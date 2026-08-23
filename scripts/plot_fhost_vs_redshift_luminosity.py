#!/usr/bin/env python3
"""Plot the catalog-native host fraction against redshift and luminosity.

The active ``run_hubble.xonsh`` inputs do not contain a direct
``f_host_2500`` measurement.  They do contain the fitted AGN fraction at
rest-frame 5100 Angstrom, so this diagnostic uses the directly supported

    f_host,5100 = 1 - fracAGN_5100_fit

definition.  The light-curve and spectral catalogs are joined one-to-one on
``object_id``.  By default, every finite matched redshift is retained.
Luminosity is the matching fitted AGN ``lambda L_lambda`` at 5100 Angstrom.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_LC_CATALOG = Path(
    "results/data/"
    "aug22_0346pm_mag_linear_svi4000w2000s500_stonechisq_"
    "specaug210827am_n8000_af3ff4c_chisq.h5"
)
DEFAULT_SPECTRA_CATALOG = Path(
    "results/data/aug21_0827am_spectrafit_9b14caf_nested_N8000_rhat.h5"
)
DEFAULT_OUTPUT_BASE = Path(
    "plots/hubble/"
    "aug22b_dereddened_nocutsrelaxed_fixedzcompletenessrange_"
    "h5aug0346pmmaglinear_specaug210827am_newcuts_newbins_n8000_quick/"
    "diagnostics/fhost_5100_vs_redshift_l5100"
)


def _decode_hdf_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind == "S":
        return values.astype(str)
    if values.dtype.kind == "O":
        return np.asarray(
            [
                value.decode("utf-8", errors="replace")
                if isinstance(value, bytes)
                else value
                for value in values
            ],
            dtype=object,
        )
    return values


def _read_hdf_columns(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    """Read selected columns from a flat LC HDF5 or a spectra-catalog HDF5."""
    with h5py.File(path, "r") as handle:
        group = handle["catalog"] if "catalog" in handle else handle
        missing = sorted(column for column in columns if column not in group)
        if missing:
            raise KeyError(f"{path} is missing required dataset(s): {missing}")
        return pd.DataFrame(
            {
                column: _decode_hdf_values(group[column][...])
                for column in columns
            }
        )


def _normalize_object_id(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        if float(value).is_integer():
            return str(int(value))
    return str(value).strip()


def _normalize_ids(frame: pd.DataFrame, *, source: Path) -> pd.DataFrame:
    frame = frame.copy()
    frame["object_id"] = frame["object_id"].map(_normalize_object_id)
    empty = frame["object_id"].eq("") | frame["object_id"].eq("nan")
    if np.any(empty):
        raise ValueError(f"{source} contains {int(np.count_nonzero(empty))} empty object_id values")
    duplicated = frame["object_id"].duplicated(keep=False)
    if np.any(duplicated):
        examples = frame.loc[duplicated, "object_id"].drop_duplicates().head(10).tolist()
        raise ValueError(f"{source} contains duplicate object_id values, including {examples}")
    return frame


def _read_spectra_catalog(requested_path: Path) -> tuple[pd.DataFrame, Path, bool]:
    """Read the configured spectra HDF5, with an explicit same-stem CSV fallback."""
    required = (
        "object_id",
        "z",
        "fit_ok",
        "fracAGN_5100_fit",
        "fracAGN_5100_fit_err",
        "log_disk_luminosity_fit",
        "log_disk_luminosity_fit_err",
    )
    if requested_path.exists():
        if requested_path.suffix.lower() in {".h5", ".hdf5"}:
            return _read_hdf_columns(requested_path, required), requested_path, False
        if requested_path.suffix.lower() == ".csv":
            return pd.read_csv(requested_path, usecols=list(required)), requested_path, False
        raise ValueError(f"Unsupported spectral catalog format: {requested_path}")

    csv_fallback = requested_path.with_suffix(".csv")
    if requested_path.suffix.lower() in {".h5", ".hdf5"} and csv_fallback.exists():
        warnings.warn(
            f"Configured spectra catalog {requested_path} is missing; "
            f"using same-stem CSV {csv_fallback}.",
            stacklevel=2,
        )
        return pd.read_csv(csv_fallback, usecols=list(required)), csv_fallback, True

    raise FileNotFoundError(f"Spectral catalog does not exist: {requested_path}")


def build_fhost_catalog(
    lc_catalog: Path,
    spectra_catalog: Path,
    *,
    z_min: float | None,
    z_max: float | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Join the two runner catalogs and derive the direct 5100-A host fraction."""
    lc_catalog = Path(lc_catalog)
    spectra_catalog = Path(spectra_catalog)
    if not lc_catalog.exists():
        raise FileNotFoundError(f"Light-curve catalog does not exist: {lc_catalog}")
    if z_min is not None and not np.isfinite(z_min):
        raise ValueError(f"z_min must be finite or None; got {z_min}")
    if z_max is not None and not np.isfinite(z_max):
        raise ValueError(f"z_max must be finite or None; got {z_max}")
    if z_min is not None and z_max is not None and z_min >= z_max:
        raise ValueError(f"Require z_min < z_max; got {z_min}, {z_max}")

    lc = _read_hdf_columns(lc_catalog, ("object_id", "z"))
    spectra, spectra_used, used_csv_fallback = _read_spectra_catalog(spectra_catalog)
    lc = _normalize_ids(lc, source=lc_catalog)
    spectra = _normalize_ids(spectra, source=spectra_used)

    fit_ok = spectra["fit_ok"]
    if not pd.api.types.is_bool_dtype(fit_ok.dtype):
        fit_ok = fit_ok.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    spectra_fit = spectra.loc[fit_ok].copy()

    merged = lc.merge(
        spectra_fit,
        on="object_id",
        how="inner",
        suffixes=("_lc", "_spectra"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("The light-curve and successful spectral-fit catalogs have no matching IDs")

    z_lc = pd.to_numeric(merged["z_lc"], errors="coerce").to_numpy(dtype=float)
    z_spectra = pd.to_numeric(merged["z_spectra"], errors="coerce").to_numpy(dtype=float)
    finite_z_pair = np.isfinite(z_lc) & np.isfinite(z_spectra)
    max_z_difference = (
        float(np.max(np.abs(z_lc[finite_z_pair] - z_spectra[finite_z_pair])))
        if np.any(finite_z_pair)
        else np.nan
    )
    if not np.isfinite(max_z_difference) or max_z_difference > 1e-8:
        raise ValueError(
            "Redshifts disagree between matched catalogs: "
            f"maximum absolute difference is {max_z_difference}"
        )

    f_agn_5100 = pd.to_numeric(
        merged["fracAGN_5100_fit"], errors="coerce"
    ).to_numpy(dtype=float)
    finite_fraction = np.isfinite(f_agn_5100)
    outside_fraction = finite_fraction & ((f_agn_5100 < 0.0) | (f_agn_5100 > 1.0))
    if np.any(outside_fraction):
        raise ValueError(
            "fracAGN_5100_fit must lie in [0, 1]; found "
            f"{int(np.count_nonzero(outside_fraction))} out-of-range value(s)"
        )

    log_l5100 = (
        pd.to_numeric(merged["log_disk_luminosity_fit"], errors="coerce")
        + np.log10(5100.0)
        + 7.0
    )
    output = pd.DataFrame(
        {
            "object_id": merged["object_id"],
            "z": z_lc,
            "log_l5100": log_l5100,
            "log_l5100_err": pd.to_numeric(
                merged["log_disk_luminosity_fit_err"], errors="coerce"
            ),
            "f_agn_5100": f_agn_5100,
            "f_host_5100": 1.0 - f_agn_5100,
            "f_host_5100_err": pd.to_numeric(
                merged["fracAGN_5100_fit_err"], errors="coerce"
            ),
        }
    )
    finite = (
        np.isfinite(output["z"])
        & np.isfinite(output["log_l5100"])
        & np.isfinite(output["f_host_5100"])
        & output["f_host_5100"].between(0.0, 1.0, inclusive="both")
    )
    if z_min is not None:
        finite &= output["z"] >= z_min
    if z_max is not None:
        finite &= output["z"] <= z_max
    output = output.loc[finite].sort_values(["z", "object_id"]).reset_index(drop=True)
    if output.empty:
        raise ValueError("No finite matched rows remain inside the requested redshift interval")

    metadata: dict[str, object] = {
        "lc_catalog": str(lc_catalog),
        "spectra_catalog_requested": str(spectra_catalog),
        "spectra_catalog_used": str(spectra_used),
        "used_same_stem_csv_fallback": bool(used_csv_fallback),
        "lc_rows": int(len(lc)),
        "spectra_rows": int(len(spectra)),
        "spectra_fit_ok_rows": int(len(spectra_fit)),
        "matched_fit_ok_rows": int(len(merged)),
        "plotted_rows": int(len(output)),
        "z_min_requested": None if z_min is None else float(z_min),
        "z_max_requested": None if z_max is None else float(z_max),
        "z_min_plotted": float(output["z"].min()),
        "z_max_plotted": float(output["z"].max()),
        "max_abs_redshift_difference": max_z_difference,
        "f_host_definition": "1 - fracAGN_5100_fit",
        "luminosity_definition": (
            "log10[lambda L_lambda(5100 A) / (erg s^-1)] = "
            "log_disk_luminosity_fit + log10(5100) + 7"
        ),
        "luminosity_source_column": "log_disk_luminosity_fit",
    }
    return output, metadata


def _binned_quantiles(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int,
    min_bin_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)
    if x.size == 0 or np.max(x) <= np.min(x):
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty

    edges = np.linspace(float(np.min(x)), float(np.max(x)), n_bins + 1)
    x_mid: list[float] = []
    y_median: list[float] = []
    y_low: list[float] = []
    y_high: list[float] = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (x >= left) & (x < right)
        if index == n_bins - 1:
            in_bin = (x >= left) & (x <= right)
        if np.count_nonzero(in_bin) < min_bin_count:
            continue
        x_mid.append(float(np.median(x[in_bin])))
        y_median.append(float(np.median(y[in_bin])))
        y_low.append(float(np.percentile(y[in_bin], 16.0)))
        y_high.append(float(np.percentile(y[in_bin], 84.0)))
    return tuple(np.asarray(values, dtype=float) for values in (x_mid, y_median, y_low, y_high))


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return np.nan
    x_rank = pd.Series(x[finite]).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y[finite]).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def plot_fhost_catalog(
    frame: pd.DataFrame,
    output_base: Path,
    *,
    n_bins: int = 12,
    min_bin_count: int = 20,
) -> tuple[Path, Path]:
    """Create PDF and PNG versions of the two-panel diagnostic."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    if min_bin_count < 1:
        raise ValueError("min_bin_count must be at least 1")

    output_base = Path(output_base)
    if output_base.suffix.lower() in {".pdf", ".png", ".csv", ".json"}:
        output_base = output_base.with_suffix("")
    output_base.parent.mkdir(parents=True, exist_ok=True)

    y = frame["f_host_5100"].to_numpy(dtype=float)
    panels = (
        (frame["z"].to_numpy(dtype=float), "Redshift $z$"),
        (
            frame["log_l5100"].to_numpy(dtype=float),
            r"$\log_{10}[\lambda L_{\lambda,\rm AGN}(5100\,\AA)/{\rm erg\,s^{-1}}]$",
        ),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), sharey=True)
    for panel_index, (ax, (x, x_label)) in enumerate(zip(axes, panels)):
        finite = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
        at_floor = finite & np.isclose(y, 1.0e-3, rtol=0.0, atol=1.0e-10)
        above_floor = finite & ~at_floor
        ax.scatter(
            x[above_floor],
            y[above_floor],
            s=8,
            alpha=0.12,
            color="0.2",
            linewidths=0,
            rasterized=True,
            label="Objects" if panel_index == 0 else None,
        )
        ax.scatter(
            x[at_floor],
            y[at_floor],
            s=12,
            alpha=0.28,
            color="#D55E00",
            marker="v",
            linewidths=0,
            rasterized=True,
            label="At fit floor" if panel_index == 0 else None,
        )
        x_mid, y_median, y_low, y_high = _binned_quantiles(
            x,
            y,
            n_bins=n_bins,
            min_bin_count=min_bin_count,
        )
        if x_mid.size:
            ax.fill_between(
                x_mid,
                y_low,
                y_high,
                color="#56B4E9",
                alpha=0.24,
                linewidth=0,
                label="16th--84th percentile" if panel_index == 0 else None,
            )
            ax.plot(
                x_mid,
                y_median,
                color="#0072B2",
                linewidth=2.2,
                marker="o",
                markersize=3.5,
                label="Binned median" if panel_index == 0 else None,
            )
        rho = _spearman_rho(x, y)
        ax.text(
            0.04,
            0.95,
            rf"$N={np.count_nonzero(finite):,}$" + "\n"
            + rf"$N_{{\rm floor}}={np.count_nonzero(at_floor):,}$" + "\n"
            + rf"$\rho_{{\rm S}}={rho:.3f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
        ax.set_xlabel(x_label)
        ax.set_yscale("log")
        ax.set_ylim(8.0e-4, 1.05)
        ax.grid(True, which="major", alpha=0.22)
        ax.grid(True, which="minor", alpha=0.08)

    axes[0].set_ylabel(r"$f_{\rm host,5100}=1-f_{\rm AGN,5100}$")
    axes[0].axhline(1.0e-3, color="#D55E00", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[1].axhline(1.0e-3, color="#D55E00", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[1].text(
        0.97,
        1.15e-3,
        r"fit floor ($10^{-3}$)",
        ha="right",
        va="bottom",
        color="#A54500",
        fontsize=9,
        transform=axes[1].get_yaxis_transform(),
    )
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    fig.suptitle("Host-galaxy fraction in the run_hubble catalogs", fontsize=14, y=0.99)
    z_min = float(frame["z"].min())
    z_max = float(frame["z"].max())
    fig.text(
        0.5,
        0.93,
        rf"Successful matched SED fits; plotted range ${z_min:.2f}\leq z\leq{z_max:.2f}$; "
        rf"$f_{{\rm host}}\equiv1-\mathrm{{fracAGN\_5100\_fit}}$",
        ha="center",
        va="top",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.13, top=0.85, wspace=0.08)

    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lc-catalog", type=Path, default=DEFAULT_LC_CATALOG)
    parser.add_argument("--spectra-catalog", type=Path, default=DEFAULT_SPECTRA_CATALOG)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument(
        "--z-min",
        type=float,
        default=None,
        help="Optional inclusive lower redshift bound (default: no bound).",
    )
    parser.add_argument(
        "--z-max",
        type=float,
        default=None,
        help="Optional inclusive upper redshift bound (default: no bound).",
    )
    parser.add_argument("--n-bins", type=int, default=12)
    parser.add_argument("--min-bin-count", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    frame, metadata = build_fhost_catalog(
        args.lc_catalog,
        args.spectra_catalog,
        z_min=args.z_min,
        z_max=args.z_max,
    )
    pdf_path, png_path = plot_fhost_catalog(
        frame,
        args.output_base,
        n_bins=args.n_bins,
        min_bin_count=args.min_bin_count,
    )

    output_base = args.output_base
    if output_base.suffix.lower() in {".pdf", ".png", ".csv", ".json"}:
        output_base = output_base.with_suffix("")
    csv_path = output_base.with_suffix(".csv")
    json_path = output_base.with_suffix(".json")
    frame.to_csv(csv_path, index=False)

    fhost = frame["f_host_5100"].to_numpy(dtype=float)
    metadata.update(
        {
            "spearman_fhost_vs_redshift": _spearman_rho(
                frame["z"].to_numpy(dtype=float), fhost
            ),
            "spearman_fhost_vs_log_l5100": _spearman_rho(
                frame["log_l5100"].to_numpy(dtype=float), fhost
            ),
            "f_host_5100_percentiles_16_50_84": [
                float(value) for value in np.percentile(fhost, (16.0, 50.0, 84.0))
            ],
            "f_host_5100_fit_floor": 1.0e-3,
            "rows_at_f_host_fit_floor": int(
                np.count_nonzero(np.isclose(fhost, 1.0e-3, rtol=0.0, atol=1.0e-10))
            ),
            "outputs": {
                "pdf": str(pdf_path),
                "png": str(png_path),
                "csv": str(csv_path),
                "json": str(json_path),
            },
        }
    )
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Matched successful spectra rows: {metadata['matched_fit_ok_rows']:,}")
    print(
        "Rows plotted across z range "
        f"[{metadata['z_min_plotted']:.3f}, {metadata['z_max_plotted']:.3f}]: "
        f"{len(frame):,}"
    )
    if metadata["used_same_stem_csv_fallback"]:
        print(f"Spectra fallback used: {metadata['spectra_catalog_used']}")
    print(f"PDF:  {pdf_path}")
    print(f"PNG:  {png_path}")
    print(f"Data: {csv_path}")
    print(f"Metadata: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
