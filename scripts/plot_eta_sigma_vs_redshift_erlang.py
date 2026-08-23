#!/usr/bin/env python3
"""Compare eta_sigma versus redshift in the latest and DHO Erlang catalogs.

The defaults compare the last Erlang HDF5 entry in ``run_hubble.xonsh``
(the 2026-08-18 catalog) with the complete 2026-08-06 Erlang DHO catalog.
Every successfully saved row with finite ``z``, ``eta_sigma``, and a positive
``eta_sigma_err`` is retained by default; no Hubble-fit redshift cut or
sampler-diagnostic cut is applied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


DEFAULT_LATEST_ERLANG = Path(
    "results/data/"
    "aug18_0955pm_erlang_preview_chisq_nofluxguard_n8000_ebdef2a_chisq.h5"
)
DEFAULT_ERLANG_DHO = Path(
    "results/data/aug06_0347pm_erlang_dhodrw_99fb39b_chisq.h5"
)
DEFAULT_OUTPUT_BASE = Path(
    "plots/hubble/erlang_eta_sigma_comparison/"
    "eta_sigma_vs_redshift_aug18_vs_aug06_dho"
)

ETA_SIGMA_PRIOR_LOCATION = -0.5
ETA_SIGMA_PRIOR_LOW = -1.5
ETA_SIGMA_PRIOR_HIGH = 0.25


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


def _normalize_object_id(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        if float(value).is_integer():
            return str(int(value))
    return str(value).strip()


def read_eta_sigma_catalog(
    path: Path,
    *,
    catalog_key: str,
    catalog_label: str,
    z_min: float | None = None,
    z_max: float | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read and validate the eta_sigma columns in a flat catalog HDF5."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Catalog does not exist: {path}")
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(f"Expected an HDF5 catalog, got: {path}")

    required = ("object_id", "z", "eta_sigma", "eta_sigma_err")
    optional = ("eta_sigma_kl",)
    with h5py.File(path, "r") as handle:
        group = handle["catalog"] if "catalog" in handle else handle
        missing = sorted(column for column in required if column not in group)
        if missing:
            raise KeyError(f"{path} is missing required dataset(s): {missing}")
        columns = {
            column: _decode_hdf_values(group[column][...])
            for column in (*required, *optional)
            if column in group
        }

    lengths = {column: len(values) for column, values in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Catalog datasets have inconsistent lengths: {lengths}")

    frame = pd.DataFrame(columns)
    frame["object_id"] = frame["object_id"].map(_normalize_object_id)
    empty_id = frame["object_id"].isin({"", "nan", "None"})
    if np.any(empty_id):
        raise ValueError(
            f"{path} contains {int(np.count_nonzero(empty_id))} empty object IDs"
        )
    duplicated = frame["object_id"].duplicated(keep=False)
    if np.any(duplicated):
        examples = frame.loc[duplicated, "object_id"].drop_duplicates().head(10)
        raise ValueError(
            f"{path} contains duplicate object IDs, including {examples.tolist()}"
        )

    for column in ("z", "eta_sigma", "eta_sigma_err", "eta_sigma_kl"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    valid = (
        np.isfinite(frame["z"])
        & np.isfinite(frame["eta_sigma"])
        & np.isfinite(frame["eta_sigma_err"])
        & (frame["eta_sigma_err"] > 0.0)
    )
    if z_min is not None:
        valid &= frame["z"] >= z_min
    if z_max is not None:
        valid &= frame["z"] <= z_max

    selected = frame.loc[valid].copy()
    if selected.empty:
        raise ValueError(f"No valid rows remain in {path}")
    selected = selected.sort_values(["z", "object_id"]).reset_index(drop=True)
    selected.insert(0, "catalog_key", catalog_key)
    selected.insert(1, "catalog_label", catalog_label)

    rho_result = spearmanr(selected["z"], selected["eta_sigma"])
    metadata: dict[str, object] = {
        "catalog_key": catalog_key,
        "catalog_label": catalog_label,
        "path": str(path),
        "total_rows": int(len(frame)),
        "plotted_rows": int(len(selected)),
        "dropped_rows": int(len(frame) - len(selected)),
        "z_min": float(selected["z"].min()),
        "z_max": float(selected["z"].max()),
        "eta_sigma_min": float(selected["eta_sigma"].min()),
        "eta_sigma_median": float(selected["eta_sigma"].median()),
        "eta_sigma_max": float(selected["eta_sigma"].max()),
        "eta_sigma_percentiles_16_50_84": [
            float(value)
            for value in np.percentile(selected["eta_sigma"], [16.0, 50.0, 84.0])
        ],
        "median_eta_sigma_err": float(selected["eta_sigma_err"].median()),
        "spearman_rho_eta_sigma_vs_z": float(rho_result.statistic),
        "spearman_pvalue_eta_sigma_vs_z": float(rho_result.pvalue),
    }
    return selected, metadata


def _common_bin_edges(frames: list[pd.DataFrame], n_bins: int) -> np.ndarray:
    all_z = np.concatenate([frame["z"].to_numpy(dtype=float) for frame in frames])
    z_min = float(np.min(all_z))
    z_max = float(np.max(all_z))
    if not z_max > z_min:
        raise ValueError("The combined redshift range must have nonzero width")
    return np.linspace(z_min, z_max, n_bins + 1)


def binned_eta_sigma(
    frame: pd.DataFrame,
    edges: np.ndarray,
    *,
    min_bin_count: int,
) -> pd.DataFrame:
    """Return median and central 68-percent summaries in common z bins."""
    z = frame["z"].to_numpy(dtype=float)
    eta_sigma = frame["eta_sigma"].to_numpy(dtype=float)
    bin_index = np.digitize(z, edges[1:-1], right=False)
    rows: list[dict[str, float | int]] = []
    for index in range(len(edges) - 1):
        mask = bin_index == index
        count = int(np.count_nonzero(mask))
        if count < min_bin_count:
            continue
        eta_values = eta_sigma[mask]
        rows.append(
            {
                "z_low": float(edges[index]),
                "z_high": float(edges[index + 1]),
                "z_median": float(np.median(z[mask])),
                "eta_sigma_median": float(np.median(eta_values)),
                "eta_sigma_p16": float(np.percentile(eta_values, 16.0)),
                "eta_sigma_p84": float(np.percentile(eta_values, 84.0)),
                "count": count,
            }
        )
    return pd.DataFrame(rows)


def make_plot(
    catalogs: list[tuple[pd.DataFrame, dict[str, object]]],
    *,
    output_base: Path,
    n_bins: int,
    min_bin_count: int,
    z_min_requested: float | None,
    z_max_requested: float | None,
) -> dict[str, object]:
    """Create the two-panel comparison and write its companion data files."""
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    edges = _common_bin_edges([frame for frame, _ in catalogs], n_bins)

    colors = ("#356CA5", "#D17625")
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.7), sharex=True, sharey=True)
    binned_metadata: dict[str, list[dict[str, object]]] = {}

    for ax, (frame, metadata), color in zip(axes, catalogs, colors, strict=True):
        ax.scatter(
            frame["z"],
            frame["eta_sigma"],
            s=9,
            alpha=0.20,
            linewidths=0,
            color=color,
            rasterized=True,
            zorder=1,
        )
        summary = binned_eta_sigma(
            frame,
            edges,
            min_bin_count=min_bin_count,
        )
        if not summary.empty:
            ax.fill_between(
                summary["z_median"],
                summary["eta_sigma_p16"],
                summary["eta_sigma_p84"],
                color=color,
                alpha=0.18,
                linewidth=0,
                zorder=2,
            )
            ax.plot(
                summary["z_median"],
                summary["eta_sigma_median"],
                color=color,
                marker="o",
                markersize=4.5,
                linewidth=2.0,
                zorder=3,
            )
        ax.axhline(
            ETA_SIGMA_PRIOR_LOCATION,
            color="0.25",
            linestyle="--",
            linewidth=1.15,
            alpha=0.85,
            zorder=0,
        )
        ax.set_title(str(metadata["catalog_label"]), loc="left", fontsize=12.5)
        ax.set_ylabel(r"Wavelength exponent $\eta_{\sigma}$")
        ax.grid(True, alpha=0.22, linewidth=0.7)
        ax.text(
            0.985,
            0.965,
            "\n".join(
                (
                    rf"$N={metadata['plotted_rows']:,}$",
                    rf"$z={metadata['z_min']:.3f}$--${metadata['z_max']:.3f}$",
                    rf"median $\eta_\sigma={metadata['eta_sigma_median']:.3f}$",
                    rf"Spearman $\rho={metadata['spearman_rho_eta_sigma_vs_z']:.3f}$",
                )
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.7,
            bbox={"facecolor": "white", "edgecolor": "0.82", "alpha": 0.88},
        )
        binned_metadata[str(metadata["catalog_key"])] = summary.to_dict(orient="records")

    combined_z = np.concatenate(
        [frame["z"].to_numpy(dtype=float) for frame, _ in catalogs]
    )
    z_span = float(np.ptp(combined_z))
    x_padding = max(0.03 * z_span, 0.03)
    axes[-1].set_xlim(float(np.min(combined_z)) - x_padding, float(np.max(combined_z)) + x_padding)
    axes[-1].set_ylim(ETA_SIGMA_PRIOR_LOW - 0.04, ETA_SIGMA_PRIOR_HIGH + 0.04)
    axes[-1].set_xlabel("Redshift $z$")
    axes[0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="0.35",
                markerfacecolor="0.35",
                linestyle="none",
                markersize=4,
                alpha=0.5,
                label="Individual fits",
            ),
            Line2D([0], [0], color="0.25", linewidth=2.0, label="Binned median"),
            Line2D(
                [0],
                [0],
                color="0.25",
                linestyle="--",
                linewidth=1.15,
                label=r"Prior location $\mu=-0.5$",
            ),
        ],
        loc="lower right",
        frameon=False,
        fontsize=9.5,
    )

    fig.suptitle(r"$\eta_{\sigma}$ versus redshift: Erlang catalog comparison", y=0.985)
    fig.text(
        0.5,
        0.952,
        "All finite saved fits; bands show the 16th--84th percentiles in common redshift bins",
        ha="center",
        va="top",
        fontsize=10,
        color="0.3",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    csv_path = output_base.with_suffix(".csv")
    json_path = output_base.with_suffix(".json")
    fig.savefig(pdf_path, dpi=220, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    combined = pd.concat([frame for frame, _ in catalogs], ignore_index=True)
    combined.to_csv(csv_path, index=False)

    metadata_out: dict[str, object] = {
        "selection": {
            "definition": (
                "successfully saved catalog rows with finite z and eta_sigma, "
                "finite positive eta_sigma_err, and optional requested z bounds"
            ),
            "sampler_diagnostic_cuts_applied": False,
            "z_min_requested": z_min_requested,
            "z_max_requested": z_max_requested,
        },
        "eta_sigma_definition": (
            "dimensionless wavelength exponent in sigma(lambda) proportional "
            "to (lambda/lambda_reference)^eta_sigma"
        ),
        "eta_sigma_prior": {
            "distribution": "TruncatedNormal",
            "location": ETA_SIGMA_PRIOR_LOCATION,
            "scale": 0.3,
            "low": ETA_SIGMA_PRIOR_LOW,
            "high": ETA_SIGMA_PRIOR_HIGH,
        },
        "common_redshift_bin_edges": [float(value) for value in edges],
        "minimum_rows_per_plotted_bin": min_bin_count,
        "catalogs": [metadata for _, metadata in catalogs],
        "binned_summaries": binned_metadata,
        "outputs": {
            "pdf": str(pdf_path),
            "png": str(png_path),
            "csv": str(csv_path),
            "json": str(json_path),
        },
    }
    json_path.write_text(json.dumps(metadata_out, indent=2) + "\n", encoding="utf-8")
    return metadata_out


def _finite_optional_bound(value: float | None, name: str) -> float | None:
    if value is not None and not np.isfinite(value):
        raise ValueError(f"{name} must be finite when provided; got {value}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot eta_sigma versus redshift for the latest and DHO Erlang catalogs."
    )
    parser.add_argument(
        "--latest-erlang",
        type=Path,
        default=DEFAULT_LATEST_ERLANG,
        help=f"Latest Erlang HDF5 catalog (default: {DEFAULT_LATEST_ERLANG})",
    )
    parser.add_argument(
        "--erlang-dho",
        type=Path,
        default=DEFAULT_ERLANG_DHO,
        help=f"Complete Erlang DHO HDF5 catalog (default: {DEFAULT_ERLANG_DHO})",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help="Output path without an extension; PDF, PNG, CSV, and JSON are written.",
    )
    parser.add_argument("--n-bins", type=int, default=12)
    parser.add_argument("--min-bin-count", type=int, default=10)
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    z_min = _finite_optional_bound(args.z_min, "z_min")
    z_max = _finite_optional_bound(args.z_max, "z_max")
    if z_min is not None and z_max is not None and z_min >= z_max:
        raise ValueError(f"Require z_min < z_max; got {z_min}, {z_max}")
    if args.n_bins < 1:
        raise ValueError(f"n_bins must be positive; got {args.n_bins}")
    if args.min_bin_count < 1:
        raise ValueError(
            f"min_bin_count must be positive; got {args.min_bin_count}"
        )

    latest = read_eta_sigma_catalog(
        args.latest_erlang,
        catalog_key="latest_erlang_aug18",
        catalog_label="Latest Erlang catalog (Aug 18)",
        z_min=z_min,
        z_max=z_max,
    )
    dho = read_eta_sigma_catalog(
        args.erlang_dho,
        catalog_key="erlang_dho_aug06",
        catalog_label="Complete Erlang DHO catalog (Aug 06)",
        z_min=z_min,
        z_max=z_max,
    )
    metadata = make_plot(
        [latest, dho],
        output_base=args.output_base,
        n_bins=args.n_bins,
        min_bin_count=args.min_bin_count,
        z_min_requested=z_min,
        z_max_requested=z_max,
    )

    print("Wrote eta_sigma versus redshift comparison:")
    for output in metadata["outputs"].values():
        print(f"  {output}")
    for catalog in metadata["catalogs"]:
        print(
            f"  {catalog['catalog_label']}: N={catalog['plotted_rows']}, "
            f"z={catalog['z_min']:.6g}--{catalog['z_max']:.6g}, "
            f"rho={catalog['spearman_rho_eta_sigma_vs_z']:.4f}"
        )


if __name__ == "__main__":
    main()
