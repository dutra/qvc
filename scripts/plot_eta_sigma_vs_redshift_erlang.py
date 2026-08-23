#!/usr/bin/env python3
"""Plot eta_sigma versus redshift for Erlang catalogs and Stone objects.

The defaults compare the last Erlang HDF5 entry in ``run_hubble.xonsh``
(the 2026-08-18 catalog) with the complete 2026-08-06 Erlang DHO catalog.
Use ``--stone-only`` for the dedicated 2026-08-10 Stone-object Erlang fit.
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
DEFAULT_STONE_ERLANG = Path(
    "results/data/"
    "aug10_0425am_erlang_dhodrw_svi4000w2000s500_99fb39b_stone_nolinear.h5"
)
DEFAULT_STONE_IDS = Path("data/stone_object_ids_chisq.csv")
DEFAULT_OUTPUT_BASE = Path(
    "plots/hubble/erlang_eta_sigma_comparison/"
    "eta_sigma_vs_redshift_aug18_vs_aug06_dho"
)
DEFAULT_STONE_OUTPUT_BASE = Path(
    "plots/hubble/erlang_eta_sigma_comparison/eta_sigma_vs_redshift_stone_objects"
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
    membership_ids_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read and validate the eta_sigma columns in a flat catalog HDF5."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Catalog does not exist: {path}")
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(f"Expected an HDF5 catalog, got: {path}")

    required = ("object_id", "z", "eta_sigma", "eta_sigma_err")
    optional = ("eta_sigma_kl", "num_divergences")
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

    membership_metadata: dict[str, object] = {}
    if membership_ids_path is not None:
        membership_ids_path = Path(membership_ids_path)
        if not membership_ids_path.is_file():
            raise FileNotFoundError(
                f"Membership ID catalog does not exist: {membership_ids_path}"
            )
        membership = pd.read_csv(
            membership_ids_path,
            usecols=["object_id"],
            dtype={"object_id": "string"},
        )
        membership["object_id"] = membership["object_id"].map(_normalize_object_id)
        membership_empty = membership["object_id"].isin({"", "nan", "None"})
        if np.any(membership_empty):
            raise ValueError(f"{membership_ids_path} contains empty object IDs")
        membership_duplicated = membership["object_id"].duplicated(keep=False)
        if np.any(membership_duplicated):
            examples = (
                membership.loc[membership_duplicated, "object_id"]
                .drop_duplicates()
                .head(10)
            )
            raise ValueError(
                f"{membership_ids_path} contains duplicate object IDs, "
                f"including {examples.tolist()}"
            )
        expected_ids = set(membership["object_id"])
        fitted_ids = set(frame["object_id"])
        unexpected_ids = sorted(fitted_ids - expected_ids)
        if unexpected_ids:
            raise ValueError(
                f"{path} contains {len(unexpected_ids)} object(s) outside "
                f"{membership_ids_path}, including {unexpected_ids[:10]}"
            )
        missing_ids = sorted(expected_ids - fitted_ids)
        membership_metadata = {
            "membership_ids_path": str(membership_ids_path),
            "canonical_membership_rows": int(len(expected_ids)),
            "catalog_membership_rows": int(len(fitted_ids)),
            "canonical_membership_coverage_fraction": float(
                len(fitted_ids) / len(expected_ids)
            ),
            "missing_membership_rows": int(len(missing_ids)),
            "missing_membership_object_ids": missing_ids,
            "unexpected_membership_rows": 0,
        }

    for column in (
        "z",
        "eta_sigma",
        "eta_sigma_err",
        "eta_sigma_kl",
        "num_divergences",
    ):
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
    if membership_metadata:
        membership_metadata["plotted_membership_rows"] = int(len(selected))

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
        **membership_metadata,
    }
    if "num_divergences" in selected:
        divergences = selected["num_divergences"].to_numpy(dtype=float)
        finite_divergences = divergences[np.isfinite(divergences)]
        metadata.update(
            {
                "rows_with_finite_num_divergences": int(len(finite_divergences)),
                "rows_with_divergences": int(
                    np.count_nonzero(finite_divergences > 0.0)
                ),
                "median_num_divergences": (
                    float(np.median(finite_divergences))
                    if finite_divergences.size
                    else None
                ),
                "max_num_divergences": (
                    float(np.max(finite_divergences))
                    if finite_divergences.size
                    else None
                ),
            }
        )
    return selected, metadata


def _common_bin_edges(
    frames: list[pd.DataFrame],
    n_bins: int,
    *,
    binning: str,
) -> np.ndarray:
    all_z = np.concatenate([frame["z"].to_numpy(dtype=float) for frame in frames])
    z_min = float(np.min(all_z))
    z_max = float(np.max(all_z))
    if not z_max > z_min:
        raise ValueError("The combined redshift range must have nonzero width")
    if binning == "equal_width":
        return np.linspace(z_min, z_max, n_bins + 1)
    if binning == "quantile":
        edges = np.unique(np.quantile(all_z, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 2:
            raise ValueError("Quantile binning produced fewer than two unique edges")
        return edges
    raise ValueError(f"Unknown redshift binning mode: {binning}")


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
    binning: str,
    z_min_requested: float | None,
    z_max_requested: float | None,
    figure_title: str,
    figure_subtitle: str,
    show_measurement_errors: bool = False,
) -> dict[str, object]:
    """Create an eta_sigma plot and write its companion data files."""
    if not catalogs:
        raise ValueError("At least one catalog is required")
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    edges = _common_bin_edges(
        [frame for frame, _ in catalogs],
        n_bins,
        binning=binning,
    )

    colors = ("#356CA5", "#D17625", "#2F855A")
    n_panels = len(catalogs)
    figure_height = 5.7 if n_panels == 1 else 4.35 * n_panels
    fig, axes_grid = plt.subplots(
        n_panels,
        1,
        figsize=(9.2, figure_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes_grid[:, 0]
    binned_metadata: dict[str, list[dict[str, object]]] = {}

    for index, (frame, metadata) in enumerate(catalogs):
        ax = axes[index]
        color = colors[index % len(colors)]
        if show_measurement_errors:
            ax.errorbar(
                frame["z"],
                frame["eta_sigma"],
                yerr=frame["eta_sigma_err"],
                fmt="none",
                ecolor=color,
                elinewidth=0.65,
                alpha=0.18,
                capsize=0,
                rasterized=True,
                zorder=0,
            )
        small_catalog = len(frame) <= 500
        ax.scatter(
            frame["z"],
            frame["eta_sigma"],
            s=18 if small_catalog else 9,
            alpha=0.65 if small_catalog else 0.20,
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
        annotation_lines = [rf"$N={metadata['plotted_rows']:,}$"]
        if "canonical_membership_rows" in metadata:
            annotation_lines.append(
                "Stone plotted "
                f"{metadata['plotted_membership_rows']:,}/"
                f"{metadata['canonical_membership_rows']:,}"
            )
        annotation_lines.extend(
            (
                rf"$z={metadata['z_min']:.3f}$--${metadata['z_max']:.3f}$",
                rf"median $\eta_\sigma={metadata['eta_sigma_median']:.3f}$",
                rf"Spearman $\rho={metadata['spearman_rho_eta_sigma_vs_z']:.3f}$",
            )
        )
        if "rows_with_divergences" in metadata:
            annotation_lines.append(
                "divergences $>0$: "
                f"{metadata['rows_with_divergences']:,}/"
                f"{metadata['rows_with_finite_num_divergences']:,}"
            )
        ax.text(
            0.985,
            0.965,
            "\n".join(annotation_lines),
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
    individual_label = (
        r"Posterior median $\pm$ 68% half-width"
        if show_measurement_errors
        else "Individual fits"
    )
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
                label=individual_label,
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

    fig.suptitle(figure_title, y=0.985)
    fig.text(
        0.5,
        0.952,
        figure_subtitle,
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
        "redshift_binning": binning,
        "minimum_rows_per_plotted_bin": min_bin_count,
        "measurement_error_bars_plotted": bool(show_measurement_errors),
        "figure_title": figure_title,
        "figure_subtitle": figure_subtitle,
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
        description=(
            "Plot eta_sigma versus redshift for Erlang catalogs or the dedicated "
            "Stone-object run."
        )
    )
    parser.add_argument(
        "--stone-only",
        action="store_true",
        help="Plot the dedicated Stone-object Erlang catalog instead of the comparison.",
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
        "--stone-erlang",
        type=Path,
        default=DEFAULT_STONE_ERLANG,
        help=f"Dedicated Stone-object Erlang HDF5 (default: {DEFAULT_STONE_ERLANG})",
    )
    parser.add_argument(
        "--stone-ids",
        type=Path,
        default=DEFAULT_STONE_IDS,
        help=f"Canonical Stone object-ID CSV (default: {DEFAULT_STONE_IDS})",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=None,
        help=(
            "Output path without an extension; mode-specific defaults are used when "
            "omitted. PDF, PNG, CSV, and JSON are written."
        ),
    )
    parser.add_argument("--n-bins", type=int, default=None)
    parser.add_argument("--min-bin-count", type=int, default=None)
    parser.add_argument(
        "--binning",
        choices=("equal_width", "quantile"),
        default=None,
        help="Redshift-bin construction; defaults to quantile for Stone-only mode.",
    )
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    z_min = _finite_optional_bound(args.z_min, "z_min")
    z_max = _finite_optional_bound(args.z_max, "z_max")
    if z_min is not None and z_max is not None and z_min >= z_max:
        raise ValueError(f"Require z_min < z_max; got {z_min}, {z_max}")
    n_bins = args.n_bins if args.n_bins is not None else (8 if args.stone_only else 12)
    min_bin_count = (
        args.min_bin_count if args.min_bin_count is not None else 10
    )
    binning = args.binning or ("quantile" if args.stone_only else "equal_width")
    if n_bins < 1:
        raise ValueError(f"n_bins must be positive; got {n_bins}")
    if min_bin_count < 1:
        raise ValueError(
            f"min_bin_count must be positive; got {min_bin_count}"
        )

    if args.stone_only:
        catalogs = [
            read_eta_sigma_catalog(
                args.stone_erlang,
                catalog_key="stone_erlang_dho_aug10_nolinear",
                catalog_label="Stone objects — Erlang DHO, no linear trend (Aug 10)",
                z_min=z_min,
                z_max=z_max,
                membership_ids_path=args.stone_ids,
            )
        ]
        output_base = args.output_base or DEFAULT_STONE_OUTPUT_BASE
        figure_title = r"$\eta_{\sigma}$ versus redshift for Stone quasars"
        figure_subtitle = (
            "All finite saved fits; error bars show posterior 68% half-widths; "
            "band shows binwise 16th--84th percentiles"
        )
        show_measurement_errors = True
        output_description = "Stone-object eta_sigma versus redshift plot"
    else:
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
        catalogs = [latest, dho]
        output_base = args.output_base or DEFAULT_OUTPUT_BASE
        figure_title = r"$\eta_{\sigma}$ versus redshift: Erlang catalog comparison"
        figure_subtitle = (
            "All finite saved fits; bands show the 16th--84th percentiles "
            "in common redshift bins"
        )
        show_measurement_errors = False
        output_description = "eta_sigma versus redshift comparison"

    metadata = make_plot(
        catalogs,
        output_base=output_base,
        n_bins=n_bins,
        min_bin_count=min_bin_count,
        binning=binning,
        z_min_requested=z_min,
        z_max_requested=z_max,
        figure_title=figure_title,
        figure_subtitle=figure_subtitle,
        show_measurement_errors=show_measurement_errors,
    )

    print(f"Wrote {output_description}:")
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
