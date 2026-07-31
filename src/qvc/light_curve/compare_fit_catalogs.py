"""Compare sampler diagnostics and parameter stability between LC fit catalogs."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr


DEFAULT_ATLAS_PARAMETERS = (
    "log_sigma_uv",
    "log_tau_uv_rf",
    "eta_sigma",
    "eta_tau",
    "log_sigma_band_g",
    "log_tau_band_g_RF",
    "log_sigma_band_r",
    "log_tau_band_r_RF",
    "log_sigma_band_i",
    "log_tau_band_i_RF",
    "linear_trend",
    "lag0",
    "lag_beta",
    "log_lag_blr_g_RF",
    "log_lag_blr_r_RF",
    "log_lag_blr_i_RF",
)

CONVERGENCE_FAMILIES = (
    ("Acceptance", "_accept_prob"),
    ("Divergences", "_num_divergences"),
    ("R-hat", "_rhat"),
    ("ESS", "_ess"),
)


def _decode_strings(values):
    values = np.asarray(values)
    if values.dtype.kind == "S":
        return np.char.decode(values, "utf-8")
    return values.astype(str)


def _load_object_ids(handle, path):
    if "object_id" not in handle:
        raise KeyError(f"{path} does not contain an object_id dataset.")
    object_ids = _decode_strings(handle["object_id"][:])
    unique, counts = np.unique(object_ids, return_counts=True)
    duplicate_ids = unique[counts > 1]
    if duplicate_ids.size:
        preview = ", ".join(duplicate_ids[:5])
        raise ValueError(f"{path} contains duplicate object_id values: {preview}")
    return object_ids


def _matched_indices(new_ids, old_ids):
    new_lookup = {object_id: index for index, object_id in enumerate(new_ids)}
    old_indices = np.asarray(
        [index for index, object_id in enumerate(old_ids) if object_id in new_lookup],
        dtype=int,
    )
    new_indices = np.asarray(
        [new_lookup[old_ids[index]] for index in old_indices],
        dtype=int,
    )
    return new_indices, old_indices


def _numeric_vector_dataset(handle, name, expected_rows):
    if name not in handle:
        return False
    dataset = handle[name]
    return (
        isinstance(dataset, h5py.Dataset)
        and dataset.shape == (expected_rows,)
        and dataset.dtype.kind in "biuf"
    )


def _read_numeric(handle, name, indices):
    return np.asarray(handle[name][:], dtype=float)[indices]


def _robust_sigma(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


def _finite_diagnostic_fields(handle, suffix):
    fields = []
    for name in handle:
        if not (name == suffix.lstrip("_") or name.endswith(suffix)):
            continue
        dataset = handle[name]
        if not isinstance(dataset, h5py.Dataset) or dataset.dtype.kind not in "biuf":
            continue
        values = np.asarray(dataset[:], dtype=float)
        if np.any(np.isfinite(values)):
            fields.append(name)
    return sorted(fields)


def _diagnostic_availability(handle):
    return {
        family: bool(_finite_diagnostic_fields(handle, suffix))
        for family, suffix in CONVERGENCE_FAMILIES
    }


def _final_sampler_values(handle):
    values = {}
    for name in ("accept_prob", "num_divergences"):
        if name not in handle:
            values[name] = np.array([], dtype=float)
            continue
        array = np.asarray(handle[name][:], dtype=float)
        values[name] = array[np.isfinite(array)]
    return values


def _sampler_summary(handle, label, path):
    availability = _diagnostic_availability(handle)
    values = _final_sampler_values(handle)
    accept = values["accept_prob"]
    divergences = values["num_divergences"]
    return {
        "catalog": label,
        "path": str(path),
        "n_objects": int(len(handle["object_id"])),
        "acceptance_available": availability["Acceptance"],
        "divergences_available": availability["Divergences"],
        "rhat_available": availability["R-hat"],
        "ess_available": availability["ESS"],
        "acceptance_finite_count": int(accept.size),
        "acceptance_median": float(np.median(accept)) if accept.size else np.nan,
        "acceptance_p05": float(np.quantile(accept, 0.05)) if accept.size else np.nan,
        "acceptance_p95": float(np.quantile(accept, 0.95)) if accept.size else np.nan,
        "acceptance_fraction_lt_0p7": (
            float(np.mean(accept < 0.7)) if accept.size else np.nan
        ),
        "acceptance_fraction_lt_0p8": (
            float(np.mean(accept < 0.8)) if accept.size else np.nan
        ),
        "divergence_finite_count": int(divergences.size),
        "divergence_free_fraction": (
            float(np.mean(divergences == 0)) if divergences.size else np.nan
        ),
        "objects_with_divergences": (
            int(np.count_nonzero(divergences > 0))
            if divergences.size
            else np.nan
        ),
        "total_divergences": (
            int(np.sum(divergences)) if divergences.size else np.nan
        ),
        "maximum_divergences": (
            int(np.max(divergences)) if divergences.size else np.nan
        ),
    }


def _eligible_parameter_names(
    new_handle,
    old_handle,
    *,
    new_rows,
    old_rows,
):
    shared_names = set(new_handle).intersection(old_handle)
    parameters = []
    for name in sorted(shared_names):
        if name.endswith("_err") or name == "object_id":
            continue
        error_name = f"{name}_err"
        if error_name not in shared_names:
            continue
        if not _numeric_vector_dataset(new_handle, name, new_rows):
            continue
        if not _numeric_vector_dataset(old_handle, name, old_rows):
            continue
        if not _numeric_vector_dataset(new_handle, error_name, new_rows):
            continue
        if not _numeric_vector_dataset(old_handle, error_name, old_rows):
            continue
        parameters.append(name)
    return parameters


def _parameter_metrics(
    name,
    new_handle,
    old_handle,
    new_indices,
    old_indices,
    new_divergences,
):
    new_value = _read_numeric(new_handle, name, new_indices)
    old_value = _read_numeric(old_handle, name, old_indices)
    new_error = _read_numeric(new_handle, f"{name}_err", new_indices)
    old_error = _read_numeric(old_handle, f"{name}_err", old_indices)

    finite = np.isfinite(new_value) & np.isfinite(old_value)
    finite_count = int(np.count_nonzero(finite))
    if finite_count:
        delta = new_value[finite] - old_value[finite]
        unique_new = int(np.unique(new_value[finite]).size)
        unique_old = int(np.unique(old_value[finite]).size)
    else:
        delta = np.array([], dtype=float)
        unique_new = 0
        unique_old = 0

    correlation = np.nan
    rank_correlation = np.nan
    if finite_count >= 3 and unique_new >= 2 and unique_old >= 2:
        correlation = float(np.corrcoef(old_value[finite], new_value[finite])[0, 1])
        rank_correlation = float(
            spearmanr(old_value[finite], new_value[finite]).statistic
        )

    finite_pull = (
        finite
        & np.isfinite(new_error)
        & np.isfinite(old_error)
        & (new_error >= 0)
        & (old_error >= 0)
        & (np.hypot(new_error, old_error) > 0)
    )
    pull = (
        (new_value[finite_pull] - old_value[finite_pull])
        / np.hypot(new_error[finite_pull], old_error[finite_pull])
    )
    finite_uncertainty = (
        np.isfinite(new_error)
        & np.isfinite(old_error)
        & (new_error > 0)
        & (old_error > 0)
    )
    uncertainty_ratio = new_error[finite_uncertainty] / old_error[finite_uncertainty]

    divergent_fraction = np.nan
    if new_divergences is not None and finite_count:
        divergent_fraction = float(np.mean(new_divergences[finite] > 0))

    median_pull = float(np.median(pull)) if pull.size else np.nan
    pull_nmad = _robust_sigma(pull)
    fraction_pull_gt3 = float(np.mean(np.abs(pull) > 3)) if pull.size else np.nan
    instability_score = (
        (abs(median_pull) if np.isfinite(median_pull) else 0.0)
        + (pull_nmad if np.isfinite(pull_nmad) else 0.0)
        + (5.0 * fraction_pull_gt3 if np.isfinite(fraction_pull_gt3) else 0.0)
        + (
            1.0 - abs(rank_correlation)
            if np.isfinite(rank_correlation)
            else 1.0
        )
    )

    row = {
        "parameter": name,
        "matched_finite": finite_count,
        "matched_fraction": finite_count / len(new_indices),
        "pearson_r": correlation,
        "spearman_rho": rank_correlation,
        "median_new_minus_old": float(np.median(delta)) if delta.size else np.nan,
        "nmad_new_minus_old": _robust_sigma(delta),
        "pull_finite": int(pull.size),
        "median_pull": median_pull,
        "pull_nmad": pull_nmad,
        "fraction_abs_pull_lt1": (
            float(np.mean(np.abs(pull) < 1)) if pull.size else np.nan
        ),
        "fraction_abs_pull_lt2": (
            float(np.mean(np.abs(pull) < 2)) if pull.size else np.nan
        ),
        "fraction_abs_pull_gt3": fraction_pull_gt3,
        "median_new_to_old_uncertainty": (
            float(np.median(uncertainty_ratio))
            if uncertainty_ratio.size
            else np.nan
        ),
        "new_divergent_fraction": divergent_fraction,
        "instability_priority_score": instability_score,
    }
    arrays = {
        "new_value": new_value,
        "old_value": old_value,
        "new_error": new_error,
        "old_error": old_error,
        "finite": finite,
        "pull": pull,
    }
    return row, arrays


def _model_variant(handle):
    if "model_variant" not in handle:
        return "not stored"
    values = _decode_strings(handle["model_variant"][:])
    unique, counts = np.unique(values, return_counts=True)
    if not unique.size:
        return "not stored"
    return str(unique[np.argmax(counts)])


def _plot_availability(ax, new_availability, old_availability, new_label, old_label):
    families = [family for family, _ in CONVERGENCE_FAMILIES]
    matrix = np.asarray(
        [
            [new_availability[family] for family in families],
            [old_availability[family] for family in families],
        ],
        dtype=float,
    )
    ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(families)), families, rotation=25, ha="right")
    ax.set_yticks([0, 1], [new_label, old_label])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                "available" if matrix[row, column] else "not stored",
                ha="center",
                va="center",
                fontsize=8,
            )
    ax.set_title("True sampler-diagnostic availability")


def _plot_acceptance(ax, sampler_values, labels):
    plotted = False
    for values, label, color in zip(
        sampler_values,
        labels,
        ("tab:blue", "tab:orange"),
    ):
        accept = values["accept_prob"]
        if not accept.size:
            continue
        ordered = np.sort(accept)
        cumulative = np.arange(1, ordered.size + 1) / ordered.size
        ax.plot(
            ordered,
            cumulative,
            lw=2,
            label=f"{label} (median={np.median(accept):.3f})",
            color=color,
        )
        plotted = True
    for reference, style in ((0.7, "--"), (0.8, ":")):
        ax.axvline(reference, color="0.35", ls=style, lw=1)
        ax.text(
            reference,
            0.03,
            f"{reference:.1f}",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
            color="0.35",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean acceptance probability")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("Final NUTS acceptance")
    ax.grid(alpha=0.2)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "not stored in either catalog", ha="center", va="center")


def _plot_divergence_free(ax, summaries, labels):
    values = [summary["divergence_free_fraction"] for summary in summaries]
    for index, (value, label, color) in enumerate(
        zip(values, labels, ("tab:blue", "tab:orange"))
    ):
        if np.isfinite(value):
            ax.bar(index, 100 * value, color=color, width=0.65)
            ax.text(
                index,
                100 * value + 1,
                f"{100 * value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
            summary = summaries[index]
            ax.text(
                index,
                3,
                (
                    f"{int(summary['objects_with_divergences']):,} objects\n"
                    f"{int(summary['total_divergences']):,} total"
                ),
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="white",
            )
        else:
            ax.bar(index, 0, color="0.85", width=0.65)
            ax.text(index, 4, "not stored", rotation=90, ha="center", va="bottom")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(0, 108)
    ax.set_ylabel("objects with zero divergences (%)")
    ax.set_title("Final NUTS divergence-free fraction")
    ax.grid(axis="y", alpha=0.2)


def _plot_divergence_vs_acceptance(ax, sampler_values, labels):
    plotted = False
    for values, label, color in zip(
        sampler_values,
        labels,
        ("tab:blue", "tab:orange"),
    ):
        accept = values["accept_prob"]
        divergences = values["num_divergences"]
        count = min(accept.size, divergences.size)
        if not count:
            continue
        accept = accept[:count]
        divergences = divergences[:count]
        bins = np.linspace(0, 1, 21)
        centers = 0.5 * (bins[:-1] + bins[1:])
        fraction = np.full(centers.size, np.nan)
        bin_count = np.zeros(centers.size, dtype=int)
        for index in range(centers.size):
            keep = (accept >= bins[index]) & (
                accept <= bins[index + 1]
                if index == centers.size - 1
                else accept < bins[index + 1]
            )
            bin_count[index] = np.count_nonzero(keep)
            if bin_count[index]:
                fraction[index] = np.mean(divergences[keep] > 0)
        keep = bin_count >= 10
        ax.plot(
            centers[keep],
            100 * fraction[keep],
            marker="o",
            ms=4,
            lw=1.5,
            color=color,
            label=label,
        )
        plotted = True
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("mean acceptance probability")
    ax.set_ylabel("objects with divergences (%)")
    ax.set_title("Divergences versus acceptance")
    ax.grid(alpha=0.2)
    if plotted:
        ax.legend(fontsize=8)


def _plot_core_metric(
    ax,
    parameter_summary,
    atlas_parameters,
    *,
    column,
    title,
    reference=None,
    x_label=None,
):
    selected = parameter_summary[
        parameter_summary["parameter"].isin(atlas_parameters)
    ].copy()
    if selected.empty:
        ax.text(0.5, 0.5, "no eligible shared parameters", ha="center", va="center")
        ax.set_axis_off()
        return
    selected["_order"] = selected["parameter"].map(
        {name: index for index, name in enumerate(atlas_parameters)}
    )
    selected = selected.sort_values("_order", ascending=False)
    values = selected[column].to_numpy(dtype=float)
    positions = np.arange(len(selected))
    ax.barh(positions, values, color="tab:purple", alpha=0.8)
    ax.set_yticks(positions, selected["parameter"], fontsize=7)
    if reference is not None:
        ax.axvline(reference, color="black", lw=1, ls="--")
    ax.set_title(title)
    ax.set_xlabel(x_label or column)
    ax.grid(axis="x", alpha=0.2)


def _plot_overview(
    output_path,
    *,
    new_handle,
    old_handle,
    new_label,
    old_label,
    new_path,
    old_path,
    sampler_summaries,
    parameter_summary,
    atlas_parameters,
    matched_count,
):
    new_availability = _diagnostic_availability(new_handle)
    old_availability = _diagnostic_availability(old_handle)
    sampler_values = [
        _final_sampler_values(new_handle),
        _final_sampler_values(old_handle),
    ]
    labels = [new_label, old_label]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    _plot_availability(
        axes[0, 0],
        new_availability,
        old_availability,
        new_label,
        old_label,
    )
    _plot_acceptance(axes[0, 1], sampler_values, labels)
    _plot_divergence_free(axes[0, 2], sampler_summaries, labels)
    _plot_divergence_vs_acceptance(axes[1, 0], sampler_values, labels)
    _plot_core_metric(
        axes[1, 1],
        parameter_summary,
        atlas_parameters[:8],
        column="spearman_rho",
        title="Cross-fit rank agreement",
        reference=1.0,
        x_label=r"Spearman $\rho$",
    )
    _plot_core_metric(
        axes[1, 2],
        parameter_summary,
        atlas_parameters[:8],
        column="pull_nmad",
        title="Cross-fit normalized-difference width",
        reference=1.0,
        x_label="pull NMAD",
    )

    new_variant = _model_variant(new_handle)
    old_variant = _model_variant(old_handle)
    fig.suptitle(
        "Light-curve fit convergence and cross-fit stability\n"
        f"{new_label}: {new_variant}  |  {old_label}: {old_variant}",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.012,
        (
            f"Matched objects: {matched_count:,}. Cross-fit parameter agreement is "
            "not an MCMC convergence statistic; the model variants differ. "
            f"New: {Path(new_path).name} | Old: {Path(old_path).name}"
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _robust_identity_limits(old_value, new_value, finite):
    combined = np.concatenate([old_value[finite], new_value[finite]])
    if combined.size < 3:
        return None
    low, high = np.nanquantile(combined, [0.005, 0.995])
    if not (np.isfinite(low) and np.isfinite(high)):
        return None
    if high <= low:
        padding = max(abs(low) * 0.05, 1e-6)
        return low - padding, high + padding
    padding = 0.04 * (high - low)
    return low - padding, high + padding


def _plot_parameter_panel(
    ax,
    name,
    row,
    arrays,
    new_divergences,
    new_label,
    old_label,
):
    finite = arrays["finite"]
    old_value = arrays["old_value"]
    new_value = arrays["new_value"]
    limits = _robust_identity_limits(old_value, new_value, finite)
    if limits is None:
        ax.text(0.5, 0.5, f"{name}\ninsufficient finite values", ha="center", va="center")
        ax.set_axis_off()
        return

    nondivergent = finite
    divergent = np.zeros_like(finite)
    if new_divergences is not None:
        divergent = finite & (new_divergences > 0)
        nondivergent = finite & ~divergent
    ax.scatter(
        old_value[nondivergent],
        new_value[nondivergent],
        s=4,
        alpha=0.12,
        color="0.15",
        linewidths=0,
        rasterized=True,
        label="new: no divergence",
    )
    if np.any(divergent):
        ax.scatter(
            old_value[divergent],
            new_value[divergent],
            s=6,
            alpha=0.35,
            color="tab:red",
            linewidths=0,
            rasterized=True,
            label="new: ≥1 divergence",
        )
    ax.plot(limits, limits, color="tab:blue", lw=1.2, ls="--")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(old_label)
    ax.set_ylabel(new_label)
    ax.set_title(name, fontsize=10)
    ax.text(
        0.03,
        0.97,
        (
            f"N={int(row['matched_finite']):,}\n"
            rf"$\rho_s$={row['spearman_rho']:.3f}"
            f"\npull median={row['median_pull']:+.2f}"
            f"\npull NMAD={row['pull_nmad']:.2f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax.grid(alpha=0.15)


def _plot_atlas(
    output_path,
    *,
    parameter_summary,
    parameter_arrays,
    atlas_parameters,
    new_divergences,
    new_label,
    old_label,
):
    eligible = [
        name for name in atlas_parameters if name in parameter_arrays
    ]
    with PdfPages(output_path) as pdf:
        if not eligible:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(
                0.5,
                0.5,
                "No requested parameters had enough matched finite values.",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            return

        rows_by_name = parameter_summary.set_index("parameter")
        panels_per_page = 4
        for start in range(0, len(eligible), panels_per_page):
            names = eligible[start : start + panels_per_page]
            fig, axes = plt.subplots(2, 2, figsize=(11, 10))
            axes = axes.ravel()
            for ax, name in zip(axes, names):
                _plot_parameter_panel(
                    ax,
                    name,
                    rows_by_name.loc[name],
                    parameter_arrays[name],
                    new_divergences,
                    new_label,
                    old_label,
                )
            for ax in axes[len(names) :]:
                ax.set_axis_off()
            fig.suptitle(
                "Matched-object LC parameter stability\n"
                "Identity agreement is diagnostic, not proof of sampler convergence",
                fontsize=14,
            )
            if new_divergences is not None:
                handles, labels = axes[0].get_legend_handles_labels()
                if handles:
                    fig.legend(
                        handles,
                        labels,
                        loc="lower center",
                        ncol=2,
                        fontsize=8,
                    )
            fig.tight_layout(rect=(0, 0.035, 1, 0.95))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def compare_light_curve_fit_catalogs(
    new_path,
    old_path,
    *,
    output_dir,
    new_label="new",
    old_label="old",
    atlas_parameters=DEFAULT_ATLAS_PARAMETERS,
    min_matched=100,
):
    """Create sampler-convergence and cross-fit stability diagnostics."""

    new_path = Path(new_path)
    old_path = Path(old_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(new_path, "r") as new_handle, h5py.File(
        old_path, "r"
    ) as old_handle:
        new_ids = _load_object_ids(new_handle, new_path)
        old_ids = _load_object_ids(old_handle, old_path)
        new_indices, old_indices = _matched_indices(new_ids, old_ids)
        if len(new_indices) < int(min_matched):
            raise ValueError(
                f"Only {len(new_indices)} object_id values overlap; at least "
                f"{int(min_matched)} are required."
            )

        sampler_summaries = [
            _sampler_summary(new_handle, new_label, new_path),
            _sampler_summary(old_handle, old_label, old_path),
        ]
        sampler_summary = pd.DataFrame(sampler_summaries)

        new_divergences = None
        if "num_divergences" in new_handle:
            new_divergences = _read_numeric(
                new_handle,
                "num_divergences",
                new_indices,
            )

        eligible = _eligible_parameter_names(
            new_handle,
            old_handle,
            new_rows=len(new_ids),
            old_rows=len(old_ids),
        )
        rows = []
        parameter_arrays = {}
        for name in eligible:
            row, arrays = _parameter_metrics(
                name,
                new_handle,
                old_handle,
                new_indices,
                old_indices,
                new_divergences,
            )
            if (
                row["matched_finite"] < int(min_matched)
                or not np.isfinite(row["spearman_rho"])
            ):
                continue
            rows.append(row)
            parameter_arrays[name] = arrays

        parameter_summary = pd.DataFrame(rows)
        if parameter_summary.empty:
            raise ValueError("No shared numeric value/error parameter pairs were eligible.")
        parameter_summary = parameter_summary.sort_values(
            "instability_priority_score",
            ascending=False,
            kind="stable",
        ).reset_index(drop=True)

        overview_path = output_dir / "light_curve_fit_convergence_comparison.pdf"
        atlas_path = output_dir / "light_curve_fit_parameter_stability_atlas.pdf"
        sampler_csv_path = output_dir / "light_curve_fit_sampler_summary.csv"
        parameter_csv_path = output_dir / "light_curve_fit_parameter_stability.csv"

        sampler_summary.to_csv(sampler_csv_path, index=False)
        parameter_summary.to_csv(parameter_csv_path, index=False)
        _plot_overview(
            overview_path,
            new_handle=new_handle,
            old_handle=old_handle,
            new_label=new_label,
            old_label=old_label,
            new_path=new_path,
            old_path=old_path,
            sampler_summaries=sampler_summaries,
            parameter_summary=parameter_summary,
            atlas_parameters=tuple(atlas_parameters),
            matched_count=len(new_indices),
        )
        _plot_atlas(
            atlas_path,
            parameter_summary=parameter_summary,
            parameter_arrays=parameter_arrays,
            atlas_parameters=tuple(atlas_parameters),
            new_divergences=new_divergences,
            new_label=new_label,
            old_label=old_label,
        )

    return {
        "overview_pdf": str(overview_path),
        "atlas_pdf": str(atlas_path),
        "sampler_summary_csv": str(sampler_csv_path),
        "parameter_stability_csv": str(parameter_csv_path),
        "matched_objects": int(len(new_indices)),
        "eligible_parameters": int(len(parameter_summary)),
    }


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare true sampler diagnostics and matched-object parameter "
            "stability between new and old light-curve fit HDF5 catalogs."
        )
    )
    parser.add_argument("new_h5")
    parser.add_argument("old_h5")
    parser.add_argument(
        "--output-dir",
        default="plots/light_curve_fit_convergence",
    )
    parser.add_argument("--new-label", default="new")
    parser.add_argument("--old-label", default="old")
    parser.add_argument("--min-matched", type=int, default=100)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    outputs = compare_light_curve_fit_catalogs(
        args.new_h5,
        args.old_h5,
        output_dir=args.output_dir,
        new_label=args.new_label,
        old_label=args.old_label,
        min_matched=args.min_matched,
    )
    for name, value in outputs.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
