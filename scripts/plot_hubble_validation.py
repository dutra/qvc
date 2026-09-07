#!/usr/bin/env python3
"""Plot recovery and selection diagnostics for a Hubble-validation campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

from astropy.cosmology import Flatw0waCDM
import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.integrate import cumulative_trapezoid
from scipy.special import expit
from scipy.stats import gaussian_kde


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble.hubble_validation import (
    ARM_NAMES,
    CORNER_PARAMETERS,
    collect_recovery_fragments,
    ensemble_summary,
    incomplete_recovery_report,
    write_dataframe_atomic,
)


ARM_STYLE = {
    "all": {"color": "0.35", "label": "All, LF catalog"},
    "selected_uncorrected": {
        "color": "tab:red",
        "label": "no completeness correction",
    },
    "selected_oracle": {"color": "#2ca02c", "label": "Detected, ideal correction"},
    "selected_estimated": {
        "color": "tab:blue",
        "label": "with completeness correction",
    },
}
CATALOG_STYLE = {
    "all": {"color": "0.35", "label": "All, LF catalog"},
    "detected": {"color": "tab:red", "label": "Detected catalog"},
}
PARAMETER_LABEL = {
    "M0_agn": r"$M^0_{\rm AGN}$",
    "alpha_agn": r"$\alpha_{\rm AGN}$",
    "beta_agn": r"$\beta_{\rm AGN}$",
    "Om0": r"$\Omega_{m,0}$",
    "w0": r"$w_0$",
    "wa": r"$w_a$",
}
POSTERIOR_PARAMETERS = (
    "M0_agn",
    "alpha_agn",
    "beta_agn",
    "Om0",
    "w0",
    "wa",
)
COMPLETENESS_CONTOUR_MAG_MIN = 21.5


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, help="Campaign directory containing manifest.json and recovery.csv.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--min-contour-points", type=int, default=8)
    parser.add_argument(
        "--no-points",
        action="store_true",
        help="Hide the individual recovered-median points in the corner plot.",
    )
    parser.add_argument(
        "--single",
        type=int,
        metavar="REALIZATION",
        help=(
            "Create posterior-corner, posterior-uncertainty Hubble, and "
            "completeness-effects plots for one realization."
        ),
    )
    return parser


def _truth_from_manifest(manifest: dict) -> dict[str, float]:
    truth = manifest["configuration"]["truth"]
    return {
        "alpha_agn": float(truth["alpha_agn"]),
        "beta_agn": float(truth["beta_agn"]),
        "Om0": float(truth["om0"]),
        "w0": float(truth["w0"]),
        "wa": float(truth["wa"]),
    }


def _cosmology_truth_from_manifest(manifest: dict) -> dict[str, float]:
    truth = manifest["configuration"]["truth"]
    return {
        "H0": float(truth.get("h0", 70.0)),
        "Om0": float(truth["om0"]),
        "w0": float(truth["w0"]),
        "wa": float(truth["wa"]),
    }


def _distance_modulus(parameters: dict[str, float], redshift: np.ndarray) -> np.ndarray | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cosmology = Flatw0waCDM(
                H0=float(parameters["H0"]),
                Om0=float(parameters["Om0"]),
                w0=float(parameters["w0"]),
                wa=float(parameters["wa"]),
            )
            distance_modulus = np.asarray(cosmology.distmod(redshift).value, dtype=float)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return distance_modulus if np.all(np.isfinite(distance_modulus)) else None


def plot_hubble_recovery(
    recovery: pd.DataFrame,
    manifest: dict,
    output_pdf: Path,
    *,
    output_png: Path | None = None,
    dpi: int = 220,
) -> Path:
    """Plot recovered median cosmologies with across-realization 1-sigma bars."""

    complete = recovery.loc[recovery["status"] == "complete"].copy()
    required = {"Om0_q50", "w0_q50", "wa_q50"}
    missing = sorted(required - set(complete.columns))
    if missing:
        raise KeyError(f"Recovery table is missing cosmology medians: {missing}")
    truth = _cosmology_truth_from_manifest(manifest)
    z_min, z_max = manifest["configuration"].get("z_range", (0.1, 4.0))
    redshift = np.linspace(max(float(z_min), 1e-3), float(z_max), 10)
    dense_redshift = np.linspace(max(float(z_min), 1e-3), float(z_max), 500)
    truth_mu = _distance_modulus(truth, redshift)
    dense_truth_mu = _distance_modulus(truth, dense_redshift)
    if truth_mu is None or dense_truth_mu is None:
        raise ValueError("Injected cosmology does not produce finite distance moduli.")

    figure, (ax_hubble, ax_residual) = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.3),
        sharex=True,
        gridspec_kw={"height_ratios": (2.15, 1.0), "hspace": 0.05},
    )
    ax_hubble.plot(
        dense_redshift,
        dense_truth_mu,
        color="black",
        lw=2.0,
        label="Injected FlatΛCDM truth",
        zorder=5,
    )
    plotted = 0
    for arm in ARM_NAMES:
        subset = complete.loc[complete["arm"] == arm]
        recovered_mu = []
        recovered_parameters = []
        for _, row in subset.iterrows():
            parameters = {
                "H0": row.get("H0_q50", truth["H0"]),
                "Om0": row["Om0_q50"],
                "w0": row["w0_q50"],
                "wa": row["wa_q50"],
            }
            curve = _distance_modulus(parameters, dense_redshift)
            if curve is not None:
                recovered_mu.append(curve)
                recovered_parameters.append(parameters)
        if not recovered_mu:
            continue
        curves = np.asarray(recovered_mu, dtype=float)
        median_parameters = {
            name: float(np.median([values[name] for values in recovered_parameters]))
            for name in ("H0", "Om0", "w0", "wa")
        }
        dense_median = _distance_modulus(median_parameters, dense_redshift)
        if dense_median is None:
            continue
        curves_at_nodes = np.asarray(
            [np.interp(redshift, dense_redshift, curve) for curve in curves],
            dtype=float,
        )
        lower, _, upper = np.quantile(
            curves_at_nodes, [0.16, 0.50, 0.84], axis=0
        )
        median = np.interp(redshift, dense_redshift, dense_median)
        yerr = np.vstack(
            (
                np.clip(median - lower, 0.0, None),
                np.clip(upper - median, 0.0, None),
            )
        )
        style = ARM_STYLE[arm]
        label = f"{style['label']} (N={curves.shape[0]})"
        ax_hubble.plot(
            dense_redshift,
            dense_median,
            color=style["color"],
            lw=1.4,
            label=label,
            zorder=3,
        )
        ax_hubble.errorbar(
            redshift,
            median,
            yerr=yerr,
            color=style["color"],
            fmt="o",
            markersize=3.8,
            elinewidth=1.0,
            capsize=2.0,
            label="_nolegend_",
            zorder=4,
        )
        ax_residual.plot(
            dense_redshift,
            dense_median - dense_truth_mu,
            color=style["color"],
            lw=1.4,
            zorder=3,
        )
        ax_residual.errorbar(
            redshift,
            median - truth_mu,
            yerr=yerr,
            color=style["color"],
            fmt="o",
            markersize=3.8,
            elinewidth=1.0,
            capsize=2.0,
            zorder=4,
        )
        plotted += 1
    if plotted == 0:
        plt.close(figure)
        raise ValueError("No successful arm has a finite recovered cosmology.")

    ax_residual.axhline(0.0, color="black", lw=1.2, ls="--")
    ax_hubble.set_xscale("linear")
    ax_hubble.set_ylabel(r"Distance modulus $\mu$ (mag)")
    ax_residual.set_xlabel("Redshift")
    ax_residual.set_ylabel(r"$\Delta\mu$ from truth (mag)")
    ax_hubble.legend(
        loc="lower right",
        frameon=False,
        fontsize=11,
        ncol=1,
        handlelength=2.8,
        handletextpad=0.8,
        labelspacing=0.7,
    )
    figure.align_ylabels([ax_hubble, ax_residual])
    figure.tight_layout()
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight")
    if output_png is not None:
        figure.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_pdf


def _read_validation_catalogs(
    campaign: Path,
    realization: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parent_parts = []
    selected_parts = []
    columns = [
        "z",
        "apparent_mag_2500",
        "injected_detection_probability",
        "injected_detected",
    ]
    if realization is None:
        run_dirs = sorted((campaign / "runs").glob("seed_*"))
    else:
        run_dirs = [campaign / "runs" / f"seed_{realization:04d}"]
    for run_dir in run_dirs:
        parent_path = run_dir / "all.csv"
        selected_path = run_dir / "selected.csv"
        if not parent_path.is_file() or not selected_path.is_file():
            continue
        parent_parts.append(pd.read_csv(parent_path, usecols=columns))
        selected_parts.append(pd.read_csv(selected_path, usecols=["z", "apparent_mag_2500"]))
    if not parent_parts:
        raise FileNotFoundError("No paired all.csv and selected.csv validation catalogs were found.")
    return (
        pd.concat(parent_parts, ignore_index=True),
        pd.concat(selected_parts, ignore_index=True),
    )


def _representative_calibration_paths(
    campaign: Path,
    recovery: pd.DataFrame,
    realization: int | None = None,
) -> tuple[int, Path, Path, Path | None]:
    complete_estimated = recovery.loc[
        (recovery["status"] == "complete")
        & (recovery["arm"] == "selected_estimated")
    ]
    if realization is None:
        preferred = complete_estimated["realization"].astype(int).tolist()
        fallback = [
            int(path.name.removeprefix("seed_"))
            for path in sorted((campaign / "runs").glob("seed_*"))
        ]
    else:
        preferred = [realization]
        fallback = []
    for realization in [*preferred, *fallback]:
        run_dir = campaign / "runs" / f"seed_{realization:04d}"
        parent = run_dir / "calibration_parent.h5"
        detected = run_dir / "calibration_detected.csv"
        checkpoint = run_dir / "posterior_selected_estimated.h5"
        if parent.is_file() and detected.is_file():
            return realization, parent, detected, checkpoint if checkpoint.is_file() else None
    raise FileNotFoundError("No persisted calibration parent/detected pair was found.")


def _calibration_completeness_map(
    campaign: Path,
    recovery: pd.DataFrame,
    realization: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    realization, parent_path, detected_path, checkpoint_path = (
        _representative_calibration_paths(campaign, recovery, realization)
    )
    metadata = {
        "mag_support": (18.0, 24.5),
        "z_support": (0.0, 4.5),
        "n_mag": 65,
        "n_z": 45,
        "sigma_mag": 0.1,
        "sigma_z": 0.3,
    }
    if checkpoint_path is not None:
        with h5py.File(checkpoint_path, "r") as checkpoint:
            dataset_keys = {
                "mag_support": "completeness_map_magnitude_support",
                "z_support": "completeness_map_redshift_support",
                "n_mag": "completeness_map_n_magnitude_bins",
                "n_z": "completeness_map_n_redshift_bins",
                "sigma_mag": "completeness_smooth_sigma_mag",
                "sigma_z": "completeness_smooth_sigma_z",
            }
            for key, dataset in dataset_keys.items():
                if dataset in checkpoint:
                    value = np.asarray(checkpoint[dataset][...])
                    metadata[key] = value.tolist() if value.ndim else value.item()
    mag_edges = np.linspace(*metadata["mag_support"], int(metadata["n_mag"]) + 1)
    z_edges = np.linspace(*metadata["z_support"], int(metadata["n_z"]) + 1)
    with h5py.File(parent_path, "r") as parent_file:
        parent_mag = np.asarray(parent_file["apparent_mag_2500"][:], dtype=float)
        parent_z = np.asarray(parent_file["z"][:], dtype=float)
        count_scale = float(parent_file.attrs.get("mock_count_scale", 1.0))
    detected = pd.read_csv(detected_path, usecols=["z", "apparent_mag_2500"])
    parent_counts, _, _ = np.histogram2d(parent_mag, parent_z, bins=[mag_edges, z_edges])
    detected_counts, _, _ = np.histogram2d(
        detected["apparent_mag_2500"].to_numpy(dtype=float),
        detected["z"].to_numpy(dtype=float),
        bins=[mag_edges, z_edges],
    )
    dm = float(np.diff(mag_edges[:2])[0])
    dz = float(np.diff(z_edges[:2])[0])
    parent_smooth = gaussian_filter(
        parent_counts,
        sigma=(float(metadata["sigma_mag"]) / dm, float(metadata["sigma_z"]) / dz),
        mode="nearest",
    )
    detected_smooth = gaussian_filter(
        detected_counts,
        sigma=(float(metadata["sigma_mag"]) / dm, float(metadata["sigma_z"]) / dz),
        mode="nearest",
    )
    completeness = detected_smooth / (max(count_scale, 1e-12) * parent_smooth + 1e-12)
    completeness[parent_smooth < 1e-12] = np.nan
    completeness = np.clip(completeness, 0.0, 1.0)
    return completeness, mag_edges, z_edges, realization


def _relative_completeness_percent(completeness: np.ndarray) -> np.ndarray:
    """Normalize a completeness map to its finite maximum and express percent."""

    completeness = np.asarray(completeness, dtype=float)
    finite = np.isfinite(completeness)
    if not np.any(finite):
        return np.full_like(completeness, np.nan)
    maximum = float(np.max(completeness[finite]))
    if maximum <= 0.0:
        return np.where(finite, 0.0, np.nan)
    return 100.0 * completeness / maximum


def _add_relative_completeness_contours(
    axis,
    relative_completeness: np.ndarray,
    mag_edges: np.ndarray,
    z_edges: np.ndarray,
    min_magnitude: float = COMPLETENESS_CONTOUR_MAG_MIN,
):
    """Overlay percentage contours on the faint side of a completeness map."""

    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    contour_values = np.array(relative_completeness, dtype=float, copy=True)
    contour_values[mag_centers < float(min_magnitude), :] = np.nan
    finite = contour_values[np.isfinite(contour_values)]
    if finite.size == 0 or float(np.max(finite)) <= float(np.min(finite)):
        return None
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    contours = axis.contour(
        mag_centers,
        z_centers,
        contour_values.T,
        colors="white",
        linewidths=1.3,
    )
    labels = axis.clabel(
        contours,
        inline=True,
        fmt=lambda level: f"{level:.0f}%",
        fontsize=7,
    )
    for label in labels:
        rotation = float(label.get_rotation()) % 360.0
        if np.sin(np.deg2rad(rotation)) < 0.0:
            label.set_rotation((rotation + 180.0) % 360.0)
    return contours


def plot_completeness_effects(
    campaign: Path,
    recovery: pd.DataFrame,
    manifest: dict,
    output_pdf: Path,
    *,
    output_png: Path | None = None,
    dpi: int = 220,
    realization: int | None = None,
) -> Path:
    """Show how the injected hard-supported sigmoid reshapes the LF sample."""

    parent, selected = _read_validation_catalogs(campaign, realization)
    configuration = manifest["configuration"]
    m50 = float(configuration["selection"]["m50"])
    width = float(configuration["selection"]["width"])
    support = configuration.get("fit", {}).get(
        "completeness_magnitude_support", [18.5, 24.0]
    )
    support = tuple(float(value) for value in support)
    z_range = tuple(float(value) for value in configuration.get("z_range", [0.1, 4.0]))
    completeness, mag_edges, z_edges, _ = _calibration_completeness_map(
        campaign, recovery, realization
    )
    relative_completeness = _relative_completeness_percent(completeness)

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))
    ax_mag, ax_z, ax_sigmoid, ax_map = axes.ravel()
    magnitude_range = (
        min(float(parent["apparent_mag_2500"].quantile(0.002)), support[0] - 0.5),
        max(float(parent["apparent_mag_2500"].quantile(0.998)), support[1] + 0.5),
    )
    bins_mag = np.linspace(*magnitude_range, 55)
    ax_mag.hist(
        parent["apparent_mag_2500"], bins=bins_mag, density=True,
        histtype="step", lw=1.8, color=CATALOG_STYLE["all"]["color"],
        label=CATALOG_STYLE["all"]["label"],
    )
    ax_mag.hist(
        selected["apparent_mag_2500"], bins=bins_mag, density=True,
        histtype="step", lw=1.8, color=CATALOG_STYLE["detected"]["color"],
        label=CATALOG_STYLE["detected"]["label"],
    )
    ax_mag.set_xlabel(r"Apparent $m_{2500}$ (mag)")
    ax_mag.set_ylabel("Normalized density")
    distribution_handles, distribution_labels = ax_mag.get_legend_handles_labels()

    bins_z = np.linspace(*z_range, 35)
    ax_z.hist(
        parent["z"], bins=bins_z, density=True, histtype="step", lw=1.8,
        color=CATALOG_STYLE["all"]["color"], label=CATALOG_STYLE["all"]["label"],
    )
    ax_z.hist(
        selected["z"], bins=bins_z, density=True, histtype="step", lw=1.8,
        color=CATALOG_STYLE["detected"]["color"],
        label=CATALOG_STYLE["detected"]["label"],
    )
    ax_z.set_xlabel("Redshift")
    ax_z.set_ylabel("Normalized density")

    probability_bins = np.linspace(magnitude_range[0], magnitude_range[1], 45)
    bin_index = np.digitize(parent["apparent_mag_2500"], probability_bins) - 1
    centers = 0.5 * (probability_bins[:-1] + probability_bins[1:])
    fraction = np.full(centers.size, np.nan)
    lower = np.full(centers.size, np.nan)
    upper = np.full(centers.size, np.nan)
    detected_flag = parent["injected_detected"].astype(bool).to_numpy()
    for index in range(centers.size):
        mask = bin_index == index
        count = int(np.count_nonzero(mask))
        if count < 20:
            continue
        successes = int(np.count_nonzero(detected_flag[mask]))
        p = successes / count
        denominator = 1.0 + 1.0 / count
        center = (p + 0.5 / count) / denominator
        half = np.sqrt(p * (1.0 - p) / count + 0.25 / count**2) / denominator
        fraction[index] = p
        lower[index] = max(0.0, center - half)
        upper[index] = min(1.0, center + half)
    valid = np.isfinite(fraction)
    empirical_yerr = np.clip(
        np.vstack(
            (fraction[valid] - lower[valid], upper[valid] - fraction[valid])
        ),
        0.0,
        None,
    )
    ax_sigmoid.errorbar(
        centers[valid],
        fraction[valid],
        yerr=empirical_yerr,
        fmt="o",
        ms=3.5,
        capsize=2,
        color="tab:red",
        label="Realized detection fraction",
    )
    magnitude_grid = np.linspace(*magnitude_range, 600)
    truth_probability = expit(-(magnitude_grid - m50) / width)
    truth_probability[(magnitude_grid < support[0]) | (magnitude_grid > support[1])] = 0.0
    ax_sigmoid.plot(
        magnitude_grid, truth_probability, color="black", lw=2.0,
        label=rf"Injected sigmoid ($m_{{50}}={m50:g}$, $s={width:g}$)",
    )
    ax_sigmoid.set_ylim(-0.04, 1.04)
    ax_sigmoid.set_xlabel(r"Apparent $m_{2500}$ (mag)")
    ax_sigmoid.set_ylabel(r"$p_{\rm det}$")
    selection_handles, selection_labels = ax_sigmoid.get_legend_handles_labels()

    image = ax_map.pcolormesh(
        mag_edges,
        z_edges,
        relative_completeness.T,
        shading="auto",
        vmin=0.0,
        vmax=100.0,
        cmap="viridis",
        edgecolors="none",
        linewidth=0.0,
        antialiased=False,
        rasterized=True,
    )
    _add_relative_completeness_contours(
        ax_map,
        relative_completeness,
        mag_edges,
        z_edges,
    )
    ax_map.grid(False, which="both")
    ax_map.xaxis.grid(False, which="both")
    ax_map.yaxis.grid(False, which="both")
    ax_map.set_xlabel(r"Apparent $m_{2500}$ (mag)")
    ax_map.set_ylabel("Redshift")
    colorbar = figure.colorbar(image, ax=ax_map, pad=0.02)
    colorbar.set_label("Relative completeness (%)")

    figure.legend(
        distribution_handles + selection_handles,
        distribution_labels + selection_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
        fontsize=11,
        columnspacing=1.8,
        handlelength=2.8,
        handleheight=1.2,
        markerscale=1.5,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight")
    if output_png is not None:
        figure.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_pdf


def _axis_limits(recovery: pd.DataFrame, truth: dict[str, float]) -> dict[str, tuple[float, float]]:
    limits = {}
    for parameter in CORNER_PARAMETERS:
        values = recovery[f"{parameter}_q50"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        values = np.append(values, truth[parameter])
        lower, upper = np.quantile(values, [0.01, 0.99])
        span = float(upper - lower)
        if not np.isfinite(span) or span <= 0.0:
            span = max(abs(float(truth[parameter])) * 0.1, 0.1)
        limits[parameter] = (float(lower - 0.15 * span), float(upper + 0.15 * span))
    return limits


def _density_thresholds(density: np.ndarray, enclosed=(0.865, 0.393)) -> list[float]:
    density = np.asarray(density, dtype=float)
    flat = density[np.isfinite(density) & (density >= 0.0)]
    if flat.size == 0 or np.sum(flat) <= 0.0:
        raise ValueError("KDE density is empty.")
    ordered = np.sort(flat)[::-1]
    cumulative = np.cumsum(ordered)
    cumulative /= cumulative[-1]
    thresholds = []
    for probability in enclosed:
        index = min(int(np.searchsorted(cumulative, probability)), ordered.size - 1)
        thresholds.append(float(ordered[index]))
    return sorted(set(thresholds))


def _draw_contours(ax, x, y, xlim, ylim, *, color, min_points) -> bool:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[finite], np.asarray(y)[finite]
    if x.size < min_points or np.linalg.matrix_rank(np.cov(np.vstack((x, y)))) < 2:
        return False
    try:
        kde = gaussian_kde(np.vstack((x, y)))
        x_grid = np.linspace(*xlim, 90)
        y_grid = np.linspace(*ylim, 90)
        xx, yy = np.meshgrid(x_grid, y_grid)
        density = kde(np.vstack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
        levels = _density_thresholds(density)
        if len(levels) < 2 or not np.all(np.diff(levels) > 0.0):
            return False
        ax.contour(xx, yy, density, levels=levels, colors=[color], linewidths=(1.0, 1.8))
    except (ValueError, np.linalg.LinAlgError) as exc:
        warnings.warn(f"Skipping singular recovery contour: {exc}", RuntimeWarning)
        return False
    return True


def plot_median_recovery_corner(
    recovery: pd.DataFrame,
    truth: dict[str, float],
    output_pdf: Path,
    *,
    output_png: Path | None = None,
    dpi: int = 220,
    min_contour_points: int = 8,
    show_points: bool = True,
) -> Path:
    """Plot contours and, optionally, one point per recovered posterior median."""

    complete = recovery.loc[recovery["status"] == "complete"].copy()
    if complete.empty:
        raise ValueError("Recovery table contains no successful fits.")
    required = [f"{parameter}_q50" for parameter in CORNER_PARAMETERS]
    missing = sorted(set(required) - set(complete.columns))
    if missing:
        raise KeyError(f"Recovery table is missing median columns: {missing}")
    limits = _axis_limits(complete, truth)
    n_parameters = len(CORNER_PARAMETERS)
    figure, axes = plt.subplots(
        n_parameters,
        n_parameters,
        figsize=(12.0, 12.0),
        squeeze=False,
    )

    for row, y_parameter in enumerate(CORNER_PARAMETERS):
        for column, x_parameter in enumerate(CORNER_PARAMETERS):
            ax = axes[row, column]
            if column > row:
                ax.set_visible(False)
                continue
            xlim = limits[x_parameter]
            if row == column:
                for arm in ARM_NAMES:
                    subset = complete.loc[complete["arm"] == arm, f"{x_parameter}_q50"].to_numpy(dtype=float)
                    subset = subset[np.isfinite(subset)]
                    if subset.size == 0:
                        continue
                    style = ARM_STYLE[arm]
                    if subset.size >= 3 and np.std(subset) > 0.0:
                        try:
                            grid = np.linspace(*xlim, 200)
                            ax.plot(grid, gaussian_kde(subset)(grid), color=style["color"], lw=1.6)
                        except (ValueError, np.linalg.LinAlgError):
                            pass
                ax.axvline(truth[x_parameter], color="black", ls="--", lw=1.2)
                ax.set_xlim(xlim)
                ax.set_yticks([])
            else:
                ylim = limits[y_parameter]
                for arm in ARM_NAMES:
                    subset = complete.loc[complete["arm"] == arm]
                    x = subset[f"{x_parameter}_q50"].to_numpy(dtype=float)
                    y = subset[f"{y_parameter}_q50"].to_numpy(dtype=float)
                    style = ARM_STYLE[arm]
                    if show_points:
                        finite = np.isfinite(x) & np.isfinite(y)
                        ax.scatter(
                            x[finite],
                            y[finite],
                            s=15,
                            color=style["color"],
                            alpha=0.55,
                            edgecolors="none",
                            zorder=2,
                        )
                    _draw_contours(
                        ax,
                        x,
                        y,
                        xlim,
                        ylim,
                        color=style["color"],
                        min_points=min_contour_points,
                    )
                ax.axvline(truth[x_parameter], color="black", ls="--", lw=0.9)
                ax.axhline(truth[y_parameter], color="black", ls="--", lw=0.9)
                ax.plot(truth[x_parameter], truth[y_parameter], marker="+", color="black", ms=8, mew=1.5)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            if row != column:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            if row == n_parameters - 1:
                ax.set_xlabel(PARAMETER_LABEL[x_parameter])
            else:
                ax.set_xticklabels([])
            if column == 0 and row > 0:
                ax.set_ylabel(PARAMETER_LABEL[y_parameter])
            elif row != column:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=8)

    handles = [
        Line2D([0], [0], color=ARM_STYLE[arm]["color"], lw=2, label=ARM_STYLE[arm]["label"])
        for arm in ARM_NAMES
        if np.any(complete["arm"] == arm)
    ]
    handles.append(Line2D([0], [0], color="black", ls="--", lw=1.2, label="Injected truth"))
    figure.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=False,
        fontsize=19,
        handlelength=3.2,
        handleheight=1.5,
        handletextpad=0.9,
        labelspacing=0.9,
    )
    figure.tight_layout(
        rect=(0.03, 0.03, 0.98, 0.97),
        pad=0.25,
        h_pad=0.1,
        w_pad=0.1,
    )
    figure.subplots_adjust(hspace=0.0, wspace=0.0)
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight")
    if output_png is not None:
        figure.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_pdf


def _posterior_truth_from_manifest(manifest: dict) -> dict[str, float]:
    truth = manifest["configuration"]["truth"]
    return {
        "M0_agn": float(truth["m0_agn"]),
        "alpha_agn": float(truth["alpha_agn"]),
        "beta_agn": float(truth["beta_agn"]),
        "Om0": float(truth["om0"]),
        "w0": float(truth["w0"]),
        "wa": float(truth["wa"]),
    }


def _decode_checkpoint_strings(values) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in np.asarray(values).reshape(-1).tolist()
    )


def _load_validation_posterior(
    checkpoint_path: Path,
    manifest: dict,
) -> np.ndarray:
    """Load one checkpoint in the common injected AGN-pivot coordinate system."""

    checkpoint_path = Path(checkpoint_path)
    with h5py.File(checkpoint_path, "r") as checkpoint:
        required_datasets = {
            "flat_samples",
            "model_labels",
            "agn_pivot_observable_names",
            "agn_pivot_values",
        }
        missing_datasets = sorted(required_datasets - set(checkpoint.keys()))
        if missing_datasets:
            raise KeyError(
                f"Checkpoint {checkpoint_path} is missing datasets: {missing_datasets}"
            )
        samples = np.asarray(checkpoint["flat_samples"], dtype=float)
        labels = _decode_checkpoint_strings(checkpoint["model_labels"])
        pivot_names = _decode_checkpoint_strings(
            checkpoint["agn_pivot_observable_names"]
        )
        pivot_values = np.asarray(checkpoint["agn_pivot_values"], dtype=float).reshape(-1)

    if samples.ndim != 2 or samples.shape[1] != len(labels):
        raise ValueError(
            f"Checkpoint {checkpoint_path} has incompatible flat_samples and model_labels."
        )
    if len(set(labels)) != len(labels):
        raise ValueError(f"Checkpoint {checkpoint_path} has duplicate model labels.")
    missing_parameters = sorted(set(POSTERIOR_PARAMETERS) - set(labels))
    if missing_parameters:
        raise KeyError(
            f"Checkpoint {checkpoint_path} is missing parameters: {missing_parameters}"
        )
    if len(pivot_names) != pivot_values.size or len(set(pivot_names)) != len(pivot_names):
        raise ValueError(f"Checkpoint {checkpoint_path} has invalid AGN pivot metadata.")
    pivot_by_name = dict(zip(pivot_names, pivot_values))
    missing_pivots = sorted(
        {"log_sigma_uv", "log_tau_uv_rf"} - set(pivot_by_name)
    )
    if missing_pivots:
        raise KeyError(
            f"Checkpoint {checkpoint_path} is missing AGN pivots: {missing_pivots}"
        )

    index = {label: position for position, label in enumerate(labels)}
    selected = np.column_stack(
        [samples[:, index[parameter]] for parameter in POSTERIOR_PARAMETERS]
    )
    finite = np.all(np.isfinite(selected), axis=1)
    selected = selected[finite]
    if selected.shape[0] < 2:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has fewer than two finite posterior samples."
        )

    truth = manifest["configuration"]["truth"]
    m0_index = POSTERIOR_PARAMETERS.index("M0_agn")
    alpha_index = POSTERIOR_PARAMETERS.index("alpha_agn")
    beta_index = POSTERIOR_PARAMETERS.index("beta_agn")
    selected[:, m0_index] += selected[:, alpha_index] * (
        float(truth["log_sigma_pivot"]) - pivot_by_name["log_sigma_uv"]
    ) + selected[:, beta_index] * (
        float(truth["log_tau_pivot"]) - pivot_by_name["log_tau_uv_rf"]
    )
    return selected


def _posterior_axis_limits(
    arm_samples: dict[str, np.ndarray],
    truth: dict[str, float],
) -> list[tuple[float, float]]:
    limits = []
    for parameter_index, parameter in enumerate(POSTERIOR_PARAMETERS):
        values = np.concatenate(
            [samples[:, parameter_index] for samples in arm_samples.values()]
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"No finite posterior values found for {parameter}.")
        lower, upper = np.quantile(values, [0.005, 0.995])
        lower = min(float(lower), truth[parameter])
        upper = max(float(upper), truth[parameter])
        span = upper - lower
        if not np.isfinite(span) or span <= 0.0:
            span = max(abs(truth[parameter]) * 0.1, 0.1)
        limits.append((lower - 0.10 * span, upper + 0.10 * span))
    return limits


def _draw_posterior_density_1d(ax, values, limits, *, color) -> bool:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2 or np.std(finite) <= 0.0:
        return False
    density, edges = np.histogram(finite, bins=200, range=limits, density=True)
    density = gaussian_filter(density.astype(float), sigma=4.0, mode="nearest")
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.fill_between(centers, 0.0, density, color=color, alpha=0.10, linewidth=0.0)
    ax.plot(centers, density, color=color, lw=1.9)
    return True


def _draw_posterior_density_2d(ax, x, y, xlim, ylim, *, color) -> bool:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 3 or np.linalg.matrix_rank(np.cov(np.vstack((x, y)))) < 2:
        return False
    density, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=120,
        range=(xlim, ylim),
    )
    density = gaussian_filter(density, sigma=4.0, mode="nearest")
    try:
        levels = _density_thresholds(density, enclosed=(0.865, 0.393))
    except ValueError:
        return False
    if len(levels) < 2 or not np.all(np.diff(levels) > 0.0):
        return False
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    maximum = float(np.max(density))
    ax.contourf(
        x_centers,
        y_centers,
        density.T,
        levels=(levels[0], maximum * (1.0 + 1e-12)),
        colors=[color],
        alpha=0.09,
    )
    ax.contourf(
        x_centers,
        y_centers,
        density.T,
        levels=(levels[1], maximum * (1.0 + 1e-12)),
        colors=[color],
        alpha=0.11,
    )
    ax.contour(
        x_centers,
        y_centers,
        density.T,
        levels=levels,
        colors=[color],
        linewidths=(1.15, 1.9),
    )
    return True


def plot_realization_posterior_corner(
    arm_samples: dict[str, np.ndarray],
    truth: dict[str, float],
    output_pdf: Path,
    *,
    output_png: Path | None = None,
    dpi: int = 220,
) -> Path:
    """Overlay the posterior distributions from one validation realization."""

    if not arm_samples:
        raise ValueError("No posterior samples were supplied for the realization.")
    unknown_arms = sorted(set(arm_samples) - set(ARM_STYLE))
    if unknown_arms:
        raise ValueError(f"Unknown validation arms: {unknown_arms}")
    expected_shape = len(POSTERIOR_PARAMETERS)
    for arm, samples in arm_samples.items():
        samples = np.asarray(samples)
        if samples.ndim != 2 or samples.shape[1] != expected_shape:
            raise ValueError(
                f"Posterior samples for {arm} must have shape (N, {expected_shape})."
            )

    limits = _posterior_axis_limits(arm_samples, truth)
    n_parameters = len(POSTERIOR_PARAMETERS)
    figure, axes = plt.subplots(
        n_parameters,
        n_parameters,
        figsize=(14.0, 14.0),
        squeeze=False,
    )
    for row, y_parameter in enumerate(POSTERIOR_PARAMETERS):
        for column, x_parameter in enumerate(POSTERIOR_PARAMETERS):
            ax = axes[row, column]
            if column > row:
                ax.set_visible(False)
                continue
            xlim = limits[column]
            if row == column:
                for arm, samples in arm_samples.items():
                    _draw_posterior_density_1d(
                        ax,
                        samples[:, column],
                        xlim,
                        color=ARM_STYLE[arm]["color"],
                    )
                ax.axvline(truth[x_parameter], color="black", ls="--", lw=1.2)
                ax.set_xlim(xlim)
                ax.set_yticks([])
            else:
                ylim = limits[row]
                for arm, samples in arm_samples.items():
                    _draw_posterior_density_2d(
                        ax,
                        samples[:, column],
                        samples[:, row],
                        xlim,
                        ylim,
                        color=ARM_STYLE[arm]["color"],
                    )
                ax.axvline(truth[x_parameter], color="black", ls="--", lw=0.9)
                ax.axhline(truth[y_parameter], color="black", ls="--", lw=0.9)
                ax.plot(
                    truth[x_parameter],
                    truth[y_parameter],
                    marker="+",
                    color="black",
                    ms=8,
                    mew=1.5,
                )
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

            ax.grid(False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            if row == n_parameters - 1:
                ax.set_xlabel(PARAMETER_LABEL[x_parameter])
            else:
                ax.set_xticklabels([])
            if column == 0 and row > 0:
                ax.set_ylabel(PARAMETER_LABEL[y_parameter])
            elif row != column:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=8)

    handles = [
        Line2D(
            [0],
            [0],
            color=ARM_STYLE[arm]["color"],
            lw=2.2,
            label=ARM_STYLE[arm]["label"],
        )
        for arm in ARM_NAMES
        if arm in arm_samples
    ]
    handles.append(
        Line2D([0], [0], color="black", ls="--", lw=1.2, label="Injected truth")
    )
    figure.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=False,
        fontsize=23,
        handlelength=3.2,
        handleheight=1.5,
        handletextpad=0.9,
        labelspacing=0.9,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.06,
        top=0.985,
        hspace=0.0,
        wspace=0.0,
    )
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight")
    if output_png is not None:
        figure.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_pdf


def _load_realization_posteriors(
    campaign: Path,
    manifest: dict,
    realization: int,
) -> dict[str, np.ndarray]:
    """Load every usable arm posterior for one configured realization."""

    campaign = Path(campaign)
    configuration = manifest["configuration"]
    seed_start = int(configuration.get("seed_start", 0))
    n_runs = int(configuration["n_runs"])
    last_realization = seed_start + n_runs - 1
    if realization < seed_start or realization > last_realization:
        raise ValueError(
            f"realization {realization} is outside the configured range "
            f"{seed_start}-{last_realization}."
        )
    arms = tuple(configuration.get("arms", ARM_NAMES))
    unknown_arms = sorted(set(arms) - set(ARM_STYLE))
    if unknown_arms:
        raise ValueError(f"Manifest contains unknown validation arms: {unknown_arms}")
    run_dir = campaign / "runs" / f"seed_{realization:04d}"
    arm_samples = {}
    load_errors = []
    missing_arms = []
    for arm in arms:
        checkpoint_path = run_dir / f"posterior_{arm}.h5"
        if not checkpoint_path.is_file():
            missing_arms.append(arm)
            continue
        try:
            arm_samples[arm] = _load_validation_posterior(
                checkpoint_path,
                manifest,
            )
        except (KeyError, OSError, ValueError) as exc:
            load_errors.append(f"{checkpoint_path}: {exc}")
    notices = []
    if missing_arms:
        notices.append(f"missing arms: {', '.join(missing_arms)}")
    if load_errors:
        notices.append("unreadable checkpoints: " + "; ".join(load_errors))
    if notices:
        warnings.warn(
            f"Realization {realization} is incomplete ({'; '.join(notices)}).",
            RuntimeWarning,
        )
    if not arm_samples:
        raise ValueError(
            f"Realization {realization} contains no usable posterior checkpoints."
        )
    return arm_samples


def _posterior_distance_modulus_curves(
    samples: np.ndarray,
    truth_cosmology: dict[str, float],
    z_range,
    *,
    max_curves: int = 4000,
    n_redshift: int = 1400,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate posterior CPL distance-modulus curves on a common dense grid."""

    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != len(POSTERIOR_PARAMETERS):
        raise ValueError(
            f"Posterior samples must have shape (N, {len(POSTERIOR_PARAMETERS)})."
        )
    z_min, z_max = (float(value) for value in z_range)
    if not 0.0 < z_min < z_max:
        raise ValueError("z_range must contain two increasing positive values.")
    if max_curves <= 0 or n_redshift < 3:
        raise ValueError("max_curves must be positive and n_redshift must be at least 3.")
    if samples.shape[0] > max_curves:
        indices = np.linspace(0, samples.shape[0] - 1, max_curves, dtype=int)
        samples = samples[indices]

    index = {name: position for position, name in enumerate(POSTERIOR_PARAMETERS)}
    om0 = samples[:, index["Om0"]][:, None]
    w0 = samples[:, index["w0"]][:, None]
    wa = samples[:, index["wa"]][:, None]
    integration_redshift = np.linspace(0.0, z_max, int(n_redshift))
    one_plus_z = 1.0 + integration_redshift[None, :]
    dark_energy = np.power(one_plus_z, 3.0 * (1.0 + w0 + wa)) * np.exp(
        -3.0 * wa * integration_redshift[None, :] / one_plus_z
    )
    expansion_squared = om0 * one_plus_z**3 + (1.0 - om0) * dark_energy
    valid_expansion = np.all(np.isfinite(expansion_squared), axis=1) & np.all(
        expansion_squared > 0.0,
        axis=1,
    )
    expansion_squared = expansion_squared[valid_expansion]
    if expansion_squared.shape[0] == 0:
        raise ValueError("No posterior samples produce finite CPL expansion histories.")
    comoving_integral = cumulative_trapezoid(
        1.0 / np.sqrt(expansion_squared),
        integration_redshift,
        axis=1,
        initial=0.0,
    )
    plot_mask = integration_redshift >= z_min
    redshift = integration_redshift[plot_mask]
    luminosity_distance_mpc = (
        299792.458
        / float(truth_cosmology["H0"])
        * (1.0 + redshift[None, :])
        * comoving_integral[:, plot_mask]
    )
    distance_modulus = 5.0 * np.log10(luminosity_distance_mpc) + 25.0
    finite_curves = np.all(np.isfinite(distance_modulus), axis=1)
    if not np.any(finite_curves):
        raise ValueError("No posterior samples produce finite distance moduli.")
    return redshift, distance_modulus[finite_curves]


def plot_single_realization_hubble(
    arm_samples: dict[str, np.ndarray],
    manifest: dict,
    output_pdf: Path,
    *,
    output_png: Path | None = None,
    dpi: int = 220,
) -> Path:
    """Plot one realization with pointwise posterior 68% Hubble uncertainties."""

    if not arm_samples:
        raise ValueError("No posterior samples were supplied for the realization.")
    truth = _cosmology_truth_from_manifest(manifest)
    z_range = manifest["configuration"].get("z_range", (0.1, 4.0))
    figure, (ax_hubble, ax_residual) = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.3),
        sharex=True,
        gridspec_kw={"height_ratios": (2.15, 1.0), "hspace": 0.05},
    )
    dense_truth_mu = None
    plotted = 0
    for arm in ARM_NAMES:
        if arm not in arm_samples:
            continue
        redshift, curves = _posterior_distance_modulus_curves(
            arm_samples[arm],
            truth,
            z_range,
        )
        lower, median, upper = np.quantile(curves, [0.16, 0.50, 0.84], axis=0)
        if dense_truth_mu is None:
            dense_truth_mu = _distance_modulus(truth, redshift)
            if dense_truth_mu is None:
                plt.close(figure)
                raise ValueError("Injected cosmology does not produce finite distances.")
            ax_hubble.plot(
                redshift,
                dense_truth_mu,
                color="black",
                lw=2.0,
                label="Injected FlatΛCDM truth",
                zorder=5,
            )
        style = ARM_STYLE[arm]
        ax_hubble.fill_between(
            redshift,
            lower,
            upper,
            color=style["color"],
            alpha=0.12,
            linewidth=0.0,
            zorder=1,
        )
        ax_hubble.plot(
            redshift,
            median,
            color=style["color"],
            lw=1.6,
            label=style["label"],
            zorder=3,
        )
        ax_residual.fill_between(
            redshift,
            lower - dense_truth_mu,
            upper - dense_truth_mu,
            color=style["color"],
            alpha=0.12,
            linewidth=0.0,
            zorder=1,
        )
        ax_residual.plot(
            redshift,
            median - dense_truth_mu,
            color=style["color"],
            lw=1.6,
            zorder=3,
        )

        error_redshift = np.linspace(float(z_range[0]), float(z_range[1]), 10)
        error_median = np.interp(error_redshift, redshift, median)
        error_lower = np.interp(error_redshift, redshift, lower)
        error_upper = np.interp(error_redshift, redshift, upper)
        error_truth = np.interp(error_redshift, redshift, dense_truth_mu)
        yerr = np.vstack((error_median - error_lower, error_upper - error_median))
        ax_hubble.errorbar(
            error_redshift,
            error_median,
            yerr=yerr,
            color=style["color"],
            fmt="o",
            markersize=3.8,
            elinewidth=1.0,
            capsize=2.0,
            label="_nolegend_",
            zorder=4,
        )
        ax_residual.errorbar(
            error_redshift,
            error_median - error_truth,
            yerr=yerr,
            color=style["color"],
            fmt="o",
            markersize=3.8,
            elinewidth=1.0,
            capsize=2.0,
            label="_nolegend_",
            zorder=4,
        )
        plotted += 1

    if plotted == 0:
        plt.close(figure)
        raise ValueError("No successful arm has finite posterior Hubble curves.")
    ax_residual.axhline(0.0, color="black", lw=1.2)
    ax_hubble.set_xscale("linear")
    ax_hubble.set_ylabel(r"Distance modulus $\mu$ (mag)")
    ax_residual.set_xlabel("Redshift")
    ax_residual.set_ylabel(r"$\Delta\mu$ from truth (mag)")
    ax_hubble.legend(
        loc="lower right",
        frameon=False,
        fontsize=12,
        ncol=1,
        handlelength=2.8,
        handletextpad=0.8,
        labelspacing=0.7,
    )
    for axis in (ax_hubble, ax_residual):
        axis.grid(False)
        axis.tick_params(direction="in", top=True, right=True)
    figure.align_ylabels((ax_hubble, ax_residual))
    figure.subplots_adjust(left=0.14, right=0.98, bottom=0.10, top=0.98)
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight")
    if output_png is not None:
        figure.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_pdf


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    campaign = args.campaign.expanduser().resolve()
    manifest_path = campaign / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Expected manifest.json under campaign directory {campaign}.")
    manifest = json.loads(manifest_path.read_text())
    recovery = collect_recovery_fragments(campaign)
    incomplete = incomplete_recovery_report(recovery, manifest["configuration"])
    write_dataframe_atomic(incomplete, campaign / "incomplete_fits.csv")
    if recovery.empty:
        raise ValueError(
            f"Campaign {campaign} contains no recovery rows; see incomplete_fits.csv."
        )
    truth = _truth_from_manifest(manifest)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else campaign / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = ensemble_summary(recovery)
    write_dataframe_atomic(summary, campaign / "ensemble_summary.csv")
    plot_median_recovery_corner(
        recovery,
        truth,
        output_dir / "median_recovery_corner.pdf",
        output_png=output_dir / "median_recovery_corner.png",
        dpi=args.dpi,
        min_contour_points=args.min_contour_points,
        show_points=not args.no_points,
    )
    plot_hubble_recovery(
        recovery,
        manifest,
        output_dir / "hubble_diagram_recovery.pdf",
        output_png=output_dir / "hubble_diagram_recovery.png",
        dpi=args.dpi,
    )
    completeness_plot = output_dir / "completeness_effects.pdf"
    try:
        plot_completeness_effects(
            campaign,
            recovery,
            manifest,
            completeness_plot,
            output_png=output_dir / "completeness_effects.png",
            dpi=args.dpi,
        )
    except FileNotFoundError as exc:
        warnings.warn(
            f"Skipping completeness-effects plot because catalog artifacts are absent: {exc}",
            RuntimeWarning,
        )
    single_corner_plot = None
    single_hubble_plot = None
    single_completeness_plot = None
    if args.single is not None:
        arm_samples = _load_realization_posteriors(
            campaign,
            manifest,
            args.single,
        )
        single_output_dir = output_dir / "single_runs"
        single_corner_plot = (
            single_output_dir / f"posterior_corner_seed_{args.single:04d}.pdf"
        )
        plot_realization_posterior_corner(
            arm_samples,
            _posterior_truth_from_manifest(manifest),
            single_corner_plot,
            output_png=single_corner_plot.with_suffix(".png"),
            dpi=args.dpi,
        )
        single_hubble_plot = (
            single_output_dir / f"hubble_diagram_seed_{args.single:04d}.pdf"
        )
        plot_single_realization_hubble(
            arm_samples,
            manifest,
            single_hubble_plot,
            output_png=single_hubble_plot.with_suffix(".png"),
            dpi=args.dpi,
        )
        single_completeness_plot = (
            single_output_dir / f"completeness_effects_seed_{args.single:04d}.pdf"
        )
        try:
            plot_completeness_effects(
                campaign,
                recovery,
                manifest,
                single_completeness_plot,
                output_png=single_completeness_plot.with_suffix(".png"),
                dpi=args.dpi,
                realization=args.single,
            )
        except FileNotFoundError as exc:
            warnings.warn(
                "Skipping single-run completeness-effects plot because catalog "
                f"artifacts are absent: {exc}",
                RuntimeWarning,
            )
            single_completeness_plot = None
    print(f"Validation plot: {output_dir / 'median_recovery_corner.pdf'}")
    print(f"Hubble recovery plot: {output_dir / 'hubble_diagram_recovery.pdf'}")
    if completeness_plot.is_file():
        print(f"Completeness-effects plot: {completeness_plot}")
    if single_corner_plot is not None:
        print(f"Single-run posterior corner: {single_corner_plot}")
        print(f"Single-run Hubble plot: {single_hubble_plot}")
        if single_completeness_plot is not None:
            print(f"Single-run completeness-effects plot: {single_completeness_plot}")
    print(f"Ensemble summary: {campaign / 'ensemble_summary.csv'}")
    print(f"Incomplete fits: {campaign / 'incomplete_fits.csv'} ({len(incomplete)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
