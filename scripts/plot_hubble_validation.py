#!/usr/bin/env python3
"""Plot median-recovery contours for a fixed-truth Hubble validation campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
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
    "all": {"color": "#1f77b4", "label": "All (unselected)"},
    "selected_uncorrected": {"color": "#d62728", "label": "Selected, uncorrected"},
    "selected_oracle": {"color": "#2ca02c", "label": "Selected, oracle"},
    "selected_estimated": {"color": "#9467bd", "label": "Selected, estimated"},
}
PARAMETER_LABEL = {
    "alpha_agn": r"$\alpha_{\rm AGN}$",
    "beta_agn": r"$\beta_{\rm AGN}$",
    "Om0": r"$\Omega_{m,0}$",
    "w0": r"$w_0$",
    "wa": r"$w_a$",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, help="Campaign directory containing manifest.json and recovery.csv.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--min-contour-points", type=int, default=8)
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
) -> Path:
    """Plot one point per run and contours of recovered posterior medians."""

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
                    ax.hist(
                        subset,
                        bins=max(5, min(12, int(np.sqrt(subset.size) * 2))),
                        range=xlim,
                        density=True,
                        histtype="step",
                        color=style["color"],
                        linewidth=1.2,
                    )
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
                    ax.scatter(x, y, s=12, alpha=0.48, color=style["color"], edgecolors="none")
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
    figure.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=False)
    figure.suptitle(
        "Distribution of recovered posterior medians across random realizations",
        y=0.995,
    )
    figure.tight_layout(rect=(0.03, 0.03, 0.98, 0.97))
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
    )
    print(f"Validation plot: {output_dir / 'median_recovery_corner.pdf'}")
    print(f"Ensemble summary: {campaign / 'ensemble_summary.csv'}")
    print(f"Incomplete fits: {campaign / 'incomplete_fits.csv'} ({len(incomplete)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
