"""Create separate LC and spectra Tier-1 diagnostic plots versus redshift."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np


DATA_DIR = Path("results/data")
LC_OUTPUT_DIR = Path("results/plots/tier1_cuts_vs_redshift_aug29_aug27_aug16")
SPECTRA_OUTPUT_DIR = Path("results/plots/tier1_spectra_cuts_vs_redshift_aug28")

LC_RUNS = (
    (
        "aug29_0158am_erlang_dho_svithennuts_psflogitnormal_iters1_"
        "svi10000lr0003w250s100_stonechisq_specaug280111pm_n8000_"
        "ce36392_chisq.h5",
        "Aug 29: Erlang DHO",
        "#0072B2",
    ),
    (
        "aug27_1219pm_erlang_legacy_svithennuts_psflogitnormal_iters3_"
        "svi4000w250s250_stonechisq_specaug240152pmv3_n8000_536f607_"
        "chisq.h5",
        "Aug 27: Erlang legacy",
        "#D55E00",
    ),
    (
        "aug16_0611am_mag_linear_svi4000w2000s500_f978995_stone.h5",
        "Aug 16: magnitude-linear",
        "#009E73",
    ),
)

SPECTRA_PATH = DATA_DIR / "jaxqsofit" / (
    "aug28_0111pm_spectrafit_f6a63a3_chisqgt20_N8000_nested_"
    "fhostpsf_delayedburst_galexauto.h5"
)

# Active thresholds in run_hubble.xonsh.
LC_CUTS = (
    ("loo_chi2_eff", 1.3, r"LC LOO $\chi^2_{\rm eff}$"),
    ("log_tau_uv_rf_rhat", 1.1, r"$\log\,\tau_{\rm UV,RF}$ R-hat"),
    ("log_sigma_uv_rhat", 1.1, r"$\log\,\sigma_{\rm UV}$ R-hat"),
)

SPECTRA_CUTS = (
    ("sed_reduced_chi2", 2.0, r"SED $\chi^2_\nu$"),
    ("spectroscopy_reduced_chi2", 1.3, r"Spectroscopy $\chi^2_\nu$"),
    ("joint_reduced_chi2", 1.3, r"Joint $\chi^2_\nu$"),
    ("m_2500_dereddened_rhat", 1.1, r"$m_{2500,\rm dered}$ R-hat"),
    (
        "m_2500_attenuated_model_rhat",
        1.1,
        r"$m_{2500,\rm atten}$ R-hat",
    ),
)


def load_columns(path: Path, cuts, group: str | None = None):
    with h5py.File(path, "r") as handle:
        source = handle[group] if group else handle
        data = {"z": np.asarray(source["z"], dtype=float)}
        for column, _, _ in cuts:
            data[column] = (
                np.asarray(source[column], dtype=float) if column in source else None
            )
    return data


def shared_limits(rows, column: str, cut: float):
    arrays = [
        row[column][np.isfinite(row[column]) & (row[column] > 0.0)]
        for row in rows
        if row[column] is not None
    ]
    arrays = [array for array in arrays if array.size]
    if not arrays:
        return cut / 2.0, cut * 2.0
    values = np.concatenate(arrays)
    return min(float(values.min()), cut) / 1.12, max(float(values.max()), cut) * 1.12


def log_ticks(lower: float, upper: float, cut: float):
    candidates = (0.05, 0.1, 0.2, 0.5, 0.7, 1.0, 1.1, 1.2, 1.3, 1.5,
                  2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0)
    return sorted({value for value in candidates if lower <= value <= upper} | {cut})


def style_panel(axis, limits, cut):
    axis.set_yscale("log")
    axis.set_xlim(0.0, 5.0)
    axis.set_ylim(*limits)
    axis.axhline(cut, color="#A51C30", lw=1.45, ls="--", zorder=1)
    axis.text(
        0.97,
        cut,
        f"cut = {cut:g}",
        transform=axis.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#A51C30",
    )
    axis.yaxis.set_major_locator(FixedLocator(log_ticks(*limits, cut)))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.grid(axis="y", which="major", color="0.86", lw=0.65, ls=":")


def plot_values(axis, z, values, cut, color):
    if values is None:
        axis.text(0.5, 0.5, "Column missing", transform=axis.transAxes,
                  ha="center", va="center", color="0.4")
        return
    finite = np.isfinite(z) & np.isfinite(values) & (values > 0.0)
    passed = finite & (values <= cut)
    failed = finite & (values > cut)
    axis.scatter(z[passed], values[passed], s=10, color=color, alpha=0.36,
                 edgecolors="none", rasterized=True)
    axis.scatter(z[failed], values[failed], s=15, color="#A51C30", marker="x",
                 alpha=0.78, linewidths=0.7, rasterized=True)
    count = int(finite.sum())
    if count:
        annotation = f"N = {count:,}\npass = {100.0 * passed.sum() / count:.1f}%"
    else:
        annotation = "No finite values"
    axis.text(0.03, 0.96, annotation, transform=axis.transAxes, ha="left",
              va="top", fontsize=8.5, color="0.25",
              bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72})


def plot_lc():
    rows = [load_columns(DATA_DIR / filename, LC_CUTS) for filename, *_ in LC_RUNS]
    limits = {column: shared_limits(rows, column, cut)
              for column, cut, _ in LC_CUTS}
    fig, axes = plt.subplots(len(rows), len(LC_CUTS), figsize=(13.0, 10.5), sharex=True)
    for row_index, ((_, label, color), data) in enumerate(zip(LC_RUNS, rows)):
        for col_index, (column, cut, title) in enumerate(LC_CUTS):
            axis = axes[row_index, col_index]
            style_panel(axis, limits[column], cut)
            plot_values(axis, data["z"], data[column], cut, color)
            if row_index == 0:
                axis.set_title(title, pad=7)
            if row_index == len(rows) - 1:
                axis.set_xlabel("Redshift")
        axes[row_index, 0].text(-0.21, 0.5, label,
                                transform=axes[row_index, 0].transAxes,
                                rotation=90, ha="center", va="center",
                                fontsize=12, color=color)
    fig.supylabel("Light-curve Tier 1 diagnostic (log scale)", x=0.015, fontsize=13)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.075, top=0.94,
                        hspace=0.25, wspace=0.28)
    LC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = LC_OUTPUT_DIR / "lc_tier1_cuts_vs_redshift"
    fig.savefig(stem.with_suffix(".png"), dpi=250)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    return stem


def plot_spectra():
    data = load_columns(SPECTRA_PATH, SPECTRA_CUTS, group="catalog")
    limits = {column: shared_limits([data], column, cut)
              for column, cut, _ in SPECTRA_CUTS}
    fig, axes = plt.subplots(1, len(SPECTRA_CUTS), figsize=(18.5, 4.1), sharex=True)
    for axis, (column, cut, title) in zip(axes, SPECTRA_CUTS):
        style_panel(axis, limits[column], cut)
        plot_values(axis, data["z"], data[column], cut, "#7A5195")
        axis.set_title(title, pad=7)
        axis.set_xlabel("Redshift")
    fig.supylabel("Spectra Tier 1 diagnostic (log scale)", x=0.01, fontsize=13)
    fig.subplots_adjust(left=0.055, right=0.992, bottom=0.18, top=0.86, wspace=0.35)
    SPECTRA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = SPECTRA_OUTPUT_DIR / "spectra_tier1_cuts_vs_redshift"
    fig.savefig(stem.with_suffix(".png"), dpi=250)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    return stem


def main():
    with plt.style.context("src/style.mplstyle"):
        lc_stem = plot_lc()
        spectra_stem = plot_spectra()
    print(f"Saved {lc_stem.with_suffix('.png')}")
    print(f"Saved {lc_stem.with_suffix('.pdf')}")
    print(f"Saved {spectra_stem.with_suffix('.png')}")
    print(f"Saved {spectra_stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
