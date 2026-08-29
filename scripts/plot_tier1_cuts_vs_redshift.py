"""Plot every current Tier-1 Hubble cut against redshift for selected LC runs."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
import numpy as np


DATA_DIR = Path("results/data")
OUTPUT_DIR = Path("results/plots/tier1_cuts_vs_redshift_aug29_aug27_aug16")

RUNS = (
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

# Active defaults set in run_hubble.xonsh on 2026-08-29.
TIER1_CUTS = (
    ("sed_reduced_chi2", 2.0, r"SED $\chi^2_\nu$"),
    ("spectroscopy_reduced_chi2", 1.3, r"Spectroscopy $\chi^2_\nu$"),
    ("joint_reduced_chi2", 1.3, r"Joint $\chi^2_\nu$"),
    ("loo_chi2_eff", 1.3, r"LC LOO $\chi^2_{\rm eff}$"),
    ("m_2500_dereddened_rhat", 1.1, r"$m_{2500,\rm dered}$ R-hat"),
    (
        "m_2500_attenuated_model_rhat",
        1.1,
        r"$m_{2500,\rm atten}$ R-hat",
    ),
    ("log_tau_uv_rf_rhat", 1.1, r"$\log\,\tau_{\rm UV,RF}$ R-hat"),
    ("log_sigma_uv_rhat", 1.1, r"$\log\,\sigma_{\rm UV}$ R-hat"),
)


def load_catalog(path: Path) -> dict[str, np.ndarray | None]:
    with h5py.File(path, "r") as handle:
        data = {"z": np.asarray(handle["z"], dtype=float)}
        for column, _, _ in TIER1_CUTS:
            data[column] = (
                np.asarray(handle[column], dtype=float) if column in handle else None
            )
    return data


def column_limits(loaded, column: str, cut: float) -> tuple[float, float]:
    values = [
        data[column][np.isfinite(data[column])]
        for data in loaded
        if data[column] is not None and np.isfinite(data[column]).any()
    ]
    if not values:
        return cut / 3.0, cut * 3.0
    combined = np.concatenate(values)
    positive = combined[combined > 0.0]
    lower = min(float(np.min(positive)), cut) / 1.12
    upper = max(float(np.max(positive)), cut) * 1.12
    return lower, upper


def main() -> None:
    loaded = [load_catalog(DATA_DIR / filename) for filename, *_ in RUNS]
    limits = {
        column: column_limits(loaded, column, cut)
        for column, cut, _ in TIER1_CUTS
    }

    with plt.style.context("src/style.mplstyle"):
        fig, axes = plt.subplots(
            len(RUNS),
            len(TIER1_CUTS),
            figsize=(27.0, 10.5),
            sharex=True,
        )

        for row, ((_, run_label, color), data) in enumerate(zip(RUNS, loaded)):
            z = data["z"]
            for col, (column, cut, title) in enumerate(TIER1_CUTS):
                axis = axes[row, col]
                values = data[column]
                axis.set_yscale("log")
                axis.set_xlim(0.0, 5.0)
                axis.set_ylim(*limits[column])
                axis.axhline(cut, color="#A51C30", lw=1.4, ls="--", zorder=1)
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
                axis.grid(axis="y", which="major", color="0.87", lw=0.65, ls=":")

                if values is None:
                    axis.text(
                        0.5,
                        0.5,
                        "Column missing",
                        transform=axis.transAxes,
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="0.4",
                    )
                else:
                    finite = np.isfinite(z) & np.isfinite(values) & (values > 0.0)
                    passed = finite & (values <= cut)
                    failed = finite & (values > cut)
                    axis.scatter(
                        z[passed],
                        values[passed],
                        s=8,
                        color=color,
                        alpha=0.32,
                        edgecolors="none",
                        rasterized=True,
                    )
                    axis.scatter(
                        z[failed],
                        values[failed],
                        s=13,
                        color="#A51C30",
                        marker="x",
                        alpha=0.75,
                        linewidths=0.65,
                        rasterized=True,
                    )
                    n_finite = int(np.sum(finite))
                    n_passed = int(np.sum(passed))
                    percent = 100.0 * n_passed / n_finite if n_finite else np.nan
                    annotation = (
                        f"N = {n_finite:,}\npass = {percent:.1f}%"
                        if n_finite
                        else "No finite values"
                    )
                    axis.text(
                        0.03,
                        0.96,
                        annotation,
                        transform=axis.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8.2,
                        color="0.25",
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.72,
                            "pad": 1.0,
                        },
                    )

                axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
                axis.yaxis.set_major_formatter(ScalarFormatter())
                axis.yaxis.set_minor_formatter(NullFormatter())
                axis.tick_params(labelsize=8.5)
                if row == 0:
                    axis.set_title(title, fontsize=11, pad=7)
                if row == len(RUNS) - 1:
                    axis.set_xlabel("Redshift", fontsize=11)

            axes[row, 0].text(
                -0.27,
                0.5,
                run_label,
                transform=axes[row, 0].transAxes,
                rotation=90,
                ha="center",
                va="center",
                fontsize=12,
                color=color,
            )

        fig.supylabel("Tier 1 diagnostic (log scale)", x=0.015, fontsize=13)
        fig.subplots_adjust(
            left=0.055,
            right=0.995,
            bottom=0.075,
            top=0.93,
            hspace=0.26,
            wspace=0.34,
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = OUTPUT_DIR / "tier1_cuts_vs_redshift"
        fig.savefig(stem.with_suffix(".png"), dpi=220)
        fig.savefig(stem.with_suffix(".pdf"))
        plt.close(fig)

    print(f"Saved {stem.with_suffix('.png')}")
    print(f"Saved {stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
