"""Compare sigma/tau MCMC R-hat against redshift for selected LC runs."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np


DATA_DIR = Path("results/data")
OUTPUT_DIR = Path("results/plots/rhat_vs_redshift_aug29_aug27_aug16")
RUNS = (
    (
        "aug29_0158am_erlang_dho_svithennuts_psflogitnormal_iters1_"
        "svi10000lr0003w250s100_stonechisq_specaug280111pm_n8000_"
        "ce36392_chisq.h5",
        "Aug 29: Erlang DHO",
        "#0072B2",
        18,
        0.62,
    ),
    (
        "aug27_1219pm_erlang_legacy_svithennuts_psflogitnormal_iters3_"
        "svi4000w250s250_stonechisq_specaug240152pmv3_n8000_536f607_"
        "chisq.h5",
        "Aug 27: Erlang legacy",
        "#D55E00",
        10,
        0.28,
    ),
    (
        "aug16_0611am_mag_linear_svi4000w2000s500_f978995_stone.h5",
        "Aug 16: magnitude-linear",
        "#009E73",
        18,
        0.62,
    ),
)

PANELS = (
    ("log_sigma_uv_rhat", r"$\log\,\sigma_{\rm UV}$ R-hat"),
    ("log_tau_uv_rf_rhat", r"$\log\,\tau_{\rm UV,RF}$ R-hat"),
)


def load_columns(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {
            key: np.asarray(handle[key], dtype=float)
            for key in ("z", *(column for column, _ in PANELS))
        }


def finite_xy(data: dict[str, np.ndarray], column: str) -> tuple[np.ndarray, np.ndarray]:
    z = data["z"]
    rhat = data[column]
    finite = np.isfinite(z) & np.isfinite(rhat)
    return z[finite], rhat[finite]


def scatter_run(axis, data, column, color, marker_size, alpha):
    z, rhat = finite_xy(data, column)
    if z.size:
        axis.scatter(
            z,
            rhat,
            s=marker_size,
            color=color,
            alpha=alpha,
            edgecolors="none",
            rasterized=True,
        )
    return z, rhat


def main() -> None:
    loaded = [load_columns(DATA_DIR / filename) for filename, *_ in RUNS]
    global_rhat_max = max(
        float(np.nanmax(data[column]))
        for data in loaded
        for column, _ in PANELS
        if np.isfinite(data[column]).any()
    )
    shared_ylim = (0.985, 1.04 * global_rhat_max)
    shared_ticks = (1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 4.0)

    with plt.style.context("src/style.mplstyle"):
        fig, axes = plt.subplots(
            len(RUNS),
            len(PANELS),
            figsize=(12.5, 13.0),
            sharex=True,
        )

        for row, ((_, label, color, marker_size, alpha), data) in enumerate(
            zip(RUNS, loaded)
        ):
            for col, (column, title) in enumerate(PANELS):
                axis = axes[row, col]
                z, rhat = scatter_run(
                    axis, data, column, color, marker_size, min(1.0, alpha * 1.15)
                )
                axis.axhline(1.1, color="0.18", lw=1.35, ls="--", zorder=0)
                axis.text(
                    0.985,
                    1.1,
                    "cut = 1.1",
                    transform=axis.get_yaxis_transform(),
                    ha="right",
                    va="bottom",
                    fontsize=9,
                    color="0.25",
                )
                axis.set_xlim(0.0, 5.0)
                axis.set_yscale("log")
                axis.set_ylabel("R-hat (log scale)")
                if row == 0:
                    axis.set_title(title, pad=8)
                if row == len(RUNS) - 1:
                    axis.set_xlabel("Redshift")

                if rhat.size:
                    axis.set_ylim(*shared_ylim)
                else:
                    axis.set_ylim(*shared_ylim)
                    axis.text(
                        0.5,
                        0.5,
                        "No finite R-hat values",
                        transform=axis.transAxes,
                        ha="center",
                        va="center",
                        color="0.35",
                    )
                axis.yaxis.set_major_locator(FixedLocator(shared_ticks))
                axis.yaxis.set_major_formatter(
                    FuncFormatter(lambda value, _: f"{value:g}")
                )
                axis.yaxis.set_minor_formatter(NullFormatter())
                axis.grid(axis="y", which="major", color="0.85", lw=0.7, ls=":")

            axes[row, 0].text(
                -0.18,
                0.5,
                label,
                transform=axes[row, 0].transAxes,
                ha="center",
                va="center",
                rotation=90,
                fontsize=13,
                color=color,
            )

        fig.subplots_adjust(
            left=0.12, right=0.985, bottom=0.07, top=0.955, hspace=0.28, wspace=0.22
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = OUTPUT_DIR / "log_sigma_tau_rhat_vs_redshift"
        fig.savefig(stem.with_suffix(".png"), dpi=300)
        fig.savefig(stem.with_suffix(".pdf"))
        plt.close(fig)

    for (filename, label, *_), data in zip(RUNS, loaded):
        counts = [np.isfinite(data[column]).sum() for column, _ in PANELS]
        print(f"{label}: sigma N={counts[0]:,}, tau N={counts[1]:,} ({filename})")
    print(f"Saved {stem.with_suffix('.png')}")
    print(f"Saved {stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
