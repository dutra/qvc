#!/usr/bin/env python
"""Plot matched CARMA(2,1) and legacy Erlang grid-recovery results."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return {
        key: np.asarray([float(row[key]) for row in rows])
        for key in rows[0]
        if key
        not in {
            "kernel_model",
            "injection_kernel_model",
            "recovery_kernel_model",
            "lag_model",
        }
    }


def interval(ax, x, data, name, color):
    median = data[f"{name}_median"]
    ax.errorbar(
        x,
        median,
        yerr=np.vstack(
            (median - data[f"{name}_low"], data[f"{name}_high"] - median)
        ),
        fmt="none",
        ecolor=color,
        alpha=0.28,
        linewidth=0.8,
        zorder=1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carma", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = [("CARMA(2,1)", load(args.carma)), ("Legacy Erlang", load(args.legacy))]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2), constrained_layout=True)
    for column, (title, data) in enumerate(datasets):
        lag = data["true_lag_rest"]
        fraction = data["true_blr_fraction"]
        recovered_fraction = data["blr_fraction_median"]

        ax = axes[0, column]
        interval(ax, lag, data, "lag_rest", "tab:blue")
        points = ax.scatter(
            lag,
            data["lag_rest_median"],
            c=np.log10(fraction),
            cmap="viridis",
            vmin=-np.log10(20),
            vmax=0,
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        ax.plot([10, 1000], [10, 1000], "k--", linewidth=1)
        ax.set(xscale="log", yscale="log", xlim=(10, 1100), ylim=(10, 1100))
        ax.set_title(title)
        ax.set_xlabel("Injected rest-frame lag [days]")
        ax.set_ylabel("Recovered rest-frame lag [days]")
        ax.grid(alpha=0.2, which="both")

        ax = axes[1, column]
        interval(ax, fraction, data, "blr_fraction", "tab:orange")
        ax.scatter(
            fraction,
            recovered_fraction,
            c=np.log10(lag),
            cmap="plasma",
            vmin=np.log10(15),
            vmax=np.log10(800),
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        ax.plot([0.04, 1.1], [0.04, 1.1], "k--", linewidth=1)
        ax.set(xscale="log", yscale="log", xlim=(0.04, 1.1), ylim=(0.04, 1.1))
        ax.set_xlabel(r"Injected $A_{\rm BLR}/A_{\rm cont}$")
        ax.set_ylabel(r"Recovered $A_{\rm BLR}/A_{\rm cont}$")
        ax.grid(alpha=0.2, which="both")

    lag_bar = fig.colorbar(points, ax=axes[0, :], pad=0.015)
    lag_bar.set_label(r"$\log_{10}(A_{\rm BLR}/A_{\rm cont})$")
    fraction_map = axes[1, 0].collections[-1]
    fraction_bar = fig.colorbar(fraction_map, ax=axes[1, :], pad=0.015)
    fraction_bar.set_label(r"$\log_{10}(\mathrm{injected\ lag/days})$")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240)
    plt.close(fig)
    print(f"Saved comparison plot: {args.output}")


if __name__ == "__main__":
    main()
