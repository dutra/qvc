from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

plt.style.use(Path(__file__).with_name("style.mplstyle"))


COLORS = {
    "u": "tab:blue",
    "g": "tab:green",
    "r": "tab:orange",
    "i": "tab:red",
    "z": "tab:brown",
    "y": "tab:gray",
}


class _ErrorbarLegendHandle:
    def __init__(self, *, markerfacecolor, markeredgecolor, markersize, linewidth, alpha, error_color, error_lw, capsize):
        self.markerfacecolor = markerfacecolor
        self.markeredgecolor = markeredgecolor
        self.markersize = markersize
        self.linewidth = linewidth
        self.alpha = alpha
        self.error_color = error_color
        self.error_lw = error_lw
        self.capsize = capsize


class _HandlerErrorbarMarker(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        cx = xdescent + 0.5 * width
        cy = ydescent + 0.5 * height
        half_err = 5
        half_cap = min(orig_handle.capsize, 0.22 * min(width, height))

        hline = Line2D(
            [cx - half_err, cx + half_err],
            [cy, cy],
            color=orig_handle.error_color,
            lw=orig_handle.error_lw,
            alpha=orig_handle.alpha,
            transform=trans,
        )
        vline = Line2D(
            [cx, cx],
            [cy - half_err, cy + half_err],
            color=orig_handle.error_color,
            lw=orig_handle.error_lw,
            alpha=orig_handle.alpha,
            transform=trans,
        )
        cap_left = Line2D(
            [cx - half_err, cx - half_err],
            [cy - half_cap, cy + half_cap],
            color=orig_handle.error_color,
            lw=orig_handle.error_lw,
            alpha=orig_handle.alpha,
            transform=trans,
        )
        cap_right = Line2D(
            [cx + half_err, cx + half_err],
            [cy - half_cap, cy + half_cap],
            color=orig_handle.error_color,
            lw=orig_handle.error_lw,
            alpha=orig_handle.alpha,
            transform=trans,
        )
        cap_bottom = Line2D(
            [cx - half_cap, cx + half_cap],
            [cy - half_err, cy - half_err],
            color=orig_handle.error_color,
            lw=orig_handle.error_lw,
            alpha=orig_handle.alpha,
            transform=trans,
        )
        cap_top = Line2D(
            [cx - half_cap, cx + half_cap],
            [cy + half_err, cy + half_err],
            color=orig_handle.error_color,
            lw=orig_handle.error_lw,
            alpha=orig_handle.alpha,
            transform=trans,
        )
        marker = Line2D(
            [cx],
            [cy],
            marker="o",
            linestyle="none",
            markersize=orig_handle.markersize,
            markerfacecolor=orig_handle.markerfacecolor,
            markeredgecolor=orig_handle.markeredgecolor,
            markeredgewidth=orig_handle.linewidth,
            alpha=orig_handle.alpha,
            transform=trans,
        )
        return [hline, vline, cap_left, cap_right, cap_bottom, cap_top, marker]


def _symmetrize_error(err):
    if err is None:
        return None
    if isinstance(err, tuple):
        low = np.asarray(err[0], dtype=float)
        high = np.asarray(err[1], dtype=float)
        sym = 0.5 * (np.abs(low) + np.abs(high))
    else:
        sym = np.asarray(err, dtype=float)
    valid = np.isfinite(sym) & (sym >= 0.0)
    return np.where(valid, sym, np.nan)


def _compute_panel_metrics(x, y, xerr=None, yerr=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return None

    x = x[mask]
    y = y[mask]
    delta = y - x
    metrics = {
        "N": int(delta.size),
        "RMSE": float(np.sqrt(np.mean(delta**2))),
        "Bias": float(np.mean(delta)),
        "sigma": float(np.std(delta)),
    }

    xerr_sym = _symmetrize_error(xerr)
    yerr_sym = _symmetrize_error(yerr)
    if xerr_sym is not None and yerr_sym is not None:
        xerr_sym = xerr_sym[mask]
        yerr_sym = yerr_sym[mask]
        sigma_tot_sq = xerr_sym**2 + yerr_sym**2
        valid_chi = np.isfinite(sigma_tot_sq) & (sigma_tot_sq > 0.0)
        if np.any(valid_chi):
            metrics["chi2_nu"] = float(
                np.sum((delta[valid_chi] ** 2) / sigma_tot_sq[valid_chi]) / valid_chi.sum()
            )
    return metrics


def _format_panel_metrics(metrics, *, header=None):
    if metrics is None:
        return None
    lines = []
    if header:
        lines.append(header)
    lines.extend(
        [
            f"N = {metrics['N']}",
            f"RMSE = {metrics['RMSE']:.3f}",
            f"Bias = {metrics['Bias']:.3f}",
            f"$\\sigma$ = {metrics['sigma']:.3f}",
        ]
    )
    return "\n".join(lines)


def _format_identity_panel_metrics(metrics, *, unit, header=None):
    if metrics is None:
        return None
    lines = []
    if header:
        lines.append(header)
    lines.extend(
        [
            f"N = {metrics['N']}",
            f"Bias = {metrics['Bias']:.2f} {unit}",
            f"$\\sigma$ = {metrics['sigma']:.2f} {unit}",
        ]
    )
    return "\n".join(lines)


def _annotate_panel_metrics(ax, metrics, *, loc="lower right", header=None, fontsize=8.5):
    text = _format_panel_metrics(metrics, header=header)
    if not text:
        return
    anchors = {
        "lower right": (0.97, 0.03, "right", "bottom"),
        "lower left": (0.03, 0.03, "left", "bottom"),
        "upper right": (0.97, 0.97, "right", "top"),
        "upper left": (0.03, 0.97, "left", "top"),
    }
    x, y, ha, va = anchors[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.9),
        zorder=20,
    )


def _annotate_identity_panel_metrics(ax, metrics, *, unit, loc="lower right", header=None, fontsize=8.5):
    text = _format_identity_panel_metrics(metrics, unit=unit, header=header)
    if not text:
        return
    anchors = {
        "lower right": (0.97, 0.03, "right", "bottom"),
        "lower left": (0.03, 0.03, "left", "bottom"),
        "upper right": (0.97, 0.97, "right", "top"),
        "upper left": (0.03, 0.97, "left", "top"),
    }
    x, y, ha, va = anchors[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.9),
        zorder=20,
    )


def _resolve_pattern(df, pattern, band):
    if pattern is None:
        return None
    if isinstance(pattern, tuple):
        names = [item.format(band=band) for item in pattern]
        if not all(name in df.columns for name in names):
            return None
        return tuple(pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float) for name in names)
    name = pattern.format(band=band)
    if name not in df.columns:
        return None
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)


def _resolve_label(template, band):
    if template is None:
        return None
    return template.replace("<<band>>", band)


def _extract_xyerr(df, keydict, band):
    x = _resolve_pattern(df, keydict["x"], band)
    y = _resolve_pattern(df, keydict["y"], band)
    if x is None or y is None:
        return None, "missing data"

    xerr = _resolve_pattern(df, keydict.get("xerr"), band)
    yerr = _resolve_pattern(df, keydict.get("yerr"), band)

    mask = np.isfinite(x) & np.isfinite(y)

    xerr_mode = keydict.get("xerr_mode", "direct")
    if isinstance(xerr, tuple):
        low = np.asarray(xerr[0], dtype=float)
        high = np.asarray(xerr[1], dtype=float)
        if xerr_mode == "bounds":
            low = x - low
            high = high - x
        valid_low = np.isfinite(low) & (low >= 0.0)
        valid_high = np.isfinite(high) & (high >= 0.0)
        mask &= valid_low & valid_high
        xerr = (np.clip(low, 0.0, 0.5), np.clip(high, 0.0, 0.5))
    elif xerr is not None:
        xerr = np.asarray(xerr, dtype=float)
        valid_xerr = np.isfinite(xerr) & (xerr >= 0.0)
        mask &= valid_xerr
        xerr = np.where(valid_xerr, xerr, np.nan)

    if isinstance(yerr, tuple):
        low = np.asarray(yerr[0], dtype=float)
        high = np.asarray(yerr[1], dtype=float)
        valid_low = np.isfinite(low) & (low >= 0.0)
        valid_high = np.isfinite(high) & (high >= 0.0)
        mask &= valid_low & valid_high
        yerr = (low, high)
    elif yerr is not None:
        yerr = np.asarray(yerr, dtype=float)
        valid_yerr = np.isfinite(yerr) & (yerr >= 0.0)
        mask &= valid_yerr
        yerr = np.where(valid_yerr, yerr, np.nan)

    if mask.sum() == 0:
        return None, "no finite matched data"

    x = x[mask]
    y = y[mask]
    if isinstance(xerr, tuple):
        xerr = (xerr[0][mask], xerr[1][mask])
    elif xerr is not None:
        xerr = xerr[mask]
    if isinstance(yerr, tuple):
        yerr = (yerr[0][mask], yerr[1][mask])
    elif yerr is not None:
        yerr = yerr[mask]
    return (x, y, xerr, yerr), None


def plot_sigma_tau_identity_grid(
    data,
    sigma_keys,
    tau_keys,
    *,
    bands,
    figsize=(16.0, 8.0),
    show=True,
    output_path=None,
    style=None,
    sigma_limits=None,
    tau_limits=None,
):
    label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 13
    metric_fontsize = 10.5
    style = {
        "point_alpha": 0.9,
        "point_size": 18,
        "point_linewidth": 0.5,
        "point_color": "k",
        "point_edgecolor": "k",
        "error_alpha": 0.28,
        "error_color": "0.35",
        "error_lw": 1.4,
        "error_capsize": 3,
        "error_capthick": 1,
        "rasterized": False,
        **(style or {}),
    }

    ncols = len(bands)
    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=figsize,
        sharex="row",
        sharey="row",
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.0, h_pad=0.01, wspace=0.0, hspace=0.05)
    axes = np.asarray(axes, dtype=object)
    if axes.ndim == 1:
        if ncols == 1:
            axes = axes[:, np.newaxis]
        else:
            axes = axes[np.newaxis, :]

    def _style_axis(ax):
        ax.minorticks_on()
        ax.tick_params(
            direction="in",
            which="major",
            top=True,
            right=True,
            length=4,
            width=1.1,
            pad=2,
            labelsize=tick_fontsize,
        )
        ax.tick_params(direction="in", which="minor", top=True, right=True, length=2.5, width=0.9)
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)
        ax.set_aspect("equal", adjustable="box")

    def _set_row_ticks(ax_row):
        major = MultipleLocator(0.5)
        minor = AutoMinorLocator(5)
        for ax in ax_row:
            ax.xaxis.set_major_locator(major)
            ax.yaxis.set_major_locator(major)
            ax.xaxis.set_minor_locator(minor)
            ax.yaxis.set_minor_locator(minor)

    def _row_limits(keydict, quantity):
        mins = []
        maxs = []
        for band in bands:
            df_band = data.get(band) if isinstance(data, dict) else data
            if df_band is None:
                continue
            panel_data, _ = _extract_xyerr(df_band, keydict, band)
            if panel_data is None:
                continue
            x, y, _, _ = panel_data
            mins.append(min(np.nanmin(x), np.nanmin(y)))
            maxs.append(max(np.nanmax(x), np.nanmax(y)))
        if not mins:
            raise ValueError(f"No finite matched {quantity} values available for any band.")
        lo = min(mins)
        hi = max(maxs)
        pad_floor = 0.2 if quantity == "sigma" else 0.5
        pad = max(0.04 * (hi - lo), pad_floor)
        return (lo - pad, hi + pad)

    sigma_limits = sigma_limits or _row_limits(sigma_keys, "sigma")
    tau_limits = tau_limits or _row_limits(tau_keys, "tau")

    # Identity residuals are computed in log space (delta = y - x), so both rows are in dex.
    metric_units = ("dex", "dex")

    for row_index, (keydict, row_limits) in enumerate(((sigma_keys, sigma_limits), (tau_keys, tau_limits))):
        for col, band in enumerate(bands):
            ax = axes[row_index, col]
            df_band = data.get(band) if isinstance(data, dict) else data
            if df_band is None:
                _style_axis(ax)
                ax.text(0.5, 0.5, f"{band}-band\nmissing data", transform=ax.transAxes, ha="center", va="center")
                continue

            panel_data, message = _extract_xyerr(df_band, keydict, band)
            ax.plot(row_limits, row_limits, ls="--", lw=2.0, color=COLORS.get(band, "0.2"), zorder=-4)
            ax.set_xlim(*row_limits)
            ax.set_ylim(*row_limits)
            _style_axis(ax)

            if panel_data is None:
                ax.text(0.5, 0.5, f"{band}-band\n{message}", transform=ax.transAxes, ha="center", va="center")
                continue

            x, y, xerr, yerr = panel_data
            metrics = _compute_panel_metrics(x, y, xerr=xerr, yerr=yerr)
            ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                fmt="none",
                color=style["error_color"],
                alpha=style["error_alpha"],
                lw=style["error_lw"],
                capsize=style["error_capsize"],
                capthick=style["error_capthick"],
                zorder=-9,
                rasterized=style["rasterized"],
            )
            ax.scatter(
                x,
                y,
                color=style["point_color"],
                s=style["point_size"],
                alpha=style["point_alpha"],
                edgecolor=style["point_edgecolor"],
                linewidths=style["point_linewidth"],
                zorder=-8,
                rasterized=style["rasterized"],
            )
            ax.set_xlabel(_resolve_label(keydict.get("xlabel"), band), labelpad=2, fontsize=label_fontsize)
            ax.set_ylabel(_resolve_label(keydict.get("ylabel"), band), labelpad=2, fontsize=label_fontsize)
            legend_handle = _ErrorbarLegendHandle(
                markerfacecolor=style["point_color"],
                markeredgecolor=style["point_edgecolor"],
                markersize=np.sqrt(style["point_size"]),
                linewidth=style["point_linewidth"],
                alpha=style["point_alpha"],
                error_color=style["error_color"],
                error_lw=style["error_lw"],
                capsize=style["error_capsize"],
            )
            ax.legend(
                handles=[legend_handle],
                labels=["AGN"],
                handler_map={_ErrorbarLegendHandle: _HandlerErrorbarMarker()},
                loc="upper left",
                frameon=False,
                fancybox=True,
                framealpha=1,
                edgecolor="0.2",
                facecolor="white",
                fontsize=legend_fontsize,
            )
            _annotate_identity_panel_metrics(
                ax,
                metrics,
                unit=metric_units[row_index],
                loc="lower right",
                fontsize=metric_fontsize,
            )

    _set_row_ticks(axes[0])
    _set_row_ticks(axes[1])

    for i in range(2):
        for j in range(1, ncols):
            axes[i, j].tick_params(labelleft=False)

    if output_path is not None:
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_sigma_tau_style_comparison(ours_df, comparison_df, *, title=None, comparison_overlays=None):
    ours_sigma = pd.to_numeric(ours_df["ours_sigma_r"], errors="coerce").to_numpy(dtype=float)
    ours_tau = pd.to_numeric(ours_df["ours_tau_r"], errors="coerce").to_numpy(dtype=float)

    ours_mask = np.isfinite(ours_sigma) & np.isfinite(ours_tau)
    ours_idx = np.flatnonzero(ours_mask)

    if comparison_overlays is None:
        comparison_overlays = [
            {
                "dataframe": comparison_df,
                "sigma_col": "macleod_sigma_r",
                "tau_col": "macleod_tau_r",
                "label": "MacLeod+2010",
                "color": "tab:orange",
            },
        ]

    processed_overlays = []
    for overlay in comparison_overlays:
        overlay_sigma = pd.to_numeric(overlay["dataframe"][overlay["sigma_col"]], errors="coerce").to_numpy(dtype=float)
        overlay_tau = pd.to_numeric(overlay["dataframe"][overlay["tau_col"]], errors="coerce").to_numpy(dtype=float)
        overlay_mask = (
            np.isfinite(overlay_sigma)
            & np.isfinite(overlay_tau)
            & (overlay_sigma > -9)
            & (overlay_tau > -9)
            & (overlay_tau < 5)
        )
        overlay_idx = np.flatnonzero(overlay_mask)
        processed_overlays.append(
            {
                "sigma": overlay_sigma,
                "tau": overlay_tau,
                "idx": overlay_idx,
                "label": overlay["label"],
                "color": overlay["color"],
            }
        )

    overlay_hist_x_positions = [0.77, 0.64, 0.51]
    tau_hist_x_positions = [0.98, 0.82, 0.66]

    summary_metric_blocks = []

    fig = plt.figure(figsize=(7.8, 7.0))
    gs = fig.add_gridspec(2, 2, width_ratios=(4.0, 1.2), height_ratios=(1.2, 4.0), hspace=0.05, wspace=0.05)
    ax_hist_x = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_hist_x)
    ax_hist_y = fig.add_subplot(gs[1, 1], sharey=ax)

    if ours_idx.size > 0:
        ax.scatter(
            ours_sigma[ours_idx],
            ours_tau[ours_idx],
            s=2.2**2,
            color="0.3",
            alpha=0.7,
            linewidths=0,
            rasterized=True,
            label="this work",
        )
        sigma_med = np.nanmedian(ours_sigma[ours_idx])
        tau_med = np.nanmedian(ours_tau[ours_idx])
        sigma_med_linear = 10.0**sigma_med
        tau_med_linear = 10.0**tau_med
        ax_hist_x.hist(ours_sigma[ours_idx], bins=30, color="0.35", histtype="stepfilled", alpha=0.25)
        ax_hist_x.axvline(sigma_med, color="k", ls="--", lw=1.5)
        ax_hist_x.text(
            0.98,
            0.90,
            f"this work: {sigma_med_linear:.2f} mag",
            ha="right",
            va="top",
            color="k",
            transform=ax_hist_x.transAxes,
            fontsize=9,
        )
        ax_hist_y.hist(ours_tau[ours_idx], bins=30, orientation="horizontal", color="0.35", histtype="stepfilled", alpha=0.25)
        ax_hist_y.axhline(tau_med, color="k", ls="--", lw=1.5)
        ax_hist_y.text(
            tau_hist_x_positions[0] - 0.05,
            0.98,
            f"this work: {int(np.round(tau_med_linear))} days",
            ha="right",
            va="top",
            rotation=-90,
            color="k",
            transform=ax_hist_y.transAxes,
            fontsize=9,
        )
        ours_metrics = _compute_panel_metrics(ours_sigma[ours_idx], ours_tau[ours_idx])
        summary_metric_blocks.append(_format_panel_metrics(ours_metrics, header="this work"))

    for overlay_index, overlay in enumerate(processed_overlays):
        overlay_idx = overlay["idx"]
        if overlay_idx.size == 0:
            continue
        overlay_sigma = overlay["sigma"]
        overlay_tau = overlay["tau"]
        overlay_color = overlay["color"]
        overlay_label = overlay["label"]
        ax.scatter(
            overlay_sigma[overlay_idx],
            overlay_tau[overlay_idx],
            s=8,
            alpha=0.35,
            linewidths=0,
            color=overlay_color,
            rasterized=True,
            label=overlay_label,
        )
        ax_hist_x.hist(overlay_sigma[overlay_idx], bins=30, color=overlay_color, histtype="step", lw=1.5, alpha=0.9)
        ax_hist_y.hist(
            overlay_tau[overlay_idx],
            bins=30,
            orientation="horizontal",
            color=overlay_color,
            histtype="step",
            lw=1.5,
            alpha=0.9,
        )
        overlay_sigma_med = np.nanmedian(overlay_sigma[overlay_idx])
        overlay_tau_med = np.nanmedian(overlay_tau[overlay_idx])
        ax_hist_x.axvline(overlay_sigma_med, color=overlay_color, ls="--", lw=1.5)
        ax_hist_y.axhline(overlay_tau_med, color=overlay_color, ls="--", lw=1.5)
        overlay_sigma_med_linear = 10.0**overlay_sigma_med
        overlay_tau_med_linear = 10.0**overlay_tau_med
        hist_x_y = overlay_hist_x_positions[min(overlay_index, len(overlay_hist_x_positions) - 1)]
        tau_hist_x = tau_hist_x_positions[min(overlay_index + 1, len(tau_hist_x_positions) - 1)]
        ax_hist_x.text(
            0.98,
            hist_x_y,
            f"{overlay_label}: {overlay_sigma_med_linear:.2f} mag",
            ha="right",
            va="top",
            color=overlay_color,
            transform=ax_hist_x.transAxes,
            fontsize=9,
        )
        ax_hist_y.text(
            tau_hist_x - 0.05,
            0.98,
            f"{overlay_label}: {int(np.round(overlay_tau_med_linear))} days",
            ha="right",
            va="top",
            rotation=-90,
            color=overlay_color,
            transform=ax_hist_y.transAxes,
            fontsize=9,
        )
        overlay_metrics = _compute_panel_metrics(overlay_sigma[overlay_idx], overlay_tau[overlay_idx])
        summary_metric_blocks.append(_format_panel_metrics(overlay_metrics, header=overlay_label))

    combined_sigma_series = [ours_sigma[ours_idx]]
    combined_tau_series = [ours_tau[ours_idx]]
    for overlay in processed_overlays:
        if overlay["idx"].size > 0:
            combined_sigma_series.append(overlay["sigma"][overlay["idx"]])
            combined_tau_series.append(overlay["tau"][overlay["idx"]])
    combined_sigma = np.concatenate(combined_sigma_series) if combined_sigma_series else np.array([])
    combined_tau = np.concatenate(combined_tau_series) if combined_tau_series else np.array([])
    if combined_sigma.size and combined_tau.size:
        xpad = 0.05 * max(np.nanmax(combined_sigma) - np.nanmin(combined_sigma), 1e-6)
        ypad = 0.05 * max(np.nanmax(combined_tau) - np.nanmin(combined_tau), 1e-6)
        ax.set_xlim(np.nanmin(combined_sigma) - xpad, np.nanmax(combined_sigma) + xpad)
        ax.set_ylim(1.1, 4.9)

    ax.set_xlabel(r"$\log \sigma_{\rm UV}$")
    ax.set_ylabel(r"$\log \tau_{\rm UV,RF}$")
    if title is not None:
        ax.set_title(title)

    legend_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=4, color="0.3", alpha=0.7, label="this work"),
    ]
    for overlay in processed_overlays:
        if overlay["idx"].size > 0:
            legend_handles.append(
                Line2D([], [], marker="o", linestyle="none", markersize=5, color=overlay["color"], alpha=0.65, label=overlay["label"])
            )
    ax.legend(handles=legend_handles, loc="lower right")

    ax_hist_x.tick_params(labelbottom=False)
    ax_hist_y.tick_params(labelleft=False)
    ax_hist_x.set_ylabel("N")
    ax_hist_y.set_xlabel("N")
    fig.tight_layout()
    return fig


__all__ = [
    "COLORS",
    "plot_sigma_tau_identity_grid",
    "plot_sigma_tau_style_comparison",
]
