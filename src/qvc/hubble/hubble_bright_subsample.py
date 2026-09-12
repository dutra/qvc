"""Redshift-dependent bright-subsample selection for the AGN Hubble fit.

The completeness map defines a faint magnitude boundary, shifted brighter by
``margin``. The same boundary filters the sample and multiplies the original
completeness function inside the likelihood normalization and plotting moments.
Completeness remains magnitude-dependent inside the retained region; this cut
does not establish that the retained sample is complete or free of selection bias.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from qvc.hubble.hubble_completeness_refactored import (
    COMPLETENESS_MAG_COL,
    prepare_completeness_magnitude_columns,
)

BRIGHT_SUBSAMPLE_JSON_ATTR = "bright_subsample_json"


@dataclass(frozen=True)
class BrightSubsampleCut:
    """Redshift-dependent hard bright cut in the selection magnitude.

    ``completeness_min`` is a fraction of the per-redshift peak completeness
    when ``relative`` is true (the default; the absolute level of the
    estimated map only reflects the catalog size relative to the LF mock and
    cancels in the Malmquist term), or an absolute detection probability
    otherwise.  ``peak_grid`` records the per-redshift peak completeness.
    """

    completeness_min: float
    margin: float
    z_grid: tuple
    threshold_grid: tuple
    relative: bool = True
    peak_grid: tuple = ()

    def __post_init__(self):
        completeness_min = float(self.completeness_min)
        margin = float(self.margin)
        if not (0.0 < completeness_min <= 1.0):
            raise ValueError("completeness_min must lie in (0, 1].")
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("margin must be a finite nonnegative magnitude offset.")
        z_grid = tuple(float(value) for value in np.asarray(self.z_grid, dtype=float).ravel())
        thresholds = tuple(
            float(value) for value in np.asarray(self.threshold_grid, dtype=float).ravel()
        )
        if len(z_grid) == 0 or len(z_grid) != len(thresholds):
            raise ValueError("z_grid and threshold_grid must be nonempty and equal in length.")
        if np.any(np.diff(z_grid) <= 0.0) or not np.all(np.isfinite(z_grid)):
            raise ValueError("z_grid must be finite and strictly increasing.")
        if not np.all(np.isfinite(thresholds)):
            raise ValueError("threshold_grid must be finite; use a very bright value where no magnitude is complete.")
        peaks = tuple(float(value) for value in np.asarray(self.peak_grid, dtype=float).ravel())
        if peaks and len(peaks) != len(z_grid):
            raise ValueError("peak_grid must be empty or match z_grid in length.")
        object.__setattr__(self, "completeness_min", completeness_min)
        object.__setattr__(self, "margin", margin)
        object.__setattr__(self, "z_grid", z_grid)
        object.__setattr__(self, "threshold_grid", thresholds)
        object.__setattr__(self, "relative", bool(self.relative))
        object.__setattr__(self, "peak_grid", peaks)

    def threshold(self, z):
        """Faintest allowed selection magnitude at redshift ``z``."""
        z = np.asarray(z, dtype=float)
        return np.interp(z, np.asarray(self.z_grid), np.asarray(self.threshold_grid))

    def mask(self, magnitude, z):
        magnitude, z = np.broadcast_arrays(
            np.asarray(magnitude, dtype=float), np.asarray(z, dtype=float)
        )
        return np.isfinite(magnitude) & np.isfinite(z) & (magnitude <= self.threshold(z))

    def run_tag(self):
        kind = "rel" if self.relative else "abs"
        return (
            f"_brightsub-{kind}{self.completeness_min:g}-dm{self.margin:g}".replace(".", "p")
        )

    def to_json(self):
        return json.dumps(
            {
                "completeness_min": self.completeness_min,
                "margin": self.margin,
                "relative": self.relative,
                "z_grid": list(self.z_grid),
                "threshold_grid": list(self.threshold_grid),
                "peak_grid": list(self.peak_grid),
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text):
        payload = json.loads(text)
        return cls(
            completeness_min=payload["completeness_min"],
            margin=payload["margin"],
            z_grid=tuple(payload["z_grid"]),
            threshold_grid=tuple(payload["threshold_grid"]),
            relative=payload.get("relative", True),
            peak_grid=tuple(payload.get("peak_grid", ())),
        )

    def summary_frame(self):
        frame = pd.DataFrame(
            {
                "z": np.asarray(self.z_grid),
                "bright_cut_magnitude": np.asarray(self.threshold_grid),
                "complete_to_magnitude": np.asarray(self.threshold_grid) + self.margin,
            }
        )
        if self.peak_grid:
            frame["peak_completeness"] = np.asarray(self.peak_grid)
        return frame


def derive_bright_subsample_cut(
    completeness_model,
    mag_centers,
    z_centers,
    *,
    completeness_min,
    margin,
    relative=True,
    unavailable_threshold=-99.0,
):
    """Tabulate the faintest complete magnitude per redshift, minus ``margin``.

    At each redshift the faintest magnitude bin whose completeness reaches
    ``completeness_min`` (times the per-redshift peak when ``relative``) is the
    completeness limit; the cut is that limit minus ``margin``.  Redshifts
    without any such bin receive ``unavailable_threshold`` so that no object
    survives there.
    """
    mag_centers = np.asarray(mag_centers, dtype=float)
    z_centers = np.asarray(z_centers, dtype=float)
    if mag_centers.ndim != 1 or z_centers.ndim != 1 or mag_centers.size < 2:
        raise ValueError("mag_centers and z_centers must be one-dimensional grids.")
    probability = np.asarray(
        completeness_model(mag_centers[None, :], z_centers[:, None]), dtype=float
    )
    probability = np.broadcast_to(probability, (z_centers.size, mag_centers.size))
    thresholds = np.full(z_centers.size, float(unavailable_threshold))
    peaks = np.nanmax(np.where(np.isfinite(probability), probability, 0.0), axis=1)
    for index in range(z_centers.size):
        level = float(completeness_min) * (peaks[index] if relative else 1.0)
        if not np.isfinite(level) or peaks[index] <= 0.0:
            continue
        complete = np.flatnonzero(probability[index] >= level)
        if complete.size == 0:
            continue
        thresholds[index] = mag_centers[complete[-1]] - float(margin)
    return BrightSubsampleCut(
        completeness_min=completeness_min,
        margin=margin,
        z_grid=tuple(z_centers),
        threshold_grid=tuple(thresholds),
        relative=relative,
        peak_grid=tuple(peaks),
    )


class BrightCutCompletenessModel:
    """Completeness model multiplied by the hard bright cut.

    Attribute access falls through to the wrapped model so that the
    likelihood, tail bookkeeping and checkpoint metadata see the original
    grids and support; caches the likelihood attaches are stored on the
    wrapper itself.
    """

    def __init__(self, inner, cut):
        if not isinstance(cut, BrightSubsampleCut):
            raise TypeError("cut must be a BrightSubsampleCut.")
        self.__dict__["_inner"] = inner
        self.__dict__["bright_subsample_cut"] = cut

    def __call__(self, mag, z, *extra):
        values = np.asarray(self._inner(mag, z, *extra), dtype=float)
        magnitude, redshift = np.broadcast_arrays(
            np.asarray(mag, dtype=float), np.asarray(z, dtype=float)
        )
        keep = self.bright_subsample_cut.mask(magnitude, redshift)
        # Some models (for example redshift-independent analytic ones) return
        # a shape that does not carry the redshift axis; broadcast both ways.
        values, keep = np.broadcast_arrays(values, keep)
        return np.where(keep, values, 0.0)

    def __getattr__(self, name):
        inner = self.__dict__.get("_inner")
        if inner is None or name.startswith("__"):
            raise AttributeError(name)
        return getattr(inner, name)

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    @property
    def grid(self):
        grid = dict(getattr(self._inner, "grid", {}))
        grid["bright_subsample_cut"] = self.bright_subsample_cut.to_json()
        return grid


def wrap_completeness_params(completeness_params, cut):
    """Return the completeness tuple with its model wrapped by the bright cut."""
    if completeness_params is None or cut is None:
        return completeness_params
    model = completeness_params[0]
    if isinstance(model, BrightCutCompletenessModel):
        if model.bright_subsample_cut != cut:
            raise ValueError("Completeness model already carries a different bright cut.")
        return completeness_params
    return (BrightCutCompletenessModel(model, cut),) + tuple(completeness_params[1:])


def apply_bright_subsample_cut(df, cut, *, completeness_magnitude):
    """Return ``(kept_frame, keep_mask)`` for the bright subsample of ``df``."""
    prepared = prepare_completeness_magnitude_columns(df, completeness_magnitude)
    magnitude = prepared[COMPLETENESS_MAG_COL].to_numpy(dtype=float)
    z = prepared["z"].to_numpy(dtype=float)
    keep = cut.mask(magnitude, z)
    kept = df.loc[keep].copy()
    kept.attrs.update(df.attrs)
    kept.attrs[BRIGHT_SUBSAMPLE_JSON_ATTR] = cut.to_json()
    return kept, keep


def plot_bright_subsample_cut(
    completeness_model,
    mag_centers,
    z_centers,
    cut,
    df,
    keep,
    *,
    z_range,
    completeness_magnitude,
    plot_path,
    filename="bright_subsample_cut.pdf",
    slice_redshifts=(0.6, 1.0, 1.5, 2.0, 2.6),
):
    """Visualize the bright-subsample cut on top of the completeness map.

    Left: the map relative to its per-redshift peak, the completeness limit
    (where the map drops below ``completeness_min`` times the peak), the cut
    (limit minus margin), and the objects kept (filled) and removed (open).
    Middle: the relative completeness versus magnitude at a few redshifts
    with the threshold level, limit and cut marked.  Right: kept and removed
    counts per redshift bin of the fit range.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    mag_centers = np.asarray(mag_centers, dtype=float)
    z_centers = np.asarray(z_centers, dtype=float)
    probability = np.asarray(
        completeness_model(mag_centers[None, :], z_centers[:, None]), dtype=float
    )
    probability = np.broadcast_to(probability, (z_centers.size, mag_centers.size))
    peak = np.nanmax(np.where(np.isfinite(probability), probability, 0.0), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(peak[:, None] > 0, probability / peak[:, None], np.nan)

    prepared = prepare_completeness_magnitude_columns(df, completeness_magnitude)
    magnitude = prepared[COMPLETENESS_MAG_COL].to_numpy(dtype=float)
    z = prepared["z"].to_numpy(dtype=float)
    keep = np.asarray(keep, dtype=bool)
    thresholds = np.asarray(cut.threshold_grid)
    limits = thresholds + cut.margin
    valid = thresholds > -50.0

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), gridspec_kw={"width_ratios": (1.35, 1.2, 0.9)})

    # --- left: map with cut and objects ---
    ax = axes[0]
    dm = float(np.median(np.diff(mag_centers)))
    dz = float(np.median(np.diff(z_centers)))
    image = ax.imshow(
        relative,
        origin="lower",
        aspect="auto",
        extent=[mag_centers[0] - dm / 2, mag_centers[-1] + dm / 2, z_centers[0] - dz / 2, z_centers[-1] + dz / 2],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(image, ax=ax, label="completeness / per-redshift peak")
    ax.plot(limits[valid], z_centers[valid], color="w", lw=1.6, ls="--", label=f"limit: map falls below {cut.completeness_min:g} x peak")
    ax.plot(thresholds[valid], z_centers[valid], color="r", lw=2.0, label=f"cut: limit - {cut.margin:g} mag (fit uses m <= cut)")
    ax.scatter(magnitude[keep], z[keep], s=7, color="w", edgecolor="k", linewidths=0.3, zorder=3, label=f"kept ({int(keep.sum())})")
    ax.scatter(magnitude[~keep], z[~keep], s=9, facecolor="none", edgecolor="orange", linewidths=0.6, zorder=3, label=f"removed ({int((~keep).sum())})")
    ax.axhspan(z_centers[0] - dz, z_range[0], color="k", alpha=0.15, lw=0)
    ax.axhspan(z_range[1], z_centers[-1] + dz, color="k", alpha=0.15, lw=0)
    ax.set_xlim(mag_centers[0], mag_centers[-1])
    ax.set_ylim(max(z_centers[0], 0.0), min(z_centers[-1], float(np.nanmax(z)) + 0.4))
    ax.set_xlabel(r"selection magnitude $m_{2500\,\mathrm{\AA}}$ (mag)")
    ax.set_ylabel("z")
    ax.set_title("bright-subsample cut on the completeness map", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.85)

    # --- middle: slices ---
    ax = axes[1]
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(slice_redshifts)))
    for color, zs in zip(colors, slice_redshifts):
        index = int(np.argmin(np.abs(z_centers - zs)))
        ax.plot(mag_centers, relative[index], color=color, lw=1.4, label=f"z = {z_centers[index]:.2f}")
        ax.axvline(cut.threshold(z_centers[index]), color=color, lw=1.0, ls="-")
        ax.axvline(cut.threshold(z_centers[index]) + cut.margin, color=color, lw=1.0, ls="--")
    ax.axhline(cut.completeness_min, color="k", lw=1.0, ls=":", label=f"threshold {cut.completeness_min:g} x peak")
    ax.set_xlim(mag_centers[0], mag_centers[-1])
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"selection magnitude $m_{2500\,\mathrm{\AA}}$ (mag)")
    ax.set_ylabel("completeness / per-redshift peak")
    ax.set_title("map slices; dashed = limit, solid = cut", fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color="gray", ls="--"), Line2D([], [], color="gray", ls="-")]
    labels += ["completeness limit", "cut (limit - margin)"]
    ax.legend(handles, labels, fontsize=7, loc="lower right")

    # --- right: counts per redshift bin ---
    ax = axes[2]
    summary = summarize_bright_subsample_cut(df, keep, cut, z_range=z_range)
    centers = np.array([interval.mid for interval in summary["z_bin"]])
    width = 0.8 * float(np.median(np.diff(centers))) if len(centers) > 1 else 0.3
    ax.bar(centers, summary["n_before"], width=width, color="lightgray", label="before cut")
    ax.bar(centers, summary["n_kept"], width=width, color="C0", label="kept")
    for x, row in zip(centers, summary.itertuples(index=False)):
        ax.text(x, row.n_before, f"{row.fraction_kept:.0%}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("z (fit range bins)")
    ax.set_ylabel("objects")
    ax.set_title("kept fraction per redshift bin", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"Bright subsample: completeness >= {cut.completeness_min:g}"
        + (" x peak" if cut.relative else " (absolute)")
        + f", margin {cut.margin:g} mag; kept {int(keep.sum())} of {keep.size}",
        fontsize=11,
    )
    fig.tight_layout()
    import os

    os.makedirs(plot_path, exist_ok=True)
    output = os.path.join(plot_path, filename)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def summarize_bright_subsample_cut(df, keep, cut, *, z_range, n_bins=6):
    """Per-redshift-bin counts before and after the cut, for the fit range."""
    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
    edges = np.linspace(float(z_range[0]), float(z_range[1]), int(n_bins) + 1)
    in_range = (z >= edges[0]) & (z <= edges[-1])
    labels = pd.cut(z, edges, include_lowest=True)
    frame = pd.DataFrame({"z_bin": labels, "in_range": in_range, "kept": np.asarray(keep, dtype=bool)})
    frame = frame[frame["in_range"]]
    grouped = frame.groupby("z_bin", observed=True)
    summary = pd.DataFrame(
        {
            "n_before": grouped.size(),
            "n_kept": grouped["kept"].sum().astype(int),
        }
    )
    summary["fraction_kept"] = summary["n_kept"] / summary["n_before"].clip(lower=1)
    centers = np.array([interval.mid for interval in summary.index])
    summary["bright_cut_magnitude_at_center"] = cut.threshold(centers)
    summary["peak_completeness_at_center"] = (
        np.interp(centers, np.asarray(cut.z_grid), np.asarray(cut.peak_grid))
        if cut.peak_grid
        else np.nan
    )
    return summary.reset_index().rename(columns={"z_bin": "z_bin"})
