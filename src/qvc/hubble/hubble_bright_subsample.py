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
from pathlib import Path

import numpy as np
import pandas as pd

from qvc.hubble.hubble_completeness_refactored import (
    COMPLETENESS_GROUP_COL,
    COMPLETENESS_MAG_COL,
    prepare_completeness_magnitude_columns,
)

BRIGHT_SUBSAMPLE_JSON_ATTR = "bright_subsample_json"


@dataclass(frozen=True)
class BrightSubsamplePlotData:
    """Original selection inputs shared across cosmology-specific exports.

    The pre-bright sample is distinct from the completeness parent: earlier
    catalog cuts may already have removed objects from the fit sample.
    """

    completeness_params: tuple
    before: pd.DataFrame
    keep: np.ndarray

    def write(self, cut, *, plot_path, z_range, completeness_magnitude, plot):
        destination = Path(plot_path) / "completeness"
        destination.mkdir(parents=True, exist_ok=True)
        if isinstance(cut, GroupedBrightSubsampleCut):
            # Per-group thresholds, counts and cut-on-map figures.
            cut.summary_frame().to_csv(destination / "bright_subsample_thresholds.csv", index=False)
            summarize_bright_subsample_cut(self.before, self.keep, cut, z_range=z_range).to_csv(
                destination / "bright_subsample_counts.csv", index=False
            )
            index = group_index_from_frame(self.before)
            group_params = self.completeness_params[0].group_params
            for position, name in enumerate(cut.groups):
                rows = index == position
                if plot:
                    plot_bright_subsample_cut(
                        *group_params[name][:3], cut.cut_for(name), self.before.loc[rows],
                        np.asarray(self.keep)[rows], z_range=z_range,
                        completeness_magnitude=completeness_magnitude, plot_path=plot_path,
                        filename=f"bright_subsample_cut_group_{name}.pdf",
                    )
            return
        cut.summary_frame().to_csv(destination / "bright_subsample_thresholds.csv", index=False)
        summarize_bright_subsample_cut(self.before, self.keep, cut, z_range=z_range).to_csv(
            destination / "bright_subsample_counts.csv", index=False
        )
        if plot:
            plot_bright_subsample_cut(
                *self.completeness_params[:3], cut, self.before, self.keep,
                z_range=z_range, completeness_magnitude=completeness_magnitude,
                plot_path=plot_path,
            )


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

    def selection_cut_magnitude(self, z, group_index=None):
        """Per-object hard cut; the likelihood ends its integral exactly here."""
        del group_index
        return self.bright_subsample_cut.threshold(z)

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
    if isinstance(cut, GroupedBrightSubsampleCut) or isinstance(model, GroupedCompletenessModel):
        if not isinstance(model, GroupedCompletenessModel) or model.bright_subsample_cut != cut:
            raise ValueError(
                "Per-group bright cuts require the grouped completeness model built "
                "with the same cuts (see build_grouped_completeness_params)."
            )
        return completeness_params
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
    if isinstance(cut, GroupedBrightSubsampleCut):
        keep = cut.mask(magnitude, z, group_index_from_frame(prepared))
    else:
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
    completeness_path = os.path.join(plot_path, "completeness")
    os.makedirs(completeness_path, exist_ok=True)
    output = os.path.join(completeness_path, filename)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def summarize_bright_subsample_cut(df, keep, cut, *, z_range, n_bins=6):
    """Per-redshift-bin counts before and after the cut, for the fit range."""
    if isinstance(cut, GroupedBrightSubsampleCut):
        index = group_index_from_frame(df)
        keep = np.asarray(keep, dtype=bool)
        frames = []
        for position, name in enumerate(cut.groups):
            rows = index == position
            frame = summarize_bright_subsample_cut(
                df.loc[rows], keep[rows], cut.cut_for(name), z_range=z_range, n_bins=n_bins
            )
            frame.insert(0, "completeness_group", name)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)
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


def plot_completeness_per_redshift_peak_faint_cut(
    completeness_grid, mag_centers, z_centers, mag_edges, z_edges,
    selection, *, plot_dir,
):
    """Show the per-redshift normalization and the actual bright selection boundary."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Patch
    import matplotlib.patheffects as pe
    from qvc.hubble.hubble_completeness_refactored import trace_completeness_step

    c = np.asarray(completeness_grid, dtype=float)
    m, z = np.asarray(mag_edges), np.asarray(z_edges)
    mc, zc = np.asarray(mag_centers), np.asarray(z_centers)
    cut = selection.threshold(zc)
    limit = cut + selection.margin
    kind = "of peak" if selection.relative else "absolute"
    peaks = c.max(axis=0)
    rel = np.divide(100*c, peaks[None, :], out=np.zeros_like(c), where=peaks[None, :] > 0)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(11.8,5.4),gridspec_kw={'width_ratios':[1.4,1]},layout='constrained')
    im=ax.pcolormesh(m,z,np.clip(rel.T,.1,100),cmap='viridis',norm=LogNorm(.1,100),rasterized=True)
    zs=np.linspace(z[0],z[-1],1200);cuts=np.interp(zs,zc,cut);lims=cuts+selection.margin
    ax.fill_betweenx(zs,cuts,m[-1],facecolor='none',edgecolor='black',hatch='///',linewidth=0)
    line1,=ax.plot(lims,zs,color='#ff9600',ls='--',lw=2,label=f'Faintest qualifying bin ({100*selection.completeness_min:g}% {kind})')
    line2,=ax.plot(cuts,zs,color='#db1f3c',lw=2,label=f'Selection limit ({selection.margin:g} mag brighter)')
    for line in (line1,line2):line.set_path_effects([pe.Stroke(linewidth=3.8,foreground='white'),pe.Normal()])
    ax.set(xlim=(m[0],m[-1]),ylim=(z[0],z[-1]),xlabel=r'$m_{2500\,\mathrm{\AA}}$ (mag)',ylabel='Redshift $z$',title='Completeness relative to the peak at each redshift')
    cb=fig.colorbar(im,ax=ax,pad=.02,fraction=.045)
    cb.set_ticks([.1,1,10,100]);cb.set_ticklabels(['0.1','1','10','100']);cb.set_label(r'$100\,p(I=1\mid m,z)/\max_m p(I=1\mid m,z)$ (%)')
    valid = np.flatnonzero((peaks > 0) & (limit >= m[0]) & (limit <= m[-1]))
    if not valid.size:
        plt.close(fig)
        return None
    j = valid[np.argmin(abs(zc[valid]-2.))]
    zz=zc[j];v=rel[:,j]
    level = 100*selection.completeness_min if selection.relative else 100*selection.completeness_min/peaks[j]
    ax.axhline(zz,color='white',lw=1,ls=':',alpha=.9)
    bx.plot(mc,v,color='#234c73',marker='o',ms=3,lw=1.5,label='Completeness at bin centers')
    bx.axhline(level,color='.35',ls=':',lw=1.2,label=f'{level:g}% of this slice’s peak')
    bx.axvline(limit[j],color='#ff9600',ls='--',lw=2)
    bx.axvline(cut[j],color='#db1f3c',lw=2)
    bx.axvspan(cut[j],m[-1],facecolor='none',edgecolor='black',hatch='///',linewidth=0)
    bx.annotate('',xy=(cut[j],65),xytext=(limit[j],65),arrowprops={'arrowstyle':'<->','color':'black','lw':1.3})
    bx.text((cut[j]+limit[j])/2,78,f'{selection.margin:g} mag',ha='center',va='bottom',fontsize=10,bbox={'facecolor':'white','edgecolor':'none','pad':2})
    bx.scatter([limit[j]],[v[np.argmin(abs(mc-limit[j]))]],s=60,facecolor='#ff9600',edgecolor='black',zorder=5)
    bx.set(xlim=(m[0],m[-1]),ylim=(.1,160),yscale='log',xlabel=r'$m_{2500\,\mathrm{\AA}}$ (mag)',ylabel='Relative completeness (%)',title=rf'Example slice: $z={zz:.2f}$')
    bx.set_yticks([.1,1,10,100],labels=['0.1','1','10','100']);bx.legend(loc='lower left',fontsize=8,framealpha=1)
    fig.legend(handles=[line1,line2,Patch(facecolor='none',edgecolor='black',hatch='///',label='Excluded: fainter than selection limit')],loc='outside lower center',ncol=3,fontsize=9,frameon=False)
    out = Path(plot_dir) / "completeness_map_per_redshift_peak_faint_cut.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    with trace_completeness_step(out.name):
        fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Per-target-selection completeness maps
# ---------------------------------------------------------------------------
COMPLETENESS_MAP_GROUP_MODES = ("single", "spectrograph")
# Default when the option is not given; falls back to "single" (with a notice)
# when its prerequisites are missing (see hubble_fit.configure_completeness_run_args).
DEFAULT_COMPLETENESS_MAP_GROUPS = "spectrograph"
# SDSS-I/II spectra end with MJD 54663 (DR7); BOSS/eBOSS spectra start at 55000+.
SPECTROGRAPH_MJD_BOUNDARY = 55000.0
SPECTROGRAPH_GROUPS = ("sdss", "boss")
SPECTROGRAPH_GROUP_RULE = (
    f"spectrum mjd < {SPECTROGRAPH_MJD_BOUNDARY:g}: sdss (SDSS-I/II); otherwise boss (BOSS/eBOSS)"
)


def normalize_completeness_map_groups(mode):
    mode = "single" if mode is None else str(mode).strip().lower()
    if mode not in COMPLETENESS_MAP_GROUP_MODES:
        raise ValueError(
            f"completeness map groups must be one of {COMPLETENESS_MAP_GROUP_MODES}; got {mode!r}."
        )
    return mode


def assign_completeness_groups(df, mode, *, allow_missing=False):
    """Return a copy of ``df`` with the integer ``COMPLETENESS_GROUP_COL``.

    ``mode='spectrograph'`` splits by the MJD of the spectrum used in the fit
    (``SPECTROGRAPH_GROUP_RULE``).  Every row must carry a finite ``mjd``
    unless ``allow_missing``, in which case such rows get group -1 (no map;
    they must not enter the selection likelihood).
    """
    mode = normalize_completeness_map_groups(mode)
    if df is None or mode == "single":
        return df
    if "mjd" not in df.columns:
        raise KeyError("--completeness-map-groups spectrograph requires an 'mjd' column.")
    mjd = pd.to_numeric(df["mjd"], errors="coerce").to_numpy(dtype=float)
    bad = ~np.isfinite(mjd)
    if np.any(bad) and not allow_missing:
        raise ValueError(
            f"--completeness-map-groups spectrograph: {int(bad.sum())} rows lack a finite spectrum mjd."
        )
    out = df.copy()
    out.attrs.update(df.attrs)
    group = np.where(mjd < SPECTROGRAPH_MJD_BOUNDARY, 0, 1)
    out[COMPLETENESS_GROUP_COL] = np.where(bad, -1, group).astype(np.int64)
    return out


def completeness_group_prerequisite_problem(df):
    """Reason the spectrograph grouping cannot be built from ``df`` (None if it can)."""
    if df is None:
        return None
    if "mjd" not in df.columns:
        return "the input catalog has no spectrum 'mjd' column"
    mjd = pd.to_numeric(df["mjd"], errors="coerce").to_numpy(dtype=float)
    missing = int(np.count_nonzero(~np.isfinite(mjd)))
    if missing:
        return f"{missing} objects lack a finite spectrum mjd"
    return None


def group_index_from_frame(df):
    if COMPLETENESS_GROUP_COL not in df.columns:
        raise KeyError(f"Per-group completeness requires the {COMPLETENESS_GROUP_COL!r} column.")
    return np.asarray(df[COMPLETENESS_GROUP_COL].to_numpy(), dtype=np.int64)


@dataclass(frozen=True)
class GroupedBrightSubsampleCut:
    """One redshift-dependent bright cut per target-selection group.

    Each group's cut is derived from that group's own completeness map with
    the same rule (``completeness_min`` x per-redshift peak, minus ``margin``).
    """

    group_mode: str
    groups: tuple
    cuts: tuple
    group_rule: str = SPECTROGRAPH_GROUP_RULE

    def __post_init__(self):
        groups = tuple(str(name) for name in self.groups)
        cuts = tuple(self.cuts)
        if len(groups) < 1 or len(groups) != len(cuts):
            raise ValueError("groups and cuts must be nonempty and equal in length.")
        if not all(isinstance(cut, BrightSubsampleCut) for cut in cuts):
            raise TypeError("cuts must be BrightSubsampleCut instances.")
        first = cuts[0]
        for cut in cuts[1:]:
            if (cut.completeness_min, cut.margin, cut.relative) != (
                first.completeness_min, first.margin, first.relative
            ):
                raise ValueError("All group cuts must share the same rule.")
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "cuts", cuts)

    completeness_min = property(lambda self: self.cuts[0].completeness_min)
    margin = property(lambda self: self.cuts[0].margin)
    relative = property(lambda self: self.cuts[0].relative)

    def _index(self, group_index, shape):
        index = np.broadcast_to(np.asarray(group_index, dtype=np.int64), shape)
        if np.any((index < 0) | (index >= len(self.groups))):
            raise ValueError("completeness group index outside the configured groups.")
        return index

    def threshold(self, z, group_index):
        z = np.asarray(z, dtype=float)
        index = self._index(group_index, z.shape)
        out = np.empty(z.shape, dtype=float)
        for position, cut in enumerate(self.cuts):
            rows = index == position
            out[rows] = cut.threshold(z[rows])
        return out

    def mask(self, magnitude, z, group_index):
        magnitude, z = np.broadcast_arrays(np.asarray(magnitude, dtype=float), np.asarray(z, dtype=float))
        return np.isfinite(magnitude) & np.isfinite(z) & (magnitude <= self.threshold(z, group_index))

    def cut_for(self, name):
        return self.cuts[self.groups.index(name)]

    def run_tag(self):
        return self.cuts[0].run_tag() + f"_compgroups-{self.group_mode}"

    def to_json(self):
        return json.dumps(
            {
                "group_mode": self.group_mode,
                "group_rule": self.group_rule,
                "groups": list(self.groups),
                "cuts": {name: json.loads(cut.to_json()) for name, cut in zip(self.groups, self.cuts)},
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text):
        payload = json.loads(text)
        groups = tuple(payload["groups"])
        return cls(
            group_mode=payload["group_mode"],
            groups=groups,
            cuts=tuple(BrightSubsampleCut.from_json(json.dumps(payload["cuts"][name])) for name in groups),
            group_rule=payload.get("group_rule", ""),
        )

    def summary_frame(self):
        frames = []
        for name, cut in zip(self.groups, self.cuts):
            frame = cut.summary_frame()
            frame.insert(0, "completeness_group", name)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)


def bright_subsample_cut_from_json(text):
    """Decode either a single or a per-group bright cut."""
    payload = json.loads(text)
    if "groups" in payload:
        return GroupedBrightSubsampleCut.from_json(text)
    return BrightSubsampleCut.from_json(text)


class GroupedCompletenessModel:
    """Per-object dispatch to one (bright-cut-wrapped) map per group.

    Called with ``group_index`` (one integer per row of the redshift axis),
    each row is evaluated with its own group's map and cut.  Called without
    it, the model returns the union selection probability
    ``sum_g p_g(m, z) 1[m <= cut_g(z)]`` (used only for plots of the map).
    All group maps share the magnitude/redshift grids and support, so grid
    attributes fall through to the first group's model.
    """

    requires_group_index = True

    def __init__(self, models, cut, group_params=None, union_params=None):
        if not isinstance(cut, GroupedBrightSubsampleCut):
            raise TypeError("cut must be a GroupedBrightSubsampleCut.")
        models = tuple(models)
        if len(models) != len(cut.groups):
            raise ValueError("Need one completeness model per group.")
        self.__dict__["_models"] = models
        self.__dict__["bright_subsample_cut"] = cut
        self.__dict__["group_params"] = dict(group_params or {})
        self.__dict__["union_params"] = union_params

    @property
    def groups(self):
        return self.bright_subsample_cut.groups

    def __call__(self, mag, z, *extra, group_index=None):
        models = self.__dict__["_models"]
        if group_index is None:
            return sum(np.asarray(model(mag, z, *extra), dtype=float) for model in models)
        z = np.asarray(z, dtype=float)
        mag = np.asarray(mag, dtype=float)
        n_rows = z.shape[0]
        index = np.asarray(group_index, dtype=np.int64).reshape(-1)
        if index.shape[0] != n_rows:
            raise ValueError("group_index must have one entry per redshift row.")
        if np.any((index < 0) | (index >= len(models))):
            raise ValueError("completeness group index outside the configured groups.")
        shape = np.broadcast_shapes(mag.shape, z.shape, *(np.shape(value) for value in extra))
        out = np.zeros(shape, dtype=float)

        def rows_of(value, rows):
            value = np.asarray(value)
            return value[rows] if value.ndim and value.shape[0] == n_rows and n_rows != 1 else value

        for position, model in enumerate(models):
            rows = index == position
            if not np.any(rows):
                continue
            out[rows] = np.broadcast_to(
                np.asarray(model(rows_of(mag, rows), z[rows], *(rows_of(v, rows) for v in extra)), dtype=float),
                (int(rows.sum()),) + shape[1:],
            )
        return out

    def selection_cut_magnitude(self, z, group_index=None):
        if group_index is None:
            raise ValueError("Grouped completeness needs group_index for its hard cuts.")
        return self.bright_subsample_cut.threshold(z, group_index)

    def __getattr__(self, name):
        models = self.__dict__.get("_models")
        if not models or name.startswith("__"):
            raise AttributeError(name)
        return getattr(models[0], name)

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    @property
    def grid(self):
        grid = dict(getattr(self.__dict__["_models"][0], "grid", {}))
        grid["bright_subsample_cut"] = self.bright_subsample_cut.to_json()
        return grid


def build_grouped_completeness_params(build_map, parent, cut=None, *, completeness_min=None,
                                      margin=None, relative=True, group_mode="spectrograph",
                                      groups=SPECTROGRAPH_GROUPS, plot_path=None, plot=False):
    """Build one completeness map per group from the same mock.

    ``build_map(frame, count_scale=..., plot=..., plot_path=...)`` returns the
    usual completeness tuple for one parent (sub)sample.  The union map is built
    first; every group map reuses its mock-count normalization, so each group
    map is ``N_selected_in_group(m, z) / (scale * N_mock(m, z))`` and the group
    maps sum to the union map (up to the [0, 1] clip).  The mock is not split:
    it stands for the whole quasar population that either selection could
    have targeted.  When ``cut`` is None the per-group cuts are derived from
    each group's map with the single-map rule.  Returns
    ``(grouped_params, cut, union_params, group_params)``.
    """
    import os

    union = build_map(parent, count_scale=None, plot=plot, plot_path=plot_path)
    scale = float(getattr(union[0], "count_scale"))
    index = group_index_from_frame(parent)
    group_params = {}
    for position, name in enumerate(groups):
        subset = parent.loc[index == position]
        if len(subset) == 0:
            raise ValueError(f"Completeness group {name!r} has no parent objects.")
        group_params[name] = build_map(
            subset,
            count_scale=scale,
            plot=plot,
            plot_path=None if plot_path is None else os.path.join(plot_path, f"completeness_group_{name}"),
        )
    if cut is None:
        cut = GroupedBrightSubsampleCut(
            group_mode=group_mode,
            groups=tuple(groups),
            cuts=tuple(
                derive_bright_subsample_cut(
                    params[0], params[1], params[2],
                    completeness_min=completeness_min, margin=margin, relative=relative,
                )
                for params in group_params.values()
            ),
        )
    models = [BrightCutCompletenessModel(group_params[name][0], cut.cut_for(name)) for name in groups]
    grouped = (GroupedCompletenessModel(models, cut, group_params, union),) + tuple(union[1:])
    return grouped, cut, union, group_params
