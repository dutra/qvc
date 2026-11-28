"""Stacked variability amplitude against rest wavelength, and the variability slope eta it predicts.

Diagnostic of the Hubble run, the method of the one-off figure variability_curve_and_band_windows_apj
(plots/diagnostics/gold_sample/drop_blue_bands_refit/scripts/s09_eta_dip_diagnosis.py, composite() and the
eta bins; s14_figure_curve.py, local slopes and predicted eta), computed for the reduction the run used:

* per quasar and band b: q_b = a_b - log10(k_b), with a_b the model-free amplitude (log10 of the excess rms of the
  SDSS Stripe 82 light curve over its mean total flux; the amplitude table of the flat-or-tilted test) and k_b the
  fraction of the total flux the run's light curves keep as varying (psf_agn_fraction_lc_posterior_<b>); a band
  is used if its noise is below 15% of the mean flux (relnoise < 0.15); quasars with at least two bands.
* stacked curve: q = a_o + S(log lambda_rest), fitted by alternating least squares (40 iterations) with one
  intercept a_o per quasar (the per-object normalization) and S on a grid of 0.03 dex in log lambda_rest from
  800 to 6000 A (each band at the grid point nearest its pivot / (1 + z)); S is set to 0 at 3000 A and kept only at
  grid points with at least 20 bands; errors from 60 bootstrap resamples over quasars.
* local slopes of S in fixed rest-wavelength ranges (weighted straight line in log lambda).
* predicted eta(z): slope of a straight line through S at the four band pivots u, g, r, i (3543, 4770, 6231,
  7625 A divided by 1 + z), only where all four lie inside the stacked curve.
* fitted eta: median of the run's eta_sigma in redshift bins (bootstrap errors, 500 resamples), the median
  posterior width (eta_sigma_err), and the predicted eta at the bin's median redshift.
* summary numbers: slope of fitted eta against redshift (per-quasar least squares over the sample, bootstrap
  error over quasars), and the rms difference between predicted and fitted eta over the bins where both exist.

If the reduction is right and the amplitude follows one power law, the curve is straight and the two eta lines
are flat and agree. Every output names the reduction (the psf-fraction option recorded in the light-curve file and
the share of line light kept). All bootstraps use local generators with fixed seeds (no global random state).

Amplitude table: the pipeline uses the table of the flat-or-tilted test (amps_cc: cloud-cut epochs, rebuilt on
24 Nov), not the table the reference figure was made from (cosmology_independent_z_diagnostics/tables/
long.parquet). About 1.3% of the band rows differ by more than 0.01 dex; on the reference sample this moves the
stacked curve by up to 0.02 dex and the predicted eta by up to 0.03. The agreement to 1e-16 with the reference
tables holds only when the reference inputs are used (tests/test_variability_curve.py).

The continuum used for the reduction label is chosen as in the subtraction test (``continuum``; never rebuilt
silently), and "line light kept" is computed by the same helper on the same rows (the sample's light-curve rows).

Inputs: the run's light-curve HDF5 (object_id, z, eta_sigma, eta_sigma_err, kept fractions; f_AGN_psf_<b> and
f_cont_psf_<b> only for the label) and the amplitude table, resolved as in qvc.hubble.subtraction_test. Missing
inputs or fewer than MIN_QUASARS quasars: one warning, None returned, nothing raised.
Outputs: <out_dir>/variability_curve_and_band_windows_<label>.pdf and .json.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from qvc.hubble.subtraction_test import (  # noqa: F401  (REPO_ROOT re-exported for the tests)
    REPO_ROOT, BANDS, LAM_EFF, MissingInput, _norm_id, _style, file_record, kept_share, light_curve_continuum,
    note_continuum, read_amplitudes, read_fit_fractions, reduction_label, resolve_input, warn_missing_style,
)

RELNOISE_MAX = 0.15
GRID = np.arange(np.log10(800), np.log10(6000) + 1e-9, 0.03)
GRID_STEP = 0.03
NORM_LAM = 3000.0
MIN_BANDS_PER_POINT = 20
ALS_ITER = 40
N_BOOT_CURVE = 60
N_BOOT_MEDIAN = 500
N_BOOT_TREND = 200
SEED = 20261125
Z_EDGES = (0.44, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.2)
Z_GRID = np.round(np.arange(0.40, 3.201, 0.01), 3)
MIN_PER_BIN = 5
MIN_Z_SPREAD = 0.3
RANGES = ((980, 1215), (1215, 1400), (1390, 2110), (2250, 2600), (2770, 3420), (3410, 4200), (4200, 5200))
PIVOTS = np.array([LAM_EFF[b] for b in BANDS])
MIN_QUASARS = 50
FILE_STEM = "variability_curve_and_band_windows"


# ----------------------------------------------------------------------------------------------- inputs
def read_run_light_curves(lc_path, first_fit=None, continuum="auto"):
    """Per object x band kept fraction (and f_AGN, f_cont for the label); per object z, eta and its width."""
    import h5py

    p = resolve_input(lc_path, "light-curve file")
    with h5py.File(p, "r") as h:
        keys = set(h.keys())
        keep_col = {}
        for b in BANDS:
            for c in (f"psf_agn_fraction_lc_posterior_{b}", f"psf_constant_flux_band_summaries_{b}_agn_fraction"):
                if c in keys:
                    keep_col[b] = c
                    break
        missing = [k for k in ("object_id", "z", "eta_sigma") if k not in keys]
        missing += [f"psf_agn_fraction_lc_posterior_{b}" for b in BANDS if b not in keep_col]
        if missing:
            raise MissingInput(f"light-curve file {p.name} lacks {', '.join(missing[:6])}")
        n = len(h["z"])
        d = {"object_id": [_norm_id(v) for v in h["object_id"][()]], "z": np.asarray(h["z"][()], float),
             "eta": np.asarray(h["eta_sigma"][()], float),
             "eta_err": np.asarray(h["eta_sigma_err"][()], float) if "eta_sigma_err" in keys else np.full(n, np.nan),
             "fit_ok": np.asarray(h["fit_ok"][()], bool) if "fit_ok" in keys else np.ones(n, bool)}
        cont, cont_rec = (light_curve_continuum(h, keys, continuum, p.name) if first_fit is None
                          else (None, dict(source_requested=continuum, source_used="first spectral fit", warning=None)))
        for b in BANDS:
            d[f"keep_{b}"] = np.asarray(h[keep_col[b]][()], float)
            d[f"cont_{b}"] = cont[b] if cont is not None else np.full(n, np.nan)
            d[f"agn_{b}"] = np.asarray(h[f"f_AGN_psf_{b}"][()], float) if f"f_AGN_psf_{b}" in keys else np.full(n, np.nan)
        mode = ",".join(np.unique(np.asarray(h["psf_fraction"][()]).astype(str))) if "psf_fraction" in keys else None
    W = pd.DataFrame(d)
    W = W[W.fit_ok].drop_duplicates("object_id", keep="last")
    O = W[["object_id", "z", "eta", "eta_err"]].reset_index(drop=True)
    T = pd.concat([pd.DataFrame({"object_id": W.object_id.to_numpy(), "band": b, "z": W.z.to_numpy(),
                                 "f_keep": W[f"keep_{b}"].to_numpy(), "f_cont": W[f"cont_{b}"].to_numpy(),
                                 "f_AGN": W[f"agn_{b}"].to_numpy()}) for b in BANDS], ignore_index=True)
    info = dict(light_curve_file=file_record(p), psf_fraction=mode, kept_fraction_column=keep_col["g"].replace("_g", "_<b>"))
    info["continuum"] = cont_rec
    if first_fit is not None:
        F, rec = read_fit_fractions(first_fit, "first spectral fit")
        T = T.drop(columns=["f_AGN", "f_cont"]).merge(F, on=["object_id", "band"], how="left")
        info["first_fit"] = rec
    return T, O, info


def band_table(T, amps, object_ids=None):
    """Per object x band q = amplitude - log10(kept fraction), with the noise cut; rows with finite q only."""
    if object_ids is not None:
        ids = pd.Index(pd.Series(list(object_ids)).map(_norm_id).unique())
        T = T[T.object_id.isin(ids)]
    T = T.merge(amps, on=["object_id", "band"], how="inner")
    T = T.assign(lam_rest=T.band.map(LAM_EFF) / (1 + T.z))
    ok = (T.relnoise < RELNOISE_MAX) & np.isfinite(T.amp)
    with np.errstate(invalid="ignore", divide="ignore"):
        T = T.assign(q=np.where(ok, T.amp - np.log10(T.f_keep), np.nan))
    return T[np.isfinite(T.q)].reset_index(drop=True)


# -------------------------------------------------------------------------------------------- estimator
def _als(o, g, q, n):
    S = np.zeros(len(GRID))
    cg = np.bincount(g, minlength=len(GRID))
    co = np.bincount(o, minlength=n)
    for _ in range(ALS_ITER):
        a = np.bincount(o, q - S[g], minlength=n) / np.maximum(co, 1)
        S = np.bincount(g, q - a[o], minlength=len(GRID)) / np.maximum(cg, 1)
        S = S - np.interp(np.log10(NORM_LAM), GRID[cg > 0], S[cg > 0])
    S[cg < MIN_BANDS_PER_POINT] = np.nan
    return S, cg


def stacked_curve(Q, n_boot=N_BOOT_CURVE, seed=SEED):
    """Stacked amplitude S(log lambda_rest) from rows (object_id, lam_rest, q); quasars with >= 2 bands."""
    ids, o = np.unique(Q.object_id.to_numpy(), return_inverse=True)
    Q = Q[np.bincount(o)[o] >= 2]
    ids, o = np.unique(Q.object_id.to_numpy(), return_inverse=True)
    g = np.clip(np.round((np.log10(Q.lam_rest.to_numpy()) - GRID[0]) / GRID_STEP).astype(int), 0, len(GRID) - 1)
    q = Q.q.to_numpy()
    S0, cg = _als(o, g, q, len(ids))
    order = np.argsort(o, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(o[order])) + 1, len(o)]
    grp = [order[starts[k]:starts[k + 1]] for k in range(len(ids))]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        rows = np.concatenate([grp[k] for k in pick])
        newo = np.repeat(np.arange(len(ids)), [len(grp[k]) for k in pick])
        bs.append(_als(newo, g[rows], q[rows], len(ids))[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # grid points empty in every resample
        err = np.nanstd(np.array(bs), axis=0) if n_boot else np.full(len(GRID), np.nan)
    C = pd.DataFrame(dict(log_lam=GRID, lam=10 ** GRID, S=S0, S_err=err, n=cg))
    return C, int(len(ids))


def local_slopes(C):
    C = C[np.isfinite(C.S) & np.isfinite(C.S_err)]
    rows = []
    for lo, hi in RANGES:
        s = C[(C.lam >= lo) & (C.lam <= hi)]
        if len(s) > 2:
            p, cov = np.polyfit(s.log_lam, s.S, 1, w=1 / s.S_err.clip(lower=1e-3), cov="unscaled")
            rows.append(dict(lam_lo=lo, lam_hi=hi, n_points=len(s), slope=float(p[0]), slope_err=float(np.sqrt(cov[0, 0]))))
        elif len(s) == 2:
            rows.append(dict(lam_lo=lo, lam_hi=hi, n_points=2, slope=float(np.polyfit(s.log_lam, s.S, 1)[0]),
                             slope_err=float("nan")))
    return pd.DataFrame(rows, columns=["lam_lo", "lam_hi", "n_points", "slope", "slope_err"])


def predicted_eta(C, zgrid=Z_GRID):
    C = C[np.isfinite(C.S)]
    lx, s = C.log_lam.to_numpy(), C.S.to_numpy()
    rows = []
    for z in zgrid:
        xb = np.log10(PIVOTS / (1 + z))
        inside = bool(len(lx) and (xb >= lx.min()).all() and (xb <= lx.max()).all())
        rows.append(dict(z=float(z), inside=inside,
                         eta_pred=float(np.polyfit(xb, np.interp(xb, lx, s), 1)[0]) if inside else np.nan))
    return pd.DataFrame(rows)


def _bmed(x, rng, nb=N_BOOT_MEDIAN):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < MIN_PER_BIN:
        return np.nan, np.nan, len(x)
    return float(np.median(x)), float(np.std(np.median(rng.choice(x, (nb, len(x))), axis=1), ddof=1)), len(x)


def eta_bins(O, EP, edges=Z_EDGES, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    e = EP[EP.inside]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        t = O[(O.z >= lo) & (O.z < hi)]
        if not len(t):
            continue
        m, me, n = _bmed(t.eta, rng)
        zm = float(t.z.median())
        ee = e[(e.z >= lo) & (e.z < hi)]
        pred = float(np.interp(zm, ee.z, ee.eta_pred)) if len(ee) and ee.z.min() <= zm <= ee.z.max() else np.nan
        rows.append(dict(z_lo=lo, z_hi=hi, z_med=zm, n=n, eta_fit=m, eta_fit_err=me,
                         eta_posterior_width_med=float(np.nanmedian(t.eta_err)) if np.isfinite(t.eta_err).any() else np.nan,
                         eta_pred=pred))
    return pd.DataFrame(rows)


def eta_trend(O, n_bins_ok, seed=SEED, n_boot=N_BOOT_TREND):
    """Least-squares slope of fitted eta against z over the quasars, with a bootstrap error over quasars.

    NaN with a note when the redshift spread is too small (5-95% range below MIN_Z_SPREAD) or fewer than two
    redshift bins hold MIN_PER_BIN quasars."""
    O = O[np.isfinite(O.eta) & np.isfinite(O.z)]
    z, e = O.z.to_numpy(), O.eta.to_numpy()
    zr = (float(z.min()), float(z.max())) if len(z) else (float("nan"), float("nan"))
    spread = float(np.subtract(*np.percentile(z, [95, 5]))) if len(z) else 0.0
    if spread < MIN_Z_SPREAD or n_bins_ok < 2:
        note = (f"not measured: redshift 5-95% range {spread:.2f} < {MIN_Z_SPREAD}" if spread < MIN_Z_SPREAD
                else f"not measured: {n_bins_ok} redshift bin(s) with at least {MIN_PER_BIN} quasars")
        return float("nan"), float("nan"), zr, int(len(z)), note
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = float(np.polyfit(z, e, 1)[0])
        rng = np.random.default_rng(seed + 2)
        b = [np.polyfit(z[k], e[k], 1)[0] for k in (rng.integers(0, len(z), len(z)) for _ in range(n_boot))]
    return s, float(np.std(b, ddof=1)), zr, int(len(z)), None


def compute(Q, O, object_ids=None, n_boot=N_BOOT_CURVE, seed=SEED):
    """Everything from the rows of band_table (object_id, lam_rest, q, f_keep, f_cont, f_AGN) and an object table
    (object_id, z, eta, eta_err), restricted to ``object_ids`` when given. The curve uses the quasars with a
    finite eta."""
    if object_ids is not None:
        ids = pd.Index(pd.Series(list(object_ids)).map(_norm_id).unique())
        O = O[O.object_id.isin(ids)]
    O = O[np.isfinite(O.eta)].reset_index(drop=True)
    Q = Q[Q.object_id.isin(O.object_id)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", getattr(getattr(np, "exceptions", np), "RankWarning", RuntimeWarning))
        return _compute(Q, O, n_boot, seed)


def _compute(Q, O, n_boot, seed):
    C, nq = stacked_curve(Q, n_boot, seed)
    if nq < MIN_QUASARS:
        raise MissingInput(f"only {nq} quasars have model-free amplitudes in at least two bands; at least {MIN_QUASARS} needed")
    EP = predicted_eta(C)
    EB = eta_bins(O, EP, seed=seed)
    tr, tr_err, zr, n_eta, tr_note = eta_trend(O, int((EB.n >= MIN_PER_BIN).sum()) if len(EB) else 0, seed)
    both = EB[np.isfinite(EB.eta_fit) & np.isfinite(EB.eta_pred)]
    rms = float(np.sqrt(np.mean((both.eta_pred - both.eta_fit) ** 2))) if len(both) else float("nan")
    zq = np.round(np.percentile(O.z, [10, 50, 90]), 1)
    return dict(n_quasars_curve=nq, n_bands_curve=int(len(Q)), n_quasars_eta=n_eta,
                curve=C[np.isfinite(C.S)].to_dict(orient="list"), local_slopes=local_slopes(C).to_dict(orient="records"),
                eta_bins=EB.to_dict(orient="records"), predicted_eta=EP[EP.inside].to_dict(orient="list"),
                eta_vs_z_slope=tr, eta_vs_z_slope_err=tr_err, eta_vs_z_note=tr_note, eta_vs_z_range=zr,
                rms_pred_minus_fit=rms,
                n_bins_compared=int(len(both)), band_bar_redshifts=[float(v) for v in zq],
                eta_median=float(np.median(O.eta)))


# ------------------------------------------------------------------------------------------------- plot
def plot(res, pdf_path, style_path=None):
    import matplotlib
    import matplotlib.style
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    styles = _style(style_path)
    C = pd.DataFrame(res["curve"])
    EB = pd.DataFrame(res["eta_bins"])
    EP = pd.DataFrame(res["predicted_eta"])
    rc = {"axes.unicode_minus": True, "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 8,
          "ytick.labelsize": 8, "legend.fontsize": 7, "axes.linewidth": 0.7}
    # Figure + Agg canvas, no pyplot: never opens a window or changes the caller's backend
    with matplotlib.style.context(styles), matplotlib.rc_context(rc):
        fig = Figure(figsize=(3.5, 5.0))
        FigureCanvasAgg(fig)
        try:
            gs = fig.add_gridspec(2, 1, left=0.185, right=0.97, bottom=0.08, top=0.985, hspace=0.32,
                                  height_ratios=[1.15, 1.0])
            ax = fig.add_subplot(gs[0])
            ax.errorbar(C.lam, C.S, C.S_err, fmt="o", color="k", mec="white", mew=0.4, ms=3.2, elinewidth=0.7, zorder=4)
            ax.plot(C.lam, C.S, color="k", lw=1.0, zorder=3)
            lo_c, hi_c = float(np.nanmin(C.S - C.S_err.fillna(0))), float(np.nanmax(C.S + C.S_err.fillna(0)))
            span = max(hi_c - lo_c, 0.2)
            xlo, xhi = float(C.lam.min()) / 1.12, float(C.lam.max()) * 1.10
            # reference: a power law with the median fitted eta, through the curve between 1390 and 2110 A
            ref = C[(C.lam >= 1390) & (C.lam <= 2110)]
            if len(ref) >= 2:
                x0, y0 = np.log10(ref.lam).mean(), ref.S.mean()
                xx = np.log10(np.array([max(1150.0, xlo), min(3300.0, xhi)]))
                yy = y0 + res["eta_median"] * (xx - x0)
                ax.plot(10 ** xx, yy, color="0.5", ls="--", lw=0.8, zorder=2)
                lo_c, hi_c = min(lo_c, yy.min()), max(hi_c, yy.max())
                span = max(hi_c - lo_c, 0.2)
            # band bars (u..i pivots) at three redshifts of the sample, below the curve; clipped to the axis
            # range of the curve, each label above the middle of its visible bar (inside the axes)
            bars = res["band_bar_redshifts"]
            ybar = [lo_c - 0.17 * span * (k + 1) for k in range(len(bars))][::-1]
            bx0, bx1 = xlo * 1.03, xhi / 1.03
            for z, y in zip(sorted(bars), ybar):
                piv = PIVOTS / (1 + z)
                a, b = max(piv[0], bx0), min(piv[-1], bx1)
                if a >= b:
                    continue
                ax.plot([a, b], [y, y], color="k", lw=1.0, solid_capstyle="butt")
                for pv in piv:
                    if bx0 <= pv <= bx1:
                        ax.plot([pv, pv], [y - 0.018 * span, y + 0.018 * span], color="k", lw=0.8)
                ax.text(np.sqrt(a * b), y + 0.03 * span, f"z = {z:.1f}", fontsize=6.3, ha="center", va="bottom",
                        color="k")
            # above the data: the line markers at the top edge, the reduction label below them on the right
            ytop = hi_c + 0.95 * span
            for lam, lab in ((1215.67, r"Ly$\alpha$"), (3646.0, "Balmer edge")):
                if xlo * 1.08 < lam < xhi / 1.25:
                    ax.plot([lam, lam], [ytop, ytop - 0.06 * span], color="k", lw=0.8, zorder=5)
                    ax.text(lam, ytop - 0.075 * span, lab, fontsize=6.3, ha="center", va="top", color="k")
            med = f"{res['eta_median']:.2f}".replace("-", "\u2212")
            ax.text(xhi / 1.05, hi_c + 0.06 * span,
                    f"{res['reduction']}\nN = {res['n_quasars_curve']:,} (curve), {res['n_quasars_eta']:,} ($\\eta$)"
                    f"\ndashed: median fitted $\\eta$ = {med}",
                    fontsize=6.0, ha="right", va="bottom", color="k", linespacing=1.3)
            ax.set_xscale("log")
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(min(ybar) - 0.08 * span, ytop)
            ticks = [t for t in (1000, 2000, 3000, 5000) if xlo < t < xhi]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(r"Rest wavelength ($\mathrm{\AA}$)")
            ax.set_ylabel("Variability amplitude (dex)")

            ax2 = fig.add_subplot(gs[1])
            ok = np.isfinite(EB.eta_fit)
            if np.isfinite(res["eta_vs_z_slope"]):
                tr = f"{res['eta_vs_z_slope']:+.3f} \u00b1 {res['eta_vs_z_slope_err']:.3f}".replace("-", "\u2212")
                fit_label = f"fits (slope with z {tr})"
            else:
                fit_label = "fits (slope with z not measured)"
            ax2.errorbar(EB.z_med[ok], EB.eta_fit[ok], EB.eta_fit_err[ok], fmt="o", color="k", mec="white", mew=0.4,
                         ms=3.5, elinewidth=0.7, zorder=4, label=fit_label)
            ax2.plot(EB.z_med[ok], EB.eta_fit[ok], color="k", lw=1.0, zorder=3)
            if len(EP):
                ax2.plot(EP.z, EP.eta_pred, color="#0072B2", lw=1.3, ls="--", zorder=2, label="predicted from the stacked curve")
            ax2.set_xlim(0.4, 3.2)
            vals = np.r_[EB.eta_fit[ok] - EB.eta_fit_err[ok], EB.eta_fit[ok] + EB.eta_fit_err[ok], EP.eta_pred if len(EP) else []]
            pad = 0.15 * max(np.ptp(vals), 0.2)
            ax2.set_ylim(vals.min() - 2.2 * pad, vals.max() + pad)
            ax2.set_xlabel("Redshift")
            ax2.set_ylabel(r"Variability slope $\eta$")
            h, lab = ax2.get_legend_handles_labels()
            ax2.legend(h[::-1], lab[::-1], loc="best", frameon=False, handlelength=2.0, handletextpad=0.4,
                       labelspacing=0.2, borderaxespad=0.3)
            Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(pdf_path)
        finally:
            fig.clear()
    return styles


# --------------------------------------------------------------------------------------------- run hook
def eta_trend_text(res):
    if not np.isfinite(res["eta_vs_z_slope"]):
        return f"eta vs z {res['eta_vs_z_note']}"
    return f"fitted eta vs z slope {res['eta_vs_z_slope']:+.3f} \u00b1 {res['eta_vs_z_slope_err']:.3f} per unit z"


def rms_text(res):
    n = res["n_bins_compared"]
    if n == 0:
        return "rms(predicted - fitted eta) not available (no bin with both)"
    return (f"rms(predicted - fitted eta) {res['rms_pred_minus_fit']:.3f} over {n} bin" + ("s" if n > 1 else "")
            + (" (one bin only: not a test)" if n == 1 else ""))


def run_diagnostic(lc_path, out_dir, object_ids=None, *, amplitudes=None, first_fit=None, continuum="auto",
                   label="postcut", style_path=None, sample_info=None, log=print):
    """Compute, plot and record; never raises. Returns the result dict, or None when skipped."""
    tag = "[variability curve]"
    try:
        T, O, info = read_run_light_curves(lc_path, first_fit, continuum)
        note_continuum(info, log, "[subtraction test / variability curve]")
        amps, amps_rec = read_amplitudes(amplitudes)
        if object_ids is not None:
            ids = pd.Index(pd.Series(list(object_ids)).map(_norm_id).unique())
            T, O = T[T.object_id.isin(ids)], O[O.object_id.isin(ids)]
        Q = band_table(T, amps)
        res = compute(Q, O)
        res.update(info)
        res["kept_line_share_median"] = kept_share(T)
        res.update(amplitudes_file=amps_rec, sample=label,
                   n_objects_requested=None if object_ids is None else int(pd.Series(list(object_ids)).nunique()))
        res["reduction"] = reduction_label(info, res["kept_line_share_median"])
        stem = f"{FILE_STEM}_{label}" if label else FILE_STEM
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        styles = plot(res, out_dir / f"{stem}.pdf", style_path)
        res["style_file"] = styles[0] if styles else None
        res["sample_selection"] = sample_info
        warn_missing_style(styles, style_path, log, tag)
        with open(out_dir / f"{stem}.json", "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        log(f"{tag} {res['reduction']}: {eta_trend_text(res)}; {rms_text(res)}; {res['n_quasars_curve']} quasars "
            f"in the curve, {res['n_quasars_eta']} with eta -> {out_dir / (stem + '.pdf')}")
        return res
    except MissingInput as exc:
        log(f"{tag} WARNING: skipped, {exc}. The Hubble fit is not affected.")
    except Exception as exc:  # a diagnostic must never stop a Hubble run
        log(f"{tag} WARNING: skipped after an error ({type(exc).__name__}: {exc}). The Hubble fit is not affected.")
    return None
