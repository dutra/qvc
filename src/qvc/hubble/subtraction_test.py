"""Cosmology-independent test of the constant-flux subtraction ("flat or tilted").

Each light-curve band is fitted after a constant flux is removed (host, and the part of the line, Fe II and
Balmer-continuum flux assumed not to vary). If the removed flux is right, the variability amplitude of a band
does not depend on how much line light the band holds; if too much (too little) is removed, bands with more
line light look more (less) variable.

Estimator (the instrumental-variable version of the one-off figure 3_subtraction_test_flat_or_tilted,
plots/diagnostics/gold_sample/responsivity_plot_for_colin/scripts/f7_fig1_simple_v3.py, with the pair
building of cosmology_independent_z_diagnostics/independent_check/scripts/lib.py):

* per quasar and band b: amplitude a_b = log10(sqrt(var_obs - noise var) / mean flux), model-free, from the
  SDSS Stripe 82 epochs (relative to the total flux, so independent of the reduction); line-light excess
  D_b = log10(f_AGN / f_cont) of the spectral fit used for the subtraction (f_AGN: everything but the host);
  kept fraction k_b = the fraction of the total flux the light curves keep as varying.
* rows: (quasar, adjacent band pair u-g, g-r, r-i) with finite amplitudes, rest wavelength > 1350 A,
  f_cont > 0.05 and noise below 15% of the mean flux in both bands; 5-sigma outliers of the amplitude colour
  removed; weights 1 / robust variance of that colour per pair.
* y = (a_b - a_b') - log10(k_b / k_b')   (amplitude ratio relative to the kept flux)
  x = D_b - D_b' (first fit),  zz = the same from a second, independent spectral fit.
  All three are demeaned in cells of (band pair, dz = 0.05), i.e. compared between quasars at fixed redshift.
* slope = cov_w(y, zz) / cov_w(x, zz) (instrumental variable: errors of the first fit do not bias it); it is
  drawn as y against x_hat = gamma zz, gamma = cov_w(x, zz) / var_w(zz), so the drawn slope is the IV slope.
  9 quantile bins of x_hat; error = std of 100 bootstrap resamples over quasars.
* verdict: "flat within 2 sigma" if |slope| < 2 x error, otherwise "tilted: too much removed" (slope > 0) or
  "tilted: too little removed" (slope < 0).

The test is computed for the kept fraction of the run and, on the same rows, for the two limits:
lines constant (k = f_cont) and lines varying like the disc (k = f_AGN, host only).

Known properties (measured on synthetic null samples by an independent review; inherited unchanged from the
reference scripts): the bootstrap error is 5-10% too small (rms pull 1.06-1.10), so the 2-sigma rule calls
"tilted" in about 7-9% of null cases instead of 5%; and trimming and weighting on the raw amplitude colour
give a small bias of +0.026 +- 0.006 in the slope under the null.

Inputs (each resolved like the pipeline's data files: QVC_DATA_DIR, the working directory, the repository
root, the repository data/ folder; the default tables then fall back to their original plots/ locations):
  light-curve HDF5 of the run: object_id, z, fit_ok, psf_agn_fraction_lc_posterior_<b>, f_AGN_psf_<b>,
      f_cont_psf_<b>. The continuum is an explicit, recorded choice (``continuum``; CLI
      --subtraction-test-continuum): "auto" (default) uses ``*_f_cont_orig_psf_<b>`` when present (written by
      reductions that rewrote f_cont_psf_<b> to the kept flux) and f_cont_psf_<b> as stored otherwise; "stored",
      "orig" and "rebuild" (f_AGN - f_line - f_feii - f_bc) force one source. Nothing is rebuilt silently. The
      closure (f_cont + f_line + f_feii + f_bc) / f_AGN is computed per band (median over quasars); if the stored
      continuum is used and any band deviates from 1 by more than CLOSURE_TOL, one warning is printed (the stored
      continuum may hold the kept flux, so the label and the "lines constant" panel may be wrong) and the label
      reads "(unverified)". A first spectral fit (--subtraction-test-first-fit) replaces f_AGN and f_cont;
  amplitudes parquet: object_id, band, Fbar, var_obs, nv_cal (cosmology_independent_z_diagnostics/
      independent_check/scripts/b2_amps.py, cloud-cut epochs);
  second spectral fit: a parquet (object_id, f_AGN_<b>, f_cont_<b>, optional spec_src; ``path#src`` selects
      spec_src) or a spectral-fit HDF5 catalog (group ``catalog`` or top level, with object_id, f_AGN_psf_<b>,
      f_cont_psf_<b>; ``#src`` is rejected there);
  first spectral fit (optional, same formats): replaces f_AGN / f_cont of the light-curve file, for reductions
      whose file does not store them (host-only reductions store only f_AGN_psf_<b>).
The JSON records the resolved path, size, modification time and sha256 of every input file.

Missing inputs: run_diagnostic never raises. If the light-curve file lacks the fractions (e.g. a minimal-format
input), an input file does not exist, or fewer than MIN_QUASARS quasars have every input (amplitudes exist only
for the SDSS Stripe 82 quasars of the amplitude table; the default second fit covers about 2,000 of them), it
prints one warning naming what is missing and returns None.

Same model: if the second fit is the same file as the first (or the light-curve file), or the two give nearly
identical line fractions (within-cell correlation > INDEPENDENCE_CORR_MAX), the instrumental variable reduces to
the plain regression, which errors in the line fractions bias. The plot and JSON are still written, but every
panel says "not tested: both spectral fits are the same model" (flat = null) and the slopes are labelled
uncorrected.

Outputs: <out_dir>/subtraction_test_flat_or_tilted_<label>.pdf and .json (slopes, errors, binned points,
N, input files), and one log line with the slope and verdict of the run.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = ("u", "g", "r", "i")
PAIRS = (("u", "g"), ("g", "r"), ("r", "i"))
LAM_EFF = dict(u=3543.0, g=4770.0, r=6231.0, i=7625.0)
LAM_REST_MIN = 1350.0
F_CONT_MIN = 0.05
RELNOISE_MAX = 0.15
TRIM_NSIG = 5.0
CELL_DZ = 0.05
N_BINS = 9
N_BOOT = 100
SEED = 6
FLAT_NSIGMA = 2.0
MIN_QUASARS = 50
INDEPENDENCE_CORR_MAX = 0.97
CONTINUUM_SOURCES = ("auto", "stored", "orig", "rebuild")
CLOSURE_TOL = 0.01
_CONTINUUM_WARNED = set()
NOT_TESTED = "not tested: both spectral fits are the same model"


def _discover_root():
    """Repository root: the first parent holding pyproject.toml (as qvc.hubble.hubble_utils does)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]


REPO_ROOT = _discover_root()
# Default input tables, relative to the data directory; the original plots/ locations are the last fallback.
DEFAULT_AMPLITUDES = "subtraction_test/sdss_s82_band_amplitudes.parquet"
DEFAULT_AMPLITUDES_FALLBACK = ("plots/diagnostics/gold_sample/cosmology_independent_z_diagnostics/"
                               "independent_check/tables/amps_cc.parquet")
DEFAULT_SECOND_FIT = "subtraction_test/second_spectral_fit_fractions.parquet#new"
DEFAULT_SECOND_FIT_FALLBACK = "plots/diagnostics/gold_sample/responsivity_test/fractions.parquet#new"
DEFAULT_STYLE = "src/style_modern.mplstyle"
FILE_STEM = "subtraction_test_flat_or_tilted"
PANELS = ("run", "constant", "disc")


class MissingInput(RuntimeError):
    """An input the test needs is not available; the message says which."""


# ----------------------------------------------------------------------------------------------- inputs
def _candidates(path_like):
    p = Path(os.path.expanduser(str(path_like)))
    if p.is_absolute():
        return [p]
    out = []
    env = os.environ.get("QVC_DATA_DIR")
    if env:
        env = Path(env).expanduser()
        out.append(env / p)
        if p.parts[:1] == ("data",):
            out.append(env / Path(*p.parts[1:]))
    out += [Path.cwd() / p, REPO_ROOT / p]
    if p.parts[:1] != ("data",):
        out.append(REPO_ROOT / "data" / p)
    return out


def resolve_input(path_like, what, fallbacks=()):
    """First existing file among the pipeline-style candidates of ``path_like``, then of each fallback."""
    tried = []
    for spec in (path_like, *fallbacks):
        for c in _candidates(spec):
            if c.exists():
                return c.resolve()
            tried.append(str(c))
    raise MissingInput(f"{what} not found (tried {', '.join(dict.fromkeys(tried))})")


def file_record(path):
    p = Path(path)
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    st = p.stat()
    return dict(path=str(p), size_bytes=int(st.st_size),
                mtime=datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(), sha256=h.hexdigest())


def _split_source(spec):
    spec = str(spec)
    if "#" in spec:
        path, src = spec.rsplit("#", 1)
        return path, (src or None)
    return spec, None


def _norm_id(v):
    s = v.decode() if isinstance(v, bytes) else str(v)
    s = s.strip()
    return s[:-2] if s.endswith(".0") else s


def light_curve_continuum(h, keys, source="auto", name="light-curve file"):
    """Continuum fraction per band of the fit the light curves were subtracted with, by an explicit choice.

    Returns ({band: array} or None if unavailable, record). The record holds the requested and used source, the
    per-band closure of the stored continuum (median over quasars of (f_cont + f_line + f_feii + f_bc) / f_AGN)
    and, when the stored continuum is used and does not close, the warning text."""
    if source not in CONTINUUM_SOURCES:
        raise MissingInput(f"continuum source '{source}' unknown (choose from {', '.join(CONTINUUM_SOURCES)})")
    orig = {b: sorted(k for k in keys if k.endswith(f"_f_cont_orig_psf_{b}")) for b in BANDS}
    have_orig = all(orig.values())
    have_stored = all(f"f_cont_psf_{b}" in keys for b in BANDS)
    comps = ("f_line_psf", "f_feii_psf", "f_bc_psf")
    have_comps = all(f"{c}_{b}" in keys for c in comps + ("f_AGN_psf",) for b in BANDS)
    rest = {b: sum(np.asarray(h[f"{c}_{b}"][()], float) for c in comps) for b in BANDS} if have_comps else None
    agn = {b: np.asarray(h[f"f_AGN_psf_{b}"][()], float) for b in BANDS} if have_comps else None
    stored = {b: np.asarray(h[f"f_cont_psf_{b}"][()], float) for b in BANDS} if have_stored else None

    def closure(cont):
        if cont is None or not have_comps:
            return None
        with np.errstate(invalid="ignore", divide="ignore"):
            return {b: float(np.nanmedian((cont[b] + rest[b]) / agn[b])) for b in BANDS}

    used = {"auto": "orig" if have_orig else "stored"}.get(source, source)
    rec = dict(source_requested=source, source_used=used, closure_stored=closure(stored), warning=None)
    if used == "orig":
        if not have_orig:
            raise MissingInput(f"{name} has no *_f_cont_orig_psf_<b> columns (continuum source 'orig')")
        cont = {b: np.asarray(h[orig[b][0]][()], float) for b in BANDS}
        rec["column"] = orig["g"][0][:-1] + "<b>"
        rec["closure_orig"] = closure(cont)
    elif used == "rebuild":
        if not have_comps:
            raise MissingInput(f"{name} lacks f_line/f_feii/f_bc/f_AGN_psf_<b> needed to rebuild the continuum")
        cont = {b: np.clip(agn[b] - rest[b], 1e-6, None) for b in BANDS}
        rec["column"] = "f_AGN_psf_<b> - f_line_psf_<b> - f_feii_psf_<b> - f_bc_psf_<b>"
    else:
        cont = stored
        rec["column"] = "f_cont_psf_<b>"
        c = rec["closure_stored"]
        if cont is not None and c and any(abs(v - 1) > CLOSURE_TOL for v in c.values()):
            rec["warning"] = (
                f"in {name} the stored continuum does not close with the components (closure "
                + ", ".join(f"{b} {v:.3f}" for b, v in c.items())
                + "; 1 = closes): f_cont_psf_<b> may have been overwritten with the kept flux, so the 'line light "
                "kept' label and the 'lines constant' panel may be wrong. Give the subtracted spectral fit with "
                "--subtraction-test-first-fit, or choose --subtraction-test-continuum rebuild (or orig)")
    return cont, rec


def note_continuum(info, log, tag):
    """Print the continuum warning once per light-curve file and source (both diagnostics read the same file)."""
    rec = info.get("continuum") or {}
    key = (info["light_curve_file"]["path"], rec.get("source_used"))
    if rec.get("warning") and key not in _CONTINUUM_WARNED:
        _CONTINUUM_WARNED.add(key)
        log(f"{tag} WARNING: {rec['warning']}")


def kept_share(T):
    """Median share of the non-continuum flux kept as varying, over the sample's (quasar, band) rows of the
    light-curve table (f_keep, f_cont, f_AGN) with more than 2% non-continuum flux; shared by both diagnostics."""
    U = T[np.isfinite(T.f_keep) & np.isfinite(T.f_cont) & np.isfinite(T.f_AGN) & ((T.f_AGN - T.f_cont) > 0.02)]
    share = ((U.f_keep - U.f_cont) / (U.f_AGN - U.f_cont)).to_numpy()
    return float(np.median(share)) if share.size else float("nan")


def reduction_label(info, share):
    """One short line naming the reduction: the psf-fraction option recorded in the file and the share kept."""
    parts = [f"psf fraction: {info['psf_fraction']}" if info.get("psf_fraction") else "psf fraction not recorded"]
    if np.isfinite(share):
        unverified = bool((info.get("continuum") or {}).get("warning"))
        parts.append(f"line light kept: {share:.0%}" + (" (unverified)" if unverified else ""))
    return "; ".join(parts)


def read_light_curve_fractions(lc_path, first_fit=None, continuum="auto"):
    """Per object x band: z, kept fraction, f_AGN and f_cont of the subtracting fit; plus a short description.

    f_AGN and f_cont come from the light-curve file unless ``first_fit`` (a spectral fit, as for the second
    fit) is given; use it for reductions whose file does not store them (e.g. host-only reductions)."""
    import h5py

    p = resolve_input(lc_path, "light-curve file")
    with h5py.File(p, "r") as h:
        keys = set(h.keys())
        need = ["object_id", "z"] + [f"psf_agn_fraction_lc_posterior_{b}" for b in BANDS]
        cont, cont_rec = (None, None)
        if first_fit is None:
            need += [f"f_AGN_psf_{b}" for b in BANDS]
            cont, cont_rec = light_curve_continuum(h, keys, continuum, p.name)
            if cont is None:
                need += [f"f_cont_psf_{b}" for b in BANDS]
        missing = [k for k in need if k not in keys]
        if missing:
            raise MissingInput(f"light-curve file {p.name} lacks the band fractions of its subtraction "
                               f"({', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''})"
                               + ("; give the spectral fit it subtracted as the first fit" if first_fit is None else ""))
        d = {"object_id": [_norm_id(v) for v in h["object_id"][()]], "z": np.asarray(h["z"][()], float)}
        d["fit_ok"] = np.asarray(h["fit_ok"][()], bool) if "fit_ok" in keys else np.ones(len(d["z"]), bool)
        for b in BANDS:
            d[f"keep_{b}"] = np.asarray(h[f"psf_agn_fraction_lc_posterior_{b}"][()], float)
            if first_fit is None:
                d[f"agn_{b}"] = np.asarray(h[f"f_AGN_psf_{b}"][()], float)
                d[f"cont_{b}"] = cont[b]
            else:
                d[f"agn_{b}"] = d[f"cont_{b}"] = np.full(len(d["z"]), np.nan)
        mode = None
        if "psf_fraction" in keys:
            vals = np.unique(np.asarray(h["psf_fraction"][()]).astype(str))
            mode = ",".join(vals)
    W = pd.DataFrame(d)
    W = W[W.fit_ok].drop_duplicates("object_id", keep="last")
    rows = []
    for b in BANDS:
        rows.append(pd.DataFrame({"object_id": W.object_id.to_numpy(), "band": b, "z": W.z.to_numpy(),
                                  "f_keep": W[f"keep_{b}"].to_numpy(), "f_AGN": W[f"agn_{b}"].to_numpy(),
                                  "f_cont": W[f"cont_{b}"].to_numpy()}))
    T = pd.concat(rows, ignore_index=True)
    info = dict(light_curve_file=file_record(p), psf_fraction=mode)
    if first_fit is None:
        info["first_fit"] = dict(source="light-curve file", columns=f"f_AGN_psf_<b>; continuum: {cont_rec['column']}",
                                 path=str(p), spec_src=None)
        info["continuum"] = cont_rec
    else:
        F, used = read_fit_fractions(first_fit, "first spectral fit")
        T = T.drop(columns=["f_AGN", "f_cont"]).merge(F, on=["object_id", "band"], how="left")
        info["first_fit"] = used
        info["continuum"] = dict(source_requested=continuum, source_used="first spectral fit", warning=None)
    return T, info


def read_amplitudes(path=None):
    """Model-free amplitude (log10 excess rms / mean flux) and relative noise per object x band."""
    if path is None:
        p = resolve_input(DEFAULT_AMPLITUDES, "per-band amplitudes", (DEFAULT_AMPLITUDES_FALLBACK,))
    else:
        p = resolve_input(path, "per-band amplitudes")
    A = pd.read_parquet(p, columns=["object_id", "band", "Fbar", "var_obs", "nv_cal"])
    A["object_id"] = A.object_id.map(_norm_id)
    V = (A.var_obs - A.nv_cal).to_numpy()
    F = A.Fbar.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        A["amp"] = np.where((V > 0) & (F > 0), 0.5 * np.log10(np.clip(V, 1e-30, None)) - np.log10(np.clip(F, 1e-30, None)),
                            np.nan)
        A["relnoise"] = np.sqrt(A.nv_cal.to_numpy()) / F
    return A[["object_id", "band", "amp", "relnoise"]].drop_duplicates(["object_id", "band"], keep="last"), file_record(p)


def read_fit_fractions(spec=None, what="second spectral fit"):
    """Per object x band f_AGN and f_cont of a spectral fit (parquet table or HDF5 catalog), and its record."""
    if spec is None:
        path, src = _split_source(DEFAULT_SECOND_FIT)
        fpath, fsrc = _split_source(DEFAULT_SECOND_FIT_FALLBACK)
        p = resolve_input(path, what, (fpath,))
        src = src if p.name == Path(path).name else fsrc
    else:
        path, src = _split_source(spec)
        p = resolve_input(path, what)
    if p.suffix.lower() in (".parquet", ".pq"):
        f = pd.read_parquet(p)
        if src is not None:
            if "spec_src" not in f:
                raise MissingInput(f"{p.name} has no spec_src column to select '{src}'")
            f = f[f.spec_src.astype(str) == src]
        names = {b: (f"f_AGN_{b}", f"f_cont_{b}") for b in BANDS}
    else:
        if src is not None:
            raise MissingInput(f"{what} {p.name} is an HDF5 catalog; '#{src}' (spec_src) applies only to parquet tables")
        import h5py
        with h5py.File(p, "r") as h:
            c = h["catalog"] if "catalog" in h else h
            cols = ["object_id"] + [f"f_{k}_psf_{b}" for k in ("AGN", "cont") for b in BANDS]
            miss = [k for k in cols if k not in c]
            if miss:
                raise MissingInput(f"{what} {p.name} lacks {', '.join(miss[:4])}")
            f = pd.DataFrame({k: c[k][()] for k in cols + (["fit_ok"] if "fit_ok" in c else [])})
        if "fit_ok" in f:
            f = f[f.fit_ok.astype(bool)]
        names = {b: (f"f_AGN_psf_{b}", f"f_cont_psf_{b}") for b in BANDS}
    f = f.copy()
    f["object_id"] = f.object_id.map(_norm_id)
    f = f.drop_duplicates("object_id", keep="last")
    if f.empty:
        raise MissingInput(f"{what} {p.name}{'#' + src if src else ''} has no rows")
    rows = []
    for b in BANDS:
        a, c = names[b]
        rows.append(pd.DataFrame({"object_id": f.object_id.to_numpy(), "band": b,
                                  "f_AGN": f[a].to_numpy(float), "f_cont": f[c].to_numpy(float)}))
    rec = file_record(p)
    rec["spec_src"] = src
    return pd.concat(rows, ignore_index=True), rec


def build_band_table(fractions, amplitudes, second, object_ids=None):
    """Merge the per object x band inputs; restrict to ``object_ids`` when given."""
    T = fractions
    if object_ids is not None:
        ids = pd.Index(pd.Series(list(object_ids)).map(_norm_id).unique())
        T = T[T.object_id.isin(ids)]
    with np.errstate(invalid="ignore", divide="ignore"):
        second = second.assign(D_second=np.log10(second.f_AGN / second.f_cont))[["object_id", "band", "D_second"]]
    T = T.merge(amplitudes, on=["object_id", "band"], how="left").merge(second, on=["object_id", "band"], how="left")
    with np.errstate(invalid="ignore", divide="ignore"):
        T["D"] = np.log10(T.f_AGN / T.f_cont)
    T["lam"] = T.band.map(LAM_EFF) / (1.0 + T.z)
    return T.reset_index(drop=True)


# -------------------------------------------------------------------------------------------- estimator
def make_pairs(T, kept=("f_keep", "f_cont", "f_AGN")):
    ok = (np.isfinite(T.amp) & np.isfinite(T.D) & (T.lam > LAM_REST_MIN) & (T.f_cont > F_CONT_MIN)
          & (T.relnoise < RELNOISE_MAX))
    W = T[ok]
    cols = ["z", "amp", "D", "D_second"] + list(kept)
    P = {b: W[W.band == b].set_index("object_id")[cols] for b in BANDS}
    out = []
    for p, (b1, b2) in enumerate(PAIRS):
        J = P[b1].join(P[b2], how="inner", lsuffix="1", rsuffix="2")
        d = pd.DataFrame({"object_id": J.index.to_numpy(), "pair": p, "z": J.z1.to_numpy()})
        d["ct"] = (J.amp1 - J.amp2).to_numpy()
        d["dD"] = (J.D1 - J.D2).to_numpy()
        d["dD2"] = (J.D_second1 - J.D_second2).to_numpy()
        for k in kept:
            with np.errstate(invalid="ignore", divide="ignore"):
                d["lk_" + k] = np.log10(J[k + "1"].to_numpy() / J[k + "2"].to_numpy())
            for j in "12":
                d[k + j] = J[k + j].to_numpy()
        out.append(d)
    C = pd.concat(out, ignore_index=True)
    return C[np.isfinite(C.ct)].reset_index(drop=True)


def _cells(C):
    return C.pair.astype(str) + "_" + np.floor(C.z / CELL_DZ).astype(int).astype(str)


def trim_and_weight(C):
    """5-sigma outliers of the amplitude colour about its cell median removed; weights 1/robust variance per pair."""
    g = _cells(C)
    r = C.ct.to_numpy() - C.groupby(g).ct.transform("median").to_numpy()
    keep = np.ones(len(C), bool)
    for p in C.pair.unique():
        m = (C.pair == p).to_numpy()
        s = 1.4826 * np.median(np.abs(r[m] - np.median(r[m])))
        keep[m] = np.abs(r[m] - np.median(r[m])) < TRIM_NSIG * s
    C = C[keep].reset_index(drop=True)
    g = _cells(C)
    r = C.ct.to_numpy() - C.groupby(g).ct.transform("median").to_numpy()
    w = np.ones(len(C))
    for p in C.pair.unique():
        m = (C.pair == p).to_numpy()
        w[m] = 1.0 / (1.4826 * np.median(np.abs(r[m] - np.median(r[m])))) ** 2
    C["w"] = w
    return C


def resid_cell(C, cols, w):
    g = C.pair.to_numpy() * 10000 + np.floor(C.z.to_numpy() / CELL_DZ).astype(int)
    _, gi = np.unique(g, return_inverse=True)
    W = np.bincount(gi, w)
    return [v - (np.bincount(gi, w * v) / W)[gi] for v in cols]


def iv_parts(C, kept):
    """Demeaned y, x_hat (first-fit x predicted from the second fit), weights, IV slope and first-stage corr."""
    w = C.w.to_numpy()
    y = C.ct.to_numpy() - C["lk_" + kept].to_numpy()
    yd, xd, zd = resid_cell(C, [y, C.dD.to_numpy(), C.dD2.to_numpy()], w)
    gam = np.sum(w * xd * zd) / np.sum(w * zd ** 2)
    xh = gam * zd
    corr = np.sum(w * xd * zd) / np.sqrt(np.sum(w * xd ** 2) * np.sum(w * zd ** 2))
    return yd, xh, w, float(np.sum(w * yd * xh) / np.sum(w * xh ** 2)), float(corr)


def binned(yd, xh, nbins=N_BINS):
    q = np.quantile(xh, np.linspace(0, 1, nbins + 1))
    q[0] -= 1e-9
    q[-1] += 1e-9
    bi = np.digitize(xh, q) - 1
    xm = np.array([xh[bi == i].mean() for i in range(nbins)])
    ym = np.array([yd[bi == i].mean() for i in range(nbins)])
    ye = np.array([yd[bi == i].std() / np.sqrt((bi == i).sum()) for i in range(nbins)])
    n = np.array([(bi == i).sum() for i in range(nbins)])
    return xm, ym, ye, n


def verdict(slope, err):
    flat = bool(abs(slope) < FLAT_NSIGMA * err)
    if flat:
        return flat, f"flat within {FLAT_NSIGMA:g}σ"
    return flat, ("tilted: too much removed" if slope > 0 else "tilted: too little removed")


def mark_same_model(res, reason):
    """Withdraw the verdicts: with one spectral model the slopes are the plain, uncorrected regression."""
    res["same_model"] = True
    res["same_model_reason"] = reason
    for P in res["panels"].values():
        P["flat"] = None
        P["verdict"] = NOT_TESTED
        P["slope_kind"] = "uncorrected (plain regression)"
    return res


def compute(T, n_boot=N_BOOT, seed=SEED):
    """Flat-or-tilted test on an object x band table (see build_band_table). Returns a dict of results."""
    kept = {"run": "f_keep", "constant": "f_cont", "disc": "f_AGN"}
    C = trim_and_weight(make_pairs(T))
    C = C[np.isfinite(C.dD2) & np.isfinite(C.lk_f_keep)].reset_index(drop=True)
    nq = int(C.object_id.nunique())
    if nq < MIN_QUASARS:
        raise MissingInput(f"only {nq} quasars have all inputs (amplitudes, both spectral fits); "
                           f"at least {MIN_QUASARS} needed")
    ids, inv = np.unique(C.object_id.to_numpy(), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(inv[order])) + 1, len(inv)]
    rows_of = [order[starts[j]:starts[j + 1]] for j in range(len(ids))]
    rng = np.random.default_rng(seed)
    picks = [rng.integers(0, len(ids), len(ids)) for _ in range(n_boot)]
    boot_idx = [np.concatenate([rows_of[j] for j in pk]) for pk in picks]
    out = dict(n_quasars=nq, n_pairs=int(len(C)), pairs_per_band_pair={
        "-".join(PAIRS[p]): int((C.pair == p).sum()) for p in range(len(PAIRS))}, panels={}, same_model=False)
    for key in PANELS:
        yd, xh, w, s, corr = iv_parts(C, kept[key])
        B = np.array([iv_parts(C.iloc[ix].reset_index(drop=True), kept[key])[3] for ix in boot_idx])
        se = float(np.std(B))
        xm, ym, ye, n = binned(yd, xh)
        flat, words = verdict(s, se)
        out["panels"][key] = dict(slope=s, err=se, flat=flat, verdict=words, slope_kind="instrumental variable",
                                  bins=dict(x=xm.tolist(), y=ym.tolist(), y_err=ye.tolist(), n=n.tolist()))
        out["first_second_fit_corr"] = corr
    out["kept_line_share_median"] = kept_share(T)
    if out["first_second_fit_corr"] > INDEPENDENCE_CORR_MAX:
        mark_same_model(out, f"within-cell correlation of the two fits' line-light differences "
                             f"{out['first_second_fit_corr']:.3f} > {INDEPENDENCE_CORR_MAX}")
    return out


# ------------------------------------------------------------------------------------------------- plot
def _pm(v, e):
    s = f"{v + 0.0:+.2f}"
    if float(s) == 0:
        s = "0.00"
    return s.replace("-", "−") + " ± " + f"{e:.2f}"


def _style(style_path):
    """Style file list for matplotlib.style.context; [] (matplotlib defaults) when the file is absent.

    The default style sheets are tracked in the repository's src/ folder: they are looked up next to the package
    (src/qvc/hubble/../../), then under the repository root, then with the data-path candidates."""
    name = style_path or DEFAULT_STYLE
    p = Path(os.path.expanduser(str(name)))
    if not p.is_absolute():
        for c in (Path(__file__).resolve().parents[2] / p.name, REPO_ROOT / p):
            if p.parts[:1] == ("src",) and c.exists():
                return [str(c.resolve())]
    try:
        return [str(resolve_input(name, "style file"))]
    except MissingInput:
        return []


_STYLE_WARNED = set()


def warn_missing_style(styles, style_path, log, tag, default=None):
    """One warning per style file when the sheet is missing and matplotlib defaults are used."""
    name = str(style_path or default or DEFAULT_STYLE)
    if not styles and name not in _STYLE_WARNED:
        _STYLE_WARNED.add(name)
        log(f"{tag} WARNING: style file {name} not found; matplotlib defaults used")


def plot(result, pdf_path, style_path=None):
    import matplotlib
    import matplotlib.style
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    styles = _style(style_path)
    share = result.get("kept_line_share_median", float("nan"))
    unver = " unverified" if (result.get("continuum") or {}).get("warning") else ""
    same = bool(result.get("same_model"))
    titles = {
        "run": "(a) This run" + (f" (line light kept: {share:.0%}{unver})" if np.isfinite(share) else ""),
        "constant": "(b) If the lines were constant",
        "disc": "(c) If the lines varied like the disc",
    }
    colours = {"run": "#0072B2", "constant": "0.45", "disc": "0.45"}
    # Figure + Agg canvas, no pyplot: never opens a window or changes the caller's backend; the style and rc
    # settings apply only inside these context managers
    with matplotlib.style.context(styles), matplotlib.rc_context({"axes.unicode_minus": True}):
        fig = Figure(figsize=(3 * 3.3, 4.4))
        FigureCanvasAgg(fig)
        axs = fig.subplots(1, 3, sharey=True, sharex=True,
                                gridspec_kw=dict(wspace=0.06))
        try:
            fig.subplots_adjust(left=0.075, right=0.995, bottom=0.13, top=0.77 if same else 0.79)
            allx = np.concatenate([result["panels"][k]["bins"]["x"] for k in PANELS])
            ally = np.concatenate([np.array(result["panels"][k]["bins"]["y"]) + s * np.array(result["panels"][k]["bins"]["y_err"])
                                   for k in PANELS for s in (-1, 1)])
            xr = 1.15 * np.max(np.abs(allx))
            yr = 1.15 * np.max(np.abs(ally))
            for ax, key in zip(axs, PANELS):
                P = result["panels"][key]
                b = P["bins"]
                ax.axhline(0, color="0.15", lw=0.8, ls="--", zorder=0)
                xx = np.array([-xr, xr])
                ax.plot(xx, P["slope"] * xx, color=colours[key], lw=1.3, zorder=1)
                ax.errorbar(b["x"], b["y"], yerr=b["y_err"], fmt="o", color="k", ms=3.6, mec="white", mew=0.5,
                            elinewidth=1.0, zorder=2)
                if same:
                    title = f"{titles[key]}\nuncorrected slope {_pm(P['slope'], P['err'])}\n{P['verdict']}"
                else:
                    title = f"{titles[key]}\nslope {_pm(P['slope'], P['err'])}; {P['verdict']}"
                ax.set_title(title, loc="left", fontsize=8 if same else 8.5, color="k", linespacing=1.4)
                if key == "run":
                    for sp in ax.spines.values():
                        sp.set_linewidth(1.4)
            axs[0].set_xlim(-xr, xr)
            axs[0].set_ylim(-yr, yr)
            axs[0].set_ylabel("Band varies more → (dex)")
            axs[1].set_xlabel("Band has more line light → (dex)")
            if same:
                third = (f"{result['n_quasars']:,} quasars. Not tested: both spectral fits are the same model, so the "
                         "slopes are the uncorrected regression.")
            else:
                third = (f"{result['n_quasars']:,} quasars, {result['n_pairs']:,} band pairs with two independent "
                         "spectral fits; x from the second fit; (b), (c): same objects, limiting cases.")
            fig.text(0.5, 0.985,
                     "Check of the constant flux removed from the light curves, without distances: if it is right,\n"
                     "bands with more line light vary as much as bands with less, and the points lie flat (dashed line).\n"
                     + third, ha="center", va="top", fontsize=8.5, linespacing=1.35)
            Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(pdf_path)
        finally:
            fig.clear()
    return styles


# --------------------------------------------------------------------------------------------- run hook
def _same_file(a, b):
    try:
        return a is not None and b is not None and os.path.samefile(a, b)
    except OSError:
        return False


def run_diagnostic(lc_path, out_dir, object_ids=None, *, amplitudes=None, second_fit=None, first_fit=None,
                   continuum="auto", label="postcut", style_path=None, sample_info=None, log=print):
    """Compute, plot and record the test; never raises. Returns the result dict, or None when skipped.

    ``amplitudes`` / ``second_fit`` None means the default tables (data directory, then the plots/ fallback)."""
    tag = "[subtraction test]"
    try:
        fr, info = read_light_curve_fractions(lc_path, first_fit, continuum)
        note_continuum(info, log, tag)
        amps, amps_rec = read_amplitudes(amplitudes)
        sec, sec_rec = read_fit_fractions(second_fit)
        T = build_band_table(fr, amps, sec, object_ids)
        res = compute(T)
        res.update(info)
        res["reduction"] = reduction_label(info, res["kept_line_share_median"])
        res.update(amplitudes_file=amps_rec, second_fit=sec_rec, sample=label,
                   n_objects_requested=None if object_ids is None else int(pd.Series(list(object_ids)).nunique()),
                   estimator="instrumental variable (second spectral fit), cells of band pair and dz 0.05, "
                             f"{N_BINS} quantile bins, {N_BOOT} bootstrap resamples over quasars",
                   flat_rule=f"|slope| < {FLAT_NSIGMA:g} x error")
        first = info["first_fit"]
        if not res["same_model"] and _same_file(sec_rec["path"], first["path"]) and \
                sec_rec.get("spec_src") == first.get("spec_src"):
            mark_same_model(res, "the second fit is the same file and source as the first")
        if not res["same_model"] and _same_file(sec_rec["path"], info["light_curve_file"]["path"]):
            mark_same_model(res, "the second fit is the light-curve file")
        if res["same_model"]:
            log(f"{tag} WARNING: {NOT_TESTED} ({res['same_model_reason']}); slopes are uncorrected")
        stem = f"{FILE_STEM}_{label}" if label else FILE_STEM
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        styles = plot(res, out_dir / f"{stem}.pdf", style_path)
        res["style_file"] = styles[0] if styles else None
        res["sample_selection"] = sample_info
        warn_missing_style(styles, style_path, log, tag)
        with open(out_dir / f"{stem}.json", "w") as fh:
            json.dump(res, fh, indent=1)
        P = res["panels"]["run"]
        kind = "uncorrected slope" if res["same_model"] else "slope"
        log(f"{tag} this run ({res['reduction']}): {kind} {P['slope']:+.3f} ± {P['err']:.3f} ({P['verdict']}); "
            f"lines constant {res['panels']['constant']['slope']:+.2f}, like the disc "
            f"{res['panels']['disc']['slope']:+.2f}; {res['n_quasars']} quasars, {res['n_pairs']} band pairs "
            f"-> {out_dir / (stem + '.pdf')}")
        return res
    except MissingInput as exc:
        log(f"{tag} WARNING: skipped, {exc}. The Hubble fit is not affected.")
    except Exception as exc:  # a diagnostic must never stop a Hubble run
        log(f"{tag} WARNING: skipped after an error ({type(exc).__name__}: {exc}). The Hubble fit is not affected.")
    return None
