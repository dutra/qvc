"""Observed band-amplitude ratios log10(sigma_b / sigma_r) against redshift, with dilution models.

Diagnostic of the Hubble run, the method of the one-off figure band_amplitude_ratios_literature_dilution_apj
(plots/diagnostics/gold_sample/drop_blue_bands_refit/scripts/s20_band_ratios.py), computed on the run's post-cut
quasars and with the run's own kept fractions as the main curve:

* data ("SDSS S82 excess rms"): per quasar and band, a_b = log10 of the noise-subtracted rms of the SDSS Stripe 82
  light curve over its mean total flux (the amplitude table of the flat-or-tilted test; model-free, no kept-fraction
  division); bands with noise below 15% of the mean flux (relnoise < 0.15). Per quasar r_b = a_b - a_r for
  b = u, g, i where both bands pass. Bins of dz = 0.1 over 0.35-3.15 (edges as in the one-off), at least 15 quasars
  per bin, median with a bootstrap error (1000 resamples, one generator seeded 20261126 and drawn in the order
  u, g, i and increasing redshift, as in the one-off; quasars sorted by object id).
  Small samples: if dz = 0.1 does not give MIN_BINS_TOTAL bins with MIN_BINS_PER_PANEL per panel, dz = 0.2 and then
  0.3 are tried (edges from 0.35); the width used is stated in the figure and the JSON (bin_width).
* model per quasar: r_b = eta log10(lambda_b / lambda_r) + log10(K_b / K_r), lambda = band pivot (the same ratio at
  every redshift), K = varying (kept) fraction of the total band flux; binned as the data (median over the same
  quasars of the bin that have K in both bands). A curve has a model value in a bin only if at least 15 quasars and
  at least 80% of the bin's quasars have K (MIN_MODEL_COVERAGE); otherwise the bin is left out of that curve's fit
  and drawing (the per-bin coverage n_<curve> / n is in the JSON).
* one global eta per curve: weighted least squares on the binned medians of all three panels (weights 1 / bootstrap
  error^2; the model is linear in eta); formal eta error = 1 / sqrt(sum w lever^2), also given scaled by
  sqrt(chi2 / dof) (the scaled one is shown in the legend); chi2 / dof with dof = bins - 1. A fit needs at least
  MIN_BINS_TOTAL bins in total and MIN_BINS_PER_PANEL in each panel; with fewer data bins only the data are drawn
  (warning). The three panels share the r-band amplitude, so their points are correlated (not in the chi2).

Curves (K per quasar and band, from the species table unless noted; every template is clipped to
[f_cont, max(f_AGN, f_cont)], as in the one-off):
  run       "This run": the kept fraction of the run's light curves (psf_agn_fraction_lc_posterior_<b>, the column
            the flat-or-tilted test and the variability curve use); the label names the reduction exactly as those
            two diagnostics print it (qvc.hubble.subtraction_test.reduction_label, same continuum choice). A file
            with the kept fractions but without f_AGN_psf_<b> / f_cont_psf_<b> (e.g. host-only reductions) is still
            used, labelled "line light kept: not recorded"; a file without any kept-fraction column is drawn with
            K = 1 and labelled as having no constant-flux subtraction;
  lines     "Lines only": emission lines constant, Fe II and Balmer continuum varying fully, host constant:
            K = f_cont + f_feii + f_bc;
  full      "Lines + Fe II + BC (constant)": the full subtraction, K = f_cont;
  lit       "Literature responsivities", the L1 / F1 fit set (= curve (a) of the one-off): K = f_cont
            + 0.45 (Lya + other UV + Balmer lines + Mg II) + 0.40 (C IV + C III]) + 0.20 Fe II (UV and optical)
            + 1.0 Balmer continuum; host constant;
  none      "No dilution (power law only)": K = 1 (thin gray dotted reference).

Caveats (also written to the JSON):
* above z ~ 1.6 the spectral fits contain no Balmer-continuum component (a coverage rule of the spectral wrapper;
  f_bc = 0 there in the species table), so the curves with a varying Balmer continuum carry no Balmer-continuum
  dilution there;
* narrow lines and host are outside the varying AGN light and are held constant in every curve (in the species
  table f_AGN + f_host + f_narrow = 1, and the broad line groups sum to f_AGN - f_cont - f_feii - f_bc);
* the three panels share the r-band amplitude and are therefore correlated; the chi2 treats them as independent.

Species table (new input; ``--band-ratio-species-fractions``; default ``subtraction_test/band_species_fractions
.parquet`` under the data-path resolver, then the original table as a fallback):
  long format, one row per object and band: object_id (str), band (u, g, r, i), z, f_AGN, f_cont, f_host, f_narrow,
  f_Lya, f_otherUV, f_CIV, f_CIII, f_MgII, f_Balmer, f_feii_uv, f_feii_opt, f_bc (fractions of the total band flux;
  f_narrow is stored for reference and not required by the reader). The builder records its provenance (source
  file, spec_src, sha256, which spectral fits) in the parquet metadata (key qvc_provenance); the JSON copies it,
  since these fits may differ from the spectra a run subtracted.
  Provenance: plots/diagnostics/gold_sample/responsivity_test/fractions.parquet, rows spec_src == "old" (the nov20
  re-prediction of the aug31 spectral fits, the fits subtracted by the nov21 and later reductions; the table the
  one-off read through cosmology_independent_z_diagnostics/tables/long.parquet). Fe II is split by the band's rest
  effective wavelength (UV if pivot / (1 + z) < 3800 A, optical otherwise; s05_species_sets.py), a band-level split,
  not a spectral one. build_species_table() makes the slim table (python -m qvc.hubble.band_ratio_test --build
  SOURCE OUT [--amplitudes AMPS]); the reader also accepts the wide source table directly (``path#spec_src``).
Coverage: the template curves use the data quasars present in the species table; the JSON records the coverage. Below
MIN_SPECIES_COVERAGE the three template curves are skipped with a warning and only the data, "This run" and the
no-dilution reference are drawn. A missing species table counts as zero coverage (warning, not a skip).

Inputs and failures as in qvc.hubble.subtraction_test: the run's light-curve HDF5 (object_id, z, fit_ok, kept
fractions; f_AGN_psf_<b> / f_cont_psf_<b> or --subtraction-test-first-fit for the label) and the amplitude table
(--subtraction-test-amplitudes). Missing inputs or fewer than MIN_QUASARS quasars with data: one warning, None
returned, nothing raised.
Figure: the look of the collaborator's figure (src/style_colin.mplstyle; three stacked panels, hspace 0.05, black
circles with capped error bars, x label z); the legend goes to the first position in the top panel that covers no
point or curve, otherwise below the figure; the gray notes go below the figure.
Outputs: <out_dir>/band_amplitude_ratios_<label>.pdf and .json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qvc.hubble.subtraction_test import (  # noqa: F401  (REPO_ROOT re-exported for the tests)
    REPO_ROOT, BANDS, LAM_EFF, MissingInput, _norm_id, _split_source, _style, file_record, kept_share,
    note_continuum, read_amplitudes, read_light_curve_fractions, reduction_label, resolve_input, warn_missing_style,
)

RELNOISE_MAX = 0.15
Z_EDGES = np.round(np.arange(0.35, 3.1501, 0.1), 3)
BIN_WIDTHS = (0.1, 0.2, 0.3)    # tried in this order; the first that gives enough bins is used
DEFAULT_STYLE_BR = "src/style_colin.mplstyle"
MIN_PER_BIN = 15
N_BOOT = 1000
SEED = 20261126
MIN_QUASARS = 50
MIN_SPECIES_COVERAGE = 0.5
MIN_MODEL_COVERAGE = 0.8
MIN_BINS_TOTAL = 5
MIN_BINS_PER_PANEL = 2
MISS_NSIGMA = 2.0
RATIO_BANDS = ("u", "g", "i")
REF_BAND = "r"
FEII_UV_MAX_REST = 3800.0
BC_ZMAX = 1.6
FILE_STEM = "band_amplitude_ratios"
DEFAULT_SPECIES = "subtraction_test/band_species_fractions.parquet"
DEFAULT_SPECIES_FALLBACK = "plots/diagnostics/gold_sample/responsivity_test/fractions.parquet#old"
SPECIES_COLS = ("f_AGN", "f_cont", "f_host", "f_narrow", "f_Lya", "f_otherUV", "f_CIV", "f_CIII", "f_MgII",
                "f_Balmer", "f_feii_uv", "f_feii_opt", "f_bc")
# literature responsivities, the L1 / F1 fit set (curve (a) of s20_band_ratios.py)
LITERATURE = dict(f_Lya=0.45, f_otherUV=0.45, f_Balmer=0.45, f_MgII=0.45, f_CIV=0.40, f_CIII=0.40,
                  f_feii_uv=0.20, f_feii_opt=0.20, f_bc=1.0)
CURVES = {
    "run": "This run",
    "lit": "Literature responsivities",
    "lines": "Lines only",
    "full": "Lines + Fe II + BC (constant)",
    "none": "No dilution (power law only)",
}
TEMPLATES = ("lit", "lines", "full")
REQUIRED_SPECIES = tuple(c for c in SPECIES_COLS if c != "f_narrow")
PROVENANCE_KEY = b"qvc_provenance"
# spectral fits behind known source tables (used by the builder when no description is given)
KNOWN_FITS = {("fractions.parquet", "old"): "nov20 re-prediction of the aug31 spectral fits (the fits subtracted by "
                                            "the nov21 and later reductions; responsivity_test/fractions.parquet)"}
CAVEAT = ("Above z ~ 1.6 the spectral fits contain no Balmer-continuum component (a coverage rule of the spectral "
          "wrapper), so the curves with a varying Balmer continuum carry no Balmer-continuum dilution there.")
CAVEAT_SHORT = "No Balmer-continuum component in the spectral fits above z ≈ 1.6"
NARROW_NOTE = "Narrow lines and host are outside the varying AGN light and are held constant in every curve."
CORRELATION_NOTE = ("The three panels share the r-band amplitude and are therefore correlated; the chi2 and the eta "
                    "errors treat the bins as independent.")
LEGEND_NOTE = "BC = Balmer continuum; 'Lines only' = lines constant, Fe II and BC varying"


# ----------------------------------------------------------------------------------------------- inputs
def build_species_table(source, object_ids=None):
    """Slim long table (object_id, band, z, SPECIES_COLS) from a wide spectral-fraction table.

    ``source``: a DataFrame or a parquet path (``path#src`` selects spec_src) with columns object_id, z,
    f_AGN_<b>, f_cont_<b>, f_host_<b>, f_narrow_<b>, f_feii_<b>, f_bc_<b>, f_Lya_<b>, f_otherUV_<b>, f_CIV_<b>,
    f_CIII_<b>, f_MgII_<b>, f_Balmer_<b> (responsivity_test/fractions.parquet). Restricted to ``object_ids``."""
    if isinstance(source, pd.DataFrame):
        f = source
    else:
        path, src = _split_source(source)
        f = pd.read_parquet(resolve_input(path, "species fractions source"))
        if src is not None:
            if "spec_src" not in f:
                raise MissingInput(f"{path} has no spec_src column to select '{src}'")
            f = f[f.spec_src.astype(str) == src]
    f = f.copy()
    f["object_id"] = f.object_id.map(_norm_id)
    f = f.drop_duplicates("object_id", keep="last")
    if object_ids is not None:
        f = f[f.object_id.isin(pd.Index(pd.Series(list(object_ids)).map(_norm_id).unique()))]
    plain = ("AGN", "cont", "host", "narrow", "Lya", "otherUV", "CIV", "CIII", "MgII", "Balmer", "bc")
    miss = [f"f_{k}_{b}" for k in plain + ("feii",) for b in BANDS if f"f_{k}_{b}" not in f]
    if miss or "z" not in f:
        raise MissingInput(f"species fractions source lacks {', '.join((miss or ['z'])[:6])}")
    rows = []
    for b in BANDS:
        z = f.z.to_numpy(float)
        d = {"object_id": f.object_id.to_numpy(), "band": b, "z": z}
        for k in plain:
            d[f"f_{k}"] = f[f"f_{k}_{b}"].to_numpy(float)
        uv = LAM_EFF[b] / (1 + z) < FEII_UV_MAX_REST
        fe = f[f"f_feii_{b}"].to_numpy(float)
        d["f_feii_uv"] = np.where(uv, fe, 0.0)
        d["f_feii_opt"] = np.where(uv, 0.0, fe)
        rows.append(pd.DataFrame(d))
    S = pd.concat(rows, ignore_index=True)
    return S[["object_id", "band", "z", *SPECIES_COLS]]


def read_species(path=None):
    """Species table (slim long format, or the wide source table) and its file record."""
    if path is None:
        spath, ssrc = DEFAULT_SPECIES, None
        fpath, fsrc = _split_source(DEFAULT_SPECIES_FALLBACK)
        p = resolve_input(spath, "species fractions", (fpath,))
        src = ssrc if p.name == Path(spath).name else fsrc
    else:
        spath, src = _split_source(path)
        p = resolve_input(spath, "species fractions")
    f = pd.read_parquet(p)
    if "band" in f:
        if src is not None:
            raise MissingInput(f"{p.name} is a slim species table; '#{src}' applies only to the wide source table")
        miss = [c for c in ("object_id", "band", *REQUIRED_SPECIES) if c not in f]
        if miss:
            raise MissingInput(f"species table {p.name} lacks {', '.join(miss[:6])}")
        S = f.copy()
        S["object_id"] = S.object_id.map(_norm_id)
        S = S.drop_duplicates(["object_id", "band"], keep="last")
    else:
        if src is not None:
            if "spec_src" not in f:
                raise MissingInput(f"{p.name} has no spec_src column to select '{src}'")
            f = f[f.spec_src.astype(str) == src]
        S = build_species_table(f)
    if S.empty:
        raise MissingInput(f"species table {p.name}{'#' + src if src else ''} has no rows")
    rec = file_record(p)
    rec["spec_src"] = src
    rec["provenance"] = table_provenance(p, src, slim="band" in f)
    return S, rec


def table_provenance(p, src=None, slim=True):
    """Which spectral fits a species table comes from: the builder's parquet metadata, or for the wide source table
    its path and spec_src (with the known description when there is one)."""
    if slim:
        try:
            import pyarrow.parquet as pq
            meta = pq.read_schema(p).metadata or {}
        except Exception:  # metadata is information only
            meta = {}
        if PROVENANCE_KEY in meta:
            return json.loads(meta[PROVENANCE_KEY].decode())
        return dict(note="not recorded in the table")
    return dict(source=str(p), spec_src=src, fits=KNOWN_FITS.get((Path(p).name, src), "not recorded"))


def write_species_table(S, out, provenance):
    """Write the slim table with its provenance in the parquet metadata (key qvc_provenance)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pa.Table.from_pandas(S, preserve_index=False)
    meta = dict(t.schema.metadata or {})
    meta[PROVENANCE_KEY] = json.dumps(provenance).encode()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t.replace_schema_metadata(meta), out)


def template_fractions(S):
    """Per object x band K of the three template curves (columns K_lit, K_lines, K_full)."""
    lo = S.f_cont.to_numpy(float)
    hi = np.maximum(S.f_AGN.to_numpy(float), lo)
    clip = lambda raw: np.clip(raw, lo, hi)  # noqa: E731
    lit = S.f_cont.to_numpy(float).copy()
    for c, R in LITERATURE.items():
        lit = lit + R * S[c].to_numpy(float)
    return pd.DataFrame({"object_id": S.object_id.to_numpy(), "band": S.band.to_numpy(),
                         "K_lit": clip(lit),
                         "K_lines": clip(S.f_cont + S.f_feii_uv + S.f_feii_opt + S.f_bc),
                         "K_full": clip(S.f_cont.to_numpy(float))})


# -------------------------------------------------------------------------------------------- estimator
def quasar_ratios(T, amps, object_ids=None):
    """Per quasar: z and r_b = a_b - a_r (NaN where a band fails the noise cut); kept fractions of the run.

    ``T``: light-curve rows (object_id, band, z, f_keep). Only quasars with amplitudes are returned."""
    if object_ids is not None:
        ids = pd.Index(pd.Series(list(object_ids)).map(_norm_id).unique())
        T = T[T.object_id.isin(ids)]
    A = amps[(amps.relnoise < RELNOISE_MAX) & np.isfinite(amps.amp)]
    Y = A.pivot_table(index="object_id", columns="band", values="amp")
    Kr = T.pivot_table(index="object_id", columns="band", values="f_keep", dropna=False)
    Z = T.drop_duplicates("object_id").set_index("object_id").z
    ids = Y.index.intersection(Z.index).sort_values()
    out = pd.DataFrame({"z": Z.reindex(ids).to_numpy()}, index=ids)
    for b in BANDS:
        out[f"a_{b}"] = Y[b].reindex(ids).to_numpy() if b in Y else np.nan
        out[f"K_run_{b}"] = Kr[b].reindex(ids).to_numpy() if b in Kr else np.nan
    for b in RATIO_BANDS:
        out[f"r_{b}"] = out[f"a_{b}"] - out[f"a_{REF_BAND}"]
    out.index.name = "object_id"
    return out


def attach_templates(Qz, S):
    """Add K_<curve>_<b> of the template curves (NaN for quasars without species rows)."""
    K = template_fractions(S)
    for key in TEMPLATES:
        P = K.pivot_table(index="object_id", columns="band", values=f"K_{key}", dropna=False)
        for b in BANDS:
            Qz[f"K_{key}_{b}"] = P[b].reindex(Qz.index).to_numpy() if b in P else np.nan
    return Qz


def bin_data(Qz, curves, edges=Z_EDGES, n_boot=N_BOOT, seed=SEED, min_per_bin=MIN_PER_BIN):
    """Binned medians of r_b with bootstrap errors, and per curve the binned median dilution log10(K_b / K_r) over
    the bin's quasars that have K; NaN unless at least ``min_per_bin`` quasars and MIN_MODEL_COVERAGE of the bin's
    quasars have K (n_<curve> and cov_<curve> = n_<curve> / n record the coverage)."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in RATIO_BANDS:
        d = Qz[np.isfinite(Qz[f"r_{b}"]) & np.isfinite(Qz.z)]
        lever = float(np.log10(LAM_EFF[b] / LAM_EFF[REF_BAND]))
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = d[(d.z >= lo) & (d.z < hi)]
            if len(s) < min_per_bin:
                continue
            x = s[f"r_{b}"].to_numpy()
            bs = np.median(rng.choice(x, (n_boot, len(x))), axis=1)
            r = dict(band=b, z_lo=float(lo), z_hi=float(hi), z_mid=float(0.5 * (lo + hi)), z_med=float(s.z.median()),
                     n=int(len(s)), data_median=float(np.median(x)), data_median_err=float(np.std(bs, ddof=1)),
                     lever=lever)
            for key in curves:
                if key == "none":
                    r[f"dil_{key}"], r[f"n_{key}"], r[f"cov_{key}"] = 0.0, int(len(s)), 1.0
                    continue
                with np.errstate(invalid="ignore", divide="ignore"):
                    kk = np.log10(s[f"K_{key}_{b}"].to_numpy(float) / s[f"K_{key}_{REF_BAND}"].to_numpy(float))
                kk = kk[np.isfinite(kk)]
                cov = kk.size / len(s)
                ok = kk.size >= min_per_bin and cov >= MIN_MODEL_COVERAGE
                r[f"dil_{key}"] = float(np.median(kk)) if ok else float("nan")
                r[f"n_{key}"] = int(kk.size)
                r[f"cov_{key}"] = float(cov)
            rows.append(r)
    return pd.DataFrame(rows)


def z_edges(dz):
    """Bin edges from 0.35 in steps of dz, up to at least 3.15 (dz = 0.1 gives Z_EDGES, as in the one-off)."""
    n = int(np.ceil(round((3.15 - 0.35) / dz, 6)))
    return np.round(0.35 + dz * np.arange(n + 1), 3)


def bins_enough(bands):
    """None if the bins (one band label per bin) allow a fit, else the reason."""
    per = {b: int(sum(1 for x in bands if x == b)) for b in RATIO_BANDS}
    total = sum(per.values())
    if total < MIN_BINS_TOTAL or min(per.values()) < MIN_BINS_PER_PANEL:
        return (f"{total} redshift bins (" + ", ".join(f"{b} {n}" for b, n in per.items())
                + f"); at least {MIN_BINS_TOTAL} in total and {MIN_BINS_PER_PANEL} per panel needed")
    return None


def fit_eta(B, key):
    """Global eta of one curve: weighted least squares on the bins where the curve exists."""
    m = np.isfinite(B[f"dil_{key}"]).to_numpy() & (B.data_median_err > 0).to_numpy()
    why = bins_enough(B.band.to_numpy()[m])
    if why is not None:
        return dict(eta=float("nan"), eta_err=float("nan"), eta_err_scaled=float("nan"), chi2=float("nan"), dof=0,
                    chi2_dof=float("nan"), n_bins=int(m.sum()), fitted=False, note=f"not fitted: {why}"), \
            np.full(len(B), np.nan), np.full(len(B), np.nan)
    y, e, lev, dil = (B[c].to_numpy()[m] for c in ("data_median", "data_median_err", "lever", f"dil_{key}"))
    w = 1 / e ** 2
    eta = float(np.sum(w * lev * (y - dil)) / np.sum(w * lev ** 2))
    eta_err = float(1 / np.sqrt(np.sum(w * lev ** 2)))
    model = eta * B.lever.to_numpy() + B[f"dil_{key}"].to_numpy()
    pull = (B.data_median.to_numpy() - model) / B.data_median_err.to_numpy()
    chi2 = float(np.sum(pull[m] ** 2))
    dof = int(m.sum() - 1)
    return dict(eta=eta, eta_err=eta_err, eta_err_scaled=float(eta_err * np.sqrt(chi2 / dof)), chi2=chi2, dof=dof,
                chi2_dof=chi2 / dof, n_bins=int(m.sum()), fitted=True, note=None), model, pull


def compute(Qz, species_ok=True, n_boot=N_BOOT, seed=SEED):
    """Binned data, curves and fits from the per-quasar table of quasar_ratios (+ attach_templates)."""
    n_data = int(np.isfinite(Qz[[f"r_{b}" for b in RATIO_BANDS]]).any(axis=1).sum())
    if n_data < MIN_QUASARS:
        raise MissingInput(f"only {n_data} quasars have model-free amplitudes in r and another band; "
                           f"at least {MIN_QUASARS} needed")
    curves = ["run"] + (list(TEMPLATES) if species_ok else []) + ["none"]
    tried = []
    for dz in BIN_WIDTHS:
        Bd = bin_data(Qz, curves, edges=z_edges(dz), n_boot=n_boot, seed=seed)
        tried.append((dz, Bd))
        if not Bd.empty and bins_enough(Bd.band.to_numpy()) is None:
            break
    else:   # no width gives enough bins: keep the one with the most bins (narrowest on ties)
        dz, Bd = max(tried, key=lambda t: (len(t[1]), -t[0]))
    B = Bd
    if B.empty:
        raise MissingInput(f"no redshift bin holds {MIN_PER_BIN} quasars (bin widths {', '.join(map(str, BIN_WIDTHS))})")
    data_note = bins_enough(B.band.to_numpy())
    fits = {}
    for key in curves:
        f, model, pull = fit_eta(B, key) if data_note is None else fit_eta(B.iloc[:0], key)
        if data_note is not None:
            model = pull = np.full(len(B), np.nan)
            f["note"] = f"not fitted: {data_note}"
        B[f"model_{key}"] = model
        B[f"pull_{key}"] = pull
        fits[key] = dict(label=CURVES[key], **f)
    drawn = [k for k in curves if fits[k]["fitted"]]
    misses = []
    if fits["run"]["fitted"]:
        miss = B[np.abs(B.pull_run) > MISS_NSIGMA]
        misses = [dict(band=r.band, z_lo=r.z_lo, z_hi=r.z_hi, pull=float(r.pull_run)) for r in miss.itertuples()]
    return dict(n_quasars_data=n_data, n_bins=int(len(B)), bin_width=float(dz), z_edges=z_edges(dz).tolist(),
                bin_widths_tried=[float(t[0]) for t in tried], bins=B.to_dict(orient="records"), curves=fits,
                curves_drawn=drawn, data_only=data_note, this_run_misses_2sigma=misses)


# ----------------------------------------------------------------------------------------------- run file
def read_run_kept(lc_path, first_fit=None, continuum="auto"):
    """The run's per object x band kept fraction (T: object_id, band, z, f_keep, f_AGN, f_cont), info, and the kind:
    "full" (as the flat-or-tilted test reads it), "kept only" (kept fractions without f_AGN / f_cont, e.g. host-only
    reductions) or "unsubtracted" (no kept-fraction column: K = 1)."""
    try:
        T, info = read_light_curve_fractions(lc_path, first_fit, continuum)
        return T, info, "full"
    except MissingInput as exc:
        if first_fit is not None or "lacks" not in str(exc):
            raise
        reason = str(exc)
    import h5py

    p = resolve_input(lc_path, "light-curve file")
    with h5py.File(p, "r") as h:
        keys = set(h.keys())
        if not {"object_id", "z"} <= keys:
            raise MissingInput(f"light-curve file {p.name} lacks object_id or z")
        keep_col = {}
        for b in BANDS:
            for c in (f"psf_agn_fraction_lc_posterior_{b}", f"psf_constant_flux_band_summaries_{b}_agn_fraction"):
                if c in keys:
                    keep_col[b] = c
                    break
        if keep_col and len(keep_col) < len(BANDS):
            raise MissingInput(f"light-curve file {p.name} has kept fractions only for bands "
                               f"{', '.join(sorted(keep_col))}")
        n = len(h["z"])
        d = {"object_id": [_norm_id(v) for v in h["object_id"][()]], "z": np.asarray(h["z"][()], float),
             "fit_ok": np.asarray(h["fit_ok"][()], bool) if "fit_ok" in keys else np.ones(n, bool)}
        for b in BANDS:
            d[f"keep_{b}"] = np.asarray(h[keep_col[b]][()], float) if keep_col else np.ones(n)
        mode = ",".join(np.unique(np.asarray(h["psf_fraction"][()]).astype(str))) if "psf_fraction" in keys else None
    W = pd.DataFrame(d)
    W = W[W.fit_ok].drop_duplicates("object_id", keep="last")
    T = pd.concat([pd.DataFrame({"object_id": W.object_id.to_numpy(), "band": b, "z": W.z.to_numpy(),
                                 "f_keep": W[f"keep_{b}"].to_numpy(), "f_AGN": np.nan, "f_cont": np.nan})
                   for b in BANDS], ignore_index=True)
    kind = "kept only" if keep_col else "unsubtracted"
    info = dict(light_curve_file=file_record(p), psf_fraction=mode, continuum=None,
                first_fit=dict(source="not available", reason=reason),
                kept_fraction_column=(keep_col["g"].replace("_g", "_<b>") if keep_col else None))
    return T, info, kind


def run_label(info, kind, share):
    if kind == "unsubtracted":
        return "no constant-flux subtraction applied: K = 1"
    if kind == "kept only":
        return reduction_label(info, float("nan")) + "; line light kept: not recorded"
    return reduction_label(info, share)


def _clean(o):
    """JSON-safe copy: non-finite floats -> None, numpy scalars and arrays -> Python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


# ------------------------------------------------------------------------------------------------- plot
STYLES = {     # the collaborator's figure: dashed C0, dotted C1, solid C3 (+ dash-dot C2, thin gray dotted)
    "run": dict(color="C3", lw=1.8, ls="-", zorder=3),
    "lit": dict(color="C2", lw=1.5, ls="-.", zorder=2),
    "lines": dict(color="C0", lw=1.5, ls="--", zorder=2),
    "full": dict(color="C1", lw=1.8, ls=":", zorder=3.5),   # above "run": visible when they agree
    "none": dict(color="0.5", lw=0.9, ls=":", zorder=1),
}
SHORT = {"run": "this run", "lit": "literature", "lines": "lines only", "full": "lines + Fe II + BC",
         "none": "no dilution"}
LEGEND_LOCS = ("upper left", "upper right", "lower left", "lower right", "upper center", "lower center",
               "center left", "center right")


def _minus(s):
    return s.replace("-", "−")


def _on_grid(s, col, edges=Z_EDGES):
    """Values of ``col`` on every redshift bin of ``edges`` (NaN where the bin is absent), so lines break there."""
    z = np.round(0.5 * (np.asarray(edges)[:-1] + np.asarray(edges)[1:]), 6)
    v = pd.Series(s[col].to_numpy(float), index=np.round(s.z_mid.to_numpy(float), 6))
    return z, v.reindex(z).to_numpy(float)


def _short_runs(y):
    """Mask of finite points in runs of fewer than 3 consecutive finite values (drawn with markers)."""
    ok = np.isfinite(y)
    out = np.zeros(len(y), bool)
    i = 0
    while i < len(y):
        if ok[i]:
            j = i
            while j < len(y) and ok[j]:
                j += 1
            if j - i < 3:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def short_reduction(res):
    """Short form of the run's reduction for the legend entry."""
    kind = res.get("run_kept_fraction_kind", "full")
    if kind == "unsubtracted":
        return "no subtraction, K = 1"
    mode = res.get("psf_fraction") or "psf fraction n/a"
    share = res.get("kept_line_share_median")
    kept = f"lines kept {share:.0%}" if share is not None and np.isfinite(share) else "lines kept n/r"
    return f"{mode}, {kept}"


def _covered(ax, bbox, pad=2.0):
    """Number of data points and sampled curve points of ``ax`` inside a display-space box."""
    n = 0
    for line in ax.lines:
        xy = np.asarray(line.get_xydata(), float)
        if xy.ndim != 2 or len(xy) == 0:
            continue
        pts = []
        for k in range(len(xy)):
            pts.append(xy[k])
            if k + 1 < len(xy) and np.isfinite(xy[k]).all() and np.isfinite(xy[k + 1]).all() \
                    and line.get_linestyle() not in ("None", "none", ""):
                t = np.linspace(0, 1, 12)[1:-1, None]
                pts.extend(xy[k] + t * (xy[k + 1] - xy[k]))
        P = np.asarray(pts, float)
        P = P[np.isfinite(P).all(axis=1)]
        if not len(P):
            continue
        D = ax.transData.transform(P)
        n += int(((D[:, 0] > bbox.x0 - pad) & (D[:, 0] < bbox.x1 + pad)
                  & (D[:, 1] > bbox.y0 - pad) & (D[:, 1] < bbox.y1 + pad)).sum())
    return n


def plot(res, pdf_path, style_path=None):
    import matplotlib
    import matplotlib.style
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    styles = _style(style_path or DEFAULT_STYLE_BR)
    B = pd.DataFrame(res["bins"])
    C = res["curves"]
    drawn = res["curves_drawn"]
    edges = np.asarray(res.get("z_edges") or Z_EDGES, float)
    order = [k for k in ("lines", "full", "run", "lit", "none") if k in drawn]

    def label(k):
        e = _minus(f"η = {C[k]['eta']:.2f} ± {C[k]['eta_err_scaled']:.2f}")
        if k == "run":
            return f"This run ({short_reduction(res)}), {e}"
        return f"{CURVES[k]}, {e}"

    # Figure + Agg canvas, no pyplot: never opens a window or changes the caller's backend
    with matplotlib.style.context(styles), matplotlib.rc_context({"axes.unicode_minus": True}):
        fig = Figure(figsize=(7.0, 9.0))
        canvas = FigureCanvasAgg(fig)
        try:
            axs = fig.subplots(3, 1, sharex=True, gridspec_kw=dict(hspace=0.05))
            handles = {}
            for ax, b in zip(axs, RATIO_BANDS):
                s = B[B.band == b] if len(B) else B
                if len(s):
                    handles["data"] = ax.errorbar(s.z_mid, s.data_median, yerr=s.data_median_err, fmt="o", ms=4,
                                                  color="k", zorder=4)
                for k in order:
                    z, y = _on_grid(s, f"model_{k}", edges)
                    (h,) = ax.plot(z, y, **STYLES[k])
                    handles[k] = h
                    short = _short_runs(y)
                    if short.any():
                        ax.plot(z[short], y[short], ls="none", marker="o", ms=2.5, color=STYLES[k]["color"],
                                zorder=STYLES[k]["zorder"])
                ax.set_ylabel(rf"$\log_{{10}}(\sigma_{b}/\sigma_r)$")
            if len(B):
                axs[-1].set_xlim(float(B.z_mid.min()) - 0.05, float(B.z_mid.max()) + 0.05)
            axs[-1].set_xlabel(r"$z$")
            keys = (["data"] if "data" in handles else []) + order
            names = {"data": "SDSS S82 excess RMS", **{k: label(k) for k in order}}
            hl = ([handles[k] for k in keys], [names[k] for k in keys])
            # legend: the first position inside the top panel that covers no point or curve; else below the figure
            canvas.draw()
            renderer = canvas.get_renderer()
            leg, legend_place = None, None
            for loc in LEGEND_LOCS:
                leg = axs[0].legend(*hl, fontsize=8, loc=loc)
                canvas.draw()
                if _covered(axs[0], leg.get_window_extent(renderer)) == 0:
                    legend_place = f"top panel, {loc}"
                    break
                leg.remove()
                leg = None
            inv = fig.transFigure.inverted()
            canvas.draw()
            y_next = inv.transform((0, axs[-1].get_tightbbox(renderer).y0))[1] - 0.01
            if leg is None:
                leg = fig.legend(*hl, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, y_next), ncol=1)
                legend_place = "below the figure"
                canvas.draw()
                y_next = inv.transform((0, leg.get_window_extent(renderer).y0))[1] - 0.008
            notes = []
            if order:
                notes.append("χ²/dof: " + ", ".join(f"{SHORT[k]} {C[k]['chi2_dof']:.1f}" for k in order))
            else:
                notes.append("Too few redshift bins for the dilution curves; data only")
            notes.append(f"Redshift bins Δz = {res.get('bin_width', 0.1):.1f}"
                         + (" (widened for a small sample)" if res.get("bin_width", 0.1) > 0.1 + 1e-9 else ""))
            if any(k in drawn for k in TEMPLATES):
                notes.append(LEGEND_NOTE)
            if any(k in drawn for k in ("lit", "lines")):
                notes.append(CAVEAT_SHORT)
            for txt in notes:
                t = fig.text(0.5, y_next, txt, fontsize=8, color="0.4", ha="center", va="top")
                canvas.draw()
                y_next = inv.transform((0, t.get_window_extent(renderer).y0))[1] - 0.009
            res["legend_place"] = legend_place
            Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(pdf_path)
        finally:
            fig.clear()
    return styles


# --------------------------------------------------------------------------------------------- run hook
def run_diagnostic(lc_path, out_dir, object_ids=None, *, amplitudes=None, species=None, first_fit=None,
                   continuum="auto", label="postcut", style_path=None, sample_info=None, log=print):
    """Compute, plot and record; never raises. Returns the result dict, or None when skipped. ``sample_info``
    (the caller's description of the sample: N, redshift range) is stored in the JSON as sample_selection."""
    tag = "[band ratios]"
    try:
        T, info, kind = read_run_kept(lc_path, first_fit, continuum)
        note_continuum(info, log, "[subtraction test / variability curve / band ratios]")
        if kind != "full":
            log(f"{tag} note: {info['first_fit']['reason']}; 'This run' uses "
                + ("the kept fractions only" if kind == "kept only" else "K = 1 (no kept-fraction column)"))
        amps, amps_rec = read_amplitudes(amplitudes)
        if object_ids is not None:
            ids = pd.Index(pd.Series(list(object_ids)).map(_norm_id).unique())
            T = T[T.object_id.isin(ids)]
        Qz = quasar_ratios(T, amps)
        data_ids = Qz.index[np.isfinite(Qz[[f"r_{b}" for b in RATIO_BANDS]]).any(axis=1)]
        try:
            S, sp_rec = read_species(species)
            sp_err = None
        except MissingInput as exc:
            S, sp_rec, sp_err = None, None, str(exc)
        if S is not None:
            Qz = attach_templates(Qz, S)
            have = np.isfinite(Qz.loc[data_ids, [f"K_lit_{b}" for b in BANDS]]).all(axis=1)
            n_cov = int(have.sum())
        else:
            n_cov = 0
        coverage = n_cov / len(data_ids) if len(data_ids) else 0.0
        species_ok = S is not None and coverage >= MIN_SPECIES_COVERAGE
        if not species_ok:
            why = sp_err or (f"species fractions cover {coverage:.0%} of the quasars with data "
                             f"(minimum {MIN_SPECIES_COVERAGE:.0%})")
            log(f"{tag} WARNING: template curves (lines only, lines + Fe II + BC, literature) skipped: {why}; "
                "data, this run and no dilution drawn")
        res = compute(Qz, species_ok)
        if res["data_only"]:
            log(f"{tag} WARNING: only the data are drawn, no curve fitted: {res['data_only']}")
        res.update(info)
        res["run_kept_fraction_kind"] = kind
        share = kept_share(T[T.object_id.isin(data_ids)]) if kind == "full" else float("nan")
        res["kept_line_share_median"] = share
        res["reduction"] = run_label(info, kind, share)
        res["curves"]["run"]["label"] = f"This run ({res['reduction']})"
        res.update(
            amplitudes_file=amps_rec, species_file=sp_rec, species_error=sp_err,
            species_fits=(sp_rec or {}).get("provenance"),
            species_coverage=dict(n_quasars_data=int(len(data_ids)), n_with_species=n_cov, fraction=coverage,
                                  minimum=MIN_SPECIES_COVERAGE, templates_drawn=bool(species_ok)),
            sample=label, n_objects_requested=None if object_ids is None else int(pd.Series(list(object_ids)).nunique()),
            literature_responsivities=LITERATURE, caveats=[CAVEAT, NARROW_NOTE, CORRELATION_NOTE],
            estimator=(f"per-quasar log10(sigma_b/sigma_r) of the SDSS S82 excess rms (relnoise < {RELNOISE_MAX}); "
                       f"bins dz 0.1 (0.2, 0.3 when 0.1 gives too few bins; bin_width), >= {MIN_PER_BIN} quasars, median with {N_BOOT} bootstrap resamples (seed {SEED}); "
                       "model eta log10(lambda_b/lambda_r) + median log10(K_b/K_r) over the same quasars (model "
                       f"only where >= {MIN_PER_BIN} quasars and >= {MIN_MODEL_COVERAGE:.0%} of the bin have K); one "
                       "global eta per curve by weighted least squares over the three panels; dof = bins - 1; "
                       f"fit needs >= {MIN_BINS_TOTAL} bins and >= {MIN_BINS_PER_PANEL} per panel; eta_err formal, "
                       "eta_err_scaled = eta_err sqrt(chi2/dof)"),
            miss_rule=f"|data - this run| > {MISS_NSIGMA:g} x bootstrap error")
        stem = f"{FILE_STEM}_{label}" if label else FILE_STEM
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        styles = plot(res, out_dir / f"{stem}.pdf", style_path)
        res["style_file"] = styles[0] if styles else None
        res["sample_selection"] = sample_info
        warn_missing_style(styles, style_path, log, tag, default=DEFAULT_STYLE_BR)
        res = _clean(res)
        with open(out_dir / f"{stem}.json", "w") as fh:
            json.dump(res, fh, indent=1, allow_nan=False)
        C = res["curves"]
        if C["run"]["fitted"]:
            head = f"this run eta {C['run']['eta']:+.2f}, chi2/dof {C['run']['chi2_dof']:.1f}; " + ", ".join(
                f"{SHORT[k]} {C[k]['chi2_dof']:.1f}" for k in res["curves_drawn"] if k != "run") + \
                f"; {len(res['this_run_misses_2sigma'])} of {res['n_bins']} bins off by > 2 sigma"
        else:
            head = f"this run not fitted ({C['run']['note']})"
        log(f"{tag} {res['reduction']}: {head}; {res['n_quasars_data']} quasars -> {out_dir / (stem + '.pdf')}")
        return res
    except MissingInput as exc:
        log(f"{tag} WARNING: skipped, {exc}. The Hubble fit is not affected.")
    except Exception as exc:  # a diagnostic must never stop a Hubble run
        log(f"{tag} WARNING: skipped after an error ({type(exc).__name__}: {exc}). The Hubble fit is not affected.")
    return None


def _main(argv=None):
    p = argparse.ArgumentParser(description="Build the slim species table of the band-ratio diagnostic.")
    p.add_argument("--build", nargs=2, metavar=("SOURCE", "OUT"), required=True,
                   help="wide source table (path#spec_src, e.g. responsivity_test/fractions.parquet#old) and output")
    p.add_argument("--amplitudes", default=None,
                   help="keep only objects of this amplitude table (default: all objects of the source)")
    p.add_argument("--fits-description", default=None,
                   help="which spectral fits the source holds (default: the known description of the source, if any)")
    a = p.parse_args(argv)
    ids = None
    if a.amplitudes:
        ids = read_amplitudes(a.amplitudes)[0].object_id.unique()
    S = build_species_table(a.build[0], ids)
    path, src = _split_source(a.build[0])
    rp = resolve_input(path, "species fractions source")
    rec = file_record(rp)
    prov = dict(source=rec["path"], source_sha256=rec["sha256"], spec_src=src,
                fits=a.fits_description or KNOWN_FITS.get((rp.name, src), "not recorded"),
                restricted_to_amplitude_table=a.amplitudes, builder="qvc.hubble.band_ratio_test --build")
    write_species_table(S, a.build[1], prov)
    print(f"wrote {a.build[1]}: {S.object_id.nunique()} objects, {len(S)} rows, {Path(a.build[1]).stat().st_size} bytes")


if __name__ == "__main__":
    _main()
