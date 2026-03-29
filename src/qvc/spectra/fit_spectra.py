#!/usr/bin/env python3
"""
Fit SDSS quasar spectra with jaxqsofit.

This module runs the current spectra workflow:

1. read the quasar sample from the input HDF5 file,
2. match objects to DR16Q,
3. download/cache the SDSS spectra,
4. run one jaxqsofit fit per object,
5. save a flat summary CSV plus the native jaxqsofit outputs.

There is no grid of fits and no collect/select stage.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import traceback
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from speclite import filters as speclite_filters
from tqdm import tqdm

num_cores = os.environ.get("NUM_CORES", max((os.cpu_count() or 2) - 2, 1))
try:
    num_cores = int(num_cores)
except ValueError:
    print(f"Invalid NUM_CORES value '{num_cores}', falling back to os.cpu_count().")
    num_cores = os.cpu_count() or 1

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_cores}"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from qvc.hubble.hubble_utils import match_radec, read_quasars_from_hdf5_flat
from jaxqsofit import QSOFit, build_default_prior_config


COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
SDSS_BANDS = ("u", "g", "r", "i", "z")
SDSS_CAL_BANDS = ("u", "g", "r", "i")
_SDSS_FILTER_CACHE = None


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def sym_percentile(x, p=[16, 50, 84], axis=0):
    lower, median, upper = np.percentile(x, p, axis=axis)
    err = 0.5 * (upper - lower)   # optional symmetric equivalent
    err_lower = median - lower
    err_upper = upper - median
    return median, err, err_lower, err_upper

def safe_float(x, default=np.nan):
    try:
        x = np.asarray(x).squeeze()
        return float(x)
    except Exception:
        return default


def coerce_scalar(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (str, bool, int, float)):
        return x
    try:
        arr = np.asarray(x)
        if arr.ndim == 0 or arr.size == 1:
            return arr.reshape(-1)[0].item()
    except Exception:
        pass
    return None


def serialize_any(x):
    if x is None:
        return None
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (str, bool, int, float)):
        return x
    if isinstance(x, dict):
        return {str(k): serialize_any(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [serialize_any(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    try:
        arr = np.asarray(x)
        return arr.tolist()
    except Exception:
        return repr(x)


def sdss_spec_filename(plate: int, mjd: int, fiber: int) -> str:
    return f"spec-{int(plate):04d}-{int(mjd):05d}-{int(fiber):04d}.fits"


def sdss_cache_file_path(cache_dir: str | Path, plate: int, mjd: int, fiber: int) -> Path:
    return Path(cache_dir) / sdss_spec_filename(plate=plate, mjd=mjd, fiber=fiber)


def load_sdss_spec_from_cache(cache_dir: str | Path, plate: int, mjd: int, fiber: int):
    cache_file = sdss_cache_file_path(cache_dir=cache_dir, plate=plate, mjd=mjd, fiber=fiber)
    if cache_file.exists():
        return fits.open(cache_file, memmap=False)
    return None


def sdss_bands_affected_by_lya(z, buffer=0.0):
    """
    SDSS ugriz bands whose rest-frame blue edge is below Ly-alpha.
    """
    edges_obs = {
        "u": (3055.11, 4030.64),
        "g": (3797.64, 5553.04),
        "r": (5418.23, 6994.42),
        "i": (6692.41, 8400.32),
        "z": (7964.70, 10873.33),
    }

    cutoff = 1216.0 + float(buffer)
    affected = []
    for band, (lo_obs, _hi_obs) in edges_obs.items():
        if (lo_obs / (1.0 + float(z))) < cutoff:
            affected.append(band)
    return affected


def get_sdss_filters():
    global _SDSS_FILTER_CACHE
    if _SDSS_FILTER_CACHE is None:
        _SDSS_FILTER_CACHE = speclite_filters.load_filters(*[f"sdss2010-{b}" for b in SDSS_CAL_BANDS])
    return _SDSS_FILTER_CACHE


def build_psf_photometry_inputs(rec):
    """
    Build PSF-photometry inputs for jaxqsofit from mean-corrected multiband values.
    """
    z = safe_float(rec.get("z"))
    dropped_bands = set(sdss_bands_affected_by_lya(z)) if np.isfinite(z) else set()

    psf_bands_all = []
    psf_mags_all = []
    psf_mag_errs_all = []

    for band in SDSS_CAL_BANDS:
        if band in dropped_bands:
            continue

        mag = safe_float(rec.get(f"mean_corrected_{band}"))
        if not np.isfinite(mag):
            continue

        # Prefer posterior mean-band errors from multiband fitting; fall back to LC scatter.
        mag_err = safe_float(rec.get(f"mean_{band}_err"))

        print(f"Band: {band} -- Mags Mean: {rec.get(f'mags_mean_{band}')}, Mean: {rec.get(f'mean_{band}')}, Corrected Mag: {mag}")

        psf_bands_all.append(band)
        psf_mags_all.append(float(mag))
        psf_mag_errs_all.append(float(mag_err))

    return psf_bands_all, psf_mags_all, psf_mag_errs_all


def estimate_m2500_from_model(q):
    """Estimate reddened and intrinsic apparent mags at rest-frame 2500A from PL draws."""
    if not hasattr(q, "numpyro_samples") or q.numpyro_samples is None:
        return np.nan, np.nan, np.nan, np.nan

    s = q.numpyro_samples

    if "PL_norm" not in s or "PL_slope" not in s:
        return np.nan, np.nan, np.nan, np.nan

    pl_norm = np.asarray(s["PL_norm"], dtype=float).reshape(-1)
    pl_slope = np.asarray(s["PL_slope"], dtype=float).reshape(-1)
    if pl_norm.size == 0 or pl_slope.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    n = min(pl_norm.size, pl_slope.size)
    pl_norm = pl_norm[:n]
    pl_slope = pl_slope[:n]

    if "scale_psf" in s:
        scale_psf = np.asarray(s["scale_psf"], dtype=float).reshape(-1)
        if scale_psf.size == 1:
            scale_psf = np.full((n,), float(scale_psf[0]), dtype=float)
        else:
            scale_psf = scale_psf[:n]
    else:
        scale_psf = np.full_like(pl_norm, float(getattr(q, "scale_psf", np.nan)), dtype=float)
        if not np.isfinite(scale_psf).any():
            scale_psf = np.ones_like(pl_norm, dtype=float)

    if "reddening_ebv" in s:
        reddening_ebv = np.asarray(s["reddening_ebv"], dtype=float).reshape(-1)
        if reddening_ebv.size == 1:
            reddening_ebv = np.full((n,), float(reddening_ebv[0]), dtype=float)
        else:
            reddening_ebv = reddening_ebv[:n]
    else:
        reddening_ebv = np.zeros((n,), dtype=float)

    prior_config = getattr(q, "_fit_prior_config", {}) or {}
    pl_pivot = float(np.asarray(prior_config.get("PL_pivot", np.nan), dtype=float))
    if not np.isfinite(pl_pivot) or pl_pivot <= 0.0:
        wave = np.asarray(getattr(q, "wave", []), dtype=float)
        if wave.size > 0 and np.all(np.isfinite(wave)):
            pl_pivot = float(0.5 * (wave[0] + wave[-1]))
        else:
            pl_pivot = 2500.0

    # NumPy equivalent of jaxqsofit.model._smc_like_reddening_jax at 2500A.
    reddening_uv_ref = float(prior_config.get("reddening_uv_ref", 2500.0))
    reddening_alpha = float(prior_config.get("reddening_alpha", 1.2))
    reddening_uv_ref = max(reddening_uv_ref, 1e-8)
    k_lambda_2500 = (2500.0 / reddening_uv_ref) ** (-reddening_alpha)
    reddening_atten_2500 = 10.0 ** (-0.4 * np.maximum(reddening_ebv, 0.0) * k_lambda_2500)

    f_lambda_2500_intrinsic = scale_psf * pl_norm * (2500.0 / pl_pivot) ** pl_slope
    f_lambda_2500_reddened = f_lambda_2500_intrinsic * reddening_atten_2500

    c_A_s = 2.99792458e18
    f_nu_reddened = (f_lambda_2500_reddened * 1e-17) * (2500.0**2) / c_A_s
    f_nu_intrinsic = (f_lambda_2500_intrinsic * 1e-17) * (2500.0**2) / c_A_s

    valid_reddened = np.isfinite(f_nu_reddened) & (f_nu_reddened > 0.0)
    valid_intrinsic = np.isfinite(f_nu_intrinsic) & (f_nu_intrinsic > 0.0)

    if np.any(valid_reddened):
        m_2500_samples = -2.5 * np.log10(f_nu_reddened[valid_reddened]) - 48.60
        m50, m_err, _, _ = sym_percentile(m_2500_samples)
    else:
        m50, m_err = np.nan, np.nan

    if np.any(valid_intrinsic):
        m_2500_intrinsic_samples = -2.5 * np.log10(f_nu_intrinsic[valid_intrinsic]) - 48.60
        m50_intrinsic, m_err_intrinsic, _, _ = sym_percentile(m_2500_intrinsic_samples)
    else:
        m50_intrinsic, m_err_intrinsic = np.nan, np.nan

    return m50, m_err, m50_intrinsic, m_err_intrinsic


def compute_derived_results(result, q, args):
    """
    Populate old fit_spectra-compatible columns from jaxqsofit outputs when possible.
    """
    # result["best"] = True
    # result["decomp_host"] = bool(args.decompose_host)
    # result["BC"] = bool(args.fit_bc)
    # result["poly"] = bool(args.fit_poly)
    # result["npca_qso"] = 10 if args.decompose_host else -1

    # result["redchi2_conti_full"] = safe_float(result.get("chi2_per_pixel"))
    # result["redchi"] = safe_float(result.get("chi2_per_pixel"))

    # if "PL_slope_blue" not in result:
    #     result["PL_slope_blue"] = safe_float(result.get("PL_slope"))
    # if "PL_slope_red" not in result:
    #     result["PL_slope_red"] = safe_float(result.get("PL_slope"))

    if "f_host_2500" not in result:
        result["f_host_2500"] = safe_float(result.get("frac_host_2500"))
    if "f_host_5100" not in result:
        result["f_host_5100"] = safe_float(result.get("frac_host_5100"))

    samples = q.numpyro_samples

    # Host/AGN fraction near spectrum center from posterior samples.
    log_frac_host = np.asarray(samples["log_frac_host"], dtype=float).reshape(-1)
    frac_host_samp = 1.0 / (1.0 + np.exp(-log_frac_host))
    p16, p50, p84 = np.nanpercentile(frac_host_samp, [16.0, 50.0, 84.0])

    m50, m_err, m16, m84 = sym_percentile(frac_host_samp)
    result["f_host_center"] = safe_float(m50)
    result["f_host_center_err"] = safe_float(m_err)

    # BC fraction
    i3000 = np.argmin(np.abs(np.asarray(q.wave) - 3000.0))

    bc_draws = np.asarray(q.pred_out["f_bc_model"], dtype=float)[:, i3000]
    pl_draws = np.asarray(q.pred_out["f_pl_model"], dtype=float)[:, i3000]

    bc_over_pl_draws = bc_draws / pl_draws
    m50, m_err, m16, m84 = sym_percentile(bc_over_pl_draws)
    result["f_bc_over_pl_3000"] = safe_float(m50)
    result["f_bc_over_pl_3000_err"] = safe_float(m_err)

    # FeUV fraction (same definition style as BC fraction)
    i3000 = np.argmin(np.abs(np.asarray(q.wave) - 3000.0))
    fe_uv_draws = np.asarray(q.pred_out["f_fe_mgii_model"], dtype=float)[:, i3000]
    fe_uv_over_pl_draws = fe_uv_draws / pl_draws
    m50, m_err, m16, m84 = sym_percentile(fe_uv_over_pl_draws)
    result["f_fe_uv_over_pl_3000"] = safe_float(m50)
    result["f_fe_uv_over_pl_3000_err"] = safe_float(m_err)

    z = safe_float(result.get("z"))
    m2500 = np.nan
    m2500_err = np.nan
    m2500_intrinsic = np.nan
    m2500_intrinsic_err = np.nan

    m2500, m2500_err, m2500_intrinsic, m2500_intrinsic_err = estimate_m2500_from_model(q)
    print(f"Estimated m2500 from model: {m2500:.3f} +/- {m2500_err:.3f}")
    print(f"Estimated intrinsic m2500 from model: {m2500_intrinsic:.3f} +/- {m2500_intrinsic_err:.3f}")

    result["apparent_mag_2500"] = m2500
    result["apparent_mag_2500_err"] = m2500_err
    result["apparent_mag_2500_intrinsic"] = m2500_intrinsic
    result["apparent_mag_2500_intrinsic_err"] = m2500_intrinsic_err


# -----------------------------------------------------------------------------
# sample building and cross-match
# -----------------------------------------------------------------------------

def load_quasar_core_list(fpath_in):
    return read_quasars_from_hdf5_flat(fpath_in)


def prepare_sample_df(sample_df, filter_sdss_name=None, filter_object_id=None, N=None, skip=None):
    sample_df = sample_df.copy()
    for band in SDSS_BANDS:
        mag_col = f"mags_mean_{band}"
        if mag_col not in sample_df.columns:
            continue
        mag_mean = pd.to_numeric(sample_df[mag_col], errors="coerce")
        mean_col = f"mean_{band}"
        if mean_col in sample_df.columns:
            mean_shift = pd.to_numeric(sample_df[mean_col], errors="coerce").fillna(0.0)
        else:
            mean_shift = 0.0
        sample_df[f"mean_corrected_{band}"] = mag_mean + mean_shift

    exclusion_sdss_names = {
        "221120.38+010905.6",  # wrong redshift
        "235133.07+005537.0",  # wrong redshift
        "024555.35+005332.6",  # weird spectrum
    }
    if "sdss_name" in sample_df.columns:
        sample_df = sample_df[~sample_df["sdss_name"].astype(str).isin(exclusion_sdss_names)]

    if filter_sdss_name is not None:
        sample_df = sample_df[sample_df["sdss_name"].astype(str).isin(filter_sdss_name)]
    if filter_object_id is not None:
        ids = [str(x) for x in filter_object_id]
        sample_df = sample_df[sample_df["object_id"].astype(str).isin(ids)]

    if skip is not None:
        sample_df = sample_df.iloc[int(skip):]
    if N is not None:
        sample_df = sample_df.iloc[: int(N)]

    sample_df = sample_df.reset_index(drop=True)
    sample_df["object_id"] = sample_df["object_id"].astype(str).str.strip()
    return sample_df



def match_to_dr16q(sample_df, dr16q_fits, max_sep_arcsec=1.0):
    cols = ["RA", "DEC", "SDSS_NAME", "PLATE", "FIBERID", "MJD", "Z_SYS", "LOGLBOL"]
    data_cat = Table.read(dr16q_fits, hdu=1)[cols].to_pandas()
    df_matched, unmatched = match_radec(
        sample_df,
        data_cat,
        populate_cols=["SDSS_NAME", "PLATE", "FIBERID", "MJD", "Z_SYS", "LOGLBOL", "RA", "DEC"],
        ra_col_a="ra",
        dec_col_a="dec",
        ra_col_b="RA",
        dec_col_b="DEC",
        max_sep_arcsec=max_sep_arcsec,
        add_prefix=False,
    )

    df_matched = df_matched.copy()
    df_matched["plate"] = df_matched["PLATE"].astype(int)
    df_matched["fiber"] = df_matched["FIBERID"].astype(int)
    df_matched["mjd"] = df_matched["MJD"].astype(int)
    df_matched["z"] = df_matched["Z_SYS"].astype(float)
    df_matched["loglbol"] = df_matched["LOGLBOL"].astype(float)
    df_matched["sdss_name"] = df_matched["SDSS_NAME"].astype(str).str.strip()
    df_matched["object_id"] = df_matched["object_id"].astype(str).str.strip()
    df_matched["ra"] = df_matched["RA"].astype(float)
    df_matched["dec"] = df_matched["DEC"].astype(float)

    print(f"Matched {len(df_matched)} objects to DR16Q. Unmatched: {len(unmatched)}")
    return df_matched



def build_records(args):
    if str(args.fpath_in).lower().endswith(".csv"):
        sample_df = pd.read_csv(args.fpath_in)
    else:
        sample_df = load_quasar_core_list(args.fpath_in)

    print("build_records filtering on: ", args.filter_object_id)
    sample_df = prepare_sample_df(
        sample_df,
        filter_sdss_name=args.filter_sdss_name,
        filter_object_id=args.filter_object_id,
        N=args.N,
        skip=args.skip,
    )
    print(f"Sample size after filtering: {len(sample_df)}")
    if len(sample_df) > 0:
        print("First few filtered object_ids:", sample_df["object_id"].head().tolist())

    df_matched = match_to_dr16q(sample_df, args.dr16q_fits, args.max_sep)
    return [row.to_dict() for _, row in df_matched.iterrows()]


def fetch_dustmaps(args):
    import dustmaps.sfd
    dustmaps.sfd.fetch()

# -----------------------------------------------------------------------------
# SDSS spectrum cache
# -----------------------------------------------------------------------------

def fetch_spectrum_fits(args, plate, fiber, mjd, cache_dir="data/spectra_cache"):
    from astroquery.sdss import SDSS

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = sdss_cache_file_path(
        cache_dir=cache_dir,
        plate=int(plate),
        mjd=int(mjd),
        fiber=int(fiber),
    )
    spec_file = sdss_spec_filename(plate=int(plate), mjd=int(mjd), fiber=int(fiber))

    if cache_file.exists():
        return fits.open(cache_file, memmap=False)
    elif args.mode != "download":
        raise RuntimeError(f"Cache file {cache_file} ({spec_file}) not found in non-download mode.")

    spec = SDSS.get_spectra(plate=int(plate), fiberID=int(fiber), mjd=int(mjd))
    if spec is None or len(spec) == 0:
        raise RuntimeError(f"No SDSS spectrum found for {spec_file}")

    hdul = spec[0]
    hdul.writeto(cache_file, overwrite=True)
    return fits.open(cache_file, memmap=False)



def load_spec_from_cache(plate, fiber, mjd, cache_dir="data/spectra_cache"):
    return load_sdss_spec_from_cache(
        cache_dir=cache_dir,
        plate=int(plate),
        mjd=int(mjd),
        fiber=int(fiber),
    )



def get_spectrum_arrays(hdul):
    tb = hdul[1].data
    lam = np.asarray(10 ** tb["loglam"], dtype=float)
    flux = np.asarray(tb["flux"], dtype=float)

    ivar = np.asarray(tb["ivar"], dtype=float)
    err = np.full_like(flux, np.nan, dtype=float)
    good_ivar = np.isfinite(ivar) & (ivar > 0)
    err[good_ivar] = 1.0 / np.sqrt(ivar[good_ivar])

    good = np.isfinite(lam) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    return lam[good], flux[good], err[good]


# -----------------------------------------------------------------------------
# saving
# -----------------------------------------------------------------------------


def extract_named_results(q):
    out = {}

    if hasattr(q, "conti_result_name") and hasattr(q, "conti_result"):
        for name, value in zip(q.conti_result_name, q.conti_result):
            out[str(name)] = coerce_scalar(value)

    if hasattr(q, "line_result_name") and hasattr(q, "line_result"):
        for name, value in zip(q.line_result_name, q.line_result):
            out[str(name)] = coerce_scalar(value)

    return out



def extract_scalar_attrs(q):
    out = {}
    for key, value in q.__dict__.items():
        if key.startswith("_"):
            continue
        if key in out:
            continue
        scalar = coerce_scalar(value)
        if scalar is not None:
            out[str(key)] = scalar
    return out



def extract_fit_stats(q):
    out = {
        "chi2": np.nan,
        "chi2_per_pixel": np.nan,
        "wrms": np.nan,
        "n_pixels": 0,
        "wave_min_rf": np.nan,
        "wave_max_rf": np.nan,
    }

    resid = np.asarray(q.flux) - np.asarray(q.model_total)
    sigma = np.asarray(q.err)
    mask = np.asarray(q.wave, dtype=float) >= 1215.67
    resid = resid[mask]

    s = q.numpyro_samples
    frac_j = safe_float(np.median(np.asarray(s.get("frac_jitter", 0.0))), 0.0)
    add_j = safe_float(np.median(np.asarray(s.get("add_jitter", 0.0))), 0.0)
    sigma = np.sqrt(sigma**2 + (frac_j * np.abs(np.asarray(q.model_total))) ** 2 + add_j**2)

    good = np.isfinite(resid) & np.isfinite(sigma) & (sigma > 0)

    z = resid[good] / sigma[good]
    out["wrms"] = float(np.sqrt(np.mean(z**2)))

    out["chi2"] = float(np.sum(z**2))
    out["chi2_per_pixel"] = float(np.mean(z**2))
    out["n_pixels"] = int(np.sum(good))

    out["wave_min_rf"] = safe_float(np.min(q.wave))
    out["wave_max_rf"] = safe_float(np.max(q.wave))

    return out


# -----------------------------------------------------------------------------
# fitting
# -----------------------------------------------------------------------------

def run_one_fit(rec, args):
    result = {
        "object_id": str(rec["object_id"]),
        "sdss_name": str(rec["sdss_name"]),
        "plate": int(rec["plate"]),
        "fiber": int(rec["fiber"]),
        "mjd": int(rec["mjd"]),
        "z": float(rec["z"]),
        "ra": float(rec["ra"]),
        "dec": float(rec["dec"]),
        "loglbol": safe_float(rec.get("loglbol")),
        "fit_ok": False,
        "error_message": "",
        "delta_mag_u": -1e9,
        "delta_mag_g": -1e9,
        "delta_mag_r": -1e9,
        "delta_mag_i": -1e9,
        "delta_mag_z": -1e9,
        "mag_synth_u": -1e9,
        "mag_synth_g": -1e9,
        "mag_synth_r": -1e9,
        "mag_synth_i": -1e9,
        "mag_synth_z": -1e9,
        "mean_corrected_u": -1e9,
        "mean_corrected_g": -1e9,
        "mean_corrected_r": -1e9,
        "mean_corrected_i": -1e9,
        "mean_corrected_z": -1e9,
        "delta_m_flux_recal": 0.0,
        "sigma_dm": 0.0,
        "dm_i": 0.0,
        "flux_scale": 1.0,
        "bands_used": "",
        "numpyro_sample_count": 0,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    result["result_dir"] = args.output_dir
    os.makedirs(args.fig_dir, exist_ok=True)
    result["fig_dir"] = args.fig_dir if args.save_fig else None

    try:
        hdul = load_spec_from_cache(
            rec["plate"],
            rec["fiber"],
            rec["mjd"],
            cache_dir=args.cache_dir,
        )
        if hdul is None:
            hdul = fetch_spectrum_fits(
                args,
                rec["plate"],
                rec["fiber"],
                rec["mjd"],
                cache_dir=args.cache_dir,
            )

        try:
            lam, flux, err = get_spectrum_arrays(hdul)
        finally:
            hdul.close()
        if len(lam) == 0:
            raise RuntimeError("Spectrum has no good pixels after ivar filtering.")

        q = QSOFit(
            lam=lam,
            flux=flux,
            err=err,
            z=float(rec["z"]),
            ra=float(rec["ra"]),
            dec=float(rec["dec"]),
            filename=f"z{rec['z']:.3f}_{rec['sdss_name']}",
            output_path=str(args.output_dir),
        )

        prior_config = build_default_prior_config(flux)
        psf_bands_all, psf_mags_all, psf_mag_errs_all = build_psf_photometry_inputs(rec)
        result["bands_used"] = "".join(psf_bands_all)
        for band in SDSS_BANDS:
            mag = safe_float(rec.get(f"mean_corrected_{band}"))
            if np.isfinite(mag):
                result[f"mean_corrected_{band}"] = float(mag)

        if args.resume:
            q = QSOFit.load_from_samples(
                filename=f"z{rec['z']:.3f}_{rec['sdss_name']}",  # important: matches fit(name=...)
                output_path=str(args.output_dir),
                kwargs_plot={"show_plot": False},
                plot_diagnostics=False,
            )
        else:
            q.fit(
                name=f"z{rec['z']:.3f}_{rec['sdss_name']}",
                fit_poly_edge_flex=args.fit_poly_edge_flex,
                deredden=not args.no_deredden,
                wave_range=(args.wave_min, args.wave_max),
                fit_lines=args.fit_lines,
                decompose_host=args.decompose_host,
                fit_pl=args.fit_pl,
                fit_fe=args.fit_fe,
                fit_bc=args.fit_bc,
                fit_poly=args.fit_poly,
                mask_lya_forest=args.mask_lya_forest,
                fit_method=args.fit_method,
                prior_config=prior_config,
                dsps_ssp_fn=args.dsps_ssp_fn,
                nuts_warmup=args.nuts_warmup,
                nuts_samples=args.nuts_samples,
                nuts_chains=args.nuts_chains,
                nuts_target_accept=args.nuts_target_accept,
                optax_steps=args.optax_steps,
                optax_lr=args.optax_lr,
                save_result=args.save_jaxqsofit_samples,
                save_fits_name=f"z{rec['z']:.3f}_{rec['sdss_name']}",
                show_plot=False,
                plot_fig=args.save_fig,
                save_fig=args.save_fig,
                verbose=args.verbose,
                kwargs_plot={"save_fig_path": args.fig_dir, 'plot_residual': args.plot_residual},
                psf_mags=psf_mags_all,
                psf_mag_errs=psf_mag_errs_all,
                psf_bands=psf_bands_all,
                use_psf_phot=True,
            )

        result.update(extract_named_results(q))
        result.update(extract_scalar_attrs(q))
        result.update(extract_fit_stats(q))
        compute_derived_results(result, q, args)
        result["fit_ok"] = True

        return result

    except Exception as exc:
        result["error_message"] = str(exc)
        if args.verbose:
            traceback.print_exc()
        return result


# -----------------------------------------------------------------------------
# runners
# -----------------------------------------------------------------------------

def run_download(args):
    records = build_records(args)
    errors = []

    for rec in tqdm(records, desc="Downloading spectra"):
        try:
            hdul = fetch_spectrum_fits(
                args,
                rec["plate"],
                rec["fiber"],
                rec["mjd"],
                cache_dir=args.cache_dir,
            )
            hdul.close()
        except Exception as exc:
            errors.append((rec["sdss_name"], str(exc)))

    print(f"Tried to download {len(records)} spectra. Errors: {len(errors)}")
    if errors:
        for name, msg in errors[:10]:
            print(f"  {name}: {msg}")



def run_fit(args):
    records = build_records(args)
    if len(records) == 0:
        raise RuntimeError("No records to process.")

    worker = partial(run_one_fit, args=args)

    if int(args.nproc) <= 1:
        rows = [worker(rec) for rec in tqdm(records, desc="Fitting spectra")]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=int(args.nproc)) as pool:
            rows = list(tqdm(pool.imap(worker, records), total=len(records), desc="Fitting spectra"))

    df = pd.DataFrame(rows)
    Path(args.fpath_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.fpath_out, index=False)
    print(f"Wrote {len(df)} rows to {args.fpath_out}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fit SDSS spectra with jaxqsofit.")

    # Preserve the existing positional CLI interface for batch scripts.
    p.add_argument("fpath_in", nargs="?", help="Input HDF5 quasar catalog.")
    p.add_argument("fpath_out", nargs="?", help="Output CSV with one row per fitted object.")
    p.add_argument("--fpath-in", dest="fpath_in_opt", default=None, help="Input HDF5 quasar catalog.")
    p.add_argument("--fpath-out", dest="fpath_out_opt", default=None, help="Output CSV with one row per fitted object.")
    p.add_argument("--mode", choices=["download", "fit", "fetch-dustmaps"], required=True)

    p.add_argument("--dr16q-fits", default="data/dr16q_prop_May01_2024.fits")
    p.add_argument("--cache-dir", default="data/spectra_cache")
    p.add_argument("--output-dir", default="results/jaxqsofit")
    p.add_argument("--max-sep", type=float, default=1.0, help="Cross-match radius in arcsec.")
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--skip", type=int, default=None)
    p.add_argument("--filter_sdss_name", nargs="+", default=None)
    p.add_argument("--filter_object_id", nargs="+", default=None)
    p.add_argument("--fit-method", choices=["optax", "nuts", "optax+nuts"], default="optax+nuts")
    p.add_argument("--dsps-ssp-fn", default="data/ssp_data_fsps_v3.2_lgmet_age.h5", help="Path to the DSPS SSP HDF5 file.")
    p.add_argument("--no-deredden", action="store_true")

    p.add_argument("--wave-min", type=float, default=1250.0, help="Rest-frame minimum wavelength.")
    p.add_argument("--wave-max", type=float, default=8000.0, help="Rest-frame maximum wavelength.")

    p.add_argument("--optax-steps", type=int, default=600)
    p.add_argument("--optax-lr", type=float, default=1e-2)
    p.add_argument("--nuts-warmup", type=int, default=50)
    p.add_argument("--nuts-samples", type=int, default=50)
    p.add_argument("--nuts-chains", type=int, default=1)
    p.add_argument("--nuts-target-accept", type=float, default=0.9)

    p.set_defaults(fit_lines=True)
    p.add_argument("--fit-lines", dest="fit_lines", action="store_true")
    p.add_argument("--no-fit-lines", dest="fit_lines", action="store_false")

    p.set_defaults(decompose_host=True)
    p.add_argument("--decompose-host", dest="decompose_host", action="store_true")
    p.add_argument("--no-decompose-host", dest="decompose_host", action="store_false")

    p.set_defaults(fit_pl=True)
    p.add_argument("--fit-pl", dest="fit_pl", action="store_true")
    p.add_argument("--no-fit-pl", dest="fit_pl", action="store_false")

    p.set_defaults(fit_fe=True)
    p.add_argument("--fit-fe", dest="fit_fe", action="store_true")
    p.add_argument("--no-fit-fe", dest="fit_fe", action="store_false")

    p.set_defaults(fit_bc=True)
    p.add_argument("--fit-bc", dest="fit_bc", action="store_true")
    p.add_argument("--no-fit-bc", dest="fit_bc", action="store_false")

    p.set_defaults(fit_poly=True)
    p.add_argument("--fit-poly", dest="fit_poly", action="store_true")
    p.add_argument("--no-fit-poly", dest="fit_poly", action="store_false")

    p.set_defaults(mask_lya_forest=True)
    p.add_argument("--mask-lya-forest", dest="mask_lya_forest", action="store_true")
    p.add_argument("--no-mask-lya-forest", dest="mask_lya_forest", action="store_false")

    p.set_defaults(fit_poly_edge_flex=True)
    p.add_argument("--fit-poly-edge-flex", dest="fit_poly_edge_flex", action="store_true")
    p.add_argument("--no-fit-poly-edge-flex", dest="fit_poly_edge_flex", action="store_false")

    p.set_defaults(save_jaxqsofit_samples=True)
    p.add_argument("--save-jaxqsofit-samples", dest="save_jaxqsofit_samples", action="store_true")
    p.add_argument("--no-save-jaxqsofit-samples", dest="save_jaxqsofit_samples", action="store_false")
    

    p.add_argument("--nproc", type=int, default=1, help="Use spawn multiprocessing when nproc > 1.")
    p.add_argument("--plot-residual", dest="plot_residual", action="store_true", default=False, help="Plot residuals in fit figures.")
    p.add_argument("--disable_rescale_flux", "--disable-rescale-flux", dest="disable_rescale_flux", action="store_true", help="Disable magnitude-based flux rescaling.")
    p.set_defaults(save_fig=True)
    p.add_argument("--save-fig", dest="save_fig", action="store_true")
    p.add_argument("--no-save-fig", dest="save_fig", action="store_false")
    p.add_argument("--fig-dir", default="plots/jaxqsofit/", help="Path to save figures")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume mode: load the saved samples from jaxqsofit.")
    p.add_argument("--dustmaps-data-dir", default="results/dustmaps", help="Directory to store dustmaps data (used for fetch-dustmaps mode)")
    args = p.parse_args()

    # Resolve optional aliases first.
    if args.fpath_in is None and args.fpath_in_opt is not None:
        args.fpath_in = args.fpath_in_opt
    if args.fpath_out is None and args.fpath_out_opt is not None:
        args.fpath_out = args.fpath_out_opt

    # fit and download need the input catalog; fit also needs output CSV.
    if args.mode in {"fit", "download"} and not args.fpath_in:
        p.error("fpath_in is required for --mode fit/download.")
    if args.mode == "fit" and not args.fpath_out:
        p.error("fpath_out is required for --mode fit.")

    return args



def main():
    args = parse_args()

    # Set dustmaps location only when it will be used.
    if args.mode == "fetch-dustmaps" or (args.mode == "fit" and not args.no_deredden):
        from dustmaps.config import config
        config["data_dir"] = args.dustmaps_data_dir

    if args.mode == "download":
        run_download(args)
    elif args.mode == "fit":
        run_fit(args)
    elif args.mode == "fetch-dustmaps":
        fetch_dustmaps(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
