#!/usr/bin/env python3
"""
Fit SDSS quasar spectra with jaxqsofit.

This is a simple adaptation of the older fit_spectra.py workflow:

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
import pickle
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

from hubble_utils import match_radec, read_quasars_from_hdf5
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


def compute_photometric_flux_rescale(rec, lam, flux, disable_rescale_flux=False):
    """
    Compute delta_m_avg and sigma_dm from synthetic-vs-fiber SDSS magnitudes.
    """
    out = {
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
        "mean_corrected_u": safe_float(rec.get("mean_corrected_u"), -1e9),
        "mean_corrected_g": safe_float(rec.get("mean_corrected_g"), -1e9),
        "mean_corrected_r": safe_float(rec.get("mean_corrected_r"), -1e9),
        "mean_corrected_i": safe_float(rec.get("mean_corrected_i"), -1e9),
        "mean_corrected_z": safe_float(rec.get("mean_corrected_z"), -1e9),
        "delta_m_avg": 0.0,
        "sigma_dm": 0.0,
        "bands_used": "",
    }
    if disable_rescale_flux:
        return out

    dropped_bands = set(sdss_bands_affected_by_lya(rec["z"]))
    delta_mags = {}
    weights = []
    bands_used = []
    filters = get_sdss_filters()

    for band, filt in zip(SDSS_CAL_BANDS, filters):
        if band in dropped_bands:
            continue

        mag_fiber = safe_float(rec.get(f"mean_corrected_{band}"))
        if not np.isfinite(mag_fiber) or mag_fiber < 0:
            continue

        try:
            mag_synth = float(
                filt.get_ab_magnitude(
                    1e-17 * flux * u.erg / u.s / u.cm**2 / u.AA,
                    lam * u.AA,
                )
            )
        except Exception:
            continue

        out[f"mag_synth_{band}"] = mag_synth
        out[f"mean_corrected_{band}"] = mag_fiber

        dm = mag_fiber - mag_synth
        delta_mags[band] = dm

        sig_m = safe_float(rec.get(f"mean_{band}_err"))
        w = 1.0 / (sig_m**2) if np.isfinite(sig_m) and sig_m > 0 else 1.0
        if np.isfinite(w) and w > 0:
            weights.append(w)
            bands_used.append(band)

    if bands_used:
        dm_arr = np.array([delta_mags[b] for b in bands_used], dtype=float)
        w_arr = np.array(weights, dtype=float)
        mask = np.isfinite(dm_arr) & np.isfinite(w_arr) & (w_arr > 0)
        if np.any(mask):
            w = w_arr[mask]
            dm = dm_arr[mask]
            out["delta_m_avg"] = float(np.sum(w * dm) / np.sum(w))
            out["sigma_dm"] = float(np.sqrt(1.0 / np.sum(w)))

    for band in SDSS_BANDS:
        if band in delta_mags:
            out[f"delta_mag_{band}"] = float(delta_mags[band])
    out["bands_used"] = "".join(bands_used)
    return out


def apply_mc_flux_scaling(flux, flux_err, delta_m_avg, sigma_dm, mc_samples):
    rng = np.random.default_rng(42)
    if mc_samples > 1:
        dm_i = rng.normal(delta_m_avg, sigma_dm)
        s = 10.0 ** (-0.4 * dm_i)
        flux_scaled = s * flux + rng.normal(0.0, s * flux_err, size=flux.shape)
        flux_err_scaled = s * flux_err * np.sqrt(2.0)
    else:
        dm_i = rng.normal(delta_m_avg, 0.0)
        s = 10.0 ** (-0.4 * dm_i)
        flux_scaled = s * flux
        flux_err_scaled = s * flux_err
    return flux_scaled, flux_err_scaled, float(dm_i), float(s)


def compute_apparent_mag_2500_astropy(logL2500, z):
    """Convert log10(lambda L_lambda at 2500A) to monochromatic AB magnitude."""
    c = 2.99792458e10
    lambda_cm = 2500e-8

    dl_cm = COSMO.luminosity_distance(z).to(u.cm).value
    log_lnu = logL2500 + np.log10(lambda_cm / c)
    log_fnu = log_lnu - np.log10(4.0 * np.pi * dl_cm**2)
    return -2.5 * log_fnu - 48.60


def estimate_m2500_from_model(q):
    """Fallback apparent mag estimate from model continuum at rest-frame 2500A."""
    if not hasattr(q, "wave") or not hasattr(q, "f_conti_model"):
        return np.nan

    s = q.numpyro_samples
    pl_norm = np.asarray(s["PL_norm_eff"], dtype=float)
    pl_slope = np.asarray(s["PL_slope"], dtype=float)

    f_lambda_2500 = pl_norm * (2500.0 / 3000.0) ** pl_slope

    c_A_s = 2.99792458e18
    f_nu = (f_lambda_2500 * 1e-17) * (2500.0**2) / c_A_s
    m_2500_samples = -2.5 * np.log10(f_nu) - 48.60

    m50, m_err, m16, m84 = sym_percentile(m_2500_samples)
    return m50, m_err


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

    m2500, m2500_err = estimate_m2500_from_model(q)

    result["apparent_mag_2500"] = m2500
    result["apparent_mag_2500_err"] = m2500_err


# -----------------------------------------------------------------------------
# sample building and cross-match
# -----------------------------------------------------------------------------

def load_quasar_core_list(fpath_in, pickled=False):
    if pickled:
        with open(fpath_in + ".pkl", "rb") as f: 
            quasar_list = pickle.load(f)
    else:
        quasar_list = read_quasars_from_hdf5(fpath_in)
    return quasar_list


def prepare_sample_df(quasar_list, filter_sdss_name=None, filter_object_id=None, N=None, skip=None):
    for q in quasar_list:
        mags_mean = q.get("mags_mean", [])
        if len(mags_mean) == 5:
            for i, band in enumerate(SDSS_BANDS):
                q[f"mags_mean_{band}"] = mags_mean[i]
        for band in SDSS_BANDS:
            mag_mean = safe_float(q.get(f"mags_mean_{band}"))
            if np.isfinite(mag_mean):
                mean_shift = safe_float(q.get(f"mean_{band}"), 0.0)
                q[f"mean_corrected_{band}"] = mag_mean + mean_shift if np.isfinite(mean_shift) else mag_mean

    sample_df = pd.DataFrame.from_records(quasar_list)

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
    quasar_list = load_quasar_core_list(args.fpath_in, pickled=args.pickled)
    print("build_records filtering on: ", args.filter_object_id)
    sample_df = prepare_sample_df(
        quasar_list,
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

def fetch_spectrum_fits(sdss_name, plate, fiber, mjd, cache_dir="data/spectra_cache"):
    from astroquery.sdss import SDSS

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{sdss_name}.fits"

    if cache_file.exists():
        return fits.open(cache_file, memmap=False)

    spec = SDSS.get_spectra(plate=int(plate), fiberID=int(fiber), mjd=int(mjd))
    if spec is None or len(spec) == 0:
        raise RuntimeError(f"No SDSS spectrum found for {sdss_name}")

    hdul = spec[0]
    hdul.writeto(cache_file, overwrite=True)
    return fits.open(cache_file, memmap=False)



def load_spec_from_cache(sdss_name, cache_dir="data/spectra_cache"):
    cache_file = Path(cache_dir) / f"{sdss_name}.fits"
    if cache_file.exists():
        return fits.open(cache_file, memmap=False)
    return None



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
        "delta_m_avg": 0.0,
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
        hdul = load_spec_from_cache(rec["sdss_name"], cache_dir=args.cache_dir)
        if hdul is None:
            hdul = fetch_spectrum_fits(
                rec["sdss_name"],
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

        flux_rescale = compute_photometric_flux_rescale(
            rec,
            lam,
            flux,
            disable_rescale_flux=args.disable_rescale_flux,
        )
        result.update(flux_rescale)

        fit_method = str(args.fit_method).lower()
        numpyro_sample_count = int(args.nuts_samples) if "nuts" in fit_method else 1
        result["numpyro_sample_count"] = int(numpyro_sample_count)

        flux_scaled, err_scaled, dm_i, flux_scale = apply_mc_flux_scaling(
            flux,
            err,
            delta_m_avg=float(flux_rescale["delta_m_avg"]),
            sigma_dm=float(flux_rescale["sigma_dm"]),
            mc_samples=int(numpyro_sample_count),
        )
        result["dm_i"] = dm_i
        result["flux_scale"] = flux_scale

        q = QSOFit(
            lam=lam,
            flux=flux_scaled,
            err=err_scaled,
            z=float(rec["z"]),
            ra=float(rec["ra"]),
            dec=float(rec["dec"]),
            filename=f"z{rec['z']:.3f}_{rec['sdss_name']}",
            output_path=str(args.output_dir),
        )

        prior_config = build_default_prior_config(flux_scaled)

        q.fit(
            name=str(rec["sdss_name"]),
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
            save_result=True,
            save_fits_name=str(rec["sdss_name"]),
            show_plot=False,
            plot_fig=args.save_fig,
            save_fig=args.save_fig,
            verbose=args.verbose,
            kwargs_plot={"save_fig_path": args.fig_dir},
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
                rec["sdss_name"],
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

    # Keep positional args for backward compatibility with older fit_spectra.py calls.
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
    p.add_argument("--pickled", action="store_true", help="Use pickled data file")

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

    p.add_argument("--nproc", type=int, default=1, help="Use spawn multiprocessing when nproc > 1.")
    p.add_argument("--MC_samples", type=int, default=1, help="Deprecated and ignored. Flux-rescale sampling now follows numpyro sample count.")
    p.add_argument("--disable_rescale_flux", "--disable-rescale-flux", dest="disable_rescale_flux", action="store_true", help="Disable magnitude-based flux rescaling.")
    p.set_defaults(save_fig=True)
    p.add_argument("--save-fig", dest="save_fig", action="store_true")
    p.add_argument("--no-save-fig", dest="save_fig", action="store_false")
    p.add_argument("--fig-dir", default="plots/jaxqsofit/", help="Path to save figures")
    p.add_argument("--verbose", action="store_true")

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
