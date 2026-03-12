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
import json
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
from astroquery.sdss import SDSS
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


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

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

    wave_rf = np.asarray(q.wave, dtype=float)
    cont_rf = np.asarray(q.f_conti_model, dtype=float)
    if wave_rf.size < 2 or cont_rf.size < 2:
        return np.nan
    if not (np.nanmin(wave_rf) <= 2500.0 <= np.nanmax(wave_rf)):
        return np.nan

    f2500_rf = np.interp(2500.0, wave_rf, cont_rf)
    z = safe_float(getattr(q, "z", np.nan))
    if not np.isfinite(f2500_rf) or f2500_rf <= 0 or not np.isfinite(z):
        return np.nan

    # jaxqsofit uses SDSS-style 1e-17 flux units.
    f_lambda_obs = (f2500_rf / (1.0 + z)) * 1e-17
    lambda_obs_cm = 2500.0 * (1.0 + z) * 1e-8
    c = 2.99792458e10
    f_nu = f_lambda_obs * (lambda_obs_cm**2) / c
    if not np.isfinite(f_nu) or f_nu <= 0:
        return np.nan
    return -2.5 * np.log10(f_nu) - 48.60


def add_legacy_aliases(result, q, args):
    """
    Populate old fit_spectra-compatible columns from jaxqsofit outputs when possible.
    """
    result["best"] = True
    result["decomp_host"] = bool(args.decompose_host)
    result["BC"] = bool(args.fit_bc)
    result["poly"] = bool(args.fit_poly)
    result["npca_qso"] = 10 if args.decompose_host else -1

    result["redchi2_conti_full"] = safe_float(result.get("chi2_per_pixel"))
    result["redchi"] = safe_float(result.get("chi2_per_pixel"))

    if "PL_slope_blue" not in result:
        result["PL_slope_blue"] = safe_float(result.get("PL_slope"))
    if "PL_slope_red" not in result:
        result["PL_slope_red"] = safe_float(result.get("PL_slope"))

    if "f_host_2500" not in result:
        result["f_host_2500"] = safe_float(result.get("frac_host_2500"))
    if "f_host_5100" not in result:
        result["f_host_5100"] = safe_float(result.get("frac_host_5100"))

    z = safe_float(result.get("z"))
    m2500 = np.nan
    m2500_err = np.nan
    m2500_red = np.nan
    m2500_red_err = np.nan

    m2500 = estimate_m2500_from_model(q)

    sn = safe_float(result.get("SN_ratio_conti"))
    if np.isfinite(sn) and sn > 0:
        m2500_err = 1.0857 / sn

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
            for i, band in enumerate(["u", "g", "r", "i", "z"]):
                q[f"mags_mean_{band}"] = mags_mean[i]

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
    sample_df = prepare_sample_df(
        quasar_list,
        filter_sdss_name=args.filter_sdss_name,
        filter_object_id=args.filter_object_id,
        N=args.N,
        skip=args.skip,
    )
    df_matched = match_to_dr16q(sample_df, args.dr16q_fits, args.max_sep)
    return [row.to_dict() for _, row in df_matched.iterrows()]


def fetch_dustmaps(args):
    import dustmaps.sfd
    dustmaps.sfd.fetch()

# -----------------------------------------------------------------------------
# SDSS spectrum cache
# -----------------------------------------------------------------------------

def fetch_spectrum_fits(sdss_name, plate, fiber, mjd, cache_dir="data/spectra_cache"):
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

def make_object_dir(args, rec):
    outdir = Path(args.output_dir) / str(rec["sdss_name"])
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir



def save_full_result_json(q, rec, outdir):
    payload = {
        "object_id": str(rec["object_id"]),
        "sdss_name": str(rec["sdss_name"]),
        "plate": int(rec["plate"]),
        "fiber": int(rec["fiber"]),
        "mjd": int(rec["mjd"]),
        "z": float(rec["z"]),
        "ra": float(rec["ra"]),
        "dec": float(rec["dec"]),
        "attrs": {},
    }

    for key, value in q.__dict__.items():
        if key.startswith("_"):
            continue
        payload["attrs"][str(key)] = serialize_any(value)

    fpath = outdir / "full_result.json"
    with open(fpath, "w") as f:
        json.dump(payload, f)
    return fpath



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

    if not hasattr(q, "model_total"):
        return out

    resid = np.asarray(q.flux) - np.asarray(q.model_total)
    sigma = np.asarray(q.err)

    if getattr(q, "numpyro_samples", None) is not None:
        s = q.numpyro_samples
        frac_j = safe_float(np.median(np.asarray(s.get("frac_jitter", 0.0))), 0.0)
        add_j = safe_float(np.median(np.asarray(s.get("add_jitter", 0.0))), 0.0)
        sigma = np.sqrt(sigma**2 + (frac_j * np.abs(np.asarray(q.model_total))) ** 2 + add_j**2)

    good = np.isfinite(resid) & np.isfinite(sigma) & (sigma > 0)
    if np.any(good):
        z = resid[good] / sigma[good]
        out["chi2"] = float(np.sum(z**2))
        out["chi2_per_pixel"] = float(np.mean(z**2))
        out["wrms"] = float(np.sqrt(np.mean(z**2)))
        out["n_pixels"] = int(np.sum(good))

    if hasattr(q, "wave"):
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
    }

    outdir = make_object_dir(args, rec)
    result["result_dir"] = str(outdir)

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

        q = QSOFit(
            lam=lam,
            flux=flux,
            err=err,
            z=float(rec["z"]),
            ra=float(rec["ra"]),
            dec=float(rec["dec"]),
            plateid=int(rec["plate"]),
            mjd=int(rec["mjd"]),
            fiberid=int(rec["fiber"]),
            path=str(outdir),
        )

        prior_config = build_default_prior_config(flux)

        q.fit(
            name=str(rec["sdss_name"]),
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
            plot_fig=args.plot_fig or args.save_fig,
            save_fig=args.save_fig,
            verbose=args.verbose,
        )

        result.update(extract_named_results(q))
        result.update(extract_scalar_attrs(q))
        result.update(extract_fit_stats(q))
        add_legacy_aliases(result, q, args)
        result["fit_ok"] = True

        if args.save_full_json:
            json_path = save_full_result_json(q, rec, outdir)
            result["full_result_json"] = str(json_path)
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

    p.set_defaults(fit_bc=False)
    p.add_argument("--fit-bc", dest="fit_bc", action="store_true")
    p.add_argument("--no-fit-bc", dest="fit_bc", action="store_false")

    p.set_defaults(fit_poly=True)
    p.add_argument("--fit-poly", dest="fit_poly", action="store_true")
    p.add_argument("--no-fit-poly", dest="fit_poly", action="store_false")

    p.set_defaults(mask_lya_forest=True)
    p.add_argument("--mask-lya-forest", dest="mask_lya_forest", action="store_true")
    p.add_argument("--no-mask-lya-forest", dest="mask_lya_forest", action="store_false")

    p.add_argument("--nproc", type=int, default=1, help="Use spawn multiprocessing when nproc > 1.")
    p.add_argument("--plot-fig", action="store_true")
    p.add_argument("--save-fig", action="store_true")
    p.add_argument("--save-full-json", action="store_true", help="Save full q.__dict__ JSON per object (large files).")
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
