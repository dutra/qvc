"""Shared utility functions for the QVC/Hubble fitting workflow."""

import json
import math
import os
import pickle
import warnings
from ast import literal_eval
from collections import defaultdict
from itertools import combinations
from statistics import NormalDist
from typing import Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import FlatwCDM, Flatw0waCDM, FlatLambdaCDM, FlatwpwaCDM
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.votable import parse
from scipy.linalg import cho_factor
from scipy import stats
from tqdm import tqdm

from hubble_cut_config import (
    DEFAULT_F_HOST_CUT,
    build_agn_cuts,
    build_log_amp_delta_blr_cuts,
)
from hubble_model import (
    M_model_agn,
    M_model_agn_err,
    agn_model_oidx,
    agn_model_pack_obs,
    agn_model_pack_params,
    get_model_params,
)

def convert_M2500_to_logL2500(M2500):
    return -1/2.5 * (M2500 - 90.0)

def sym_percentile(x, p=[16, 50, 84], axis=0):
    lower, median, upper = np.percentile(x, p, axis=axis)
    err = 0.5 * (upper - lower)   # optional symmetric equivalent
    err_lower = median - lower
    err_upper = upper - median
    return median, err, err_lower, err_upper


def match_radec(df_a, df_b, populate_cols=[], ra_col_a='ra', dec_col_a='dec', ra_col_b='ra', dec_col_b='dec', max_sep_arcsec=1.0, add_prefix=False):
    """
    Match objects in df_a to df_b by sky coordinates within max_sep_arcsec.
    Returns a DataFrame with indices and separation for matches.
    """
    coords_a = SkyCoord(ra=df_a[ra_col_a].values * u.deg, dec=df_a[dec_col_a].values * u.deg)
    coords_b = SkyCoord(ra=df_b[ra_col_b].values * u.deg, dec=df_b[dec_col_b].values * u.deg)

    idx, d2d, _ = coords_a.match_to_catalog_sky(coords_b)
    match_mask = d2d < max_sep_arcsec * u.arcsec

    # Prepare result DataFrame
    result = df_a.copy()
    result['matched_idx_b'] = np.where(match_mask, idx, -1)
    result['matched_sep_arcsec'] = d2d.arcsec
    # Optionally, add matched object_id or similar column if present in df_b
    for col in populate_cols:
        matched_values = np.where(match_mask, df_b.iloc[idx][col].values, None)
        if add_prefix:
            result[f'matched_{col}'] = matched_values
        else:
            result[col] = matched_values

    # Print warnings for unmatched
    unmatched_object_ids = []
    for i, matched in enumerate(match_mask):
        if not matched:
            unmatched_object_ids.append(df_a.iloc[i]['object_id'])
    return result, unmatched_object_ids

def populate_xray(df, table_fpath="data/cscresults.vot"):
    # Read the CSC VOTable and normalize the column names.
    vo = parse(table_fpath)
    table = vo.get_first_table().to_table()

    fields = vo.get_first_table().fields
    new_names = [f.name for f in fields]
    print(new_names)
    for old, new in zip(table.colnames, new_names):
        table.rename_column(old, new)

    df_csc = table.to_pandas()

    # Match to the CSC catalog and copy the flux bounds.
    df_matched, unmatched_object_ids = match_radec(
        df, df_csc,
        populate_cols=['flux_aper_b', 'flux_aper_hilim_b', 'flux_aper_lolim_b'],
        max_sep_arcsec=1.0
    )
    print(f"Matched {len(df_matched) - len(unmatched_object_ids)} out of {len(df)} objects to CSC3 catalog.")

    # Ensure the matched flux columns are numeric.
    for c in ["flux_aper_b", "flux_aper_hilim_b", "flux_aper_lolim_b"]:
        df_matched[c] = pd.to_numeric(df_matched[c], errors="coerce")

    best = df_matched["flux_aper_b"]
    hi   = df_matched["flux_aper_hilim_b"]
    lo   = df_matched["flux_aper_lolim_b"]

    # Use the provided bounds to estimate a symmetric 1σ error.
    err = np.where(~hi.isna() & ~lo.isna(),
                   0.5 * (hi - lo),
                   np.where(~hi.isna(),
                            (hi - best),
                            np.where(~lo.isna(),
                                     (best - lo),
                                     np.nan)))
    df_matched["flux_aper_err_b"] = np.clip(err, a_min=0.0, a_max=None)

    # Build the luminosity distance in cm.
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    z = pd.to_numeric(df_matched["z"], errors="coerce")
    DL_cm = cosmo.luminosity_distance(z.values).to('cm').value

    # Adopt a fixed photon index and derive the nominal K-correction.
    GAMMA_X = 1.9
    Kcorr = np.power(1.0 + z.values, GAMMA_X - 2.0)  # = (1+z)^(-0.1) for Gamma=1.9
    # Keep the no-K-correction convention used elsewhere in this workflow.
    Kcorr = np.ones_like(Kcorr)

    # Pull the measured flux and error arrays.
    xray_flux     = df_matched["flux_aper_b"].replace(0, np.nan).values
    xray_flux_err = df_matched["flux_aper_err_b"].replace(0, np.nan).values

    # Compute 2-10 keV and monochromatic 2 keV luminosities.
    L_xray_210kev = 4.0 * np.pi * (DL_cm**2) * xray_flux * Kcorr
    L_xray_2 = L_xray_210kev * (2.0 - GAMMA_X) / (10.0**(2.0 - GAMMA_X) - 2.0**(2.0 - GAMMA_X)) * 2.0**(1.0 - GAMMA_X)

    df_matched["log_Lxray"] = np.log10(L_xray_2)

    # Propagate the X-ray flux error into log luminosity.
    df_matched["log_Lxray_err"] = (1.0 / np.log(10.0)) * (xray_flux_err / xray_flux)

    # Use the de-reddened UV luminosity for alpha_OX.
    df_matched["alphaOX"] = -(df_matched["log_Lxray"] - df_matched["log_L2500_int_fs"]) / 2.605
    df_matched["alphaOX_err"] = np.sqrt(
        df_matched["log_Lxray_err"]**2 + df_matched["log_L2500_int_fs_err"]**2
    ) / 2.605

    # Expected alpha_OX from Just et al. (2007), Equation 3.
    a = -0.140
    sigma_a = 0.007
    b = 2.705
    sigma_b = 0.212

    alphaOx_expected = a *  df_matched["log_L2500_int_fs"] + b

    x =  df_matched["log_L2500_int_fs"]
    sigma_x =  df_matched["log_L2500_int_fs_err"]

    alphaOx_expected_err = np.sqrt(
        (x**2) * (sigma_a**2) +
        (sigma_b**2) +
        (a**2) * (sigma_x**2)
    )

    df_matched["alphaOX_exp"] = alphaOx_expected
    df_matched["alphaOX_exp_err"] = alphaOx_expected_err

    # Compare the observed and expected alpha_OX values.
    df_matched["delta_alphaOX"] = df_matched["alphaOX"] - df_matched["alphaOX_exp"]
    df_matched["delta_alphaOX_err"] = np.sqrt(
        df_matched["alphaOX_err"]**2 + df_matched["alphaOX_exp_err"]**2
    )

    return df_matched


def populate_lc_info(df, lc_info_csv):
    fields = {
        'number_points': int,
        'cadence': float,
        'cadence_err': float,
        't_std': float,
    }
    # Load and concatenate two CSV files
    df_lcinfo = pd.read_csv(
        lc_info_csv,
        dtype={'object_id': str},
        converters=fields
    )
    merged = df.merge(df_lcinfo, on='object_id', how='left', suffixes=('', '_lcinfo'))

    print("Length of lc info merged DataFrame:", len(merged))
    missing_ids = set(df['object_id']) - set(df_lcinfo['object_id'])
    print("object_id not in merged:", list(missing_ids))

    for col in fields.keys():
        if f'{col}_lcinfo' in merged.columns:
            df[f'{col}'] = merged[f'{col}_lcinfo']
        else:
            df[col] = merged[col]

    return df

def populate_chisq_info(df, chisq_info_csv):
    fields = {
        'chi_sq_g': float,
        'chi_sq_all': float,
    }
    # Load and concatenate two CSV files
    df_chisqinfo = pd.read_csv(
        chisq_info_csv,
        dtype={'object_id': str},
        converters=fields
    )
    merged = df.merge(df_chisqinfo, on='object_id', how='left', suffixes=('', '_chisqinfo'))

    print("Length of chisq info merged DataFrame:", len(merged))
    missing_ids = set(df['object_id']) - set(df_chisqinfo['object_id'])
    print("object_id not in merged:", list(missing_ids))

    for col in fields.keys():
        if f'{col}_chisqinfo' in merged.columns:
            df[f'{col}'] = merged[f'{col}_chisqinfo']
        else:
            df[col] = merged[col]

    return df

def _norm_name(s):
    s = (s or "").replace("\ufeff", "").strip()
    return s

def _wrap_converters(fields):
    """Wrap converters to report column+value; do NOT include 'object_id' here."""
    wrapped = {}
    for col, fn in fields.items():
        if col == "object_id":
            continue
        def make(fn, col):
            def _conv(x):
                x = None if x == "" else x
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return np.nan
                try:
                    return fn(x)
                except Exception as e:
                    raise ValueError(f"Converter failed for column {col!r} with value {x!r}") from e
            return _conv
        wrapped[col] = make(fn, col)
    return wrapped

def _ensure_object_id(df):
    # rescue an index named object_id
    if df.index.name and _norm_name(df.index.name).lower() == "object_id" and "object_id" not in df.columns:
        df = df.reset_index()
    if "object_id" not in df.columns:
        raise KeyError("DataFrame lacks 'object_id' column after normalization.")
    df["object_id"] = df["object_id"].astype(str)
    return df

def parse_list(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if not s:
        return []
    # Try Python literal syntax first.
    try:
        v = literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [str(t) for t in v]
    except Exception:
        pass
    # Fall back to JSON.
    try:
        v = json.loads(s)
        if isinstance(v, (list, tuple)):
            return [str(t) for t in v]
    except Exception:
        pass
    # Finally, treat the value as a comma-separated string.
    return [t.strip() for t in s.split(",") if t.strip()]

def populate_spectra_fit(df, spectra_fit_csvs, best=True):
    # Spectral-fit columns to merge into the AGN catalog.
    fields = {
        'object_id': str,
        'f_host_2500': float,
        'f_host_5100': float,
        'ebv_fs': float,
        'euv_fs': float,
        'conti_a_0': float,
        'apparent_mag_2500_reddened': float,
        'apparent_mag_2500_reddened_err': float,
        'apparent_mag_2500': float,
        'apparent_mag_2500_err': float,
        'apparent_mag_i_rest': float,
        'apparent_mag_i_rest_err': float,
        'apparent_mag_i_obs': float,
        'apparent_mag_i_obs_err': float,
        'delta_m_flux_recal': float,
        'alpha_lambda': float,
        'alpha_lambda_err': float,
        'aic': float,
        'bic': float,
        'decomp_host': bool,
        'BC': bool,
        'best': bool,
        'bands_used': parse_list,
        'PL_slope': float,
        'PL_slope_err': float,
        'PL_slope_blue': float,
        'PL_slope_blue_err': float,
        'PL_slope_red': float,
        'PL_break_wave': float,
        'iron_frac': float,
        'PL_break_wave_inbounds': bool,
        'lam_rf_min': float,
        'lam_rf_max': float,
        'f_fe_uv_over_pl_3000': float,
        'f_bc_over_pl_3000': float,
        'f_host_center': float,
        'wrms': float,
        'frac_host_psf_2500': float
    }

    # Drop existing derived columns before re-merging them from the fit tables.
    drop_targets = [c for c in fields.keys() if c != "object_id"]
    existing_to_drop = [c for c in drop_targets if c in df.columns]
    if existing_to_drop:
        df = df.drop(columns=existing_to_drop)

    # Normalize the merge key on the left-hand table.
    df = _ensure_object_id(df)

    expanded_df = df  # will be replaced if best=False

    for i, csv_path in enumerate(spectra_fit_csvs):
        print(f"\033[96mLoading spectra fit CSV ({i+1}/{len(spectra_fit_csvs)}): {csv_path}\033[0m")

        wanted = set(fields.keys()) | {"object_id"}
        conv = _wrap_converters({k: v for k, v in fields.items() if k not in {"best", "BC", "decomp_host", "poly"}})

        df_spectra = pd.read_csv(
            csv_path,
            usecols=lambda c: _norm_name(c) in wanted,
            converters=conv,
            encoding="utf-8-sig",
            skipinitialspace=True,
        )

        # Normalize headers and ensure the merge key is present.
        df_spectra.columns = [_norm_name(c) for c in df_spectra.columns]
        df_spectra = _ensure_object_id(df_spectra)

        # Normalize boolean columns that may have been serialized as strings.
        for bcol in ["best", "BC", "decomp_host", "poly"]:
            if bcol in df_spectra.columns:
                df_spectra[bcol] = (
                    df_spectra[bcol]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(["true", "1", "t", "yes"])
                )

        if best:
            # Keep only the preferred fit for each object.
            if "best" in df_spectra.columns and df_spectra["best"].any():
                df_spectra = df_spectra[df_spectra["best"] == True].reset_index(drop=True)
                print(f"Length after best==True: {len(df_spectra)}")
            merged = df.merge(
                df_spectra,
                on="object_id",
                how="left",
                suffixes=("_old", "_spectralfit"),
                validate="one_to_one",
            )
            print("Length of merged DataFrame (best=True):", len(merged))

            # Copy spectral-fit columns onto matched rows only.
            matched_mask = df["object_id"].isin(df_spectra["object_id"])
            for col in list(fields.keys()):
                if col == "object_id":
                    continue
                src_col = f"{col}_spectralfit" if f"{col}_spectralfit" in merged.columns else col
                if src_col not in merged.columns:
                    continue
                if col in df.columns:
                    # overwrite only for matched rows
                    df.loc[matched_mask, col] = merged.loc[matched_mask, src_col].values
                else:
                    df.loc[matched_mask, col] = merged.loc[matched_mask, src_col].values

        else:
            # Preserve every fit row by expanding the table one-to-many.
            expanded_df = expanded_df.merge(
                df_spectra,
                on="object_id",
                how="left",
                validate="one_to_many",
            )
            print("Length after expanding with one-to-many merge:", len(expanded_df))

    # Continue with the selected output frame.
    out = df if best else expanded_df

    # Derive additional columns when the required inputs are present.
    if "redchi" in out.columns:
        out["log_redchi"] = np.log10(out["redchi"].replace(0, np.nan))
    if "ebv_fs" in out.columns:
        out["log_ebv_fs"] = np.log10(out["ebv_fs"].replace(0, np.nan))
    if "euv_fs" in out.columns:
        out["log_euv_fs"] = np.log10(out["euv_fs"].replace(0, np.nan))
    if {"apparent_mag_2500_reddened", "apparent_mag_2500"}.issubset(out.columns):
        out["dm_red"] = out["apparent_mag_2500_reddened"] - out["apparent_mag_2500"]
    if {"apparent_mag_2500_reddened_err", "apparent_mag_2500_err"}.issubset(out.columns):
        out["dm_red_err"] = np.sqrt(
            out["apparent_mag_2500_reddened_err"]**2 + out["apparent_mag_2500_err"]**2
        )
    
    if "reddening_integral" in out.columns:
        out["log_reddening_integral"] = np.log10(out["reddening_integral"].replace(0, np.nan))
    if "delta_qso01_redchi2" in out.columns:
        out["log_delta_qso01_redchi2"] = np.log10(np.abs(out["delta_qso01_redchi2"].replace(0, np.nan)))
    if "conti_a_0" in out.columns:
        out["log_conti_a_0"] = np.log10(out["conti_a_0"].replace(0, 1e-9))  # avoid log(0)

    # If 'alpha_lambda' is missing but 'PL_slope' exists, set alpha_lambda = PL_slope
    if "alpha_lambda" not in out.columns and "PL_slope" in out.columns:
        out["alpha_lambda"] = out["PL_slope"]
    if "alpha_lambda_err" not in out.columns and "PL_slope_err" in out.columns:
        out["alpha_lambda_err"] = out["PL_slope_err"]
    
    out['alpha_nu'] = -out['alpha_lambda'] - 2
    out['alpha_nu_err'] = out['alpha_lambda_err']

    # Convert apparent magnitude at 2500 A into log luminosity.
    out['log_L2500_int_fs'] = -0.4 * (out['apparent_mag_2500'])
    out['log_L2500_int_fs'] += 36.0  # 90 * 0.4 = 36

    # Propagate the magnitude error into log luminosity.
    out['log_L2500_int_fs_err'] = 0.4 * out['apparent_mag_2500_err']

    out['iron_frac'] = out['f_fe_uv_over_pl_3000']

    return out
def populate_sdss_fields(objs, progress_bar=True):
    print(f"Populating SDSS fields: {len(objs)}", flush=True)
    cat = pd.read_parquet(f"data/S82/Catalog.parquet").set_index('idx')
    hdul = fits.open('data/dr16q_prop_May01_2024.fits')
    fits_data = hdul[1].data  # Assuming the data is in the first extension    
    fits_data_2 = hdul[2].data  # Assuming the data is in the first extension    
    for d in tqdm(objs, desc="Populating SDSS fields", disable=(not progress_bar)):
        obj = cat.loc[cat['objectId'] == d['object_id']].iloc[0]
        c1 = SkyCoord(fits_data['RA'], fits_data['DEC'], unit='deg')
        c2 = SkyCoord(obj['RA'], obj['DEC'], unit='deg')
        sep = c1.separation(c2).to(u.arcsec)
        i = np.argwhere(sep < 1*u.arcsec).flatten()
        if len(i) == 0:
            print(f"Skipping entry {d['object_id']} as it does not exist in the fits data.")
            continue
        if len(i) > 1:
            print(f"Warning: {d['sdss_name']} found multiple times in SDSS data")
        i = i[0]  # Get the first index if there are multiple matches
        d['ra'] = obj['RA']
        d['dec'] = obj['DEC']
        d['z'] = fits_data['Z_SYS'][i]
        d['z_err'] = fits_data['Z_SYS_ERR'][i]
        d['sdss_name'] = fits_data['SDSS_NAME'][i]
        d['log_lbol'] = -999.0
        if d['z'] < 0.7:
            d['log_lbol'] = np.log10(5.15) + fits_data['LOGL3000'][i]
            d['log_lbol_err'] = fits_data['LOGL3000_ERR'][i]
        else:
            d['log_lbol'] = fits_data['LOGLBOL'][i]
            d["log_lbol_err"] = fits_data['LOGLBOL_ERR'][i]

        d['LOGMBH'] = fits_data['LOGMBH'][i]
        d['LOGMBH_ERR'] = fits_data['LOGMBH_ERR'][i]
        d['LOGLEDD_RATIO'] = fits_data['LOGLEDD_RATIO'][i]
        d['LOGLEDD_RATIO_ERR'] = fits_data['LOGLEDD_RATIO_ERR'][i]
        d['ebv_wu'] = fits_data['EBV'][i]
        d['sn_median_all'] = fits_data['SN_MEDIAN_ALL'][i]
        d['M_i'] = fits_data_2['M_I'][i]
        for b in ['u', 'g', 'r', 'i', 'z']:
            filters = {'u':0, 'g':1, 'r':2, 'i':3, 'z':4}
            d[f'PSFMAG_{b}'] = fits_data_2['PSFMAG'][i, filters[b]]
        d['LOGLBOL'] = fits_data['LOGLBOL'][i]
        d['LOGL1350'] = fits_data['LOGL1350'][i]
        d['LOGL1700'] = fits_data['LOGL1700'][i]
        d['LOGL2500_wu'] = fits_data['LOGL2500'][i]
        d['LOGL3000'] = fits_data['LOGL3000'][i]
        d['LOGL5100'] = fits_data['LOGL5100'][i]
        d['LOGL1350_ERR'] = fits_data['LOGL1350_ERR'][i]
        d['LOGL1700_ERR'] = fits_data['LOGL1700_ERR'][i]
        d['LOGL2500_ERR_wu'] = fits_data['LOGL2500_ERR'][i]
        d['LOGL3000_ERR'] = fits_data['LOGL3000_ERR'][i]
        d['LOGL5100_ERR'] = fits_data['LOGL5100_ERR'][i]

        d['FHOST_5100'] = fits_data['FHOST_5100'][i]
        d['EXTINCTION'] = fits_data_2['EXTINCTION'][i, filters['i']]

        d['fhost'] = fits_data['FHOST_5100'][i]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for b in ['u', 'g', 'r', 'i', 'z']:
                d[f'apparent_mag_{b}'] = -2.5 * np.log10(fits_data_2['PSFFLUX'][i, filters[b]]) + 22.5
                d[f'apparent_mag_{b}_err'] = 2.5/np.log(10) * np.sqrt(1/fits_data_2['PSFFLUX_IVAR'][i, filters[b]])/fits_data_2['PSFFLUX'][i, filters[b]]
        if any(issubclass(warning.category, RuntimeWarning) for warning in w):
            pass

    return objs

def read_quasars_from_hdf5(file_path, N=None):
    quasar_list = []

    with h5py.File(file_path, "r") as hdf:
        for group_name in tqdm(list(hdf.keys()), desc="Reading quasars from HDF5"):
            group = hdf[group_name]
            quasar = {"object_id": group_name}
            for key, value in group.attrs.items():
                quasar[key] = value
            for sub_group_name in group.keys():
                sub_group = group[sub_group_name]
                quasar[sub_group_name] = {sub_key: sub_group[sub_key][...] for sub_key in sub_group.keys()}
            quasar_list.append(quasar)
            if (N is not None) and (N >= 0) and (len(quasar_list) >= N):
                break
    return quasar_list

def load_agn_data(file_path, populate_sdss=False, apply_cut=True, fhost_cut=DEFAULT_F_HOST_CUT,
                  exclude_object_ids_csv=None,
                  residuals_sigma_clip=None, residuals_csv=None,
                  spectra_fit_csv=None, only_load=False,
                  pickled=False,
                  correct_sigma_uv_host=False,
                  iron_frac_cut=None, wrms_cut=None,
                  lc_info_csv="data/aug4_sample_chisqg10_ebv005sn3_lcdata.csv",
                  chisq_info_csv="data/aug4_sample_chisqg10_ebv005sn3.csv",
                  z_range=(0.44, 3.16),
                  plot_path="plots/hubble"):
    from hubble_plotting import (
        plot_Mi_relation,
        plot_cut_diagnostics,
        plot_m2500_vs_z_colorpanels,
        plot_sigma_uv_host_correction,
        plot_tau_sigma_vs_wu_catalog,
        plot_tau_sigma_vs_redshift,
    )

    if exclude_object_ids_csv is None:
        exclude_object_ids_csv = []
    
    if pickled:
        with open(file_path + ".pkl", "rb") as f: 
            quasar_list = pickle.load(f)
    else:
        quasar_list = read_quasars_from_hdf5(file_path)
    print("Number of quasars loaded:", len(quasar_list))

    if populate_sdss:
        print("Populating SDSS fields...")
        populate_sdss_fields(quasar_list)
        write_hdf5_file(quasar_list, file_path)

    for quasar in quasar_list:
        if 'ebv_wu' not in quasar.keys():
            print("Populating SDSS fields...")
            populate_sdss_fields(quasar_list)
            write_hdf5_file(quasar_list, file_path)
            break
        
    bands = ['u', 'g', 'r', 'i', 'z']
    if len(quasar_list[0]['mags_mean']) == 5:
        bands = ['u', 'g', 'r', 'i', 'z']
    elif len(quasar_list[0]['mags_mean']) == 3:
        bands = ['g', 'r', 'i']
    else:
        raise ValueError("Unexpected number of bands in mags_means")
    
    for q in quasar_list:
        clean_bands = set()
        for i, b in enumerate(bands):
            q[f'mags_mean_{b}'] = q['mags_mean'][i]

        del q['mags_mean']

    for q in quasar_list:
        q['len_dropped_bands'] = len(q['dropped_bands'])

    df = pd.DataFrame(quasar_list)

    dropped_bands = df['dropped_bands']
    jitter_total_sq = np.zeros(len(df))
    amp_delta_blr_total_sq = np.zeros(len(df))

    for b in ['u', 'g', 'r', 'i']:
        #df.loc[dropped_bands.apply(lambda s: b in s), f'log_jitter_{b}'] = np.nan
        jitter = 10**df[f'log_jitter_{b}'].values
        jitter[dropped_bands.apply(lambda s: b in s)] = 0.0
        df[f'jitter_{b}'] = jitter
        jitter_total_sq = jitter_total_sq + jitter**2

        #df.loc[dropped_bands.apply(lambda s: b in s), f'log_amp_delta_blr_{b}'] = np.nan
        amp_delta_blr = 10**df[f'log_amp_delta_blr_{b}'].values
        amp_delta_blr[dropped_bands.apply(lambda s: b in s)] = 0.0
        amp_delta_blr_total_sq = amp_delta_blr_total_sq + amp_delta_blr**2
        df[f'amp_delta_blr_{b}'] = amp_delta_blr
        blr2_col = f'log_amp_delta_blr2_{b}'
        if blr2_col in df.columns:
            amp_delta_blr2 = 10**df[blr2_col].values
            amp_delta_blr2[dropped_bands.apply(lambda s: b in s)] = 0.0
            amp_delta_blr_total_sq = amp_delta_blr_total_sq + amp_delta_blr2**2
            df[f'amp_delta_blr2_{b}'] = amp_delta_blr2

    df['log_jitter_total'] = np.log10(np.sqrt(jitter_total_sq))
    df['log_amp_delta_blr_total'] = np.log10(np.sqrt(amp_delta_blr_total_sq))

    # df['log_sigma_UV'] = df['log_sigma_UV'] + 1/2 * np.log10(1 + df['z'])

    df_sample = pd.read_csv("data/aug4_sample_chisqg10_ebv005sn3.csv", dtype={'object_id': str})
    print("Entire Sample file length:", len(df_sample))
    # TODO: populate z
    #plot_redshift_histogram(df_sample.copy(), df.copy(), bins=30, cut_info="Chisq sample vs fitted LCs")

    if spectra_fit_csv is not None:
        print("Populating spectra fit data from:", spectra_fit_csv)
        df = populate_spectra_fit(df, spectra_fit_csv)
    else:
        print("[WARNING] spectra_fit_csv not provided, assuming spectral fit fields are in agn h5 file")
        if 'alpha_lambda' not in df.columns:
            raise ValueError("spectra_fit_csv not provided and spectral fields not found in agn h5 file")
            #raise ValueError("spectra_fit_csv must be provided if alpha_lambda not in agn h5 file")

    if "log_sigma_UV" in df.columns:
        df["log_sigma_UV_uncorrected"] = pd.to_numeric(df["log_sigma_UV"], errors="coerce")
    if correct_sigma_uv_host:
        if {"log_sigma_UV_uncorrected", "frac_host_psf_2500"}.issubset(df.columns):
            frac_host_psf = pd.to_numeric(df["frac_host_psf_2500"], errors="coerce")
            valid_frac_host = np.isfinite(frac_host_psf) & (frac_host_psf != -1.0)
            agn_frac_psf = 1.0 - frac_host_psf
            valid_hostcorr = valid_frac_host & np.isfinite(agn_frac_psf) & (agn_frac_psf > 0.0)
            df["sigma_UV_hostcorr_factor"] = np.where(valid_hostcorr, 1.0 / agn_frac_psf, np.nan)
            df["log_sigma_UV"] = np.where(
                valid_hostcorr,
                df["log_sigma_UV_uncorrected"] + np.log10(df["sigma_UV_hostcorr_factor"]),
                df["log_sigma_UV_uncorrected"],
            )
            print(
                "Applied sigma_UV host correction using frac_host_psf_2500: "
                "sigma_UV_corrected = sigma_UV / (1 - frac_host_psf_2500)"
            )
            plot_sigma_uv_host_correction(df, plot_path=plot_path, show=False)
        else:
            raise KeyError("correct_sigma_uv_host=True requires 'log_sigma_UV' and 'frac_host_psf_2500'.")

    if {"z", "log_tau_UV_RF", "log_sigma_UV"}.issubset(df.columns):
        plot_tau_sigma_vs_redshift(df, plot_path=plot_path, show=False)
    if {"log_tau_UV_RF", "log_sigma_UV", "LOGMBH", "LOGLEDD_RATIO"}.issubset(df.columns):
        plot_tau_sigma_vs_wu_catalog(df, plot_path=plot_path, show=False)

    # Remove objects with implausibly bright or faint apparent magnitude at 2500 A.
    mag_mask = ((df['apparent_mag_2500'] >= 16) & (df['apparent_mag_2500'] < 24))
    num_removed = np.sum(~mag_mask)
    print(f"Cut on apparent_mag_2500 : {num_removed} objects removed")
    plot_cut_diagnostics(df.copy(), df[mag_mask], bins=30, cut_info="16 < apparent_mag_2500 < 24")
    df = df[mag_mask].reset_index(drop=True)

    df = populate_xray(df)
    
    if lc_info_csv is not None:
        print("Populating LC info from:", lc_info_csv)
        df = populate_lc_info(df, lc_info_csv)
    else:
        print("[WARNING] lc_info_csv not provided")
    if chisq_info_csv is not None:
        print("Populating LC info from:", chisq_info_csv)
        df = populate_chisq_info(df, chisq_info_csv)
    else:
        print("[WARNING] chisq_info_csv not provided")
    

    num_quasars_z_0_1_before = len(df[(df['z'] > 0) & (df['z'] <= 1.0)])
    num_quasars_z_gt_3_before = len(df[df['z'] > 3])
    print("Number of quasars with 0 < z <= 1.0:", num_quasars_z_0_1_before)
    print("Number of quasars with z > 3:", num_quasars_z_gt_3_before)
    print("Highest redshift quasar:", df['z'].max())

    df_all = df.copy()

    if only_load:
        return df

    df['log_t_rf_length'] = np.log10(df['t_rf_length'])

    df['log_f_host_2500'] = np.where(df['f_host_2500'] > 0, np.log10(df['f_host_2500']), np.nan)
    df['log_f_host_5100'] = np.where(df['f_host_5100'] > 0, np.log10(df['f_host_5100']), np.nan)
    
    df = df.reset_index(drop=True)
    
    # Apply a small hand-maintained exclusion list.
    exclusion_object_ids = []
    mask_exclude = ~df['object_id'].astype(str).isin(exclusion_object_ids)

    exclusion_sdss_names = [
        '221120.38+010905.6', # removed because wrong redshift
        '024555.35+005332.6', # removed because the spectrum is anomalous
        '015802.36+002917.3', # next to a star
    ]
    mask_exclude &= (~df['sdss_name'].astype(str).isin(exclusion_sdss_names))
    print(f"Excluding {np.sum(~mask_exclude)} objects by exclusion list")
    df = df[mask_exclude].reset_index(drop=True)


    mask_valid = (df['log_tau_UV_RF'] > 2*df['log_sigma_UV'] + 2.5)
    num_removed = np.sum(~mask_valid)
    print(f"Cut on tau vs sigma diagram: {num_removed} objects removed")
    plot_cut_diagnostics(df.copy(), df[mask_valid], bins=30, cut_info="tau > 2*sigma + 2.5")

    df = df[mask_valid].reset_index(drop=True)
    # mask_in  = df_agn["z"].between(0.44, 3.16)

    # Remove outliers listed in external CSV files.
    for exclude_csv in exclude_object_ids_csv:
        if os.path.exists(exclude_csv):
            exclude_df = pd.read_csv(exclude_csv)
            exclude_ids = set(exclude_df['object_id'].astype(str))
            mask_exclude = ~df['object_id'].astype(str).isin(exclude_ids)
            num_excluded = np.sum(~mask_exclude)
            print(f"Excluding {num_excluded} objects from DataFrame based on {exclude_csv}")
            plot_cut_diagnostics(df.copy(), df[mask_exclude], bins=30, cut_info="exclude csv")
            df = df[mask_exclude].reset_index(drop=True)
        else:
            print(f"[WARNING] Exclusion CSV not found: {exclude_csv}")

    # Remove objects with too many dropped bands.
    mask_dropped = ~df['len_dropped_bands'].isin([4, 5])
    num_removed_dropped = np.sum(~mask_dropped)
    print(f"Removed {num_removed_dropped} objects with len_dropped_bands == 4 or 5")
    plot_cut_diagnostics(df.copy(), df[mask_dropped], bins=30, cut_info="dropped bands 4 or 5")
    df = df[mask_dropped].reset_index(drop=True)
    blr_amp_cuts = build_log_amp_delta_blr_cuts()
    print("Active per-band BLR amplitude cuts:")
    for col, lower, upper in blr_amp_cuts:
        print(f"  {col}: lower={lower}, upper={upper}, allow_missing=True")

    for col, lower, upper in blr_amp_cuts:
        mask = np.ones(len(df), dtype=bool)
        if lower is not None:
            mask &= df[col] >= lower
        if upper is not None:
            mask &= df[col] < upper
        mask |= df[col].isna()
        num_removed = np.sum(~mask)
        cut_desc = f"{lower}<{col}<{upper} or NaN"
        print(f"Removed {num_removed} objects failing {cut_desc}")
        plot_cut_diagnostics(df.copy(), df[mask], bins=30, cut_info=cut_desc)
        df = df[mask].reset_index(drop=True)

    cuts = build_agn_cuts(
        f_host_cut=fhost_cut,
        iron_frac_cut=iron_frac_cut if iron_frac_cut is not None else None,
        wrms_cut=wrms_cut if wrms_cut is not None else None,
    )
    print("Active AGN scalar cuts:")
    for col, lower, upper in cuts:
        print(f"  {col}: lower={lower}, upper={upper}")

    if apply_cut:
        initial_count = len(df)
        mask = np.ones(len(df), dtype=bool)
        for col, lower, upper in cuts:
            col_mask = np.ones(len(df), dtype=bool)
            if lower is not None:
                col_mask &= df[col] >= lower
            if upper is not None:
                col_mask &= df[col] <= upper
            cut_count = np.sum(~col_mask)
            print(f"Cut on {col}: {cut_count} objects removed")
            plot_cut_diagnostics(df.copy(), df[col_mask], bins=30, cut_info=f"{lower}<{col}<{upper}")
            mask &= col_mask
        df = df[mask]

        print(f"Total objects removed by all cuts: {initial_count - len(df)}")
    # Drop rows that still lack the core continuum fit parameters.
    remove_nans_columns = ['alpha_lambda', 'alpha_lambda_err']
    for col in remove_nans_columns:
        nan_mask = ~df[col].isna()
        num_nans = (~nan_mask).sum()
        print(f"Removing {num_nans} objects with NaN in column '{col}'")
        df = df[nan_mask]
        plot_cut_diagnostics(df.copy(), df[mask], bins=30, cut_info=f"{col} not NaN")

    df = df.reset_index(drop=True)

    if residuals_sigma_clip is not None and residuals_csv is not None:
        if os.path.exists(residuals_csv):
            residual_df = pd.read_csv(residuals_csv)
            if 'residuals' not in residual_df.columns:
                raise ValueError(f"'residuals' column not found in {residuals_csv}")
            mu_zscore = dict(zip(residual_df['object_id'].astype(str), residual_df['mu_zscore']))
            df['mu_zscore'] = df['object_id'].astype(str).map(mu_zscore)
            mask_residual = df['mu_zscore'].abs() < residuals_sigma_clip
            num_removed = np.sum(~mask_residual)
            print(f"Cut on sigma clip with mu_score < {residuals_sigma_clip}: {num_removed} objects removed")
            df = df.drop(columns=['mu_zscore'])
            plot_cut_diagnostics(df.copy(), df[mask_residual], bins=30, cut_info=f"|mu_zscore|<{residuals_sigma_clip}")

            df = df[mask_residual].reset_index(drop=True)
        else:
            print(f"[WARNING] Residual CSV not found: {residuals_csv}")
            raise ValueError(f"Residual CSV not found: {residuals_csv}")

    # Require a positive finite magnitude uncertainty at 2500 A.
    num_before = len(df)
    mask = (df['apparent_mag_2500_err'] > 0) & np.isfinite(df['apparent_mag_2500_err'])
    plot_cut_diagnostics(df.copy(), df[mask], bins=30, cut_info=f"0<apparent_mag_2500_err<inf")

    df = df[mask].reset_index(drop=True)
    num_after = len(df)
    print(f"Dropped {num_before - num_after} objects with apparent_mag_2500_err <= 0 or not finite")

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    y_log_meas_err = 0.4 * np.asarray(df['apparent_mag_2500_err'].fillna(1e9))
    actual_M2500 = df['apparent_mag_2500'] - cosmo.distmod(df['z']).value
    actual_logL2500 = convert_M2500_to_logL2500(actual_M2500)
    yerr_linear = 10**actual_logL2500 * np.log(10) * y_log_meas_err
    mask = yerr_linear/(10**actual_logL2500) < 0.5
    plot_cut_diagnostics(df.copy(), df[mask], bins=30, cut_info=f"frac_err_logL2500<0.5")

    num_removed = np.sum(~mask)
    print(f"\033[93mRemoved {num_removed} objects with fractional logL2500 error >= 0.5\033[0m")
    df = df[mask]

    num_quasars_z_0_1 = len(df[(df['z'] > 0) & (df['z'] <= 1.0)])
    num_quasars_z_gt_3 = len(df[df['z'] > 3])
    print("Number of quasars with 0 < z <= 1.0:", num_quasars_z_0_1)
    print("Number of dropped quasars with 0 < z <= 1.0:", num_quasars_z_0_1_before - num_quasars_z_0_1)
    print("Number of quasars with z > 3:", num_quasars_z_gt_3)
    print(f"\nTotal number of objects removed by all cuts: {len(df_all) - len(df)}")
    print("Final number of quasars:", len(df))
    plot_cut_diagnostics(df_all.copy(), df.copy(), bins=30, cut_info="all cuts")
    colorpanel_cols = [
        col for col in ("f_host_center", "f_fe_uv_over_pl_3000", "f_bc_over_pl_3000", "wrms")
        if col in df_all.columns
    ]
    if len(colorpanel_cols) > 0 and "z" in df_all.columns and "apparent_mag_2500" in df_all.columns:
        cuts_plot_dir = os.path.join("plots", "hubble", "cuts")
        os.makedirs(cuts_plot_dir, exist_ok=True)
        fig_colorpanels, _ = plot_m2500_vs_z_colorpanels(
            df_all,
            df_keep=df,
            color_cols=tuple(colorpanel_cols),
            z_range=z_range,
        )
        fig_colorpanels.savefig(
            os.path.join(cuts_plot_dir, "m2500_vs_z_colorpanels.pdf"),
            bbox_inches="tight",
        )
        plt.close(fig_colorpanels)
    plot_Mi_relation(df_all.copy())
    
    return df, df_all

def load_pantheon_data():
    """
    Load Pantheon+SH0ES data and build the covariance *for the selected subset*:
      selection = (zHD > 0.01) OR (IS_CALIBRATOR == True)

    Returns:
        df_pantheon_sel : pandas.DataFrame (filtered to the selection)
        sna_logdetCov   : float, log|C_sel|
        sna_L           : ndarray, Cholesky factor of C_sel (lower)
        sna_lower       : bool (True) for cho_solve
    """

    # Load the Pantheon+ catalog.
    print("Loading Pantheon+ supernova data...")
    df_pantheon = pd.read_csv(
        # "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat",
        "data/Pantheon+SH0ES.dat",
        sep=r"\s+"
    )
    
    # Select cosmological SNe plus calibrators.
    is_calib = np.asarray(df_pantheon["IS_CALIBRATOR"], dtype=bool)
    sel_mask = (df_pantheon["zHD"].values > 0.01) | is_calib

    # Load the full covariance and apply the same selection.
    print("Loading SN covariance matrix...")
    n_sn = len(df_pantheon)
    cov_flat = np.loadtxt(
        # "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov",
        "data/Pantheon+SH0ES_STAT+SYS.cov",
        skiprows=1
    )
    cov_matrix = cov_flat.reshape((n_sn, n_sn))
    assert cov_matrix.shape == (n_sn, n_sn), f"Expected ({n_sn},{n_sn}), got {cov_matrix.shape}"

    cov_sel = cov_matrix[sel_mask][:, sel_mask]
    df_pantheon_sel = df_pantheon.loc[sel_mask].reset_index(drop=True)

    # Cholesky factorization of the selected covariance.
    try:
        sna_L, sna_lower = cho_factor(cov_sel, lower=True)
    except np.linalg.LinAlgError:
        raise ValueError("Selected covariance submatrix is not positive-definite!")

    # Compute log|C| from the Cholesky factor.
    sna_logdetCov = 2.0 * np.sum(np.log(np.diag(sna_L)))

    n_sel = cov_sel.shape[0]
    print(f"Cholesky factorization successful. "
          f"Selected SNe: {n_sel} / {n_sn} "
          f"(kept {(sel_mask).sum()}; dropped {n_sn - (sel_mask).sum()}).")

    return df_pantheon_sel, sna_logdetCov, sna_L, sna_lower

LN10 = math.log(10.0)
LN2  = math.log(2.0)
TWOPI = 2.0 * math.pi
HALF_LN_TWOPI = 0.5 * math.log(TWOPI)
Phi = NormalDist()

def _bayes_factor_repr_from_delta(delta, delta_err=None):
    log10K = delta / LN10
    s_main = f"10^{log10K:.2f}"
    ci_tuple = None
    if delta_err is not None:
        lo = (delta - delta_err) / LN10
        hi = (delta + delta_err) / LN10
        s_ci = f"[10^{lo:.2f}, 10^{hi:.2f}]"
        ci_tuple = (lo, hi, s_ci)
    return log10K, s_main, ci_tuple

def _jeffreys_strength(abs_delta, thresholds):
    t1, t2, t3 = thresholds
    if abs_delta < t1:
        return "barely worth mentioning"
    elif abs_delta < t2:
        return "substantial"
    elif abs_delta < t3:
        return "Strong"
    else:
        return "very strong"

def odds_sigmas_from_delta(delta):
    """Return Kass&Raftery 1995 sigma Z from |Δln Z| odds, stably."""
    absD = abs(float(delta))
    sigma = np.sqrt(2.0 * absD)
    return sigma


def compare_models_by_log_evidence_all(
        df_agn,
        cosmo_models_dict,
        jeffreys_thresholds=(1.0, 2.5, 5.0),   # |Δln Z| bands
        z_decisive=2.0,
        write_path="plots/hubble/"
):
    """
    Compare MANY models by log-evidence.
    Inputs
    ------
    cosmo_models_dict : dict
        { model_label: {"logZ": float, "logZerr": float}, ... }

    Returns
    -------
    result : dict with:
      - 'ranking': list of per-model dicts sorted by logZ (desc)
      - 'preferred_model': label of top model
      - 'top_vs_runnerup': dict with ΔlnZ, Δerr, z_mc, odds-based Z (1- & 2-sided),
                           Jeffreys strength, Bayes-factor strings
      - 'pairwise': dict of dicts with pairwise deltas and Z’s
      - 'text_path': path to saved human-readable summary
    """
    # Collect and validate the model evidence inputs.
    items = []
    for label, vals in cosmo_models_dict.items():
        try:
            z = float(vals["logZ"])
            e = float(vals["logZerr"])
        except Exception as exc:
            raise ValueError(f"Model '{label}' missing numeric 'logZ'/'logZerr'") from exc
        items.append((label, z, e))
    if len(items) < 2:
        raise ValueError("Need at least two models to compare.")

    # Sort models by descending log evidence.
    items.sort(key=lambda t: t[1], reverse=True)
    labels = [t[0] for t in items]
    logZs  = np.array([t[1] for t in items], dtype=float)
    errs   = np.array([t[2] for t in items], dtype=float)

    top_label, top_logZ, top_err = items[0]
    preferred_model = top_label

    # Compute per-model comparisons relative to the top-ranked model.
    ranking = []
    for (label, z, e) in items:
        d = z - top_logZ   # <= 0 for all except top (0)
        de = float(np.hypot(e, top_err))
        z_mc = np.inf if de == 0 else d / de
        z2 = odds_sigmas_from_delta(d)
        log10K, B_str, B_ci = _bayes_factor_repr_from_delta(d, de)
        strength = _jeffreys_strength(abs(d), jeffreys_thresholds)
        ranking.append({
            "model": label,
            "logZ": z,
            "logZerr": e,
            "delta_logZ_vs_top": d,
            "delta_logZ_err_vs_top": de,
            "z_mc_vs_top": z_mc,
            #"sigma_one_sided_vs_top": z1,
            "sigma_two_sided_vs_top": z2,
            "jeffreys_strength_vs_top": strength,
            "log10_Bayes_factor_vs_top": log10K,
            "Bayes_factor_str_vs_top": B_str,
            "Bayes_factor_ci_1sigma_vs_top": B_ci,
        })

    # Summarize the top-vs-runner-up comparison.
    if len(items) >= 2:
        ru_label, ru_logZ, ru_err = items[1]
        delta = top_logZ - ru_logZ
        delta_err = float(np.hypot(top_err, ru_err))
        z_mc_head = np.inf if delta_err == 0 else delta / delta_err
        absD = abs(delta)
        sigma_one = 0
        sigma_two = odds_sigmas_from_delta(delta)
        # CI via ±1σ on Δ
        def _odds_sigmas_at(d):
            return odds_sigmas_from_delta(d)
        s2_lo = _odds_sigmas_at(delta - delta_err)
        s2_hi = _odds_sigmas_at(delta + delta_err)

        log10K, B_str, B_ci = _bayes_factor_repr_from_delta(delta, delta_err)
        strength = _jeffreys_strength(abs(delta), jeffreys_thresholds)
        decisive = abs(z_mc_head) >= z_decisive

        top_vs_runnerup = {
            "preferred_model": top_label,
            "runner_up": ru_label,
            "delta_logZ": delta,
            "delta_logZ_err": delta_err,
            "z_mc": z_mc_head,
            "sigma_from_odds_one_sided": sigma_one,
            "sigma_from_odds_two_sided": sigma_two,
            #"sigma_from_odds_one_sided_ci_1sigma": (s1_lo, s1_hi),
            "sigma_from_odds_two_sided_ci_1sigma": (s2_lo, s2_hi),
            "log10_Bayes_factor": log10K,
            "Bayes_factor_str": B_str,
            "Bayes_factor_ci_1sigma": B_ci,
            "jeffreys_strength": strength,
            "decisive_zmc_ge_thresh": decisive,
        }
    else:
        top_vs_runnerup = None

    # Build the full pairwise comparison matrix.
    pairwise = {}
    for i, (li, zi, ei) in enumerate(items):
        pairwise[li] = {}
        for j, (lj, zj, ej) in enumerate(items):
            if i == j:
                pairwise[li][lj] = {
                    "delta_logZ": 0.0,
                    "delta_logZ_err": float(np.hypot(ei, ej)),
                    "z_mc": np.nan,
                    "sigma_one_sided": np.nan,
                    "sigma_two_sided": np.nan,
                    "jeffreys_strength": "—",
                    "log10_Bayes_factor": 0.0,
                    "Bayes_factor_str": "1:1",
                    "Bayes_factor_ci_1sigma": None,
                }
            else:
                d = zi - zj
                de = float(np.hypot(ei, ej))
                zmc = np.inf if de == 0 else d / de
                z2 = odds_sigmas_from_delta(d)
                log10K, B_str, B_ci = _bayes_factor_repr_from_delta(d, de)
                strength = _jeffreys_strength(abs(d), jeffreys_thresholds)
                pairwise[li][lj] = {
                    "delta_logZ": d,
                    "delta_logZ_err": de,
                    "z_mc": zmc,
                    #"sigma_one_sided": z1,
                    "sigma_two_sided": z2,
                    "jeffreys_strength": strength,
                    "log10_Bayes_factor": log10K,
                    "Bayes_factor_str": B_str,
                    "Bayes_factor_ci_1sigma": B_ci,
                }

    # Build a human-readable text summary.
    lines = []
    lines.append("Bayesian Model Comparison (multi-model)\n\n")
    lines.append("Models (sorted by ln Z):\n")
    for r in ranking:
        star = "  *" if r["model"] == preferred_model else "   "
        lines.append(
            f"{star} {r['model']}: ln Z = {r['logZ']:.3f} ± {r['logZerr']:.3f} ; "
            f"ΔlnZ(top) = {r['delta_logZ_vs_top']:.3f} ± {r['delta_logZ_err_vs_top']:.3f} ; "
            f"Z_two = {r['sigma_two_sided_vs_top']:.3f}σ ; "
            f"{r['jeffreys_strength_vs_top']}\n"
        )
    lines.append("\nPreferred model: " + preferred_model + "\n")

    if top_vs_runnerup is not None:
        t = top_vs_runnerup
        lines.append(
            f"\nTop vs runner-up ({t['preferred_model']} vs {t['runner_up']}):\n"
            f"Δln Z = {t['delta_logZ']:.3f} ± {t['delta_logZ_err']:.3f}  "
            f"(z_mc = {t['z_mc']:.2f})\n"
            f"Two-sided Z (from odds): {t['sigma_from_odds_two_sided']:.4f}σ  "
            f"[{t['sigma_from_odds_two_sided_ci_1sigma'][0]:.4f}, "
            f"{t['sigma_from_odds_two_sided_ci_1sigma'][1]:.4f}]\n"
            f"Jeffreys strength: {t['jeffreys_strength']}; "
            f"decisive (|z_mc|≥{z_decisive:.1f})? {'yes' if t['decisive_zmc_ge_thresh'] else 'no'}\n"
            f"Number of AGNs: {len(df_agn)}\n"
        )

    # Print and save the text summary.
    for line in lines:
        print(line, end="")
    os.makedirs(write_path, exist_ok=True)
    text_path = os.path.join(write_path, "compare_all_models.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return {
        "ranking": ranking,
        "preferred_model": preferred_model,
        "top_vs_runnerup": top_vs_runnerup,
        "pairwise": pairwise,
        "text_path": text_path,
    }



def write_hdf5_file(quasar_list, file_path):
    print(f"Writing {len(quasar_list)} quasars to {file_path}", flush=True)
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    with h5py.File(file_path, "w") as hdf:
        for quasar in quasar_list:
            object_id = quasar["object_id"]

            group = hdf.create_group(object_id)
            for key, value in quasar.items():
                if isinstance(value, dict):
                    sub_group = group.create_group(key)
                    for sub_key, sub_value in value.items():
                        sub_group.create_dataset(sub_key, data=sub_value)
                else:
                    group.attrs[key] = value

def extract_cosmo_results_from_samples(
    samples,
    cosmo_model,
    only_sna,
    logZ_tuple=None,
    *,
    format_for_latex=False,
    value_fmt="{:.3f}",
):
    """
    Extract summary stats for all cosmological parameters from posterior samples.

    Parameters
    ----------
    samples : (N, P) ndarray
        Posterior samples. Columns must align with `model_labels` from get_model_params(cosmo_model).
    cosmo_model : str
        'FlatwCDM' or 'Flatw0waCDM'
    only_sna : bool
        True if SN Ia only; False if SN Ia + AGN.
    logZ_tuple : (float, float) or None
        (logZ, logZerr) from dynesty; None for emcee.
    format_for_latex : bool, optional
        If True, values are strings like r"$x \pm y$". Otherwise tuples (mean, std).
    value_fmt : str, optional
        Format for numbers when format_for_latex=True (e.g., "{:.2f}").

    Returns
    -------
    dict
        {
          "model": "<latex model name>",
          "data": "<latex data label>",
          "params": { "<param_name>": (mean, std) or "$x \\pm y$", ... },
          "param_order": [ "<param_name>", ... ],              # for consistent table columns
          "param_labels_latex": { "<param_name>": "<latex>", ... },
          "logZ": { "value": float, "err": float } or None
        }
    """
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)

    # Validate the sample array shape.
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError("`samples` must be a 2D array of shape (n_samples, n_params).")
    if samples.shape[1] != len(model_labels):
        raise ValueError(
            f"`samples` has {samples.shape[1]} columns but model expects {len(model_labels)} "
            f"({model_labels}). Ensure column order matches `model_labels`."
        )

    data_label = "SN~Ia" if only_sna else "SN~Ia + AGN"
    if cosmo_model == "Flatw0waCDM":
        model_name_latex = "flat $w_0w_a$CDM"
    elif cosmo_model == "FlatwCDM":
        model_name_latex = "flat $w$CDM"
    elif cosmo_model == "FlatLambdaCDM":
        model_name_latex = "flat $\Lambda$CDM"
    else:
        model_name_latex = cosmo_model

    # Compute mean and standard deviation for every sampled parameter.
    means = np.mean(samples, axis=0)
    stds  = np.std(samples, axis=0, ddof=0)

    def pack(m, s):
        if format_for_latex:
            return rf"${value_fmt.format(m)} \pm {value_fmt.format(s)}$"
        return (m, s)

    params = {name: pack(means[i], stds[i]) for i, name in enumerate(model_labels)}
    param_labels_latex = {name: model_labels_latex[i] for i, name in enumerate(model_labels)}

    logZ_out = None
    if logZ_tuple is not None:
        logZ_val, logZ_err = logZ_tuple
        logZ_out = {
            "value": float(logZ_val),
            "err": float(logZ_err),
        }

    return {
        "model": model_name_latex,
        "data": data_label,
        "params": params,                       # all params present
        "param_order": list(model_labels),      # preserve column order for LaTeX
        "param_labels_latex": param_labels_latex,
        "logZ": logZ_out,
    }

def display_results_summary(samples, cosmo_model, z_pivot_agn):
    """
    Print median and 16/84% intervals for sampled params, plus derived w0 (and wa)
    when applicable. If cosmo_model == 'Flatw0waCDM', w0 is computed from (wp, wa)
    at the supplied z_pivot_agn.
    """
    _, model_labels, _ = get_model_params(cosmo_model)
    samples = np.asarray(samples)

    # Print the sampled parameter summaries first.
    med = np.median(samples, axis=0)
    lo  = np.percentile(samples, 16, axis=0)
    hi  = np.percentile(samples, 84, axis=0)
    for name, m, l, h in zip(model_labels, med, lo, hi):
        print(f"{name:>15}: {m:.4f} (+{h - m:.4f}, -{m - l:.4f})")

    # Resolve parameter names across model variants.
    def _idx(labels, *cands):
        for c in cands:
            if c in labels:
                return labels.index(c)
        return None  # not found

    # Print derived summaries for the relevant cosmology.
    if cosmo_model == "Flatw0waCDM":
        i_wp = _idx(model_labels, "wp", "w_p")
        i_wa = _idx(model_labels, "wa", "w_a")
        if i_wp is None or i_wa is None:
            return  # nothing to do

        a_p = 1.0 / (1.0 + float(z_pivot_agn))
        wp  = samples[:, i_wp]
        wa  = samples[:, i_wa]
        w0  = wp - (1.0 - a_p) * wa

        for name, arr in (("w0", w0), ("wa", wa)):
            m = np.median(arr)
            l = np.percentile(arr, 16)
            h = np.percentile(arr, 84)
            print(f"{name:>15}: {m:.4f} (+{h - m:.4f}, -{m - l:.4f})")

    else:
        i_w0 = _idx(model_labels, "w0", "w_0", "w")
        if i_w0 is not None:
            arr = samples[:, i_w0]
            m = np.median(arr); l = np.percentile(arr, 16); h = np.percentile(arr, 84)
            print(f"{'w0':>15}: {m:.4f} (+{h - m:.4f}, -{m - l:.4f})")

        i_wa = _idx(model_labels, "wa", "w_a")
        if i_wa is not None:
            arr = samples[:, i_wa]
            m = np.median(arr); l = np.percentile(arr, 16); h = np.percentile(arr, 84)
            print(f"{'wa':>15}: {m:.4f} (+{h - m:.4f}, -{m - l:.4f})")
    return

def _weighted_quantile(x, q, w=None):
    """
    Weighted quantiles of x at probabilities q in [0,1].
    If w is None, falls back to np.quantile.
    """
    x = np.asarray(x)
    q = np.atleast_1d(q)
    if w is None:
        return np.quantile(x, q)
    w = np.asarray(w)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        raise ValueError("No finite values for weighted quantile.")
    x, w = x[m], w[m]
    order = np.argsort(x)
    x, w = x[order], w[order]
    cum_w = np.cumsum(w)
    cum_w /= cum_w[-1]
    return np.interp(q, cum_w, x)

def compute_age_universe_with_error(samples, cosmo_model, weights=None, ci=(0.68, 0.95), max_eval=None, random_seed=None):
    """
    Compute the posterior distribution of the Universe age and summarize it.

    Parameters
    ----------
    samples : (N, P) array
        Posterior samples aligned with `model_labels` returned by get_model_params(cosmo_model).
    cosmo_model : str
        One of {"Flatw0waCDM","FlatwCDM","FlatLambdaCDM","FlatwpwaCDM"}.
    weights : (N,) array or None
        Optional sample weights (e.g., from nested sampling). If None, treats samples equally.
    ci : tuple
        Credible levels to report as central intervals, e.g., (0.68, 0.95).
    max_eval : int or None
        If set, randomly subsample to at most this many evaluations for speed.
    random_seed : int or None
        RNG seed for reproducible subsampling.

    Returns
    -------
    stats : dict
        {
          "mean": float, "std": float,
          "q16": float, "q50": float, "q84": float,
          "q2.5": float, "q97.5": float,
          "ages": np.ndarray (valid posterior ages in Gyr),
          "n_total": int, "n_valid": int, "n_invalid": int
        }
    """
    # Get parameter names & any needed priors (e.g., zp for FlatwpwaCDM)
    priors, model_labels, _ = get_model_params(cosmo_model)

    N = samples.shape[0]
    idx = np.arange(N)

    # Optionally subsample the posterior for speed.
    if (max_eval is not None) and (N > max_eval):
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(idx, size=max_eval, replace=False)
        samples = samples[idx]
        if weights is not None:
            weights = np.asarray(weights)[idx]

    ages = np.full(samples.shape[0], np.nan, dtype=float)

    # Evaluate the cosmological age sample by sample.
    for j in range(samples.shape[0]):
        params = {name: samples[j, i] for i, name in enumerate(model_labels)}
        try:
            if cosmo_model == "Flatw0waCDM":
                cosmo = Flatw0waCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'], wa=params['wa'])
            elif cosmo_model == "FlatwCDM":
                cosmo = FlatwCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'])
            elif cosmo_model == "FlatLambdaCDM":
                cosmo = FlatLambdaCDM(H0=params['H0'], Om0=params['Om0'])
            elif cosmo_model == "FlatwpwaCDM":
                cosmo = FlatwpwaCDM(H0=params['H0'], Om0=params['Om0'],
                                     wp=params['wp'], wa=params['wa'],
                                     zp=priors.get('zp', 0.0))
            else:
                raise ValueError(f"Unknown cosmology model: {cosmo_model}")

            age = cosmo.age(0).to(u.Gyr).value
            if np.isfinite(age) and (age > 0):
                ages[j] = age
        except Exception:
            continue

    m = np.isfinite(ages)
    n_valid = int(np.sum(m))
    n_total = ages.size
    n_invalid = n_total - n_valid
    if n_valid == 0:
        raise RuntimeError("All age evaluations failed; check parameter ranges.")

    a = ages[m]
    w = None if weights is None else np.asarray(weights)[m]

    # Summarize the valid age samples.
    mean = np.average(a, weights=w) if w is not None else float(np.mean(a))
    if w is not None:
        w_norm = w / np.sum(w)
        var = np.average((a - mean) ** 2, weights=w_norm)
        std = float(np.sqrt(var))
    else:
        std = float(np.std(a, ddof=1))

    q16, q50, q84 = _weighted_quantile(a, [0.16, 0.50, 0.84], w)
    q025, q975 = _weighted_quantile(a, [0.025, 0.975], w)

    print(f"Age of universe: {q50:.3f} (+{q84 - q50:.3f}/-{q50 - q16:.3f}) Gyr  "
          f"[mean={mean:.3f}±{std:.3f} Gyr; valid {n_valid}/{n_total}, skipped {n_invalid}]")

    return mean, std

def compute_pivot_redshift(flat_samples, cosmo_model, z_min=0.0, z_max=4.0):
    """
    Compute the optimal pivot redshift z_p* from MCMC samples of (wp, wa),
    with a constraint z_p* >= z_min (default 0).

    Returns
    -------
    z_p_star : float
        Constrained optimal pivot redshift (decorrelates w_p and w_a as much as
        possible under the z >= z_min constraint).
    """
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    idx = {name: model_labels.index(name) for name in model_labels}

    # Samples
    wp = flat_samples[:, idx['wp']]
    wa = flat_samples[:, idx['wa']]

    # Read the current pivot from the priors if present.
    z_p_current = None
    for k in ('zp', 'z_p', 'z_pivot_agn', 'z_pivot'):
        if k in priors:
            z_p_current = priors[k] if np.isscalar(priors[k]) else float(priors[k])
            break
    if z_p_current is None:
        z_p_current = 0.0

    a_p = 1.0 / (1.0 + z_p_current)

    # Compute the covariance structure of the pivot parameters.
    C = np.cov(np.vstack([wp, wa]))
    cov_wp_wa = C[0, 1]
    var_wa    = C[1, 1]

    # Compute the unconstrained optimal pivot in scale factor.
    tiny = np.finfo(float).tiny
    a_p_star = a_p + cov_wp_wa / max(var_wa, tiny)

    # Enforce the requested redshift bounds.
    a_max_allowed = 1.0 / (1.0 + z_min)  # for z_min=0, this is 1
    a_min_allowed = 1.0 / (1.0 + (z_max if z_max is not None else np.inf))

    if z_max is not None:
        a_p_star = max(a_p_star, a_min_allowed)
    a_p_star = min(a_p_star, a_max_allowed)

    # Guard against non-physical scale factors.
    a_p_star = max(a_p_star, np.finfo(float).eps)
    z_p_star = 1.0 / a_p_star - 1.0
    return z_p_star

def posterior_corr(flat_samples, cosmo_model, z_pivot_agn):
    """
    Pearson correlation between posterior w0 and wa.

    If cosmo_model is 'Flatw0waCDM', w0 is derived from (wp, wa) at the given pivot redshift.
    Otherwise, tries to read w0 directly from samples.

    Parameters
    ----------
    flat_samples : (N, P) array
        Flattened MCMC samples: N total draws by P parameters.
    cosmo_model : str
        Cosmological model string.
    z_pivot_agn : float, optional
        Pivot redshift to compute w0 from wp, wa for Flatw0waCDM.

    Returns
    -------
    rho : float
        Posterior Pearson correlation coefficient corr(w0, wa).
    """
    priors, model_labels, _ = get_model_params(cosmo_model)
    flat_samples = np.asarray(flat_samples)

    def _idx(labels, *names):
        for name in names:
            if name in labels:
                return labels.index(name)
        raise ValueError(f"Parameter {names} not found in model labels.")

    if cosmo_model == "Flatw0waCDM":
        i_w0 = _idx(model_labels, "w0", "w_0")
        i_wa = _idx(model_labels, "wa", "w_a")
        w0 = flat_samples[:, i_w0]
        wa = flat_samples[:, i_wa]
    elif cosmo_model == "FlatwpwaCDM":
        i_wp = _idx(model_labels, "wp", "w_p")
        i_wa = _idx(model_labels, "wa", "w_a")
        a_p = 1.0 / (1.0 + float(z_pivot_agn))
        wp = flat_samples[:, i_wp]
        wa = flat_samples[:, i_wa]
        w0 = wp - (1.0 - a_p) * wa
    else:
        i_w0 = _idx(model_labels, "w0", "w_0", "w")
        i_wa = _idx(model_labels, "wa", "w_a")
        w0 = flat_samples[:, i_w0]
        wa = flat_samples[:, i_wa]

    mask = np.isfinite(w0) & np.isfinite(wa)
    if not np.any(mask):
        raise ValueError("No finite samples to compute correlation.")

    return np.corrcoef(w0[mask], wa[mask])[0, 1]

import math


def _to_finite_float(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def format_result_errors(
    val,
    err=None,
    err_lower=None,
    err_upper=None,
    unit=None,
    nd=2,
    sci_threshold=3,
):
    """
    Return a LaTeX-ready string with fixed decimal places.

    Examples
    --------
    12.35 \\pm 0.46
    (1.23 \\pm 0.05) \\times 10^{5}
    0.84^{+0.07}_{-0.05}
    (3.21^{+0.11}_{-0.09}) \\times 10^{-4}

    Notes
    -----
    - If either err_lower or err_upper is provided, asymmetric errors are used.
    - `unit` should already be LaTeX-ready, e.g. r"\\Msun" or r"\\mathrm{km\\,s^{-1}}".
    """

    valf = _to_finite_float(val)
    if valf is None:
        return "N/A"

    errf = _to_finite_float(err)
    errlf = _to_finite_float(err_lower)
    erruf = _to_finite_float(err_upper)

    # Use asymmetric mode if either asymmetric error is provided
    use_asym = (errlf is not None) or (erruf is not None)

    mags = [abs(valf)]
    if errf is not None:
        mags.append(abs(errf))
    if errlf is not None:
        mags.append(abs(errlf))
    if erruf is not None:
        mags.append(abs(erruf))

    max_mag = max(mags) if mags else 0.0
    exponent = int(math.floor(math.log10(max_mag))) if max_mag > 0 else 0

    use_sci = abs(exponent) >= sci_threshold
    scale = 10.0 ** exponent if use_sci else 1.0

    v = valf / scale
    fmt = f"{{:.{nd}f}}"

    if use_asym:
        el = abs(errlf) / scale if errlf is not None else 0.0
        eu = abs(erruf) / scale if erruf is not None else 0.0
        core = rf"{fmt.format(v)}^{{+{fmt.format(eu)}}}_{{-{fmt.format(el)}}}"
    elif errf is not None:
        e = abs(errf) / scale
        core = rf"{fmt.format(v)} \pm {fmt.format(e)}"
    else:
        core = fmt.format(v)

    if use_sci:
        out = rf"({core}) \times 10^{{{exponent}}}"
    else:
        out = core

    if unit:
        out += rf"\,\mathrm{{{unit}}}"

    return out

def write_results_tex_variables(
    df_agn, df_agn_all, df_pantheon, z_range, cosmo_model_joint_samples, cosmo_model_sna_samples, compare_r,
    write_path, result_prefix="", chisq_dict=None, cosmo_models_result_dict=None
):
    """
    Write key cosmological parameters AND model comparison results
    to a LaTeX file.
    """

    # Capitalize the first letter of result_prefix if present
    if result_prefix:
        result_prefix = result_prefix[0].upper() + result_prefix[1:]

    lines = []
    lines.append(r"% Auto-generated cosmological and evidence results")
    lines.append(r"% Do not edit manually; regenerated by write_results_tex_variables()")

    # Local formatting helpers.
    def _clean(name):
        return name.replace("0", "Zero").replace("Λ", "Lambda").replace("_", "")

    def _cmd(name, content, model_suffix=""):
        cmd_name = f"result{result_prefix}{_clean(model_suffix)}{name}"
        return f"\\newcommand{{\\{cmd_name}}}{{\\ensuremath{{{content}}}}}"

    def _sym_percentile(data, percentiles=[16, 50, 84]):
        if len(data) == 0: return np.nan, np.nan
        p = np.percentile(data, percentiles)
        return p[1], (p[2] - p[0]) / 2.0

    # Global AGN and SN summary counts.
    obs_arr, err_arr, pivots_arr = agn_model_pack_obs(df_agn)
    log_sigma_UV_pivot = pivots_arr[agn_model_oidx["log_sigma_UV"]]
    log_tau_UV_RF_pivot = pivots_arr[agn_model_oidx["log_tau_UV_RF"]]
    n_fitted = len(df_agn[df_agn['z'].between(z_range[0], z_range[1])])
    lines.append(_cmd("NumAGNInitial", len(df_agn_all)))
    lines.append(_cmd("NumAGNCut", len(df_agn_all)-len(df_agn)))

    lines.append(_cmd("NumAGNPlotted", len(df_agn)))
    lines.append(_cmd("NumAGNFitted", n_fitted))

    is_calib_bool = np.asarray(df_pantheon['IS_CALIBRATOR'], dtype=bool)
    mask = (df_pantheon['zHD'] > 0.01) | is_calib_bool
    lines.append(_cmd("NumSNaPlotted", len(df_pantheon)))
    lines.append(_cmd("NumSNaFitted", len(df_pantheon[mask])))
    lines.append(_cmd("SigmaUVPivot", f"{10**log_sigma_UV_pivot:.1f}"))
    lines.append(_cmd("TauUVRFPivot", f"{10**log_tau_UV_RF_pivot:.0f}"))

    for model_name, flat_samples in cosmo_model_sna_samples.items():
        flat_samples = np.asarray(flat_samples)
        priors, model_labels, _ = get_model_params(model_name)
        results = {}
        for i, key in enumerate(model_labels):
            median, err, err_lower, err_upper = sym_percentile(flat_samples[:, i])
            results[key] = median
            results[f"{key}_err"] = err
            results[f"{key}_err_lower"] = err_lower
            results[f"{key}_err_upper"] = err_upper

        if 'M0_sn' in results:
            lines.append(_cmd("SNMZero", format_result_errors(results['M0_sn'],results['M0_sn_err']), model_suffix=model_name))
        if 'Om0' in results:
            lines.append(_cmd("SNOmZero", format_result_errors(results['Om0'],results['Om0_err']), model_suffix=model_name))
        if 'w0' in results:
            lines.append(_cmd("SNwZero", format_result_errors(results['w0'],err_lower=results['w0_err_lower'], err_upper=results['w0_err_upper']), model_suffix=model_name))
        if 'wa' in results:
            lines.append(_cmd("SNwa", format_result_errors(results['wa'],err_lower=results['wa_err_lower'], err_upper=results['wa_err_upper'], nd=1), model_suffix=model_name))
        if 'H0' in results:
             lines.append(_cmd("SNHZero", format_result_errors(results['H0'],results['H0_err']), model_suffix=model_name))


    # Per-model summary parameters.
    for model_name, flat_samples in cosmo_model_joint_samples.items():
        flat_samples = np.asarray(flat_samples)
        priors, model_labels, _ = get_model_params(model_name)
        results = {}
        for i, key in enumerate(model_labels):
            median, err, err_lower, err_upper = sym_percentile(flat_samples[:, i])
            results[key] = median
            results[f"{key}_err"] = err
            results[f"{key}_err_lower"] = err_lower
            results[f"{key}_err_upper"] = err_upper

        if 'Om0' in results:
            lines.append(_cmd("OmZero", format_result_errors(results['Om0'],results['Om0_err']), model_suffix=model_name))
        if 'w0' in results:
            lines.append(_cmd("wZero", 
                            format_result_errors(results['w0'],err_lower=results['w0_err_lower'], err_upper=results['w0_err_upper']),
                            model_suffix=model_name))
        if 'wa' in results:
            lines.append(_cmd("wa", 
                              format_result_errors(results['wa'],err_lower=results['wa_err_lower'], err_upper=results['wa_err_upper'], nd=1),
                              model_suffix=model_name))
        if 'H0' in results:
             lines.append(_cmd("HZero", format_result_errors(results['H0'],results['H0_err']), model_suffix=model_name))
        if 'alpha_agn' in results:
            lines.append(_cmd("AlphaAGN", format_result_errors(results['alpha_agn'],results['alpha_agn_err']), model_suffix=model_name))
        if 'beta_agn' in results:
            lines.append(_cmd("BetaAGN", format_result_errors(results['beta_agn'],results['beta_agn_err']), model_suffix=model_name))
        if 'M0_agn' in results:
            lines.append(_cmd("MZeroAGN", format_result_errors(results['M0_agn'],results['M0_agn_err']), model_suffix=model_name))

        result = cosmo_models_result_dict[model_name]
        lines.append(_cmd("AgeUniverse", format_result_errors(result["age"], result["age_err"], unit=r"Gyr"), model_suffix=model_name))


        try:
            idx_M0 = model_labels.index('M0_agn')
            idx_alpha = model_labels.index('alpha_agn')
            idx_beta = model_labels.index('beta_agn')
            idx_logf = model_labels.index('log_f')

            L_intercept = np.power(10, (90 - flat_samples[:, idx_M0]) / 2.5)
            alpha_L = flat_samples[:, idx_alpha] * (-1/2.5)
            beta_L = flat_samples[:, idx_beta] * (-1/2.5)
            hd_scatter = np.exp(flat_samples[:, idx_logf])
            l_scatter = hd_scatter / 2.5

            lines.append(_cmd("LIntercept", format_result_errors(*_sym_percentile(L_intercept), unit=r"erg\,s^{-1}"), model_suffix=model_name))
            lines.append(_cmd("AlphaAGNL", format_result_errors(*_sym_percentile(alpha_L)), model_suffix=model_name))
            lines.append(_cmd("BetaAGNL", format_result_errors(*_sym_percentile(beta_L)), model_suffix=model_name))
            lines.append(_cmd("ScatterHD", format_result_errors(*_sym_percentile(hd_scatter), unit=r"mag"), model_suffix=model_name))
            lines.append(_cmd("ScatterL", format_result_errors(*_sym_percentile(l_scatter), unit=r"dex"), model_suffix=model_name))
        except ValueError:
            pass

        if chisq_dict and model_name in chisq_dict:
            lines.append(_cmd("ChiSqRed", format_result_errors(chisq_dict[model_name]), model_suffix=model_name))

    # Model-comparison summaries.
    if compare_r:
        lines.append(r"% --- Model Comparisons ---")
        if "preferred_model" in compare_r:
            lines.append(_cmd("PreferredModelOverall", compare_r["preferred_model"]))

        for r in compare_r.get("ranking", []):
            m_name = r["model"]
            lines.append(_cmd("LogZ", f"{r['logZ']:.1f}", model_suffix=m_name))
            lines.append(_cmd("LogZerr", f"{r['logZerr']:.1f}", model_suffix=m_name))
            lines.append(_cmd("DeltaLogZ", f"{r['delta_logZ_vs_top']:.1f}", model_suffix=m_name))
            lines.append(_cmd("Sigma", f"{r['sigma_two_sided_vs_top']:.1f}", model_suffix=m_name))
            lines.append(_cmd("JeffreysStrength", r['jeffreys_strength_vs_top'], model_suffix=m_name))

        pw = compare_r.get("pairwise", {})
        ranked_models = [r["model"] for r in compare_r.get("ranking", [])]
        
        for a, b in combinations(ranked_models, 2):
            pair = pw.get(a, {}).get(b) or pw.get(b, {}).get(a)
            pair_name = f"{_clean(a)}{_clean(b)}" 

            if pair:
                lines.append(_cmd(f"DeltaLogZ{pair_name}", f"{pair['delta_logZ']:.1f} \\pm {pair['delta_logZ_err']:.1f}"))
                lines.append(_cmd(f"Sigma{pair_name}", f"{pair['sigma_two_sided']:.1f}"))
                lines.append(_cmd(f"JeffreysStrength{pair_name}", pair['jeffreys_strength']))
            else:
                lines.append(_cmd(f"DeltaLogZ{pair_name}", "N/A"))

    # Write the LaTeX command file.
    filename = f"param_results_{result_prefix.lower()}.tex" if result_prefix else "param_results.tex"
    tex_path = os.path.join(write_path, filename)
    os.makedirs(write_path, exist_ok=True)
    
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")
    
    print(f"Wrote result parameters LaTeX commands to {tex_path}")

def reduced_chi_squared(residuals,
                        model_err,
                        extra_err=None,
                        n_params=0,
                        min_err=1e-12):
    """
    Compute reduced chi^2 from residuals and per-point uncertainties.

    Parameters
    ----------
    residuals : array-like
        Data minus model, in the SAME units as the uncertainties
        (here: log10 L_2500).
    model_err : array-like
        1σ model spread at each data point (e.g., 0.5*(hi-lo) of the ribbon),
        in log10 units.
    extra_err : None or array-like, optional
        Optional additional 1σ errors per point to add in quadrature
        (e.g., measurement error in log10 L, propagated x-error in log space).
    n_params : int, optional
        Number of fitted parameters used to define the model (for DoF).
    min_err : float, optional
        Floor to avoid division by ~0.

    Returns
    -------
    chi2_red : float
        Reduced chi-squared.
    meta : dict
        {'chi2': float, 'dof': int, 'N_eff': int, 'n_params': int}
    """
    r = np.asarray(residuals, dtype=float)
    s = np.asarray(model_err,  dtype=float)

    if extra_err is not None:
        s = np.sqrt(s**2 + np.asarray(extra_err, dtype=float)**2)

    # Guard against non-finite/zero sigmas
    finite = np.isfinite(r) & np.isfinite(s) & (s > 0)
    r = r[finite]
    s = np.maximum(s[finite], min_err)

    N = r.size
    dof = max(1, N - int(n_params))

    chi2 = np.sum((r / s)**2)
    chi2_red = chi2 / dof

    return chi2_red, {'chi2': chi2, 'dof': dof, 'N_eff': N, 'n_params': int(n_params)}

def cosmo_model_label_latex(cosmo_model):
    if   cosmo_model == 'FlatwCDM':      label = r"flat $w$CDM model"
    elif cosmo_model == 'Flatw0waCDM':   label = r"flat $w_0w_a$CDM model"
    elif cosmo_model == 'FlatLambdaCDM': label = r"flat $\Lambda$CDM model"
    elif cosmo_model == 'FlatwpwaCDM':   label = r"flat $w_p\!-\!w_a$CDM model"
    else:
        raise ValueError("Invalid cosmology model.")
    
    return label


def _save_mapping_hdf5(filename, **kwargs):
    """Save keyword arrays/scalars into a flat HDF5 file."""
    with h5py.File(filename, 'w') as f:
        for name, data in kwargs.items():
            if isinstance(data, (np.ndarray, list)):
                f.create_dataset(name, data=data, compression="gzip")
            else:
                f.create_dataset(name, data=data)
    print(f"Saved: {list(kwargs.keys())} to {filename}")


def _load_mapping_hdf5(filename):
    """Load a flat HDF5 file into a dictionary."""
    results = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            results[key] = f[key][()]
    return results


def save_chains(filename, **kwargs):
    """Backward-compatible wrapper for saving posterior chains to HDF5."""
    _save_mapping_hdf5(filename, **kwargs)


def load_chains(filename):
    """Backward-compatible wrapper for loading posterior chains from HDF5."""
    return _load_mapping_hdf5(filename)

def save_cosmo_results_hdf5(filename, models_dict):
    """
    Saves a dictionary of cosmological models to HDF5.
    Input structure: {'ModelName': {'param': val, 'param_err': val, ...}}
    """
    with h5py.File(filename, 'w') as f:
        for model_name, params in models_dict.items():
            # Create a dedicated Group for each model (e.g., 'FlatLambdaCDM')
            grp = f.create_group(model_name)
            
            for param_name, value in params.items():
                if value is None:
                    value = np.nan  # HDF5 doesn't support None, use NaN for missing values
                grp.create_dataset(param_name, data=value)
    
    print(f"Saved models: {list(models_dict.keys())} to {filename}")

def select_agn_subset_uniform_with_replacement(
    df_agn,
    z_range,
    N=None,
    subset_seed=42,
    id_col="object_id",
    z_uniform_min=0.44,
    kde_bw_method=None,
    max_weight_ratio=20.0,
):
    """
    Deterministic weighted sampling WITH replacement, using continuous
    inverse-density weights in redshift so the selected sample is
    approximately uniform in z between z_uniform_min and z_range[1].

    Same inputs -> same returned sample.
    Duplicates are allowed.
    """
    zmin, zmax = z_range
    z0 = max(zmin, z_uniform_min)

    df_sel = df_agn.copy()
    df_sel = df_sel[df_sel["z"].between(z0, zmax)].copy()

    n_avail = len(df_sel)
    print(f"AGN available after cuts: {n_avail}")

    if n_avail == 0:
        raise ValueError("No AGN available after z_range cut.")

    if N is None:
        df_sel = df_sel.sort_values([id_col, "z"]).reset_index(drop=True)
        print("Using all AGN in range.")
        print(f"+++ Length of selected AGNs: {len(df_sel)}")
        print(f"+++ Redshift range of selected AGNs: {df_sel['z'].min()} to {df_sel['z'].max()}")
        return df_sel

    z = df_sel["z"].to_numpy()

    # estimate parent density p(z)
    kde = gaussian_kde(z, bw_method=kde_bw_method)
    pz = kde(z)

    # target uniform in z => w(z) ∝ 1 / p(z)
    w = 1.0 / np.clip(pz, 1e-12, None)

    # cap extreme weights for stability
    w_med = np.median(w)
    w = np.minimum(w, max_weight_ratio * w_med)

    # normalize after sorting so results do not depend on input row order
    df_sel = df_sel.sort_values([id_col, "z"]).reset_index(drop=True)
    z = df_sel["z"].to_numpy()
    pz = kde(z)
    w = 1.0 / np.clip(pz, 1e-12, None)
    w_med = np.median(w)
    w = np.minimum(w, max_weight_ratio * w_med)
    probs = w / w.sum()

    rng = np.random.default_rng(subset_seed)
    idx = rng.choice(len(df_sel), size=N, replace=True, p=probs)

    df_out = df_sel.iloc[idx].reset_index(drop=True)

    print(f"Selected N={len(df_out)} AGN with subset_seed={subset_seed} (with replacement)")
    print(f"+++ Redshift range of selected AGNs: {df_out['z'].min()} to {df_out['z'].max()}")

    n_unique = df_out[id_col].nunique()
    print(f"+++ Unique AGNs in selected sample: {n_unique}/{len(df_out)}")

    return df_out


def report_pivots(df_agn):
    print("\nAGN pivot summary")
    print("-" * 68)
    print(f"{'Quantity':<15}{'Type':<18}{'log10 value':>14}{'linear value':>16}")

    rows = [
        ("sigma_UV", "computed mean", np.mean(df_agn["log_sigma_UV"])),
        ("tau_UV_RF", "computed mean", np.mean(df_agn["log_tau_UV_RF"])),
    ]

    _, _, pivots_arr = agn_model_pack_obs(df_agn)

    rows.extend([
        ("sigma_UV", "fixed pivot", pivots_arr[agn_model_oidx["log_sigma_UV"]]),
        ("tau_UV_RF", "fixed pivot", pivots_arr[agn_model_oidx["log_tau_UV_RF"]]),
    ])

    for name, kind, log_val in rows:
        lin_val = 10**log_val
        print(f"{name:<15}{kind:<18}{log_val:>14.4f}{lin_val:>16.4f}")
