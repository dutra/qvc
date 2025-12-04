import os
prefix = os.environ.get('PREFIX', 'test')


import matplotlib.pyplot as plt
import numpy as np
import h5py
from astropy.cosmology import FlatwCDM, Flatw0waCDM, FlatLambdaCDM, FlatwpwaCDM
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import interp1d
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u
#from astroquery.vizier import Vizier
from tqdm import tqdm
import warnings
import re
from scipy import stats
from scipy.stats import norm, sigmaclip, multivariate_normal
from scipy.interpolate import RegularGridInterpolator

from dynesty.utils import resample_equal
from hubble_model import get_model_params, M_model_agn, M_model_agn_err, agn_model_pack_obs, agn_model_pack_params, agn_model_oidx
from hubble_completeness import make_dm_function

from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.stats import linregress
from scipy.stats import pearsonr

bands = ['u', 'g', 'r', 'i', 'z']#, 'y']
bands_idx = {b: i for i, b in enumerate(bands)}
filters = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5} # harcoded filter order for SDSS

def convert_M2500_to_logL2500(M2500):
    return -1/2.5 * (M2500 - 90.0)

def convert_logL2500_to_M2500(logL2500):
    return -2.5 * logL2500 + 90.0

def sym_percentile(x, p=[16, 50, 84], axis=0):
    lower, median, upper = np.percentile(x, p, axis=axis)
    return median, 0.5 * (upper - lower)

def find_optimal_pivot(flat_samples,
                       cosmo_model,
                       df_agn):
    """
    Find the pivot redshift z_pivot that minimizes the correlation between 
    M_pred(z_pivot) and H0 using AGN samples and model.

    Parameters
    ----------
    flat_samples : ndarray of shape (nsamples, nparams)
        Posterior samples from MCMC or nested sampling.
    param_indices : dict
        Dictionary mapping parameter names to column indices in flat_samples.
    df_agn : DataFrame
        AGN data with columns: 'z', 'alpha_nu', 'log_sigma_UV', 'log_tau_UV_RF'.
    apparent_mag : array-like
        Observed apparent magnitude at 2500 Å for each AGN.
    z_grid : array-like
        Grid of redshifts to scan as possible pivot values.
    alpha_nu_default : float
        Default alpha_nu if not given in df_agn.

    Returns
    -------
    z_best : float
        Optimal pivot redshift where Corr(H0, M_pred(z)) ≈ 0.
    z_grid : array
        Array of pivot redshifts tested.
    corrs : array
        Pearson correlations between M_pred(z_pivot) and H0 for each pivot redshift.
    """
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    param_indices = {name: model_labels.index(name) for name in model_labels}
    # If alpha_nu not in df_agn, use constant
    alpha_nu = df_agn['alpha_nu'].values

    z_grid = np.linspace(0.1, 3.0, 1000)  # Grid of redshifts to test as pivot
    # Precompute redshift-independent K-correction difference terms for each z_pivot
    z_data = df_agn['z'].values

    H0_samples = flat_samples[:, param_indices['H0']]
    apparent_mag_2500 = df_agn['apparent_mag_2500'].values
    corrs = []

    for z_pivot in z_grid:

        M_pred_samples = np.array([
            apparent_mag_2500 - M_model_agn(
                s[param_indices['M0_agn']],
                s[param_indices['log_sigma_UV_break']],
                s[param_indices['eta_A1_agn']],
                s[param_indices['eta_A2_agn']],
                s[param_indices['eta_break_agn']],
                s[param_indices['beta_agn']],
                df_agn['log_sigma_UV'].values,
                df_agn['log_tau_UV_RF'].values
            )
            for s in flat_samples
        ])  # shape: (nsamples, n_agn)

        # Collapse across AGNs: average M_pred per sample
        M_pred_mean = M_pred_samples.mean(axis=1)

        r, _ = pearsonr(M_pred_mean, H0_samples)
        corrs.append(r)

    corrs = np.array(corrs)
    z_best = z_grid[np.argmin(np.abs(corrs))]

    return z_best, z_grid, corrs

def sn_completeness_function(m_b, z, mlim=24.1, sigma=0.5):
    """
    Return probability of SN detection given apparent magnitude m_b and redshift z.
    mlim: effective magnitude limit for SN survey.
    sigma: sharpness of detection efficiency curve.
    """
    return 1.0 - norm.cdf(m_b, loc=mlim, scale=sigma)

def log_nuLnu_to_m2500(log_nuLnu, z):
    raise NotImplementedError("Use apparent_mag_2500_from_mi_rest in hubble_completeness_refactored.py instead.")
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    DL_cm = cosmo.luminosity_distance(z).to('cm').value
    m_AB = (
        -2.5 * log_nuLnu
        + 5 * np.log10(DL_cm)
        + 2.5 * np.log10(4 * np.pi)
        - 48.6
    )
    return m_AB

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
            #print(f"Warning: No match found for index {i} (RA={df_a.iloc[i][ra_col_a]}, DEC={df_a.iloc[i][dec_col_a]})")
            unmatched_object_ids.append(df_a.iloc[i]['object_id'])
    return result, unmatched_object_ids

from astropy.table import Table
from astropy.io.votable import parse

def populate_xray(df, table_fpath="data/cscresults.vot"):
    # Parse the VOTable
    vo = parse(table_fpath)
    table = vo.get_first_table().to_table()

    # Fix column names using FIELD 'name' attributes
    fields = vo.get_first_table().fields
    new_names = [f.name for f in fields]
    print(new_names)
    for old, new in zip(table.colnames, new_names):
        table.rename_column(old, new)

    # <-- this must be outside the loop
    df_csc = table.to_pandas()

    # Match to CSC and bring over flux + bounds
    df_matched, unmatched_object_ids = match_radec(
        df, df_csc,
        populate_cols=['flux_aper_b', 'flux_aper_hilim_b', 'flux_aper_lolim_b'],
        max_sep_arcsec=1.0
    )
    print(f"Matched {len(df_matched) - len(unmatched_object_ids)} out of {len(df)} objects to CSC3 catalog.")

    # Ensure numeric
    for c in ["flux_aper_b", "flux_aper_hilim_b", "flux_aper_lolim_b"]:
        df_matched[c] = pd.to_numeric(df_matched[c], errors="coerce")

    best = df_matched["flux_aper_b"]
    hi   = df_matched["flux_aper_hilim_b"]
    lo   = df_matched["flux_aper_lolim_b"]

    # Symmetric 1σ error from provided bounds (fallback to one-sided if needed)
    err = np.where(~hi.isna() & ~lo.isna(),
                   0.5 * (hi - lo),
                   np.where(~hi.isna(),
                            (hi - best),
                            np.where(~lo.isna(),
                                     (best - lo),
                                     np.nan)))
    df_matched["flux_aper_err_b"] = np.clip(err, a_min=0.0, a_max=None)

    # --- Cosmology & redshift array ---
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    z = pd.to_numeric(df_matched["z"], errors="coerce")
    DL_cm = cosmo.luminosity_distance(z.values).to('cm').value

    # --- Adopt photon index and K-correction per Eq. (5) ---
    GAMMA_X = 1.9  # requested
    # For energy flux in a fixed observed band: Kcorr = (1+z)^(Gamma - 2)
    Kcorr = np.power(1.0 + z.values, GAMMA_X - 2.0)  # = (1+z)^(-0.1) for Gamma=1.9

    # Pull flux and error
    xray_flux     = df_matched["flux_aper_b"].replace(0, np.nan).values
    xray_flux_err = df_matched["flux_aper_err_b"].replace(0, np.nan).values

    # Luminosity with K-correction (no absorption correction)
    L_xray = 4.0 * np.pi * (DL_cm**2) * xray_flux * Kcorr
    df_matched["log_Lxray"] = np.log10(L_xray)

    # Error on log L_xray (treat z as exact, so Kcorr & constants drop out)
    df_matched["log_Lxray_err"] = (1.0 / np.log(10.0)) * (xray_flux_err / xray_flux)

    # alpha_OX (using your existing 2500 Å columns)
    df_matched["alphaOX"] = -(df_matched["log_Lxray"] - df_matched["log_L2500_fs"]) / 2.605
    df_matched["alphaOX_int"] = -(df_matched["log_Lxray"] - df_matched["log_L2500_int_fs"]) / 2.605

    df_matched["alphaOX_err"] = np.sqrt(
        df_matched["log_Lxray_err"]**2 + df_matched["log_L2500_fs_err"]**2
    ) / 2.605

    df_matched["alphaOX_int_err"] = np.sqrt(
        df_matched["log_Lxray_err"]**2 + df_matched["log_L2500_int_fs_err"]**2
    ) / 2.605

    return df_matched


def populate_zquery(df, zquery_csv):
    fields = {
        'specObjID': str,
        'plate': str,
        'mjd': str,
        'fiberID': str,
        'z': float,
        'zErr': float,
        'zWarning': str,
        'class': str,
        'subClass': str,
    }
    # Load and concatenate two CSV files
    df_zquery = pd.read_csv(
        zquery_csv,
        dtype={'object_id': str},
        converters=fields
    )

    print("-------------------------- Z query report --------------------------")
    #print("Length of zquery csv file:", len(df_zquery))

    merged = df.merge(df_zquery, on='object_id', how='left', suffixes=('', '_zquery'))

    #print("Length of zquery merged DataFrame:", len(merged))
    missing_ids = set(df['object_id']) - set(df_zquery['object_id'])
    #print("object_id not in merged:", list(missing_ids))

    df['sameZ'] = np.isclose(merged['z'], merged['z_zquery'], atol=1e-1, equal_nan=True)
    not_sameZ = ~df['sameZ'].fillna(False)

    for col in fields.keys():
        if f'{col}_zquery' in merged.columns:
            df[f'{col}_zquery'] = merged[f'{col}_zquery']
        else:
            df[col] = merged[col]

    # Convert zWarning and sameZ to int, fill missing or NaN with -99
    for col in ['zWarning', 'sameZ']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-99).astype(int)

    # Convert 'class' and 'subClass' columns to numeric codes
    for col in ['class', 'subClass']:
        df[f'{col}_code'] = df[col].astype('category').cat.codes
        #print(f"Column '{col}' code conversion:")
        #print(dict(enumerate(df[col].astype('category').cat.categories)))

    return df

def populate_sdss_mags(df, 
                       sdss_mags_csv='results/data/nov2_sdss_mags.csv',
                       ps1_mags_csv='results/data/nov12_ps1_mags.csv'):
    fields_sdss = {
        'psfMag_u': float,
        'fiberMag_u': float,
        'petroRad_u': float,
        'psfMag_g': float,
        'fiberMag_g': float,
        'petroRad_g': float,        
        'psfMag_i': float,
        'fiberMag_i': float,
        'petroRad_i': float,
        'psfMag_r': float,
        'fiberMag_r': float,
        'petroRad_r': float,
        'psfMag_z': float,
        'fiberMag_z': float,
        'petroRad_z': float,
    }
    # SDSS magnitudes
    df_mags = pd.read_csv(
        sdss_mags_csv,
        dtype={'object_id': str},
        converters=fields_sdss
    )
    merged = df.merge(df_mags, on='object_id', how='left', suffixes=('', '_sdss'))

    print("Length of sdss mags merged DataFrame:", len(merged))
    missing_ids = set(df['object_id']) - set(df_mags['object_id'])
    print("object_id not in merged:", list(missing_ids))

    for col in fields_sdss.keys():
        if f'{col}_sdss' in merged.columns:
            df[f'{col}_sdss'] = merged[f'{col}_sdss']
        else:
            df[f'{col}_sdss'] = merged[col]

    for b in ['u', 'g', 'r', 'i', 'z']:
        df[f'psf_sdss_minus_fiber_sdss_{b}'] = df[f'psfMag_{b}_sdss'] - df[f'fiberMag_{b}_sdss']
    # df['log_psf_minus_fiber_r'] = np.log10(df['psf_minus_fiber_r'])
    # df['log_petroRad_r'] = np.log10(df['petroRad_r'])


    # PS1 magnitudes

    fields_ps1 = {
        'psfMag_g': float,
        'psfMag_i': float,
        'psfMag_r': float,
        'psfMag_z': float,
    }


    df_mags = pd.read_csv(
        ps1_mags_csv,
        dtype={'object_id': str},
        #converters=fields_ps1
    )
    merged = df.merge(df_mags, on='object_id', how='left', suffixes=('', '_ps1'))

    print("Length of sdss mags merged DataFrame:", len(merged))
    missing_ids = set(df['object_id']) - set(df_mags['object_id'])
    print("object_id not in merged:", list(missing_ids))

    for col in fields_ps1.keys():
        if f'{col}_ps1' in merged.columns:
            df[f'{col}_ps1'] = merged[f'{col}_ps1']
        else:
            df[f'{col}_ps1'] = merged[col]

    for b in ['g', 'r', 'i', 'z']:
        df[f'psf_ps1_minus_fiber_sdss_{b}'] = df[f'psfMag_{b}_ps1'] - df[f'fiberMag_{b}_sdss']

    return df

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

from ast import literal_eval
def parse_list(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if not s:
        return []
    # Try Python literal (e.g., "['g','r']") first (fast & permissive but safe)
    try:
        v = literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [str(t) for t in v]
    except Exception:
        pass
    # Try JSON (e.g., '["g","r"]')
    try:
        v = json.loads(s)
        if isinstance(v, (list, tuple)):
            return [str(t) for t in v]
    except Exception:
        pass
    # Fallback: comma-separated string (e.g., "g,r,i")
    return [t.strip() for t in s.split(",") if t.strip()]

def compute_apparent_mag_2500_astropy(
    logL2500,                 # log10(ν Lν) at 2500 Å (erg s^-1)
    logL2500_err,        # 1-σ uncertainty in log10(ν Lν), in dex
    z,
):
    cosmo   = FlatLambdaCDM(H0=70, Om0=0.3)
    c_cms   = 2.99792458e10     # cm/s
    lambda_cm = 2500e-8         # 2500 Å in cm

    logL2500 = np.asarray(logL2500, dtype=float)
    z = np.asarray(z, dtype=float)

    # Luminosity distance in cm
    DL_cm = cosmo.luminosity_distance(z).to(u.cm).value

    # Convert log10(ν Lν) -> log10 Lν (dex), then to log10 fν(obs)
    log_Lnu = logL2500 + np.log10(lambda_cm / c_cms)
    log_fnu = log_Lnu - np.log10(4 * np.pi * DL_cm**2)# * (1 + z))
    # AB magnitude
    m_ab = -2.5 * log_fnu - 48.60

    # Since m_ab = -2.5 * log_fnu and log_fnu depends linearly on log_Lnu,
    # σ_m from logL2500_err (dex) is simply:
    m_ab_err = 2.5 * np.abs(logL2500_err)

    return m_ab, m_ab_err

def populate_spectra_fit(df, spectra_fit_csvs, best=True):
    # Columns expected from spectral-fit CSVs (exclude 'object_id' from the drop list!)
    fields = {
        'object_id': str,                   # merge key (do not drop from df)
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
        'delta_m_avg': float,
        'alpha_lambda': float,
        'alpha_lambda_err': float,
        'redchi': float,
        'redchi2_conti_full': float,
        'aic': float,
        'bic': float,
        'npca_qso': int,
        'decomp_host': bool,
        'BC': bool,
        'poly': bool,
        'best': bool,
        'log_L2500_fs': float,
        'log_L2500_fs_err': float,
        'log_L2500_int_fs': float,
        'log_L2500_int_fs_err': float,
        'reddening_integral': float,
        'reddening_proxy': float,
        'bands_used': parse_list,
        'PL_slope_blue': float,
        'PL_slope_blue_err': float,
        'PL_slope_red': float,
        'PL_break_wave': float,
        'iron_frac': float,
        'PL_break_wave_inbounds': bool,
        'lam_rf_min': float,
        'lam_rf_max': float,
    }

    # Never drop the merge key
    drop_targets = [c for c in fields.keys() if c != "object_id"]
    existing_to_drop = [c for c in drop_targets if c in df.columns]
    if existing_to_drop:
        df = df.drop(columns=existing_to_drop)

    # Ensure left DF has the key and is string-typed
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

        # Normalize headers and ensure key
        df_spectra.columns = [_norm_name(c) for c in df_spectra.columns]
        df_spectra = _ensure_object_id(df_spectra)

        # Coerce booleans that may have come in as strings
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
            # Keep only best fits, then one-to-one merge and copy columns over
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

            # Update/insert columns for matched rows only
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
            # IMPORTANT: preserve EVERY spectra-fit row -> expand via one-to-many merge
            # We already dropped overlapping 'fields' from df, so no suffix collisions.
            expanded_df = expanded_df.merge(
                df_spectra,
                on="object_id",
                how="left",
                validate="one_to_many",
            )
            print("Length after expanding with one-to-many merge:", len(expanded_df))

    # Choose which frame to proceed with
    out = df if best else expanded_df

    # -------- Derived columns (guarded) on the chosen output --------
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

    # Optional save
    # out_csv = f"plots/hubble/{prefix}/merged.csv"
    # os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    # cols_to_save = ['object_id', 'sdss_name', 'apparent_mag_2500', 'f_host_2500', 'z'] + [c for c in fields if c != "object_id"]
    # cols_to_save_unique = [c for c in cols_to_save if c in out.columns]
    # out[cols_to_save_unique].to_csv(out_csv, index=False)
    # print(f"Saved merged DataFrame to {out_csv} with columns: {cols_to_save_unique}")
    
    out['alpha_nu'] = -out['alpha_lambda'] - 2
    out['alpha_nu_err'] = out['alpha_lambda_err']

    # out = out.drop(columns=['apparent_mag_2500', 'apparent_mag_2500_err'], errors='ignore')

    # m_2500, m_2500_err = compute_apparent_mag_2500_astropy(
    #     out['log_L2500_int_fs'],
    #     out['log_L2500_int_fs_err'],
    #     out['z'])
    # out['apparent_mag_2500'] = m_2500
    # out['apparent_mag_2500_err'] = m_2500_err


    # Keep a default error if needed
    # out['apparent_mag_2500_err'] = 0.1 * np.ones(len(out))

    return out

# Constants
c = 2.99792458e18  # speed of light in Angstrom/s


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
        d['sdss_name'] = fits_data['SDSS_NAME'][i]  # Extract SDSS_NAME
        d['log_lbol'] = -999.0
        if d['z'] < 0.7:
            d['log_lbol'] = np.log10(5.15) + fits_data['LOGL3000'][i]
            d['log_lbol_err'] = fits_data['LOGL3000_ERR'][i]
        else:
            d['log_lbol'] = fits_data['LOGLBOL'][i]  # Extract log Lbol values
            d["log_lbol_err"] = fits_data['LOGLBOL_ERR'][i]  # Extract log Lbol error values

        d['LOGMBH'] = fits_data['LOGMBH'][i]  # Extract log MBH values
        d['LOGMBH_ERR'] = fits_data['LOGMBH_ERR'][i]  # Extract log MBH error values
        d['LOGLEDD_RATIO'] = fits_data['LOGLEDD_RATIO'][i]  # Extract log L/edd values
        d['LOGLEDD_RATIO_ERR'] = fits_data['LOGLEDD_RATIO_ERR'][i]  # Extract log L/edd error values
        d['ebv_wu'] = fits_data['EBV'][i]
        d['sn_median_all'] = fits_data['SN_MEDIAN_ALL'][i]
        d['M_i'] = fits_data_2['M_I'][i]
        for b in ['u', 'g', 'r', 'i', 'z']:
            filters = {'u':0, 'g':1, 'r':2, 'i':3, 'z':4}
            d[f'PSFMAG_{b}'] = fits_data_2['PSFMAG'][i, filters[b]]
        # d['CIV'] = fits_data['CIV'][i, 0]
        # d['FEII_UV_EW'] = fits_data['FEII_UV_EW'][i]
        # d['FEII_OPT_EW'] = fits_data['FEII_OPT_EW'][i]
        # d['HBETA'] = fits_data['HBETA'][i, 0]
        # d['HALPHA'] = fits_data['HALPHA'][i, 0]

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


        # Other fields, not used
        # d['IF_BOSS_SDSS'] = fits_data['IF_BOSS_SDSS'][i]
        d['FHOST_5100'] = fits_data['FHOST_5100'][i]
        # d['BAL_PROB'] = fits_data_2['BAL_PROB'][i]
        d['EXTINCTION'] = fits_data_2['EXTINCTION'][i, filters['i']]
        # d['XMM_TOTAL_FLUX'] = fits_data_2['XMM_TOTAL_FLUX'][i]
        # d['XMM_HARD_FLUX'] = fits_data_2['XMM_HARD_FLUX'][i]
        # d['XMM_SOFT_FLUX'] = fits_data_2['XMM_SOFT_FLUX'][i]


        d['fhost'] = fits_data['FHOST_5100'][i]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for b in ['u', 'g', 'r', 'i', 'z']:
                d[f'apparent_mag_{b}'] = -2.5 * np.log10(fits_data_2['PSFFLUX'][i, filters[b]]) + 22.5
                d[f'apparent_mag_{b}_err'] = 2.5/np.log(10) * np.sqrt(1/fits_data_2['PSFFLUX_IVAR'][i, filters[b]])/fits_data_2['PSFFLUX'][i, filters[b]]
            # #     d[f'extinction_{b}'] = fits_data_2['EXTINCTION'][i, filters[b]]
            # d['color'] = -2.5 * np.log10(fits_data_2['PSFFLUX'][i, 0] / fits_data_2['PSFFLUX'][i, 3])
            # # Error propagation for color
            # flux_g = fits_data_2['PSFFLUX'][i, 0]
            # flux_i = fits_data_2['PSFFLUX'][i, 3]
            # ivar_g = fits_data_2['PSFFLUX_IVAR'][i, 0]
            # ivar_i = fits_data_2['PSFFLUX_IVAR'][i, 3]
            # # Convert inverse variance to variance, handle zero/negative ivar
            # var_g = 1.0 / ivar_g if ivar_g > 0 else np.inf
            # var_i = 1.0 / ivar_i if ivar_i > 0 else np.inf
            # Error propagation formula for color = -2.5 * log10(flux_g / flux_i)
            # σ_color^2 = (2.5/ln10)^2 * [ (σ_g/flux_g)^2 + (σ_i/flux_i)^2 ]
            # if flux_g > 0 and flux_i > 0 and np.isfinite(var_g) and np.isfinite(var_i):
            #     d['color_err'] = 2.5 / np.log(10) * np.sqrt(var_g / flux_g**2 + var_i / flux_i**2)
            # else:
            #     d['color_err'] = np.nan
        if any(issubclass(warning.category, RuntimeWarning) for warning in w):
            pass
            # print(f"RuntimeWarning occurred for z={d['z']:.2f} {d['sdss_name']} log_lbol={d['log_lbol']:.2f}")
            # print(fits_data_2['PSFFLUX'][i,:])
            # print(w)

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

def plot_m_vs_redshift(
    df_before, df_after, cut_info="", save_path=f"plots/hubble/{prefix}/cuts/"
):
    """
    Plot apparent_mag_2500 (AB) vs redshift in two panels:
      - Left: All (before) vs Kept (after)
      - Right: All (before) vs Removed (before - after)

    Notes
    -----
    * `bins` kept for backward compatibility (unused).
    * Expects columns: 'object_id', 'z', 'apparent_mag_2500'.
    """
    import os, re
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    if len(df_before) == len(df_after):
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Identify removed rows by object_id
    before_ids  = set(df_before['object_id'].astype(str))
    after_ids   = set(df_after['object_id'].astype(str))
    removed_ids = before_ids - after_ids
    df_removed  = df_before[df_before['object_id'].astype(str).isin(removed_ids)]

    # Extract finite data
    def _finite(df):
        z  = df['z'].to_numpy(dtype=float)
        m  = df['apparent_mag_2500'].to_numpy(dtype=float)
        ok = np.isfinite(z) & np.isfinite(m)
        return z[ok], m[ok]

    z_all,    m_all    = _finite(df_before)
    z_kept,   m_kept   = _finite(df_after)
    z_removed, m_removed = _finite(df_removed)

    # Axis limits shared across panels
    if z_all.size:
        x_min, x_max = np.nanmin(z_all), np.nanmax(z_all)
    else:
        x_min, x_max = 0.0, 1.0
    m_concat = np.concatenate([m_all, m_kept, m_removed]) if m_all.size else np.array([])
    if m_concat.size:
        y_min, y_max = np.nanmin(m_concat), np.nanmax(m_concat)
    else:
        y_min, y_max = 16.0, 28.0

    # Figure with two panels
    fig = plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.15)

    # Panel 1: All vs Kept
    ax1 = fig.add_subplot(gs[0])
    if z_all.size:
        ax1.scatter(z_all, m_all, s=6, alpha=0.8, c='blue', linewidths=0, label='All', rasterized=True)
    if z_kept.size:
        ax1.scatter(z_kept, m_kept, s=8, alpha=0.8, c='orange', linewidths=0, label='Kept', rasterized=True)
    ax1.set_xlabel("Redshift $z$")
    ax1.set_ylabel(r"$m_{2500}$ (AB)")
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(18, 30)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("All vs Kept")
    ax1.legend(loc="best", frameon=False)

    # Panel 2: All vs Removed
    ax2 = fig.add_subplot(gs[1])
    if z_all.size:
        ax2.scatter(z_all, m_all, s=6, alpha=0.8, c='blue', linewidths=0, label='All', rasterized=True)
    if z_removed.size:
        ax2.scatter(z_removed, m_removed, s=8, alpha=0.8, c='red', linewidths=0, label='Removed', rasterized=True)
    ax2.set_xlabel("Redshift $z$")
    ax2.set_ylabel(r"$m_{2500}$ (AB)")
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(18, 30)
    ax2.grid(True, alpha=0.3)

    ax2.set_title("All vs Removed")
    ax2.legend(loc="best", frameon=False)

    # Annotate cut info
    if cut_info:
        fig.text(0.5, 0.01, f"Cut info: {cut_info}", ha='center', va='bottom', fontsize=11, color='k')

    # Save
    safe_cut_info = re.sub(r'[^A-Za-z0-9._-]+', '_', str(cut_info)) if cut_info else ""
    filename = f"m2500_vs_z_cuts_{safe_cut_info}.png" if safe_cut_info else "m2500_vs_z_cuts.png"
    plot_path = os.path.join(os.path.dirname(save_path), filename)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(plot_path, dpi=150)
    plt.close()


def plot_redshift_histogram(df_before, df_after, bins=30, cut_info="", save_path=f"plots/hubble/{prefix}/cuts/"):
    """
    Plot a histogram of object counts vs redshift and save the figure.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'z' column for redshift.
    bins : int or sequence, optional
        Number of bins or bin edges for the histogram.
    save_path : str, optional
        Path to save the output figure.
    """
    import matplotlib.gridspec as gridspec
    
    if len(df_before) == len(df_after):
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Prepare data
    before_ids = set(df_before['object_id'].astype(str))
    after_ids = set(df_after['object_id'].astype(str))
    removed_ids = before_ids - after_ids
    df_removed = df_before[df_before['object_id'].astype(str).isin(removed_ids)]

    hist_before, bin_edges = np.histogram(df_before['z'].dropna(), bins=bins)
    hist_after, _ = np.histogram(df_after['z'].dropna(), bins=bin_edges)
    hist_removed, _ = np.histogram(df_removed['z'].dropna(), bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Create a figure with two panels side by side
    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    # Panel 1: Histogram of all vs kept
    ax1 = fig.add_subplot(gs[0])
    ax1.hist(df_before['z'].dropna(), bins=bin_edges, color='tab:blue', alpha=0.7, edgecolor='k', label='All')
    ax1.hist(df_after['z'].dropna(), bins=bin_edges, color='tab:orange', alpha=0.7, edgecolor='k', label='Kept')
    ax1.set_xlabel("Redshift (z)")
    ax1.set_ylabel("Number of objects")
    ax1.set_title("Histogram of Objects vs Redshift")
    ax1.legend()

    # Panel 2: Histogram of removed objects
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(df_before['z'].dropna(), bins=bin_edges, color='tab:blue', alpha=0.7, edgecolor='k', label='All')
    ax2.bar(bin_centers, hist_removed, width=np.diff(bin_edges), color='tab:red', alpha=0.7, edgecolor='k', label='Removed')
    ax2.set_xlabel("Redshift (z)")
    ax2.set_ylabel("Number removed")
    ax2.set_title("Removed Objects by Redshift")
    ax2.legend()

    # Annotate the cut_info as text on the figure
    if cut_info:
        fig.text(0.5, 0.01, f"Cut info: {cut_info}", ha='center', va='bottom', fontsize=12, color='k')

    plt.tight_layout()

    # Build the output file path using cut_info
    # Make cut_info safe for filenames (replace spaces and special chars)
    safe_cut_info = re.sub(r'[^A-Za-z0-9._-]+', '_', str(cut_info)) if cut_info else ""
    filename = f"redshift_histogram_{safe_cut_info}.png" if safe_cut_info else "redshift_histogram.png"
    plot_path = os.path.join(os.path.dirname(save_path), filename)
    plt.savefig(plot_path, dpi=150)
    plt.close()

def populate_chi_sq_from_csv(df, csv_path):
    """
    Populate the 'chi_sq' field in df by matching 'object_id' with the CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least an 'object_id' column.
    csv_path : str
        Path to the CSV file containing 'object_id' and 'chi_sq' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new/updated 'chi_sq' column.
    """
    chi_sq_df = pd.read_csv(csv_path)
    #print(chi_sq_df.head())
    # Ensure object_id is string for matching
    df['object_id'] = df['object_id'].astype(str)
    chi_sq_df['object_id'] = chi_sq_df['object_id'].astype(str)
    # Merge on object_id
    merged = pd.merge(df, chi_sq_df[['object_id', 'chi_sq_g', 'chi_sq_all']], on='object_id', how='left')
    # If chi_sq already exists, overwrite; else, add
    df['chi_sq_g'] = merged['chi_sq_g']
    df['chi_sq_all'] = merged['chi_sq_all']
    return df

def populate_magdiff(df):
        # Merge in PSF/fiber magnitude differences and overwrite chosen fields.
    # Edit this list to include the exact column names from the CSV you want to overwrite.
    fields_to_overwrite = ['PL_break_wave', 'lam_min', 'lam_max', 'PL_break_wave_inbounds']
    for b in ['u', 'g', 'r', 'i', 'z']:
        fields_to_overwrite.append(f'mag_psffiber_diff_{b}')

    psf_csv = "results/data/psffiber_mag_diff.csv"
    if os.path.exists(psf_csv):
        df_psf = pd.read_csv(psf_csv, dtype={"object_id": str})
        # normalize column names (trim whitespace)
        df_psf.columns = [c.strip() for c in df_psf.columns]

        # ensure both frames have object_id as str
        df["object_id"] = df["object_id"].astype(str)
        df_psf["object_id"] = df_psf["object_id"].astype(str)

        # select only requested fields that actually exist in CSV
        available = [f for f in fields_to_overwrite if f in df_psf.columns]
        missing_in_csv = [f for f in fields_to_overwrite if f not in df_psf.columns]
        if missing_in_csv:
            print(f"[populate] These requested fields were not found in {psf_csv}: {missing_in_csv}")

        if len(available) == 0:
            print(f"[populate] No requested overwrite fields found in {psf_csv}; nothing to merge.")
        else:
            merged = df.merge(
                df_psf[["object_id"] + available],
                on="object_id",
                how="left",
                suffixes=("", "_psf")
            )

            # For each available field, overwrite df where CSV provided a non-NaN value.
            for f in available:
                psf_col = f  # from CSV
                # use the merged column (will be present as psf_col since suffix only applies to conflicts)
                new_vals = merged[psf_col]
                mask_replace = new_vals.notna()
                n_replaced = int(mask_replace.sum())
                if f in df.columns:
                    df.loc[mask_replace, f] = new_vals[mask_replace].values
                else:
                    # column didn't exist in df: add it using CSV values for matched rows, NaN otherwise
                    df[f] = np.nan
                    df.loc[mask_replace, f] = new_vals[mask_replace].values
                print(f"[populate] Overwrote {n_replaced} values for column '{f}' from {psf_csv}.")

            # report overall matching stats
            n_matched = merged[available[0]].notna().sum() if len(available) else 0
            print(f"[populate] PSF CSV merge complete. At least one column matched for {n_matched} objects.")
    else:
        print(f"[populate] PSF CSV not found: {psf_csv}; skipping PSF merge.")

    return df


import numpy as np
import pandas as pd

def make_psf_minus_fiber_correction_fn(z, psf_minus_fiber_i, z_window=0.5):
    """
    Given arrays of z and psf_minus_fiber_i, returns two callables:

        dm_of_z(zq)      -> interpolated median(psf_minus_fiber_i) vs z
        dm_err_of_z(zq)  -> interpolated SEM of psf_minus_fiber_i vs z

    The medians and SEMs are computed in redshift bins using
    scipy.stats.binned_statistic. Interpolation is linear in z with
    flat extrapolation at the edges (conservative).
    """
    from scipy.stats import binned_statistic
    from scipy.interpolate import interp1d

    # Clean and sort
    d = (
        pd.DataFrame({'z': z, 'psf_minus_fiber_i': psf_minus_fiber_i})
        .dropna()
        .sort_values('z')
    )

    if len(d) < 5:
        raise ValueError("Not enough data points to build correction.")

    z_min = d['z'].min()
    z_max = d['z'].max()
    z_span = z_max - z_min

    if z_span <= 0:
        raise ValueError("z must span a non-zero range.")

    # Choose number of bins so that z_window ~ bin width
    nbins = max(5, int(np.ceil(z_span / z_window)))

    z_vals = d['z'].to_numpy()
    dm_vals = d['psf_minus_fiber_i'].to_numpy()

    # 1) Median in bins
    dm_binned, edges, _ = binned_statistic(
        z_vals,
        dm_vals,
        statistic='median',
        bins=nbins,
        range=(z_min, z_max),
    )

    # Bin centers
    z_centers = 0.5 * (edges[:-1] + edges[1:])

    # 2) Counts per bin (for SEM)
    counts, _, _ = binned_statistic(
        z_vals,
        dm_vals,
        statistic='count',
        bins=nbins,
        range=(z_min, z_max),
    )

    # 3) Standard deviation per bin
    std_binned, _, _ = binned_statistic(
        z_vals,
        dm_vals,
        statistic='std',
        bins=nbins,
        range=(z_min, z_max),
    )

    # SEM = std / sqrt(N) where N > 1; otherwise NaN
    sem_binned = np.where(counts > 1, std_binned / np.sqrt(counts), np.nan)

    # Mask out empty median bins
    mask_dm = np.isfinite(dm_binned)
    z_fit = z_centers[mask_dm]
    dm_fit = dm_binned[mask_dm]
    sem_fit = sem_binned[mask_dm]

    if len(z_fit) < 2:
        raise ValueError("Not enough non-empty bins to build correction function.")

    # Interpolator for the meditian correction (flat extrapolation at edges)
    from scipy.interpolate import interp1d
    dm_interp = interp1d(
        z_fit, dm_fit,
        kind="linear",
        bounds_error=False,
        fill_value=(dm_fit[0], dm_fit[-1]),
    )

    # For the error, drop NaN SEM bins (e.g. bins with only 1 object)
    mask_sem = np.isfinite(sem_fit)
    if np.sum(mask_sem) >= 2:
        z_fit_err = z_fit[mask_sem]
        sem_fit_err = sem_fit[mask_sem]

        dm_err_interp = interp1d(
            z_fit_err, sem_fit_err,
            kind="linear",
            bounds_error=False,
            fill_value=(sem_fit_err[0], sem_fit_err[-1]),
        )

        def dm_err_of_z(zq):
            """Interpolated SEM of psf_minus_fiber_i at given zq."""
            return dm_err_interp(np.asarray(zq, float))

    else:
        raise ValueError("Not enough non-empty SEM bins to build error function.")

    def dm_of_z(zq):
        """Interpolated psf_minus_fiber_i correction at given zq."""
        return dm_interp(np.asarray(zq, float))

    return dm_of_z, dm_err_of_z


def plot_m2500_correction(dm_of_z, z, m2500_uncorrected,
                          title="m2500 vs redshift (correction comparison)",
                          show=False, alpha=0.5, s=5):

    # Mask out m2500_uncorrected values outside the range [1, 30]
    valid_mask = (m2500_uncorrected >= 1) & (m2500_uncorrected <= 30)
    z = z[valid_mask]
    m2500_uncorrected = m2500_uncorrected[valid_mask]

    dm_values = dm_of_z(z)

    # Plot dm_of_z vs z
    plt.figure(figsize=(8, 6))
    plt.scatter(z, dm_values, label='dm_of_z', color='blue', s=0.5)
    plt.xlabel('Redshift (z)')
    plt.ylabel('dm_of_z')
    plt.title('dm_of_z vs Redshift')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(f"plots/hubble/{prefix}/diagnostics/", exist_ok=True)
    plt.savefig(f"plots/hubble/{prefix}/diagnostics/dm_of_z_vs_redshift.png", dpi=200)
    if show:
        plt.show()
    plt.close()

    m2500_corrected = m2500_uncorrected - dm_values
    mc = np.asarray(m2500_corrected, dtype=float)
    mu = np.asarray(m2500_uncorrected, dtype=float)

    # Basic validation
    if not (z.shape == mc.shape == mu.shape):
        raise ValueError("z, m2500_corrected, and m2500_uncorrected must have the same shape.")
    mask = np.isfinite(z) & np.isfinite(mc) & np.isfinite(mu)
    if mask.sum() < 3:
        raise ValueError("Not enough finite points to plot.")

    z, mc, mu = z[mask], mc[mask], mu[mask]
    dmag = mc - mu

    # Sort by redshift for optional line joins (keeps scatter as-is)
    idx = np.argsort(z)
    z_s, mc_s, mu_s, dmag_s = z[idx], mc[idx], mu[idx], dmag[idx]

    fig = plt.figure(figsize=(7.5, 7.5))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.0, 1.0], hspace=0.08)

    # Top: m vs z
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(z, mu, s=s, alpha=alpha, label="m2500 (uncorrected)")
    ax1.scatter(z, mc, s=s, alpha=alpha, label="m2500 (corrected)")

    ax1.set_ylabel(r"$m_{2500}$")
    ax1.set_title(title)
    ax1.legend(loc="best", frameon=False)
    ax1.grid(True, ls=":", alpha=0.3)

    # Bottom: residuals Δm
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.axhline(0.0, lw=1.0, color="k", alpha=0.6)
    ax2.scatter(z, dmag, s=s, alpha=alpha, label=r"$\Delta m = m_{2500}^{\rm corr} - m_{2500}^{\rm uncorr}$")
    ax2.plot(z_s, dmag_s, lw=0.8, alpha=0.5)

    ax2.set_xlabel("redshift z")
    ax2.set_ylabel(r"$\Delta m$")
    ax2.grid(True, ls=":", alpha=0.3)
    ax2.legend(loc="best", frameon=False)

    plot_path = f"plots/hubble/{prefix}/diagnostics/"
    os.makedirs(plot_path, exist_ok=True)
    fig.savefig(f"{plot_path}/m2500_correction_comparison.png", dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def load_agn_data(file_path, populate_sdss=False, apply_cut=True, fhost_cut=10,
                  exclude_object_ids_csv=[],
                  residuals_sigma_clip=None, residuals_csv=None,
                  spectra_fit_csv=None, zquery_csv=None, only_load=False,
                  iron_frac_cut=None, redchi2_cut=None,
                  sdss_mags_csv=None, lc_info_csv="data/aug4_sample_chisqg10_ebv005sn3_lcdata.csv"):
    #quasar_list = read_quasars_from_hdf5(file_path)
    import pickle
    with open(file_path + ".pkl", "rb") as f: 
        quasar_list = pickle.load(f)
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

    # ALWAYS Remove objects with apparent_mag_2500 or apparent_mag_i too bright or too faint
    mag_mask = ((df['apparent_mag_2500'] >= 16) & (df['apparent_mag_2500'] < 24))
    num_removed = np.sum(~mag_mask)
    print(f"Cut on apparent_mag_2500 : {num_removed} objects removed")
    plot_redshift_histogram(df.copy(), df[mag_mask], bins=30, cut_info="16 < apparent_mag_2500 < 24")
    plot_m_vs_redshift(df.copy(), df[mag_mask], cut_info="16 < apparent_mag_2500 < 24")
    df = df[mag_mask].reset_index(drop=True)

    if zquery_csv is not None:
        print("Populating zquery data from:", zquery_csv)
        df = populate_zquery(df, zquery_csv)
    else:
        print("[WARNING] zquery_csv not provided, assuming zquery fields are in agn h5 file")
        if 'zWarning' not in df.columns:
            print("[WARNING] zquery fields not in data, setting zWarning and sameZ to -99")
            df['zWarning'] = -99
            df['sameZ'] = -99

    df = populate_xray(df)

    df = populate_sdss_mags(df)
    
    if lc_info_csv is not None:
        print("Populating LC info from:", lc_info_csv)
        df = populate_lc_info(df, lc_info_csv)
    else:
        print("[WARNING] lc_info_csv not provided")

    num_quasars_z_0_1_before = len(df[(df['z'] > 0) & (df['z'] <= 1.0)])
    num_quasars_z_gt_3_before = len(df[df['z'] > 3])
    print("Number of quasars with 0 < z <= 1.0:", num_quasars_z_0_1_before)
    print("Number of quasars with z > 3:", num_quasars_z_gt_3_before)
    print("Highest redshift quasar:", df['z'].max())

    # Correct for psf - fiber magnitude differences using rolling median
    #dm_of_z = df["psf_sdss_minus_fiber_sdss_i"].values
    dm_of_z, dm_of_z_err = make_psf_minus_fiber_correction_fn(
    df["z"].values,
    df["psf_sdss_minus_fiber_sdss_i"].values,
    z_window=0.1,
    )
    plot_m2500_correction(dm_of_z, df["z"].values, df["apparent_mag_2500"].values, show=False)
    dm = dm_of_z(df["z"].values)
    df['dm_psf_correction'] = dm
    df['dm_psf_correction_err'] = dm_of_z_err(df["z"].values)
    #dm = df["psf_sdss_minus_fiber_sdss_r"].values
    
    df['apparent_mag_2500_uncorrectedpsf'] = df['apparent_mag_2500'].values
    df['apparent_mag_2500'] = df['apparent_mag_2500'].values - dm

    df_all = df.copy()

    if only_load:
        return df

    # Remove infinite values from numeric columns using a mask
    # numeric_cols = df.select_dtypes(include=[np.number]).columns
    # mask_finite = np.isfinite(df[numeric_cols]).all(axis=1)
    # num_removed = (~mask_finite).sum()
    # if num_removed > 0:
    #     print(f"Removing {num_removed} rows with inf/-inf in numeric columns")
    # df = df[mask_finite].reset_index(drop=True)

    # plot_redshift_histogram(df_all, df_all[mask_finite], bins=30, cut_info="remove_inf")


    #log_tau_band_RF = np.array([df[f'log_tau_band_{b}_RF'] for b in ['u', 'g', 'r', 'i', 'z']])
    # Mask values <= 0
    #masked_tau = np.where(log_tau_band_RF > 0, np.power(10, log_tau_band_RF), np.nan)
    #tau_band_RF_mean = np.nanmean(masked_tau, axis=0)

    #df['log_tau_band_RF_mean'] = np.log10(tau_band_RF_mean)
    df['log_t_rf_length'] = np.log10(df['t_rf_length'])
    #df['tau_band_RF_mean'] = tau_band_RF_mean
    #df['log_rho'] = np.log10(tau_band_RF_mean / df['t_rf_length'])

    df['log_f_host_2500'] = np.where(df['f_host_2500'] > 0, np.log10(df['f_host_2500']), np.nan)
    df['log_f_host_5100'] = np.where(df['f_host_5100'] > 0, np.log10(df['f_host_5100']), np.nan)
    # Replace NaNs with 0 in all columns
    #df = df.fillna(0)
    
    df = df.reset_index(drop=True)
    
    # Exclude objects whose object_id is in an exclusion list/array (if provided)
    exclusion_object_ids = []
    mask_exclude = ~df['object_id'].astype(str).isin(exclusion_object_ids)

    exclusion_sdss_names = [
        '221120.38+010905.6', # removed because wrong redshift
        '024555.35+005332.6' # remove because weird spectra
        '015802.36+002917.3' # next to star
                            ]
    mask_exclude &= (~df['sdss_name'].astype(str).isin(exclusion_sdss_names))
    print(f"Excluding {np.sum(~mask_exclude)} objects by exclusion list")
    df = df[mask_exclude].reset_index(drop=True)


    mask_valid = (df['log_tau_UV_RF'] > 2*df['log_sigma_UV'] + 2.5)
    num_removed = np.sum(~mask_valid)
    print(f"Cut on tau vs sigma diagram: {num_removed} objects removed")
    plot_redshift_histogram(df.copy(), df[mask_valid], bins=30, cut_info="tau > 2*sigma + 2.5")
    plot_m_vs_redshift(df.copy(), df[mask_valid], cut_info="tau > 2*sigma + 2.5")

    df = df[mask_valid].reset_index(drop=True)
    # mask_in  = df_agn["z"].between(0.44, 3.16)

    # Remove outliers by excluding object_ids listed in the outlier CSV
    for exclude_csv in exclude_object_ids_csv:
        if os.path.exists(exclude_csv):
            exclude_df = pd.read_csv(exclude_csv)
            exclude_ids = set(exclude_df['object_id'].astype(str))
            mask_exclude = ~df['object_id'].astype(str).isin(exclude_ids)
            num_excluded = np.sum(~mask_exclude)
            print(f"Excluding {num_excluded} objects from DataFrame based on {exclude_csv}")
            plot_redshift_histogram(df.copy(), df[mask_exclude], bins=30, cut_info="exclude csv")
            df = df[mask_exclude].reset_index(drop=True)
        else:
            print(f"[WARNING] Exclusion CSV not found: {exclude_csv}")




    # Remove objects with len_dropped_bands == 4 or 5
    mask_dropped = ~df['len_dropped_bands'].isin([4, 5])
    num_removed_dropped = np.sum(~mask_dropped)
    print(f"Removed {num_removed_dropped} objects with len_dropped_bands == 4 or 5")
    plot_redshift_histogram(df.copy(), df[mask_dropped], bins=30, cut_info="dropped bands 4 or 5")
    plot_m_vs_redshift(df.copy(), df[mask_dropped], cut_info="dropped bands 4 or 5")
    df = df[mask_dropped].reset_index(drop=True)

    mask = ~df['npca_qso'].isin([0])
    num_removed_dropped = np.sum(~mask)
    print(f"Removed {num_removed_dropped} objects with npca_qso = 0")
    plot_redshift_histogram(df.copy(), df[mask], bins=30, cut_info="npca_qso 0")
    plot_m_vs_redshift(df.copy(), df[mask], cut_info="npca_qso 0")
    df = df[mask].reset_index(drop=True)

    # Select objects with BC == False (if column exists) and report how many were dropped
    n_before = len(df)
    keep_mask = df['BC'] == False
    n_kept = int(np.sum(keep_mask))
    n_dropped = int(n_before - n_kept)
    print(f"Selecting BC==False: {n_dropped} objects removed (kept {n_kept} of {n_before})")
    plot_redshift_histogram(df.copy(), df[keep_mask], bins=30, cut_info=f"BCFalse")
    plot_m_vs_redshift(df.copy(), df[keep_mask], cut_info=f"BCFalse")
    df = df[keep_mask].reset_index(drop=True)

    n_before = len(df)
    keep_mask = df['poly'] == False
    n_kept = int(np.sum(keep_mask))
    n_dropped = int(n_before - n_kept)
    print(f"Selecting poly==False: {n_dropped} objects removed (kept {n_kept} of {n_before})")
    plot_redshift_histogram(df.copy(), df[keep_mask], bins=30, cut_info=f"polyFalse")
    plot_m_vs_redshift(df.copy(), df[keep_mask], cut_info=f"polyFalse")
    df = df[keep_mask].reset_index(drop=True)

    n_before = len(df)
    keep_mask = df['decomp_host'] == False
    n_kept = int(np.sum(keep_mask))
    n_dropped = int(n_before - n_kept)
    print(f"Selecting decomp_host==False: {n_dropped} objects removed (kept {n_kept} of {n_before})")
    plot_redshift_histogram(df.copy(), df[keep_mask], bins=30, cut_info=f"decomp_hostFalse")
    plot_m_vs_redshift(df.copy(), df[keep_mask], cut_info=f"polyFalse")
    df = df[keep_mask].reset_index(drop=True)

    # Convenience logs / coercions
    # def _safelog(a):
    #     return np.log10(np.abs(a + 1e-10)
    # if 'dm_red' in df_agn: df_agn['log_dm_red'] = _safelog(df_agn['dm_red'])
    # if 'reddening_integral' in df_agn: df_agn['log_reddening_integral'] = _safelog(df_agn['reddening_integral'])
    # if 'reddening_proxy' in df_agn: df_agn['log_reddening_proxy'] = _safelog(df_agn['reddening_proxy'])
    # if 'redchi' in df_agn: df_agn['log_redchi'] = _safelog(df_agn['redchi'])
    # if 'redchi2_conti_full' in df_agn: df_agn['log_redchi2_conti_full'] = _safelog(df_agn['redchi2_conti_full'])
    # if 'apparent_mag_2500_err' in df_agn: df_agn['log_apparent_mag_2500_err'] = _safelog(df_agn['apparent_mag_2500_err'])
    # if 'log_sigma_UV_err' in df_agn: df_agn['log_log_sigma_UV_err'] = _safelog(df_agn['log_sigma_UV_err'])
    # if 'log_tau_UV_RF_err' in df_agn: df_agn['log_log_tau_UV_RF_err'] = _safelog(df_agn['log_tau_UV_RF_err'])
    # if 'psf_minus_fiber_r' in df_agn: df_agn['log_psf_minus_fiber_r'] = _safelog(df_agn['psf_minus_fiber_r'])
    # if 'petroRad_r' in df_agn: df_agn['log_petroRad_r'] = _safelog(df_agn['petroRad_r'])
    # if 'log_tau_drw0_rhat' in df_agn: df_agn['log_log_tau_drw0_rhat'] = _safelog(df_agn['tau_drw0_rhat'])
    # for col in ['BC', 'decomp_host', 'poly']:
    #     if col in df_agn:
    #         df_agn[col] = df_agn[col].replace(
    #             {True: 1, False: 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0}
    #         )


    # mask = ~((df['z'] < 0.6) & (df['decomp_host'] == False))
    # num_removed = np.sum(~mask)
    # print(f"Cut on z < 0.6 and decomp_host False: {num_removed} objects removed")
    # plot_redshift_histogram(df.copy(), df[mask], bins=30, cut_info="z < 2 and m_2500 > 23")
    # plot_m_vs_redshift(df.copy(), df[mask], cut_info="z < 2 and m_2500 > 23")
    # df = df[mask].reset_index(drop=True)
    # mask = ~((df['z'] < 0.6) & (df['psf_minus_fiber_r'] < -0.3))
    # num_removed = np.sum(~mask)
    # print(f"Cut on z < 0.6 and psf_minus_fiber_r > -0.3: {num_removed} objects removed")
    # df = df[mask].reset_index(drop=True)

    # mask = ~((df['z'] < 0.6) & (df['iron_frac'] > 1.19))
    # num_removed = np.sum(~mask)
    # print(f"Cut on z < 0.6 and psf_minus_fiber_r > -0.3: {num_removed} objects removed")
    # df = df[mask].reset_index(drop=True)

    # mask = ~((df['z'] < 0.6) & (df['petroRad_r'] > 1.5))
    # num_removed = np.sum(~mask)
    # print(f"Cut on z < 0.6 and psf_minus_fiber_r > -0.3: {num_removed} objects removed")
    # df = df[mask].reset_index(drop=True)

    # mask = ~((df['z'] < 0.6) & (df['sn_median_all'] > 6))
    # num_removed = np.sum(~mask)
    # print(f"Cut on z < 0.6 and psf_minus_fiber_r > -0.3: {num_removed} objects removed")
    # df = df[mask].reset_index(drop=True)

    for b in ['u', 'g', 'r', 'i']:
        mask = (df[f'log_amp_delta_blr_{b}'] < 0) | df[f'log_amp_delta_blr_{b}'].isna()
        num_removed = np.sum(~mask)
        print(f"Removed {num_removed} objects with log_amp_delta_blr_{b} >= 0")
        plot_redshift_histogram(df.copy(), df[mask], bins=30, cut_info=f"log_amp_delta_blr_{b} < 0 or NaN")
        plot_m_vs_redshift(df.copy(), df[mask], cut_info=f"log_amp_delta_blr_{b} < 0 or NaN")
        df = df[mask].reset_index(drop=True)

    # Define cuts as (column, lower_limit, upper_limit)
    cuts = [
        #('z', 1, None),
        #('log_lbol', 45.4, None),
        #('dm_red', None, dm_red_cut),
        ('log_tau_UV_RF', 1.5, 4),
        #('redchi', None, 5),
        ('redchi2_conti_full', None, 1.2),
        #('apparent_mag_i', 12, 40),
        ('t_rf_length', 1700, None),
        # ('f_host_2500', -1, -1),
        # ('f_host_5100', -1, -1),
        ('alpha_lambda', None, -0.01),
        ('iron_frac', None, 10),
        #('log_amp_delta_blr_total', None, 0),
        #('eta_tau2', None, 1),
        ('log_tau_UV_RF_err', 0, 1.0),
        ('log_sigma_UV_err', 0, 0.3),

        #('eta_A1_rhat', None, 1.1),
        # ('apparent_mag_2500_err', None, 0.04),
        # ('npca_qso', -1, -1),
        # ('iron_frac', None, 1.2),
        #('PL_slope_blue', None, -1),
        # ('log_tau_fast0', None, 0.5),
        # ('bwb_alpha', 0.45, None),

        # ('eta_A1', -1, -0.5),
        # ('eta_tau1', 0, 1),
        # ('PL_slope_blue', -2, 1)




        #('conti_a_0', 0.0001, None),
 
        #('psf_minus_fiber_r', -0.3, None)
        # new cuts
        
        # ('eta_A1', -0.8, -0.5),

        # ('psf_minus_fiber_r', -0.35, -0.16),
        # ('log_petroRad_r', None, 0.4),
        # ('log_redchi', None, 0.3),
        



        # ('PL_slope_blue', None, -1),

        # ('bwb_alpha_z', 0.52, None),
        # ('bwb_alpha_i', 0.52, None),
        # ('bwb_alpha_r', 0.52, None),
        # ('bwb_alpha_u', 0.52, None),
        # ('redchi2_conti_full', 1.05, None),
        
        
        #('npca_qso', None, -1),
        
        #('apparent_mag_2500_err', 0, 1),
        #('z', None, 0.5),
        #('alpha_lambda', None, -1),
        # ('sameZ', 0.9, 1.1),
        #('log_tau_fast0', None, 0.5),
    ]

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
            plot_redshift_histogram(df.copy(), df[col_mask], bins=30, cut_info=f"{lower}<{col}<{upper}")
            plot_m_vs_redshift(df.copy(), df[col_mask], cut_info=f"{lower}<{col}<{upper}")
            mask &= col_mask
        df = df[mask]

        print(f"Total objects removed by all cuts: {initial_count - len(df)}")
    remove_nans_columns = ['alpha_lambda', 'alpha_lambda_err']
    for col in remove_nans_columns:
        nan_mask = ~df[col].isna()
        num_nans = (~nan_mask).sum()
        print(f"Removing {num_nans} objects with NaN in column '{col}'")
        df = df[nan_mask]
        plot_redshift_histogram(df.copy(), df[mask], bins=30, cut_info=f"{col} not NaN")

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
            plot_redshift_histogram(df.copy(), df[mask_residual], bins=30, cut_info=f"|mu_zscore|<{residuals_sigma_clip}")

            df = df[mask_residual].reset_index(drop=True)
        else:
            print(f"[WARNING] Residual CSV not found: {residuals_csv}")
            raise ValueError(f"Residual CSV not found: {residuals_csv}")

    # Drop rows where apparent_mag_2500_err is exactly zero or not finite
    num_before = len(df)
    mask = (df['apparent_mag_2500_err'] > 0) & np.isfinite(df['apparent_mag_2500_err'])
    plot_redshift_histogram(df.copy(), df[mask], bins=30, cut_info=f"0<apparent_mag_2500_err<inf")

    df = df[mask].reset_index(drop=True)
    num_after = len(df)
    print(f"Dropped {num_before - num_after} objects with apparent_mag_2500_err <= 0 or not finite")

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    y_log_meas_err = 0.4 * np.asarray(df['apparent_mag_2500_err'].fillna(1e9))
    actual_M2500 = df['apparent_mag_2500'] - cosmo.distmod(df['z']).value
    actual_logL2500 = convert_M2500_to_logL2500(actual_M2500)
    yerr_linear = 10**actual_logL2500 * np.log(10) * y_log_meas_err
    mask = yerr_linear/(10**actual_logL2500) < 0.5
    plot_redshift_histogram(df.copy(), df[mask], bins=30, cut_info=f"frac_err_logL2500<0.5")

    num_removed = np.sum(~mask)
    print(f"\033[93mRemoved {num_removed} objects with fractional logL2500 error >= 0.5\033[0m")
    df = df[mask]

    # mask = df['log_L2500_fs_err'] < 1
    # num_removed = np.sum(~mask)
    # print(f"\033[93mRemoved {num_removed} objects with log_L2500_fs_err error >= 1\033[0m")
    # df = df[mask]

    # Remove objects whose object_id is listed in the specified CSV file
    # remove_csv_path = "plots/hubble/oct27a_oct26a_oct28a_preview_fhost0_dev_redchi2.0cut_carma/Flatw0waCDM_joint_dev/remove.csv"
    # if os.path.exists(remove_csv_path):
    #     remove_df = pd.read_csv(remove_csv_path, dtype={'object_id': str})
    #     remove_ids = set(remove_df['object_id'].astype(str))
    #     mask_keep = ~df['object_id'].astype(str).isin(remove_ids)
    #     num_removed = np.sum(~mask_keep)
    #     print(f"Removed {num_removed} objects based on {remove_csv_path}")
    #     plot_redshift_histogram(df.copy(), df[mask_keep], bins=30, cut_info="remove.csv")
    #     df = df[mask_keep].reset_index(drop=True)
    # else:
    #     print(f"[WARNING] Remove CSV not found: {remove_csv_path}")

    # dusty = (df.get("conti_a_0", 0.0) > 0.08)
    # df = df[ ~((df["z"] < 1.0) & dusty & (df["poly"] == False)) ].copy()

    #df = df[~((df["z"]<1) & (df['dm_red'] < 0.6))].copy()

    num_quasars_z_0_1 = len(df[(df['z'] > 0) & (df['z'] <= 1.0)])
    num_quasars_z_gt_3 = len(df[df['z'] > 3])
    print("Number of quasars with 0 < z <= 1.0:", num_quasars_z_0_1)
    print("Number of dropped quasars with 0 < z <= 1.0:", num_quasars_z_0_1_before - num_quasars_z_0_1)
    print("Number of quasars with z > 3:", num_quasars_z_gt_3)
    print(f"\nTotal number of objects removed by all cuts: {len(df_all) - len(df)}")
    print("Final number of quasars:", len(df))
    plot_redshift_histogram(df_all.copy(), df.copy(), bins=30, cut_info=f"all cuts")
    plot_m_vs_redshift(df_all.copy(), df.copy(), cut_info=f"all cuts")

    
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

    # --- Load catalog ---
    print("Loading Pantheon+ supernova data...")
    df_pantheon = pd.read_csv(
        # "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat",
        "data/Pantheon+SH0ES.dat",
        sep=r"\s+"
    )
    
    # --- Selection mask: use only (zHD > 0.01) OR calibrators ---
    is_calib = np.asarray(df_pantheon["IS_CALIBRATOR"], dtype=bool)
    sel_mask = (df_pantheon["zHD"].values > 0.01) | is_calib

    # --- Load full covariance (stat+sys), then subset with the same mask ---
    print("Loading SN covariance matrix...")
    n_sn = len(df_pantheon)
    cov_flat = np.loadtxt(
        # "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov",
        "data/Pantheon+SH0ES_STAT+SYS.cov",
        skiprows=1
    )
    cov_matrix = cov_flat.reshape((n_sn, n_sn))
    assert cov_matrix.shape == (n_sn, n_sn), f"Expected ({n_sn},{n_sn}), got {cov_matrix.shape}"

    # --- Apply the SAME selection to covariance and dataframe ---
    cov_sel = cov_matrix[sel_mask][:, sel_mask]
    df_pantheon_sel = df_pantheon.loc[sel_mask].reset_index(drop=True)

    # --- Cholesky on the selected submatrix ---
    try:
        sna_L, sna_lower = cho_factor(cov_sel, lower=True)
    except np.linalg.LinAlgError:
        raise ValueError("Selected covariance submatrix is not positive-definite!")

    # log|C| = 2 * sum(log(diag(L))) for lower-triangular L from Cholesky
    sna_logdetCov = 2.0 * np.sum(np.log(np.diag(sna_L)))

    n_sel = cov_sel.shape[0]
    print(f"Cholesky factorization successful. "
          f"Selected SNe: {n_sel} / {n_sn} "
          f"(kept {(sel_mask).sum()}; dropped {n_sn - (sel_mask).sum()}).")

    return df_pantheon_sel, sna_logdetCov, sna_L, sna_lower




def soft_clip(x, floor=1e-5, sharpness=5):
    # Smoother logistic-like clipping
    return floor + (1 - floor) * (1 / (1 + np.exp(-sharpness * (x - floor))))

import numpy as np
from statistics import NormalDist

import os, math
import numpy as np
from statistics import NormalDist

LN10 = math.log(10.0)
LN2  = math.log(2.0)
TWOPI = 2.0 * math.pi
HALF_LN_TWOPI = 0.5 * math.log(TWOPI)
Phi = NormalDist()

def _log1pexp_pos(x):
    return x + math.log1p(math.exp(-x))

def _stable_logistic(x):
    return 0.5 * (1.0 + math.tanh(0.5 * x))

def _log_eps_from_delta_abs(absD):
    if absD < 40:
        return -math.log1p(math.exp(absD))
    return -(_log1pexp_pos(absD))

def _norm_isf_from_logeps(log_eps, use_phi_cut=-36.0):
    if log_eps > use_phi_cut:
        eps = math.exp(log_eps)
        return Phi.inv_cdf(1.0 - eps)
    L = -log_eps
    z = math.sqrt(2.0 * L)
    for _ in range(5):
        f = log_eps + 0.5 * z * z + math.log(z) + HALF_LN_TWOPI
        fp = z + 1.0 / z
        z -= f / fp
        if z <= 0 or not math.isfinite(z):
            z = max(1.0, math.sqrt(2.0 * L))
    return z

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

def _latex_escape_text(s: str) -> str:
    """Escape LaTeX special chars in plain text model names."""
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(s))

import os, math
import numpy as np

# Assumed available in your codebase (same helpers you already use):
#   _bayes_factor_repr_from_delta(delta_logZ, delta_logZ_err)
#   _stable_logistic(delta)
#   _log_eps_from_delta_abs(abs_delta)
#   _norm_isf_from_logeps(log_eps)
#   LN2  (= math.log(2.0))

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

def _odds_sigmas_from_delta(delta):
    """Return (one-sided Z, two-sided Z) from |Δln Z| odds, stably."""
    absD = abs(float(delta))
    log_eps = _log_eps_from_delta_abs(absD)
    log_eps_half = log_eps - LN2
    return (_norm_isf_from_logeps(log_eps),
            _norm_isf_from_logeps(log_eps_half))

def compare_models_by_log_evidence_all(
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
    # ---- Collect & validate ----
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

    # ---- Sort by ln Z (desc) ----
    items.sort(key=lambda t: t[1], reverse=True)
    labels = [t[0] for t in items]
    logZs  = np.array([t[1] for t in items], dtype=float)
    errs   = np.array([t[2] for t in items], dtype=float)

    top_label, top_logZ, top_err = items[0]
    preferred_model = top_label

    # ---- Per-model stats relative to TOP ----
    ranking = []
    for (label, z, e) in items:
        d = z - top_logZ   # <= 0 for all except top (0)
        de = float(np.hypot(e, top_err))
        z_mc = np.inf if de == 0 else d / de
        # odds-based sigma (one-/two-sided)
        z1, z2 = _odds_sigmas_from_delta(d)
        # Bayes factor repr and Jeffreys strength
        log10K, B_str, B_ci = _bayes_factor_repr_from_delta(d, de)
        strength = _jeffreys_strength(abs(d), jeffreys_thresholds)
        ranking.append({
            "model": label,
            "logZ": z,
            "logZerr": e,
            "delta_logZ_vs_top": d,
            "delta_logZ_err_vs_top": de,
            "z_mc_vs_top": z_mc,
            "sigma_one_sided_vs_top": z1,
            "sigma_two_sided_vs_top": z2,
            "jeffreys_strength_vs_top": strength,
            "log10_Bayes_factor_vs_top": log10K,
            "Bayes_factor_str_vs_top": B_str,
            "Bayes_factor_ci_1sigma_vs_top": B_ci,
        })

    # ---- Top vs Runner-up headline ----
    if len(items) >= 2:
        ru_label, ru_logZ, ru_err = items[1]
        delta = top_logZ - ru_logZ
        delta_err = float(np.hypot(top_err, ru_err))
        z_mc_head = np.inf if delta_err == 0 else delta / delta_err
        # odds to Z
        absD = abs(delta)
        log_eps = _log_eps_from_delta_abs(absD)
        log_eps_half = log_eps - LN2
        sigma_one = _norm_isf_from_logeps(log_eps)
        sigma_two = _norm_isf_from_logeps(log_eps_half)
        # CI via ±1σ on Δ
        def _odds_sigmas_at(d):
            return _odds_sigmas_from_delta(d)
        s1_lo, s2_lo = _odds_sigmas_at(delta - delta_err)
        s1_hi, s2_hi = _odds_sigmas_at(delta + delta_err)

        # Bayes factor & Jeffreys strength
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
            "sigma_from_odds_one_sided_ci_1sigma": (s1_lo, s1_hi),
            "sigma_from_odds_two_sided_ci_1sigma": (s2_lo, s2_hi),
            "log10_Bayes_factor": log10K,
            "Bayes_factor_str": B_str,
            "Bayes_factor_ci_1sigma": B_ci,
            "jeffreys_strength": strength,
            "decisive_zmc_ge_thresh": decisive,
        }
    else:
        top_vs_runnerup = None

    # ---- Full pairwise matrix ----
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
                z1, z2 = _odds_sigmas_from_delta(d)
                log10K, B_str, B_ci = _bayes_factor_repr_from_delta(d, de)
                strength = _jeffreys_strength(abs(d), jeffreys_thresholds)
                pairwise[li][lj] = {
                    "delta_logZ": d,
                    "delta_logZ_err": de,
                    "z_mc": zmc,
                    "sigma_one_sided": z1,
                    "sigma_two_sided": z2,
                    "jeffreys_strength": strength,
                    "log10_Bayes_factor": log10K,
                    "Bayes_factor_str": B_str,
                    "Bayes_factor_ci_1sigma": B_ci,
                }

    # ---- Human-readable summary ----
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
        )

    # Print & save
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
    # Create directory if it doesn't exist
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

from collections import defaultdict
import os

def make_cosmo_table_latex(
    results,
    *,
    include_lnZ=True,
    caption="Marginalized Cosmological Parameters and Bayesian Evidence",
    label="tab:cosmoparams",
    value_fmt="{:.3f}",
    write_path="plots/hubble/"
):
    """
    Build the LaTeX table string using exact keys: H0, Om0, w0, wa.
    results: list[dict] with keys: 'model', 'data', 'params', 'logZ'.
    """

    # ---------- local helpers (simple formatting only) ----------
    def _fmt_val(v):
        """Format v which can be a LaTeX string, a (mean,std) tuple, or None."""
        if v is None:
            return r"--"
        if isinstance(v, str):
            return v
        if isinstance(v, (tuple, list)) and len(v) == 2:
            m, s = v
            return rf"${value_fmt.format(m)} \pm {value_fmt.format(s)}$"
        return str(v)

    def _fmt_logZ(d):
        if not d:
            return r"--"
        return rf"${d['value']:.1f} \pm {d['err']:.1f}$"

    # ---------- columns ----------
    col_keys   = ["H0", "Om0", "w0", "wa"]
    col_labels = {
        "H0":  r"$H_0$\tnote{a}",
        "Om0": r"$\Omega_m$",
        "w0":  r"$w_0$",
        "wa":  r"$w_a$",
    }

    # ---------- model order ----------
    model_order = [r"flat $\Lambda$CDM", r"flat $w$CDM", r"flat $w_0w_a$CDM"]

    # ---------- external rows (fully combined; no extend/append) ----------
    external_rows = {
        r"flat $\Lambda$CDM": [
            {
                "data": r"Pantheon+ \& SH0ES",
                "params": {"H0": r"$73.6 \pm 1.1$", "Om0": r"$0.334 \pm 0.018$", "w0": r"$-1$", "wa": r"--"},
                "logZ": None,
            },
            {
                "data": r"DES-SN5YR",
                "params": {"H0": r"--", "Om0": r"$0.352 \pm 0.017$", "w0": r"$-1$", "wa": r"--"},
                "logZ": None,
            },
            {
                "data": r"Planck 2018",
                "params": {"H0": r"$67.66 \pm 0.42$", "Om0": r"$0.3111 \pm 0.0056$", "w0": r"$-1$", "wa": r"--"},
                "logZ": None,
            },
            {
                "data": r"DESI DR2",
                "params": {"H0": r"--", "Om0": r"$0.2975 \pm 0.0086$", "w0": r"$-1$", "wa": r"--"},
                "logZ": None,
            },
        ],
        r"flat $w$CDM": [
            {
                "data": r"Pantheon+ \& SH0ES",
                "params": {"H0": r"$73.5 \pm 1.1$", "Om0": r"$0.309^{+0.063}_{-0.069}$", "w0": r"$-0.90 \pm 0.14$", "wa": r"--"},
                "logZ": None,
            },
            {
                "data": r"DES-SN5YR",
                "params": {"H0": r"--", "Om0": r"$0.264^{+0.074}_{-0.096}$", "w0": r"$-0.80^{+0.14}_{-0.16}$", "wa": r"--"},
                "logZ": None,
            },
            {
                "data": r"DESI DR2",
                "params": {"H0": r"--", "Om0": r"$0.2969 \pm 0.0089$", "w0": r"$-0.916 \pm 0.078$", "wa": r"--"},
                "logZ": None,
            },
        ],
        r"flat $w_0w_a$CDM": [
            {
                "data": r"Pantheon+ \& SH0ES",
                "params": {"H0": r"$73.3 \pm 1.1$", "Om0": r"$0.403^{+0.054}_{-0.098}$", "w0": r"$-0.93 \pm 0.15$", "wa": r"$-0.1^{+0.9}_{-2.0}$"},
                "logZ": None,
            },
            {
                "data": r"DES-SN5YR",
                "params": {"H0": r"--", "Om0": r"$0.495^{+0.033}_{-0.043}$", "w0": r"$-0.36^{+0.36}_{-0.30}$", "wa": r"$-8.8^{+3.7}_{-4.5}$"},
                "logZ": None,
            },
            {
                "data": r"DESI DR2",
                "params": {"H0": r"--", "Om0": r"$0.352^{+0.041}_{-0.018}$", "w0": r"$-0.48^{+0.35}_{-0.17}$", "wa": r"$< -1.34$"},
                "logZ": None,
            },
        ],
    }


    # ---------- group results by model and prepend external rows ----------
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)
    for m in model_order:
        if m not in by_model:
            by_model[m] = []
        for row in external_rows.get(m, []):
            by_model[m].insert(0, {
                "model": m,
                "data": row["data"],
                "params": row["params"],
                "logZ": row.get("logZ"),
            })

    # ---------- build LaTeX ----------
    lines = []

    ncols = 1 + len(col_keys) + (1 if include_lnZ else 0)  # Dataset + params + optional lnZ
    lines.append(r"\begin{tabular}{" + "l" + "c" * (ncols - 1) + "}")
    lines.append(r"\toprule")

    header = ["Dataset"] + [col_labels[k] for k in col_keys]
    if include_lnZ:
        header.append(r"$\ln \mathcal{Z}$")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    row_order = [r"Pantheon+ \& SH0ES", r"DES-SN5YR", 
                 r"Planck 2018",
                 r"DESI DR2",
                 r"SN~Ia", r"SN~Ia + AGN"]

    for model in model_order:
        rows = by_model.get(model, [])
        if not rows:
            continue
        lines.append(rf"\multicolumn{{{ncols}}}{{l}}{{\underline{{\textbf{{{model}}}}}}} \\")
        rows = sorted(rows, key=lambda d: row_order.index(d["data"]) if d["data"] in row_order else 999)

        for r in rows:
            ds = r["data"]
            params = r.get("params", {})
            cells = [ds] + [_fmt_val(params.get(k)) for k in col_keys]
            if include_lnZ:
                cells.append(_fmt_logZ(r.get("logZ")))
            # Bold entire row for SN-only or SN+AGN rows
            if ds in (r"SN~Ia", r"SN~Ia + AGN"):
                cells = [rf"\textbf{{{c}}}" for c in cells]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines.pop()

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    lines.append(r"\begin{tablenotes}")
    lines.append(r"\item[a] Units: km s$^{-1}$ Mpc$^{-1}$.")
    lines.append(r"\end{tablenotes}")

    latex_str = "\n".join(lines)

    # write to file
    os.makedirs(os.path.dirname(write_path), exist_ok=True)
    write_path = os.path.join(write_path, "cosmo_table.tex")
    with open(write_path, "w") as f:
        f.write(latex_str)

    return latex_str


import numpy as np

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

    # Defensive checks
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

    # Compute mean/std for every parameter in the model
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

    # --- base params ---
    med = np.median(samples, axis=0)
    lo  = np.percentile(samples, 16, axis=0)
    hi  = np.percentile(samples, 84, axis=0)
    for name, m, l, h in zip(model_labels, med, lo, hi):
        print(f"{name:>15}: {m:.4f} (+{h - m:.4f}, -{m - l:.4f})")

    # --- helpers ---
    def _idx(labels, *cands):
        for c in cands:
            if c in labels:
                return labels.index(c)
        return None  # not found

    # --- derived summaries ---
    # Case 1: Flatw0waCDM -> derive w0 from (wp, wa) at pivot; print w0 and wa
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

    # Case 2: FlatwCDM (or any model that directly has w0/wa in samples)
    else:
        # Print w0 if present in samples
        i_w0 = _idx(model_labels, "w0", "w_0", "w")
        if i_w0 is not None:
            arr = samples[:, i_w0]
            m = np.median(arr); l = np.percentile(arr, 16); h = np.percentile(arr, 84)
            print(f"{'w0':>15}: {m:.4f} (+{h - m:.4f}, -{m - l:.4f})")

        # Print wa if present in samples (some models may include it explicitly)
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

    # Optional thinning for speed
    if (max_eval is not None) and (N > max_eval):
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(idx, size=max_eval, replace=False)
        samples = samples[idx]
        if weights is not None:
            weights = np.asarray(weights)[idx]

    ages = np.full(samples.shape[0], np.nan, dtype=float)

    # Compute ages sample-by-sample; skip pathological draws cleanly
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

            age = cosmo.age(0).to(u.Gyr).value  # age today in Gyr
            if np.isfinite(age) and (age > 0):
                ages[j] = age
        except Exception:
            # Unphysical / numerically problematic draw; leave as NaN
            continue

    m = np.isfinite(ages)
    n_valid = int(np.sum(m))
    n_total = ages.size
    n_invalid = n_total - n_valid
    if n_valid == 0:
        raise RuntimeError("All age evaluations failed; check parameter ranges.")

    a = ages[m]
    w = None if weights is None else np.asarray(weights)[m]

    # Summary stats
    mean = np.average(a, weights=w) if w is not None else float(np.mean(a))
    # For std, use weighted variance if weights are given
    if w is not None:
        w_norm = w / np.sum(w)
        var = np.average((a - mean) ** 2, weights=w_norm)
        std = float(np.sqrt(var))
    else:
        std = float(np.std(a, ddof=1))

    # Equal-tailed credible intervals
    q16, q50, q84 = _weighted_quantile(a, [0.16, 0.50, 0.84], w)
    q025, q975 = _weighted_quantile(a, [0.025, 0.975], w)

    # Pretty print (68% CI by default)
    print(f"Age of universe: {q50:.3f} (+{q84 - q50:.3f}/-{q50 - q16:.3f}) Gyr  "
          f"[mean={mean:.3f}±{std:.3f} Gyr; valid {n_valid}/{n_total}, skipped {n_invalid}]")

    return mean, std

def display_diagnostics(sampler, cosmo_model, fitting_method=False):
    priors, model_labels, _ = get_model_params(cosmo_model)
    if fitting_method == 'dynesty':
        # For dynesty, use weighted samples
        samples = sampler.results.samples
        weights = np.exp(sampler.results.logwt - sampler.results.logz[-1])
        # Resample to equal weights for diagnostics
        samples_equal = resample_equal(samples, weights)
        # ArviZ expects (chain, draw, param), so fake a single chain
        chain = samples_equal[np.newaxis, :, :]
        # No log_prob for dynesty, so skip
        idata = az.from_dict(posterior={name: chain[:, :, i] for i, name in enumerate(model_labels)})
    elif fitting_method == 'emcee':
        # --- Convergence diagnostics for emcee sampler ---
        # Extract chains and log probabilities
        chain = sampler.get_chain()  # shape: (nsteps, nwalkers, ndim)
        log_prob = sampler.get_log_prob()  # shape: (nsteps, nwalkers)
        # Transpose to match (chain, draw, dim) expected by ArviZ
        chain = np.transpose(chain, (1, 0, 2))
        log_prob = np.transpose(log_prob, (1, 0))
        idata = az.from_dict(
            posterior={name: chain[:, :, i] for i, name in enumerate(model_labels)},
            log_likelihood={"log_likelihood": log_prob}
        )
    else:
        raise ValueError("Unsupported fitting method. Use 'dynesty' or 'emcee'.")
    # Compute Rhat and ESS
    rhat = az.rhat(idata)
    ess = az.ess(idata)
    print("Gelman-Rubin Rhat diagnostic:")
    print(rhat)
    print("Effective Sample Size (ESS):")
    print(ess)
    print(az.summary(idata, round_to=3))

# --- Super-simple CMB likelihood: Gaussian on 100*theta_* ---

import numpy as np
from scipy import stats
from scipy.integrate import quad
import astropy.units as u
from astropy import constants as const
from astropy.table import Table as AstroTable

# --- DESI (Planck) prior on 100*theta_* ---
CMB_100THETA_MEAN = 1.04110
CMB_100THETA_SIGMA = 0.00053

# --- Defaults ---
STD_T_CMB = 2.7255
STD_NEFF  = 3.046
BBN_OMEGABH2_MEAN = 0.02218   # standard Neff BBN mean

# ---------- helpers ----------
def _safe_omegabh2(cosmo, fallback=BBN_OMEGABH2_MEAN):
    h = cosmo.H0.value / 100.0
    Ob0 = getattr(cosmo, "Ob0", None)
    if (Ob0 is None) or (float(Ob0) <= 0.0) or (not np.isfinite(Ob0)):
        return float(fallback)
    return float(Ob0) * h * h

def _omega_gamma_h2(Tcmb=STD_T_CMB):
    return 2.469e-5 * (Tcmb / 2.7255)**4

def _omega_r(cosmo, Tcmb=STD_T_CMB):
    """Total radiation density Ω_r = Ω_γ + Ω_ν,rel using Neff (massless approx)."""
    h = cosmo.H0.value / 100.0
    Om_gamma = _omega_gamma_h2(Tcmb=Tcmb) / (h*h)
    Neff = getattr(cosmo, "Neff", STD_NEFF)
    f_nu = 7.0/8.0 * (4.0/11.0)**(4.0/3.0) * float(Neff)  # ≈ 0.2271 * Neff
    return Om_gamma * (1.0 + f_nu)

def _z_star_hu_sugiyama(omega_b_h2, omega_m_h2):
    """Robust Hu–Sugiyama / Eisenstein–Hu z* fit with safe lower bounds."""
    wb = float(np.maximum(omega_b_h2, 1e-6))
    wm = float(np.maximum(omega_m_h2, 1e-6))
    g1 = 0.0783 * wb**(-0.238) / (1.0 + 39.5 * wb**0.763)
    g2 = 0.560 / (1.0 + 21.1 * wb**1.81)
    return 1048.0 * (1.0 + 0.00124 * wb**(-0.738)) * (1.0 + g1 * wm**g2)

def _r_s_comoving_robust(cosmo, zstar, omega_b_h2, Tcmb=STD_T_CMB, z_switch=1.0e4):
    """
    r_s(z*) in Mpc: numeric ∫ from z* to z_switch + analytic radiation-era tail above z_switch.
    This avoids endpoint issues and quad's extrapolation problems.
    """
    h = cosmo.H0.value / 100.0
    Om_b     = float(omega_b_h2) / (h*h)
    Om_gamma = _omega_gamma_h2(Tcmb=Tcmb) / (h*h)
    Om_r     = _omega_r(cosmo, Tcmb=Tcmb)

    # Numeric piece: z* -> z_switch
    c_si = const.c.to_value(u.m/u.s)
    pref_R = 0.75 * (Om_b / Om_gamma)  # R(z) = pref_R / (1+z)

    def integrand(z):
        a = 1.0 / (1.0 + z)
        R = pref_R * a
        c_s = c_si / np.sqrt(3.0 * (1.0 + R))
        Hz = cosmo.H(z).to_value(u.s**-1)
        return c_s / Hz  # [m]

    z_lo = float(zstar)
    z_hi = float(max(z_switch, z_lo * 1.01))  # ensure upper > lower
    rs_num_m, _ = quad(integrand, z_lo, z_hi, epsabs=0.0, epsrel=1e-6, limit=400)

    # Analytic tail: z_switch -> ∞ in pure radiation era with c_s ≈ c/sqrt(3), H ≈ H0 sqrt(Ω_r) (1+z)^2
    H0_SI = cosmo.H0.to_value(u.s**-1)
    rs_tail_m = (c_si / np.sqrt(3.0)) / (H0_SI * np.sqrt(Om_r)) * (1.0 / (1.0 + z_hi))

    rs_mpc = ((rs_num_m + rs_tail_m) * u.m).to_value(u.Mpc)
    return rs_mpc

# ---------- simple, stable CMB likelihood ----------
def loglike_cmb_theta_simple(cosmo,
                             omega_b_h2=None,
                             Tcmb=STD_T_CMB,
                             mean_100theta=CMB_100THETA_MEAN,
                             sigma_100theta=CMB_100THETA_SIGMA):
    """
    Gaussian likelihood on 100*theta_* with robust r_s and safe baryon fallback.
    Returns (lnL, diagnostics).
    """
    # baryons
    omega_b_h2 = _safe_omegabh2(cosmo) if (omega_b_h2 is None) else float(omega_b_h2)

    # z*, r_s, D_M
    h = cosmo.H0.value / 100.0
    omega_m_h2 = float(cosmo.Om0) * h * h
    zstar = _z_star_hu_sugiyama(omega_b_h2, omega_m_h2)

    D_A = cosmo.angular_diameter_distance(zstar).to_value(u.Mpc)
    if not np.isfinite(D_A) or D_A <= 0.0:
        # If the user feeds a pathological cosmology, bail out gracefully.
        return -np.inf, {"reason": "DA_nonfinite", "z_star": zstar}

    D_M = (1.0 + zstar) * D_A
    r_s = _r_s_comoving_robust(cosmo, zstar, omega_b_h2, Tcmb=Tcmb)

    hundred_theta = 100.0 * (r_s / D_M)
    ll = stats.norm.logpdf(hundred_theta, loc=mean_100theta, scale=sigma_100theta)

    return float(ll), {"100theta": hundred_theta, "z_star": zstar, "r_s_Mpc": r_s, "D_M_Mpc": D_M}



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

    # Current pivot; default z_p=0 -> a_p=1
    z_p_current = None
    for k in ('zp', 'z_p', 'z_pivot_agn', 'z_pivot'):
        if k in priors:
            z_p_current = priors[k] if np.isscalar(priors[k]) else float(priors[k])
            break
    if z_p_current is None:
        z_p_current = 0.0

    a_p = 1.0 / (1.0 + z_p_current)

    # Covariances
    C = np.cov(np.vstack([wp, wa]))
    cov_wp_wa = C[0, 1]
    var_wa    = C[1, 1]

    # Unconstrained optimal pivot (in scale factor)
    tiny = np.finfo(float).tiny
    a_p_star = a_p + cov_wp_wa / max(var_wa, tiny)

    # Enforce z >= z_min  <=>  a <= 1/(1+z_min)
    a_max_allowed = 1.0 / (1.0 + z_min)  # for z_min=0, this is 1
    # Enforce z <= z_max  <=>  a >= 1/(1+z_max)
    a_min_allowed = 1.0 / (1.0 + (z_max if z_max is not None else np.inf))

    if z_max is not None:
        a_p_star = max(a_p_star, a_min_allowed)
    a_p_star = min(a_p_star, a_max_allowed)

    # Guard against <= 0
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
        # Directly take w0 from samples
        i_w0 = _idx(model_labels, "w0", "w_0", "w")
        i_wa = _idx(model_labels, "wa", "w_a")
        w0 = flat_samples[:, i_w0]
        wa = flat_samples[:, i_wa]

    mask = np.isfinite(w0) & np.isfinite(wa)
    if not np.any(mask):
        raise ValueError("No finite samples to compute correlation.")

    return np.corrcoef(w0[mask], wa[mask])[0, 1]

def get_w0_wa_from_pivot(flat_samples, cosmo_model, z_p):
    """
    Convert posterior samples from (w_p, w_a) at a pivot redshift z_p
    into (w_0, w_a).

    Parameters
    ----------
    flat_samples : ndarray, shape (N, P)
        Flattened MCMC samples: N total draws by P parameters.
    cosmo_model : str
        Cosmological model name, used by get_model_params().
    z_p : float
        Pivot redshift.

    Returns
    -------
    w0 : ndarray, shape (N,)
        Posterior samples of w_0.
    wa : ndarray, shape (N,)
        Posterior samples of w_a.
    """
    # Get parameter labels for this cosmological model
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)

    # Locate wp and wa indices
    i_wp = model_labels.index("wp")
    i_wa = model_labels.index("wa")

    wp = flat_samples[:, i_wp]
    wa = flat_samples[:, i_wa]

    # Transform to w0 at the given pivot redshift
    w0 = wp - wa * (z_p / (1.0 + z_p))

    return w0, wa

import math
from typing import Tuple, Optional
from astropy.cosmology import FlatLambdaCDM

def _round_sig(x: float, sig: int) -> float:
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))

def format_value_uncertainty(
    median: float,
    err: Optional[float] = None,
    *,
    twosig_when=(1, 2, 3),
    sci: bool = True,
    latex: bool = True,
    unit: Optional[str] = None,
    max_decimals_no_sci: int = 2,          # <- cap decimals in non-sci mode
    prefer_fewer_decimals: bool = True     # <- use 1 sig fig if 2 would exceed the cap
) -> Tuple[float, Optional[float], int, str]:
    """
    Returns (value_rounded, err_rounded, exponent, formatted_string).
    - Uncertainty: 1 sig fig, except 2 when leading digit in twosig_when,
      but if that would exceed `max_decimals_no_sci` in non-sci mode, fall back to 1.
    - Value rounded to same decimal place as the (rounded) uncertainty.
    - No '±' when err is None.
    - Avoid sci for 0.1 <= |value| < 10 (plain decimals).
    - Special-case: fold 1×10^{-1} or 1×10^{-2} into 0.1 / 0.01.
    """
    # choose exponent (initial)
    if median == 0 or not math.isfinite(median):
        exponent = 0
    else:
        exponent = int(math.floor(math.log10(abs(median)))) if sci else 0

    # avoid sci in [0.1, 10)
    if sci and 0.1 <= abs(median) < 10:
        exponent = 0

    scale = 10.0 ** exponent if exponent != 0 else 1.0
    v = median / scale

    def decimals_needed(x: float) -> int:
        if x == 0: return 0
        return max(0, -int(math.floor(math.log10(abs(x)))))  # e.g. 0.02 -> 2

    if err is None:
        v_rounded = _round_sig(v, 3) if exponent != 0 else _round_sig(v, 3)
        e_rounded = None
    else:
        if err < 0 or not math.isfinite(err):
            raise ValueError("err must be finite and non-negative.")
        e = err / scale

        if e == 0:
            e_rounded = 0.0
            v_rounded = round(v, 6)
        else:
            # candidate with 2 sig figs?
            e1 = _round_sig(e, 1)
            lead = int(abs(e1) / (10 ** math.floor(math.log10(abs(e1)))))
            use_two = lead in twosig_when

            # if non-sci and two sig figs would exceed decimal cap, fall back to 1
            if use_two:
                e_two = _round_sig(e, 2)
                if exponent == 0 and prefer_fewer_decimals and decimals_needed(e_two) > max_decimals_no_sci:
                    use_two = False

            sig_unc = 2 if use_two else 1
            e_rounded = _round_sig(e, sig_unc)

            # round value to same decimal place as e_rounded
            digits = decimals_needed(e_rounded)
            v_rounded = round(v, digits)

            # re-enforce after potential order change
            e_rounded = _round_sig(e, sig_unc)
            digits = decimals_needed(e_rounded)
            v_rounded = round(v, digits)

    # special-case: (±)1×10^{-1 or -2} -> fold to decimal
    if sci and exponent in (-1, -2) and (abs(v_rounded) == 1.0):
        v_rounded *= 10 ** exponent
        if err is not None:
            e_rounded *= 10 ** exponent
        exponent = 0
        scale = 1.0

    # build string
    if latex:
        if exponent != 0:
            s = (f"{v_rounded}\\times 10^{{{exponent}}}"
                 if err is None else f"({v_rounded} \\pm {e_rounded})\\times 10^{{{exponent}}}")
        else:
            s = (f"{v_rounded}" if err is None else f"{v_rounded} \\pm {e_rounded}")
        if unit:
            s += f"\\,\\mathrm{{{unit}}}"
    else:
        if exponent != 0:
            s = (f"{v_rounded}×10^{exponent}"
                 if err is None else f"({v_rounded} ± {e_rounded})×10^{exponent}")
        else:
            s = (f"{v_rounded}" if err is None else f"{v_rounded} ± {e_rounded}")
        if unit:
            s += f" {unit}"

    val_out = v_rounded * (10 ** exponent if exponent != 0 else 1.0)
    err_out = (e_rounded * (10 ** exponent) if (err is not None and exponent != 0) else (e_rounded if err is not None else None))
    
    return s
    return val_out, err_out, exponent, s

def write_results_tex_variables(
    df_agn, flat_samples, cosmo_model, compare_r,
    write_path, chisq_dict=None, age=None
):
    """
    Write key cosmological parameters AND model comparison results
    to a LaTeX file as \newcommand definitions.
    """
    import os
    import numpy as np
    from itertools import combinations

    flat_samples = np.asarray(flat_samples)

    # --- AGN pivots
    obs_arr, err_arr, pivots_arr = agn_model_pack_obs(df_agn)
    log_sigma_UV_pivot  = pivots_arr[agn_model_oidx["log_sigma_UV"]]
    log_tau_UV_RF_pivot = pivots_arr[agn_model_oidx["log_tau_UV_RF"]]

    priors, model_labels, _ = get_model_params(cosmo_model)
    results = {key: sym_percentile(flat_samples[:, i])
               for i, key in enumerate(model_labels)}

    lines = []
    lines.append(r"% Auto-generated cosmological and evidence results")
    lines.append(r"% Do not edit manually; regenerated by write_results_tex_variables()")
    lines.append(r"\newcommand{\resultNumAGN}{%d}" % len(df_agn))
    if age is not None:
        age, age_err = age
    else:
        age, age_err = np.nan, np.nan
    lines.append(r"\newcommand{\resultAgeUniverse}{\ensuremath{%.2f\pm%.2f\,\mathrm{Gyr}}}" % (age, age_err))

    # ===============================
    # --- Model comparison results ---
    # ===============================
    if compare_r is not None:
        # Preferred model overall
        preferred = compare_r["preferred_model"]
        lines.append(r"\newcommand{\resultPreferredModelOverall}{%s}" % preferred)

        # Per-model stats relative to TOP
        for r in compare_r["ranking"]:
            model = r["model"]
            safe = model.replace("0", "Zero").replace("Λ", "Lambda")  # latex-safe key
            lines.append(r"\newcommand{\resultLogZ%s}{%.1f}" %
                         (safe, r["logZ"]))
            lines.append(r"\newcommand{\resultLogZerr%s}{%.1f}" %
                         (safe, r["logZerr"]))
            lines.append(r"\newcommand{\resultDeltaLogZ%s}{%.1f}" %
                         (safe, r["delta_logZ_vs_top"]))
            lines.append(r"\newcommand{\resultSigma%s}{%.1f}" %
                         (safe, r["sigma_two_sided_vs_top"]))
            lines.append(r"\newcommand{\resultJeffreysStrength%s}{%s}" %
                         (safe, r["jeffreys_strength_vs_top"]))

        # ---------- Helpers ----------
        def _latex_model_token(name: str) -> str:
            return {
                "Flatw0waCDM": "FlatwZeroWaCDM",
                "FlatwCDM": "FlatwCDM",
                "FlatLambdaCDM": "FlatLambdaCDM",
            }.get(name, name.replace("0", "Zero").replace("Λ", "Lambda"))

        def _get_pair(a: str, b: str):
            """Fetch pair dict for (a,b) regardless of direction."""
            pw = compare_r.get("pairwise", {})
            return pw.get(a, {}).get(b) or pw.get(b, {}).get(a)

        def _emit_pair(lines_list, a: str, b: str):
            pair = _get_pair(a, b)
            base = f"{_latex_model_token(a)}{_latex_model_token(b)}"
            if pair:
                lines_list.append(
                    r"\newcommand{\resultDeltaLogZ%s}{\ensuremath{%.1f \pm %.1f}}" %
                    (base, pair["delta_logZ"], pair["delta_logZ_err"])
                )
                lines_list.append(
                    r"\newcommand{\resultSigma%s}{%.1f}" %
                    (base, pair["sigma_two_sided"])
                )
                lines_list.append(
                    r"\newcommand{\resultJeffreysStrength%s}{%s}" %
                    (base, pair["jeffreys_strength"])
                )
                lines_list.append(
                    r"\newcommand{\resultZmc%s}{%.1f}" %
                    (base, pair["z_mc"])
                )
            else:
                lines_list.append(r"\newcommand{\resultDeltaLogZ%s}{N/A}" % base)
                lines_list.append(r"\newcommand{\resultSigma%s}{N/A}" % base)
                lines_list.append(r"\newcommand{\resultJeffreysStrength%s}{N/A}" % base)
                lines_list.append(r"\newcommand{\resultZmc%s}{N/A}" % base)

        # ---------- Iterate over all model pairs (no hardcoding) ----------
        models = [r["model"] for r in compare_r.get("ranking", [])]
        for a, b in combinations(models, 2):
            _emit_pair(lines, a, b)

    else:
        lines.append(r"\newcommand{\resultPreferredModelOverall}{N/A}")

    # ===============================
    # --- AGN relation results ---
    # ===============================
    lines.append(r"\newcommand{\resultAlphaAGN}{\ensuremath{%s}}" %
                 format_value_uncertainty(results['alpha_agn'][0], results['alpha_agn'][1]))
    lines.append(r"\newcommand{\resultBetaAGN}{\ensuremath{%s}}" %
                 format_value_uncertainty(results['beta_agn'][0], results['beta_agn'][1]))
    lines.append(r"\newcommand{\resultSigmaUVPivot}{\ensuremath{%.1f}}" %
                 10**log_sigma_UV_pivot)
    lines.append(r"\newcommand{\resultTauUVRFPivot}{\ensuremath{%.0f}}" %
                 10**log_tau_UV_RF_pivot)

    # Cosmological parameters
    lines.append(r"\newcommand{\resultOmZero}{\ensuremath{%s}}" % f"{results['Om0'][0]:.2f} \pm {results['Om0'][1]:.2f}")
    lines.append(r"\newcommand{\resultwZero}{\ensuremath{%s}}" % f"{results['w0'][0]:.2f} \pm {results['w0'][1]:.2f}")
    if cosmo_model in ('Flatw0waCDM', 'FlatwpwaCDM'):
        lines.append(r"\newcommand{\resultwa}{%s}" % f"{results['wa'][0]:.2f} \pm {results['wa'][1]:.2f}")


    # Derived intercepts
    M0_agn_samples = flat_samples[:, model_labels.index('M0_agn')]
    alpha_agn_samples = flat_samples[:, model_labels.index('alpha_agn')]
    beta_agn_samples  = flat_samples[:, model_labels.index('beta_agn')]
    alpha_AGN_L_samples = alpha_agn_samples * (-1/2.5)
    beta_AGN_L_samples  = beta_agn_samples  * (-1/2.5)
    L_intercept_samples = np.power(10, (90 - M0_agn_samples) / 2.5)

    lines.append(r"\newcommand{\resultLIntercept}{\ensuremath{%s}}" %
                 format_value_uncertainty(*sym_percentile(L_intercept_samples), unit=r"erg\,s^{-1}"))
    lines.append(r"\newcommand{\resultAlphaAGNL}{\ensuremath{%s}}" %
                 format_value_uncertainty(*sym_percentile(alpha_AGN_L_samples)))
    lines.append(r"\newcommand{\resultBetaAGNL}{\ensuremath{%s}}" %
                 format_value_uncertainty(*sym_percentile(beta_AGN_L_samples)))

    hd_scatter_samples = np.exp(flat_samples[:, model_labels.index('log_f')])
    lines.append(r"\newcommand{\resultScatterHD}{\ensuremath{%s}}" %
                 format_value_uncertainty(*sym_percentile(hd_scatter_samples), unit=r"mag"))
    l_scatter_samples = hd_scatter_samples / 2.5
    lines.append(r"\newcommand{\resultScatterL}{\ensuremath{%s}}" %
                 format_value_uncertainty(*sym_percentile(l_scatter_samples), unit=r"dex"))

    if chisq_dict is not None:
        for key, val in chisq_dict.items():
            lines.append(r"\newcommand{\result%sChiSqRed}{\ensuremath{%s}}" %
                         (key, format_value_uncertainty(val, None)))

    # --- Save file ---
    tex_path = os.path.join(write_path, "param_results.tex")
    os.makedirs(write_path, exist_ok=True)
    with open(tex_path, "w") as f:
        for line in lines:
            print(line)
            f.write(line + "\n")
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

def load_df_nearby():
    f_show = read_quasars_from_hdf5("results/data/oct21a_nearbylcs_1w4000s500t14co4ch4.h5")
    df_show = pd.DataFrame(df_show)

    # Load the CSV file
    df_nearby_csv = pd.read_csv("data/nearby_lcs/nearby_lcs_m2500.csv")

    # Drop columns from df_show that are also present in df_nearby_csv except 'object_id'
    columns_to_drop = [col for col in df_show.columns if col in df_nearby_csv.columns and col != 'object_id']
    df_show = df_show.drop(columns=columns_to_drop)
    # Merge with df_show on 'object_id'
    df_show = df_show.merge(df_nearby_csv, on='object_id', how='left')

    # Overwrite columns from df_nearby_csv to df_show
    for col in df_nearby_csv.columns:
        if col != 'object_id':  # Skip the 'object_id' column
            df_show[col] = df_show[col].fillna(df_nearby_csv.set_index('object_id')[col])

    #df_show = df_show[~df_show['object_id'].isin(['ngc4395', 'ucg06728', 'ngc4593'])]
    bad_lcs = ['ngc4395', 'ucg06728', 'ngc4593']
    df_show['bad'] = df_show['object_id'].isin(bad_lcs)

    return df_show

def make_agn_latex_table(
    agn_df,
    mu,
    mu_err,
    dm_interp,
    sort_by = None, ascending = True,
    max_rows = None,
    write_path = f"plots/hubble/{prefix}"
) -> str:

    def _is_bad(x):
        return x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))

    def fmt_num(x, nd):
        return r"$\dots$" if _is_bad(x) else rf"${float(x):.{nd}f}$"

    def fmt_signed_dec(x, nd):
        return r"$\dots$" if _is_bad(x) else f"${float(x):+.{nd}f}$"

    def name_to_bold(nm):
        s = str(nm).replace("-", "$-$")
        return rf"\textbf{{J{s}}}"

    def fmt_with_sym_err(row, base_col, nd_val, nd_err, err_col=None):
        v = row[base_col]
        if _is_bad(v):
            return r"$\dots$"
        v = float(v)
        if err_col is None:
            err_col = f"{base_col}_err"
        if err_col in row and not _is_bad(row[err_col]):
            e = abs(float(row[err_col]))
            return rf"${v:.{nd_val}f} \pm {e:.{nd_err}f}$"
        return rf"${v:.{nd_val}f}$"

    df = agn_df.copy()

    df['mu'] = mu
    df['mu_err'] = mu_err

    #dm_interp = make_dm_function(np.array(df["apparent_mag_2500"].values), np.array(df['z'].values), dms)
    pts = np.column_stack([df['z'], df['apparent_mag_2500']])
    m_2500_corr = (df['apparent_mag_2500'] - dm_interp(pts))
    df['apparent_mag_2500_corr'] = m_2500_corr
    df['apparent_mag_2500_corr_err'] = df['apparent_mag_2500_err']

    if max_rows is not None:
        df = df.sample(n=max_rows, random_state=42)
    if sort_by is not None:
        df = df.sort_values(sort_by, ascending=ascending)

    lines = [
        #r"\begin{adjustbox}{max width=\textwidth, max totalheight=\textheight, keepaspectratio}",
        r"\begin{tabular}{@{}lccccccccc@{}}",
        r"\hline\hline",
        r"\textbf{SDSS Name} & RA & Dec & $z$ & $m_{2500}$ & $m_{2500}^{\mathrm{uncorr}}$ & $\mu$ & $\log\tau_{\mathrm{UV,RF}}$ & $\log\sigma_{\mathrm{UV}}$ & $\mathrm{Cov}(\log\sigma_{\mathrm{UV}},\,\log\tau_{\mathrm{UV,RF}})$ \\",
        r"& (deg) & (deg) &  & (mag) & (mag) & (mag) & (days) & (mag) &  \\"
        r"\hline",
    ]

    for _, row in df.iterrows():
        nm   = name_to_bold(row['sdss_name'])
        ra   = fmt_num(row['ra'], 4)
        dec  = fmt_signed_dec(row['dec'], 4)

        zz   = fmt_with_sym_err(row, 'z', 4, 4)
        m25v_corr = fmt_with_sym_err(row, 'apparent_mag_2500_corr', 2, 2)
        m25v_uncorr = fmt_with_sym_err(row, 'apparent_mag_2500', 2, 2)
        mu_str  = fmt_with_sym_err(row, 'mu',            2,    2)

        tau_str = fmt_with_sym_err(row, 'log_tau_UV_RF', 2, 2, err_col='log_tau_UV_RF_std_psd')
        sig_str = fmt_with_sym_err(row, 'log_sigma_UV',  2, 2, err_col='log_sigma_UV_std_psd')
        tau_sig_cov = fmt_num(row['log_sigma_UV_log_tau_UV_RF_cov_psd'], 3)

        lines.append(
            f"{nm} & {ra} & {dec} & {zz} & {m25v_corr} & {m25v_uncorr} & {mu_str} & {tau_str} & {sig_str} & {tau_sig_cov} \\\\"
        )

    lines += [
        r"\hline",
        r"\end{tabular}%",                   
        #r"\end{adjustbox}"
    ]

    latex_str = "\n".join(lines)
    os.makedirs(os.path.dirname(write_path), exist_ok=True)
    out_path = os.path.join(write_path, "agn_table.tex")
    with open(out_path, "w") as f:
        f.write(latex_str)
    return latex_str