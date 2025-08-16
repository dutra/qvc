import matplotlib.pyplot as plt
import os
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
from astroquery.vizier import Vizier
from tqdm import tqdm
import warnings
from scipy import stats
from scipy.stats import norm, sigmaclip, multivariate_normal
from scipy.interpolate import RegularGridInterpolator
import arviz as az
from dynesty.utils import resample_equal
from hubble_model import get_model_params, M_model_agn, M_model_agn_err, M_model_SN, K_corr
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.stats import linregress
from scipy.stats import pearsonr

bands = ['u', 'g', 'r', 'i', 'z']#, 'y']
bands_idx = {b: i for i, b in enumerate(bands)}
filters = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5} # harcoded filter order for SDSS

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
        AGN data with columns: 'z', 'alpha_nu', 'log_sigma0', 'log_tau_UV_RF'.
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
        delta_K = K_corr(z_data, alpha_nu) - K_corr(z_pivot, alpha_nu)

        M_pred_samples = np.array([
            apparent_mag_2500 - delta_K - M_model_agn(
                s[param_indices['M0_agn']],
                s[param_indices['log_sigma0_break']],
                s[param_indices['eta_A1_agn']],
                s[param_indices['eta_A2_agn']],
                s[param_indices['eta_break_agn']],
                s[param_indices['beta_agn']],
                df_agn['log_sigma0'].values,
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
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    DL_cm = cosmo.luminosity_distance(z).to('cm').value
    m_AB = (
        -2.5 * log_nuLnu
        + 5 * np.log10(DL_cm)
        + 2.5 * np.log10(4 * np.pi)
        - 48.6
    )
    return m_AB

def compute_apparent_mag_2500_colin(df):
    # Load Colin's SDSS QSO 2500A magnitudes and merge with df on SDSS_NAME
    fields = {
            'f_host_4200': float,
            'apparent_mag_2500': float,
            'apparent_mag_2500_err': float,
            'alpha_lambda': float,
            'alpha_lambda_err': float,
            'redchi': float
        }
    # Load and concatenate two CSV files
    colin_df1 = pd.read_csv(
        'data/sample_stone_fittedm2500.csv',
        dtype={'object_id': str},
        converters=fields
    )
    colin_df2 = pd.read_csv(
        'data/sample_chisqg10_ebv005sn3_fittedm2500.csv',
        dtype={'object_id': str},
        converters=fields
    )
    # Merge on object_id, giving priority to colin_df2 values where available
    colin_df = pd.merge(
        colin_df1, colin_df2, 
        on='object_id', 
        how='outer', 
        suffixes=('', '_2')
    )
    # For each field, prefer colin_df2 value if present, else colin_df1
    for col in fields.keys():
        colin_df[col] = colin_df[f"{col}_2"].combine_first(colin_df[col])
        if f"{col}_2" in colin_df:
            colin_df.drop(columns=[f"{col}_2"], inplace=True)
    
    # Discard rows with apparent_mag_2500_err <= 0
    #colin_df = colin_df[colin_df['apparent_mag_2500_err'] > 0].reset_index(drop=True)
    
    # Fill apparent_mag_2500_err == 0 with mean of nonzero errors
    mean_err = colin_df.loc[colin_df['apparent_mag_2500_err'] > 0, 'apparent_mag_2500_err'].mean()
    colin_df.loc[colin_df['apparent_mag_2500_err'] == 0, 'apparent_mag_2500_err'] = mean_err

    print("Length of colin_df:", len(colin_df))
    print("Number with apparent_mag_2500 > 0:", np.sum(colin_df['apparent_mag_2500'] > 0))
    # Merge on SDSS_NAME, bring in apparent_mag_2500
    merged = df.merge(colin_df, on='object_id', how='left', suffixes=('', '_colin'))
    print("Length of merged DataFrame:", len(merged))
    missing_ids = set(df['object_id']) - set(colin_df['object_id'])
    print("object_id not in merged:", list(missing_ids))
    for col in fields.keys():
        df[col] = merged[col]
    return df


def compute_apparent_mag_2500(df, logL_col='MY_LOGL2500', logL_err_col='MY_LOGL2500_ERR', z_col='z', H0=70, Om0=0.3):
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    c = 2.99792458e10  # cm/s
    lambda_ = 2500e-8  # cm

    z = df[z_col].values
    logL_2500 = df[logL_col].values
    logL_2500_err = df[logL_err_col].values

    DL = cosmo.luminosity_distance(z).to(u.cm).value  # cm

    log_Lnu = logL_2500 + np.log10(lambda_ / c)
    log_fnu = log_Lnu - np.log10(4 * np.pi * DL**2 * (1 + z))
    m_ab = -2.5 * log_fnu - 48.60
    m_ab_err = 2.5 * logL_2500_err

    df['apparent_mag_2500'] = m_ab
    df['apparent_mag_2500_err'] = m_ab_err
    return df


def calculate_all_alpha_nu(df):
    """
    Estimate alpha_nu from total monochromatic luminosities L (in erg/s),
    propagating errors from LOGLxxxx_ERR columns.

    Assumes columns LOGLxxxx contain log10(lambda * L_lambda) and their corresponding errors LOGLxxxx_ERR.

    Fits:
        log10(L_lambda) ∝ -(alpha_nu + 2) * log10(lambda)

    So:
        alpha_nu = -slope - 2

    Error propagation based on linear fit uncertainty.
    """
    # Wavelengths in Angstroms
    band_waves = {
        'LOGL1350': 1350,
        'LOGL1700': 1700,
        'LOGL2500': 2500,
        'LOGL3000': 3000,
        'LOGL5100': 5100,
    }

    valid_bands = [k for k, v in band_waves.items() if 1216 <= v <= 5200]
    waves = np.array([band_waves[k] for k in valid_bands])  # in Å
    log_lambda = np.log10(waves)

    alpha_nu_list = []
    alpha_nu_err_list = []

    for _, row in df.iterrows():
        logL_total = np.array([row[k] for k in valid_bands])
        logL_err = np.array([row[f'{k}_ERR'] for k in valid_bands])

        mask = np.isfinite(logL_total) & (logL_total > 0) & np.isfinite(logL_err) & (logL_err > 0)

        if mask.sum() >= 3:
            logL_lambda = logL_total[mask] - log_lambda[mask]
            logL_lambda_err = logL_err[mask]

            p, cov = np.polyfit(log_lambda[mask], logL_lambda, 1, w=1/logL_lambda_err, cov=True)
            slope_err = np.sqrt(np.diag(cov))[0]

            alpha_nu = -p[0] - 2
            alpha_nu_err = slope_err

        elif mask.sum() == 2:
            logL_lambda = logL_total[mask] - log_lambda[mask]
            logL_lambda_err = logL_err[mask]

            # Simple two-point fit without covariance
            slope = (logL_lambda[1] - logL_lambda[0]) / (log_lambda[1] - log_lambda[0])
            slope_err = np.sqrt(logL_lambda_err[0]**2 + logL_lambda_err[1]**2) / abs(log_lambda[1] - log_lambda[0])

            alpha_nu = -slope - 2
            alpha_nu_err = slope_err

        else:
            alpha_nu = np.nan
            alpha_nu_err = np.nan

        alpha_nu_list.append(alpha_nu)
        alpha_nu_err_list.append(alpha_nu_err)
    df['alpha_nu'] = alpha_nu_list
    df['alpha_nu_err'] = alpha_nu_err_list
    df['alpha_nu'] = -0.5
    df['alpha_nu_err'] = 0.1  # Default values if no valid data

    return df

def compute_MY_LOGL2500(df):
    """
    Compute MY_LOGL2500 and its propagated uncertainty from available LOGLxxxx bands and alpha_nu.

    Parameters:
    - df: DataFrame with columns LOGLxxxx, LOGLxxxx_ERR, and alpha_nu

    Returns:
    - Two pandas Series:
        MY_LOGL2500      : mean log10 L_2500 in erg/s
        MY_LOGL2500_ERR  : propagated uncertainty [dex]
    """
    lambda_target = 2500  # Å
    log_lambda_target = np.log10(lambda_target)

    bands = {
        'LOGL1350': 1350,
        'LOGL1700': 1700,
        'LOGL2500': 2500,  # included directly
        'LOGL3000': 3000,
        'LOGL5100': 5100,
    }

    log_lambda_bands = {band: np.log10(lam) for band, lam in bands.items()}

    logL_vals = []
    logL_errs = []

    for _, row in df.iterrows():
        alpha = row.get('alpha_nu', np.nan)
        if not np.isfinite(alpha):
            logL_vals.append(np.nan)
            logL_errs.append(np.nan)
            continue

        est_list = []
        var_list = []

        for band, lam in bands.items():
            logL = row.get(band, np.nan)
            logL_err = row.get(f"{band}_ERR", np.nan)

            if np.isfinite(logL) and logL > 0 and np.isfinite(logL_err) and logL_err > 0:
                delta = log_lambda_target - log_lambda_bands[band]
                logL2500 = logL + (-(alpha + 1)) * delta
                logL2500_err = logL_err  # Only propagate observational error

                est_list.append(logL2500)
                var_list.append(logL2500_err**2)

        if len(est_list) == 0:
            logL_vals.append(np.nan)
            logL_errs.append(np.nan)
        else:
            # Inverse-variance weighted average
            weights = 1.0 / np.array(var_list)
            avg = np.sum(weights * est_list) / np.sum(weights)
            err = np.sqrt(1.0 / np.sum(weights))

            logL_vals.append(avg)
            logL_errs.append(err)

    return (
        pd.Series(logL_vals, index=df.index, name='MY_LOGL2500'),
        pd.Series(logL_errs, index=df.index, name='MY_LOGL2500_ERR')
    )


# Constants
c = 2.99792458e18  # speed of light in Angstrom/s


def calc_Mi_from_M2500(M_2500, alpha_nu, z):
    """
    Compute SDSS i-band absolute magnitude at observed frame (z) from M_2500.
    Assumes f_nu ∝ ν^alpha_nu.
    
    Parameters
    ----------
    M_2500 : array-like
        Absolute magnitude at rest-frame 2500 Å (AB system)
    alpha_nu : array-like
        Power-law slope of the quasar spectrum (f_nu ∝ ν^alpha)
    z : array-like
        Redshift of each source

    Returns
    -------
    M_i_z : ndarray
        Absolute magnitude in observed-frame SDSS i-band
    """

    # Load SDSS i-band filter curve
    df_filter = pd.read_csv("data/sdss_i.dat", sep=r'\s+', skiprows=6, header=None,
                            names=['wavelength', 'throughput_1', 'throughput_2', 'throughput_3', 'atm_trans'])

    wavelengths = df_filter['wavelength'].values  # Angstrom
    transmission = df_filter['throughput_1'].values

    lambda_2500 = 2500.0  # Angstrom

    # Convert inputs to arrays
    M_2500 = np.atleast_1d(M_2500).astype(float)
    alpha_nu = np.atleast_1d(alpha_nu).astype(float)
    z = np.atleast_1d(z).astype(float)

    # Check that all arrays are the same shape
    if not (M_2500.shape == alpha_nu.shape == z.shape):
        raise ValueError("M_2500, alpha_nu, and z must have the same shape")

    M_i_z = np.full_like(M_2500, np.nan, dtype=float)

    for i in range(len(M_2500)):
        λ_obs = wavelengths
        T = transmission
        α = alpha_nu[i]
        z_i = z[i]

        # observed λ corresponds to rest-frame λ_rest = λ_obs / (1 + z)
        λ_rest = λ_obs / (1 + z_i)

        # Monochromatic correction: 2500 → λ_eff
        λ_eff = np.average(λ_rest, weights=T)
        mono_corr = -2.5 * (α + 2) * np.log10(λ_eff / lambda_2500)

        # Broadband correction using power-law weighting
        integrand = T * λ_obs**(-(α + 2))
        numerator = np.trapezoid(integrand, λ_obs)
        denominator = np.trapezoid(T, λ_obs)
        broadband_weighted = numerator / denominator

        broadband_corr = -2.5 * np.log10(broadband_weighted / λ_eff**(-(α + 2)))

        delta_M = mono_corr + broadband_corr
        M_i_z[i] = M_2500[i] + delta_M

    return M_i_z

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
        d['z'] = obj['Z_DR16Q']
        d['sdss_name'] = fits_data['SDSS_NAME'][i]  # Extract SDSS_NAME
        d['log_lbol'] = -999.0
        if d['z'] < 0.7:
            d['log_lbol'] = np.log10(5.15) + fits_data['LOGL3000'][i]
            d['log_lbol_err'] = fits_data['LOGL3000_ERR'][i]
        else:
            d['log_lbol'] = fits_data['LOGLBOL'][i]  # Extract log Lbol values
            d["log_lbol_err"] = fits_data['LOGLBOL_ERR'][i]  # Extract log Lbol error values
        d['LOGLBOL'] = d['log_lbol']
        d['LOGMBH'] = fits_data['LOGMBH'][i]  # Extract log MBH values
        d['LOGMBH_ERR'] = fits_data['LOGMBH_ERR'][i]  # Extract log MBH error values
        d['LOGLEDD_RATIO'] = fits_data['LOGLEDD_RATIO'][i]  # Extract log L/edd values
        d['LOGLEDD_RATIO_ERR'] = fits_data['LOGLEDD_RATIO_ERR'][i]  # Extract log L/edd error values
        d['ebv'] = fits_data['EBV'][i]
        d['sn_median_all'] = fits_data['SN_MEDIAN_ALL'][i]
        d['M_i'] = fits_data_2['M_I'][i]
        # d['CIV'] = fits_data['CIV'][i, 0]
        # d['FEII_UV_EW'] = fits_data['FEII_UV_EW'][i]
        # d['FEII_OPT_EW'] = fits_data['FEII_OPT_EW'][i]
        # d['HBETA'] = fits_data['HBETA'][i, 0]
        # d['HALPHA'] = fits_data['HALPHA'][i, 0]

        d['LOGLBOL'] = fits_data['LOGLBOL'][i]
        d['LOGL1350'] = fits_data['LOGL1350'][i]
        d['LOGL1700'] = fits_data['LOGL1700'][i]
        d['LOGL2500'] = fits_data['LOGL2500'][i]
        d['LOGL3000'] = fits_data['LOGL3000'][i]
        d['LOGL5100'] = fits_data['LOGL5100'][i]
        d['LOGL1350_ERR'] = fits_data['LOGL1350_ERR'][i]
        d['LOGL1700_ERR'] = fits_data['LOGL1700_ERR'][i]
        d['LOGL2500_ERR'] = fits_data['LOGL2500_ERR'][i]
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

def read_quasars_from_hdf5(file_path):
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
    return quasar_list

def filter_unresolved_quasars(df):
    print("Filtering unresolved quasars...")

    # Set Vizier to return all columns and a reasonable row limit
    Vizier.columns = ['*']
    Vizier.ROW_LIMIT = -1

    # Prepare coordinates for the query
    coords = SkyCoord(ra=df['ra'].values * u.deg, dec=df['dec'].values * u.deg)

    # Query Vizier catalog V/154/sdss16 for matches within 2 arcsec
    result = Vizier.query_region(coords, radius=2 * u.arcsec, catalog='V/154/sdss16')

    # If matches are found, extract the class for each source
    if len(result) > 0:
        sdss_table = result[0]
        # Build a DataFrame for easy merging
        sdss_df = sdss_table.to_pandas()
        # Merge on coordinates (within 2 arcsec)
        idx, d2d, _ = match_coordinates_sky(coords, SkyCoord(ra=sdss_df['RA_ICRS'].values*u.deg, dec=sdss_df['DE_ICRS'].values*u.deg))
        matched = d2d.arcsec < 2
        df['sdss16_class'] = None
        df.loc[matched, 'sdss16_class'] = sdss_df.iloc[idx[matched]]['class'].values
        df = df[matched]
        df['sdss16_class'] = df['sdss16_class'].fillna(0)
        # Filter out unresolved quasars
        df = df[df['sdss16_class'] == 6]
        df = df.drop(columns=['sdss16_class'])
        df = df.reset_index(drop=True)
    else:
        raise ValueError("No matches found in the SDSS catalog.")

    return df

def sigma_clip_in_bins(df, bin_width=0.1, sigma=2):
    df_clean = []
    z_min, z_max = df['z'].min(), df['z'].max()
    bins = np.arange(z_min, z_max + bin_width, bin_width)

    for i in range(len(bins) - 1):
        bin_df = df[(df['z'] >= bins[i]) & (df['z'] < bins[i+1])]
        if len(bin_df) < 5:
            continue
        y = bin_df['apparent_mag_2500'] - bin_df['M_i']
        clipped, _, _ = sigmaclip(y, low=sigma, high=sigma)
        df_clean.append(bin_df[y.isin(clipped)])

    return pd.concat(df_clean)

def populate_chi_sq_from_csv(df, csv_path="data/aug4_sample_chisqg10_ebv005sn3.csv"):
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

def load_quasar_data(file_path, populate_sdss=False, apply_cut=True):
    quasar_list = read_quasars_from_hdf5(file_path)
    print("Number of quasars loaded:", len(quasar_list))

    if populate_sdss:
        print("Populating SDSS fields...")
        populate_sdss_fields(quasar_list)
        write_hdf5_file(quasar_list, file_path)

    for quasar in quasar_list:
        if 'ebv' not in quasar.keys():
            print("Populating SDSS fields...")
            populate_sdss_fields(quasar_list)
            write_hdf5_file(quasar_list, file_path)
            break

    df = pd.DataFrame(quasar_list)

    
    df = populate_chi_sq_from_csv(df)

    #df = calculate_all_alpha_nu(df)
    df['alpha_nu'] = -0.5  # Default value
    df['alpha_nu_err'] = 0.1  # Default error
    #df['MY_LOGL2500'], df['MY_LOGL2500_ERR'] = compute_MY_LOGL2500(df)
    

    # df = compute_apparent_mag_2500(df, logL_col='LOGL2500', logL_err_col='LOGL2500_ERR')
    # df['alpha_lambda'] = -1.5
    # df['redchi'] = 1.0
    df = compute_apparent_mag_2500_colin(df)

    num_quasars_z_0_1_before = len(df[(df['z'] > 0) & (df['z'] <= 1.0)])
    num_quasars_z_gt_3_before = len(df[df['z'] > 3])
    print("Number of quasars with 0 < z <= 1.0:", num_quasars_z_0_1_before)
    print("Number of quasars with z > 3:", num_quasars_z_gt_3_before)
    print("Highest redshift quasar:", df['z'].max())
    # Remove infinite values from numeric columns
    columns_with_nans = df.columns[df.isna().any()].tolist()
    print("Columns with NaNs:", columns_with_nans)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].where(~np.isinf(df[numeric_cols]), np.nan)


    # Replace NaNs with 0 in all columns
    #df = df.fillna(0)

    if apply_cut is False:
        print("Skipping data cuts as apply_cut is False.")
        return df
    
    df = df.reset_index(drop=True)

    # Define cuts as (column, lower_limit, upper_limit)
    cuts = [
        #('f_host', None, 0.6),
        ('alpha_lambda', None, 0),
        ('redchi', None, 10),
        ('apparent_mag_2500', 1, 40),
        ('apparent_mag_i', 15, 25)
        # Uncomment/add more cuts as needed
        # ('z', 1, None),
        # ('z', None, 3.2),
        # ('ebv', None, 0.05),
    ]

    initial_count = len(df)
    mask = np.ones(len(df), dtype=bool)
    for col, lower, upper in cuts:
        col_mask = np.ones(len(df), dtype=bool)
        if lower is not None:
            col_mask &= df[col] >= lower
        if upper is not None:
            col_mask &= df[col] < upper
        cut_count = np.sum(~col_mask)
        print(f"Cut on {col}: {cut_count} objects removed")
        mask &= col_mask

    df = df[mask]
    print(f"Total objects removed by all cuts: {initial_count - len(df)}")

    df = df.reset_index(drop=True)
    
    num_quasars_z_0_1 = len(df[(df['z'] > 0) & (df['z'] <= 1.0)])
    num_quasars_z_gt_3 = len(df[df['z'] > 3])
    print("Number of quasars with 0 < z <= 1.0:", num_quasars_z_0_1)
    print("Number of dropped quasars with 0 < z <= 1.0:", num_quasars_z_0_1_before - num_quasars_z_0_1)
    print("Number of quasars with z > 3:", num_quasars_z_gt_3)
    print("Final number of quasars:", len(df))
    return df

def load_data(file_path, populate_sdss=False):
    print("Loading quasar data...")
    #df_agn = load_quasar_data("data/may12_objs_tauwavelength_taublr_redbands_ds4_merged.h5")
    df_agn = load_quasar_data(file_path=file_path, populate_sdss=populate_sdss)
    # Return 200 randomly sampled AGNs for speed
    #df_agn = df_agn.sample(n=500, random_state=42).reset_index(drop=True)

    # Load Pantheon+ SN metadata
    print("Loading Pantheon+ supernova data...")
    df_pantheon = pd.read_csv(
        #"https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat",
        "data/Pantheon+SH0ES.dat",
        sep=r"\s+"
    )

    print("Loading SN covariance matrix...")
    n_sn = len(df_pantheon)

    # Load .cov file with NumPy, skipping the first line (which contains just "1701")
    cov_flat = np.loadtxt(
        #"https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov",
        "data/Pantheon+SH0ES_STAT+SYS.cov",
        skiprows=1
    )

    # Reshape into square matrix
    cov_matrix = cov_flat.reshape((n_sn, n_sn))

    # Confirm shape is correct
    assert cov_matrix.shape == (n_sn, n_sn), f"Expected ({n_sn},{n_sn}), got {cov_matrix.shape}"

    # Pre-compute Cholesky decomposition and log-determinant for SN likelihood
    try:
        sna_L, sna_lower = cho_factor(cov_matrix, lower=True)
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix is not positive-definite!")

    # Compute log-determinant: log|C| = 2 * sum(log(diagonal of Cholesky factor))
    sna_logdetCov = 2.0 * np.sum(np.log(np.diag(sna_L)))

    print("Cholesky factorization successful. Data loaded. ")

    return df_agn, df_pantheon, sna_logdetCov, sna_L, sna_lower




def soft_clip(x, floor=1e-5, sharpness=5):
    # Smoother logistic-like clipping
    return floor + (1 - floor) * (1 / (1 + np.exp(-sharpness * (x - floor))))

def compare_models_by_log_evidence(logZ_1, logZerr_1, logZ_2, logZerr_2, model_1_name="Model 1", model_2_name="Model 2"):
    """
    Compare two models based on their log-evidence (logZ) and uncertainties.

    Parameters:
        logZ_1 (float): Log-evidence of model 1
        logZerr_1 (float): Uncertainty in logZ_1
        logZ_2 (float): Log-evidence of model 2
        logZerr_2 (float): Uncertainty in logZ_2
        model_1_name (str): Name of model 1
        model_2_name (str): Name of model 2

    Returns:
        dict: A dictionary containing Bayes factor, delta_logZ, sigma_equivalent, and evidence strength.
    """
    delta_logZ = logZ_1 - logZ_2
    delta_logZ_err = np.sqrt(logZerr_1**2 + logZerr_2**2)
    bayes_factor = np.exp(delta_logZ)
    sigma_equiv = np.sqrt(2 * abs(delta_logZ))

    # Interpret strength using Jeffreys' scale
    abs_delta = abs(delta_logZ)
    if abs_delta < 1:
        strength = "Not worth more than a bare mention"
    elif abs_delta < 2.5:
        strength = "Substantial evidence"
    elif abs_delta < 5:
        strength = "Strong evidence"
    else:
        strength = "Very strong evidence"

    preferred_model = model_1_name if delta_logZ > 0 else model_2_name

    result = {
        "delta_logZ": delta_logZ,
        "delta_logZ_err": delta_logZ_err,
        "Bayes_factor": bayes_factor,
        "preferred_model": preferred_model,
        "strength": strength,
        "sigma_equivalent": sigma_equiv
    }

    print(f"\nBayesian Model Comparison:")
    print(f"  ΔlogZ = {delta_logZ:.2f} ± {delta_logZ_err:.2f}")
    print(f"  Bayes factor (B_12) = {bayes_factor:.2f}")
    print(f"  Sigma-equivalent ≈ {sigma_equiv:.2f}σ")
    print(f"  Preferred model: {preferred_model}")
    print(f"  Evidence strength: {strength}")

    return result

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

def generate_cosmo_table_latex(results):
    """
    Generate a LaTeX table for cosmological parameter results.

    Parameters
    ----------
    results : list of dict
        Each dictionary must contain:
        - model (str)
        - data (str): "SN~Ia" or "SN~Ia + AGN"
        - Om0 (tuple): (mean, err)
        - H0 (tuple): (mean, err)
        - w0 (tuple): (mean, err)
        - wa (tuple or None): (mean, err) or None
        - logZ (tuple or None): (mean, err) or None
    """
    lines = []
    lines.append("\\begin{table*}")
    lines.append("\\centering")
    lines.append("\\caption{Marginalized Cosmological Parameters and Bayesian Evidence}")
    lines.append("\\label{tab:cosmoparams}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\hline\\hline")
    lines.append("Model & Data & $\\Omega_m$ & $H_0$ [km s$^{-1}$ Mpc$^{-1}$] & $w$ / $w_0$ & $w_a$ & $\\ln \\mathcal{Z}$ \\\\")
    lines.append("\\hline")

    for res in results:
        Om0 = f"${res['Om0'][0]:.3f} \\pm {res['Om0'][1]:.3f}$"
        H0 = f"${res['H0'][0]:.1f} \\pm {res['H0'][1]:.1f}$"
        w0 = f"${res['w0'][0]:+.2f} \\pm {res['w0'][1]:.2f}$"
        wa = "--" if res['wa'] is None else f"${res['wa'][0]:+.1f} \\pm {res['wa'][1]:.1f}$"
        logZ = "--" if res['logZ'] is None else f"${res['logZ'][0]:.1f} \\pm {res['logZ'][1]:.1f}$"
        lines.append(f"{res['model']} & {res['data']} & {Om0} & {H0} & {w0} & {wa} & {logZ} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    latex_table = "\n".join(lines)
    filename = "plots/hubble/table.tex"
    with open(filename, "w") as f:
        f.write(latex_table)
        print(f"LaTeX table written to: {filename}")


def extract_cosmo_results_from_sampler(sampler, cosmo_model, only_sna, dynasty=False, logZ_tuple=None):
    """
    Extract mean and stddev of cosmological parameters from a sampler.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler or dynesty.DynamicNestedSampler
        The sampler object from your pipeline.
    cosmo_model : str
        'FlatwCDM' or 'Flatw0waCDM'
    only_sna : bool
        True if SN Ia only; False if SN Ia + AGN.
    dynasty : bool
        True if using dynesty; False if using emcee.
    logZ_tuple : tuple or None
        (logZ, logZerr), only available from dynesty.

    Returns
    -------
    dict
        Result row for LaTeX table.
    """
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    data_label = "SN~Ia" if only_sna else "SN~Ia + AGN"

    # Flatten or resample samples
    if dynasty:
        samples = sampler.results.samples
        weights = np.exp(sampler.results.logwt - sampler.results.logz[-1])
        samples = resample_equal(samples, weights)
    else:
        samples = sampler.get_chain(flat=True)

    # Extract parameter stats
    def mean_std(param_name):
        idx = model_labels.index(param_name)
        return np.mean(samples[:, idx]), np.std(samples[:, idx])

    param_stats = {
        "Om0": mean_std("Om0"),
        "H0": mean_std("H0"),
        "w0": mean_std("w0"),
        "wa": mean_std("wa") if cosmo_model == "Flatw0waCDM" else None
    }

    return {
        "model": "Flat$w_0w_a$CDM" if cosmo_model == "Flatw0waCDM" else "Flat$w$CDM",
        "data": data_label,
        "Om0": param_stats["Om0"],
        "H0": param_stats["H0"],
        "w0": param_stats["w0"],
        "wa": param_stats["wa"],
        "logZ": logZ_tuple
    }

def display_results_summary(samples, cosmo_model, z_agn_pivot):
    """
    Print median and 16/84% intervals for sampled params, plus derived w0 (and wa)
    when applicable. If cosmo_model == 'Flatw0waCDM', w0 is computed from (wp, wa)
    at the supplied z_agn_pivot.
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

        a_p = 1.0 / (1.0 + float(z_agn_pivot))
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


def approximate_sound_horizon(Om0, h, Omega_b=0.049):
    """
    Eisenstein & Hu (1998) approximation to sound horizon at drag epoch (in Mpc).
    """
    omega_m = Om0 * h**2
    omega_b = Omega_b * h**2
    b1 = 0.313 * omega_m**(-0.419) * (1 + 0.607 * omega_m**0.674)
    b2 = 0.238 * omega_m**0.223
    z_drag = 1291 * omega_m**0.251 / (1 + 0.659 * omega_m**0.828) * (1 + b1 * omega_b**b2)
    R_drag = 31.5 * omega_b * 1e3 / (z_drag * omega_m * 1e3)
    return 44.5 * np.log(9.83 / omega_m) / np.sqrt(1 + 10 * R_drag)

def make_psd(matrix, epsilon=1e-10):
    """
    Enforce symmetric positive semi-definiteness.
    """
    matrix = (matrix + matrix.T) / 2
    eigvals, eigvecs = eigh(matrix)
    eigvals[eigvals < epsilon] = epsilon
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def log_likelihood_planck2018_cmb(cosmo, Omega_b=0.049):
    """
    Log-likelihood from Planck 2018 compressed CMB distance priors:
    (ℓ_A, R, z_*) for flat w0waCDM.
    
    Parameters
    ----------
    cosmo : astropy.cosmology.FRW
        Cosmology instance (Flatw0waCDM or similar).
    Omega_b : float
        Baryon density (default from Planck 2018 best-fit).

    Returns
    -------
    float
        Log-likelihood contribution from Planck 2018 CMB data.
    """
    h = cosmo.H0.value / 100
    Om0 = cosmo.Om0
    z_star = 1089.92  # Planck best-fit decoupling redshift

    # Comoving distance to z_star
    D_M = cosmo.comoving_distance(z_star).value  # in Mpc

    # Sound horizon
    r_s = approximate_sound_horizon(Om0, h, Omega_b)

    # Acoustic scale and shift parameter
    l_A = np.pi * D_M / r_s
    R = np.sqrt(Om0) * D_M * cosmo.H0.value / 299792.458  # c in km/s

    # Planck 2018 compressed parameters
    PLANCK_MEAN = np.array([301.77, 1.7492, z_star])
    PLANCK_COV = np.array([
        [0.090**2,  0.00045,  0.057],
        [0.00045,   0.0042**2, 0.0036],
        [0.057,     0.0036,    0.25**2]
    ])
    PLANCK_COV = make_psd(PLANCK_COV)

    data = np.array([l_A, R, z_star])
    return multivariate_normal.logpdf(data, mean=PLANCK_MEAN, cov=PLANCK_COV)



def log_likelihood_cmb_distance_priors_simpler(cosmo):
    z_star = 1089 # # Last scattering surface
    h = cosmo.H0.value / 100

    # Planck values in h^-1 Mpc
    D_M_CMB_hinv = 1394.4
    sigma_D_M_hinv = 61.0

    # Convert to Mpc using model H0
    D_M_CMB = D_M_CMB_hinv / h
    sigma_D_M = sigma_D_M_hinv / h

    # model's prediction
    D_M_model = (1 + z_star) * cosmo.angular_diameter_distance(z_star).value

    # Log-likelihood
    ll_cmb = norm.logpdf(D_M_model, loc=D_M_CMB, scale=sigma_D_M)
    return ll_cmb



def predict_uncensored_magnitudes_from_observed(df_agn, completeness_params, nsig=4, n_grid=500):
    """
    Computes bias-corrected apparent magnitudes using a log-space stabilized posterior.

    Parameters
    ----------
    df_agn : pd.DataFrame
        Must contain 'apparent_mag_2500', 'apparent_mag_2500_err', 'z'.
    completeness_params : tuple
        Output from get_completeness_function_2d.
    nsig : float
        Number of sigmas for grid range.
    n_grid : int
        Number of magnitude samples.

    Returns
    -------
    np.ndarray
        Posterior-mean corrected magnitudes.
    """
    completeness2d, *_ = completeness_params

    m_obs = df_agn['apparent_mag_2500'].values.astype(np.float64)
    m_err = df_agn['apparent_mag_2500_err'].values.astype(np.float64)
    z = df_agn['z'].values.astype(np.float64)
    N = len(m_obs)

    # Build expanded mag grid
    m_offsets = np.linspace(-nsig, nsig, n_grid)
    m_grid = m_obs[:, None] + m_err[:, None] * m_offsets  # shape (N, n_grid)

    # Gaussian log-prior
    log_prior = norm.logpdf(m_grid, loc=m_obs[:, None], scale=m_err[:, None])

    # Evaluate completeness and soft-clip in log-space
    z_grid = np.tile(z[:, None], (1, n_grid))
    p_det = completeness2d(m_grid, z_grid)

    # Estimate more adaptive floor (5th percentile of nonzero completeness)
    finite_p = p_det[np.isfinite(p_det) & (p_det > 0)]
    min_c = np.percentile(finite_p, 5) if len(finite_p) > 0 else 1e-4
    p_det = soft_clip(p_det, floor=min_c, sharpness=20)

    log_p_det = np.log(p_det + 1e-300)

    # Log posterior (unnormalized)
    log_post = log_prior + log_p_det
    log_post -= np.max(log_post, axis=1, keepdims=True)  # for stability

    post = np.exp(log_post)
    post /= np.trapz(post, m_grid, axis=1)[:, None] + 1e-12

    m_corr = np.trapz(m_grid * post, m_grid, axis=1)

    return m_corr

def predict_uncensored_magnitudes(df_agn, m_model, mu_err, completeness_params, nsig=4, n_grid=300):
    z = df_agn['z'].values
    p_detect_fn, *_ = completeness_params

    uncensored_samples = []
    min_c = p_detect_fn.min_completeness_value

    for i in range(len(df_agn)):
        m = m_model[i]
        sigma = mu_err[i]
        zval = z[i]

        m_offsets = np.linspace(-nsig, nsig, n_grid)
        m_grid = m + sigma * m_offsets
        prior = stats.norm.pdf(m_grid, loc=m, scale=sigma)

        p_det = p_detect_fn(m_grid, np.full_like(m_grid, zval))
        p_det = soft_clip(p_det, floor=min_c, sharpness=20)

        if np.sum(p_det) < 1e-6:
            uncensored_samples.append(m)
            continue

        posterior = prior * p_det
        posterior /= np.trapezoid(posterior, m_grid)

        mean_m = np.trapezoid(m_grid * posterior, m_grid)
        uncensored_samples.append(mean_m)

    return np.array(uncensored_samples)

def apply_forward_completeness_correction(df_agn, params, cosmo_model, completeness_params):
    """
    Apply completeness-aware correction to model-predicted apparent magnitudes.

    Parameters
    ----------
    df_agn : pd.DataFrame
        AGN data including redshift, apparent magnitude error, and variability features.
    params : dict
        Dictionary of model parameters.
    cosmo : astropy cosmology instance
        Cosmology model used to compute distance moduli.
    completeness_params : tuple
        Output from get_completeness_function_2d.

    Returns
    -------
    np.ndarray
        Completeness-corrected apparent magnitudes (posterior means).
    """

    priors, model_labels, _ = get_model_params(cosmo_model)
    # Common post-processing
    
    #params = dict(zip(model_labels, params))
    
    if cosmo_model == 'Flatw0waCDM':
        #cosmo = Flatw0waCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'], wa=params['wa'])
        cosmo = FlatwpwaCDM(H0=params['H0'], Om0=params['Om0'], wp=params['wp'], wa=params['wa'], zp=z_agn_pivot)
    elif cosmo_model == 'FlatwCDM':
        cosmo = FlatwCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'])
    elif cosmo_model == 'FlatLambdaCDM':
        cosmo = FlatLambdaCDM(H0=params['H0'], Om0=params['Om0'])
    else:
        raise ValueError("cosmo_model must be 'FlatwCDM', 'Flatw0waCDM' or 'FlatLambdaCDM'")

    # Compute model-predicted absolute magnitude
    M_model = M_model_agn(
        params['M0_agn'],
        params['log_sigma0_break'],
        params['eta_A1_agn'], params['eta_A2_agn'],
        params['eta_break_agn'],
        params['beta_agn'],
        df_agn['log_sigma0'].values,
        df_agn['log_tau_UV_RF'].values
    )

    # Cosmological distance modulus + K-correction
    mu_cosmo = cosmo.distmod(df_agn['z'].values).value
    m_model = mu_cosmo + (K_corr(df_agn['z'].values) - K_corr(2)) + M_model

    # Total uncertainty on m_model
    mu_err = np.sqrt(
        df_agn['apparent_mag_2500_err'].values**2 +
        M_model_agn_err(
            params['M0_agn'],
            params['log_sigma0_break'],
            params['eta_A1_agn'], params['eta_A2_agn'],
            params['eta_break_agn'],
            params['beta_agn'],
            df_agn['log_sigma0'].values,
            df_agn['log_sigma0_err'].values,
            df_agn['log_tau_UV_RF'].values
        )**2 +
        (2.5 * 0.3 * np.log10(1 + df_agn['z'].values))**2 +
        (0.055 * df_agn['z'].values)**2 +
        np.exp(2 * params['log_f'])
    )

    # Apply marginalization over completeness
    m_corr = predict_uncensored_magnitudes(df_agn, m_model, mu_err, completeness_params)
    return m_corr

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
    for k in ('zp', 'z_p', 'z_agn_pivot', 'z_pivot'):
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

import numpy as np

def posterior_corr(flat_samples, cosmo_model, z_agn_pivot):
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
    z_agn_pivot : float, optional
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
        i_wp = _idx(model_labels, "wp", "w_p")
        i_wa = _idx(model_labels, "wa", "w_a")
        a_p = 1.0 / (1.0 + float(z_agn_pivot))
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


