#!/usr/bin/env python3
from functools import partial
import multiprocessing
from multiprocessing import Pool, cpu_count
import os
import matplotlib.pyplot as plt

num_cores = os.environ.get("NUM_CORES", os.cpu_count()-2)
try:
    num_cores = int(num_cores)
except ValueError:
    print(f"Invalid NUM_CORES value '{num_cores}', ignoring.")
    num_cores = os.cpu_count()

if multiprocessing.current_process().name == "MainProcess":
    print(f"CPU Num Cores: {num_cores}")
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_cores}"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

prefix = os.environ.get('PREFIX', "test")
suffix = os.environ.get('SUFFIX', "test")

print(f"PREFIX: {prefix}, SUFFIX: {suffix}")

import sys
import timeit
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
import csv

from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from hubble_utils import load_agn_data, write_hdf5_file

from astroquery.sdss import SDSS
#warnings.filterwarnings("ignore")

bands = ['u', 'g', 'r', 'i', 'z']

colors = {'u': 'tab:blue',
          'g': 'tab:green',
          'r': 'tab:orange',
          'i': 'tab:red',
          'z': 'tab:brown',
          'y': 'tab:gray'}

def plot_f_host_4200_vs_z(quasar_dict_list):
    """
    Plot log10(f_host_4200) vs redshift for all objects and save the figure.
    """
    zs = []
    f_host_4200s = []
    for q in quasar_dict_list:
        z = q.get("z", np.nan)
        f_host_4200 = q.get("f_host_4200", np.nan)
        if np.isfinite(z) and np.isfinite(f_host_4200) and f_host_4200 > 0:
            zs.append(z)
            f_host_4200s.append(f_host_4200)
    if zs and f_host_4200s:
        plt.figure(figsize=(7,5))
        plt.scatter(zs, np.log10(f_host_4200s), s=15, alpha=0.7)
        plt.xlabel("Redshift (z)")
        plt.ylabel("log$_{10}$(f$_{host,4200}$)")
        plt.title("log$_{10}$(f$_{host,4200}$) vs Redshift")
        plt.tight_layout()
        os.makedirs(f"plots/pyqsofit/{prefix}", exist_ok=True)
        plt.savefig(f"plots/pyqsofit/{prefix}/f_host_4200_vs_z.png")
        plt.close()

def plot_delta_mags_vs_z_single_quasar(quasar):
    """
    Plot delta_m_avg and delta_mags for a single quasar as a function of its redshift.
    Each delta_mags point is colored by its band.
    """
    z = quasar.get("z", np.nan)
    delta_m_avg = quasar.get("delta_m_avg", np.nan)
    delta_mags = quasar.get("delta_mags", {})
    obj_id = str(quasar.get("object_id", ""))

    # Plot delta_m_avg vs z (single point)
    plt.figure(figsize=(8,6))
    if np.isfinite(z) and np.isfinite(delta_m_avg):
        plt.scatter([z], [delta_m_avg], color='red', s=60, marker='*', label=f"object_id={obj_id}")
    plt.xlabel("z")
    plt.ylabel("delta_m_avg")
    plt.title(f"delta_m_avg vs z (object_id={obj_id})")
    plt.legend()
    plt.tight_layout()
    os.makedirs(f"plots/pyqsofit/{prefix}/delta_m_avg_single", exist_ok=True)
    plt.savefig(f"plots/pyqsofit/{prefix}/delta_m_avg_single/delta_mags_vs_z_avg_{obj_id}.png")
    plt.close()

    # Plot all delta_mags vs z (colored by band)
    if np.isfinite(z) and isinstance(delta_mags, dict) and len(delta_mags) > 0:
        plt.figure(figsize=(8,6))
        for band, dm in delta_mags.items():
            if np.isfinite(dm):
                color = colors.get(band, 'gray')
                plt.scatter([z], [dm], color=color, s=40, alpha=0.7, label=band)
        plt.scatter([z], [delta_m_avg], color='red', s=60, marker='*', label="delta_m_avg")
        plt.xlabel("z")
        plt.ylabel("delta_mags (all bands)")
        plt.title(f"delta_mags (all bands) vs z (object_id={obj_id})")
        plt.legend()
        plt.tight_layout()
        os.makedirs(f"plots/pyqsofit/{prefix}/delta_mags_single", exist_ok=True)
        plt.savefig(f"plots/pyqsofit/{prefix}/delta_mags_single/delta_mags_all_vs_z_{obj_id}.png")
        plt.close()
        
def plot_delta_mags_vs_z(quasar_dict_list):
    zs = []
    delta_m_avgs = []
    delta_mags_by_band = {b: ([], []) for b in bands}
    for q in quasar_dict_list:
        z = q.get("z", np.nan)
        delta_m_avg = q.get("delta_m_avg", np.nan)
        delta_mags = q.get("delta_mags", {})
        if np.isfinite(z) and np.isfinite(delta_m_avg) and delta_m_avg > -1e8:
            zs.append(z)
            delta_m_avgs.append(delta_m_avg)
        if np.isfinite(z) and isinstance(delta_mags, dict) and len(delta_mags) > 0:
            for band, dm in delta_mags.items():
                if np.isfinite(dm):
                    delta_mags_by_band[band][0].append(z)
                    delta_mags_by_band[band][1].append(dm)

    # Plot delta_m_avg vs z
    plt.figure(figsize=(6,4))
    plt.scatter(zs, delta_m_avgs, s=10, alpha=0.7)
    plt.xlabel("z")
    plt.ylabel("delta_m_avg")
    plt.title("delta_m_avg vs z")
    plt.tight_layout()
    os.makedirs(f"plots/pyqsofit/{prefix}", exist_ok=True)
    plt.savefig(f"plots/pyqsofit/{prefix}/delta_m_avg.png")
    plt.close()

    # Plot all delta_mags vs z, colored by band
    plt.figure(figsize=(8,6))
    for band in bands:
        z_arr, dm_arr = delta_mags_by_band[band]
        if len(z_arr) > 0:
            plt.scatter(z_arr, dm_arr, s=10, alpha=0.7, color=colors.get(band, 'gray'), label=band)
    plt.xlabel("z")
    plt.ylabel("delta_mags (all bands)")
    plt.title("delta_mags (all bands) vs z")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/pyqsofit/{prefix}/delta_mags_all.png")
    plt.close()

def bands_bluer_than_lyman_alpha(z, buffer=100):
    """
    Returns a list of bands with rest-frame effective wavelength < Lyman-alpha (1216 Å).

    Args:
        z (float): Redshift.

    Returns:
        list: Bands bluer than Lyman-alpha at rest frame.
    """
    lyman_alpha = 1216
    bands = {
        'u': {'lambda_eff': 3551},
        'g': {'lambda_eff': 4686},
        'r': {'lambda_eff': 6165},
        'i': {'lambda_eff': 7481},
        'z': {'lambda_eff': 8931},
        'y': {'lambda_eff': 9700}
    }

    bluer_bands = []
    for band, props in bands.items():
        rest_lambda_eff = props['lambda_eff'] / (1 + z)
        if rest_lambda_eff < (lyman_alpha - buffer):
            bluer_bands.append(band)

    return bluer_bands
# --------------------------- Utility ---------------------------------
def _safe_float(x):
    try:
        # unwrap numpy scalars/arrays/masked values
        x = np.asarray(x).squeeze()
        if isinstance(x, (bytes, bytearray)):
            x = x.decode("ascii", "ignore")
        return float(x)
    except Exception:
        return np.nan
def compute_apparent_mag_2500_astropy(conti_table, logL_col, logL_err_col,
                                      z_col='z', H0=70, Om0=0.3):
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    c = 2.99792458e10  # cm/s
    lambda_ = 2500e-8  # cm

    z = conti_table[z_col]
    logL_2500 = conti_table[logL_col]
    logL_2500_err = conti_table[logL_err_col]

    DL = cosmo.luminosity_distance(z).to(u.cm).value  # cm

    log_Lnu = logL_2500 + np.log10(lambda_ / c)
    log_fnu = log_Lnu - np.log10(4 * np.pi * DL**2 * (1 + z))
    m_ab = -2.5 * log_fnu - 48.60
    m_ab_err = 2.5 * logL_2500_err

    return m_ab, m_ab_err


def fetch_spectrum_fits(sdss_name, plate, fiber, mjd, cache_dir="data/spectra_cache"):
    """
    Download SDSS spectrum and cache to FITS file if not already present.
    Returns (hdulist, from_cache: bool). Raises on failure.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{sdss_name}.fits")

    if os.path.exists(cache_file):
        return fits.open(cache_file, memmap=False), True

    spec = SDSS.get_spectra(plate=plate, fiberID=fiber, mjd=mjd)
    if spec is None or len(spec) == 0:
        raise ValueError(f"No spectrum found for SDSS_NAME={sdss_name} (p={plate}, f={fiber}, mjd={mjd})")

    hdul = spec[0]
    hdul.writeto(cache_file, overwrite=True)
    # reopen without memory mapping to unify return type
    return fits.open(cache_file, memmap=False), False

from astroquery.sdss import SDSS
from astropy.table import Table
from astropy.io.ascii.core import InconsistentTableError

def zquery(plate, mjd, fiberid, dr=16, timeout=60):
    """
    Query SDSS for redshift and classification by (plate, mjd, fiberid).

    Returns:
        astropy.table.Table row (length-1) or None if no match / error.
    """
    # Make sure these are ints to avoid odd SQL formatting
    plate   = int(plate)
    mjd     = int(mjd)
    fiberid = int(fiberid)

    q = f"""
    SELECT TOP 1
        specObjID, plate, mjd, fiberID,
        z, zErr, zWarning, class, subClass
    FROM SpecObjAll
    WHERE plate={plate} AND mjd={mjd} AND fiberID={fiberid}
    """

    try:
        res = SDSS.query_sql(q, data_release=dr, timeout=timeout, cache=False)
    except InconsistentTableError as e:
        # This usually means the server returned HTML instead of CSV (e.g., error page)
        print("SDSS returned a non-CSV response (likely an error page).")
        print("Offending SQL:\n", q)
        print("Astropy parsing error:", e)
        raise e
    except Exception as e:
        print("SDSS query failed:", e)
        print("Offending SQL:\n", q)
        raise e

    if res is None or len(res) == 0:
        print("No rows returned for:", (plate, mjd, fiberid))
        return None

    # Print and return the single row
    return dict(res[0])


def load_spec_from_cache(sdss_name, cache_dir="data/spectra_cache"):
    """Return cached FITS HDUList if available, else None."""
    cache_file = os.path.join(cache_dir, f"{sdss_name}.fits")
    if os.path.exists(cache_file):
        return fits.open(cache_file, memmap=False)
    return None


def match_sample_to_dr16q(sample_df, dr16q_fits, max_sep_arcsec=2.0):
    """Load sample CSV and DR16Q, crossmatch within max_sep_arcsec, return (data_cat_table, sample_df_matched).
    Ensures 1–to–1 matches by keeping the closest pair per AGN (and per SDSS if needed)."""
    #sample_df = pd.read_csv(sample_csv)


    with fits.open(dr16q_fits) as hdul:
        data_cat_full = hdul[1].data

    coords_sdss = SkyCoord(ra=data_cat_full['RA'],  dec=data_cat_full['DEC'],  unit=(u.deg, u.deg), frame='icrs')
    coords_agn  = SkyCoord(ra=sample_df['ra'],       dec=sample_df['dec'],    unit=(u.deg, u.deg), frame='icrs')

    idx_sdss_all, idx_agn_all, sep2d_all, _ = coords_agn.search_around_sky(coords_sdss, max_sep_arcsec * u.arcsec)
    if len(idx_sdss_all) == 0:
        raise RuntimeError("No matches found between sample and DR16Q with the given separation.")

    # Build all matches then keep the closest per AGN
    matches = pd.DataFrame({
        'agn_idx': np.asarray(idx_agn_all, dtype=int),
        'sdss_idx': np.asarray(idx_sdss_all, dtype=int),
        'sep_arcsec': sep2d_all.arcsec
    }).sort_values('sep_arcsec', kind='mergesort')

    # 1) keep closest per AGN
    matches = matches.groupby('agn_idx', as_index=False).first()

    # Optional: also enforce one-to-one on SDSS side (keep closest remaining per SDSS)
    matches = matches.sort_values('sep_arcsec', kind='mergesort')
    matches = matches.drop_duplicates(subset='sdss_idx', keep='first')

    # Now we have a 1–to–1 list of pairs
    agn_keep   = matches['agn_idx'].to_numpy()
    sdss_keep  = matches['sdss_idx'].to_numpy()

    # Slice both tables to the same length and consistent ordering
    data_cat_arr = data_cat_full[sdss_keep]  # numpy recarray
    data_cat = Table(data_cat_arr, copy=True)  # convert to Astropy Table

    sample_df_matched = sample_df.iloc[agn_keep].reset_index(drop=True)

    # Add photometry columns (now lengths match)
    for b in ['u', 'g', 'r', 'i', 'z']:
        mags_mean_col = f'mags_mean_{b}' # direct mean from LC
        mean_col = f'mean_{b}' # LC fitting mean (dm)
        if mags_mean_col in sample_df_matched.columns:
            # If mean_col is not >= 0, assign 0
            mean_vals = sample_df_matched[mean_col].to_numpy()
            data_cat[f'mean_corrected_{b}'] = sample_df_matched[mags_mean_col].to_numpy() + mean_vals

    data_cat['object_id'] = sample_df_matched['object_id'].to_numpy()
    #data_cat['clean_bands'] = sample_df_matched['clean_bands'].to_numpy()

    return data_cat



def create_qsopar_fits(path_ex='data/', parfilename='qsopar.fits', overwrite=True, author='Hengxiao Guo'):
    """
    Create the QSOFit parameter file 'qsopar.fits' with the following HDUs:
      - PRIMARY        : header only (Author)
      - line_priors    : emission-line priors (wavelengths, windows, Gaussian setup, ties)
      - conti_windows  : continuum-fitting windows [Å]
      - conti_priors   : continuum parameter priors (Fe UV/optical, PL, Balmer, poly)
      - measure_info   : continuum luminosity wavelengths and Fe flux range(s)

    Parameters
    ----------
    path_ex : str
        Directory where 'qsopar.fits' will be written.
    overwrite : bool
        Overwrite an existing file if present.
    author : str
        Value placed in PRIMARY header.

    Returns
    -------
    str
        Absolute path to the written FITS file.
    """
    import os
    import numpy as np
    from astropy.io import fits
    from astropy.table import Table

    os.makedirs(path_ex, exist_ok=True)

    # ------------------------ PRIMARY HDU ------------------------
    hdr0 = fits.Header()
    hdr0['Author'] = author
    primary_hdu = fits.PrimaryHDU(header=hdr0)

    # ------------------------ line_priors ------------------------
    lp_dtype = np.dtype([
        ('lambda',  'f4'),
        ('compname','S20'),
        ('minwav',  'f4'),
        ('maxwav',  'f4'),
        ('linename','S20'),
        ('ngauss',  'i4'),
        ('inisca',  'f4'),
        ('minsca',  'f4'),
        ('maxsca',  'f4'),
        ('inisig',  'f4'),
        ('minsig',  'f4'),
        ('maxsig',  'f4'),
        ('voff',    'f4'),
        ('vindex',  'i4'),
        ('windex',  'i4'),
        ('findex',  'i4'),
        ('fvalue',  'f4'),
        ('vary',    'i4'),
    ])

    line_priors = np.rec.fromrecords([
        #  lambda    comp  minwav maxwav  name         ngauss  inisca  minsca  maxsca  inisig   minsig   maxsig   voff    vindex windex findex fvalue vary
        (6564.61,  'Ha',  6400,  6800,  'Ha_br',         2,    0.0,    0.0,   1e10,   5e-3,   0.004,    0.05,   0.015,     0,     0,     0,   0.05,   1),
        (6564.61,  'Ha',  6400,  6800,  'Ha_na',         1,    0.0,    0.0,   1e10,   1e-3,   5e-4,   0.00169,  0.01,      1,     1,     0,   0.002,  1),
        (6549.85,  'Ha',  6400,  6800,  'NII6549',       1,    0.0,    0.0,   1e10,   1e-3,  2.3e-4,  0.00169,  5e-3,      1,     1,     1,   0.001,  1),
        (6585.28,  'Ha',  6400,  6800,  'NII6585',       1,    0.0,    0.0,   1e10,   1e-3,  2.3e-4,  0.00169,  5e-3,      1,     1,     1,   0.003,  1),
        (6718.29,  'Ha',  6400,  6800,  'SII6718',       1,    0.0,    0.0,   1e10,   1e-3,  2.3e-4,  0.00169,  5e-3,      1,     1,     2,   0.001,  1),
        (6732.67,  'Ha',  6400,  6800,  'SII6732',       1,    0.0,    0.0,   1e10,   1e-3,  2.3e-4,  0.00169,  5e-3,      1,     1,     2,   0.001,  1),

        (4862.68,  'Hb',  4640,  5100,  'Hb_br',         2,    0.0,    0.0,   1e10,   5e-3,   0.004,    0.05,   0.01,      0,     0,     0,   0.01,   1),
        (4862.68,  'Hb',  4640,  5100,  'Hb_na',         1,    0.0,    0.0,   1e10,   1e-3,  2.3e-4,  0.00169,  0.01,      1,     1,     0,   0.002,  1),
        (4960.30,  'Hb',  4640,  5100,  'OIII4959c',     1,    0.0,    0.0,   1e10,   1e-3,  2.3e-4,  0.00169,  0.01,      1,     1,     0,   0.002,  1),
        (5008.24,  'Hb',  4640,  5100,  'OIII5007c',     1,    0.0,    0.0,   1e10,   1e-3,  2.3e-4,  0.00169,  0.01,      1,     1,     0,   0.004,  1),
        (4960.30,  'Hb',  4640,  5100,  'OIII4959w',     1,    0.0,    0.0,   1e10,   3e-3,  2.3e-4,   0.004,   0.01,      2,     2,     0,   0.001,  1),
        (5008.24,  'Hb',  4640,  5100,  'OIII5007w',     1,    0.0,    0.0,   1e10,   3e-3,  2.3e-4,   0.004,   0.01,      2,     2,     0,   0.002,  1),

        (4341.68,  'Hg',  4200,  4400,  'Hg_br',         1,    0.0,    0.0,   1e10,   5e-3,   0.004,    0.05,   0.01,      0,     0,     0,   0.01,   1),
        (4341.68,  'Hg',  4200,  4400,  'Hg_na',         1,    0.0,    0.0,   1e10,   1e-3,  2.3e-4,  0.00169,  0.01,      1,     1,     0,   0.002,  1),
        (4102.89,  'Hd',  4000,  4150,  'Hd_br',         1,    0.0,    0.0,   1e10,   5e-3,   0.004,    0.05,   0.01,      0,     0,     0,   0.01,   1),
        (4102.89,  'Hd',  4000,  4150,  'Hd_na',         1,    0.0,    0.0,   1e10,   1e-3,  2.3e-4,  0.00169,  0.01,      1,     1,     0,   0.002,  1),

        (2798.75, 'MgII', 2700,  2900,  'MgII_br',       2,    0.0,    0.0,   1e10,   5e-3,   0.004,    0.05,   0.015,     0,     0,     0,   0.05,   1),
        (2798.75, 'MgII', 2700,  2900,  'MgII_na',       1,    0.0,    0.0,   1e10,   1e-3,   5e-4,   0.00169,  0.01,      1,     1,     0,   0.002,  1),

        (1908.73, 'CIII', 1700,  1970,  'CIII_br',       2,    0.0,    0.0,   1e10,   5e-3,   0.004,    0.05,   0.015,    99,     0,     0,   0.01,   1),

        (1549.06,  'CIV', 1500,  1700,  'CIV_br',        2,    0.0,    0.0,   1e10,   5e-3,   0.004,    0.05,   0.015,     0,     0,     0,   0.05,   1),

        (1402.06, 'SiIV', 1290,  1450,  'SiIV_OIV1',     1,    0.0,    0.0,   1e10,   5e-3,   0.002,    0.05,   0.015,     1,     1,     0,   0.05,   1),

        (1215.67,  'Lya', 1150,  1290,  'Lya_br',        3,    0.0,    0.0,   1e10,   5e-3,   0.002,    0.05,    0.02,     0,     0,     0,   0.05,   1),
        (1240.14,  'Lya', 1150,  1290,  'NV1240',        1,    0.0,    0.0,   1e10,   2e-3,   0.001,    0.01,   0.005,     0,     0,     0,   0.002,  1),
    ], dtype=lp_dtype)

    hdr1 = fits.Header()
    hdr1['lambda'] = 'Vacuum wavelength [Ang]'
    hdr1['minwav'] = 'Lower complex fitting wavelength range [Ang]'
    hdr1['maxwav'] = 'Upper complex fitting wavelength range [Ang]'
    hdr1['ngauss'] = 'Number of Gaussians for the line'
    hdr1['inisca'] = 'Initial guess of line scale [flux]'
    hdr1['minsca'] = 'Lower range of line scale [flux]'
    hdr1['maxsca'] = 'Upper range of line scale [flux]'
    hdr1['inisig'] = 'Initial guess of line sigma [lnlambda]'
    hdr1['minsig'] = 'Lower range of line sigma [lnlambda]'
    hdr1['maxsig'] = 'Upper range of line sigma [lnlambda]'
    hdr1['voff']   = 'Velocity offset from the central wavelength [lnlambda]'
    hdr1['vindex'] = 'Same NONZERO vindex => same velocity'
    hdr1['windex'] = 'Same NONZERO windex => same width'
    hdr1['findex'] = 'Same NONZERO findex => constrained flux ratios'
    hdr1['fvalue'] = 'Relative scale factor when tying via findex'
    hdr1['vary']   = '0 = fixed; 1 = free'
    hdu1 = fits.BinTableHDU(data=line_priors, header=hdr1, name='line_priors')

    # ------------------------ conti_windows ------------------------
    cw_dtype = np.dtype([('min','f4'), ('max','f4')])
    conti_windows = np.rec.fromrecords([
        # (1150., 1170.),  # often masked due to Lyα forest
        (1275., 1290.),
        (1350., 1360.),
        (1445., 1465.),
        (1690., 1705.),
        (1770., 1810.),
        (1970., 2400.),
        (2480., 2675.),
        (2925., 3400.),
        (3775., 3832.),
        (4000., 4050.),
        (4200., 4230.),
        (4435., 4640.),
        (5100., 5535.),
        (6005., 6035.),
        (6110., 6250.),
        (6800., 7000.),
        (7160., 7180.),
        (7500., 7800.),
        (8050., 8150.),
    ], dtype=cw_dtype)
    hdu2 = fits.BinTableHDU(data=conti_windows, name='conti_windows')

    # ------------------------ conti_priors ------------------------
    cp_dtype = np.dtype([
        ('parname','S20'),
        ('initial','f4'),
        ('min',    'f4'),
        ('max',    'f4'),
        ('vary',   'i4'),
    ])
    conti_priors = np.rec.fromrecords([
        ('Fe_uv_norm',   0.0,   0.0,   1e10,  1),  # MgII Fe template normalization [flux]
        ('Fe_uv_FWHM',   3000,  1200,  18000, 1),  # MgII Fe template FWHM [km/s or AA in template units]
        ('Fe_uv_shift',  0.0,  -0.01,  0.01,  1),  # MgII Fe template shift [lnlambda]
        ('Fe_op_norm',   0.0,   0.0,   1e10,  1),  # Hβ/Hα Fe template normalization [flux]
        ('Fe_op_FWHM',   3000,  1200,  18000, 1),  # Hβ/Hα Fe template FWHM
        ('Fe_op_shift',  0.0,  -0.01,  0.01,  1),  # Hβ/Hα Fe template shift [lnlambda]
        ('PL_norm',      1.0,   0.0,   1e10,  1),  # Power-law normalization (f_λ ∝ (λ/3000)^-α)
        ('PL_slope',    -1.5,  -5.0,   -0.2,   1),  # Power-law slope α
        ('Blamer_norm',  0.0,   0.0,   1e10,  1),  # Balmer continuum normalization (< 3646 Å)
        ('Balmer_Te',  15000, 10000, 50000,  1),   # Balmer continuum Te
        ('Balmer_Tau',   0.5,   0.1,   2.0,   1),  # Balmer continuum τ
        ('Balmer_vel',  3000, 12000, 18000,  1), # Velocity broadening of the Balmer continuum at < 3646 AA [lnlambda]
        ('conti_a_0',    -2.0,   -100.0,  100.0,  1),# Polynomial terms
        ('conti_a_1',    -2.0,   -100.0,  100.0,  1),
        ('conti_a_2',    -2.0,   -100.0,  100.0,  1),
    ], dtype=cp_dtype)

    hdr3 = fits.Header()
    hdr3['vary'] = '0 = fixed; 1 = free'
    hdu3 = fits.BinTableHDU(data=conti_priors, header=hdr3, name='conti_priors')

    # ------------------------ measure_info ------------------------
    # Use fixed-length array columns so FITS writes cleanly (no var-length arrays needed).
    cont_loc = np.array([[1350, 1450, 1700, 2500, 3000, 3500, 4200, 5100]], dtype='f4')  # shape (1, 8)
    fe_flux  = np.array([[4435, 4685]], dtype='f4')                                      # shape (1, 2)

    measure_info = Table()
    measure_info['cont_loc'] = cont_loc
    measure_info['Fe_flux_range'] = fe_flux

    hdu4 = fits.table_to_hdu(measure_info)
    hdu4.name = 'measure_info'
    hdr4 = fits.Header()
    hdr4['cont_loc'] = 'Continuum luminosity wavelengths reported'
    hdr4['Fe_flux_range'] = 'Fe emission wavelength range(s) reported'
    # Merge custom header cards without clobbering standard FITS table cards
    hdu4.header.extend(hdr4, update=True, end=True)

    # ------------------------ Write file ------------------------
    hdul = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3, hdu4])
    outpath = os.path.abspath(os.path.join(path_ex, parfilename))
    hdul.writeto(outpath, overwrite=overwrite)

    # Quick sanity: ensure expected HDUs exist
    with fits.open(outpath, memmap=False) as chk:
        names = {(h.header.get('EXTNAME', '') or '').strip().lower() for h in chk[1:]}
        required = {'line_priors', 'conti_windows', 'conti_priors', 'measure_info'}
        missing = required - names
        if missing:
            raise RuntimeError(f"qsopar.fits written but missing HDUs: {sorted(missing)}")

    return outpath


def run_qsofit_record(rec, npca_qso, cache_dir="data/spectra_cache", 
                      path_ex=f'data/pyqsofit', parfilename=f'qsopar.fits',
                      save_fig_path=f'./plots/pyqso/'):
    """
    Worker-safe version of QSOFit runner.
    `rec` is a plain dict containing only the fields needed for one object.
    """
    from speclite import filters
    from pyqsofit.PyQSOFit import QSOFit

    QSOFit.set_mpl_style()


    # default result (so we always return a complete row even on error)
    result = dict(
        z=rec["z"],
        delta_mags={},
        delta_m_avg=-1e9,
        object_id=rec["object_id"],
        sdss_name=rec["sdss_name"],
        apparent_mag_i_rest=-1e9,
        apparent_mag_2500=-1e9,
        apparent_mag_2500_err=-1e9,
        apparent_mag_2500_reddened=-1e9,
        apparent_mag_2500_reddened_err=-1e9,
        f_host_2500=-99,
        f_host_4200=-99,
        f_host_5100=-99,
        alpha_lambda=-1e9,
        alpha_lambda_err=-1e9,
        redchi=1e9,
        L2500=-1e9,
        L2500_err=-1e9,
        L2500_int=-1e9,
        L2500_int_err=-1e9,
        EBV_ccm89=-1e9,
        EBV=-1e9,
        bands_obj=[],
        bands_used=[],
        bands_dropped=[]
    )

    try:
        # cached spectrum
        hdul = load_spec_from_cache(rec["sdss_name"], cache_dir=cache_dir)
        if hdul is None:
            # keep default values; return early
            return result

        lam  = 10.0 ** hdul[1].data['loglam']                 # [Å]
        flux = hdul[1].data['flux']                           # [1e-17 erg/s/cm^2/Å in SDSS]
        ivar = hdul[1].data['ivar']
        err  = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.inf)

        # --- Choose bands per object (NO globals) ---
        bands_obj = rec.get('bands', ['g', 'r', 'i'])
        bands_dropped = rec['bands_dropped']
        bands_obj = [b for b in bands_obj if b not in bands_dropped]
        if len(bands_obj) == 0:
            sdss_filters = []
        else:
            sdss_filters = filters.load_filters(*[f'sdss2010-{b}' for b in bands_obj])

        # --- Per-band delta magnitudes and weights ---
        delta_mags = {}
        weights = []

        for b, filt in zip(bands_obj, sdss_filters):
            mag_fiber = rec['mags'].get(b, np.nan)
            if not np.isfinite(mag_fiber) or mag_fiber < 0:
                continue

            try:
                # synthetic magnitude from the (unscaled) spectrum
                mag_synth = filt.get_ab_magnitude(
                    1e-17 * flux * u.erg / u.s / u.cm**2 / u.AA,  # SDSS flux units
                    lam * u.AA
                )
            except Exception as e:
                print(f"Error computing synthetic mag for {rec['sdss_name']} band {b}: {e}")
                continue
            dm = mag_fiber - mag_synth
            delta_mags[b] = dm

            # weight by photometric mag uncertainty if available; else equal weight
            sig_m = rec.get('mags_err', {}).get(b, np.nan)
            w = 1.0 / (sig_m**2) if np.isfinite(sig_m) and sig_m > 0 else 1.0
            weights.append(w)

        bands_used = list(delta_mags.keys())
        dm_arr = np.array([delta_mags[b] for b in bands_used], dtype=float)
        w_arr  = np.array(weights[:len(bands_used)], dtype=float)
        mask   = np.isfinite(dm_arr) & np.isfinite(w_arr) & (w_arr > 0)

        if np.any(mask):
            w = w_arr[mask]
            dm = dm_arr[mask]
            delta_m_avg = np.sum(w * dm) / np.sum(w)
            # standard error of weighted mean (for optional calibration inflation)
            sigma_dm = np.sqrt(1.0 / np.sum(w))
        else:
            print(f"[WARN] No usable bands after drops for {rec['sdss_name']} (z={rec['z']:.2f}); scale=1.")
            delta_m_avg = 0.0
            sigma_dm = 0.0

        # --- Apply absolute flux scale ---
        scale = 10.0 ** (-0.4 * delta_m_avg)
        flux_scaled = flux * scale
        err_scaled  = err  * scale      # IMPORTANT: scale the uncertainties too
        # (If you keep ivar anywhere: ivar_scaled = ivar / scale**2)

        # --- Optional: include calibration (zeropoint) uncertainty in quadrature ---
        # This treats a fully correlated term as if it were per-pixel (conservative).
        if sigma_dm > 0:
            frac_s = np.log(10.0) / 2.5 * sigma_dm   # σ_s / s from mag error
            err_scaled = np.sqrt(err_scaled**2 + (flux_scaled * frac_s)**2)

        # Proceed to fitting
        q_mle = QSOFit(lam, flux_scaled, err_scaled, rec["z"], path=path_ex)
        q_mle.Fit(
            name=f"{rec['z']:.2f}_{rec['sdss_name']}_{rec['plate']}-{rec['mjd']}-{rec['fiber']}",  # customize the name of given targets. Default: plate-mjd-fiber
            
            # preprocessing parameters
            nsmooth=1,              # do n-pixel smoothing to the raw input flux and err spectra
            and_mask=False,         # delete the and masked pixels
            or_mask=False,          # delete the or masked pixels
            reject_badpix=True,    # reject 10 most possible outliers by the test of pointDistGESD
            deredden=True,          # correct the Galactic extinction
            #wave_range=[1150, 1e9], # trim input wavelength
            wave_range=[1200, 1e9], # trim input wavelength
            wave_mask=None,         # 2-D array, mask the given range(s)

            # host decomposition parameters
            decompose_host=(rec["loglbol"] < 46),  # If True, the host galaxy-QSO decomposition will be applied
            host_prior=False,         # If True, adopt prior-informed method to assist decomposition (PCA only)
            host_prior_scale=0.2,     # scale of prior penalty; smaller if prior affects fitting too much

            host_line_mask=True,      # mask galaxy line region when subtracting from original spectra
            decomp_na_mask=True,      # mask narrow line region during decomposition
            qso_type='global',        # PCA template name for quasar

            # npca_qso=10,              # number of quasar templates
            # host_type='BC03',         # PCA template name for galaxy
            # npca_gal=5,               # number of galaxy templates

            npca_qso=npca_qso,              # number of quasar templates
            host_type='BC03',         # PCA template name for galaxy
            npca_gal=10,               # number of galaxy templates
            
            # continuum model fit parameters
            Fe_uv_op=True,            # If True, fit continuum with UV and optical FeII template
            poly=True,                # If True, include polynomial component to account for dust reddening
            BC=False,                 # If True, fit continuum with Balmer continua from 1000 to 3646A
            initial_guess=None,       # initial parameters for continuum model
            rej_abs_conti=False,      # iteratively reject 3σ outlier absorption pixels in continuum
            n_pix_min_conti=100,      # minimum negative pixels for host continuum fit rejection

            # emission line fit parameters
            linefit=True,             # If True, fit emission lines
            rej_abs_line=False,       # If True, iteratively reject 3σ outlier absorption pixels in lines

            # fitting method selection
            MC=True,                  # Monte Carlo resampling for error array
            MCMC=False,               # Markov Chain Monte Carlo sampling
            nsamp=20,                 # number of MC trials or MCMC samples

            # advanced fitting parameters
            param_file_name=parfilename,  # qso fitting parameter FITS file
            nburn=20,                 # burn-in samples for MCMC
            nthin=10,                 # return every n-th MCMC sample
            epsilon_jitter=0.,        # initial jitter for Gaussians to avoid local minima

            # customize the results
            save_result=True,         # save fitting results to a FITS file
            save_fits_name=None,      # output name for result FITS
            save_fits_path='./results/pysqo_fits',       # output path for result FITS
            plot_fig=True,            # plot fitting results
            save_fig=True,            # save fitting figures
            plot_corner=True,         # plot corner plot if MCMC=True

            # debugging mode
            verbose=False,            # turn debugging output on/off

            # sublevel parameters for figure plot and emcee
            kwargs_plot={
                'save_fig_path': save_fig_path,  # path to save figures
                'broad_fwhm': 1200                 # km/s, lower limit to classify as broad component
            },
            kwargs_conti_emcee={},
            kwargs_line_emcee={}
        )

        def _safe_float(x):
            try:
                # unwrap numpy scalars/arrays/masked values
                x = np.asarray(x).squeeze()
                if isinstance(x, (bytes, bytearray)):
                    x = x.decode("ascii", "ignore")
                return float(x)
            except Exception:
                return np.nan

        conti_dict = {
            name: _safe_float(val)
            for name, val in zip(q_mle.conti_result_name, q_mle.conti_result)
        }
        conti_dict['z'] = rec["z"]

        # L2500_int
        L_ok = np.isfinite(conti_dict.get('L2500_int', np.nan)) and np.isfinite(conti_dict.get('L2500_int_err', np.nan))
                
        if L_ok:
            m_2500, m_2500_err = compute_apparent_mag_2500_astropy(conti_dict, logL_col='L2500_int', logL_err_col='L2500_int_err')
            mag_errs = np.array([mag_err if (np.isfinite(mag_err) and mag_err >=0) else 0.0 
                                            for mag_err in rec["mags_err"].values()])
            m_2500_err = np.sqrt(m_2500_err**2 + np.mean(mag_errs)**2)
        else:
            m_2500, m_2500_err = -1e9, -1e9

        # L2500 reddened
        L_ok = np.isfinite(conti_dict.get('L2500', np.nan)) and np.isfinite(conti_dict.get('L2500_err', np.nan))
        if L_ok:
            m_2500_reddened, m_2500_reddened_err = compute_apparent_mag_2500_astropy(conti_dict, logL_col='L2500', logL_err_col='L2500_err')
            mag_errs = np.array([mag_err if (np.isfinite(mag_err) and mag_err >=0) else 0.0 
                                for mag_err in rec["mags_err"].values()])
            m_2500_reddened_err = np.sqrt(m_2500_reddened_err**2 + np.mean(mag_errs)**2)
        else:
            m_2500_reddened, m_2500_reddened_err = -1e9, -1e9


        try:
            alpha_lambda = conti_dict.get('PL_slope', -99)
            z = conti_dict['z']

            filt_i = filters.load_filter('sdss2010-i')
            host_contr = q_mle.host if q_mle.decompose_host else 0.0
            lam_obs = q_mle.wave * (1 + z)
            #f_lam_obs = (q_mle.flux - host_contr) / (1.0 + z)
            f_lam_obs = q_mle.f_conti_model / (1.0 + z)
            apparent_mag_i_obs = filt_i.get_ab_magnitude(1e-17*f_lam_obs*u.erg/u.s/u.cm**2/u.AA, lam_obs*u.AA)
            K_i = 2.5*(alpha_lambda + 1.0)*np.log10(1.0 + z)
            apparent_mag_i_rest = apparent_mag_i_obs - K_i
        except Exception as e:
            print(f"[ERROR] apparent_mag_i_rest {rec.get('object_id','?')} ({rec.get('sdss_name','?')}) (z {rec.get('z','?')}): {e}")
            apparent_mag_i_rest, apparent_mag_i_obs = -1e9, -1e9
            delta_m_avg = -1e9
            delta_mags = {}

        result.update(
            delta_m_avg=delta_m_avg,
            delta_mags=delta_mags,
            apparent_mag_i_rest=apparent_mag_i_rest,
            apparent_mag_i_obs=apparent_mag_i_obs,
            apparent_mag_2500=m_2500,
            apparent_mag_2500_err=m_2500_err,
            apparent_mag_2500_reddened=m_2500_reddened,
            apparent_mag_2500_reddened_err=m_2500_reddened_err,
            f_host_2500=conti_dict.get('frac_host_2500', 0),
            f_host_4200=conti_dict.get('frac_host_4200', 0),
            f_host_5100=conti_dict.get('frac_host_5100', 0),
            alpha_lambda=conti_dict['PL_slope'],
            alpha_lambda_err=conti_dict['PL_slope_err'],
            redchi=q_mle.conti_fit.redchi,
            L2500=conti_dict['L2500'],
            L2500_err=conti_dict['L2500_err'],
            L2500_int=conti_dict['L2500_int'],
            L2500_int_err=conti_dict['L2500_int_err'],
            EBV=conti_dict['EBV'],
            EBV_ccm89=conti_dict['EBV_ccm89'],
            bands_obj=bands_obj,
            bands_used=bands_used,
            bands_dropped=bands_dropped,
        )

        print(f"[INFO] ({rec.get('sdss_name','?')}): redchi={result['redchi']:.3f}, npca_qso={npca_qso}, delta_m_avg={result['delta_m_avg']:.3f}, m_i_rest={result['apparent_mag_i_rest']:.3f}, m_2500={result['apparent_mag_2500']:.3f}±{result['apparent_mag_2500_err']:.3f}, f_host_5100={result['f_host_5100']:.3f}, alpha_lambda={result['alpha_lambda']:.3f}±{result['alpha_lambda_err']:.3f}")
        print(f"  EBV={result['EBV']:.3f}, EBV_ccm89={result['EBV_ccm89']:.3f}")

        return result

    except Exception as e:
        # swallow errors per object; keep defaults
        print(f"[ERROR] z {rec.get('z','?')} Object {rec.get('object_id','?')} ({rec.get('sdss_name','?')}): {e}")
        return result

# --------------------------- CLI & Main ---------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DR16Q crossmatch, optional SDSS spectrum download, and QSOFit processing.")
    # p.add_argument("--input-csv",help="Path to sample CSV.")
    p.add_argument("fpath_in", help="Path to either csv or h5 fits with mag means.")
    p.add_argument("fpath_out", help="Output file for QSOFit results.")
    p.add_argument("--filter_csv", default=None,
                   help="Optional CSV file with object_id column to filter input.")
    p.add_argument("--dr16q-fits", default="data/dr16q_prop_May01_2024.fits",
                   help="Path to DR16Q FITS catalog.")
    p.add_argument("--cache-dir", default="data/spectra_cache",
                   help="Directory for cached spectra FITS.")
    p.add_argument("--max-sep", type=float, default=1.0,
                   help="Max match separation in arcsec.")
    p.add_argument("--N", type=int, default=None,
                   help="Optional limit on number of rows from input CSV to consider before matching.")
    p.add_argument("--skip", type=int, default=None, help="Optional number of rows to skip at start of input CSV.")
    p.add_argument("--download", action="store_true",
                   help="If set, download (and cache) all matched spectra and exit.")
    # p.add_argument("--nproc", type=int, default=max(1, (os.cpu_count() or 2) - 1),
    #            help="Number of parallel worker processes for QSOFit.")
    p.add_argument("--filter_object_id", nargs="+", help="List of object IDs to filter.")
    p.add_argument("--filter_sdss_name", nargs="+", help="List of sdss_names to filter.")
    p.add_argument("--sequential", action="store_true",
                   help="If set, use sequential (non-parallel) QSOFit processing.")

    p.add_argument("--spectral_fit_csv", default=None, type=str, 
                   help="Optional CSV file with spectral fit results to merge into output.")
    p.add_argument("--zquery", action="store_true", help="If set, perform zquery on all matched spectra and exit.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.fpath_in.endswith(".h5"):
        sample_df = load_agn_data(args.fpath_in, apply_cut=False, only_load=True)
    elif args.fpath_in.endswith(".csv"):
        sample_df = pd.read_csv(args.fpath_in, dtype={"object_id": str})
    else:
        raise ValueError(f"Unknown input file type: {args.fpath_in}")

    exclusion_sdss_names = [
        '221120.38+010905.6', # wrong redshift
        '024555.35+005332.6'  # weird spectra
    ]
    mask_exclude = ~sample_df['sdss_name'].astype(str).isin(exclusion_sdss_names)
    print(f"Excluding {np.sum(~mask_exclude)} objects by sdss_name in exclusion list")
    sample_df = sample_df[mask_exclude].reset_index(drop=True)

    if args.filter_csv is not None:
        filter_df = pd.read_csv(args.filter_csv)
        if 'object_id' not in filter_df.columns:
            raise ValueError(f"Filter CSV {args.filter_csv} missing object_id column")
        filter_ids = set(str(oid).strip() for oid in filter_df['object_id'].values if str(oid).strip())
        sample_df = sample_df[sample_df['object_id'].astype(str).str.strip().isin(filter_ids)]
        print(f"[INFO] After filtering with {args.filter_csv}, {len(sample_df)} rows remain")

    if args.filter_sdss_name is not None:
        sample_df = sample_df[sample_df['sdss_name'].astype(str).isin(args.filter_sdss_name)]
        print(f"[INFO] After filtering with {len(args.filter_sdss_name)} sdss_names, {len(sample_df)} rows remain")



    sample_df = sample_df[args.skip:] if args.skip is not None else sample_df
    sample_df = sample_df[:args.N] if args.N is not None else sample_df
    sample_df['object_id'] = sample_df['object_id'].astype(str).str.strip()
    #quasar_dict_list = read_quasars_from_hdf5(args.fpath_in, N=args.N)
    quasar_dict_list = sample_df.to_dict(orient="records")

    # 1) Match sample to DR16Q
    data_cat = match_sample_to_dr16q(
        sample_df=sample_df,
        dr16q_fits=args.dr16q_fits,
        max_sep_arcsec=args.max_sep,
    )

    # 2) If --download, fetch all spectra and exit
    if args.download:
        SDSS.clear_cache()
        N = len(data_cat)
        errors = []
        for i in tqdm(range(N), desc="Downloading spectra"):
            row = data_cat[i]
            plate, fiber, mjd = int(row['PLATE']), int(row['FIBERID']), int(row['MJD'])
            sdss_name = str(row['SDSS_NAME'])
            try:
                _, from_cache = fetch_spectrum_fits(sdss_name, plate, fiber, mjd, cache_dir=args.cache_dir)
                # you can log from_cache if you want
            except Exception as e:
                errors.append((i, sdss_name, str(e)))
        print(f"[DONE] Attempted {N} downloads. Errors: {len(errors)}")
        if errors:
            for i, name, msg in errors[:10]:
                print(f"  - {i}:{name} -> {msg}")
        return  # Exit after download-only path
    
    if args.zquery:
        SDSS.clear_cache()
        N = len(data_cat)
        results = []
        errors = []
        for i in tqdm(range(N), desc="Downloading spectra"):
            row = data_cat[i]
            plate, fiber, mjd = int(row['PLATE']), int(row['FIBERID']), int(row['MJD'])
            sdss_name = str(row['SDSS_NAME'])
            try:
                res = zquery(plate, mjd, fiber)
                res |= dict(object_id=str(row['object_id']), sdss_name=sdss_name, ra=float(row['RA']), dec=float(row['DEC']))
                if res is not None:
                    results.append(res)
                else:
                    errors.append((i, sdss_name, "zquery returned None"))
            except Exception as e:
                errors.append((i, sdss_name, str(e)))
            #break  # TEMPORARY: only do one for testing
        print(f"[DONE] Attempted {N} zquery. Errors: {len(errors)}")
        if errors:
            for i, name, msg in errors[:10]:
                print(f"  - {i}:{name} -> {msg}")
        if results:
            csv_file = args.fpath_out
            field_names = list(results[0].keys())
            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=field_names)
                writer.writeheader()
                for row in results:
                    writer.writerow(row)
            print(f"[OK] Saved zquery results to {csv_file}")
        return  # Exit after zquery path

    os.makedirs('results/pysqo_fits', exist_ok=True)
    os.makedirs('data/pyqsofit', exist_ok=True)
    # 3) Otherwise, proceed to QSOFit processing (expects cached spectra)
    create_qsopar_fits(path_ex=f'data/pyqsofit', parfilename=f'qsopar_{prefix}_{suffix}.fits', overwrite=True)
    # Build worker records so we don't try to pickle big astropy tables
    records = []
    colnames = set(data_cat.colnames)
    for i in range(len(data_cat)):
        row = data_cat[i]
        z = float(row['Z_SYS'])
        bands_dropped = bands_bluer_than_lyman_alpha(z)
        #bands_dropped = row['bands_dropped']
        # TEMPORARY: skip objects without any dropped bands
        # if len(bands_dropped) == 0:
        #     continue 
        bands_obj = [b for b in ['g', 'r', 'i'] if b not in bands_dropped]
        rec = dict(
            bands=bands_obj,
            bands_dropped=bands_dropped,
            object_id=str(row['object_id']),
            sdss_name=str(row['SDSS_NAME']),
            plate=int(row['PLATE']),
            fiber=int(row['FIBERID']),
            mjd=int(row['MJD']),
            z=float(row['Z_SYS']),
            loglbol=float(row['LOGLBOL']),
            mags={
                b: (float(row[f'mean_corrected_{b}']) if f'mean_corrected_{b}' in colnames else np.nan)
                for b in bands_obj
            },
            mags_err={
                b: float(row[f"mean_{b}_err"]) if f"mean_{b}_err" in colnames else np.nan
                for b in bands_obj
            },
            ra=float(row['RA']),
            dec=float(row['DEC']),
        )
        records.append(rec)

    # Run QSOFit twice: once with npca_qso=0, once with npca_qso=2
    results_0 = {}
    results_1 = {}
    results_2 = {}

    for npca_qso, results_dict in [(0, results_0), (1, results_1), (2, results_2)]:
        save_fig_path = os.path.join('plots', 'pyqsofit', prefix, f'npca_qso_{npca_qso}')
        os.makedirs(save_fig_path, exist_ok=True)
        worker = partial(run_qsofit_record, npca_qso=npca_qso, cache_dir=args.cache_dir, 
                        path_ex=f'data/pyqsofit', parfilename=f'qsopar_{prefix}_{suffix}.fits',
                        save_fig_path=save_fig_path)
        chunksize = 1
        if args.sequential:
            # Sequential processing
            for rec in tqdm(records, desc=f"Processing npca_qso={npca_qso} (sequential)"):
                res = worker(rec)
                obj_id = res.get("object_id", None)
                if obj_id is not None:
                    res['npca_qso'] = npca_qso
                    results_dict[obj_id] = res
            print(f"Collected {len(results_dict)} results for npca_qso={npca_qso} (sequential)")
        else:
            # Parallel processing
            with Pool(processes=num_cores) as pool:
                with tqdm(total=len(records), desc=f"Processing npca_qso={npca_qso}", dynamic_ncols=True, smoothing=0.0) as pbar:
                    for res in pool.imap_unordered(worker, records, chunksize=chunksize):
                        obj_id = res.get("object_id", None)
                        if obj_id is not None:
                            res['npca_qso'] = npca_qso
                            results_dict[obj_id] = res
                        pbar.update(1)
            print(f"Collected {len(results_dict)} results for npca_qso={npca_qso}")

    # Select best result (lowest redchi) for each object
    results = {}
    for obj_id in results_0.keys():
        res0 = results_0[obj_id]
        res1 = results_1[obj_id]
        res2 = results_2[obj_id]

        #best_res = min([res0, res1, res2], key=lambda r: r["redchi"])

        # Start with the simplest model
        best_res = res0

        # Only accept npca_qso=1 if it improves redchi by at least 20%
        if res1["redchi"] <= 0.8 * best_res["redchi"]:
            best_res = res1

        # Only accept npca_qso=2 if it improves redchi by at least 20% over the current best
        if res2["redchi"] <= 0.8 * best_res["redchi"]:
            best_res = res2

        # TODO: If chi2 is still bad, use BC=True models

        # Add redchi for each npca_qso to the best result
        best_res["redchi_npca_qso0"] = res0.get("redchi", np.nan)
        best_res["redchi_npca_qso1"] = res1.get("redchi", np.nan)
        best_res["redchi_npca_qso2"] = res2.get("redchi", np.nan)

        results[obj_id] = best_res
        print(f"Object {obj_id}: selected npca_qso={best_res['npca_qso']} with redchi={best_res['redchi']:.3f} (0:{res0['redchi']:.3f}, 1:{res1['redchi']:.3f}, 2:{res2['redchi']:.3f})")

    # Update each quasar dict with fields from results
    for quasar in quasar_dict_list:
        obj_id = str(quasar.get('object_id'))
        quasar.update(results[obj_id])

    # Call for each object
    for q in quasar_dict_list:
        plot_delta_mags_vs_z_single_quasar(q)

    plot_delta_mags_vs_z(quasar_dict_list)
    plot_f_host_4200_vs_z(quasar_dict_list)

    write_hdf5_file(quasar_dict_list, args.fpath_out)
    
    # Also write results to CSV
    csv_file=args.fpath_out.replace(".h5", "_specfitted.csv")

    field_names = [
        'object_id',
        'plate',
        'mjd',
        'fiber',
        'z',
        "delta_m_avg",
        "delta_mags",
        "apparent_mag_i_rest",
        "apparent_mag_i_obs",
        "apparent_mag_2500",
        "apparent_mag_2500_err",
        "apparent_mag_2500_reddened",
        "apparent_mag_2500_reddened_err",
        "f_host_2500",
        "f_host_4200",
        "f_host_5100",
        "alpha_lambda",
        "alpha_lambda_err",
        'redchi_npca_qso0',
        'redchi_npca_qso1',
        'redchi_npca_qso2',
        "redchi",
        "npca_qso",
        'sdss_name',               
        'EBV',
        'EBV_ccm89',
        'L2500_int',
        'L2500_int_err',
        'L2500',
        'L2500_err',
        'bands_obj',
        'bands_used',
        'bands_dropped',
    ]

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for obj_id in results:
            writer.writerow(results[obj_id])

    print(f"[OK] Saved CSV results to {csv_file}")

    print(f"[OK] Saved results to {args.fpath_out}")
if __name__ == "__main__":
    main()
