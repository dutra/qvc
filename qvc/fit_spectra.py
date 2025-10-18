#!/usr/bin/env python3
from functools import partial
import multiprocessing
from multiprocessing import Pool, cpu_count
import os, csv, glob, shutil, argparse

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

prefix = os.environ.get('PREFIX', "pyqsofit")
suffix = os.environ.get('SUFFIX', "pyqsofit")
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
import csv
import traceback
from speclite.filters import FilterResponse
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from hubble_utils import read_quasars_from_hdf5, write_hdf5_file, match_radec
import warnings
import h5py

from astroquery.sdss import SDSS
#warnings.filterwarnings("ignore")

bands = ['u', 'g', 'r', 'i', 'z']


def _as_value(x, unit=None):
    """Return float array from Quantity or ndarray. If ndarray and a unit is
    provided, just assume that unit (no conversion)."""
    if hasattr(x, "to"):  # Astropy Quantity
        return np.asarray(x.to(unit).value if unit is not None else x.value, dtype=float)
    return np.asarray(x, dtype=float)

def synth_ab_mag_overlap(
    filt,
    lam,                # Quantity [Å] or ndarray (assumed Å)
    f_lambda,           # Quantity [erg s^-1 cm^-2 Å^-1] or ndarray in those units
    tmin_rel=0.01,
    min_overlap_frac=0.85,
    min_pts=30,
    debug=False,
):
    """
    Match speclite FilterResponse.get_ab_magnitude exactly, but allow partial
    spectral coverage by trimming the filter to the spectrum–filter overlap.
    """

    # --- Filter grid (robust to ndarray or Quantity) ---
    w_attr = getattr(filt, "wavelength", None)
    if w_attr is None:
        w_attr = getattr(filt, "waves", None)
    if w_attr is None:
        raise ValueError("Filter has no wavelength grid (expected .wavelength or .waves).")

    w = _as_value(w_attr, u.AA)                    # Å
    t = _as_value(getattr(filt, "response"), None) # dimensionless throughput

    if not np.any(np.isfinite(t)):
        raise ValueError("Filter response has no finite values.")
    tmax = np.nanmax(t)
    if not np.isfinite(tmax) or tmax <= 0:
        raise ValueError("Filter maximum throughput is not positive/finite.")

    # Effective passband threshold (avoid tiny tails)
    in_eff = t >= (tmin_rel * tmax)
    if not np.any(in_eff):
        raise ValueError("No effective band found under tmin_rel threshold.")
    w_eff = w[in_eff]
    wmin_eff, wmax_eff = w_eff[0], w_eff[-1]
    band_width_eff = wmax_eff - wmin_eff

    # --- Spectrum (robust to ndarray or Quantity) ---
    lam_A = _as_value(lam, u.AA)                               # Å
    fl_A  = _as_value(f_lambda, u.erg/u.s/u.cm**2/u.AA)        # erg s^-1 cm^-2 Å^-1

    m = np.isfinite(lam_A) & np.isfinite(fl_A)
    if not np.any(m):
        raise ValueError("Spectrum has no finite wavelength/flux values.")
    lam_A = lam_A[m]; fl_A = fl_A[m]
    if lam_A.ndim != 1: lam_A = lam_A.ravel()
    if fl_A.ndim  != 1: fl_A  = fl_A.ravel()
    if not np.all(np.diff(lam_A) > 0):
        idx = np.argsort(lam_A)
        lam_A, fl_A = lam_A[idx], fl_A[idx]

    # Overlap with effective passband
    spec_min, spec_max = lam_A[0], lam_A[-1]
    ov_min = max(wmin_eff, spec_min)
    ov_max = min(wmax_eff, spec_max)
    if ov_max <= ov_min:
        raise ValueError(f"No overlap with effective band [{wmin_eff:.1f}, {wmax_eff:.1f}] Å.")

    overlap_frac = float((ov_max - ov_min) / band_width_eff)
    if overlap_frac < min_overlap_frac:
        raise ValueError(f"Insufficient overlap: {overlap_frac:.2f} < {min_overlap_frac:.2f}.")

    # If full coverage, defer to speclite (bit-for-bit match).
    if (spec_min <= w[0]) and (spec_max >= w[-1]):
        lam_q = lam if hasattr(lam, "to") else (lam * u.AA)
        f_q   = f_lambda if hasattr(f_lambda, "to") else (f_lambda * u.erg/u.s/u.cm**2/u.AA)
        mag = filt.get_ab_magnitude(f_q, lam_q)
        if debug:
            print(f"[synth_ab_mag_overlap] full coverage; overlap=1.00 mag={float(mag):.4f}")
        return float(mag), 1.0

    # Build a trimmed filter on a dense grid strictly within the overlap
    n_grid = max(min_pts, 3)
    w_trim = np.linspace(ov_min, ov_max, n_grid)

    # Evaluate throughput on this grid; prefer speclite's callable interpolation if present
    if callable(getattr(filt, "__call__", None)):
        t_trim = np.asarray(filt(w_trim * u.AA), dtype=float)
    else:
        t_trim = np.interp(w_trim, w, t)

    # Construct a temporary FilterResponse; this recomputes the AB zeropoint
    # with speclite's photon weighting on the *trimmed* band.
    tmp_meta = getattr(filt, "meta", None)
    tmp = FilterResponse(w_trim * u.AA, t_trim, meta=tmp_meta)

    lam_q = lam if hasattr(lam, "to") else (lam * u.AA)
    f_q   = f_lambda if hasattr(f_lambda, "to") else (f_lambda * u.erg/u.s/u.cm**2/u.AA)

    mag = tmp.get_ab_magnitude(f_q, lam_q)

    if debug:
        print(f"[synth_ab_mag_overlap] eff=[{wmin_eff:.1f},{wmax_eff:.1f}]Å "
              f"spec=[{spec_min:.1f},{spec_max:.1f}]Å overlap={overlap_frac:.3f} "
              f"n_grid={n_grid} mag={float(mag):.4f}")

    return float(mag), overlap_frac


def sdss_bands_affected_by_lya(z, buffer=100.0):
    """
    SDSS ugriz bands whose *rest-frame blue edge* falls below 1216+buffer Å,
    i.e., likely contaminated by Lyα line/forest.

    Args:
        z (float): source redshift
        buffer (float): safety margin in Å (typ. 50–200)

    Returns:
        list[str]: bands to drop (contaminated)
    """
    # SDSS (full transmission) edges in observed Å (λ_min, λ_max)
    edges_obs = {
        "u": (3055.11, 4030.64),
        "g": (3797.64, 5553.04),
        "r": (5418.23, 6994.42),
        "i": (6692.41, 8400.32),
        "z": (7964.70, 10873.33),
    }

    cutoff = 1216.0 + buffer
    affected = []
    for b, (lo_obs, _hi_obs) in edges_obs.items():
        lo_rest = lo_obs / (1.0 + z)
        if lo_rest < cutoff:
            affected.append(b)
    return affected
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


def compute_apparent_mag_2500_astropy(logL2500, logL2500_err, z):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    c = 2.99792458e10  # cm/s
    lambda_ = 2500e-8  # cm

    DL = cosmo.luminosity_distance(z).to(u.cm).value  # cm

    log_Lnu = logL2500 + np.log10(lambda_ / c)
    log_fnu = log_Lnu - np.log10(4 * np.pi * DL**2 * (1 + z))
    m_ab = -2.5 * log_fnu - 48.60
    m_ab_err = 2.5 * logL2500_err

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
    
    #print(sample_df[['object_id', 'sdss_name']])
    data_cat['object_id'] = sample_df_matched['object_id'].to_numpy()
    if 'z' in sample_df_matched.columns:
        data_cat['z'] = sample_df_matched['z'].to_numpy()
    else:
        data_cat['z'] = data_cat['Z_SYS']  # fallback to DR16Q redshift if not in sample

    # Check that z is close to Z_SYS
    if not np.allclose(sample_df_matched['z'], data_cat_full[sdss_keep]['Z_SYS'], rtol=0, atol=1e-3):
        not_close = ~np.isclose(sample_df_matched['z'], data_cat_full[sdss_keep]['Z_SYS'], rtol=0, atol=1e-3)
        if np.any(not_close):
            print("Redshift mismatch for the following objects:")
            for idx in np.where(not_close)[0]:
                print(f"object_id: {sample_df_matched.iloc[idx]['object_id']}, "
                      f"sdss_name: {sample_df_matched.iloc[idx]['sdss_name']}, "
                      f"sample z: {sample_df_matched.iloc[idx]['z']}, "
                      f"DR16Q Z_SYS: {data_cat_full[sdss_keep][idx]['Z_SYS']}")
    # Add photometry columns (now lengths match)
    for b in ['u', 'g', 'r', 'i', 'z']:
        mags_mean_col = f'mags_mean_{b}' # direct mean from LC
        mean_col = f'mean_{b}'  # LC fitting mean (dm)
        # If mean_col is not >= 0, assign 0
        mean_vals = sample_df_matched[mean_col].to_numpy()
        mean_vals = np.where(mean_vals >= 0, mean_vals, 0)
        data_cat[f'mean_corrected_{b}'] = sample_df_matched[mags_mean_col].to_numpy() + mean_vals
        data_cat[f'{mean_col}_err'] = sample_df_matched[f'{mean_col}_err'].to_numpy()

    #print(data_cat['z'])
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

        (3728.48, 'OII', 3650, 3800, 'OII3728',          1,    0.0, 0.0, 1e10, 1e-3, 3.333e-4, 0.00169, 0.01, 1, 1, 0, 0.001, 1),
        (3426.84, 'NeV', 3380, 3480, 'NeV3426',          1,    0.0, 0.0, 1e10, 1e-3, 3.333e-4, 0.00169, 0.01, 0, 0, 0, 0.001, 1),

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
        # (1445., 1465.), # often affected by absorption
        (1690., 1705.),
        (1770., 1810.),
        (1970., 2400.),
        (2480., 2675.),
        (2925., 3400.),
        (3450., 3700.), # BC bump
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
        ('PL_slope_blue',    -0.5,  -5.0,  0.0,   1), # Blue slope of the power-law (PL) continuum
        ('PL_slope_red',     -0.1,  -5.0,  0.0,   1), # Red slope of the power-law (PL) continuum
        ('PL_break_wave',    4000,  2000,  5000, 1), # Break wavelength of the power-law (PL) continuum
        ('Balmer_norm',  0.0,   0.0,   1e10,  1),  # Balmer continuum normalization (< 3646 Å)
        ('Balmer_Te',  15000, 10000, 50000,  1),   # Balmer continuum Te
        ('Balmer_Tau',   0.5,   0.1,   2.0,   1),  # Balmer continuum τ
        ('Balmer_vel',  3000, 12000, 18000,  1), # Velocity broadening of the Balmer continuum at < 3646 AA [lnlambda]
        ('conti_a_0',    0.01,   0.0,  .1,  1), # Polynomial terms
        ('conti_a_1',    -2.0,   -100.0,  100.0,  0),
        ('conti_a_2',    -2.0,   -100.0,  100.0,  0),
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


def run_qsofit_record(rec, npca_qso, decomp_host, BC, cache_dir="data/spectra_cache", 
                      path_ex=f'data/pyqsofit', parfilename=f'qsopar.fits',
                      save_fig_path=f'./plots/pyqso/', save_fits_path=f'./results/pyqso_fits/',
                      allow_partial_band_overlap=False, MC_samples=50, poly=False):
    """
    Worker-safe version of QSOFit runner.
    `rec` is a plain dict containing only the fields needed for one object.
    """
    from speclite import filters
    from pyqsofit.PyQSOFit import QSOFit

    #print("\n\n====================== Working on SDSS_NAME={} (z={:.2f}) =================".format(rec["sdss_name"], rec["z"]))            

    QSOFit.set_mpl_style()


    # default result (so we always return a complete row even on error)
    result = dict(
        z=rec["z"],
        plate=rec["plate"],
        fiber=rec["fiber"],
        mjd=rec["mjd"],
        delta_mag_u=-1e9,
        delta_mag_r=-1e9,
        delta_mag_g=-1e9,
        delta_mag_i=-1e9,
        delta_mag_z=-1e9,
        delta_m_avg=-1e9,
        loglbol=rec["loglbol"],
        object_id=rec["object_id"],
        sdss_name=rec["sdss_name"],
        apparent_mag_i_rest=-1e9,
        apparent_mag_2500=-1e9,
        apparent_mag_2500_err=-1e9,
        apparent_mag_2500_reddened=-1e9,
        apparent_mag_2500_reddened_err=-1e9,
        f_host_2500=-99,
        #f_host_4200=-99,
        f_host_5100=-99,
        alpha_lambda=-1e9,
        alpha_lambda_err=-1e9,
        redchi=1e9,
        aic=1e9,
        bic=1e9,
        redchi2_conti_full=1e9,
        ebv_fs=-1e9,
        euv_fs=-1e9,
        log_L2500_fs=-1e9,
        log_L2500_fs_err=-1e9,
        log_L2500_int_fs=-1e9,
        log_L2500_int_fs_err=-1e9,
        reddening_integral=-1e9,
        reddening_proxy=-1e9,
        conti_a_0=-1e9,
        conti_a_0_err=-1e9,
        bands_used=[]
    )


    try:
    # cached spectrum
        hdul = load_spec_from_cache(rec["sdss_name"], cache_dir=cache_dir)
        if hdul is None:
            # keep default values; return early
            return result

        lam  = 10 ** hdul[1].data['loglam']                  # [Å]
        flux = hdul[1].data['flux']                          # [erg/s/cm^2/Å]
        err  = 1.0 / np.sqrt(hdul[1].data['ivar'])           # 1-sigma

        # Absolute flux calibration (g,r,i)
        sdss_filters = filters.load_filters(*[f'sdss2010-{b}' for b in bands])
        delta_mags, weights = {}, []

        dropped_bands = sdss_bands_affected_by_lya(rec['z'])
        bands_used = []

        for b, filt in zip(bands, sdss_filters):
            if b in dropped_bands:
                print(f"[INFO] Dropping band {b} for {rec['sdss_name']} (z={rec['z']:.2f}) due to Lyα forest contamination.")
                continue
            #print(f"[INFO] Processing band {b} for {rec['sdss_name']} (z={rec['z']:.2f})")
            #print(f"[DEBUG] lam range: {lam.min():.1f} - {lam.max():.1f} Å")
            try:
                mag_fiber = rec[f"mean_corrected_{b}"]
                if not np.isfinite(mag_fiber) or mag_fiber < 0:
                    print(f"[WARN] Invalid mag_fiber for band {b} for {rec['sdss_name']}: {mag_fiber}")
                    continue
                if allow_partial_band_overlap:
                    # Compute synthetic mag robustly over the overlap only
                    mag_synth, overlap_frac = synth_ab_mag_overlap(
                        filt, lam * u.AA, 1e-17 * flux * u.erg / u.s / u.cm**2 / u.AA,
                        tmin_rel=0.01,          # 1% throughput threshold
                        min_overlap_frac=0.85,  # accept slightly partial coverage
                        min_pts=30
                    )
                else:
                    mag_synth = filt.get_ab_magnitude(
                        1e-17 * flux * u.erg / u.s / u.cm**2 / u.AA,
                        lam * u.AA
                    )
                    overlap_frac = 1.0
                dm = mag_fiber - mag_synth
                delta_mags[b] = dm
                #print(f"[INFO] Band {b}: mag_fiber={mag_fiber:.3f}, mag_synth={mag_synth:.3f}, Δm={dm:.3f} (overlap={overlap_frac:.2f})")
                
                # weight by photometric mag uncertainty if available; else equal weight
                sig_m = rec[f'mean_{b}_err']
                w = (1.0 / (sig_m**2) if np.isfinite(sig_m) and sig_m > 0 else 1.0) * overlap_frac
                weights.append(w)
                bands_used.append(b)
            except Exception as e:
                print(f"[ERROR mag_fiber] Error processing band {b} for {rec['sdss_name']}: {e}")
                continue

        mag_errs = np.array([rec[f'mean_{b}_err'] if (np.isfinite(rec[f'mean_{b}_err']) and rec[f'mean_{b}_err'] >=0) else 0.0 
                                    for b in bands_used])  # only for used bands
        dm_arr = np.array([delta_mags[b] for b in bands_used], dtype=float)
        w_arr  = np.array(weights, dtype=float)
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
        # print(f"[INFO] Using bands {bands_used} for {rec['sdss_name']}: delta_m_avg={delta_m_avg:.3f} ± {sigma_dm:.3f} mag")
        scale = 10 ** (-0.4 * delta_m_avg)

        scale = 1.0  # TEMPORARY OVERRIDE: disable absolute flux rescaling

        flux_scaled = flux * scale

        err_scaled  = err  * scale      # IMPORTANT: scale the uncertainties too
        
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
            #wave_range=[1200, 1e9],  # trim input wavelength
            wave_range=[1200, 7997.75],  # trim input wavelength to avoid edge effects in host decomposition
            wave_mask=None,         # 2-D array, mask the given range(s)

            # host decomposition parameters
            decompose_host=decomp_host,      # If True, the host galaxy-QSO decomposition will be applied
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
            poly=poly,                # If True, include polynomial component to account for dust reddening
            BC=BC,                 # If True, fit continuum with Balmer continua from 1000 to 3646A
            initial_guess=None,       # initial parameters for continuum model
            rej_abs_conti=True,      # iteratively reject 3σ outlier absorption pixels in continuum
            n_pix_min_conti=100,      # minimum negative pixels for host continuum fit rejection

            # emission line fit parameters
            linefit=True,             # If True, fit emission lines
            rej_abs_line=False,       # If True, iteratively reject 3σ outlier absorption pixels in lines

            # fitting method selection
            MC=(MC_samples>0),                  # Monte Carlo resampling for error array
            MCMC=False,               # Markov Chain Monte Carlo sampling
            nsamp=MC_samples,                 # number of MC trials or MCMC samples

            # advanced fitting parameters
            param_file_name=parfilename,  # qso fitting parameter FITS file
            nburn=20,                 # burn-in samples for MCMC
            nthin=10,                 # return every n-th MCMC sample
            epsilon_jitter=0.,        # initial jitter for Gaussians to avoid local minima

            # customize the results
            save_result=True,         # save fitting results to a FITS file
            save_fits_name=None,      # output name for result FITS
            save_fits_path=save_fits_path,       # output path for result FITS
            plot_fig=True,            # plot fitting results
            save_fig=True,            # save fitting figures
            plot_corner=True,         # plot corner plot if MCMC=True

            # debugging mode
            verbose=False,            # turn debugging output on/off

            # sublevel parameters for figure plot and emcee
            kwargs_plot={
                'save_fig_path': save_fig_path,  # path to save figures
                'broad_fwhm': 1200,                 # km/s, lower limit to classify as broad component
                'disable_secondary_plot': True,  # if True, disable the secondary plot with masked regions
            },
            kwargs_conti_emcee={},
            kwargs_line_emcee={}
        )

        conti_dict = {
            name: _safe_float(val)
            for name, val in zip(q_mle.conti_result_name, q_mle.conti_result)
        }
        conti_dict['z'] = rec["z"]

        L_ok = np.isfinite(conti_dict['L2500_int']) and np.isfinite(conti_dict.get('L2500_int_err', -1e9))
        if L_ok:
            m_2500, m_2500_err = compute_apparent_mag_2500_astropy(conti_dict['L2500_int'], conti_dict.get('L2500_int_err', -1e9), z=rec['z'])
            m_2500_err = np.sqrt(m_2500_err**2 + np.mean(mag_errs)**2)
        else:
            print(f"[WARN] L2500_int not finite for {rec['sdss_name']} (z={rec['z']:.2f})")
            m_2500, m_2500_err = -1e9, -1e9

        # L2500 reddened
        L_ok = np.isfinite(conti_dict['L2500']) and np.isfinite(conti_dict.get('L2500_err', -1e9))
        if L_ok:
            m_2500_reddened, m_2500_reddened_err = compute_apparent_mag_2500_astropy(conti_dict['L2500'], conti_dict.get('L2500_err', -1e9), z=rec['z'])
            m_2500_reddened_err = np.sqrt(m_2500_reddened_err**2 + np.mean(mag_errs)**2)
        else:
            print(f"[WARN] L2500 not finite for {rec['sdss_name']} (z={rec['z']:.2f})")
            m_2500_reddened, m_2500_reddened_err = -1e9, -1e9


        try:
            alpha_lambda = conti_dict.get('PL_slope_blue', -99)
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
            delta_mags = np.array([])


        try:
            # compute reddened integral
            wave_eval = np.linspace(2500-1500, 2500+1500, 5000)
            poly = q_mle.F_poly_conti(wave_eval, q_mle.conti_fit.params.valuesdict())
            reddening_integral = -np.trapz(poly, wave_eval)
            #print(f"[INFO] reddening_integral {rec.get('object_id','?')} ({rec.get('sdss_name','?')}) (z {rec.get('z','?')}): {reddening_integral:.3f}")
        except Exception as e:
            print(f"[ERROR] reddening_integral {rec.get('object_id','?')} ({rec.get('sdss_name','?')}) (z {rec.get('z','?')}): {e}")
            traceback.print_exc()
            reddening_integral = -1e9

        try:
            wave_eval = np.linspace(2500-1500, 2500+1500, 5000)
            lam = wave_eval
            poly = q_mle.F_poly_conti(wave_eval, q_mle.conti_fit.params.valuesdict())
            df = poly                         # PL-subtracted in linear units
            f_pl = q_mle.PL(lam, q_mle.conti_fit.params.valuesdict())        # reconstruct your PL continuum used in the subtraction
            r = np.clip(df / np.maximum(f_pl, 1e-300), -1e6, 1e6)  # fractional residual
            x = np.log(lam)
            reddening_proxy = - np.trapz(r, x) / (x.max() - x.min())      # dimensionless
            #print(f"[INFO] reddening_proxy {rec.get('object_id','?')} ({rec.get('sdss_name','?')}) (z {rec.get('z','?')}): {reddening_proxy:.3f}")
        except Exception as e:
            print(f"[ERROR] reddening_proxy {rec.get('object_id','?')} ({rec.get('sdss_name','?')}) (z {rec.get('z','?')}): {e}")
            traceback.print_exc()
            reddening_proxy = -1e9

        # print("redchi: ", q_mle.conti_fit.redchi)
        # print("aic: ", q_mle.conti_fit.aic)
        # print("bic: ", q_mle.conti_fit.bic)

        result.update(
            delta_m_avg=delta_m_avg,
            delta_mag_u=delta_mags.get('u', -1e9),
            delta_mag_r=delta_mags.get('r', -1e9),
            delta_mag_g=delta_mags.get('g', -1e9),
            delta_mag_i=delta_mags.get('i', -1e9),
            delta_mag_z=delta_mags.get('z', -1e9),
            apparent_mag_i_rest=apparent_mag_i_rest,
            apparent_mag_i_obs=apparent_mag_i_obs,
            apparent_mag_2500=m_2500,
            apparent_mag_2500_err=m_2500_err,
            apparent_mag_2500_reddened=m_2500_reddened,
            apparent_mag_2500_reddened_err=m_2500_reddened_err,
            f_host_2500=conti_dict.get('frac_host_4200', -1), # in pyqsofit, frac_host_4200 is actually at 2500A
            #f_host_4200=conti_dict.get('frac_host_4200', 0),
            f_host_5100=conti_dict.get('frac_host_5100', -1),
            conti_a_0=conti_dict['conti_a_0'],
            conti_a_0_err=conti_dict.get('conti_a_0_err', -1e9),
            alpha_lambda=conti_dict['PL_slope_blue'],
            alpha_lambda_err=conti_dict.get('PL_slope_blue_err', -1e9),
            redchi=q_mle.conti_fit.redchi,
            aic=q_mle.conti_fit.aic,
            bic=q_mle.conti_fit.bic,
            redchi2_conti_full=q_mle.redchi2_conti_full,
            ebv_fs=conti_dict.get('EBV', -99),
            euv_fs=conti_dict.get('EUV', -99),
            log_L2500_fs=conti_dict.get('L2500', -1e9),
            log_L2500_fs_err=conti_dict.get('L2500_err', -1e9),
            log_L2500_int_fs=conti_dict.get('L2500_int', -1e9),
            log_L2500_int_fs_err=conti_dict.get('L2500_int_err', -1e9),
            reddening_integral=reddening_integral,
            reddening_proxy=reddening_proxy,
            bands_used=bands_used,
        )
        #print(f"[INFO] Result for {rec['sdss_name']} (z={rec['z']:.2f}): {result}")
        return result | conti_dict

    except Exception as e:
        # swallow errors per object; keep defaults but print traceback
        print(f"[ERROR] z {rec.get('z','?')} Object {rec.get('object_id','?')} ({rec.get('sdss_name','?')}): {e}")
        traceback.print_exc()
        return result

# --------------------------- CLI & Main ---------------------------------
# ------------------------------
# Provided writer: DO NOT CHANGE
# ------------------------------
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

# ------------------------------
# Argparse
# ------------------------------


# ------------------------------
# argparse (minor tweak to wording)
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DR16Q crossmatch, optional SDSS spectrum download, QSOFit processing → CSV (collect/select).")
    p.add_argument("fpath_in", help="Path to HDF5 with mag means (input to build the sample).")
    p.add_argument("fpath_out", help="Output CSV file for all QSOFit runs (collect) / Input CSV (select).")
    p.add_argument("--mode", choices=["collect", "select"], required=True,
                   help="collect: run all configs, write one CSV; select: read CSV, mark best=True per object_id.")
    p.add_argument("--filter_csv", default=None,
                   help="Optional CSV with an object_id column to filter input rows.")
    p.add_argument("--dr16q-fits", default="data/dr16q_prop_May01_2024.fits",
                   help="Path to DR16Q FITS catalog.")
    p.add_argument("--cache-dir", default="data/spectra_cache",
                   help="Directory for cached spectra FITS.")
    p.add_argument("--max-sep", type=float, default=1.0,
                   help="Max match separation in arcsec.")
    p.add_argument("--N", type=int, default=None,
                   help="Optional limit on number of rows before matching.")
    p.add_argument("--skip", type=int, default=None, help="Optional number of rows to skip at start.")
    p.add_argument("--download", action="store_true",
                   help="If set, download (and cache) all matched spectra and exit.")
    p.add_argument("--filter_object_id", nargs="+", help="List of object IDs to filter.")
    p.add_argument("--filter_sdss_name", nargs="+", help="List of sdss_names to filter.")
    p.add_argument("--spectral_fit_csv", default=None, type=str,
                   help="Optional CSV of external spectral-fit results to merge into output (by object_id if present, else sdss_name).")
    p.add_argument("--allow_partial_band_overlap", action="store_true",
                   help="Allow bands with partial wavelength overlap when computing synthetic mags.")
    p.add_argument("--enable_BC", action="store_true",
                   help="Include BC=True runs (otherwise only BC=False).")
    p.add_argument("--nproc", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="Parallel worker processes for QSOFit.")
    p.add_argument("--MC_samples", type=int, default=50,
                   help="Number of Monte Carlo samples per object (0 to disable MC).")
    p.add_argument("--enable_poly", action="store_true",
                   help="Include polynomial component in continuum fit to account for dust reddening.")

    return p.parse_args()

# ------------------------------
# Utilities you already have
# ------------------------------
def option_label(npca_qso, decomp_host, BC, poly):
    return f"npca_qso={npca_qso}|decomp_host={decomp_host}|BC={BC}|poly={poly}"

def load_quasar_core_list(fpath_in):
    return read_quasars_from_hdf5(fpath_in)

def prepare_sample_df(quasar_list, filter_sdss_name, filter_csv, N, skip):
    for q in quasar_list:
        for i, b in enumerate(['u', 'g', 'r', 'i', 'z']):
            if len(q['mags_means']) != 5:
                raise ValueError(f"Expected 5 mags_means, got {len(q['mags_means'])} for object_id {q.get('object_id','?')}")
            q[f'mags_mean_{b}'] = q['mags_means'][i]
            if f'mean_{b}' in q:
                q[f'mean_corrected_{b}'] = q[f'mags_mean_{b}'] + q[f'mean_{b}']
            else:
                q[f'mean_corrected_{b}'] = q[f'mags_mean_{b}']
            #print(f"[DEBUG] object_id {q.get('object_id','?')} band {b}: mags_mean {q[f'mags_mean_{b}']}, mean_{b} {q.get(f'mean_{b}','?')} -> mean_corrected_{b} {q[f'mean_corrected_{b}']}")
    sample_df = pd.DataFrame.from_records(quasar_list)

    exclusion_sdss_names = [
        '221120.38+010905.6', # wrong redshift
        '024555.35+005332.6'  # weird spectra
    ]
    mask_exclude = ~sample_df['sdss_name'].astype(str).isin(exclusion_sdss_names)
    print(f"Excluding {np.sum(~mask_exclude)} objects by sdss_name in exclusion list")
    sample_df = sample_df[mask_exclude].reset_index(drop=True)

    if filter_sdss_name is not None:
        sample_df = sample_df[sample_df['sdss_name'].astype(str).isin(filter_sdss_name)]

    if filter_csv is not None:
        filter_df = pd.read_csv(filter_csv)
        if 'object_id' not in filter_df.columns:
            raise ValueError(f"Filter CSV {filter_csv} missing object_id column")
        filter_ids = set(str(oid).strip() for oid in filter_df['object_id'].values if str(oid).strip())
        sample_df = sample_df[sample_df['object_id'].astype(str).str.strip().isin(filter_ids)]
        print(f"[INFO] After filtering with {filter_csv}, {len(sample_df)} rows remain")

    sample_df = sample_df[skip:] if skip is not None else sample_df
    sample_df = sample_df[:N] if N is not None else sample_df
    sample_df['object_id'] = sample_df['object_id'].astype(str).str.strip()
    return sample_df

def match_to_dr16q(sample_df, dr16q_fits, max_sep_arcsec=1.0):
    cols = ["RA", "DEC", "SDSS_NAME", "PLATE", "FIBERID", "MJD", "Z_SYS", "LOGLBOL"]
    data_cat_table = Table.read(dr16q_fits, hdu=1)
    data_cat_full = data_cat_table[cols].to_pandas()
    print(f"[INFO] Loaded DR16Q catalog with {len(data_cat_full)} rows from {dr16q_fits}")

    df_matched, unmatched_object_ids = match_radec(
        sample_df, data_cat_full,
        populate_cols=['SDSS_NAME', 'PLATE', 'FIBERID', 'MJD', 'Z_SYS', 'LOGLBOL', 'RA', 'DEC'],
        ra_col_a='ra', dec_col_a='dec', ra_col_b='RA', dec_col_b='DEC',
        max_sep_arcsec=max_sep_arcsec, add_prefix=False
    )
    df_matched['plate'] = df_matched['PLATE']
    df_matched['fiber'] = df_matched['FIBERID']
    df_matched['mjd'] = df_matched['MJD']
    df_matched['z'] = df_matched['Z_SYS']
    df_matched['loglbol'] = df_matched['LOGLBOL']
    df_matched['sdss_name'] = df_matched['SDSS_NAME'].astype(str).str.strip()
    df_matched['object_id'] = df_matched['object_id'].astype(str).str.strip()
    df_matched['ra'] = df_matched['RA']
    df_matched['dec'] = df_matched['DEC']
    print(f"[INFO] After matching to DR16Q, {len(df_matched)} objects matched, {len(unmatched_object_ids)} unmatched.")
    return df_matched, unmatched_object_ids

# ------------------------------
# Ranking helper (same logic as before)
# ------------------------------
def pick_best_fit(models, redchi_ok=(0.7, 1.5), target=1.0):
    ranked = sorted(
        enumerate(models),
        key=lambda t: (abs(t[1].get('redchi2_conti_full', np.inf) - target),
                       t[1].get('aic', np.inf),
                       t[1].get('bic', np.inf),
                       t[0])
    )
    i_best, best = ranked[0]
    lo, hi = redchi_ok
    if not (lo <= best.get('redchi2_conti_full', np.inf) <= hi):
        for _, m in ranked:
            if lo <= m.get('redchi2_conti_full', np.inf) <= hi:
                return m
    return best

# ------------------------------
# COLLECT → write ALL runs to ONE CSV
# ------------------------------
def run_collect(args):
    num_cores = args.nproc

    quasar_list = load_quasar_core_list(args.fpath_in)
    sample_df = prepare_sample_df(
        quasar_list, args.filter_sdss_name, args.filter_csv, args.N, args.skip
    )

    df_matched, _ = match_to_dr16q(sample_df, args.dr16q_fits, args.max_sep)
    records = df_matched.to_dict(orient='records')

    # Optional download-only branch
    if args.download:
        SDSS.clear_cache()
        N = len(records)
        errors = []
        for i in tqdm(range(N), desc="Downloading spectra"):
            row = records[i]
            plate, fiber, mjd = int(row['PLATE']), int(row['FIBERID']), int(row['MJD'])
            sdss_name = str(row['SDSS_NAME'])
            try:
                _, _ = fetch_spectrum_fits(sdss_name, plate, fiber, mjd, cache_dir=args.cache_dir)
            except Exception as e:
                errors.append((i, sdss_name, str(e)))
        print(f"[DONE] Attempted {N} downloads. Errors: {len(errors)}")
        if errors:
            for i, name, msg in errors[:10]:
                print(f"  - {i}:{name} -> {msg}")
        return

    os.makedirs('results/pysqo_fits', exist_ok=True)
    os.makedirs('data/pyqsofit', exist_ok=True)
    create_qsopar_fits(path_ex='data/pyqsofit', parfilename=f'qsopar_{prefix}_{suffix}.fits', overwrite=True)

    rows = []  # accumulate CSV rows

    def run_parallel(npca_qso, decomp_host, BC, poly):
        save_fig_path = os.path.join('plots', 'pyqsofit', prefix, f'npca_qso_{npca_qso}_decomp_host_{decomp_host}_BC_{BC}_poly_{poly}')
        os.makedirs(save_fig_path, exist_ok=True)
        save_fits_path = os.path.join('results', 'pysqo_fits', prefix, f'npca_qso_{npca_qso}_decomp_host_{decomp_host}_BC_{BC}_poly_{poly}')
        os.makedirs(save_fits_path, exist_ok=True)

        worker = partial(
            run_qsofit_record,
            npca_qso=npca_qso, decomp_host=decomp_host, BC=BC, poly=poly,
            cache_dir=args.cache_dir, path_ex='data/pyqsofit',
            parfilename=f'qsopar_{prefix}_{suffix}.fits',
            save_fits_path=save_fits_path,
            allow_partial_band_overlap=args.allow_partial_band_overlap,
            save_fig_path=save_fig_path,
            MC_samples=args.MC_samples,
        )

        chunksize = 1
        with Pool(processes=num_cores) as pool:
            with tqdm(total=len(records), desc=f"Processing npca_qso={npca_qso} decomp_host={decomp_host} BC={BC} poly={poly}",
                       dynamic_ncols=True, smoothing=0.0) as pbar:
                for res in pool.imap_unordered(worker, records, chunksize=chunksize):
                    # Build a flat row for CSV
                    obj_id = str(res.get("object_id", ""))

                    row = {
                        "object_id": obj_id,
                        "sdss_name": res.get("sdss_name", ""),
                        "plate": res.get("plate", res.get("PLATE")),
                        "fiber": res.get("fiber", res.get("FIBERID")),
                        "mjd": res.get("mjd", res.get("MJD")),
                        "z": res.get("z", res.get("Z_SYS")),
                        "loglbol": res.get("loglbol", res.get("LOGLBOL")),
                        "ra": res.get("ra", res.get("RA")),
                        "dec": res.get("dec", res.get("DEC")),
                        "npca_qso": npca_qso,
                        "decomp_host": bool(decomp_host),
                        "BC": bool(BC),
                        "poly": bool(poly),
                        "run_label": option_label(npca_qso, decomp_host, BC, poly),                    
                    } | res  # merge all result keys
                    rows.append(row)
                    pbar.update(1)

    poly_list = [False, True] if args.enable_poly else [False]
    BC_list = [False, True] if args.enable_BC else [False]
    for poly in poly_list:
        for BC in BC_list:
            run_parallel(npca_qso=-1, decomp_host=False, BC=BC, poly=poly)  # no host decomposition
            for npca in [0, 1, 2, 5, 10]:
                run_parallel(npca_qso=npca, decomp_host=True, BC=BC, poly=poly)

    # poly = False
    # BC = False
    # npca_qso = -1
    # decomp_host = False
    # run_parallel(npca_qso=npca_qso, decomp_host=decomp_host, BC=BC, poly=poly)

    # Optional merge of external spectral-fit CSV (row-wise add columns)
    # if args.spectral_fit_csv and os.path.exists(args.spectral_fit_csv):
    #     df_out = pd.DataFrame(rows)
    #     df_spec = pd.read_csv(args.spectral_fit_csv)

    #     join_key = "object_id" if "object_id" in df_spec.columns else ("sdss_name" if "sdss_name" in df_spec.columns else None)
    #     if join_key is None:
    #         print(f"[WARN] spectral_fit_csv has neither object_id nor sdss_name; skipping merge.")
    #     else:
    #         df_out = df_out.merge(df_spec, on=join_key, how="left", suffixes=("", "_spec"))
    #     df_out["best"] = False  # not selected yet
    #     df_out.to_csv(args.fpath_out, index=False)
    # else:
    df_out = pd.DataFrame(rows)
    df_out["best"] = False
    df_out.to_csv(args.fpath_out, index=False)

    print(f"[OK] Wrote all runs to CSV: {args.fpath_out}")

# ------------------------------
# SELECT → mark best per object_id in SAME CSV
# ------------------------------
def run_select(args):

    # ---- Load
    csv_path = args.fpath_out
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded CSV with {len(df)} rows from {csv_path}")

    # ---- Coerce essentials
    # ---- Minimal coercions
    df["object_id"] = df["object_id"].astype(str)
    for c in ["redchi2_conti_full", "aic", "bic", "loglbol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Coerce booleans simply
    for c in ["decomp_host", "BC", "poly"]:
        if c in df.columns and df[c].dtype != bool:
            df[c] = df[c].astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"])

    # set npca_qso to -1 when there is no host decomposition
    df.loc[~df["decomp_host"], "npca_qso"] = -1


    # ---- Drop unusable rows
    df = df[np.isfinite(df["redchi2_conti_full"])].copy()
    if df.empty:
        return  # nothing to do

    # ---- Define the proxy: start with redchi2, then penalize
    df["redchip"] = df["redchi2_conti_full"]

    # do not consider failed host decompositions
    mask_failed = (df["decomp_host"] == True) & (df.get("f_host_2500", 1.0) < 1e-3)
    if mask_failed.any():
        df.loc[mask_failed, "redchip"] = 1e9

    # do not consider polynomial fits
    df.loc[df["poly"] == True, "redchip"] *= 1e9

    # 20% penalty if BC=True
    df.loc[df["BC"] == True, "redchip"] *= 1

    # 100% penalty if npca_qso == 0
    df.loc[(df["npca_qso"].isin([0])), "redchip"] *= 1e9


    # low lbol
    # 50% penalty if decomp_host == False
    df.loc[(df["loglbol"] < 46.5) & (df["decomp_host"] == False), "redchip"] *= 1.5
    # 20% penalty if npca_qso 2
    df.loc[(df["loglbol"] < 46.5) & (df["decomp_host"] == True) & (df["npca_qso"].isin([1,2])), "redchip"] *= 1
    # 20% penalty if npca_qso != 0 and 5, 10
    df.loc[(df["loglbol"] < 46.5) & (df["decomp_host"] == True) & (df["npca_qso"].isin([5, 10])), "redchip"] *= 1.5

    
    # high lbol
    # 50% penalty if npca_qso == 0
    #df.loc[(df["loglbol"] >= 46.5) & (df["decomp_host"] == True), "redchip"] *= 1.5

    # ---- Pick the minimum redchip per object
    idx_best = df.groupby("object_id", sort=False)["redchip"].idxmin()

    # ---- Mark winners
    df["best"] = False
    df.loc[idx_best, "best"] = True
    print(f"[INFO] Marked {df['best'].sum()} best fits out of {len(df)} total rows.")

    # ---- Save back (overwrite input for simplicity)
    df.to_csv(csv_path, index=False)

    print(f"[OK] Selected best fits and updated CSV: {csv_path}")

# ------------------------------
# Main
# ------------------------------
def main():
    args = parse_args()
    if args.mode == "collect":
        run_collect(args)
    else:
        run_select(args)

if __name__ == "__main__":
    main()