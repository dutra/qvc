#!/usr/bin/env python3
import os
from functools import partial
from multiprocessing import Pool, cpu_count
import sys
import timeit
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import trange, tqdm

from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

from astroquery.sdss import SDSS
warnings.filterwarnings("ignore")



# --------------------------- Utility ---------------------------------
def compute_apparent_mag_2500_astropy(conti_table, logL_col='L2500', logL_err_col='L2500_err',
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


def load_spec_from_cache(sdss_name, cache_dir="data/spectra_cache"):
    """Return cached FITS HDUList if available, else None."""
    cache_file = os.path.join(cache_dir, f"{sdss_name}.fits")
    if os.path.exists(cache_file):
        return fits.open(cache_file, memmap=False)
    return None


def match_sample_to_dr16q(sample_csv, dr16q_fits, max_sep_arcsec=2.0, limit=None):
    """Load sample CSV and DR16Q, crossmatch within max_sep_arcsec, return (data_cat_table, sample_df_matched).
    Ensures 1–to–1 matches by keeping the closest pair per AGN (and per SDSS if needed)."""
    sample_df = pd.read_csv(sample_csv)
    if limit is not None:
        sample_df = sample_df.iloc[:limit].copy()
    sample_df['object_id'] = sample_df['object_id'].astype(str).str.strip()

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
        col = f'mags_mean_{b}'
        if col in sample_df_matched.columns:
            data_cat[f'mean_corrected_{b}'] = sample_df_matched[col].to_numpy()

    data_cat['object_id'] = sample_df_matched['object_id'].to_numpy()

    return data_cat, sample_df_matched



def run_qsofit_record(rec, cache_dir="data/spectra_cache", path_ex="."):
    """
    Worker-safe version of QSOFit runner.
    `rec` is a plain dict containing only the fields needed for one object.
    """
    from speclite import filters
    from pyqsofit.PyQSOFit import QSOFit

    QSOFit.set_mpl_style()


    # default result (so we always return a complete row even on error)
    result = dict(
        object_id=rec["object_id"],
        sdss_name=rec["sdss_name"],
        apparent_mag_2500=-1e9,
        apparent_mag_2500_err=-1e9,
        f_host_4200=-1e9,
        alpha_lambda=-1e9,
        alpha_lambda_err=-1e9,
        redchi=-1e9,
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
        bands = ['g', 'r', 'i']
        sdss_filters = filters.load_filters(*[f'sdss2010-{b}' for b in bands])
        delta_mags, weights = [], []

        for b, filt in zip(bands, sdss_filters):
            mag_fiber = rec["mags"].get(b, np.nan)
            if not np.isfinite(mag_fiber) or mag_fiber < 0:
                continue
            mag_synth = filt.get_ab_magnitude(
                1e-17 * flux * u.erg / u.s / u.cm**2 / u.AA,
                lam * u.AA
            )
            delta_mags.append(mag_fiber - mag_synth)
            weights.append(1.0)

        delta_mags = np.array(delta_mags) if delta_mags else np.array([0.0])
        weights    = np.array(weights)    if weights    else np.array([1.0])
        mask = np.isfinite(delta_mags)
        delta_m_avg = np.average(delta_mags[mask], weights=weights[mask]) if np.any(mask) else 0.0

        scale = 10 ** (-0.4 * delta_m_avg)
        flux_scaled = flux * scale

        q_mle = QSOFit(lam, flux_scaled, err, rec["z"], path=path_ex)
        q_mle.Fit(
            name=f"{rec['z']:.2f}_{rec['sdss_name']}_{rec['plate']}-{rec['mjd']}-{rec['fiber']}",  # customize the name of given targets. Default: plate-mjd-fiber
            
            # preprocessing parameters
            nsmooth=1,              # do n-pixel smoothing to the raw input flux and err spectra
            and_mask=False,         # delete the and masked pixels
            or_mask=False,          # delete the or masked pixels
            reject_badpix=False,    # reject 10 most possible outliers by the test of pointDistGESD
            deredden=True,          # correct the Galactic extinction
            wave_range=[1150, 1e9], # trim input wavelength
            wave_mask=None,         # 2-D array, mask the given range(s)

            # host decomposition parameters
            decompose_host=(rec["loglbol"] < 46),  # If True, the host galaxy-QSO decomposition will be applied
            host_prior=False,         # If True, adopt prior-informed method to assist decomposition (PCA only)
            host_prior_scale=0.2,     # scale of prior penalty; smaller if prior affects fitting too much

            host_line_mask=True,      # mask galaxy line region when subtracting from original spectra
            decomp_na_mask=True,      # mask narrow line region during decomposition
            qso_type='global',        # PCA template name for quasar
            npca_qso=10,              # number of quasar templates
            host_type='BC03',         # PCA template name for galaxy
            npca_gal=5,               # number of galaxy templates
            
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
            param_file_name='qsopar.fits',  # qso fitting parameter FITS file
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
                'save_fig_path': './plots/pyqso',  # path to save figures
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

        L_ok = np.isfinite(conti_dict.get('L2500', np.nan)) and np.isfinite(conti_dict.get('L2500_err', np.nan))
        if L_ok:
            m_2500, m_2500_err = compute_apparent_mag_2500_astropy(conti_dict)
        else:
            m_2500, m_2500_err = -1e9, -1e9

        result.update(
            apparent_mag_2500=m_2500,
            apparent_mag_2500_err=m_2500_err,
            f_host_4200=conti_dict.get('frac_host_4200', -99),
            alpha_lambda=conti_dict.get('PL_slope', -99),
            alpha_lambda_err=conti_dict.get('PL_slope_err', -99),
            redchi=q_mle.conti_fit.redchi
        )
        return result

    except Exception:
        # swallow errors per object; keep defaults
        return result

# --------------------------- CLI & Main ---------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DR16Q crossmatch, optional SDSS spectrum download, and QSOFit processing.")
    p.add_argument("--input-csv", default="data/csv/aug4_sample_chisqg10_ebv005sn3_magsmean.csv",
                   help="Path to sample CSV.")
    p.add_argument("--dr16q-fits", default="data/dr16q_prop_May01_2024.fits",
                   help="Path to DR16Q FITS catalog.")
    p.add_argument("--cache-dir", default="data/spectra_cache",
                   help="Directory for cached spectra FITS.")
    p.add_argument("--max-sep", type=float, default=1.0,
                   help="Max match separation in arcsec.")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional limit on number of rows from input CSV to consider before matching.")
    p.add_argument("--download", action="store_true",
                   help="If set, download (and cache) all matched spectra and exit.")
    p.add_argument("--out-csv", default="data/csv/aug11_sample_chisqg10_ebv005sn3_fittedm2500.csv",
                   help="Output CSV for QSOFit results.")
    p.add_argument("--nproc", type=int, default=max(1, (os.cpu_count() or 2) - 1),
               help="Number of parallel worker processes for QSOFit.")
    return p.parse_args()


def main():
    args = parse_args()

    # 1) Match sample to DR16Q
    data_cat, sample_df_matched = match_sample_to_dr16q(
        sample_csv=args.input_csv,
        dr16q_fits=args.dr16q_fits,
        max_sep_arcsec=args.max_sep,
        limit=args.limit
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

    # 3) Otherwise, proceed to QSOFit processing (expects cached spectra)
    # Build worker records so we don't try to pickle big astropy tables
    records = []
    colnames = set(data_cat.colnames)
    for i in range(len(data_cat)):
        row = data_cat[i]
        rec = dict(
            object_id=str(row['object_id']),
            sdss_name=str(row['SDSS_NAME']),
            plate=int(row['PLATE']),
            fiber=int(row['FIBERID']),
            mjd=int(row['MJD']),
            z=float(row['Z_SYS']),
            loglbol=float(row['LOGLBOL']),
            mags={
                b: (float(row[f'mean_corrected_{b}']) if f'mean_corrected_{b}' in colnames else np.nan)
                for b in ['g', 'r', 'i']
            },
        )
        records.append(rec)

    # Build a quick lookup of original order from the input CSV
    input_df = pd.read_csv(args.input_csv)
    #original_order = list(input_df["object_id"].astype(str))
    original_order = list(sample_df_matched["object_id"].astype(str))

    worker = partial(run_qsofit_record, cache_dir=args.cache_dir, path_ex=".")

    chunksize = 1  # small so progress bar updates frequently
    results_dict = {}

    with Pool(processes=args.nproc) as pool:
        with tqdm(total=len(records), desc="Processing objects", dynamic_ncols=True, smoothing=0.0) as pbar:
            for res in pool.imap_unordered(worker, records, chunksize=chunksize):
                obj_id = res.get("object_id", None)
                if obj_id is not None:
                    results_dict[obj_id] = res
                pbar.update(1)

    print(f"Collected {len(results_dict)} results out of {len(records)} records")

    # Reassemble results in the same order as input_csv, removing blank results
    ordered_results = [results_dict.get(oid, {}) for oid in original_order]
    filtered_results = [res for res in ordered_results if res]  # remove blank dicts

    pd.DataFrame(filtered_results).to_csv(args.out_csv, index=False)
    print(f"[OK] Saved results to {args.out_csv}")
if __name__ == "__main__":
    main()
