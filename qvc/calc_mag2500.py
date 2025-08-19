import numpy as np
import pandas as pd
from astropy.io import fits
import warnings

import glob, os, sys, timeit
import matplotlib
import numpy as np

from pyqsofit.PyQSOFit import QSOFit
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import warnings

from astropy.coordinates import SkyCoord

from speclite import filters

warnings.filterwarnings("ignore")

QSOFit.set_mpl_style()

import h5py
from tqdm import tqdm



agn_df = pd.read_csv('data/csv/aug4_sample_chisqg10_ebv005sn3_magsmean.csv')
agn_df = agn_df[:4000]
#agn_df = pd.concat([agn_df.iloc[:1000], agn_df.iloc[2000:3000]], ignore_index=True)

#agn_df = pd.read_csv('data/csv/aug8_stone_merged_magsmean.csv')

#
print(agn_df.head())
print("Length of agn_df:", len(agn_df))

bands = ['u', 'g', 'r', 'i', 'z']

from astropy import units as u
from astropy.table import Table

with fits.open('data/dr16q_prop_May01_2024.fits') as hdul:
    data_cat = hdul[1].data
    #data_cat = data_cat[data_cat['Z_SYS'] <1]

    coords_sdss = SkyCoord(
        ra=data_cat['RA'],
        dec=data_cat['DEC'],
        unit=(u.deg, u.deg),
        frame='icrs'
    )

    # ...existing code...
    # Ensure all object_ids are strings and stripped
    agn_df['object_id'] = agn_df['object_id'].astype(str).str.strip()
    #obj_mean_corr_map = {str(obj['object_id']).strip(): obj['mean_corrected'] for obj in objs}

    coords_agn = SkyCoord(
        ra=agn_df['ra'],
        dec=agn_df['dec'],
        unit=(u.deg, u.deg),
        frame='icrs'
    )

    # Match using search_around_sky to find all pairs within a certain separation
    max_sep = 1.0 * u.arcsec  # maximum separation for matching

    idx_sdss, idx_agn, sep2d, _ = coords_agn.search_around_sky(coords_sdss, max_sep)

    # Store matches in a DataFrame for further analysis
    matches_df = pd.DataFrame({
        'agn_idx': idx_agn,
        'sdss_idx': idx_sdss,
        'sep_arcsec': sep2d.arcsec
    })

    # Only keep data_cat rows where there is a match
    matched_sdss_indices = np.unique(idx_sdss)
    data_cat = data_cat[matched_sdss_indices]
    # Add apparent_mag_2500 from agn_df into data_cat by matching indices
    # Assume agn_df has a column 'apparent_mag_2500' and matches to data_cat via matched_sdss_indices

    # Only keep rows in agn_df that correspond to matched_sdss_indices
    agn_df_matched = agn_df.iloc[idx_agn].reset_index(drop=True)
    print("Length of agn_df_matched:", len(agn_df_matched))

    # Add apparent_mag_2500 from agn_df_matched to data_cat as a new column
    # Assumes agn_df_matched['apparent_mag_2500'] exists and matches the order of data_cat after filtering


    # Convert data_cat to Table if not already
    if not isinstance(data_cat, Table):
        data_cat = Table(data_cat)

    #data_cat['apparent_mag_2500'] = agn_df_matched['apparent_mag_2500'].values

    # Add the mean_corrected magnitudes for each band to data_cat
    # The order of agn_df_matched matches data_cat after filtering

    for i, band in enumerate(bands):
        data_cat[f'mean_corrected_{band}'] = agn_df_matched[f'mags_mean_{band}'].values
    data_cat['object_id'] = agn_df_matched['object_id'].values

# %%
path_ex = '.' #os.path.join(pyqsofit.__path__[0], '..', 'example')

# create a header
hdr0 = fits.Header()
hdr0['Author'] = 'Hengxiao Guo'
primary_hdu = fits.PrimaryHDU(header=hdr0)

line_priors = np.rec.array([
    (6564.61, 'Ha', 6400, 6800, 'Ha_br', 2, 0.0, 0.0, 1e10, 5e-3, 0.004, 0.05, 0.015, 0, 0, 0, 0.05, 1),
    (6564.61, 'Ha', 6400, 6800, 'Ha_na', 1, 0.0, 0.0, 1e10, 1e-3, 5e-4, 0.00169, 0.01, 1, 1, 0, 0.002, 1),
    (6549.85, 'Ha', 6400, 6800, 'NII6549', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 5e-3, 1, 1, 1, 0.001, 1),
    (6585.28, 'Ha', 6400, 6800, 'NII6585', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 5e-3, 1, 1, 1, 0.003, 1),
    (6718.29, 'Ha', 6400, 6800, 'SII6718', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 5e-3, 1, 1, 2, 0.001, 1),
    (6732.67, 'Ha', 6400, 6800, 'SII6732', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 5e-3, 1, 1, 2, 0.001, 1),

    (4862.68, 'Hb', 4640, 5100, 'Hb_br', 2, 0.0, 0.0, 1e10, 5e-3, 0.004, 0.05, 0.01, 0, 0, 0, 0.01, 1),
    (4862.68, 'Hb', 4640, 5100, 'Hb_na', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 0.01, 1, 1, 0, 0.002, 1),
    (4960.30, 'Hb', 4640, 5100, 'OIII4959c', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 0.01, 1, 1, 0, 0.002, 1),
    (5008.24, 'Hb', 4640, 5100, 'OIII5007c', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 0.01, 1, 1, 0, 0.004, 1),
    (4960.30, 'Hb', 4640, 5100, 'OIII4959w',   1, 0.0, 0.0, 1e10, 3e-3, 2.3e-4, 0.004,  0.01,  2, 2, 0, 0.001, 1),
    (5008.24, 'Hb', 4640, 5100, 'OIII5007w',   1, 0.0, 0.0, 1e10, 3e-3, 2.3e-4, 0.004,  0.01,  2, 2, 0, 0.002, 1),
    #(4687.02, 'Hb', 4640, 5100, 'HeII4687_br', 1, 0.0, 0.0, 1e10, 5e-3, 0.004,  0.05,   0.005, 0, 0, 0, 0.001, 1),
    #(4687.02, 'Hb', 4640, 5100, 'HeII4687_na', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 0.005, 1, 1, 0, 0.001, 1),
    
    (4341.68, 'Hg', 4200, 4400, 'Hg_br', 1, 0.0, 0.0, 1e10, 5e-3, 0.004, 0.05, 0.01, 0, 0, 0, 0.01, 1),
    (4341.68, 'Hg', 4200, 4400, 'Hg_na', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 0.01, 1, 1, 0, 0.002, 1),
    (4102.89, 'Hd', 4000, 4150, 'Hd_br', 1, 0.0, 0.0, 1e10, 5e-3, 0.004, 0.05, 0.01, 0, 0, 0, 0.01, 1),
    (4102.89, 'Hd', 4000, 4150, 'Hd_na', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 0.01, 1, 1, 0, 0.002, 1),

    #(3934.78, 'CaII', 3900, 3960, 'CaII3934' , 2, 0.0, 0.0, 1e10, 1e-3, 3.333e-4, 0.00169, 0.01, 99, 0, 0, -0.001, 1),

    #(3728.48, 'OII', 3650, 3800, 'OII3728', 1, 0.0, 0.0, 1e10, 1e-3, 3.333e-4, 0.00169, 0.01, 1, 1, 0, 0.001, 1),

    #(3426.84, 'NeV', 3380, 3480, 'NeV3426',    1, 0.0, 0.0, 1e10, 1e-3, 3.333e-4, 0.00169, 0.01, 0, 0, 0, 0.001, 1),
    #(3426.84, 'NeV', 3380, 3480, 'NeV3426_br', 1, 0.0, 0.0, 1e10, 5e-3, 0.0025,   0.02,   0.01, 0, 0, 0, 0.001, 1),

    (2798.75, 'MgII', 2700, 2900, 'MgII_br', 2, 0.0, 0.0, 1e10, 5e-3, 0.004, 0.05, 0.015, 0, 0, 0, 0.05, 1),
    (2798.75, 'MgII', 2700, 2900, 'MgII_na', 1, 0.0, 0.0, 1e10, 1e-3, 5e-4, 0.00169, 0.01, 1, 1, 0, 0.002, 1),

    (1908.73, 'CIII', 1700, 1970, 'CIII_br', 2, 0.0, 0.0, 1e10, 5e-3, 0.004, 0.05, 0.015, 99, 0, 0, 0.01, 1),
    #(1908.73, 'CIII', 1700, 1970, 'CIII_na',   1, 0.0, 0.0, 1e10, 1e-3, 5e-4,  0.00169, 0.01,  1, 1, 0, 0.002, 1),
    #(1892.03, 'CIII', 1700, 1970, 'SiIII1892', 1, 0.0, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.003, 1, 1, 0, 0.005, 1),
    #(1857.40, 'CIII', 1700, 1970, 'AlIII1857', 1, 0.0, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.003, 1, 1, 0, 0.005, 1),
    #(1816.98, 'CIII', 1700, 1970, 'SiII1816',  1, 0.0, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.0002, 1),
    #(1786.7,  'CIII', 1700, 1970, 'FeII1787',  1, 0.0, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.0002, 1),
    #(1750.26, 'CIII', 1700, 1970, 'NIII1750',  1, 0.0, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.001, 1),
    #(1718.55, 'CIII', 1700, 1900, 'NIV1718',   1, 0.0, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.001, 1),

    (1549.06, 'CIV', 1500, 1700, 'CIV_br', 2, 0.0, 0.0, 1e10, 5e-3, 0.004, 0.05, 0.015, 0, 0, 0, 0.05, 1),
    # (1549.06, 'CIV', 1500, 1700, 'CIV_na', 1, 0.0, 0.0, 1e10, 1e-3, 5e-4, 0.00169, 0.01, 1, 1, 0, 0.002, 1),
    #(1640.42, 'CIV', 1500, 1700, 'HeII1640',    1, 0.0, 0.0, 1e10, 1e-3, 5e-4,   0.00169, 0.008, 1, 1, 0, 0.002, 1),
    #(1663.48, 'CIV', 1500, 1700, 'OIII1663',    1, 0.0, 0.0, 1e10, 1e-3, 5e-4,   0.00169, 0.008, 1, 1, 0, 0.002, 1),
    #(1640.42, 'CIV', 1500, 1700, 'HeII1640_br', 1, 0.0, 0.0, 1e10, 5e-3, 0.0025, 0.02,   0.008, 1, 1, 0, 0.002, 1),
    #(1663.48, 'CIV', 1500, 1700, 'OIII1663_br', 1, 0.0, 0.0, 1e10, 5e-3, 0.0025, 0.02,   0.008, 1, 1, 0, 0.002, 1),

    (1402.06, 'SiIV', 1290, 1450, 'SiIV_OIV1', 1, 0.0, 0.0, 1e10, 5e-3, 0.002, 0.05,  0.015, 1, 1, 0, 0.05, 1),
    #(1396.76, 'SiIV', 1290, 1450, 'SiIV_OIV2', 1, 0.0, 0.0, 1e10, 5e-3, 0.002, 0.05,  0.015, 1, 1, 0, 0.05, 1),
    #(1335.30, 'SiIV', 1290, 1450, 'CII1335',   1, 0.0, 0.0, 1e10, 2e-3, 0.001, 0.015, 0.01,  1, 1, 0, 0.001, 1),
    #(1304.35, 'SiIV', 1290, 1450, 'OI1304',    1, 0.0, 0.0, 1e10, 2e-3, 0.001, 0.015, 0.01,  1, 1, 0, 0.001, 1),

    (1215.67, 'Lya', 1150, 1290, 'Lya_br', 3, 0.0, 0.0, 1e10, 5e-3, 0.002, 0.05, 0.02, 0, 0, 0, 0.05, 1),
    (1240.14, 'Lya', 1150, 1290, 'NV1240', 1, 0.0, 0.0, 1e10, 2e-3, 0.001, 0.01, 0.005, 0, 0, 0, 0.002, 1),
    # (1215.67, 'Lya', 1150, 1290, 'Lya_na', 1, 0.0, 0.0, 1e10, 1e-3, 5e-4, 0.00169, 0.01, 0, 0, 0, 0.002, 1),
    ],

    formats='float32,    a20,  float32, float32,      a20,  int32, float32, float32, float32, float32, float32, float32, float32,   int32,  int32,  int32, float32, int32',
    names=' lambda, compname,   minwav,  maxwav, linename, ngauss,  inisca,  minsca,  maxsca,  inisig,  minsig,  maxsig,    voff,  vindex, windex, findex,  fvalue,  vary')

# Header
hdr1 = fits.Header()
hdr1['lambda'] = 'Vacuum Wavelength in Ang'
hdr1['minwav'] = 'Lower complex fitting wavelength range'
hdr1['maxwav'] = 'Upper complex fitting wavelength range'
hdr1['ngauss'] = 'Number of Gaussians for the line'

# Can be set to negative for absorption lines if you want
hdr1['inisca'] = 'Initial guess of line scale [flux]'
hdr1['minsca'] = 'Lower range of line scale [flux]'
hdr1['maxsca'] = 'Upper range of line scale [flux]'

hdr1['inisig'] = 'Initial guess of linesigma [lnlambda]'
hdr1['minsig'] = 'Lower range of line sigma [lnlambda]'
hdr1['maxsig'] = 'Upper range of line sigma [lnlambda]'

hdr1['voff  '] = 'Limits on velocity offset from the central wavelength [lnlambda]'
hdr1['vindex'] = 'Entries w/ same NONZERO vindex constrained to have same velocity'
hdr1['windex'] = 'Entries w/ same NONZERO windex constrained to have same width'
hdr1['findex'] = 'Entries w/ same NONZERO findex have constrained flux ratios'
hdr1['fvalue'] = 'Relative scale factor for entries w/ same findex'

hdr1['vary'] = 'Whether or not to vary the parameter (set to 0 to fix the line parameter to initial values)'

# Save line info
hdu1 = fits.BinTableHDU(data=line_priors, header=hdr1, name='line_priors')

conti_windows = np.rec.array([
    #(1150., 1170.), 
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
    (8050., 8150.), # Continuum fitting windows (to avoid emission line, etc.)  [AA]
    ], 
    formats = 'float32,  float32',
    names =    'min,     max')

hdu2 = fits.BinTableHDU(data=conti_windows, name='conti_windows')

conti_priors = np.rec.array([
    ('Fe_uv_norm',  0.0,   0.0,   1e10,  1), # Normalization of the MgII Fe template [flux]
    ('Fe_uv_FWHM',  3000,  1200,  18000, 1), # FWHM of the MgII Fe template [AA]
    ('Fe_uv_shift', 0.0,   -0.01, 0.01,  1), # Wavelength shift of the MgII Fe template [lnlambda]
    ('Fe_op_norm',  0.0,   0.0,   1e10,  1), # Normalization of the Hbeta/Halpha Fe template [flux]
    ('Fe_op_FWHM',  3000,  1200,  18000, 1), # FWHM of the Hbeta/Halpha Fe template [AA]
    ('Fe_op_shift', 0.0,   -0.01, 0.01,  1), # Wavelength shift of the Hbeta/Halpha Fe template [lnlambda]
    ('PL_norm',     1.0,   0.0,   1e10,  1), # Normalization of the power-law (PL) continuum f_lambda = (lambda/3000)^-alpha
    ('PL_slope',    -1.5,  -5.0,  3.0,   1), # Slope of the power-law (PL) continuum
    ('Blamer_norm', 0.0,   0.0,   1e10,  1), # Normalization of the Balmer continuum at < 3646 AA [flux] (Dietrich et al. 2002)
    ('Balmer_Te',   15000, 10000, 50000, 1), # Te of the Balmer continuum at < 3646 AA [K?]
    ('Balmer_Tau',  0.5,   0.1,   2.0,   1), # Tau of the Balmer continuum at < 3646 AA
    ('conti_a_0',   0.0,   None,  None,  0), # 1st coefficient of the polynomial continuum
    ('conti_a_1',   0.0,   None,  None,  0), # 2nd coefficient of the polynomial continuum
    ('conti_a_2',   0.0,   None,  None,  0), # 3rd coefficient of the polynomial continuum
    # Note: The min/max bounds on the conti_a_0 coefficients are ignored by the code,
    # so they can be determined automatically for numerical stability.
    ],

    formats = 'a20,  float32, float32, float32, int32',
    names = 'parname, initial,   min,     max,     vary')

hdr3 = fits.Header()
hdr3['ini'] = 'Initial guess of line scale [flux]'
hdr3['min'] = 'FWHM of the MgII Fe template'
hdr3['max'] = 'Wavelength shift of the MgII Fe template'

hdr3['vary'] = 'Whether or not to vary the parameter (set to 0 to fix the continuum parameter to initial values)'


hdu3 = fits.BinTableHDU(data=conti_priors, header=hdr3, name='conti_priors')

measure_info = Table(
    [
        [[1350, 1450, 1700, 2500, 3000, 3500, 4200, 5100]],
        [[
            # [2240, 2650], 
            [4435, 4685],
        ]]
    ],
    names=([
        'cont_loc',
        'Fe_flux_range'
    ]),
    dtype=([
        'float32',
        'float32'
    ])
)
hdr4 = fits.Header()
hdr4['cont_loc'] = 'The wavelength of continuum luminosity in results'
hdr4['Fe_flux_range'] = 'Fe emission wavelength range calculated in results'

hdu4 = fits.BinTableHDU(data=measure_info, header=hdr4, name='measure_info')

hdu_list = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3, hdu4])
hdu_list.writeto(os.path.join(path_ex, 'qsopar.fits'), overwrite=True)

# %%
from astroquery.sdss import SDSS
from astropy.table import Table
import pyqsofit


import os
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
import astropy.units as u

not_cached = []

def load_spec(sdss_name, plate, fiber, mjd, cache_dir="data/spectra_cache"):
    
    # sdss_name = data_cat['SDSS_NAME'][i]
    # plate, fiber, mjd = data_cat['PLATE'][i], data_cat['FIBERID'][i], data_cat['MJD'][i]
    #cache_file = os.path.join(cache_dir, f"{sdss_name}_p{plate}_f{fiber}_m{mjd}.fits")
    cache_file = os.path.join(cache_dir, f"{sdss_name}.fits")
    
    # Return cached FITS if available
    if os.path.exists(cache_file):
        from astropy.io import fits
        print(f"Loaded cached spectrum for {sdss_name} from {cache_file}")
        return fits.open(cache_file, memmap=False)
    else:
        print("WARNING: No cached spectrum found for", sdss_name)
        not_cached.append(sdss_name)

def get_alpha(i):

    # Get the SDSS_NAME for the current index i
    sdss_name = data_cat['SDSS_NAME'][i]
    print(f"Processing SDSS_NAME: {sdss_name}")
    plate = data_cat['PLATE'][i]
    fiber = data_cat['FIBERID'][i]
    mjd = data_cat['MJD'][i]
    z = data_cat['Z_SYS'][i]
    loglbol = data_cat['LOGLBOL'][i]

    # Query SDSS for the spectrum using the SDSS_NAME
    #try:
    if True:
        # Use the new caching function
        data = load_spec(sdss_name, plate, fiber, mjd)
        if data is None:
            print(f"Data is none for {sdss_name}. Skipping...")
            return None
        # Optional: compute the SkyCoord if you still need the coordinates
        coord = SkyCoord(
            ra=data_cat['RA'][i] * u.deg,
            dec=data_cat['DEC'][i] * u.deg,
            frame='icrs'
        )
        # Extract the spectrum data from the FITS file
        # Requried
        lam = 10 ** data[1].data['loglam']  # OBS wavelength [A]
        flux = data[1].data['flux']  # OBS flux [erg/s/cm^2/A]
        #print(np.mean(flux), np.std(flux))
        err = 1 / np.sqrt(data[1].data['ivar'])  # 1 sigma error
        z = data_cat['Z_SYS'][i]  # Redshift

        # if np.min(lam) > 3619.0:
        #     print("NO g band!")
        #     return -1, -1, -1


        # Absolute flux calibration
        bands = ['g', 'r', 'i']
        sdss_filters = filters.load_filters(*[f'sdss2010-{b}' for b in bands])

        # Extract the 'mean' arrays from each object in objs

        # Step 3: Compute Δm per band
        delta_mags = []
        weights = []

        for b, filt in zip(bands, sdss_filters):
            try:
                mag_fiber = data_cat[f'mean_corrected_{b}'][i]
                #print(mag_fiber)
                if not np.isfinite(mag_fiber) or mag_fiber < 0:
                    continue

                # Synthetic magnitude
                mag_synth = filt.get_ab_magnitude(1e-17*flux*u.erg/u.s/u.cm**2/u.AA, lam*u.AA)
                #print(mag_synth, mag_fiber)
                delta_m = mag_fiber - mag_synth
                delta_mags.append(delta_m)
                # TODO: Add mags_mean_err to weights
                # check how weights are defined in np.average
                weights.append(1.0)  # could add mag error weights if available
            except Exception as e:
                print(f"Error processing band {b} for SDSS_NAME {sdss_name}: {e}")
                continue

        if len(delta_mags) == 0:
            delta_mags = 0.0
            weights = 1.0
            #raise ValueError("No valid photometry–spectrum comparisons found.")

        delta_mags = np.array(delta_mags)
        weights = np.array(weights)
        mask = np.isfinite(delta_mags)
        if np.any(mask):
            delta_m_avg = np.average(delta_mags[mask], weights=weights[mask])
        else:
            delta_m_avg = 0

        scale_factor = 10 ** (-0.4 * delta_m_avg)
        flux_scaled = flux * scale_factor

        print("scale_factor", scale_factor)
        if scale_factor < 0.5 or scale_factor > 1.5:
            print(f"WARNING: scale_factor={scale_factor:.3f} is outside the expected range [0.5, 1.5]")


        # Prepare data
        q_mle = QSOFit(lam, flux, err, z, path=path_ex)

        # Double check the installation path with the PCA / Fe template files
        # print('install path:', q_mle.install_path)

        # Change it if you installed them somewhere else
        #q_mle.install_path = '...'

        start = timeit.default_timer()
        # Do the fitting
        decompose_host = True if loglbol < 46 else False

        q_mle.Fit(name=f"{z:.2f}_{sdss_name}_{plate}-{mjd}-{fiber}",  # customize the name of given targets. Default: plate-mjd-fiber
                # prepocessing parameters
                nsmooth=1,  # do n-pixel smoothing to the raw input flux and err spectra
                and_mask=False,  # delete the and masked pixels
                or_mask=False,  # delete the or masked pixels
                reject_badpix=False,  # reject 10 most possible outliers by the test of pointDistGESD
                deredden=True,  # correct the Galactic extinction
                wave_range=[1150, 1e9],  # trim input wavelength
                wave_mask=None,  # 2-D array, mask the given range(s)

                # host decomposition parameters
                decompose_host=decompose_host,  # If True, the host galaxy-QSO decomposition will be applied
                host_prior=False, # If True, the code will adopt prior-informed method to assist decomposition. Currently, only 'CZBIN1' and 'DZBIN1' model for QSO PCA are available. And the model for galaxy must be PCA too.
                host_prior_scale=0.2, # scale of prior panelty. Usually, 0.2 works fine for SDSS spectra. Adjust it smaller if you find the prior affect the fitting results too much.

                host_line_mask=True, # If True, the line region of galaxy will be masked when subtracted from original spectra.
                decomp_na_mask=True, # If True, the narrow line region will be masked when perform decomposition
                qso_type='global', # PCA template name for quasar
                npca_qso=10, # numebr of quasar templates
                host_type='BC03', # template name for galaxy
                npca_gal=5, # number of galaxy templates
                
                # continuum model fit parameters
                Fe_uv_op=True,  # If True, fit continuum with UV and optical FeII template
                poly=True,  # If True, fit continuum with the polynomial component to account for the dust reddening
                BC=False,  # If True, fit continuum with Balmer continua from 1000 to 3646A
                initial_guess=None,  # Initial parameters for continuum model, read the annotation of this function for detail
                rej_abs_conti=False,  # If True, it will iterately reject 3 sigma outlier absorption pixels in the continuum
                n_pix_min_conti=100,  # Minimum number of negative pixels for host continuuum fit to be rejected.

                # emission line fit parameters
                linefit=True,  # If True, the emission line will be fitted
                rej_abs_line=False,
                # If True, it will iterately reject 3 sigma outlier absorption pixels in the emission lines

                # fitting method selection
                MC=True,
                # If True, do Monte Carlo resampling of the spectrum based on the input error array to produce the MC error array
                MCMC=False,
                # If True, do Markov Chain Monte Carlo sampling of the posterior probability densities to produce the error array
                nsamp=20,
                # The number of trials of the MC process (if MC=True) or number samples to run MCMC chain (if MCMC=True)

                # advanced fitting parameters
                param_file_name='qsopar.fits',  # Name of the qso fitting parameter FITS file.
                nburn=20,  # The number of burn-in samples to run MCMC chain
                nthin=10,  # To set the MCMC chain returns every n samples
                epsilon_jitter=0.,
                # Initial jitter for every initial guass to avoid local minimum. (Under test, not recommanded to change)

                # customize the results
                save_result=True,  # If True, all the fitting results will be saved to a fits file
                save_fits_name=None,  # The output name of the result fits
                save_fits_path='.',  # The output path of the result fits
                plot_fig=True,  # If True, the fitting results will be plotted
                save_fig=True,  # If True, the figure will be saved
                plot_corner=True,  # Whether or not to plot the corner plot results if MCMC=True

                # debugging mode
                verbose=False,  # turn on (True) or off (False) debugging output

                # sublevel parameters for figure plot and emcee
                kwargs_plot={
                    'save_fig_path': './plots/pyqso',  # The output path of the figure
                    'broad_fwhm'   : 1200  # km/s, lower limit that code decide if a line component belongs to broad component
                },
                kwargs_conti_emcee={},
                kwargs_line_emcee={}
                )

    #except:
    #    return -99, -99, -99

    end = timeit.default_timer()

    print(f'Fitting finished in {np.round(end - start, 1)}s')

    # Build a dictionary from q_mle.conti_result_name and q_mle.conti_result
    conti_dict = {name: val for name, val in zip(q_mle.conti_result_name, q_mle.conti_result)}
    # Convert string values to float if needed
    for k, v in conti_dict.items():
        if isinstance(v, str):
            try:
                conti_dict[k] = float(v)
            except Exception:
                conti_dict[k] = np.nan

    conti_dict['z'] = z

    # Get m_i conti rest (host-subtracted / QSO only)
    filt = filters.load_filters('sdss2010-i')
    apparent_mag_i_rest = filt.get_ab_magnitude(1e-17*q_mle.qso*u.erg/u.s/u.cm**2/u.AA, q_mle.wave*u.AA)

    # Compute m_2500_new using the dictionary
    m_2500, m_2500_err = compute_apparent_mag_2500_astropy(conti_dict)

    result = {
        'apparent_mag_2500': m_2500,
        'apparent_mag_2500_err': m_2500_err,
        'apparent_mag_i_rest': apparent_mag_i_rest
        'f_host_4200': conti_dict.get('frac_host_4200', -99),
        'alpha_lambda': conti_dict.get('PL_slope', -99),
        'alpha_lambda_err': conti_dict.get('PL_slope_err', -99),
        'redchi': q_mle.conti_fit.redchi
    }
    #plt.show()
    return result

from astropy.cosmology import FlatLambdaCDM
from tqdm import trange
import astropy.units as u

def compute_apparent_mag_2500_astropy(conti_table, logL_col='L2500', logL_err_col='L2500_err', z_col='z', H0=70, Om0=0.3):
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

ntest = len(data_cat)
#ntest = 500

new_data = []

results = []
for i in trange(ntest, desc="Processing objects"):
    object_id = data_cat['object_id'][i]
    sdss_name = data_cat['SDSS_NAME'][i]
    # Initialize result with default -99 values
    result = dict(
        object_id=object_id,
        sdss_name=sdss_name,
        apparent_mag_2500=-1e9,
        apparent_mag_2500_err=-1e9,
        f_host_4200=-1e9,
        alpha_lambda=-1e9,
        alpha_lambda_err=-1e9,
        redchi=-1e9,
    )

    try:
        r = get_alpha(i)
        print(r)
        if r:
            result |= r
        else:
            print(f"Error processing object {object_id} ({sdss_name})")
    except Exception as e:
        print(f"Exception processing object {object_id} ({sdss_name}): {e}")

    results.append(result)


# Save results to CSV


results_df = pd.DataFrame(results)

#results_df.to_csv('data/csv/aug10_stone_merged_fittedm2500.csv', index=False)
results_df.to_csv('data/csv/aug11_sample_chisqg10_ebv005sn3_magsmean_fittedm2500.csv', index=False)

print("Not cached objects:", not_cached)
