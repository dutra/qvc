import matplotlib.pyplot as plt

import numpy as np
import h5py
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
from scipy.stats import norm
from scipy.interpolate import RegularGridInterpolator

bands = ['u', 'g', 'r', 'i', 'z']#, 'y']
bands_idx = {b: i for i, b in enumerate(bands)}

def populate_sdss_fields(d, fits_data, fits_data_2):
    i = np.argwhere(d['sdss_name'] == fits_data['SDSS_NAME']).flatten()
    if len(i) == 0:
        print(f"Warning: {d['sdss_name']} not found in SDSS data")
        return d
    i = i[0]
    if np.any(fits_data_2['PSFFLUX'][i,:] <= 0):
        return d
    d['log_mbh'] = fits_data['LOGMBH'][i]
    d['log_mbh_err'] = fits_data['LOGMBH_ERR'][i]
    d['log_ledd_ratio'] = fits_data['LOGLEDD_RATIO'][i]
    d['log_ledd_ratio_err'] = fits_data['LOGLEDD_RATIO_ERR'][i]
    d['ebv'] = fits_data['EBV'][i]
    d['M_i'] = fits_data_2['M_I'][i]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d['log_lbol'] = fits_data['LOGLBOL'][i]
        d['log_lbol_err'] = fits_data['LOGLBOL_ERR'][i]
        d['apparent_mag_z'] = -2.5 * np.log10(fits_data_2['PSFFLUX'][i,4]) + 22.5
        d['apparent_mag_i'] = -2.5 * np.log10(fits_data_2['PSFFLUX'][i,3]) + 22.5
        d['apparent_mag_r'] = -2.5 * np.log10(fits_data_2['PSFFLUX'][i,2]) + 22.5
        d['apparent_mag_g'] = -2.5 * np.log10(fits_data_2['PSFFLUX'][i,1]) + 22.5
        d['apparent_mag_u'] = -2.5 * np.log10(fits_data_2['PSFFLUX'][i,0]) + 22.5
        d['apparent_mag_z_err'] = 2.5/np.log(10) * np.sqrt(1/fits_data_2['PSFFLUX_IVAR'][i,4])/fits_data_2['PSFFLUX'][i,4]
        d['apparent_mag_i_err'] = 2.5/np.log(10) * np.sqrt(1/fits_data_2['PSFFLUX_IVAR'][i,3])/fits_data_2['PSFFLUX'][i,3]
        d['apparent_mag_r_err'] = 2.5/np.log(10) * np.sqrt(1/fits_data_2['PSFFLUX_IVAR'][i,2])/fits_data_2['PSFFLUX'][i,2]
        d['apparent_mag_g_err'] = 2.5/np.log(10) * np.sqrt(1/fits_data_2['PSFFLUX_IVAR'][i,1])/fits_data_2['PSFFLUX'][i,1]
        d['apparent_mag_u_err'] = 2.5/np.log(10) * np.sqrt(1/fits_data_2['PSFFLUX_IVAR'][i,0])/fits_data_2['PSFFLUX'][i,0]
        d['color'] = -2.5 * np.log10(fits_data_2['PSFFLUX'][i, 0] / fits_data_2['PSFFLUX'][i, 3])
    if any(issubclass(warning.category, RuntimeWarning) for warning in w):
        print(f"RuntimeWarning occurred for {d['sdss_name']}")
        print(fits_data_2['PSFFLUX'][i,:])
        print(w)
    return d


def read_quasars_from_hdf5(file_path):
    quasar_list = []

    hdul = fits.open('data/dr16q_prop_May01_2024.fits')
    fits_data = hdul[1].data  # Assuming the data is in the first extension    
    fits_data_2 = hdul[2].data  # Assuming the data is in the first extension    


    with h5py.File(file_path, "r") as hdf:
        for group_name in tqdm(hdf.keys(), desc="Reading quasars from HDF5"):
            group = hdf[group_name]
            quasar = {"object_id": group_name}
            for key, value in group.attrs.items():
                quasar[key] = value
            for sub_group_name in group.keys():
                sub_group = group[sub_group_name]
                quasar[sub_group_name] = {sub_key: sub_group[sub_key][...] for sub_key in sub_group.keys()}
            populate_sdss_fields(quasar, fits_data, fits_data_2)
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

def load_quasar_data(file_path):

    quasar_list = read_quasars_from_hdf5(file_path)
    print("Number of quasars loaded:", len(quasar_list))


    df = pd.DataFrame(quasar_list)
    #df = df.set_index('object_id')
    #df = df.drop(columns=['log_jitter', 'magerrs_mean', 'log_sigma_band', 'log_sigma_band_err', 'clean_bands'])
    df = df.drop(columns=['mags', 'times', 'magerrs'])

    df_all = df.copy()

    # data cuts
    df = df[df['log_sigma_UV'] < 0]
    df = df[df['log_tau_UV_RF'] > 1]
    # df = df[df['log_lbol'] > 44]
    # df = df[df['log_mbh'] > 1]
    # df = df[df['apparent_mag_i'] < 30]
    # #df = df[df['M_i'] > -27]
    df = df[df['ebv'] < 0.05]
    #df = df[df['apparent_mag_i'].between(18, 20)]
    #df = df[df['z'] < 1.2]
    #df = filter_unresolved_quasars(df)
    
    # Remove infinite values from numeric columns
    columns_with_nans = df.columns[df.isna().any()].tolist()
    print("Columns with NaNs:", columns_with_nans)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].where(~np.isinf(df[numeric_cols]), np.nan)
    df = df.dropna()
    
    num_quasars_z_0_1 = len(df[(df['z'] > 0) & (df['z'] <= 1)])
    num_quasars_z_gt_3 = len(df[df['z'] > 3])
    
    print("Number of quasars with 0 < z <= 1:", num_quasars_z_0_1)
    print("Number of quasars with z > 3:", num_quasars_z_gt_3)
    print("Final number of quasars:", len(df))
    return df

def get_completeness_function_simple(mag_lim, center=20):
    """
    Simpler completeness function based on a normal CDF.
    Returns p(I=1|m) = 1 - normal.cdf(m, center=mag_lim).
    """
    raise NotImplementedError("This function is not implemented yet.")
    def completeness(m):
        return 1 - norm.cdf(m, loc=center, scale=mag_lim)

    mag_eval = np.linspace(14, 26, 500)
    dm = mag_eval[1] - mag_eval[0]
    p_detect = completeness(mag_eval)

    plt.plot(mag_eval, p_detect, label=f"Center={center}, Mag Lim={mag_lim}")
    plt.xlabel("i-band magnitude")
    plt.ylabel("Completeness")
    plt.title("Simplified completeness function")
    plt.legend()
    plt.savefig("plots/completeness_function_simple.png", dpi=200)
    #plt.show()
    plt.close()
    return p_detect, mag_eval, dm

# Example usage
# p_detect, mag_eval, dm = get_completeness_function_simple(mag_lim=1.0, center=20)

def get_completeness_function(df_agn):
    n_bins_completeness = 26
    file_path = "stacked_sampled_apparent_magnitudes.h5"

    with h5py.File(file_path, "r") as f:
        mags_true = f["stacked_apparent_magnitudes"][:]

    mags_obs = df_agn['apparent_mag_i'].values

    # Clean both
    mags_true = mags_true[np.isfinite(mags_true)]
    mags_obs = mags_obs[np.isfinite(mags_obs)]

    mag_min, mag_max = 14, 26
    bin_edges = np.linspace(mag_min, mag_max, n_bins_completeness)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    hist_true, _ = np.histogram(mags_true, bins=bin_edges)
    hist_obs, _ = np.histogram(mags_obs, bins=bin_edges)

    mask = hist_true > 0
    completeness_ratio = np.zeros_like(hist_true, dtype=float)
    completeness_ratio[mask] = hist_obs[mask] / hist_true[mask]
    completeness_ratio = gaussian_filter1d(completeness_ratio, sigma=0.5)
    completeness_ratio = np.clip(completeness_ratio, 0, 1)

    # Use fixed eval grid for later convolution
    mag_eval = np.linspace(mag_min, mag_max, 500)
    dm = mag_eval[1] - mag_eval[0]
    interp_fn = interp1d(
        bin_centers, completeness_ratio,
        kind='quadratic', bounds_error=False, fill_value=(1.0, 0.0)
    )
    p_detect = interp_fn(mag_eval)
    plt.scatter(mag_eval, p_detect)
    plt.xlabel("i-band magnitude"); plt.ylabel("Completeness")
    plt.title("Empirical completeness function")
    plt.savefig("plots/completeness_function.png", dpi=200)
    #plt.show()
    plt.close()
    
    return p_detect, mag_eval, dm

class Completeness2D:
    def __init__(self, mag_centers, z_centers, completeness_ratio):
        self.mag_centers = mag_centers
        self.z_centers = z_centers
        self.interp_fn = RegularGridInterpolator(
            (mag_centers, z_centers),
            completeness_ratio,
            bounds_error=False, fill_value=0.0
        )

    def __call__(self, mag, z):
        pts = np.column_stack([np.ravel(mag), np.ravel(z)])
        vals = self.interp_fn(pts)
        return vals.reshape(np.shape(mag))
    
def get_completeness_function_2d(df_agn, sim_file="sampled_apparent_magnitudes_by_redshift.h5",
                                 n_mag_bins=26, n_z_bins=16, mag_min=14, mag_max=26):
    """
    Returns a completeness function p_detect(mag, z) as a callable.
    Uses observed AGN sample and simulated sample from HDF5 file
    with 'redshift_bins' group (one dataset per redshift bin).
    """
    import numpy as np
    import h5py
    from scipy.ndimage import gaussian_filter
    from scipy.interpolate import RegularGridInterpolator

    # Load simulated (true) sample: flatten all bins into arrays
    mags_true_list = []
    z_true_list = []
    with h5py.File(sim_file, "r") as f:
        grp = f["redshift_bins"]
        for ds_name in grp:
            ds = grp[ds_name]
            mags = ds[()]
            z_bin = ds.attrs["redshift"]
            mags_true_list.append(mags)
            z_true_list.append(np.full_like(mags, z_bin, dtype=float))
    mags_true = np.concatenate(mags_true_list)
    z_true = np.concatenate(z_true_list)

    # Observed sample
    mags_obs = df_agn['apparent_mag_i'].values
    z_obs = df_agn['z'].values

    # Clean both
    mask_true = np.isfinite(mags_true) & np.isfinite(z_true)
    mags_true = mags_true[mask_true]
    z_true = z_true[mask_true]

    mask_obs = np.isfinite(mags_obs) & np.isfinite(z_obs)
    mags_obs = mags_obs[mask_obs]
    z_obs = z_obs[mask_obs]

    # Bin edges and centers
    mag_edges = np.linspace(mag_min, mag_max, n_mag_bins)
    z_edges = np.linspace(np.min(z_obs), np.max(z_obs), n_z_bins)
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    # 2D histograms
    hist_true, _, _ = np.histogram2d(mags_true, z_true, bins=[mag_edges, z_edges])
    hist_obs, _, _ = np.histogram2d(mags_obs, z_obs, bins=[mag_edges, z_edges])

    # Compute completeness ratio
    mask = hist_true > 0
    completeness_ratio = np.zeros_like(hist_true, dtype=float)
    completeness_ratio[mask] = hist_obs[mask] / hist_true[mask]
    completeness_ratio = gaussian_filter(completeness_ratio, sigma=0.7)
    completeness_ratio = np.clip(completeness_ratio, 0, 1)

    # Interpolator: returns completeness for any (mag, z)
    interp_fn = RegularGridInterpolator(
        (mag_centers, z_centers), completeness_ratio,  # (mag, z) order
        bounds_error=False, fill_value=0.0
    )

    # Compute grid spacing for mag and z
    dm = mag_centers[1] - mag_centers[0]
    dz = z_centers[1] - z_centers[0]

    return Completeness2D(mag_centers, z_centers, completeness_ratio), mag_centers, z_centers, dm, dz