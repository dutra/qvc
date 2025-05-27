import matplotlib.pyplot as plt
import os
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
from scipy.stats import norm

bands = ['u', 'g', 'r', 'i', 'z']#, 'y']
bands_idx = {b: i for i, b in enumerate(bands)}

def sn_completeness_function(m_b, z, mlim=24.1, sigma=0.5):
    """
    Return probability of SN detection given apparent magnitude m_b and redshift z.
    mlim: effective magnitude limit for SN survey.
    sigma: sharpness of detection efficiency curve.
    """
    return 1.0 - norm.cdf(m_b, loc=mlim, scale=sigma)

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
        d['z'] = obj['Z_SYS']
        d['sdss_name'] = fits_data['SDSS_NAME'][i]  # Extract SDSS_NAME
        d['log_lbol'] = -999.0
        if d['z'] < 0.7:
            d['log_lbol'] = np.log10(5.15) + fits_data['LOGL3000'][i]
            d['log_lbol_err'] = fits_data['LOGL3000_ERR'][i]
        else:
            d['log_lbol'] = fits_data['LOGLBOL'][i]  # Extract log Lbol values
            d["log_lbol_err"] = fits_data['LOGLBOL_ERR'][i]  # Extract log Lbol error values
        d['log_mbh'] = fits_data['LOGMBH'][i]  # Extract log MBH values
        d['log_mbh_err'] = fits_data['LOGMBH_ERR'][i]  # Extract log MBH error values
        d['log_ledd_ratio'] = fits_data['LOGLEDD_RATIO'][i]  # Extract log L/edd values
        d['log_ledd_ratio_err'] = fits_data['LOGLEDD_RATIO_ERR'][i]  # Extract log L/edd error values
        d['ebv'] = fits_data['EBV'][i]
        d['M_i'] = fits_data_2['M_I'][i]
        d['sn_median_all'] = fits_data['SN_MEDIAN_ALL'][i]
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

    return objs



def read_quasars_from_hdf5(file_path):
    quasar_list = []

    with h5py.File(file_path, "r") as hdf:
        for group_name in tqdm(hdf.keys(), desc="Reading quasars from HDF5"):
            group = hdf[group_name]
            quasar = {"object_id": group_name}
            for key, value in group.attrs.items():
                quasar[key] = value
            for sub_group_name in group.keys():
                sub_group = group[sub_group_name]
                quasar[sub_group_name] = {sub_key: sub_group[sub_key][...] for sub_key in sub_group.keys()}
            quasar_list.append(quasar)
                #populate_sdss_fields(quasar, fits_data, fits_data_2)
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

def load_quasar_data(file_path, populate_sdss=False):

    quasar_list = read_quasars_from_hdf5(file_path)
    print("Number of quasars loaded:", len(quasar_list))
    if populate_sdss:
        print("Populating SDSS fields...")
        populate_sdss_fields(quasar_list)
        write_hdf5_file(quasar_list, file_path)


    df = pd.DataFrame(quasar_list)
    #df = df.set_index('object_id')
    #df = df.drop(columns=['log_jitter', 'magerrs_mean', 'log_sigma_band', 'log_sigma_band_err', 'clean_bands'])
    #df = df.drop(columns=['mags', 'times', 'magerrs'])

    # data cuts

    df = df[
        df['log_sigma_hat_UV'].between(-3, 0) &
        (df['log_sigma_hat_UV_err'] > 0) & (df['log_sigma_hat_UV_err'] < 0.5) &
        (df['apparent_mag_i'] < 26) &
        (df['apparent_mag_i_err'] < 0.5) &
        (df['M_i'] < -21) &
        (df['z'] > 0) &
        (df['log_lbol'] > 1) &
        (df['log_mbh'] > 1) &
        (df['ebv'] < 0.05)
    ].dropna()
    
    # Remove infinite values from numeric columns
    columns_with_nans = df.columns[df.isna().any()].tolist()
    print("Columns with NaNs:", columns_with_nans)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].where(~np.isinf(df[numeric_cols]), np.nan)
    #df = df.dropna()
    
    df = df.reset_index(drop=True)
    num_quasars_z_0_1 = len(df[(df['z'] > 0) & (df['z'] <= 1)])
    num_quasars_z_gt_3 = len(df[df['z'] > 3])
    print("Number of quasars with 0 < z <= 1:", num_quasars_z_0_1)
    print("Number of quasars with z > 3:", num_quasars_z_gt_3)
    print("Final number of quasars:", len(df))
    return df

def load_data(file_path, populate_sdss=False):
    print("Loading quasar data...")
    #df_agn = load_quasar_data("data/may12_objs_tauwavelength_taublr_redbands_ds4_merged.h5")
    df_agn = load_quasar_data(file_path=file_path, populate_sdss=populate_sdss)
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

    # Invert covariance and pre-compute log-determinant for SN likelihood
    #global Cov_inv, logdetCov
    Cov_inv = np.linalg.inv(cov_matrix)
    sign, logdet = np.linalg.slogdet(cov_matrix)
    if sign <= 0:
        raise ValueError("Covariance matrix is not positive-definite!")
    logdetCov = logdet
    L = np.linalg.cholesky(np.linalg.inv(Cov_inv))
    print("Data loaded. Running joint cosmographic fits...")
    return df_agn, df_pantheon, Cov_inv, logdetCov, L


def completeness(m, center, mag_lim):
    print("m shape:", np.shape(m))
    print("center shape:", np.shape(center))
    print("mag_lim shape:", np.shape(mag_lim))
    return 1 - norm.cdf(m, loc=center, scale=mag_lim)
def get_completeness_function_simple(mag_lim, center=20):
    """
    Simpler completeness function based on a normal CDF.
    Returns p(I=1|m) = 1 - normal.cdf(m, center=mag_lim).
    """
    #raise NotImplementedError("This function is not implemented yet.")


    mag_eval = np.linspace(14, 26, 500)
    dm = mag_eval[1] - mag_eval[0]
    p_detect = completeness(mag_eval, center, mag_lim)

    plt.plot(mag_eval, p_detect, label=f"Center={center}, Mag Lim={mag_lim}")
    plt.xlabel("i-band magnitude")
    plt.ylabel("Completeness")
    plt.title("Simplified completeness function")
    plt.legend()
    plt.savefig("plots/completeness_function_simple.png", dpi=200)
    #plt.show()
    plt.close()
    return p_detect, mag_eval, dm

def compute_delta_mag_bias(agn_magnitudes, mag_lim, center=20):
    p_detect, mag_grid, dm = get_completeness_function_simple(mag_lim, center)
    C_interp = interp1d(mag_grid, p_detect, kind='linear', bounds_error=False, fill_value=(p_detect[0], p_detect[-1]))

    delta_mags = []
    for m_i in agn_magnitudes:
        C_i = C_interp(m_i)
        if C_i <= 0:
            delta_mags.append(np.nan)  # Avoid divide-by-zero
            continue

        # Assume flat prior for true magnitude distribution over mag_grid
        weights = 1.0 - p_detect  # Non-detection weights
        mean_mag = np.sum(mag_grid * weights) * dm / (np.sum(weights) * dm)
        delta_mag = mean_mag - m_i  # Shift toward undetected fainter population
        delta_mags.append(delta_mag)

    return np.array(delta_mags)

# Example usage
# p_detect, mag_eval, dm = get_completeness_function_simple(mag_lim=1.0, center=20)

def get_completeness_function(df_agn):
    n_bins_completeness = 12
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
    
def get_completeness_function_2d(df_agn, sim_file="sampled_apparent_magnitudes_redshift_vol.h5",
                                 n_mag_bins=36, n_z_bins=36, mag_min=14, mag_max=26):
    """
    Returns a completeness function p_detect(mag, z) as a callable.
    Uses observed AGN sample and simulated sample from HDF5 file
    with 'redshift_bins' group (one dataset per redshift bin).
    """
    import numpy as np
    import h5py
    from scipy.ndimage import gaussian_filter
    from scipy.interpolate import RegularGridInterpolator

    # Load simulated (true) sample
    mags_true_list, z_true_list = [], []
    with h5py.File(sim_file, "r") as f:
        for name in f["redshift_bin"]:
            ds = f["redshift_bin"][name]
            mags = ds[()]
            z_bin = ds.attrs["redshift"]
            mags_true_list.append(mags)
            z_true_list.append(np.full_like(mags, z_bin, dtype=float))
    mags_true = np.concatenate(mags_true_list)
    z_true = np.concatenate(z_true_list)

    mags_obs = df_agn['apparent_mag_i'].values
    z_obs = df_agn['z'].values

    # Clean
    mask_true = np.isfinite(mags_true) & np.isfinite(z_true)
    mags_true = mags_true[mask_true]
    z_true = z_true[mask_true]

    mask_obs = np.isfinite(mags_obs) & np.isfinite(z_obs)
    mags_obs = mags_obs[mask_obs]
    z_obs = z_obs[mask_obs]

    # Bin edges and centers
    mag_edges = np.linspace(mag_min, mag_max, n_mag_bins + 1)
    z_edges = np.linspace(0.0, np.max(z_true)+0.5, n_z_bins + 1)
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    # 2D histograms
    hist_true, _, _ = np.histogram2d(mags_true, z_true, bins=[mag_edges, z_edges])
    hist_obs, _, _ = np.histogram2d(mags_obs, z_obs, bins=[mag_edges, z_edges])

    # Compute completeness ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        completeness_ratio = np.zeros_like(hist_true, dtype=float)
        mask = hist_true > 0
        completeness_ratio[mask] = hist_obs[mask] / hist_true[mask]

    # Apply gentle smoothing
    completeness_ratio = gaussian_filter(completeness_ratio, sigma=0.4)
    completeness_ratio = np.clip(completeness_ratio, 0.0, 1.0)

    # Interpolator
    interp_fn = RegularGridInterpolator(
        (mag_centers, z_centers), completeness_ratio,
        bounds_error=False, fill_value=0.0
    )

    # Compute grid spacing
    dm = mag_centers[1] - mag_centers[0]
    dz = z_centers[1] - z_centers[0]

    # Return as a wrapper class with callable behavior
    return Completeness2D(mag_centers, z_centers, completeness_ratio), mag_centers, z_centers, dm, dz


from scipy.interpolate import RegularGridInterpolator


def compute_delta_mag_bias_2d_zbins(df_agn, completeness2d, mag_centers, z_centers, dm, n_z_bins=12):
    """
    Estimate AGN magnitude bias corrections using completeness-corrected intrinsic magnitudes
    binned by redshift, and interpolate them across the full sample.

    Parameters:
    -----------
    df_agn : DataFrame
        Contains 'apparent_mag_i', 'z', and optionally 'apparent_mag_i_err'.
    completeness2d : callable
        Completeness function p_detect(m, z).
    mag_centers : array_like
        Centers of magnitude bins.
    z_centers : array_like
        Centers of redshift bins.
    dm : float
        Bin width in magnitude.
    n_z_bins : int
        Number of redshift bins for intrinsic mag estimation.

    Returns:
    --------
    delta_mags : ndarray
        Δm = ⟨m_intrinsic⟩ - m_observed for each AGN.
    delta_mag_errs : ndarray
        Propagated uncertainty combining measurement error and intrinsic scatter.
    """
    import numpy as np
    from scipy.interpolate import interp1d

    apparent_mags = df_agn['apparent_mag_i'].values
    z_vals = df_agn['z'].values
    mag_errs = df_agn['apparent_mag_i_err'].values if 'apparent_mag_i_err' in df_agn.columns else np.zeros_like(apparent_mags)

    z_bins = np.linspace(z_vals.min(), z_vals.max(), n_z_bins + 1)
    z_bin_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

    mean_mags = np.full(n_z_bins, np.nan)
    std_mags = np.full(n_z_bins, np.nan)

    for i in range(n_z_bins):
        z_low, z_high = z_bins[i], z_bins[i + 1]
        z_bin_mask = (z_centers >= z_low) & (z_centers < z_high)
        if not np.any(z_bin_mask):
            continue

        mag_grid, z_grid = np.meshgrid(mag_centers, z_centers[z_bin_mask], indexing='ij')
        completeness = completeness2d(mag_grid, z_grid)
        completeness = np.clip(completeness, 1e-3, 1.0)
        weights = 1.0 / completeness

        weighted_mags = mag_grid * weights
        weighted_sum = np.nansum(weighted_mags)
        total_weights = np.nansum(weights)

        if total_weights > 0:
            mean_mags[i] = weighted_sum / total_weights
            variance = np.nansum(weights * (mag_grid - mean_mags[i])**2) / total_weights
            std_mags[i] = np.sqrt(variance)

    # Interpolate over z to get delta_m for each object
    interp_mean = interp1d(z_bin_centers[~np.isnan(mean_mags)], mean_mags[~np.isnan(mean_mags)], kind='linear', fill_value='extrapolate')
    interp_std = interp1d(z_bin_centers[~np.isnan(std_mags)], std_mags[~np.isnan(std_mags)], kind='linear', fill_value='extrapolate')

    delta_mags = interp_mean(z_vals) - apparent_mags
    delta_mag_errs = np.sqrt(interp_std(z_vals)**2 + mag_errs**2)

    return delta_mags, delta_mag_errs


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
    import numpy as np
    from dynesty.utils import resample_equal
    from hubble_model import get_model_params

    priors, model_labels = get_model_params(cosmo_model)
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
