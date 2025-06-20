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

def sigma_clip_in_bins(df, bin_width=0.1, sigma=2):
    df_clean = []
    z_min, z_max = df['z'].min(), df['z'].max()
    bins = np.arange(z_min, z_max + bin_width, bin_width)

    for i in range(len(bins) - 1):
        bin_df = df[(df['z'] >= bins[i]) & (df['z'] < bins[i+1])]
        if len(bin_df) < 5:
            continue
        y = bin_df['apparent_mag_i'] - bin_df['M_i']
        clipped, _, _ = sigmaclip(y, low=sigma, high=sigma)
        df_clean.append(bin_df[y.isin(clipped)])

    return pd.concat(df_clean)

def load_quasar_data(file_path):

    quasar_list = read_quasars_from_hdf5(file_path)
    print("Number of quasars loaded:", len(quasar_list))

    #if populate_sdss:
    for quasar in quasar_list:
        if 'apparent_mag_i' not in quasar.keys():
            print("Populating SDSS fields...")
            populate_sdss_fields(quasar_list)
            write_hdf5_file(quasar_list, file_path)
            break

    df = pd.DataFrame(quasar_list)
    df = df[
        #(df['z'] > 0.4) &
        (df['eta_A1_err'] > 0.1) &
        df['log_sigma_hat_UV'].between(-3, 0) &
        (df['log_sigma_hat_UV_err'] > 0) & (df['log_sigma_hat_UV_err'] < 0.5) &
        (df['log_tau_UV_RF'] > 2) &
        (df['apparent_mag_i'] < 26) &
        (df['apparent_mag_i_err'] < 0.5) &
        (df['M_i'] < -21) &
        (df['z'] > 0) &
        (df['log_lbol'] > 1) &
        (df['log_mbh'] > 1) &
        (df['ebv'] < 0.05)
    ].dropna()
        
    #df = df.set_index('object_id')
    #df = df.drop(columns=['log_jitter', 'magerrs_mean', 'log_sigma_band', 'log_sigma_band_err', 'clean_bands'])
    #df = df.drop(columns=['mags', 'times', 'magerrs'])


    
    # Remove infinite values from numeric columns
    columns_with_nans = df.columns[df.isna().any()].tolist()
    print("Columns with NaNs:", columns_with_nans)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].where(~np.isinf(df[numeric_cols]), np.nan)
    df = df.dropna()
    
    df = df.reset_index(drop=True)

    # sigma clip in bins can almost replace all the previous cuts!
    df = sigma_clip_in_bins(df)

    num_quasars_z_0_1 = len(df[(df['z'] > 0) & (df['z'] <= 1)])
    num_quasars_z_gt_3 = len(df[df['z'] > 3])
    print("Number of quasars with 0 < z <= 1:", num_quasars_z_0_1)
    print("Number of quasars with z > 3:", num_quasars_z_gt_3)
    print("Final number of quasars:", len(df))
    return df

def load_data(file_path):
    print("Loading quasar data...")
    #df_agn = load_quasar_data("data/may12_objs_tauwavelength_taublr_redbands_ds4_merged.h5")
    df_agn = load_quasar_data(file_path=file_path)
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


# class Completeness2D:
#     def __init__(self, mag_centers, z_centers, completeness_ratio):
#         self.mag_centers = mag_centers
#         self.z_centers = z_centers
#         self.interp_fn = RegularGridInterpolator(
#             (mag_centers, z_centers),
#             completeness_ratio,
#             bounds_error=False, fill_value=0.0
#         )

#     def __call__(self, mag, z):
#         pts = np.column_stack([np.ravel(mag), np.ravel(z)])
#         vals = self.interp_fn(pts)
#         return vals.reshape(np.shape(mag))
    
class Completeness2D:
    def __init__(self, mag_centers, z_centers, completeness_map):
        self.mag_centers = mag_centers
        self.z_centers = z_centers

        # Clip NaNs and store minimum finite completeness
        completeness_map_clean = np.nan_to_num(completeness_map, nan=0.0)
        self.min_completeness_value = float(np.nanmin(completeness_map_clean))

        self.mag_min = mag_centers[0]
        self.mag_max = mag_centers[-1]
        self.z_min = z_centers[0]
        self.z_max = z_centers[-1]

        self.interp_fn = RegularGridInterpolator(
            (mag_centers, z_centers),
            completeness_map_clean,
            bounds_error=False,
            fill_value=0.0
        )

    def __call__(self, mag, z):
        mag = np.asarray(mag)
        z = np.asarray(z)
        mag_b, z_b = np.broadcast_arrays(mag, z)

        mag_clipped = np.clip(mag_b, self.mag_min, self.mag_max)
        z_clipped = np.clip(z_b, self.z_min, self.z_max)

        pts = np.column_stack([mag_clipped.ravel(), z_clipped.ravel()])
        vals = self.interp_fn(pts)
        return vals.reshape(mag_b.shape)

    def get_completeness_map(self):
        return self.interp_fn.values


def get_completeness_function_2d(df_agn,
                                 sim_file="sampled_apparent_magnitudes_redshift_vol.h5",
                                 n_mag_bins=20, n_z_bins=30,
                                 mag_min=15, mag_max=24,
                                 sigma_mag=1.0, sigma_z=0.7,
                                 normalize=True,
                                 plot=False):
    # --- Load simulated (true) sample
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

    # --- Load observed sample
    mags_obs = df_agn['apparent_mag_i'].values
    z_obs = df_agn['z'].values

    # --- Clean NaNs/Infs
    mask_true = np.isfinite(mags_true) & np.isfinite(z_true)
    mags_true = mags_true[mask_true]
    z_true = z_true[mask_true]

    mask_obs = np.isfinite(mags_obs) & np.isfinite(z_obs)
    mags_obs = mags_obs[mask_obs]
    z_obs = z_obs[mask_obs]

    # --- Bin edges and centers
    z_min, z_max = np.min(z_true), np.max(z_true)
    if z_max - z_min < 1e-3:
        z_min -= 0.01
        z_max += 0.01

    mag_edges = np.linspace(mag_min, mag_max, n_mag_bins + 1)
    z_edges = np.linspace(z_min, z_max, n_z_bins + 1)
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    # --- Histogram both samples
    hist_true, _, _ = np.histogram2d(mags_true, z_true, bins=[mag_edges, z_edges])
    hist_obs, _, _ = np.histogram2d(mags_obs, z_obs, bins=[mag_edges, z_edges])

    # --- Compute completeness ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        completeness = np.zeros_like(hist_true, dtype=float)
        valid = hist_true > 0
        completeness[valid] = hist_obs[valid] / hist_true[valid]

    # --- Optional smoothing
    completeness_smoothed = gaussian_filter(completeness, sigma=(sigma_mag, sigma_z), mode='nearest')

    # --- Clip to [0, 1] and normalize if requested
    completeness_smoothed = np.clip(completeness_smoothed, 0.0, 1.0)
    if normalize and np.nanmax(completeness_smoothed) > 0:
        completeness_smoothed /= np.nanmax(completeness_smoothed)

    # --- Compute bin widths for completeness convolution
    dm = mag_centers[1] - mag_centers[0]
    dz = z_centers[1] - z_centers[0]

    # --- Optional diagnostic plot
    if plot:
        import matplotlib.pyplot as plt
        plt.imshow(completeness_smoothed.T, origin='lower', aspect='auto',
                   extent=[mag_edges[0], mag_edges[-1], z_edges[0], z_edges[-1]])
        plt.xlabel('Apparent Magnitude')
        plt.ylabel('Redshift')
        plt.title('Completeness Map (Smoothed)')
        plt.colorbar(label='p(detect | m, z)')
        plt.tight_layout()
        plt.show()

    return Completeness2D(mag_centers, z_centers, completeness_smoothed), mag_centers, z_centers, dm, dz



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

def display_results_summary(samples, cosmo_model):
    _, model_labels, _ = get_model_params(cosmo_model)
    median_samples = np.median(samples, axis=0)
    lowers = np.percentile(samples, 16, axis=0)
    uppers = np.percentile(samples, 84, axis=0)
    
    for name, med, lo, hi in zip(model_labels, median_samples, lowers, uppers):
        print(f"{name:>15}: {med:.4f} (+{hi - med:.4f}, -{med - lo:.4f})")

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
        Must contain 'apparent_mag_i', 'apparent_mag_i_err', 'z'.
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

    m_obs = df_agn['apparent_mag_i'].values.astype(np.float64)
    m_err = df_agn['apparent_mag_i_err'].values.astype(np.float64)
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
        cosmo = Flatw0waCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'], wa=params['wa'])
    elif cosmo_model == 'FlatwCDM':
        cosmo = FlatwCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'])
    elif cosmo_model == 'FlatLambdaCDM':
        cosmo = FlatLambdaCDM(H0=params['H0'], Om0=params['Om0'])
    else:
        raise ValueError("cosmo_model must be 'FlatwCDM', 'Flatw0waCDM' or 'FlatLambdaCDM'")

    # Compute model-predicted absolute magnitude
    M_model = M_model_agn(
        params['M0_sn'] + params['delta_M0_agn'],
        params['log_sigma_hat_sq_break'],
        params['eta_A1_agn'], params['eta_A2_agn'],
        params['eta_break_agn'],
        params['beta_agn'],
        df_agn['log_sigma_hat_UV'].values,
        df_agn['log_tau_UV_RF'].values
    )

    # Cosmological distance modulus + K-correction
    mu_cosmo = cosmo.distmod(df_agn['z'].values).value
    m_model = mu_cosmo + (K_corr(df_agn['z'].values) - K_corr(2)) + M_model

    # Total uncertainty on m_model
    mu_err = np.sqrt(
        df_agn['apparent_mag_i_err'].values**2 +
        M_model_agn_err(
            params['M0_sn'] + params['delta_M0_agn'],
            params['log_sigma_hat_sq_break'],
            params['eta_A1_agn'], params['eta_A2_agn'],
            params['eta_break_agn'],
            params['beta_agn'],
            df_agn['log_sigma_hat_UV'].values,
            df_agn['log_sigma_hat_UV_err'].values,
            df_agn['log_tau_UV_RF'].values
        )**2 +
        (2.5 * 0.3 * np.log10(1 + df_agn['z'].values))**2 +
        (0.055 * df_agn['z'].values)**2 +
        np.exp(2 * params['log_f'])
    )

    # Apply marginalization over completeness
    m_corr = predict_uncensored_magnitudes(df_agn, m_model, mu_err, completeness_params)
    return m_corr

