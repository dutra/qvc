import numpy as np
import pandas as pd
import h5py
from astropy.io import fits
from astropy.cosmology import FlatwCDM, Flatw0waCDM
from scipy import stats
from scipy.signal import fftconvolve
from scipy.stats import gaussian_kde
import emcee
import corner
import multiprocessing
from matplotlib.lines import Line2D
import warnings
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astroquery.vizier import Vizier
from astropy.coordinates import match_coordinates_sky
from astropy.coordinates import SkyCoord
from astropy.table import Table
from scipy.stats import norm
import astropy.units as u

import matplotlib.pyplot as plt

plt.style.use('style.mplstyle')


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

def load_quasar_data(file_path="s82_objs.h5"):

    quasar_list = read_quasars_from_hdf5("data/may12_objs_tauwavelength_taublr_redbands_ds4_merged.h5")
    #quasar_list = read_quasars_from_hdf5("data/may13_objs_tauwavelength_taublr_freebreak_newpriors4_merged.h5")
    #quasar_list = read_quasars_from_hdf5("data/may13_objs_tauwavelength_taublr_freebreak_newpriors_all_merged.h5")

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
    import numpy as np
    import h5py
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import interp1d

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

    return p_detect, mag_eval, dm


    plt.scatter(mag_eval, p_detect)
    plt.xlabel("i-band magnitude"); plt.ylabel("Completeness")
    plt.title("Empirical completeness function")
    plt.savefig("plots/completeness_function.png", dpi=200)
    #plt.show()
    plt.close()

    return p_detect, mag_eval, dm

# --- Constants ---
lambda_pivot = {
    'u': 3543,  # SDSS u-band
    'g': 4770,  # SDSS g-band
    'r': 6231,  # SDSS r-band
    'i': 7625,  # SDSS i-band
    'z': 9134,  # SDSS z-band
    'y': 9633,  # PS1 y-band
}

# z_pivot = df['z'].median()
sigma_pivot = -0.8
tau_pivot = 2.0

# --- Model Functions ---
def M_model_single(M0, alpha, log_sigma_UV, log_tau_UV_RF):
    """Variability-luminosity relation for one AGN."""
    #return M0 + alpha * (log_sigma_UV - sigma_pivot) + delta * (log_tau_UV_RF - tau_pivot)
    return M0 + alpha * 2*(log_sigma_UV - sigma_pivot) - (log_tau_UV_RF - tau_pivot)

def K_corr(z, alpha_nu=-0.5):
    """Simple K-correction."""
    return -2.5 * (1 + alpha_nu) * np.log10(1 + z)

# --- Priors ---
priors = {
    "alpha": (-10, 10),
    "M0": (-30, -10),
    "log_f": (-3, 1),
    "H0": (60, 80),
    "Om0": (0.2, 0.7),
    "w0": (-3, 0),
    "wa": (-3, 3),
}
labels = list(priors.keys())

def log_likelihood(theta, cosmo_model, model_labels, model_priors, df_agn, df_pantheon, completeness_params, only_sna=False):
    params = dict(zip(model_labels, theta))
  
    # Check prior bounds
    for key, (low, high) in model_priors.items():
        if not (low < params[key] < high):
            return -np.inf

    # Cosmology model selection
    if cosmo_model == 'FlatwCDM':
        cosmo = FlatwCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'])
    else:
        cosmo = Flatw0waCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'], wa=params['wa'])

    # AGN data
    z = df_agn['z'].values
    m_obs = df_agn['apparent_mag_i'].values
    m_err = df_agn['apparent_mag_i_err'].values
    log_sigma = df_agn['log_sigma_UV'].values
    log_tau = df_agn['log_tau_UV_RF'].values
    log_sigma_err = df_agn['log_sigma_UV_err'].values
    log_tau_err = df_agn['log_tau_UV_RF_err'].values

    mu_cosmo = cosmo.distmod(z).value
    Kcorr = K_corr(z) + K_corr(2)
    M_pred = M_model_single(params['M0'], params['alpha'], log_sigma, log_tau)
    mu_pred = m_obs - M_pred - Kcorr

    mu_err = np.sqrt(
        m_err**2 +
        (params['alpha'] * np.sqrt((2*log_sigma_err)**2+log_tau_err**2))**2 +
        (2.5 * 0.3 * np.log10(1 + z))**2 +
        #(K_corr(z) * 0.05)**2 +
        (0.055 * z)**2 +
        np.exp(2 * params['log_f'])
    )

    dmu = mu_pred - mu_cosmo
    ll_agn = np.sum(stats.norm.logpdf(dmu, scale=mu_err))

    # Selection correction
    if completeness_params:
        p_detect, mag_eval, dm = completeness_params
        m_model = M_pred + mu_cosmo

        integrals = np.zeros(len(df_agn))
        unique_errors = np.round(mu_err, 4)
        unique_vals = np.unique(unique_errors)

        # Precompute kernel grid
        x_kernel = mag_eval - np.median(mag_eval)

        for sigma in unique_vals:
            if sigma <= 0 or not np.isfinite(sigma):
                continue

            mask = np.abs(mu_err - sigma) < 1e-6
            if np.sum(mask) == 0:
                continue

            kernel = stats.norm.pdf(x_kernel, loc=0, scale=sigma)
            conv = fftconvolve(p_detect, kernel, mode="same") * dm
            # Avoid extrapolation issues
            conv = np.clip(conv, 1e-12, 1.0)

            # Interpolate safely
            integrals[mask] = np.interp(m_model[mask], mag_eval, conv, left=1e-12, right=1e-12)

        # Avoid NaNs in log
        integrals = np.clip(integrals, 1e-12, None)
        norm_correction = np.sum(np.log(integrals))
    # SNIa likelihood
    mu_snia = cosmo.distmod(df_pantheon['zHD']).value
    dmu_snia = df_pantheon['MU_SH0ES'] - mu_snia
    ll_snia = np.sum(stats.norm.logpdf(dmu_snia, scale=df_pantheon['MU_SH0ES_ERR_DIAG']))

    if only_sna:
        return ll_snia
    if completeness_params:
        return ll_snia + ll_agn - norm_correction
    else:
        return ll_snia + ll_agn


def run_mcmc_pipeline(df_agn, df_pantheon, cosmo_model='Flatw0waCDM', only_sna=False, completeness=True):
    if cosmo_model == 'FlatwCDM':
        cosmo_params = ['H0', 'Om0', 'w0']
    elif cosmo_model == 'Flatw0waCDM':
        cosmo_params = ['H0', 'Om0', 'w0', 'wa']
    else:
        raise ValueError("cosmo_model must be 'FlatwCDM' or 'Flatw0waCDM'")

    model_labels = ['alpha', 'M0', 'log_f'] + cosmo_params
    ndim = len(model_labels)

    # Reduced priors for selected cosmology
    model_priors = {key: priors[key] for key in model_labels}

    nwalkers = ndim * 2 * 15
    num_warmup, num_samples = 200, 500

    initial_pos = np.array([
        np.random.uniform(low, high, nwalkers) for low, high in model_priors.values()
    ]).T


    # passing all the data to the log_likelihood function makes it SLOW!
    df_pantheon_filtered = df_pantheon[['zHD', 'MU_SH0ES', 'MU_SH0ES_ERR_DIAG']]
    df_agn_filtered = df_agn[['z', 'apparent_mag_i', 'apparent_mag_i_err', 'log_sigma_UV', 'log_sigma_UV_err', 'log_tau_UV_RF', 'log_tau_UV_RF_err']]
    if completeness:
        completeness_params = get_completeness_function(df_agn_filtered)
    else:
        completeness_params = None
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_likelihood,
            args=(cosmo_model, model_labels, model_priors, df_agn_filtered, df_pantheon_filtered, completeness_params, only_sna), pool=pool,
        )
        state = sampler.run_mcmc(initial_pos, num_warmup, progress=True)

        sampler.reset()
        sampler.run_mcmc(state, num_samples, progress=True)
    return sampler, model_labels


def plot_corner(sampler, model_labels):
    flat_samples = sampler.get_chain(flat=True)
    fig = corner.corner(flat_samples, labels=model_labels, truths=None, show_titles=True, title_fmt=".2f")
    if only_sna:
        fig.suptitle("SNIa only", fontsize=16)
        plt.savefig(f"plots/posterior_{cosmo_model}_sna.png", dpi=200)
    else:
        fig.suptitle("SNIa + AGN", fontsize=16)
        plt.savefig(f"plots/posterior_{cosmo_model}_agn.png", dpi=200)
    #plt.show()
    plt.close()

def plot_cosmo_corner(sampler_sna, sampler_agn, cosmo_model='Flatw0waCDM'):
# === Parameter setup ===
    if cosmo_model == 'FlatwCDM':
        param_names = ["H0", "Om0", "w0"]
        labels = [r"$H_0$", r"$\Omega_M$", r"$w_0$"]
    elif cosmo_model == 'Flatw0waCDM':
        param_names = ["H0", "Om0", "w0", "wa"]
        labels = [r"$H_0$", r"$\Omega_M$", r"$w_0$", r"$w_a$"]
    n_params = len(labels)
    param_indices = [list(priors.keys()).index(p) for p in param_names]

    # === Get flattened MCMC chains ===
    sna_data = sampler_sna.get_chain(flat=True)[:, param_indices]
    agn_data = sampler_agn.get_chain(flat=True)[:, param_indices]

    # === Fast KDE-level calculator ===
    def get_density_levels(values, probs=[0.393, 0.865]):
        z = values.ravel()
        z_sorted = np.sort(z)
        cdf = np.cumsum(z_sorted)
        cdf /= cdf[-1]
        levels = [z_sorted[np.searchsorted(cdf, 1 - p)] for p in probs]
        return np.unique(np.sort(levels))  # ensure strictly increasing

    # === Fast 2D KDE plot with filled contours ===
    def fast_filled_kde(ax, x, y, color, base_alpha=0.4, levels=[0.393, 0.865]):
        data = np.vstack([x, y])
        kde = gaussian_kde(data)

        # Low-resolution grid
        xmin, xmax = np.percentile(x, [0.5, 99.5])
        ymin, ymax = np.percentile(y, [0.5, 99.5])
        xgrid = np.linspace(xmin, xmax, 100)
        ygrid = np.linspace(ymin, ymax, 100)
        xx, yy = np.meshgrid(xgrid, ygrid)
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

        contour_levels = get_density_levels(zz, levels)
        for i in range(len(contour_levels)-1, -1, -1):
            ax.contourf(xx, yy, zz,
                        levels=[contour_levels[i], zz.max()],
                        colors=[color],
                        alpha=base_alpha * (i + 1) / len(contour_levels))
        ax.contour(xx, yy, zz, levels=contour_levels, colors=[color], linewidths=1.2)

    # === Triangle plot ===
    fig, axes = plt.subplots(n_params, n_params, figsize=(10, 10))

    for i in tqdm(range(n_params), desc="Generating corner plot"):
        for j in range(n_params):
            ax = axes[i, j]
            ax.tick_params(direction='in')

            if i < j:
                ax.axis("off")
            elif i == j:
                for data, color in [(sna_data, "blue"), (agn_data, "red")]:
                    kde = gaussian_kde(data[:, i], bw_method=0.1)
                    xmin = data[:, i].min()
                    xmax = data[:, i].max()
                    margin = 0.1 * (xmax - xmin)
                    x_vals = np.linspace(xmin - margin, xmax + margin, 300)
                    ax.plot(x_vals, kde(x_vals), color=color, lw=1.8)
            else:
                fast_filled_kde(ax, sna_data[:, j], sna_data[:, i], "blue", base_alpha=0.4)
                fast_filled_kde(ax, agn_data[:, j], agn_data[:, i], "red", base_alpha=0.4)

            if j == 0:
                ax.set_ylabel(labels[i])
            else:
                ax.set_yticklabels([])

            if i == n_params - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])

    # === Legend ===
    legend_elements = [
        Line2D([0], [0], color="blue", lw=4, label="SN Ia"),
        Line2D([0], [0], color="red", lw=4, label="SN Ia + AGN"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=26, frameon=False, markerscale=1.5)

    # === Layout & Save ===
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
    fig.savefig(f"plots/corner_kde_{cosmo_model}.pdf", bbox_inches="tight", transparent=True)
    fig.savefig(f"plots/corner_kde_{cosmo_model}.png", bbox_inches="tight")
    #plt.show()
    plt.close()

def plot_hubble(sampler, df_agn, df_pantheon, cosmo_model, show=False):
    """Plot Hubble diagram + residuals, classic Pantheon+ style."""

    flat_samples = sampler.get_chain(thin=15, flat=True)
    
    z_grid = np.linspace(0.0001, df_agn['z'].max(), len(df_agn))

    if cosmo_model == 'FlatwCDM':
        label = r"Flat$w$CDM Model"
        cosmo_params = ['H0', 'Om0', 'w0']
        mu_models = np.array([Flatw0waCDM(H0=s[-3], Om0=s[-2], w0=s[-1]).distmod(z_grid).value for s in flat_samples])
    elif cosmo_model == 'Flatw0waCDM':
        label = r"Flat$w_0w_a$CDM Model"
        cosmo_params = ['H0', 'Om0', 'w0', 'wa']
        mu_models = np.array([Flatw0waCDM(H0=s[-4], Om0=s[-3], w0=s[-2], wa=s[-1]).distmod(z_grid).value for s in flat_samples])

    model_labels = ['alpha', 'M0', 'log_f'] + cosmo_params
    results = {key: np.percentile(flat_samples[:, i], [16, 50, 84]) for i, key in enumerate(model_labels)}


    # --- Cosmology model ---    
    mu_model_median = np.percentile(mu_models, 50, axis=0)
    mu_model_16th = np.percentile(mu_models, 16, axis=0)
    mu_model_84th = np.percentile(mu_models, 84, axis=0)

    # --- AGN distance modulus ---
    Kcorr = K_corr(df_agn['z']) + K_corr(2)
    mu_pred = np.array([
        df_agn['apparent_mag_i'] - M_model_single(s[1], s[0], df_agn['log_sigma_UV'], df_agn['log_tau_UV_RF']) - Kcorr
        for s in flat_samples
    ])
    mu_pred_median = np.percentile(mu_pred, 50, axis=0)
    mu_pred_16th = np.percentile(mu_pred, 16, axis=0)
    mu_pred_84th = np.percentile(mu_pred, 84, axis=0)
    mu_pred_std = np.sqrt(df_agn['apparent_mag_i_err']**2 +
            np.abs(0.5 * (mu_pred_84th - mu_pred_16th))**2 +
                 (-2.5 * 0.3 * np.log10(1 + df_agn["z"]))**2 +
                 #(Kcorr * 0.05)**2 +
                (results["alpha"][1] * np.sqrt((2*df_agn['log_sigma_UV_err'])**2+df_agn['log_tau_UV_RF_err']**2))**2)

    #--- Residuals ---
    mu_interp = np.interp(df_agn["z"], z_grid, mu_model_median)
    residuals = mu_pred_median - mu_interp

    # --- Binning ---
    bins = np.linspace(df_agn["z"].min(), df_agn["z"].max(), 25)
    bin_indices = np.digitize(df_agn["z"], bins)
    binned_mu_pred_median = [np.mean(mu_pred_median[bin_indices == i]) for i in range(1, len(bins))]
    binned_mu_pred_std = [np.std(mu_pred_median[bin_indices == i])/np.sqrt(len(mu_pred_median[bin_indices == i])) for i in range(1, len(bins))]
    binned_z = [np.median(df_agn["z"][bin_indices == i]) for i in range(1, len(bins))]


    # --- Plot setup ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    inset_ax = inset_axes(ax, width="40%", height="40%", loc="lower right", borderpad=1.5)

    # --- Inset plot ---
    inset_ax.scatter(df_agn["z"], mu_pred_median, s=2, label="AGN", alpha=0.5, color="black",zorder=-9)
    inset_ax.scatter(df_pantheon["zHD"], df_pantheon["MU_SH0ES"], color="dodgerblue", s=2, label="SN Ia", alpha=0.5, zorder=-10)
    inset_ax.plot(z_grid, mu_model_median, alpha=0.9, color="purple", zorder=-9, lw=0.5, label="ΛCDM Model")
    inset_ax.fill_between(z_grid, mu_model_16th, mu_model_84th, color="purple", alpha=0.9)
    inset_ax.set_xscale('log')
    inset_ax.set_xlim(0.02, 4.2)
    inset_ax.set_ylim(32, 51)
    inset_ax.set_xlabel(r"$z$", fontsize=12, labelpad=-10)
    inset_ax.set_ylabel(r"$\mu$ (mag)", fontsize=12)
    inset_ax.tick_params(axis='both', which='major', labelsize=10)
#    inset_ax.legend(frameon=False, fontsize=6)

    # --- Main Hubble diagram ---
    # Original AGNs
    ax.errorbar(df_agn["z"], mu_pred_median, yerr=mu_pred_std, fmt='o', linestyle='none',
                markersize=2, alpha=0.2, color='k', lw=1.5, zorder=-10, label="AGN")
    # Binned AGNs
    ax.errorbar(binned_z, binned_mu_pred_median, yerr=binned_mu_pred_std, label="Binned AGN",
                fmt='o', markersize=4, capsize=3, lw=1.5,alpha=0.9, color="red", zorder=-7)

    # SNIa points
    ax.errorbar(df_pantheon["zHD"], df_pantheon["MU_SH0ES"], yerr=df_pantheon["MU_SH0ES_ERR_DIAG"], 
                fmt='s', markersize=3, color="dodgerblue", linestyle='none', lw=1, label="SN Ia", alpha=0.7, zorder=-8)

    # Cosmo model band
    ax.plot(z_grid, mu_model_median, alpha=0.9, color="m", zorder=-5, lw=2, label=label)
    ax.fill_between(z_grid, mu_model_16th, mu_model_84th, color="purple", alpha=0.9, zorder=-5)

    ax.set_ylabel(r"$\mu$ (mag)")
    ax.set_xlabel(r"$z$")
    ax.set_xlim(-0.2, df_agn['z'].max())
    ax.set_ylim(26, 51)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.3, 0.05))

    # Ticks styling
    for axi in [ax, inset_ax]:
        axi.minorticks_on()
        axi.tick_params(axis='both', which='minor', direction='in', length=4, top=True, right=True, width=2)
        axi.tick_params(axis='both', which='major', direction='in', length=8, top=True, right=True)

    fig.tight_layout()
    plt.savefig(f"plots/hubble_diagram_{cosmo_model}.pdf", dpi=300)
    plt.savefig(f"plots/hubble_diagram_{cosmo_model}.png")
    if show:
        plt.show()
    plt.close()
    return residuals, mu_pred_std



def plot_predicted_vs_actual_Mi(sampler, df_agn, cosmo_model, show=False):
    """Plot predicted vs actual M_i binned by redshift in multiple panels."""

    flat_samples = sampler.get_chain(flat=True, thin=15)
    # --- Extract parameters ---
    if cosmo_model == 'FlatwCDM':
        cosmo_params = ['H0', 'Om0', 'w0']
    elif cosmo_model == 'Flatw0waCDM':
        cosmo_params = ['H0', 'Om0', 'w0', 'wa']
    else:
        raise ValueError("Invalid cosmology model.")

    model_labels = ['alpha', 'M0', 'log_f'] + cosmo_params
    results = {key: np.percentile(flat_samples[:, i], [16, 50, 84]) for i, key in enumerate(model_labels)}

    # --- Predicted M_i ---
    Kcorr = K_corr(df_agn['z']) + K_corr(2)
    M_pred = np.array([
        M_model_single(s[1], s[0], df_agn['log_sigma_UV'], df_agn['log_tau_UV_RF'])- Kcorr
        for s in flat_samples
    ]) 


    M_pred_median = np.percentile(M_pred, 50, axis=0)
    M_pred_16th = np.percentile(M_pred, 16, axis=0)
    M_pred_84th = np.percentile(M_pred, 84, axis=0)

    # --- Binning by redshift ---
    z_bins = np.linspace(df_agn['z'].min(), df_agn['z'].max(), 12)
    bin_indices = np.digitize(df_agn['z'], z_bins)
    binned_z = [np.median(df_agn['z'][bin_indices == i]) for i in range(1, len(z_bins))]
    binned_M_pred = [np.median(M_pred_median[bin_indices == i]) for i in range(1, len(z_bins))]
    binned_M_actual = [np.median(df_agn['M_i'][bin_indices == i]) for i in range(1, len(z_bins))]
    binned_M_pred_err = [np.std(M_pred_median[bin_indices == i]) for i in range(1, len(z_bins))]
    binned_M_actual_err = [np.std(df_agn['M_i'][bin_indices == i]) for i in range(1, len(z_bins))]

    # --- Plot setup ---
    fig, axes = plt.subplots(len(binned_z)//3, 3, figsize=(12, 8), sharey=True, sharex=True)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i >= len(binned_z):
            ax.axis("off")
            continue

        mask = bin_indices == i + 1
        ax.errorbar(
            df_agn['M_i'][mask], M_pred_median[mask],
            xerr=df_agn['M_i'][mask].std(), yerr=M_pred_median[mask].std(),
            fmt='o', markersize=2, alpha=0.6, label=f"z ~ {binned_z[i]:.2f}", lw=1
        )
        ax.plot([-30, -20], [-30, -20], 'k--', lw=1, label="1:1 Line")
        ax.set_xlim(-30, -20)
        ax.set_ylim(-30, -20)
        ax.set_xlabel(r"Actual $M_i$")
        if i == 0:
            ax.set_ylabel(r"Predicted $M_i$")
        ax.legend(frameon=False, fontsize=8)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.suptitle(f"Predicted vs Actual $M_i$ Binned by Redshift ({cosmo_model})", fontsize=16)
    #fig.tight_layout()
    plt.savefig(f"plots/predicted_vs_actual_Mi_{cosmo_model}.png", dpi=300)
    plt.savefig(f"plots/predicted_vs_actual_Mi_{cosmo_model}.pdf", dpi=300)
    if show:
        plt.show()
    plt.close()



def main():
    print("Loading quasar data...")
    df_agn = load_quasar_data()
    
    print("Loading Pantheon data...")
    df_pantheon = pd.read_csv('https://github.com/PantheonPlusSH0ES/DataRelease/blob/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat?raw=true',
                            sep=r'\s+')
    print("Done loading data.")

    for cosmo_model in ['Flatw0waCDM', 'FlatwCDM']:
        print(f"Plotting corner plot for {cosmo_model}...")
        print("Running MCMC for SNIa only...")
        sampler_sna, _ = run_mcmc_pipeline(df_agn, df_pantheon, only_sna=True, cosmo_model=cosmo_model)
        print("Running MCMC for SNIa + AGN...")
        sampler_agn, _ = run_mcmc_pipeline(df_agn, df_pantheon, only_sna=False, cosmo_model=cosmo_model)
        print("Plotting Hubble diagram...")
        plot_hubble(sampler_agn, df_agn, df_pantheon, cosmo_model=cosmo_model)

        print("Plotting corner plot...")
        plot_cosmo_corner(sampler_sna, sampler_agn, cosmo_model=cosmo_model)
        print(f"Corner plot for {cosmo_model} saved.")
        plot_predicted_vs_actual_Mi(sampler_agn, df_agn, cosmo_model=cosmo_model)
        print(f"Predicted vs Actual M_i plot for {cosmo_model} saved.")
        #break
    print("All plots saved.")

if __name__ == "__main__":
    main()