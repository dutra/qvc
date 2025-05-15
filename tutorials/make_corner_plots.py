import numpy as np
import pandas as pd
import h5py
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM
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


import matplotlib.pyplot as plt

bands = ['u', 'g', 'r', 'i', 'z']#, 'y']
bands_idx = {b: i for i, b in enumerate(bands)}

def read_quasars_from_hdf5(file_path):
    quasar_list = []

    hdul = fits.open('data/dr16q_prop_May01_2024.fits')
    fits_data = hdul[1].data  # Assuming the data is in the first extension    
    fits_data_2 = hdul[2].data  # Assuming the data is in the first extension    


    with h5py.File(file_path, "r") as hdf:
        for group_name in hdf.keys():
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

def load_quasar_data(file_path="s82_objs.h5"):

    quasar_list = read_quasars_from_hdf5("data/may12_objs_tauwavelength_taublr_redbands_ds4_merged.h5")
    #quasar_list = read_quasars_from_hdf5("data/may13_objs_tauwavelength_taublr_freebreak_merged.h5")

    print("Number of quasars loaded:", len(quasar_list))

    filtered_quasar_list = []
    for q in quasar_list:
        accept = True    
        for i, b in enumerate(q['clean_bands']):
            if len(q['magerrs'][b]) == 0:
                continue
            if not ((10**q[f'log_sigma_band'][bands_idx[b]])**2 > 1*(np.nanmean(q['magerrs'][b])**2 + (10**q[f'log_jitter'][i])**2)):
                accept = False

        
        #calc_lam_RF(q)
        #q['delta_time'] = q['times']['r'].max() - q['times']['r'].min()
        if True:
            filtered_quasar_list.append(q)


    df = pd.DataFrame(filtered_quasar_list)
    #df = df.set_index('object_id')
    #df = df.drop(columns=['log_jitter', 'magerrs_mean', 'log_sigma_band', 'log_sigma_band_err', 'clean_bands'])
    df = df.drop(columns=['mags', 'times', 'magerrs'])

    df_all = df.copy()

    # Remove infinite values from numeric columns
    columns_with_nans = df.columns[df.isna().any()].tolist()
    print("Columns with NaNs:", columns_with_nans)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].where(~np.isinf(df[numeric_cols]), np.nan)
    df = df.dropna()

    # data cuts
    # df = df[df['log_sigma_UV'] < 0]
    # df = df[df['log_tau_UV_RF'] > 1]
    # df = df[df['log_lbol'] > 44]
    # df = df[df['log_mbh'] > 1]
    # df = df[df['apparent_mag_i'] < 30]
    # #df = df[df['M_i'] > -27]
    #df = df[df['ebv'] < 0.05]

    num_quasars_z_0_1 = len(df[(df['z'] > 0) & (df['z'] <= 1)])
    num_quasars_z_gt_3 = len(df[df['z'] > 3])

    print("Number of quasars with 0 < z <= 1:", num_quasars_z_0_1)
    print("Number of quasars with z > 3:", num_quasars_z_gt_3)
    print("Final number of quasars:", len(df))
    return df


print("Loading quasar data...")
df = load_quasar_data()
print("Loading Pantheon data...")
df_pantheon = pd.read_csv('https://github.com/PantheonPlusSH0ES/DataRelease/blob/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat?raw=true',
                        delim_whitespace=True)

print("Done loading data.")

def get_completeness_function(df):
    n_bins_completeness = 26

    # Load the magnitude true dist data
    file_path = "stacked_sampled_apparent_magnitudes.h5"

    with h5py.File(file_path, "r") as f:
        mags_true = f["stacked_apparent_magnitudes"][:]

    mags_obs = df['apparent_mag_i'].values

    # Clean both datasets
    mags_true = mags_true[np.isfinite(mags_true)]
    mags_obs = mags_obs[np.isfinite(mags_obs)]

    # Histogram bins
    bin_edges = np.linspace(14, 26, n_bins_completeness)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dm = np.diff(bin_edges)[0]

    # Histogram both samples
    hist_true, _ = np.histogram(mags_true, bins=bin_edges)
    hist_obs, _ = np.histogram(mags_obs, bins=bin_edges)

    # Avoid division by zero
    mask = hist_true > 0
    completeness_ratio = np.zeros_like(hist_true, dtype=float)
    completeness_ratio[mask] = hist_obs[mask] / hist_true[mask]

    # Smooth or regularize if needed
    # Apply Gaussian smoothing to the completeness ratio
    #completeness_ratio = gaussian_filter1d(completeness_ratio, sigma=0.5)
    completeness_ratio = np.clip(completeness_ratio, 0, 1)

    # Interpolation
    p_detect_interp = interp1d(
        bin_centers,
        completeness_ratio,
        kind='quadratic',
        bounds_error=False,
        fill_value=(1.0, 0.0)
    )

    # --- Build Empirical Completeness Function ---
    mag_eval = np.linspace(14, 28, 500)
    dm = mag_eval[1] - mag_eval[0]
    p_detect = p_detect_interp(mag_eval)
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
z_pivot = df['z'].median()
sigma_pivot = -0.8
tau_pivot = 2.0

# --- Model Functions ---

def M_model_single(M0, alpha, delta, log_sigma_UV, log_tau_UV_RF):
    """Variability-luminosity relation for one AGN."""
    return M0 + alpha * (log_sigma_UV - sigma_pivot) + delta * (log_tau_UV_RF - tau_pivot)

def K_corr(z, alpha_nu=-0.5):
    """Simple K-correction."""
    return -2.5 * (1 + alpha_nu) * np.log10(1 + z)



# --- Priors ---
priors = {
    "alpha": (0, 10),
    "delta": (-5, 5),
    "M0": (-30, -10),
    "log_f": (-3, 1),
    "H0": (60, 80),
    "Om0": (0.2, 0.7),
    "w0": (-3, 0),
    "wa": (-3, 3),
}
labels = list(priors.keys())

def log_likelihood(theta, cosmo_model, model_labels, model_priors, only_sna=False):
    params = dict(zip(model_labels, theta))

    # Check prior bounds
    for key, (low, high) in model_priors.items():
        if not (low < params[key] < high):
            return -np.inf

    # Cosmology model selection
    if cosmo_model == 'FlatLambdaCDM':
        cosmo = FlatLambdaCDM(H0=params['H0'], Om0=params['Om0'])
    else:
        cosmo = Flatw0waCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'], wa=params['wa'])

    # AGN data
    z = df['z'].values
    m_obs = df['apparent_mag_i'].values
    m_err = df['apparent_mag_i_err'].values
    log_sigma = df['log_sigma_UV'].values
    log_tau = df['log_tau_UV_RF'].values
    log_sigma_err = df['log_sigma_UV_err'].values
    log_tau_err = df['log_tau_UV_RF_err'].values

    mu_cosmo = cosmo.distmod(z).value
    Kcorr = K_corr(z) + K_corr(2)
    M_pred = M_model_single(params['M0'], params['alpha'], params['delta'], log_sigma, log_tau)
    mu_pred = m_obs - M_pred - Kcorr

    mu_err = np.sqrt(
        m_err**2 +
        (params['alpha'] * log_sigma_err)**2 +
        (params['delta'] * log_tau_err)**2 +
        (K_corr(z) * 0.05)**2 +
        (0.055 * z)**2 +
        np.exp(2 * params['log_f'])
    )

    dmu = mu_pred - mu_cosmo
    ll_agn = np.sum(stats.norm.logpdf(dmu, scale=mu_err))

    # Selection correction
    p_detect, mag_eval, dm = get_completeness_function(df)
    m_model = M_pred + mu_cosmo
    integrals = np.zeros(len(df))
    unique_errors = np.round(mu_err, 4)
    unique_vals = np.unique(unique_errors)

    for sigma in unique_vals:
        mask = (np.abs(mu_err - sigma) < 1e-6)
        if np.sum(mask) == 0:
            continue

        kernel = stats.norm.pdf(mag_eval, loc=0.0, scale=sigma)
        conv = fftconvolve(p_detect, kernel, mode="same") * dm
        integrals[mask] = np.interp(m_model[mask], mag_eval, conv)

    norm_correction = np.sum(np.log(integrals + 1e-20))

    # SNIa likelihood
    mu_snia = cosmo.distmod(df_pantheon['zHD']).value
    dmu_snia = df_pantheon['MU_SH0ES'] - mu_snia
    ll_snia = np.sum(stats.norm.logpdf(dmu_snia, scale=df_pantheon['MU_SH0ES_ERR_DIAG']))

    if only_sna:
        return ll_snia
    
    return ll_snia + ll_agn - norm_correction


def run_mcmc_pipeline(cosmo_model='Flatw0waCDM', only_sna=False):
    if cosmo_model == 'FlatLambdaCDM':
        cosmo_params = ['H0', 'Om0']
    elif cosmo_model == 'Flatw0waCDM':
        cosmo_params = ['H0', 'Om0', 'w0', 'wa']
    else:
        raise ValueError("cosmo_model must be 'FlatLambdaCDM' or 'Flatw0waCDM'")

    model_labels = ['alpha', 'delta', 'M0', 'log_f'] + cosmo_params
    ndim = len(model_labels)

    # Reduced priors for selected cosmology
    model_priors = {key: priors[key] for key in model_labels}

    nwalkers = ndim * 2 * 15
    nsteps_burnin, nsteps_production = 250, 250

    initial_pos = np.array([
        np.random.uniform(low, high, nwalkers) for low, high in model_priors.values()
    ]).T

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_likelihood,
            args=(cosmo_model, model_labels, model_priors, only_sna), pool=pool,
        )
        state = sampler.run_mcmc(initial_pos, nsteps_burnin, progress=True)

        sampler.reset()
        sampler.run_mcmc(state, nsteps_production, progress=True)

    flat_samples = sampler.get_chain(flat=True)

    fig = corner.corner(flat_samples, labels=model_labels, truths=None, show_titles=True, title_fmt=".2f")
    if only_sna:
        fig.suptitle("SNIa only", fontsize=16)
        plt.savefig("plots/corner_plot_sna.png", dpi=200)
    else:
        fig.suptitle("SNIa + AGN", fontsize=16)
        plt.savefig("plots/corner_plot_agn.png", dpi=200)
    #plt.show()
    plt.close()

    return flat_samples, sampler


def plot_corner(sampler_sna, sampler_agn, cosmo_model='Flatw0waCDM'):
# === Parameter setup ===
    if cosmo_model == 'FlatLambdaCDM':
        param_names = ["H0", "Om0"]
        labels = [r"$H_0$", r"$\Omega_M$"]
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

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            ax.tick_params(direction='in')

            if i < j:
                ax.axis("off")
            elif i == j:
                ax.hist(sna_data[:, i], bins=40, density=True, color="blue", histtype="step", linewidth=1.5)
                ax.hist(agn_data[:, i], bins=40, density=True, color="red", histtype="step", linewidth=1.5)
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
        Line2D([0], [0], color="blue", lw=2, label="SN Ia"),
        Line2D([0], [0], color="red", lw=2, label="AGN"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=14, frameon=False)

    # === Layout & Save ===
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
    fig.savefig(f"plots/corner_kde_{cosmo_model}.pdf", bbox_inches="tight", transparent=True)
    fig.savefig(f"plots/corner_kde_{cosmo_model}.png", bbox_inches="tight")
    #plt.show()
    plt.close()

def main():
    for cosmo_model in ['FlatLambdaCDM', 'Flatw0waCDM']:
        print(f"Plotting corner plot for {cosmo_model}...")
        print("Running MCMC for SNIa only...")
        flat_samples_sna, sampler_sna = run_mcmc_pipeline(only_sna=True, cosmo_model=cosmo_model)
        print("Running MCMC for SNIa + AGN...")
        flat_samples_agn, sampler_agn = run_mcmc_pipeline(only_sna=False, cosmo_model=cosmo_model)

        print("Plotting corner plot...")
        plot_corner(sampler_sna, sampler_agn, cosmo_model=cosmo_model)
        print(f"Corner plot for {cosmo_model} saved.")
    print("All corner plots saved.")

main()