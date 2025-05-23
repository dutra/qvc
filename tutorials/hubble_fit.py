import numpy as np
import pandas as pd
import emcee
import multiprocessing
import matplotlib.pyplot as plt
import corner
from astropy.cosmology import FlatwCDM, Flatw0waCDM
from scipy import stats
from scipy.signal import fftconvolve
import numpy as np
import pandas as pd
import h5py
from astropy.io import fits
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
from scipy.special import expit  # numerically stable sigmoid
import astropy.units as u
import math

import matplotlib.pyplot as plt

plt.style.use('style.mplstyle')

from hubble_utils import *
#from hubble_plotting import *
from hubble_model import *

# Placeholders for SN covariance (will be loaded in main)
Cov_inv = None
logdetCov = None

# --- Log-likelihood ---
def log_likelihood(theta, cosmo_model,
                   df_agn, df_pantheon, completeness_params,
                   log_sigma_hat_pivot,
                   only_sna=False, use_full_cov=False,
                   return_params=False):
    
    priors, model_labels = get_model_params(cosmo_model)
    model_priors = {key: priors[key] for key in model_labels}
    params = dict(zip(model_labels, theta))

    for key, (low, high) in model_priors.items():
        if not (low < params[key] < high):
            return -np.inf

    # Cosmology
    if cosmo_model == 'FlatwCDM':
        cosmo = FlatwCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'])
    else:
        cosmo = Flatw0waCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'], wa=params['wa'])

    # SN model
    mu_theory = cosmo.distmod(df_pantheon['zHD'].values).value
    mu_obs = M_model_SN(
        df_pantheon['m_b_corr'].values,
        df_pantheon['x1'].values,
        df_pantheon['c'].values,
        df_pantheon['biasCor_m_b'].values,
        df_pantheon['HOST_LOGMASS'].values,
        params['alpha_sn'], params['beta_sn'], params['M0_sn'],
        params['gamma_sn'], params['tau_Ms']
    )

    # --- Residuals between SN model and theory ---
    res_snia = mu_obs - mu_theory

    # Compute main SN likelihood (with or without covariance)
    if use_full_cov:
        global Cov_inv, logdetCov
        ll_snia = -0.5 * res_snia @ Cov_inv @ res_snia - 0.5 * logdetCov - 0.5 * len(res_snia) * np.log(2 * np.pi)
        # Use diagonal from covariance as proxy for Cepheid calibration residuals
        sigma = np.sqrt(np.diag(Cov_inv)**-1)  # approx only if Cov_inv is well-conditioned
    else:
        sigma = df_pantheon['MU_SH0ES_ERR_DIAG'].values
        ll_snia = np.sum(stats.norm.logpdf(res_snia, scale=sigma))

    # --- Cepheid calibration: always applied ---
    mask_calib = df_pantheon['IS_CALIBRATOR'].values == 1
    if np.any(mask_calib):
        mu_ceph = df_pantheon['CEPH_DIST'].values[mask_calib]
        mu_sn_calib = mu_obs[mask_calib]
        mu_ceph_err = sigma[mask_calib]  # proxy for CEPH_DIST error
        ll_calib = np.sum(stats.norm.logpdf(mu_sn_calib - mu_ceph, scale=mu_ceph_err))
    else:
        ll_calib = 0.0

    if only_sna:
        return ll_snia + ll_calib

    # AGN model
    z = df_agn['z'].values
    m_obs = df_agn['apparent_mag_i'].values
    m_err = df_agn['apparent_mag_i_err'].values
    log_sigma_hat = df_agn['log_sigma_hat_UV'].values
    log_sigma_hat_err = df_agn['log_sigma_hat_UV_err'].values

    mu_cosmo = cosmo.distmod(z).value
    M_pred = M_model_agn(params['M0_agn'], params['alpha_agn'], log_sigma_hat)
    mu_pred = m_obs - (K_corr(z) - K_corr(2)) - M_pred

    mu_err = np.sqrt(
        m_err**2 +
        (params['alpha_agn'] * 2*log_sigma_hat_err)**2 +
        (2.5 * 0.3 * np.log10(1 + z))**2 +
        (0.055 * z)**2 +
        np.exp(2 * params['log_f'])
    )

    dmu = mu_pred - mu_cosmo

    # if use_full_cov:  # TODO: test this
    #     # Compute AGN covariance matrix
    #     D = mu_err**2 - (0.055 * z)**2
    #     u = 0.055 * z
    #     inv_D = 1.0 / D

    #     # Compute quadratic form
    #     dmu = mu_pred - mu_cosmo
    #     inv_D_dmu = inv_D * dmu
    #     inv_D_u = inv_D * u
    #     alpha = 1.0 + np.dot(u, inv_D_u)
    #     quad = np.dot(dmu, inv_D_dmu) - (np.dot(inv_D_u, dmu) ** 2) / alpha

    #     # Log determinant
    #     logdet = np.sum(np.log(D)) + np.log(alpha)

    #     ll_agn = -0.5 * quad - 0.5 * logdet - 0.5 * len(dmu) * np.log(2 * np.pi)    
    # # else:
    ll_agn = np.sum(stats.norm.logpdf(dmu, scale=mu_err))

    # Corrected? AGN completeness correction 2D
    # Optional AGN completeness correction 2D
    norm_correction = 0.0
    if completeness_params is not None:
        completeness2d, mag_centers, z_centers, dm, dz = completeness_params
        m_model = M_pred + mu_cosmo

        integrals = np.zeros(len(df_agn))
        unique_err = np.round(mu_err, 4)

        mag_range = mag_centers - np.mean(mag_centers)  # Center kernel around zero explicitly

        for sigma in np.unique(unique_err):
            if sigma <= 0 or not np.isfinite(sigma):
                continue
            mask = np.abs(mu_err - sigma) < 1e-6

            # Symmetric kernel centered on zero magnitude offset
            kernel = stats.norm.pdf(mag_range, loc=0, scale=sigma)
            kernel /= kernel.sum() * dm  # normalize kernel properly

            conv_values = []
            for zval in z[mask]:
                p_z = completeness2d(mag_centers, np.full_like(mag_centers, zval))
                conv = fftconvolve(p_z, kernel, mode="same") * dm
                conv = np.clip(conv, 1e-12, 1.0)
                conv_values.append(conv)

            for i, idx in enumerate(np.where(mask)[0]):
                val = np.interp(m_model[idx], mag_centers, conv_values[i], left=1e-12, right=1e-12)
                integrals[idx] = val

        norm_correction = np.sum(np.log(np.clip(integrals, 1e-12, None)))


    # # Optional AGN completeness correction 2D
    # norm_correction = 0.0
    # if completeness_params is not None:
    #     p_detect, mag_centers, z_centers, dm, dz = completeness_params
    #     m_model = M_pred + mu_cosmo

    #     integrals = np.zeros(len(df_agn))
    #     unique_err = np.round(mu_err, 4)

    #     for sigma in np.unique(unique_err):
    #         if sigma <= 0 or not np.isfinite(sigma): continue
    #         mask = np.abs(mu_err - sigma) < 1e-6

    #         # 2D grid for convolution
    #         kernel = stats.norm.pdf(mag_centers - np.median(mag_centers), loc=0, scale=sigma)
    #         conv_values = []
    #         for zval in z[mask]:
    #             p_z = p_detect(mag_centers, np.full_like(mag_centers, zval))
    #             conv = fftconvolve(p_z, kernel, mode="same") * dm
    #             conv = np.clip(conv, 1e-12, 1.0)
    #             conv_values.append(conv)
    #         for i, idx in enumerate(np.where(mask)[0]):
    #             val = np.interp(m_model[idx], mag_centers, conv_values[i], left=1e-12, right=1e-12)
    #             integrals[idx] = val

    #     norm_correction = np.sum(np.log(np.clip(integrals, 1e-12, None)))
    # Optional AGN completeness correction 1D
    # norm_correction = 0
    # if completeness_params is not None:
    #     p_detect, mag_eval, dm = completeness_params
    #     m_model = M_pred + mu_cosmo
    #     integrals = np.zeros(len(df_agn))
    #     unique_err = np.round(mu_err, 4)
    #     for sigma in np.unique(unique_err):
    #         if sigma <= 0 or not np.isfinite(sigma): continue
    #         mask = np.abs(mu_err - sigma) < 1e-6
    #         kernel = stats.norm.pdf(mag_eval - np.median(mag_eval), loc=0, scale=sigma)
    #         conv = fftconvolve(p_detect, kernel, mode="same") * dm
    #         conv = np.clip(conv, 1e-12, 1.0)
    #         integrals[mask] = np.interp(m_model[mask], mag_eval, conv, left=1e-12, right=1e-12)
        # norm_correction = np.sum(np.log(np.clip(integrals, 1e-12, None)))

    if return_params:
        return ll_snia + ll_calib + ll_agn - norm_correction, norm_correction, m_model
    
    #print(norm_correction/ll_agn)
    return ll_snia + ll_calib + ll_agn - norm_correction

def run_mcmc_pipeline(df_agn, df_pantheon, cosmo_model='Flatw0waCDM', 
                      only_sna=False, completeness=True, use_full_cov=False,
                      num_samples=2000, num_warmup=2000):
    priors, model_labels = get_model_params(cosmo_model)
    ndim = len(model_labels)
    model_priors = {key: priors[key] for key in model_labels}

    # MCMC initialization
    nwalkers = ndim * 2 * 15
    num_warmup = num_warmup    # increased burn-in steps for better convergence
    num_samples = num_samples  # increased MCMC sample steps for robust posterior
    # Random initial positions drawn uniformly within prior bounds
    initial_pos = np.array([
        np.random.uniform(low, high, nwalkers) for low, high in model_priors.values()
    ]).T

    # Prepare data (reduce DataFrame size for efficiency)
    df_pantheon_filtered = df_pantheon[['zHD', 'MU_SH0ES', 'MU_SH0ES_ERR_DIAG', 'CEPH_DIST', 'IS_CALIBRATOR',
                                        'm_b_corr', 'x1', 'c', 'biasCor_m_b', 'HOST_LOGMASS']]  # SN data needed for likelihood
    df_agn_filtered = df_agn[['z', 'apparent_mag_i', 'apparent_mag_i_err',
                               'log_sigma_hat_UV', 'log_sigma_hat_UV_err']]
    completeness_params = get_completeness_function_2d(df_agn_filtered) if completeness else None
    #completeness_params = get_completeness_function_simple(20) if completeness else None
    log_sigma_hat_pivot = df_agn_filtered['log_sigma_hat_UV'].mean()
    print(f"Log sigma hat pivot: {log_sigma_hat_pivot:.3f}")

    # Run MCMC using EnsembleSampler with multiprocessing for speed
    num_cpus = multiprocessing.cpu_count()-2
    with multiprocessing.Pool(processes=num_cpus) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, 
                                        args=(cosmo_model, 
                                              df_agn_filtered, df_pantheon_filtered, 
                                              completeness_params, log_sigma_hat_pivot,
                                              only_sna, use_full_cov), 
                                        pool=pool)
        # Burn-in
        state = sampler.run_mcmc(initial_pos, num_warmup, progress=True)
        sampler.reset()
        # Main MCMC sampling
        sampler.run_mcmc(state, num_samples, progress=True)


    
    samples = sampler.get_chain(flat=True)
    # Compute median of each parameter from the samples
    median_samples = np.median(samples, axis=0)
    params = dict(zip(model_labels, median_samples))
    print("Median samples:", median_samples)
    print(f"Sampler shape: {samples.shape}")

    # Calculate m_model and mu_cosmo using the median samples
    if cosmo_model == 'Flatw0waCDM':
        mu_cosmo = Flatw0waCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'], wa=params['wa']).distmod(df_agn_filtered['z'].values).value
    else:
        mu_cosmo = FlatwCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0']).distmod(df_agn_filtered['z'].values).value

    mu_err = np.sqrt(
        df_agn_filtered['apparent_mag_i_err'].values**2 +
        (params['alpha_agn'] * 2*df_agn_filtered['log_sigma_hat_UV_err'].values)**2 +
        (2.5 * 0.3 * np.log10(1 + df_agn_filtered['z'].values))**2 +
        (0.055 * df_agn_filtered['z'].values)**2 +
        np.exp(2 * params['log_f'])
    )
    # Predict median uncensored AGN magnitudes
    M_model_agn_samples = M_model_agn(params['M0_agn'], params['alpha_agn'], df_agn_filtered['log_sigma_hat_UV'])
    m_model = mu_cosmo + (K_corr(df_agn_filtered['z'].values) - K_corr(2)) + M_model_agn_samples

    mag_corr = predict_uncensored_magnitudes(df_agn_filtered, m_model, mu_err)

    return sampler, model_labels, mag_corr


def predict_uncensored_magnitudes(df_agn, m_model, mu_err):
    z = df_agn['z'].values

    p_detect, mag_centers, z_centers, dm, dz = get_completeness_function_2d(df_agn)

    uncensored_samples = []

    for i in range(len(df_agn)):
        m = m_model[i]
        sigma = mu_err[i]
        zval = z[i]

        # Grid of possible m* values
        m_grid = np.linspace(m - 5*sigma, m + 5*sigma, 500)

        # Gaussian prior
        prior = stats.norm.pdf(m_grid, loc=m, scale=sigma)

        # Detection probability at each m
        p_det = p_detect(m_grid, np.full_like(m_grid, zval))
        p_det = np.clip(p_detect(m_grid, np.full_like(m_grid, zval)), 1e-12, 1.0)

        if np.all(p_det == 0):
            sampled_m = df_agn['apparent_mag_i'].iloc[i]
            uncensored_samples.append(sampled_m)
            continue

        # Posterior (up to normalization)
        posterior = prior * p_det
        posterior /= np.trapz(posterior, m_grid)  # normalize

        # Sample or get expected value
        #sampled_m = np.random.choice(m_grid, p=posterior / posterior.sum())
        mean_m = np.trapz(m_grid * posterior, m_grid)

        uncensored_samples.append(mean_m)

    return np.array(uncensored_samples)



def main():
    print("Loading quasar data...")
    #df_agn = load_quasar_data("data/may12_objs_tauwavelength_taublr_redbands_ds4_merged.h5")
    df_agn = load_quasar_data("data/N20_w500_grace/may21_joint_fits_N20_merged.h5")
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
    global Cov_inv, logdetCov
    Cov_inv = np.linalg.inv(cov_matrix)
    sign, logdet = np.linalg.slogdet(cov_matrix)
    if sign <= 0:
        raise ValueError("Covariance matrix is not positive-definite!")
    logdetCov = logdet
    print("Data loaded. Running joint cosmographic fits...")

    num_warmup, num_samples = 250, 50
    # Run MCMC fits for SNIa only and SNIa+AGN, for each cosmological model
    for cosmo_model in ['Flatw0waCDM', 'FlatwCDM']:
        # print(f"Running MCMC for {cosmo_model}: SNIa only")
        # sampler_snia, _ = run_mcmc_pipeline(df_agn, df_pantheon, cosmo_model=cosmo_model, only_sna=True, completeness=True, use_full_cov=True,
        #                                      num_warmup=num_warmup, num_samples=num_samples)
        # plot_corner(sampler_snia, cosmo_model=cosmo_model, only_sna=True)
        
        print(f"Running MCMC for {cosmo_model}: SNIa + AGN")
        sampler_joint, _ = run_mcmc_pipeline(df_agn, df_pantheon, cosmo_model=cosmo_model, only_sna=False, completeness=True, use_full_cov=False,
                                             num_warmup=num_warmup, num_samples=num_samples)
        plot_corner(sampler_joint, cosmo_model=cosmo_model, only_sna=False)
        
        # Plot results
        print("Plotting Hubble diagram...")
        plot_hubble(sampler_joint, df_agn, df_pantheon, cosmo_model=cosmo_model)

        print("Plotting cosmological posteriors corner plot...")
        plot_cosmo_corner(sampler_joint, sampler_joint, cosmo_model=cosmo_model)
        print("Plotting AGN M_i predictions vs actual...")
        plot_predicted_vs_actual_Mi(sampler_joint, df_agn, cosmo_model=cosmo_model)
        print("Plotting completeness vs magnitude at redshifts...")
        p_detect, mag_centers, z_centers, dm, dz = get_completeness_function_2d(df_agn)
        plot_completeness_vs_mag_at_redshifts(p_detect, mag_centers, z_centers)
        print(f"Finished plots for {cosmo_model}\n")
        #break
    print("All analyses complete. Results saved to 'plots/' directory.")

def test():
    #df_agn = load_quasar_data("data/may12_objs_tauwavelength_taublr_redbands_ds4_merged.h5")
    df_agn = load_quasar_data("data/N20_w500_grace/may21_joint_fits_N20_merged.h5")
    print("Plotting completeness vs magnitude at redshifts...")
    p_detect, mag_centers, z_centers, dm, dz = get_completeness_function_2d(df_agn)
    plot_completeness_vs_mag_at_redshifts(p_detect, mag_centers, z_centers)

    #df_agn = df_agn.sample(n=500, random_state=42).reset_index(drop=True)
    # Load Pantheon+ SN metadata
    df_pantheon = pd.read_csv(
        #"https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat",
        "data/Pantheon+SH0ES.dat",
        sep=r"\s+"
    )

    cosmo_model = 'Flatw0waCDM'
    sampler_joint, _ = run_mcmc_pipeline(df_agn, df_pantheon, cosmo_model=cosmo_model, 
                                         only_sna=False, completeness=True, use_full_cov=False,
                                         num_warmup=20, num_samples=50)
    plot_corner(sampler_joint, cosmo_model=cosmo_model, only_sna=False)
    print("Plotting Hubble diagram...")
    plot_hubble(sampler_joint, df_agn, df_pantheon, cosmo_model=cosmo_model)
    print("Plotting cosmological posteriors corner plot...")
    plot_cosmo_corner(sampler_joint, sampler_joint, cosmo_model=cosmo_model)
    print("Plotting AGN M_i predictions vs actual...")
    plot_predicted_vs_actual_Mi(sampler_joint, df_agn, cosmo_model=cosmo_model)

if __name__ == "__main__":
    main()
    #test()