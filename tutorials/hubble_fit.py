import numpy as np
import pandas as pd
import emcee
import matplotlib.pyplot as plt
from astropy.cosmology import FlatwCDM, Flatw0waCDM, FlatLambdaCDM
from scipy import stats
from scipy.signal import fftconvolve
import numpy as np
import pandas as pd
from scipy import stats
import corner
from tqdm import tqdm
from dynesty import DynamicNestedSampler
from dynesty.utils import resample_equal
import pickle

import matplotlib.pyplot as plt

plt.style.use('style.mplstyle')

from hubble_utils import *
from hubble_plotting import *
from hubble_model import *

# Placeholders for SN covariance (will be loaded in main)
Cov_inv = None
logdetCov = None
L = None
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
    elif cosmo_model == 'Flatw0waCDM':
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
        global Cov_inv, logdetCov, L
        #ll_snia = -0.5 * res_snia @ Cov_inv @ res_snia - 0.5 * logdetCov - 0.5 * len(res_snia) * np.log(2 * np.pi)
        # Use Cholesky decomposition
        y = L @ res_snia
        ll_snia = -0.5 * np.dot(y, y) - np.sum(np.log(np.diag(L))) - 0.5 * len(res_snia) * np.log(2 * np.pi)
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
    log_tau_UV_RF = df_agn['log_tau_UV_RF'].values
    log_tau_UV_RF_err = df_agn['log_tau_UV_RF_err'].values

    mu_cosmo = cosmo.distmod(z).value
    M_pred = M_model_agn(params['M0_sn'], params['delta_M_agn'], params['alpha_agn'], params['beta_agn'], log_sigma_hat, log_tau_UV_RF)
    mu_pred = m_obs - (K_corr(z) - K_corr(2)) - M_pred

    mu_err = np.sqrt(
        m_err**2 +
        (params['alpha_agn'] * 2*log_sigma_hat_err)**2 +
        (params['beta_agn'] * log_tau_UV_RF_err)**2 +
        (2.5 * 0.3 * np.log10(1 + z))**2 +
        (0.055 * z)**2 +
        np.exp(2 * params['log_f'])
    )

    dmu = mu_pred - mu_cosmo

    if use_full_cov:  # TODO: test this
        # Compute AGN covariance matrix
        D = mu_err**2 - (0.055 * z)**2
        u = 0.055 * z
        inv_D = 1.0 / D

        # Compute quadratic form
        dmu = mu_pred - mu_cosmo
        inv_D_dmu = inv_D * dmu
        inv_D_u = inv_D * u
        alpha = 1.0 + np.dot(u, inv_D_u)
        quad = np.dot(dmu, inv_D_dmu) - (np.dot(inv_D_u, dmu) ** 2) / alpha

        # Log determinant
        logdet = np.sum(np.log(D)) + np.log(alpha)

        ll_agn = -0.5 * quad - 0.5 * logdet - 0.5 * len(dmu) * np.log(2 * np.pi)  
    else:
        ll_agn = np.sum(stats.norm.logpdf(dmu, scale=mu_err))

    # Corrected? AGN completeness correction 2D
    # Optional AGN completeness correction 2D
    per_agn_log_weights = 0.0
    m_model = M_pred + (K_corr(df_agn['z'].values) - K_corr(2)) + mu_cosmo  # model-predicted magnitude
    sigma = mu_err               # total uncertainty on m_model
    if False and completeness_params is not None:
        completeness2d, mag_centers, z_centers, dm, dz = completeness_params
        m_model = M_pred + (K_corr(df_agn['z'].values) - K_corr(2)) + mu_cosmo  # model-predicted magnitude

        sigma = mu_err               # total uncertainty on m_model

        # Define the magnitude grid for integration (must cover m_model ± ~4σ)
        m_grid = mag_centers
        m_grid_broadcasted = np.tile(m_grid, (len(z), 1))      # (N_obj, N_grid)
        z_broadcasted = np.tile(z[:, None], (1, len(m_grid)))  # (N_obj, N_grid)

        # Evaluate the Gaussian p(m | m_model, sigma) for each AGN
        gauss_weights = stats.norm.pdf(m_grid_broadcasted, loc=m_model[:, None], scale=sigma[:, None])
        gauss_weights /= np.sum(gauss_weights, axis=1)[:, None] * dm

        # Evaluate completeness function p(I=1 | m, z_i)
        p_detect = completeness2d(m_grid_broadcasted, z_broadcasted)
        integrals = np.sum(gauss_weights * p_detect, axis=1) * dm
        integrals = np.clip(integrals, 1e-12, 1.0)

        # Apply selection correction to log-likelihood
        per_agn_log_weights = np.log(integrals)
        ll_agn -= np.sum(per_agn_log_weights)

    if return_params:
        return ll_snia + ll_calib + ll_agn, per_agn_log_weights, m_model
    
    return ll_snia + ll_calib + ll_agn

# Globals used by dynesty
_dynesty_config = {}

def prior_transform_dynesty(unit_cube):
    priors = _dynesty_config['model_priors']
    keys = _dynesty_config['model_labels']
    return [priors[key][0] + (priors[key][1] - priors[key][0]) * x
            for x, key in zip(unit_cube, keys)]

def loglike_dynesty(theta):
    cfg = _dynesty_config
    return log_likelihood(theta,
                          cfg['cosmo_model'],
                          cfg['df_agn'],
                          cfg['df_pantheon'],
                          cfg['completeness_params'],
                          cfg['log_sigma_hat_pivot'],
                          cfg['only_sna'],
                          cfg['use_full_cov'])

def run_mcmc_pipeline(df_agn, df_pantheon, cosmo_model='Flatw0waCDM', 
                      only_sna=False, completeness=True, use_full_cov=False,
                      num_samples=2000, num_warmup=2000, 
                      use_dynesty=False, nlive=500, maxiter=None, n_effective=5000):
    import multiprocessing

    priors, model_labels = get_model_params(cosmo_model)
    model_priors = {key: priors[key] for key in model_labels}
    ndim = len(model_labels)

    # Prepare data
    df_pantheon_filtered = df_pantheon[['zHD', 'MU_SH0ES', 'MU_SH0ES_ERR_DIAG', 'CEPH_DIST', 'IS_CALIBRATOR',
                                        'm_b_corr', 'x1', 'c', 'biasCor_m_b', 'HOST_LOGMASS']].copy()
    df_agn_filtered = df_agn[['z', 'apparent_mag_i', 'apparent_mag_i_err', 'M_i',
                              'log_sigma_hat_UV', 'log_sigma_hat_UV_err', 'log_tau_UV_RF', 'log_tau_UV_RF_err']].copy()
    completeness_params = get_completeness_function_2d(df_agn_filtered) if completeness else None
    log_sigma_hat_pivot = df_agn_filtered['log_sigma_hat_UV'].mean()
    print(f"Log sigma hat pivot: {log_sigma_hat_pivot:.3f}")
    print(f"Mean AGN M_i: ", df_agn_filtered['M_i'].mean())

    if use_dynesty:
        # Set dynesty global context
        _dynesty_config.update({
            'model_priors': model_priors,
            'model_labels': model_labels,
            'cosmo_model': cosmo_model,
            'df_agn': df_agn_filtered,
            'df_pantheon': df_pantheon_filtered,
            'completeness_params': completeness_params,
            'log_sigma_hat_pivot': log_sigma_hat_pivot,
            'only_sna': only_sna,
            'use_full_cov': use_full_cov,
        })

        num_cpus = multiprocessing.cpu_count() - 2
        with multiprocessing.Pool(processes=num_cpus) as pool:
            sampler = DynamicNestedSampler(
                loglike_dynesty,
                prior_transform_dynesty,
                ndim,
                bound='single', # assuming gaussian like posterior, if multimodal use 'multi'
                sample='unif', # fast works well for moderately correlated parameters, use 'auto' if unsure
                nlive=nlive,
                pool=pool,
                queue_size=num_cpus
            )
            sampler.run_nested(maxiter=maxiter, n_effective=n_effective, print_progress=True)
            results = sampler.results
        logZ, logZerr = results.logz[-1], results.logzerr[-1]
        print(f"\nBayesian evidence logZ = {logZ:.2f} ± {logZerr:.2f}")
        if logZerr > 1:
            print("Warning: logZ error is large, consider increasing nlive or maxiter.")

        samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
        resampled = resample_equal(samples, weights)
        median_samples = np.median(resampled, axis=0)
    else:
        # --- emcee logic ---
        nwalkers = ndim * 2 * 15
        initial_pos = np.array([
            np.random.uniform(low, high, nwalkers) for low, high in model_priors.values()
        ]).T

        num_cpus = multiprocessing.cpu_count() - 2
        with multiprocessing.Pool(processes=num_cpus) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood,
                                            args=(cosmo_model,
                                                  df_agn_filtered, df_pantheon_filtered,
                                                  completeness_params, log_sigma_hat_pivot,
                                                  only_sna, use_full_cov),
                                            pool=pool)
            state = sampler.run_mcmc(initial_pos, num_warmup, progress=True)
            sampler.reset()
            sampler.run_mcmc(state, num_samples, progress=True)

        samples = sampler.get_chain(flat=True)

        # Check convergence using autocorrelation time
        try:
            tau = sampler.get_autocorr_time(quiet=True)
            converged = np.all((sampler.iteration > 50 * tau) & (tau * 100 < sampler.iteration))
            print(f"Autocorr time: {tau}")
            if converged:
                print("MCMC appears to have converged.")
            else:
                print("Warning: MCMC may not have converged. Consider running for more steps.")
        except Exception as e:
            print(f"Could not compute autocorrelation time: {e}")
        
        # save sampler and samples
        with open(f"data/sampler_emcee_{cosmo_model}_{'sna' if only_sna else 'agn'}.pkl", "wb") as f:
            pickle.dump(sampler, f)
        np.save(f"data/samples_emcee_{cosmo_model}_{'sna' if only_sna else 'agn'}.npy", samples)

        median_samples = np.median(samples, axis=0)
        logZ = logZerr = None  # Not available from emcee

    # Common post-processing
    params = dict(zip(model_labels, median_samples))
    if cosmo_model == 'Flatw0waCDM':
        mu_cosmo = Flatw0waCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'], wa=params['wa']).distmod(df_agn_filtered['z'].values).value
    elif cosmo_model == 'FlatwCDM':
        mu_cosmo = FlatwCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0']).distmod(df_agn_filtered['z'].values).value
    elif cosmo_model == 'FlatLambdaCDM':
        mu_cosmo = FlatLambdaCDM(H0=params['H0'], Om0=params['Om0']).distmod(df_agn_filtered['z'].values).value
    else:
        raise ValueError("cosmo_model must be 'FlatwCDM', 'Flatw0waCDM' or 'FlatLambdaCDM'")

    mu_err = np.sqrt(
        df_agn_filtered['apparent_mag_i_err'].values**2 +
        (params['alpha_agn'] * 2 * df_agn_filtered['log_sigma_hat_UV_err'].values)**2 +
        (params['beta_agn'] * df_agn_filtered['log_tau_UV_RF_err'].values)**2 +
        (2.5 * 0.3 * np.log10(1 + df_agn_filtered['z'].values))**2 +
        (0.055 * df_agn_filtered['z'].values)**2 +
        np.exp(2 * params['log_f'])
    )

    M_model_agn_samples = M_model_agn(params['M0_sn'], params['delta_M_agn'], params['alpha_agn'], params['beta_agn'], 
                                      df_agn_filtered['log_sigma_hat_UV'], df_agn_filtered['log_tau_UV_RF'])
    m_model = mu_cosmo + (K_corr(df_agn_filtered['z'].values) - K_corr(2)) + M_model_agn_samples
    mag_corr = predict_uncensored_magnitudes(df_agn_filtered, m_model, mu_err)
    df_agn['apparent_mag_i_corr'] = mag_corr

    return sampler, model_labels, mag_corr, logZ, logZerr

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
        posterior = prior #* p_det
        posterior /= np.trapezoid(posterior, m_grid)  # normalize

        # Sample or get expected value
        sampled_m = np.random.choice(m_grid, p=posterior / posterior.sum())
        #mean_m = np.trapz(m_grid * posterior, m_grid)

        #uncensored_samples.append(mean_m)
        uncensored_samples.append(sampled_m)

    return np.array(uncensored_samples)

def compare_models():
    priors, model_labels = get_model_params('Flatw0waCDM')
    ndim = len(priors.keys())
    nlive = 25 * ndim # basic
    # nlive = 50 * ndim # modeerate precision
    # nlive = 100 * ndim # high precision
    #maxiter = 2000 # TESTING
    maxiter = None # full run
    n_effective = 100 # Testing
    #n_effective = 2000 # moderate quality
    #n_effective = 5000 # high quality
    #n_effective = 10000 # publication quality
    
    use_full_cov = True
    completeness = True

    global Cov_inv, logdetCov, L
    df_agn, df_pantheon, Cov_inv, logdetCov, L = load_data("data/N20_w500_grace/may21_joint_fits_N20_merged.h5")

    # Run MCMC pipeline for FlatLambdaCDM model
    sampler_FlatwCDM, _, _, logZ_FlatwCDM, logZerr_FlatwCDM = run_mcmc_pipeline(
                                        df_agn, df_pantheon, cosmo_model='FlatwCDM', 
                                        only_sna=False, completeness=completeness, use_full_cov=use_full_cov,
                                        use_dynesty=True, nlive=nlive, maxiter=maxiter, n_effective=n_effective)
    #plot_posterior_corner(sampler_FlatwCDM, cosmo_model='FlatwCDM', only_sna=False, dynasty=True)
    plot_traces(sampler_FlatwCDM, cosmo_model='FlatwCDM', only_sna=False, dynasty=True)

    # Run MCMC pipeline for Flatw0waCDM model
    sampler_Flatw0waCDM, _, _, logZ_Flatw0waCDM, logZerr_Flatw0waCDM = run_mcmc_pipeline(
                                        df_agn, df_pantheon, cosmo_model='Flatw0waCDM', 
                                        only_sna=False, completeness=completeness, use_full_cov=use_full_cov,
                                        use_dynesty=True, nlive=nlive, maxiter=maxiter, n_effective=n_effective)

    #plot_posterior_corner(sampler_Flatw0waCDM, cosmo_model='Flatw0waCDM', only_sna=False, dynasty=True)
    plot_traces(sampler_Flatw0waCDM, cosmo_model='Flatw0waCDM', only_sna=False, dynasty=True)
    
    compare_models_by_log_evidence(logZ_FlatwCDM, logZerr_FlatwCDM, logZ_Flatw0waCDM, logZerr_Flatw0waCDM,
                                    model_1_name="FlatwCDM", model_2_name="Flatw0waCDM")

def main():
    # Load data
    global Cov_inv, logdetCov, L
    df_agn, df_pantheon, Cov_inv, logdetCov, L = load_data("data/may23_all_merged.h5", populate_sdss=False)
    #df_agn, df_pantheon, Cov_inv, logdetCov, L = load_data("data/N20_w500_grace/may21_joint_fits_N20_merged.h5", False)
    
    results = []

    num_warmup, num_samples = 50, 50
    use_full_cov = True
    completeness = True
    # Run MCMC fits for SNIa only and SNIa+AGN, for each cosmological model
    for cosmo_model in ['Flatw0waCDM', 'FlatwCDM', 'FlatLambdaCDM']:
        # print(f"Running MCMC for {cosmo_model}: SNIa only")
        # sampler_snia, _, _, _, _ = run_mcmc_pipeline(
        #                                     df_agn, df_pantheon, cosmo_model=cosmo_model, only_sna=True, 
        #                                     completeness=completeness, use_full_cov=use_full_cov,
        #                                     num_warmup=num_warmup, num_samples=num_samples)
        # results.append(extract_cosmo_results_from_sampler(sampler_snia, cosmo_model, only_sna=True, dynasty=False))
        # plot_posterior_corner(sampler_snia, cosmo_model=cosmo_model, only_sna=True)
        # plot_traces(sampler_snia, only_sna=True, cosmo_model=cosmo_model, show=False, dynasty=False)

        print(f"Running MCMC for {cosmo_model}: SNIa + AGN")
        sampler_joint, model_labels, mag_corr, logZ, logZerr = run_mcmc_pipeline(
                                                    df_agn, df_pantheon, cosmo_model=cosmo_model, only_sna=False, 
                                                    completeness=completeness, use_full_cov=use_full_cov,
                                                    num_warmup=num_warmup, num_samples=num_samples)
        results.append(extract_cosmo_results_from_sampler(sampler_joint, cosmo_model, only_sna=False, dynasty=False))
        plot_posterior_corner(sampler_joint, cosmo_model=cosmo_model, only_sna=False)
        plot_traces(sampler_joint, only_sna=False, cosmo_model=cosmo_model, show=False, dynasty=False)
        
        # Plot results
        print("Plotting Hubble diagram...")
        plot_hubble(sampler_joint, df_agn, df_pantheon, cosmo_model=cosmo_model, completeness=True, show_uncorrected=False)
        plot_hubble(sampler_joint, df_agn, df_pantheon, cosmo_model=cosmo_model, completeness=True, show_uncorrected=True)
        
        print("Plotting cosmological posteriors corner plot...")
        plot_cosmo_corner(sampler_joint, sampler_joint, cosmo_model=cosmo_model)
        print("Plotting AGN M_i predictions vs actual...")
        plot_predicted_vs_actual_Mi(sampler_joint, df_agn, cosmo_model=cosmo_model)
        print("Plotting completeness vs magnitude at redshifts...")
        p_detect, mag_centers, z_centers, dm, dz = get_completeness_function_2d(df_agn)
        plot_completeness_vs_mag_at_redshifts(p_detect, mag_centers, z_centers)

        print("Plotting AGN predicted sigma hat vs luminosity...")
        plot_predicted_sigma_hat_vs_luminosity(sampler_joint, df_agn, cosmo_model=cosmo_model, show=True)

        print(f"Finished plots for {cosmo_model}\n")
        break
    latex_code = generate_cosmo_table_latex(results)
    print("All analyses complete. Results saved to 'plots/hubble' directory.")



def test():
    # Load data
    global Cov_inv, logdetCov, L
    #df_agn, df_pantheon, Cov_inv, logdetCov, L = load_data("data/N20_w500_grace/may21_joint_fits_N20_merged.h5")

    #df_agn, df_pantheon, Cov_inv, logdetCov, L = load_data("data/may23_all_merged.h5", populate_sdss=False)
    df_agn, df_pantheon, Cov_inv, logdetCov, L = load_data("data/N20_w500_grace/may21_joint_fits_N20_merged.h5", False)

    cosmo_model = 'Flatw0waCDM'
    sampler_joint, model_labels, mag_corr, logZ, logZerr = run_mcmc_pipeline(df_agn, df_pantheon, cosmo_model=cosmo_model, 
                                        only_sna=False, completeness=True, use_full_cov=True,
                                        num_warmup=50, num_samples=50)
    plot_posterior_corner(sampler_joint, cosmo_model=cosmo_model, only_sna=False)
    print("Plotting AGN M_i predictions vs actual...")
    plot_predicted_vs_actual_Mi(sampler_joint, df_agn, cosmo_model=cosmo_model)
    print("Plotting Hubble diagram...")
    plot_hubble(sampler_joint, df_agn, df_pantheon, cosmo_model=cosmo_model, show_uncorrected=True, completeness=True)
    print("Plotting cosmological posteriors corner plot...")
    plot_cosmo_corner(sampler_joint, sampler_joint, cosmo_model=cosmo_model)

    print("Plotting AGN M_i predictions vs actual...")
    plot_predicted_vs_actual_Mi(sampler_joint, df_agn, cosmo_model=cosmo_model)
    print("Plotting completeness vs magnitude at redshifts...")
    p_detect, mag_centers, z_centers, dm, dz = get_completeness_function_2d(df_agn)
    plot_completeness_vs_mag_at_redshifts(p_detect, mag_centers, z_centers)

    print("Plotting AGN predicted sigma hat vs luminosity...")
    plot_predicted_sigma_hat_vs_luminosity(sampler_joint, df_agn, cosmo_model=cosmo_model, show=False)
if __name__ == "__main__":
    main()
    #test()
    #compare_models()
