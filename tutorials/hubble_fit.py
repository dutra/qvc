import numpy as np
import pandas as pd
import emcee
from emcee.moves import DEMove, DESnookerMove
import matplotlib.pyplot as plt
from astropy.cosmology import FlatwCDM, Flatw0waCDM, FlatLambdaCDM, FlatwpwaCDM
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
import multiprocessing
from scipy.linalg import cho_solve

import matplotlib.pyplot as plt

plt.style.use('style.mplstyle')

from hubble_utils import *
from hubble_plotting import *
from hubble_model import *
import os
import yaml

# Placeholders for global data (improves speed?)
_agn_data = None
_pantheon_data = None

_sna_LogdetCov, _sna_L, _sna_Lower = None, None, None

z_agn_pivot = 1.4

# --- Log-likelihood ---
def log_likelihood(theta, cosmo_model,
                   completeness_params,
                   only_sna=False, use_full_cov=False,
                   return_params=False):
    
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    model_priors = {key: priors[key] for key in model_labels}
    params = dict(zip(model_labels, theta))

    for key, (low, high) in model_priors.items():
        if not (low < params[key] < high):
            return -np.inf

    # Cosmology
    if cosmo_model == 'FlatwCDM':
        cosmo = FlatwCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'])
    elif cosmo_model == 'Flatw0waCDM':
        a_pivot = 1 / (1 + z_agn_pivot)
        wp = params['w0'] + (1 - a_pivot) * params['wa']
        cosmo = FlatwpwaCDM(H0=params['H0'], Om0=params['Om0'], wp=wp, wa=params['wa'], zp=z_agn_pivot)
        #cosmo = Flatw0waCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'], wa=params['wa'])
    elif cosmo_model == 'FlatLambdaCDM':
        cosmo = FlatLambdaCDM(H0=params['H0'], Om0=params['Om0'])


    # SN model: compute host mass correction
    delta_host = params['gamma_sn'] * expit(-( _pantheon_data['HOST_LOGMASS'] - 10) / params['tau_Ms']) - params['gamma_sn'] / 2

    # Start with cosmological prediction
    sn_mu_model = cosmo.distmod(_pantheon_data['zHD']).value

    # Apply Cepheid distances for calibrator hosts
    mask_calib = _pantheon_data['IS_CALIBRATOR'] == 1
    sn_mu_model[mask_calib] = _pantheon_data['CEPH_DIST'][mask_calib]

    # Residuals: observed standardized SN magnitude minus theoretical prediction
    res_snia = _pantheon_data['m_b_corr'] - (sn_mu_model + params['M0_sn'] - delta_host)

    # Compute main SN likelihood (with or without covariance)
    if use_full_cov:
        global _sna_L, _sna_Lower, _sna_LogdetCov  # Ensure these are set when loading data
        quad_form = res_snia.T @ cho_solve((_sna_L, _sna_Lower), res_snia)
        ll_snia = -0.5 * quad_form - 0.5 * _sna_LogdetCov - 0.5 * len(res_snia) * np.log(2 * np.pi)
    else:
        sigma = _pantheon_data['MU_SH0ES_ERR_DIAG']
        ll_snia = np.sum(stats.norm.logpdf(res_snia, scale=sigma))

    if only_sna:
        return ll_snia

    # AGN model
    z = _agn_data['z']
    m_obs = _agn_data['apparent_mag_i']
    m_err = _agn_data['apparent_mag_i_err']
    log_sigma_hat = _agn_data['log_sigma_hat_UV']
    log_sigma_hat_err = _agn_data['log_sigma_hat_UV_err']
    log_tau_UV_RF = _agn_data['log_tau_UV_RF']
    log_tau_UV_RF_err = _agn_data['log_tau_UV_RF_err']

    mu_cosmo = cosmo.distmod(z).value
    M_pred = M_model_agn(params['M0_sn']+params['delta_M0_agn'], 
                         params['log_sigma_hat_sq_break'], 
                         params['eta_A1_agn'], params['eta_A2_agn'], 
                         params['eta_break_agn'],
                         params['beta_agn'], 
                         log_sigma_hat, log_tau_UV_RF)


    mu_pred = m_obs - (K_corr(z) - K_corr(2)) - M_pred
    M_i_pred_err = M_model_agn_err(params['M0_sn']+params['delta_M0_agn'],
                        params['log_sigma_hat_sq_break'],
                        params['eta_A1_agn'], params['eta_A2_agn'], 
                        params['eta_break_agn'],
                        params['beta_agn'],
                        log_sigma_hat, log_sigma_hat_err,
                        log_tau_UV_RF_err)
    
    mu_err = np.sqrt(
        m_err**2 +
        M_i_pred_err**2 +
        (2.5 * 0.3 * np.log10(1 + z))**2 +
        (0.055 * z)**2 +
        np.exp(2 * params['log_f'])
    )

    # we can predict uncensored magnitudes instead
    # check predict_uncensored_magnitudes function call
    if False and use_full_cov:
        # Residuals
        dmu = mu_pred - mu_cosmo

        # Diagonal variance excluding correlated PV
        sigma_diag_sq = (
            m_err**2 +
            M_i_pred_err**2 +
            (2.5 * 0.3 * np.log10(1 + z))**2 +
            np.exp(2 * params['log_f'])
        )
        D = sigma_diag_sq
        u = 0.055 * z

        # Add small floor to avoid division by zero or negative variance
        D = np.maximum(sigma_diag_sq, 1e-6)  # [mag^2], adjust if needed
        inv_D = 1.0 / D

        # Sherman–Morrison terms
        inv_D_dmu = inv_D * dmu
        inv_D_u = inv_D * u
        alpha = 1.0 + np.dot(u, inv_D_u)

        quad = np.dot(dmu, inv_D_dmu) - (np.dot(inv_D_u, dmu) ** 2) / alpha
        logdet = np.sum(np.log(D)) + np.log(alpha)

        ll_agn = -0.5 * quad - 0.5 * logdet - 0.5 * len(dmu) * np.log(2 * np.pi)
    else:
        ll_agn = np.sum(stats.norm.logpdf(mu_pred - mu_cosmo, scale=mu_err))

    # Corrected? AGN completeness correction 2D
    # Optional AGN completeness correction 2D
    m_model = M_pred + mu_cosmo #+ (K_corr(_agn_data['z']) - K_corr(2)) + mu_cosmo  # model-predicted magnitude

    ll_completeness = 0.0
    # selection bias correction during inference
    if False and completeness_params is not None:
        completeness2d, mag_centers, _, dm, _ = completeness_params

        m_grid = mag_centers
        N_obj = len(z)

        # Shape: (N_obj, N_grid)
        m_grid_broadcasted = np.tile(m_grid, (N_obj, 1))
        z_broadcasted = np.tile(z[:, None], (1, len(m_grid)))

        # Gaussian weights: P(m | model)
        gauss_weights = stats.norm.pdf(m_grid_broadcasted, loc=m_model[:, None], scale=mu_err[:, None])
        gauss_weights /= np.trapezoid(gauss_weights, m_grid, axis=1)[:, None] + 1e-12

        # Evaluate p(detect | m, z)
        p_detect = completeness2d(m_grid_broadcasted, z_broadcasted)
        p_detect = soft_clip(p_detect, floor=1e-4, sharpness=20)

        # Marginalized likelihood: ∫ P(m | model) × p_detect(m, z) dm
        integrals = np.trapezoid(gauss_weights * p_detect, m_grid, axis=1)
        integrals = np.maximum(integrals, 1e-12)

        ll_completeness = np.sum(np.log(integrals))


    return ll_snia + ll_agn + ll_completeness

# Globals used by dynesty
_dynesty_config = {}

def prior_transform_dynesty(unit_cube):
    global _dynesty_config
    priors = _dynesty_config['model_priors']
    keys = _dynesty_config['model_labels']
    return [priors[key][0] + (priors[key][1] - priors[key][0]) * x
            for x, key in zip(unit_cube, keys)]

def loglike_dynesty(theta):
    global _dynesty_config
    cfg = _dynesty_config
    return log_likelihood(theta,
                          cfg['cosmo_model'],
                          cfg['completeness_params'],
                          cfg['only_sna'],
                          cfg['use_full_cov'])

def dynesty_initializer(agn_data, pantheon_data, dynesty_config, sna_LogdetCov, sna_L, sna_Lower):
    global _agn_data, _pantheon_data, _dynesty_config
    global _sna_LogdetCov, _sna_L, _sna_Lower
    _agn_data = agn_data
    _pantheon_data = pantheon_data
    _dynesty_config = dynesty_config
    _sna_LogdetCov = sna_LogdetCov
    _sna_L = sna_L
    _sna_Lower = sna_Lower

def run_mcmc_pipeline(df_agn, df_pantheon, cosmo_model='Flatw0waCDM', 
                      only_sna=False, completeness=True, use_full_cov=False,
                      resume=False, dlogz_init=np.inf, nlive_init=25, nlive_batch=10,
                      num_samples=2000, num_warmup=2000, 
                      fitting_method=None):

    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    ndim = len(model_labels)

    # Prepare data
    df_pantheon_filtered = df_pantheon[['zHD', 'MU_SH0ES', 'MU_SH0ES_ERR_DIAG', 'CEPH_DIST', 'IS_CALIBRATOR',
                                        'm_b_corr', 'x1', 'c', 'biasCor_m_b', 'HOST_LOGMASS']].copy()
    df_agn_filtered = df_agn[['z', 'apparent_mag_i', 'apparent_mag_i_err', 'M_i',
                              'log_sigma_hat_UV', 'log_sigma_hat_UV_err', 'log_tau_UV_RF', 'log_tau_UV_RF_err']].copy()

    if completeness:
        completeness_params = get_completeness_function_2d(df_agn_filtered)
   
        # Optionally load initial parameter values from file if available
        # initial_params = None
        # params_yaml_path = f"data/median_params_{cosmo_model}.yaml"
        # if os.path.exists(params_yaml_path):
        #     print(f"Loading initial parameter values from {params_yaml_path}")
        #     with open(params_yaml_path, "r") as f:
        #         initial_params = yaml.safe_load(f)
        #         print(initial_params)
        #     if not set(model_labels).issubset(initial_params.keys()):
        #         raise ValueError(f"YAML file does not contain all required model labels: {model_labels}")
        # if initial_params is not None:
        #     print("Applying forward completeness correction to AGN apparent magnitudes before fitting...")
        #     mag_corr = apply_forward_completeness_correction(df_agn_filtered, initial_params, cosmo_model, completeness_params)
        #     df_agn.loc[df_agn_filtered.index, 'apparent_mag_i'] = mag_corr.astype(np.float32)

    z_pivot = (1 / np.exp(np.mean(np.log(1 / (1 + df_agn_filtered['z']))))) - 1
    #print(f"Log sigma hat pivot: {log_sigma_hat_pivot:.3f}, log tau UV RF pivot: {df_agn_filtered['log_tau_UV_RF'].median():.3f}")
    print("log tau UV RF pivot: ", np.average(df_agn_filtered['log_tau_UV_RF'], weights=1 / df_agn_filtered['log_tau_UV_RF_err']**2))
    print(f"z mean: {df_agn_filtered['z'].mean():.3f},  z_agn_pivot: {z_pivot:.3f}")

    print(f"Mean AGN M_i: {df_agn_filtered['M_i'].mean()}, delta_M_agn ~ {df_agn_filtered['M_i'].mean()-(-19.3):.3f}")


    global _agn_data, _pantheon_data
    _agn_data = {col: df_agn_filtered[col].values for col in df_agn_filtered.columns}
    _pantheon_data = {col: df_pantheon_filtered[col].values for col in df_pantheon_filtered.columns}

    if fitting_method == 'dynesty':
        # Set dynesty global context
        global _dynesty_config
        _dynesty_config.update({
            'model_priors': priors,
            'model_labels': model_labels,
            'cosmo_model': cosmo_model,
            'completeness_params': completeness_params,
            'only_sna': only_sna,
            'use_full_cov': use_full_cov,
        })


        num_cpus = multiprocessing.cpu_count()
        with multiprocessing.get_context("spawn").Pool(
            processes=num_cpus,
            initializer=dynesty_initializer,
            initargs=(_agn_data, _pantheon_data, _dynesty_config, 
                      _sna_LogdetCov, _sna_L, _sna_Lower)
        ) as pool:            
            # use NestedSampler for precise log-evidence estimates (e.g., model selection)
            # use DynamicNestSampler for Cosmological parameter inference
            u = np.random.rand(1000, ndim)
            v = pool.map(prior_transform_dynesty, u)
            l = pool.map(loglike_dynesty, v)
            print("Fraction of finite likelihoods:", np.sum(np.isfinite(l)) / len(l))

            if resume:
                sampler = DynamicNestedSampler.restore(f'data/dynesty_{cosmo_model}.save', pool=pool)
            else:
                sampler = DynamicNestedSampler(
                    loglike_dynesty,
                    prior_transform_dynesty,
                    ndim,
                    update_interval=10*ndim,
                    bound='single',
                    sample='rwalk',
                    pool=pool,
                    queue_size=num_cpus
                )
            sampler.run_nested(
                resume=resume,
                checkpoint_file=f'data/dynesty_{cosmo_model}.save',
                print_progress=True,
                dlogz_init=dlogz_init,
                n_effective=10,               
                nlive_init=5 * ndim,         
                nlive_batch=2 * ndim  # 2 * ndim is low, but seems to work
            )

        results = sampler.results
        logZ, logZerr = results.logz[-1], results.logzerr[-1]
        print(f"\nBayesian evidence logZ = {logZ:.2f} ± {logZerr:.2f}")
        if logZerr > 1:
            print("Warning: logZ error is large, consider increasing nlive or maxiter.")

        unweighted_samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
        flat_samples = resample_equal(unweighted_samples, weights)
        median_samples = np.median(flat_samples, axis=0)

        print("Dynesty results stats:")
        print("  samples shape:", unweighted_samples.shape)
        print("  weights max:", np.max(weights))
        print("  effective samples:", np.sum(weights)**2 / np.sum(weights**2))
        display_results_summary(flat_samples, cosmo_model)
        plot_dynesty(results, cosmo_model)

    elif fitting_method == 'emcee':
        # --- emcee logic ---
        print("Number of parameters:", ndim)
        nwalkers = ndim * 2 * 10 #* 15
        initial_pos = np.array([
            np.random.uniform(low, high, nwalkers) for low, high in model_priors.values()
        ]).T

        num_cpus = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_cpus) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood,
                                            args=(cosmo_model,
                                                    df_agn_filtered, df_pantheon_filtered,
                                                    completeness_params,
                                                    only_sna, use_full_cov),
                                            moves=[(DEMove(), 0.8), (DESnookerMove(), 0.2)],
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
    else:
        raise ValueError("fitting_method must be 'emcee', 'dynesty', or 'ultranest'")

    params = dict(zip(model_labels, median_samples))

    # Save median parameter values to file
    # Convert all parameter values to plain Python floats before saving
    with open(f"data/median_params_{cosmo_model}.yaml", "w") as f:
        yaml.dump( {k: float(v) for k, v in params.items()}, f)

    completeness_params = get_completeness_function_2d(df_agn_filtered)
    mag_corr = apply_forward_completeness_correction(df_agn_filtered, params, cosmo_model, completeness_params)
    df_agn.loc[df_agn_filtered.index, 'apparent_mag_i_corr'] = mag_corr.astype(np.float32)

    #display_diagnostics(sampler, cosmo_model, fitting_method=fitting_method)
    np.save(f"data/flat_samples_{cosmo_model}_{'sna' if only_sna else 'agn'}.npy", flat_samples)
    return sampler, flat_samples, model_labels, mag_corr, logZ, logZerr

def compare_models():
    priors, model_labels = get_model_params('Flatw0waCDM')
    ndim = len(priors.keys())
    nlive = 25 * ndim # basic
    # nlive = 50 * ndim # modeerate precision
    # nlive = 100 * ndim # high precision
    #maxiter = 2000 # TESTING
    maxiter = None # full run
    #n_effective = 100 # Testing
    n_effective = 2000 # moderate quality
    #n_effective = 5000 # high quality
    #n_effective = 10000 # publication quality
    
    use_full_cov = True
    completeness = True

    global _sna_LogdetCov, _sna_L, _sna_Lower
    df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/N20_w500_grace/may21_joint_fits_N20_merged.h5")

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
    #df_agn, df_pantheon, Cov_inv, logdetCov, L = load_data("data/may23_all_merged.h5", populate_sdss=True)
    #df_agn, df_pantheon, Cov_inv, logdetCov, L = load_data("data/N20_w500_grace/may21_joint_fits_N20_merged.h5", False)
    df_agn, df_pantheon, Cov_inv, logdetCov, L = load_data("data/june1_joint_N20w2000s1000_fits_merged.h5")
    print(len(df_pantheon))
    
    results = []

    num_warmup, num_samples = 250, 100
    use_full_cov = True
    completeness = True

    # Run MCMC fits for SNIa only and SNIa+AGN, for each cosmological model
    for cosmo_model in ['Flatw0waCDM', 'FlatwCDM']:
        print(f"Running MCMC for {cosmo_model}: SNIa + AGN")
        sampler_joint, model_labels, mag_corr, logZ, logZerr = run_mcmc_pipeline(
                                                    df_agn, df_pantheon, cosmo_model=cosmo_model, only_sna=False, 
                                                    completeness=completeness, use_full_cov=use_full_cov,
                                                    num_warmup=num_warmup, num_samples=num_samples)
        results.append(extract_cosmo_results_from_sampler(sampler_joint, cosmo_model, only_sna=False, dynasty=False))
        plot_posterior_corner(sampler_joint, cosmo_model=cosmo_model, only_sna=False)
        plot_traces(sampler_joint, only_sna=False, cosmo_model=cosmo_model, show=False, dynasty=False)
        plot_cosmo_corner(sampler_joint, sampler_joint, cosmo_model=cosmo_model)

        print(f"Running MCMC for {cosmo_model}: SNIa only")
        sampler_snia, _, _, _, _ = run_mcmc_pipeline(
                                            df_agn, df_pantheon, cosmo_model=cosmo_model, only_sna=True, 
                                            completeness=completeness, use_full_cov=use_full_cov,
                                            num_warmup=num_warmup, num_samples=num_samples)
        results.append(extract_cosmo_results_from_sampler(sampler_snia, cosmo_model, only_sna=True, dynasty=False))
        plot_posterior_corner(sampler_snia, cosmo_model=cosmo_model, only_sna=True)
        plot_traces(sampler_snia, only_sna=True, cosmo_model=cosmo_model, show=False, dynasty=False)


        # Plot results
        print("Plotting Hubble diagram...")
        #plot_hubble(sampler_joint, df_agn, df_pantheon, cosmo_model=cosmo_model, completeness=True, show_uncorrected=False)
        plot_hubble(sampler_joint, df_agn, df_pantheon, cosmo_model=cosmo_model, completeness=True, show_uncorrected=True, show_true=True)
        
        print("Plotting cosmological posteriors corner plot...")
        plot_cosmo_corner(sampler_snia, sampler_joint, cosmo_model=cosmo_model)
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
    cosmo_model = 'Flatw0waCDM'

    # Load data
    global _sna_LogdetCov, _sna_L, _sna_Lower

    #df_agn, df_pantheon, Cov_inv, logdetCov, L = load_data("data/N20_w500_grace/may21_joint_fits_N20_merged.h5")
    #df_agn, df_pantheon, Cov_inv, logdetCov, L = load_data("data/may23_all_merged.h5", populate_sdss=False)
    df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/june1_joint_N20w2000s1000_fits_merged.h5")
    fitting_method = 'dynesty'

    sampler_joint, flat_samples, model_labels, mag_corr, logZ, logZerr = run_mcmc_pipeline(df_agn, df_pantheon, cosmo_model=cosmo_model, 
                                        only_sna=False, completeness=True, use_full_cov=True,
                                        fitting_method=fitting_method,
                                        num_warmup=8000, num_samples=1000, resume=False)
    try:
        plot_posterior_corner(flat_samples, cosmo_model=cosmo_model, only_sna=False)
    except Exception as e:
        print(f"Could not plot posterior corner: {e}")
        #plot_traces(sampler_joint, only_sna=False, cosmo_model=cosmo_model, show=False, use_dynesty=use_dynesty)
    print("Plotting AGN predicted sigma hat vs luminosity...")
    plot_Mi_vs_log_sigma_hat_sq(flat_samples, df_agn, cosmo_model=cosmo_model, show=False)
    
    print("Plotting AGN M_i predictions vs actual...")
    plot_predicted_vs_actual_Mi(flat_samples, df_agn, cosmo_model=cosmo_model)
    
    print("Plotting Hubble diagram...")
    plot_hubble(flat_samples, df_agn, df_pantheon, cosmo_model=cosmo_model, show_uncorrected=False, completeness=False)
    
    #print("Plotting cosmological posteriors corner plot...")
    #plot_cosmo_corner(sampler_joint, sampler_joint, cosmo_model=cosmo_model)

    
    print("Plotting completeness vs magnitude at redshifts...")
    p_detect, mag_centers, z_centers, dm, dz = get_completeness_function_2d(df_agn)
    plot_completeness_vs_mag_at_redshifts(p_detect, mag_centers, z_centers)
    plot_completeness_diagnostics(df_agn, p_detect, mag_centers, z_centers)

    #plot_predicted_sigma_hat_vs_luminosity(sampler_joint, df_agn, cosmo_model=cosmo_model, show=False)
if __name__ == "__main__":
    #main()
    test()
    #compare_models()
