import numpy as np
import pandas as pd
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

z_agn_pivot = 1.7

# --- Log-likelihood ---
def log_likelihood(theta, cosmo_model,
                   completeness_params,
                   only_sna=False, use_full_cov=False,
                   return_params=False):
    
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    model_priors = {key: priors[key] for key in model_labels}
    params = dict(zip(model_labels, theta))

    for key, (low, high) in model_priors.items():
        if low > high:
            raise ValueError(f"For key {key} prior: Low {low} > high {high}")
        # Check if parameter is within prior bounds 
        if not (low < params[key] < high):
            return -np.inf

    # Cosmology
    if cosmo_model == 'FlatwCDM':
        cosmo = FlatwCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'])
    elif cosmo_model == 'Flatw0waCDM':
        #a_pivot = 1 / (1 + z_agn_pivot)
        #wp = params['w0'] + (1 - a_pivot) * params['wa']
        cosmo = FlatwpwaCDM(H0=params['H0'], Om0=params['Om0'], wp=params['wp'], wa=params['wa'], zp=z_agn_pivot)
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
    m_obs = _agn_data['apparent_mag_2500']
    m_err = _agn_data['apparent_mag_2500_err']
    log_sigma0 = _agn_data['log_sigma0']
    log_sigma0_err = _agn_data['log_sigma0_err']
    log_tau_UV_RF = _agn_data['log_tau_UV_RF']
    log_tau_UV_RF_err = _agn_data['log_tau_UV_RF_err']
    f_host = _agn_data['f_host']
    f_host_err = _agn_data['f_host_err']

    mu_cosmo = cosmo.distmod(z).value
    M_pred = M_model_agn(params['M0_agn'], 
                         params['alpha_agn'], params['beta_agn'],
                         log_sigma0, log_tau_UV_RF,
                         f_host)
    

    mu_pred = m_obs - M_pred 
    M_i_pred_err = M_model_agn_err(params['M0_agn'],
                         params['alpha_agn'], params['beta_agn'],
                        log_sigma0, log_sigma0_err,
                        log_tau_UV_RF_err,
                        f_host_err)
    
    mu_err = np.sqrt(
        m_err**2 +
        M_i_pred_err**2 +
        # (2.5 * 0.3 * np.log10(1 + z))**2 +
        # (2.5 * alpha_nu_err * np.log10(1 + z))**2 +
        (0.055 * z)**2 +
        np.exp(2 * params['log_f'])
    )

    ll_agn = np.sum(stats.norm.logpdf(mu_pred - mu_cosmo, scale=mu_err))

    # Corrected? AGN completeness correction 2D
    # Optional AGN completeness correction 2D
    m_model = M_pred + mu_cosmo  # model-predicted magnitude

    ll_completeness = 0.0
    # selection bias correction during inference
    if completeness_params is not None:
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
        p_detect = soft_clip(p_detect, floor=1e-12, sharpness=10)

        # Marginalized likelihood: ∫ P(m | model) × p_detect(m, z) dm
        integrals = np.trapezoid(gauss_weights * p_detect, m_grid, axis=1)
        integrals = np.maximum(integrals, 1e-12)

        ll_completeness = np.sum(np.log(integrals))

    # print(f"Log-likelihood components: ll_snia={ll_snia:.2f}, ll_agn={ll_agn:.2f}, ll_completeness={ll_completeness:.2f}")
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
                      resume=False, dlogz_init=np.inf, nlive_init=25, nlive_batch=10):

    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    ndim = len(model_labels)

    # Prepare data
    df_pantheon_filtered = df_pantheon[['zHD', 'MU_SH0ES', 'MU_SH0ES_ERR_DIAG', 'CEPH_DIST', 'IS_CALIBRATOR',
                                        'm_b_corr', 'x1', 'c', 'biasCor_m_b', 'HOST_LOGMASS']].copy()
    df_agn_filtered = df_agn[['z', 'apparent_mag_2500', 'apparent_mag_2500_err', 'apparent_mag_i',
                              'log_sigma0', 'log_sigma0_err', 'log_tau_UV_RF', 'log_tau_UV_RF_err',
                              'f_host', 'f_host_err',
                              ]].copy()

    if completeness:
        completeness_params = get_completeness_function_2d(df_agn_filtered)
    else:
        completeness_params = None

    z_pivot = (1 / np.exp(np.mean(np.log(1 / (1 + df_agn_filtered['z']))))) - 1
    #print(f"Log sigma hat pivot: {log_sigma0_pivot:.3f}, log tau UV RF pivot: {df_agn_filtered['log_tau_UV_RF'].median():.3f}")
    print("log tau UV RF mean: ", np.average(df_agn_filtered['log_tau_UV_RF']))
    print("log tau UV RF pivot: ", np.average(df_agn_filtered['log_tau_UV_RF'], weights=1 / df_agn_filtered['log_tau_UV_RF_err']**2))
    
    print("log sigma0 mean: ", np.average(df_agn_filtered['log_sigma0']))
    print("log sigma0 pivot: ", np.average(df_agn_filtered['log_sigma0'], weights=1 / (df_agn_filtered['log_sigma0_err'])**2))
    
    print(f"z mean: {df_agn_filtered['z'].mean():.3f},  z_agn_pivot: {z_pivot:.3f}")

    #print(f"Mean AGN M: {df_agn_filtered['M_2500'].mean()}, delta_M_agn ~ {df_agn_filtered['M_2500'].mean()-(-19.3):.3f}")


    global _agn_data, _pantheon_data
    _agn_data = {col: df_agn_filtered[col].values for col in df_agn_filtered.columns}
    _pantheon_data = {col: df_pantheon_filtered[col].values for col in df_pantheon_filtered.columns}

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
        if resume:
            sampler = DynamicNestedSampler.restore(f'data/dynesty_{cosmo_model}.save', pool=pool)
        else:
            print("Testing likelihood and prior transform with random samples...")
            u = np.random.rand(10, ndim)
            v = pool.map(prior_transform_dynesty, u)
            l = pool.map(loglike_dynesty, v)
            print("Fraction of finite likelihoods:", np.sum(np.isfinite(l)) / len(l))

            sampler = DynamicNestedSampler(
                loglike_dynesty,
                prior_transform_dynesty,
                ndim,
                update_interval=10*ndim,
                bound='multi',
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
            nlive_init=20 * ndim,         
            nlive_batch=10 * ndim  # 2 * ndim is low, but seems to work
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
    display_results_summary(flat_samples, cosmo_model, z_agn_pivot)
    plot_dynesty(results, cosmo_model)

    params = dict(zip(model_labels, median_samples))

    # Save median parameter values to file
    # Convert all parameter values to plain Python floats before saving
    with open(f"data/median_params_{cosmo_model}.yaml", "w") as f:
        yaml.dump( {k: float(v) for k, v in params.items()}, f)

    #completeness_params = get_completeness_function_2d(df_agn_filtered)
    #mag_corr = apply_forward_completeness_correction(df_agn_filtered, params, cosmo_model, completeness_params)
    #df_agn.loc[df_agn_filtered.index, 'apparent_mag_2500_corr'] = mag_corr.astype(np.float32)
    mag_corr = None

    #z_pivot_best, _, _ = find_optimal_pivot(flat_samples, cosmo_model, df_agn_filtered)
    #print(f"Optimal z pivot for {cosmo_model}: {z_pivot_best:.3f}")

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
    global _sna_LogdetCov, _sna_L, _sna_Lower
    #df_agn, df_pantheon, Cov_inv, logdetCov, L = load_data("data/may23_all_merged.h5", populate_sdss=True)
    #df_agn, df_pantheon, Cov_inv, logdetCov, L = load_data("data/N20_w500_grace/may21_joint_fits_N20_merged.h5", False)
    df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("results/aug10_stonebwb_N20w2000s1000t6c4.h5", populate_sdss=False)
 
    print(len(df_pantheon))
    
    results = []

    use_full_cov = True
    completeness = True

    # Run MCMC fits for SNIa only and SNIa+AGN, for each cosmological model
    for cosmo_model in ['Flatw0waCDM']:#, 'FlatwCDM']:
        print(f"Running MCMC for {cosmo_model}: SNIa + AGN")
        sampler_joint, flat_samples_joint, model_labels, mag_corr, logZ, logZerr = run_mcmc_pipeline(df_agn, df_pantheon, cosmo_model=cosmo_model, 
                                            only_sna=False, completeness=True, use_full_cov=True,
                                            resume=False)
        
        #results.append(extract_cosmo_results_from_sampler(sampler_joint, cosmo_model, only_sna=False, dynasty=False))

        plot_predicted_vs_actual_M2500(flat_samples_joint, df_agn, cosmo_model=cosmo_model, z_agn_pivot=z_agn_pivot, show=False)
        
        print("Plotting Hubble diagram...")
        residuals, mu_pred_median, mu_pred_std = plot_hubble(flat_samples_joint, df_agn, df_pantheon, 
                                                            cosmo_model=cosmo_model, z_agn_pivot=z_agn_pivot, show_true=False, show=False)
        #plot_predicted_L2500_vs_sigmahat(flat_samples_joint, df_agn, cosmo_model=cosmo_model, z_agn_pivot=z_agn_pivot, show=False)
        #plot_full_residuals(df_agn, residuals, flat_samples_joint, cosmo_model, z_agn_pivot, show=False)
        
        #plot_posterior_corner(sampler_joint, cosmo_model=cosmo_model, only_sna=False)
        #plot_traces(sampler_joint, only_sna=False, cosmo_model=cosmo_model, show=False, dynasty=False)
        #plot_cosmo_corner(sampler_joint, sampler_joint, cosmo_model=cosmo_model)

        print(f"Running MCMC for {cosmo_model}: SNIa only")
        sampler_snia, flat_samples_snia, _, _, _, _ = run_mcmc_pipeline(df_agn, df_pantheon, cosmo_model=cosmo_model, 
                                            only_sna=True, completeness=True, use_full_cov=True,
                                            resume=False)
        
        plot_cosmo_corner(flat_samples_snia, flat_samples_joint, cosmo_model, z_agn_pivot, show=False)
        print(f"Finished plots for {cosmo_model}\n")

    # latex_code = generate_cosmo_table_latex(results)
    # print("All analyses complete. Results saved to 'plots/hubble' directory.")



def test():
    cosmo_model = 'Flatw0waCDM'
    #cosmo_model = 'FlatwCDM'
    # cosmo_model = 'FlatLambdaCDM'
    only_sna = False

    # Load data
    global _sna_LogdetCov, _sna_L, _sna_Lower

    #df_agn, df_pantheon, Cov_inv, logdetCov, L = load_data("data/N20_w500_grace/may21_joint_fits_N20_merged.h5")
    #df_agn, df_pantheon, Cov_inv, logdetCov, L = load_data("data/may23_all_merged.h5", populate_sdss=False)
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/june1_joint_N20w2000s1000_fits_merged.h5")
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/june30_joint_allebv005_N20w500s250_merged.h5")
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/july5_joint_chisq_lcrf2500_mean1_N20w2000s1000_merged.h5")

    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/july12_joint_mean1_zsort_N20w2000s500_merged.h5")
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/july14_chi10_extinction04_ebv005_N20w500s250_merged.h5")
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/july8_joint_chisq_lcrf_exactly2000_N20w500s250_merged.h5")
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/july15_chi2_otherfilters_N30w500s250_merged.h5")
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/july15_nochi2_filters_N30w250s100_merged.h5")
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/july16_nochi2_filters_N30w500s250_merged.h5")
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/july17_goodsources_chisq5and10_N30w250s100_merged.h5")
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/july17_goodsources_chisq5and10_mean01_N30w250s100_merged.h5")
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/july17_goodsources_chisq5and10_mean01_N20w4000s500_merged.h5")
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/july18_goodsources_chisq5and10_mean1_N20w4000s500_merged.h5")
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/july19_goodsources_chisq2_otherfilters_mean1_N20w4000s500_merged.h5")
    
    
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/july19_goodsources_chisq10_otherfilters_mean1_N20w4000s500_merged.h5")
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/july19_goodsources_chisq5and10_mean1_N20w4000s500_merged.h5", populate_sdss=True)
    
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/july20_goodsources_chisq5and10_hostpl_N20w4000s500_merged.h5")
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/july21_chisq2_hostpl_N20w4000s200_merged.h5")
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/july27_chisq2_preview_tree8_N20w4000s500_merged.h5", populate_sdss=True)
    
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/july31_chisq2_preview_tree8_N10w1000s250_merged.h5", populate_sdss=False)
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/july31_chisq2_preview_grae_tree8_N10w1000s250_grace_merged.h5", populate_sdss=False)
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/aug1_fhostshen_wred_N10t8w2000s500_merged.h5", populate_sdss=False)
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/aug1_wored_N10t8w2000s500_merged.h5", populate_sdss=False)
    
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/aug4_fhostshen11_N10t6w1000s500_merged.h5", populate_sdss=False)
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/aug4_fhostzero_N10t6w1000s500_merged.h5", populate_sdss=False)
    
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/aug5_fshen11_N20t6w4000s500_merged.h5", populate_sdss=False)
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/aug5_fshen11_fhostnocap_N10t6w1000s500_merged.h5", populate_sdss=False)

    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/aug6_fshen11_fhostnocap_bplzero_N10t6w1000s500_merged.h5", populate_sdss=False)
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/aug6_fshen11_fhostnocap_bplzero_N10t6w2000s500_merged.h5", populate_sdss=False)
    
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("results/aug9_stone_N10w1000s500t6c4_merged.h5", populate_sdss=True)
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("results/aug9_stone_bwb_N10w1000s500t6c4_merged.h5", populate_sdss=True)
    
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("results/aug10_stonebwb_N20w2000s1000t6c4.h5", populate_sdss=False)
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("results/aug10_stonebwb_N30w2000s1000t6c4.h5", populate_sdss=False)

    df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("results/data/aug12_chisq_qscpu_N20w4000s1000t8c4.h5", populate_sdss=False)

    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/july19_goodsources_chisq5and10_mean1_N20w4000s500_merged.h5", populate_sdss=True)
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data("data/june1_joint_N20w2000s1000_fits_merged.h5")
    #df_agn = df_agn[:400]
    #df_agn = df_agn[df_agn['z'] > 1]  # Filter AGN data to z < 2.5

    sampler_joint, flat_samples, model_labels, mag_corr, logZ, logZerr = run_mcmc_pipeline(df_agn, df_pantheon, cosmo_model=cosmo_model, 
                                        only_sna=only_sna, completeness=True, use_full_cov=True,
                                        resume=False)
    if cosmo_model == 'Flatw0waCDM':
        zp = compute_pivot_redshift(flat_samples, cosmo_model)
        print("Pivot redshift: ", zp)
    # try:
    #     plot_posterior_corner(flat_samples, cosmo_model=cosmo_model, only_sna=False, show=False)
    # except Exception as e:
    #     print(f"Could not plot posterior corner: {e}")
    
    #plot_traces(sampler_joint, only_sna=False, cosmo_model=cosmo_model, show=False, use_dynesty=use_dynesty)
    #print("Plotting AGN predicted sigma hat vs luminosity...")
    #plot_Mi_vs_log_sigma0(flat_samples, df_agn, cosmo_model=cosmo_model, show=False)
    
    #print("Plotting AGN M_i predictions vs actual...")
    plot_predicted_vs_actual_M2500(flat_samples, df_agn, cosmo_model=cosmo_model, z_agn_pivot=z_agn_pivot, show=False)
    #plot_predicted_vs_actual_Mi(flat_samples, df_agn, cosmo_model=cosmo_model, show=False)
    
    print("Plotting Hubble diagram...")
    residuals, mu_pred_median, mu_pred_std = plot_hubble(flat_samples, df_agn, df_pantheon, 
                                                         cosmo_model=cosmo_model, z_agn_pivot=z_agn_pivot, show_true=False, show=False)
    
    #plot_Mi_vs_sigmahat(df_agn, cosmo_model=cosmo_model, show=False)
    #plot_predicted_M2500_vs_sigmahat(flat_samples, df_agn, cosmo_model=cosmo_model, show=False)
    plot_predicted_L2500_vs_sigmahat(flat_samples, df_agn, cosmo_model=cosmo_model, z_agn_pivot=z_agn_pivot, show=False)
    #plot_inverted_sigmahat_vs_l2500_pl(flat_samples, df_agn, cosmo_model=cosmo_model, show=False)
    #print("Plotting cosmological posteriors corner plot...")
    plot_cosmo_corner(None, flat_samples, cosmo_model, z_agn_pivot, show=False)

    
    print("Plotting completeness vs magnitude at redshifts...")
    #p_detect, mag_centers, z_centers, dm, dz = get_completeness_function_2d(df_agn)
    #plot_completeness_vs_mag_at_redshifts(p_detect, mag_centers, z_centers)
    #plot_completeness_diagnostics(df_agn, p_detect, mag_centers, z_centers)

    print("Plotting residuals...")
    plot_full_residuals(df_agn, residuals, flat_samples, cosmo_model, z_agn_pivot, show=False)

    # Example usage:
    # Assuming `samples` is a dict from your MCMC run
    if cosmo_model == 'Flatw0waCDM':
        rho_w0_wa = posterior_corr(flat_samples, cosmo_model, z_agn_pivot)
        print(f"Posterior correlation coefficient (w0, wa) at z_p={z_agn_pivot}: {rho_w0_wa:.3f}")


    #plot_predicted_sigma_hat_vs_luminosity(sampler_joint, df_agn, cosmo_model=cosmo_model, show=False)
if __name__ == "__main__":
    #main()
    test()
    #compare_models()
