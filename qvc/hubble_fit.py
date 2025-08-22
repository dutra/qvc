import os

num_cores = os.environ.get("NUM_CORES", os.cpu_count()-2)

try:
    num_cores = int(num_cores)
except ValueError:
    print(f"Invalid NUM_CORES value '{num_cores}', ignoring.")
    num_cores = os.cpu_count()-2

print(f"CPU Num Cores: {num_cores}")
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_cores}"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

prefix = os.environ.get("PREFIX", "")

import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatwCDM, Flatw0waCDM, FlatLambdaCDM, FlatwpwaCDM
from scipy import stats
from scipy.signal import fftconvolve
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from dynesty import DynamicNestedSampler
from dynesty.utils import resample_equal
import multiprocessing
from scipy.linalg import cho_solve
from dynesty import utils as dyfunc

import matplotlib.pyplot as plt

plt.style.use('style.mplstyle')

from hubble_utils import *
from hubble_plotting import *
from hubble_model import *
from hubble_completeness import *
import argparse
from scipy.interpolate import interp1d

# Placeholders for global data (improves speed?)
_agn_data = None
_pantheon_data = None

_sna_LogdetCov, _sna_L, _sna_Lower = None, None, None

z_pivot_sna = 0.0
z_pivot_agn = 1.5


def completeness_loglike(m_model, mu_err, z, completeness2d, m_grid, sigma_completeness, tiny=1e-300):
    """
    m_model : array (N_obj,) model-predicted apparent magnitudes
    mu_err  : array (N_obj,) Gaussian sigma for each magnitude
    z       : array (N_obj,) redshifts
    m_grid  : array (N_grid,) magnitude grid (e.g., the map's mag_centers)
    """
    m_grid = np.asarray(m_grid)
    z      = np.asarray(z)
    m_model = np.asarray(m_model)
    mu_err  = np.asarray(mu_err)

    # Gaussian *pdf* over the real line, evaluated on m_grid
    # Do NOT renormalize row-wise over m_grid.
    sigma = np.maximum(mu_err, 1e-9)  # avoid zero-sigma
    pdf = stats.norm.pdf(m_grid[None, :],
        loc=m_model[:, None],
        scale=np.sqrt(sigma[:, None]**2 + sigma_completeness**2)) # If not adding scatter to mags_true
        #scale=sigma[:, None])

    # p_detect(m, z)
    p_det = completeness2d(m_grid[None, :], z[:, None])  # shape (N_obj, N_grid)
    wpdf = pdf * p_det

    # ∫ pdf(m) * p_det(m, z) dm  (outside-grid p_det=0 by construction)
    integrals = np.trapz(wpdf, m_grid, axis=1)
    integrals = np.clip(integrals, tiny, 1.0)            # numerical guard

    # Average
    m_integrals = np.trapz(wpdf * m_grid[None, :], m_grid, axis=1)
    m_integrals = np.clip(m_integrals, tiny, None)        # numerical guard (can be > 1; units=mag)
    dmi = m_integrals / integrals - m_model

    return np.sum(np.log(integrals)), (integrals, dmi)

# --- Log-likelihood ---
def log_likelihood(theta, cosmo_model,
                   completeness_params,
                   only_sna=False, use_full_cov=False,
                   return_params=False):
    
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model, only_sna=only_sna)
    model_priors = {key: priors[key] for key in model_labels}
    params = dict(zip(model_labels, theta))

    for key, (low, high) in model_priors.items():
        if low > high:
            raise ValueError(f"For key {key} prior: Low {low} > high {high}")
        # Check if parameter is within prior bounds 
        if not (low < params[key] < high):
            return -np.inf, np.array([])  # Return -inf log-likelihood and zero blobs

    # Cosmology
    if cosmo_model == 'FlatwCDM':
        cosmo = FlatwCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'])
    elif cosmo_model == 'Flatw0waCDM':
        #a_pivot = 1 / (1 + z_pivot_agn)
        #wp = params['w0'] + (1 - a_pivot) * params['wa']
        z_pivot = z_pivot_sna if only_sna else z_pivot_agn
        cosmo = FlatwpwaCDM(H0=params['H0'], Om0=params['Om0'], wp=params['wp'], wa=params['wa'], zp=z_pivot)
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

    z = _agn_data['z']
    
    if only_sna:
        return ll_snia, np.array([])

    # AGN model
    m_obs = _agn_data['apparent_mag_2500']
    m_err = _agn_data['apparent_mag_2500_err']
    log_sigma_UV = _agn_data['log_sigma_UV']
    log_sigma_UV_err = _agn_data['log_sigma_UV_err']
    log_tau_UV_RF = _agn_data['log_tau_UV_RF']
    log_tau_UV_RF_err = _agn_data['log_tau_UV_RF_err']
    bwb_beta = _agn_data['bwb_beta']
    bwb_beta_err = _agn_data['bwb_beta_err']

    mu_cosmo = cosmo.distmod(z).value
    M_pred = M_model_agn(params['M0_agn'], 
                         params['alpha_agn'],
                         params['beta_agn'],
                         params['gamma_agn'],
                         log_sigma_UV, log_tau_UV_RF,
                         bwb_beta)
    

    mu_pred = m_obs - M_pred 
    M_i_pred_err = M_model_agn_err(params['M0_agn'],
                         params['alpha_agn'],
                        params['beta_agn'],
                        params['gamma_agn'],
                        log_sigma_UV, log_sigma_UV_err,
                        log_tau_UV_RF_err,
                        bwb_beta_err)
    
    mu_err = np.sqrt(
        m_err**2 +
        M_i_pred_err**2 +
        (0.055 * z)**2 +
        np.exp(params['log_f'])**2
    )

    ll_agn = np.sum(stats.norm.logpdf(mu_pred - mu_cosmo, scale=mu_err))

    m_model = M_pred + mu_cosmo  # model-predicted magnitude

    ll_completeness = 0.0
    if completeness_params is not None:
        completeness2d, mag_centers, _, _, _, completeness_scatter = completeness_params
        ll_completeness, blobs = completeness_loglike(
            m_model=m_model, mu_err=mu_err, z=z,
            completeness2d=completeness2d, m_grid=mag_centers,
            sigma_completeness=completeness_scatter
        )

    #ll_theta, _cmb = loglike_cmb_theta_simple(cosmo)  # or pass omega_b_h2 if you prefer
    
    # print(f"Log-likelihood components: ll_snia={ll_snia:.2f}, ll_agn={ll_agn:.2f}, ll_completeness={ll_completeness:.2f}")
    return ll_snia + ll_agn - ll_completeness, np.array(blobs)

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
                      only_sna=False, completeness=True, use_full_cov=True,
                      resume=False, speed="production"):

    priors, model_labels, model_labels_latex = get_model_params(cosmo_model, only_sna=only_sna)
    ndim = len(model_labels)
    print(f"Running sampling with {ndim} parameters for cosmological model: {cosmo_model}")

    # Prepare data
    df_pantheon_filtered = df_pantheon[['zHD', 'MU_SH0ES', 'MU_SH0ES_ERR_DIAG', 'CEPH_DIST', 'IS_CALIBRATOR',
                                        'm_b_corr', 'x1', 'c', 'biasCor_m_b', 'HOST_LOGMASS']].copy()
    df_agn_filtered = df_agn[['z', 'apparent_mag_2500', 'apparent_mag_2500_err', 'apparent_mag_i_rest', 'apparent_mag_i',
                              'log_sigma_UV', 'log_sigma_UV_err', 'log_tau_UV_RF', 'log_tau_UV_RF_err',
                              'bwb_beta', 'bwb_beta_err', 'ra', 'dec'
                              ]].copy()

    if completeness:
        completeness_params = get_completeness_function_2d(df_agn_filtered, plot=True)
    else:
        completeness_params = None

    
    print(f"log tau UV RF mean: {np.average(df_agn_filtered['log_tau_UV_RF']):.4f}")
    print(f"log tau UV RF pivot (weighted): {np.average(df_agn_filtered['log_tau_UV_RF'], weights=1 / df_agn_filtered['log_tau_UV_RF_err']**2):.4f}")

    print(f"log sigma0 mean: {np.average(df_agn_filtered['log_sigma_UV']):.4f}")
    print(f"log sigma0 pivot (weighted): {np.average(df_agn_filtered['log_sigma_UV'], weights=1 / df_agn_filtered['log_sigma_UV_err']**2):.4f}")

    #print(f"bwb_beta pivot (weighted): {np.average(df_agn_filtered['bwb_beta'], weights=1 / df_agn_filtered['bwb_beta_err']**2):.4f}")

    _z_pivot = (1 / np.exp(np.mean(np.log(1 / (1 + df_agn_filtered['z']))))) - 1
    print(f"z mean: {df_agn_filtered['z'].mean():.3f}, calculated z pivot: {_z_pivot:.3f}")


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
    checkpoint_folder = 'results/hubble'
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    checkpoint_file = os.path.join(checkpoint_folder, 
                                   f'dynesty_checkpoint_{cosmo_model}_{'sna' if only_sna else 'joint'}_{speed}.save')
    print(f"Checkpoint file: {checkpoint_file}")
    with multiprocessing.get_context("spawn").Pool(
        processes=num_cores,
        initializer=dynesty_initializer,
        initargs=(_agn_data, _pantheon_data, _dynesty_config, 
                    _sna_LogdetCov, _sna_L, _sna_Lower)
    ) as pool:            
        # use NestedSampler for precise log-evidence estimates (e.g., model selection)
        # use DynamicNestSampler for Cosmological parameter inference
        if resume:
            print("[WARNING] Resuming from checkpoint file...")
            if isinstance(resume, str):
                checkpoint_file = resume
                print(f"Resuming from checkpoint file: {checkpoint_file}")
            elif resume is True:
                print(f"Resuming from default checkpoint file: {checkpoint_file}")
            sampler = DynamicNestedSampler.restore(checkpoint_file, pool=pool)
        else:
            #print("Testing likelihood and prior transform with random samples...")
            #u = np.random.rand(10, ndim)
            #v = pool.map(prior_transform_dynesty, u)
            #l = pool.map(loglike_dynesty, v)
            #print("Fraction of finite likelihoods:", np.sum(np.isfinite(l)) / len(l))

            sampler = DynamicNestedSampler(
                loglike_dynesty,
                prior_transform_dynesty,
                ndim,
                update_interval=10*ndim,
                bound='multi',
                sample='rwalk',
                pool=pool,
                queue_size=num_cores,
                blob=True
            )
            if speed == 'fast':
                print("[Warning] Starting fast run...")
                sampler.run_nested(
                    resume=resume,
                    checkpoint_file=checkpoint_file,
                    print_progress=True,
                    dlogz_init=10,                 
                    n_effective=50,                # 300–1000 typical for model comparison
                    nlive_init=10,   # bump live points
                    nlive_batch=10   # reasonable batch size for dynamic allocation
                )
            elif speed == "production":
                print("Starting production run...")
                # Production run?
                sampler.run_nested(
                    resume=resume,
                    checkpoint_file=checkpoint_file,
                    print_progress=True,
                    dlogz_init=0.01,                 
                    n_effective=500,                # 300–1000 typical for model comparison
                    nlive_init=max(500, 50*ndim),   # bump live points
                    nlive_batch=max(250, 25*ndim)   # reasonable batch size for dynamic allocation
                    # optional: sample='rwalk', walks=50, bound='multi' if you expect multi-modality
                )
            elif speed == "test":
                print("[Warning] Starting TEST run...")
                # "Fast" test run?
                sampler.run_nested(
                    resume=resume,
                    checkpoint_file=checkpoint_file,
                    print_progress=True,
                    dlogz_init=10,                 
                    n_effective=200,                # 300–1000 typical for model comparison
                    nlive_init=max(200, 20*ndim),   # bump live points
                    nlive_batch=max(100, 10*ndim)   # reasonable batch size for dynamic allocation
                )


    results = sampler.results
    logZ, logZerr = results.logz[-1], results.logzerr[-1]
    print(f"\nBayesian evidence logZ = {logZ:.2f} ± {logZerr:.2f}")
    if logZerr > 1:
        print("Warning: logZ error is large, consider increasing nlive or maxiter.")
    
    # --- pull arrays from results ---
    samples = results.samples                               # (nsamp, ndim)
    logl    = results.logl                                  # (nsamp,)
    weights = np.exp(results.logwt - results.logz[-1])      # (nsamp,)
    blobs   = results.blob                                 # (nsamp, nobj) if blob=True

    # Keep equal-weight resampling
    idx = np.arange(weights.size)
    flat_idx = dyfunc.resample_equal(idx, weights)          # (nsamp,)
    flat_samples = samples[flat_idx]
    flat_blobs   = blobs[flat_idx]

    if only_sna:
        return sampler, flat_samples, model_labels, None, logZ, logZerr

    # --- safety checks ---
    if blobs is None:
        raise RuntimeError("results.blobs is None. Did you run with blob=True and return (logl, blob)?")

    z = _agn_data['z']

    # ===== Highest posterior weight (MAP-ish) sample =====
    idx_max_weight = np.argmax(weights)
    integrals_max_w = blobs[idx_max_weight,:][0]  # this is integrals for that sample, shape: (nobj,)
    dmi_max_w = blobs[idx_max_weight,:][1]  # this is dmi for that sample, shape: (nobj,)
    
    # Bin dmi in redshift
    # Interpolate dmi vs redshift for smooth plotting or further analysis (no binning)
    dmi_interp = interp1d(z, dmi_max_w, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # Plot dmi_interp vs z for the highest-weight sample
    z_plot = np.linspace(z.min(), z.max(), 200)
    plt.figure(figsize=(8, 5))
    plt.plot(z_plot, dmi_interp(z_plot), label="dmi_interp(z)")
    plt.xlabel("Redshift (z)")
    plt.ylabel("dmi (mag)")
    plt.title("Interpolated dmi vs z — highest posterior weight sample")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/completeness/dmi_interp_vs_z_highest_weight.png", dpi=150)
    plt.close()

    print("\nHighest-weight (posterior) sample:")
    print("  idx:", idx_max_weight)
    print("  logl:", float(logl[idx_max_weight]))
    print("  weight:", float(weights[idx_max_weight]))
    print("  (preview) integrals[:10]:", integrals_max_w[:10])

    # Plot log(integrals) vs redshift for highest-weight sample
    plt.figure(figsize=(8, 5))
    plt.scatter(z, integrals_max_w, s=16, alpha=0.75)
    plt.xlabel("Redshift (z)")
    plt.ylabel("integral  (completeness)")
    plt.title("Completeness integrals vs z — highest posterior weight sample")
    plt.grid(True)
    plt.tight_layout()
    # Optional: save to disk
    plt.savefig("plots/completeness/integrals_vs_z_highest_weight.png", dpi=150)
    #plt.show()
    plt.close()


    # Posterior summaries over resampled blobs (per-object)
    posterior_mean_logint = np.mean(flat_blobs, axis=0)
    posterior_med_logint  = np.median(flat_blobs, axis=0)

    # print("\nPosterior (equal-weight) blob summaries:")
    # print("  per-object mean (first 10):", posterior_mean_logint[:10])
    # print("  per-object median (first 10):", posterior_med_logint[:10])

    # Stats
    neff = (weights.sum()**2) / (weights**2).sum()
    print("\nDynesty results stats:")
    print("  samples shape:", samples.shape)
    print("  blobs shape:", blobs.shape)
    print("  weights max:", float(weights.max()))
    print("  effective samples (ESS):", float(neff))
    print("  resampled samples shape:", flat_samples.shape)
    print("  resampled blobs shape:", flat_blobs.shape)

    # Optional: median params from equal-weight posterior
    median_samples = np.median(flat_samples, axis=0)
    print("\nMedian parameters (equal-weight posterior):")
    print(median_samples)

    # If you want to keep your existing summary:
    display_results_summary(flat_samples, cosmo_model, z_pivot_agn)

    #z_pivot_best, _, _ = find_optimal_pivot(flat_samples, cosmo_model, df_agn_filtered)
    #print(f"Optimal z pivot for {cosmo_model}: {z_pivot_best:.3f}")

    #display_diagnostics(sampler, cosmo_model, fitting_method=fitting_method)
    #np.save(f"results/hubble/flat_samples_{cosmo_model}_{'sna' if only_sna else 'agn'}.npy", flat_samples)

    return sampler, flat_samples, model_labels, dmi_max_w, logZ, logZerr




def run_single(df_agn, cosmo_model, completeness=True, use_full_cov=True, 
               N=None, resume=False, only_sna=False, speed="production"):

    # Load data
    #global _sna_LogdetCov, _sna_L, _sna_Lower
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data(agn_data_filepath, populate_sdss=populate_sdss_fields)

    if N is not None:
        print(f"Limiting AGN data to first {N} entries for speed...")
        df_agn = df_agn.head(N)
        #df_pantheon = df_pantheon.head(N)

    sampler, flat_samples, model_labels, dmag_corr, logZ, logZerr = run_mcmc_pipeline(df_agn, df_pantheon, cosmo_model=cosmo_model, 
                                                         only_sna=only_sna, completeness=completeness, use_full_cov=use_full_cov,
                                                         resume=resume, speed=speed)

    plot_path = f"plots/hubble/{prefix}/{cosmo_model}_{'sna' if only_sna else 'joint'}_{speed}"
    os.makedirs(plot_path, exist_ok=True)

    print("Plotting full dynesty corner...")
    plot_dynesty(sampler.results, cosmo_model, plot_path)

    if only_sna:
        print("Skipping AGN-specific plots for SNe-only run.")
        return sampler, flat_samples, model_labels, dmag_corr, logZ, logZerr

    print("Plotting predicted vs actual M2500...")
    plot_predicted_vs_actual_M2500(flat_samples, df_agn, cosmo_model=cosmo_model, z_pivot_agn=z_pivot_agn, debias=False, show=False, plot_path=plot_path)
    plot_predicted_vs_actual_M2500(flat_samples, df_agn, cosmo_model=cosmo_model, z_pivot_agn=z_pivot_agn, debias=True, show=False, dms=dmag_corr, plot_path=plot_path)

    print("Plotting Hubble diagram...")
    residuals, mu_pred_median, mu_pred_std = plot_hubble(flat_samples, df_agn, df_pantheon, 
                                                         cosmo_model=cosmo_model, z_pivot_agn=z_pivot_agn, 
                                                         show_true=False, show=False, debias=False, plot_path=plot_path)
    debiased_residuals, _, _ = plot_hubble(flat_samples, df_agn, df_pantheon, 
                                                         cosmo_model=cosmo_model, z_pivot_agn=z_pivot_agn, 
                                                         show_true=False, show=False, debias=True, dms=dmag_corr, plot_path=plot_path)

    print("Plotting predicted L2500 vs ...")
    plot_predicted_L2500_vs_sigmahat(flat_samples, df_agn, cosmo_model=cosmo_model, z_pivot_agn=z_pivot_agn, show=False, plot_path=plot_path)
    
    print("Plotting cosmological posteriors corner plot...")
    plot_cosmo_corner(None, flat_samples, cosmo_model, z_pivot_sna, z_pivot_agn, show=False, plot_path=plot_path)

    print("Plotting completeness vs magnitude at redshifts...")
    p_detect, mag_centers, z_centers, dm, dz, completeness_scatter = get_completeness_function_2d(df_agn, plot=True)
    plot_completeness_vs_mag_at_redshifts(p_detect, mag_centers, z_centers)

    print("Plotting debiased residuals...")
    plot_full_residuals(df_agn, debiased_residuals, flat_samples, cosmo_model, z_pivot_agn, show=False, plot_path=plot_path)

    # Example usage:
    # Assuming `samples` is a dict from your MCMC run
    if cosmo_model == 'Flatw0waCDM':
        rho_w0_wa = posterior_corr(flat_samples, cosmo_model, z_pivot_agn)
        print(f"Posterior correlation coefficient (w0, wa) at z_p={z_pivot_agn}: {rho_w0_wa:.3f}")

    if cosmo_model == 'Flatw0waCDM':
        zp = compute_pivot_redshift(flat_samples, cosmo_model)
        print("Computed pivot redshift: ", zp)

    return sampler, flat_samples, model_labels, dmag_corr, logZ, logZerr


def run_all(df_agn, cosmo_model, speed="production", resume=False, N=None):
    cosmo_models = ['Flatw0waCDM', 'FlatwCDM']
    cosmo_models_latex = {'Flatw0waCDM': r'Flat$w_0w_a$CDM', 'FlatwCDM': r'Flat$w$CDM'}
    cosmo_models_dict = {k: {} for k in cosmo_models}
    results_latex = []
    for cosmo_model in cosmo_models:
        _, samples_joint, _, _, logZ_joint, logZerr_joint = run_single(df_agn, cosmo_model=cosmo_model, 
                                                                       only_sna=False, 
                                                                       resume=resume, 
                                                                       speed=speed, N=N)
        _, samples_sna, _, _, logZ_sna, logZerr_sna = run_single(df_agn, cosmo_model=cosmo_model, 
                                                         only_sna=True, resume=resume, speed=speed, N=N)

        plot_cosmo_corner(samples_sna, samples_joint, cosmo_model, z_pivot_sna, z_pivot_agn, show=False)

        cosmo_models_dict[cosmo_model]['logZ'] = logZ_joint
        cosmo_models_dict[cosmo_model]['logZerr'] = logZerr_joint
        r_sna   = extract_cosmo_results_from_samples(samples_sna, cosmo_model, True,  
                                                    logZ_tuple=(logZ_sna, logZerr_sna), format_for_latex=True, value_fmt="{:.2f}")
        r_joint   = extract_cosmo_results_from_samples(samples_joint, cosmo_model, False,  
                                                    logZ_tuple=(logZ_joint, logZerr_joint), format_for_latex=True, value_fmt="{:.2f}")
        results_latex.extend([r_sna, r_joint])

    make_cosmo_table_latex(results_latex, write_path=f"plots/hubble/{prefix}/")


    logZ_1 = cosmo_models_dict[cosmo_models[0]]['logZ']
    logZerr_1 = cosmo_models_dict[cosmo_models[0]]['logZerr']
    model_1_name = cosmo_models_latex[cosmo_models[0]]
    logZ_2 = cosmo_models_dict[cosmo_models[1]]['logZ']
    logZerr_2 = cosmo_models_dict[cosmo_models[1]]['logZerr']
    model_2_name = cosmo_models_latex[cosmo_models[1]]
    print(f"Comparing models {cosmo_models[0]} and {cosmo_models[1]} by log-evidence:")
    print(f"  {model_1_name}: logZ = {logZ_1:.2f} ± {logZerr_1:.2f}")
    print(f"  {model_2_name}: logZ = {logZ_2:.2f} ± {logZerr_2:.2f}")
    compare_models_by_log_evidence(logZ_1=logZ_1, logZerr_1=logZerr_1, 
                                   logZ_2=logZ_2, logZerr_2=logZerr_2,
                                   model_1_name=model_1_name,
                                   model_2_name=model_2_name)

if __name__ == "__main__":
    #global _sna_LogdetCov, _sna_L, _sna_Lower

    parser = argparse.ArgumentParser(description="Run Hubble fit pipeline.", allow_abbrev=True)
    parser.add_argument("agn_data_filepath", type=str, help="Path to AGN data file")
    parser.add_argument("--force_populate_fields", action="store_true", help="Force populate fields")
    parser.add_argument("--cosmo_model", type=str, default="FlatwCDM", help="Cosmological model (default: FlatwCDM)")
    parser.add_argument("--disable_completeness", action="store_true", default=False, help="Enable completeness correction (default: True)")
    parser.add_argument("--disable_full_covariance", action="store_true", default=False, help="Use full covariance matrix for SNIa likelihood (default: False)")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume previous MCMC run (default: False)")
    parser.add_argument("--run", type=str, choices=["full", "single"], default="single", help="Run mode: compare_models, compare_sna, full, or single (default: single)")
    parser.add_argument("--speed", type=str, choices=["production", "test", "fast"], default="production", help="Sampling speed: production, test, or fast (default: production)")
    parser.add_argument("--N", type=int, default=None, help="Number of AGNs to run (default: all)")
    args = parser.parse_args()

    if args.disable_full_covariance:
        print("Warning: Running without full covariance may lead to underestimated uncertainties.")
    if args.disable_completeness:
        print("Warning: Running without completeness correction may lead to biased results.")
    if args.resume:
        print("Warning: Resuming previous MCMC run.")


    df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data(args.agn_data_filepath, populate_sdss=args.force_populate_fields)

    if args.run == "single": # default
        run_single(df_agn=df_agn, cosmo_model=args.cosmo_model,
             completeness=not args.disable_completeness, use_full_cov=not args.disable_full_covariance, resume=args.resume,
             speed=args.speed, N=args.N, only_sna=True)
    elif args.run == "full":
        run_all(df_agn=df_agn, cosmo_model=args.cosmo_model, speed=args.speed, resume=args.resume, N=args.N)
    
    print(f"Finished running Hubble fit pipeline for {args.cosmo_model} with only SNIa={args.run}.")
