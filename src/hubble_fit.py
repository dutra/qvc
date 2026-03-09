import os
import multiprocessing

num_cores = os.environ.get("NUM_CORES", os.cpu_count()-2)
try:
    num_cores = int(num_cores)
except ValueError:
    print(f"Invalid NUM_CORES value '{num_cores}', ignoring.")
    num_cores = os.cpu_count()-2

if multiprocessing.current_process().name == "MainProcess":
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
from scipy import stats
from dynesty import DynamicNestedSampler
from dynesty import utils as dyfunc
import argparse
from scipy.interpolate import interp1d
from functools import partial

import matplotlib.pyplot as plt

plt.style.use('style.mplstyle')
z_pivot_sna = 0.0
z_pivot_agn = 1.5

from hubble_utils import *
from hubble_likelihood import *
from hubble_plotting import *
from hubble_model import *
from hubble_completeness_refactored import *

def prior_transform_dynesty(unit_cube, priors, model_labels):
    return [priors[key][0] + (priors[key][1] - priors[key][0]) * x
            for x, key in zip(unit_cube, model_labels)]

def run_mcmc_pipeline(df_agn, df_agn_all, df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov, 
                      df_calibrators=None,
                      cosmo_model='Flatw0waCDM', z_pivot_agn=1.5,
                      only_sna=False, completeness=True, use_full_cov=True,
                      resume=False, speed="production", use_mu_sh0es=False,
                      z_range=(0.44, 3.16),
                      N=None,
                      ):


    # Restrict AGN data to redshift range in
    df_agn = df_agn.copy()
    df_agn = df_agn[df_agn['z'].between(z_range[0], z_range[1])].reset_index(drop=True)

    n_avail = len(df_agn)
    print(f"AGN available after cuts: {n_avail}")

    if N is not None:
        if N > n_avail:
            raise ValueError(f"Requested N={N}, but only {n_avail} AGN available after cuts.")

        subset_seed = 42  # fixed seed for reproducibility

        rng = np.random.default_rng(subset_seed)
        idx = rng.choice(n_avail, size=N, replace=False)
        df_agn = df_agn.iloc[np.sort(idx)].reset_index(drop=True)

        print(f"Randomly selected N={N} AGN with subset_seed={subset_seed}")

    priors, model_labels, model_labels_latex = get_model_params(cosmo_model, only_sna=only_sna)
    ndim = len(model_labels)
    print(f"Running sampling with {ndim} parameters for cosmological model: {cosmo_model}")

    if completeness:
        completeness_params = get_completeness_function_2d(df_agn, plot=True)
    else:
        completeness_params = None

    print(f"log_sigma mean: {np.mean(df_agn['log_sigma_UV']):.2f}")
    print(f"sigma mean: {np.mean(10**df_agn['log_sigma_UV']):.2f}")
    print(f"log_tau UV RF mean: {np.mean(df_agn['log_tau_UV_RF']):.2f}")
    print(f"tau UV RF mean: {np.mean(10**df_agn['log_tau_UV_RF']):.2f}")

    _z_pivot = (1 / np.exp(np.mean(np.log(1 / (1 + df_agn['z']))))) - 1
    print(f"z mean: {df_agn['z'].mean():.3f}, calculated z pivot: {_z_pivot:.3f}")


    agn_fields = agn_model_req_params + agn_model_req_obs + agn_model_req_errs
    agn_fields += ('apparent_mag_2500', 'apparent_mag_2500_err', 'z', 'z_err', 'object_id')
    agn_data = {col: df_agn[col].values for col in agn_fields if col in df_agn.columns}

    pantheon_fields = ['zHD', 'm_b_corr', 'IS_CALIBRATOR']
    pantheon_data = {col: df_pantheon[col].values for col in df_pantheon.columns}

    agn_calibrators_fields = ('MU_CAL', 'MU_CAL_ERR', 'AGN_IS_CALIBRATOR') + agn_fields
    if df_calibrators is None:
        agn_calibrators_data = None
    else:
        agn_calibrators_data = {col: df_calibrators[col].values for col in agn_calibrators_fields if col in df_calibrators.columns}

    checkpoint_folder = f'results/hubble_posteriors/{prefix}'
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    checkpoint_file = os.path.join(checkpoint_folder,
                                   f"posteriors_{cosmo_model}_{'sna' if only_sna else 'joint'}_{speed}.h5")
    print(f"Checkpoint file: {checkpoint_file}")
    print(f"Starting Hubble Fit with {len(agn_data['z'])} AGNs and {len(pantheon_data['zHD'])} SNes...")

    if resume:
        print("[WARNING] Resuming from checkpoint file...")
        if isinstance(resume, str):
            checkpoint_file = resume
            print(f"Resuming from checkpoint file: {checkpoint_file}")
        elif resume is True:                
            print(f"Resuming from default checkpoint file: {checkpoint_file}")
        if os.path.exists(checkpoint_file):
            #sampler = DynamicNestedSampler.restore(checkpoint_file, pool=pool)
            r = load_chains(checkpoint_file)
            flat_samples = r["flat_samples"]
            dmi_max_w = r["dmi_max_w"]
            logZ = r["logZ"]
            logZerr = r["logZerr"]
            integrals_max_w = r["integrals_max_w"]
        else:
            print(f"Checkpoint file {checkpoint_file} does not exist. Starting fresh run.")
            #raise RuntimeError("Checkpoint file does not exist.")
            resume = False  # Start fresh if checkpoint doesn't exist


    if not resume:
        with multiprocessing.get_context("spawn").Pool(
            processes=num_cores
        ) as pool:            
            # use NestedSampler for precise log-evidence estimates (e.g., model selection)
            # use DynamicNestSampler for Cosmological parameter inference
            logl_kwargs = dict(
                agn_data=agn_data,
                agn_calibrators_data=agn_calibrators_data,
                pantheon_data=pantheon_data,
                _sna_L=_sna_L,
                _sna_Lower=_sna_Lower,
                _sna_LogdetCov=_sna_LogdetCov,
                cosmo_model=cosmo_model,
                z_pivot_agn=z_pivot_agn,
                completeness_params=completeness_params,
                only_sna=only_sna,
                use_full_cov=use_full_cov,
                use_mu_sh0es=use_mu_sh0es
            )
            ptform_kwargs = dict(priors=priors, model_labels=model_labels)
            sampler = DynamicNestedSampler(
                log_likelihood_nearbylcs if agn_calibrators_data is not None else log_likelihood,
                prior_transform_dynesty,
                ndim,
                logl_kwargs=logl_kwargs,
                ptform_kwargs=ptform_kwargs,
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
                    checkpoint_file=checkpoint_file.replace('.hdf5', '.save'),
                    print_progress=True,
                    dlogz_init=10,                 
                    n_effective=50,                # 300–1000 typical for model comparison
                    nlive_init=20,   # bump live points
                    nlive_batch=5   # reasonable batch size for dynamic allocation
                )
            elif speed == "production":
                print("Starting production run...")
                # Production run?
                sampler.run_nested(
                    resume=resume,
                    checkpoint_file=checkpoint_file.replace('.hdf5', '.save'),
                    print_progress=True,
                    dlogz_init=0.01,                 
                    n_effective=2000,                # 300–1000 typical for model comparison
                    nlive_init=max(1000, 50*ndim),   # bump live points
                    nlive_batch=max(500, 25*ndim)   # reasonable batch size for dynamic allocation
                    # optional: sample='rwalk', walks=50, bound='multi' if you expect multi-modality
                )
            elif speed == "dev":
                print("[Warning] Starting DEV run...")
                # "Fast" test run?
                sampler.run_nested(
                    resume=resume,
                    checkpoint_file=checkpoint_file.replace('.hdf5', '.save'),
                    print_progress=True,
                    dlogz_init=0.01,                 
                    n_effective=500,                # 300–1000 typical for model comparison
                    nlive_init=50,   # bump live points
                    nlive_batch=30   # reasonable batch size for dynamic allocation
                )

            elif speed == "test":
                print("[Warning] Starting TEST run...")
                # "Fast" test run?
                sampler.run_nested(
                    resume=resume,
                    checkpoint_file=checkpoint_file.replace('.hdf5', '.save'),
                    print_progress=True,
                    dlogz_init=0.01,                 
                    n_effective=1000,                # 300–1000 typical for model comparison
                    nlive_init=250,   # bump live points
                    nlive_batch=100   # reasonable batch size for dynamic allocation
                )


        results = sampler.results
        print("Plotting full dynesty corner...")
        plot_dynesty(sampler.results, cosmo_model, checkpoint_folder, only_sna=only_sna, speed=speed)
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

        # if only_sna:
        #     return None, flat_samples, model_labels, None, logZ, logZerr

        # --- safety checks ---
        if blobs is None:
            raise RuntimeError("results.blobs is None. Did you run with blob=True and return (logl, blob)?")
    
        # ===== Highest posterior weight (MAP-ish) sample =====
        idx_max_weight = np.argmax(weights)
        integrals_max_w = blobs[idx_max_weight,:][0]  # this is integrals for that sample, shape: (nobj,)
        dmi_max_w = blobs[idx_max_weight,:][1]  # this is dmi for that sample, shape: (nobj,)
        
        print("\nHighest-weight (posterior) sample:")
        print("  idx:", idx_max_weight)
        print("  logl:", float(logl[idx_max_weight]))
        print("  weight:", float(weights[idx_max_weight]))
        print("  (preview) integrals[:10]:", integrals_max_w[:10])

        # Optional: median params from equal-weight posterior
        median_samples = np.median(flat_samples, axis=0)
        print("\nMedian parameters (equal-weight posterior):")
        print(median_samples)
        # Stats
        neff = (weights.sum()**2) / (weights**2).sum()
        print("\nDynesty results stats:")
        print("  samples shape:", samples.shape)
        print("  blobs shape:", blobs.shape)
        print("  weights max:", float(weights.max()))
        print("  effective samples (ESS):", float(neff))
        print("  resampled samples shape:", flat_samples.shape)
        print("  resampled blobs shape:", flat_blobs.shape)
        
        print("1 sigma scatter on HD (magnitudes)")
        sigma_intrinsic = float(np.exp(median_samples[model_labels.index('log_f')]))
        print("  sigma_intrinsic:", sigma_intrinsic)

        # we should save flat_samples, dmi_max_w, logZ, logZerr
        save_chains(checkpoint_file, flat_samples=flat_samples, dmi_max_w=dmi_max_w, logZ=logZ, logZerr=logZerr, integrals_max_w=integrals_max_w)

        # Bin dmi in redshift
        # Interpolate dmi vs redshift for smooth plotting or further analysis (no binning)
        #dmi_interp = interp1d(z, dmi_max_w)
    dm_interp = make_dm_function(df_agn['apparent_mag_2500'].values, df_agn['z'].values, dmi_max_w)
    
    plot_completeness_diagnostics(dmi_max_w, agn_data['z'], integrals_max_w)

    return flat_samples, model_labels, dm_interp, logZ, logZerr




def run_single(df_agn, df_agn_all, df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov, 
               cosmo_model, completeness=True, use_full_cov=True, 
               N=None, resume=False, only_sna=False, speed="production", use_mu_sh0es=False, cosmo_model_samples={}, verbose=True,
               z_range=(0.44, 3.16),
               z_pivot_agn=1.5, skip_plots=False, residuals_sigma_clip=None, df_calibrators=None):

    flat_samples, model_labels, dm_interp, logZ, logZerr = run_mcmc_pipeline(
                                                        df_agn, df_agn_all,
                                                        df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov,
                                                        df_calibrators=df_calibrators,
                                                        cosmo_model=cosmo_model, z_pivot_agn=z_pivot_agn,
                                                        only_sna=only_sna, completeness=completeness, use_full_cov=use_full_cov,
                                                        z_range=z_range,
                                                        N=N,
                                                        resume=resume, speed=speed, use_mu_sh0es=use_mu_sh0es)
    
    display_results_summary(flat_samples, cosmo_model, z_pivot_agn)
    print("Computing age of the universe with error propagation...")
    age, age_err = compute_age_universe_with_error(flat_samples, cosmo_model, max_eval=200)

    if skip_plots:
        return flat_samples, model_labels, dm_interp, logZ, logZerr, None, (age, age_err)

    plot_path = f"plots/hubble/{prefix}/{cosmo_model}_{'sna' if only_sna else 'joint'}_{speed}"
    print(f"Saving plots to ", plot_path)
    os.makedirs(plot_path, exist_ok=True)

    if only_sna:
        print("Skipping AGN-specific plots for SNe-only run.")
        return flat_samples, model_labels, dm_interp, logZ, logZerr, None, None

    print("Plotting predicted L2500 vs ...")
    # plot_predicted_L2500_vs_sigmahat(flat_samples, df_agn, cosmo_model=cosmo_model, z_pivot_agn=z_pivot_agn, 
    #                                  debias=False, show_residuals=False,
    #                                  show=False, plot_path=plot_path)
    L_residuals_debiased, L_pred_std_debiased = plot_predicted_L2500_vs_sigmahat(flat_samples, df_agn, cosmo_model=cosmo_model, z_pivot_agn=z_pivot_agn, 
                                                            debias=True, dm_interp=dm_interp, show_residuals=False,
                                                            show=False, plot_path=plot_path, df_calibrators=df_calibrators)
    
    chisq_red_L2500, _ = reduced_chi_squared(L_residuals_debiased, L_pred_std_debiased, n_params=len(model_labels)-1)

    print("Plotting Hubble diagram...")

    r = plot_hubble(flat_samples, df_agn, df_pantheon, 
                    cosmo_model=cosmo_model, z_pivot_agn=z_pivot_agn, 
                    show_true=False, show=False, debias=True, dm_interp=dm_interp, plot_path=plot_path,
                    cosmo_model_samples=cosmo_model_samples, verbose=verbose, residuals_sigma_clip=residuals_sigma_clip,
                    df_calibrators=df_calibrators)
    debiased_residuals, debiased_residuals_err, mu_pred_median_debiased, mu_pred_std_debiased, mu_pred_std_debiased_with_scatter = r
    plot_hubble(flat_samples, df_agn, df_pantheon, 
                cosmo_model=cosmo_model, z_pivot_agn=z_pivot_agn, show_residuals=True,
                show_true=False, show=False, debias=False, plot_path=plot_path, verbose=False)    
    if cosmo_model == 'Flatw0waCDM':
        make_agn_latex_table(df_agn, mu_pred_median_debiased, mu_pred_std_debiased_with_scatter, dm_interp=dm_interp, max_rows=30, sort_by='z')

    chisq_red_hubble_debiased, _ = reduced_chi_squared(debiased_residuals, mu_pred_std_debiased, n_params=len(model_labels)-1)



    print("Plotting predicted vs actual M2500...")
    plot_predicted_vs_actual_M2500(flat_samples, df_agn, cosmo_model=cosmo_model, z_pivot_agn=z_pivot_agn, debias=False, show=False, plot_path=plot_path)
    M2500_residuals_debiased, M2500_std_debiased, M2500_binned_residuals_debiased, _ = plot_predicted_vs_actual_M2500(flat_samples, df_agn, cosmo_model=cosmo_model, 
                                                                                  z_pivot_agn=z_pivot_agn, debias=True, show=False, dm_interp=dm_interp,
                                                                                  plot_path=plot_path)
    chisq_red_M2500_debiased, _ = reduced_chi_squared(M2500_residuals_debiased, M2500_std_debiased, n_params=len(model_labels)-1)
    if cosmo_model == 'Flatw0waCDM':
        print("Plotting debiased residuals...")
        plot_full_residuals(df_agn, debiased_residuals, debiased_residuals_err, flat_samples, cosmo_model, z_pivot_agn, debias=True, dm_interp=dm_interp, show=False, plot_path=plot_path)
        plot_full_residuals(df_agn, debiased_residuals, debiased_residuals_err, flat_samples, cosmo_model, z_pivot_agn, debias=True, dm_interp=dm_interp, 
                            show=False, plot_path=plot_path, z_cut=1.0)

        plot_full_residuals(df_agn, debiased_residuals, debiased_residuals_err, flat_samples, cosmo_model, z_pivot_agn, debias=True, dm_interp=dm_interp, 
                            show=False, plot_path=plot_path, key_y='z', key_color='residuals')
        plot_full_residuals(df_agn, debiased_residuals, debiased_residuals_err, flat_samples, cosmo_model, z_pivot_agn, debias=True, dm_interp=dm_interp, 
                            show=False, plot_path=plot_path, z_cut=1.0, key_y='z', key_color='residuals')

        #plot_full_residuals(df_agn, residuals, flat_samples, cosmo_model, z_pivot_agn, debias=False, show=False, plot_path=plot_path)

    
    print("Plotting cosmological posteriors corner plot...")
    plot_cosmo_corner(None, flat_samples, cosmo_model, z_pivot_sna, z_pivot_agn, show=False, 
                      plot_path=plot_path, speed=speed,
                      gauss_sigma=1.5, kde_bw_scale=1.5)

    print("Plotting completeness vs magnitude at redshifts...")
    p_detect, mag_centers, z_centers, dm, dz, completeness_scatter = get_completeness_function_2d(df_agn, plot=True)
    plot_completeness_vs_mag_at_redshifts(p_detect, mag_centers, z_centers)


    # Example usage:
    # Assuming `samples` is a dict from your MCMC run
    if cosmo_model in ['FlatwpwaCDM', 'Flatw0waCDM']:
        rho_w0_wa = posterior_corr(flat_samples, cosmo_model, z_pivot_agn)
        print(f"Posterior correlation coefficient (w0, wa) at z_p={z_pivot_agn}: {rho_w0_wa:.3f}")

    if cosmo_model == 'FlatwpwaCDM':
        zp = compute_pivot_redshift(flat_samples, cosmo_model)
        print("Computed pivot redshift: ", zp)
    
    print('std debiased residuals:', np.std(debiased_residuals))
    # TODO: Subtract typical mu error in quadrature
    
    print(f"\033[94mReduced chi-squared (debiased) M2500: {chisq_red_M2500_debiased:.3f}\033[0m")
    print(f"\033[94mReduced chi-squared (debiased) Hubble: {chisq_red_hubble_debiased:.3f}\033[0m")
    print(f"\033[94mReduced chi-squared (debiased) L2500: {chisq_red_L2500:.3f}\033[0m")
    chisq_dict = {
        'M2500': chisq_red_M2500_debiased,
        'Hubble': chisq_red_hubble_debiased,
        'L2500': chisq_red_L2500
    }
    # try:
    #     write_results_tex_variables(df_agn, flat_samples, cosmo_model, None, 
    #                                 z_pivot_agn, plot_path, 
    #                                 result_prefix=result_prefix,
    #                                 chisq_dict=chisq_dict, age=age)
    # except Exception as e:
    #     print("Error writing TeX variables:", e)

    plot_residuals_vs_alphaOX(df_agn, debiased_residuals, debiased_residuals_err, show=False, plot_path=plot_path)

    return flat_samples, model_labels, dm_interp, logZ, logZerr, debiased_residuals, (age, age_err)


def run_all(df_agn, df_agn_all, df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov, 
            cosmo_models, z_pivot_agn=1.5, skip_plots=False,
            residuals_sigma_clip=None,
            result_prefix="", z_range=(0.44, 3.16),
            speed="production", resume=False, N=None, use_mu_sh0es=False):
    #cosmo_models = ['Flatw0waCDM', 'FlatLambdaCDM', 'FlatwCDM']

    cosmo_models_dict = {k: {} for k in cosmo_models}
    results_latex = []
    cosmo_model_samples = {}

    for cosmo_model in cosmo_models:
        r = run_single(df_agn, df_agn_all, df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov, 
                       cosmo_model=cosmo_model, only_sna=False, 
                       resume=resume, speed=speed, N=N,
                       z_pivot_agn=z_pivot_agn, skip_plots=skip_plots,
                       residuals_sigma_clip=residuals_sigma_clip,
                       z_range=z_range,
                       cosmo_model_samples=cosmo_model_samples)
        
        samples_joint, _, _, logZ_joint, logZerr_joint, _, age = r
        #print(f"For model {cosmo_model}, universe age: {age:.3f} Gyr")
        r = run_single(df_agn, df_agn_all, df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov, 
                       cosmo_model=cosmo_model, only_sna=True, 
                       z_pivot_agn=z_pivot_agn, skip_plots=skip_plots,
                       residuals_sigma_clip=residuals_sigma_clip,
                       z_range=z_range,
                       resume=resume, speed=speed, N=N, use_mu_sh0es=use_mu_sh0es)
        samples_sna, _, _, logZ_sna, logZerr_sna, _, _ = r
        
        plot_cosmo_corner(samples_sna, samples_joint, cosmo_model, z_pivot_sna, z_pivot_agn, show=False, 
                          plot_path=f"plots/hubble/{prefix}", speed=speed,
                          gauss_sigma=1.5, kde_bw_scale=1.5)

        cosmo_models_dict[cosmo_model]['logZ'] = logZ_joint
        cosmo_models_dict[cosmo_model]['logZerr'] = logZerr_joint
        cosmo_models_dict[cosmo_model]['age'] = age

        r_sna   = extract_cosmo_results_from_samples(samples_sna, cosmo_model, True,  
                                                    logZ_tuple=(logZ_sna, logZerr_sna), format_for_latex=True, value_fmt="{:.2f}")
        r_joint   = extract_cosmo_results_from_samples(samples_joint, cosmo_model, False,  
                                                    logZ_tuple=(logZ_joint, logZerr_joint), format_for_latex=True, value_fmt="{:.2f}")
        cosmo_model_samples[cosmo_model] = samples_joint
        results_latex.extend([r_sna, r_joint])
    
    make_cosmo_table_latex(results_latex, write_path=f"plots/hubble/{prefix}/")

    compare_r = compare_models_by_log_evidence_all(cosmo_models_dict, write_path=f"plots/hubble/{prefix}/")
    write_results_tex_variables(df_agn, z_range, cosmo_model_samples, compare_r,
                                f"plots/hubble/{prefix}", result_prefix=result_prefix, age_dict=cosmo_models_dict)
    
    print("================================================================\n\n")
    return cosmo_models_dict, cosmo_model_samples, results_latex, compare_r

if __name__ == "__main__":
    #global _sna_LogdetCov, _sna_L, _sna_Lower

    parser = argparse.ArgumentParser(description="Run Hubble fit pipeline.", allow_abbrev=True)
    parser.add_argument("agn_data_filepath", type=str, help="Path to AGN data file")
    parser.add_argument("--force_populate_fields", action="store_true", help="Force populate fields")
    parser.add_argument("--cosmo_models", type=str, nargs='+',  default=["FlatwCDM"], 
                        choices=["FlatwCDM", "Flatw0waCDM", "FlatLambdaCDM", "FlatwpwaCDM"],
                        help="Cosmological models list (default: FlatwCDM)")
    parser.add_argument("--disable_completeness", action="store_true", default=False, help="Enable completeness correction (default: True)")
    parser.add_argument("--disable_full_covariance", action="store_true", default=False, help="Use full covariance matrix for SNIa likelihood (default: False)")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="Resume previous MCMC run (default: False). If a string is provided, it is used as the checkpoint file.")
    parser.add_argument("--run", type=str, choices=["full", "single"], default="single", help="Run mode: compare_models, compare_sna, full, or single (default: single)")
    parser.add_argument("--speed", type=str, choices=["production", "test", "fast", "dev"], default="production", help="Sampling speed: production, test, or fast (default: production)")
    parser.add_argument("--N", type=int, default=None, help="Number of AGNs to run (default: all)")
    parser.add_argument("--only_sna", action="store_true", default=False, help="Run SNIa-only fit (default: False)")
    parser.add_argument("--use_mu_sh0es", action="store_true", default=False, help="Use MU_SH0ES for SNIa fit (default: False)")
    parser.add_argument("--spectra_fit_csv", type=str, nargs='+', help="Path(s) to spectra fit CSV file(s)")
    parser.add_argument("--zquery_csv", type=str, help="Path to zquery CSV file")
    parser.add_argument("--no_cuts", action="store_true", default=False, help="Disable AGN data cuts (default: False)")
    parser.add_argument("--z_pivot_agn", type=float, default=1.5, help="Pivot redshift for AGN standardization (default: 1.5)")
    parser.add_argument("--skip_plots", action="store_true", default=False, help="Skip plotting steps (default: False)")
    parser.add_argument("--fhost_cut", type=float, default=np.inf, help="Optional fhost cut value (default: 10)")
    parser.add_argument("--exclude_object_ids_csv", type=str, nargs='+', default=[], help="Path(s) to CSV file(s) containing object IDs to exclude")
    parser.add_argument("--residuals_sigma_clip", type=float, default=None, help="Optional residual cut value to exclude outliers (default: None)")
    parser.add_argument("--residuals_csv", type=str, default=None, help="Path to CSV file containing residuals for outlier exclusion (default: None)")
    parser.add_argument("--agn_calibrators", type=str, default=None, help="Path to H5 or CSV file containing AGN data to use as calibrators (default: None)")
    parser.add_argument("--redchi2_cut", type=float, default=None, help="Optional reduced chi-squared cut value to exclude outliers (default: None)")
    parser.add_argument("--iron_frac_cut", type=float, default=None, help="Optional iron fraction cut value to exclude outliers (default: None)")
    parser.add_argument("--sdss_mags_csv", type=str, default=None, help="Path to CSV file containing SDSS magnitudes (default: None)")
    parser.add_argument("--result_prefix", type=str, default="", help="Prefix for result files (default: empty)")
    parser.add_argument("--z_range", type=float, nargs=2, default=[0.44, 3.16], 
                        help="Redshift range for AGN data (default: [0.44, 3.16])")
    parser.add_argument("--pickled", action="store_true", default=False, help="Use pickled data file (default: False)")

    args = parser.parse_args()

    print("Running Hubble fit with the following settings:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    if args.disable_full_covariance:
        print("Warning: Running without full covariance may lead to underestimated uncertainties.")
    if args.disable_completeness:
        print("Warning: Running without completeness correction may lead to biased results.")
    if args.resume:
        print("Warning: Resuming previous MCMC run.")

    df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_pantheon_data()
    df_agn, df_agn_all = load_agn_data(args.agn_data_filepath, populate_sdss=args.force_populate_fields, 
                           apply_cut=not args.no_cuts, fhost_cut=args.fhost_cut,
                           residuals_sigma_clip=args.residuals_sigma_clip, residuals_csv=args.residuals_csv,
                           exclude_object_ids_csv=args.exclude_object_ids_csv,
                           spectra_fit_csv=args.spectra_fit_csv, zquery_csv=args.zquery_csv,
                           redchi2_cut=args.redchi2_cut, iron_frac_cut=args.iron_frac_cut,
                           sdss_mags_csv=args.sdss_mags_csv, pickled=args.pickled)
    
    plot_Mi_relation(df_agn)
    
    if args.agn_calibrators:
        if args.agn_calibrators.endswith('.h5'):
            df_calibrators = read_quasars_from_hdf5(args.agn_data_show)
            df_calibrators = pd.DataFrame(df_calibrators)
        elif args.agn_calibrators.endswith('.csv'):
            df_calibrators = pd.read_csv(args.agn_calibrators)
        else:
            raise ValueError("Unsupported file format for agn_calibrators. Use .h5 or .csv")
    else:
        df_calibrators = None


    # if args.N and args.N > 0:
    #     # df_agn = df_agn.sample(n=args.N, random_state=42)
    #     df_fit = df_fit[:args.N]

    if args.run == "single": # default
        cosmo_models_dict = {k: {} for k in args.cosmo_models}
        for cosmo_model in args.cosmo_models:
            r = run_single(df_agn=df_agn, df_agn_all=df_agn_all, df_pantheon=df_pantheon, _sna_L=_sna_L, _sna_Lower=_sna_Lower, _sna_LogdetCov=_sna_LogdetCov, 
                           cosmo_model=cosmo_model,
                completeness=not args.disable_completeness, use_full_cov=not args.disable_full_covariance, resume=args.resume,
                speed=args.speed, N=args.N, only_sna=args.only_sna, use_mu_sh0es=args.use_mu_sh0es,
                skip_plots=args.skip_plots, residuals_sigma_clip=args.residuals_sigma_clip,
                z_pivot_agn=args.z_pivot_agn, df_calibrators=df_calibrators)
            _, samples_joint, _, _, logZ_joint, logZerr_joint, _, age = r
            cosmo_models_dict[cosmo_model]['logZ'] = logZ_joint
            cosmo_models_dict[cosmo_model]['logZerr'] = logZerr_joint
            cosmo_models_dict[cosmo_model]['age'] = age
        compare_r = compare_models_by_log_evidence_all(cosmo_models_dict, write_path=f"plots/hubble/{prefix}/")
    elif args.run == "full":
        run_all(df_agn=df_agn, df_agn_all=df_agn_all, df_pantheon=df_pantheon, _sna_L=_sna_L, _sna_Lower=_sna_Lower, _sna_LogdetCov=_sna_LogdetCov, 
                cosmo_models=args.cosmo_models, z_pivot_agn=args.z_pivot_agn, skip_plots=args.skip_plots,
                residuals_sigma_clip=args.residuals_sigma_clip,
                result_prefix=args.result_prefix, z_range=args.z_range,
                speed=args.speed, resume=args.resume, N=args.N, use_mu_sh0es=args.use_mu_sh0es)
    
    print(f"Finished running Hubble fit pipeline for {args.cosmo_models}")
