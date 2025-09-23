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
from hubble_completeness import *

def prior_transform_dynesty(unit_cube, priors, model_labels):
    return [priors[key][0] + (priors[key][1] - priors[key][0]) * x
            for x, key in zip(unit_cube, model_labels)]

def run_mcmc_pipeline(df_agn, df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov, cosmo_model='Flatw0waCDM', 
                      only_sna=False, completeness=True, use_full_cov=True,
                      resume=False, speed="production", use_mu_sh0es=False):

    priors, model_labels, model_labels_latex = get_model_params(cosmo_model, only_sna=only_sna)
    ndim = len(model_labels)
    print(f"Running sampling with {ndim} parameters for cosmological model: {cosmo_model}")

    if completeness:
        completeness_params = get_completeness_function_2d(df_agn, plot=True)
    else:
        completeness_params = None

    print(f"log tau UV RF mean: {np.average(df_agn['log_tau_UV_RF']):.4f}")
    print(f"log tau UV RF pivot (weighted): {np.average(df_agn['log_tau_UV_RF'], weights=1 / df_agn['log_tau_UV_RF_err']**2):.4f}")

    print(f"log sigma0 mean: {np.average(df_agn['log_sigma_UV']):.4f}")
    print(f"log sigma0 pivot (weighted): {np.average(df_agn['log_sigma_UV'], weights=1 / df_agn['log_sigma_UV_err']**2):.4f}")

    print(f"alpha_nu mean: {np.average(df_agn['alpha_nu']):.4f}")
    print(f"alpha_nu pivot (weighted): {np.average(df_agn['alpha_nu'], weights=1 / df_agn['alpha_nu_err']**2):.4f}")


    _z_pivot = (1 / np.exp(np.mean(np.log(1 / (1 + df_agn['z']))))) - 1
    print(f"z mean: {df_agn['z'].mean():.3f}, calculated z pivot: {_z_pivot:.3f}")


    agn_fields = agn_model_req_params + agn_model_req_obs + agn_model_req_errs
    agn_fields += ('apparent_mag_2500', 'apparent_mag_2500_err', 'z', 'z_err', 'object_id')
    agn_data = {col: df_agn[col].values for col in agn_fields if col in df_agn.columns}

    pantheon_fields = ['zHD', 'm_b_corr', 'IS_CALIBRATOR']
    pantheon_data = {col: df_pantheon[col].values for col in df_pantheon.columns}

    checkpoint_folder = f'results/dynesty_checkpoint/{prefix}'
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    checkpoint_file = os.path.join(checkpoint_folder,
                                   f"dynesty_checkpoint_{cosmo_model}_{'sna' if only_sna else 'joint'}_{speed}.save")
    print(f"Checkpoint file: {checkpoint_file}")
    print(f"Starting Hubble Fit with {len(agn_data['z'])} AGNs and {len(pantheon_data['zHD'])} SNes...")
    with multiprocessing.get_context("spawn").Pool(
        processes=num_cores
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
            logl_kwargs = dict(
                agn_data=agn_data,
                pantheon_data=pantheon_data,
                _sna_L=_sna_L,
                _sna_Lower=_sna_Lower,
                _sna_LogdetCov=_sna_LogdetCov,
                cosmo_model=cosmo_model,
                completeness_params=completeness_params,
                only_sna=only_sna,
                use_full_cov=use_full_cov,
                use_mu_sh0es=use_mu_sh0es
            )
            ptform_kwargs = dict(priors=priors, model_labels=model_labels)
            sampler = DynamicNestedSampler(
                log_likelihood,
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
                    checkpoint_file=checkpoint_file,
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
                    checkpoint_file=checkpoint_file,
                    print_progress=True,
                    dlogz_init=0.01,                 
                    n_effective=2000,                # 300–1000 typical for model comparison
                    nlive_init=max(500, 50*ndim),   # bump live points
                    nlive_batch=max(250, 25*ndim)   # reasonable batch size for dynamic allocation
                    # optional: sample='rwalk', walks=50, bound='multi' if you expect multi-modality
                )
            elif speed == "dev":
                print("[Warning] Starting DEV run...")
                # "Fast" test run?
                sampler.run_nested(
                    resume=resume,
                    checkpoint_file=checkpoint_file,
                    print_progress=True,
                    dlogz_init=0.1,                 
                    n_effective=200,                # 300–1000 typical for model comparison
                    nlive_init=max(100, 25*ndim),   # bump live points
                    nlive_batch=max(50, 15*ndim)   # reasonable batch size for dynamic allocation
                )

            elif speed == "test":
                print("[Warning] Starting TEST run...")
                # "Fast" test run?
                sampler.run_nested(
                    resume=resume,
                    checkpoint_file=checkpoint_file,
                    print_progress=True,
                    dlogz_init=0.1,                 
                    n_effective=200,                # 300–1000 typical for model comparison
                    nlive_init=max(200, 30*ndim),   # bump live points
                    nlive_batch=max(100, 20*ndim)   # reasonable batch size for dynamic allocation
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

    z = agn_data['z']

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
    os.makedirs("plots/completeness", exist_ok=True)
    plt.savefig("plots/completeness/dmi_interp_vs_z_highest_weight.png", dpi=150)
    plt.close()

    print("\nHighest-weight (posterior) sample:")
    print("  idx:", idx_max_weight)
    print("  logl:", float(logl[idx_max_weight]))
    print("  weight:", float(weights[idx_max_weight]))
    print("  (preview) integrals[:10]:", integrals_max_w[:10])

    # Plot log(integrals) vs redshift for highest-weight sample
    plt.figure(figsize=(8, 5))
    plt.scatter(z, integrals_max_w, s=16, alpha=0.3)
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

    print("1 sigma scatter on HD (magnitudes)")
    sigma_intrinsic = float(np.exp(median_samples[model_labels.index('log_f')]))
    print("  sigma_intrinsic:", sigma_intrinsic)

    return sampler, flat_samples, model_labels, dmi_max_w, logZ, logZerr




def run_single(df_agn, df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov, cosmo_model, completeness=True, use_full_cov=True, 
               N=None, resume=False, only_sna=False, speed="production", use_mu_sh0es=False, cosmo_model_samples={}, verbose=True):

    # Load data
    #global _sna_LogdetCov, _sna_L, _sna_Lower
    #df_agn, df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_data(agn_data_filepath, populate_sdss=populate_sdss_fields)

    if N is not None:
        print(f"Limiting AGN data to first {N} entries for speed...")
        df_agn = df_agn.head(N)
        #df_pantheon = df_pantheon.head(N)

    sampler, flat_samples, model_labels, dmag_corr, logZ, logZerr = run_mcmc_pipeline(df_agn, df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov, cosmo_model=cosmo_model, 
                                                         only_sna=only_sna, completeness=completeness, use_full_cov=use_full_cov,
                                                         resume=resume, speed=speed, use_mu_sh0es=use_mu_sh0es)

    plot_path = f"plots/hubble/{prefix}/{cosmo_model}_{'sna' if only_sna else 'joint'}_{speed}"
    print(f"Saving plots to ", plot_path)
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
                                                         show_true=False, show=False, debias=False, plot_path=plot_path, verbose=False)
    debiased_residuals, _, _ = plot_hubble(flat_samples, df_agn, df_pantheon, 
                                                         cosmo_model=cosmo_model, z_pivot_agn=z_pivot_agn, 
                                                         show_true=False, show=False, debias=True, dms=dmag_corr, plot_path=plot_path,
                                                         cosmo_model_samples=cosmo_model_samples, verbose=verbose)

    print("Plotting predicted L2500 vs ...")
    plot_predicted_L2500_vs_sigmahat(flat_samples, df_agn, cosmo_model=cosmo_model, z_pivot_agn=z_pivot_agn, 
                                     debias=False, show_residuals=False,
                                     show=False, plot_path=plot_path)
    plot_predicted_L2500_vs_sigmahat(flat_samples, df_agn, cosmo_model=cosmo_model, z_pivot_agn=z_pivot_agn, 
                                     debias=True, dms=dmag_corr, show_residuals=False,
                                     show=False, plot_path=plot_path)
    
    print("Plotting cosmological posteriors corner plot...")
    plot_cosmo_corner(None, flat_samples, cosmo_model, z_pivot_sna, z_pivot_agn, show=False, plot_path=plot_path)

    print("Plotting completeness vs magnitude at redshifts...")
    p_detect, mag_centers, z_centers, dm, dz, completeness_scatter = get_completeness_function_2d(df_agn, plot=True)
    plot_completeness_vs_mag_at_redshifts(p_detect, mag_centers, z_centers)

    print("Plotting debiased residuals...")
    plot_full_residuals(df_agn, debiased_residuals, flat_samples, cosmo_model, z_pivot_agn, debias=True, dms=dmag_corr, show=False, plot_path=plot_path)
    plot_full_residuals(df_agn, residuals, flat_samples, cosmo_model, z_pivot_agn, debias=False, show=False, plot_path=plot_path)

    # Example usage:
    # Assuming `samples` is a dict from your MCMC run
    if cosmo_model in ['FlatwpwaCDM', 'Flatw0waCDM']:
        rho_w0_wa = posterior_corr(flat_samples, cosmo_model, z_pivot_agn)
        print(f"Posterior correlation coefficient (w0, wa) at z_p={z_pivot_agn}: {rho_w0_wa:.3f}")

    if cosmo_model == 'FlatwpwaCDM':
        zp = compute_pivot_redshift(flat_samples, cosmo_model)
        print("Computed pivot redshift: ", zp)
    
    display_results_summary(flat_samples, cosmo_model, z_pivot_agn)

    print('std debiased residuals:', np.std(debiased_residuals))
    # TODO: Subtract typical mu error in quadrature

    return sampler, flat_samples, model_labels, dmag_corr, logZ, logZerr, debiased_residuals


def run_all(df_agn, df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov, cosmo_model, speed="production", resume=False, N=None, use_mu_sh0es=False):
    cosmo_models = ['Flatw0waCDM', 'FlatLambdaCDM', 'FlatwCDM']

    cosmo_models_latex = {'Flatw0waCDM': r'Flat$w_0w_a$CDM', 'FlatwCDM': r'Flat$w$CDM', 'FlatLambdaCDM': r'Flat$\Lambda$CDM'}
    cosmo_models_dict = {k: {} for k in cosmo_models}
    results_latex = []
    cosmo_model_samples = {}

    for cosmo_model in cosmo_models:
        r = run_single(df_agn, df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov, 
                       cosmo_model=cosmo_model, only_sna=False, 
                       resume=resume, speed=speed, N=N,
                       cosmo_model_samples=cosmo_model_samples)
        _, samples_joint, _, _, logZ_joint, logZerr_joint, debiased_residuals = r
        r = run_single(df_agn, df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov, 
                       cosmo_model=cosmo_model, only_sna=True, 
                       resume=resume, speed=speed, N=N, use_mu_sh0es=use_mu_sh0es)
        _, samples_sna, _, _, logZ_sna, logZerr_sna = r
        
        plot_cosmo_corner(samples_sna, samples_joint, cosmo_model, z_pivot_sna, z_pivot_agn, show=False, plot_path=f"plots/hubble/{prefix}")

        cosmo_models_dict[cosmo_model]['logZ'] = logZ_joint
        cosmo_models_dict[cosmo_model]['logZerr'] = logZerr_joint
        r_sna   = extract_cosmo_results_from_samples(samples_sna, cosmo_model, True,  
                                                    logZ_tuple=(logZ_sna, logZerr_sna), format_for_latex=True, value_fmt="{:.2f}")
        r_joint   = extract_cosmo_results_from_samples(samples_joint, cosmo_model, False,  
                                                    logZ_tuple=(logZ_joint, logZerr_joint), format_for_latex=True, value_fmt="{:.2f}")
        cosmo_model_samples[cosmo_model] = samples_joint
        results_latex.extend([r_sna, r_joint])
    
    make_cosmo_table_latex(results_latex, write_path=f"plots/hubble/{prefix}/")


    model_1 = 'Flatw0waCDM'
    model_2 = 'FlatwCDM'
    logZ_1 = cosmo_models_dict[model_1]['logZ']
    logZerr_1 = cosmo_models_dict[model_1]['logZerr']
    model_1_name = cosmo_models_latex[model_1]
    logZ_2 = cosmo_models_dict[model_2]['logZ']
    logZerr_2 = cosmo_models_dict[model_2]['logZerr']
    model_2_name = cosmo_models_latex[model_2]
    print(f"Comparing models {model_1} and {model_2} by log-evidence:")
    print(f"  {model_1_name}: logZ = {logZ_1:.2f} ± {logZerr_1:.2f}")
    print(f"  {model_2_name}: logZ = {logZ_2:.2f} ± {logZerr_2:.2f}")
    compare_r = compare_models_by_log_evidence(logZ_1=logZ_1, logZerr_1=logZerr_1, 
                                   logZ_2=logZ_2, logZerr_2=logZerr_2,
                                   model_1_name=model_1_name,
                                   model_2_name=model_2_name,
                                   write_path=f"plots/hubble/{prefix}/")
    
    write_results_tex_variables(df_agn, cosmo_model_samples['Flatw0waCDM'], 'Flatw0waCDM', compare_r, z_pivot_agn,
                                f"plots/hubble/{prefix}")


if __name__ == "__main__":
    #global _sna_LogdetCov, _sna_L, _sna_Lower

    parser = argparse.ArgumentParser(description="Run Hubble fit pipeline.", allow_abbrev=True)
    parser.add_argument("agn_data_filepath", type=str, help="Path to AGN data file")
    parser.add_argument("--force_populate_fields", action="store_true", help="Force populate fields")
    parser.add_argument("--cosmo_model", type=str,  default="FlatwCDM", choices=["FlatwCDM", "Flatw0waCDM", "FlatLambdaCDM"],
                         help="Cosmological model (default: FlatwCDM)")
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
    df_agn = load_agn_data(args.agn_data_filepath, populate_sdss=args.force_populate_fields, 
                           spectra_fit_csv=args.spectra_fit_csv, zquery_csv=args.zquery_csv)

    if args.N and args.N > 0:
        # df_agn = df_agn.sample(n=args.N, random_state=42)
        df_agn = df_agn[:args.N]
    if args.run == "single": # default
        run_single(df_agn=df_agn, df_pantheon=df_pantheon, _sna_L=_sna_L, _sna_Lower=_sna_Lower, _sna_LogdetCov=_sna_LogdetCov, cosmo_model=args.cosmo_model,
             completeness=not args.disable_completeness, use_full_cov=not args.disable_full_covariance, resume=args.resume,
             speed=args.speed, N=args.N, only_sna=args.only_sna, use_mu_sh0es=args.use_mu_sh0es)
    elif args.run == "full":
        run_all(df_agn=df_agn, df_pantheon=df_pantheon, _sna_L=_sna_L, _sna_Lower=_sna_Lower, _sna_LogdetCov=_sna_LogdetCov, 
                cosmo_model=args.cosmo_model, speed=args.speed, resume=args.resume, N=args.N, use_mu_sh0es=args.use_mu_sh0es)
    
    print(f"Finished running Hubble fit pipeline for {args.cosmo_model}")
