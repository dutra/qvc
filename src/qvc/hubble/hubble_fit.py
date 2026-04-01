import os
import multiprocessing
import traceback

import argparse
from functools import partial
from pathlib import Path

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatwCDM, Flatw0waCDM, FlatLambdaCDM, FlatwpwaCDM
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy import stats
from dynesty import DynamicNestedSampler
from dynesty import utils as dyfunc

plt.style.use(Path(__file__).with_name("style.mplstyle"))
z_pivot_sna = 0.0
z_pivot_agn = 1.5
DEFAULT_COMPLETENESS_SIM_FILE = "data/nov9_mock_mag_z_moresources.h5"

from qvc.hubble.hubble_utils import (
    compare_models_by_log_evidence_all,
    compute_age_universe_with_error,
    compute_pivot_redshift,
    display_results_summary,
    extract_cosmo_results_from_samples,
    get_qvc_result_dir,
    load_agn_data,
    load_chains,
    load_pantheon_data,
    posterior_corr,
    reduced_chi_squared,
    read_quasars_from_hdf5_flat,
    report_pivots,
    save_chains,
    save_cosmo_results_hdf5,
    select_agn_subset_uniform_with_replacement,
    sym_percentile,
    write_results_tex_variables,
)
from qvc.hubble.hubble_likelihood import log_likelihood, log_likelihood_nearbylcs
from qvc.hubble.hubble_plotting import (
    plot_blr_line_lags_vs_l2500,
    plot_completeness_diagnostics,
    plot_completeness_vs_mag_at_redshifts,
    plot_cosmo_corner,
    plot_debias_impact_diagnostics,
    plot_delta_m_flux_recal_vs_redshift,
    plot_dynesty,
    plot_fast_vs_uv_variability,
    plot_full_residuals,
    plot_full_residuals_rz,
    plot_hubble,
    plot_predicted_L2500_vs_sigmahat,
    plot_predicted_vs_actual_M2500,
    plot_redshift_histograms,
    plot_redshift_bin_residual_summary,
    plot_residuals_vs_alphaOX,
)
from qvc.hubble.hubble_model import (
    agn_model_pack_obs,
    evaluate_log_f,
    get_agn_model_spec,
    get_model_params,
)
from qvc.hubble.hubble_completeness_refactored import (
    get_completeness_function_2d,
    get_completeness_function_3d_fhost,
    get_completeness_function_4d_fhost_alpha,
    make_dm_function,
)
from qvc.hubble.hubble_cut_config import (
    DEFAULT_BC_FRAC_CUT,
    DEFAULT_CHI_SQ_CUT,
    DEFAULT_F_HOST_CUT,
    DEFAULT_IRON_FRAC_CUT,
    DEFAULT_REDDENING_EBV_CUT,
    DEFAULT_WRMS_CUT,
)

VALID_COMPLETENESS_MODES = ("2d", "3d_fhost", "4d_fhost_alpha")


def validate_completeness_mode(completeness_mode):
    if completeness_mode not in VALID_COMPLETENESS_MODES:
        raise ValueError(
            f"Invalid completeness_mode={completeness_mode!r}. "
            f"Expected one of {VALID_COMPLETENESS_MODES}."
        )


def prior_transform_dynesty(unit_cube, priors, model_labels):
    return [priors[key][0] + (priors[key][1] - priors[key][0]) * x
            for x, key in zip(unit_cube, model_labels)]

def make_run_tag(
    cosmo_model,
    only_sna,
    speed,
    N,
    z_range,
    use_alpha_lambda_term=False,
    use_redshift_log_f_term=False,
):
    zmin, zmax = z_range
    n_tag = "all" if N is None else f"N{N}"
    z_tag = f"z{zmin:.2f}_{zmax:.2f}".replace(".", "p")
    alpha_tag = "_alphaLam" if use_alpha_lambda_term else ""
    logf_tag = "_logfz" if use_redshift_log_f_term else ""
    return f"{cosmo_model}_{'sna' if only_sna else 'joint'}_{speed}_{n_tag}_{z_tag}{alpha_tag}{logf_tag}"


def validate_resume_checkpoint(results, checkpoint_file, ndim, n_agn):
    required_keys = {
        "flat_samples",
        "dmi_max_w",
        "dmi_posterior_sigma",
        "integrals_max_w",
        "logZ",
        "logZerr",
    }
    missing_keys = sorted(required_keys - set(results.keys()))
    if missing_keys:
        raise RuntimeError(
            f"Resume checkpoint '{checkpoint_file}' is missing required dataset(s): {missing_keys}. "
            "This usually means the file is stale or was written by an older pipeline version. "
            "Delete it or pass resume=False to start a fresh run."
        )

    flat_samples = np.asarray(results["flat_samples"])
    if flat_samples.ndim != 2:
        raise RuntimeError(
            f"Resume checkpoint '{checkpoint_file}' has flat_samples with shape {flat_samples.shape}, "
            "but a 2D array is required. The checkpoint is incompatible with the current pipeline."
        )
    if flat_samples.shape[1] != ndim:
        raise RuntimeError(
            f"Resume checkpoint '{checkpoint_file}' was created for a different parameterization: "
            f"flat_samples has {flat_samples.shape[1]} columns, but the current model expects {ndim}. "
            "This usually happens when resuming with a different cosmology model or code version. "
            "Delete the checkpoint or use a fresh resume path."
        )

    for key in ("dmi_max_w", "integrals_max_w"):
        value = np.asarray(results[key])
        if value.ndim == 0:
            continue
        if value.shape[0] != n_agn:
            raise RuntimeError(
                f"Resume checkpoint '{checkpoint_file}' is incompatible with the current AGN selection: "
                f"{key} has length {value.shape[0]}, but the current run has {n_agn} AGN objects. "
                "This usually means the checkpoint was created with a different input sample or redshift cut. "
                "Delete the checkpoint or use a new output filename."
            )
    if "dmi_posterior_median" in results:
        value = np.asarray(results["dmi_posterior_median"])
        if value.ndim != 0 and value.shape[0] != n_agn:
            raise RuntimeError(
                f"Resume checkpoint '{checkpoint_file}' is incompatible with the current AGN selection: "
                f"dmi_posterior_median has length {value.shape[0]}, but the current run has {n_agn} AGN objects. "
                "Delete the checkpoint or use a new output filename."
            )
    value = np.asarray(results["dmi_posterior_sigma"])
    if value.ndim != 0 and value.shape[0] != n_agn:
        raise RuntimeError(
            f"Resume checkpoint '{checkpoint_file}' is incompatible with the current AGN selection: "
            f"dmi_posterior_sigma has length {value.shape[0]}, but the current run has {n_agn} AGN objects. "
            "Delete the checkpoint or use a new output filename."
        )


def resolve_resume_checkpoint_path(resume, checkpoint_file):
    if not resume:
        return None

    if isinstance(resume, str):
        resume_stripped = resume.strip()
        resume_lower = resume_stripped.lower()
        if resume_lower in {"true", "1", "yes"}:
            resolved_checkpoint = checkpoint_file
        elif resume_lower in {"false", "0", "no"}:
            return None
        else:
            resolved_checkpoint = resume_stripped
    else:
        resolved_checkpoint = checkpoint_file

    if not os.path.exists(resolved_checkpoint):
        raise FileNotFoundError(
            f"Resume was requested, but checkpoint file '{resolved_checkpoint}' does not exist."
        )
    return resolved_checkpoint

def run_mcmc_pipeline(df_agn, df_agn_all, df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov, 
                      df_calibrators=None,
                      cosmo_model='Flatw0waCDM',
                      only_sna=False, completeness=True, use_full_cov=True,
                      resume=False, speed="production",
                      z_range=(0.44, 3.16),
                      prefix="default",
                      completeness_sim_file=DEFAULT_COMPLETENESS_SIM_FILE,
                      completeness_mode="2d",
                      N=None,
                      use_alpha_lambda_term=False,
                      use_redshift_log_f_term=False,
                      ):
    validate_completeness_mode(completeness_mode)
    run_tag = make_run_tag(
        cosmo_model,
        only_sna,
        speed,
        N,
        z_range,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    plot_path = f"plots/hubble/{prefix}/{run_tag}"
    os.makedirs(plot_path, exist_ok=True)

    priors, model_labels, model_labels_latex = get_model_params(
        cosmo_model,
        only_sna=only_sna,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    ndim = len(model_labels)
    print(f"Running sampling with {ndim} parameters for cosmological model: {cosmo_model}")
    if not use_full_cov:
        print("[WARNING] use_full_cov=False: fitting with diagonal SN uncertainties instead of the full covariance matrix.")

    if use_alpha_lambda_term:
        for required_col in ("alpha_lambda", "alpha_lambda_err"):
            if required_col not in df_agn.columns:
                raise KeyError(f"--fit_alpha_lambda_term requires df_agn[{required_col!r}].")
            bad = ~np.isfinite(df_agn[required_col].to_numpy(dtype=float))
            if np.any(bad):
                raise ValueError(
                    f"--fit_alpha_lambda_term requires finite {required_col} for all AGN used in the fit; "
                    f"found {np.count_nonzero(bad)} non-finite rows."
                )

    if completeness:
        if completeness_mode in ("3d_fhost", "4d_fhost_alpha"):
            if "f_host_center" not in df_agn.columns:
                raise KeyError(f"completeness_mode={completeness_mode!r} requires df_agn['f_host_center'].")
            bad_fhost = ~np.isfinite(df_agn["f_host_center"].to_numpy(dtype=float))
            if np.any(bad_fhost):
                raise ValueError(
                    f"completeness_mode={completeness_mode!r} requires finite f_host_center for all AGN used in the fit; "
                    f"found {np.count_nonzero(bad_fhost)} non-finite rows."
                )
        if completeness_mode == "4d_fhost_alpha":
            if "alpha_lambda" not in df_agn.columns:
                raise KeyError("completeness_mode='4d_fhost_alpha' requires df_agn['alpha_lambda'].")
            bad_alpha = ~np.isfinite(df_agn["alpha_lambda"].to_numpy(dtype=float))
            if np.any(bad_alpha):
                raise ValueError(
                    "completeness_mode='4d_fhost_alpha' requires finite alpha_lambda for all AGN used in the fit; "
                    f"found {np.count_nonzero(bad_alpha)} non-finite rows."
                )
        if completeness_sim_file is None:
            print(f"Building {completeness_mode} completeness map using default mock catalog file.")
        else:
            print(f"Building {completeness_mode} completeness map using mock catalog: {completeness_sim_file}")
        if completeness_mode == "4d_fhost_alpha":
            completeness_params = get_completeness_function_4d_fhost_alpha(
                df_agn, sim_file=completeness_sim_file, plot=True, plot_path=plot_path
            )
        elif completeness_mode == "3d_fhost":
            completeness_params = get_completeness_function_3d_fhost(
                df_agn, sim_file=completeness_sim_file, plot=True, plot_path=plot_path
            )
        else:
            completeness_params = get_completeness_function_2d(
                df_agn, sim_file=completeness_sim_file, plot=True, plot_path=plot_path
            )
    else:
        completeness_params = None

    agn_model_req_params, agn_model_req_obs, agn_model_req_errs = get_agn_model_spec(
        use_alpha_lambda_term=use_alpha_lambda_term
    )
    agn_fields = agn_model_req_params + agn_model_req_obs + agn_model_req_errs
    agn_fields += ('apparent_mag_2500', 'apparent_mag_2500_err', 'z', 'z_err', 'object_id')
    if 'f_host_center' in df_agn.columns:
        agn_fields += ('f_host_center',)
    if 'alpha_lambda' in df_agn.columns:
        agn_fields += ('alpha_lambda',)
    agn_data = {col: df_agn[col].values for col in agn_fields if col in df_agn.columns}

    pantheon_fields = ['zHD', 'm_b_corr', 'IS_CALIBRATOR', 'CEPH_DIST', 'MU_SH0ES_ERR_DIAG']
    pantheon_data = {col: df_pantheon[col].values for col in pantheon_fields if col in df_pantheon.columns}

    agn_calibrators_fields = ('MU_CAL', 'MU_CAL_ERR', 'AGN_IS_CALIBRATOR') + agn_fields
    if df_calibrators is None:
        agn_calibrators_data = None
    else:
        agn_calibrators_data = {col: df_calibrators[col].values for col in agn_calibrators_fields if col in df_calibrators.columns}

    checkpoint_folder = get_qvc_result_dir() / "hubble_posteriors" / prefix
    checkpoint_folder.mkdir(parents=True, exist_ok=True)

    checkpoint_file = str(checkpoint_folder / f"posteriors_{run_tag}.h5")
    print(f"Checkpoint file: {checkpoint_file}")
    print(f"Starting Hubble Fit with {len(agn_data['z'])} AGNs and {len(pantheon_data['zHD'])} SNes...")

    if resume:
        print("[WARNING] Resuming from checkpoint file...")
        checkpoint_file = resolve_resume_checkpoint_path(resume, checkpoint_file)
        print(f"Resuming from default checkpoint file: {checkpoint_file}")
        #sampler = DynamicNestedSampler.restore(checkpoint_file, pool=pool)
        try:
            r = load_chains(checkpoint_file)
            validate_resume_checkpoint(
                r,
                checkpoint_file=checkpoint_file,
                ndim=ndim,
                n_agn=len(agn_data["z"]),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to resume from checkpoint '{checkpoint_file}'. "
                "The checkpoint appears incompatible with the current run configuration "
                "(for example: different cosmology model, different selected AGN sample, "
                "or an older file format). Start a fresh run or remove the stale checkpoint."
            ) from exc
        flat_samples = r["flat_samples"]
        dmi_max_w = r["dmi_max_w"]
        dmi_posterior_median = r.get("dmi_posterior_median", dmi_max_w)
        dmi_posterior_sigma = r["dmi_posterior_sigma"]
        logZ = r["logZ"]
        logZerr = r["logZerr"]
        integrals_max_w = r["integrals_max_w"]


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
                use_alpha_lambda_term=use_alpha_lambda_term,
                use_redshift_log_f_term=use_redshift_log_f_term,
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
                    print_progress=True,
                    dlogz_init=10,                 
                    n_effective=50,                # 300–1000 typical for model comparison
                    nlive_init=20,   # bump live points
                    nlive_batch=5   # reasonable batch size for dynamic allocation
                )
            elif speed == "production":
                print("Starting production run...")
                sampler.run_nested(
                    print_progress=True,
                    dlogz_init=0.01,                 
                    n_effective=2000,                # 300–1000 typical for model comparison
                    nlive_init=max(1000, 50*ndim),   # bump live points
                    nlive_batch=max(500, 25*ndim)   # reasonable batch size for dynamic allocation
                    # optional: sample='rwalk', walks=50, bound='multi' if you expect multi-modality
                )
            elif speed == "dev":
                print("[Warning] Starting DEV run...")
                sampler.run_nested(
                    print_progress=True,
                    dlogz_init=0.01,                 
                    n_effective=200,                # 300–1000 typical for model comparison
                    nlive_init=25,   # bump live points
                    nlive_batch=15   # reasonable batch size for dynamic allocation
                )

            elif speed == "test":
                print("[Warning] Starting TEST run...")
                sampler.run_nested(
                    print_progress=True,
                    dlogz_init=0.01,                 
                    n_effective=1000,                # 300–1000 typical for model comparison
                    nlive_init=250,   # bump live points
                    nlive_batch=100   # reasonable batch size for dynamic allocation
                )


        results = sampler.results
        print("Plotting full dynesty corner...")
        plot_dynesty(sampler.results, cosmo_model, plot_path, only_sna=only_sna, speed=speed)
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
        dmi_posterior_median = np.median(flat_blobs[:, 1, :], axis=0)
        dmi_posterior_sigma = 0.5 * (
            np.percentile(flat_blobs[:, 1, :], 84, axis=0)
            - np.percentile(flat_blobs[:, 1, :], 16, axis=0)
        )
        
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
        median_params = dict(zip(model_labels, median_samples))
        sigma_intrinsic = float(
            np.exp(
                evaluate_log_f(
                    median_params,
                    np.array([z_pivot_agn]),
                    z_pivot=z_pivot_agn,
                    use_redshift_log_f_term=use_redshift_log_f_term,
                )[0]
            )
        )
        print("  sigma_intrinsic(z_pivot):", sigma_intrinsic)

        print("Debias correction summary:")
        print("  median |dmi_max_w|:", float(np.nanmedian(np.abs(dmi_max_w))))
        print("  median |dmi_posterior_median|:", float(np.nanmedian(np.abs(dmi_posterior_median))))
        print("  median sigma_dmi:", float(np.nanmedian(dmi_posterior_sigma)))

        save_chains(
            checkpoint_file,
            flat_samples=flat_samples,
            dmi_max_w=dmi_max_w,
            dmi_posterior_median=dmi_posterior_median,
            dmi_posterior_sigma=dmi_posterior_sigma,
            logZ=logZ,
            logZerr=logZerr,
            integrals_max_w=integrals_max_w,
        )

        # Bin dmi in redshift
        # Interpolate dmi vs redshift for smooth plotting or further analysis (no binning)
        #dmi_interp = interp1d(z, dmi_max_w)
    dm_interp = make_dm_function(
        df_agn['apparent_mag_2500'].values,
        df_agn['z'].values,
        dmi_posterior_median,
        f_host_center=df_agn["f_host_center"].values if "f_host_center" in df_agn.columns else None,
        alpha_lambda=df_agn["alpha_lambda"].values if "alpha_lambda" in df_agn.columns else None,
    )

    print("Plotting completeness diagnostics...")

    
    plot_completeness_diagnostics(
        dmi_posterior_median,
        agn_data['z'],
        agn_data['apparent_mag_2500'],
        integrals_max_w,
        plot_path=plot_path,
    )

    return flat_samples, model_labels, dm_interp, logZ, logZerr, dmi_posterior_sigma


def run_single(df_agn, df_agn_all, df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov, 
               cosmo_model, completeness=True, use_full_cov=True, 
               N=None, resume=False, only_sna=False, speed="production", 
               cosmo_model_joint_samples={}, cosmo_model_sna_samples={},
               verbose=True,
               z_range=(0.44, 3.16),
               skip_plots=False, residuals_sigma_clip=None, df_calibrators=None,
               prefix="default", uniform_redshift_distribution=False,
               completeness_sim_file=DEFAULT_COMPLETENESS_SIM_FILE,
               completeness_mode="2d",
               use_alpha_lambda_term=False,
               use_redshift_log_f_term=False):
    validate_completeness_mode(completeness_mode)
    run_tag = make_run_tag(
        cosmo_model,
        only_sna,
        speed,
        N,
        z_range,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    plot_path = f"plots/hubble/{prefix}/{run_tag}"
    os.makedirs(plot_path, exist_ok=True)
    print(f"Saving plots to ", plot_path)
    if completeness:
        if completeness_sim_file is None:
            print("Completeness enabled with default mock catalog file.")
        else:
            print(f"Completeness enabled with mock catalog file: {completeness_sim_file}")

    if uniform_redshift_distribution:
        df_agn_fit_selection = select_agn_subset_uniform_with_replacement(
            df_agn,
            z_range=z_range,
            N=N,
        )
        plot_redshift_histograms(df_pantheon, df_agn_fit_selection, xscale="linear", plot_path=plot_path)
    else:
        df_agn_fit_selection = df_agn[df_agn["z"].between(z_range[0], z_range[1])].copy()
        plot_redshift_histograms(df_pantheon, df_agn, xscale="log", plot_path=plot_path)

    plot_delta_m_flux_recal_vs_redshift(df_agn_fit_selection, plot_path=plot_path)

    report_pivots(df_agn_fit_selection)

    flat_samples, model_labels, dm_interp, logZ, logZerr, dmi_posterior_sigma = run_mcmc_pipeline(
                                                        df_agn_fit_selection, df_agn_all,
                                                        df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov,
                                                        df_calibrators=df_calibrators,
                                                        cosmo_model=cosmo_model,
                                                        only_sna=only_sna, completeness=completeness, use_full_cov=use_full_cov,
                                                        z_range=z_range,
                                                        N=N,
                                                        resume=resume, speed=speed,
                                                        prefix=prefix,
                                                        completeness_sim_file=completeness_sim_file,
                                                        completeness_mode=completeness_mode,
                                                        use_alpha_lambda_term=use_alpha_lambda_term,
                                                        use_redshift_log_f_term=use_redshift_log_f_term)
    display_results_summary(
        flat_samples,
        cosmo_model,
        z_pivot_agn,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    print("Computing age of the universe with error propagation...")
    age, age_err = compute_age_universe_with_error(flat_samples, cosmo_model, max_eval=200)

    if skip_plots or only_sna:
        print("Skipping plots, returning results...")
        return flat_samples, model_labels, dm_interp, logZ, logZerr, None, age, age_err

    # if only_sna:
    #     print("Skipping AGN-specific plots for SNe-only run.")
    #     return flat_samples, model_labels, dm_interp, logZ, logZerr, None, age, age_err

    print("Plotting predicted L2500 vs ...")

    L_residuals_debiased, L_pred_std_debiased = plot_predicted_L2500_vs_sigmahat(flat_samples, df_agn, cosmo_model=cosmo_model, z_pivot_agn=z_pivot_agn, 
                                                            debias=True, dm_interp=dm_interp, show_residuals=False,
                                                            show=False, plot_path=plot_path, df_calibrators=df_calibrators, z_range=z_range)

    plot_blr_line_lags_vs_l2500(
        flat_samples,
        df_agn,
        cosmo_model,
        z_pivot_agn,
        dm_interp,
        plot_path=plot_path,
        show=False,
    )
    
    chisq_red_L2500, _ = reduced_chi_squared(L_residuals_debiased, L_pred_std_debiased, n_params=len(model_labels)-1)

    print("Plotting Hubble diagram...")
    dmi_posterior_sigma_full = None
    if uniform_redshift_distribution:
        print(
            "Uniform-redshift selection uses resampling with replacement; "
            "disabling sigma_dmi overlay on full-sample Hubble plots."
        )
    else:
        if len(df_agn_fit_selection) != len(dmi_posterior_sigma):
            raise ValueError(
                "Fit/plot alignment failure: "
                f"df_agn_fit_selection has length {len(df_agn_fit_selection)}, "
                f"but dmi_posterior_sigma has length {len(dmi_posterior_sigma)}."
            )
        fit_indices = pd.Index(df_agn_fit_selection.index)
        if not fit_indices.isin(df_agn.index).all():
            missing = fit_indices[~fit_indices.isin(df_agn.index)].tolist()[:10]
            raise ValueError(
                "Fit/plot alignment failure: fitted AGN selection contains index values "
                f"not present in df_agn: {missing}"
            )
        dmi_posterior_sigma_full = np.full(len(df_agn), np.nan, dtype=float)
        df_agn_index_positions = pd.Series(np.arange(len(df_agn)), index=df_agn.index)
        dmi_posterior_sigma_full[df_agn_index_positions.loc[fit_indices].to_numpy()] = np.asarray(
            dmi_posterior_sigma, dtype=float
        )
    # Debiased (Bias corrected)
    r = plot_hubble(flat_samples, df_agn, df_pantheon, 
                    cosmo_model=cosmo_model, z_pivot_agn=z_pivot_agn, 
                    show_true=False, show=False, debias=True, dm_interp=dm_interp, plot_path=plot_path,
                    cosmo_model_samples=cosmo_model_joint_samples, verbose=verbose, residuals_sigma_clip=residuals_sigma_clip,
                    df_calibrators=df_calibrators, dmi_sigma=dmi_posterior_sigma_full)
    debiased_residuals, debiased_residuals_err, mu_pred_median_debiased, mu_pred_std_debiased, mu_pred_std_debiased_with_scatter = r
    # Biased
    r = plot_hubble(flat_samples, df_agn, df_pantheon, 
                cosmo_model=cosmo_model, z_pivot_agn=z_pivot_agn, show_residuals=True,
                show_true=False, show=False, debias=False, plot_path=plot_path, verbose=False)
    biased_residuals, biased_residuals_err, _, _, _ = r

    chisq_red_hubble_debiased, _ = reduced_chi_squared(
        debiased_residuals,
        mu_pred_std_debiased_with_scatter,
        n_params=len(model_labels)-1,
    )



    print("Plotting predicted vs actual M2500...")
    plot_predicted_vs_actual_M2500(flat_samples, df_agn, cosmo_model=cosmo_model, z_pivot_agn=z_pivot_agn, debias=False, show=False, plot_path=plot_path)
    M2500_residuals_debiased, M2500_std_debiased, M2500_binned_residuals_debiased, _ = plot_predicted_vs_actual_M2500(flat_samples, df_agn, cosmo_model=cosmo_model, 
                                                                                  z_pivot_agn=z_pivot_agn, debias=True, show=False, dm_interp=dm_interp,
                                                                                  plot_path=plot_path)
    chisq_red_M2500_debiased, _ = reduced_chi_squared(M2500_residuals_debiased, M2500_std_debiased, n_params=len(model_labels)-1)
    print("Plotting debiased residuals...")
    plot_full_residuals(df_agn, debiased_residuals, debiased_residuals_err, flat_samples, cosmo_model, z_pivot_agn, debias=True, dm_interp=dm_interp, show=False, plot_path=plot_path, z_range=z_range)
    plot_full_residuals(
        df_agn,
        L_residuals_debiased,
        L_pred_std_debiased,
        flat_samples,
        cosmo_model,
        z_pivot_agn,
        debias=True,
        dm_interp=dm_interp,
        show=False,
        plot_path=plot_path,
        z_range=z_range,
        residual_label='L2500_sigma_tau_residuals',
        output_tag='full_residuals_l2500_sigma_tau',
    )
    plot_full_residuals(df_agn, debiased_residuals, debiased_residuals_err, flat_samples, cosmo_model, z_pivot_agn, debias=True, dm_interp=dm_interp, 
                        show=False, plot_path=plot_path, key_y='z', key_color='residuals', z_range=z_range)
    plot_full_residuals_rz(
        df_agn,
        debiased_residuals,
        debiased_residuals_err,
        flat_samples,
        cosmo_model,
        z_pivot_agn,
        debias=True,
        dm_interp=dm_interp,
        show=False,
        plot_path=plot_path,
        z_range=z_range,
    )
    plot_debias_impact_diagnostics(
        df_agn,
        biased_residuals,
        debiased_residuals,
        plot_path=plot_path,
        show=False,
    )
    plot_redshift_bin_residual_summary(
        df_agn,
        biased_residuals,
        biased_residuals_err,
        debiased_residuals,
        debiased_residuals_err,
        plot_path=plot_path,
        show=False,
    )
    plot_fast_vs_uv_variability(df_agn, plot_path=plot_path, show=False)

    
    print("Plotting cosmological posteriors corner plot...")
    plot_cosmo_corner(None, flat_samples, cosmo_model, z_pivot_sna, z_pivot_agn, show=False, 
                      plot_path=plot_path, speed=speed,
                      gauss_sigma=1.5, kde_bw_scale=1.5)

    if completeness:
        if completeness_mode == "4d_fhost_alpha":
            print("Plotting host-aware/color-aware 4D completeness diagnostics...")
            get_completeness_function_4d_fhost_alpha(
                df_agn, sim_file=completeness_sim_file, plot=True, plot_path=plot_path
            )
        elif completeness_mode == "3d_fhost":
            print("Plotting host-aware 3D completeness diagnostics...")
            get_completeness_function_3d_fhost(
                df_agn, sim_file=completeness_sim_file, plot=True, plot_path=plot_path
            )
        else:
            print("Plotting completeness vs magnitude at redshifts...")
            p_detect, mag_centers, z_centers, dm, dz, completeness_scatter = get_completeness_function_2d(
                df_agn, sim_file=completeness_sim_file, plot=True, plot_path=plot_path
            )
            plot_completeness_vs_mag_at_redshifts(
                p_detect, mag_centers, z_centers, plot_path=plot_path
            )

    # TODO: Subtract typical mu error in quadrature
    
    print(f"\033[94mReduced chi-squared (debiased) M2500: {chisq_red_M2500_debiased:.3f}\033[0m")
    print(f"\033[94mReduced chi-squared (debiased) Hubble: {chisq_red_hubble_debiased:.3f}\033[0m")
    print(f"\033[94mReduced chi-squared (debiased) L2500: {chisq_red_L2500:.3f}\033[0m")
    chisq_dict = {
        'M2500': chisq_red_M2500_debiased,
        'Hubble': chisq_red_hubble_debiased,
        'L2500': chisq_red_L2500
    }

    plot_residuals_vs_alphaOX(df_agn, debiased_residuals, debiased_residuals_err, show=False, plot_path=plot_path)

    return flat_samples, model_labels, dm_interp, logZ, logZerr, debiased_residuals, age, age_err


def run_all(df_agn, df_agn_all, df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov, 
            cosmo_models, skip_plots=False,
            residuals_sigma_clip=None,
            z_range=(0.44, 3.16),
            speed="production", resume=False, N=None,
            prefix="default", result_prefix="", uniform_redshift_distribution=False,
            completeness_sim_file=DEFAULT_COMPLETENESS_SIM_FILE,
            completeness_mode="2d",
            use_alpha_lambda_term=False,
            use_redshift_log_f_term=False):

    zmin, zmax = z_range
    n_tag = "all" if N is None else f"N{N}"
    z_tag = f"z{zmin:.2f}_{zmax:.2f}".replace(".", "p")
    compare_run_tag = f"model_compare_{speed}_{n_tag}_{z_tag}"
    compare_plot_path = f"plots/hubble/{prefix}/{compare_run_tag}"
    os.makedirs(compare_plot_path, exist_ok=True)

    cosmo_models_result_dict = {k: {} for k in cosmo_models}
    cosmo_models_sna_result_dict = {k: {} for k in cosmo_models}
    results_latex = []
    cosmo_model_joint_samples = {}
    cosmo_model_sna_samples = {}
    for cosmo_model in cosmo_models:
        r = run_single(df_agn, df_agn_all, df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov, 
                       cosmo_model=cosmo_model, only_sna=False, 
                       resume=resume, speed=speed, N=N,
                       skip_plots=skip_plots,
                       residuals_sigma_clip=residuals_sigma_clip,
                       z_range=z_range,
                       cosmo_model_joint_samples=cosmo_model_joint_samples,
                       prefix=prefix, uniform_redshift_distribution=uniform_redshift_distribution,
                       completeness_sim_file=completeness_sim_file,
                       completeness_mode=completeness_mode,
                       use_alpha_lambda_term=use_alpha_lambda_term,
                       use_redshift_log_f_term=use_redshift_log_f_term)
        
        samples_joint, model_labels_joint, dm_interp_joint, logZ_joint, logZerr_joint, debiased_residuals_joint, age_joint, age_err_joint = r
        #print(f"For model {cosmo_model}, universe age: {age:.3f} Gyr")
        r = run_single(df_agn, df_agn_all, df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov, 
                       cosmo_model=cosmo_model, only_sna=True, 
                       skip_plots=skip_plots,
                       residuals_sigma_clip=residuals_sigma_clip,
                       z_range=z_range,
                       resume=resume, speed=speed, N=N,
                       prefix=prefix, uniform_redshift_distribution=uniform_redshift_distribution,
                       completeness_sim_file=completeness_sim_file,
                       completeness_mode=completeness_mode,
                       use_alpha_lambda_term=use_alpha_lambda_term,
                       use_redshift_log_f_term=use_redshift_log_f_term)
        samples_sna, model_labels_sna, dm_interp_sna, logZ_sna, logZerr_sna, debiased_residuals_sna, age_sna, age_sna_err = r
        
        plot_cosmo_corner(samples_sna, samples_joint, cosmo_model, z_pivot_sna, z_pivot_agn, show=False, 
                          plot_path=compare_plot_path, speed=speed,
                          gauss_sigma=1.5, kde_bw_scale=1.5, include_alpha_beta=False)
        plot_cosmo_corner(samples_sna, samples_joint, cosmo_model, z_pivot_sna, z_pivot_agn, show=False, 
                          plot_path=compare_plot_path, speed=speed,
                          gauss_sigma=1.5, kde_bw_scale=1.5, include_alpha_beta=True)
        
        cosmo_models_result_dict[cosmo_model]['logZ'] = logZ_joint
        cosmo_models_result_dict[cosmo_model]['logZerr'] = logZerr_joint
        cosmo_models_result_dict[cosmo_model]['age'] = age_joint
        cosmo_models_result_dict[cosmo_model]['age_err'] = age_err_joint
        cosmo_models_sna_result_dict[cosmo_model]['logZ'] = logZ_sna
        cosmo_models_sna_result_dict[cosmo_model]['logZerr'] = logZerr_sna
        cosmo_models_sna_result_dict[cosmo_model]['age'] = age_sna
        cosmo_models_sna_result_dict[cosmo_model]['age_err'] = age_sna_err



        r_sna   = extract_cosmo_results_from_samples(samples_sna, cosmo_model, True,  
                                                    logZ_tuple=(logZ_sna, logZerr_sna), format_for_latex=True, value_fmt="{:.2f}")
        r_joint   = extract_cosmo_results_from_samples(samples_joint, cosmo_model, False,  
                                                    logZ_tuple=(logZ_joint, logZerr_joint), format_for_latex=True, value_fmt="{:.2f}")

        cosmo_model_joint_samples[cosmo_model] = samples_joint
        cosmo_model_sna_samples[cosmo_model] = samples_sna
        results_latex.extend([r_sna, r_joint])

        cosmo_models_result_dict[cosmo_model] |= dict(N=N, z_i=z_range[0], z_f=z_range[1])

        for i, key in enumerate(model_labels_joint):
            median, err, lower, upper = sym_percentile(samples_joint[:, i])
            cosmo_models_result_dict[cosmo_model][key] = median
            cosmo_models_result_dict[cosmo_model][f"{key}_err"] = err
            cosmo_models_result_dict[cosmo_model][f"{key}_err_lower"] = lower
            cosmo_models_result_dict[cosmo_model][f"{key}_err_upper"] = upper

        for i, key in enumerate(model_labels_sna):
            median, err, lower, upper = sym_percentile(samples_sna[:, i])
            cosmo_models_sna_result_dict[cosmo_model][key] = median
            cosmo_models_sna_result_dict[cosmo_model][f"{key}_err"] = err
            cosmo_models_sna_result_dict[cosmo_model][f"{key}_err_lower"] = lower
            cosmo_models_sna_result_dict[cosmo_model][f"{key}_err_upper"] = upper

    compare_r = compare_models_by_log_evidence_all(df_agn, cosmo_models_result_dict, write_path=f"{compare_plot_path}/")
    is_calib_bool = np.asarray(df_pantheon['IS_CALIBRATOR'], dtype=bool)
    sn_fit_mask = (df_pantheon['zHD'] > 0.01) | is_calib_bool
    compare_r_sna = compare_models_by_log_evidence_all(
        df_agn,
        cosmo_models_sna_result_dict,
        write_path=f"{compare_plot_path}/",
        sample_label="SNe Ia",
        sample_count=int(np.count_nonzero(sn_fit_mask)),
        output_filename="compare_all_models_sn_only.txt",
    )
    write_results_tex_variables(df_agn, df_agn_all, df_pantheon, z_range, 
                                cosmo_model_joint_samples, cosmo_model_sna_samples, 
                                compare_r, compare_plot_path, 
                                result_prefix=result_prefix, cosmo_models_result_dict=cosmo_models_result_dict,
                                cosmo_models_sna_result_dict=cosmo_models_sna_result_dict,
                                compare_r_sna=compare_r_sna)

    cosmo_output_dir = get_qvc_result_dir() / "cosmo" / prefix
    cosmo_output_dir.mkdir(parents=True, exist_ok=True)
    save_cosmo_results_hdf5(
        str(cosmo_output_dir / f"cosmo_results_{n_tag}_{z_tag}.hdf5"),
        cosmo_models_result_dict
    )
    
    print("================================================================\n\n")
    return cosmo_models_result_dict, cosmo_model_joint_samples, results_latex, compare_r

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
    parser.add_argument("--spectra_fit_csv", type=str, nargs='+', help="Path(s) to spectra fit CSV file(s)")
    parser.add_argument("--no_cuts", action="store_true", default=False, help="Disable AGN data cuts (default: False)")
    parser.add_argument("--skip_plots", action="store_true", default=False, help="Skip plotting steps (default: False)")
    parser.add_argument(
        "--fhost_cut",
        type=float,
        default=DEFAULT_F_HOST_CUT,
        help=f"Upper limit for f_host_center in the default AGN cuts (default: {DEFAULT_F_HOST_CUT})",
    )
    parser.add_argument("--exclude_object_ids_csv", type=str, nargs='+', default=[], help="Path(s) to CSV file(s) containing object IDs to exclude")
    parser.add_argument("--residuals_sigma_clip", type=float, default=None, help="Optional residual cut value to exclude outliers (default: None)")
    parser.add_argument("--residuals_csv", type=str, default=None, help="Path to CSV file containing residuals for outlier exclusion (default: None)")
    parser.add_argument("--agn_calibrators", type=str, default=None, help="Path to H5 or CSV file containing AGN data to use as calibrators (default: None)")
    parser.add_argument("--wrms_cut", type=float, default=DEFAULT_WRMS_CUT, help="Optional reduced chi-squared cut value to exclude outliers (default: None)")
    parser.add_argument("--iron_frac_cut", type=float, default=DEFAULT_IRON_FRAC_CUT, help="Optional iron fraction cut value to exclude outliers (default: None)")
    parser.add_argument("--bc_frac_cut", type=float, default=DEFAULT_BC_FRAC_CUT, help="Optional BC cut value to exclude outliers (default: None)")
    parser.add_argument(
        "--variability_chi_sq_cut",
        type=float,
        default=DEFAULT_CHI_SQ_CUT,
        help="Optional reduced g-band chi-squared cut value; keeps rows with variability_chi_sq_red_g >= cut.",
    )
    parser.add_argument(
        "--reddening_ebv_cut",
        type=float,
        default=DEFAULT_REDDENING_EBV_CUT,
        help="Optional upper limit on reddening_ebv to exclude reddened objects (default: disabled).",
    )
    parser.add_argument("--prefix", type=str, default="default", help="Prefix directory under plots/hubble/ and results/, and result variable prefix.")
    parser.add_argument("--result_prefix", type=str, default="", help="Prefix for result variable names in LaTeX output (default: empty string)")
    parser.add_argument("--z_range", type=float, nargs=2, default=[0.44, 3.16], 
                        help="Redshift range for AGN data (default: [0.44, 3.16])")
    parser.add_argument("--uniform_redshift_distribution", action="store_true", default=False, help="Select AGN subset with uniform redshift distribution (default: False)")
    parser.add_argument(
        "--completeness_sim_file",
        type=str,
        default=DEFAULT_COMPLETENESS_SIM_FILE,
        help="Mock catalog HDF5 file to use when building the completeness map.",
    )
    parser.add_argument(
        "--completeness_mode",
        type=str,
        choices=list(VALID_COMPLETENESS_MODES),
        default="2d",
        help="Completeness model to use: 2D p(det|m,z), 3D p(det|m,z,f_host_center), or 4D p(det|m,z,f_host_center,alpha_lambda).",
    )
    parser.add_argument(
        "--correct-sigma-uv-host",
        action="store_true",
        default=False,
        help="Correct log_sigma_uv using f_host_center, propagate f_host_center_err into log_sigma_uv_std_psd, and save diagnostics plots.",
    )
    parser.add_argument(
        "--fit_alpha_lambda_term",
        action="store_true",
        default=False,
        help="Fit an additional linear alpha_lambda term in the AGN standardization relation.",
    )
    parser.add_argument(
        "--fit_redshift_log_f_term",
        action="store_true",
        default=False,
        help="Fit log_f(z) = log_f0 + gamma_f * log10((1+z)/(1+z_pivot)).",
    )
    parser.add_argument(
        "--use_jax",
        action="store_true",
        default=False,
        help="Use the experimental JAX/NumPyro nested-sampling pipeline instead of the default Dynesty pipeline.",
    )

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
    agn_plot_path = f"plots/hubble/{args.prefix}"
    df_agn, df_agn_all = load_agn_data(args.agn_data_filepath, populate_sdss=args.force_populate_fields, 
                           apply_cut=not args.no_cuts, fhost_cut=args.fhost_cut,
                           residuals_sigma_clip=args.residuals_sigma_clip, residuals_csv=args.residuals_csv,
                           exclude_object_ids_csv=args.exclude_object_ids_csv,
                           spectra_fit_csv=args.spectra_fit_csv,
                           wrms_cut=args.wrms_cut, iron_frac_cut=args.iron_frac_cut,
                           bc_frac_cut=args.bc_frac_cut,
                           variability_chi_sq_cut=args.variability_chi_sq_cut,
                           reddening_ebv_cut=args.reddening_ebv_cut,
                           correct_sigma_uv_host=args.correct_sigma_uv_host,
                           z_range=tuple(args.z_range), plot_path=agn_plot_path)
    if args.N is not None:
        df_agn = df_agn.sample(n=args.N, random_state=42)
    if args.agn_calibrators:
        if args.agn_calibrators.endswith('.h5'):
            df_calibrators = read_quasars_from_hdf5_flat(args.agn_calibrators)
        elif args.agn_calibrators.endswith('.csv'):
            df_calibrators = pd.read_csv(args.agn_calibrators)
        else:
            raise ValueError("Unsupported file format for agn_calibrators. Use .h5 or .csv")
    else:
        df_calibrators = None

    if args.use_jax:
        if args.run != "single":
            raise NotImplementedError("--use_jax currently supports only --run single.")
        if args.resume:
            raise NotImplementedError("--use_jax does not support --resume yet.")
        if args.agn_calibrators is not None:
            raise NotImplementedError("--use_jax does not support --agn_calibrators yet.")
        from qvc.hubble.hubble_fit_jax import run_single_jax

        for cosmo_model in args.cosmo_models:
            run_single_jax(
                df_agn=df_agn,
                df_agn_all=df_agn_all,
                df_pantheon=df_pantheon,
                _sna_L=_sna_L,
                _sna_Lower=_sna_Lower,
                _sna_LogdetCov=_sna_LogdetCov,
                cosmo_model=cosmo_model,
                completeness=not args.disable_completeness,
                z_range=tuple(args.z_range),
                speed=args.speed,
                prefix=args.prefix,
                completeness_sim_file=args.completeness_sim_file,
                completeness_mode=args.completeness_mode,
                only_sna=args.only_sna,
                N=args.N,
                uniform_redshift_distribution=args.uniform_redshift_distribution,
                use_alpha_lambda_term=args.fit_alpha_lambda_term,
                use_redshift_log_f_term=args.fit_redshift_log_f_term,
            )
    elif args.run == "single": # default
        cosmo_models_dict = {k: {} for k in args.cosmo_models}
        for cosmo_model in args.cosmo_models:
            r = run_single(df_agn=df_agn, df_agn_all=df_agn_all, df_pantheon=df_pantheon, _sna_L=_sna_L, _sna_Lower=_sna_Lower, _sna_LogdetCov=_sna_LogdetCov, 
                           cosmo_model=cosmo_model,
                completeness=not args.disable_completeness, use_full_cov=not args.disable_full_covariance, resume=args.resume, z_range=args.z_range,
                speed=args.speed, N=args.N, only_sna=args.only_sna,
                skip_plots=args.skip_plots, residuals_sigma_clip=args.residuals_sigma_clip,
                df_calibrators=df_calibrators,
                prefix=args.prefix,
                completeness_sim_file=args.completeness_sim_file,
                completeness_mode=args.completeness_mode,
                use_alpha_lambda_term=args.fit_alpha_lambda_term,
                use_redshift_log_f_term=args.fit_redshift_log_f_term)
            samples_joint, model_labels, dm_interp, logZ_joint, logZerr_joint, debiased_residuals, age, age_err = r
            cosmo_models_dict[cosmo_model]['logZ'] = logZ_joint
            cosmo_models_dict[cosmo_model]['logZerr'] = logZerr_joint
            cosmo_models_dict[cosmo_model]['age'] = age
            cosmo_models_dict[cosmo_model]['age_err'] = age_err
        zmin, zmax = args.z_range
        n_tag = "all" if args.N is None else f"N{args.N}"
        z_tag = f"z{zmin:.2f}_{zmax:.2f}".replace(".", "p")
        compare_path = f"plots/hubble/{args.prefix}/single_compare_{args.speed}_{n_tag}_{z_tag}"
        os.makedirs(compare_path, exist_ok=True)
        if len(cosmo_models_dict) >= 2:
            compare_r = compare_models_by_log_evidence_all(
                df_agn,
                cosmo_models_dict,
                write_path=f"{compare_path}/",
            )
        else:
            print(
                "Skipping evidence comparison because only one cosmology model was requested: "
                f"{args.cosmo_models}"
            )
    elif args.run == "full":
        run_all(df_agn=df_agn, df_agn_all=df_agn_all, df_pantheon=df_pantheon, _sna_L=_sna_L, _sna_Lower=_sna_Lower, _sna_LogdetCov=_sna_LogdetCov, 
                cosmo_models=args.cosmo_models, skip_plots=args.skip_plots,
                residuals_sigma_clip=args.residuals_sigma_clip,
                z_range=args.z_range,
                speed=args.speed, resume=args.resume, N=args.N,
                prefix=args.prefix, result_prefix=args.result_prefix, uniform_redshift_distribution=args.uniform_redshift_distribution,
                completeness_sim_file=args.completeness_sim_file,
                completeness_mode=args.completeness_mode,
                use_alpha_lambda_term=args.fit_alpha_lambda_term,
                use_redshift_log_f_term=args.fit_redshift_log_f_term)
    
    print(f"Finished running Hubble fit pipeline for {args.cosmo_models}")
