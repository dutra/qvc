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
DEFAULT_COMPLETENESS_SIM_FILE = None
DEFAULT_COMPLETENESS_MOCK_AREA_DEG2 = 5.0

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
    plot_hubble_residual_normality,
    plot_predicted_L2500_vs_sigmahat,
    plot_predicted_vs_actual_M2500,
    plot_redshift_histograms,
    plot_redshift_bin_residual_summary,
    plot_residuals_vs_alphaOX,
    plot_sigma_uv_mpred_correction,
)
from qvc.hubble.tex_utils import make_agn_csv_table, make_agn_latex_table
from qvc.hubble.hubble_model import (
    agn_model_pack_obs,
    evaluate_log_f,
    get_agn_model_spec,
    get_model_params,
)
from qvc.hubble.hubble_completeness_refactored import (
    evaluate_dm_interp,
    get_completeness_function_2d,
    get_completeness_function_3d_fhost,
    get_completeness_function_4d_fhost_alpha,
    make_dm_function,
)
from qvc.hubble.completeness_mock_catalog import (
    COSMO as COMPLETENESS_MOCK_COSMO,
    build_shen_lf,
    mock_m_per_zbin,
    save_mock_catalog,
)

VALID_COMPLETENESS_MODES = ("2d", "3d_fhost", "4d_fhost_alpha")


def validate_completeness_mode(completeness_mode):
    if completeness_mode not in VALID_COMPLETENESS_MODES:
        raise ValueError(
            f"Invalid completeness_mode={completeness_mode!r}. "
            f"Expected one of {VALID_COMPLETENESS_MODES}."
        )


def subsample_dataframe_at_most(df, n, *, random_state=42, label="rows"):
    """Return at most ``n`` rows without crashing when ``n`` exceeds the population."""

    if n is None:
        return df, None

    n = int(n)
    if n < 0:
        raise ValueError(f"Requested sample size must be non-negative, got {n}.")

    available = len(df)
    if n >= available:
        if n > available:
            print(
                f"Requested N={n} but only {available} {label} are available after cuts; "
                f"using all {available}."
            )
        return df, available

    return df.sample(n=n, random_state=random_state), n


def prior_transform_dynesty(unit_cube, priors, model_labels):
    return [priors[key][0] + (priors[key][1] - priors[key][0]) * x
            for x, key in zip(unit_cube, model_labels)]

def make_run_tag(
    cosmo_model,
    only_sna,
    speed,
    N,
    z_range,
    completeness=True,
    disable_ceph_dist_calibration=False,
    use_alpha_lambda_term=False,
    use_eta_sigma_term=False,
    use_redshift_log_f_term=False,
):
    zmin, zmax = z_range
    n_tag = "all" if N is None else f"N{N}"
    z_tag = f"z{zmin:.2f}_{zmax:.2f}".replace(".", "p")
    completeness_tag = "" if completeness else "_disable_completeness"
    ceph_tag = "_nocephdist_planckh0" if disable_ceph_dist_calibration else ""
    alpha_tag = "_alphaLam" if use_alpha_lambda_term else ""
    eta_sigma_tag = "_etaSigma" if use_eta_sigma_term else ""
    logf_tag = "_logfz" if use_redshift_log_f_term else ""
    return (
        f"{cosmo_model}_{'sna' if only_sna else 'joint'}_{speed}_{n_tag}_{z_tag}"
        f"{completeness_tag}{ceph_tag}{alpha_tag}{eta_sigma_tag}{logf_tag}"
    )


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
    if "dmi_selection_sigma_posterior_median" in results:
        value = np.asarray(results["dmi_selection_sigma_posterior_median"])
        if value.ndim != 0 and value.shape[0] != n_agn:
            raise RuntimeError(
                f"Resume checkpoint '{checkpoint_file}' is incompatible with the current AGN selection: "
                f"dmi_selection_sigma_posterior_median has length {value.shape[0]}, "
                f"but the current run has {n_agn} AGN objects. "
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


def _infer_radec_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def estimate_sky_box_area_deg2(df_agn_all):
    """
    Estimate the survey footprint area using the smallest RA arc enclosing all
    source coordinates and a Dec bounding box.
    """
    ra_col = _infer_radec_column(df_agn_all, ("ra", "RA"))
    dec_col = _infer_radec_column(df_agn_all, ("dec", "DEC"))
    if ra_col is None or dec_col is None:
        print(
            "[WARNING] Could not estimate sky-box area from df_agn_all because "
            "RA/Dec columns are missing; using default mock area "
            f"{DEFAULT_COMPLETENESS_MOCK_AREA_DEG2:.1f} deg^2."
        )
        return DEFAULT_COMPLETENESS_MOCK_AREA_DEG2

    ra = np.mod(pd.to_numeric(df_agn_all[ra_col], errors="coerce").to_numpy(dtype=float), 360.0)
    dec = pd.to_numeric(df_agn_all[dec_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(ra) & np.isfinite(dec)
    ra = ra[finite]
    dec = dec[finite]
    if ra.size < 2:
        print(
            "[WARNING] Too few finite RA/Dec rows to estimate sky-box area; "
            f"using default mock area {DEFAULT_COMPLETENESS_MOCK_AREA_DEG2:.1f} deg^2."
        )
        return DEFAULT_COMPLETENESS_MOCK_AREA_DEG2

    ra_sorted = np.sort(ra)
    gaps = np.diff(np.concatenate([ra_sorted, [ra_sorted[0] + 360.0]]))
    largest_gap_idx = int(np.argmax(gaps))
    ra_span_deg = float(360.0 - gaps[largest_gap_idx])
    ra_min = float(ra_sorted[(largest_gap_idx + 1) % ra_sorted.size])
    ra_max = float(np.mod(ra_min + ra_span_deg, 360.0))
    dec_min = float(np.min(dec))
    dec_max = float(np.max(dec))

    area_sr = np.deg2rad(ra_span_deg) * (
        np.sin(np.deg2rad(dec_max)) - np.sin(np.deg2rad(dec_min))
    )
    area_deg2 = float(abs(area_sr) * (180.0 / np.pi) ** 2)
    if not np.isfinite(area_deg2) or area_deg2 <= 0.0:
        print(
            "[WARNING] Invalid sky-box area estimate from RA/Dec; using default "
            f"mock area {DEFAULT_COMPLETENESS_MOCK_AREA_DEG2:.1f} deg^2."
        )
        return DEFAULT_COMPLETENESS_MOCK_AREA_DEG2

    print(
        "Estimated pre-cut sky-box area from df_agn_all: "
        f"{area_deg2:.1f} deg^2 "
        f"(ra_col={ra_col}, dec_col={dec_col}, RA span={ra_span_deg:.2f} deg, "
        f"RA box={ra_min:.2f}->{ra_max:.2f} deg, Dec box={dec_min:.2f}->{dec_max:.2f} deg)"
    )
    return area_deg2


def generate_fresh_completeness_sim_file(plot_path, *, area_deg2, seed=123):
    """Generate a fresh Shen-LF mock catalog for completeness-map construction."""
    completeness_dir = Path(plot_path) / "completeness"
    completeness_dir.mkdir(parents=True, exist_ok=True)
    output_path = completeness_dir / "mock_completeness_catalog_fresh.h5"
    thinning_probability = min(
        1.0,
        float(DEFAULT_COMPLETENESS_MOCK_AREA_DEG2) / max(float(area_deg2), 1e-12),
    )

    rng = np.random.default_rng(seed)
    phi_log10, m_grid, z_bins = build_shen_lf(None)
    _, _, _, _, z_all, m_all, m_rest_all, _ = mock_m_per_zbin(
        phi_log10,
        m_grid,
        z_bins,
        float(area_deg2),
        -0.5,
        0.3,
        COMPLETENESS_MOCK_COSMO,
        z_res=512,
        m_scatter=0.0,
        kcorr_zref=2.0,
        m_lim=28.0,
        thinning_probability=thinning_probability,
        rng=rng,
        return_z=True,
        return_global=True,
    )
    n_generated = int(np.size(z_all))
    print(
        f"Fresh completeness mock generated {n_generated} sources "
        f"after in-generator thinning (p_keep={thinning_probability:.4g})."
    )
    save_mock_catalog(
        output_path,
        z_all,
        m_all,
        m_rest_all,
        m_limit=28.0,
        thinning_probability=thinning_probability,
        rng=rng,
        area_deg2=area_deg2,
    )
    print(
        f"Generated fresh completeness mock catalog: {output_path} "
        f"(area_deg2={float(area_deg2):.1f}, p_keep={thinning_probability:.4g})"
    )
    return str(output_path)

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
                      compare_sigma_only=False,
                      disable_ceph_dist_calibration=False,
                      use_alpha_lambda_term=False,
                      use_eta_sigma_term=False,
                      use_redshift_log_f_term=False,
                      ):
    validate_completeness_mode(completeness_mode)
    run_tag = make_run_tag(
        cosmo_model,
        only_sna,
        speed,
        N,
        z_range,
        completeness=completeness,
        disable_ceph_dist_calibration=disable_ceph_dist_calibration,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    plot_path = f"plots/hubble/{prefix}/{run_tag}"
    os.makedirs(plot_path, exist_ok=True)

    priors, model_labels, model_labels_latex = get_model_params(
        cosmo_model,
        only_sna=only_sna,
        use_planck_h0_prior=disable_ceph_dist_calibration,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
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
    if use_eta_sigma_term:
        for required_col in ("eta_sigma", "eta_sigma_err"):
            if required_col not in df_agn.columns:
                raise KeyError(f"--fit_eta_sigma_term requires df_agn[{required_col!r}].")
            bad = ~np.isfinite(df_agn[required_col].to_numpy(dtype=float))
            if np.any(bad):
                raise ValueError(
                    f"--fit_eta_sigma_term requires finite {required_col} for all AGN used in the fit; "
                    f"found {np.count_nonzero(bad)} non-finite rows."
                )

    if completeness:
        if completeness_sim_file is None:
            completeness_area_deg2 = estimate_sky_box_area_deg2(df_agn_all)
            completeness_sim_file = generate_fresh_completeness_sim_file(
                plot_path,
                area_deg2=completeness_area_deg2,
            )
        if completeness_mode in ("3d_fhost", "4d_fhost_alpha"):
            if "f_host_2500" not in df_agn.columns:
                raise KeyError(f"completeness_mode={completeness_mode!r} requires df_agn['f_host_2500'].")
            bad_fhost = ~np.isfinite(df_agn["f_host_2500"].to_numpy(dtype=float))
            if np.any(bad_fhost):
                raise ValueError(
                    f"completeness_mode={completeness_mode!r} requires finite f_host_2500 for all AGN used in the fit; "
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
        print(f"Building {completeness_mode} completeness map using mock catalog: {completeness_sim_file}")
        if completeness_mode == "4d_fhost_alpha":
            completeness_params = get_completeness_function_4d_fhost_alpha(
                df_agn, sim_file=completeness_sim_file, plot=not compare_sigma_only, plot_path=plot_path
            )
        elif completeness_mode == "3d_fhost":
            completeness_params = get_completeness_function_3d_fhost(
                df_agn, sim_file=completeness_sim_file, plot=not compare_sigma_only, plot_path=plot_path
            )
        else:
            completeness_params = get_completeness_function_2d(
                df_agn, sim_file=completeness_sim_file, plot=not compare_sigma_only, plot_path=plot_path
            )
    else:
        completeness_params = None

    agn_model_req_params, agn_model_req_obs, agn_model_req_errs = get_agn_model_spec(
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    agn_fields = agn_model_req_params + agn_model_req_obs + agn_model_req_errs
    agn_fields += ('apparent_mag_2500', 'apparent_mag_2500_err', 'z', 'z_err', 'object_id')
    if 'f_host_2500' in df_agn.columns:
        agn_fields += ('f_host_2500',)
    if 'alpha_lambda' in df_agn.columns:
        agn_fields += ('alpha_lambda',)
    if 'eta_sigma' in df_agn.columns:
        agn_fields += ('eta_sigma',)
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
        dmi_selection_sigma_posterior_median = r.get("dmi_selection_sigma_posterior_median")
        if dmi_selection_sigma_posterior_median is not None:
            dmi_selection_sigma_posterior_median = np.asarray(
                dmi_selection_sigma_posterior_median,
                dtype=float,
            )
            if (
                dmi_selection_sigma_posterior_median.ndim == 0
                and not np.isfinite(dmi_selection_sigma_posterior_median)
            ):
                dmi_selection_sigma_posterior_median = None
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
                use_planck_h0_prior=disable_ceph_dist_calibration,
                use_ceph_dist_calibration=not disable_ceph_dist_calibration,
                use_alpha_lambda_term=use_alpha_lambda_term,
                use_eta_sigma_term=use_eta_sigma_term,
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
                    nlive_batch=10   # reasonable batch size for dynamic allocation
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
        if compare_sigma_only:
            print("compare_sigma_only=True: skipping dynesty plot generation.")
        else:
            print("Plotting full dynesty corner...")
            plot_dynesty(
                sampler.results,
                cosmo_model,
                plot_path,
                only_sna=only_sna,
                speed=speed,
                use_alpha_lambda_term=use_alpha_lambda_term,
                use_eta_sigma_term=use_eta_sigma_term,
                use_redshift_log_f_term=use_redshift_log_f_term,
            )
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
        if completeness and flat_blobs.ndim == 3 and flat_blobs.shape[1] >= 3:
            dmi_selection_sigma_posterior_median = np.median(flat_blobs[:, 2, :], axis=0)
        else:
            dmi_selection_sigma_posterior_median = None
        
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
        if dmi_selection_sigma_posterior_median is not None:
            print(
                "  median sigma_sel:",
                float(np.nanmedian(dmi_selection_sigma_posterior_median)),
            )

        save_chains(
            checkpoint_file,
            flat_samples=flat_samples,
            dmi_max_w=dmi_max_w,
            dmi_posterior_median=dmi_posterior_median,
            dmi_posterior_sigma=dmi_posterior_sigma,
            dmi_selection_sigma_posterior_median=dmi_selection_sigma_posterior_median,
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
        f_host_2500=df_agn["f_host_2500"].values if "f_host_2500" in df_agn.columns else None,
        alpha_lambda=df_agn["alpha_lambda"].values if "alpha_lambda" in df_agn.columns else None,
    )
    dmi_selection_sigma_interp = None
    if dmi_selection_sigma_posterior_median is not None:
        dmi_selection_sigma_interp = make_dm_function(
            df_agn['apparent_mag_2500'].values,
            df_agn['z'].values,
            dmi_selection_sigma_posterior_median,
            f_host_2500=df_agn["f_host_2500"].values if "f_host_2500" in df_agn.columns else None,
            alpha_lambda=df_agn["alpha_lambda"].values if "alpha_lambda" in df_agn.columns else None,
        )

    if compare_sigma_only:
        print("compare_sigma_only=True: skipping completeness diagnostics plots.")
    else:
        print("Plotting completeness diagnostics...")
        plot_completeness_diagnostics(
            dmi_posterior_median,
            agn_data['z'],
            agn_data['apparent_mag_2500'],
            integrals_max_w,
            plot_path=plot_path,
        )

    return (
        flat_samples,
        model_labels,
        dm_interp,
        dmi_selection_sigma_interp,
        logZ,
        logZerr,
        dmi_posterior_median,
        dmi_posterior_sigma,
        dmi_selection_sigma_posterior_median,
    )


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
               compare_sigma_only=False,
               disable_ceph_dist_calibration=False,
               use_alpha_lambda_term=False,
               use_eta_sigma_term=False,
               use_redshift_log_f_term=False):
    validate_completeness_mode(completeness_mode)
    run_tag = make_run_tag(
        cosmo_model,
        only_sna,
        speed,
        N,
        z_range,
        completeness=completeness,
        disable_ceph_dist_calibration=disable_ceph_dist_calibration,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    plot_path = f"plots/hubble/{prefix}/{run_tag}"
    os.makedirs(plot_path, exist_ok=True)
    print(f"Saving plots to ", plot_path)
    if completeness:
        if completeness_sim_file is None:
            print("Completeness enabled with a freshly generated mock catalog.")
            completeness_area_deg2 = estimate_sky_box_area_deg2(df_agn_all)
            completeness_sim_file = generate_fresh_completeness_sim_file(
                plot_path,
                area_deg2=completeness_area_deg2,
            )
        else:
            print(f"Completeness enabled with mock catalog file: {completeness_sim_file}")

    if uniform_redshift_distribution:
        df_agn_fit_selection = select_agn_subset_uniform_with_replacement(
            df_agn,
            z_range=z_range,
            N=N,
        )
        if not compare_sigma_only:
            plot_redshift_histograms(df_pantheon, df_agn_fit_selection, xscale="linear", plot_path=plot_path)
    else:
        df_agn_fit_selection = df_agn[df_agn["z"].between(z_range[0], z_range[1])].copy()
        if not compare_sigma_only:
            plot_redshift_histograms(df_pantheon, df_agn, xscale="log", plot_path=plot_path)

    if not compare_sigma_only:
        plot_delta_m_flux_recal_vs_redshift(df_agn_fit_selection, plot_path=plot_path)

    report_pivots(df_agn_fit_selection)

    (
        flat_samples,
        model_labels,
        dm_interp,
        dmi_selection_sigma_interp,
        logZ,
        logZerr,
        dmi_posterior_median,
        dmi_posterior_sigma,
        dmi_selection_sigma_posterior_median,
    ) = run_mcmc_pipeline(
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
                                                        compare_sigma_only=compare_sigma_only,
                                                        disable_ceph_dist_calibration=disable_ceph_dist_calibration,
                                                        use_alpha_lambda_term=use_alpha_lambda_term,
                                                        use_eta_sigma_term=use_eta_sigma_term,
                                                        use_redshift_log_f_term=use_redshift_log_f_term)
    display_results_summary(
        flat_samples,
        cosmo_model,
        z_pivot_agn,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
        sigma_sel_posterior_median=dmi_selection_sigma_posterior_median,
    )
    print("Computing age of the universe with error propagation...")
    age, age_err = compute_age_universe_with_error(
        flat_samples,
        cosmo_model,
        max_eval=200,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )

    if compare_sigma_only or skip_plots or only_sna:
        print("Skipping plots, returning results...")
        return flat_samples, model_labels, dm_interp, logZ, logZerr, None, age, age_err

    # if only_sna:
    #     print("Skipping AGN-specific plots for SNe-only run.")
    #     return flat_samples, model_labels, dm_interp, logZ, logZerr, None, age, age_err

    alpha_agn_idx = model_labels.index("alpha_agn")
    alpha_agn_median = float(np.nanmedian(flat_samples[:, alpha_agn_idx]))
    plot_sigma_uv_mpred_correction(
        df_agn,
        alpha_agn_median,
        plot_path=plot_path,
        show=False,
        filename="sigma_uv_mpred_correction_postcut.pdf",
    )

    print("Plotting predicted L2500 vs ...")
    dmi_posterior_median_full = None
    if uniform_redshift_distribution:
        print(
            "Uniform-redshift selection uses resampling with replacement; "
            "using dm_interp-only debiasing in L2500 plots."
        )
    else:
        if len(df_agn_fit_selection) != len(dmi_posterior_median):
            raise ValueError(
                "Fit/plot alignment failure: "
                f"df_agn_fit_selection has length {len(df_agn_fit_selection)}, "
                f"but dmi_posterior_median has length {len(dmi_posterior_median)}."
            )
        fit_indices = pd.Index(df_agn_fit_selection.index)
        if not fit_indices.isin(df_agn.index).all():
            missing = fit_indices[~fit_indices.isin(df_agn.index)].tolist()[:10]
            raise ValueError(
                "Fit/plot alignment failure: fitted AGN selection contains index values "
                f"not present in df_agn: {missing}"
            )
        dmi_posterior_median_full = np.full(len(df_agn), np.nan, dtype=float)
        df_agn_index_positions = pd.Series(np.arange(len(df_agn)), index=df_agn.index)
        dmi_posterior_median_full[df_agn_index_positions.loc[fit_indices].to_numpy()] = np.asarray(
            dmi_posterior_median,
            dtype=float,
        )

    plot_predicted_L2500_vs_sigmahat(
        flat_samples,
        df_agn,
        cosmo_model=cosmo_model,
        z_pivot_agn=z_pivot_agn,
        debias=False,
        show_residuals=False,
        show=False,
        plot_path=plot_path,
        df_calibrators=df_calibrators,
        z_range=z_range,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    plot_predicted_L2500_vs_sigmahat(
        flat_samples,
        df_agn,
        cosmo_model=cosmo_model,
        z_pivot_agn=z_pivot_agn,
        debias=False,
        show_residuals=True,
        show=False,
        plot_path=plot_path,
        df_calibrators=df_calibrators,
        z_range=z_range,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    plot_predicted_L2500_vs_sigmahat(
        flat_samples,
        df_agn,
        cosmo_model=cosmo_model,
        z_pivot_agn=z_pivot_agn,
        debias=True,
        dm_interp=dm_interp,
        dmi_values=dmi_posterior_median_full,
        dmi_selection_sigma_interp=dmi_selection_sigma_interp,
        show_residuals=False,
        show=False,
        plot_path=plot_path,
        df_calibrators=df_calibrators,
        z_range=z_range,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    L_residuals_debiased, L_pred_std_debiased = plot_predicted_L2500_vs_sigmahat(
        flat_samples,
        df_agn,
        cosmo_model=cosmo_model,
        z_pivot_agn=z_pivot_agn,
        debias=True,
        dm_interp=dm_interp,
        dmi_values=dmi_posterior_median_full,
        dmi_selection_sigma_interp=dmi_selection_sigma_interp,
        show_residuals=True,
        show=False,
        plot_path=plot_path,
        df_calibrators=df_calibrators,
        z_range=z_range,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )

    plot_blr_line_lags_vs_l2500(
        flat_samples,
        df_agn,
        cosmo_model,
        z_pivot_agn,
        dm_interp,
        plot_path=plot_path,
        show=False,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    
    chisq_red_L2500, _ = reduced_chi_squared(L_residuals_debiased, L_pred_std_debiased, n_params=len(model_labels)-1)

    print("Plotting Hubble diagram...")
    dmi_posterior_sigma_full = None
    dmi_selection_sigma_full = None
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
        if dmi_selection_sigma_posterior_median is not None:
            dmi_selection_sigma_full = np.full(len(df_agn), np.nan, dtype=float)
            dmi_selection_sigma_full[df_agn_index_positions.loc[fit_indices].to_numpy()] = np.asarray(
                dmi_selection_sigma_posterior_median,
                dtype=float,
            )
    if dmi_selection_sigma_interp is not None:
        interp_cols = [
            np.asarray(df_agn["z"].values, dtype=float),
            np.asarray(df_agn["apparent_mag_2500"].values, dtype=float),
        ]
        if "f_host_2500" in df_agn.columns:
            interp_cols.append(np.asarray(df_agn["f_host_2500"].values, dtype=float))
            if "alpha_lambda" in df_agn.columns:
                interp_cols.append(np.asarray(df_agn["alpha_lambda"].values, dtype=float))
        dmi_selection_sigma_full = np.asarray(
            dmi_selection_sigma_interp(np.column_stack(interp_cols)),
            dtype=float,
        )
    if dmi_posterior_median_full is not None:
        dmi_posterior_median_full_plot = np.where(
            np.isfinite(dmi_posterior_median_full),
            dmi_posterior_median_full,
            evaluate_dm_interp(
                dm_interp,
                df_agn["z"].values,
                df_agn["apparent_mag_2500"].values,
                f_host_2500=df_agn["f_host_2500"].values if "f_host_2500" in df_agn.columns else None,
                alpha_lambda=df_agn["alpha_lambda"].values if "alpha_lambda" in df_agn.columns else None,
            ),
        )
        plot_completeness_diagnostics(
            dmi_posterior_median_full_plot,
            df_agn["z"].values,
            df_agn["apparent_mag_2500"].values,
            integrals_max_w=None,
            plot_path=plot_path,
            z_range=z_range,
        )
    # Debiased (Bias corrected)
    r = plot_hubble(flat_samples, df_agn, df_pantheon, 
                    cosmo_model=cosmo_model, z_pivot_agn=z_pivot_agn, 
                    show_true=False, show=False, debias=True, dm_interp=dm_interp, plot_path=plot_path,
                    cosmo_model_samples=cosmo_model_joint_samples, verbose=verbose, residuals_sigma_clip=residuals_sigma_clip,
                    df_calibrators=df_calibrators,
                    dmi_values=dmi_posterior_median_full,
                    dmi_sigma=dmi_posterior_sigma_full,
                    dmi_selection_sigma=dmi_selection_sigma_full,
                    use_alpha_lambda_term=use_alpha_lambda_term,
                    use_eta_sigma_term=use_eta_sigma_term,
                    use_redshift_log_f_term=use_redshift_log_f_term)
    debiased_residuals, debiased_residuals_err, mu_pred_median_debiased, mu_pred_std_debiased, mu_pred_std_debiased_with_scatter = r
    if cosmo_model == "Flatw0waCDM":
        make_agn_csv_table(
            df_agn,
            mu_pred_median_debiased,
            mu_pred_std_debiased_with_scatter,
            dm_interp,
            sort_by="z",
            ascending=True,
            write_path=plot_path,
        )
        make_agn_latex_table(
            df_agn,
            mu_pred_median_debiased,
            mu_pred_std_debiased_with_scatter,
            dm_interp,
            sort_by="z",
            ascending=True,
            max_rows=30,
            write_path=plot_path,
        )
    # Biased
    r = plot_hubble(flat_samples, df_agn, df_pantheon, 
                cosmo_model=cosmo_model, z_pivot_agn=z_pivot_agn, show_residuals=True,
                show_true=False, show=False, debias=False, plot_path=plot_path, verbose=False,
                use_alpha_lambda_term=use_alpha_lambda_term,
                use_eta_sigma_term=use_eta_sigma_term,
                use_redshift_log_f_term=use_redshift_log_f_term)
    biased_residuals, biased_residuals_err, _, _, _ = r

    chisq_red_hubble_debiased, _ = reduced_chi_squared(
        debiased_residuals,
        mu_pred_std_debiased_with_scatter,
        n_params=len(model_labels)-1,
    )
    plot_hubble_residual_normality(
        debiased_residuals,
        mu_pred_std_debiased_with_scatter,
        plot_path=plot_path,
        show=False,
        filename="hubble_residual_normality_debiased.pdf",
    )



    print("Plotting predicted vs actual M2500...")
    plot_predicted_vs_actual_M2500(
        flat_samples,
        df_agn,
        cosmo_model=cosmo_model,
        z_pivot_agn=z_pivot_agn,
        debias=False,
        show=False,
        plot_path=plot_path,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    M2500_residuals_debiased, M2500_std_debiased, M2500_binned_residuals_debiased, _ = plot_predicted_vs_actual_M2500(
        flat_samples,
        df_agn,
        cosmo_model=cosmo_model,
        z_pivot_agn=z_pivot_agn,
        debias=True,
        show=False,
        dm_interp=dm_interp,
        dmi_selection_sigma_interp=dmi_selection_sigma_interp,
        plot_path=plot_path,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    chisq_red_M2500_debiased, _ = reduced_chi_squared(M2500_residuals_debiased, M2500_std_debiased, n_params=len(model_labels)-1)
    print("Plotting debiased residuals...")
    plot_full_residuals(
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
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    plot_full_residuals(
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
        z_cut=1.5,
        z_range=z_range,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
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
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    plot_full_residuals(
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
        key_y='z',
        key_color='residuals',
        z_range=z_range,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
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
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
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
                      gauss_sigma=1.5, kde_bw_scale=1.5,
                      use_alpha_lambda_term=use_alpha_lambda_term,
                      use_eta_sigma_term=use_eta_sigma_term,
                      use_redshift_log_f_term=use_redshift_log_f_term)

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
            completeness=True,
            prefix="default", result_prefix="", uniform_redshift_distribution=False,
            completeness_sim_file=DEFAULT_COMPLETENESS_SIM_FILE,
            completeness_mode="2d",
            compare_sigma_only=False,
            disable_ceph_dist_calibration=False,
            use_alpha_lambda_term=False,
            use_eta_sigma_term=False,
            use_redshift_log_f_term=False):

    zmin, zmax = z_range
    n_tag = "all" if N is None else f"N{N}"
    z_tag = f"z{zmin:.2f}_{zmax:.2f}".replace(".", "p")
    completeness_tag = "" if completeness else "_disable_completeness"
    ceph_tag = "_nocephdist_planckh0" if disable_ceph_dist_calibration else ""
    compare_run_tag = f"model_compare_{speed}_{n_tag}_{z_tag}{completeness_tag}{ceph_tag}"
    if use_alpha_lambda_term:
        compare_run_tag += "_alphaLam"
    if use_eta_sigma_term:
        compare_run_tag += "_etaSigma"
    if use_redshift_log_f_term:
        compare_run_tag += "_logfz"
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
                       completeness=completeness,
                       resume=resume, speed=speed, N=N,
                       skip_plots=skip_plots,
                       residuals_sigma_clip=residuals_sigma_clip,
                       z_range=z_range,
                       cosmo_model_joint_samples=cosmo_model_joint_samples,
                       prefix=prefix, uniform_redshift_distribution=uniform_redshift_distribution,
                       completeness_sim_file=completeness_sim_file,
                       completeness_mode=completeness_mode,
                       compare_sigma_only=compare_sigma_only,
                       disable_ceph_dist_calibration=disable_ceph_dist_calibration,
                       use_alpha_lambda_term=use_alpha_lambda_term,
                       use_eta_sigma_term=use_eta_sigma_term,
                       use_redshift_log_f_term=use_redshift_log_f_term)
        
        samples_joint, model_labels_joint, dm_interp_joint, logZ_joint, logZerr_joint, debiased_residuals_joint, age_joint, age_err_joint = r
        #print(f"For model {cosmo_model}, universe age: {age:.3f} Gyr")
        r = run_single(df_agn, df_agn_all, df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov, 
                       cosmo_model=cosmo_model, only_sna=True, 
                       completeness=completeness,
                       skip_plots=skip_plots,
                       residuals_sigma_clip=residuals_sigma_clip,
                       z_range=z_range,
                       resume=resume, speed=speed, N=N,
                       prefix=prefix, uniform_redshift_distribution=uniform_redshift_distribution,
                       completeness_sim_file=completeness_sim_file,
                       completeness_mode=completeness_mode,
                       compare_sigma_only=compare_sigma_only,
                       disable_ceph_dist_calibration=disable_ceph_dist_calibration,
                       use_alpha_lambda_term=use_alpha_lambda_term,
                       use_eta_sigma_term=use_eta_sigma_term,
                       use_redshift_log_f_term=use_redshift_log_f_term)
        samples_sna, model_labels_sna, dm_interp_sna, logZ_sna, logZerr_sna, debiased_residuals_sna, age_sna, age_sna_err = r
        if not compare_sigma_only:
            plot_cosmo_corner(samples_sna, samples_joint, cosmo_model, z_pivot_sna, z_pivot_agn, show=False, 
                              plot_path=compare_plot_path, speed=speed,
                              gauss_sigma=1.5, kde_bw_scale=1.5, include_alpha_beta=False,
                              use_alpha_lambda_term=use_alpha_lambda_term,
                              use_eta_sigma_term=use_eta_sigma_term,
                              use_redshift_log_f_term=use_redshift_log_f_term)
            plot_cosmo_corner(samples_sna, samples_joint, cosmo_model, z_pivot_sna, z_pivot_agn, show=False, 
                              plot_path=compare_plot_path, speed=speed,
                              gauss_sigma=1.5, kde_bw_scale=1.5, include_alpha_beta=True,
                              use_alpha_lambda_term=use_alpha_lambda_term,
                              use_eta_sigma_term=use_eta_sigma_term,
                              use_redshift_log_f_term=use_redshift_log_f_term)
        
        cosmo_models_result_dict[cosmo_model]['logZ'] = logZ_joint
        cosmo_models_result_dict[cosmo_model]['logZerr'] = logZerr_joint
        cosmo_models_result_dict[cosmo_model]['age'] = age_joint
        cosmo_models_result_dict[cosmo_model]['age_err'] = age_err_joint
        cosmo_models_sna_result_dict[cosmo_model]['logZ'] = logZ_sna
        cosmo_models_sna_result_dict[cosmo_model]['logZerr'] = logZerr_sna
        cosmo_models_sna_result_dict[cosmo_model]['age'] = age_sna
        cosmo_models_sna_result_dict[cosmo_model]['age_err'] = age_sna_err



        r_sna   = extract_cosmo_results_from_samples(
            samples_sna,
            cosmo_model,
            True,
            logZ_tuple=(logZ_sna, logZerr_sna),
            format_for_latex=True,
            value_fmt="{:.2f}",
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
            use_redshift_log_f_term=use_redshift_log_f_term,
        )
        r_joint   = extract_cosmo_results_from_samples(
            samples_joint,
            cosmo_model,
            False,
            logZ_tuple=(logZ_joint, logZerr_joint),
            format_for_latex=True,
            value_fmt="{:.2f}",
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
            use_redshift_log_f_term=use_redshift_log_f_term,
        )

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
    parser.add_argument(
        "--disable_ceph_dist_calibration",
        action="store_true",
        default=False,
        help="Disable the Pantheon calibrator CEPH_DIST replacement and switch the H0 prior to the Planck 2018 interval.",
    )
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="Resume previous MCMC run (default: False). If a string is provided, it is used as the checkpoint file.")
    parser.add_argument("--run", type=str, choices=["full", "single"], default="single", help="Run mode: compare_models, compare_sna, full, or single (default: single)")
    parser.add_argument("--speed", type=str, choices=["production", "test", "fast", "dev"], default="production", help="Sampling speed: production, test, or fast (default: production)")
    parser.add_argument("--N", type=int, default=None, help="Number of AGNs to run (default: all)")
    parser.add_argument("--only_sna", action="store_true", default=False, help="Run SNIa-only fit (default: False)")
    parser.add_argument("--spectra_fit_csv", type=str, nargs='+', help="Path(s) to spectra fit CSV file(s)")
    parser.add_argument("--no_cuts", action="store_true", default=False, help="Disable AGN data cuts (default: False)")
    parser.add_argument("--skip_plots", action="store_true", default=False, help="Skip plotting steps (default: False)")
    parser.add_argument(
        "--compare_sigma_only",
        action="store_true",
        default=False,
        help="Run the full fit and evidence calculation, but skip non-essential plots and keep only text/console model-comparison sigma outputs.",
    )
    parser.add_argument("--exclude_object_ids_csv", type=str, nargs='+', default=[], help="Path(s) to CSV file(s) containing object IDs to exclude")
    parser.add_argument("--residuals_sigma_clip", type=float, default=None, help="Optional residual cut value to exclude outliers (default: None)")
    parser.add_argument("--residuals_csv", type=str, default=None, help="Path to CSV file containing residuals for outlier exclusion (default: None)")
    parser.add_argument("--agn_calibrators", type=str, default=None, help="Path to H5 or CSV file containing AGN data to use as calibrators (default: None)")
    parser.add_argument("--prefix", type=str, default="default", help="Prefix directory under plots/hubble/ and results/, and result variable prefix.")
    parser.add_argument("--result_prefix", type=str, default="", help="Prefix for result variable names in LaTeX output (default: empty string)")
    parser.add_argument("--z_range", type=float, nargs=2, default=[0.44, 3.16], 
                        help="Redshift range for AGN data (default: [0.44, 3.16])")
    parser.add_argument("--uniform_redshift_distribution", action="store_true", default=False, help="Select AGN subset with uniform redshift distribution (default: False)")
    parser.add_argument(
        "--completeness_sim_file",
        type=str,
        default=DEFAULT_COMPLETENESS_SIM_FILE,
        help="Optional mock catalog HDF5 override. If omitted, generate a fresh mock catalog for each run.",
    )
    parser.add_argument(
        "--completeness_mode",
        type=str,
        choices=list(VALID_COMPLETENESS_MODES),
        default="2d",
        help="Completeness model to use: 2D p(det|m,z), 3D p(det|m,z,f_host_2500), or 4D p(det|m,z,f_host_2500,alpha_lambda).",
    )
    parser.add_argument(
        "--correct-sigma-uv-host",
        action="store_true",
        default=False,
        help="Correct log_sigma_uv using f_host_2500, propagate f_host_2500_err into log_sigma_uv_std_psd, and save diagnostics plots.",
    )
    parser.add_argument(
        "--fit_alpha_lambda_term",
        action="store_true",
        default=False,
        help="Fit an additional linear alpha_lambda term in the AGN standardization relation.",
    )
    parser.add_argument(
        "--fit_eta_sigma_term",
        action="store_true",
        default=False,
        help="Fit an additional linear eta_sigma term in the AGN standardization relation.",
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
    if args.disable_ceph_dist_calibration:
        print("Warning: Running without CEPH_DIST calibration; using the Planck H0 prior instead.")
    if args.resume:
        print("Warning: Resuming previous MCMC run.")

    df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_pantheon_data()
    agn_plot_path = f"plots/hubble/{args.prefix}"
    cut_report_path = Path(agn_plot_path) / "cut_summary.txt"
    df_agn, df_agn_all = load_agn_data(args.agn_data_filepath, populate_sdss=args.force_populate_fields, 
                           apply_cut=not args.no_cuts,
                           residuals_sigma_clip=args.residuals_sigma_clip, residuals_csv=args.residuals_csv,
                           exclude_object_ids_csv=args.exclude_object_ids_csv,
                           spectra_fit_csv=args.spectra_fit_csv,
                           correct_sigma_uv_host=args.correct_sigma_uv_host,
                           z_range=tuple(args.z_range), plot_path=agn_plot_path,
                           cut_report_path=cut_report_path)
    df_agn, effective_N = subsample_dataframe_at_most(
        df_agn,
        args.N,
        random_state=42,
        label="AGN objects",
    )
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
                N=effective_N,
                uniform_redshift_distribution=args.uniform_redshift_distribution,
                disable_ceph_dist_calibration=args.disable_ceph_dist_calibration,
                use_alpha_lambda_term=args.fit_alpha_lambda_term,
                use_eta_sigma_term=args.fit_eta_sigma_term,
                use_redshift_log_f_term=args.fit_redshift_log_f_term,
            )
    elif args.run == "single": # default
        cosmo_models_dict = {k: {} for k in args.cosmo_models}
        for cosmo_model in args.cosmo_models:
            r = run_single(df_agn=df_agn, df_agn_all=df_agn_all, df_pantheon=df_pantheon, _sna_L=_sna_L, _sna_Lower=_sna_Lower, _sna_LogdetCov=_sna_LogdetCov, 
                           cosmo_model=cosmo_model,
                completeness=not args.disable_completeness, use_full_cov=not args.disable_full_covariance, resume=args.resume, z_range=args.z_range,
                speed=args.speed, N=effective_N, only_sna=args.only_sna,
                skip_plots=args.skip_plots, residuals_sigma_clip=args.residuals_sigma_clip,
                df_calibrators=df_calibrators,
                prefix=args.prefix,
                completeness_sim_file=args.completeness_sim_file,
                completeness_mode=args.completeness_mode,
                compare_sigma_only=args.compare_sigma_only,
                disable_ceph_dist_calibration=args.disable_ceph_dist_calibration,
                use_alpha_lambda_term=args.fit_alpha_lambda_term,
                use_eta_sigma_term=args.fit_eta_sigma_term,
                use_redshift_log_f_term=args.fit_redshift_log_f_term)
            samples_joint, model_labels, dm_interp, logZ_joint, logZerr_joint, debiased_residuals, age, age_err = r
            cosmo_models_dict[cosmo_model]['logZ'] = logZ_joint
            cosmo_models_dict[cosmo_model]['logZerr'] = logZerr_joint
            cosmo_models_dict[cosmo_model]['age'] = age
            cosmo_models_dict[cosmo_model]['age_err'] = age_err
        zmin, zmax = args.z_range
        n_tag = "all" if effective_N is None else f"N{effective_N}"
        z_tag = f"z{zmin:.2f}_{zmax:.2f}".replace(".", "p")
        completeness_tag = "" if not args.disable_completeness else "_disable_completeness"
        ceph_tag = "_nocephdist_planckh0" if args.disable_ceph_dist_calibration else ""
        alpha_tag = "_alphaLam" if args.fit_alpha_lambda_term else ""
        eta_sigma_tag = "_etaSigma" if args.fit_eta_sigma_term else ""
        logf_tag = "_logfz" if args.fit_redshift_log_f_term else ""
        compare_path = (
            f"plots/hubble/{args.prefix}/single_compare_{args.speed}_{n_tag}_{z_tag}"
            f"{completeness_tag}{ceph_tag}{alpha_tag}{eta_sigma_tag}{logf_tag}"
        )
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
                speed=args.speed, resume=args.resume, N=effective_N,
                completeness=not args.disable_completeness,
                prefix=args.prefix, result_prefix=args.result_prefix, uniform_redshift_distribution=args.uniform_redshift_distribution,
                completeness_sim_file=args.completeness_sim_file,
                completeness_mode=args.completeness_mode,
                compare_sigma_only=args.compare_sigma_only,
                disable_ceph_dist_calibration=args.disable_ceph_dist_calibration,
                use_alpha_lambda_term=args.fit_alpha_lambda_term,
                use_eta_sigma_term=args.fit_eta_sigma_term,
                use_redshift_log_f_term=args.fit_redshift_log_f_term)
    
    print(f"Finished running Hubble fit pipeline for {args.cosmo_models}")
