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
DEFAULT_COMPLETENESS_FOOTPRINT_AREA_DEG2 = 5.0

from qvc.hubble.hubble_utils import (
    compare_models_by_log_evidence_all,
    compute_alpha_ox,
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
from qvc.hubble.hubble_likelihood import (
    log_likelihood,
    log_likelihood_nearbylcs,
    sigma_lens_from_dc,
    sigma_mu_from_z_err,
)
from qvc.hubble.hubble_plotting import (
    plot_blr_diagnostics_summary,
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
    plot_hubble_residual_tail_diagnostics,
    plot_parameter_residual_diagnostics,
    plot_predicted_L2500_vs_sigmahat,
    plot_L2500_vs_sigma_tau_separate,
    plot_catalog_quantity_vs_sigma_tau_separate,
    plot_predicted_vs_actual_M2500,
    plot_redshift_histograms,
    plot_redshift_bin_residual_summary,
    plot_redshift_wiggle_diagnostics,
    plot_residuals_vs_alphaOX,
    plot_sigma_uv_mpred_correction,
)
from qvc.hubble.tex_utils import make_agn_csv_table, make_agn_latex_table
from qvc.hubble.hubble_model import (
    AGN_LOGF_Z_PARAM,
    AgnPivotContext,
    agn_model_pack_obs,
    agn_model_pack_params,
    build_agn_pivot_context,
    evaluate_log_f,
    get_agn_model_spec,
    get_model_params,
    M_model_agn,
    M_model_agn_err,
    resolve_model_option_flags,
)
from qvc.hubble.hubble_completeness_refactored import (
    COMPLETENESS_FHOST_COL,
    COMPLETENESS_MAG_COL,
    COMPLETENESS_MAG_ERR_COL,
    VALID_COMPLETENESS_MAGNITUDES,
    get_completeness_function_2d,
    get_completeness_function_3d_fhost,
    get_completeness_function_4d_fhost_alpha,
    make_dm_function,
    normalize_completeness_magnitude,
    prepare_completeness_magnitude_columns,
)
from qvc.hubble.completeness_mock_catalog import (
    COSMO as COMPLETENESS_MOCK_COSMO,
    build_shen_lf,
    mock_m_per_zbin,
    save_mock_catalog,
)

VALID_COMPLETENESS_MODES = ("2d", "3d_fhost", "4d_fhost_alpha")
SPEED_CHOICES = ("fastest", "quick", "standard", "production")
SIGMA_CLIP_SECOND_PASS_MODES = ("warm", "fresh")
AGN_PIVOT_CHECKPOINT_KEYS = (
    "agn_pivot_observable_names",
    "agn_pivot_values",
    "agn_pivot_z_range",
    "agn_pivot_reference_object_ids",
    "agn_pivot_rule",
)


def validate_completeness_mode(completeness_mode):
    if completeness_mode not in VALID_COMPLETENESS_MODES:
        raise ValueError(
            f"Invalid completeness_mode={completeness_mode!r}. "
            f"Expected one of {VALID_COMPLETENESS_MODES}."
        )


def normalize_speed(speed):
    normalized = str(speed).strip().lower()
    if normalized in SPEED_CHOICES:
        return normalized
    else:
        raise ValueError(
            f"Invalid speed={speed!r}. Expected one of {SPEED_CHOICES}. "
            "Ordered fastest to slowest: fastest, quick, standard, production."
        )


def normalize_sigma_clip_second_pass_mode(mode):
    normalized = str(mode).strip().lower()
    if normalized in SIGMA_CLIP_SECOND_PASS_MODES:
        return normalized
    raise ValueError(
        f"Invalid sigma_clip_second_pass_mode={mode!r}. "
        f"Expected one of {SIGMA_CLIP_SECOND_PASS_MODES}."
    )


def _fit_mode_label(only_sna=False, only_agn=False):
    if only_sna and only_agn:
        raise ValueError("only_sna and only_agn cannot both be True.")
    if only_sna:
        return "sna"
    if only_agn:
        return "agn"
    return "joint"


def _cosmo_from_params(cosmo_model, params, zp):
    if cosmo_model == "FlatwCDM":
        return FlatwCDM(H0=params["H0"], Om0=params["Om0"], w0=params["w0"])
    if cosmo_model == "Flatw0waCDM":
        return Flatw0waCDM(
            H0=params["H0"],
            Om0=params["Om0"],
            w0=params["w0"],
            wa=params["wa"],
        )
    if cosmo_model == "FlatLambdaCDM":
        return FlatLambdaCDM(H0=params["H0"], Om0=params["Om0"])
    if cosmo_model == "FlatwpwaCDM":
        return FlatwpwaCDM(
            H0=params["H0"],
            Om0=params["Om0"],
            wp=params["wp"],
            wa=params["wa"],
            zp=zp,
        )
    raise ValueError(f"Invalid cosmology model: {cosmo_model!r}")


def _compute_alpha_ox_from_posterior_median(
    df_agn,
    flat_samples,
    model_labels,
    *,
    cosmo_model,
    z_pivot,
):
    median_params = dict(zip(model_labels, np.median(flat_samples, axis=0)))
    cosmology = _cosmo_from_params(cosmo_model, median_params, z_pivot)
    print(
        "Computing alpha-OX with posterior-median "
        f"{cosmo_model} parameters: {median_params}"
    )
    return compute_alpha_ox(df_agn, cosmology=cosmology)


def _resolve_table_debias_values_for_frame(df_agn, *, dmi_values):
    """Validate direct per-object corrections used in the AGN results table."""
    dmi = np.asarray(dmi_values, dtype=float)
    if dmi.shape != (len(df_agn),):
        raise ValueError(f"dmi_values has shape {dmi.shape}, but expected {(len(df_agn),)}.")
    return dmi


def _compute_debiased_agn_table_mu(
    flat_samples,
    model_labels,
    df_agn,
    cosmo_model,
    *,
    z_pivot_agn,
    agn_pivot_context,
    dmi_values,
    only_agn=False,
    use_alpha_lambda_term=False,
    use_eta_sigma_term=False,
    use_redshift_log_f_term=False,
):
    """Compute debiased AGN distance-modulus table values without making plots."""
    samples = np.asarray(flat_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != len(model_labels):
        raise ValueError(
            f"Expected flat_samples shape (n, {len(model_labels)}), got {samples.shape}."
        )
    if len(df_agn) == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    option_flags = resolve_model_option_flags(
        cosmo_model,
        samples.shape[1],
        only_agn=only_agn,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    param_indices = {label: idx for idx, label in enumerate(model_labels)}
    m_obs = df_agn["apparent_mag_2500"].to_numpy(dtype=float)
    agn_obs_arr, agn_err_arr, agn_pivot_arr = agn_model_pack_obs(
        df_agn,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        pivot_context=agn_pivot_context,
    )

    mu_samples = []
    for sample in samples:
        sample_params = {label: sample[param_indices[label]] for label in model_labels}
        agn_params_arr = agn_model_pack_params(
            sample_params,
            use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
            use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        )
        predicted_M2500 = M_model_agn(
            agn_params_arr,
            agn_obs_arr,
            agn_pivot_arr,
            use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
            use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        )
        mu_samples.append(m_obs - predicted_M2500)

    debias_values = _resolve_table_debias_values_for_frame(
        df_agn,
        dmi_values=dmi_values,
    )
    mu_samples = np.asarray(mu_samples, dtype=float) - debias_values
    mu_median = np.percentile(mu_samples, 50, axis=0)

    median_params = {
        label: float(np.nanmedian(samples[:, idx]))
        for idx, label in enumerate(model_labels)
    }
    agn_params_arr = agn_model_pack_params(
        median_params,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
    )
    predicted_M2500_err = M_model_agn_err(
        agn_params_arr,
        agn_obs_arr,
        agn_err_arr,
        agn_pivot_arr,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
    )
    cosmo = _cosmo_from_params(cosmo_model, median_params, z_pivot_agn)
    z = df_agn["z"].to_numpy(dtype=float)
    m_err = df_agn["apparent_mag_2500_err"].to_numpy(dtype=float)
    z_err = df_agn["z_err"].to_numpy(dtype=float)
    log_f_eff = evaluate_log_f(
        median_params,
        z,
        z_pivot=z_pivot_agn,
        use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
    )
    mu_err = np.sqrt(
        m_err**2
        + predicted_M2500_err**2
        + sigma_mu_from_z_err(z, z_err, cosmo)**2
        + sigma_lens_from_dc(z, cosmo)**2
        + np.exp(log_f_eff) ** 2
    )
    return mu_median, mu_err


def _agn_likelihood_param_labels(
    model_labels,
    cosmo_model,
    *,
    use_alpha_lambda_term=False,
    use_eta_sigma_term=False,
    use_redshift_log_f_term=False,
):
    allowed_labels = {
        "M0_agn",
        "alpha_agn",
        "beta_agn",
        "log_f",
        "H0",
        "Om0",
    }
    if cosmo_model == "FlatwCDM":
        allowed_labels.add("w0")
    elif cosmo_model == "Flatw0waCDM":
        allowed_labels.update({"w0", "wa"})
    elif cosmo_model == "FlatwpwaCDM":
        allowed_labels.update({"wp", "wa"})
    elif cosmo_model != "FlatLambdaCDM":
        raise ValueError(f"Invalid cosmology model: {cosmo_model!r}")

    req_params, _, _ = get_agn_model_spec(
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    allowed_labels.update(req_params)
    if use_redshift_log_f_term:
        allowed_labels.add(AGN_LOGF_Z_PARAM)

    return [label for label in model_labels if label in allowed_labels]


def compute_agn_likelihood_space_reduced_chi2(
    flat_samples,
    model_labels,
    df_agn_fit_selection,
    cosmo_model,
    *,
    z_pivot_agn,
    agn_pivot_context,
    use_alpha_lambda_term=False,
    use_eta_sigma_term=False,
    use_redshift_log_f_term=False,
):
    """Compute AGN chi2 with the same residual and variance as the AGN likelihood."""
    agn_likelihood_labels = _agn_likelihood_param_labels(
        model_labels,
        cosmo_model,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    n_agn_params = len(agn_likelihood_labels)
    if df_agn_fit_selection is None or len(df_agn_fit_selection) == 0:
        return np.nan, {"chi2": np.nan, "dof": 0, "N_eff": 0, "n_params": n_agn_params}

    samples = np.asarray(flat_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != len(model_labels):
        raise ValueError(
            f"Expected flat_samples shape (n, {len(model_labels)}), got {samples.shape}."
        )

    median_params = {
        label: float(np.nanmedian(samples[:, idx]))
        for idx, label in enumerate(model_labels)
    }
    cosmo = _cosmo_from_params(cosmo_model, median_params, z_pivot_agn)

    agn_params_arr = agn_model_pack_params(
        median_params,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    agn_obs_arr, agn_err_arr, agn_pivot_arr = agn_model_pack_obs(
        df_agn_fit_selection,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        pivot_context=agn_pivot_context,
    )
    M_pred = M_model_agn(
        agn_params_arr,
        agn_obs_arr,
        agn_pivot_arr,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    M_pred_err, _ = M_model_agn_err(
        agn_params_arr,
        agn_obs_arr,
        agn_err_arr,
        agn_pivot_arr,
        check_negative=True,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )

    z = df_agn_fit_selection["z"].to_numpy(dtype=float)
    z_err = df_agn_fit_selection["z_err"].to_numpy(dtype=float)
    m_obs = df_agn_fit_selection["apparent_mag_2500"].to_numpy(dtype=float)
    m_err = df_agn_fit_selection["apparent_mag_2500_err"].to_numpy(dtype=float)

    log_f_eff = evaluate_log_f(
        median_params,
        z,
        z_pivot=z_pivot_agn,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    mu_pred = m_obs - M_pred
    mu_cosmo = cosmo.distmod(z).value
    mu_err = np.sqrt(
        m_err**2
        + M_pred_err**2
        + sigma_mu_from_z_err(z, z_err, cosmo)**2
        + sigma_lens_from_dc(z, cosmo)**2
        + np.exp(log_f_eff)**2
    )

    return reduced_chi_squared(
        mu_pred - mu_cosmo,
        mu_err,
        n_params=n_agn_params,
    )


def get_dynesty_speed_settings(speed, ndim, *, warm_start=False):
    speed = normalize_speed(speed)
    if speed == "fastest":
        settings = dict(dlogz_init=10, n_effective=50, nlive_init=20, nlive_batch=5)
    elif speed == "quick":
        settings = dict(dlogz_init=0.01, n_effective=500, nlive_init=50, nlive_batch=20)
    elif speed == "standard":
        settings = dict(dlogz_init=0.01, n_effective=1000, nlive_init=250, nlive_batch=100)
    elif speed == "production":
        settings = dict(
            dlogz_init=0.01,
            n_effective=2000,
            nlive_init=max(1000, 50 * ndim),
            nlive_batch=max(500, 25 * ndim),
        )
    else:
        raise ValueError(f"Invalid speed={speed!r}. Expected one of {SPEED_CHOICES}.")

    if not warm_start:
        return settings

    return {
        "dlogz_init": settings["dlogz_init"],
        "n_effective": max(25, int(np.ceil(0.10 * settings["n_effective"]))),
        "nlive_init": max(2 * ndim + 1, int(np.ceil(0.10 * settings["nlive_init"]))),
        "nlive_batch": max(5, int(np.ceil(0.10 * settings["nlive_batch"]))),
    }


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


def inverse_prior_transform_dynesty(samples, priors, model_labels, *, eps=1e-9):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != len(model_labels):
        raise ValueError(
            f"Expected samples with shape (n, {len(model_labels)}), got {samples.shape}."
        )
    unit = np.empty_like(samples, dtype=float)
    for j, key in enumerate(model_labels):
        lo, hi = priors[key]
        width = float(hi) - float(lo)
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError(f"Prior for {key!r} must have finite positive width, got {(lo, hi)}.")
        unit[:, j] = (samples[:, j] - float(lo)) / width
    return np.clip(unit, eps, 1.0 - eps)


def build_warm_start_live_points(
    flat_samples,
    *,
    priors,
    model_labels,
    nlive,
    loglike_func,
    logl_kwargs,
    rng_seed=12345,
    jitter_scale=1e-3,
    eps=1e-9,
):
    flat_samples = np.asarray(flat_samples, dtype=float)
    if flat_samples.ndim != 2 or flat_samples.shape[1] != len(model_labels):
        raise ValueError(
            f"Warm-start flat_samples must have shape (n, {len(model_labels)}), got {flat_samples.shape}."
        )
    finite = np.all(np.isfinite(flat_samples), axis=1)
    flat_samples = flat_samples[finite]
    if flat_samples.size == 0:
        raise ValueError("Cannot build warm-start live points from an empty/non-finite pass-1 posterior.")

    rng = np.random.default_rng(rng_seed)
    replace = len(flat_samples) < int(nlive)
    selected = flat_samples[
        rng.choice(len(flat_samples), size=int(nlive), replace=replace)
    ]
    live_u = inverse_prior_transform_dynesty(selected, priors, model_labels, eps=eps)
    if jitter_scale and jitter_scale > 0.0:
        live_u = np.clip(
            live_u + rng.normal(0.0, float(jitter_scale), size=live_u.shape),
            eps,
            1.0 - eps,
        )
    live_v = np.asarray(
        [prior_transform_dynesty(row, priors, model_labels) for row in live_u],
        dtype=float,
    )

    live_logl = []
    live_blobs = []
    for theta in live_v:
        value = loglike_func(theta, **logl_kwargs)
        if isinstance(value, tuple):
            logl, blob = value
        else:
            logl, blob = value, None
        live_logl.append(float(logl))
        if blob is not None:
            live_blobs.append(np.asarray(blob, dtype=float))

    live_logl = np.asarray(live_logl, dtype=float)
    if len(live_blobs) == len(live_logl):
        return [live_u, live_v, live_logl, np.asarray(live_blobs, dtype=float)]
    return [live_u, live_v, live_logl]


def make_run_tag(
    cosmo_model,
    only_sna,
    speed,
    N,
    z_range,
    only_agn=False,
    completeness=True,
    completeness_mode="2d",
    completeness_magnitude="dereddened",
    disable_ceph_dist_calibration=False,
    use_planck_h0_prior=False,
    use_planck_om_prior=False,
    use_alpha_lambda_term=False,
    use_eta_sigma_term=False,
    use_redshift_log_f_term=False,
):
    speed = normalize_speed(speed)
    zmin, zmax = z_range
    n_tag = "all" if N is None else f"N{N}"
    z_tag = f"z{zmin:.2f}_{zmax:.2f}".replace(".", "p")
    completeness_magnitude = normalize_completeness_magnitude(
        completeness_magnitude
    )
    completeness_tag = (
        f"_{completeness_mode}_compmag-{completeness_magnitude}"
        if completeness
        else "_disable_completeness"
    )
    ceph_tag = "_nocephdist_planckh0" if disable_ceph_dist_calibration else ""
    planck_h0_tag = "_planckh0" if use_planck_h0_prior and not disable_ceph_dist_calibration else ""
    planck_om_tag = "_planckom" if use_planck_om_prior else ""
    alpha_tag = "_alphaLam" if use_alpha_lambda_term else ""
    eta_sigma_tag = "_etaSigma" if use_eta_sigma_term else ""
    logf_tag = "_logfz" if use_redshift_log_f_term else ""
    return (
        f"{cosmo_model}_{_fit_mode_label(only_sna, only_agn)}_{speed}_{n_tag}_{z_tag}"
        f"{completeness_tag}{ceph_tag}{planck_h0_tag}{planck_om_tag}{alpha_tag}{eta_sigma_tag}{logf_tag}"
    )


def _agn_pivot_checkpoint_payload(agn_pivot_context):
    if not isinstance(agn_pivot_context, AgnPivotContext):
        raise TypeError(
            "agn_pivot_context must be an AgnPivotContext; "
            f"got {type(agn_pivot_context).__name__}."
        )
    return {
        "agn_pivot_observable_names": agn_pivot_context.observable_names,
        "agn_pivot_values": agn_pivot_context.values,
        "agn_pivot_z_range": agn_pivot_context.z_range,
        "agn_pivot_reference_object_ids": agn_pivot_context.reference_object_ids,
        "agn_pivot_rule": agn_pivot_context.rule,
    }


def _checkpoint_string_tuple(value, *, field_name, checkpoint_file):
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_file}' field {field_name!r} must be 1D, "
            f"got shape {arr.shape}."
        )
    return tuple(
        item.decode("utf-8")
        if isinstance(item, (bytes, np.bytes_))
        else str(item)
        for item in arr.tolist()
    )


def _checkpoint_reference_object_id_tuple(
    value,
    *,
    field_name,
    checkpoint_file,
):
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_file}' field {field_name!r} must be 1D, "
            f"got shape {arr.shape}."
        )
    return tuple(
        item.decode("utf-8")
        if isinstance(item, (bytes, np.bytes_))
        else item
        for item in arr.tolist()
    )


def _load_agn_pivot_context_from_checkpoint(
    results,
    *,
    checkpoint_file,
    use_alpha_lambda_term=False,
    use_eta_sigma_term=False,
):
    missing = sorted(set(AGN_PIVOT_CHECKPOINT_KEYS) - set(results))
    if missing:
        raise RuntimeError(
            f"AGN checkpoint '{checkpoint_file}' is missing required immutable "
            f"pivot metadata: {missing}. No legacy fallback is supported; start "
            "a fresh fit with the current code."
        )

    try:
        observable_names = _checkpoint_string_tuple(
            results["agn_pivot_observable_names"],
            field_name="agn_pivot_observable_names",
            checkpoint_file=checkpoint_file,
        )
        reference_object_ids = _checkpoint_reference_object_id_tuple(
            results["agn_pivot_reference_object_ids"],
            field_name="agn_pivot_reference_object_ids",
            checkpoint_file=checkpoint_file,
        )
        values = tuple(
            float(value)
            for value in np.asarray(results["agn_pivot_values"], dtype=float).tolist()
        )
        stored_z_range = tuple(
            float(value)
            for value in np.asarray(results["agn_pivot_z_range"], dtype=float).tolist()
        )
        rule_value = np.asarray(results["agn_pivot_rule"])
        if rule_value.ndim == 0:
            rule = rule_value.item()
        elif rule_value.size == 1:
            rule = rule_value.reshape(()).item()
        else:
            raise ValueError(
                "agn_pivot_rule must contain exactly one string value."
            )
        if isinstance(rule, bytes):
            rule = rule.decode("utf-8")
        context = AgnPivotContext(
            observable_names=observable_names,
            values=values,
            z_range=stored_z_range,
            reference_object_ids=reference_object_ids,
            rule=str(rule),
        )
        context.as_array(
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"AGN checkpoint '{checkpoint_file}' contains invalid or incompatible "
            "pivot metadata."
        ) from exc
    return context


def _validate_agn_pivot_checkpoint_reference_provenance(
    agn_pivot_context,
    results,
    *,
    checkpoint_file,
):
    """Verify pivot reference IDs against immutable checkpoint sample provenance."""

    if not isinstance(agn_pivot_context, AgnPivotContext):
        raise TypeError(
            "agn_pivot_context must be an AgnPivotContext; "
            f"got {type(agn_pivot_context).__name__}."
        )

    stage = _checkpoint_stage_from_results(results)
    provenance_field = (
        "object_id_initial_fit_selection"
        if stage in {"pass1", "pass2"}
        else "object_id_fit_selection"
    )
    if provenance_field not in results:
        raise RuntimeError(
            f"AGN checkpoint '{checkpoint_file}' is missing required immutable "
            f"pivot-reference provenance field {provenance_field!r} for "
            f"sigma_clip_pass_stage={stage!r}. No legacy fallback is supported."
        )

    provenance_ids = tuple(
        _normalize_object_id_array(
            results[provenance_field],
            field_name=provenance_field,
            checkpoint_file=checkpoint_file,
        ).tolist()
    )
    if provenance_ids != agn_pivot_context.reference_object_ids:
        raise RuntimeError(
            f"AGN checkpoint '{checkpoint_file}' has incompatible pivot reference "
            f"object IDs: agn_pivot_reference_object_ids does not exactly match "
            f"{provenance_field!r}, including order and multiplicity."
        )
    return agn_pivot_context


def _validate_agn_pivot_context_for_reference(
    agn_pivot_context,
    df_agn_reference,
    *,
    z_range,
    use_alpha_lambda_term=False,
    use_eta_sigma_term=False,
    require_reference_ids=True,
):
    if not isinstance(agn_pivot_context, AgnPivotContext):
        raise TypeError(
            "agn_pivot_context must be an AgnPivotContext; "
            f"got {type(agn_pivot_context).__name__}."
        )
    requested_z_range = tuple(float(value) for value in z_range)
    if agn_pivot_context.z_range != requested_z_range:
        raise ValueError(
            "AGN pivot redshift range does not match the requested fit range: "
            f"stored={agn_pivot_context.z_range!r}, requested={requested_z_range!r}."
        )
    agn_pivot_context.as_array(
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    if require_reference_ids:
        expected_ids = tuple(df_agn_reference["object_id"].astype(str).tolist())
        if agn_pivot_context.reference_object_ids != expected_ids:
            raise ValueError(
                "AGN pivot reference object IDs do not exactly match the initial "
                "fitted subset, including order and replacement-sampling multiplicity."
            )
    return agn_pivot_context


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


def _validate_resume_replot_checkpoint_params(results, checkpoint_file, ndim):
    if "flat_samples" not in results:
        raise RuntimeError(
            f"Resume-replot checkpoint '{checkpoint_file}' is missing required dataset 'flat_samples'."
        )
    flat_samples = np.asarray(results["flat_samples"])
    if flat_samples.ndim != 2:
        raise RuntimeError(
            f"Resume-replot checkpoint '{checkpoint_file}' has flat_samples with shape {flat_samples.shape}, "
            "but a 2D array is required."
        )
    if flat_samples.shape[1] != ndim:
        raise RuntimeError(
            f"Resume-replot checkpoint '{checkpoint_file}' was created for a different parameterization: "
            f"flat_samples has {flat_samples.shape[1]} columns, but the current model expects {ndim}."
        )


def _remap_resume_replot_checkpoint(results, checkpoint_file, df_agn_fit_selection, ndim):
    """Return checkpoint payload remapped to the current cut AGN fit selection."""

    _validate_resume_replot_checkpoint_params(results, checkpoint_file, ndim)
    if "object_id_fit_selection" not in results:
        raise RuntimeError(
            f"Resume-replot checkpoint '{checkpoint_file}' is missing required dataset "
            "'object_id_fit_selection'. Replotting with new cuts requires a checkpoint "
            "written by a version that stores fit object IDs."
        )

    saved_ids = _normalize_object_id_array(
        results["object_id_fit_selection"],
        field_name="object_id_fit_selection",
        checkpoint_file=checkpoint_file,
    )
    current_ids = df_agn_fit_selection["object_id"].astype(str).to_numpy()
    saved_index = {}
    duplicate_saved = set()
    for idx, object_id in enumerate(saved_ids):
        if object_id in saved_index:
            duplicate_saved.add(object_id)
        saved_index[object_id] = idx
    if duplicate_saved:
        preview = ", ".join(sorted(duplicate_saved)[:5])
        raise RuntimeError(
            f"Resume-replot checkpoint '{checkpoint_file}' has duplicate object_id_fit_selection "
            f"entries ({len(duplicate_saved)} duplicate IDs; examples: {preview})."
        )

    missing_ids = [object_id for object_id in current_ids if object_id not in saved_index]
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        raise RuntimeError(
            f"Resume-replot checkpoint '{checkpoint_file}' does not contain all AGNs requested "
            f"by the current cuts. Missing {len(missing_ids)} / {len(current_ids)} current "
            f"object IDs; examples: {preview}. The posterior H5 lacks per-object debias "
            "arrays for these newly included AGNs. This mode can only remove or reorder "
            "AGNs that were present in the original checkpoint."
        )

    remap_idx = np.array([saved_index[object_id] for object_id in current_ids], dtype=int)
    out = dict(results)
    out["object_id_fit_selection"] = current_ids
    per_object_keys = (
        "dmi_max_w",
        "dmi_posterior_median",
        "dmi_posterior_sigma",
        "integrals_max_w",
        "dmi_selection_sigma_posterior_median",
    )
    required_per_object = {
        "dmi_max_w",
        "dmi_posterior_sigma",
        "integrals_max_w",
    }
    for key in per_object_keys:
        if key not in results:
            if key in required_per_object:
                raise RuntimeError(
                    f"Resume-replot checkpoint '{checkpoint_file}' is missing required dataset {key!r}."
                )
            continue
        value = np.asarray(results[key])
        if value.ndim == 0:
            out[key] = value
            continue
        if value.shape[0] != saved_ids.shape[0]:
            raise RuntimeError(
                f"Resume-replot checkpoint '{checkpoint_file}' has incompatible dataset {key!r}: "
                f"length {value.shape[0]}, but object_id_fit_selection has length {saved_ids.shape[0]}."
            )
        out[key] = value[remap_idx]
    if "dmi_posterior_median" not in out:
        out["dmi_posterior_median"] = out["dmi_max_w"]
    return out


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


def normalize_resume_by_model(resume, cosmo_models):
    models = list(cosmo_models)
    if not models:
        return {}

    if resume is False or resume is None:
        return {model: False for model in models}

    if resume is True:
        return {model: True for model in models}

    if isinstance(resume, str):
        resume_stripped = resume.strip()
        resume_lower = resume_stripped.lower()
        if resume_lower in {"true", "1", "yes"}:
            return {model: True for model in models}
        if resume_lower in {"false", "0", "no"}:
            return {model: False for model in models}
        resume_paths = [resume_stripped]
    else:
        resume_values = list(resume)
        if len(resume_values) == 0:
            return {model: True for model in models}
        if len(resume_values) == 1:
            resume_stripped = str(resume_values[0]).strip()
            resume_lower = resume_stripped.lower()
            if resume_lower in {"true", "1", "yes"}:
                return {model: True for model in models}
            if resume_lower in {"false", "0", "no"}:
                return {model: False for model in models}
        resume_paths = [str(value).strip() for value in resume_values]

    if len(resume_paths) != len(models):
        raise ValueError(
            "Explicit --resume checkpoint paths must match --cosmo_models one-for-one. "
            f"Got {len(resume_paths)} resume path(s) for {len(models)} model(s): {models}."
        )
    return dict(zip(models, resume_paths))


def _normalize_resume_stage(resume_stage):
    stage = str(resume_stage).strip().lower()
    if stage not in {"both", "pass1", "pass2"}:
        raise ValueError(
            f"Invalid resume_stage={resume_stage!r}. Expected one of ('both', 'pass1', 'pass2')."
        )
    return stage


def _build_checkpoint_paths(prefix, run_tag):
    checkpoint_folder = get_qvc_result_dir() / "hubble_posteriors" / prefix
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    base = checkpoint_folder / f"posteriors_{run_tag}.h5"
    return {
        "single": str(base),
        "pass1": str(checkpoint_folder / f"posteriors_{run_tag}_pass1.h5"),
        "pass2": str(checkpoint_folder / f"posteriors_{run_tag}_pass2.h5"),
    }


def _checkpoint_stage_from_results(results):
    if "sigma_clip_pass_stage" not in results:
        raise RuntimeError(
            "Checkpoint is missing required field 'sigma_clip_pass_stage'. "
            "No legacy fallback is supported."
        )
    stage = results["sigma_clip_pass_stage"]
    if isinstance(stage, np.ndarray):
        if stage.ndim == 0:
            stage = stage.item()
        elif stage.size == 1:
            stage = stage.reshape(()).item()
    if isinstance(stage, bytes):
        stage = stage.decode("utf-8")
    stage = str(stage)
    if stage not in {"single", "pass1", "pass2"}:
        raise RuntimeError(
            f"Checkpoint contains invalid sigma_clip_pass_stage={stage!r}; expected 'single', 'pass1', or 'pass2'."
        )
    return stage


def _normalize_object_id_array(values, *, field_name, checkpoint_file):
    arr = np.asarray(values, dtype=object)
    if arr.ndim != 1:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_file}' field {field_name!r} must be 1D, got shape {arr.shape}."
        )
    for item in arr.tolist():
        missing = pd.isna(item)
        if np.ndim(missing) != 0 or bool(missing):
            raise RuntimeError(
                f"Checkpoint '{checkpoint_file}' field {field_name!r} contains "
                "a missing or non-scalar object ID."
            )
    return np.asarray(
        [
            item.decode("utf-8")
            if isinstance(item, (bytes, np.bytes_))
            else str(item)
            for item in arr.tolist()
        ],
        dtype=str,
    )


def _load_resume_replot_object_ids(resume):
    resume_path = str(resume).strip()
    if resume_path.lower() in {"true", "1", "yes", "false", "0", "no"}:
        raise ValueError("--resume_replot_with_cuts requires --resume to be an explicit posterior H5 path.")
    if not os.path.exists(resume_path):
        raise FileNotFoundError(
            f"Resume-replot checkpoint file '{resume_path}' does not exist. "
            "Replace the placeholder with the posterior H5 you want to replot."
        )
    results = load_chains(resume_path)
    if "object_id_fit_selection" not in results:
        raise RuntimeError(
            f"Resume-replot checkpoint '{resume_path}' is missing required dataset "
            "'object_id_fit_selection'. Replotting with new cuts requires a checkpoint "
            "written by a version that stores fit object IDs."
        )
    object_ids = _normalize_object_id_array(
        results["object_id_fit_selection"],
        field_name="object_id_fit_selection",
        checkpoint_file=resume_path,
    )
    if len(np.unique(object_ids)) != len(object_ids):
        raise RuntimeError(f"Resume-replot checkpoint '{resume_path}' has duplicate object_id_fit_selection entries.")
    return resume_path, object_ids


def _select_resume_replot_fit_selection(df_agn, resume):
    resume_path, saved_ids = _load_resume_replot_object_ids(resume)
    current_ids = df_agn["object_id"].astype(str)
    saved_order = {object_id: idx for idx, object_id in enumerate(saved_ids)}
    keep_mask = current_ids.isin(saved_order)
    filtered = df_agn.loc[keep_mask].copy()
    if filtered.empty:
        raise RuntimeError(
            f"None of the {len(saved_ids)} AGNs saved in resume checkpoint '{resume_path}' survived the current cuts."
        )
    filtered["_resume_replot_order"] = filtered["object_id"].astype(str).map(saved_order)
    filtered = (
        filtered.sort_values("_resume_replot_order", kind="stable")
        .drop(columns=["_resume_replot_order"])
        .reset_index(drop=True)
    )
    print(
        "Resume-replot fit selection: "
        f"using {len(filtered)} / {len(saved_ids)} saved checkpoint AGNs that survived "
        f"the current cuts; plotting will still use all {len(df_agn)} current-cut AGNs."
    )
    return filtered


def restrict_agn_to_resume_replot_sample(df_agn, resume):
    return _select_resume_replot_fit_selection(df_agn, resume).reset_index(drop=True)


def _extract_pass1_state_from_checkpoint(
    results,
    checkpoint_file,
    df_agn_full_sample,
    *,
    sigma_clip_threshold,
):
    required = [
        "sigma_clip_threshold",
        "object_id_full_sample",
        "keep_mask_full",
        "mu_zscore_pass1",
        "residuals_pass1",
        "residuals_err_pass1",
    ]
    missing = [key for key in required if key not in results]
    if missing:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_file}' is missing embedded pass-1 clipping state: {missing}. "
            "This checkpoint cannot be used to skip the first sigma-clip pass."
        )

    stored_threshold = float(np.asarray(results["sigma_clip_threshold"]).reshape(()))
    if not np.isclose(stored_threshold, float(sigma_clip_threshold), rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"Checkpoint '{checkpoint_file}' was created with sigma_clip_threshold={stored_threshold:.6g}, "
            f"but the current run requested {float(sigma_clip_threshold):.6g}."
        )

    object_ids_full = _normalize_object_id_array(
        results["object_id_full_sample"],
        field_name="object_id_full_sample",
        checkpoint_file=checkpoint_file,
    )
    current_object_ids = df_agn_full_sample["object_id"].astype(str).to_numpy()
    if object_ids_full.shape[0] != len(df_agn_full_sample) or not np.array_equal(object_ids_full, current_object_ids):
        raise RuntimeError(
            f"Checkpoint '{checkpoint_file}' embedded pass-1 state does not align with the current full AGN sample. "
            "Object IDs differ in length or order."
        )

    keep_mask_full = np.asarray(results["keep_mask_full"], dtype=bool)
    if keep_mask_full.shape != (len(df_agn_full_sample),):
        raise RuntimeError(
            f"Checkpoint '{checkpoint_file}' keep_mask_full has shape {keep_mask_full.shape}, "
            f"but expected {(len(df_agn_full_sample),)}."
        )

    residuals = np.asarray(results["residuals_pass1"], dtype=float)
    residuals_err = np.asarray(results["residuals_err_pass1"], dtype=float)
    mu_zscore = np.asarray(results["mu_zscore_pass1"], dtype=float)
    for name, value in (
        ("residuals_pass1", residuals),
        ("residuals_err_pass1", residuals_err),
        ("mu_zscore_pass1", mu_zscore),
    ):
        if value.shape != (len(df_agn_full_sample),):
            raise RuntimeError(
                f"Checkpoint '{checkpoint_file}' field {name!r} has shape {value.shape}, "
                f"but expected {(len(df_agn_full_sample),)}."
            )

    diagnostics_df = df_agn_full_sample.copy()
    diagnostics_df["residuals"] = residuals
    diagnostics_df["residuals_err"] = residuals_err
    diagnostics_df["mu_zscore"] = mu_zscore
    diagnostics_df["was_clipped"] = ~keep_mask_full
    return {
        "keep_mask_full": keep_mask_full,
        "pass1_diagnostics_df": diagnostics_df,
    }


def _write_stage_checkpoint(
    target_checkpoint_file,
    *,
    source_checkpoint_file=None,
    sigma_clip_pass_stage,
    sigma_clip_threshold=None,
    df_agn_full_sample=None,
    df_agn_plot_sample=None,
    df_agn_fit_selection=None,
    df_agn_initial_fit_selection=None,
    keep_mask_full=None,
    pass1_diagnostics_df=None,
    sigma_clip_second_pass_mode=None,
    sigma_clip_warm_start_from_pass1=None,
    logZ_is_approximate=None,
):
    source_checkpoint_file = source_checkpoint_file or target_checkpoint_file
    payload = load_chains(source_checkpoint_file)
    payload["sigma_clip_pass_stage"] = str(sigma_clip_pass_stage)
    if sigma_clip_threshold is not None:
        payload["sigma_clip_threshold"] = float(sigma_clip_threshold)
    if sigma_clip_second_pass_mode is not None:
        payload["sigma_clip_second_pass_mode"] = str(sigma_clip_second_pass_mode)
    if sigma_clip_warm_start_from_pass1 is not None:
        payload["sigma_clip_warm_start_from_pass1"] = bool(sigma_clip_warm_start_from_pass1)
    if logZ_is_approximate is not None:
        payload["logZ_is_approximate"] = bool(logZ_is_approximate)
    if df_agn_full_sample is not None:
        payload["object_id_full_sample"] = df_agn_full_sample["object_id"].astype(str).to_numpy()
    if df_agn_plot_sample is not None:
        payload["object_id_plot_sample"] = df_agn_plot_sample["object_id"].astype(str).to_numpy()
    if df_agn_fit_selection is not None:
        payload["object_id_fit_selection"] = df_agn_fit_selection["object_id"].astype(str).to_numpy()
    if df_agn_initial_fit_selection is not None:
        payload["object_id_initial_fit_selection"] = (
            df_agn_initial_fit_selection["object_id"].astype(str).to_numpy()
        )
    if keep_mask_full is not None:
        payload["keep_mask_full"] = np.asarray(keep_mask_full, dtype=bool)
    if pass1_diagnostics_df is not None:
        payload["residuals_pass1"] = pass1_diagnostics_df["residuals"].to_numpy(dtype=float)
        payload["residuals_err_pass1"] = pass1_diagnostics_df["residuals_err"].to_numpy(dtype=float)
        payload["mu_zscore_pass1"] = pass1_diagnostics_df["mu_zscore"].to_numpy(dtype=float)
    save_chains(target_checkpoint_file, **payload)


def _resolve_two_pass_resume_checkpoint(resume, resume_stage, checkpoint_paths):
    if not resume:
        return None
    if isinstance(resume, str):
        resume_stripped = resume.strip()
        resume_lower = resume_stripped.lower()
        if resume_lower not in {"true", "1", "yes"}:
            return resolve_resume_checkpoint_path(resume, checkpoint_paths["single"])

    search_order = {
        "both": ("pass2", "pass1", "single"),
        "pass1": ("pass1", "single"),
        "pass2": ("pass2", "pass1", "single"),
    }[_normalize_resume_stage(resume_stage)]
    for stage in search_order:
        candidate = checkpoint_paths[stage]
        if os.path.exists(candidate):
            return candidate
    preferred = ", ".join(checkpoint_paths[stage] for stage in search_order)
    raise FileNotFoundError(
        "Resume was requested, but no compatible checkpoint file was found for "
        f"resume_stage={resume_stage!r}. Looked for: {preferred}"
    )


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
            f"{DEFAULT_COMPLETENESS_FOOTPRINT_AREA_DEG2:.1f} deg^2."
        )
        return DEFAULT_COMPLETENESS_FOOTPRINT_AREA_DEG2

    ra = np.mod(pd.to_numeric(df_agn_all[ra_col], errors="coerce").to_numpy(dtype=float), 360.0)
    dec = pd.to_numeric(df_agn_all[dec_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(ra) & np.isfinite(dec)
    ra = ra[finite]
    dec = dec[finite]
    if ra.size < 2:
        print(
            "[WARNING] Too few finite RA/Dec rows to estimate sky-box area; "
            f"using default mock area {DEFAULT_COMPLETENESS_FOOTPRINT_AREA_DEG2:.1f} deg^2."
        )
        return DEFAULT_COMPLETENESS_FOOTPRINT_AREA_DEG2

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
            f"mock area {DEFAULT_COMPLETENESS_FOOTPRINT_AREA_DEG2:.1f} deg^2."
        )
        return DEFAULT_COMPLETENESS_FOOTPRINT_AREA_DEG2

    print(
        "Estimated pre-cut sky-box area from df_agn_all: "
        f"{area_deg2:.1f} deg^2 "
        f"(ra_col={ra_col}, dec_col={dec_col}, RA span={ra_span_deg:.2f} deg, "
        f"RA box={ra_min:.2f}->{ra_max:.2f} deg, Dec box={dec_min:.2f}->{dec_max:.2f} deg)"
    )
    return area_deg2


def _select_agn_fit_selection(
    df_agn,
    *,
    z_range,
    N,
    uniform_redshift_distribution,
):
    if uniform_redshift_distribution:
        return select_agn_subset_uniform_with_replacement(
            df_agn,
            z_range=z_range,
            N=N,
            z_uniform_min=float(z_range[0]),
        )
    df_selection = df_agn[
        df_agn["z"].between(z_range[0], z_range[1], inclusive="both")
    ].copy()
    df_selection, _ = subsample_dataframe_at_most(
        df_selection,
        N,
        random_state=42,
        label="in-range AGN objects",
    )
    return df_selection


def _prepare_shared_agn_pivot_context(
    df_agn,
    *,
    cosmo_models,
    resume_by_model,
    z_range,
    N,
    uniform_redshift_distribution,
    only_sna,
    only_agn,
    speed,
    completeness,
    completeness_mode,
    disable_ceph_dist_calibration,
    use_planck_h0_prior,
    use_planck_om_prior,
    use_alpha_lambda_term,
    use_eta_sigma_term,
    use_redshift_log_f_term,
    disable_sigma_clip_pass,
    resume_stage,
    prefix,
    completeness_magnitude="dereddened",
    resume_replot_with_cuts=False,
):
    """Build once, or strictly load once, for cosmologies sharing a fit sample."""

    if only_sna:
        return None

    reference_selection = _select_agn_fit_selection(
        df_agn,
        z_range=z_range,
        N=None if resume_replot_with_cuts else N,
        uniform_redshift_distribution=(
            False if resume_replot_with_cuts else uniform_redshift_distribution
        ),
    )
    loaded_contexts = []
    for cosmo_model in cosmo_models:
        model_resume = resume_by_model[cosmo_model]
        if not model_resume:
            continue
        run_tag = make_run_tag(
            cosmo_model,
            only_sna,
            speed,
            N,
            z_range,
            only_agn=only_agn,
            completeness=completeness,
            completeness_mode=completeness_mode,
            completeness_magnitude=completeness_magnitude,
            disable_ceph_dist_calibration=disable_ceph_dist_calibration,
            use_planck_h0_prior=use_planck_h0_prior,
            use_planck_om_prior=use_planck_om_prior,
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
            use_redshift_log_f_term=use_redshift_log_f_term,
        )
        checkpoint_paths = _build_checkpoint_paths(prefix, run_tag)
        apply_two_pass = (
            not disable_sigma_clip_pass
            and not resume_replot_with_cuts
        )
        if apply_two_pass:
            checkpoint_file = _resolve_two_pass_resume_checkpoint(
                model_resume,
                resume_stage,
                checkpoint_paths,
            )
        else:
            checkpoint_file = resolve_resume_checkpoint_path(
                model_resume,
                checkpoint_paths["single"],
            )
        results = load_chains(checkpoint_file)
        context = _load_agn_pivot_context_from_checkpoint(
            results,
            checkpoint_file=checkpoint_file,
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
        )
        _validate_agn_pivot_checkpoint_reference_provenance(
            context,
            results,
            checkpoint_file=checkpoint_file,
        )
        _validate_agn_pivot_context_for_reference(
            context,
            reference_selection,
            z_range=z_range,
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
            require_reference_ids=not resume_replot_with_cuts,
        )
        loaded_contexts.append((checkpoint_file, context))

    if loaded_contexts:
        shared_context = loaded_contexts[0][1]
        incompatible = [
            checkpoint_file
            for checkpoint_file, context in loaded_contexts[1:]
            if context != shared_context
        ]
        if incompatible:
            raise RuntimeError(
                "Cosmology checkpoints do not share one identical AGN pivot "
                f"context. Incompatible checkpoint(s): {incompatible}."
            )
        return shared_context

    return build_agn_pivot_context(
        reference_selection,
        z_range,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )


def _build_completeness_params(
    df_agn_completeness,
    df_agn_all,
    *,
    completeness,
    completeness_mode,
    completeness_sim_file,
    plot_path,
    plot=False,
):
    if not completeness:
        return None

    missing_magnitude_columns = {
        COMPLETENESS_MAG_COL,
        COMPLETENESS_MAG_ERR_COL,
    } - set(df_agn_completeness.columns)
    if missing_magnitude_columns:
        raise KeyError(
            "Completeness requires prepared 2500-A magnitude columns: "
            f"{sorted(missing_magnitude_columns)}."
        )

    if completeness_mode in ("3d_fhost", "4d_fhost_alpha"):
        if COMPLETENESS_FHOST_COL not in df_agn_completeness.columns:
            raise KeyError(
                f"completeness_mode={completeness_mode!r} requires "
                f"df_agn_completeness[{COMPLETENESS_FHOST_COL!r}]."
            )
        bad_fhost = ~np.isfinite(
            df_agn_completeness[COMPLETENESS_FHOST_COL].to_numpy(dtype=float)
        )
        if np.any(bad_fhost):
            raise ValueError(
                f"completeness_mode={completeness_mode!r} requires finite "
                f"{COMPLETENESS_FHOST_COL} for all AGN used to estimate the "
                "completeness map; "
                f"found {np.count_nonzero(bad_fhost)} non-finite rows."
            )
    if completeness_mode == "4d_fhost_alpha":
        if "alpha_lambda" not in df_agn_completeness.columns:
            raise KeyError(
                "completeness_mode='4d_fhost_alpha' requires "
                "df_agn_completeness['alpha_lambda']."
            )
        bad_alpha = ~np.isfinite(
            df_agn_completeness["alpha_lambda"].to_numpy(dtype=float)
        )
        if np.any(bad_alpha):
            raise ValueError(
                "completeness_mode='4d_fhost_alpha' requires finite alpha_lambda "
                "for all AGN used to estimate the completeness map; "
                f"found {np.count_nonzero(bad_alpha)} non-finite rows."
            )

    if completeness_mode == "4d_fhost_alpha":
        return get_completeness_function_4d_fhost_alpha(
            df_agn_completeness,
            sim_file=completeness_sim_file,
            plot=plot,
            plot_path=plot_path,
            df_agn_fhost_population=df_agn_all,
        )
    if completeness_mode == "3d_fhost":
        return get_completeness_function_3d_fhost(
            df_agn_completeness,
            sim_file=completeness_sim_file,
            plot=plot,
            plot_path=plot_path,
            df_agn_fhost_population=df_agn_all,
        )
    return get_completeness_function_2d(
        df_agn_completeness,
        sim_file=completeness_sim_file,
        plot=plot,
        plot_path=plot_path,
    )


def _map_fit_values_to_plot_sample(
    df_plot,
    df_fit_selection,
    fit_values,
    *,
    value_name,
    uniform_redshift_distribution=False,
):
    if uniform_redshift_distribution:
        raise ValueError(
            f"{value_name} alignment is not supported with uniform_redshift_distribution=True."
        )
    if len(df_fit_selection) != len(fit_values):
        raise ValueError(
            "Fit/plot alignment failure: "
            f"df_agn_fit_selection has length {len(df_fit_selection)}, "
            f"but {value_name} has length {len(fit_values)}."
        )
    fit_indices = pd.Index(df_fit_selection.index)
    if not fit_indices.isin(df_plot.index).all():
        missing = fit_indices[~fit_indices.isin(df_plot.index)].tolist()[:10]
        raise ValueError(
            "Fit/plot alignment failure: fitted AGN selection contains index values "
            f"not present in df_agn: {missing}"
        )
    full_values = np.full(len(df_plot), np.nan, dtype=float)
    df_plot_index_positions = pd.Series(np.arange(len(df_plot)), index=df_plot.index)
    full_values[df_plot_index_positions.loc[fit_indices].to_numpy()] = np.asarray(
        fit_values,
        dtype=float,
    )
    return full_values


def _extract_fit_values_from_plot_sample(
    df_plot,
    df_fit_selection,
    plot_values,
    *,
    value_name,
):
    plot_values = np.asarray(plot_values, dtype=float)
    if plot_values.shape[0] != len(df_plot):
        raise ValueError(
            f"{value_name} has length {plot_values.shape[0]}, but df_agn has length {len(df_plot)}."
        )
    fit_indices = pd.Index(df_fit_selection.index)
    if not fit_indices.isin(df_plot.index).all():
        missing = fit_indices[~fit_indices.isin(df_plot.index)].tolist()[:10]
        raise ValueError(
            f"Could not align {value_name} back to the fit selection; missing indices: {missing}"
        )
    df_plot_index_positions = pd.Series(np.arange(len(df_plot)), index=df_plot.index)
    return plot_values[df_plot_index_positions.loc[fit_indices].to_numpy()]


def _compute_direct_full_sample_completeness_summaries(
    flat_samples,
    *,
    df_agn_fit_selection,
    df_agn_plot_sample,
    df_pantheon,
    _sna_L,
    _sna_Lower,
    _sna_LogdetCov,
    cosmo_model,
    completeness_params,
    z_pivot_agn,
    agn_pivot_context,
    use_full_cov,
    disable_ceph_dist_calibration,
    use_planck_h0_prior,
    use_planck_om_prior,
    only_agn=False,
    use_alpha_lambda_term=False,
    use_eta_sigma_term=False,
    use_redshift_log_f_term=False,
    early_de_guard=False,
):
    n_plot = len(df_agn_plot_sample)
    if n_plot == 0:
        empty = np.empty(0, dtype=float)
        return empty, empty, None

    if len(df_agn_fit_selection) == 0:
        raise ValueError("Cannot compute direct full-sample completeness summaries with an empty fit selection.")

    if completeness_params is None:
        zeros = np.zeros(n_plot, dtype=float)
        return zeros, zeros, None

    dmi_draws = []
    sigma_sel_draws = []
    for theta in np.asarray(flat_samples, dtype=float):
        _, blob = log_likelihood(
            theta,
            agn_data=df_agn_plot_sample,
            pantheon_data=df_pantheon,
            _sna_L=_sna_L,
            _sna_Lower=_sna_Lower,
            _sna_LogdetCov=_sna_LogdetCov,
            cosmo_model=cosmo_model,
            completeness_params=completeness_params,
            z_pivot_agn=z_pivot_agn,
            agn_calibrators_data=None,
            agn_pivot_context=agn_pivot_context,
            use_planck_h0_prior=use_planck_h0_prior,
            use_planck_om_prior=use_planck_om_prior,
            use_ceph_dist_calibration=not disable_ceph_dist_calibration,
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
            use_redshift_log_f_term=use_redshift_log_f_term,
            early_de_guard=early_de_guard,
            only_sna=False,
            only_agn=only_agn,
            use_full_cov=use_full_cov,
        )
        blob = np.asarray(blob, dtype=float)
        dmi_draws.append(blob[1])
        sigma_sel_draws.append(blob[2])

    dmi_draws = np.asarray(dmi_draws, dtype=float)
    sigma_sel_draws = np.asarray(sigma_sel_draws, dtype=float)
    dmi_posterior_median_full_direct = np.median(dmi_draws, axis=0)
    dmi_posterior_sigma_full_direct = 0.5 * (
        np.percentile(dmi_draws, 84, axis=0)
        - np.percentile(dmi_draws, 16, axis=0)
    )
    dmi_selection_sigma_full_direct = np.median(sigma_sel_draws, axis=0)
    return (
        dmi_posterior_median_full_direct,
        dmi_posterior_sigma_full_direct,
        dmi_selection_sigma_full_direct,
    )


def _build_sigma_clip_diagnostics(
    df_agn_diagnostics,
    residuals,
    residuals_err,
    *,
    sigma_clip_threshold,
):
    residuals = np.asarray(residuals, dtype=float)
    residuals_err = np.asarray(residuals_err, dtype=float)
    if residuals.shape[0] != len(df_agn_diagnostics):
        raise ValueError(
            "Residual diagnostics alignment failure: "
            f"{residuals.shape[0]} residuals for {len(df_agn_diagnostics)} AGN."
        )
    if residuals_err.shape[0] != len(df_agn_diagnostics):
        raise ValueError(
            "Residual diagnostics alignment failure: "
            f"{residuals_err.shape[0]} residual errors for {len(df_agn_diagnostics)} AGN."
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        mu_zscore = np.abs(residuals) / residuals_err
    keep_mask = np.isfinite(mu_zscore) & (mu_zscore < float(sigma_clip_threshold))

    diagnostics_df = df_agn_diagnostics.copy()
    diagnostics_df["residuals"] = residuals
    diagnostics_df["residuals_err"] = residuals_err
    diagnostics_df["mu_zscore"] = mu_zscore
    diagnostics_df["was_clipped"] = ~keep_mask
    preferred_columns = [
        "object_id",
        "sdss_name",
        "ra",
        "dec",
        "z",
        "residuals",
        "residuals_err",
        "mu_zscore",
        "was_clipped",
    ]
    remaining_columns = [
        col for col in diagnostics_df.columns if col not in preferred_columns
    ]
    diagnostics_df = diagnostics_df[[
        col for col in preferred_columns if col in diagnostics_df.columns
    ] + remaining_columns]
    return diagnostics_df, keep_mask


def _write_sigma_clip_diagnostics(
    diagnostics_df,
    plot_path,
    *,
    residuals_filename,
    clipped_filename=None,
):
    residuals_path = Path(plot_path) / residuals_filename
    diagnostics_df.to_csv(residuals_path, index=False)
    print(f"Saved sigma-clipping diagnostics to {residuals_path}")
    if clipped_filename is None:
        return

    clipped_df = diagnostics_df[diagnostics_df["was_clipped"]].copy()
    clipped_df = clipped_df.sort_values(
        by=["mu_zscore", "object_id"],
        ascending=[False, True],
        na_position="last",
    )
    clipped_path = Path(plot_path) / clipped_filename
    clipped_df.to_csv(clipped_path, index=False)
    print(f"Saved clipped-object diagnostics to {clipped_path}")


def _write_sigma_clip_membership_audit(
    df_agn_full_sample,
    keep_mask_full,
    plot_path,
    *,
    filename,
    df_agn_fit_selection=None,
):
    keep_mask_full = np.asarray(keep_mask_full, dtype=bool)
    if keep_mask_full.shape[0] != len(df_agn_full_sample):
        raise ValueError(
            "Sigma-clip membership audit alignment failure: "
            f"{keep_mask_full.shape[0]} mask entries for {len(df_agn_full_sample)} AGN."
        )

    audit_df = df_agn_full_sample.copy()
    audit_df["is_in_pass2_sample"] = keep_mask_full
    audit_df["is_in_pass2_plot_sample"] = keep_mask_full
    audit_df["was_clipped_pass1"] = ~keep_mask_full
    audit_df["is_in_pass2_fit_selection"] = False
    if df_agn_fit_selection is not None:
        fit_indices = pd.Index(df_agn_fit_selection.index)
        if not fit_indices.isin(audit_df.index).all():
            missing = fit_indices[~fit_indices.isin(audit_df.index)].tolist()[:10]
            raise ValueError(
                "Sigma-clip membership audit alignment failure: "
                f"fit-selection indices missing from full sample: {missing}"
            )
        audit_df.loc[fit_indices, "is_in_pass2_fit_selection"] = True
    preferred_columns = [
        "object_id",
        "sdss_name",
        "ra",
        "dec",
        "z",
        "is_in_pass2_sample",
        "is_in_pass2_plot_sample",
        "is_in_pass2_fit_selection",
        "was_clipped_pass1",
    ]
    remaining_columns = [col for col in audit_df.columns if col not in preferred_columns]
    audit_df = audit_df[[col for col in preferred_columns if col in audit_df.columns] + remaining_columns]

    audit_path = Path(plot_path) / filename
    audit_df.to_csv(audit_path, index=False)
    print(f"Saved sigma-clip membership audit to {audit_path}")
    return audit_df


def _run_fit_stage(
    df_agn_fit_selection,
    df_agn_all,
    df_pantheon,
    _sna_L,
    _sna_Lower,
    _sna_LogdetCov,
    *,
    agn_pivot_context,
    df_calibrators,
    cosmo_model,
    only_sna,
    only_agn,
    completeness,
    use_full_cov,
    z_range,
    N,
    resume,
    speed,
    prefix,
    completeness_sim_file,
    completeness_mode,
    compare_sigma_only,
    minimal_plots=False,
    disable_ceph_dist_calibration,
    use_planck_h0_prior,
    use_planck_om_prior,
    use_alpha_lambda_term,
    use_eta_sigma_term,
    use_redshift_log_f_term,
    early_de_guard=False,
    checkpoint_file_override=None,
    resume_replot_with_cuts=False,
    warm_start_flat_samples=None,
    logZ_is_approximate=False,
    df_agn_completeness=None,
):
    use_planck_h0_prior = use_planck_h0_prior or disable_ceph_dist_calibration
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
        df_agn_fit_selection,
        df_agn_all,
        df_pantheon,
        _sna_L,
        _sna_Lower,
        _sna_LogdetCov,
        agn_pivot_context=agn_pivot_context,
        df_calibrators=df_calibrators,
        cosmo_model=cosmo_model,
        only_sna=only_sna,
        only_agn=only_agn,
        completeness=completeness,
        use_full_cov=use_full_cov,
        z_range=z_range,
        N=N,
        resume=resume,
        speed=speed,
        prefix=prefix,
        checkpoint_file_override=checkpoint_file_override,
        completeness_sim_file=completeness_sim_file,
        completeness_mode=completeness_mode,
        completeness_magnitude=df_agn_fit_selection.attrs.get(
            "completeness_magnitude",
            "dereddened",
        ),
        compare_sigma_only=compare_sigma_only,
        minimal_plots=minimal_plots,
        disable_ceph_dist_calibration=disable_ceph_dist_calibration,
        use_planck_h0_prior=use_planck_h0_prior,
        use_planck_om_prior=use_planck_om_prior,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
        early_de_guard=early_de_guard,
        resume_replot_with_cuts=resume_replot_with_cuts,
        warm_start_flat_samples=warm_start_flat_samples,
        logZ_is_approximate=logZ_is_approximate,
        df_agn_completeness=df_agn_completeness,
    )
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
        age,
        age_err,
    )


def generate_fresh_completeness_sim_file(plot_path, *, area_deg2, seed=123):
    """Generate a fresh Shen-LF mock catalog for completeness-map construction."""
    completeness_dir = Path(plot_path) / "completeness"
    completeness_dir.mkdir(parents=True, exist_ok=True)
    output_path = completeness_dir / "mock_completeness_catalog_fresh.h5"
    thinning_probability = 1.0

    rng = np.random.default_rng(seed)
    phi_log10, m_grid, z_bins = build_shen_lf(None)
    alpha_nu_parent_mean = -0.5
    alpha_nu_parent_sigma = 0.3
    _, _, _, _, z_all, m_all, m_rest_all, _, alpha_lambda_all = mock_m_per_zbin(
        phi_log10,
        m_grid,
        z_bins,
        float(area_deg2),
        alpha_nu_parent_mean,
        alpha_nu_parent_sigma,
        COMPLETENESS_MOCK_COSMO,
        z_res=512,
        m_scatter=0.0,
        kcorr_zref=2.0,
        m_lim=28.0,
        thinning_probability=thinning_probability,
        rng=rng,
        return_z=True,
        return_global=True,
        return_alpha=True,
    )
    n_generated = int(np.size(z_all))
    print(
        f"Fresh completeness mock generated {n_generated} sources "
        f"for the full {float(area_deg2):.1f} deg^2 footprint "
        f"(p_keep={thinning_probability:.4g})."
    )
    save_mock_catalog(
        output_path,
        z_all,
        m_all,
        m_rest_all,
        m_limit=28.0,
        alpha_lambda_all=alpha_lambda_all,
        thinning_probability=thinning_probability,
        rng=rng,
        area_deg2=area_deg2,
        alpha_nu_parent_mean=alpha_nu_parent_mean,
        alpha_nu_parent_sigma=alpha_nu_parent_sigma,
    )
    print(
        f"Generated fresh completeness mock catalog: {output_path} "
        f"(area_deg2={float(area_deg2):.1f}, p_keep={thinning_probability:.4g})"
    )
    return str(output_path)

def run_mcmc_pipeline(df_agn, df_agn_all, df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov,
                      *,
                      agn_pivot_context,
                      df_calibrators=None,
                      cosmo_model='Flatw0waCDM',
                      only_sna=False, only_agn=False, completeness=True, use_full_cov=True,
                      resume=False, speed="production",
                      z_range=(0.44, 3.16),
                      prefix="default",
                      checkpoint_file_override=None,
                      completeness_sim_file=DEFAULT_COMPLETENESS_SIM_FILE,
                      completeness_mode="2d",
                      completeness_magnitude="dereddened",
                      N=None,
                      compare_sigma_only=False,
                      minimal_plots=False,
                      disable_ceph_dist_calibration=False,
                      use_planck_h0_prior=False,
                      use_planck_om_prior=False,
                      use_alpha_lambda_term=False,
                      use_eta_sigma_term=False,
                      use_redshift_log_f_term=False,
                      early_de_guard=False,
                      resume_replot_with_cuts=False,
                      warm_start_flat_samples=None,
                      logZ_is_approximate=False,
                      df_agn_completeness=None,
                      ):
    validate_completeness_mode(completeness_mode)
    completeness_magnitude = normalize_completeness_magnitude(
        df_agn.attrs.get("completeness_magnitude", completeness_magnitude)
    )
    if completeness and COMPLETENESS_MAG_COL not in df_agn.columns:
        df_agn = prepare_completeness_magnitude_columns(
            df_agn,
            completeness_magnitude,
        )
    if completeness and COMPLETENESS_MAG_COL not in df_agn_all.columns:
        df_agn_all = prepare_completeness_magnitude_columns(
            df_agn_all,
            completeness_magnitude,
        )
    if (
        completeness
        and df_agn_completeness is not None
        and COMPLETENESS_MAG_COL not in df_agn_completeness.columns
    ):
        df_agn_completeness = prepare_completeness_magnitude_columns(
            df_agn_completeness,
            completeness_magnitude,
        )
    speed = normalize_speed(speed)
    _fit_mode_label(only_sna, only_agn)
    use_planck_h0_prior = use_planck_h0_prior or disable_ceph_dist_calibration
    if only_sna:
        if agn_pivot_context is not None:
            raise ValueError("SNe-only runs must not receive AGN pivot metadata.")
    else:
        _validate_agn_pivot_context_for_reference(
            agn_pivot_context,
            df_agn,
            z_range=z_range,
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
            require_reference_ids=False,
        )
    run_tag = make_run_tag(
        cosmo_model,
        only_sna,
        speed,
        N,
        z_range,
        only_agn=only_agn,
        completeness=completeness,
        completeness_mode=completeness_mode,
        completeness_magnitude=completeness_magnitude,
        disable_ceph_dist_calibration=disable_ceph_dist_calibration,
        use_planck_h0_prior=use_planck_h0_prior,
        use_planck_om_prior=use_planck_om_prior,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    plot_path = f"plots/hubble/{prefix}/{run_tag}"
    os.makedirs(plot_path, exist_ok=True)

    priors, model_labels, model_labels_latex = get_model_params(
        cosmo_model,
        only_sna=only_sna,
        only_agn=only_agn,
        use_planck_h0_prior=use_planck_h0_prior,
        use_planck_om_prior=use_planck_om_prior,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    ndim = len(model_labels)
    print(f"Running sampling with {ndim} parameters for cosmological model: {cosmo_model}")
    if resume_replot_with_cuts and not resume:
        raise ValueError("--resume_replot_with_cuts requires --resume to point at an existing posterior H5 file.")
    if resume_replot_with_cuts and only_sna:
        raise ValueError("--resume_replot_with_cuts is only supported for joint AGN Hubble fits, not --only_sna.")
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

    if df_agn_completeness is None:
        df_agn_completeness = df_agn

    if completeness and not resume_replot_with_cuts:
        if completeness_sim_file is None:
            completeness_area_deg2 = estimate_sky_box_area_deg2(df_agn_all)
            completeness_sim_file = generate_fresh_completeness_sim_file(
                plot_path,
                area_deg2=completeness_area_deg2,
            )
        print(f"Building {completeness_mode} completeness map using mock catalog: {completeness_sim_file}")
        completeness_params = _build_completeness_params(
            df_agn_completeness,
            df_agn_all,
            completeness=completeness,
            completeness_mode=completeness_mode,
            completeness_sim_file=completeness_sim_file,
            plot_path=plot_path,
            plot=not (compare_sigma_only or minimal_plots),
        )
    else:
        completeness_params = None

    agn_model_req_params, agn_model_req_obs, agn_model_req_errs = get_agn_model_spec(
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    agn_fields = agn_model_req_params + agn_model_req_obs + agn_model_req_errs
    agn_fields += ('apparent_mag_2500', 'apparent_mag_2500_err', 'z', 'z_err', 'object_id')
    if completeness:
        agn_fields += (COMPLETENESS_MAG_COL, COMPLETENESS_MAG_ERR_COL)
    if COMPLETENESS_FHOST_COL in df_agn.columns:
        agn_fields += (COMPLETENESS_FHOST_COL,)
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

    checkpoint_file = (
        str(checkpoint_file_override)
        if checkpoint_file_override is not None
        else _build_checkpoint_paths(prefix, run_tag)["single"]
    )
    print(f"Checkpoint file: {checkpoint_file}")
    n_sne = len(pantheon_data["zHD"]) if "zHD" in pantheon_data else 0
    if only_agn:
        print(f"Starting AGN-only Hubble Fit with {len(agn_data['z'])} AGNs; SN likelihood disabled.")
    else:
        print(f"Starting Hubble Fit with {len(agn_data['z'])} AGNs and {n_sne} SNes...")

    if resume:
        print("[WARNING] Resuming from checkpoint file...")
        checkpoint_file = resolve_resume_checkpoint_path(resume, checkpoint_file)
        print(f"Resuming from default checkpoint file: {checkpoint_file}")
        #sampler = DynamicNestedSampler.restore(checkpoint_file, pool=pool)
        try:
            r = load_chains(checkpoint_file)
            stored_pivot_context = None
            if not only_sna:
                stored_pivot_context = _load_agn_pivot_context_from_checkpoint(
                    r,
                    checkpoint_file=checkpoint_file,
                    use_alpha_lambda_term=use_alpha_lambda_term,
                    use_eta_sigma_term=use_eta_sigma_term,
                )
                _validate_agn_pivot_checkpoint_reference_provenance(
                    stored_pivot_context,
                    r,
                    checkpoint_file=checkpoint_file,
                )
            if resume_replot_with_cuts:
                r = _remap_resume_replot_checkpoint(
                    r,
                    checkpoint_file,
                    df_agn,
                    ndim,
                )
                print(
                    "Resume-replot with cuts: loaded posterior samples and remapped "
                    f"per-AGN arrays to {len(df_agn)} current AGN by object_id."
                )
            else:
                validate_resume_checkpoint(
                    r,
                    checkpoint_file=checkpoint_file,
                    ndim=ndim,
                    n_agn=len(agn_data["z"]),
                )
            if not only_sna:
                if stored_pivot_context != agn_pivot_context:
                    raise RuntimeError(
                        f"AGN checkpoint '{checkpoint_file}' pivot metadata does not "
                        "exactly match the immutable pivot context for this run."
                    )
        except Exception as exc:
            if resume_replot_with_cuts:
                raise RuntimeError(
                    f"Failed to resume/replot from checkpoint '{checkpoint_file}' with the current AGN cuts. "
                    "The checkpoint must match the current parameterization and include object_id_fit_selection "
                    "metadata covering every current AGN."
                ) from exc
            else:
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
                agn_pivot_context=agn_pivot_context,
                completeness_params=completeness_params,
                only_sna=only_sna,
                only_agn=only_agn,
                use_full_cov=use_full_cov,
                use_planck_h0_prior=use_planck_h0_prior,
                use_planck_om_prior=use_planck_om_prior,
                use_ceph_dist_calibration=not disable_ceph_dist_calibration,
                use_alpha_lambda_term=use_alpha_lambda_term,
                use_eta_sigma_term=use_eta_sigma_term,
                use_redshift_log_f_term=use_redshift_log_f_term,
                early_de_guard=early_de_guard,
            )
            ptform_kwargs = dict(priors=priors, model_labels=model_labels)
            loglike_func = (
                log_likelihood_nearbylcs
                if agn_calibrators_data is not None and not only_sna
                else log_likelihood
            )
            sampler = DynamicNestedSampler(
                loglike_func,
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
            warm_start = warm_start_flat_samples is not None
            speed_settings = get_dynesty_speed_settings(speed, ndim, warm_start=warm_start)
            live_points = None
            if warm_start:
                print(
                    "[Warning] Starting warm-start sigma-clipped top-up run "
                    f"with {speed!r} settings: {speed_settings}"
                )
                live_points = build_warm_start_live_points(
                    warm_start_flat_samples,
                    priors=priors,
                    model_labels=model_labels,
                    nlive=speed_settings["nlive_init"],
                    loglike_func=loglike_func,
                    logl_kwargs=logl_kwargs,
                )
            else:
                print(f"Starting {speed} run with settings: {speed_settings}")
            sampler.run_nested(
                print_progress=True,
                dlogz_init=speed_settings["dlogz_init"],
                n_effective=speed_settings["n_effective"],
                nlive_init=speed_settings["nlive_init"],
                nlive_batch=speed_settings["nlive_batch"],
                live_points=live_points,
            )


        results = sampler.results
        if compare_sigma_only or minimal_plots:
            print("Skipping dynesty plot generation.")
        else:
            print("Plotting full dynesty corner...")
            plot_dynesty(
                sampler.results,
                cosmo_model,
                plot_path,
                only_sna=only_sna,
                only_agn=only_agn,
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

        checkpoint_payload = dict(
            flat_samples=flat_samples,
            dmi_max_w=dmi_max_w,
            dmi_posterior_median=dmi_posterior_median,
            dmi_posterior_sigma=dmi_posterior_sigma,
            dmi_selection_sigma_posterior_median=dmi_selection_sigma_posterior_median,
            object_id_fit_selection=df_agn["object_id"].astype(str).to_numpy(),
            sigma_clip_pass_stage="single",
            logZ=logZ,
            logZerr=logZerr,
            logZ_is_approximate=bool(logZ_is_approximate),
            integrals_max_w=integrals_max_w,
        )
        if not only_sna:
            checkpoint_payload.update(
                _agn_pivot_checkpoint_payload(agn_pivot_context)
            )
        save_chains(checkpoint_file, **checkpoint_payload)

        # Bin dmi in redshift
        # Interpolate dmi vs redshift for smooth plotting or further analysis (no binning)
        #dmi_interp = interp1d(z, dmi_max_w)
    debias_magnitude = (
        df_agn[COMPLETENESS_MAG_COL]
        if completeness
        else df_agn["apparent_mag_2500"]
    )
    dm_interp = make_dm_function(
        debias_magnitude.values,
        df_agn['z'].values,
        dmi_posterior_median,
        f_host_2500_psf=df_agn[COMPLETENESS_FHOST_COL].values if COMPLETENESS_FHOST_COL in df_agn.columns else None,
        alpha_lambda=df_agn["alpha_lambda"].values if "alpha_lambda" in df_agn.columns else None,
    )
    dmi_selection_sigma_interp = None
    if dmi_selection_sigma_posterior_median is not None:
        dmi_selection_sigma_interp = make_dm_function(
            df_agn[COMPLETENESS_MAG_COL].values,
            df_agn['z'].values,
            dmi_selection_sigma_posterior_median,
            f_host_2500_psf=df_agn[COMPLETENESS_FHOST_COL].values if COMPLETENESS_FHOST_COL in df_agn.columns else None,
            alpha_lambda=df_agn["alpha_lambda"].values if "alpha_lambda" in df_agn.columns else None,
        )

    if compare_sigma_only or minimal_plots:
        print("Skipping completeness diagnostics plots.")
    else:
        print("Plotting completeness diagnostics...")
        plot_completeness_diagnostics(
            dmi_posterior_median,
            agn_data['z'],
            agn_data[COMPLETENESS_MAG_COL] if completeness else agn_data["apparent_mag_2500"],
            integrals_max_w,
            plot_path=plot_path,
            z_range=z_range,
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
               N=None, resume=False, only_sna=False, only_agn=False, speed="production", 
               cosmo_model_joint_samples={}, cosmo_model_sna_samples={},
               verbose=True,
               z_range=(0.44, 3.16),
               skip_plots=False, residuals_sigma_clip=None, df_calibrators=None,
               disable_sigma_clip_pass=False, sigma_clip_threshold=3.0,
               resume_stage="both",
               sigma_clip_second_pass_mode="warm",
               prefix="default", uniform_redshift_distribution=False,
               completeness_sim_file=DEFAULT_COMPLETENESS_SIM_FILE,
               completeness_mode="2d",
               completeness_magnitude="dereddened",
               compare_sigma_only=False,
               minimal_plots=False,
               disable_ceph_dist_calibration=False,
               use_planck_h0_prior=False,
               use_planck_om_prior=False,
               use_alpha_lambda_term=False,
               use_eta_sigma_term=False,
               use_redshift_log_f_term=False,
               early_de_guard=False,
               resume_replot_with_cuts=False,
               agn_pivot_context=None):
    validate_completeness_mode(completeness_mode)
    completeness_magnitude = normalize_completeness_magnitude(
        completeness_magnitude
    )
    df_agn = prepare_completeness_magnitude_columns(
        df_agn,
        completeness_magnitude,
    )
    df_agn_all = prepare_completeness_magnitude_columns(
        df_agn_all,
        completeness_magnitude,
    )
    speed = normalize_speed(speed)
    _fit_mode_label(only_sna, only_agn)
    use_planck_h0_prior = use_planck_h0_prior or disable_ceph_dist_calibration
    sigma_clip_second_pass_mode = normalize_sigma_clip_second_pass_mode(sigma_clip_second_pass_mode)
    run_tag = make_run_tag(
        cosmo_model,
        only_sna,
        speed,
        N,
        z_range,
        only_agn=only_agn,
        completeness=completeness,
        completeness_mode=completeness_mode,
        completeness_magnitude=completeness_magnitude,
        disable_ceph_dist_calibration=disable_ceph_dist_calibration,
        use_planck_h0_prior=use_planck_h0_prior,
        use_planck_om_prior=use_planck_om_prior,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    plot_path = f"plots/hubble/{prefix}/{run_tag}"
    os.makedirs(plot_path, exist_ok=True)
    print(f"Saving plots to ", plot_path)
    if completeness:
        print(
            "Completeness magnitude: "
            f"{completeness_magnitude} "
            f"({df_agn.attrs['completeness_magnitude_source']})."
        )
        if completeness_sim_file is None:
            if resume_replot_with_cuts:
                print("Completeness diagnostics enabled with a freshly generated mock catalog.")
            else:
                print("Completeness enabled with a freshly generated mock catalog.")
            completeness_area_deg2 = estimate_sky_box_area_deg2(df_agn_all)
            completeness_sim_file = generate_fresh_completeness_sim_file(
                plot_path,
                area_deg2=completeness_area_deg2,
            )
        else:
            print(f"Completeness enabled with mock catalog file: {completeness_sim_file}")

    disable_sigma_clip_pass = bool(disable_sigma_clip_pass)
    if resume_replot_with_cuts:
        if not resume:
            raise ValueError("resume_replot_with_cuts=True requires resume to point at an existing posterior H5 file.")
        if not disable_sigma_clip_pass and not only_sna:
            print(
                "[INFO] resume_replot_with_cuts=True: bypassing the internal two-pass "
                "sigma-clipping refit and regenerating plots for the current cut sample."
            )
        disable_sigma_clip_pass = True
    apply_two_pass_sigma_clip = (not disable_sigma_clip_pass) and (not only_sna)
    if apply_two_pass_sigma_clip and skip_plots:
        raise ValueError("Two-pass sigma clipping requires skip_plots=False so Hubble residuals can be computed.")
    if apply_two_pass_sigma_clip and uniform_redshift_distribution:
        raise ValueError(
            "Two-pass sigma clipping does not support uniform_redshift_distribution=True "
            "because fit rows are resampled with replacement."
        )
    resume_stage = _normalize_resume_stage(resume_stage)
    if not apply_two_pass_sigma_clip and resume_stage != "both":
        print(
            f"[INFO] Ignoring resume_stage={resume_stage!r} because two-pass sigma clipping is disabled."
        )

    df_agn_full_sample_preclip = df_agn.copy()
    df_agn_pass2_plot_sample = df_agn_full_sample_preclip.copy()
    if resume_replot_with_cuts:
        if uniform_redshift_distribution:
            raise ValueError("--resume_replot_with_cuts does not support uniform_redshift_distribution=True.")
        if N is not None:
            print(
                "[INFO] --resume_replot_with_cuts uses the current cuts for the fit/debias "
                "selection; N is ignored for the resumed fit selection."
            )
        print(
            "[INFO] --resume_replot_with_cuts will use the checkpoint object_id list "
            "only to remap and validate saved per-AGN debias arrays."
        )
        df_agn_pass2_fit_selection = _select_agn_fit_selection(
            df_agn_pass2_plot_sample,
            z_range=z_range,
            N=None,
            uniform_redshift_distribution=False,
        )
    else:
        df_agn_pass2_fit_selection = _select_agn_fit_selection(
            df_agn_pass2_plot_sample,
            z_range=z_range,
            N=N,
            uniform_redshift_distribution=uniform_redshift_distribution,
        )
    pass1_diagnostics_df = None
    keep_mask_full = None
    checkpoint_paths = _build_checkpoint_paths(prefix, run_tag)
    pass1_checkpoint_file = checkpoint_paths["pass1"]
    pass2_checkpoint_file = checkpoint_paths["pass2"]
    single_checkpoint_file = checkpoint_paths["single"]
    initial_agn_fit_selection = df_agn_pass2_fit_selection.copy()

    if only_sna:
        if agn_pivot_context is not None:
            raise ValueError("SNe-only runs must not receive an AGN pivot context.")
    elif resume:
        if apply_two_pass_sigma_clip:
            pivot_checkpoint_file = _resolve_two_pass_resume_checkpoint(
                resume,
                resume_stage,
                checkpoint_paths,
            )
        else:
            pivot_checkpoint_file = resolve_resume_checkpoint_path(
                resume,
                single_checkpoint_file,
            )
        pivot_checkpoint_results = load_chains(pivot_checkpoint_file)
        stored_pivot_context = _load_agn_pivot_context_from_checkpoint(
            pivot_checkpoint_results,
            checkpoint_file=pivot_checkpoint_file,
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
        )
        _validate_agn_pivot_checkpoint_reference_provenance(
            stored_pivot_context,
            pivot_checkpoint_results,
            checkpoint_file=pivot_checkpoint_file,
        )
        if agn_pivot_context is None:
            agn_pivot_context = stored_pivot_context
        elif agn_pivot_context != stored_pivot_context:
            raise RuntimeError(
                f"AGN checkpoint '{pivot_checkpoint_file}' does not share the "
                "same immutable pivot context as the other cosmology runs."
            )
        _validate_agn_pivot_context_for_reference(
            agn_pivot_context,
            initial_agn_fit_selection,
            z_range=z_range,
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
            require_reference_ids=not resume_replot_with_cuts,
        )
    else:
        if agn_pivot_context is None:
            agn_pivot_context = build_agn_pivot_context(
                initial_agn_fit_selection,
                z_range,
                use_alpha_lambda_term=use_alpha_lambda_term,
                use_eta_sigma_term=use_eta_sigma_term,
            )
        _validate_agn_pivot_context_for_reference(
            agn_pivot_context,
            initial_agn_fit_selection,
            z_range=z_range,
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
            require_reference_ids=True,
        )
    direct_completeness_params = None

    def _get_direct_completeness_params():
        nonlocal direct_completeness_params
        if direct_completeness_params is None:
            direct_completeness_params = _build_completeness_params(
                df_agn_full_sample_preclip,
                df_agn_all,
                completeness=completeness,
                completeness_mode=completeness_mode,
                completeness_sim_file=completeness_sim_file,
                plot_path=plot_path,
                plot=False,
            )
        return direct_completeness_params

    skip_pass1_sampling = False
    pass1_resume_arg = False
    pass2_resume_arg = False
    selected_resume_checkpoint = None
    selected_resume_results = None
    pass2_warm_start_flat_samples = None

    if apply_two_pass_sigma_clip:
        if resume:
            selected_resume_checkpoint = _resolve_two_pass_resume_checkpoint(
                resume,
                resume_stage,
                checkpoint_paths,
            )
            selected_resume_results = load_chains(selected_resume_checkpoint)
            selected_resume_stage = _checkpoint_stage_from_results(selected_resume_results)
            if resume_stage == "pass1":
                if selected_resume_stage == "pass2":
                    raise RuntimeError(
                        f"Checkpoint '{selected_resume_checkpoint}' is a pass-2 checkpoint and cannot be used with resume_stage='pass1'."
                    )
                pass1_resume_arg = selected_resume_checkpoint
            elif selected_resume_stage in {"pass1", "pass2"}:
                extracted_state = _extract_pass1_state_from_checkpoint(
                    selected_resume_results,
                    selected_resume_checkpoint,
                    df_agn_full_sample_preclip,
                    sigma_clip_threshold=sigma_clip_threshold,
                )
                keep_mask_full = extracted_state["keep_mask_full"]
                pass1_diagnostics_df = extracted_state["pass1_diagnostics_df"]
                skip_pass1_sampling = True
                if selected_resume_stage == "pass2":
                    pass2_resume_arg = selected_resume_checkpoint
            elif resume_stage == "pass2":
                raise RuntimeError(
                    f"Checkpoint '{selected_resume_checkpoint}' does not contain embedded pass-1 clipping state needed for resume_stage='pass2'."
                )
            else:
                pass1_resume_arg = selected_resume_checkpoint

        if not skip_pass1_sampling:
            df_agn_pass1_fit_selection = df_agn_pass2_fit_selection.copy()
            print(
                f"Running Hubble fit for {len(df_agn_pass1_fit_selection)} AGN "
                f"with sigma clipping threshold |mu_zscore| < {sigma_clip_threshold:.2f}."
            )
            (
                flat_samples_pass1,
                model_labels_pass1,
                dm_interp_pass1,
                dmi_selection_sigma_interp_pass1,
                _logZ_pass1,
                _logZerr_pass1,
                dmi_posterior_median_pass1,
                dmi_posterior_sigma_pass1,
                dmi_selection_sigma_posterior_median_pass1,
                _age_pass1,
                _age_err_pass1,
            ) = _run_fit_stage(
                df_agn_pass1_fit_selection,
                df_agn_all,
                df_pantheon,
                _sna_L,
                _sna_Lower,
                _sna_LogdetCov,
                agn_pivot_context=agn_pivot_context,
                df_calibrators=df_calibrators,
                cosmo_model=cosmo_model,
                only_sna=only_sna,
                only_agn=only_agn,
                completeness=completeness,
                use_full_cov=use_full_cov,
                z_range=z_range,
                N=N,
                resume=pass1_resume_arg,
                speed=speed,
                prefix=prefix,
                completeness_sim_file=completeness_sim_file,
                completeness_mode=completeness_mode,
                compare_sigma_only=compare_sigma_only,
                minimal_plots=minimal_plots,
                disable_ceph_dist_calibration=disable_ceph_dist_calibration,
                use_planck_h0_prior=use_planck_h0_prior,
                use_planck_om_prior=use_planck_om_prior,
                use_alpha_lambda_term=use_alpha_lambda_term,
                use_eta_sigma_term=use_eta_sigma_term,
                use_redshift_log_f_term=use_redshift_log_f_term,
                early_de_guard=early_de_guard,
                checkpoint_file_override=pass1_checkpoint_file,
                resume_replot_with_cuts=False,
                df_agn_completeness=df_agn_full_sample_preclip,
            )
            (
                dmi_posterior_median_pass1_full,
                dmi_posterior_sigma_pass1_full,
                dmi_selection_sigma_pass1_full,
            ) = _compute_direct_full_sample_completeness_summaries(
                flat_samples_pass1,
                df_agn_fit_selection=df_agn_pass1_fit_selection,
                df_agn_plot_sample=df_agn_full_sample_preclip,
                df_pantheon=df_pantheon,
                _sna_L=_sna_L,
                _sna_Lower=_sna_Lower,
                _sna_LogdetCov=_sna_LogdetCov,
                cosmo_model=cosmo_model,
                completeness_params=_get_direct_completeness_params(),
                z_pivot_agn=z_pivot_agn,
                agn_pivot_context=agn_pivot_context,
                use_full_cov=use_full_cov,
                disable_ceph_dist_calibration=disable_ceph_dist_calibration,
                use_planck_h0_prior=use_planck_h0_prior,
                use_planck_om_prior=use_planck_om_prior,
                only_agn=only_agn,
                use_alpha_lambda_term=use_alpha_lambda_term,
                use_eta_sigma_term=use_eta_sigma_term,
                use_redshift_log_f_term=use_redshift_log_f_term,
                early_de_guard=early_de_guard,
            )
            pass1_residuals_full, pass1_clipping_sigma_full, _, _, _ = plot_hubble(
                flat_samples_pass1,
                df_agn_full_sample_preclip,
                df_pantheon,
                cosmo_model=cosmo_model,
                z_pivot_agn=z_pivot_agn,
                show_true=False,
                show=False,
                debias=True,
                dm_interp=dm_interp_pass1,
                plot_path=plot_path,
                cosmo_model_samples=cosmo_model_joint_samples,
                verbose=verbose,
                residuals_sigma_clip=residuals_sigma_clip,
                df_calibrators=df_calibrators,
                dmi_values=dmi_posterior_median_pass1_full,
                dmi_sigma=dmi_posterior_sigma_pass1_full,
                dmi_selection_sigma=dmi_selection_sigma_pass1_full,
                filename="hubble_diagram_pass1_full_sample_debiased.pdf",
                residuals_csv_filename=None if minimal_plots else "hubble_plot_residuals_pass1.csv",
                compute_only=minimal_plots,
                sigma_clip_threshold=sigma_clip_threshold,
                z_range=z_range,
                use_alpha_lambda_term=use_alpha_lambda_term,
                use_eta_sigma_term=use_eta_sigma_term,
                use_redshift_log_f_term=use_redshift_log_f_term,
                only_agn=only_agn,
                agn_pivot_context=agn_pivot_context,
            )
            pass1_diagnostics_df, keep_mask_full = _build_sigma_clip_diagnostics(
                df_agn_full_sample_preclip,
                pass1_residuals_full,
                pass1_clipping_sigma_full,
                sigma_clip_threshold=sigma_clip_threshold,
            )
            df_agn_pass2_plot_sample = df_agn_full_sample_preclip.loc[keep_mask_full].copy()
            df_agn_pass2_fit_selection = _select_agn_fit_selection(
                df_agn_pass2_plot_sample,
                z_range=z_range,
                N=N,
                uniform_redshift_distribution=uniform_redshift_distribution,
            )
            if sigma_clip_second_pass_mode == "warm":
                pass2_warm_start_flat_samples = flat_samples_pass1
            _write_sigma_clip_diagnostics(
                pass1_diagnostics_df,
                plot_path,
                residuals_filename="residuals_pass1.csv",
                clipped_filename="clipped_objects_pass1.csv",
            )
            _write_sigma_clip_membership_audit(
                df_agn_full_sample_preclip,
                keep_mask_full,
                plot_path,
                filename="sigma_clip_membership_pass1.csv",
                df_agn_fit_selection=df_agn_pass2_fit_selection,
            )
            _write_stage_checkpoint(
                pass1_checkpoint_file,
                source_checkpoint_file=pass1_resume_arg if pass1_resume_arg else pass1_checkpoint_file,
                sigma_clip_pass_stage="pass1",
                sigma_clip_threshold=sigma_clip_threshold,
                df_agn_full_sample=df_agn_full_sample_preclip,
                df_agn_plot_sample=df_agn_pass2_plot_sample,
                df_agn_fit_selection=df_agn_pass1_fit_selection,
                df_agn_initial_fit_selection=initial_agn_fit_selection,
                keep_mask_full=keep_mask_full,
                pass1_diagnostics_df=pass1_diagnostics_df,
            )
            n_before = len(df_agn_full_sample_preclip)
            n_after = int(np.count_nonzero(keep_mask_full))
            print(f"Sigma-clipping pass 1 kept {n_after} / {n_before} AGN and clipped {n_before - n_after}.")
            clipped_mask_pass1_full = ~keep_mask_full
            if not minimal_plots:
                plot_hubble(
                    flat_samples_pass1,
                    df_agn_full_sample_preclip,
                    df_pantheon,
                    cosmo_model=cosmo_model,
                    z_pivot_agn=z_pivot_agn,
                    show_true=False,
                    show=False,
                    debias=True,
                    dm_interp=dm_interp_pass1,
                    plot_path=plot_path,
                    cosmo_model_samples=cosmo_model_joint_samples,
                    verbose=verbose,
                    residuals_sigma_clip=residuals_sigma_clip,
                    df_calibrators=df_calibrators,
                    dmi_values=dmi_posterior_median_pass1_full,
                    dmi_sigma=dmi_posterior_sigma_pass1_full,
                    dmi_selection_sigma=dmi_selection_sigma_pass1_full,
                    clipped_mask=clipped_mask_pass1_full,
                    filename="hubble_diagram_pass1_full_sample_clipped_debiased.pdf",
                    residuals_csv_filename="hubble_plot_residuals_pass1_clipped.csv",
                    sigma_clip_threshold=sigma_clip_threshold,
                    z_range=z_range,
                    use_alpha_lambda_term=use_alpha_lambda_term,
                    use_eta_sigma_term=use_eta_sigma_term,
                    use_redshift_log_f_term=use_redshift_log_f_term,
                    only_agn=only_agn,
                    agn_pivot_context=agn_pivot_context,
                )
            if resume_stage == "pass1":
                print("Stopping after resumed pass-1 fit as requested by resume_stage='pass1'.")
                return flat_samples_pass1, model_labels_pass1, dm_interp_pass1, _logZ_pass1, _logZerr_pass1, None, _age_pass1, _age_err_pass1
        else:
            df_agn_pass2_plot_sample = df_agn_full_sample_preclip.loc[keep_mask_full].copy()
            df_agn_pass2_fit_selection = _select_agn_fit_selection(
                df_agn_pass2_plot_sample,
                z_range=z_range,
                N=N,
                uniform_redshift_distribution=uniform_redshift_distribution,
            )
            if sigma_clip_second_pass_mode == "warm" and not pass2_resume_arg:
                if selected_resume_results is None or "flat_samples" not in selected_resume_results:
                    raise RuntimeError(
                        "Cannot warm-start sigma-clipped pass 2 from the resumed pass-1 checkpoint "
                        "because it does not contain flat_samples."
                    )
                expected_pass1_fit_selection = _select_agn_fit_selection(
                    df_agn_full_sample_preclip,
                    z_range=z_range,
                    N=N,
                    uniform_redshift_distribution=uniform_redshift_distribution,
                )
                if "object_id_fit_selection" not in selected_resume_results:
                    raise RuntimeError(
                        f"Cannot warm-start sigma-clipped pass 2 from '{selected_resume_checkpoint}' "
                        "because the pass-1 checkpoint does not contain object_id_fit_selection. "
                        "Rerun pass 1 with the current code."
                    )
                saved_pass1_ids = _normalize_object_id_array(
                    selected_resume_results["object_id_fit_selection"],
                    field_name="object_id_fit_selection",
                    checkpoint_file=selected_resume_checkpoint,
                )
                expected_pass1_ids = expected_pass1_fit_selection["object_id"].astype(str).to_numpy()
                if saved_pass1_ids.shape != expected_pass1_ids.shape or not np.array_equal(saved_pass1_ids, expected_pass1_ids):
                    raise RuntimeError(
                        f"Cannot warm-start sigma-clipped pass 2 from '{selected_resume_checkpoint}' "
                        "because its pass-1 object_id_fit_selection does not match the current pass-1 fit selection."
                    )
                validate_resume_checkpoint(
                    selected_resume_results,
                    checkpoint_file=selected_resume_checkpoint,
                    ndim=len(get_model_params(
                        cosmo_model,
                        only_sna=only_sna,
                        only_agn=only_agn,
                        use_planck_h0_prior=use_planck_h0_prior,
                        use_planck_om_prior=use_planck_om_prior,
                        use_alpha_lambda_term=use_alpha_lambda_term,
                        use_eta_sigma_term=use_eta_sigma_term,
                        use_redshift_log_f_term=use_redshift_log_f_term,
                    )[1]),
                    n_agn=len(expected_pass1_fit_selection),
                )
                pass2_warm_start_flat_samples = selected_resume_results["flat_samples"]
            _write_sigma_clip_diagnostics(
                pass1_diagnostics_df,
                plot_path,
                residuals_filename="residuals_pass1.csv",
                clipped_filename="clipped_objects_pass1.csv",
            )
            _write_sigma_clip_membership_audit(
                df_agn_full_sample_preclip,
                keep_mask_full,
                plot_path,
                filename="sigma_clip_membership_pass1.csv",
                df_agn_fit_selection=df_agn_pass2_fit_selection,
            )

        # Stage 2 removes all pass-1 sigma-clipped objects globally from the
        # plotting sample, while keeping outside-z survivors in the final plots.
        # Only the surviving in-range objects are used for the stage-2 fit.
        if not uniform_redshift_distribution:
            expected_pass2_fit = _select_agn_fit_selection(
                df_agn_pass2_plot_sample,
                z_range=z_range,
                N=N,
                uniform_redshift_distribution=False,
            )
            if not expected_pass2_fit["object_id"].tolist() == df_agn_pass2_fit_selection["object_id"].tolist():
                raise RuntimeError(
                    "Stage-2 fit selection is inconsistent with the surviving in-range plot sample."
                )
        if pass2_resume_arg:
            print("Resuming second Hubble-fit pass from the pass-2 checkpoint.")
        elif sigma_clip_second_pass_mode == "warm":
            print("Running warm-start second Hubble-fit pass on the clipped AGN sample.")
        else:
            print("Running fresh second Hubble-fit pass on the clipped AGN sample.")

    if uniform_redshift_distribution:
        if not (compare_sigma_only or minimal_plots):
            plot_redshift_histograms(df_pantheon, df_agn_pass2_fit_selection, xscale="linear", plot_path=plot_path, only_agn=only_agn)
    else:
        if not (compare_sigma_only or minimal_plots):
            plot_redshift_histograms(df_pantheon, df_agn_pass2_plot_sample, xscale="log", plot_path=plot_path, only_agn=only_agn)

    if not (compare_sigma_only or minimal_plots):
        plot_delta_m_flux_recal_vs_redshift(df_agn_pass2_fit_selection, plot_path=plot_path)

    if not only_sna:
        report_pivots(
            df_agn_pass2_fit_selection,
            agn_pivot_context=agn_pivot_context,
        )

    warm_start_pass2 = (
        apply_two_pass_sigma_clip
        and sigma_clip_second_pass_mode == "warm"
        and not pass2_resume_arg
    )
    if warm_start_pass2 and pass2_warm_start_flat_samples is None:
        raise RuntimeError(
            "Sigma-clipped pass 2 is configured for warm-start mode, but no pass-1 posterior samples are available."
        )
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
        age,
        age_err,
    ) = _run_fit_stage(
        df_agn_pass2_fit_selection,
        df_agn_all,
        df_pantheon,
        _sna_L,
        _sna_Lower,
        _sna_LogdetCov,
        agn_pivot_context=agn_pivot_context,
        df_calibrators=df_calibrators,
        cosmo_model=cosmo_model,
        only_sna=only_sna,
        only_agn=only_agn,
        completeness=completeness,
        use_full_cov=use_full_cov,
        z_range=z_range,
        N=N,
        resume=pass2_resume_arg if apply_two_pass_sigma_clip else resume,
        speed=speed,
        prefix=prefix,
        checkpoint_file_override=pass2_checkpoint_file if apply_two_pass_sigma_clip else single_checkpoint_file,
        completeness_sim_file=completeness_sim_file,
        completeness_mode=completeness_mode,
        compare_sigma_only=compare_sigma_only,
        minimal_plots=minimal_plots,
        disable_ceph_dist_calibration=disable_ceph_dist_calibration,
        use_planck_h0_prior=use_planck_h0_prior,
        use_planck_om_prior=use_planck_om_prior,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
        early_de_guard=early_de_guard,
        resume_replot_with_cuts=resume_replot_with_cuts,
        warm_start_flat_samples=pass2_warm_start_flat_samples if warm_start_pass2 else None,
        logZ_is_approximate=warm_start_pass2,
        df_agn_completeness=df_agn_full_sample_preclip,
    )
    if apply_two_pass_sigma_clip:
        _write_stage_checkpoint(
            pass2_checkpoint_file,
            source_checkpoint_file=pass2_resume_arg if pass2_resume_arg else pass2_checkpoint_file,
            sigma_clip_pass_stage="pass2",
            sigma_clip_threshold=sigma_clip_threshold,
            sigma_clip_second_pass_mode=sigma_clip_second_pass_mode,
            sigma_clip_warm_start_from_pass1=warm_start_pass2,
            logZ_is_approximate=warm_start_pass2,
            df_agn_full_sample=df_agn_full_sample_preclip,
            df_agn_plot_sample=df_agn_pass2_plot_sample,
            df_agn_fit_selection=df_agn_pass2_fit_selection,
            df_agn_initial_fit_selection=initial_agn_fit_selection,
            keep_mask_full=keep_mask_full,
            pass1_diagnostics_df=pass1_diagnostics_df,
        )

    if compare_sigma_only or skip_plots or only_sna:
        print("Skipping plots, returning results...")
        return flat_samples, model_labels, dm_interp, logZ, logZerr, None, age, age_err

    if minimal_plots:
        (
            dmi_posterior_median_full,
            dmi_posterior_sigma_full,
            dmi_selection_sigma_full,
        ) = _compute_direct_full_sample_completeness_summaries(
            flat_samples,
            df_agn_fit_selection=df_agn_pass2_fit_selection,
            df_agn_plot_sample=df_agn_pass2_plot_sample,
            df_pantheon=df_pantheon,
            _sna_L=_sna_L,
            _sna_Lower=_sna_Lower,
            _sna_LogdetCov=_sna_LogdetCov,
            cosmo_model=cosmo_model,
            completeness_params=_get_direct_completeness_params(),
            z_pivot_agn=z_pivot_agn,
            agn_pivot_context=agn_pivot_context,
            use_full_cov=use_full_cov,
            disable_ceph_dist_calibration=disable_ceph_dist_calibration,
            use_planck_h0_prior=use_planck_h0_prior,
            use_planck_om_prior=use_planck_om_prior,
            only_agn=only_agn,
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
            use_redshift_log_f_term=use_redshift_log_f_term,
            early_de_guard=early_de_guard,
        )
        df_agn_agn_likelihood_chi2_selection = df_agn_pass2_fit_selection
        if resume_replot_with_cuts:
            df_agn_agn_likelihood_chi2_selection = df_agn_pass2_plot_sample[
                df_agn_pass2_plot_sample["z"].between(z_range[0], z_range[1])
            ].copy()
        chisq_red_agn_likelihood_space, _ = compute_agn_likelihood_space_reduced_chi2(
            flat_samples,
            model_labels,
            df_agn_agn_likelihood_chi2_selection,
            cosmo_model,
            z_pivot_agn=z_pivot_agn,
            agn_pivot_context=agn_pivot_context,
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
            use_redshift_log_f_term=use_redshift_log_f_term,
        )
        df_agn_agn_likelihood_chi2_selection_zgt1 = df_agn_agn_likelihood_chi2_selection[
            df_agn_agn_likelihood_chi2_selection["z"].between(
                1.0, z_range[1], inclusive="right"
            )
        ]
        chisq_red_agn_likelihood_space_zgt1, _ = compute_agn_likelihood_space_reduced_chi2(
            flat_samples,
            model_labels,
            df_agn_agn_likelihood_chi2_selection_zgt1,
            cosmo_model,
            z_pivot_agn=z_pivot_agn,
            agn_pivot_context=agn_pivot_context,
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
            use_redshift_log_f_term=use_redshift_log_f_term,
        )
        (
            debiased_residuals,
            _debiased_clipping_sigma,
            _mu_pred_median_debiased,
            _mu_pred_std_debiased,
            _mu_pred_std_debiased_with_scatter,
        ) = plot_hubble(
            flat_samples,
            df_agn_pass2_plot_sample,
            df_pantheon,
            cosmo_model=cosmo_model,
            z_pivot_agn=z_pivot_agn,
            show_true=False,
            show=False,
            debias=True,
            dm_interp=dm_interp,
            plot_path=plot_path,
            cosmo_model_samples=cosmo_model_joint_samples,
            verbose=verbose,
            residuals_sigma_clip=residuals_sigma_clip,
            df_calibrators=df_calibrators,
            dmi_values=dmi_posterior_median_full,
            dmi_sigma=dmi_posterior_sigma_full,
            dmi_selection_sigma=dmi_selection_sigma_full,
            residuals_csv_filename="hubble_plot_residuals.csv",
            sigma_clip_threshold=sigma_clip_threshold if apply_two_pass_sigma_clip else None,
            z_range=z_range,
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
            use_redshift_log_f_term=use_redshift_log_f_term,
            only_agn=only_agn,
            agn_likelihood_space_chi2=chisq_red_agn_likelihood_space,
            agn_likelihood_space_chi2_zgt1=chisq_red_agn_likelihood_space_zgt1,
            agn_pivot_context=agn_pivot_context,
        )
        print(
            "minimal_plots=True: retained hubble_diagram_debiased.pdf and "
            "hubble_plot_residuals.csv; skipped other figures."
        )
        return (
            flat_samples,
            model_labels,
            dm_interp,
            logZ,
            logZerr,
            debiased_residuals,
            age,
            age_err,
        )

    # if only_sna:
    #     print("Skipping AGN-specific plots for SNe-only run.")
    #     return flat_samples, model_labels, dm_interp, logZ, logZerr, None, age, age_err

    alpha_agn_idx = model_labels.index("alpha_agn")
    alpha_agn_median = float(np.nanmedian(flat_samples[:, alpha_agn_idx]))
    plot_sigma_uv_mpred_correction(
        df_agn_pass2_plot_sample,
        alpha_agn_median,
        plot_path=plot_path,
        show=False,
        filename="sigma_uv_mpred_correction_postcut.pdf",
    )

    (
        dmi_posterior_median_full_direct,
        dmi_posterior_sigma_full_direct,
        dmi_selection_sigma_full_direct,
    ) = _compute_direct_full_sample_completeness_summaries(
        flat_samples,
        df_agn_fit_selection=df_agn_pass2_fit_selection,
        df_agn_plot_sample=df_agn_pass2_plot_sample,
        df_pantheon=df_pantheon,
        _sna_L=_sna_L,
        _sna_Lower=_sna_Lower,
        _sna_LogdetCov=_sna_LogdetCov,
        cosmo_model=cosmo_model,
        completeness_params=_get_direct_completeness_params(),
        z_pivot_agn=z_pivot_agn,
        agn_pivot_context=agn_pivot_context,
        use_full_cov=use_full_cov,
        disable_ceph_dist_calibration=disable_ceph_dist_calibration,
        use_planck_h0_prior=use_planck_h0_prior,
        use_planck_om_prior=use_planck_om_prior,
        only_agn=only_agn,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
        early_de_guard=early_de_guard,
    )
    dmi_posterior_median_full = dmi_posterior_median_full_direct
    dmi_posterior_sigma_full = dmi_posterior_sigma_full_direct
    dmi_selection_sigma_full = dmi_selection_sigma_full_direct

    print("Plotting predicted L2500 vs ...")
    plot_predicted_L2500_vs_sigmahat(
        flat_samples,
        df_agn_pass2_plot_sample,
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
        agn_pivot_context=agn_pivot_context,
    )
    plot_predicted_L2500_vs_sigmahat(
        flat_samples,
        df_agn_pass2_plot_sample,
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
        agn_pivot_context=agn_pivot_context,
    )
    plot_predicted_L2500_vs_sigmahat(
        flat_samples,
        df_agn_pass2_plot_sample,
        cosmo_model=cosmo_model,
        z_pivot_agn=z_pivot_agn,
        debias=True,
        dm_interp=dm_interp,
        dmi_values=dmi_posterior_median_full,
        dmi_selection_sigma=dmi_selection_sigma_full,
        show_residuals=False,
        show=False,
        plot_path=plot_path,
        df_calibrators=df_calibrators,
        z_range=z_range,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
        agn_pivot_context=agn_pivot_context,
    )
    L_residuals_debiased, L_pred_std_debiased = plot_predicted_L2500_vs_sigmahat(
        flat_samples,
        df_agn_pass2_plot_sample,
        cosmo_model=cosmo_model,
        z_pivot_agn=z_pivot_agn,
        debias=True,
        dm_interp=dm_interp,
        dmi_values=dmi_posterior_median_full,
        dmi_selection_sigma=dmi_selection_sigma_full,
        show_residuals=True,
        show=False,
        plot_path=plot_path,
        df_calibrators=df_calibrators,
        z_range=z_range,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
        agn_pivot_context=agn_pivot_context,
    )
    plot_L2500_vs_sigma_tau_separate(
        flat_samples,
        df_agn_pass2_plot_sample,
        cosmo_model=cosmo_model,
        z_pivot_agn=z_pivot_agn,
        debias=True,
        dm_interp=dm_interp,
        dmi_values=dmi_posterior_median_full,
        dmi_selection_sigma=dmi_selection_sigma_full,
        show_residuals=False,
        show=False,
        plot_path=plot_path,
        z_range=z_range,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
        agn_pivot_context=agn_pivot_context,
    )
    plot_L2500_vs_sigma_tau_separate(
        flat_samples,
        df_agn_pass2_plot_sample,
        cosmo_model=cosmo_model,
        z_pivot_agn=z_pivot_agn,
        debias=True,
        dm_interp=dm_interp,
        dmi_values=dmi_posterior_median_full,
        dmi_selection_sigma=dmi_selection_sigma_full,
        show_residuals=True,
        show=False,
        plot_path=plot_path,
        z_range=z_range,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
        agn_pivot_context=agn_pivot_context,
    )
    plot_catalog_quantity_vs_sigma_tau_separate(
        df_agn_pass2_plot_sample,
        y_col="LOGMBH",
        yerr_col="LOGMBH_ERR",
        y_label=r"$\log M_{\rm BH}$",
        filename="MBH_vs_sigma_tau_separate.pdf",
        plot_path=plot_path,
        show=False,
        z_range=z_range,
    )
    plot_catalog_quantity_vs_sigma_tau_separate(
        df_agn_pass2_plot_sample,
        y_col="LOGLEDD_RATIO",
        yerr_col="LOGLEDD_RATIO_ERR",
        y_label=r"$\log (L/L_{\rm Edd})$",
        filename="Eddington_ratio_vs_sigma_tau_separate.pdf",
        plot_path=plot_path,
        show=False,
        z_range=z_range,
    )

    plot_blr_line_lags_vs_l2500(
        flat_samples,
        df_agn_pass2_plot_sample,
        cosmo_model,
        z_pivot_agn,
        dm_interp=dm_interp,
        dmi_values=dmi_posterior_median_full,
        plot_path=plot_path,
        show=False,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    plot_blr_diagnostics_summary(
        df_agn_pass2_plot_sample,
        plot_path=plot_path,
        show=False,
    )

    chisq_red_L2500, _ = reduced_chi_squared(L_residuals_debiased, L_pred_std_debiased, n_params=len(model_labels)-1)
    df_agn_agn_likelihood_chi2_selection = df_agn_pass2_fit_selection
    if resume_replot_with_cuts:
        df_agn_agn_likelihood_chi2_selection = df_agn_pass2_plot_sample[
            df_agn_pass2_plot_sample["z"].between(z_range[0], z_range[1])
        ].copy()
    chisq_red_agn_likelihood_space, _ = compute_agn_likelihood_space_reduced_chi2(
        flat_samples,
        model_labels,
        df_agn_agn_likelihood_chi2_selection,
        cosmo_model,
        z_pivot_agn=z_pivot_agn,
        agn_pivot_context=agn_pivot_context,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    df_agn_agn_likelihood_chi2_selection_zgt1 = df_agn_agn_likelihood_chi2_selection[
        df_agn_agn_likelihood_chi2_selection["z"].between(1.0, z_range[1], inclusive="right")
    ]
    chisq_red_agn_likelihood_space_zgt1, _ = compute_agn_likelihood_space_reduced_chi2(
        flat_samples,
        model_labels,
        df_agn_agn_likelihood_chi2_selection_zgt1,
        cosmo_model,
        z_pivot_agn=z_pivot_agn,
        agn_pivot_context=agn_pivot_context,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )

    print("Plotting Hubble diagram...")
    if dmi_posterior_median_full is not None:
        plot_completeness_diagnostics(
            dmi_posterior_median_full,
            df_agn_pass2_plot_sample["z"].values,
            df_agn_pass2_plot_sample[COMPLETENESS_MAG_COL].values,
            integrals_max_w=None,
            plot_path=plot_path,
            z_range=z_range,
        )
    # Debiased (Bias corrected)
    r = plot_hubble(flat_samples, df_agn_pass2_plot_sample, df_pantheon,
                    cosmo_model=cosmo_model, z_pivot_agn=z_pivot_agn, 
                    show_true=False, show=False, debias=True, dm_interp=dm_interp, plot_path=plot_path,
                    cosmo_model_samples=cosmo_model_joint_samples, verbose=verbose, residuals_sigma_clip=residuals_sigma_clip,
                    df_calibrators=df_calibrators,
                    dmi_values=dmi_posterior_median_full,
                    dmi_sigma=dmi_posterior_sigma_full,
                    dmi_selection_sigma=dmi_selection_sigma_full,
                    residuals_csv_filename="hubble_plot_residuals.csv",
                    sigma_clip_threshold=sigma_clip_threshold if apply_two_pass_sigma_clip else None,
                    z_range=z_range,
                    use_alpha_lambda_term=use_alpha_lambda_term,
                    use_eta_sigma_term=use_eta_sigma_term,
                    use_redshift_log_f_term=use_redshift_log_f_term,
                    only_agn=only_agn,
                    agn_likelihood_space_chi2=chisq_red_agn_likelihood_space,
                    agn_likelihood_space_chi2_zgt1=chisq_red_agn_likelihood_space_zgt1,
                    agn_pivot_context=agn_pivot_context)
    debiased_residuals, debiased_clipping_sigma, mu_pred_median_debiased, mu_pred_std_debiased, mu_pred_std_debiased_with_scatter = r
    if apply_two_pass_sigma_clip:
        final_diagnostics_df, keep_mask_pass2 = _build_sigma_clip_diagnostics(
            df_agn_pass2_plot_sample,
            debiased_residuals,
            debiased_clipping_sigma,
            sigma_clip_threshold=sigma_clip_threshold,
        )
        final_diagnostics_df = final_diagnostics_df.rename(
            columns={
                "mu_zscore": "mu_zscore_pass2",
                "was_clipped": "was_clipped_pass2",
            }
        )
        if pass1_diagnostics_df is not None:
            final_diagnostics_df["mu_zscore_pass1"] = pass1_diagnostics_df.loc[
                final_diagnostics_df.index, "mu_zscore"
            ].to_numpy(dtype=float)
            final_diagnostics_df["was_clipped_pass1"] = pass1_diagnostics_df.loc[
                final_diagnostics_df.index, "was_clipped"
            ].to_numpy(dtype=bool)
        final_diagnostics_df["is_in_pass2_sample"] = True
        final_diagnostics_df["is_in_pass2_plot_sample"] = True
        final_diagnostics_df["is_in_pass2_fit_selection"] = final_diagnostics_df["z"].between(
            z_range[0], z_range[1]
        )
        _write_sigma_clip_diagnostics(
            final_diagnostics_df,
            plot_path,
            residuals_filename="residuals.csv",
        )
        if keep_mask_full is not None:
            if pass1_diagnostics_df is None:
                pass2_membership_audit = df_agn_full_sample_preclip.copy()
                pass2_membership_audit["is_in_pass2_sample"] = True
                pass2_membership_audit["is_in_pass2_plot_sample"] = True
                pass2_membership_audit["is_in_pass2_fit_selection"] = False
                pass2_membership_audit["was_clipped_pass1"] = False
            else:
                pass2_membership_audit = pass1_diagnostics_df.copy()
                pass2_membership_audit = pass2_membership_audit.rename(
                    columns={
                        "mu_zscore": "mu_zscore_pass1",
                        "was_clipped": "was_clipped_pass1",
                    }
                )
                pass2_membership_audit["is_in_pass2_sample"] = keep_mask_full
                pass2_membership_audit["is_in_pass2_plot_sample"] = keep_mask_full
                pass2_membership_audit["is_in_pass2_fit_selection"] = False
            fit_indices_pass2 = pd.Index(df_agn_pass2_fit_selection.index)
            pass2_membership_audit.loc[fit_indices_pass2, "is_in_pass2_fit_selection"] = True
            pass2_membership_audit["mu_zscore_pass2"] = np.nan
            pass2_membership_audit["was_clipped_pass2"] = pd.Series(
                pd.array([pd.NA] * len(pass2_membership_audit), dtype="boolean"),
                index=pass2_membership_audit.index,
            )
            pass2_membership_audit.loc[final_diagnostics_df.index, "mu_zscore_pass2"] = (
                final_diagnostics_df["mu_zscore_pass2"].to_numpy(dtype=float)
            )
            pass2_membership_audit.loc[final_diagnostics_df.index, "was_clipped_pass2"] = (
                final_diagnostics_df["was_clipped_pass2"].to_numpy(dtype=bool)
            )
            pass2_membership_audit.to_csv(
                Path(plot_path) / "sigma_clip_membership_pass2.csv",
                index=False,
            )
    if cosmo_model == "Flatw0waCDM":
        df_agn_table_sample = df_agn_full_sample_preclip
        table_sample_matches_plot_sample = (
            len(df_agn_table_sample) == len(df_agn_pass2_plot_sample)
            and df_agn_table_sample["object_id"].astype(str).reset_index(drop=True).equals(
                df_agn_pass2_plot_sample["object_id"].astype(str).reset_index(drop=True)
            )
        )
        if table_sample_matches_plot_sample:
            dmi_posterior_median_table = dmi_posterior_median_full
        else:
            dmi_posterior_median_table, _, _ = _compute_direct_full_sample_completeness_summaries(
                flat_samples,
                df_agn_fit_selection=df_agn_pass2_fit_selection,
                df_agn_plot_sample=df_agn_table_sample,
                df_pantheon=df_pantheon,
                _sna_L=_sna_L,
                _sna_Lower=_sna_Lower,
                _sna_LogdetCov=_sna_LogdetCov,
                cosmo_model=cosmo_model,
                completeness_params=_get_direct_completeness_params(),
                z_pivot_agn=z_pivot_agn,
                agn_pivot_context=agn_pivot_context,
                use_full_cov=use_full_cov,
                disable_ceph_dist_calibration=disable_ceph_dist_calibration,
                use_planck_h0_prior=use_planck_h0_prior,
                use_planck_om_prior=use_planck_om_prior,
                only_agn=only_agn,
                use_alpha_lambda_term=use_alpha_lambda_term,
                use_eta_sigma_term=use_eta_sigma_term,
                use_redshift_log_f_term=use_redshift_log_f_term,
                early_de_guard=early_de_guard,
            )
        mu_table, mu_err_table = _compute_debiased_agn_table_mu(
            flat_samples,
            model_labels,
            df_agn_table_sample,
            cosmo_model,
            z_pivot_agn=z_pivot_agn,
            agn_pivot_context=agn_pivot_context,
            dmi_values=dmi_posterior_median_table,
            only_agn=only_agn,
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
            use_redshift_log_f_term=use_redshift_log_f_term,
        )
        make_agn_csv_table(
            df_agn_table_sample,
            mu_table,
            mu_err_table,
            dm_interp,
            dmi_values=dmi_posterior_median_table,
            sort_by="z",
            ascending=True,
            write_path=plot_path,
        )
        make_agn_latex_table(
            df_agn_pass2_plot_sample,
            mu_pred_median_debiased,
            mu_pred_std_debiased_with_scatter,
            dm_interp,
            dmi_values=dmi_posterior_median_full,
            sort_by="z",
            ascending=True,
            max_rows=30,
            write_path=plot_path,
        )
    # Biased
    r = plot_hubble(flat_samples, df_agn_pass2_plot_sample, df_pantheon,
                cosmo_model=cosmo_model, z_pivot_agn=z_pivot_agn, show_residuals=True,
                show_true=False, show=False, debias=False, plot_path=plot_path, verbose=False,
                sigma_clip_threshold=sigma_clip_threshold if apply_two_pass_sigma_clip else None,
                z_range=z_range,
                use_alpha_lambda_term=use_alpha_lambda_term,
                use_eta_sigma_term=use_eta_sigma_term,
                use_redshift_log_f_term=use_redshift_log_f_term,
                only_agn=only_agn,
                agn_pivot_context=agn_pivot_context)
    biased_residuals, biased_residuals_err, _, _, _ = r

    hubble_chi2_mask = df_agn_pass2_plot_sample["z"].between(z_range[0], z_range[1]).to_numpy(dtype=bool)
    if np.any(hubble_chi2_mask):
        chisq_red_hubble_debiased, _ = reduced_chi_squared(
            debiased_residuals[hubble_chi2_mask],
            mu_pred_std_debiased[hubble_chi2_mask],
            n_params=len(model_labels)-1,
        )
    else:
        chisq_red_hubble_debiased = np.nan
    chisq_red_hubble_debiased_no_mpred_err = np.nan
    hdbudget_path = Path(plot_path) / "diagnostics" / "hubble_error_budget_per_object_debiased.csv"
    if np.any(hubble_chi2_mask) and hdbudget_path.exists():
        hdbudget_df = pd.read_csv(hdbudget_path)
        required_no_mpred_cols = {
            "z",
            "residuals",
            "apparent_mag_2500_err_term",
            "sigma_lens_term",
            "z_err_term",
            "intrinsic_scatter_term",
        }
        if required_no_mpred_cols.issubset(hdbudget_df.columns):
            budget_mask = hdbudget_df["z"].between(z_range[0], z_range[1]).to_numpy(dtype=bool)
            sigma_no_mpred = np.sqrt(
                np.square(hdbudget_df["apparent_mag_2500_err_term"].to_numpy(dtype=float))
                + np.square(hdbudget_df["sigma_lens_term"].to_numpy(dtype=float))
                + np.square(hdbudget_df["z_err_term"].to_numpy(dtype=float))
                + np.square(hdbudget_df["intrinsic_scatter_term"].to_numpy(dtype=float))
            )
            if np.any(budget_mask):
                chisq_red_hubble_debiased_no_mpred_err, _ = reduced_chi_squared(
                    hdbudget_df["residuals"].to_numpy(dtype=float)[budget_mask],
                    sigma_no_mpred[budget_mask],
                    n_params=len(model_labels)-1,
                )
        else:
            missing_no_mpred_cols = sorted(required_no_mpred_cols.difference(hdbudget_df.columns))
            print(
                "[WARNING] Skipping 'Hubble no Mpred err' reduced chi-squared "
                f"because diagnostics CSV is missing required columns: {missing_no_mpred_cols}"
            )
    plot_hubble_residual_normality(
        debiased_residuals[hubble_chi2_mask],
        mu_pred_std_debiased_with_scatter[hubble_chi2_mask],
        plot_path=plot_path,
        show=False,
        filename="hubble_residual_normality_debiased.pdf",
    )
    plot_hubble(
        flat_samples,
        df_agn_pass2_plot_sample,
        df_pantheon,
        cosmo_model=cosmo_model,
        z_pivot_agn=z_pivot_agn,
        show_true=False,
        show=False,
        debias=True,
        dm_interp=dm_interp,
        plot_path=plot_path,
        cosmo_model_samples=cosmo_model_joint_samples,
        verbose=False,
        residuals_sigma_clip=residuals_sigma_clip,
        df_calibrators=df_calibrators,
        dmi_values=dmi_posterior_median_full,
        dmi_sigma=dmi_posterior_sigma_full,
        dmi_selection_sigma=dmi_selection_sigma_full,
        filename="hubble_diagram_debiased_no_logf.pdf",
        residuals_csv_filename="hubble_plot_residuals_no_logf.csv",
        sigma_clip_threshold=sigma_clip_threshold if apply_two_pass_sigma_clip else None,
        z_range=z_range,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
        only_agn=only_agn,
        use_intrinsic_scatter_in_residual_sigma=False,
        agn_likelihood_space_chi2=chisq_red_agn_likelihood_space,
        agn_likelihood_space_chi2_zgt1=chisq_red_agn_likelihood_space_zgt1,
        diagnostics_suffix="_debiased_no_logf",
        agn_pivot_context=agn_pivot_context,
    )
    plot_hubble_residual_normality(
        debiased_residuals[hubble_chi2_mask],
        mu_pred_std_debiased[hubble_chi2_mask],
        plot_path=plot_path,
        show=False,
        filename="hubble_residual_normality_debiased_no_logf.pdf",
    )
    plot_hubble_residual_tail_diagnostics(
        df_agn_pass2_plot_sample.loc[hubble_chi2_mask].copy(),
        debiased_residuals[hubble_chi2_mask],
        mu_pred_std_debiased_with_scatter[hubble_chi2_mask],
        sigma_dmi=dmi_posterior_sigma_full[hubble_chi2_mask] if dmi_posterior_sigma_full is not None else None,
        sigma_sel=dmi_selection_sigma_full[hubble_chi2_mask] if dmi_selection_sigma_full is not None else None,
        plot_path=plot_path,
        show=False,
    )



    print("Plotting predicted vs actual M2500...")
    plot_predicted_vs_actual_M2500(
        flat_samples,
        df_agn_pass2_plot_sample,
        cosmo_model=cosmo_model,
        z_pivot_agn=z_pivot_agn,
        debias=False,
        show=False,
        plot_path=plot_path,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
        agn_pivot_context=agn_pivot_context,
    )
    M2500_residuals_debiased, M2500_std_debiased, M2500_binned_residuals_debiased, _ = plot_predicted_vs_actual_M2500(
        flat_samples,
        df_agn_pass2_plot_sample,
        cosmo_model=cosmo_model,
        z_pivot_agn=z_pivot_agn,
        debias=True,
        show=False,
        dm_interp=dm_interp,
        dmi_values=dmi_posterior_median_full,
        dmi_selection_sigma=dmi_selection_sigma_full,
        plot_path=plot_path,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
        agn_pivot_context=agn_pivot_context,
    )
    chisq_red_M2500_debiased, _ = reduced_chi_squared(M2500_residuals_debiased, M2500_std_debiased, n_params=len(model_labels)-1)
    print("Plotting debiased residuals...")
    plot_full_residuals(
        df_agn_pass2_plot_sample,
        debiased_residuals,
        debiased_clipping_sigma,
        flat_samples,
        cosmo_model,
        z_pivot_agn,
        debias=True,
        dm_interp=dm_interp,
        dmi_values=dmi_posterior_median_full,
        show=False,
        plot_path=plot_path,
        z_range=z_range,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    plot_full_residuals(
        df_agn_pass2_plot_sample,
        debiased_residuals,
        debiased_clipping_sigma,
        flat_samples,
        cosmo_model,
        z_pivot_agn,
        debias=True,
        dm_interp=dm_interp,
        dmi_values=dmi_posterior_median_full,
        show=False,
        plot_path=plot_path,
        z_cut=1.5,
        z_range=z_range,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    plot_full_residuals(
        df_agn_pass2_plot_sample,
        L_residuals_debiased,
        L_pred_std_debiased,
        flat_samples,
        cosmo_model,
        z_pivot_agn,
        debias=True,
        dm_interp=dm_interp,
        dmi_values=dmi_posterior_median_full,
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
        df_agn_pass2_plot_sample,
        debiased_residuals,
        debiased_clipping_sigma,
        flat_samples,
        cosmo_model,
        z_pivot_agn,
        debias=True,
        dm_interp=dm_interp,
        dmi_values=dmi_posterior_median_full,
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
        df_agn_pass2_plot_sample,
        debiased_residuals,
        debiased_clipping_sigma,
        flat_samples,
        cosmo_model,
        z_pivot_agn,
        debias=True,
        dm_interp=dm_interp,
        dmi_values=dmi_posterior_median_full,
        show=False,
        plot_path=plot_path,
        z_range=z_range,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    plot_debias_impact_diagnostics(
        df_agn_pass2_plot_sample,
        biased_residuals,
        debiased_residuals,
        plot_path=plot_path,
        show=False,
    )
    plot_redshift_bin_residual_summary(
        df_agn_pass2_plot_sample,
        biased_residuals,
        biased_residuals_err,
        debiased_residuals,
        debiased_clipping_sigma,
        plot_path=plot_path,
        show=False,
    )
    plot_redshift_wiggle_diagnostics(
        df_agn_pass2_plot_sample,
        biased_residuals,
        biased_residuals_err,
        debiased_residuals,
        debiased_clipping_sigma,
        plot_path=plot_path,
        z_range=z_range,
        show=False,
    )
    plot_parameter_residual_diagnostics(
        df_agn_pass2_plot_sample,
        debiased_residuals,
        debiased_clipping_sigma,
        plot_path=plot_path,
        z_range=z_range,
        show=False,
    )
    plot_fast_vs_uv_variability(df_agn_pass2_plot_sample, plot_path=plot_path, show=False)

    
    print("Plotting cosmological posteriors corner plot...")
    plot_cosmo_corner(None, flat_samples, cosmo_model, z_pivot_sna, z_pivot_agn, show=False, 
                      plot_path=plot_path, speed=speed,
                      gauss_sigma=1.5, kde_bw_scale=1.5,
                      include_alpha_beta=True,
                      only_agn=only_agn,
                      use_alpha_lambda_term=use_alpha_lambda_term,
                      use_eta_sigma_term=use_eta_sigma_term,
                      use_redshift_log_f_term=use_redshift_log_f_term)

    if completeness:
        df_agn_completeness_plot_sample = df_agn_full_sample_preclip
        if completeness_mode == "4d_fhost_alpha":
            print("Plotting host-aware/color-aware 4D completeness diagnostics...")
            get_completeness_function_4d_fhost_alpha(
                df_agn_completeness_plot_sample,
                sim_file=completeness_sim_file,
                plot=True,
                plot_path=plot_path,
                df_agn_fhost_population=df_agn_all,
            )
        elif completeness_mode == "3d_fhost":
            print("Plotting host-aware 3D completeness diagnostics...")
            get_completeness_function_3d_fhost(
                df_agn_completeness_plot_sample,
                sim_file=completeness_sim_file,
                plot=True,
                plot_path=plot_path,
                df_agn_fhost_population=df_agn_all,
            )
        else:
            print("Plotting completeness vs magnitude at redshifts...")
            p_detect, mag_centers, z_centers, dm, dz, completeness_scatter = get_completeness_function_2d(
                df_agn_completeness_plot_sample, sim_file=completeness_sim_file, plot=True, plot_path=plot_path
            )
            plot_completeness_vs_mag_at_redshifts(
                p_detect, mag_centers, z_centers, plot_path=plot_path
            )

    print(f"\033[94mReduced chi-squared (debiased) M2500: {chisq_red_M2500_debiased:.3f}\033[0m")
    print(f"\033[94mReduced chi-squared (debiased) Hubble: {chisq_red_hubble_debiased:.3f}\033[0m")
    print(f"\033[94mReduced chi-squared (AGN likelihood-space residuals): {chisq_red_agn_likelihood_space:.3f}\033[0m")
    print(f"\033[94mReduced chi-squared (debiased) Hubble no Mpred err: {chisq_red_hubble_debiased_no_mpred_err:.3f}\033[0m")
    print(f"\033[94mReduced chi-squared (debiased) L2500: {chisq_red_L2500:.3f}\033[0m")
    chisq_dict = {
        'M2500': chisq_red_M2500_debiased,
        'Hubble': chisq_red_hubble_debiased,
        'AGN_likelihood_space': chisq_red_agn_likelihood_space,
        'L2500': chisq_red_L2500
    }

    df_agn_alpha_ox = _compute_alpha_ox_from_posterior_median(
        df_agn_pass2_plot_sample,
        flat_samples,
        model_labels,
        cosmo_model=cosmo_model,
        z_pivot=z_pivot_agn,
    )
    plot_residuals_vs_alphaOX(
        df_agn_alpha_ox,
        debiased_residuals,
        debiased_clipping_sigma,
        show=False,
        plot_path=plot_path,
    )

    return flat_samples, model_labels, dm_interp, logZ, logZerr, debiased_residuals, age, age_err


def run_all(df_agn, df_agn_all, df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov, 
            cosmo_models, skip_plots=False,
            minimal_plots=False,
            residuals_sigma_clip=None,
            disable_sigma_clip_pass=False,
            sigma_clip_threshold=3.0,
            z_range=(0.44, 3.16),
            speed="production", resume=False, N=None,
            resume_stage="both",
            sigma_clip_second_pass_mode="warm",
            completeness=True,
            prefix="default", result_prefix="", uniform_redshift_distribution=False,
            completeness_sim_file=DEFAULT_COMPLETENESS_SIM_FILE,
            completeness_mode="2d",
            completeness_magnitude="dereddened",
            compare_sigma_only=False,
            disable_ceph_dist_calibration=False,
            use_planck_h0_prior=False,
            use_planck_om_prior=False,
            only_agn=False,
            use_alpha_lambda_term=False,
            use_eta_sigma_term=False,
            use_redshift_log_f_term=False,
            early_de_guard=False):

    validate_completeness_mode(completeness_mode)
    completeness_magnitude = normalize_completeness_magnitude(
        completeness_magnitude
    )
    speed = normalize_speed(speed)
    if only_agn:
        print("Running full model comparison in AGN-only mode; SNe-only comparison branch is disabled.")
    use_planck_h0_prior = use_planck_h0_prior or disable_ceph_dist_calibration
    sigma_clip_second_pass_mode = normalize_sigma_clip_second_pass_mode(sigma_clip_second_pass_mode)
    zmin, zmax = z_range
    n_tag = "all" if N is None else f"N{N}"
    z_tag = f"z{zmin:.2f}_{zmax:.2f}".replace(".", "p")
    completeness_tag = (
        f"_{completeness_mode}_compmag-{completeness_magnitude}"
        if completeness
        else "_disable_completeness"
    )
    ceph_tag = "_nocephdist_planckh0" if disable_ceph_dist_calibration else ""
    planck_h0_tag = "_planckh0" if use_planck_h0_prior and not disable_ceph_dist_calibration else ""
    planck_om_tag = "_planckom" if use_planck_om_prior else ""
    mode_tag = _fit_mode_label(False, only_agn)
    compare_run_tag = f"model_compare_{mode_tag}_{speed}_{n_tag}_{z_tag}{completeness_tag}{ceph_tag}{planck_h0_tag}{planck_om_tag}"
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
    resume_by_model = normalize_resume_by_model(resume, cosmo_models)
    agn_pivot_context = _prepare_shared_agn_pivot_context(
        df_agn,
        cosmo_models=cosmo_models,
        resume_by_model=resume_by_model,
        z_range=z_range,
        N=N,
        uniform_redshift_distribution=uniform_redshift_distribution,
        only_sna=False,
        only_agn=only_agn,
        speed=speed,
        completeness=completeness,
        completeness_mode=completeness_mode,
        completeness_magnitude=completeness_magnitude,
        disable_ceph_dist_calibration=disable_ceph_dist_calibration,
        use_planck_h0_prior=use_planck_h0_prior,
        use_planck_om_prior=use_planck_om_prior,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
        disable_sigma_clip_pass=disable_sigma_clip_pass,
        resume_stage=resume_stage,
        prefix=prefix,
    )
    for cosmo_model in cosmo_models:
        model_resume = resume_by_model[cosmo_model]
        r = run_single(df_agn, df_agn_all, df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov, 
                       cosmo_model=cosmo_model, only_sna=False, 
                       only_agn=only_agn,
                       completeness=completeness,
                       resume=model_resume, speed=speed, N=N,
                       skip_plots=skip_plots,
                       residuals_sigma_clip=residuals_sigma_clip,
                       disable_sigma_clip_pass=disable_sigma_clip_pass,
                       sigma_clip_threshold=sigma_clip_threshold,
                       resume_stage=resume_stage,
                       sigma_clip_second_pass_mode=sigma_clip_second_pass_mode,
                       z_range=z_range,
                       cosmo_model_joint_samples=cosmo_model_joint_samples,
                       prefix=prefix, uniform_redshift_distribution=uniform_redshift_distribution,
                       completeness_sim_file=completeness_sim_file,
                       completeness_mode=completeness_mode,
                       completeness_magnitude=completeness_magnitude,
                       compare_sigma_only=compare_sigma_only,
                       minimal_plots=minimal_plots,
                       disable_ceph_dist_calibration=disable_ceph_dist_calibration,
                       use_planck_h0_prior=use_planck_h0_prior,
                       use_planck_om_prior=use_planck_om_prior,
                       use_alpha_lambda_term=use_alpha_lambda_term,
                       use_eta_sigma_term=use_eta_sigma_term,
                       use_redshift_log_f_term=use_redshift_log_f_term,
                       early_de_guard=early_de_guard,
                       agn_pivot_context=agn_pivot_context)
        
        samples_joint, model_labels_joint, dm_interp_joint, logZ_joint, logZerr_joint, debiased_residuals_joint, age_joint, age_err_joint = r
        #print(f"For model {cosmo_model}, universe age: {age:.3f} Gyr")
        if only_agn:
            samples_sna = None
            model_labels_sna = []
            logZ_sna = np.nan
            logZerr_sna = np.nan
            age_sna = np.nan
            age_sna_err = np.nan
        else:
            r = run_single(df_agn, df_agn_all, df_pantheon, _sna_L, _sna_Lower, _sna_LogdetCov, 
                           cosmo_model=cosmo_model, only_sna=True, 
                           completeness=completeness,
                           skip_plots=skip_plots,
                           residuals_sigma_clip=residuals_sigma_clip,
                           disable_sigma_clip_pass=disable_sigma_clip_pass,
                           sigma_clip_threshold=sigma_clip_threshold,
                           resume_stage=resume_stage,
                           sigma_clip_second_pass_mode=sigma_clip_second_pass_mode,
                           z_range=z_range,
                           resume=model_resume, speed=speed, N=N,
                           prefix=prefix, uniform_redshift_distribution=uniform_redshift_distribution,
                           completeness_sim_file=completeness_sim_file,
                           completeness_mode=completeness_mode,
                           completeness_magnitude=completeness_magnitude,
                           compare_sigma_only=compare_sigma_only,
                           minimal_plots=minimal_plots,
                           disable_ceph_dist_calibration=disable_ceph_dist_calibration,
                           use_planck_h0_prior=use_planck_h0_prior,
                           use_planck_om_prior=use_planck_om_prior,
                           use_alpha_lambda_term=use_alpha_lambda_term,
                           use_eta_sigma_term=use_eta_sigma_term,
                           use_redshift_log_f_term=use_redshift_log_f_term,
                           early_de_guard=early_de_guard,
                           agn_pivot_context=None)
            samples_sna, model_labels_sna, dm_interp_sna, logZ_sna, logZerr_sna, debiased_residuals_sna, age_sna, age_sna_err = r
        if not compare_sigma_only and not minimal_plots and not only_agn:
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



        r_sna = None
        if not only_agn:
            r_sna = extract_cosmo_results_from_samples(
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
            only_agn=only_agn,
            logZ_tuple=(logZ_joint, logZerr_joint),
            format_for_latex=True,
            value_fmt="{:.2f}",
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
            use_redshift_log_f_term=use_redshift_log_f_term,
        )

        cosmo_model_joint_samples[cosmo_model] = samples_joint
        if not only_agn:
            cosmo_model_sna_samples[cosmo_model] = samples_sna
            results_latex.append(r_sna)
        results_latex.append(r_joint)

        cosmo_models_result_dict[cosmo_model] |= dict(N=N, z_i=z_range[0], z_f=z_range[1])

        for i, key in enumerate(model_labels_joint):
            median, err, lower, upper = sym_percentile(samples_joint[:, i])
            cosmo_models_result_dict[cosmo_model][key] = median
            cosmo_models_result_dict[cosmo_model][f"{key}_err"] = err
            cosmo_models_result_dict[cosmo_model][f"{key}_err_lower"] = lower
            cosmo_models_result_dict[cosmo_model][f"{key}_err_upper"] = upper

        if not only_agn:
            for i, key in enumerate(model_labels_sna):
                median, err, lower, upper = sym_percentile(samples_sna[:, i])
                cosmo_models_sna_result_dict[cosmo_model][key] = median
                cosmo_models_sna_result_dict[cosmo_model][f"{key}_err"] = err
                cosmo_models_sna_result_dict[cosmo_model][f"{key}_err_lower"] = lower
                cosmo_models_sna_result_dict[cosmo_model][f"{key}_err_upper"] = upper

    compare_r = compare_models_by_log_evidence_all(df_agn, cosmo_models_result_dict, write_path=f"{compare_plot_path}/")
    compare_r_sna = None
    if not only_agn:
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
                                compare_r_sna=compare_r_sna,
                                agn_pivot_context=agn_pivot_context)

    cosmo_output_dir = get_qvc_result_dir() / "cosmo" / prefix
    cosmo_output_dir.mkdir(parents=True, exist_ok=True)
    save_cosmo_results_hdf5(
        str(cosmo_output_dir / f"cosmo_results_{n_tag}_{z_tag}.hdf5"),
        cosmo_models_result_dict
    )
    
    print("================================================================\n\n")
    return cosmo_models_result_dict, cosmo_model_joint_samples, results_latex, compare_r


def validate_plot_mode_args(args):
    """Reject plotting modes whose output contracts are mutually exclusive."""
    if args.minimal_plots and args.skip_plots:
        raise ValueError("--minimal-plots cannot be combined with --skip_plots.")
    if args.minimal_plots and args.compare_sigma_only:
        raise ValueError("--minimal-plots cannot be combined with --compare_sigma_only.")
    if args.minimal_plots and args.only_sna:
        raise ValueError("--minimal-plots cannot be used with a direct --only_sna run.")
    if args.minimal_plots and args.use_jax:
        raise ValueError("--minimal-plots is not supported with --use_jax.")


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
    parser.add_argument(
        "--use_planck_h0_prior",
        action="store_true",
        default=False,
        help="Use the Planck 2018 top-hat H0 prior without changing Cepheid distance calibration.",
    )
    parser.add_argument(
        "--use_planck_om_prior",
        action="store_true",
        default=False,
        help="Use the Planck 2018 top-hat Om0 prior.",
    )
    parser.add_argument(
        "--early-de-guard",
        action="store_true",
        default=False,
        help="Reject Flatw0waCDM samples with w0 + wa >= 0. Disabled by default.",
    )
    parser.add_argument(
        "--resume",
        nargs="*",
        default=False,
        help=(
            "Resume previous MCMC run(s). With no paths, each model uses its default checkpoint. "
            "With explicit paths, pass one H5 path per --cosmo_models entry."
        ),
    )
    parser.add_argument(
        "--resume_stage",
        type=str,
        choices=["both", "pass1", "pass2"],
        default="both",
        help="For two-pass sigma clipping, choose which stage to resume: overall workflow ('both'), only the first pass ('pass1'), or only the second pass using embedded pass-1 state ('pass2'). Ignored when two-pass sigma clipping is disabled.",
    )
    parser.add_argument("--run", type=str, choices=["full", "single"], default="single", help="Run mode: compare_models, compare_sna, full, or single (default: single)")
    parser.add_argument(
        "--speed",
        type=str,
        choices=SPEED_CHOICES,
        default="production",
        help=(
            "Sampling speed preset. Preferred names, fastest to slowest: "
            "fastest, quick, standard, production."
        ),
    )
    parser.add_argument("--N", type=int, default=None, help="Number of AGNs to run (default: all)")
    parser.add_argument("--only_sna", action="store_true", default=False, help="Run SNIa-only fit (default: False)")
    parser.add_argument("--only_agn", action="store_true", default=False, help="Run AGN-only fit with the Supernova likelihood and M0_sn disabled (default: False)")
    parser.add_argument(
        "--spectra_fit_csv",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Path(s) to CSV output from fit_spectra_jaxsedfit_joint.py. "
            "Legacy spectral-fit formats are not supported."
        ),
    )
    parser.add_argument(
        "--magnitude-convention",
        type=str,
        choices=["dereddened", "attenuated"],
        required=True,
        help=(
            "Choose whether the Hubble workflow uses the joint SED fit's "
            "dereddened or attenuated-model 2500-A magnitude."
        ),
    )
    parser.add_argument(
        "--spectra_sdss_run2d",
        type=str,
        choices=["all", "v5_13_2", "26"],
        default="all",
        help="Optional SDSS_RUN2D filter for spectra-matched AGN rows. Applies only when cuts are enabled.",
    )
    parser.add_argument(
        "--no-cuts",
        "--no_cuts",
        dest="no_cuts",
        action="store_true",
        default=False,
        help="Disable all AGN data cuts (default: False).",
    )
    parser.add_argument("--skip_plots", action="store_true", default=False, help="Skip plotting steps (default: False)")
    parser.add_argument(
        "--minimal-plots",
        action="store_true",
        default=False,
        help=(
            "Run the normal fit and evidence comparison while retaining only the "
            "debiased Hubble diagram and its residual CSV."
        ),
    )
    parser.add_argument(
        "--compare_sigma_only",
        action="store_true",
        default=False,
        help="Run the full fit and evidence calculation, but skip non-essential plots and keep only text/console model-comparison sigma outputs.",
    )
    parser.add_argument("--exclude_object_ids_csv", type=str, nargs='+', default=[], help="Path(s) to CSV file(s) containing object IDs to exclude")
    parser.add_argument("--residuals_sigma_clip", type=float, default=None, help="Optional residual cut value to exclude outliers (default: None)")
    parser.add_argument("--residuals_csv", type=str, default=None, help="Path to CSV file containing residuals for outlier exclusion (default: None)")
    parser.add_argument(
        "--disable_sigma_clip_pass",
        action="store_true",
        default=False,
        help="Disable the internal second-pass AGN sigma-clipping rerun. By default the sigma-clip pass pathway runs.",
    )
    parser.add_argument(
        "--sigma_clip_threshold",
        type=float,
        default=3.0,
        help="Absolute mu_zscore clipping threshold for internal two-pass Hubble fitting (default: 3.0).",
    )
    parser.add_argument(
        "--sigma_clip_second_pass_mode",
        type=str,
        choices=SIGMA_CLIP_SECOND_PASS_MODES,
        default="warm",
        help=(
            "Second-stage behavior after sigma clipping: 'warm' seeds a small approximate "
            "top-up run from the pass-1 posterior; 'fresh' reruns the clipped fit from the prior."
        ),
    )
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
        help="Completeness model to use: 2D p(det|m,z), 3D p(det|m,z,f_host_2500_psf), or 4D p(det|m,z,f_host_2500_psf,alpha_lambda).",
    )
    parser.add_argument(
        "--completeness_magnitude",
        type=str,
        choices=list(VALID_COMPLETENESS_MAGNITUDES),
        default="dereddened",
        help=(
            "m_2500 definition used by the completeness model: "
            "'dereddened' (default) or 'attenuated'."
        ),
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
    parser.add_argument(
        "--resume_replot_with_cuts",
        action="store_true",
        default=False,
        help=(
            "Load posterior samples from --resume, remap saved per-AGN debias arrays by object_id "
            "to the current cut AGN sample, and regenerate plots without rerunning sampling."
        ),
    )

    args = parser.parse_args()
    args.speed = normalize_speed(args.speed)

    print("Running Hubble fit with the following settings:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    resume_by_model = normalize_resume_by_model(args.resume, args.cosmo_models)
    if args.only_sna and args.only_agn:
        raise ValueError("--only_sna and --only_agn cannot be used together.")
    validate_plot_mode_args(args)

    if args.disable_full_covariance:
        print("Warning: Running without full covariance may lead to underestimated uncertainties.")
    if args.disable_completeness:
        print("Warning: Running without completeness correction may lead to biased results.")
    if args.disable_ceph_dist_calibration:
        print("Warning: Running without CEPH_DIST calibration; using the Planck H0 prior instead.")
    effective_use_planck_h0_prior = args.use_planck_h0_prior or args.disable_ceph_dist_calibration
    if args.use_planck_h0_prior and not args.disable_ceph_dist_calibration:
        print("Using the Planck H0 prior with Cepheid distance calibration enabled.")
    if args.use_planck_om_prior:
        print("Using the Planck Om0 prior.")
    has_resume = any(bool(value) for value in resume_by_model.values())
    if has_resume:
        print("Warning: Resuming previous MCMC run.")
    if args.resume_replot_with_cuts:
        if not has_resume:
            raise ValueError("--resume_replot_with_cuts requires --resume path/to/posteriors.h5.")
        if args.run != "single":
            raise NotImplementedError("--resume_replot_with_cuts currently supports only --run single.")
        if args.use_jax:
            raise NotImplementedError("--resume_replot_with_cuts is not supported with --use_jax.")

    df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_pantheon_data()
    agn_plot_path = f"plots/hubble/{args.prefix}"
    cut_report_path = Path(agn_plot_path) / "cut_summary.txt"
    df_agn, df_agn_all = load_agn_data(args.agn_data_filepath, populate_sdss=args.force_populate_fields, 
                           apply_cut=not args.no_cuts,
                           residuals_sigma_clip=args.residuals_sigma_clip, residuals_csv=args.residuals_csv,
                           exclude_object_ids_csv=args.exclude_object_ids_csv,
                           spectra_fit_csv=args.spectra_fit_csv,
                           magnitude_convention=args.magnitude_convention,
                           spectra_sdss_run2d=args.spectra_sdss_run2d,
                           correct_sigma_uv_host=args.correct_sigma_uv_host,
                           z_range=tuple(args.z_range), plot_path=agn_plot_path,
                           cut_report_path=cut_report_path,
                           plot_diagnostics=not args.minimal_plots)
    effective_N = args.N
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
        if has_resume:
            raise NotImplementedError("--use_jax does not support --resume yet.")
        if args.agn_calibrators is not None:
            raise NotImplementedError("--use_jax does not support --agn_calibrators yet.")
        from qvc.hubble.hubble_fit_jax import run_single_jax

        agn_pivot_context = _prepare_shared_agn_pivot_context(
            df_agn,
            cosmo_models=args.cosmo_models,
            resume_by_model=resume_by_model,
            z_range=tuple(args.z_range),
            N=effective_N,
            uniform_redshift_distribution=args.uniform_redshift_distribution,
            only_sna=args.only_sna,
            only_agn=args.only_agn,
            speed=args.speed,
            completeness=not args.disable_completeness,
            completeness_mode=args.completeness_mode,
            completeness_magnitude=args.completeness_magnitude,
            disable_ceph_dist_calibration=args.disable_ceph_dist_calibration,
            use_planck_h0_prior=effective_use_planck_h0_prior,
            use_planck_om_prior=args.use_planck_om_prior,
            use_alpha_lambda_term=args.fit_alpha_lambda_term,
            use_eta_sigma_term=args.fit_eta_sigma_term,
            use_redshift_log_f_term=args.fit_redshift_log_f_term,
            disable_sigma_clip_pass=True,
            resume_stage="both",
            prefix=args.prefix,
        )
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
                completeness_magnitude=args.completeness_magnitude,
                only_sna=args.only_sna,
                only_agn=args.only_agn,
                N=effective_N,
                uniform_redshift_distribution=args.uniform_redshift_distribution,
                disable_ceph_dist_calibration=args.disable_ceph_dist_calibration,
                use_planck_h0_prior=effective_use_planck_h0_prior,
                use_planck_om_prior=args.use_planck_om_prior,
                use_alpha_lambda_term=args.fit_alpha_lambda_term,
                use_eta_sigma_term=args.fit_eta_sigma_term,
                use_redshift_log_f_term=args.fit_redshift_log_f_term,
                early_de_guard=args.early_de_guard,
                agn_pivot_context=agn_pivot_context,
            )
    elif args.run == "single": # default
        cosmo_models_dict = {k: {} for k in args.cosmo_models}
        agn_pivot_context = _prepare_shared_agn_pivot_context(
            df_agn,
            cosmo_models=args.cosmo_models,
            resume_by_model=resume_by_model,
            z_range=tuple(args.z_range),
            N=effective_N,
            uniform_redshift_distribution=args.uniform_redshift_distribution,
            only_sna=args.only_sna,
            only_agn=args.only_agn,
            speed=args.speed,
            completeness=not args.disable_completeness,
            completeness_mode=args.completeness_mode,
            completeness_magnitude=args.completeness_magnitude,
            disable_ceph_dist_calibration=args.disable_ceph_dist_calibration,
            use_planck_h0_prior=effective_use_planck_h0_prior,
            use_planck_om_prior=args.use_planck_om_prior,
            use_alpha_lambda_term=args.fit_alpha_lambda_term,
            use_eta_sigma_term=args.fit_eta_sigma_term,
            use_redshift_log_f_term=args.fit_redshift_log_f_term,
            disable_sigma_clip_pass=args.disable_sigma_clip_pass,
            resume_stage=args.resume_stage,
            prefix=args.prefix,
            resume_replot_with_cuts=args.resume_replot_with_cuts,
        )
        for cosmo_model in args.cosmo_models:
            r = run_single(df_agn=df_agn, df_agn_all=df_agn_all, df_pantheon=df_pantheon, _sna_L=_sna_L, _sna_Lower=_sna_Lower, _sna_LogdetCov=_sna_LogdetCov, 
                           cosmo_model=cosmo_model,
                completeness=not args.disable_completeness, use_full_cov=not args.disable_full_covariance, resume=resume_by_model[cosmo_model], z_range=args.z_range,
                speed=args.speed, N=effective_N, only_sna=args.only_sna, only_agn=args.only_agn,
                skip_plots=args.skip_plots, residuals_sigma_clip=args.residuals_sigma_clip,
                disable_sigma_clip_pass=args.disable_sigma_clip_pass,
                sigma_clip_threshold=args.sigma_clip_threshold,
                resume_stage=args.resume_stage,
                sigma_clip_second_pass_mode=args.sigma_clip_second_pass_mode,
                df_calibrators=df_calibrators,
                prefix=args.prefix,
                completeness_sim_file=args.completeness_sim_file,
                completeness_mode=args.completeness_mode,
                completeness_magnitude=args.completeness_magnitude,
                compare_sigma_only=args.compare_sigma_only,
                minimal_plots=args.minimal_plots,
                disable_ceph_dist_calibration=args.disable_ceph_dist_calibration,
                use_planck_h0_prior=effective_use_planck_h0_prior,
                use_planck_om_prior=args.use_planck_om_prior,
                use_alpha_lambda_term=args.fit_alpha_lambda_term,
                use_eta_sigma_term=args.fit_eta_sigma_term,
                use_redshift_log_f_term=args.fit_redshift_log_f_term,
                early_de_guard=args.early_de_guard,
                resume_replot_with_cuts=args.resume_replot_with_cuts,
                agn_pivot_context=agn_pivot_context)
            samples_joint, model_labels, dm_interp, logZ_joint, logZerr_joint, debiased_residuals, age, age_err = r
            cosmo_models_dict[cosmo_model]['logZ'] = logZ_joint
            cosmo_models_dict[cosmo_model]['logZerr'] = logZerr_joint
            cosmo_models_dict[cosmo_model]['age'] = age
            cosmo_models_dict[cosmo_model]['age_err'] = age_err
        zmin, zmax = args.z_range
        n_tag = "all" if effective_N is None else f"N{effective_N}"
        z_tag = f"z{zmin:.2f}_{zmax:.2f}".replace(".", "p")
        completeness_tag = (
            f"_{args.completeness_mode}_compmag-{args.completeness_magnitude}"
            if not args.disable_completeness
            else "_disable_completeness"
        )
        ceph_tag = "_nocephdist_planckh0" if args.disable_ceph_dist_calibration else ""
        planck_h0_tag = "_planckh0" if effective_use_planck_h0_prior and not args.disable_ceph_dist_calibration else ""
        planck_om_tag = "_planckom" if args.use_planck_om_prior else ""
        alpha_tag = "_alphaLam" if args.fit_alpha_lambda_term else ""
        eta_sigma_tag = "_etaSigma" if args.fit_eta_sigma_term else ""
        logf_tag = "_logfz" if args.fit_redshift_log_f_term else ""
        mode_tag = _fit_mode_label(args.only_sna, args.only_agn)
        compare_path = (
            f"plots/hubble/{args.prefix}/single_compare_{mode_tag}_{args.speed}_{n_tag}_{z_tag}"
            f"{completeness_tag}{ceph_tag}{planck_h0_tag}{planck_om_tag}{alpha_tag}{eta_sigma_tag}{logf_tag}"
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
                disable_sigma_clip_pass=args.disable_sigma_clip_pass,
                sigma_clip_threshold=args.sigma_clip_threshold,
                z_range=args.z_range,
                speed=args.speed, resume=args.resume, N=effective_N,
                resume_stage=args.resume_stage,
                sigma_clip_second_pass_mode=args.sigma_clip_second_pass_mode,
                completeness=not args.disable_completeness,
                prefix=args.prefix, result_prefix=args.result_prefix, uniform_redshift_distribution=args.uniform_redshift_distribution,
                completeness_sim_file=args.completeness_sim_file,
                completeness_mode=args.completeness_mode,
                completeness_magnitude=args.completeness_magnitude,
                compare_sigma_only=args.compare_sigma_only,
                minimal_plots=args.minimal_plots,
                disable_ceph_dist_calibration=args.disable_ceph_dist_calibration,
                use_planck_h0_prior=effective_use_planck_h0_prior,
                use_planck_om_prior=args.use_planck_om_prior,
                only_agn=args.only_agn,
                use_alpha_lambda_term=args.fit_alpha_lambda_term,
                use_eta_sigma_term=args.fit_eta_sigma_term,
                use_redshift_log_f_term=args.fit_redshift_log_f_term,
                early_de_guard=args.early_de_guard)
    
    print(f"Finished running Hubble fit pipeline for {args.cosmo_models}")
