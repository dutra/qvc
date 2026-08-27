"""
Experimental JAX/NumPyro nested-sampling Hubble fit pipeline.

This mirrors the main ``hubble_fit.py`` pipeline as closely as practical while:

- evaluating the AGN+SN likelihood in JAX
- using a JAX-native completeness interpolation path
- sampling with NumPyro's nested sampler when the optional dependencies are installed

``jax_cosmo`` is treated as optional here. If it imports cleanly it can be used
for future cosmology backends, but the current implementation uses an internal
JAX distance integral so the fitter remains runnable even when the environment's
``jax_cosmo`` install is incomplete.

It intentionally reuses the existing QVC data loading, cuts, completeness map
construction, and plotting utilities so that the inputs/outputs remain aligned
with the NumPy/Dynesty pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import config as jax_config
    from jax.scipy.linalg import solve_triangular
except Exception:  # pragma: no cover - optional dependency
    jax = None
    jnp = None
    jax_config = None
    solve_triangular = None

try:  # pragma: no cover - optional dependency
    import jax_cosmo as jc
except Exception:
    jc = None

try:  # pragma: no cover - optional dependency
    import numpyro
    import numpyro.distributions as dist
    from numpyro.contrib.nested_sampling import NestedSampler
except Exception:
    numpyro = None
    dist = None
    NestedSampler = None

from qvc.hubble.hubble_completeness_refactored import (
    COMPLETENESS_MAG_COL,
    COMPLETENESS_MAG_ERR_COL,
    VALID_COMPLETENESS_MAGNITUDES,
    Completeness2D,
    get_completeness_function_2d,
    make_dm_function,
    normalize_completeness_magnitude,
    prepare_completeness_magnitude_columns,
)
from qvc.hubble.hubble_fit import (
    COMPLETENESS_MOCK_MAX_ROWS_ENV,
    COMPLETENESS_MOCK_OVERSAMPLE_ENV,
    COMPLETENESS_MOCK_PROPOSAL_AREA_ENV,
    DEFAULT_COMPLETENESS_MOCK_MAX_ROWS,
    DEFAULT_COMPLETENESS_MOCK_OVERSAMPLE,
    DEFAULT_COMPLETENESS_SIM_FILE,
    SPEED_CHOICES,
    VALID_COMPLETENESS_MODES,
    _select_agn_fit_selection,
    _parse_completeness_mock_proposal_area,
    _completeness_stratification_checkpoint_payload,
    estimate_sky_box_area_deg2,
    generate_fresh_completeness_sim_file,
    make_run_tag,
    normalize_speed,
    standardization_plot_posterior_view,
    _validate_agn_pivot_context_for_reference,
    validate_completeness_mode,
    z_pivot_agn,
    z_pivot_sna,
)
from qvc.hubble.completeness_strata import (
    COMPLETENESS_STRATIFICATION_CHOICES,
    COMPLETENESS_STRATUM_CODE_COL,
    COMPLETENESS_STRATUM_COL,
    StratifiedCompletenessBundle,
    build_completeness_params as build_completeness_params_for_strata,
    get_completeness_stratification_preset,
    make_stratified_dm_function,
    normalize_completeness_stratification,
    write_completeness_stratum_counts,
)
from qvc.hubble.hubble_likelihood import (
    _magnitude_integration_grid,
    _validate_observed_magnitude_support,
    log_likelihood,
)
from qvc.hubble.completeness_closure import (
    simulate_hubble_posterior_closure,
    write_completeness_closure_diagnostics,
)
from qvc.hubble.program_color_completeness import (
    build_hubble_completeness_map,
    read_color_completeness_artifact,
)
from qvc.hubble.hubble_model import (
    AgnPivotContext,
    agn_model_req_errs,
    agn_model_req_obs,
    agn_model_req_params,
    build_agn_pivot_context,
    get_model_params,
    validate_agn_observable_uncertainties,
)
from qvc.hubble.hubble_plotting import (
    HubblePosteriorDrawSelection,
    get_hubble_posterior_sample_indices,
    plot_blr_line_lags_vs_l2500,
    plot_completeness_diagnostics,
    plot_cosmo_corner,
    plot_delta_m_flux_recal_vs_redshift,
    plot_full_residuals,
    plot_hubble,
    plot_hubble_reddening_redshift_diagnostic,
    plot_L2500_vs_sigma_tau_separate,
    plot_catalog_quantity_vs_sigma_tau_separate,
    plot_predicted_L2500_vs_sigmahat,
    plot_redshift_histograms,
    plot_sigma_uv_mpred_correction,
)
from qvc.hubble.cuts import (
    SDSS_TARGET_SELECTION_CHOICES,
    normalize_sdss_target_selection,
)
from qvc.hubble.hubble_utils import (
    compute_age_universe_with_error,
    display_results_summary,
    get_qvc_result_dir,
    load_agn_data,
    load_pantheon_data,
    reduced_chi_squared,
    report_pivots,
    save_chains,
)


if jax_config is not None:  # pragma: no branch
    jax_config.update("jax_enable_x64", True)


def _require_jax_stack() -> None:
    missing = []
    if jax is None or jnp is None:
        missing.append("jax")
    if numpyro is None or dist is None or NestedSampler is None:
        missing.append("numpyro.contrib.nested_sampling")
    if missing:
        raise ImportError(
            "hubble_fit_jax.py requires optional dependencies that are not installed: "
            + ", ".join(missing)
        )


def _trapz_jax(y: jnp.ndarray, x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    return jnp.trapezoid(y, x=x, axis=axis)


JAX_COSMOLOGY_GRID_SIZE = 8192


def _normal_logpdf(x: jnp.ndarray, loc: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    scale = jnp.maximum(scale, 1e-12)
    z = (x - loc) / scale
    return -0.5 * z**2 - jnp.log(scale) - 0.5 * jnp.log(2.0 * jnp.pi)


def _sigma_lens_from_dc_jax(z: jnp.ndarray, dc: jnp.ndarray, amp: float = 0.06, z_ref: float = 1.0, power: float = 1.5) -> jnp.ndarray:
    z = jnp.asarray(z)
    dc_ref = jnp.interp(jnp.asarray([z_ref]), z, dc, left=dc[0], right=dc[-1])[0]
    ratio = jnp.clip(dc / jnp.maximum(dc_ref, 1e-12), 0.0)
    return amp * ratio**power


def _convert_wpwa_to_w0(params: dict[str, Any], zp: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    ap = 1.0 / (1.0 + zp)
    wp = params["wp"]
    wa = params["wa"]
    w0 = wp - (1.0 - ap) * wa
    return w0, wa


def _ez_inv_flat_jax(z: jnp.ndarray, params: dict[str, Any], cosmo_model: str, zp: float) -> jnp.ndarray:
    Om0 = params["Om0"]
    Ode0 = 1.0 - Om0
    if cosmo_model == "FlatLambdaCDM":
        w0 = -1.0
        wa = 0.0
    elif cosmo_model == "FlatwCDM":
        w0 = params["w0"]
        wa = 0.0
    elif cosmo_model == "Flatw0waCDM":
        w0 = params["w0"]
        wa = params["wa"]
    elif cosmo_model == "FlatwpwaCDM":
        w0, wa = _convert_wpwa_to_w0(params, zp)
    else:
        raise ValueError(f"Unsupported cosmo_model={cosmo_model!r} for JAX pipeline.")
    zp1 = 1.0 + z
    de = zp1 ** (3.0 * (1.0 + w0 + wa)) * jnp.exp(-3.0 * wa * z / zp1)
    ez2 = Om0 * zp1**3 + Ode0 * de
    return jax.lax.rsqrt(jnp.maximum(ez2, 1e-18))


def _comoving_distance_jax(
    z: jnp.ndarray,
    params: dict[str, Any],
    cosmo_model: str,
    zp: float,
) -> jnp.ndarray:
    """Return distances from one shared cumulative integration grid."""
    z = jnp.asarray(z)
    H0 = params["H0"]
    c_kms = 299792.458
    z_flat = jnp.ravel(z)
    z_max = jnp.maximum(jnp.max(z_flat), 1e-8)
    grid = jnp.linspace(0.0, z_max, JAX_COSMOLOGY_GRID_SIZE)
    integrand = _ez_inv_flat_jax(grid, params, cosmo_model, zp)
    increments = 0.5 * (integrand[1:] + integrand[:-1]) * (
        grid[1:] - grid[:-1]
    )
    cumulative = jnp.concatenate(
        [jnp.zeros(1, dtype=integrand.dtype), jnp.cumsum(increments)]
    )
    dc_flat = (c_kms / H0) * jnp.interp(z_flat, grid, cumulative)
    return jnp.reshape(dc_flat, z.shape)


def _distance_modulus_from_dc_jax(
    dc: jnp.ndarray,
    z_photon: jnp.ndarray,
) -> jnp.ndarray:
    dl = dc * (1.0 + jnp.asarray(z_photon))
    return 5.0 * jnp.log10(jnp.maximum(dl, 1e-12)) + 25.0


def _distance_modulus_from_redshifts_jax(
    z_distance: jnp.ndarray,
    z_photon: jnp.ndarray,
    params: dict[str, Any],
    cosmo_model: str,
    zp: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return distance modulus using separate distance and photon redshifts."""
    z_distance = jnp.asarray(z_distance)
    z_photon = jnp.asarray(z_photon)
    dc = _comoving_distance_jax(z_distance, params, cosmo_model, zp)
    mu = _distance_modulus_from_dc_jax(dc, z_photon)
    return mu, dc


def _distance_modulus_jax(z: jnp.ndarray, params: dict[str, Any], cosmo_model: str, zp: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return distance modulus and comoving distance for a single redshift."""
    return _distance_modulus_from_redshifts_jax(
        z,
        z,
        params,
        cosmo_model,
        zp,
    )


def _sigma_mu_from_z_err_jax(
    z: jnp.ndarray,
    z_err: jnp.ndarray,
    params: dict[str, Any],
    cosmo_model: str,
    zp: float,
    use_redshift_mu_term: bool = False,
    dc: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Propagate z uncertainty with the analytic evolved-mean derivative."""
    z = jnp.asarray(z)
    z_err = jnp.asarray(z_err)
    if dc is None:
        dc = _comoving_distance_jax(z, params, cosmo_model, zp)
    c_kms = 299792.458
    d_dc_dz = (c_kms / params["H0"]) * _ez_inv_flat_jax(
        z, params, cosmo_model, zp
    )
    d_mu_dz = (5.0 / jnp.log(10.0)) * (
        1.0 / (1.0 + z) + d_dc_dz / jnp.maximum(dc, 1e-12)
    )
    if use_redshift_mu_term:
        d_mu_dz = d_mu_dz + params["gamma_mu_z"] / (
            jnp.log(10.0) * (1.0 + z)
        )
    sigma_mu = jnp.abs(d_mu_dz) * z_err
    return jnp.where(jnp.isfinite(z_err) & (z_err > 0.0), sigma_mu, 0.0)


def _prepare_completeness_for_jax(
    completeness_params,
    *,
    selection_magnitude=None,
):
    if completeness_params is None:
        return None
    if isinstance(completeness_params, StratifiedCompletenessBundle):
        prepared = [
            _prepare_completeness_for_jax(
                params,
                selection_magnitude=selection_magnitude,
            )
            for params in completeness_params.params_by_stratum
        ]
        reference = prepared[0]
        for other in prepared[1:]:
            if other["mode"] != reference["mode"]:
                raise ValueError("Completeness strata have incompatible JAX modes.")
            for key in (
                "mag_centers",
                "z_centers",
                "magnitude_support",
                "integration_mag_grid",
            ):
                if key in reference and not np.allclose(
                    np.asarray(reference[key]), np.asarray(other[key])
                ):
                    raise ValueError(
                        f"Completeness strata have incompatible JAX grid {key!r}."
                    )
        return {
            **reference,
            "cube": jnp.stack([item["cube"] for item in prepared], axis=0),
            "sigma": jnp.stack([item["sigma"] for item in prepared], axis=0),
            "stratified": True,
        }
    model = completeness_params[0]
    magnitude_grid = np.asarray(completeness_params[1], dtype=float)
    magnitude_support = getattr(
        model,
        "magnitude_support",
        (float(magnitude_grid[0]), float(magnitude_grid[-1])),
    )
    integration_mag_grid = _magnitude_integration_grid(
        magnitude_grid,
        magnitude_support,
    )
    if selection_magnitude is not None:
        _validate_observed_magnitude_support(
            selection_magnitude,
            magnitude_support,
        )
    support_payload = {
        "magnitude_support": jnp.asarray(magnitude_support),
        "integration_mag_grid": jnp.asarray(integration_mag_grid),
    }
    if isinstance(model, Completeness2D):
        cmap = jnp.asarray(model._interp.values)
        return {
            **support_payload,
            "mode": "2d",
            "mag_centers": jnp.asarray(model.mag_centers),
            "z_centers": jnp.asarray(model.z_centers),
            "cube": cmap,
            "sigma": jnp.asarray(0.0),
        }
    raise TypeError(f"Unsupported completeness model type: {type(model)!r}")


def _interp_regular_2d(x, y, x_grid, y_grid, values):
    # Magnitude remains bounded; redshift extrapolates from the outermost cell.
    valid = (
        (x >= x_grid[0])
        & (x <= x_grid[-1])
        & jnp.isfinite(y)
    )
    ix = jnp.clip(
        jnp.searchsorted(x_grid, x, side="right") - 1,
        0,
        x_grid.shape[0] - 2,
    )
    iy = jnp.clip(
        jnp.searchsorted(y_grid, y, side="right") - 1,
        0,
        y_grid.shape[0] - 2,
    )
    tx = jnp.clip(
        (x - x_grid[ix]) / (x_grid[ix + 1] - x_grid[ix]), 0.0, 1.0
    )
    ty = (y - y_grid[iy]) / (y_grid[iy + 1] - y_grid[iy])
    v00 = values[ix, iy]
    v10 = values[ix + 1, iy]
    v01 = values[ix, iy + 1]
    v11 = values[ix + 1, iy + 1]
    interp = (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )
    return jnp.where(valid, jnp.clip(interp, 0.0, 1.0), 0.0)














def _completeness_loglike_jax(
    m_model,
    mu_err,
    z,
    completeness,
    stratum_codes=None,
):
    """Evaluate only a frozen two-dimensional C(m_HD, z) map."""
    if completeness is None:
        return 0.0
    if completeness["mode"] != "2d":
        raise TypeError("The JAX Hubble likelihood accepts only frozen 2D completeness maps.")
    m_grid = completeness.get("integration_mag_grid", completeness["mag_centers"])
    map_m_grid = jnp.clip(m_grid, completeness["mag_centers"][0], completeness["mag_centers"][-1])
    is_stratified = bool(completeness.get("stratified", False))
    if is_stratified:
        if stratum_codes is None:
            raise ValueError("Stratified JAX completeness requires stratum codes.")
        stratum_codes = jnp.asarray(stratum_codes, dtype=jnp.int32)
        sigma = completeness["sigma"][stratum_codes, None]
    else:
        sigma = completeness["sigma"]
    sig = jnp.sqrt(mu_err[:, None] ** 2 + sigma**2)

    def interpolate(cube):
        return _interp_regular_2d(
            map_m_grid[None, :], z[:, None], completeness["mag_centers"],
            completeness["z_centers"], cube,
        )

    if is_stratified:
        all_values = jax.vmap(interpolate)(completeness["cube"])
        p_det = all_values[stratum_codes, jnp.arange(z.shape[0]), :]
    else:
        p_det = interpolate(completeness["cube"])
    pdf_model = jnp.exp(_normal_logpdf(m_grid[None, :], m_model[:, None], sig))
    normalization = _trapz_jax(pdf_model * p_det, m_grid, axis=1)
    return jnp.sum(jnp.log(jnp.clip(normalization, 1e-300)))

def _prepare_agn_arrays(
    agn_data: dict[str, np.ndarray],
    *,
    agn_pivot_context: AgnPivotContext,
) -> dict[str, jnp.ndarray]:
    if agn_pivot_context is None:
        raise ValueError("AGN array preparation requires an explicit AgnPivotContext.")
    out = {
        k: jnp.asarray(v)
        for k, v in agn_data.items()
        if k not in {"object_id", COMPLETENESS_STRATUM_COL}
    }
    out["object_id"] = np.asarray(agn_data["object_id"]).astype(str)
    if COMPLETENESS_STRATUM_COL in agn_data:
        out[COMPLETENESS_STRATUM_COL] = np.asarray(
            agn_data[COMPLETENESS_STRATUM_COL]
        ).astype(str)
    obs = jnp.stack([out[k] for k in agn_model_req_obs], axis=0)
    err_numpy = np.stack(
        [np.asarray(agn_data[k], dtype=float) for k in agn_model_req_errs],
        axis=0,
    )
    validate_agn_observable_uncertainties(
        err_numpy,
        object_ids=out["object_id"],
    )
    err = jnp.asarray(err_numpy)
    pivots = jnp.asarray(
        agn_pivot_context.as_array(
            use_alpha_lambda_term=False,
            use_eta_sigma_term=False,
        )
    )
    out["_obs_arr"] = obs
    out["_err_arr"] = err
    out["_pivot_arr"] = pivots
    return out


def _prepare_pantheon_arrays(pantheon_data: dict[str, np.ndarray], L, lower, logdet):
    required = {
        "zHD",
        "zHEL",
        "m_b_corr",
        "IS_CALIBRATOR",
        "CEPH_DIST",
        "MU_SH0ES_ERR_DIAG",
    }
    missing = required - set(pantheon_data)
    if missing:
        raise KeyError(
            "Pantheon JAX likelihood is missing required fields "
            f"{sorted(missing)}; zHEL has no zHD fallback."
        )

    z_hd = np.asarray(pantheon_data["zHD"], dtype=float)
    z_hel = np.asarray(pantheon_data["zHEL"], dtype=float)
    if z_hd.shape != z_hel.shape:
        raise ValueError(
            "Pantheon zHD and zHEL must have identical shapes; "
            f"got {z_hd.shape} and {z_hel.shape}."
        )
    if not np.all(np.isfinite(z_hd)):
        raise ValueError("Pantheon zHD must contain only finite values.")
    if not np.all(np.isfinite(z_hel)):
        raise ValueError("Pantheon zHEL must contain only finite values.")

    return {
        "zHD": jnp.asarray(z_hd),
        "zHEL": jnp.asarray(z_hel),
        "m_b_corr": jnp.asarray(pantheon_data["m_b_corr"]),
        "IS_CALIBRATOR": jnp.asarray(np.asarray(pantheon_data["IS_CALIBRATOR"], dtype=bool)),
        "CEPH_DIST": jnp.asarray(pantheon_data["CEPH_DIST"]),
        "MU_SH0ES_ERR_DIAG": jnp.asarray(pantheon_data["MU_SH0ES_ERR_DIAG"]),
        "_sna_L": jnp.asarray(L),
        "_sna_lower": bool(lower),
        "_sna_logdet": jnp.asarray(float(logdet)),
    }


def _agn_model_jax(params_vec, obs_arr, pivot_arr):
    M0_agn, alpha_agn, beta_agn = params_vec
    log_sigma_uv = obs_arr[agn_model_req_obs.index("log_sigma_uv")]
    log_tau_uv = obs_arr[agn_model_req_obs.index("log_tau_uv_rf")]
    sig_piv = pivot_arr[agn_model_req_obs.index("log_sigma_uv")]
    tau_piv = pivot_arr[agn_model_req_obs.index("log_tau_uv_rf")]
    return M0_agn + alpha_agn * (log_sigma_uv - sig_piv) + beta_agn * (log_tau_uv - tau_piv)


def _agn_model_err_jax(params_vec, err_arr):
    _, alpha_agn, beta_agn = params_vec
    sig_std = err_arr[agn_model_req_errs.index("log_sigma_uv_std_psd")]
    tau_std = err_arr[agn_model_req_errs.index("log_tau_uv_rf_std_psd")]
    cov = err_arr[agn_model_req_errs.index("log_sigma_uv_log_tau_uv_rf_cov_psd")]
    var = (alpha_agn * sig_std) ** 2 + (beta_agn * tau_std) ** 2 + 2.0 * alpha_agn * beta_agn * cov
    return jnp.sqrt(jnp.maximum(var, 0.0))


def _pack_param_dict(theta: jnp.ndarray, model_labels: list[str]) -> dict[str, jnp.ndarray]:
    return {k: theta[i] for i, k in enumerate(model_labels)}


def _log_likelihood_jax(
    theta: jnp.ndarray,
    *,
    model_labels: list[str],
    cosmo_model: str,
    agn_data_jax: dict[str, Any],
    pantheon_jax: dict[str, Any],
    completeness_jax: dict[str, Any] | None,
    only_sna: bool,
    only_agn: bool,
    use_ceph_dist_calibration: bool,
    early_de_guard: bool,
    use_redshift_mu_term: bool = False,
) -> jnp.ndarray:
    params = _pack_param_dict(theta, model_labels)
    if early_de_guard and cosmo_model == "Flatw0waCDM":
        early_de_ok = params["w0"] + params["wa"] < 0.0
    else:
        early_de_ok = True

    # Evaluate the cosmology once on a shared cumulative grid for every
    # redshift needed by this likelihood call.  The final z=1 entry supplies
    # the weak-lensing reference distance without another integration.
    distance_redshift_parts = []
    if not only_agn:
        z_sn_hd = pantheon_jax["zHD"]
        distance_redshift_parts.append(z_sn_hd)
    if not only_sna:
        z_agn = agn_data_jax["z"]
        distance_redshift_parts.extend(
            [z_agn, jnp.asarray([1.0], dtype=z_agn.dtype)]
        )
    all_distance_redshifts = jnp.concatenate(distance_redshift_parts)
    all_dc = _comoving_distance_jax(
        all_distance_redshifts, params, cosmo_model, z_pivot_agn
    )
    distance_offset = 0

    if only_agn:
        ll_sn = 0.0
    else:
        z_sn_hel = pantheon_jax["zHEL"]
        is_cal = pantheon_jax["IS_CALIBRATOR"]
        n_sn = z_sn_hd.shape[0]
        dc_sn = all_dc[distance_offset:distance_offset + n_sn]
        distance_offset += n_sn
        mu_sn = _distance_modulus_from_dc_jax(dc_sn, z_sn_hel)
        if use_ceph_dist_calibration:
            mu_sn = jnp.where(is_cal, pantheon_jax["CEPH_DIST"], mu_sn)
        res_sn = pantheon_jax["m_b_corr"] - (mu_sn + params["M0_sn"])
        y = solve_triangular(pantheon_jax["_sna_L"], res_sn, lower=pantheon_jax["_sna_lower"])
        ll_sn = -0.5 * jnp.dot(y, y) - 0.5 * pantheon_jax["_sna_logdet"] - 0.5 * res_sn.shape[0] * jnp.log(2.0 * jnp.pi)
    if only_sna:
        return jnp.where(early_de_ok, ll_sn, -jnp.inf)

    agn_param_vec = jnp.stack([params[k] for k in agn_model_req_params], axis=0)
    M_pred = _agn_model_jax(agn_param_vec, agn_data_jax["_obs_arr"], agn_data_jax["_pivot_arr"])
    M_pred_err = _agn_model_err_jax(agn_param_vec, agn_data_jax["_err_arr"])
    n_agn = z_agn.shape[0]
    dc = all_dc[distance_offset:distance_offset + n_agn]
    dc_ref = all_dc[distance_offset + n_agn]
    mu_cosmo = _distance_modulus_from_dc_jax(dc, z_agn)
    sigma_lens = 0.06 * jnp.clip(
        dc / jnp.maximum(dc_ref, 1e-12), 0.0
    ) ** 1.5
    sigma_mu_z = _sigma_mu_from_z_err_jax(
        z_agn,
        agn_data_jax["z_err"],
        params,
        cosmo_model,
        z_pivot_agn,
        use_redshift_mu_term=use_redshift_mu_term,
        dc=dc,
    )
    mu_err = jnp.sqrt(
        agn_data_jax["apparent_mag_2500_err"] ** 2
        + M_pred_err**2
        + sigma_mu_z**2
        + sigma_lens**2
        + jnp.exp(params["log_f"]) ** 2
    )
    mu_pred = agn_data_jax["apparent_mag_2500"] - M_pred
    if use_redshift_mu_term:
        delta_mu_z = params["gamma_mu_z"] * jnp.log10(
            (1.0 + z_agn) / (1.0 + z_pivot_agn)
        )
    else:
        delta_mu_z = jnp.zeros_like(z_agn)
    mu_model = mu_cosmo + delta_mu_z
    ll_agn = jnp.sum(_normal_logpdf(mu_pred - mu_model, 0.0, mu_err))

    m_model = M_pred + mu_model
    if completeness_jax is not None:
        selection_magnitude = agn_data_jax[COMPLETENESS_MAG_COL]
        selection_magnitude_error = agn_data_jax[COMPLETENESS_MAG_ERR_COL]
        attenuation_offset = (
            selection_magnitude - agn_data_jax["apparent_mag_2500"]
        )
        selection_model_magnitude = m_model + attenuation_offset
        non_magnitude_variance = jnp.maximum(
            mu_err**2 - agn_data_jax["apparent_mag_2500_err"] ** 2,
            0.0,
        )
        selection_total_error = jnp.sqrt(
            non_magnitude_variance + selection_magnitude_error**2
        )
        ll_comp = _completeness_loglike_jax(
            selection_model_magnitude,
            selection_total_error,
            z_agn,
            completeness_jax,
            agn_data_jax.get(COMPLETENESS_STRATUM_CODE_COL),
        )
    else:
        ll_comp = 0.0
    return jnp.where(early_de_ok, ll_sn + ll_agn - ll_comp, -jnp.inf)


def _build_numpyro_nested_model(model_labels, priors, loglike_fn):
    _require_jax_stack()

    def model():
        theta = []
        for label in model_labels:
            low, high = priors[label]
            prior_distribution = dist.Uniform(
                low=jnp.asarray(low, dtype=jnp.float64),
                high=jnp.asarray(high, dtype=jnp.float64),
            )
            theta.append(
                numpyro.sample(
                    label,
                    prior_distribution,
                )
            )
        theta = jnp.stack(theta, axis=0)
        numpyro.factor("log_like", loglike_fn(theta))

    return model


def _run_numpyro_nested(model, model_labels, *, seed: int, num_live_points: int, max_samples: int, dlogz: float):
    key = jax.random.PRNGKey(seed)
    constructor_kwargs = {"num_live_points": num_live_points, "verbose": True}
    termination_kwargs = {"dlogZ": dlogz}
    if max_samples is not None:
        termination_kwargs["max_samples"] = max_samples

    ns = NestedSampler(
        model,
        constructor_kwargs=constructor_kwargs,
        termination_kwargs=termination_kwargs,
    )
    ns.run(key)

    sample_key = jax.random.PRNGKey(seed + 1)
    flat_samples = None
    results = getattr(ns, "_results", None)
    total_num_samples = None
    if results is not None and hasattr(results, "total_num_samples"):
        total_num_samples = int(np.asarray(results.total_num_samples))
    for kwargs in (
        {"rng_key": sample_key, "num_samples": total_num_samples, "group_by_chain": False},
        {"rng_key": sample_key, "num_samples": total_num_samples},
        {"rng_key": sample_key, "group_by_chain": False},
        {"rng_key": sample_key},
        {},
    ):
        try:
            if kwargs.get("num_samples") is None:
                continue
            samples = ns.get_samples(**kwargs)
            flat_samples = np.column_stack([np.asarray(samples[label]) for label in model_labels])
            break
        except TypeError:
            continue
    if flat_samples is None and hasattr(ns, "get_weighted_samples"):
        weighted_samples, _ = ns.get_weighted_samples()
        flat_samples = np.column_stack([np.asarray(weighted_samples[label]) for label in model_labels])
    if flat_samples is None and results is not None and hasattr(results, "samples"):
        flat_samples = np.column_stack([np.asarray(results.samples[label]) for label in model_labels])
    if flat_samples is None:
        raise RuntimeError("Could not extract posterior samples from NumPyro NestedSampler.")

    logZ = None
    logZerr = None
    if results is not None:
        for key_name in ("log_Z_mean", "logZ", "log_z", "log_evidence"):
            if hasattr(results, key_name):
                arr = np.asarray(getattr(results, key_name))
                logZ = float(arr[-1] if arr.ndim > 0 else arr)
                break
        for key_name in ("log_Z_uncert", "logZerr", "log_z_err", "log_evidence_err"):
            if hasattr(results, key_name):
                arr = np.asarray(getattr(results, key_name))
                logZerr = float(arr[-1] if arr.ndim > 0 else arr)
                break
    extra_fields = None
    for getter in ("get_extra_fields",):
        if hasattr(ns, getter):
            try:
                extra_fields = getattr(ns, getter)()
            except TypeError:
                extra_fields = None
            if extra_fields:
                break
    if isinstance(extra_fields, dict):
        for key_name in ("logZ", "log_z", "log_evidence"):
            if key_name in extra_fields:
                arr = np.asarray(extra_fields[key_name])
                logZ = float(arr[-1] if arr.ndim > 0 else arr)
                break
        for key_name in ("logZerr", "log_z_err", "log_evidence_err"):
            if key_name in extra_fields:
                arr = np.asarray(extra_fields[key_name])
                logZerr = float(arr[-1] if arr.ndim > 0 else arr)
                break
    return ns, flat_samples, logZ, logZerr


def _nested_speed_preset(speed: str, ndim: int) -> tuple[int, int, float]:
    """Match the Dynesty speed presets as closely as practical for NumPyro."""
    speed = normalize_speed(speed)
    if speed == "fastest":
        return 20, 10_000, 10.0
    if speed == "production":
        return max(1000, 50 * ndim), 500_000, 0.01
    if speed == "quick":
        return 25, 10_000, 0.01
    if speed == "standard":
        return 250, 100_000, 0.01
    raise ValueError(f"Unknown speed preset: {speed!r}")


def _compute_numpy_blobs_from_samples(
    flat_samples,
    *,
    model_labels,
    agn_data,
    pantheon_data,
    _sna_L,
    _sna_Lower,
    _sna_LogdetCov,
    cosmo_model,
    completeness_params,
    z_pivot_agn,
    agn_pivot_context: AgnPivotContext,
    only_sna,
    only_agn,
    disable_ceph_dist_calibration,
    use_planck_h0_prior,
    use_planck_om_prior,
    use_redshift_mu_term=False,
    early_de_guard=False,
):
    logls = []
    blobs = []
    for theta in np.asarray(flat_samples, dtype=float):
        logl, blob = log_likelihood(
            theta,
            agn_data=agn_data,
            pantheon_data=pantheon_data,
            _sna_L=_sna_L,
            _sna_Lower=_sna_Lower,
            _sna_LogdetCov=_sna_LogdetCov,
            cosmo_model=cosmo_model,
            completeness_params=completeness_params,
            z_pivot_agn=z_pivot_agn,
            agn_pivot_context=agn_pivot_context,
            agn_calibrators_data=None,
            use_planck_h0_prior=use_planck_h0_prior,
            use_planck_om_prior=use_planck_om_prior,
            use_redshift_mu_term=use_redshift_mu_term,
            use_ceph_dist_calibration=not disable_ceph_dist_calibration,
            early_de_guard=early_de_guard,
            only_sna=only_sna,
            only_agn=only_agn,
            use_full_cov=True,
        )
        logls.append(float(logl))
        blobs.append(np.asarray(blob, dtype=float))
    logls = np.asarray(logls, dtype=float)
    blobs = np.asarray(blobs, dtype=float)
    return logls, blobs


def run_single_jax(
    df_agn,
    df_agn_all,
    df_pantheon,
    _sna_L,
    _sna_Lower,
    _sna_LogdetCov,
    *,
    cosmo_model="Flatw0waCDM",
    completeness=True,
    z_range=(0.44, 3.16),
    speed="fastest",
    prefix="default_jax",
    completeness_sim_file=DEFAULT_COMPLETENESS_SIM_FILE,
    completeness_mode="old",
    color_completeness_artifact=None,
    completeness_stratification="none",
    completeness_magnitude="dereddened",
    only_sna=False,
    N=None,
    uniform_redshift_distribution=False,
    disable_ceph_dist_calibration=False,
    use_planck_h0_prior=False,
    use_planck_om_prior=False,
    only_agn=False,
    use_alpha_lambda_term=False,
    use_eta_sigma_term=False,
    use_redshift_log_f_term=False,
    use_redshift_mu_term=False,
    early_de_guard=False,
    completeness_closure_test=False,
    seed=42,
    agn_pivot_context=None,
    df_agn_completeness_parent=None,
):
    use_redshift_mu_term = bool(use_redshift_mu_term and not only_sna)
    completeness_stratification = normalize_completeness_stratification(
        completeness_stratification
    )
    if completeness_stratification != "none" and (not completeness or only_sna):
        raise ValueError(
            "Completeness stratification requires completeness and an AGN likelihood."
        )
    if (
        completeness_stratification != "none"
        and df_agn.attrs.get("sdss_target_selection", "all") != "all"
    ):
        raise ValueError(
            "Completeness stratification requires an unrestricted "
            "sdss_target_selection='all' parent sample."
        )
    _require_jax_stack()
    if use_alpha_lambda_term:
        raise NotImplementedError("run_single_jax does not support --fit_alpha_lambda_term yet.")
    if use_eta_sigma_term:
        raise NotImplementedError("run_single_jax does not support --fit_eta_sigma_term yet.")
    if use_redshift_log_f_term:
        raise NotImplementedError("run_single_jax does not support --fit_redshift_log_f_term yet.")
    validate_completeness_mode(completeness_mode)
    if completeness_mode != "old" and not color_completeness_artifact:
        raise ValueError("Color-dependent completeness requires a frozen artifact.")
    completeness_magnitude = normalize_completeness_magnitude(
        completeness_magnitude
    )
    if completeness:
        df_agn = prepare_completeness_magnitude_columns(
            df_agn,
            completeness_magnitude,
        )
        df_agn_all = prepare_completeness_magnitude_columns(
            df_agn_all,
            completeness_magnitude,
        )
        if df_agn_completeness_parent is None:
            df_agn_completeness_parent = df_agn.copy()
        else:
            df_agn_completeness_parent = prepare_completeness_magnitude_columns(
                df_agn_completeness_parent,
                completeness_magnitude,
            )
    speed = normalize_speed(speed)
    if only_sna and only_agn:
        raise ValueError("only_sna and only_agn cannot both be True.")
    use_planck_h0_prior = use_planck_h0_prior or disable_ceph_dist_calibration

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
        use_alpha_lambda_term=False,
        use_eta_sigma_term=False,
        use_redshift_mu_term=use_redshift_mu_term,
        completeness_stratification=completeness_stratification,
    )
    plot_path = f"plots/hubble/{prefix}/{run_tag}"
    os.makedirs(plot_path, exist_ok=True)
    print("Saving plots to", plot_path)
    if completeness:
        print(
            "Completeness magnitude: "
            f"{completeness_magnitude} "
            f"({df_agn.attrs['completeness_magnitude_source']})."
        )

    df_agn_fit = _select_agn_fit_selection(
        df_agn,
        z_range=z_range,
        N=N,
        uniform_redshift_distribution=uniform_redshift_distribution,
    )
    active_stratification = get_completeness_stratification_preset(
        completeness_stratification
    )
    if active_stratification is not None:
        for frame_name, frame in (("fit", df_agn_fit), ("parent", df_agn_all)):
            missing = {
                COMPLETENESS_STRATUM_COL,
                COMPLETENESS_STRATUM_CODE_COL,
            } - set(frame.columns)
            if missing:
                raise KeyError(
                    f"Stratified JAX completeness {frame_name} dataframe is "
                    f"missing {sorted(missing)}."
                )
        fit_codes = set(
            df_agn_fit[COMPLETENESS_STRATUM_CODE_COL]
            .to_numpy(dtype=int)
            .tolist()
        )
        expected_codes = set(range(len(active_stratification.strata)))
        if fit_codes != expected_codes:
            raise ValueError(
                "Fitted JAX sample must contain every active completeness stratum."
            )
        stratum_counts = write_completeness_stratum_counts(
            preset_name=completeness_stratification,
            before_cuts=df_agn_all,
            after_quality_cuts=df_agn,
            fitted=df_agn_fit,
            output_path=Path(plot_path) / "completeness_strata_summary.csv",
            cut_summary_path=Path("plots/hubble") / prefix / "cut_summary.txt",
        )
        print("Completeness stratum counts:")
        print(stratum_counts.to_string(index=False))
    if uniform_redshift_distribution:
        plot_redshift_histograms(df_pantheon, df_agn_fit, xscale="linear", plot_path=plot_path, only_agn=only_agn)
    else:
        plot_redshift_histograms(df_pantheon, df_agn, xscale="log", plot_path=plot_path, only_agn=only_agn)

    if only_sna:
        if agn_pivot_context is not None:
            raise ValueError("SNe-only JAX runs must not receive AGN pivot metadata.")
    else:
        if agn_pivot_context is None:
            agn_pivot_context = build_agn_pivot_context(
                df_agn_fit,
                z_range,
                use_alpha_lambda_term=False,
                use_eta_sigma_term=False,
            )
        _validate_agn_pivot_context_for_reference(
            agn_pivot_context,
            df_agn_fit,
            z_range=z_range,
            use_alpha_lambda_term=False,
            use_eta_sigma_term=False,
            require_reference_ids=True,
        )
    plot_delta_m_flux_recal_vs_redshift(df_agn_fit, plot_path=plot_path)
    if not only_sna:
        report_pivots(df_agn_fit, agn_pivot_context=agn_pivot_context)

    if completeness:
        if completeness_sim_file is None:
            completeness_area_deg2 = estimate_sky_box_area_deg2(df_agn_all)
            completeness_sim_file = generate_fresh_completeness_sim_file(
                plot_path,
                area_deg2=completeness_area_deg2,
                seed=seed,
                z_range=z_range,
                completeness_magnitude=completeness_magnitude,
            )
        if completeness_stratification == "none":
            completeness_params = get_completeness_function_2d(
                df_agn_completeness_parent,
                sim_file=completeness_sim_file,
                plot=True,
                plot_path=plot_path,
            )
        else:
            if completeness_mode != "old":
                raise ValueError("Color-dependent modes do not use count stratification.")
            completeness_params = build_completeness_params_for_strata(
                df_agn_fit,
                df_agn_all,
                completeness_mode="2d",
                completeness_sim_file=completeness_sim_file,
                plot=True,
                plot_path=plot_path,
                stratification=completeness_stratification,
            )
        if completeness_mode != "old":
            artifact = read_color_completeness_artifact(
                color_completeness_artifact
            )
            model_2d = build_hubble_completeness_map(
                completeness_mode,
                old_completeness=completeness_params[0],
                artifact=artifact,
                hd_magnitude=df_agn_fit[COMPLETENESS_MAG_COL].to_numpy(dtype=float),
                hd_redshift=df_agn_fit["z"].to_numpy(dtype=float),
            )
            completeness_params = (
                model_2d,
                model_2d.mag_centers,
                model_2d.z_centers,
                float(np.diff(model_2d.mag_centers)[0]),
                float(np.diff(model_2d.z_centers)[0]),
                0.0,
            )
    else:
        completeness_params = None

    agn_fields = agn_model_req_params + agn_model_req_obs + agn_model_req_errs
    agn_fields += ("apparent_mag_2500", "apparent_mag_2500_err", "z", "z_err", "object_id")
    if completeness:
        agn_fields += (COMPLETENESS_MAG_COL, COMPLETENESS_MAG_ERR_COL)
    if active_stratification is not None:
        agn_fields += (COMPLETENESS_STRATUM_COL, COMPLETENESS_STRATUM_CODE_COL)
    if "alpha_lambda" in df_agn_fit.columns:
        agn_fields += ("alpha_lambda",)
    agn_data = {}
    for col in agn_fields:
        if col not in df_agn_fit.columns:
            continue
        if col.endswith("_draws"):
            values = np.stack(df_agn_fit[col].to_numpy())
            agn_data[col] = values
        else:
            agn_data[col] = df_agn_fit[col].values

    pantheon_fields = ["zHD", "zHEL", "m_b_corr", "IS_CALIBRATOR", "CEPH_DIST", "MU_SH0ES_ERR_DIAG"]
    pantheon_data = {col: df_pantheon[col].values for col in pantheon_fields if col in df_pantheon.columns}

    if only_sna:
        agn_data_jax = {}
    else:
        agn_data_jax = _prepare_agn_arrays(
            agn_data,
            agn_pivot_context=agn_pivot_context,
        )
    pantheon_jax = _prepare_pantheon_arrays(pantheon_data, _sna_L, _sna_Lower, _sna_LogdetCov)
    completeness_jax = _prepare_completeness_for_jax(
        completeness_params,
        selection_magnitude=(
            agn_data.get(COMPLETENESS_MAG_COL)
            if completeness_params is not None
            else None
        ),
    )

    priors, model_labels, _ = get_model_params(
        cosmo_model,
        only_sna=only_sna,
        only_agn=only_agn,
        use_planck_h0_prior=use_planck_h0_prior,
        use_planck_om_prior=use_planck_om_prior,
        use_redshift_mu_term=use_redshift_mu_term,
    )
    loglike_fn = jax.jit(
        lambda theta: _log_likelihood_jax(
            theta,
            model_labels=model_labels,
            cosmo_model=cosmo_model,
            agn_data_jax=agn_data_jax,
            pantheon_jax=pantheon_jax,
            completeness_jax=completeness_jax,
            only_sna=only_sna,
            only_agn=only_agn,
            use_ceph_dist_calibration=not disable_ceph_dist_calibration,
            early_de_guard=early_de_guard,
            use_redshift_mu_term=use_redshift_mu_term,
        )
    )
    model = _build_numpyro_nested_model(model_labels, priors, loglike_fn)

    num_live_points, max_samples, dlogz = _nested_speed_preset(speed, len(model_labels))
    print(
        "Running NumPyro nested sampler with Dynesty-matched speed preset "
        f"{speed!r}: {num_live_points=} {max_samples=} {dlogz=}"
    )
    _, flat_samples, logZ, logZerr = _run_numpyro_nested(
        model,
        model_labels,
        seed=seed,
        num_live_points=num_live_points,
        max_samples=max_samples,
        dlogz=dlogz,
    )

    logls, blobs = _compute_numpy_blobs_from_samples(
        flat_samples,
        model_labels=model_labels,
        agn_data=agn_data,
        pantheon_data=pantheon_data,
        _sna_L=_sna_L,
        _sna_Lower=_sna_Lower,
        _sna_LogdetCov=_sna_LogdetCov,
        cosmo_model=cosmo_model,
        completeness_params=completeness_params,
        z_pivot_agn=z_pivot_agn,
        agn_pivot_context=agn_pivot_context,
        use_redshift_mu_term=use_redshift_mu_term,
        only_sna=only_sna,
        only_agn=only_agn,
        disable_ceph_dist_calibration=disable_ceph_dist_calibration,
        use_planck_h0_prior=use_planck_h0_prior,
        use_planck_om_prior=use_planck_om_prior,
        early_de_guard=early_de_guard,
    )
    idx_max_weight = int(np.argmax(logls))
    integrals_max_w = blobs[idx_max_weight, 0, :]
    dmi_max_w = blobs[idx_max_weight, 1, :]
    dmi_posterior_median = np.median(blobs[:, 1, :], axis=0)
    dmi_posterior_sigma = 0.5 * (
        np.percentile(blobs[:, 1, :], 84, axis=0)
        - np.percentile(blobs[:, 1, :], 16, axis=0)
    )
    posterior_sample_indices = get_hubble_posterior_sample_indices(
        len(flat_samples)
    )
    dmi_posterior_draws = HubblePosteriorDrawSelection(
        values=blobs[posterior_sample_indices, 1, :],
        sample_indices=posterior_sample_indices,
        object_ids=tuple(
            str(value) for value in df_agn_fit["object_id"].to_numpy()
        ),
    )
    dmi_selection_sigma_posterior_median = None
    if blobs.ndim == 3 and blobs.shape[1] >= 3:
        dmi_selection_sigma_posterior_median = np.median(blobs[:, 2, :], axis=0)

    checkpoint_folder = get_qvc_result_dir() / "hubble_posteriors" / prefix
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    checkpoint_file = str(checkpoint_folder / f"posteriors_{run_tag}_jax.h5")
    checkpoint_payload = dict(
        flat_samples=flat_samples,
        dmi_max_w=dmi_max_w,
        dmi_posterior_median=dmi_posterior_median,
        dmi_posterior_sigma=dmi_posterior_sigma,
        dmi_selection_sigma_posterior_median=dmi_selection_sigma_posterior_median,
        sigma_clip_pass_stage="single",
        logZ=logZ if logZ is not None else np.nan,
        logZerr=logZerr if logZerr is not None else np.nan,
        integrals_max_w=integrals_max_w,
        model_labels=np.asarray(model_labels, dtype=str),
        use_redshift_mu_term=bool(use_redshift_mu_term),
        completeness_mode=str(completeness_mode),
    )
    if completeness and not isinstance(completeness_params, StratifiedCompletenessBundle):
        active_map = completeness_params[0]
        checkpoint_payload["completeness_artifact_hash"] = str(
            getattr(active_map, "artifact_content_hash", "")
        )
        checkpoint_payload["completeness_old_hash"] = str(
            getattr(active_map, "old_completeness_hash", "")
        )
        checkpoint_payload["completeness_clipped_cells"] = np.argwhere(
            getattr(active_map, "clipped_cell_mask", np.zeros((0, 0), bool))
        )
    if not only_sna:
        checkpoint_payload.update(
            object_id_fit_selection=agn_data["object_id"],
            agn_pivot_observable_names=agn_pivot_context.observable_names,
            agn_pivot_values=agn_pivot_context.values,
            agn_pivot_z_range=agn_pivot_context.z_range,
            agn_pivot_reference_object_ids=agn_pivot_context.reference_object_ids,
            agn_pivot_rule=agn_pivot_context.rule,
        )
    checkpoint_payload.update(
        _completeness_stratification_checkpoint_payload(
            completeness_stratification, df_agn_fit
        )
    )
    save_chains(checkpoint_file, **checkpoint_payload)

    if completeness_closure_test and completeness and not only_sna:
        closure_bin_width = 0.2
        closure_z_lo = closure_bin_width * np.floor(z_range[0] / closure_bin_width)
        closure_z_hi = closure_bin_width * np.ceil(z_range[1] / closure_bin_width)
        closure_result = simulate_hubble_posterior_closure(
            posterior_samples=flat_samples,
            agn_data=df_agn_fit,
            cosmo_model=cosmo_model,
            z_pivot_agn=z_pivot_agn,
            agn_pivot_context=agn_pivot_context,
            completeness_params=completeness_params,
            redshift_bins=np.arange(
                closure_z_lo,
                closure_z_hi + 0.5 * closure_bin_width,
                closure_bin_width,
            ),
            seed=seed + 7679,
            max_posterior_draws=100,
            max_abs_mean_zscore=4.0,
            min_detected_per_bin=25,
            only_agn=only_agn,
            use_planck_h0_prior=use_planck_h0_prior,
            use_planck_om_prior=use_planck_om_prior,
            use_redshift_mu_term=use_redshift_mu_term,
        )
        closure_paths = write_completeness_closure_diagnostics(
            closure_result, plot_path
        )
        print(
            "Completeness posterior-predictive closure: "
            f"{'PASS' if closure_result.all_bins_pass else 'FAIL'}; "
            f"summary={closure_paths['summary_csv']}"
        )

    display_results_summary(
        flat_samples,
        cosmo_model,
        z_pivot_agn,
        sigma_sel_posterior_median=dmi_selection_sigma_posterior_median,
        use_redshift_mu_term=use_redshift_mu_term,
        model_labels_override=model_labels,
    )
    age, age_err = compute_age_universe_with_error(
        flat_samples,
        cosmo_model,
        max_eval=200,
        use_redshift_mu_term=use_redshift_mu_term,
        model_labels_override=model_labels,
    )

    if only_sna:
        print("Skipping AGN-specific post-processing and plots for SNe-only run.")
        return flat_samples, model_labels, logZ, logZerr, age, age_err

    authoritative_flat_samples = flat_samples
    authoritative_model_labels = model_labels
    flat_samples, model_labels = standardization_plot_posterior_view(
        flat_samples,
        model_labels,
    )

    debias_magnitude = (
        agn_data[COMPLETENESS_MAG_COL]
        if completeness
        else agn_data["apparent_mag_2500"]
    )
    if active_stratification is not None:
        dm_interp = make_stratified_dm_function(df_agn_fit, dmi_posterior_median)
    else:
        dm_interp = make_dm_function(
            debias_magnitude,
            agn_data["z"],
            dmi_posterior_median,
        )
    dmi_selection_sigma_interp = None
    if dmi_selection_sigma_posterior_median is not None:
        if active_stratification is not None:
            dmi_selection_sigma_interp = make_stratified_dm_function(
                df_agn_fit,
                dmi_selection_sigma_posterior_median,
            )
        else:
            dmi_selection_sigma_interp = make_dm_function(
                agn_data[COMPLETENESS_MAG_COL],
                agn_data["z"],
                dmi_selection_sigma_posterior_median,
            )

    plot_cosmo_corner(
        None,
        flat_samples,
        cosmo_model,
        z_pivot_sna,
        z_pivot_agn,
        show=False,
        plot_path=plot_path,
        speed=f"{speed}_jax",
        only_agn=only_agn,
        use_redshift_mu_term=use_redshift_mu_term,
    )
    plot_predicted_L2500_vs_sigmahat(
        flat_samples,
        df_agn_fit,
        cosmo_model=cosmo_model,
        z_pivot_agn=z_pivot_agn,
        debias=False,
        show_residuals=False,
        show=False,
        plot_path=plot_path,
        df_calibrators=None,
        z_range=z_range,
        agn_pivot_context=agn_pivot_context,
        use_redshift_mu_term=use_redshift_mu_term,
    )
    plot_predicted_L2500_vs_sigmahat(
        flat_samples,
        df_agn_fit,
        cosmo_model=cosmo_model,
        z_pivot_agn=z_pivot_agn,
        debias=False,
        show_residuals=True,
        show=False,
        plot_path=plot_path,
        df_calibrators=None,
        z_range=z_range,
        agn_pivot_context=agn_pivot_context,
        use_redshift_mu_term=use_redshift_mu_term,
    )
    plot_predicted_L2500_vs_sigmahat(
        flat_samples,
        df_agn_fit,
        cosmo_model=cosmo_model,
        z_pivot_agn=z_pivot_agn,
        debias=True,
        dm_interp=dm_interp,
        dmi_selection_sigma_interp=dmi_selection_sigma_interp,
        show_residuals=False,
        show=False,
        plot_path=plot_path,
        df_calibrators=None,
        z_range=z_range,
        agn_pivot_context=agn_pivot_context,
        use_redshift_mu_term=use_redshift_mu_term,
    )
    L_residuals_debiased, L_pred_std_debiased = plot_predicted_L2500_vs_sigmahat(
        flat_samples,
        df_agn_fit,
        cosmo_model=cosmo_model,
        z_pivot_agn=z_pivot_agn,
        debias=True,
        dm_interp=dm_interp,
        dmi_selection_sigma_interp=dmi_selection_sigma_interp,
        show_residuals=True,
        show=False,
        plot_path=plot_path,
        df_calibrators=None,
        z_range=z_range,
        agn_pivot_context=agn_pivot_context,
        use_redshift_mu_term=use_redshift_mu_term,
    )
    plot_L2500_vs_sigma_tau_separate(
        flat_samples,
        df_agn_fit,
        cosmo_model=cosmo_model,
        z_pivot_agn=z_pivot_agn,
        debias=True,
        dm_interp=dm_interp,
        dmi_selection_sigma_interp=dmi_selection_sigma_interp,
        show_residuals=False,
        show=False,
        plot_path=plot_path,
        z_range=z_range,
        agn_pivot_context=agn_pivot_context,
        use_redshift_mu_term=use_redshift_mu_term,
    )
    plot_L2500_vs_sigma_tau_separate(
        flat_samples,
        df_agn_fit,
        cosmo_model=cosmo_model,
        z_pivot_agn=z_pivot_agn,
        debias=True,
        dm_interp=dm_interp,
        dmi_selection_sigma_interp=dmi_selection_sigma_interp,
        show_residuals=True,
        show=False,
        plot_path=plot_path,
        z_range=z_range,
        agn_pivot_context=agn_pivot_context,
        use_redshift_mu_term=use_redshift_mu_term,
    )
    plot_catalog_quantity_vs_sigma_tau_separate(
        df_agn_fit,
        y_col="LOGMBH",
        yerr_col="LOGMBH_ERR",
        y_label=r"$\log M_{\rm BH}$",
        filename="MBH_vs_sigma_tau_separate.pdf",
        plot_path=plot_path,
        show=False,
        z_range=z_range,
    )
    plot_catalog_quantity_vs_sigma_tau_separate(
        df_agn_fit,
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
        df_agn_fit,
        cosmo_model,
        z_pivot_agn,
        dm_interp,
        plot_path=plot_path,
        show=False,
        use_redshift_mu_term=use_redshift_mu_term,
    )
    alpha_agn_idx = model_labels.index("alpha_agn")
    alpha_agn_median = float(np.nanmedian(flat_samples[:, alpha_agn_idx]))
    plot_sigma_uv_mpred_correction(
        df_agn_fit,
        alpha_agn_median,
        plot_path=plot_path,
        show=False,
        filename="sigma_uv_mpred_correction_postcut.pdf",
    )
    chisq_red_L2500, _ = reduced_chi_squared(L_residuals_debiased, L_pred_std_debiased, n_params=len(model_labels) - 1)
    print(f"Reduced chi-squared (debiased) M2500: {chisq_red_L2500:.3f}")
    plot_full_residuals(
        df_agn_fit,
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
        residual_label="L2500_sigma_tau_residuals",
        output_tag="full_residuals_l2500_sigma_tau",
        use_redshift_mu_term=use_redshift_mu_term,
    )

    r = plot_hubble(
        flat_samples,
        df_agn_fit,
        df_pantheon,
        cosmo_model=cosmo_model,
        z_pivot_agn=z_pivot_agn,
        show_true=False,
        show=False,
        debias=True,
        dm_interp=dm_interp,
        plot_path=plot_path,
        cosmo_model_samples={},
        verbose=True,
        residuals_sigma_clip=None,
        df_calibrators=None,
        dmi_values=dmi_posterior_median,
        dmi_sigma=dmi_posterior_sigma,
        dmi_selection_sigma=dmi_selection_sigma_posterior_median,
        z_range=z_range,
        only_agn=only_agn,
        dmi_posterior_draws=dmi_posterior_draws,
        posterior_sample_indices=posterior_sample_indices,
        agn_pivot_context=agn_pivot_context,
        use_redshift_mu_term=use_redshift_mu_term,
    )
    (
        debiased_residuals,
        _debiased_clipping_sigma,
        _,
        mu_pred_std_debiased,
        mu_pred_std_debiased_with_scatter,
    ) = r
    for cut_stage, sample_label in (
        ("precut", "pre-cut sample (sigma clipping disabled)"),
        ("postcut", "post-cut sample (sigma clipping disabled)"),
    ):
        plot_hubble_reddening_redshift_diagnostic(
            df_agn_fit,
            debiased_residuals,
            plot_path=plot_path,
            show=False,
            filename=f"hubble_reddening_redshift_diagnostic_{cut_stage}.pdf",
            sample_label=sample_label,
        )
    n_agn_params = sum(label != "M0_sn" for label in model_labels)
    hubble_chi2_mask = (
        df_agn_fit["z"]
        .between(z_range[0], z_range[1])
        .to_numpy(dtype=bool)
        & np.isfinite(debiased_residuals)
        & np.isfinite(mu_pred_std_debiased)
        & np.isfinite(mu_pred_std_debiased_with_scatter)
        & (mu_pred_std_debiased > 0.0)
        & (mu_pred_std_debiased_with_scatter > 0.0)
    )
    if np.count_nonzero(hubble_chi2_mask) > n_agn_params:
        chisq_red_hubble_debiased_full, _ = reduced_chi_squared(
            debiased_residuals[hubble_chi2_mask],
            mu_pred_std_debiased_with_scatter[hubble_chi2_mask],
            n_params=n_agn_params,
        )
        chisq_red_hubble_debiased_data_only, _ = reduced_chi_squared(
            debiased_residuals[hubble_chi2_mask],
            mu_pred_std_debiased[hubble_chi2_mask],
            n_params=n_agn_params,
        )
    else:
        chisq_red_hubble_debiased_full = np.nan
        chisq_red_hubble_debiased_data_only = np.nan
    print(
        "Reduced chi-squared (debiased) Hubble, full: "
        f"{chisq_red_hubble_debiased_full:.3f}"
    )
    print(
        "Reduced chi-squared (debiased) Hubble, data only: "
        f"{chisq_red_hubble_debiased_data_only:.3f}"
    )

    plot_completeness_diagnostics(
        dmi_posterior_median,
        agn_data["z"],
        agn_data[COMPLETENESS_MAG_COL] if completeness else agn_data["apparent_mag_2500"],
        integrals_max_w,
        plot_path=plot_path,
        z_range=z_range,
        completeness_strata=agn_data.get(COMPLETENESS_STRATUM_COL),
    )
    return (
        authoritative_flat_samples,
        authoritative_model_labels,
        logZ,
        logZerr,
        age,
        age_err,
    )


def main():
    parser = argparse.ArgumentParser(description="Experimental JAX/NumPyro nested-sampling Hubble-fit pipeline.", allow_abbrev=True)
    parser.add_argument(
        "--fit_redshift_mu_term",
        action="store_true",
        default=False,
        help=(
            "Fit gamma_mu_z * log10((1+z)/(1+z_pivot)) as an AGN "
            "mean distance-modulus evolution term."
        ),
    )
    parser.add_argument("agn_data_filepath", type=str, help="Path to AGN data file")
    parser.add_argument("--cosmo_model", type=str, default="Flatw0waCDM", choices=["FlatLambdaCDM", "FlatwCDM", "Flatw0waCDM", "FlatwpwaCDM"])
    parser.add_argument("--speed", type=str, choices=SPEED_CHOICES, default="production")
    parser.add_argument("--spectra_fit_h5", type=str, nargs="+", required=True)
    parser.add_argument("--sdss-target-metadata-h5", default=None)
    parser.add_argument(
        "--sdss-target-selection",
        "--sdss_target_selection",
        dest="sdss_target_selection",
        type=normalize_sdss_target_selection,
        choices=SDSS_TARGET_SELECTION_CHOICES,
        default="all",
        help=(
            "SDSS targeting population to fit. Unlike quality cuts, this sample "
            "definition remains active with --no-cuts."
        ),
    )
    parser.add_argument(
        "--magnitude-convention",
        type=str,
        choices=["dereddened", "attenuated"],
        required=True,
        help=(
            "Choose which spectral 2500-A magnitude populates the Hubble-workflow "
            "apparent_mag_2500 aliases. This option is required."
        ),
    )
    parser.add_argument("--prefix", type=str, default="default_jax")
    parser.add_argument("--z_range", type=float, nargs=2, default=[0.44, 3.16])
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--only_sna", action="store_true", default=False)
    parser.add_argument("--only_agn", action="store_true", default=False)
    parser.add_argument("--uniform_redshift_distribution", action="store_true", default=False)
    parser.add_argument("--disable_completeness", action="store_true", default=False)
    parser.add_argument("--disable_ceph_dist_calibration", action="store_true", default=False)
    parser.add_argument("--use_planck_h0_prior", action="store_true", default=False)
    parser.add_argument("--use_planck_om_prior", action="store_true", default=False)
    parser.add_argument(
        "--early-de-guard",
        action="store_true",
        default=False,
        help="Reject Flatw0waCDM samples with w0 + wa >= 0. Disabled by default.",
    )
    parser.add_argument(
        "--completeness_sim_file",
        type=str,
        default=DEFAULT_COMPLETENESS_SIM_FILE,
        help="Optional mock catalog HDF5 override. If omitted, generate or reuse a validated area-scaled mock cache.",
    )
    parser.add_argument(
        "--completeness-mock-oversample",
        type=float,
        default=DEFAULT_COMPLETENESS_MOCK_OVERSAMPLE,
    )
    parser.add_argument(
        "--completeness-mock-max-rows",
        type=int,
        default=DEFAULT_COMPLETENESS_MOCK_MAX_ROWS,
    )
    parser.add_argument(
        "--completeness-mock-proposal-area",
        default="full_sky",
    )
    parser.add_argument(
        "--allow-spectra-catalog-v1",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--completeness-mode", dest="completeness_mode", type=str,
        choices=list(VALID_COMPLETENESS_MODES), default="old",
    )
    parser.add_argument("--color-completeness-artifact", default=None)
    parser.add_argument(
        "--completeness-stratification",
        "--completeness_stratification",
        dest="completeness_stratification",
        type=normalize_completeness_stratification,
        choices=COMPLETENESS_STRATIFICATION_CHOICES,
        default="none",
    )
    parser.add_argument(
        "--completeness_magnitude",
        type=str,
        choices=list(VALID_COMPLETENESS_MAGNITUDES),
        default="dereddened",
    )
    parser.add_argument("--correct-sigma-uv-host", action="store_true", default=False)
    parser.add_argument(
        "--completeness-closure-test", action="store_true", default=False
    )
    parser.add_argument(
        "--no-cuts",
        "--no_cuts",
        dest="no_cuts",
        action="store_true",
        default=False,
        help="Disable all AGN data cuts (default: False).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not np.isfinite(args.completeness_mock_oversample) or args.completeness_mock_oversample < 1.0:
        parser.error("--completeness-mock-oversample must be at least one.")
    if args.completeness_mock_max_rows <= 0:
        parser.error("--completeness-mock-max-rows must be positive.")
    try:
        _parse_completeness_mock_proposal_area(args.completeness_mock_proposal_area)
    except ValueError as exc:
        parser.error(str(exc))
    os.environ[COMPLETENESS_MOCK_OVERSAMPLE_ENV] = str(
        args.completeness_mock_oversample
    )
    os.environ[COMPLETENESS_MOCK_MAX_ROWS_ENV] = str(
        args.completeness_mock_max_rows
    )
    os.environ[COMPLETENESS_MOCK_PROPOSAL_AREA_ENV] = str(
        args.completeness_mock_proposal_area
    )
    args.speed = normalize_speed(args.speed)
    if args.only_sna and args.only_agn:
        raise ValueError("--only_sna and --only_agn cannot be used together.")
    if args.completeness_stratification != "none":
        if args.disable_completeness:
            raise ValueError(
                "--completeness-stratification requires completeness."
            )
        if args.sdss_target_selection != "all":
            raise ValueError(
                "--completeness-stratification requires --sdss-target-selection all."
            )
        if args.only_sna:
            raise ValueError(
                "--completeness-stratification requires an AGN likelihood."
            )
    effective_use_planck_h0_prior = args.use_planck_h0_prior or args.disable_ceph_dist_calibration

    _require_jax_stack()
    df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_pantheon_data()
    agn_plot_path = f"plots/hubble/{args.prefix}"
    df_agn, df_agn_all = load_agn_data(
        args.agn_data_filepath,
        apply_cut=not args.no_cuts,
        spectra_fit_h5=args.spectra_fit_h5,
        sdss_target_metadata_h5=args.sdss_target_metadata_h5,
        allow_spectra_catalog_v1=args.allow_spectra_catalog_v1,
        approximate_v1_fhost_2500_psf=False,
        magnitude_convention=args.magnitude_convention,
        completeness_magnitude=args.completeness_magnitude,
        sdss_target_selection=args.sdss_target_selection,
        completeness_stratification=args.completeness_stratification,
        correct_sigma_uv_host=args.correct_sigma_uv_host,
        z_range=tuple(args.z_range),
        plot_path=agn_plot_path,
        cut_report_path=Path(agn_plot_path) / "cut_summary.txt",
    )
    run_single_jax(
        df_agn,
        df_agn_all,
        df_pantheon,
        _sna_L,
        _sna_Lower,
        _sna_LogdetCov,
        cosmo_model=args.cosmo_model,
        completeness=not args.disable_completeness,
        z_range=tuple(args.z_range),
        speed=args.speed,
        prefix=args.prefix,
        completeness_sim_file=args.completeness_sim_file,
        completeness_mode=args.completeness_mode,
        color_completeness_artifact=args.color_completeness_artifact,
        completeness_stratification=args.completeness_stratification,
        completeness_magnitude=args.completeness_magnitude,
        only_sna=args.only_sna,
        only_agn=args.only_agn,
        N=args.N,
        uniform_redshift_distribution=args.uniform_redshift_distribution,
        disable_ceph_dist_calibration=args.disable_ceph_dist_calibration,
        use_planck_h0_prior=effective_use_planck_h0_prior,
        use_planck_om_prior=args.use_planck_om_prior,
        early_de_guard=args.early_de_guard,
        use_redshift_mu_term=args.fit_redshift_mu_term,
        completeness_closure_test=args.completeness_closure_test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
