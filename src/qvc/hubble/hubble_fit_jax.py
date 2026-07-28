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
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import config as jax_config
    from jax.scipy.linalg import solve_triangular
    from jax.scipy.special import ndtr as jax_ndtr
except Exception:  # pragma: no cover - optional dependency
    jax = None
    jnp = None
    jax_config = None
    solve_triangular = None
    jax_ndtr = None

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
    COMPLETENESS_FHOST_COL,
    Completeness2D,
    Completeness3D,
    Completeness4D,
    get_completeness_function_2d,
    get_completeness_function_3d_fhost,
    get_completeness_function_4d_fhost_alpha,
    get_relative_selection_function_2d,
    make_dm_function,
)
from qvc.hubble.hubble_fit import (
    DEFAULT_COMPLETENESS,
    DEFAULT_COMPLETENESS_MODE,
    DEFAULT_COMPLETENESS_SIM_FILE,
    SPEED_CHOICES,
    _select_agn_fit_selection,
    _relative_selection_checkpoint_payload,
    add_completeness_cli_arguments,
    make_run_tag,
    normalize_speed,
    resolve_completeness_sim_file,
    _validate_agn_pivot_context_for_reference,
    validate_completeness_mode,
    z_pivot_agn,
    z_pivot_sna,
)
from qvc.hubble.hubble_likelihood import log_likelihood
from qvc.hubble.hubble_model import (
    AgnPivotContext,
    agn_model_req_errs,
    agn_model_req_obs,
    agn_model_req_params,
    build_agn_pivot_context,
    get_model_params,
)
from qvc.hubble.hubble_plotting import (
    plot_blr_line_lags_vs_l2500,
    plot_completeness_diagnostics,
    plot_cosmo_corner,
    plot_delta_m_flux_recal_vs_redshift,
    plot_full_residuals,
    plot_hubble,
    plot_L2500_vs_sigma_tau_separate,
    plot_catalog_quantity_vs_sigma_tau_separate,
    plot_predicted_L2500_vs_sigmahat,
    plot_redshift_histograms,
    plot_sigma_uv_mpred_correction,
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


def _distance_modulus_jax(z: jnp.ndarray, params: dict[str, Any], cosmo_model: str, zp: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return distance modulus and comoving distance in Mpc."""
    z = jnp.asarray(z)
    H0 = params["H0"]
    c_kms = 299792.458

    def one_distance(zi):
        grid = jnp.linspace(0.0, jnp.maximum(zi, 1e-8), 256)
        integrand = _ez_inv_flat_jax(grid, params, cosmo_model, zp)
        dc = (c_kms / H0) * _trapz_jax(integrand, grid, axis=0)
        dl = dc * (1.0 + zi)
        mu = 5.0 * jnp.log10(jnp.maximum(dl, 1e-12)) + 25.0
        return mu, dc

    mu, dc = jax.vmap(one_distance)(z)
    return mu, dc


def _sigma_mu_from_z_err_jax(
    z: jnp.ndarray,
    z_err: jnp.ndarray,
    params: dict[str, Any],
    cosmo_model: str,
    zp: float,
) -> jnp.ndarray:
    z = jnp.asarray(z)
    z_err = jnp.asarray(z_err)
    z_lo = jnp.maximum(z - z_err, 1e-8)
    z_hi = jnp.maximum(z + z_err, z_lo + 1e-8)
    mu_lo, _ = _distance_modulus_jax(z_lo, params, cosmo_model, zp)
    mu_hi, _ = _distance_modulus_jax(z_hi, params, cosmo_model, zp)
    sigma_mu = 0.5 * jnp.abs(mu_hi - mu_lo)
    return jnp.where(jnp.isfinite(z_err) & (z_err > 0.0), sigma_mu, 0.0)


def _prepare_completeness_for_jax(completeness_params):
    if completeness_params is None:
        return None
    model = completeness_params[0]
    if isinstance(model, Completeness3D):
        cube = jnp.asarray(model._interp.values)
        return {
            "mode": "3d_fhost",
            "mag_centers": jnp.asarray(model.mag_centers),
            "z_centers": jnp.asarray(model.z_centers),
            "fhost_centers": jnp.asarray(model.fhost_centers),
            "cube": cube,
            "sigma": jnp.asarray(0.0),
        }
    if isinstance(model, Completeness4D):
        cube = jnp.asarray(model._interp.values)
        return {
            "mode": "4d_fhost_alpha",
            "mag_centers": jnp.asarray(model.mag_centers),
            "z_centers": jnp.asarray(model.z_centers),
            "fhost_centers": jnp.asarray(model.fhost_centers),
            "alpha_centers": jnp.asarray(model.alpha_centers),
            "cube": cube,
            "sigma": jnp.asarray(0.0),
        }
    if isinstance(model, Completeness2D):
        cmap = jnp.asarray(model._interp.values)
        return {
            "mode": model.mode,
            "mag_centers": jnp.asarray(model.mag_centers),
            "z_centers": jnp.asarray(model.z_centers),
            "cube": cmap,
            "sigma": jnp.asarray(0.0),
        }
    raise TypeError(f"Unsupported completeness model type: {type(model)!r}")


def _interp_regular_2d(x, y, x_grid, y_grid, values):
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    ux = (x - x_grid[0]) / dx
    uy = (jnp.clip(y, y_grid[0], y_grid[-1]) - y_grid[0]) / dy
    valid = (
        jnp.isfinite(x)
        & jnp.isfinite(y)
        & (ux >= 0.0)
        & (ux <= (x_grid.shape[0] - 1))
    )
    ix = jnp.clip(jnp.floor(ux).astype(jnp.int32), 0, x_grid.shape[0] - 2)
    iy = jnp.clip(jnp.floor(uy).astype(jnp.int32), 0, y_grid.shape[0] - 2)
    tx = jnp.clip(ux - ix, 0.0, 1.0)
    ty = jnp.clip(uy - iy, 0.0, 1.0)
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
    return jnp.where(valid, interp, 0.0)


def _interp_regular_3d(x, y, z, x_grid, y_grid, z_grid, values):
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    dz = z_grid[1] - z_grid[0]
    ux = (x - x_grid[0]) / dx
    uy = (jnp.clip(y, y_grid[0], y_grid[-1]) - y_grid[0]) / dy
    uz = (jnp.clip(z, z_grid[0], z_grid[-1]) - z_grid[0]) / dz
    valid = (
        jnp.isfinite(x)
        & jnp.isfinite(y)
        & jnp.isfinite(z)
        & (ux >= 0.0)
        & (ux <= (x_grid.shape[0] - 1))
    )
    ix = jnp.clip(jnp.floor(ux).astype(jnp.int32), 0, x_grid.shape[0] - 2)
    iy = jnp.clip(jnp.floor(uy).astype(jnp.int32), 0, y_grid.shape[0] - 2)
    iz = jnp.clip(jnp.floor(uz).astype(jnp.int32), 0, z_grid.shape[0] - 2)
    tx = jnp.clip(ux - ix, 0.0, 1.0)
    ty = jnp.clip(uy - iy, 0.0, 1.0)
    tz = jnp.clip(uz - iz, 0.0, 1.0)

    c000 = values[ix, iy, iz]
    c100 = values[ix + 1, iy, iz]
    c010 = values[ix, iy + 1, iz]
    c110 = values[ix + 1, iy + 1, iz]
    c001 = values[ix, iy, iz + 1]
    c101 = values[ix + 1, iy, iz + 1]
    c011 = values[ix, iy + 1, iz + 1]
    c111 = values[ix + 1, iy + 1, iz + 1]
    interp = (
        c000 * (1 - tx) * (1 - ty) * (1 - tz)
        + c100 * tx * (1 - ty) * (1 - tz)
        + c010 * (1 - tx) * ty * (1 - tz)
        + c110 * tx * ty * (1 - tz)
        + c001 * (1 - tx) * (1 - ty) * tz
        + c101 * tx * (1 - ty) * tz
        + c011 * (1 - tx) * ty * tz
        + c111 * tx * ty * tz
    )
    return jnp.where(valid, interp, 0.0)


def _interp_regular_4d(x, y, z, w, x_grid, y_grid, z_grid, w_grid, values):
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    dz = z_grid[1] - z_grid[0]
    dw = w_grid[1] - w_grid[0]
    ux = (x - x_grid[0]) / dx
    uy = (jnp.clip(y, y_grid[0], y_grid[-1]) - y_grid[0]) / dy
    uz = (jnp.clip(z, z_grid[0], z_grid[-1]) - z_grid[0]) / dz
    uw = (jnp.clip(w, w_grid[0], w_grid[-1]) - w_grid[0]) / dw
    valid = (
        jnp.isfinite(x)
        & jnp.isfinite(y)
        & jnp.isfinite(z)
        & jnp.isfinite(w)
        & (ux >= 0.0)
        & (ux <= (x_grid.shape[0] - 1))
    )
    ix = jnp.clip(jnp.floor(ux).astype(jnp.int32), 0, x_grid.shape[0] - 2)
    iy = jnp.clip(jnp.floor(uy).astype(jnp.int32), 0, y_grid.shape[0] - 2)
    iz = jnp.clip(jnp.floor(uz).astype(jnp.int32), 0, z_grid.shape[0] - 2)
    iw = jnp.clip(jnp.floor(uw).astype(jnp.int32), 0, w_grid.shape[0] - 2)
    tx = jnp.clip(ux - ix, 0.0, 1.0)
    ty = jnp.clip(uy - iy, 0.0, 1.0)
    tz = jnp.clip(uz - iz, 0.0, 1.0)
    tw = jnp.clip(uw - iw, 0.0, 1.0)

    out = 0.0
    for ox in (0, 1):
        wx = tx if ox else (1.0 - tx)
        for oy in (0, 1):
            wy = ty if oy else (1.0 - ty)
            for oz in (0, 1):
                wz = tz if oz else (1.0 - tz)
                for ow in (0, 1):
                    ww = tw if ow else (1.0 - tw)
                    out = out + (
                        values[ix + ox, iy + oy, iz + oz, iw + ow]
                        * wx * wy * wz * ww
                    )
    return jnp.where(valid, out, 0.0)


def _completeness_loglike_jax(
    m_model,
    mu_err,
    z,
    completeness,
    f_host_2500_psf,
    alpha_lambda,
    *,
    return_blob=False,
):
    if completeness is None:
        if return_blob:
            empty = jnp.zeros((3, jnp.size(m_model)))
            return 0.0, empty
        return 0.0
    m_grid = completeness["mag_centers"]
    sigma = completeness["sigma"]
    sig = jnp.sqrt(mu_err[:, None] ** 2 + sigma**2)
    if completeness["mode"] == "4d_fhost_alpha":
        p_det = _interp_regular_4d(
            m_grid[None, :],
            z[:, None],
            f_host_2500_psf[:, None],
            alpha_lambda[:, None],
            completeness["mag_centers"],
            completeness["z_centers"],
            completeness["fhost_centers"],
            completeness["alpha_centers"],
            completeness["cube"],
        )
    elif completeness["mode"] == "3d_fhost":
        p_det = _interp_regular_3d(
            m_grid[None, :],
            z[:, None],
            f_host_2500_psf[:, None],
            completeness["mag_centers"],
            completeness["z_centers"],
            completeness["fhost_centers"],
            completeness["cube"],
        )
    else:
        p_det = _interp_regular_2d(
            m_grid[None, :],
            z[:, None],
            completeness["mag_centers"],
            completeness["z_centers"],
            completeness["cube"],
        )
    pdf_model = jnp.exp(_normal_logpdf(m_grid[None, :], m_model[:, None], sig))
    weighted_pdf = pdf_model * p_det
    Z = _trapz_jax(weighted_pdf, m_grid, axis=1)
    m_Z = _trapz_jax(weighted_pdf * m_grid[None, :], m_grid, axis=1)
    m2_Z = _trapz_jax(weighted_pdf * m_grid[None, :] ** 2, m_grid, axis=1)
    m_bright = m_grid[0]
    a = (m_bright - m_model) / sig[:, 0]
    cdf_bright = jax_ndtr(a)
    pdf_bright = jnp.exp(-0.5 * a**2) / jnp.sqrt(2.0 * jnp.pi)
    p_bright = p_det[:, 0]
    Z = Z + p_bright * cdf_bright
    m_Z = m_Z + p_bright * (
        m_model * cdf_bright - sig[:, 0] * pdf_bright
    )
    m2_Z = m2_Z + p_bright * (
        (m_model**2 + sig[:, 0] ** 2) * cdf_bright
        - sig[:, 0] * (m_model + m_bright) * pdf_bright
    )
    Z = jnp.clip(Z, 1e-300)
    loglike = jnp.sum(jnp.log(Z))
    if not return_blob:
        return loglike
    valid_Z = Z > 1e-298
    expected_mag = jnp.where(valid_Z, m_Z / Z, m_model)
    expected_mag2 = jnp.where(valid_Z, m2_Z / Z, m_model**2)
    dmi = expected_mag - m_model
    selection_sigma = jnp.sqrt(
        jnp.clip(expected_mag2 - expected_mag**2, 0.0)
    )
    return loglike, jnp.stack([Z, dmi, selection_sigma], axis=0)


def _prepare_agn_arrays(
    agn_data: dict[str, np.ndarray],
    *,
    agn_pivot_context: AgnPivotContext,
) -> dict[str, jnp.ndarray]:
    if agn_pivot_context is None:
        raise ValueError("AGN array preparation requires an explicit AgnPivotContext.")
    out = {k: jnp.asarray(v) for k, v in agn_data.items() if k != "object_id"}
    out["object_id"] = np.asarray(agn_data["object_id"]).astype(str)
    obs = jnp.stack([out[k] for k in agn_model_req_obs], axis=0)
    err = jnp.stack([out[k] for k in agn_model_req_errs], axis=0)
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
    return {
        "zHD": jnp.asarray(pantheon_data["zHD"]),
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
    return jnp.sqrt(jnp.maximum(var, 1e-18))


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
) -> jnp.ndarray:
    params = _pack_param_dict(theta, model_labels)
    if early_de_guard and cosmo_model == "Flatw0waCDM":
        early_de_ok = params["w0"] + params["wa"] < 0.0
    else:
        early_de_ok = True

    if only_agn:
        ll_sn = 0.0
    else:
        z_sn = pantheon_jax["zHD"]
        is_cal = pantheon_jax["IS_CALIBRATOR"]
        mu_sn, _ = _distance_modulus_jax(z_sn, params, cosmo_model, z_pivot_agn)
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
    z_agn = agn_data_jax["z"]
    mu_cosmo, dc = _distance_modulus_jax(z_agn, params, cosmo_model, z_pivot_agn)
    sigma_lens = _sigma_lens_from_dc_jax(z_agn, dc)
    sigma_mu_z = _sigma_mu_from_z_err_jax(
        z_agn,
        agn_data_jax["z_err"],
        params,
        cosmo_model,
        z_pivot_agn,
    )
    mu_err = jnp.sqrt(
        agn_data_jax["apparent_mag_2500_err"] ** 2
        + M_pred_err**2
        + sigma_mu_z**2
        + sigma_lens**2
        + jnp.exp(params["log_f"]) ** 2
    )
    mu_pred = agn_data_jax["apparent_mag_2500"] - M_pred
    ll_agn = jnp.sum(_normal_logpdf(mu_pred - mu_cosmo, 0.0, mu_err))

    m_model = M_pred + mu_cosmo
    if completeness_jax is not None:
        f_host_2500_psf = agn_data_jax.get(COMPLETENESS_FHOST_COL)
        ll_comp = _completeness_loglike_jax(
            m_model,
            mu_err,
            z_agn,
            completeness_jax,
            f_host_2500_psf,
            agn_data_jax.get("alpha_lambda"),
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
            theta.append(
                numpyro.sample(
                    label,
                    dist.Uniform(
                        low=jnp.asarray(low, dtype=jnp.float64),
                        high=jnp.asarray(high, dtype=jnp.float64),
                    ),
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
    completeness=DEFAULT_COMPLETENESS,
    z_range=(0.44, 3.16),
    speed="fastest",
    prefix="default_jax",
    completeness_sim_file=DEFAULT_COMPLETENESS_SIM_FILE,
    completeness_mode=DEFAULT_COMPLETENESS_MODE,
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
    early_de_guard=False,
    seed=42,
    agn_pivot_context=None,
):
    _require_jax_stack()
    if use_alpha_lambda_term:
        raise NotImplementedError("run_single_jax does not support --fit_alpha_lambda_term yet.")
    if use_eta_sigma_term:
        raise NotImplementedError("run_single_jax does not support --fit_eta_sigma_term yet.")
    if use_redshift_log_f_term:
        raise NotImplementedError("run_single_jax does not support --fit_redshift_log_f_term yet.")
    validate_completeness_mode(completeness_mode)
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
        disable_ceph_dist_calibration=disable_ceph_dist_calibration,
        use_planck_h0_prior=use_planck_h0_prior,
        use_planck_om_prior=use_planck_om_prior,
        use_alpha_lambda_term=False,
        use_eta_sigma_term=False,
    )
    plot_path = f"plots/hubble/{prefix}/{run_tag}"
    os.makedirs(plot_path, exist_ok=True)
    print("Saving plots to", plot_path)

    df_agn_fit = _select_agn_fit_selection(
        df_agn,
        z_range=z_range,
        N=N,
        uniform_redshift_distribution=uniform_redshift_distribution,
    )
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
        completeness_sim_file = resolve_completeness_sim_file(
            completeness=completeness,
            completeness_sim_file=completeness_sim_file,
            plot_path=plot_path,
            df_agn_all=df_agn_all,
            seed=seed,
        )
        if completeness_mode == "4d_fhost_alpha":
            completeness_params = get_completeness_function_4d_fhost_alpha(
                df_agn_fit,
                sim_file=completeness_sim_file,
                plot=True,
                plot_path=plot_path,
                df_agn_fhost_population=df_agn_all,
            )
        elif completeness_mode == "3d_fhost":
            completeness_params = get_completeness_function_3d_fhost(
                df_agn_fit,
                sim_file=completeness_sim_file,
                plot=True,
                plot_path=plot_path,
                df_agn_fhost_population=df_agn_all,
            )
        elif completeness_mode == "2d_relative_support":
            completeness_params = get_relative_selection_function_2d(
                df_agn_fit,
                sim_file=completeness_sim_file,
                plot=True,
                plot_path=plot_path,
            )
        else:
            completeness_params = get_completeness_function_2d(
                df_agn_fit, sim_file=completeness_sim_file, plot=True, plot_path=plot_path
            )
    else:
        completeness_params = None

    agn_fields = agn_model_req_params + agn_model_req_obs + agn_model_req_errs
    agn_fields += ("apparent_mag_2500", "apparent_mag_2500_err", "z", "z_err", "object_id")
    if COMPLETENESS_FHOST_COL in df_agn_fit.columns:
        agn_fields += (COMPLETENESS_FHOST_COL,)
    if "alpha_lambda" in df_agn_fit.columns:
        agn_fields += ("alpha_lambda",)
    agn_data = {col: df_agn_fit[col].values for col in agn_fields if col in df_agn_fit.columns}

    pantheon_fields = ["zHD", "m_b_corr", "IS_CALIBRATOR", "CEPH_DIST", "MU_SH0ES_ERR_DIAG"]
    pantheon_data = {col: df_pantheon[col].values for col in pantheon_fields if col in df_pantheon.columns}

    if only_sna:
        agn_data_jax = {}
    else:
        agn_data_jax = _prepare_agn_arrays(
            agn_data,
            agn_pivot_context=agn_pivot_context,
        )
    pantheon_jax = _prepare_pantheon_arrays(pantheon_data, _sna_L, _sna_Lower, _sna_LogdetCov)
    completeness_jax = _prepare_completeness_for_jax(completeness_params)

    priors, model_labels, _ = get_model_params(
        cosmo_model,
        only_sna=only_sna,
        only_agn=only_agn,
        use_planck_h0_prior=use_planck_h0_prior,
        use_planck_om_prior=use_planck_om_prior,
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
        dmi_selection_sigma_posterior_median=dmi_selection_sigma_posterior_median,
        sigma_clip_pass_stage="single",
        logZ=logZ if logZ is not None else np.nan,
        logZerr=logZerr if logZerr is not None else np.nan,
        integrals_max_w=integrals_max_w,
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
    if completeness and completeness_mode == "2d_relative_support":
        checkpoint_payload.update(
            _relative_selection_checkpoint_payload(completeness_params)
        )
    save_chains(checkpoint_file, **checkpoint_payload)

    display_results_summary(
        flat_samples,
        cosmo_model,
        z_pivot_agn,
        sigma_sel_posterior_median=dmi_selection_sigma_posterior_median,
    )
    age, age_err = compute_age_universe_with_error(flat_samples, cosmo_model, max_eval=200)

    if only_sna:
        print("Skipping AGN-specific post-processing and plots for SNe-only run.")
        return flat_samples, model_labels, logZ, logZerr, age, age_err

    dm_interp = make_dm_function(
        agn_data["apparent_mag_2500"],
        agn_data["z"],
        dmi_posterior_median,
        f_host_2500_psf=agn_data.get(COMPLETENESS_FHOST_COL),
        alpha_lambda=agn_data.get("alpha_lambda"),
    )
    dmi_selection_sigma_interp = None
    if dmi_selection_sigma_posterior_median is not None:
        dmi_selection_sigma_interp = make_dm_function(
            agn_data["apparent_mag_2500"],
            agn_data["z"],
            dmi_selection_sigma_posterior_median,
            f_host_2500_psf=agn_data.get(COMPLETENESS_FHOST_COL),
            alpha_lambda=agn_data.get("alpha_lambda"),
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
        z_range=z_range,
        only_agn=only_agn,
        agn_pivot_context=agn_pivot_context,
    )
    debiased_residuals, _debiased_clipping_sigma, _, mu_pred_std_debiased, _ = r
    hubble_chi2_mask = df_agn_fit["z"].between(z_range[0], z_range[1]).to_numpy(dtype=bool)
    if np.any(hubble_chi2_mask):
        chisq_red_hubble_debiased, _ = reduced_chi_squared(
            debiased_residuals[hubble_chi2_mask],
            mu_pred_std_debiased[hubble_chi2_mask],
            n_params=len(model_labels) - 1,
        )
    else:
        chisq_red_hubble_debiased = np.nan
    print(f"Reduced chi-squared (debiased) Hubble: {chisq_red_hubble_debiased:.3f}")

    plot_completeness_diagnostics(
        dmi_posterior_median,
        agn_data["z"],
        agn_data["apparent_mag_2500"],
        integrals_max_w,
        plot_path=plot_path,
        z_range=z_range,
    )
    return flat_samples, model_labels, logZ, logZerr, age, age_err


def main():
    parser = argparse.ArgumentParser(description="Experimental JAX/NumPyro nested-sampling Hubble-fit pipeline.", allow_abbrev=True)
    parser.add_argument("agn_data_filepath", type=str, help="Path to AGN data file")
    parser.add_argument("--cosmo_model", type=str, default="Flatw0waCDM", choices=["FlatLambdaCDM", "FlatwCDM", "Flatw0waCDM", "FlatwpwaCDM"])
    parser.add_argument("--speed", type=str, choices=SPEED_CHOICES, default="production")
    parser.add_argument("--spectra_fit_csv", type=str, nargs="+", required=True)
    parser.add_argument(
        "--magnitude-convention",
        type=str,
        choices=["intrinsic", "observed"],
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
    add_completeness_cli_arguments(parser)
    parser.add_argument("--disable_ceph_dist_calibration", action="store_true", default=False)
    parser.add_argument("--use_planck_h0_prior", action="store_true", default=False)
    parser.add_argument("--use_planck_om_prior", action="store_true", default=False)
    parser.add_argument(
        "--early-de-guard",
        action="store_true",
        default=False,
        help="Reject Flatw0waCDM samples with w0 + wa >= 0. Disabled by default.",
    )
    parser.add_argument("--correct-sigma-uv-host", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.speed = normalize_speed(args.speed)
    if args.only_sna and args.only_agn:
        raise ValueError("--only_sna and --only_agn cannot be used together.")
    effective_use_planck_h0_prior = args.use_planck_h0_prior or args.disable_ceph_dist_calibration

    _require_jax_stack()
    df_pantheon, _sna_LogdetCov, _sna_L, _sna_Lower = load_pantheon_data()
    agn_plot_path = f"plots/hubble/{args.prefix}"
    df_agn, df_agn_all = load_agn_data(
        args.agn_data_filepath,
        apply_cut=True,
        spectra_fit_csv=args.spectra_fit_csv,
        magnitude_convention=args.magnitude_convention,
        correct_sigma_uv_host=args.correct_sigma_uv_host,
        z_range=tuple(args.z_range),
        plot_path=agn_plot_path,
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
        only_sna=args.only_sna,
        only_agn=args.only_agn,
        N=args.N,
        uniform_redshift_distribution=args.uniform_redshift_distribution,
        disable_ceph_dist_calibration=args.disable_ceph_dist_calibration,
        use_planck_h0_prior=effective_use_planck_h0_prior,
        use_planck_om_prior=args.use_planck_om_prior,
        early_de_guard=args.early_de_guard,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
