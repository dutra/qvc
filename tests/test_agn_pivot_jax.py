from pathlib import Path
import sys

import numpy as np
import pytest


pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astropy.cosmology import FlatLambdaCDM

from qvc.hubble.hubble_fit_jax import (
    _agn_model_jax,
    _log_likelihood_jax,
    _prepare_agn_arrays,
    _prepare_pantheon_arrays,
    _sigma_mu_from_z_err_jax,
)
from qvc.hubble.hubble_likelihood import (
    log_likelihood as numpy_log_likelihood,
    pantheon_distance_modulus,
    sigma_mu_model_from_z_err,
)
from qvc.hubble.hubble_model import (
    M_model_agn,
    agn_model_pack_obs,
    build_agn_pivot_context,
    get_model_params,
)


def _skewed_agn_data():
    sigma_linear = np.array([0.24, 0.26, 8.0])
    tau_days = np.array([140.0, 260.0, 20_000.0])
    return {
        "object_id": np.array(["agn-a", "agn-b", "agn-c"]),
        "z": np.array([0.7, 0.8, 0.9]),
        "log_sigma_uv": np.log10(sigma_linear),
        "log_tau_uv_rf": np.log10(tau_days),
        "log_sigma_uv_std_psd": np.full(3, 0.04),
        "log_tau_uv_rf_std_psd": np.full(3, 0.05),
        "log_sigma_uv_log_tau_uv_rf_cov_psd": np.zeros(3),
    }


def _pivot_context(data):
    return build_agn_pivot_context(data, z_range=(0.5, 1.0))


def test_prepare_agn_arrays_uses_context_rounded_medians_not_means():
    data = _skewed_agn_data()
    context = _pivot_context(data)

    prepared = _prepare_agn_arrays(data, agn_pivot_context=context)
    actual = np.asarray(prepared["_pivot_arr"])
    expected = np.log10(np.array([0.3, 300.0]))
    arithmetic_means = np.array(
        [
            np.mean(data["log_sigma_uv"]),
            np.mean(data["log_tau_uv_rf"]),
        ]
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
    assert not np.allclose(actual, arithmetic_means, rtol=0.0, atol=1e-3)


def test_cpu_and_jax_agn_predictions_share_exact_pivot_context():
    data = _skewed_agn_data()
    context = _pivot_context(data)
    params = np.array([-23.1, -1.7, 0.8])

    obs_arr, _, pivot_arr = agn_model_pack_obs(
        data,
        pivot_context=context,
    )
    cpu_prediction = M_model_agn(params, obs_arr, pivot_arr)

    prepared = _prepare_agn_arrays(data, agn_pivot_context=context)
    jax_prediction = _agn_model_jax(
        jnp.asarray(params),
        prepared["_obs_arr"],
        prepared["_pivot_arr"],
    )

    np.testing.assert_allclose(
        np.asarray(jax_prediction),
        cpu_prediction,
        rtol=1e-12,
        atol=1e-12,
    )


def test_agn_array_preparation_rejects_missing_pivot_context():
    with pytest.raises(
        ValueError,
        match="requires an explicit AgnPivotContext",
    ):
        _prepare_agn_arrays(
            _skewed_agn_data(),
            agn_pivot_context=None,
        )


@pytest.mark.parametrize("covariance", [0.0021, -0.0021])
def test_agn_array_preparation_rejects_non_psd_covariance(covariance):
    data = _skewed_agn_data()
    data["log_sigma_uv_log_tau_uv_rf_cov_psd"][1] = covariance

    with pytest.raises(ValueError, match="covariance"):
        _prepare_agn_arrays(
            data,
            agn_pivot_context=_pivot_context(data),
        )


def test_jax_redshift_error_uses_complete_analytic_mean_derivative():
    z = np.array([0.6, 1.5, 2.8])
    z_err = np.array([0.01, 0.0, 0.02])
    params = {"H0": 70.0, "Om0": 0.3, "gamma_mu_z": 1.7}
    actual = np.asarray(
        _sigma_mu_from_z_err_jax(
            jnp.asarray(z),
            jnp.asarray(z_err),
            params,
            "FlatLambdaCDM",
            1.5,
            use_redshift_mu_term=True,
        )
    )
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    dc = cosmo.comoving_distance(z).value
    d_dc_dz = (299792.458 / params["H0"]) * cosmo.inv_efunc(z)
    expected = np.abs(
        (5.0 / np.log(10.0))
        * (1.0 / (1.0 + z) + d_dc_dz / dc)
        + params["gamma_mu_z"] / (np.log(10.0) * (1.0 + z))
    ) * z_err
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)

    # The analytic first-order propagation should also remain close to the
    # previous centered finite-difference reference for realistic z errors.
    finite_difference = sigma_mu_model_from_z_err(
        z, z_err, cosmo, params, z_pivot=1.5, use_redshift_mu_term=True
    )
    np.testing.assert_allclose(actual, finite_difference, rtol=4e-4, atol=2e-7)


def test_joint_jax_likelihood_matches_numpy_with_shared_cosmology_grid():
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    agn_data = {
        "object_id": np.array(["agn-a", "agn-b"]),
        "z": np.array([0.7, 2.1]),
        "z_err": np.zeros(2),
        "apparent_mag_2500_err": np.full(2, 0.05),
        "log_sigma_uv": np.array([-0.8, -0.6]),
        "log_tau_uv_rf": np.array([2.4, 2.7]),
        "log_sigma_uv_std_psd": np.full(2, 0.04),
        "log_tau_uv_rf_std_psd": np.full(2, 0.05),
        "log_sigma_uv_log_tau_uv_rf_cov_psd": np.zeros(2),
    }
    pivot_context = build_agn_pivot_context(agn_data, z_range=(0.5, 2.5))
    priors, labels, _ = get_model_params(
        "FlatLambdaCDM", use_redshift_mu_term=True
    )
    params = {label: 0.5 * sum(priors[label]) for label in labels}
    params.update(H0=70.0, Om0=0.3, gamma_mu_z=1.2)
    theta = np.array([params[label] for label in labels])
    agn_params = np.array(
        [params["M0_agn"], params["alpha_agn"], params["beta_agn"]]
    )
    obs, _, pivots = agn_model_pack_obs(
        agn_data, pivot_context=pivot_context
    )
    absolute_magnitude = M_model_agn(agn_params, obs, pivots)
    delta_mu = params["gamma_mu_z"] * np.log10(
        (1.0 + agn_data["z"]) / 2.5
    )
    agn_data["apparent_mag_2500"] = (
        absolute_magnitude + cosmo.distmod(agn_data["z"]).value + delta_mu
    )

    z_hd = np.array([0.02, 0.3, 1.4])
    z_hel = np.array([0.021, 0.29, 1.41])
    sn_sigma = np.full(3, 0.1)
    pantheon_data = {
        "zHD": z_hd,
        "zHEL": z_hel,
        "m_b_corr": pantheon_distance_modulus(cosmo, z_hd, z_hel)
        + params["M0_sn"],
        "IS_CALIBRATOR": np.zeros(3, dtype=bool),
        "CEPH_DIST": np.zeros(3),
        "MU_SH0ES_ERR_DIAG": sn_sigma,
    }
    covariance_cholesky = np.diag(sn_sigma)
    covariance_logdet = 2.0 * np.sum(np.log(sn_sigma))

    actual = _log_likelihood_jax(
        jnp.asarray(theta),
        model_labels=labels,
        cosmo_model="FlatLambdaCDM",
        agn_data_jax=_prepare_agn_arrays(
            agn_data, agn_pivot_context=pivot_context
        ),
        pantheon_jax=_prepare_pantheon_arrays(
            pantheon_data,
            covariance_cholesky,
            True,
            covariance_logdet,
        ),
        completeness_jax=None,
        only_sna=False,
        only_agn=False,
        use_ceph_dist_calibration=True,
        early_de_guard=False,
        use_redshift_mu_term=True,
    )
    expected, _ = numpy_log_likelihood(
        theta,
        agn_data=agn_data,
        pantheon_data=pantheon_data,
        _sna_L=covariance_cholesky,
        _sna_Lower=True,
        _sna_LogdetCov=covariance_logdet,
        cosmo_model="FlatLambdaCDM",
        completeness_params=None,
        z_pivot_agn=1.5,
        agn_pivot_context=pivot_context,
        use_redshift_mu_term=True,
        use_full_cov=True,
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0.0, atol=2e-5)
