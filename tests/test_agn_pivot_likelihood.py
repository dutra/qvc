import numpy as np

from qvc.hubble import hubble_likelihood
from qvc.hubble.hubble_model import build_agn_pivot_context, get_model_params


def test_nearby_calibrators_and_main_sample_use_same_pivot_context(monkeypatch):
    agn_data = {
        "object_id": np.array(["main", "cal"]),
        "z": np.array([0.8, 1.0]),
        "z_err": np.array([0.001, 0.001]),
        "apparent_mag_2500": np.array([20.0, 20.5]),
        "apparent_mag_2500_err": np.array([0.05, 0.05]),
        "log_sigma_uv": np.array([-0.8, -0.5]),
        "log_tau_uv_rf": np.array([2.4, 2.8]),
        "log_sigma_uv_std_psd": np.array([0.04, 0.04]),
        "log_tau_uv_rf_std_psd": np.array([0.05, 0.05]),
        "log_sigma_uv_log_tau_uv_rf_cov_psd": np.array([0.0, 0.0]),
    }
    calibrators = {
        "object_id": np.array(["cal"]),
        "AGN_IS_CALIBRATOR": np.array([True]),
        "MU_CAL": np.array([40.0]),
        "MU_CAL_ERR": np.array([0.1]),
        "z": np.array([0.01]),
        "z_err": np.array([0.001]),
        "apparent_mag_2500": np.array([17.0]),
        "apparent_mag_2500_err": np.array([0.05]),
        "log_sigma_uv": np.array([3.0]),
        "log_tau_uv_rf": np.array([-3.0]),
        "log_sigma_uv_std_psd": np.array([0.04]),
        "log_tau_uv_rf_std_psd": np.array([0.05]),
        "log_sigma_uv_log_tau_uv_rf_cov_psd": np.array([0.0]),
    }
    pivot_context = build_agn_pivot_context(agn_data, (0.5, 1.5))
    priors, labels, _ = get_model_params(
        "FlatLambdaCDM",
        only_agn=True,
    )
    theta = np.array(
        [(priors[label][0] + priors[label][1]) / 2.0 for label in labels]
    )
    observed_contexts = []
    real_pack = hubble_likelihood.agn_model_pack_obs

    def recording_pack(*args, **kwargs):
        observed_contexts.append(kwargs["pivot_context"])
        return real_pack(*args, **kwargs)

    monkeypatch.setattr(
        hubble_likelihood,
        "agn_model_pack_obs",
        recording_pack,
    )

    log_likelihood, _ = hubble_likelihood.log_likelihood_nearbylcs(
        theta,
        agn_data=agn_data,
        agn_calibrators_data=calibrators,
        pantheon_data={},
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM",
        completeness_params=None,
        z_pivot_agn=1.5,
        agn_pivot_context=pivot_context,
        only_agn=True,
    )

    assert np.isfinite(log_likelihood)
    assert observed_contexts == [pivot_context, pivot_context]
    assert all(context is pivot_context for context in observed_contexts)


def test_nearby_calibrator_mean_uses_mu_cal_plus_redshift_evolution(monkeypatch):
    agn_data = {
        "object_id": np.array(["main"]),
        "z": np.array([0.8]),
        "z_err": np.array([0.001]),
        "apparent_mag_2500": np.array([20.0]),
        "apparent_mag_2500_err": np.array([0.05]),
        "log_sigma_uv": np.array([-0.8]),
        "log_tau_uv_rf": np.array([2.4]),
        "log_sigma_uv_std_psd": np.array([0.04]),
        "log_tau_uv_rf_std_psd": np.array([0.05]),
        "log_sigma_uv_log_tau_uv_rf_cov_psd": np.array([0.0]),
    }
    calibrators = {
        "object_id": np.array(["cal"]),
        "AGN_IS_CALIBRATOR": np.array([True]),
        "MU_CAL": np.array([40.0]),
        "MU_CAL_ERR": np.array([0.1]),
        "z": np.array([0.2]),
        "z_err": np.array([0.002]),
        "apparent_mag_2500": np.array([17.0]),
        "apparent_mag_2500_err": np.array([0.05]),
        "log_sigma_uv": np.array([-0.7]),
        "log_tau_uv_rf": np.array([2.5]),
        "log_sigma_uv_std_psd": np.array([0.04]),
        "log_tau_uv_rf_std_psd": np.array([0.05]),
        "log_sigma_uv_log_tau_uv_rf_cov_psd": np.array([0.0]),
    }
    pivot_context = build_agn_pivot_context(agn_data, (0.5, 1.0))
    priors, labels, _ = get_model_params(
        "FlatLambdaCDM", only_agn=True, use_redshift_mu_term=True
    )
    params = {label: 0.5 * sum(priors[label]) for label in labels}
    params["gamma_mu_z"] = 1.4
    theta = np.array([params[label] for label in labels])
    captured_residuals = []
    real_logpdf = hubble_likelihood._normal_logpdf_sum

    def capture_logpdf(residual, sigma):
        captured_residuals.append(np.asarray(residual, dtype=float))
        return real_logpdf(residual, sigma)

    monkeypatch.setattr(hubble_likelihood, "_normal_logpdf_sum", capture_logpdf)
    log_likelihood, _ = hubble_likelihood.log_likelihood_nearbylcs(
        theta,
        agn_data=agn_data,
        agn_calibrators_data=calibrators,
        pantheon_data={},
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM",
        completeness_params=None,
        z_pivot_agn=1.5,
        agn_pivot_context=pivot_context,
        only_agn=True,
        use_redshift_mu_term=True,
    )

    agn_params = hubble_likelihood.agn_model_pack_params(params)
    obs, _, pivots = hubble_likelihood.agn_model_pack_obs(
        calibrators, pivot_context=pivot_context
    )
    m_absolute = hubble_likelihood.M_model_agn(agn_params, obs, pivots)
    delta = params["gamma_mu_z"] * np.log10((1.0 + calibrators["z"]) / 2.5)
    expected_calibrator_residual = (
        calibrators["apparent_mag_2500"] - m_absolute
        - calibrators["MU_CAL"] - delta
    )
    assert np.isfinite(log_likelihood)
    np.testing.assert_allclose(captured_residuals[1], expected_calibrator_residual)
