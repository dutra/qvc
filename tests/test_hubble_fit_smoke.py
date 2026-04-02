import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.cosmology import FlatLambdaCDM


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble import hubble_fit, hubble_likelihood, hubble_model


def _make_fake_agn_sample(n_agn=24, seed=123):
    rng = np.random.default_rng(seed)
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)

    z = np.linspace(0.5, 2.2, n_agn)
    log_sigma_uv = rng.normal(-0.8, 0.12, size=n_agn)
    log_tau_uv = 2.7 + 0.35 * (z - np.mean(z)) + rng.normal(0.0, 0.08, size=n_agn)
    log_sigma_hat0 = log_sigma_uv - 1.4 + rng.normal(0.0, 0.05, size=n_agn)

    true_params = {
        "M0_agn": -23.4,
        "alpha_agn": -1.8,
        "beta_agn": -0.9,
        "M0_sn": -19.2,
        "H0": 70.0,
        "Om0": 0.3,
    }

    obs_dict = {
        "log_sigma_hat0": log_sigma_hat0,
        "log_sigma_uv": log_sigma_uv,
        "log_tau_uv_rf": log_tau_uv,
        "log_sigma_hat0_err": np.full(n_agn, 0.04),
        "log_sigma_uv_std_psd": np.full(n_agn, 0.05),
        "log_tau_uv_rf_std_psd": np.full(n_agn, 0.06),
        "log_sigma_uv_log_tau_uv_rf_cov_psd": np.full(n_agn, 0.001),
    }
    params_arr = hubble_model.agn_model_pack_params(true_params)
    obs_arr, _, pivots = hubble_model.agn_model_pack_obs(obs_dict)
    absolute_mag = hubble_model.M_model_agn(params_arr, obs_arr, pivots)
    mu = cosmo.distmod(z).value
    apparent_mag = absolute_mag + mu + rng.normal(0.0, 0.04, size=n_agn)

    return pd.DataFrame(
        {
            "object_id": [f"agn_{i:03d}" for i in range(n_agn)],
            "z": z,
            "z_err": np.full(n_agn, 0.002),
            "apparent_mag_2500": apparent_mag,
            "apparent_mag_2500_err": np.full(n_agn, 0.04),
            "log_sigma_hat0": log_sigma_hat0,
            "log_sigma_uv": log_sigma_uv,
            "log_tau_uv_rf": log_tau_uv,
            "log_sigma_hat0_err": np.full(n_agn, 0.04),
            "log_sigma_uv_std_psd": np.full(n_agn, 0.05),
            "log_tau_uv_rf_std_psd": np.full(n_agn, 0.06),
            "log_sigma_uv_log_tau_uv_rf_cov_psd": np.full(n_agn, 0.001),
            "delta_m_flux_recal": rng.normal(0.0, 0.02, size=n_agn),
        }
    )


def _make_fake_pantheon_sample(n_sne=18, seed=456):
    rng = np.random.default_rng(seed)
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    z = np.linspace(0.015, 0.9, n_sne)
    mu = cosmo.distmod(z).value
    m_b_corr = mu - 19.2 + rng.normal(0.0, 0.08, size=n_sne)
    return pd.DataFrame(
        {
            "zHD": z,
            "m_b_corr": m_b_corr,
            "IS_CALIBRATOR": np.zeros(n_sne, dtype=int),
            "CEPH_DIST": np.full(n_sne, -9.0),
            "MU_SH0ES_ERR_DIAG": np.full(n_sne, 0.08),
        }
    )


@pytest.fixture
def fake_data():
    df_agn = _make_fake_agn_sample()
    df_pantheon = _make_fake_pantheon_sample()
    return df_agn, df_pantheon


def test_log_likelihood_finite_on_fake_lcdm_data(fake_data):
    df_agn, df_pantheon = fake_data
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)

    agn_fields = hubble_model.agn_model_req_obs + hubble_model.agn_model_req_errs
    agn_fields += ("apparent_mag_2500", "apparent_mag_2500_err", "z", "z_err", "object_id")
    agn_data = {col: df_agn[col].to_numpy() for col in agn_fields}
    pantheon_data = {col: df_pantheon[col].to_numpy() for col in df_pantheon.columns}

    logl, blob = hubble_likelihood.log_likelihood(
        theta,
        agn_data=agn_data,
        pantheon_data=pantheon_data,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM",
        completeness_params=None,
        z_pivot_agn=hubble_fit.z_pivot_agn,
        agn_calibrators_data=None,
        only_sna=False,
        use_full_cov=False,
    )

    assert np.isfinite(logl)
    assert blob.shape == (2, len(df_agn))


def test_run_single_skip_plots_smoke(fake_data, monkeypatch, tmp_path):
    df_agn, df_pantheon = fake_data
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    flat_samples = np.tile(theta[None, :], (8, 1))
    dmi_posterior_sigma = np.full(len(df_agn), 0.05)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hubble_fit, "plot_redshift_histograms", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_delta_m_flux_recal_vs_redshift", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "report_pivots", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "display_results_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "compute_age_universe_with_error", lambda *args, **kwargs: (13.8, 0.1))
    monkeypatch.setattr(
        hubble_fit,
        "run_mcmc_pipeline",
        lambda *args, **kwargs: (
            flat_samples,
            model_labels,
            lambda pts: np.zeros(len(np.atleast_2d(pts))),
            -50.0,
            0.2,
            dmi_posterior_sigma,
        ),
    )

    result = hubble_fit.run_single(
        df_agn=df_agn,
        df_agn_all=df_agn.copy(),
        df_pantheon=df_pantheon,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM",
        completeness=False,
        use_full_cov=False,
        only_sna=False,
        speed="fast",
        z_range=(0.44, 3.16),
        skip_plots=True,
        prefix="unit",
    )

    samples_out, labels_out, _, logz, logzerr, residuals, age, age_err = result
    assert samples_out.shape == flat_samples.shape
    assert labels_out == model_labels
    assert logz == -50.0
    assert logzerr == 0.2
    assert residuals is None
    assert age == 13.8
    assert age_err == 0.1


def test_run_single_only_sna_smoke(fake_data, monkeypatch, tmp_path):
    df_agn, df_pantheon = fake_data
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=True)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    flat_samples = np.tile(theta[None, :], (8, 1))
    dmi_posterior_sigma = np.full(len(df_agn), 0.05)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hubble_fit, "plot_redshift_histograms", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_delta_m_flux_recal_vs_redshift", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "report_pivots", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "display_results_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "compute_age_universe_with_error", lambda *args, **kwargs: (13.7, 0.2))
    monkeypatch.setattr(
        hubble_fit,
        "run_mcmc_pipeline",
        lambda *args, **kwargs: (
            flat_samples,
            model_labels,
            lambda pts: np.zeros(len(np.atleast_2d(pts))),
            -25.0,
            0.15,
            dmi_posterior_sigma,
        ),
    )

    result = hubble_fit.run_single(
        df_agn=df_agn,
        df_agn_all=df_agn.copy(),
        df_pantheon=df_pantheon,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM",
        completeness=False,
        use_full_cov=False,
        only_sna=True,
        speed="fast",
        z_range=(0.44, 3.16),
        skip_plots=False,
        prefix="unit",
    )

    samples_out, labels_out, _, logz, logzerr, residuals, age, age_err = result
    assert samples_out.shape == flat_samples.shape
    assert labels_out == model_labels
    assert logz == -25.0
    assert logzerr == 0.15
    assert residuals is None
    assert age == 13.7
    assert age_err == 0.2


@pytest.mark.parametrize("resume_value, use_default_checkpoint", [(True, True), ("custom_resume.h5", False)])
def test_resolve_resume_checkpoint_path_requires_existing_file(tmp_path, resume_value, use_default_checkpoint):
    default_checkpoint = tmp_path / "default_resume.h5"
    expected_name = default_checkpoint.name if use_default_checkpoint else Path("custom_resume.h5").name

    with pytest.raises(FileNotFoundError, match=expected_name):
        hubble_fit.resolve_resume_checkpoint_path(resume_value, str(default_checkpoint))


@pytest.mark.parametrize("resume_value", ["True", "true", "1", "yes"])
def test_resolve_resume_checkpoint_path_treats_true_like_default_checkpoint(tmp_path, resume_value):
    default_checkpoint = tmp_path / "default_resume.h5"

    with pytest.raises(FileNotFoundError, match=default_checkpoint.name):
        hubble_fit.resolve_resume_checkpoint_path(resume_value, str(default_checkpoint))


def test_subsample_dataframe_at_most_clamps_oversized_requests_without_reordering():
    df = pd.DataFrame({"object_id": ["a", "b", "c"], "value": [1, 2, 3]})

    sampled, effective_n = hubble_fit.subsample_dataframe_at_most(df, 10, random_state=42, label="AGN objects")

    assert effective_n == 3
    assert sampled.equals(df)
