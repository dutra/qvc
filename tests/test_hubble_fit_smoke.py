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
            "sdss_name": [f"{i:06d}.00+000000.0" for i in range(n_agn)],
            "ra": np.linspace(10.0, 20.0, n_agn),
            "dec": np.linspace(-5.0, 5.0, n_agn),
            "z": z,
            "z_err": np.full(n_agn, 0.002),
            "apparent_mag_2500": apparent_mag,
            "apparent_mag_2500_err": np.full(n_agn, 0.04),
            "log_sigma_hat0": log_sigma_hat0,
            "log_sigma_uv": log_sigma_uv,
            "log_sigma_uv_uncorrected": log_sigma_uv + 0.02,
            "log_tau_uv_rf": log_tau_uv,
            "log_sigma_hat0_err": np.full(n_agn, 0.04),
            "log_sigma_uv_std_psd": np.full(n_agn, 0.05),
            "log_tau_uv_rf_std_psd": np.full(n_agn, 0.06),
            "log_sigma_uv_log_tau_uv_rf_cov_psd": np.full(n_agn, 0.001),
            "f_host_2500": np.full(n_agn, 0.2),
            "f_host_2500_err": np.full(n_agn, 0.03),
            "f_bc_3000": np.full(n_agn, 0.12),
            "f_bc_3000_err": np.full(n_agn, 0.02),
            "f_fe_uv_3000": np.full(n_agn, 0.18),
            "f_fe_uv_3000_err": np.full(n_agn, 0.03),
            "f_na": np.full(n_agn, 0.04),
            "f_na_err": np.full(n_agn, 0.01),
            "f_br": np.full(n_agn, 0.06),
            "f_br_err": np.full(n_agn, 0.015),
            "delta_m_flux_recal": rng.normal(0.0, 0.02, size=n_agn),
            "eta_sigma": rng.normal(-0.45, 0.06, size=n_agn),
            "eta_sigma_err": np.full(n_agn, 0.03),
            "alpha_lambda": rng.normal(-1.7, 0.15, size=n_agn),
            "alpha_lambda_err": np.full(n_agn, 0.08),
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
    assert blob.shape == (3, len(df_agn))


def test_run_single_skip_plots_smoke(fake_data, monkeypatch, tmp_path):
    df_agn, df_pantheon = fake_data
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    flat_samples = np.tile(theta[None, :], (8, 1))
    dmi_posterior_median = np.zeros(len(df_agn))
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
            None,
            -50.0,
            0.2,
            dmi_posterior_median,
            dmi_posterior_sigma,
            None,
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
    dmi_posterior_median = np.zeros(len(df_agn))
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
            None,
            -25.0,
            0.15,
            dmi_posterior_median,
            dmi_posterior_sigma,
            None,
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


def test_run_single_calls_agn_table_only_for_joint_flatw0wa(monkeypatch, tmp_path):
    df_agn = _make_fake_agn_sample()
    df_pantheon = _make_fake_pantheon_sample()
    priors, model_labels, _ = hubble_model.get_model_params("Flatw0waCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    flat_samples = np.tile(theta[None, :], (8, 1))
    dmi_posterior_median = np.zeros(len(df_agn))
    dmi_posterior_sigma = np.full(len(df_agn), 0.05)
    latex_calls = []
    csv_calls = []

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
            None,
            -50.0,
            0.2,
            dmi_posterior_median,
            dmi_posterior_sigma,
            None,
        ),
    )
    monkeypatch.setattr(hubble_fit, "plot_predicted_L2500_vs_sigmahat", lambda *args, **kwargs: (np.zeros(len(df_agn)), np.ones(len(df_agn))))
    monkeypatch.setattr(hubble_fit, "plot_blr_line_lags_vs_l2500", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        hubble_fit,
        "plot_hubble",
        lambda *args, **kwargs: (
            np.zeros(len(df_agn)),
            np.ones(len(df_agn)),
            np.full(len(df_agn), 44.0),
            np.full(len(df_agn), 0.1),
            np.full(len(df_agn), 0.2),
        ),
    )
    monkeypatch.setattr(hubble_fit, "plot_hubble_residual_normality", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_predicted_vs_actual_M2500", lambda *args, **kwargs: (np.zeros(len(df_agn)), np.ones(len(df_agn)), None, None))
    monkeypatch.setattr(hubble_fit, "plot_full_residuals", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_full_residuals_rz", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_debias_impact_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_redshift_bin_residual_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_fast_vs_uv_variability", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_cosmo_corner", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_residuals_vs_alphaOX", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "make_agn_latex_table", lambda *args, **kwargs: latex_calls.append((args, kwargs)))
    monkeypatch.setattr(hubble_fit, "make_agn_csv_table", lambda *args, **kwargs: csv_calls.append((args, kwargs)))

    hubble_fit.run_single(
        df_agn=df_agn,
        df_agn_all=df_agn.copy(),
        df_pantheon=df_pantheon,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="Flatw0waCDM",
        completeness=False,
        use_full_cov=False,
        only_sna=False,
        speed="fast",
        z_range=(0.44, 3.16),
        skip_plots=False,
        prefix="unit",
    )

    assert len(latex_calls) == 1
    assert len(csv_calls) == 1


def test_run_single_does_not_call_agn_table_for_only_sna(monkeypatch, tmp_path):
    df_agn = _make_fake_agn_sample()
    df_pantheon = _make_fake_pantheon_sample()
    priors, model_labels, _ = hubble_model.get_model_params("Flatw0waCDM", only_sna=True)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    flat_samples = np.tile(theta[None, :], (8, 1))
    dmi_posterior_median = np.zeros(len(df_agn))
    dmi_posterior_sigma = np.full(len(df_agn), 0.05)
    latex_calls = []
    csv_calls = []

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
            None,
            -25.0,
            0.15,
            dmi_posterior_median,
            dmi_posterior_sigma,
            None,
        ),
    )
    monkeypatch.setattr(hubble_fit, "make_agn_latex_table", lambda *args, **kwargs: latex_calls.append((args, kwargs)))
    monkeypatch.setattr(hubble_fit, "make_agn_csv_table", lambda *args, **kwargs: csv_calls.append((args, kwargs)))

    hubble_fit.run_single(
        df_agn=df_agn,
        df_agn_all=df_agn.copy(),
        df_pantheon=df_pantheon,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="Flatw0waCDM",
        completeness=False,
        use_full_cov=False,
        only_sna=True,
        speed="fast",
        z_range=(0.44, 3.16),
        skip_plots=False,
        prefix="unit",
    )

    assert latex_calls == []
    assert csv_calls == []


def test_run_single_does_not_call_agn_table_when_skip_plots(monkeypatch, tmp_path):
    df_agn = _make_fake_agn_sample()
    df_pantheon = _make_fake_pantheon_sample()
    priors, model_labels, _ = hubble_model.get_model_params("Flatw0waCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    flat_samples = np.tile(theta[None, :], (8, 1))
    dmi_posterior_median = np.zeros(len(df_agn))
    dmi_posterior_sigma = np.full(len(df_agn), 0.05)
    latex_calls = []
    csv_calls = []

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
            None,
            -50.0,
            0.2,
            dmi_posterior_median,
            dmi_posterior_sigma,
            None,
        ),
    )
    monkeypatch.setattr(hubble_fit, "make_agn_latex_table", lambda *args, **kwargs: latex_calls.append((args, kwargs)))
    monkeypatch.setattr(hubble_fit, "make_agn_csv_table", lambda *args, **kwargs: csv_calls.append((args, kwargs)))

    hubble_fit.run_single(
        df_agn=df_agn,
        df_agn_all=df_agn.copy(),
        df_pantheon=df_pantheon,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="Flatw0waCDM",
        completeness=False,
        use_full_cov=False,
        only_sna=False,
        speed="fast",
        z_range=(0.44, 3.16),
        skip_plots=True,
        prefix="unit",
    )

    assert latex_calls == []
    assert csv_calls == []


def test_run_single_compare_sigma_only_skips_plotting_but_keeps_fit_outputs(monkeypatch, tmp_path):
    df_agn = _make_fake_agn_sample()
    df_pantheon = _make_fake_pantheon_sample()
    priors, model_labels, _ = hubble_model.get_model_params("Flatw0waCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    flat_samples = np.tile(theta[None, :], (8, 1))
    dmi_posterior_median = np.zeros(len(df_agn))
    dmi_posterior_sigma = np.full(len(df_agn), 0.05)

    monkeypatch.chdir(tmp_path)
    redshift_calls = []
    delta_m_calls = []
    monkeypatch.setattr(
        hubble_fit,
        "plot_redshift_histograms",
        lambda *args, **kwargs: redshift_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        hubble_fit,
        "plot_delta_m_flux_recal_vs_redshift",
        lambda *args, **kwargs: delta_m_calls.append((args, kwargs)),
    )
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
            None,
            -50.0,
            0.2,
            dmi_posterior_median,
            dmi_posterior_sigma,
            None,
        ),
    )

    result = hubble_fit.run_single(
        df_agn=df_agn,
        df_agn_all=df_agn.copy(),
        df_pantheon=df_pantheon,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="Flatw0waCDM",
        completeness=False,
        use_full_cov=False,
        only_sna=False,
        speed="fast",
        z_range=(0.44, 3.16),
        skip_plots=False,
        compare_sigma_only=True,
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
    assert redshift_calls == []
    assert delta_m_calls == []


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


def test_run_mcmc_pipeline_requires_eta_sigma_columns_when_flag_enabled(fake_data, monkeypatch, tmp_path):
    df_agn, df_pantheon = fake_data
    df_agn = df_agn.drop(columns=["eta_sigma"])
    monkeypatch.chdir(tmp_path)

    with pytest.raises(KeyError, match="fit_eta_sigma_term"):
        hubble_fit.run_mcmc_pipeline(
            df_agn=df_agn,
            df_agn_all=df_agn.copy(),
            df_pantheon=df_pantheon,
            _sna_L=None,
            _sna_Lower=True,
            _sna_LogdetCov=None,
            cosmo_model="FlatLambdaCDM",
            completeness=False,
            use_full_cov=False,
            speed="fast",
            use_eta_sigma_term=True,
        )


def test_run_mcmc_pipeline_requires_finite_eta_sigma_err_when_flag_enabled(fake_data, monkeypatch, tmp_path):
    df_agn, df_pantheon = fake_data
    df_agn = df_agn.copy()
    df_agn.loc[df_agn.index[0], "eta_sigma_err"] = np.nan
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="fit_eta_sigma_term"):
        hubble_fit.run_mcmc_pipeline(
            df_agn=df_agn,
            df_agn_all=df_agn.copy(),
            df_pantheon=df_pantheon,
            _sna_L=None,
            _sna_Lower=True,
            _sna_LogdetCov=None,
            cosmo_model="FlatLambdaCDM",
            completeness=False,
            use_full_cov=False,
            speed="fast",
            use_eta_sigma_term=True,
        )


def test_run_mcmc_pipeline_compare_sigma_only_skips_completeness_plots_on_resume(monkeypatch, tmp_path):
    df_agn = pd.DataFrame(
        {
            "object_id": ["agn_001", "agn_002"],
            "z": [0.6, 1.1],
            "z_err": [0.01, 0.01],
            "apparent_mag_2500": [20.1, 20.4],
            "apparent_mag_2500_err": [0.1, 0.1],
        }
    )
    df_pantheon = pd.DataFrame(
        {
            "zHD": [0.05, 0.1],
            "m_b_corr": [16.0, 17.0],
            "IS_CALIBRATOR": [0, 0],
            "CEPH_DIST": [-9.0, -9.0],
            "MU_SH0ES_ERR_DIAG": [0.1, 0.1],
        }
    )
    result_root = tmp_path / "result_root"
    expected = result_root / "hubble_posteriors" / "unit" / "posteriors_FlatLambdaCDM_joint_fast_all_z0p44_3p16.h5"
    completeness_calls = []
    diagnostics_calls = []

    monkeypatch.setattr(hubble_fit, "get_qvc_result_dir", lambda: result_root)
    monkeypatch.setattr(hubble_fit, "get_model_params", lambda *args, **kwargs: ({"H0": (60.0, 80.0)}, ["H0"], ["H0"]))
    monkeypatch.setattr(hubble_fit, "get_agn_model_spec", lambda *args, **kwargs: ((), (), ()))
    monkeypatch.setattr(hubble_fit, "make_dm_function", lambda *args, **kwargs: "interp")
    monkeypatch.setattr(
        hubble_fit,
        "plot_completeness_diagnostics",
        lambda *args, **kwargs: diagnostics_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        hubble_fit,
        "load_chains",
        lambda path: {
            "flat_samples": np.ones((3, 1)),
            "dmi_max_w": np.zeros(len(df_agn)),
            "dmi_posterior_median": np.zeros(len(df_agn)),
            "dmi_posterior_sigma": np.full(len(df_agn), 0.05),
            "integrals_max_w": np.ones(len(df_agn)),
            "logZ": -1.0,
            "logZerr": 0.2,
        },
    )
    monkeypatch.setattr(hubble_fit.os.path, "exists", lambda path: str(path) == str(expected))
    monkeypatch.setattr(
        hubble_fit,
        "get_completeness_function_2d",
        lambda *args, **kwargs: completeness_calls.append(kwargs.get("plot")),
    )

    result = hubble_fit.run_mcmc_pipeline(
        df_agn=df_agn,
        df_agn_all=df_agn.copy(),
        df_pantheon=df_pantheon,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM",
        completeness=True,
        use_full_cov=False,
        resume=True,
        speed="fast",
        prefix="unit",
        compare_sigma_only=True,
        completeness_sim_file="dummy_completeness.h5",
    )

    assert result[0].shape == (3, 1)
    assert diagnostics_calls == []
    assert completeness_calls == [False]
