import ast
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

from qvc.hubble import hubble_fit, hubble_likelihood, hubble_model, hubble_plotting, hubble_utils


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
            "apparent_mag_2500_intrinsic": apparent_mag,
            "apparent_mag_2500_intrinsic_err": np.full(n_agn, 0.04),
            "flux_aper_b": np.full(n_agn, 1.0e-14),
            "flux_aper_err_b": np.full(n_agn, 2.0e-15),
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
            "log_sigma0": log_sigma_uv - 0.25,
            "log_sigma0_err": np.full(n_agn, 0.04),
            "log_amp_delta_blr_u": np.full(n_agn, -0.30),
            "log_amp_delta_blr_u_err": np.full(n_agn, 0.04),
            "log_amp_delta_blr_g": np.full(n_agn, -0.25),
            "log_amp_delta_blr_g_err": np.full(n_agn, 0.04),
            "log_amp_delta_blr_r": np.full(n_agn, -0.20),
            "log_amp_delta_blr_r_err": np.full(n_agn, 0.04),
            "log_amp_delta_blr_i": np.full(n_agn, -0.15),
            "log_amp_delta_blr_i_err": np.full(n_agn, 0.04),
            "log_sigma_band_u": log_sigma_uv - 0.10,
            "log_sigma_band_u_err": np.full(n_agn, 0.04),
            "log_sigma_band_g": log_sigma_uv - 0.08,
            "log_sigma_band_g_err": np.full(n_agn, 0.04),
            "log_sigma_band_r": log_sigma_uv - 0.06,
            "log_sigma_band_r_err": np.full(n_agn, 0.04),
            "log_sigma_band_i": log_sigma_uv - 0.04,
            "log_sigma_band_i_err": np.full(n_agn, 0.04),
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


def test_alpha_ox_cosmology_uses_equal_weight_posterior_medians(monkeypatch):
    captured = {}
    source = pd.DataFrame({"object_id": ["agn_001"]})

    def fake_compute_alpha_ox(df, *, cosmology):
        captured["cosmology"] = cosmology
        return df.assign(alphaOX=1.23)

    monkeypatch.setattr(hubble_fit, "compute_alpha_ox", fake_compute_alpha_ox)
    out = hubble_fit._compute_alpha_ox_from_posterior_median(
        source,
        np.array(
            [
                [60.0, 0.10],
                [72.0, 0.30],
                [90.0, 0.80],
            ]
        ),
        ["H0", "Om0"],
        cosmo_model="FlatLambdaCDM",
        z_pivot=1.5,
    )

    assert captured["cosmology"].H0.value == pytest.approx(72.0)
    assert captured["cosmology"].Om0 == pytest.approx(0.30)
    assert out.loc[0, "alphaOX"] == pytest.approx(1.23)


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


def test_log_likelihood_only_agn_skips_pantheon_and_sn_parameter(fake_data):
    df_agn, _ = fake_data
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_agn=True)
    assert "M0_sn" not in model_labels
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)

    agn_fields = hubble_model.agn_model_req_obs + hubble_model.agn_model_req_errs
    agn_fields += ("apparent_mag_2500", "apparent_mag_2500_err", "z", "z_err", "object_id")
    agn_data = {col: df_agn[col].to_numpy() for col in agn_fields}

    logl, blob = hubble_likelihood.log_likelihood(
        theta,
        agn_data=agn_data,
        pantheon_data={},
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM",
        completeness_params=None,
        z_pivot_agn=hubble_fit.z_pivot_agn,
        agn_calibrators_data=None,
        only_agn=True,
        use_full_cov=False,
    )

    assert np.isfinite(logl)
    assert blob.shape == (3, len(df_agn))


def test_flatw0wa_early_de_guard_is_opt_in(fake_data):
    df_agn, df_pantheon = fake_data
    priors, model_labels, _ = hubble_model.get_model_params("Flatw0waCDM", only_sna=False)
    params = {key: (priors[key][0] + priors[key][1]) / 2.0 for key in model_labels}
    params["w0"] = -0.2
    params["wa"] = 0.3
    theta = np.array([params[key] for key in model_labels], dtype=float)

    agn_fields = hubble_model.agn_model_req_obs + hubble_model.agn_model_req_errs
    agn_fields += ("apparent_mag_2500", "apparent_mag_2500_err", "z", "z_err", "object_id")
    agn_data = {col: df_agn[col].to_numpy() for col in agn_fields}
    pantheon_data = {col: df_pantheon[col].to_numpy() for col in df_pantheon.columns}

    logl_default, blob_default = hubble_likelihood.log_likelihood(
        theta,
        agn_data=agn_data,
        pantheon_data=pantheon_data,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="Flatw0waCDM",
        completeness_params=None,
        z_pivot_agn=hubble_fit.z_pivot_agn,
        agn_calibrators_data=None,
        only_sna=False,
        use_full_cov=False,
    )
    logl_guarded, blob_guarded = hubble_likelihood.log_likelihood(
        theta,
        agn_data=agn_data,
        pantheon_data=pantheon_data,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="Flatw0waCDM",
        completeness_params=None,
        z_pivot_agn=hubble_fit.z_pivot_agn,
        agn_calibrators_data=None,
        early_de_guard=True,
        only_sna=False,
        use_full_cov=False,
    )

    assert params["w0"] + params["wa"] >= 0.0
    assert np.isfinite(logl_default)
    assert blob_default.shape == (3, len(df_agn))
    assert logl_guarded == -np.inf
    np.testing.assert_array_equal(blob_guarded, np.zeros((3, len(df_agn))))


def test_compute_agn_likelihood_space_reduced_chi2_uses_only_agn_relevant_dof(fake_data):
    df_agn, _ = fake_data
    priors, model_labels, _ = hubble_model.get_model_params(
        "Flatw0waCDM",
        only_sna=False,
        use_alpha_lambda_term=True,
        use_eta_sigma_term=True,
        use_redshift_log_f_term=True,
    )
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    flat_samples = np.tile(theta[None, :], (4, 1))

    chi2_red, meta = hubble_fit.compute_agn_likelihood_space_reduced_chi2(
        flat_samples,
        model_labels,
        df_agn,
        "Flatw0waCDM",
        z_pivot_agn=hubble_fit.z_pivot_agn,
        use_alpha_lambda_term=True,
        use_eta_sigma_term=True,
        use_redshift_log_f_term=True,
    )

    expected_labels = {
        "M0_agn",
        "alpha_agn",
        "beta_agn",
        hubble_model.AGN_ALPHA_LAMBDA_PARAM,
        hubble_model.AGN_ETA_SIGMA_PARAM,
        "log_f",
        hubble_model.AGN_LOGF_Z_PARAM,
        "H0",
        "Om0",
        "w0",
        "wa",
    }

    assert np.isfinite(chi2_red)
    assert set(
        hubble_fit._agn_likelihood_param_labels(
            model_labels,
            "Flatw0waCDM",
            use_alpha_lambda_term=True,
            use_eta_sigma_term=True,
            use_redshift_log_f_term=True,
        )
    ) == expected_labels
    assert "M0_sn" in model_labels
    assert "M0_sn" not in expected_labels
    assert meta["n_params"] == len(expected_labels)
    assert meta["dof"] == len(df_agn) - len(expected_labels)


def test_table_debias_requires_direct_per_object_values(fake_data):
    df_agn, _ = fake_data
    expected = np.linspace(-0.2, 0.1, len(df_agn))

    actual = hubble_fit._resolve_table_debias_values_for_frame(
        df_agn,
        dmi_values=expected,
    )

    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="expected"):
        hubble_fit._resolve_table_debias_values_for_frame(
            df_agn,
            dmi_values=expected[:-1],
        )


def test_compute_direct_full_sample_completeness_summaries_freezes_fit_pivots(fake_data):
    df_agn, df_pantheon = fake_data
    df_fit = df_agn.iloc[:3].copy()
    df_plot = pd.concat(
        [
            df_fit,
            df_fit.iloc[[0]].assign(
                object_id="agn_outside",
                z=3.4,
                log_sigma_uv=0.8,
                log_tau_uv_rf=4.2,
                apparent_mag_2500=22.1,
            ),
        ],
        ignore_index=True,
    )

    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    params = {key: (priors[key][0] + priors[key][1]) / 2.0 for key in model_labels}
    params["alpha_agn"] = -2.0
    params["beta_agn"] = -1.0
    theta = np.array([params[key] for key in model_labels], dtype=float)
    flat_samples = np.tile(theta[None, :], (5, 1))

    class SimpleCompleteness:
        mode = "2d"

        def __call__(self, m_grid, z):
            m_grid = np.asarray(m_grid, dtype=float)
            z = np.asarray(z, dtype=float)
            return 1.0 / (1.0 + np.exp((m_grid - (22.2 + 0.15 * z)) / 0.2))

    completeness_params = (
        SimpleCompleteness(),
        np.linspace(18.0, 24.5, 96),
    )

    _, _, fit_pivots = hubble_model.agn_model_pack_obs(df_fit)
    _, _, plot_pivots = hubble_model.agn_model_pack_obs(df_plot)
    _, fit_blob = hubble_likelihood.log_likelihood(
        theta,
        agn_data=df_fit,
        pantheon_data=df_pantheon,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM",
        completeness_params=completeness_params,
        z_pivot_agn=hubble_fit.z_pivot_agn,
        agn_calibrators_data=None,
        agn_pivot_arr=fit_pivots,
        only_sna=False,
        use_full_cov=False,
    )
    _, naive_plot_blob = hubble_likelihood.log_likelihood(
        theta,
        agn_data=df_plot,
        pantheon_data=df_pantheon,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM",
        completeness_params=completeness_params,
        z_pivot_agn=hubble_fit.z_pivot_agn,
        agn_calibrators_data=None,
        only_sna=False,
        use_full_cov=False,
    )

    (
        dmi_full_direct,
        dmi_sigma_full_direct,
        sigma_sel_full_direct,
    ) = hubble_fit._compute_direct_full_sample_completeness_summaries(
        flat_samples,
        df_agn_fit_selection=df_fit,
        df_agn_plot_sample=df_plot,
        df_pantheon=df_pantheon,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM",
        completeness_params=completeness_params,
        z_pivot_agn=hubble_fit.z_pivot_agn,
        use_full_cov=False,
        disable_ceph_dist_calibration=False,
        use_planck_h0_prior=False,
        use_planck_om_prior=False,
        use_alpha_lambda_term=False,
        use_eta_sigma_term=False,
        use_redshift_log_f_term=False,
    )

    np.testing.assert_allclose(dmi_full_direct[:-1], fit_blob[1], atol=1e-10)
    np.testing.assert_allclose(sigma_sel_full_direct[:-1], fit_blob[2], atol=1e-10)
    np.testing.assert_allclose(dmi_sigma_full_direct, 0.0, atol=1e-12)
    assert not np.allclose(plot_pivots, fit_pivots, atol=1e-12)
    assert np.isfinite(dmi_full_direct[-1])
    assert np.isfinite(sigma_sel_full_direct[-1])
    assert not np.allclose(naive_plot_blob[1][:-1], fit_blob[1], atol=1e-10)


def test_run_single_skip_plots_smoke(fake_data, monkeypatch, tmp_path):
    df_agn, df_pantheon = fake_data
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    flat_samples = np.tile(theta[None, :], (8, 1))
    dmi_posterior_median = np.zeros(len(df_agn))
    dmi_posterior_sigma = np.full(len(df_agn), 0.05)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hubble_fit, "get_qvc_result_dir", lambda: Path.cwd() / "results")
    monkeypatch.setattr(hubble_fit, "plot_redshift_histograms", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_delta_m_flux_recal_vs_redshift", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "report_pivots", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "display_results_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "compute_age_universe_with_error", lambda *args, **kwargs: (13.8, 0.1))
    monkeypatch.setattr(
        hubble_fit,
        "run_mcmc_pipeline",
        lambda *args, **kwargs: (
            _write_fake_checkpoint(
                kwargs["checkpoint_file_override"],
                flat_samples,
                dmi_posterior_median,
                dmi_posterior_sigma,
                logz=-50.0,
                logzerr=0.2,
            ),
            (
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
        )[1],
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
        speed="fastest",
        z_range=(0.44, 3.16),
        skip_plots=True,
        disable_sigma_clip_pass=True,
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


def test_run_single_threads_direct_full_sample_debias_arrays_to_plots(monkeypatch, tmp_path):
    df_agn = _make_fake_agn_sample(n_agn=5)
    df_agn = pd.concat(
        [
            df_agn,
            df_agn.iloc[[0]].assign(object_id="agn_outside", z=3.5),
        ],
        ignore_index=True,
    )
    df_pantheon = _make_fake_pantheon_sample()
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    flat_samples = np.tile(theta[None, :], (8, 1))

    direct_dmi = np.linspace(0.01, 0.06, len(df_agn))
    direct_sigma = np.full(len(df_agn), 0.02)
    direct_sigma_sel = np.full(len(df_agn), 0.07)

    hubble_calls = []
    l2500_calls = []
    m2500_calls = []
    blr_calls = []
    full_residual_calls = []
    full_residual_rz_calls = []

    monkeypatch.chdir(tmp_path)
    _patch_run_single_plot_stack(monkeypatch)

    def fake_run_mcmc_pipeline(df_agn_arg, *args, **kwargs):
        n = len(df_agn_arg)
        _write_fake_checkpoint(
            kwargs["checkpoint_file_override"],
            flat_samples,
            np.zeros(n),
            np.full(n, 0.05),
            logz=-21.0,
            logzerr=0.2,
        )
        return (
            flat_samples,
            model_labels,
            lambda pts: np.zeros(len(np.atleast_2d(pts))),
            lambda pts: np.full(len(np.atleast_2d(pts)), 0.09),
            -21.0,
            0.2,
            np.zeros(n),
            np.full(n, 0.05),
            np.full(n, 0.09),
        )

    def fake_plot_hubble(*args, **kwargs):
        hubble_calls.append(kwargs)
        n = len(args[1])
        return (
            np.zeros(n, dtype=float),
            np.ones(n, dtype=float),
            np.full(n, 44.0),
            np.full(n, 0.1),
            np.full(n, 0.2),
        )

    def fake_plot_predicted_L2500_vs_sigmahat(*args, **kwargs):
        l2500_calls.append(kwargs)
        n = len(args[1])
        return np.zeros(n, dtype=float), np.ones(n, dtype=float)

    def fake_plot_predicted_vs_actual_M2500(*args, **kwargs):
        m2500_calls.append(kwargs)
        n = len(args[1])
        return np.zeros(n, dtype=float), np.ones(n, dtype=float), None, None

    def fake_plot_blr_line_lags_vs_l2500(*args, **kwargs):
        blr_calls.append(kwargs)

    def fake_plot_full_residuals(*args, **kwargs):
        full_residual_calls.append(kwargs)

    def fake_plot_full_residuals_rz(*args, **kwargs):
        full_residual_rz_calls.append(kwargs)

    monkeypatch.setattr(hubble_fit, "run_mcmc_pipeline", fake_run_mcmc_pipeline)
    monkeypatch.setattr(
        hubble_fit,
        "_compute_direct_full_sample_completeness_summaries",
        lambda *args, **kwargs: (
            direct_dmi.copy(),
            direct_sigma.copy(),
            direct_sigma_sel.copy(),
        ),
    )
    monkeypatch.setattr(hubble_fit, "plot_hubble", fake_plot_hubble)
    monkeypatch.setattr(hubble_fit, "plot_predicted_L2500_vs_sigmahat", fake_plot_predicted_L2500_vs_sigmahat)
    monkeypatch.setattr(hubble_fit, "plot_predicted_vs_actual_M2500", fake_plot_predicted_vs_actual_M2500)
    monkeypatch.setattr(hubble_fit, "plot_blr_line_lags_vs_l2500", fake_plot_blr_line_lags_vs_l2500)
    monkeypatch.setattr(hubble_fit, "plot_full_residuals", fake_plot_full_residuals)
    monkeypatch.setattr(hubble_fit, "plot_full_residuals_rz", fake_plot_full_residuals_rz)

    hubble_fit.run_single(
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
        speed="fastest",
        z_range=(0.44, 3.16),
        disable_sigma_clip_pass=True,
        prefix="unit",
    )

    debiased_hubble_call = next(call for call in hubble_calls if call.get("debias"))
    np.testing.assert_allclose(debiased_hubble_call["dmi_values"], direct_dmi)
    np.testing.assert_allclose(debiased_hubble_call["dmi_sigma"], direct_sigma)
    np.testing.assert_allclose(debiased_hubble_call["dmi_selection_sigma"], direct_sigma_sel)

    debiased_l2500_call = next(call for call in l2500_calls if call.get("debias"))
    np.testing.assert_allclose(debiased_l2500_call["dmi_values"], direct_dmi)
    np.testing.assert_allclose(debiased_l2500_call["dmi_selection_sigma"], direct_sigma_sel)

    debiased_m2500_call = next(call for call in m2500_calls if call.get("debias"))
    np.testing.assert_allclose(debiased_m2500_call["dmi_values"], direct_dmi)
    np.testing.assert_allclose(debiased_m2500_call["dmi_selection_sigma"], direct_sigma_sel)

    np.testing.assert_allclose(blr_calls[0]["dmi_values"], direct_dmi)
    assert all(np.allclose(call["dmi_values"], direct_dmi) for call in full_residual_calls)
    np.testing.assert_allclose(full_residual_rz_calls[0]["dmi_values"], direct_dmi)


def test_run_single_only_sna_smoke(fake_data, monkeypatch, tmp_path):
    df_agn, df_pantheon = fake_data
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=True)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    flat_samples = np.tile(theta[None, :], (8, 1))
    dmi_posterior_median = np.zeros(len(df_agn))
    dmi_posterior_sigma = np.full(len(df_agn), 0.05)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hubble_fit, "get_qvc_result_dir", lambda: Path.cwd() / "results")
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
        speed="fastest",
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


def test_run_single_only_agn_keeps_agn_hubble_plots(fake_data, monkeypatch, tmp_path):
    df_agn, df_pantheon = fake_data
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_agn=True)
    assert "M0_sn" not in model_labels
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    flat_samples = np.tile(theta[None, :], (8, 1))
    plot_hubble_calls = []
    plot_cosmo_corner_calls = []

    monkeypatch.chdir(tmp_path)
    _patch_run_single_plot_stack(monkeypatch)
    monkeypatch.setattr(
        hubble_fit,
        "run_mcmc_pipeline",
        lambda df_agn_arg, *args, **kwargs: (
            flat_samples,
            model_labels,
            lambda pts: np.zeros(len(np.atleast_2d(pts))),
            None,
            -24.0,
            0.12,
            np.zeros(len(df_agn_arg)),
            np.full(len(df_agn_arg), 0.05),
            None,
        ),
    )

    def fake_plot_hubble(*args, **kwargs):
        plot_hubble_calls.append(kwargs)
        n = len(args[1])
        return (
            np.zeros(n, dtype=float),
            np.ones(n, dtype=float),
            np.full(n, 44.0),
            np.full(n, 0.1),
            np.full(n, 0.2),
        )

    monkeypatch.setattr(hubble_fit, "plot_hubble", fake_plot_hubble)
    monkeypatch.setattr(
        hubble_fit,
        "plot_cosmo_corner",
        lambda *args, **kwargs: plot_cosmo_corner_calls.append(kwargs),
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
        only_agn=True,
        speed="fastest",
        z_range=(0.44, 3.16),
        disable_sigma_clip_pass=True,
        skip_plots=False,
        prefix="unit",
    )

    samples_out, labels_out, _, logz, logzerr, residuals, age, age_err = result
    assert samples_out.shape == flat_samples.shape
    assert labels_out == model_labels
    assert logz == -24.0
    assert logzerr == 0.12
    assert residuals is not None
    assert age == 13.8
    assert age_err == 0.1
    assert plot_hubble_calls
    assert all(call.get("only_agn") is True for call in plot_hubble_calls)
    assert plot_cosmo_corner_calls
    assert all(call.get("only_agn") is True for call in plot_cosmo_corner_calls)


@pytest.mark.parametrize("include_alpha_beta", [False, True])
def test_plot_cosmo_corner_only_agn_legend_label_and_font_size(
    monkeypatch,
    tmp_path,
    include_alpha_beta,
):
    rng = np.random.default_rng(987)
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_agn=True)
    center = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    samples = np.tile(center[None, :], (80, 1))
    samples += rng.normal(
        0.0,
        np.array([0.05, 0.08, 0.06, 0.03, 0.2, 0.03], dtype=float),
        size=samples.shape,
    )
    captured = {}

    def fake_save_figure(fig, *args, **kwargs):
        captured["fig"] = fig

    monkeypatch.setattr(hubble_plotting, "_save_figure", fake_save_figure)

    hubble_plotting.plot_cosmo_corner(
        None,
        samples,
        "FlatLambdaCDM",
        z_pivot_sna=0.01,
        z_pivot_agn=1.0,
        plot_path=str(tmp_path),
        show=False,
        smooth=24,
        only_agn=True,
        include_alpha_beta=include_alpha_beta,
    )

    fig = captured["fig"]
    legend_labels = [
        text.get_text()
        for legend in fig.legends
        for text in legend.get_texts()
    ]
    assert "AGN" in legend_labels
    assert "SN Ia + AGN" not in legend_labels
    assert {
        text.get_fontsize()
        for legend in fig.legends
        for text in legend.get_texts()
    } == {hubble_plotting._COSMO_CORNER_LEGEND_FONTSIZE}
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_hubble_debiased_returns_clipping_sigma_and_writes_distinct_diagnostics(monkeypatch, tmp_path):
    df_agn = _make_fake_agn_sample(n_agn=6).copy()
    df_agn["wrms"] = np.linspace(0.1, 0.2, len(df_agn))
    df_pantheon = _make_fake_pantheon_sample(n_sne=6).copy()
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    df_pantheon["MU_SH0ES"] = cosmo.distmod(df_pantheon["zHD"].to_numpy(dtype=float)).value
    df_pantheon["biasCor_m_b"] = np.zeros(len(df_pantheon), dtype=float)

    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    flat_samples = np.tile(theta[None, :], (6, 1))

    monkeypatch.setattr(hubble_plotting, "_save_figure", lambda fig, path, **kwargs: path)

    residuals, clipping_sigma, _, _, mu_pred_std_with_scatter = hubble_plotting.plot_hubble(
        flat_samples,
        df_agn,
        df_pantheon,
        cosmo_model="FlatLambdaCDM",
        z_pivot_agn=hubble_fit.z_pivot_agn,
        plot_path=str(tmp_path),
        show=False,
        debias=True,
        dm_interp=None,
        dmi_values=np.zeros(len(df_agn), dtype=float),
        dmi_selection_sigma=np.full(len(df_agn), 7.0, dtype=float),
        residuals_csv_filename="residuals.csv",
    )

    assert residuals.shape == (len(df_agn),)
    assert clipping_sigma.shape == (len(df_agn),)
    assert mu_pred_std_with_scatter.shape == (len(df_agn),)
    np.testing.assert_allclose(clipping_sigma, mu_pred_std_with_scatter)
    assert not np.allclose(clipping_sigma, np.full(len(df_agn), 7.0, dtype=float))

    residuals_df = pd.read_csv(tmp_path / "residuals.csv")
    for col in ("mu_pred_std", "mu_pred_std_with_scatter", "clipping_sigma", "chi2_sigma", "sigma_sel", "mu_zscore"):
        assert col in residuals_df.columns
    np.testing.assert_allclose(residuals_df["sigma_sel"].to_numpy(dtype=float), 7.0)
    np.testing.assert_allclose(
        residuals_df["chi2_sigma"].to_numpy(dtype=float),
        residuals_df["mu_pred_std"].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(
        residuals_df["mu_zscore"].to_numpy(dtype=float),
        np.abs(residuals_df["residuals"].to_numpy(dtype=float)) / residuals_df["clipping_sigma"].to_numpy(dtype=float),
    )


def test_plot_predicted_vs_actual_m2500_marks_out_of_range_objects(monkeypatch, tmp_path):
    from matplotlib.axes import Axes
    import matplotlib.pyplot as plt

    df_agn = _make_fake_agn_sample(n_agn=3).copy()
    # The lower boundary is in range; the other objects are out of range, with
    # the final one clipped.
    df_agn["z"] = [0.44, 3.20, 3.25]
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    flat_samples = np.tile(theta[None, :], (4, 1))

    scatter_calls = []
    errorbar_calls = []
    captured = {}
    original_scatter = Axes.scatter
    original_errorbar = Axes.errorbar

    def capture_scatter(self, *args, **kwargs):
        scatter_calls.append(kwargs.copy())
        return original_scatter(self, *args, **kwargs)

    def capture_errorbar(self, *args, **kwargs):
        errorbar_calls.append(kwargs.copy())
        return original_errorbar(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", capture_scatter)
    monkeypatch.setattr(Axes, "errorbar", capture_errorbar)
    monkeypatch.setattr(
        hubble_plotting,
        "_save_figure",
        lambda fig, *args, **kwargs: captured.setdefault("fig", fig),
    )

    hubble_plotting.plot_predicted_vs_actual_M2500(
        flat_samples,
        df_agn,
        "FlatLambdaCDM",
        z_pivot_agn=hubble_fit.z_pivot_agn,
        plot_path=str(tmp_path),
        show=False,
        z_range=(0.44, 3.16),
        clipped_mask=np.array([False, False, True]),
        completeness=False,
        show_sigma_band=False,
        show_cosmo_uncertainty_band=False,
    )

    assert any(
        call.get("marker") == "D"
        and np.allclose(
            hubble_plotting.mpl.colors.to_rgba(call["facecolors"]),
            hubble_plotting._OUT_OF_RANGE_AGN_MARKER_COLOR,
        )
        for call in scatter_calls
    )
    assert any(
        call.get("marker") == "D" and call.get("facecolors") == "tab:green"
        for call in scatter_calls
    )
    assert any(
        call.get("facecolors") == "k" and call.get("marker") is None
        for call in scatter_calls
    )
    assert any(
        np.allclose(
            hubble_plotting.mpl.colors.to_rgba(call["ecolor"]),
            hubble_plotting._OUT_OF_RANGE_AGN_ERROR_COLOR,
        )
        for call in errorbar_calls
    )
    plt.close(captured["fig"])


def test_plot_hubble_residual_chi2_annotation_includes_high_z(monkeypatch, tmp_path):
    from matplotlib.axes import Axes

    df_agn = _make_fake_agn_sample(n_agn=6).copy()
    df_agn["wrms"] = np.linspace(0.1, 0.2, len(df_agn))
    df_pantheon = _make_fake_pantheon_sample(n_sne=6).copy()
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    df_pantheon["MU_SH0ES"] = cosmo.distmod(df_pantheon["zHD"].to_numpy(dtype=float)).value
    df_pantheon["biasCor_m_b"] = np.zeros(len(df_pantheon), dtype=float)

    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    flat_samples = np.tile(theta[None, :], (6, 1))

    text_calls = []
    original_text = Axes.text

    def capture_text(self, x, y, s, *args, **kwargs):
        if r"\chi^2_\nu" in str(s):
            text_calls.append(
                {
                    "x": x,
                    "y": y,
                    "text": str(s),
                    "ha": kwargs.get("ha"),
                    "va": kwargs.get("va"),
                }
            )
        return original_text(self, x, y, s, *args, **kwargs)

    monkeypatch.setattr(Axes, "text", capture_text)
    monkeypatch.setattr(hubble_plotting, "_save_figure", lambda fig, path, **kwargs: path)

    hubble_plotting.plot_hubble(
        flat_samples,
        df_agn,
        df_pantheon,
        cosmo_model="FlatLambdaCDM",
        z_pivot_agn=hubble_fit.z_pivot_agn,
        plot_path=str(tmp_path),
        show=False,
        debias=True,
        dm_interp=None,
        dmi_values=np.zeros(len(df_agn), dtype=float),
        agn_likelihood_space_chi2=1.23,
        agn_likelihood_space_chi2_zgt1=2.34,
        residuals_csv_filename=None,
    )

    annotation = next(
        call for call in text_calls
        if r"$\chi^2_\nu(0.44<z<3.16) = 1.23$" in call["text"]
    )
    assert rf"$\chi^2_\nu(1.00<z<3.16) = 2.34$" in annotation["text"]
    assert "\n" in annotation["text"]
    assert annotation["x"] == 0.02
    assert annotation["y"] == 0.08
    assert annotation["ha"] == "left"
    assert annotation["va"] == "bottom"

    text_calls.clear()
    hubble_plotting.plot_hubble(
        flat_samples,
        df_agn,
        df_pantheon,
        cosmo_model="FlatLambdaCDM",
        z_pivot_agn=hubble_fit.z_pivot_agn,
        plot_path=str(tmp_path),
        show=False,
        debias=False,
        z_range=(0.44, 0.9),
        residuals_csv_filename=None,
    )

    chi2_text = next(call["text"] for call in text_calls if r"\chi^2_\nu" in call["text"])
    assert r"$\chi^2_\nu(0.44<z<0.90)" in chi2_text
    assert "(1<z<" not in chi2_text


def test_run_single_two_pass_sigma_clip_uses_plot_hubble_clipping_sigma(monkeypatch, tmp_path):
    df_agn = _make_fake_agn_sample(n_agn=4)
    df_pantheon = _make_fake_pantheon_sample()
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    pipeline_calls = []

    monkeypatch.chdir(tmp_path)
    _patch_run_single_plot_stack(monkeypatch)

    def fake_run_mcmc_pipeline(df_agn, *args, **kwargs):
        pipeline_calls.append(
            {
                "object_ids": df_agn["object_id"].tolist(),
                "completeness_object_ids": kwargs["df_agn_completeness"]["object_id"].tolist(),
                "warm_start_flat_samples": kwargs.get("warm_start_flat_samples"),
                "logZ_is_approximate": kwargs.get("logZ_is_approximate"),
            }
        )
        flat_samples = np.tile(theta[None, :], (8, 1))
        n = len(df_agn)
        _write_fake_checkpoint(
            kwargs["checkpoint_file_override"],
            flat_samples,
            np.zeros(n),
            np.full(n, 0.05),
            logz=-70.0 - len(pipeline_calls),
            logzerr=0.2,
        )
        return (
            flat_samples,
            model_labels,
            lambda pts: np.zeros(len(np.atleast_2d(pts))),
            None,
            -70.0 - len(pipeline_calls),
            0.2,
            np.zeros(n),
            np.full(n, 0.05),
            None,
        )

    def fake_plot_hubble(*args, **kwargs):
        n = len(args[1])
        if len(pipeline_calls) == 1 and kwargs.get("filename") == "hubble_diagram_pass1_full_sample_debiased.pdf":
            residuals = np.array([4.0, 4.0, 0.1, 0.1], dtype=float)
            clipping_sigma = np.array([1.0, 10.0, 1.0, 1.0], dtype=float)
            mu_pred_std_with_scatter = np.array([10.0, 1.0, 1.0, 1.0], dtype=float)
            return (
                residuals,
                clipping_sigma,
                np.full(n, 44.0),
                np.full(n, 0.1),
                mu_pred_std_with_scatter,
            )
        return (
            np.zeros(n, dtype=float),
            np.ones(n, dtype=float),
            np.full(n, 44.0),
            np.full(n, 0.1),
            np.full(n, 0.2),
        )

    monkeypatch.setattr(hubble_fit, "run_mcmc_pipeline", fake_run_mcmc_pipeline)
    monkeypatch.setattr(hubble_fit, "plot_hubble", fake_plot_hubble)

    hubble_fit.run_single(
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
        speed="fastest",
        z_range=(0.44, 3.16),
        disable_sigma_clip_pass=False,
        sigma_clip_threshold=3.0,
        prefix="unit",
    )

    assert len(pipeline_calls) == 2
    assert pipeline_calls[0]["object_ids"] == ["agn_000", "agn_001", "agn_002", "agn_003"]
    assert pipeline_calls[1]["object_ids"] == ["agn_001", "agn_002", "agn_003"]
    assert pipeline_calls[0]["warm_start_flat_samples"] is None
    assert pipeline_calls[1]["warm_start_flat_samples"] is not None


def test_run_single_only_agn_two_pass_passes_only_agn_to_pass1_clipped_plot(monkeypatch, tmp_path):
    df_agn = _make_fake_agn_sample(n_agn=4)
    df_pantheon = _make_fake_pantheon_sample()
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_agn=True)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    pipeline_calls = []
    plot_calls = []

    monkeypatch.chdir(tmp_path)
    _patch_run_single_plot_stack(monkeypatch)

    def fake_run_mcmc_pipeline(df_agn_arg, *args, **kwargs):
        pipeline_calls.append(
            {
                "object_ids": df_agn_arg["object_id"].tolist(),
                "only_agn": kwargs.get("only_agn"),
            }
        )
        flat_samples = np.tile(theta[None, :], (8, 1))
        n = len(df_agn_arg)
        _write_fake_checkpoint(
            kwargs["checkpoint_file_override"],
            flat_samples,
            np.zeros(n),
            np.full(n, 0.05),
            logz=-80.0 - len(pipeline_calls),
            logzerr=0.2,
        )
        return (
            flat_samples,
            model_labels,
            lambda pts: np.zeros(len(np.atleast_2d(pts))),
            None,
            -80.0 - len(pipeline_calls),
            0.2,
            np.zeros(n),
            np.full(n, 0.05),
            None,
        )

    def fake_plot_hubble(*args, **kwargs):
        plot_calls.append(kwargs)
        n = len(args[1])
        if kwargs.get("filename") == "hubble_diagram_pass1_full_sample_debiased.pdf":
            return (
                np.array([4.0, 0.1, 0.1, 0.1], dtype=float),
                np.ones(n, dtype=float),
                np.full(n, 44.0),
                np.full(n, 0.1),
                np.full(n, 0.2),
            )
        return (
            np.zeros(n, dtype=float),
            np.ones(n, dtype=float),
            np.full(n, 44.0),
            np.full(n, 0.1),
            np.full(n, 0.2),
        )

    monkeypatch.setattr(hubble_fit, "run_mcmc_pipeline", fake_run_mcmc_pipeline)
    monkeypatch.setattr(hubble_fit, "plot_hubble", fake_plot_hubble)

    hubble_fit.run_single(
        df_agn=df_agn,
        df_agn_all=df_agn.copy(),
        df_pantheon=df_pantheon,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM",
        completeness=False,
        use_full_cov=False,
        only_agn=True,
        speed="fastest",
        z_range=(0.44, 3.16),
        disable_sigma_clip_pass=False,
        sigma_clip_threshold=3.0,
        prefix="unit",
    )

    assert len(pipeline_calls) == 2
    assert all(call["only_agn"] is True for call in pipeline_calls)
    pass1_plot_calls = [
        call for call in plot_calls
        if call.get("filename") in {
            "hubble_diagram_pass1_full_sample_debiased.pdf",
            "hubble_diagram_pass1_full_sample_clipped_debiased.pdf",
        }
    ]
    assert {call["filename"] for call in pass1_plot_calls} == {
        "hubble_diagram_pass1_full_sample_debiased.pdf",
        "hubble_diagram_pass1_full_sample_clipped_debiased.pdf",
    }
    assert all(call.get("only_agn") is True for call in pass1_plot_calls)


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
            _write_fake_checkpoint(
                kwargs["checkpoint_file_override"],
                flat_samples,
                dmi_posterior_median,
                dmi_posterior_sigma,
                logz=-50.0,
                logzerr=0.2,
            ),
            (
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
        )[1],
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
        speed="fastest",
        z_range=(0.44, 3.16),
        skip_plots=False,
        prefix="unit",
    )

    assert len(latex_calls) == 1
    assert len(csv_calls) == 1
    csv_df_arg = csv_calls[0][0][0]
    assert csv_df_arg["object_id"].tolist() == df_agn["object_id"].tolist()


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
        speed="fastest",
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
        speed="fastest",
        z_range=(0.44, 3.16),
        skip_plots=True,
        disable_sigma_clip_pass=True,
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
        speed="fastest",
        z_range=(0.44, 3.16),
        skip_plots=False,
        compare_sigma_only=True,
        disable_sigma_clip_pass=True,
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


def test_run_single_minimal_plots_keeps_only_debiased_hubble_plot(monkeypatch, tmp_path):
    df_agn = _make_fake_agn_sample(n_agn=6)
    df_pantheon = _make_fake_pantheon_sample()
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    flat_samples = np.tile(theta[None, :], (8, 1))
    pipeline_kwargs = []
    hubble_calls = []
    expensive_calls = []

    monkeypatch.chdir(tmp_path)
    _patch_run_single_plot_stack(monkeypatch)
    for name in (
        "plot_sigma_uv_mpred_correction",
        "plot_predicted_L2500_vs_sigmahat",
        "plot_blr_diagnostics_summary",
        "plot_completeness_diagnostics",
        "plot_cosmo_corner",
    ):
        monkeypatch.setattr(
            hubble_fit,
            name,
            lambda *args, _name=name, **kwargs: expensive_calls.append(_name),
        )

    def fake_run_mcmc_pipeline(df_agn_arg, *args, **kwargs):
        pipeline_kwargs.append(kwargs)
        return (
            flat_samples,
            model_labels,
            lambda pts: np.zeros(len(np.atleast_2d(pts))),
            None,
            -20.0,
            0.1,
            np.zeros(len(df_agn_arg)),
            np.full(len(df_agn_arg), 0.05),
            None,
        )

    def fake_plot_hubble(*args, **kwargs):
        hubble_calls.append(kwargs)
        n = len(args[1])
        return (
            np.arange(n, dtype=float),
            np.ones(n, dtype=float),
            np.full(n, 44.0),
            np.full(n, 0.1),
            np.full(n, 0.2),
        )

    monkeypatch.setattr(hubble_fit, "run_mcmc_pipeline", fake_run_mcmc_pipeline)
    monkeypatch.setattr(hubble_fit, "plot_hubble", fake_plot_hubble)
    monkeypatch.setattr(
        hubble_fit,
        "compute_agn_likelihood_space_reduced_chi2",
        lambda *args, **kwargs: (1.0, len(args[2])),
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
        speed="fastest",
        z_range=(0.44, 3.16),
        disable_sigma_clip_pass=True,
        minimal_plots=True,
        prefix="unit",
    )

    assert len(pipeline_kwargs) == 1
    assert pipeline_kwargs[0]["minimal_plots"] is True
    assert len(hubble_calls) == 1
    assert hubble_calls[0]["debias"] is True
    assert hubble_calls[0]["residuals_csv_filename"] == "hubble_plot_residuals.csv"
    assert "filename" not in hubble_calls[0]
    assert expensive_calls == []
    assert result[5].tolist() == list(range(len(df_agn)))


@pytest.mark.parametrize(
    ("conflicting_flag", "message"),
    [
        ("skip_plots", "--skip_plots"),
        ("compare_sigma_only", "--compare_sigma_only"),
        ("only_sna", "--only_sna"),
        ("use_jax", "--use_jax"),
    ],
)
def test_validate_plot_mode_args_rejects_minimal_plot_conflicts(conflicting_flag, message):
    args = type(
        "Args",
        (),
        {
            "minimal_plots": True,
            "skip_plots": False,
            "compare_sigma_only": False,
            "only_sna": False,
            "use_jax": False,
        },
    )()
    setattr(args, conflicting_flag, True)

    with pytest.raises(ValueError, match=message):
        hubble_fit.validate_plot_mode_args(args)


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


def test_normalize_resume_by_model_no_resume():
    assert hubble_fit.normalize_resume_by_model(
        False,
        ["FlatLambdaCDM", "FlatwCDM"],
    ) == {"FlatLambdaCDM": False, "FlatwCDM": False}


def test_normalize_resume_by_model_flag_only_uses_defaults():
    assert hubble_fit.normalize_resume_by_model(
        [],
        ["FlatLambdaCDM", "FlatwCDM"],
    ) == {"FlatLambdaCDM": True, "FlatwCDM": True}


def test_normalize_resume_by_model_single_model_single_path():
    assert hubble_fit.normalize_resume_by_model(
        ["flatlambda.h5"],
        ["FlatLambdaCDM"],
    ) == {"FlatLambdaCDM": "flatlambda.h5"}


def test_normalize_resume_by_model_multiple_models_matching_paths():
    assert hubble_fit.normalize_resume_by_model(
        ["flatlambda.h5", "flatw.h5"],
        ["FlatLambdaCDM", "FlatwCDM"],
    ) == {"FlatLambdaCDM": "flatlambda.h5", "FlatwCDM": "flatw.h5"}


@pytest.mark.parametrize(
    "resume_values",
    [
        ["shared.h5"],
        ["flatlambda.h5", "flatw.h5"],
        ["flatlambda.h5", "flatw.h5", "flatw0wa.h5", "extra.h5"],
    ],
)
def test_normalize_resume_by_model_rejects_mismatched_path_count(resume_values):
    models = ["FlatLambdaCDM", "FlatwCDM", "Flatw0waCDM"]

    with pytest.raises(ValueError, match="one-for-one"):
        hubble_fit.normalize_resume_by_model(resume_values, models)


def test_run_all_dispatches_resume_path_per_cosmo_model(monkeypatch, tmp_path):
    calls = []

    def fake_run_single(*args, **kwargs):
        calls.append(
            {
                "cosmo_model": kwargs["cosmo_model"],
                "only_sna": kwargs["only_sna"],
                "resume": kwargs["resume"],
            }
        )
        return (
            np.ones((4, 2), dtype=float),
            ["H0", "Om0"],
            None,
            -10.0,
            0.1,
            None,
            13.8,
            0.1,
        )

    monkeypatch.setattr(hubble_fit, "run_single", fake_run_single)
    monkeypatch.setattr(
        hubble_fit,
        "compare_models_by_log_evidence_all",
        lambda *args, **kwargs: {"delta_logZ": 0.0},
    )
    monkeypatch.setattr(hubble_fit, "extract_cosmo_results_from_samples", lambda *args, **kwargs: {})
    monkeypatch.setattr(hubble_fit, "write_results_tex_variables", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "save_cosmo_results_hdf5", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "get_qvc_result_dir", lambda: tmp_path)

    df_agn = _make_fake_agn_sample(n_agn=4)
    df_pantheon = _make_fake_pantheon_sample(n_sne=4)

    hubble_fit.run_all(
        df_agn,
        df_agn,
        df_pantheon,
        np.eye(len(df_pantheon)),
        np.eye(len(df_pantheon)),
        0.0,
        cosmo_models=["FlatLambdaCDM", "FlatwCDM"],
        resume=["flatlambda.h5", "flatw.h5"],
        speed="fastest",
        compare_sigma_only=True,
        disable_sigma_clip_pass=True,
        prefix=str(tmp_path / "plots"),
    )

    assert calls == [
        {"cosmo_model": "FlatLambdaCDM", "only_sna": False, "resume": "flatlambda.h5"},
        {"cosmo_model": "FlatLambdaCDM", "only_sna": True, "resume": "flatlambda.h5"},
        {"cosmo_model": "FlatwCDM", "only_sna": False, "resume": "flatw.h5"},
        {"cosmo_model": "FlatwCDM", "only_sna": True, "resume": "flatw.h5"},
    ]


def test_resolve_two_pass_resume_checkpoint_prefers_pass2_then_pass1(tmp_path):
    paths = {
        "single": str(tmp_path / "single.h5"),
        "pass1": str(tmp_path / "pass1.h5"),
        "pass2": str(tmp_path / "pass2.h5"),
    }
    Path(paths["pass1"]).touch()
    assert hubble_fit._resolve_two_pass_resume_checkpoint(True, "both", paths) == paths["pass1"]
    Path(paths["pass2"]).touch()
    assert hubble_fit._resolve_two_pass_resume_checkpoint(True, "both", paths) == paths["pass2"]
    assert hubble_fit._resolve_two_pass_resume_checkpoint(True, "pass2", paths) == paths["pass2"]


def test_write_stage_checkpoint_roundtrips_pass1_state(tmp_path):
    checkpoint = tmp_path / "pass1.h5"
    df_agn = _make_fake_agn_sample(n_agn=4)
    df_fit = df_agn[df_agn["z"].between(0.44, 3.16)].copy()
    keep_mask = np.array([True, False, True, True], dtype=bool)
    diagnostics_df = df_agn.copy()
    diagnostics_df["residuals"] = np.array([0.1, 3.2, 0.2, 0.3])
    diagnostics_df["residuals_err"] = np.ones(4)
    diagnostics_df["mu_zscore"] = np.array([0.1, 3.2, 0.2, 0.3])
    diagnostics_df["was_clipped"] = ~keep_mask
    _write_fake_checkpoint(checkpoint, np.ones((3, 1)), np.zeros(len(df_fit)), np.full(len(df_fit), 0.05))

    hubble_fit._write_stage_checkpoint(
        str(checkpoint),
        sigma_clip_pass_stage="pass1",
        sigma_clip_threshold=3.0,
        df_agn_full_sample=df_agn,
        df_agn_fit_selection=df_fit,
        keep_mask_full=keep_mask,
        pass1_diagnostics_df=diagnostics_df,
    )

    loaded = hubble_fit.load_chains(str(checkpoint))
    assert hubble_fit._checkpoint_stage_from_results(loaded) == "pass1"
    extracted = hubble_fit._extract_pass1_state_from_checkpoint(
        loaded,
        str(checkpoint),
        df_agn,
        sigma_clip_threshold=3.0,
    )
    np.testing.assert_array_equal(extracted["keep_mask_full"], keep_mask)
    assert extracted["pass1_diagnostics_df"]["mu_zscore"].tolist() == [0.1, 3.2, 0.2, 0.3]


def test_cosmo_results_hdf5_roundtrips_nested_model_results(tmp_path):
    output_path = tmp_path / "cosmo_results.hdf5"
    models_dict = {
        "FlatLambdaCDM": {
            "H0": 73.2,
            "N": 2000,
            "samples": np.array([0.31, 0.32, 0.33]),
            "missing": None,
        },
        "Flatw0waCDM": {
            "w0": -0.9,
            "wa": -1.2,
            "logZ": 4154.1,
        },
    }

    hubble_utils.save_cosmo_results_hdf5(str(output_path), models_dict)
    loaded = hubble_utils.load_cosmo_results_hdf5(str(output_path))

    assert set(loaded) == set(models_dict)
    assert set(loaded["FlatLambdaCDM"]) == set(models_dict["FlatLambdaCDM"])
    assert set(loaded["Flatw0waCDM"]) == set(models_dict["Flatw0waCDM"])
    assert loaded["FlatLambdaCDM"]["H0"] == pytest.approx(73.2)
    assert int(loaded["FlatLambdaCDM"]["N"]) == 2000
    np.testing.assert_allclose(loaded["FlatLambdaCDM"]["samples"], [0.31, 0.32, 0.33])
    assert np.isnan(loaded["FlatLambdaCDM"]["missing"])
    assert loaded["Flatw0waCDM"]["w0"] == pytest.approx(-0.9)
    assert loaded["Flatw0waCDM"]["wa"] == pytest.approx(-1.2)
    assert loaded["Flatw0waCDM"]["logZ"] == pytest.approx(4154.1)


def test_build_warm_start_live_points_uses_unit_cube_and_blobs():
    priors = {"x": (0.0, 10.0), "y": (-5.0, 5.0)}
    model_labels = ["x", "y"]
    flat_samples = np.array([[1.0, -1.0], [2.0, 0.0], [3.0, 1.0]], dtype=float)
    calls = []

    def fake_loglike(theta, **kwargs):
        theta = np.asarray(theta, dtype=float)
        calls.append(theta)
        return -float(np.sum(theta**2)), np.array([theta[0], theta[1], theta.sum()])

    live_u, live_v, live_logl, live_blobs = hubble_fit.build_warm_start_live_points(
        flat_samples,
        priors=priors,
        model_labels=model_labels,
        nlive=5,
        loglike_func=fake_loglike,
        logl_kwargs={},
        rng_seed=7,
        jitter_scale=1e-4,
    )

    assert live_u.shape == (5, 2)
    assert live_v.shape == (5, 2)
    assert live_logl.shape == (5,)
    assert live_blobs.shape == (5, 3)
    assert len(calls) == 5
    assert np.all(live_u > 0.0)
    assert np.all(live_u < 1.0)
    np.testing.assert_allclose(live_blobs[:, 2], live_v.sum(axis=1))


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
            speed="fastest",
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
            speed="fastest",
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
    expected = result_root / "hubble_posteriors" / "unit" / "posteriors_FlatLambdaCDM_joint_fastest_all_z0p44_3p16_2d.h5"
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
        speed="fastest",
        prefix="unit",
        compare_sigma_only=True,
        completeness_sim_file="dummy_completeness.h5",
    )

    assert result[0].shape == (3, 1)
    assert diagnostics_calls == []
    assert completeness_calls == [False]


def test_run_mcmc_pipeline_uses_explicit_parent_sample_for_completeness_map(monkeypatch, tmp_path):
    df_fit = _make_fake_agn_sample(n_agn=2)
    df_parent = _make_fake_agn_sample(n_agn=4)
    df_pantheon = _make_fake_pantheon_sample()
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    completeness_sample_ids = []

    class FakeResults:
        samples = np.tile(theta[None, :], (6, 1))
        logl = np.full(6, -1.0)
        logwt = np.zeros(6)
        logz = np.array([-3.0])
        logzerr = np.array([0.1])
        blob = np.zeros((6, 3, len(df_fit)))
        blobs = blob

    class FakeSampler:
        def __init__(self, *args, **kwargs):
            pass

        def run_nested(self, *args, **kwargs):
            self.results = FakeResults()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hubble_fit, "get_qvc_result_dir", lambda: Path.cwd() / "results")
    monkeypatch.setattr(hubble_fit, "DynamicNestedSampler", FakeSampler)
    monkeypatch.setattr(hubble_fit, "plot_dynesty", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        hubble_fit,
        "get_completeness_function_2d",
        lambda df_arg, *args, **kwargs: (
            completeness_sample_ids.append(df_arg["object_id"].tolist()),
            (
                np.ones((2, 2)),
                np.array([19.0, 20.0]),
                np.array([0.5, 1.0]),
                0.5,
                0.1,
                0.0,
            ),
        )[1],
    )
    monkeypatch.setattr(
        hubble_fit,
        "log_likelihood",
        lambda theta_arg, **kwargs: (
            -1.0,
            np.zeros((3, len(kwargs["agn_data"]["object_id"]))),
        ),
    )

    result = hubble_fit.run_mcmc_pipeline(
        df_agn=df_fit,
        df_agn_all=df_parent.copy(),
        df_pantheon=df_pantheon,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM",
        completeness=True,
        use_full_cov=False,
        speed="fastest",
        prefix="unit",
        completeness_sim_file="dummy_completeness.h5",
        df_agn_completeness=df_parent,
    )

    assert result[6].shape == (len(df_fit),)
    assert completeness_sample_ids == [df_parent["object_id"].tolist()]


def _patch_run_single_plot_stack(monkeypatch):
    monkeypatch.setattr(hubble_fit, "get_qvc_result_dir", lambda: Path.cwd() / "results")
    monkeypatch.setattr(hubble_fit, "plot_redshift_histograms", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_delta_m_flux_recal_vs_redshift", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "report_pivots", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "display_results_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "compute_age_universe_with_error", lambda *args, **kwargs: (13.8, 0.1))
    monkeypatch.setattr(hubble_fit, "plot_sigma_uv_mpred_correction", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_predicted_L2500_vs_sigmahat", lambda *args, **kwargs: (np.zeros(len(args[1])), np.ones(len(args[1]))))
    monkeypatch.setattr(hubble_fit, "plot_L2500_vs_sigma_tau_separate", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_catalog_quantity_vs_sigma_tau_separate", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_blr_line_lags_vs_l2500", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_blr_diagnostics_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_hubble_residual_normality", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_hubble_residual_tail_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_predicted_vs_actual_M2500", lambda *args, **kwargs: (np.zeros(len(args[1])), np.ones(len(args[1])), None, None))
    monkeypatch.setattr(hubble_fit, "plot_full_residuals", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_full_residuals_rz", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_debias_impact_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_redshift_bin_residual_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_fast_vs_uv_variability", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_cosmo_corner", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_residuals_vs_alphaOX", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_completeness_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "make_agn_latex_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "make_agn_csv_table", lambda *args, **kwargs: None)


def _write_fake_checkpoint(path, flat_samples, dmi_posterior_median, dmi_posterior_sigma, *, logz=-1.0, logzerr=0.2):
    hubble_fit.save_chains(
        str(path),
        flat_samples=flat_samples,
        dmi_max_w=np.asarray(dmi_posterior_median, dtype=float),
        dmi_posterior_median=np.asarray(dmi_posterior_median, dtype=float),
        dmi_posterior_sigma=np.asarray(dmi_posterior_sigma, dtype=float),
        dmi_selection_sigma_posterior_median=np.nan,
        logZ=float(logz),
        logZerr=float(logzerr),
        integrals_max_w=np.ones(len(dmi_posterior_median), dtype=float),
    )


def test_run_single_resume_replot_with_cuts_bypasses_sampling_passes_and_plots_current_cut_sample(monkeypatch, tmp_path):
    df_agn = _make_fake_agn_sample(n_agn=4)
    df_agn.loc[:, "z"] = np.array([0.5, 1.2, 3.3, 2.0], dtype=float)
    df_pantheon = _make_fake_pantheon_sample()
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    pipeline_calls = []
    plot_hubble_calls = []
    completeness_plot_calls = []
    agn_chi2_calls = []

    monkeypatch.chdir(tmp_path)
    _patch_run_single_plot_stack(monkeypatch)
    generated_completeness = tmp_path / "generated_completeness.h5"
    monkeypatch.setattr(hubble_fit, "estimate_sky_box_area_deg2", lambda *args, **kwargs: 5.0)
    monkeypatch.setattr(hubble_fit, "generate_fresh_completeness_sim_file", lambda *args, **kwargs: str(generated_completeness))
    monkeypatch.setattr(
        hubble_fit,
        "get_completeness_function_2d",
        lambda *args, **kwargs: (
            completeness_plot_calls.append(kwargs.get("sim_file")),
            (
                np.ones((2, 2)),
                np.array([19.0, 20.0]),
                np.array([0.5, 1.0]),
                0.5,
                0.1,
                0.0,
            ),
        )[1],
    )
    monkeypatch.setattr(hubble_fit, "plot_completeness_vs_mag_at_redshifts", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        hubble_fit,
        "_load_resume_replot_object_ids",
        lambda resume: pytest.fail("_load_resume_replot_object_ids should not select the resume-replot sample"),
    )

    def fake_run_mcmc_pipeline(df_agn_arg, *args, **kwargs):
        pipeline_calls.append(
            {
                "object_ids": df_agn_arg["object_id"].tolist(),
                "resume": kwargs.get("resume"),
                "resume_replot_with_cuts": kwargs.get("resume_replot_with_cuts"),
                "completeness_sim_file": kwargs.get("completeness_sim_file"),
            }
        )
        n = len(df_agn_arg)
        return (
            np.tile(theta[None, :], (8, 1)),
            model_labels,
            lambda pts: np.zeros(len(np.atleast_2d(pts))),
            None,
            -5.0,
            0.2,
            np.arange(n, dtype=float),
            np.full(n, 0.05),
            None,
        )

    def fake_plot_hubble(flat_samples, df_plot, df_pantheon, *args, **kwargs):
        plot_hubble_calls.append(
            {
                "object_ids": df_plot["object_id"].tolist(),
                "filename": kwargs.get("filename"),
                "agn_likelihood_space_chi2": kwargs.get("agn_likelihood_space_chi2"),
                "agn_likelihood_space_chi2_zgt1": kwargs.get("agn_likelihood_space_chi2_zgt1"),
            }
        )
        n = len(df_plot)
        return np.zeros(n), np.ones(n), np.zeros(n), np.ones(n), np.ones(n)

    def fake_compute_agn_likelihood_space_reduced_chi2(flat_samples, model_labels, df_agn_arg, *args, **kwargs):
        object_ids = df_agn_arg["object_id"].tolist()
        agn_chi2_calls.append(object_ids)
        value = 9.87 if len(agn_chi2_calls) == 1 else 6.54
        return value, {"chi2": value, "dof": max(len(object_ids) - len(model_labels), 1)}

    monkeypatch.setattr(hubble_fit, "run_mcmc_pipeline", fake_run_mcmc_pipeline)
    monkeypatch.setattr(
        hubble_fit,
        "_compute_direct_full_sample_completeness_summaries",
        lambda *args, **kwargs: (
            np.zeros(len(df_agn), dtype=float),
            np.zeros(len(df_agn), dtype=float),
            None,
        ),
    )
    monkeypatch.setattr(hubble_fit, "plot_hubble", fake_plot_hubble)
    monkeypatch.setattr(
        hubble_fit,
        "compute_agn_likelihood_space_reduced_chi2",
        fake_compute_agn_likelihood_space_reduced_chi2,
    )

    hubble_fit.run_single(
        df_agn,
        df_agn.copy(),
        df_pantheon,
        None,
        True,
        None,
        cosmo_model="FlatLambdaCDM",
        completeness=True,
        use_full_cov=False,
        resume=str(tmp_path / "posterior.h5"),
        speed="fastest",
        prefix="unit",
        resume_replot_with_cuts=True,
    )

    expected_fit_ids = ["agn_000", "agn_001", "agn_003"]
    expected_plot_ids = df_agn["object_id"].tolist()
    assert len(pipeline_calls) == 1
    assert pipeline_calls[0]["object_ids"] == expected_fit_ids
    assert pipeline_calls[0]["resume"] == str(tmp_path / "posterior.h5")
    assert pipeline_calls[0]["resume_replot_with_cuts"] is True
    assert pipeline_calls[0]["completeness_sim_file"] == str(generated_completeness)
    assert plot_hubble_calls[0]["object_ids"] == expected_plot_ids
    assert plot_hubble_calls[0]["filename"] is None
    assert agn_chi2_calls == [
        ["agn_000", "agn_001", "agn_003"],
        ["agn_001", "agn_003"],
    ]
    assert plot_hubble_calls[0]["agn_likelihood_space_chi2"] == 9.87
    assert plot_hubble_calls[0]["agn_likelihood_space_chi2_zgt1"] == 6.54
    assert completeness_plot_calls == [str(generated_completeness), str(generated_completeness)]


def test_remap_resume_replot_checkpoint_rejects_current_cut_ids_missing_from_checkpoint():
    df_agn = pd.DataFrame({"object_id": ["agn_000", "agn_new"]})
    results = {
        "flat_samples": np.ones((4, 2), dtype=float),
        "object_id_fit_selection": np.array(["agn_000"], dtype=str),
        "dmi_max_w": np.zeros(1, dtype=float),
        "dmi_posterior_sigma": np.full(1, 0.05),
        "integrals_max_w": np.ones(1, dtype=float),
    }

    with pytest.raises(RuntimeError, match="lacks per-object debias arrays"):
        hubble_fit._remap_resume_replot_checkpoint(
            results,
            "posterior.h5",
            df_agn,
            ndim=2,
        )


def test_run_single_two_pass_sigma_clip_filters_outliers_and_writes_diagnostics(monkeypatch, tmp_path):
    df_agn = _make_fake_agn_sample(n_agn=5)
    df_agn = pd.concat(
        [
            df_agn,
            df_agn.loc[[1]].assign(object_id="agn_005", z=0.2),
        ],
        ignore_index=True,
    )
    df_pantheon = _make_fake_pantheon_sample()
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    pipeline_calls = []
    plot_hubble_calls = []

    monkeypatch.chdir(tmp_path)
    _patch_run_single_plot_stack(monkeypatch)

    def fake_run_mcmc_pipeline(df_agn, *args, **kwargs):
        pipeline_calls.append(
            {
                "object_ids": df_agn["object_id"].tolist(),
                "completeness_object_ids": kwargs["df_agn_completeness"]["object_id"].tolist(),
                "warm_start_flat_samples": kwargs.get("warm_start_flat_samples"),
                "logZ_is_approximate": kwargs.get("logZ_is_approximate"),
            }
        )
        flat_samples = np.tile(theta[None, :], (8, 1))
        n = len(df_agn)
        logz = -10.0 - len(pipeline_calls)
        _write_fake_checkpoint(
            kwargs["checkpoint_file_override"],
            flat_samples,
            np.zeros(n),
            np.full(n, 0.05),
            logz=logz,
            logzerr=0.2,
        )
        return (
            flat_samples,
            model_labels,
            lambda pts: np.zeros(len(np.atleast_2d(pts))),
            None,
            logz,
            0.2,
            np.zeros(n),
            np.full(n, 0.05),
            None,
        )

    def fake_plot_hubble(flat_samples, df_plot, df_pantheon, *args, **kwargs):
        plot_hubble_calls.append(
            {
                "object_ids": df_plot["object_id"].tolist(),
                "filename": kwargs.get("filename"),
                "clipped_mask": kwargs.get("clipped_mask"),
                "sigma_clip_threshold": kwargs.get("sigma_clip_threshold"),
            }
        )
        n = len(df_plot)
        if len(plot_hubble_calls) == 1:
            residuals = np.zeros(n, dtype=float)
            residuals[:6] = np.array([0.1, 3.5, np.nan, 0.2, 0.3, 4.2], dtype=float)
            residuals_err = np.ones(n, dtype=float)
            return (
                residuals,
                residuals_err,
                np.full(n, 44.0),
                np.full(n, 0.1),
                np.full(n, 0.2),
            )
        return (
            np.zeros(n, dtype=float),
            np.ones(n, dtype=float),
            np.full(n, 44.0),
            np.full(n, 0.1),
            np.full(n, 0.2),
        )

    monkeypatch.setattr(hubble_fit, "run_mcmc_pipeline", fake_run_mcmc_pipeline)
    monkeypatch.setattr(hubble_fit, "plot_hubble", fake_plot_hubble)

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
        speed="fastest",
        z_range=(0.44, 3.16),
        disable_sigma_clip_pass=False,
        sigma_clip_threshold=3.0,
        prefix="unit",
    )

    assert len(pipeline_calls) == 2
    assert pipeline_calls[0]["object_ids"] == ["agn_000", "agn_001", "agn_002", "agn_003", "agn_004"]
    assert pipeline_calls[0]["completeness_object_ids"] == df_agn["object_id"].tolist()
    assert pipeline_calls[0]["warm_start_flat_samples"] is None
    assert pipeline_calls[0]["logZ_is_approximate"] is False
    assert pipeline_calls[1]["object_ids"] == ["agn_000", "agn_003", "agn_004"]
    assert pipeline_calls[1]["completeness_object_ids"] == df_agn["object_id"].tolist()
    np.testing.assert_allclose(pipeline_calls[1]["warm_start_flat_samples"], np.tile(theta[None, :], (8, 1)))
    assert pipeline_calls[1]["logZ_is_approximate"] is True
    assert result[3] == -12.0
    assert plot_hubble_calls[0]["object_ids"] == df_agn["object_id"].tolist()
    assert plot_hubble_calls[0]["filename"] == "hubble_diagram_pass1_full_sample_debiased.pdf"
    assert plot_hubble_calls[0]["clipped_mask"] is None
    assert plot_hubble_calls[0]["sigma_clip_threshold"] == 3.0
    np.testing.assert_array_equal(
        plot_hubble_calls[1]["clipped_mask"],
        np.array([False, True, True, False, False, True], dtype=bool),
    )
    assert plot_hubble_calls[1]["filename"] == "hubble_diagram_pass1_full_sample_clipped_debiased.pdf"
    assert plot_hubble_calls[1]["sigma_clip_threshold"] == 3.0

    run_dir = tmp_path / "plots" / "hubble" / "unit" / "FlatLambdaCDM_joint_fastest_all_z0p44_3p16_disable_completeness"
    run_tag = hubble_fit.make_run_tag("FlatLambdaCDM", False, "fastest", None, (0.44, 3.16), completeness=False)
    checkpoint_paths = hubble_fit._build_checkpoint_paths("unit", run_tag)
    pass1_df = pd.read_csv(run_dir / "residuals_pass1.csv")
    clipped_df = pd.read_csv(run_dir / "clipped_objects_pass1.csv")
    final_df = pd.read_csv(run_dir / "residuals.csv")
    pass1_membership_df = pd.read_csv(run_dir / "sigma_clip_membership_pass1.csv")
    pass2_membership_df = pd.read_csv(run_dir / "sigma_clip_membership_pass2.csv")
    assert set(pass1_df["object_id"]) == set(df_agn["object_id"])
    assert pass1_df.loc[pass1_df["object_id"] == "agn_005", "was_clipped"].item()
    assert set(pass1_df.loc[pass1_df["was_clipped"], "object_id"]) == {"agn_001", "agn_002", "agn_005"}
    assert set(clipped_df["object_id"]) == {"agn_001", "agn_002", "agn_005"}
    assert set(final_df["object_id"]) == {"agn_000", "agn_003", "agn_004"}
    assert final_df["was_clipped_pass1"].eq(False).all()
    assert final_df["was_clipped_pass2"].eq(False).all()
    assert final_df["is_in_pass2_sample"].eq(True).all()
    assert final_df["is_in_pass2_plot_sample"].eq(True).all()
    assert final_df["is_in_pass2_fit_selection"].eq(True).all()
    assert "mu_zscore_pass1" in final_df.columns
    assert "mu_zscore_pass2" in final_df.columns
    assert set(pass1_membership_df.loc[pass1_membership_df["was_clipped_pass1"], "object_id"]) == {"agn_001", "agn_002", "agn_005"}
    assert set(pass2_membership_df.loc[pass2_membership_df["is_in_pass2_sample"], "object_id"]) == {"agn_000", "agn_003", "agn_004"}
    assert pass2_membership_df["is_in_pass2_plot_sample"].equals(pass2_membership_df["is_in_pass2_sample"])
    assert set(pass2_membership_df.loc[pass2_membership_df["is_in_pass2_fit_selection"], "object_id"]) == {"agn_000", "agn_003", "agn_004"}
    assert not set(pass2_membership_df.loc[pass2_membership_df["was_clipped_pass1"], "object_id"]) & set(
        pass2_membership_df.loc[pass2_membership_df["is_in_pass2_sample"], "object_id"]
    )
    pass1_checkpoint = hubble_fit.load_chains(checkpoint_paths["pass1"])
    pass2_checkpoint = hubble_fit.load_chains(checkpoint_paths["pass2"])
    assert hubble_fit._checkpoint_stage_from_results(pass1_checkpoint) == "pass1"
    assert hubble_fit._checkpoint_stage_from_results(pass2_checkpoint) == "pass2"
    assert pass2_checkpoint["sigma_clip_second_pass_mode"] == "warm"
    assert bool(pass2_checkpoint["sigma_clip_warm_start_from_pass1"])
    assert bool(pass2_checkpoint["logZ_is_approximate"])
    assert "keep_mask_full" in pass1_checkpoint
    assert "mu_zscore_pass1" in pass1_checkpoint
    assert "keep_mask_full" in pass2_checkpoint
    assert "mu_zscore_pass1" in pass2_checkpoint
    assert len(pass1_checkpoint["object_id_fit_selection"]) == len(pass1_checkpoint["dmi_posterior_median"])
    assert len(pass2_checkpoint["object_id_fit_selection"]) == len(pass2_checkpoint["dmi_posterior_median"])
    assert set(pass1_checkpoint["object_id_fit_selection"].astype(str)) == {
        "agn_000", "agn_001", "agn_002", "agn_003", "agn_004"
    }
    assert set(pass2_checkpoint["object_id_fit_selection"].astype(str)) == {"agn_000", "agn_003", "agn_004"}
    assert set(pass2_checkpoint["object_id_plot_sample"].astype(str)) == {"agn_000", "agn_003", "agn_004"}


def test_run_single_two_pass_sigma_clip_fresh_mode_reruns_without_warm_start(monkeypatch, tmp_path):
    df_agn = _make_fake_agn_sample(n_agn=4)
    df_pantheon = _make_fake_pantheon_sample()
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    pipeline_calls = []
    plot_hubble_calls = []

    monkeypatch.chdir(tmp_path)
    _patch_run_single_plot_stack(monkeypatch)

    def fake_run_mcmc_pipeline(df_agn, *args, **kwargs):
        pipeline_calls.append(
            {
                "object_ids": df_agn["object_id"].tolist(),
                "warm_start_flat_samples": kwargs.get("warm_start_flat_samples"),
                "logZ_is_approximate": kwargs.get("logZ_is_approximate"),
            }
        )
        flat_samples = np.tile(theta[None, :], (8, 1))
        n = len(df_agn)
        _write_fake_checkpoint(
            kwargs["checkpoint_file_override"],
            flat_samples,
            np.zeros(n),
            np.full(n, 0.05),
            logz=-20.0 - len(pipeline_calls),
            logzerr=0.2,
        )
        return (
            flat_samples,
            model_labels,
            lambda pts: np.zeros(len(np.atleast_2d(pts))),
            None,
            -20.0 - len(pipeline_calls),
            0.2,
            np.zeros(n),
            np.full(n, 0.05),
            None,
        )

    def fake_plot_hubble(flat_samples, df_plot, df_pantheon, *args, **kwargs):
        plot_hubble_calls.append(kwargs)
        n = len(df_plot)
        return (
            np.full(n, 0.5, dtype=float),
            np.ones(n, dtype=float),
            np.full(n, 44.0),
            np.full(n, 0.1),
            np.full(n, 0.2),
        )

    monkeypatch.setattr(hubble_fit, "run_mcmc_pipeline", fake_run_mcmc_pipeline)
    monkeypatch.setattr(hubble_fit, "plot_hubble", fake_plot_hubble)

    hubble_fit.run_single(
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
        speed="fastest",
        z_range=(0.44, 3.16),
        disable_sigma_clip_pass=False,
        sigma_clip_threshold=3.0,
        sigma_clip_second_pass_mode="fresh",
        prefix="unit",
    )

    assert len(pipeline_calls) == 2
    assert pipeline_calls[1]["object_ids"] == pipeline_calls[0]["object_ids"]
    assert pipeline_calls[0]["warm_start_flat_samples"] is None
    assert pipeline_calls[1]["warm_start_flat_samples"] is None
    assert pipeline_calls[0]["logZ_is_approximate"] is False
    assert pipeline_calls[1]["logZ_is_approximate"] is False
    assert len(plot_hubble_calls) == 5
    assert plot_hubble_calls[0]["filename"] == "hubble_diagram_pass1_full_sample_debiased.pdf"
    assert plot_hubble_calls[0].get("clipped_mask") is None
    assert plot_hubble_calls[0]["sigma_clip_threshold"] == 3.0
    np.testing.assert_array_equal(plot_hubble_calls[1]["clipped_mask"], np.zeros(len(df_agn), dtype=bool))
    assert plot_hubble_calls[1]["filename"] == "hubble_diagram_pass1_full_sample_clipped_debiased.pdf"
    assert plot_hubble_calls[1]["sigma_clip_threshold"] == 3.0
    assert plot_hubble_calls[4]["filename"] == "hubble_diagram_debiased_no_logf.pdf"
    assert plot_hubble_calls[4]["use_intrinsic_scatter_in_residual_sigma"] is False
    assert plot_hubble_calls[4]["diagnostics_suffix"] == "_debiased_no_logf"


def test_run_single_two_pass_sigma_clip_removes_clipped_object_ids_from_second_pass_outputs(monkeypatch, tmp_path):
    df_agn = _make_fake_agn_sample(n_agn=5)
    df_agn = pd.concat(
        [
            df_agn,
            df_agn.loc[[1]].assign(z=0.2),
        ],
        ignore_index=True,
    )
    df_pantheon = _make_fake_pantheon_sample()
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    plot_hubble_calls = []
    l2500_calls = []
    m2500_calls = []
    full_residual_calls = []
    full_residual_rz_calls = []
    blr_calls = []
    blr_pdf_calls = []
    debias_impact_calls = []
    alphaox_calls = []

    monkeypatch.chdir(tmp_path)
    _patch_run_single_plot_stack(monkeypatch)

    def _record_df_call(calls, df_arg, kwargs):
        calls.append(
            {
                "object_ids": df_arg["object_id"].tolist(),
                "filename": kwargs.get("filename"),
                "clipped_mask": kwargs.get("clipped_mask"),
                "sigma_clip_threshold": kwargs.get("sigma_clip_threshold"),
            }
        )

    def fake_run_mcmc_pipeline(df_agn, *args, **kwargs):
        flat_samples = np.tile(theta[None, :], (8, 1))
        n = len(df_agn)
        _write_fake_checkpoint(
            kwargs["checkpoint_file_override"],
            flat_samples,
            np.zeros(n),
            np.full(n, 0.05),
            logz=-30.0,
            logzerr=0.2,
        )
        return (
            flat_samples,
            model_labels,
            lambda pts: np.zeros(len(np.atleast_2d(pts))),
            None,
            -30.0,
            0.2,
            np.zeros(n),
            np.full(n, 0.05),
            None,
        )

    def fake_plot_hubble(*args, **kwargs):
        _record_df_call(plot_hubble_calls, args[1], kwargs)
        n = len(args[1])
        if len(plot_hubble_calls) == 1:
            residuals = np.zeros(n, dtype=float)
            residuals[:6] = np.array([0.1, 3.5, np.nan, 0.2, 0.3, 4.2], dtype=float)
        else:
            residuals = np.zeros(n, dtype=float)
        return (
            residuals,
            np.ones(n, dtype=float),
            np.full(n, 44.0),
            np.full(n, 0.1),
            np.full(n, 0.2),
        )

    monkeypatch.setattr(hubble_fit, "run_mcmc_pipeline", fake_run_mcmc_pipeline)
    monkeypatch.setattr(hubble_fit, "plot_hubble", fake_plot_hubble)
    monkeypatch.setattr(
        hubble_fit,
        "plot_predicted_L2500_vs_sigmahat",
        lambda *args, **kwargs: (l2500_calls.append(args[1]["object_id"].tolist()), (np.zeros(len(args[1])), np.ones(len(args[1]))))[1],
    )
    monkeypatch.setattr(hubble_fit, "plot_blr_line_lags_vs_l2500", lambda *args, **kwargs: blr_calls.append(args[1]["object_id"].tolist()))
    monkeypatch.setattr(hubble_fit, "plot_blr_diagnostics_summary", lambda *args, **kwargs: blr_pdf_calls.append(args[0]["object_id"].tolist()))
    monkeypatch.setattr(
        hubble_fit,
        "plot_predicted_vs_actual_M2500",
        lambda *args, **kwargs: (m2500_calls.append(args[1]["object_id"].tolist()), (np.zeros(len(args[1])), np.ones(len(args[1])), None, None))[1],
    )
    monkeypatch.setattr(hubble_fit, "plot_full_residuals", lambda *args, **kwargs: full_residual_calls.append(args[0]["object_id"].tolist()))
    monkeypatch.setattr(hubble_fit, "plot_full_residuals_rz", lambda *args, **kwargs: full_residual_rz_calls.append(args[0]["object_id"].tolist()))
    monkeypatch.setattr(hubble_fit, "plot_debias_impact_diagnostics", lambda *args, **kwargs: debias_impact_calls.append(args[0]["object_id"].tolist()))
    monkeypatch.setattr(hubble_fit, "plot_residuals_vs_alphaOX", lambda *args, **kwargs: alphaox_calls.append(args[0]["object_id"].tolist()))

    hubble_fit.run_single(
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
        speed="fastest",
        z_range=(0.44, 3.16),
        disable_sigma_clip_pass=False,
        sigma_clip_threshold=3.0,
        prefix="unit",
    )

    expected_full_ids = ["agn_000", "agn_001", "agn_002", "agn_003", "agn_004", "agn_001"]
    expected_second_pass_ids = ["agn_000", "agn_003", "agn_004"]
    assert plot_hubble_calls[0]["object_ids"] == expected_full_ids
    assert plot_hubble_calls[0]["filename"] == "hubble_diagram_pass1_full_sample_debiased.pdf"
    assert plot_hubble_calls[0]["clipped_mask"] is None
    assert plot_hubble_calls[0]["sigma_clip_threshold"] == 3.0
    assert plot_hubble_calls[1]["object_ids"] == expected_full_ids
    assert plot_hubble_calls[1]["filename"] == "hubble_diagram_pass1_full_sample_clipped_debiased.pdf"
    np.testing.assert_array_equal(
        plot_hubble_calls[1]["clipped_mask"],
        np.array([False, True, True, False, False, True], dtype=bool),
    )
    assert plot_hubble_calls[1]["sigma_clip_threshold"] == 3.0
    for call in plot_hubble_calls[2:]:
        assert call["object_ids"] == expected_second_pass_ids
    for call_ids in l2500_calls:
        assert call_ids == expected_second_pass_ids
    for call_ids in m2500_calls:
        assert call_ids == expected_second_pass_ids
    for call_ids in full_residual_calls:
        assert call_ids == expected_second_pass_ids
    for call_ids in full_residual_rz_calls:
        assert call_ids == expected_second_pass_ids
    assert blr_calls[0] == expected_second_pass_ids
    assert blr_pdf_calls[0] == expected_second_pass_ids
    assert debias_impact_calls[0] == expected_second_pass_ids
    assert alphaox_calls[0] == expected_second_pass_ids


def test_run_single_two_pass_sigma_clip_keeps_new_pass2_outlier(monkeypatch, tmp_path):
    df_agn = _make_fake_agn_sample(n_agn=5)
    df_pantheon = _make_fake_pantheon_sample()
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    plot_hubble_calls = []

    monkeypatch.chdir(tmp_path)
    _patch_run_single_plot_stack(monkeypatch)

    def fake_run_mcmc_pipeline(df_agn, *args, **kwargs):
        flat_samples = np.tile(theta[None, :], (8, 1))
        n = len(df_agn)
        _write_fake_checkpoint(
            kwargs["checkpoint_file_override"],
            flat_samples,
            np.zeros(n),
            np.full(n, 0.05),
            logz=-33.0,
            logzerr=0.2,
        )
        return (
            flat_samples,
            model_labels,
            lambda pts: np.zeros(len(np.atleast_2d(pts))),
            None,
            -33.0,
            0.2,
            np.zeros(n),
            np.full(n, 0.05),
            None,
        )

    def fake_plot_hubble(*args, **kwargs):
        df_plot = args[1]
        plot_hubble_calls.append(
            {
                "object_ids": df_plot["object_id"].tolist(),
                "filename": kwargs.get("filename"),
                "sigma_clip_threshold": kwargs.get("sigma_clip_threshold"),
            }
        )
        n = len(df_plot)
        if len(plot_hubble_calls) == 1:
            residuals = np.array([0.1, 2.9, 0.2, 0.3, 0.4], dtype=float)
        elif kwargs.get("filename") is None and kwargs.get("debias"):
            residuals = np.array([0.1, 3.1, 0.2, 0.3, 0.4], dtype=float)
        else:
            residuals = np.zeros(n, dtype=float)
        return (
            residuals,
            np.ones(n, dtype=float),
            np.full(n, 44.0),
            np.full(n, 0.1),
            np.full(n, 0.2),
        )

    monkeypatch.setattr(hubble_fit, "run_mcmc_pipeline", fake_run_mcmc_pipeline)
    monkeypatch.setattr(hubble_fit, "plot_hubble", fake_plot_hubble)

    hubble_fit.run_single(
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
        speed="fastest",
        z_range=(0.44, 3.16),
        disable_sigma_clip_pass=False,
        sigma_clip_threshold=3.0,
        prefix="unit",
    )

    run_dir = tmp_path / "plots" / "hubble" / "unit" / "FlatLambdaCDM_joint_fastest_all_z0p44_3p16_disable_completeness"
    final_df = pd.read_csv(run_dir / "residuals.csv")
    assert set(final_df["object_id"]) == set(df_agn["object_id"])
    assert final_df.loc[final_df["object_id"] == "agn_001", "mu_zscore_pass1"].item() < 3.0
    assert final_df.loc[final_df["object_id"] == "agn_001", "mu_zscore_pass2"].item() > 3.0
    assert final_df.loc[final_df["object_id"] == "agn_001", "was_clipped_pass1"].item() == False
    assert final_df.loc[final_df["object_id"] == "agn_001", "is_in_pass2_sample"].item() == True


def test_run_single_two_pass_sigma_clip_keeps_out_of_range_survivor_in_stage2_plots(monkeypatch, tmp_path):
    df_agn = _make_fake_agn_sample(n_agn=5)
    df_agn = pd.concat(
        [
            df_agn,
            df_agn.loc[[1]].assign(object_id="agn_out_survivor", z=3.4),
            df_agn.loc[[2]].assign(object_id="agn_out_clipped", z=3.5),
        ],
        ignore_index=True,
    )
    df_pantheon = _make_fake_pantheon_sample()
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    pipeline_calls = []
    plot_hubble_calls = []
    l2500_calls = []
    m2500_calls = []
    full_residual_calls = []
    full_residual_rz_calls = []
    blr_calls = []
    blr_pdf_calls = []
    debias_impact_calls = []
    alphaox_calls = []

    monkeypatch.chdir(tmp_path)
    _patch_run_single_plot_stack(monkeypatch)

    def fake_run_mcmc_pipeline(df_agn, *args, **kwargs):
        pipeline_calls.append(df_agn["object_id"].tolist())
        flat_samples = np.tile(theta[None, :], (8, 1))
        n = len(df_agn)
        _write_fake_checkpoint(
            kwargs["checkpoint_file_override"],
            flat_samples,
            np.zeros(n),
            np.full(n, 0.05),
            logz=-60.0,
            logzerr=0.2,
        )
        return (
            flat_samples,
            model_labels,
            lambda pts: np.zeros(len(np.atleast_2d(pts))),
            None,
            -60.0,
            0.2,
            np.zeros(n),
            np.full(n, 0.05),
            None,
        )

    def fake_plot_hubble(*args, **kwargs):
        df_plot = args[1]
        plot_hubble_calls.append(
            {
                "object_ids": df_plot["object_id"].tolist(),
                "filename": kwargs.get("filename"),
                "residuals_csv_filename": kwargs.get("residuals_csv_filename"),
            }
        )
        n = len(df_plot)
        if len(plot_hubble_calls) == 1:
            residuals = np.array([0.1, 3.5, np.nan, 0.2, 0.3, 0.4, 4.2], dtype=float)
        else:
            residuals = np.zeros(n, dtype=float)
        return (
            residuals,
            np.ones(n, dtype=float),
            np.full(n, 44.0),
            np.full(n, 0.1),
            np.full(n, 0.2),
        )

    monkeypatch.setattr(hubble_fit, "run_mcmc_pipeline", fake_run_mcmc_pipeline)
    monkeypatch.setattr(hubble_fit, "plot_hubble", fake_plot_hubble)
    monkeypatch.setattr(
        hubble_fit,
        "plot_predicted_L2500_vs_sigmahat",
        lambda *args, **kwargs: (l2500_calls.append(args[1]["object_id"].tolist()), (np.zeros(len(args[1])), np.ones(len(args[1]))))[1],
    )
    monkeypatch.setattr(hubble_fit, "plot_blr_line_lags_vs_l2500", lambda *args, **kwargs: blr_calls.append(args[1]["object_id"].tolist()))
    monkeypatch.setattr(hubble_fit, "plot_blr_diagnostics_summary", lambda *args, **kwargs: blr_pdf_calls.append(args[0]["object_id"].tolist()))
    monkeypatch.setattr(
        hubble_fit,
        "plot_predicted_vs_actual_M2500",
        lambda *args, **kwargs: (m2500_calls.append(args[1]["object_id"].tolist()), (np.zeros(len(args[1])), np.ones(len(args[1])), None, None))[1],
    )
    monkeypatch.setattr(hubble_fit, "plot_full_residuals", lambda *args, **kwargs: full_residual_calls.append(args[0]["object_id"].tolist()))
    monkeypatch.setattr(hubble_fit, "plot_full_residuals_rz", lambda *args, **kwargs: full_residual_rz_calls.append(args[0]["object_id"].tolist()))
    monkeypatch.setattr(hubble_fit, "plot_debias_impact_diagnostics", lambda *args, **kwargs: debias_impact_calls.append(args[0]["object_id"].tolist()))
    monkeypatch.setattr(hubble_fit, "plot_residuals_vs_alphaOX", lambda *args, **kwargs: alphaox_calls.append(args[0]["object_id"].tolist()))

    hubble_fit.run_single(
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
        speed="fastest",
        z_range=(0.44, 3.16),
        disable_sigma_clip_pass=False,
        sigma_clip_threshold=3.0,
        prefix="unit",
    )

    expected_stage2_plot_ids = ["agn_000", "agn_003", "agn_004", "agn_out_survivor"]
    expected_stage2_fit_ids = ["agn_000", "agn_003", "agn_004"]
    assert pipeline_calls[0] == ["agn_000", "agn_001", "agn_002", "agn_003", "agn_004"]
    assert pipeline_calls[1] == expected_stage2_fit_ids
    assert plot_hubble_calls[0]["object_ids"] == df_agn["object_id"].tolist()
    assert plot_hubble_calls[2]["object_ids"] == expected_stage2_plot_ids
    assert plot_hubble_calls[2]["residuals_csv_filename"] == "hubble_plot_residuals.csv"
    assert plot_hubble_calls[3]["object_ids"] == expected_stage2_plot_ids
    for call_ids in l2500_calls:
        assert call_ids == expected_stage2_plot_ids
    for call_ids in m2500_calls:
        assert call_ids == expected_stage2_plot_ids
    for call_ids in full_residual_calls:
        assert call_ids == expected_stage2_plot_ids
    for call_ids in full_residual_rz_calls:
        assert call_ids == expected_stage2_plot_ids
    assert blr_calls[0] == expected_stage2_plot_ids
    assert blr_pdf_calls[0] == expected_stage2_plot_ids
    assert debias_impact_calls[0] == expected_stage2_plot_ids
    assert alphaox_calls[0] == expected_stage2_plot_ids

    run_dir = tmp_path / "plots" / "hubble" / "unit" / "FlatLambdaCDM_joint_fastest_all_z0p44_3p16_disable_completeness"
    final_df = pd.read_csv(run_dir / "residuals.csv")
    pass2_membership_df = pd.read_csv(run_dir / "sigma_clip_membership_pass2.csv")
    assert set(final_df["object_id"]) == set(expected_stage2_plot_ids)
    assert final_df["was_clipped_pass1"].eq(False).all()
    assert final_df["is_in_pass2_plot_sample"].eq(True).all()
    assert set(final_df.loc[final_df["is_in_pass2_fit_selection"], "object_id"]) == set(expected_stage2_fit_ids)
    assert set(final_df.loc[~final_df["is_in_pass2_fit_selection"], "object_id"]) == {"agn_out_survivor"}
    assert final_df.loc[final_df["object_id"] == "agn_out_survivor", "z"].item() > 3.16
    assert set(pass2_membership_df.loc[pass2_membership_df["is_in_pass2_plot_sample"], "object_id"]) == set(expected_stage2_plot_ids)
    assert set(pass2_membership_df.loc[pass2_membership_df["is_in_pass2_fit_selection"], "object_id"]) == set(expected_stage2_fit_ids)
    assert not {"agn_001", "agn_002", "agn_out_clipped"} & set(final_df["object_id"])


def test_run_single_resume_stage_pass2_skips_first_pass(monkeypatch, tmp_path):
    df_agn = _make_fake_agn_sample(n_agn=5)
    df_agn = pd.concat([df_agn, df_agn.loc[[1]].assign(object_id="agn_005", z=0.2)], ignore_index=True)
    df_pantheon = _make_fake_pantheon_sample()
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)

    monkeypatch.chdir(tmp_path)
    _patch_run_single_plot_stack(monkeypatch)
    monkeypatch.setattr(
        hubble_fit,
        "plot_hubble",
        lambda *args, **kwargs: (
            np.zeros(len(args[1]), dtype=float),
            np.ones(len(args[1]), dtype=float),
            np.full(len(args[1]), 44.0),
            np.full(len(args[1]), 0.1),
            np.full(len(args[1]), 0.2),
        ),
    )

    run_tag = hubble_fit.make_run_tag("FlatLambdaCDM", False, "fastest", None, (0.44, 3.16), completeness=False)
    checkpoint_paths = hubble_fit._build_checkpoint_paths("unit", run_tag)
    flat_samples_pass1 = np.tile(theta[None, :], (8, 1))
    _write_fake_checkpoint(checkpoint_paths["pass1"], flat_samples_pass1, np.zeros(5), np.full(5, 0.05), logz=-11.0)
    pass1_diag = df_agn.copy()
    pass1_diag["residuals"] = np.array([0.1, 3.5, 0.2, 0.3, 0.4, 4.2], dtype=float)
    pass1_diag["residuals_err"] = np.ones(len(df_agn), dtype=float)
    pass1_diag["mu_zscore"] = np.array([0.1, 3.5, 0.2, 0.3, 0.4, 4.2], dtype=float)
    pass1_diag["was_clipped"] = np.array([False, True, False, False, False, True], dtype=bool)
    keep_mask_full = ~pass1_diag["was_clipped"].to_numpy(dtype=bool)
    hubble_fit._write_stage_checkpoint(
        checkpoint_paths["pass1"],
        sigma_clip_pass_stage="pass1",
        sigma_clip_threshold=3.0,
        df_agn_full_sample=df_agn,
        df_agn_fit_selection=df_agn[df_agn["z"].between(0.44, 3.16)].copy(),
        keep_mask_full=keep_mask_full,
        pass1_diagnostics_df=pass1_diag,
    )

    pipeline_calls = []

    def fake_run_mcmc_pipeline(df_agn, *args, **kwargs):
        pipeline_calls.append(
            {
                "object_ids": df_agn["object_id"].tolist(),
                "resume": kwargs.get("resume"),
                "warm_start_flat_samples": kwargs.get("warm_start_flat_samples"),
                "logZ_is_approximate": kwargs.get("logZ_is_approximate"),
            }
        )
        flat_samples = np.tile(theta[None, :], (8, 1))
        n = len(df_agn)
        _write_fake_checkpoint(kwargs["checkpoint_file_override"], flat_samples, np.zeros(n), np.full(n, 0.05), logz=-12.0)
        return (
            flat_samples,
            model_labels,
            lambda pts: np.zeros(len(np.atleast_2d(pts))),
            None,
            -12.0,
            0.2,
            np.zeros(n),
            np.full(n, 0.05),
            None,
        )

    monkeypatch.setattr(hubble_fit, "run_mcmc_pipeline", fake_run_mcmc_pipeline)

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
        speed="fastest",
        z_range=(0.44, 3.16),
        disable_sigma_clip_pass=False,
        sigma_clip_threshold=3.0,
        resume=True,
        resume_stage="pass2",
        prefix="unit",
    )

    assert len(pipeline_calls) == 1
    assert pipeline_calls[0]["resume"] is False
    assert pipeline_calls[0]["object_ids"] == ["agn_000", "agn_002", "agn_003", "agn_004"]
    np.testing.assert_allclose(pipeline_calls[0]["warm_start_flat_samples"], flat_samples_pass1)
    assert pipeline_calls[0]["logZ_is_approximate"] is True
    assert result[3] == -12.0


def test_run_single_resume_stage_pass2_rejects_legacy_checkpoint(monkeypatch, tmp_path):
    df_agn = _make_fake_agn_sample(n_agn=4)
    df_pantheon = _make_fake_pantheon_sample()
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)

    monkeypatch.chdir(tmp_path)
    _patch_run_single_plot_stack(monkeypatch)

    run_tag = hubble_fit.make_run_tag("FlatLambdaCDM", False, "fastest", None, (0.44, 3.16), completeness=False)
    checkpoint_paths = hubble_fit._build_checkpoint_paths("unit", run_tag)
    _write_fake_checkpoint(checkpoint_paths["single"], np.tile(theta[None, :], (8, 1)), np.zeros(len(df_agn)), np.full(len(df_agn), 0.05), logz=-9.0)

    with pytest.raises(RuntimeError, match="embedded pass-1 clipping state"):
        hubble_fit.run_single(
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
            speed="fastest",
            z_range=(0.44, 3.16),
            disable_sigma_clip_pass=False,
            sigma_clip_threshold=3.0,
            resume=True,
            resume_stage="pass2",
            prefix="unit",
        )


def test_run_single_resume_stage_pass1_stops_before_second_pass(monkeypatch, tmp_path):
    df_agn = _make_fake_agn_sample(n_agn=4)
    df_pantheon = _make_fake_pantheon_sample()
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    pipeline_calls = []

    monkeypatch.chdir(tmp_path)
    _patch_run_single_plot_stack(monkeypatch)

    def fake_run_mcmc_pipeline(df_agn, *args, **kwargs):
        pipeline_calls.append({"object_ids": df_agn["object_id"].tolist(), "resume": kwargs.get("resume")})
        flat_samples = np.tile(theta[None, :], (8, 1))
        n = len(df_agn)
        _write_fake_checkpoint(kwargs["checkpoint_file_override"], flat_samples, np.zeros(n), np.full(n, 0.05), logz=-14.0)
        return (
            flat_samples,
            model_labels,
            lambda pts: np.zeros(len(np.atleast_2d(pts))),
            None,
            -14.0,
            0.2,
            np.zeros(n),
            np.full(n, 0.05),
            None,
        )

    monkeypatch.setattr(hubble_fit, "run_mcmc_pipeline", fake_run_mcmc_pipeline)
    monkeypatch.setattr(
        hubble_fit,
        "plot_hubble",
        lambda *args, **kwargs: (
            np.full(len(args[1]), 0.1, dtype=float),
            np.ones(len(args[1]), dtype=float),
            np.full(len(args[1]), 44.0),
            np.full(len(args[1]), 0.1),
            np.full(len(args[1]), 0.2),
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
        speed="fastest",
        z_range=(0.44, 3.16),
        disable_sigma_clip_pass=False,
        sigma_clip_threshold=3.0,
        resume_stage="pass1",
        prefix="unit",
    )

    assert len(pipeline_calls) == 1
    assert pipeline_calls[0]["object_ids"] == df_agn["object_id"].tolist()
    assert result[3] == -14.0


def test_run_single_disable_sigma_clip_pass_skips_two_pass_branch(monkeypatch, tmp_path):
    df_agn = _make_fake_agn_sample(n_agn=4)
    df_pantheon = _make_fake_pantheon_sample()
    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    pipeline_calls = []
    plot_hubble_calls = []
    l2500_calls = []
    m2500_calls = []
    full_residual_calls = []
    full_residual_rz_calls = []
    blr_calls = []
    blr_pdf_calls = []
    debias_impact_calls = []
    alphaox_calls = []

    monkeypatch.chdir(tmp_path)
    _patch_run_single_plot_stack(monkeypatch)

    def fake_run_mcmc_pipeline(df_agn, *args, **kwargs):
        pipeline_calls.append(df_agn["object_id"].tolist())
        flat_samples = np.tile(theta[None, :], (8, 1))
        n = len(df_agn)
        _write_fake_checkpoint(
            kwargs["checkpoint_file_override"],
            flat_samples,
            np.zeros(n),
            np.full(n, 0.05),
            logz=-40.0,
            logzerr=0.2,
        )
        return (
            flat_samples,
            model_labels,
            lambda pts: np.zeros(len(np.atleast_2d(pts))),
            None,
            -40.0,
            0.2,
            np.zeros(n),
            np.full(n, 0.05),
            None,
        )

    def fake_plot_hubble(*args, **kwargs):
        plot_hubble_calls.append(kwargs)
        n = len(args[1])
        return (
            np.full(n, 0.5, dtype=float),
            np.ones(n, dtype=float),
            np.full(n, 44.0),
            np.full(n, 0.1),
            np.full(n, 0.2),
        )

    monkeypatch.setattr(hubble_fit, "run_mcmc_pipeline", fake_run_mcmc_pipeline)
    monkeypatch.setattr(hubble_fit, "plot_hubble", fake_plot_hubble)
    monkeypatch.setattr(
        hubble_fit,
        "plot_predicted_L2500_vs_sigmahat",
        lambda *args, **kwargs: (l2500_calls.append(kwargs), (np.zeros(len(args[1])), np.ones(len(args[1]))))[1],
    )
    monkeypatch.setattr(hubble_fit, "plot_blr_line_lags_vs_l2500", lambda *args, **kwargs: blr_calls.append(kwargs))
    monkeypatch.setattr(hubble_fit, "plot_blr_diagnostics_summary", lambda *args, **kwargs: blr_pdf_calls.append(kwargs))
    monkeypatch.setattr(
        hubble_fit,
        "plot_predicted_vs_actual_M2500",
        lambda *args, **kwargs: (m2500_calls.append(kwargs), (np.zeros(len(args[1])), np.ones(len(args[1])), None, None))[1],
    )
    monkeypatch.setattr(hubble_fit, "plot_full_residuals", lambda *args, **kwargs: full_residual_calls.append(kwargs))
    monkeypatch.setattr(hubble_fit, "plot_full_residuals_rz", lambda *args, **kwargs: full_residual_rz_calls.append(kwargs))
    monkeypatch.setattr(hubble_fit, "plot_debias_impact_diagnostics", lambda *args, **kwargs: debias_impact_calls.append(kwargs))
    monkeypatch.setattr(hubble_fit, "plot_residuals_vs_alphaOX", lambda *args, **kwargs: alphaox_calls.append(kwargs))

    hubble_fit.run_single(
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
        speed="fastest",
        z_range=(0.44, 3.16),
        disable_sigma_clip_pass=True,
        sigma_clip_threshold=3.0,
        prefix="unit",
    )

    assert len(pipeline_calls) == 1
    assert pipeline_calls[0] == df_agn["object_id"].tolist()
    assert len(plot_hubble_calls) == 3
    assert plot_hubble_calls[2]["filename"] == "hubble_diagram_debiased_no_logf.pdf"
    assert plot_hubble_calls[2]["use_intrinsic_scatter_in_residual_sigma"] is False
    for kwargs in plot_hubble_calls:
        assert "clipped_mask" not in kwargs
    for kwargs in l2500_calls:
        assert "clipped_mask" not in kwargs
    for kwargs in m2500_calls:
        assert "clipped_mask" not in kwargs
    for kwargs in full_residual_calls:
        assert "clipped_mask" not in kwargs
    for kwargs in full_residual_rz_calls:
        assert "clipped_mask" not in kwargs
    for kwargs in blr_calls:
        assert "clipped_mask" not in kwargs
    for kwargs in blr_pdf_calls:
        assert "clipped_mask" not in kwargs
    for kwargs in debias_impact_calls:
        assert "clipped_mask" not in kwargs
    for kwargs in alphaox_calls:
        assert "clipped_mask" not in kwargs


def test_load_agn_data_residuals_csv_cut_remains_available(monkeypatch, tmp_path):
    agn_df = _make_fake_agn_sample(n_agn=3).copy()
    agn_df["object_id"] = ["agn_a", "agn_b", "agn_c"]
    agn_df["mags_mean_u"] = [20.0, 20.1, 20.2]
    agn_df["mags_mean_g"] = [19.8, 19.9, 20.0]
    agn_df["mags_mean_r"] = [19.6, 19.7, 19.8]
    agn_df["mags_mean_i"] = [19.4, 19.5, 19.6]
    agn_df["dlog_amp_blr_u"] = -1.0
    agn_df["dlog_amp_blr_g"] = -1.0
    agn_df["dlog_amp_blr_r"] = -1.0
    agn_df["dlog_amp_blr_i"] = -1.0
    agn_df["log_jitter_u"] = -2.0
    agn_df["log_jitter_g"] = -2.0
    agn_df["log_jitter_r"] = -2.0
    agn_df["log_jitter_i"] = -2.0
    agn_df["dropped_bands"] = [[], [], []]
    agn_df["t_rf_length"] = [120.0, 140.0, 160.0]
    residuals_path = tmp_path / "residuals.csv"
    cut_report_path = tmp_path / "minimal" / "cut_summary.txt"
    pd.DataFrame(
        {
            "object_id": ["agn_a", "agn_b", "agn_c"],
            "residuals": [0.1, 0.2, 0.3],
            "mu_zscore": [1.0, 3.4, 2.5],
        }
    ).to_csv(residuals_path, index=False)

    monkeypatch.setattr(hubble_utils, "resolve_qvc_data_path", lambda path: str(residuals_path) if str(path).endswith("residuals.csv") else str(path))
    monkeypatch.setattr(hubble_utils, "read_quasars_from_hdf5_flat", lambda path: agn_df.copy())
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda df, *args, **kwargs: df)
    monkeypatch.setattr(hubble_plotting, "plot_cut_diagnostics", lambda *args, **kwargs: None)

    filtered_df, all_df = hubble_utils.load_agn_data(
        "dummy.h5",
        magnitude_convention="intrinsic",
        apply_cut=False,
        residuals_sigma_clip=3.0,
        residuals_csv=str(residuals_path),
        lc_info_csv=None,
        cut_report_path=cut_report_path,
        plot_diagnostics=False,
    )

    assert all_df["object_id"].tolist() == ["agn_a", "agn_b", "agn_c"]
    assert filtered_df["object_id"].tolist() == ["agn_a", "agn_c"]
    assert "mu_zscore" not in filtered_df.columns
    assert cut_report_path.is_file()
    cut_report_text = cut_report_path.read_text(encoding="utf-8")
    assert "removed z < 1.5" in cut_report_text
    assert "removed z >= 1.5" in cut_report_text

    diagnostics_path = cut_report_path.parent / "cut_diagnostics_by_z.csv"
    diagnostics_df = pd.read_csv(diagnostics_path)
    assert {
        "removed_z_lt_0p44",
        "removed_z_0p44_to_1",
        "removed_z_1_to_2",
        "removed_z_2_to_3p16",
        "removed_z_gt_3p16",
        "removed_z_lt_1p5",
        "removed_z_ge_1p5",
    }.issubset(diagnostics_df.columns)
    residual_cut = diagnostics_df.loc[
        diagnostics_df["step"] == "residual_sigma_clip"
    ].iloc[0]
    assert residual_cut["removed_z_lt_1p5"] == 1
    assert residual_cut["removed_z_ge_1p5"] == 0


def test_hubble_fit_cli_declares_and_forwards_spectra_sdss_run2d():
    source_path = Path(hubble_fit.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    parser_declared = False
    load_kwargs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                if node.args and isinstance(node.args[0], ast.Constant):
                    if node.args[0].value == "--spectra_sdss_run2d":
                        parser_declared = True
            if isinstance(node.func, ast.Name) and node.func.id == "load_agn_data":
                load_kwargs.update(kw.arg for kw in node.keywords if kw.arg is not None)

    assert parser_declared
    assert "spectra_sdss_run2d" in load_kwargs


def test_hubble_fit_cli_declares_and_forwards_magnitude_convention():
    source_path = Path(hubble_fit.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    parser_has_default = False
    parser_required = None
    parser_choices = None
    forwarded_value = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--magnitude-convention"
            ):
                keywords = {kw.arg: kw.value for kw in node.keywords}
                parser_has_default = "default" in keywords
                parser_required = ast.literal_eval(keywords["required"])
                parser_choices = ast.literal_eval(keywords["choices"])
        if isinstance(node.func, ast.Name) and node.func.id == "load_agn_data":
            for keyword in node.keywords:
                if keyword.arg == "magnitude_convention":
                    forwarded_value = keyword.value

    assert parser_has_default is False
    assert parser_required is True
    assert parser_choices == ["intrinsic", "observed"]
    assert isinstance(forwarded_value, ast.Attribute)
    assert isinstance(forwarded_value.value, ast.Name)
    assert forwarded_value.value.id == "args"
    assert forwarded_value.attr == "magnitude_convention"


def test_hubble_fit_jax_cli_declares_and_forwards_magnitude_convention():
    source_path = SRC / "qvc" / "hubble" / "hubble_fit_jax.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    parser_has_default = False
    parser_required = None
    parser_choices = None
    forwarded_value = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--magnitude-convention"
            ):
                keywords = {kw.arg: kw.value for kw in node.keywords}
                parser_has_default = "default" in keywords
                parser_required = ast.literal_eval(keywords["required"])
                parser_choices = ast.literal_eval(keywords["choices"])
        if isinstance(node.func, ast.Name) and node.func.id == "load_agn_data":
            for keyword in node.keywords:
                if keyword.arg == "magnitude_convention":
                    forwarded_value = keyword.value

    assert parser_has_default is False
    assert parser_required is True
    assert parser_choices == ["intrinsic", "observed"]
    assert isinstance(forwarded_value, ast.Attribute)
    assert isinstance(forwarded_value.value, ast.Name)
    assert forwarded_value.value.id == "args"
    assert forwarded_value.attr == "magnitude_convention"


def test_hubble_fit_cli_declares_only_agn_and_rejects_only_sna_combo():
    source = Path(hubble_fit.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    parser_declared = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                if node.args and isinstance(node.args[0], ast.Constant):
                    if node.args[0].value == "--only_agn":
                        parser_declared = True

    assert parser_declared
    assert "--only_sna and --only_agn cannot be used together." in source


def test_hubble_fit_cli_declares_and_forwards_minimal_plots():
    source_path = Path(hubble_fit.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    parser_declared = False
    run_single_kwargs = set()
    run_all_kwargs = set()
    load_kwargs = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant):
                parser_declared |= node.args[0].value == "--minimal-plots"
        if isinstance(node.func, ast.Name):
            kwargs = {kw.arg for kw in node.keywords if kw.arg is not None}
            if node.func.id == "run_single":
                run_single_kwargs.update(kwargs)
            elif node.func.id == "run_all":
                run_all_kwargs.update(kwargs)
            elif node.func.id == "load_agn_data":
                load_kwargs.update(kwargs)

    assert parser_declared
    assert "minimal_plots" in run_single_kwargs
    assert "minimal_plots" in run_all_kwargs
    assert "plot_diagnostics" in load_kwargs
