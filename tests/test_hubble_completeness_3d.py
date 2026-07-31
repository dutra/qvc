import os
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble import hubble_completeness_refactored as hcr
from qvc.hubble import hubble_fit, hubble_likelihood, hubble_model
from qvc.hubble.completeness_mock_catalog import (
    AB_ABSOLUTE_MAG_ZEROPOINT,
    LOG10_MAG_JACOBIAN,
    NU_2500_HZ,
    _configure_shen_paths,
    build_shen_lf,
    log_nu_lnu_to_ab_absolute_magnitude,
    save_mock_catalog,
)
from qvc.hubble.hubble_likelihood import completeness_loglike


def test_configure_shen_paths_overrides_checkout_config(tmp_path):
    shen_config = types.SimpleNamespace(
        homepath="/stale/quasarlf/pubtools/",
        datapath="/stale/quasarlf/pubtools/data/",
    )

    obdata_path = _configure_shen_paths(shen_config, tmp_path)
    expected_homepath = f"{tmp_path.resolve()}{os.sep}"

    assert shen_config.homepath == expected_homepath
    assert shen_config.datapath == f"{expected_homepath}data{os.sep}"
    assert obdata_path == f"{expected_homepath}obdata_copy{os.sep}"


def test_build_shen_lf_uses_extinction_convolved_physical_2500_channel(
    tmp_path, monkeypatch
):
    """Gold test that the Shen mock parent is the observed 2500 A LF."""
    calls = []
    log_nu_lnu = np.array([44.0, 45.0])
    log_phi_dex = np.array([-5.0, -6.0])

    def fake_return_qlf_in_band(redshift, nu, model):
        calls.append((redshift, nu, model))
        return log_nu_lnu, log_phi_dex

    fake_utilities = types.ModuleType("utilities")
    fake_utilities.return_qlf_in_band = fake_return_qlf_in_band
    monkeypatch.setitem(sys.modules, "utilities", fake_utilities)

    phi_log10, m_grid, z_bins = build_shen_lf(tmp_path)

    assert len(calls) == len(z_bins) == 40
    assert all(np.isclose(nu, NU_2500_HZ) and model == "B" for _, nu, model in calls)
    np.testing.assert_allclose(
        phi_log10,
        np.tile(log_phi_dex + LOG10_MAG_JACOBIAN, (len(z_bins), 1)),
    )
    np.testing.assert_allclose(
        m_grid,
        AB_ABSOLUTE_MAG_ZEROPOINT
        - 2.5 * (log_nu_lnu - np.log10(NU_2500_HZ)),
    )


def test_log_nu_lnu_to_ab_absolute_magnitude_gold_value():
    target_magnitude = -25.0
    log_lnu = (AB_ABSOLUTE_MAG_ZEROPOINT - target_magnitude) / 2.5
    log_nu_lnu = log_lnu + np.log10(NU_2500_HZ)
    np.testing.assert_allclose(
        log_nu_lnu_to_ab_absolute_magnitude(log_nu_lnu, NU_2500_HZ),
        target_magnitude,
        atol=1e-12,
    )


def _build_pivot_context(df_agn):
    z = df_agn["z"].to_numpy(dtype=float)
    return hubble_model.build_agn_pivot_context(
        df_agn,
        z_range=(float(np.min(z)), float(np.max(z))),
    )


def test_completeness_loglike_includes_bright_gaussian_tail():
    mag_centers = np.linspace(18.5, 24.0, 60)
    z_centers = np.linspace(0.0, 4.0, 20)
    completeness = hcr.Completeness2D(
        mag_centers,
        z_centers,
        np.ones((mag_centers.size, z_centers.size)),
    )

    _, blob = completeness_loglike(
        m_obs=np.array([17.5]),
        m_obs_err=np.array([0.05]),
        m_model=np.array([17.5]),
        mu_err=np.array([0.3]),
        z=np.array([1.0]),
        completeness_model=completeness,
        m_grid=mag_centers,
    )

    np.testing.assert_allclose(blob[0], 1.0, atol=1e-4)
    np.testing.assert_allclose(blob[1], 0.0, atol=1e-4)
    np.testing.assert_allclose(blob[2], 0.3, atol=1e-4)


def test_selection_correction_matches_truncated_normal_and_recovers_parent_mean():
    """Regression test the full correction against a known magnitude-limit solution."""

    class HardMagnitudeLimit:
        mode = "2d"

        def __init__(self, limit):
            self.limit = float(limit)

        def __call__(self, mag, z):
            mag, z = np.broadcast_arrays(
                np.asarray(mag, dtype=float),
                np.asarray(z, dtype=float),
            )
            # Half weight at the discontinuity is the trapezoid-rule convention.
            return np.where(mag < self.limit, 1.0, np.where(mag == self.limit, 0.5, 0.0))

    m_model = 22.3
    sigma = 0.45
    m_limit = 22.0
    mag_grid = np.linspace(18.5, 24.5, 6001)

    log_z, blob = completeness_loglike(
        m_obs=np.array([m_model]),
        m_obs_err=np.array([0.05]),
        m_model=np.array([m_model]),
        mu_err=np.array([sigma]),
        z=np.array([1.2]),
        completeness_model=HardMagnitudeLimit(m_limit),
        m_grid=mag_grid,
    )

    alpha = (m_limit - m_model) / sigma
    expected_z = ndtr(alpha)
    inverse_mills = np.exp(-0.5 * alpha**2) / (np.sqrt(2.0 * np.pi) * expected_z)
    expected_bias = -sigma * inverse_mills
    expected_sigma = sigma * np.sqrt(1.0 - alpha * inverse_mills - inverse_mills**2)

    np.testing.assert_allclose(np.exp(log_z), expected_z, rtol=2e-6)
    np.testing.assert_allclose(blob[0, 0], expected_z, rtol=2e-6)
    np.testing.assert_allclose(blob[1, 0], expected_bias, atol=2e-6)
    np.testing.assert_allclose(blob[2, 0], expected_sigma, atol=2e-6)

    selected_mean = m_model + blob[1, 0]
    corrected_mean = selected_mean - blob[1, 0]
    np.testing.assert_allclose(corrected_mean, m_model, atol=1e-12)


def _make_fake_fhost_df(n=200, seed=123):
    rng = np.random.default_rng(seed)
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    logL = np.linspace(42.3, 45.6, n)
    z = np.linspace(0.4, 2.8, n)
    f_mean = hcr.generalized_sigmoid_fhost(logL, 44.2, 2.8, 0.9)
    f_host = np.clip(
        hcr.expit(hcr.logit(np.clip(f_mean, 1e-3, 1.0 - 1e-3)) + rng.normal(0.0, 0.45, size=n)),
        0.0,
        1.0,
    )
    M2500 = 90.0 - 2.5 * logL
    m2500 = M2500 + cosmo.distmod(z).value
    return pd.DataFrame(
        {
            "object_id": [f"agn_{i:04d}" for i in range(n)],
            "z": z,
            "apparent_mag_2500": m2500,
            "m_2500_dereddened": m2500,
            "m_2500_attenuated_model": m2500,
            hcr.COMPLETENESS_MAG_COL: m2500,
            "f_host_2500": f_host,
            "f_host_2500_psf": f_host,
        }
    )


def _make_fake_agn_sample_with_fhost(n_agn=24, seed=123):
    rng = np.random.default_rng(seed)
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)

    z = np.linspace(0.5, 2.2, n_agn)
    log_sigma_uv = rng.normal(-0.8, 0.12, size=n_agn)
    log_tau_uv = 2.7 + 0.35 * (z - np.mean(z)) + rng.normal(0.0, 0.08, size=n_agn)
    log_sigma_hat0 = log_sigma_uv - 1.4 + rng.normal(0.0, 0.05, size=n_agn)
    logL = np.linspace(42.8, 45.4, n_agn)
    M2500_true = 90.0 - 2.5 * logL

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
    object_ids = [f"agn_{i:03d}" for i in range(n_agn)]
    pivot_df = pd.DataFrame(
        {
            "object_id": object_ids,
            "z": z,
            **obs_dict,
        }
    )
    pivot_context = _build_pivot_context(pivot_df)
    params_arr = hubble_model.agn_model_pack_params(true_params)
    obs_arr, _, pivots = hubble_model.agn_model_pack_obs(
        obs_dict,
        pivot_context=pivot_context,
    )
    absolute_mag = hubble_model.M_model_agn(params_arr, obs_arr, pivots)
    mu = cosmo.distmod(z).value
    apparent_mag = absolute_mag + mu + rng.normal(0.0, 0.04, size=n_agn)
    f_host = np.clip(hcr.generalized_sigmoid_fhost(logL, 44.2, 2.8, 0.9), 0.0, 1.0)

    return pd.DataFrame(
        {
            "object_id": object_ids,
            "z": z,
            "z_err": np.full(n_agn, 0.002),
            "apparent_mag_2500": apparent_mag,
            "apparent_mag_2500_err": np.full(n_agn, 0.04),
            "m_2500_dereddened": apparent_mag,
            "m_2500_dereddened_err": np.full(n_agn, 0.04),
            "m_2500_attenuated_model": apparent_mag + 0.35,
            "m_2500_attenuated_model_err": np.full(n_agn, 0.06),
            hcr.COMPLETENESS_MAG_COL: apparent_mag,
            hcr.COMPLETENESS_MAG_ERR_COL: np.full(n_agn, 0.04),
            "log_sigma_hat0": log_sigma_hat0,
            "log_sigma_uv": log_sigma_uv,
            "log_tau_uv_rf": log_tau_uv,
            "log_sigma_hat0_err": np.full(n_agn, 0.04),
            "log_sigma_uv_std_psd": np.full(n_agn, 0.05),
            "log_tau_uv_rf_std_psd": np.full(n_agn, 0.06),
            "log_sigma_uv_log_tau_uv_rf_cov_psd": np.full(n_agn, 0.001),
            "delta_m_flux_recal": rng.normal(0.0, 0.02, size=n_agn),
            "f_host_2500": f_host,
            "f_host_2500_psf": f_host,
        }
    )


def _make_fake_agn_sample_with_fhost_alpha(n_agn=32, seed=123, alpha_center=-1.25):
    rng = np.random.default_rng(seed)
    df = _make_fake_agn_sample_with_fhost(n_agn=n_agn, seed=seed)
    trend = 0.08 * (df["z"].to_numpy(dtype=float) - float(df["z"].mean()))
    df["alpha_lambda"] = alpha_center + trend + rng.normal(0.0, 0.06, size=n_agn)
    return df


def _write_fake_sim_file(
    path,
    n=2000,
    seed=321,
    include_alpha=False,
    alpha_center=-1.2,
    include_fhost=False,
):
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.1, 3.5, size=n)
    logL = rng.uniform(42.2, 46.0, size=n)
    M2500 = 90.0 - 2.5 * logL
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    m2500 = M2500 + cosmo.distmod(z).value
    with h5py.File(path, "w") as handle:
        handle.create_dataset("z", data=z)
        handle.create_dataset("apparent_mag_2500", data=m2500)
        if include_alpha:
            alpha_lambda = alpha_center + 0.12 * (z - np.mean(z)) + rng.normal(0.0, 0.08, size=n)
            handle.create_dataset("alpha_lambda", data=alpha_lambda)
        if include_fhost:
            f_host = np.clip(0.25 + 0.08 * (z - np.mean(z)) + rng.normal(0.0, 0.05, size=n), 0.01, 0.9)
            handle.create_dataset("f_host_2500_psf", data=f_host)


def test_completeness_2d_plot_smoothing_is_display_only(tmp_path):
    df_agn = _make_fake_agn_sample_with_fhost(n_agn=36)
    sim_file = tmp_path / "mock2d.h5"
    _write_fake_sim_file(sim_file, n=1200)

    comp_no_plot, mag_centers, z_centers, *_ = hcr.get_completeness_function_2d(
        df_agn,
        sim_file=str(sim_file),
        n_mag_bins=12,
        n_z_bins=10,
        plot=False,
    )
    comp_with_plot, mag_centers_plot, z_centers_plot, *_ = hcr.get_completeness_function_2d(
        df_agn,
        sim_file=str(sim_file),
        n_mag_bins=12,
        n_z_bins=10,
        plot=True,
        plot_path=str(tmp_path),
    )

    mag_grid, z_grid = np.meshgrid(mag_centers, z_centers, indexing="ij")
    np.testing.assert_allclose(mag_centers_plot, mag_centers)
    np.testing.assert_allclose(z_centers_plot, z_centers)
    np.testing.assert_allclose(comp_with_plot(mag_grid, z_grid), comp_no_plot(mag_grid, z_grid))
    assert (tmp_path / "completeness" / "completeness_map.pdf").exists()


def test_fit_fhost_2500_model_monotonic_and_bounded():
    df = _make_fake_fhost_df()
    model = hcr.fit_fhost_2500_l2500_model(df)
    assert np.isfinite(model["x0"])
    assert np.isfinite(model["k"])
    assert np.isfinite(model["nu"])
    assert np.isfinite(model["sigma_host_logit"])

    grid = np.linspace(42.5, 45.5, 200)
    pred = hcr.predict_fhost_2500_from_logL2500(grid, model)
    assert np.all(np.isfinite(pred))
    assert np.all((pred > 0.0) & (pred < 1.0))
    assert np.all(np.diff(pred) <= 1e-8)


def test_sample_fhost_2500_from_model_is_bounded_and_luminosity_dependent():
    df = _make_fake_fhost_df()
    model = hcr.fit_fhost_2500_l2500_model(df)
    rng = np.random.default_rng(0)
    low = hcr.sample_fhost_2500_from_logL2500(np.full(5000, 43.0), model, rng)
    high = hcr.sample_fhost_2500_from_logL2500(np.full(5000, 45.2), model, rng)
    assert np.all((low > 0.0) & (low < 1.0))
    assert np.all((high > 0.0) & (high < 1.0))
    assert np.mean(low) > np.mean(high)


def test_completeness3d_shape_and_likelihood_matches_2d_when_host_independent():
    mag_centers = np.linspace(18.5, 24.0, 9)
    z_centers = np.linspace(0.0, 4.0, 7)
    fhost_centers = np.linspace(0.05, 0.95, 5)
    alpha_centers = np.linspace(-2.4, -0.8, 4)
    mm, zz = np.meshgrid(mag_centers, z_centers, indexing="ij")
    c2 = np.clip(np.exp(-0.12 * zz) / (1.0 + np.exp((mm - 22.0) / 0.35)), 0.0, 1.0)
    c3 = np.repeat(c2[:, :, None], len(fhost_centers), axis=2)
    c4 = np.repeat(c3[:, :, :, None], len(alpha_centers), axis=3)

    comp2 = hcr.Completeness2D(mag_centers, z_centers, c2)
    comp3 = hcr.Completeness3D(mag_centers, z_centers, fhost_centers, c3)
    comp4 = hcr.Completeness4D(mag_centers, z_centers, fhost_centers, alpha_centers, c4)

    q = comp3(np.array([[20.0, 21.0]]), np.array([[1.0, 2.0]]), np.array([[0.2, 0.8]]))
    assert q.shape == (1, 2)
    assert np.all((q >= 0.0) & (q <= 1.0))
    q4 = comp4(
        np.array([[20.0, 21.0]]),
        np.array([[1.0, 2.0]]),
        np.array([[0.2, 0.8]]),
        np.array([[-1.9, -1.2]]),
    )
    assert q4.shape == (1, 2)
    assert np.all((q4 >= 0.0) & (q4 <= 1.0))

    m_obs = np.array([20.5, 21.3, 22.1])
    m_model = np.array([20.4, 21.1, 22.0])
    mu_err = np.array([0.15, 0.18, 0.20])
    z = np.array([0.8, 1.5, 2.2])
    f_host = np.array([0.2, 0.5, 0.8])
    alpha_lambda = np.array([-1.8, -1.5, -1.1])

    ll2, blob2 = hubble_likelihood.completeness_loglike(
        m_obs=m_obs,
        m_obs_err=np.full_like(m_obs, 0.05),
        m_model=m_model,
        mu_err=mu_err,
        z=z,
        completeness_model=comp2,
        m_grid=mag_centers,
    )
    ll3, blob3 = hubble_likelihood.completeness_loglike(
        m_obs=m_obs,
        m_obs_err=np.full_like(m_obs, 0.05),
        m_model=m_model,
        mu_err=mu_err,
        z=z,
        completeness_model=comp3,
        m_grid=mag_centers,
        f_host_2500_psf=f_host,
    )
    ll4, blob4 = hubble_likelihood.completeness_loglike(
        m_obs=m_obs,
        m_obs_err=np.full_like(m_obs, 0.05),
        m_model=m_model,
        mu_err=mu_err,
        z=z,
        completeness_model=comp4,
        m_grid=mag_centers,
        f_host_2500_psf=f_host,
        alpha_lambda=alpha_lambda,
    )

    assert np.allclose(ll2, ll3, rtol=1e-10, atol=1e-10)
    assert np.allclose(blob2, blob3, rtol=1e-10, atol=1e-10)
    assert np.allclose(ll2, ll4, rtol=1e-10, atol=1e-10)
    assert np.allclose(blob2, blob4, rtol=1e-10, atol=1e-10)


def test_completeness_loglike_caches_detection_grid_across_parameter_calls():
    mag_centers = np.linspace(18.5, 24.0, 9)
    z_centers = np.linspace(0.0, 4.0, 7)
    mm, zz = np.meshgrid(mag_centers, z_centers, indexing="ij")
    c2 = np.clip(np.exp(-0.12 * zz) / (1.0 + np.exp((mm - 22.0) / 0.35)), 0.0, 1.0)
    base = hcr.Completeness2D(mag_centers, z_centers, c2)

    class CountingCompleteness:
        mode = "2d"

        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            return self.wrapped(*args, **kwargs)

    comp = CountingCompleteness(base)
    m_obs = np.array([20.5, 21.3, 22.1])
    m_model = np.array([20.4, 21.1, 22.0])
    mu_err = np.array([0.15, 0.18, 0.20])
    z = np.array([0.8, 1.5, 2.2])

    ll1, blob1 = hubble_likelihood.completeness_loglike(
        m_obs=m_obs,
        m_obs_err=np.full_like(m_obs, 0.05),
        m_model=m_model,
        mu_err=mu_err,
        z=z,
        completeness_model=comp,
        m_grid=mag_centers,
    )
    ll2, blob2 = hubble_likelihood.completeness_loglike(
        m_obs=m_obs,
        m_obs_err=np.full_like(m_obs, 0.05),
        m_model=m_model + 0.05,
        mu_err=mu_err,
        z=z,
        completeness_model=comp,
        m_grid=mag_centers,
    )

    assert comp.calls == 1
    assert np.isfinite(ll1)
    assert np.isfinite(ll2)
    assert blob1.shape == blob2.shape == (3, len(m_obs))


def test_log_likelihood_does_not_use_completeness_smoothing_as_extra_scatter(monkeypatch):
    df_agn = _make_fake_agn_sample_with_fhost(n_agn=4)
    df_pantheon = pd.DataFrame(
        {
            "zHD": np.linspace(0.02, 0.8, 8),
            "zHEL": np.linspace(0.02, 0.8, 8),
            "m_b_corr": np.linspace(15.0, 18.0, 8),
            "IS_CALIBRATOR": np.zeros(8, dtype=int),
            "CEPH_DIST": np.full(8, -9.0),
            "MU_SH0ES_ERR_DIAG": np.full(8, 0.08),
        }
    )
    mag_centers = np.linspace(18.5, 24.0, 9)
    z_centers = np.linspace(0.0, 4.0, 7)
    c2 = np.ones((len(mag_centers), len(z_centers)))
    completeness_params = (
        hcr.Completeness2D(mag_centers, z_centers, c2),
        mag_centers,
        z_centers,
        0.5,
        0.1,
        99.0,
    )
    captured = {}

    def fake_completeness_loglike(*args, **kwargs):
        captured["sigma_completeness"] = kwargs["sigma_completeness"]
        n = len(kwargs["z"])
        return 0.0, np.ones((3, n), dtype=float)

    monkeypatch.setattr(hubble_likelihood, "completeness_loglike", fake_completeness_loglike)

    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    agn_fields = hubble_model.agn_model_req_obs + hubble_model.agn_model_req_errs
    agn_fields += ("apparent_mag_2500", "apparent_mag_2500_err", "z", "z_err", "object_id")
    agn_fields += (hcr.COMPLETENESS_MAG_COL, hcr.COMPLETENESS_MAG_ERR_COL)
    agn_data = {col: df_agn[col].to_numpy() for col in agn_fields}
    pantheon_data = {col: df_pantheon[col].to_numpy() for col in df_pantheon.columns}
    pivot_context = _build_pivot_context(df_agn)

    logl, _ = hubble_likelihood.log_likelihood(
        theta,
        agn_data=agn_data,
        pantheon_data=pantheon_data,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM",
        completeness_params=completeness_params,
        z_pivot_agn=hubble_fit.z_pivot_agn,
        agn_pivot_context=pivot_context,
        agn_calibrators_data=None,
        only_sna=False,
        use_full_cov=False,
    )

    assert np.isfinite(logl)
    assert captured["sigma_completeness"] == 0.0


def test_completeness_callables_return_zero_for_nonfinite_queries():
    mag_centers = np.linspace(18.5, 24.0, 5)
    z_centers = np.linspace(0.1, 3.0, 4)
    fhost_centers = np.linspace(0.05, 0.95, 3)
    alpha_centers = np.linspace(-2.5, -0.5, 3)
    c2 = np.ones((len(mag_centers), len(z_centers)))
    c3 = np.ones((len(mag_centers), len(z_centers), len(fhost_centers)))
    c4 = np.ones((len(mag_centers), len(z_centers), len(fhost_centers), len(alpha_centers)))

    comp2 = hcr.Completeness2D(mag_centers, z_centers, c2)
    comp3 = hcr.Completeness3D(mag_centers, z_centers, fhost_centers, c3)
    comp4 = hcr.Completeness4D(mag_centers, z_centers, fhost_centers, alpha_centers, c4)

    np.testing.assert_allclose(comp2([20.0, np.nan], [1.0, 1.0]), [1.0, 0.0])
    np.testing.assert_allclose(comp3([20.0, np.nan], [1.0, 1.0], [0.2, 0.2]), [1.0, 0.0])
    np.testing.assert_allclose(
        comp4([20.0, np.nan], [1.0, 1.0], [0.2, 0.2], [-1.5, -1.5]),
        [1.0, 0.0],
    )


def test_get_completeness_function_4d_fhost_alpha_uses_mock_alpha_dataset(tmp_path):
    df_agn = _make_fake_agn_sample_with_fhost_alpha(alpha_center=-1.15)
    df_pre = _make_fake_agn_sample_with_fhost_alpha(n_agn=64, seed=456, alpha_center=-0.95)
    sim_file = tmp_path / "mock4d_alpha.h5"
    _write_fake_sim_file(sim_file, include_alpha=True, alpha_center=-0.95, include_fhost=True)

    completeness_params = hcr.get_completeness_function_4d_fhost_alpha(
        df_agn,
        sim_file=str(sim_file),
        plot=False,
        n_mag_bins=8,
        n_z_bins=8,
        n_fhost_bins=4,
        n_alpha_bins=5,
        sigma_mag=0.25,
        sigma_z_abs=0.25,
        sigma_fhost=0.2,
        sigma_alpha=0.4,
        fit_logL_max=99.0,
        df_agn_fhost_population=df_pre,
    )

    assert completeness_params[0].mode == "4d_fhost_alpha"
    alpha_model = completeness_params[-1]
    assert alpha_model["source"] == "mock_h5_dataset:alpha_lambda"
    assert alpha_model["n_mock"] > 0
    assert abs(alpha_model["alpha_mean"] - (-0.95)) < 0.25
    host_model = completeness_params[-2]
    assert host_model["source"] == "mock_h5_dataset:f_host_2500_psf"
    assert host_model["observed_fit_source"] == "precut_f_host_2500_psf_vs_l2500"
    assert host_model["n_fit"] == len(df_pre)
    assert host_model["n_mock"] > 0


def test_get_completeness_function_4d_fhost_alpha_falls_back_to_observed_alpha(tmp_path):
    df_agn = _make_fake_agn_sample_with_fhost_alpha(alpha_center=-0.85)
    sim_file = tmp_path / "mock4d_no_alpha.h5"
    _write_fake_sim_file(sim_file, include_alpha=False)

    completeness_params = hcr.get_completeness_function_4d_fhost_alpha(
        df_agn,
        sim_file=str(sim_file),
        plot=False,
        n_mag_bins=8,
        n_z_bins=8,
        n_fhost_bins=4,
        n_alpha_bins=5,
        sigma_mag=0.25,
        sigma_z_abs=0.25,
        sigma_fhost=0.2,
        sigma_alpha=0.4,
    )

    alpha_model = completeness_params[-1]
    assert alpha_model["source"] == "observed_alpha_lambda"
    assert alpha_model["n_fit"] == len(df_agn)
    assert abs(alpha_model["alpha_mean"] - np.median(df_agn["alpha_lambda"])) < 0.05


def test_save_mock_catalog_persists_alpha_lambda(tmp_path):
    out = tmp_path / "mock_with_alpha.h5"
    z = np.array([0.5, 1.0, 1.5])
    m_i = np.array([20.1, 21.2, 22.3])
    m2500 = np.array([19.9, 21.0, 22.1])
    alpha_lambda = np.array([-1.2, -1.4, -1.6])

    save_mock_catalog(
        out,
        z,
        m_i,
        m2500,
        alpha_lambda_all=alpha_lambda,
        alpha_nu_parent_mean=-0.5,
        alpha_nu_parent_sigma=0.3,
    )

    with h5py.File(out, "r") as handle:
        np.testing.assert_allclose(handle["alpha_lambda"][:], alpha_lambda)
        np.testing.assert_allclose(handle["alpha_nu"][:], -alpha_lambda - 2.0)
        assert handle.attrs["alpha_lambda_parent_mean"] == -1.5
        assert handle.attrs["alpha_lambda_parent_sigma"] == 0.3
        assert handle.attrs["thinning_probability"] == 1.0
        assert handle.attrs["mock_count_scale"] == 1.0


def test_generate_fresh_completeness_uses_full_area_without_thinning(
    tmp_path,
    monkeypatch,
):
    calls = {}
    z_all = np.array([0.5, 1.0])
    m_all = np.array([20.0, 21.0])
    m_2500_all = np.array([19.8, 20.8])
    alpha_lambda_all = np.array([-1.5, -1.6])

    monkeypatch.setattr(
        hubble_fit,
        "build_shen_lf",
        lambda _: (
            np.zeros((2, 2)),
            np.array([-24.0, -23.0]),
            np.array([0.0, 1.0]),
        ),
    )

    def fake_mock_m_per_zbin(
        phi_log10,
        m_grid,
        z_bins,
        area_deg2,
        *args,
        thinning_probability,
        **kwargs,
    ):
        calls["mock_area_deg2"] = area_deg2
        calls["mock_thinning_probability"] = thinning_probability
        return (
            [],
            np.array([]),
            [],
            np.array([]),
            z_all,
            m_all,
            m_2500_all,
            np.array([0, 0]),
            alpha_lambda_all,
        )

    def fake_save_mock_catalog(
        output_path,
        z,
        m,
        m_2500,
        *,
        thinning_probability,
        area_deg2,
        **kwargs,
    ):
        calls["save_thinning_probability"] = thinning_probability
        calls["save_area_deg2"] = area_deg2

    monkeypatch.setattr(hubble_fit, "mock_m_per_zbin", fake_mock_m_per_zbin)
    monkeypatch.setattr(hubble_fit, "save_mock_catalog", fake_save_mock_catalog)

    output_path = hubble_fit.generate_fresh_completeness_sim_file(
        tmp_path,
        area_deg2=74.1,
    )

    assert output_path.endswith("completeness/mock_completeness_catalog_fresh.h5")
    assert calls == {
        "mock_area_deg2": 74.1,
        "mock_thinning_probability": 1.0,
        "save_thinning_probability": 1.0,
        "save_area_deg2": 74.1,
    }


def test_get_completeness_function_3d_fhost_and_loglikelihood_smoke(tmp_path):
    df_agn = _make_fake_agn_sample_with_fhost()
    df_pantheon = pd.DataFrame(
        {
            "zHD": np.linspace(0.02, 0.8, 12),
            "zHEL": np.linspace(0.02, 0.8, 12),
            "m_b_corr": np.linspace(15.0, 18.0, 12),
            "IS_CALIBRATOR": np.zeros(12, dtype=int),
            "CEPH_DIST": np.full(12, -9.0),
            "MU_SH0ES_ERR_DIAG": np.full(12, 0.08),
        }
    )
    sim_file = tmp_path / "mock3d.h5"
    _write_fake_sim_file(sim_file)

    completeness_params = hcr.get_completeness_function_3d_fhost(df_agn, sim_file=str(sim_file), plot=False)
    assert completeness_params[0].mode == "3d_fhost"

    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)

    agn_fields = hubble_model.agn_model_req_obs + hubble_model.agn_model_req_errs
    agn_fields += ("apparent_mag_2500", "apparent_mag_2500_err", "z", "z_err", "object_id", "f_host_2500_psf")
    agn_fields += (hcr.COMPLETENESS_MAG_COL, hcr.COMPLETENESS_MAG_ERR_COL)
    agn_data = {col: df_agn[col].to_numpy() for col in agn_fields}
    pantheon_data = {col: df_pantheon[col].to_numpy() for col in df_pantheon.columns}
    pivot_context = _build_pivot_context(df_agn)

    logl, blob = hubble_likelihood.log_likelihood(
        theta,
        agn_data=agn_data,
        pantheon_data=pantheon_data,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM",
        completeness_params=completeness_params,
        z_pivot_agn=hubble_fit.z_pivot_agn,
        agn_pivot_context=pivot_context,
        agn_calibrators_data=None,
        only_sna=False,
        use_full_cov=False,
    )

    assert np.isfinite(logl)
    assert blob.shape == (3, len(df_agn))


def test_get_completeness_function_3d_ignores_legacy_fhost_column(tmp_path):
    df_agn = _make_fake_agn_sample_with_fhost()
    df_agn["f_host_2500"] = np.nan
    sim_file = tmp_path / "mock3d_legacy_fhost_ignored.h5"
    _write_fake_sim_file(sim_file, include_fhost=False)

    completeness_params = hcr.get_completeness_function_3d_fhost(
        df_agn,
        sim_file=str(sim_file),
        plot=False,
        n_mag_bins=8,
        n_z_bins=8,
        n_fhost_bins=4,
        sigma_mag=0.25,
        sigma_z_abs=0.25,
        sigma_fhost=0.2,
        fit_logL_max=99.0,
    )

    assert completeness_params[0].mode == "3d_fhost"
    assert completeness_params[-1]["observed_fit_source"] == "fit_sample_f_host_2500_psf_vs_l2500"


def test_get_completeness_function_3d_fhost_fits_host_population_on_precut_sample(tmp_path):
    df_postcut = _make_fake_agn_sample_with_fhost(n_agn=12, seed=123)
    df_precut = _make_fake_agn_sample_with_fhost(n_agn=48, seed=456)
    sim_file = tmp_path / "mock3d_no_fhost.h5"
    _write_fake_sim_file(sim_file, include_fhost=False)

    completeness_params = hcr.get_completeness_function_3d_fhost(
        df_postcut,
        sim_file=str(sim_file),
        plot=False,
        n_mag_bins=8,
        n_z_bins=8,
        n_fhost_bins=4,
        sigma_mag=0.25,
        sigma_z_abs=0.25,
        sigma_fhost=0.2,
        fit_logL_max=99.0,
        df_agn_fhost_population=df_precut,
    )

    host_model = completeness_params[-1]
    assert host_model["source"] == "observed_fhost_model"
    assert host_model["observed_fit_source"] == "precut_f_host_2500_psf_vs_l2500"
    assert host_model["n_observed_population"] == len(df_precut)
    assert host_model["n_fit"] == len(df_precut)
