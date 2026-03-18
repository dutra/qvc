import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble import hubble_completeness_refactored as hcr
from qvc.hubble import hubble_fit, hubble_likelihood, hubble_model


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
            "f_host_center": f_host,
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
        "log_sigma_UV": log_sigma_uv,
        "log_tau_UV_RF": log_tau_uv,
        "log_sigma_hat0_err": np.full(n_agn, 0.04),
        "log_sigma_UV_std_psd": np.full(n_agn, 0.05),
        "log_tau_UV_RF_std_psd": np.full(n_agn, 0.06),
        "log_sigma_UV_log_tau_UV_RF_cov_psd": np.full(n_agn, 0.001),
    }
    params_arr = hubble_model.agn_model_pack_params(true_params)
    obs_arr, _, pivots = hubble_model.agn_model_pack_obs(obs_dict)
    absolute_mag = hubble_model.M_model_agn(params_arr, obs_arr, pivots)
    mu = cosmo.distmod(z).value
    apparent_mag = absolute_mag + mu + rng.normal(0.0, 0.04, size=n_agn)
    f_host = np.clip(hcr.generalized_sigmoid_fhost(logL, 44.2, 2.8, 0.9), 0.0, 1.0)

    return pd.DataFrame(
        {
            "object_id": [f"agn_{i:03d}" for i in range(n_agn)],
            "z": z,
            "z_err": np.full(n_agn, 0.002),
            "apparent_mag_2500": apparent_mag,
            "apparent_mag_2500_err": np.full(n_agn, 0.04),
            "log_sigma_hat0": log_sigma_hat0,
            "log_sigma_UV": log_sigma_uv,
            "log_tau_UV_RF": log_tau_uv,
            "log_sigma_hat0_err": np.full(n_agn, 0.04),
            "log_sigma_UV_std_psd": np.full(n_agn, 0.05),
            "log_tau_UV_RF_std_psd": np.full(n_agn, 0.06),
            "log_sigma_UV_log_tau_UV_RF_cov_psd": np.full(n_agn, 0.001),
            "delta_m_flux_recal": rng.normal(0.0, 0.02, size=n_agn),
            "f_host_center": f_host,
        }
    )


def _write_fake_sim_file(path, n=2000, seed=321):
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.1, 3.5, size=n)
    logL = rng.uniform(42.2, 46.0, size=n)
    M2500 = 90.0 - 2.5 * logL
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    m2500 = M2500 + cosmo.distmod(z).value
    with h5py.File(path, "w") as handle:
        handle.create_dataset("z", data=z)
        handle.create_dataset("apparent_mag_2500", data=m2500)


def test_fit_fhost_center_model_monotonic_and_bounded():
    df = _make_fake_fhost_df()
    model = hcr.fit_fhost_center_l2500_model(df)
    assert np.isfinite(model["x0"])
    assert np.isfinite(model["k"])
    assert np.isfinite(model["nu"])
    assert np.isfinite(model["sigma_host_logit"])

    grid = np.linspace(42.5, 45.5, 200)
    pred = hcr.predict_fhost_center_from_logL2500(grid, model)
    assert np.all(np.isfinite(pred))
    assert np.all((pred > 0.0) & (pred < 1.0))
    assert np.all(np.diff(pred) <= 1e-8)


def test_sample_fhost_center_from_model_is_bounded_and_luminosity_dependent():
    df = _make_fake_fhost_df()
    model = hcr.fit_fhost_center_l2500_model(df)
    rng = np.random.default_rng(0)
    low = hcr.sample_fhost_center_from_logL2500(np.full(5000, 43.0), model, rng)
    high = hcr.sample_fhost_center_from_logL2500(np.full(5000, 45.2), model, rng)
    assert np.all((low > 0.0) & (low < 1.0))
    assert np.all((high > 0.0) & (high < 1.0))
    assert np.mean(low) > np.mean(high)


def test_completeness3d_shape_and_likelihood_matches_2d_when_host_independent():
    mag_centers = np.linspace(18.5, 24.0, 9)
    z_centers = np.linspace(0.0, 4.0, 7)
    fhost_centers = np.linspace(0.05, 0.95, 5)
    mm, zz = np.meshgrid(mag_centers, z_centers, indexing="ij")
    c2 = np.clip(np.exp(-0.12 * zz) / (1.0 + np.exp((mm - 22.0) / 0.35)), 0.0, 1.0)
    c3 = np.repeat(c2[:, :, None], len(fhost_centers), axis=2)

    comp2 = hcr.Completeness2D(mag_centers, z_centers, c2)
    comp3 = hcr.Completeness3D(mag_centers, z_centers, fhost_centers, c3)

    q = comp3(np.array([[20.0, 21.0]]), np.array([[1.0, 2.0]]), np.array([[0.2, 0.8]]))
    assert q.shape == (1, 2)
    assert np.all((q >= 0.0) & (q <= 1.0))

    m_obs = np.array([20.5, 21.3, 22.1])
    m_model = np.array([20.4, 21.1, 22.0])
    mu_err = np.array([0.15, 0.18, 0.20])
    z = np.array([0.8, 1.5, 2.2])
    f_host = np.array([0.2, 0.5, 0.8])

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
        f_host_center=f_host,
    )

    assert np.allclose(ll2, ll3, rtol=1e-10, atol=1e-10)
    assert np.allclose(blob2, blob3, rtol=1e-10, atol=1e-10)


def test_get_completeness_function_3d_fhost_and_loglikelihood_smoke(tmp_path):
    df_agn = _make_fake_agn_sample_with_fhost()
    df_pantheon = pd.DataFrame(
        {
            "zHD": np.linspace(0.02, 0.8, 12),
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
    agn_fields += ("apparent_mag_2500", "apparent_mag_2500_err", "z", "z_err", "object_id", "f_host_center")
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
        completeness_params=completeness_params,
        z_pivot_agn=hubble_fit.z_pivot_agn,
        agn_calibrators_data=None,
        only_sna=False,
        use_full_cov=False,
    )

    assert np.isfinite(logl)
    assert blob.shape == (2, len(df_agn))
