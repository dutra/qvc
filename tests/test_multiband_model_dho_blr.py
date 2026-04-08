import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.light_curve.multiband_model_dho_blr import (
    ContiBLRFluxLinearized_SHO_Wrapper,
    ContiBLR_SHO_Wrapper,
    OverdampedSHOBaseQS,
    TwoStageFluxMixDisplayModel,
    make_multiband_dho_blr_flux_linearized_model,
    make_multiband_dho_blr_model,
    qs_psd,
    relative_flux_std_to_mag_std,
    relative_flux_to_mag_residual,
)


def test_make_multiband_dho_blr_model_random_data():
    rng = np.random.default_rng(42)
    n_band = 3
    n_obs = 36

    t = np.sort(rng.uniform(0.0, 100.0, size=n_obs))
    band = rng.integers(0, n_band, size=n_obs)
    yerr = rng.uniform(0.01, 0.05, size=n_obs)
    y = rng.normal(0.0, yerr, size=n_obs)

    model = make_multiband_dho_blr_model(
        X=(jnp.asarray(t), jnp.asarray(band)),
        y=jnp.asarray(y),
        yerr=jnp.asarray(yerr),
        n_band=n_band,
        zero_mean=True,
        has_jitter=True,
    )

    assert model is not None
    assert hasattr(model, "log_prob")
    assert hasattr(model, "pred")

    tau_fast = jnp.asarray(rng.uniform(5.0, 15.0, size=n_band))
    tau_slow = tau_fast + jnp.asarray(rng.uniform(20.0, 80.0, size=n_band))
    base_kernel = OverdampedSHOBaseQS(tau_fast=tau_fast, tau_slow=tau_slow)
    params = {
        "amp_cont": jnp.asarray(rng.uniform(0.1, 0.4, size=n_band)),
        "amp_blr": jnp.asarray(rng.uniform(0.01, 0.08, size=n_band)),
        "amp_blr2": jnp.asarray(rng.uniform(0.01, 0.08, size=n_band)),
        "lag_disk": jnp.asarray(rng.uniform(0.5, 5.0, size=n_band)),
        "lag_blr": jnp.asarray(rng.uniform(5.0, 30.0, size=n_band)),
        "lag_blr2": jnp.asarray(rng.uniform(15.0, 60.0, size=n_band)),
    }
    wrapper = ContiBLR_SHO_Wrapper(kernel=base_kernel, params=params)

    design = base_kernel.design_matrix()
    stationary = base_kernel.stationary_covariance()
    transition = wrapper.transition_matrix((jnp.array(0.0), jnp.array(1)), (jnp.array(3.0), jnp.array(1)))
    obs = wrapper.observation_model((jnp.array(0.0), jnp.array(1)))

    assert design.shape == (2 * n_band, 2 * n_band)
    assert stationary.shape == (2 * n_band, 2 * n_band)
    assert transition.shape == (2 * n_band, 2 * n_band)
    assert obs.shape == (2 * n_band,)
    assert np.all(np.isfinite(np.asarray(design)))
    assert np.all(np.isfinite(np.asarray(stationary)))
    assert np.all(np.isfinite(np.asarray(transition)))
    assert np.all(np.isfinite(np.asarray(obs)))
    assert not np.allclose(
        np.asarray(stationary)[:n_band, n_band:],
        0.0,
        atol=1e-10,
    )


def test_exact_overdamped_sho_base_kernel_matches_unit_rms_covariance():
    tau_fast = jnp.asarray([10.0], dtype=float)
    tau_slow = jnp.asarray([100.0], dtype=float)
    kernel = OverdampedSHOBaseQS(tau_fast=tau_fast, tau_slow=tau_slow)

    P = np.asarray(kernel.stationary_covariance(), dtype=float)
    h0 = np.asarray(kernel.observation_model((jnp.array(0.0), jnp.array(0))), dtype=float)
    phi = np.asarray(
        kernel.transition_matrix(
            (jnp.array(0.0), jnp.array(0)),
            (jnp.array(25.0), jnp.array(0)),
        ),
        dtype=float,
    )
    cov0 = float(h0 @ P @ h0)
    cov25 = float(h0 @ phi @ P @ h0)
    rho = float(tau_fast[0] / tau_slow[0])
    expected25 = (
        np.exp(-25.0 / float(tau_slow[0]))
        - rho * np.exp(-25.0 / float(tau_fast[0]))
    ) / (1.0 - rho)

    assert np.isclose(cov0, 1.0, rtol=1e-6, atol=1e-8)
    assert np.isclose(cov25, expected25, rtol=1e-6, atol=1e-8)


def test_linearized_wrapper_matches_relflux_wrapper_shape():
    rng = np.random.default_rng(0)
    n_band = 3
    tau_fast = jnp.asarray(rng.uniform(5.0, 15.0, size=n_band))
    tau_slow = tau_fast + jnp.asarray(rng.uniform(20.0, 80.0, size=n_band))
    base_kernel = OverdampedSHOBaseQS(tau_fast=tau_fast, tau_slow=tau_slow)
    amp_cont_relflux = jnp.asarray(rng.uniform(0.08, 0.2, size=n_band))
    amp_bc_relflux = jnp.asarray(rng.uniform(0.01, 0.05, size=n_band))
    amp_blr_relflux = jnp.asarray(rng.uniform(0.01, 0.08, size=n_band))

    params_old = {
        "amp_cont": amp_cont_relflux,
        "amp_bc": amp_bc_relflux,
        "amp_blr": amp_blr_relflux,
        "amp_blr2": jnp.zeros(n_band, dtype=float),
        "lag_disk": jnp.asarray(rng.uniform(0.5, 5.0, size=n_band)),
        "lag_bc": jnp.asarray(rng.uniform(1.0, 8.0, size=n_band)),
        "lag_blr": jnp.asarray(rng.uniform(5.0, 30.0, size=n_band)),
        "lag_blr2": jnp.zeros(n_band, dtype=float),
    }
    params_new = {
        "amp_cont_relflux": amp_cont_relflux,
        "amp_bc_relflux": amp_bc_relflux,
        "amp_blr_relflux": amp_blr_relflux,
        "amp_blr2_relflux": jnp.zeros(n_band, dtype=float),
        "lag_disk": params_old["lag_disk"],
        "lag_bc": params_old["lag_bc"],
        "lag_blr": params_old["lag_blr"],
        "lag_blr2": params_old["lag_blr2"],
    }
    old_wrapper = ContiBLR_SHO_Wrapper(kernel=base_kernel, params=params_old)
    new_wrapper = ContiBLRFluxLinearized_SHO_Wrapper(kernel=base_kernel, params=params_new)

    for band in range(n_band):
        obs_old = old_wrapper.observation_model((jnp.array(0.0), jnp.array(band)))
        obs_new = new_wrapper.observation_model((jnp.array(0.0), jnp.array(band)))
        assert np.allclose(np.asarray(obs_new), np.asarray(obs_old), rtol=1e-6, atol=1e-8)


def test_make_multiband_dho_blr_flux_linearized_model_random_data():
    rng = np.random.default_rng(7)
    n_band = 2
    n_obs = 24

    t = np.sort(rng.uniform(0.0, 80.0, size=n_obs))
    band = rng.integers(0, n_band, size=n_obs)
    yerr = rng.uniform(0.01, 0.05, size=n_obs)
    y = rng.normal(0.0, yerr, size=n_obs)
    baseline_flux_by_band = jnp.asarray([1.2, 0.75], dtype=float)

    model = make_multiband_dho_blr_flux_linearized_model(
        X=(jnp.asarray(t), jnp.asarray(band)),
        y=jnp.asarray(y),
        yerr=jnp.asarray(yerr),
        n_band=n_band,
        baseline_flux_by_band=baseline_flux_by_band,
        zero_mean=True,
        has_jitter=False,
    )

    tau_fast = jnp.asarray(rng.uniform(5.0, 15.0, size=n_band))
    tau_slow = tau_fast + jnp.asarray(rng.uniform(20.0, 80.0, size=n_band))
    amp_cont_relflux = jnp.asarray(rng.uniform(0.08, 0.2, size=n_band))
    params = {
        "log_kernel_param": jnp.log(jnp.concatenate([tau_fast, tau_slow])),
        "amp_cont_relflux": amp_cont_relflux,
        "amp_bc_relflux": jnp.asarray(rng.uniform(0.01, 0.05, size=n_band)),
        "amp_blr_relflux": jnp.asarray(rng.uniform(0.01, 0.08, size=n_band)),
        "amp_blr2_relflux": jnp.zeros(n_band, dtype=float),
        "lag_disk": jnp.asarray(rng.uniform(0.5, 5.0, size=n_band)),
        "lag_bc": jnp.asarray(rng.uniform(1.0, 8.0, size=n_band)),
        "lag_blr": jnp.asarray(rng.uniform(5.0, 30.0, size=n_band)),
        "lag_blr2": jnp.zeros(n_band, dtype=float),
    }

    log_prob = model.log_prob(params)
    pred_mean_rel, pred_std_rel = model.pred(
        params,
        (
            jnp.linspace(0.0, 90.0, 12, dtype=float),
            jnp.zeros(12, dtype=int),
        ),
    )
    pred_mean_mag, pred_std_mag = model.prediction_to_display((pred_mean_rel, pred_std_rel))
    expected_log_amp = np.log((2.5 / np.log(10.0)) * np.asarray(amp_cont_relflux))
    gp, _ = model._build_gp(params)
    omega = jnp.asarray(np.logspace(-3, 0, 8), dtype=float)
    expected_psd = np.asarray(qs_psd(gp.kernel, omega, b=0, sigma_n2=0.0)) * (2.5 / np.log(10.0)) ** 2

    assert np.isfinite(float(log_prob))
    assert np.all(np.isfinite(np.asarray(pred_mean_rel)))
    assert np.all(np.isfinite(np.asarray(pred_std_rel)))
    assert np.all(np.isfinite(np.asarray(pred_mean_mag)))
    assert np.all(np.isfinite(np.asarray(pred_std_mag)))
    assert np.isclose(float(relative_flux_to_mag_residual(0.0)), 0.0)
    assert np.isclose(float(relative_flux_std_to_mag_std(0.0, 0.1)), (2.5 / np.log(10.0)) * 0.1)
    assert np.allclose(np.asarray(model.my_amp_transform(params)), expected_log_amp, rtol=1e-6, atol=1e-8)
    assert np.allclose(np.asarray(model.psd(params, omega, b=0, sigma_n2=0.0)), expected_psd, rtol=1e-6, atol=1e-8)


def test_two_stage_fluxmix_display_soft_floors_large_negative_relflux():
    class _StubContinuumModel:
        X = (jnp.asarray([0.0]), jnp.asarray([0], dtype=int))
        nBand = 1

    model = TwoStageFluxMixDisplayModel(
        continuum_model=_StubContinuumModel(),
        basis_grid_t=jnp.asarray([0.0, 1.0], dtype=float),
        basis_relflux_norm=jnp.asarray([[-1.0, -1.0]], dtype=float),
        t_ref=jnp.asarray([0.0, 1.0], dtype=float),
        zero_mean=True,
        min_total_flux_ratio=0.05,
        floor_softness=0.01,
    )
    params = {
        "amp_cont": jnp.asarray([1.0], dtype=float),
        "amp_blr": jnp.asarray([0.0], dtype=float),
        "amp_bc": jnp.asarray([0.0], dtype=float),
        "lag_blr": jnp.asarray([0.0], dtype=float),
        "lag_bc": jnp.asarray([0.0], dtype=float),
    }
    pred_mu, pred_std = model.pred(
        params,
        (jnp.asarray([0.5], dtype=float), jnp.asarray([0], dtype=int)),
    )

    assert np.all(np.isfinite(np.asarray(pred_mu)))
    assert np.all(np.isfinite(np.asarray(pred_std)))
    assert float(np.asarray(pred_mu)[0]) < 4.0
