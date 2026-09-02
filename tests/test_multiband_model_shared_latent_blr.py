import numpy as np
from types import SimpleNamespace
import jax
import jax.numpy as jnp
from numpyro.handlers import seed, trace
from scipy.linalg import expm

from qvc.light_curve.multiband_model_shared_latent_blr import (
    SharedLatentDiskBLRQS,
    SharedLatentDiskBLRRelativeFluxModel,
    continuum_effective_timescale,
)
from qvc.light_curve.fit_light_curves import build_single_object_model_mag_flux_linearized


def _kernel():
    return SharedLatentDiskBLRQS(
        tau_fast=jnp.array([15.0]),
        tau_slow=jnp.array([180.0]),
        lag_disk=jnp.array([2.0, 4.0]),
        lag_blr=jnp.array([30.0, 70.0]),
        amp_cont=jnp.array([0.10, 0.08]),
        amp_blr=jnp.array([0.03, 0.02]),
        disk_order=4,
        blr_order=3,
    )


def test_shared_latent_stationary_covariance_solves_lyapunov_equation():
    kernel = _kernel()
    A = np.asarray(kernel.design_matrix())
    P = np.asarray(kernel.stationary_covariance())
    base = kernel._base()
    A0 = np.asarray(base.design_matrix())
    P0 = np.asarray(base.stationary_covariance())
    Q = np.zeros_like(P)
    Q[:2, :2] = -(A0 @ P0 + P0 @ A0.T)

    np.testing.assert_allclose(A @ P + P @ A.T + Q, 0.0, atol=2e-8)
    assert np.linalg.eigvalsh(P).min() > -1e-10


def test_shared_latent_transition_matches_matrix_exponential():
    kernel = _kernel()
    analytic = np.asarray(kernel.transition_matrix((1.0, 0), (8.5, 1)))
    numeric = expm(np.asarray(kernel.design_matrix()) * 7.5)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=2e-7)


def test_disk_and_delayed_loadings_are_stationary_rms_normalized():
    kernel = _kernel()
    P = np.asarray(kernel.stationary_covariance())
    for band in range(2):
        h = np.asarray(kernel.observation_model((0.0, band)))
        disk_endpoint = 2 + (band + 1) * kernel.disk_order - 1
        blr_endpoint = 2 + 2 * kernel.disk_order + (band + 1) * kernel.blr_order - 1
        disk_rms = h[disk_endpoint] * np.sqrt(P[disk_endpoint, disk_endpoint])
        blr_rms = h[blr_endpoint] * np.sqrt(P[blr_endpoint, blr_endpoint])
        np.testing.assert_allclose(disk_rms, np.asarray(kernel.amp_cont)[band], rtol=2e-6)
        np.testing.assert_allclose(blr_rms, np.asarray(kernel.amp_blr)[band], rtol=2e-6)


def test_shared_latent_kernel_has_one_driver_pair():
    kernel = _kernel()
    expected = 2 + 2 * kernel.disk_order + 2 * kernel.blr_order
    assert kernel.design_matrix().shape == (expected, expected)
    assert float(kernel.evaluate((0.0, 0), (10.0, 1))) != 0.0


def test_stationary_rms_matches_zero_lag_kernel_variance():
    kernel = _kernel()
    expected = np.sqrt(
        [float(kernel.evaluate((0.0, band), (0.0, band))) for band in range(2)]
    )
    np.testing.assert_allclose(kernel.stationary_rms(), expected, rtol=2e-6)


def test_shared_model_kernel_driver_is_independent_of_band_arrays_and_order():
    model = SimpleNamespace(disk_order=3, blr_order=2)
    params = {
        "tau_fast_driver": 15.0,
        "tau_slow_driver": 180.0,
        # Deliberately inconsistent legacy arrays: the shared driver must not
        # consume either their values or their first-band identity.
        "tau_fast_band": jnp.array([1.0, 999.0]),
        "tau_slow_band": jnp.array([2.0, 888.0]),
        "lag_disk": jnp.array([2.0, 4.0]),
        "lag_blr": jnp.array([30.0, 70.0]),
        "amp_cont_relflux": jnp.array([0.10, 0.08]),
        "amp_blr_relflux": jnp.array([0.03, 0.02]),
    }
    kernel = SharedLatentDiskBLRRelativeFluxModel._build_kernel(model, params)
    reversed_params = {
        **params,
        "tau_fast_band": params["tau_fast_band"][::-1],
        "tau_slow_band": params["tau_slow_band"][::-1],
        "lag_disk": params["lag_disk"][::-1],
        "lag_blr": params["lag_blr"][::-1],
        "amp_cont_relflux": params["amp_cont_relflux"][::-1],
        "amp_blr_relflux": params["amp_blr_relflux"][::-1],
    }
    reversed_kernel = SharedLatentDiskBLRRelativeFluxModel._build_kernel(
        model, reversed_params
    )

    np.testing.assert_allclose(kernel.tau_fast, [15.0])
    np.testing.assert_allclose(kernel.tau_slow, [180.0])
    np.testing.assert_allclose(reversed_kernel.tau_fast, kernel.tau_fast)
    np.testing.assert_allclose(reversed_kernel.tau_slow, kernel.tau_slow)


def test_effective_timescale_matches_integrated_autocorrelation():
    kernel = _kernel()
    tau_effective = np.asarray(kernel.effective_timescales())
    # Integrate the independently evaluated covariance far beyond the slowest
    # driver/filter timescale. The exact state-space result should agree.
    lags = np.linspace(0.0, 4000.0, 20001)
    for band in range(2):
        covariance = np.asarray(
            jax.vmap(lambda lag: kernel.evaluate((lag, band), (0.0, band)))(
                jnp.asarray(lags)
            )
        )
        numeric = np.trapezoid(covariance, lags) / covariance[0]
        np.testing.assert_allclose(tau_effective[band], numeric, rtol=2e-4)


def test_continuum_timescale_matches_numerical_covariance_and_excludes_blr():
    kernel = _kernel()
    exact = np.asarray(kernel.continuum_effective_timescales())
    A = np.asarray(kernel.design_matrix())
    P = np.asarray(kernel.stationary_covariance())
    slices, size = kernel._chain_slices()
    endpoints = np.asarray([chain_slice.stop - 1 for chain_slice in slices])
    stds = np.sqrt(np.diag(P)[endpoints])
    lags = np.linspace(0.0, 4000.0, 20001)

    for band in range(2):
        h = np.zeros(size)
        h[2 + (band + 1) * kernel.disk_order - 1] = 1.0 / stds[band]
        covariance = np.asarray([h @ expm(A * lag) @ P @ h for lag in lags])
        numeric = np.trapezoid(covariance, lags) / covariance[0]
        np.testing.assert_allclose(exact[band], numeric, rtol=2e-4)

    changed_blr = SharedLatentDiskBLRQS(
        tau_fast=kernel.tau_fast,
        tau_slow=kernel.tau_slow,
        lag_disk=kernel.lag_disk,
        lag_blr=jnp.array([300.0, 700.0]),
        amp_cont=kernel.amp_cont,
        amp_blr=jnp.array([0.3, 0.5]),
        disk_order=kernel.disk_order,
        blr_order=kernel.blr_order,
    )
    np.testing.assert_allclose(
        changed_blr.continuum_effective_timescales(), exact, rtol=2e-6
    )
    assert not np.allclose(
        changed_blr.effective_timescales(), kernel.effective_timescales()
    )


def test_synthetic_continuum_timescale_changes_with_disk_response():
    base = float(continuum_effective_timescale(15.0, 180.0, 2.0, disk_order=3))
    longer_lag = float(
        continuum_effective_timescale(15.0, 180.0, 8.0, disk_order=3)
    )
    higher_order = float(
        continuum_effective_timescale(15.0, 180.0, 2.0, disk_order=5)
    )
    assert longer_lag > base
    assert not np.isclose(higher_order, base)


def test_shared_latent_fit_uses_convolved_disk_wavelength_law():
    band = np.tile(np.arange(3), 4)
    obj = {
        "X": (jnp.arange(12.0), jnp.asarray(band)),
        "y": jnp.zeros(12),
        "yerr": jnp.full(12, 0.03),
        "survey_idx": np.zeros(12, dtype=np.int32),
        "z": 1.0,
        "bands": ["g", "r", "i"],
        "mags_means": np.full(3, 20.0),
        "log_jitter_active_mask": np.ones((3, 3), dtype=bool),
        "survey_offset_active_mask": np.zeros((3, 3), dtype=bool),
    }
    lam_rf = jnp.array([2400.0, 3100.0, 3750.0])
    model = build_single_object_model_mag_flux_linearized(
        obj,
        lam_rf,
        np.full((3, 3), -4.0),
        shared_latent=True,
        disk_order=4,
    )
    sites = trace(seed(model, jax.random.PRNGKey(3))).get_trace()
    lag_disk = np.asarray(sites["lag_disk"]["value"])
    expected_ratio = np.asarray((lam_rf / lam_rf[0]) ** (4.0 / 3.0))

    np.testing.assert_allclose(lag_disk / lag_disk[0], expected_ratio, rtol=2e-6)
    assert "eta_tau" not in sites
    assert np.ndim(np.asarray(sites["tau_fast_driver"]["value"])) == 0
    assert np.ndim(np.asarray(sites["tau_slow_driver"]["value"])) == 0
