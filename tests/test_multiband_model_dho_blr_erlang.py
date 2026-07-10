import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp

from qvc.light_curve.multiband_model_dho_blr_erlang import (
    ErlangResponseDHOQS,
    erlang_impulse_response,
    erlang_response_moments,
    make_multiband_dho_blr_flux_linearized_erlang_model,
)


jax.config.update("jax_enable_x64", True)


def test_erlang_response_has_requested_deterministic_centroid():
    lag = 120.0
    order = 4
    t = np.linspace(0.0, 800.0, 20001)
    response = np.asarray(erlang_impulse_response(t, lag, order))
    norm = np.trapezoid(response, t)
    centroid = np.trapezoid(t * response, t) / norm
    variance = np.trapezoid((t - centroid) ** 2 * response, t) / norm
    expected_centroid, expected_std = erlang_response_moments(lag, order)

    assert np.isclose(norm, 1.0, rtol=2e-4)
    assert np.isclose(centroid, float(expected_centroid), rtol=2e-4)
    assert np.isclose(np.sqrt(variance), float(expected_std), rtol=3e-4)
    assert np.isclose(t[np.argmax(response)], lag * (order - 1) / order, atol=0.1)


def test_erlang_augmented_stationary_covariance_is_psd_and_solves_lyapunov():
    kernel = ErlangResponseDHOQS(
        tau_fast=jnp.array([20.0]),
        tau_slow=jnp.array([150.0]),
        lag_blr=jnp.array([80.0]),
        amp_cont=jnp.array([0.7]),
        amp_blr=jnp.array([0.2]),
        order=3,
    )
    A = np.asarray(kernel.design_matrix())
    P = np.asarray(kernel.stationary_covariance())
    base = kernel._base()
    A0 = np.asarray(base.design_matrix())
    P0 = np.asarray(base.stationary_covariance())
    Q = np.zeros_like(P)
    Q[: A0.shape[0], : A0.shape[0]] = -(A0 @ P0 + P0 @ A0.T)

    residual = A @ P + P @ A.T + Q
    eigvals = np.linalg.eigvalsh(P)
    assert np.max(np.abs(residual)) < 1e-8
    assert eigvals.min() > -1e-9


def test_erlang_state_impulse_response_peaks_after_zero():
    lag = 100.0
    order = 5
    kernel = ErlangResponseDHOQS(
        tau_fast=jnp.array([20.0]),
        tau_slow=jnp.array([150.0]),
        lag_blr=jnp.array([lag]),
        amp_cont=jnp.array([0.0]),
        amp_blr=jnp.array([1.0]),
        order=order,
    )
    A = np.asarray(kernel.design_matrix())
    n0 = 2
    # A unit impulse in the driver causes the first filter state to jump by
    # rate; subsequent propagation must reproduce the Erlang response.
    initial = np.zeros(A.shape[0])
    initial[n0] = order / lag
    t = np.linspace(0.0, 300.0, 1201)
    from scipy.linalg import expm

    output = np.array([(expm(A * ti) @ initial)[-1] for ti in t])
    expected = np.asarray(erlang_impulse_response(t, lag, order))
    np.testing.assert_allclose(output, expected, rtol=2e-7, atol=2e-10)
    assert t[np.argmax(output)] > 0.0
    assert np.isclose(t[np.argmax(output)], lag * (order - 1) / order, atol=0.3)


def test_deterministic_erlang_response_recovers_injected_centroid_lag():
    injected_lag = 75.0
    order = 4
    t = np.linspace(0.0, 350.0, 1401)
    observed = np.asarray(erlang_impulse_response(t, injected_lag, order))
    lag_grid = np.linspace(55.0, 95.0, 161)
    losses = np.array(
        [
            np.mean(
                (observed - np.asarray(erlang_impulse_response(t, trial_lag, order))) ** 2
            )
            for trial_lag in lag_grid
        ]
    )
    recovered_lag = lag_grid[np.argmin(losses)]
    assert np.isclose(recovered_lag, injected_lag, atol=0.26)


def test_erlang_quasisep_gp_has_finite_likelihood():
    times = jnp.array([0.0, 7.0, 18.0, 35.0])
    band = jnp.zeros(times.shape, dtype=jnp.int32)
    model = make_multiband_dho_blr_flux_linearized_erlang_model(
        (times, band),
        jnp.zeros(times.shape),
        jnp.full(times.shape, 0.02),
        n_band=1,
        survey_idx=jnp.zeros(times.shape, dtype=jnp.int32),
    )
    params = {
        "tau_fast_band": jnp.array([20.0]),
        "tau_slow_band": jnp.array([150.0]),
        "lag_blr": jnp.array([80.0]),
        "amp_cont_relflux": jnp.array([0.1]),
        "amp_blr_relflux": jnp.array([0.02]),
        "log_jitter": jnp.full((1, 3), -10.0),
        "survey_delta_mag": jnp.zeros((1, 3)),
        "mean": jnp.zeros(1),
        "linear_trend": jnp.asarray(0.0),
        "linear_trend_band_offset": jnp.zeros(1),
    }
    assert np.isfinite(float(model.log_prob(params)))
