import os
import sys
from pathlib import Path

import numpy as np
import pytest


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
from qvc.light_curve.multiband_dho_core import (
    mag_residual_to_relative_flux,
    relative_flux_to_mag_residual,
)


jax.config.update("jax_enable_x64", True)


def _original_loop_design_matrix(kernel):
    """Pre-vectorization implementation retained as a numerical oracle."""

    base = kernel._base()
    A0 = base.design_matrix()
    B, n0, n = kernel._dimensions()
    dtype = A0.dtype
    A = jnp.zeros((n, n), dtype=dtype).at[:n0, :n0].set(A0)
    for b in range(B):
        rate = int(kernel.order) / jnp.maximum(jnp.asarray(kernel.lag_blr)[b], 1e-12)
        start = n0 + b * int(kernel.order)
        driver_h = base.observation_model(
            (jnp.array(0.0, dtype=dtype), jnp.array(b, dtype=jnp.int32))
        )
        A = A.at[start, :n0].set(rate * driver_h)
        A = A.at[start, start].set(-rate)
        for k in range(1, int(kernel.order)):
            idx = start + k
            A = A.at[idx, idx - 1].set(rate)
            A = A.at[idx, idx].set(-rate)
    return A


@pytest.mark.parametrize("n_band", [1, 2, 5])
@pytest.mark.parametrize("order", [1, 2, 4, 6])
def test_vectorized_design_matrix_matches_original_loops(n_band, order):
    tau_fast = jnp.linspace(8.0, 28.0, n_band)
    tau_slow = jnp.linspace(90.0, 330.0, n_band)
    lag_blr = jnp.linspace(25.0, 180.0, n_band)
    kernel = ErlangResponseDHOQS(
        tau_fast=tau_fast,
        tau_slow=tau_slow,
        lag_blr=lag_blr,
        amp_cont=jnp.linspace(0.1, 0.3, n_band),
        amp_blr=jnp.linspace(0.02, 0.09, n_band),
        order=order,
    )
    vectorized = np.asarray(kernel.design_matrix())
    original = np.asarray(_original_loop_design_matrix(kernel))
    np.testing.assert_array_equal(vectorized, original)


def test_vectorized_design_matrix_lag_gradient_matches_original_loops():
    def objective(lags, original):
        kernel = ErlangResponseDHOQS(
            tau_fast=jnp.array([11.0, 23.0]),
            tau_slow=jnp.array([120.0, 275.0]),
            lag_blr=lags,
            amp_cont=jnp.array([0.15, 0.28]),
            amp_blr=jnp.array([0.03, 0.08]),
            order=4,
        )
        A = _original_loop_design_matrix(kernel) if original else kernel.design_matrix()
        weights = jnp.arange(A.size, dtype=float).reshape(A.shape) + 1.0
        return jnp.sum(A * weights)

    lags = jnp.array([42.0, 135.0])
    vectorized_grad = jax.grad(lambda x: objective(x, False))(lags)
    original_grad = jax.grad(lambda x: objective(x, True))(lags)
    np.testing.assert_allclose(vectorized_grad, original_grad, rtol=1e-13, atol=1e-13)


def _make_kernel(n_band, order, lag_scale=1.0):
    return ErlangResponseDHOQS(
        tau_fast=jnp.linspace(8.0, 28.0, n_band),
        tau_slow=jnp.linspace(90.0, 330.0, n_band),
        lag_blr=lag_scale * jnp.linspace(25.0, 180.0, n_band),
        amp_cont=jnp.linspace(0.1, 0.3, n_band),
        amp_blr=jnp.linspace(0.02, 0.09, n_band),
        order=order,
    )


@pytest.mark.parametrize("n_band", [1, 2, 5])
@pytest.mark.parametrize("order", [1, 2, 4, 6])
def test_closed_form_transition_matrix_matches_expm(n_band, order):
    kernel = _make_kernel(n_band, order)
    for dt in [0.0, 0.05, 0.37, 5.0, 41.7, 333.0, -12.5]:
        X1 = (jnp.asarray(100.0), jnp.asarray(0))
        X2 = (jnp.asarray(100.0 + dt), jnp.asarray(min(1, n_band - 1)))
        closed = np.asarray(kernel.transition_matrix(X1, X2))
        oracle = np.asarray(kernel._transition_matrix_expm(X1, X2))
        np.testing.assert_allclose(closed, oracle, rtol=1e-9, atol=1e-12)


def test_closed_form_transition_matrix_gradients_match_expm():
    def objective(theta, use_oracle):
        kernel = ErlangResponseDHOQS(
            tau_fast=theta["tau_fast"],
            tau_slow=theta["tau_slow"],
            lag_blr=theta["lag_blr"],
            amp_cont=jnp.array([0.15, 0.28]),
            amp_blr=jnp.array([0.03, 0.08]),
            order=3,
        )
        X1 = (jnp.asarray(0.0), jnp.asarray(0))
        X2 = (jnp.asarray(17.3), jnp.asarray(1))
        fn = kernel._transition_matrix_expm if use_oracle else kernel.transition_matrix
        phi = fn(X1, X2)
        weights = jnp.arange(phi.size, dtype=float).reshape(phi.shape) + 1.0
        return jnp.sum(phi * weights)

    theta = {
        "tau_fast": jnp.array([11.0, 23.0]),
        "tau_slow": jnp.array([120.0, 275.0]),
        "lag_blr": jnp.array([42.0, 135.0]),
    }
    grad_closed = jax.grad(lambda th: objective(th, False))(theta)
    grad_oracle = jax.grad(lambda th: objective(th, True))(theta)
    for key in theta:
        np.testing.assert_allclose(
            grad_closed[key], grad_oracle[key], rtol=1e-8, atol=1e-12
        )


def _gp_log_likelihood(kernel, use_oracle_transition=False):
    rng = np.random.default_rng(42)
    n_band = kernel.tau_fast.shape[0]
    t = np.sort(rng.uniform(0.0, 1500.0, 400))
    band = rng.integers(0, n_band, t.size)
    y = rng.normal(0.0, 0.1, t.size)
    X = (jnp.asarray(t), jnp.asarray(band, dtype=jnp.int32))

    if use_oracle_transition:
        class OracleKernel(ErlangResponseDHOQS):
            def transition_matrix(self, X1, X2):
                return self._transition_matrix_expm(X1, X2)

        kernel = OracleKernel(
            tau_fast=kernel.tau_fast,
            tau_slow=kernel.tau_slow,
            lag_blr=kernel.lag_blr,
            amp_cont=kernel.amp_cont,
            amp_blr=kernel.amp_blr,
            order=kernel.order,
        )

    from tinygp import GaussianProcess

    gp = GaussianProcess(kernel, X, diag=jnp.full(t.size, 4e-4), assume_sorted=True)
    return float(gp.log_probability(jnp.asarray(y)))


@pytest.mark.parametrize("order", [1, 3])
def test_closed_form_transition_preserves_gp_log_likelihood(order):
    kernel = _make_kernel(3, order)
    closed = _gp_log_likelihood(kernel, use_oracle_transition=False)
    oracle = _gp_log_likelihood(kernel, use_oracle_transition=True)
    np.testing.assert_allclose(closed, oracle, rtol=1e-10)


@pytest.mark.parametrize("n_band", [1, 2, 5])
@pytest.mark.parametrize("order", [1, 2, 4, 6])
def test_block_sylvester_stationary_covariance_matches_kron(n_band, order):
    kernel = _make_kernel(n_band, order)
    blockwise = np.asarray(kernel.stationary_covariance())
    oracle = np.asarray(kernel._stationary_covariance_kron())
    np.testing.assert_allclose(blockwise, oracle, rtol=1e-10, atol=1e-13)


def test_block_sylvester_stationary_covariance_gradients_match_kron():
    def objective(theta, use_oracle):
        kernel = ErlangResponseDHOQS(
            tau_fast=theta["tau_fast"],
            tau_slow=theta["tau_slow"],
            lag_blr=theta["lag_blr"],
            amp_cont=jnp.array([0.15, 0.28]),
            amp_blr=jnp.array([0.03, 0.08]),
            order=3,
        )
        fn = (
            kernel._stationary_covariance_kron
            if use_oracle
            else kernel.stationary_covariance
        )
        P = fn()
        weights = jnp.arange(P.size, dtype=float).reshape(P.shape) + 1.0
        return jnp.sum(P * weights)

    theta = {
        "tau_fast": jnp.array([11.0, 23.0]),
        "tau_slow": jnp.array([120.0, 275.0]),
        "lag_blr": jnp.array([42.0, 135.0]),
    }
    grad_blockwise = jax.grad(lambda th: objective(th, False))(theta)
    grad_oracle = jax.grad(lambda th: objective(th, True))(theta)
    for key in theta:
        np.testing.assert_allclose(
            grad_blockwise[key], grad_oracle[key], rtol=1e-8, atol=1e-12
        )


def _collision_kernel(order, lag_blr):
    """Kernel whose band-0 response rate can collide with a continuum rate."""

    return ErlangResponseDHOQS(
        tau_fast=jnp.array([20.0, 11.0]),
        tau_slow=jnp.array([150.0, 240.0]),
        lag_blr=jnp.asarray(lag_blr),
        amp_cont=jnp.array([0.15, 0.28]),
        amp_blr=jnp.array([0.03, 0.08]),
        order=order,
    )


@pytest.mark.parametrize("order", [1, 3, 6])
@pytest.mark.parametrize("tau_collide", [20.0, 150.0])
@pytest.mark.parametrize("z_rel", [0.0, 1e-15, 1e-10, 1e-6, 1e-3])
def test_transition_matrix_is_stable_at_pole_collisions(order, tau_collide, z_rel):
    # Response rate q = order / lag equals a continuum rate 1/tau (up to a
    # relative offset z_rel), the singular point of the naive closed form.
    lag0 = order * tau_collide * (1.0 + z_rel)
    kernel = _collision_kernel(order, [lag0, 90.0])
    for dt in [0.3, 12.0, 75.0, 400.0]:
        X1 = (jnp.asarray(0.0), jnp.asarray(0))
        X2 = (jnp.asarray(dt), jnp.asarray(0))
        closed = np.asarray(kernel.transition_matrix(X1, X2))
        oracle = np.asarray(kernel._transition_matrix_expm(X1, X2))
        assert np.all(np.isfinite(closed))
        np.testing.assert_allclose(closed, oracle, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("z_rel", [0.0, 1e-12, 1e-6])
def test_transition_matrix_gradients_are_finite_and_match_at_collisions(z_rel):
    order = 3

    def objective(theta, use_oracle):
        kernel = _collision_kernel(order, theta["lag_blr"])
        X1 = (jnp.asarray(0.0), jnp.asarray(0))
        X2 = (jnp.asarray(37.0), jnp.asarray(0))
        fn = kernel._transition_matrix_expm if use_oracle else kernel.transition_matrix
        phi = fn(X1, X2)
        weights = jnp.arange(phi.size, dtype=float).reshape(phi.shape) + 1.0
        return jnp.sum(phi * weights)

    theta = {"lag_blr": jnp.array([order * 20.0 * (1.0 + z_rel), 90.0])}
    grad_closed = jax.grad(lambda th: objective(th, False))(theta)
    grad_oracle = jax.grad(lambda th: objective(th, True))(theta)
    assert np.all(np.isfinite(np.asarray(grad_closed["lag_blr"])))
    np.testing.assert_allclose(
        grad_closed["lag_blr"], grad_oracle["lag_blr"], rtol=1e-7, atol=1e-12
    )


def test_gp_log_likelihood_finite_and_matches_oracle_at_collision():
    order = 3
    kernel = _collision_kernel(order, [order * 20.0, 90.0])
    closed = _gp_log_likelihood(kernel, use_oracle_transition=False)
    oracle = _gp_log_likelihood(kernel, use_oracle_transition=True)
    assert np.isfinite(closed)
    np.testing.assert_allclose(closed, oracle, rtol=1e-10)


@pytest.mark.parametrize("lag0", [1e-12, 1e-6, 1e4])
def test_transition_matrix_gradients_finite_at_extreme_lags(lag0):
    def objective(lags):
        kernel = _collision_kernel(3, lags)
        X1 = (jnp.asarray(0.0), jnp.asarray(0))
        X2 = (jnp.asarray(250.0), jnp.asarray(0))
        return jnp.sum(kernel.transition_matrix(X1, X2))

    grad = jax.grad(objective)(jnp.array([lag0, 90.0]))
    assert np.all(np.isfinite(np.asarray(grad)))


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


def test_erlang_cross_covariance_has_causal_direction_and_matches_convolution():
    """Check direction against an external, past-only convolution oracle.

    This deliberately does not use the augmented Erlang state to construct the
    oracle.  For R(t) = integral psi(u) C(t-u) du, u >= 0, the cross covariance
    is

        Cov[C(0), R(s)] = integral psi(u) k_C(s-u) du.

    This test would fail if the continuum/response cross covariance were
    accidentally transposed while remaining symmetric and positive definite.
    """

    from scipy.linalg import expm

    lag = 80.0
    order = 4
    kernel = ErlangResponseDHOQS(
        tau_fast=jnp.array([20.0]),
        tau_slow=jnp.array([150.0]),
        lag_blr=jnp.array([lag]),
        amp_cont=jnp.array([1.0]),
        amp_blr=jnp.array([1.0]),
        order=order,
    )

    A = np.asarray(kernel.design_matrix())
    P = np.asarray(kernel.stationary_covariance())
    base = kernel._base()
    A0 = np.asarray(base.design_matrix())
    P0 = np.asarray(base.stationary_covariance())
    h0 = np.asarray(base.observation_model((jnp.array(0.0), jnp.array(0))))

    h_cont = np.zeros(A.shape[0])
    h_cont[: A0.shape[0]] = h0
    h_resp = np.zeros(A.shape[0])
    h_resp[-1] = 1.0

    # The integration grid is independent of the state-space implementation.
    u = np.linspace(0.0, 12.0 * lag, 24001)
    psi = np.asarray(erlang_impulse_response(u, lag, order))

    def base_covariance(delta):
        # The scalar stationary base covariance is even in delta.
        F = expm(A0 * abs(float(delta)))
        return h0 @ P0 @ F.T @ h0

    def convolution_oracle(separation):
        values = np.array([base_covariance(separation - ui) for ui in u])
        return np.trapezoid(psi * values, u)

    def augmented_cross_covariance(separation):
        F = expm(A * abs(float(separation)))
        if separation >= 0.0:
            # C(0), followed by R(separation).
            return h_resp @ F @ P @ h_cont
        # R(separation), followed by C(0).
        return h_cont @ F @ P @ h_resp

    separations = np.array([-160.0, -80.0, 0.0, 40.0, 80.0, 160.0])
    state_values = np.array([augmented_cross_covariance(s) for s in separations])
    oracle_values = np.array([convolution_oracle(s) for s in separations])
    np.testing.assert_allclose(state_values, oracle_values, rtol=2e-6, atol=2e-9)

    # A delayed response must correlate more strongly with an earlier
    # continuum than with a continuum the same distance in its future.
    forward = augmented_cross_covariance(lag)
    reverse = augmented_cross_covariance(-lag)
    assert forward > reverse

    # Exercise TinyGP's exact transition convention explicitly.  For t1<t2 it
    # contracts h2 @ Pinf @ transition(t1,t2) @ h1.
    X1 = (jnp.asarray(0.0), jnp.asarray(0))
    X2 = (jnp.asarray(lag), jnp.asarray(0))
    tinygp_forward = (
        h_resp
        @ np.asarray(kernel.transition_matrix(X1, X2))
        @ np.asarray(kernel.stationary_covariance())
        @ h_cont
    )
    np.testing.assert_allclose(tinygp_forward, forward, rtol=2e-10, atol=2e-12)


def test_causal_qsm_matches_pairwise_dense_covariance_and_is_psd():
    kernel = ErlangResponseDHOQS(
        tau_fast=jnp.array([18.0, 27.0]),
        tau_slow=jnp.array([140.0, 230.0]),
        lag_blr=jnp.array([55.0, 105.0]),
        amp_cont=jnp.array([0.7, 0.5]),
        amp_blr=jnp.array([0.12, 0.3]),
        order=3,
    )
    times = jnp.array([0.0, 0.0, 17.0, 43.0, 43.0, 91.0, 180.0])
    bands = jnp.array([0, 1, 1, 0, 1, 0, 1], dtype=jnp.int32)
    X = (times, bands)
    qsm_dense = np.asarray(kernel.to_symm_qsm(X).to_dense())
    pairwise_dense = np.asarray(
        jax.vmap(lambda x1: jax.vmap(lambda x2: kernel.evaluate(x1, x2))(X))(X)
    )
    np.testing.assert_allclose(qsm_dense, pairwise_dense, rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(qsm_dense, qsm_dense.T, rtol=0.0, atol=2e-12)
    assert np.linalg.eigvalsh(qsm_dense).min() > -1e-10


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


def test_psf_agn_fraction_dilutes_covariance_in_observation_space():
    times = jnp.array([0.0, 5.0, 12.0, 20.0])
    band = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
    model = make_multiband_dho_blr_flux_linearized_erlang_model(
        (times, band),
        jnp.zeros(times.shape),
        jnp.full(times.shape, 0.02),
        n_band=2,
        survey_idx=jnp.zeros(times.shape, dtype=jnp.int32),
    )
    params = {
        "tau_fast_band": jnp.array([20.0, 25.0]),
        "tau_slow_band": jnp.array([150.0, 180.0]),
        "lag_blr": jnp.array([80.0, 90.0]),
        "amp_cont_relflux": jnp.array([0.1, 0.12]),
        "amp_blr_relflux": jnp.array([0.02, 0.03]),
    }
    undiluted = np.asarray(model._build_kernel(params).to_symm_qsm((times, band)).to_dense())
    fractions = np.array([0.4, 0.7])
    params["agn_fraction_by_band"] = jnp.asarray(fractions)
    diluted = np.asarray(model._build_kernel(params).to_symm_qsm((times, band)).to_dense())
    point_fractions = fractions[np.asarray(band)]

    np.testing.assert_allclose(
        diluted,
        undiluted * np.outer(point_fractions, point_fractions),
        rtol=2e-10,
        atol=2e-12,
    )


def _deterministic_continuum(t):
    """A smooth, non-periodic test continuum with several time scales."""

    t = np.asarray(t, dtype=float)
    return (
        0.65 * np.exp(-0.5 * ((t - 85.0) / 16.0) ** 2)
        - 0.45 * np.exp(-0.5 * ((t - 205.0) / 24.0) ** 2)
        + 0.30 * np.exp(-0.5 * ((t - 330.0) / 19.0) ** 2)
    )


def _causal_erlang_convolution(signal, dt, lag, order=4):
    delay_grid = np.arange(signal.size, dtype=float) * dt
    response = np.array(erlang_impulse_response(delay_grid, lag, order), copy=True)
    response /= np.sum(response) * dt
    return np.convolve(signal, response, mode="full")[: signal.size] * dt


@pytest.mark.parametrize(
    "lag,continuum_amp,blr_amp",
    [
        (25.0, 0.05, 0.01),
        (60.0, 0.12, 0.03),
        (110.0, 0.20, 0.08),
        (160.0, 0.30, 0.12),
    ],
)
def test_continuum_plus_erlang_blr_recovers_lag_after_magnitude_conversion(
    lag,
    continuum_amp,
    blr_amp,
):
    """Flux-additive continuum+BLR injections retain their lag in magnitudes."""

    dt = 0.5
    t = np.arange(0.0, 600.0, dt)
    driver = _deterministic_continuum(t)
    delayed = _causal_erlang_convolution(driver, dt, lag, order=4)
    rel_flux = continuum_amp * driver + blr_amp * delayed
    assert np.min(1.0 + rel_flux) > 0.5

    mag = np.asarray(relative_flux_to_mag_residual(rel_flux))
    recovered_rel_flux = np.asarray(mag_residual_to_relative_flux(mag))
    np.testing.assert_allclose(recovered_rel_flux, rel_flux, rtol=2e-6, atol=2e-7)

    lag_grid = np.arange(max(5.0, lag - 30.0), lag + 30.1, 0.5)
    losses = []
    for trial_lag in lag_grid:
        trial_delayed = _causal_erlang_convolution(driver, dt, trial_lag, order=4)
        trial_rel_flux = continuum_amp * driver + blr_amp * trial_delayed
        trial_mag = np.asarray(relative_flux_to_mag_residual(trial_rel_flux))
        losses.append(np.mean((mag - trial_mag) ** 2))

    recovered_lag = lag_grid[int(np.argmin(losses))]
    assert np.isclose(recovered_lag, lag, atol=0.51)


@pytest.mark.parametrize("lag", [35.0, 90.0, 150.0])
def test_small_magnitude_noise_preserves_erlang_lag_recovery(lag):
    """Representative magnitude noise does not erase deterministic lag recovery."""

    dt = 1.0
    t = np.arange(0.0, 650.0, dt)
    driver = _deterministic_continuum(t)
    continuum_amp = 0.18
    blr_amp = 0.08
    delayed = _causal_erlang_convolution(driver, dt, lag, order=4)
    rel_flux = continuum_amp * driver + blr_amp * delayed
    mag = np.asarray(relative_flux_to_mag_residual(rel_flux))
    rng = np.random.default_rng(int(lag))
    mag_obs = mag + rng.normal(0.0, 0.005, size=mag.size)

    lag_grid = np.arange(max(5.0, lag - 40.0), lag + 40.1, 1.0)
    losses = []
    for trial_lag in lag_grid:
        trial_delayed = _causal_erlang_convolution(driver, dt, trial_lag, order=4)
        trial_flux = continuum_amp * driver + blr_amp * trial_delayed
        trial_mag = np.asarray(relative_flux_to_mag_residual(trial_flux))
        losses.append(np.mean((mag_obs - trial_mag) ** 2))

    recovered_lag = lag_grid[int(np.argmin(losses))]
    assert abs(recovered_lag - lag) <= 5.0
