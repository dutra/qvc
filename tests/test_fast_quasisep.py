"""Exactness of the fused single-scan solver against the tinygp path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tinygp import GaussianProcess

from qvc.light_curve.fast_quasisep import erlang_transitions, fused_log_probability
from qvc.light_curve.multiband_model_dho_blr_erlang import (
    ErlangResponseDHOQS,
    make_multiband_dho_blr_flux_linearized_erlang_model,
)

jax.config.update("jax_enable_x64", True)

B = 4


def _data(n_epochs, rng, exact_ties):
    t_ep = np.sort(rng.uniform(0.0, 2000.0, n_epochs))
    t = np.repeat(t_ep, B)
    if not exact_ties:
        t = t + np.tile(np.arange(B) * 1e-3, n_epochs)
    band = np.tile(np.arange(B), n_epochs)
    keep = rng.uniform(size=t.size) > 0.35
    t, band = t[keep], band[keep]
    order = np.argsort(t + 1e-9 * band)
    t, band = t[order], band[order]
    y = rng.normal(0.0, 0.1, t.size)
    yerr = np.full(t.size, 0.02)
    return (jnp.asarray(t), jnp.asarray(band, dtype=jnp.int32)), jnp.asarray(y), jnp.asarray(yerr)


def _kernel_params(rng, order):
    return dict(
        tau_fast=jnp.asarray(rng.uniform(5.0, 30.0, B)),
        tau_slow=jnp.asarray(rng.uniform(80.0, 300.0, B)),
        lag_blr=jnp.asarray(rng.uniform(20.0, 200.0, B)),
        amp_cont=jnp.asarray(rng.uniform(0.05, 0.2, B)),
        amp_blr=jnp.asarray(rng.uniform(0.01, 0.08, B)),
        order=order,
    )


@pytest.mark.parametrize("order", [1, 3, 5])
def test_erlang_transitions_matches_vmapped_kernel(order):
    rng = np.random.default_rng(order)
    kernel = ErlangResponseDHOQS(**_kernel_params(rng, order))
    gaps = jnp.asarray(np.concatenate([[0.0], rng.uniform(0.0, 50.0, 40), [0.0, 1e-9]]))
    t = jnp.cumsum(gaps)
    band = jnp.zeros(t.shape, dtype=jnp.int32)
    tp = jnp.append(t[0], t[:-1])
    expected = jax.vmap(kernel.transition_matrix)((tp, band), (t, band))
    # dt must be derived from t exactly as the solver does — near the
    # series/difference branch boundary, a differently-rounded dt legitimately
    # selects a different (equally valid) branch.
    got = erlang_transitions(kernel, jnp.diff(t, prepend=t[0]))
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("exact_ties", [True, False])
@pytest.mark.parametrize("order", [1, 3])
def test_fused_log_probability_matches_tinygp(exact_ties, order):
    rng = np.random.default_rng(7 * order + exact_ties)
    X, y, yerr = _data(120, rng, exact_ties)
    diag = yerr**2
    kp = _kernel_params(rng, order)

    def lp_ref(kp):
        gp = GaussianProcess(ErlangResponseDHOQS(**kp), X, diag=diag, assume_sorted=True)
        return gp.log_probability(y)

    def lp_fast(kp):
        return fused_log_probability(ErlangResponseDHOQS(**kp), X, diag, y, sort_time=X[0])

    v0, v1 = lp_ref(kp), lp_fast(kp)
    np.testing.assert_allclose(float(v1), float(v0), rtol=1e-10)

    diff_kp = {k: v for k, v in kp.items() if k != "order"}
    g0 = jax.grad(lambda d: lp_ref({**d, "order": order}))(diff_kp)
    g1 = jax.grad(lambda d: lp_fast({**d, "order": order}))(diff_kp)
    for k in g0:
        np.testing.assert_allclose(
            np.asarray(g1[k]), np.asarray(g0[k]), rtol=5e-7, atol=1e-10
        )


@pytest.mark.parametrize("has_jitter", [False, True])
def test_model_log_prob_fast_matches_reference(has_jitter):
    rng = np.random.default_rng(3)
    X, y, yerr = _data(90, rng, True)

    def build(fast):
        return make_multiband_dho_blr_flux_linearized_erlang_model(
            X, y, yerr, n_band=B, has_jitter=has_jitter, use_fast_solver=fast
        )

    m_ref, m_fast = build(False), build(True)
    params = {
        "tau_fast_band": jnp.asarray(rng.uniform(5.0, 30.0, B)),
        "tau_slow_band": jnp.asarray(rng.uniform(80.0, 300.0, B)),
        "lag_blr": jnp.asarray(rng.uniform(20.0, 200.0, B)),
        "amp_cont": jnp.asarray(rng.uniform(0.05, 0.2, B)),
        "amp_blr": jnp.asarray(rng.uniform(0.01, 0.08, B)),
        "mean_offset": jnp.zeros(B),
        "mean_slope": jnp.zeros(B),
    }
    if has_jitter:
        params["log_jitter"] = jnp.full(B, -5.0)

    v0 = float(m_ref.log_prob(params))
    v1 = float(m_fast.log_prob(params))
    np.testing.assert_allclose(v1, v0, rtol=1e-10)

    g0 = jax.grad(lambda p: m_ref.log_prob(p))(params)
    g1 = jax.grad(lambda p: m_fast.log_prob(p))(params)
    for k in g0:
        np.testing.assert_allclose(
            np.asarray(g1[k]), np.asarray(g0[k]), rtol=5e-7, atol=1e-10
        )
