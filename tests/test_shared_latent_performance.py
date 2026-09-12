"""Numerical contracts for the shared, bounded-memory prediction paths."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm, solve_continuous_lyapunov
from tinygp import GaussianProcess

from qvc.light_curve import fit_light_curves as fit
from qvc.light_curve.multiband_model_shared_latent_blr import (
    make_multiband_shared_latent_blr_model,
)
from qvc.light_curve.quasisep_prediction import marginal_prediction, regularized_loo_residuals

jax.config.update("jax_enable_x64", True)


def example(variant, n=16):
    rng = np.random.default_rng(92)
    times = np.sort(rng.uniform(0, 2000, n))
    times[1] = times[0]
    perm = rng.permutation(n)
    X = jnp.asarray(times[perm]), jnp.asarray((np.arange(n) % 2)[perm])
    y = jnp.asarray(rng.normal(0, .08, n))
    err = jnp.linspace(.01, .03, n)
    params = dict(
        tau_fast_driver=jnp.array(.05), tau_slow_driver=jnp.array(300.),
        lag_disk=jnp.array([2., 5.]), lag_blr=jnp.array([40., 80.]),
        amp_cont_relflux=jnp.array([.1, .08]), amp_blr_relflux=jnp.array([.02, .03]),
        mean=jnp.array([.013, -.021]), linear_trend=jnp.array(.012),
        linear_trend_band_offset=jnp.array([.002, -.002]),
        log_jitter=jnp.log(jnp.full((2, 3), .012)),
        survey_delta_mag=jnp.array([[.01, -.02, 0.], [-.005, .014, 0.]]),
        seeing_mean_slope=jnp.full((2, 3), .02),
        seeing_scatter_slope=jnp.full((2, 3), .1),
    )
    factory = make_multiband_shared_latent_blr_model
    model = factory(X, y, err, n_band=2, survey_idx=jnp.arange(n) % 3,
        seeing_covariate=jnp.linspace(-.4, .8, n), zero_mean=False,
        has_jitter=True, disk_order=3)
    return model, params, X


def scipy_covariance(kernel, x, u):
    # General Schur Lyapunov/expm oracle; does not use production transitions,
    # stationary-covariance solver, or QSM assembly.
    A = np.asarray(kernel.design_matrix())
    Q = np.zeros_like(A)
    base = kernel._base()
    a, p = map(np.asarray, (base.design_matrix(), base.stationary_covariance()))
    Q[:2, :2] = -(a @ p + p @ a.T)
    P = solve_continuous_lyapunov(A, -Q)
    ends = np.array([sl.stop - 1 for sl in kernel._chain_slices()[0]])
    stds = np.sqrt(P[ends, ends])
    h = np.asarray(jax.vmap(lambda b: kernel._observation_model_with_stds(
        (0., b), jnp.asarray(stds)))(jnp.arange(2)))
    out = np.empty((len(x[0]), len(u[0])))
    cache = {}
    for i, (t, b) in enumerate(zip(*map(np.asarray, x))):
        for j, (v, c) in enumerate(zip(*map(np.asarray, u))):
            dt = abs(float(t - v))
            if dt not in cache:
                cache[dt] = expm(A * dt)
            late, early = (b, c) if t >= v else (c, b)
            out[i, j] = h[late] @ cache[dt] @ P @ h[early]
    return out


@pytest.mark.parametrize("variant", ["slb"])
def test_marginals_training_order_and_loo_against_scipy(variant):
    model, params, X = example(variant)
    gp, inds = model._build_gp(params)
    y = model._observed_y_sorted(params, inds)
    rng = np.random.default_rng(18)
    t = np.r_[[-500., 2500., X[0][0], X[0][0]], rng.uniform(0, 2200, 125)]
    query = jnp.asarray(t), jnp.asarray(np.arange(len(t)) % 2)
    C = scipy_covariance(gp.kernel, gp.X, gp.X) + np.diag(gp.noise.diagonal())
    alpha = np.linalg.solve(C, y - gp.loc)
    cross = scipy_covariance(gp.kernel, query, gp.X)
    prior = np.diag(scipy_covariance(gp.kernel, query, query))
    expected_mean = np.asarray(jax.vmap(gp.mean_function)(query)) + cross @ alpha
    expected_var = prior + np.sqrt(np.finfo(float).eps) - np.sum(
        np.linalg.solve(np.linalg.cholesky(C), cross.T)**2, axis=0)
    mean, std = model.pred(params, query)
    np.testing.assert_allclose(mean, expected_mean, atol=2e-10, rtol=2e-8)
    np.testing.assert_allclose(std**2, expected_var, atol=2e-11, rtol=2e-8)
    training_cross = scipy_covariance(gp.kernel, X, gp.X)
    np.testing.assert_allclose(model.pred_training_mean(params),
        jax.vmap(gp.mean_function)(X) + training_cross @ alpha, atol=2e-10)
    C += np.eye(len(y)) * (1e-10 * max(np.nanmedian(np.diag(C)), 1.))
    precision = np.linalg.inv(C)
    expected_loo = precision @ (y - gp.loc) / np.sqrt(np.diag(precision))
    np.testing.assert_allclose(regularized_loo_residuals(gp.solver.matrix, y-gp.loc),
                               expected_loo, atol=2e-9, rtol=2e-8)
    diagnostics = fit.compute_loo_short_lag_residual_diagnostics(model,
        {k: np.stack([v, v]) for k, v in params.items()},
        {"object_id": "synthetic", "z": 1.}, ["g", "r"])
    expected_bins = fit.binned_loo_residual_pair_correlation(
        np.asarray(gp.X[0]) / 2, np.asarray(gp.X[1]), expected_loo,
        bin_edges=fit.LOO_RESIDUAL_RF_BIN_EDGES, bands=["g", "r"])
    assert diagnostics["loo_resid_valid"]
    np.testing.assert_allclose(diagnostics["loo_chi2_eff"], np.mean(expected_loo**2), rtol=2e-8)
    for key in expected_bins:
        np.testing.assert_allclose(diagnostics[key], expected_bins[key],
                                   rtol=2e-8, atol=2e-9, equal_nan=True)
    empty = (jnp.empty(0), jnp.empty(0, dtype=jnp.int32))
    assert model.pred(params, empty)[0].shape == (0,)


@pytest.mark.parametrize("variant", ["slb"])
def test_training_mean_restores_both_permutations(monkeypatch, variant):
    model, params, X = example(variant)
    original = type(model)._build_gp

    def reordered(self, p):
        gp, inds = original(self, p)
        order = jnp.lexsort((-jnp.arange(len(gp.loc)), gp.X[0]))
        gp = GaussianProcess(gp.kernel, tuple(x[order] for x in gp.X),
            diag=gp.noise.diagonal()[order], mean=gp.mean_function, assume_sorted=True)
        return gp, inds[order]

    monkeypatch.setattr(type(model), "_build_gp", reordered)
    gp, inds = model._build_gp(params)
    assert not np.array_equal(inds, np.arange(len(inds)))
    expected = jax.jit(lambda p: model._build_gp(p)[0].condition(
        model._observed_y_sorted(p, inds), X).gp.loc)(params)
    np.testing.assert_allclose(model.pred_training_mean(params), expected, atol=2e-11)
    # Refinement must not call the marginal-variance path.
    monkeypatch.setattr(type(model), "pred", lambda *a: pytest.fail("dense refinement"))
    obj = {"X": X, "y": model.y[model.input_inverse_order],
           "yerr": jnp.full(len(inds), .03)}
    pseudo, error = fit._flux_linearized_pseudo_data_from_prediction(obj, model, params)
    assert np.isfinite(pseudo).all() and np.isfinite(error).all()


@pytest.mark.parametrize("variant", ["slb"])
@pytest.mark.parametrize("pole_case", ["ordinary"])
def test_marginal_prediction_gradients(variant, pole_case):
    model, params, _ = example(variant, n=6)
    X = jnp.array([-2000., 0., 1e-10, 4000.]), jnp.array([1, 0, 1, 0])

    def values(theta, bounded):
        p = {**params, "tau_fast_driver": jnp.exp(theta[0]),
             "amp_cont_relflux": jnp.exp(theta[1]) * jnp.array([.1, .08])}
        gp, inds = model._build_gp(p)
        y = model._observed_y_sorted(p, inds)
        if bounded:
            mean, var = marginal_prediction(gp, y, X, block_size=3)
        else:
            pred = gp.condition(y, X).gp
            mean, var = pred.loc, pred.variance
        return jnp.sum(mean), jnp.sum(var)

    theta = jnp.array([np.log(.05 if pole_case == "ordinary" else 2.), 0.])
    for component in [0, 1]:
        actual = jax.jit(jax.value_and_grad(lambda p: values(p, True)[component]))(theta)
        expected = jax.jit(jax.value_and_grad(lambda p: values(p, False)[component]))(theta)
        assert all(np.isfinite(x).all() for x in actual)
        for x, y in zip(actual, expected):
            np.testing.assert_allclose(x, y, atol=2e-9, rtol=2e-7)
