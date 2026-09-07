import numpy as np
import pytest
from types import SimpleNamespace
import jax
import jax.numpy as jnp
from numpyro.handlers import seed, trace
from scipy.linalg import expm, solve_continuous_lyapunov
from tinygp import GaussianProcess

from qvc.light_curve.multiband_model_shared_latent_blr import (
    SharedLatentDiskBLRQS,
    SharedLatentDiskBLRRelativeFluxModel,
    continuum_effective_timescale,
    make_multiband_shared_latent_blr_model,
)
from qvc.light_curve.fit_light_curves import (
    _flux_linearized_pseudo_data_from_prediction,
    build_single_object_model_mag_flux_linearized,
)


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


def _prediction_kernel(disk_order=3, tau_fast=0.05, tau_slow=32.0, n_band=2):
    return SharedLatentDiskBLRQS(
        tau_fast=jnp.array([tau_fast]),
        tau_slow=jnp.array([tau_slow]),
        lag_disk=jnp.array([0.41, 0.6, 0.86])[:n_band],
        lag_blr=jnp.array([36.4, 63.1, 83.1])[:n_band],
        amp_cont=jnp.array([0.147, 0.128, 0.113])[:n_band],
        amp_blr=jnp.array([0.037, 0.05, 0.045])[:n_band],
        disk_order=disk_order,
        blr_order=3,
    )


def _independent_covariance(kernel, X1, X2):
    """Build the physical process in a different basis, using only parameters.

    The driver state is (x, velocity / omega), with unit stationary variance
    for x. Each causal Erlang chain is driven directly by x. No QVC state
    matrices, transitions, covariance solver, or loadings enter this oracle.
    """

    fast, slow = float(kernel.tau_fast[0]), float(kernel.tau_slow[0])
    omega, gamma = 1.0 / np.sqrt(fast * slow), 1.0 / fast + 1.0 / slow
    n_band = len(kernel.lag_disk)
    orders = [kernel.disk_order] * n_band + [kernel.blr_order] * n_band
    lags = np.concatenate([kernel.lag_disk, kernel.lag_blr])
    size = 2 + sum(orders)
    A = np.zeros((size, size))
    A[:2, :2] = [[0.0, omega], [-omega, -gamma]]
    Q = np.zeros_like(A)
    Q[1, 1] = 2.0 * gamma
    offset, endpoints = 2, []
    for lag, order in zip(lags, orders):
        rate = order / lag
        for j in range(order):
            A[offset + j, offset + j] = -rate
            A[offset + j, 0 if j == 0 else offset + j - 1] = rate
        offset += order
        endpoints.append(offset - 1)
    P = solve_continuous_lyapunov(A, -Q)
    H = np.zeros((n_band, size))
    for band in range(n_band):
        disk, blr = endpoints[band], endpoints[n_band + band]
        H[band, disk] = float(kernel.amp_cont[band]) / np.sqrt(P[disk, disk])
        H[band, blr] = float(kernel.amp_blr[band]) / np.sqrt(P[blr, blr])

    times1, bands1 = map(np.asarray, X1)
    times2, bands2 = map(np.asarray, X2)
    covariance = np.empty((len(times1), len(times2)))
    transitions = {}
    for i, (t1, b1) in enumerate(zip(times1, bands1)):
        for j, (t2, b2) in enumerate(zip(times2, bands2)):
            dt = abs(float(t1 - t2))
            if dt not in transitions:
                transitions[dt] = expm(A * dt)
            late, early = (b1, b2) if t1 >= t2 else (b2, b1)
            covariance[i, j] = H[late] @ transitions[dt] @ P @ H[early]
    return covariance


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


def test_conditional_prediction_matches_dense_gp_regression():
    """Regression: predictions must use the causal (non-reversible) cross-covariance.

    tinygp's inherited ``to_general_qsm`` assumes ``transition_matrix`` returns
    the transpose of the forward transition; this kernel returns the forward
    transition itself, so conditional means computed through the inherited
    path were silently wrong (errors larger than the signal amplitude).
    """

    kernel = SharedLatentDiskBLRQS(
        tau_fast=jnp.array([0.05]),
        tau_slow=jnp.array([32.0]),
        lag_disk=jnp.array([0.41, 0.6, 0.86]),
        lag_blr=jnp.array([36.4, 63.1, 83.1]),
        amp_cont=jnp.array([0.147, 0.128, 0.113]),
        amp_blr=jnp.array([0.037, 0.05, 0.045]),
        disk_order=3,
        blr_order=3,
    )
    rng = np.random.default_rng(0)
    n = 80
    t = np.sort(rng.uniform(0.0, 2000.0, n))
    b = rng.integers(0, 3, n)
    order = np.argsort(t + 1e-9 * b)
    t, b = t[order], b[order]
    X = (jnp.asarray(t), jnp.asarray(b))
    diag = jnp.full(n, 0.05**2)
    gp = GaussianProcess(kernel, X, diag=diag, assume_sorted=True)
    y = np.asarray(gp.sample(jax.random.PRNGKey(1)))

    K = np.asarray(
        jax.vmap(
            lambda ti, bi: jax.vmap(lambda tj, bj: kernel.evaluate((ti, bi), (tj, bj)))(*X)
        )(*X)
    )
    alpha = np.linalg.solve(K + np.diag(np.asarray(diag)), y)

    # At the training inputs and at shifted test inputs.
    for shift in (0.0, 0.3):
        X_test = (X[0] + shift, X[1])
        Ks = np.asarray(
            jax.vmap(
                lambda ti, bi: jax.vmap(lambda tj, bj: kernel.evaluate((ti, bi), (tj, bj)))(*X)
            )(*X_test)
        )
        mu_dense = Ks @ alpha
        _, cond = gp.condition(y, X_test, diag=0.0)
        np.testing.assert_allclose(np.asarray(cond.loc), mu_dense, rtol=0.0, atol=1e-8)
        var_dense = np.diag(
            np.asarray(
                jax.vmap(
                    lambda ti, bi: jax.vmap(lambda tj, bj: kernel.evaluate((ti, bi), (tj, bj)))(*X_test)
                )(*X_test)
            )
        ) - np.einsum("ij,ji->i", Ks, np.linalg.solve(K + np.diag(np.asarray(diag)), Ks.T))
        np.testing.assert_allclose(np.asarray(cond.variance), var_dense, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize(
    "disk_order,tau_fast,tau_slow,n_band",
    [(1, 1.0, 180.0, 1), (3, 0.05, 32.0, 2), (4, 15.0, 180.0, 3)],
)
def test_general_qsm_matches_independent_state_space(
    disk_order, tau_fast, tau_slow, n_band, monkeypatch
):
    kernel = _prediction_kernel(disk_order, tau_fast, tau_slow, n_band)
    # The production GP sorts by time, without a secondary band sort at ties.
    X = (
        jnp.array([0.0, 0.0, 0.001, 12.0, 12.0, 70.0, 150.0, 400.0]),
        jnp.array([2, 0, 1, 2, 1, 0, 2, 1]) % n_band,
    )
    # Unsorted queries, simultaneous bands, and extrapolation at both ends.
    Y = (
        jnp.array([600.0, -300.0, 12.0, 12.0, 55.0, 0.0]),
        jnp.array([1, 0, 0, 1, 2, 1]) % n_band,
    )
    if n_band == 1:
        # Identical query coordinates describe the same latent random variable.
        # Keep the conditional covariance nonsingular without query noise.
        Y = tuple(x[jnp.array([0, 1, 2, 4, 5])] for x in Y)
    K = _independent_covariance(kernel, X, X)
    cross = _independent_covariance(kernel, Y, X)
    Kyy = _independent_covariance(kernel, Y, Y)
    rhs = jnp.asarray(np.column_stack([np.sin(np.arange(8)), np.cos(np.arange(8))]))

    for query, expected in [(X, K), (Y, cross)]:
        general = kernel.to_general_qsm(query, X)
        np.testing.assert_allclose(general @ jnp.eye(8), expected, rtol=2e-8, atol=2e-10)
        for values in [rhs[:, 0], rhs]:
            np.testing.assert_allclose(
                kernel.matmul(query, X, values), expected @ values,
                rtol=2e-8, atol=2e-10,
            )
    np.testing.assert_allclose(kernel.matmul(X, rhs), K @ rhs, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(kernel.matmul(X, y=rhs), K @ rhs, rtol=2e-8, atol=2e-10)

    y = 0.08 * rhs[:, 0]
    noise = jnp.linspace(0.015, 0.025, 8) ** 2
    C = K + np.diag(noise)
    alpha = np.linalg.solve(C, y)
    gp = GaussianProcess(kernel, X, diag=noise, assume_sorted=True)
    expected_logp = -0.5 * (y @ alpha + np.linalg.slogdet(C)[1] + 8 * np.log(2 * np.pi))
    np.testing.assert_allclose(gp.log_probability(y), expected_logp, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        gp.condition(y, X).gp.loc, gp.condition(y).gp.loc, rtol=1e-10, atol=1e-10
    )
    conditional = gp.condition(y, Y, diag=0.0).gp
    np.testing.assert_allclose(conditional.loc, cross @ alpha, rtol=2e-8, atol=2e-10)
    expected_cov = Kyy - cross @ np.linalg.solve(C, cross.T)
    np.testing.assert_allclose(conditional.covariance, expected_cov, rtol=2e-8, atol=2e-10)

    def forbid_dense_evaluation(*args):
        raise AssertionError("Cross multiplication must use the generalized QSM")

    monkeypatch.setattr(SharedLatentDiskBLRQS, "evaluate", forbid_dense_evaluation)
    np.testing.assert_allclose(kernel.matmul(Y, X, rhs), cross @ rhs, rtol=2e-8, atol=2e-10)


def test_prediction_gradients_match_independent_finite_differences():
    X = (jnp.array([0.0, 2000.0]), jnp.array([0, 1]))
    Y = (jnp.array([-2000.0, 0.3, 20.0, 2000.3, 4000.0]), jnp.array([0, 1, 0, 0, 1]))
    y = jnp.array([0.1, -0.2])
    log_params = np.log([0.05, 32.0, 0.41, 0.6, 36.4, 63.1, 0.147, 0.128, 0.037, 0.05])

    def build(values):
        p = jnp.exp(values)
        return SharedLatentDiskBLRQS(
            tau_fast=p[:1], tau_slow=p[1:2], lag_disk=p[2:4], lag_blr=p[4:6],
            amp_cont=p[6:8], amp_blr=p[8:10], disk_order=3, blr_order=3,
        )

    def predictions(values):
        kernel = build(values)
        gp = GaussianProcess(kernel, X, diag=0.0025, assume_sorted=True)
        return jnp.array([
            jnp.sum(kernel.matmul(Y, X, y)),
            jnp.sum(kernel(Y, X) @ y),
            jnp.sum(gp.condition(y, Y, diag=0.0025).gp.loc),
        ])

    def independent_predictions(values):
        kernel = build(values)
        cross = _independent_covariance(kernel, Y, X)
        C = _independent_covariance(kernel, X, X) + 0.0025 * np.eye(2)
        return np.array([(cross @ y).sum(), (cross @ y).sum(), (cross @ np.linalg.solve(C, y)).sum()])

    gradients = np.asarray(jax.jit(jax.jacrev(predictions))(jnp.asarray(log_params)))
    assert np.isfinite(gradients).all()
    steps = 1e-4 * np.eye(len(log_params))
    finite_differences = np.column_stack([
        (independent_predictions(log_params + step) - independent_predictions(log_params - step)) / 2e-4
        for step in steps
    ])
    np.testing.assert_allclose(predictions(log_params), independent_predictions(log_params), rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(gradients, finite_differences, rtol=2e-5, atol=2e-8)


def test_flux_refinement_uses_independently_verified_conditional_mean():
    kernel = _prediction_kernel()
    # Exercise prediction back into the original, unsorted observation order.
    X = (
        jnp.array([70.0, 0.0, 0.001, 12.0, 12.0, 0.0, 400.0, 150.0]),
        jnp.array([1, 1, 0, 1, 0, 0, 1, 0]),
    )
    y = 0.08 * jnp.sin(jnp.arange(8))
    errors = jnp.linspace(0.015, 0.025, 8)
    params = dict(
        tau_fast_driver=kernel.tau_fast[0], tau_slow_driver=kernel.tau_slow[0],
        lag_disk=kernel.lag_disk, lag_blr=kernel.lag_blr,
        amp_cont_relflux=kernel.amp_cont, amp_blr_relflux=kernel.amp_blr,
    )
    model = make_multiband_shared_latent_blr_model(
        X, y, errors, n_band=2, zero_mean=True, has_jitter=False, disk_order=3,
    )
    K = _independent_covariance(kernel, X, X)
    expected_mean = K @ np.linalg.solve(K + np.diag(errors**2), y)
    mean, _ = model.pred(params, X)
    np.testing.assert_allclose(mean, expected_mean, rtol=2e-8, atol=2e-10)

    c = 2.5 / np.log(10.0)
    obj = dict(X=X, y=-c * jnp.log1p(y), yerr=jnp.full(8, 0.03))
    pseudo_y, pseudo_errors = _flux_linearized_pseudo_data_from_prediction(obj, model, params)
    derivative = -c / (1.0 + expected_mean)
    expected_pseudo_y = expected_mean + (obj["y"] + c * np.log1p(expected_mean)) / derivative
    np.testing.assert_allclose(pseudo_y, expected_pseudo_y, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(pseudo_errors, obj["yerr"] / abs(derivative), rtol=2e-8, atol=2e-10)
