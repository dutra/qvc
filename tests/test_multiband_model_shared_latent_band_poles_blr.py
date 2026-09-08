"""Independent solver, inference and catalog checks for the experimental variant."""

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.handlers import seed, trace
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.integrate import quad
from tinygp import GaussianProcess
from tinygp.solvers import QuasisepSolver

from qvc.light_curve import fit_light_curves as fit
from qvc.light_curve import multiband_fit_utils as utils
from qvc.light_curve.band_poles_fit import (
    build_explicit_band_poles_params,
    build_single_object_model_band_poles,
    add_band_poles_prediction_params,
)
from qvc.light_curve.multiband_model_shared_latent_band_poles_blr import (
    VARIANT,
    SharedLatentBandPolesBLRQS,
    continuum_effective_timescale,
    make_multiband_shared_latent_band_poles_blr_model,
)
from qvc.light_curve.multiband_model_shared_latent_blr import SharedLatentDiskBLRQS

jax.config.update("jax_enable_x64", True)
X = (jnp.array([0.0, 0.0, 0.2, 7.0, 100.0, 2000.0]), jnp.array([1, 0, 1, 0, 1, 0]))
XT = (jnp.array([2500.0, -500.0, 7.0, 0.1, 0.0]), jnp.array([0, 1, 1, 0, 1]))
Y = jnp.array([0.02, 0.01, -0.01, 0.04, -0.03, 0.01])
D = jnp.linspace(0.01, 0.025, len(Y)) ** 2


@pytest.fixture(params=["analytic", "expm"])
def transition(request):
    return request.param


def kernel(theta=None, order=3, transition="analytic"):
    if theta is None:
        theta = jnp.array(
            [np.log(0.05), np.log(300.0), 0.51, np.log(3.0), np.log(60.0), 0.0]
        )
    f, suv, ld, lb = jnp.exp(theta[jnp.array([0, 1, 3, 4])])
    sb = suv * (jnp.array([0.7, 1.5])) ** theta[2]
    return SharedLatentBandPolesBLRQS(
        jnp.atleast_1d(f),
        sb,
        jnp.atleast_1d(suv),
        ld * jnp.array([1.0, 1.5]),
        lb * jnp.array([1.0, 1.3]),
        jnp.exp(theta[5]) * jnp.array([0.12, 0.08]),
        jnp.array([0.02, 0.03]),
        disk_order=order,
        blr_order=3,
        transition=transition,
    )


def independent_state(k):
    """Different basis: parallel OU coordinates with shared white noise.

    Uses only physical parameters, never production A/P/transition/loadings.
    Suitable for tests whose band/reference poles differ from the common pole.
    """
    f = float(k.tau_fast[0])
    ss = np.r_[k.tau_slow_band, k.tau_slow_uv]
    B = len(k.lag_disk)
    ndriver = B + 2
    orders = [k.disk_order] * B + [k.blr_order] * B
    n = ndriver + sum(orders)
    A = np.zeros((n, n))
    Q = np.zeros((n, n))
    Q[:ndriver, :ndriver] = 1.0
    A[np.arange(ndriver), np.arange(ndriver)] = -1 / np.r_[f, ss]
    start, ends = ndriver, []
    for i, (order, L) in enumerate(zip(orders, np.r_[k.lag_disk, k.lag_blr])):
        q = order / L
        parent = i + 1 if i < B else B + 1
        coeff = np.sqrt(2 * (f + ss[parent - 1])) / (ss[parent - 1] - f)
        for j in range(order):
            A[start + j, start + j] = -q
            if j:
                A[start + j, start + j - 1] = q
            else:
                A[start, 0], A[start, parent] = -q * coeff, q * coeff
        start += order
        ends.append(start - 1)
    P = solve_continuous_lyapunov(A, -Q)
    H = np.zeros((B, n))
    for b in range(B):
        H[b, ends[b]] = float(k.amp_cont[b]) / np.sqrt(P[ends[b], ends[b]])
        H[b, ends[B + b]] = float(k.amp_blr[b]) / np.sqrt(P[ends[B + b], ends[B + b]])
    return A, P, H


def independent_cascade_state(k):
    """SciPy SDE oracle that remains nonsingular in its formulas at f=s.

    Constructed from physical inputs, without production matrix/P/loadings.
    Unlike the production recurrence, P uses a general Schur Lyapunov solve.
    """
    B = len(k.lag_disk)
    orders = [k.disk_order] * B + [k.blr_order] * B
    n = B + 2 + sum(orders)
    A = np.zeros((n, n))
    Q = np.zeros_like(A)
    f = float(k.tau_fast[0])
    A[0, 0] = -1 / f
    Q[0, 0] = 2 / f
    for i, s in enumerate(np.r_[k.tau_slow_band, k.tau_slow_uv], 1):
        A[i, 0] = 1 / s
        A[i, i] = -1 / s
    start = B + 2
    ends = []
    for i, (order, L) in enumerate(zip(orders, np.r_[k.lag_disk, k.lag_blr])):
        parent = i + 1 if i < B else B + 1
        for j in range(order):
            A[start + j, start + j] = -order / L
            A[start + j, start + j - 1 if j else parent] = order / L
        start += order
        ends.append(start - 1)
    P = solve_continuous_lyapunov(A, -Q)
    H = np.zeros((B, n))
    for b in range(B):
        H[b, ends[b]] = float(k.amp_cont[b]) / np.sqrt(P[ends[b], ends[b]])
        H[b, ends[B + b]] = float(k.amp_blr[b]) / np.sqrt(P[ends[B + b], ends[B + b]])
    return A, P, H


def dense(k, x1=X, x2=X, *, state_builder=independent_state):
    A, P, H = state_builder(k)
    result = np.empty((len(x1[0]), len(x2[0])))
    cache = {}
    for i, (t, b) in enumerate(zip(*map(np.asarray, x1))):
        for j, (u, c) in enumerate(zip(*map(np.asarray, x2))):
            dt = abs(float(t - u))
            if dt not in cache:
                cache[dt] = expm(A * dt)
            late, early = (b, c) if t >= u else (c, b)
            result[i, j] = H[late] @ cache[dt] @ P @ H[early]
    return result


@pytest.mark.parametrize(
    "eta,order", [(0.51, 3), (0.0, 3), (-0.4, 1), (0.5, 2), (-0.3, 5)]
)
def test_covariance_solvers_predictions_and_moments(eta, order, transition):
    theta = jnp.array(
        [np.log(0.05), np.log(300.0), eta, np.log(0.15), np.log(900.0), 0.0]
    )
    k = kernel(theta, order, transition)
    K = dense(k)
    Ks = dense(k, XT, X)
    Kss = dense(k, XT, XT)
    np.testing.assert_allclose(k.to_symm_qsm(X).to_dense(), K, atol=2e-10, rtol=2e-8)
    np.testing.assert_allclose(
        k.to_general_qsm(XT, X) @ jnp.eye(len(Y)), Ks, atol=2e-10, rtol=2e-8
    )
    np.testing.assert_allclose(k.matmul(X, Y), K @ Y, atol=2e-10)
    gp = GaussianProcess(k, X, diag=D)
    assert isinstance(gp.solver, QuasisepSolver)
    noisy = K + np.diag(D)
    alpha = np.linalg.solve(noisy, Y)
    logp = -0.5 * (Y @ alpha + np.linalg.slogdet(noisy)[1] + len(Y) * np.log(2 * np.pi))
    np.testing.assert_allclose(gp.log_probability(Y), logp, atol=2e-8)
    np.testing.assert_allclose(
        gp.solver.solve_triangular(Y),
        np.linalg.solve(np.linalg.cholesky(noisy), Y),
        atol=2e-8,
    )
    for points, cross, test in [(None, K, K), (XT, Ks, Kss)]:
        pred = gp.condition(Y, points, diag=0.0).gp
        np.testing.assert_allclose(pred.loc, cross @ alpha, atol=2e-9)
        np.testing.assert_allclose(
            pred.variance,
            np.diag(test - cross @ np.linalg.solve(noisy, cross.T)),
            atol=2e-10,
        )
    A = np.asarray(k.design_matrix())
    P = np.asarray(k.stationary_covariance())
    Q = np.zeros_like(A)
    Q[0, 0] = 2 / float(k.tau_fast[0])
    np.testing.assert_allclose(A @ P + P @ A.T, -Q, atol=2e-10)
    assert np.linalg.eigvalsh(P).min() > -1e-10
    phi = lambda t: np.asarray(k.transition_matrix((0.0, 0), (t, 1)))
    for dt in [0.0, 1e-7, 0.05, 13.0, 2000.0]:
        np.testing.assert_allclose(phi(dt), expm(A * dt), atol=2e-9, rtol=2e-7)
    np.testing.assert_allclose(phi(100.0), phi(13.0) @ phi(87.0), atol=2e-9)
    ar, pr, hr = independent_state(k)
    tau = [-h @ np.linalg.solve(ar, pr @ h) / (h @ pr @ h) for h in hr]
    np.testing.assert_allclose(k.effective_timescales(), tau, rtol=2e-8)
    if eta == 0:
        old = SharedLatentDiskBLRQS(
            k.tau_fast,
            k.tau_slow_uv,
            k.lag_disk,
            k.lag_blr,
            k.amp_cont,
            k.amp_blr,
            disk_order=order,
            blr_order=3,
        )
        np.testing.assert_allclose(old.to_symm_qsm(X).to_dense(), K, atol=2e-10)


@pytest.mark.parametrize("slow", [2.0, 2.0 + 1e-10, 0.1])
def test_coincident_near_coincident_and_crossed_poles(slow, transition):
    k = SharedLatentBandPolesBLRQS(
        jnp.array([2.0]),
        jnp.array([slow]),
        jnp.array([2.0]),
        jnp.array([6.0]),
        jnp.array([6.0]),
        jnp.ones(1),
        jnp.zeros(1),
        disk_order=3,
        blr_order=3,
        transition=transition,
    )
    tau = float(k.continuum_effective_timescales()[0])
    # Independent frequency-domain integral of the cascade's PSD; arbitrary
    # driver/filter coincidences require no special limiting expressions.
    shape = lambda w: 1 / (
        (1 + (2 * w) ** 2) * (1 + (slow * w) ** 2) * (1 + (2 * w) ** 2) ** 3
    )
    integral = quad(shape, 0, np.inf, epsabs=1e-11)[0]
    np.testing.assert_allclose(tau, np.pi / (2 * integral), rtol=2e-8)
    qx = (jnp.array([0.0, 1.0, 20.0]), jnp.zeros(3, dtype=int))
    gp = GaussianProcess(k, qx, diag=0.01)
    assert np.isfinite(gp.log_probability(jnp.zeros(3)))
    assert np.all(np.isfinite(gp.condition(jnp.zeros(3), qx, diag=0.0).gp.variance))

    def quantities(log_s):
        varied = eqx.tree_at(
            lambda q: q.tau_slow_band, k, jnp.atleast_1d(jnp.exp(log_s))
        )
        process = GaussianProcess(varied, qx, diag=0.01)
        pred = process.condition(jnp.array([0.1, -0.1, 0.2]), qx, diag=0.0).gp
        return jnp.stack(
            [
                process.log_probability(jnp.array([0.1, -0.1, 0.2])),
                jnp.sum(pred.loc),
                jnp.sum(pred.variance),
            ]
        )

    assert np.all(np.isfinite(jax.jacrev(quantities)(jnp.log(slow))))


def statistics(theta, transition="analytic"):
    k = kernel(theta, transition=transition)
    gp = GaussianProcess(k, X, diag=D)
    p = gp.condition(Y, XT, diag=0.0).gp
    weights = jnp.arange(1.0, len(XT[0]) + 1)
    return jnp.stack(
        [
            gp.log_probability(Y),
            weights @ p.loc,
            weights @ p.variance,
            k.evaluate((-2000.0, 0), (2000.0, 1)),
        ]
    )


statistics_jacobian = jax.jit(jax.jacrev(statistics), static_argnames=("transition",))


def independent_statistics(theta, state_builder=independent_state):
    k = kernel(jnp.asarray(theta))
    K = dense(k, state_builder=state_builder) + np.diag(D)
    cross = dense(k, XT, X, state_builder=state_builder)
    test = dense(k, XT, XT, state_builder=state_builder)
    alpha = np.linalg.solve(K, Y)
    weights = np.arange(1.0, len(XT[0]) + 1)
    return np.array(
        [
            -0.5 * (Y @ alpha + np.linalg.slogdet(K)[1] + len(Y) * np.log(2 * np.pi)),
            weights @ (cross @ alpha),
            weights @ np.diag(test - cross @ np.linalg.solve(K, cross.T)),
            dense(
                k,
                (np.array([-2000.0]), np.array([0])),
                (np.array([2000.0]), np.array([1])),
                state_builder=state_builder,
            )[0, 0],
        ]
    )


@pytest.mark.parametrize("eta", [0.51, 0.0])
def test_likelihood_and_prediction_gradients_against_dense(eta, transition):
    theta = np.array(
        [np.log(0.05), np.log(300.0), eta, np.log(0.15), np.log(900.0), 0.0]
    )
    got = np.asarray(statistics_jacobian(theta, transition))
    assert np.all(np.isfinite(got))
    expected = np.column_stack(
        [
            (
                independent_statistics(theta + e * 1e-4)
                - independent_statistics(theta - e * 1e-4)
            )
            / 2e-4
            for e in np.eye(6)
        ]
    )
    np.testing.assert_allclose(got, expected, rtol=4e-4, atol=4e-6)


def object_data():
    bands = ("g", "i")
    obj = dict(
        object_id="band-pole-test",
        X=X,
        y=Y,
        yerr=jnp.sqrt(D),
        survey_idx=np.zeros(len(Y), dtype=int),
        z=1.0,
        bands=bands,
        clean_bands=bands,
        mags_means=np.array([20.0, 20.0]),
        log_jitter_active_mask=np.ones((2, 3), bool),
        survey_offset_active_mask=np.zeros((2, 3), bool),
        survey_names=("sdss", "ps1", "ztf"),
    )
    lam = jnp.array([utils.lambda_pivot[b] / 2 for b in bands])
    return obj, lam


def trace_values(transition="analytic"):
    obj, lam = object_data()
    model = build_single_object_model_band_poles(
        obj,
        lam,
        np.full((2, 3), -4.0),
        disable_linear_trend=True,
        band_poles_transition=transition,
    )
    sites = trace(seed(model, jax.random.PRNGKey(4))).get_trace()
    values = {
        k: np.asarray(v["value"])
        for k, v in sites.items()
        if v["type"] in {"sample", "deterministic"} and not v.get("is_observed", False)
    }
    return obj, lam, sites, values


def test_builder_priors_and_explicit_parameters():
    obj, lam, sites, values = trace_values()
    assert float(sites["eta_tau"]["fn"].loc) == 0.5
    assert float(sites["eta_tau"]["fn"].scale) == 0.5
    prior = sites["log_tau_slow_uv_driver"]["fn"]
    np.testing.assert_allclose(prior.low, np.log(10 * (1 + obj["z"])))
    np.testing.assert_allclose(prior.high, np.log(1e4 * (1 + obj["z"])))
    assert np.isfinite(sites["loglike"]["fn"].log_factor)
    p = add_band_poles_prediction_params(values, lam)
    expected = p["tau_slow_uv_driver"] * (np.asarray(lam) / 2500.0) ** p["eta_tau"]
    np.testing.assert_allclose(p["tau_slow_band"], expected)
    np.testing.assert_allclose(p["tau_fast_band"], p["tau_fast_driver"])
    ld = p["lag0"] * (2500.0 / p["lambda_center_rf"]) ** (4 / 3)
    np.testing.assert_allclose(
        np.exp(p["log_tau_uv"]),
        continuum_effective_timescale(
            p["tau_fast_driver"], p["tau_slow_uv_driver"], ld
        ),
    )
    # Rebuilding explicit samples must not drift their amplitude conventions.
    again = add_band_poles_prediction_params(p, lam)
    for key in ["log_sigma_uv", "amp_cont_relflux", "log_tau_uv", "tau_slow_band"]:
        np.testing.assert_allclose(again[key], p[key])


@pytest.mark.parametrize(
    "profile,expected",
    [(None, "modified"), ("default", "default"), ("modified", "modified")],
)
def test_profile_resolution(profile, expected):
    obj, lam = object_data()
    sites = trace(
        seed(
            build_single_object_model_band_poles(
                obj, lam, None, eta_prior_profile=profile
            ),
            jax.random.PRNGKey(3),
        )
    ).get_trace()
    prior = fit.eta_tau_prior(expected)
    np.testing.assert_allclose(
        sites["eta_tau"]["fn"].log_prob(0.4), prior.log_prob(0.4)
    )
    np.testing.assert_allclose(
        sites["eta_sigma"]["fn"].log_prob(-0.6),
        fit.eta_sigma_prior(expected).log_prob(-0.6),
    )
    assert fit.resolve_eta_prior_profile(profile, VARIANT) == expected
    assert fit.resolve_eta_prior_profile(None, "shared_latent_blr") == "default"
    assert (
        fit.resolve_eta_prior_profile(None, "mag_flux_linearized_erlang") == "default"
    )


def test_catalog_hubble_definitions_and_serialization(
    tmp_path, monkeypatch, transition
):
    obj, lam, sites, raw = trace_values(transition)
    draws = {k: np.stack([v] * 5) for k, v in raw.items()}
    draws["log_tau_slow_uv_driver"] += np.linspace(-0.2, 0.2, 5)
    draws["log_sigma_center0"] += np.linspace(-0.03, 0.03, 5)
    # A deliberately crossed band remains unswapped and is reported.
    draws["eta_tau"] = np.linspace(-9.0, -7.0, 5)
    p = add_band_poles_prediction_params(draws, lam)
    flat = utils.flatten_flat_samples_per_band(p, obj["bands"], obj["survey_names"])
    kls = fit.compute_parameter_kls(
        flat,
        bands=obj["bands"],
        survey_names=obj["survey_names"],
        t_ref=X[0],
        z=obj["z"],
        lambda_center_rf=float(raw["lambda_center_rf"]),
        log_jitter_mean=np.full((2, 3), -4.0),
        model_variant=VARIANT,
        disable_linear_trend=True,
    )
    assert "eta_tau_kl" in kls
    assert "log_tau_slow_uv_driver_kl" in kls
    assert "log_tau_separation_raw_kl" in kls
    assert "log_tau_fast_center0_kl" not in kls
    assert "log_tau_slow_center0_kl" not in kls
    result = utils.process_samples(flat, obj, bands=obj["bands"], model_variant=VARIANT)
    from qvc.light_curve import merge_results

    payload = result[utils.LIGHT_CURVE_POSTERIOR_DRAW_PAYLOAD_KEY]
    assert payload["format"] == utils.LIGHT_CURVE_POSTERIOR_DRAW_FORMAT
    indices = payload["posterior_index"][:5]
    np.testing.assert_allclose(
        payload["log_sigma_uv"][:5], p["log_sigma_uv"][indices] / np.log(10), rtol=1e-6
    )
    sigma = p["log_sigma_uv"] / np.log(10)
    tau = p["log_tau_uv"] / np.log(10) - np.log10(2.0)
    vx, vy, cov = utils.regularize_cov_from_percentiles(
        *np.percentile(sigma, [16, 84]),
        *np.percentile(tau, [16, 84]),
        np.cov(sigma, tau)[0, 1],
    )
    np.testing.assert_allclose(result["cov_log_sigma_uv_log_tau_uv_rf"], cov)
    expected = p["log_tau_uv"] / np.log(10) - np.log10(1 + obj["z"])
    np.testing.assert_allclose(result["log_tau_uv_rf"], np.median(expected))
    for b in obj["bands"]:
        assert f"log_sigma_total_rms_band_{b}" in result
        assert f"log_tau_cont_band_{b}_RF" in result
        expected_cross = np.mean(flat[f"tau_slow_band_{b}"] < flat["tau_fast_driver"])
        assert result[f"band_pole_below_common_fraction_{b}"] == expected_cross
        np.testing.assert_allclose(
            result[f"log_tau_fast_band_{b}_RF"],
            np.median(np.log10(p["tau_fast_driver"]) - np.log10(2.0)),
        )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(utils, "prefix", "band-test", raising=False)
    monkeypatch.setattr(utils, "suffix", "new", raising=False)
    from qvc.light_curve.band_poles_fit import posterior_band_poles_moments

    moments = posterior_band_poles_moments(flat, obj["bands"])
    saved = {
        **p,
        "tau_continuum_band_obs": moments["tau_continuum"],
        "tau_total_band_obs": moments["tau_total"],
        "rms_total_band_relflux": moments["rms_total"],
    }
    utils.save_obj_samples_to_hdf5(
        saved,
        obj["object_id"],
        model_metadata={
            "eta_prior_profile": "modified",
            "band_poles_transition": transition,
        },
    )
    path = tmp_path / "results/samples/band-test/band-pole-test_new.h5"
    with h5py.File(path) as h:
        assert h.attrs["model_variant"] == VARIANT
        assert h.attrs["eta_prior_profile"] == "modified"
        assert h.attrs["band_poles_transition"] == transition
        assert "expm" in h.attrs["transition_implementation"]
        np.testing.assert_array_equal(h["eta_tau"][:], draws["eta_tau"])
        np.testing.assert_allclose(h["log_tau_uv"][:], p["log_tau_uv"])
        np.testing.assert_allclose(
            h["tau_continuum_band_obs"][:], moments["tau_continuum"]
        )
        np.testing.assert_allclose(h["tau_total_band_obs"][:], moments["tau_total"])
    rows = [dict(object_id=obj["object_id"], z=obj["z"], suffix="new")]
    merged = merge_results.collect_light_curve_posterior_draws(rows, path.parent)
    assert merged["format"] == utils.LIGHT_CURVE_POSTERIOR_DRAW_FORMAT
    np.testing.assert_allclose(
        merged["log_tau_uv_rf"][0, :5], tau[merged["posterior_index"][0, :5]], rtol=1e-6
    )
    rows[0].update(result)
    embedded = merge_results.collect_light_curve_posterior_draws(rows, path.parent)
    np.testing.assert_array_equal(merged["log_tau_uv_rf"], embedded["log_tau_uv_rf"])
    from qvc.light_curve.posterior_draws import read_light_curve_posterior_draw_group

    merged_path = tmp_path / "merged.h5"
    rows[0]["model_variant"] = VARIANT
    merge_results.write_quasars_to_h5_flat(
        rows, merged_path, posterior_draw_payload=embedded
    )
    with h5py.File(merged_path) as h:
        loaded = read_light_curve_posterior_draw_group(h)
    np.testing.assert_array_equal(loaded["log_tau_uv_rf"], embedded["log_tau_uv_rf"])
    from qvc.hubble.hubble_utils import (
        read_quasars_from_hdf5_flat,
        LIGHT_CURVE_LOG_TAU_RF_DRAW_COL,
    )

    frame = read_quasars_from_hdf5_flat(
        merged_path, include_light_curve_posterior_draws=True
    )
    assert frame.loc[0, "model_variant"] == VARIANT
    np.testing.assert_array_equal(
        frame.loc[0, LIGHT_CURVE_LOG_TAU_RF_DRAW_COL], embedded["log_tau_uv_rf"][0]
    )
    # Changing time units from observer to rest must commute with UV moments.
    f, s, l = 2.0, 400.0, 5.0
    np.testing.assert_allclose(
        continuum_effective_timescale(f * 3, s * 3, l * 3) / 3,
        continuum_effective_timescale(f, s, l),
    )


def test_refinement_prediction_and_psd_use_new_kernel():
    obj, lam, sites, raw = trace_values()
    p = add_band_poles_prediction_params(raw, lam)
    model = make_multiband_shared_latent_band_poles_blr_model(
        X, Y, jnp.sqrt(D), n_band=2, zero_mean=True, has_jitter=False
    )
    gp, inds = model._build_gp(p)
    assert isinstance(gp.kernel, SharedLatentBandPolesBLRQS)
    r, _ = model.pred(p, X)
    yp, ep = fit._flux_linearized_pseudo_data_from_prediction(obj, model, p)
    ratio = np.maximum(1 + np.asarray(r), fit.FLUX_LINEARIZED_MIN_TOTAL_FLUX_RATIO)
    deriv = -(2.5 / np.log(10)) / ratio
    np.testing.assert_allclose(yp, r + (Y + 2.5 * np.log10(ratio)) / deriv)
    assert np.all(np.isfinite(ep))
    psd = np.asarray(model.psd(p, jnp.array([0.001, 0.01, 0.1]), b=0))
    assert np.all(np.isfinite(psd)) and np.all(psd > 0)


@pytest.mark.parametrize(
    "flag", ["drw_parameterization", "use_fast_solver", "enforce_positive_flux_guard"]
)
def test_incompatible_modes_are_rejected(flag):
    obj, lam = object_data()
    with pytest.raises(ValueError):
        build_single_object_model_band_poles(obj, lam, None, **{flag: True})


@pytest.mark.parametrize(
    "strategy,expected_nuts", [("nuts_each", 2), ("svi_then_nuts", 1)]
)
def test_refinement_dispatch_preserves_eta_and_kernel(
    monkeypatch, strategy, expected_nuts, transition
):
    obj, lam, sites, raw = trace_values()
    calls = []
    from qvc.light_curve import band_poles_fit

    factory = make_multiband_shared_latent_band_poles_blr_model

    def check_factory(*args, **kwargs):
        assert kwargs["transition"] == transition
        model = factory(*args, **kwargs)
        assert model.transition == transition
        calls.append("factory")
        return model

    monkeypatch.setattr(
        band_poles_fit,
        "make_multiband_shared_latent_band_poles_blr_model",
        check_factory,
    )
    monkeypatch.setattr(
        fit, "make_multiband_shared_latent_band_poles_blr_model", check_factory
    )

    def svi(model, key, **kwargs):
        values = fit._model_params_at_values(model, key, raw)
        assert "tau_slow_uv_driver" in values
        assert float(values["eta_tau"]) == float(raw["eta_tau"])
        calls.append("svi")
        return raw, 1.0, {}

    def nuts(model, key, **kwargs):
        values = fit._model_params_at_values(model, key, raw)
        draws = {k: np.stack([v] * 4) for k, v in values.items()}
        calls.append("nuts")
        return (
            draws,
            {k: v[None] for k, v in draws.items()},
            dict(accept_prob=0.8, num_divergences=0, elapsed_sec=0.01),
        )

    monkeypatch.setattr(fit, "run_svi_warm_start", svi)
    monkeypatch.setattr(fit, "_run_nuts_inference", nuts)
    monkeypatch.setattr(fit, "print_and_validate_svi_warm_start", lambda *a, **kw: None)
    monkeypatch.setattr(
        fit, "summarize_final_hubble_nuts_posterior", lambda *a, **kw: {}
    )
    samples, chains, fitted, diagnostics, summary = (
        fit.run_iterated_mag_flux_linearized_inference(
            obj,
            lam,
            None,
            rng_key=jax.random.PRNGKey(2),
            fit_method="svi+nuts",
            num_warmup=2,
            num_samples=4,
            num_chains=1,
            chain_method="sequential",
            progress_bar=False,
            dense_mass=False,
            max_tree_depth=1,
            svi_steps=2,
            svi_lr=0.0003,
            disable_linear_trend=True,
            model_variant=VARIANT,
            band_poles_transition=transition,
            refinement_strategy=strategy,
            refinement_iters=2,
        )
    )
    assert calls.count("nuts") == expected_nuts
    assert calls.count("svi") == 2
    assert calls.count("factory") >= 4
    np.testing.assert_allclose(samples["eta_tau"], raw["eta_tau"])
    assert np.all(np.isfinite(samples["log_tau_uv"]))
    assert np.isfinite(diagnostics["flux_linearized_iter2_pseudo_delta_rms"])


def test_small_synthetic_svi_nuts_refinement_smoke(transition):
    """Interface smoke only: deliberately too short for scientific inference."""
    obj, lam = object_data()
    samples, chains, fitted, diagnostics, summary = (
        fit.run_iterated_mag_flux_linearized_inference(
            obj,
            lam,
            None,
            rng_key=jax.random.PRNGKey(7),
            fit_method="svi+nuts",
            num_warmup=2,
            num_samples=4,
            num_chains=1,
            chain_method="sequential",
            progress_bar=False,
            dense_mass=False,
            max_tree_depth=1,
            svi_steps=2,
            svi_lr=0.0003,
            disable_linear_trend=True,
            model_variant=VARIANT,
            band_poles_transition=transition,
            disk_order=1,
            erlang_order=1,
            refinement_strategy="svi_then_nuts",
            refinement_iters=2,
        )
    )
    assert diagnostics["flux_linearized_nuts_runs"] == 1
    assert np.all(np.isfinite(samples["eta_tau"]))
    assert np.all(np.isfinite(samples["log_tau_uv"]))
    assert np.isfinite(diagnostics["flux_linearized_iter2_pseudo_delta_rms"])


def test_hpc_flags_forward_new_variant_without_submission(monkeypatch):
    import sys
    from hpc_scripts.sfitlc import parse_args

    flags = [
        "--model_variant",
        VARIANT,
        "--disk_order",
        "2",
        "--eta_prior_profile",
        "default",
        "--band_poles_transition",
        "expm",
    ]
    monkeypatch.setattr(sys, "argv", ["sfitlc.py", "--fit", "stone", *flags])
    assert parse_args().extra_fit_flags == tuple(flags)


@pytest.mark.parametrize("order", [1, 3])
def test_model_structure_function_includes_disk_filter(order, transition):
    from qvc.light_curve.band_poles_fit import continuum_structure_function

    k = kernel(order=order)
    k = eqx.tree_at(lambda q: q.amp_blr, k, jnp.zeros(2))
    samples = {
        "tau_fast_driver": k.tau_fast,
        "tau_slow_uv_driver": k.tau_slow_uv,
        "tau_slow_band_g": np.atleast_1d(k.tau_slow_band[0]),
        "lag_disk_g": np.atleast_1d(k.lag_disk[0]),
        "amp_cont_g": np.atleast_1d(k.amp_cont[0]),
    }
    tau = np.geomspace(0.001, 3000.0, 40)
    points = (tau * 2, np.zeros(len(tau), int))
    cov = dense(k, points, (np.array([0.0]), np.array([0])))[:, 0]
    expected = np.sqrt(np.maximum(2 * (float(k.amp_cont[0]) ** 2 - cov), 0))
    got = continuum_structure_function(
        samples, "g", tau, z=1.0, disk_order=order, transition=transition
    )
    np.testing.assert_allclose(got, expected, rtol=2e-4, atol=3e-9)
    result = fit.compute_model_structure_function_equivalent(
        samples,
        "g",
        tau,
        z=1.0,
        return_series=True,
        disk_order=order,
        band_poles_transition=transition,
    )
    np.testing.assert_allclose(
        result["sf_model_curve_ref_band"], expected, rtol=2e-4, atol=3e-9
    )


@pytest.mark.parametrize(
    "relative_gap", [0.0, -1e-10, 1e-10, 0.999e-3, 1.001e-3, -0.999e-3, -1.002e-3]
)
def test_fallback_boundary_values_and_gradients(relative_gap, transition):
    theta = np.array(
        [
            np.log(2.0),
            np.log(2.0 / (1 - relative_gap)),
            0.0,
            np.log(6.0),
            np.log(6.0),
            0.0,
        ]
    )
    k = kernel(theta, transition=transition)
    A, P, H = independent_cascade_state(k)
    times = jnp.array([0.0, 1e-10, 1e-7, 0.1, 2.0, 100.0, 2000.0])
    np.testing.assert_allclose(
        k._transition_matrices(times),
        np.stack([expm(A * t) for t in times]),
        atol=3e-10,
        rtol=3e-8,
    )
    got = np.asarray(statistics_jacobian(theta, transition))
    expected = np.column_stack(
        [
            (
                independent_statistics(theta + e * 2e-6, independent_cascade_state)
                - independent_statistics(theta - e * 2e-6, independent_cascade_state)
            )
            / 4e-6
            for e in np.eye(6)
        ]
    )
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, atol=8e-6, rtol=5e-4)


def test_time_batch_skips_local_expm_until_rates_coincide(monkeypatch):
    from qvc.light_curve import multiband_model_shared_latent_band_poles_blr as module

    calls = []
    original = module.expm

    def watched(matrix, **kwargs):
        jax.debug.callback(
            lambda _: calls.append(matrix.shape[0]), matrix[0, 0], ordered=True
        )
        return original(matrix, **kwargs)

    monkeypatch.setattr(module, "expm", watched)

    @jax.jit
    def run(slow):
        k = SharedLatentBandPolesBLRQS(
            jnp.array([2.0]),
            jnp.atleast_1d(slow),
            jnp.atleast_1d(slow),
            jnp.array([6.0]),
            jnp.array([60.0]),
            jnp.array([0.1]),
            jnp.array([0.02]),
            transition="analytic",
        )
        return k._transition_matrices(jnp.array([0.0, 1.0, 100.0]))

    run(300.0).block_until_ready()
    jax.effects_barrier()
    assert calls == []
    run(2.0).block_until_ready()
    jax.effects_barrier()
    assert calls and set(calls) == {5}


@pytest.mark.parametrize(
    "requested,saved,expected",
    [
        (None, None, "analytic"),
        ("expm", None, "expm"),
        (None, {}, "expm"),
        (
            None,
            {"transition_implementation": "cascade_float64_expm_max_squarings_64"},
            "expm",
        ),
        (None, {"band_poles_transition": "analytic"}, "analytic"),
        ("analytic", {"band_poles_transition": "expm"}, "analytic"),
        ("expm", {"band_poles_transition": "analytic"}, "expm"),
    ],
)
def test_backend_defaults_and_resume_provenance(requested, saved, expected):
    from qvc.light_curve.multiband_model_shared_latent_band_poles_blr import (
        resolve_transition,
        run_transition_metadata,
        saved_transition,
    )

    backend = resolve_transition(requested, saved_metadata=saved)
    assert backend == expected
    meta = run_transition_metadata(backend, saved_metadata=saved)
    assert meta["band_poles_transition"] == (
        saved_transition(saved) if saved is not None else expected
    )
    assert meta["prediction_band_poles_transition"] == expected
    assert resolve_transition(model_variant="shared_latent_blr") is None
    with pytest.raises(ValueError, match="requires --model_variant"):
        resolve_transition("expm", model_variant="shared_latent_blr")
    with pytest.raises(ValueError, match="Unknown"):
        resolve_transition("invalid")


@pytest.mark.parametrize("flag", ["--band_poles_transition", "--band-poles-transition"])
def test_cli_backend_aliases_parse_without_running_fit(monkeypatch, flag, transition):
    import sys

    class Parsed(Exception):
        pass

    seen = []

    def stop(args):
        seen.append(args)
        raise Parsed

    monkeypatch.setattr(fit, "apply_resume_sample_save_policy", stop)
    monkeypatch.setattr(
        sys, "argv", ["fit_light_curves", "--model_variant", VARIANT, flag, transition]
    )
    with pytest.raises(Parsed):
        fit.main()
    assert seen[0].band_poles_transition == transition


def test_sample_metadata_round_trip_preserves_fit_backend(
    tmp_path, monkeypatch, transition
):
    from qvc.light_curve.multiband_model_shared_latent_band_poles_blr import (
        run_transition_metadata,
        resolve_transition,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(utils, "prefix", "backend")
    monkeypatch.setattr(utils, "suffix", "test")
    samples = {"tau_slow_uv_driver": np.ones(3), "log_tau_uv": np.ones(3)}
    meta = run_transition_metadata(transition)
    utils.save_obj_samples_to_hdf5(samples, "object", model_metadata=meta)
    loaded, saved = utils.load_obj_samples_from_hdf5("object", return_metadata=True)
    assert resolve_transition(saved_metadata=saved) == transition
    other = "expm" if transition == "analytic" else "analytic"
    replot = run_transition_metadata(other, saved_metadata=saved)
    assert replot["band_poles_transition"] == transition
    assert replot["prediction_band_poles_transition"] == other
    utils.save_all_samples_to_hdf5(samples, model_metadata=meta)
    with h5py.File("results/samples/backend/all_test.h5") as h:
        assert h.attrs["band_poles_transition"] == transition
