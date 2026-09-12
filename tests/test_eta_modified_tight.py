"""Default truncated eta profile: support, inference, diagnostics and provenance."""

import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import truncnorm

from qvc.light_curve import fit_light_curves as fit
from qvc.light_curve import multiband_fit_utils as utils

jax.config.update("jax_enable_x64", True)
PROFILE = "default"

def object_data():
    X = (jnp.array([0., 1., 3., 5., 8., 13.]), jnp.array([0, 1, 0, 1, 0, 1]))
    Y = jnp.array([.01, -.02, .03, .02, -.01, .01])
    D = jnp.full(6, .001)
    bands = ("g", "i")
    obj = dict(
        object_id="prior-test",
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


@pytest.mark.parametrize("name,loc,low,high", [
    ("eta_sigma", -.8, -1.5, 0.), ("eta_tau", .5, 0., 1.5),
])
def test_distribution_matches_normalized_scipy_and_support(name, loc, low, high):
    prior = getattr(fit, f"{name}_prior")(PROFILE)
    assert type(prior).__name__.endswith("TruncatedDistribution")
    assert float(prior.base_dist.loc) == pytest.approx(loc)
    assert float(prior.base_dist.scale) == pytest.approx(.25)
    assert float(prior.low) == low
    assert float(prior.high) == high
    reference = truncnorm((low-loc)/.25, (high-loc)/.25, loc=loc, scale=.25)
    x = np.linspace(low, high, 51)
    np.testing.assert_allclose(prior.log_prob(x), reference.logpdf(x), atol=2e-13)
    # Independent quadrature checks the normalization rather than just shape.
    nodes, weights = np.polynomial.legendre.leggauss(64)
    grid = (high + low) / 2 + (high - low) * nodes / 2
    integral = np.sum(np.exp(prior.log_prob(grid)) * weights) * (high - low) / 2
    np.testing.assert_allclose(integral, 1., atol=2e-13)
    outside = jnp.array([low-.01, high+.01])
    assert not np.asarray(prior.support(outside)).any()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert np.isneginf(np.asarray(prior.log_prob(outside))).all()
    samples = prior.sample(jax.random.PRNGKey(13), (2048,))
    assert np.asarray(prior.support(samples)).all()
    assert np.isfinite(jax.grad(prior.log_prob)(jnp.asarray(loc)))


@pytest.mark.parametrize("variant", ["shared_latent_blr", "mag_flux_linearized_erlang"])
@pytest.mark.parametrize("explicit", [False, True])
def test_inference_and_kl_use_the_selected_truncated_profile(variant, explicit):
    obj, lam = object_data()
    shared = variant == "shared_latent_blr"
    profile_kwargs = {"eta_prior_profile": PROFILE} if explicit else {}
    kwargs = dict(disable_linear_trend=True, erlang_order=1, **profile_kwargs)
    model = fit.build_single_object_model_mag_flux_linearized(
        obj, lam, None, shared_latent=shared, disk_order=1, **kwargs)
    sites = fit.trace(fit.seed(model, jax.random.PRNGKey(4))).get_trace()
    names = ["eta_sigma"] if shared else ["eta_sigma", "eta_tau"]
    assert ("eta_tau" in sites) == (not shared)
    for name in names:
        prior = sites[name]["fn"]
        expected = getattr(fit, f"{name}_prior")(PROFILE)
        assert float(prior.base_dist.scale) == .25
        np.testing.assert_allclose(prior.log_prob(sites[name]["value"]),
                                   expected.log_prob(sites[name]["value"]))
    raw = {k: np.asarray(v["value"]) for k, v in sites.items()
           if v["type"] in {"sample", "deterministic"} and not v.get("is_observed", False)}
    draws = {k: np.stack([v] * 32) for k, v in raw.items()}
    draws["eta_sigma"] = np.linspace(-1., -.6, 32)
    if not shared:
        draws["eta_tau"] = np.linspace(.2, .8, 32)
    explicit = fit.add_model_prediction_params(draws, lam, model_variant=variant, disk_order=1)
    flat = utils.flatten_flat_samples_per_band(explicit, obj["bands"], obj["survey_names"])
    result = fit.compute_parameter_kls(flat, bands=obj["bands"], survey_names=obj["survey_names"],
        t_ref=obj["X"][0], z=obj["z"], lambda_center_rf=float(raw["lambda_center_rf"]),
        log_jitter_mean=np.full((2, 3), -4.), model_variant=variant,
        disable_linear_trend=True, **profile_kwargs)
    assert ("eta_tau_kl" in result) == (not shared)
    for name in names:
        x = draws[name]
        loc, low, high = (-.8, -1.5, 0.) if name == "eta_sigma" else (.5, 0., 1.5)
        prior = truncnorm((low-loc)/.25, (high-loc)/.25, loc=loc, scale=.25)
        variance = x.var(ddof=1) + 1e-12
        log_q = -.5 * np.log(2*np.pi*variance) - .5 * (x-x.mean())**2 / variance
        np.testing.assert_allclose(result[f"{name}_kl"], np.mean(log_q-prior.logpdf(x)), atol=2e-12)


@pytest.mark.parametrize("flag", ["--eta_prior_profile", "--eta-prior-profile"])
@pytest.mark.parametrize("variant", ["shared_latent_blr", "mag_flux_linearized_erlang"])
@pytest.mark.parametrize("profile", [None, "default", "relaxed", "modified"])
def test_cli_accepts_profile_without_running_fit(monkeypatch, flag, variant, profile):
    class Parsed(Exception):
        pass

    def stop(args):
        assert args.eta_prior_profile == (profile or PROFILE)
        assert args.model_variant == variant
        raise Parsed

    monkeypatch.setattr(fit, "apply_resume_sample_save_policy", stop)
    monkeypatch.setattr(sys, "argv", ["fit_light_curves", "--model_variant", variant] + ([flag, profile] if profile else []))
    with pytest.raises(Parsed):
        fit.main()


@pytest.mark.parametrize("name", ["eta_sigma", "eta_tau"])
def test_default_helpers_match_tight_and_reject_retired_name(name):
    helper = getattr(fit, f"{name}_prior")
    default, explicit = helper(), helper(PROFILE)
    grid = jnp.linspace(float(explicit.low), float(explicit.high), 51)
    np.testing.assert_array_equal(default.log_prob(grid), explicit.log_prob(grid))
    with pytest.raises(ValueError, match="eta_prior_profile must be one of"):
        helper("modified_tight")


@pytest.mark.parametrize("flag", ["--eta_prior_profile", "--eta-prior-profile"])
def test_cli_rejects_retired_modified_tight_name(monkeypatch, capsys, flag):
    monkeypatch.setattr(sys, "argv", ["fit_light_curves", flag, "modified_tight"])
    with pytest.raises(SystemExit) as exc:
        fit.main()
    assert exc.value.code == 2
    assert "invalid choice: 'modified_tight'" in capsys.readouterr().err
