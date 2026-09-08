"""Opt-in truncated eta profile: support, inference, diagnostics and provenance."""

import sys
import warnings

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import truncnorm

from qvc.light_curve import fit_light_curves as fit
from qvc.light_curve import multiband_fit_utils as utils
from qvc.light_curve.band_poles_fit import build_single_object_model_band_poles
from test_multiband_model_shared_latent_band_poles_blr import object_data

jax.config.update("jax_enable_x64", True)
PROFILE = "modified_tight"


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


@pytest.mark.parametrize("variant", ["shared_latent_blr", "mag_flux_linearized_erlang",
                                     "shared_latent_band_poles_blr"])
def test_inference_and_kl_use_the_selected_truncated_profile(variant):
    obj, lam = object_data()
    shared = variant == "shared_latent_blr"
    kwargs = dict(eta_prior_profile=PROFILE, disable_linear_trend=True, erlang_order=1)
    if variant == "shared_latent_band_poles_blr":
        model = build_single_object_model_band_poles(obj, lam, None, disk_order=1, **kwargs)
    else:
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
        disable_linear_trend=True, eta_prior_profile=PROFILE)
    assert ("eta_tau_kl" in result) == (not shared)
    for name in names:
        x = draws[name]
        loc, low, high = (-.8, -1.5, 0.) if name == "eta_sigma" else (.5, 0., 1.5)
        prior = truncnorm((low-loc)/.25, (high-loc)/.25, loc=loc, scale=.25)
        variance = x.var(ddof=1) + 1e-12
        log_q = -.5 * np.log(2*np.pi*variance) - .5 * (x-x.mean())**2 / variance
        np.testing.assert_allclose(result[f"{name}_kl"], np.mean(log_q-prior.logpdf(x)), atol=2e-12)


@pytest.mark.parametrize("flag", ["--eta_prior_profile", "--eta-prior-profile"])
@pytest.mark.parametrize("variant", ["shared_latent_blr", "shared_latent_band_poles_blr"])
def test_cli_accepts_profile_without_running_fit(monkeypatch, flag, variant):
    class Parsed(Exception):
        pass

    def stop(args):
        assert args.eta_prior_profile == PROFILE
        assert args.model_variant == variant
        raise Parsed

    monkeypatch.setattr(fit, "apply_resume_sample_save_policy", stop)
    monkeypatch.setattr(sys, "argv", ["fit_light_curves", "--model_variant", variant, flag, PROFILE])
    with pytest.raises(Parsed):
        fit.main()


def test_profile_metadata_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(utils, "prefix", "tight-prior")
    monkeypatch.setattr(utils, "suffix", "test")
    metadata = {"eta_prior_profile": fit.resolve_eta_prior_profile(PROFILE,
        "shared_latent_band_poles_blr"), "band_poles_transition": "analytic"}
    samples = {"tau_slow_uv_driver": np.ones(3), "log_tau_uv": np.ones(3),
               "eta_sigma": np.full(3, -.8), "eta_tau": np.full(3, .5)}
    utils.save_obj_samples_to_hdf5(samples, "object", model_metadata=metadata)
    loaded, saved = utils.load_obj_samples_from_hdf5("object", return_metadata=True)
    assert saved["eta_prior_profile"] == PROFILE
    assert fit.resolve_eta_prior_profile(saved["eta_prior_profile"]) == PROFILE
    for key in samples:
        np.testing.assert_array_equal(loaded[key], samples[key])
    utils.save_all_samples_to_hdf5(samples, model_metadata=metadata)
    with h5py.File("results/samples/tight-prior/all_test.h5") as h:
        assert h.attrs["eta_prior_profile"] == PROFILE
