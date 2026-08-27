import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qvc.light_curve.fraction_marginalization import (
    empirical_logmeanexp,
    fit_logit_normal,
    responsibility_resample_fractions,
    scale_prediction_samples_by_fraction,
    scale_variable_relflux_amplitudes,
    sigmoid_fraction_draws,
)


def test_empirical_logmeanexp_matches_manual_mixture_and_has_finite_gradient():
    values = jnp.array([-3.0, -1.0, -2.5])
    expected = np.log(np.mean(np.exp(np.asarray(values))))
    np.testing.assert_allclose(float(empirical_logmeanexp(values)), expected, rtol=1e-6)
    gradient = jax.grad(lambda x: empirical_logmeanexp(x))(values)
    assert np.all(np.isfinite(np.asarray(gradient)))
    np.testing.assert_allclose(np.sum(np.asarray(gradient)), 1.0, rtol=1e-6)


def test_scaling_changes_only_variable_relative_flux_amplitudes():
    params = {
        "amp_cont_relflux": jnp.array([1.0, 2.0]),
        "amp_bc_relflux": jnp.array([0.1, 0.2]),
        "amp_blr_relflux": jnp.array([0.3, 0.4]),
        "amp_blr2_relflux": jnp.array([0.5, 0.6]),
        "log_jitter": jnp.array([-3.0, -4.0]),
        "survey_delta_mag": jnp.array([0.01, -0.02]),
    }
    fractions = jnp.array([0.25, 0.75])
    scaled = scale_variable_relflux_amplitudes(params, fractions)

    for key in (
        "amp_cont_relflux",
        "amp_bc_relflux",
        "amp_blr_relflux",
        "amp_blr2_relflux",
    ):
        np.testing.assert_allclose(scaled[key], params[key] * fractions)
    np.testing.assert_allclose(scaled["log_jitter"], params["log_jitter"])
    np.testing.assert_allclose(scaled["survey_delta_mag"], params["survey_delta_mag"])


def test_scaling_dilutes_complete_bandwise_linear_trend():
    params = {
        "linear_trend": jnp.asarray(0.4),
        "linear_trend_band_offset": jnp.array([0.1, -0.1]),
        "mean": jnp.array([0.02, -0.03]),
    }
    fractions = jnp.array([0.25, 0.75])

    scaled = scale_variable_relflux_amplitudes(params, fractions)

    np.testing.assert_allclose(scaled["linear_trend_band"], [0.125, 0.225])
    np.testing.assert_allclose(scaled["linear_trend"], params["linear_trend"])
    np.testing.assert_allclose(
        scaled["linear_trend_band_offset"], params["linear_trend_band_offset"]
    )
    np.testing.assert_allclose(scaled["mean"], params["mean"])


def test_scaling_dilutes_shared_erlang_trend_by_band():
    scaled = scale_variable_relflux_amplitudes(
        {"linear_trend": jnp.asarray(0.4)},
        jnp.array([0.25, 0.75]),
    )

    np.testing.assert_allclose(scaled["linear_trend_band"], [0.1, 0.3])


def test_prediction_scaling_dilutes_batched_linear_trend():
    samples = {
        "psf_agn_fraction": np.array([[0.25, 0.75], [0.5, 0.8]]),
        "linear_trend": np.array([0.4, 0.2]),
        "linear_trend_band_offset": np.array([[0.1, -0.1], [0.0, 0.1]]),
    }

    prediction = scale_prediction_samples_by_fraction(samples)

    np.testing.assert_allclose(
        prediction["linear_trend_band"],
        [[0.125, 0.225], [0.1, 0.24]],
    )


def test_identical_empirical_draws_equal_fixed_fraction_likelihood():
    fractions = jnp.array([0.4, 0.7])
    params = {"amp_cont_relflux": jnp.array([2.0, 3.0])}

    def fixed_loglike(fraction):
        scaled = scale_variable_relflux_amplitudes(params, fraction)
        return -jnp.sum(scaled["amp_cont_relflux"] ** 2)

    component = fixed_loglike(fractions)
    mixture = empirical_logmeanexp(
        jax.vmap(fixed_loglike)(jnp.repeat(fractions[None, :], 64, axis=0))
    )
    np.testing.assert_allclose(mixture, component, rtol=1e-6)


def test_logit_normal_matches_empirical_moments_and_handles_singular_draws():
    draws = np.array(
        [
            [0.2, 0.4, 0.7],
            [0.25, 0.5, 0.75],
            [0.3, 0.6, 0.8],
            [0.35, 0.7, 0.85],
        ]
    )
    mean, covariance, scale_tril = fit_logit_normal(draws)
    logits = np.log(draws) - np.log1p(-draws)
    np.testing.assert_allclose(mean, logits.mean(axis=0))
    np.testing.assert_allclose(covariance, np.cov(logits, rowvar=False), atol=1e-7)
    assert np.all(np.linalg.eigvalsh(scale_tril @ scale_tril.T) > 0.0)

    singular = np.repeat([[0.2, 0.4, 0.8]], 8, axis=0)
    _, _, singular_scale = fit_logit_normal(singular)
    assert np.all(np.linalg.eigvalsh(singular_scale @ singular_scale.T) > 0.0)
    fractions = sigmoid_fraction_draws(
        np.random.default_rng(2).normal(size=(20, 3)),
        *fit_logit_normal(singular)[::2],
    )
    assert np.all((fractions > 0.0) & (fractions < 1.0))


def test_responsibility_resampling_preserves_joint_vectors_for_prediction():
    draws = np.array([[0.2, 0.8], [0.7, 0.3]])
    responsibilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    selected = responsibility_resample_fractions(
        draws, responsibilities, seed=9
    )
    np.testing.assert_allclose(selected, draws)

    samples = {
        "psf_agn_fraction": selected,
        "amp_cont_relflux": np.ones((2, 2)),
        "log_sigma_uv": np.array([-2.0, -2.0]),
    }
    prediction = scale_prediction_samples_by_fraction(samples)
    np.testing.assert_allclose(prediction["amp_cont_relflux"], draws)
    np.testing.assert_allclose(prediction["log_sigma_uv"], samples["log_sigma_uv"])


def test_simulated_diluted_lc_recovers_intrinsic_amplitude_and_fraction_uncertainty():
    rng = np.random.default_rng(12)
    x = np.linspace(-1.0, 1.0, 80)
    intrinsic_amplitude = 0.35
    fractions = np.clip(rng.normal(0.58, 0.045, size=64), 0.4, 0.75)
    observed = intrinsic_amplitude * np.median(fractions) * x
    noise = 0.012
    amplitude_grid = np.linspace(0.15, 0.60, 901)

    def posterior_summary(fraction_samples):
        slopes = amplitude_grid[:, None] * fraction_samples[None, :]
        component_chi2 = (
            np.sum(observed**2)
            - 2.0 * slopes * np.sum(observed * x)
            + slopes**2 * np.sum(x**2)
        ) / noise**2
        component_loglikes = -0.5 * component_chi2
        max_component = component_loglikes.max(axis=1, keepdims=True)
        loglike = (
            max_component[:, 0]
            + np.log(np.mean(np.exp(component_loglikes - max_component), axis=1))
        )
        posterior = np.exp(loglike - np.max(loglike))
        posterior /= posterior.sum()
        cdf = np.cumsum(posterior)
        q16, q50, q84 = np.interp((0.16, 0.5, 0.84), cdf, amplitude_grid)
        log_sigma_error = 0.5 * (np.log10(q84) - np.log10(q16))
        return q50, 0.5 * (q84 - q16), log_sigma_error

    empirical_median, empirical_error, empirical_log_sigma_error = posterior_summary(
        fractions
    )
    logit_mean, _, logit_scale = fit_logit_normal(fractions[:, None])
    latent_fractions = sigmoid_fraction_draws(
        rng.normal(size=(4096, 1)), logit_mean, logit_scale
    )[:, 0]
    latent_median, latent_error, latent_log_sigma_error = posterior_summary(
        latent_fractions
    )
    fixed_median, fixed_error, fixed_log_sigma_error = posterior_summary(
        np.array([np.median(fractions)])
    )

    assert empirical_median == pytest.approx(intrinsic_amplitude, abs=0.025)
    assert latent_median == pytest.approx(intrinsic_amplitude, abs=0.025)
    assert empirical_error > fixed_error
    assert latent_error > fixed_error
    assert empirical_log_sigma_error > fixed_log_sigma_error
    assert latent_log_sigma_error > fixed_log_sigma_error
