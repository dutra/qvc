import jax
import jax.numpy as jnp
import numpy as np

from qvc.light_curve.fraction_marginalization import (
    empirical_logmeanexp,
    fit_logit_normal,
    responsibility_resample_fractions,
    scale_prediction_samples_by_fraction,
    scale_variable_relflux_amplitudes,
    select_fraction_draws_for_bands,
    sigmoid_fraction_draws,
)


def test_empirical_logmeanexp_matches_manual_mixture_and_has_finite_gradient():
    values = jnp.array([-3.0, -1.0, -2.5])
    expected = np.log(np.mean(np.exp(np.asarray(values))))

    actual = empirical_logmeanexp(values)
    gradient = jax.grad(lambda x: empirical_logmeanexp(x))(values)

    np.testing.assert_allclose(float(actual), expected, rtol=1e-6)
    assert np.all(np.isfinite(np.asarray(gradient)))
    np.testing.assert_allclose(np.sum(np.asarray(gradient)), 1.0, rtol=1e-6)


def test_scaling_changes_stochastic_amplitudes_but_not_observed_terms_or_trend():
    params = {
        "amp_cont_relflux": jnp.array([1.0, 2.0]),
        "amp_bc_relflux": jnp.array([0.1, 0.2]),
        "amp_blr_relflux": jnp.array([0.3, 0.4]),
        "amp_blr2_relflux": jnp.array([0.5, 0.6]),
        "linear_trend": jnp.asarray(0.4),
        "linear_trend_band_offset": jnp.array([0.1, -0.1]),
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
    for key in (
        "linear_trend",
        "linear_trend_band_offset",
        "log_jitter",
        "survey_delta_mag",
    ):
        np.testing.assert_allclose(scaled[key], params[key])


def test_logit_normal_preserves_band_dimension_and_handles_singular_draws():
    draws = np.repeat([[0.2, 0.4, 0.8]], 8, axis=0)

    mean, covariance, scale_tril = fit_logit_normal(draws)
    fractions = sigmoid_fraction_draws(
        np.random.default_rng(2).normal(size=(20, 3)),
        mean,
        scale_tril,
    )

    assert covariance.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(scale_tril @ scale_tril.T) > 0.0)
    assert fractions.shape == (20, 3)
    assert np.all((fractions > 0.0) & (fractions < 1.0))


def test_fraction_selection_reorders_bands_without_breaking_joint_rows():
    source = {
        "psf_agn_fraction_bands": ("u", "g", "r", "i", "z"),
        "psf_agn_fraction_draws": np.array(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.6, 0.7, 0.8, 0.9, 1.0],
            ]
        ),
        "psf_agn_fraction_valid_count": 2,
    }

    selected = select_fraction_draws_for_bands(source, ("r", "g", "z"))

    np.testing.assert_allclose(
        selected,
        [[0.3, 0.2, 0.5], [0.8, 0.7, 1.0]],
    )


def test_responsibility_resampling_and_prediction_scaling_preserve_joint_vectors():
    draws = np.array([[0.2, 0.8], [0.7, 0.3]])
    responsibilities = np.array([[1.0, 0.0], [0.0, 1.0]])

    selected = responsibility_resample_fractions(
        draws,
        responsibilities,
        seed=9,
    )
    samples = {
        "psf_agn_fraction": selected,
        "amp_cont_relflux": np.ones((2, 2)),
        "linear_trend": np.array([0.4, 0.2]),
        "log_sigma_uv": np.array([-2.0, -2.0]),
    }
    prediction = scale_prediction_samples_by_fraction(samples)

    np.testing.assert_allclose(selected, draws)
    np.testing.assert_allclose(prediction["amp_cont_relflux"], draws)
    np.testing.assert_allclose(prediction["linear_trend"], samples["linear_trend"])
    np.testing.assert_allclose(prediction["log_sigma_uv"], samples["log_sigma_uv"])
