import math

import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

from qvc.light_curve.dho_drw_parameterization import (
    IntegratedTimescaleDHOBaseQS,
    LOG_QUALITY_FACTOR_PRIOR_SIGMA,
    MAX_QUALITY_FACTOR,
    MIN_QUALITY_FACTOR,
    dho_log_timescales_from_drw,
    dho_timescales_from_drw,
    log_perturbation_ratio_prior,
    log_quality_factor_prior,
)
from qvc.light_curve.multiband_model_dho_blr_erlang_drw import (
    make_multiband_dho_blr_flux_linearized_erlang_drw_model,
)


def test_drw_coordinates_preserve_integrated_correlation_time_and_ratio():
    tau_drw = jnp.asarray([10.0, 100.0])
    rho = jnp.asarray([0.01, 0.4])
    tau_fast, tau_slow = dho_timescales_from_drw(tau_drw, rho)

    assert np.allclose(np.asarray(tau_fast + tau_slow), np.asarray(tau_drw))
    assert np.allclose(np.asarray(tau_fast / tau_slow), np.asarray(rho))
    assert np.all(np.asarray(tau_fast) < np.asarray(tau_slow))


def test_log_drw_coordinates_match_linear_mapping():
    tau_drw = 300.0
    rho = 0.2
    logit_rho = np.log(rho) - np.log1p(-rho)
    log_fast, log_slow, mapped_rho = dho_log_timescales_from_drw(
        jnp.log(tau_drw), logit_rho
    )
    fast, slow = dho_timescales_from_drw(tau_drw, rho)

    assert np.isclose(float(jnp.exp(log_fast)), float(fast))
    assert np.isclose(float(jnp.exp(log_slow)), float(slow))
    assert np.isclose(float(mapped_rho), rho)


def test_all_regime_carma21_has_unit_variance_and_requested_integral_time():
    tau_drw = 80.0
    for quality_factor in (0.2, 0.5, 3.0):
        for tau_perturb in (0.2, 4.0, 30.0):
            kernel = IntegratedTimescaleDHOBaseQS.from_drw(
                tau_drw=jnp.asarray([tau_drw]),
                quality_factor=jnp.asarray([quality_factor]),
                tau_perturb=jnp.asarray([tau_perturb]),
            )
            covariance = np.asarray(kernel.stationary_covariance())
            h = np.asarray(kernel.observation_model((0.0, 0)))
            assert np.isclose(h @ covariance @ h, 1.0, rtol=2e-5)

            integrated_transition = -np.linalg.inv(
                np.asarray(kernel.design_matrix())
            )
            integrated_corr = h @ integrated_transition @ covariance @ h
            assert np.isclose(integrated_corr, tau_drw, rtol=2e-5)


def test_underdamped_dho_can_produce_qpo_like_negative_correlation():
    kernel = IntegratedTimescaleDHOBaseQS.from_drw(
        tau_drw=jnp.asarray([30.0]),
        quality_factor=jnp.asarray([4.0]),
        tau_perturb=jnp.asarray([0.6]),
    )
    covariance = np.asarray(kernel.stationary_covariance())
    h = np.asarray(kernel.observation_model((0.0, 0)))
    grid = np.linspace(0.0, 1000.0, 500)
    corr = np.array(
        [
            h
            @ np.asarray(kernel.transition_matrix((0.0, 0), (float(t), 0)))
            @ covariance
            @ h
            for t in grid
        ]
    )
    assert np.min(corr) < -0.1


def test_underdamped_erlang_gp_has_finite_likelihood():
    times = jnp.array([0.0, 7.0, 18.0, 35.0])
    band = jnp.zeros(times.shape, dtype=jnp.int32)
    model = make_multiband_dho_blr_flux_linearized_erlang_drw_model(
        (times, band),
        jnp.zeros(times.shape),
        jnp.full(times.shape, 0.02),
        n_band=1,
        survey_idx=jnp.zeros(times.shape, dtype=jnp.int32),
        erlang_order=2,
    )
    params = {
        "tau_drw_band": jnp.array([100.0]),
        "tau_perturb_band": jnp.array([2.0]),
        "quality_factor": jnp.asarray(3.0),
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

    kernel = model._build_kernel(params)
    transition = np.asarray(kernel.transition_matrix((0.0, 0), (7.0, 0)))
    dense_oracle = expm(np.asarray(kernel.design_matrix()) * 7.0)
    assert np.allclose(transition, dense_oracle, rtol=2e-5, atol=2e-7)


def test_erlang_line_amplitude_is_stationary_response_rms_at_any_lag():
    times = jnp.array([0.0, 10.0])
    band = jnp.zeros(times.shape, dtype=jnp.int32)
    model = make_multiband_dho_blr_flux_linearized_erlang_drw_model(
        (times, band),
        jnp.zeros(times.shape),
        jnp.full(times.shape, 0.02),
        n_band=1,
        survey_idx=jnp.zeros(times.shape, dtype=jnp.int32),
        erlang_order=3,
    )
    target_rms = 0.07
    base_params = {
        "tau_drw_band": jnp.array([100.0]),
        "tau_perturb_band": jnp.array([2.0]),
        "quality_factor": jnp.asarray(0.2),
        "amp_cont_relflux": jnp.array([0.0]),
        "amp_blr_relflux": jnp.array([target_rms]),
    }

    for lag in (5.0, 5000.0):
        params = {**base_params, "lag_blr": jnp.array([lag])}
        kernel = model._build_kernel(params)
        variance = kernel.evaluate((0.0, 0), (0.0, 0))
        assert np.isclose(float(variance), target_rms**2, rtol=2e-6)


def test_positive_flux_guard_is_diagnostic_by_default_and_opt_in_penalty():
    times = jnp.array([0.0, 7.0, 18.0, 35.0])
    band = jnp.zeros(times.shape, dtype=jnp.int32)
    model = make_multiband_dho_blr_flux_linearized_erlang_drw_model(
        (times, band),
        jnp.zeros(times.shape),
        jnp.full(times.shape, 0.02),
        n_band=1,
        survey_idx=jnp.zeros(times.shape, dtype=jnp.int32),
        erlang_order=2,
    )
    guarded_model = make_multiband_dho_blr_flux_linearized_erlang_drw_model(
        (times, band),
        jnp.zeros(times.shape),
        jnp.full(times.shape, 0.02),
        n_band=1,
        survey_idx=jnp.zeros(times.shape, dtype=jnp.int32),
        erlang_order=2,
        enforce_positive_flux_guard=True,
    )
    params = {
        "tau_drw_band": jnp.array([100.0]),
        "tau_perturb_band": jnp.array([2.0]),
        "quality_factor": jnp.asarray(0.6),
        "lag_blr": jnp.array([20.0]),
        "amp_cont_relflux": jnp.array([0.05]),
        "amp_blr_relflux": jnp.array([0.01]),
        "log_jitter": jnp.full((1, 3), -10.0),
        "survey_delta_mag": jnp.zeros((1, 3)),
        "mean": jnp.zeros(1),
        "linear_trend": jnp.asarray(0.0),
        "linear_trend_band_offset": jnp.zeros(1),
    }

    safe_penalty = float(model.positive_flux_log_penalty(params))
    pathological = dict(params)
    pathological["amp_cont_relflux"] = jnp.array([1.0])
    pathological["amp_blr_relflux"] = jnp.array([1.0])
    pathological_penalty = float(model.positive_flux_log_penalty(pathological))

    assert np.isfinite(safe_penalty)
    assert safe_penalty > -1e-8
    assert np.isfinite(pathological_penalty)
    assert pathological_penalty < -100.0
    unguarded_log_prob = float(model.log_prob(pathological))
    guarded_log_prob = float(guarded_model.log_prob(pathological))
    assert np.isclose(
        guarded_log_prob,
        unguarded_log_prob + pathological_penalty,
        rtol=1e-6,
    )

    safe_negative_probability = np.asarray(
        model.negative_total_flux_probability(params)
    )
    pathological_negative_probability = np.asarray(
        model.negative_total_flux_probability(pathological)
    )
    assert np.all(safe_negative_probability < 1e-8)
    assert np.all(pathological_negative_probability > 0.1)


def test_quality_factor_prior_is_legacy_centered_with_small_qpo_tail():
    prior = log_quality_factor_prior()

    assert np.isclose(float(jnp.exp(prior.support.lower_bound)), MIN_QUALITY_FACTOR)
    assert np.isclose(float(jnp.exp(prior.support.upper_bound)), MAX_QUALITY_FACTOR)
    assert jnp.isclose(prior.base_dist.loc, jnp.log(0.1))
    assert jnp.isclose(prior.base_dist.scale, LOG_QUALITY_FACTOR_PRIOR_SIGMA)

    def normal_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    loc = math.log(0.1)
    scale = 1.0
    z_low = (math.log(MIN_QUALITY_FACTOR) - loc) / scale
    z_high = (math.log(MAX_QUALITY_FACTOR) - loc) / scale
    z_critical = (math.log(0.5) - loc) / scale
    qpo_mass = (
        normal_cdf(z_high) - normal_cdf(z_critical)
    ) / (
        normal_cdf(z_high) - normal_cdf(z_low)
    )
    assert 0.0 < qpo_mass < 0.1


def test_carma21_perturbation_ratio_prior_is_positive_and_bounded():
    prior = log_perturbation_ratio_prior()

    assert float(jnp.exp(prior.support.lower_bound)) > 0.0
    assert float(jnp.exp(prior.support.upper_bound)) <= 0.5
    assert jnp.isclose(prior.base_dist.loc, jnp.log(0.02))
