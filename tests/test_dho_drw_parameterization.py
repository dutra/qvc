import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm
from tinygp import GaussianProcess

import qvc.light_curve.fast_quasisep as fast_quasisep
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
    ErlangResponseIntegratedDHOQS,
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


def _integrated_erlang_kernel_from_log_params(log_params, *, order=3):
    tau_drw = jnp.exp(log_params[0]) * jnp.asarray([0.8, 1.2])
    quality_factor = jnp.exp(log_params[1])
    tau_perturb = jnp.exp(log_params[2]) * tau_drw
    lag_blr = jnp.exp(log_params[3]) * jnp.asarray([0.9, 1.1])
    base = IntegratedTimescaleDHOBaseQS.from_drw(
        tau_drw,
        quality_factor,
        tau_perturb,
    )
    return ErlangResponseIntegratedDHOQS(
        tau_fast=jnp.full_like(tau_drw, 0.5),
        tau_slow=jnp.full_like(tau_drw, 0.5),
        lag_blr=lag_blr,
        amp_cont=jnp.asarray([0.08, 0.1]),
        amp_blr=jnp.asarray([0.02, 0.03]),
        order=order,
        carma_omega0=base.omega0,
        carma_damping=base.damping,
        carma_obs_position=base.obs_position,
        carma_obs_velocity=base.obs_velocity,
    )


def _reference_integrated_erlang_transitions(kernel, gaps):
    design = kernel.design_matrix()
    bands, _n_driver, n_state = kernel._dimensions()
    indices = kernel._band_state_indices()
    blocks = design[indices[:, :, None], indices[:, None, :]]

    def one(gap):
        transition_blocks = jax.vmap(
            lambda block: jax.scipy.linalg.expm(block * gap)
        )(blocks)
        transition = jnp.zeros((n_state, n_state), dtype=design.dtype)
        band = jnp.arange(bands)
        return transition.at[
            indices[band][:, :, None],
            indices[band][:, None, :],
        ].set(transition_blocks)

    return jax.vmap(one)(gaps)


def test_integrated_erlang_structured_transitions_match_expm_and_gradients():
    gaps = jnp.asarray([0.0, 1e-10, 0.03, 3.0, 100.0, 1000.0])
    weights = jnp.sin(jnp.arange(gaps.size * 100, dtype=float)).reshape(
        gaps.size,
        10,
        10,
    )

    def structured_objective(log_params):
        kernel = _integrated_erlang_kernel_from_log_params(log_params)
        transitions = kernel.transition_matrices_from_dt(gaps)
        return jnp.sum(transitions * weights)

    def reference_objective(log_params):
        kernel = _integrated_erlang_kernel_from_log_params(log_params)
        transitions = _reference_integrated_erlang_transitions(kernel, gaps)
        return jnp.sum(transitions * weights)

    structured_value_and_grad = jax.jit(
        jax.value_and_grad(structured_objective)
    )
    reference_value_and_grad = jax.jit(
        jax.value_and_grad(reference_objective)
    )
    for quality_factor in (
        0.05,
        0.49,
        0.5 * (1.0 - 1e-8),
        0.5,
        0.5 * (1.0 + 1e-8),
        0.51,
        3.0,
    ):
        log_params = jnp.log(
            jnp.asarray([100.0, quality_factor, 0.02, 70.0])
        )
        structured_value, structured_gradient = structured_value_and_grad(
            log_params
        )
        reference_value, reference_gradient = reference_value_and_grad(log_params)

        np.testing.assert_allclose(
            np.asarray(structured_value),
            np.asarray(reference_value),
            rtol=2e-10,
            atol=2e-11,
        )
        np.testing.assert_allclose(
            np.asarray(structured_gradient),
            np.asarray(reference_gradient),
            rtol=2e-8,
            atol=2e-9,
        )


def test_integrated_erlang_transitions_match_expm_at_response_pole_collisions():
    order = 3
    tau_drw = jnp.asarray([80.0, 120.0])
    base = IntegratedTimescaleDHOBaseQS.from_drw(
        tau_drw,
        jnp.asarray(0.2),
        0.03 * tau_drw,
    )
    half_damping = 0.5 * base.damping
    modal_offset = jnp.sqrt(half_damping**2 - base.omega0**2)
    collision_rates = jnp.asarray(
        [
            half_damping[0] - modal_offset[0],
            half_damping[1] + modal_offset[1],
        ]
    )
    gaps = jnp.asarray([0.0, 1e-10, 0.03, 3.0, 100.0, 1000.0])
    weights = jnp.cos(jnp.arange(gaps.size * 100, dtype=float)).reshape(
        gaps.size,
        10,
        10,
    )

    def kernel_from_rate_offsets(log_rate_offsets):
        response_rates = collision_rates * jnp.exp(log_rate_offsets)
        return ErlangResponseIntegratedDHOQS(
            tau_fast=jnp.full_like(tau_drw, 0.5),
            tau_slow=jnp.full_like(tau_drw, 0.5),
            lag_blr=order / response_rates,
            amp_cont=jnp.asarray([0.08, 0.1]),
            amp_blr=jnp.asarray([0.02, 0.03]),
            order=order,
            carma_omega0=base.omega0,
            carma_damping=base.damping,
            carma_obs_position=base.obs_position,
            carma_obs_velocity=base.obs_velocity,
        )

    def structured_objective(log_rate_offsets):
        transitions = kernel_from_rate_offsets(
            log_rate_offsets
        ).transition_matrices_from_dt(gaps)
        return jnp.sum(transitions * weights)

    def reference_objective(log_rate_offsets):
        transitions = _reference_integrated_erlang_transitions(
            kernel_from_rate_offsets(log_rate_offsets),
            gaps,
        )
        return jnp.sum(transitions * weights)

    log_rate_offsets = jnp.zeros(2)
    structured_value, structured_gradient = jax.jit(
        jax.value_and_grad(structured_objective)
    )(log_rate_offsets)
    reference_value, reference_gradient = jax.jit(
        jax.value_and_grad(reference_objective)
    )(log_rate_offsets)
    np.testing.assert_allclose(
        np.asarray(structured_value),
        np.asarray(reference_value),
        rtol=2e-10,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        np.asarray(structured_gradient),
        np.asarray(reference_gradient),
        rtol=2e-8,
        atol=2e-9,
    )


def test_integrated_erlang_block_likelihood_matches_tinygp_and_gradients():
    epoch_times = jnp.asarray([0.0, 3.0, 20.0, 100.0])
    times = jnp.repeat(epoch_times, 2)
    bands = jnp.tile(jnp.arange(2, dtype=jnp.int32), epoch_times.size)
    coordinates = (times, bands)
    diagonal = jnp.linspace(0.01, 0.03, times.size) ** 2
    residual = 0.03 * jnp.sin(times / 7.0 + bands)

    def reference_objective(log_params):
        kernel = _integrated_erlang_kernel_from_log_params(log_params)
        return GaussianProcess(
            kernel,
            coordinates,
            diag=diagonal,
            assume_sorted=True,
        ).log_probability(residual)

    def block_objective(log_params):
        kernel = _integrated_erlang_kernel_from_log_params(log_params)
        return fast_quasisep.block_diagonal_log_probability(
            kernel,
            coordinates,
            diagonal,
            residual,
            sort_time=times,
        )

    block_value_and_grad = jax.jit(jax.value_and_grad(block_objective))
    reference_value_and_grad = jax.jit(
        jax.value_and_grad(reference_objective)
    )
    for quality_factor in (0.2, 0.5, 2.0):
        log_params = jnp.log(
            jnp.asarray([100.0, quality_factor, 0.02, 70.0])
        )
        block_value, block_gradient = block_value_and_grad(log_params)
        reference_value, reference_gradient = reference_value_and_grad(log_params)
        np.testing.assert_allclose(
            np.asarray(block_value),
            np.asarray(reference_value),
            rtol=2e-10,
            atol=2e-11,
        )
        np.testing.assert_allclose(
            np.asarray(block_gradient),
            np.asarray(reference_gradient),
            rtol=2e-8,
            atol=2e-9,
        )


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


def test_positive_flux_guard_penalizes_pathological_total_rms():
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
    assert float(model.log_prob(pathological)) < float(model.log_prob(params))


def test_positive_flux_guard_applies_psf_dilution_to_mean_and_covariance():
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
    fraction = jnp.asarray(0.4)
    diluted = {
        "tau_drw_band": jnp.array([100.0]),
        "tau_perturb_band": jnp.array([2.0]),
        "quality_factor": jnp.asarray(0.6),
        "lag_blr": jnp.array([20.0]),
        "amp_cont_relflux": jnp.array([0.05]),
        "amp_blr_relflux": jnp.array([0.01]),
        "agn_fraction_by_band": jnp.reshape(fraction, (1,)),
        "log_jitter": jnp.full((1, 3), -10.0),
        "survey_delta_mag": jnp.zeros((1, 3)),
        "mean": jnp.array([-0.8]),
        "linear_trend": jnp.asarray(0.01),
        "linear_trend_band_offset": jnp.array([-0.02]),
    }
    manually_diluted = dict(diluted)
    manually_diluted.pop("agn_fraction_by_band")
    for name in (
        "amp_cont_relflux",
        "amp_blr_relflux",
        "mean",
        "linear_trend",
        "linear_trend_band_offset",
    ):
        manually_diluted[name] = manually_diluted[name] * fraction

    np.testing.assert_allclose(
        np.asarray(model.positive_flux_margin(diluted)),
        np.asarray(model.positive_flux_margin(manually_diluted)),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        np.asarray(model.positive_flux_log_penalty(diluted)),
        np.asarray(model.positive_flux_log_penalty(manually_diluted)),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        np.asarray(model.log_prob(diluted)),
        np.asarray(model.log_prob(manually_diluted)),
        rtol=2e-12,
        atol=2e-12,
    )


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
