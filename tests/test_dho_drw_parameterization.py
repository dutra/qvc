import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
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
    _scaled_phi_pairs,
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


def _reference_scaled_phi_pairs(inputs, max_order, series_terms=64):
    """High-order real-pair series oracle for the scaled phi functions."""

    z_real, z_imag, decay = inputs
    coefficients = jnp.asarray(
        [
            [
                1.0 / math.factorial(power + order)
                for order in range(1, int(max_order) + 1)
            ]
            for power in range(int(series_terms))
        ],
        dtype=z_real.dtype,
    )
    powers = jnp.arange(int(series_terms))
    z = z_real + 1j * z_imag
    values = jnp.exp(-decay) * jnp.sum(
        z ** powers[:, None] * coefficients,
        axis=0,
    )
    return jnp.concatenate([jnp.real(values), jnp.imag(values)])


def test_scaled_phi_branch_boundary_matches_high_order_values_and_jacobians():
    for erlang_order in (1, 3, 5, 8):
        max_order = erlang_order + 5

        def candidate(inputs):
            real, imag = _scaled_phi_pairs(
                inputs[0], inputs[1], inputs[2], max_order
            )
            return jnp.concatenate([real, imag])

        def reference(inputs):
            return _reference_scaled_phi_pairs(inputs, max_order)

        candidate_jacobian = jax.jit(jax.jacrev(candidate))
        reference_jacobian = jax.jit(jax.jacrev(reference))
        for radius, angle, decay in (
            (0.0, 0.0, 0.2),
            (1.0 - 1.0e-9, 0.0, 0.2),
            (1.0, 0.7, 2.0),
            (1.0 + 1.0e-9, 0.0, 0.2),
            (1.0 + 1.0e-9, 0.7, 2.0),
        ):
            inputs = jnp.asarray(
                [radius * np.cos(angle), radius * np.sin(angle), decay]
            )
            np.testing.assert_allclose(
                np.asarray(candidate(inputs)),
                np.asarray(reference(inputs)),
                rtol=2.0e-12,
                atol=2.0e-13,
            )
            np.testing.assert_allclose(
                np.asarray(candidate_jacobian(inputs)),
                np.asarray(reference_jacobian(inputs)),
                rtol=2.0e-11,
                atol=2.0e-12,
            )


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


def test_integrated_erlang_fused_likelihood_matches_block_tinygp_and_gradients():
    epoch_times = jnp.asarray([0.0, 1.0e-12, 3.0, 20.0, 100.0])
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

    def fused_objective(log_params):
        kernel = _integrated_erlang_kernel_from_log_params(log_params)
        return fast_quasisep.fused_log_probability(
            kernel,
            coordinates,
            diagonal,
            residual,
            sort_time=times,
        )

    transition_nonzero_indices = tuple(
        np.flatnonzero(
            np.diff(np.asarray(times), prepend=float(times[0])) != 0.0
        ).tolist()
    )
    assert 2 in transition_nonzero_indices

    def compact_fused_objective(log_params):
        kernel = _integrated_erlang_kernel_from_log_params(log_params)
        return fast_quasisep.fused_log_probability(
            kernel,
            coordinates,
            diagonal,
            residual,
            sort_time=times,
            transition_nonzero_indices=transition_nonzero_indices,
        )

    block_value_and_grad = jax.jit(jax.value_and_grad(block_objective))
    fused_value_and_grad = jax.jit(jax.value_and_grad(fused_objective))
    compact_fused_value_and_grad = jax.jit(
        jax.value_and_grad(compact_fused_objective)
    )
    reference_value_and_grad = jax.jit(
        jax.value_and_grad(reference_objective)
    )
    for quality_factor in (
        0.05,
        0.2,
        0.49,
        0.5 - 1e-8,
        0.5,
        0.5 + 1e-8,
        0.51,
        2.0,
        3.0,
    ):
        log_params = jnp.log(
            jnp.asarray([100.0, quality_factor, 0.02, 70.0])
        )
        block_value, block_gradient = block_value_and_grad(log_params)
        fused_value, fused_gradient = fused_value_and_grad(log_params)
        compact_fused_value, compact_fused_gradient = (
            compact_fused_value_and_grad(log_params)
        )
        reference_value, reference_gradient = reference_value_and_grad(log_params)
        np.testing.assert_allclose(
            np.asarray(compact_fused_value),
            np.asarray(fused_value),
            rtol=2e-10,
            atol=2e-11,
        )
        np.testing.assert_allclose(
            np.asarray(compact_fused_gradient),
            np.asarray(fused_gradient),
            rtol=2e-8,
            atol=2e-9,
        )
        np.testing.assert_allclose(
            np.asarray(fused_value),
            np.asarray(block_value),
            rtol=2e-10,
            atol=2e-11,
        )
        np.testing.assert_allclose(
            np.asarray(fused_gradient),
            np.asarray(block_gradient),
            rtol=2e-8,
            atol=2e-9,
        )
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


def test_compact_fused_likelihood_handles_all_zero_gaps():
    times = jnp.zeros(4)
    bands = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
    coordinates = (times, bands)
    diagonal = jnp.full(4, 0.02**2)
    residual = jnp.asarray([0.01, -0.02, 0.03, -0.01])

    def objective(log_params, compact):
        kernel = _integrated_erlang_kernel_from_log_params(log_params)
        return fast_quasisep.fused_log_probability(
            kernel,
            coordinates,
            diagonal,
            residual,
            sort_time=times,
            transition_nonzero_indices=() if compact else None,
        )

    log_params = jnp.log(jnp.asarray([100.0, 0.5, 0.02, 70.0]))
    full = jax.jit(jax.value_and_grad(lambda value: objective(value, False)))(
        log_params
    )
    compact = jax.jit(jax.value_and_grad(lambda value: objective(value, True)))(
        log_params
    )
    np.testing.assert_allclose(
        np.asarray(compact[0]), np.asarray(full[0]), rtol=2e-10, atol=2e-11
    )
    np.testing.assert_allclose(
        np.asarray(compact[1]), np.asarray(full[1]), rtol=2e-8, atol=2e-9
    )


def test_compact_fused_likelihood_rejects_stale_or_invalid_indices():
    times = jnp.asarray([0.0, 0.0, 1.0, 2.0])
    bands = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
    coordinates = (times, bands)
    diagonal = jnp.full(4, 0.02**2)
    residual = jnp.asarray([0.01, -0.02, 0.03, -0.01])
    kernel = _integrated_erlang_kernel_from_log_params(
        jnp.log(jnp.asarray([100.0, 0.5, 0.02, 70.0]))
    )

    stale_value = jax.jit(
        lambda: fast_quasisep.fused_log_probability(
            kernel,
            coordinates,
            diagonal,
            residual,
            sort_time=times,
            transition_nonzero_indices=(),
        )
    )()
    assert jnp.isnan(stale_value)

    with pytest.raises(ValueError, match="sorted and unique"):
        fast_quasisep.fused_log_probability(
            kernel,
            coordinates,
            diagonal,
            residual,
            sort_time=times,
            transition_nonzero_indices=(2, 2),
        )
    for invalid_indices in ((-1, 2), (2, 4)):
        with pytest.raises(ValueError, match="out-of-range"):
            fast_quasisep.fused_log_probability(
                kernel,
                coordinates,
                diagonal,
                residual,
                sort_time=times,
                transition_nonzero_indices=invalid_indices,
            )


def test_compact_full_model_matches_block_solver_for_unsorted_tied_epochs():
    times = jnp.asarray([2.0, 0.0, 1.0, 0.0, 2.0, 1.0 + 1e-12])
    bands = jnp.asarray([1, 0, 1, 1, 0, 0], dtype=jnp.int32)
    surveys = jnp.asarray([0, 1, 2, 0, 1, 2], dtype=jnp.int32)
    observations = jnp.asarray([0.02, -0.01, 0.03, -0.02, 0.01, -0.015])
    errors = jnp.asarray([0.02, 0.025, 0.018, 0.023, 0.021, 0.019])
    sorted_times = np.sort(np.asarray(times))
    transition_nonzero_indices = tuple(
        np.flatnonzero(
            np.diff(sorted_times, prepend=sorted_times[0]) != 0.0
        ).tolist()
    )
    model = make_multiband_dho_blr_flux_linearized_erlang_drw_model(
        (times, bands),
        observations,
        errors,
        n_band=2,
        survey_idx=surveys,
        erlang_order=3,
        transition_nonzero_indices=transition_nonzero_indices,
    )

    initial = jnp.concatenate(
        [
            jnp.log(jnp.asarray([80.0, 120.0])),
            jnp.log(jnp.asarray([0.015, 0.025])),
            jnp.log(jnp.asarray([0.6])),
            jnp.log(jnp.asarray([60.0, 90.0])),
            jnp.log(jnp.asarray([0.07, 0.09])),
            jnp.log(jnp.asarray([0.015, 0.02])),
            jnp.full(6, -8.0),
            jnp.zeros(6),
            jnp.asarray([0.01, -0.015]),
            jnp.asarray([2e-4]),
            jnp.asarray([1e-4, -1e-4]),
            jnp.asarray([3.0, 4.0]),
        ]
    )

    def unpack(value):
        tau_drw = jnp.exp(value[0:2])
        return {
            "tau_drw_band": tau_drw,
            "tau_perturb_band": tau_drw * jnp.exp(value[2:4]),
            "quality_factor": jnp.exp(value[4]),
            "lag_blr": jnp.exp(value[5:7]),
            "amp_cont_relflux": jnp.exp(value[7:9]),
            "amp_blr_relflux": jnp.exp(value[9:11]),
            "log_jitter": value[11:17].reshape(2, 3),
            "survey_delta_mag": value[17:23].reshape(2, 3),
            "mean": value[23:25],
            "linear_trend": value[25],
            "linear_trend_band_offset": value[26:28],
            "agn_fraction_by_band": jax.nn.sigmoid(value[28:30]),
        }

    def compact_objective(value):
        return model._log_prob_impl(unpack(value))

    def block_objective(value):
        return model._block_log_prob_impl(unpack(value))

    compact = jax.jit(jax.value_and_grad(compact_objective))(initial)
    block = jax.jit(jax.value_and_grad(block_objective))(initial)
    np.testing.assert_allclose(
        np.asarray(compact[0]), np.asarray(block[0]), rtol=2e-10, atol=2e-11
    )
    np.testing.assert_allclose(
        np.asarray(compact[1]), np.asarray(block[1]), rtol=2e-8, atol=2e-9
    )

    _, tangent = jax.jvp(
        model.log_prob,
        (unpack(initial),),
        (unpack(jnp.ones_like(initial)),),
    )
    assert np.isfinite(float(tangent))


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
