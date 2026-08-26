from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from qvc.hubble.latent_alpha_completeness import (
    BETA_ALPHA_L_PARAMETER,
    GAUSS_HERMITE_ORDER,
    RESPONSE_COEFFICIENT_PRIOR_SIGMA,
    LatentAlphaConfig,
    M2500_TO_LOG_NU_LNU_INTERCEPT,
    absolute_m2500_to_log_nu_lnu,
    bounded_alpha_completeness,
    bounded_response_from_kappa,
    deterministic_joint_draw_indices,
    latent_alpha_config_hash,
    latent_alpha_parameter_prior_specs,
    latent_alpha_provenance,
    marginalized_alpha_completeness,
    normal_gauss_hermite_nodes,
    parent_alpha_logpdf,
    parent_alpha_mean,
    parent_alpha_mean_from_config,
    resolve_lf_luminosity_state,
    response_coefficient_names,
    response_coefficient_prior_specs,
    response_design_matrix,
    select_deterministic_joint_draws,
    solve_response_kappa,
    stable_sigmoid,
)
from scipy.special import ndtri


def _nonzero_coefficients(include_magnitude_interactions=False):
    coefficients = {
        "alpha_sel_z_p0_linear": 0.70,
        "alpha_sel_z_p0_quadratic": -0.25,
        "alpha_sel_z_p1_linear": 0.40,
        "alpha_sel_z_p2_quadratic": 0.15,
    }
    if include_magnitude_interactions:
        coefficients.update(
            {
                "alpha_sel_mag_z_p0_linear": -0.8,
                "alpha_sel_mag_z_p1_quadratic": 0.35,
            }
        )
    return coefficients


def test_config_is_immutable_and_validates_mode_specific_beta():
    config = LatentAlphaConfig()
    with pytest.raises(FrozenInstanceError):
        config.mode = "joint"

    with pytest.raises(ValueError, match="fixed mode requires"):
        LatentAlphaConfig(mode="fixed")
    with pytest.raises(ValueError, match="only valid in fixed mode"):
        LatentAlphaConfig(mode="joint", fixed_beta_l=0.1)
    with pytest.raises(ValueError, match="sigma must be positive"):
        LatentAlphaConfig(sigma=0.0)
    with pytest.raises(ValueError, match="quadrature_order is fixed"):
        LatentAlphaConfig(quadrature_order=8)
    with pytest.raises(ValueError, match="strictly ordered"):
        LatentAlphaConfig(beta_l_prior=(0.5, -0.5))

    custom_prior = LatentAlphaConfig(mode="joint", beta_l_prior=(-0.2, 0.35))
    assert latent_alpha_parameter_prior_specs(custom_prior)[
        BETA_ALPHA_L_PARAMETER
    ]["low"] == -0.2
    assert latent_alpha_parameter_prior_specs(custom_prior)[
        BETA_ALPHA_L_PARAMETER
    ]["high"] == 0.35


def test_off_is_identical_to_fixed_zero_and_joint_reads_named_parameter():
    log_luminosity = np.array([44.5, 45.5, 46.5])
    off = LatentAlphaConfig(mode="off")
    fixed_zero = LatentAlphaConfig(mode="fixed", fixed_beta_l=0.0)

    np.testing.assert_array_equal(
        parent_alpha_mean_from_config(log_luminosity, off),
        parent_alpha_mean_from_config(log_luminosity, fixed_zero),
    )

    joint = LatentAlphaConfig(mode="joint")
    with pytest.raises(KeyError, match=BETA_ALPHA_L_PARAMETER):
        joint.beta_l()
    assert joint.beta_l({BETA_ALPHA_L_PARAMETER: 0.12}) == pytest.approx(0.12)
    assert joint.joint_parameter_names() == (BETA_ALPHA_L_PARAMETER,)


def test_positive_beta_means_more_luminous_parent_is_bluer():
    luminosities = np.array([44.5, 45.5, 46.5])
    positive = parent_alpha_mean(luminosities, 0.2)
    negative = parent_alpha_mean(luminosities, -0.2)

    np.testing.assert_allclose(positive, [-0.7, -0.5, -0.3])
    assert np.all(np.diff(positive) > 0.0)
    assert np.all(np.diff(negative) < 0.0)

    logpdf_at_mean = parent_alpha_logpdf(positive, luminosities, 0.2)
    expected = -np.log(0.3 * np.sqrt(2.0 * np.pi))
    np.testing.assert_allclose(logpdf_at_mean, expected)


def test_m2500_conversion_is_exact_monochromatic_ab_nu_lnu():
    magnitudes = np.array([-25.0, -23.0])
    converted = absolute_m2500_to_log_nu_lnu(magnitudes)
    np.testing.assert_allclose(
        converted,
        M2500_TO_LOG_NU_LNU_INTERCEPT - 0.4 * magnitudes,
        rtol=0.0,
        atol=1e-14,
    )
    assert converted[0] - converted[1] == pytest.approx(0.8)
    # Guard against reintroducing QVC's historical rounded M=90-2.5logL
    # convenience conversion into this physically defined parent coordinate.
    assert converted[0] != pytest.approx(-0.4 * (-25.0 - 90.0))


@pytest.mark.parametrize(
    ("lf_model", "shen_mode", "expected"),
    [
        ("shen", "type1_intrinsic", "dereddened"),
        ("shen", "type1_attenuated", "attenuated"),
        ("shen", "all_nh_attenuated", "attenuated"),
        ("wang2026_type1_lade_a", None, "attenuated"),
        ("palanque2016_ple_lede", None, "attenuated"),
        ("kulkarni2019_type1_model1", None, "attenuated"),
        ("kulkarni2019_type1_model2", None, "attenuated"),
        ("kulkarni2019_type1_model3", None, "attenuated"),
    ],
)
def test_lf_resolves_the_scientifically_matching_luminosity_state(
    lf_model, shen_mode, expected
):
    assert (
        resolve_lf_luminosity_state(lf_model, shen_lf_mode=shen_mode) == expected
    )
    config = LatentAlphaConfig.for_lf(
        lf_model=lf_model, shen_lf_mode=shen_mode
    )
    assert config.luminosity_state == expected


def test_lf_resolver_rejects_semantic_mismatch_and_unknown_fallback():
    with pytest.raises(ValueError, match="requires attenuated"):
        resolve_lf_luminosity_state(
            "wang2026_type1_lade_a", requested_state="dereddened"
        )
    with pytest.raises(ValueError, match="Unknown LF model"):
        resolve_lf_luminosity_state("not_a_real_lf")


def test_response_labels_and_priors_are_stable_and_shrunk():
    base_names = response_coefficient_names(False)
    full_names = response_coefficient_names(True)
    assert len(base_names) == 8
    assert len(full_names) == 16
    assert full_names[:8] == base_names
    assert base_names[:4] == (
        "alpha_sel_z_p0_linear",
        "alpha_sel_z_p0_quadratic",
        "alpha_sel_z_p1_linear",
        "alpha_sel_z_p1_quadratic",
    )

    priors = response_coefficient_prior_specs(True)
    assert tuple(priors) == full_names
    for name, prior in priors.items():
        assert prior["distribution"] == "truncated_normal"
        assert prior["mean"] == 0.0
        assert prior["sigma"] == RESPONSE_COEFFICIENT_PRIOR_SIGMA == 0.5
        assert prior["low"] < 0.0 < prior["high"]
        if "_mag_" in name:
            assert (prior["low"], prior["high"]) == (-2.0, 2.0)
        else:
            assert (prior["low"], prior["high"]) == (-3.0, 3.0)

    joint_priors = latent_alpha_parameter_prior_specs(
        LatentAlphaConfig(mode="joint")
    )
    assert joint_priors[BETA_ALPHA_L_PARAMETER] == {
        "distribution": "uniform",
        "low": -0.5,
        "high": 0.5,
        "units": "alpha_nu_per_dex",
    }


def test_response_design_uses_four_legendre_terms_and_centered_alpha_basis():
    # At the alpha reference mean u=0 and q=-1.  At the redshift midpoint,
    # [P0,P1,P2,P3] = [1,0,-1/2,0].
    design = response_design_matrix(-0.5, 1.8)
    np.testing.assert_allclose(
        design,
        [0.0, -1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        atol=1e-15,
    )
    assert design.shape == (8,)

    full = response_design_matrix(
        -0.2,
        1.8,
        magnitude=21.25,
        include_magnitude_interactions=True,
    )
    assert full.shape == (16,)
    np.testing.assert_array_equal(full[8:], np.zeros(8))


def test_response_coordinates_are_edge_clamped_to_calibration_support():
    at_low_z = response_design_matrix(-0.2, 0.44)
    below_low_z = response_design_matrix(-0.2, -10.0)
    at_high_z = response_design_matrix(-0.2, 3.16)
    above_high_z = response_design_matrix(-0.2, 20.0)
    np.testing.assert_array_equal(below_low_z, at_low_z)
    np.testing.assert_array_equal(above_high_z, at_high_z)

    low_mag = response_design_matrix(
        -0.2,
        1.5,
        magnitude=18.5,
        include_magnitude_interactions=True,
    )
    below_low_mag = response_design_matrix(
        -0.2,
        1.5,
        magnitude=0.0,
        include_magnitude_interactions=True,
    )
    np.testing.assert_array_equal(below_low_mag, low_mag)


def test_twelve_point_gauss_hermite_rule_has_standard_normal_moments():
    nodes, weights = normal_gauss_hermite_nodes()
    assert nodes.shape == weights.shape == (GAUSS_HERMITE_ORDER,)
    np.testing.assert_allclose(np.sum(weights), 1.0, atol=1e-15)
    np.testing.assert_allclose(np.sum(weights * nodes), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.sum(weights * nodes**2), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.sum(weights * nodes**4), 3.0, atol=1e-13)

    # The returned arrays cannot mutate the cached canonical quadrature.
    nodes[0] = 999.0
    fresh_nodes, _ = normal_gauss_hermite_nodes()
    assert fresh_nodes[0] != 999.0


def test_vectorized_kappa_solver_is_stable_and_normalizes_each_row():
    _, weights = normal_gauss_hermite_nodes()
    offsets = np.linspace(-2.0, 2.0, GAUSS_HERMITE_ORDER)
    offsets = np.stack((offsets, -0.4 * offsets, offsets**2 - 1.0))
    base = np.array([1.0e-10, 0.25, 0.999999])
    kappa = solve_response_kappa(base, offsets, weights=weights)

    response = stable_sigmoid(
        np.log(base / (1.0 - base))[:, None] + offsets - kappa[:, None]
    )
    np.testing.assert_allclose(response @ weights, base, rtol=2e-12, atol=1e-15)
    assert np.all(np.isfinite(kappa))


def test_precalibrated_response_broadcasts_over_joint_alpha_draws():
    base = np.array([0.2, 0.8])
    offsets_at_nodes = np.zeros((2, GAUSS_HERMITE_ORDER))
    kappa = solve_response_kappa(base, offsets_at_nodes)
    draw_offsets = np.array([[-1.0, 0.0, 1.0], [0.5, -0.5, 0.0]])
    response = bounded_response_from_kappa(
        base[:, None], draw_offsets, kappa[:, None]
    )
    assert response.shape == draw_offsets.shape
    assert np.all((response >= 0.0) & (response <= 1.0))
    np.testing.assert_allclose(response[:, 2], [0.40460968, 0.8], rtol=1e-7)


def test_bounded_response_marginalizes_exactly_to_base_c3():
    config = LatentAlphaConfig(mode="fixed", fixed_beta_l=0.2)
    coefficients = _nonzero_coefficients()
    base = np.array([0.0, 0.02, 0.30, 0.88, 1.0])
    redshift = np.linspace(0.44, 3.16, base.size)
    log_luminosity = np.linspace(44.5, 46.5, base.size)

    marginalized = marginalized_alpha_completeness(
        base,
        redshift,
        log_luminosity,
        coefficients,
        config=config,
    )
    np.testing.assert_allclose(marginalized, base, rtol=0.0, atol=3e-15)

    parent_mean = parent_alpha_mean_from_config(log_luminosity, config)
    conditional = bounded_alpha_completeness(
        base,
        parent_mean,
        redshift,
        log_luminosity,
        coefficients,
        config=config,
    )
    assert np.all(np.isfinite(conditional))
    assert np.all((conditional >= 0.0) & (conditional <= 1.0))
    assert conditional[0] == 0.0
    assert conditional[-1] == 1.0


def test_off_and_fixed_zero_give_identical_normalized_response():
    coefficients = _nonzero_coefficients()
    base = np.array([0.1, 0.4, 0.9])
    alpha = np.array([-1.1, -0.5, 0.2])
    redshift = np.array([0.6, 1.7, 3.0])
    log_luminosity = np.array([44.7, 45.5, 46.2])
    off = LatentAlphaConfig(mode="off")
    fixed_zero = LatentAlphaConfig(mode="fixed", fixed_beta_l=0.0)

    result_off = bounded_alpha_completeness(
        base, alpha, redshift, log_luminosity, coefficients, config=off
    )
    result_fixed = bounded_alpha_completeness(
        base, alpha, redshift, log_luminosity, coefficients, config=fixed_zero
    )
    np.testing.assert_array_equal(result_off, result_fixed)


def test_magnitude_interactions_vanish_at_pivot_and_change_off_pivot_response():
    config = LatentAlphaConfig(include_magnitude_interactions=True)
    interaction_only = {
        "alpha_sel_mag_z_p0_linear": 1.2,
        "alpha_sel_mag_z_p1_quadratic": -0.6,
    }
    zeros = {}
    args = {
        "base_completeness": 0.45,
        "alpha_nu": -0.15,
        "redshift": 2.0,
        "log_luminosity": 45.8,
        "config": config,
    }

    at_pivot = bounded_alpha_completeness(
        **args, coefficients=interaction_only, magnitude=config.magnitude_pivot
    )
    zero_at_pivot = bounded_alpha_completeness(
        **args, coefficients=zeros, magnitude=config.magnitude_pivot
    )
    assert at_pivot == pytest.approx(zero_at_pivot, abs=1e-14)

    off_pivot = bounded_alpha_completeness(
        **args, coefficients=interaction_only, magnitude=23.5
    )
    zero_off_pivot = bounded_alpha_completeness(
        **args, coefficients=zeros, magnitude=23.5
    )
    assert abs(float(off_pivot) - float(zero_off_pivot)) > 0.02
    marginalized = marginalized_alpha_completeness(
        args["base_completeness"],
        args["redshift"],
        args["log_luminosity"],
        interaction_only,
        config=config,
        magnitude=23.5,
    )
    assert marginalized == pytest.approx(args["base_completeness"], abs=2e-15)


def test_injection_recovers_redshift_independent_parent_after_inverse_selection():
    config = LatentAlphaConfig(mode="off")
    coefficients = {
        "alpha_sel_z_p0_linear": 0.8,
        "alpha_sel_z_p1_linear": 1.15,
        "alpha_sel_z_p2_quadratic": -0.35,
    }
    alpha = np.linspace(-2.3, 1.3, 20_001)
    parent = np.exp(parent_alpha_logpdf(alpha, 45.5, 0.0))
    selected_means = []
    recovered_means = []
    for redshift in np.linspace(0.5, 3.1, 7):
        response = bounded_alpha_completeness(
            0.42,
            alpha,
            redshift,
            45.5,
            coefficients,
            config=config,
        )
        selected_density = parent * response
        selected_means.append(
            np.trapezoid(alpha * selected_density, alpha)
            / np.trapezoid(selected_density, alpha)
        )
        inverse_weighted = selected_density / response
        recovered_means.append(
            np.trapezoid(alpha * inverse_weighted, alpha)
            / np.trapezoid(inverse_weighted, alpha)
        )
    assert np.ptp(selected_means) > 0.15
    np.testing.assert_allclose(recovered_means, config.mu, atol=2e-9)


def test_injection_recovers_luminosity_parent_with_magnitude_interaction():
    config = LatentAlphaConfig(
        mode="fixed",
        fixed_beta_l=0.25,
        include_magnitude_interactions=True,
    )
    coefficients = {
        "alpha_sel_z_p0_linear": 0.65,
        "alpha_sel_z_p1_quadratic": -0.25,
        "alpha_sel_mag_z_p0_linear": 0.9,
        "alpha_sel_mag_z_p2_quadratic": 0.3,
    }
    alpha = np.linspace(-2.5, 1.5, 25_001)
    recovered = []
    expected = []
    selected = []
    for log_luminosity, magnitude in ((44.5, 19.0), (46.5, 23.0)):
        mean = float(parent_alpha_mean_from_config(log_luminosity, config))
        parent = np.exp(
            parent_alpha_logpdf(
                alpha,
                log_luminosity,
                config.fixed_beta_l,
                mu=config.mu,
                sigma=config.sigma,
                logl_pivot=config.logl_pivot,
            )
        )
        response = bounded_alpha_completeness(
            0.5,
            alpha,
            1.8,
            log_luminosity,
            coefficients,
            config=config,
            magnitude=magnitude,
        )
        selected_density = parent * response
        selected.append(
            np.trapezoid(alpha * selected_density, alpha)
            / np.trapezoid(selected_density, alpha)
        )
        inverse_weighted = selected_density / response
        recovered.append(
            np.trapezoid(alpha * inverse_weighted, alpha)
            / np.trapezoid(inverse_weighted, alpha)
        )
        expected.append(mean)
    assert abs((selected[1] - selected[0]) - (expected[1] - expected[0])) > 0.05
    np.testing.assert_allclose(recovered, expected, atol=2e-9)


def test_deterministic_16_draw_likelihood_quadrature_tracks_64_draw_reference():
    config = LatentAlphaConfig(mode="off")
    quantiles = (np.arange(64, dtype=float) + 0.5) / 64.0
    alpha = config.mu + config.sigma * ndtri(quantiles)
    response = bounded_alpha_completeness(
        0.38,
        alpha,
        2.1,
        45.5,
        {
            "alpha_sel_z_p0_linear": 0.35,
            "alpha_sel_z_p1_quadratic": -0.15,
        },
        config=config,
    )
    integrand = np.exp(parent_alpha_logpdf(alpha, 45.5, 0.0)) * response
    compact = integrand[deterministic_joint_draw_indices()]
    assert np.mean(compact) == pytest.approx(np.mean(integrand), rel=0.012)


def test_deterministic_16_of_64_selection_preserves_joint_draw_indexing():
    expected = np.arange(2, 64, 4)
    indices = deterministic_joint_draw_indices()
    np.testing.assert_array_equal(indices, expected)

    joint = {
        "alpha": np.arange(2 * 64).reshape(2, 64),
        "host": 1000 + np.arange(2 * 64).reshape(2, 64),
    }
    selected = select_deterministic_joint_draws(joint)
    np.testing.assert_array_equal(selected["alpha"], joint["alpha"][:, expected])
    np.testing.assert_array_equal(
        selected["host"] - selected["alpha"], np.full((2, 16), 1000)
    )
    with pytest.raises(ValueError, match="expected exactly 64"):
        select_deterministic_joint_draws(np.arange(63))


def test_serialization_hash_and_provenance_are_stable_and_complete():
    config = LatentAlphaConfig.for_lf(
        lf_model="shen",
        shen_lf_mode="type1_intrinsic",
        mode="joint",
        include_magnitude_interactions=True,
    )
    round_trip = LatentAlphaConfig.from_json(config.to_json())
    assert round_trip == config

    coefficients_a = {
        "alpha_sel_z_p1_linear": 0.1,
        "alpha_sel_z_p0_linear": -0.2,
    }
    coefficients_reordered = dict(reversed(tuple(coefficients_a.items())))
    hash_a = latent_alpha_config_hash(config, coefficients_a)
    assert hash_a == latent_alpha_config_hash(config, coefficients_reordered)
    assert hash_a != latent_alpha_config_hash(
        config, {**coefficients_a, "alpha_sel_z_p1_linear": 0.2}
    )

    provenance = latent_alpha_provenance(config, coefficients_a)
    json.dumps(provenance, allow_nan=False)
    assert provenance["config_hash_sha256"] == hash_a
    assert provenance["parent_distribution"]["redshift_evolution"] == "none"
    assert provenance["parent_distribution"]["luminosity_state"] == "dereddened"
    assert provenance["selection_response"]["range"] == [0.0, 1.0]
    assert provenance["joint_draw_selection"]["indices"] == list(range(2, 64, 4))
    assert provenance["coefficient_priors"]["alpha_sel_z_p0_linear"][
        "distribution"
    ] == "truncated_normal"
