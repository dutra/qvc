import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble.empirical_luminosity_functions import (
    EMPIRICAL_LF_MODEL_IDS,
    KULKARNI2019_SOURCE_COSMOLOGY,
    KULKARNI2019_TYPE1_MODEL1,
    KULKARNI2019_TYPE1_MODEL2,
    KULKARNI2019_TYPE1_MODEL3,
    KULKARNI2019_TYPE1_MODEL_IDS,
    LFGrid,
    PALANQUE2016_PLE_LEDE,
    PALANQUE2016_SOURCE_COSMOLOGY,
    WANG2026_SOURCE_COSMOLOGY,
    WANG2026_TYPE1_LADE_A,
    build_empirical_lf,
    build_kulkarni2019_type1_model1,
    build_kulkarni2019_type1_model2,
    build_kulkarni2019_type1_model3,
    build_palanque2016_ple_lede,
    build_wang2026_type1_lade_a,
    kulkarni2019_evolving_parameters,
    kulkarni2019_model1_evolving_parameters,
    kulkarni2019_model2_evolving_parameters,
    kulkarni2019_model3_evolving_parameters,
    palanque2016_evolving_parameters,
    wang2026_e1,
    wang2026_e2,
)


LOG10_TWO = np.log10(2.0)


def test_empirical_lf_model_ids_are_explicit_and_stable():
    assert EMPIRICAL_LF_MODEL_IDS == (
        "wang2026_type1_lade_a",
        "palanque2016_ple_lede",
        "kulkarni2019_type1_model1",
        "kulkarni2019_type1_model2",
        "kulkarni2019_type1_model3",
    )
    assert WANG2026_TYPE1_LADE_A == EMPIRICAL_LF_MODEL_IDS[0]
    assert PALANQUE2016_PLE_LEDE == EMPIRICAL_LF_MODEL_IDS[1]
    assert KULKARNI2019_TYPE1_MODEL_IDS == EMPIRICAL_LF_MODEL_IDS[2:]


def test_wang2026_evolution_normalization_and_z2p2_gold_values():
    np.testing.assert_allclose(wang2026_e1(0.0), 1.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(wang2026_e2(0.0), 1.0, rtol=0.0, atol=1e-14)

    density_evolution = wang2026_e1(2.2)
    luminosity_evolution = wang2026_e2(2.2)
    evolving_break = -22.345 - 2.5 * np.log10(luminosity_evolution)

    np.testing.assert_allclose(
        density_evolution,
        1.81004294684,
        rtol=0.0,
        atol=5e-12,
    )
    np.testing.assert_allclose(
        luminosity_evolution,
        40.9532939287,
        rtol=0.0,
        atol=5e-11,
    )
    np.testing.assert_allclose(
        evolving_break,
        -26.375722096,
        rtol=0.0,
        atol=5e-11,
    )


def test_wang2026_builder_is_phi_star_over_two_at_the_z0_break():
    grid = build_wang2026_type1_lade_a(
        target_magnitude_grid=np.array([-22.345]),
        redshift_grid=np.array([0.0]),
        target_cosmology=WANG2026_SOURCE_COSMOLOGY,
    )

    assert isinstance(grid, LFGrid)
    assert grid.model_id == WANG2026_TYPE1_LADE_A
    assert grid.native_magnitude_name == "M_1450_AB"
    np.testing.assert_allclose(
        grid.phi_log10,
        [[-6.635 - LOG10_TWO]],
        rtol=0.0,
        atol=1e-12,
    )


def test_palanque2016_pivot_and_z3p16_parameter_gold_values():
    pivot = palanque2016_evolving_parameters(2.2)
    np.testing.assert_allclose(pivot.m_star, -26.639, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        pivot.log10_phi_star,
        -5.93,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        pivot.alpha_bright,
        -3.89,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        pivot.beta_faint,
        -1.47,
        rtol=0.0,
        atol=1e-12,
    )

    high_redshift = palanque2016_evolving_parameters(3.16)
    np.testing.assert_allclose(
        high_redshift.m_star,
        -26.7734,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        high_redshift.log10_phi_star,
        -6.426896,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        high_redshift.alpha_bright,
        -3.5828,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        high_redshift.beta_faint,
        -1.47,
        rtol=0.0,
        atol=1e-12,
    )

    grid = build_palanque2016_ple_lede(
        target_magnitude_grid=np.array([pivot.m_star, high_redshift.m_star]),
        redshift_grid=np.array([2.2, 3.16]),
        target_cosmology=PALANQUE2016_SOURCE_COSMOLOGY,
    )
    assert isinstance(grid, LFGrid)
    assert grid.model_id == PALANQUE2016_PLE_LEDE
    assert grid.native_magnitude_name == "M_g(z=2)_AB"
    np.testing.assert_allclose(
        grid.phi_log10[0, 0],
        -5.93 - LOG10_TWO,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        grid.phi_log10[1, 1],
        -6.426896 - LOG10_TWO,
        rtol=0.0,
        atol=1e-12,
    )


def test_palanque2016_z1_cosmology_remap_gold_values_and_break_density():
    redshift = 1.0
    target_cosmology = FlatLambdaCDM(H0=70.0, Om0=0.3)
    delta_distance_modulus = float(
        target_cosmology.distmod(redshift).value
        - PALANQUE2016_SOURCE_COSMOLOGY.distmod(redshift).value
    )
    log10_volume_ratio = float(
        np.log10(
            PALANQUE2016_SOURCE_COSMOLOGY.differential_comoving_volume(
                redshift
            ).value
            / target_cosmology.differential_comoving_volume(redshift).value
        )
    )

    np.testing.assert_allclose(
        delta_distance_modulus,
        -0.057813862591253,
        rtol=0.0,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        log10_volume_ratio,
        0.0331898129268792,
        rtol=0.0,
        atol=2e-13,
    )

    parameters = palanque2016_evolving_parameters(redshift)
    # The builder evaluates M_source = M_target + DM_target - DM_source.
    target_break = parameters.m_star - delta_distance_modulus
    grid = build_palanque2016_ple_lede(
        target_magnitude_grid=np.array([target_break]),
        redshift_grid=np.array([redshift]),
        target_cosmology=target_cosmology,
    )
    expected_log10_density = (
        parameters.log10_phi_star - LOG10_TWO + log10_volume_ratio
    )
    np.testing.assert_allclose(
        grid.phi_log10,
        [[expected_log10_density]],
        rtol=0.0,
        atol=2e-12,
    )


def test_kulkarni2019_final_table3_z1_parameter_golds_for_all_models():
    cases = (
        (
            KULKARNI2019_TYPE1_MODEL1,
            kulkarni2019_model1_evolving_parameters,
            (-24.660, -6.382, -3.739, -1.8302578995588528),
        ),
        (
            KULKARNI2019_TYPE1_MODEL2,
            kulkarni2019_model2_evolving_parameters,
            (-24.536, -6.310, -3.661, -1.7945606992823826),
        ),
        (
            KULKARNI2019_TYPE1_MODEL3,
            kulkarni2019_model3_evolving_parameters,
            (-24.506, -6.286, -3.654, -1.766),
        ),
    )
    for model_id, wrapper, expected in cases:
        generic = kulkarni2019_evolving_parameters(1.0, model_id)
        specific = wrapper(1.0)
        for parameters in (generic, specific):
            actual = (
                parameters.m_star,
                parameters.log10_phi_star,
                parameters.alpha_bright,
                parameters.beta_faint,
            )
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-14)


def test_kulkarni2019_broken_beta_pivots_and_model3_linear_evolution():
    model1 = kulkarni2019_model1_evolving_parameters(3.773)
    model2 = kulkarni2019_model2_evolving_parameters(2.379)
    model3 = kulkarni2019_model3_evolving_parameters(
        np.array([0.0, 1.0, 3.16])
    )

    # At z=c32, zeta=0 and both denominator terms equal one.
    np.testing.assert_allclose(
        model1.beta_faint,
        -2.312 + 0.559 / 2.0,
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        model2.beta_faint,
        -2.264 + 0.530 / 2.0,
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        model3.beta_faint,
        -1.602 - 0.082 * (1.0 + np.array([0.0, 1.0, 3.16])),
        rtol=0.0,
        atol=2e-15,
    )
    assert model3.beta_faint.flags.writeable is False


@pytest.mark.parametrize(
    "model_id",
    KULKARNI2019_TYPE1_MODEL_IDS,
)
def test_kulkarni2019_vector_parameter_evaluation_matches_scalar_calls(model_id):
    redshifts = np.array([0.6, 1.0, 3.16, 6.5])
    vector = kulkarni2019_evolving_parameters(redshifts, model_id)

    for field in ("m_star", "log10_phi_star", "alpha_bright", "beta_faint"):
        expected = np.array(
            [
                getattr(kulkarni2019_evolving_parameters(z, model_id), field)
                for z in redshifts
            ]
        )
        np.testing.assert_allclose(
            getattr(vector, field),
            expected,
            rtol=0.0,
            atol=2e-14,
        )


def test_kulkarni2019_builders_match_final_table3_lf_golds_at_z3p16():
    cases = (
        (
            KULKARNI2019_TYPE1_MODEL1,
            build_kulkarni2019_type1_model1,
            -6.602502549440217,
        ),
        (
            KULKARNI2019_TYPE1_MODEL2,
            build_kulkarni2019_type1_model2,
            -6.435558316423153,
        ),
        (
            KULKARNI2019_TYPE1_MODEL3,
            build_kulkarni2019_type1_model3,
            -6.596149840874216,
        ),
    )
    for model_id, builder, expected_log10_phi in cases:
        grid = builder(
            target_magnitude_grid=np.array([-25.0]),
            redshift_grid=np.array([3.16]),
            target_cosmology=KULKARNI2019_SOURCE_COSMOLOGY,
        )
        dispatched = build_empirical_lf(
            model_id,
            target_magnitude_grid=np.array([-25.0]),
            redshift_grid=np.array([3.16]),
            target_cosmology=KULKARNI2019_SOURCE_COSMOLOGY,
        )

        assert grid.model_id == model_id
        assert dispatched.model_id == model_id
        np.testing.assert_allclose(
            grid.phi_log10,
            [[expected_log10_phi]],
            rtol=0.0,
            atol=2e-13,
        )
        np.testing.assert_array_equal(dispatched.phi_log10, grid.phi_log10)
        assert grid.native_magnitude_name == "M_1450_AB"
        assert grid.reference_wavelength_angstrom == 1450.0
        assert grid.native_to_monochromatic_ab_offset == 0.0
        assert grid.calibration_redshift_range == (0.6, 6.5)
        assert "table3_published_rounded_medians" in grid.formula_version


@pytest.mark.parametrize(
    ("model_id", "builder"),
    (
        (KULKARNI2019_TYPE1_MODEL1, build_kulkarni2019_type1_model1),
        (KULKARNI2019_TYPE1_MODEL2, build_kulkarni2019_type1_model2),
        (KULKARNI2019_TYPE1_MODEL3, build_kulkarni2019_type1_model3),
    ),
)
def test_kulkarni2019_each_model_is_phi_star_over_two_at_its_break(
    model_id,
    builder,
):
    redshift = 2.4
    parameters = kulkarni2019_evolving_parameters(redshift, model_id)
    grid = builder(
        target_magnitude_grid=np.array([parameters.m_star]),
        redshift_grid=np.array([redshift]),
        target_cosmology=KULKARNI2019_SOURCE_COSMOLOGY,
    )

    np.testing.assert_allclose(
        grid.phi_log10[0, 0],
        parameters.log10_phi_star - LOG10_TWO,
        rtol=0.0,
        atol=2e-13,
    )


def test_kulkarni2019_model2_uses_final_not_obsolete_preprint_coefficients():
    parameters = kulkarni2019_model2_evolving_parameters(1.0)

    # The archived 2018-preprint/Shen-loader values at z=1 are
    # (-24.468, -6.25, -3.667, -1.6400348982).  This freezes the materially
    # different final 2019 Table 3 result as provenance protection.
    np.testing.assert_allclose(parameters.m_star, -24.536, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        parameters.log10_phi_star,
        -6.310,
        rtol=0.0,
        atol=1e-14,
    )
    assert not np.isclose(parameters.m_star, -24.468, rtol=0.0, atol=1e-6)
    assert not np.isclose(
        parameters.beta_faint,
        -1.6400348982117574,
        rtol=0.0,
        atol=1e-6,
    )

    final_grid = build_kulkarni2019_type1_model2(
        target_magnitude_grid=np.array([-25.0]),
        redshift_grid=np.array([3.16]),
        target_cosmology=KULKARNI2019_SOURCE_COSMOLOGY,
    )
    stale_preprint_log10_phi = -6.581761400921176
    np.testing.assert_allclose(
        final_grid.phi_log10[0, 0],
        -6.435558316423153,
        rtol=0.0,
        atol=2e-13,
    )
    assert not np.isclose(
        final_grid.phi_log10[0, 0],
        stale_preprint_log10_phi,
        rtol=0.0,
        atol=1e-8,
    )


def test_kulkarni2019_cosmology_remap_and_stable_extreme_magnitudes():
    redshift = 1.0
    target_cosmology = FlatLambdaCDM(H0=67.9, Om0=0.3065)
    delta_distance_modulus = float(
        target_cosmology.distmod(redshift).value
        - KULKARNI2019_SOURCE_COSMOLOGY.distmod(redshift).value
    )
    log10_volume_ratio = float(
        np.log10(
            KULKARNI2019_SOURCE_COSMOLOGY.differential_comoving_volume(
                redshift
            ).value
            / target_cosmology.differential_comoving_volume(redshift).value
        )
    )
    np.testing.assert_allclose(
        delta_distance_modulus,
        0.057813862591253,
        rtol=0.0,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        log10_volume_ratio,
        -0.0331898129268792,
        rtol=0.0,
        atol=2e-13,
    )

    parameters = kulkarni2019_model2_evolving_parameters(redshift)
    target_break = parameters.m_star - delta_distance_modulus
    grid = build_kulkarni2019_type1_model2(
        target_magnitude_grid=np.array([target_break, -1.0e6, 1.0e6]),
        redshift_grid=np.array([redshift]),
        target_cosmology=target_cosmology,
    )
    np.testing.assert_allclose(
        grid.phi_log10[0, 0],
        parameters.log10_phi_star - LOG10_TWO + log10_volume_ratio,
        rtol=0.0,
        atol=2e-12,
    )
    assert np.all(np.isfinite(grid.phi_log10))


def test_empirical_outputs_are_galactic_corrected_with_internal_dust_implicit():
    cases = (
        (
            build_wang2026_type1_lade_a,
            WANG2026_SOURCE_COSMOLOGY,
            "corrected_in_inherited_k19_photometry",
        ),
        (
            build_palanque2016_ple_lede,
            PALANQUE2016_SOURCE_COSMOLOGY,
            "corrected_in_g_dered_using_schlegel1998",
        ),
        *(
            (
                builder,
                KULKARNI2019_SOURCE_COSMOLOGY,
                "corrected_in_input_psf_photometry_using_schlegel1998",
            )
            for builder in (
                build_kulkarni2019_type1_model1,
                build_kulkarni2019_type1_model2,
                build_kulkarni2019_type1_model3,
            )
        ),
    )
    forbidden_knob_fragments = (
        "attenuat",
        "redden",
        "extinct",
        "dust",
        "ebv",
        "column_density",
        "n_h",
    )

    for builder, source_cosmology, galactic_extinction in cases:
        grid = builder(
            target_magnitude_grid=np.array([-24.0]),
            redshift_grid=np.array([2.2]),
            target_cosmology=source_cosmology,
        )
        semantics = grid.reddening_semantics

        assert semantics.luminosity_semantics.startswith("empirical_")
        assert semantics.galactic_extinction == galactic_extinction
        assert (
            semantics.internal_extinction
            == "not_corrected_or_explicitly_modeled"
        )
        assert semantics.apply_additional_internal_extinction is False

        parameter_names = tuple(inspect.signature(builder).parameters)
        assert not any(
            fragment in parameter_name.lower()
            for parameter_name in parameter_names
            for fragment in forbidden_knob_fragments
        )
