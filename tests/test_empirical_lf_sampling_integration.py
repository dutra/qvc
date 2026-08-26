import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qvc-empirical-lf-test-matplotlib")

from qvc.hubble import completeness_mock_catalog as mock_catalog
from qvc.hubble.completeness_mock_catalog import (
    COMPLETENESS_MOCK_SCHEMA_VERSION,
    COMPLETENESS_MOCK_SEMANTICS_VERSION,
    COSMO,
    EMPIRICAL_LF_NATIVE_MAGNITUDE_GRID,
    KULKARNI2019_REFERENCE_WAVELENGTH_ANGSTROM,
    KULKARNI2019_TYPE1_MODEL1,
    KULKARNI2019_TYPE1_MODEL2,
    KULKARNI2019_TYPE1_MODEL3,
    LF_CONVERSION_SLOPE_CONVENTION,
    LF_CONVERSION_SLOPE_PARAMETER,
    PALANQUE2016_NATIVE_TO_MONOCHROMATIC_AB_OFFSET,
    PALANQUE2016_PLE_LEDE,
    PALANQUE2016_REFERENCE_WAVELENGTH_ANGSTROM,
    WANG2026_REFERENCE_WAVELENGTH_ANGSTROM,
    WANG2026_TYPE1_LADE_A,
    build_completeness_lf,
    completeness_lf_magnitude_state_match,
    completeness_lf_static_metadata,
    mock_m_per_zbin,
    native_absolute_magnitude_to_m2500,
    sample_alpha_nu_lf_conversion_conditional_on_m2500_support,
    save_mock_catalog,
)
from qvc.hubble.empirical_luminosity_functions import build_empirical_lf


EMPIRICAL_CASES = (
    (WANG2026_TYPE1_LADE_A, WANG2026_REFERENCE_WAVELENGTH_ANGSTROM, 0.0),
    (
        PALANQUE2016_PLE_LEDE,
        PALANQUE2016_REFERENCE_WAVELENGTH_ANGSTROM,
        PALANQUE2016_NATIVE_TO_MONOCHROMATIC_AB_OFFSET,
    ),
    (KULKARNI2019_TYPE1_MODEL1, KULKARNI2019_REFERENCE_WAVELENGTH_ANGSTROM, 0.0),
    (KULKARNI2019_TYPE1_MODEL2, KULKARNI2019_REFERENCE_WAVELENGTH_ANGSTROM, 0.0),
    (KULKARNI2019_TYPE1_MODEL3, KULKARNI2019_REFERENCE_WAVELENGTH_ANGSTROM, 0.0),
)
KULKARNI_CASES = (
    (
        KULKARNI2019_TYPE1_MODEL1,
        1,
        "kulkarni2019_model1_eq7_eq13_eq16_to_eq18_table3_"
        "published_rounded_medians",
        "broken_power_law_in_1_plus_z",
        True,
    ),
    (
        KULKARNI2019_TYPE1_MODEL2,
        2,
        "kulkarni2019_model2_eq7_eq13_eq16_to_eq18_table3_"
        "published_rounded_medians",
        "broken_power_law_in_1_plus_z",
        False,
    ),
    (
        KULKARNI2019_TYPE1_MODEL3,
        3,
        "kulkarni2019_model3_eq7_eq13_eq16_table3_"
        "published_rounded_medians",
        "linear_chebyshev_in_1_plus_z",
        False,
    ),
)
M2500_SUPPORT = (18.5, 24.0)


def test_standalone_cli_names_the_population_lf_conversion_slope(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "completeness_mock_catalog.py",
            "--lf-model",
            WANG2026_TYPE1_LADE_A,
            "--output",
            str(tmp_path / "mock.h5"),
            "--alpha-nu-lf-conversion",
            "-0.65",
            "--dalpha-nu-lf-conversion",
            "0.22",
        ],
    )

    args = mock_catalog.parse_args()

    assert args.alpha_nu_lf_conversion == pytest.approx(-0.65)
    assert args.dalpha_nu_lf_conversion == pytest.approx(0.22)
    assert not hasattr(args, "alpha_nu")
    assert not hasattr(args, "dalpha_nu")


def test_standalone_cli_rejects_the_ambiguous_legacy_slope_flag(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "completeness_mock_catalog.py",
            "--lf-model",
            WANG2026_TYPE1_LADE_A,
            "--output",
            str(tmp_path / "mock.h5"),
            "--alpha-nu",
            "-0.65",
        ],
    )

    with pytest.raises(SystemExit):
        mock_catalog.parse_args()


@pytest.mark.parametrize(
    (
        "reference_wavelength_angstrom",
        "native_to_monochromatic_ab_offset",
        "expected_shift",
    ),
    (
        (WANG2026_REFERENCE_WAVELENGTH_ANGSTROM, 0.0, -0.2957150080463284),
        (
            KULKARNI2019_REFERENCE_WAVELENGTH_ANGSTROM,
            0.0,
            -0.2957150080463284,
        ),
        (
            PALANQUE2016_REFERENCE_WAVELENGTH_ANGSTROM,
            PALANQUE2016_NATIVE_TO_MONOCHROMATIC_AB_OFFSET,
            0.9356226582671712,
        ),
    ),
)
def test_native_to_m2500_wavelength_shift_gold_values(
    reference_wavelength_angstrom,
    native_to_monochromatic_ab_offset,
    expected_shift,
):
    native_magnitude = np.array([-26.0, -24.0])
    converted = native_absolute_magnitude_to_m2500(
        native_magnitude,
        alpha_nu_lf_conversion=-0.5,
        reference_wavelength_angstrom=reference_wavelength_angstrom,
        native_to_monochromatic_ab_offset=(
            native_to_monochromatic_ab_offset
        ),
    )

    np.testing.assert_allclose(
        converted - native_magnitude,
        expected_shift,
        rtol=0.0,
        atol=2e-14,
    )


@pytest.mark.parametrize(
    ("reference_wavelength_angstrom", "native_to_monochromatic_ab_offset"),
    tuple((case[1], case[2]) for case in EMPIRICAL_CASES),
)
def test_truncated_gaussian_slopes_are_finite_and_exactly_supported(
    reference_wavelength_angstrom,
    native_to_monochromatic_ab_offset,
):
    rng = np.random.default_rng(20260824)
    sample_size = 4096
    native_magnitude = np.full(sample_size, -23.0)
    # Put half the unconverted rows just brightward and half just faintward of
    # the allowed interval so conditioning genuinely truncates both tails.
    unconverted_apparent = np.where(
        np.arange(sample_size) % 2 == 0,
        M2500_SUPPORT[0] - 0.1,
        M2500_SUPPORT[1] + 0.1,
    )
    distance_modulus = unconverted_apparent - native_magnitude

    alpha_nu_lf_conversion = (
        sample_alpha_nu_lf_conversion_conditional_on_m2500_support(
            rng,
            native_magnitude,
            distance_modulus,
            reference_wavelength_angstrom=reference_wavelength_angstrom,
            m2500_support=M2500_SUPPORT,
            alpha_nu_lf_conversion_mean=-0.5,
            alpha_nu_lf_conversion_sigma=0.3,
            native_to_monochromatic_ab_offset=(
                native_to_monochromatic_ab_offset
            ),
        )
    )
    apparent_m2500 = (
        native_absolute_magnitude_to_m2500(
            native_magnitude,
            alpha_nu_lf_conversion,
            reference_wavelength_angstrom,
            native_to_monochromatic_ab_offset,
        )
        + distance_modulus
    )

    assert np.all(np.isfinite(alpha_nu_lf_conversion))
    assert np.all(np.isfinite(apparent_m2500))
    assert np.all(apparent_m2500 >= M2500_SUPPORT[0])
    assert np.all(apparent_m2500 <= M2500_SUPPORT[1])

    # A zero-width slope distribution placed exactly on either endpoint must
    # also be accepted: the declared support is inclusive, not open.
    coefficient = 2.5 * np.log10(
        2500.0 / reference_wavelength_angstrom
    )
    for endpoint in M2500_SUPPORT:
        unconverted = 21.0
        endpoint_alpha = (
            endpoint
            - unconverted
            - native_to_monochromatic_ab_offset
        ) / coefficient
        fixed_alpha = (
            sample_alpha_nu_lf_conversion_conditional_on_m2500_support(
                rng,
                native_magnitude=np.array([-23.0]),
                distance_modulus=np.array([unconverted + 23.0]),
                reference_wavelength_angstrom=reference_wavelength_angstrom,
                m2500_support=M2500_SUPPORT,
                alpha_nu_lf_conversion_mean=endpoint_alpha,
                alpha_nu_lf_conversion_sigma=0.0,
                native_to_monochromatic_ab_offset=(
                    native_to_monochromatic_ab_offset
                ),
            )
        )
        fixed_m2500 = (
            native_absolute_magnitude_to_m2500(
                -23.0,
                fixed_alpha,
                reference_wavelength_angstrom,
                native_to_monochromatic_ab_offset,
            )
            + unconverted
            + 23.0
        )
        np.testing.assert_allclose(
            fixed_m2500,
            endpoint,
            rtol=0.0,
            atol=2e-13,
        )


def test_truncated_slope_draws_match_the_conditional_gaussian_mean():
    rng = np.random.default_rng(90210)
    sample_size = 80_000
    parent_mean = -0.5
    parent_sigma = 0.3
    lower_alpha = -0.4
    coefficient = 2.5 * np.log10(
        2500.0 / WANG2026_REFERENCE_WAVELENGTH_ANGSTROM
    )
    base_apparent = M2500_SUPPORT[0] - coefficient * lower_alpha
    native_magnitude = np.full(sample_size, -23.0)
    distance_modulus = np.full(sample_size, base_apparent + 23.0)

    draws = sample_alpha_nu_lf_conversion_conditional_on_m2500_support(
        rng,
        native_magnitude,
        distance_modulus,
        reference_wavelength_angstrom=WANG2026_REFERENCE_WAVELENGTH_ANGSTROM,
        m2500_support=M2500_SUPPORT,
        alpha_nu_lf_conversion_mean=parent_mean,
        alpha_nu_lf_conversion_sigma=parent_sigma,
    )
    standardized_lower = (lower_alpha - parent_mean) / parent_sigma
    standard_normal_density = np.exp(-0.5 * standardized_lower**2) / np.sqrt(
        2.0 * np.pi
    )
    expected_mean = parent_mean + parent_sigma * standard_normal_density / (
        1.0 - ndtr(standardized_lower)
    )

    assert np.all(draws >= lower_alpha - 2e-14)
    np.testing.assert_allclose(
        np.mean(draws),
        expected_mean,
        rtol=0.0,
        atol=2.5e-3,
    )


@pytest.mark.parametrize(
    (
        "lf_model",
        "reference_wavelength_angstrom",
        "native_to_monochromatic_ab_offset",
    ),
    EMPIRICAL_CASES,
)
def test_empirical_lf_build_and_sampling_never_use_shen_weighting(
    monkeypatch,
    lf_model,
    reference_wavelength_angstrom,
    native_to_monochromatic_ab_offset,
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Empirical LF path called Shen population/dust code")

    for name in (
        "build_shen_lf",
        "shen_type1_fraction",
        "_shen_type1_nh_bin_fractions",
        "_build_shen_type1_lf_at_redshift",
        "_load_shen_c_backend",
    ):
        monkeypatch.setattr(mock_catalog, name, fail_if_called)

    grid = build_completeness_lf(
        lf_model,
        z_range=(0.7, 0.8),
        target_cosmology=COSMO,
    )
    direct_grid = build_empirical_lf(
        lf_model,
        EMPIRICAL_LF_NATIVE_MAGNITUDE_GRID,
        grid.redshift_grid,
        COSMO,
    )
    metadata = completeness_lf_static_metadata(lf_model)
    np.testing.assert_array_equal(grid.phi_log10, direct_grid.phi_log10)
    assert grid.formula_version == metadata["formula_version"]
    assert grid.native_magnitude_name == metadata["native_magnitude_name"]
    assert grid.calibration_redshift_range == tuple(
        metadata["calibration_redshift_range"]
    )
    assert grid.reference_wavelength_angstrom == pytest.approx(
        metadata["reference_wavelength_angstrom"]
    )
    assert grid.native_to_monochromatic_ab_offset == pytest.approx(
        metadata["native_to_monochromatic_ab_offset"]
    )
    assert grid.reddening_semantics.apply_additional_internal_extinction is False

    result = mock_m_per_zbin(
        grid.phi_log10,
        grid.native_magnitude_grid,
        grid.redshift_grid,
        area_deg2=5.0,
        alpha_nu_lf_conversion=-0.5,
        dalpha_nu_lf_conversion=0.3,
        cosmo=COSMO,
        z_res=32,
        kcorr_zref=2.0,
        reference_wavelength_angstrom=reference_wavelength_angstrom,
        native_to_monochromatic_ab_offset=(
            native_to_monochromatic_ab_offset
        ),
        m2500_support=M2500_SUPPORT,
        z_range=(0.7, 0.8),
        rng=np.random.default_rng(12345),
        return_z=True,
        return_global=True,
        return_alpha_nu_lf_conversion=True,
    )
    z_all, apparent_i, apparent_m2500, alpha_nu_lf_conversion = (
        result[4],
        result[5],
        result[6],
        result[8],
    )

    assert z_all.size > 0
    assert (
        z_all.shape
        == apparent_i.shape
        == apparent_m2500.shape
        == alpha_nu_lf_conversion.shape
    )
    assert np.all(np.isfinite(z_all))
    assert np.all(np.isfinite(apparent_i))
    assert np.all(np.isfinite(apparent_m2500))
    assert np.all(np.isfinite(alpha_nu_lf_conversion))
    assert np.all(apparent_m2500 >= M2500_SUPPORT[0])
    assert np.all(apparent_m2500 <= M2500_SUPPORT[1])
    expected_count = float(np.sum(result[1]))
    assert int(np.sum(result[3])) == z_all.size
    assert abs(z_all.size - expected_count) < 8.0 * np.sqrt(expected_count)


@pytest.mark.parametrize(
    (
        "lf_model",
        "reference_wavelength_angstrom",
        "native_to_monochromatic_ab_offset",
    ),
    EMPIRICAL_CASES,
)
def test_magnitude_state_mismatch_and_semantics_are_persisted(
    tmp_path,
    lf_model,
    reference_wavelength_angstrom,
    native_to_monochromatic_ab_offset,
):
    metadata = completeness_lf_static_metadata(lf_model)
    is_match, expected_state = completeness_lf_magnitude_state_match(
        lf_model,
        "dereddened",
    )
    semantics = metadata["semantics"]

    assert is_match is False
    assert expected_state == "attenuated"
    assert semantics["expected_completeness_magnitude"] == "attenuated"
    assert semantics["internal_dust_treatment"] == "implicit_no_reapplication"
    assert semantics["conversion_is_approximate"] is True
    assert (
        semantics["lf_conversion_slope_parameter"]
        == "alpha_nu_lf_conversion"
    )
    assert semantics["lf_conversion_continuum_state"] == (
        "attenuation_retaining_empirical_lf_continuum_proxy"
    )
    assert semantics["lf_conversion_dust_operation"] == (
        "none_preserve_lf_population_attenuation_state"
    )
    assert semantics["lf_conversion_internal_dust_correction_applied"] is False
    assert semantics["lf_conversion_is_jaxsedfit_intrinsic_slope"] is False

    output_path = tmp_path / f"{lf_model}.h5"
    save_mock_catalog(
        output_path,
        z_all=np.array([0.44, 3.16]),
        m_all=np.array([19.0, 23.5]),
        m_2500_all=np.array(M2500_SUPPORT),
        alpha_nu_lf_conversion_all=np.array([-0.5, -0.3]),
        alpha_nu_lf_conversion_parent_mean=-0.5,
        alpha_nu_lf_conversion_parent_sigma=0.3,
        lf_model=lf_model,
        lf_metadata=metadata,
        reference_wavelength_angstrom=reference_wavelength_angstrom,
        native_to_monochromatic_ab_offset=(
            native_to_monochromatic_ab_offset
        ),
        m2500_support=M2500_SUPPORT,
        z_range=(0.44, 3.16),
        completeness_magnitude_state="dereddened",
        lf_magnitude_state_match=is_match,
    )

    with h5py.File(output_path, "r") as handle:
        saved_metadata = json.loads(handle.attrs["lf_metadata_json"])
        np.testing.assert_array_equal(
            handle["apparent_mag_2500"][:],
            M2500_SUPPORT,
        )
        assert np.all(np.isfinite(handle["apparent_mag_2500"][:]))
        np.testing.assert_allclose(
            handle["alpha_nu_lf_conversion"][:],
            [-0.5, -0.3],
        )
        assert "alpha_lambda_lf_conversion" not in handle
        assert "alpha_lambda" not in handle
        assert "alpha_nu" not in handle
        assert handle.attrs["lf_model"] == lf_model
        assert "shen_lf_mode" not in handle.attrs
        assert handle.attrs["completeness_magnitude_state"] == "dereddened"
        assert bool(handle.attrs["lf_magnitude_state_match"]) is False
        assert (
            handle.attrs["lf_internal_dust_treatment"]
            == "implicit_no_reapplication"
        )
        assert bool(handle.attrs["lf_conversion_is_approximate"]) is True
        assert (
            int(handle.attrs["completeness_mock_schema_version"])
            == COMPLETENESS_MOCK_SCHEMA_VERSION
            == 4
        )
        assert (
            handle.attrs["lf_semantics_version"]
            == COMPLETENESS_MOCK_SEMANTICS_VERSION
        )
        assert (
            handle.attrs["lf_conversion_slope_parameter"]
            == LF_CONVERSION_SLOPE_PARAMETER
        )
        assert (
            handle.attrs["lf_conversion_slope_convention"]
            == LF_CONVERSION_SLOPE_CONVENTION
        )
        assert handle.attrs["lf_conversion_continuum_state"] == (
            "attenuation_retaining_empirical_lf_continuum_proxy"
        )
        assert handle.attrs["lf_conversion_dust_operation"] == (
            "none_preserve_lf_population_attenuation_state"
        )
        assert bool(
            handle.attrs["lf_conversion_internal_dust_correction_applied"]
        ) is False
        assert bool(
            handle.attrs["lf_conversion_is_jaxsedfit_intrinsic_slope"]
        ) is False
        assert handle.attrs[
            "alpha_nu_lf_conversion_parent_mean"
        ] == pytest.approx(-0.5)
        assert handle.attrs[
            "alpha_nu_lf_conversion_parent_sigma"
        ] == pytest.approx(0.3)
        assert "alpha_nu_parent_mean" not in handle.attrs
        assert "alpha_nu_parent_sigma" not in handle.attrs
        assert handle.attrs["lf_native_to_monochromatic_ab_offset"] == pytest.approx(
            native_to_monochromatic_ab_offset
        )
        assert (
            saved_metadata["semantics"]["expected_completeness_magnitude"]
            == "attenuated"
        )


@pytest.mark.parametrize(
    (
        "lf_model",
        "model_number",
        "formula_version",
        "beta_evolution",
        "approximate_samples_included",
    ),
    KULKARNI_CASES,
)
def test_kulkarni_static_metadata_captures_sample_and_magnitude_semantics(
    lf_model,
    model_number,
    formula_version,
    beta_evolution,
    approximate_samples_included,
):
    metadata = completeness_lf_static_metadata(lf_model)
    semantics = metadata["semantics"]
    sample = metadata["sample_provenance"]
    source = metadata["source_provenance"]

    assert metadata["model_id"] == lf_model
    assert metadata["native_magnitude_name"] == "M_1450_AB"
    assert metadata["reference_wavelength_angstrom"] == pytest.approx(1450.0)
    assert metadata["native_to_monochromatic_ab_offset"] == 0.0
    assert metadata["calibration_redshift_range"] == [0.6, 6.5]
    assert metadata["formula_version"] == formula_version
    assert metadata["model_provenance"] == {
        "model_number": model_number,
        "beta_faint_evolution": beta_evolution,
    }

    assert source["parameter_summary"] == "published_rounded_posterior_medians"
    assert source["publication_version"] == "final_published_2019_mnras_table3"
    assert source["parameter_covariance_propagated"] is False
    assert source["reference_repository_revision"].startswith("77c2a80")
    assert source["reference_repository_tag_v3_used"] is False

    assert sample["boss_dr9_excluded_redshift_interval"] == [2.2, 3.5]
    assert (
        sample["approximate_selection_samples_included"]
        is approximate_samples_included
    )
    included = sample["included_approximate_selection_sample_ids"]
    excluded = sample["excluded_approximate_selection_sample_ids"]
    if approximate_samples_included:
        assert included == [17, 18, 19, 20]
        assert excluded == []
    else:
        assert included == []
        assert excluded == [17, 18, 19, 20]

    assert semantics["population_scope"] == "observational_uv_optical_type1"
    assert semantics["uv_attenuation_state"] == "empirical_rest_M1450"
    assert semantics["internal_dust_treatment"] == "implicit_no_reapplication"
    assert semantics["object_level_internal_dust_correction_applied"] is False
    assert semantics["intrinsic_agn_spectrum_phrase_interpretation"].endswith(
        "not_a_dust_free_lf"
    )
    assert "sdss_and_2slaq" in semantics["host_contamination_treatment"]
    assert semantics["coordinate_conversion"].endswith("M1450_to_M2500_proxy")
    assert semantics["expected_completeness_magnitude"] == "attenuated"
    assert semantics["heavily_reddened_type1_recovery"] == "not_demonstrated"
    assert semantics["bal_treatment"] == "not_separately_modeled_or_excluded"

    # Metadata must remain directly usable by the canonical cache JSON hash.
    json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def test_kulkarni_model1_metadata_records_the_paper_identified_feature():
    metadata = completeness_lf_static_metadata(KULKARNI2019_TYPE1_MODEL1)

    assert metadata["parameters"]["beta_faint_break_redshift"] == 3.773
    caveats = " ".join(metadata["caveats"]["model_specific"])
    assert "probably_unphysical" in caveats
    assert "z_3p773" in caveats
    assert "default_qvc_zmax_3p16_ends_before" in caveats
    assert "discontinuity" in caveats


def test_palanque_metadata_is_explicit_and_unknown_fallback_is_impossible(
    monkeypatch,
):
    metadata = completeness_lf_static_metadata(PALANQUE2016_PLE_LEDE)
    assert metadata["model_id"] == PALANQUE2016_PLE_LEDE
    assert metadata["native_magnitude_name"] == "M_g(z=2)_AB"

    registered_without_metadata = "registered_without_static_metadata"
    monkeypatch.setattr(
        mock_catalog,
        "COMPLETENESS_LF_MODELS",
        (*mock_catalog.COMPLETENESS_LF_MODELS, registered_without_metadata),
    )
    with pytest.raises(RuntimeError, match="no explicit static metadata"):
        completeness_lf_static_metadata(registered_without_metadata)


def test_kulkarni_warnings_cover_low_z_boss_gap_and_model1_feature(capsys):
    build_completeness_lf(
        KULKARNI2019_TYPE1_MODEL1,
        z_range=(0.44, 3.9),
        target_cosmology=COSMO,
    )

    warning = capsys.readouterr().out
    assert "excluded z<0.6" in warning
    assert "excluded the BOSS DR9 sample" in warning
    assert "smooth global-model interpolation without credible fit data" in warning
    assert "probably unphysical sharp faint-end-slope transition" in warning
    assert "centered near z=3.773" in warning


def test_kulkarni_model2_boss_warning_does_not_claim_model1_feature(capsys):
    build_completeness_lf(
        KULKARNI2019_TYPE1_MODEL2,
        z_range=(0.7, 3.16),
        target_cosmology=COSMO,
    )

    warning = capsys.readouterr().out
    assert "excluded the BOSS DR9 sample" in warning
    assert "sharp faint-end-slope transition" not in warning
