import argparse
import ctypes
import gzip
import importlib
import io
import json
import os
import sys
import tempfile
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr, ndtri, ndtri_exp
from tqdm.auto import tqdm

from qvc.hubble.empirical_luminosity_functions import (
    EMPIRICAL_LF_MODEL_IDS,
    KULKARNI2019_REFERENCE_WAVELENGTH_ANGSTROM,
    KULKARNI2019_SOURCE_COSMOLOGY,
    KULKARNI2019_TYPE1_MODEL1,
    KULKARNI2019_TYPE1_MODEL2,
    KULKARNI2019_TYPE1_MODEL3,
    KULKARNI2019_TYPE1_MODEL_IDS,
    LFGrid,
    PALANQUE2016_NATIVE_TO_MONOCHROMATIC_AB_OFFSET,
    PALANQUE2016_PLE_LEDE,
    PALANQUE2016_REFERENCE_WAVELENGTH_ANGSTROM,
    PALANQUE2016_SOURCE_COSMOLOGY,
    ReddeningSemantics,
    WANG2026_REFERENCE_WAVELENGTH_ANGSTROM,
    WANG2026_SOURCE_COSMOLOGY,
    WANG2026_TYPE1_LADE_A,
    build_empirical_lf,
)


COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)
L_SUN_ERG_S = 3.828e33
L0 = 1e10 * L_SUN_ERG_S
LOG10_MAG_JACOBIAN = np.log10(0.4)
NU_2500_HZ = 2.99792458e18 / 2500.0
AB_ABSOLUTE_MAG_ZEROPOINT = 51.59477721004232
SHEN_GLOBAL_FIT = "A"
SHEN_DEFAULT_LF_MODE = "all_nh_attenuated"
SHEN_LF_MODES = (
    SHEN_DEFAULT_LF_MODE,
    "type1_intrinsic",
    "type1_attenuated",
)
COMPLETENESS_LF_MODELS = ("shen", *EMPIRICAL_LF_MODEL_IDS)
SHEN_L_SUN_ERG_S = 3.9e33
SHEN_TYPE1_NH_MIN = 20.0
SHEN_TYPE1_NH_SPLIT = 21.0
SHEN_TYPE1_NH_MAX = 22.0
SHEN_TYPE1_NH_SAMPLES_PER_DEX = 64
FULL_SKY_AREA_DEG2 = float(4.0 * np.pi * (180.0 / np.pi) ** 2)
COMPLETENESS_MOCK_SCHEMA_VERSION = 4
COMPLETENESS_MOCK_SEMANTICS_VERSION = "lf_semantics_v2_lf_conversion_slope"
LF_CONVERSION_SLOPE_PARAMETER = "alpha_nu_lf_conversion"
LF_CONVERSION_SLOPE_CONVENTION = "f_nu_proportional_to_nu_power_alpha_nu"
DEFAULT_M2500_SUPPORT = (18.5, 24.0)
EMPIRICAL_LF_NATIVE_MAGNITUDE_GRID = np.linspace(-33.0, -16.0, 341)
EMPIRICAL_LF_REDSHIFT_STEP = 0.05
MAGNITUDE_SAMPLE_CHUNK_ROWS = 4096
KULKARNI2019_SOURCE_REVISION = (
    "77c2a80da35458bc461424cce13946211d3718be"
)
KULKARNI2019_BOSS_EXCLUDED_REDSHIFT_INTERVAL = (2.2, 3.5)
KULKARNI2019_MODEL1_FEATURE_REDSHIFT_INTERVAL = (3.5, 4.0)


def _lf_conversion_slope_semantics(continuum_state):
    """Describe the population-level slope used only for LF coordinates.

    This proxy transports an LF from its published reference wavelength to
    2500 Angstrom and supplies the mock K-correction.  It is deliberately
    distinct from any intrinsic accretion-disk slope inferred by JAXSedFit.
    No object-level internal-dust correction is performed by this conversion.
    """

    return {
        "lf_conversion_slope_parameter": LF_CONVERSION_SLOPE_PARAMETER,
        "lf_conversion_slope_convention": LF_CONVERSION_SLOPE_CONVENTION,
        "lf_conversion_continuum_state": str(continuum_state),
        "lf_conversion_dust_operation": (
            "none_preserve_lf_population_attenuation_state"
        ),
        "lf_conversion_internal_dust_correction_applied": False,
        "lf_conversion_is_jaxsedfit_intrinsic_slope": False,
    }


def _kulkarni2019_static_metadata(
    lf_model,
    *,
    model_number,
    parameters,
    formula_version,
    equations,
    beta_evolution,
    include_approximate_selection_samples,
    model_specific_caveats,
):
    """Return shared and model-specific Kulkarni et al. provenance."""

    approximate_sample_ids = [17, 18, 19, 20]
    return {
        "model_id": lf_model,
        "native_magnitude_name": "M_1450_AB",
        "reference_wavelength_angstrom": (
            KULKARNI2019_REFERENCE_WAVELENGTH_ANGSTROM
        ),
        "native_to_monochromatic_ab_offset": 0.0,
        "source_cosmology": {
            "H0": float(KULKARNI2019_SOURCE_COSMOLOGY.H0.value),
            "Om0": float(KULKARNI2019_SOURCE_COSMOLOGY.Om0),
        },
        # The analytic implementation deliberately uses the conservative
        # range supported by the homogenized global-fit sample.  Model 1 also
        # included three z>6.5 objects with approximate selection functions;
        # they are recorded below rather than promoted to calibrated support.
        "calibration_redshift_range": [0.6, 6.5],
        "formula_version": formula_version,
        "equations": list(equations),
        "parameters": parameters,
        "source_provenance": {
            "paper": "Kulkarni_et_al_2019_AGN_UV_luminosity_function",
            "publication_version": "final_published_2019_mnras_table3",
            "parameter_table": 3,
            "parameter_summary": "published_rounded_posterior_medians",
            "parameter_covariance_propagated": False,
            "reference_repository_revision": KULKARNI2019_SOURCE_REVISION,
            "reference_repository_tag_v3_used": False,
            "reference_repository_tag_v3_note": (
                "v3.0 predates the final published parameter table and has "
                "materially different coefficients"
            ),
        },
        "sample_provenance": {
            "population": "heterogeneous_uv_optical_type1_quasar_compilation",
            "global_fit_minimum_redshift": 0.6,
            "boss_dr9_excluded_redshift_interval": list(
                KULKARNI2019_BOSS_EXCLUDED_REDSHIFT_INTERVAL
            ),
            "boss_dr9_exclusion_reason": (
                "selection_function_systematics; global LF is a smooth "
                "interpolation across this interval"
            ),
            "approximate_selection_sample_ids": approximate_sample_ids,
            "approximate_selection_samples_included": bool(
                include_approximate_selection_samples
            ),
            "included_approximate_selection_sample_ids": (
                approximate_sample_ids
                if include_approximate_selection_samples
                else []
            ),
            "excluded_approximate_selection_sample_ids": (
                []
                if include_approximate_selection_samples
                else approximate_sample_ids
            ),
            "high_redshift_z_gt_6p5_treatment": (
                "three_approximate_selection_constraints_included_but_"
                "outside_conservative_calibration_range"
                if include_approximate_selection_samples
                else "approximate_selection_constraints_excluded"
            ),
        },
        "model_provenance": {
            "model_number": int(model_number),
            "beta_faint_evolution": beta_evolution,
        },
        "caveats": {
            "low_redshift_extrapolation_below": 0.6,
            "low_redshift_exclusion_reason": (
                "residual_host_galaxy_correction_and_extended_source_"
                "selection_systematics"
            ),
            "interpolation_without_credible_fit_data": list(
                KULKARNI2019_BOSS_EXCLUDED_REDSHIFT_INTERVAL
            ),
            "model_specific": list(model_specific_caveats),
        },
        "semantics": {
            **_lf_conversion_slope_semantics(
                "attenuation_retaining_empirical_lf_continuum_proxy"
            ),
            "population_scope": "observational_uv_optical_type1",
            "type1_definition": (
                "heterogeneous_uv_optical_color_selected_quasar_compilation"
            ),
            "uv_attenuation_state": "empirical_rest_M1450",
            "galactic_foreground_treatment": (
                "corrected_in_input_psf_photometry_using_schlegel1998_"
                "unless_noted_by_source_survey"
            ),
            "internal_dust_treatment": "implicit_no_reapplication",
            "object_level_internal_dust_correction_applied": False,
            "intrinsic_agn_spectrum_phrase_interpretation": (
                "k_correction_bandpass_template_not_a_dust_free_lf"
            ),
            "host_contamination_treatment": (
                "sdss_and_2slaq_z_lt_2p2_corrected_following_croom2009; "
                "z_lt_0p6_excluded_from_global_fit"
            ),
            "selection_function_treatment": (
                "survey_specific_photometric_imaging_and_spectroscopic_"
                "selection_functions_included_in_lf_fit"
            ),
            "coordinate_conversion": "drawn_power_law_M1450_to_M2500_proxy",
            "conversion_is_approximate": True,
            "expected_completeness_magnitude": "attenuated",
            "heavily_reddened_type1_recovery": "not_demonstrated",
            "bal_treatment": "not_separately_modeled_or_excluded",
        },
    }


def plan_area_scaled_mock_sampling(
    target_area_deg2,
    *,
    proposal_area_deg2=FULL_SKY_AREA_DEG2,
    oversample=4.0,
):
    """Return a uniformly thinned proposal-area sampling plan."""

    target_area_deg2 = float(target_area_deg2)
    proposal_area_deg2 = float(proposal_area_deg2)
    oversample = float(oversample)
    if not np.isfinite(target_area_deg2) or target_area_deg2 <= 0.0:
        raise ValueError("target_area_deg2 must be finite and positive.")
    if not np.isfinite(proposal_area_deg2) or proposal_area_deg2 <= 0.0:
        raise ValueError("proposal_area_deg2 must be finite and positive.")
    if proposal_area_deg2 < target_area_deg2:
        raise ValueError("proposal_area_deg2 cannot be smaller than target_area_deg2.")
    if not np.isfinite(oversample) or oversample < 1.0:
        raise ValueError("oversample must be finite and at least one.")

    effective_sampled_area_deg2 = min(
        proposal_area_deg2,
        oversample * target_area_deg2,
    )
    thinning_probability = effective_sampled_area_deg2 / proposal_area_deg2
    mock_count_scale = target_area_deg2 / effective_sampled_area_deg2
    return {
        "target_area_deg2": target_area_deg2,
        "proposal_area_deg2": proposal_area_deg2,
        "effective_sampled_area_deg2": effective_sampled_area_deg2,
        "thinning_probability": thinning_probability,
        "mock_count_scale": mock_count_scale,
        "requested_oversample": oversample,
        "realized_oversample": effective_sampled_area_deg2 / target_area_deg2,
    }


def log_nu_lnu_to_ab_absolute_magnitude(log_nu_lnu, frequency_hz):
    """Convert log10(nu L_nu / erg s^-1) to monochromatic absolute AB magnitude."""
    log_lnu = np.asarray(log_nu_lnu, dtype=float) - np.log10(float(frequency_hz))
    return AB_ABSOLUTE_MAG_ZEROPOINT - 2.5 * log_lnu


def _candidate_existing_path(*candidates):
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.exists():
            return path
    return None


def _discover_qvc_root() -> Path:
    """Find the repository root by walking upward to the pyproject file."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]


def default_shen_pubtools_path():
    qvc_root = _discover_qvc_root()
    return _candidate_existing_path(
        os.environ.get("SHEN_PUBTOOLS_PATH"),
        os.environ.get("HOPKINS_PUBTOOLS_PATH"),
        qvc_root / "quasarlf" / "pubtools",
        qvc_root.parent / "quasarlf" / "pubtools",
        qvc_root.parent / "quasarlf" / "pubtools",
        qvc_root / "external" / "quasarlf" / "pubtools",
    )


def default_ananna_xlf_path():
    qvc_root = _discover_qvc_root()
    return _candidate_existing_path(
        os.environ.get("ANANNA_XLF_PATH"),
        qvc_root / "ananna_xlf" / "final_sol_all.npy.gz",
        qvc_root.parent / "ananna_xlf" / "final_sol_all.npy.gz",
        Path.home() / "ananna_xlf" / "final_sol_all.npy.gz",
    )


@contextmanager
def _temporary_cwd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _configure_shen_paths(shen_config, pubtools_path):
    """Point Shen's module-level paths at the selected pubtools checkout."""
    homepath = f"{Path(pubtools_path).resolve()}{os.sep}"
    shen_config.homepath = homepath
    shen_config.datapath = f"{homepath}data{os.sep}"
    return f"{homepath}obdata_copy{os.sep}"


def normalize_shen_lf_mode(mode):
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized not in SHEN_LF_MODES:
        raise ValueError(
            f"Unknown Shen LF mode {mode!r}; expected one of {SHEN_LF_MODES}."
        )
    return normalized


def normalize_completeness_lf_model(model):
    normalized = str(model).strip().lower().replace("-", "_")
    if normalized not in COMPLETENESS_LF_MODELS:
        raise ValueError(
            f"Unknown completeness LF model {model!r}; expected one of "
            f"{COMPLETENESS_LF_MODELS}."
        )
    return normalized


def completeness_lf_static_metadata(
    lf_model,
    *,
    shen_lf_mode=SHEN_DEFAULT_LF_MODE,
):
    """Return cache-safe scientific provenance without evaluating an LF."""

    lf_model = normalize_completeness_lf_model(lf_model)
    if lf_model == "shen":
        shen_lf_mode = normalize_shen_lf_mode(shen_lf_mode)
        semantics_by_mode = {
            "all_nh_attenuated": {
                **_lf_conversion_slope_semantics(
                    "attenuation_retaining_shen_all_nh_lf_continuum_proxy"
                ),
                "population_scope": "all_agn_nh20_26",
                "type1_definition": "none",
                "uv_attenuation_state": "shen_nh_attenuated",
                "galactic_foreground_treatment": "not_in_model",
                "internal_dust_treatment": "full_nh20_26_once",
                "coordinate_conversion": "native_monochromatic_M2500",
                "conversion_is_approximate": False,
                "expected_completeness_magnitude": "attenuated",
            },
            "type1_intrinsic": {
                **_lf_conversion_slope_semantics(
                    "intrinsic_shen_type1_lf_continuum_proxy"
                ),
                "population_scope": "shen_nh_lt22_proxy",
                "type1_definition": "nh_lt22",
                "uv_attenuation_state": "intrinsic_unreddened",
                "galactic_foreground_treatment": "not_in_model",
                "internal_dust_treatment": "none",
                "coordinate_conversion": "native_monochromatic_M2500",
                "conversion_is_approximate": False,
                "expected_completeness_magnitude": "dereddened",
            },
            "type1_attenuated": {
                **_lf_conversion_slope_semantics(
                    "attenuation_retaining_shen_type1_lf_continuum_proxy"
                ),
                "population_scope": "shen_nh_lt22_proxy",
                "type1_definition": "nh_lt22",
                "uv_attenuation_state": "shen_nh_attenuated",
                "galactic_foreground_treatment": "not_in_model",
                "internal_dust_treatment": "nh20_22_once",
                "coordinate_conversion": "native_monochromatic_M2500",
                "conversion_is_approximate": False,
                "expected_completeness_magnitude": "attenuated",
            },
        }
        return {
            "model_id": "shen",
            "shen_lf_mode": shen_lf_mode,
            "native_magnitude_name": "M_2500_AB",
            "reference_wavelength_angstrom": 2500.0,
            "native_to_monochromatic_ab_offset": 0.0,
            "source_cosmology": {"H0": 70.0, "Om0": 0.3},
            "calibration_redshift_range": [0.0, 7.0],
            "formula_version": f"shen2020_global_fit_a_{shen_lf_mode}",
            "equations": ["Shen2020_global_fit_A_band_convolution"],
            "parameters": {
                "global_fit": SHEN_GLOBAL_FIT,
                "type1_log_nh_min": SHEN_TYPE1_NH_MIN,
                "type1_log_nh_max": SHEN_TYPE1_NH_MAX,
                "type1_log_nh_split": SHEN_TYPE1_NH_SPLIT,
                "nh_samples_per_dex": SHEN_TYPE1_NH_SAMPLES_PER_DEX,
            },
            "semantics": semantics_by_mode[shen_lf_mode],
        }

    if lf_model == WANG2026_TYPE1_LADE_A:
        return {
            "model_id": lf_model,
            "native_magnitude_name": "M_1450_AB",
            "reference_wavelength_angstrom": (
                WANG2026_REFERENCE_WAVELENGTH_ANGSTROM
            ),
            "native_to_monochromatic_ab_offset": 0.0,
            "source_cosmology": {
                "H0": float(WANG2026_SOURCE_COSMOLOGY.H0.value),
                "Om0": float(WANG2026_SOURCE_COSMOLOGY.Om0),
            },
            "calibration_redshift_range": [0.1, 7.5],
            "formula_version": "wang2026_model_a_eq8_eq9_eq11_eq12_table2",
            "equations": [8, 9, 11, 12],
            "parameters": {
                "log10_phi_star": -6.635,
                "M_star": -22.345,
                "beta_faint": -1.675,
                "gamma_bright": -3.819,
                "p1": 2.112,
                "p2": 1.222,
                "p3": 9.100,
                "k1": 0.066,
                "k2": -0.863,
                "k3": 2.224,
            },
            "semantics": {
                **_lf_conversion_slope_semantics(
                    "attenuation_retaining_empirical_lf_continuum_proxy"
                ),
                "population_scope": "observational_type1",
                "type1_definition": "optical_type1_compilation",
                "uv_attenuation_state": "empirical_rest_M1450",
                "galactic_foreground_treatment": (
                    "corrected_in_inherited_k19_photometry"
                ),
                "internal_dust_treatment": "implicit_no_reapplication",
                "coordinate_conversion": "drawn_power_law_M1450_to_M2500_proxy",
                "conversion_is_approximate": True,
                "expected_completeness_magnitude": "attenuated",
                "heavily_reddened_type1_recovery": "not_demonstrated",
            },
        }

    if lf_model == PALANQUE2016_PLE_LEDE:
        return {
            "model_id": PALANQUE2016_PLE_LEDE,
            "native_magnitude_name": "M_g(z=2)_AB",
            "reference_wavelength_angstrom": (
                PALANQUE2016_REFERENCE_WAVELENGTH_ANGSTROM
            ),
            "native_to_monochromatic_ab_offset": (
                PALANQUE2016_NATIVE_TO_MONOCHROMATIC_AB_OFFSET
            ),
            "source_cosmology": {
                "H0": float(PALANQUE2016_SOURCE_COSMOLOGY.H0.value),
                "Om0": float(PALANQUE2016_SOURCE_COSMOLOGY.Om0),
            },
            "calibration_redshift_range": [0.68, 4.0],
            "formula_version": "palanque2016_ple_lede_eq6_to_eq10_table4",
            "equations": [6, 7, 8, 9, 10],
            "parameters": {
                "M_star_0": -22.25,
                "log10_phi_star_0": -5.93,
                "alpha_bright": -3.89,
                "beta_faint": -1.47,
                "k1": 1.59,
                "k2": -0.36,
                "z_pivot": 2.2,
                "c1a": -0.46,
                "c1b": -0.06,
                "c2": -0.14,
                "c3": 0.32,
            },
            "semantics": {
                **_lf_conversion_slope_semantics(
                    "attenuation_retaining_empirical_lf_continuum_proxy"
                ),
                "population_scope": "observational_optical_quasar",
                "type1_definition": "variability_plus_spectroscopic_quasar",
                "uv_attenuation_state": "empirical_Mg_z2",
                "galactic_foreground_treatment": (
                    "corrected_in_g_dered_using_schlegel1998"
                ),
                "internal_dust_treatment": "implicit_no_reapplication",
                "coordinate_conversion": (
                    "approximate_Mg_z2_Knormalized_to_monochromatic_1557A_"
                    "then_power_law_to_M2500_proxy"
                ),
                "native_magnitude_reference_redshift": 2.0,
                "reference_redshift_ab_normalization_mag": (
                    PALANQUE2016_NATIVE_TO_MONOCHROMATIC_AB_OFFSET
                ),
                "conversion_is_approximate": True,
                "expected_completeness_magnitude": "attenuated",
                "heavily_reddened_type1_recovery": "not_demonstrated",
                "bal_treatment": "not_separately_modeled_or_excluded",
            },
        }

    if lf_model == KULKARNI2019_TYPE1_MODEL1:
        return _kulkarni2019_static_metadata(
            lf_model,
            model_number=1,
            formula_version=(
                "kulkarni2019_model1_eq7_eq13_eq16_to_eq18_table3_"
                "published_rounded_medians"
            ),
            equations=[7, 13, 16, 17, 18],
            beta_evolution="broken_power_law_in_1_plus_z",
            include_approximate_selection_samples=True,
            parameters={
                "chebyshev_argument": "1+z_unscaled",
                "log10_phi_star_chebyshev_coefficients": [
                    -7.798,
                    1.128,
                    -0.120,
                ],
                "M_star_chebyshev_coefficients": [
                    -17.163,
                    -5.512,
                    0.593,
                    -0.024,
                ],
                "alpha_bright_chebyshev_coefficients": [-3.223, -0.258],
                "beta_faint_broken_power_law_coefficients": [
                    -2.312,
                    0.559,
                    3.773,
                    141.884,
                    -0.171,
                ],
                "beta_faint_break_redshift": 3.773,
            },
            model_specific_caveats=[
                "paper_identifies_an_extremely_sharp_probably_unphysical_"
                "faint_end_slope_transition_with_fitted_break_at_z_3p773",
                "default_qvc_zmax_3p16_ends_before_this_transition",
                "transition_causes_a_discontinuity_in_cumulative_faint_agn_"
                "number_density",
                "faint_number_density_diverges_when_extrapolated_to_high_z_"
                "and_arbitrarily_faint_magnitudes",
            ],
        )

    if lf_model == KULKARNI2019_TYPE1_MODEL2:
        return _kulkarni2019_static_metadata(
            lf_model,
            model_number=2,
            formula_version=(
                "kulkarni2019_model2_eq7_eq13_eq16_to_eq18_table3_"
                "published_rounded_medians"
            ),
            equations=[7, 13, 16, 17, 18],
            beta_evolution="broken_power_law_in_1_plus_z",
            include_approximate_selection_samples=False,
            parameters={
                "chebyshev_argument": "1+z_unscaled",
                "log10_phi_star_chebyshev_coefficients": [
                    -7.432,
                    0.953,
                    -0.112,
                ],
                "M_star_chebyshev_coefficients": [
                    -15.412,
                    -6.869,
                    0.778,
                    -0.032,
                ],
                "alpha_bright_chebyshev_coefficients": [-2.959, -0.351],
                "beta_faint_broken_power_law_coefficients": [
                    -2.264,
                    0.530,
                    2.379,
                    12.527,
                    -0.229,
                ],
                "beta_faint_break_redshift": 2.379,
            },
            model_specific_caveats=[
                "approximate_selection_samples_17_to_20_excluded",
                "faint_number_density_diverges_when_extrapolated_to_high_z_"
                "and_arbitrarily_faint_magnitudes",
            ],
        )

    if lf_model == KULKARNI2019_TYPE1_MODEL3:
        return _kulkarni2019_static_metadata(
            lf_model,
            model_number=3,
            formula_version=(
                "kulkarni2019_model3_eq7_eq13_eq16_table3_"
                "published_rounded_medians"
            ),
            equations=[7, 13, 16],
            beta_evolution="linear_chebyshev_in_1_plus_z",
            include_approximate_selection_samples=False,
            parameters={
                "chebyshev_argument": "1+z_unscaled",
                "log10_phi_star_chebyshev_coefficients": [
                    -6.942,
                    0.629,
                    -0.086,
                ],
                "M_star_chebyshev_coefficients": [
                    -15.038,
                    -7.046,
                    0.772,
                    -0.030,
                ],
                "alpha_bright_chebyshev_coefficients": [-2.888, -0.383],
                "beta_faint_chebyshev_coefficients": [-1.602, -0.082],
            },
            model_specific_caveats=[
                "approximate_selection_samples_17_to_20_excluded",
                "does_not_match_the_measured_z_gt_4_faint_end_slope_as_well_"
                "as_model1",
            ],
        )

    raise RuntimeError(
        "Completeness LF model is registered but has no explicit static "
        f"metadata implementation: {lf_model!r}."
    )


def completeness_lf_magnitude_state_match(
    lf_model,
    completeness_magnitude,
    *,
    shen_lf_mode=SHEN_DEFAULT_LF_MODE,
):
    metadata = completeness_lf_static_metadata(
        lf_model,
        shen_lf_mode=shen_lf_mode,
    )
    expected = metadata["semantics"]["expected_completeness_magnitude"]
    return str(completeness_magnitude) == expected, expected


def shen_absorbed_fraction(log_lx_erg_s, redshift):
    """Return Shen/Ueda absorbed Compton-thin fraction psi(L_X, z)."""

    log_lx = np.asarray(log_lx_erg_s, dtype=float)
    z_capped = min(max(float(redshift), 0.0), 2.0)
    psi_43_75 = 0.43 * (1.0 + z_capped) ** 0.48
    return np.clip(psi_43_75 - 0.24 * (log_lx - 43.75), 0.20, 0.84)


def shen_type1_fraction(log_lx_erg_s, redshift, *, f_ctk=1.0):
    """Return the N_H < 1e22 fraction of Shen's total AGN population."""

    psi = shen_absorbed_fraction(log_lx_erg_s, redshift)
    return (1.0 - psi) / (1.0 + float(f_ctk) * psi)


def _shen_type1_nh_bin_fractions(log_lx_erg_s, redshift):
    """Return total-population weights in the 20--21 and 21--22 N_H bins."""

    psi = shen_absorbed_fraction(log_lx_erg_s, redshift)
    epsilon = 1.7
    threshold = (1.0 + epsilon) / (3.0 + epsilon)
    low_branch = psi < threshold
    f1 = np.where(
        low_branch,
        1.0 - ((2.0 + epsilon) / (1.0 + epsilon)) * psi,
        2.0 / 3.0 - ((3.0 + 2.0 * epsilon) / (3.0 + 3.0 * epsilon)) * psi,
    )
    f2 = np.where(
        low_branch,
        psi / (1.0 + epsilon),
        1.0 / 3.0 - (epsilon / (3.0 + 3.0 * epsilon)) * psi,
    )
    normalization = 1.0 + psi
    f1 = f1 / normalization
    f2 = f2 / normalization
    expected = shen_type1_fraction(log_lx_erg_s, redshift)
    if not np.allclose(f1 + f2, expected, rtol=1e-12, atol=1e-14):
        raise RuntimeError("Shen Type-1 N_H weights do not reproduce f_Type1.")
    return f1, f2


def _load_shen_c_backend(pubtools_path):
    library_path = Path(pubtools_path) / "clib" / "convolve.so"
    if not library_path.is_file():
        raise FileNotFoundError(
            f"Shen convolution library not found: {library_path}. "
            "Compile the released pubtools C extension first."
        )
    library = ctypes.CDLL(str(library_path))
    library.l_band.argtypes = [ctypes.c_double, ctypes.c_double]
    library.l_band.restype = ctypes.c_double
    library.l_band_dispersion.argtypes = [ctypes.c_double, ctypes.c_double]
    library.l_band_dispersion.restype = ctypes.c_double
    library.return_tau.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
    library.return_tau.restype = ctypes.c_double
    return library


def _normal_density_grid(grid, means, sigmas):
    delta = (grid[:, None] - means[None, :]) / sigmas[None, :]
    return np.exp(-0.5 * delta**2) / (np.sqrt(2.0 * np.pi) * sigmas[None, :])


def _build_shen_type1_lf_at_redshift(
    redshift,
    *,
    return_bolometric_qlf,
    return_dtg,
    backend,
    attenuated,
):
    """Numerically convolve Shen's bolometric LF into a Type-1 UV LF."""

    log_lbol_erg_s, log_phi_bol = return_bolometric_qlf(
        redshift,
        model=SHEN_GLOBAL_FIT,
    )
    log_lbol_erg_s = np.asarray(log_lbol_erg_s, dtype=float)
    log_phi_bol = np.asarray(log_phi_bol, dtype=float)
    if (
        log_lbol_erg_s.ndim != 1
        or log_phi_bol.shape != log_lbol_erg_s.shape
        or log_lbol_erg_s.size < 2
        or not np.all(np.isfinite(log_lbol_erg_s))
        or not np.all(np.isfinite(log_phi_bol))
    ):
        raise ValueError("Shen bolometric LF returned an invalid grid.")

    log_lsun = np.log10(SHEN_L_SUN_ERG_S)
    log_lbol_lsun = log_lbol_erg_s - log_lsun
    mean_log_uv = np.asarray(
        [
            np.log10(backend.l_band(value, NU_2500_HZ)) + log_lsun
            for value in log_lbol_lsun
        ],
        dtype=float,
    )
    sigma_log_uv = np.asarray(
        [backend.l_band_dispersion(value, NU_2500_HZ) for value in log_lbol_lsun],
        dtype=float,
    )
    log_lx = np.asarray(
        [
            np.log10(backend.l_band(value, -4.0)) + log_lsun
            for value in log_lbol_lsun
        ],
        dtype=float,
    )
    if (
        not np.all(np.isfinite(mean_log_uv))
        or not np.all(np.isfinite(sigma_log_uv))
        or np.any(sigma_log_uv <= 0.0)
        or not np.all(np.isfinite(log_lx))
    ):
        raise ValueError("Shen band conversion returned invalid UV or X-ray values.")

    log_uv_grid = np.linspace(
        float(np.min(mean_log_uv)),
        float(np.max(mean_log_uv)),
        log_lbol_erg_s.size,
    )
    integration_weight = 10.0**log_phi_bol * np.gradient(log_lbol_erg_s)

    if not attenuated:
        type1_weight = shen_type1_fraction(log_lx, redshift)
        phi_uv = _normal_density_grid(
            log_uv_grid,
            mean_log_uv,
            sigma_log_uv,
        ) @ (integration_weight * type1_weight)
        return log_uv_grid, phi_uv

    f_nh_20_21, f_nh_21_22 = _shen_type1_nh_bin_fractions(log_lx, redshift)
    n_nh = SHEN_TYPE1_NH_SAMPLES_PER_DEX
    nh_20_21 = np.linspace(
        SHEN_TYPE1_NH_MIN,
        SHEN_TYPE1_NH_SPLIT,
        n_nh,
        endpoint=False,
    ) + 0.5 / n_nh
    nh_21_22 = np.linspace(
        SHEN_TYPE1_NH_SPLIT,
        SHEN_TYPE1_NH_MAX,
        n_nh,
        endpoint=False,
    ) + 0.5 / n_nh
    dust_to_gas = float(return_dtg(redshift))
    phi_uv = np.zeros_like(log_uv_grid)
    for nh_grid, bin_weight in (
        (nh_20_21, f_nh_20_21),
        (nh_21_22, f_nh_21_22),
    ):
        weighted_bolometric = integration_weight * bin_weight / len(nh_grid)
        for log_nh in nh_grid:
            tau = float(backend.return_tau(log_nh, NU_2500_HZ, dust_to_gas))
            attenuated_mean = mean_log_uv - tau / np.log(10.0)
            phi_uv += _normal_density_grid(
                log_uv_grid,
                attenuated_mean,
                sigma_log_uv,
            ) @ weighted_bolometric
    return log_uv_grid, phi_uv


def bolometric_correction_shen20(
    L_bol,
    c1=4.073,
    k1=-0.026,
    c2=12.60,
    k2=0.278,
):
    x = L_bol / L0
    return c1 * x**k1 + c2 * x**k2


def lbol_from_loglx_shen20(loglx_array):
    loglx_array = np.asarray(loglx_array, dtype=float)
    lx_array = 10.0**loglx_array
    lbol_array = np.empty_like(lx_array)

    for i, lx in enumerate(lx_array):
        def equation(lb):
            return lb - bolometric_correction_shen20(lb) * lx

        lbol_array[i] = brentq(equation, lx, 1e4 * lx)

    return np.log10(lbol_array), lbol_array


def build_shen_lf(pubtools_path, mode=SHEN_DEFAULT_LF_MODE, *, progress=False):
    """Build an explicit Shen global-fit-A 2500 A population mode."""

    mode = normalize_shen_lf_mode(mode)
    if pubtools_path is None:
        pubtools_path = default_shen_pubtools_path()
    if pubtools_path is None:
        raise FileNotFoundError(
            "Shen pubtools path not found. Pass --shen-pubtools-path or set SHEN_PUBTOOLS_PATH."
        )
    pubtools_path = Path(pubtools_path).expanduser().resolve()
    if not pubtools_path.exists():
        raise FileNotFoundError(f"Shen pubtools path not found: {pubtools_path}")

    sys.path.insert(0, str(pubtools_path))
    added_obdata_path = None
    progress_stream = sys.stderr
    try:
        with _temporary_cwd(pubtools_path):
            silent_stream = io.StringIO()
            with redirect_stdout(silent_stream), redirect_stderr(silent_stream):
                config_path = pubtools_path / "config.py"
                if config_path.is_file():
                    shen_config = importlib.import_module("config")
                    added_obdata_path = _configure_shen_paths(
                        shen_config,
                        pubtools_path,
                    )
                    sys.path.insert(0, added_obdata_path)
                z_bins = np.linspace(0.0, 8.0, 40)
                shen_utilities = importlib.import_module("utilities")
                if mode == SHEN_DEFAULT_LF_MODE:
                    qlf_values = [
                        shen_utilities.return_qlf_in_band(
                            redshift=z,
                            nu=NU_2500_HZ,
                            model=SHEN_GLOBAL_FIT,
                        )
                        for z in tqdm(
                            z_bins,
                            desc=f"Building Shen LF ({mode})",
                            unit="redshift",
                            disable=not progress,
                            mininterval=1.0,
                            dynamic_ncols=True,
                            file=progress_stream,
                        )
                    ]
                else:
                    backend = _load_shen_c_backend(pubtools_path)
                    attenuated = mode == "type1_attenuated"
                    qlf_values = [
                        _build_shen_type1_lf_at_redshift(
                            z,
                            return_bolometric_qlf=(
                                shen_utilities.return_bolometric_qlf
                            ),
                            return_dtg=shen_utilities.return_dtg,
                            backend=backend,
                            attenuated=attenuated,
                        )
                        for z in tqdm(
                            z_bins,
                            desc=f"Building Shen LF ({mode})",
                            unit="redshift",
                            disable=not progress,
                            mininterval=1.0,
                            dynamic_ncols=True,
                            file=progress_stream,
                        )
                    ]
    finally:
        if added_obdata_path is not None:
            try:
                sys.path.remove(added_obdata_path)
            except ValueError:
                pass
        try:
            sys.path.remove(str(pubtools_path))
        except ValueError:
            pass

    luminosity_grids = np.asarray([qlf[0] for qlf in qlf_values], dtype=float)
    if (
        luminosity_grids.ndim != 2
        or np.any(~np.isfinite(luminosity_grids))
        or not np.allclose(
            luminosity_grids,
            luminosity_grids[0],
            rtol=1e-12,
            atol=1e-12,
        )
    ):
        raise ValueError(f"Shen {mode} returned inconsistent luminosity grids.")
    luminosities = luminosity_grids[0]
    phi_values = np.asarray([qlf[1] for qlf in qlf_values], dtype=float)
    if mode == SHEN_DEFAULT_LF_MODE:
        phi_log10 = phi_values + LOG10_MAG_JACOBIAN
    else:
        if np.any(~np.isfinite(phi_values)) or np.any(phi_values < 0.0):
            raise ValueError(f"Shen {mode} convolution returned invalid densities.")
        with np.errstate(divide="ignore"):
            phi_log10 = np.log10(phi_values) + LOG10_MAG_JACOBIAN

    # all_nh_attenuated evaluates Shen's physical 2500 A channel, including
    # bolometric-correction scatter and the full N_H extinction distribution.
    # type1_intrinsic applies the N_H<1e22 population weight before the
    # intrinsic UV convolution. type1_attenuated additionally integrates the
    # released attenuation law over the two unobscured N_H bins.
    # m_grid is monochromatic rest-frame absolute AB M_2500, converted from
    # Shen's nu*L_nu(2500 A); it is not apparent, bolometric, or band-integrated.
    m_grid = log_nu_lnu_to_ab_absolute_magnitude(luminosities, NU_2500_HZ)
    return phi_log10, m_grid, z_bins


def build_completeness_lf(
    lf_model,
    *,
    shen_lf_mode=SHEN_DEFAULT_LF_MODE,
    z_range=(0.44, 3.16),
    shen_pubtools_path=None,
    target_cosmology=COSMO,
    progress=False,
):
    """Build any supported completeness LF without adding a dust screen."""

    lf_model = normalize_completeness_lf_model(lf_model)
    z_min, z_max = (float(value) for value in z_range)
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min >= z_max:
        raise ValueError("z_range must contain two finite increasing values.")
    metadata = completeness_lf_static_metadata(
        lf_model,
        shen_lf_mode=shen_lf_mode,
    )
    semantics = metadata["semantics"]
    reddening_semantics = ReddeningSemantics(
        luminosity_semantics=str(semantics["uv_attenuation_state"]),
        galactic_extinction=str(semantics["galactic_foreground_treatment"]),
        internal_extinction=str(semantics["internal_dust_treatment"]),
        selection_dust_treatment=str(semantics["population_scope"]),
        apply_additional_internal_extinction=False,
    )

    if lf_model == "shen":
        shen_lf_mode = normalize_shen_lf_mode(shen_lf_mode)
        if not (
            np.isclose(
                float(target_cosmology.H0.value),
                float(COSMO.H0.value),
                rtol=0.0,
                atol=1e-12,
            )
            and np.isclose(
                float(target_cosmology.Om0),
                float(COSMO.Om0),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise NotImplementedError(
                "The Shen grid is native to H0=70, Om0=0.3 and is not yet "
                "implemented for a different target cosmology."
            )
        phi_log10, magnitude_grid, redshift_grid = build_shen_lf(
            shen_pubtools_path,
            mode=shen_lf_mode,
            progress=progress,
        )
        return LFGrid(
            model_id="shen",
            phi_log10=phi_log10,
            native_magnitude_grid=magnitude_grid,
            redshift_grid=redshift_grid,
            reference_wavelength_angstrom=2500.0,
            native_to_monochromatic_ab_offset=0.0,
            native_magnitude_name="M_2500_AB",
            source_cosmology=COSMO,
            target_cosmology=target_cosmology,
            calibration_redshift_range=(0.0, 7.0),
            formula_version=metadata["formula_version"],
            reddening_semantics=reddening_semantics,
        )

    n_redshift = max(
        2,
        int(np.ceil((z_max - z_min) / EMPIRICAL_LF_REDSHIFT_STEP)) + 1,
    )
    redshift_grid = np.linspace(z_min, z_max, n_redshift)
    grid = build_empirical_lf(
        lf_model,
        EMPIRICAL_LF_NATIVE_MAGNITUDE_GRID,
        redshift_grid,
        target_cosmology,
    )
    extrapolated = grid.redshift_grid[grid.redshift_extrapolation_mask]
    if extrapolated.size:
        extrapolated_ranges = []
        calibration_min, calibration_max = grid.calibration_redshift_range
        if z_min < calibration_min:
            extrapolated_ranges.append(
                f"[{z_min:.3f}, {min(z_max, calibration_min):.3f})"
            )
        if z_max > calibration_max:
            extrapolated_ranges.append(
                f"({max(z_min, calibration_max):.3f}, {z_max:.3f}]"
            )
        print(
            "[WARNING] Completeness LF "
            f"{lf_model!r} is being analytically extrapolated over "
            f"z={' and '.join(extrapolated_ranges)}; calibrated range is "
            f"{grid.calibration_redshift_range}."
        )
    if lf_model == WANG2026_TYPE1_LADE_A and z_min < 0.6:
        print(
            "[WARNING] Wang 2026 low-redshift faint cells extend beyond the "
            "paper's conservative luminosity cuts; this extrapolation is "
            "recorded in the mock provenance."
        )
    if lf_model in KULKARNI2019_TYPE1_MODEL_IDS:
        if z_min < 0.6:
            print(
                "[WARNING] Kulkarni 2019 excluded z<0.6 from its global fits "
                "because of residual host-galaxy correction and extended-source "
                "selection systematics. The requested low-redshift interval is "
                "an analytic extrapolation and is recorded in mock provenance."
            )
        boss_min, boss_max = KULKARNI2019_BOSS_EXCLUDED_REDSHIFT_INTERVAL
        if z_min < boss_max and z_max > boss_min:
            overlap_min = max(z_min, boss_min)
            overlap_max = min(z_max, boss_max)
            print(
                "[WARNING] Kulkarni 2019 excluded the BOSS DR9 sample over "
                f"2.2<=z<3.5 because of selection-function systematics. "
                f"Requested z={overlap_min:.3f}--{overlap_max:.3f} is a smooth "
                "global-model interpolation without credible fit data; this is "
                "recorded in mock provenance."
            )
        feature_min, feature_max = (
            KULKARNI2019_MODEL1_FEATURE_REDSHIFT_INTERVAL
        )
        if (
            lf_model == KULKARNI2019_TYPE1_MODEL1
            and z_min < feature_max
            and z_max > feature_min
        ):
            print(
                "[WARNING] Kulkarni 2019 Model 1 has a paper-identified, "
                "probably unphysical sharp faint-end-slope transition centered "
                "near z=3.773. It can create a discontinuity in cumulative faint "
                "AGN counts; the requested range overlaps the feature."
            )
    return grid


def build_ananna_lf(ananna_xlf_path):
    if ananna_xlf_path is None:
        ananna_xlf_path = default_ananna_xlf_path()
    if ananna_xlf_path is None:
        raise FileNotFoundError(
            "Ananna XLF file not found. Pass --ananna-xlf-path or set ANANNA_XLF_PATH."
        )
    ananna_xlf_path = Path(ananna_xlf_path).expanduser().resolve()
    if not ananna_xlf_path.exists():
        raise FileNotFoundError(f"Ananna XLF file not found: {ananna_xlf_path}")

    lumbins = np.linspace(41.0, 47.0, 150)
    zbin1 = np.arange(0.002, 0.1, 0.005)
    zbin2 = np.arange(0.1, 1.0, 0.05)
    zbin3 = np.arange(1.0, 5.05, 0.05)
    zbins = np.concatenate([zbin1, zbin2, zbin3])
    nhbins = np.linspace(20.0, 26.0, 80)

    with gzip.GzipFile(ananna_xlf_path, "r") as handle:
        loading_matr = np.load(handle)

    lum_func = RegularGridInterpolator(
        (zbins, lumbins, nhbins),
        loading_matr[0],
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    mask_unobs = (nhbins >= 20.0) & (nhbins < 22.0)
    nh_unobs = nhbins[mask_unobs]
    z_bins = np.linspace(0.01, 4.5, 20)
    lx_grid = lumbins

    phi_unobs_2d = np.zeros((len(z_bins), len(lx_grid)))
    for i, z in enumerate(z_bins):
        zz, ll, nh = np.meshgrid(np.array([z]), lx_grid, nh_unobs, indexing="ij")
        pts = np.column_stack([zz.ravel(), ll.ravel(), nh.ravel()])
        phi_3d = lum_func(pts).reshape(1, len(lx_grid), len(nh_unobs))
        phi_unobs_2d[i, :] = np.trapezoid(phi_3d[0], x=nh_unobs, axis=-1)

    loglbol_grid, _ = lbol_from_loglx_shen20(lx_grid)
    jac = np.empty_like(lx_grid)
    jac[1:-1] = (lx_grid[2:] - lx_grid[:-2]) / (loglbol_grid[2:] - loglbol_grid[:-2])
    jac[0] = (lx_grid[1] - lx_grid[0]) / (loglbol_grid[1] - loglbol_grid[0])
    jac[-1] = (lx_grid[-1] - lx_grid[-2]) / (loglbol_grid[-1] - loglbol_grid[-2])

    phi_bol_lin = np.clip(phi_unobs_2d, 1e-40, None) * jac[None, :]
    phi_bol_log10 = np.log10(np.clip(phi_bol_lin, 1e-40, None)) + LOG10_MAG_JACOBIAN
    m_grid = 91.0 - 2.5 * loglbol_grid
    return phi_bol_log10, m_grid, z_bins


def native_absolute_magnitude_to_m2500(
    native_magnitude,
    alpha_nu_lf_conversion,
    reference_wavelength_angstrom,
    native_to_monochromatic_ab_offset=0.0,
):
    """Convert an LF magnitude to rest-frame M_2500 without dereddening.

    The population-level conversion proxy follows ``f_nu`` proportional to
    ``nu**alpha_nu_lf_conversion``.  For empirical Type-1 LFs it retains the
    papers' implicit internal attenuation: this is only a wavelength-coordinate
    conversion and never applies or removes a dust screen.  It is not the
    intrinsic JAXSedFit accretion-disk slope.
    """

    reference_wavelength_angstrom = float(reference_wavelength_angstrom)
    if (
        not np.isfinite(reference_wavelength_angstrom)
        or reference_wavelength_angstrom <= 0.0
    ):
        raise ValueError("reference_wavelength_angstrom must be positive.")
    native_magnitude = np.asarray(native_magnitude, dtype=float)
    alpha_nu_lf_conversion = np.asarray(
        alpha_nu_lf_conversion,
        dtype=float,
    )
    native_to_monochromatic_ab_offset = float(
        native_to_monochromatic_ab_offset
    )
    if not np.isfinite(native_to_monochromatic_ab_offset):
        raise ValueError("native_to_monochromatic_ab_offset must be finite.")
    coefficient = 2.5 * np.log10(2500.0 / reference_wavelength_angstrom)
    return (
        native_magnitude
        + native_to_monochromatic_ab_offset
        + coefficient * alpha_nu_lf_conversion
    )


def _m2500_lf_conversion_alpha_bounds(
    native_magnitude,
    distance_modulus,
    reference_wavelength_angstrom,
    m2500_support,
    native_to_monochromatic_ab_offset=0.0,
):
    """Return the LF-conversion alpha_nu interval allowed by m2500 support."""

    m_min, m_max = (float(value) for value in m2500_support)
    if not np.isfinite(m_min) or not np.isfinite(m_max) or m_min >= m_max:
        raise ValueError("m2500_support must contain two finite increasing values.")
    coefficient = 2.5 * np.log10(
        2500.0 / float(reference_wavelength_angstrom)
    )
    base = np.asarray(native_magnitude, dtype=float) + np.asarray(
        distance_modulus,
        dtype=float,
    ) + float(native_to_monochromatic_ab_offset)
    if np.isclose(coefficient, 0.0, rtol=0.0, atol=1e-14):
        accepted = (base >= m_min) & (base <= m_max)
        lower = np.where(accepted, -np.inf, np.inf)
        upper = np.where(accepted, np.inf, -np.inf)
        return lower, upper
    bound_1 = (m_min - base) / coefficient
    bound_2 = (m_max - base) / coefficient
    return np.minimum(bound_1, bound_2), np.maximum(bound_1, bound_2)


def m2500_support_probability(
    native_magnitude,
    distance_modulus,
    *,
    reference_wavelength_angstrom,
    m2500_support,
    alpha_nu_lf_conversion_mean,
    alpha_nu_lf_conversion_sigma,
    native_to_monochromatic_ab_offset=0.0,
):
    """Probability that a Gaussian LF-conversion slope gives supported m2500."""

    alpha_nu_lf_conversion_mean = float(alpha_nu_lf_conversion_mean)
    alpha_nu_lf_conversion_sigma = float(abs(alpha_nu_lf_conversion_sigma))
    lower, upper = _m2500_lf_conversion_alpha_bounds(
        native_magnitude,
        distance_modulus,
        reference_wavelength_angstrom,
        m2500_support,
        native_to_monochromatic_ab_offset,
    )
    if alpha_nu_lf_conversion_sigma == 0.0:
        return (
            (alpha_nu_lf_conversion_mean >= lower)
            & (alpha_nu_lf_conversion_mean <= upper)
        ).astype(float)
    standardized_lower = (
        lower - alpha_nu_lf_conversion_mean
    ) / alpha_nu_lf_conversion_sigma
    standardized_upper = (
        upper - alpha_nu_lf_conversion_mean
    ) / alpha_nu_lf_conversion_sigma
    return _standard_normal_interval_probability(
        standardized_lower,
        standardized_upper,
    )


def _standard_normal_interval_probability(lower, upper):
    """Return Phi(upper)-Phi(lower) without catastrophic tail cancellation."""

    lower, upper = np.broadcast_arrays(
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )
    probability = np.zeros(lower.shape, dtype=float)
    valid = lower <= upper

    left_tail = valid & (upper <= 0.0)
    if np.any(left_tail):
        log_upper = log_ndtr(upper[left_tail])
        log_lower = log_ndtr(lower[left_tail])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            log_probability = log_upper + np.log1p(
                -np.exp(np.minimum(log_lower - log_upper, 0.0))
            )
        probability[left_tail] = np.exp(log_probability)

    right_tail = valid & (lower >= 0.0) & ~left_tail
    if np.any(right_tail):
        log_survival_lower = log_ndtr(-lower[right_tail])
        log_survival_upper = log_ndtr(-upper[right_tail])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            log_probability = log_survival_lower + np.log1p(
                -np.exp(
                    np.minimum(
                        log_survival_upper - log_survival_lower,
                        0.0,
                    )
                )
            )
        probability[right_tail] = np.exp(log_probability)

    central = valid & ~left_tail & ~right_tail
    if np.any(central):
        probability[central] = (
            ndtr(upper[central]) - ndtr(lower[central])
        )
    return np.clip(probability, 0.0, 1.0)


def sample_alpha_nu_lf_conversion_conditional_on_m2500_support(
    rng,
    native_magnitude,
    distance_modulus,
    *,
    reference_wavelength_angstrom,
    m2500_support,
    alpha_nu_lf_conversion_mean,
    alpha_nu_lf_conversion_sigma,
    native_to_monochromatic_ab_offset=0.0,
):
    """Draw LF-conversion slopes conditional on requested m2500 support."""

    native_magnitude = np.asarray(native_magnitude, dtype=float)
    distance_modulus = np.asarray(distance_modulus, dtype=float)
    alpha_nu_lf_conversion_mean = float(alpha_nu_lf_conversion_mean)
    alpha_nu_lf_conversion_sigma = float(abs(alpha_nu_lf_conversion_sigma))
    lower, upper = _m2500_lf_conversion_alpha_bounds(
        native_magnitude,
        distance_modulus,
        reference_wavelength_angstrom,
        m2500_support,
        native_to_monochromatic_ab_offset,
    )
    if np.any(lower > upper):
        raise ValueError(
            "Cannot sample alpha_nu_lf_conversion for objects outside "
            "m2500 support."
        )
    if alpha_nu_lf_conversion_sigma == 0.0:
        clipped_alpha = np.clip(alpha_nu_lf_conversion_mean, lower, upper)
        if not np.allclose(
            clipped_alpha,
            alpha_nu_lf_conversion_mean,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "Fixed alpha_nu_lf_conversion lies outside the conditional "
                "support."
            )
        return np.asarray(clipped_alpha, dtype=float)
    standardized_lower = (
        lower - alpha_nu_lf_conversion_mean
    ) / alpha_nu_lf_conversion_sigma
    standardized_upper = (
        upper - alpha_nu_lf_conversion_mean
    ) / alpha_nu_lf_conversion_sigma
    probability = _standard_normal_interval_probability(
        standardized_lower,
        standardized_upper,
    )
    if np.any(probability <= 0.0) or np.any(~np.isfinite(probability)):
        raise ValueError(
            "Cannot sample a zero-probability alpha_nu_lf_conversion interval."
        )
    unit_uniforms = np.clip(
        rng.random(probability.shape),
        np.finfo(float).eps,
        1.0 - np.finfo(float).eps,
    )
    standardized_draws = np.empty(probability.shape, dtype=float)

    left_tail = standardized_upper <= 0.0
    if np.any(left_tail):
        log_cdf_lower = log_ndtr(standardized_lower[left_tail])
        log_cdf_upper = log_ndtr(standardized_upper[left_tail])
        uniforms = unit_uniforms[left_tail]
        log_cdf_draw = np.logaddexp(
            log_cdf_lower + np.log1p(-uniforms),
            log_cdf_upper + np.log(uniforms),
        )
        standardized_draws[left_tail] = ndtri_exp(log_cdf_draw)

    right_tail = (standardized_lower >= 0.0) & ~left_tail
    if np.any(right_tail):
        log_survival_lower = log_ndtr(-standardized_lower[right_tail])
        log_survival_upper = log_ndtr(-standardized_upper[right_tail])
        uniforms = unit_uniforms[right_tail]
        log_survival_draw = np.logaddexp(
            log_survival_lower + np.log1p(-uniforms),
            log_survival_upper + np.log(uniforms),
        )
        standardized_draws[right_tail] = -ndtri_exp(log_survival_draw)

    central = ~left_tail & ~right_tail
    if np.any(central):
        cdf_lower = ndtr(standardized_lower[central])
        cdf_upper = ndtr(standardized_upper[central])
        cdf_draw = cdf_lower + unit_uniforms[central] * (
            cdf_upper - cdf_lower
        )
        standardized_draws[central] = ndtri(cdf_draw)
    return (
        alpha_nu_lf_conversion_mean
        + alpha_nu_lf_conversion_sigma * standardized_draws
    )


def _sample_piecewise_linear_1d(rng, grid, density, size):
    """Sample a non-negative density represented linearly between grid nodes."""

    grid = np.asarray(grid, dtype=float)
    density = np.asarray(density, dtype=float)
    widths = np.diff(grid)
    interval_mass = 0.5 * (density[:-1] + density[1:]) * widths
    total_mass = float(np.sum(interval_mass))
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("Cannot sample a zero or non-finite piecewise density.")
    cumulative_mass = np.cumsum(interval_mass)
    mass_draw = rng.random(int(size)) * total_mass
    interval_index = np.searchsorted(
        cumulative_mass,
        mass_draw,
        side="right",
    )
    interval_index = np.clip(interval_index, 0, widths.size - 1)
    cumulative_lower = np.zeros_like(mass_draw)
    has_lower = interval_index > 0
    cumulative_lower[has_lower] = cumulative_mass[interval_index[has_lower] - 1]
    selected_mass = interval_mass[interval_index]
    quantile = np.clip(
        (mass_draw - cumulative_lower) / selected_mass,
        0.0,
        1.0,
    )
    left_density = density[interval_index]
    right_density = density[interval_index + 1]
    fraction = _invert_linear_density_quantile(
        left_density,
        right_density,
        quantile,
    )
    return grid[interval_index] + fraction * widths[interval_index]


def _invert_linear_density_quantile(left_density, right_density, quantile):
    """Invert the CDF of a linearly varying density on the unit interval."""

    left_density, right_density, quantile = np.broadcast_arrays(
        np.asarray(left_density, dtype=float),
        np.asarray(right_density, dtype=float),
        np.asarray(quantile, dtype=float),
    )
    delta = right_density - left_density
    mean_density = 0.5 * (left_density + right_density)
    target = np.clip(quantile, 0.0, 1.0) * mean_density
    fraction = np.empty(target.shape, dtype=float)
    effectively_flat = np.isclose(
        delta,
        0.0,
        rtol=1e-12,
        atol=np.finfo(float).tiny,
    )
    fraction[effectively_flat] = quantile[effectively_flat]
    varying = ~effectively_flat
    if np.any(varying):
        discriminant = np.maximum(
            left_density[varying] ** 2
            + 2.0 * delta[varying] * target[varying],
            0.0,
        )
        denominator = left_density[varying] + np.sqrt(discriminant)
        fraction[varying] = np.divide(
            2.0 * target[varying],
            denominator,
            out=np.zeros_like(target[varying]),
            where=denominator > 0.0,
        )
    return np.clip(fraction, 0.0, 1.0)


def _restricted_redshift_edges(z_bins, z_range):
    z_bins = np.asarray(z_bins, dtype=float)
    if z_bins.ndim != 1 or z_bins.size < 2 or np.any(np.diff(z_bins) <= 0.0):
        raise ValueError("z_bins must be a strictly increasing one-dimensional grid.")
    if z_range is None:
        return z_bins
    z_min, z_max = (float(value) for value in z_range)
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min >= z_max:
        raise ValueError("z_range must contain two finite increasing values.")
    if z_min < z_bins[0] or z_max > z_bins[-1]:
        raise ValueError(
            "z_range lies outside the LF redshift support: "
            f"requested=[{z_min}, {z_max}], support=[{z_bins[0]}, {z_bins[-1]}]."
        )
    interior = z_bins[(z_bins > z_min) & (z_bins < z_max)]
    return np.concatenate(([z_min], interior, [z_max]))


def _evaluate_lf_density(rgi, redshift, magnitude, *, valid=None):
    """Evaluate a log-LF interpolator as a finite linear density array."""

    magnitude = np.asarray(magnitude, dtype=float)
    redshift = np.asarray(redshift, dtype=float)
    if magnitude.ndim != 2 or redshift.ndim != 1:
        raise ValueError("magnitude must be 2D and redshift must be 1D.")
    if magnitude.shape[0] != redshift.size:
        raise ValueError("magnitude rows must match the redshift vector.")
    if valid is None:
        valid = np.ones(magnitude.shape, dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != magnitude.shape:
            raise ValueError("valid mask must match the magnitude array.")
    density = np.zeros(magnitude.shape, dtype=float)
    if not np.any(valid):
        return density
    redshift_2d = np.broadcast_to(redshift[:, None], magnitude.shape)
    points = np.column_stack(
        (redshift_2d[valid], magnitude[valid])
    )
    log_density = rgi(points)
    finite = np.isfinite(log_density)
    evaluated = np.zeros(log_density.shape, dtype=float)
    evaluated[finite] = 10.0 ** log_density[finite]
    density[valid] = evaluated
    return density


def _native_magnitude_interval_components(
    rgi,
    redshift,
    magnitude_grid,
    distance_modulus,
    *,
    reference_wavelength_angstrom,
    native_to_monochromatic_ab_offset,
    m2500_support,
    alpha_nu_lf_conversion_mean,
    alpha_nu_lf_conversion_sigma,
):
    """Return endpoints, endpoint densities, and masses for native-M cells.

    The returned interval masses include the Gaussian LF-conversion-slope
    acceptance probability. For empirical LFs this proxy preserves their
    attenuation-retaining magnitude state. When support is deterministic in
    native magnitude (notably a native M_2500 LF), cells are clipped at the
    exact apparent-magnitude boundaries before integration. This prevents edge
    leakage and preserves the Poisson normalization.
    """

    redshift = np.asarray(redshift, dtype=float)
    magnitude_grid = np.asarray(magnitude_grid, dtype=float)
    distance_modulus = np.asarray(distance_modulus, dtype=float)
    if redshift.ndim != 1 or distance_modulus.shape != redshift.shape:
        raise ValueError("redshift and distance_modulus must be matching vectors.")
    coefficient = 2.5 * np.log10(
        2500.0 / float(reference_wavelength_angstrom)
    )
    deterministic_support = (
        np.isclose(coefficient, 0.0, rtol=0.0, atol=1e-14)
        or float(abs(alpha_nu_lf_conversion_sigma)) == 0.0
    )
    n_rows = redshift.size
    n_intervals = magnitude_grid.size - 1

    if deterministic_support:
        fixed_color = (
            float(native_to_monochromatic_ab_offset)
            + coefficient * float(alpha_nu_lf_conversion_mean)
        )
        native_lower = (
            float(m2500_support[0]) - distance_modulus - fixed_color
        )
        native_upper = (
            float(m2500_support[1]) - distance_modulus - fixed_color
        )
        left = np.maximum(
            magnitude_grid[:-1][None, :],
            native_lower[:, None],
        )
        right = np.minimum(
            magnitude_grid[1:][None, :],
            native_upper[:, None],
        )
        valid = right > left
        left_density = _evaluate_lf_density(
            rgi,
            redshift,
            left,
            valid=valid,
        )
        right_density = _evaluate_lf_density(
            rgi,
            redshift,
            right,
            valid=valid,
        )
    else:
        nodes = np.broadcast_to(
            magnitude_grid[None, :],
            (n_rows, magnitude_grid.size),
        )
        lf_density = _evaluate_lf_density(rgi, redshift, nodes)
        support_probability = m2500_support_probability(
            nodes,
            distance_modulus[:, None],
            reference_wavelength_angstrom=reference_wavelength_angstrom,
            m2500_support=m2500_support,
            alpha_nu_lf_conversion_mean=(
                alpha_nu_lf_conversion_mean
            ),
            alpha_nu_lf_conversion_sigma=(
                alpha_nu_lf_conversion_sigma
            ),
            native_to_monochromatic_ab_offset=(
                native_to_monochromatic_ab_offset
            ),
        )
        weighted_density = lf_density * support_probability
        left = np.broadcast_to(
            magnitude_grid[:-1][None, :],
            (n_rows, n_intervals),
        )
        right = np.broadcast_to(
            magnitude_grid[1:][None, :],
            (n_rows, n_intervals),
        )
        left_density = weighted_density[:, :-1]
        right_density = weighted_density[:, 1:]

    width = np.maximum(right - left, 0.0)
    interval_mass = 0.5 * (left_density + right_density) * width
    interval_mass = np.where(
        np.isfinite(interval_mass) & (interval_mass > 0.0),
        interval_mass,
        0.0,
    )
    return left, right, left_density, right_density, interval_mass


def _sample_native_magnitudes_for_redshifts(
    rgi,
    rng,
    redshift,
    magnitude_grid,
    cosmo,
    *,
    reference_wavelength_angstrom,
    native_to_monochromatic_ab_offset,
    m2500_support,
    alpha_nu_lf_conversion_mean,
    alpha_nu_lf_conversion_sigma,
    chunk_rows=MAGNITUDE_SAMPLE_CHUNK_ROWS,
):
    """Draw native magnitudes from the support-weighted conditional LF."""

    redshift = np.asarray(redshift, dtype=float)
    sampled_chunks = []
    for start in range(0, redshift.size, int(chunk_rows)):
        stop = min(start + int(chunk_rows), redshift.size)
        redshift_chunk = redshift[start:stop]
        distance_modulus = cosmo.distmod(redshift_chunk).value
        (
            left,
            right,
            left_density,
            right_density,
            interval_mass,
        ) = _native_magnitude_interval_components(
            rgi,
            redshift_chunk,
            magnitude_grid,
            distance_modulus,
            reference_wavelength_angstrom=reference_wavelength_angstrom,
            native_to_monochromatic_ab_offset=(
                native_to_monochromatic_ab_offset
            ),
            m2500_support=m2500_support,
            alpha_nu_lf_conversion_mean=(
                alpha_nu_lf_conversion_mean
            ),
            alpha_nu_lf_conversion_sigma=(
                alpha_nu_lf_conversion_sigma
            ),
        )
        row_mass = np.sum(interval_mass, axis=1)
        if np.any(~np.isfinite(row_mass)) or np.any(row_mass <= 0.0):
            raise RuntimeError(
                "Sampled redshift has zero support-weighted LF density."
            )
        mass_draw = rng.random(redshift_chunk.size) * row_mass
        cumulative_mass = np.cumsum(interval_mass, axis=1)
        interval_index = np.sum(
            cumulative_mass < mass_draw[:, None],
            axis=1,
        )
        interval_index = np.clip(
            interval_index,
            0,
            interval_mass.shape[1] - 1,
        )
        rows = np.arange(redshift_chunk.size)
        cumulative_lower = np.zeros(redshift_chunk.size, dtype=float)
        has_lower = interval_index > 0
        cumulative_lower[has_lower] = cumulative_mass[
            rows[has_lower],
            interval_index[has_lower] - 1,
        ]
        selected_mass = interval_mass[rows, interval_index]
        quantile = np.clip(
            (mass_draw - cumulative_lower) / selected_mass,
            0.0,
            1.0,
        )
        selected_left = left[rows, interval_index]
        selected_right = right[rows, interval_index]
        fraction = _invert_linear_density_quantile(
            left_density[rows, interval_index],
            right_density[rows, interval_index],
            quantile,
        )
        sampled_chunks.append(
            selected_left + fraction * (selected_right - selected_left)
        )
    if not sampled_chunks:
        return np.empty(0, dtype=float)
    return np.concatenate(sampled_chunks)


def mock_m_per_zbin(
    phi_log10,
    m_grid,
    z_bins,
    area_deg2,
    alpha_nu_lf_conversion,
    dalpha_nu_lf_conversion,
    cosmo,
    *,
    z_res=512,
    m_scatter=0.0,
    kcorr_zref=2.0,
    reference_wavelength_angstrom=None,
    native_to_monochromatic_ab_offset=0.0,
    m2500_support=None,
    z_range=None,
    completeness=None,
    m_lim=None,
    thinning_probability=1.0,
    rng=None,
    return_z=False,
    return_global=False,
    return_alpha_nu_lf_conversion=False,
    verbose=False,
    progress=False,
):
    rng = np.random.default_rng() if rng is None else rng
    thinning_probability = float(thinning_probability)
    if not np.isfinite(thinning_probability) or not (0.0 < thinning_probability <= 1.0):
        raise ValueError(
            "thinning_probability must be finite and in (0, 1], "
            f"got {thinning_probability}."
        )

    m_grid = np.asarray(m_grid, dtype=float)
    if m_grid.ndim != 1 or m_grid.size < 2 or np.any(~np.isfinite(m_grid)):
        raise ValueError("m_grid must contain at least two finite values.")
    order = np.argsort(m_grid)
    m_grid = m_grid[order]
    n_mag = len(m_grid)

    alpha_nu_lf_conversion = float(alpha_nu_lf_conversion)
    dalpha_nu_lf_conversion = float(abs(dalpha_nu_lf_conversion))
    if not np.isfinite(alpha_nu_lf_conversion) or not np.isfinite(
        dalpha_nu_lf_conversion
    ):
        raise ValueError(
            "alpha_nu_lf_conversion and dalpha_nu_lf_conversion must be "
            "finite."
        )
    if reference_wavelength_angstrom is None:
        reference_wavelength_angstrom = (
            7480.0 / (1.0 + float(kcorr_zref))
            if kcorr_zref is not None
            else 2500.0
        )
    reference_wavelength_angstrom = float(reference_wavelength_angstrom)
    native_to_monochromatic_ab_offset = float(
        native_to_monochromatic_ab_offset
    )
    if (
        not np.isfinite(reference_wavelength_angstrom)
        or reference_wavelength_angstrom <= 0.0
    ):
        raise ValueError("reference_wavelength_angstrom must be positive.")
    if not np.isfinite(native_to_monochromatic_ab_offset):
        raise ValueError("native_to_monochromatic_ab_offset must be finite.")
    if m2500_support is not None:
        m2500_support = tuple(float(value) for value in m2500_support)
        if (
            len(m2500_support) != 2
            or not np.all(np.isfinite(m2500_support))
            or m2500_support[0] >= m2500_support[1]
        ):
            raise ValueError(
                "m2500_support must contain two finite increasing values."
            )
        if m_scatter > 0.0:
            raise ValueError(
                "m_scatter cannot be combined with exact m2500_support; "
                "the support-weighted Poisson intensity assumes zero added "
                "post-selection magnitude scatter."
            )

    z_bins = np.asarray(z_bins, dtype=float)
    z_mids = 0.5 * (z_bins[:-1] + z_bins[1:])

    phi = np.asarray(phi_log10, dtype=float)
    if phi.shape == (len(z_mids), len(order)):
        z_support = z_mids
    elif phi.shape == (len(z_bins), len(order)):
        z_support = z_bins
    elif phi.shape == (len(order), len(z_mids)):
        phi = phi.T
        z_support = z_mids
    elif phi.shape == (len(order), len(z_bins)):
        phi = phi.T
        z_support = z_bins
    else:
        raise ValueError("phi_log10 has incompatible shape for the supplied z and magnitude grids.")

    phi = phi[:, order]
    rgi = RegularGridInterpolator(
        (z_support, m_grid),
        phi,
        method="linear",
        bounds_error=False,
        fill_value=-np.inf,
    )

    sampling_z_edges = _restricted_redshift_edges(z_bins, z_range)
    area_sr = area_deg2 * (np.pi / 180.0) ** 2
    per_z_m = []
    per_z_m_rest = []
    per_z = []
    per_z_alpha_nu_lf_conversion = []
    nexp_per_bin = np.zeros(len(sampling_z_edges) - 1)
    nsel_per_bin = np.zeros(len(sampling_z_edges) - 1, dtype=int)

    redshift_intervals = list(zip(sampling_z_edges[:-1], sampling_z_edges[1:]))
    progress_bar = tqdm(
        redshift_intervals,
        desc="Sampling completeness mock",
        unit="z-bin",
        disable=not progress,
        mininterval=1.0,
        dynamic_ncols=True,
    )
    for i, (z1, z2) in enumerate(progress_bar):
        z = np.linspace(z1, z2, z_res)
        dvdz = cosmo.differential_comoving_volume(z).to_value(u.Mpc**3 / u.sr) * area_sr
        if m2500_support is None:
            nodes = np.broadcast_to(m_grid[None, :], (z.size, n_mag))
            phi_zm = _evaluate_lf_density(rgi, z, nodes)
            phi_int_z = np.trapezoid(phi_zm, x=m_grid, axis=1)
        else:
            distance_modulus_z = cosmo.distmod(z).value
            *_, magnitude_interval_mass = _native_magnitude_interval_components(
                rgi,
                z,
                m_grid,
                distance_modulus_z,
                reference_wavelength_angstrom=reference_wavelength_angstrom,
                native_to_monochromatic_ab_offset=(
                    native_to_monochromatic_ab_offset
                ),
                m2500_support=m2500_support,
                alpha_nu_lf_conversion_mean=alpha_nu_lf_conversion,
                alpha_nu_lf_conversion_sigma=dalpha_nu_lf_conversion,
            )
            phi_int_z = np.sum(magnitude_interval_mass, axis=1)

        nexp = np.trapezoid(phi_int_z * dvdz, z)
        nexp_per_bin[i] = nexp
        if not np.isfinite(nexp) or nexp <= 0.0:
            per_z_m.append(np.empty(0, dtype=float))
            per_z_m_rest.append(np.empty(0, dtype=float))
            per_z.append(np.empty(0, dtype=float))
            per_z_alpha_nu_lf_conversion.append(np.empty(0, dtype=float))
            continue
        n_draw = rng.poisson(nexp * thinning_probability)
        if n_draw == 0:
            per_z_m.append(np.empty(0, dtype=float))
            per_z_m_rest.append(np.empty(0, dtype=float))
            per_z.append(np.empty(0, dtype=float))
            per_z_alpha_nu_lf_conversion.append(np.empty(0, dtype=float))
            continue

        wz = phi_int_z * dvdz
        z_samp = _sample_piecewise_linear_1d(rng, z, wz, n_draw)

        if m2500_support is None:
            # Preserve the legacy no-support behavior using the LF grid as
            # piecewise-linear magnitude density, now in bounded chunks.
            m_abs = _sample_native_magnitudes_for_redshifts(
                rgi,
                rng,
                z_samp,
                m_grid,
                cosmo,
                reference_wavelength_angstrom=reference_wavelength_angstrom,
                native_to_monochromatic_ab_offset=(
                    native_to_monochromatic_ab_offset
                ),
                m2500_support=(-np.inf, np.inf),
                alpha_nu_lf_conversion_mean=alpha_nu_lf_conversion,
                alpha_nu_lf_conversion_sigma=0.0,
            )
        else:
            m_abs = _sample_native_magnitudes_for_redshifts(
                rgi,
                rng,
                z_samp,
                m_grid,
                cosmo,
                reference_wavelength_angstrom=reference_wavelength_angstrom,
                native_to_monochromatic_ab_offset=(
                    native_to_monochromatic_ab_offset
                ),
                m2500_support=m2500_support,
                alpha_nu_lf_conversion_mean=alpha_nu_lf_conversion,
                alpha_nu_lf_conversion_sigma=dalpha_nu_lf_conversion,
            )

        dm_s = cosmo.distmod(z_samp).value
        if m2500_support is None:
            alpha_nu_lf_conversion_samp = alpha_nu_lf_conversion + rng.normal(
                0.0,
                dalpha_nu_lf_conversion,
                size=z_samp.shape,
            )
        else:
            alpha_nu_lf_conversion_samp = (
                sample_alpha_nu_lf_conversion_conditional_on_m2500_support(
                    rng,
                    m_abs,
                    dm_s,
                    reference_wavelength_angstrom=(
                        reference_wavelength_angstrom
                    ),
                    m2500_support=m2500_support,
                    alpha_nu_lf_conversion_mean=alpha_nu_lf_conversion,
                    alpha_nu_lf_conversion_sigma=(
                        dalpha_nu_lf_conversion
                    ),
                    native_to_monochromatic_ab_offset=(
                        native_to_monochromatic_ab_offset
                    ),
                )
            )
        if kcorr_zref is None:
            kcorr = (
                -2.5
                * (1.0 + alpha_nu_lf_conversion_samp)
                * np.log10(1.0 + z_samp)
            )
            m_i_reference = (
                m_abs + native_to_monochromatic_ab_offset
            )
        else:
            kcorr = (
                -2.5
                * (1.0 + alpha_nu_lf_conversion_samp)
                * np.log10((1.0 + z_samp) / (1.0 + kcorr_zref))
            )
            lambda_i_reference = 7480.0 / (1.0 + float(kcorr_zref))
            m_i_reference = native_absolute_magnitude_to_m2500(
                m_abs,
                alpha_nu_lf_conversion_samp,
                reference_wavelength_angstrom,
                native_to_monochromatic_ab_offset,
            ) - 2.5 * alpha_nu_lf_conversion_samp * np.log10(
                2500.0 / lambda_i_reference
            ) - 2.5 * np.log10(1.0 + float(kcorr_zref))

        m_obs = m_i_reference + dm_s + kcorr
        m_2500_abs = native_absolute_magnitude_to_m2500(
            m_abs,
            alpha_nu_lf_conversion_samp,
            reference_wavelength_angstrom,
            native_to_monochromatic_ab_offset,
        )
        m_2500_obs = m_2500_abs + dm_s

        if m_scatter > 0:
            m_obs = m_obs + rng.normal(0.0, m_scatter, size=m_obs.size)
            m_2500_obs = m_2500_obs + rng.normal(0.0, m_scatter, size=m_2500_obs.size)

        if m2500_support is not None and m_2500_obs.size > 0:
            keep = (
                (m_2500_obs >= m2500_support[0])
                & (m_2500_obs <= m2500_support[1])
            )
            m_obs = m_obs[keep]
            m_2500_obs = m_2500_obs[keep]
            z_samp = z_samp[keep]
            alpha_nu_lf_conversion_samp = (
                alpha_nu_lf_conversion_samp[keep]
            )

        if m_lim is not None and m_obs.size > 0:
            keep = m_obs < m_lim
            m_obs = m_obs[keep]
            m_2500_obs = m_2500_obs[keep]
            z_samp = z_samp[keep]
            alpha_nu_lf_conversion_samp = (
                alpha_nu_lf_conversion_samp[keep]
            )

        if completeness is not None and m_obs.size > 0:
            p = np.clip(completeness(m_obs, z_samp), 0.0, 1.0)
            keep = rng.random(m_obs.size) < p
            m_obs = m_obs[keep]
            m_2500_obs = m_2500_obs[keep]
            z_samp = z_samp[keep]
            alpha_nu_lf_conversion_samp = (
                alpha_nu_lf_conversion_samp[keep]
            )

        per_z_m.append(m_obs)
        per_z_m_rest.append(m_2500_obs)
        per_z.append(z_samp)
        per_z_alpha_nu_lf_conversion.append(
            alpha_nu_lf_conversion_samp
        )
        nsel_per_bin[i] = m_obs.size
        progress_bar.set_postfix(
            z=f"{z1:.2f}-{z2:.2f}",
            saved=f"{int(np.sum(nsel_per_bin)):,}",
            refresh=False,
        )

    if not (return_z or return_global):
        out = (per_z_m, nexp_per_bin)
        if return_alpha_nu_lf_conversion:
            out = out + (per_z_alpha_nu_lf_conversion,)
        return out

    out = (per_z_m, nexp_per_bin, per_z, nsel_per_bin)
    if return_global:
        nonempty = [i for i, arr in enumerate(per_z_m) if len(arr)]
        if nonempty:
            z_all = np.concatenate([per_z[i] for i in nonempty])
            m_all = np.concatenate([per_z_m[i] for i in nonempty])
            m_rest_all = np.concatenate([per_z_m_rest[i] for i in nonempty])
            bin_index = np.concatenate([np.full(len(per_z_m[i]), i, dtype=int) for i in nonempty])
            alpha_nu_lf_conversion_all = np.concatenate(
                [per_z_alpha_nu_lf_conversion[i] for i in nonempty]
            )
        else:
            z_all = np.empty(0, dtype=float)
            m_all = np.empty(0, dtype=float)
            m_rest_all = np.empty(0, dtype=float)
            bin_index = np.empty(0, dtype=int)
            alpha_nu_lf_conversion_all = np.empty(0, dtype=float)
        out = out + (z_all, m_all, m_rest_all, bin_index)
        if return_alpha_nu_lf_conversion:
            out = out + (alpha_nu_lf_conversion_all,)
    elif return_alpha_nu_lf_conversion:
        out = out + (per_z_alpha_nu_lf_conversion,)
    return out


def save_mock_catalog(
    output_path,
    z_all,
    m_all,
    m_2500_all,
    m_limit=None,
    *,
    alpha_nu_lf_conversion_all,
    thinning_probability=1.0,
    rng=None,
    area_deg2=None,
    alpha_nu_lf_conversion_parent_mean=None,
    alpha_nu_lf_conversion_parent_sigma=None,
    lf_model=None,
    shen_lf_mode=None,
    lf_metadata=None,
    reference_wavelength_angstrom=None,
    native_to_monochromatic_ab_offset=None,
    m2500_support=None,
    z_range=None,
    completeness_magnitude_state=None,
    lf_magnitude_state_match=None,
    target_area_deg2=None,
    proposal_area_deg2=None,
    effective_sampled_area_deg2=None,
    mock_count_scale=None,
    requested_oversample=None,
    realized_oversample=None,
    expected_full_sky_count=None,
    random_seed=None,
    config_hash=None,
):
    """Save a completeness mock with explicit LF-conversion-slope semantics.

    ``alpha_nu_lf_conversion`` is a population-level continuum proxy used for
    LF wavelength conversion and mock K-correction. For empirical LFs it
    retains the source LF's implicit attenuation and is never an intrinsic
    JAXSedFit continuum parameter.
    """

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    thinning_probability = float(thinning_probability)
    if not np.isfinite(thinning_probability) or not (0.0 < thinning_probability <= 1.0):
        raise ValueError(
            "thinning_probability must be finite and in (0, 1], "
            f"got {thinning_probability}."
        )
    mask = np.ones_like(z_all, dtype=bool)
    if m_limit is not None:
        mask &= m_all < m_limit
    if m2500_support is not None:
        m2500_support = tuple(float(value) for value in m2500_support)
        if len(m2500_support) != 2 or m2500_support[0] >= m2500_support[1]:
            raise ValueError("m2500_support must contain two increasing values.")
        finite_m2500 = np.isfinite(m_2500_all)
        inside_m2500 = (
            finite_m2500
            & (np.asarray(m_2500_all) >= m2500_support[0])
            & (np.asarray(m_2500_all) <= m2500_support[1])
        )
        if not np.all(inside_m2500):
            raise ValueError(
                "Completeness mock contains rows outside its declared m_2500 "
                f"support {m2500_support}: count={np.count_nonzero(~inside_m2500)}."
            )
    if z_range is not None:
        z_range = tuple(float(value) for value in z_range)
        if len(z_range) != 2 or z_range[0] >= z_range[1]:
            raise ValueError("z_range must contain two increasing values.")
        finite_z = np.isfinite(z_all)
        inside_z = (
            finite_z
            & (np.asarray(z_all) >= z_range[0])
            & (np.asarray(z_all) <= z_range[1])
        )
        if not np.all(inside_z):
            raise ValueError(
                "Completeness mock contains rows outside its declared redshift "
                f"support {z_range}: count={np.count_nonzero(~inside_z)}."
            )
    alpha_nu_lf_conversion_all = np.asarray(
        alpha_nu_lf_conversion_all,
        dtype=float,
    )
    if alpha_nu_lf_conversion_all.shape != np.shape(z_all):
        raise ValueError(
            "alpha_nu_lf_conversion_all must have the same shape as "
            f"z_all; got {alpha_nu_lf_conversion_all.shape} and "
            f"{np.shape(z_all)}."
        )
    if np.any(~np.isfinite(alpha_nu_lf_conversion_all)):
        raise ValueError("alpha_nu_lf_conversion_all must be finite.")
    n_before_thin = int(np.count_nonzero(mask))
    n_after_thin = int(np.count_nonzero(mask))
    if mock_count_scale is None:
        mock_count_scale = 1.0 / thinning_probability
    mock_count_scale = float(mock_count_scale)
    if not np.isfinite(mock_count_scale) or mock_count_scale <= 0.0:
        raise ValueError("mock_count_scale must be finite and positive.")

    area_metadata = (
        target_area_deg2,
        proposal_area_deg2,
        effective_sampled_area_deg2,
    )
    if any(value is not None for value in area_metadata):
        if any(value is None for value in area_metadata):
            raise ValueError(
                "target, proposal, and effective sampled areas must be supplied together."
            )
        target_area_deg2 = float(target_area_deg2)
        proposal_area_deg2 = float(proposal_area_deg2)
        effective_sampled_area_deg2 = float(effective_sampled_area_deg2)
        expected_scale = target_area_deg2 / effective_sampled_area_deg2
        if not np.isclose(mock_count_scale, expected_scale, rtol=1e-10, atol=0.0):
            raise ValueError(
                "mock_count_scale is inconsistent with target/effective areas: "
                f"{mock_count_scale} versus {expected_scale}."
            )

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(fd)
    try:
        with h5py.File(temporary_name, "w") as h5file:
            h5file.create_dataset("z", data=z_all[mask], compression="gzip", shuffle=True)
            h5file.create_dataset("apparent_mag_i", data=m_all[mask], compression="gzip", shuffle=True)
            # This is an apparent-magnitude proxy at rest-frame 2500 A, not rest-frame i-band.
            h5file.create_dataset("apparent_mag_2500", data=m_2500_all[mask], compression="gzip", shuffle=True)
            # Keep the legacy key so existing completeness readers do not break.
            h5file.create_dataset("apparent_mag_i_rest", data=m_2500_all[mask], compression="gzip", shuffle=True)
            alpha_saved = alpha_nu_lf_conversion_all[mask]
            h5file.create_dataset(
                "alpha_nu_lf_conversion",
                data=alpha_saved,
                compression="gzip",
                shuffle=True,
            )
            h5file.attrs["alpha_nu_lf_conversion_mean"] = float(
                np.mean(alpha_saved)
            )
            h5file.attrs["alpha_nu_lf_conversion_sigma"] = (
                float(np.std(alpha_saved, ddof=1))
                if alpha_saved.size > 1
                else 0.0
            )
            h5file.attrs["completeness_mock_schema_version"] = COMPLETENESS_MOCK_SCHEMA_VERSION
            h5file.attrs["thinning_probability"] = thinning_probability
            h5file.attrs["mock_count_scale"] = mock_count_scale
            h5file.attrs["stored_object_count"] = n_after_thin
            if (
                alpha_nu_lf_conversion_parent_mean is not None
                and np.isfinite(alpha_nu_lf_conversion_parent_mean)
            ):
                h5file.attrs[
                    "alpha_nu_lf_conversion_parent_mean"
                ] = float(alpha_nu_lf_conversion_parent_mean)
            if (
                alpha_nu_lf_conversion_parent_sigma is not None
                and np.isfinite(alpha_nu_lf_conversion_parent_sigma)
            ):
                h5file.attrs[
                    "alpha_nu_lf_conversion_parent_sigma"
                ] = float(abs(alpha_nu_lf_conversion_parent_sigma))
            if area_deg2 is not None and np.isfinite(area_deg2):
                h5file.attrs["area_deg2"] = float(area_deg2)
            if target_area_deg2 is not None:
                h5file.attrs["target_area_deg2"] = target_area_deg2
                h5file.attrs["proposal_area_deg2"] = proposal_area_deg2
                h5file.attrs["effective_sampled_area_deg2"] = effective_sampled_area_deg2
            if requested_oversample is not None:
                h5file.attrs["requested_oversample"] = float(requested_oversample)
            if realized_oversample is not None:
                h5file.attrs["realized_oversample"] = float(realized_oversample)
            if expected_full_sky_count is not None:
                h5file.attrs["expected_full_sky_count"] = float(expected_full_sky_count)
            if random_seed is not None:
                h5file.attrs["random_seed"] = int(random_seed)
            if config_hash is not None:
                h5file.attrs["config_hash"] = str(config_hash)
            if lf_model is not None:
                h5file.attrs["lf_model"] = str(lf_model)
            if shen_lf_mode is not None:
                h5file.attrs["shen_lf_mode"] = normalize_shen_lf_mode(shen_lf_mode)
            h5file.attrs["lf_semantics_version"] = COMPLETENESS_MOCK_SEMANTICS_VERSION
            h5file.attrs["lf_conversion_slope_parameter"] = (
                LF_CONVERSION_SLOPE_PARAMETER
            )
            h5file.attrs["lf_conversion_slope_convention"] = (
                LF_CONVERSION_SLOPE_CONVENTION
            )
            h5file.attrs["lf_conversion_slope_purpose"] = (
                "native_lf_wavelength_conversion_and_mock_k_correction"
            )
            h5file.attrs["lf_conversion_slope_jaxsedfit_relationship"] = (
                "independent_population_proxy_not_intrinsic_jaxsedfit_slope"
            )
            if lf_metadata is not None:
                metadata_json = json.dumps(
                    lf_metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                h5file.attrs["lf_metadata_json"] = metadata_json
                semantics = dict(lf_metadata.get("semantics", {}))
                for key in (
                    "population_scope",
                    "type1_definition",
                    "uv_attenuation_state",
                    "galactic_foreground_treatment",
                    "internal_dust_treatment",
                    "coordinate_conversion",
                ):
                    if key in semantics:
                        h5file.attrs[f"lf_{key}"] = str(semantics[key])
                for key in (
                    "lf_conversion_slope_parameter",
                    "lf_conversion_slope_convention",
                    "lf_conversion_continuum_state",
                    "lf_conversion_dust_operation",
                    "lf_conversion_internal_dust_correction_applied",
                    "lf_conversion_is_jaxsedfit_intrinsic_slope",
                ):
                    if key in semantics:
                        h5file.attrs[key] = semantics[key]
                if "conversion_is_approximate" in semantics:
                    h5file.attrs["lf_conversion_is_approximate"] = bool(
                        semantics["conversion_is_approximate"]
                    )
            if reference_wavelength_angstrom is not None:
                h5file.attrs["lf_native_reference_wavelength_angstrom"] = float(
                    reference_wavelength_angstrom
                )
            if native_to_monochromatic_ab_offset is not None:
                h5file.attrs[
                    "lf_native_to_monochromatic_ab_offset"
                ] = float(native_to_monochromatic_ab_offset)
            if m2500_support is not None:
                h5file.attrs["m2500_support_min"] = m2500_support[0]
                h5file.attrs["m2500_support_max"] = m2500_support[1]
            if z_range is not None:
                h5file.attrs["mock_redshift_min"] = z_range[0]
                h5file.attrs["mock_redshift_max"] = z_range[1]
            if completeness_magnitude_state is not None:
                h5file.attrs["completeness_magnitude_state"] = str(
                    completeness_magnitude_state
                )
            if lf_magnitude_state_match is not None:
                h5file.attrs["lf_magnitude_state_match"] = bool(
                    lf_magnitude_state_match
                )
        os.replace(temporary_name, output_path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    print(
        "Saved mock catalog with "
        f"{n_after_thin} / {n_before_thin} sources after support/cut selection "
        f"(p_keep={thinning_probability:.4g}, mock_count_scale={mock_count_scale:.4g})"
    )


def plot_mock_catalog(z_all, m_values, plot_path, title, ylabel, bin_index):
    plot_path = Path(plot_path).expanduser().resolve()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(z_all, m_values, c=bin_index, cmap="viridis", s=3, linewidths=0, rasterized=True)
    fig.colorbar(sc, ax=ax, label="Redshift Bin Index")
    ax.set_xlabel("Redshift")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, 5)
    ax.set_ylim(15, 30)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Generate mock completeness catalogs from Shen, Wang, Palanque, "
            "Kulkarni, or Ananna luminosity functions."
        )
    )
    parser.add_argument(
        "--lf-model",
        choices=["shen", *EMPIRICAL_LF_MODEL_IDS, "ananna"],
        required=True,
        help="Luminosity-function model to use.",
    )
    parser.add_argument("--output", required=True, help="Output HDF5 file path.")
    parser.add_argument(
        "--area-deg2",
        type=float,
        default=None,
        help=(
            "Survey area in deg^2. Defaults: 5 for Shen/empirical LFs, "
            "50 for Ananna."
        ),
    )
    parser.add_argument(
        "--target-area-deg2",
        type=float,
        default=None,
        help="Observed footprint area to which a larger proposal mock is scaled.",
    )
    parser.add_argument(
        "--proposal-area-deg2",
        type=float,
        default=FULL_SKY_AREA_DEG2,
        help="Large proposal area used with --target-area-deg2.",
    )
    parser.add_argument(
        "--oversample",
        type=float,
        default=4.0,
        help="Effective sampled-area multiple used with --target-area-deg2.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=2_000_000,
        help="Maximum saved rows for an area-scaled proposal mock.",
    )
    parser.add_argument(
        "--alpha-nu-lf-conversion",
        type=float,
        default=-0.5,
        help=(
            "Mean population-level f_nu slope used only to convert the LF "
            "reference wavelength and compute mock K-corrections. Empirical "
            "LF attenuation is retained; this is not an intrinsic JAXSedFit "
            "disk slope."
        ),
    )
    parser.add_argument(
        "--dalpha-nu-lf-conversion",
        type=float,
        default=0.3,
        help="Scatter in the population-level LF-conversion slope.",
    )
    parser.add_argument(
        "--m-limit",
        type=float,
        default=None,
        help=(
            "Optional legacy observed-band magnitude cut. The primary mock "
            "support is controlled by --m2500-min/--m2500-max."
        ),
    )
    parser.add_argument(
        "--m2500-min",
        type=float,
        default=DEFAULT_M2500_SUPPORT[0],
        help="Inclusive lower apparent-m2500 support bound.",
    )
    parser.add_argument(
        "--m2500-max",
        type=float,
        default=DEFAULT_M2500_SUPPORT[1],
        help="Inclusive upper apparent-m2500 support bound.",
    )
    parser.add_argument(
        "--completeness-magnitude",
        choices=("attenuated", "dereddened"),
        default="attenuated",
        help=(
            "Magnitude state represented by the generated completeness mock; "
            "used for LF compatibility warnings and provenance."
        ),
    )
    parser.add_argument(
        "--z-range",
        type=float,
        nargs=2,
        default=(0.44, 3.16),
        metavar=("Z_MIN", "Z_MAX"),
        help="Redshift support to generate (default: 0.44 3.16).",
    )
    parser.add_argument("--m-scatter", type=float, default=0.0, help="Additional Gaussian apparent-magnitude scatter.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--z-res", type=int, default=512, help="Redshift resolution inside each z bin.")
    parser.add_argument("--plot", action="store_true", help="Save a diagnostic scatter plot.")
    parser.add_argument("--plot-path", default=None, help="Optional plot output path.")
    parser.add_argument("--plot-rest", action="store_true", help="Plot rest-frame 2500A apparent magnitudes instead of observed survey-band magnitudes.")
    parser.add_argument(
        "--shen-pubtools-path",
        default=None,
        help="Path to the Shen QLF pubtools directory. If omitted, use SHEN_PUBTOOLS_PATH or common repo-relative locations.",
    )
    parser.add_argument(
        "--shen-lf-mode",
        choices=SHEN_LF_MODES,
        default=SHEN_DEFAULT_LF_MODE,
        help=(
            "Shen population/attenuation mode: the existing all-N_H observed "
            "2500-A LF, an intrinsic N_H<1e22 Type-1 LF, or its attenuated "
            "selection-space counterpart."
        ),
    )
    parser.add_argument(
        "--ananna-xlf-path",
        default=None,
        help="Path to the gzipped Ananna XLF matrix. If omitted, use ANANNA_XLF_PATH or common local locations.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-bin sampling diagnostics.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Shen-LF and redshift-bin progress bars.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    z_range = tuple(float(value) for value in args.z_range)
    m2500_support = (float(args.m2500_min), float(args.m2500_max))
    if z_range[0] < 0.0 or z_range[0] >= z_range[1]:
        raise ValueError("--z-range must be increasing and non-negative.")
    if (
        not np.all(np.isfinite(m2500_support))
        or m2500_support[0] >= m2500_support[1]
    ):
        raise ValueError("--m2500-min/--m2500-max must be finite and increasing.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    rng = np.random.default_rng(args.seed)

    lf_metadata = None
    reference_wavelength_angstrom = None
    native_to_monochromatic_ab_offset = 0.0
    lf_magnitude_state_match = None
    if args.lf_model != "ananna":
        lf_grid = build_completeness_lf(
            args.lf_model,
            shen_lf_mode=args.shen_lf_mode,
            z_range=z_range,
            shen_pubtools_path=args.shen_pubtools_path,
            target_cosmology=COSMO,
            progress=not args.no_progress,
        )
        phi_log10 = lf_grid.phi_log10
        m_grid = lf_grid.native_magnitude_grid
        z_bins = lf_grid.redshift_grid
        reference_wavelength_angstrom = (
            lf_grid.reference_wavelength_angstrom
        )
        native_to_monochromatic_ab_offset = (
            lf_grid.native_to_monochromatic_ab_offset
        )
        lf_metadata = completeness_lf_static_metadata(
            args.lf_model,
            shen_lf_mode=args.shen_lf_mode,
        )
        lf_metadata = json.loads(json.dumps(lf_metadata))
        lf_metadata["target_cosmology"] = {
            "H0": float(COSMO.H0.value),
            "Om0": float(COSMO.Om0),
        }
        lf_metadata["requested_redshift_range"] = list(z_range)
        lf_metadata["requested_m2500_support"] = list(m2500_support)
        calibration_min, calibration_max = lf_metadata[
            "calibration_redshift_range"
        ]
        lf_metadata["redshift_extrapolation"] = bool(
            z_range[0] < calibration_min or z_range[1] > calibration_max
        )
        (
            lf_magnitude_state_match,
            expected_magnitude_state,
        ) = completeness_lf_magnitude_state_match(
            args.lf_model,
            args.completeness_magnitude,
            shen_lf_mode=args.shen_lf_mode,
        )
        if not lf_magnitude_state_match:
            descriptor = (
                f"shen/{args.shen_lf_mode}"
                if args.lf_model == "shen"
                else args.lf_model
            )
            print(
                "[WARNING] LF/completeness magnitude-state mismatch: "
                f"{descriptor} expects {expected_magnitude_state!r}, but "
                f"{args.completeness_magnitude!r} was requested."
            )
        area_deg2 = 5.0 if args.area_deg2 is None else args.area_deg2
    else:
        if args.shen_lf_mode != SHEN_DEFAULT_LF_MODE:
            raise ValueError("--shen-lf-mode is only valid with --lf-model shen.")
        phi_log10, m_grid, z_bins = build_ananna_lf(args.ananna_xlf_path)
        reference_wavelength_angstrom = 7480.0 / 3.0
        if z_range[0] < z_bins[0] or z_range[1] > z_bins[-1]:
            raise ValueError(
                "--z-range lies outside the Ananna LF support "
                f"[{z_bins[0]}, {z_bins[-1]}]."
            )
        lf_metadata = {
            "model_id": "ananna",
            "formula_version": "legacy_ananna_xlf_to_bolometric_proxy",
            "native_magnitude_name": "legacy_bolometric_proxy_magnitude",
            "reference_wavelength_angstrom": reference_wavelength_angstrom,
            "native_to_monochromatic_ab_offset": 0.0,
            "source_cosmology": {"H0": 70.0, "Om0": 0.3},
            "target_cosmology": {"H0": 70.0, "Om0": 0.3},
            "calibration_redshift_range": [float(z_bins[0]), float(z_bins[-1])],
            "requested_redshift_range": list(z_range),
            "requested_m2500_support": list(m2500_support),
            "redshift_extrapolation": False,
            "semantics": {
                **_lf_conversion_slope_semantics(
                    "legacy_proxy_unspecified_attenuation_state"
                ),
                "population_scope": "legacy_ananna_nh20_22_proxy",
                "uv_attenuation_state": "legacy_bolometric_conversion_proxy",
                "coordinate_conversion": "legacy_i_z2_to_M2500_proxy",
                "conversion_is_approximate": True,
            },
        }
        area_deg2 = 50.0 if args.area_deg2 is None else args.area_deg2

    lf_metadata["lf_conversion_slope_distribution"] = {
        "parameter": LF_CONVERSION_SLOPE_PARAMETER,
        "convention": LF_CONVERSION_SLOPE_CONVENTION,
        "parent_distribution": "normal",
        "parent_mean": float(args.alpha_nu_lf_conversion),
        "parent_sigma": float(abs(args.dalpha_nu_lf_conversion)),
        "support_conditioning": "conditional_on_declared_m2500_support",
        "internal_dust_operation": "none_preserve_lf_population_state",
        "jaxsedfit_relationship": (
            "independent_population_proxy_not_intrinsic_jaxsedfit_slope"
        ),
    }

    sampling_plan = None
    if args.target_area_deg2 is not None:
        if args.area_deg2 is not None:
            raise ValueError(
                "Use either --area-deg2 for a legacy equal-area mock or "
                "--target-area-deg2 for an area-scaled proposal mock."
            )
        if args.max_rows <= 0:
            raise ValueError("--max-rows must be positive.")
        sampling_plan = plan_area_scaled_mock_sampling(
            args.target_area_deg2,
            proposal_area_deg2=args.proposal_area_deg2,
            oversample=args.oversample,
        )
        area_deg2 = sampling_plan["proposal_area_deg2"]
        thinning_probability = sampling_plan["thinning_probability"]
    else:
        thinning_probability = 1.0

    (
        _,
        nexp,
        _,
        nsel,
        z_all,
        m_all,
        m_rest_all,
        bin_index,
        alpha_nu_lf_conversion_all,
    ) = mock_m_per_zbin(
        phi_log10,
        m_grid,
        z_bins,
        area_deg2,
        args.alpha_nu_lf_conversion,
        args.dalpha_nu_lf_conversion,
        COSMO,
        z_res=args.z_res,
        m_scatter=args.m_scatter,
        kcorr_zref=2.0,
        reference_wavelength_angstrom=reference_wavelength_angstrom,
        native_to_monochromatic_ab_offset=(
            native_to_monochromatic_ab_offset
        ),
        m2500_support=m2500_support,
        z_range=z_range,
        m_lim=args.m_limit,
        thinning_probability=thinning_probability,
        rng=rng,
        return_z=True,
        return_global=True,
        return_alpha_nu_lf_conversion=True,
        verbose=args.verbose,
        progress=not args.no_progress,
    )

    cap_probability = 1.0
    generated_count = len(z_all)
    if sampling_plan is not None and generated_count > args.max_rows:
        cap_probability = args.max_rows / generated_count
        selected = np.sort(
            rng.choice(generated_count, size=args.max_rows, replace=False)
        )
        z_all = z_all[selected]
        m_all = m_all[selected]
        m_rest_all = m_rest_all[selected]
        bin_index = bin_index[selected]
        alpha_nu_lf_conversion_all = (
            alpha_nu_lf_conversion_all[selected]
        )

    if sampling_plan is None:
        combined_probability = thinning_probability
        target_area = proposal_area = effective_area = None
        count_scale = None
        realized_oversample = None
    else:
        combined_probability = thinning_probability * cap_probability
        target_area = sampling_plan["target_area_deg2"]
        proposal_area = sampling_plan["proposal_area_deg2"]
        effective_area = proposal_area * combined_probability
        count_scale = target_area / effective_area
        realized_oversample = effective_area / target_area

    save_mock_catalog(
        args.output,
        z_all,
        m_all,
        m_rest_all,
        m_limit=args.m_limit,
        alpha_nu_lf_conversion_all=alpha_nu_lf_conversion_all,
        thinning_probability=combined_probability,
        area_deg2=(target_area if target_area is not None else area_deg2),
        target_area_deg2=target_area,
        proposal_area_deg2=proposal_area,
        effective_sampled_area_deg2=effective_area,
        mock_count_scale=count_scale,
        requested_oversample=(args.oversample if sampling_plan is not None else None),
        realized_oversample=realized_oversample,
        expected_full_sky_count=(
            float(np.sum(nexp) * FULL_SKY_AREA_DEG2 / area_deg2)
            if sampling_plan is not None
            else None
        ),
        random_seed=args.seed,
        alpha_nu_lf_conversion_parent_mean=(
            args.alpha_nu_lf_conversion
        ),
        alpha_nu_lf_conversion_parent_sigma=(
            args.dalpha_nu_lf_conversion
        ),
        lf_model=args.lf_model,
        shen_lf_mode=(args.shen_lf_mode if args.lf_model == "shen" else None),
        lf_metadata=lf_metadata,
        reference_wavelength_angstrom=reference_wavelength_angstrom,
        native_to_monochromatic_ab_offset=(
            native_to_monochromatic_ab_offset
        ),
        m2500_support=m2500_support,
        z_range=z_range,
        completeness_magnitude_state=args.completeness_magnitude,
        lf_magnitude_state_match=lf_magnitude_state_match,
    )
    print(f"Saved mock catalog to {args.output}")
    print(f"Generated {len(z_all)} total mock sources before save cut.")

    if args.plot:
        if args.plot_path is None:
            stem = Path(args.output).with_suffix("")
            mode_suffix = (
                f"_{args.shen_lf_mode}" if args.lf_model == "shen" else ""
            )
            plot_path = f"{stem}_{args.lf_model}{mode_suffix}.pdf"
        else:
            plot_path = args.plot_path
        if args.plot_rest:
            plot_mock_catalog(
                z_all,
                m_rest_all,
                plot_path,
                (
                    f"Mock Survey from {args.lf_model.title()} LF"
                    + (
                        f" ({args.shen_lf_mode})"
                        if args.lf_model == "shen"
                        else ""
                    )
                ),
                "Apparent Magnitude at 2500A",
                bin_index,
            )
        else:
            plot_mock_catalog(
                z_all,
                m_all,
                plot_path,
                (
                    f"Mock Survey from {args.lf_model.title()} LF"
                    + (
                        f" ({args.shen_lf_mode})"
                        if args.lf_model == "shen"
                        else ""
                    )
                ),
                "Observed-band Apparent Magnitude",
                bin_index,
            )
        print(f"Saved diagnostic plot to {plot_path}")


if __name__ == "__main__":
    main()
