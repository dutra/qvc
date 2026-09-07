"""Analytic empirical Type-1 quasar luminosity functions.

This module keeps the published luminosity-function conventions separate from
the Shen bolometric/attenuation machinery.  Its grids are expressed in each
paper's native absolute-magnitude convention, remapped to the requested
cosmology but not converted between rest wavelengths.  The reference
wavelength stored on :class:`LFGrid` lets the mock sampler perform that color
conversion after drawing a continuum slope.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable

import numpy as np
from astropy.cosmology import FlatLambdaCDM


WANG2026_TYPE1_LADE_A = "wang2026_type1_lade_a"
PALANQUE2016_PLE_LEDE = "palanque2016_ple_lede"
KULKARNI2019_TYPE1_MODEL1 = "kulkarni2019_type1_model1"
KULKARNI2019_TYPE1_MODEL2 = "kulkarni2019_type1_model2"
KULKARNI2019_TYPE1_MODEL3 = "kulkarni2019_type1_model3"
KULKARNI2019_TYPE1_MODEL_IDS = (
    KULKARNI2019_TYPE1_MODEL1,
    KULKARNI2019_TYPE1_MODEL2,
    KULKARNI2019_TYPE1_MODEL3,
)
EMPIRICAL_LF_MODEL_IDS = (
    WANG2026_TYPE1_LADE_A,
    PALANQUE2016_PLE_LEDE,
    *KULKARNI2019_TYPE1_MODEL_IDS,
)

WANG2026_SOURCE_COSMOLOGY = FlatLambdaCDM(
    H0=70.0,
    Om0=0.3,
    name="Wang et al. (2026)",
)
PALANQUE2016_SOURCE_COSMOLOGY = FlatLambdaCDM(
    H0=67.9,
    Om0=0.3065,
    name="Palanque-Delabrouille et al. (2016)",
)
KULKARNI2019_SOURCE_COSMOLOGY = FlatLambdaCDM(
    H0=70.0,
    Om0=0.3,
    name="Kulkarni et al. (2019)",
)

WANG2026_REFERENCE_WAVELENGTH_ANGSTROM = 1450.0
KULKARNI2019_REFERENCE_WAVELENGTH_ANGSTROM = 1450.0
# M_g(z=2) is referenced to the rest wavelength sampled by observed SDSS g at
# z=2.  The 4670-A effective wavelength follows the convention used alongside
# this LF in Croom et al. (2009).
PALANQUE2016_REFERENCE_WAVELENGTH_ANGSTROM = 4670.0 / 3.0
# M_g(z=2) is defined with K(z)-K(2), so at z=2 it equals g-DM.  A physical
# monochromatic AB absolute magnitude at the corresponding emitted wavelength
# instead equals g-DM+2.5*log10(1+2).  Apply this reference-redshift
# normalization before the continuum color conversion to 2500 A.
PALANQUE2016_NATIVE_TO_MONOCHROMATIC_AB_OFFSET = 2.5 * np.log10(3.0)


@dataclass(frozen=True, slots=True)
class ReddeningSemantics:
    """Dust/reddening interpretation attached to a published LF."""

    luminosity_semantics: str
    galactic_extinction: str
    internal_extinction: str
    selection_dust_treatment: str
    apply_additional_internal_extinction: bool


WANG2026_REDDENING_SEMANTICS = ReddeningSemantics(
    luminosity_semantics="empirical_observed_rest_uv_type1",
    galactic_extinction="corrected_in_inherited_k19_photometry",
    internal_extinction="not_corrected_or_explicitly_modeled",
    selection_dust_treatment="inherited_survey_specific_not_explicit",
    apply_additional_internal_extinction=False,
)
PALANQUE2016_REDDENING_SEMANTICS = ReddeningSemantics(
    luminosity_semantics="empirical_galactic_dereddened_selection_space_quasar",
    galactic_extinction="corrected_in_g_dered_using_schlegel1998",
    internal_extinction="not_corrected_or_explicitly_modeled",
    selection_dust_treatment="variability_selection_without_explicit_internal_dust_model",
    apply_additional_internal_extinction=False,
)
KULKARNI2019_REDDENING_SEMANTICS = ReddeningSemantics(
    luminosity_semantics=(
        "empirical_selection_corrected_galactic_dereddened_"
        "host_corrected_rest_uv_type1"
    ),
    galactic_extinction=(
        "corrected_in_input_psf_photometry_using_schlegel1998"
    ),
    internal_extinction="not_corrected_or_explicitly_modeled",
    selection_dust_treatment=(
        "heterogeneous_uv_optical_color_selection_without_explicit_"
        "internal_dust_model"
    ),
    apply_additional_internal_extinction=False,
)


def _readonly_float_array(values, *, name, ndim):
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional; got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    array.setflags(write=False)
    return array


def _readonly_log_density_array(values, *, name):
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional; got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if np.any(np.isnan(array)) or np.any(np.isposinf(array)):
        raise ValueError(
            f"{name} can contain finite log densities or -inf for zero density, "
            "but not NaN or +inf."
        )
    array.setflags(write=False)
    return array


def _validate_flat_cosmology(cosmology, *, name):
    if not isinstance(cosmology, FlatLambdaCDM):
        raise TypeError(f"{name} must be an astropy.cosmology.FlatLambdaCDM instance.")
    return cosmology


@dataclass(frozen=True, slots=True, eq=False)
class LFGrid:
    """Immutable analytic LF evaluated on a native-magnitude/redshift grid.

    ``phi_log10`` is log10(dN/dV/dM) in the *target* cosmology,
    with shape ``(redshift_grid.size, native_magnitude_grid.size)`` and units
    of comoving Mpc^-3 mag^-1.  Magnitudes retain the paper's
    native wavelength convention; they are not yet converted to M_2500.
    """

    model_id: str
    phi_log10: np.ndarray
    native_magnitude_grid: np.ndarray
    redshift_grid: np.ndarray
    reference_wavelength_angstrom: float
    native_to_monochromatic_ab_offset: float
    native_magnitude_name: str
    source_cosmology: FlatLambdaCDM
    target_cosmology: FlatLambdaCDM
    calibration_redshift_range: tuple[float, float]
    formula_version: str
    reddening_semantics: ReddeningSemantics

    def __post_init__(self):
        magnitudes = _readonly_float_array(
            self.native_magnitude_grid,
            name="native_magnitude_grid",
            ndim=1,
        )
        redshifts = _readonly_float_array(
            self.redshift_grid,
            name="redshift_grid",
            ndim=1,
        )
        phi_log10 = _readonly_log_density_array(
            self.phi_log10,
            name="phi_log10",
        )
        expected_shape = (redshifts.size, magnitudes.size)
        if phi_log10.shape != expected_shape:
            raise ValueError(
                "phi_log10 must have shape (n_redshift, n_magnitude); "
                f"expected {expected_shape}, got {phi_log10.shape}."
            )
        if not np.all(redshifts >= 0.0):
            raise ValueError("redshift_grid cannot contain negative values.")

        wavelength = float(self.reference_wavelength_angstrom)
        if not np.isfinite(wavelength) or wavelength <= 0.0:
            raise ValueError("reference_wavelength_angstrom must be finite and positive.")
        magnitude_offset = float(self.native_to_monochromatic_ab_offset)
        if not np.isfinite(magnitude_offset):
            raise ValueError(
                "native_to_monochromatic_ab_offset must be finite."
            )

        calibration_range = tuple(float(value) for value in self.calibration_redshift_range)
        if (
            len(calibration_range) != 2
            or not np.all(np.isfinite(calibration_range))
            or calibration_range[0] < 0.0
            or calibration_range[1] <= calibration_range[0]
        ):
            raise ValueError(
                "calibration_redshift_range must be a finite increasing pair of "
                "non-negative redshifts."
            )
        _validate_flat_cosmology(self.source_cosmology, name="source_cosmology")
        _validate_flat_cosmology(self.target_cosmology, name="target_cosmology")
        if not isinstance(self.reddening_semantics, ReddeningSemantics):
            raise TypeError("reddening_semantics must be a ReddeningSemantics record.")

        object.__setattr__(self, "native_magnitude_grid", magnitudes)
        object.__setattr__(self, "redshift_grid", redshifts)
        object.__setattr__(self, "phi_log10", phi_log10)
        object.__setattr__(self, "reference_wavelength_angstrom", wavelength)
        object.__setattr__(
            self,
            "native_to_monochromatic_ab_offset",
            magnitude_offset,
        )
        object.__setattr__(self, "calibration_redshift_range", calibration_range)

    @property
    def phi_log10_per_magnitude(self):
        """Alias spelling out the density convention carried by ``phi_log10``."""

        return self.phi_log10

    @property
    def magnitude_grid(self):
        """Compatibility alias for the native absolute-magnitude grid."""

        return self.native_magnitude_grid

    @property
    def redshift_extrapolation_mask(self):
        """Read-only mask marking requested redshifts outside calibration."""

        lower, upper = self.calibration_redshift_range
        mask = (self.redshift_grid < lower) | (self.redshift_grid > upper)
        mask.setflags(write=False)
        return mask

    def to_metadata(self):
        """Return flat, scalar provenance suitable for HDF5 attributes."""

        lower, upper = self.calibration_redshift_range
        reddening = self.reddening_semantics
        return {
            "model_id": self.model_id,
            "formula_version": self.formula_version,
            "native_magnitude_name": self.native_magnitude_name,
            "reference_wavelength_angstrom": self.reference_wavelength_angstrom,
            "native_to_monochromatic_ab_offset": (
                self.native_to_monochromatic_ab_offset
            ),
            "source_cosmology_h0_km_s_mpc": float(self.source_cosmology.H0.value),
            "source_cosmology_omega_m": float(self.source_cosmology.Om0),
            "target_cosmology_h0_km_s_mpc": float(self.target_cosmology.H0.value),
            "target_cosmology_omega_m": float(self.target_cosmology.Om0),
            "calibration_redshift_min": lower,
            "calibration_redshift_max": upper,
            "luminosity_semantics": reddening.luminosity_semantics,
            "galactic_extinction": reddening.galactic_extinction,
            "internal_extinction": reddening.internal_extinction,
            "selection_dust_treatment": reddening.selection_dust_treatment,
            "apply_additional_internal_extinction": (
                reddening.apply_additional_internal_extinction
            ),
        }


@dataclass(frozen=True, slots=True)
class Palanque2016EvolvingParameters:
    """PLE+LEDE parameters evaluated at one or more redshifts."""

    m_star: float | np.ndarray
    log10_phi_star: float | np.ndarray
    alpha_bright: float | np.ndarray
    beta_faint: float | np.ndarray


@dataclass(frozen=True, slots=True)
class Kulkarni2019EvolvingParameters:
    """Flexible-DPL parameters evaluated at one or more redshifts."""

    m_star: float | np.ndarray
    log10_phi_star: float | np.ndarray
    alpha_bright: float | np.ndarray
    beta_faint: float | np.ndarray


def _as_redshift_values(redshift):
    values = np.asarray(redshift, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("redshift must contain only finite values.")
    if np.any(values < 0.0):
        raise ValueError("redshift cannot contain negative values.")
    return values


def _scalar_or_readonly(values, *, scalar):
    if scalar:
        return float(np.asarray(values))
    array = np.array(values, dtype=float, copy=True)
    array.setflags(write=False)
    return array


# Wang et al. (2026), Model A: Table 2 and Equations (8), (9), (11), (12).
_WANG_LOG10_PHI_STAR = -6.635
_WANG_M_STAR = -22.345
_WANG_BETA_FAINT = -1.675
_WANG_GAMMA_BRIGHT = -3.819
_WANG_P1 = 2.112
_WANG_P2 = 1.222
_WANG_P3 = 9.100
_WANG_K1 = 0.066
_WANG_K2 = -0.863
_WANG_K3 = 2.224


def wang2026_e1(redshift):
    """Return the normalized Model-A density-evolution factor e1(z)."""

    z = _as_redshift_values(redshift)
    scalar = z.ndim == 0
    normalization = 1.0 + (1.0 + _WANG_P1) ** (-_WANG_P3)
    values = (
        normalization
        * (1.0 + z) ** _WANG_P2
        / (1.0 + ((1.0 + z) / (1.0 + _WANG_P1)) ** _WANG_P3)
    )
    return _scalar_or_readonly(values, scalar=scalar)


def wang2026_e2(redshift):
    """Return the normalized Model-A luminosity-evolution factor e2(z)."""

    z = _as_redshift_values(redshift)
    scalar = z.ndim == 0
    numerator = 10.0 ** (-_WANG_K1 * _WANG_K3) + 10.0 ** (
        -_WANG_K2 * _WANG_K3
    )
    denominator = 10.0 ** (_WANG_K1 * (z - _WANG_K3)) + 10.0 ** (
        _WANG_K2 * (z - _WANG_K3)
    )
    return _scalar_or_readonly(numerator / denominator, scalar=scalar)


# Kulkarni et al. (2019), final MNRAS Table 3 and Equations (7), (13),
# (16)--(18).  These are the published, rounded marginal-posterior medians.
# They intentionally differ from the obsolete 2018-preprint coefficients
# still distributed by some downstream Shen pubtools checkouts.
_KULKARNI2019_COEFFICIENTS = MappingProxyType(
    {
        KULKARNI2019_TYPE1_MODEL1: (
            (-7.798, 1.128, -0.120),
            (-17.163, -5.512, 0.593, -0.024),
            (-3.223, -0.258),
            (-2.312, 0.559, 3.773, 141.884, -0.171),
        ),
        KULKARNI2019_TYPE1_MODEL2: (
            (-7.432, 0.953, -0.112),
            (-15.412, -6.869, 0.778, -0.032),
            (-2.959, -0.351),
            (-2.264, 0.530, 2.379, 12.527, -0.229),
        ),
        KULKARNI2019_TYPE1_MODEL3: (
            (-6.942, 0.629, -0.086),
            (-15.038, -7.046, 0.772, -0.030),
            (-2.888, -0.383),
            (-1.602, -0.082),
        ),
    }
)


def _kulkarni2019_broken_beta(redshift, coefficients):
    """Evaluate Equations (17)--(18) without overflowing either power."""

    c30, c31, break_redshift, low_z_power, high_z_power = coefficients
    zeta = (
        np.log1p(redshift) - np.log1p(break_redshift)
    ) / np.log(10.0)
    log_denominator = np.logaddexp(
        np.log(10.0) * low_z_power * zeta,
        np.log(10.0) * high_z_power * zeta,
    )
    return c30 + c31 * np.exp(-log_denominator)


def kulkarni2019_evolving_parameters(redshift, model_id):
    """Evaluate a final-publication Kulkarni et al. (2019) global model.

    The Chebyshev polynomials are evaluated directly at ``1 + z`` without
    rescaling to ``[-1, 1]``, matching both Equation (16) and the authors'
    public implementation.  Models 1 and 2 use the broken-power-law
    faint-slope evolution in Equations (17)--(18); Model 3 uses the published
    linear Chebyshev evolution.
    """

    try:
        c0, c1, c2, c3 = _KULKARNI2019_COEFFICIENTS[str(model_id)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown Kulkarni et al. (2019) model {model_id!r}; expected "
            f"one of {KULKARNI2019_TYPE1_MODEL_IDS}."
        ) from exc

    z = _as_redshift_values(redshift)
    scalar = z.ndim == 0
    x = 1.0 + z
    log10_phi_star = np.polynomial.chebyshev.chebval(x, c0)
    m_star = np.polynomial.chebyshev.chebval(x, c1)
    alpha_bright = np.polynomial.chebyshev.chebval(x, c2)
    if str(model_id) == KULKARNI2019_TYPE1_MODEL3:
        beta_faint = np.polynomial.chebyshev.chebval(x, c3)
    else:
        beta_faint = _kulkarni2019_broken_beta(z, c3)

    evaluated = (m_star, log10_phi_star, alpha_bright, beta_faint)
    if not all(np.all(np.isfinite(values)) for values in evaluated):
        raise ValueError(
            "Kulkarni et al. (2019) parameter evolution overflowed; "
            "use a finite astrophysical redshift range."
        )
    return Kulkarni2019EvolvingParameters(
        m_star=_scalar_or_readonly(m_star, scalar=scalar),
        log10_phi_star=_scalar_or_readonly(log10_phi_star, scalar=scalar),
        alpha_bright=_scalar_or_readonly(alpha_bright, scalar=scalar),
        beta_faint=_scalar_or_readonly(beta_faint, scalar=scalar),
    )


def kulkarni2019_model1_evolving_parameters(redshift):
    """Evaluate final-publication Kulkarni et al. (2019) Model 1."""

    return kulkarni2019_evolving_parameters(redshift, KULKARNI2019_TYPE1_MODEL1)


def kulkarni2019_model2_evolving_parameters(redshift):
    """Evaluate final-publication Kulkarni et al. (2019) Model 2."""

    return kulkarni2019_evolving_parameters(redshift, KULKARNI2019_TYPE1_MODEL2)


def kulkarni2019_model3_evolving_parameters(redshift):
    """Evaluate final-publication Kulkarni et al. (2019) Model 3."""

    return kulkarni2019_evolving_parameters(redshift, KULKARNI2019_TYPE1_MODEL3)


# Palanque-Delabrouille et al. (2016), Table 4 PLE+LEDE fit and Equations
# (6)--(10).  The low-redshift PLE branch is anchored by the tabulated M*(0).
_PALANQUE_Z_PIVOT = 2.2
_PALANQUE_M_STAR_0 = -22.25
_PALANQUE_LOG10_PHI_STAR_0 = -5.93
_PALANQUE_ALPHA_BRIGHT_0 = -3.89
_PALANQUE_BETA_FAINT = -1.47
_PALANQUE_K1 = 1.59
_PALANQUE_K2 = -0.36
_PALANQUE_C1A = -0.46
_PALANQUE_C1B = -0.06
_PALANQUE_C2 = -0.14
_PALANQUE_C3 = 0.32


def palanque2016_evolving_parameters(redshift):
    """Evaluate the continuous Table-4 PLE+LEDE parameters.

    The LF is continuous at z=2.2, although its redshift derivative has the
    published branch-change kink there.
    """

    z = _as_redshift_values(redshift)
    scalar = z.ndim == 0
    m_star_low = _PALANQUE_M_STAR_0 - 2.5 * (
        _PALANQUE_K1 * z + _PALANQUE_K2 * z**2
    )
    m_star_pivot = _PALANQUE_M_STAR_0 - 2.5 * (
        _PALANQUE_K1 * _PALANQUE_Z_PIVOT
        + _PALANQUE_K2 * _PALANQUE_Z_PIVOT**2
    )
    delta_z = z - _PALANQUE_Z_PIVOT
    high_redshift = z > _PALANQUE_Z_PIVOT

    m_star = np.where(
        high_redshift,
        m_star_pivot + _PALANQUE_C2 * delta_z,
        m_star_low,
    )
    log10_phi_star = np.where(
        high_redshift,
        _PALANQUE_LOG10_PHI_STAR_0
        + _PALANQUE_C1A * delta_z
        + _PALANQUE_C1B * delta_z**2,
        _PALANQUE_LOG10_PHI_STAR_0,
    )
    alpha_bright = np.where(
        high_redshift,
        _PALANQUE_ALPHA_BRIGHT_0 + _PALANQUE_C3 * delta_z,
        _PALANQUE_ALPHA_BRIGHT_0,
    )
    beta_faint = np.full_like(z, _PALANQUE_BETA_FAINT, dtype=float)
    return Palanque2016EvolvingParameters(
        m_star=_scalar_or_readonly(m_star, scalar=scalar),
        log10_phi_star=_scalar_or_readonly(log10_phi_star, scalar=scalar),
        alpha_bright=_scalar_or_readonly(alpha_bright, scalar=scalar),
        beta_faint=_scalar_or_readonly(beta_faint, scalar=scalar),
    )


def _double_power_law_log10(
    magnitude,
    *,
    m_star,
    log10_phi_star,
    faint_slope,
    bright_slope,
):
    faint_term = 0.4 * (faint_slope + 1.0) * (magnitude - m_star)
    bright_term = 0.4 * (bright_slope + 1.0) * (magnitude - m_star)
    log10_denominator = np.logaddexp(
        np.log(10.0) * faint_term,
        np.log(10.0) * bright_term,
    ) / np.log(10.0)
    return log10_phi_star - log10_denominator


def _cosmology_remap(redshift_grid, *, source_cosmology, target_cosmology):
    """Return DM_target-DM_source and dV_source/dV_target.

    The analytic z->0 limits avoid the undefined subtraction of two distance
    moduli and the 0/0 differential-volume ratio at exactly zero redshift.
    """

    source = _validate_flat_cosmology(source_cosmology, name="source_cosmology")
    target = _validate_flat_cosmology(target_cosmology, name="target_cosmology")
    z = np.asarray(redshift_grid, dtype=float)
    delta_distance_modulus = np.empty_like(z)
    volume_ratio = np.empty_like(z)
    positive = z > 0.0

    if np.any(positive):
        delta_distance_modulus[positive] = (
            target.distmod(z[positive]).value - source.distmod(z[positive]).value
        )
        volume_ratio[positive] = (
            source.differential_comoving_volume(z[positive]).value
            / target.differential_comoving_volume(z[positive]).value
        )

    if np.any(~positive):
        source_h0 = float(source.H0.value)
        target_h0 = float(target.H0.value)
        delta_distance_modulus[~positive] = 5.0 * np.log10(
            source_h0 / target_h0
        )
        volume_ratio[~positive] = (target_h0 / source_h0) ** 3

    if (
        not np.all(np.isfinite(delta_distance_modulus))
        or not np.all(np.isfinite(volume_ratio))
        or np.any(volume_ratio <= 0.0)
    ):
        raise ValueError("Cosmology remapping produced invalid distance or volume factors.")
    return delta_distance_modulus, volume_ratio


def _prepare_builder_inputs(
    target_magnitude_grid,
    redshift_grid,
    target_cosmology,
):
    magnitudes = _readonly_float_array(
        target_magnitude_grid,
        name="target_magnitude_grid",
        ndim=1,
    )
    redshifts = _readonly_float_array(
        redshift_grid,
        name="redshift_grid",
        ndim=1,
    )
    if np.any(redshifts < 0.0):
        raise ValueError("redshift_grid cannot contain negative values.")
    target = _validate_flat_cosmology(target_cosmology, name="target_cosmology")
    return magnitudes, redshifts, target


def build_wang2026_type1_lade_a(
    target_magnitude_grid,
    redshift_grid,
    target_cosmology,
):
    """Build Wang et al. (2026) Model A on target-cosmology M_1450.

    No internal extinction is added: the published LF is an empirical
    selection-space Type-1 UV LF, not a dust-free luminosity distribution.
    """

    magnitudes, redshifts, target = _prepare_builder_inputs(
        target_magnitude_grid,
        redshift_grid,
        target_cosmology,
    )
    delta_dm, volume_ratio = _cosmology_remap(
        redshifts,
        source_cosmology=WANG2026_SOURCE_COSMOLOGY,
        target_cosmology=target,
    )
    source_magnitude = magnitudes[None, :] + delta_dm[:, None]
    e1 = np.asarray(wang2026_e1(redshifts), dtype=float)[:, None]
    e2 = np.asarray(wang2026_e2(redshifts), dtype=float)[:, None]
    translated_magnitude = source_magnitude + 2.5 * np.log10(e2)
    phi_log10 = _double_power_law_log10(
        translated_magnitude,
        m_star=_WANG_M_STAR,
        log10_phi_star=_WANG_LOG10_PHI_STAR,
        faint_slope=_WANG_BETA_FAINT,
        bright_slope=_WANG_GAMMA_BRIGHT,
    )
    # Equation (8) has only the e1 prefactor; there is no extra 1/e2
    # luminosity-Jacobian factor for a per-magnitude LF.
    phi_log10 += np.log10(e1) + np.log10(volume_ratio[:, None])
    return LFGrid(
        model_id=WANG2026_TYPE1_LADE_A,
        phi_log10=phi_log10,
        native_magnitude_grid=magnitudes,
        redshift_grid=redshifts,
        reference_wavelength_angstrom=WANG2026_REFERENCE_WAVELENGTH_ANGSTROM,
        native_to_monochromatic_ab_offset=0.0,
        native_magnitude_name="M_1450_AB",
        source_cosmology=WANG2026_SOURCE_COSMOLOGY,
        target_cosmology=target,
        calibration_redshift_range=(0.1, 7.5),
        formula_version="wang2026_model_a_eq8_eq9_eq11_eq12_table2",
        reddening_semantics=WANG2026_REDDENING_SEMANTICS,
    )


def build_palanque2016_ple_lede(
    target_magnitude_grid,
    redshift_grid,
    target_cosmology,
):
    """Build the Palanque-Delabrouille et al. (2016) PLE+LEDE LF.

    ``target_magnitude_grid`` is M_g(z=2) expressed in the target cosmology.
    The published LF is evaluated at

    ``M_source = M_target + DM_target - DM_source``

    and its density is multiplied by ``dV_source/dV_target``.  Redshifts
    outside 0.68--4.0 are allowed but marked by LFGrid's extrapolation mask.
    """

    magnitudes, redshifts, target = _prepare_builder_inputs(
        target_magnitude_grid,
        redshift_grid,
        target_cosmology,
    )
    delta_dm, volume_ratio = _cosmology_remap(
        redshifts,
        source_cosmology=PALANQUE2016_SOURCE_COSMOLOGY,
        target_cosmology=target,
    )
    source_magnitude = magnitudes[None, :] + delta_dm[:, None]
    parameters = palanque2016_evolving_parameters(redshifts)
    m_star = np.asarray(parameters.m_star, dtype=float)[:, None]
    log10_phi_star = np.asarray(parameters.log10_phi_star, dtype=float)[:, None]
    alpha_bright = np.asarray(parameters.alpha_bright, dtype=float)[:, None]
    beta_faint = np.asarray(parameters.beta_faint, dtype=float)[:, None]
    phi_log10 = _double_power_law_log10(
        source_magnitude,
        m_star=m_star,
        log10_phi_star=log10_phi_star,
        faint_slope=beta_faint,
        bright_slope=alpha_bright,
    )
    phi_log10 += np.log10(volume_ratio[:, None])
    return LFGrid(
        model_id=PALANQUE2016_PLE_LEDE,
        phi_log10=phi_log10,
        native_magnitude_grid=magnitudes,
        redshift_grid=redshifts,
        reference_wavelength_angstrom=PALANQUE2016_REFERENCE_WAVELENGTH_ANGSTROM,
        native_to_monochromatic_ab_offset=(
            PALANQUE2016_NATIVE_TO_MONOCHROMATIC_AB_OFFSET
        ),
        native_magnitude_name="M_g(z=2)_AB",
        source_cosmology=PALANQUE2016_SOURCE_COSMOLOGY,
        target_cosmology=target,
        calibration_redshift_range=(0.68, 4.0),
        formula_version="palanque2016_ple_lede_eq6_to_eq10_table4",
        reddening_semantics=PALANQUE2016_REDDENING_SEMANTICS,
    )


def _build_kulkarni2019_type1(
    model_id,
    target_magnitude_grid,
    redshift_grid,
    target_cosmology,
):
    magnitudes, redshifts, target = _prepare_builder_inputs(
        target_magnitude_grid,
        redshift_grid,
        target_cosmology,
    )
    delta_dm, volume_ratio = _cosmology_remap(
        redshifts,
        source_cosmology=KULKARNI2019_SOURCE_COSMOLOGY,
        target_cosmology=target,
    )
    source_magnitude = magnitudes[None, :] + delta_dm[:, None]
    parameters = kulkarni2019_evolving_parameters(redshifts, model_id)
    phi_log10 = _double_power_law_log10(
        source_magnitude,
        m_star=np.asarray(parameters.m_star, dtype=float)[:, None],
        log10_phi_star=np.asarray(
            parameters.log10_phi_star,
            dtype=float,
        )[:, None],
        faint_slope=np.asarray(parameters.beta_faint, dtype=float)[:, None],
        bright_slope=np.asarray(parameters.alpha_bright, dtype=float)[:, None],
    )
    phi_log10 += np.log10(volume_ratio[:, None])
    model_number = str(model_id).rsplit("model", maxsplit=1)[-1]
    if str(model_id) == KULKARNI2019_TYPE1_MODEL3:
        formula_version = (
            "kulkarni2019_model3_eq7_eq13_eq16_table3_"
            "published_rounded_medians"
        )
    else:
        formula_version = (
            f"kulkarni2019_model{model_number}_eq7_eq13_eq16_to_eq18_"
            "table3_published_rounded_medians"
        )
    return LFGrid(
        model_id=str(model_id),
        phi_log10=phi_log10,
        native_magnitude_grid=magnitudes,
        redshift_grid=redshifts,
        reference_wavelength_angstrom=(
            KULKARNI2019_REFERENCE_WAVELENGTH_ANGSTROM
        ),
        native_to_monochromatic_ab_offset=0.0,
        native_magnitude_name="M_1450_AB",
        source_cosmology=KULKARNI2019_SOURCE_COSMOLOGY,
        target_cosmology=target,
        calibration_redshift_range=(0.6, 6.5),
        formula_version=formula_version,
        reddening_semantics=KULKARNI2019_REDDENING_SEMANTICS,
    )


def build_kulkarni2019_type1_model1(
    target_magnitude_grid,
    redshift_grid,
    target_cosmology,
):
    """Build final-publication Kulkarni et al. (2019) Model 1."""

    return _build_kulkarni2019_type1(
        KULKARNI2019_TYPE1_MODEL1,
        target_magnitude_grid,
        redshift_grid,
        target_cosmology,
    )


def build_kulkarni2019_type1_model2(
    target_magnitude_grid,
    redshift_grid,
    target_cosmology,
):
    """Build final-publication Kulkarni et al. (2019) Model 2."""

    return _build_kulkarni2019_type1(
        KULKARNI2019_TYPE1_MODEL2,
        target_magnitude_grid,
        redshift_grid,
        target_cosmology,
    )


def build_kulkarni2019_type1_model3(
    target_magnitude_grid,
    redshift_grid,
    target_cosmology,
):
    """Build final-publication Kulkarni et al. (2019) Model 3."""

    return _build_kulkarni2019_type1(
        KULKARNI2019_TYPE1_MODEL3,
        target_magnitude_grid,
        redshift_grid,
        target_cosmology,
    )


_BUILDERS: MappingProxyType[
    str,
    Callable[[np.ndarray, np.ndarray, FlatLambdaCDM], LFGrid],
] = MappingProxyType(
    {
        WANG2026_TYPE1_LADE_A: build_wang2026_type1_lade_a,
        PALANQUE2016_PLE_LEDE: build_palanque2016_ple_lede,
        KULKARNI2019_TYPE1_MODEL1: build_kulkarni2019_type1_model1,
        KULKARNI2019_TYPE1_MODEL2: build_kulkarni2019_type1_model2,
        KULKARNI2019_TYPE1_MODEL3: build_kulkarni2019_type1_model3,
    }
)


def build_empirical_lf(
    model_id,
    target_magnitude_grid,
    redshift_grid,
    target_cosmology,
):
    """Dispatch an exact fixed empirical-LF model by its public ID."""

    try:
        builder = _BUILDERS[str(model_id)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown empirical LF model {model_id!r}; expected one of "
            f"{EMPIRICAL_LF_MODEL_IDS}."
        ) from exc
    return builder(target_magnitude_grid, redshift_grid, target_cosmology)


__all__ = [
    "EMPIRICAL_LF_MODEL_IDS",
    "KULKARNI2019_REDDENING_SEMANTICS",
    "KULKARNI2019_REFERENCE_WAVELENGTH_ANGSTROM",
    "KULKARNI2019_SOURCE_COSMOLOGY",
    "KULKARNI2019_TYPE1_MODEL1",
    "KULKARNI2019_TYPE1_MODEL2",
    "KULKARNI2019_TYPE1_MODEL3",
    "KULKARNI2019_TYPE1_MODEL_IDS",
    "Kulkarni2019EvolvingParameters",
    "LFGrid",
    "PALANQUE2016_PLE_LEDE",
    "PALANQUE2016_NATIVE_TO_MONOCHROMATIC_AB_OFFSET",
    "PALANQUE2016_REDDENING_SEMANTICS",
    "PALANQUE2016_REFERENCE_WAVELENGTH_ANGSTROM",
    "PALANQUE2016_SOURCE_COSMOLOGY",
    "Palanque2016EvolvingParameters",
    "ReddeningSemantics",
    "WANG2026_REDDENING_SEMANTICS",
    "WANG2026_REFERENCE_WAVELENGTH_ANGSTROM",
    "WANG2026_SOURCE_COSMOLOGY",
    "WANG2026_TYPE1_LADE_A",
    "build_empirical_lf",
    "build_kulkarni2019_type1_model1",
    "build_kulkarni2019_type1_model2",
    "build_kulkarni2019_type1_model3",
    "build_palanque2016_ple_lede",
    "build_wang2026_type1_lade_a",
    "kulkarni2019_evolving_parameters",
    "kulkarni2019_model1_evolving_parameters",
    "kulkarni2019_model2_evolving_parameters",
    "kulkarni2019_model3_evolving_parameters",
    "palanque2016_evolving_parameters",
    "wang2026_e1",
    "wang2026_e2",
]
