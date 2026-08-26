"""Pure latent-continuum-slope completeness utilities.

This module contains the numerical pieces needed to add a redshift-independent
parent distribution for the dereddened quasar continuum slope to the Hubble
likelihood.  It intentionally has no dependency on either Hubble sampler.  The
array formulas use only NumPy operations that have direct JAX equivalents, so
the same parameterization can be used by the NumPy/Dynesty and JAX/NumPyro
implementations.

The selection response is a bounded logit perturbation of an existing base
completeness ``C3``.  A vectorized scalar offset, ``kappa``, is solved so that
the 12-point Gauss--Hermite marginal of the response is exactly ``C3`` (to
floating-point precision).  This makes adding latent alpha structure neutral
to the already calibrated marginal completeness.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
from typing import Any, Literal

import numpy as np


LATENT_ALPHA_SCHEMA_VERSION = "qvc_latent_alpha_completeness_v1"
LATENT_ALPHA_MODES = ("off", "fixed", "joint")
LUMINOSITY_STATES = ("attenuated", "dereddened")

DEFAULT_ALPHA_MU = -0.5
DEFAULT_ALPHA_SIGMA = 0.3
DEFAULT_LOGL_PIVOT = 45.5
DEFAULT_REDSHIFT_MIN = 0.44
DEFAULT_REDSHIFT_MAX = 3.16
DEFAULT_MAGNITUDE_PIVOT = 21.25
DEFAULT_MAGNITUDE_SCALE = 2.75

# Monochromatic AB conversion at rest-frame 2500 Angstrom.  The absolute-AB
# zero point is the luminosity-density form of the 3631 Jy convention:
# M_AB = AB_ABSOLUTE_MAG_ZEROPOINT - 2.5 log10(L_nu / erg s^-1 Hz^-1).
NU_2500_HZ = 2.99792458e18 / 2500.0
AB_ABSOLUTE_MAG_ZEROPOINT = 51.59477721004232
M2500_TO_LOG_NU_LNU_INTERCEPT = (
    np.log10(NU_2500_HZ) + AB_ABSOLUTE_MAG_ZEROPOINT / 2.5
)

BETA_ALPHA_L_PARAMETER = "beta_alpha_L"
BETA_ALPHA_L_PRIOR = (-0.5, 0.5)
BASE_RESPONSE_COEFFICIENT_PRIOR = (-3.0, 3.0)
MAGNITUDE_RESPONSE_COEFFICIENT_PRIOR = (-2.0, 2.0)
RESPONSE_COEFFICIENT_PRIOR_SIGMA = 0.5

GAUSS_HERMITE_ORDER = 12
JOINT_DRAW_INPUT_COUNT = 64
JOINT_DRAW_SELECTED_COUNT = 16


_EMPIRICAL_ATTENUATED_LFS = frozenset(
    {
        "wang2026_type1_lade_a",
        "palanque2016_ple_lede",
        "kulkarni2019_type1_model1",
        "kulkarni2019_type1_model2",
        "kulkarni2019_type1_model3",
    }
)
_SHEN_LUMINOSITY_STATES = {
    "all_nh_attenuated": "attenuated",
    "type1_attenuated": "attenuated",
    "type1_intrinsic": "dereddened",
}


def _normalise_identifier(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def resolve_lf_luminosity_state(
    lf_model: str,
    *,
    shen_lf_mode: str | None = None,
    requested_state: str = "auto",
) -> Literal["attenuated", "dereddened"]:
    """Resolve the luminosity state used by an LF, rejecting mismatches.

    The empirical Type-1 LFs retain internal attenuation and therefore use the
    attenuated luminosity.  Shen's intrinsic Type-1 mode uses the dereddened
    luminosity; its two attenuation-integrated modes use attenuated luminosity.
    An explicit ``requested_state`` is allowed only when it agrees with the LF
    semantics, preventing a silent change of the alpha--luminosity relation.
    """

    model = _normalise_identifier(lf_model)
    if model == "shen":
        mode = _normalise_identifier(shen_lf_mode or "all_nh_attenuated")
        try:
            resolved = _SHEN_LUMINOSITY_STATES[mode]
        except KeyError as exc:
            valid = ", ".join(sorted(_SHEN_LUMINOSITY_STATES))
            raise ValueError(
                f"Unknown Shen LF mode {shen_lf_mode!r}; expected one of {valid}."
            ) from exc
    elif model in _EMPIRICAL_ATTENUATED_LFS:
        resolved = "attenuated"
    else:
        valid = ", ".join(["shen", *sorted(_EMPIRICAL_ATTENUATED_LFS)])
        raise ValueError(f"Unknown LF model {lf_model!r}; expected one of {valid}.")

    requested = _normalise_identifier(requested_state)
    if requested != "auto" and requested not in LUMINOSITY_STATES:
        raise ValueError(
            f"Unknown luminosity state {requested_state!r}; expected auto, "
            f"attenuated, or dereddened."
        )
    if requested != "auto" and requested != resolved:
        raise ValueError(
            f"LF {lf_model!r} requires {resolved} luminosity, not {requested}."
        )
    return resolved  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class LatentAlphaConfig:
    """Immutable definition of the latent-alpha selection model.

    ``mode='off'`` fixes the luminosity coefficient to zero.  ``fixed`` uses
    ``fixed_beta_l`` and ``joint`` obtains :data:`BETA_ALPHA_L_PARAMETER` from
    the cosmology parameter mapping.  Positive beta means more luminous
    quasars have larger (bluer) :math:`alpha_nu`.
    """

    mode: Literal["off", "fixed", "joint"] = "off"
    fixed_beta_l: float | None = None
    mu: float = DEFAULT_ALPHA_MU
    sigma: float = DEFAULT_ALPHA_SIGMA
    logl_pivot: float = DEFAULT_LOGL_PIVOT
    beta_l_prior: tuple[float, float] = BETA_ALPHA_L_PRIOR
    luminosity_state: Literal["attenuated", "dereddened"] = "attenuated"
    lf_model: str = "unspecified"
    shen_lf_mode: str | None = None
    include_magnitude_interactions: bool = False
    redshift_min: float = DEFAULT_REDSHIFT_MIN
    redshift_max: float = DEFAULT_REDSHIFT_MAX
    magnitude_pivot: float = DEFAULT_MAGNITUDE_PIVOT
    magnitude_scale: float = DEFAULT_MAGNITUDE_SCALE
    quadrature_order: int = GAUSS_HERMITE_ORDER
    schema_version: str = LATENT_ALPHA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        mode = _normalise_identifier(self.mode)
        state = _normalise_identifier(self.luminosity_state)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "luminosity_state", state)
        object.__setattr__(self, "lf_model", _normalise_identifier(self.lf_model))
        object.__setattr__(
            self,
            "shen_lf_mode",
            (
                None
                if self.shen_lf_mode is None
                else _normalise_identifier(self.shen_lf_mode)
            ),
        )
        try:
            beta_l_prior = tuple(float(value) for value in self.beta_l_prior)
        except (TypeError, ValueError) as exc:
            raise ValueError("beta_l_prior must contain two finite bounds.") from exc
        object.__setattr__(self, "beta_l_prior", beta_l_prior)

        if mode not in LATENT_ALPHA_MODES:
            raise ValueError(
                f"Unknown latent-alpha mode {self.mode!r}; "
                f"expected one of {LATENT_ALPHA_MODES}."
            )
        if state not in LUMINOSITY_STATES:
            raise ValueError(
                f"Unknown luminosity state {self.luminosity_state!r}; "
                f"expected one of {LUMINOSITY_STATES}."
            )
        if mode == "fixed":
            if self.fixed_beta_l is None or not np.isfinite(self.fixed_beta_l):
                raise ValueError("fixed mode requires a finite fixed_beta_l.")
        elif self.fixed_beta_l is not None:
            raise ValueError(
                f"fixed_beta_l is only valid in fixed mode, not {mode!r}."
            )

        finite_values = {
            "mu": self.mu,
            "sigma": self.sigma,
            "logl_pivot": self.logl_pivot,
            "redshift_min": self.redshift_min,
            "redshift_max": self.redshift_max,
            "magnitude_pivot": self.magnitude_pivot,
            "magnitude_scale": self.magnitude_scale,
        }
        for name, value in finite_values.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive.")
        if (
            len(beta_l_prior) != 2
            or not np.all(np.isfinite(beta_l_prior))
            or beta_l_prior[0] >= beta_l_prior[1]
        ):
            raise ValueError(
                "beta_l_prior must contain two finite, strictly ordered bounds."
            )
        if self.redshift_max <= self.redshift_min:
            raise ValueError("redshift_max must exceed redshift_min.")
        if self.magnitude_scale <= 0.0:
            raise ValueError("magnitude_scale must be positive.")
        if self.quadrature_order != GAUSS_HERMITE_ORDER:
            raise ValueError(
                f"quadrature_order is fixed at {GAUSS_HERMITE_ORDER} for "
                "sampler and cache compatibility."
            )
        if self.schema_version != LATENT_ALPHA_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {self.schema_version!r}; expected "
                f"{LATENT_ALPHA_SCHEMA_VERSION!r}."
            )

    @classmethod
    def for_lf(
        cls,
        *,
        lf_model: str,
        shen_lf_mode: str | None = None,
        requested_luminosity_state: str = "auto",
        **kwargs: Any,
    ) -> "LatentAlphaConfig":
        """Construct a config whose luminosity state is resolved from the LF."""

        state = resolve_lf_luminosity_state(
            lf_model,
            shen_lf_mode=shen_lf_mode,
            requested_state=requested_luminosity_state,
        )
        if "luminosity_state" in kwargs:
            raise TypeError(
                "for_lf resolves luminosity_state; use requested_luminosity_state "
                "to validate an explicit state."
            )
        normalized_model = _normalise_identifier(lf_model)
        normalized_shen_mode = (
            _normalise_identifier(shen_lf_mode or "all_nh_attenuated")
            if normalized_model == "shen"
            else None
        )
        return cls(
            luminosity_state=state,
            lf_model=normalized_model,
            shen_lf_mode=normalized_shen_mode,
            **kwargs,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LatentAlphaConfig":
        return cls(**dict(payload))

    @classmethod
    def from_json(cls, payload: str) -> "LatentAlphaConfig":
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise ValueError("Latent-alpha config JSON must encode an object.")
        return cls.from_dict(decoded)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    def beta_l(self, parameters: Mapping[str, Any] | None = None) -> float:
        """Return the effective alpha--luminosity coefficient for this mode."""

        if self.mode == "off":
            return 0.0
        if self.mode == "fixed":
            # Validation in __post_init__ proves this is finite and non-None.
            return float(self.fixed_beta_l)  # type: ignore[arg-type]
        if parameters is None or BETA_ALPHA_L_PARAMETER not in parameters:
            raise KeyError(
                f"joint mode requires parameter {BETA_ALPHA_L_PARAMETER!r}."
            )
        value = float(parameters[BETA_ALPHA_L_PARAMETER])
        if not np.isfinite(value):
            raise ValueError(f"{BETA_ALPHA_L_PARAMETER} must be finite.")
        return value

    def joint_parameter_names(self) -> tuple[str, ...]:
        return (BETA_ALPHA_L_PARAMETER,) if self.mode == "joint" else ()


def parent_alpha_mean(
    log_luminosity: Any,
    beta_l: Any,
    *,
    mu: float = DEFAULT_ALPHA_MU,
    logl_pivot: float = DEFAULT_LOGL_PIVOT,
) -> np.ndarray:
    """Mean of the redshift-independent alpha parent at a luminosity."""

    log_luminosity_array = np.asarray(log_luminosity, dtype=float)
    beta_array = np.asarray(beta_l, dtype=float)
    return mu + beta_array * (log_luminosity_array - logl_pivot)


def absolute_m2500_to_log_nu_lnu(absolute_m2500: Any) -> np.ndarray:
    """Convert rest-frame monochromatic ``M_2500,AB`` to log10(nu Lnu).

    The return value is in ``erg s^-1`` and uses the exact 2500-A frequency,
    rather than the historical QVC ``M=90-2.5 logL`` approximation.  This
    matters when interpreting the explicitly exposed luminosity pivot.
    """

    magnitude = np.asarray(absolute_m2500, dtype=float)
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("absolute_m2500 must be finite.")
    return M2500_TO_LOG_NU_LNU_INTERCEPT - 0.4 * magnitude


def parent_alpha_logpdf(
    alpha_nu: Any,
    log_luminosity: Any,
    beta_l: Any,
    *,
    mu: float = DEFAULT_ALPHA_MU,
    sigma: float = DEFAULT_ALPHA_SIGMA,
    logl_pivot: float = DEFAULT_LOGL_PIVOT,
) -> np.ndarray:
    """Log-density of the Normal parent distribution for ``alpha_nu``."""

    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive.")
    mean = parent_alpha_mean(
        log_luminosity, beta_l, mu=mu, logl_pivot=logl_pivot
    )
    standardized = (np.asarray(alpha_nu, dtype=float) - mean) / sigma
    return -0.5 * standardized**2 - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)


def parent_alpha_pdf(
    alpha_nu: Any,
    log_luminosity: Any,
    beta_l: Any,
    *,
    mu: float = DEFAULT_ALPHA_MU,
    sigma: float = DEFAULT_ALPHA_SIGMA,
    logl_pivot: float = DEFAULT_LOGL_PIVOT,
) -> np.ndarray:
    return np.exp(
        parent_alpha_logpdf(
            alpha_nu,
            log_luminosity,
            beta_l,
            mu=mu,
            sigma=sigma,
            logl_pivot=logl_pivot,
        )
    )


def parent_alpha_mean_from_config(
    log_luminosity: Any,
    config: LatentAlphaConfig,
    *,
    parameters: Mapping[str, Any] | None = None,
) -> np.ndarray:
    return parent_alpha_mean(
        log_luminosity,
        config.beta_l(parameters),
        mu=config.mu,
        logl_pivot=config.logl_pivot,
    )


def response_coefficient_names(
    include_magnitude_interactions: bool = False,
) -> tuple[str, ...]:
    """Return coefficient names in the exact numerical design-matrix order."""

    names: list[str] = []
    for order in range(4):
        names.extend(
            (
                f"alpha_sel_z_p{order}_linear",
                f"alpha_sel_z_p{order}_quadratic",
            )
        )
    if include_magnitude_interactions:
        for order in range(4):
            names.extend(
                (
                    f"alpha_sel_mag_z_p{order}_linear",
                    f"alpha_sel_mag_z_p{order}_quadratic",
                )
            )
    return tuple(names)


def response_coefficient_prior_specs(
    include_magnitude_interactions: bool = False,
) -> dict[str, dict[str, float | str]]:
    """Return ordered, JSON-safe shrinkage-prior specifications.

    Response terms use a zero-centered Normal with sigma 0.5, truncated at
    broad logit-response bounds.  The common scale regularizes the otherwise
    weakly identified redshift/alpha basis while the truncation prevents rare
    sampler excursions from producing numerically saturated response surfaces.
    """

    specs: dict[str, dict[str, float | str]] = {}
    for name in response_coefficient_names(include_magnitude_interactions):
        bounds = (
            MAGNITUDE_RESPONSE_COEFFICIENT_PRIOR
            if "_mag_" in name
            else BASE_RESPONSE_COEFFICIENT_PRIOR
        )
        specs[name] = {
            "distribution": "truncated_normal",
            "mean": 0.0,
            "sigma": RESPONSE_COEFFICIENT_PRIOR_SIGMA,
            "low": bounds[0],
            "high": bounds[1],
        }
    return specs


def latent_alpha_parameter_prior_specs(
    config: LatentAlphaConfig,
) -> dict[str, dict[str, float | str]]:
    specs: dict[str, dict[str, float | str]] = {}
    if config.mode == "joint":
        specs[BETA_ALPHA_L_PARAMETER] = {
            "distribution": "uniform",
            "low": config.beta_l_prior[0],
            "high": config.beta_l_prior[1],
            "units": "alpha_nu_per_dex",
        }
    specs.update(
        response_coefficient_prior_specs(config.include_magnitude_interactions)
    )
    return specs


def response_coefficient_vector(
    coefficients: Mapping[str, Any] | Any,
    *,
    include_magnitude_interactions: bool = False,
) -> np.ndarray:
    """Convert an ordered vector or named mapping into the canonical vector.

    Missing mapping entries mean a zero response term.  Unknown entries are an
    error, which prevents a misspelled sampled parameter from being ignored.
    """

    names = response_coefficient_names(include_magnitude_interactions)
    if isinstance(coefficients, Mapping):
        unknown = set(coefficients).difference(names)
        if unknown:
            raise KeyError(f"Unknown alpha-response coefficients: {sorted(unknown)}")
        values = np.asarray([coefficients.get(name, 0.0) for name in names], dtype=float)
    else:
        values = np.asarray(coefficients, dtype=float)
        if values.shape != (len(names),):
            raise ValueError(
                f"Expected {len(names)} alpha-response coefficients, got "
                f"shape {values.shape}."
            )
    if not np.all(np.isfinite(values)):
        raise ValueError("Alpha-response coefficients must all be finite.")
    return values


def _legendre_basis_order3(coordinate: np.ndarray) -> np.ndarray:
    p0 = np.ones_like(coordinate)
    p1 = coordinate
    p2 = 0.5 * (3.0 * coordinate**2 - 1.0)
    p3 = 0.5 * (5.0 * coordinate**3 - 3.0 * coordinate)
    return np.stack((p0, p1, p2, p3), axis=-1)


def response_design_matrix(
    alpha_nu: Any,
    redshift: Any,
    *,
    alpha_reference_mean: float = DEFAULT_ALPHA_MU,
    alpha_reference_sigma: float = DEFAULT_ALPHA_SIGMA,
    redshift_min: float = DEFAULT_REDSHIFT_MIN,
    redshift_max: float = DEFAULT_REDSHIFT_MAX,
    magnitude: Any | None = None,
    include_magnitude_interactions: bool = False,
    magnitude_pivot: float = DEFAULT_MAGNITUDE_PIVOT,
    magnitude_scale: float = DEFAULT_MAGNITUDE_SCALE,
) -> np.ndarray:
    """Build four Legendre-z linear/quadratic-alpha response blocks.

    Alpha is standardized against the fixed reference parent (not its
    luminosity-shifted mean), so changing ``beta_alpha_L`` changes the population
    passed through a fixed selection response.  The quadratic basis is
    ``u**2 - 1``.  Redshift and optional magnitude coordinates are edge-clamped
    to their calibrated support, keeping every response basis bounded.
    """

    if not np.isfinite(alpha_reference_sigma) or alpha_reference_sigma <= 0.0:
        raise ValueError("alpha_reference_sigma must be finite and positive.")
    if not redshift_max > redshift_min:
        raise ValueError("redshift_max must exceed redshift_min.")
    if not np.isfinite(magnitude_scale) or magnitude_scale <= 0.0:
        raise ValueError("magnitude_scale must be finite and positive.")

    alpha_array = np.asarray(alpha_nu, dtype=float)
    redshift_array = np.asarray(redshift, dtype=float)
    if include_magnitude_interactions:
        if magnitude is None:
            raise ValueError(
                "magnitude is required when magnitude interactions are enabled."
            )
        alpha_array, redshift_array, magnitude_array = np.broadcast_arrays(
            alpha_array, redshift_array, np.asarray(magnitude, dtype=float)
        )
    else:
        alpha_array, redshift_array = np.broadcast_arrays(alpha_array, redshift_array)
        magnitude_array = None

    if not np.all(np.isfinite(alpha_array)) or not np.all(np.isfinite(redshift_array)):
        raise ValueError("alpha_nu and redshift must be finite.")
    if magnitude_array is not None and not np.all(np.isfinite(magnitude_array)):
        raise ValueError("magnitude must be finite.")

    u = (alpha_array - alpha_reference_mean) / alpha_reference_sigma
    q = u**2 - 1.0
    z_coordinate = 2.0 * (redshift_array - redshift_min) / (
        redshift_max - redshift_min
    ) - 1.0
    z_coordinate = np.clip(z_coordinate, -1.0, 1.0)
    legendre = _legendre_basis_order3(z_coordinate)

    base_features: list[np.ndarray] = []
    for order in range(4):
        base_features.extend((legendre[..., order] * u, legendre[..., order] * q))
    design = np.stack(base_features, axis=-1)

    if magnitude_array is not None:
        magnitude_coordinate = np.clip(
            (magnitude_array - magnitude_pivot) / magnitude_scale, -1.0, 1.0
        )
        interaction_features: list[np.ndarray] = []
        for order in range(4):
            interaction_features.extend(
                (
                    magnitude_coordinate * legendre[..., order] * u,
                    magnitude_coordinate * legendre[..., order] * q,
                )
            )
        design = np.concatenate(
            (design, np.stack(interaction_features, axis=-1)), axis=-1
        )
    return design


def response_logit_offset(
    alpha_nu: Any,
    redshift: Any,
    coefficients: Mapping[str, Any] | Any,
    *,
    config: LatentAlphaConfig,
    magnitude: Any | None = None,
) -> np.ndarray:
    vector = response_coefficient_vector(
        coefficients,
        include_magnitude_interactions=config.include_magnitude_interactions,
    )
    design = response_design_matrix(
        alpha_nu,
        redshift,
        alpha_reference_mean=config.mu,
        alpha_reference_sigma=config.sigma,
        redshift_min=config.redshift_min,
        redshift_max=config.redshift_max,
        magnitude=magnitude,
        include_magnitude_interactions=config.include_magnitude_interactions,
        magnitude_pivot=config.magnitude_pivot,
        magnitude_scale=config.magnitude_scale,
    )
    return np.sum(design * vector, axis=-1)


@lru_cache(maxsize=None)
def _normal_gauss_hermite_nodes_cached(order: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if order <= 0:
        raise ValueError("Gauss-Hermite order must be positive.")
    raw_nodes, raw_weights = np.polynomial.hermite.hermgauss(order)
    nodes = np.sqrt(2.0) * raw_nodes
    weights = raw_weights / np.sqrt(np.pi)
    return tuple(float(value) for value in nodes), tuple(float(value) for value in weights)


def normal_gauss_hermite_nodes(
    order: int = GAUSS_HERMITE_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nodes and normalized weights for expectations under N(0, 1)."""

    nodes, weights = _normal_gauss_hermite_nodes_cached(int(order))
    # New arrays prevent a caller from mutating the cached canonical values.
    return np.asarray(nodes, dtype=float), np.asarray(weights, dtype=float)


def stable_sigmoid(value: Any) -> np.ndarray:
    """Overflow-safe logistic function using operations shared with JAX."""

    value_array = np.asarray(value, dtype=float)
    return np.exp(-np.logaddexp(0.0, -value_array))


def stable_logit(probability: Any) -> np.ndarray:
    """Logit on the closed unit interval, returning +/-inf at endpoints."""

    probability_array = np.asarray(probability, dtype=float)
    if not np.all(np.isfinite(probability_array)):
        raise ValueError("Probabilities must be finite.")
    if np.any((probability_array < 0.0) | (probability_array > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1].")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(probability_array) - np.log1p(-probability_array)


def solve_response_kappa(
    base_completeness: Any,
    node_logit_offsets: Any,
    *,
    weights: Any | None = None,
    iterations: int = 96,
) -> np.ndarray:
    """Solve the normalization offset for every leading array element.

    ``node_logit_offsets`` has quadrature along its final axis.  The returned
    kappa satisfies

    ``sum(w * sigmoid(logit(C3) + offset - kappa)) == C3``.

    A fixed-iteration bisection is deterministic, vectorized over all leading
    axes, and translates directly to ``jax.lax.fori_loop`` when integrated.
    """

    offsets = np.asarray(node_logit_offsets, dtype=float)
    if offsets.ndim < 1:
        raise ValueError("node_logit_offsets must have a quadrature axis.")
    if not np.all(np.isfinite(offsets)):
        raise ValueError("node_logit_offsets must be finite.")
    quadrature_size = offsets.shape[-1]
    if weights is None:
        _, weight_array = normal_gauss_hermite_nodes(quadrature_size)
    else:
        weight_array = np.asarray(weights, dtype=float)
    if weight_array.shape != (quadrature_size,):
        raise ValueError(
            f"Expected {quadrature_size} quadrature weights, got "
            f"shape {weight_array.shape}."
        )
    if (
        not np.all(np.isfinite(weight_array))
        or np.any(weight_array < 0.0)
        or not float(np.sum(weight_array)) > 0.0
    ):
        raise ValueError("Quadrature weights must be finite, nonnegative, and nonzero.")
    weight_array = weight_array / np.sum(weight_array)
    if iterations < 1:
        raise ValueError("iterations must be positive.")

    base = np.asarray(base_completeness, dtype=float)
    if not np.all(np.isfinite(base)) or np.any((base < 0.0) | (base > 1.0)):
        raise ValueError("base_completeness must be finite and lie in [0, 1].")
    leading_shape = np.broadcast_shapes(base.shape, offsets.shape[:-1])
    base = np.broadcast_to(base, leading_shape)
    offsets = np.broadcast_to(offsets, leading_shape + (quadrature_size,))

    # Endpoints are handled analytically.  The interior logit can safely use
    # machine-tiny clipping without changing any ordinary completeness value.
    tiny = np.finfo(float).tiny
    interior_base = np.clip(base, tiny, 1.0 - np.finfo(float).eps)
    base_logit = np.log(interior_base) - np.log1p(-interior_base)

    # Relative to the population offsets, +/-64 brackets every practically
    # representable interior probability while retaining a tight root interval.
    low = np.min(offsets, axis=-1) - 64.0
    high = np.max(offsets, axis=-1) + 64.0
    for _ in range(iterations):
        midpoint = 0.5 * (low + high)
        response = stable_sigmoid(
            base_logit[..., np.newaxis] + offsets - midpoint[..., np.newaxis]
        )
        marginalized = np.sum(response * weight_array, axis=-1)
        # The marginalized response is strictly decreasing in kappa.
        move_low_up = marginalized > base
        low = np.where(move_low_up, midpoint, low)
        high = np.where(move_low_up, high, midpoint)

    result = 0.5 * (low + high)
    result = np.where(base == 0.0, np.inf, result)
    result = np.where(base == 1.0, -np.inf, result)
    return result


def _broadcast_context(
    base_completeness: Any,
    redshift: Any,
    log_luminosity: Any,
    magnitude: Any | None,
    *,
    require_magnitude: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    base = np.asarray(base_completeness, dtype=float)
    z = np.asarray(redshift, dtype=float)
    log_l = np.asarray(log_luminosity, dtype=float)
    if require_magnitude:
        if magnitude is None:
            raise ValueError(
                "magnitude is required when magnitude interactions are enabled."
            )
        base, z, log_l, mag = np.broadcast_arrays(
            base, z, log_l, np.asarray(magnitude, dtype=float)
        )
        return base, z, log_l, mag
    base, z, log_l = np.broadcast_arrays(base, z, log_l)
    return base, z, log_l, None


def calibrate_response_kappa(
    base_completeness: Any,
    redshift: Any,
    log_luminosity: Any,
    coefficients: Mapping[str, Any] | Any,
    *,
    config: LatentAlphaConfig,
    magnitude: Any | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Calibrate kappa against the configured luminosity-dependent parent."""

    base, z, log_l, mag = _broadcast_context(
        base_completeness,
        redshift,
        log_luminosity,
        magnitude,
        require_magnitude=config.include_magnitude_interactions,
    )
    standard_nodes, weights = normal_gauss_hermite_nodes(config.quadrature_order)
    mean = parent_alpha_mean_from_config(log_l, config, parameters=parameters)
    alpha_nodes = mean[..., np.newaxis] + config.sigma * standard_nodes
    z_nodes = np.broadcast_to(z[..., np.newaxis], alpha_nodes.shape)
    mag_nodes = (
        None
        if mag is None
        else np.broadcast_to(mag[..., np.newaxis], alpha_nodes.shape)
    )
    offsets = response_logit_offset(
        alpha_nodes,
        z_nodes,
        coefficients,
        config=config,
        magnitude=mag_nodes,
    )
    return solve_response_kappa(base, offsets, weights=weights)


def bounded_response_from_kappa(
    base_completeness: Any,
    logit_offset: Any,
    kappa: Any,
) -> np.ndarray:
    """Evaluate a bounded response from a pre-calibrated normalization.

    This low-level helper avoids recalibrating kappa when evaluating many
    jointly indexed alpha posterior draws for the same object.  Callers can use
    ``base[..., None]``, ``kappa[..., None]``, and draw-shaped logit offsets.
    """

    base, offset, normalization = np.broadcast_arrays(
        np.asarray(base_completeness, dtype=float),
        np.asarray(logit_offset, dtype=float),
        np.asarray(kappa, dtype=float),
    )
    response = stable_sigmoid(stable_logit(base) + offset - normalization)
    response = np.where(base == 0.0, 0.0, response)
    response = np.where(base == 1.0, 1.0, response)
    return np.clip(response, 0.0, 1.0)


def bounded_alpha_completeness(
    base_completeness: Any,
    alpha_nu: Any,
    redshift: Any,
    log_luminosity: Any,
    coefficients: Mapping[str, Any] | Any,
    *,
    config: LatentAlphaConfig,
    magnitude: Any | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Evaluate the normalized, bounded alpha-dependent completeness."""

    if config.include_magnitude_interactions:
        if magnitude is None:
            raise ValueError(
                "magnitude is required when magnitude interactions are enabled."
            )
        base, alpha, z, log_l, mag = np.broadcast_arrays(
            np.asarray(base_completeness, dtype=float),
            np.asarray(alpha_nu, dtype=float),
            np.asarray(redshift, dtype=float),
            np.asarray(log_luminosity, dtype=float),
            np.asarray(magnitude, dtype=float),
        )
    else:
        base, alpha, z, log_l = np.broadcast_arrays(
            np.asarray(base_completeness, dtype=float),
            np.asarray(alpha_nu, dtype=float),
            np.asarray(redshift, dtype=float),
            np.asarray(log_luminosity, dtype=float),
        )
        mag = None
    kappa = calibrate_response_kappa(
        base,
        z,
        log_l,
        coefficients,
        config=config,
        magnitude=mag,
        parameters=parameters,
    )
    offset = response_logit_offset(
        alpha, z, coefficients, config=config, magnitude=mag
    )
    return bounded_response_from_kappa(base, offset, kappa)


def marginalized_alpha_completeness(
    base_completeness: Any,
    redshift: Any,
    log_luminosity: Any,
    coefficients: Mapping[str, Any] | Any,
    *,
    config: LatentAlphaConfig,
    magnitude: Any | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Marginalize the bounded response over the configured alpha parent."""

    base, z, log_l, mag = _broadcast_context(
        base_completeness,
        redshift,
        log_luminosity,
        magnitude,
        require_magnitude=config.include_magnitude_interactions,
    )
    standard_nodes, weights = normal_gauss_hermite_nodes(config.quadrature_order)
    mean = parent_alpha_mean_from_config(log_l, config, parameters=parameters)
    alpha_nodes = mean[..., np.newaxis] + config.sigma * standard_nodes
    z_nodes = np.broadcast_to(z[..., np.newaxis], alpha_nodes.shape)
    mag_nodes = (
        None
        if mag is None
        else np.broadcast_to(mag[..., np.newaxis], alpha_nodes.shape)
    )
    offsets = response_logit_offset(
        alpha_nodes,
        z_nodes,
        coefficients,
        config=config,
        magnitude=mag_nodes,
    )
    kappa = solve_response_kappa(base, offsets, weights=weights)
    response = bounded_response_from_kappa(
        base[..., np.newaxis], offsets, kappa[..., np.newaxis]
    )
    return np.sum(response * weights, axis=-1)


def deterministic_joint_draw_indices(
    input_count: int = JOINT_DRAW_INPUT_COUNT,
    selected_count: int = JOINT_DRAW_SELECTED_COUNT,
) -> np.ndarray:
    """Select deterministic midpoint-quantile indices from joint draws.

    The canonical 16-of-64 selection is ``[2, 6, ..., 62]``.  Applying the
    returned indices to every posterior field preserves their joint indexing.
    """

    if input_count < 1 or selected_count < 1:
        raise ValueError("Draw counts must be positive.")
    if selected_count > input_count:
        raise ValueError("selected_count cannot exceed input_count.")
    indices = np.floor(
        (np.arange(selected_count, dtype=float) + 0.5)
        * float(input_count)
        / float(selected_count)
    ).astype(int)
    if np.unique(indices).size != selected_count:
        raise RuntimeError("Midpoint draw selection unexpectedly produced duplicates.")
    return indices


def select_deterministic_joint_draws(
    draws: Mapping[str, Any] | Any,
    *,
    axis: int = -1,
    input_count: int = JOINT_DRAW_INPUT_COUNT,
    selected_count: int = JOINT_DRAW_SELECTED_COUNT,
) -> dict[str, np.ndarray] | np.ndarray:
    """Apply one deterministic index set to an array or mapping of joint draws."""

    indices = deterministic_joint_draw_indices(input_count, selected_count)

    def select_one(value: Any, label: str) -> np.ndarray:
        array = np.asarray(value)
        normalized_axis = axis if axis >= 0 else array.ndim + axis
        if normalized_axis < 0 or normalized_axis >= array.ndim:
            raise ValueError(f"axis {axis} is invalid for {label} with ndim={array.ndim}.")
        if array.shape[normalized_axis] != input_count:
            raise ValueError(
                f"{label} has {array.shape[normalized_axis]} draws on axis {axis}; "
                f"expected exactly {input_count}."
            )
        return np.take(array, indices, axis=normalized_axis)

    if isinstance(draws, Mapping):
        if not draws:
            raise ValueError("Joint-draw mapping cannot be empty.")
        return {str(name): select_one(value, str(name)) for name, value in draws.items()}
    return select_one(draws, "draws")


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def latent_alpha_serialization_payload(
    config: LatentAlphaConfig,
    coefficients: Mapping[str, Any] | Any | None = None,
) -> dict[str, Any]:
    """Return the canonical cache/checkpoint payload for this model."""

    names = response_coefficient_names(config.include_magnitude_interactions)
    if coefficients is None:
        coefficient_payload = None
    else:
        vector = response_coefficient_vector(
            coefficients,
            include_magnitude_interactions=config.include_magnitude_interactions,
        )
        coefficient_payload = {
            name: float(value) for name, value in zip(names, vector, strict=True)
        }
    return {
        "schema_version": LATENT_ALPHA_SCHEMA_VERSION,
        "config": config.to_dict(),
        "coefficient_names": list(names),
        "coefficient_priors": response_coefficient_prior_specs(
            config.include_magnitude_interactions
        ),
        "beta_parameter": BETA_ALPHA_L_PARAMETER,
        "beta_prior": {
            "distribution": "uniform",
            "low": config.beta_l_prior[0],
            "high": config.beta_l_prior[1],
            "units": "alpha_nu_per_dex",
        },
        "coefficients": coefficient_payload,
        "quadrature": {
            "kind": "gauss_hermite_standard_normal",
            "order": config.quadrature_order,
        },
        "joint_draw_selection": {
            "input_count": JOINT_DRAW_INPUT_COUNT,
            "selected_count": JOINT_DRAW_SELECTED_COUNT,
            "indices": deterministic_joint_draw_indices().tolist(),
        },
    }


def latent_alpha_config_hash(
    config: LatentAlphaConfig,
    coefficients: Mapping[str, Any] | Any | None = None,
) -> str:
    payload = latent_alpha_serialization_payload(config, coefficients)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def latent_alpha_provenance(
    config: LatentAlphaConfig,
    coefficients: Mapping[str, Any] | Any | None = None,
) -> dict[str, Any]:
    """Return structured, JSON-safe scientific provenance."""

    payload = latent_alpha_serialization_payload(config, coefficients)
    payload.update(
        {
            "config_hash_sha256": latent_alpha_config_hash(config, coefficients),
            "parent_distribution": {
                "family": "normal",
                "mean_equation": "mu + beta_alpha_L * (logL - logL_pivot)",
                "redshift_evolution": "none",
                "positive_beta_meaning": "more_luminous_is_bluer",
                "luminosity_state": config.luminosity_state,
                "lf_model": config.lf_model,
                "shen_lf_mode": config.shen_lf_mode,
                "luminosity_definition": "log10(nu_Lnu_2500 / erg_s^-1)",
                "m2500_ab_conversion": (
                    "log10(nuLnu)=log10(nu_2500)+"
                    "AB_absolute_mag_zeropoint/2.5-0.4*M2500"
                ),
                "nu_2500_hz": NU_2500_HZ,
                "ab_absolute_mag_zeropoint": AB_ABSOLUTE_MAG_ZEROPOINT,
            },
            "selection_response": {
                "equation": "sigmoid(logit(C3) + eta(alpha,z,m) - kappa)",
                "alpha_basis": ["u", "u^2 - 1"],
                "redshift_basis": ["P0", "P1", "P2", "P3"],
                "magnitude_interactions": config.include_magnitude_interactions,
                "normalization": "gauss_hermite_marginal_equals_C3",
                "range": [0.0, 1.0],
            },
        }
    )
    return payload


__all__ = [
    "BASE_RESPONSE_COEFFICIENT_PRIOR",
    "BETA_ALPHA_L_PARAMETER",
    "BETA_ALPHA_L_PRIOR",
    "AB_ABSOLUTE_MAG_ZEROPOINT",
    "DEFAULT_ALPHA_MU",
    "DEFAULT_ALPHA_SIGMA",
    "DEFAULT_LOGL_PIVOT",
    "GAUSS_HERMITE_ORDER",
    "JOINT_DRAW_INPUT_COUNT",
    "JOINT_DRAW_SELECTED_COUNT",
    "LATENT_ALPHA_MODES",
    "LATENT_ALPHA_SCHEMA_VERSION",
    "MAGNITUDE_RESPONSE_COEFFICIENT_PRIOR",
    "M2500_TO_LOG_NU_LNU_INTERCEPT",
    "NU_2500_HZ",
    "RESPONSE_COEFFICIENT_PRIOR_SIGMA",
    "LatentAlphaConfig",
    "absolute_m2500_to_log_nu_lnu",
    "bounded_alpha_completeness",
    "bounded_response_from_kappa",
    "calibrate_response_kappa",
    "deterministic_joint_draw_indices",
    "latent_alpha_config_hash",
    "latent_alpha_parameter_prior_specs",
    "latent_alpha_provenance",
    "latent_alpha_serialization_payload",
    "marginalized_alpha_completeness",
    "normal_gauss_hermite_nodes",
    "parent_alpha_logpdf",
    "parent_alpha_mean",
    "parent_alpha_mean_from_config",
    "parent_alpha_pdf",
    "resolve_lf_luminosity_state",
    "response_coefficient_names",
    "response_coefficient_prior_specs",
    "response_coefficient_vector",
    "response_design_matrix",
    "response_logit_offset",
    "select_deterministic_joint_draws",
    "solve_response_kappa",
    "stable_logit",
    "stable_sigmoid",
]
