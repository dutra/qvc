"""Simple, marginal-preserving fitted-colour completeness utilities.

The empirical two- and three-dimensional completeness surfaces used by QVC
already average over the colours of the targeted population.  This module
therefore redistributes that base probability as a function of a fitted
SDSS-PSF ``g-i`` colour; it does not multiply by a second survey efficiency.

The parent colour model is represented by a hash-validated regular grid of
qsogen mean colours.  At fixed host fraction the parent is a Normal around the
grid mean.  The two-dimensional model is the corresponding weighted mixture
over the existing QVC host-fraction population.  All parent lookups are strict:
extrapolation and edge clamping are deliberately forbidden.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Literal

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import expit, logit, ndtr


COLOR_PARENT_CACHE_FORMAT = "qvc_qsogen_color_parent_v2"
FITTED_COLOR_SCHEMA_VERSION = "qvc_fitted_color_completeness_v1"
COLOR_STRENGTH_PARAMETER = "s_color"
COLOR_STRENGTH_PRIOR = (-1.0, 1.0)
DEFAULT_COLOR_PARENT_SIGMA = 0.20
MAX_COLOR_PARENT_FHOST_SPACING = 0.002
COLOR_MODEL = "qsogen_delta_gi"

JOINT_COLOR_DRAW_INPUT_COUNT = 64
JOINT_COLOR_DRAW_SELECTED_COUNT = 16
DEFAULT_HOST_QUADRATURE_ORDER = 12

QSOGEN_COMMIT = "d2f9abf1ad23c489da8857f7e3c1bca862105d22"
QSOGEN_SOURCE_URL = "https://github.com/MJTemple/qsogen"
QSOGEN_LICENSE = "MIT"
JAXSEDFIT_FILTER_COMMIT = "bc9da74735260bd33b3da2076fd7929fdd592e0d"
REFERENCE_COSMOLOGY = {"H0_km_s_Mpc": 70.0, "Omega_m": 0.3}


class ColorParentSupportError(ValueError):
    """Raised when a parent-colour lookup would require extrapolation."""


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _normalise_path(path: str | os.PathLike[str]) -> str:
    value = os.fspath(path).strip()
    if not value:
        raise ValueError("parent_file cannot be empty.")
    return value


def _validate_sha256(value: str, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA256 digest.")
    return normalized


@dataclass(frozen=True, slots=True)
class FittedColorConfig:
    """Immutable definition of the active one-parameter colour model.

    ``parent_cache_sha256`` is the scientific content hash stored inside the
    parent HDF5 file, not the byte hash of the container.  The path is retained
    for provenance but is intentionally excluded from :func:`fitted_color_config_hash`,
    so moving an identical cache does not change the model identity.
    """

    parent_file: str
    parent_cache_sha256: str
    parent_sigma: float = DEFAULT_COLOR_PARENT_SIGMA
    model: Literal["qsogen_delta_gi"] = COLOR_MODEL
    input_draw_count: int = JOINT_COLOR_DRAW_INPUT_COUNT
    selected_draw_count: int = JOINT_COLOR_DRAW_SELECTED_COUNT
    schema_version: str = FITTED_COLOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "parent_file", _normalise_path(self.parent_file))
        object.__setattr__(
            self,
            "parent_cache_sha256",
            _validate_sha256(
                self.parent_cache_sha256, name="parent_cache_sha256"
            ),
        )
        if self.model != COLOR_MODEL:
            raise ValueError(f"model must be {COLOR_MODEL!r}.")
        if not np.isfinite(self.parent_sigma) or self.parent_sigma <= 0.0:
            raise ValueError("parent_sigma must be finite and positive.")
        if self.input_draw_count != JOINT_COLOR_DRAW_INPUT_COUNT:
            raise ValueError(
                f"input_draw_count is fixed at {JOINT_COLOR_DRAW_INPUT_COUNT}."
            )
        if self.selected_draw_count != JOINT_COLOR_DRAW_SELECTED_COUNT:
            raise ValueError(
                f"selected_draw_count is fixed at {JOINT_COLOR_DRAW_SELECTED_COUNT}."
            )
        if self.schema_version != FITTED_COLOR_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {FITTED_COLOR_SCHEMA_VERSION!r}."
            )

    @classmethod
    def from_parent_file(
        cls,
        parent_file: str | os.PathLike[str],
        *,
        parent_sigma: float = DEFAULT_COLOR_PARENT_SIGMA,
    ) -> "FittedColorConfig":
        cache = read_qsogen_color_parent_cache(parent_file)
        cache.assert_converged_host_grid()
        return cls(
            parent_file=os.fspath(parent_file),
            parent_cache_sha256=cache.content_hash_sha256,
            parent_sigma=parent_sigma,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def color_parameter_prior_spec() -> dict[str, dict[str, float | str]]:
    """Return the single sampled colour-response prior."""

    return {
        COLOR_STRENGTH_PARAMETER: {
            "distribution": "uniform",
            "low": COLOR_STRENGTH_PRIOR[0],
            "high": COLOR_STRENGTH_PRIOR[1],
        }
    }


def deterministic_color_draw_indices(
    input_count: int = JOINT_COLOR_DRAW_INPUT_COUNT,
    selected_count: int = JOINT_COLOR_DRAW_SELECTED_COUNT,
) -> np.ndarray:
    """Return deterministic midpoint indices shared by all colour quantities."""

    if not isinstance(input_count, (int, np.integer)) or input_count <= 0:
        raise ValueError("input_count must be a positive integer.")
    if not isinstance(selected_count, (int, np.integer)) or selected_count <= 0:
        raise ValueError("selected_count must be a positive integer.")
    if selected_count > input_count:
        raise ValueError("selected_count cannot exceed input_count.")
    indices = np.floor(
        (np.arange(selected_count, dtype=float) + 0.5)
        * float(input_count)
        / float(selected_count)
    ).astype(np.int64)
    if np.unique(indices).size != selected_count:
        raise RuntimeError("Midpoint draw selection unexpectedly produced duplicates.")
    return indices


def select_deterministic_color_draws(
    draws: Any,
    *,
    axis: int = -1,
    input_count: int = JOINT_COLOR_DRAW_INPUT_COUNT,
    selected_count: int = JOINT_COLOR_DRAW_SELECTED_COUNT,
) -> np.ndarray:
    array = np.asarray(draws)
    normalized_axis = axis if axis >= 0 else array.ndim + axis
    if normalized_axis < 0 or normalized_axis >= array.ndim:
        raise ValueError(f"axis {axis} is invalid for an array with ndim={array.ndim}.")
    if array.shape[normalized_axis] != input_count:
        raise ValueError(
            f"Expected {input_count} draws on axis {axis}; got shape {array.shape}."
        )
    return np.take(
        array,
        deterministic_color_draw_indices(input_count, selected_count),
        axis=normalized_axis,
    )


def fitted_psf_g_minus_i(
    fluxes_mjy: Any,
    *,
    bands: tuple[str, ...] | list[str] = ("u", "g", "r", "i", "z"),
) -> np.ndarray:
    """Convert fitted total-PSF model fluxes to an AB ``g-i`` colour.

    Both bands share the AB zeropoint, so the colour is exactly
    ``-2.5*log10(F_g/F_i)`` for fluxes in any common linear unit.  No noisy
    catalogue photometry or DR16Q fallback is accepted here.
    """

    array = np.asarray(fluxes_mjy, dtype=float)
    if array.ndim < 1:
        raise ValueError("fluxes_mjy must have a final band axis.")
    normalized_bands = tuple(
        str(name)
        .strip()
        .lower()
        .replace("sdss:", "")
        .replace("_sdss", "")
        for name in bands
    )
    if array.shape[-1] != len(normalized_bands):
        raise ValueError(
            f"fluxes_mjy has {array.shape[-1]} bands but bands has "
            f"{len(normalized_bands)} entries."
        )
    if normalized_bands.count("g") != 1 or normalized_bands.count("i") != 1:
        raise ValueError("bands must contain exactly one SDSS g and one SDSS i band.")
    g_flux = array[..., normalized_bands.index("g")]
    i_flux = array[..., normalized_bands.index("i")]
    if (
        not np.all(np.isfinite(g_flux))
        or not np.all(np.isfinite(i_flux))
        or np.any(g_flux <= 0.0)
        or np.any(i_flux <= 0.0)
    ):
        raise ValueError("Fitted g and i total-PSF fluxes must be finite and positive.")
    return -2.5 * np.log10(g_flux / i_flux)


@lru_cache(maxsize=None)
def _normal_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(order, (int, np.integer)) or order <= 0:
        raise ValueError("quadrature order must be a positive integer.")
    raw_nodes, raw_weights = np.polynomial.hermite.hermgauss(int(order))
    nodes = np.sqrt(2.0) * raw_nodes
    weights = raw_weights / np.sqrt(np.pi)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def fixed_reference_log_l2500(magnitude: Any, redshift: Any) -> np.ndarray:
    """Return QVC's historical log-L2500 coordinate at fixed reference cosmology.

    This deliberately reproduces the coordinate used to fit the existing host
    population: ``M2500=m2500-DM`` and ``logL2500=(90-M2500)/2.5``.  It is a
    fixed preprocessing coordinate and cannot feed sampled cosmology back into
    the colour parent.
    """

    from astropy.cosmology import FlatLambdaCDM

    magnitude, redshift = np.broadcast_arrays(
        np.asarray(magnitude, dtype=float), np.asarray(redshift, dtype=float)
    )
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("magnitude must be finite.")
    if not np.all(np.isfinite(redshift)) or np.any(redshift <= 0.0):
        raise ValueError("redshift must be finite and strictly positive.")
    cosmology = FlatLambdaCDM(
        H0=REFERENCE_COSMOLOGY["H0_km_s_Mpc"],
        Om0=REFERENCE_COSMOLOGY["Omega_m"],
    )
    absolute_magnitude = magnitude - cosmology.distmod(redshift).value
    return (90.0 - absolute_magnitude) / 2.5


def fixed_reference_host_fraction_quadrature(
    magnitude: Any,
    redshift: Any,
    host_population_model: Mapping[str, Any],
    *,
    order: int = DEFAULT_HOST_QUADRATURE_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic logit-Normal host nodes for the 2D colour parent.

    The required model fields are the existing QVC generalized-sigmoid
    parameters ``x0``, ``k``, ``nu`` and ``sigma_host_logit``.  Returned nodes
    have ``broadcast(magnitude, redshift).shape + (order,)``; returned weights
    are the shared one-dimensional Normal-quadrature weights.
    """

    required = {"x0", "k", "nu", "sigma_host_logit"}
    missing = sorted(required.difference(host_population_model))
    if missing:
        raise ValueError(f"host_population_model lacks required fields: {missing}.")
    parameters = {
        name: float(host_population_model[name])
        for name in required
    }
    if not all(np.isfinite(value) for value in parameters.values()):
        raise ValueError("Host-population parameters must be finite.")
    if parameters["k"] <= 0.0 or parameters["nu"] <= 0.0:
        raise ValueError("Host-population k and nu must be positive.")
    if parameters["sigma_host_logit"] < 0.0:
        raise ValueError("sigma_host_logit cannot be negative.")
    clip_eps = float(host_population_model.get("clip_eps", 1.0e-6))
    if not np.isfinite(clip_eps) or not 0.0 < clip_eps < 0.5:
        raise ValueError("Host-population clip_eps must lie in (0, 0.5).")
    log_l2500 = fixed_reference_log_l2500(magnitude, redshift)
    argument = np.clip(
        parameters["k"] * (log_l2500 - parameters["x0"]), -60.0, 60.0
    )
    mean = 1.0 / np.power(1.0 + np.exp(argument), parameters["nu"])
    mean = np.clip(mean, clip_eps, 1.0 - clip_eps)
    normal_nodes, weights = _normal_quadrature(order)
    nodes = expit(
        logit(mean)[..., np.newaxis]
        + parameters["sigma_host_logit"] * normal_nodes
    )
    return np.asarray(nodes, dtype=float), np.array(weights, copy=True)


def _validate_response_inputs(
    base_completeness: Any,
    parent_percentile: Any,
    color_strength: Any,
    *,
    require_positive_base: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base, percentile, strength = np.broadcast_arrays(
        np.asarray(base_completeness, dtype=float),
        np.asarray(parent_percentile, dtype=float),
        np.asarray(color_strength, dtype=float),
    )
    if not np.all(np.isfinite(base)) or np.any((base < 0.0) | (base > 1.0)):
        raise ValueError("base_completeness must be finite and lie in [0, 1].")
    if require_positive_base and np.any(base <= 0.0):
        raise ValueError(
            "Selected objects require strictly positive base completeness."
        )
    if not np.all(np.isfinite(percentile)) or np.any(
        (percentile < 0.0) | (percentile > 1.0)
    ):
        raise ValueError("parent_percentile must be finite and lie in [0, 1].")
    if not np.all(np.isfinite(strength)) or np.any(
        (strength < COLOR_STRENGTH_PRIOR[0])
        | (strength > COLOR_STRENGTH_PRIOR[1])
    ):
        raise ValueError("color_strength must be finite and lie in [-1, 1].")
    return base, percentile, strength


def bounded_color_completeness_xp(
    base_completeness: Any,
    parent_percentile: Any,
    color_strength: Any,
    *,
    xp: Any = np,
) -> Any:
    """Backend-neutral unchecked response primitive for NumPy or JAX."""

    base = xp.asarray(base_completeness)
    percentile = xp.asarray(parent_percentile)
    strength = xp.asarray(color_strength)
    return base + strength * base * (1.0 - base) * (1.0 - 2.0 * percentile)


def bounded_color_completeness(
    base_completeness: Any,
    parent_percentile: Any,
    color_strength: Any,
) -> np.ndarray:
    """Evaluate ``B + s B(1-B)(1-2q)`` after strict validation."""

    base, percentile, strength = _validate_response_inputs(
        base_completeness,
        parent_percentile,
        color_strength,
        require_positive_base=False,
    )
    response = bounded_color_completeness_xp(base, percentile, strength)
    # The expression is analytically bounded for the validated domain.  A
    # tolerance check catches regressions without hiding them via clipping.
    tolerance = 16.0 * np.finfo(float).eps
    if np.any((response < -tolerance) | (response > 1.0 + tolerance)):
        raise RuntimeError("Bounded colour-completeness invariant was violated.")
    return np.asarray(response)


def color_relative_selection_factor_xp(
    base_completeness: Any,
    parent_percentile: Any,
    color_strength: Any,
    *,
    xp: Any = np,
) -> Any:
    """Backend-neutral ``C_color / B`` primitive for a selected object."""

    base = xp.asarray(base_completeness)
    percentile = xp.asarray(parent_percentile)
    strength = xp.asarray(color_strength)
    return 1.0 + strength * (1.0 - base) * (1.0 - 2.0 * percentile)


def color_relative_selection_factor(
    base_completeness: Any,
    parent_percentile: Any,
    color_strength: Any,
) -> np.ndarray:
    """Evaluate the selected-object relative factor with ``B=0`` rejected."""

    base, percentile, strength = _validate_response_inputs(
        base_completeness,
        parent_percentile,
        color_strength,
        require_positive_base=True,
    )
    factor = np.asarray(
        color_relative_selection_factor_xp(base, percentile, strength), dtype=float
    )
    tolerance = 16.0 * np.finfo(float).eps
    if np.any((factor < -tolerance) | (factor > 2.0 + tolerance)):
        raise RuntimeError("Colour relative-factor invariant was violated.")
    return factor


def mean_color_relative_factor(
    base_completeness: Any,
    parent_percentile: Any,
    color_strength: Any,
    *,
    axis: int = -1,
) -> np.ndarray:
    return np.mean(
        color_relative_selection_factor(
            base_completeness, parent_percentile, color_strength
        ),
        axis=axis,
    )


def log_mean_color_relative_factor(
    base_completeness: Any,
    parent_percentile: Any,
    color_strength: Any,
    *,
    axis: int = -1,
) -> np.ndarray:
    mean_factor = mean_color_relative_factor(
        base_completeness, parent_percentile, color_strength, axis=axis
    )
    if np.any(mean_factor <= 0.0):
        raise ValueError(
            "Mean colour relative factor is zero; the selected object is impossible "
            "under this boundary response."
        )
    return np.log(mean_factor)


def _readonly_float64(values: Any, *, name: str, ndim: int) -> np.ndarray:
    array = np.array(values, dtype=np.float64, order="C", copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}; got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    array.setflags(write=False)
    return array


def _validate_axis(values: Any, *, name: str, bounds: tuple[float, float] | None = None) -> np.ndarray:
    axis = _readonly_float64(values, name=name, ndim=1)
    if axis.size < 2:
        raise ValueError(f"{name} must contain at least two grid points.")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    if bounds is not None and (axis[0] < bounds[0] or axis[-1] > bounds[1]):
        raise ValueError(f"{name} must lie within [{bounds[0]}, {bounds[1]}].")
    return axis


def _validate_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    try:
        normalized = json.loads(_canonical_json(dict(provenance)))
    except (TypeError, ValueError) as exc:
        raise ValueError("provenance must be a finite JSON-safe mapping.") from exc
    required = {
        "construction",
        "qsogen_commit",
        "qsogen_source_url",
        "qsogen_license",
        "jaxsedfit_filter_commit",
        "asset_sha256",
        "filter_names",
        "reference_cosmology",
        "magnitude_state",
        "source_internal_attenuation",
        "host_scaling",
        "m2500_to_qsogen_luminosity",
        "parent_population_interpretation",
        "residual_scatter_limitation",
    }
    missing = sorted(required.difference(normalized))
    if missing:
        raise ValueError(f"qsogen parent provenance lacks required fields: {missing}.")
    if normalized["qsogen_commit"] != QSOGEN_COMMIT:
        raise ValueError(
            f"qsogen_commit must be the pinned commit {QSOGEN_COMMIT}."
        )
    if normalized["qsogen_source_url"] != QSOGEN_SOURCE_URL:
        raise ValueError(f"qsogen_source_url must be {QSOGEN_SOURCE_URL!r}.")
    if normalized["qsogen_license"] != QSOGEN_LICENSE:
        raise ValueError(f"qsogen_license must be {QSOGEN_LICENSE!r}.")
    if normalized["jaxsedfit_filter_commit"] != JAXSEDFIT_FILTER_COMMIT:
        raise ValueError(
            "jaxsedfit_filter_commit must match the spectra-fit revision "
            f"{JAXSEDFIT_FILTER_COMMIT}."
        )
    if normalized["filter_names"] != ["g_sdss", "i_sdss"]:
        raise ValueError("filter_names must be ['g_sdss', 'i_sdss'] in that order.")
    if normalized["magnitude_state"] != "attenuation_retaining":
        raise ValueError("magnitude_state must be 'attenuation_retaining'.")
    assets = normalized["asset_sha256"]
    if not isinstance(assets, dict) or not assets:
        raise ValueError("asset_sha256 must be a nonempty mapping.")
    for name, digest in assets.items():
        _validate_sha256(digest, name=f"asset_sha256[{name!r}]")
    cosmology = normalized["reference_cosmology"]
    if not isinstance(cosmology, dict) or set(cosmology) != set(REFERENCE_COSMOLOGY):
        raise ValueError(
            f"reference_cosmology must contain exactly {sorted(REFERENCE_COSMOLOGY)}."
        )
    for name, expected in REFERENCE_COSMOLOGY.items():
        if not np.isclose(float(cosmology[name]), expected, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"reference_cosmology[{name!r}] must equal {expected}."
            )
    return normalized


def _content_hash(
    magnitude_grid: np.ndarray,
    redshift_grid: np.ndarray,
    f_host_grid: np.ndarray,
    agn_only_mean_g_minus_i: np.ndarray,
    total_mean_g_minus_i: np.ndarray,
    provenance: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(COLOR_PARENT_CACHE_FORMAT.encode("ascii"))
    digest.update(_canonical_json(provenance).encode("utf-8"))
    for name, values in (
        ("apparent_magnitude_2500", magnitude_grid),
        ("redshift", redshift_grid),
        ("f_host_2500_psf", f_host_grid),
        ("agn_only_mean_g_minus_i", agn_only_mean_g_minus_i),
        ("total_mean_g_minus_i", total_mean_g_minus_i),
    ):
        array = np.ascontiguousarray(values, dtype="<f8")
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class QsogenColorParentCache:
    """Regular-grid qsogen mean colours used to evaluate parent percentiles."""

    magnitude_grid: np.ndarray
    redshift_grid: np.ndarray
    f_host_grid: np.ndarray
    agn_only_mean_g_minus_i: np.ndarray
    total_mean_g_minus_i: np.ndarray
    provenance: Mapping[str, Any]
    content_hash_sha256: str | None = None

    def __post_init__(self) -> None:
        magnitude = _validate_axis(self.magnitude_grid, name="magnitude_grid")
        redshift = _validate_axis(self.redshift_grid, name="redshift_grid")
        if redshift[0] <= 0.0:
            raise ValueError("redshift_grid must be strictly positive.")
        f_host = _validate_axis(
            self.f_host_grid, name="f_host_grid", bounds=(0.0, 1.0)
        )
        if f_host[0] != 0.0 or f_host[-1] != 1.0:
            raise ValueError("f_host_grid must span the closed interval [0, 1].")
        agn = _readonly_float64(
            self.agn_only_mean_g_minus_i,
            name="agn_only_mean_g_minus_i",
            ndim=2,
        )
        total = _readonly_float64(
            self.total_mean_g_minus_i, name="total_mean_g_minus_i", ndim=3
        )
        expected_agn = (magnitude.size, redshift.size)
        expected_total = expected_agn + (f_host.size,)
        if agn.shape != expected_agn:
            raise ValueError(
                f"agn_only_mean_g_minus_i has shape {agn.shape}; expected {expected_agn}."
            )
        if total.shape != expected_total:
            raise ValueError(
                f"total_mean_g_minus_i has shape {total.shape}; expected {expected_total}."
            )
        if not np.allclose(total[..., 0], agn, rtol=0.0, atol=2e-10):
            raise ValueError(
                "total_mean_g_minus_i at f_host=0 must equal the AGN-only mean."
            )
        provenance = _validate_provenance(self.provenance)
        computed_hash = _content_hash(magnitude, redshift, f_host, agn, total, provenance)
        if self.content_hash_sha256 is not None:
            supplied_hash = _validate_sha256(
                self.content_hash_sha256, name="content_hash_sha256"
            )
            if supplied_hash != computed_hash:
                raise ValueError(
                    "qsogen parent cache content hash does not match its arrays/provenance."
                )
        object.__setattr__(self, "magnitude_grid", magnitude)
        object.__setattr__(self, "redshift_grid", redshift)
        object.__setattr__(self, "f_host_grid", f_host)
        object.__setattr__(self, "agn_only_mean_g_minus_i", agn)
        object.__setattr__(self, "total_mean_g_minus_i", total)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "content_hash_sha256", computed_hash)

    @property
    def support(self) -> dict[str, tuple[float, float]]:
        return {
            "apparent_magnitude_2500": (
                float(self.magnitude_grid[0]),
                float(self.magnitude_grid[-1]),
            ),
            "redshift": (
                float(self.redshift_grid[0]),
                float(self.redshift_grid[-1]),
            ),
            "f_host_2500_psf": (
                float(self.f_host_grid[0]),
                float(self.f_host_grid[-1]),
            ),
        }

    def assert_converged_host_grid(
        self,
        *,
        max_spacing: float = MAX_COLOR_PARENT_FHOST_SPACING,
    ) -> None:
        if not np.isfinite(max_spacing) or max_spacing <= 0.0:
            raise ValueError("max_spacing must be finite and positive.")
        realized = float(np.max(np.diff(self.f_host_grid)))
        tolerance = 32.0 * np.finfo(float).eps
        if realized > float(max_spacing) + tolerance:
            raise ValueError(
                "qsogen parent f_host grid is too coarse for fitted-color "
                f"inference: maximum spacing={realized:.8g}, required "
                f"<={float(max_spacing):.8g}. Regenerate the v2 parent cache."
            )

    def assert_support(
        self,
        magnitude: Any,
        redshift: Any,
        f_host: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if f_host is None:
            mag, z = np.broadcast_arrays(
                np.asarray(magnitude, dtype=float), np.asarray(redshift, dtype=float)
            )
            host = None
        else:
            mag, z, host = np.broadcast_arrays(
                np.asarray(magnitude, dtype=float),
                np.asarray(redshift, dtype=float),
                np.asarray(f_host, dtype=float),
            )
        for name, values, grid in (
            ("apparent_magnitude_2500", mag, self.magnitude_grid),
            ("redshift", z, self.redshift_grid),
        ):
            if not np.all(np.isfinite(values)):
                raise ColorParentSupportError(f"{name} query contains nonfinite values.")
            outside = (values < grid[0]) | (values > grid[-1])
            if np.any(outside):
                raise ColorParentSupportError(
                    f"{name} query is outside strict support [{grid[0]}, {grid[-1]}]; "
                    f"count={int(np.count_nonzero(outside))}."
                )
        if host is not None:
            if not np.all(np.isfinite(host)):
                raise ColorParentSupportError("f_host_2500_psf query contains nonfinite values.")
            outside = (host < self.f_host_grid[0]) | (host > self.f_host_grid[-1])
            if np.any(outside):
                raise ColorParentSupportError(
                    "f_host_2500_psf query is outside strict support "
                    f"[{self.f_host_grid[0]}, {self.f_host_grid[-1]}]; "
                    f"count={int(np.count_nonzero(outside))}."
                )
        return mag, z, host

    def agn_only_mean_color(self, magnitude: Any, redshift: Any) -> np.ndarray:
        mag, z, _ = self.assert_support(magnitude, redshift)
        interpolator = RegularGridInterpolator(
            (self.magnitude_grid, self.redshift_grid),
            self.agn_only_mean_g_minus_i,
            method="linear",
            bounds_error=True,
        )
        points = np.column_stack((mag.ravel(), z.ravel()))
        return np.asarray(interpolator(points), dtype=float).reshape(mag.shape)

    def total_mean_color(
        self, magnitude: Any, redshift: Any, f_host: Any
    ) -> np.ndarray:
        mag, z, host = self.assert_support(magnitude, redshift, f_host)
        assert host is not None
        interpolator = RegularGridInterpolator(
            (self.magnitude_grid, self.redshift_grid, self.f_host_grid),
            self.total_mean_g_minus_i,
            method="linear",
            bounds_error=True,
        )
        points = np.column_stack((mag.ravel(), z.ravel(), host.ravel()))
        return np.asarray(interpolator(points), dtype=float).reshape(mag.shape)

    def percentile_3d(
        self,
        color_g_minus_i: Any,
        magnitude: Any,
        redshift: Any,
        f_host: Any,
        *,
        sigma: float = DEFAULT_COLOR_PARENT_SIGMA,
    ) -> np.ndarray:
        sigma = _validate_parent_sigma(sigma)
        color, mag, z, host = np.broadcast_arrays(
            np.asarray(color_g_minus_i, dtype=float),
            np.asarray(magnitude, dtype=float),
            np.asarray(redshift, dtype=float),
            np.asarray(f_host, dtype=float),
        )
        if not np.all(np.isfinite(color)):
            raise ValueError("color_g_minus_i must be finite.")
        mean = self.total_mean_color(mag, z, host)
        return np.asarray(ndtr((color - mean) / sigma), dtype=float)

    def percentile_2d(
        self,
        color_g_minus_i: Any,
        magnitude: Any,
        redshift: Any,
        host_fraction_nodes: Any,
        host_fraction_weights: Any,
        *,
        sigma: float = DEFAULT_COLOR_PARENT_SIGMA,
    ) -> np.ndarray:
        """Evaluate the host-marginal parent CDF as a Normal mixture.

        Nodes and weights may be one-dimensional (shared by every context) or
        have shape ``broadcast(color, magnitude, redshift).shape + (K,)``.
        This keeps the host-population model outside this cache and avoids a
        hidden dependence on sampled cosmology.
        """

        sigma = _validate_parent_sigma(sigma)
        color, mag, z = np.broadcast_arrays(
            np.asarray(color_g_minus_i, dtype=float),
            np.asarray(magnitude, dtype=float),
            np.asarray(redshift, dtype=float),
        )
        if not np.all(np.isfinite(color)):
            raise ValueError("color_g_minus_i must be finite.")
        self.assert_support(mag, z)
        nodes = np.asarray(host_fraction_nodes, dtype=float)
        weights = np.asarray(host_fraction_weights, dtype=float)
        if nodes.ndim == 1:
            nodes = np.broadcast_to(nodes, color.shape + nodes.shape)
        if weights.ndim == 1:
            weights = np.broadcast_to(weights, color.shape + weights.shape)
        if nodes.ndim != color.ndim + 1 or nodes.shape[:-1] != color.shape:
            raise ValueError(
                "host_fraction_nodes must be one-dimensional or have context shape plus K."
            )
        if weights.shape != nodes.shape:
            raise ValueError("host_fraction_weights must have the same shape as nodes.")
        if nodes.shape[-1] < 1:
            raise ValueError("At least one host-fraction node is required.")
        if not np.all(np.isfinite(nodes)) or np.any((nodes < 0.0) | (nodes > 1.0)):
            raise ValueError("host_fraction_nodes must be finite and lie in [0, 1].")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("host_fraction_weights must be finite and nonnegative.")
        weight_sum = np.sum(weights, axis=-1, keepdims=True)
        if np.any(weight_sum <= 0.0):
            raise ValueError("host_fraction_weights must have positive row sums.")
        normalized_weights = weights / weight_sum
        means = self.total_mean_color(
            np.broadcast_to(mag[..., np.newaxis], nodes.shape),
            np.broadcast_to(z[..., np.newaxis], nodes.shape),
            nodes,
        )
        component_cdf = ndtr((color[..., np.newaxis] - means) / sigma)
        return np.sum(normalized_weights * component_cdf, axis=-1)


def assert_parent_cache_matches_config(
    cache: QsogenColorParentCache, config: FittedColorConfig
) -> None:
    if cache.content_hash_sha256 != config.parent_cache_sha256:
        raise ValueError(
            "Loaded qsogen parent cache does not match FittedColorConfig."
        )


def _select_aligned_fitted_inputs(
    flux_draws_mjy: Any,
    magnitude_draws: Any,
    redshift: Any,
    *,
    bands: tuple[str, ...] | list[str],
    f_host_draws: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    flux = np.asarray(flux_draws_mjy, dtype=float)
    if flux.ndim < 2 or flux.shape[-2] != JOINT_COLOR_DRAW_INPUT_COUNT:
        raise ValueError(
            "flux_draws_mjy must have shape (..., 64, n_bands)."
        )
    magnitude = np.asarray(magnitude_draws, dtype=float)
    expected_scalar_shape = flux.shape[:-1]
    if magnitude.shape != expected_scalar_shape:
        raise ValueError(
            f"magnitude_draws has shape {magnitude.shape}; expected "
            f"{expected_scalar_shape} aligned with flux draws."
        )
    leading_shape = flux.shape[:-2]
    z = np.asarray(redshift, dtype=float)
    try:
        z = np.broadcast_to(z, leading_shape)
    except ValueError as exc:
        raise ValueError(
            f"redshift shape {z.shape} cannot broadcast to object shape {leading_shape}."
        ) from exc
    indices = deterministic_color_draw_indices()
    selected_flux = np.take(flux, indices, axis=-2)
    selected_magnitude = np.take(magnitude, indices, axis=-1)
    selected_color = fitted_psf_g_minus_i(selected_flux, bands=bands)
    selected_z = np.broadcast_to(
        z[..., np.newaxis], leading_shape + (JOINT_COLOR_DRAW_SELECTED_COUNT,)
    )
    if f_host_draws is None:
        selected_host = None
    else:
        host = np.asarray(f_host_draws, dtype=float)
        if host.shape != expected_scalar_shape:
            raise ValueError(
                f"f_host_draws has shape {host.shape}; expected "
                f"{expected_scalar_shape} aligned with flux draws."
            )
        selected_host = np.take(host, indices, axis=-1)
    return selected_color, selected_magnitude, selected_z, selected_host


def aligned_fitted_color_percentiles_3d(
    cache: QsogenColorParentCache,
    flux_draws_mjy: Any,
    magnitude_draws: Any,
    redshift: Any,
    f_host_draws: Any,
    *,
    bands: tuple[str, ...] | list[str] = ("u", "g", "r", "i", "z"),
    sigma: float = DEFAULT_COLOR_PARENT_SIGMA,
) -> np.ndarray:
    """Turn 64 aligned fitted photometry/host draws into 16 parent percentiles."""

    color, magnitude, z, host = _select_aligned_fitted_inputs(
        flux_draws_mjy,
        magnitude_draws,
        redshift,
        bands=bands,
        f_host_draws=f_host_draws,
    )
    assert host is not None
    return cache.percentile_3d(
        color, magnitude, z, host, sigma=sigma
    )


def aligned_fitted_color_percentiles_2d(
    cache: QsogenColorParentCache,
    flux_draws_mjy: Any,
    magnitude_draws: Any,
    redshift: Any,
    host_population_model: Mapping[str, Any],
    *,
    bands: tuple[str, ...] | list[str] = ("u", "g", "r", "i", "z"),
    sigma: float = DEFAULT_COLOR_PARENT_SIGMA,
    host_quadrature_order: int = DEFAULT_HOST_QUADRATURE_ORDER,
) -> np.ndarray:
    """Turn 64 aligned fitted draws into 16 host-marginal parent percentiles."""

    color, magnitude, z, _ = _select_aligned_fitted_inputs(
        flux_draws_mjy,
        magnitude_draws,
        redshift,
        bands=bands,
    )
    host_nodes, host_weights = fixed_reference_host_fraction_quadrature(
        magnitude,
        z,
        host_population_model,
        order=host_quadrature_order,
    )
    return cache.percentile_2d(
        color,
        magnitude,
        z,
        host_nodes,
        host_weights,
        sigma=sigma,
    )


def _validate_parent_sigma(value: float) -> float:
    sigma = float(value)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("parent sigma must be finite and positive.")
    return sigma


def write_qsogen_color_parent_cache(
    path: str | os.PathLike[str], cache: QsogenColorParentCache
) -> Path:
    """Atomically write a validated qsogen parent cache."""

    if not isinstance(cache, QsogenColorParentCache):
        raise TypeError("cache must be a QsogenColorParentCache.")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["qvc_qsogen_color_parent_format"] = COLOR_PARENT_CACHE_FORMAT
            handle.attrs["content_hash_sha256"] = cache.content_hash_sha256
            handle.attrs["provenance_json"] = _canonical_json(cache.provenance)
            grids = handle.create_group("grids")
            grids.create_dataset("apparent_magnitude_2500", data=cache.magnitude_grid)
            grids.create_dataset("redshift", data=cache.redshift_grid)
            grids.create_dataset("f_host_2500_psf", data=cache.f_host_grid)
            means = handle.create_group("mean_color")
            means.create_dataset(
                "agn_only_g_minus_i",
                data=cache.agn_only_mean_g_minus_i,
                compression="gzip",
                shuffle=True,
            )
            means.create_dataset(
                "total_g_minus_i",
                data=cache.total_mean_g_minus_i,
                compression="gzip",
                shuffle=True,
            )
            handle.flush()
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output


def read_qsogen_color_parent_cache(
    path: str | os.PathLike[str],
    *,
    expected_content_hash: str | None = None,
) -> QsogenColorParentCache:
    """Load and fully revalidate a parent cache, including its content hash."""

    source = Path(path)
    with h5py.File(source, "r") as handle:
        raw_format = handle.attrs.get("qvc_qsogen_color_parent_format")
        if isinstance(raw_format, bytes):
            raw_format = raw_format.decode("utf-8", errors="replace")
        if raw_format != COLOR_PARENT_CACHE_FORMAT:
            raise ValueError(
                f"Parent cache {source} has format {raw_format!r}; expected "
                f"{COLOR_PARENT_CACHE_FORMAT!r}."
            )
        if set(handle) != {"grids", "mean_color"}:
            raise ValueError("Parent cache must contain exactly grids and mean_color groups.")
        if set(handle["grids"]) != {
            "apparent_magnitude_2500",
            "redshift",
            "f_host_2500_psf",
        }:
            raise ValueError("Parent cache grids group has an incompatible dataset set.")
        if set(handle["mean_color"]) != {
            "agn_only_g_minus_i",
            "total_g_minus_i",
        }:
            raise ValueError("Parent cache mean_color group has an incompatible dataset set.")
        raw_hash = handle.attrs.get("content_hash_sha256")
        if isinstance(raw_hash, bytes):
            raw_hash = raw_hash.decode("utf-8", errors="replace")
        if raw_hash is None:
            raise ValueError("Parent cache lacks content_hash_sha256.")
        raw_provenance = handle.attrs.get("provenance_json")
        if isinstance(raw_provenance, bytes):
            raw_provenance = raw_provenance.decode("utf-8", errors="replace")
        if raw_provenance is None:
            raise ValueError("Parent cache lacks provenance_json.")
        try:
            provenance = json.loads(str(raw_provenance))
        except json.JSONDecodeError as exc:
            raise ValueError("Parent cache provenance_json is malformed.") from exc
        cache = QsogenColorParentCache(
            magnitude_grid=handle["grids/apparent_magnitude_2500"][:],
            redshift_grid=handle["grids/redshift"][:],
            f_host_grid=handle["grids/f_host_2500_psf"][:],
            agn_only_mean_g_minus_i=handle["mean_color/agn_only_g_minus_i"][:],
            total_mean_g_minus_i=handle["mean_color/total_g_minus_i"][:],
            provenance=provenance,
            content_hash_sha256=str(raw_hash),
        )
    if expected_content_hash is not None:
        expected = _validate_sha256(
            expected_content_hash, name="expected_content_hash"
        )
        if cache.content_hash_sha256 != expected:
            raise ValueError(
                "Parent cache does not match the expected scientific content hash."
            )
    return cache


def fitted_color_serialization_payload(config: FittedColorConfig) -> dict[str, Any]:
    return {
        "schema_version": FITTED_COLOR_SCHEMA_VERSION,
        "model": COLOR_MODEL,
        "parent_cache_sha256": config.parent_cache_sha256,
        "parent_sigma_mag": float(config.parent_sigma),
        "parameter_prior": color_parameter_prior_spec(),
        "joint_draw_selection": {
            "input_count": config.input_draw_count,
            "selected_count": config.selected_draw_count,
            "indices": deterministic_color_draw_indices().tolist(),
        },
    }


def fitted_color_config_hash(config: FittedColorConfig) -> str:
    return hashlib.sha256(
        _canonical_json(fitted_color_serialization_payload(config)).encode("utf-8")
    ).hexdigest()


def fitted_color_provenance(config: FittedColorConfig) -> dict[str, Any]:
    payload = fitted_color_serialization_payload(config)
    payload.update(
        {
            "config_hash_sha256": fitted_color_config_hash(config),
            "parent_file": config.parent_file,
            "fitted_color_definition": (
                "SDSS observed-frame total captured-PSF, MW-dereddened AB g-i; "
                "source attenuation, host, lines, FeII, Balmer continuum and IGM retained"
            ),
            "parent_percentile_definition": (
                "q=P_parent(g-i <= fitted_g-i | apparent_m2500,z[,f_host_2500_psf])"
            ),
            "response": {
                "equation": "C=B+s_color*B*(1-B)*(1-2*q)",
                "relative_factor": "C/B=1+s_color*(1-B)*(1-2*q)",
                "normalization": "E_parent[C|context]=B because q is Uniform(0,1)",
                "positive_s_color": "redder_objects_are_less_complete",
                "range": [0.0, 1.0],
            },
            "likelihood": {
                "selected_object_term": "log(mean_16_draws(C/B))",
                "normalization_integral": "unchanged_base_2d_or_3d_integral",
                "derived_color_prior_correction": "none",
                "cosmology_coupling": (
                    "none_with_fixed_parent_and_unweighted_log_mean_relative_factor; "
                    "factorizes_from_cosmology"
                ),
            },
            "parent_interpretation": {
                "role": "selection_sensitivity_reference_not_unbiased_parent_truth",
                "calibration_caveat": (
                    "qsogen_Temple_mean_colors_are_calibrated_to_observed_selected_DR16Q"
                ),
                "dust_tail_caveat": (
                    "E(B-V)=0_mean_plus_symmetric_Normal_scatter_omits_asymmetric_"
                    "internal_dust_red_tail"
                ),
            },
            "scientific_label": "global_qsogen_delta_gi_sensitivity",
            "scientific_use": "targeting_diagnostic_not_cosmology_correction",
            "not_exact_historical_targeting": True,
        }
    )
    return payload


__all__ = [
    "COLOR_MODEL",
    "COLOR_PARENT_CACHE_FORMAT",
    "COLOR_STRENGTH_PARAMETER",
    "COLOR_STRENGTH_PRIOR",
    "DEFAULT_COLOR_PARENT_SIGMA",
    "DEFAULT_HOST_QUADRATURE_ORDER",
    "FITTED_COLOR_SCHEMA_VERSION",
    "JOINT_COLOR_DRAW_INPUT_COUNT",
    "JOINT_COLOR_DRAW_SELECTED_COUNT",
    "JAXSEDFIT_FILTER_COMMIT",
    "MAX_COLOR_PARENT_FHOST_SPACING",
    "QSOGEN_COMMIT",
    "QSOGEN_LICENSE",
    "QSOGEN_SOURCE_URL",
    "REFERENCE_COSMOLOGY",
    "ColorParentSupportError",
    "FittedColorConfig",
    "QsogenColorParentCache",
    "aligned_fitted_color_percentiles_2d",
    "aligned_fitted_color_percentiles_3d",
    "assert_parent_cache_matches_config",
    "bounded_color_completeness",
    "bounded_color_completeness_xp",
    "color_parameter_prior_spec",
    "color_relative_selection_factor",
    "color_relative_selection_factor_xp",
    "deterministic_color_draw_indices",
    "fitted_psf_g_minus_i",
    "fixed_reference_host_fraction_quadrature",
    "fixed_reference_log_l2500",
    "fitted_color_config_hash",
    "fitted_color_provenance",
    "fitted_color_serialization_payload",
    "log_mean_color_relative_factor",
    "mean_color_relative_factor",
    "read_qsogen_color_parent_cache",
    "select_deterministic_color_draws",
    "write_qsogen_color_parent_cache",
]
