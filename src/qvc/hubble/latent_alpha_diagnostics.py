"""Diagnostics for the latent intrinsic-slope completeness model.

The diagnostics in this module are deliberately sampler-independent.  They
consume the authoritative aligned draws from a spectra-catalog v3 data frame,
one representative latent-alpha parameter vector, and the same three-
dimensional host-aware completeness callable used by the likelihood.  A
posterior may additionally be supplied to derive the representative vector
and to report correlations of ``beta_alpha_L`` with cosmology.

No derived-slope prior correction is applied here, matching the likelihood's
compact-posterior-draw approximation.  All 64 catalog draws are retained for
the diagnostic calculations; the canonical 16 draws are also evaluated so
their agreement can be recorded explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qvc.hubble.latent_alpha_completeness import (
    BETA_ALPHA_L_PARAMETER,
    JOINT_DRAW_INPUT_COUNT,
    bounded_response_from_kappa,
    calibrate_response_kappa,
    deterministic_joint_draw_indices,
    latent_alpha_provenance,
    normal_gauss_hermite_nodes,
    parent_alpha_mean_from_config,
    parent_alpha_pdf,
    response_coefficient_names,
    response_logit_offset,
    LatentAlphaConfig,
    absolute_m2500_to_log_nu_lnu,
)


LATENT_ALPHA_DIAGNOSTICS_SCHEMA_VERSION = "qvc_latent_alpha_diagnostics_v1"

_ALIGNED_DRAW_COLUMNS = (
    "f_host_2500_psf_draws",
    "alpha_nu_intrinsic_1450_2500_draws",
    "alpha_nu_attenuated_1450_2500_draws",
    "m_2500_dereddened_draws",
    "m_2500_attenuated_model_draws",
    "a_2500_galaxy_draws",
    "a_2500_internal_draws",
    "a_2500_total_draws",
)

_COSMOLOGY_PARAMETER_NAMES = frozenset(
    {
        "H0",
        "Om0",
        "Ode0",
        "Ok0",
        "w0",
        "wp",
        "wa",
        "q0",
        "j0",
        "zt",
    }
)


@dataclass(frozen=True, slots=True)
class LatentAlphaDiagnosticsResult:
    """Paths and in-memory summary produced by the diagnostics writer."""

    json_path: Path
    plot_paths: tuple[Path, ...]
    summary: dict[str, Any]


def _draw_matrix(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise KeyError(f"Missing required aligned v3 draw column {column!r}.")
    raw = np.asarray(frame[column])
    if raw.ndim == 1 and raw.dtype == object:
        try:
            raw = np.stack(raw)
        except ValueError as exc:
            raise ValueError(
                f"Could not stack aligned posterior column {column!r}."
            ) from exc
    try:
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Aligned posterior column {column!r} must be numeric."
        ) from exc
    expected = (len(frame), JOINT_DRAW_INPUT_COUNT)
    if values.shape != expected:
        raise ValueError(
            f"Aligned posterior column {column!r} has shape {values.shape}; "
            f"expected {expected}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"Aligned posterior column {column!r} must contain 64 finite draws "
            "for every object."
        )
    return values


def _finite_vector(values: Any, *, length: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape ({length},).")
    return result


def _posterior_mapping(
    posterior_samples: Mapping[str, Any] | Any | None,
    model_labels: Sequence[str] | None,
) -> dict[str, np.ndarray] | None:
    if posterior_samples is None:
        return None
    if isinstance(posterior_samples, Mapping):
        if model_labels is not None:
            missing = set(model_labels).difference(posterior_samples)
            if missing:
                raise KeyError(
                    "model_labels includes labels absent from posterior_samples: "
                    f"{sorted(missing)}"
                )
        labels = list(posterior_samples) if model_labels is None else list(model_labels)
        arrays = {label: np.asarray(posterior_samples[label], dtype=float) for label in labels}
        lengths = {array.size for array in arrays.values() if array.ndim == 1}
        if any(array.ndim != 1 for array in arrays.values()) or len(lengths) != 1:
            raise ValueError(
                "Mapping posterior_samples must contain equal-length 1D arrays."
            )
        if not lengths or next(iter(lengths)) < 2:
            raise ValueError("posterior_samples must contain at least two draws.")
        return arrays

    samples = np.asarray(posterior_samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("Array posterior_samples must have shape (draw, parameter).")
    if model_labels is None:
        raise ValueError("model_labels is required for array posterior_samples.")
    labels = [str(label) for label in model_labels]
    if samples.shape[1] != len(labels):
        raise ValueError(
            f"posterior_samples has {samples.shape[1]} columns but received "
            f"{len(labels)} model_labels."
        )
    if len(set(labels)) != len(labels):
        raise ValueError("model_labels must be unique.")
    if samples.shape[0] < 2:
        raise ValueError("posterior_samples must contain at least two draws.")
    return {label: samples[:, index] for index, label in enumerate(labels)}


def _clean_posterior(
    posterior: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray] | None:
    if posterior is None:
        return None
    arrays = list(posterior.values())
    finite = np.ones(arrays[0].shape, dtype=bool)
    for array in arrays:
        finite &= np.isfinite(array)
    if np.count_nonzero(finite) < 2:
        raise ValueError("Fewer than two jointly finite posterior draws remain.")
    return {label: array[finite] for label, array in posterior.items()}


def _representative_parameters(
    config: LatentAlphaConfig,
    parameters: Mapping[str, Any] | None,
    posterior: dict[str, np.ndarray] | None,
) -> dict[str, float]:
    required = list(response_coefficient_names(config.include_magnitude_interactions))
    if config.mode == "joint":
        required.append(BETA_ALPHA_L_PARAMETER)
    result: dict[str, float] = {}
    if parameters is not None:
        for label, value in parameters.items():
            array = np.asarray(value, dtype=float)
            if array.ndim == 0 and np.isfinite(array):
                result[str(label)] = float(array)
    elif posterior is None:
        raise ValueError(
            "Provide either representative parameters or posterior_samples."
        )

    missing = []
    for label in required:
        if label not in result:
            if posterior is not None and label in posterior:
                result[label] = float(np.median(posterior[label]))
            else:
                missing.append(label)
    if missing:
        raise KeyError(
            "Latent-alpha diagnostics are missing representative model "
            f"parameters: {missing}."
        )
    return result


def _evaluate_c3(
    model: Any,
    magnitude: Any,
    redshift: Any,
    f_host: Any,
    kwargs: Mapping[str, Any] | None,
) -> np.ndarray:
    raw_magnitude = np.asarray(magnitude, dtype=float)
    evaluated_magnitude = raw_magnitude
    inside_magnitude_support = None
    if hasattr(model, "mag_centers") and hasattr(model, "magnitude_support"):
        centers = np.asarray(model.mag_centers, dtype=float)
        support = np.asarray(model.magnitude_support, dtype=float)
        if centers.ndim != 1 or centers.size < 2 or support.shape != (2,):
            raise ValueError(
                "3D completeness exposes malformed magnitude grid/support metadata."
            )
        inside_magnitude_support = (
            np.isfinite(raw_magnitude)
            & (raw_magnitude >= support[0])
            & (raw_magnitude <= support[1])
        )
        evaluated_magnitude = np.clip(
            raw_magnitude, centers[0], centers[-1]
        )
    try:
        values = model(
            evaluated_magnitude,
            np.asarray(redshift, dtype=float),
            np.asarray(f_host, dtype=float),
            **dict(kwargs or {}),
        )
    except TypeError as exc:
        raise TypeError(
            "completeness_model must be callable as "
            "completeness_model(magnitude, redshift, f_host, **kwargs)."
        ) from exc
    result = np.asarray(values, dtype=float)
    expected = np.broadcast_shapes(
        np.shape(magnitude), np.shape(redshift), np.shape(f_host)
    )
    if result.shape != expected:
        try:
            result = np.broadcast_to(result, expected)
        except ValueError as exc:
            raise ValueError(
                f"3D completeness returned shape {result.shape}; expected a "
                f"shape broadcastable to {expected}."
            ) from exc
    if inside_magnitude_support is not None:
        result = np.where(
            np.broadcast_to(inside_magnitude_support, expected), result, 0.0
        )
    if not np.all(np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("3D completeness must return finite values in [0, 1].")
    return np.asarray(result, dtype=float)


def _safe_correlation(x: Any, y: Any) -> float | None:
    x_array, y_array = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    )
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    if np.count_nonzero(finite) < 2:
        return None
    x_use = x_array[finite]
    y_use = y_array[finite]
    if np.std(x_use) == 0.0 or np.std(y_use) == 0.0:
        return None
    return float(np.corrcoef(x_use, y_use)[0, 1])


def _linear_trend(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    correlation = _safe_correlation(x, y)
    centered = x - np.mean(x)
    denominator = float(np.sum(centered**2))
    slope = None if denominator == 0.0 else float(
        np.sum(centered * (y - np.mean(y))) / denominator
    )
    return {"slope": slope, "pearson_r": correlation}


def _quantile_summary(values: Any) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    quantiles = np.quantile(array, [0.0, 0.16, 0.5, 0.84, 1.0])
    return {
        "minimum": float(quantiles[0]),
        "p16": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p84": float(quantiles[3]),
        "maximum": float(quantiles[4]),
    }


def _binned_median(x: np.ndarray, y: np.ndarray, *, bins: int = 10):
    edges = np.linspace(float(np.min(x)), float(np.max(x)), bins + 1)
    centers: list[float] = []
    medians: list[float] = []
    low: list[float] = []
    high: list[float] = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        include = (x >= left) & (x < right)
        if right == edges[-1]:
            include |= x == right
        if np.count_nonzero(include) < 2:
            continue
        quantiles = np.quantile(y[include], [0.16, 0.5, 0.84])
        centers.append(float(np.median(x[include])))
        low.append(float(quantiles[0]))
        medians.append(float(quantiles[1]))
        high.append(float(quantiles[2]))
    return np.asarray(centers), np.asarray(medians), np.asarray(low), np.asarray(high)


def _json_safe(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return {str(key): _json_safe(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_json_safe(value) for value in payload]
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, np.ndarray):
        return [_json_safe(value) for value in payload.tolist()]
    if isinstance(payload, np.generic):
        return payload.item()
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_latent_alpha_diagnostics(
    agn_data: pd.DataFrame,
    *,
    config: LatentAlphaConfig,
    completeness_model: Any,
    output_dir: str | Path,
    parameters: Mapping[str, Any] | None = None,
    posterior_samples: Mapping[str, Any] | Any | None = None,
    model_labels: Sequence[str] | None = None,
    log_luminosity: Any | None = None,
    distance_modulus: Any | None = None,
    magnitude: Any | None = None,
    redshift_column: str = "z",
    completeness_kwargs: Mapping[str, Any] | None = None,
    cosmology_parameter_names: Sequence[str] | None = None,
    filename_prefix: str = "latent_alpha",
    plot_format: str = "pdf",
    alpha_grid_size: int = 241,
    saturation_threshold: float = 1.0e-4,
    inverse_weight_floor: float = 1.0e-6,
) -> LatentAlphaDiagnosticsResult:
    """Write the complete latent-alpha diagnostic bundle.

    Parameters
    ----------
    agn_data
        Fitted AGN frame containing ``z``, ``joint_posterior_valid_count``, and
        all eight aligned v3 ``*_draws`` columns.  Each draw column must be an
        ``(object, 64)`` numeric array or an object-valued Series of 64-vectors.
    config
        Authoritative LF-resolved latent-alpha configuration.
    completeness_model
        Host-aware base completeness, callable as ``C3(m, z, f_host)``.
    parameters / posterior_samples
        Supply a representative parameter mapping, posterior samples, or both.
        If only a posterior is supplied its coordinate-wise median is used.
        An array posterior requires ``model_labels``; a mapping posterior does
        not.  The posterior is otherwise used only for beta/cosmology
        correlations.
    log_luminosity
        Optional explicit object-level LF-state ``log10(nu L_nu(2500) /
        erg s^-1)`` for controlled tests.  Authoritative run diagnostics
        provide ``distance_modulus`` so luminosity is derived draw-by-draw
        from the aligned selected-state apparent magnitudes.

    Notes
    -----
    All 64 draws are used for the authoritative summaries.  The standard
    deterministic 16-of-64 subset is evaluated in parallel and its differences
    are recorded.  No derived-slope prior correction is applied.
    """

    if not isinstance(agn_data, pd.DataFrame):
        raise TypeError("agn_data must be a pandas DataFrame.")
    if agn_data.empty:
        raise ValueError("agn_data must contain at least one object.")
    if not isinstance(config, LatentAlphaConfig):
        raise TypeError("config must be a LatentAlphaConfig.")
    if not callable(completeness_model):
        raise TypeError("completeness_model must be callable.")
    if redshift_column not in agn_data:
        raise KeyError(f"Missing redshift column {redshift_column!r}.")
    if not filename_prefix or Path(filename_prefix).name != filename_prefix:
        raise ValueError("filename_prefix must be a nonempty plain filename stem.")
    plot_format = str(plot_format).strip().lower().lstrip(".")
    if plot_format not in {"pdf", "png"}:
        raise ValueError("plot_format must be 'pdf' or 'png'.")
    if int(alpha_grid_size) < 81:
        raise ValueError("alpha_grid_size must be at least 81.")
    saturation_threshold = float(saturation_threshold)
    if not 0.0 < saturation_threshold < 0.5:
        raise ValueError("saturation_threshold must lie strictly between 0 and 0.5.")
    inverse_weight_floor = float(inverse_weight_floor)
    if not 0.0 < inverse_weight_floor < 1.0:
        raise ValueError("inverse_weight_floor must lie strictly between 0 and 1.")

    n_objects = len(agn_data)
    redshift = _finite_vector(
        agn_data[redshift_column], length=n_objects, name=redshift_column
    )
    if np.any(redshift <= 0.0):
        raise ValueError("Latent-alpha diagnostics require positive redshifts.")
    if "joint_posterior_valid_count" not in agn_data:
        raise KeyError("Missing required column 'joint_posterior_valid_count'.")
    valid_count = np.asarray(agn_data["joint_posterior_valid_count"], dtype=int)
    if valid_count.shape != (n_objects,) or np.any(
        valid_count < JOINT_DRAW_INPUT_COUNT
    ):
        raise ValueError(
            "Latent-alpha diagnostics require 64 valid aligned joint posterior "
            "draws for every object."
        )

    draws = {column: _draw_matrix(agn_data, column) for column in _ALIGNED_DRAW_COLUMNS}
    host_draws = draws["f_host_2500_psf_draws"]
    if np.any((host_draws < 0.0) | (host_draws > 1.0)):
        raise ValueError("f_host_2500_psf_draws must lie in [0, 1].")
    alpha_intrinsic = draws["alpha_nu_intrinsic_1450_2500_draws"]
    alpha_attenuated = draws["alpha_nu_attenuated_1450_2500_draws"]
    if np.any(alpha_attenuated > alpha_intrinsic + 1.0e-8):
        raise ValueError(
            "Attenuated alpha_nu draws cannot be bluer than aligned intrinsic draws."
        )
    attenuation_sum = (
        draws["a_2500_galaxy_draws"] + draws["a_2500_internal_draws"]
    )
    if not np.allclose(
        draws["a_2500_total_draws"], attenuation_sum, rtol=1.0e-5, atol=1.0e-6
    ):
        raise ValueError(
            "Aligned A_2500_total draws do not equal galaxy + internal attenuation."
        )

    state_is_attenuated = config.luminosity_state == "attenuated"
    # The selected-data latent variable is always the intrinsic disk-only
    # secant.  LF semantics control only the magnitude/log-luminosity state.
    alpha_state_column = "alpha_nu_intrinsic_1450_2500_draws"
    magnitude_state_column = (
        "m_2500_attenuated_model_draws"
        if state_is_attenuated
        else "m_2500_dereddened_draws"
    )
    alpha_state = draws[alpha_state_column]
    magnitude_state_draws = draws[magnitude_state_column]
    if magnitude is None:
        magnitude_values = np.median(magnitude_state_draws, axis=1)
    else:
        magnitude_values = _finite_vector(
            magnitude, length=n_objects, name="magnitude"
        )

    if distance_modulus is not None:
        distance_modulus_values = _finite_vector(
            distance_modulus, length=n_objects, name="distance_modulus"
        )
        log_luminosity_draws = absolute_m2500_to_log_nu_lnu(
            magnitude_state_draws - distance_modulus_values[:, None]
        )
        log_luminosity_values = np.median(log_luminosity_draws, axis=1)
        luminosity_draw_semantics = "aligned_m2500_draws_minus_distance_modulus"
    else:
        if log_luminosity is None:
            raise ValueError(
                "Provide log_luminosity explicitly for a controlled scalar "
                "diagnostic calculation, or distance_modulus for authoritative "
                "aligned-draw diagnostics."
            )
        log_luminosity_values = _finite_vector(
            log_luminosity, length=n_objects, name="log_luminosity"
        )
        log_luminosity_draws = np.broadcast_to(
            log_luminosity_values[:, None], magnitude_state_draws.shape
        )
        luminosity_draw_semantics = "object_scalar_broadcast_for_diagnostics"

    posterior = _clean_posterior(_posterior_mapping(posterior_samples, model_labels))
    representative = _representative_parameters(config, parameters, posterior)
    coefficient_labels = response_coefficient_names(
        config.include_magnitude_interactions
    )
    coefficients = {label: representative[label] for label in coefficient_labels}
    parent_means = parent_alpha_mean_from_config(
        log_luminosity_values, config, parameters=representative
    )
    parent_means_draws = parent_alpha_mean_from_config(
        log_luminosity_draws, config, parameters=representative
    )

    selected_indices = deterministic_joint_draw_indices()
    z_draws = redshift[:, None]
    logl_draws = log_luminosity_draws
    magnitude_draws = magnitude_state_draws
    base_64 = _evaluate_c3(
        completeness_model,
        magnitude_draws,
        z_draws,
        host_draws,
        completeness_kwargs,
    )
    kappa_64 = calibrate_response_kappa(
        base_64,
        z_draws,
        logl_draws,
        coefficients,
        config=config,
        magnitude=(magnitude_draws if config.include_magnitude_interactions else None),
        parameters=representative,
    )
    offsets_64 = response_logit_offset(
        alpha_state,
        z_draws,
        coefficients,
        config=config,
        magnitude=(magnitude_draws if config.include_magnitude_interactions else None),
    )
    with np.errstate(invalid="ignore", over="ignore"):
        selected_completeness_64 = bounded_response_from_kappa(
            base_64, offsets_64, kappa_64
        )
    if not np.all(np.isfinite(selected_completeness_64)):
        raise RuntimeError("Latent-alpha response produced nonfinite completeness.")

    nodes, node_weights = normal_gauss_hermite_nodes(config.quadrature_order)
    alpha_nodes = parent_means_draws[:, :, None] + config.sigma * nodes
    node_offsets = response_logit_offset(
        alpha_nodes,
        redshift[:, None, None],
        coefficients,
        config=config,
        magnitude=(
            magnitude_draws[:, :, None]
            if config.include_magnitude_interactions
            else None
        ),
    )
    with np.errstate(invalid="ignore", over="ignore"):
        node_response_64 = bounded_response_from_kappa(
            base_64[:, :, None], node_offsets, kappa_64[:, :, None]
        )
    marginalized_64 = np.sum(node_response_64 * node_weights, axis=-1)
    normalization_residual = marginalized_64 - base_64
    selected_completeness_16 = selected_completeness_64[:, selected_indices]

    # Parent and predicted selected distributions, averaged across objects at
    # their median host fraction and selected-state magnitude.
    alpha_median_intrinsic = np.median(alpha_intrinsic, axis=1)
    alpha_median_attenuated = np.median(alpha_attenuated, axis=1)
    alpha_median_state = np.median(alpha_state, axis=1)
    alpha_min = min(
        float(np.quantile(alpha_state, 0.001)),
        float(np.min(parent_means) - 5.0 * config.sigma),
    )
    alpha_max = max(
        float(np.quantile(alpha_state, 0.999)),
        float(np.max(parent_means) + 5.0 * config.sigma),
    )
    if alpha_max <= alpha_min:
        alpha_min, alpha_max = config.mu - 5 * config.sigma, config.mu + 5 * config.sigma
    alpha_grid = np.linspace(alpha_min, alpha_max, int(alpha_grid_size))
    parent_density = parent_alpha_pdf(
        alpha_grid[None, :],
        log_luminosity_values[:, None],
        config.beta_l(representative),
        mu=config.mu,
        sigma=config.sigma,
        logl_pivot=config.logl_pivot,
    )
    host_median = np.median(host_draws, axis=1)
    base_object = _evaluate_c3(
        completeness_model,
        magnitude_values,
        redshift,
        host_median,
        completeness_kwargs,
    )
    response_grid = np.empty_like(parent_density)
    positive_base = base_object > np.finfo(float).eps
    if np.any(positive_base):
        grid_kappa = calibrate_response_kappa(
            base_object[positive_base],
            redshift[positive_base],
            log_luminosity_values[positive_base],
            coefficients,
            config=config,
            magnitude=(
                magnitude_values[positive_base]
                if config.include_magnitude_interactions
                else None
            ),
            parameters=representative,
        )
        grid_offsets = response_logit_offset(
            alpha_grid[None, :],
            redshift[positive_base, None],
            coefficients,
            config=config,
            magnitude=(
                magnitude_values[positive_base, None]
                if config.include_magnitude_interactions
                else None
            ),
        )
        response_grid[positive_base] = bounded_response_from_kappa(
            base_object[positive_base, None],
            grid_offsets,
            grid_kappa[:, None],
        )
    response_grid[~positive_base] = 0.0
    selected_density = np.full_like(parent_density, np.nan)
    selected_density[positive_base] = (
        parent_density[positive_base]
        * response_grid[positive_base]
        / base_object[positive_base, None]
    )
    selected_norm = np.trapezoid(
        selected_density[positive_base], alpha_grid, axis=1
    )
    if np.any(~np.isfinite(selected_norm)) or np.any(selected_norm <= 0.0):
        raise RuntimeError(
            "Could not normalize the predicted selected alpha distribution."
        )
    selected_density[positive_base] /= selected_norm[:, None]
    mean_parent_density = np.mean(parent_density, axis=0)
    mean_selected_density = (
        np.mean(selected_density[positive_base], axis=0)
        if np.any(positive_base)
        else np.full(alpha_grid.shape, np.nan)
    )
    selected_mean_by_object = np.full(n_objects, np.nan)
    selected_mean_by_object[positive_base] = np.trapezoid(
        selected_density[positive_base] * alpha_grid[None, :],
        alpha_grid,
        axis=1,
    )

    node_response = node_response_64[:, selected_indices]
    lower_saturation = float(
        np.mean(
            np.sum(
                (node_response <= saturation_threshold) * node_weights,
                axis=-1,
            )
        )
    )
    upper_saturation = float(
        np.mean(
            np.sum(
                (node_response >= 1.0 - saturation_threshold) * node_weights,
                axis=-1,
            )
        )
    )

    object_detection_64 = np.mean(selected_completeness_64, axis=1)
    object_detection_16 = np.mean(selected_completeness_16, axis=1)
    inverse_weights = 1.0 / np.maximum(object_detection_64, inverse_weight_floor)
    inverse_weights_16 = 1.0 / np.maximum(
        object_detection_16, inverse_weight_floor
    )
    ess = float(np.sum(inverse_weights) ** 2 / np.sum(inverse_weights**2))
    base_inverse_weights = 1.0 / np.maximum(
        np.mean(base_64, axis=1), inverse_weight_floor
    )
    base_ess = float(
        np.sum(base_inverse_weights) ** 2 / np.sum(base_inverse_weights**2)
    )

    alpha_redshift_summary = {
        "intrinsic": {
            **_quantile_summary(alpha_intrinsic),
            **_linear_trend(redshift, alpha_median_intrinsic),
        },
        "attenuated": {
            **_quantile_summary(alpha_attenuated),
            **_linear_trend(redshift, alpha_median_attenuated),
        },
        "latent_slope_state": "intrinsic_disk_only",
        "lf_luminosity_state": config.luminosity_state,
        "attenuation_shift": _quantile_summary(alpha_attenuated - alpha_intrinsic),
    }
    selected_valid = selected_mean_by_object[positive_base]
    parent_vs_selected = {
        "objects_with_positive_base_completeness": int(np.count_nonzero(positive_base)),
        "parent_mean": float(np.mean(parent_means)),
        "parent_population_sigma": float(config.sigma),
        "predicted_selected_mean": (
            float(np.mean(selected_valid)) if selected_valid.size else None
        ),
        "observed_compact_draw_mean": float(np.mean(alpha_state)),
        "observed_object_median_mean": float(np.mean(alpha_median_state)),
        "derived_slope_prior_correction": "none",
    }
    luminosity_summary = {
        "log_luminosity": _quantile_summary(log_luminosity_values),
        "beta_alpha_L": float(config.beta_l(representative)),
        "positive_beta_meaning": "more_luminous_is_bluer",
        "parent_mean_trend": _linear_trend(log_luminosity_values, parent_means),
        "observed_slope_trend": _linear_trend(
            log_luminosity_values, alpha_median_state
        ),
    }
    normalization_summary = {
        "maximum_absolute_residual": float(np.max(np.abs(normalization_residual))),
        "rms_residual": float(np.sqrt(np.mean(normalization_residual**2))),
        "mean_residual": float(np.mean(normalization_residual)),
        "saturation_threshold": saturation_threshold,
        "lower_saturation_fraction": lower_saturation,
        "upper_saturation_fraction": upper_saturation,
        "response_minimum": float(np.min(selected_completeness_64)),
        "response_maximum": float(np.max(selected_completeness_64)),
    }
    ess_summary = {
        "n_objects": n_objects,
        "ess": ess,
        "ess_fraction": ess / n_objects,
        "base_c3_ess": base_ess,
        "base_c3_ess_fraction": base_ess / n_objects,
        "inverse_weight_floor": inverse_weight_floor,
        "objects_at_inverse_weight_floor": int(
            np.count_nonzero(object_detection_64 < inverse_weight_floor)
        ),
        "inverse_weight": _quantile_summary(inverse_weights),
        "mean_absolute_16_vs_64_weight_difference": float(
            np.mean(np.abs(inverse_weights_16 - inverse_weights))
        ),
        "maximum_absolute_16_vs_64_detection_difference": float(
            np.max(np.abs(object_detection_16 - object_detection_64))
        ),
    }

    beta_correlations: dict[str, Any] = {
        "available": False,
        "posterior_draw_count": 0 if posterior is None else len(next(iter(posterior.values()))),
        "correlations": {},
    }
    correlation_names = set(cosmology_parameter_names or _COSMOLOGY_PARAMETER_NAMES)
    if posterior is not None and BETA_ALPHA_L_PARAMETER in posterior:
        beta_draws = posterior[BETA_ALPHA_L_PARAMETER]
        correlations = {
            label: value
            for label in sorted(correlation_names.intersection(posterior))
            if (value := _safe_correlation(beta_draws, posterior[label])) is not None
        }
        beta_correlations.update(
            {
                "available": bool(correlations),
                "beta_summary": _quantile_summary(beta_draws),
                "correlations": correlations,
            }
        )

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []

    # Imports remain local so catalog/schema-only workflows do not initialize a
    # Matplotlib backend.
    import matplotlib.pyplot as plt

    def save_figure(figure, suffix: str) -> Path:
        path = output_path / f"{filename_prefix}_{suffix}.{plot_format}"
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        plot_paths.append(path)
        return path

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.scatter(redshift, alpha_median_intrinsic, s=8, alpha=0.2, label="intrinsic")
    axis.scatter(redshift, alpha_median_attenuated, s=8, alpha=0.2, label="attenuated")
    for values, color in ((alpha_median_intrinsic, "C0"), (alpha_median_attenuated, "C1")):
        centers, medians, low, high = _binned_median(redshift, values)
        axis.plot(centers, medians, color=color, lw=2.2)
        axis.fill_between(centers, low, high, color=color, alpha=0.15)
    axis.axhline(config.mu, color="black", ls="--", lw=1, label="parent pivot mean")
    axis.set(xlabel="Redshift", ylabel=r"$\alpha_\nu^{1450-2500}$")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    save_figure(figure, "alpha_vs_redshift")

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.plot(alpha_grid, mean_parent_density, lw=2.2, label="parent")
    if np.any(positive_base):
        axis.plot(alpha_grid, mean_selected_density, lw=2.2, label="predicted selected")
    axis.hist(
        alpha_state.ravel(), bins=45, density=True, histtype="step", lw=1.5,
        label="observed compact draws",
    )
    axis.set(xlabel=r"$\alpha_\nu^{1450-2500}$", ylabel="Probability density")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    save_figure(figure, "parent_vs_selected")

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.scatter(log_luminosity_values, alpha_median_state, s=10, alpha=0.25, label="observed")
    order = np.argsort(log_luminosity_values)
    axis.plot(
        log_luminosity_values[order], parent_means[order], color="black", lw=2.2,
        label="parent mean",
    )
    if np.any(positive_base):
        selected_order = order[np.isfinite(selected_mean_by_object[order])]
        axis.plot(
            log_luminosity_values[selected_order], selected_mean_by_object[selected_order],
            color="C3", lw=1.5, alpha=0.8, label="predicted selected mean",
        )
    axis.set(
        xlabel=r"$\log_{10}[\nu L_\nu(2500\,\mathrm{\AA})/\mathrm{erg\,s^{-1}}]$",
        ylabel=r"$\alpha_\nu^{1450-2500}$",
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    save_figure(figure, "luminosity_dependence")

    slice_magnitudes = np.quantile(magnitude_values, [0.15, 0.5, 0.85])
    slice_redshifts = np.quantile(redshift, [0.15, 0.5, 0.85])
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 4.3), sharey=True)
    slice_records: list[dict[str, float]] = []
    for axis, slice_magnitude in zip(axes, slice_magnitudes, strict=True):
        for slice_redshift in slice_redshifts:
            slice_base = float(
                _evaluate_c3(
                    completeness_model,
                    slice_magnitude,
                    slice_redshift,
                    float(np.median(host_median)),
                    completeness_kwargs,
                )
            )
            slice_kappa = calibrate_response_kappa(
                slice_base,
                slice_redshift,
                float(np.median(log_luminosity_values)),
                coefficients,
                config=config,
                magnitude=(
                    slice_magnitude if config.include_magnitude_interactions else None
                ),
                parameters=representative,
            )
            slice_offsets = response_logit_offset(
                alpha_grid,
                slice_redshift,
                coefficients,
                config=config,
                magnitude=(slice_magnitude if config.include_magnitude_interactions else None),
            )
            slice_response = bounded_response_from_kappa(
                slice_base, slice_offsets, slice_kappa
            )
            axis.plot(alpha_grid, slice_response, lw=1.8, label=f"z={slice_redshift:.2f}")
            slice_records.append(
                {
                    "magnitude": float(slice_magnitude),
                    "redshift": float(slice_redshift),
                    "f_host": float(np.median(host_median)),
                    "log_luminosity": float(np.median(log_luminosity_values)),
                    "base_completeness": slice_base,
                    "response_minimum": float(np.min(slice_response)),
                    "response_maximum": float(np.max(slice_response)),
                }
            )
        axis.set_title(f"m={slice_magnitude:.2f}")
        axis.set_xlabel(r"$\alpha_\nu^{1450-2500}$")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Completeness")
    axes[-1].legend(frameon=False)
    save_figure(figure, "completeness_slices")

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    axes[0].scatter(base_64.ravel(), normalization_residual.ravel(), s=5, alpha=0.2)
    axes[0].axhline(0.0, color="black", lw=1)
    axes[0].set(xlabel=r"Base $C_3$", ylabel=r"$\int C_\alpha p_0\,da-C_3$")
    axes[1].hist(selected_completeness_64.ravel(), bins=40, histtype="stepfilled", alpha=0.55)
    axes[1].axvline(saturation_threshold, color="C3", ls="--")
    axes[1].axvline(1.0 - saturation_threshold, color="C3", ls="--")
    axes[1].set(xlabel=r"$C_\alpha$ at observed draws", ylabel="Count")
    for axis in axes:
        axis.grid(alpha=0.2)
    save_figure(figure, "normalization_and_saturation")

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.scatter(redshift, inverse_weights, s=10, alpha=0.3)
    axis.axhline(1.0, color="black", lw=1)
    axis.set(
        xlabel="Redshift",
        ylabel=r"Inverse mean $C_\alpha$ weight",
        title=f"ESS = {ess:.1f}/{n_objects} ({ess / n_objects:.1%})",
    )
    axis.set_yscale("log")
    axis.grid(alpha=0.2)
    save_figure(figure, "inverse_weight_ess")

    if beta_correlations["available"]:
        labels = list(beta_correlations["correlations"])
        correlations = [beta_correlations["correlations"][label] for label in labels]
        figure, axis = plt.subplots(figsize=(max(5.0, 1.2 * len(labels)), 4.2))
        axis.bar(labels, correlations, color="C4", alpha=0.8)
        axis.axhline(0.0, color="black", lw=1)
        axis.set(ylabel=r"Pearson $r(\beta_{\alpha L},\theta)$", ylim=(-1.0, 1.0))
        axis.grid(axis="y", alpha=0.2)
        save_figure(figure, "beta_cosmology_correlations")

    summary: dict[str, Any] = {
        "schema_version": LATENT_ALPHA_DIAGNOSTICS_SCHEMA_VERSION,
        "model": latent_alpha_provenance(config, coefficients),
        "n_objects": n_objects,
        "inputs": {
            "redshift_column": redshift_column,
            "luminosity_state": config.luminosity_state,
            "latent_slope_state": "intrinsic_disk_only",
            "lf_luminosity_state": config.luminosity_state,
            "alpha_state_column": alpha_state_column,
            "magnitude_state_column": magnitude_state_column,
            "joint_draw_input_count": JOINT_DRAW_INPUT_COUNT,
            "joint_draw_likelihood_count": len(selected_indices),
            "joint_draw_likelihood_indices": selected_indices.tolist(),
            "all_64_used_for_diagnostics": True,
            "joint_magnitude_host_slope_covariance_used": (
                luminosity_draw_semantics
                == "aligned_m2500_draws_minus_distance_modulus"
            ),
            "luminosity_draw_semantics": luminosity_draw_semantics,
            "derived_slope_prior_correction": "none",
            "magnitude": _quantile_summary(magnitude_values),
            "base_completeness": _quantile_summary(base_64),
        },
        "representative_parameters": {
            label: float(representative[label])
            for label in [*coefficient_labels, *config.joint_parameter_names()]
        },
        "alpha_vs_redshift": alpha_redshift_summary,
        "parent_vs_selected": parent_vs_selected,
        "luminosity_dependence": luminosity_summary,
        "completeness_slices": slice_records,
        "normalization_and_saturation": normalization_summary,
        "inverse_weight_ess": ess_summary,
        "beta_cosmology_correlations": beta_correlations,
        "plot_files": [path.name for path in plot_paths],
    }
    json_path = output_path / f"{filename_prefix}_diagnostics.json"
    _write_json_atomic(json_path, summary)
    return LatentAlphaDiagnosticsResult(
        json_path=json_path,
        plot_paths=tuple(plot_paths),
        summary=summary,
    )


__all__ = [
    "LATENT_ALPHA_DIAGNOSTICS_SCHEMA_VERSION",
    "LatentAlphaDiagnosticsResult",
    "write_latent_alpha_diagnostics",
]
