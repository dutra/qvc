"""AGN-fraction priors and likelihood marginalization for light-curve fits."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp


VARIABLE_RELFLUX_AMPLITUDE_KEYS = (
    "amp_cont_relflux",
    "amp_bc_relflux",
    "amp_blr_relflux",
    "amp_blr2_relflux",
)


def empirical_logmeanexp(component_log_likelihoods):
    """Return the log likelihood of an equal-weight finite mixture."""

    values = jnp.asarray(component_log_likelihoods)
    if values.ndim != 1 or values.shape[0] == 0:
        raise ValueError("component_log_likelihoods must be a non-empty 1D array.")
    return logsumexp(values) - jnp.log(values.shape[0])


def scale_variable_relflux_amplitudes(
    params: Mapping[str, object], fractions
) -> dict[str, object]:
    """Dilute stochastic AGN amplitudes while preserving observed-unit terms.

    Linear trends are intentionally left unchanged here. They require a
    separate modeling decision about whether the fitted trend is intrinsic AGN
    variability or an observed-frame calibration term.
    """

    fractions = jnp.asarray(fractions)
    scaled = dict(params)
    for key in VARIABLE_RELFLUX_AMPLITUDE_KEYS:
        if key in scaled:
            scaled[key] = jnp.asarray(scaled[key]) * fractions
    return scaled


def fit_logit_normal(
    fraction_draws,
    *,
    clip: float = 1e-6,
    covariance_eigenvalue_floor: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a stable multivariate normal to correlated fractions in logit space."""

    draws = np.asarray(fraction_draws, dtype=float)
    if draws.ndim != 2 or draws.shape[0] == 0 or draws.shape[1] == 0:
        raise ValueError(
            "fraction_draws must have shape (draw, band) with at least one draw."
        )
    if not np.all(np.isfinite(draws)):
        raise ValueError("fraction_draws must be finite.")
    if np.any((draws <= 0.0) | (draws >= 1.0)):
        draws = np.clip(draws, clip, 1.0 - clip)
    logits = np.log(draws) - np.log1p(-draws)
    mean = np.mean(logits, axis=0)
    if len(logits) == 1:
        covariance = np.zeros((draws.shape[1], draws.shape[1]), dtype=float)
    else:
        covariance = np.atleast_2d(np.cov(logits, rowvar=False, ddof=1))

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    floor = max(float(covariance_eigenvalue_floor), np.finfo(float).eps * scale)
    regularized = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    regularized = 0.5 * (regularized + regularized.T)
    scale_tril = np.linalg.cholesky(regularized)
    return mean, covariance, scale_tril


def sigmoid_fraction_draws(standard_normal_draws, mean, scale_tril):
    """Transform standard-normal draws through a fitted correlated logit normal."""

    normal = np.asarray(standard_normal_draws, dtype=float)
    mean = np.asarray(mean, dtype=float)
    scale_tril = np.asarray(scale_tril, dtype=float)
    logits = mean + normal @ scale_tril.T
    positive = logits >= 0
    fractions = np.empty_like(logits)
    fractions[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    fractions[~positive] = exp_logits / (1.0 + exp_logits)
    return fractions


def select_fraction_draws_for_bands(source, bands):
    """Select valid correlated draws in the retained light-curve band order."""

    stored_bands = tuple(str(band) for band in source["psf_agn_fraction_bands"])
    draws = np.asarray(source["psf_agn_fraction_draws"], dtype=float)
    valid_count = int(source.get("psf_agn_fraction_valid_count", len(draws)))
    draws = draws[:valid_count]
    try:
        indices = [stored_bands.index(str(band)) for band in bands]
    except ValueError as exc:
        raise ValueError(
            f"Retained LC bands {tuple(bands)!r} are not covered by spectral bands "
            f"{stored_bands!r}."
        ) from exc
    selected = draws[:, indices]
    jointly_valid = np.all(np.isfinite(selected), axis=1) & np.all(
        (selected > 0.0) & (selected <= 1.0), axis=1
    )
    selected = selected[jointly_valid]
    if len(selected) == 0:
        raise ValueError(
            "No valid joint PSF AGN-fraction draws remain for the retained LC bands."
        )
    return selected


def responsibility_resample_fractions(
    fraction_draws,
    responsibilities,
    *,
    seed: int = 0,
):
    """Draw one joint fraction vector per light-curve posterior sample."""

    draws = np.asarray(fraction_draws, dtype=float)
    weights = np.asarray(responsibilities, dtype=float)
    if weights.ndim != 2 or weights.shape[1] != draws.shape[0]:
        raise ValueError(
            "responsibilities must have shape (posterior, fraction_draw)."
        )
    rng = np.random.default_rng(seed)
    indices = np.empty(weights.shape[0], dtype=int)
    for index, row in enumerate(weights):
        total = np.sum(row)
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("Each responsibility row must have positive finite mass.")
        indices[index] = rng.choice(draws.shape[0], p=row / total)
    return draws[indices]


def scale_prediction_samples_by_fraction(samples):
    """Apply posterior fraction vectors to stochastic prediction amplitudes."""

    out = dict(samples)
    if "psf_agn_fraction" not in out:
        return out
    fractions = np.asarray(out["psf_agn_fraction"])
    for key in (
        *VARIABLE_RELFLUX_AMPLITUDE_KEYS,
        "amp_cont",
        "amp_bc",
        "amp_blr",
        "amp_blr2",
    ):
        if key in out:
            out[key] = np.asarray(out[key]) * fractions
    return out
