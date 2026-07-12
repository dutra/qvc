#!/usr/bin/env python3
"""Single-object multiband light-curve fitter using the DHO+BC+BLR model."""

import os
import sys
import argparse
import logging
import time
import traceback
from functools import lru_cache

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import curve_fit, least_squares
from scipy.special import gammaln
from scipy.stats import kurtosis, median_abs_deviation, normaltest, skew
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

# ---------- CPU & threading hygiene ----------
num_cores = os.environ.get("NUM_CORES", os.cpu_count() - 2)
try:
    num_cores = int(num_cores)
except ValueError:
    print(f"Invalid NUM_CORES value '{num_cores}', ignoring.")
    num_cores = os.cpu_count() - 2

if __name__ == "__main__" and (os.environ.get("PYTHON_EXECUTION_CONTEXT") != "worker"):
    print(f"CPU Num Cores: {num_cores}")

os.environ["XLA_FLAGS"] = (
    f"--xla_force_host_platform_device_count={num_cores} "
    f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=1"
)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.pop("NUMEXPR_MAX_THREADS", None)
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

prefix = os.environ.get("PREFIX", "test")
suffix = os.environ.get("SUFFIX", "test")

# ---------- JAX/NumPyro ----------
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)
import jax.numpy as jnp
from jax import lax
from jax import random, device_get
from jax.tree_util import tree_map

import numpyro

numpyro.set_host_device_count(num_cores)
numpyro.enable_x64()
numpyro.enable_validation(False)
import numpyro.distributions as dist
from numpyro.diagnostics import print_summary as numpyro_print_summary
from numpyro.handlers import seed, trace
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_value
from numpyro.optim import Adam

try:
    from numpyro.contrib.nested_sampling import NestedSampler
except ImportError:
    NestedSampler = None

from qvc.light_curve.multiband_fit_plotting import *
from qvc.light_curve.multiband_fit_plotting import (
    apply_window_response_correction,
    subtract_gp_mean_for_psd,
)
from qvc.light_curve.multiband_fit_utils import *
from qvc.light_curve.multiband_generate_lc import *
from qvc.light_curve.multiband_model_dho_blr import (
    mag_residual_to_relative_flux,
    magerr_residual_to_relative_fluxerr,
    make_multiband_dho_blr_flux_linearized_model,
    make_multiband_dho_blr_fluxmix_fast_display_model,
    make_multiband_dho_blr_model,
    make_linear_mean_func,
    relative_flux_to_mag_residual,
)
from qvc.light_curve.multiband_model_dho_blr_erlang import (
    DEFAULT_ERLANG_ORDER,
    make_multiband_dho_blr_flux_linearized_erlang_model,
)
from qvc.light_curve.psf_constant_flux_correction import (
    apply_constant_flux_correction_to_objects,
    print_constant_flux_correction_summary,
)
from qvc.light_curve.variability_metrics import compute_variability_metrics_for_cleaned_lc


zero_mean = False
has_jitter = True
BALMER_EDGE_REST_WAVELENGTH = 3646.0
BALMER_EDGE_ATTENUATION_WIDTH = 250.0
ETA_SIGMA_LOW = -5.0
LAG0_HIGH = 100.0
LAG_BETA_HIGH = 5.0
LOG_LAG_BLR_LOW = np.log(10.0)
LOG_LAG_BLR_HIGH = np.log(1e3)
LOG_LAG_RATIO_BC_TO_BLR_LOW = np.log(0.1)
LOG_LAG_RATIO_BC_TO_BLR_HIGH = np.log(0.3)
SF_MODEL_TAU_MIN_RF = 10.0
SF_MODEL_TAU_MAX_RF = 1e4
SF_MODEL_TAU_N_PLOT = 400
LOG_SF_INF_TO_RMS = 0.5 * np.log10(2.0)
RELFLUX_TO_MAG_SCALE = float(2.5 / np.log(10.0))
LOG_RELFLUX_TO_MAG_SCALE = float(np.log(RELFLUX_TO_MAG_SCALE))
LINEAR_TREND_RF_SIGMA_MAG_PER_DAY = 1e-4
FLUXMIX_CONT_SCALE_PRIOR_SIGMA = 1.0
FLUXMIX_BASIS_GRID_MIN_POINTS = 512
FLUXMIX_BASIS_GRID_MAX_POINTS = 2048
FLUXMIX_PREDICTION_FORWARD_PAD = 400.0
FLUXMIX_MIN_TOTAL_FLUX_RATIO = 0.05
FLUXMIX_TOTAL_FLUX_FLOOR_SOFTNESS = 0.01
FLUXMIX_TOTAL_FLUX_FLOOR_PENALTY = 20.0
FLUX_LINEARIZED_REFINEMENT_ITERS = 3
FLUX_LINEARIZED_MIN_TOTAL_FLUX_RATIO = 0.05
SDSS_FILTER_BLUE_EDGE_OBS = {
    "u": 3055.11,
    "g": 3797.64,
    "r": 5418.23,
    "i": 6692.41,
    "z": 7964.70,
    "y": 9469.38,
}
LC_SURVEY_NAMES = tuple(SURVEY_NAMES)
LC_SURVEY_TO_IDX = {name: idx for idx, name in enumerate(LC_SURVEY_NAMES)}


def _normalize_survey_name(value):
    if value is None:
        return "sdss"
    text = str(value).strip().lower()
    if text in {"", "nan", "none"}:
        return "sdss"
    if text == "s":
        return "sdss"
    if text == "p":
        return "ps1"
    if text == "z":
        return "ztf"
    if text in LC_SURVEY_TO_IDX:
        return text
    if text.startswith("panstarr") or text.startswith("pan-starr") or text == "panstarrs":
        return "ps1"
    raise ValueError(f"Unsupported survey label {value!r}. Expected one of {LC_SURVEY_NAMES}.")


def _default_survey_labels_for_band(band, size):
    if int(size) <= 0:
        return np.array([], dtype=str)
    if str(band) == "u":
        default = "sdss"
    else:
        default = "ztf"
    return np.full(int(size), default, dtype=f"<U{len(default)}")


def _survey_indices_from_labels(labels):
    labels = np.asarray(labels, dtype=str)
    if labels.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=str)
    normalized = np.asarray([_normalize_survey_name(value) for value in labels], dtype=str)
    survey_idx = np.asarray([LC_SURVEY_TO_IDX[value] for value in normalized], dtype=np.int32)
    return survey_idx, normalized


def _compute_log_jitter_mean_grid(yerr, band_idx, survey_idx, n_bands):
    yerr = np.asarray(yerr, dtype=float)
    band_idx = np.asarray(band_idx, dtype=np.int32)
    survey_idx = np.asarray(survey_idx, dtype=np.int32)
    n_surveys = len(LC_SURVEY_NAMES)
    log_jitter_mean = np.full((n_bands, n_surveys), np.log(1e-3), dtype=float)
    active_mask = np.zeros((n_bands, n_surveys), dtype=bool)
    fallback = np.log(1e-3)
    for i in range(int(n_bands)):
        band_mask = band_idx == i
        good_band = band_mask & np.isfinite(yerr) & (yerr > 0.0)
        if np.any(good_band):
            fallback = float(np.log(np.mean(yerr[good_band])))
        for j in range(n_surveys):
            mask = good_band & (survey_idx == j)
            active_mask[i, j] = bool(np.any(mask))
            log_jitter_mean[i, j] = (
                float(np.log(np.mean(yerr[mask])))
                if np.any(mask)
                else fallback
            )
    return jnp.asarray(log_jitter_mean, dtype=float), active_mask


def _compute_survey_offset_active_mask(band_idx, survey_idx, n_bands):
    band_idx = np.asarray(band_idx, dtype=np.int32)
    survey_idx = np.asarray(survey_idx, dtype=np.int32)
    n_surveys = len(LC_SURVEY_NAMES)
    active_mask = np.zeros((n_bands, n_surveys), dtype=bool)
    for i in range(int(n_bands)):
        band_surveys = survey_idx[band_idx == i]
        if band_surveys.size == 0:
            continue
        active_surveys = np.unique(band_surveys)
        ref_survey = int(np.min(active_surveys))
        active_mask[i, active_surveys] = True
        active_mask[i, ref_survey] = False
    return active_mask


def _get_object_active_noise_calibration_masks(obj_dict, n_bands):
    band_idx = np.asarray(
        obj_dict.get("band_idx", np.asarray(obj_dict["X"][1], dtype=np.int32)),
        dtype=np.int32,
    )
    survey_idx = np.asarray(
        obj_dict.get("survey_idx", np.zeros_like(band_idx, dtype=np.int32)),
        dtype=np.int32,
    )
    log_jitter_active_mask = obj_dict.get("log_jitter_active_mask")
    if log_jitter_active_mask is None:
        _, log_jitter_active_mask = _compute_log_jitter_mean_grid(
            np.asarray(obj_dict["yerr"], dtype=float),
            band_idx,
            survey_idx,
            n_bands,
        )
    survey_offset_active_mask = obj_dict.get("survey_offset_active_mask")
    if survey_offset_active_mask is None:
        survey_offset_active_mask = _compute_survey_offset_active_mask(
            band_idx,
            survey_idx,
            n_bands,
        )
    return (
        np.asarray(log_jitter_active_mask, dtype=bool),
        np.asarray(survey_offset_active_mask, dtype=bool),
    )


def _coerce_log_jitter_mean_grid(log_jitter_mean, n_bands):
    log_jitter_mean = np.asarray(log_jitter_mean, dtype=float)
    if log_jitter_mean.ndim == 2:
        return jnp.asarray(log_jitter_mean, dtype=float)
    if log_jitter_mean.ndim == 1 and log_jitter_mean.shape[0] == int(n_bands):
        return jnp.asarray(
            np.repeat(log_jitter_mean[:, None], len(LC_SURVEY_NAMES), axis=1),
            dtype=float,
        )
    raise ValueError(
        f"log_jitter_mean must have shape ({int(n_bands)},) or "
        f"({int(n_bands)}, {len(LC_SURVEY_NAMES)}); got {log_jitter_mean.shape}."
    )


def _sample_log_jitter_grid(log_jitter_mean, active_mask):
    log_jitter_mean = jnp.asarray(log_jitter_mean, dtype=float)
    active_mask = np.asarray(active_mask, dtype=bool)
    active_indices = np.argwhere(active_mask)
    log_jitter = jnp.array(log_jitter_mean)
    if active_indices.size:
        active_means = jnp.asarray(log_jitter_mean[active_mask], dtype=float)
        log_jitter_active = numpyro.sample(
            "log_jitter_active",
            log_jitter_prior(active_means),
        )
        for idx_flat, (band_idx, survey_idx) in enumerate(active_indices):
            log_jitter = log_jitter.at[int(band_idx), int(survey_idx)].set(log_jitter_active[idx_flat])
    return numpyro.deterministic("log_jitter", log_jitter)


def _sample_survey_delta_mag_grid(active_mask):
    active_mask = np.asarray(active_mask, dtype=bool)
    survey_delta_mag = jnp.zeros(active_mask.shape, dtype=float)
    active_indices = np.argwhere(active_mask)
    if active_indices.size:
        survey_delta_mag_active = numpyro.sample(
            "survey_delta_mag_active",
            survey_delta_mag_prior().expand((active_indices.shape[0],)),
        )
        for idx_flat, (band_idx, survey_idx) in enumerate(active_indices):
            survey_delta_mag = survey_delta_mag.at[int(band_idx), int(survey_idx)].set(
                survey_delta_mag_active[idx_flat]
            )
    return numpyro.deterministic("survey_delta_mag", survey_delta_mag)


def compute_lambda_center_rf(lam_rf):
    """Geometric-mean rest wavelength of the kept bands for one object."""

    lam_rf = jnp.asarray(lam_rf)
    lam_rf = jnp.maximum(lam_rf, jnp.array(1e-12, dtype=lam_rf.dtype))
    return jnp.exp(jnp.mean(jnp.log(lam_rf)))


def reference_flux_from_mean_magnitudes(mags_means):
    """Convert per-band mean magnitudes into reference fluxes."""

    return jnp.asarray(
        [10.0 ** (-0.4 * float(val)) for val in np.asarray(mags_means, dtype=float)],
        dtype=float,
    )


def infer_nested_sampler_u_ndims(model, rng_key, *args, **kwargs):
    """Mirror NumPyro nested-sampler latent-dimension counting."""

    prototype_trace = trace(seed(model, rng_key)).get_trace(*args, **kwargs)
    u_ndims = 0
    for site in prototype_trace.values():
        if (
            site["type"] == "sample"
            and not site["is_observed"]
            and site["infer"].get("enumerate", "") != "parallel"
        ):
            shape = tuple(int(dim) for dim in site["fn"].shape())
            u_ndims += int(np.prod(shape, dtype=int)) if shape else 1
    return u_ndims


def align_nested_sampler_num_live_points(model, rng_key, requested_num_live_points=None):
    """Return a num_live_points value compatible with the active JAX device count."""

    if requested_num_live_points is None:
        requested_num_live_points = infer_nested_sampler_u_ndims(model, rng_key) * 2

    requested_num_live_points = int(requested_num_live_points)
    device_count = max(1, len(jax.devices()))
    aligned_num_live_points = (
        (requested_num_live_points + device_count - 1) // device_count
    ) * device_count
    return aligned_num_live_points, requested_num_live_points, device_count


def run_svi_warm_start(model, rng_key, *, num_steps, learning_rate, progress_bar=False):
    """Run AutoNormal SVI and return guide-median init values for NUTS."""

    guide = AutoNormal(model)
    svi = SVI(model, guide, Adam(learning_rate), Trace_ELBO())
    svi_state = svi.init(rng_key)

    if progress_bar:
        final_loss = jnp.array(jnp.nan, dtype=jnp.float64)
        for _ in tqdm(range(int(num_steps)), desc="SVI warm-start", leave=False):
            svi_state, final_loss = svi.update(svi_state)
    else:
        def body_fn(_, carry):
            state, _ = carry
            state, loss = svi.update(state)
            return state, loss

        svi_state, final_loss = lax.fori_loop(
            0,
            int(num_steps),
            body_fn,
            (svi_state, jnp.array(jnp.nan, dtype=jnp.float64)),
        )
    params = svi.get_params(svi_state)
    init_values = guide.median(params)
    return init_values, float(device_get(final_loss))


def _expand_last(x):
    x = jnp.asarray(x)
    return jnp.expand_dims(x, axis=-1) if x.ndim > 0 else x


def _coerce_band_array(x, lam_rf):
    x = jnp.asarray(x, dtype=jnp.asarray(lam_rf).dtype)
    B = int(jnp.asarray(lam_rf).shape[0])
    if x.ndim == 0:
        return jnp.broadcast_to(x, (B,))
    if x.shape[-1] == B:
        return x
    return jnp.broadcast_to(x, x.shape[:-1] + (B,))


def compute_lam_lya_suppression_rf(bands, z):
    """Return the rest-frame band edge used for Lyman-related band diagnostics."""

    return jnp.asarray(
        [
            SDSS_FILTER_BLUE_EDGE_OBS.get(str(band), lambda_pivot[str(band)]) / (1.0 + float(z))
            for band in bands
        ],
        dtype=float,
    )


@lru_cache(maxsize=1)
def _load_sdss_lightcurve_filter_curves():
    """Load SDSS filter curves once for band-integrated IGM transmission."""

    from speclite import filters as speclite_filters

    filters = speclite_filters.load_filters(*[f"sdss2010-{b}" for b in ("u", "g", "r", "i")])
    out = {}
    for band, filt in zip(("u", "g", "r", "i"), filters):
        wave_obs = np.asarray(filt.wavelength, dtype=float)
        transmission = np.asarray(filt.response, dtype=float)
        good = np.isfinite(wave_obs) & np.isfinite(transmission) & (transmission > 0.0)
        if np.any(good):
            out[band] = (wave_obs[good], transmission[good])
    return out


def _build_igm_cache_np_nm(rest_wave_nm):
    """Build wavelength-only helper arrays for the copied grahspj IGM model."""

    wavelength = np.asarray(rest_wave_nm, dtype=float)
    n_transitions_low = 10
    n_transitions_max = 31
    lambda_limit = 91.2
    n_arr = np.arange(n_transitions_max, dtype=float)
    lambda_n = np.ones_like(n_arr, dtype=float)
    valid_n = n_arr >= 2
    lambda_n[valid_n] = lambda_limit / (1.0 - 1.0 / (n_arr[valid_n] * n_arr[valid_n]))
    z_n = wavelength[None, :] / lambda_n[:, None] - 1.0
    n_eval = np.arange(3, n_transitions_max, dtype=float)
    fact = np.array([1.0, 1.0, 1.0, 0.348, 0.179, 0.109, 0.0722, 0.0508, 0.0373, 0.0283], dtype=float)
    fact_eval = np.zeros_like(n_eval, dtype=float)
    fact_mask = n_eval <= 9.0
    fact_eval[fact_mask] = fact[n_eval[fact_mask].astype(np.int32)]
    val_gt9_coeff = 720.0 / (n_eval * (n_eval * n_eval * n_eval - 1.0))
    z_l = wavelength / lambda_limit - 1.0
    wl_ratio = wavelength / lambda_limit
    n = np.arange(n_transitions_low - 1, dtype=float)
    factorial_n = np.exp(gammaln(n + 1.0))
    term2 = np.sum(np.power(-1.0, n) / (factorial_n * (2.0 * n - 1.0)))
    ni = np.arange(1, n_transitions_low, dtype=float)
    factorial_ni = np.exp(gammaln(ni + 1.0))
    coeff = 2.0 * np.power(-1.0, ni) / (factorial_ni * ((6.0 * ni - 5.0) * (2.0 * ni - 1.0)))
    return {
        "wavelength": wavelength,
        "z_n2": z_n[2],
        "z_eval": z_n[3:],
        "z_n9": z_n[9],
        "z_l": z_l,
        "wl_ratio": wl_ratio,
        "fact": fact,
        "fact_eval": fact_eval,
        "n_eval": n_eval,
        "val_gt9_coeff": val_gt9_coeff,
        "term2": term2,
        "coeff": coeff,
    }


def _evaluate_igm_transmission_np(igm_cache, redshift):
    """Evaluate the copied grahspj IGM transmission on an observed-frame wavelength grid."""

    n_transitions_low = 10
    gamma = 0.2788
    n0 = 0.25
    z = float(redshift)
    wavelength = igm_cache["wavelength"]
    z_n2 = igm_cache["z_n2"]
    z_eval = igm_cache["z_eval"]
    z_n9 = igm_cache["z_n9"]
    z_l = igm_cache["z_l"]
    wl_ratio = igm_cache["wl_ratio"]
    fact = igm_cache["fact"]
    fact_eval = igm_cache["fact_eval"]
    n_eval = igm_cache["n_eval"]

    tau_a = 0.00211 * (1.0 + z) ** 3.7 if z <= 4.0 else 0.00058 * (1.0 + z) ** 4.5
    tau2 = np.where(
        z <= 4.0,
        0.00211 * (1.0 + z_n2) ** 3.7,
        0.00058 * (1.0 + z_n2) ** 4.5,
    )
    tau2 = np.where(z_n2 >= z, 0.0, tau2)
    val_le5 = np.where(
        z_eval < 3.0,
        tau_a * fact_eval[:, None] * (0.25 * (1.0 + z_eval)) ** (1.0 / 3.0),
        tau_a * fact_eval[:, None] * (0.25 * (1.0 + z_eval)) ** (1.0 / 6.0),
    )
    val_6_9 = tau_a * fact_eval[:, None] * (0.25 * (1.0 + z_eval)) ** (1.0 / 3.0)
    tau9 = tau_a * fact[9] * (0.25 * (1.0 + z_n9)) ** (1.0 / 3.0)
    val_gt9 = tau9[None, :] * igm_cache["val_gt9_coeff"][:, None]
    val_eval = np.where(
        n_eval[:, None] <= 5.0,
        val_le5,
        np.where(n_eval[:, None] <= 9.0, val_6_9, val_gt9),
    )
    tau_taun = tau2 + np.sum(np.where(z_eval >= z, 0.0, val_eval), axis=0)
    w = z_l < z
    tau_l_igm = np.where(
        w,
        0.805 * (1.0 + z_l) ** 3 * (1.0 / (1.0 + z_l) - 1.0 / (1.0 + z)),
        0.0,
    )
    term1 = gamma - np.exp(-1.0)
    term2 = igm_cache["term2"]
    term3 = (1.0 + z) * wl_ratio ** 1.5 - wl_ratio ** 2.5
    ni = np.arange(1, n_transitions_low, dtype=float)
    coeff = igm_cache["coeff"]
    term4_terms = coeff[:, None] * (
        (1.0 + z) ** (2.5 - (3.0 * ni[:, None])) * wl_ratio[None, :] ** (3.0 * ni[:, None])
        - wl_ratio[None, :] ** 2.5
    )
    term4 = np.sum(term4_terms, axis=0)
    tau_l_lls = np.where(w, n0 * ((term1 - term2) * term3 - term4), 0.0)
    lambda_min_igm = (1.0 + z) * 70.0
    weight = np.where(wavelength < lambda_min_igm, (wavelength / lambda_min_igm) ** 2, 1.0)
    transmission = np.exp(-tau_taun - tau_l_igm - tau_l_lls) * weight
    return np.clip(transmission, 1.0e-12, 1.0)


def compute_igm_transmission_obs_wave(obs_wave_angstrom, z):
    """Return copied grahspj IGM transmission on an observed-frame wavelength grid in Angstrom."""

    obs_wave_nm = np.asarray(obs_wave_angstrom, dtype=float) / 10.0
    igm_cache = _build_igm_cache_np_nm(obs_wave_nm)
    return _evaluate_igm_transmission_np(igm_cache, z)


def compute_igm_transmission_rest_wave(rest_wave_angstrom, z):
    """Return copied grahspj IGM transmission for rest-frame wavelengths in Angstrom."""

    obs_wave_angstrom = np.asarray(rest_wave_angstrom, dtype=float) * (1.0 + float(z))
    return compute_igm_transmission_obs_wave(obs_wave_angstrom, z)


def compute_band_igm_transmission(bands, z):
    """Return one effective IGM transmission factor per band, integrated under SDSS curves."""

    z = float(z)
    filters = _load_sdss_lightcurve_filter_curves()
    transmissions = []
    for band in bands:
        band = str(band)
        if band in filters:
            wave_obs, response = filters[band]
            igm = compute_igm_transmission_obs_wave(wave_obs, z)
            denom = np.maximum(np.trapezoid(response, wave_obs), 1.0e-30)
            eff = np.trapezoid(response * igm, wave_obs) / denom
        else:
            wave_obs = np.asarray(
                [SDSS_FILTER_BLUE_EDGE_OBS.get(band, lambda_pivot[band])],
                dtype=float,
            )
            eff = compute_igm_transmission_obs_wave(wave_obs, z)[0]
        transmissions.append(np.clip(eff, 1.0e-12, 1.0))
    return jnp.asarray(transmissions, dtype=float)


def compute_log_igm_transmission_band(bands, z):
    """Return per-band log IGM transmission for diagnostics and component bookkeeping."""

    return jnp.log(compute_band_igm_transmission(bands, z))


def balmer_continuum_weight(
    lam_rf,
    transition=BALMER_EDGE_REST_WAVELENGTH,
    width=BALMER_EDGE_ATTENUATION_WIDTH,
):
    """Smooth weight that approaches 1 blueward of the Balmer edge and 0 redward."""

    lam_rf = jnp.asarray(lam_rf)
    return 1.0 / (1.0 + jnp.exp((lam_rf - transition) / width))


def _dist_log_prob_array(distribution, x):
    return np.asarray(distribution.log_prob(jnp.asarray(x)))


def kl_from_samples(x, log_prior_fn):
    """Approximate KL(q||p) with q=N(sample mean, sample var) and user-provided log p."""

    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan

    mu = x.mean()
    var = x.var(ddof=1) + 1e-12
    log_q = -0.5 * np.log(2.0 * np.pi * var) - 0.5 * (x - mu) ** 2 / var
    log_p = np.asarray(log_prior_fn(x), dtype=float)
    good = np.isfinite(log_q) & np.isfinite(log_p)
    if not np.any(good):
        return np.nan
    return float(np.mean(log_q[good] - log_p[good]))


def conditional_kl_from_samples(x, build_prior_log_prob_fn, *conditioners):
    """Approximate KL(q||p) when the prior parameters depend on per-sample conditioners."""

    x = np.asarray(x).ravel()
    conds = [np.asarray(c).ravel() for c in conditioners]
    if any(c.shape != x.shape for c in conds):
        raise ValueError("Conditioner arrays must have the same shape as x.")

    good = np.isfinite(x)
    for c in conds:
        good &= np.isfinite(c)
    if np.sum(good) < 2:
        return np.nan

    x_good = x[good]
    conds_good = [c[good] for c in conds]
    var = x_good.var(ddof=1) + 1e-12
    mu = x_good.mean()
    log_q = -0.5 * np.log(2.0 * np.pi * var) - 0.5 * (x_good - mu) ** 2 / var
    log_p = np.asarray(build_prior_log_prob_fn(x_good, *conds_good), dtype=float)
    good_lp = np.isfinite(log_q) & np.isfinite(log_p)
    if not np.any(good_lp):
        return np.nan
    return float(np.mean(log_q[good_lp] - log_p[good_lp]))


def linear_mean_time_scaling(t_ref):
    """Return the global time centering/scaling used by the linear mean function."""

    t_ref = np.asarray(t_ref, dtype=float)
    if t_ref.size == 0:
        return 0.0, 1.0
    t_center = 0.5 * (np.min(t_ref) + np.max(t_ref))
    t_std = max(float(np.std(t_ref)), 1e-6)
    return float(t_center), t_std


def posterior_median_mean_function(flat_samples, t_eval, band, *, t_ref=None, survey_idx=None, survey_names=None):
    """Return the posterior-median fitted mean function for one band."""

    t_eval = np.asarray(t_eval, dtype=float)
    if t_ref is None:
        t_ref = t_eval
    t_ref = np.asarray(t_ref, dtype=float)
    mean_key = f"mean_{band}"
    mean_level = float(np.nanmedian(np.asarray(flat_samples[mean_key], dtype=float))) if mean_key in flat_samples else 0.0
    linear_trend = (
        float(np.nanmedian(np.asarray(flat_samples["linear_trend"], dtype=float)))
        if "linear_trend" in flat_samples
        else 0.0
    )
    band_linear_trend = linear_trend
    band_slope_key = f"linear_trend_band_{band}"
    band_offset_key = f"linear_trend_band_offset_{band}"
    if band_slope_key in flat_samples:
        band_linear_trend = float(np.nanmedian(np.asarray(flat_samples[band_slope_key], dtype=float)))
    elif band_offset_key in flat_samples:
        band_linear_trend = linear_trend + float(
            np.nanmedian(np.asarray(flat_samples[band_offset_key], dtype=float))
        )
    t_center, t_std = linear_mean_time_scaling(t_ref)
    time_scaled = (t_eval - t_center) / t_std
    mean_curve = mean_level + band_linear_trend * time_scaled
    survey_offset_mag = np.zeros_like(t_eval, dtype=float)
    if survey_idx is not None:
        survey_idx = np.asarray(survey_idx, dtype=np.int32)
        if survey_names is None:
            survey_names = LC_SURVEY_NAMES
        survey_names = tuple(str(name) for name in survey_names)
        if survey_idx.shape == t_eval.shape:
            for survey_i, survey_name in enumerate(survey_names):
                key = f"survey_delta_mag_{band}_{survey_name}"
                if key not in flat_samples:
                    continue
                mask = survey_idx == survey_i
                if not np.any(mask):
                    continue
                survey_offset_mag[mask] = float(np.nanmedian(np.asarray(flat_samples[key], dtype=float)))
    if "log_sigma_uv_relflux" in flat_samples or "amp_cont_relflux" in flat_samples:
        mean_curve = np.asarray(-2.5 * np.log10(np.clip(1.0 + mean_curve, 1e-12, None)), dtype=float)
        return mean_curve + survey_offset_mag
    return mean_curve + survey_offset_mag


def compute_band_adf(values, *, regression="c", autolag="AIC"):
    """Safely run the Augmented Dickey-Fuller test on one 1D series."""

    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    result = {
        "adf_stat": np.nan,
        "adf_pvalue": np.nan,
        "adf_usedlag": np.nan,
        "adf_nobs": float(values.size),
        "adf_valid": False,
    }
    if values.size < 4:
        return result
    if np.allclose(values, values[0], equal_nan=False):
        return result

    try:
        stat, pvalue, usedlag, nobs, *_ = adfuller(values, regression=regression, autolag=autolag)
    except Exception as exc:
        logging.warning("ADF failed for series of length %d: %s", values.size, exc)
        return result

    result.update(
        adf_stat=float(stat),
        adf_pvalue=float(pvalue),
        adf_usedlag=float(usedlag),
        adf_nobs=float(nobs),
        adf_valid=True,
    )
    return result


def compute_object_adf_diagnostics(flat_samples, obj, bands):
    """Compute per-band ADF diagnostics after subtracting the posterior mean function."""

    t_all = np.asarray(obj["X"][0], dtype=float)
    y_all = np.asarray(obj["y"], dtype=float)
    band_idx = np.asarray(obj["band_idx"])
    survey_idx_all = np.asarray(obj.get("survey_idx", np.zeros_like(band_idx, dtype=np.int32)), dtype=np.int32)
    survey_names = tuple(obj.get("survey_names", LC_SURVEY_NAMES))

    out = {}
    pvalues = []
    for i, band in enumerate(bands):
        mask = band_idx == i
        t_band = t_all[mask]
        y_band = y_all[mask]
        fitted_mean = posterior_median_mean_function(
            flat_samples,
            t_band,
            band,
            t_ref=t_all,
            survey_idx=survey_idx_all[mask],
            survey_names=survey_names,
        )
        detrended = y_band - fitted_mean
        adf = compute_band_adf(detrended)
        out[f"adf_stat_{band}"] = adf["adf_stat"]
        out[f"adf_pvalue_{band}"] = adf["adf_pvalue"]
        out[f"adf_usedlag_{band}"] = adf["adf_usedlag"]
        out[f"adf_nobs_{band}"] = adf["adf_nobs"]
        out[f"adf_valid_{band}"] = adf["adf_valid"]
        if np.isfinite(adf["adf_pvalue"]):
            pvalues.append(adf["adf_pvalue"])

    out["adf_min_pvalue"] = float(np.min(pvalues)) if pvalues else np.nan
    out["adf_any_pvalue_lt_0p05"] = bool(np.any(np.asarray(pvalues) < 0.05)) if pvalues else False
    return out


def compute_multiband_residual_normality_diagnostics(
    flat_samples,
    obj,
    bands,
    *,
    z=None,
    return_series=True,
):
    """Summarize per-band normality diagnostics for detrended residuals."""

    out = {}
    pvalues = []
    for band in bands:
        try:
            _, _, residual, yerr = extract_band_detrended_series(
                flat_samples,
                obj,
                bands,
                band,
                z=z,
                subtract_mean=True,
            )
        except KeyError:
            continue

        residual = np.asarray(residual, dtype=float).ravel()
        yerr = np.asarray(yerr, dtype=float).ravel()
        mask = np.isfinite(residual)
        if yerr.shape == residual.shape:
            mask &= np.isfinite(yerr) | ~np.isfinite(yerr)
        residual = residual[mask]
        yerr = yerr[mask]

        stats = {
            f"resid_normality_nobs_{band}": float(residual.size),
            f"resid_normality_mean_{band}": np.nan,
            f"resid_normality_std_{band}": np.nan,
            f"resid_normality_skew_{band}": np.nan,
            f"resid_normality_kurtosis_{band}": np.nan,
            f"resid_normality_k2_{band}": np.nan,
            f"resid_normality_pvalue_{band}": np.nan,
            f"resid_normality_valid_{band}": False,
        }
        if residual.size == 0:
            out.update(stats)
            if return_series:
                out[f"resid_normality_residual_{band}"] = residual
                out[f"resid_normality_zscore_{band}"] = np.array([], dtype=float)
                out[f"resid_normality_yerr_{band}"] = yerr
            continue

        mean = float(np.mean(residual))
        std = float(np.std(residual, ddof=1)) if residual.size > 1 else np.nan
        zscore = (
            (residual - mean) / std
            if np.isfinite(std) and std > 0.0
            else np.full_like(residual, np.nan, dtype=float)
        )
        skewness = float(skew(residual, bias=False)) if residual.size > 2 else np.nan
        excess_kurt = float(kurtosis(residual, fisher=True, bias=False)) if residual.size > 3 else np.nan
        k2_stat = np.nan
        p_norm = np.nan
        if residual.size >= 8 and np.isfinite(std) and std > 0.0:
            try:
                k2_stat, p_norm = normaltest(np.asarray(residual, dtype=np.float64))
                k2_stat = float(k2_stat)
                p_norm = float(p_norm)
            except Exception as exc:
                logging.warning(
                    "Normality test failed for %s band residuals of %s: %s",
                    band,
                    obj.get("object_id", "<unknown>"),
                    exc,
                )

        stats.update(
            {
                f"resid_normality_mean_{band}": mean,
                f"resid_normality_std_{band}": std,
                f"resid_normality_skew_{band}": skewness,
                f"resid_normality_kurtosis_{band}": excess_kurt,
                f"resid_normality_k2_{band}": k2_stat,
                f"resid_normality_pvalue_{band}": p_norm,
                f"resid_normality_valid_{band}": bool(np.isfinite(std) and std > 0.0),
            }
        )
        out.update(stats)
        if np.isfinite(p_norm):
            pvalues.append(p_norm)
        if return_series:
            out[f"resid_normality_residual_{band}"] = residual
            out[f"resid_normality_zscore_{band}"] = zscore
            out[f"resid_normality_yerr_{band}"] = yerr

    out["resid_normality_min_pvalue"] = float(np.min(pvalues)) if pvalues else np.nan
    out["resid_normality_any_pvalue_lt_0p05"] = bool(np.any(np.asarray(pvalues) < 0.05)) if pvalues else False
    return out


LOO_RESIDUAL_RF_BIN_EDGES = (0.0, 10.0, 30.0, 100.0, 300.0)


def _format_rest_frame_bin_edge(value):
    value = float(value)
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


def binned_loo_residual_pair_correlation(
    times_rf,
    band_idx,
    standardized_residuals,
    *,
    bin_edges=LOO_RESIDUAL_RF_BIN_EDGES,
    bands=None,
):
    """Summarize within-band LOO residual products in rest-frame lag bins."""

    times_rf = np.asarray(times_rf, dtype=float).ravel()
    band_idx = np.asarray(band_idx, dtype=np.int32).ravel()
    residuals = np.asarray(standardized_residuals, dtype=float).ravel()
    if not (times_rf.shape == band_idx.shape == residuals.shape):
        raise ValueError("times_rf, band_idx, and standardized_residuals must match")
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("bin_edges must be a strictly increasing 1D sequence")

    finite = np.isfinite(times_rf) & np.isfinite(residuals)
    times_rf = times_rf[finite]
    band_idx = band_idx[finite]
    residuals = residuals[finite]
    if bands is None:
        band_values = np.unique(band_idx)
        band_names = [str(int(value)) for value in band_values]
    else:
        band_names = [str(band) for band in bands]
        band_values = np.arange(len(band_names), dtype=np.int32)

    out = {}
    for edge_index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        suffix = (
            f"{_format_rest_frame_bin_edge(low)}_"
            f"{_format_rest_frame_bin_edge(high)}d"
        )
        all_products = []
        products_by_band = {}
        for band_value, band_name in zip(band_values, band_names):
            mask = band_idx == band_value
            t_band = times_rf[mask]
            r_band = residuals[mask]
            if t_band.size < 2:
                products = np.array([], dtype=float)
            else:
                dt = np.abs(t_band[:, None] - t_band[None, :])
                products_matrix = r_band[:, None] * r_band[None, :]
                upper = np.triu(np.ones(dt.shape, dtype=bool), k=1)
                in_bin = upper & (dt >= low) & (
                    dt <= high if edge_index == len(edges) - 2 else dt < high
                )
                products = np.asarray(products_matrix[in_bin], dtype=float)
                products = products[np.isfinite(products)]
            products_by_band[band_name] = products
            if products.size:
                all_products.append(products)

        combined = (
            np.concatenate(all_products) if all_products else np.array([], dtype=float)
        )
        for label, products in [(None, combined), *products_by_band.items()]:
            key_suffix = suffix if label is None else f"{suffix}_{label}"
            count = int(products.size)
            mean = float(np.mean(products)) if count else np.nan
            stderr = (
                float(np.std(products, ddof=1) / np.sqrt(count))
                if count > 1
                else np.nan
            )
            zscore = mean / stderr if np.isfinite(stderr) and stderr > 0.0 else np.nan
            out[f"loo_resid_pair_count_rf_{key_suffix}"] = count
            out[f"loo_resid_corr_rf_{key_suffix}"] = mean
            out[f"loo_resid_corr_stderr_rf_{key_suffix}"] = stderr
            out[f"loo_resid_corr_z_rf_{key_suffix}"] = zscore
    return out


def compute_loo_short_lag_residual_diagnostics(
    model,
    samples,
    obj,
    bands,
    *,
    bin_edges=LOO_RESIDUAL_RF_BIN_EDGES,
):
    """Compute exact Gaussian LOO standardized-residual correlations."""

    oid = obj.get("object_id", "<unknown>")
    try:
        params = tree_map(jnp.asarray, _posterior_median_params(samples))
        gp, inds = model._build_gp(params)
        y_sorted = np.asarray(model._observed_y_sorted(params, inds), dtype=float)
        mean_sorted = np.asarray(gp.loc, dtype=float)
        covariance = np.asarray(gp.covariance, dtype=float)
        covariance = 0.5 * (covariance + covariance.T)
        scale = max(float(np.nanmedian(np.diag(covariance))), 1.0)
        covariance = covariance + np.eye(covariance.shape[0]) * (1e-10 * scale)
        chol = np.linalg.cholesky(covariance)
        centered = y_sorted - mean_sorted
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, centered))
        precision = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(chol.shape[0])))
        precision_diag = np.diag(precision)
        loo_standardized = alpha / np.sqrt(np.maximum(precision_diag, 1e-300))
        finite_loo = np.isfinite(loo_standardized)
        loo_standardized_finite = loo_standardized[finite_loo]
        if not loo_standardized_finite.size:
            raise ValueError("No finite LOO standardized residuals.")
        loo_chi2_eff = float(np.mean(np.square(loo_standardized_finite)))
        times_sorted = np.asarray(gp.X[0], dtype=float)
        bands_sorted = np.asarray(gp.X[1], dtype=np.int32)
        times_rf = times_sorted / (1.0 + float(obj.get("z", 0.0)))
        result = binned_loo_residual_pair_correlation(
            times_rf,
            bands_sorted,
            loo_standardized,
            bin_edges=bin_edges,
            bands=bands,
        )
        result["loo_resid_valid"] = True
        result["loo_resid_nobs"] = int(loo_standardized_finite.size)
        result["loo_chi2_eff"] = loo_chi2_eff
        result["loo_rms"] = float(np.sqrt(loo_chi2_eff)) if np.isfinite(loo_chi2_eff) else np.nan
    except Exception as exc:
        logging.warning("[%s] LOO residual diagnostic failed: %s", oid, exc)
        result = {
            "loo_resid_valid": False,
            "loo_resid_nobs": 0,
            "loo_chi2_eff": np.nan,
            "loo_rms": np.nan,
            "loo_resid_error": str(exc),
        }

    if not result["loo_resid_valid"]:
        print(
            f"[{oid}] LOO standardized-residual diagnostic unavailable: "
            f"{result.get('loo_resid_error', 'unknown error')}"
        )
        return result

    print(
        f"[{oid}] LOO standardized-residual summary: "
        f"chi2_eff={result['loo_chi2_eff']:.4f}, "
        f"rms={result['loo_rms']:.4f}, "
        f"nobs={result['loo_resid_nobs']}"
    )
    print(f"[{oid}] LOO standardized-residual correlation (rest-frame bins):")
    edges = np.asarray(bin_edges, dtype=float)
    for low, high in zip(edges[:-1], edges[1:]):
        suffix = (
            f"{_format_rest_frame_bin_edge(low)}_"
            f"{_format_rest_frame_bin_edge(high)}d"
        )
        print(
            f"  {low:g}–{high:g} d: "
            f"corr={result.get(f'loo_resid_corr_rf_{suffix}', np.nan):+.4f}, "
            f"SE={result.get(f'loo_resid_corr_stderr_rf_{suffix}', np.nan):.4f}, "
            f"z={result.get(f'loo_resid_corr_z_rf_{suffix}', np.nan):+.2f}, "
            f"pairs={result.get(f'loo_resid_pair_count_rf_{suffix}', 0)}"
        )
    return result


def extract_band_detrended_series(flat_samples, obj, bands, band, *, z=None, subtract_mean=True):
    """Return observed/rest-frame times, values, and errors for one band."""

    if band not in bands:
        raise KeyError(f"Missing {band} band required for residual diagnostics.")

    t_all = np.asarray(obj["X"][0], dtype=float)
    y_all = np.asarray(obj["y"], dtype=float)
    yerr_all = np.asarray(obj.get("yerr", np.full_like(y_all, np.nan)), dtype=float)
    band_idx = np.asarray(obj["band_idx"])
    survey_idx_all = np.asarray(obj.get("survey_idx", np.zeros_like(band_idx, dtype=np.int32)), dtype=np.int32)
    survey_names = tuple(obj.get("survey_names", LC_SURVEY_NAMES))

    i = bands.index(band)
    mask = band_idx == i
    t_band = t_all[mask]
    y_band = y_all[mask]
    yerr_band = yerr_all[mask]
    if subtract_mean:
        fitted_mean = posterior_median_mean_function(
            flat_samples,
            t_band,
            band,
            t_ref=t_all,
            survey_idx=survey_idx_all[mask],
            survey_names=survey_names,
        )
        values = y_band - fitted_mean
    else:
        values = y_band

    z_val = float(obj.get("z", 0.0) if z is None else z)
    t_rf = t_band / (1.0 + z_val)
    return t_band, t_rf, values, yerr_band


def bin_series_mean_and_variance(t, y, yerr=None, *, bin_width=1000.0, min_count=3):
    """Bin a 1D time series and summarize the mean and variance in each time bin."""

    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if yerr is None:
        yerr = np.full_like(y, np.nan, dtype=float)
    else:
        yerr = np.asarray(yerr, dtype=float).ravel()

    mask = np.isfinite(t) & np.isfinite(y)
    if yerr.shape == y.shape:
        mask &= np.isfinite(yerr) | ~np.isfinite(yerr)
    t = t[mask]
    y = y[mask]
    yerr = yerr[mask]

    out = {
        "bin_center": np.array([], dtype=float),
        "bin_count": np.array([], dtype=int),
        "mean": np.array([], dtype=float),
        "mean_err": np.array([], dtype=float),
        "variance": np.array([], dtype=float),
        "variance_err": np.array([], dtype=float),
    }
    if t.size == 0:
        return out

    t0 = float(np.min(t))
    t1 = float(np.max(t))
    if not np.isfinite(bin_width) or bin_width <= 0.0:
        raise ValueError("bin_width must be positive.")
    edges = np.arange(t0, t1 + bin_width, bin_width, dtype=float)
    if edges.size < 2:
        edges = np.array([t0, t0 + bin_width], dtype=float)
    if edges[-1] <= t1:
        edges = np.append(edges, edges[-1] + bin_width)

    centers = []
    counts = []
    means = []
    mean_errs = []
    variances = []
    variance_errs = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (t >= lo) & (t < hi if hi < edges[-1] else t <= hi)
        if int(np.sum(in_bin)) < int(min_count):
            continue

        t_bin = t[in_bin]
        y_bin = y[in_bin]
        yerr_bin = yerr[in_bin]
        n = y_bin.size
        if n < 2:
            continue

        good_w = np.isfinite(yerr_bin) & (yerr_bin > 0.0)
        if np.any(good_w):
            w = 1.0 / np.square(yerr_bin[good_w])
            mean = float(np.average(y_bin[good_w], weights=w))
            mean_err = float(np.sqrt(1.0 / np.sum(w)))
        else:
            mean = float(np.mean(y_bin))
            mean_err = float(np.std(y_bin, ddof=1) / np.sqrt(n))

        variance = float(np.var(y_bin, ddof=1))
        variance_err = float(variance * np.sqrt(2.0 / max(n - 1, 1)))

        centers.append(float(np.mean(t_bin)))
        counts.append(int(n))
        means.append(mean)
        mean_errs.append(mean_err)
        variances.append(variance)
        variance_errs.append(variance_err)

    out["bin_center"] = np.asarray(centers, dtype=float)
    out["bin_count"] = np.asarray(counts, dtype=int)
    out["mean"] = np.asarray(means, dtype=float)
    out["mean_err"] = np.asarray(mean_errs, dtype=float)
    out["variance"] = np.asarray(variances, dtype=float)
    out["variance_err"] = np.asarray(variance_errs, dtype=float)
    return out


def fit_binned_linear_trend(x, y, yerr):
    """Fit y = a + b (x - x0) and return slope/intercept summaries."""

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    yerr = np.asarray(yerr, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0.0)
    x = x[mask]
    y = y[mask]
    yerr = yerr[mask]

    result = {
        "valid": False,
        "t_center": np.nan,
        "slope": np.nan,
        "slope_err": np.nan,
        "slope_snr": np.nan,
        "intercept": np.nan,
        "intercept_err": np.nan,
        "n_points": int(x.size),
    }
    if x.size < 3:
        return result

    x0 = float(np.median(x))
    x_centered = x - x0
    try:
        coeffs, cov = np.polyfit(
            x_centered,
            y,
            1,
            w=1.0 / np.maximum(yerr, 1e-12),
            cov=True,
        )
    except Exception:
        return result

    slope = float(coeffs[0])
    intercept = float(coeffs[1])
    slope_err = float(np.sqrt(max(cov[0, 0], 0.0)))
    intercept_err = float(np.sqrt(max(cov[1, 1], 0.0)))
    result.update(
        valid=np.isfinite(slope) and np.isfinite(slope_err),
        t_center=x0,
        slope=slope,
        slope_err=slope_err,
        slope_snr=float(slope / slope_err) if np.isfinite(slope_err) and slope_err > 0.0 else np.nan,
        intercept=intercept,
        intercept_err=intercept_err,
    )
    return result


def compute_g_band_residual_drift_diagnostics(
    flat_samples,
    obj,
    bands,
    *,
    z=None,
    bin_width_rf_days=1000.0,
    min_count=3,
    return_series=False,
):
    """Summarize mean/variance drift in detrended g-band residuals over rest-frame time."""

    out = {
        "g_resid_bin_width_rf_days": float(bin_width_rf_days),
        "g_resid_n_bins": 0,
        "g_resid_mean_trend_valid": False,
        "g_resid_mean_slope": np.nan,
        "g_resid_mean_slope_err": np.nan,
        "g_resid_mean_slope_snr": np.nan,
        "g_resid_mean_intercept": np.nan,
        "g_resid_mean_intercept_err": np.nan,
        "g_resid_mean_fit_t_center_rf": np.nan,
        "g_resid_var_trend_valid": False,
        "g_resid_var_slope": np.nan,
        "g_resid_var_slope_err": np.nan,
        "g_resid_var_slope_snr": np.nan,
        "g_resid_var_intercept": np.nan,
        "g_resid_var_intercept_err": np.nan,
        "g_resid_var_fit_t_center_rf": np.nan,
    }
    if return_series:
        out |= {
            "g_resid_bin_center_rf": np.array([], dtype=float),
            "g_resid_bin_count": np.array([], dtype=int),
            "g_resid_bin_mean": np.array([], dtype=float),
            "g_resid_bin_mean_err": np.array([], dtype=float),
            "g_resid_bin_variance": np.array([], dtype=float),
            "g_resid_bin_variance_err": np.array([], dtype=float),
        }

    try:
        _, t_rf, residual, yerr = extract_band_detrended_series(flat_samples, obj, bands, "g", z=z)
    except KeyError:
        return out

    binned = bin_series_mean_and_variance(
        t_rf,
        residual,
        yerr,
        bin_width=bin_width_rf_days,
        min_count=min_count,
    )
    out["g_resid_n_bins"] = int(binned["bin_center"].size)
    if return_series:
        out["g_resid_bin_center_rf"] = binned["bin_center"]
        out["g_resid_bin_count"] = binned["bin_count"]
        out["g_resid_bin_mean"] = binned["mean"]
        out["g_resid_bin_mean_err"] = binned["mean_err"]
        out["g_resid_bin_variance"] = binned["variance"]
        out["g_resid_bin_variance_err"] = binned["variance_err"]

    mean_fit = fit_binned_linear_trend(binned["bin_center"], binned["mean"], binned["mean_err"])
    var_fit = fit_binned_linear_trend(binned["bin_center"], binned["variance"], binned["variance_err"])

    out.update(
        g_resid_mean_trend_valid=bool(mean_fit["valid"]),
        g_resid_mean_slope=mean_fit["slope"],
        g_resid_mean_slope_err=mean_fit["slope_err"],
        g_resid_mean_slope_snr=mean_fit["slope_snr"],
        g_resid_mean_intercept=mean_fit["intercept"],
        g_resid_mean_intercept_err=mean_fit["intercept_err"],
        g_resid_mean_fit_t_center_rf=mean_fit["t_center"],
        g_resid_var_trend_valid=bool(var_fit["valid"]),
        g_resid_var_slope=var_fit["slope"],
        g_resid_var_slope_err=var_fit["slope_err"],
        g_resid_var_slope_snr=var_fit["slope_snr"],
        g_resid_var_intercept=var_fit["intercept"],
        g_resid_var_intercept_err=var_fit["intercept_err"],
        g_resid_var_fit_t_center_rf=var_fit["t_center"],
    )
    return out


def compute_g_band_raw_drift_diagnostics(
    flat_samples,
    obj,
    bands,
    *,
    z=None,
    bin_width_rf_days=1000.0,
    min_count=3,
):
    """Summarize mean/variance drift in raw g-band light-curve values over rest-frame time."""

    out = {
        "g_raw_bin_width_rf_days": float(bin_width_rf_days),
        "g_raw_n_bins": 0,
        "g_raw_mean_trend_valid": False,
        "g_raw_mean_slope": np.nan,
        "g_raw_mean_slope_err": np.nan,
        "g_raw_mean_slope_snr": np.nan,
        "g_raw_mean_intercept": np.nan,
        "g_raw_mean_intercept_err": np.nan,
        "g_raw_mean_fit_t_center_rf": np.nan,
        "g_raw_var_trend_valid": False,
        "g_raw_var_slope": np.nan,
        "g_raw_var_slope_err": np.nan,
        "g_raw_var_slope_snr": np.nan,
        "g_raw_var_intercept": np.nan,
        "g_raw_var_intercept_err": np.nan,
        "g_raw_var_fit_t_center_rf": np.nan,
    }

    try:
        _, t_rf, values, yerr = extract_band_detrended_series(
            flat_samples,
            obj,
            bands,
            "g",
            z=z,
            subtract_mean=False,
        )
    except KeyError:
        return out

    binned = bin_series_mean_and_variance(
        t_rf,
        values,
        yerr,
        bin_width=bin_width_rf_days,
        min_count=min_count,
    )
    out["g_raw_n_bins"] = int(binned["bin_center"].size)

    mean_fit = fit_binned_linear_trend(binned["bin_center"], binned["mean"], binned["mean_err"])
    var_fit = fit_binned_linear_trend(binned["bin_center"], binned["variance"], binned["variance_err"])

    out.update(
        g_raw_mean_trend_valid=bool(mean_fit["valid"]),
        g_raw_mean_slope=mean_fit["slope"],
        g_raw_mean_slope_err=mean_fit["slope_err"],
        g_raw_mean_slope_snr=mean_fit["slope_snr"],
        g_raw_mean_intercept=mean_fit["intercept"],
        g_raw_mean_intercept_err=mean_fit["intercept_err"],
        g_raw_mean_fit_t_center_rf=mean_fit["t_center"],
        g_raw_var_trend_valid=bool(var_fit["valid"]),
        g_raw_var_slope=var_fit["slope"],
        g_raw_var_slope_err=var_fit["slope_err"],
        g_raw_var_slope_snr=var_fit["slope_snr"],
        g_raw_var_intercept=var_fit["intercept"],
        g_raw_var_intercept_err=var_fit["intercept_err"],
        g_raw_var_fit_t_center_rf=var_fit["t_center"],
    )
    return out


def bending_power_law_psd(freq, log_sigma, log_tau, log_noise_floor=-99.0, alpha_high=-2.0):
    """Single-break PSD with flat low-frequency slope and configurable high-frequency slope."""

    freq = np.asarray(freq, dtype=float)
    sigma = np.power(10.0, float(log_sigma))
    tau = np.power(10.0, float(log_tau))
    slope = -float(alpha_high)
    denom = 1.0 + np.power(np.clip(2.0 * np.pi * freq * tau, 1e-30, None), slope)
    noise_floor = np.power(10.0, float(log_noise_floor))
    return 2.0 * sigma * sigma * tau / denom + noise_floor


def fit_bending_power_law_psd(freq, power, power_lo=None, power_hi=None):
    """Fit a zero-floor bending PSD in log-space and return sigma/tau summaries."""

    freq = np.asarray(freq, dtype=float)
    power = np.asarray(power, dtype=float)
    mask = np.isfinite(freq) & np.isfinite(power) & (freq > 0.0) & (power > 0.0)
    if np.count_nonzero(mask) < 4:
        return {
            "log_sigma_bpl": np.nan,
            "log_sigma_bpl_err": np.nan,
            "log_tau_bpl": np.nan,
            "log_tau_bpl_err": np.nan,
            "log_noise_floor_bpl": np.nan,
            "log_noise_floor_bpl_err": np.nan,
            "psd_bpl_alpha_high": np.nan,
            "psd_bpl_alpha_high_err": np.nan,
            "psd_bpl_valid": False,
            "psd_bpl_nbins": float(np.count_nonzero(mask)),
        }

    freq_fit = freq[mask]
    power_fit = power[mask]
    log_power = np.log10(power_fit)

    if power_lo is not None and power_hi is not None:
        lo = np.asarray(power_lo, dtype=float)[mask]
        hi = np.asarray(power_hi, dtype=float)[mask]
        log_err = np.full_like(log_power, 0.25)
        valid_lo = np.isfinite(lo) & (lo > 0.0) & (lo < power_fit)
        valid_hi = np.isfinite(hi) & (hi > power_fit)
        err_lo = np.full_like(log_power, np.nan)
        err_hi = np.full_like(log_power, np.nan)
        err_lo[valid_lo] = log_power[valid_lo] - np.log10(lo[valid_lo])
        err_hi[valid_hi] = np.log10(hi[valid_hi]) - log_power[valid_hi]
        both = valid_lo & valid_hi
        only_lo = valid_lo & ~valid_hi
        only_hi = valid_hi & ~valid_lo
        log_err[both] = 0.5 * (err_lo[both] + err_hi[both])
        log_err[only_lo] = err_lo[only_lo]
        log_err[only_hi] = err_hi[only_hi]
        log_err = np.clip(log_err, 0.05, 0.60)
    else:
        log_err = np.full_like(log_power, 0.2)

    f_mid = np.exp(np.mean(np.log(freq_fit)))
    tau_init = 1.0 / (2.0 * np.pi * f_mid)
    sigma_init = np.sqrt(
        np.clip(
            np.median(power_fit) * (1.0 + (2.0 * np.pi * f_mid * tau_init) ** 2) / (2.0 * tau_init),
            1e-12,
            None,
        )
    )
    alpha_high_init = -2.0
    fixed_log_noise_floor = -99.0

    def model_log10(freq_val, log_sigma, log_tau, alpha_high):
        psd = bending_power_law_psd(freq_val, log_sigma, log_tau, fixed_log_noise_floor, alpha_high)
        return np.log10(np.clip(psd, 1e-300, None))

    try:
        popt, pcov = curve_fit(
            model_log10,
            freq_fit,
            log_power,
            p0=(np.log10(sigma_init), np.log10(tau_init), alpha_high_init),
            sigma=log_err,
            absolute_sigma=True,
            bounds=([-6.0, -1.0, -2.5], [3.0, 6.0, -1.5]),
            maxfev=20000,
        )
        perr = np.sqrt(np.diag(pcov))
    except Exception as exc:
        logging.warning("Bending-PSD fit failed: %s", exc)
        return {
            "log_sigma_bpl": np.nan,
            "log_sigma_bpl_err": np.nan,
            "log_tau_bpl": np.nan,
            "log_tau_bpl_err": np.nan,
            "log_noise_floor_bpl": np.nan,
            "log_noise_floor_bpl_err": np.nan,
            "psd_bpl_alpha_high": np.nan,
            "psd_bpl_alpha_high_err": np.nan,
            "psd_bpl_valid": False,
            "psd_bpl_nbins": float(freq_fit.size),
        }

    tau_char = 10.0 ** float(popt[1])
    tau_min = float(np.nanmin(1.0 / (2.0 * np.pi * freq_fit)))
    tau_max = float(np.nanmax(1.0 / (2.0 * np.pi * freq_fit)))
    near_tau_lower_bound = np.isclose(float(popt[1]), -1.0, atol=0.05)
    near_tau_upper_bound = np.isclose(float(popt[1]), 6.0, atol=0.05)
    near_slope_lower_bound = np.isclose(float(popt[2]), -2.5, atol=0.03)
    near_slope_upper_bound = np.isclose(float(popt[2]), -1.5, atol=0.03)
    turnover_bracketed = np.isfinite(tau_char) and (tau_min < tau_char < tau_max)

    return {
        "log_sigma_bpl": float(popt[0]),
        "log_sigma_bpl_err": float(perr[0]) if np.all(np.isfinite(perr)) else np.nan,
        "log_tau_bpl": float(popt[1]),
        "log_tau_bpl_err": float(perr[1]) if np.all(np.isfinite(perr)) else np.nan,
        "log_noise_floor_bpl": fixed_log_noise_floor,
        "log_noise_floor_bpl_err": 0.0,
        "psd_bpl_alpha_high": float(popt[2]),
        "psd_bpl_alpha_high_err": float(perr[2]) if np.all(np.isfinite(perr)) else np.nan,
        "psd_bpl_valid": bool(
            np.all(np.isfinite(popt))
            and not near_tau_lower_bound
            and not near_tau_upper_bound
            and not near_slope_lower_bound
            and not near_slope_upper_bound
            and turnover_bracketed
        ),
        "psd_bpl_nbins": float(freq_fit.size),
    }


def compute_lomb_scargle_break_diagnostics(model, samples, obj, z, *, n_freq=500):
    """Fit a bending power law to the plotted combined Lomb-Scargle PSD and convert to UV."""

    ls_bins_per_decade = 3
    ls_min_per_bin = 5
    bands = list(obj["bands"])
    lam_rf = np.asarray([lambda_pivot[band] / (1.0 + float(z)) for band in bands], dtype=float)
    ref_idx = int(np.argmin(np.abs(lam_rf - 2500.0)))
    ref_band = bands[ref_idx]
    lam_ref_band = float(lam_rf[ref_idx])
    posterior_median = {k: np.median(v, axis=0) for k, v in samples.items()}
    freqs = np.logspace(-6, 2, n_freq)
    y_psd = subtract_gp_mean_for_psd(
        model,
        posterior_median,
        obj["X"],
        obj["y"],
        survey_idx=obj.get("survey_idx"),
    )

    f_bin_norm, p_bin_norm, p_lo_norm, p_hi_norm, counts_norm, p_noise_norm = combined_lomb_scargle_from_model(
        model,
        obj["y"],
        obj["yerr"],
        posterior_median,
        2.0 * np.pi * freqs,
        amp_scaling_mode="absolute_gp_normalized",
    )
    f_bin_raw, _p_raw_raw, p_bin_raw, p_lo_raw, p_hi_raw, counts_raw, p_noise_raw = combined_raw_band_lomb_scargle(
        obj["X"],
        y_psd,
        obj["yerr"],
        posterior_median,
        2.0 * np.pi * freqs,
        ref_band_idx=ref_idx,
        bins_per_decade=ls_bins_per_decade,
        min_per_bin=ls_min_per_bin,
        band_wavelength_rf=lam_rf,
        survey_idx=obj.get("survey_idx"),
    )
    f_response_raw, response_raw = estimate_model_window_response(
        model,
        samples,
        obj["X"],
        obj["yerr"],
        2.0 * np.pi * freqs,
        ref_band_idx=ref_idx,
        bins_per_decade=ls_bins_per_decade,
        min_per_bin=ls_min_per_bin,
        band_wavelength_rf=lam_rf,
        survey_idx=obj.get("survey_idx"),
    )
    p_bin_raw, p_lo_raw, p_hi_raw, _response_at_raw_bin = apply_window_response_correction(
        f_bin_raw,
        p_bin_raw,
        p_lo_raw,
        p_hi_raw,
        f_response_raw,
        response_raw,
    )

    model_psd = np.asarray(
        model.psd(
            {k: jnp.array(v) for k, v in posterior_median.items()},
            2.0 * np.pi * freqs,
            b=ref_idx,
            sigma_n2=0.0,
        )
    )
    if "log_tau_uv" in samples:
        n_total = int(len(np.asarray(samples["log_tau_uv"])))
    else:
        n_total = 0
        for value in samples.values():
            arr = np.asarray(value)
            if arr.ndim > 0 and arr.shape[0] > 1:
                n_total = int(arr.shape[0])
                break
    psd_display_samples = []
    for i in range(min(50, n_total)):
        sample_params = posterior_sample_params_at_index(samples, i, n_total)
        psd_i = model.psd(
            sample_params,
            2.0 * np.pi * freqs,
            b=ref_idx,
            sigma_n2=0.0,
        )
        psd_display_samples.append(np.asarray(psd_i))
    psd_display_median = (
        np.median(np.stack(psd_display_samples, axis=0), axis=0)
        if psd_display_samples
        else model_psd
    )
    if f_bin_norm.size > 0 and p_bin_norm.size > 0 and np.all(np.isfinite(model_psd)):
        model_at_f0 = float(np.interp(f_bin_norm[0], freqs, model_psd))
        scale = model_at_f0 / max(float(p_bin_norm[0]), 1e-30)
        p_bin_fit_norm = p_bin_norm * scale
        p_lo_fit_norm = p_lo_norm * scale
        p_hi_fit_norm = p_hi_norm * scale
    else:
        p_bin_fit_norm = p_bin_norm
        p_lo_fit_norm = p_lo_norm
        p_hi_fit_norm = p_hi_norm

    psd_bpl_fit_fmax = 2e-3
    fit_norm_mask = (
        np.isfinite(f_bin_norm)
        & np.isfinite(p_bin_fit_norm)
        & np.isfinite(p_lo_fit_norm)
        & np.isfinite(p_hi_fit_norm)
        & (f_bin_norm <= psd_bpl_fit_fmax)
    )
    fit_norm = fit_bending_power_law_psd(
        f_bin_norm[fit_norm_mask],
        p_bin_fit_norm[fit_norm_mask],
        p_lo_fit_norm[fit_norm_mask],
        p_hi_fit_norm[fit_norm_mask],
    )

    psd_xlim = (8e-6, 1.5e-2)
    psd_ymin = 2e-2
    signal_finite = (
        np.isfinite(f_bin_raw)
        & np.isfinite(p_bin_raw)
        & np.isfinite(p_lo_raw)
        & np.isfinite(p_hi_raw)
    )
    signal_plot = signal_finite.copy()
    p_bin_display = np.asarray(p_bin_raw, dtype=float).copy()
    model_at_bin = np.interp(f_bin_raw, freqs, psd_display_median, left=np.nan, right=np.nan)
    display_floor = np.maximum(1.35 * psd_ymin, 0.35 * model_at_bin)
    display_floor = np.where(np.isfinite(display_floor), display_floor, 1.35 * psd_ymin)
    floor_plotted = signal_plot & (p_bin_display <= 0.0)
    zero_cross = signal_plot & (p_bin_display > 0.0) & (p_lo_raw <= 0.0)
    lower_span = np.clip(p_bin_display - p_lo_raw, 1e-300, None)
    zero_cross_frac = np.clip(-p_lo_raw / lower_span, 0.0, 1.0)
    shrink = np.clip(0.25 + 1.75 * zero_cross_frac, 0.0, 1.0)
    p_bin_display[zero_cross] = (
        (1.0 - shrink[zero_cross]) * p_bin_display[zero_cross]
        + shrink[zero_cross] * display_floor[zero_cross]
    )
    p_bin_display[floor_plotted] = display_floor[floor_plotted]
    signal_plot &= p_bin_display > 0.0
    p_display_err_lo = np.clip(
        p_bin_display - np.clip(p_lo_raw, 1e-300, None),
        0.0,
        None,
    )
    p_display_err_hi = np.clip(p_hi_raw - p_bin_display, 0.0, None)
    p_display_err_lo[floor_plotted] = 0.0
    display_fit_mask = (
        signal_plot
        & (f_bin_raw >= psd_xlim[0])
        & (f_bin_raw <= psd_bpl_fit_fmax)
    )
    display_fit_raw = fit_bending_power_law_to_display_points(
        f_bin_raw[display_fit_mask],
        p_bin_display[display_fit_mask],
        p_display_err_lo[display_fit_mask],
        p_display_err_hi[display_fit_mask],
    )
    if display_fit_raw is None:
        fit_raw = {
            "log_sigma_bpl": np.nan,
            "log_sigma_bpl_err": np.nan,
            "log_tau_bpl": np.nan,
            "log_tau_bpl_err": np.nan,
            "log_noise_floor_bpl": np.nan,
            "log_noise_floor_bpl_err": np.nan,
            "psd_bpl_alpha_high": np.nan,
            "psd_bpl_alpha_high_err": np.nan,
            "psd_bpl_valid": False,
            "psd_bpl_nbins": float(np.count_nonzero(display_fit_mask)),
        }
    else:
        fit_raw = {
            "log_sigma_bpl": display_fit_raw["log_sigma"],
            "log_sigma_bpl_err": display_fit_raw["log_sigma_err"],
            "log_tau_bpl": display_fit_raw["log_tau"],
            "log_tau_bpl_err": display_fit_raw["log_tau_err"],
            "log_noise_floor_bpl": -99.0,
            "log_noise_floor_bpl_err": 0.0,
            "psd_bpl_alpha_high": display_fit_raw["alpha_high"],
            "psd_bpl_alpha_high_err": display_fit_raw["alpha_high_err"],
            "psd_bpl_valid": display_fit_raw["valid"],
            "psd_bpl_nbins": float(display_fit_raw["n_points"]),
        }
    eta_sigma = float(np.nanmedian(np.asarray(samples["eta_sigma"], dtype=float)))
    eta_tau = float(np.nanmedian(np.asarray(samples["eta_tau"], dtype=float)))
    log_sigma_uv = (
        fit_norm["log_sigma_bpl"] + log_single_pl(2500.0, lam_ref_band, eta_sigma)
        if np.isfinite(fit_norm["log_sigma_bpl"]) else np.nan
    )
    log_tau_uv_obs = (
        fit_norm["log_tau_bpl"] + log_single_pl(2500.0, lam_ref_band, eta_tau)
        if np.isfinite(fit_norm["log_tau_bpl"]) else np.nan
    )
    log_tau_rf = log_tau_uv_obs - np.log10(1.0 + float(z)) if np.isfinite(log_tau_uv_obs) else np.nan
    log_tau_rf_err = fit_norm["log_tau_bpl_err"]
    log_tau_ls_obs = fit_raw["log_tau_bpl"]
    sigma_ls = np.power(10.0, fit_raw["log_sigma_bpl"]) if np.isfinite(fit_raw["log_sigma_bpl"]) else np.nan
    sigma_ls_err = (
        np.log(10.0) * sigma_ls * fit_raw["log_sigma_bpl_err"]
        if np.isfinite(sigma_ls) and np.isfinite(fit_raw["log_sigma_bpl_err"])
        else np.nan
    )
    log_tau_ls = (
        fit_raw["log_tau_bpl"] - np.log10(1.0 + float(z))
        if np.isfinite(fit_raw["log_tau_bpl"])
        else np.nan
    )
    tau_ls = np.power(10.0, log_tau_ls) if np.isfinite(log_tau_ls) else np.nan
    tau_ls_err = (
        np.log(10.0) * tau_ls * fit_raw["log_tau_bpl_err"]
        if np.isfinite(tau_ls) and np.isfinite(fit_raw["log_tau_bpl_err"])
        else np.nan
    )
    tau_ls_obs = np.power(10.0, log_tau_ls_obs) if np.isfinite(log_tau_ls_obs) else np.nan
    psd_noise_floor_norm = (
        np.power(10.0, fit_norm["log_noise_floor_bpl"])
        if np.isfinite(fit_norm["log_noise_floor_bpl"])
        else np.nan
    )
    psd_noise_floor_raw = (
        np.power(10.0, fit_raw["log_noise_floor_bpl"])
        if np.isfinite(fit_raw["log_noise_floor_bpl"])
        else np.nan
    )

    out = {
        "psd_bpl_ref_band": ref_band,
        "psd_bpl_ref_lambda_rf": lam_ref_band,
        "log_sigma_bpl_ref_band": fit_norm["log_sigma_bpl"],
        "log_sigma_bpl_ref_band_err": fit_norm["log_sigma_bpl_err"],
        "log_tau_bpl_ref_band": fit_norm["log_tau_bpl"],
        "log_tau_bpl_ref_band_err": fit_norm["log_tau_bpl_err"],
        "log_sigma_uv_bpl": log_sigma_uv,
        "log_sigma_uv_bpl_err": fit_norm["log_sigma_bpl_err"],
        "log_tau_uv_bpl": log_tau_uv_obs,
        "log_tau_uv_bpl_err": fit_norm["log_tau_bpl_err"],
        "log_noise_floor_bpl": fit_norm["log_noise_floor_bpl"],
        "log_noise_floor_bpl_err": fit_norm["log_noise_floor_bpl_err"],
        "psd_bpl_alpha_high": fit_norm["psd_bpl_alpha_high"],
        "psd_bpl_alpha_high_err": fit_norm["psd_bpl_alpha_high_err"],
        "log_tau_uv_rf_bpl": log_tau_rf,
        "log_tau_uv_rf_bpl_err": log_tau_rf_err,
        "psd_bpl_valid": fit_norm["psd_bpl_valid"],
        "psd_bpl_nbins": fit_norm["psd_bpl_nbins"],
        "psd_noise_floor": psd_noise_floor_norm,
        "log_sigma_ls": fit_raw["log_sigma_bpl"],
        "log_sigma_ls_err": fit_raw["log_sigma_bpl_err"],
        "sigma_ls": sigma_ls,
        "sigma_ls_err": sigma_ls_err,
        "log_tau_ls_obs": log_tau_ls_obs,
        "tau_ls_obs": tau_ls_obs,
        "log_tau_ls": log_tau_ls,
        "log_tau_ls_err": fit_raw["log_tau_bpl_err"],
        "tau_ls": tau_ls,
        "tau_ls_err": tau_ls_err,
        "alpha_high_ls": fit_raw["psd_bpl_alpha_high"],
        "alpha_high_ls_err": fit_raw["psd_bpl_alpha_high_err"],
        "log_noise_floor_ls": fit_raw["log_noise_floor_bpl"],
        "log_noise_floor_ls_err": fit_raw["log_noise_floor_bpl_err"],
        "psd_noise_floor_ls": psd_noise_floor_raw,
        "psd_ls_valid": fit_raw["psd_bpl_valid"],
        "psd_ls_nbins": fit_raw["psd_bpl_nbins"],
    }
    if np.isfinite(fit_norm["log_tau_bpl"]):
        out["log_nu_break_bpl"] = -np.log10(2.0 * np.pi) - fit_norm["log_tau_bpl"]
    else:
        out["log_nu_break_bpl"] = np.nan
    if np.isfinite(fit_raw["log_tau_bpl"]):
        out["log_nu_break_ls"] = -np.log10(2.0 * np.pi) - fit_raw["log_tau_bpl"]
    else:
        out["log_nu_break_ls"] = np.nan
    return out


def _structure_function_pairs(t, y, yerr=None, *, use_inverse_variance_weights=True):
    """Return same-band pair lags, noise-subtracted squared differences, and pair weights."""

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.zeros_like(y) if yerr is None else np.asarray(yerr, dtype=float)

    n = t.size
    if n < 2:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    dt = np.abs(t[:, None] - t[None, :])
    dy2 = np.square(y[:, None] - y[None, :])
    noise2 = np.square(yerr[:, None]) + np.square(yerr[None, :])

    iu = np.triu_indices(n, k=1)
    tau = dt[iu]
    noise2_term = noise2[iu]
    sf2_term = dy2[iu] - noise2_term
    if use_inverse_variance_weights:
        sf2_weight = 1.0 / np.maximum(np.square(noise2_term), 1e-8)
    else:
        sf2_weight = np.ones_like(sf2_term)
    return tau, sf2_term, sf2_weight


def _bin_structure_function_pairs(
    tau,
    sf2_term,
    sf2_weight=None,
    *,
    bins_per_decade=2,
    min_pairs=1,
    edges=None,
):
    """Bin precomputed SF pair terms in linear-lag bins."""

    tau = np.asarray(tau, dtype=float)
    sf2_term = np.asarray(sf2_term, dtype=float)
    if sf2_weight is None:
        sf2_weight = np.ones_like(sf2_term)
    sf2_weight = np.asarray(sf2_weight, dtype=float)

    good = (
        np.isfinite(tau)
        & np.isfinite(sf2_term)
        & np.isfinite(sf2_weight)
        & (tau > 0.0)
        & (sf2_weight > 0.0)
    )
    if np.count_nonzero(good) < min_pairs:
        return np.array([]), np.array([]), np.array([]), np.array([])

    tau = tau[good]
    sf2_term = sf2_term[good]
    sf2_weight = sf2_weight[good]
    fixed_edges = edges is not None
    if not fixed_edges:
        tmin = np.min(tau)
        tmax = np.max(tau)
        if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
            return np.array([]), np.array([]), np.array([]), np.array([])

        decades = np.log10(tmax) - np.log10(tmin)
        n_bins = max(1, int(np.ceil(bins_per_decade * decades)))
        edges = np.linspace(tmin, tmax, n_bins + 1)
    else:
        edges = np.asarray(edges, dtype=float)
        n_bins = max(1, edges.size - 1)
    which = np.clip(np.digitize(tau, edges) - 1, 0, n_bins - 1)

    tau_bin, sf_bin, sf_lo, sf_hi = [], [], [], []
    for k in range(n_bins):
        sel = which == k
        if np.count_nonzero(sel) < min_pairs:
            if fixed_edges:
                tau_bin.append(0.5 * (edges[k] + edges[k + 1]))
                sf_bin.append(np.nan)
                sf_lo.append(np.nan)
                sf_hi.append(np.nan)
            continue
        tau_chunk = tau[sel]
        sf2_chunk = sf2_term[sel]
        weight_chunk = sf2_weight[sel]
        weight_sum = float(np.sum(weight_chunk))
        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
            if fixed_edges:
                tau_bin.append(0.5 * (edges[k] + edges[k + 1]))
                sf_bin.append(np.nan)
                sf_lo.append(np.nan)
                sf_hi.append(np.nan)
            continue

        sf2_lo, sf2_mean, sf2_hi = _weighted_quantile(
            sf2_chunk,
            weight_chunk,
            [0.16, 0.5, 0.84],
        )
        sf_bin_val = np.sqrt(max(float(sf2_mean), 0.0))

        tau_bin.append(
            0.5 * (edges[k] + edges[k + 1])
            if fixed_edges
            else np.mean(tau_chunk)
        )
        sf_bin.append(sf_bin_val)
        sf_lo.append(np.sqrt(max(float(sf2_lo), 0.0)) if np.isfinite(sf2_lo) else np.nan)
        sf_hi.append(np.sqrt(max(float(sf2_hi), 0.0)) if np.isfinite(sf2_hi) else np.nan)

    return (
        np.asarray(tau_bin, dtype=float),
        np.asarray(sf_bin, dtype=float),
        np.asarray(sf_lo, dtype=float),
        np.asarray(sf_hi, dtype=float),
    )


def _structure_function_bin_edges(tau, *, bins_per_decade=2):
    """Return linear-spaced SF bin edges from the finite positive pair lags."""

    tau = np.asarray(tau, dtype=float)
    tau = tau[np.isfinite(tau) & (tau > 0.0)]
    if tau.size == 0:
        return np.array([], dtype=float)
    tmin = float(np.min(tau))
    tmax = float(np.max(tau))
    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
        return np.array([], dtype=float)
    decades = np.log10(tmax) - np.log10(tmin)
    n_bins = max(1, int(np.ceil(bins_per_decade * decades)))
    return np.linspace(tmin, tmax, n_bins + 1)


def _weighted_quantile(values, weights, quantiles):
    """Return weighted quantiles for finite 1D samples."""

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if np.count_nonzero(mask) == 0:
        return np.full_like(quantiles, np.nan, dtype=float)

    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return np.interp(np.clip(quantiles, 0.0, 1.0), cdf, values)


def empirical_structure_function(
    t,
    y,
    yerr=None,
    *,
    bins_per_decade=2,
    min_pairs=1,
    use_inverse_variance_weights=True,
):
    """Compute a binned empirical structure function from one band.

    Uses the Stone-style estimator in each lag bin:
    SF(tau) = sqrt(mean(dm^2 - err_i^2 - err_j^2)).
    """

    tau, sf2_term, sf2_weight = _structure_function_pairs(
        t,
        y,
        yerr,
        use_inverse_variance_weights=use_inverse_variance_weights,
    )
    return _bin_structure_function_pairs(
        tau,
        sf2_term,
        sf2_weight,
        bins_per_decade=bins_per_decade,
        min_pairs=min_pairs,
    )


def _noise_corrected_rms(y, yerr):
    """Return sqrt(var(y) - mean(yerr^2)) with finite-data guards."""

    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yerr)
    if np.count_nonzero(mask) < 2:
        return np.nan
    var_signal = float(np.var(y[mask], ddof=1))
    var_noise = float(np.mean(np.square(yerr[mask])))
    var_intrinsic = var_signal - var_noise
    return np.sqrt(var_intrinsic) if np.isfinite(var_intrinsic) and var_intrinsic > 0.0 else np.nan


def _posterior_median_band_survey_jitter(samples, band, survey):
    """Return posterior-median white-noise jitter for one band-survey cell."""

    jitter_key = f"log_jitter_{band}_{survey}"
    if jitter_key not in samples:
        legacy_key = f"log_jitter_{band}"
        if legacy_key not in samples:
            return 0.0
        jitter_key = legacy_key
    log_jitter = np.asarray(samples[jitter_key], dtype=float)
    finite = np.isfinite(log_jitter)
    if not np.any(finite):
        return 0.0
    return float(np.exp(np.nanmedian(log_jitter[finite])))


def empirical_structure_function_g_reference_from_all_bands(
    samples,
    obj,
    z,
    bands,
    ref_band,
    *,
    bins_per_decade=2,
    min_pairs=1,
    use_inverse_variance_weights=True,
    n_bootstrap=16,
    bootstrap_seed=0,
):
    """Combine all same-band raw SF pairs after scaling amplitudes to the g-band RMS."""

    bands = list(bands)
    if ref_band not in bands:
        raise KeyError(f"Reference band '{ref_band}' is not available in bands={bands}.")

    band_idx = np.asarray(obj["band_idx"], dtype=int)
    t_rf_all = np.asarray(obj["X"][0], dtype=float) / (1.0 + float(z))
    y_all = np.asarray(obj["y"], dtype=float)
    yerr_all = np.asarray(obj["yerr"], dtype=float)
    survey_idx_all = np.asarray(
        obj.get("survey_idx", np.zeros_like(band_idx, dtype=np.int32)),
        dtype=np.int32,
    )
    survey_names = tuple(obj.get("survey_names", LC_SURVEY_NAMES))

    band_payloads = {}
    sigma_ref = np.nan
    for i_band, band in enumerate(bands):
        mask = band_idx == i_band
        if np.count_nonzero(mask) < 2:
            continue
        jitter_eff = np.asarray(
            [
                _posterior_median_band_survey_jitter(
                    samples,
                    band,
                    survey_names[int(survey_idx)],
                )
                for survey_idx in survey_idx_all[mask]
            ],
            dtype=float,
        )
        yerr_eff = np.hypot(yerr_all[mask], jitter_eff)
        sigma_band = _noise_corrected_rms(y_all[mask], yerr_eff)
        band_payloads[band] = {
            "t_rf": t_rf_all[mask],
            "y": y_all[mask],
            "yerr": yerr_eff,
            "sigma": sigma_band,
        }
        if band == ref_band:
            sigma_ref = sigma_band

    if not np.isfinite(sigma_ref) or sigma_ref <= 0.0:
        sigma_ref = 1.0

    tau_chunks = []
    sf2_chunks = []
    sf2_weight_chunks = []
    bands_used = []
    for band in bands:
        payload = band_payloads.get(band)
        if payload is None:
            continue
        sigma_band = payload["sigma"]
        amp_scale_to_ref = (
            sigma_band / sigma_ref
            if np.isfinite(sigma_band) and sigma_band > 0.0
            else 1.0
        )
        tau_band, sf2_band, sf2_weight_band = _structure_function_pairs(
            payload["t_rf"],
            payload["y"] / amp_scale_to_ref,
            payload["yerr"] / amp_scale_to_ref,
            use_inverse_variance_weights=use_inverse_variance_weights,
        )
        if tau_band.size == 0:
            continue
        tau_chunks.append(tau_band)
        sf2_chunks.append(sf2_band)
        sf2_weight_chunks.append(sf2_weight_band)
        bands_used.append(band)

    if not tau_chunks:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            [],
        )

    tau_all = np.concatenate(tau_chunks)
    sf2_all = np.concatenate(sf2_chunks)
    sf2_weight_all = np.concatenate(sf2_weight_chunks)
    edges = _structure_function_bin_edges(tau_all, bins_per_decade=bins_per_decade)
    tau_sf, sf_med, sf_lo, sf_hi = _bin_structure_function_pairs(
        tau_all,
        sf2_all,
        sf2_weight_all,
        bins_per_decade=bins_per_decade,
        min_pairs=min_pairs,
        edges=edges,
    )
    if n_bootstrap <= 1 or tau_sf.size == 0 or edges.size < 2:
        return tau_sf, sf_med, sf_lo, sf_hi, bands_used

    rng = np.random.default_rng(bootstrap_seed)
    n_bins = edges.size - 1
    sf_boot = np.full((int(n_bootstrap), n_bins), np.nan, dtype=float)
    for i_boot in range(int(n_bootstrap)):
        tau_boot_chunks = []
        sf2_boot_chunks = []
        sf2_weight_boot_chunks = []
        for band in bands_used:
            payload = band_payloads[band]
            n_band = payload["t_rf"].size
            if n_band < 2:
                continue
            draw = rng.integers(0, n_band, size=n_band)
            sigma_band = payload["sigma"]
            amp_scale_to_ref = (
                sigma_band / sigma_ref
                if np.isfinite(sigma_band) and sigma_band > 0.0
                else 1.0
            )
            tau_band, sf2_band, sf2_weight_band = _structure_function_pairs(
                payload["t_rf"][draw],
                payload["y"][draw] / amp_scale_to_ref,
                payload["yerr"][draw] / amp_scale_to_ref,
                use_inverse_variance_weights=use_inverse_variance_weights,
            )
            if tau_band.size == 0:
                continue
            tau_boot_chunks.append(tau_band)
            sf2_boot_chunks.append(sf2_band)
            sf2_weight_boot_chunks.append(sf2_weight_band)

        if not tau_boot_chunks:
            continue
        _, sf_boot_med, _, _ = _bin_structure_function_pairs(
            np.concatenate(tau_boot_chunks),
            np.concatenate(sf2_boot_chunks),
            np.concatenate(sf2_weight_boot_chunks),
            min_pairs=min_pairs,
            edges=edges,
        )
        n_fill = min(n_bins, sf_boot_med.size)
        sf_boot[i_boot, :n_fill] = sf_boot_med[:n_fill]

    sf_lo_boot = np.full(n_bins, np.nan, dtype=float)
    sf_hi_boot = np.full(n_bins, np.nan, dtype=float)
    for k in range(n_bins):
        sf_boot_k = sf_boot[:, k]
        sf_boot_k = sf_boot_k[np.isfinite(sf_boot_k)]
        if sf_boot_k.size < 4:
            continue
        sf_lo_boot[k], sf_hi_boot[k] = np.percentile(sf_boot_k, [16.0, 84.0])
    n_fill = min(sf_med.size, sf_lo_boot.size)
    sf_lo = sf_lo.copy()
    sf_hi = sf_hi.copy()
    valid_boot = (
        np.isfinite(sf_lo_boot[:n_fill])
        & np.isfinite(sf_hi_boot[:n_fill])
        & (sf_lo_boot[:n_fill] <= sf_med[:n_fill])
        & (sf_hi_boot[:n_fill] >= sf_med[:n_fill])
    )
    sf_lo[:n_fill] = np.where(valid_boot, sf_lo_boot[:n_fill], sf_lo[:n_fill])
    sf_hi[:n_fill] = np.where(valid_boot, sf_hi_boot[:n_fill], sf_hi[:n_fill])
    return tau_sf, sf_med, sf_lo, sf_hi, bands_used


def _sf_bending_power_law_model(tau_val, log_sf_inf, log_tau, alpha_short):
    """Evaluate a flat-large-lag bending SF with a bounded short-lag slope."""

    tau_val = np.asarray(tau_val, dtype=float)
    sf_inf = 10.0 ** float(log_sf_inf)
    tau_break = 10.0 ** float(log_tau)
    ratio = np.clip(tau_val / tau_break, 1e-12, None)
    return sf_inf / (1.0 + np.power(ratio, float(alpha_short)))


def fit_structure_function(tau, sf, sf_lo=None, sf_hi=None):
    """Fit a bending power law with fixed large-lag slope and return sigma/tau."""

    tau = np.asarray(tau, dtype=float)
    sf = np.asarray(sf, dtype=float)
    mask = np.isfinite(tau) & np.isfinite(sf) & (tau > 0.0) & (sf > 0.0)
    if np.count_nonzero(mask) < 4:
        return {
            "log_sigma_sf": np.nan,
            "log_sigma_sf_err": np.nan,
            "log_tau_sf": np.nan,
            "log_tau_sf_err": np.nan,
            "sf_alpha_short": np.nan,
            "sf_alpha_short_err": np.nan,
            "sf_valid": False,
            "sf_nbins": float(np.count_nonzero(mask)),
        }

    tau_fit = tau[mask]
    sf_fit = sf[mask]

    if sf_lo is not None and sf_hi is not None:
        lo = np.asarray(sf_lo, dtype=float)[mask]
        hi = np.asarray(sf_hi, dtype=float)[mask]
        sf_err = 0.5 * (
            np.clip(sf_fit - lo, 1e-4, None) +
            np.clip(hi - sf_fit, 1e-4, None)
        )
    else:
        sf_err = np.full_like(sf_fit, np.nan)
    sf_err_floor = np.maximum(0.15 * sf_fit, max(np.median(sf_fit) * 0.05, 1e-3))
    sf_err = np.where(np.isfinite(sf_err), sf_err, sf_err_floor)
    sf_err = np.maximum(sf_err, sf_err_floor)

    sf_inf_init = np.clip(np.max(sf_fit), 1e-4, None)
    tau_init = np.exp(np.mean(np.log(tau_fit)))

    log_tau_low = np.log10(200.0)
    log_tau_high = 4.0

    alpha_short_init = -0.5
    alpha_short_low = -1.0
    alpha_short_high = -0.25
    log_sf_inf_high = 1.0
    log_sf_inf_init = np.clip(np.log10(sf_inf_init), -6.0 + 1e-3, log_sf_inf_high - 1e-3)
    log_tau_init = np.clip(np.log10(tau_init), log_tau_low + 1e-3, log_tau_high - 1e-3)

    try:
        result = least_squares(
            lambda pars: (
                _sf_bending_power_law_model(tau_fit, pars[0], pars[1], pars[2]) - sf_fit
            ) / sf_err,
            x0=np.array([log_sf_inf_init, log_tau_init, alpha_short_init], dtype=float),
            bounds=(
                np.array([-6.0, log_tau_low, alpha_short_low], dtype=float),
                np.array([log_sf_inf_high, log_tau_high, alpha_short_high], dtype=float),
            ),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=20000,
        )
        popt = result.x
        if result.jac.shape[0] > result.jac.shape[1]:
            _, svals, vh = np.linalg.svd(result.jac, full_matrices=False)
            threshold = np.finfo(float).eps * max(result.jac.shape) * svals[0]
            keep = svals > threshold
            if np.any(keep):
                pcov = (vh[keep].T / np.square(svals[keep])) @ vh[keep]
                perr = np.sqrt(np.diag(pcov))
            else:
                perr = np.full_like(popt, np.nan)
        else:
            perr = np.full_like(popt, np.nan)
    except Exception as exc:
        logging.warning("Structure-function fit failed: %s", exc)
        return {
            "log_sigma_sf": np.nan,
            "log_sigma_sf_err": np.nan,
            "log_tau_sf": np.nan,
            "log_tau_sf_err": np.nan,
            "sf_alpha_short": np.nan,
            "sf_alpha_short_err": np.nan,
            "sf_valid": False,
            "sf_nbins": float(tau_fit.size),
        }

    tau_char = 10.0 ** float(popt[1])
    tau_min = float(np.nanmin(tau_fit))
    tau_max = float(np.nanmax(tau_fit))
    near_lower_bound = np.isclose(float(popt[1]), log_tau_low, atol=0.05)
    near_upper_bound = np.isclose(float(popt[1]), log_tau_high, atol=0.05)
    slope_near_lower_bound = np.isclose(float(popt[2]), alpha_short_low, atol=0.02)
    slope_near_upper_bound = np.isclose(float(popt[2]), alpha_short_high, atol=0.02)
    turnover_bracketed = np.isfinite(tau_char) and (tau_min < tau_char < tau_max)
    sf_valid = bool(
        np.all(np.isfinite(popt))
        and not near_lower_bound
        and not near_upper_bound
        and not slope_near_lower_bound
        and not slope_near_upper_bound
        and turnover_bracketed
    )

    return {
        "log_sigma_sf": float(popt[0]),
        "log_sigma_sf_err": float(perr[0]) if np.all(np.isfinite(perr)) else np.nan,
        "log_tau_sf": float(popt[1]),
        "log_tau_sf_err": float(perr[1]) if np.all(np.isfinite(perr)) else np.nan,
        "sf_alpha_short": float(popt[2]),
        "sf_alpha_short_err": float(perr[2]) if np.all(np.isfinite(perr)) else np.nan,
        "sf_valid": sf_valid,
        "sf_nbins": float(tau_fit.size),
    }


def dho_structure_function(tau, amp, tau_fast, tau_slow):
    """Analytic SF of the exact unit-RMS continuum-only overdamped-SHO process."""

    tau = np.asarray(tau, dtype=float)
    amp = np.asarray(amp, dtype=float)
    tau_fast_ord, tau_slow_ord = ordered_dho_taus(tau_fast, tau_slow)
    rho = tau_fast_ord / np.maximum(tau_slow_ord, 1e-12)
    denom = np.maximum(1.0 - rho, 1e-12)
    cov_factor = (
        np.exp(-tau / tau_slow_ord)
        - rho * np.exp(-tau / np.maximum(tau_fast_ord, 1e-12))
    ) / denom
    sf2 = 2.0 * np.square(amp) * (1.0 - cov_factor)
    return np.sqrt(np.clip(sf2, 0.0, None))


def _build_structure_function_lag_grid(t_band, tau_sf):
    """Choose a rest-frame lag grid that reflects the observed g-band coverage."""

    tau_sf = np.asarray(tau_sf, dtype=float)
    valid_tau_sf = tau_sf[np.isfinite(tau_sf) & (tau_sf > 0.0)]
    if valid_tau_sf.size >= 4:
        return valid_tau_sf

    t_band = np.asarray(t_band, dtype=float)
    if t_band.size < 3:
        return np.array([], dtype=float)
    dt = np.abs(t_band[:, None] - t_band[None, :])
    iu = np.triu_indices(t_band.size, k=1)
    tau_pairs = dt[iu]
    tau_pairs = tau_pairs[np.isfinite(tau_pairs) & (tau_pairs > 0.0)]
    if tau_pairs.size < 4:
        return np.array([], dtype=float)

    tau_min = float(np.nanmin(tau_pairs))
    tau_max = float(np.nanmax(tau_pairs))
    if not np.isfinite(tau_min) or not np.isfinite(tau_max) or tau_max <= tau_min:
        return np.array([], dtype=float)
    return np.logspace(np.log10(tau_min), np.log10(tau_max), 16)


def compute_model_structure_function_equivalent(samples, ref_band, tau_grid, *, z=0.0, return_series=False):
    """Fit the same DRW SF form to the model-implied DHO SF in the reference band."""

    nan_out = {
        "log_sigma_sf_model_ref_band": np.nan,
        "log_sigma_sf_model_ref_band_err": np.nan,
        "log_tau_sf_model_ref_band": np.nan,
        "log_tau_sf_model_ref_band_err": np.nan,
        "sf_model_valid": False,
    }
    if return_series:
        nan_out |= {
            "sf_model_tau_ref_band": np.array([], dtype=float),
            "sf_model_curve_ref_band": np.array([], dtype=float),
        }

    amp_key = f"amp_cont_{ref_band}"
    tau_fast_key = f"tau_fast_{ref_band}"
    tau_slow_key = f"tau_slow_{ref_band}"
    if any(key not in samples for key in (amp_key, tau_fast_key, tau_slow_key)):
        return nan_out

    amp = np.asarray(samples[amp_key], dtype=float)
    tau_fast = np.asarray(samples[tau_fast_key], dtype=float)
    tau_slow = np.asarray(samples[tau_slow_key], dtype=float)
    mask = np.isfinite(amp) & np.isfinite(tau_fast) & np.isfinite(tau_slow) & (amp > 0.0) & (tau_fast > 0.0) & (tau_slow > 0.0)
    if np.count_nonzero(mask) == 0:
        return nan_out

    amp_med = float(np.nanmedian(amp[mask]))
    one_plus_z = 1.0 + float(z)
    tau_fast_med = float(np.nanmedian(tau_fast[mask])) / one_plus_z
    tau_slow_med = float(np.nanmedian(tau_slow[mask])) / one_plus_z

    tau_grid = np.asarray(tau_grid, dtype=float)
    tau_grid = tau_grid[np.isfinite(tau_grid) & (tau_grid > 0.0)]
    if tau_grid.size >= 4:
        sf_model_fit_grid = dho_structure_function(tau_grid, amp_med, tau_fast_med, tau_slow_med)
        fit = fit_structure_function(tau_grid, sf_model_fit_grid)
    else:
        fit = {
            "log_sigma_sf": np.nan,
            "log_tau_sf": np.nan,
            "sf_valid": False,
        }

    out = {
        "log_sigma_sf_model_ref_band": (
            fit["log_sigma_sf"] - LOG_SF_INF_TO_RMS
            if np.isfinite(fit["log_sigma_sf"])
            else np.nan
        ),
        "log_sigma_sf_model_ref_band_err": np.nan,
        "log_tau_sf_model_ref_band": fit["log_tau_sf"],
        "log_tau_sf_model_ref_band_err": np.nan,
        "sf_model_valid": bool(fit["sf_valid"]),
    }
    if return_series:
        tau_model_plot = np.logspace(
            np.log10(SF_MODEL_TAU_MIN_RF),
            np.log10(SF_MODEL_TAU_MAX_RF),
            SF_MODEL_TAU_N_PLOT,
        )
        sf_model_plot = dho_structure_function(
            tau_model_plot,
            amp_med,
            tau_fast_med,
            tau_slow_med,
        )
        out |= {
            "sf_model_tau_ref_band": tau_model_plot,
            "sf_model_curve_ref_band": sf_model_plot,
        }
    return out


def compute_structure_function_diagnostics(
    samples,
    obj,
    z,
    *,
    use_inverse_variance_weights=True,
    n_bootstrap=16,
    bootstrap_seed=0,
    return_series=False,
):
    """Fit a raw all-band SF after scaling each band's amplitude to the g-band RMS."""

    bands = list(obj["bands"])
    if "g" not in bands:
        raise KeyError("Missing g band required for SF diagnostics.")
    lam_rf = np.asarray([lambda_pivot[band] / (1.0 + float(z)) for band in bands], dtype=float)
    ref_idx = bands.index("g")
    ref_band = bands[ref_idx]
    lam_ref_band = float(lam_rf[ref_idx])

    band_idx = np.asarray(obj["band_idx"])
    ref_mask = band_idx == ref_idx
    t_band = np.asarray(obj["X"][0], dtype=float)[ref_mask] / (1.0 + float(z))
    ref_surveys = np.asarray(
        obj.get("survey_idx", np.zeros_like(band_idx, dtype=np.int32))[ref_mask],
        dtype=np.int32,
    )
    survey_names = tuple(obj.get("survey_names", LC_SURVEY_NAMES))
    jitter_ref_samples = np.asarray(
        [
            _posterior_median_band_survey_jitter(samples, ref_band, survey_names[int(survey_idx)])
            for survey_idx in ref_surveys
        ],
        dtype=float,
    )
    finite_jitter = jitter_ref_samples[np.isfinite(jitter_ref_samples) & (jitter_ref_samples > 0.0)]
    jitter_ref_band = float(np.median(finite_jitter)) if finite_jitter.size else 0.0
    tau_sf, sf_med, sf_lo, sf_hi, sf_bands_used = (
        empirical_structure_function_g_reference_from_all_bands(
            samples,
            obj,
            z,
            bands,
            ref_band,
            use_inverse_variance_weights=use_inverse_variance_weights,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
        )
    )
    fit = fit_structure_function(tau_sf, sf_med, sf_lo, sf_hi)
    tau_grid_model = _build_structure_function_lag_grid(t_band, tau_sf)
    model_sf_fit = compute_model_structure_function_equivalent(
        samples,
        ref_band,
        tau_grid_model,
        z=float(z),
        return_series=return_series,
    )

    log_sigma_uv = (
        fit["log_sigma_sf"] - LOG_SF_INF_TO_RMS
        if np.isfinite(fit["log_sigma_sf"])
        else np.nan
    )
    log_tau_rf = fit["log_tau_sf"]
    log_tau_uv_obs = log_tau_rf + np.log10(1.0 + float(z)) if np.isfinite(log_tau_rf) else np.nan

    out = {
        "sf_ref_band": ref_band,
        "sf_source_bands": ",".join(sf_bands_used),
        "sf_inverse_variance_weighted": bool(use_inverse_variance_weights),
        "sf_n_bootstrap": int(n_bootstrap),
        "sf_ref_lambda_rf": lam_ref_band,
        "log_jitter_sf_ref_band": np.log10(jitter_ref_band) if jitter_ref_band > 0.0 else np.nan,
        "log_sigma_sf_ref_band": log_sigma_uv,
        "log_sigma_sf_ref_band_err": fit["log_sigma_sf_err"],
        "log_tau_sf_ref_band": fit["log_tau_sf"],
        "log_tau_sf_ref_band_err": fit["log_tau_sf_err"],
        "sf_alpha_short_ref_band": fit["sf_alpha_short"],
        "sf_alpha_short_ref_band_err": fit["sf_alpha_short_err"],
        "log_sigma_uv_sf": log_sigma_uv,
        "log_sigma_uv_sf_err": fit["log_sigma_sf_err"],
        "log_tau_uv_sf": log_tau_uv_obs,
        "log_tau_uv_sf_err": fit["log_tau_sf_err"],
        "log_tau_uv_rf_sf": log_tau_rf,
        "log_tau_uv_rf_sf_err": fit["log_tau_sf_err"],
        "sf_valid": fit["sf_valid"],
        "sf_nbins": fit["sf_nbins"],
        **model_sf_fit,
    }
    if return_series:
        out |= {
            "sf_tau_ref_band": tau_sf,
            "sf_curve_ref_band": sf_med,
            "sf_curve_lo_ref_band": sf_lo,
            "sf_curve_hi_ref_band": sf_hi,
        }
    return out


def log_nonfinite_sample_summary(samples_dict, *, label, max_items=20):
    """Log per-parameter non-finite fractions to diagnose pathological chains."""

    bad = []
    for key, value in samples_dict.items():
        arr = np.asarray(value)
        if arr.size == 0:
            continue
        finite_frac = float(np.mean(np.isfinite(arr)))
        if finite_frac < 1.0:
            bad.append((key, finite_frac, arr.shape))

    if not bad:
        logging.info("[%s] All sampled parameters are finite.", label)
        return

    bad.sort(key=lambda item: item[1])
    preview = ", ".join(
        f"{key}{shape}={finite_frac:.1%} finite"
        for key, finite_frac, shape in bad[:max_items]
    )
    logging.warning(
        "[%s] Non-finite samples detected in %d parameter(s): %s%s",
        label,
        len(bad),
        preview,
        " ..." if len(bad) > max_items else "",
    )


def print_light_curve_posterior_summary(
    object_id,
    *,
    samples_per_chain=None,
    flat_samples=None,
):
    """Print one NumPyro posterior summary after an object's final fit."""

    if samples_per_chain is not None:
        samples = samples_per_chain
        group_by_chain = True
    elif flat_samples is not None:
        samples = flat_samples
        group_by_chain = False
    else:
        logging.warning("[%s] No posterior samples available for NumPyro summary.", object_id)
        return

    print(f"\n[{object_id}] NumPyro posterior summary:")
    try:
        numpyro_print_summary(samples, prob=0.90, group_by_chain=group_by_chain)
    except (AssertionError, ValueError) as exc:
        # NumPyro requires at least four draws for split-Rhat/ESS. Keep very
        # short smoke fits usable instead of failing after inference succeeds.
        print(f"Summary unavailable: {exc or 'at least four posterior draws are required.'}")
    print()


def sigma_shift_to_uv(eta_sigma, lambda_center_rf, lambda_uv=2500.0):
    return jnp.log(10.0) * log_single_pl(lambda_uv, lambda_center_rf, eta_sigma)


def tau_shift_to_uv(eta_tau, lambda_center_rf, lambda_uv=2500.0):
    return jnp.log(10.0) * log_single_pl(lambda_uv, lambda_center_rf, eta_tau)


def eta_sigma_prior():
    return dist.TruncatedNormal(-0.5, 1.0) #, low=ETA_SIGMA_LOW, high=0.0)


def eta_tau_prior():
    return dist.Normal(0.5, 0.5)


def log_sigma_center0_prior(eta_sigma, lambda_center_rf):
    sigma_shift = sigma_shift_to_uv(eta_sigma, lambda_center_rf)
    return dist.Normal(-0.6 * jnp.log(10.0) - sigma_shift, 1.0 * jnp.log(10.0))


def log_sigma_center0_relflux_prior(eta_sigma, lambda_center_rf):
    sigma_shift = sigma_shift_to_uv(eta_sigma, lambda_center_rf)
    return dist.Normal(
        -0.6 * jnp.log(10.0) - sigma_shift - LOG_RELFLUX_TO_MAG_SCALE,
        1.0 * jnp.log(10.0),
    )


def log_tau_slow_center0_prior(eta_tau, z, lambda_center_rf):
    shift = tau_shift_to_uv(eta_tau, lambda_center_rf)
    log_tau_uv_high = jnp.log(10**4.0 * (1.0 + z))
    log_tau_uv_low = jnp.log(10.0 * (1.0 + z))
    return dist.TruncatedNormal(
        jnp.log(10**2.5 * (1.0 + z)) - shift,
        1.2 * jnp.log(10.0),
        low=log_tau_uv_low - shift,
        high=log_tau_uv_high - shift,
    )


TAU_FAST_TO_SLOW_PRIOR_RATIO = 150.0
TAU_FAST_PRIOR_LOGSIGMA_DEX = 0.4


def log_tau_fast_center0_prior(log_tau_slow_center0, *, tau_fast_truncated=False):
    mean = jnp.asarray(log_tau_slow_center0) - jnp.log(TAU_FAST_TO_SLOW_PRIOR_RATIO)
    sigma = TAU_FAST_PRIOR_LOGSIGMA_DEX * jnp.log(10.0)
    if tau_fast_truncated:
        return dist.TruncatedNormal(
            mean,
            sigma,
            high=jnp.asarray(log_tau_slow_center0),
        )
    return dist.Normal(mean, sigma)


def log_tau_fast_separation_raw_prior():
    """Prior coordinate for a smooth, strictly positive log-timescale gap."""

    target_gap = jnp.log(TAU_FAST_TO_SLOW_PRIOR_RATIO)
    raw_loc = jnp.log(jnp.expm1(target_gap))
    return dist.Normal(raw_loc, TAU_FAST_PRIOR_LOGSIGMA_DEX * jnp.log(10.0))


def ordered_log_tau_fast(log_tau_slow, separation_raw):
    """Map an unconstrained coordinate to ``log_tau_fast < log_tau_slow``."""

    return jnp.asarray(log_tau_slow) - jax.nn.softplus(jnp.asarray(separation_raw))


def linear_trend_prior(*, t_ref, z):
    _, t_std_obs = linear_mean_time_scaling(t_ref)
    scale = max(
        LINEAR_TREND_RF_SIGMA_MAG_PER_DAY * float(t_std_obs) / float(1.0 + z),
        1e-8,
    )
    return dist.Normal(0.0, scale)


def linear_trend_prior_relflux(*, t_ref, z):
    mag_prior = linear_trend_prior(t_ref=t_ref, z=z)
    return dist.Normal(0.0, float(mag_prior.scale) / RELFLUX_TO_MAG_SCALE)


def linear_trend_band_offset_raw_prior():
    return dist.Normal(0.0, 0.02)


def linear_trend_band_offset_raw_prior_relflux():
    return dist.Normal(0.0, 0.02 / RELFLUX_TO_MAG_SCALE)


def linear_trend_band_offset_prior(B, *, relflux=False):
    """Marginal prior for the centered per-band slope offsets."""

    if B <= 1:
        return dist.Normal(0.0, 1e-6)
    base_scale = (
        0.02 / RELFLUX_TO_MAG_SCALE
        if relflux
        else 0.02
    )
    return dist.Normal(0.0, base_scale * jnp.sqrt((B - 1) / B))


def lag0_prior(z=0.0):
    one_plus_z = 1.0 + jnp.asarray(z, dtype=float)
    return dist.TruncatedNormal(
        5.0 * one_plus_z,
        5.0 * one_plus_z,
        low=0.0,
        high=LAG0_HIGH * one_plus_z,
    )


def log_lag0_prior(z=0.0):
    return dist.TransformedDistribution(
        lag0_prior(z=z),
        dist.transforms.ExpTransform().inv,
    )


def lag_beta_prior():
    return dist.TruncatedNormal(4.0 / 3.0, 0.2, low=0.0, high=LAG_BETA_HIGH)


def mean_prior():
    return dist.Normal(0.0, 0.2)


def mean_prior_relflux():
    return dist.Normal(0.0, 0.2 / RELFLUX_TO_MAG_SCALE)


def log_jitter_prior(log_jitter_mean):
    return dist.Normal(log_jitter_mean, 1.0)


def survey_delta_mag_prior():
    return dist.Normal(0.0, 0.02)


def dlog_amp_blr_prior():
    return dist.Normal(-1.0, 1.0)


def log_lag_blr_prior(z=0.0):
    # Keep the model lag in observed-frame days, but anchor the prior in rest-frame days.
    log_1pz = jnp.log1p(jnp.asarray(z, dtype=float))
    return dist.TruncatedNormal(
        loc=jnp.log(1e2) + log_1pz,
        scale=jnp.log(10.0),
        low=LOG_LAG_BLR_LOW + log_1pz,
        high=LOG_LAG_BLR_HIGH + log_1pz,
    )


def relative_log_lag_blr_prior(*, z=0.0, log_lag0=0.0):
    # Shifting a TruncatedNormal with an inverse AffineTransform causes
    # NumPyro to report unconstrained Real support for the transformed
    # distribution.  Since TruncatedNormal.log_prob does not itself mask
    # values outside its support, NUTS can then explore lags beyond the
    # intended bounds.  Shift all parameters explicitly so the interval
    # support is retained by the sampler.
    prior = log_lag_blr_prior(z=z)
    log_lag0 = jnp.asarray(log_lag0, dtype=float)
    return dist.TruncatedNormal(
        loc=prior.base_dist.loc - log_lag0,
        scale=prior.base_dist.scale,
        low=prior.low - log_lag0,
        high=prior.high - log_lag0,
    )


def dlog_amp_bc_prior():
    return dist.Normal(-1.0, 1.0)


def log_lag_ratio_bc_to_blr_prior():
    return dist.TruncatedNormal(
        loc=jnp.log(0.2),
        scale=0.15,
        low=LOG_LAG_RATIO_BC_TO_BLR_LOW,
        high=LOG_LAG_RATIO_BC_TO_BLR_HIGH,
    )


def log_cont_scale_prior():
    return dist.Normal(0.0, FLUXMIX_CONT_SCALE_PRIOR_SIGMA)


def sample_linear_trend_with_band_offsets(
    *,
    B,
    disable_linear_trend=False,
    trend_prior_dist=None,
    band_offset_raw_prior_dist=None,
    shared_linear_trend=False,
):
    """Sample a global slope plus zero-sum per-band slope offsets."""

    if trend_prior_dist is None:
        trend_prior_dist = dist.Normal(0.0, 0.1)
    if band_offset_raw_prior_dist is None:
        band_offset_raw_prior_dist = linear_trend_band_offset_raw_prior()

    zeros = jnp.zeros(B, dtype=float)
    if disable_linear_trend:
        linear_trend = numpyro.deterministic("linear_trend", 0.0)
        linear_trend_band_offset = numpyro.deterministic(
            "linear_trend_band_offset",
            zeros,
        )
        linear_trend_band = numpyro.deterministic(
            "linear_trend_band",
            zeros,
        )
        return linear_trend, linear_trend_band_offset, linear_trend_band

    linear_trend = numpyro.sample("linear_trend", trend_prior_dist)
    if shared_linear_trend:
        linear_trend_band_offset = numpyro.deterministic(
            "linear_trend_band_offset",
            zeros,
        )
        linear_trend_band = numpyro.deterministic(
            "linear_trend_band",
            jnp.full(B, linear_trend),
        )
        return linear_trend, linear_trend_band_offset, linear_trend_band

    with numpyro.plate("band", B):
        linear_trend_band_offset_raw = numpyro.sample(
            "linear_trend_band_offset_raw",
            band_offset_raw_prior_dist,
        )
    linear_trend_band_offset = numpyro.deterministic(
        "linear_trend_band_offset",
        linear_trend_band_offset_raw - jnp.mean(linear_trend_band_offset_raw),
    )
    linear_trend_band = numpyro.deterministic(
        "linear_trend_band",
        linear_trend + linear_trend_band_offset,
    )
    return linear_trend, linear_trend_band_offset, linear_trend_band


def compute_flux_line_ratio_offsets(
    lam_rf,
    *,
    lambda_center_rf,
    eta_sigma,
    log_igm_transmission_band=None,
):
    """Offsets mapping sampled line/continuum log-ratios back to legacy amplitude deltas."""

    lam_rf = jnp.asarray(lam_rf, dtype=float)
    lambda_center_rf = jnp.asarray(lambda_center_rf, dtype=lam_rf.dtype)
    eta_sigma = jnp.asarray(eta_sigma, dtype=lam_rf.dtype)
    if log_igm_transmission_band is None:
        log_igm_transmission_band = jnp.zeros_like(lam_rf, dtype=lam_rf.dtype)
    log_igm_transmission_band = jnp.asarray(log_igm_transmission_band, dtype=lam_rf.dtype)
    log_igm_transmission_band = jnp.broadcast_to(log_igm_transmission_band, lam_rf.shape)

    lambda_uv = jnp.array(2500.0, dtype=lam_rf.dtype)
    sigma_shift_to_uv = jnp.log(10.0) * log_single_pl(lambda_uv, lambda_center_rf, eta_sigma)
    sigma_shift_to_band = jnp.log(10.0) * log_single_pl(
        lam_rf,
        _expand_last(lambda_center_rf),
        _expand_last(eta_sigma),
    )
    log_ratio_offset_blr = _expand_last(sigma_shift_to_uv) - sigma_shift_to_band - log_igm_transmission_band

    bc_weight = balmer_continuum_weight(lam_rf)
    log_ratio_offset_bc_band = log_ratio_offset_blr + jnp.log(jnp.maximum(bc_weight, 1e-12))
    bc_ref_weights = jnp.maximum(bc_weight, 1e-6)
    log_ratio_offset_bc_ref = jnp.sum(bc_ref_weights * log_ratio_offset_bc_band) / jnp.sum(
        bc_ref_weights
    )

    return {
        "blr_band": log_ratio_offset_blr,
        "bc_ref": log_ratio_offset_bc_ref,
    }


def sample_flux_line_latent_params(
    *,
    B,
    z,
    log_lag0,
    log_jitter_mean,
    log_jitter_active_mask,
    survey_offset_active_mask,
    line_ratio_offsets,
    mean_prior_dist=None,
    disable_lag_blr=False,
    disable_lag_bc=False,
    n_blr_terms=1,
    shared_blr_lag=False,
    blr_lag_band_scatter=None,
):
    """Sample flux line parameters in ratio space and reconstruct legacy deltas exactly."""

    if disable_lag_blr or disable_lag_bc:
        dlog_amp_bc = None
        log_lag_ratio_bc_to_blr = None
    else:
        log_amp_ratio_bc = numpyro.sample(
            "log_amp_ratio_bc",
            dist.Normal(line_ratio_offsets["bc_ref"] - 1.0, 1.0),
        )
        dlog_amp_bc = numpyro.deterministic(
            "dlog_amp_bc",
            log_amp_ratio_bc - line_ratio_offsets["bc_ref"],
        )
        log_lag_ratio_bc_to_blr = numpyro.sample(
            "log_lag_ratio_bc_to_blr",
            log_lag_ratio_bc_to_blr_prior(),
        )

    log_amp_ratio_blr_loc = line_ratio_offsets["blr_band"] - 1.0
    if mean_prior_dist is None:
        mean_prior_dist = mean_prior()
    log_jitter = _sample_log_jitter_grid(log_jitter_mean, log_jitter_active_mask)
    survey_delta_mag = _sample_survey_delta_mag_grid(survey_offset_active_mask)

    shared_log_lag_blr = None
    use_pooled_blr_lag = shared_blr_lag or blr_lag_band_scatter is not None
    band_scatter = 0.0 if shared_blr_lag else blr_lag_band_scatter
    if use_pooled_blr_lag and not disable_lag_blr:
        if n_blr_terms != 1:
            raise ValueError("Pooled BLR lags currently require n_blr_terms=1.")
        if band_scatter is None or float(band_scatter) < 0.0:
            raise ValueError("blr_lag_band_scatter must be non-negative.")
        shared_delta_log_lag_blr_raw = numpyro.sample(
            "delta_log_lag_blr_shared_raw",
            relative_log_lag_blr_prior(z=z, log_lag0=log_lag0),
        )
        shared_log_lag_blr = shared_delta_log_lag_blr_raw + log_lag0

    with numpyro.plate("band", B):
        mean = numpyro.sample("mean", mean_prior_dist)

        if disable_lag_blr:
            dlog_amp_blr = numpyro.deterministic(
                "dlog_amp_blr",
                jnp.full(B, -9.0),
            )
            log_lag_blr = numpyro.deterministic(
                "log_lag_blr",
                jnp.full(B, -9.0),
            )
            dlog_amp_blr2 = numpyro.deterministic(
                "dlog_amp_blr2",
                jnp.full(B, -9.0),
            )
            log_lag_blr2 = numpyro.deterministic(
                "log_lag_blr2",
                jnp.full(B, -9.0),
            )
        elif n_blr_terms <= 1:
            log_amp_ratio_blr_raw = numpyro.sample(
                "log_amp_ratio_blr_raw",
                dist.Normal(log_amp_ratio_blr_loc, 1.0),
            )
            if shared_log_lag_blr is None:
                delta_log_lag_blr_raw = numpyro.sample(
                    "delta_log_lag_blr_raw",
                    relative_log_lag_blr_prior(z=z, log_lag0=log_lag0),
                )
                log_lag_blr_raw = delta_log_lag_blr_raw + log_lag0
            else:
                if float(band_scatter) > 0.0:
                    blr_lag_band_offset_raw = numpyro.sample(
                        "blr_lag_band_offset_raw",
                        dist.Normal(0.0, 1.0),
                    )
                    log_lag_blr_raw = (
                        shared_log_lag_blr
                        + jnp.asarray(band_scatter) * blr_lag_band_offset_raw
                    )
                else:
                    log_lag_blr_raw = jnp.broadcast_to(shared_log_lag_blr, (B,))
            dlog_amp_blr = numpyro.deterministic(
                "dlog_amp_blr",
                log_amp_ratio_blr_raw - line_ratio_offsets["blr_band"],
            )
            log_lag_blr = numpyro.deterministic(
                "log_lag_blr",
                log_lag_blr_raw,
            )
            dlog_amp_blr2 = numpyro.deterministic(
                "dlog_amp_blr2",
                jnp.full(B, -9.0),
            )
            log_lag_blr2 = numpyro.deterministic(
                "log_lag_blr2",
                jnp.full(B, -9.0),
            )
        else:
            log_amp_ratio_blr_raw = numpyro.sample(
                "log_amp_ratio_blr_raw",
                dist.Normal(log_amp_ratio_blr_loc, 1.0),
            )
            delta_log_lag_blr_raw = numpyro.sample(
                "delta_log_lag_blr_raw",
                relative_log_lag_blr_prior(z=z, log_lag0=log_lag0),
            )
            log_lag_blr_raw = delta_log_lag_blr_raw + log_lag0
            log_amp_ratio_blr2_raw = numpyro.sample(
                "log_amp_ratio_blr2_raw",
                dist.Normal(log_amp_ratio_blr_loc, 1.0),
            )
            delta_log_lag_blr2_raw = numpyro.sample(
                "delta_log_lag_blr2_raw",
                relative_log_lag_blr_prior(z=z, log_lag0=log_lag0),
            )
            log_lag_blr2_raw = delta_log_lag_blr2_raw + log_lag0

            first_is_short = log_lag_blr_raw <= log_lag_blr2_raw
            log_lag_blr = numpyro.deterministic(
                "log_lag_blr",
                jnp.where(first_is_short, log_lag_blr_raw, log_lag_blr2_raw),
            )
            log_lag_blr2 = numpyro.deterministic(
                "log_lag_blr2",
                jnp.where(first_is_short, log_lag_blr2_raw, log_lag_blr_raw),
            )
            dlog_amp_blr = numpyro.deterministic(
                "dlog_amp_blr",
                jnp.where(first_is_short, log_amp_ratio_blr_raw, log_amp_ratio_blr2_raw)
                - line_ratio_offsets["blr_band"],
            )
            dlog_amp_blr2 = numpyro.deterministic(
                "dlog_amp_blr2",
                jnp.where(first_is_short, log_amp_ratio_blr2_raw, log_amp_ratio_blr_raw)
                - line_ratio_offsets["blr_band"],
            )

    return (
        mean,
        dlog_amp_blr,
        dlog_amp_blr2,
        log_lag_blr,
        log_lag_blr2,
        log_jitter,
        survey_delta_mag,
        dlog_amp_bc,
        log_lag_ratio_bc_to_blr,
    )


def compute_parameter_kls(
    flat_samples,
    *,
    bands,
    survey_names,
    t_ref,
    z,
    lambda_center_rf,
    log_jitter_mean,
    model_variant="mag_linear",
    disable_linear_trend=False,
    disable_lag_blr=False,
    disable_lag_bc=False,
    drop_band_lyman_alpha=False,
    tau_fast_truncated=False,
    n_blr_terms=1,
):
    """Return approximate KL(q||p) for sampled light-curve parameters."""

    kls = {}
    eta_sigma = np.asarray(flat_samples["eta_sigma"])
    eta_tau = np.asarray(flat_samples["eta_tau"])
    sigma_center0_key = (
        "log_sigma_center0_relflux"
        if model_variant in ("mag_flux_linearized", "mag_flux_linearized_erlang") and "log_sigma_center0_relflux" in flat_samples
        else "log_sigma_center0"
    )
    sigma_prior_fn = (
        log_sigma_center0_relflux_prior if model_variant in ("mag_flux_linearized", "mag_flux_linearized_erlang") else log_sigma_center0_prior
    )
    linear_trend_prior_fn = (
        linear_trend_prior_relflux(t_ref=t_ref, z=z)
        if model_variant in ("mag_flux_linearized", "mag_flux_linearized_erlang")
        else linear_trend_prior(t_ref=t_ref, z=z)
    )
    linear_trend_band_offset_prior_fn = linear_trend_band_offset_prior(
        len(bands),
        relflux=(model_variant in ("mag_flux_linearized", "mag_flux_linearized_erlang")),
    )
    mean_prior_fn = mean_prior_relflux if model_variant in ("mag_flux_linearized", "mag_flux_linearized_erlang") else mean_prior
    log_jitter_mean_arr = np.asarray(log_jitter_mean, dtype=float)

    kls["eta_sigma_kl"] = kl_from_samples(
        eta_sigma,
        lambda x: _dist_log_prob_array(eta_sigma_prior(), x),
    )
    kls["eta_tau_kl"] = kl_from_samples(
        eta_tau,
        lambda x: _dist_log_prob_array(eta_tau_prior(), x),
    )

    if sigma_center0_key in flat_samples:
        kls["log_sigma_center0_kl"] = conditional_kl_from_samples(
            flat_samples[sigma_center0_key],
            lambda x, eta: _dist_log_prob_array(
                sigma_prior_fn(
                    eta,
                    lambda_center_rf,
                ),
                x,
            ),
            eta_sigma,
        )

    if "log_tau_slow_center0" in flat_samples:
        kls["log_tau_slow_center0_kl"] = conditional_kl_from_samples(
            flat_samples["log_tau_slow_center0"],
            lambda x, eta: _dist_log_prob_array(
                log_tau_slow_center0_prior(
                    eta,
                    z,
                    lambda_center_rf,
                ),
                x,
            ),
            eta_tau,
        )

    if "log_tau_fast_center0" in flat_samples:
        kls["log_tau_fast_center0_kl"] = conditional_kl_from_samples(
            flat_samples["log_tau_fast_center0"],
            lambda x, log_tau_slow: _dist_log_prob_array(
                log_tau_fast_center0_prior(
                    log_tau_slow,
                    tau_fast_truncated=tau_fast_truncated,
                ),
                x,
            ),
            flat_samples["log_tau_slow_center0"],
        )

    if not disable_linear_trend and "linear_trend" in flat_samples:
        kls["linear_trend_kl"] = kl_from_samples(
            flat_samples["linear_trend"],
            lambda x: _dist_log_prob_array(linear_trend_prior_fn, x),
        )

    if model_variant != "mag_flux_linearized_erlang" and "lag0" in flat_samples:
        kls["lag0_kl"] = kl_from_samples(
            flat_samples["lag0"],
            lambda x: _dist_log_prob_array(lag0_prior(z=z), x),
        )
    if model_variant != "mag_flux_linearized_erlang" and "lag_beta" in flat_samples:
        kls["lag_beta_kl"] = kl_from_samples(
            flat_samples["lag_beta"],
            lambda x: _dist_log_prob_array(lag_beta_prior(), x),
        )

    if not disable_lag_blr and not disable_lag_bc:
        if "dlog_amp_bc" in flat_samples:
            kls["dlog_amp_bc_kl"] = kl_from_samples(
                flat_samples["dlog_amp_bc"],
                lambda x: _dist_log_prob_array(dlog_amp_bc_prior(), x),
            )
        if "log_lag_ratio_bc_to_blr" in flat_samples:
            kls["log_lag_ratio_bc_to_blr_kl"] = kl_from_samples(
                flat_samples["log_lag_ratio_bc_to_blr"],
                lambda x: _dist_log_prob_array(log_lag_ratio_bc_to_blr_prior(), x),
            )

    for i, band in enumerate(bands):
        if not disable_linear_trend:
            band_offset_key = f"linear_trend_band_offset_{band}"
            if band_offset_key in flat_samples:
                kls[f"{band_offset_key}_kl"] = kl_from_samples(
                    flat_samples[band_offset_key],
                    lambda x: _dist_log_prob_array(linear_trend_band_offset_prior_fn, x),
                )

        mean_key = f"mean_{band}"
        if mean_key in flat_samples:
            kls[f"{mean_key}_kl"] = kl_from_samples(
                flat_samples[mean_key],
                lambda x: _dist_log_prob_array(mean_prior_fn(), x),
            )

        band_jitter_key = f"log_jitter_{band}"
        if band_jitter_key in flat_samples:
            if log_jitter_mean_arr.ndim == 1:
                jitter_prior_mean = float(log_jitter_mean_arr[i])
            else:
                jitter_prior_mean = float(np.nanmean(log_jitter_mean_arr[i]))
            kls[f"{band_jitter_key}_kl"] = kl_from_samples(
                flat_samples[band_jitter_key],
                lambda x: _dist_log_prob_array(
                    log_jitter_prior(jitter_prior_mean),
                    x,
                ),
            )

        for j, survey in enumerate(survey_names):
            jitter_key = f"log_jitter_{band}_{survey}"
            if jitter_key in flat_samples:
                if log_jitter_mean_arr.ndim == 1:
                    jitter_prior_mean = float(log_jitter_mean_arr[i])
                else:
                    jitter_prior_mean = float(log_jitter_mean_arr[i, j])
                kls[f"{jitter_key}_kl"] = kl_from_samples(
                    flat_samples[jitter_key],
                    lambda x: _dist_log_prob_array(
                        log_jitter_prior(jitter_prior_mean),
                        x,
                    ),
                )
            survey_delta_key = f"survey_delta_mag_{band}_{survey}"
            if survey_delta_key in flat_samples:
                kls[f"{survey_delta_key}_kl"] = kl_from_samples(
                    flat_samples[survey_delta_key],
                    lambda x: _dist_log_prob_array(survey_delta_mag_prior(), x),
                )

        if disable_lag_blr:
            continue

        amp_keys = [f"dlog_amp_blr_{band}"]
        if n_blr_terms >= 2:
            amp_keys.append(f"dlog_amp_blr2_{band}")
        for amp_key in amp_keys:
            if amp_key in flat_samples:
                kls[f"{amp_key}_kl"] = kl_from_samples(
                    flat_samples[amp_key],
                    lambda x: _dist_log_prob_array(dlog_amp_blr_prior(), x),
                )

        lag_keys = [f"log_lag_blr_{band}"]
        if n_blr_terms >= 2:
            lag_keys.append(f"log_lag_blr2_{band}")
        for lag_key in lag_keys:
            if lag_key in flat_samples:
                kls[f"{lag_key}_kl"] = kl_from_samples(
                    flat_samples[lag_key],
                    lambda x: _dist_log_prob_array(log_lag_blr_prior(z=z), x),
                )

    finite_kls = [v for v in kls.values() if np.isfinite(v)]
    if finite_kls:
        kls["kl_total"] = float(np.sum(finite_kls))
    return kls


def make_lc(
    data,
    bands,
    inject_fake=False,
    alpha_sigma=-0.5,
    beta_tau=0.2,
    drop_band_lyman_alpha=False,
    verbose=True
):
    """Prepare one object's multiband time series into model-ready arrays."""

    if drop_band_lyman_alpha:
        dropped_bands = sdss_bands_affected_by_lya(data["z"]) + ["z"]
        if verbose:
            logging.info(
                f"Excluding Ly-alpha-affected bands {dropped_bands} for object {data['object_id']} at z={data['z']}"
            )
    else:
        dropped_bands = ["z"]
        if verbose:
            logging.info(
                f"Excluding default bands {dropped_bands} for object {data['object_id']} at z={data['z']}; "
                "keeping Ly-alpha-affected bands with smooth attenuation."
        )

    bands = [b for b in bands if b not in dropped_bands]
    times = data["times"]
    mags = data["mags"]
    magerrs = data["magerrs"]
    surveys = data.get("surveys", {})

    if len(bands) == 0:
        print(f"No usable bands for {data['object_id']}, skipping.", flush=True)
        return None

    all_times = np.concatenate([np.asarray(times[b]) for b in bands])
    all_mags = np.concatenate([np.asarray(mags[b]) for b in bands])
    all_magerrs = np.concatenate([np.asarray(magerrs[b]) for b in bands])
    all_surveys = np.concatenate(
        [
            np.asarray(
                surveys.get(b, _default_survey_labels_for_band(b, len(times[b]))),
                dtype=str,
            )
            for b in bands
        ]
    )
    band_idx = np.concatenate([np.full(len(times[b]), i) for i, b in enumerate(bands)]).astype(
        np.int64, copy=False
    )

    if len(all_times) == 0:
        print(f"No points for {data['object_id']}, skipping.", flush=True)
        return None

    tie_eps = 10.0 * np.finfo(all_times.dtype).eps
    key = all_times + band_idx.astype(all_times.dtype) * tie_eps
    order = np.argsort(key, kind="mergesort")
    all_times, all_mags, all_magerrs, all_surveys, band_idx = (
        all_times[order],
        all_mags[order],
        all_magerrs[order],
        all_surveys[order],
        band_idx[order],
    )

    if inject_fake:
        lam_rf_bands = (
            np.asarray([lambda_pivot[band] for band in bands], dtype=float) / (1.0 + float(data["z"]))
        )
        lam_ref = 2500.0

        key = random.PRNGKey(0)
        key = random.fold_in(key, int(data["object_id"]))
        key, k_tau0, k_sig0, k_eps, k_y0, k_noise = random.split(key, 6)

        one_plus_z = float(1.0 + data["z"])
        log_tau0_obs_min = 1.8
        log_tau0_rf_min = log_tau0_obs_min - np.log10(one_plus_z)
        log_tau0_rf = random.uniform(k_tau0, minval=log_tau0_rf_min, maxval=3.0)
        log_sigma0 = random.uniform(k_sig0, minval=-0.5, maxval=1.0)
        tau0_rf = 10.0 ** float(log_tau0_rf)
        sigma0 = 10.0 ** float(log_sigma0)

        tau_rf_band = tau0_rf * (lam_rf_bands / lam_ref) ** beta_tau
        tau_obs_band = tau_rf_band * one_plus_z
        sigma_band = sigma0 * (lam_rf_bands / lam_ref) ** alpha_sigma

        global_times = np.unique(np.asarray(all_times, dtype=float))
        dt_g = np.diff(global_times, prepend=global_times[0])
        eps_g = np.array(random.normal(k_eps, shape=(global_times.size,)))
        pos_on_global = np.searchsorted(global_times, np.asarray(all_times, dtype=float))

        uniq_bands = np.unique(band_idx)
        noise_keys = random.split(k_noise, len(uniq_bands))
        init_keys = random.split(k_y0, len(uniq_bands))

        for bk, ik, b in zip(noise_keys, init_keys, uniq_bands):
            b = int(b)
            mask = band_idx == b
            idx_band = np.nonzero(mask)[0]
            pos_b = pos_on_global[mask]

            tau_b = float(tau_obs_band[b])
            sigma_b = float(sigma_band[b])

            y_g = np.empty_like(global_times)
            y_g[0] = float(random.normal(ik)) * sigma_b
            for k in range(1, global_times.size):
                a = np.exp(-dt_g[k] / tau_b)
                q = (sigma_b**2) * (1.0 - np.exp(-2.0 * dt_g[k] / tau_b))
                y_g[k] = a * y_g[k - 1] + np.sqrt(q) * eps_g[k]

            e_b = np.asarray(all_magerrs[idx_band], dtype=float)
            eta = np.array(random.normal(bk, shape=(pos_b.size,)))
            y_b = y_g[pos_b] + eta * e_b
            all_mags[idx_band] = y_b

    mfin = np.isfinite(all_mags) & np.isfinite(all_magerrs) & np.isfinite(all_times)
    all_times, all_mags, all_magerrs, all_surveys, band_idx = (
        all_times[mfin],
        all_mags[mfin],
        all_magerrs[mfin],
        all_surveys[mfin],
        band_idx[mfin],
    )
    if len(all_times) == 0:
        print(f"No finite values for {data['object_id']}, skipping.", flush=True)
        return None

    window_size = 6
    keep = np.ones(len(all_times), dtype=bool)
    for b in np.unique(band_idx):
        mask = band_idx == b
        yb = all_mags[mask]
        idx_b = np.where(mask)[0]
        if len(yb) < 2 * window_size + 1:
            continue
        windows = sliding_window_view(yb, 2 * window_size + 1)
        centers = yb[window_size:-window_size]
        medians = np.nanmean(windows, axis=1)
        mads = median_abs_deviation(windows, axis=1, nan_policy="omit")
        is_out = np.abs(centers - medians) > 2.5 * mads
        keep[idx_b[window_size:-window_size][is_out]] = False

    all_times, all_mags, all_magerrs, all_surveys, band_idx = (
        all_times[keep],
        all_mags[keep],
        all_magerrs[keep],
        all_surveys[keep],
        band_idx[keep],
    )

    if len(all_times) == 0:
        print(f"All points excluded for {data['object_id']}, skipping.", flush=True)
        return None

    t_obs_length = float(np.max(all_times) - np.min(all_times))
    t_rf_length = float(t_obs_length / (1.0 + data["z"]))
    if verbose:
        print(f"[{data['object_id']}] Δt_obs={t_obs_length:.2f} d, Δt_rf={t_rf_length:.2f} d")

    B = len(bands)
    mags_means = np.empty(B)
    mags_stds = np.empty(B)
    for i in range(B):
        m = band_idx == i
        mu = np.nanmean(all_mags[m]) if np.any(m) else np.nan
        sd = np.nanstd(all_mags[m]) if np.any(m) else np.nan
        mags_means[i] = mu
        mags_stds[i] = sd
        if np.any(m):
            all_mags[m] = all_mags[m] - mu

    time0 = np.min(all_times)
    survey_idx, survey_labels = _survey_indices_from_labels(all_surveys)
    X = (jnp.array(all_times) - jnp.min(all_times), jnp.array(band_idx))
    y = jnp.array(all_mags)
    yerr = jnp.array(all_magerrs)

    out = {
        "X": X,
        "time0": time0,
        "y": y,
        "yerr": yerr,
        "band_idx": band_idx,
        "survey_idx": survey_idx,
        "survey_labels": survey_labels,
        "survey_names": LC_SURVEY_NAMES,
        "z": data["z"],
        "mags_means": mags_means,
        "mags_stds": mags_stds,
        "dropped_bands": dropped_bands,
        "t_obs_length": t_obs_length,
        "t_rf_length": t_rf_length,
        "bands": bands,
        "cadence": data["cadence"],
        "cadence_err": data["cadence_err"],
        "number_points": data["number_points"],
    }
    out.update(compute_variability_metrics_for_cleaned_lc(out))

    if inject_fake:
        out["log_tau_fake"] = float(np.log(10 ** float(log_tau0_rf) * (1 + data["z"])))
        out["log_sigma_fake"] = float(np.log(10 ** float(log_sigma0)))
        out["alpha_sigma"] = float(alpha_sigma)
        out["beta_tau"] = float(beta_tau)
    else:
        out["log_tau_fake"] = -99.0
        out["log_sigma_fake"] = -99.0

    return out


def build_explicit_model_params(raw_params, lam_rf, *, lam_lya_rf=None):
    """Convert sampled high-level parameters into explicit model arrays."""

    lam_rf = jnp.asarray(lam_rf)
    if lam_lya_rf is None:
        lam_lya_rf = raw_params.get("lam_lya_rf", lam_rf)
    lam_lya_rf = jnp.asarray(lam_lya_rf, dtype=lam_rf.dtype)
    lambda_uv = jnp.array(2500.0, dtype=lam_rf.dtype)
    lambda_center_rf = jnp.asarray(
        raw_params.get("lambda_center_rf", compute_lambda_center_rf(lam_rf))
    )

    eta_sigma = jnp.asarray(raw_params["eta_sigma"])
    eta_tau = jnp.asarray(raw_params["eta_tau"])
    dlog_amp_blr = jnp.asarray(raw_params["dlog_amp_blr"])
    dlog_amp_blr2 = jnp.asarray(
        raw_params.get(
            "dlog_amp_blr2",
            jnp.full_like(dlog_amp_blr, -1e9),
        )
    )
    log_igm_transmission_band = raw_params.get("log_igm_transmission_band")
    if log_igm_transmission_band is None:
        log_igm_transmission_band = jnp.zeros_like(lam_rf, dtype=lam_rf.dtype)
    log_igm_transmission_band = _coerce_band_array(log_igm_transmission_band, lam_rf)
    log_lag_blr = jnp.asarray(raw_params["log_lag_blr"])
    log_lag_blr2 = jnp.asarray(raw_params.get("log_lag_blr2", log_lag_blr))
    has_bc_lag = "dlog_amp_bc" in raw_params
    if has_bc_lag:
        dlog_amp_bc = jnp.asarray(raw_params["dlog_amp_bc"])
        log_lag_ratio_bc_to_blr = jnp.asarray(
            raw_params.get("log_lag_ratio_bc_to_blr", jnp.log(0.2))
        )
    lag0 = jnp.asarray(raw_params["lag0"])
    lag_beta = jnp.asarray(raw_params["lag_beta"])
    sigma_shift_to_uv = jnp.log(10.0) * log_single_pl(lambda_uv, lambda_center_rf, eta_sigma)
    tau_shift_to_uv = jnp.log(10.0) * log_single_pl(lambda_uv, lambda_center_rf, eta_tau)

    if "log_sigma_center0" in raw_params:
        log_sigma_center0 = jnp.asarray(raw_params["log_sigma_center0"])
        log_sigma_uv = log_sigma_center0 + sigma_shift_to_uv
    else:
        log_sigma_uv = jnp.asarray(raw_params["log_sigma_uv"])
        log_sigma_center0 = log_sigma_uv - sigma_shift_to_uv

    if "log_tau_slow_center0" in raw_params:
        log_tau_slow_center0 = jnp.asarray(raw_params["log_tau_slow_center0"])
        log_tau_uv = log_tau_slow_center0 + tau_shift_to_uv
    else:
        log_tau_uv = jnp.asarray(raw_params["log_tau_uv"])
        log_tau_slow_center0 = log_tau_uv - tau_shift_to_uv

    if "log_tau_fast_center0" in raw_params:
        log_tau_fast_center0 = jnp.asarray(raw_params["log_tau_fast_center0"])
        log_tau_fast_uv = log_tau_fast_center0 + tau_shift_to_uv
    else:
        log_tau_fast_uv = jnp.asarray(raw_params["log_tau_fast_uv"])
        log_tau_fast_center0 = log_tau_fast_uv - tau_shift_to_uv

    log_sigma_uv_exp = _expand_last(log_sigma_uv)
    log_sigma_center0_exp = _expand_last(log_sigma_center0)
    eta_sigma_exp = _expand_last(eta_sigma)
    eta_tau_exp = _expand_last(eta_tau)
    lag0_exp = _expand_last(lag0)
    lag_beta_exp = _expand_last(lag_beta)
    log_tau_fast_center0_exp = _expand_last(log_tau_fast_center0)
    log_tau_slow_center0_exp = _expand_last(log_tau_slow_center0)
    lambda_center_rf_exp = _expand_last(lambda_center_rf)

    log_sigma_band = log_sigma_center0_exp + jnp.log(10.0) * log_single_pl(
        lam_rf,
        lambda_center_rf_exp,
        eta_sigma_exp,
    )
    bc_weight = balmer_continuum_weight(lam_rf)

    # Static multiplicative IGM absorption shifts the band mean, not mag residual amplitudes.
    amp_cont = jnp.exp(log_sigma_band)
    amp_blr = jnp.exp(log_sigma_uv_exp + dlog_amp_blr)
    amp_blr2 = jnp.exp(log_sigma_uv_exp + dlog_amp_blr2)
    lag_disk = lag0_exp * (lam_rf / lambda_center_rf_exp) ** lag_beta_exp
    lag_blr = jnp.exp(log_lag_blr)
    lag_blr2 = jnp.exp(log_lag_blr2)
    if has_bc_lag:
        dlog_amp_bc_exp = _expand_last(dlog_amp_bc)
        log_lag_bc_shared = jnp.mean(log_lag_blr, axis=-1) + jnp.asarray(log_lag_ratio_bc_to_blr)
        amp_bc = jnp.exp(log_sigma_uv_exp + dlog_amp_bc_exp) * bc_weight
        lag_bc = jnp.broadcast_to(
            _expand_last(jnp.exp(log_lag_bc_shared)),
            lag_blr.shape,
        )
    else:
        amp_bc = jnp.zeros_like(amp_cont)
        lag_bc = jnp.zeros_like(lag_blr)
    log_tau_scale = jnp.log(10.0) * log_single_pl(
        lam_rf,
        lambda_center_rf_exp,
        eta_tau_exp,
    )
    log_tau_fast_band = log_tau_fast_center0_exp + log_tau_scale
    log_tau_slow_band = log_tau_slow_center0_exp + log_tau_scale
    log_kernel_param = jnp.concatenate([log_tau_fast_band, log_tau_slow_band], axis=-1)

    explicit = dict(raw_params)
    explicit["lambda_center_rf"] = lambda_center_rf
    explicit["log_sigma_center0"] = log_sigma_center0
    explicit["log_tau_slow_center0"] = log_tau_slow_center0
    explicit["log_tau_fast_center0"] = log_tau_fast_center0
    explicit["log_sigma_uv"] = log_sigma_uv
    explicit["log_tau_uv"] = log_tau_uv
    explicit["log_tau_fast_uv"] = log_tau_fast_uv
    explicit["log_igm_transmission_band"] = log_igm_transmission_band
    explicit["igm_transmission_band"] = jnp.exp(log_igm_transmission_band)
    explicit["bc_weight"] = bc_weight
    explicit["amp_cont"] = amp_cont
    explicit["amp_bc"] = amp_bc
    explicit["amp_blr"] = amp_blr
    explicit["amp_blr2"] = amp_blr2
    explicit["lag_disk"] = lag_disk
    explicit["lag_bc"] = lag_bc
    explicit["lag_blr"] = lag_blr
    explicit["lag_blr2"] = lag_blr2
    explicit["tau_fast_band"] = jnp.exp(log_tau_fast_band)
    explicit["tau_slow_band"] = jnp.exp(log_tau_slow_band)
    explicit["log_kernel_param"] = log_kernel_param
    if has_bc_lag:
        explicit["dlog_amp_bc"] = dlog_amp_bc
        explicit["log_lag_ratio_bc_to_blr"] = log_lag_ratio_bc_to_blr
    return explicit


def build_explicit_model_params_relflux(raw_params, lam_rf, *, lam_lya_rf=None):
    """Convert sampled relative-flux parameters into internal and legacy arrays."""

    lam_rf = jnp.asarray(lam_rf)
    if lam_lya_rf is None:
        lam_lya_rf = raw_params.get("lam_lya_rf", lam_rf)
    lam_lya_rf = jnp.asarray(lam_lya_rf, dtype=lam_rf.dtype)
    lambda_uv = jnp.array(2500.0, dtype=lam_rf.dtype)
    lambda_center_rf = jnp.asarray(
        raw_params.get("lambda_center_rf", compute_lambda_center_rf(lam_rf))
    )

    eta_sigma = jnp.asarray(raw_params["eta_sigma"])
    eta_tau = jnp.asarray(raw_params["eta_tau"])
    dlog_amp_blr = jnp.asarray(raw_params["dlog_amp_blr"])
    dlog_amp_blr2 = jnp.asarray(
        raw_params.get(
            "dlog_amp_blr2",
            jnp.full_like(dlog_amp_blr, -1e9),
        )
    )
    log_igm_transmission_band = raw_params.get("log_igm_transmission_band")
    if log_igm_transmission_band is None:
        log_igm_transmission_band = jnp.zeros_like(lam_rf, dtype=lam_rf.dtype)
    log_igm_transmission_band = _coerce_band_array(log_igm_transmission_band, lam_rf)
    log_lag_blr = jnp.asarray(raw_params["log_lag_blr"])
    log_lag_blr2 = jnp.asarray(raw_params.get("log_lag_blr2", log_lag_blr))
    has_bc_lag = "dlog_amp_bc" in raw_params
    if has_bc_lag:
        dlog_amp_bc = jnp.asarray(raw_params["dlog_amp_bc"])
        log_lag_ratio_bc_to_blr = jnp.asarray(
            raw_params.get("log_lag_ratio_bc_to_blr", jnp.log(0.2))
        )
    lag0 = jnp.asarray(raw_params["lag0"])
    lag_beta = jnp.asarray(raw_params["lag_beta"])
    sigma_shift = jnp.log(10.0) * log_single_pl(lambda_uv, lambda_center_rf, eta_sigma)
    tau_shift = jnp.log(10.0) * log_single_pl(lambda_uv, lambda_center_rf, eta_tau)

    if "log_sigma_center0_relflux" in raw_params:
        log_sigma_center0_relflux = jnp.asarray(raw_params["log_sigma_center0_relflux"])
        log_sigma_uv_relflux = log_sigma_center0_relflux + sigma_shift
    elif "log_sigma_center0" in raw_params:
        log_sigma_center0_relflux = jnp.asarray(raw_params["log_sigma_center0"])
        log_sigma_uv_relflux = log_sigma_center0_relflux + sigma_shift
    else:
        log_sigma_uv_relflux = jnp.asarray(
            raw_params.get("log_sigma_uv_relflux", raw_params["log_sigma_uv"])
        )
        log_sigma_center0_relflux = log_sigma_uv_relflux - sigma_shift

    if "log_tau_slow_center0" in raw_params:
        log_tau_slow_center0 = jnp.asarray(raw_params["log_tau_slow_center0"])
        log_tau_uv = log_tau_slow_center0 + tau_shift
    else:
        log_tau_uv = jnp.asarray(raw_params["log_tau_uv"])
        log_tau_slow_center0 = log_tau_uv - tau_shift

    if "log_tau_fast_center0" in raw_params:
        log_tau_fast_center0 = jnp.asarray(raw_params["log_tau_fast_center0"])
        log_tau_fast_uv = log_tau_fast_center0 + tau_shift
    else:
        log_tau_fast_uv = jnp.asarray(raw_params["log_tau_fast_uv"])
        log_tau_fast_center0 = log_tau_fast_uv - tau_shift

    log_sigma_uv_relflux_exp = _expand_last(log_sigma_uv_relflux)
    log_sigma_center0_relflux_exp = _expand_last(log_sigma_center0_relflux)
    eta_sigma_exp = _expand_last(eta_sigma)
    eta_tau_exp = _expand_last(eta_tau)
    lag0_exp = _expand_last(lag0)
    lag_beta_exp = _expand_last(lag_beta)
    log_tau_fast_center0_exp = _expand_last(log_tau_fast_center0)
    log_tau_slow_center0_exp = _expand_last(log_tau_slow_center0)
    lambda_center_rf_exp = _expand_last(lambda_center_rf)

    log_sigma_band_relflux = log_sigma_center0_relflux_exp + jnp.log(10.0) * log_single_pl(
        lam_rf,
        lambda_center_rf_exp,
        eta_sigma_exp,
    )
    bc_weight = balmer_continuum_weight(lam_rf)

    # Residual relative flux is measured around the observed mean, so fixed IGM cancels.
    amp_cont_relflux = jnp.exp(log_sigma_band_relflux)
    amp_blr_relflux = jnp.exp(log_sigma_uv_relflux_exp + dlog_amp_blr)
    amp_blr2_relflux = jnp.exp(log_sigma_uv_relflux_exp + dlog_amp_blr2)
    lag_disk = lag0_exp * (lam_rf / lambda_center_rf_exp) ** lag_beta_exp
    lag_blr = jnp.exp(log_lag_blr)
    lag_blr2 = jnp.exp(log_lag_blr2)
    if has_bc_lag:
        dlog_amp_bc_exp = _expand_last(dlog_amp_bc)
        log_lag_bc_shared = jnp.mean(log_lag_blr, axis=-1) + jnp.asarray(log_lag_ratio_bc_to_blr)
        amp_bc_relflux = jnp.exp(log_sigma_uv_relflux_exp + dlog_amp_bc_exp) * bc_weight
        lag_bc = jnp.broadcast_to(
            _expand_last(jnp.exp(log_lag_bc_shared)),
            lag_blr.shape,
        )
    else:
        amp_bc_relflux = jnp.zeros_like(amp_cont_relflux)
        lag_bc = jnp.zeros_like(lag_blr)
    log_tau_scale = jnp.log(10.0) * log_single_pl(
        lam_rf,
        lambda_center_rf_exp,
        eta_tau_exp,
    )
    log_tau_fast_band = log_tau_fast_center0_exp + log_tau_scale
    log_tau_slow_band = log_tau_slow_center0_exp + log_tau_scale
    log_kernel_param = jnp.concatenate([log_tau_fast_band, log_tau_slow_band], axis=-1)

    scale = jnp.asarray(RELFLUX_TO_MAG_SCALE, dtype=lam_rf.dtype)
    log_sigma_center0 = log_sigma_center0_relflux + LOG_RELFLUX_TO_MAG_SCALE
    log_sigma_uv = log_sigma_uv_relflux + LOG_RELFLUX_TO_MAG_SCALE
    amp_cont = scale * amp_cont_relflux
    amp_bc = scale * amp_bc_relflux
    amp_blr = scale * amp_blr_relflux
    amp_blr2 = scale * amp_blr2_relflux

    explicit = dict(raw_params)
    explicit["lambda_center_rf"] = lambda_center_rf
    explicit["log_sigma_center0_relflux"] = log_sigma_center0_relflux
    explicit["log_sigma_uv_relflux"] = log_sigma_uv_relflux
    explicit["log_sigma_center0"] = log_sigma_center0
    explicit["log_tau_slow_center0"] = log_tau_slow_center0
    explicit["log_tau_fast_center0"] = log_tau_fast_center0
    explicit["log_sigma_uv"] = log_sigma_uv
    explicit["log_tau_uv"] = log_tau_uv
    explicit["log_tau_fast_uv"] = log_tau_fast_uv
    explicit["log_igm_transmission_band"] = log_igm_transmission_band
    explicit["igm_transmission_band"] = jnp.exp(log_igm_transmission_band)
    explicit["bc_weight"] = bc_weight
    explicit["amp_cont_relflux"] = amp_cont_relflux
    explicit["amp_bc_relflux"] = amp_bc_relflux
    explicit["amp_blr_relflux"] = amp_blr_relflux
    explicit["amp_blr2_relflux"] = amp_blr2_relflux
    explicit["amp_cont"] = amp_cont
    explicit["amp_bc"] = amp_bc
    explicit["amp_blr"] = amp_blr
    explicit["amp_blr2"] = amp_blr2
    explicit["lag_disk"] = lag_disk
    explicit["lag_bc"] = lag_bc
    explicit["lag_blr"] = lag_blr
    explicit["lag_blr2"] = lag_blr2
    explicit["tau_fast_band"] = jnp.exp(log_tau_fast_band)
    explicit["tau_slow_band"] = jnp.exp(log_tau_slow_band)
    explicit["log_kernel_param"] = log_kernel_param
    if has_bc_lag:
        explicit["dlog_amp_bc"] = dlog_amp_bc
        explicit["log_lag_ratio_bc_to_blr"] = log_lag_ratio_bc_to_blr
    return explicit


def build_explicit_model_params_continuum_only(raw_params, lam_rf, *, lam_lya_rf=None):
    """Explicit continuum-only parameters with line amplitudes forced to zero."""

    lam_rf = jnp.asarray(lam_rf)
    B = int(lam_rf.shape[0])
    raw = dict(raw_params)
    zeros = jnp.zeros(B, dtype=lam_rf.dtype)
    raw.setdefault("dlog_amp_blr", jnp.full(B, -9.0, dtype=lam_rf.dtype))
    raw.setdefault("dlog_amp_blr2", jnp.full(B, -9.0, dtype=lam_rf.dtype))
    raw.setdefault("log_lag_blr", jnp.full(B, -9.0, dtype=lam_rf.dtype))
    raw.setdefault("log_lag_blr2", jnp.full(B, -9.0, dtype=lam_rf.dtype))
    explicit = build_explicit_model_params(raw, lam_rf, lam_lya_rf=lam_lya_rf)
    explicit["amp_bc"] = zeros
    explicit["amp_blr"] = zeros
    explicit["amp_blr2"] = zeros
    explicit["lag_bc"] = zeros
    explicit["lag_blr"] = zeros
    explicit["lag_blr2"] = zeros
    return explicit


def build_explicit_model_params_fluxmix_fast(raw_params, lam_rf, *, lam_lya_rf=None):
    """Explicit parameters for the two-stage flux-mixing approximation."""

    explicit = build_explicit_model_params(raw_params, lam_rf, lam_lya_rf=lam_lya_rf)
    explicit["log_cont_scale"] = jnp.asarray(raw_params.get("log_cont_scale", 0.0))
    return explicit


def add_model_prediction_params(samples, lam_rf, *, model_variant=None, lam_lya_rf=None):
    """Add explicit model parameters needed for prediction/plotting."""

    out = dict(samples)
    use_fluxmix = model_variant == "mag_fluxmix_fast" or "log_cont_scale" in out
    use_relflux = (
        model_variant in ("mag_flux_linearized", "mag_flux_linearized_erlang")
        or "log_sigma_uv_relflux" in out
        or "amp_cont_relflux" in out
    )
    if use_fluxmix and all(
        key in out
        for key in (
            "log_kernel_param",
            "amp_cont",
            "amp_bc",
            "amp_blr",
            "amp_blr2",
            "lag_disk",
            "lag_bc",
            "lag_blr",
            "lag_blr2",
            "tau_fast_band",
            "tau_slow_band",
            "log_sigma_uv",
            "log_tau_uv",
            "log_tau_fast_uv",
            "log_igm_transmission_band",
            "igm_transmission_band",
            "lambda_center_rf",
        )
    ):
        return out
    if use_fluxmix:
        explicit = build_explicit_model_params_fluxmix_fast(
            out,
            lam_rf,
            lam_lya_rf=lam_lya_rf,
        )
        out["log_kernel_param"] = np.asarray(explicit["log_kernel_param"])
        out["amp_cont"] = np.asarray(explicit["amp_cont"])
        out["amp_bc"] = np.asarray(explicit["amp_bc"])
        out["amp_blr"] = np.asarray(explicit["amp_blr"])
        out["amp_blr2"] = np.asarray(explicit["amp_blr2"])
        out["lag_disk"] = np.asarray(explicit["lag_disk"])
        out["lag_bc"] = np.asarray(explicit["lag_bc"])
        out["lag_blr"] = np.asarray(explicit["lag_blr"])
        out["lag_blr2"] = np.asarray(explicit["lag_blr2"])
        out["tau_fast_band"] = np.asarray(explicit["tau_fast_band"])
        out["tau_slow_band"] = np.asarray(explicit["tau_slow_band"])
        out["log_sigma_center0"] = np.asarray(explicit["log_sigma_center0"])
        out["log_tau_slow_center0"] = np.asarray(explicit["log_tau_slow_center0"])
        out["log_tau_fast_center0"] = np.asarray(explicit["log_tau_fast_center0"])
        out["log_sigma_uv"] = np.asarray(explicit["log_sigma_uv"])
        out["log_tau_uv"] = np.asarray(explicit["log_tau_uv"])
        out["log_tau_fast_uv"] = np.asarray(explicit["log_tau_fast_uv"])
        out["log_igm_transmission_band"] = np.asarray(explicit["log_igm_transmission_band"])
        out["igm_transmission_band"] = np.asarray(explicit["igm_transmission_band"])
        out["log_cont_scale"] = np.asarray(explicit["log_cont_scale"])
        if "dlog_amp_bc" in explicit:
            out["dlog_amp_bc"] = np.asarray(explicit["dlog_amp_bc"])
        if "log_lag_ratio_bc_to_blr" in explicit:
            out["log_lag_ratio_bc_to_blr"] = np.asarray(explicit["log_lag_ratio_bc_to_blr"])
        if "dlog_amp_blr2" in explicit:
            out["dlog_amp_blr2"] = np.asarray(explicit["dlog_amp_blr2"])
        if "log_lag_blr2" in explicit:
            out["log_lag_blr2"] = np.asarray(explicit["log_lag_blr2"])
        out["lambda_center_rf"] = np.asarray(explicit["lambda_center_rf"])
        return out
    if use_relflux and all(
        key in out
        for key in (
            "log_kernel_param",
            "amp_cont_relflux",
            "amp_bc_relflux",
            "amp_blr_relflux",
            "amp_blr2_relflux",
            "amp_cont",
            "amp_bc",
            "amp_blr",
            "amp_blr2",
            "lag_disk",
            "lag_bc",
            "lag_blr",
            "lag_blr2",
            "tau_fast_band",
            "tau_slow_band",
            "log_sigma_uv_relflux",
            "log_sigma_uv",
            "log_tau_uv",
            "log_tau_fast_uv",
            "log_igm_transmission_band",
            "igm_transmission_band",
            "lambda_center_rf",
        )
    ):
        return out
    if use_relflux:
        explicit = build_explicit_model_params_relflux(
            out,
            lam_rf,
            lam_lya_rf=lam_lya_rf,
        )
        out["log_kernel_param"] = np.asarray(explicit["log_kernel_param"])
        out["amp_cont_relflux"] = np.asarray(explicit["amp_cont_relflux"])
        out["amp_bc_relflux"] = np.asarray(explicit["amp_bc_relflux"])
        out["amp_blr_relflux"] = np.asarray(explicit["amp_blr_relflux"])
        out["amp_blr2_relflux"] = np.asarray(explicit["amp_blr2_relflux"])
        out["amp_cont"] = np.asarray(explicit["amp_cont"])
        out["amp_bc"] = np.asarray(explicit["amp_bc"])
        out["amp_blr"] = np.asarray(explicit["amp_blr"])
        out["amp_blr2"] = np.asarray(explicit["amp_blr2"])
        out["lag_disk"] = np.asarray(explicit["lag_disk"])
        out["lag_bc"] = np.asarray(explicit["lag_bc"])
        out["lag_blr"] = np.asarray(explicit["lag_blr"])
        out["lag_blr2"] = np.asarray(explicit["lag_blr2"])
        out["tau_fast_band"] = np.asarray(explicit["tau_fast_band"])
        out["tau_slow_band"] = np.asarray(explicit["tau_slow_band"])
        out["log_sigma_center0_relflux"] = np.asarray(explicit["log_sigma_center0_relflux"])
        out.setdefault("log_sigma_center0", np.asarray(explicit["log_sigma_center0_relflux"]))
        out["log_sigma_center0_mag_equiv"] = np.asarray(explicit["log_sigma_center0"])
        out["log_tau_slow_center0"] = np.asarray(explicit["log_tau_slow_center0"])
        out["log_tau_fast_center0"] = np.asarray(explicit["log_tau_fast_center0"])
        out["log_sigma_uv_relflux"] = np.asarray(explicit["log_sigma_uv_relflux"])
        out["log_sigma_uv"] = np.asarray(explicit["log_sigma_uv"])
        out["log_tau_uv"] = np.asarray(explicit["log_tau_uv"])
        out["log_tau_fast_uv"] = np.asarray(explicit["log_tau_fast_uv"])
        out["log_igm_transmission_band"] = np.asarray(explicit["log_igm_transmission_band"])
        out["igm_transmission_band"] = np.asarray(explicit["igm_transmission_band"])
        if "dlog_amp_bc" in explicit:
            out["dlog_amp_bc"] = np.asarray(explicit["dlog_amp_bc"])
        if "log_lag_ratio_bc_to_blr" in explicit:
            out["log_lag_ratio_bc_to_blr"] = np.asarray(explicit["log_lag_ratio_bc_to_blr"])
        if "dlog_amp_blr2" in explicit:
            out["dlog_amp_blr2"] = np.asarray(explicit["dlog_amp_blr2"])
        if "log_lag_blr2" in explicit:
            out["log_lag_blr2"] = np.asarray(explicit["log_lag_blr2"])
        out["lambda_center_rf"] = np.asarray(explicit["lambda_center_rf"])
        return out

    if all(
        key in out
        for key in (
            "log_kernel_param",
            "amp_cont",
            "amp_bc",
            "amp_blr",
            "amp_blr2",
            "lag_disk",
            "lag_bc",
            "lag_blr",
            "lag_blr2",
            "tau_fast_band",
            "tau_slow_band",
            "log_sigma_uv",
            "log_tau_uv",
            "log_tau_fast_uv",
            "log_igm_transmission_band",
            "igm_transmission_band",
            "lambda_center_rf",
        )
    ):
        return out

    explicit = build_explicit_model_params(
        out,
        lam_rf,
        lam_lya_rf=lam_lya_rf,
    )
    out["log_kernel_param"] = np.asarray(explicit["log_kernel_param"])
    out["amp_cont"] = np.asarray(explicit["amp_cont"])
    out["amp_bc"] = np.asarray(explicit["amp_bc"])
    out["amp_blr"] = np.asarray(explicit["amp_blr"])
    out["amp_blr2"] = np.asarray(explicit["amp_blr2"])
    out["lag_disk"] = np.asarray(explicit["lag_disk"])
    out["lag_bc"] = np.asarray(explicit["lag_bc"])
    out["lag_blr"] = np.asarray(explicit["lag_blr"])
    out["lag_blr2"] = np.asarray(explicit["lag_blr2"])
    out["tau_fast_band"] = np.asarray(explicit["tau_fast_band"])
    out["tau_slow_band"] = np.asarray(explicit["tau_slow_band"])
    out["log_sigma_center0"] = np.asarray(explicit["log_sigma_center0"])
    out["log_tau_slow_center0"] = np.asarray(explicit["log_tau_slow_center0"])
    out["log_tau_fast_center0"] = np.asarray(explicit["log_tau_fast_center0"])
    out["log_sigma_uv"] = np.asarray(explicit["log_sigma_uv"])
    out["log_tau_uv"] = np.asarray(explicit["log_tau_uv"])
    out["log_tau_fast_uv"] = np.asarray(explicit["log_tau_fast_uv"])
    out["log_igm_transmission_band"] = np.asarray(explicit["log_igm_transmission_band"])
    out["igm_transmission_band"] = np.asarray(explicit["igm_transmission_band"])
    if "dlog_amp_bc" in explicit:
        out["dlog_amp_bc"] = np.asarray(explicit["dlog_amp_bc"])
    if "log_lag_ratio_bc_to_blr" in explicit:
        out["log_lag_ratio_bc_to_blr"] = np.asarray(explicit["log_lag_ratio_bc_to_blr"])
    if "dlog_amp_blr2" in explicit:
        out["dlog_amp_blr2"] = np.asarray(explicit["dlog_amp_blr2"])
    if "log_lag_blr2" in explicit:
        out["log_lag_blr2"] = np.asarray(explicit["log_lag_blr2"])
    out["lambda_center_rf"] = np.asarray(explicit["lambda_center_rf"])
    return out


def build_single_object_model(
    obj_dict,
    lam_rf,
    log_jitter_mean,
    *,
    lam_lya_rf=None,
    disable_linear_trend=False,
    disable_lag_blr=False,
    disable_lag_bc=False,
    drop_band_lyman_alpha=False,
    tau_fast_truncated=False,
    n_blr_terms=1,
    shared_blr_lag=False,
    blr_lag_band_scatter=None,
):
    """Return the NumPyro model for one object."""

    (t, bidx) = obj_dict["X"]
    y = obj_dict["y"]
    yerr = obj_dict["yerr"]
    survey_idx = jnp.asarray(obj_dict["survey_idx"], dtype=jnp.int32)
    z = float(obj_dict["z"])
    B = int(len(lam_rf))
    lambda_center_rf = compute_lambda_center_rf(lam_rf)
    if lam_lya_rf is None:
        lam_lya_rf = lam_rf
    lam_lya_rf = jnp.asarray(lam_lya_rf, dtype=lam_rf.dtype)
    bands = tuple(str(b) for b in obj_dict.get("bands", []))
    if bands:
        log_igm_transmission_band = compute_log_igm_transmission_band(bands, z)
    else:
        log_igm_transmission_band = jnp.zeros(B, dtype=lam_rf.dtype)
    lambda_uv = jnp.array(2500.0, dtype=lam_rf.dtype)
    log_jitter_active_mask, survey_offset_active_mask = _get_object_active_noise_calibration_masks(
        obj_dict, B
    )
    log_jitter_mean_grid = _coerce_log_jitter_mean_grid(log_jitter_mean, B)

    def model():
        eta_sigma = numpyro.sample("eta_sigma", eta_sigma_prior())

        eta_tau = numpyro.sample("eta_tau", eta_tau_prior())

        log_tau_slow_center0 = numpyro.sample(
            "log_tau_slow_center0",
            log_tau_slow_center0_prior(
                eta_tau,
                z,
                lambda_center_rf,
            ),
        )

        log_tau_fast_center0 = numpyro.sample(
            "log_tau_fast_center0",
            log_tau_fast_center0_prior(
                log_tau_slow_center0,
                tau_fast_truncated=tau_fast_truncated,
            ),
        )

        log_sigma_center0 = numpyro.sample(
            "log_sigma_center0",
            log_sigma_center0_prior(
                eta_sigma,
                lambda_center_rf,
            ),
        )
        (
            linear_trend,
            linear_trend_band_offset,
            _linear_trend_band,
        ) = sample_linear_trend_with_band_offsets(
            B=B,
            disable_linear_trend=disable_linear_trend,
            trend_prior_dist=linear_trend_prior(t_ref=t, z=z),
            band_offset_raw_prior_dist=linear_trend_band_offset_raw_prior(),
        )

        lag0 = numpyro.sample("lag0", lag0_prior(z=z))
        log_lag0 = numpyro.deterministic("log_lag0", jnp.log(lag0))
        lag_beta = numpyro.sample("lag_beta", lag_beta_prior())

        line_ratio_offsets = compute_flux_line_ratio_offsets(
            lam_rf,
            lambda_center_rf=lambda_center_rf,
            eta_sigma=eta_sigma,
            log_igm_transmission_band=log_igm_transmission_band,
        )
        (
            mean,
            dlog_amp_blr,
            dlog_amp_blr2,
            log_lag_blr,
            log_lag_blr2,
            log_jitter,
            survey_delta_mag,
            dlog_amp_bc,
            log_lag_ratio_bc_to_blr,
        ) = sample_flux_line_latent_params(
            B=B,
            z=z,
            log_lag0=log_lag0,
            log_jitter_mean=log_jitter_mean_grid,
            log_jitter_active_mask=log_jitter_active_mask,
            survey_offset_active_mask=survey_offset_active_mask,
            line_ratio_offsets=line_ratio_offsets,
            disable_lag_blr=disable_lag_blr,
            disable_lag_bc=disable_lag_bc,
            n_blr_terms=n_blr_terms,
            shared_blr_lag=shared_blr_lag,
            blr_lag_band_scatter=blr_lag_band_scatter,
        )

        _ = numpyro.deterministic("log_tau_fake", float(obj_dict.get("log_tau_fake", -99.0)))
        _ = numpyro.deterministic("log_sigma_fake", float(obj_dict.get("log_sigma_fake", -99.0)))

        raw_params = dict(
            log_tau_slow_center0=log_tau_slow_center0,
            log_tau_fast_center0=log_tau_fast_center0,
            log_sigma_center0=log_sigma_center0,
            lambda_center_rf=lambda_center_rf,
            linear_trend=linear_trend,
            linear_trend_band_offset=linear_trend_band_offset,
            mean=mean,
            dlog_amp_blr=dlog_amp_blr,
            dlog_amp_blr2=dlog_amp_blr2,
            log_lag_blr=log_lag_blr,
            log_lag_blr2=log_lag_blr2,
            log_jitter=log_jitter,
            survey_delta_mag=survey_delta_mag,
            lag0=lag0,
            lag_beta=lag_beta,
            log_igm_transmission_band=log_igm_transmission_band,
            eta_sigma=eta_sigma,
            eta_tau=eta_tau,
        )
        if dlog_amp_bc is not None:
            raw_params["dlog_amp_bc"] = dlog_amp_bc
            raw_params["log_lag_ratio_bc_to_blr"] = log_lag_ratio_bc_to_blr

        params = build_explicit_model_params(
            raw_params,
            lam_rf,
            lam_lya_rf=lam_lya_rf,
        )

        numpyro.deterministic("lambda_center_rf", params["lambda_center_rf"])
        numpyro.deterministic("log_sigma_uv", params["log_sigma_uv"])
        numpyro.deterministic("log_tau_uv", params["log_tau_uv"])
        numpyro.deterministic("log_tau_fast_uv", params["log_tau_fast_uv"])
        numpyro.deterministic("tau_fast", params["tau_fast_band"])
        numpyro.deterministic("tau_slow", params["tau_slow_band"])
        numpyro.deterministic("amp_cont", params["amp_cont"])
        numpyro.deterministic("amp_bc", params["amp_bc"])
        numpyro.deterministic("amp_blr", params["amp_blr"])
        numpyro.deterministic("amp_blr2", params["amp_blr2"])
        numpyro.deterministic("log_igm_transmission_band", params["log_igm_transmission_band"])
        numpyro.deterministic("igm_transmission_band", params["igm_transmission_band"])
        numpyro.deterministic("lag_disk", params["lag_disk"])
        numpyro.deterministic("lag_bc", params["lag_bc"])
        numpyro.deterministic("lag_blr", params["lag_blr"])
        numpyro.deterministic("lag_blr2", params["lag_blr2"])

        m = make_multiband_dho_blr_model(
            X=(t, bidx),
            y=y,
            yerr=yerr,
            n_band=B,
            survey_idx=survey_idx,
            zero_mean=zero_mean,
            has_jitter=has_jitter,
        )
        numpyro.factor("loglike", m.log_prob(params))

    return model


def build_single_object_model_mag_flux_linearized(
    obj_dict,
    lam_rf,
    log_jitter_mean,
    *,
    lam_lya_rf=None,
    disable_linear_trend=False,
    disable_lag_blr=False,
    disable_lag_bc=False,
    drop_band_lyman_alpha=False,
    tau_fast_truncated=False,
    n_blr_terms=1,
    use_erlang=False,
    shared_blr_lag=False,
    erlang_order=DEFAULT_ERLANG_ORDER,
    blr_lag_band_scatter=None,
):
    """Return the relative-flux quasi-separable model for one object."""

    if n_blr_terms != 1:
        raise ValueError(
            "model_variant='mag_flux_linearized' currently supports only n_blr_terms=1."
        )

    (t, bidx) = obj_dict["X"]
    y = obj_dict["y"]
    yerr = obj_dict["yerr"]
    survey_idx = jnp.asarray(obj_dict["survey_idx"], dtype=jnp.int32)
    z = float(obj_dict["z"])
    B = int(len(lam_rf))
    lambda_center_rf = compute_lambda_center_rf(lam_rf)
    if lam_lya_rf is None:
        lam_lya_rf = lam_rf
    lam_lya_rf = jnp.asarray(lam_lya_rf, dtype=lam_rf.dtype)
    bands = tuple(str(b) for b in obj_dict.get("bands", []))
    if bands:
        log_igm_transmission_band = compute_log_igm_transmission_band(bands, z)
    else:
        log_igm_transmission_band = jnp.zeros(B, dtype=lam_rf.dtype)
    baseline_flux_by_band = reference_flux_from_mean_magnitudes(obj_dict["mags_means"])
    if "y_relflux_fit" in obj_dict and "yerr_relflux_fit" in obj_dict:
        y_relflux = jnp.asarray(obj_dict["y_relflux_fit"], dtype=float)
        yerr_relflux = jnp.asarray(obj_dict["yerr_relflux_fit"], dtype=float)
    else:
        y_relflux = mag_residual_to_relative_flux(y)
        yerr_relflux = magerr_residual_to_relative_fluxerr(y, yerr)
    bidx_np = np.asarray(bidx)
    yerr_relflux_np = np.asarray(yerr_relflux, dtype=float)
    log_jitter_mean_relflux, log_jitter_active_mask_relflux = _compute_log_jitter_mean_grid(
        yerr_relflux_np,
        bidx_np,
        np.asarray(survey_idx, dtype=np.int32),
        B,
    )
    _, survey_offset_active_mask = _get_object_active_noise_calibration_masks(obj_dict, B)

    def model():
        eta_sigma = numpyro.sample("eta_sigma", eta_sigma_prior())
        eta_tau = numpyro.sample("eta_tau", eta_tau_prior())

        log_tau_slow_center0 = numpyro.sample(
            "log_tau_slow_center0",
            log_tau_slow_center0_prior(
                eta_tau,
                z,
                lambda_center_rf,
            ),
        )
        if use_erlang:
            log_tau_separation_raw = numpyro.sample(
                "log_tau_separation_raw",
                log_tau_fast_separation_raw_prior(),
            )
            log_tau_fast_center0 = numpyro.deterministic(
                "log_tau_fast_center0",
                ordered_log_tau_fast(log_tau_slow_center0, log_tau_separation_raw),
            )
        else:
            log_tau_fast_center0 = numpyro.sample(
                "log_tau_fast_center0",
                log_tau_fast_center0_prior(
                    log_tau_slow_center0,
                    tau_fast_truncated=tau_fast_truncated,
                ),
            )
        log_sigma_center0 = numpyro.sample(
            "log_sigma_center0",
            log_sigma_center0_relflux_prior(
                eta_sigma,
                lambda_center_rf,
            ),
        )
        (
            linear_trend,
            linear_trend_band_offset,
            _linear_trend_band,
        ) = sample_linear_trend_with_band_offsets(
            B=B,
            disable_linear_trend=disable_linear_trend,
            trend_prior_dist=linear_trend_prior_relflux(t_ref=t, z=z),
            band_offset_raw_prior_dist=linear_trend_band_offset_raw_prior_relflux(),
            shared_linear_trend=use_erlang,
        )

        if use_erlang:
            # This first implementation gives the prompt continuum zero delay
            # and assigns all inferred response delay to the causal BLR chain.
            lag0 = numpyro.deterministic("lag0", jnp.asarray(0.0))
            log_lag0 = numpyro.deterministic("log_lag0", jnp.asarray(0.0))
            lag_beta = numpyro.deterministic("lag_beta", jnp.asarray(0.0))
        else:
            lag0 = numpyro.sample("lag0", lag0_prior(z=z))
            log_lag0 = numpyro.deterministic("log_lag0", jnp.log(lag0))
            lag_beta = numpyro.sample("lag_beta", lag_beta_prior())

        line_ratio_offsets = compute_flux_line_ratio_offsets(
            lam_rf,
            lambda_center_rf=lambda_center_rf,
            eta_sigma=eta_sigma,
            log_igm_transmission_band=log_igm_transmission_band,
        )
        (
            mean,
            dlog_amp_blr,
            dlog_amp_blr2,
            log_lag_blr,
            log_lag_blr2,
            log_jitter,
            survey_delta_mag,
            dlog_amp_bc,
            log_lag_ratio_bc_to_blr,
        ) = sample_flux_line_latent_params(
            B=B,
            z=z,
            log_lag0=log_lag0,
            log_jitter_mean=log_jitter_mean_relflux,
            log_jitter_active_mask=log_jitter_active_mask_relflux,
            survey_offset_active_mask=survey_offset_active_mask,
            line_ratio_offsets=line_ratio_offsets,
            mean_prior_dist=mean_prior_relflux(),
            disable_lag_blr=disable_lag_blr,
            disable_lag_bc=(True if use_erlang else disable_lag_bc),
            n_blr_terms=n_blr_terms,
            shared_blr_lag=shared_blr_lag,
            blr_lag_band_scatter=blr_lag_band_scatter,
        )

        _ = numpyro.deterministic("log_tau_fake", float(obj_dict.get("log_tau_fake", -99.0)))
        _ = numpyro.deterministic("log_sigma_fake", float(obj_dict.get("log_sigma_fake", -99.0)))

        raw_params = dict(
            log_tau_slow_center0=log_tau_slow_center0,
            log_tau_fast_center0=log_tau_fast_center0,
            log_sigma_center0=log_sigma_center0,
            lambda_center_rf=lambda_center_rf,
            linear_trend=linear_trend,
            linear_trend_band_offset=linear_trend_band_offset,
            mean=mean,
            dlog_amp_blr=dlog_amp_blr,
            dlog_amp_blr2=dlog_amp_blr2,
            log_lag_blr=log_lag_blr,
            log_lag_blr2=log_lag_blr2,
            log_jitter=log_jitter,
            survey_delta_mag=survey_delta_mag,
            lag0=lag0,
            lag_beta=lag_beta,
            log_igm_transmission_band=log_igm_transmission_band,
            eta_sigma=eta_sigma,
            eta_tau=eta_tau,
        )
        if dlog_amp_bc is not None:
            raw_params["dlog_amp_bc"] = dlog_amp_bc
            raw_params["log_lag_ratio_bc_to_blr"] = log_lag_ratio_bc_to_blr

        params = build_explicit_model_params_relflux(
            raw_params,
            lam_rf,
            lam_lya_rf=lam_lya_rf,
        )

        numpyro.deterministic("lambda_center_rf", params["lambda_center_rf"])
        numpyro.deterministic("log_sigma_center0_relflux", params["log_sigma_center0_relflux"])
        numpyro.deterministic("log_sigma_uv_relflux", params["log_sigma_uv_relflux"])
        numpyro.deterministic("log_sigma_uv", params["log_sigma_uv"])
        numpyro.deterministic("log_tau_uv", params["log_tau_uv"])
        numpyro.deterministic("log_tau_fast_uv", params["log_tau_fast_uv"])
        numpyro.deterministic("tau_fast", params["tau_fast_band"])
        numpyro.deterministic("tau_slow", params["tau_slow_band"])
        numpyro.deterministic("amp_cont_relflux", params["amp_cont_relflux"])
        numpyro.deterministic("amp_bc_relflux", params["amp_bc_relflux"])
        numpyro.deterministic("amp_blr_relflux", params["amp_blr_relflux"])
        numpyro.deterministic("amp_blr2_relflux", params["amp_blr2_relflux"])
        numpyro.deterministic("amp_cont", params["amp_cont"])
        numpyro.deterministic("amp_bc", params["amp_bc"])
        numpyro.deterministic("amp_blr", params["amp_blr"])
        numpyro.deterministic("amp_blr2", params["amp_blr2"])
        numpyro.deterministic("log_igm_transmission_band", params["log_igm_transmission_band"])
        numpyro.deterministic("igm_transmission_band", params["igm_transmission_band"])
        numpyro.deterministic("lag_disk", params["lag_disk"])
        numpyro.deterministic("lag_bc", params["lag_bc"])
        numpyro.deterministic("lag_blr", params["lag_blr"])
        numpyro.deterministic("lag_blr2", params["lag_blr2"])
        numpyro.deterministic("F0_cont_band", baseline_flux_by_band)

        use_erlang_response = use_erlang and not disable_lag_blr
        model_factory = (
            make_multiband_dho_blr_flux_linearized_erlang_model
            if use_erlang_response
            else make_multiband_dho_blr_flux_linearized_model
        )
        m = model_factory(
            X=(t, bidx),
            y=y_relflux,
            yerr=yerr_relflux,
            n_band=B,
            survey_idx=survey_idx,
            baseline_flux_by_band=baseline_flux_by_band,
            zero_mean=zero_mean,
            has_jitter=has_jitter,
            **({"erlang_order": erlang_order} if use_erlang_response else {}),
        )
        numpyro.factor("loglike", m.log_prob(params))

    return model


def build_single_object_model_continuum_only(
    obj_dict,
    lam_rf,
    log_jitter_mean,
    *,
    lam_lya_rf=None,
    disable_linear_trend=False,
    drop_band_lyman_alpha=False,
    tau_fast_truncated=False,
):
    """Return the continuum-only mag-space QS model for stage 1."""

    (t, bidx) = obj_dict["X"]
    y = obj_dict["y"]
    yerr = obj_dict["yerr"]
    survey_idx = jnp.asarray(obj_dict["survey_idx"], dtype=jnp.int32)
    z = float(obj_dict["z"])
    B = int(len(lam_rf))
    lambda_center_rf = compute_lambda_center_rf(lam_rf)
    if lam_lya_rf is None:
        lam_lya_rf = lam_rf
    lam_lya_rf = jnp.asarray(lam_lya_rf, dtype=lam_rf.dtype)
    bands = tuple(str(b) for b in obj_dict.get("bands", []))
    if bands:
        log_igm_transmission_band = compute_log_igm_transmission_band(bands, z)
    else:
        log_igm_transmission_band = jnp.zeros(B, dtype=lam_rf.dtype)
    log_jitter_active_mask, survey_offset_active_mask = _get_object_active_noise_calibration_masks(
        obj_dict, B
    )
    log_jitter_mean_grid = _coerce_log_jitter_mean_grid(log_jitter_mean, B)

    def model():
        eta_sigma = numpyro.sample("eta_sigma", eta_sigma_prior())
        eta_tau = numpyro.sample("eta_tau", eta_tau_prior())

        log_tau_slow_center0 = numpyro.sample(
            "log_tau_slow_center0",
            log_tau_slow_center0_prior(
                eta_tau,
                z,
                lambda_center_rf,
            ),
        )
        log_tau_fast_center0 = numpyro.sample(
            "log_tau_fast_center0",
            log_tau_fast_center0_prior(
                log_tau_slow_center0,
                tau_fast_truncated=tau_fast_truncated,
            ),
        )
        log_sigma_center0 = numpyro.sample(
            "log_sigma_center0",
            log_sigma_center0_prior(
                eta_sigma,
                lambda_center_rf,
            ),
        )
        (
            linear_trend,
            linear_trend_band_offset,
            _linear_trend_band,
        ) = sample_linear_trend_with_band_offsets(
            B=B,
            disable_linear_trend=disable_linear_trend,
            trend_prior_dist=linear_trend_prior(t_ref=t, z=z),
            band_offset_raw_prior_dist=linear_trend_band_offset_raw_prior(),
        )

        lag0 = numpyro.sample("lag0", lag0_prior(z=z))
        lag_beta = numpyro.sample("lag_beta", lag_beta_prior())
        log_jitter = _sample_log_jitter_grid(
            log_jitter_mean_grid,
            log_jitter_active_mask,
        )
        survey_delta_mag = _sample_survey_delta_mag_grid(
            survey_offset_active_mask,
        )
        with numpyro.plate("band", B):
            mean = numpyro.sample("mean", mean_prior())

        raw_params = dict(
            log_tau_slow_center0=log_tau_slow_center0,
            log_tau_fast_center0=log_tau_fast_center0,
            log_sigma_center0=log_sigma_center0,
            lambda_center_rf=lambda_center_rf,
            linear_trend=linear_trend,
            linear_trend_band_offset=linear_trend_band_offset,
            mean=mean,
            log_jitter=log_jitter,
            survey_delta_mag=survey_delta_mag,
            lag0=lag0,
            lag_beta=lag_beta,
            log_igm_transmission_band=log_igm_transmission_band,
            eta_sigma=eta_sigma,
            eta_tau=eta_tau,
        )
        params = build_explicit_model_params_continuum_only(
            raw_params,
            lam_rf,
            lam_lya_rf=lam_lya_rf,
        )
        numpyro.deterministic("lambda_center_rf", params["lambda_center_rf"])
        numpyro.deterministic("log_sigma_uv", params["log_sigma_uv"])
        numpyro.deterministic("log_tau_uv", params["log_tau_uv"])
        numpyro.deterministic("log_tau_fast_uv", params["log_tau_fast_uv"])
        numpyro.deterministic("tau_fast", params["tau_fast_band"])
        numpyro.deterministic("tau_slow", params["tau_slow_band"])
        numpyro.deterministic("amp_cont", params["amp_cont"])
        numpyro.deterministic("amp_bc", params["amp_bc"])
        numpyro.deterministic("amp_blr", params["amp_blr"])
        numpyro.deterministic("amp_blr2", params["amp_blr2"])
        numpyro.deterministic("log_igm_transmission_band", params["log_igm_transmission_band"])
        numpyro.deterministic("igm_transmission_band", params["igm_transmission_band"])
        numpyro.deterministic("lag_disk", params["lag_disk"])
        numpyro.deterministic("lag_bc", params["lag_bc"])
        numpyro.deterministic("lag_blr", params["lag_blr"])
        numpyro.deterministic("lag_blr2", params["lag_blr2"])

        m = make_multiband_dho_blr_model(
            X=(t, bidx),
            y=y,
            yerr=yerr,
            n_band=B,
            survey_idx=survey_idx,
            zero_mean=zero_mean,
            has_jitter=has_jitter,
        )
        numpyro.factor("loglike", m.log_prob(params))

    return model


def build_single_object_model_mag_fluxmix_stage2(
    obj_dict,
    lam_rf,
    *,
    lam_lya_rf=None,
    stage1_raw_median,
    stage1_params_median,
    basis_grid_t,
    basis_relflux_norm,
    disable_lag_blr=False,
    disable_lag_bc=False,
    n_blr_terms=1,
):
    """Return the conditional flux-mixing stage-2 NUTS model."""

    if n_blr_terms != 1:
        raise ValueError(
            "model_variant='mag_fluxmix_fast' currently supports only n_blr_terms=1."
        )

    if disable_lag_blr:
        disable_lag_bc = True

    (t, bidx) = obj_dict["X"]
    y = jnp.asarray(obj_dict["y"], dtype=float)
    yerr = jnp.asarray(obj_dict["yerr"], dtype=float)
    survey_idx = jnp.asarray(obj_dict["survey_idx"], dtype=jnp.int32)
    z = float(obj_dict["z"])
    B = int(len(lam_rf))
    if lam_lya_rf is None:
        lam_lya_rf = lam_rf
    lam_lya_rf = jnp.asarray(lam_lya_rf, dtype=lam_rf.dtype)
    bands = tuple(str(b) for b in obj_dict.get("bands", []))
    if bands:
        log_igm_transmission_band_fixed = compute_log_igm_transmission_band(bands, z)
    else:
        log_igm_transmission_band_fixed = jnp.zeros(B, dtype=lam_rf.dtype)
    mean_func = make_linear_mean_func(t, zero_mean=zero_mean)
    mean_obs = mean_func(stage1_raw_median, (t, bidx))
    survey_delta_mag_fixed = jnp.asarray(
        stage1_raw_median.get("survey_delta_mag", jnp.zeros((B, len(LC_SURVEY_NAMES)), dtype=float)),
        dtype=float,
    )
    if survey_delta_mag_fixed.ndim == 2:
        mean_obs = mean_obs + survey_delta_mag_fixed[bidx, survey_idx]
    log_jitter_fixed = jnp.asarray(stage1_raw_median["log_jitter"], dtype=float)
    if has_jitter:
        if log_jitter_fixed.ndim == 2:
            jitter_term = jnp.exp(log_jitter_fixed[bidx, survey_idx])
        else:
            jitter_term = jnp.exp(log_jitter_fixed[bidx])
        yerr_eff = jnp.sqrt(yerr**2 + jitter_term ** 2)
    else:
        yerr_eff = yerr

    lambda_center_rf = jnp.asarray(stage1_params_median["lambda_center_rf"], dtype=float)
    stage1_eta_sigma = jnp.asarray(stage1_raw_median["eta_sigma"], dtype=float)
    stage1_log_sigma_center0 = jnp.asarray(stage1_raw_median["log_sigma_center0"], dtype=float)
    log_lag0_fixed = jnp.log(jnp.asarray(stage1_raw_median["lag0"], dtype=float))
    basis_grid_t = jnp.asarray(basis_grid_t, dtype=float)
    basis_relflux_norm = jnp.asarray(basis_relflux_norm, dtype=float)

    def _interp_band(tt, bb, lag):
        return jnp.interp(
            tt - lag,
            basis_grid_t,
            basis_relflux_norm[bb],
            left=0.0,
            right=0.0,
        )

    def model():
        eta_sigma = numpyro.sample("eta_sigma", eta_sigma_prior())
        log_sigma_center0 = numpyro.sample(
            "log_sigma_center0",
            log_sigma_center0_prior(
                eta_sigma,
                lambda_center_rf,
            ),
        )
        numpyro.deterministic("stage1_basis_eta_sigma", stage1_eta_sigma)
        numpyro.deterministic("stage1_basis_log_sigma_center0", stage1_log_sigma_center0)
        numpyro.deterministic("delta_eta_sigma", eta_sigma - stage1_eta_sigma)
        numpyro.deterministic(
            "delta_log_sigma_center0",
            log_sigma_center0 - stage1_log_sigma_center0,
        )
        numpyro.deterministic(
            "log_cont_scale",
            log_sigma_center0 - stage1_log_sigma_center0,
        )
        numpyro.deterministic("log_igm_transmission_band", log_igm_transmission_band_fixed)
        numpyro.deterministic("igm_transmission_band", jnp.exp(log_igm_transmission_band_fixed))
        line_ratio_offsets = compute_flux_line_ratio_offsets(
            lam_rf,
            lambda_center_rf=lambda_center_rf,
            eta_sigma=eta_sigma,
            log_igm_transmission_band=log_igm_transmission_band_fixed,
        )

        if disable_lag_blr:
            dlog_amp_blr = numpyro.deterministic("dlog_amp_blr", jnp.full(B, -9.0))
            log_lag_blr = numpyro.deterministic("log_lag_blr", jnp.full(B, -9.0))
            dlog_amp_blr2 = numpyro.deterministic("dlog_amp_blr2", jnp.full(B, -9.0))
            log_lag_blr2 = numpyro.deterministic("log_lag_blr2", jnp.full(B, -9.0))
            dlog_amp_bc = None
            log_lag_ratio_bc_to_blr = None
        else:
            if disable_lag_bc:
                dlog_amp_bc = None
                log_lag_ratio_bc_to_blr = None
            else:
                log_amp_ratio_bc = numpyro.sample(
                    "log_amp_ratio_bc",
                    dist.Normal(line_ratio_offsets["bc_ref"] - 1.0, 1.0),
                )
                dlog_amp_bc = numpyro.deterministic(
                    "dlog_amp_bc",
                    log_amp_ratio_bc - line_ratio_offsets["bc_ref"],
                )
                log_lag_ratio_bc_to_blr = numpyro.sample(
                    "log_lag_ratio_bc_to_blr",
                    log_lag_ratio_bc_to_blr_prior(),
                )

            with numpyro.plate("band", B):
                log_amp_ratio_blr_raw = numpyro.sample(
                    "log_amp_ratio_blr_raw",
                    dist.Normal(line_ratio_offsets["blr_band"] - 1.0, 1.0),
                )
                delta_log_lag_blr_raw = numpyro.sample(
                    "delta_log_lag_blr_raw",
                    relative_log_lag_blr_prior(z=z, log_lag0=log_lag0_fixed),
                )
                log_lag_blr_raw = delta_log_lag_blr_raw + log_lag0_fixed
                dlog_amp_blr = numpyro.deterministic(
                    "dlog_amp_blr",
                    log_amp_ratio_blr_raw - line_ratio_offsets["blr_band"],
                )
                log_lag_blr = numpyro.deterministic("log_lag_blr", log_lag_blr_raw)
                dlog_amp_blr2 = numpyro.deterministic("dlog_amp_blr2", jnp.full(B, -9.0))
                log_lag_blr2 = numpyro.deterministic("log_lag_blr2", jnp.full(B, -9.0))

        stage2_raw_params = dict(stage1_raw_median)
        stage2_raw_params["lambda_center_rf"] = lambda_center_rf
        stage2_raw_params["eta_sigma"] = eta_sigma
        stage2_raw_params["log_sigma_center0"] = log_sigma_center0
        stage2_raw_params["log_igm_transmission_band"] = log_igm_transmission_band_fixed
        stage2_raw_params["dlog_amp_blr"] = dlog_amp_blr
        stage2_raw_params["log_lag_blr"] = log_lag_blr
        stage2_raw_params["dlog_amp_blr2"] = dlog_amp_blr2
        stage2_raw_params["log_lag_blr2"] = log_lag_blr2
        if dlog_amp_bc is not None:
            stage2_raw_params["dlog_amp_bc"] = dlog_amp_bc
            stage2_raw_params["log_lag_ratio_bc_to_blr"] = log_lag_ratio_bc_to_blr
        explicit_params = build_explicit_model_params_fluxmix_fast(
            stage2_raw_params,
            lam_rf,
            lam_lya_rf=lam_lya_rf,
        )
        amp_cont = jnp.asarray(explicit_params["amp_cont"], dtype=float)
        amp_blr = jnp.asarray(explicit_params["amp_blr"], dtype=float)
        amp_bc = jnp.asarray(explicit_params["amp_bc"], dtype=float)
        lag_blr = jnp.asarray(explicit_params["lag_blr"], dtype=float)
        lag_bc = jnp.asarray(explicit_params["lag_bc"], dtype=float)

        def _relflux_one(tt, bb):
            prompt = amp_cont[bb] * _interp_band(tt, bb, 0.0)
            blr = amp_blr[bb] * _interp_band(tt, bb, lag_blr[bb])
            bc = amp_bc[bb] * _interp_band(tt, bb, lag_bc[bb])
            return prompt + blr + bc

        rel_flux = jax.vmap(_relflux_one)(t, bidx)
        total_flux_ratio = 1.0 + rel_flux
        floor_violation = jax.nn.softplus(
            (FLUXMIX_MIN_TOTAL_FLUX_RATIO - total_flux_ratio)
            / FLUXMIX_TOTAL_FLUX_FLOOR_SOFTNESS
        ) * FLUXMIX_TOTAL_FLUX_FLOOR_SOFTNESS
        numpyro.factor(
            "fluxmix_total_flux_floor_penalty",
            -FLUXMIX_TOTAL_FLUX_FLOOR_PENALTY
            * jnp.sum((floor_violation / FLUXMIX_TOTAL_FLUX_FLOOR_SOFTNESS) ** 2),
        )
        y_pred = mean_obs + relative_flux_to_mag_residual(
            rel_flux,
            min_total_flux_ratio=FLUXMIX_MIN_TOTAL_FLUX_RATIO,
            floor_softness=FLUXMIX_TOTAL_FLUX_FLOOR_SOFTNESS,
        )
        numpyro.sample("obs", dist.Normal(y_pred, yerr_eff), obs=y)

    return model


_RECOMPUTE_EXPLICIT_KEYS = {
    "log_kernel_param",
    "amp_cont",
    "amp_bc",
    "amp_blr",
    "amp_blr2",
    "lag_disk",
    "lag_bc",
    "lag_blr",
    "lag_blr2",
    "tau_fast_band",
    "tau_slow_band",
    "tau_fast",
    "tau_slow",
    "log_sigma_uv",
    "log_tau_uv",
    "log_tau_fast_uv",
    "log_igm_transmission_band",
    "igm_transmission_band",
    "lambda_center_rf",
}


def _strip_explicit_prediction_keys(samples):
    return {k: v for k, v in samples.items() if k not in _RECOMPUTE_EXPLICIT_KEYS}


def _flatten_per_chain_samples(samples_per_chain):
    flat = {}
    for key, value in samples_per_chain.items():
        value = np.asarray(value)
        flat[key] = value.reshape((-1,) + value.shape[2:])
    return flat


def _trim_per_chain_samples(samples_per_chain, n_draws):
    return {k: np.asarray(v)[:, :n_draws] for k, v in samples_per_chain.items()}


def _run_nuts_inference(
    numpyro_model,
    rng_key,
    *,
    num_warmup,
    num_samples,
    num_chains,
    chain_method,
    progress_bar,
    dense_mass,
    max_tree_depth,
    target_accept=0.9,
    init_strategy=None,
):
    if init_strategy is None:
        init_strategy = numpyro.infer.init_to_median()
    nuts = NUTS(
        numpyro_model,
        init_strategy=init_strategy,
        dense_mass=dense_mass,
        max_tree_depth=max_tree_depth,
        target_accept_prob=target_accept,
    )
    mcmc = MCMC(
        nuts,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=max(1, num_chains),
        chain_method=chain_method,
        progress_bar=progress_bar,
    )
    t0 = time.perf_counter()
    mcmc.run(rng_key, extra_fields=("accept_prob", "diverging"))
    elapsed = time.perf_counter() - t0
    samples_flat = tree_map(lambda x: np.asarray(device_get(x)), mcmc.get_samples(group_by_chain=False))
    samples_per_chain = tree_map(lambda x: np.asarray(device_get(x)), mcmc.get_samples(group_by_chain=True))
    extra_fields = {
        k: np.asarray(device_get(v))
        for k, v in mcmc.get_extra_fields().items()
    }
    diagnostics = {
        "accept_prob": float(np.mean(extra_fields["accept_prob"])) if "accept_prob" in extra_fields else np.nan,
        "num_divergences": int(np.sum(extra_fields["diverging"])) if "diverging" in extra_fields else 0,
        "elapsed_sec": float(elapsed),
    }
    return samples_flat, samples_per_chain, diagnostics


def _flux_linearized_fit_object(obj_dict, y_relflux, yerr_relflux):
    fit_obj = dict(obj_dict)
    fit_obj["y_relflux_fit"] = jnp.asarray(y_relflux, dtype=float)
    fit_obj["yerr_relflux_fit"] = jnp.asarray(yerr_relflux, dtype=float)
    return fit_obj


def _flux_linearized_initial_arrays(obj_dict):
    return (
        mag_residual_to_relative_flux(obj_dict["y"]),
        magerr_residual_to_relative_fluxerr(obj_dict["y"], obj_dict["yerr"]),
    )


def _posterior_median_params(samples_flat):
    return {
        key: np.nanmedian(np.asarray(value), axis=0)
        for key, value in samples_flat.items()
    }


def _flux_linearized_pseudo_data_from_prediction(obj_dict, model, params):
    """Build one Gauss-Newton pseudo-data update for the magnitude likelihood."""

    r_star, _ = model.pred(params, obj_dict["X"])
    r_star = np.asarray(device_get(r_star), dtype=float)
    y_mag = np.asarray(obj_dict["y"], dtype=float)
    yerr_mag = np.asarray(obj_dict["yerr"], dtype=float)

    total_flux_ratio = np.maximum(
        1.0 + r_star,
        FLUX_LINEARIZED_MIN_TOTAL_FLUX_RATIO,
    )
    mag_model = -2.5 * np.log10(total_flux_ratio)
    dmag_dr = -(2.5 / np.log(10.0)) / total_flux_ratio
    y_relflux = r_star + (y_mag - mag_model) / dmag_dr
    yerr_relflux = yerr_mag / np.maximum(np.abs(dmag_dr), 1e-12)
    return jnp.asarray(y_relflux, dtype=float), jnp.asarray(yerr_relflux, dtype=float)


def _build_mag_flux_linearized_model_for_fit(obj_dict, lam_rf, log_jitter_mean, **kwargs):
    return build_single_object_model_mag_flux_linearized(
        obj_dict,
        lam_rf,
        log_jitter_mean=log_jitter_mean,
        **kwargs,
    )


def run_iterated_mag_flux_linearized_inference(
    obj_dict,
    lam_rf,
    log_jitter_mean,
    *,
    lam_lya_rf=None,
    rng_key,
    fit_method,
    num_warmup,
    num_samples,
    num_chains,
    chain_method,
    progress_bar,
    dense_mass,
    max_tree_depth,
    svi_steps,
    svi_lr,
    target_accept=0.9,
    disable_linear_trend=False,
    disable_lag_blr=False,
    disable_lag_bc=False,
    drop_band_lyman_alpha=False,
    tau_fast_truncated=False,
    n_blr_terms=1,
    model_variant="mag_flux_linearized",
    shared_blr_lag=False,
    erlang_order=DEFAULT_ERLANG_ORDER,
    blr_lag_band_scatter=None,
):
    """Iteratively refit the relative-flux QS model using local magnitude-likelihood pseudo-data."""

    if fit_method == "ns":
        raise ValueError(
            "model_variant='mag_flux_linearized' uses iterative local likelihood refinement "
            "and currently requires --fit_method nuts or svi+nuts."
        )

    y_fit, yerr_fit = _flux_linearized_initial_arrays(obj_dict)
    samples_flat = None
    samples_per_chain = None
    diagnostics = {
        "flux_linearized_refinement_iters": int(FLUX_LINEARIZED_REFINEMENT_ITERS),
        "flux_linearized_min_total_flux_ratio": float(FLUX_LINEARIZED_MIN_TOTAL_FLUX_RATIO),
    }

    model_kwargs = dict(
        lam_lya_rf=lam_lya_rf,
        disable_linear_trend=disable_linear_trend,
        disable_lag_blr=disable_lag_blr,
        disable_lag_bc=disable_lag_bc,
        drop_band_lyman_alpha=drop_band_lyman_alpha,
        tau_fast_truncated=tau_fast_truncated,
        n_blr_terms=n_blr_terms,
        use_erlang=(model_variant == "mag_flux_linearized_erlang"),
        shared_blr_lag=shared_blr_lag,
        erlang_order=erlang_order,
        blr_lag_band_scatter=blr_lag_band_scatter,
    )

    for iter_idx in range(int(FLUX_LINEARIZED_REFINEMENT_ITERS)):
        fit_obj = _flux_linearized_fit_object(obj_dict, y_fit, yerr_fit)
        iter_model = _build_mag_flux_linearized_model_for_fit(
            fit_obj,
            lam_rf,
            log_jitter_mean,
            **model_kwargs,
        )
        iter_key = random.fold_in(rng_key, iter_idx)
        if fit_method == "svi+nuts":
            svi_key, mcmc_key = random.split(iter_key)
            init_values, svi_final_loss = run_svi_warm_start(
                iter_model,
                svi_key,
                num_steps=svi_steps,
                learning_rate=svi_lr,
                progress_bar=progress_bar,
            )
            init_strategy = init_to_value(values=init_values)
            diagnostics[f"flux_linearized_iter{iter_idx + 1}_svi_final_loss"] = float(svi_final_loss)
        else:
            mcmc_key = iter_key
            init_strategy = numpyro.infer.init_to_median()

        samples_flat, samples_per_chain, iter_diag = _run_nuts_inference(
            iter_model,
            mcmc_key,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=progress_bar,
            dense_mass=dense_mass,
            max_tree_depth=max_tree_depth,
            target_accept=target_accept,
            init_strategy=init_strategy,
        )
        diagnostics[f"flux_linearized_iter{iter_idx + 1}_accept_prob"] = iter_diag["accept_prob"]
        diagnostics[f"flux_linearized_iter{iter_idx + 1}_num_divergences"] = iter_diag["num_divergences"]
        diagnostics[f"flux_linearized_iter{iter_idx + 1}_elapsed_sec"] = iter_diag["elapsed_sec"]

        samples_for_prediction = add_model_prediction_params(
            samples_flat,
            lam_rf,
            model_variant="mag_flux_linearized",
            lam_lya_rf=lam_lya_rf,
        )
        params_median = _posterior_median_params(samples_for_prediction)
        use_erlang_response = (
            model_variant == "mag_flux_linearized_erlang" and not disable_lag_blr
        )
        display_factory = (
            make_multiband_dho_blr_flux_linearized_erlang_model
            if use_erlang_response
            else make_multiband_dho_blr_flux_linearized_model
        )
        display_model = display_factory(
            obj_dict["X"],
            y_fit,
            yerr_fit,
            n_band=int(len(lam_rf)),
            survey_idx=obj_dict["survey_idx"],
            baseline_flux_by_band=reference_flux_from_mean_magnitudes(obj_dict["mags_means"]),
            zero_mean=zero_mean,
            has_jitter=has_jitter,
            **({"erlang_order": erlang_order} if use_erlang_response else {}),
        )
        y_next, yerr_next = _flux_linearized_pseudo_data_from_prediction(
            obj_dict,
            display_model,
            params_median,
        )
        diagnostics[f"flux_linearized_iter{iter_idx + 1}_pseudo_delta_rms"] = float(
            np.sqrt(np.nanmean(np.square(np.asarray(y_next) - np.asarray(y_fit))))
        )
        if iter_idx < int(FLUX_LINEARIZED_REFINEMENT_ITERS) - 1:
            y_fit, yerr_fit = y_next, yerr_next

    diagnostics["accept_prob"] = diagnostics[
        f"flux_linearized_iter{int(FLUX_LINEARIZED_REFINEMENT_ITERS)}_accept_prob"
    ]
    diagnostics["num_divergences"] = diagnostics[
        f"flux_linearized_iter{int(FLUX_LINEARIZED_REFINEMENT_ITERS)}_num_divergences"
    ]
    diagnostics["elapsed_sec"] = sum(
        diagnostics[f"flux_linearized_iter{i + 1}_elapsed_sec"]
        for i in range(int(FLUX_LINEARIZED_REFINEMENT_ITERS))
    )
    return samples_flat, samples_per_chain, fit_obj, diagnostics


def _fluxmix_stage1_raw_median_params(samples_flat, lam_rf):
    """Recover the stage-1 continuum median parameters from combined samples."""

    B = int(len(lam_rf))
    mean_default = np.zeros(B, dtype=float)
    jitter_default = np.full((B, len(LC_SURVEY_NAMES)), np.log(1e-3), dtype=float)
    survey_delta_default = np.zeros((B, len(LC_SURVEY_NAMES)), dtype=float)
    linear_trend_band_offset_default = np.zeros(B, dtype=float)
    linear_trend_default = 0.0
    if "stage1_basis_log_sigma_center0" in samples_flat:
        log_sigma_center0_stage1 = np.asarray(samples_flat["stage1_basis_log_sigma_center0"], dtype=float)
    else:
        log_cont_scale = np.asarray(
            samples_flat.get(
                "log_cont_scale",
                np.zeros_like(np.asarray(samples_flat["log_sigma_center0"], dtype=float)),
            ),
            dtype=float,
        )
        log_sigma_center0_stage1 = np.asarray(samples_flat["log_sigma_center0"], dtype=float) - log_cont_scale
    if "stage1_basis_eta_sigma" in samples_flat:
        eta_sigma_stage1 = np.asarray(samples_flat["stage1_basis_eta_sigma"], dtype=float)
    else:
        eta_sigma_stage1 = np.asarray(samples_flat["eta_sigma"], dtype=float)
    if "log_igm_transmission_band" in samples_flat:
        log_igm_transmission_band = np.asarray(samples_flat["log_igm_transmission_band"], dtype=float)
    else:
        log_igm_transmission_band = np.zeros((1, B), dtype=float)
    return dict(
        log_tau_slow_center0=jnp.asarray(np.median(np.asarray(samples_flat["log_tau_slow_center0"], dtype=float), axis=0), dtype=float),
        log_tau_fast_center0=jnp.asarray(np.median(np.asarray(samples_flat["log_tau_fast_center0"], dtype=float), axis=0), dtype=float),
        log_sigma_center0=jnp.asarray(
            np.median(log_sigma_center0_stage1, axis=0),
            dtype=float,
        ),
        lambda_center_rf=compute_lambda_center_rf(lam_rf),
        linear_trend=jnp.asarray(
            np.median(
                np.asarray(
                    samples_flat.get("linear_trend", linear_trend_default),
                    dtype=float,
                ),
                axis=0,
            ),
            dtype=float,
        ),
        linear_trend_band_offset=jnp.asarray(
            np.median(
                np.asarray(
                    samples_flat.get(
                        "linear_trend_band_offset",
                        linear_trend_band_offset_default,
                    ),
                    dtype=float,
                ),
                axis=0,
            ),
            dtype=float,
        ),
        mean=jnp.asarray(
            np.median(np.asarray(samples_flat.get("mean", mean_default), dtype=float), axis=0),
            dtype=float,
        ),
        log_jitter=jnp.asarray(
            np.median(np.asarray(samples_flat.get("log_jitter", jitter_default), dtype=float), axis=0),
            dtype=float,
        ),
        survey_delta_mag=jnp.asarray(
            np.median(
                np.asarray(samples_flat.get("survey_delta_mag", survey_delta_default), dtype=float),
                axis=0,
            ),
            dtype=float,
        ),
        lag0=jnp.asarray(np.median(np.asarray(samples_flat["lag0"], dtype=float), axis=0), dtype=float),
        lag_beta=jnp.asarray(np.median(np.asarray(samples_flat["lag_beta"], dtype=float), axis=0), dtype=float),
        log_igm_transmission_band=jnp.asarray(np.median(log_igm_transmission_band, axis=0), dtype=float),
        eta_sigma=jnp.asarray(np.median(eta_sigma_stage1, axis=0), dtype=float),
        eta_tau=jnp.asarray(np.median(np.asarray(samples_flat["eta_tau"], dtype=float), axis=0), dtype=float),
    )


def _build_fluxmix_continuum_basis(obj_dict, lam_rf, stage1_raw_median, *, lam_lya_rf=None):
    """Build the fixed normalized continuum basis used by stage 2."""

    B = int(len(lam_rf))
    continuum_params = build_explicit_model_params_continuum_only(
        stage1_raw_median,
        lam_rf,
        lam_lya_rf=lam_lya_rf,
    )
    continuum_model = make_multiband_dho_blr_model(
        obj_dict["X"],
        obj_dict["y"],
        obj_dict["yerr"],
        n_band=B,
        survey_idx=obj_dict["survey_idx"],
        zero_mean=zero_mean,
        has_jitter=has_jitter,
    )

    t_obs = np.asarray(obj_dict["X"][0], dtype=float)
    lag_pad = float(np.exp(LOG_LAG_BLR_HIGH))
    n_grid = int(
        np.clip(
            max(FLUXMIX_BASIS_GRID_MIN_POINTS, 4 * len(t_obs)),
            FLUXMIX_BASIS_GRID_MIN_POINTS,
            FLUXMIX_BASIS_GRID_MAX_POINTS,
        )
    )
    grid_t = jnp.linspace(
        float(np.min(t_obs)) - lag_pad,
        float(np.max(t_obs)) + FLUXMIX_PREDICTION_FORWARD_PAD,
        n_grid,
        dtype=float,
    )

    basis_rows = []
    for band in range(B):
        band_idx_grid = jnp.full_like(grid_t, band, dtype=int)
        pred_mean, _ = continuum_model.pred(continuum_params, (grid_t, band_idx_grid))
        mean_grid = continuum_model.get_mean(
            continuum_model.zero_mean,
            continuum_params,
            (grid_t, band_idx_grid),
        )
        prompt_dm = pred_mean - mean_grid
        prompt_relflux = mag_residual_to_relative_flux(prompt_dm)
        amp_cont_band = jnp.maximum(jnp.asarray(continuum_params["amp_cont"], dtype=float)[band], 1e-12)
        basis_rows.append(prompt_relflux / amp_cont_band)

    basis_relflux_norm = jnp.stack(basis_rows, axis=0)
    return {
        "continuum_model": continuum_model,
        "continuum_params": continuum_params,
        "basis_grid_t": grid_t,
        "basis_relflux_norm": basis_relflux_norm,
    }


def _prediction_sample_subset(samples_flat, max_samples=64):
    relevant_keys = (
        "mean",
        "linear_trend",
        "amp_cont",
        "amp_blr",
        "amp_bc",
        "lag_blr",
        "lag_bc",
    )
    available = {
        key: np.asarray(samples_flat[key])
        for key in relevant_keys
        if key in samples_flat
    }
    if not available:
        return None
    n_samples = len(next(iter(available.values())))
    if n_samples == 0:
        return None
    n_keep = min(max_samples, n_samples)
    indices = np.linspace(0, n_samples - 1, n_keep, dtype=int)
    return {key: value[indices] for key, value in available.items()}


def _fluxmix_zero_line_amplitudes(params):
    """Return params with BLR/BC amplitudes zeroed for continuum-only predictions."""

    out = dict(params)
    for key in (
        "amp_blr",
        "amp_blr2",
        "amp_bc",
        "amp_blr_relflux",
        "amp_blr2_relflux",
        "amp_bc_relflux",
    ):
        if key in out:
            arr = np.asarray(out[key], dtype=float)
            out[key] = np.zeros_like(arr)
    return out


def _combine_fluxmix_stage_samples(stage1_per_chain, stage2_per_chain, lam_rf, *, lam_lya_rf=None):
    """Merge stage-1 and stage-2 per-chain samples into the public posterior dict."""

    n_draws = min(
        int(next(iter(stage1_per_chain.values())).shape[1]),
        int(next(iter(stage2_per_chain.values())).shape[1]),
    )
    stage1_trim = _trim_per_chain_samples(stage1_per_chain, n_draws)
    stage2_trim = _trim_per_chain_samples(stage2_per_chain, n_draws)

    combined_per_chain_raw = _strip_explicit_prediction_keys(stage1_trim)
    combined_per_chain_raw.update(stage2_trim)
    combined_per_chain_raw["stage1_basis_eta_sigma"] = np.asarray(stage1_trim["eta_sigma"], dtype=float)
    combined_per_chain_raw["stage1_basis_log_sigma_center0"] = np.asarray(
        stage1_trim["log_sigma_center0"],
        dtype=float,
    )
    combined_per_chain = add_model_prediction_params(
        combined_per_chain_raw,
        lam_rf,
        model_variant="mag_fluxmix_fast",
        lam_lya_rf=lam_lya_rf,
    )
    combined_flat = _flatten_per_chain_samples(combined_per_chain)
    combined_flat = add_model_prediction_params(
        combined_flat,
        lam_rf,
        model_variant="mag_fluxmix_fast",
        lam_lya_rf=lam_lya_rf,
    )
    return combined_flat, combined_per_chain


def _make_fluxmix_display_model(obj_dict, basis_info, combined_flat):
    """Construct the staged display model for plotting and basis updates."""

    return make_multiband_dho_blr_fluxmix_fast_display_model(
        continuum_model=basis_info["continuum_model"],
        basis_grid_t=basis_info["basis_grid_t"],
        basis_relflux_norm=basis_info["basis_relflux_norm"],
        t_ref=obj_dict["X"][0],
        zero_mean=zero_mean,
        prediction_samples=_prediction_sample_subset(combined_flat),
        min_total_flux_ratio=FLUXMIX_MIN_TOTAL_FLUX_RATIO,
        floor_softness=FLUXMIX_TOTAL_FLUX_FLOOR_SOFTNESS,
    )


def _fluxmix_line_only_mag_residual(display_model, combined_flat, X):
    """Approximate the line-only magnitude residual from the staged display model."""

    posterior_median = {k: np.median(v, axis=0) for k, v in combined_flat.items()}
    total_mu, _ = display_model.pred(posterior_median, X)
    cont_mu, _ = display_model.pred(_fluxmix_zero_line_amplitudes(posterior_median), X)
    return np.asarray(total_mu, dtype=float) - np.asarray(cont_mu, dtype=float)


def build_mag_fluxmix_fast_display_model(obj_dict, lam_rf, samples_flat):
    """Reconstruct the staged display model from saved combined samples."""

    stage1_raw_median = _fluxmix_stage1_raw_median_params(samples_flat, lam_rf)
    basis_info = _build_fluxmix_continuum_basis(obj_dict, lam_rf, stage1_raw_median)
    prediction_samples = _prediction_sample_subset(samples_flat)
    return make_multiband_dho_blr_fluxmix_fast_display_model(
        continuum_model=basis_info["continuum_model"],
        basis_grid_t=basis_info["basis_grid_t"],
        basis_relflux_norm=basis_info["basis_relflux_norm"],
        t_ref=obj_dict["X"][0],
        zero_mean=zero_mean,
        prediction_samples=prediction_samples,
        min_total_flux_ratio=FLUXMIX_MIN_TOTAL_FLUX_RATIO,
        floor_softness=FLUXMIX_TOTAL_FLUX_FLOOR_SOFTNESS,
    )


def apply_resume_sample_save_policy(args):
    """Disable per-object posterior sample writes when reusing saved samples."""
    if getattr(args, "resume", False) and getattr(args, "save_sample_file", False):
        logging.info("--resume set; disabling per-object posterior sample file saving.")
        args.save_sample_file = False
    return args


def run_two_stage_fluxmix_fast_inference(
    obj_dict,
    lam_rf,
    log_jitter_mean,
    *,
    lam_lya_rf=None,
    rng_key,
    num_warmup,
    num_samples,
    num_chains,
    chain_method,
    progress_bar,
    dense_mass,
    max_tree_depth,
    target_accept=0.9,
    disable_linear_trend=False,
    disable_lag_blr=False,
    disable_lag_bc=False,
    drop_band_lyman_alpha=False,
    tau_fast_truncated=False,
    n_blr_terms=1,
    outer_iters=1,
):
    """Run the staged NUTS flux-mixing approximation and return combined samples."""

    if n_blr_terms != 1:
        raise ValueError(
            "model_variant='mag_fluxmix_fast' currently supports only n_blr_terms=1."
        )
    if int(outer_iters) < 1:
        raise ValueError("outer_iters must be >= 1.")
    stage1_fit_obj = dict(obj_dict)
    combined_flat = None
    combined_per_chain = None
    basis_info = None
    display_model = None
    diagnostics = {
        "fluxmix_outer_iters": int(outer_iters),
        "stage2_conditioned_on_median_basis": True,
    }

    for outer_iter in range(int(outer_iters)):
        iter_idx = outer_iter + 1
        iter_key = random.fold_in(rng_key, outer_iter)

        stage1_model = build_single_object_model_continuum_only(
            stage1_fit_obj,
            lam_rf,
            log_jitter_mean=log_jitter_mean,
            lam_lya_rf=lam_lya_rf,
            disable_linear_trend=disable_linear_trend,
            drop_band_lyman_alpha=drop_band_lyman_alpha,
            tau_fast_truncated=tau_fast_truncated,
        )
        stage1_flat, stage1_per_chain, stage1_diag = _run_nuts_inference(
            stage1_model,
            iter_key,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=progress_bar,
            dense_mass=dense_mass,
            max_tree_depth=max_tree_depth,
            target_accept=target_accept,
        )

        stage1_raw_median = _fluxmix_stage1_raw_median_params(stage1_flat, lam_rf)
        basis_info = _build_fluxmix_continuum_basis(
            obj_dict,
            lam_rf,
            stage1_raw_median,
            lam_lya_rf=lam_lya_rf,
        )

        stage2_model = build_single_object_model_mag_fluxmix_stage2(
            obj_dict,
            lam_rf,
            lam_lya_rf=lam_lya_rf,
            stage1_raw_median=stage1_raw_median,
            stage1_params_median=basis_info["continuum_params"],
            basis_grid_t=basis_info["basis_grid_t"],
            basis_relflux_norm=basis_info["basis_relflux_norm"],
            disable_lag_blr=disable_lag_blr,
            disable_lag_bc=disable_lag_bc,
            n_blr_terms=n_blr_terms,
        )
        stage2_key = random.fold_in(iter_key, 1)
        stage2_flat, stage2_per_chain, stage2_diag = _run_nuts_inference(
            stage2_model,
            stage2_key,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=progress_bar,
            dense_mass=dense_mass,
            max_tree_depth=max_tree_depth,
            target_accept=target_accept,
            init_strategy=init_to_value(
                values={
                    "eta_sigma": np.asarray(stage1_raw_median["eta_sigma"], dtype=float),
                    "log_sigma_center0": np.asarray(stage1_raw_median["log_sigma_center0"], dtype=float),
                }
            ),
        )

        combined_flat, combined_per_chain = _combine_fluxmix_stage_samples(
            stage1_per_chain,
            stage2_per_chain,
            lam_rf,
            lam_lya_rf=lam_lya_rf,
        )
        display_model = _make_fluxmix_display_model(obj_dict, basis_info, combined_flat)

        diagnostics[f"stage1_iter{iter_idx}_accept_prob"] = stage1_diag["accept_prob"]
        diagnostics[f"stage2_iter{iter_idx}_accept_prob"] = stage2_diag["accept_prob"]
        diagnostics[f"stage1_iter{iter_idx}_num_divergences"] = stage1_diag["num_divergences"]
        diagnostics[f"stage2_iter{iter_idx}_num_divergences"] = stage2_diag["num_divergences"]
        diagnostics[f"stage1_iter{iter_idx}_elapsed_sec"] = stage1_diag["elapsed_sec"]
        diagnostics[f"stage2_iter{iter_idx}_elapsed_sec"] = stage2_diag["elapsed_sec"]

        if outer_iter < int(outer_iters) - 1:
            line_only = _fluxmix_line_only_mag_residual(display_model, combined_flat, obj_dict["X"])
            stage1_fit_obj = dict(obj_dict)
            stage1_fit_obj["y"] = np.asarray(obj_dict["y"], dtype=float) - line_only
            diagnostics[f"stage1_iter{iter_idx + 1}_line_subtracted_rms"] = float(
                np.sqrt(np.mean(np.square(line_only)))
            )

    diagnostics["stage1_accept_prob"] = diagnostics[f"stage1_iter{int(outer_iters)}_accept_prob"]
    diagnostics["stage2_accept_prob"] = diagnostics[f"stage2_iter{int(outer_iters)}_accept_prob"]
    diagnostics["stage1_num_divergences"] = diagnostics[f"stage1_iter{int(outer_iters)}_num_divergences"]
    diagnostics["stage2_num_divergences"] = diagnostics[f"stage2_iter{int(outer_iters)}_num_divergences"]
    diagnostics["stage1_elapsed_sec"] = diagnostics[f"stage1_iter{int(outer_iters)}_elapsed_sec"]
    diagnostics["stage2_elapsed_sec"] = diagnostics[f"stage2_iter{int(outer_iters)}_elapsed_sec"]
    return combined_flat, combined_per_chain, display_model, diagnostics


def main():
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Fit quasars one-by-one with the DHO+BLR light-curve model.")
    parser.add_argument("--filter_object_id", nargs="+", help="List of object IDs to filter.")
    parser.add_argument("--N", type=int, help="Number of objects to process.")
    parser.add_argument("--skip", type=int, help="Number of objects to skip.")
    parser.add_argument("--filter_file", type=str, help="Path to file containing object IDs.")
    parser.add_argument("--plot", action="store_true", help="Enable plotting of results.")
    parser.add_argument("--progress", action="store_true", help="Show progress bar.")
    parser.add_argument("--nwarm", type=int, default=500, help="Warmup steps for MCMC.")
    parser.add_argument("--nsamp", type=int, default=250, help="Samples per chain for MCMC.")
    parser.add_argument("--nchains", type=int, default=2, help="Number of chains (>=1).")
    parser.add_argument(
        "--fit_method",
        type=str,
        choices=("ns", "nuts", "svi+nuts", "two_stage_nuts", "alternating_two_stage_nuts"),
        default="nuts",
        help="Posterior fitting backend: nested sampling ('ns'), HMC/NUTS ('nuts'), SVI warm-start plus NUTS ('svi+nuts'), the staged two-NUTS flux-mixing approximation ('two_stage_nuts'), or the alternating staged flux-mixing backend ('alternating_two_stage_nuts').",
    )
    parser.add_argument("--inject_fake", action="store_true", help="Inject fake light curves.")
    parser.add_argument("--max_tree_depth", type=int, default=8, help="NUTS max tree depth.")
    parser.add_argument(
        "--target_accept",
        type=float,
        default=0.7,
        help="Target NUTS acceptance probability (default: 0.7).",
    )
    parser.add_argument(
        "--dense_mass",
        action="store_true",
        dest="dense_mass",
        help="Use dense mass matrix adaptation for NUTS (default).",
    )
    parser.add_argument(
        "--no_dense_mass",
        action="store_false",
        dest="dense_mass",
        help="Use diagonal rather than dense mass matrix adaptation for NUTS.",
    )
    parser.set_defaults(dense_mass=True)
    parser.add_argument(
        "--svi_steps",
        type=int,
        default=1000,
        help="SVI warm-start steps used only with --fit_method svi+nuts.",
    )
    parser.add_argument(
        "--svi_lr",
        type=float,
        default=1e-2,
        help="SVI learning rate used only with --fit_method svi+nuts.",
    )
    parser.add_argument(
        "--fluxmix_outer_iters",
        type=int,
        default=2,
        help="Number of outer stage1/stage2 alternations used only with --fit_method alternating_two_stage_nuts.",
    )
    parser.add_argument(
        "--ns_num_live_points",
        type=int,
        default=None,
        help="Nested sampler live-point count. Default uses NumPyro/JAXNS heuristic.",
    )
    parser.add_argument(
        "--ns_max_samples",
        type=int,
        default=None,
        help="Nested sampler maximum internal sample count. Default uses NumPyro/JAXNS heuristic.",
    )
    parser.add_argument(
        "--ns_dlogz",
        type=float,
        default=1.0,
        help="Nested sampler evidence tolerance for termination. Smaller is stricter. Default: 1.",
    )
    parser.add_argument("--resume", action="store_true", help="Load saved samples (debug).")
    parser.add_argument("--save_sample_file", dest="save_sample_file", action="store_true", help="Save per-object posterior samples to HDF5.")
    parser.add_argument("--no_save_sample_file", dest="save_sample_file", action="store_false", help="Do not save per-object posterior samples to HDF5.")
    parser.set_defaults(save_sample_file=True)
    parser.add_argument("--disable_linear_trend", action="store_true", help="Disable trend.")
    parser.add_argument("--rf_length_cut", type=int, default=-1, help="Rest-frame cut (days).")
    parser.add_argument("--exact_same_length", action="store_true", help="Exact same RF length cut.")
    parser.add_argument("--load_stone_lcs", action="store_true", default=False, help="Use Stone LCs.")
    parser.add_argument("--disable_trace_plot", action="store_true", default=False, help="Disable MCMC trace plot.")
    parser.add_argument("--disable_combined_plot", action="store_true", default=False, help="Disable combined light-curve fit plot.")
    parser.add_argument("--disable_color_magnitude_plot", action="store_true", default=False, help="Disable color-magnitude plot.")
    parser.add_argument("--disable_correlation_plot", action="store_true", default=False, help="Disable correlation matrix plot.")
    parser.add_argument("--disable_histogram_plot", action="store_true", default=False, help="Disable posterior histogram plot.")
    parser.add_argument("--disable_corner_plot", action="store_true", default=False, help="Disable corner plot.")
    parser.add_argument(
        "--plot_ls_broken_pl",
        action="store_true",
        default=False,
        help="Overlay the fitted Lomb-Scargle broken power law and uncorrected LS PSD points on the PSD subplot.",
    )
    parser.add_argument(
        "--show_combined_light_curve_component_overlay",
        action="store_true",
        default=False,
        help="Show the dashed component overlay on the combined light-curve fit plot.",
    )
    parser.add_argument(
        "--corner_plot_mode",
        type=str,
        choices=("fast", "full"),
        default="fast",
        help="Corner plot row selection: fast subsampling or full posterior samples.",
    )
    parser.add_argument("--disable_lag_blr", action="store_true", default=False, help="Disable BLR lag model.")
    parser.add_argument(
        "--shared_blr_lag",
        action="store_true",
        default=False,
        help="Sample one BLR lag and use it in every retained band; per-band BLR amplitudes remain free.",
    )
    parser.add_argument(
        "--blr_lag_band_scatter",
        type=float,
        default=0.4,
        help="Partially pool band BLR lags around one shared log lag with this fixed natural-log scatter (default: 0.4).",
    )
    parser.add_argument(
        "--independent_blr_lags",
        action="store_const",
        const=None,
        dest="blr_lag_band_scatter",
        help="Use the legacy fully independent prior for each band's BLR lag.",
    )
    parser.set_defaults(blr_lag_band_scatter=0.4)
    parser.add_argument("--disable_lag_bc", action="store_true", default=False, help="Disable Balmer-continuum lag model.")
    parser.add_argument("--disable_plot_psd", action="store_true", default=False, help="Disable PSD sub-plot.")
    parser.add_argument("--disable_sigma_tau_lambda_plot", action="store_true", default=False, help="Disable sigma-tau versus wavelength summary plot.")
    parser.add_argument("--disable_recovery_plot", action="store_true", default=False, help="Disable fake-data recovery summary plot.")
    parser.add_argument("--inject_random_fake_etas", action="store_true", default=False, help="Randomize fake etas.")
    parser.add_argument("--beta_tau", type=float, default=0.2, help="beta_tau for fake curves.")
    parser.add_argument(
        "--drop_band_lyman_alpha",
        action="store_true",
        default=False,
        help="Drop bands blueward of Ly-alpha instead of keeping them with smooth attenuation.",
    )
    parser.add_argument("--load_nearby_lc_csv", type=str, default=None, help="CSV listing nearby LCs to load.")
    parser.add_argument("--tau_fast_truncated", action="store_true", default=False, help="Truncated prior for tau_fast0.")
    parser.add_argument("--n_blr_terms", type=int, choices=(1, 2), default=1, help="Number of BLR lag terms to fit.")
    parser.add_argument(
        "--erlang_order",
        type=int,
        default=DEFAULT_ERLANG_ORDER,
        help=f"Positive Erlang BLR response order for --model_variant mag_flux_linearized_erlang (default: {DEFAULT_ERLANG_ORDER}).",
    )
    parser.add_argument(
        "--model_variant",
        choices=("mag_linear", "mag_flux_linearized", "mag_flux_linearized_erlang", "mag_fluxmix_fast"),
        default="mag_linear",
        help="Light-curve model path: the legacy additive-magnitude quasi-sep solver, the relative-flux quasi-sep approximation with mag-equivalent outputs, or the staged fast flux-mixing approximation.",
    )
    parser.add_argument(
        "--spectra_fit_csv",
        nargs="+",
        default=None,
        help="Spectra-fit CSV file(s) used to derive per-band PSF PL/total fractions.",
    )
    parser.add_argument(
        "--subtract_psf_constant_flux",
        action="store_true",
        default=False,
        help="Subtract spectra-derived constant contaminating flux in PSF light curves before GP fitting.",
    )
    args = parser.parse_args()
    args = apply_resume_sample_save_policy(args)
    print("Args:", args)

    if args.load_stone_lcs:
        objs = load_stone_lcs(filter_object_ids=args.filter_object_id)
        print(f"Loaded {len(objs)} Stone light curves.")
    elif args.load_nearby_lc_csv is not None:
        objs = load_nearby_lcs(args.load_nearby_lc_csv)
        print(f"Loaded {len(objs)} nearby light curves from {args.load_nearby_lc_csv}.")
    else:
        objs = concat_light_curves(
            filter_object_ids=args.filter_object_id,
            progress_bar=args.progress,
            N=args.N,
            skip=args.skip,
        )
    print(f"Loaded {len(objs)} objects.")

    objs = populate_sdss_fields(objs, progress_bar=args.progress)
    for obj in objs:
        obj["psf_constant_flux_n_bands_corrected"] = 0
        obj["psf_constant_flux_corrected"] = False
    if args.rf_length_cut > 0:
        objs = cut_light_curve_restframe_window(
            objs,
            n_days=args.rf_length_cut,
            same_length=args.exact_same_length,
        )
        print(f"After restframe cut, {len(objs)} objects remain.")

    if args.subtract_psf_constant_flux:
        if not args.spectra_fit_csv:
            raise ValueError("--subtract_psf_constant_flux requires --spectra_fit_csv.")
        objs, correction_summary = apply_constant_flux_correction_to_objects(
            objs,
            spectra_fit_csvs=args.spectra_fit_csv,
            progress_bar=args.progress,
        )
        print_constant_flux_correction_summary(correction_summary)

    if args.inject_random_fake_etas:
        rng = np.random.default_rng()
        alpha_sigma = float(rng.uniform(-1.0, 0.0))
        beta_tau = float(rng.uniform(-0.5, 2.0))
        print(f"Randomized alpha_sigma={alpha_sigma:.3f}, beta_tau={beta_tau:.3f}")
    else:
        alpha_sigma = -0.5
        beta_tau = float(args.beta_tau)
        print(f"Using fixed alpha_sigma={alpha_sigma:.3f}, beta_tau={beta_tau:.3f}")

    results = []
    chain_method = "parallel" if args.nchains and args.nchains > 1 else "sequential"
    if args.model_variant == "mag_fluxmix_fast" and args.fit_method not in ("two_stage_nuts", "alternating_two_stage_nuts"):
        raise ValueError(
            "model_variant='mag_fluxmix_fast' requires --fit_method two_stage_nuts or alternating_two_stage_nuts."
        )
    if args.fit_method in ("two_stage_nuts", "alternating_two_stage_nuts") and args.model_variant != "mag_fluxmix_fast":
        raise ValueError(
            f"--fit_method {args.fit_method} is only supported with --model_variant mag_fluxmix_fast."
        )
    if args.model_variant in ("mag_flux_linearized", "mag_flux_linearized_erlang") and args.fit_method == "ns":
        raise ValueError(
            "flux-linearized model variants use iterative local likelihood refinement "
            "and requires --fit_method nuts or svi+nuts."
        )
    if args.fit_method == "alternating_two_stage_nuts" and args.fluxmix_outer_iters < 2:
        raise ValueError("--fit_method alternating_two_stage_nuts requires --fluxmix_outer_iters >= 2.")
    if args.shared_blr_lag:
        args.blr_lag_band_scatter = 0.0
    if args.model_variant == "mag_fluxmix_fast":
        args.blr_lag_band_scatter = None
    if args.shared_blr_lag and args.n_blr_terms != 1:
        raise ValueError("--shared_blr_lag currently requires --n_blr_terms 1.")
    if args.blr_lag_band_scatter is not None and args.blr_lag_band_scatter < 0.0:
        raise ValueError("--blr_lag_band_scatter must be non-negative.")
    if args.blr_lag_band_scatter is not None and args.n_blr_terms != 1:
        raise ValueError("--blr_lag_band_scatter currently requires --n_blr_terms 1.")
    if args.shared_blr_lag and args.model_variant == "mag_fluxmix_fast":
        raise ValueError("--shared_blr_lag is not yet supported by model_variant='mag_fluxmix_fast'.")
    if args.erlang_order < 1:
        raise ValueError("--erlang_order must be at least 1.")
    if not 0.0 < args.target_accept < 1.0:
        raise ValueError("--target_accept must be strictly between 0 and 1.")
    if args.erlang_order != DEFAULT_ERLANG_ORDER and args.model_variant != "mag_flux_linearized_erlang":
        raise ValueError("--erlang_order is only used by --model_variant mag_flux_linearized_erlang.")
    if args.fit_method == "ns":
        if NestedSampler is None:
            raise ImportError(
                "Requested --fit_method ns, but numpyro.contrib.nested_sampling is unavailable. "
                "This NumPyro wrapper requires the optional 'jaxns' package to be installed."
            )
        if args.nchains != 1:
            logging.warning(
                "--fit_method ns does not use multiple chains; ignoring --nchains=%s.",
                args.nchains,
            )
        if args.nwarm > 0:
            logging.warning(
                "--fit_method ns does not use warmup; ignoring --nwarm=%s.",
                args.nwarm,
            )
        if args.max_tree_depth != parser.get_default("max_tree_depth"):
            logging.warning(
                "--fit_method ns does not use max_tree_depth; ignoring --max_tree_depth=%s.",
                args.max_tree_depth,
            )
    elif args.fit_method == "nuts":
        if args.svi_steps != parser.get_default("svi_steps"):
            logging.warning(
                "--fit_method nuts does not use SVI warm-start; ignoring --svi_steps=%s.",
                args.svi_steps,
            )
        if args.svi_lr != parser.get_default("svi_lr"):
            logging.warning(
                "--fit_method nuts does not use SVI warm-start; ignoring --svi_lr=%s.",
                args.svi_lr,
            )
    elif args.fit_method in ("two_stage_nuts", "alternating_two_stage_nuts"):
        if args.svi_steps != parser.get_default("svi_steps"):
            logging.warning(
                "--fit_method %s does not use SVI warm-start; ignoring --svi_steps=%s.",
                args.fit_method,
                args.svi_steps,
            )
        if args.svi_lr != parser.get_default("svi_lr"):
            logging.warning(
                "--fit_method %s does not use SVI warm-start; ignoring --svi_lr=%s.",
                args.fit_method,
                args.svi_lr,
            )

    iterator = tqdm(objs, desc="Fitting", disable=not args.progress)
    for idx, obj in enumerate(iterator):
        oid = str(obj["object_id"])
        try:
            default_bands = ["u", "g", "r", "i", "z"]
            if args.load_stone_lcs or args.load_nearby_lc_csv is not None:
                default_bands = ["g", "r", "i"]

            lc = make_lc(
                obj,
                bands=default_bands,
                inject_fake=args.inject_fake,
                alpha_sigma=alpha_sigma,
                beta_tau=beta_tau,
                drop_band_lyman_alpha=args.drop_band_lyman_alpha,
            )
            if lc is None:
                continue

            obj |= lc

            bands = obj["bands"]
            lam_rf = jnp.array([lambda_pivot[b] for b in bands], dtype=float) / (1.0 + float(obj["z"]))
            lam_lya_rf = compute_lam_lya_suppression_rf(bands, obj["z"])
            lambda_center_rf = compute_lambda_center_rf(lam_rf)
            print(f"[{oid}] Using bands: {bands}")
            print(f"[{oid}] lam_rf = {lam_rf}")
            print(f"[{oid}] lambda_center_rf = {lambda_center_rf}")

            bidx = obj["band_idx"]
            yerr = np.asarray(obj["yerr"])
            survey_idx = np.asarray(obj["survey_idx"], dtype=np.int32)
            B = len(bands)
            log_jitter_mean, log_jitter_active_mask = _compute_log_jitter_mean_grid(
                yerr,
                bidx,
                survey_idx,
                B,
            )
            survey_offset_active_mask = _compute_survey_offset_active_mask(
                bidx,
                survey_idx,
                B,
            )
            obj["log_jitter_active_mask"] = log_jitter_active_mask
            obj["survey_offset_active_mask"] = survey_offset_active_mask
            log_jitter_mean_fit = log_jitter_mean
            if args.model_variant in ("mag_flux_linearized", "mag_flux_linearized_erlang"):
                y_relflux = np.asarray(mag_residual_to_relative_flux(obj["y"]), dtype=float)
                yerr_relflux = np.asarray(
                    magerr_residual_to_relative_fluxerr(obj["y"], obj["yerr"]),
                    dtype=float,
                )
                log_jitter_mean_fit, _ = _compute_log_jitter_mean_grid(
                    yerr_relflux,
                    bidx,
                    survey_idx,
                    B,
                )
            if args.model_variant == "mag_linear":
                model_builder = build_single_object_model
            elif args.model_variant in ("mag_flux_linearized", "mag_flux_linearized_erlang"):
                if args.n_blr_terms != 1:
                    raise ValueError(
                        f"model_variant={args.model_variant!r} currently supports only n_blr_terms=1."
                    )
                model_builder = None
                logging.info(
                    "[%s] %s using iterative local magnitude-likelihood refinement with relative-flux QS solves; active_components=%s",
                    oid,
                    args.model_variant,
                    "cont" if args.disable_lag_blr else ("cont,blr" if args.model_variant == "mag_flux_linearized_erlang" or args.disable_lag_bc else "cont,blr,bc"),
                )
            elif args.model_variant == "mag_fluxmix_fast":
                if args.n_blr_terms != 1:
                    raise ValueError(
                        "model_variant='mag_fluxmix_fast' currently supports only n_blr_terms=1."
                    )
                model_builder = None
                active_components = "cont" if args.disable_lag_blr else ("cont,blr" if args.disable_lag_bc else "cont,blr,bc")
                if args.fit_method == "alternating_two_stage_nuts":
                    logging.info(
                        "[%s] mag_fluxmix_fast using alternating two-stage NUTS with %d outer iterations: continuum-only QS alternated with conditional flux mixing; active_components=%s",
                        oid,
                        args.fluxmix_outer_iters,
                        active_components,
                    )
                else:
                    logging.info(
                        "[%s] mag_fluxmix_fast using two-stage NUTS: continuum-only QS followed by conditional flux mixing; active_components=%s",
                        oid,
                        active_components,
                    )
            else:
                raise ValueError(f"Unknown model_variant '{args.model_variant}'.")
            if model_builder is not None:
                numpyro_model = model_builder(
                    obj,
                    lam_rf,
                    log_jitter_mean=log_jitter_mean_fit,
                    lam_lya_rf=lam_lya_rf,
                    disable_linear_trend=args.disable_linear_trend,
                    disable_lag_blr=args.disable_lag_blr,
                    disable_lag_bc=args.disable_lag_bc,
                    drop_band_lyman_alpha=args.drop_band_lyman_alpha,
                    tau_fast_truncated=args.tau_fast_truncated,
                    n_blr_terms=args.n_blr_terms,
                    shared_blr_lag=args.shared_blr_lag,
                    blr_lag_band_scatter=args.blr_lag_band_scatter,
                )
            else:
                numpyro_model = None

            stage_diagnostics = {}
            flux_linearized_fit_obj = None
            if args.resume:
                logging.warning("[DEBUG] Loading saved samples (flat) — developer mode.")
                obj_flat_samples = load_obj_samples_from_hdf5(oid)
                samples_per_chain = None
            else:
                key = random.PRNGKey(0)
                key = random.fold_in(key, idx)
                if args.fit_method in ("two_stage_nuts", "alternating_two_stage_nuts"):
                    obj_flat_samples, samples_per_chain, m, stage_diagnostics = run_two_stage_fluxmix_fast_inference(
                        obj,
                        lam_rf,
                        log_jitter_mean=log_jitter_mean,
                        lam_lya_rf=lam_lya_rf,
                        rng_key=key,
                        num_warmup=args.nwarm,
                        num_samples=args.nsamp,
                        num_chains=args.nchains,
                        chain_method=chain_method,
                        progress_bar=args.progress,
                        dense_mass=args.dense_mass,
                        max_tree_depth=args.max_tree_depth,
                        target_accept=args.target_accept,
                        disable_linear_trend=args.disable_linear_trend,
                        disable_lag_blr=args.disable_lag_blr,
                        disable_lag_bc=args.disable_lag_bc,
                        drop_band_lyman_alpha=args.drop_band_lyman_alpha,
                        tau_fast_truncated=args.tau_fast_truncated,
                        n_blr_terms=args.n_blr_terms,
                        outer_iters=(args.fluxmix_outer_iters if args.fit_method == "alternating_two_stage_nuts" else 1),
                    )
                elif args.model_variant in ("mag_flux_linearized", "mag_flux_linearized_erlang"):
                    obj_flat_samples, samples_per_chain, flux_linearized_fit_obj, stage_diagnostics = run_iterated_mag_flux_linearized_inference(
                        obj,
                        lam_rf,
                        log_jitter_mean=log_jitter_mean_fit,
                        lam_lya_rf=lam_lya_rf,
                        rng_key=key,
                        fit_method=args.fit_method,
                        num_warmup=args.nwarm,
                        num_samples=args.nsamp,
                        num_chains=args.nchains,
                        chain_method=chain_method,
                        progress_bar=args.progress,
                        dense_mass=args.dense_mass,
                        max_tree_depth=args.max_tree_depth,
                        svi_steps=args.svi_steps,
                        svi_lr=args.svi_lr,
                        target_accept=args.target_accept,
                        disable_linear_trend=args.disable_linear_trend,
                        disable_lag_blr=args.disable_lag_blr,
                        disable_lag_bc=args.disable_lag_bc,
                        drop_band_lyman_alpha=args.drop_band_lyman_alpha,
                        tau_fast_truncated=args.tau_fast_truncated,
                        n_blr_terms=args.n_blr_terms,
                        model_variant=args.model_variant,
                        shared_blr_lag=args.shared_blr_lag,
                        erlang_order=args.erlang_order,
                        blr_lag_band_scatter=args.blr_lag_band_scatter,
                    )
                elif args.fit_method in ("nuts", "svi+nuts"):
                    if args.fit_method == "svi+nuts":
                        svi_key, mcmc_key = random.split(key)
                        init_values, svi_final_loss = run_svi_warm_start(
                            numpyro_model,
                            svi_key,
                            num_steps=args.svi_steps,
                            learning_rate=args.svi_lr,
                            progress_bar=args.progress,
                        )
                        logging.info(
                            "[%s] Completed SVI warm-start for NUTS with %d steps at lr=%g; final ELBO loss=%s.",
                            oid,
                            args.svi_steps,
                            args.svi_lr,
                            svi_final_loss,
                        )
                        init_strategy = init_to_value(values=init_values)
                    else:
                        mcmc_key = key
                        init_strategy = numpyro.infer.init_to_median()
                    nuts = NUTS(
                        numpyro_model,
                        init_strategy=init_strategy,
                        dense_mass=args.dense_mass,
                        max_tree_depth=args.max_tree_depth,
                        target_accept_prob=args.target_accept,
                    )
                    mcmc = MCMC(
                        nuts,
                        num_warmup=args.nwarm,
                        num_samples=args.nsamp,
                        num_chains=max(1, args.nchains),
                        chain_method=chain_method,
                        progress_bar=args.progress,
                    )
                    mcmc.run(mcmc_key)
                    samples_flat = mcmc.get_samples(group_by_chain=False)
                    samples_per_chain = mcmc.get_samples(group_by_chain=True)
                    samples_flat = tree_map(lambda x: np.asarray(device_get(x)), samples_flat)
                    samples_per_chain = tree_map(lambda x: np.asarray(device_get(x)), samples_per_chain)
                    obj_flat_samples = samples_flat
                else:
                    ns_constructor_kwargs = {}
                    ns_termination_kwargs = {}
                    (
                        aligned_num_live_points,
                        requested_num_live_points,
                        ns_device_count,
                    ) = align_nested_sampler_num_live_points(
                        numpyro_model,
                        random.fold_in(key, 11),
                        requested_num_live_points=args.ns_num_live_points,
                    )
                    if aligned_num_live_points != requested_num_live_points:
                        logging.info(
                            "[%s] Adjusting nested-sampler num_live_points from %d to %d "
                            "to match %d active JAX devices.",
                            oid,
                            requested_num_live_points,
                            aligned_num_live_points,
                            ns_device_count,
                        )
                    ns_constructor_kwargs["num_live_points"] = aligned_num_live_points
                    if args.ns_max_samples is not None:
                        ns_constructor_kwargs["max_samples"] = args.ns_max_samples
                    ns_termination_kwargs["dlogZ"] = args.ns_dlogz

                    ns = NestedSampler(
                        numpyro_model,
                        constructor_kwargs=ns_constructor_kwargs,
                        termination_kwargs=ns_termination_kwargs,
                    )
                    ns.run(key)
                    samples_flat = ns.get_samples(
                        random.fold_in(key, 1),
                        num_samples=args.nsamp,
                        group_by_chain=False,
                    )
                    samples_per_chain = ns.get_samples(
                        random.fold_in(key, 2),
                        num_samples=args.nsamp,
                        group_by_chain=True,
                    )
                    samples_flat = tree_map(lambda x: np.asarray(device_get(x)), samples_flat)
                    samples_per_chain = tree_map(lambda x: np.asarray(device_get(x)), samples_per_chain)
                    obj_flat_samples = samples_flat

            print_light_curve_posterior_summary(
                oid,
                samples_per_chain=samples_per_chain,
                flat_samples=obj_flat_samples,
            )

            obj_flat_samples = add_model_prediction_params(
                obj_flat_samples,
                lam_rf,
                model_variant=args.model_variant,
                lam_lya_rf=lam_lya_rf,
            )

            log_nonfinite_sample_summary(obj_flat_samples, label=oid)

            obj_flat_samples_flatten_per_band = flatten_flat_samples_per_band(
                obj_flat_samples,
                bands=bands,
                survey_names=obj["survey_names"],
            )
            log_nonfinite_sample_summary(obj_flat_samples_flatten_per_band, label=f"{oid} per-band")

            diagnostics = {}
            if samples_per_chain is not None and args.fit_method in ("nuts", "two_stage_nuts"):
                obj_samples_per_chain_flatten_per_band = flatten_per_chain_samples_per_band(
                    samples_per_chain,
                    bands=bands,
                    survey_names=obj["survey_names"],
                )
                diagnostics = diagnostics_for_per_chain_samples(obj_samples_per_chain_flatten_per_band)
            diagnostics |= stage_diagnostics

            if args.model_variant in ("mag_flux_linearized", "mag_flux_linearized_erlang"):
                fit_obj_for_display = flux_linearized_fit_obj if flux_linearized_fit_obj is not None else obj
                y_relflux_display, yerr_relflux_display = (
                    (
                        fit_obj_for_display["y_relflux_fit"],
                        fit_obj_for_display["yerr_relflux_fit"],
                    )
                    if "y_relflux_fit" in fit_obj_for_display and "yerr_relflux_fit" in fit_obj_for_display
                    else (
                        mag_residual_to_relative_flux(obj["y"]),
                        magerr_residual_to_relative_fluxerr(obj["y"], obj["yerr"]),
                    )
                )
                display_factory = (
                    make_multiband_dho_blr_flux_linearized_erlang_model
                    if args.model_variant == "mag_flux_linearized_erlang"
                    else make_multiband_dho_blr_flux_linearized_model
                )
                m = display_factory(
                    obj["X"],
                    y_relflux_display,
                    yerr_relflux_display,
                    n_band=B,
                    survey_idx=obj["survey_idx"],
                    baseline_flux_by_band=reference_flux_from_mean_magnitudes(obj["mags_means"]),
                    zero_mean=zero_mean,
                    has_jitter=has_jitter,
                )
            elif args.model_variant == "mag_fluxmix_fast":
                m = build_mag_fluxmix_fast_display_model(
                    obj,
                    lam_rf,
                    obj_flat_samples,
                )
            else:
                m = make_multiband_dho_blr_model(
                    obj["X"],
                    obj["y"],
                    obj["yerr"],
                    n_band=B,
                    survey_idx=obj["survey_idx"],
                    zero_mean=zero_mean,
                    has_jitter=has_jitter,
                )
            plot_samples = obj_flat_samples

            loo_residual_result = compute_loo_short_lag_residual_diagnostics(
                m,
                plot_samples,
                obj,
                bands,
            )
            if args.save_sample_file:
                save_obj_samples_to_hdf5(
                    obj_flat_samples,
                    oid,
                    scalar_diagnostics={
                        "loo_chi2_eff": loo_residual_result["loo_chi2_eff"],
                        "loo_rms": loo_residual_result["loo_rms"],
                    },
                )

            result = process_samples(
                obj_flat_samples_flatten_per_band,
                obj,
                bands=bands,
            )
            adf_result = compute_object_adf_diagnostics(
                obj_flat_samples_flatten_per_band,
                obj,
                bands=bands,
            )
            drift_result = compute_g_band_residual_drift_diagnostics(
                obj_flat_samples_flatten_per_band,
                obj,
                bands,
                z=float(obj["z"]),
            )
            raw_drift_result = compute_g_band_raw_drift_diagnostics(
                obj_flat_samples_flatten_per_band,
                obj,
                bands,
                z=float(obj["z"]),
            )
            psd_break_result = compute_lomb_scargle_break_diagnostics(
                m,
                plot_samples,
                obj,
                float(obj["z"]),
            )
            sf_result = compute_structure_function_diagnostics(
                obj_flat_samples_flatten_per_band,
                obj,
                float(obj["z"]),
            )
            kl_result = compute_parameter_kls(
                obj_flat_samples_flatten_per_band,
                bands=bands,
                survey_names=obj["survey_names"],
                t_ref=np.asarray(obj["X"][0], dtype=float),
                z=float(obj["z"]),
                lambda_center_rf=float(lambda_center_rf),
                log_jitter_mean=np.asarray(log_jitter_mean_fit),
                model_variant=args.model_variant,
                disable_linear_trend=args.disable_linear_trend,
                disable_lag_blr=args.disable_lag_blr,
                disable_lag_bc=args.disable_lag_bc,
                drop_band_lyman_alpha=args.drop_band_lyman_alpha,
                tau_fast_truncated=args.tau_fast_truncated,
                n_blr_terms=args.n_blr_terms,
            )

            if args.plot:
                try:
                    plot_data = result | psd_break_result
                    if not args.disable_trace_plot:
                        plot_mcmc_traces(obj_flat_samples_flatten_per_band, obj)
                    if not args.disable_combined_plot:
                        save_combined_plot(
                            plot_samples,
                            m,
                            obj["X"],
                            obj["y"],
                            obj["yerr"],
                            obj["band_idx"],
                            obj["mags_means"],
                            obj["survey_times"],
                            plot_data,
                            time0=obj["time0"],
                            bands=bands,
                            plot_psd=(not args.disable_plot_psd),
                            plot_bpl_fit=args.plot_ls_broken_pl,
                            show_combined_light_curve_component_overlay=(
                                args.show_combined_light_curve_component_overlay
                            ),
                        )
                    if not args.disable_color_magnitude_plot:
                        save_color_magnitude_plot(
                            plot_samples,
                            m,
                            obj["X"],
                            obj["y"],
                            obj["yerr"],
                            obj["band_idx"],
                            obj["mags_means"],
                            plot_data,
                            time0=obj["time0"],
                            bands=bands,
                        )
                    drift_plot_result = compute_g_band_residual_drift_diagnostics(
                        obj_flat_samples_flatten_per_band,
                        obj,
                        bands,
                        z=float(obj["z"]),
                        return_series=True,
                    )
                    save_g_band_binned_residual_drift_plot(
                        drift_plot_result,
                        obj | dict(prefix=prefix, suffix=suffix),
                    )
                    sf_plot_result = compute_structure_function_diagnostics(
                        obj_flat_samples_flatten_per_band,
                        obj,
                        float(obj["z"]),
                        return_series=True,
                    )
                    save_structure_function_plot(
                        sf_plot_result,
                        obj | dict(prefix=prefix, suffix=suffix),
                    )
                    normality_plot_result = compute_multiband_residual_normality_diagnostics(
                        obj_flat_samples_flatten_per_band,
                        obj,
                        bands,
                        z=float(obj["z"]),
                        return_series=True,
                    )
                    save_multiband_residual_normality_plot(
                        normality_plot_result,
                        obj | dict(prefix=prefix, suffix=suffix, bands=bands),
                    )
                    save_dm_df_over_f_distribution_plot(
                        obj | dict(prefix=prefix, suffix=suffix, bands=bands),
                    )
                    if not args.disable_correlation_plot:
                        plot_correlation_matrix(obj_flat_samples_flatten_per_band, obj)
                    if not args.disable_histogram_plot:
                        plot_all_histograms(obj_flat_samples_flatten_per_band, obj)
                    if not args.disable_corner_plot:
                        plot_posterior(
                            obj_flat_samples_flatten_per_band,
                            obj,
                            sample_mode=args.corner_plot_mode,
                        )
                except Exception as e:
                    logging.error(f"[{oid}] Plotting error: {e}")
                    logging.error(traceback.format_exc())

            final_result = obj | result | adf_result | drift_result | raw_drift_result | psd_break_result | sf_result | kl_result | loo_residual_result | diagnostics | dict(prefix=prefix, suffix=suffix, model_variant=args.model_variant)
            log_sigma_uv = final_result.get("log_sigma_uv")
            log_sigma_uv_err = final_result.get("log_sigma_uv_err")
            log_tau_uv_rf = final_result.get("log_tau_uv_rf")
            log_tau_uv_rf_err = final_result.get("log_tau_uv_rf_err")
            log_sigma_uv_bpl = final_result.get("log_sigma_uv_bpl")
            log_tau_uv_rf_bpl = final_result.get("log_tau_uv_rf_bpl")
            log_sigma_uv_sf = final_result.get("log_sigma_uv_sf")
            log_tau_uv_rf_sf = final_result.get("log_tau_uv_rf_sf")
            sigma_ls = final_result.get("sigma_ls")
            tau_ls = final_result.get("tau_ls")
            print(
                f"[{oid}] log_sigma_uv = {log_sigma_uv} ± {log_sigma_uv_err} ; "
                f"log_tau_uv_rf = {log_tau_uv_rf} ± {log_tau_uv_rf_err} ; "
                f"log_sigma_uv_bpl = {log_sigma_uv_bpl} ; "
                f"log_tau_uv_rf_bpl = {log_tau_uv_rf_bpl} ; "
                f"log_sigma_uv_sf = {log_sigma_uv_sf} ; "
                f"log_tau_uv_rf_sf = {log_tau_uv_rf_sf} ; "
                f"sigma_ls = {sigma_ls} ; "
                f"tau_ls = {tau_ls}"
            )

            results.append(final_result)

            if args.inject_fake:
                compare_pairs = [
                    ("log_tau_fake", "log_tau_uv", "log10_tau"),
                    ("log_sigma_fake", "log_sigma_uv", "log10_sigma"),
                ]
                summarize_fake_true_vs_recovered(final_result, diagnostics, compare_pairs=compare_pairs)

        except Exception as e:
            logging.error(f"[{oid}] Error during fit: {e}")
            logging.error(traceback.format_exc())
            continue

    save_quasar_list_hdf5(results, ignored_keys=["X", "y", "yerr", "band_idx"])

    if not args.disable_sigma_tau_lambda_plot:
        try:
            plot_sigma_tau_vs_lambda_with_model(
                results,
                inject_fake=args.inject_fake,
            )
        except Exception as e:
            logging.error(f"plot_sigma_tau_vs_lambda_with_model error: {e}")
            logging.error(traceback.format_exc())

    if args.inject_fake and not args.disable_recovery_plot:
        try:
            plot_recovery(results)
        except Exception as e:
            logging.error(f"plot_recovery error: {e}")
            logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
