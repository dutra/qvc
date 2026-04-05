#!/usr/bin/env python3
"""Single-object multiband light-curve fitter using the DHO+BC+BLR model."""

import os
import sys
import argparse
import logging
import traceback

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import curve_fit, least_squares
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
numpyro.enable_validation(True)
import numpyro.distributions as dist
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
from qvc.light_curve.multiband_fit_utils import *
from qvc.light_curve.multiband_generate_lc import *
from qvc.light_curve.multiband_model_dho_blr import make_multiband_dho_blr_model
from qvc.light_curve.psf_constant_flux_correction import (
    apply_constant_flux_correction_to_objects,
    print_constant_flux_correction_summary,
)
from qvc.light_curve.variability_metrics import compute_variability_metrics_for_cleaned_lc


zero_mean = False
has_jitter = True
LYA_REST_WAVELENGTH = 1216.0
LYA_ATTENUATION_WIDTH = 150.0
BALMER_EDGE_REST_WAVELENGTH = 3646.0
BALMER_EDGE_ATTENUATION_WIDTH = 250.0
ETA_SIGMA_LOW = -5.0
LOG_AMP_DELTA_LYA_LOW = -5.0
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


def compute_lambda_center_rf(lam_rf):
    """Geometric-mean rest wavelength of the kept bands for one object."""

    lam_rf = jnp.asarray(lam_rf)
    lam_rf = jnp.maximum(lam_rf, jnp.array(1e-12, dtype=lam_rf.dtype))
    return jnp.exp(jnp.mean(jnp.log(lam_rf)))


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


def run_svi_warm_start(model, rng_key, *, num_steps, learning_rate):
    """Run AutoNormal SVI and return guide-median init values for NUTS."""

    guide = AutoNormal(model)
    svi = SVI(model, guide, Adam(learning_rate), Trace_ELBO())
    svi_state = svi.init(rng_key)

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


def lya_variability_weight(lam_rf, transition=LYA_REST_WAVELENGTH, width=LYA_ATTENUATION_WIDTH):
    """Smooth weight that approaches 1 blueward of Lyα and 0 redward of it."""

    lam_rf = jnp.asarray(lam_rf)
    return 1.0 / (1.0 + jnp.exp((lam_rf - transition) / width))


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


def posterior_median_mean_function(flat_samples, t_eval, band, *, t_ref=None):
    """Return the posterior-median fitted mean function for one band."""

    t_eval = np.asarray(t_eval, dtype=float)
    if t_ref is None:
        t_ref = t_eval
    t_ref = np.asarray(t_ref, dtype=float)
    mean_key = f"mean_{band}"
    mean_level = float(np.nanmedian(np.asarray(flat_samples[mean_key], dtype=float))) if mean_key in flat_samples else 0.0
    poly1 = float(np.nanmedian(np.asarray(flat_samples["poly1"], dtype=float))) if "poly1" in flat_samples else 0.0
    t_center, t_std = linear_mean_time_scaling(t_ref)
    time_scaled = (t_eval - t_center) / t_std
    return mean_level + poly1 * time_scaled


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

    out = {}
    pvalues = []
    for i, band in enumerate(bands):
        mask = band_idx == i
        t_band = t_all[mask]
        y_band = y_all[mask]
        fitted_mean = posterior_median_mean_function(flat_samples, t_band, band, t_ref=t_all)
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


def extract_band_detrended_series(flat_samples, obj, bands, band, *, z=None, subtract_mean=True):
    """Return observed/rest-frame times, values, and errors for one band."""

    if band not in bands:
        raise KeyError(f"Missing {band} band required for residual diagnostics.")

    t_all = np.asarray(obj["X"][0], dtype=float)
    y_all = np.asarray(obj["y"], dtype=float)
    yerr_all = np.asarray(obj.get("yerr", np.full_like(y_all, np.nan)), dtype=float)
    band_idx = np.asarray(obj["band_idx"])

    i = bands.index(band)
    mask = band_idx == i
    t_band = t_all[mask]
    y_band = y_all[mask]
    yerr_band = yerr_all[mask]
    if subtract_mean:
        fitted_mean = posterior_median_mean_function(flat_samples, t_band, band, t_ref=t_all)
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
    """Fit a bending PSD in log-space and return sigma/tau summaries."""

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
        err_lo = np.clip(log_power - np.log10(np.clip(lo, 1e-300, None)), 1e-3, None)
        err_hi = np.clip(np.log10(np.clip(hi, 1e-300, None)) - log_power, 1e-3, None)
        log_err = 0.5 * (err_lo + err_hi)
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
    noise_floor_init = np.clip(np.percentile(power_fit, 10), 1e-12, None)

    alpha_high_init = -2.0

    def model_log10(freq_val, log_sigma, log_tau, log_noise_floor, alpha_high):
        psd = bending_power_law_psd(freq_val, log_sigma, log_tau, log_noise_floor, alpha_high)
        return np.log10(np.clip(psd, 1e-300, None))

    try:
        popt, pcov = curve_fit(
            model_log10,
            freq_fit,
            log_power,
            p0=(np.log10(sigma_init), np.log10(tau_init), np.log10(noise_floor_init), alpha_high_init),
            sigma=log_err,
            absolute_sigma=True,
            bounds=([-6.0, -1.0, -12.0, -2.5], [3.0, 6.0, 8.0, -1.5]),
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
    near_slope_lower_bound = np.isclose(float(popt[3]), -2.5, atol=0.03)
    near_slope_upper_bound = np.isclose(float(popt[3]), -1.5, atol=0.03)
    turnover_bracketed = np.isfinite(tau_char) and (tau_min < tau_char < tau_max)

    return {
        "log_sigma_bpl": float(popt[0]),
        "log_sigma_bpl_err": float(perr[0]) if np.all(np.isfinite(perr)) else np.nan,
        "log_tau_bpl": float(popt[1]),
        "log_tau_bpl_err": float(perr[1]) if np.all(np.isfinite(perr)) else np.nan,
        "log_noise_floor_bpl": float(popt[2]),
        "log_noise_floor_bpl_err": float(perr[2]) if np.all(np.isfinite(perr)) else np.nan,
        "psd_bpl_alpha_high": float(popt[3]),
        "psd_bpl_alpha_high_err": float(perr[3]) if np.all(np.isfinite(perr)) else np.nan,
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

    bands = list(obj["bands"])
    lam_rf = np.asarray([lambda_pivot[band] / (1.0 + float(z)) for band in bands], dtype=float)
    ref_idx = int(np.argmin(np.abs(lam_rf - 2500.0)))
    ref_band = bands[ref_idx]
    lam_ref_band = float(lam_rf[ref_idx])
    posterior_median = {k: np.median(v, axis=0) for k, v in samples.items()}
    freqs = np.logspace(-6, 2, n_freq)

    f_bin_norm, p_bin_norm, p_lo_norm, p_hi_norm, counts_norm, p_noise_norm = combined_lomb_scargle_from_model(
        model,
        obj["y"],
        obj["yerr"],
        posterior_median,
        2.0 * np.pi * freqs,
        amp_scaling_mode="absolute_gp_normalized",
    )
    f_bin_raw, p_bin_raw, p_lo_raw, p_hi_raw, counts_raw, p_noise_raw = combined_lomb_scargle_from_model(
        model,
        obj["y"],
        obj["yerr"],
        posterior_median,
        2.0 * np.pi * freqs,
        amp_scaling_mode="relative_to_2500",
        band_wavelength_rf=lam_rf,
    )

    model_psd = (2.0 * np.pi) * np.asarray(
        model.psd(
            {k: jnp.array(v) for k, v in posterior_median.items()},
            2.0 * np.pi * freqs,
            b=0,
            sigma_n2=0.0,
        )
    )
    if f_bin_norm.size > 0 and p_bin_norm.size > 0 and np.all(np.isfinite(model_psd)):
        model_at_f0 = float(np.interp(f_bin_norm[0], freqs, model_psd))
        scale = model_at_f0 / max(float(p_bin_norm[0]), 1e-30)
        p_bin_fit_norm = p_bin_norm * scale
        p_lo_fit_norm = p_lo_norm * scale
        p_hi_fit_norm = p_hi_norm * scale
        p_noise_fit_norm = float(p_noise_norm) * scale if np.isfinite(p_noise_norm) else np.nan
    else:
        p_bin_fit_norm = p_bin_norm
        p_lo_fit_norm = p_lo_norm
        p_hi_fit_norm = p_hi_norm
        p_noise_fit_norm = float(p_noise_norm) if np.isfinite(p_noise_norm) else np.nan

    fit_norm = fit_bending_power_law_psd(f_bin_norm, p_bin_fit_norm, p_lo_fit_norm, p_hi_fit_norm)
    fit_raw = fit_bending_power_law_psd(f_bin_raw, p_bin_raw, p_lo_raw, p_hi_raw)
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
        "psd_noise_floor": p_noise_fit_norm,
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
        "psd_noise_floor_ls": float(p_noise_raw) if np.isfinite(p_noise_raw) else np.nan,
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


def _posterior_median_band_jitter(samples, band):
    """Return posterior-median white-noise jitter in linear mag units for one band."""

    jitter_key = f"log_jitter_{band}"
    if jitter_key not in samples:
        return 0.0
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

    band_payloads = {}
    sigma_ref = np.nan
    for i_band, band in enumerate(bands):
        mask = band_idx == i_band
        if np.count_nonzero(mask) < 2:
            continue
        yerr_eff = np.hypot(yerr_all[mask], _posterior_median_band_jitter(samples, band))
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
    """Analytic SF of the continuum-only overdamped-SHO process."""

    tau = np.asarray(tau, dtype=float)
    amp = np.asarray(amp, dtype=float)
    variance_factor = dho_stationary_variance_factor(tau_fast, tau_slow)
    tau_fast_ord, tau_slow_ord = ordered_dho_taus(tau_fast, tau_slow)
    denom = np.maximum(tau_slow_ord - tau_fast_ord, 1e-12)
    c_fast = -tau_fast_ord / denom
    c_slow = tau_slow_ord / denom
    sf2 = 2.0 * np.square(amp) * (
        np.square(c_fast) * (1.0 - np.exp(-tau / tau_fast_ord))
        + np.square(c_slow) * (1.0 - np.exp(-tau / tau_slow_ord))
    )
    sf2 = np.where(np.isfinite(variance_factor), sf2, np.nan)
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
    jitter_ref_band = _posterior_median_band_jitter(samples, ref_band)
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


def log_tau_fast_center0_prior(log_tau_slow_center0, *, tau_fast_truncated=False):
    mean = jnp.asarray(log_tau_slow_center0) - jnp.log(10.0)
    sigma = 0.5 * jnp.log(10.0)
    if tau_fast_truncated:
        return dist.TruncatedNormal(
            mean,
            sigma,
            high=jnp.asarray(log_tau_slow_center0),
        )
    return dist.Normal(mean, sigma)


def poly1_prior():
    return dist.Normal(0.0, 0.1)


def lag0_prior():
    return dist.TruncatedNormal(5.0, 5.0, low=0.0, high=LAG0_HIGH)


def lag_beta_prior():
    return dist.TruncatedNormal(4.0 / 3.0, 0.2, low=0.0, high=LAG_BETA_HIGH)


def log_amp_delta_lya_prior():
    return dist.TruncatedNormal(-0.5, 0.75, low=LOG_AMP_DELTA_LYA_LOW, high=0.0)


def mean_prior():
    return dist.Normal(0.0, 0.2)


def log_jitter_prior(log_jitter_mean):
    return dist.Normal(log_jitter_mean, 1.0)


def log_amp_delta_blr_prior():
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


def log_amp_delta_bc_prior():
    return dist.Normal(-1.0, 1.0)


def log_lag_ratio_bc_to_blr_prior():
    return dist.TruncatedNormal(
        loc=jnp.log(0.2),
        scale=0.15,
        low=LOG_LAG_RATIO_BC_TO_BLR_LOW,
        high=LOG_LAG_RATIO_BC_TO_BLR_HIGH,
    )


def compute_parameter_kls(
    flat_samples,
    *,
    bands,
    z,
    lambda_center_rf,
    log_jitter_mean,
    disable_poly1=False,
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

    kls["eta_sigma_kl"] = kl_from_samples(
        eta_sigma,
        lambda x: _dist_log_prob_array(eta_sigma_prior(), x),
    )
    kls["eta_tau_kl"] = kl_from_samples(
        eta_tau,
        lambda x: _dist_log_prob_array(eta_tau_prior(), x),
    )

    if "log_sigma_center0" in flat_samples:
        kls["log_sigma_center0_kl"] = conditional_kl_from_samples(
            flat_samples["log_sigma_center0"],
            lambda x, eta: _dist_log_prob_array(
                log_sigma_center0_prior(
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

    if not disable_poly1 and "poly1" in flat_samples:
        kls["poly1_kl"] = kl_from_samples(
            flat_samples["poly1"],
            lambda x: _dist_log_prob_array(poly1_prior(), x),
        )

    if "lag0" in flat_samples:
        kls["lag0_kl"] = kl_from_samples(
            flat_samples["lag0"],
            lambda x: _dist_log_prob_array(lag0_prior(), x),
        )
    if "lag_beta" in flat_samples:
        kls["lag_beta_kl"] = kl_from_samples(
            flat_samples["lag_beta"],
            lambda x: _dist_log_prob_array(lag_beta_prior(), x),
        )

    if not drop_band_lyman_alpha and "log_amp_delta_lya" in flat_samples:
        kls["log_amp_delta_lya_kl"] = kl_from_samples(
            flat_samples["log_amp_delta_lya"],
            lambda x: _dist_log_prob_array(log_amp_delta_lya_prior(), x),
        )

    if not disable_lag_blr and not disable_lag_bc:
        if "log_amp_delta_bc" in flat_samples:
            kls["log_amp_delta_bc_kl"] = kl_from_samples(
                flat_samples["log_amp_delta_bc"],
                lambda x: _dist_log_prob_array(log_amp_delta_bc_prior(), x),
            )
        if "log_lag_ratio_bc_to_blr" in flat_samples:
            kls["log_lag_ratio_bc_to_blr_kl"] = kl_from_samples(
                flat_samples["log_lag_ratio_bc_to_blr"],
                lambda x: _dist_log_prob_array(log_lag_ratio_bc_to_blr_prior(), x),
            )

    for i, band in enumerate(bands):
        mean_key = f"mean_{band}"
        if mean_key in flat_samples:
            kls[f"{mean_key}_kl"] = kl_from_samples(
                flat_samples[mean_key],
                lambda x: _dist_log_prob_array(mean_prior(), x),
            )

        jitter_key = f"log_jitter_{band}"
        if jitter_key in flat_samples:
            kls[f"{jitter_key}_kl"] = kl_from_samples(
                flat_samples[jitter_key],
                lambda x: _dist_log_prob_array(log_jitter_prior(float(log_jitter_mean[i])), x),
            )

        if disable_lag_blr:
            continue

        amp_keys = [f"log_amp_delta_blr_{band}"]
        if n_blr_terms >= 2:
            amp_keys.append(f"log_amp_delta_blr2_{band}")
        for amp_key in amp_keys:
            if amp_key in flat_samples:
                kls[f"{amp_key}_kl"] = kl_from_samples(
                    flat_samples[amp_key],
                    lambda x: _dist_log_prob_array(log_amp_delta_blr_prior(), x),
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

    if len(bands) == 0:
        print(f"No usable bands for {data['object_id']}, skipping.", flush=True)
        return None

    all_times = np.concatenate([np.asarray(times[b]) for b in bands])
    all_mags = np.concatenate([np.asarray(mags[b]) for b in bands])
    all_magerrs = np.concatenate([np.asarray(magerrs[b]) for b in bands])
    band_idx = np.concatenate([np.full(len(times[b]), i) for i, b in enumerate(bands)]).astype(
        np.int64, copy=False
    )

    if len(all_times) == 0:
        print(f"No points for {data['object_id']}, skipping.", flush=True)
        return None

    tie_eps = 10.0 * np.finfo(all_times.dtype).eps
    key = all_times + band_idx.astype(all_times.dtype) * tie_eps
    order = np.argsort(key, kind="mergesort")
    all_times, all_mags, all_magerrs, band_idx = (
        all_times[order],
        all_mags[order],
        all_magerrs[order],
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
    all_times, all_mags, all_magerrs, band_idx = (
        all_times[mfin],
        all_mags[mfin],
        all_magerrs[mfin],
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

    all_times, all_mags, all_magerrs, band_idx = (
        all_times[keep],
        all_mags[keep],
        all_magerrs[keep],
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
    X = (jnp.array(all_times) - jnp.min(all_times), jnp.array(band_idx))
    y = jnp.array(all_mags)
    yerr = jnp.array(all_magerrs)

    out = {
        "X": X,
        "time0": time0,
        "y": y,
        "yerr": yerr,
        "band_idx": band_idx,
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


def build_explicit_model_params(raw_params, lam_rf):
    """Convert sampled high-level parameters into explicit model arrays."""

    lam_rf = jnp.asarray(lam_rf)
    lambda_uv = jnp.array(2500.0, dtype=lam_rf.dtype)
    lambda_center_rf = jnp.asarray(
        raw_params.get("lambda_center_rf", compute_lambda_center_rf(lam_rf))
    )

    eta_sigma = jnp.asarray(raw_params["eta_sigma"])
    eta_tau = jnp.asarray(raw_params["eta_tau"])
    log_amp_delta_blr = jnp.asarray(raw_params["log_amp_delta_blr"])
    log_amp_delta_blr2 = jnp.asarray(
        raw_params.get(
            "log_amp_delta_blr2",
            jnp.full_like(log_amp_delta_blr, -1e9),
        )
    )
    log_amp_delta_lya = jnp.asarray(raw_params.get("log_amp_delta_lya", 0.0))
    log_lag_blr = jnp.asarray(raw_params["log_lag_blr"])
    log_lag_blr2 = jnp.asarray(raw_params.get("log_lag_blr2", log_lag_blr))
    has_bc_lag = "log_amp_delta_bc" in raw_params
    if has_bc_lag:
        log_amp_delta_bc = jnp.asarray(raw_params["log_amp_delta_bc"])
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
    log_amp_delta_lya_exp = _expand_last(log_amp_delta_lya)
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
    log_amp_delta_lya_band = log_amp_delta_lya_exp * lya_variability_weight(lam_rf)
    bc_weight = balmer_continuum_weight(lam_rf)

    amp_cont = jnp.exp(log_sigma_band + log_amp_delta_lya_band)
    amp_blr = jnp.exp(log_sigma_uv_exp + log_amp_delta_blr)
    amp_blr2 = jnp.exp(log_sigma_uv_exp + log_amp_delta_blr2)
    lag_disk = lag0_exp * (lam_rf / lambda_center_rf_exp) ** lag_beta_exp
    lag_blr = jnp.exp(log_lag_blr)
    lag_blr2 = jnp.exp(log_lag_blr2)
    if has_bc_lag:
        log_amp_delta_bc_exp = _expand_last(log_amp_delta_bc)
        log_lag_bc_shared = jnp.mean(log_lag_blr, axis=-1) + jnp.asarray(log_lag_ratio_bc_to_blr)
        amp_bc = jnp.exp(log_sigma_uv_exp + log_amp_delta_bc_exp) * bc_weight
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
    explicit["log_amp_delta_lya"] = log_amp_delta_lya
    explicit["log_amp_delta_lya_band"] = log_amp_delta_lya_band
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
        explicit["log_amp_delta_bc"] = log_amp_delta_bc
        explicit["log_lag_ratio_bc_to_blr"] = log_lag_ratio_bc_to_blr
    return explicit


def add_model_prediction_params(samples, lam_rf):
    """Add explicit model parameters needed for prediction/plotting."""

    out = dict(samples)
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
            "log_amp_delta_lya_band",
            "lambda_center_rf",
        )
    ):
        return out

    explicit = build_explicit_model_params(
        out,
        lam_rf,
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
    out["log_amp_delta_lya"] = np.asarray(explicit["log_amp_delta_lya"])
    out["log_amp_delta_lya_band"] = np.asarray(explicit["log_amp_delta_lya_band"])
    if "log_amp_delta_bc" in explicit:
        out["log_amp_delta_bc"] = np.asarray(explicit["log_amp_delta_bc"])
    if "log_lag_ratio_bc_to_blr" in explicit:
        out["log_lag_ratio_bc_to_blr"] = np.asarray(explicit["log_lag_ratio_bc_to_blr"])
    if "log_amp_delta_blr2" in explicit:
        out["log_amp_delta_blr2"] = np.asarray(explicit["log_amp_delta_blr2"])
    if "log_lag_blr2" in explicit:
        out["log_lag_blr2"] = np.asarray(explicit["log_lag_blr2"])
    out["lambda_center_rf"] = np.asarray(explicit["lambda_center_rf"])
    return out


def build_single_object_model(
    obj_dict,
    lam_rf,
    log_jitter_mean,
    *,
    disable_poly1=False,
    disable_lag_blr=False,
    disable_lag_bc=False,
    drop_band_lyman_alpha=False,
    tau_fast_truncated=False,
    n_blr_terms=1,
):
    """Return the NumPyro model for one object."""

    (t, bidx) = obj_dict["X"]
    y = obj_dict["y"]
    yerr = obj_dict["yerr"]
    z = float(obj_dict["z"])
    B = int(len(lam_rf))
    lambda_center_rf = compute_lambda_center_rf(lam_rf)
    lambda_uv = jnp.array(2500.0, dtype=lam_rf.dtype)

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

        if disable_poly1:
            poly1 = numpyro.deterministic("poly1", 0.0)
        else:
            poly1 = numpyro.sample("poly1", poly1_prior())

        lag0 = numpyro.sample("lag0", lag0_prior())
        lag_beta = numpyro.sample("lag_beta", lag_beta_prior())

        if drop_band_lyman_alpha:
            log_amp_delta_lya = numpyro.deterministic("log_amp_delta_lya", 0.0)
        else:
            log_amp_delta_lya = numpyro.sample("log_amp_delta_lya", log_amp_delta_lya_prior())

        if disable_lag_blr or disable_lag_bc:
            log_amp_delta_bc = None
            log_lag_ratio_bc_to_blr = None
        else:
            log_amp_delta_bc = numpyro.sample("log_amp_delta_bc", log_amp_delta_bc_prior())
            log_lag_ratio_bc_to_blr = numpyro.sample(
                "log_lag_ratio_bc_to_blr",
                log_lag_ratio_bc_to_blr_prior(),
            )

        with numpyro.plate("band", B):
            mean = numpyro.sample("mean", mean_prior())

            if disable_lag_blr:
                log_amp_delta_blr = numpyro.deterministic(
                    "log_amp_delta_blr",
                    jnp.full(B, -9.0),
                )
                log_lag_blr = numpyro.deterministic(
                    "log_lag_blr",
                    jnp.full(B, -9.0),
                )
                log_amp_delta_blr2 = numpyro.deterministic(
                    "log_amp_delta_blr2",
                    jnp.full(B, -9.0),
                )
                log_lag_blr2 = numpyro.deterministic(
                    "log_lag_blr2",
                    jnp.full(B, -9.0),
                )
            elif n_blr_terms <= 1:
                log_amp_delta_blr_raw = numpyro.sample("log_amp_delta_blr_raw", log_amp_delta_blr_prior())
                log_lag_blr_raw = numpyro.sample("log_lag_blr_raw", log_lag_blr_prior(z=z))
                log_amp_delta_blr = numpyro.deterministic(
                    "log_amp_delta_blr",
                    log_amp_delta_blr_raw,
                )
                log_lag_blr = numpyro.deterministic(
                    "log_lag_blr",
                    log_lag_blr_raw,
                )
                log_amp_delta_blr2 = numpyro.deterministic(
                    "log_amp_delta_blr2",
                    jnp.full(B, -9.0),
                )
                log_lag_blr2 = numpyro.deterministic(
                    "log_lag_blr2",
                    jnp.full(B, -9.0),
                )
            else:
                log_amp_delta_blr_raw = numpyro.sample("log_amp_delta_blr_raw", log_amp_delta_blr_prior())
                log_lag_blr_raw = numpyro.sample("log_lag_blr_raw", log_lag_blr_prior(z=z))
                log_amp_delta_blr2_raw = numpyro.sample("log_amp_delta_blr2_raw", log_amp_delta_blr_prior())
                log_lag_blr2_raw = numpyro.sample("log_lag_blr2_raw", log_lag_blr_prior(z=z))

                first_is_short = log_lag_blr_raw <= log_lag_blr2_raw
                log_lag_blr = numpyro.deterministic(
                    "log_lag_blr",
                    jnp.where(first_is_short, log_lag_blr_raw, log_lag_blr2_raw),
                )
                log_lag_blr2 = numpyro.deterministic(
                    "log_lag_blr2",
                    jnp.where(first_is_short, log_lag_blr2_raw, log_lag_blr_raw),
                )
                log_amp_delta_blr = numpyro.deterministic(
                    "log_amp_delta_blr",
                    jnp.where(first_is_short, log_amp_delta_blr_raw, log_amp_delta_blr2_raw),
                )
                log_amp_delta_blr2 = numpyro.deterministic(
                    "log_amp_delta_blr2",
                    jnp.where(first_is_short, log_amp_delta_blr2_raw, log_amp_delta_blr_raw),
                )

            log_jitter = numpyro.sample("log_jitter", log_jitter_prior(log_jitter_mean))

        _ = numpyro.deterministic("log_tau_fake", float(obj_dict.get("log_tau_fake", -99.0)))
        _ = numpyro.deterministic("log_sigma_fake", float(obj_dict.get("log_sigma_fake", -99.0)))

        raw_params = dict(
            log_tau_slow_center0=log_tau_slow_center0,
            log_tau_fast_center0=log_tau_fast_center0,
            log_sigma_center0=log_sigma_center0,
            lambda_center_rf=lambda_center_rf,
            poly1=poly1,
            mean=mean,
            log_amp_delta_blr=log_amp_delta_blr,
            log_amp_delta_blr2=log_amp_delta_blr2,
            log_lag_blr=log_lag_blr,
            log_lag_blr2=log_lag_blr2,
            log_jitter=log_jitter,
            lag0=lag0,
            lag_beta=lag_beta,
            log_amp_delta_lya=log_amp_delta_lya,
            eta_sigma=eta_sigma,
            eta_tau=eta_tau,
        )
        if log_amp_delta_bc is not None:
            raw_params["log_amp_delta_bc"] = log_amp_delta_bc
            raw_params["log_lag_ratio_bc_to_blr"] = log_lag_ratio_bc_to_blr

        params = build_explicit_model_params(
            raw_params,
            lam_rf,
        )

        numpyro.deterministic("lambda_center_rf", params["lambda_center_rf"])
        numpyro.deterministic("log_sigma_uv", params["log_sigma_uv"])
        numpyro.deterministic("log_tau_uv", params["log_tau_uv"])
        numpyro.deterministic("log_tau_fast_uv", params["log_tau_fast_uv"])
        log_sigma_hat_uv = params["log_sigma_uv"] - 0.5 * params["log_tau_uv"]
        numpyro.deterministic("log_sigma_hat_uv", log_sigma_hat_uv)
        numpyro.deterministic("log_sigma_hat0", log_sigma_hat_uv)
        numpyro.deterministic("tau_fast", params["tau_fast_band"])
        numpyro.deterministic("tau_slow", params["tau_slow_band"])
        numpyro.deterministic("amp_cont", params["amp_cont"])
        numpyro.deterministic("amp_bc", params["amp_bc"])
        numpyro.deterministic("amp_blr", params["amp_blr"])
        numpyro.deterministic("amp_blr2", params["amp_blr2"])
        numpyro.deterministic("log_amp_delta_lya_band", params["log_amp_delta_lya_band"])
        numpyro.deterministic("lag_disk", params["lag_disk"])
        numpyro.deterministic("lag_bc", params["lag_bc"])
        numpyro.deterministic("lag_blr", params["lag_blr"])
        numpyro.deterministic("lag_blr2", params["lag_blr2"])

        m = make_multiband_dho_blr_model(
            X=(t, bidx),
            y=y,
            yerr=yerr,
            n_band=B,
            zero_mean=zero_mean,
            has_jitter=has_jitter,
        )
        numpyro.factor("loglike", m.log_prob(params))

    return model


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
        choices=("ns", "nuts", "svi+nuts"),
        default="nuts",
        help="Posterior fitting backend: nested sampling ('ns'), HMC/NUTS ('nuts'), or SVI warm-start plus NUTS ('svi+nuts').",
    )
    parser.add_argument("--inject_fake", action="store_true", help="Inject fake light curves.")
    parser.add_argument("--max_tree_depth", type=int, default=8, help="NUTS max tree depth.")
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
    parser.add_argument("--load_sample_file", action="store_true", help="Load saved samples (debug).")
    parser.add_argument("--save_sample_file", dest="save_sample_file", action="store_true", help="Save per-object posterior samples to HDF5.")
    parser.add_argument("--no_save_sample_file", dest="save_sample_file", action="store_false", help="Do not save per-object posterior samples to HDF5.")
    parser.set_defaults(save_sample_file=True)
    parser.add_argument("--disable_poly1", action="store_true", help="Disable trend.")
    parser.add_argument("--rf_length_cut", type=int, default=-1, help="Rest-frame cut (days).")
    parser.add_argument("--exact_same_length", action="store_true", help="Exact same RF length cut.")
    parser.add_argument("--load_stone_lcs", action="store_true", default=False, help="Use Stone LCs.")
    parser.add_argument("--disable_trace_plot", action="store_true", default=False, help="Disable MCMC trace plot.")
    parser.add_argument("--disable_combined_plot", action="store_true", default=False, help="Disable combined light-curve fit plot.")
    parser.add_argument("--disable_color_magnitude_plot", action="store_true", default=False, help="Disable color-magnitude plot.")
    parser.add_argument("--disable_correlation_plot", action="store_true", default=False, help="Disable correlation matrix plot.")
    parser.add_argument("--disable_histogram_plot", action="store_true", default=False, help="Disable posterior histogram plot.")
    parser.add_argument("--disable_corner_plot", action="store_true", default=False, help="Disable corner plot.")
    parser.add_argument("--plot_ls_broken_pl", action="store_true", default=False, help="Overlay the fitted Lomb-Scargle broken power law on the PSD subplot.")
    parser.add_argument(
        "--corner_plot_mode",
        type=str,
        choices=("fast", "full"),
        default="fast",
        help="Corner plot row selection: fast subsampling or full posterior samples.",
    )
    parser.add_argument("--disable_lag_blr", action="store_true", default=False, help="Disable BLR lag model.")
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
            lambda_center_rf = compute_lambda_center_rf(lam_rf)
            print(f"[{oid}] Using bands: {bands}")
            print(f"[{oid}] lam_rf = {lam_rf}")
            print(f"[{oid}] lambda_center_rf = {lambda_center_rf}")

            bidx = obj["band_idx"]
            yerr = np.asarray(obj["yerr"])
            B = len(bands)
            ljm = np.empty(B)
            for i in range(B):
                m = (bidx == i) & np.isfinite(yerr) & (yerr < 10)
                ljm[i] = np.log(np.mean(yerr[m])) if np.any(m) else np.log(1e-3)
            log_jitter_mean = jnp.array(ljm)

            numpyro_model = build_single_object_model(
                obj,
                lam_rf,
                log_jitter_mean=log_jitter_mean,
                disable_poly1=args.disable_poly1,
                disable_lag_blr=args.disable_lag_blr,
                disable_lag_bc=args.disable_lag_bc,
                drop_band_lyman_alpha=args.drop_band_lyman_alpha,
                tau_fast_truncated=args.tau_fast_truncated,
                n_blr_terms=args.n_blr_terms,
            )

            if args.load_sample_file:
                logging.warning("[DEBUG] Loading saved samples (flat) — developer mode.")
                obj_flat_samples = load_obj_samples_from_hdf5(oid)
                samples_per_chain = None
            else:
                key = random.PRNGKey(0)
                key = random.fold_in(key, idx)
                if args.fit_method in ("nuts", "svi+nuts"):
                    if args.fit_method == "svi+nuts":
                        svi_key, mcmc_key = random.split(key)
                        init_values, svi_final_loss = run_svi_warm_start(
                            numpyro_model,
                            svi_key,
                            num_steps=args.svi_steps,
                            learning_rate=args.svi_lr,
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
                        dense_mass=True,
                        max_tree_depth=args.max_tree_depth,
                        target_accept_prob=0.9,
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
                if args.save_sample_file:
                    save_obj_samples_to_hdf5(obj_flat_samples, oid)

            explicit_scalar = build_explicit_model_params(obj_flat_samples, lam_rf)
            for key, explicit_key in (
                ("log_sigma_uv", "log_sigma_uv"),
                ("log_tau_uv", "log_tau_uv"),
                ("log_tau_fast_uv", "log_tau_fast_uv"),
                ("lambda_center_rf", "lambda_center_rf"),
                ("amp_cont", "amp_cont"),
                ("amp_bc", "amp_bc"),
                ("amp_blr", "amp_blr"),
                ("amp_blr2", "amp_blr2"),
                ("lag_disk", "lag_disk"),
                ("lag_bc", "lag_bc"),
                ("lag_blr", "lag_blr"),
                ("lag_blr2", "lag_blr2"),
                ("tau_fast", "tau_fast_band"),
                ("tau_slow", "tau_slow_band"),
                ("log_amp_delta_lya_band", "log_amp_delta_lya_band"),
            ):
                obj_flat_samples[key] = np.asarray(explicit_scalar[explicit_key])

            log_nonfinite_sample_summary(obj_flat_samples, label=oid)

            obj_flat_samples_flatten_per_band = flatten_flat_samples_per_band(
                obj_flat_samples,
                bands=bands,
            )
            log_nonfinite_sample_summary(obj_flat_samples_flatten_per_band, label=f"{oid} per-band")

            diagnostics = {}
            if samples_per_chain is not None and args.fit_method == "nuts":
                obj_samples_per_chain_flatten_per_band = flatten_per_chain_samples_per_band(
                    samples_per_chain,
                    bands=bands,
                )
                diagnostics = diagnostics_for_per_chain_samples(obj_samples_per_chain_flatten_per_band)

            m = make_multiband_dho_blr_model(
                obj["X"],
                obj["y"],
                obj["yerr"],
                n_band=B,
                zero_mean=zero_mean,
                has_jitter=has_jitter,
            )
            plot_samples = add_model_prediction_params(
                obj_flat_samples,
                lam_rf,
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
                z=float(obj["z"]),
                lambda_center_rf=float(lambda_center_rf),
                log_jitter_mean=np.asarray(log_jitter_mean),
                disable_poly1=args.disable_poly1,
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

            final_result = obj | result | adf_result | drift_result | raw_drift_result | psd_break_result | sf_result | kl_result | dict(prefix=prefix, suffix=suffix) 
            # final_result |= diagnostics
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
