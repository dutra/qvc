#!/usr/bin/env python3
"""Single-object multiband light-curve fitter using the DHO+BLR model."""

import os
import sys
import argparse
import logging
import traceback

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation
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
from jax import random, device_get
from jax.tree_util import tree_map

import numpyro

numpyro.set_host_device_count(num_cores)
numpyro.enable_x64()
numpyro.enable_validation(True)
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from qvc.light_curve.multiband_fit_plotting import *
from qvc.light_curve.multiband_fit_utils import *
from qvc.light_curve.multiband_generate_lc import *
from qvc.light_curve.multiband_model_dho_blr import make_multiband_dho_blr_model
from qvc.light_curve.variability_metrics import compute_variability_metrics_for_cleaned_lc


zero_mean = False
has_jitter = True
LYA_REST_WAVELENGTH = 1216.0
LYA_ATTENUATION_WIDTH = 150.0
ETA_SIGMA_LOW = -5.0
LOG_AMP_DELTA_LYA_LOW = -5.0
LAG0_HIGH = 100.0
LAG_BETA_HIGH = 5.0


def compute_lambda_center_rf(lam_rf):
    """Geometric-mean rest wavelength of the kept bands for one object."""

    lam_rf = jnp.asarray(lam_rf)
    lam_rf = jnp.maximum(lam_rf, jnp.array(1e-12, dtype=lam_rf.dtype))
    return jnp.exp(jnp.mean(jnp.log(lam_rf)))


def _expand_last(x):
    x = jnp.asarray(x)
    return jnp.expand_dims(x, axis=-1) if x.ndim > 0 else x


def lya_variability_weight(lam_rf, transition=LYA_REST_WAVELENGTH, width=LYA_ATTENUATION_WIDTH):
    """Smooth weight that approaches 1 blueward of Lyα and 0 redward of it."""

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


def bending_power_law_psd(freq, log_sigma, log_tau, log_noise_floor=-99.0):
    """Single-break PSD with flat low-frequency slope, -2 high-frequency slope, and white-noise floor."""

    freq = np.asarray(freq, dtype=float)
    sigma = np.power(10.0, float(log_sigma))
    tau = np.power(10.0, float(log_tau))
    denom = 1.0 + np.square(2.0 * np.pi * freq * tau)
    noise_floor = np.power(10.0, float(log_noise_floor))
    return 2.0 * sigma * sigma * tau / denom + noise_floor


def fit_bending_power_law_psd(freq, power, power_lo=None, power_hi=None):
    """Fit a DRW-like bending PSD in log-space and return sigma/tau summaries."""

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

    def model_log10(freq_val, log_sigma, log_tau, log_noise_floor):
        psd = bending_power_law_psd(freq_val, log_sigma, log_tau, log_noise_floor)
        return np.log10(np.clip(psd, 1e-300, None))

    try:
        popt, pcov = curve_fit(
            model_log10,
            freq_fit,
            log_power,
            p0=(np.log10(sigma_init), np.log10(tau_init), np.log10(noise_floor_init)),
            sigma=log_err,
            absolute_sigma=True,
            bounds=([-6.0, -1.0, -12.0], [3.0, 6.0, 8.0]),
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
            "psd_bpl_valid": False,
            "psd_bpl_nbins": float(freq_fit.size),
        }

    return {
        "log_sigma_bpl": float(popt[0]),
        "log_sigma_bpl_err": float(perr[0]) if np.all(np.isfinite(perr)) else np.nan,
        "log_tau_bpl": float(popt[1]),
        "log_tau_bpl_err": float(perr[1]) if np.all(np.isfinite(perr)) else np.nan,
        "log_noise_floor_bpl": float(popt[2]),
        "log_noise_floor_bpl_err": float(perr[2]) if np.all(np.isfinite(perr)) else np.nan,
        "psd_bpl_valid": True,
        "psd_bpl_nbins": float(freq_fit.size),
    }


def compute_lomb_scargle_break_diagnostics(model, samples, obj, z, *, n_freq=500):
    """Fit a bending power law to the band nearest rest-frame 2500 A and convert to UV."""

    bands = list(obj["bands"])
    lam_rf = np.asarray([lambda_pivot[band] / (1.0 + float(z)) for band in bands], dtype=float)
    ref_idx = int(np.argmin(np.abs(lam_rf - 2500.0)))
    ref_band = bands[ref_idx]
    lam_ref_band = float(lam_rf[ref_idx])
    posterior_median = {k: np.median(v, axis=0) for k, v in samples.items()}
    freqs = np.logspace(-6, 2, n_freq)
    f_bin, p_bin, p_lo, p_hi, counts, p_noise = combined_lomb_scargle_from_model(
        model,
        obj["y"],
        obj["yerr"],
        posterior_median,
        2.0 * np.pi * freqs,
        amp_reference="selected_band",
        #selected_band=ref_idx,
    )
    fit = fit_bending_power_law_psd(f_bin, p_bin, p_lo, p_hi)
    eta_sigma = float(np.nanmedian(np.asarray(samples["eta_sigma"], dtype=float)))
    eta_tau = float(np.nanmedian(np.asarray(samples["eta_tau"], dtype=float)))
    log_sigma_uv = (
        fit["log_sigma_bpl"] + log_single_pl(2500.0, lam_ref_band, eta_sigma)
        if np.isfinite(fit["log_sigma_bpl"]) else np.nan
    )
    log_tau_uv_obs = (
        fit["log_tau_bpl"] + log_single_pl(2500.0, lam_ref_band, eta_tau)
        if np.isfinite(fit["log_tau_bpl"]) else np.nan
    )
    log_tau_rf = log_tau_uv_obs - np.log10(1.0 + float(z)) if np.isfinite(log_tau_uv_obs) else np.nan
    log_tau_rf_err = fit["log_tau_bpl_err"]

    out = {
        "psd_bpl_ref_band": ref_band,
        "psd_bpl_ref_lambda_rf": lam_ref_band,
        "log_sigma_bpl_ref_band": fit["log_sigma_bpl"],
        "log_sigma_bpl_ref_band_err": fit["log_sigma_bpl_err"],
        "log_tau_bpl_ref_band": fit["log_tau_bpl"],
        "log_tau_bpl_ref_band_err": fit["log_tau_bpl_err"],
        "log_sigma_uv_bpl": log_sigma_uv,
        "log_sigma_uv_bpl_err": fit["log_sigma_bpl_err"],
        "log_tau_uv_bpl": log_tau_uv_obs,
        "log_tau_uv_bpl_err": fit["log_tau_bpl_err"],
        "log_noise_floor_bpl": fit["log_noise_floor_bpl"],
        "log_noise_floor_bpl_err": fit["log_noise_floor_bpl_err"],
        "log_tau_uv_rf_bpl": log_tau_rf,
        "log_tau_uv_rf_bpl_err": log_tau_rf_err,
        "psd_bpl_valid": fit["psd_bpl_valid"],
        "psd_bpl_nbins": fit["psd_bpl_nbins"],
        "psd_noise_floor": float(p_noise) if np.isfinite(p_noise) else np.nan,
    }
    if np.isfinite(fit["log_tau_bpl"]):
        out["log_nu_break_bpl"] = -np.log10(2.0 * np.pi) - fit["log_tau_bpl"]
    else:
        out["log_nu_break_bpl"] = np.nan
    return out


def empirical_structure_function(t, y, yerr=None, *, bins_per_decade=3, min_pairs=8):
    """Compute a binned empirical structure function from one band."""

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.zeros_like(y) if yerr is None else np.asarray(yerr, dtype=float)

    n = t.size
    if n < 3:
        return np.array([]), np.array([]), np.array([]), np.array([])

    dt = np.abs(t[:, None] - t[None, :])
    dy2 = np.square(y[:, None] - y[None, :])
    noise2 = np.square(yerr[:, None]) + np.square(yerr[None, :])

    iu = np.triu_indices(n, k=1)
    tau = dt[iu]
    sf2 = np.maximum(dy2[iu] - noise2[iu], 0.0)
    good = np.isfinite(tau) & np.isfinite(sf2) & (tau > 0.0)
    if np.count_nonzero(good) < min_pairs:
        return np.array([]), np.array([]), np.array([]), np.array([])

    tau = tau[good]
    sf2 = sf2[good]
    tmin = np.min(tau)
    tmax = np.max(tau)
    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
        return np.array([]), np.array([]), np.array([]), np.array([])

    decades = np.log10(tmax) - np.log10(tmin)
    n_bins = max(1, int(np.ceil(bins_per_decade * decades)))
    edges = np.logspace(np.log10(tmin), np.log10(tmax), n_bins + 1)
    which = np.digitize(tau, edges) - 1

    tau_bin, sf_bin, sf_lo, sf_hi = [], [], [], []
    for k in range(n_bins):
        sel = which == k
        if np.count_nonzero(sel) < min_pairs:
            continue
        tau_chunk = tau[sel]
        sf_chunk = np.sqrt(sf2[sel])
        tau_bin.append(10.0 ** np.mean(np.log10(tau_chunk)))
        sf_bin.append(np.median(sf_chunk))
        sf_lo.append(np.percentile(sf_chunk, 16))
        sf_hi.append(np.percentile(sf_chunk, 84))

    return (
        np.asarray(tau_bin, dtype=float),
        np.asarray(sf_bin, dtype=float),
        np.asarray(sf_lo, dtype=float),
        np.asarray(sf_hi, dtype=float),
    )


def fit_structure_function(tau, sf, sf_lo=None, sf_hi=None):
    """Fit SF_inf * sqrt(1 - exp(-tau/tau_char)) and return sigma,tau."""

    tau = np.asarray(tau, dtype=float)
    sf = np.asarray(sf, dtype=float)
    mask = np.isfinite(tau) & np.isfinite(sf) & (tau > 0.0) & (sf > 0.0)
    if np.count_nonzero(mask) < 4:
        return {
            "log_sigma_sf": np.nan,
            "log_sigma_sf_err": np.nan,
            "log_tau_sf": np.nan,
            "log_tau_sf_err": np.nan,
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
        sf_err = np.full_like(sf_fit, max(np.median(sf_fit) * 0.15, 1e-3))

    sf_inf_init = np.clip(np.max(sf_fit), 1e-4, None)
    tau_init = np.exp(np.mean(np.log(tau_fit)))

    def sf_model(tau_val, log_sf_inf, log_tau):
        sf_inf = 10.0 ** log_sf_inf
        tau_char = 10.0 ** log_tau
        return sf_inf * np.sqrt(np.clip(1.0 - np.exp(-tau_val / tau_char), 0.0, None))

    log_tau_low = 1.0
    log_tau_high = 4.0

    log_sf_inf_high = np.log10(np.sqrt(2.0))

    try:
        popt, pcov = curve_fit(
            sf_model,
            tau_fit,
            sf_fit,
            p0=(np.log10(sf_inf_init), np.log10(tau_init)),
            sigma=sf_err,
            absolute_sigma=True,
            bounds=([-6.0, log_tau_low], [log_sf_inf_high, log_tau_high]),
            maxfev=20000,
        )
        perr = np.sqrt(np.diag(pcov))
    except Exception as exc:
        logging.warning("Structure-function fit failed: %s", exc)
        return {
            "log_sigma_sf": np.nan,
            "log_sigma_sf_err": np.nan,
            "log_tau_sf": np.nan,
            "log_tau_sf_err": np.nan,
            "sf_valid": False,
            "sf_nbins": float(tau_fit.size),
        }

    tau_char = 10.0 ** float(popt[1])
    tau_min = float(np.nanmin(tau_fit))
    tau_max = float(np.nanmax(tau_fit))
    near_lower_bound = np.isclose(float(popt[1]), log_tau_low, atol=0.05)
    near_upper_bound = np.isclose(float(popt[1]), log_tau_high, atol=0.05)
    turnover_bracketed = np.isfinite(tau_char) and (tau_min < tau_char < tau_max)
    sf_valid = bool(np.all(np.isfinite(popt)) and not near_lower_bound and not near_upper_bound and turnover_bracketed)

    return {
        "log_sigma_sf": float(popt[0] - np.log10(np.sqrt(2.0))),
        "log_sigma_sf_err": float(perr[0]) if np.all(np.isfinite(perr)) else np.nan,
        "log_tau_sf": float(popt[1]),
        "log_tau_sf_err": float(perr[1]) if np.all(np.isfinite(perr)) else np.nan,
        "sf_valid": sf_valid,
        "sf_nbins": float(tau_fit.size),
    }


def compute_structure_function_diagnostics(samples, obj, z):
    """Fit SF in the g band using rest-frame lags and convert to UV."""

    bands = list(obj["bands"])
    if "g" not in bands:
        raise KeyError("Missing g band required for SF diagnostics.")
    lam_rf = np.asarray([lambda_pivot[band] / (1.0 + float(z)) for band in bands], dtype=float)
    ref_idx = bands.index("g")
    ref_band = bands[ref_idx]
    lam_ref_band = float(lam_rf[ref_idx])

    band_idx = np.asarray(obj["band_idx"])
    mask = band_idx == ref_idx
    t_band = np.asarray(obj["X"][0], dtype=float)[mask] / (1.0 + float(z))
    y_band = np.asarray(obj["y"], dtype=float)[mask]
    yerr_band = np.asarray(obj["yerr"], dtype=float)[mask]
    tau_sf, sf_med, sf_lo, sf_hi = empirical_structure_function(t_band, y_band, yerr_band)
    fit = fit_structure_function(tau_sf, sf_med, sf_lo, sf_hi)

    eta_sigma = float(np.nanmedian(np.asarray(samples["eta_sigma"], dtype=float)))
    eta_tau = float(np.nanmedian(np.asarray(samples["eta_tau"], dtype=float)))
    log_sigma_uv = (
        fit["log_sigma_sf"] + log_single_pl(2500.0, lam_ref_band, eta_sigma)
        if np.isfinite(fit["log_sigma_sf"]) else np.nan
    )
    log_tau_rf = (
        fit["log_tau_sf"] + log_single_pl(2500.0, lam_ref_band, eta_tau)
        if np.isfinite(fit["log_tau_sf"]) else np.nan
    )
    log_tau_uv_obs = log_tau_rf + np.log10(1.0 + float(z)) if np.isfinite(log_tau_rf) else np.nan

    return {
        "sf_ref_band": ref_band,
        "sf_ref_lambda_rf": lam_ref_band,
        "log_sigma_sf_ref_band": fit["log_sigma_sf"],
        "log_sigma_sf_ref_band_err": fit["log_sigma_sf_err"],
        "log_tau_sf_ref_band": fit["log_tau_sf"],
        "log_tau_sf_ref_band_err": fit["log_tau_sf_err"],
        "log_sigma_uv_sf": log_sigma_uv,
        "log_sigma_uv_sf_err": fit["log_sigma_sf_err"],
        "log_tau_uv_sf": log_tau_uv_obs,
        "log_tau_uv_sf_err": fit["log_tau_sf_err"],
        "log_tau_uv_rf_sf": log_tau_rf,
        "log_tau_uv_rf_sf_err": fit["log_tau_sf_err"],
        "sf_valid": fit["sf_valid"],
        "sf_nbins": fit["sf_nbins"],
    }


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


def log_lag_blr_prior():
    return dist.Normal(jnp.log(1e2), jnp.log(50.0))


def compute_parameter_kls(
    flat_samples,
    *,
    bands,
    z,
    lambda_center_rf,
    log_jitter_mean,
    disable_poly1=False,
    disable_lag_blr=False,
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
                    lambda x: _dist_log_prob_array(log_lag_blr_prior(), x),
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

    amp_cont = jnp.exp(log_sigma_band + log_amp_delta_lya_band)
    amp_blr = jnp.exp(log_sigma_uv_exp + log_amp_delta_blr)
    amp_blr2 = jnp.exp(log_sigma_uv_exp + log_amp_delta_blr2)
    lag_disk = lag0_exp * (lam_rf / lambda_center_rf_exp) ** lag_beta_exp
    lag_blr = jnp.exp(log_lag_blr)
    lag_blr2 = jnp.exp(log_lag_blr2)
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
    explicit["amp_cont"] = amp_cont
    explicit["amp_blr"] = amp_blr
    explicit["amp_blr2"] = amp_blr2
    explicit["lag_disk"] = lag_disk
    explicit["lag_blr"] = lag_blr
    explicit["lag_blr2"] = lag_blr2
    explicit["tau_fast_band"] = jnp.exp(log_tau_fast_band)
    explicit["tau_slow_band"] = jnp.exp(log_tau_slow_band)
    explicit["log_kernel_param"] = log_kernel_param
    return explicit


def add_model_prediction_params(samples, lam_rf):
    """Add explicit model parameters needed for prediction/plotting."""

    out = dict(samples)
    if all(
        key in out
        for key in (
            "log_kernel_param",
            "amp_cont",
            "amp_blr",
            "amp_blr2",
            "lag_disk",
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
    out["amp_blr"] = np.asarray(explicit["amp_blr"])
    out["amp_blr2"] = np.asarray(explicit["amp_blr2"])
    out["lag_disk"] = np.asarray(explicit["lag_disk"])
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
                log_lag_blr_raw = numpyro.sample("log_lag_blr_raw", log_lag_blr_prior())
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
                log_lag_blr_raw = numpyro.sample("log_lag_blr_raw", log_lag_blr_prior())
                log_amp_delta_blr2_raw = numpyro.sample("log_amp_delta_blr2_raw", log_amp_delta_blr_prior())
                log_lag_blr2_raw = numpyro.sample("log_lag_blr2_raw", log_lag_blr_prior())

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
        numpyro.deterministic("amp_blr", params["amp_blr"])
        numpyro.deterministic("amp_blr2", params["amp_blr2"])
        numpyro.deterministic("log_amp_delta_lya_band", params["log_amp_delta_lya_band"])
        numpyro.deterministic("lag_disk", params["lag_disk"])
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
    parser.add_argument("--inject_fake", action="store_true", help="Inject fake light curves.")
    parser.add_argument("--max_tree_depth", type=int, default=8, help="NUTS max tree depth.")
    parser.add_argument("--load_sample_file", action="store_true", help="Load saved samples (debug).")
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
    parser.add_argument("--disable_lag_blr", action="store_true", default=False, help="Disable BLR lag model.")
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
                drop_band_lyman_alpha=args.drop_band_lyman_alpha,
                tau_fast_truncated=args.tau_fast_truncated,
                n_blr_terms=args.n_blr_terms,
            )

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

            if args.load_sample_file:
                logging.warning("[DEBUG] Loading saved samples (flat) — developer mode.")
                obj_flat_samples = load_obj_samples_from_hdf5(oid)
                samples_per_chain = None
            else:
                key = random.PRNGKey(0)
                key = random.fold_in(key, idx)
                mcmc.run(key)
                samples_flat = mcmc.get_samples(group_by_chain=False)
                samples_per_chain = mcmc.get_samples(group_by_chain=True)
                samples_flat = tree_map(lambda x: np.asarray(device_get(x)), samples_flat)
                samples_per_chain = tree_map(lambda x: np.asarray(device_get(x)), samples_per_chain)
                obj_flat_samples = samples_flat
                save_obj_samples_to_hdf5(obj_flat_samples, oid)

            explicit_scalar = build_explicit_model_params(obj_flat_samples, lam_rf)
            for key, explicit_key in (
                ("log_sigma_uv", "log_sigma_uv"),
                ("log_tau_uv", "log_tau_uv"),
                ("log_tau_fast_uv", "log_tau_fast_uv"),
                ("lambda_center_rf", "lambda_center_rf"),
                ("amp_cont", "amp_cont"),
                ("amp_blr", "amp_blr"),
                ("amp_blr2", "amp_blr2"),
                ("lag_disk", "lag_disk"),
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
            if samples_per_chain is not None:
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
                    if not args.disable_correlation_plot:
                        plot_correlation_matrix(obj_flat_samples_flatten_per_band, obj)
                    if not args.disable_histogram_plot:
                        plot_all_histograms(obj_flat_samples_flatten_per_band, obj)
                    if not args.disable_corner_plot:
                        plot_posterior_fast(obj_flat_samples_flatten_per_band, obj)
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
            print(
                f"[{oid}] log_sigma_uv = {log_sigma_uv} ± {log_sigma_uv_err} ; "
                f"log_tau_uv_rf = {log_tau_uv_rf} ± {log_tau_uv_rf_err} ; "
                f"log_sigma_uv_bpl = {log_sigma_uv_bpl} ; "
                f"log_tau_uv_rf_bpl = {log_tau_uv_rf_bpl} ; "
                f"log_sigma_uv_sf = {log_sigma_uv_sf} ; "
                f"log_tau_uv_rf_sf = {log_tau_uv_rf_sf}"
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
