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


def bending_power_law_psd(freq, log_sigma, log_tau):
    """Single-break PSD with flat low-frequency slope and -2 high-frequency slope."""

    freq = np.asarray(freq, dtype=float)
    sigma = np.power(10.0, float(log_sigma))
    tau = np.power(10.0, float(log_tau))
    denom = 1.0 + np.square(2.0 * np.pi * freq * tau)
    return 2.0 * sigma * sigma * tau / denom


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

    def model_log10(freq_val, log_sigma, log_tau):
        psd = bending_power_law_psd(freq_val, log_sigma, log_tau)
        return np.log10(np.clip(psd, 1e-300, None))

    try:
        popt, pcov = curve_fit(
            model_log10,
            freq_fit,
            log_power,
            p0=(np.log10(sigma_init), np.log10(tau_init)),
            sigma=log_err,
            absolute_sigma=True,
            bounds=([-6.0, -1.0], [3.0, 6.0]),
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
            "psd_bpl_valid": False,
            "psd_bpl_nbins": float(freq_fit.size),
        }

    return {
        "log_sigma_bpl": float(popt[0]),
        "log_sigma_bpl_err": float(perr[0]) if np.all(np.isfinite(perr)) else np.nan,
        "log_tau_bpl": float(popt[1]),
        "log_tau_bpl_err": float(perr[1]) if np.all(np.isfinite(perr)) else np.nan,
        "psd_bpl_valid": True,
        "psd_bpl_nbins": float(freq_fit.size),
    }


def compute_lomb_scargle_break_diagnostics(model, samples, obj, z, *, n_freq=500):
    """Fit a bending power law to the combined Lomb-Scargle PSD break."""

    posterior_median = {k: np.median(v, axis=0) for k, v in samples.items()}
    freqs = np.logspace(-6, 2, n_freq)
    f_bin, p_bin, p_lo, p_hi, counts, p_noise = combined_lomb_scargle_from_model(
        model,
        obj["y"],
        obj["yerr"],
        posterior_median,
        2.0 * np.pi * freqs,
        amp_reference="uv",
    )
    fit = fit_bending_power_law_psd(f_bin, p_bin, p_lo, p_hi)
    log_tau_rf = fit["log_tau_bpl"] - np.log10(1.0 + float(z)) if np.isfinite(fit["log_tau_bpl"]) else np.nan
    log_tau_rf_err = fit["log_tau_bpl_err"]

    out = {
        "log_sigma_uv_bpl": fit["log_sigma_bpl"],
        "log_sigma_uv_bpl_err": fit["log_sigma_bpl_err"],
        "log_tau_uv_bpl": fit["log_tau_bpl"],
        "log_tau_uv_bpl_err": fit["log_tau_bpl_err"],
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
    log_tau_uv_low = 0.0
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

        for amp_key in (f"log_amp_delta_blr_{band}", f"log_amp_delta_blr2_{band}"):
            if amp_key in flat_samples:
                kls[f"{amp_key}_kl"] = kl_from_samples(
                    flat_samples[amp_key],
                    lambda x: _dist_log_prob_array(log_amp_delta_blr_prior(), x),
                )

        for lag_key in (f"log_lag_blr_{band}", f"log_lag_blr2_{band}"):
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
    parser.add_argument("--disable_corner_plot", action="store_true", default=False, help="Disable corner plot.")
    parser.add_argument("--disable_lag_blr", action="store_true", default=False, help="Disable BLR lag model.")
    parser.add_argument("--disable_plot_psd", action="store_true", default=False, help="Disable PSD sub-plot.")
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
            psd_break_result = compute_lomb_scargle_break_diagnostics(
                m,
                plot_samples,
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
            )

            if args.plot:
                try:
                    plot_mcmc_traces(obj_flat_samples_flatten_per_band, obj)
                    plot_data = result | psd_break_result
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
                        plot_bpl_fit=True,
                        filename_suffix=f"{suffix}_bpl",
                    )
                    plot_correlation_matrix(obj_flat_samples_flatten_per_band, obj)
                    plot_all_histograms(obj_flat_samples_flatten_per_band, obj)
                    if not args.disable_corner_plot:
                        plot_posterior_fast(obj_flat_samples_flatten_per_band, obj)
                except Exception as e:
                    logging.error(f"[{oid}] Plotting error: {e}")
                    logging.error(traceback.format_exc())

            final_result = obj | result | adf_result | psd_break_result | kl_result | dict(prefix=prefix, suffix=suffix) 
            # final_result |= diagnostics
            log_sigma_uv = final_result.get("log_sigma_uv")
            log_sigma_uv_err = final_result.get("log_sigma_uv_err")
            log_tau_uv_rf = final_result.get("log_tau_uv_rf")
            log_tau_uv_rf_err = final_result.get("log_tau_uv_rf_err")
            log_sigma_uv_bpl = final_result.get("log_sigma_uv_bpl")
            log_tau_uv_rf_bpl = final_result.get("log_tau_uv_rf_bpl")
            print(
                f"[{oid}] log_sigma_uv = {log_sigma_uv} ± {log_sigma_uv_err} ; "
                f"log_tau_uv_rf = {log_tau_uv_rf} ± {log_tau_uv_rf_err} ; "
                f"log_sigma_uv_bpl = {log_sigma_uv_bpl} ; "
                f"log_tau_uv_rf_bpl = {log_tau_uv_rf_bpl}"
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

    try:
        plot_sigma_tau_vs_lambda_with_model(
            results,
            inject_fake=args.inject_fake,
        )
    except Exception as e:
        logging.error(f"plot_sigma_tau_vs_lambda_with_model error: {e}")
        logging.error(traceback.format_exc())

    if args.inject_fake:
        try:
            plot_recovery(results)
        except Exception as e:
            logging.error(f"plot_recovery error: {e}")
            logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
