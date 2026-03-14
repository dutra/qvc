#!/usr/bin/env python3
"""
Single-object multiband OU/BLR fitter (one light curve at a time).

Key changes vs. the old joint version:
- No vmap, no batch plates, no padding, no fixed 5-band assumptions.
- The NumPyro model runs per-object; number of bands is inferred per object.
- Works with arbitrary band sets/order; we pass a per-object `bands` list.
- Keeps your plotting, saving, and per-band flatten utilities.

Requires your modules:
  - multiband_fit_utils, multiband_fit_plotting, multiband_generate_lc
  - multiband_models + specific Model variants
"""

import os
import sys
import argparse
import logging
import traceback
import math
from pathlib import Path

import numpy as np
import pandas as pd
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
from jax import random
from jax import random, device_get
from jax.tree_util import tree_map
import numpyro
numpyro.set_host_device_count(num_cores)
numpyro.enable_x64()
numpyro.enable_validation(True)
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC

from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import median_abs_deviation
from tinygp import kernels

from multiband_fit_utils import *
from multiband_fit_plotting import *
from multiband_generate_lc import *

from multiband_model_blr_mag_new import MyMultiVarModel_SMAG_New


# ---------- Global toggles ----------
zero_mean = False
has_jitter = True
has_lag = True


# ---------- Helper: build per-object LC ----------
def make_lc(
    ModelClass,
    data,
    bands,
    inject_fake=False,
    alpha_sigma=-0.5,
    beta_tau=0.2,
    disable_band_drop=False,
):
    """Prepare one object's multiband time series into the model-ready arrays.

    Returns dict with keys:
      X=(times_shifted, band_idx), y, yerr, band_idx, mags_means, mags_stds,
      dropped_bands, t_obs_length, t_rf_length, bands (list-of-str), and
      optionally log_tau_fake/log_sigma_fake/alpha_sigma/beta_tau.
    """
    # If bands not specified, infer from present keys and keep SDSS order if possible

    if disable_band_drop:
        dropped_bands = ["z"]
    else:
        dropped_bands = sdss_bands_affected_by_lya(data["z"]) + ["z"]

    logging.info(
        f"Excluding blue bands {dropped_bands} for object {data['object_id']} at z={data['z']}"
    )

    bands = [b for b in bands if b not in dropped_bands]

    times = data["times"]
    mags = data["mags"]
    magerrs = data["magerrs"]

    if len(bands) == 0:
        print(f"No usable bands for {data['object_id']}, skipping.", flush=True)
        return None

    # Combine arrays
    all_times = np.concatenate([np.asarray(times[b]) for b in bands])
    print("ALL times: ", all_times[0])
    all_mags = np.concatenate([np.asarray(mags[b]) for b in bands])
    all_magerrs = np.concatenate([np.asarray(magerrs[b]) for b in bands])
    band_idx = np.concatenate([np.full(len(times[b]), i) for i, b in enumerate(bands)]).astype(
        np.int64, copy=False
    )

    if len(all_times) == 0:
        print(f"No points for {data['object_id']}, skipping.", flush=True)
        return None

    # Stable sort by (time, band) key
    tie_eps = 10.0 * np.finfo(all_times.dtype).eps
    key = all_times + band_idx.astype(all_times.dtype) * tie_eps
    order = np.argsort(key, kind="mergesort")
    all_times, all_mags, all_magerrs, band_idx = (
        all_times[order],
        all_mags[order],
        all_magerrs[order],
        band_idx[order],
    )

    # Optional fake injection (kept as in your version, per object)
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

    # Finite mask
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

    # Outlier reject per band (rolling median/MAD)
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

    # Observed/rest-frame lengths
    t_obs_length = float(np.max(all_times) - np.min(all_times))
    t_rf_length = float(t_obs_length / (1.0 + data["z"]))
    print(f"[{data['object_id']}] Δt_obs={t_obs_length:.2f} d, Δt_rf={t_rf_length:.2f} d")

    # Center per band (and inflate error for dropped bands)
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

    # Build arrays
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
        'cadence': data['cadence'],
        'cadence_err': data['cadence_err'],
        'number_points': data['number_points'],
    }

    if inject_fake:
        out["log_tau_fake"] = float(np.log(10 ** float(log_tau0_rf) * (1 + data["z"])))  # from above scope
        out["log_sigma_fake"] = float(np.log(10 ** float(log_sigma0)))
        out["alpha_sigma"] = float(alpha_sigma)
        out["beta_tau"] = float(beta_tau)
    else:
        out["log_tau_fake"] = -99.0
        out["log_sigma_fake"] = -99.0

    return out


# ---------- Per-object NumPyro model ----------
def build_single_object_model(
    ModelClass,
    obj_dict,
    lam_rf,                # (B,) array for this object
    f_host_2500,          # scalar float
    log_jitter_mean,      # (B,) array (per-band)
    *,
    bwb=True,
    disable_poly1=False,
    disable_lag_blr=False,
    free_eta_break=False,
    sigma_tau_uniform=False,
    broken_pl=False,
    sigma_tau_plane_cut=True,
    lmc_q_groups=None,
    yupriors=False,
    nearby_lc=False,
    tau_fast_truncated=False
):
    """
    Returns a NumPyro model function for ONE object with B bands.
    """
    # Extract per-object data
    (t, bidx) = obj_dict["X"]
    y = obj_dict["y"]
    yerr = obj_dict["yerr"]
    cadence = obj_dict['cadence']
    cadence_err = obj_dict['cadence_err']
    z = float(obj_dict["z"])
    B = int(len(lam_rf))  # number of bands (static Python int for plate size)

    # Reference centers
    log_tau_drw0_c = jnp.log(10**2.5 * (1.0 + z))

    def model():
        # ---- Global-ish means (these can be kept per-object to retain your prior structure)

        if yupriors:
            eta_A1 = numpyro.sample("eta_A1", dist.Normal(-0.746, 0.030))
            eta_A2 = numpyro.sample("eta_A2", dist.Normal(-0.746, 0.030))
            eta_tau1 = numpyro.sample("eta_tau1", dist.Normal(0.388, 0.083))
            eta_tau2 = numpyro.sample("eta_tau2", dist.Normal(0.388, 0.083))
        else:
            if nearby_lc:
                eta_A1 = numpyro.sample("eta_A1", dist.Normal(-0.8, 0.3))
                eta_tau1 = numpyro.sample("eta_tau1", dist.Normal(0.4, 0.2))
            else:
                eta_A1 = numpyro.sample("eta_A1", dist.TruncatedNormal(-0.5, 1.0, high=0.0))
                eta_tau1 = numpyro.sample("eta_tau1", dist.Normal(0.5, 0.5))

            if broken_pl:
                eta_A2 = numpyro.sample("eta_A2", dist.TruncatedNormal(-0.5, 1.0, high=0.0))
                eta_tau2 = numpyro.sample("eta_tau2", dist.Normal(0.5, 0.5))
            else:
                eta_A2 = numpyro.deterministic("eta_A2", 0.0)
                eta_tau2 = numpyro.deterministic("eta_tau2", 0.0)

        if free_eta_break:
            s = 0.4
            median = 0.1
            mu = jnp.log(median)
            sigma = jnp.sqrt(jnp.log((1 + jnp.sqrt(1 + 4 * (s / median) ** 2)) / 2))
            eta_break = numpyro.sample("eta_break", dist.LogNormal(mu, sigma))
            lam_s = numpyro.sample("lam_s", dist.Normal(2500.0, 100.0))
        else:
            eta_break = numpyro.deterministic("eta_break", 0.1)
            lam_s = numpyro.deterministic("lam_s", 2500.0)

        # Core OU params
        log_tau_drw0_high = jnp.log(10**4.0 * (1.0 + z))
        log_tau_drw0_low = 0.0
        if sigma_tau_uniform:
            log_tau_drw0 = numpyro.sample("log_tau_drw0", dist.Uniform(log_tau_drw0_low, log_tau_drw0_high))
        else:
            log_tau_drw0 = numpyro.sample(
                "log_tau_drw0",
                dist.TruncatedNormal(log_tau_drw0_c, 1.2 * jnp.log(10), low=log_tau_drw0_low, high=log_tau_drw0_high),
            )
        
        log_tau_fast0_low = 0.0
        log_tau_fast0_high = jnp.log(100.0 * (1.0 + z))
        log_tau_fast0_c = jnp.log(10 * (1.0 + z))
        log_tau_fast0 = numpyro.sample("log_tau_fast0", 
                                       dist.TruncatedNormal(log_tau_fast0_c, jnp.log(25), 
                                                            low=log_tau_fast0_low, high=log_tau_fast0_high))


        if sigma_tau_uniform:
            log_sigma0 = numpyro.sample("log_sigma0", dist.Uniform(-2.0 * jnp.log(10), 0.2 * jnp.log(10)))
        else:
            log_sigma0 = numpyro.sample("log_sigma0", dist.Normal(-0.6 * jnp.log(10), 1.0 * jnp.log(10)))
        log_sigma_hat0 = numpyro.deterministic("log_sigma_hat0", log_sigma0 - 0.5 * log_tau_drw0)

        # Host dilution
        alpha_host = numpyro.sample("alpha_host", dist.Normal(1.0, 0.1))
        alpha_agn = numpyro.sample("alpha_agn", dist.Normal(-1.5, 0.3))
        f_host = numpyro.deterministic("f_host", float(f_host_2500))

        # Mean function detrending
        if disable_poly1:
            poly1 = numpyro.deterministic("poly1", 0.0)
        else:
            if nearby_lc:
                poly1 = numpyro.sample("poly1", dist.Normal(0.0, 0.2))
            else:
                poly1 = numpyro.sample("poly1", dist.Normal(0.0, 0.1))

        # Disk lags
        lag0 = numpyro.sample("lag0", dist.TruncatedNormal(10.0, 5.0, low=0.0))
        lag_beta = numpyro.sample("lag_beta", dist.TruncatedNormal(4.0 / 3.0, 0.2, low=0.0))

        # per band bwb scaling
        if bwb:
            bwb_alpha = numpyro.sample("bwb_alpha", dist.Uniform(0.0, 1.0))
        else:
            bwb_alpha = numpyro.deterministic("bwb_alpha", 1.0)

        # Per-band plate
        with numpyro.plate("band", B):
            mean = numpyro.sample("mean", dist.Normal(jnp.zeros(B), 0.2))


            if disable_lag_blr:
                log_amp_delta_blr = numpyro.deterministic("log_amp_delta_blr", jnp.full(B, -1e9))
                log_lag_blr = numpyro.deterministic("log_lag_blr", jnp.full(B, -9.0))
            else:
                log_amp_delta_blr = numpyro.sample("log_amp_delta_blr", dist.Normal(jnp.full(B, -1.0), 1.0))
                log_lag_blr = numpyro.sample("log_lag_blr", dist.Uniform(jnp.log(2.0), jnp.log(5000.0)))

            # Convolution widths (kept as deterministic heuristics)
            width_blr = numpyro.deterministic("width_blr", 0.2 * jnp.exp(log_tau_drw0_c) * jnp.ones(B))
            width_cont = numpyro.deterministic("width_cont", 0.2 * jnp.exp(log_tau_drw0_c) * jnp.ones(B))

            # Per-band jitter prior mean provided (shape B)
            log_jitter = numpyro.sample("log_jitter", dist.Normal(log_jitter_mean, 1.0))

        # Fake flags (bookkeeping)
        _ = numpyro.deterministic("log_tau_fake", float(obj_dict.get("log_tau_fake", -99.0)))
        _ = numpyro.deterministic("log_sigma_fake", float(obj_dict.get("log_sigma_fake", -99.0)))

        # Log-likelihood via your Model class
        m = ModelClass(
            X=(t, bidx),
            y=y,
            yerr=yerr,
            kernel=kernels.quasisep.Exp(jnp.array([1.0, 1.0])),
            zero_mean=zero_mean,
            has_jitter=has_jitter,
            has_lag=has_lag,
            lam_rf=lam_rf,
            z=z,
            q_groups=lmc_q_groups,
            use_bwb=bwb,
            broken_pl=broken_pl,
            bwb=bwb
        )

        params = dict(
            log_tau_drw0=log_tau_drw0,
            log_tau_fast0=log_tau_fast0,
            log_sigma0=log_sigma0,
            alpha_host=alpha_host,
            alpha_agn=alpha_agn,
            f_host=f_host,
            poly1=poly1,
            mean=mean,
            log_amp_delta_blr=log_amp_delta_blr,
            log_lag_blr=log_lag_blr,
            log_jitter=log_jitter,
            lag0=lag0,
            lag_beta=lag_beta,
            bwb_alpha=bwb_alpha,
            #bwb_beta=bwb_beta,
            width_blr=width_blr,
            width_cont=width_cont,
            eta_A1=eta_A1,
            eta_A2=eta_A2,
            eta_tau1=eta_tau1,
            eta_tau2=eta_tau2,
            eta_break=eta_break,
            lam_s=lam_s,
        )

        numpyro.factor("loglike", m.log_prob(params))

    return model


# ---------- Main ----------
def main():
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Fit quasars one-by-one (no joint batching).")
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
    parser.add_argument("--bwb", action="store_true", help="Enable BWB model.")
    parser.add_argument("--max_tree_depth", type=int, default=8, help="NUTS max tree depth.")
    parser.add_argument("--load_sample_file", action="store_true", help="Load saved samples (debug).")
    parser.add_argument("--disable_poly1", action="store_true", help="Disable trend.")
    parser.add_argument("--jax_trace", action="store_true", help="Enable JAX trace (compile profile).")
    parser.add_argument("--rf_length_cut", type=int, default=-1, help="Rest-frame cut (days).")
    parser.add_argument("--exact_same_length", action="store_true", help="Exact same RF length cut.")
    parser.add_argument("--load_stone_lcs", action="store_true", default=False, help="Use Stone LCs.")
    parser.add_argument("--free_eta_break", action="store_true", default=False, help="Free eta_break, lam_s.")
    parser.add_argument("--disable_corner_plot", action="store_true", default=False, help="Disable corner plot.")
    parser.add_argument("--couple_sigma_tau", action="store_true", default=False, help="Couple sigma/tau prior.")
    parser.add_argument("--disable_lag_blr", action="store_true", default=False, help="Disable BLR lag model.")
    parser.add_argument("--sigma_tau_uniform", action="store_true", default=False, help="Uniform priors for sigma/tau.")
    parser.add_argument("--lmc", type=int, default=-6, help="LMC Q groups (0/1/2/3).")
    parser.add_argument("--disable_plot_psd", action="store_true", default=False, help="Disable PSD sub-plot.")
    parser.add_argument("--inject_random_fake_etas", action="store_true", default=False, help="Randomize fake etas.")
    parser.add_argument("--fhost_csv", type=str, default=None, help="CSV with columns: object_id,f_host_2500")
    parser.add_argument("--disable_fhost", action="store_true", default=False, help="Set all f_host_2500=0.")
    parser.add_argument("--broken_pl", action="store_true", default=False, help="Use broken power law.")
    parser.add_argument("--log_sigma_eta_tau_sigma", type=float, default=0.2, help="Stddev for log_sigma_eta_tau priors.")
    parser.add_argument("--beta_tau", type=float, default=0.2, help="beta_tau for fake curves.")
    parser.add_argument("--disable_band_drop", action="store_true", default=False, help="Disable Lya band drop.")
    parser.add_argument("--load_nearby_lc_csv", type=str, default=None, help="CSV listing nearby LCs to load.")
    parser.add_argument("--load_yu_priors", action="store_true", default=False, help="Use Yu+2023 priors.")
    parser.add_argument("--disable_sigma_tau_plane_cut", action="store_true", default=False, help="Disable sigma–tau plane cut.")
    parser.add_argument("--tau_fast_truncated", action="store_true", default=False, help="Truncated prior for tau_fast0.")
    args = parser.parse_args()
    print("Args:", args)

    # Load objects
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
            objs, n_days=args.rf_length_cut, same_length=args.exact_same_length
        )
        print(f"After restframe cut, {len(objs)} objects remain.")

    # f_host setup
    if args.disable_fhost:
        print("[WARNING] Disabling f_host: setting f_host_2500=0.0 for all objects.")
        for obj in objs:
            obj["f_host_2500"] = 0.0
    else:
        if args.fhost_csv is None:
            raise ValueError("Must provide --fhost_csv if not using --disable_fhost.")
        fhost_df = pd.read_csv(args.fhost_csv, dtype={"object_id": str})
        fhost_map = fhost_df.set_index("object_id")[["f_host_2500"]].to_dict(orient="index")
        for obj in objs:
            oid = str(obj["object_id"])
            if oid in fhost_map and fhost_map[oid]["f_host_2500"] >= 0:
                obj["f_host_2500"] = float(fhost_map[oid]["f_host_2500"])
            else:
                raise ValueError(f"Object {oid} missing/invalid f_host_2500 in {args.fhost_csv}")

    ModelClass = MyMultiVarModel_SMAG_New

    # Fake eta draw (global for this run)
    if args.inject_random_fake_etas:
        rng = np.random.default_rng()
        alpha_sigma = float(rng.uniform(-1.0, 0.0))
        beta_tau = float(rng.uniform(-0.5, 2.0))
        print(f"Randomized alpha_sigma={alpha_sigma:.3f}, beta_tau={beta_tau:.3f}")
    else:
        alpha_sigma = -0.5
        beta_tau = float(args.beta_tau)
        print(f"Using fixed alpha_sigma={alpha_sigma:.3f}, beta_tau={beta_tau:.3f}")

    # Iterate one object at a time
    results = []
    chain_method = "parallel" if args.nchains and args.nchains > 1 else "sequential"

    iterator = tqdm(objs, desc="Fitting", disable=not args.progress)
    for idx, obj in enumerate(iterator):
        oid = str(obj["object_id"])
        try:
            # Select default band set: SDSS-like unless Stone
            default_bands = ["u", "g", "r", "i", "z"]
            if args.load_stone_lcs:
                default_bands = ["g", "r", "i"]
            if args.load_nearby_lc_csv is not None:
                default_bands = ["g", "r", "i"]
            lc = make_lc(
                ModelClass,
                obj,
                bands=default_bands,
                inject_fake=args.inject_fake,
                alpha_sigma=alpha_sigma,
                beta_tau=beta_tau,
                disable_band_drop=args.disable_band_drop,
            )
            if lc is None:
                continue

            obj |= lc  # merge dicts for downstream plotting/processing

            # Per-object lam_rf for *actual* bands used
            bands = obj["bands"]
            lam_rf = jnp.array([lambda_pivot[b] for b in bands], dtype=float) / (1.0 + float(obj["z"]))
            print(f"[{oid}] Using bands: {bands}")
            print(f"[{oid}] lam_rf = {lam_rf}")

            # Per-band log_jitter mean from yerr per band
            bidx = obj["band_idx"]
            yerr = np.asarray(obj["yerr"])
            B = len(bands)
            ljm = np.empty(B)
            for i in range(B):
                m = (bidx == i) & np.isfinite(yerr) & (yerr < 10)
                val = np.log(np.mean(yerr[m])) if np.any(m) else np.log(1e-3)
                ljm[i] = val
            log_jitter_mean = jnp.array(ljm)

            # Build per-object model
            numpyro_model = build_single_object_model(
                ModelClass,
                obj,
                lam_rf,
                f_host_2500=float(obj["f_host_2500"]),
                log_jitter_mean=log_jitter_mean,
                bwb=args.bwb,
                disable_poly1=args.disable_poly1,
                disable_lag_blr=args.disable_lag_blr,
                free_eta_break=args.free_eta_break,
                sigma_tau_uniform=args.sigma_tau_uniform,
                lmc_q_groups=args.lmc,
                broken_pl=args.broken_pl,
                sigma_tau_plane_cut=(not args.disable_sigma_tau_plane_cut),
                yupriors=args.load_yu_priors,
                nearby_lc=(args.load_nearby_lc_csv is not None),
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
                # Save/diagnostics for this object
                obj_flat_samples = samples_flat  # already single-object
                save_obj_samples_to_hdf5(obj_flat_samples, oid)
            obj_flat_samples_flatten_per_band = flatten_flat_samples_per_band(obj_flat_samples, bands=bands)
                

            diagnostics = {}
            if samples_per_chain is not None:
                obj_samples_per_chain_flatten_per_band = flatten_per_chain_samples_per_band(samples_per_chain, bands=bands)
                diagnostics = diagnostics_for_per_chain_samples(obj_samples_per_chain_flatten_per_band)

            # Summarize parameters for this object
            result = process_samples(obj_flat_samples_flatten_per_band, obj, broken_pl=args.broken_pl, bands=bands)

            # Plots (do not crash entire run if a single plot fails)
            if args.plot:
                try:
                    plot_mcmc_traces(obj_flat_samples_flatten_per_band, obj)
                    m = ModelClass(
                        obj["X"],
                        obj["y"],
                        obj["yerr"],
                        kernels.quasisep.Exp(jnp.array([1.0, 1.0])),
                        zero_mean=zero_mean,
                        has_jitter=has_jitter,
                        has_lag=has_lag,
                        lam_rf=lam_rf,
                        z=obj["z"],
                        use_bwb=args.bwb,
                        q_groups=args.lmc,
                        broken_pl=args.broken_pl,
                        bwb=args.bwb,
                    )
                    save_combined_plot(
                        obj_flat_samples,
                        m,
                        obj["X"],
                        obj["y"],
                        obj["yerr"],
                        obj["band_idx"],
                        obj["mags_means"],
                        obj["survey_times"],
                        result,
                        time0=obj["time0"],
                        bands=bands,
                        plot_psd=(not args.disable_plot_psd),
                    )
                    plot_correlation_matrix(obj_flat_samples_flatten_per_band, obj)
                    plot_all_histograms(obj_flat_samples_flatten_per_band, obj)
                    if not args.disable_corner_plot:
                        plot_posterior_fast(obj_flat_samples_flatten_per_band, obj)
                except Exception as e:
                    logging.error(f"[{oid}] Plotting error: {e}")
                    logging.error(traceback.format_exc())

            final_result = obj | diagnostics | result | dict(prefix=prefix, suffix=suffix)
            log_sigma_UV = final_result.get("log_sigma_UV")
            log_sigma_UV_err = final_result.get("log_sigma_UV_err")
            log_tau_UV_RF = final_result.get("log_tau_UV_RF")
            log_tau_UV_RF_err = final_result.get("log_tau_UV_RF_err")
            print(
                f"[{oid}] log_sigma_UV = {log_sigma_UV} ± {log_sigma_UV_err} ; "
                f"log_tau_UV_RF = {log_tau_UV_RF} ± {log_tau_UV_RF_err}"
            )
            
            results.append(final_result)

            # Optional fake recovery summary
            if args.inject_fake:
                compare_pairs = [
                    ("log_tau_fake", "log_tau_drw0", "log10_tau"),
                    ("log_sigma_fake", "log_sigma0", "log10_sigma"),
                ]
                summarize_fake_true_vs_recovered(final_result, diagnostics, compare_pairs=compare_pairs)
            
        except Exception as e:
            logging.error(f"[{oid}] Error during fit: {e}")
            logging.error(traceback.format_exc())
            continue

    # Save list (excluding heavy arrays)
    save_quasar_list_hdf5(results, ignored_keys=["X", "y", "yerr", "band_idx"])

    # Aggregate sigma–tau vs lambda plot (optional)
    try:
        plot_sigma_tau_vs_lambda_with_model(results, inject_fake=args.inject_fake, broken_pl=args.broken_pl)
    except Exception as e:
        logging.error(f"plot_sigma_tau_vs_lambda_with_model error: {e}")
        logging.error(traceback.format_exc())

    # Final fake recovery pack
    if args.inject_fake:
        try:
            plot_recovery(results)
        except Exception as e:
            logging.error(f"plot_recovery error: {e}")
            logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
