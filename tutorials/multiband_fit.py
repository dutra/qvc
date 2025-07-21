
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import os
from collections.abc import Callable
import equinox as eqx
from tinygp.helpers import JAXArray
from numpy.typing import NDArray
import tinygp
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm, trange
from scipy.optimize import minimize
import scipy.stats as st
import numpyro
from numpyro import infer
from numpyro.infer import MCMC, NUTS, AIES, Predictive
import numpyro.distributions as dist

from tinygp import kernels, solvers

learning_rate=0.001
import warnings
import jax.scipy as jsp
from jax.scipy.special import erfc, logsumexp

from jax import lax
import numpy as np
import optax
from eztaox.fitter import fit
from eztaox.initializers import DRWInit, UniformInit
from eztaox.models import MultiVarModel, MultiVarModelFFT
from eztaox.utils import formatlc

from eztaox.kernels.quasisep import MultibandLowRank
from tinygp import kernels
from tinygp import GaussianProcess
from eztaox.kernels import direct, quasisep

from astropy.coordinates import SkyCoord
from astropy import units as u

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

warnings.filterwarnings("ignore", category=RuntimeWarning)

from astropy.io import fits
from multiprocessing import Pool, get_context
import h5py
import sys

from tinygp.helpers import JAXArray
import argparse
import traceback

from multiband_fit_utils import *
from multiband_fit_plotting import *
from multiband_generate_lc import *
from multiband_models import *

from solvers import DirectFullRank

#bands = ['g', 'r', 'i']


# define params
zero_mean = False
has_jitter = True
has_lag = True


def initSampler(key, nSample, nBand, X, y, yerr, clean_bands, z):
    # split keys
    subkeys = jax.random.split(key, 14)

    # uniform sampler
    lagSampler = UniformInit(nBand-1, [-10, 10])
    loglagBLRSampler = UniformInit(nBand, [0, 5])
    logtauBLRSampler = UniformInit(1, [jnp.log(10**2.5), jnp.log(10**4.5)])
    meanSampler = UniformInit(nBand, [-1, 1])
    alphaHostFracSampler = UniformInit(1, [0.0, 1.0])
    fHostFracSampler = UniformInit(1, [0.0, 1.0])
    poly1Sampler = UniformInit(1, [-10, 10])
    logAmpDeltaSampler = UniformInit(nBand-1, [-2.0, 0.0])
    logAmpDeltaBLRSampler = UniformInit(nBand, [-5.0, -2.0])
    logJitterSampler = UniformInit(nBand, [jnp.log(1e-4), jnp.log(0.1)])

    print(f"initSampler logJitterSampler {key}", nBand)
    # power laws
    #etaBreakSampler = UniformInit(1, [0, 3])
    #lamsSampler = UniformInit(1, [2400.0, 2600.0])
    # sigma
    etaA1Sampler = UniformInit(1, [-2, 2])
    etaA2Sampler = UniformInit(1, [-2, 2])
    # tau
    etaTau1Sampler = UniformInit(1, [-2, 2])
    etaTau2Sampler = UniformInit(1, [-2, 2])

    # kernel init
    kernelSampler = DRWInit([jnp.log(10**2.5), jnp.log(10**4.5)], [jnp.log(0.01), jnp.log(1.5)])
    logwSampler = UniformInit(1, [0.0, 1.0])

    # --- Build model instance ---
    initial_drw_params = {"log_kernel_param": jnp.log(np.array([500.0, 0.35]))}
    k = kernels.quasisep.Exp(*jnp.exp(initial_drw_params["log_kernel_param"]))
    m = MultiVarModel(
        X, y, yerr, k,
        zero_mean=zero_mean,
        has_jitter=has_jitter,
        has_lag=has_lag,
        clean_bands=clean_bands,
        z=z
    )
    
    # MLE fit of log_prob of m
    @jax.jit
    def loss(params) -> JAXArray:
        return -m.log_prob(params)

    def single_loop(params) -> tuple[dict[str, JAXArray], JAXArray]:
        opt_state = optimizer.init(params)
        for _ in range(nIter):
            grads = jax.grad(loss)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

        return params, loss(params)


    # Collect initial params from m (use bestP or m.default_params)
    init_params = {
        "log_kernel_param": kernelSampler(subkeys[0], 1),
        "log_amp_delta": logAmpDeltaSampler(subkeys[1], 1),
        "log_amp_delta_blr": logAmpDeltaBLRSampler(subkeys[2], 1),
        "mean": meanSampler(subkeys[3], 1),
        "alpha_host": alphaHostFracSampler(subkeys[4], 1),
        "f_host": fHostFracSampler(subkeys[5], 1),
        "poly1": poly1Sampler(subkeys[6], 1),
        "lag": lagSampler(subkeys[7], 1),
        "log_tau_drw_blr": logtauBLRSampler(subkeys[8], 1),
        "log_jitter": logJitterSampler(subkeys[9], 1),
        "eta_A1": etaA1Sampler(subkeys[10], 1),
        "eta_A2": etaA2Sampler(subkeys[11], 1),
        "eta_tau1": etaTau1Sampler(subkeys[12], 1),
        "eta_tau2": etaTau2Sampler(subkeys[13], 1),
        #"eta_break": etaBreakSampler(subkeys[14], 1),
        #"lam_s": lamsSampler(subkeys[15], 1),
    }

    print('Starting MLE')

    import jaxopt

    # jaxopt optimize
    opt = jaxopt.ScipyMinimize(fun=loss, method="SLSQP",)
    soln = opt.run(init_params)
    best_param = soln.params
    print('done MLE')

    print("best",best_param)
    print("MLE loss: ", soln.state.fun_val)

    best_param['log_sigma_hat0'] = 0.5 * (2*best_param['log_kernel_param'][1] - best_param['log_kernel_param'][0])
    best_param['log_tau_drw0'] = best_param['log_kernel_param'][0]
    best_param['log_sigma_band'] = best_param['log_kernel_param'][1] + best_param['log_amp_delta']
    
    print(best_param['log_kernel_param'])
    print(best_param['log_sigma_hat0'],  best_param['log_tau_drw0'], ' <<<<<<<')

    return best_param

def numpyro_model(Model, X, yerr, y=None, bestP=None, clean_bands=None, z=None):
    # --- Kernel and lag parameters ---
    #log_w = numpyro.sample("log_w", dist.Normal(jnp.full_like(bestP["log_w"], jnp.log(20.0)), 1.0))
    log_amp_delta_blr = numpyro.sample("log_amp_delta_blr", dist.Normal(jnp.full_like(bestP["log_amp_delta_blr"], jnp.log(1e-3)), 2.0))
    lag = numpyro.sample("lag", dist.Normal(jnp.full_like(bestP["lag"], 0.0), 10))
    #log_lag_blr = numpyro.sample("log_lag_blr", dist.Normal(jnp.full_like(bestP["log_lag_blr"], jnp.log(1e2)), 2.0))
    log_tau_drw_blr = numpyro.sample("log_tau_drw_blr", dist.Normal(jnp.log(1e2), 2.0))

    log_tau_drw_0 = numpyro.sample("log_tau_drw0", dist.Uniform(bestP['log_kernel_param'][0], jnp.log(1e5))) # tau_drw at 2500 AA
    log_sigma_hat_0 = numpyro.sample("log_sigma_hat0", dist.Uniform(bestP['log_kernel_param'][1], jnp.log(1e2))) # sigma_hat at 2500 AA


    #log_tau_drw_0 = numpyro.sample("log_tau_drw0", dist.Uniform(jnp.log(1e1), jnp.log(1e5))) # tau_drw at 2500 AA
    #log_sigma_hat_0 = numpyro.sample("log_sigma_hat0", dist.Uniform(jnp.log(1e-3), jnp.log(1e2))) # sigma_hat at 2500 AA


    # Initialize jitter using the mean yerr
    mean_yerr = jnp.mean(yerr)
    log_jitter_init = jnp.log(mean_yerr + 1e-6)

    log_jitter = numpyro.sample("log_jitter", dist.Normal(bestP["log_jitter"], 1.0))

    # --- Mean function parameters ---
    mean = numpyro.sample("mean", dist.Normal(bestP["mean"], 1.0))
    poly1 = numpyro.sample("poly1", dist.Normal(0.0, 10.0))

    # --- Power law parameters ---
    # WARNING: SINGLE NOT JOINT
    powerlaw_priors = {
        "eta_A1": (-0.75, 0.1),
        "eta_A2": (-0.6, 0.1),
        "eta_tau1": (0.05, 0.1),
        "eta_tau2": (0.01, 0.1),
        #"eta_break": (3, 0.1),
        #"lam_s": (2500.0, 100.0),
    }
    powerlaw_samples = {
        k: numpyro.sample(k, dist.Normal(loc, scale))
        for k, (loc, scale) in powerlaw_priors.items()
    }

    # --- Build model instance ---
    k = kernels.quasisep.Exp(jnp.array([1, 1]))
    m = Model(
        X, y, yerr, k,
        zero_mean=zero_mean,
        has_jitter=has_jitter,
        has_lag=has_lag,
        clean_bands=clean_bands,
        z=z
    )

    # --- Collect parameters for the model ---
    sample_params = {
        "log_tau_drw0": log_tau_drw_0,
        "log_sigma_hat0": log_sigma_hat_0,
        #"log_w": log_w,
        "log_amp_delta_blr": log_amp_delta_blr,
        "lag": lag,
        #"log_lag_blr": log_lag_blr,
        "log_tau_drw_blr": log_tau_drw_blr,
        "mean": mean,
        "poly1": poly1,
        "log_jitter": log_jitter,
        **powerlaw_samples,
    }

    # --- Evaluate model likelihood ---
    m.sample(sample_params)

import jax.numpy as jnp
from jax import numpy as jnp
from typing import List

def pad_batch_data(batch_data):
    batch_size = len(batch_data)

    # Assuming X is a tuple, where first element is array with shape (N_i, feature_dim)
    first_X_array = batch_data[0]['X'][0]  # e.g. times array

    if len(first_X_array.shape) > 1:
        feature_dim = first_X_array.shape[1]
    else:
        feature_dim = 1

    max_len = max(obj['X'][0].shape[0] for obj in batch_data)  # max time points length

    # Prepare padded arrays
    Xs = jnp.zeros((batch_size, max_len, feature_dim))
    ys = jnp.zeros((batch_size, max_len))
    yerrs = jnp.zeros((batch_size, max_len))
    mask = jnp.zeros((batch_size, max_len), dtype=bool)
    clean_bands_list = []
    zs = []

    for i, obj in enumerate(batch_data):
        Xi = obj['X'][0]  # first element of tuple, e.g. times, shape (N_i, feature_dim)
        N = Xi.shape[0]
        if feature_dim == 1:
            Xs = Xs.at[i, :N, 0].set(Xi)
        else:
            Xs = Xs.at[i, :N, :].set(Xi)
        ys = ys.at[i, :N].set(obj['y'])
        yerrs = yerrs.at[i, :N].set(obj['yerr'])
        mask = mask.at[i, :N].set(True)
        clean_bands_list.append(obj['clean_bands'])
        zs.append(obj['z'])

    zs = jnp.array(zs)
    return Xs, ys, yerrs, mask, clean_bands_list, zs

def numpyro_joint_model(Model, batch_data):
    batch_size = len(batch_data)
    nBands = 5  # or use from config
    band_lag_count = 4  # if e.g. lag is only defined for 4 bands

    # Shared across all objects
    powerlaw_samples = {
        k: numpyro.sample(k, dist.Normal(loc, scale))
        for k, (loc, scale) in {
            "eta_A1": (0.0, 1.0),
            "eta_A2": (0.0, 1.0),
            "eta_tau1": (0.0, 1.0),
            "eta_tau2": (0.0, 1.0),
        }.items()
    }

    # Extract object-level prior means
    log_tau_drw0_mean = jnp.array([obj['bestP']['log_tau_drw0'] for obj in batch_data])
    log_sigma_hat0_mean = jnp.array([obj['bestP']['log_sigma_hat0'] for obj in batch_data])
    log_amp_delta_blr_mean = jnp.stack([jnp.array(obj['bestP']['log_amp_delta_blr']) for obj in batch_data])  # (B, 5)
    lag_mean = jnp.stack([jnp.array(obj['bestP']['lag']) for obj in batch_data])                              # (B, 4)
    mean_mean = jnp.stack([jnp.array(obj['bestP']['mean']) for obj in batch_data])                            # (B, 5)
    log_jitter_mean = jnp.stack([jnp.array(obj['bestP']['log_jitter']) for obj in batch_data])                # (B, 5)

    with numpyro.plate("objects", batch_size):
        # Object-level parameters (shape: [B])
        log_tau_drw0 = numpyro.sample("log_tau_drw0", dist.Normal(log_tau_drw0_mean, 1.0))
        log_sigma_hat0 = numpyro.sample("log_sigma_hat0", dist.Normal(log_sigma_hat0_mean, 1.0))
        log_tau_drw_blr = numpyro.sample("log_tau_drw_blr", dist.Normal(jnp.log(1e2), 2.0))
        alpha_host = numpyro.sample("alpha_host", dist.Normal(0.5, 1.0))
        f_host = numpyro.sample("f_host", dist.Uniform(0.0, 1.0))
        poly1 = numpyro.sample("poly1", dist.Normal(0.0, 10.0))

    with numpyro.plate("objects", batch_size, dim=-2):
        with numpyro.plate("band", nBands, dim=-1):
            # Parameters with shape [B, nBands]
            mean = numpyro.sample("mean", dist.Normal(mean_mean, 1.0))
            log_amp_delta_blr = numpyro.sample("log_amp_delta_blr", dist.Normal(log_amp_delta_blr_mean, 2.0))
            log_jitter = numpyro.sample("log_jitter", dist.Normal(log_jitter_mean, 1.0))

        with numpyro.plate("band_lag", nBands-1):  # Only first 4 bands have lags?
            lag = numpyro.sample("lag", dist.Normal(lag_mean, 10.0))

    # Prepare padded observations ahead of time
    Xs, ys, yerrs, mask, clean_bands_list, zs = pad_batch_data(batch_data)

    def log_prob_fn(i):
        # Collect params for object i
        params = {
            "log_tau_drw0": log_tau_drw0[i],
            "log_sigma_hat0": log_sigma_hat0[i],
            "log_tau_drw_blr": log_tau_drw_blr[i],
            "alpha_host": alpha_host[i],
            "f_host": f_host[i],
            "poly1": poly1[i],
            "mean": mean[i],
            "log_amp_delta_blr": log_amp_delta_blr[i],
            "log_jitter": log_jitter[i],
            "lag": lag[i],
            **powerlaw_samples,
        }

        X_i = Xs[i]      # shape (max_len, feature_dim)
        y_i = ys[i]      # shape (max_len,)
        yerr_i = yerrs[i]  # shape (max_len,)

        # Slice valid data points using mask[i]
        valid_idx = mask[i]
        X_masked = jnp.where(valid_idx[:, None], X_i, 0.0)
        y_masked = jnp.where(valid_idx, y_i, 0.0)
        yerr_masked = jnp.where(valid_idx, yerr_i, 99999.0)

        # Mask Lyman-alpha affected bands
        band_idx = X_masked[:, 1].astype(int)  # assumes 2nd column of X is band index
        lambda_obs = jnp.array([3551., 4686., 6165., 7481., 8931.])  # ugriz in Å
        lambda_rest = lambda_obs[band_idx] / (1 + zs[i])
        yerr_masked = jnp.where(lambda_rest < 1216.0, 99999.0, yerr_masked)
        # TODO: pad width
        
        m = Model(
            X_masked, y_masked, yerr_masked,
            kernels.quasisep.Exp(jnp.array([1.0, 1.0])),
            zero_mean=zero_mean,
            has_jitter=has_jitter,
            has_lag=has_lag,
            clean_bands=['u', 'g', 'r', 'i', 'z'],
            z=zs[i],
        )
        return m.log_prob(params)

    total_log_prob = jax.vmap(log_prob_fn)(jnp.arange(batch_size)).sum()
    numpyro.factor("likelihood", total_log_prob)


def numpyro_joint_model_OLD(Model, batch_data):
    # --- Shared (universal) parameters ---
    # These priors can be broader
    #powerlaw_priors = {
        #"eta_A1": (-1.25, 0.2),
        #"eta_A2": (-0.3, 0.2),
        #"eta_tau1": (0.0, 0.2),
        #"eta_tau2": (0.0, 0.2),
        #"eta_break": (1, 0.1),
        #"lam_s": (2500.0, 100.0),
    #}
    powerlaw_priors = {
        "eta_A1": (0, 1.),
        "eta_A2": (0, 1.),
        "eta_tau1": (0.0, 1.),
        "eta_tau2": (0.0, 1.),
    }
    powerlaw_samples = {
        k: numpyro.sample(k, dist.Normal(loc, scale))
        for k, (loc, scale) in powerlaw_priors.items()
    }

    for i, data in enumerate(batch_data):
        # Object-specific parameters
        #log_w = numpyro.sample(f"log_w_{i}", dist.Normal(jnp.log(20.0), 1.0))

        log_tau_drw_0 = numpyro.sample(f"log_tau_drw0_{i}", dist.Normal(bestP['log_tau_drw0'], 1.0))
        log_sigma_hat_0 = numpyro.sample(f"log_sigma_hat0_{i}", dist.Normal(bestP['log_sigma_hat0'], 1.0))
        log_amp_delta_blr = numpyro.sample(f"log_amp_delta_blr_{i}", dist.Normal(jnp.full_like(bestP["log_amp_delta_blr"], jnp.log(1e-3)), 2.0))
        lag = numpyro.sample(f"lag_{i}", dist.Normal(jnp.full_like(bestP["lag"], 0.0), 10.0))
        #log_lag_blr = numpyro.sample(f"log_lag_blr_{i}", dist.Normal(jnp.full_like(bestP["log_lag_blr"], jnp.log(1e2)), 2.0))
        log_tau_drw_blr = numpyro.sample(f"log_tau_drw_blr_{i}", dist.Normal(jnp.log(1e2), 2.0))
        mean = numpyro.sample(f"mean_{i}", dist.Normal(bestP["mean"], 1.0))
        alpha_host = numpyro.sample(f"alpha_host_{i}", dist.Normal(0.5, 1.0))
        f_host = numpyro.sample(f"f_host_{i}", dist.Uniform(0.0, 1.0))
        poly1 = numpyro.sample(f"poly1_{i}", dist.Normal(0.0, 10.0))
        mean_yerr = jnp.mean(data['yerr'])
        log_jitter_init = jnp.log(mean_yerr + 1e-6)
        log_jitter = numpyro.sample(f"log_jitter_{i}", dist.Normal(jnp.full_like(bestP["log_jitter"], log_jitter_init), 1.0))

        params = {
            #"log_w": log_w,
            "log_tau_drw0": log_tau_drw_0,
            "log_sigma_hat0": log_sigma_hat_0,
            "log_amp_delta_blr": log_amp_delta_blr,
            "lag": lag,
            #"log_lag_blr": log_lag_blr,
            "log_tau_drw_blr": log_tau_drw_blr,
            "mean": mean,
            "alpha_host": alpha_host,
            "f_host": f_host,
            "poly1": poly1,
            "log_jitter": log_jitter,
            **powerlaw_samples,
        }

        m = Model(
            data['X'], data['y'], data['yerr'],
            kernels.quasisep.Exp(jnp.array([1, 1])),  # Placeholder, your Model will build the kernel
            zero_mean=zero_mean, has_jitter=has_jitter, has_lag=has_lag,
            clean_bands=data['clean_bands'], z=data['z']
        )
        log_prob = m.log_prob(params)
        #jax.debug.print("log_prob: {lp} {i}", lp=log_prob, i=i)
        #log_prob = jnp.where(jnp.isfinite(log_prob), log_prob, -1e10)
        numpyro.factor(f"loglike_{i}", log_prob)

def fit_multiband(Model, data, nwarm=500, nsamp=250, progress_bar=False, plot=False, svi=False, fit=True):
    times = data['times']
    mags = data['mags']
    data['mags_means'] = np.array([np.nanmean(mags[band]) for band in mags.keys()])
    for band in mags.keys():
       mags[band] = mags[band] - np.nanmean(mags[band])  # Center the magnitudes
    magerrs = data['magerrs']
    
    #red_bands = bands_redder_than_5000(data['z'])
    blue_bands = bands_bluer_than_lyman_alpha(data['z'])

    clean_bands = list(set(bands))
    # Reorder clean_bands to match the desired order
    clean_bands = list(sorted(clean_bands, key=lambda band: ['u', 'g', 'r', 'i', 'z', 'y'].index(band)))
    #clean_bands = bands
    data['clean_bands'] = clean_bands
    print(f"Bands: {bands}, Clean Bands: {clean_bands}")
    if len(clean_bands) == 0:
        print(f"No clean bands for quasar {data['object_id']}, skipping.", flush=True)
        return None
    # Combine
    print(times.keys())
    all_times = np.concatenate([times[b] for b in clean_bands])
    all_mags = np.concatenate([mags[b] for b in clean_bands]) 
    all_magerrs = np.concatenate([magerrs[b] for b in clean_bands])
    band_idx = np.concatenate([np.full(len(times[b]), i) for i, b in enumerate(clean_bands)])

    if len(all_times) == 0 or len(all_mags) == 0 or len(all_magerrs) == 0:
        print(f"No magnitudes or errors for quasar {data['object_id']}, skipping.", flush=True)
        return None
    # Check for NaNs
    if np.all(~np.isfinite(all_times)) or np.all(~np.isfinite(all_mags)) or np.all(~np.isfinite(all_magerrs)):
        print(f"NaN values ({len(~np.isfinite(all_mags))}/{len(all_mags)}) found in data for quasar {data['object_id']}, skipping.", flush=True)
        return None

    # Sort in time
    sort_idx = np.argsort(all_times)
    all_times = all_times[sort_idx]
    all_mags = all_mags[sort_idx]
    all_magerrs = all_magerrs[sort_idx]
    band_idx = band_idx[sort_idx]

    # Mask NaNs
    mask = np.isfinite(all_mags)
    all_times = all_times[mask]
    all_mags = all_mags[mask]
    all_magerrs = all_magerrs[mask]
    band_idx = band_idx[mask]

    # Define X, y, yerr, t
    # X = (all_times, band_idx)
    X = (jnp.array(all_times)-jnp.min(all_times), jnp.array(band_idx))
    y = np.array(all_mags)
    yerr = np.array(all_magerrs)
    t = np.array(all_times)

    # Reject outliers in moving window per band
    window_size = 6
    mask_outlier = np.ones(len(y), dtype=bool)

    for band in np.unique(band_idx):
        band_mask = band_idx == band
        band_y = y[band_mask]
        band_times = all_times[band_mask]

        for i in range(len(band_y)):
            if i < window_size or i >= len(band_y) - window_size:
                continue
            window = band_y[i - window_size:i + window_size + 1]
            if jnp.abs(band_y[i] - jnp.nanmean(window)) > 2.5 * st.median_abs_deviation(window):
                mask_outlier[np.where(band_mask)[0][i]] = False

    X = (jnp.array(all_times[mask_outlier]) - jnp.min(all_times[mask_outlier]), jnp.array(band_idx[mask_outlier]))
    y = jnp.array(y[mask_outlier])
    yerr = jnp.array(yerr[mask_outlier])
    t = jnp.array(t[mask_outlier])

    if fit == False:
        batch_dict = {'X': X, 'y': y, 'yerr': yerr, 'clean_bands': clean_bands, 'z': data['z'], 'band_idx': band_idx[mask_outlier]}
        return batch_dict

    # define kernel
    initial_drw_params = {"log_kernel_param": jnp.log(np.array([100.0, 0.35]))}
    k = kernels.quasisep.Exp(*jnp.exp(initial_drw_params["log_kernel_param"]))

    # define model
    m = Model(
        X, y, yerr, k, zero_mean=zero_mean, has_jitter=has_jitter, has_lag=has_lag, clean_bands=clean_bands, z=data['z']
    )

    print("Initializing bestP.")
    bestP = initSampler(jax.random.PRNGKey(0), 1, len(clean_bands), X, y, yerr)
    print(bestP)

    for k in bestP.keys():
        bestP[k] += 1e-4 * np.random.randn(*bestP[k].shape)

    print('Starting EMCEE MCMC')
    try:
        #init_strategy = numpyro.infer.init_to_value(values=bestP)
        init_strategy = numpyro.infer.init_to_sample()

        # emcee works better than NUTS for multimodal posteriors
        nuts_kernel = AIES(
            partial(numpyro_model, bestP=bestP, clean_bands=clean_bands, z=data['z']),
            moves={AIES.DEMove() : 0.5, AIES.StretchMove() : 0.5},
            init_strategy=init_strategy,
            )

        num_params = sum(p.size for p in bestP.values())
        #print(f"Number of parameters: {num_params}")

        mcmc = MCMC(
            nuts_kernel,
            num_warmup=nwarm, # This could be less than num_samples
            num_samples=nsamp,
            num_chains=2*num_params,
            progress_bar=True,
            chain_method="vectorized",
        )

        mcmc.run(jax.random.PRNGKey(int(data['object_id'])), Model, X, yerr, y=y)
        samples = mcmc.get_samples(group_by_chain=False)
        diagnostics = mcmc.get_extra_fields()
    except Exception as e:
        print(f"Error during MCMC for quasar {data['object_id']}: {e}", flush=True)
        print("Traceback details:")
        traceback.print_exc()            
        return None

        #print(samples)

        #if np.all(diagnostics['diverging']):
        #    print(f"Diverging MCMC for quasar {data['object_id']}, skipping.", flush=True)
        #    #return None

    result = process_samples(samples, data)
    
    if plot:
        psd_results = compute_psd_from_samples(samples, clean_bands)
        save_combined_plot(samples, m, X, y, yerr, band_idx[mask_outlier], result, psd_results=psd_results)
        plot_traces(samples)
        # plot_mcmc_traces(samples, result)
        # plot_posterior(samples, data, clean_bands=clean_bands)
        # psd_results = compute_psd_from_samples(samples, clean_bands)
        # d['psd'] = psd_results
        # plot_psd(psd_results, data['object_id'])    
    return result


def process_quasar(Model, i_data, n=0, **kwargs):
    i, data = i_data
    # Load the quasar data
    result = fit_multiband(Model, data, **kwargs)
    if result is None:
        print(f"Skipping quasar {data['object_id']}.")
        return None
    data['i'] = i
    data |= result
    return data
                    
if __name__ == '__main__': 
    print("Starting multiband fit", flush=True)

    parser = argparse.ArgumentParser(description="Process quasars with optional filtering.")
    parser.add_argument("--filter_object_id", nargs="+", help="List of object IDs to filter.")
    parser.add_argument("--N", type=int, help="Number of objects to process.")
    parser.add_argument("--skip", type=int, help="Number of objects to skip.")
    parser.add_argument("--chunk_size", type=int, default=500, help="Chunk size for processing objects.")
    parser.add_argument("-f", "--file", type=str, help="Path to the file to append (read and write) objects.") 
    parser.add_argument("--lc_file", type=str, help="Path to the light curve file.")
    parser.add_argument("--filter_file", type=str, help="Path to the file containing object IDs to filter.")
    parser.add_argument("--plot", action="store_true", help="Enable plotting of results.")
    parser.add_argument("--svi", action="store_true", help="Use stochastic variation inference (SVI).")
    parser.add_argument("--ignore_existing", action="store_true", help="Ignore sources already in the HDF5 file.")
    parser.add_argument("--create_lc", action="store_true", help="Only create LC file and exit.")
    parser.add_argument("--progress", action="store_true", help="Show progress bar.")
    parser.add_argument("--joint", action="store_true", help="Use joint model fitting.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU.")
    parser.add_argument("--nwarm", type=int, default=500, help="Number of warmup steps for MCMC.")
    parser.add_argument("--nsamp", type=int, default=250, help="Number of samples for MCMC.")
    parser.add_argument("--nchains", type=int, default=-1, help="Number of chains for MCMC.")
    parser.add_argument("--latent", action="store_true", help="Use latent variable model.")
    parser.add_argument("--choose_N", type=int, default=-1, help="Sample choose_N objects.")
    parser.add_argument("--job_id", type=int, default=-1, help="Job Index for parallel processing.")
    parser.add_argument("--job_N", type=int, default=-1, help="Number of objects to divide.")

    args = parser.parse_args()

    check_64bit(gpu=not bool(args.cpu))

    if args.create_lc:
        objs = concat_light_curves(save_file_path=args.lc_file, progress_bar=args.progress)
        sys.exit("Created LC file. Exiting the program as requested.")

    # Filter objects by object_id that exist in the HDF5 file
    existing_object_ids = set()
    if args.ignore_existing:
        if os.path.exists(args.file):
            with h5py.File(args.file, "r") as hdf:
                existing_object_ids = set(hdf.keys())
                print(f"Found {len(existing_object_ids)} existing object IDs in {args.file}")
        else:
            print("WARNING! --ignore_existing flag but no existing file")

    filter_object_ids = args.filter_object_id if args.filter_object_id else []
    filter_object_ids = pd.read_csv(args.filter_file, dtype={"object_id": str})["object_id"].values if args.filter_file else filter_object_ids
    print(f"Loaded {len(filter_object_ids)=}")
    if args.choose_N > 0:
        filter_object_ids = np.random.choice(filter_object_ids, size=args.choose_N, replace=False)
        print(f"After choosing, total of {len(filter_object_ids)=}")

    elif args.job_id > -1:
        subarrays = [filter_object_ids[i:i + args.job_N] for i in range(0, len(filter_object_ids), args.job_N)]
        filter_object_ids = subarrays[args.job_id]
        print(f"Job ID {args.job_id} processing {filter_object_ids=}")

    if len(filter_object_ids) > 0:
        print(f"Filtering object IDs: {len(filter_object_ids)}")

    objs = concat_light_curves(filter_object_ids=filter_object_ids, existing_object_ids=existing_object_ids, N=args.N, skip=args.skip, save_file_path=args.lc_file, progress_bar=args.progress)
    if args.create_lc:
        sys.exit("Created LC file. Exiting the program as requested.")
    print(f"Loaded {len(objs)} objects from concat_light_curves")

    #objs = populate_sdss_fields(objs)

    Model = MyMultiVarModel
    if args.latent:
        print("Using latent model (with BLR contribution)")
        Model = MyMultiVarModelLatent

    # After loading objs
    print("--- Joint fitting")
    batch_data = []
    for i, obj in enumerate(objs):
        # Prepare each object's data for the joint model
        result = fit_multiband(Model, obj, nwarm=args.nwarm, nsamp=args.nsamp, progress_bar=args.progress, plot=False, svi=False, fit=False)
        if result is None:
            continue
        obj['i'] = i
        obj |= result
        # Run bestP for each object
        n_bands = len(obj['clean_bands'])
        bestP = initSampler(jax.random.PRNGKey(i), 1, 5, obj['X'], obj['y'], obj['yerr'], obj['clean_bands'], obj['z'])
        m = Model(
            obj['X'], obj['y'], obj['yerr'], 
            kernels.quasisep.Exp(jnp.array([1, 1])),
            zero_mean=has_lag, has_jitter=has_jitter, has_lag=has_lag,
            clean_bands=obj['clean_bands'], z=obj['z']
        )
        save_combined_plot(bestP, m, obj['X'], obj['y'], obj['yerr'], obj['band_idx'], obj, fit_bestP=True)

        num_params = sum(p.size for p in bestP.values())
        batch_data.append({
            'object_id': obj['object_id'],
            'X': obj['X'],
            'y': obj['y'],
            'yerr': obj['yerr'],
            'clean_bands': obj['clean_bands'],
            'band_idx': obj['band_idx'],
            'z': obj['z'],
            'bestP': bestP,
            # add any other fields needed by your model
        })
    num_params = sum(p.size for p in batch_data[0]['bestP'].values())
    num_objects = len(batch_data)
    print(f"Running joint fit on {len(batch_data)} objects...")

    estimated_nchains = 2*((num_params - 6)*len(batch_data) + 6)
    if args.nchains < 1:
        nchains = estimated_nchains
    else:
        nchains = args.nchains
    print(f"{args.nwarm=}, {args.nsamp=}, {args.nchains=}, estimated num_chains: {estimated_nchains}, {num_params=}, {len(batch_data)=}")

    # Print estimated memory usage
    memory_est = estimate_numpyro_gpu_memory_vectorized(num_params_per_object=num_params, num_objects=num_objects, num_chains=nchains, num_samples=args.nsamp, num_warmup=args.nwarm)
    print(f"Estimated GPU memory usage (vectorized): {memory_est:.2f} GB")
    
    init_strategy = numpyro.infer.init_to_sample()
    print("Done with numpyro.infer.init_to_sample")

    # emcee works better than NUTS for multimodal posteriors
    nuts_kernel = AIES(
        numpyro_joint_model,
        moves={AIES.DEMove() : 0.9, AIES.StretchMove() : 0.1},
        init_strategy=init_strategy,
        )
    #nuts_kernel = NUTS(numpyro_joint_model, init_strategy=init_strategy)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=args.nwarm,
        num_samples=args.nsamp,
        #num_chains=(2*num_params - 6)*len(batch_data) + 6,
        num_chains=nchains,
        progress_bar=args.progress,
        chain_method="vectorized",
    )
    mcmc.run(jax.random.PRNGKey(0), Model, batch_data)
    samples_flat = mcmc.get_samples(group_by_chain=False)
    diagnostics = mcmc.get_extra_fields()

    print("Done with MCMC run")

    # Save and plot the results
    results = []
    for i, obj in enumerate(batch_data):
        for k, v in samples_flat.items():
            print(v.shape, k)
        # The universal parameters are 1D
        obj_samples_clean = {
            k: v[:, i] if k not in ['eta_A1', 'eta_A2', 'eta_tau1', 'eta_tau2'] else v
            for k, v in samples_flat.items()
        }
        print(obj_samples_clean['log_jitter'].shape)
        result = process_samples(obj_samples_clean, obj)
        # plot
        if args.plot:
            m = Model(
                obj['X'], obj['y'], obj['yerr'], 
                kernels.quasisep.Exp(jnp.array([1, 1])),
                zero_mean=has_lag, has_jitter=has_jitter, has_lag=has_lag,
                clean_bands=['u','g','r','i','z'], z=obj['z']
            )
            psd_results = compute_psd_from_samples(obj_samples_clean, obj["clean_bands"])
            save_combined_plot(obj_samples_clean, m, obj['X'], obj['y'], obj['yerr'], obj['band_idx'], result, fit_bestP=False, psd_results=psd_results)
            dump_mcmc_diagnostics(mcmc, obj, i, len(batch_data))
            plot_trace_numpyro_for_object(mcmc, obj, i, len(batch_data))
            plot_posterior_for_object(mcmc, obj, i, len(batch_data))
        results.append(obj | result)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++", flush=True)
        print(f"Quasar {i+1}/{len(batch_data)} Object ID: {obj['object_id']}", flush=True)

        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ Done fitting all objects")
        if args.file:
            print("Saving results to ", args.file)
            append_hdf5_file(results, args.file)
        else:
            print("Warning!! Not saving results to file.")

    sys.exit("Exiting the program as requested.")
