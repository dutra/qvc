
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

# define params
zero_mean = False
has_jitter = True
has_lag = True


def initSampler(key, nSample, nBand=None):
    # split keys
    subkeys = jax.random.split(key, 14)

    # uniform sampler
    lagSampler = UniformInit(nBand-1, [-10, 10])
    loglagBLRSampler = UniformInit(nBand, [0, 5])
    logtauBLRSampler = UniformInit(1, [jnp.log(10**2.5), jnp.log(10**4.5)])
    meanSampler = UniformInit(nBand, [-1, 1])
    poly1Sampler = UniformInit(1, [-10, 10])
    logAmpDeltaSampler = UniformInit(nBand-1, [-2.0, 0.0])
    logAmpDeltaBLRSampler = UniformInit(nBand, [-5.0, -2.0])
    logJitterSampler = UniformInit(nBand, [jnp.log(1e-4), jnp.log(0.1)])

    # power laws
    etaBreakSampler = UniformInit(1, [2, 5])
    lamsSampler = UniformInit(1, [2400.0, 2600.0])
    # Colins
    # # sigma
    # etaA1Sampler = UniformInit(1, [-0.8, -0.6])
    # etaA2Sampler = UniformInit(1, [-0.1, 0.1])
    # # tau
    # etaTau1Sampler = UniformInit(1, [0.2,0.6])
    # etaA2Sampler = UniformInit(1, [-0.2, 0.2])
    # Updated (Kelly+2022, Yu+2022)
    # sigma
    etaA1Sampler = UniformInit(1, [-0.25, -0.05])
    etaA2Sampler = UniformInit(1, [-0.25, -0.05])
    # tau
    etaTau1Sampler = UniformInit(1, [0.2, 0.6])
    etaTau2Sampler = UniformInit(1, [0.2, 0.6])

    # kernel init
    kernelSampler = DRWInit([jnp.log(10**2.5), jnp.log(10**4.5)], [jnp.log(0.1), jnp.log(1.0)])
    logwSampler = UniformInit(1, [0.0, 1.0])
    
    return {
        "log_kernel_param": kernelSampler(subkeys[0], nSample),
        "log_w": logwSampler(subkeys[1], nSample),
        "log_amp_delta": logAmpDeltaSampler(subkeys[1], nSample),
        "log_amp_delta_blr": logAmpDeltaBLRSampler(subkeys[2], nSample),
        "mean": meanSampler(subkeys[3], nSample),
        "poly1": poly1Sampler(subkeys[4], nSample),
        "lag": lagSampler(subkeys[5], nSample),
        "log_lag_blr": loglagBLRSampler(subkeys[6], nSample),
        "log_tau_drw_blr": logtauBLRSampler(subkeys[7], nSample),
        "log_jitter": logJitterSampler(subkeys[8], nSample),
        # power laws
        "eta_A1": etaA1Sampler(subkeys[9], nSample),
        "eta_A2": etaA2Sampler(subkeys[10], nSample),
        "eta_tau1": etaTau1Sampler(subkeys[11], nSample),
        "eta_tau2": etaTau2Sampler(subkeys[12], nSample),
        "eta_break": etaBreakSampler(subkeys[13], nSample),
        "lam_s": lamsSampler(subkeys[14], nSample),
    }

def numpyro_model(Model, X, yerr, y=None, bestP=None, clean_bands=None, z=None):
    # --- Kernel and lag parameters ---
    log_w = numpyro.sample("log_w", dist.Normal(jnp.full_like(bestP["log_w"], jnp.log(5.0)), 1.0))
    log_amp_delta_blr = numpyro.sample("log_amp_delta_blr", dist.Normal(jnp.full_like(bestP["log_amp_delta_blr"], -8.0), 2.0))
    lag = numpyro.sample("lag", dist.Normal(jnp.full_like(bestP["lag"], 0.0), 10))
    log_lag_blr = numpyro.sample("log_lag_blr", dist.Normal(jnp.full_like(bestP["log_lag_blr"], 0.0), 2.0))
    log_tau_drw_blr = numpyro.sample("log_tau_drw_blr", dist.Normal(2.8, 2.0))
    log_jitter = numpyro.sample("log_jitter", dist.Normal(jnp.full_like(bestP["log_jitter"], np.log(1e-4)), 1.0))

    # --- Mean function parameters ---
    mean = numpyro.sample("mean", dist.Normal(jnp.full_like(bestP["mean"], 0.0), 0.1))
    poly1 = numpyro.sample("poly1", dist.Normal(0.0, 10.0))

    # --- Power law parameters ---
    powerlaw_priors = {
        "eta_A1": (-1.0, 1.0),
        "eta_A2": (-0.2, 1.0),
        "eta_tau1": (0.8, 1.0),
        "eta_tau2": (0.1, 1.0),
        "eta_break": (4, 0.1),
        "lam_s": (2500.0, 1.0),
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
        "log_w": log_w,
        "log_amp_delta_blr": log_amp_delta_blr,
        "lag": lag,
        "log_lag_blr": log_lag_blr,
        "log_tau_drw_blr": log_tau_drw_blr,
        "mean": mean,
        "poly1": poly1,
        "log_jitter": log_jitter,
        **powerlaw_samples,
    }

    # --- Evaluate model likelihood ---
    m.sample(sample_params)

def numpyro_joint_model(Model, batch_data):
    # --- Shared (universal) parameters ---
    # TODO: update the priors from the previous batch
    eta_A1 = numpyro.sample("eta_A1", dist.Normal(-1.0, 0.1))
    eta_A2 = numpyro.sample("eta_A2", dist.Normal(-0.2, 0.1))
    eta_tau1 = numpyro.sample("eta_tau1", dist.Normal(1.0, 0.1))
    eta_tau2 = numpyro.sample("eta_tau2", dist.Normal(0.1, 0.1))
    lam_s = numpyro.sample("lam_s", dist.Normal(2500.0, 50.0))
    eta_break = numpyro.sample("eta_break", dist.Normal(4, 0.2))

    for i, data in enumerate(batch_data):
        # Object-specific parameters
        log_kernel_param = numpyro.sample(f"log_kernel_param_{i}", dist.Uniform(jnp.array([2.0, -3.0]), jnp.array([10.0, 0.5])))
        log_amp_delta_blr = numpyro.sample(f"log_amp_delta_blr_{i}", dist.Normal(jnp.full((len(data['clean_bands']),), -8.0), 2.0))
        lag = numpyro.sample(f"lag_{i}", dist.Normal(jnp.zeros(len(data['clean_bands'])-1), 10))
        log_lag_blr = numpyro.sample(f"log_lag_blr_{i}", dist.Normal(jnp.zeros(len(data['clean_bands'])), 2.0))
        log_tau_drw_blr = numpyro.sample(f"log_tau_drw_blr_{i}", dist.Normal(2.8, 2.0))
        log_jitter = numpyro.sample(f"log_jitter_{i}", dist.Normal(jnp.full((len(data['clean_bands']),), np.log(1e-4)), 1.0))
        mean = numpyro.sample(f"mean_{i}", dist.Normal(jnp.zeros(len(data['clean_bands'])), 0.1))
        poly1 = numpyro.sample(f"poly1_{i}", dist.Normal(0.0, 10.0))

        params = {
            "log_kernel_param": log_kernel_param,
            "log_amp_delta_blr": log_amp_delta_blr,
            "lag": lag,
            "log_lag_blr": log_lag_blr,
            "log_tau_drw_blr": log_tau_drw_blr,
            "log_jitter": log_jitter,
            "mean": mean,
            "poly1": poly1,
            # --- Shared parameters ---
            "eta_A1": eta_A1,
            "eta_A2": eta_A2,
            "eta_tau1": eta_tau1,
            "eta_tau2": eta_tau2,
            "lam_s": lam_s,
            "eta_break": eta_break,
        }

        m = Model(
            data['X'], data['y'], data['yerr'],
            kernels.quasisep.Exp(*jnp.exp(log_kernel_param)),
            zero_mean=zero_mean, has_jitter=has_jitter, has_lag=has_lag,
            clean_bands=data['clean_bands'], z=data['z']
        )
        log_prob = m.log_prob(params)
        # numpyro does the summation for us
        numpyro.factor(f"loglike_{i}", log_prob)


def fit_multiband(Model, data, nwarm=500, nsamp=250, progress_bar=False, plot=False, svi=False, fit=True):
    times = data['times']
    mags = data['mags']
    data['mags_means'] = np.array([np.nanmean(mags[band]) for band in mags.keys()])
    for band in mags.keys():
       mags[band] = mags[band] - np.nanmean(mags[band])  # Center the magnitudes
    magerrs = data['magerrs']
    
    red_bands = bands_redder_than_5000(data['z'])
    blue_bands = bands_bluer_than_lyman_alpha(data['z'])

    clean_bands = list(set(bands) - set(blue_bands) - set(red_bands))
    # Reorder clean_bands to match the desired order
    clean_bands = list(sorted(clean_bands, key=lambda band: ['u', 'g', 'r', 'i', 'z', 'y'].index(band)))
    #clean_bands = bands
    data['clean_bands'] = clean_bands
    if len(clean_bands) == 0:
        print(f"No clean bands for quasar {data['object_id']}, skipping.", flush=True)
        return None
    # Combine
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
        batch_dict = {'X': X, 'y': y, 'yerr': yerr, 'clean_bands': clean_bands, 'z': data['z']}
        return batch_dict

    # define kernel
    initial_drw_params = {"log_kernel_param": jnp.log(np.array([100.0, 0.35]))}
    k = kernels.quasisep.Exp(*jnp.exp(initial_drw_params["log_kernel_param"]))

    # define model
    m1 = Model(
        X, y, yerr, k, zero_mean=zero_mean, has_jitter=has_jitter, has_lag=has_lag, clean_bands=clean_bands, z=data['z']
    )

    print("Initializing bestP.")
    bestP = initSampler(jax.random.PRNGKey(0), 1, len(clean_bands))
    print(bestP)

    for k in bestP.keys():
        bestP[k] += 1e-4 * np.random.randn(*bestP[k].shape)

    if svi == True:

        print('Starting SVI')

        # SVI
        #guide = numpyro.infer.autoguide.AutoDiagonalNormal(numpyro_model)
        #guide = numpyro.infer.autoguide.AutoLowRankMultivariateNormal(numpyro_model)
        guide = numpyro.infer.autoguide.AutoMultivariateNormal(numpyro_model)

        svi = numpyro.infer.SVI(
            model=numpyro_model,
            guide=guide,
            optim=numpyro.optim.Adam(1e-2),
            loss=numpyro.infer.Trace_ELBO(),
        )

        svi_state = svi.init(jax.random.PRNGKey(0), X, yerr, y=y, bestP=bestP, clean_bands=clean_bands)

        # Training loop
        def run_svi_training(svi_state):
            def svi_step(carry, _):
                svi_state = carry
                svi_state, loss = svi.update(svi_state, X, yerr, y=y, bestP=bestP, clean_bands=clean_bands)
                return svi_state, loss

            return lax.scan(svi_step, svi_state, None, length=1000)

        # JIT the wrapper function
        run_svi_training_jit = jax.jit(run_svi_training)
        svi_state, losses = run_svi_training_jit(svi_state)

        print(losses)

        params = svi.get_params(svi_state)
        print(guide.get_posterior(params))
        samples = guide.sample_posterior(jax.random.PRNGKey(1), params, sample_shape=(250,))
        print(samples)

    else:

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
        save_combined_plot(samples, m1, X, y, yerr, band_idx[mask_outlier], result)
        #plot_mcmc_traces(samples, result)
        plot_posterior(samples, data, clean_bands=clean_bands)
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
    parser.add_argument("--latent", action="store_true", help="Use latent variable model.")

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
            existing_object_ids = set()

    filter_object_ids = set(args.filter_object_id) if args.filter_object_id else set()
    filter_object_ids = set(pd.read_csv(args.filter_file, dtype={"object_id": str})["object_id"].values) if args.filter_file else filter_object_ids
    filter_object_ids = set(filter_object_ids) - set(existing_object_ids)
    if len(filter_object_ids) > 0:
        print(f"Filtering object IDs: {len(filter_object_ids)}")
    objs = concat_light_curves(filter_object_ids=filter_object_ids, N=args.N, skip=args.skip, save_file_path=args.lc_file, progress_bar=args.progress)
    if args.create_lc:
        sys.exit("Created LC file. Exiting the program as requested.")
    print(f"Loaded {len(objs)} objects from concat_light_curves")

    #objs = populate_sdss_fields(objs)

    Model = MyMultiVarModel

    if args.latent:
        print("Using latent model (with BLR contribution)")
        Model = MyMultiVarModelLatent

    # After loading objs
    if args.joint:
        batch_data = []
        for i, obj in enumerate(objs):
            # Prepare each object's data for the joint model
            result = fit_multiband(Model, obj, nwarm=args.nwarm, nsamp=args.nsamp, progress_bar=args.progress, plot=False, svi=False, fit=False)
            if result is None:
                continue
            obj['i'] = i
            obj |= result
            batch_data.append({
                'X': obj['X'],
                'y': obj['y'],
                'yerr': obj['yerr'],
                'clean_bands': obj['clean_bands'],
                'z': obj['z'],
                # add any other fields needed by your model
            })
            print(f"Running joint fit on {len(batch_data)} objects...")
            init_strategy = numpyro.infer.init_to_sample()

            # emcee works better than NUTS for multimodal posteriors
            nuts_kernel = AIES(
                partial(numpyro_model, bestP=bestP, clean_bands=clean_bands, z=data['z']),
                moves={AIES.DEMove() : 0.5, AIES.StretchMove() : 0.5},
                init_strategy=init_strategy,
                )
            mcmc = MCMC(
                nuts_kernel,
                num_warmup=nwarm, # This could be less than num_samples
                num_samples=nsamp,
                num_chains=2*num_params,
                progress_bar=True,
                chain_method="vectorized",
            )
            mcmc.run(jax.random.PRNGKey(0), batch_data)
            samples = mcmc.get_samples(group_by_chain=False)
            diagnostics = mcmc.get_extra_fields()

    else:
        for i, obj in enumerate(objs):
            print(f"Processing quasar {i}/{len(objs)} ({obj['object_id']})", flush=True)
            q = process_quasar(Model, (i, obj), n=len(objs), nwarm=args.nwarm, nsamp=args.nsamp, progress_bar=args.progress, plot=args.plot, svi=args.svi)
            if q is None:
                #print(f"Skipping quasar {obj['object_id']}, no data", flush=True)
                continue
            fields_to_filter = ['times', 'mags', 'magerrs']
            #filtered_q = {k: v for k, v in q.items() if k not in fields_to_filter}
            #print(filtered_q)
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n", flush=True)
            print(f"Quasar {i+1}/{len(objs)} Object ID: {q['object_id']}", flush=True)
            if args.file:
                append_hdf5_file([q], args.file)

    sys.exit("Exiting the program as requested.")
