
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["NUMEXPR_NUM_THREADS"] = "32"
os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["VECLIB_MAXIMUM_THREADS"] = "32"
os.environ["NUMBA_NUM_THREADS"] = "32"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

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

import jax
import jax.numpy as jnp


import numpyro
from numpyro import infer
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist

from tinygp import kernels, solvers

print("Total device count:", jax.local_device_count())
numpyro.set_host_device_count(30)
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import warnings

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

from multiband_fit_utils import *
from multiband_fit_plotting import *

num_samples = 250

# define params
zero_mean = False
has_jitter = True
has_lag = True

lambda_pivot = {
    'u': 3543,  # SDSS u-band
    'g': 4770,  # SDSS g-band
    'r': 6231,  # SDSS r-band
    'i': 7625,  # SDSS i-band
    'z': 9134,  # SDSS z-band
    'y': 9633,  # PS1 y-band
}

filters = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5} # harcoded filter order for SDSS
bands = ['u', 'g', 'r', 'i', 'z']#, 'y']

# class MyMultibandLowRank(MultibandLowRank):
#     amplitudes: jnp.ndarray
#     amplitudes_blr: jnp.ndarray
#     lag_blr: jnp.ndarray

#     def observation_model(self, X) -> JAXArray:
#         return (self.amplitudes[X[1]] * self.kernel.observation_model(X[0]) + 
#                 self.amplitudes_blr[X[1]] * self.kernel.observation_model(X[0] - self.lag_blr[X[1]]))
    
class MyMultibandLowRank(tinygp.kernels.Kernel):
    kernel: tinygp.kernels.Kernel
    amplitudes: jnp.ndarray
    amplitudes_blr: jnp.ndarray
    lag_blr: jnp.ndarray

    def __init__(self, kernel, amplitudes, amplitudes_blr, lag_blr) -> None:
        self.kernel = kernel
        self.amplitudes = amplitudes
        self.amplitudes_blr = amplitudes_blr
        self.lag_blr = lag_blr

    def evaluate(self, X1, X2) -> JAXArray:
        t1, b1 = X1
        t2, b2 = X2

        # This does a cross-band average of the lag
        lag_band = (self.lag_blr[b1] + self.lag_blr[b2])/2

        k_conti = self.amplitudes[b1] * self.amplitudes[b2] * self.kernel.evaluate(t1, t2)
        k_blr = self.amplitudes_blr[b1] * self.amplitudes_blr[b2] * self.kernel.evaluate(t1, t2 - lag_band)

        return k_conti + k_blr
# Override MultiVarModel
class MyMultiVarModel(MultiVarModel):
    clean_bands: JAXArray

    def __init__(
        self,
        X: JAXArray,
        y: JAXArray | NDArray,
        yerr: JAXArray | NDArray,
        kernel: tinygp.kernels.quasisep.Quasisep,
        **kwargs,
    ) -> None:
        super().__init__(X, y, yerr, kernel, **kwargs)
        self.clean_bands = kwargs.get("clean_bands", None)

    @staticmethod
    def mean_func(
        zero_mean: bool, nBand: int, params: dict[str, JAXArray], X: JAXArray
    ) -> JAXArray:
        if zero_mean is True:
            means = jnp.zeros(nBand)
        else:
            time_center = (jnp.max(X[0]) + jnp.min(X[0])) / 2
            means = jnp.atleast_1d(params["mean"]) + params["poly1"] * (X[0] / (365.25*1000))
            #means = jnp.atleast_1d(params["mean"]) + params["poly1"] * X[0]# / (365.25*100)
            #means = jnp.atleast_1d(params["mean"])# + jnp.expm1(jnp.log1p(params["poly1"]) + (jnp.log1p(X[0]/365.25*10000)))
            #means = jnp.atleast_1d(params["mean"]) + params["poly1"] * (X[0] / (365.25*100)) + params["poly2"] * (X[0] / (365.25*100))**2 
        return means[X[1]]
    
    def _build_gp(
        self, params: dict[str, JAXArray]
    ) -> tuple[GaussianProcess, JAXArray]:
        # log amp + mean
        log_amps = self.my_amp_transform(params)
        log_amps_blr = self.my_amp_transform_blr(params)
        means = partial(
            MyMultiVarModel.mean_func, self.zero_mean, log_amps.shape[0], params
        )

        # time axis transform: t and band are not sorted,
        # inds gives the sorted indices for the new_t
        X, inds = self.lag_transform(self.X, self.has_lag, params)
        t = X[0]
        band = X[1]

        # add jitter to the diagonal
        if self.has_jitter is True:
            diags = self.diag[inds] + (jnp.exp(params["log_jitter"]) ** 2)[band[inds]]
        else:
            diags = self.diag[inds]

        # def kernel
        # kernel = quasisep.MultibandLowRank(
        #     amplitudes=jnp.exp(log_amps),
        #     kernel=self.kernel_def(jnp.exp(params["log_kernel_param"])),
        # )
        kernel = MyMultibandLowRank(
            amplitudes=jnp.exp(log_amps),
            amplitudes_blr=jnp.exp(log_amps_blr),
            lag_blr=jnp.exp(params["log_lag_blr"])[band[inds]],
            kernel=self.kernel_def(jnp.exp(params["log_kernel_param"])),
        )
        return (
            GaussianProcess(
                kernel,
                (t[inds], band[inds]),
                diag=diags,
                mean=means,
            ),
            inds,
        )
    def my_amp_transform_blr(self, params: dict[str, JAXArray]) -> JAXArray:
        return jnp.atleast_1d(params["log_amp_delta_blr"])

    def my_amp_transform(self, params: dict[str, JAXArray]) -> JAXArray:
        b = params["beta"]
        params["log_amp_delta"] = jnp.array([b*np.log(lambda_pivot[band]/lambda_pivot['u']) for band in self.clean_bands[1:]]) # comment this out for old version
        r = jnp.insert(jnp.atleast_1d(params["log_amp_delta"]), 0, 0.0)
        return r
    
    @eqx.filter_jit
    def log_prob(self, params: dict[str, JAXArray]) -> JAXArray:
        """Calculate the log probability of the input parameters.

        Args:
            params (dict[str, JAXArray]): Model parameters.

        Returns:
            JAXArray: Log probability of the input parameters.
        """
        gp, inds = self._build_gp(params)
        log_prob = gp.log_probability(y=self.y[inds])
        jax.debug.print("Log probability: {log_prob}", log_prob=log_prob)
        return log_prob

    def sample(self, params: dict[str, JAXArray]) -> None:
        """A convience function for intergrating with numpyro for MCMC sampling.

        Args:
            params (dict[str, JAXArray]): Model parameters.
        """
        gp, inds = self._build_gp(params)
        numpyro.sample("gp", gp.numpyro_dist(), obs=self.y[inds])

    @eqx.filter_jit
    def pred(
        self, params: dict[str, JAXArray], X: JAXArray
    ) -> tuple[JAXArray, JAXArray]:
        """Make conditional GP prediction.

        Args:
            params (dict[str, JAXArray]): A dictionary containing model
                parameters.
            X (JAXArray): The time and band information for creating the
                conditional GP prediction.

        Returns:
            tuple[JAXArray, JAXArray]: A tuple of the mean GP prediction and
        """
        # transform time axis
        new_X, inds = self.lag_transform(X, self.has_lag, params)

        # build gp, cond
        gp, inds = self._build_gp(params)
        _, cond = gp.condition(self.y[inds], new_X)

        return cond.loc, jnp.sqrt(cond.variance)

    
def initSampler(key, nSample, nBand=len(bands)):
    # split keys
    subkeys = jax.random.split(key, 10)

    # uniform sampler
    lagSampler = UniformInit(nBand-1, [-10, 10])
    lagBLRSampler = UniformInit(nBand, [-2, np.log(20)])
    meanSampler = UniformInit(nBand, [-1, 1])
    poly1Sampler = UniformInit(1, [-1000, 1000])
    #poly2Sampler = UniformInit(1, [-10, 10])
    logAmpDeltaSampler = UniformInit(nBand-1, [-2, 0.0])
    logAmpDeltaBLRSampler = UniformInit(nBand, [-2, 0.0])
    logJitterSampler = UniformInit(nBand, [jnp.log(1e-10), jnp.log(0.1)])
    betaSampler = UniformInit(1, [-2.0, 0.0])

    # kernel init
    kernelSampler = DRWInit([jnp.log(10**2.5), jnp.log(10**4.5)], [jnp.log(0.01), jnp.log(1.0)])

    return {
        "log_kernel_param": kernelSampler(subkeys[0], nSample),
        "log_amp_delta": logAmpDeltaSampler(subkeys[1], nSample),
        "log_amp_delta_blr": logAmpDeltaBLRSampler(subkeys[7], nSample),
        "mean": meanSampler(subkeys[2], nSample),
        "poly1": poly1Sampler(subkeys[6], nSample),
        #"poly2": poly2Sampler(subkeys[7], nSample),
        "lag": lagSampler(subkeys[3], nSample),
        "log_lag_blr": lagBLRSampler(subkeys[8], nSample),
        "log_jitter": logJitterSampler(subkeys[4], nSample),
        "beta": betaSampler(subkeys[5], nSample),
    }

def numpyro_model(X, yerr, y=None, bestP=None, clean_bands=None):
    # kernel param
    flat_normal = dist.Normal(bestP["log_kernel_param"], jnp.array([5.0, 5.0]))
    diag_normal = dist.Independent(flat_normal, 1)
    log_kernel_param = numpyro.sample("log_kernel_param", diag_normal)

    # log amp delta
    #log_amp_delta = numpyro.sample(
    #   "log_amp_delta", dist.Normal(bestP["log_amp_delta"], 2.0)
    #) # comment this out when using beta
    log_amp_delta_blr = numpyro.sample(
      "log_amp_delta_blr", dist.Normal(bestP["log_amp_delta_blr"], 2.0)
    )

    # lag
    lag = numpyro.sample("lag", dist.Normal(bestP['lag'], 10))
    log_lag_blr = numpyro.sample("log_lag_blr", dist.Normal(bestP['log_lag_blr'], 2))
    
    # log jitter, mean => the prior for these two should be set small, otherwise
    # it is hard to converge
    log_jitter = numpyro.sample("log_jitter", dist.Normal(bestP["log_jitter"], 0.1))
    
    mean = numpyro.sample("mean", dist.Normal(bestP['mean'], .1))
    poly1 = numpyro.sample("poly1", dist.Normal(bestP['poly1'], 1))
    #poly2 = numpyro.sample("poly2", dist.Normal(bestP['poly2'], 2)) 
    beta = numpyro.sample("beta", dist.Normal(bestP['beta'], 0.5))


    # kernel
    k = kernels.quasisep.Exp(*jnp.exp(log_kernel_param))
    m1 = MyMultiVarModel(X, y, yerr, k, zero_mean=zero_mean, has_jitter=has_jitter, has_lag=has_lag, clean_bands=clean_bands)

    sample_params = {
        "log_kernel_param": log_kernel_param,
        #"log_amp_delta": log_amp_delta, # comment this out when using beta
        "log_amp_delta_blr": log_amp_delta_blr, 
        "lag": lag,
        "log_lag_blr": log_lag_blr,
        "mean": mean,
        "poly1": poly1, 
        #"poly2": poly2,
        "log_jitter": log_jitter,
        "beta": beta,
    }
    m1.sample(sample_params)


def fit_multiband(data, progress_bar=False, plot=False):
    times = data['times']
    mags = data['mags']
    magerrs = data['magerrs']
    
    red_bands = bands_redder_than_5000(data['z'])
    blue_bands = bands_bluer_than_lyman_alpha(data['z'])

    clean_bands = list(set(bands) - set(blue_bands))
    # Reorder clean_bands to match the desired order
    clean_bands = list(sorted(clean_bands, key=lambda band: ['u', 'g', 'r', 'i', 'z', 'y'].index(band)))
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

    # Reject outliers in moving window
    window_size = 20
    mask_outlier = np.ones(len(y), dtype=bool)
    for i in range(len(y)):
        if i < window_size or i >= len(y) - window_size:
            continue
        window = y[i - window_size:i + window_size + 1]
        if jnp.abs(y[i] - jnp.nanmean(window)) > 2.5 * st.median_abs_deviation(window):
            mask_outlier[i] = False

    # Reject outliers using a percentile-based approach
    # lower_percentile = 2
    # upper_percentile = 98
    # mask_outlier = (y >= np.percentile(y, lower_percentile)) & (y <= np.percentile(y, upper_percentile))

    X = (jnp.array(all_times[mask_outlier]) - jnp.min(all_times[mask_outlier]), jnp.array(band_idx[mask_outlier]))
    y = jnp.array(y[mask_outlier])
    yerr = jnp.array(yerr[mask_outlier])
    t = jnp.array(t[mask_outlier])

    # define kernel
    initial_drw_params = {"log_kernel_param": jnp.log(np.array([100.0, 0.35]))}
    k = kernels.quasisep.Exp(*jnp.exp(initial_drw_params["log_kernel_param"]))


    # define model
    m1 = MyMultiVarModel(
        X, y, yerr, k, zero_mean=zero_mean, has_jitter=has_jitter, has_lag=has_lag, clean_bands=clean_bands
    )
    bestP, logProb = fit(model=m1, 
                        optimizer=optax.adam(learning_rate=0.1),
                        initSampler=initSampler,
                        prng_key=jax.random.PRNGKey(0),
                        nSample=1, nIter=2, nBest=5)
    print(bestP)
    data['clean_bands'] = clean_bands
    save_combined_plot_bestp(bestP, m1, X, y, yerr, band_idx[mask_outlier], data)

    for k in bestP.keys():
        bestP[k] += 1e-4 * np.random.randn(*bestP[k].shape) 

    nuts_kernel = NUTS(
        partial(numpyro_model, bestP=bestP, clean_bands=clean_bands),
        dense_mass=True,
        target_accept_prob=0.9,
        # adapt_step_size=True,
    )

    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_samples,
        num_samples=num_samples,
        num_chains=2,
        progress_bar=progress_bar,
    )

    mcmc.run(jax.random.PRNGKey(1), X, yerr, y=y)
    samples = mcmc.get_samples(group_by_chain=False)
    diagnostics = mcmc.get_extra_fields()
    if np.all(diagnostics['diverging']):
        print(f"Diverging MCMC for quasar {data['object_id']}, skipping.", flush=True)
        return None
    
    log_tau_RF = np.log10(np.exp(samples['log_kernel_param'][:, 0])/(1+data['z']))
    lower, median, upper = np.percentile(log_tau_RF, [16, 50, 84])
    log_tau_RF_err = 0.5 * (upper - lower) # symmetric uncertainties
    log_tau_RF = median


    lambda_ref = 2500 # Any reference wavelength
    lambda_pivot_RF = lambda_pivot['u']/(1 + data['z'])
    
    log_sigma_RF = np.log10(np.exp(samples['log_kernel_param'][:, 1] + samples['beta']*np.log(lambda_ref/lambda_pivot_RF)))
    lower, median, upper = np.percentile(log_sigma_RF, [16, 50, 84])
    log_sigma_RF_err = 0.5 * (upper - lower) # symmetric uncertainties
    log_sigma_RF = median

    log_sigma = np.log10(np.exp(samples['log_kernel_param'][:, 1]))
    beta = samples['beta']
    log_amp_delta = np.array([beta*np.log(lambda_pivot[band]/lambda_pivot['u']) for band in clean_bands])
    #log_amp_delta = samples["log_amp_delta"].T # comment out this line when using beta
    #log_amp_delta = jnp.insert(log_amp_delta, 0, np.zeros(log_amp_delta.shape[1]), axis=0) # comment out this line when using beta

    log_sigma_band = log_sigma+log_amp_delta

    # beta
    lower, median, upper = np.percentile(beta, [16, 50, 84])
    beta_err = 0.5 * (upper - lower) # symmetric uncertainties
    beta = median

    # log_amp_delta
    lower, median, upper = np.percentile(log_amp_delta, [16, 50, 84], axis=1)
    log_amp_delta_err = 0.5 * (upper - lower) # symmetric uncertainties
    log_amp_delta = median

    # log_sigma_band
    lower, median, upper = np.percentile(log_sigma_band, [16, 50, 84], axis=1)    
    log_sigma_band_err = 0.5 * (upper - lower) # symmetric uncertainties
    log_sigma_band = median

    # log_sigma
    lower, median, upper = np.percentile(log_sigma, [16, 50, 84], axis=0)
    log_sigma_err = 0.5 * (upper - lower) # symmetric uncertainties
    log_sigma = median

    log_jitter = np.percentile(np.log10(np.exp(2*samples['log_jitter'])), 50, axis=0)

    lower, median, upper = np.percentile(samples['poly1'], [16, 50, 84], axis=0)
    poly1_err = 0.5 * (upper - lower) # symmetric uncertainties
    poly1 = median

    # lower, median, upper = np.percentile(samples['poly2'], [16, 50, 84], axis=0)
    # poly2_err = 0.5 * (upper - lower) # symmetric uncertainties
    # poly2 = median

    lower, median, upper = np.percentile(samples['mean'], [16, 50, 84], axis=0)
    mean_err = 0.5 * (upper - lower) # symmetric uncertainties
    mean = median

    lower, median, upper = np.percentile(samples['log_amp_delta_blr'], [16, 50, 84], axis=0)
    log_amp_delta_blr_err = 0.5 * (upper - lower) # symmetric uncertainties
    log_amp_delta_blr = median

    lower, median, upper = np.percentile(samples['log_lag_blr'], [16, 50, 84], axis=0)
    log_lag_blr_err = 0.5 * (upper - lower) # symmetric uncertainties
    log_lag_blr = median

    lower, median, upper = np.percentile(samples['lag'], [16, 50, 84], axis=0)
    lag_err = 0.5 * (upper - lower) # symmetric uncertainties
    lag = median



    d = dict(object_id=data['object_id'],
            z=data['z'],
            log_tau_RF=log_tau_RF,
            log_tau_RF_err=log_tau_RF_err,
            beta=beta,
            beta_err=beta_err,
            log_sigma_RF=log_sigma_RF,
            log_sigma_RF_err=log_sigma_RF_err,
            log_sigma_band=log_sigma_band,
            log_sigma_band_err=log_sigma_band_err,
            log_amp_delta=log_amp_delta,
            log_amp_delta_err=log_amp_delta_err,
            log_sigma=log_sigma,
            log_sigma_err=log_sigma_err,
            log_jitter=log_jitter,
            poly1=poly1,
            poly1_err=poly1_err,
            # poly2=poly2,
            # poly2_err=poly2_err,
            mean=mean,
            mean_err=mean_err,
            clean_bands=clean_bands,
            log_amp_delta_blr=log_amp_delta_blr,
            log_amp_delta_blr_err=log_amp_delta_blr_err,
            log_lag_blr=log_lag_blr,
            log_lag_blr_err=log_lag_blr_err,
            lag=lag,
            lag_err=lag_err,)
    
    if plot:
        save_combined_plot(samples, m1, X, y, yerr, band_idx[mask_outlier], d)
        plot_mcmc_traces(samples, d)
        plot_posterior(samples, data, clean_bands=clean_bands)
    
    return d


def process_quasar(i_data, n=0, progress_bar=False, plot=False):
    i, data = i_data
    #print(f"Processing quasar {i}/{n} ({data['object_id']})", flush=True)

    # Load the quasar data
    result = fit_multiband(data, progress_bar=progress_bar, plot=plot)
    if result is None:
        print(f"Skipping quasar {data['object_id']} due to diverging MCMC.")
        return None
    data['i'] = i
    data |= result


    #print(f"Quasar {i}/{n} ({data['object_id']}): log_tau_RF={data['log_tau_RF']:.3f}±{data['log_tau_RF_err']:.3f}, log_sigma_RF={data['log_sigma_RF']}±{data['log_sigma_RF_err']}", flush=True)
    return data

def concat_light_curves(N=None, skip=None, filter_object_ids=None, save_file_path=None):
    if save_file_path and os.path.exists(save_file_path):
        print(f"Loading data from {save_file_path}")
        s82_objs = load_s82_from_hdf5(save_file_path)
        if filter_object_ids is not None:
            # Filter the loaded objects based on the provided object IDs
            s82_objs = [obj for obj in s82_objs if obj['object_id'] in filter_object_ids]
        #return s82_objs
    else: 
        s82_objs = []
    # Load the S82 data from the FITS file
    hdul = fits.open('data/dr16q_prop_May01_2024.fits')
    fits_data = hdul[1].data  # Assuming the data is in the first extension    

    cat = pd.read_parquet(f"data/S82/Catalog.parquet").set_index('idx')
    sdss = pd.read_parquet(f"data/S82/dr16s82_sdssLCRaw.parquet")
    sdss = sdss[sdss.mjd.notna() & (len(sdss.mjd) > 0)]
    ps1 = pd.read_parquet(f"data/S82/dr16s82_ps1LCRaw.parquet")
    ztf = pd.read_parquet(f"data/S82/dr16s82_ZuberLCRaw.parquet")

    # Find elements in cat where objectId exists in the list of objectId of sdss
    match_object_ids = set(sdss.objectId) if filter_object_ids is None else filter_object_ids
    matching_indices = cat[cat.objectId.isin(match_object_ids)].index
    cat = cat.loc[matching_indices]
    cat = cat[skip:] if skip else cat
    cat = cat[:N] if N else cat

    print(f"Found {len(cat)} matching objects in concat_light_curves", len(cat))

    # Loop through the data and extract the relevant information        
    for idx, row in tqdm(cat.iterrows(), total=len(cat), desc="Processing quasars"):
        object_id = row['objectId']
        if object_id in [obj['object_id'] for obj in s82_objs]:
            continue

        # Filter light curves for the current object_id
        sdss_lc = sdss[sdss.objectId == row['objectId']].copy()
        ps1_lc = ps1[ps1.ps1objID == int(row['ps1objID'])].copy() if row['ps1objID'] is not None else pd.DataFrame()
        ztf_lc = ztf[ztf.ps1objID == int(row['ps1objID'])].copy() if row['ps1objID'] is not None else pd.DataFrame()

        #print("LC len: ", len(sdss_lc), len(ps1_lc), len(ztf_lc))

        # Combine light curves from different catalogs
        times = {}
        mags = {}
        magerrs = {}
        magerrs_mean = []

        for band in bands:  
            sdss_ps1_offset = {
                'g': row.sdss_g_qg - row.ps1_g_qg,
                'r': row.sdss_r_qg - row.ps1_r_qg,
                'i': row.sdss_i_qg - row.ps1_i_qg,
                'z': row.sdss_z_qg - row.ps1_z_qg,
            }

            offset = sdss_ps1_offset[band] if band in sdss_ps1_offset else 0.0

            times[band] = np.concatenate([
                sdss_lc[sdss_lc.filterID == filters[band]].mjd.values if not sdss_lc.empty else [],
                ps1_lc[ps1_lc.filterID == filters[band]].obsTime.values if not ps1_lc.empty else [],
                ztf_lc[ztf_lc.filterID == filters[band]].mjd.values if not ztf_lc.empty else []
            ])

            mags[band] = np.concatenate([
                sdss_lc[sdss_lc.filterID == filters[band]].psMag.values if not sdss_lc.empty else [],
                ps1_lc[ps1_lc.filterID == filters[band]].psfMag.values + offset if not ps1_lc.empty else [],
                ztf_lc[ztf_lc.filterID == filters[band]].mag.values + offset if not ztf_lc.empty else []
            ])
            mags[band] = mags[band] - np.nanmean(mags[band])  # Center the magnitudes

            magerrs[band] = np.concatenate([
                sdss_lc[sdss_lc.filterID == filters[band]].psMagErr_p3.values if not sdss_lc.empty else [],
                ps1_lc[ps1_lc.filterID == filters[band]].psfMagErr_p3.values if not ps1_lc.empty else [],
                ztf_lc[ztf_lc.filterID == filters[band]].magerr_p3.values if not ztf_lc.empty else []
            ])
            # Select NaNs from mags and magerrs
            nan_mask = np.isnan(mags[band]) | np.isnan(magerrs[band])

            # Drop NaNs from mags and magerrs using the same indexes
            times[band] = times[band][~nan_mask]
            mags[band] = mags[band][~nan_mask]
            magerrs[band] = magerrs[band][~nan_mask]

            if len(times[band]) == 0 or len(mags[band]) == 0 or len(magerrs[band]) == 0:
                continue

            # Sort times, mags, magerrs by time
            sort_idx = np.argsort(times[band])
            times[band] = times[band][sort_idx]
            mags[band] = mags[band][sort_idx]
            magerrs[band] = magerrs[band][sort_idx]

            # Keep only the last X days of light curve in the rest frame
            # rest_frame_times = (times[band]-times[band].min()) / (1 + row['Z_SYS'])
            # mask_days = (rest_frame_times >= (rest_frame_times.max() - 2600)) 
            # times[band] = times[band][mask_days]
            # mags[band] = mags[band][mask_days]
            # magerrs[band] = magerrs[band][mask_days]

            # Ensure magerrs_mean calculation handles empty arrays
            #if len(magerrs[band]) == 0:
            #    magerrs_mean.append(np.nan)  # Default to 0.0 if no data is available
            #else:
            #    magerrs_mean.append(np.mean(magerrs[band]))

        # Skip if no data is available for the object
        if all((len(times[band]) == 0 or len(mags[band]) == 0 or len(magerrs[band]) == 0) for band in bands):
            print(f"No data available for object {object_id}, skipping.", flush=True)
            continue
        
        #rest_frame_times = (times['g']-times['g'].min()) / (1 + row['Z_SYS'])
        #print(f"Object {object_id} - Band g: {rest_frame_times.min()=} {rest_frame_times.max()=} diff: {rest_frame_times.max()-rest_frame_times.min()=}", flush=True)
        # Check if any band has rest frame time of 2800
        # has_valid_rest_frame_time = any(
        #     (((times[band] - times[band].min()) / (1 + row['Z_SYS'])).max() >= 2500)
        #     if len(times[band]) > 0 else False
        #     for band in bands
        # ) 
        # if not has_valid_rest_frame_time:
        #     #print(f"No bands with rest frame time of 2800 for object {object_id}, skipping.", flush=True)
        #     continue

        s82_objs.append({
            'object_id': object_id,
            'times': times,
            'mags': mags,
            'magerrs': magerrs,
            'magerrs_mean': magerrs_mean
        })

        #save_lc_plot(times, mags, magerrs, object_id, bands=bands)

    hdul.close()

    print(f"Found {len(s82_objs)} objects in concat_light_curves after time cut", len(s82_objs))

    s82_objs = populate_sdss_fields(s82_objs)

    if save_file_path:
        with h5py.File(save_file_path, "w") as hdf:
            for obj in s82_objs:
                object_id = obj["object_id"]
                group = hdf.create_group(object_id)

                # Save all attributes
                for key, value in obj.items():
                    if isinstance(value, dict):
                        sub_group = group.create_group(key)
                        for sub_key, sub_value in value.items():
                            sub_group.create_dataset(sub_key, data=sub_value)
                    else:
                        group.attrs[key] = value
        print(f"Saved {len(s82_objs)} LCs to {save_file_path}")

    return s82_objs

def load_s82_from_hdf5(file_path="s82_objs.h5"):
    s82_objs = []

    with h5py.File(file_path, "r") as hdf:
        for object_id in hdf.keys():
            group = hdf[object_id]
            data = {"object_id": object_id}

            # Load attributes
            for attr_key in group.attrs.keys():
                data[attr_key] = group.attrs[attr_key]

            # Load datasets
            for key in group.keys():
                if isinstance(group[key], h5py.Group):
                    data[key] = {}
                    for sub_key in group[key].keys():
                        data[key][sub_key] = group[key][sub_key][...]
                else:
                    data[key] = group[key][...]

            s82_objs.append(data)

    return s82_objs

def populate_sdss_fields(s82_objs):
    print(f"Populating SDSS fields: {len(s82_objs)}", flush=True)
    cat = pd.read_parquet(f"data/S82/Catalog.parquet").set_index('idx')
    hdul = fits.open('data/dr16q_prop_May01_2024.fits')
    fits_data = hdul[1].data  # Assuming the data is in the first extension    
    fits_data_2 = hdul[2].data  # Assuming the data is in the first extension    
    for d in tqdm(s82_objs, desc="Populating SDSS fields"):
        obj = cat.loc[cat['objectId'] == d['object_id']].iloc[0]
        c1 = SkyCoord(fits_data['RA'], fits_data['DEC'], unit='deg')
        c2 = SkyCoord(obj['RA'], obj['DEC'], unit='deg')
        sep = c1.separation(c2).to(u.arcsec)
        j = np.argwhere(sep < 1*u.arcsec).flatten()
        if len(j) == 0:
            print(f"Skipping entry {d['object_id']} as it does not exist in the fits data.")
            continue
        
        j = j[0]  # Get the first index if there are multiple matches
        d['ra'] = obj['RA']
        d['dec'] = obj['DEC']
        d['z'] = obj['Z_SYS']
        d['sdss_name'] = fits_data['SDSS_NAME'][j]  # Extract SDSS_NAME
        d['log_lbol'] = fits_data['LOGLBOL'][j]  # Extract log Lbol values
        d["log_lbol_err"] = fits_data['LOGLBOL_ERR'][j]  # Extract log Lbol error values
        d['log_mbh'] = fits_data['LOGMBH'][j]  # Extract log MBH values
        d['log_mbh_err'] = fits_data['LOGMBH_ERR'][j]  # Extract log MBH error values
        d['log_ledd_ratio'] = fits_data['LOGLEDD_RATIO'][j]  # Extract log L/edd values
        d['log_ledd_ratio_err'] = fits_data['LOGLEDD_RATIO_ERR'][j]  # Extract log L/edd error values

    return s82_objs

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Process quasars with optional filtering.")
    parser.add_argument("--filter_object_id", nargs="+", help="List of object IDs to filter.")
    parser.add_argument("--N", type=int, help="Number of objects to process.")
    parser.add_argument("--skip", type=int, help="Number of objects to skip.")
    parser.add_argument("--chunk_size", type=int, default=500, help="Chunk size for processing objects.")
    parser.add_argument("-f", "--file", type=str, help="Path to the file to append (read and write) objects.") 
    parser.add_argument("--lc_file", type=str, help="Path to the light curve file.")
    parser.add_argument("--plot", action="store_true", help="Enable plotting of results.")
    args = parser.parse_args()


    filter_object_ids = set(args.filter_object_id) if args.filter_object_id else None
    if filter_object_ids is not None:
        print(f"Filtering object IDs: {filter_object_ids}")
        objs = concat_light_curves(filter_object_ids=filter_object_ids)
        objs = populate_sdss_fields(objs)
        for i, obj in enumerate(objs):
            q = process_quasar((i, obj), n=len(objs), progress_bar=True, plot=True)
            fields_to_filter = ['times', 'mags', 'magerrs']
            q = {k: v for k, v in q.items() if k not in fields_to_filter}
            print(q['z'], q['clean_bands'])
    
        sys.exit("Exiting the program as requested.")


    # Filter objects by object_id that exist in the HDF5 file
    existing_object_ids = set()
    if os.path.exists(args.file):
        with h5py.File(args.file, "r") as hdf:
            existing_object_ids = set(hdf.keys())
    else:
        existing_object_ids = []

    cat = pd.read_parquet(f"data/S82/Catalog.parquet").set_index('idx')
    sdss = pd.read_parquet(f"data/S82/dr16s82_sdssLCRaw.parquet")
    sdss = sdss[sdss.mjd.notna() & (len(sdss.mjd) > 0)]

    # Find elements in cat where objectId exists in the list of objectId of sdss
    #filter_object_ids = set(pd.read_csv("data/object_ids_test.csv")["object_id"].values)
    match_object_ids = set(sdss.objectId) if filter_object_ids is None else filter_object_ids
    matching_indices = cat[cat.objectId.isin(match_object_ids)].index
    cat = cat[cat.objectId.isin(match_object_ids)]
    #cat = cat.loc[matching_indices]
    print(f"Total objects in catalog: {len(cat)}")
    print(f"Number of objects already processed in HDF5 file: {len(existing_object_ids)}")
    print(f"Number of objects that should be processed: {len(cat)-len(existing_object_ids)}")

    cat = cat[~cat['objectId'].isin(existing_object_ids)]
    filter_object_ids = set(cat['objectId'].values)
    print(f"Number of objects to process: {len(filter_object_ids)}")
    #objs = [obj for obj in objs if obj['object_id'] not in existing_object_ids]
    objs = concat_light_curves(filter_object_ids=filter_object_ids, N=args.N, skip=args.skip, save_file_path=args.lc_file)
    print(f"Loaded {len(objs)} objects from concat_light_curves")
    #sys.exit("Exiting the program as requested.")
    #objs = populate_sdss_fields(objs)

    for start_idx in range(0, len(objs), args.chunk_size):
        chunk = objs[start_idx:start_idx + args.chunk_size]
        print(f"({100*start_idx/len(objs):.1f}%) Processing chunk {start_idx // args.chunk_size + 1}/{(len(objs) + args.chunk_size - 1) // args.chunk_size}...")
        
        ctx = get_context("spawn")  # Safer for JAX when using multiprocessing
        results = []

        with ctx.Pool(processes=15) as pool:
            results = pool.map(partial(process_quasar, n=len(chunk), plot=args.plot), enumerate(chunk))

        quasar_list = [q for q in results if q is not None]

        # Append to HDF5 file if it exists, otherwise create a new one
        with h5py.File(args.file, "a") as hdf:
            for quasar in quasar_list:
                object_id = quasar["object_id"]
                if object_id in hdf:
                    continue

                group = hdf.create_group(object_id)
                for key, value in quasar.items():
                    if isinstance(value, dict):
                        sub_group = group.create_group(key)
                        for sub_key, sub_value in value.items():
                            sub_group.create_dataset(sub_key, data=sub_value)
                    else:
                        group.attrs[key] = value