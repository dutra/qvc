
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
from numpyro.infer import MCMC, NUTS, AIES, Predictive
import numpyro.distributions as dist

from tinygp import kernels, solvers

print("Total device count:", jax.local_device_count())
#numpyro.set_host_device_count(30)
jax.config.update("jax_enable_x64", True)
#jax.config.update("jax_platform_name", "cpu")
learning_rate=0.001

# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "gpu")
#learning_rate=0.0001

import warnings
import jax.scipy as jsp
from jax.scipy.special import erfc

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

from multiband_fit_utils import *
from multiband_fit_plotting import *

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

class MyMultibandLowRankOld(MultibandLowRank):
     amplitudes: jnp.ndarray
     amplitudes_blr: jnp.ndarray
     lag_blr: jnp.ndarray

     def observation_model(self, X) -> JAXArray:
         return (self.amplitudes[X[1]] * self.kernel.observation_model(X[0]) +
                self.amplitudes_blr[X[1]] * self.kernel.observation_model(X[0] - self.lag_blr[X[1]]))
    
class MyMultibandLowRank(tinygp.kernels.Kernel):
    sigma: float
    amplitudes: jnp.ndarray
    amplitudes_blr: jnp.ndarray
    lag_blr: jnp.ndarray
    tau_drw: jnp.ndarray
    tau_drw_blr: float

    def __init__(self, sigma, scale, amplitudes, amplitudes_blr, lag_blr, taus, tau_drw_blr) -> None:
        self.sigma = sigma
        self.amplitudes = amplitudes
        self.amplitudes_blr = amplitudes_blr
        self.lag_blr = lag_blr
        self.tau_drw = scale * taus
        self.tau_drw_blr = tau_drw_blr

    def coord_to_sortable(self, X) -> JAXArray:
        return X[0]

    def k(self, tau, tau_drw) -> JAXArray:
        tau = jnp.abs(tau)
        drw = jnp.exp(-tau / tau_drw)
        return drw

    #def k(self, tau, tau_drw, w=5) -> JAXArray:
    #    # Compute the analytic convolution of DRW and Gaussian kernels
    #    prefactor = 1 / (jnp.sqrt(2 * jnp.pi) * w)
    #    # IDEA: take w out of the prefactor multiply it back after
    #    exp_term = jnp.exp((w**2) / (2 * tau_drw**2) - jnp.abs(tau) / tau_drw)
    #    erfc_term = erfc((w / jnp.sqrt(2) / tau_drw) - (jnp.abs(tau) / jnp.sqrt(2) / w))
    #    return prefactor * exp_term * erfc_term

    def evaluate(self, X1, X2) -> JAXArray:
        t1, b1 = X1
        t2, b2 = X2

        # a is cont at t1
        # b is blr at t1
        # c is cont at t2
        # d is blr at t2
        cov_ac = (
            self.amplitudes[b1]
            * self.amplitudes[b2]
            * self.sigma**2
            * jnp.sqrt(
                self.k(t2 - t1, self.tau_drw[b1])
                * self.k(t2 - t1, self.tau_drw[b2])
            )
        )
        cov_ad = (
            self.amplitudes[b1]
            * self.amplitudes_blr[b2]
            * self.sigma**2
            * jnp.sqrt(
                self.k(t2 - t1, self.tau_drw[b1])
                * self.k(t2 - t1 - self.lag_blr[b2], self.tau_drw_blr)
            )
        )
        cov_bc = (
            self.amplitudes_blr[b1]
            * self.amplitudes[b2]
            * self.sigma**2
            * jnp.sqrt(
                self.k(t2 - t1 - self.lag_blr[b1], self.tau_drw_blr)
                * self.k(t2 - t1, self.tau_drw[b2])
            )
        )
        cov_bd = (
            self.amplitudes_blr[b1]
            * self.amplitudes_blr[b2]
            * self.sigma**2
            * jnp.sqrt(
                self.k(t2 - t1 - self.lag_blr[b1], self.tau_drw_blr)
                * self.k(t2 - t1 - self.lag_blr[b2], self.tau_drw_blr)
            )
        )

        return cov_ac + cov_ad + cov_bc + cov_bd

def log_broken_pl(lam, lam_s, d1, ep):
    return np.log10( ((lam/lam_s)**d1 + (lam/lam_s)**(d1+ep))**-1 )

# Override MultiVarModel
class MyMultiVarModel(MultiVarModel):
    clean_bands: JAXArray
    z: float

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
        self.z = kwargs.get("z", 0.0)

    @staticmethod
    def mean_func(
        zero_mean: bool, nBand: int, params: dict[str, JAXArray], X: JAXArray
    ) -> JAXArray:
        if zero_mean is True:
            means = jnp.zeros(nBand)
        else:
            time_centered = (X[0] - jnp.nanmean(X[0]))
            time_scaled = time_centered #/ (jnp.nanmax(X[0]) - jnp.nanmin(X[0]))
            means = jnp.atleast_1d(params["mean"]) + params["poly1"] * time_scaled
        return means[X[1]]
    
    def _build_gp(
        self, params: dict[str, JAXArray]
    ) -> tuple[GaussianProcess, JAXArray]:
        # log amp + mean
        log_amps = self.my_amp_transform(params)
        log_amps_blr = self.my_amp_transform_blr(params)
        log_taus = self.my_tau_drw_transform(params)

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
            sigma=jnp.exp(params["log_kernel_param"][1]),
            scale=jnp.exp(params["log_kernel_param"][0]),
            taus=jnp.exp(log_taus)[band[inds]],
            tau_drw_blr=jnp.exp(params["log_tau_drw_blr"]),
        )
        return (
            GaussianProcess(
                kernel,
                (t[inds], band[inds]),
                diag=diags + 1e-6,
                mean=means,
            ),
            inds,
        )
    def my_amp_transform_blr(self, params: dict[str, JAXArray]) -> JAXArray:
        return jnp.atleast_1d(params["log_amp_delta_blr"])

    def my_tau_drw_transform(self, params: dict[str, JAXArray]) -> JAXArray:
        eta_tau1 = params["eta_tau1"]
        eta_tau2 = eta_tau1 + params["ep_tau"]
        lam_s = params["lam_s"]/(1 + self.z)
        params["log_tau_delta"] = jnp.log(10) * jnp.array([log_broken_pl(lambda_pivot[band], lam_s, eta_tau1, eta_tau2) for band in self.clean_bands])
        return params["log_tau_delta"]

    def my_amp_transform(self, params: dict[str, JAXArray]) -> JAXArray:
        eta_A1 = params["eta_A1"]
        eta_A2 = eta_A1 + params["ep_A"]
        lam_s = params["lam_s"]/(1 + self.z)
        params["log_amp_delta"] = jnp.log(10) * jnp.array([log_broken_pl(lambda_pivot[band], lam_s, eta_A1, eta_A2) for band in self.clean_bands])
        return params["log_amp_delta"]
    
    @eqx.filter_jit
    def log_prob(self, params: dict[str, JAXArray]) -> JAXArray:
        """Calculate the log probability of the input parameters.

        Args:
            params (dict[str, JAXArray]): Model parameters.

        Returns:
            JAXArray: Log probability of the input parameters.
        """
        gp, inds = self._build_gp(params)
        #jax.debug.print("amps {a}, amps_blr {b}, lag_blr {l}", a=self.kernel.amplitudes, b=self.kernel.amplitudes_blr, l=self.kernel.lag_blr)
        log_prob = gp.log_probability(y=self.y[inds])
        #jax.debug.print("Log probability: {log_prob}", log_prob=log_prob)
        return log_prob

    def sample(self, params: dict[str, JAXArray]) -> None:
        """A convience function for intergrating with numpyro for MCMC sampling.

        Args:
            params (dict[str, JAXArray]): Model parameters.
        """
        gp, inds = self._build_gp(params)
        log_prob = gp.log_probability(y=self.y[inds])
        jax.debug.print("Log probability: {log_prob}", log_prob=log_prob)
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

def compute_psd_from_samples(samples, clean_bands, num_points=1000, time_range=(0, 365*20)):
    """
    Compute the Power Spectral Density (PSD) using the kernel parameters from MCMC samples.

    Args:
        samples (dict): MCMC samples containing kernel parameters.
        clean_bands (list): List of clean bands used in the model.
        num_points (int): Number of points to sample in the time range.
        time_range (tuple): A tuple specifying the range of time lags (min_time, max_time).

    Returns:
        dict: A dictionary containing frequencies and PSD for each band.
    """
    # Extract kernel parameters from samples
    log_kernel_param = samples["log_kernel_param"]
    log_amp_delta_blr = samples["log_amp_delta_blr"]
    log_lag_blr = samples["log_lag_blr"]
    eta_A1 = samples["eta_A1"]
    eta_A2 = eta_A1 + samples["ep_A"]
    eta_tau1 = samples["eta_tau1"]
    eta_tau2 = eta_tau1 + samples["ep_tau"]

    # Compute the median values of the parameters
    kernel_param = jnp.exp(jnp.median(log_kernel_param, axis=0))
    amp_delta_blr = jnp.exp(jnp.median(log_amp_delta_blr, axis=0))
    lag_blr = jnp.exp(jnp.median(log_lag_blr, axis=0))
    beta = jnp.median(beta)
    delta = jnp.median(delta)

    # Compute amplitudes and taus for each band
    amplitudes = jnp.exp(log_kernel_param[1] + jnp.log(10) * jnp.array([log_broken_pl(lambda_pivot[band], params["lam_s"], eta_A1, eta_A2) for band in clean_bands]))
    taus = jnp.exp(log_kernel_param[0] + jnp.log(10) * jnp.array([log_broken_pl(lambda_pivot[band], params["lam_s"], eta_tau1, eta_tau2) for band in self.clean_bands]))

    # Instantiate the MyMultibandLowRank kernel
    kernel = MyMultibandLowRank(
        sigma=kernel_param[1],
        scale=kernel_param[0],
        amplitudes=amplitudes,
        amplitudes_blr=amp_delta_blr,
        lag_blr=lag_blr,
        taus=taus,
    )

    # Generate time lags
    min_time, max_time = time_range
    time_lags = jnp.linspace(min_time, max_time, num_points)

    # Compute PSD for each band
    psd_results = {}
    for band_idx, band in enumerate(clean_bands):
        # Compute the covariance for the given band
        covariances = jnp.array([kernel.evaluate((0, band_idx), (lag, band_idx)) for lag in time_lags])

        # Apply the Fourier Transform to compute the PSD
        fft_result = jnp.fft.fft(covariances)
        freqs = jnp.fft.fftfreq(num_points, d=(max_time - min_time) / num_points)

        # Compute the PSD (magnitude squared of the FFT)
        psd = jnp.abs(fft_result) ** 2

        # Store only the positive frequencies and corresponding PSD values
        positive_freqs = freqs[:num_points // 2]
        positive_psd = psd[:num_points // 2]

        psd_results[band] = {"freqs": np.array(positive_freqs), "psd": np.array(positive_psd)}

    return psd_results

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
    logJitterSampler = UniformInit(nBand, [jnp.log(1e-6), jnp.log(0.1)])

    # power laws
    etaA1Sampler = UniformInit(1, [0.5, 0.1])
    etaTau1Sampler = UniformInit(1, [-1.5, -2.0])
    epTauSampler = UniformInit(1, [-0.1, 0.1])
    epASampler = UniformInit(1, [-0.1, 0.1])
    lamsSampler = UniformInit(1, [2000.0, 2500.0])

    # kernel init
    kernelSampler = DRWInit([jnp.log(10**2.5), jnp.log(10**4.5)], [jnp.log(0.1), jnp.log(1.0)])
    
    return {
        "log_kernel_param": kernelSampler(subkeys[0], nSample),
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
        "eta_tau1": etaTau1Sampler(subkeys[10], nSample),
        "ep_A": epASampler(subkeys[11], nSample),
        "ep_tau": epTauSampler(subkeys[12], nSample),
        "lam_s": lamsSampler(subkeys[13], nSample),
    }

def numpyro_model(X, yerr, y=None, bestP=None, clean_bands=None):
    # kernel param
    #flat_normal = dist.Normal(bestP["log_kernel_param"], jnp.array([0.1, 0.1]))
    # This works better with the direct GP solver
    flat_normal = dist.Uniform(jnp.array([2.0, -3.0]), jnp.array([10.0, 0.5]))
    diag_normal = dist.Independent(flat_normal, 1)
    log_kernel_param = numpyro.sample("log_kernel_param", diag_normal)
    #jax.debug.print("{x} log_kernel_param_numpro", x=log_kernel_param)
    #jax.debug.print("{x} log_kernel_param_numpro_bestp", x=bestP["log_kernel_param"])

    log_amp_delta_blr = numpyro.sample(
      "log_amp_delta_blr", dist.Normal(jnp.full_like(bestP["log_amp_delta_blr"], -2.0), 2.0)
    )

    # lag
    lag = numpyro.sample("lag", dist.Normal(jnp.full_like(bestP["lag"], 0.0), 10))
    log_lag_blr = numpyro.sample("log_lag_blr", dist.Normal(jnp.full_like(bestP["log_lag_blr"], 5.0), 2.0))

    # log tau drw blr
    log_tau_drw_blr = numpyro.sample("log_tau_drw_blr", dist.Normal(2.8, 2.0))
    
    # log jitter, mean => the prior for these two should be set small, otherwise
    # it is hard to converge
    log_jitter = numpyro.sample("log_jitter", dist.Normal(np.full_like(bestP["log_jitter"], -4.0), 2.0))
    
    mean = numpyro.sample("mean", dist.Normal(jnp.full_like(bestP["mean"], 0.0), 0.1))
    poly1 = numpyro.sample("poly1", dist.Normal(0.0, 10.0))

    # power laws
    ep_A = numpyro.sample("beta", dist.Normal(0.2, 0.1))
    ep_tau = numpyro.sample("delta", dist.Normal(-0.5, 0.1))

    eta_A1 = numpyro.sample("eta_A1", dist.Normal(0.0, 0.1))
    eta_tau1 = numpyro.sample("eta_tau1", dist.Normal(0.0, 0.1))
    lams = numpyro.sample("lam_s", dist.Normal(2300.0, 100.0))

    # kernel
    k = kernels.quasisep.Exp(*jnp.exp(log_kernel_param))
    m1 = MyMultiVarModel(X, y, yerr, k, zero_mean=zero_mean, has_jitter=has_jitter, has_lag=has_lag, clean_bands=clean_bands, z=data['z'])

    sample_params = {
        "log_kernel_param": log_kernel_param,
        "log_amp_delta_blr": log_amp_delta_blr, 
        "lag": lag,
        "log_lag_blr": log_lag_blr,
        "log_tau_drw_blr": log_tau_drw_blr,
        "mean": mean,
        "poly1": poly1, 
        "log_jitter": log_jitter,
        # power laws
        "ep_A": ep_A,
        "ep_tau": ep_tau,
        "eta_A1": eta_A1,
        "eta_tau1": eta_tau1,
        "lam_s": lams,
    }
    m1.sample(sample_params)


def fit_multiband(data, progress_bar=False, plot=False, svi=False):
    times = data['times']
    mags = data['mags']
    data['mags_means'] = np.array([np.nanmean(mags[band]) for band in mags.keys()])
    for band in mags.keys():
       mags[band] = mags[band] - np.nanmean(mags[band])  # Center the magnitudes
    magerrs = data['magerrs']
    
    red_bands = bands_redder_than_5000(data['z'])
    blue_bands = bands_bluer_than_lyman_alpha(data['z'])

    clean_bands = list(set(bands) - set(blue_bands))
    # Reorder clean_bands to match the desired order
    clean_bands = list(sorted(clean_bands, key=lambda band: ['u', 'g', 'r', 'i', 'z', 'y'].index(band)))
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

    # define kernel
    initial_drw_params = {"log_kernel_param": jnp.log(np.array([100.0, 0.35]))}
    k = kernels.quasisep.Exp(*jnp.exp(initial_drw_params["log_kernel_param"]))

    # define model
    m1 = MyMultiVarModel(
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
                partial(numpyro_model, bestP=bestP, clean_bands=clean_bands),
                moves={AIES.DEMove() : 0.5, AIES.StretchMove() : 0.5},
                init_strategy=init_strategy,
                )

            num_params = sum(p.size for p in bestP.values())
            print(f"Number of parameters: {num_params}")

            mcmc = MCMC(
                nuts_kernel,
                num_warmup=250, # This could be less than num_samples
                num_samples=100,
                num_chains=2*33,
                progress_bar=progress_bar,
                chain_method="vectorized",
            )

            mcmc.run(jax.random.PRNGKey(int(data['object_id'])), X, yerr, y=y)
            samples = mcmc.get_samples(group_by_chain=False)
            diagnostics = mcmc.get_extra_fields()
        except Exception as e:
            print(f"Error during MCMC for quasar {data['object_id']}: {e}", flush=True)
            return None

        #print(samples)

        #if np.all(diagnostics['diverging']):
        #    print(f"Diverging MCMC for quasar {data['object_id']}, skipping.", flush=True)
        #    #return None

    # power laws
    eta_A1 = samples["eta_A1"]
    eta_A2 = eta_A1 + samples["ep_A"]
    eta_tau1 = samples["eta_tau1"]
    eta_tau2 = eta_tau1 + samples["ep_tau"]
    
    lambda_ref = 2500 # Any reference wavelength
    lambda_s_RF = samples["lam_s"]/(1 + data['z'])
    
    samples_log_sigma_UV = np.log10(np.exp(samples['log_kernel_param'][:, 1] + np.log(10) * log_broken_pl(lambda_ref, lambda_s_RF, eta_A1, eta_A2)))
    samples_log_tau_UV = np.log10(np.exp(samples['log_kernel_param'][:, 0] + np.log(10) * log_broken_pl(lambda_ref, lambda_s_RF, eta_tau1, eta_tau2)))
    samples_log_tau_UV_RF = log_tau_UV - np.log10(1 + data['z']) # time dilation correction

    def sym_perecentile(x, p=[16, 50, 84], axis=0):
        lower, median, upper = np.percentile(x, p, axis=axis)
        return median, 0.5 * (upper - lower)

    # parameter estimates
    log_jitter, log_jitter_err = sym_perecentile(np.log10(np.exp(2*samples['log_jitter'])))
    poly1, poly1_err = sym_perecentile(samples['poly1'])
    mean, mean_err = sym_perecentile(samples['mean'])
    log_amp_delta_blr, log_amp_delta_blr_err = sym_perecentile(np.log10(np.exp(samples['log_amp_delta_blr'])))
    log_lag_blr, log_lag_blr_err = sym_perecentile(np.log10(np.exp(samples['log_lag_blr'])))
    lag, lag_err = sym_perecentile(samples['lag'])

    log_tau, log_tau_err = sym_perecentile(np.log10(np.exp(samples['log_kernel_param'][:, 0])))
    log_sigma, log_sigma_err = sym_perecentile(np.log10(np.exp(samples['log_kernel_param'][:, 1])))

    log_tau_UV_RF, log_tau_UV_RF_err = sym_perecentile(samples_log_tau_UV_RF)
    log_sigma_UV, log_sigma_UV_err = sym_perecentile(log_sigma_UV)

    # BLR
    log_tau_blr, log_tau_blr_err = sym_perecentile(np.log10(np.exp(samples['log_tau_drw_blr'])))
    log_sigma_blr, log_sigma_blr_err = sym_perecentile(np.log10(np.exp(samples['log_kernel_param'][:, 1])*np.exp(samples['log_amp_delta_blr'])))

    lambda_s_RF, lambda_s_RF_err = sym_perecentile(lambda_s_RF)

    # Construct the result dictionary
    d = dict(object_id=data['object_id'],
            z=data['z'],
            # kernel params latent
            log_tau_UV_RF=log_tau_UV_RF,
            log_tau_UV_RF_err=log_tau_UV_RF_err,
            log_sigma_UV=log_sigma_UV,
            log_sigma_UV_err=log_sigma_UV_err,
            eta_A1=eta_A1,
            eta_A2=eta_A2,
            eta_tau1=eta_tau1,
            eta_tau2=eta_tau2,
            lam_s=lambda_s_RF,
            lam_s_err=lambda_s_RF_err,
            # kernel params band
            log_sigma_band=log_sigma_band,
            log_sigma_band_err=log_sigma_band_err,
            # kernel params
            log_sigma=log_sigma,
            log_sigma_err=log_sigma_err,
            log_tau=log_tau,
            log_tau_err=log_tau_err,
            #BLR
            log_sigma_blr=log_sigma_blr,
            log_sigma_blr_err=log_sigma_blr_err,
            log_tau_blr=log_tau_blr,
            log_tau_blr_err=log_tau_blr_err,
            # other
            log_jitter=log_jitter,
            poly1=poly1,
            poly1_err=poly1_err,
            mean=mean,
            mean_err=mean_err,
            clean_bands=clean_bands,
            log_lag_blr=log_lag_blr,
            log_lag_blr_err=log_lag_blr_err,
            lag=lag,
            lag_err=lag_err,
            )
    
    if plot:
        save_combined_plot(samples, m1, X, y, yerr, band_idx[mask_outlier], d)
        #plot_mcmc_traces(samples, d)
        #plot_posterior(samples, data, clean_bands=clean_bands)
    # psd_results = compute_psd_from_samples(samples, clean_bands)
    # d['psd'] = psd_results
    # plot_psd(psd_results, data['object_id'])    
    return d


def process_quasar(i_data, n=0, progress_bar=False, plot=False, svi=False):
    i, data = i_data
    #print(f"Processing quasar {i}/{n} ({data['object_id']})", flush=True)

    # Load the quasar data
    result = fit_multiband(data, progress_bar=progress_bar, plot=plot, svi=svi)
    if result is None:
        print(f"Skipping quasar {data['object_id']}.")
        return None
    data['i'] = i
    data |= result


    #print(f"Quasar {i}/{n} ({data['object_id']}): log_tau_RF={data['log_tau_RF']:.3f}±{data['log_tau_RF_err']:.3f}, log_sigma_RF={data['log_sigma_RF']}±{data['log_sigma_RF_err']}", flush=True)
    return data

def concat_light_curves(N=None, skip=None, filter_object_ids=None, save_file_path=None):
    print(f"concat_light_curves args: {N=}, {skip=}, {len(filter_object_ids)=}, {save_file_path=}")
    if save_file_path and os.path.exists(save_file_path):
        print(f"concat_light_curves Loading LC data from {save_file_path}")
        s82_objs = load_s82_from_hdf5(save_file_path)
        print(f"Loaded {len(s82_objs)} objs from {save_file_path}")
        if filter_object_ids is not None:
            # Filter the loaded objects based on the provided object IDs
            s82_objs = [obj for obj in s82_objs if obj['object_id'] in filter_object_ids]
            print(f"After filtering {len(filter_object_ids)}, loaded {len(s82_objs)}")
        if skip is not None:
            s82_objs = s82_objs[skip:]
            print(f"After skipping {skip}, loaded {len(s82_objs)} objs")
        if N is not None:
            s82_objs = s82_objs[:N]
        return s82_objs
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
            mags_means = [np.nanmean(mags[band]) for band in mags.keys()]
            #mags[band] = mags[band] - np.nanmean(mags[band])  # Center the magnitudes

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

        # Skip if no data is available for the object
        if all((len(times[band]) == 0 or len(mags[band]) == 0 or len(magerrs[band]) == 0) for band in bands):
            print(f"No data available for object {object_id}, skipping.", flush=True)
            continue


        s82_objs.append({
            'object_id': object_id,
            'times': times,
            'mags': mags,
            'mags_mean': mags_means,
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

def append_hdf5_file(quasar_list, file_path):
    # Append to HDF5 file if it exists, otherwise create a new one
    print(f"Appending {len(quasar_list)} quasars to {file_path}", flush=True)
    with h5py.File(file_path, "a") as hdf:
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

    args = parser.parse_args()


    # Filter objects by object_id that exist in the HDF5 file
    existing_object_ids = set()
    if args.ignore_existing:
        if os.path.exists(args.file):
            with h5py.File(args.file, "r") as hdf:
                existing_object_ids = set(hdf.keys())
                print(f"Found {len(existing_object_ids)} existing object IDs in {args.file}")
        else:
            existing_object_ids = set()

    filter_object_ids = set(args.filter_object_id) if args.filter_object_id else None
    filter_object_ids = set(pd.read_csv(args.filter_file, dtype={"object_id": str})["object_id"].values) if args.filter_file else filter_object_ids
    filter_object_ids = set(filter_object_ids) - set(existing_object_ids)
    if filter_object_ids is not None:
        print(f"Filtering object IDs: {len(filter_object_ids)}")
    objs = concat_light_curves(filter_object_ids=filter_object_ids, N=args.N, skip=args.skip, save_file_path=args.lc_file)
    if args.create_lc:
        sys.exit("Created LC file. Exiting the program as requested.")
    print(f"Loaded {len(objs)} objects from concat_light_curves")
    #objs = populate_sdss_fields(objs)
    for i, obj in enumerate(objs):
        print(f"Processing quasar {i}/{len(objs)} ({obj['object_id']})", flush=True)
        q = process_quasar((i, obj), n=len(objs), progress_bar=True, plot=args.plot, svi=args.svi)
        if q is None:
            #print(f"Skipping quasar {obj['object_id']}, no data", flush=True)
            continue
        fields_to_filter = ['times', 'mags', 'magerrs']
        filtered_q = {k: v for k, v in q.items() if k not in fields_to_filter}
        #print(filtered_q)
        print(f"Quasar {i}/{len(objs)} ({q['object_id']}): log_tau_RF={q['log_tau_RF']:.3f}±{q['log_tau_RF_err']:.3f}, log_sigma_RF={q['log_sigma_RF']}±{q['log_sigma_RF_err']}", flush=True)
        if args.file:
            append_hdf5_file([q], args.file)

    sys.exit("Exiting the program as requested.")



    # cat = pd.read_parquet(f"data/S82/Catalog.parquet").set_index('idx')
    # sdss = pd.read_parquet(f"data/S82/dr16s82_sdssLCRaw.parquet")
    # sdss = sdss[sdss.mjd.notna() & (len(sdss.mjd) > 0)]

    # # Find elements in cat where objectId exists in the list of objectId of sdss
    # #filter_object_ids = set(pd.read_csv("data/object_ids_test.csv")["object_id"].values)
    # match_object_ids = set(sdss.objectId) if filter_object_ids is None else filter_object_ids
    # matching_indices = cat[cat.objectId.isin(match_object_ids)].index
    # cat = cat[cat.objectId.isin(match_object_ids)]
    # #cat = cat.loc[matching_indices]
    # print(f"Total objects in catalog: {len(cat)}")
    # print(f"Number of objects already processed in HDF5 file: {len(existing_object_ids)}")
    # print(f"Number of objects that should be processed: {len(cat)-len(existing_object_ids)}")

    # cat = cat[~cat['objectId'].isin(existing_object_ids)]
    # filter_object_ids = set(cat['objectId'].values)
    # print(f"Number of objects to process: {len(filter_object_ids)}")
    # #objs = [obj for obj in objs if obj['object_id'] not in existing_object_ids]
    # objs = concat_light_curves(filter_object_ids=filter_object_ids, N=args.N, skip=args.skip, save_file_path=args.lc_file)
    # print(f"Loaded {len(objs)} objects from concat_light_curves")
    # #sys.exit("Exiting the program as requested.")
    # #objs = populate_sdss_fields(objs)

    # for start_idx in range(0, len(objs), args.chunk_size):
    #     chunk = objs[start_idx:start_idx + args.chunk_size]
    #     print(f"({100*start_idx/len(objs):.1f}%) Processing chunk {start_idx // args.chunk_size + 1}/{(len(objs) + args.chunk_size - 1) // args.chunk_size}...")
        
    #     ctx = get_context("spawn")  # Safer for JAX when using multiprocessing
    #     results = []

    #     with ctx.Pool(processes=15) as pool:
    #         results = pool.map(partial(process_quasar, n=len(chunk), plot=args.plot, svi=args.svi), enumerate(chunk))

    #     quasar_list = [q for q in results if q is not None]

    #     # Append to HDF5 file if it exists, otherwise create a new one
    #     with h5py.File(args.file, "a") as hdf:
    #         for quasar in quasar_list:
    #             object_id = quasar["object_id"]
    #             if object_id in hdf:
    #                 continue

    #             group = hdf.create_group(object_id)
    #             for key, value in quasar.items():
    #                 if isinstance(value, dict):
    #                     sub_group = group.create_group(key)
    #                     for sub_key, sub_value in value.items():
    #                         sub_group.create_dataset(sub_key, data=sub_value)
    #                 else:
    #                     group.attrs[key] = value
