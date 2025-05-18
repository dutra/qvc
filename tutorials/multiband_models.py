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

from solvers import DirectFullRank


# define params
zero_mean = False
has_jitter = True
has_lag = True

class MyMultibandContiBLR(tinygp.kernels.Kernel):
    amplitudes: jnp.ndarray
    amplitudes_blr: jnp.ndarray
    lag_blr: jnp.ndarray
    tau_drw: jnp.ndarray
    tau_drw_blr: float
    w: jnp.ndarray

    def __init__(self, amplitudes, amplitudes_blr, lag_blr, taus, tau_drw_blr, log_w) -> None:
        self.amplitudes = amplitudes
        self.amplitudes_blr = amplitudes_blr
        self.lag_blr = jnp.zeros_like(lag_blr)
        self.tau_drw = taus
        self.tau_drw_blr = tau_drw_blr
        self.w = jnp.exp(log_w)

    def coord_to_sortable(self, X) -> JAXArray:
        return X[0]

    def k(self, tau, tau_drw) -> JAXArray:
        tau = jnp.abs(tau)
        drw = jnp.exp(-tau / tau_drw)
        return drw

    def ktophat(self, tau, tau_d) -> JAXArray:
        delta_t = jnp.abs(tau)
        width = self.w

        t_L = delta_t - width
        t_H = delta_t         # delay = 0, so t_H = delta_t - 0

        A = 1.0 / width  # normalized top-hat amplitude
        prefactor = A

        def case1(t_L, t_H):
            return jnp.exp(-t_L / tau_d) - jnp.exp(-t_H / tau_d)

        def case2(t_L, t_H):
            return jnp.exp(t_H / tau_d) - jnp.exp(t_L / tau_d)

        def case3(t_L, t_H):
            return 2.0 - jnp.exp(t_L / tau_d) - jnp.exp(-t_H / tau_d)

        return prefactor * jnp.select(
            [t_L > 0, t_H < 0, (t_L <= 0) & (t_H >= 0)],
            [case1(t_L, t_H), case2(t_L, t_H), case3(t_L, t_H)],
            default=0.0
        )

    def evaluate(self, X1, X2) -> JAXArray:
        t1, b1 = X1
        t2, b2 = X2

        amplitudes_b1 = jnp.sqrt( (self.amplitudes[b1])**2 * self.tau_drw[b1] )
        amplitudes_b2 = jnp.sqrt( (self.amplitudes[b2])**2 * self.tau_drw[b2] )
        amplitudes_blr_b1 = jnp.sqrt( (self.amplitudes_blr[b1])**2 * self.tau_drw_blr )
        amplitudes_blr_b2 = jnp.sqrt( (self.amplitudes_blr[b2])**2 * self.tau_drw_blr )

        # a is cont at t1
        # b is blr at t1
        # c is cont at t2
        # d is blr at t2
        cov_ac = (
            amplitudes_b1
            * amplitudes_b2
            * jnp.sqrt(
                self.k(t2 - t1, self.tau_drw[b1])
                * self.k(t2 - t1, self.tau_drw[b2])
            )
        )
        cov_ad = (
            amplitudes_b1
            * amplitudes_blr_b2
            * jnp.sqrt(
                self.k(t2 - t1, self.tau_drw[b1])
                * self.k(t2 - t1 - self.lag_blr[b2], self.tau_drw_blr)
            )
        )
        cov_bc = (
            amplitudes_blr_b1
            * amplitudes_b2
            * jnp.sqrt(
                self.k(t2 - t1 - self.lag_blr[b1], self.tau_drw_blr)
                * self.k(t2 - t1, self.tau_drw[b2])
            )
        )
        cov_bd = (
            amplitudes_blr_b1
            * amplitudes_blr_b2
            * jnp.sqrt(
                self.k(t2 - t1 - self.lag_blr[b1], self.tau_drw_blr)
                * self.k(t2 - t1 - self.lag_blr[b2], self.tau_drw_blr)
            )
        )

        return cov_ac + cov_ad + cov_bc + cov_bd

class MyMultibandConti(tinygp.kernels.Kernel):
    sigma_hat: float
    tau_drw: jnp.ndarray
    w: jnp.ndarray

    def __init__(self, sigma_hat, scale, tau_drw, log_w) -> None:
        self.tau_drw = tau_drw
        self.sigma_hat = sigma_hat
        self.w = jnp.exp(log_w)

    def coord_to_sortable(self, X) -> JAXArray:
        return X[0]

    def k(self, tau, tau_drw) -> JAXArray:
        tau = jnp.abs(tau)
        drw = jnp.exp(-tau / tau_drw)
        return drw

    def kktophat(self, tau, tau_d) -> JAXArray:
        delta_t = jnp.abs(tau)
        width = self.w

        t_L = delta_t - width
        t_H = delta_t         # delay = 0, so t_H = delta_t - 0

        A = 1.0 / width  # normalized top-hat amplitude
        prefactor = A

        def case1(t_L, t_H):
            return jnp.exp(-t_L / tau_d) - jnp.exp(-t_H / tau_d)

        def case2(t_L, t_H):
            return jnp.exp(t_H / tau_d) - jnp.exp(t_L / tau_d)

        def case3(t_L, t_H):
            return 2.0 - jnp.exp(t_L / tau_d) - jnp.exp(-t_H / tau_d)

        return prefactor * jnp.select(
            [t_L > 0, t_H < 0, (t_L <= 0) & (t_H >= 0)],
            [case1(t_L, t_H), case2(t_L, t_H), case3(t_L, t_H)],
            default=0.0
        )

    def evaluate(self, X1, X2) -> JAXArray:
        t1, b1 = X1
        t2, b2 = X2

        sigma_b1 = jnp.sqrt(self.sigma_hat**2 * self.tau_drw[b1])
        sigma_b2 = jnp.sqrt(self.sigma_hat**2 * self.tau_drw[b2])

        # a is cont at t1
        # b is blr at t1
        # c is cont at t2
        # d is blr at t2
        cov_ac = (
            sigma_b1
            * sigma_b2
            * jnp.sqrt(
                self.k(t2 - t1, self.tau_drw[b1])
                * self.k(t2 - t1, self.tau_drw[b2])
            )
        )

        return cov_ac

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
        self.z = kwargs.get("z", None)

    @staticmethod
    def mean_func(
        zero_mean: bool, nBand: int, params: dict[str, JAXArray], X: JAXArray
    ) -> JAXArray:
        #zero_mean = True
        if zero_mean is True:
            means = jnp.zeros(nBand)[X[1]]
        else:
            time_centered = (X[0] - jnp.nanmean(X[0]))
            time_scaled = time_centered #/ (jnp.nanmax(X[0]) - jnp.nanmin(X[0]))
            means = jnp.atleast_1d(params["mean"])[X[1]] + params["poly1"] * time_scaled / 100000
        return means
    
    def _build_gp(
        self, params: dict[str, JAXArray]
    ) -> tuple[GaussianProcess, JAXArray]:
        # log amp + mean
        log_sigma_hat_band = self.my_amp_transform(params)
        log_sigma_hat_band_blr = self.my_amp_transform_blr(params)
        log_tau_band_rf = self.my_tau_drw_transform(params)

        means = partial(
            MyMultiVarModel.mean_func, self.zero_mean, log_sigma_hat_band.shape[0], params
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

        kernel = MyMultibandContiBLR(
            amplitudes=jnp.exp(log_sigma_hat_band),
            taus=jnp.exp(log_tau_band_rf),
            amplitudes_blr=jnp.exp(log_sigma_hat_band_blr),
            tau_drw_blr=jnp.exp(params["log_tau_drw_blr"] - jnp.log(1 + self.z)),
            lag_blr=jnp.exp(params["log_lag_blr"] - jnp.log(1 + self.z))[band[inds]],
            log_w=params["log_w"] - jnp.log(1 + self.z),
        )
        return (
            GaussianProcess(
                kernel,
                (t[inds], band[inds]),
                diag=diags + 1e-6,
                mean=means), 
        inds,)

    def my_amp_transform_blr(self, params: dict[str, JAXArray]) -> JAXArray:
        return params["log_sigma_hat0"] + jnp.atleast_1d(params["log_amp_delta_blr"])
    
    def my_tau_drw_transform(self, params: dict[str, JAXArray]) -> JAXArray:
        eta_tau1 = params["eta_tau1"]
        eta_tau2 = params["eta_tau2"]
        lam_s = params["lam_s"]
        eta_break = params["eta_break"]
        params["log_tau_band_RF"] = params["log_tau_drw0"] - jnp.log(1 + self.z) + jnp.log(10) * jnp.array([log_broken_pl(lambda_pivot[band]/(1 + self.z), lam_s, eta_tau1, eta_tau2, eta_break) for band in self.clean_bands])
        return params["log_tau_band_RF"]

    def my_amp_transform(self, params: dict[str, JAXArray]) -> JAXArray:
        eta_A1 = params["eta_A1"]
        eta_A2 = params["eta_A2"]
        lam_s = params["lam_s"]
        eta_break = params["eta_break"]
        params["log_sigma_hat_band"] = params["log_sigma_hat0"] + jnp.log(10) * jnp.array([log_broken_pl(lambda_pivot[band]/(1 + self.z), lam_s, eta_A1, eta_A2, eta_break) for band in self.clean_bands])
        return params["log_sigma_hat_band"]
    
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
        #jax.debug.print("Log probability: {log_prob}", log_prob=log_prob)
        return log_prob

    def sample(self, params: dict[str, JAXArray]) -> None:
        """A convience function for intergrating with numpyro for MCMC sampling.

        Args:
            params (dict[str, JAXArray]): Model parameters.
        """
        gp, inds = self._build_gp(params)
        log_prob = gp.log_probability(y=self.y[inds])
        #K = gp.kernel(gp.X, gp.X) + gp.noise
        #jax.debug.print("sym: {s}", s= jnp.allclose(K, K.T, atol=1e-6))
        #jax.debug.print("Log probability: {log_prob}", log_prob=log_prob)
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


class MyMultiVarModelLatent(MyMultiVarModel):

    def _build_latent_model(self, params: dict[str, JAXArray]) -> tuple[GaussianProcess, JAXArray]:
        log_amps = self.my_amp_transform(params)
        log_amps_blr = self.my_amp_transform_blr(params)
        log_taus = self.my_tau_drw_transform(params)

        X, inds = self.lag_transform(self.X, self.has_lag, params)
        t_obs, band_obs = X
        t_obs = t_obs[inds]
        band_obs = band_obs[inds]
        y_obs = self.y[inds]

        amp_conti = jnp.exp(log_amps)
        amp_blr = jnp.exp(log_amps_blr)
        lag_blr = jnp.exp(params["log_lag_blr"] - jnp.log(1 + self.z))[band_obs]

        # Clip amplitudes to lower limit of 1e-2
        amp_conti = jnp.clip(amp_conti, 1e-4, None)
        amp_blr = jnp.clip(amp_blr, 1e-4, None)

        # Noise (diagonal)
        noise_diag = self.diag[inds]
        if self.has_jitter:
            noise_diag += (jnp.exp(params["log_jitter"]) ** 2)[band_obs]

        # Get unique latent points
        t_latent, band_latent, inv_d, inv_l = get_unique_times(t_obs, band_obs, lag_blr)
        M = len(t_latent)

        # Construct latent-space kernel
        kernel = MyMultibandConti(
            sigma_hat=jnp.exp(params["log_sigma_hat_0"]),
            scale=jnp.exp(params["log_tau_drw_0"] - jnp.log(1+self.z)),
            tau_drw=jnp.exp(log_taus)[band_latent],
            log_w=params["log_w"] - jnp.log(1 + self.z),
        )
        X_latent = (t_latent, band_latent)
        K_latent = kernel(X_latent, X_latent) + 1e-6 * jnp.eye(M)

        # Construct observation operator H
        H = build_H(t_obs, band_obs, inv_d, inv_l, amp_conti, amp_blr, M)

        print("amp_conti.shape[0]", amp_conti.shape[0])

        # Mean function
        mu_obs = self.mean_func(self.zero_mean, amp_conti.shape[0], params, (t_obs, band_obs))
        #mu_obs = mean_fn((t_obs, band_obs))

        return {
            "H": H,
            "K_latent": K_latent,
            "D": noise_diag,
            "mu_obs": mu_obs,
            "y_obs": y_obs
        }

    @eqx.filter_jit
    def log_prob(self, params: dict[str, JAXArray]) -> JAXArray:
        latent = self._build_latent_model(params)
        H, K_latent, D, mu_obs, y_obs = (
            latent["H"], latent["K_latent"], latent["D"], latent["mu_obs"], latent["y_obs"]
        )

        # Build full covariance in observation space: K_obs = H K_latent H.T + D
        K_obs = H @ K_latent @ H.T
        K_obs += jnp.diag(D)  # add diagonal noise

        #jax.debug.print("H: {h}", h=H)

        # Subtract mean
        y_centered = y_obs - mu_obs

        # Cholesky solve for log-likelihood
        L = jnp.linalg.cholesky(K_obs + 1e-4 * jnp.eye(K_obs.shape[0]))  # small jitter for stability
        alpha = jax.scipy.linalg.cho_solve((L, True), y_centered)

        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        log_likelihood = -0.5 * (jnp.dot(y_centered, alpha) + logdet + y_obs.size * jnp.log(2 * jnp.pi))


        log_likelihood = jnp.where(jnp.isfinite(log_likelihood), log_likelihood, -jnp.inf)

        return log_likelihood

    def sample(self, params: dict[str, JAXArray]) -> None:
        """
        NumPyro-compatible likelihood evaluation using Woodbury identity for efficiency.
        Wraps the custom log-likelihood into a custom Distribution for NumPyro sampling.
        """

        #jax.debug.print("log_likelihood: {x}", x=self.log_prob(params))

        # Use numpyro.factor or wrap in a custom Distribution
        numpyro.factor("gp_loglike", self.log_prob(params))

    @eqx.filter_jit
    def pred(self, params: dict[str, JAXArray], X_new: JAXArray) -> tuple[JAXArray, JAXArray]:
        latent = self._build_latent_model(params)
        H, K_latent, D, mu_obs, y_obs = (
            latent["H"], latent["K_latent"], latent["D"], latent["mu_obs"], latent["y_obs"]
        )

        # Build full observed-space covariance matrix
        K_obs = H @ K_latent @ H.T + jnp.diag(D)
        y_centered = y_obs - mu_obs

        # Cholesky solve for alpha
        L = jnp.linalg.cholesky(K_obs + 1e-4 * jnp.eye(K_obs.shape[0]))
        alpha = jax.scipy.linalg.cho_solve((L, True), y_centered)

        # Prepare new latent inputs
        X_new_lagged, _ = self.lag_transform(X_new, self.has_lag, params)
        t_new, band_new = X_new_lagged
        lag_blr = jnp.exp(params["log_lag_blr"])[band_new]

        log_amps = self.my_amp_transform(params)
        log_amps_blr = self.my_amp_transform_blr(params)
        amp_conti = jnp.exp(log_amps)
        amp_blr = jnp.exp(log_amps_blr)

        t_latent, band_latent, _, _ = get_unique_times(
            self.X[0], self.X[1], jnp.exp(params["log_lag_blr"])[self.X[1]]
        )
        X_latent = (t_latent, band_latent)

        kernel = MyMultibandConti(
            sigma_hat=jnp.exp(params["log_sigma_hat_0"]),
            scale=jnp.exp(params["log_tau_drw_0"] - jnp.log(1+self.z)),
            tau_drw=jnp.exp(log_taus)[band_latent], # I think?
            log_w=params["log_w"] - jnp.log(1 + self.z),
        )

        t_new_latent, band_new_latent, inv_d_new, inv_l_new = get_unique_times(t_new, band_new, lag_blr)
        X_new_latent = (t_new_latent, band_new_latent)
        H_new = build_H(t_new, band_new, inv_d_new, inv_l_new, amp_conti, amp_blr, len(t_new_latent))

        K_lat_new_obs = H_new @ kernel(X_new_latent, X_latent) @ H.T
        pred_mean = self.mean_func(self.zero_mean, amp_conti.shape[0], params, (t_new, band_new)) + K_lat_new_obs @ alpha

        # Predictive variance (optional)
        v = jax.scipy.linalg.cho_solve((L, True), K_lat_new_obs.T)
        K_new = H_new @ kernel(X_new_latent, X_new_latent) @ H_new.T + 1e-6 * jnp.eye(H_new.shape[0])
        pred_cov = K_new - K_lat_new_obs @ v

        return pred_mean, jnp.sqrt(jnp.diag(pred_cov))