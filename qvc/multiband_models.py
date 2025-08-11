import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
import os
import sys

import tinygp

from tinygp.helpers import JAXArray
from numpy.typing import NDArray

from functools import partial

import numpyro
import numpyro.distributions as dist

from tinygp import kernels
from tinygp import GaussianProcess

from eztaox.models import MultiVarModel

from multiband_fit_utils import *
from multiband_fit_plotting import *
from multiband_generate_lc import *

class MyMultibandContiBLR(tinygp.kernels.Kernel):
    amplitudes: jnp.ndarray
    amplitudes_blr: jnp.ndarray
    lag_blr: jnp.ndarray
    tau_drw: float
    w: jnp.ndarray
    s_b: jnp.ndarray

    def __init__(self, amplitudes, amplitudes_blr, lag_blr, taus, log_w, s_b) -> None:
        self.amplitudes = amplitudes
        self.amplitudes_blr = amplitudes_blr
        self.lag_blr = lag_blr
        self.tau_drw = taus
        self.w = jnp.exp(log_w)
        self.s_b = s_b

    def coord_to_sortable(self, X) -> JAXArray:
        return X[0]

    def k(self, t1, t2, tau_drw) -> JAXArray:
        tau = jnp.abs(t1 - t2)
        drw = jnp.exp(-tau / tau_drw)
        return drw

    def evaluate(self, X1, X2) -> JAXArray:
        t1, b1 = X1
        t2, b2 = X2

        amplitudes_b1 = self.amplitudes[b1]
        amplitudes_b2 = self.amplitudes[b2]
        amplitudes_blr_b1 = self.amplitudes_blr[b1]
        amplitudes_blr_b2 = self.amplitudes_blr[b2]

        k_ac = self.k(t1, t2, self.tau_drw)

        # a is cont at t1
        # b is blr at t1
        # c is cont at t2
        # d is blr at t2
        cov_ac = (
            amplitudes_b1
            * amplitudes_b2
            * k_ac
        )
        cov_ad = (
            amplitudes_b1
            * amplitudes_blr_b2
            * self.k(t1, t2 - self.lag_blr[b2], self.tau_drw)
        )
        cov_bc = (
            amplitudes_blr_b1
            * amplitudes_b2
            * self.k(t1 - self.lag_blr[b1], t2, self.tau_drw)
        )
        cov_bd = (
            amplitudes_blr_b1
            * amplitudes_blr_b2
            * self.k(t1 - self.lag_blr[b1], t2 - self.lag_blr[b2], self.tau_drw)
        )

        # BWB
        q1 = self.s_b[b1] * amplitudes_b1
        q2 = self.s_b[b2] * amplitudes_b2
        cov_ac = cov_ac + 2.0 * q1 * q2 * k_ac * k_ac

        return cov_ac + cov_ad + cov_bc + cov_bd


# Override MultiVarModel
class MyMultiVarModel(MultiVarModel):
    yerr: JAXArray | NDArray
    clean_bands: JAXArray
    z: float
    lam_rf: JAXArray

    def __init__(
        self,
        X: JAXArray,
        y: JAXArray | NDArray,
        yerr: JAXArray | NDArray,
        kernel: tinygp.kernels.quasisep.Quasisep,
        **kwargs,
    ) -> None:
        super().__init__(X, y, yerr, kernel, **kwargs)
        self.yerr = yerr
        self.clean_bands = kwargs.get("clean_bands", None)
        self.z = kwargs.get("z", None)
        self.lam_rf = jnp.array([lambda_pivot[band] for band in self.clean_bands]) / (1 + self.z)

    @staticmethod
    def mean_func(
        zero_mean: bool, nBand: int, t_center: float, t_std: float, params: dict[str, JAXArray], X: JAXArray
    ) -> JAXArray:

        band_idx = X[1]

        if zero_mean is True:
            mean_per_obs = jnp.zeros(nBand)[band_idx]
        else:
            time_centered = (X[0] - t_center)
            time_scaled = time_centered/t_std
            #coeffs = jnp.stack([params["poly1"], params["mean"][band_idx]])
            #mean_per_obs = jnp.polyval(coeffs, time_scaled)
            mean_per_obs = params["poly1"] * time_scaled + params["mean"][band_idx]

        return mean_per_obs
    
    def _build_gp(
        self, params: dict[str, JAXArray]
    ) -> tuple[GaussianProcess, JAXArray]:
        # log amp + mean
        log_sigma_band = self.my_amp_transform(params)
        log_sigma_band_blr = self.my_amp_transform_blr(params)
        log_tau_band = self.my_tau_drw_transform(params)

        # time axis transform: t and band are not sorted,
        # inds gives the sorted indices for the new_t
        X, inds = self.my_lag_transform(self.X, self.has_lag, params)

        t_center = jnp.mean(X[0])
        t_std = jnp.std(X[0])

        means = partial(
            MyMultiVarModel.mean_func, self.zero_mean, log_sigma_band.shape[0], 
            t_center, t_std, params
        )

        t = X[0]
        band = X[1]

        # add jitter to the diagonal
        if self.has_jitter is True:
            diags = self.diag[inds] + (jnp.exp(params["log_jitter"]) ** 2)[band[inds]]
        else:
            diags = self.diag[inds]

        # BWB
        s_b = params["bwb_alpha"] + params["bwb_beta"] * jnp.log(self.lam_rf / 2500.0)  # shape (n_band,)

        kernel = MyMultibandContiBLR(
            amplitudes=jnp.exp(log_sigma_band),
            taus=jnp.exp(log_tau_band),
            amplitudes_blr=jnp.exp(log_sigma_band_blr),
            lag_blr=jnp.exp(params["log_lag_blr"]),
            log_w=0,
            s_b=s_b
        )

        # Check if kernel covariance is symmetric
        #cov_matrix = kernel((t[inds], band[inds]), (t[inds], band[inds]))
        #is_symmetric = jnp.allclose(cov_matrix, cov_matrix.T, atol=1e-5)
        #jax.debug.print("Kernel covariance symmetric: {sym}", sym=is_symmetric)

        return (
            GaussianProcess(
                kernel,
                (t[inds], band[inds]),
                diag=diags + 1e-6,
                mean=means), 
        inds,)

    def my_lag_transform(
        self, X: JAXArray, has_lag: bool, params: dict[str, JAXArray]
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray]:
        if has_lag is True:
            lags = params["lag0"] * (self.lam_rf / 2500.0) ** params["lag_beta"]
            lags = jnp.insert(lags, 0, 0.0)
        else:
            nBand = params["log_amp_delta"].size + 1
            lags = jnp.zeros(nBand)
        t, band = X
        new_t = t - lags[band]
        inds = jnp.argsort(new_t)
        return (new_t, band), inds

    def my_amp_transform_blr(self, params: dict[str, JAXArray]) -> JAXArray:
        return params["log_sigma0"] + jnp.atleast_1d(params["log_amp_delta_blr"])
    
    def my_tau_drw_transform(self, params: dict[str, JAXArray]) -> JAXArray:
        eta_tau1 = params["eta_tau1"]
        eta_tau2 = params["eta_tau2"]
        lam_s = params["lam_s"]
        eta_break = params["eta_break"]
        log_tau_band = params["log_tau_drw0"] + jnp.log(10) * log_broken_pl(self.lam_rf, lam_s, eta_tau1, eta_tau2, eta_break)
        return jnp.mean(log_tau_band)

    def my_amp_transform(self, params: dict[str, JAXArray]) -> JAXArray:
        eta_A1 = params["eta_A1"]
        eta_A2 = params["eta_A2"]
        lam_s = params["lam_s"]
        eta_break = params["eta_break"]

        # Host dilution: apply per-band correction
        # Host galaxy contribution modeled as a power-law in wavelength
        alpha_AGN = -1.5 # alpha_lam
        host_frac = params["f_host"] * (self.lam_rf / 5100.0) ** (params["alpha_host"] - alpha_AGN)
        dilution_factor = 1.0 / (1.0 + host_frac)
        log_dilution = jnp.log(dilution_factor)

        # Power-law scaling across rest-frame wavelength
        log_sigma_band = params["log_sigma0"] + log_dilution + jnp.log(10) * log_broken_pl(self.lam_rf, lam_s, eta_A1, eta_A2, eta_break)

        return log_sigma_band
    
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
        new_X, inds = self.my_lag_transform(X, self.has_lag, params)

        # build gp, cond
        gp, inds = self._build_gp(params)
        _, cond = gp.condition(self.y[inds], new_X)

        return cond.loc, jnp.sqrt(cond.variance)
