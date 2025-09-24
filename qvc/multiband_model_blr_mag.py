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

import jax
import jax.numpy as jnp
from typing import Sequence
import tinygp
from tinygp.kernels import quasisep as qs

class ContiBLRQS_Mag(qs.Wrapper):
    # keep your familiar names
    tau_drw: float
    width_cont: jnp.ndarray
    width_blr: jnp.ndarray
    amp_cont: jnp.ndarray     # <- now equals A_b (mag gain)
    amp_blr: jnp.ndarray      # <- now equals B_b (mag gain)
    lag_blr: jnp.ndarray
    lag_disk: jnp.ndarray
    bwb_alpha: jnp.ndarray
    bwb_beta: jnp.ndarray
    s: jnp.ndarray
    kernel: qs.Kernel
    kernel2: qs.Kernel

    def __init__(self, amp_cont, amp_blr, lag_blr, lag_disk, tau_drw,
                 bwb_alpha, bwb_beta, width_cont, width_blr, s):
        self.amp_cont = amp_cont   # interpret as A_b (mag)
        self.amp_blr  = amp_blr    # interpret as B_b (mag)
        self.lag_blr  = lag_blr
        self.lag_disk = lag_disk
        self.tau_drw  = tau_drw
        self.bwb_alpha = bwb_alpha
        self.bwb_beta  = bwb_beta
        self.width_cont = width_cont
        self.width_blr  = width_blr
        self.s = s

        # OU in magnitudes for c(t); "squared" OU is exact OU with tau/2
        self.kernel  = qs.Exp(scale=self.tau_drw,     sigma=1.0)
        self.kernel2 = qs.Exp(scale=self.tau_drw / 2, sigma=1.0) # TODO: beta

    def coord_to_sortable(self, X):
        t, b = X
        return t + jnp.asarray(b, t.dtype) * (10.0 * jnp.finfo(t.dtype).eps)

    def _gain_top_hat(self, width, tau):
        x = width / jnp.maximum(tau, 1e-12)
        return jnp.where(x < 1e-8, 1.0 - 0.5*x + x**2/6.0,
                         -jnp.expm1(-x) / jnp.maximum(x, 1e-12))

    def _obs0(self):
        return self.kernel.observation_model(jnp.array(0.0))

    def _conv_obs(self, lag, width, s):
        h0  = self._obs0()
        Phi = self.kernel.transition_matrix(0.0, s * lag)
        G   = self._gain_top_hat(width, self.tau_drw)
        return (h0 @ Phi) * G

    # block matrices (same pattern as before)
    def design_matrix(self):
        A0, A1 = self.kernel.design_matrix(), self.kernel2.design_matrix()
        z0 = jnp.zeros((A0.shape[0], A1.shape[1]))
        z1 = jnp.zeros((A1.shape[0], A0.shape[1]))
        return jnp.block([[A0, z0],[z1, A1]])

    def stationary_covariance(self):
        P0, P1 = self.kernel.stationary_covariance(), self.kernel2.stationary_covariance()
        z01 = jnp.zeros((P0.shape[0], P1.shape[1]))
        z10 = jnp.zeros((P1.shape[0], P0.shape[1]))
        return jnp.block([[P0, z01],[z10, P1]])

    def transition_matrix(self, X1, X2):
        t1, _ = X1
        t2, b2 = X2
        dt = self.s[b2] * (t2 - t1)
        Phi0 = self.kernel.transition_matrix(0.0, dt)
        Phi1 = self.kernel2.transition_matrix(0.0, dt)
        z01 = jnp.zeros((Phi0.shape[0], Phi1.shape[1]))
        z10 = jnp.zeros((Phi1.shape[0], Phi0.shape[1]))
        return jnp.block([[Phi0, z01],[z10, Phi1]])

    def observation_model(self, X):
        _, b = X
        b = jnp.asarray(b, dtype=int)

        # both arms apply to the SAME latent magnitude process c(t)
        h_cont = self.amp_cont[b] * self._conv_obs(self.lag_disk[b],
                                                   self.width_cont[b], self.s[b])
        h_blr  = self.amp_blr[b]  * self._conv_obs(self.lag_disk[b] + self.lag_blr[b],
                                                   self.width_blr[b],  self.s[b])

        h_base = h_cont + h_blr

        # BWB phenomenology (squared OU -> OU with tau/2)
        h_sq  = self.kernel2.observation_model(jnp.array(0.0))
        q_b   = self.bwb_alpha * (self.amp_cont[b] ** 2) # TODO: self.bwb_alpha[b]?
        h_bwb = jnp.sqrt(2.0) * q_b * h_sq

        return jnp.concatenate([h_base, h_bwb], axis=0)

    def psd(self, omega: JAXArray, b: int, sigma_n2: float = 0.0) -> JAXArray:
        """
        Auto-PSD S_y^{(b)}(ω) via general state-space formula.

        Parameters
        ----------
        omega : array
            Angular frequencies [rad / time units].
        b : int
            Band index.
        sigma_n2 : float, optional
            White-noise level to add (default 0.0).

        Returns
        -------
        PSD : array, shape = omega.shape
            Real non-negative power spectral density.
        """
        # System matrices
        A = self.design_matrix()
        P = self.stationary_covariance()
        Qc = -(A @ P + P @ A.T)

        # Observation vector for band b (evaluate at t=0)
        h = self.observation_model((jnp.array(0.0), jnp.array(int(b))))

        I = jnp.eye(A.shape[0], dtype=A.dtype)

        def one_w(w):
            # v = ((-iωI - A^T)^(-1)) h
            v = jnp.linalg.solve((-1j * w) * I - A.T, h)
            return (v.conj().T @ (Qc @ v)).real + sigma_n2

        return 2.0 * jax.vmap(one_w)(omega)

# Override MultiVarModel
class MyMultiVarModel_SMAG(MultiVarModel):
    yerr: JAXArray | NDArray
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
        self.z = kwargs.get("z", None)
        self.lam_rf = kwargs.get("lam_rf", None)

    @staticmethod
    def mean_func(
        zero_mean: bool, nBand: int, t_center: float, t_std: float, params: dict[str, JAXArray], X: JAXArray
    ) -> JAXArray:

        band_idx = X[1]

        if zero_mean is True:
            mean_per_obs = jnp.zeros(nBand)[band_idx]
        else:
            time_centered = (X[0] - t_center)
            t_std = jnp.maximum(t_std, 1e-6)
            time_scaled = time_centered/t_std
            #coeffs = jnp.stack([params["poly1"], params["mean"][band_idx]])
            #mean_per_obs = jnp.polyval(coeffs, time_scaled)
            mean_per_obs = params["poly1"] * time_scaled + params["mean"][band_idx]

        return mean_per_obs
    
    def _build_gp(self, params):
        log_sigma_band      = self.my_amp_transform(params)
        log_sigma_band_blr  = self.my_amp_transform_blr(params)
        log_tau_band, s     = self.my_tau_drw_transform(params)

        # DO NOT sort or reindex
        t, band = self.X
        
        #s = ContiBLRQS.coord_to_sortable(self, (t, band))  # or kernel.coord_to_sortable((t, band))
        #jax.debug.print("sorted_by_kernel? {}", jnp.all(jnp.diff(s) >= 0))

        t_center = jnp.mean(t)
        t_std    = jnp.std(t)

        means = partial(MyMultiVarModel_SMAG.mean_func, self.zero_mean, log_sigma_band.shape[0],
                        t_center, t_std, params)

        # diagonal noise in original order
        if self.has_jitter:
            diags = self.diag + (jnp.exp(params["log_jitter"]) ** 2)[band]
        else:
            diags = self.diag

        lag_disk = params["lag0"] * (self.lam_rf / 2500.0) ** params["lag_beta"]

        kernel = ContiBLRQS_Mag(
            amp_cont=jnp.exp(log_sigma_band),
            amp_blr=jnp.exp(log_sigma_band_blr),
            tau_drw=jnp.exp(log_tau_band),
            lag_disk=lag_disk, 
            lag_blr=jnp.exp(params["log_lag_blr"]),
            bwb_alpha=params["bwb_alpha"],
            bwb_beta=params["bwb_beta"],
            width_cont=params["width_cont"],
            width_blr=params["width_blr"],
            s=s
        )

        gp = GaussianProcess(kernel, (t, band), diag=diags + 1e-6, mean=means)
        return gp, jnp.arange(t.shape[0])

    def my_lag_transform(
        self, X: JAXArray, has_lag: bool, params: dict[str, JAXArray]
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray]:
        t, band = X
        return (t, band), jnp.arange(t.shape[0])

    def my_amp_transform_blr(self, params: dict[str, JAXArray]) -> JAXArray:
        return params["log_sigma0"] + jnp.atleast_1d(params["log_amp_delta_blr"])
    
    # def my_tau_drw_transform(self, params: dict[str, JAXArray]) -> JAXArray:
    #      eta_tau1 = params["eta_tau1"]
    #      eta_tau2 = params["eta_tau2"]
    #      lam_s = params["lam_s"]
    #      eta_break = params["eta_break"]
    #      lam_rf_mean = jnp.mean(self.lam_rf)
    #      log_tau_band_mean = params["log_tau_drw0"] + jnp.log(10) * log_broken_pl(lam_rf_mean, lam_s, eta_tau1, eta_tau2, eta_break)
    #      return log_tau_band_mean

    def my_tau_drw_transform(self, params: dict[str, JAXArray]) -> JAXArray:
        eta_tau1 = params["eta_tau1"]
        eta_tau2 = params["eta_tau2"]
        lam_s = params["lam_s"]
        lam_rf_mean = jnp.mean(self.lam_rf)
        eta_break = params["eta_break"]
        log_tau_band_mean = params["log_tau_drw0"] + jnp.log(10) * log_broken_pl(lam_rf_mean, lam_s, eta_tau1, eta_tau2, eta_break)
        log_tau_band = params["log_tau_drw0"] + jnp.log(10) * log_broken_pl(self.lam_rf, lam_s, eta_tau1, eta_tau2, eta_break)
        return log_tau_band_mean, jnp.exp(log_tau_band_mean)/jnp.exp(log_tau_band)

    def my_amp_transform(self, params: dict[str, JAXArray]) -> JAXArray:
        """
        Transform the amplitude parameters for the model.
        """
        eta_A1 = params["eta_A1"]
        eta_A2 = params["eta_A2"]
        lam_s = params["lam_s"]
        eta_break = params["eta_break"]

        # Host dilution: apply per-band correction
        # Host galaxy contribution modeled as a power-law in wavelength
        host_frac = params["f_host"] * (self.lam_rf / 5100.0) ** (params["alpha_host"] - params["alpha_agn"])
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
        gp, _ = self._build_gp(params)
        return gp.log_probability(y=self.y)

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
        # 1) Build GP; _build_gp should return the training sort order it used.
        gp, order = self._build_gp(params)
        y_sorted = self.y[order]

        # 2) Sort test inputs by tie-broken key (time + tiny band offset)
        t_test, b_test = X
        b_test = jnp.asarray(b_test, dtype=jnp.int32)

        tie_eps  = 10.0 * jnp.finfo(t_test.dtype).eps
        key_test = t_test + b_test.astype(t_test.dtype) * tie_eps
        otest    = jnp.argsort(key_test)

        t_pred = t_test[otest]
        b_pred = b_test[otest]

        # 3) Condition on sorted test inputs; add tiny jitter for numerical PD
        _, cond = gp.condition(y_sorted, (t_pred, b_pred), diag=1e-10)

        # 4) Map predictions back to the original test order
        inv = jnp.empty_like(otest)
        inv = inv.at[otest].set(jnp.arange(otest.shape[0]))

        mean = cond.loc[inv]
        var  = cond.variance[inv]
        std  = jnp.sqrt(jnp.clip(var, 0.0, jnp.inf))
        return mean, std

    def psd(
        self, params: dict[str, JAXArray], omega: JAXArray, b: int, sigma_n2: float = 0.0
    ) -> JAXArray:
        """
        Get the power spectral density (PSD) for a specific band.

        Parameters
        ----------
        params : dict[str, JAXArray]
            Model parameters.
        omega : JAXArray
            Angular frequencies [rad / time units].
        b : int
            Band index.
        sigma_n2 : float, optional
            White-noise level to add (default 0.0).

        Returns
        -------
        JAXArray
            Real non-negative power spectral density.
        """
        gp, _ = self._build_gp(params)
        return gp.kernel.psd(omega, b, sigma_n2)

def sample_drw_tinygp(key, t, tau, sigma, noise=0.0, mean=0.0):
    """
    Draw y ~ GP(mean, k), k(Δt) = sigma^2 * exp(-|Δt|/tau)
    t: (N,) times (irregular OK); tau>0; sigma is the long-term std (k(0)^{1/2})
    """
    k = (sigma**2) * kernels.Exp(scale=tau) 
    gp = GaussianProcess(k, t, diag=(noise**2), mean=mean)
    return gp.sample(key, shape=(len(t),))