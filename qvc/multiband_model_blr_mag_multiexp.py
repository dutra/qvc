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


import jax
import jax.numpy as jnp
from tinygp.kernels import quasisep as qs
from tinygp.helpers import JAXArray

import jax
import jax.numpy as jnp
from tinygp.kernels import quasisep as qs
from tinygp.helpers import JAXArray

import jax
import jax.numpy as jnp
from tinygp.kernels import quasisep as qs
from tinygp.helpers import JAXArray

import jax.numpy as jnp
import tinygp.kernels.quasisep as qs

# New multi-band OU kernel with band-dependent timescales
class MultiExp(qs.Quasisep):
    """Multiband OU kernel (LMC with fully correlated noise) for B bands.
    Each band b has its own timescale tau[b], but all share a common driving process.
    This yields cross-band correlations and preserves PSD.
    """
    tau: jnp.ndarray  # array of shape (B,) with tau for each band

    def coord_to_sortable(self, X):
        # Sort by time (band ordering handled via tiny offset in wrapper if needed)
        return X[0]

    def observation_model(self, X):
        # Return observation vector (length B) that picks out the state for band b
        t, b = X
        B = self.tau.shape[0]
        obs = jnp.zeros(B, dtype=self.tau.dtype)
        obs = obs.at[b].set(1.0)
        return obs

    def design_matrix(self):
        # State matrix A: diagonal of -1/tau_b for each band state
        lam = 1.0 / jnp.maximum(self.tau, 1e-12)  # decay rates per band
        return -jnp.diag(lam)

    def stationary_covariance(self):
        # Solve A*P + P*A^T + Q = 0 for stationary cov P. Using common noise:
        # Choose Q = c c^T with c_i = sqrt(2*lam_i) so that Var(state_i)=1.
        tau_i = self.tau[:, None]
        tau_j = self.tau[None, :]
        # P_ij = 2 * sqrt(tau_i * tau_j) / (tau_i + tau_j)
        P = 2.0 * jnp.sqrt(tau_i * tau_j) / jnp.maximum(tau_i + tau_j, 1e-12)
        # Ensure symmetry (numerical safety)
        return (P + P.T) / 2.0

    def transition_matrix(self, t1, t2):
        # State transition over time difference dt = t2 - t1 (A is diagonal)
        dt = t2 - t1
        lam = 1.0 / jnp.maximum(self.tau, 1e-12)
        return jnp.diag(jnp.exp(-lam * dt))


class ContiBLR_QS(qs.Wrapper):
    """
    PSD-safe multiband OU/DRW kernel in magnitudes with continuum+BLR components.
    Now uses a MultiExp latent kernel to capture wavelength-dependent tau.
    """
    amp_cont:   jnp.ndarray
    amp_blr:    jnp.ndarray
    lag_blr:    jnp.ndarray
    lag_disk:   jnp.ndarray
    width_cont: jnp.ndarray
    width_blr:  jnp.ndarray
    s:          jnp.ndarray        # tau_ref / tau_band for each band
    kernel:     qs.Quasisep        # underlying latent kernel (MultiExp)

    def __init__(self, amp_cont, amp_blr, lag_blr, lag_disk,
                 width_cont, width_blr, s, tau_drw):
        eps = 1e-12
        self.amp_cont   = jnp.asarray(amp_cont)
        self.amp_blr    = jnp.asarray(amp_blr)
        self.lag_blr    = jnp.asarray(lag_blr)
        self.lag_disk   = jnp.asarray(lag_disk)
        self.width_cont = jnp.maximum(jnp.asarray(width_cont), 0.0)
        self.width_blr  = jnp.maximum(jnp.asarray(width_blr),  0.0)
        self.s          = jnp.maximum(jnp.asarray(s), eps)
        # Compute per-band tau: tau_band = tau_drw (ref) / s[b]
        tau_ref = jnp.maximum(jnp.asarray(tau_drw), eps)
        tau_band = tau_ref / self.s  # array of length B

        # Latent multi-band OU kernel with band-specific timescales
        self.kernel = MultiExp(tau=tau_band)

    def coord_to_sortable(self, X):
        # Shared clock: sort primarily by time, break ties by band index
        t, b = X
        return t + 1e-9 * jnp.asarray(b, jnp.int32)

    @staticmethod
    def _gain_top_hat(width, tau):
        # Top-hat convolution gain for an OU process with timescale tau
        x = width / jnp.maximum(tau, 1e-12)
        # Use series expansion for very small x to avoid numerical issues
        small = 1.0 - 0.5 * x + (x * x) / 6.0
        full  = -jnp.expm1(-x) / jnp.maximum(x, 1e-12)  # (1 - exp(-x)) / x
        return jnp.where(x < 1e-8, small, full)

    def _conv_obs(self, band_idx, lag, width):
        # Compute the convolution observation vector for a given band, lag, and width
        # band_idx: integer index of the band
        # 1. Start with the band’s state at t (observation at time 0):
        h0 = self.kernel.observation_model((jnp.array(0.0), jnp.array(band_idx)))
        # 2. Evolve state from t to t+lag using transition matrix:
        Phi = self.kernel.transition_matrix(jnp.array(0.0), jnp.array(lag))
        # h0 is 1xB, Phi is BxB -> h0 @ Phi yields 1xB (the band’s state decayed over lag)
        h_lag = h0 @ Phi  # this is a 1xB row vector
        # Since h0 selects the band_idx state, h_lag will have a non-zero at band_idx
        # 3. Apply top-hat filter gain for the band’s timescale:
        tau_b = self.kernel.tau[band_idx]            # band-specific tau
        G     = self._gain_top_hat(width, tau_b)     # scalar gain
        return h_lag * G    # 1xB vector (mostly with band_idx component)

    def observation_model(self, X):
        # X = (t_array, band_array) for an observation
        # For each observation, return a 1xB observation vector
        t, b = X
        b = jnp.asarray(b, dtype=jnp.int32)
        # Continuum component (disk) for this band
        h_cont = self.amp_cont[b] * self._conv_obs(b, self.lag_disk[b], self.width_cont[b])
        # BLR component for this band (disk lag + additional BLR lag)
        h_blr  = self.amp_blr[b]  * self._conv_obs(b, self.lag_disk[b] + self.lag_blr[b],
                                                  self.width_blr[b])
        # Sum continuum and BLR contributions
        return h_cont + h_blr

def qs_psd(kernel, omega, b: int, sigma_n2: float = 0.0):
    """
    One-sided PSD for any tinygp.quasisep kernel (including qs.Sum).
    Assumes your kernel's observation_model accepts X=(t, band).
    """
    A  = kernel.design_matrix()
    P  = kernel.stationary_covariance()
    Qc = -(A @ P + P @ A.T)
    I  = jnp.eye(A.shape[0], dtype=A.dtype)

    # observation vector for band b at t=0 (no loss of generality for stationary kernels)
    h  = kernel.observation_model((jnp.array(0.0), jnp.array(int(b))))

    def one_w(w):
        v = jnp.linalg.solve((-1j * w) * I - A.T, h)
        return (v.conj().T @ (Qc @ v)).real + sigma_n2

    return 2.0 * jax.vmap(one_w)(omega)


# Override MultiVarModel
class MyMultiVarModel_SMAG_MultiExp(MultiVarModel):
    yerr: JAXArray | NDArray
    z: float
    lam_rf: JAXArray
    broken_pl: bool

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
        self.broken_pl =kwargs["broken_pl"]

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
        log_tau_center, s     = self.my_tau_drw_transform(params)

        t, band = self.X
        t_center = jnp.mean(t)
        t_std    = jnp.std(t)

        means = partial(MyMultiVarModel_SMAG_MultiExp.mean_func, self.zero_mean, log_sigma_band.shape[0],
                        t_center, t_std, params)

        # diagonal noise in original order
        if self.has_jitter:
            diags = self.diag + (jnp.exp(params["log_jitter"]) ** 2)[band]
        else:
            diags = self.diag

        lag_disk = params["lag0"] * (self.lam_rf / 2500.0) ** params["lag_beta"]

        kernel_contBLR = ContiBLR_QS(
            amp_cont=jnp.exp(log_sigma_band),
            amp_blr=jnp.exp(log_sigma_band_blr),
            tau_drw=jnp.exp(log_tau_center),
            lag_disk=lag_disk, 
            lag_blr=jnp.exp(params["log_lag_blr"]),
            #bwb_alpha=params["bwb_alpha"],
            #bwb_beta=params["bwb_beta"],
            width_cont=params["width_cont"],
            width_blr=params["width_blr"],
            s=s
        )

        kernel = kernel_contBLR #+ kernel_bwb

        u = kernel.coord_to_sortable((t, band))
        order = jnp.argsort(u)
        t, band = t[order], band[order]
        diags = diags[order]

        gp = GaussianProcess(kernel, (t, band), diag=diags + 1e-6, mean=means)
        return gp, order

    def my_lag_transform(
        self, X: JAXArray, has_lag: bool, params: dict[str, JAXArray]
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray]:
        t, band = X
        return (t, band), jnp.arange(t.shape[0])

    def my_amp_transform_blr(self, params: dict[str, JAXArray]) -> JAXArray:
        return params["log_sigma0"] + jnp.atleast_1d(params["log_amp_delta_blr"])
    
    def my_tau_drw_transform(self, params: dict[str, JAXArray]) -> tuple[JAXArray, JAXArray]:
        log_pl = log_broken_pl if self.broken_pl else log_single_pl
        eta_tau1 = params["eta_tau1"]
        eta_tau2 = params["eta_tau2"]
        lam_s = params["lam_s"]
        lam_rf_mean = jnp.mean(self.lam_rf)
        eta_break = params["eta_break"]
        log_tau_band_mean = params["log_tau_drw0"] + jnp.log(10) * log_pl(lam_rf_mean, lam_s, eta_tau1, eta_tau2, eta_break)
        log_tau_band = params["log_tau_drw0"] + jnp.log(10) * log_pl(self.lam_rf, lam_s, eta_tau1, eta_tau2, eta_break)
        return log_tau_band_mean, jnp.exp(log_tau_band_mean)/jnp.exp(log_tau_band)

    def my_amp_transform(self, params: dict[str, JAXArray]) -> JAXArray:
        """
        Transform the amplitude parameters for the model.
        """
        log_pl = log_broken_pl if self.broken_pl else log_single_pl
        
        eta_A1 = params["eta_A1"]
        eta_A2 = params["eta_A2"]
        lam_s = params["lam_s"]
        eta_break = params["eta_break"]

        # Host dilution: apply per-band correction
        # Host galaxy contribution modeled as a power-law in wavelength
        host_frac = params["f_host"] * (self.lam_rf / 2500.0) ** (params["alpha_host"] - params["alpha_agn"])
        dilution_factor = 1.0 / (1.0 + host_frac)
        log_dilution = jnp.log(dilution_factor)

        # Power-law scaling across rest-frame wavelength
        log_sigma_band = params["log_sigma0"] + log_dilution + jnp.log(10) * log_pl(self.lam_rf, lam_s, eta_A1, eta_A2, eta_break)

        return log_sigma_band
    
    @eqx.filter_jit
    def log_prob(self, params: dict[str, JAXArray]) -> JAXArray:
        gp, order = self._build_gp(params)
        y_sorted = self.y[order]
        return gp.log_probability(y=y_sorted)

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
        y_test = self.y[order]

        # 2) Sort test inputs by tie-broken key (time + tiny band offset)
        t_test, b_test = X
        b_test = jnp.asarray(b_test, dtype=jnp.int32)

        # 3) Condition on sorted test inputs; add tiny jitter for numerical PD
        _, cond = gp.condition(y_test, (t_test, b_test), diag=1e-10)

        mean = cond.loc
        var  = cond.variance
        jax.debug.print(" var: {}", var)
        jax.debug.print("var min/max: {}/{}", jnp.min(var), jnp.max(var))
        jax.debug.print("mean min/max: {}/{}", jnp.min(mean), jnp.max(mean))
        std  = jnp.sqrt(var)    
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
        return qs_psd(kernel=gp.kernel, omega=omega, b=b, sigma_n2=sigma_n2)

def sample_drw_tinygp(key, t, tau, sigma, noise=0.0, mean=0.0):
    """
    Draw y ~ GP(mean, k), k(Δt) = sigma^2 * exp(-|Δt|/tau)
    t: (N,) times (irregular OK); tau>0; sigma is the long-term std (k(0)^{1/2})
    """
    k = (sigma**2) * kernels.Exp(scale=tau) 
    gp = GaussianProcess(k, t, diag=(noise**2), mean=mean)
    return gp.sample(key, shape=(len(t),))