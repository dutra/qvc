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

class ContiBLRQS_Mag(qs.Wrapper):
    """
    Multiband OU/DRW kernel in *magnitudes* with band-dependent correlation
    timescales τ_b = tau0 / s[b], per-band continuum/BLR gains, lags, and
    rectangular smoothing. Designed to keep bands coherent.

    - No global time warp for sorting (shared clock).
    - Band dependence enters via *dt* scaling:
        within-band: dt <- s[b] * dt         (τ_b effect)
        cross-band:  dt <- sqrt(s[b1]*s[b2]) * (t2 - t1)
    """

    amp_cont:  jnp.ndarray
    amp_blr:   jnp.ndarray
    lag_blr:   jnp.ndarray
    lag_disk:  jnp.ndarray
    width_cont:jnp.ndarray
    width_blr: jnp.ndarray
    s:         jnp.ndarray        # >0, encodes τ_b = tau0 / s[b]
    kernel:    qs.Kernel          # base OU with scale=tau0, sigma=1

    def __init__(self, amp_cont, amp_blr, lag_blr, lag_disk,
                 width_cont, width_blr, s, tau_drw):
        # Safety floors
        eps = 1e-12

        self.amp_cont   = jnp.asarray(amp_cont)
        self.amp_blr    = jnp.asarray(amp_blr)
        self.lag_blr    = jnp.asarray(lag_blr)
        self.lag_disk   = jnp.asarray(lag_disk)

        # Clamp widths to >= 0
        self.width_cont = jnp.maximum(jnp.asarray(width_cont), 0.0)
        self.width_blr  = jnp.maximum(jnp.asarray(width_blr),  0.0)

        # Positive stretch factors (encode band τ)
        self.s          = jnp.maximum(jnp.asarray(s), eps)

        # One latent OU with "center" τ = tau_drw; unit variance
        self.kernel     = qs.Exp(scale=jnp.maximum(jnp.asarray(tau_drw), eps),
                                 sigma=1.0)

    # ---------- Sorting on a shared clock ----------
    def coord_to_sortable(self, X):
        t, b = X
        # shared clock; tiny tie-break on band to avoid equal keys
        return t + 1e-9 * (jnp.asarray(b, jnp.int32))

    # ---------- Stable top-hat gain in u (scaled) units ----------
    @staticmethod
    def _gain_top_hat_u(width_u, tau0):
        tau0 = jnp.maximum(tau0, 1e-12)
        x = width_u / tau0
        small = 1.0 - 0.5 * x + (x * x) / 6.0
        full  = -jnp.expm1(-x) / jnp.maximum(x, 1e-12)
        return jnp.where(x < 1e-8, small, full)

    # ---------- Within-band convolution/lag (uses band s[b]) ----------
    def _conv_obs_band(self, lag_t, width_t, s_b):
        # Map to the latent's "u" units so τ_b = tau0 / s_b
        lag_u   = s_b * lag_t
        width_u = s_b * width_t

        h0  = self.kernel.observation_model(jnp.array(0.0))       # (1,)
        Phi = self.kernel.transition_matrix(0.0, lag_u)           # (1,1)
        G   = self._gain_top_hat_u(width_u, self.kernel.scale)    # scalar in [0,1]
        return (h0 @ Phi) * G                                     # (1,)

    # ---------- Band-specific observation vector ----------
    def observation_model(self, X):
        _, b = X
        b = jnp.asarray(b, dtype=jnp.int32)

        h_cont = self.amp_cont[b] * self._conv_obs_band(
            self.lag_disk[b], self.width_cont[b], self.s[b]
        )
        h_blr  = self.amp_blr[b] * self._conv_obs_band(
            self.lag_disk[b] + self.lag_blr[b], self.width_blr[b], self.s[b]
        )
        return h_cont + h_blr

    # ---------- Cross-/within-band state propagation with symmetric stretch ----------
    def transition_matrix(self, X1, X2):
        """
        Coherent clock between any two observations:
          dt_eff = sqrt(s[b1] * s[b2]) * (t2 - t1)
        This yields τ_b effects while avoiding cross-band dephasing.
        """
        t1, b1 = X1
        t2, b2 = X2
        b1 = jnp.asarray(b1, dtype=jnp.int32)
        b2 = jnp.asarray(b2, dtype=jnp.int32)

        s_eff = jnp.sqrt(self.s[b1] * self.s[b2])
        dt    = s_eff * (t2 - t1)
        return self.kernel.transition_matrix(0.0, dt)

    # ---------- Optional PSD helper (guaranteed ≥ 0) ----------
    def psd(self, omega: JAXArray, b: int, sigma_n2: float = 0.0) -> JAXArray:
        A  = self.design_matrix()
        P  = self.stationary_covariance()
        Qc = -(A @ P + P @ A.T)
        I  = jnp.eye(A.shape[0], dtype=A.dtype)
        h  = self.observation_model((jnp.array(0.0), jnp.array(int(b))))

        def one_w(w):
            v = jnp.linalg.solve((-1j * w) * I - A.T, h)
            return (v.conj().T @ (Qc @ v)).real + sigma_n2

        return 2.0 * jax.vmap(one_w)(omega)



# Override MultiVarModel
class MyMultiVarModel_SMAG(MultiVarModel):
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
            tau_drw=jnp.exp(log_tau_center),
            lag_disk=lag_disk, 
            lag_blr=jnp.exp(params["log_lag_blr"]),
            #bwb_alpha=params["bwb_alpha"],
            #bwb_beta=params["bwb_beta"],
            width_cont=params["width_cont"],
            width_blr=params["width_blr"],
            s=s
        )

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
    
    def my_tau_drw_transform(self, params: dict[str, JAXArray]) -> JAXArray:
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
        return gp.kernel.psd(omega, b, sigma_n2)

def sample_drw_tinygp(key, t, tau, sigma, noise=0.0, mean=0.0):
    """
    Draw y ~ GP(mean, k), k(Δt) = sigma^2 * exp(-|Δt|/tau)
    t: (N,) times (irregular OK); tau>0; sigma is the long-term std (k(0)^{1/2})
    """
    k = (sigma**2) * kernels.Exp(scale=tau) 
    gp = GaussianProcess(k, t, diag=(noise**2), mean=mean)
    return gp.sample(key, shape=(len(t),))