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
from jax.scipy.linalg import expm
from tinygp.kernels import quasisep as qs
import jax
import jax.numpy as jnp
from tinygp.kernels import quasisep as qs

# -------- Conti+BLR shared-clock QS kernel (TinyGP Quasisep) --------
import jax
import jax.numpy as jnp
import equinox as eqx
from jax.scipy.linalg import expm
from tinygp.helpers import JAXArray
from tinygp.kernels.quasisep import Quasisep

import jax.numpy as jnp
import tinygp.kernels.quasisep as qs
import jax.numpy as jnp
import tinygp.kernels.quasisep as qs

class ContiBLR_QS(qs.Quasisep):
    """
    Single-class, PSD-safe multiband OU/DRW kernel with continuum + BLR components.

    Args
    ----
    tau:        (B,) OU timescale per band (same units as lags/widths)
    amp_cont:   (B,) continuum gain per band
    amp_blr:    (B,) BLR gain per band
    lag_blr:    (B,) additional BLR lag per band
    lag_disk:   (B,) disk lag per band
    width_cont: (B,) top-hat width for continuum per band
    width_blr:  (B,) top-hat width for BLR per band
    """
    tau:        jnp.ndarray
    amp_cont:   jnp.ndarray
    amp_blr:    jnp.ndarray
    lag_blr:    jnp.ndarray
    lag_disk:   jnp.ndarray
    width_cont: jnp.ndarray
    width_blr:  jnp.ndarray

    # ---------- tinygp hooks ----------
    def coord_to_sortable(self, X):
        t, b = X
        return t + 1e-9 * jnp.asarray(b, jnp.int32)

    def design_matrix(self):
        lam = 1.0 / jnp.maximum(self.tau, 1e-12)
        return -jnp.diag(lam)

    def stationary_covariance(self):
        # P_ij = 2 sqrt(tau_i tau_j) / (tau_i + tau_j)
        tau = jnp.maximum(self.tau, 1e-12)
        ti = tau[:, None]
        tj = tau[None, :]
        P  = 2.0 * jnp.sqrt(ti * tj) / jnp.maximum(ti + tj, 1e-12)
        return 0.5 * (P + P.T)

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        t1, b1 = X1
        t2, b2 = X2
        dt  = t2 - t1
        lam = 1.0 / jnp.maximum(self.tau, 1e-12)
        return jnp.diag(jnp.exp(-lam * dt))

    def observation_model(self, X: JAXArray) -> JAXArray:
        _t, b = X
        b = jnp.asarray(b, dtype=jnp.int32)

        amp_c = self.amp_cont[b]
        amp_r = self.amp_blr[b]
        lag_d = jnp.maximum(self.lag_disk[b], 0.0)
        lag_r = jnp.maximum(self.lag_blr[b],  0.0)
        w_c   = jnp.maximum(self.width_cont[b], 0.0)
        w_r   = jnp.maximum(self.width_blr[b],  0.0)

        B   = self.tau.shape[0]
        e_b = jnp.zeros(B, dtype=self.tau.dtype).at[b].set(1.0)

        Phi_d   = self.transition_matrix((0.0, 0.0), (lag_d, jnp.zeros_like(lag_d)))
        Phi_dr  = self.transition_matrix((0.0, 0.0), (lag_d + lag_r, jnp.zeros_like(lag_d)))
        h_cont0 = e_b @ Phi_d
        h_blr0  = e_b @ Phi_dr

        tau_b = jnp.maximum(self.tau[b], 1e-12)
        Gc    = _top_hat_gain(w_c, tau_b)
        Gr    = _top_hat_gain(w_r, tau_b)

        return amp_c * (Gc * h_cont0) + amp_r * (Gr * h_blr0)

import jax
import jax.numpy as jnp
from tinygp.kernels import quasisep as qs
from tinygp.helpers import JAXArray

def _safe_pos(x, eps=1e-12):
    return jnp.maximum(x, eps)

def _top_hat_gain(width, tau_b):
    w = _safe_pos(width)
    t = _safe_pos(tau_b)
    return jnp.where(w < 1e-9, 1.0, (t / w) * (1.0 - jnp.exp(-w / t)))

class ContiBLR_CARMA2_QS(qs.Quasisep):
    """
    Multiband PSD-safe CARMA(2) via two shared-noise OU blocks (fast/slow).
    BLR is a perfect lagged replica of the CARMA(2) continuum (same fast/slow mix),
    with its own amplitude and (optional) top-hat smoothing width.

    Args
    ----
    tau_fast:         (B,) fast OU timescale per band
    tau_slow:         (B,) slow OU timescale per band
    amp_cont:     (B,) continuum gain per band
    amp_blr:      (B,) BLR gain per band
    mix:          (B,) in (0,1): fraction from fast block (1-mix from slow) for BOTH cont & BLR
    lag_disk:     (B,) disk lag per band (applied to both cont & BLR)
    lag_blr:      (B,) extra BLR lag per band (so BLR lag = lag_disk + lag_blr)
    width_cont:   (B,) top-hat width for continuum per band
    width_blr:    (B,) top-hat width for BLR per band
    """

    tau_fast:        jnp.ndarray
    tau_slow:        jnp.ndarray
    amp_cont:    jnp.ndarray
    amp_blr:     jnp.ndarray
    mix:         jnp.ndarray
    lag_disk:    jnp.ndarray
    lag_blr:     jnp.ndarray
    width_cont:  jnp.ndarray
    width_blr:   jnp.ndarray

    # ---------- tinygp hooks ----------
    def coord_to_sortable(self, X):
        t, b = X
        return t + 1e-9 * jnp.asarray(b, jnp.int32)

    def _B(self):
        return self.tau_fast.shape[0]

    def design_matrix(self):
        # Two independent OU blocks (fast/slow)
        lam1 = 1.0 / _safe_pos(self.tau_fast)
        lam2 = 1.0 / _safe_pos(self.tau_slow)
        A1 = -jnp.diag(lam1)
        A2 = -jnp.diag(lam2)
        return jax.scipy.linalg.block_diag(A1, A2)

    def stationary_covariance(self):
        # Shared-noise cross-band covariance per block: P_ij = 2 sqrt(ti tj) / (ti + tj)
        def _P(tau):
            tau = _safe_pos(tau)
            ti, tj = tau[:, None], tau[None, :]
            P = 2.0 * jnp.sqrt(ti * tj) / _safe_pos(ti + tj)
            return 0.5 * (P + P.T)

        P1 = _P(self.tau_fast)
        P2 = _P(self.tau_slow)
        Z  = jnp.zeros_like(P1)
        return jnp.block([[P1, Z],
                          [Z,  P2]])

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        t1, _ = X1
        t2, _ = X2
        dt = t2 - t1
        lam1 = 1.0 / _safe_pos(self.tau_fast)
        lam2 = 1.0 / _safe_pos(self.tau_slow)
        Phi1 = jnp.diag(jnp.exp(-lam1 * dt))
        Phi2 = jnp.diag(jnp.exp(-lam2 * dt))
        return jax.scipy.linalg.block_diag(Phi1, Phi2)

    def _basis_vectors(self, b: jnp.ndarray):
        B = self._B()
        e_fast = jnp.zeros(2 * B, dtype=self.tau_fast.dtype).at[b].set(1.0)
        e_slow = jnp.zeros(2 * B, dtype=self.tau_fast.dtype).at[B + b].set(1.0)
        return e_fast, e_slow

    def _lagged_obs(self, e_vec, lag):
        Phi = self.transition_matrix((0.0, 0), (lag, 0))
        return e_vec @ Phi

    def observation_model(self, X: JAXArray) -> JAXArray:
        _t, b = X
        b = jnp.asarray(b, dtype=jnp.int32)

        # Per-band params
        amp_c = self.amp_cont[b]
        amp_r = self.amp_blr[b]
        lag_d = jnp.maximum(self.lag_disk[b], 0.0)
        lag_r = jnp.maximum(self.lag_blr[b],  0.0)
        w_c   = jnp.maximum(self.width_cont[b], 0.0)
        w_r   = jnp.maximum(self.width_blr[b],  0.0)
        #mixb  = jnp.clip(self.mix[b], 0.0, 1.0)
        mixb  = jnp.clip(self.mix, 0.0, 1.0)

        # Basis for fast/slow blocks, then apply lags
        e_fast, e_slow = self._basis_vectors(b)
        h_fast_cont = self._lagged_obs(e_fast, lag_d)
        h_slow_cont = self._lagged_obs(e_slow, lag_d)
        h_fast_blr  = self._lagged_obs(e_fast, lag_d + lag_r)
        h_slow_blr  = self._lagged_obs(e_slow, lag_d + lag_r)

        # Top-hat gains using the band's τ in each block
        Gc_fast = _top_hat_gain(w_c, _safe_pos(self.tau_fast[b]))
        Gc_slow = _top_hat_gain(w_c, _safe_pos(self.tau_slow[b]))
        Gr_fast = _top_hat_gain(w_r, _safe_pos(self.tau_fast[b]))
        Gr_slow = _top_hat_gain(w_r, _safe_pos(self.tau_slow[b]))

        # Same fast/slow mixture for both cont and BLR (BLR is a lagged copy)
        cont = amp_c * ( mixb * (Gc_fast * h_fast_cont) + (Gc_slow * h_slow_cont) )
        blr  = amp_r * ( mixb * (Gr_fast * h_fast_blr)  + (Gr_slow * h_slow_blr) )

        return cont + blr

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
class MyMultiVarModel_SMAG_New(MultiVarModel):
    yerr: JAXArray | NDArray
    z: float
    lam_rf: JAXArray
    broken_pl: bool
    bwb: bool

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
        self.bwb = kwargs["bwb"]

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
        log_tau_center, log_tau_band     = self.my_tau_drw_transform(params, "log_tau_drw0")
        log_tau_fast_center, log_tau_fast_band     = self.my_tau_drw_transform(params, "log_tau_fast0")

        t, band = self.X
        t_center = jnp.mean(t)
        t_std    = jnp.std(t)

        means = partial(MyMultiVarModel_SMAG_New.mean_func, self.zero_mean, log_sigma_band.shape[0],
                        t_center, t_std, params)

        # diagonal noise in original order
        if self.has_jitter:
            diags = self.diag + (jnp.exp(params["log_jitter"]) ** 2)[band]
        else:
            diags = self.diag

        lag_disk = params["lag0"] * (self.lam_rf / 2500.0) ** params["lag_beta"]
        
        if self.bwb:
            kernel = ContiBLR_CARMA2_QS(
                tau_fast=jnp.exp(log_tau_fast_band),
                tau_slow=jnp.exp(log_tau_band),
                mix=params["bwb_alpha"],
                amp_cont=jnp.exp(log_sigma_band),
                amp_blr=jnp.exp(log_sigma_band_blr),
                lag_disk=lag_disk,
                lag_blr=jnp.exp(params["log_lag_blr"]),
                width_cont=params["width_cont"],
                width_blr=params["width_blr"]
            )
        else:
            kernel = ContiBLR_QS(
            amp_cont=jnp.exp(log_sigma_band),
            amp_blr=jnp.exp(log_sigma_band_blr),
            lag_disk=lag_disk,
            lag_blr=jnp.exp(params["log_lag_blr"]),
            width_cont=params["width_cont"],
            width_blr=params["width_blr"],
            tau=jnp.exp(log_tau_band)
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
    
    def my_tau_drw_transform(self, params: dict[str, JAXArray], key: str) -> tuple[JAXArray, JAXArray]:
        log_pl = log_broken_pl if self.broken_pl else log_single_pl
        eta_tau1 = params["eta_tau1"]
        eta_tau2 = params["eta_tau2"]
        lam_s = params["lam_s"]
        lam_rf_mean = jnp.mean(self.lam_rf)
        eta_break = params["eta_break"]
        log_tau_band_mean = params[key] + jnp.log(10) * log_pl(lam_rf_mean, lam_s, eta_tau1, eta_tau2, eta_break)
        log_tau_band = params[key] + jnp.log(10) * log_pl(self.lam_rf, lam_s, eta_tau1, eta_tau2, eta_break)
        return log_tau_band_mean, log_tau_band

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
        # jax.debug.print(" var: {}", var)
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