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

# ----- helper (kept outside class to avoid retracing in some setups) -----
def _top_hat_gain(width, tau):
    """
    Gain for convolving an OU(τ) with a top-hat of width W:
      G = (1 - exp(-W/τ)) / (W/τ)
    with stable series for tiny W/τ.
    """
    x = width / jnp.maximum(tau, 1e-12)
    small = 1.0 - 0.5 * x + (x * x) / 6.0   # 1 - x/2 + x^2/6
    full  = -jnp.expm1(-x) / jnp.maximum(x, 1e-16)
    return jnp.where(x < 1e-8, small, full)



class ContiBLR_BWB_QS(qs.Quasisep):
    """
    Continuum+BLR kernel in magnitudes, with per-band τ and built-in BWB.

    State size = 2B: [u_0..u_{B-1}, v_0..v_{B-1}]
      u_b ~ OU(τ_u[b]), v_b ~ OU(τ_v[b]) with τ_u[b] = tau_band[b],
                                        and τ_v[b] = beta * tau_band[b].

    Continuum (band b, after disk lag & smoothing):
        weights on [u_b, v_b] = [ a_b * G(width_cont[b], τ_ref[b]),
                                  c_b * G(width_cont[b], τ_ref[b]) ]
        with c_b = sigma_bwb * (amp_cont[b]**exp_bwb) * G(width_bwb[b], τ_ref[b])

    BLR (band b, after disk+BLR lag & smoothing):
        amp_blr[b] * [ a_b, (kappa_blr_bwb * c_b) ] with G(width_blr[b], τ_ref[b])

    Args
    ----
    tau_band        : (B,)   per-band base OU timescale (sets τ_u)
    beta            : (B,) or ()  factor so τ_v = beta * tau_band
    rho             : ()     global u-v correlation (-1<rho<1) at stationarity
    amp_cont        : (B,)   per-band continuum loading (== your old amp_cont)
    sigma_bwb       : (B,) or ()  BWB scale
    exp_bwb         : ()     exponent for amp_cont in c_b
    width_bwb       : (B,)   top-hat width for BWB smoothing
    tau_ref         : (B,)   reference τ for gains (often tau_band)
    amp_blr         : (B,)   BLR amplitude scaling
    kappa_blr_bwb   : (B,) or ()  relative BWB strength inside BLR (1 = inherit)
    lag_disk        : (B,)   disk lag (>=0)
    lag_blr         : (B,)   extra BLR lag (>=0)
    width_cont      : (B,)   continuum top-hat width
    width_blr       : (B,)   BLR top-hat width
    """

    # --- timescales ---
    tau_band: jnp.ndarray     # (B,)
    beta: jnp.ndarray | float # () or (B,)
    rho: float

    # --- continuum + BWB ---
    amp_cont: jnp.ndarray     # (B,)
    sigma_bwb: jnp.ndarray    # (B,) or ()
    exp_bwb: float
    width_bwb: jnp.ndarray    # (B,)
    tau_ref: jnp.ndarray      # (B,) #TODO, remove

    # --- BLR ---
    amp_blr: jnp.ndarray       # (B,)
    kappa_blr_bwb: jnp.ndarray # (B,) or ()
    lag_disk: jnp.ndarray      # (B,)
    lag_blr: jnp.ndarray       # (B,)
    width_cont: jnp.ndarray    # (B,)
    width_blr: jnp.ndarray     # (B,)

    # ---------- tinygp hooks ----------
    def coord_to_sortable(self, X):
        t, b = X
        return t + 1e-9 * jnp.asarray(b, jnp.int32)

    # A = diag(-1/τ_u, ..., -1/τ_u, -1/τ_v, ..., -1/τ_v)
    def design_matrix(self) -> JAXArray:
        tau_u = jnp.maximum(self.tau_band, 1e-12)
        beta  = self.beta if jnp.ndim(self.beta) > 0 else jnp.full_like(tau_u, self.beta)
        tau_v = jnp.maximum(beta, 1e-12) * tau_u
        lam_u = 1.0 / tau_u
        lam_v = 1.0 / tau_v
        return -jnp.diag(jnp.concatenate([lam_u, lam_v], axis=0))

    # Stationary covariance with OU cross-band coupling and u-v correlation ρ
    def stationary_covariance(self) -> JAXArray:
        def _P_from_tau(ti, tj):
            ti = jnp.maximum(ti, 1e-12)
            tj = jnp.maximum(tj, 1e-12)
            return 2.0 * jnp.sqrt(ti[:, None] * tj[None, :]) / jnp.maximum(ti[:, None] + tj[None, :], 1e-12)

        tau_u = jnp.maximum(self.tau_band, 1e-12)
        beta  = self.beta if jnp.ndim(self.beta) > 0 else jnp.full_like(tau_u, self.beta)
        tau_v = jnp.maximum(beta, 1e-12) * tau_u

        Puu = _P_from_tau(tau_u, tau_u)  # (B,B)
        Pvv = _P_from_tau(tau_v, tau_v)  # (B,B)

        r   = jnp.clip(self.rho, -0.999, 0.999)
        Puv = r * _P_from_tau(tau_u, tau_v)  # (B,B)
        Pvu = Puv.T

        P = jnp.block([[Puu, Puv],
                       [Pvu, Pvv]])
        return 0.5 * (P + P.T)

    # Φ = diag(exp(-Δt/τ_u...), exp(-Δt/τ_v...))
    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        t1, _ = X1
        t2, _ = X2
        dt = t2 - t1
        tau_u = jnp.maximum(self.tau_band, 1e-12)
        beta  = self.beta if jnp.ndim(self.beta) > 0 else jnp.full_like(tau_u, self.beta)
        tau_v = jnp.maximum(beta, 1e-12) * tau_u
        phi_u = jnp.exp(-dt / tau_u)
        phi_v = jnp.exp(-dt / tau_v)
        return jnp.diag(jnp.concatenate([phi_u, phi_v], axis=0))

    # h(X) row: only band b's u_b and v_b are observed (plus BLR-delayed copies)
    def observation_model(self, X: JAXArray) -> JAXArray:
        _t, b = X
        b = jnp.asarray(b, dtype=jnp.int32)
        B = self.amp_cont.shape[0]

        # reference τ for gains (often tau_band)
        tau_ref_b = jnp.maximum(self.tau_band[b], 1e-12)

        # smoothing gains
        Gc = _top_hat_gain(self.width_cont[b], tau_ref_b)
        Gr = _top_hat_gain(self.width_blr[b],  tau_ref_b)
        Gb = _top_hat_gain(self.width_bwb[b],  tau_ref_b)

        # continuum weights (pre-lag)
        a_b = self.amp_cont[b]
        c_b = self.sigma_bwb * (jnp.abs(a_b) ** self.exp_bwb) * Gb  # BWB term

        # BLR scaling (same chromatic structure as continuum)
        amp   = self.amp_blr[b]
        kap   = self.kappa_blr_bwb
        kap_b = kap if jnp.ndim(kap) == 0 else kap[b]

        # indices in the 2B state
        idx_u = b
        idx_v = B + b

        # compute per-band lags and decay factors
        tau_u_b = jnp.maximum(self.tau_band[b], 1e-12)
        beta_b  = self.beta if jnp.ndim(self.beta) == 0 else self.beta[b]
        tau_v_b = jnp.maximum(beta_b, 1e-12) * tau_u_b

        lag_d  = jnp.maximum(self.lag_disk[b], 0.0)
        lag_rb = jnp.maximum(self.lag_blr[b],  0.0)

        phi_u_d  = jnp.exp(-lag_d  / tau_u_b)
        phi_v_d  = jnp.exp(-lag_d  / tau_v_b)
        phi_u_dr = jnp.exp(-(lag_d + lag_rb) / tau_u_b)
        phi_v_dr = jnp.exp(-(lag_d + lag_rb) / tau_v_b)

        # assemble observation row
        h = jnp.zeros(2 * B, dtype=self.amp_cont.dtype)

        # continuum contribution (delayed by disk lag)
        h = h.at[idx_u].set(h[idx_u] + (a_b * Gc) * phi_u_d)
        h = h.at[idx_v].set(h[idx_v] + (c_b * Gc) * phi_v_d)

        # BLR contribution (disk+BLR lag, BLR width)
        h = h.at[idx_u].set(h[idx_u] + amp * (a_b * Gr)            * phi_u_dr)
        h = h.at[idx_v].set(h[idx_v] + amp * (kap_b * c_b * Gr)    * phi_v_dr)

        return h


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
        log_tau_center, log_tau_band     = self.my_tau_drw_transform(params)

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

        #kernel_contBLR = ContiBLR_QS(
        #    amp_cont=jnp.exp(log_sigma_band),
        #    amp_blr=jnp.exp(log_sigma_band_blr),
        #    lag_disk=lag_disk,
        #    lag_blr=jnp.exp(params["log_lag_blr"]),
        #    width_cont=params["width_cont"],
        #    width_blr=params["width_blr"],
        #    tau=jnp.exp(log_tau_band)
        #)
        
        kernel_bwb = ContiBLR_BWB_QS(
            tau_band=jnp.exp(log_tau_band),
            beta=params["bwb_beta"],
            rho=1.0,
            amp_cont=jnp.exp(log_sigma_band),
            sigma_bwb=params["bwb_alpha"],
            exp_bwb=1.0,
            width_bwb=params["width_cont"],
            tau_ref=jnp.exp(log_tau_center),
            lag_disk=lag_disk,
            # BLR parameters
            amp_blr=jnp.exp(log_sigma_band_blr),
            kappa_blr_bwb=1.0,
            lag_blr=jnp.exp(params["log_lag_blr"]),
            width_cont=params["width_cont"],
            width_blr=params["width_blr"]
        )

        kernel = kernel_bwb #kernel_contBLR + kernel_bwb

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