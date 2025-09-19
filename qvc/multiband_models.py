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

class ContiBLRQS(qs.Wrapper):
    """
    Quasi-separable Gaussian process kernel for multiband AGN light curves, 
    combining continuum, BLR, and bluer-when-brighter (BWB) contributions.

    The covariance between band b1 at time t1 and band b2 at time t2 is

        cov = A_b1 A_b2 k(τ)
            + A_b1 A_blr_b2 (k * TopHat_{Δ_b2,w_b2})(τ)
            + A_blr_b1 A_b2 (k * TopHat_{Δ_b1,w_b1})(-τ)
            + A_blr_b1 A_blr_b2 (k * TopHat_{Δ_b1,w_b1} * TopHat_{Δ_b2,w_b2})(τ)
            + 2 q_b1 q_b2 [k(τ)]^2,

    where τ = t2 - t1, k(τ) is an OU/DRW kernel with timescale τ_drw, and 
    TopHat_{Δ,w} denotes convolution with a top-hat transfer function centered 
    at lag Δ and width w. The last term models the BWB effect as a squared-kernel 
    contribution with amplitude weights q_b.

    Implementation details
    ----------------------
    - Continuum contribution: direct OU process evaluated at time t.
    - BLR contribution: same latent OU, but evaluated at (t - Δ_b), ensuring 
      that a positive lag Δ_b corresponds to the BLR light curve lagging behind 
      the continuum.
    - Top-hat smoothing: each component is rescaled by a factor 
      sinh(w / 2τ_drw) / (w / 2τ_drw), or its series expansion for small width.
    - BWB term: implemented as a second OU kernel with effective timescale 
      τ_drw / bwb_beta (exact square if bwb_beta = 2), and observation weights 
      √2 · q_b with q_b = bwb_alpha · (A_cont · fac_cont)^2.

    Parameters
    ----------
    tau_drw : float
        DRW/OU timescale for the latent continuum kernel.
    width_cont : array_like [B]
        Per-band top-hat width for the continuum component.
    width_blr : array_like [B]
        Per-band top-hat width for the BLR component.
    amp_cont : array_like [B]
        Per-band continuum amplitudes.
    amp_blr : array_like [B]
        Per-band BLR amplitudes.
    lag_blr : array_like [B]
        Per-band BLR lags (positive values imply BLR lags behind the continuum).
    bwb_alpha : array_like [B]
        Scaling coefficient for the BWB squared-kernel term.
    bwb_beta : array_like [B]
        Timescale modifier for the BWB kernel (default ~2 for exact squaring).
    """

    tau_drw: float
    width_cont: jnp.ndarray
    width_blr: jnp.ndarray
    amp_cont: jnp.ndarray
    amp_blr: jnp.ndarray
    lag_blr: jnp.ndarray
    lag_disk: jnp.ndarray
    bwb_alpha: jnp.ndarray
    bwb_beta: jnp.ndarray
    kernel2: qs.Kernel

    def __init__(self, amp_cont, amp_blr, lag_blr, lag_disk, tau_drw, bwb_alpha, bwb_beta, width_cont, width_blr) -> None:
        self.amp_cont = amp_cont
        self.amp_blr = amp_blr
        self.lag_blr = lag_blr
        self.lag_disk  = lag_disk
        self.tau_drw = tau_drw
        self.bwb_alpha = bwb_alpha
        self.bwb_beta = bwb_beta
        self.width_cont = width_cont
        self.width_blr = width_blr
        self.kernel = qs.Exp(scale=self.tau_drw, sigma=1.0)
        self.kernel2 = qs.Exp(scale=self.tau_drw / self.bwb_beta, sigma=1.0)

    def coord_to_sortable(self, X) -> JAXArray:
        t, b = X
        # break ties across bands without changing any modelled times
        return t + jnp.asarray(b, t.dtype) * (10.0 * jnp.finfo(t.dtype).eps)

    # Base matrices (block 0)
    def _A0(self):
        return self.kernel.design_matrix()

    def _P0(self):
        return self.kernel.stationary_covariance()

    def _Phi0(self, dt):
        return self.kernel.transition_matrix(0.0, dt)  # QS kernels ignore absolute times

    # Base matrices (block 1)
    def _A1(self):
        return self.kernel2.design_matrix()

    def _P1(self):
        return self.kernel2.stationary_covariance()

    def _Phi1(self, dt):
        return self.kernel2.transition_matrix(0.0, dt)

    def design_matrix(self) -> JAXArray:
        # Block-diagonal of base and k^2 generators
        A0 = self._A0()
        A1 = self._A1()
        z0 = jnp.zeros((A0.shape[0], A1.shape[1]))
        z1 = jnp.zeros((A1.shape[0], A0.shape[1]))
        return jnp.block([[A0, z0],
                          [z1, A1]])

    def stationary_covariance(self) -> JAXArray:
        P0 = self._P0()
        P1 = self._P1()
        z01 = jnp.zeros((P0.shape[0], P1.shape[1]))
        z10 = jnp.zeros((P1.shape[0], P0.shape[1]))
        return jnp.block([[P0, z01],
                          [z10, P1]])

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        t1, _ = X1
        t2, _ = X2
        dt = t2 - t1
        Phi0 = self._Phi0(dt)
        Phi1 = self._Phi1(dt)
        z01 = jnp.zeros((Phi0.shape[0], Phi1.shape[1]))
        z10 = jnp.zeros((Phi1.shape[0], Phi0.shape[1]))
        return jnp.block([[Phi0, z01],
                          [z10, Phi1]])

    def _quadrature_nodes_weights(self, width: JAXArray):
        """
        Return offsets u_k in [0, width] and mean-weights γ_k (sum=1) for the
        causal average (1/width) ∫_0^{width} f(u) du ≈ Σ γ_k f(u_k).
        """
        nodes = jnp.array([0.1127016653792583, 0.5, 0.8872983346207417])
        weights = jnp.array([5/18, 8/18, 5/18])  # sums to 1
        u = width * nodes
        gamma = weights  # already normalized to sum=1 for the mean
        return u, gamma

    def conv_observation_model(self, t: JAXArray, lag: JAXArray, width: JAXArray) -> JAXArray:
        """
        Causal top-hat average of the latent OU/DRW observed at time t:

            h_eff(t; lag, width)
            = (1/width) ∫_{u=0}^{width} h(t - lag - u) du
            ≈ Σ_k γ_k · h(t - lag - u_k) · Φ(+ (lag + u_k)),

        where:
        - h(·) = kernel.observation_model(·) is the row vector that maps the latent state
                at that (earlier) time into an observation.
        - Φ(+δ) = kernel.transition_matrix(0, δ) is the *forward* state transition over
                δ = lag + u ≥ 0, which for an OU gives exp(-δ/τ) attenuation.
        - {u_k, γ_k} are positive nodes/weights of a quadrature rule over [0, width].

        Returns
        -------
        A row vector in the base-state space (same shape as kernel.observation_model(t)).
        """

        # Quadrature nodes/weights over u ∈ [0, width]; both are ≥ 0 and Σ_k γ_k ≈ 1.
        u_k, gamma_k = self._quadrature_nodes_weights(width)

        def tap(u, g):
            # Row vector evaluated at the earlier time t - (lag + u)
            C = self.kernel.observation_model(t - (lag + u))
            # Forward transition over +δ so the contribution decays with delay (OU: e^{-δ/τ})
            Phi = self.kernel.transition_matrix(0.0, (lag + u))
            # Effective row at time t contributed by this tap
            return g * (C @ Phi)

        # Sum contributions of all taps; shapes: (K, m) -> (m,)
        taps = jax.vmap(tap, in_axes=(0, 0))(u_k, gamma_k)
        return taps.sum(axis=0)


    def observation_model(self, X: JAXArray) -> JAXArray:
        """
        Return a single observation vector h_tot for the augmented state:
            h_tot = [ h_base_total , h_bwb ].
        """
        t, b = X
        b = jnp.asarray(b, dtype=int)

        # Continuum at (t - lag_disk[b])
        h_cont0 = self.conv_observation_model(t, self.lag_disk[b], self.width_cont[b])
        # BLR lags behind continuum by lag_blr[b]  => total shift is lag_disk + lag_blr
        h_blr0  = self.conv_observation_model(t, self.lag_disk[b] + self.lag_blr[b], self.width_blr[b])

        h_cont = self.amp_cont[b] * h_cont0
        h_blr  = self.amp_blr[b]  * h_blr0
        h_base_total = h_cont + h_blr

        h_sq = self.kernel2.observation_model(t)
        q_b  = self.bwb_alpha * (self.amp_cont[b]) ** 2
        h_bwb = jnp.sqrt(2.0) * q_b * h_sq

        return jnp.concatenate([h_base_total, h_bwb], axis=0)

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
class MyMultiVarModel(MultiVarModel):
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
        log_tau_band        = self.my_tau_drw_transform(params)

        # DO NOT sort or reindex
        t, band = self.X
        
        s = ContiBLRQS.coord_to_sortable(self, (t, band))  # or kernel.coord_to_sortable((t, band))
        #jax.debug.print("sorted_by_kernel? {}", jnp.all(jnp.diff(s) >= 0))

        t_center = jnp.mean(t)
        t_std    = jnp.std(t)

        means = partial(MyMultiVarModel.mean_func, self.zero_mean, log_sigma_band.shape[0],
                        t_center, t_std, params)

        # diagonal noise in original order
        if self.has_jitter:
            diags = self.diag + (jnp.exp(params["log_jitter"]) ** 2)[band]
        else:
            diags = self.diag

        lag_disk = params["lag0"] * (self.lam_rf / 2500.0) ** params["lag_beta"]

        kernel = ContiBLRQS(
            amp_cont=jnp.exp(log_sigma_band),
            amp_blr=jnp.exp(log_sigma_band_blr),
            tau_drw=jnp.exp(log_tau_band),
            lag_disk=lag_disk,                              # NEW
            lag_blr=jnp.exp(params["log_lag_blr"]),
            bwb_alpha=params["bwb_alpha"],
            bwb_beta=params["bwb_beta"],
            width_cont=params["width_cont"],
            width_blr=params["width_blr"],
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
        return log_tau_band_mean

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
        gp, _ = self._build_gp(params)
        _, cond = gp.condition(self.y, X)   # no sorting
        return cond.loc, jnp.sqrt(cond.variance)

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