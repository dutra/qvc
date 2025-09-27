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
import jax.nn as jnn  # NEW: for one_hot
from typing import Sequence
import tinygp
from tinygp.kernels import quasisep as qs

JAXArray = jnp.ndarray

def inv_softplus(y):
    # numerically stable inverse softplus
    return jnp.where(y > 20.0, y, jnp.log(jnp.expm1(y)))

def softmin_logits(dist2, log_temp):
    """
    Convert squared distances to *soft* responsibilities with temperature.
    We add a tiny epsilon floor to avoid one-hot collapse.
    """
    temp = jnp.exp(log_temp)              # temperature in distance space
    logits = -dist2 / jnp.clip(temp, 1e-6, None)
    # stable softmax with a small floor to reduce brittleness
    resp = jax.nn.softmax(logits, axis=-1)
    eps = 1e-3
    resp = resp * (1.0 - resp.shape[-1] * eps) + eps
    resp = resp / resp.sum(axis=-1, keepdims=True)
    return resp

class ContiBLR_LMC_QS(qs.Wrapper):
    """
    QS Linear Model of Coregionalization (LMC) for multiband AGN light curves.

    Latents:
      - Q shared DRW latents k_q(τ; τ_q) (drive continuum + BLR across all bands).
      - Optional Q BWB-surrogate latents with τ_q / bwb_beta (instantaneous, no lag).

    Observation row for band b at time t is the concat over latent blocks:
      For each shared latent q:
          h_q(b,t) = a_cont[b,q] * H_q(t; lag_disk[b],            width_cont[b])
                   + a_blr [b,q] * H_q(t; lag_disk[b]+lag_blr[b], width_blr [b])
      For each BWB latent q (optional):
          h_q^bwb(b,t) = √2 * q_bwb[b,q] * h_q^bwb_raw(t)

    where H_q is the causal top-hat average of the latent OU observed at time t:
        H_q(t; lag, width) ≈ Σ_k γ_k · h_q(t - lag - u_k) · Φ_q(+ lag + u_k)

    Shapes
    ------
    B: #bands, Q: #latents
    tau_latents : (Q,)
    a_cont, a_blr, q_bwb : (B, Q)
    lag_disk, lag_blr, width_cont, width_blr : (B,)
    """

    tau_latents: jnp.ndarray          # (Q,)
    a_cont: jnp.ndarray               # (B,Q)
    a_blr: jnp.ndarray                # (B,Q)
    lag_disk: jnp.ndarray             # (B,)
    lag_blr: jnp.ndarray              # (B,)
    width_cont: jnp.ndarray           # (B,)
    width_blr: jnp.ndarray            # (B,)

    use_bwb: bool
    bwb_beta: float
    q_bwb: jnp.ndarray | None         # (B,Q) or None

    kernels_q: Sequence[qs.Kernel]            # Q shared DRW latents
    kernels_bwb: Sequence[qs.Kernel] | None   # Q BWB latents (optional)

    def __init__(
        self,
        tau_latents: jnp.ndarray,        # (Q,)
        a_cont: jnp.ndarray,             # (B,Q)
        a_blr: jnp.ndarray,              # (B,Q)
        lag_disk: jnp.ndarray,           # (B,)
        lag_blr: jnp.ndarray,            # (B,)
        width_cont: jnp.ndarray,         # (B,)
        width_blr: jnp.ndarray,          # (B,)
        *,
        use_bwb: bool = False,
        bwb_beta: float = 2.0,
        q_bwb: jnp.ndarray | None = None,   # (B,Q) if use_bwb
    ):
        # store tensors
        self.tau_latents  = tau_latents
        self.a_cont       = a_cont
        self.a_blr        = a_blr
        self.lag_disk     = lag_disk
        self.lag_blr      = lag_blr
        self.width_cont   = width_cont
        self.width_blr    = width_blr
        self.use_bwb      = bool(use_bwb)
        self.bwb_beta     = bwb_beta
        self.q_bwb        = q_bwb

        # shared DRW latents
        Q = int(tau_latents.shape[0])
        self.kernels_q = tuple(qs.Exp(scale=tau_latents[q], sigma=1.0) for q in range(Q))

        # Satisfy qs.Wrapper's required dataclass field:
        self.kernel = self.kernels_q[0]
        
        # optional BWB latents (same Q, faster timescale)
        if self.use_bwb:
            if q_bwb is None:
                raise ValueError("q_bwb (B,Q) must be provided when use_bwb=True.")
            self.kernels_bwb = tuple(qs.Exp(scale=tau_latents[q] / self.bwb_beta, sigma=1.0) for q in range(Q))
        else:
            self.kernels_bwb = None

    # ---- QS plumbing helpers ----
    def coord_to_sortable(self, X) -> JAXArray:
        t, b = X
        return t + jnp.asarray(b, t.dtype) * (10.0 * jnp.finfo(t.dtype).eps)

    @staticmethod
    def _hstack(mats):         # for design matrices
        return jnp.concatenate(mats, axis=1)

    @staticmethod
    def _block_diag(mats):     # for P (stationary cov) and Φ (transition)
        sizes = [Mi.shape[0] for Mi in mats]
        m = sum(sizes)
        out = jnp.zeros((m, m), dtype=mats[0].dtype)
        off = 0
        for Mi in mats:
            s = Mi.shape[0]
            out = out.at[off:off+s, off:off+s].set(Mi)
            off += s
        return out

    def _quadrature_nodes_weights(self, width: JAXArray):
        nodes   = jnp.array([0.1127016653792583, 0.5, 0.8872983346207417], dtype=width.dtype)
        weights = jnp.array([5/18, 8/18, 5/18], dtype=width.dtype)  # sum=1
        return width * nodes, weights

    def _conv_obs_with(self, kernel: qs.Kernel, t: JAXArray, lag: JAXArray, width: JAXArray) -> JAXArray:
        # Causal top-hat average of kernel.observation_model evaluated at (t - lag - u),
        # attenuated by forward transition over + (lag + u).
        u_k, gamma_k = self._quadrature_nodes_weights(width)
        def tap(u, g):
            C   = kernel.observation_model(t - (lag + u))
            Phi = kernel.transition_matrix(0.0, (lag + u))
            return g * (C @ Phi)
        return jax.vmap(tap, in_axes=(0, 0))(u_k, gamma_k).sum(axis=0)

    # ---- QS state-space pieces (concat over latent blocks) ----
    def design_matrix(self) -> JAXArray:
        mats = [k.design_matrix() for k in self.kernels_q]
        if self.kernels_bwb is not None:
            mats += [k.design_matrix() for k in self.kernels_bwb]
        return self._block_diag(mats)  # <- was _hstack

    def stationary_covariance(self) -> JAXArray:
        mats = [k.stationary_covariance() for k in self.kernels_q]
        if self.kernels_bwb is not None:
            mats += [k.stationary_covariance() for k in self.kernels_bwb]
        return self._block_diag(mats)

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        t1, _ = X1
        t2, _ = X2
        dt = t2 - t1
        mats = [k.transition_matrix(0.0, dt) for k in self.kernels_q]
        if self.kernels_bwb is not None:
            mats += [k.transition_matrix(0.0, dt) for k in self.kernels_bwb]
        return self._block_diag(mats)

    # ---- Observation row at (t, b) ----
    def observation_model(self, X: JAXArray) -> JAXArray:
        t, b = X
        b = jnp.asarray(b, dtype=jnp.int32)

        # For each shared latent q, build a row for its block:
        #   h_q = a_cont[b,q] * H_q(disk) + a_blr[b,q] * H_q(BLR)
        rows_q = []
        B, Q = self.a_cont.shape
        for q in range(Q):
            kq = self.kernels_q[q]
            h_cont = self._conv_obs_with(kq, t, self.lag_disk[b],                    self.width_cont[b])
            h_blr  = self._conv_obs_with(kq, t, self.lag_disk[b] + self.lag_blr[b],  self.width_blr[b])
            hq = self.a_cont[b, q] * h_cont + self.a_blr[b, q] * h_blr   # shape (m_q,)
            rows_q.append(hq)

        # Optional BWB surrogate rows (instantaneous, no lag/width)
        rows_bwb = []
        if self.kernels_bwb is not None:
            for q in range(Q):
                kb = self.kernels_bwb[q]
                h_raw = kb.observation_model(t)                       # shape (m_q,)
                rows_bwb.append(jnp.sqrt(2.0) * self.q_bwb[b, q] * h_raw)

        # Concatenate all latent blocks into one long row
        if rows_bwb:
            return jnp.concatenate(rows_q + rows_bwb, axis=0)
        else:
            return jnp.concatenate(rows_q, axis=0)
        
    def psd(self, omega: jnp.ndarray, b: int, sigma_n2: float = 0.0) -> jnp.ndarray:
        # State-space PSD (same approach as in the previous kernel)
        A = self.design_matrix()
        P = self.stationary_covariance()
        Qc = -(A @ P + P @ A.T)

        # Observation at t=0 for band b
        h = self.observation_model((jnp.array(0.0), jnp.array(int(b))))

        I = jnp.eye(A.shape[0], dtype=A.dtype)

        def one_w(w):
            v = jnp.linalg.solve((-1j * w) * I - A.T, h)
            return (v.conj().T @ (Qc @ v)).real + sigma_n2

        return 2.0 * jax.vmap(one_w)(omega)

class MyMultiVarModel_BLR_LMC(MultiVarModel):
    yerr: JAXArray | NDArray
    z: float
    lam_rf: JAXArray
    use_bwb: bool
    q_groups: int

    def __init__(
        self,
        X: JAXArray,
        y: JAXArray | NDArray,
        yerr: JAXArray | NDArray,
        kernel: tinygp.kernels.quasisep.Quasisep,
        
        **kwargs,
    ) -> None:
        # Sort
        ind_sort = jnp.argsort(X[0])
        super().__init__((X[0][ind_sort], X[1][ind_sort]), y[ind_sort], yerr[ind_sort], kernel, **kwargs)
        self.yerr = yerr
        self.z = kwargs["z"]
        self.lam_rf = kwargs["lam_rf"]
        self.use_bwb = kwargs["use_bwb"]
        self.q_groups = kwargs["q_groups"] # 1, 2 or 3

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
        log_sigma_band     = self.my_amp_transform(params)      # (B,)
        log_sigma_band_blr = self.my_amp_transform_blr(params)  # (B,)

        # NEW: get centers (Q,) and gate (B,Q) from the transform
        centers_log_tau, gate = self.my_tau_drw_transform(params)

        ind_sort = jnp.argsort(centers_log_tau)

        # basic plumbing (unchanged)
        t, band = self.X

        t_center = jnp.mean(t); t_std = jnp.std(t)
        means = partial(MyMultiVarModel_BLR_LMC.mean_func, self.zero_mean,
                        log_sigma_band.shape[0], t_center, t_std, params)
        diags = self.diag + (jnp.exp(params["log_jitter"]) ** 2)[band] if self.has_jitter else self.diag

        # per-band disk lags (unchanged)
        lag_disk = params["lag0"] * (self.lam_rf / 2500.0) ** params["lag_beta"]

        B = log_sigma_band.size
        Q = centers_log_tau.shape[0]

        # Loadings: diagonal per-band amplitude × gate
        a_cont = jnp.exp(log_sigma_band)[:, None]     * gate   # (B,Q)
        a_blr  = jnp.exp(log_sigma_band_blr)[:, None] * gate   # (B,Q)

        # Latent timescales
        tau_latents = jnp.exp(centers_log_tau)                # (Q,)

        # Optional BWB term (use same gate)
        use_bwb = bool(self.use_bwb)
        if use_bwb:
            base  = params["bwb_alpha"] * jnp.exp(2.0 * log_sigma_band)  # (B,)
            q_bwb = base[:, None] * gate                                 # (B,Q)  ← use the same gate
        else:
            q_bwb = None

        kernel = ContiBLR_LMC_QS(
            tau_latents=tau_latents,
            a_cont=a_cont, a_blr=a_blr,
            lag_disk=lag_disk,
            lag_blr=jnp.exp(params["log_lag_blr"]),
            width_cont=params["width_cont"],
            width_blr=params["width_blr"],
            use_bwb=use_bwb, bwb_beta=params["bwb_beta"], q_bwb=q_bwb,
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

    def my_tau_drw_transform(self, params: dict[str, JAXArray]) -> tuple[JAXArray, JAXArray]:
        lam_rf = self.lam_rf.astype(jnp.float64)        # (B,)
        B = lam_rf.size
        Q = int(self.q_groups) if self.q_groups else 1

        # λ-quantile centers (fixed)
        q_centers = (jnp.arange(Q, dtype=lam_rf.dtype) + 0.5) / Q
        lam_cent  = jnp.quantile(lam_rf, q_centers, method="linear")     # (Q,)
        log_lam   = jnp.log(lam_rf)[:, None]                             # (B,1)
        log_cent  = jnp.log(lam_cent)[None, :]                           # (1,Q)

        # bandwidth = fraction of bin width in log-λ space
        step = (jnp.max(log_cent) - jnp.min(log_cent)) / jnp.maximum(Q - 1, 1)
        sigma = 0.7 * jnp.maximum(step, 1e-6)

        # SOFT weights (B,Q), row-normalized
        logits = -0.5 * ((log_lam - log_cent) / sigma) ** 2
        gate   = jax.nn.softmax(logits, axis=1)                          # (B,Q)

        # Optional: scale rows to stabilize amplitude–τ leakage
        gate   = gate * jnp.sqrt(Q)

        # τ centers from your wavelength law (still fixed by λ-centers)
        log_tau0  = params["log_tau_drw0"]
        lam_s     = params["lam_s"]; eta_break = params["eta_break"]
        eta_tau1  = params["eta_tau1"]; eta_tau2 = params["eta_tau2"]
        centers_log_tau = (
            log_tau0 + jnp.log(10.0) * log_broken_pl(lam_cent, lam_s, eta_tau1, eta_tau2, eta_break)
        )  # (Q,)

        return centers_log_tau, gate

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
        host_frac = params["f_host"] * (self.lam_rf / 2500.0) ** (params["alpha_host"] - params["alpha_agn"])
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