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
    tau_drw: jnp.ndarray
    tau_drw_blr: float
    w: jnp.ndarray

    def __init__(self, amplitudes, amplitudes_blr, lag_blr, taus, tau_drw_blr, log_w=1) -> None:
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

    def ke(self, tau, tau_d) -> JAXArray:
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
    amplitudes: jnp.ndarray
    tau_drw: jnp.ndarray
    w: jnp.ndarray

    def __init__(self, amplitudes, taus, log_w=1) -> None:
        self.tau_drw = taus
        self.amplitudes = amplitudes
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

        sigma_b1 = jnp.sqrt( (self.amplitudes[b1])**2 * self.tau_drw[b1] )
        sigma_b2 = jnp.sqrt( (self.amplitudes[b2])**2 * self.tau_drw[b2] )

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
        zero_mean: bool, nBand: int, params: dict[str, JAXArray], X: JAXArray
    ) -> JAXArray:

        band_idx = X[1]

        if zero_mean is True:
            mean_per_obs = jnp.zeros(nBand)[band_idx]
        else:
            time_centered = (X[0] - params['t_center'])
            time_scaled = time_centered/params['t_std']
            #coeffs = jnp.stack([params["poly1"], params["mean"][band_idx]])
            #mean_per_obs = jnp.polyval(coeffs, time_scaled)
            mean_per_obs = params["poly1"] * time_scaled + params["mean"][band_idx]

        return mean_per_obs
    
    def _build_gp(
        self, params: dict[str, JAXArray]
    ) -> tuple[GaussianProcess, JAXArray]:
        # log amp + mean
        log_sigma_hat_band = self.my_amp_transform(params)
        log_sigma_hat_band_blr = self.my_amp_transform_blr(params)
        log_tau_band_rf = self.my_tau_drw_transform(params)

        # time axis transform: t and band are not sorted,
        # inds gives the sorted indices for the new_t
        X, inds = self.my_lag_transform(self.X, self.has_lag, params)

        params_mean = dict(params)
        params_mean['t_center'] = jnp.mean(X[0])
        params_mean['t_std'] = jnp.std(X[0])

        means = partial(
            MyMultiVarModel.mean_func, self.zero_mean, log_sigma_hat_band.shape[0], params_mean
        )

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
            tau_drw_blr=jnp.exp(params["log_tau_drw0"] - jnp.log(1 + self.z)), # Assume DRW tau is 2500AA
            #log_w=params["log_w"] - jnp.log(1 + self.z),
            lag_blr=jnp.zeros_like(log_sigma_hat_band_blr),
            log_w=0
        )

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
        return params["log_sigma_hat0"] + jnp.atleast_1d(params["log_amp_delta_blr"])
    
    def my_tau_drw_transform(self, params: dict[str, JAXArray]) -> JAXArray:
        eta_tau1 = params["eta_tau1"]
        eta_tau2 = params["eta_tau2"]
        lam_s = 2500
        eta_break = 1.0
        log_tau_band_RF = params["log_tau_drw0"] - jnp.log(1 + self.z) + jnp.log(10) * log_broken_pl(self.lam_rf, lam_s, eta_tau1, eta_tau2, eta_break)
        return log_tau_band_RF

    def my_amp_transform(self, params: dict[str, JAXArray]) -> JAXArray:
        eta_A1 = params["eta_A1"]
        eta_A2 = params["eta_A2"]
        lam_s = 2500
        eta_break = 1.0

        # Host dilution: apply per-band correction
        # Host galaxy contribution modeled as a power-law in wavelength
        host_frac = params["f_host"] * (self.lam_rf / 5100.0) ** params["alpha_host"]
        dilution_factor = 1.0 / (1.0 + host_frac)
        log_dilution = jnp.log(dilution_factor)

        # Power-law scaling across rest-frame wavelength
        log_sigma_hat_band = params["log_sigma_hat0"] + log_dilution + jnp.log(10) * log_broken_pl(self.lam_rf, lam_s, eta_A1, eta_A2, eta_break)

        return log_sigma_hat_band
    
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

    def sample(self, params: dict[str, JAXArray], i) -> None:
            """A convience function for integrating with numpyro for MCMC sampling.

            Args:
                params (dict[str, JAXArray]): Model parameters.
            """

            X, inds = self.my_lag_transform(self.X, self.has_lag, params)
            gp, inds = self._build_gp(params)

            f = numpyro.sample(f"gp_{i}", gp.numpyro_dist())

            # Compute s_b = bwb_A * log(lambda_b / 2500 Å)
            s_b = params["bwb_A"] * jnp.log(self.lam_rf / 2500.0)
            s_per_obs = s_b[X[1][inds]]

            # Model mean with BWB nonlinear effect
            mean = f + 0. * f

            numpyro.sample(f"obs_{i}", dist.Normal(mean), obs=self.y[inds])

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


class MyMultiVarModelLatent(MyMultiVarModel):

    def _get_unique_times(self, t, band, lag_blr):
        # Step 1: build full latent time grid (t, t - lag_blr)
        t_direct = t
        t_lagged = t - lag_blr

        t_latent = jnp.concatenate([t_direct, t_lagged])
        band_latent = jnp.concatenate([band, band])

        # Step 2: sort the combined time array (to make GP matrix construction more stable)
        sort_idx = jnp.argsort(t_latent)
        t_latent_sorted = t_latent[sort_idx]
        band_latent_sorted = band_latent[sort_idx]

        # Step 3: construct index maps back to sorted array
        def find_index(t_query, t_sorted):
            return jnp.argmin(jnp.abs(t_query[:, None] - t_sorted[None, :]), axis=1)

        inv_direct = find_index(t_direct, t_latent_sorted)
        inv_lagged = find_index(t_lagged, t_latent_sorted)

        return t_latent_sorted, band_latent_sorted, inv_direct, inv_lagged

    def _build_H(self, t, band, inv_direct, inv_lagged, A_c, A_b, M):
        # Observation operator H: shape (N, M)
        N = len(t)
        rows = jnp.arange(N)
        H = jnp.zeros((N, M))

        # Add direct term: A_c[band] * f(t)
        H = H.at[rows, inv_direct].add(A_c[band])

        # Add lagged term: A_b[band] * f(t - tau[band])
        H = H.at[rows, inv_lagged].add(A_b[band])
        return H

    def _build_latent_model(self, params: dict[str, JAXArray]) -> tuple[GaussianProcess, JAXArray]:
        log_amps = self.my_amp_transform(params)
        log_amps_blr = self.my_amp_transform_blr(params)
        log_taus = self.my_tau_drw_transform(params)

        X, inds = self.my_lag_transform(self.X, self.has_lag, params)
        t_obs, band_obs = X
        t_obs = t_obs[inds]
        band_obs = band_obs[inds]
        y_obs = self.y[inds]

        amp_conti = jnp.exp(log_amps)
        amp_blr = jnp.exp(log_amps_blr)
        lag_blr = jnp.exp(params["log_lag_blr"] - jnp.log(1 + self.z))[band_obs]

        # Noise (diagonal)
        noise_diag = self.diag[inds]
        if self.has_jitter:
            noise_diag += (jnp.exp(params["log_jitter"]) ** 2)[band_obs]

        # Latent lag transformation
        t_latent, band_latent, inv_d, inv_l = self._get_unique_times(t_obs, band_obs, lag_blr)
        M = len(t_latent)

        # Construct latent-space kernel
        kernel = MyMultibandConti(
            amplitudes=jnp.ones_like(amp_conti),
            taus=jnp.exp(log_taus),
        )
        # Assume BLR is scaled + lagged version of continuum in each band
        X_latent = (t_latent, band_latent)
        K_latent = kernel(X_latent, X_latent) + 1e-6 * jnp.eye(M)

        params_mean_latent = params
        params_mean_latent['t_center'] = jnp.mean(t_latent)
        params_mean_latent['t_std'] = jnp.std(t_latent)

        means_latent = partial(MyMultiVarModelLatent.mean_func, self.zero_mean, amp_conti.shape[0], params_mean_latent)

        # Construct observation operator H
        H = self._build_H(t_obs, band_obs, inv_d, inv_l, amp_conti, amp_blr, M)

        # Mean function
        params_mean = params
        params_mean['t_center'] = jnp.mean(t_obs)
        params_mean['t_std'] = jnp.std(t_obs)

        means = partial(MyMultiVarModelLatent.mean_func, self.zero_mean, amp_conti.shape[0], params_mean)
        mu_obs = means((t_obs, band_obs))

        # Build GP
        gp = GaussianProcess(
                kernel,
                X_latent,
                diag=1e-6 * jnp.ones_like(t_latent),
                mean=means_latent)

        return gp, H, K_latent, noise_diag, mu_obs, y_obs

    @eqx.filter_jit
    def log_prob(self, params: dict[str, JAXArray]) -> JAXArray:
        gp, H, K_latent, D, mu_obs, y_obs = self._build_latent_model(params)

        # Build full covariance in observation space: K_obs = H K_latent H.T + D
        K_obs = H @ K_latent @ H.T
        K_obs += jnp.diag(D)  # add diagonal noise

        # Check if K_obs is symmetric
        #is_symmetric = jnp.allclose(K_obs, K_obs.T, atol=1e-6)
        #jax.debug.print("K_obs symmetric: {sym}", sym=is_symmetric)
        # Check if K_obs is positive semi-definite
        #eigvals = jnp.linalg.eigvalsh(K_obs)
        #is_psd = jnp.all(eigvals >= -1e-8)
        #jax.debug.print("K_obs PSD: {x}", x=is_psd)

        # Subtract mean
        y_centered = y_obs - mu_obs

        # Cholesky solve for log-likelihood
        L = jnp.linalg.cholesky(K_obs + 1e-6 * jnp.eye(K_obs.shape[0]))  # small jitter for stability
        alpha = jax.scipy.linalg.cho_solve((L, True), y_centered)

        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        log_likelihood = -0.5 * (jnp.dot(y_centered, alpha) + logdet + y_obs.size * jnp.log(2 * jnp.pi))

        log_likelihood = jnp.where(jnp.isfinite(log_likelihood), log_likelihood, -jnp.inf)
        #jax.debug.print("log_likelihood: {x}", x=log_likelihood)   
        return log_likelihood

    def sample(self, params: dict[str, JAXArray], i) -> None:
        """A convience function for intergrating with numpyro for MCMC sampling.

        Args:
            params (dict[str, JAXArray]): Model parameters.
        """
        gp, H, K_latent, D, mu_obs, y_obs = self._build_latent_model(params)
        # Latent GP
        f_latent = numpyro.sample(f"gp_{i}", gp.numpyro_dist())

        # Compute s_b = bwb_A * log(lambda_b / 2500 Å)
        s_b = jnp.array([
            params["bwb_A"] * jnp.log(lambda_pivot[band] / 2500.0)
            for band in self.clean_bands
        ])
        s_per_obs = s_b[self.X[1]]

        # Model mean with BWB nonlinear effect
        mean = H @ f_latent + s_per_obs * (H @ f_latent) ** 2

        std = jnp.sqrt(jnp.clip(D, a_min=1e-6))
        numpyro.sample(f"obs_{i}", dist.Normal(mean, std), obs=y_obs)

    @eqx.filter_jit
    def pred(self, params: dict[str, JAXArray], X_new: JAXArray) -> tuple[JAXArray, JAXArray, JAXArray, JAXArray, JAXArray, JAXArray]:
        # Build latent model from training data
        gp, H_train, K_latent, D, mu_obs, y_obs = self._build_latent_model(params)

        # Observation covariance and Cholesky factor for training data
        K_obs = H_train @ K_latent @ H_train.T + jnp.diag(D)
        y_centered = y_obs - mu_obs
        L = jnp.linalg.cholesky(K_obs + 1e-6 * jnp.eye(K_obs.shape[0]))
        alpha = jax.scipy.linalg.cho_solve((L, True), y_centered)

        # Prepare new latent inputs (times, bands, lags)
        X_new_lagged, _ = self.my_lag_transform(X_new, self.has_lag, params)
        t_new, band_new = X_new_lagged
        lag_blr = jnp.exp(params["log_lag_blr"])[band_new]

        log_amps = self.my_amp_transform(params)
        log_amps_blr = self.my_amp_transform_blr(params)
        log_taus = self.my_tau_drw_transform(params)
        amp_conti = jnp.exp(log_amps)
        amp_blr = jnp.exp(log_amps_blr)

        # Build latent kernel and latent inputs for prediction
        kernel = MyMultibandConti(
            amplitudes=jnp.ones_like(amp_conti),
            taus=jnp.exp(log_taus),
        )
        t_latent, band_latent, _, _ = self._get_unique_times(
            self.X[0], self.X[1], jnp.exp(params["log_lag_blr"])[self.X[1]]
        )
        X_latent = (t_latent, band_latent)

        t_new_latent, band_new_latent, inv_d_new, inv_l_new = self._get_unique_times(t_new, band_new, lag_blr)
        X_new_latent = (t_new_latent, band_new_latent)

        # Helper to build H matrix given amplitudes (A_c, A_b)
        def build_H_for_amp(A_c, A_b):
            return self._build_H(t_new, band_new, inv_d_new, inv_l_new, A_c, A_b, len(t_new_latent))

        # Full H_new with all amps
        H_new_full = build_H_for_amp(amp_conti, amp_blr)

        # Predictive cross-covariance and covariance
        K_cross = H_new_full @ kernel(X_new_latent, X_latent) @ H_train.T
        K_new = H_new_full @ kernel(X_new_latent, X_new_latent) @ H_new_full.T + 1e-6 * jnp.eye(H_new_full.shape[0])

        # Predictive mean and covariance in observation space
        mean_fn = partial(MyMultiVarModelLatent.mean_func, self.zero_mean, amp_conti.shape[0], params)
        mu_new_obs = mean_fn((t_new, band_new))

        # Continuum-only prediction: zero BLR amps
        H_new_cont = build_H_for_amp(amp_conti, jnp.zeros_like(amp_blr))
        K_cross_cont = H_new_cont @ kernel(X_new_latent, X_latent) @ H_train.T
        K_new_cont = H_new_cont @ kernel(X_new_latent, X_new_latent) @ H_new_cont.T + 1e-6 * jnp.eye(H_new_cont.shape[0])

        pred_mean_cont = mu_new_obs + K_cross_cont @ alpha
        pred_cov_cont = K_new_cont - K_cross_cont @ jax.scipy.linalg.cho_solve((L, True), K_cross_cont.T)
        pred_cov_cont = 0.5 * (pred_cov_cont + pred_cov_cont.T)
        pred_std_cont = jnp.sqrt(jnp.clip(jnp.diag(pred_cov_cont), a_min=0.0))

        # BLR-only prediction: zero continuum amps
        H_new_blr = build_H_for_amp(jnp.zeros_like(amp_conti), amp_blr)
        K_cross_blr = H_new_blr @ kernel(X_new_latent, X_latent) @ H_train.T
        K_new_blr = H_new_blr @ kernel(X_new_latent, X_new_latent) @ H_new_blr.T + 1e-6 * jnp.eye(H_new_blr.shape[0])

        pred_mean_blr = mu_new_obs + K_cross_blr @ alpha
        pred_cov_blr = K_new_blr - K_cross_blr @ jax.scipy.linalg.cho_solve((L, True), K_cross_blr.T)
        pred_cov_blr = 0.5 * (pred_cov_blr + pred_cov_blr.T)
        pred_std_blr = jnp.sqrt(jnp.clip(jnp.diag(pred_cov_blr), a_min=0.0))

        # Debug prints
        #jax.debug.print("pred_cov min eigenvalue: {x}", x=jnp.min(jnp.linalg.eigvalsh(pred_cov)))
        #jax.debug.print("any nan in pred_cov diag: {x}", x=jnp.any(jnp.isnan(jnp.diag(pred_cov))))

        return (pred_mean_cont + pred_mean_blr, jnp.sqrt(pred_std_cont**2 + pred_std_blr**2),
                pred_mean_cont, pred_std_cont,
                pred_mean_blr, pred_std_blr)