"""Shared-latent DHO with disk and generic delayed responses in every band.

All observed components are linear filters of one unit-RMS DHO realization.
Finite Erlang filter chains keep the augmented process Markovian and therefore
compatible with tinygp's quasi-separable solver.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from tinygp import GaussianProcess
from tinygp.kernels import quasisep as qs
from tinygp.solvers.quasisep.core import DiagQSM, StrictLowerTriQSM, SymmQSM

from qvc.light_curve.multiband_dho_core import (
    ContiBLRRelativeFlux_SHO_Model,
    OverdampedSHOBaseQS,
    _safe_pos,
    make_linear_mean_func,
)


DEFAULT_DISK_ORDER = 3


class SharedLatentDiskBLRQS(qs.Quasisep):
    """One DHO driver feeding narrow disk and broad delayed filter chains."""

    tau_fast: jnp.ndarray
    tau_slow: jnp.ndarray
    lag_disk: jnp.ndarray
    lag_blr: jnp.ndarray
    amp_cont: jnp.ndarray
    amp_blr: jnp.ndarray
    disk_order: int = eqx.field(static=True, default=DEFAULT_DISK_ORDER)
    blr_order: int = eqx.field(static=True, default=3)

    def coord_to_sortable(self, X):
        t, b = X
        return t + 1e-9 * jnp.asarray(b, dtype=jnp.int32)

    def _base(self):
        # The shared driver has one pair of poles, not one pair per band.
        return OverdampedSHOBaseQS(
            jnp.atleast_1d(self.tau_fast)[0:1],
            jnp.atleast_1d(self.tau_slow)[0:1],
        )

    def _response_specs(self):
        specs = []
        for band in range(int(jnp.asarray(self.lag_disk).shape[0])):
            specs.append((band, "disk", int(self.disk_order)))
        for band in range(int(jnp.asarray(self.lag_blr).shape[0])):
            specs.append((band, "blr", int(self.blr_order)))
        return tuple(specs)

    def _response_lags(self):
        return jnp.concatenate(
            [jnp.asarray(self.lag_disk, dtype=float), jnp.asarray(self.lag_blr, dtype=float)]
        )

    def _chain_slices(self):
        start = 2
        slices = []
        for _band, _kind, order in self._response_specs():
            slices.append(slice(start, start + order))
            start += order
        return tuple(slices), start

    @staticmethod
    def _chain_matrix(rate, order, dtype):
        state_rates = jnp.full(order, rate, dtype=dtype)
        block = -jnp.diag(state_rates)
        if order > 1:
            rows = jnp.arange(1, order, dtype=jnp.int32)
            block = block.at[rows, rows - 1].set(rate)
        return block

    def _subsystem_matrix(self, response_index):
        base = self._base()
        A0 = base.design_matrix()
        order = self._response_specs()[response_index][2]
        rate = order / _safe_pos(self._response_lags()[response_index])
        chain = self._chain_matrix(rate, order, A0.dtype)
        driver = jnp.zeros((order, 2), dtype=A0.dtype)
        driver = driver.at[0].set(rate * base.observation_model((0.0, 0)))
        return jnp.block(
            [[A0, jnp.zeros((2, order), dtype=A0.dtype)], [driver, chain]]
        )

    def design_matrix(self):
        A0 = self._base().design_matrix()
        slices, size = self._chain_slices()
        A = jnp.zeros((size, size), dtype=A0.dtype).at[:2, :2].set(A0)
        for response_index, chain_slice in enumerate(slices):
            subsystem = self._subsystem_matrix(response_index)
            order = subsystem.shape[0] - 2
            A = A.at[chain_slice, :2].set(subsystem[2:, :2])
            A = A.at[chain_slice, chain_slice].set(subsystem[2:, 2:])
        return A

    def _response_stds(self):
        covariance = self.stationary_covariance()
        slices, _size = self._chain_slices()
        endpoints = jnp.asarray([chain_slice.stop - 1 for chain_slice in slices])
        return jnp.sqrt(_safe_pos(covariance[endpoints, endpoints]))

    def stationary_covariance(self):
        """Solve ``A P + P A.T = -Q`` using A's triangular structure."""

        base = self._base()
        A = self.design_matrix()
        P0 = base.stationary_covariance()
        A0 = base.design_matrix()
        Q0 = -(A0 @ P0 + P0 @ A0.T)
        size = A.shape[0]
        Q = jnp.zeros_like(A).at[:2, :2].set(Q0)
        indices = jnp.arange(size)

        def update(flat_index, P):
            i = flat_index // size
            j = flat_index % size
            left = jnp.sum(
                jnp.where(indices < i, A[i, :] * P[:, j], 0.0)
            )
            right = jnp.sum(
                jnp.where(indices < j, P[i, :] * A[j, :], 0.0)
            )
            value = (-Q[i, j] - left - right) / (A[i, i] + A[j, j])
            return P.at[i, j].set(value)

        P = jax.lax.fori_loop(0, size * size, update, jnp.zeros_like(A))
        return 0.5 * (P + P.T)

    def _observation_model_with_stds(self, X, stds):
        _t, band = X
        band = jnp.asarray(band, dtype=jnp.int32)
        B = int(jnp.asarray(self.lag_disk).shape[0])
        slices, size = self._chain_slices()
        h = jnp.zeros(size, dtype=jnp.asarray(self.tau_fast).dtype)
        disk_index = band
        blr_index = B + band
        disk_endpoint = 2 + (band + 1) * int(self.disk_order) - 1
        blr_endpoint = 2 + B * int(self.disk_order) + (band + 1) * int(self.blr_order) - 1
        h = h.at[disk_endpoint].set(
            _safe_pos(jnp.asarray(self.amp_cont))[band] / stds[disk_index]
        )
        h = h.at[blr_endpoint].set(
            _safe_pos(jnp.asarray(self.amp_blr))[band] / stds[blr_index]
        )
        return h

    def observation_model(self, X):
        return self._observation_model_with_stds(X, self._response_stds())

    @staticmethod
    def _driver_to_chain_columns(dt, lam, e_lam, q, eq, order, sign):
        j = np.arange(1, order + 1)
        z = q - lam
        zt = z * dt
        use_series = jnp.abs(zt) <= 1.0
        p = np.arange(26)
        inv_fact_pj = jnp.asarray(
            [[1.0 / math.factorial(int(pi) + int(ji)) for pi in p] for ji in j]
        )
        zt_series = jnp.where(use_series, zt, 0.0)
        series = (
            eq
            * jnp.power(q * dt, j)
            * (jnp.power(zt_series, p) @ inv_fact_pj.T)
        )
        z_safe = jnp.where(use_series, 1.0, z)
        zt_safe = jnp.where(use_series, 1.0, zt)
        m = np.arange(order)
        partial = jnp.cumsum(
            jnp.power(zt_safe, m)
            * jnp.asarray([1.0 / math.factorial(int(v)) for v in m])
        )
        difference = jnp.power(q / z_safe, j) * (e_lam - eq * partial)
        return sign * jnp.where(use_series, series, difference)

    def transition_matrix(self, X1, X2):
        t1, _ = X1
        t2, _ = X2
        dt = t2 - t1
        base = self._base()
        tau_fast, tau_slow = base._ordered_taus()
        lam_fast, lam_slow = 1.0 / tau_fast[0], 1.0 / tau_slow[0]
        e_fast, e_slow = jnp.exp(-lam_fast * dt), jnp.exp(-lam_slow * dt)
        obs_scale = base._obs_scale()[0]
        slices, size = self._chain_slices()
        phi = jnp.zeros((size, size), dtype=e_fast.dtype)
        phi = phi.at[0, 0].set(e_fast)
        phi = phi.at[1, 1].set(e_slow)
        lags = self._response_lags()
        for response_index, chain_slice in enumerate(slices):
            order = self._response_specs()[response_index][2]
            q = order / _safe_pos(lags[response_index])
            eq = jnp.exp(-q * dt)
            d = np.subtract.outer(np.arange(order), np.arange(order))
            chain = eq * jnp.power(q * dt, np.maximum(d, 0)) * jnp.asarray(
                [[1.0 / math.factorial(int(v)) if v >= 0 else 0.0 for v in row] for row in d]
            )
            phi = phi.at[chain_slice, chain_slice].set(chain)
            col_fast = obs_scale * self._driver_to_chain_columns(
                dt, lam_fast, e_fast, q, eq, order, +1.0
            )
            col_slow = obs_scale * self._driver_to_chain_columns(
                dt, lam_slow, e_slow, q, eq, order, -1.0
            )
            phi = phi.at[chain_slice, 0].set(col_fast)
            phi = phi.at[chain_slice, 1].set(col_slow)
        return phi

    def to_symm_qsm(self, X):
        Pinf = self.stationary_covariance()
        slices, _size = self._chain_slices()
        endpoints = jnp.asarray([chain_slice.stop - 1 for chain_slice in slices])
        stds = jnp.sqrt(_safe_pos(Pinf[endpoints, endpoints]))
        Xprev = jax.tree_util.tree_map(lambda y: jnp.append(y[0], y[:-1]), X)
        transitions = jax.vmap(self.transition_matrix)(Xprev, X)
        h = jax.vmap(lambda Xi: self._observation_model_with_stds(Xi, stds))(X)
        diag = jnp.einsum("ni,ij,nj->n", h, Pinf, h)
        p = jax.vmap(lambda hi, Fi: hi @ Fi)(h, transitions)
        q = h @ Pinf.T
        return SymmQSM(
            diag=DiagQSM(d=diag),
            lower=StrictLowerTriQSM(p=p, q=q, a=transitions),
        )

    def evaluate(self, X1, X2):
        Pinf = self.stationary_covariance()
        slices, _size = self._chain_slices()
        endpoints = jnp.asarray([chain_slice.stop - 1 for chain_slice in slices])
        stds = jnp.sqrt(_safe_pos(Pinf[endpoints, endpoints]))
        h1 = self._observation_model_with_stds(X1, stds)
        h2 = self._observation_model_with_stds(X2, stds)
        return jnp.where(
            self.coord_to_sortable(X1) < self.coord_to_sortable(X2),
            h2 @ self.transition_matrix(X1, X2) @ Pinf @ h1,
            h1 @ self.transition_matrix(X2, X1) @ Pinf @ h2,
        )

    def effective_timescales(self):
        """Return exact integral-correlation timescales for all observed bands."""

        A = self.design_matrix()
        Pinf = self.stationary_covariance()
        slices, _size = self._chain_slices()
        endpoints = jnp.asarray([chain_slice.stop - 1 for chain_slice in slices])
        stds = jnp.sqrt(_safe_pos(Pinf[endpoints, endpoints]))
        B = int(jnp.asarray(self.lag_disk).shape[0])
        h = jax.vmap(
            lambda band: self._observation_model_with_stds((0.0, band), stds)
        )(jnp.arange(B, dtype=jnp.int32))
        variance = jnp.einsum("bi,ij,bj->b", h, Pinf, h)
        integrated = jax.vmap(
            lambda hb: -hb @ jnp.linalg.solve(A, Pinf @ hb)
        )(h)
        return integrated / _safe_pos(variance)

    def stationary_rms(self):
        """Return the total stationary RMS of each observed band."""

        Pinf = self.stationary_covariance()
        slices, _size = self._chain_slices()
        endpoints = jnp.asarray([chain_slice.stop - 1 for chain_slice in slices])
        stds = jnp.sqrt(_safe_pos(Pinf[endpoints, endpoints]))
        B = int(jnp.asarray(self.lag_disk).shape[0])
        h = jax.vmap(
            lambda band: self._observation_model_with_stds((0.0, band), stds)
        )(jnp.arange(B, dtype=jnp.int32))
        variance = jnp.einsum("bi,ij,bj->b", h, Pinf, h)
        return jnp.sqrt(_safe_pos(variance))

    def continuum_effective_timescales(self):
        """Return exact integral timescales for the disk continua alone."""

        A = self.design_matrix()
        Pinf = self.stationary_covariance()
        slices, size = self._chain_slices()
        endpoints = jnp.asarray([chain_slice.stop - 1 for chain_slice in slices])
        stds = jnp.sqrt(_safe_pos(Pinf[endpoints, endpoints]))
        B = int(jnp.asarray(self.lag_disk).shape[0])

        def disk_observation(band):
            endpoint = 2 + (band + 1) * int(self.disk_order) - 1
            h = jnp.zeros(size, dtype=jnp.asarray(self.tau_fast).dtype)
            return h.at[endpoint].set(1.0 / stds[band])

        h = jax.vmap(disk_observation)(jnp.arange(B, dtype=jnp.int32))
        variance = jnp.einsum("bi,ij,bj->b", h, Pinf, h)
        integrated = jax.vmap(
            lambda hb: -hb @ jnp.linalg.solve(A, Pinf @ hb)
        )(h)
        return integrated / _safe_pos(variance)


def continuum_effective_timescale(tau_fast, tau_slow, lag_disk, *, disk_order):
    """Exact integral timescale for one continuum-only Erlang response."""

    lag_disk = jnp.atleast_1d(jnp.asarray(lag_disk, dtype=float))
    kernel = SharedLatentDiskBLRQS(
        tau_fast=jnp.atleast_1d(jnp.asarray(tau_fast, dtype=float)),
        tau_slow=jnp.atleast_1d(jnp.asarray(tau_slow, dtype=float)),
        lag_disk=lag_disk,
        lag_blr=jnp.ones_like(lag_disk),
        amp_cont=jnp.ones_like(lag_disk),
        amp_blr=jnp.zeros_like(lag_disk),
        disk_order=int(disk_order),
        blr_order=1,
    )
    return kernel.continuum_effective_timescales()[0]


class SharedLatentDiskBLRRelativeFluxModel(ContiBLRRelativeFlux_SHO_Model):
    """Relative-flux likelihood for the shared latent response kernel."""

    disk_order: int
    blr_order: int

    def __init__(self, *args, disk_order=DEFAULT_DISK_ORDER, blr_order=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.disk_order = int(disk_order)
        self.blr_order = int(blr_order)

    def _build_kernel(self, params):
        return SharedLatentDiskBLRQS(
            tau_fast=jnp.atleast_1d(params["tau_fast_driver"]),
            tau_slow=jnp.atleast_1d(params["tau_slow_driver"]),
            lag_disk=jnp.asarray(params["lag_disk"]),
            lag_blr=jnp.asarray(params["lag_blr"]),
            amp_cont=jnp.asarray(params["amp_cont_relflux"]),
            amp_blr=jnp.asarray(params["amp_blr_relflux"]),
            disk_order=self.disk_order,
            blr_order=self.blr_order,
        )

    def _build_gp(self, params):
        means = lambda X: self.get_mean(self.zero_mean, params, X)
        t, band = self.X
        inds = jnp.argsort(t)
        diag = self.diag
        if self.has_jitter:
            diag = diag + self._jitter_diag(params, band)
        return (
            GaussianProcess(
                self._build_kernel(params),
                (t[inds], band[inds]),
                diag=diag[inds],
                mean=means,
                assume_sorted=True,
            ),
            inds,
        )


def make_multiband_shared_latent_blr_model(
    X,
    y,
    yerr,
    n_band=None,
    *,
    survey_idx=None,
    seeing_covariate=None,
    baseline_flux_by_band=None,
    zero_mean=False,
    has_jitter=True,
    disk_order=DEFAULT_DISK_ORDER,
    blr_order=3,
):
    """Construct the shared-driver disk plus delayed-response model."""

    del baseline_flux_by_band
    if n_band is None:
        n_band = int(jnp.max(jnp.asarray(X[1], dtype=jnp.int32))) + 1
    t = jnp.asarray(X[0])
    return SharedLatentDiskBLRRelativeFluxModel(
        X,
        y,
        yerr,
        base_kernel=OverdampedSHOBaseQS(jnp.array([10.0]), jnp.array([100.0])),
        nBand=n_band,
        mean_func=make_linear_mean_func(t, zero_mean=zero_mean),
        survey_idx=survey_idx,
        seeing_covariate=seeing_covariate,
        zero_mean=zero_mean,
        has_jitter=has_jitter,
        has_lag=False,
        disk_order=disk_order,
        blr_order=blr_order,
    )


__all__ = [
    "DEFAULT_DISK_ORDER",
    "SharedLatentDiskBLRQS",
    "continuum_effective_timescale",
    "SharedLatentDiskBLRRelativeFluxModel",
    "make_multiband_shared_latent_blr_model",
]
