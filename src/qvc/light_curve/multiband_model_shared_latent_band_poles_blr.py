"""Experimental common-forcing continuum with wavelength-dependent poles.

All rates are positive. Band poles may cross the common pole without sorting
or clipping: the cascaded state and its matrix exponential remain well defined.
"""

from __future__ import annotations

import math
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from tinygp import GaussianProcess
from tinygp.kernels import quasisep as qs
from tinygp.solvers.quasisep.core import DiagQSM, StrictLowerTriQSM, SymmQSM
from tinygp.solvers.quasisep.general import GeneralQSM
from qvc.light_curve.multiband_dho_core import (
    ContiBLRRelativeFlux_SHO_Model,
    OverdampedSHOBaseQS,
    make_linear_mean_func,
)

VARIANT = "shared_latent_band_poles_blr"
DEFAULT_DISK_ORDER = 3
DEFAULT_TRANSITION = "analytic"
TRANSITION_CHOICES = ("analytic", "expm")
ANALYTIC_LOCAL_EXPM_RTOL = 1e-3
TRANSITION_IMPLEMENTATIONS = {
    "analytic": "cascade_float64_analytic_local_expm_rtol_1e-3",
    "expm": "cascade_float64_expm_max_squarings_64",
}
MODEL_METADATA = {
    "tau_uv_definition": "continuum_only_2500A_disk_filtered_integral_rest_frame_days",
    "common_pole_definition": "tau_fast_driver_is_common_OU_forcing_pole_observer_days",
    "reference_pole_definition": "tau_slow_uv_driver_is_2500A_relaxation_pole_observer_days_not_integral_time",
    "band_poles_definition": "shared_common_pole_and_power_law_band_relaxation_poles",
    "blr_driver_definition": "2500A_reference_relaxation_state",
}


def transition_metadata(transition=DEFAULT_TRANSITION):
    if transition not in TRANSITION_CHOICES:
        raise ValueError(f"Unknown band-pole transition: {transition!r}")
    return {
        **MODEL_METADATA,
        "band_poles_transition": transition,
        "transition_implementation": TRANSITION_IMPLEMENTATIONS[transition],
    }


def saved_transition(metadata):
    """Historical band-pole files predate the selector and used expm."""
    name = metadata.get("band_poles_transition")
    if isinstance(name, bytes):
        name = name.decode()
    if name is None:
        implementation = metadata.get("transition_implementation")
        if isinstance(implementation, bytes):
            implementation = implementation.decode()
        if implementation is None:
            return "expm"
        reverse = {value: key for key, value in TRANSITION_IMPLEMENTATIONS.items()}
        if implementation not in reverse:
            raise ValueError(f"Unknown saved band-pole transition: {implementation!r}")
        name = reverse[implementation]
    transition_metadata(name)
    return name


def resolve_transition(requested=None, *, model_variant=VARIANT, saved_metadata=None):
    if model_variant != VARIANT:
        if requested is not None:
            raise ValueError(
                f"--band_poles_transition requires --model_variant {VARIANT}"
            )
        return None
    name = (
        requested
        if requested is not None
        else (
            saved_transition(saved_metadata)
            if saved_metadata is not None
            else DEFAULT_TRANSITION
        )
    )
    transition_metadata(name)
    return name


def run_transition_metadata(transition, *, saved_metadata=None):
    fit_backend = (
        transition if saved_metadata is None else saved_transition(saved_metadata)
    )
    return {
        **transition_metadata(fit_backend),
        "prediction_band_poles_transition": transition,
        "prediction_transition_implementation": TRANSITION_IMPLEMENTATIONS[transition],
    }


def _exp_power(x, powers):
    """exp(-x) x**powers without overflow or log(0) in inactive branches."""
    safe_x = jnp.where(x > 0, x, 1.0)
    values = jnp.exp(-x + powers * jnp.log(safe_x))
    return jnp.where((x == 0) & (powers > 0), 0.0, values)


def _driver_to_chain(dt, lam, q, order):
    """Response of each unit-DC-gain Erlang state to exp(-lam*t).

    The confluent Taylor series handles q == lam with its parameter
    derivatives. Inactive expressions receive safe inputs before evaluation.
    This helper is independent of the original SLB implementation.
    """
    j = jnp.arange(1, order + 1)
    p = jnp.arange(26)
    zt = (q - lam) * dt
    use_series = jnp.abs(zt) <= 1.0
    series_arg = jnp.where(use_series, zt, 0.0)
    coefficients = jnp.asarray(
        [[1.0 / math.factorial(n + k) for n in range(26)] for k in range(1, order + 1)]
    )
    series = _exp_power(q * dt, j) * (coefficients @ (series_arg**p))
    z_safe = jnp.where(use_series, 1.0, q - lam)
    zt_safe = jnp.where(use_series, 1.0, zt)
    m = jnp.arange(order)
    partial = jnp.cumsum(
        zt_safe**m * jnp.asarray([1.0 / math.factorial(n) for n in range(order)])
    )
    difference = (q / z_safe) ** j * (jnp.exp(-lam * dt) - jnp.exp(-q * dt) * partial)
    return jnp.where(use_series, series, difference)


class SharedLatentBandPolesBLRQS(qs.Quasisep):
    tau_fast: jax.Array
    tau_slow_band: jax.Array
    tau_slow_uv: jax.Array
    lag_disk: jax.Array
    lag_blr: jax.Array
    amp_cont: jax.Array
    amp_blr: jax.Array
    disk_order: int = eqx.field(static=True, default=3)
    blr_order: int = eqx.field(static=True, default=3)
    transition: str = eqx.field(static=True, default=DEFAULT_TRANSITION)

    def __post_init__(self):
        if self.transition not in TRANSITION_CHOICES:
            raise ValueError(f"Unknown band-pole transition: {self.transition!r}")

    def coord_to_sortable(self, X):
        return X[0]

    def _chain_slices(self):
        B = self.lag_disk.shape[0]
        start = B + 2
        slices = []
        for order in [self.disk_order] * B + [self.blr_order] * B:
            slices.append(slice(start, start + order))
            start += order
        return tuple(slices), start

    def design_matrix(self):
        if not jax.config.x64_enabled:
            raise ValueError("The band-pole kernel requires jax_enable_x64=True")
        B = self.lag_disk.shape[0]
        slices, size = self._chain_slices()
        f = jnp.ravel(self.tau_fast)[0]
        slow = jnp.concatenate(
            [jnp.ravel(self.tau_slow_band), jnp.ravel(self.tau_slow_uv)]
        )
        A = jnp.zeros((size, size), dtype=jnp.float64).at[0, 0].set(-1.0 / f)
        for b in range(B + 1):
            A = A.at[b + 1, 0].set(1.0 / slow[b])
            A = A.at[b + 1, b + 1].set(-1.0 / slow[b])
        lags = jnp.concatenate([self.lag_disk, self.lag_blr])
        for k, sl in enumerate(slices):
            order = sl.stop - sl.start
            rate = order / lags[k]
            parent = k + 1 if k < B else B + 1
            for j in range(order):
                A = A.at[sl.start + j, sl.start + j].set(-rate)
                A = A.at[sl.start + j, sl.start + j - 1 if j else parent].set(rate)
        return A

    def stationary_covariance(self):
        """Solve AP + PA.T + Q = 0 without dividing by pole differences.

        For lower-triangular A, each row-major element depends only on
        previously computed elements. P may be singular when states coincide.
        """
        A = self.design_matrix()
        size = A.shape[0]
        Q = jnp.zeros_like(A).at[0, 0].set(2.0 / jnp.ravel(self.tau_fast)[0])
        indices = jnp.arange(size)

        def update(k, P):
            i, j = k // size, k % size
            left = jnp.sum(jnp.where(indices < i, A[i] * P[:, j], 0.0))
            right = jnp.sum(jnp.where(indices < j, P[i] * A[j], 0.0))
            return P.at[i, j].set((-Q[i, j] - left - right) / (A[i, i] + A[j, j]))

        P = jax.lax.fori_loop(0, size * size, update, jnp.zeros_like(A))
        return (P + P.T) * 0.5

    def _observation_model_with_stds(self, X, stds):
        b = jnp.asarray(X[1], dtype=jnp.int32)
        B = self.lag_disk.shape[0]
        slices, size = self._chain_slices()
        endpoints = jnp.array([sl.stop - 1 for sl in slices])
        h = jnp.zeros(size, dtype=jnp.float64)
        h = h.at[endpoints[b]].set(self.amp_cont[b] / stds[b])
        return h.at[endpoints[B + b]].set(self.amp_blr[b] / stds[B + b])

    def observation_model(self, X):
        P = self.stationary_covariance()
        endpoints = jnp.array([sl.stop - 1 for sl in self._chain_slices()[0]])
        return self._observation_model_with_stds(X, jnp.sqrt(P[endpoints, endpoints]))

    def transition_matrix(self, X1, X2):
        return self._transition_matrices(jnp.atleast_1d(X2[0] - X1[0]))[0]

    def _transition_batch(self, X1, X2):
        return self._transition_matrices(jnp.asarray(X2[0]) - jnp.asarray(X1[0]))

    def _transition_matrices(self, dt):
        A = self.design_matrix()
        if self.transition == "expm":
            return jax.vmap(lambda t: expm(A * t, max_squarings=64))(dt)
        B = self.lag_disk.shape[0]
        a = -A[0, 0]
        phi = jnp.zeros((dt.size, *A.shape), dtype=jnp.float64)
        phi = phi.at[:, 0, 0].set(jnp.exp(-a * dt))
        for parent in range(1, B + 2):
            b = -A[parent, parent]
            phi = phi.at[:, parent, parent].set(jnp.exp(-b * dt))
            phi = phi.at[:, parent, 0].set(
                jax.vmap(lambda t: _driver_to_chain(t, a, b, 1)[0])(dt)
            )
        for index, sl in enumerate(self._chain_slices()[0]):
            order = sl.stop - sl.start
            parent = index + 1 if index < B else B + 1
            b, q = -A[parent, parent], -A[sl.start, sl.start]
            powers = np.subtract.outer(np.arange(order), np.arange(order))
            inv_factorial = jnp.asarray(
                [
                    [1.0 / math.factorial(int(n)) if n >= 0 else 0.0 for n in row]
                    for row in powers
                ]
            )
            block = (
                _exp_power(q * dt[:, None, None], jnp.asarray(np.maximum(powers, 0)))
                * inv_factorial
            )
            phi = phi.at[:, sl, sl].set(block)

            def analytic_columns(times):
                rb = jax.vmap(lambda t: _driver_to_chain(t, b, q, order))(times)
                ra = jax.vmap(lambda t: _driver_to_chain(t, a, q, order))(times)
                # A safe denominator also protects callers that vmap over
                # parameters, for which JAX evaluates both cond branches.
                gap = jnp.where(
                    jnp.abs(a - b) <= ANALYTIC_LOCAL_EXPM_RTOL * jnp.maximum(a, b),
                    1.0,
                    a - b,
                )
                return jnp.stack([b * (rb - ra) / gap, rb], axis=-1)

            def local_columns(times):
                indices = jnp.asarray([0, parent, *range(sl.start, sl.stop)])
                local = A[jnp.ix_(indices, indices)]
                return jax.vmap(lambda t: expm(local * t, max_squarings=64)[2:, :2])(
                    times
                )

            # The predicate is scalar and outside the time vmap: ordinary
            # likelihoods do not execute the local expm branch at every epoch.
            columns = jax.lax.cond(
                jnp.abs(a - b) <= ANALYTIC_LOCAL_EXPM_RTOL * jnp.maximum(a, b),
                local_columns,
                analytic_columns,
                dt,
            )
            phi = phi.at[:, sl, 0].set(columns[:, :, 0])
            phi = phi.at[:, sl, parent].set(columns[:, :, 1])
        return phi

    def to_symm_qsm(self, X):
        Pinf = self.stationary_covariance()
        slices, _size = self._chain_slices()
        endpoints = jnp.asarray([chain_slice.stop - 1 for chain_slice in slices])
        stds = jnp.sqrt(Pinf[endpoints, endpoints])
        Xprev = jax.tree_util.tree_map(lambda y: jnp.append(y[0], y[:-1]), X)
        transitions = self._transition_batch(Xprev, X)
        h = jax.vmap(lambda Xi: self._observation_model_with_stds(Xi, stds))(X)
        diag = jnp.einsum("ni,ij,nj->n", h, Pinf, h)
        p = jax.vmap(lambda hi, Fi: hi @ Fi)(h, transitions)
        q = h @ Pinf.T
        return SymmQSM(
            diag=DiagQSM(d=diag),
            lower=StrictLowerTriQSM(p=p, q=q, a=transitions),
        )

    def to_general_qsm(self, X1, X2):
        """Cross-covariance using this kernel's forward state transitions.

        X2 must be sorted by time; X1 may be in any order. Search physical
        times so simultaneous observations may have any band ordering, as in
        the training GP. For later i and earlier j, the covariance is
        ``h_i F_ij Pinf h_j.T``. The placement of Pinf therefore differs from
        tinygp's base implementation, which expects transposed transitions.
        """

        t1, t2 = jnp.asarray(X1[0]), jnp.asarray(X2[0])
        idx = jnp.searchsorted(t2, t1, side="right") - 1
        Xprev = jax.tree_util.tree_map(lambda x: jnp.append(x[0], x[:-1]), X2)
        transitions = self._transition_batch(Xprev, X2)
        Pinf = self.stationary_covariance()
        slices, _size = self._chain_slices()
        endpoints = jnp.asarray([chain_slice.stop - 1 for chain_slice in slices])
        stds = jnp.sqrt(Pinf[endpoints, endpoints])
        observe = lambda Xi: self._observation_model_with_stds(Xi, stds)
        h1, h2 = jax.vmap(observe)(X1), jax.vmap(observe)(X2)

        left = jnp.clip(idx, 0, t2.size - 1)
        Xleft = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[left], X2)
        # GeneralQSM masks the absent side when extrapolating. Use a zero-time
        # transition there too: evaluating an unused negative time can overflow
        # and contaminate parameter gradients despite that final mask.
        Xleft = jax.tree_util.tree_map(
            lambda x, query: jnp.where(idx >= 0, x, query), Xleft, X1
        )
        pl = jax.vmap(jnp.dot)(h1, self._transition_batch(Xleft, X1))

        right = jnp.clip(idx + 1, 0, t2.size - 1)
        Xright = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[right], X2)
        Xright = jax.tree_util.tree_map(
            lambda x, query: jnp.where(idx < t2.size - 1, x, query), Xright, X1
        )
        qu = jax.vmap(jnp.dot)(self._transition_batch(X1, Xright), h1 @ Pinf.T)
        return GeneralQSM(pl=pl, ql=h2 @ Pinf.T, pu=h2, qu=qu, a=transitions, idx=idx)

    def evaluate(self, X1, X2):
        Pinf = self.stationary_covariance()
        slices, _size = self._chain_slices()
        endpoints = jnp.asarray([chain_slice.stop - 1 for chain_slice in slices])
        stds = jnp.sqrt(Pinf[endpoints, endpoints])
        # Choose the causal direction before evaluating the transition. A
        # jnp.where over two covariances also evaluates the negative-time branch
        # and can produce NaN gradients when that unused branch overflows.
        first_is_earlier = X1[0] <= X2[0]
        earlier = jax.tree_util.tree_map(
            lambda x1, x2: jnp.where(first_is_earlier, x1, x2), X1, X2
        )
        later = jax.tree_util.tree_map(
            lambda x1, x2: jnp.where(first_is_earlier, x2, x1), X1, X2
        )
        h_early = self._observation_model_with_stds(earlier, stds)
        h_late = self._observation_model_with_stds(later, stds)
        return h_late @ self.transition_matrix(earlier, later) @ Pinf @ h_early

    def matmul(self, X1, X2=None, y=None):
        """Multiply using the matching symmetric or generalized QSM."""

        if y is None:
            if X2 is None:
                raise ValueError("Missing right-hand side for kernel matmul")
            y = X2
            X2 = None
        if X2 is None:
            return self.to_symm_qsm(X1) @ y
        return self.to_general_qsm(X1, X2) @ y

    def _moments(self, continuum_only=False):
        A, P = self.design_matrix(), self.stationary_covariance()
        B = self.lag_disk.shape[0]
        endpoints = jnp.array([sl.stop - 1 for sl in self._chain_slices()[0]])
        stds = jnp.sqrt(P[endpoints, endpoints])
        h = jax.vmap(lambda b: self._observation_model_with_stds((0.0, b), stds))(
            jnp.arange(B)
        )
        if continuum_only:
            h = (
                jnp.zeros_like(h)
                .at[jnp.arange(B), endpoints[:B]]
                .set(self.amp_cont / stds[:B])
            )
        variance = jnp.einsum("bi,ij,bj->b", h, P, h)
        area = jax.vmap(lambda hb: -hb @ jnp.linalg.solve(A, P @ hb))(h)
        return area / variance, jnp.sqrt(variance)

    def effective_timescales(self):
        return self._moments()[0]

    def stationary_rms(self):
        return self._moments()[1]

    def continuum_effective_timescales(self):
        return self._moments(continuum_only=True)[0]


def continuum_effective_timescale(tau_fast, tau_slow_uv, lag_disk_uv, *, disk_order=3):
    k = SharedLatentBandPolesBLRQS(
        tau_fast=jnp.atleast_1d(tau_fast),
        tau_slow_uv=jnp.atleast_1d(tau_slow_uv),
        tau_slow_band=jnp.atleast_1d(tau_slow_uv),
        lag_disk=jnp.atleast_1d(lag_disk_uv),
        lag_blr=jnp.ones(1),
        amp_cont=jnp.ones(1),
        amp_blr=jnp.zeros(1),
        disk_order=disk_order,
        blr_order=1,
    )
    return k.continuum_effective_timescales()[0]


class SharedLatentBandPolesBLRRelativeFluxModel(ContiBLRRelativeFlux_SHO_Model):
    disk_order: int
    blr_order: int
    transition: str = eqx.field(static=True)

    def __init__(
        self, *args, disk_order=3, blr_order=3, transition=DEFAULT_TRANSITION, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.disk_order, self.blr_order = int(disk_order), int(blr_order)
        transition_metadata(transition)
        self.transition = transition

    def _build_kernel(self, params):
        return SharedLatentBandPolesBLRQS(
            tau_fast=jnp.atleast_1d(params["tau_fast_driver"]),
            tau_slow_uv=jnp.atleast_1d(params["tau_slow_uv_driver"]),
            tau_slow_band=jnp.asarray(params["tau_slow_band"]),
            lag_disk=jnp.asarray(params["lag_disk"]),
            lag_blr=jnp.asarray(params["lag_blr"]),
            amp_cont=jnp.asarray(params["amp_cont_relflux"]),
            amp_blr=jnp.asarray(params["amp_blr_relflux"]),
            disk_order=self.disk_order,
            blr_order=self.blr_order,
            transition=self.transition,
        )

    def _build_gp(self, params):
        t, band = self.X
        inds = jnp.argsort(t)
        diag = self.diag + (self._jitter_diag(params, band) if self.has_jitter else 0.0)
        return (
            GaussianProcess(
                self._build_kernel(params),
                (t[inds], band[inds]),
                diag=diag[inds],
                mean=lambda X: self.get_mean(self.zero_mean, params, X),
                assume_sorted=True,
            ),
            inds,
        )


def make_multiband_shared_latent_band_poles_blr_model(
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
    disk_order=3,
    blr_order=3,
    transition=DEFAULT_TRANSITION,
):
    del baseline_flux_by_band
    if disk_order < 1 or blr_order < 1:
        raise ValueError("Response orders must be positive integers")
    if n_band is None:
        n_band = int(jnp.max(X[1])) + 1
    return SharedLatentBandPolesBLRRelativeFluxModel(
        X,
        y,
        yerr,
        base_kernel=OverdampedSHOBaseQS(jnp.array([10.0]), jnp.array([100.0])),
        nBand=n_band,
        mean_func=make_linear_mean_func(jnp.asarray(X[0]), zero_mean=zero_mean),
        survey_idx=survey_idx,
        seeing_covariate=seeing_covariate,
        zero_mean=zero_mean,
        has_jitter=has_jitter,
        has_lag=False,
        disk_order=disk_order,
        blr_order=blr_order,
        transition=transition,
    )
