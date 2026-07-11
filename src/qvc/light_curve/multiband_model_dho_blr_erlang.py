"""Quasi-separable DHO continuum with causal Erlang BLR responses.

Unlike the legacy ``h @ Phi(lag)`` loading, the response states below retain a
filtered history of the continuum.  A chain of ``order`` identical first-order
filters has an Erlang impulse response with centroid ``lag`` and standard
deviation ``lag / sqrt(order)``.
"""

import math
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import expm
from tinygp import GaussianProcess
from tinygp.kernels import quasisep as qs

from qvc.light_curve.multiband_model_dho_blr import (
    ContiBLRRelativeFlux_SHO_Model,
    OverdampedSHOBaseQS,
    _safe_pos,
    make_linear_mean_func,
)


DEFAULT_ERLANG_ORDER = 3


def erlang_impulse_response(t, lag, order=DEFAULT_ERLANG_ORDER):
    """Return the unit-area Erlang response whose centroid is ``lag``."""

    t = jnp.asarray(t, dtype=float)
    lag = _safe_pos(jnp.asarray(lag, dtype=t.dtype))
    order = int(order)
    if order < 1:
        raise ValueError("Erlang order must be at least one.")
    theta = lag / order
    log_response = (
        (order - 1) * jnp.log(jnp.maximum(t, 1e-300))
        - t / theta
        - order * jnp.log(theta)
        - jax.scipy.special.gammaln(order)
    )
    return jnp.where(t >= 0.0, jnp.exp(log_response), 0.0)


def erlang_response_moments(lag, order=DEFAULT_ERLANG_ORDER):
    """Return the analytic centroid and standard deviation of the response."""

    lag = jnp.asarray(lag, dtype=float)
    return lag, lag / jnp.sqrt(jnp.asarray(float(order)))


class ErlangResponseDHOQS(qs.Quasisep):
    """DHO driver augmented by one causal Erlang response chain per band."""

    tau_fast: jnp.ndarray
    tau_slow: jnp.ndarray
    lag_blr: jnp.ndarray
    amp_cont: jnp.ndarray
    amp_blr: jnp.ndarray
    order: int = eqx.field(static=True, default=DEFAULT_ERLANG_ORDER)

    def coord_to_sortable(self, X):
        t, b = X
        return t + 1e-9 * jnp.asarray(b, dtype=jnp.int32)

    def _base(self):
        return OverdampedSHOBaseQS(self.tau_fast, self.tau_slow)

    def _dimensions(self):
        B = int(self.tau_fast.shape[0])
        return B, 2 * B, 2 * B + B * int(self.order)

    def design_matrix(self):
        base = self._base()
        A0 = base.design_matrix()
        B, n0, n = self._dimensions()
        order = int(self.order)
        dtype = A0.dtype
        n_response = B * order
        rates = self._response_rates().astype(dtype)
        state_rates = jnp.repeat(rates, order)

        # Each response chain has -q on its diagonal and q directly below the
        # diagonal, except at boundaries between bands.
        response = -jnp.diag(state_rates)
        sub_rows = jnp.arange(1, n_response, dtype=jnp.int32)
        within_chain = (sub_rows % order) != 0
        response = response.at[sub_rows, sub_rows - 1].set(
            jnp.where(within_chain, state_rates[sub_rows], 0.0)
        )

        # The first state in each chain is driven by q times that band's DHO
        # observation. Build all band loadings at once from the base scales.
        obs_scale = base._obs_scale()
        eye = jnp.eye(B, dtype=dtype)
        driver_loadings = jnp.concatenate(
            [obs_scale[:, None] * eye, -obs_scale[:, None] * eye],
            axis=1,
        )
        driver = jnp.zeros((n_response, n0), dtype=dtype)
        chain_starts = jnp.arange(B, dtype=jnp.int32) * order
        driver = driver.at[chain_starts].set(rates[:, None] * driver_loadings)

        zero_top_right = jnp.zeros((n0, n_response), dtype=dtype)
        return jnp.block([[A0, zero_top_right], [driver, response]])

    def _band_state_indices(self):
        """State indices grouped per band: (fast, slow, chain_1..chain_k)."""

        B, n0, _n = self._dimensions()
        k = int(self.order)
        b = jnp.arange(B)
        return jnp.concatenate(
            [
                b[:, None],
                B + b[:, None],
                n0 + b[:, None] * k + jnp.arange(k)[None, :],
            ],
            axis=1,
        )

    def stationary_covariance(self):
        """Solve the continuous Lyapunov equation for the augmented state.

        The design matrix has no dynamical coupling between bands, so the
        Lyapunov equation splits into one small Sylvester equation per band
        pair, ``A_b P_bb' + P_bb' A_b'^T = -Q_bb'``, instead of one dense
        Kronecker system over the full augmented state.
        """

        base = self._base()
        A = self.design_matrix()
        P0 = base.stationary_covariance()
        A0 = base.design_matrix()
        Q0 = -(A0 @ P0 + P0 @ A0.T)
        B, n0, n = self._dimensions()
        k = int(self.order)
        m = 2 + k

        idx = self._band_state_indices()
        A_blocks = A[idx[:, :, None], idx[:, None, :]]
        b = jnp.arange(B)
        driver_idx = jnp.stack([b, B + b], axis=1)
        eye_m = jnp.eye(m, dtype=A.dtype)

        def solve_pair(bi, bj):
            # White noise only enters through the driver states, so Q_bb'
            # is nonzero only in the leading 2x2 block.
            Q_pair = jnp.zeros((m, m), dtype=A.dtype)
            Q_pair = Q_pair.at[:2, :2].set(
                Q0[driver_idx[bi][:, None], driver_idx[bj][None, :]]
            )
            # Row-major vec: vec(A_bi X + X A_bj^T) =
            # (A_bi kron I + I kron A_bj) vec(X).
            operator = jnp.kron(A_blocks[bi], eye_m) + jnp.kron(eye_m, A_blocks[bj])
            return jnp.linalg.solve(operator, -Q_pair.reshape(-1)).reshape((m, m))

        bi, bj = jnp.meshgrid(b, b, indexing="ij")
        blocks = jax.vmap(solve_pair)(bi.ravel(), bj.ravel())
        P = jnp.zeros((n, n), dtype=A.dtype)
        P = P.at[idx[bi.ravel()][:, :, None], idx[bj.ravel()][:, None, :]].set(blocks)
        return 0.5 * (P + P.T)

    def _stationary_covariance_kron(self):
        """Numerical oracle for stationary_covariance(), retained for tests."""

        base = self._base()
        A = self.design_matrix()
        P0 = base.stationary_covariance()
        A0 = base.design_matrix()
        Q0 = -(A0 @ P0 + P0 @ A0.T)
        n0 = A0.shape[0]
        n = A.shape[0]
        Q = jnp.zeros((n, n), dtype=A.dtype).at[:n0, :n0].set(Q0)

        # vec(A P + P A^T) = (I kron A + A kron I) vec(P), using
        # row-major vectorization consistently on both sides.
        I = jnp.eye(n, dtype=A.dtype)
        operator = jnp.kron(A, I) + jnp.kron(I, A)
        P = jnp.linalg.solve(operator, -Q.reshape(-1)).reshape((n, n))
        return 0.5 * (P + P.T)

    def observation_model(self, X):
        _t, b = X
        b = jnp.asarray(b, dtype=jnp.int32)
        base = self._base()
        B, n0, n = self._dimensions()
        h_driver = base.observation_model((jnp.array(0.0), b))
        h = jnp.zeros(n, dtype=h_driver.dtype)
        h = h.at[:n0].set(_safe_pos(jnp.asarray(self.amp_cont))[b] * h_driver)
        response_idx = n0 + b * int(self.order) + int(self.order) - 1
        h = h.at[response_idx].set(_safe_pos(jnp.asarray(self.amp_blr))[b])
        return h

    def _response_rates(self):
        return int(self.order) / _safe_pos(jnp.asarray(self.lag_blr, dtype=float))

    def _driver_to_chain_columns(self, dt, lam, e_lam, q, eq, sign):
        """Closed-form response of each chain state to a unit driver state.

        A driver mode ``e^{-lam t}`` fed through ``j`` identical first-order
        filters of rate ``q`` gives, with ``z = q - lam``,

            x_j(dt) = (q / z)^j [e^{-lam dt} - e^{-q dt} S_{j-1}(z dt)]
                    = e^{-q dt} (q dt)^j  sum_{p>=0} (z dt)^p / (p + j)!

        where ``S_{j-1}`` is the exponential partial sum.  The first form
        suffers catastrophic cancellation for small ``|z dt|`` (error grows
        as ``(q/z)^j``), so it is only used for ``|z dt| > 1``; the second
        converges quickly there and is exact at ``z = 0``.  Each branch sees
        clamped inputs so the unselected branch stays finite under autodiff.
        """

        k = int(self.order)
        j = np.arange(1, k + 1)
        z = q - lam
        zt = z * dt
        use_series = jnp.abs(zt) <= 1.0

        n_terms = 26
        p = np.arange(n_terms)
        inv_fact_pj = jnp.asarray(
            [[1.0 / math.factorial(int(pi) + int(ji)) for pi in p] for ji in j]
        )
        zt_series = jnp.where(use_series, zt, 0.0)
        zt_pows = jnp.power(zt_series[:, None], p)
        series = (
            eq[:, None]
            * jnp.power(q[:, None] * dt, j)
            * jnp.einsum("bp,jp->bj", zt_pows, inv_fact_pj)
        )

        z_safe = jnp.where(use_series, 1.0, z)
        zt_safe = jnp.where(use_series, 1.0, zt)
        m = np.arange(k)
        inv_fact_m = jnp.asarray([1.0 / math.factorial(int(i)) for i in m])
        partial_sums = jnp.cumsum(jnp.power(zt_safe[:, None], m) * inv_fact_m, axis=1)
        difference = jnp.power((q / z_safe)[:, None], j) * (
            e_lam[:, None] - eq[:, None] * partial_sums
        )

        return sign * jnp.where(use_series[:, None], series, difference)

    def transition_matrix(self, X1, X2):
        """Analytic ``expm(A^T dt)`` exploiting the block band structure.

        The design matrix has no dynamical coupling between bands: each band
        is an independent block of two continuum poles plus one Erlang chain,
        so every block of the matrix exponential is available in closed form.
        tinygp's Quasisep convention contracts ``Pinf @ transition`` on the
        right, so the assembled matrix is the transpose of the usual
        column-state transition.
        """

        t1, _ = X1
        t2, _ = X2
        dt = t2 - t1
        B, n0, n = self._dimensions()
        k = int(self.order)
        base = self._base()
        tau_fast, tau_slow = base._ordered_taus()
        lam_fast = 1.0 / tau_fast
        lam_slow = 1.0 / tau_slow
        obs_scale = base._obs_scale()
        q = self._response_rates()

        e_fast = jnp.exp(-lam_fast * dt)
        e_slow = jnp.exp(-lam_slow * dt)
        eq = jnp.exp(-q * dt)

        # Chain-to-chain block: the exponential of a Jordan-like block is
        # lower-triangular Toeplitz, e^{-q dt} (q dt)^d / d! on subdiagonal d.
        d = np.subtract.outer(np.arange(k), np.arange(k))
        inv_fact_d = jnp.asarray(
            [[1.0 / math.factorial(int(v)) if v >= 0 else 0.0 for v in row] for row in d]
        )
        chain = (
            eq[:, None, None]
            * jnp.power(q[:, None, None] * dt, np.maximum(d, 0))
            * inv_fact_d
        )

        col_fast = obs_scale[:, None] * self._driver_to_chain_columns(
            dt, lam_fast, e_fast, q, eq, +1.0
        )
        col_slow = obs_scale[:, None] * self._driver_to_chain_columns(
            dt, lam_slow, e_slow, q, eq, -1.0
        )

        # Assemble the standard (rows = to-state) transition, then transpose.
        b = jnp.arange(B)
        chain_idx = n0 + b[:, None] * k + jnp.arange(k)[None, :]
        phi = jnp.zeros((n, n), dtype=e_fast.dtype)
        phi = phi.at[b, b].set(e_fast)
        phi = phi.at[B + b, B + b].set(e_slow)
        phi = phi.at[chain_idx[:, :, None], chain_idx[:, None, :]].set(chain)
        phi = phi.at[chain_idx, b[:, None]].set(col_fast)
        phi = phi.at[chain_idx, B + b[:, None]].set(col_slow)
        return phi.T

    def _transition_matrix_expm(self, X1, X2):
        """Numerical oracle for transition_matrix(), retained for tests."""

        t1, _ = X1
        t2, _ = X2
        return expm(self.design_matrix().T * (t2 - t1))


class ContiBLRErlangRelativeFluxModel(ContiBLRRelativeFlux_SHO_Model):
    """Relative-flux model that constructs the augmented Erlang kernel."""

    erlang_order: int

    def __init__(self, *args, erlang_order=DEFAULT_ERLANG_ORDER, **kwargs):
        super().__init__(*args, **kwargs)
        self.erlang_order = int(erlang_order)

    def _build_gp(self, params):
        means = partial(self.get_mean, self.zero_mean, params)
        X, inds = self.lag_transform(False, params, self.X)
        t, band = X
        diags = self.diag
        if self.has_jitter:
            diags = diags + self._jitter_diag(params, band)

        tau_fast = jnp.asarray(params["tau_fast_band"])
        tau_slow = jnp.asarray(params["tau_slow_band"])
        amp_cont = jnp.asarray(
            params["amp_cont_relflux"] if "amp_cont_relflux" in params else params["amp_cont"]
        )
        amp_blr = jnp.asarray(
            params["amp_blr_relflux"] if "amp_blr_relflux" in params else params["amp_blr"]
        )
        kernel = ErlangResponseDHOQS(
            tau_fast=tau_fast,
            tau_slow=tau_slow,
            lag_blr=jnp.asarray(params["lag_blr"]),
            amp_cont=amp_cont,
            amp_blr=amp_blr,
            order=self.erlang_order,
        )
        return (
            GaussianProcess(
                kernel,
                (t[inds], band[inds]),
                diag=diags[inds],
                mean=means,
                assume_sorted=True,
            ),
            inds,
        )


def make_multiband_dho_blr_flux_linearized_erlang_model(
    X,
    y,
    yerr,
    n_band=None,
    *,
    survey_idx=None,
    baseline_flux_by_band=None,
    zero_mean=False,
    has_jitter=True,
    erlang_order=DEFAULT_ERLANG_ORDER,
):
    """Construct the relative-flux DHO plus causal Erlang BLR model."""

    del baseline_flux_by_band
    if n_band is None:
        n_band = int(jnp.max(jnp.asarray(X[1], dtype=jnp.int32))) + 1
    t = jnp.asarray(X[0])
    return ContiBLRErlangRelativeFluxModel(
        X,
        y,
        yerr,
        base_kernel=OverdampedSHOBaseQS(
            tau_fast=jnp.full(n_band, 10.0),
            tau_slow=jnp.full(n_band, 100.0),
        ),
        nBand=n_band,
        mean_func=make_linear_mean_func(t, zero_mean=zero_mean),
        survey_idx=survey_idx,
        zero_mean=zero_mean,
        has_jitter=has_jitter,
        has_lag=False,
        erlang_order=erlang_order,
    )


__all__ = [
    "DEFAULT_ERLANG_ORDER",
    "ContiBLRErlangRelativeFluxModel",
    "ErlangResponseDHOQS",
    "erlang_impulse_response",
    "erlang_response_moments",
    "make_multiband_dho_blr_flux_linearized_erlang_model",
]
