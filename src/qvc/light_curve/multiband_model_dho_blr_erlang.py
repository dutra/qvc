"""Quasi-separable DHO continuum with causal Erlang BLR responses.

Unlike the legacy ``h @ Phi(lag)`` loading, the response states below retain a
filtered history of the continuum.  A chain of ``order`` identical first-order
filters has an Erlang impulse response with centroid ``lag`` and standard
deviation ``lag / sqrt(order)``.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from tinygp import GaussianProcess
from tinygp.kernels import quasisep as qs

from qvc.light_curve.multiband_model_dho_blr import (
    ContiBLRRelativeFlux_SHO_Model,
    OverdampedSHOBaseQS,
    _safe_pos,
    make_linear_mean_func,
)


DEFAULT_ERLANG_ORDER = 4


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
    order: int = DEFAULT_ERLANG_ORDER

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
        rates = order / _safe_pos(jnp.asarray(self.lag_blr, dtype=dtype))
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

    def stationary_covariance(self):
        """Solve the continuous Lyapunov equation for the augmented state."""

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

    def transition_matrix(self, X1, X2):
        t1, _ = X1
        t2, _ = X2
        # tinygp's Quasisep convention contracts ``Pinf @ transition`` on the
        # right, so this is the transpose of the usual column-state transition.
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
