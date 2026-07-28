"""Causal Erlang model with a DRW-style CARMA(2,1) continuum."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from tinygp import GaussianProcess

from qvc.light_curve.dho_drw_parameterization import IntegratedTimescaleDHOBaseQS
from qvc.light_curve.multiband_dho_core import make_linear_mean_func
from qvc.light_curve.multiband_model_dho_blr_erlang import (
    DEFAULT_ERLANG_ORDER,
    ContiBLRErlangRelativeFluxModel,
    ErlangResponseDHOQS,
)

POSITIVE_FLUX_N_SIGMA = 4.0
POSITIVE_FLUX_MARGIN_SOFTNESS = 0.01


class ErlangResponseIntegratedDHOQS(ErlangResponseDHOQS):
    """Erlang response driven by a CARMA(2,1) continuum."""

    carma_omega0: jnp.ndarray | None = None
    carma_damping: jnp.ndarray | None = None
    carma_obs_position: jnp.ndarray | None = None
    carma_obs_velocity: jnp.ndarray | None = None

    def _base(self):
        return IntegratedTimescaleDHOBaseQS(
            self.carma_omega0,
            self.carma_damping,
            self.carma_obs_position,
            self.carma_obs_velocity,
        )

    def _dimensions(self):
        B = int(self.carma_omega0.shape[0])
        return B, 2 * B, 2 * B + B * int(self.order)

    def design_matrix(self):
        base = self._base()
        A0 = base.design_matrix()
        B, n0, _ = self._dimensions()
        order = int(self.order)
        dtype = A0.dtype
        n_response = B * order
        rates = self._response_rates().astype(dtype)
        state_rates = jnp.repeat(rates, order)

        response = -jnp.diag(state_rates)
        sub_rows = jnp.arange(1, n_response, dtype=jnp.int32)
        within_chain = (sub_rows % order) != 0
        response = response.at[sub_rows, sub_rows - 1].set(
            jnp.where(within_chain, state_rates[sub_rows], 0.0)
        )

        bands = jnp.arange(B, dtype=jnp.int32)
        driver_loadings = jax.vmap(
            lambda b: base.observation_model((jnp.asarray(0.0), b))
        )(bands)
        driver = jnp.zeros((n_response, n0), dtype=dtype)
        chain_starts = bands * order
        driver = driver.at[chain_starts].set(rates[:, None] * driver_loadings)

        zero_top_right = jnp.zeros((n0, n_response), dtype=dtype)
        return jnp.block([[A0, zero_top_right], [driver, response]])

    def transition_matrix(self, X1, X2):
        t1, _ = X1
        t2, _ = X2
        dt = t2 - t1
        A = self.design_matrix()
        B, _n0, n = self._dimensions()
        idx = self._band_state_indices()
        A_blocks = A[idx[:, :, None], idx[:, None, :]]
        transition_blocks = jax.vmap(lambda block: expm(block * dt))(A_blocks)
        transition = jnp.zeros((n, n), dtype=A.dtype)
        bands = jnp.arange(B)
        return transition.at[
            idx[bands][:, :, None],
            idx[bands][:, None, :],
        ].set(transition_blocks)


class ContiBLRErlangIntegratedDHOModel(ContiBLRErlangRelativeFluxModel):
    """Relative-flux wrapper for the integrated-timescale DHO kernel."""

    def _build_kernel(self, params):
        tau_drw = jnp.asarray(params["tau_drw_band"])
        amp_cont = jnp.asarray(
            params["amp_cont_relflux"] if "amp_cont_relflux" in params else params["amp_cont"]
        )
        amp_blr_rms = jnp.asarray(
            params["amp_blr_relflux"] if "amp_blr_relflux" in params else params["amp_blr"]
        )
        base = IntegratedTimescaleDHOBaseQS.from_drw(
            tau_drw,
            jnp.asarray(params["quality_factor"]),
            jnp.asarray(params["tau_perturb_band"]),
        )
        lag_blr = jnp.asarray(params["lag_blr"])

        # Parameterize the line component by its stationary output RMS instead
        # of the Erlang filter's DC gain.  The latter becomes arbitrarily weak
        # for lag >> tau_drw and creates a gain-lag ridge in which enormous
        # coefficients have innocuous covariance.  Unit-response
        # normalization removes that ridge while retaining the same kernel
        # family and an interpretable continuum/line RMS ratio.
        unit_response = ErlangResponseIntegratedDHOQS(
            tau_fast=jnp.full_like(tau_drw, 0.5),
            tau_slow=jnp.full_like(tau_drw, 0.5),
            lag_blr=lag_blr,
            amp_cont=jnp.zeros_like(amp_cont),
            amp_blr=jnp.ones_like(amp_blr_rms),
            order=self.erlang_order,
            carma_omega0=base.omega0,
            carma_damping=base.damping,
            carma_obs_position=base.obs_position,
            carma_obs_velocity=base.obs_velocity,
        )
        bands = jnp.arange(self.nBand, dtype=jnp.int32)
        zero = jnp.asarray(0.0, dtype=tau_drw.dtype)
        unit_response_var = jax.vmap(
            lambda band: unit_response.evaluate((zero, band), (zero, band))
        )(bands)
        amp_blr_dc_gain = amp_blr_rms / jnp.sqrt(
            jnp.maximum(unit_response_var, 1e-12)
        )
        return ErlangResponseIntegratedDHOQS(
            tau_fast=jnp.full_like(tau_drw, 0.5),
            tau_slow=jnp.full_like(tau_drw, 0.5),
            lag_blr=lag_blr,
            amp_cont=amp_cont,
            amp_blr=amp_blr_dc_gain,
            order=self.erlang_order,
            carma_omega0=base.omega0,
            carma_damping=base.damping,
            carma_obs_position=base.obs_position,
            carma_obs_velocity=base.obs_velocity,
        )

    def positive_flux_margin(self, params):
        """Return the per-band four-sigma margin above zero total flux."""
        kernel = self._build_kernel(params)
        bands = jnp.arange(self.nBand, dtype=jnp.int32)
        zero = jnp.asarray(0.0, dtype=self.y.dtype)
        variance = jax.vmap(
            lambda band: kernel.evaluate((zero, band), (zero, band))
        )(bands)
        stationary_std = jnp.sqrt(jnp.maximum(variance, 0.0))

        means = jax.vmap(partial(self.get_mean, self.zero_mean, params))(self.X)
        observed_bands = jnp.asarray(self.X[1], dtype=jnp.int32)
        min_mean = jax.vmap(
            lambda band: jnp.min(
                jnp.where(observed_bands == band, means, jnp.inf)
            )
        )(bands)
        return (
            1.0
            + min_mean
            - jnp.asarray(POSITIVE_FLUX_N_SIGMA, dtype=self.y.dtype)
            * stationary_std
        )

    def positive_flux_log_penalty(self, params):
        """Smooth diagnostic penalty retained for reporting and tests."""

        margin = self.positive_flux_margin(params)
        scaled_violation = -margin / jnp.asarray(
            POSITIVE_FLUX_MARGIN_SOFTNESS,
            dtype=self.y.dtype,
        )
        return -0.5 * jnp.sum(jax.nn.softplus(scaled_violation) ** 2)

    @eqx.filter_jit
    def log_prob(self, params):
        # Keep the boundary differentiable so AutoNormal SVI can initialize;
        # at the chosen softness, even a 0.1 flux-ratio violation costs about
        # 50 log-probability units per affected band.
        return super().log_prob(params) + self.positive_flux_log_penalty(params)


def make_multiband_dho_blr_flux_linearized_erlang_drw_model(
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
    """Construct the all-regime CARMA(2,1) plus causal Erlang response model."""

    del baseline_flux_by_band
    if n_band is None:
        n_band = int(jnp.max(jnp.asarray(X[1], dtype=jnp.int32))) + 1
    t = jnp.asarray(X[0])
    return ContiBLRErlangIntegratedDHOModel(
        X,
        y,
        yerr,
        base_kernel=IntegratedTimescaleDHOBaseQS.from_drw(
            tau_drw=jnp.full(n_band, 100.0),
            quality_factor=jnp.full(n_band, 0.5),
            tau_perturb=jnp.full(n_band, 2.0),
        ),
        nBand=n_band,
        mean_func=make_linear_mean_func(t, zero_mean=zero_mean),
        has_lag=False,
        has_jitter=has_jitter,
        zero_mean=zero_mean,
        survey_idx=survey_idx,
        erlang_order=erlang_order,
        use_fast_solver=False,
    )


__all__ = [
    "ContiBLRErlangIntegratedDHOModel",
    "ErlangResponseIntegratedDHOQS",
    "POSITIVE_FLUX_MARGIN_SOFTNESS",
    "POSITIVE_FLUX_N_SIGMA",
    "make_multiband_dho_blr_flux_linearized_erlang_drw_model",
]
