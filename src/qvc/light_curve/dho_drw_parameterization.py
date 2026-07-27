"""DRW-style coordinates for the CARMA(2,1) DHO continuum.

The user-facing timescale is always the normalized integrated correlation time

    tau_drw = integral_0^infinity k(t) / k(0) dt

including when the moving-average numerator is nonzero.  The kernel is
normalized to unit stationary variance, so the calling model's amplitude is
the stationary RMS.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from tinygp.helpers import JAXArray
from tinygp.kernels import quasisep as qs


DEFAULT_RHO = 1.0 / 150.0
RHO_LOGIT_PRIOR_SIGMA_DEX = 0.4
DEFAULT_QUALITY_FACTOR = 0.1
LOG_QUALITY_FACTOR_PRIOR_SIGMA = 1.0
MIN_QUALITY_FACTOR = 0.05
MAX_QUALITY_FACTOR = 3.0
DEFAULT_PERTURBATION_TO_DRW_RATIO = 0.02
PERTURBATION_RATIO_PRIOR_SIGMA_DEX = 0.5
MIN_PERTURBATION_TO_DRW_RATIO = 1e-3
MAX_PERTURBATION_TO_DRW_RATIO = 0.5


def dho_timescales_from_drw(tau_drw, rho):
    """Return ``(tau_fast, tau_slow)`` from integrated time and pole ratio."""

    tau_drw = jnp.asarray(tau_drw)
    rho = jnp.asarray(rho, dtype=tau_drw.dtype)
    tau_slow = tau_drw / (1.0 + rho)
    tau_fast = rho * tau_slow
    return tau_fast, tau_slow


def dho_log_timescales_from_drw(log_tau_drw, logit_rho):
    """Log-space version of :func:`dho_timescales_from_drw`."""

    log_tau_drw = jnp.asarray(log_tau_drw)
    rho = jax.nn.sigmoid(jnp.asarray(logit_rho, dtype=log_tau_drw.dtype))
    log_tau_slow = log_tau_drw - jnp.log1p(rho)
    log_tau_fast = log_tau_slow + jnp.log(rho)
    return log_tau_fast, log_tau_slow, rho


def dho_logit_rho_prior():
    """Weakly informative pole-ratio prior matching the legacy center."""

    rho0 = jnp.asarray(DEFAULT_RHO)
    loc = jnp.log(rho0) - jnp.log1p(-rho0)
    scale = RHO_LOGIT_PRIOR_SIGMA_DEX * jnp.log(10.0)
    return dist.Normal(loc, scale)


def log_quality_factor_prior():
    """Legacy-centered prior with a tail into the oscillatory regime.

    Most prior mass is overdamped, near the quality factor implied by the
    legacy pole-ratio prior. Genuine QPO-like data can still move above
    critical damping, while the upper bound excludes extremely coherent,
    poorly identified solutions on a finite light-curve baseline.
    """

    return dist.TruncatedNormal(
        jnp.log(jnp.asarray(DEFAULT_QUALITY_FACTOR)),
        jnp.asarray(LOG_QUALITY_FACTOR_PRIOR_SIGMA),
        low=jnp.log(jnp.asarray(MIN_QUALITY_FACTOR)),
        high=jnp.log(jnp.asarray(MAX_QUALITY_FACTOR)),
    )


def log_perturbation_ratio_prior():
    """Prior for ``tau_perturb / tau_drw`` in the CARMA(2,1) numerator."""

    return dist.TruncatedNormal(
        jnp.log(jnp.asarray(DEFAULT_PERTURBATION_TO_DRW_RATIO)),
        jnp.asarray(PERTURBATION_RATIO_PRIOR_SIGMA_DEX * jnp.log(10.0)),
        low=jnp.log(jnp.asarray(MIN_PERTURBATION_TO_DRW_RATIO)),
        high=jnp.log(jnp.asarray(MAX_PERTURBATION_TO_DRW_RATIO)),
    )


def carma21_response_parameters(tau_drw, quality_factor, tau_perturb):
    """Return response ``(omega0, damping, ma_ratio)`` for DRW-style inputs.

    ``ma_ratio = omega0 * tau_perturb`` is obtained from the unique positive
    solution of ``x + x**3 = tau_perturb / (Q * tau_drw)``.  This choice makes
    the normalized observed process have integral correlation time exactly
    ``tau_drw`` despite the CARMA(2,1) numerator.
    """

    tau = jnp.maximum(jnp.asarray(tau_drw), 1e-12)
    q = jnp.maximum(jnp.asarray(quality_factor, dtype=tau.dtype), 1e-6)
    q = jnp.broadcast_to(q, tau.shape)
    tau_perturb = jnp.maximum(
        jnp.broadcast_to(jnp.asarray(tau_perturb, dtype=tau.dtype), tau.shape),
        1e-12,
    )
    cubic_rhs = tau_perturb / (q * tau)
    ma_ratio = (2.0 / jnp.sqrt(3.0)) * jnp.sinh(
        jnp.arcsinh(1.5 * jnp.sqrt(3.0) * cubic_rhs) / 3.0
    )
    response_tau = tau * (1.0 + ma_ratio**2)
    omega0 = 1.0 / (q * response_tau)
    damping = omega0 / q
    return omega0, damping, ma_ratio


class IntegratedTimescaleDHOBaseQS(qs.Quasisep):
    """Unit-RMS CARMA(2,1) DHO in integral-timescale coordinates."""

    omega0: jnp.ndarray
    damping: jnp.ndarray
    obs_position: jnp.ndarray
    obs_velocity: jnp.ndarray

    @classmethod
    def from_drw(cls, tau_drw, quality_factor, tau_perturb):
        """Construct the cached state-space coefficients from public inputs."""

        tau = jnp.maximum(jnp.asarray(tau_drw), 1e-12)
        q = jnp.maximum(jnp.asarray(quality_factor, dtype=tau.dtype), 1e-6)
        q = jnp.broadcast_to(q, tau.shape)
        tau_perturb = jnp.broadcast_to(
            jnp.asarray(tau_perturb, dtype=tau.dtype),
            tau.shape,
        )
        omega0, damping, ma_ratio = carma21_response_parameters(
            tau,
            q,
            tau_perturb,
        )
        norm = jnp.sqrt(1.0 + ma_ratio**2)
        return cls(
            omega0=omega0,
            damping=damping,
            obs_position=1.0 / norm,
            obs_velocity=tau_perturb / norm,
        )

    def coord_to_sortable(self, X):
        t, b = X
        return t + 1e-9 * jnp.asarray(b, dtype=jnp.int32)

    def design_matrix(self):
        omega0 = jnp.asarray(self.omega0)
        damping = jnp.asarray(self.damping)
        B = int(omega0.shape[0])
        zero = jnp.zeros((B, B), dtype=omega0.dtype)
        eye = jnp.eye(B, dtype=omega0.dtype)
        return jnp.block(
            [
                [zero, eye],
                [-jnp.diag(omega0**2), -jnp.diag(damping)],
            ]
        )

    def stationary_covariance(self):
        """Solve small per-band-pair Lyapunov systems for the shared driver."""

        omega0 = jnp.asarray(self.omega0)
        damping = jnp.asarray(self.damping)
        B = int(omega0.shape[0])
        A = self.design_matrix()
        beta = jnp.sqrt(2.0 * damping * omega0**2)
        bands = jnp.arange(B)
        idx = jnp.stack([bands, B + bands], axis=1)
        A_blocks = A[idx[:, :, None], idx[:, None, :]]
        eye2 = jnp.eye(2, dtype=A.dtype)

        def solve_pair(bi, bj):
            forcing_i = jnp.array([0.0, beta[bi]], dtype=A.dtype)
            forcing_j = jnp.array([0.0, beta[bj]], dtype=A.dtype)
            diffusion = forcing_i[:, None] * forcing_j[None, :]
            operator = (
                jnp.kron(A_blocks[bi], eye2)
                + jnp.kron(eye2, A_blocks[bj])
            )
            return jnp.linalg.solve(
                operator,
                -diffusion.reshape(-1),
            ).reshape((2, 2))

        bi, bj = jnp.meshgrid(bands, bands, indexing="ij")
        blocks = jax.vmap(solve_pair)(bi.ravel(), bj.ravel())
        covariance = jnp.zeros((2 * B, 2 * B), dtype=A.dtype)
        covariance = covariance.at[
            idx[bi.ravel()][:, :, None],
            idx[bj.ravel()][:, None, :],
        ].set(blocks)
        return 0.5 * (covariance + covariance.T)

    def observation_model(self, X: JAXArray) -> JAXArray:
        _t, b = X
        b = jnp.asarray(b, dtype=jnp.int32)
        B = int(self.omega0.shape[0])
        h = jnp.zeros(2 * B, dtype=jnp.asarray(self.omega0).dtype)
        h = h.at[b].set(self.obs_position[b])
        h = h.at[B + b].set(self.obs_velocity[b])
        return h

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        from jax.scipy.linalg import expm

        t1, _ = X1
        t2, _ = X2
        return expm(self.design_matrix() * (t2 - t1))


__all__ = [
    "DEFAULT_RHO",
    "RHO_LOGIT_PRIOR_SIGMA_DEX",
    "DEFAULT_QUALITY_FACTOR",
    "LOG_QUALITY_FACTOR_PRIOR_SIGMA",
    "MIN_QUALITY_FACTOR",
    "MAX_QUALITY_FACTOR",
    "DEFAULT_PERTURBATION_TO_DRW_RATIO",
    "PERTURBATION_RATIO_PRIOR_SIGMA_DEX",
    "MIN_PERTURBATION_TO_DRW_RATIO",
    "MAX_PERTURBATION_TO_DRW_RATIO",
    "IntegratedTimescaleDHOBaseQS",
    "carma21_response_parameters",
    "dho_logit_rho_prior",
    "dho_log_timescales_from_drw",
    "dho_timescales_from_drw",
    "log_quality_factor_prior",
    "log_perturbation_ratio_prior",
]
