"""Bounded-memory marginal predictions for the two forward-aware SLB kernels.

These routines do not construct joint conditional distributions. In particular,
independent draws from query blocks would not be joint posterior samples.
"""

import jax
import jax.numpy as jnp
from tinygp.solvers.quasisep.core import DiagQSM


def supports_shared_prediction(value):
    # Local imports avoid a cycle through the shared model adapter.
    from .multiband_model_shared_latent_blr import (
        SharedLatentDiskBLRQS, SharedLatentDiskBLRRelativeFluxModel,
    )
    from .multiband_model_shared_latent_band_poles_blr import (
        SharedLatentBandPolesBLRQS, SharedLatentBandPolesBLRRelativeFluxModel,
    )
    return isinstance(value, (
        SharedLatentDiskBLRQS, SharedLatentDiskBLRRelativeFluxModel,
        SharedLatentBandPolesBLRQS, SharedLatentBandPolesBLRRelativeFluxModel,
    ))


def marginal_prediction(gp, y, X, *, block_size=64):
    """Return conditional mean and variance in caller query order.

    Only an N_train x block_size cross covariance and a block-sized identity are
    needed. X2 is sorted for GeneralQSM; duplicate padded queries are harmless
    since the query covariance is never factorized.
    """
    X = jax.tree_util.tree_map(jnp.asarray, X)
    n = X[0].size
    if not n:
        empty = jnp.empty((0,), dtype=gp.loc.dtype)
        return empty, empty
    size = min(block_size, n)
    padding = (-n) % size
    blocks = jax.tree_util.tree_map(
        lambda x: jnp.pad(x, (0, padding), mode="edge").reshape(-1, size), X
    )
    alpha = gp.solver.solve_triangular(y - gp.loc)
    alpha = gp.solver.solve_triangular(alpha, transpose=True)
    identity = jnp.eye(size, dtype=gp.loc.dtype)

    def predict_block(query):
        order = jnp.argsort(query[0])
        query = jax.tree_util.tree_map(lambda x: x[order], query)
        cross = gp.kernel.to_general_qsm(gp.X, query) @ identity
        solved = gp.solver.solve_triangular(cross)
        mean = jax.vmap(gp.mean_function)(query) + cross.T @ alpha
        prior_diag = jax.vmap(gp.kernel.evaluate_diag)(query)
        # Match GaussianProcess.condition's default prediction noise exactly.
        diag = jnp.sqrt(jnp.finfo(mean.dtype).eps)
        variance = prior_diag + diag - jnp.sum(solved * solved, axis=0)
        inverse = jnp.argsort(order)
        return mean[inverse], variance[inverse]

    mean, variance = jax.lax.map(predict_block, blocks)
    return mean.reshape(-1)[:n], variance.reshape(-1)[:n]


@jax.jit
def regularized_loo_residuals(matrix, centered):
    """Exact QSM precision residuals with the historical LOO regularization."""
    scale = jnp.maximum(jnp.nanmedian(matrix.diag.d), 1.0)
    regularized = matrix + DiagQSM(jnp.full_like(matrix.diag.d, 1e-10 * scale))
    precision = regularized.inv()
    return (precision @ centered) / jnp.sqrt(jnp.maximum(precision.diag.d, 1e-300))
