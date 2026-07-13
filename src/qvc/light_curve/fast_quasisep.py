"""Fused single-scan quasiseparable GP log-likelihood with a custom adjoint.

Computes exactly the same value as ``GaussianProcess(kernel, X, diag=...)
.log_probability(y)`` for ``tinygp.kernels.quasisep.Quasisep`` kernels, but:

- factor + solve + normalization run in one forward ``lax.scan``;
- the reverse pass is a hand-written adjoint scan (3 gemms/step, exploiting
  the symmetry of the Cholesky auxiliary matrix), instead of jax's scan VJP;
- points with an exactly-zero time gap (simultaneous multiband observations)
  take a gemm-free branch — the transition is the identity there and carries
  no parameter gradient;
- for ``ErlangResponseDHOQS`` the batched transition build uses a single
  static-index scatter (``erlang_transitions``) whose reverse pass is one
  gather, replacing the per-point ``.at[].set`` assembly whose batched VJP
  dominated the construction gradient.

Agreement with the tinygp path is ~1e-12 relative in value and ~1e-8 in
gradients (see ``tests/test_fast_quasisep.py``).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["fused_log_probability", "scan_loglike", "erlang_transitions"]


def _driver_to_chain_columns_horner(dt, lam, e_lam, q, eq, sign, order):
    """Equivalent of ``ErlangResponseDHOQS._driver_to_chain_columns``.

    Same two-branch closed form (series for |zt| <= 1, partial-sum difference
    otherwise), but the series is evaluated by Horner recursion and the integer
    powers by cumulative products.  This matches the kernel's implementation to
    machine precision while keeping the reverse pass to a short chain of fused
    multiply-adds — the kernel's power-table einsum has a VJP that would
    otherwise dominate the fast path's construction gradient.
    """
    k = int(order)
    z = q - lam
    zt = z * dt
    use_series = jnp.abs(zt) <= 1.0

    n_terms = 20  # remainder below 1/21! ~ 2e-20 for |zt| <= 1
    coeffs = np.asarray(
        [[1.0 / math.factorial(pi + ji) for ji in range(1, k + 1)] for pi in range(n_terms)]
    )
    zt_series = jnp.where(use_series, zt, 0.0)[:, None]
    acc = jnp.broadcast_to(jnp.asarray(coeffs[-1]), zt_series.shape[:1] + (k,))
    for pi in range(n_terms - 2, -1, -1):
        acc = acc * zt_series + coeffs[pi]
    qdt = q * dt
    qdt_pows = jnp.cumprod(jnp.broadcast_to(qdt[:, None], qdt.shape + (k,)), axis=1)
    series = eq[:, None] * qdt_pows * acc

    z_safe = jnp.where(use_series, 1.0, z)
    zt_safe = jnp.where(use_series, 1.0, zt)
    inv_fact_m = jnp.asarray([1.0 / math.factorial(i) for i in range(k)])
    if k > 1:
        zt_pows_m = jnp.concatenate(
            [
                jnp.ones_like(zt_safe)[:, None],
                jnp.cumprod(jnp.broadcast_to(zt_safe[:, None], zt_safe.shape + (k - 1,)), axis=1),
            ],
            axis=1,
        )
    else:
        zt_pows_m = jnp.ones_like(zt_safe)[:, None]
    partial_sums = jnp.cumsum(zt_pows_m * inv_fact_m, axis=1)
    qz = q / z_safe
    qz_pows = jnp.cumprod(jnp.broadcast_to(qz[:, None], qz.shape + (k,)), axis=1)
    difference = qz_pows * (e_lam[:, None] - eq[:, None] * partial_sums)

    return sign * jnp.where(use_series[:, None], series, difference)


def _erlang_scatter_indices(n_band, order):
    """Static (rows, cols) enumerating the forward causal transition ``phi``.

    Layout matches ``ErlangResponseDHOQS``: states ``[fast(B), slow(B),
    chains(B*k)]``; ``transition_matrix`` returns the usual column-state
    transition ``phi``.
    """
    B, k = int(n_band), int(order)
    n0 = 2 * B
    rows, cols = [], []
    for b in range(B):
        rows.append(b), cols.append(b)
    for b in range(B):
        rows.append(B + b), cols.append(B + b)
    # phi[chain_i, chain_j] (lower-tri Toeplitz incl. diagonal)
    for b in range(B):
        for i in range(k):
            for j in range(i + 1):
                rows.append(n0 + b * k + i), cols.append(n0 + b * k + j)
    # phi[chain_i, driver_b] columns
    for b in range(B):
        for i in range(k):
            rows.append(n0 + b * k + i), cols.append(b)
    for b in range(B):
        for i in range(k):
            rows.append(n0 + b * k + i), cols.append(B + b)
    return np.asarray(rows), np.asarray(cols)


def erlang_transitions(kernel, dt):
    """Vectorized ``vmap(kernel.transition_matrix)`` for the Erlang kernel."""
    B = int(kernel.tau_fast.shape[0])
    k = int(kernel.order)
    D = 2 * B + B * k
    base = kernel._base()
    tau_fast, tau_slow = base._ordered_taus()
    obs_scale = base._obs_scale()
    q = kernel._response_rates()

    dt = jnp.asarray(dt)
    e_fast = jnp.exp(-dt[:, None] / tau_fast)  # (N, B)
    e_slow = jnp.exp(-dt[:, None] / tau_slow)
    eq = jnp.exp(-q * dt[:, None])

    # chain lower-tri Toeplitz entries eq * (q dt)^d / d!, d = 0..k-1
    qdt = q * dt[:, None]
    toep = [eq]
    fact = 1.0
    for d_ in range(1, k):
        fact *= d_
        toep.append(eq * qdt ** d_ / fact)
    toep = jnp.stack(toep, axis=-1)  # (N, B, k)

    cols_f = jax.vmap(
        lambda dtk, ef, eqk: _driver_to_chain_columns_horner(dtk, 1.0 / tau_fast, ef, q, eqk, +1.0, k)
    )(dt, e_fast, eq)
    cols_s = jax.vmap(
        lambda dtk, es, eqk: _driver_to_chain_columns_horner(dtk, 1.0 / tau_slow, es, q, eqk, -1.0, k)
    )(dt, e_slow, eq)
    cols_f = obs_scale[None, :, None] * cols_f
    cols_s = obs_scale[None, :, None] * cols_s

    chain_vals = jnp.stack(
        [toep[:, b, i - j] for b in range(B) for i in range(k) for j in range(i + 1)],
        axis=1,
    )
    values = jnp.concatenate(
        [e_fast, e_slow, chain_vals, cols_f.reshape(dt.shape[0], -1), cols_s.reshape(dt.shape[0], -1)],
        axis=1,
    )
    rows, cols = _erlang_scatter_indices(B, k)
    out = jnp.zeros((dt.shape[0], D, D), dtype=values.dtype)
    return out.at[:, rows, cols].set(values)


def _sym(m):
    return 0.5 * (m + m.T)


def _scan_loglike_fwd_impl(d, p, q, a, r, ties):
    D = p.shape[1]
    F0 = jnp.zeros((D, D), dtype=d.dtype)
    f0 = jnp.zeros((D,), dtype=d.dtype)

    def impl(carry, data):
        F, f = carry
        d_k, p_k, q_k, a_k, r_k, tie_k = data
        Fp = F @ p_k
        s = d_k - p_k @ Fp
        c = jnp.sqrt(s)

        def full(_):
            u = q_k - a_k @ Fp
            w = u / c
            F_next = a_k @ (F @ a_k.T) + jnp.outer(w, w)
            return w, F_next, a_k @ f

        def tie(_):
            u = q_k - Fp
            w = u / c
            return w, F + jnp.outer(w, w), f

        w, F_next, af = jax.lax.cond(tie_k, tie, full, None)
        x = (r_k - p_k @ f) / c
        f_next = af + w * x
        return (F_next, f_next), (jnp.log(c), x, w, F, f)

    (_, _), (logc, xs, ws, Fs, fs) = jax.lax.scan(impl, (F0, f0), (d, p, q, a, r, ties))
    value = -jnp.sum(logc) - 0.5 * jnp.sum(jnp.square(xs))
    return value, (logc, xs, ws, Fs, fs)


@jax.custom_vjp
def scan_loglike(d, p, q, a, r, ties):
    """``-sum(log c) - 0.5 * ||L^{-1} r||^2`` for the quasisep factorization.

    Args:
        d (n,): diagonal (kernel variance + noise).
        p, q (n, D): quasisep generator vectors (tinygp convention).
        a (n, D, D): transition matrices between consecutive points.
        r (n,): residuals (y minus mean).
        ties (n,) bool: True where the time gap to the previous point is
            exactly zero (transition exactly the identity).
    """
    value, _ = _scan_loglike_fwd_impl(d, p, q, a, r, ties)
    return value


def _scan_loglike_fwd(d, p, q, a, r, ties):
    value, res = _scan_loglike_fwd_impl(d, p, q, a, r, ties)
    return value, (d, p, q, a, r, ties, res)


def _scan_loglike_bwd(saved, g):
    d, p, q, a, r, ties, (logc, xs, ws, Fs, fs) = saved
    cs = jnp.exp(logc)
    D = p.shape[1]

    # lF is kept symmetric throughout: every primal F perturbation is
    # symmetric, so projecting the cotangent onto the symmetric subspace is
    # exact and enables the 3-gemm reverse step below.
    def impl(carry, data):
        lF, lf = carry
        p_k, a_k, tie_k, c, x, w, F, f = data
        Fp = F @ p_k

        x_bar = -x + lf @ w
        w_bar = 2.0 * (lF @ w) + lf * x
        c_bar = -1.0 / c - (w_bar @ w + x_bar * x) / c
        s_bar = 0.5 * c_bar / c
        u_bar = w_bar / c
        v_bar = x_bar / c

        def full(_):
            H = lF @ a_k                              # gemm 1
            Fp_bar = -(a_k.T @ u_bar) - s_bar * p_k
            F_bar = a_k.T @ H                         # gemm 2 (symmetric)
            a_bar = 2.0 * (H @ F.T)                   # gemm 3
            a_bar = a_bar - jnp.outer(u_bar, Fp) + jnp.outer(lf, f)
            return Fp_bar, F_bar, a_bar, a_k.T @ lf

        def tie(_):
            Fp_bar = -u_bar - s_bar * p_k
            return Fp_bar, lF, jnp.zeros_like(a_k), lf

        Fp_bar, F_bar, a_bar, f_bar = jax.lax.cond(tie_k, tie, full, None)
        F_bar = F_bar + _sym(jnp.outer(Fp_bar, p_k))
        p_bar = -s_bar * Fp - v_bar * f + F @ Fp_bar
        f_bar = f_bar - v_bar * p_k
        return (F_bar, f_bar), (s_bar, p_bar, u_bar, a_bar, v_bar)

    lF0 = jnp.zeros((D, D), dtype=d.dtype)
    lf0 = jnp.zeros((D,), dtype=d.dtype)
    (_, _), (d_bar, p_bar, q_bar, a_bar, r_bar) = jax.lax.scan(
        impl,
        (lF0, lf0),
        (p, a, ties, cs, xs, ws, Fs, fs),
        reverse=True,
    )
    return (g * d_bar, g * p_bar, g * q_bar, g * a_bar, g * r_bar, None)


scan_loglike.defvjp(_scan_loglike_fwd, _scan_loglike_bwd)


def fused_log_probability(kernel, X, diag, resid, *, sort_time=None):
    """Exact quasisep GP log-likelihood in one fused pass.

    Equivalent to ``GaussianProcess(kernel, X, diag=diag,
    assume_sorted=True).log_probability(resid + mean)`` for a zero-mean GP on
    ``resid``; pass ``resid = y - mean``.
    """
    Pinf = kernel.stationary_covariance()
    if sort_time is None:
        t = X[0] if isinstance(X, (tuple, list)) else kernel.coord_to_sortable(X)
    else:
        t = sort_time
    t = jnp.asarray(t)
    dt = jnp.diff(t, prepend=t[0])

    if hasattr(kernel, "_driver_to_chain_columns"):
        a = erlang_transitions(kernel, dt)
    else:
        Xp = jax.tree_util.tree_map(lambda v: jnp.append(v[0], v[:-1]), X)
        a = jax.vmap(kernel.transition_matrix)(Xp, X)

    h = jax.vmap(kernel.observation_model)(X)
    ph = h @ Pinf
    d = jnp.sum(ph * h, axis=1) + diag
    # Direct causal QSM generators. For i > j,
    # K_ij = h_i F_i ... F_{j+1} Pinf h_j. This differs from tinygp's
    # reversible-process default and is required for delayed responses.
    p = jax.vmap(jnp.dot)(h, a)
    q = h @ Pinf.T

    ties = jnp.concatenate([jnp.zeros((1,), bool), jnp.diff(t) == 0.0])
    value = scan_loglike(d, p, q, a, resid, ties)
    n = d.shape[0]
    return value - 0.5 * n * jnp.log(2 * jnp.pi)
