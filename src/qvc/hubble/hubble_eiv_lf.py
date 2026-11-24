"""Latent-luminosity errors-in-variables AGN likelihood (``--eiv-mode latent-lf``).

Model, per AGN at redshift z (x = (s, t) are log sigma_UV and log tau minus the
fit pivots; L is the true dereddened absolute magnitude):

    L                 ~ psi(L + mu(z) + Delta_att | z)         parent LF (the
                                                               completeness mock's LF)
    t | L, z          ~ N(tau0 + tau_L (L - L_ref) + tau_z zeta, s_t^2)
    s | L, t          ~ N((L - M0 - beta t) / alpha, eta^2)    scatter in x at fixed L
    m - mu(z) | L     ~ N(L, sigma_M^2 + e_m^2 + sigma_lens^2 + sigma_muz^2)
                                                               scatter in M at fixed x
    x_obs | x         ~ N(x, k^2 C_i)                          light-curve covariance

with zeta = log10((1 + z) / 2.5), sigma_M = exp(log_f) and psi the parent
luminosity function expressed in the selection (attenuated) magnitude
u = L + mu(z) + Delta_att.  The selection depends on (m_att, z) only, so

    p(m_att, x_obs | z, selected) = p(m_att | z, selected) * p(x_obs | m_att, z).

The first factor contains none of the relation parameters and, because psi is
fixed in apparent magnitude at each z, none of the cosmological parameters
either (only sigma_M through a negligible smoothing).  The likelihood used here
is therefore the second factor, in which the completeness map cancels exactly:

    p(x_obs | m, z) = int dL p(L | m, z) N(x_obs; xbar(L), Sigma + k^2 C_i)
    p(L | m, z)     propto psi(u) N(u; m_att, v_i),   u = m_att + (L - (m - mu))

Both integrals over L are evaluated with K-node Gauss-Hermite quadrature on
the Gaussian N(u; m_att, v_i); the dependence on x is analytic.  xbar(L) and
Sigma follow from the two conditionals above:

    tbar = tau0 + tau_L (L - L_ref) + tau_z zeta,   sbar = (L - M0 - beta tbar) / alpha
    Sigma = [[(beta/alpha)^2 s_t^2 + eta^2, -(beta/alpha) s_t^2],
             [-(beta/alpha) s_t^2,           s_t^2]]

The limits eta -> 0 (scatter in M at fixed x only) and sigma_M -> 0 (scatter
in x at fixed L only) are both contained.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import logsumexp

EIV_MODE_NONE = "none"
EIV_MODE_LATENT_LF = "latent-lf"
EIV_MODES = (EIV_MODE_NONE, EIV_MODE_LATENT_LF)

EIV_LOG_ETA_PARAM = "eiv_log_eta"
EIV_TAU0_PARAM = "eiv_tau0"
EIV_TAU_LUM_PARAM = "eiv_tau_lum"
EIV_TAU_Z_PARAM = "eiv_tau_z"
EIV_LOG_TAU_STD_PARAM = "eiv_log_tau_std"
EIV_LOG_ERR_SCALE_PARAM = "eiv_log_err_scale"

EIV_HYPER_PRIORS = (
    (EIV_LOG_ETA_PARAM, (float(np.log(0.003)), float(np.log(0.5)))),
    (EIV_TAU0_PARAM, (-2.0, 2.0)),
    (EIV_TAU_LUM_PARAM, (-1.0, 1.0)),
    (EIV_TAU_Z_PARAM, (-5.0, 5.0)),
    (EIV_LOG_TAU_STD_PARAM, (float(np.log(0.01)), float(np.log(2.0)))),
)
EIV_ERR_SCALE_PRIOR = (EIV_LOG_ERR_SCALE_PARAM, (float(np.log(0.25)), float(np.log(4.0))))
# In latent-lf mode log_f is the luminosity scatter at fixed x, which may vanish.
EIV_LOG_F_PRIOR = (float(np.log(0.01)), float(np.log(1.5)))


def eiv_log_f_prior(*, sigma_m_fixed=None, sigma_m_max=None):
    """Bounds of log_f (sigma_M = exp(log_f)) in latent-lf mode.

    Default: :data:`EIV_LOG_F_PRIOR`. ``sigma_m_max`` lowers the upper bound
    to ln(sigma_m_max), the bound the diagnostic MAP fits imposed from outside
    (``HI[log_f] = ln 0.1``). ``sigma_m_fixed`` fixes sigma_M: the bounds
    collapse to (ln v, ln v), so the prior transform is constant and the
    likelihood rejects any other value (the convention of ``fixed_h0``).
    Both given: the fixed value must not exceed the bound.
    """
    low, high = EIV_LOG_F_PRIOR
    if sigma_m_max is not None:
        sigma_m_max = float(sigma_m_max)
        if not np.isfinite(sigma_m_max) or not np.exp(low) < sigma_m_max <= np.exp(high):
            raise ValueError(
                f"The sigma_M upper bound must lie in ({np.exp(low):g}, {np.exp(high):g}] mag "
                f"(the default latent-lf bounds), got {sigma_m_max!r}."
            )
    if sigma_m_fixed is not None:
        sigma_m_fixed = float(sigma_m_fixed)
        if not (
            np.isfinite(sigma_m_fixed)
            and sigma_m_fixed > 0.0
            and low - 1e-12 <= np.log(sigma_m_fixed) <= high + 1e-12
        ):
            raise ValueError(
                f"The fixed sigma_M must lie in [{np.exp(low):g}, {np.exp(high):g}] mag "
                f"(the default latent-lf bounds), got {sigma_m_fixed!r}."
            )
        if sigma_m_max is not None and sigma_m_fixed > sigma_m_max:
            raise ValueError(
                f"Inconsistent sigma_M settings: fixed value {sigma_m_fixed:g} mag exceeds the "
                f"upper bound {sigma_m_max:g} mag."
            )
        value = float(np.log(sigma_m_fixed))
        return (value, value)
    if sigma_m_max is not None:
        return (low, float(np.log(sigma_m_max)))
    return EIV_LOG_F_PRIOR
EIV_LATEX_LABELS = {
    EIV_LOG_ETA_PARAM: r"$\ln\eta_\sigma$",
    EIV_TAU0_PARAM: r"$\tau_0$",
    EIV_TAU_LUM_PARAM: r"$\tau_L$",
    EIV_TAU_Z_PARAM: r"$\tau_z$",
    EIV_LOG_TAU_STD_PARAM: r"$\ln s_\tau$",
    EIV_LOG_ERR_SCALE_PARAM: r"$\ln k_{\rm err}$",
}

EIV_TAU_LUM_PIVOT = -25.0  # L_ref (mag)
EIV_Z_PIVOT = 1.5  # zeta = log10((1+z)/(1+EIV_Z_PIVOT))
EIV_DEFAULT_NODES = 24
EIV_DEFAULT_LF_MODEL = "wang2026_type1_lade_a"
EIV_ALPHA_NU_MEAN = -0.5  # parent continuum slope of the completeness mock
EIV_ALPHA_NU_SIGMA = 0.3
EIV_U_GRID = (10.0, 32.0, 0.05)  # attenuated apparent magnitude table (min, max, step)

EIV_LNPSI_COL = "eiv_lnpsi_table"
EIV_U0_COL = "eiv_lnpsi_u0"
EIV_DU_COL = "eiv_lnpsi_du"
EIV_Z0_COL = "eiv_lnpsi_z0"
EIV_DZ_COL = "eiv_lnpsi_dz"

_LOG_2PI = float(np.log(2.0 * np.pi))


def normalize_eiv_mode(mode):
    value = EIV_MODE_NONE if mode is None else str(mode).strip().lower().replace("_", "-")
    if value not in EIV_MODES:
        raise ValueError(f"Unknown eiv_mode {mode!r}; expected one of {EIV_MODES}.")
    return value


def eiv_is_active(mode):
    return normalize_eiv_mode(mode) != EIV_MODE_NONE


def eiv_hyper_priors(*, free_error_scale=False):
    """Ordered (label, bounds) of the extra latent-lf hyperparameters."""
    priors = list(EIV_HYPER_PRIORS)
    if free_error_scale:
        priors.append(EIV_ERR_SCALE_PRIOR)
    return priors


def eiv_hyper_labels(*, free_error_scale=False):
    return [label for label, _ in eiv_hyper_priors(free_error_scale=free_error_scale)]


# ---------------------------------------------------------------------------
# Parent luminosity function in the selection magnitude
# ---------------------------------------------------------------------------
def lf_log_density_grid(
    lf_grid,
    z_nodes,
    u_nodes,
    *,
    mock_cosmology,
    alpha_nu_mean=EIV_ALPHA_NU_MEAN,
    alpha_nu_sigma=EIV_ALPHA_NU_SIGMA,
    n_alpha=21,
):
    """ln psi(u | z) on a (z, u) grid, up to a z-dependent constant.

    psi is the density in apparent (attenuated) m_2500 = M_2500 + DM_mock(z) of
    the parent LF that generates the completeness mock: the native-magnitude LF
    convolved with the Gaussian distribution of continuum slopes used by the
    mock generator.  Its shape at fixed z does not depend on the fitted
    cosmology.  Values outside the LF magnitude grid are extrapolated linearly
    in ln psi from the last two finite values.
    """
    z_nodes = np.asarray(z_nodes, dtype=float)
    u_nodes = np.asarray(u_nodes, dtype=float)
    native = np.asarray(lf_grid.native_magnitude_grid, dtype=float)
    interp = RegularGridInterpolator(
        (np.asarray(lf_grid.redshift_grid, dtype=float), native),
        np.asarray(lf_grid.phi_log10, dtype=float),
        bounds_error=False,
        fill_value=None,
    )
    g, w = np.polynomial.hermite_e.hermegauss(int(n_alpha))
    w = w / w.sum()
    colour = 2.5 * np.log10(2500.0 / float(lf_grid.reference_wavelength_angstrom))
    offset = float(lf_grid.native_to_monochromatic_ab_offset)
    dm = mock_cosmology.distmod(z_nodes).value
    out = np.empty((z_nodes.size, u_nodes.size))
    for j, (zj, dmj) in enumerate(zip(z_nodes, dm)):
        m_abs = u_nodes - dmj  # M_2500
        acc = np.zeros(u_nodes.size)
        for gq, wq in zip(g, w):
            alpha = alpha_nu_mean + alpha_nu_sigma * gq
            m_native = m_abs - offset - colour * alpha
            pts = np.column_stack([np.full(u_nodes.size, zj), np.clip(m_native, native[0], native[-1])])
            lp = interp(pts)
            # linear extrapolation of log10 phi outside the native grid
            lo = interp([[zj, native[0]], [zj, native[1]]])
            hi = interp([[zj, native[-2]], [zj, native[-1]]])
            slope_lo = (lo[1] - lo[0]) / (native[1] - native[0])
            slope_hi = (hi[1] - hi[0]) / (native[-1] - native[-2])
            lp = np.where(m_native < native[0], lo[0] + slope_lo * (m_native - native[0]), lp)
            lp = np.where(m_native > native[-1], hi[1] + slope_hi * (m_native - native[-1]), lp)
            acc += wq * 10.0 ** np.clip(lp, -300.0, 300.0)
        out[j] = np.log(np.clip(acc, 1e-300, None))
    return out


def build_eiv_lf_table(
    z,
    *,
    lf_model=EIV_DEFAULT_LF_MODEL,
    lf_grid=None,
    mock_cosmology=None,
    u_grid=EIV_U_GRID,
    z_step=0.01,
    alpha_nu_mean=EIV_ALPHA_NU_MEAN,
    alpha_nu_sigma=EIV_ALPHA_NU_SIGMA,
):
    """Compact (z, u) table of ln psi(u | z) covering the redshifts ``z``.

    Returns a dict with EIV_LNPSI_COL (n_z, n_u; float32, each row shifted to
    a maximum of zero, which cancels in the likelihood), EIV_Z0_COL,
    EIV_DZ_COL, EIV_U0_COL and EIV_DU_COL that is merged into ``agn_data``;
    the likelihood interpolates it bilinearly.  The table is kept small
    (about 0.5 MB) because ``agn_data`` is pickled to the sampler workers.
    """
    from qvc.hubble.completeness_mock_catalog import COSMO, build_completeness_lf

    z = np.asarray(z, dtype=float)
    mock_cosmology = COSMO if mock_cosmology is None else mock_cosmology
    u0, u1, du = (float(v) for v in u_grid)
    u_nodes = np.arange(u0, u1 + 0.5 * du, du)
    z_lo = max(0.0, np.floor((z.min() - 0.05) / z_step) * z_step)
    z_hi = np.ceil((z.max() + 0.05) / z_step) * z_step
    z_nodes = np.arange(z_lo, z_hi + 0.5 * z_step, z_step)
    if lf_grid is None:
        lf_grid = build_completeness_lf(
            lf_model,
            z_range=(max(z_nodes[0], 0.05), z_nodes[-1] + 0.05),
            target_cosmology=mock_cosmology,
        )
    table = lf_log_density_grid(
        lf_grid,
        z_nodes,
        u_nodes,
        mock_cosmology=mock_cosmology,
        alpha_nu_mean=alpha_nu_mean,
        alpha_nu_sigma=alpha_nu_sigma,
    )
    table = (table - table.max(axis=1, keepdims=True)).astype(np.float32)
    return {
        EIV_LNPSI_COL: table,
        EIV_Z0_COL: float(z_nodes[0]),
        EIV_DZ_COL: float(z_step),
        EIV_U0_COL: u0,
        EIV_DU_COL: du,
    }


def eiv_lf_row_weights(agn_data, z):
    """Row index j and weight f for linear interpolation of the table in z."""
    nz = np.asarray(agn_data[EIV_LNPSI_COL]).shape[0]
    pos = np.clip(
        (np.asarray(z, dtype=float) - float(agn_data[EIV_Z0_COL])) / float(agn_data[EIV_DZ_COL]),
        0.0,
        nz - 1.000001,
    )
    j = np.floor(pos).astype(np.int64)
    return j, pos - j


def _interp_rows(table, u0, du, u, rows=None):
    """Linear interpolation in u of ``table`` at ``u[i, :]``.

    ``rows=None``: row i of a per-object table.  ``rows=(j, f)``: bilinear
    interpolation between rows j[i] and j[i] + 1 of a (z, u) table.
    """
    nrow, nu = table.shape
    pos = np.clip((u - u0) / du, 0.0, nu - 1.000001)
    k = pos.astype(np.int64)
    fu = pos - k
    flat = table.reshape(-1)
    if rows is None:
        base = (np.arange(nrow, dtype=np.int64) * nu)[:, None]
        a = flat[base + k].astype(float)
        b = flat[base + k + 1].astype(float)
        return a + fu * (b - a)
    j, fz = rows
    base0 = (j * nu)[:, None]
    base1 = (np.minimum(j + 1, nrow - 1) * nu)[:, None]
    a0 = flat[base0 + k].astype(float)
    b0 = flat[base0 + k + 1].astype(float)
    a1 = flat[base1 + k].astype(float)
    b1 = flat[base1 + k + 1].astype(float)
    v0 = a0 + fu * (b0 - a0)
    v1 = a1 + fu * (b1 - a1)
    return v0 + fz[:, None] * (v1 - v0)


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------
_GH_CACHE = {}
EIV_DISTANCE_GRID_STEP = 0.002


def eiv_distance_terms(z, z_err, cosmo, *, lens_amp=0.06, lens_z_ref=1.0, lens_power=1.5):
    """mu(z), sigma_lens(z) and sigma_mu(z_err) from one comoving-distance call.

    Same definitions as hubble_likelihood.sigma_lens_from_dc and
    sigma_mu_from_z_err (with no mean-redshift term), but the comoving
    distance is evaluated once on a fine redshift grid and interpolated
    linearly (relative error < 1e-7), which removes most of astropy's
    per-call overhead.  Flat cosmologies only (all models used here).
    """
    if abs(float(getattr(cosmo, "Ok0", 0.0))) > 1e-12:
        raise ValueError("eiv_distance_terms assumes a flat cosmology.")
    z = np.asarray(z, dtype=float)
    z_err = np.asarray(z_err, dtype=float)
    has_err = np.isfinite(z_err) & (z_err > 0.0)
    err = np.where(has_err, z_err, 0.0)
    z_lo = np.maximum(z - err, 1e-8)
    z_hi = np.maximum(z + err, z_lo + 1e-8)
    lo = max(1e-8, min(float(z_lo.min()), lens_z_ref))
    hi = max(float(z_hi.max()), lens_z_ref)
    n = int(np.ceil((hi - lo) / EIV_DISTANCE_GRID_STEP)) + 2
    grid = np.linspace(lo, hi, n)
    dc = cosmo.comoving_distance(grid).to_value("Mpc")

    def distmod(zz):
        return 5.0 * np.log10((1.0 + zz) * np.interp(zz, grid, dc)) + 25.0

    mu = distmod(z)
    ratio = np.clip(np.interp(z, grid, dc) / np.interp(lens_z_ref, grid, dc), 0.0, None)
    sigma_lens = lens_amp * ratio**lens_power
    sigma_mu_z = np.where(has_err, 0.5 * np.abs(distmod(z_hi) - distmod(z_lo)), 0.0)
    return mu, sigma_lens, sigma_mu_z


def _logsumexp_rows(a):
    m = np.max(a, axis=1)
    return m + np.log(np.sum(np.exp(a - m[:, None]), axis=1))


def _gh_nodes(k):
    k = int(k)
    if k not in _GH_CACHE:
        g, w = np.polynomial.hermite_e.hermegauss(k)
        _GH_CACHE[k] = (g, np.log(w / w.sum()))
    return _GH_CACHE[k]


def eiv_lf_conditional(
    params,
    *,
    dm_obs,
    m_att,
    v_mag,
    s_obs,
    t_obs,
    var_s,
    var_t,
    cov_st,
    zeta,
    lnpsi_table,
    u0,
    du,
    n_nodes=EIV_DEFAULT_NODES,
    return_details=False,
    lnpsi_rows=None,
):
    """Per-object ln p(x_obs | m, z) of the latent-lf model.

    dm_obs = m_dered - mu(z) (observed absolute magnitude), m_att is the
    selection magnitude, v_mag the total variance between m - mu and L
    (sigma_M^2 included), (s_obs, t_obs) the regressors minus the pivots and
    (var_s, var_t, cov_st) their (unscaled) measurement covariance.
    ``lnpsi_table`` is a per-object table (one row per object) or, with
    ``lnpsi_rows=(j, f)`` from :func:`eiv_lf_row_weights`, the compact (z, u)
    table.
    """
    M0 = float(params["M0_agn"])
    alpha = float(params["alpha_agn"])
    beta = float(params["beta_agn"])
    eta2 = np.exp(2.0 * float(params[EIV_LOG_ETA_PARAM]))
    st2 = np.exp(2.0 * float(params[EIV_LOG_TAU_STD_PARAM]))
    k2 = np.exp(2.0 * float(params.get(EIV_LOG_ERR_SCALE_PARAM, 0.0)))
    tau0 = float(params[EIV_TAU0_PARAM])
    tau_l = float(params[EIV_TAU_LUM_PARAM])
    tau_z = float(params[EIV_TAU_Z_PARAM])

    g, lnw = _gh_nodes(n_nodes)
    sd = np.sqrt(v_mag)
    offs = sd[:, None] * g[None, :]  # (N, K)
    lnprior = lnw[None, :] + _interp_rows(lnpsi_table, u0, du, m_att[:, None] + offs, rows=lnpsi_rows)
    L = dm_obs[:, None] + offs
    tbar = tau0 + tau_l * (L - EIV_TAU_LUM_PIVOT) + (tau_z * zeta)[:, None]
    sbar = (L - M0 - beta * tbar) / alpha
    ratio = beta / alpha
    s_ss = ratio * ratio * st2 + eta2 + k2 * var_s
    s_tt = st2 + k2 * var_t
    s_st = -ratio * st2 + k2 * cov_st
    det = s_ss * s_tt - s_st * s_st
    if np.any(~(det > 0.0)):
        raise ValueError("Non-positive-definite latent-lf regressor covariance.")
    rs = s_obs[:, None] - sbar
    rt = t_obs[:, None] - tbar
    quad = (s_tt[:, None] * rs * rs - 2.0 * s_st[:, None] * rs * rt + s_ss[:, None] * rt * rt) / det[:, None]
    lnx = -0.5 * quad - 0.5 * np.log(det)[:, None] - _LOG_2PI
    a = lnprior + lnx
    log_num = _logsumexp_rows(a)
    log_den = _logsumexp_rows(lnprior)
    ll = log_num - log_den
    if not return_details:
        return ll
    q = np.exp(lnprior - log_den[:, None])  # p(L | m) weights
    p = np.exp(a - log_num[:, None])  # p(L | m, x_obs) weights
    e_l_m = np.sum(q * L, axis=1)
    var_l_m = np.clip(np.sum(q * L * L, axis=1) - e_l_m**2, 0.0, None)
    e_l_mx = np.sum(p * L, axis=1)
    var_l_mx = np.clip(np.sum(p * L * L, axis=1) - e_l_mx**2, 0.0, None)
    return ll, dict(
        E_L_given_m=e_l_m,
        var_L_given_m=var_l_m,
        E_L_given_mx=e_l_mx,
        var_L_given_mx=var_l_mx,
    )


def _eiv_inputs(params, agn_data, *, mu_cosmo, sigma_lens, sigma_mu_z, pivots, include_sigma_m=True):
    """Arrays fed to :func:`eiv_lf_conditional` for one parameter point."""
    from qvc.hubble.hubble_completeness_refactored import COMPLETENESS_MAG_COL

    for key in (EIV_LNPSI_COL, EIV_Z0_COL, EIV_DZ_COL, EIV_U0_COL, EIV_DU_COL):
        if key not in agn_data:
            raise KeyError(
                f"latent-lf EIV mode requires the precomputed parent-LF table {key!r} in agn_data "
                "(see qvc.hubble.hubble_eiv_lf.build_eiv_lf_table)."
            )
    if COMPLETENESS_MAG_COL not in agn_data:
        raise KeyError(
            f"latent-lf EIV mode requires the selection magnitude {COMPLETENESS_MAG_COL!r} "
            "(the parent LF is defined in that magnitude); run with completeness enabled."
        )
    m = np.asarray(agn_data["apparent_mag_2500"], dtype=float)
    m_err = np.asarray(agn_data["apparent_mag_2500_err"], dtype=float)
    m_att = np.asarray(agn_data[COMPLETENESS_MAG_COL], dtype=float)
    z = np.asarray(agn_data["z"], dtype=float)
    sigma_m2 = np.exp(2.0 * float(params["log_f"])) if include_sigma_m else 0.0
    v_mag = sigma_m2 + m_err**2 + np.asarray(sigma_lens) ** 2 + np.asarray(sigma_mu_z) ** 2
    es = np.asarray(agn_data["log_sigma_uv_std_psd"], dtype=float)
    et = np.asarray(agn_data["log_tau_uv_rf_std_psd"], dtype=float)
    return dict(
        dm_obs=m - np.asarray(mu_cosmo, dtype=float),
        m_att=m_att,
        v_mag=v_mag,
        s_obs=np.asarray(agn_data["log_sigma_uv"], dtype=float) - float(pivots[0]),
        t_obs=np.asarray(agn_data["log_tau_uv_rf"], dtype=float) - float(pivots[1]),
        var_s=es**2,
        var_t=et**2,
        cov_st=np.asarray(agn_data["log_sigma_uv_log_tau_uv_rf_cov_psd"], dtype=float),
        zeta=np.log10((1.0 + z) / (1.0 + EIV_Z_PIVOT)),
        lnpsi_table=np.asarray(agn_data[EIV_LNPSI_COL]),
        lnpsi_rows=eiv_lf_row_weights(agn_data, z),
        u0=float(agn_data[EIV_U0_COL]),
        du=float(agn_data[EIV_DU_COL]),
    )


def eiv_lf_agn_loglike(params, agn_data, *, mu_cosmo, sigma_lens, sigma_mu_z, pivots, n_nodes=EIV_DEFAULT_NODES):
    """Sum of ln p(x_obs | m, z) and the (3, N) blob.

    Blob rows: [ones, dmi, sigma_sel] with dmi = (m - mu) - E[L | m, z] (the
    Eddington correction given the magnitude alone, so that the pipeline's
    debiased residual m - dmi - M_pred(x_obs) - mu = E[L | m] - M_pred(x_obs))
    and sigma_sel = sd(L | m, z).
    """
    alpha = float(params["alpha_agn"])
    if not np.isfinite(alpha) or abs(alpha) < 1e-3:
        return -np.inf, np.zeros((3, len(mu_cosmo)))
    inputs = _eiv_inputs(
        params, agn_data, mu_cosmo=mu_cosmo, sigma_lens=sigma_lens, sigma_mu_z=sigma_mu_z, pivots=pivots
    )
    ll, det = eiv_lf_conditional(params, n_nodes=n_nodes, return_details=True, **inputs)
    dmi = inputs["dm_obs"] - det["E_L_given_m"]
    blob = np.vstack([np.ones_like(dmi), dmi, np.sqrt(det["var_L_given_m"])])
    return float(np.sum(ll)), blob


def eiv_object_terms(params, agn_data, *, mu_cosmo, sigma_lens, sigma_mu_z, pivots, n_nodes=EIV_DEFAULT_NODES):
    """Per-object Hubble-diagram quantities of the latent-lf model at one parameter point.

    With a = (alpha, beta), C_i the light-curve covariance of (s, t) and k
    the error scale, the model implies alpha s_obs + beta t_obs | L =
    L - M0 + alpha eta eps + a.(measurement noise) (the population scatter of t
    cancels), so for M_pred = M0 + alpha s_obs + beta t_obs:

    * ``L_hat`` = E[L | m, z] (Eddington-corrected luminosity; same nodes and
      psi weights as the likelihood) and ``dmi`` = (m - mu) - L_hat;
    * ``resid_debiased`` = L_hat - M_pred (zero mean at fixed L) and
      ``resid_biased`` = (m - mu) - M_pred;
    * ``err_meas`` = sqrt(a (k^2 C_i) a + Var(L | m, z; sigma_M = 0)): light-curve,
      photometric, redshift and lensing terms only;
    * ``err_total`` = sqrt(a (k^2 C_i) a + alpha^2 eta^2 + Var(L | m, z)): the
      predictive standard deviation of the residual given m, z (Var(L | m, z)
      already contains sigma_M^2, e_m^2, sigma_lens^2 and sigma_muz^2, reshaped
      by psi, so they are not added again).
    """
    inputs = _eiv_inputs(
        params, agn_data, mu_cosmo=mu_cosmo, sigma_lens=sigma_lens, sigma_mu_z=sigma_mu_z, pivots=pivots
    )
    _, det = eiv_lf_conditional(params, n_nodes=n_nodes, return_details=True, **inputs)
    inputs_meas = _eiv_inputs(
        params, agn_data, mu_cosmo=mu_cosmo, sigma_lens=sigma_lens, sigma_mu_z=sigma_mu_z, pivots=pivots,
        include_sigma_m=False,
    )
    _, det_meas = eiv_lf_conditional(params, n_nodes=n_nodes, return_details=True, **inputs_meas)
    alpha = float(params["alpha_agn"])
    beta = float(params["beta_agn"])
    k2 = np.exp(2.0 * float(params.get(EIV_LOG_ERR_SCALE_PARAM, 0.0)))
    eta2 = np.exp(2.0 * float(params[EIV_LOG_ETA_PARAM]))
    lc_var = k2 * (
        alpha**2 * inputs["var_s"] + beta**2 * inputs["var_t"] + 2.0 * alpha * beta * inputs["cov_st"]
    )
    m_pred = float(params["M0_agn"]) + alpha * inputs["s_obs"] + beta * inputs["t_obs"]
    l_hat = det["E_L_given_m"]
    return dict(
        M_pred=m_pred,
        dm_obs=inputs["dm_obs"],
        L_hat=l_hat,
        dmi=inputs["dm_obs"] - l_hat,
        resid_debiased=l_hat - m_pred,
        resid_biased=inputs["dm_obs"] - m_pred,
        var_L_given_m=det["var_L_given_m"],
        var_L_given_m_meas=det_meas["var_L_given_m"],
        lc_var=lc_var,
        intrinsic_x_var=np.full_like(lc_var, alpha**2 * eta2),
        err_meas=np.sqrt(lc_var + det_meas["var_L_given_m"]),
        err_total=np.sqrt(lc_var + alpha**2 * eta2 + det["var_L_given_m"]),
    )
