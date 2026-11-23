"""Tests for the latent-luminosity errors-in-variables AGN likelihood."""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from qvc.hubble import hubble_eiv_lf as E
from qvc.hubble.hubble_completeness_refactored import COMPLETENESS_MAG_COL, COMPLETENESS_MAG_ERR_COL
from qvc.hubble.hubble_likelihood import log_likelihood
from qvc.hubble.hubble_model import AgnPivotContext, get_model_params

PIVOTS = (np.log10(0.2), np.log10(200.0))
TRUTH = {
    "M0_agn": -22.0,
    "alpha_agn": 7.5,
    "beta_agn": -2.7,
    "log_f": np.log(0.3),
    E.EIV_LOG_ETA_PARAM: np.log(0.08),
    E.EIV_TAU0_PARAM: 0.05,
    E.EIV_TAU_LUM_PARAM: -0.1,
    E.EIV_TAU_Z_PARAM: -0.6,
    E.EIV_LOG_TAU_STD_PARAM: np.log(0.25),
}


def _pivot_context():
    return AgnPivotContext(
        observable_names=("log_sigma_uv", "log_tau_uv_rf"),
        values=PIVOTS,
        z_range=(0.44, 3.16),
        reference_object_ids=("a",),
        rule="fixed_sigma_tau_v1",
    )


def _quadratic_table(n, u0=10.0, u1=32.0, du=0.05, slope=1.5, curv=-0.08, centre=21.0):
    u = np.arange(u0, u1 + 0.5 * du, du)
    row = slope * (u - centre) + curv * (u - centre) ** 2
    return np.tile(row, (n, 1)), u0, du, (slope, curv, centre)


def _mock(n=3000, seed=3, sigma_m=0.3, eta=0.08, slope=1.5, curv=-0.08, centre=21.0, cosmo=None):
    """Draw from the latent-lf generative model with a quadratic ln psi(u) and a flux limit."""
    rng = np.random.default_rng(seed)
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3) if cosmo is None else cosmo
    z = rng.uniform(0.5, 3.0, 20 * n)
    # u ~ psi(u) on [17, 24.4]: rejection sampling of the quadratic log density
    grid = np.linspace(17.0, 24.4, 2000)
    lp = slope * (grid - centre) + curv * (grid - centre) ** 2
    cdf = np.cumsum(np.exp(lp - lp.max()))
    cdf /= cdf[-1]
    u = np.interp(rng.random(z.size), cdf, grid)
    mu = cosmo.distmod(z).value
    off = rng.normal(0.2, 0.05, z.size)
    L = u - mu - off
    p = TRUTH
    zeta = np.log10((1 + z) / 2.5)
    t = p[E.EIV_TAU0_PARAM] + p[E.EIV_TAU_LUM_PARAM] * (L - E.EIV_TAU_LUM_PIVOT) + p[E.EIV_TAU_Z_PARAM] * zeta
    t = t + rng.normal(0, np.exp(p[E.EIV_LOG_TAU_STD_PARAM]), z.size)
    s = (L - p["M0_agn"] - p["beta_agn"] * t) / p["alpha_agn"] + rng.normal(0, eta, z.size)
    m_err = np.full(z.size, 0.02)
    m = L + mu + rng.normal(0, sigma_m, z.size) + rng.normal(0, 1, z.size) * m_err
    m_att = m + off
    keep = np.flatnonzero(m_att < rng.uniform(22.5, 23.5, z.size))[:n]  # magnitude-only selection
    assert keep.size == n
    es = np.full(keep.size, 0.11)
    et = np.full(keep.size, 0.15)
    cv = 0.86 * es * et
    g1, g2 = rng.normal(size=(2, keep.size))
    s_obs = s[keep] + es * g1
    t_obs = t[keep] + cv / es * g1 + np.sqrt(et**2 - (cv / es) ** 2) * g2
    data = {
        "object_id": np.arange(keep.size).astype(str),
        "z": z[keep],
        "z_err": np.zeros(keep.size),
        "apparent_mag_2500": m[keep],
        "apparent_mag_2500_err": m_err[keep],
        COMPLETENESS_MAG_COL: m_att[keep],
        COMPLETENESS_MAG_ERR_COL: m_err[keep],
        "log_sigma_uv": s_obs + PIVOTS[0],
        "log_tau_uv_rf": t_obs + PIVOTS[1],
        "log_sigma_uv_std_psd": es,
        "log_tau_uv_rf_std_psd": et,
        "log_sigma_uv_log_tau_uv_rf_cov_psd": cv,
    }
    # compact (z, u) table: the quadratic ln psi is redshift independent here
    table, u0, du, _ = _quadratic_table(2, slope=slope, curv=curv, centre=centre)
    data.update({E.EIV_LNPSI_COL: table.astype(np.float32), E.EIV_Z0_COL: 0.0, E.EIV_DZ_COL: 5.0,
                 E.EIV_U0_COL: u0, E.EIV_DU_COL: du})
    return data


def _theta(params, cosmo_values, labels):
    full = dict(params)
    full.update(cosmo_values)
    return np.array([full[k] for k in labels])


def test_default_model_params_unchanged_and_eiv_labels_appended():
    base = get_model_params("FlatLambdaCDM", only_agn=True, prior_profile="centered_lcdm_om01")
    eiv = get_model_params(
        "FlatLambdaCDM", only_agn=True, prior_profile="centered_lcdm_om01", eiv_mode="latent-lf"
    )
    assert base[1] == ["M0_agn", "alpha_agn", "beta_agn", "log_f", "H0", "Om0"]
    assert eiv[1][: len(base[1])] == base[1]
    assert eiv[1][len(base[1]):] == E.eiv_hyper_labels()
    assert eiv[0]["log_f"] == E.EIV_LOG_F_PRIOR
    assert base[0]["log_f"] != E.EIV_LOG_F_PRIOR
    free = get_model_params("FlatLambdaCDM", only_agn=True, eiv_mode="latent-lf", eiv_free_error_scale=True)
    assert free[1][-1] == E.EIV_LOG_ERR_SCALE_PARAM
    with pytest.raises(ValueError):
        E.normalize_eiv_mode("hierarchical")


def _x_marginal(p, L, zeta, s_obs, t_obs, C):
    """Analytic N(x_obs; xbar(L), Sigma + C) of the latent-lf model."""
    a, b = p["alpha_agn"], p["beta_agn"]
    st2 = np.exp(2 * p[E.EIV_LOG_TAU_STD_PARAM])
    eta2 = np.exp(2 * p[E.EIV_LOG_ETA_PARAM])
    tbar = p[E.EIV_TAU0_PARAM] + p[E.EIV_TAU_LUM_PARAM] * (L - E.EIV_TAU_LUM_PIVOT) + p[E.EIV_TAU_Z_PARAM] * zeta
    sbar = (L - p["M0_agn"] - b * tbar) / a
    S = np.array([[(b / a) ** 2 * st2 + eta2, -(b / a) * st2], [-(b / a) * st2, st2]]) + C
    r = np.stack([s_obs - sbar, t_obs - tbar], -1)
    q = np.einsum("...i,ij,...j->...", r, np.linalg.inv(S), r)
    return np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(np.linalg.det(S)))


def test_regressor_marginal_matches_direct_integration():
    """N(x_obs; xbar, Sigma + C) equals int N(x_obs; x, C) p(t | L) p(s | L, t) ds dt."""
    p = TRUTH
    C = np.array([[0.11**2, 0.86 * 0.11 * 0.15], [0.86 * 0.11 * 0.15, 0.15**2]])
    Ci = np.linalg.inv(C)
    st = np.exp(p[E.EIV_LOG_TAU_STD_PARAM])
    eta = np.exp(p[E.EIV_LOG_ETA_PARAM])
    for L, zeta, s_obs, t_obs in ((-25.3, 0.1, -0.3, 0.2), (-23.9, -0.15, -0.1, -0.4), (-26.8, 0.0, -0.6, 0.5)):
        tbar = p[E.EIV_TAU0_PARAM] + p[E.EIV_TAU_LUM_PARAM] * (L - E.EIV_TAU_LUM_PIVOT) + p[E.EIV_TAU_Z_PARAM] * zeta
        ts = tbar + np.linspace(-7 * st, 7 * st, 801)
        T = ts[:, None]
        sbar = (L - p["M0_agn"] - p["beta_agn"] * T) / p["alpha_agn"]
        u = np.linspace(-7, 7, 801)[None, :]
        S_ = sbar + eta * u  # (nt, ns) true s nodes, uniform in u
        dx_s = s_obs - S_
        dx_t = t_obs - T
        meas = np.exp(-0.5 * (Ci[0, 0] * dx_s**2 + 2 * Ci[0, 1] * dx_s * dx_t + Ci[1, 1] * dx_t**2)) / (
            2 * np.pi * np.sqrt(np.linalg.det(C))
        )
        inner = np.trapezoid(meas * np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi), u[0], axis=1)
        direct = np.trapezoid(inner * np.exp(-0.5 * ((ts - tbar) / st) ** 2) / (np.sqrt(2 * np.pi) * st), ts)
        assert _x_marginal(p, L, zeta, s_obs, t_obs, C) == pytest.approx(direct, rel=1e-6)


def test_conditional_matches_brute_force_integration():
    rng = np.random.default_rng(1)
    n = 5
    table, u0, du, (slope, curv, centre) = _quadratic_table(n)
    kw = dict(
        dm_obs=rng.normal(-25.0, 1.0, n),
        m_att=rng.uniform(18.0, 22.0, n),
        v_mag=rng.uniform(0.05, 0.6, n) ** 2,
        s_obs=rng.normal(0.0, 0.2, n),
        t_obs=rng.normal(0.0, 0.3, n),
        var_s=np.full(n, 0.11**2),
        var_t=np.full(n, 0.15**2),
        cov_st=np.full(n, 0.86 * 0.11 * 0.15),
        zeta=rng.uniform(-0.2, 0.2, n),
    )
    ll = E.eiv_lf_conditional(TRUTH, lnpsi_table=table, u0=u0, du=du, **kw)
    for i in range(n):
        sd = np.sqrt(kw["v_mag"][i])
        g = np.linspace(-9, 9, 4001)
        L = kw["dm_obs"][i] + sd * g
        u = kw["m_att"][i] + sd * g
        prior = np.exp(slope * (u - centre) + curv * (u - centre) ** 2 - 0.5 * g * g)
        C = np.array([[kw["var_s"][i], kw["cov_st"][i]], [kw["cov_st"][i], kw["var_t"][i]]])
        px = _x_marginal(TRUTH, L, kw["zeta"][i], kw["s_obs"][i], kw["t_obs"][i], C)
        brute = np.log(np.trapezoid(prior * px, g) / np.trapezoid(prior, g))
        assert ll[i] == pytest.approx(brute, abs=1e-4)


def test_quadrature_converges_with_nodes():
    data = _mock(n=400)
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    kw = dict(mu_cosmo=cosmo.distmod(data["z"]).value, sigma_lens=np.zeros(400), sigma_mu_z=np.zeros(400), pivots=PIVOTS)
    ll24, blob = E.eiv_lf_agn_loglike(TRUTH, data, n_nodes=24, **kw)
    ll96, _ = E.eiv_lf_agn_loglike(TRUTH, data, n_nodes=96, **kw)
    assert ll24 == pytest.approx(ll96, abs=1e-3)
    assert blob.shape == (3, 400)
    assert np.all(blob[2] > 0)


def test_flat_lf_limit_is_gaussian_inverse_regression():
    n = 50
    rng = np.random.default_rng(4)
    table = np.zeros((n, 441))
    dm = rng.normal(-25, 1, n)
    zeta = np.zeros(n)
    s_obs = rng.normal(0, 0.2, n)
    t_obs = rng.normal(0, 0.3, n)
    p = dict(TRUTH)
    p["log_f"] = np.log(1e-4)
    v = np.full(n, 1e-8)
    ll = E.eiv_lf_conditional(
        p, dm_obs=dm, m_att=np.full(n, 20.0), v_mag=v, s_obs=s_obs, t_obs=t_obs,
        var_s=np.full(n, 0.01), var_t=np.full(n, 0.02), cov_st=np.full(n, 0.005), zeta=zeta,
        lnpsi_table=table, u0=10.0, du=0.05,
    )
    a, b = p["alpha_agn"], p["beta_agn"]
    st2 = np.exp(2 * p[E.EIV_LOG_TAU_STD_PARAM])
    eta2 = np.exp(2 * p[E.EIV_LOG_ETA_PARAM])
    tbar = p[E.EIV_TAU0_PARAM] + p[E.EIV_TAU_LUM_PARAM] * (dm - E.EIV_TAU_LUM_PIVOT)
    sbar = (dm - p["M0_agn"] - b * tbar) / a
    S = np.array([[(b / a) ** 2 * st2 + eta2 + 0.01, -(b / a) * st2 + 0.005], [-(b / a) * st2 + 0.005, st2 + 0.02]])
    r = np.column_stack([s_obs - sbar, t_obs - tbar])
    expected = -0.5 * np.einsum("ni,ij,nj->n", r, np.linalg.inv(S), r) - 0.5 * np.log(np.linalg.det(S)) - np.log(2 * np.pi)
    np.testing.assert_allclose(ll, expected, atol=1e-9)


def test_log_likelihood_eiv_ignores_selection_and_prefers_truth():
    data = _mock(n=2500)
    ctx = _pivot_context()
    _, labels, _ = get_model_params(
        "FlatLambdaCDM", only_agn=True, eiv_mode="latent-lf", prior_profile="centered_lcdm_om01"
    )
    common = dict(
        agn_data=data, pantheon_data=None, _sna_L=None, _sna_Lower=None, _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM", z_pivot_agn=None, agn_pivot_context=ctx, only_agn=True,
        eiv_mode="latent-lf", prior_profile="centered_lcdm_om01",
    )
    theta = _theta(TRUTH, {"H0": 70.0, "Om0": 0.3}, labels)
    ll_none, blob = log_likelihood(theta, completeness_params=None, **common)
    ll_dummy, _ = log_likelihood(theta, completeness_params=(object(), np.linspace(16, 25, 10)), **common)
    assert np.isfinite(ll_none) and ll_none == ll_dummy
    assert blob.shape == (3, 2500)
    # The truth beats an attenuated slope and a high-Om0 cosmology.
    for change in ({"alpha_agn": 6.5}, {"alpha_agn": 8.5}):
        alt = dict(TRUTH, **change)
        assert log_likelihood(_theta(alt, {"H0": 70.0, "Om0": 0.3}, labels), completeness_params=None, **common)[0] < ll_none
    ll_high, _ = log_likelihood(_theta(TRUTH, {"H0": 70.0, "Om0": 0.95}, labels), completeness_params=None, **common)
    assert np.isfinite(ll_high) and np.isfinite(ll_none)
    assert ll_high < ll_none


def test_log_likelihood_eiv_rejects_unsupported_terms():
    data = _mock(n=100)
    _, labels, _ = get_model_params("FlatLambdaCDM", only_agn=True, eiv_mode="latent-lf")
    with pytest.raises(NotImplementedError):
        log_likelihood(
            _theta(TRUTH, {"H0": 70.0, "Om0": 0.3}, labels), agn_data=data, pantheon_data=None, _sna_L=None,
            _sna_Lower=None, _sna_LogdetCov=None, cosmo_model="FlatLambdaCDM", completeness_params=None,
            z_pivot_agn=None, agn_pivot_context=_pivot_context(), only_agn=True, eiv_mode="latent-lf",
            light_curve_uncertainty_mode="posterior-draws",
        )


def test_compact_table_bilinear_matches_per_object_rows():
    rng = np.random.default_rng(5)
    table = rng.normal(size=(7, 50))
    z = rng.uniform(0.0, 0.6, 30)
    u = rng.uniform(10.0, 12.4, (30, 4))
    data = {E.EIV_LNPSI_COL: table, E.EIV_Z0_COL: 0.0, E.EIV_DZ_COL: 0.1}
    j, f = E.eiv_lf_row_weights(data, z)
    rows = (1 - f)[:, None] * table[j] + f[:, None] * table[np.minimum(j + 1, 6)]
    np.testing.assert_allclose(
        E._interp_rows(table, 10.0, 0.05, u, rows=(j, f)), E._interp_rows(rows, 10.0, 0.05, u), atol=1e-12
    )


def test_lf_log_density_grid_recovers_exponential_slope():
    native = np.linspace(-33.0, -16.0, 341)
    zg = np.linspace(0.3, 3.3, 31)
    # log10 phi linear in M: d ln psi / du = -0.7 ln 10 for every z
    phi = -6.0 - 0.7 * (native[None, :] + 26.0) + 0.0 * zg[:, None]
    lf = SimpleNamespace(
        native_magnitude_grid=native, redshift_grid=zg, phi_log10=phi,
        reference_wavelength_angstrom=1450.0, native_to_monochromatic_ab_offset=0.0,
    )
    u = np.arange(15.0, 25.0, 0.05)
    tab = E.lf_log_density_grid(lf, np.array([1.0, 2.0]), u, mock_cosmology=FlatLambdaCDM(H0=70.0, Om0=0.3))
    np.testing.assert_allclose(np.diff(tab, axis=1) / 0.05, -0.7 * np.log(10.0), rtol=1e-6)


def test_fast_distance_terms_match_reference_functions():
    from astropy.cosmology import Flatw0waCDM
    from qvc.hubble.hubble_likelihood import sigma_lens_from_dc, sigma_mu_from_z_err

    rng = np.random.default_rng(6)
    z = rng.uniform(0.44, 3.16, 500)
    z_err = np.where(rng.random(500) < 0.2, 0.0, rng.uniform(1e-4, 5e-3, 500))
    for cosmo in (FlatLambdaCDM(H0=73.5, Om0=0.361), Flatw0waCDM(H0=70.0, Om0=0.3, w0=-0.8, wa=-0.5)):
        mu, sl, sz = E.eiv_distance_terms(z, z_err, cosmo)
        np.testing.assert_allclose(mu, cosmo.distmod(z).value, atol=2e-6)
        np.testing.assert_allclose(sl, sigma_lens_from_dc(z, cosmo), rtol=1e-5)
        np.testing.assert_allclose(sz, sigma_mu_from_z_err(z, z_err, cosmo, 0.0), atol=2e-5)


def _kw(data, cosmo=None):
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3) if cosmo is None else cosmo
    mu, sl, sz = E.eiv_distance_terms(data["z"], data["z_err"], cosmo)
    return dict(mu_cosmo=mu, sigma_lens=sl, sigma_mu_z=sz, pivots=PIVOTS)


def test_eiv_requires_selection_magnitude():
    data = _mock(n=50)
    data.pop(COMPLETENESS_MAG_COL)
    with pytest.raises(KeyError):
        E.eiv_lf_agn_loglike(TRUTH, data, **_kw(data))


def test_eiv_object_terms_residual_identities_and_errors():
    data = _mock(n=400)
    kw = _kw(data)
    terms = E.eiv_object_terms(TRUTH, data, **kw)
    _, blob = E.eiv_lf_agn_loglike(TRUTH, data, **kw)
    m = data["apparent_mag_2500"]
    np.testing.assert_allclose(terms["dmi"], blob[1], atol=1e-12)
    # pipeline convention: m - dmi - M_pred - mu is the debiased residual
    np.testing.assert_allclose(m - terms["dmi"] - terms["M_pred"] - kw["mu_cosmo"], terms["resid_debiased"], atol=1e-10)
    np.testing.assert_allclose(terms["resid_biased"] - terms["resid_debiased"], terms["dmi"], atol=1e-10)
    a, b = TRUTH["alpha_agn"], TRUTH["beta_agn"]
    lc = a**2 * data["log_sigma_uv_std_psd"] ** 2 + b**2 * data["log_tau_uv_rf_std_psd"] ** 2 + 2 * a * b * data["log_sigma_uv_log_tau_uv_rf_cov_psd"]
    np.testing.assert_allclose(terms["lc_var"], lc, rtol=1e-12)
    eta2 = np.exp(2 * TRUTH[E.EIV_LOG_ETA_PARAM])
    np.testing.assert_allclose(terms["err_total"] ** 2, lc + a**2 * eta2 + terms["var_L_given_m"], rtol=1e-12)
    np.testing.assert_allclose(terms["err_meas"] ** 2, lc + terms["var_L_given_m_meas"], rtol=1e-12)
    assert np.all(terms["var_L_given_m"] > terms["var_L_given_m_meas"])
    assert np.all(np.abs(terms["dmi"]) > 0)


def test_eiv_posterior_products_and_binned_errors(tmp_path):
    from qvc.hubble.hubble_eiv_plots import (
        binned_residual_table,
        eiv_posterior_hubble_products,
        write_eiv_hubble_products,
    )

    data = _mock(n=1500)
    rng = np.random.default_rng(9)
    _, labels, _ = get_model_params("FlatLambdaCDM", only_agn=True, prior_profile="centered_lcdm_om01")
    hyper_labels = E.eiv_hyper_labels()
    n = 40
    std = np.array([[TRUTH["M0_agn"] + rng.normal(0, 0.05), TRUTH["alpha_agn"] + rng.normal(0, 0.1),
                     TRUTH["beta_agn"], TRUTH["log_f"] + rng.normal(0, 0.2), 70.0, 0.3 + rng.normal(0, 0.02)] for _ in range(n)])
    hyp = np.array([[TRUTH[k] for k in hyper_labels] for _ in range(n)])
    prod = eiv_posterior_hubble_products(std, labels, hyp, hyper_labels, data, cosmo_model="FlatLambdaCDM", pivots=PIVOTS)
    assert prod["n_draws"] == n
    assert np.all(prod["sigma_dmi"] > 0)
    assert not np.allclose(prod["mu_obj_debiased"], prod["mu_obj_biased"])
    assert np.all(prod["err_total"] > prod["err_meas"])
    meta = {"object_id": data["object_id"], "apparent_mag_2500": data["apparent_mag_2500"]}
    csv, summary, tables = write_eiv_hubble_products(prod, meta, np.ones(1500, bool), str(tmp_path), n_params=11)
    assert (tmp_path / "hubble_diagram_eiv_debiased.png").exists() and (tmp_path / "eiv_hubble_binned_residuals.csv").exists()
    t = binned_residual_table(prod["z"], prod["resid_debiased"], prod["err_total"], edges=(0.5, 1.0, 1.5, 2.0, 3.0))
    np.testing.assert_allclose(t["median_err_boot"], t["median_err"], rtol=0.35)
    np.testing.assert_allclose(t["wmean_err_boot"], t["wmean_err"] * np.sqrt(t["chi2_red_in_bin"]), rtol=0.35)


def test_joint_sn_plus_eiv_likelihood_is_the_sum_of_its_parts():
    from qvc.hubble.hubble_likelihood import log_likelihood_pantheon_cephdist

    rng = np.random.default_rng(11)
    data = _mock(n=600)
    nsn = 60
    zsn = rng.uniform(0.02, 1.5, nsn)
    cos_true = FlatLambdaCDM(H0=72.0, Om0=0.33)
    pantheon = {
        "zHD": zsn, "zHEL": zsn, "IS_CALIBRATOR": np.zeros(nsn, bool), "CEPH_DIST": np.zeros(nsn),
        "MU_SH0ES_ERR_DIAG": np.full(nsn, 0.15),
        "m_b_corr": cos_true.distmod(zsn).value - 19.25 + rng.normal(0, 0.15, nsn),
    }
    _, labels, _ = get_model_params(
        "FlatLambdaCDM", only_agn=False, eiv_mode="latent-lf", prior_profile="centered_lcdm_om01"
    )
    assert labels[0] == "M0_sn" and labels[-len(E.eiv_hyper_labels()):] == E.eiv_hyper_labels()
    params = dict(TRUTH, M0_sn=-19.25, H0=72.0, Om0=0.33)
    theta = np.array([params[k] for k in labels])
    ll, blob = log_likelihood(
        theta, agn_data=data, pantheon_data=pantheon, _sna_L=None, _sna_Lower=None, _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM", completeness_params=None, z_pivot_agn=None, agn_pivot_context=_pivot_context(),
        only_agn=False, use_full_cov=False, eiv_mode="latent-lf", prior_profile="centered_lcdm_om01",
    )
    cosmo = FlatLambdaCDM(H0=72.0, Om0=0.33)
    ll_sn = log_likelihood_pantheon_cephdist(params, pantheon, None, None, None, cosmo, False)
    ll_agn, blob_agn = E.eiv_lf_agn_loglike(params, data, **_kw(data, cosmo))
    assert np.isfinite(ll) and np.isfinite(ll_sn) and np.isfinite(ll_agn)
    assert ll == pytest.approx(ll_sn + ll_agn, abs=1e-9)
    np.testing.assert_allclose(blob, blob_agn)
    # H0 is not absorbed by the AGN term: changing H0 at fixed M0_agn changes the AGN likelihood
    params2 = dict(params, H0=65.0)
    ll_agn2, _ = E.eiv_lf_agn_loglike(params2, data, **_kw(data, FlatLambdaCDM(H0=65.0, Om0=0.33)))
    assert abs(ll_agn2 - ll_agn) > 1.0
