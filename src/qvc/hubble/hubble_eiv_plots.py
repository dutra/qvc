"""Hubble-diagram products of the latent-lf errors-in-variables model.

Everything here is computed from the latent-lf model itself (the forward
likelihood's completeness-map debiasing has no meaning in that mode):

* per posterior draw theta and object i (see hubble_eiv_lf.eiv_object_terms):
  L_hat_i = E[L | m_i, z_i, theta], M_pred_i = M0 + alpha s_obs + beta t_obs,
  debiased residual r_i = L_hat_i - M_pred_i, biased residual (m_i - mu_i) - M_pred_i,
  object distance modulus mu_obj_i = mu(z_i; theta) + r_i = m_i + E[d | m, z] - M_pred_i;
* errors err_meas = sqrt(a k^2 C_i a + Var(L | m, z; sigma_M = 0)) and
  err_total = sqrt(a k^2 C_i a + alpha^2 eta^2 + Var(L | m, z));
* posterior summaries over (at most) ``max_draws`` evenly spaced draws.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM, FlatwCDM

from qvc.hubble.hubble_eiv_lf import (
    EIV_LNPSI_COL,
    build_eiv_lf_table,
    eiv_distance_terms,
    eiv_object_terms,
)

SN_BESTFIT_H0 = 73.5
SN_BESTFIT_OM0 = 0.361
EIV_Z_EDGES = (0.44, 0.7, 0.9, 1.1, 1.3, 1.5, 1.75, 2.0, 2.3, 2.6, 3.16)
EIV_RESIDUAL_COLUMN_DOCS = {
    "mu_model_median": "posterior median of the model distance modulus mu(z; theta)",
    "M_pred_median": "posterior median of M_pred = M0 + alpha s_obs + beta t_obs",
    "L_hat_median": "posterior median of E[L | m, z, theta] (Eddington-corrected absolute magnitude)",
    "dmi_median": "posterior median of the EIV debias term (m - mu) - E[L | m, z]; residual = m - dmi - M_pred - mu",
    "sigma_dmi": "half the 16-84% posterior range of dmi",
    "mu_obj_biased": "posterior median of m - M_pred (object distance modulus without the Eddington term)",
    "mu_obj_debiased": "posterior median of m + E[d | m, z] - M_pred = mu(z) + resid_debiased",
    "resid_biased": "posterior median of (m - mu(z; theta)) - M_pred",
    "resid_debiased": "posterior median of E[L | m, z] - M_pred (zero mean at fixed L under the model)",
    "err_lc": "sqrt(a k^2 C_i a), a = (alpha, beta), C_i the full (s, t) light-curve covariance",
    "err_intrinsic_x": "alpha * eta (intrinsic scatter of log sigma at fixed L, in magnitudes)",
    "sd_L_given_m": "sd(L | m, z) with the full magnitude variance (sigma_M, e_m, lensing, redshift error)",
    "sd_L_given_m_meas": "sd(L | m, z) with sigma_M = 0 (e_m, lensing, redshift error only)",
    "err_meas": "measurement-only error sqrt(err_lc^2 + sd_L_given_m_meas^2)",
    "err_total": "total predictive error of the residual sqrt(err_lc^2 + err_intrinsic_x^2 + sd_L_given_m^2); used for error bars and binned errors",
    "resid_vs_sn_debiased": "mu_obj_debiased - mu_SN(z), SN best fit flat LCDM H0=73.5 Om0=0.361 (no offset removed)",
    "resid_vs_sn_biased": "mu_obj_biased - mu_SN(z)",
}


def _cosmology(cosmo_model, params):
    if cosmo_model == "FlatLambdaCDM":
        return FlatLambdaCDM(H0=params["H0"], Om0=params["Om0"])
    if cosmo_model == "FlatwCDM":
        return FlatwCDM(H0=params["H0"], Om0=params["Om0"], w0=params["w0"])
    if cosmo_model == "Flatw0waCDM":
        return Flatw0waCDM(H0=params["H0"], Om0=params["Om0"], w0=params["w0"], wa=params["wa"])
    raise ValueError(f"Unsupported cosmology for EIV plots: {cosmo_model!r}")


def _evenly_spaced(n, max_draws):
    if n <= max_draws:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, int(max_draws)).round().astype(int))


def eiv_posterior_hubble_products(
    flat_samples,
    model_labels,
    hyper_samples,
    hyper_labels,
    agn_data,
    *,
    cosmo_model,
    pivots,
    max_draws=500,
    lf_model=None,
):
    """Posterior summaries of the per-object EIV Hubble-diagram quantities.

    ``flat_samples``/``model_labels`` are the standard columns and
    ``hyper_samples``/``hyper_labels`` the latent-lf hyperparameters of the
    same equal-weight posterior rows.  ``agn_data`` is a dict (or DataFrame)
    with the likelihood columns for the objects to plot.
    """
    flat_samples = np.asarray(flat_samples, dtype=float)
    hyper_samples = np.asarray(hyper_samples, dtype=float)
    if flat_samples.shape[0] != hyper_samples.shape[0]:
        raise ValueError("Standard and EIV hyperparameter samples must have the same rows.")
    labels = list(model_labels) + list(hyper_labels)
    samples = np.column_stack([flat_samples, hyper_samples])
    data = {k: np.asarray(agn_data[k]) for k in agn_data.keys()} if isinstance(agn_data, pd.DataFrame) else dict(agn_data)
    if EIV_LNPSI_COL not in data:
        kwargs = {} if lf_model is None else {"lf_model": lf_model}
        data.update(build_eiv_lf_table(np.asarray(data["z"], dtype=float), **kwargs))
    z = np.asarray(data["z"], dtype=float)
    z_err = np.asarray(data["z_err"], dtype=float)
    keys = ("mu", "M_pred", "L_hat", "dmi", "resid_debiased", "resid_biased", "err_meas", "err_total",
            "lc_var", "intrinsic_x_var", "var_L_given_m", "var_L_given_m_meas")
    draws = {k: [] for k in keys}
    rows = _evenly_spaced(len(samples), max_draws)
    for row in rows:
        params = dict(zip(labels, samples[row]))
        cosmo = _cosmology(cosmo_model, params)
        mu, sl, sz = eiv_distance_terms(z, z_err, cosmo)
        terms = eiv_object_terms(params, data, mu_cosmo=mu, sigma_lens=sl, sigma_mu_z=sz, pivots=pivots)
        draws["mu"].append(mu)
        for k in keys[1:]:
            draws[k].append(terms[k])
    draws = {k: np.asarray(v) for k, v in draws.items()}
    med = {k: np.median(v, axis=0) for k, v in draws.items()}
    dmi16, dmi84 = np.percentile(draws["dmi"], [16, 84], axis=0)
    out = dict(
        n_draws=len(rows),
        z=z,
        mu_model_median=med["mu"],
        M_pred_median=med["M_pred"],
        L_hat_median=med["L_hat"],
        dmi_median=med["dmi"],
        sigma_dmi=0.5 * (dmi84 - dmi16),
        mu_obj_biased=np.median(draws["mu"] + draws["resid_biased"], axis=0),
        mu_obj_debiased=np.median(draws["mu"] + draws["resid_debiased"], axis=0),
        resid_biased=med["resid_biased"],
        resid_debiased=med["resid_debiased"],
        err_lc=np.sqrt(med["lc_var"]),
        err_intrinsic_x=np.sqrt(med["intrinsic_x_var"]),
        sd_L_given_m=np.sqrt(med["var_L_given_m"]),
        sd_L_given_m_meas=np.sqrt(med["var_L_given_m_meas"]),
        err_meas=med["err_meas"],
        err_total=med["err_total"],
    )
    sn = FlatLambdaCDM(H0=SN_BESTFIT_H0, Om0=SN_BESTFIT_OM0).distmod(z).value
    out["mu_sn"] = sn
    out["resid_vs_sn_debiased"] = out["mu_obj_debiased"] - sn
    out["resid_vs_sn_biased"] = out["mu_obj_biased"] - sn
    # posterior-median cosmology curve for plotting
    med_params = dict(zip(labels, np.median(samples, axis=0)))
    out["median_params"] = med_params
    out["model_curve"] = lambda zz: _cosmology(cosmo_model, med_params).distmod(np.asarray(zz)).value
    return out


def binned_residual_table(z, resid, err, *, edges=EIV_Z_EDGES, n_boot=1000, seed=12345):
    """Per-bin median and inverse-variance mean with analytic and bootstrap errors."""
    rng = np.random.default_rng(seed)
    z = np.asarray(z, dtype=float)
    resid = np.asarray(resid, dtype=float)
    err = np.asarray(err, dtype=float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = np.flatnonzero((z >= lo) & (z < hi) & np.isfinite(resid) & np.isfinite(err))
        n = sel.size
        if n < 5:
            continue
        r, e = resid[sel], err[sel]
        w = 1.0 / e**2
        wmean = np.sum(w * r) / np.sum(w)
        boot_med = np.empty(n_boot)
        boot_wm = np.empty(n_boot)
        for b in range(n_boot):
            j = rng.integers(0, n, n)
            boot_med[b] = np.median(r[j])
            boot_wm[b] = np.sum(w[j] * r[j]) / np.sum(w[j])
        rows.append(dict(
            z_lo=lo, z_hi=hi, z_median=float(np.median(z[sel])), n=n,
            median=float(np.median(r)), median_err=float(1.2533 * np.std(r, ddof=1) / np.sqrt(n)),
            median_err_boot=float(np.std(boot_med, ddof=1)),
            wmean=float(wmean), wmean_err=float(1.0 / np.sqrt(np.sum(w))),
            wmean_err_boot=float(np.std(boot_wm, ddof=1)),
            chi2_red_in_bin=float(np.sum(((r - wmean) / e) ** 2) / max(n - 1, 1)),
        ))
    return pd.DataFrame(rows)


def residual_step(z, resid, *, n_boot=1000, seed=7):
    """Median(z >= 2) - median(z < 1.5) with a bootstrap error."""
    rng = np.random.default_rng(seed)
    z = np.asarray(z, dtype=float)
    resid = np.asarray(resid, dtype=float)
    hi = np.flatnonzero(z >= 2.0)
    lo = np.flatnonzero(z < 1.5)
    value = np.median(resid[hi]) - np.median(resid[lo])
    boots = [np.median(resid[rng.choice(hi, hi.size)]) - np.median(resid[rng.choice(lo, lo.size)]) for _ in range(n_boot)]
    return float(value), float(np.std(boots, ddof=1))


def write_eiv_hubble_products(products, df_meta, fit_mask, plot_path, *, n_params, label="EIV latent-lf",
                              shift_to_agn_scale=True, sn_points=None):
    """Residual CSV, binned tables, summary and Hubble-diagram figures.

    ``shift_to_agn_scale=True`` (AGN-only fits, where M0 is degenerate with H0)
    shifts the SN and model curves to the AGN median.  For joint SN + AGN fits
    use ``False``: the curves are drawn as fitted (joint posterior-median
    cosmology and the SN-only best fit) and the AGN median offset is reported.
    ``sn_points`` = (z, mu, err) draws the supernovae in the top panel.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_path, exist_ok=True)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    z = products["z"]
    cols = list(EIV_RESIDUAL_COLUMN_DOCS)
    df = pd.DataFrame({"object_id": np.asarray(df_meta["object_id"]).astype(str), "is_fit_selection": fit_mask, "z": z,
                       "apparent_mag_2500": np.asarray(df_meta["apparent_mag_2500"], dtype=float)})
    for c in cols:
        df[c] = products[c]
    csv_path = os.path.join(plot_path, "eiv_hubble_residuals.csv")
    df.to_csv(csv_path, index=False)
    with open(os.path.join(plot_path, "eiv_hubble_residuals_columns.txt"), "w") as fh:
        fh.write(f"Columns of eiv_hubble_residuals.csv ({label}; posterior medians over {products['n_draws']} draws)\n")
        for c in cols:
            fh.write(f"{c}: {EIV_RESIDUAL_COLUMN_DOCS[c]}\n")

    zf = z[fit_mask]
    err_t = products["err_total"][fit_mask]
    err_m = products["err_meas"][fit_mask]
    summary = {"n_fit": int(fit_mask.sum()), "n_draws": products["n_draws"],
               "median_err_total": float(np.median(err_t)), "median_err_meas": float(np.median(err_m)),
               "median_err_lc": float(np.median(products["err_lc"][fit_mask])),
               "median_err_intrinsic_x": float(np.median(products["err_intrinsic_x"][fit_mask])),
               "median_sd_L_given_m": float(np.median(products["sd_L_given_m"][fit_mask])),
               "median_abs_dmi": float(np.median(np.abs(products["dmi_median"][fit_mask]))),
               "median_sigma_dmi": float(np.median(products["sigma_dmi"][fit_mask])),
               "agn_median_offset_vs_sn_bestfit": float(np.median(products["resid_vs_sn_debiased"][fit_mask])),
               "agn_median_offset_vs_model": float(np.median(products["resid_debiased"][fit_mask])),
               "curves_shifted_to_agn_scale": bool(shift_to_agn_scale)}
    tables = {}
    for kind in ("debiased", "biased"):
        r_model = products[f"resid_{kind}"][fit_mask]
        r_sn = products[f"resid_vs_sn_{kind}"][fit_mask]
        dof = max(r_model.size - int(n_params), 1)
        summary[f"chi2_red_{kind}_err_total"] = float(np.sum((r_model / err_t) ** 2) / dof)
        summary[f"chi2_red_{kind}_err_meas"] = float(np.sum((r_model / err_m) ** 2) / dof)
        summary[f"step_vs_model_{kind}"] = residual_step(zf, r_model)
        summary[f"step_vs_sn_{kind}"] = residual_step(zf, r_sn)
        t_model = binned_residual_table(zf, r_model, err_t)
        t_sn = binned_residual_table(zf, r_sn - np.median(r_sn), err_t)
        t_model.insert(0, "residual", f"{kind} vs AGN posterior model")
        t_sn.insert(0, "residual", f"{kind} vs SN best fit (overall median removed)")
        tables[kind] = (t_model, t_sn)
    pd.concat([t for pair in tables.values() for t in pair]).to_csv(
        os.path.join(plot_path, "eiv_hubble_binned_residuals.csv"), index=False)
    pd.Series({k: (v if not isinstance(v, tuple) else "%.4f +- %.4f" % v) for k, v in summary.items()}).to_csv(
        os.path.join(plot_path, "eiv_hubble_summary.csv"), header=["value"])

    model_curve = products["model_curve"]
    om = products["median_params"].get("Om0", np.nan)
    zz = np.linspace(0.4, 3.2, 300)
    sn_curve = FlatLambdaCDM(H0=SN_BESTFIT_H0, Om0=SN_BESTFIT_OM0).distmod(zz).value
    for kind in ("debiased", "biased"):
        mu_obj = products[f"mu_obj_{kind}"][fit_mask]
        r_sn = products[f"resid_vs_sn_{kind}"][fit_mask]
        offset = float(np.median(r_sn)) if shift_to_agn_scale else 0.0
        agn_curve = model_curve(zz)
        agn_offset = float(np.median(mu_obj - model_curve(zf))) if shift_to_agn_scale else 0.0
        t_model, t_sn = tables[kind]
        fig, ax = plt.subplots(3, 1, figsize=(7.5, 10), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1, 1], hspace=0.06))
        ax[0].errorbar(zf, mu_obj, yerr=err_t, fmt="none", ecolor="0.75", alpha=0.25, lw=0.5, zorder=1)
        ax[0].plot(zf, mu_obj, ".", ms=2, color="0.55", alpha=0.5, zorder=2, label="AGN (individual, err_total)")
        if sn_points is not None:
            zs, mus, es = (np.asarray(v, dtype=float) for v in sn_points)
            ax[0].errorbar(zs, mus, yerr=es, fmt="none", ecolor="#eda100", alpha=0.3, lw=0.5, zorder=1)
            ax[0].plot(zs, mus, ".", ms=2.5, color="#eda100", zorder=3, label="SNe Ia (m_b_corr - M0_sn, joint median)")
        shift_note = " (shifted to AGN scale)" if shift_to_agn_scale else " (H0=%.1f, as fitted)" % SN_BESTFIT_H0
        ax[0].plot(zz, sn_curve + offset, "-", color="#e34948", lw=2,
                   label=r"SN-only best fit: flat $\Lambda$CDM, $\Omega_m$=%.3f%s" % (SN_BESTFIT_OM0, shift_note))
        h0 = products["median_params"].get("H0", np.nan)
        ax[0].plot(zz, agn_curve + agn_offset, "--", color="#2a78d6", lw=2,
                   label=r"%s posterior median, $\Omega_m$=%.3f, $H_0$=%.1f" % ("EIV" if shift_to_agn_scale else "joint EIV", om, h0))
        zmed = t_sn["z_median"].to_numpy()
        ax[0].errorbar(zmed, t_sn["median"].to_numpy() + offset + FlatLambdaCDM(H0=SN_BESTFIT_H0, Om0=SN_BESTFIT_OM0).distmod(zmed).value,
                       yerr=t_sn["median_err"].to_numpy(), fmt="o", color="k", ms=5, capsize=2, label="AGN binned median", zorder=5)
        ax[0].set_ylabel(r"distance modulus $\mu$ (%s)" % ("AGN scale" if shift_to_agn_scale else "as fitted"))
        ylo = np.percentile(mu_obj, 0.5) - 0.5
        if sn_points is not None:
            ylo = min(ylo, float(np.min(sn_points[1])) - 0.5)
        ax[0].set_ylim(ylo, np.percentile(mu_obj, 99.5) + 0.5)
        if sn_points is not None:
            ax[0].set_xscale("log")
        ax[0].legend(fontsize=7.5, loc="lower right", frameon=False)
        ax[0].set_title(f"{label}: {kind} (N={fit_mask.sum()}, {products['n_draws']} posterior draws)", fontsize=10)
        ax[1].axhline(0, color="#e34948", lw=1.5)
        ax[1].plot(zz, agn_curve + agn_offset - sn_curve - offset, "--", color="#2a78d6", lw=2)
        ax[1].errorbar(zmed, t_sn["median"], yerr=t_sn["median_err"], fmt="o", color="k", ms=5, capsize=2)
        ax[1].set_ylabel(r"$\mu_{\rm AGN}-\mu_{\rm SN}$ (median removed)", fontsize=8.5)
        if not shift_to_agn_scale:
            ax[1].text(0.02, 0.88, "AGN median offset vs SN-only best fit (removed here): %+.3f mag" % float(np.median(r_sn)),
                       transform=ax[1].transAxes, fontsize=8)
        s_sn = summary[f"step_vs_sn_{kind}"]
        ax[1].text(0.02, 0.06, "step med(z>=2) - med(z<1.5) = %+.3f +- %.3f mag" % s_sn, transform=ax[1].transAxes, fontsize=8)
        ax[2].axhline(0, color="#2a78d6", lw=1.5, ls="--")
        ax[2].errorbar(t_model["z_median"], t_model["median"], yerr=t_model["median_err"], fmt="o", color="k", ms=5, capsize=2,
                       label="binned median")
        ax[2].errorbar(t_model["z_median"] + 0.02, t_model["wmean"], yerr=t_model["wmean_err"], fmt="s", color="#1baf7a", ms=4,
                       capsize=2, label="binned inverse-variance mean (err_total)")
        ax[2].set_ylabel("residual vs AGN model (mag)", fontsize=8.5)
        ax[2].legend(fontsize=7.5, frameon=False, loc="lower left")
        ax[2].set_xlabel("redshift")
        for a in ax[1:]:
            lim = np.nanmax(np.abs(np.concatenate([t_sn["median"], t_model["median"]]))) + 0.25
            a.set_ylim(-max(lim, 0.4), max(lim, 0.4))
            a.grid(alpha=0.3)
        ax[0].grid(alpha=0.3)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(plot_path, f"hubble_diagram_eiv_{kind}.{ext}"), dpi=160, bbox_inches="tight")
        plt.close(fig)
    return csv_path, summary, tables
