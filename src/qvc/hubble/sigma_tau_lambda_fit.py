"""Shared sigma/tau wavelength broken power-law fit utilities."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


SDSS_LAMBDA_PIVOT = {
    "u": 3543.0,
    "g": 4770.0,
    "r": 6231.0,
    "i": 7625.0,
    "z": 9134.0,
}


def log_broken_pl(lam, lam_s, d1, d2, ds):
    """Smooth broken power law in log10 space, normalized to zero at lam_s."""
    lam = np.asarray(lam, dtype=float)
    ds = max(float(ds), 1e-3)
    ln10 = np.log(10.0)
    log10x = (np.log(lam) - np.log(float(lam_s))) / ln10
    a = log10x / ds
    log10_1p10a = np.logaddexp(0.0, a * ln10) / ln10
    return d1 * log10x + ((d2 - d1) * ds) * (log10_1p10a - np.log10(2.0))


def jac_log_broken_pl(lam, lam_s, d1, d2, ds):
    """Jacobian of log_broken_pl with respect to d1 and d2."""
    lam = np.asarray(lam, dtype=float)
    ds = max(float(ds), 1e-3)
    ln10 = np.log(10.0)
    log10x = (np.log(lam) - np.log(float(lam_s))) / ln10
    a = log10x / ds
    q = np.logaddexp(0.0, a * ln10) / ln10
    t = q - np.log10(2.0)
    return np.vstack([log10x - ds * t, ds * t])


def _collect_sigma_tau_lambda_data(df, target, *, bands, lam_s):
    del lam_s
    if target == "sigma":
        value_template = "log_sigma_band_{}"
        err_template = "log_sigma_band_{}_err"
        uv_col = "log_sigma_uv"
    elif target == "tau":
        value_template = "log_tau_band_{}_RF"
        err_template = "log_tau_band_{}_RF_err"
        uv_col = "log_tau_uv_rf"
    else:
        raise ValueError("target must be 'sigma' or 'tau'")

    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
    uv = pd.to_numeric(df[uv_col], errors="coerce").to_numpy(dtype=float)
    lam_list = []
    y_abs_list = []
    y_res_list = []
    err_list = []
    group_list = []
    band_list = []
    for band in bands:
        if band not in SDSS_LAMBDA_PIVOT:
            continue
        value_col = value_template.format(band)
        if value_col not in df.columns:
            continue
        values = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
        err_col = err_template.format(band)
        if err_col in df.columns:
            errs = pd.to_numeric(df[err_col], errors="coerce").to_numpy(dtype=float)
        else:
            errs = np.ones(len(df), dtype=float)
        lam_rf = SDSS_LAMBDA_PIVOT[band] / (1.0 + z)
        mask = (
            np.isfinite(lam_rf)
            & (lam_rf > 0.0)
            & np.isfinite(values)
            & np.isfinite(uv)
        )
        if not np.any(mask):
            continue
        clean_errs = np.where(np.isfinite(errs) & (errs > 0.0), errs, 1.0)
        idx = np.flatnonzero(mask)
        lam_list.append(lam_rf[mask])
        y_abs_list.append(values[mask])
        y_res_list.append(values[mask] - uv[mask])
        err_list.append(clean_errs[mask])
        group_list.append(idx)
        band_list.extend([band] * int(np.count_nonzero(mask)))

    if not lam_list:
        return None
    lam = np.concatenate(lam_list)
    y_abs = np.concatenate(y_abs_list)
    y_res = np.concatenate(y_res_list)
    err = np.concatenate(err_list)
    group = np.concatenate(group_list)
    _, group = np.unique(group, return_inverse=True)
    weight = 1.0 / np.maximum(err * err, 1e-12)
    return {
        "lam": lam,
        "x": np.log10(lam),
        "y_abs": y_abs,
        "y_res": y_res,
        "weight": weight,
        "group": group.astype(int),
        "band": np.asarray(band_list, dtype=object),
    }


def _profile_sse(y, model, weight, group):
    n_group = int(group.max()) + 1
    num = np.zeros(n_group, dtype=float)
    den = np.zeros(n_group, dtype=float)
    np.add.at(num, group, weight * (y - model))
    np.add.at(den, group, weight)
    intercept = num / np.where(den > 0.0, den, 1.0)
    fit = intercept[group] + model
    return float(np.sum(weight * (y - fit) ** 2))


def _hessian_num(objective, theta, eps=1e-3):
    theta = np.asarray(theta, dtype=float)
    h = eps * np.maximum(1.0, np.abs(theta))
    hessian = np.zeros((2, 2), dtype=float)
    for i in range(2):
        ei = np.zeros(2, dtype=float)
        ei[i] = h[i]
        for j in range(2):
            ej = np.zeros(2, dtype=float)
            ej[j] = h[j]
            hessian[i, j] = (
                objective(theta + ei + ej)
                - objective(theta + ei - ej)
                - objective(theta - ei + ej)
                + objective(theta - ei - ej)
            ) / (4.0 * h[i] * h[j])
    return 0.5 * (hessian + hessian.T)


def _fit_slopes(data, *, lam_s, ds_fixed, bounds):
    def objective(theta):
        model = log_broken_pl(data["lam"], lam_s, float(theta[0]), float(theta[1]), ds_fixed)
        return 0.5 * _profile_sse(data["y_abs"], model, data["weight"], data["group"])

    theta = None
    try:
        result = minimize(
            objective,
            x0=np.array([0.0, 0.0], dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if result.success and np.all(np.isfinite(result.x)):
            theta = np.asarray(result.x, dtype=float)
    except Exception:
        pass

    if theta is None:
        grid_1 = np.linspace(bounds[0][0], bounds[0][1], 41)
        grid_2 = np.linspace(bounds[1][0], bounds[1][1], 41)
        best_value = np.inf
        best = (0.0, 0.0)
        for d1 in grid_1:
            for d2 in grid_2:
                value = objective((d1, d2))
                if value < best_value:
                    best_value = value
                    best = (float(d1), float(d2))
        theta = np.asarray(best, dtype=float)

    sse = 2.0 * objective(theta)
    dof = max(1, len(data["lam"]) - (int(data["group"].max()) + 1) - 2)
    redchi = float(sse / dof)
    cov = None
    try:
        hessian = _hessian_num(objective, theta)
        cov = np.linalg.inv(hessian) * redchi
        if not np.all(np.isfinite(cov)):
            cov = None
    except Exception:
        cov = None

    err = np.full(2, np.nan, dtype=float)
    if cov is not None:
        err = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    return {
        "d1": float(theta[0]),
        "d2": float(theta[1]),
        "d1_err": float(err[0]),
        "d2_err": float(err[1]),
        "cov": cov,
        "redchi": redchi,
    }


def _global_intercept(data, *, lam_s, d1, d2, ds):
    model = log_broken_pl(data["lam"], lam_s, d1, d2, ds)
    den = np.sum(data["weight"])
    if not np.isfinite(den) or den <= 0.0:
        return 0.0
    return float(np.sum(data["weight"] * (data["y_res"] - model)) / den)


def fit_sigma_tau_lambda_broken_pl(
    df,
    *,
    bands=("u", "g", "r", "i", "z"),
    lam_s=2500.0,
    ds_fixed_sigma=0.1,
    ds_fixed_tau=0.1,
    min_points=3,
    include_plot_payload=False,
):
    """Fit the postcut sigma/tau wavelength broken power-law diagnostic."""
    required = {"z", "log_sigma_uv", "log_tau_uv_rf"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"missing columns {missing}")

    sigma_data = _collect_sigma_tau_lambda_data(df, "sigma", bands=bands, lam_s=lam_s)
    tau_data = _collect_sigma_tau_lambda_data(df, "tau", bands=bands, lam_s=lam_s)
    sigma_n = 0 if sigma_data is None else len(sigma_data["lam"])
    tau_n = 0 if tau_data is None else len(tau_data["lam"])
    if sigma_n < min_points or tau_n < min_points:
        raise ValueError(f"insufficient finite points (sigma={sigma_n}, tau={tau_n})")

    fit_sigma = _fit_slopes(
        sigma_data,
        lam_s=lam_s,
        ds_fixed=ds_fixed_sigma,
        bounds=((-6.0, 6.0), (-6.0, 6.0)),
    )
    fit_tau = _fit_slopes(
        tau_data,
        lam_s=lam_s,
        ds_fixed=ds_fixed_tau,
        bounds=((-10.0, 10.0), (-10.0, 10.0)),
    )
    fit_sigma["intercept"] = _global_intercept(
        sigma_data,
        lam_s=lam_s,
        d1=fit_sigma["d1"],
        d2=fit_sigma["d2"],
        ds=ds_fixed_sigma,
    )
    fit_tau["intercept"] = _global_intercept(
        tau_data,
        lam_s=lam_s,
        d1=fit_tau["d1"],
        d2=fit_tau["d2"],
        ds=ds_fixed_tau,
    )

    result = {
        "eta_sigma_blue": fit_sigma["d1"],
        "eta_sigma_blue_err": fit_sigma["d1_err"],
        "eta_sigma_red": fit_sigma["d2"],
        "eta_sigma_red_err": fit_sigma["d2_err"],
        "eta_tau_blue": fit_tau["d1"],
        "eta_tau_blue_err": fit_tau["d1_err"],
        "eta_tau_red": fit_tau["d2"],
        "eta_tau_red_err": fit_tau["d2_err"],
        "fit_sigma": fit_sigma,
        "fit_tau": fit_tau,
    }

    if include_plot_payload:
        result["sigma_data"] = sigma_data
        result["tau_data"] = tau_data
    return result


def std_from_slope_cov(fit, lam_grid, *, lam_s, ds_fixed):
    """Return the fit-curve standard deviation from a d1/d2 covariance."""
    cov = fit.get("cov")
    if cov is None:
        return None
    jac = jac_log_broken_pl(lam_grid, lam_s, fit["d1"], fit["d2"], ds_fixed)
    var = np.einsum("in,ij,jn->n", jac, cov, jac)
    std = np.sqrt(np.clip(var, 0.0, np.inf))
    return std if np.all(np.isfinite(std)) else None
