"""Shared sigma/tau wavelength broken power-law fit utilities."""

import numpy as np
import pandas as pd


SDSS_LAMBDA_PIVOT = {
    "u": 3543.0,
    "g": 4770.0,
    "r": 6231.0,
    "i": 7625.0,
    "z": 9134.0,
}


def log_broken_pl(lam, lam_s, d1, d2):
    """Continuous, sharp broken power law anchored at zero at ``lam_s``."""
    x = np.log10(np.asarray(lam, dtype=float) / float(lam_s))
    return d1 * np.minimum(x, 0.0) + d2 * np.maximum(x, 0.0)


def _continuum_tau_logs_from_medians(df, *, bands, lam_s, disk_order):
    """Return median-parameter continuum-only log tau values when available."""
    required = {
        "z",
        "tau_fast_driver",
        "tau_slow_driver",
        "lag0",
        "lambda_center_rf",
    }
    available_bands = [band for band in bands if f"lag_disk_{band}" in df.columns]
    if not required.issubset(df.columns) or not available_bands:
        return None

    import jax
    import jax.numpy as jnp

    from qvc.light_curve.multiband_model_shared_latent_blr import (
        continuum_effective_timescale,
    )

    numeric = lambda name: pd.to_numeric(df[name], errors="coerce").to_numpy(float)
    tau_fast = numeric("tau_fast_driver")
    tau_slow = numeric("tau_slow_driver")
    lag0 = numeric("lag0")
    lambda_center = numeric("lambda_center_rf")
    z = numeric("z")
    evaluate = jax.jit(
        jax.vmap(
            lambda fast, slow, lag: continuum_effective_timescale(
                fast, slow, lag, disk_order=disk_order
            )
        )
    )

    def log_tau(lag):
        valid = (
            np.isfinite(tau_fast)
            & (tau_fast > 0.0)
            & np.isfinite(tau_slow)
            & (tau_slow > 0.0)
            & np.isfinite(lag)
            & (lag > 0.0)
            & np.isfinite(z)
            & (z > -1.0)
        )
        result = np.full(len(df), np.nan)
        result[valid] = np.asarray(
            evaluate(
                jnp.asarray(tau_fast[valid]),
                jnp.asarray(tau_slow[valid]),
                jnp.asarray(lag[valid]),
            )
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(result) - np.log10(1.0 + z)

    lag_reference = lag0 * (float(lam_s) / lambda_center) ** (4.0 / 3.0)
    values = {"uv": log_tau(lag_reference)}
    values.update(
        {band: log_tau(numeric(f"lag_disk_{band}")) for band in available_bands}
    )
    return values


def _collect_sigma_tau_lambda_data(df, target, *, bands, lam_s, disk_order):
    if target == "sigma":
        value_template = "log_sigma_band_{}"
        uv_col = "log_sigma_uv"
        continuum_tau = None
    elif target == "tau":
        value_template = "log_tau_band_{}_RF"
        uv_col = "log_tau_uv_rf"
        continuum_tau = _continuum_tau_logs_from_medians(
            df,
            bands=bands,
            lam_s=lam_s,
            disk_order=disk_order,
        )
    else:
        raise ValueError("target must be 'sigma' or 'tau'")

    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
    uv = (
        continuum_tau["uv"]
        if continuum_tau is not None
        else pd.to_numeric(df[uv_col], errors="coerce").to_numpy(dtype=float)
    )
    lam_list = []
    y_abs_list = []
    y_res_list = []
    group_list = []
    band_list = []
    for band in bands:
        if band not in SDSS_LAMBDA_PIVOT:
            continue
        value_col = value_template.format(band)
        if continuum_tau is not None and target == "tau":
            if band not in continuum_tau:
                continue
            values = continuum_tau[band]
        elif value_col in df.columns:
            values = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
        else:
            continue
        lam_rf = SDSS_LAMBDA_PIVOT[band] / (1.0 + z)
        mask = (
            np.isfinite(lam_rf)
            & (lam_rf > 0.0)
            & np.isfinite(values)
            & np.isfinite(uv)
        )
        if not np.any(mask):
            continue
        idx = np.flatnonzero(mask)
        lam_list.append(lam_rf[mask])
        y_abs_list.append(values[mask])
        y_res_list.append(values[mask] - uv[mask])
        group_list.append(idx)
        band_list.extend([band] * int(np.count_nonzero(mask)))

    if not lam_list:
        return None
    lam = np.concatenate(lam_list)
    y_abs = np.concatenate(y_abs_list)
    y_res = np.concatenate(y_res_list)
    group = np.concatenate(group_list)
    return {
        "lam": lam,
        "x": np.log10(lam),
        "y_abs": y_abs,
        "y_res": y_res,
        "group": group.astype(int),
        "n_objects": len(df),
        "band": np.asarray(band_list, dtype=object),
        "value_definition": (
            "continuum_median_parameter_approximation"
            if continuum_tau is not None and target == "tau"
            else "stored_catalog_values"
        ),
    }


def _fit_slopes(data, *, lam_s, positive, bootstrap_replicates, bootstrap_seed):
    """Fit per-object lines, then a normal or lognormal slope population.

    Missing sides remain NaN. Bootstrap entire catalog rows so all bands and
    both slopes from a quasar move together, including differing side coverage.
    Measurement errors are not used or deconvolved from population scatter.
    """
    x = np.log10(data["lam"] / float(lam_s))
    design = np.column_stack([np.minimum(x, 0.0), np.maximum(x, 0.0)])
    y = data["y_res"]
    group = data["group"]
    n_group = data["n_objects"]
    num = np.zeros((n_group, 2))
    den = np.zeros_like(num)
    np.add.at(num, group, design * y[:, None])
    np.add.at(den, group, design**2)
    total_den = den.sum(axis=0)
    if np.any(total_den <= 1e-12):
        raise ValueError("broken power-law fit requires points on both sides of lam_s")
    object_slopes = np.divide(
        num, den, out=np.full_like(num, np.nan), where=den > 1e-12
    )
    counts = np.isfinite(object_slopes).sum(axis=0)
    if np.any(counts < 2):
        raise ValueError("population fit requires at least two objects on each side")
    if positive and np.any(object_slopes[np.isfinite(object_slopes)] <= 0):
        raise ValueError(
            "lognormal tau population requires strictly positive per-object slopes"
        )
    transformed = np.log(object_slopes) if positive else object_slopes

    def population_mean(values):
        location = np.nanmean(values, axis=0)
        variance = np.nanvar(values, axis=0)
        mean = np.exp(location + 0.5 * variance) if positive else location
        return mean, location, np.sqrt(variance)

    theta, location, scale = population_mean(transformed)
    rng = np.random.default_rng(bootstrap_seed)
    means = []
    for _ in range(bootstrap_replicates):
        sampled = transformed[rng.integers(n_group, size=n_group)]
        if np.any(np.isfinite(sampled).sum(axis=0) == 0):
            continue
        means.append(population_mean(sampled)[0])
    means = np.asarray(means)
    if len(means) < 2 or not np.all(np.isfinite(means)):
        raise ValueError("insufficient finite whole-object bootstrap fits")
    cov = np.cov(means, rowvar=False, ddof=1)
    err = np.sqrt(np.diag(cov))
    mean_percentiles = np.percentile(means, [16, 84], axis=0).T
    residual = y - design @ theta
    # Descriptive shape check, not a calibrated goodness-of-fit p-value.
    from scipy.stats import norm
    empirical = np.nanpercentile(object_slopes, [16, 50, 84, 95], axis=0).T
    modeled = location[:, None] + scale[:, None] * norm.ppf([0.16, 0.5, 0.84, 0.95])
    if positive:
        modeled = np.exp(modeled)
    return {
        "d1": float(theta[0]), "d2": float(theta[1]),
        "d1_err": float(err[0]), "d2_err": float(err[1]),
        "cov": cov, "intercept": 0.0,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mean_slope_percentiles": mean_percentiles,
        "bootstrap_mean_slopes": means,
        "bootstrap_replicates_valid": len(means),
        "population_distribution": "lognormal" if positive else "normal",
        "population_location": location, "population_scale": scale,
        "empirical_population_quantiles": empirical,
        "modeled_population_quantiles": modeled,
        "object_slopes": object_slopes, "objects_per_slope": counts,
    }


def fit_sigma_tau_lambda_broken_pl(
    df,
    *,
    bands=("u", "g", "r", "i", "z"),
    lam_s=2500.0,
    min_points=3,
    include_plot_payload=False,
    bootstrap_replicates=1000,
    bootstrap_seed=2500,
    disk_order=3,
):
    """Fit population mean slopes with whole-quasar bootstrap uncertainty.

    Tau slopes follow a positive lognormal; sigma slopes follow a normal.
    Mean uncertainties come from bootstrap refits; the plotted population band
    uses empirical per-object slope percentiles. When driver and disk fields are
    available, tau is reconstructed as a continuum-only median-parameter
    approximation. Measurement errors are not used.
    """
    if (
        not isinstance(bootstrap_replicates, (int, np.integer))
        or bootstrap_replicates < 2
    ):
        raise ValueError("bootstrap_replicates must be an integer >= 2")
    if not np.isfinite(lam_s) or lam_s <= 0:
        raise ValueError("lam_s must be finite and positive")
    required = {"z", "log_sigma_uv", "log_tau_uv_rf"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"missing columns {missing}")
    result = {}
    for target in ("sigma", "tau"):
        data = _collect_sigma_tau_lambda_data(
            df,
            target,
            bands=bands,
            lam_s=lam_s,
            disk_order=disk_order,
        )
        n = 0 if data is None else len(data["lam"])
        if n < min_points:
            raise ValueError(f"insufficient finite points ({target}={n})")
        fit = _fit_slopes(
            data,
            lam_s=lam_s,
            positive=(target == "tau"),
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        )
        result[f"fit_{target}"] = fit
        for side, key, i in (("blue", "d1", 0), ("red", "d2", 1)):
            prefix = f"eta_{target}_{side}"
            result[prefix] = fit[key]
            result[prefix + "_err"] = fit[key + "_err"]
            result[prefix + "_p16"] = float(fit["mean_slope_percentiles"][i, 0])
            result[prefix + "_p84"] = float(fit["mean_slope_percentiles"][i, 1])
        if include_plot_payload:
            result[f"{target}_data"] = data
    return result


def slope_population_band(fit, lam_grid, *, lam_s):
    """Return the empirical 16th-84th percentile per-object slope envelope."""
    q = fit["empirical_population_quantiles"][:, [0, 2]]
    edge1 = log_broken_pl(lam_grid, lam_s, q[0, 0], q[1, 0])
    edge2 = log_broken_pl(lam_grid, lam_s, q[0, 1], q[1, 1])
    # Negative log wavelength reverses the ordering of the slope quantiles.
    return np.minimum(edge1, edge2), np.maximum(edge1, edge2)
