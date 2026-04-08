import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
from pathlib import Path

plt.style.use(Path(__file__).with_name("style.mplstyle"))
import corner
import numpy as np
import os
import re
import math
import jax.numpy as jnp
from scipy.stats import norm, probplot

from astropy.timeseries import LombScargle

prefix = os.environ.get('PREFIX', "test")
suffix = os.environ.get('SUFFIX', "test")

from qvc.light_curve.multiband_fit_utils import log_single_pl

import logging

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

lambda_pivot = {
    'u': 3543,  # SDSS u-band
    'g': 4770,  # SDSS g-band
    'r': 6231,  # SDSS r-band
    'i': 7625,  # SDSS i-band
    'z': 9134,  # SDSS z-band
    'y': 9633,  # PS1 y-band
}

colors = {'u': 'tab:blue',
          'g': 'tab:green', 
          'r': 'tab:orange', 
          'i': 'tab:red', 
          'z': 'tab:brown', 
          'y': 'tab:gray'}

SF_LAG_PLOT_MIN_RF = 10.0
SF_LAG_PLOT_MAX_RF = 1e4


def _get_param(sample_dict, primary, fallback=None):
    if primary in sample_dict:
        return sample_dict[primary]
    if fallback is not None and fallback in sample_dict:
        return sample_dict[fallback]
    raise KeyError(f"Missing parameter '{primary}'" + (f" (fallback '{fallback}')" if fallback else ""))


def _prediction_to_display(model, pred_result):
    if hasattr(model, "prediction_to_display"):
        return model.prediction_to_display(pred_result)
    return pred_result


def _component_only_params(params, *, component):
    """Return a copy of params with only the requested variability component active."""

    out = dict(params)
    if component == "continuum":
        zero_keys = (
            "amp_blr",
            "amp_blr2",
            "amp_bc",
            "amp_blr_relflux",
            "amp_blr2_relflux",
            "amp_bc_relflux",
        )
    elif component == "line":
        zero_keys = (
            "amp_cont",
            "amp_cont_relflux",
        )
    else:
        raise ValueError(f"Unknown component {component!r}")

    for key in zero_keys:
        if key in out:
            arr = np.asarray(out[key], dtype=float)
            out[key] = np.zeros_like(arr)
    return out


def _posterior_sample_params_at_index(samples, i, reference_n):
    """Build per-draw params while tolerating singleton or deterministic entries."""

    out = {}
    for key, value in samples.items():
        arr = np.asarray(value)
        if arr.ndim == 0:
            out[key] = jnp.array(arr)
        elif arr.shape[0] == reference_n:
            out[key] = jnp.array(arr[i])
        elif arr.shape[0] == 1:
            out[key] = jnp.array(arr[0])
        else:
            out[key] = jnp.array(arr)
    return out


POSTERIOR_PLOT_KEY_GROUPS = {
    "continuum": {
        "exact": [
            "eta_sigma",
            "eta_tau",
            "log_sigma_uv",
            "log_tau_uv",
            "log_tau_fast_uv",
            "lag0",
            "lag_beta",
            "linear_trend",
        ],
        "prefixes": (),
    },
    "line_response": {
        "exact": [
            "dlog_amp_blr",
            "log_lag_blr",
            "dlog_amp_bc",
            "log_lag_ratio_bc_to_blr",
        ],
        "prefixes": (
            "dlog_amp_blr_",
            "log_lag_blr_",
        ),
    },
    "mean_and_noise": {
        "exact": [
            "mean",
            "log_jitter",
        ],
        "prefixes": (
            "mean_",
            "log_jitter_",
        ),
    },
}


def _posterior_plot_labels(samples_flat):
    """Return an ordered curated subset of posterior keys for diagnostic plots."""

    internal_skip_keys = {
        "log_jitter_active",
    }
    all_labels = [label for label in samples_flat.keys() if label not in internal_skip_keys]
    selected = []
    seen = set()

    def _add(label):
        if label in samples_flat and label not in seen:
            selected.append(label)
            seen.add(label)

    for group in POSTERIOR_PLOT_KEY_GROUPS.values():
        for label in group["exact"]:
            _add(label)
        for prefix in group["prefixes"]:
            for label in all_labels:
                if label.startswith(prefix):
                    _add(label)

    if not selected:
        selected = list(all_labels)

    return all_labels, selected


def _corner_plot_labels(samples_flat):
    """Return ordered candidate and filtered corner-plot parameter labels."""

    return _posterior_plot_labels(samples_flat)


def _trace_plot_labels(samples_flat):
    """Return ordered trace-plot labels, including fitted survey offsets."""

    all_labels, selected = _posterior_plot_labels(samples_flat)
    trace_labels = list(selected)
    seen = set(trace_labels)

    for label in all_labels:
        if not label.startswith("survey_delta_mag_") or label in seen:
            continue
        values = np.asarray(samples_flat[label], dtype=float)
        if values.size == 0 or np.allclose(values, 0.0, atol=0.0, rtol=0.0):
            continue
        trace_labels.append(label)
        seen.add(label)

    return all_labels, trace_labels


def save_lc_plot(times, mags, magerrs, object_id):
    logging.info("Saving LC plot")
    # Plot and save the light curves
    fig, ax = plt.subplots(figsize=(10, 6))
    for band in bands:
        if len(times[band]) > 0:
            ax.errorbar(times[band], mags[band], yerr=magerrs[band], fmt='o', label=f'{band}-band', alpha=0.7)

    ax.set_xlabel('Time (MJD)', fontsize=14)
    ax.set_ylabel('Magnitude', fontsize=14)
    ax.invert_yaxis()  # Magnitudes are brighter when lower
    ax.legend()
    #ax.set_title(f'Light Curve for Object {object_id}', fontsize=16)
    plt.tight_layout()

    # Save the plot as a PNG file
    output_dir = "light_curves"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join("light_curves", f'{object_id}_light_curve.png'))
    plt.close(fig)
    logging.info(f"Saved LC plot to light_curves/{object_id}_light_curve.png")

def plot_posterior_fast_OLD(
    samples_flat,
    data,
    bins=15,
    max_points=50_000,
    p_lo=0.5,
    p_hi=99.5,
    const_ptp=1e-12,         # threshold for "near-constant"
    jitter_rel=1e-6,         # relative jitter scale (× |mean|)
    jitter_abs=1e-8          # absolute floor for jitter
):
    """
    Faster corner plot for large MCMC draws, keeps constant params by jittering.

    Parameters
    ----------
    samples_flat : dict[str, array_like]
        Dict of MCMC samples for each param; shape (n_samples, ...) per entry.
    data : dict
        Must contain 'object_id' and 'z'. Optional: 'prefix', 'suffix'.
    bins : int
        Number of bins (1D) and base for 2D via hist2d_kwargs.
    max_points : int
        Cap the number of points used in the plot (random subsample).
    p_lo, p_hi : float
        Percentile clipping for plotting ranges (mitigates outliers → faster hist).
    const_ptp : float
        If ptp (max-min) <= const_ptp, treat as near-constant and jitter.
    jitter_rel : float
        Jitter sigma = max(jitter_abs, jitter_rel * |mean|) for near-constant cols.
    jitter_abs : float
        Absolute minimum jitter sigma.
    """
    logging.info("Saving posterior plot (fast path)")
    object_id = data["object_id"]
    z = data["z"]

    print(samples_flat)

    # Stable column order
    labels = list(samples_flat.keys())
    labels = np.array(labels)

    print('1')

    # Shared subsample index (do this BEFORE stacking to cut memory/compute)
    first = np.asarray(samples_flat[labels[0]]).ravel()
    n_total = first.shape[0]
    if n_total > max_points:
        rng = np.random.default_rng()
        idx = rng.choice(n_total, size=max_points, replace=False)
    else:
        idx = None

    print('2')

    # Build columns with shared subsampling; transform log_ → log10; cast to float32
    cols = []
    for k in labels:
        a = np.asarray(samples_flat[k]).ravel()
        if idx is not None:
            a = a[idx]
        if k.startswith("log_"):
            a = a / np.log(10.0)  # ln → log10
        cols.append(a.astype(np.float32, copy=False))

    X = np.column_stack(cols)

    print('3')

    # Drop any rows with NaN/Inf across columns (keeps alignment)
    finite = np.all(np.isfinite(X), axis=1)
    X = X[finite]
    if X.shape[0] == 0:
        raise ValueError("No finite samples to plot after cleaning.")

    print('4')

    # Identify near-constant columns and jitter them (keep them in the plot)
    ptp = np.ptp(X, axis=0)
    const_mask = ptp <= const_ptp
    if np.any(const_mask):
        for j in np.where(const_mask)[0]:
            col = X[:, j]
            mu = float(np.mean(col))
            sigma = max(jitter_abs, abs(mu) * jitter_rel)
            # Add zero-mean jitter; keep dtype float32
            X[:, j] = (col + np.random.normal(0.0, sigma, size=col.shape)).astype(np.float32)
            print("Corner Constant param (jittered):", labels[j])

    print('5')

    # Robust ranges via percentiles; guarantee positive width even after jitter
    lo = np.percentile(X, p_lo, axis=0)
    hi = np.percentile(X, p_hi, axis=0)
    eps = 1e-12
    rng = []
    for j, (l, h) in enumerate(zip(lo, hi)):
        if not np.isfinite(l) or not np.isfinite(h):
            col = X[:, j]
            l, h = np.min(col), np.max(col)
        if h <= l + eps:
            c = float(l)
            pad = max(const_ptp, abs(c) * 1e-6, jitter_abs)
            l, h = c - pad, c + pad
        rng.append((float(l), float(h)))
    rng = np.array(rng)

    # Corner kwargs tuned for speed (hist-only, modest bins)
    fig = corner.corner(
        X[:, ~const_mask],
        labels=labels[~const_mask],
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
        bins=bins,
        range=rng[~const_mask],
        plot_datapoints=False,
        plot_contours=False,         # avoid KDE for speed
        hist2d_kwargs={"bins": bins},
        quiet=True
    )

    # Save
    output_dir = f"plots/multiband/{prefix}/corner/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{z:.1f}_{object_id}_posterior_{suffix}.png")
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    logging.info(f"Saved posterior corner plot to {save_path}")

def plot_posterior(
    samples_flat,
    data,
    sample_mode="fast",
    bins=15,
    max_points=20_000,
    p_lo=0.5,
    p_hi=99.5,
    const_ptp=1e-12,         # threshold for "near-constant"
    jitter_rel=1e-6,         # relative jitter scale (× |mean|)
    jitter_abs=1e-8,         # absolute floor for jitter
    panel_budget=1_000_000,  # ~constant-cost target: keep N so N*D^2 <= panel_budget
    trim_percentiles=True,   # TRIM: drop rows outside [p_lo, p_hi] in *all* dims
    rng_seed=None
):
    """
    Corner plot for posterior samples with fast-safe row selection.

    Both `fast` and `full` use the same bounded sampling and trimming path to
    keep memory and runtime under control. `full` only changes the rendering
    style to look denser.
    """
    import os, logging
    import numpy as np
    import matplotlib.pyplot as plt
    import corner

    logging.info("Saving posterior plot (%s mode)", sample_mode)
    object_id = data["object_id"]
    z = data["z"]

    if sample_mode not in {"fast", "full"}:
        raise ValueError(f"Unsupported sample_mode={sample_mode!r}; expected 'fast' or 'full'.")

    all_labels, labels_for_corner = _corner_plot_labels(samples_flat)
    logging.info("Corner candidate parameters: %s", all_labels)
    logging.info("Corner plotted parameters: %s", labels_for_corner)

    samples_for_corner = {label: samples_flat[label] for label in labels_for_corner}

    # Stable column order
    all_labels = np.array(labels_for_corner)
    D = len(all_labels)

    first = np.asarray(samples_for_corner[all_labels[0]])
    first = first.reshape(first.shape[0], -1)[:, 0]
    n_total = first.shape[0]
    # Both modes share the same bounded row-selection path. "full" differs only
    # in rendering style, not in how many rows are plotted.
    rng = np.random.default_rng(rng_seed)
    n_by_budget = max(2_000, int(panel_budget // max(D * D, 1)))
    n_target = min(n_total, max_points, n_by_budget)
    idx = None
    if n_target < n_total:
        idx = rng.choice(n_total, size=n_target, replace=False)
    prep_max_points = max_points

    sampled = {}
    for k in all_labels:
        a = np.asarray(samples_for_corner[k]).reshape(n_total, -1)[:, 0]
        if idx is not None:
            a = a[idx]
        sampled[k] = a

    X, labels, const_mask = _prep_matrix(
        sampled,
        max_points=prep_max_points,
        const_ptp=const_ptp,
        jitter_rel=jitter_rel,
        jitter_abs=jitter_abs,
        log10_if_startswith="log_",
    )
    labels = np.array(labels)

    # Percentile ranges
    lo = np.percentile(X, p_lo, axis=0)
    hi = np.percentile(X, p_hi, axis=0)
    eps = 1e-12
    l_out = np.minimum(lo, hi)
    h_out = np.maximum(lo, hi)

    # Fix degenerate ranges
    bad = (h_out <= l_out + eps) | ~np.isfinite(l_out) | ~np.isfinite(h_out)
    if np.any(bad):
        cmin = X.min(axis=0)
        cmax = X.max(axis=0)
        l_out[bad] = cmin[bad]
        h_out[bad] = cmax[bad]
        still = h_out <= l_out + eps
        if np.any(still):
            c = 0.5 * (l_out[still] + h_out[still])
            pad = np.maximum.reduce([
                np.full_like(c, const_ptp),
                np.abs(c) * 1e-6,
                np.full_like(c, jitter_abs)
            ])
            l_out[still] = c - pad
            h_out[still] = c + pad
    rng_bounds = np.stack([l_out.astype(float), h_out.astype(float)], axis=1)

    # === TRIM SAMPLES ===
    if trim_percentiles:
        # Keep only rows inside the central percentile box across *all* dims
        m = (X >= l_out) & (X <= h_out)
        keep = m.all(axis=1)
        kept = int(keep.sum())
        if kept >= max(500, 0.05 * X.shape[0]):  # avoid pathological over-trimming
            X = X[keep]
        # If too few remain, fall back to untrimmed (still subsampled above)

    # Hide near-constant columns on the grid (they were jittered so titles etc. still OK)
    sel = ~const_mask
    X_plot = X[:, sel] if np.any(sel) else X
    labels_plot = labels[sel] if np.any(sel) else labels
    ranges_plot = (rng_bounds[sel] if np.any(sel) else rng_bounds)

    if X_plot.shape[1] > X_plot.shape[0]:
        keep_dim = max(1, X_plot.shape[0])
        logging.warning(
            "Skipping %d posterior dimension(s) in corner plot because dims=%d > samples=%d. "
            "Keeping the first %d dimensions.",
            X_plot.shape[1] - keep_dim,
            X_plot.shape[1],
            X_plot.shape[0],
            keep_dim,
        )
        X_plot = X_plot[:, :keep_dim]
        labels_plot = labels_plot[:keep_dim]
        ranges_plot = ranges_plot[:keep_dim]

    plot_datapoints = False
    plot_contours = False
    hist2d_kwargs = {"bins": int(bins)}
    if sample_mode == "full":
        logging.info("Full mode uses fast-safe sampling with denser rendering.")
        plot_contours = True
        hist2d_kwargs = {"bins": max(8, int(bins // 2)), "levels": [0.393, 0.865, 0.989]}

    fig = corner.corner(
        X_plot,
        labels=labels_plot,
        color="black",
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
        bins=int(bins),
        range=[tuple(r) for r in ranges_plot],
        plot_datapoints=plot_datapoints,
        plot_contours=plot_contours,
        hist2d_kwargs=hist2d_kwargs,
        fill_contours=False,
        no_fill_contours=True,
        smooth=(0.8 if sample_mode == "full" else None),
        smooth1d=(0.8 if sample_mode == "full" else None),
        max_n_ticks=3,
        quiet=True,
        use_math_text=False,
        labelpad=0.3,
        title_fmt=".3g",
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 9},
    )

    # Save
    output_dir = f"plots/multiband/{prefix}/corner/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{z:.1f}_{object_id}_posterior_{suffix}.pdf")
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    if isinstance(fig, plt.Figure):
        plt.close(fig)
    logging.info(f"Saved posterior corner plot to {save_path}")



def plot_broken_power_law(samples, data):
    """
    Plot the single power-law wavelength scaling using posterior medians.
      Top: sigma scaling
      Bottom: tau scaling
    Both share x = log10(lambda) and show a linear-lambda axis on top.

    Parameters
    ----------
    samples : dict
        Posterior samples with keys:
        eta_sigma, eta_tau
    data : unused (placeholder for future use)
    """
    # --- posterior medians ---
    def _median_or_default(key, default):
        if key not in samples:
            return default
        arr = np.asarray(samples[key], dtype=float)
        if arr.size == 0 or not np.any(np.isfinite(arr)):
            return default
        return float(np.nanmedian(arr))

    pm = {
        "eta_sigma": _median_or_default("eta_sigma", np.nan),
        "eta_tau": _median_or_default("eta_tau", np.nan),
    }
    eta_sigma = pm["eta_sigma"]
    eta_tau = pm["eta_tau"]
    lam_s = 2500.0

    # --- wavelength grid ---
    xlog = np.linspace(2.9, 3.9, 600)
    lam = 10.0**xlog
    y_amp = log_single_pl(lam, lam_s, eta_sigma)
    y_tau = log_single_pl(lam, lam_s, eta_tau)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 4*2), sharex=True, constrained_layout=True
    )

    def prettify(ax):
        ax.grid(True, which="both", alpha=0.25, linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.axhline(0, ls="-", lw=0.8, color="k", alpha=0.5)
        ax.axvline(np.log10(lam_s), ls="--", lw=1.0, color="gray", alpha=0.8, label=r'$\lambda_s$')

    # --- top panel ---
    ax1.plot(xlog, y_amp, lw=2.0,
             label=fr'$\eta_\sigma={eta_sigma:.2f}$')
    prettify(ax1)
    ax1.set_ylabel(r'$\log_{10}\,f_A(\lambda)$')
    ax1.legend(frameon=False, loc="best")

    # --- bottom panel ---
    ax2.plot(xlog, y_tau, lw=2.0,
             label=fr'$\eta_\tau={eta_tau:.2f}$')
    prettify(ax2)
    ax2.set_xlabel(r'$\log_{10}\,\lambda\ \mathrm{(\AA)}$')
    ax2.set_ylabel(r'$\log_{10}\,f_\tau(\lambda)$')
    ax2.legend(frameon=False, loc="best")

    # --- secondary λ-axis ---
    secax = ax1.secondary_xaxis(
        'top',
        functions=(lambda x: 10.0**x, lambda l: np.log10(l))
    )
    secax.set_xlabel(r'$\lambda\ \mathrm{(\AA)}$')
    secax.xaxis.set_major_formatter(EngFormatter(unit="Å"))

    # --- save ---
    output_dir = f"plots/multiband/{prefix}/broken_power_law"
    os.makedirs(output_dir, exist_ok=True)
    fpath = os.path.join(output_dir, f'broken_power_law_{suffix}.png')
    logging.info(f"Saving figure to {fpath}")
    fig.savefig(fpath, dpi=200)
    plt.close(fig)

def combined_lomb_scargle_from_model(
    model,
    y,
    yerr,
    params: dict,
    omega: np.ndarray,
    *,
    bins_per_decade: int = 2,
    min_per_bin: int = 1,
    normalization: str = "psd",
    amp_scaling_mode: str = "absolute_gp_normalized",
    amp_reference: str = "band0",
    band_wavelength_rf: np.ndarray | None = None,
    lambda_target_rf: float = 2500.0,
):
    """
    Compute Lomb–Scargle PSD from a MyMultiVarModel, using a provided
    angular frequency grid (omega, in rad / time-unit).
    
    Steps:
      - lag-subtract (my_lag_transform)
      - mean-subtract (mean_func)
      - optionally normalize amplitudes using GP-informed band scaling
      - Lomb–Scargle combining all bands
      - log-bin in frequency space

    Parameters
    ----------
    model : MyMultiVarModel
    params : dict
        Parameter dictionary.
    omega : array
        Angular frequencies [rad / time-unit].
    bins_per_decade : int
        Number of log-frequency bins per decade.
    min_per_bin : int
        Minimum raw samples per bin.
    normalization : str
        LS normalization (default "psd").

    Returns
    -------
    dict with keys:
      "omega" : input angular frequencies
      "f_raw" : frequencies in cycles/time-unit
      "P_raw" : raw LS power
      "f_bin","P_bin","P_lo","P_hi","bin_counts"
    """
    # Convert omega -> frequency grid
    omega = np.asarray(omega, float)
    f_raw = omega / (2.0 * np.pi)

    # Lag subtraction
    if hasattr(model, "my_lag_transform"):
        (t_lag, band_idx), _ = model.my_lag_transform(model.X, model.has_lag, params)
    else:
        (t_lag, band_idx), _ = model.lag_transform(model.has_lag, params, model.X)
    t_lag = np.asarray(t_lag, float)
    band_idx = np.asarray(band_idx, int)

    # Mean subtraction via mean_func
    t_center = float(np.mean(t_lag))
    t_std = float(np.std(t_lag))
    try:
        mean_vals = model.mean_func(params, (t_lag, band_idx))
    except TypeError:
        mean_vals = model.mean_func(
            model.zero_mean,
            int(np.max(band_idx)) + 1,
            t_center,
            t_std,
            params,
            (t_lag, band_idx),
        )
    if hasattr(model, "mean_to_display"):
        mean_vals = model.mean_to_display(mean_vals)
    y = np.asarray(y, float).copy() - np.asarray(mean_vals, float)
    if (
        getattr(model, "survey_idx", None) is not None
        and "survey_delta_mag" in params
    ):
        survey_delta_mag = np.asarray(params["survey_delta_mag"], dtype=float)
        if survey_delta_mag.ndim == 2:
            survey_idx = np.asarray(model.survey_idx, dtype=np.int32)
            y = y - survey_delta_mag[band_idx, survey_idx]
    yerr = np.asarray(yerr, float).copy()

    #if "log_jitter" in params:
    #    jitter_band = np.exp(np.asarray(params["log_jitter"], dtype=float))
    #    if jitter_band.ndim == 0:
    #        jitter_band = np.full(int(np.max(band_idx)) + 1, float(jitter_band))
    #    jitter_band = np.clip(np.asarray(jitter_band, dtype=float), 0.0, None)
    #    yerr = np.sqrt(yerr**2 + jitter_band[band_idx] ** 2)

    if amp_scaling_mode == "absolute_gp_normalized":
        if hasattr(model, "my_amp_transform"):
            log_sigma_band = np.asarray(model.my_amp_transform(params))
        else:
            log_sigma_band = np.log(np.asarray(params["amp_cont"]))
        if amp_reference == "uv" and "log_sigma_uv" in params:
            s0 = float(np.exp(np.asarray(params["log_sigma_uv"])))
        else:
            s0 = float(np.exp(log_sigma_band[0]))
        s_b = np.exp(log_sigma_band)
        scale = s0 / s_b[band_idx]
    elif amp_scaling_mode == "relative_to_2500":
        if band_wavelength_rf is None:
            raise ValueError("band_wavelength_rf is required for amp_scaling_mode='relative_to_2500'.")
        if "eta_sigma" not in params:
            raise KeyError("Missing required parameter 'eta_sigma' for raw LS wavelength calibration.")
        scale_band = relative_to_2500_amplitude_scale(
            band_wavelength_rf,
            float(np.asarray(params["eta_sigma"])),
            lambda_target_rf=lambda_target_rf,
        )
        scale = scale_band[band_idx]
    else:
        raise ValueError(f"Unknown amp_scaling_mode '{amp_scaling_mode}'.")
    y *= scale

    # Sort by time (optional, not required for LS)
    order = np.argsort(t_lag)
    t_lag, y = t_lag[order], y[order]
    yerr *= scale
    yerr = yerr[order]

    # Lomb–Scargle
    ls = LombScargle(t_lag, y, yerr, fit_mean=False)
    P_raw = ls.power(f_raw, normalization=normalization)

    # The additive floor is fit downstream with the broken-PSD model.
    # Avoid subtracting an empirical high-frequency median here, because it
    # blends measurement noise with real short-timescale signal and window
    # leakage.
    P_noise = np.nan

    # Log-binning in f
    fmin, fmax = np.min(f_raw), np.max(f_raw)
    decades = np.log10(fmax) - np.log10(fmin)
    n_bins = int(np.ceil(bins_per_decade * decades))
    edges = np.logspace(np.log10(fmin), np.log10(fmax), n_bins + 1)

    which = np.digitize(f_raw, edges) - 1
    f_bin, P_bin, P_lo, P_hi, counts = [], [], [], [], []
    for k in range(n_bins):
        sel = (which == k)
        if np.count_nonzero(sel) >= min_per_bin:
            f_chunk = f_raw[sel]
            P_chunk = P_raw[sel]
            f_center = 10.0 ** (np.mean(np.log10(f_chunk)))
            f_bin.append(f_center)
            P_bin.append(np.median(P_chunk))
            P_lo.append(np.percentile(P_chunk, 16))
            P_hi.append(np.percentile(P_chunk, 84))
            counts.append(np.count_nonzero(sel))

    return np.array(f_bin), np.array(P_bin), np.array(P_lo), np.array(P_hi), np.array(counts), P_noise


def relative_to_2500_amplitude_scale(band_wavelength_rf, eta_sigma, *, lambda_target_rf=2500.0):
    """Return multiplicative per-band amplitude factors implied by the GP wavelength law."""

    band_wavelength_rf = np.asarray(band_wavelength_rf, dtype=float)
    log_scale_band = np.asarray(
        [log_single_pl(lambda_target_rf, lam_rf, eta_sigma) for lam_rf in band_wavelength_rf],
        dtype=float,
    )
    return np.power(10.0, log_scale_band)

import numpy as np

def bootstrap_lomb_scargle(
    model,
    y,
    yerr,
    posterior_median,
    freqs,
    n_boot=500,
    random_state=None,
):
    """
    Bootstrap the Lomb–Scargle PSD n_boot times by resampling the light curve
    (with replacement) and collecting the binned PSD results.
    """
    rng = np.random.default_rng(random_state)

    f_bin_ref = None
    P_bin_boot = []
    P_lo_boot = []
    P_hi_boot = []
    cts_boot = []
    P_noise_boot = []

    n = len(y)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_boot   = y[idx]
        yerr_boot = yerr[idx]

        f_bin_i, P_bin_i, P_lo_i, P_hi_i, cts_i, P_noise_i = combined_lomb_scargle_from_model(
            model,
            y_boot,
            yerr_boot,
            posterior_median,
            2 * np.pi * freqs,
        )

        if f_bin_ref is None:
            f_bin_ref = f_bin_i  # assume identical bins each time

        P_bin_boot.append(P_bin_i)
        P_lo_boot.append(P_lo_i)
        P_hi_boot.append(P_hi_i)
        cts_boot.append(cts_i)
        P_noise_boot.append(P_noise_i)

    # Stack into arrays of shape (n_boot, n_bins)
    P_bin_boot = np.vstack(P_bin_boot)
    P_lo_boot = np.vstack(P_lo_boot)
    P_hi_boot = np.vstack(P_hi_boot)
    cts_boot = np.vstack(cts_boot)
    P_noise_boot = np.vstack(P_noise_boot)

    # at the end of bootstrap_lomb_scargle
    return {
        "f_bin": f_bin_ref,
        "P_bin_boot": P_bin_boot,
        "P_noise_boot": P_noise_boot,
        "P_bin_med": np.median(P_bin_boot, axis=0),
        "P_bin_lo": np.percentile(P_bin_boot, 16, axis=0),
        "P_bin_hi": np.percentile(P_bin_boot, 84, axis=0),
        "P_noise_med": np.median(P_noise_boot, axis=0),
        "P_noise_lo": np.percentile(P_noise_boot, 16, axis=0),
        "P_noise_hi": np.percentile(P_noise_boot, 84, axis=0),
        # add cts etc. similarly if needed
    }


def _bending_power_law_psd_plot(freq, log_sigma, log_tau, alpha_high=-2.0, log_noise_floor=-99.0):
    freq = np.asarray(freq, dtype=float)
    sigma = 10.0 ** float(log_sigma)
    tau = 10.0 ** float(log_tau)
    slope = -float(alpha_high)
    noise_floor = 10.0 ** float(log_noise_floor)
    return 2.0 * sigma * sigma * tau / (
        1.0 + np.power(np.clip(2.0 * np.pi * freq * tau, 1e-30, None), slope)
    ) + noise_floor


def _binned_median_relation(x, y, *, n_bins=40, min_per_bin=20):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < min_per_bin:
        return np.array([]), np.array([]), np.array([]), np.array([])

    x = x[mask]
    y = y[mask]
    edges = np.linspace(np.min(x), np.max(x), n_bins + 1)
    which = np.digitize(x, edges) - 1

    x_mid, y_med, y_lo, y_hi = [], [], [], []
    for k in range(n_bins):
        sel = which == k
        if np.count_nonzero(sel) < min_per_bin:
            continue
        x_chunk = x[sel]
        y_chunk = y[sel]
        x_mid.append(float(np.median(x_chunk)))
        y_med.append(float(np.median(y_chunk)))
        y_lo.append(float(np.percentile(y_chunk, 16)))
        y_hi.append(float(np.percentile(y_chunk, 84)))

    return (
        np.asarray(x_mid, dtype=float),
        np.asarray(y_med, dtype=float),
        np.asarray(y_lo, dtype=float),
        np.asarray(y_hi, dtype=float),
    )


def save_color_magnitude_plot(
    samples,
    model,
    X,
    y,
    yerr,
    band_idx,
    mags_means,
    data,
    bands=['u', 'g', 'r', 'i', 'z'],
    show=False,
    time0=0.0,
    filename_suffix=None,
):
    import os
    import logging
    import numpy as np
    import matplotlib.pyplot as plt

    logging.info("Saving color-magnitude plot")

    if 'i' not in bands:
        logging.warning("Skipping color-magnitude plot because i band is unavailable.")
        return

    object_id = data['object_id']
    band_idx_map = {i: b for i, b in enumerate(bands)}
    posterior_median = {k: np.median(v, axis=0) for k, v in samples.items()}
    cm_params = dict(posterior_median)
    if "lag0" in cm_params:
        cm_params["lag0"] = np.zeros_like(np.asarray(cm_params["lag0"], dtype=float))
    if "dlog_amp_blr" in cm_params:
        cm_params["dlog_amp_blr"] = np.full_like(
            np.asarray(cm_params["dlog_amp_blr"], dtype=float),
            -20.0,
        )
    if "dlog_amp_blr2" in cm_params:
        cm_params["dlog_amp_blr2"] = np.full_like(
            np.asarray(cm_params["dlog_amp_blr2"], dtype=float),
            -20.0,
        )
    if "amp_blr" in cm_params:
        cm_params["amp_blr"] = np.full_like(np.asarray(cm_params["amp_blr"], dtype=float), 1e-20)
    if "amp_blr2" in cm_params:
        cm_params["amp_blr2"] = np.full_like(np.asarray(cm_params["amp_blr2"], dtype=float), 1e-20)
    if "amp_blr_relflux" in cm_params:
        cm_params["amp_blr_relflux"] = np.full_like(
            np.asarray(cm_params["amp_blr_relflux"], dtype=float),
            1e-20,
        )
    if "amp_blr2_relflux" in cm_params:
        cm_params["amp_blr2_relflux"] = np.full_like(
            np.asarray(cm_params["amp_blr2_relflux"], dtype=float),
            1e-20,
        )
    t = X[0] + time0
    t_test = np.linspace(t.min() - 400, t.max() + 400, 4000)

    model_cont_by_band = {}
    model_cont_std_by_band = {}
    for n in np.unique(band_idx):
        result = _prediction_to_display(
            model,
            model.pred(cm_params, (t_test - time0, jnp.full_like(t_test, n, dtype=int))),
        )
        if len(result) == 2:
            mu, std = result
            model_cont_by_band[n] = np.asarray(mu, dtype=float)
            model_cont_std_by_band[n] = np.asarray(std, dtype=float)
        else:
            _, _, mu_cont, std_cont, _, _ = result
            model_cont_by_band[n] = np.asarray(mu_cont, dtype=float)
            model_cont_std_by_band[n] = np.asarray(std_cont, dtype=float)

    i_idx = bands.index('i')
    mu_i = model_cont_by_band.get(i_idx)
    std_i = model_cont_std_by_band.get(i_idx)
    if mu_i is None:
        logging.warning("Skipping color-magnitude plot because no i-band model prediction is available.")
        return

    i_mag_model = mu_i + mags_means[i_idx]
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for n in np.unique(band_idx):
        if n == i_idx or n not in model_cont_by_band:
            continue
        band_name = band_idx_map[n]
        mu_band = model_cont_by_band[n] + mags_means[n]
        color_model = mu_band - i_mag_model
        x, y_color, y_lo, y_hi = _binned_median_relation(i_mag_model, color_model)
        if x.size == 0:
            continue
        order = np.argsort(x)
        ax.plot(
            x[order],
            y_color[order],
            color=colors[band_name],
            lw=2.0,
            alpha=0.95,
            label=f"{band_name} - i",
        )
        ax.fill_between(
            x[order],
            y_lo[order],
            y_hi[order],
            color=colors[band_name],
            alpha=0.18,
            lw=0.0,
        )

    ax.set_xlabel("i-band magnitude (continuum model)")
    ax.set_ylabel("Color (continuum model)")
    ax.invert_xaxis()
    ax.grid(False)
    ax.legend(loc='best')
    plt.tight_layout()

    output_dir = f"plots/multiband/{prefix}/color_magnitude"
    os.makedirs(output_dir, exist_ok=True)
    save_suffix = suffix if filename_suffix is None else filename_suffix
    fpath = os.path.join(output_dir, f'{data["z"]:.1f}_{object_id}_color_magnitude_{save_suffix}.pdf')
    plt.savefig(fpath, dpi=600)
    logging.info(f"Saving figure to {fpath}")
    if show:
        plt.show()
    plt.close(fig)


def save_g_band_binned_residual_drift_plot(diagnostic, data, show=False, filename_suffix=None):
    """Save a per-object g-band residual mean/variance drift diagnostic plot."""

    object_id = data["object_id"]
    z = float(data["z"])
    x = np.asarray(diagnostic.get("g_resid_bin_center_rf", []), dtype=float)
    y_mean = np.asarray(diagnostic.get("g_resid_bin_mean", []), dtype=float)
    y_mean_err = np.asarray(diagnostic.get("g_resid_bin_mean_err", []), dtype=float)
    y_var = np.asarray(diagnostic.get("g_resid_bin_variance", []), dtype=float)
    y_var_err = np.asarray(diagnostic.get("g_resid_bin_variance_err", []), dtype=float)
    counts = np.asarray(diagnostic.get("g_resid_bin_count", []), dtype=int)

    if x.size == 0:
        logging.warning("Skipping g-band residual drift plot because no binned residual diagnostics are available.")
        return

    fig, (ax_mean, ax_var) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax_mean.errorbar(
        x,
        y_mean,
        yerr=y_mean_err,
        fmt="o",
        color=colors["g"],
        ecolor=colors["g"],
        capsize=3,
        markersize=4,
        alpha=0.85,
    )
    if bool(diagnostic.get("g_resid_mean_trend_valid", False)):
        x0 = float(diagnostic["g_resid_mean_fit_t_center_rf"])
        x_line = np.linspace(np.min(x), np.max(x), 200)
        y_line = diagnostic["g_resid_mean_intercept"] + diagnostic["g_resid_mean_slope"] * (x_line - x0)
        ax_mean.plot(x_line, y_line, color="black", lw=1.5)
    ax_mean.axhline(0.0, color="0.5", linestyle="--", lw=1.0)
    ax_mean.set_ylabel("Residual mean (mag)")
    ax_mean.set_title(
        f"g-band detrended residuals: binned mean and variance\n"
        f"N bins = {x.size}, total points = {int(np.sum(counts))}"
    )

    ax_var.errorbar(
        x,
        y_var,
        yerr=y_var_err,
        fmt="o",
        color=colors["g"],
        ecolor=colors["g"],
        capsize=3,
        markersize=4,
        alpha=0.85,
    )
    if bool(diagnostic.get("g_resid_var_trend_valid", False)):
        x0 = float(diagnostic["g_resid_var_fit_t_center_rf"])
        x_line = np.linspace(np.min(x), np.max(x), 200)
        y_line = diagnostic["g_resid_var_intercept"] + diagnostic["g_resid_var_slope"] * (x_line - x0)
        ax_var.plot(x_line, y_line, color="black", lw=1.5)
    ax_var.set_xlabel("Rest-frame time (days)")
    ax_var.set_ylabel(r"Residual variance (mag$^2$)")

    fig.tight_layout()

    output_dir = f"plots/multiband/{prefix}/g_band_residual_drift"
    os.makedirs(output_dir, exist_ok=True)
    save_suffix = suffix if filename_suffix is None else filename_suffix
    fpath = os.path.join(output_dir, f"{z:.1f}_{object_id}_g_band_residual_drift_{save_suffix}.pdf")
    plt.savefig(fpath, dpi=600)
    logging.info(f"Saving figure to {fpath}")
    if show:
        plt.show()
    plt.close(fig)


def save_multiband_residual_normality_plot(diagnostic, data, show=False, filename_suffix=None):
    """Save per-band histogram and Q-Q diagnostics for detrended residuals."""

    object_id = data["object_id"]
    z = float(data["z"])
    bands = list(data.get("bands", []))
    if not bands:
        bands = [
            key.removeprefix("resid_normality_residual_")
            for key in diagnostic
            if key.startswith("resid_normality_residual_")
        ]

    available_bands = [
        band
        for band in bands
        if f"resid_normality_residual_{band}" in diagnostic
    ]
    if not available_bands:
        logging.warning("Skipping multiband residual normality plot because no detrended residuals are available.")
        return

    fig, axes = plt.subplots(
        len(available_bands),
        2,
        figsize=(10.8, max(3.2 * len(available_bands), 3.8)),
        squeeze=False,
    )

    for row, band in enumerate(available_bands):
        ax_hist, ax_qq = axes[row]
        residual = np.asarray(diagnostic.get(f"resid_normality_residual_{band}", []), dtype=float)
        zscore = np.asarray(diagnostic.get(f"resid_normality_zscore_{band}", []), dtype=float)
        residual = residual[np.isfinite(residual)]
        zscore = zscore[np.isfinite(zscore)]

        if zscore.size == 0:
            ax_hist.text(
                0.5,
                0.5,
                f"No finite detrended residuals\nfor {band} band",
                ha="center",
                va="center",
                transform=ax_hist.transAxes,
            )
            ax_qq.set_axis_off()
            continue

        x_limit = max(4.0, float(np.nanmax(np.abs(zscore))) * 1.1)
        x_limit = min(x_limit, 8.0)
        x_grid = np.linspace(-x_limit, x_limit, 512)
        ax_hist.hist(
            zscore,
            bins=min(20, max(8, int(np.sqrt(zscore.size)))),
            density=True,
            color=colors.get(band, "0.4"),
            alpha=0.35,
            edgecolor="white",
        )
        ax_hist.plot(
            x_grid,
            norm.pdf(x_grid, loc=0.0, scale=1.0),
            color="black",
            lw=1.6,
        )
        ax_hist.axvline(0.0, color="black", ls="--", lw=1.0, alpha=0.8)
        ax_hist.set_xlim(-x_limit, x_limit)
        ax_hist.set_ylabel(f"{band}-band density")
        if row == len(available_bands) - 1:
            ax_hist.set_xlabel("Standardized detrended residual")
        ax_hist.text(
            0.98,
            0.98,
            (
                f"N={int(diagnostic.get(f'resid_normality_nobs_{band}', 0))}\n"
                f"mean={diagnostic.get(f'resid_normality_mean_{band}', np.nan):.3f}\n"
                f"std={diagnostic.get(f'resid_normality_std_{band}', np.nan):.3f}\n"
                f"skew={diagnostic.get(f'resid_normality_skew_{band}', np.nan):.2f}\n"
                f"kurt={diagnostic.get(f'resid_normality_kurtosis_{band}', np.nan):.2f}\n"
                f"K2 p={diagnostic.get(f'resid_normality_pvalue_{band}', np.nan):.2g}"
            ),
            ha="right",
            va="top",
            transform=ax_hist.transAxes,
        )
        if row == 0:
            ax_hist.set_title("Histogram")

        osm, osr = probplot(zscore, dist="norm", fit=False)
        q_lo = min(float(np.min(osm)), float(np.min(osr)))
        q_hi = max(float(np.max(osm)), float(np.max(osr)))
        ax_qq.scatter(
            osm,
            osr,
            s=12,
            alpha=0.6,
            color=colors.get(band, "0.3"),
            linewidths=0,
            rasterized=True,
        )
        ax_qq.plot([q_lo, q_hi], [q_lo, q_hi], color="tab:red", lw=1.4)
        if row == len(available_bands) - 1:
            ax_qq.set_xlabel("Normal quantiles")
        ax_qq.set_ylabel(f"{band}-band quantiles")
        if row == 0:
            ax_qq.set_title("Q-Q")

    fig.suptitle("Detrended light-curve residual normality by band", y=0.995)
    fig.tight_layout()

    output_dir = f"plots/multiband/{prefix}/residual_normality"
    os.makedirs(output_dir, exist_ok=True)
    save_suffix = suffix if filename_suffix is None else filename_suffix
    fpath = os.path.join(output_dir, f"{z:.1f}_{object_id}_residual_normality_{save_suffix}.pdf")
    plt.savefig(fpath, dpi=600)
    logging.info(f"Saving figure to {fpath}")
    if show:
        plt.show()
    plt.close(fig)


def save_dm_df_over_f_distribution_plot(data, show=False, filename_suffix=None):
    """Save per-band histograms of dm and dF/F with Gaussian fits."""

    object_id = data["object_id"]
    z = float(data["z"])
    bands = list(data.get("bands", []))
    band_idx = np.asarray(data.get("band_idx", []), dtype=int)
    dm_all = np.asarray(data.get("y", []), dtype=float)

    if dm_all.size == 0 or len(bands) == 0:
        logging.warning("Skipping dm/dF/F distribution plot because no cleaned light-curve residuals are available.")
        return

    fig, axes = plt.subplots(
        len(bands),
        2,
        figsize=(10.5, max(3.0 * len(bands), 4.2)),
        squeeze=False,
    )

    def _draw_hist_with_gaussian(ax, values, *, color, xlabel, band):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            ax.text(
                0.5,
                0.5,
                f"No finite {xlabel}\nfor {band} band",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        bins = min(30, max(8, int(np.sqrt(values.size))))
        ax.hist(
            values,
            bins=bins,
            density=True,
            color=color,
            alpha=0.35,
            edgecolor="white",
        )

        mu = float(np.mean(values))
        sigma = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        if np.isfinite(sigma) and sigma > 1e-12:
            lo = float(np.min(values))
            hi = float(np.max(values))
            pad = max(1e-8, 0.1 * (hi - lo))
            grid = np.linspace(lo - pad, hi + pad, 512)
            ax.plot(grid, norm.pdf(grid, loc=mu, scale=sigma), color="black", lw=1.6)
            ax.axvline(mu, color="black", ls="--", lw=1.0, alpha=0.9)
        else:
            ax.axvline(mu, color="black", ls="--", lw=1.0, alpha=0.9)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{band}-band density")
        ax.text(
            0.98,
            0.98,
            f"N={values.size}\nmu={mu:.3g}\nsigma={sigma:.3g}",
            ha="right",
            va="top",
            transform=ax.transAxes,
        )

    for row, band in enumerate(bands):
        mask = band_idx == row
        dm_band = dm_all[mask]
        df_over_f_band = np.power(10.0, -0.4 * dm_band) - 1.0

        _draw_hist_with_gaussian(
            axes[row, 0],
            dm_band,
            color=colors.get(band, "0.4"),
            xlabel="dm (mag)",
            band=band,
        )
        _draw_hist_with_gaussian(
            axes[row, 1],
            df_over_f_band,
            color=colors.get(band, "0.4"),
            xlabel="dF/F",
            band=band,
        )

        if row == 0:
            axes[row, 0].set_title("Magnitude Residual Distribution")
            axes[row, 1].set_title("Relative-Flux Residual Distribution")

    fig.suptitle("Per-band dm and dF/F distributions with Gaussian fits", y=0.995)
    fig.tight_layout()

    output_dir = f"plots/multiband/{prefix}/dm_df_over_f_distributions"
    os.makedirs(output_dir, exist_ok=True)
    save_suffix = suffix if filename_suffix is None else filename_suffix
    fpath = os.path.join(output_dir, f"{z:.1f}_{object_id}_dm_df_over_f_{save_suffix}.pdf")
    plt.savefig(fpath, dpi=600)
    logging.info(f"Saving figure to {fpath}")
    if show:
        plt.show()
    plt.close(fig)


def _sf_bending_power_law_model_curve(tau, log_sigma_sf, log_tau_sf, alpha_short):
    """Evaluate the fitted flat-large-lag SF curve from an RMS-normalized amplitude."""

    tau = np.asarray(tau, dtype=float)
    if (
        not np.isfinite(log_sigma_sf)
        or not np.isfinite(log_tau_sf)
        or not np.isfinite(alpha_short)
    ):
        return np.full_like(tau, np.nan, dtype=float)
    sf_inf = np.sqrt(2.0) * np.power(10.0, float(log_sigma_sf))
    tau_break = np.power(10.0, float(log_tau_sf))
    ratio = np.clip(tau / tau_break, 1e-12, None)
    return sf_inf / (1.0 + np.power(ratio, float(alpha_short)))


def save_structure_function_plot(diagnostic, data, show=False, filename_suffix=None):
    """Save a per-object all-band empirical SF plot tied to the g-band amplitude scale."""

    object_id = data["object_id"]
    z = float(data["z"])
    ref_band = str(diagnostic.get("sf_ref_band", "g"))
    sf_source_bands = str(diagnostic.get("sf_source_bands", ref_band) or ref_band)
    sf_weighted = bool(diagnostic.get("sf_inverse_variance_weighted", True))
    tau_sf = np.asarray(diagnostic.get("sf_tau_ref_band", []), dtype=float)
    sf_med = np.asarray(diagnostic.get("sf_curve_ref_band", []), dtype=float)
    sf_lo = np.asarray(diagnostic.get("sf_curve_lo_ref_band", []), dtype=float)
    sf_hi = np.asarray(diagnostic.get("sf_curve_hi_ref_band", []), dtype=float)
    tau_model = np.asarray(diagnostic.get("sf_model_tau_ref_band", []), dtype=float)
    sf_model = np.asarray(diagnostic.get("sf_model_curve_ref_band", []), dtype=float)

    valid_sf = (
        np.isfinite(tau_sf)
        & np.isfinite(sf_med)
        & (tau_sf > 0.0)
        & (sf_med > 0.0)
    )
    valid_model = (
        np.isfinite(tau_model)
        & np.isfinite(sf_model)
        & (tau_model > 0.0)
        & (sf_model > 0.0)
    )
    if not np.any(valid_sf) and not np.any(valid_model):
        logging.warning("Skipping structure-function plot because no finite empirical/model SF points are available.")
        return

    fig, ax = plt.subplots(figsize=(7.4, 5.6))

    if np.any(valid_sf):
        yerr_lo = np.clip(sf_med[valid_sf] - sf_lo[valid_sf], 0.0, None)
        yerr_hi = np.clip(sf_hi[valid_sf] - sf_med[valid_sf], 0.0, None)
        ax.errorbar(
            tau_sf[valid_sf],
            sf_med[valid_sf],
            yerr=[yerr_lo, yerr_hi],
            fmt="o",
            color=colors.get(ref_band, "k"),
            ecolor=colors.get(ref_band, "k"),
            capsize=3,
            markersize=5,
            alpha=0.9,
            label=(
                f"Empirical SF ({sf_source_bands} -> {ref_band} RMS, "
                f"{'IVW' if sf_weighted else 'unweighted'})"
            ),
        )

        tau_dense = np.logspace(
            np.log10(SF_LAG_PLOT_MIN_RF),
            np.log10(SF_LAG_PLOT_MAX_RF),
            400,
        )
        sf_fit_curve = _sf_bending_power_law_model_curve(
            tau_dense,
            diagnostic.get("log_sigma_sf_ref_band", np.nan),
            diagnostic.get("log_tau_sf_ref_band", np.nan),
            diagnostic.get("sf_alpha_short_ref_band", np.nan),
        )
        if np.any(np.isfinite(sf_fit_curve) & (sf_fit_curve > 0.0)):
            ax.plot(
                tau_dense,
                sf_fit_curve,
                color="black",
                lw=1.8,
                zorder=5,
                label="Bending-PL fit to empirical SF",
            )

    if np.any(valid_model):
        order = np.argsort(tau_model[valid_model])
        ax.plot(
            tau_model[valid_model][order],
            sf_model[valid_model][order],
            color="magenta",
            lw=1.8,
            alpha=0.9,
            label="GP-implied SF",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(SF_LAG_PLOT_MIN_RF, SF_LAG_PLOT_MAX_RF)
    ax.set_xlabel("Rest-frame lag (days)")
    ax.set_ylabel("SF (mag)")
    ax.set_title(
        f"Object {object_id}, z={z:.3f}, all-band SF -> {ref_band}-band RMS\n"
        f"N bins={int(diagnostic.get('sf_nbins', 0))}, "
        f"{'IVW' if sf_weighted else 'unweighted'}, "
        f"log tau_SF={diagnostic.get('log_tau_sf_ref_band', np.nan):.3g}, "
        f"alpha_short={diagnostic.get('sf_alpha_short_ref_band', np.nan):.2g}, "
        f"log tau_GP-SF={diagnostic.get('log_tau_sf_model_ref_band', np.nan):.3g}"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    output_dir = f"plots/multiband/{prefix}/structure_function"
    os.makedirs(output_dir, exist_ok=True)
    save_suffix = suffix if filename_suffix is None else filename_suffix
    fpath = os.path.join(output_dir, f"{z:.1f}_{object_id}_structure_function_{save_suffix}.pdf")
    plt.savefig(fpath, dpi=600)
    logging.info(f"Saving figure to {fpath}")
    if show:
        plt.show()
    plt.close(fig)


def save_combined_plot(samples, model, X, y, yerr, band_idx, mags_means, survey_times,
                       data, bands=['u', 'g', 'r', 'i', 'z'], plot_psd=True, show=False,
                       time0=0.0, plot_bpl_fit=False, filename_suffix=None):
    import os
    import logging
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms

    logging.info("Saving combined plot")

    object_id = data['object_id']
    band_idx_map = {i: b for i, b in enumerate(bands)}

    fig, (ax_lc, ax_psd) = plt.subplots(
        2, 1, figsize=(10, 10), sharex=False,
        gridspec_kw={'height_ratios': [1.5, 1]}
    )
    offsets = np.arange(len(bands)) * 0.25 + mags_means[bands.index('r')]

    t = X[0] + time0
    posterior_median = {k: np.median(v, axis=0) for k, v in samples.items()}
    t_test = np.linspace(t.min() - 400, t.max() + 400, 1000)

    for n in np.unique(band_idx):
        mask = (band_idx == n) & (yerr < 10.0)

        # Plot the observed data
        n_red = bands.index('r')
        offset = offsets[n] - offsets[n_red]
        if band_idx_map[n] != 'r':
            sign = '$+$' if offset > 0 else '$-$'
            label = f"{band_idx_map[n]}-band {sign} {np.abs(offset):.2f} mag"
        else:
            label = f"{band_idx_map[n]}-band"

        ax_lc.errorbar(
            t[mask], y[mask] + offsets[n], yerr=yerr[mask], fmt='o',
            label=label, alpha=0.7, color=colors[band_idx_map[n]],
            lw=1, capsize=2, markersize=3
        )

        # Compute predictions using the model
        result = _prediction_to_display(
            model,
            model.pred(posterior_median, (t_test - time0, jnp.full_like(t_test, n, dtype=int))),
        )

        # Plot the predictions
        if len(result) == 2:
            mu, std = result
            ax_lc.plot(t_test, mu + offsets[n], alpha=0.8, color=colors[band_idx_map[n]], lw=1.5)
            ax_lc.fill_between(
                t_test, mu + offsets[n] - std, mu + offsets[n] + std,
                alpha=0.3, lw=0.5, color=colors[band_idx_map[n]]
            )

            cont_result = _prediction_to_display(
                model,
                model.pred(
                    _component_only_params(posterior_median, component="continuum"),
                    (t_test - time0, jnp.full_like(t_test, n, dtype=int)),
                ),
            )
            mu_cont = np.asarray(cont_result[0], dtype=float)

            ax_lc.plot(
                t_test,
                mu_cont + offsets[n],
                alpha=0.75,
                color=colors[band_idx_map[n]],
                lw=1.0,
                linestyle='--',
            )
        else:
            mu, std, mu_cont, std_cont, _mu_blr, _std_blr = result

            ax_lc.plot(
                t_test, mu_cont + offsets[n], alpha=0.5,
                color=colors[band_idx_map[n]], lw=1.0,
                label=f'{band_idx_map[n]}-band continuum', linestyle='--'
            )
            ax_lc.fill_between(
                t_test, mu_cont + offsets[n] - std_cont,
                mu_cont + offsets[n] + std_cont,
                alpha=0.15, lw=0.5, color=colors[band_idx_map[n]]
            )

            ax_lc.plot(t_test, mu + offsets[n], alpha=0.8, color=colors[band_idx_map[n]], lw=1.0)
            ax_lc.fill_between(
                t_test, mu + offsets[n] - std, mu + offsets[n] + std,
                alpha=0.3, lw=0.5, color=colors[band_idx_map[n]]
            )

    ax_lc.set_ylim(ax_lc.get_ylim()[0] - 0.24, ax_lc.get_ylim()[1])
    ax_lc.set_xlabel('Time (modified Julian days)')
    ax_lc.set_ylabel('Apparent magnitude')
    ax_lc.invert_yaxis()
    ax_lc.set_xlim(np.min(t_test), np.max(t_test))

    # ---------------------------------------------------------
    # Survey span lines (using survey_times dict)
    # ---------------------------------------------------------
    trans = mtransforms.blended_transform_factory(ax_lc.transData, ax_lc.transAxes)

    survey_order = ['sdss', 'ps1', 'ztf']
    survey_y = 0.03

    for survey in survey_order:
        times_s = np.asarray(survey_times.get(survey, []), dtype=float)
        times_s = times_s[np.isfinite(times_s)]

        if len(times_s) == 0:
            continue

        t0, t1 = np.min(times_s), np.max(times_s)
        yfrac = survey_y

        ax_lc.plot(
            [t0, t1], [yfrac, yfrac],
            transform=trans,
            color='0.5', lw=3, alpha=0.9, solid_capstyle='butt',
            zorder=10
        )

        ax_lc.text(
            0.5 * (t0 + t1), yfrac - 0.015, survey.upper(),
            transform=trans,
            color='0.35', fontsize=10,
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=1),
            zorder=11
        )

    ax_lc.legend(loc='upper right')

    if plot_psd:
        # Ensure all elements of posterior_median are jnp arrays
        for k in posterior_median:
            posterior_median[k] = jnp.array(posterior_median[k])

        print("Plotting PSD...")
        freqs = np.logspace(-6, 2, 500)

        # Model PSD
        psd_samples = []
        tau_samples_for_psd = np.asarray(samples['log_tau_uv'])
        n_total = int(len(tau_samples_for_psd))
        n_samp = np.min([50, n_total])
        for i in range(n_samp):
            sample_params = _posterior_sample_params_at_index(samples, i, n_total)
            psd_i = (2.0 * jnp.pi) * model.psd(sample_params, 2 * np.pi * freqs, b=0, sigma_n2=0.0)
            psd_samples.append(np.asarray(psd_i))
        psd_samples = np.stack(psd_samples, axis=0)

        psd_median = np.median(psd_samples, axis=0)
        psd_lo = np.percentile(psd_samples, 16, axis=0)
        psd_hi = np.percentile(psd_samples, 84, axis=0)

        ax_psd.plot(freqs, psd_median, lw=2, color='m', alpha=0.8, label="Model PSD", zorder=4)
        ax_psd.fill_between(freqs, psd_lo, psd_hi, color='m', alpha=0.2, zorder=3)

        f_bin, P_bin_med, P_lo, P_hi, counts, P_noise = combined_lomb_scargle_from_model(
            model,
            y,
            yerr,
            posterior_median,
            2.0 * np.pi * freqs,
            amp_scaling_mode="absolute_gp_normalized",
        )
        # Renormalize data PSD to match model PSD at first bin
        model_at_f0 = np.interp(f_bin[0], freqs, psd_median)
        scale = model_at_f0 / max(P_bin_med[0], 1e-30)

        P_bin_med   = P_bin_med   * scale
        P_lo        = P_lo        * scale
        P_hi        = P_hi        * scale

        ax_psd.errorbar(
            f_bin,
            P_bin_med,
            yerr=[P_bin_med - P_lo, P_hi - P_bin_med],
            markersize=4,
            fmt="o",
            color='k',
            ecolor="k",
            elinewidth=0.8,
            capsize=4.0,
            capthick=0.8,
            alpha=0.9,
            zorder=5,
            label="Lomb–Scargle PSD",
        )

        if plot_bpl_fit:
            log_sigma_ls = data.get("log_sigma_ls")
            log_tau_ls_obs = data.get("log_tau_ls_obs", data.get("log_tau_bpl_ref_band"))
            alpha_high_ls = data.get("alpha_high_ls", data.get("psd_bpl_alpha_high"))
            log_noise_floor_ls = data.get("log_noise_floor_ls", data.get("log_noise_floor_bpl"))
            sigma_ls = data.get("sigma_ls")
            tau_ls = data.get("tau_ls")
            if all(np.isfinite(val) for val in (log_sigma_ls, log_tau_ls_obs, alpha_high_ls)):
                psd_ls_fit = _bending_power_law_psd_plot(
                    freqs,
                    log_sigma_ls,
                    log_tau_ls_obs,
                    alpha_high=alpha_high_ls,
                    log_noise_floor=(log_noise_floor_ls if np.isfinite(log_noise_floor_ls) else -99.0),
                )
                ax_psd.plot(
                    freqs,
                    psd_ls_fit,
                    lw=1.8,
                    color='tab:blue',
                    alpha=0.95,
                    linestyle='--',
                    label="LS broken-PL fit (raw, 2500A-calibrated)",
                    zorder=4.5,
                )
                if np.isfinite(sigma_ls) and np.isfinite(tau_ls):
                    ax_psd.text(
                        0.98,
                        0.98,
                        (
                            "LS fit\n"
                            f"$\\tau_{{\\rm ls,rf}}$={tau_ls:.1f} d\n"
                            f"$\\sigma_{{\\rm ls}}$={sigma_ls:.3g}\n"
                            f"$\\alpha_{{\\rm hi}}$={alpha_high_ls:.2f}"
                        ),
                        transform=ax_psd.transAxes,
                        ha='right',
                        va='top',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.8', alpha=0.9),
                    )
            log_sigma_bpl = data.get("log_sigma_bpl_ref_band")
            log_tau_bpl_obs = data.get("log_tau_bpl_ref_band")
            alpha_high_bpl = data.get("psd_bpl_alpha_high")
            log_noise_floor_bpl = data.get("log_noise_floor_bpl")
            psd_noise_floor_bpl = data.get("psd_noise_floor")
            if all(np.isfinite(val) for val in (log_sigma_bpl, log_tau_bpl_obs, alpha_high_bpl)):
                psd_bpl_fit = _bending_power_law_psd_plot(
                    freqs,
                    log_sigma_bpl,
                    log_tau_bpl_obs,
                    alpha_high=alpha_high_bpl,
                    log_noise_floor=(log_noise_floor_bpl if np.isfinite(log_noise_floor_bpl) else -99.0),
                )
                ax_psd.plot(
                    freqs,
                    psd_bpl_fit,
                    lw=1.4,
                    color='tab:cyan',
                    alpha=0.85,
                    linestyle=':',
                    label="LS broken-PL fit (GP-normalized)",
                    zorder=4.4,
                )
            if np.isfinite(psd_noise_floor_bpl):
                ax_psd.axhline(
                    psd_noise_floor_bpl,
                    color='gray',
                    linestyle='solid',
                    lw=3,
                    label="Noise floor",
                    zorder=-10,
                )

        tau    = jnp.exp(posterior_median['log_tau_uv'])
        tau_lo = jnp.exp(jnp.percentile(tau_samples_for_psd, 16))
        tau_hi = jnp.exp(jnp.percentile(tau_samples_for_psd, 84))

        nu    = 1.0 / (2 * np.pi * float(tau))
        nu_lo = 1.0 / (2 * np.pi * float(tau_hi))
        nu_hi = 1.0 / (2 * np.pi * float(tau_lo))

        xerr = np.array([
            [nu - nu_lo],
            [nu_hi - nu]
        ])

        print("Plotting vertical line at nu =", nu, "corresponding to tau =", tau)
        ax_psd.errorbar(
            nu, 2e4,
            xerr=xerr,
            yerr=None,
            fmt='o',
            color='m',
            markersize=5,
            capsize=4,
            elinewidth=2,
            alpha=0.8,
            label=r"$1/(2\,\pi\,\tau_{\mathrm{UV}})$",
            zorder=6,
        )

        ax_psd.set_xlabel("Frequency (days$^{-1}$)")
        ax_psd.set_ylabel(r"PSD ($\mathrm{mag}^2$ $\mathrm{days}$)")
        ax_psd.set_xscale("log")
        ax_psd.set_yscale("log")
        ax_psd.grid(False)
        ax_psd.legend(loc='lower left')
        ax_psd.set_ylim(2e-2, 9e4)
        ax_psd.set_xlim(2e-6, 1.5e-2)

    plt.tight_layout()

    # Save the plot as a PNG file
    output_dir = f"plots/multiband/{prefix}/light_curves_fits"
    os.makedirs(output_dir, exist_ok=True)
    save_suffix = suffix if filename_suffix is None else filename_suffix
    fpath = os.path.join(output_dir, f'{data["z"]:.1f}_{object_id}_light_curve_{save_suffix}.pdf')
    plt.savefig(fpath, dpi=600)

    logging.info(f"Saving figure to {fpath}")
    if show:
        plt.show()
    plt.close(fig)
    

def plot_mcmc_traces(samples_dict, data):
    """
    Generalized MCMC trace plotter for any set of parameters.

    Parameters:
    - samples_dict: dict with keys as parameter names and values as arrays of shape (n_samples, ...)
    - data: dict, must contain 'object_id'
    """
    logging.info("Plotting MCMC Traces")

    _all_labels, labels_for_trace = _trace_plot_labels(samples_dict)
    trace_items = [(key, samples_dict[key]) for key in labels_for_trace]

    total_traces = len(trace_items)
    if total_traces == 0:
        logging.warning("No trace parameters left to plot after exclusions.")
        return

    def _trace_transform(key, values):
        arr = np.asarray(values, dtype=float)
        label = key

        if key.startswith("log_"):
            return arr / np.log(10), label

        is_lag_like = ("lag" in key)
        is_amp_like = key.startswith("amp_") or ("amp_" in key)
        if is_lag_like or is_amp_like:
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = np.log10(arr)
            return arr, f"log10_{key}"

        return arr, label

    fig, axes = plt.subplots(total_traces, 1, figsize=(12, 2.5 * total_traces), sharex=True)
    if total_traces == 1:
        axes = [axes]

    for idx, (key, values) in enumerate(trace_items):
        y, ylabel = _trace_transform(key, values)
        axes[idx].plot(y, alpha=0.85, color="black", lw=0.8)
        axes[idx].set_ylabel(ylabel)
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Sample index")
    plt.tight_layout()

    output_dir = f"plots/multiband/{prefix}/mcmc_traces/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{data['z']:.1f}_{data['object_id']}_mcmc_traces_{suffix}.png")
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    logging.info(f"Saved trace plot to {save_path}")

    """
    # Plot eta_sigma vs. log_tau trace if both are present
    if 'eta_sigma' in samples_dict and 'log_tau_uv' in samples_dict:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.scatter(samples_dict['log_tau_uv'], samples_dict['eta_sigma'], alpha=0.7, lw=0.7)
        ax2.set_xlabel('log_tau_uv')
        ax2.set_ylabel('eta_sigma')
        ax2.set_title('Trace: eta_sigma vs. log_tau_uv')
        ax2.grid(True)
        save_path2 = os.path.join(output_dir, f"{data['z']:.1f}_{data['object_id']}_eta_sigma_vs_logtau.png")
        plt.tight_layout()
        plt.savefig(save_path2, dpi=100)
        plt.close(fig2)
        print("Saved eta_sigma vs. log_tau trace plot to", save_path2)

    # Plot eta_sigma vs. log_sigma_hat_uv trace if both are present
    if 'eta_sigma' in samples_dict and 'log_sigma_hat_uv' in samples_dict:
        fig_eta_sigma, ax_eta_sigma = plt.subplots(figsize=(6, 5))
        ax_eta_sigma.scatter(samples_dict['log_sigma_hat_uv'], samples_dict['eta_sigma'], alpha=0.7, lw=0.7)
        ax_eta_sigma.set_xlabel('log_sigma_hat_uv')
        ax_eta_sigma.set_ylabel('eta_sigma')
        ax_eta_sigma.set_title('Trace: eta_sigma vs. log_sigma_hat_uv')
        ax_eta_sigma.grid(True)
        save_path_eta_sigma = os.path.join(output_dir, f"{data['z']:.1f}_{data['object_id']}_eta_sigma_vs_logsigma.png")
        plt.tight_layout()
        plt.savefig(save_path_eta_sigma, dpi=100)
        plt.close(fig_eta_sigma)
        logging.info(f"Saved eta_sigma vs. log_sigma_hat_uv trace plot to {save_path_eta_sigma}")

    # Plot log_tau_uv vs. log_sigma_hat_uv trace if both are present
    if 'log_tau_uv' in samples_dict and 'log_sigma_hat_uv' in samples_dict:
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        ax3.scatter(samples_dict['log_tau_uv'], samples_dict['log_sigma_hat_uv'], alpha=0.7, lw=0.7)
        ax3.set_xlabel('log_tau_uv')
        ax3.set_ylabel('log_sigma_hat_uv')
        ax3.set_title('Trace: log_tau_uv vs. log_sigma_hat_uv')
        ax3.grid(True)
        save_path3 = os.path.join(output_dir, f"{data['z']:.1f}_{data['object_id']}_logtau_vs_logsigma.png")
        plt.tight_layout()
        plt.savefig(save_path3, dpi=100)
        plt.close(fig3)
        logging.info(f"Saved log_tau_uv vs. log_sigma_hat_uv trace plot to {save_path3}")
    """
    
    
import os, math, logging
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Shared data prep
# ---------------------------
def _prep_matrix(
    samples_flat: dict,
    max_points: int = 40_000,
    const_ptp: float = 1e-12,
    jitter_rel: float = 1e-6,
    jitter_abs: float = 1e-8,
    log10_if_startswith: str = "log_",
):
    """
    Build an (N x D) float32 matrix for plotting, with:
      - row subsampling (shared across all columns),
      - ln->log10 conversion for names starting with `log10_if_startswith`,
      - non-finite row drop (single pass),
      - small jitter for near-constant columns to avoid singular axes.

    Returns
    -------
    X : (n, d) float32
    labels : list[str]
    const_mask : (d,) bool (columns that were near-constant before jitter)
    """
    labels = list(samples_flat.keys())
    if not labels:
        raise ValueError("samples_flat is empty")

    # shared row subsample (before stacking)
    n_total = np.asarray(samples_flat[labels[0]]).ravel().shape[0]
    idx = None
    if n_total > max_points:
        idx = np.random.default_rng().choice(n_total, size=max_points, replace=False)

    cols = []
    kept_labels = []
    dropped_cols = []
    for k in labels:
        a = np.asarray(samples_flat[k]).ravel()
        if idx is not None:
            a = a[idx]
        if log10_if_startswith and k.startswith(log10_if_startswith):
            a = a / np.log(10.0)  # ln -> log10
        finite_frac = float(np.mean(np.isfinite(a))) if a.size else 0.0
        if finite_frac < 1.0:
            dropped_cols.append((k, finite_frac))
            continue
        cols.append(a.astype(np.float32, copy=False))
        kept_labels.append(k)

    if dropped_cols:
        preview = ", ".join(f"{name}({frac:.1%} finite)" for name, frac in dropped_cols[:12])
        logging.warning(
            "Dropping %d non-finite parameter column(s) from plotting matrix: %s%s",
            len(dropped_cols),
            preview,
            " ..." if len(dropped_cols) > 12 else "",
        )

    if not cols:
        detail = ", ".join(f"{name}({frac:.1%})" for name, frac in dropped_cols[:20])
        raise ValueError(f"No fully finite sample columns available for plotting. Offenders: {detail}")

    X = np.column_stack(cols)
    labels = kept_labels

    # drop rows with any NaN/Inf
    finite = np.all(np.isfinite(X), axis=1)
    X = X[finite]
    if X.shape[0] == 0:
        raise ValueError(
            f"No finite samples to plot after cleaning for columns: {labels}"
        )

    # jitter near-constant columns (keep them visible)
    ptp = np.ptp(X, axis=0)
    const_mask = ptp <= const_ptp
    if np.any(const_mask):
        for j in np.where(const_mask)[0]:
            col = X[:, j]
            mu = float(np.mean(col))
            sigma = max(jitter_abs, abs(mu) * jitter_rel)
            X[:, j] = (col + np.random.normal(0.0, sigma, size=col.shape)).astype(np.float32)
            #print("Jittered near-constant param:", labels[j])

    return X, labels, const_mask


# ---------------------------
# 1) All 1D histograms
# ---------------------------
def plot_all_histograms(
    samples_flat: dict,
    data: dict,
    bins: int = 24,
    p_lo: float = 0.5,
    p_hi: float = 99.5,
    max_points: int = 40_000,
    base_cols: int = 6,
    dpi: int = 140,
):
    """
    Plot 1D marginals (histograms) for ALL parameters, rescaled to nice units.

    - Shared subsample/clean/jitter via _prep_matrix (assumed available)
    - Per-parameter autoscale to powers of 10^3 for readable axes
    - Percentile-based x-lims (p_lo/p_hi)
    - Median (50th) line + 16/84% band, with a numeric annotation

    Returns
    -------
    fig, axes, save_path
    """

    # ---- helper: choose a unit scale (×10^{k}, k multiple of 3) based on IQR ----
    def _autoscale_unit(x: np.ndarray):
        # robust spread
        q25, q75 = np.percentile(x, [25, 75])
        iqr = max(1e-30, float(q75 - q25))
        # exponent of the IQR
        exp10 = int(np.floor(np.log10(iqr)))
        # snap to steps of 3 (…,-9,-6,-3,0,3,6,9,…)
        exp3 = 3 * int(np.floor(exp10 / 3))
        # scale to apply to data for plotting (x_scaled = x * 10^{-exp3})
        scale = 10.0 ** (-exp3)
        return scale, exp3  # (multiply-by, original exponent)

    # Prep matrix (rows=subsamples; cols=parameters)
    X, labels, _ = _prep_matrix(samples_flat, max_points=max_points)
    d = X.shape[1]
    n_cols = min(base_cols, d) if d > 0 else 1
    n_rows = math.ceil(d / n_cols)

    # Figure sizing
    fig_w = max(12, 2.2 * n_cols)
    fig_h = max(4.0, 1.6 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=dpi, squeeze=False)

    # robust ranges on ORIGINAL scale (we'll rescale them per panel)
    lo = np.percentile(X, p_lo, axis=0)
    hi = np.percentile(X, p_hi, axis=0)
    eps = 1e-12

    # draw
    panel = 0
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            if panel >= d:
                ax.axis("off")
                continue

            x = X[:, panel]
            l, h = float(lo[panel]), float(hi[panel])
            if not (np.isfinite(l) and np.isfinite(h)):
                l, h = np.min(x), np.max(x)
            if h <= l + eps:
                center = l
                pad = max(1e-12, abs(center) * 1e-6, 1e-8)
                l, h = center - pad, center + pad

            # --- autoscale to “nice” units ---
            scale, exp3 = _autoscale_unit(x)
            x_s   = x * scale
            l_s   = l * scale
            h_s   = h * scale
            x_span = float(np.nanmax(x_s) - np.nanmin(x_s)) if np.all(np.isfinite(x_s)) else np.inf
            const_tol = max(1e-12, abs(float(np.nanmedian(x_s))) * 1e-12) if np.any(np.isfinite(x_s)) else 1e-12
            hist_bins = 1 if (not np.isfinite(x_span) or x_span <= const_tol or not np.isfinite(l_s) or not np.isfinite(h_s) or h_s <= l_s) else bins

            # density=True is fine after rescaling; values stay reasonable
            ax.hist(x_s, bins=hist_bins, range=(l_s, h_s), density=True, edgecolor="none")

            # median + 68% interval (in scaled units)
            q16, q50, q84 = np.percentile(x_s, [16, 50, 84])
            err = 0.5 * (q84 - q16)
            ax.axvline(q50, linestyle="--", linewidth=1)
            ax.axvspan(q16, q84, alpha=0.12)

            # formatted annotation: median ± error
            txt = f"{q50:.3g} ± {err:.2g}"
            ax.text(
                0.98, 0.95, txt,
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, bbox=dict(facecolor="white", alpha=0.6, linewidth=0)
            )

            # title with unit scale suffix
            unit_txt = f" ×10^{exp3}" if exp3 != 0 else ""
            ax.set_title(f"{labels[panel]}{unit_txt}", fontsize=9, pad=2)

            # no scientific offset text on ticks
            ax.ticklabel_format(axis="both", style="plain", useOffset=False)
            for axis in (ax.xaxis, ax.yaxis):
                fmt = ScalarFormatter(useMathText=True)
                fmt.set_scientific(False)
                fmt.set_useOffset(False)
                axis.set_major_formatter(fmt)

            ax.tick_params(axis="both", labelsize=8)
            ax.margins(x=0)

            panel += 1

    fig.tight_layout()

    # Save
    object_id = data["object_id"]
    z = data["z"]
    out_dir = f"plots/multiband/{prefix}/marginals"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{z:.1f}_{object_id}_marginals_all_{suffix}.png")
    fig.savefig(save_path, dpi=dpi)  # no tight bbox (faster)
    logging.info(f"Saved ALL histograms to {save_path}")
    plt.close(fig)
    return fig, axes, save_path


# ---------------------------
# 2) Full correlation matrix
# ---------------------------
def plot_correlation_matrix(
    samples_flat: dict,
    data: dict,
    max_points: int = 40_000,
    reorder: str = "spectral",  # 'none' | 'spectral'
    heatmap_tick_cap: int = 60,
    dpi: int = 140,
    cmap: str = "coolwarm",
):
    """
    Plot the full correlation matrix of ALL parameters.

    - Row subsample, clean, jitter constants (shared helper).
    - Optional **spectral reordering** (sort by leading eigenvector of |corr|),
      which tends to cluster correlated blocks for readability without SciPy.
    - Tick labels are sparsified to avoid clutter on large-D problems.

    Returns: (fig, ax, save_path)
    """
    X, labels, _ = _prep_matrix(samples_flat, max_points=max_points)

    # z-score for numerics; compute corr
    std = X.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    Xs = (X - X.mean(axis=0)) / std
    C = np.corrcoef(Xs, rowvar=False)

    # Optional spectral reordering (cheap, helps show blocks)
    order = np.arange(C.shape[0])
    if reorder == "spectral" and C.shape[0] > 2:
        # use |corr| to emphasize structure, then top eigenvector of Laplacian
        A = np.abs(C)
        np.fill_diagonal(A, 0.0)
        d = A.sum(axis=1)
        L = np.diag(d) - A
        # smallest non-zero eigenvector (Fiedler) by eigh (symmetric)
        w, v = np.linalg.eigh(L)
        # choose the 2nd smallest eigenvector if possible
        if len(w) >= 2:
            fiedler = v[:, 1]
        else:
            fiedler = v[:, 0]
        order = np.argsort(fiedler)
        C = C[order][:, order]
        labels = [labels[i] for i in order]

    # Figure
    d = C.shape[0]
    fig_w = max(10, min(22, 0.18 * d + 6))  # scale width with d, cap at 22"
    fig_h = max(4.5, min(12, 0.12 * d + 3)) # scale height, cap at 12"
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    im = ax.imshow(C, vmin=-1, vmax=1, interpolation="nearest", aspect="auto", cmap=cmap)
    ax.set_title("Correlation matrix", fontsize=12, pad=6)

    # sparsify tick labels for readability & speed
    step = max(1, d // heatmap_tick_cap)
    ticks = np.arange(0, d, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], rotation=90, fontsize=8)
    ax.set_yticklabels([labels[i] for i in ticks], fontsize=8)

    # light gridlines (optional)
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.set_ylabel("ρ", rotation=0, labelpad=10)

    fig.tight_layout()

    # Save
    object_id = data["object_id"]
    z = data["z"]
    out_dir = f"plots/multiband/{prefix}/correlations/"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{z:.1f}_{object_id}_corr_all_{suffix}.png")
    fig.savefig(save_path, dpi=dpi)
    logging.info(f"Saved correlation matrix to {save_path}")
    plt.close(fig)

def plot_recovery(results):
    # Collect all fields from each result into arrays for plotting
    log_sigma_fake = np.array([r['log_sigma_fake'] for r in results])
    log_sigma_uv = np.array([r['log_sigma_uv'] for r in results])
    log_sigma_uv_err = np.array([r['log_sigma_uv_err'] for r in results])
    log_tau_fake = np.array([r['log_tau_fake'] for r in results])
    log_tau_uv = np.array([r['log_tau_uv'] for r in results])
    log_tau_uv_err = np.array([r['log_tau_uv_err'] for r in results])
    t_obs_length = np.array([r['t_obs_length'] for r in results])
    t_rf_length = np.array([r['t_rf_length'] for r in results])

    rho = 10**log_tau_uv / t_rf_length

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: log_sigma_fake vs log_sigma_uv, color by rho
    x1 = log_sigma_fake
    y1 = log_sigma_uv
    y1err = log_sigma_uv_err
    c1 = rho
    xmin1, xmax1 = -2, 0.5
    sc1 = axes[0].scatter(x1, y1, c=c1, cmap='plasma', s=40, edgecolor='k', alpha=0.8)
    axes[0].errorbar(x1, y1, yerr=y1err, fmt='none', ecolor='gray', alpha=0.4, elinewidth=1, capsize=2)
    axes[0].plot([xmin1, xmax1], [xmin1, xmax1], 'm--', lw=2)
    axes[0].set_xlabel(r'$\log_{10}\,\sigma_\mathrm{fake}$')
    axes[0].set_ylabel(r'$\log_{10}\,\sigma_{\mathrm{UV}}$')
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlim(xmin1, xmax1)
    axes[0].set_ylim(xmin1, xmax1)

    # Panel 2: log_tau_fake vs log_tau_uv (swapped x and y), color by rho
    x2 = log_tau_fake
    y2 = log_tau_uv
    y2err = log_tau_uv_err
    c2 = rho
    xmin2, xmax2 = 0, 6
    sc2 = axes[1].scatter(x2, y2, c=c2, cmap='plasma', s=40, edgecolor='k', alpha=0.8)
    axes[1].errorbar(x2, y2, yerr=y2err, fmt='none', ecolor='gray', alpha=0.4, elinewidth=1, capsize=2)
    axes[1].plot([xmin2, xmax2], [xmin2, xmax2], 'm--', lw=2)
    axes[1].set_xlabel(r'$\log_{10}\,\tau_\mathrm{fake}$')
    axes[1].set_ylabel(r'$\log_{10}\,\tau_\mathrm{UV}$')
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_xlim(xmin2, xmax2)
    axes[1].set_ylim(xmin2, xmax2)


    # Add colorbar for the second panel (rho)
    cax = fig.add_axes([axes[1].get_position().x1 + 0.01, axes[1].get_position().y0, 0.015, axes[1].get_position().height])
    cbar2 = plt.colorbar(sc2, cax=cax)
    cbar2.set_label(r'$\rho = \tau_\mathrm{DRW,0} / t_\mathrm{RF}$')

    # Save
    out_dir = f"plots/multiband/{prefix}/sigmatau/"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{suffix}.png")
    fig.savefig(save_path, dpi=300)
    print(f"Saved injected vs recovery sigma and tau plot to {save_path}")
    plt.close(fig)

def plot_sigma_tau_vs_lambda_with_model(
    rows,                 # list[dict], each dict = one object
    bands=('u', 'g', 'r', 'i', 'z'),
    *,
    inject_fake=False,
    residual=True,        # subtract UV from BOTH σ and τ
    show=False,
    debug=True,
):
    """
    Plot log10 σ_band and log10 τ_band,RF vs log10 λ_RF with population ribbons.

    This version is robust to missing bands/fields on a per-object basis:
    - If a row lacks a given band or parameter, it is treated as NaN and ignored.
    - The model ribbon uses medians over available rows/fields.

    Requires external: lambda_pivot (dict band->Å), colors (dict band->color),
                       log_single_pl, prefix, suffix.
    """
    if not rows:
        raise ValueError("`rows` is empty.")

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import linregress

    # ---------- helpers ----------
    def getf(row, key, default=np.nan):
        """Fetch row[key] if present and finite-castable; else default."""
        v = row.get(key, default)
        try:
            return float(v)
        except Exception:
            return default

    def arr(key):
        """Array over rows; missing keys -> NaN."""
        return np.asarray([getf(r, key) for r in rows], dtype=float)

    def arr_band(prefix_key, b):
        """Array over rows for per-band fields; missing -> NaN."""
        key = f"{prefix_key}_{b}"
        return np.asarray([getf(r, key) for r in rows], dtype=float)

    def med(key, default=np.nan):
        a = arr(key)
        if a.size == 0 or not np.any(np.isfinite(a)):
            return float(default)
        return float(np.nanmedian(a))

    def med_err(key, default=np.nan):
        # Median of per-row 1σ uncertainties (missing -> NaN)
        a = arr(key)
        if a.size == 0 or not np.any(np.isfinite(a)):
            return float(default)
        return float(np.nanmedian(a))

    # UV refs (may be missing per-row)
    z      = arr('z')
    tau_uv = arr('log_tau_uv_rf') if residual else np.zeros(len(rows))
    sig_uv = arr('log_sigma_uv')  if residual else np.zeros(len(rows))

    # ---------- figure ----------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 6.2),
                                   sharex=True, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.02)

    def inward(ax):
        ax.tick_params(direction='in', which='both', top=True, right=True, length=3, pad=2)
        for s in ax.spines.values():
            s.set_linewidth(1.0)

    # ---------- scatter (robust per-band/per-row) ----------
    x_all, x_all_tau, y_all_tau = [], [], []
    plotted_bands = []

    for b in bands:
        # If pivot lambda for this band is unknown, skip the band entirely
        if b not in lambda_pivot:
            continue

        lam_rf = lambda_pivot[b] / (1.0 + z)  # Å; z may contain NaNs -> handled below
        x = np.log10(lam_rf)

        y_sigma_raw = arr_band('log_sigma_band', b)          # log σ_band
        y_tau_abs   = arr_band('log_tau_band', b + '_RF')    # log τ_band,RF

        # residualization (handles NaNs naturally)
        y_sigma = y_sigma_raw - sig_uv if residual else y_sigma_raw
        y_tau   = y_tau_abs   - tau_uv if residual else y_tau_abs

        m1 = np.isfinite(x) & np.isfinite(y_sigma)
        m2 = np.isfinite(x) & np.isfinite(y_tau)

        if m1.any() or m2.any():
            plotted_bands.append(b)

        if m1.any():
            ax1.scatter(x[m1], y_sigma[m1], s=40, alpha=0.8, color=colors.get(b, None),
                        edgecolor='none', zorder=7)
            x_all.append(x[m1])

        if m2.any():
            ax2.scatter(x[m2], y_tau[m2], s=40, alpha=0.8, color=colors.get(b, None),
                        edgecolor='none', zorder=7)
            x_all_tau.append(x[m2])
            y_all_tau.append(y_tau[m2])

    x_all     = np.concatenate(x_all)     if x_all     else np.array([])
    x_all_tau = np.concatenate(x_all_tau) if x_all_tau else np.array([])
    y_all_tau = np.concatenate(y_all_tau) if y_all_tau else np.array([])

    if x_all.size:
        xmin, xmax = float(np.nanmin(x_all)) - 0.1, float(np.nanmax(x_all)) + 0.1
        ax2.set_xlim(xmin, xmax)
    else:
        # fallback window (Å: ~2e3–1.6e4)
        xmin, xmax = 3.3, 4.2
        ax2.set_xlim(xmin, xmax)

    # ---------- model ribbons from η ± 1σ (no MC) ----------
    lam_grid    = np.linspace(10**xmin, 10**xmax, 400).astype(float)
    loglam_grid = np.log10(lam_grid)

    # Population medians (robust to missing values)
    lam_s_med = 2500.0
    eta_sigma_med = med('eta_sigma')
    eta_tau_med = med('eta_tau')

    # Median 1σ (per-object errors summarized by median)
    sig_eta_sigma = med_err('eta_sigma_err')
    sig_eta_t1 = med_err('eta_tau_err')

    # Intercepts (already log10). For τ, convert to RF then median; residualize if requested.
    sig0_all = arr('log_sigma_uv') - (sig_uv if residual else 0.0)
    tau0_rf_all = arr('log_tau_uv') - np.log10(1.0 + z)
    if residual:
        tau0_rf_all = tau0_rf_all - tau_uv

    sig0_med = float(np.nanmedian(sig0_all))
    tau0_med = float(np.nanmedian(tau0_rf_all))

    def shp(e1):
        return log_single_pl(lam_grid, lam_s_med, e1)

    # Central curves
    center_sigma = sig0_med + shp(eta_sigma_med)
    center_tau   = tau0_med + shp(eta_tau_med)

    # Four-corner envelopes (η1±σ1, η2±σ2); handle NaN sigmas gracefully
    def _nan_safe(v, dv):
        lo = v - dv if np.isfinite(dv) else v
        hi = v + dv if np.isfinite(dv) else v
        return lo, hi

    A1_lo, A1_hi = _nan_safe(eta_sigma_med, sig_eta_sigma)
    T1_lo, T1_hi = _nan_safe(eta_tau_med, sig_eta_t1)

    sigma_corners = np.vstack([
        sig0_med + shp(A1_lo),
        sig0_med + shp(A1_hi),
    ])
    tau_corners = np.vstack([
        tau0_med + shp(T1_lo),
        tau0_med + shp(T1_hi),
    ])

    sigma_lo = np.nanmin(sigma_corners, axis=0)
    sigma_hi = np.nanmax(sigma_corners, axis=0)
    tau_lo   = np.nanmin(tau_corners,   axis=0)
    tau_hi   = np.nanmax(tau_corners,   axis=0)

    # Plot population-median curve + η-error ribbon
    ax1.plot(loglam_grid, center_sigma, lw=1.6, color='m', zorder=3)
    ax1.fill_between(loglam_grid, sigma_lo, sigma_hi, color='m', alpha=0.28, zorder=2)
    ax2.plot(loglam_grid, center_tau,   lw=1.6, color='m', zorder=3)
    ax2.fill_between(loglam_grid, tau_lo,   tau_hi,   color='m', alpha=0.28, zorder=2)

    # ---------- injected ("fake") single-slope overlays (optional) ----------
    have_fake_fields = all(k in rows[0] for k in ('alpha_sigma', 'beta_tau'))
    alpha_sigma_med = beta_tau_med = None

    if inject_fake and have_fake_fields:
        alpha_sigma_med = med('alpha_sigma')
        beta_tau_med    = med('beta_tau')

        fake_sigma_curve = sig0_med + shp(alpha_sigma_med)
        fake_tau_curve   = tau0_med + shp(beta_tau_med)

        ax1.plot(loglam_grid, fake_sigma_curve, ls='--', lw=1.2, color='0.25',
                 zorder=4, label='Injected σ-slope')
        ax2.plot(loglam_grid, fake_tau_curve,   ls='--', lw=1.2, color='0.25',
                 zorder=4, label='Injected τ-slope')

    # ---------- labels & axes ----------
    if residual:
        sig_lab = r'$\log(\sigma_{\mathrm{band}}/\sigma_{\mathrm{UV}})$'
        tau_lab = r'$\log(\tau_{\mathrm{band,RF}}/\tau_{\mathrm{UV,RF}})$'
    else:
        sig_lab = r'$\log\!\,\sigma_{\mathrm{band}}$'
        tau_lab = r'$\log\!\,\tau_{\mathrm{band,RF}}$'

    ax1.set_ylabel(sig_lab)
    ax2.set_ylabel(tau_lab)
    ax2.set_xlabel(r'$\log(\lambda_{\mathrm{RF}}/\mathrm{\AA})$')

    inward(ax1); inward(ax2)

    # top linear-Å axis
    def loglam_to_A(x): return np.power(10.0, x)
    def A_to_loglam(x): return np.log10(np.clip(x, 1e-12, None))
    secax = ax1.secondary_xaxis('top', functions=(loglam_to_A, A_to_loglam))
    secax.set_xlabel(r'$\lambda_{\mathrm{RF}}\;(\mathrm{\AA})$')
    secax.tick_params(direction='in', which='both', top=True)

    lam_min, lam_max = 10**xmin, 10**xmax
    rng = lam_max - lam_min
    steps = np.array([100, 200, 500, 1000, 2000, 4000], dtype=float)
    step = steps[np.argmin(np.abs(rng / steps - 5.0))]
    ticks = np.arange(np.floor(lam_min / step) * step,
                      np.ceil(lam_max / step) * step + step, step)
    secax.set_xticks(np.union1d(ticks, [1000.0]))

    # ---------- legend (only bands that actually plotted) ----------
    band_handles = [
        Line2D([0], [0], linestyle='none', marker='o', markersize=6,
               markerfacecolor=colors.get(b, None), markeredgecolor='none',
               label=f'Band {b}')
        for b in plotted_bands
    ]
    model_handles = [Line2D([0], [0], color='m', lw=1.6, label='Median model (η ± 1σ)')]
    inj_handles = []
    if inject_fake and have_fake_fields:
        inj_handles.append(Line2D([0], [0], color='0.25', lw=1.2, ls='--',
                                  label='Injected slope'))

    handles = band_handles + model_handles + inj_handles
    if handles:
        ax1.legend(handles=handles, loc='best', frameon=False, ncol=2, fontsize=9)

    # ---------- annotations (median params; robust to NaNs) ----------
    txt_sigma = rf'$\eta_{{\sigma}} = {eta_sigma_med:+.3f}\,\pm\,{sig_eta_sigma:.3f}$'
    if inject_fake and have_fake_fields and np.isfinite(alpha_sigma_med):
        d1 = eta_sigma_med - alpha_sigma_med
        txt_sigma += ('\n' +
                      rf'$\alpha_\sigma^\mathrm{{(inj)}} = {alpha_sigma_med:+.3f}$' '\n' +
                      rf'$\Delta\eta_{{\sigma}} = {d1:+.3f}$')
    ax1.text(0.02, 0.96, txt_sigma, transform=ax1.transAxes, va='top', ha='left', alpha=1.0,
             fontsize=10, bbox=dict(boxstyle='round,pad=0.25', fc='white', lw=0.8), zorder=10)

    txt_tau = rf'$\eta_{{\tau}} = {eta_tau_med:+.3f}\,\pm\,{sig_eta_t1:.3f}$'
    if inject_fake and have_fake_fields and np.isfinite(beta_tau_med):
        d1t = eta_tau_med - beta_tau_med
        txt_tau += ('\n' +
                    rf'$\beta_\tau^\mathrm{{(inj)}} = {beta_tau_med:+.3f}$' '\n' +
                    rf'$\Delta\eta_{{\tau}} = {d1t:+.3f}$')
    ax2.text(0.02, 0.96, txt_tau, transform=ax2.transAxes, va='top', ha='left', alpha=1.0,
             fontsize=10, bbox=dict(boxstyle='round,pad=0.25', fc='white', lw=0.8), zorder=10)

    # ---------- diagnostics ----------
    if debug and x_all_tau.size:
        slope_pts = linregress(x_all_tau, y_all_tau).slope
        slope_model = np.gradient(center_tau, loglam_grid).mean()
        print(f"[diag] slope(points) d logτ / d logλ ≈ {slope_pts:+.3f}")
        print(f"[diag] slope(model ) d logτ / d logλ ≈ {slope_model:+.3f}")
        print(f"[diag] medians: ησ={eta_sigma_med:+.3f}±{sig_eta_sigma:.3f}, "
              f"ητ={eta_tau_med:+.3f}±{sig_eta_t1:.3f}")

    # ---------- save ----------
    out_dir = f"plots/multiband/{prefix}/powerlaw/"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{suffix}.png")
    fig.savefig(save_path, dpi=300)
    print(f"Saved power law plot to {save_path}")

    if show:
        plt.show()
    plt.close()
