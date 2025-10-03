import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D

plt.style.use("style.mplstyle")
import corner
import numpy as np
import os
import re
import math
import jax.numpy as jnp

from astropy.timeseries import LombScargle

prefix = os.environ.get('PREFIX', "test")
suffix = os.environ.get('SUFFIX', "test")

from multiband_fit_utils import log_broken_pl, log_single_pl

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

def plot_posterior(samples_flat, data, bins=20):
    """
    Generalized corner plot of posterior parameters

    Parameters
    ----------
    samples_flat : dict
        Dict of MCMC samples, shape (n_samples, ...) for each param.
    data : dict
        Object metadata (must contain 'object_id' and 'z').
    bins : int
        Number of bins for corner plot.
    """
    logging.info("Saving posterior plot")
    object_id = data['object_id']
    z = data['z']
    flat_labels = list(samples_flat.keys())
    flat_arrays = [np.asarray(samples_flat[k]).flatten() for k in flat_labels]

    corner_data = np.vstack(flat_arrays).T

    for i in range(corner_data.shape[1]):
        lo, hi = corner_data[:, i].min(), corner_data[:, i].max()
        if lo == hi:  # constant parameter
            #print("Corner Constant param: ", flat_labels[i])
            corner_data[:, i] += np.random.normal(0, 1e-6, size=corner_data.shape[0])
        if 'log_' in flat_labels[i]:
            corner_data[:, i] = corner_data[:, i] / np.log(10)

    fig = corner.corner(corner_data, labels=flat_labels, show_titles=True, 
                        quantiles=[0.16, 0.5, 0.84], bins=bins, plot_datapoints=False, plot_contours=False)

    # Save plot
    output_dir = f"plots/multiband/{prefix}/posterior/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{z:.1f}_{object_id}_posterior_{suffix}.pdf")
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    logging.info(f"Saved posterior corner plot to {save_path}")

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

def plot_posterior_fast(
    samples_flat,
    data,
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
    Faster corner plot for large MCMC draws.

    Speed tactics:
      - Subsample to meet a compute budget ~ O(N * D^2)
      - Trim rows outside the central [p_lo, p_hi] percentile box across all dims
      - Keep "constant" params by applying tiny jitter (so titles render)
      - Avoid redundant copies; use float32
    """
    import os, logging
    import numpy as np
    import matplotlib.pyplot as plt
    import corner

    logging.info("Saving posterior plot (fast path)")
    object_id = data["object_id"]
    z = data["z"]

    # Stable column order
    labels = np.array(list(samples_flat.keys()))
    D = len(labels)

    # Determine a target N that respects both `max_points` and the D^2 cost
    # (ensure at least a modest minimum so 1D histograms look sane)
    rng = np.random.default_rng(rng_seed)
    first = np.asarray(samples_flat[labels[0]])
    first = first.reshape(first.shape[0], -1)[:, 0]
    n_total = first.shape[0]
    n_by_budget = max(2_000, int(panel_budget // max(D * D, 1)))
    n_target = min(n_total, max_points, n_by_budget)

    if n_target < n_total:
        idx = rng.choice(n_total, size=n_target, replace=False)
        n_use = n_target
    else:
        idx = None
        n_use = n_total

    # Build matrix X (N, D) with shared subsample; convert ln(*) -> log10(*) for "log_*"
    X = np.empty((n_use, D), dtype=np.float32)
    ln10 = np.log(10.0)
    for j, k in enumerate(labels):
        a = np.asarray(samples_flat[k]).reshape(n_total, -1)[:, 0]
        if idx is not None:
            a = a[idx]
        if k.startswith("log_"):
            a = a / ln10  # natural log -> log10
        X[:, j] = a.astype(np.float32, copy=False)

    # Drop rows with any NaN/Inf
    finite = np.isfinite(X).all(axis=1)
    if not finite.any():
        raise ValueError("No finite samples to plot after cleaning.")
    if not finite.all():
        X = X[finite]

    # Identify near-constant columns via ptp
    ptp = np.ptp(X, axis=0)
    const_mask = ptp <= const_ptp

    # Jitter constant columns so they render
    if np.any(const_mask):
        const_idx = np.where(const_mask)[0]
        mu = X[:, const_idx].mean(axis=0, dtype=np.float64)
        sig = np.maximum(jitter_abs, np.abs(mu) * jitter_rel).astype(np.float32)
        noise = rng.normal(0.0, 1.0, size=(X.shape[0], const_idx.size)).astype(np.float32)
        X[:, const_idx] += noise * sig
        for j in const_idx:
            logging.debug(f"Corner constant param (jittered): {labels[j]}")

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

    # Corner tuned for speed
    fig = corner.corner(
        X_plot,
        labels=labels_plot,
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
        bins=int(bins),
        range=[tuple(r) for r in ranges_plot],
        plot_datapoints=False,
        plot_contours=False,        # no KDE
        hist2d_kwargs={"bins": int(bins)},
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
    plt.close(fig)
    logging.info(f"Saved posterior corner plot to {save_path}")



def plot_broken_power_law(samples, data, broken_pl):
    """
    Plot two stacked panels of the smooth broken power law using posterior medians.
      Top:    (eta_A1, eta_A2)
      Bottom: (eta_tau1, eta_tau2)
    Both share x = log10(lambda) and show a linear-lambda axis on top.

    Parameters
    ----------
    samples : dict
        Posterior samples with keys:
        eta_A1, eta_A2, eta_tau1, eta_tau2, eta_break, lam_s
    data : unused (placeholder for future use)
    """
    log_pl = log_broken_pl if broken_pl else log_single_pl
    # --- posterior medians ---
    pm = {k: np.median(np.asarray(samples[k])) for k in
          ["eta_A1","eta_A2","eta_tau1","eta_tau2","eta_break","lam_s"]}
    eta_A1, eta_A2   = pm["eta_A1"], pm["eta_A2"]
    eta_tau1, eta_tau2 = pm["eta_tau1"], pm["eta_tau2"]
    eta_break, lam_s = pm["eta_break"], pm["lam_s"]

    # --- wavelength grid ---
    xlog = np.linspace(2.9, 3.9, 600)
    lam = 10.0**xlog
    y_amp = log_pl(lam, lam_s, eta_A1, eta_A2, eta_break)
    y_tau = log_pl(lam, lam_s, eta_tau1, eta_tau2, eta_break)

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
             label=fr'$\eta_A=({eta_A1:.2f},{eta_A2:.2f}),\ s={eta_break:.2f}$')
    prettify(ax1)
    ax1.set_ylabel(r'$\log_{10}\,f_A(\lambda)$')
    ax1.legend(frameon=False, loc="best")

    # --- bottom panel ---
    ax2.plot(xlog, y_tau, lw=2.0,
             label=fr'$\eta_\tau=({eta_tau1:.2f},{eta_tau2:.2f}),\ s={eta_break:.2f}$')
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
    params: dict,
    omega: np.ndarray,
    *,
    bins_per_decade: int = 2,
    min_per_bin: int = 1,
    normalization: str = "psd",
):
    """
    Compute Lomb–Scargle PSD from a MyMultiVarModel, using a provided
    angular frequency grid (omega, in rad / time-unit).
    
    Steps:
      - lag-subtract (my_lag_transform)
      - mean-subtract (mean_func)
      - normalize amplitudes to band 0 scale (my_amp_transform)
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
    (t_lag, band_idx), _ = model.my_lag_transform(model.X, model.has_lag, params)
    t_lag = np.asarray(t_lag, float)
    band_idx = np.asarray(band_idx, int)

    # Mean subtraction via mean_func
    t_center = float(np.mean(t_lag))
    t_std = float(np.std(t_lag))
    mean_vals = model.mean_func(
        model.zero_mean,
        int(np.max(band_idx)) + 1,
        t_center,
        t_std,
        params,
        (t_lag, band_idx),
    )
    y = np.asarray(model.y, float).copy() - np.asarray(mean_vals, float)
    yerr = np.asarray(model.yerr, float).copy()

    # Normalize amplitudes to band 0 scale
    log_sigma_band = np.asarray(model.my_amp_transform(params))
    s0 = float(np.exp(log_sigma_band[0]))
    s_b = np.exp(log_sigma_band)
    scale = s0 / s_b[band_idx]
    y *= scale

    # Sort by time (optional, not required for LS)
    order = np.argsort(t_lag)
    t_lag, y = t_lag[order], y[order]
    yerr *= scale
    yerr = yerr[order]

    # Lomb–Scargle
    ls = LombScargle(t_lag, y, yerr, fit_mean=False)
    P_raw = ls.power(f_raw, normalization=normalization)

    P_noise = np.median(P_raw[f_raw > 1/20])

    P_raw = np.maximum(P_raw - P_noise, 0.0)  # keep non-negative

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


def save_combined_plot(samples, model, X, y, yerr, band_idx, data, bands=['u', 'g', 'r', 'i', 'z'], plot_psd=True):
    logging.info("Saving combined plot")

    object_id = data['object_id']
    band_idx_map = {i: b for i, b in enumerate(bands)}

    fig, (ax_lc, ax_psd) = plt.subplots(2, 1, figsize=(10, 10), sharex=False, gridspec_kw={'height_ratios': [1.5, 1]})
    offsets = np.arange(len(bands)) * 0.25

    t = X[0]
    for n in np.unique(band_idx):
        mask = (band_idx == n) & (yerr < 10.0)
        # Plot the observed data
        ax_lc.errorbar(t[mask], y[mask]+offsets[n], yerr=yerr[mask], fmt='o', 
                label=f'{band_idx_map[n]}-band', alpha=0.7, color=colors[band_idx_map[n]], lw=1.0, capsize=1, markersize=1)
        # Generate test times for predictions
        t_test = np.linspace(t.min() - 400, t.max() + 400, 1000)
        # Compute predictions using the model
        posterior_median = {k: np.median(v, axis=0) for k, v in samples.items()}
        result = model.pred(posterior_median, (t_test, jnp.full_like(t_test, n, dtype=int)))

        # Plot the predictions
        if len(result) == 2:
            mu, std = result
            ax_lc.plot(t_test, mu+offsets[n], alpha=0.8, color=colors[band_idx_map[n]], lw=1.0)
            ax_lc.fill_between(t_test, mu+offsets[n]-std, mu+offsets[n]+std, alpha=0.3, 
                lw=0.5, color=colors[band_idx_map[n]])
        else:
            mu, std, mu_cont, std_cont, mu_blr, std_blr = result
            # Plot the continuum and BLR components if available
            ax_lc.plot(t_test, mu_cont + offsets[n], alpha=0.5
                    , color=colors[band_idx_map[n]], lw=1.0, label=f'{band_idx_map[n]}-band continuum', linestyle='--')
            ax_lc.fill_between(t_test, mu_cont + offsets[n] - std_cont,
                               mu_cont + offsets[n] + std_cont, alpha=0.15, lw=0.5, color=colors[band_idx_map[n]])
            ax_lc.plot(t_test, mu_blr + offsets[n], alpha=0.5,
                    color=colors[band_idx_map[n]], lw=1.0, label=f'{band_idx_map[n]}-band BLR', linestyle=':')
            ax_lc.fill_between(t_test, mu_blr + offsets[n] - std_blr,
                               mu_blr + offsets[n] + std_blr, alpha=0.15, lw=0.5, color=colors[band_idx_map[n]])
            # Total
            ax_lc.plot(t_test, mu + offsets[n], alpha=0.8, color=colors[band_idx_map[n]], lw=1.0)
            ax_lc.fill_between(t_test, mu + offsets[n] - std, mu + offsets[n] + std, alpha=0.3,
                               lw=0.5, color=colors[band_idx_map[n]])
    
    if plot_psd:
        # Ensure all elements of posterior_median are jnp arrays
        for k in posterior_median:
            posterior_median[k] = jnp.array(posterior_median[k])

        print("Plotting PSD...")
        # PSD calculation and plotting
        freqs = np.logspace(-6, 2, 500)

        # Data PSD
        f_bin, P_bin, P_lo, P_hi, cts, P_noise = combined_lomb_scargle_from_model(model, posterior_median, 2*np.pi*freqs)
        ax_psd.errorbar(f_bin, P_bin, yerr=[P_bin - P_lo, P_hi - P_bin], label="Lomb-Scargle PSD", lw=4, color='k')

        # Plot a vertical line at the posterior median log_tau_drw0 (if present)
        # TODO: log_tau_eff = model.my_tau_drw_transform(posterior_median)  # scalar
        tau = jnp.exp(posterior_median['log_tau_drw0']) # obs frame
        tau_lo = jnp.exp(jnp.percentile(samples['log_tau_drw0'], 16))
        tau_hi = jnp.exp(jnp.percentile(samples['log_tau_drw0'], 84))
        ax_psd.axvspan(1.0 / (2*np.pi*tau_hi), 1.0 / (2*np.pi*tau_lo), color='r', alpha=0.15)
        ax_psd.axvline(1.0 / (2*np.pi*tau), color='r', linestyle='--', lw=1.5, alpha=0.7, label=r"$1/\tau_{\mathrm{DRW}}$")

        # Model PSD
        # Compute model PSD for each posterior sample and plot the median and 16/84 percentiles
        psd_samples = []
        n_samp = np.min([50, len(samples['log_tau_drw0'])])
        for i in range(n_samp):
            sample_params = {k: jnp.array(v[i]) for k, v in samples.items()}
            psd_i = (2.0 * jnp.pi) * model.psd(sample_params, 2 * np.pi * freqs, b=0, sigma_n2=0.0)
            psd_samples.append(np.asarray(psd_i))
        psd_samples = np.stack(psd_samples, axis=0)
        psd_median = np.median(psd_samples, axis=0)
        psd_lo = np.percentile(psd_samples, 16, axis=0)
        psd_hi = np.percentile(psd_samples, 84, axis=0)

        ax_psd.plot(freqs, psd_median, lw=2, color='m', alpha=0.8, label="Model PSD")
        ax_psd.fill_between(freqs, psd_lo, psd_hi, color='m', alpha=0.2)

        # Plot the noise level
        ax_psd.axhline(np.median(P_noise), color='gray', linestyle='--', lw=1.5, label="Noise Level")

        ax_lc.set_xlabel('MJD')
        ax_lc.set_ylabel('Magnitude + arbitrary offset')
        ax_lc.invert_yaxis()
        ax_lc.set_xlim(np.min(t_test), np.max(t_test))
        #ax_lc.legend(loc='best')

        # PSD axis formatting
        ax_psd.set_xlabel("Frequency (days$^{-1}$)")
        ax_psd.set_ylabel(r"PSD ($\mathrm{mag}^2$ $\mathrm{days}$)")
        ax_psd.set_xscale("log")
        ax_psd.set_yscale("log")
        ax_psd.grid(False)

        # DRW
        # Plot a line with slope -2 for reference, normalized to match the PSD
        ref_freqs = np.linspace(np.nanmin(freqs), np.nanmax(freqs), 100)
        ref_psd2 = ref_freqs**-2
        ref_psd4 = ref_freqs**-4
        # Normalize the reference line to match the PSD at the median frequency
        median_freq = 1e-2
        median_psd = np.interp(median_freq, freqs, psd_median)
        ref_psd2 *= median_psd / np.interp(median_freq, ref_freqs, ref_psd2)
        ref_psd4 *= median_psd / np.interp(median_freq, ref_freqs, ref_psd4)
        ax_psd.plot(ref_freqs, 10*ref_psd2, 'k--', label="-2")
        ax_psd.plot(ref_freqs, 10*ref_psd4, 'k:', label="-4")
        ax_psd.set_ylim(1e-3, 1e4)
        ax_psd.set_xlim(1e-6, 1e1)

    plt.tight_layout()

    # Save the plot as a PNG file
    output_dir = f"plots/multiband/{prefix}/light_curves_fits"
    os.makedirs(output_dir, exist_ok=True)
    fpath = os.path.join(output_dir, f'{data["z"]:.1f}_{object_id}_light_curve_{suffix}.png')
    logging.info(f"Saving figure to {fpath}")
    plt.savefig(fpath, dpi=120)
    plt.close(fig)
    

def plot_mcmc_traces(samples_dict, data):
    """
    Generalized MCMC trace plotter for any set of parameters.

    Parameters:
    - samples_dict: dict with keys as parameter names and values as arrays of shape (n_samples, ...)
    - data: dict, must contain 'object_id'
    """
    logging.info("Plotting MCMC Traces")

    total_traces = len(samples_dict)
    fig, axes = plt.subplots(total_traces, 1, figsize=(12, 2.5 * total_traces), sharex=True)
    if total_traces == 1:
        axes = [axes]

    for idx, key in enumerate(samples_dict.keys()):
        if 'log_' in key:
            axes[idx].plot(samples_dict[key] / np.log(10), alpha=0.7)
        else:
            axes[idx].plot(samples_dict[key], alpha=0.7)
        axes[idx].set_ylabel(key)
        axes[idx].grid(True)

    axes[-1].set_xlabel("Sample index")
    plt.tight_layout()

    output_dir = f"plots/multiband/{prefix}/mcmc_traces/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{data['z']:.1f}_{data['object_id']}_mcmc_traces_{suffix}.png")
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    logging.info(f"Saved trace plot to {save_path}")

    """
    # Plot eta_A1 vs. log_tau trace if both are present
    if 'eta_A1' in samples_dict and 'log_tau_drw0' in samples_dict:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.scatter(samples_dict['log_tau_drw0'], samples_dict['eta_A1'], alpha=0.7, lw=0.7)
        ax2.set_xlabel('log_tau_drw0')
        ax2.set_ylabel('eta_A1')
        ax2.set_title('Trace: eta_A1 vs. log_tau_drw0')
        ax2.grid(True)
        save_path2 = os.path.join(output_dir, f"{data['z']:.1f}_{data['object_id']}_etaA1_vs_logtau.png")
        plt.tight_layout()
        plt.savefig(save_path2, dpi=100)
        plt.close(fig2)
        print("Saved eta_A1 vs. log_tau trace plot to", save_path2)

    # Plot eta_A1 vs. log_sigma_hat0 trace if both are present
    if 'eta_A1' in samples_dict and 'log_sigma_hat0' in samples_dict:
        fig_eta_sigma, ax_eta_sigma = plt.subplots(figsize=(6, 5))
        ax_eta_sigma.scatter(samples_dict['log_sigma_hat0'], samples_dict['eta_A1'], alpha=0.7, lw=0.7)
        ax_eta_sigma.set_xlabel('log_sigma_hat0')
        ax_eta_sigma.set_ylabel('eta_A1')
        ax_eta_sigma.set_title('Trace: eta_A1 vs. log_sigma_hat0')
        ax_eta_sigma.grid(True)
        save_path_eta_sigma = os.path.join(output_dir, f"{data['z']:.1f}_{data['object_id']}_etaA1_vs_logsigma.png")
        plt.tight_layout()
        plt.savefig(save_path_eta_sigma, dpi=100)
        plt.close(fig_eta_sigma)
        logging.info(f"Saved eta_A1 vs. log_sigma_hat0 trace plot to {save_path_eta_sigma}")

    # Plot log_tau_drw0 vs. log_sigma_hat0 trace if both are present
    if 'log_tau_drw0' in samples_dict and 'log_sigma_hat0' in samples_dict:
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        ax3.scatter(samples_dict['log_tau_drw0'], samples_dict['log_sigma_hat0'], alpha=0.7, lw=0.7)
        ax3.set_xlabel('log_tau_drw0')
        ax3.set_ylabel('log_sigma_hat0')
        ax3.set_title('Trace: log_tau_drw0 vs. log_sigma_hat0')
        ax3.grid(True)
        save_path3 = os.path.join(output_dir, f"{data['z']:.1f}_{data['object_id']}_logtau_vs_logsigma.png")
        plt.tight_layout()
        plt.savefig(save_path3, dpi=100)
        plt.close(fig3)
        logging.info(f"Saved log_tau_drw0 vs. log_sigma_hat0 trace plot to {save_path3}")
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
    for k in labels:
        a = np.asarray(samples_flat[k]).ravel()
        if idx is not None:
            a = a[idx]
        if log10_if_startswith and k.startswith(log10_if_startswith):
            a = a / np.log(10.0)  # ln -> log10
        cols.append(a.astype(np.float32, copy=False))
    X = np.column_stack(cols)

    # drop rows with any NaN/Inf
    finite = np.all(np.isfinite(X), axis=1)
    X = X[finite]
    if X.shape[0] == 0:
        raise ValueError("No finite samples to plot after cleaning.")

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

            # density=True is fine after rescaling; values stay reasonable
            ax.hist(x_s, bins=bins, range=(l_s, h_s), density=True, edgecolor="none")

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
    log_sigma0 = np.array([r['log_sigma0'] for r in results])
    log_sigma0_err = np.array([r['log_sigma0_err'] for r in results])
    log_tau_fake = np.array([r['log_tau_fake'] for r in results])
    log_tau_drw0 = np.array([r['log_tau_drw0'] for r in results])
    log_tau_drw0_err = np.array([r['log_tau_drw0_err'] for r in results])
    t_obs_length = np.array([r['t_obs_length'] for r in results])
    t_rf_length = np.array([r['t_rf_length'] for r in results])

    rho = 10**log_tau_drw0 / t_rf_length

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: log_sigma_fake vs log_sigma0, color by rho
    x1 = log_sigma_fake
    y1 = log_sigma0
    y1err = log_sigma0_err
    c1 = rho
    xmin1, xmax1 = -2, 0.5
    sc1 = axes[0].scatter(x1, y1, c=c1, cmap='plasma', s=40, edgecolor='k', alpha=0.8)
    axes[0].errorbar(x1, y1, yerr=y1err, fmt='none', ecolor='gray', alpha=0.4, elinewidth=1, capsize=2)
    axes[0].plot([xmin1, xmax1], [xmin1, xmax1], 'm--', lw=2)
    axes[0].set_xlabel(r'$\log_{10}\,\sigma_\mathrm{fake}$')
    axes[0].set_ylabel(r'$\log_{10}\,\sigma_0$')
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlim(xmin1, xmax1)
    axes[0].set_ylim(xmin1, xmax1)

    # Panel 2: log_tau_fake vs log_tau_drw0 (swapped x and y), color by rho
    x2 = log_tau_fake
    y2 = log_tau_drw0
    y2err = log_tau_drw0_err
    c2 = rho
    xmin2, xmax2 = 0, 6
    sc2 = axes[1].scatter(x2, y2, c=c2, cmap='plasma', s=40, edgecolor='k', alpha=0.8)
    axes[1].errorbar(x2, y2, yerr=y2err, fmt='none', ecolor='gray', alpha=0.4, elinewidth=1, capsize=2)
    axes[1].plot([xmin2, xmax2], [xmin2, xmax2], 'm--', lw=2)
    axes[1].set_xlabel(r'$\log_{10}\,\tau_\mathrm{fake}$')
    axes[1].set_ylabel(r'$\log_{10}\,\tau_\mathrm{DRW,0}$')
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
    rows,                 # list of dicts (each dict = one object)
    bands=('u', 'g', 'r', 'i', 'z'),
    *,
    broken_pl=False,  # if False, use single PL instead of broken PL for model ribbon
    inject_fake=False,
    residual=True,       # subtract UV from BOTH σ and τ
    show=False,
    debug=True,
):
    """
    Plot log10 σ_band and log10 τ_band,RF vs log10 λ_RF with population ribbons.

    Each item in `rows` must provide:
      lam_s, eta_A1, eta_A2, eta_tau1, eta_tau2, eta_break, log_sigma0, log_tau_drw0, z,
      and per-band: log_sigma_band_{b}, log_tau_band_{b}_RF.

    If residual=True, also requires per-row UV references:
      'log_tau_UV_RF' and 'log_sigma_UV' (these names are fixed by design).

    NEW: the model ribbons (both σ and τ) come from analytical propagation of the
         η-slope 1σ uncertainties using the per-row *_err fields:
           eta_A1_err, eta_A2_err, eta_tau1_err, eta_tau2_err.
         We build median curves and an envelope from the 4 corner combos (±1σ each).

    NEW (inject_fake=True): also plot the injected single-slope models using
         alpha_sigma (for σ) and beta_tau (for τ) and include them in the annotations.
    """
    if not rows:
        raise ValueError("`rows` is empty.")

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import linregress

    # ---- helpers (list-of-dicts → arrays) ----
    def arr(key):
        return np.asarray([row[key] for row in rows], dtype=float)

    def med(key):
        a = arr(key)
        return float(np.nanmedian(a))

    def med_err(key):  # median of per-row 1σ uncertainties
        a = arr(key)
        return float(np.nanmedian(a))

    z       = arr('z')
    tau_uv  = arr('log_tau_UV_RF')   if residual else np.zeros(len(rows))
    sig_uv  = arr('log_sigma_UV')    if residual else np.zeros(len(rows))

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 6.2),
                                   sharex=True, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.02)

    def inward(ax):
        ax.tick_params(direction='in', which='both', top=True, right=True, length=3, pad=2)
        for s in ax.spines.values():
            s.set_linewidth(1.0)

    # ---- scatter ----
    x_all, x_all_tau, y_all_tau = [], [], []
    plotted_bands = []

    for b in bands:
        lam_rf = (lambda_pivot[b]) / (1.0 + z)      # Å, rest-frame per row
        x = np.log10(lam_rf)

        y_sigma = arr(f'log_sigma_band_{b}')
        y_tau_abs = arr(f'log_tau_band_{b}_RF')

        y_sigma = y_sigma - sig_uv if residual else y_sigma
        y_tau   = y_tau_abs - tau_uv if residual else y_tau_abs

        m1 = np.isfinite(x) & np.isfinite(y_sigma)
        m2 = np.isfinite(x) & np.isfinite(y_tau)
        if m1.any() or m2.any():
            plotted_bands.append(b)

        ax1.scatter(x[m1], y_sigma[m1], s=40, alpha=0.8, color=colors.get(b),
                    edgecolor='none', zorder=7)
        ax2.scatter(x[m2], y_tau[m2],   s=40, alpha=0.8, color=colors.get(b),
                    edgecolor='none', zorder=7)

        if m1.any(): x_all.append(x[m1])
        if m2.any(): x_all_tau.append(x[m2]); y_all_tau.append(y_tau[m2])

    x_all     = np.concatenate(x_all)     if x_all     else np.array([])
    x_all_tau = np.concatenate(x_all_tau) if x_all_tau else np.array([])
    y_all_tau = np.concatenate(y_all_tau) if y_all_tau else np.array([])

    if x_all.size:
        xmin, xmax = float(np.nanmin(x_all)) - 0.1, float(np.nanmax(x_all)) + 0.1
        ax2.set_xlim(xmin, xmax)
    else:
        xmin, xmax = 3.3, 4.2

    # ---- model ribbons from η ± 1σ (no Monte Carlo) ----
    lam_grid    = np.linspace(10**xmin, 10**xmax, 400).astype(float)
    loglam_grid = np.log10(lam_grid)

    # Center (population medians)
    lam_s_med = med('lam_s')
    ds_med    = med('eta_break')

    eta_A1_med,   eta_A2_med   = med('eta_A1'),   med('eta_A2')
    eta_tau1_med, eta_tau2_med = med('eta_tau1'), med('eta_tau2')

    # Median 1σ (per-object errors summarized by median)
    sig_eta_A1 = med_err('eta_A1_err')
    sig_eta_A2 = med_err('eta_A2_err')
    sig_eta_t1 = med_err('eta_tau1_err')
    sig_eta_t2 = med_err('eta_tau2_err')

    # Median intercepts (already log10). For τ, convert to RF first then median.
    sig0_all = arr('log_sigma0') - (arr('log_sigma_UV') if residual else 0.0)
    tau0_rf_all = arr('log_tau_drw0') - np.log10(1.0 + arr('z'))
    if residual:
        tau0_rf_all = tau0_rf_all - arr('log_tau_UV_RF')

    sig0_med  = float(np.nanmedian(sig0_all))
    tau0_med  = float(np.nanmedian(tau0_rf_all))

    # Shape function (broken power-law; same for σ and τ)
    def shp(e1, e2):
        if broken_pl:
            return log_broken_pl(lam_grid, lam_s_med, e1, e2, ds_med)
        else:
            return log_single_pl(lam_grid, lam_s_med, e1)

    # Central curves
    center_sigma = sig0_med + shp(eta_A1_med,   eta_A2_med)
    center_tau   = tau0_med + shp(eta_tau1_med, eta_tau2_med)

    # Four-corner envelopes (η1±σ1, η2±σ2)
    A1_lo, A1_hi = eta_A1_med - sig_eta_A1, eta_A1_med + sig_eta_A1
    A2_lo, A2_hi = eta_A2_med - sig_eta_A2, eta_A2_med + sig_eta_A2

    T1_lo, T1_hi = eta_tau1_med - sig_eta_t1, eta_tau1_med + sig_eta_t1
    T2_lo, T2_hi = eta_tau2_med - sig_eta_t2, eta_tau2_med + sig_eta_t2

    sigma_corners = np.vstack([
        sig0_med + shp(A1_lo, A2_lo),
        sig0_med + shp(A1_lo, A2_hi),
        sig0_med + shp(A1_hi, A2_lo),
        sig0_med + shp(A1_hi, A2_hi),
    ])
    tau_corners = np.vstack([
        tau0_med + shp(T1_lo, T2_lo),
        tau0_med + shp(T1_lo, T2_hi),
        tau0_med + shp(T1_hi, T2_lo),
        tau0_med + shp(T1_hi, T2_hi),
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

    # ---- injected ("fake") slope overlays & comparisons ----
    have_fake_fields = all(k in rows[0] for k in ('alpha_sigma', 'beta_tau'))
    alpha_sigma_med = beta_tau_med = None

    if inject_fake and have_fake_fields:
        # Median injected slopes across objects
        alpha_sigma_med = med('alpha_sigma')
        beta_tau_med    = med('beta_tau')

        # Use single-slope shapes (same slope on both sides of the break)
        fake_sigma_curve = sig0_med + shp(alpha_sigma_med, alpha_sigma_med)
        fake_tau_curve   = tau0_med + shp(beta_tau_med,    beta_tau_med)

        # Plot as dashed gray overlays
        ax1.plot(loglam_grid, fake_sigma_curve, ls='--', lw=1.2, color='0.25',
                 zorder=4, label='Injected σ-slope')
        ax2.plot(loglam_grid, fake_tau_curve,   ls='--', lw=1.2, color='0.25',
                 zorder=4, label='Injected τ-slope')

    # ---- labels & axes ----
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
    def A_to_loglam(x): return np.log10(x)
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

    # ---- legend (bands + model + injected) ----
    band_handles = [
        Line2D([0], [0], linestyle='none', marker='o', markersize=6,
               markerfacecolor=colors.get(b), markeredgecolor='none',
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

    # ---- annotations (medians; include injected if requested) ----
    txt_sigma = (rf'$\eta_{{A,1}} = {eta_A1_med:+.3f}\,\pm\,{sig_eta_A1:.3f}$' '\n'
                 rf'$\eta_{{A,2}} = {eta_A2_med:+.3f}\,\pm\,{sig_eta_A2:.3f}$')
    if inject_fake and have_fake_fields:
        d1 = eta_A1_med - alpha_sigma_med
        d2 = eta_A2_med - alpha_sigma_med
        txt_sigma += ('\n' +
                      rf'$\alpha_\sigma^\mathrm{{(inj)}} = {alpha_sigma_med:+.3f}$' '\n' +
                      rf'$\Delta\eta_{{A,1}} = {d1:+.3f},\;\Delta\eta_{{A,2}} = {d2:+.3f}$')

    ax1.text(0.02, 0.96, txt_sigma, transform=ax1.transAxes, va='top', ha='left', alpha=1.0,
             fontsize=10, bbox=dict(boxstyle='round,pad=0.25', fc='white', lw=0.8), zorder=10)

    txt_tau = (rf'$\eta_{{\tau,1}} = {eta_tau1_med:+.3f}\,\pm\,{sig_eta_t1:.3f}$' '\n'
               rf'$\eta_{{\tau,2}} = {eta_tau2_med:+.3f}\,\pm\,{sig_eta_t2:.3f}$')
    if inject_fake and have_fake_fields:
        d1t = eta_tau1_med - beta_tau_med
        d2t = eta_tau2_med - beta_tau_med
        txt_tau += ('\n' +
                    rf'$\beta_\tau^\mathrm{{(inj)}} = {beta_tau_med:+.3f}$' '\n' +
                    rf'$\Delta\eta_{{\tau,1}} = {d1t:+.3f},\;\Delta\eta_{{\tau,2}} = {d2t:+.3f}$')

    ax2.text(0.02, 0.96, txt_tau, transform=ax2.transAxes, va='top', ha='left', alpha=1.0,
             fontsize=10, bbox=dict(boxstyle='round,pad=0.25', fc='white', lw=0.8), zorder=10)

    # ---- quick diag ----
    if debug and x_all_tau.size:
        slope_pts = linregress(x_all_tau, y_all_tau).slope
        slope_model = np.gradient(center_tau, loglam_grid).mean()
        print(f"[diag] slope(points) d logτ / d logλ ≈ {slope_pts:+.3f}")
        print(f"[diag] slope(model ) d logτ / d logλ ≈ {slope_model:+.3f}")
        print(f"[diag] medians: ηA1={eta_A1_med:+.3f}±{sig_eta_A1:.3f}, "
              f"ηA2={eta_A2_med:+.3f}±{sig_eta_A2:.3f}, "
              f"ητ1={eta_tau1_med:+.3f}±{sig_eta_t1:.3f}, "
              f"ητ2={eta_tau2_med:+.3f}±{sig_eta_t2:.3f}")
        if inject_fake and have_fake_fields:
            print(f"[diag] injected: α_σ={alpha_sigma_med:+.3f}, β_τ={beta_tau_med:+.3f}")
            print(f"[diag] deltas: ΔηA1={eta_A1_med-alpha_sigma_med:+.3f}, "
                  f"ΔηA2={eta_A2_med-alpha_sigma_med:+.3f}, "
                  f"Δητ1={eta_tau1_med-beta_tau_med:+.3f}, "
                  f"Δητ2={eta_tau2_med-beta_tau_med:+.3f}")

    out_dir = f"plots/multiband/{prefix}/powerlaw/"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{suffix}.png")
    fig.savefig(save_path, dpi=300)
    print(f"Saved power law plot to {save_path}")

    if show:
        plt.show()
    plt.close()