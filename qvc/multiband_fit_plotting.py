import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from matplotlib.ticker import ScalarFormatter
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

from multiband_fit_utils import log_broken_pl

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
    save_path = os.path.join(output_dir, f"{z:.1f}_{object_id}_posterior_{suffix}.png")
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    logging.info(f"Saved posterior corner plot to {save_path}")

def plot_posterior_fast(
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

    # Stable column order
    labels = list(samples_flat.keys())

    # Shared subsample index (do this BEFORE stacking to cut memory/compute)
    first = np.asarray(samples_flat[labels[0]]).ravel()
    n_total = first.shape[0]
    if n_total > max_points:
        rng = np.random.default_rng()
        idx = rng.choice(n_total, size=max_points, replace=False)
    else:
        idx = None

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

    # Drop any rows with NaN/Inf across columns (keeps alignment)
    finite = np.all(np.isfinite(X), axis=1)
    X = X[finite]
    if X.shape[0] == 0:
        raise ValueError("No finite samples to plot after cleaning.")

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

    # Corner kwargs tuned for speed (hist-only, modest bins)
    fig = corner.corner(
        X,
        labels=labels,
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
        bins=bins,
        range=rng,
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


def plot_broken_power_law(samples, data):
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

    # --- posterior medians ---
    pm = {k: np.median(np.asarray(samples[k])) for k in
          ["eta_A1","eta_A2","eta_tau1","eta_tau2","eta_break","lam_s"]}
    eta_A1, eta_A2   = pm["eta_A1"], pm["eta_A2"]
    eta_tau1, eta_tau2 = pm["eta_tau1"], pm["eta_tau2"]
    eta_break, lam_s = pm["eta_break"], pm["lam_s"]

    # --- wavelength grid ---
    xlog = np.linspace(2.9, 3.9, 600)
    lam = 10.0**xlog
    y_amp = log_broken_pl(lam, lam_s, eta_A1, eta_A2, eta_break)
    y_tau = log_broken_pl(lam, lam_s, eta_tau1, eta_tau2, eta_break)

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
        for i in range(len(samples['log_tau_drw0'])):
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
