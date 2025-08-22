import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
plt.style.use("style.mplstyle")
import corner
import numpy as np
import os
import re
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
            print("Corner Constant param: ", flat_labels[i])
            corner_data[:, i] += np.random.normal(0, 1e-6, size=corner_data.shape[0])
        if 'log_' in flat_labels[i]:
            corner_data[:, i] = corner_data[:, i] / np.log(10)

    fig = corner.corner(corner_data, labels=flat_labels, show_titles=True, 
                        quantiles=[0.16, 0.5, 0.84], bins=bins)

    # Save plot
    output_dir = f"results/posterior_plots/{prefix}"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{z:.1f}_{object_id}_posterior_{suffix}.png")
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    logging.info(f"Saved posterior corner plot to {save_path}")
    return fig

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
    output_dir = f"results/broken_power_law/{prefix}"
    os.makedirs(output_dir, exist_ok=True)
    fpath = os.path.join(output_dir, f'broken_power_law_{suffix}.png')
    logging.info(f"Saving figure to {fpath}")
    fig.savefig(fpath, dpi=200)
    plt.close(fig)

import numpy as np

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
    # --- Convert omega -> frequency grid
    omega = np.asarray(omega, float)
    f_raw = omega / (2.0 * np.pi)

    # --- Lag subtraction
    (t_lag, band_idx), _ = model.my_lag_transform(model.X, model.has_lag, params)
    t_lag = np.asarray(t_lag, float)
    band_idx = np.asarray(band_idx, int)

    # --- Mean subtraction via mean_func
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

    # --- Normalize amplitudes to band 0 scale
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

    # --- Lomb–Scargle
    ls = LombScargle(t_lag, y, yerr)
    P_raw = ls.power(f_raw, normalization=normalization)

    hf_sel = (f_raw > 1/20.0)
    P_noise = np.nanmedian(P_raw[hf_sel])
    print("P_noise:", P_noise)
    P_raw = np.maximum(P_raw - P_noise, 0.0)  # keep non-negative

    # --- Log-binning in f
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

    return np.array(f_bin), np.array(P_bin), np.array(P_lo), np.array(P_hi), np.array(counts)


def save_combined_plot(samples, model, X, y, yerr, band_idx, data, psd_results=None):
    logging.info("Saving combined plot")

    clean_bands = data['clean_bands']
    object_id = data['object_id']
    band_idx_map = {i: b for i, b in enumerate(clean_bands)}

    fig, (ax_lc, ax_psd) = plt.subplots(2, 1, figsize=(10, 10), sharex=False, gridspec_kw={'height_ratios': [1.5, 1]})
    offsets = np.arange(len(clean_bands)) * 0.25

    t = X[0]
    for n in np.unique(band_idx):
        m = band_idx == n
        # Plot the observed data
        ax_lc.errorbar(t[m], y[m]+offsets[n], yerr=yerr[m], fmt='o', 
                label=f'{band_idx_map[n]}-band', alpha=0.7, color=colors[band_idx_map[n]], lw=1.0, capsize=1, markersize=1)
        # Generate test times for predictions
        t_test = np.linspace(t.min(), t.max(), 1000)
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

    # Ensure all elements of posterior_median are jnp arrays
    for k in posterior_median:
        posterior_median[k] = jnp.array(posterior_median[k])

    # PSD calculation and plotting
    freqs = np.logspace(-6, -1, 250)

    # Data PSD
    f_bin, P_bin, P_lo, P_hi, cts = combined_lomb_scargle_from_model(model, posterior_median, 2*np.pi*freqs)
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
        psd_i = model.psd(sample_params, 2 * np.pi * freqs, b=0, sigma_n2=0.0)
        psd_samples.append(np.asarray(psd_i))
    psd_samples = np.stack(psd_samples, axis=0)
    psd_median = np.median(psd_samples, axis=0)
    psd_lo = np.percentile(psd_samples, 16, axis=0)
    psd_hi = np.percentile(psd_samples, 84, axis=0)

    ax_psd.plot(freqs, psd_median, lw=2, color='m', alpha=0.8, label="Model PSD")
    ax_psd.fill_between(freqs, psd_lo, psd_hi, color='m', alpha=0.2)
    
    ax_lc.set_xlabel('MJD')
    ax_lc.set_ylabel('Magnitude + arbitrary offset')
    ax_lc.invert_yaxis()
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
    ax_psd.set_xlim(1e-6, 1e-1)

    plt.tight_layout()

    # Save the plot as a PNG file
    output_dir = f"results/light_curves_fits/{prefix}"
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

    output_dir = f"results/mcmc_traces/{prefix}/"
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