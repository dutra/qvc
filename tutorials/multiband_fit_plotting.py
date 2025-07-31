import matplotlib.pyplot as plt
plt.style.use("style.mplstyle")
import corner
import numpy as np
import os
import re
import jax.numpy as jnp

prefix = os.environ.get('PREFIX', "test")
suffix = os.environ.get('SUFFIX', "test")

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

def plot_posterior_for_object(samples_flat, data, i, batch_data_len, include=None, exclude=['gp', 'f', 'log_amp_delta_blr', 'raw'], bins=20):
    """
    Generalized corner plot of posterior parameters for a specific object.

    Parameters
    ----------
    samples_flat : dict
        Dict of MCMC samples, shape (n_samples, n_objects, ...) or (n_samples, ...) for global params.
    data : dict
        Object metadata (must contain 'object_id' and 'z').
    i : int
        Index of the object in the batch.
    batch_data_len : int
        Total number of objects in the batch.
    include : list or None
        List of parameter names to include (default: all).
    exclude : list or None
        List of parameter names to exclude (default: none).
    bins : int
        Number of bins for corner plot.
    """
    logging.info("Saving posterior plot")
    object_id = data.get('object_id', f'obj_{i}')
    z = data.get('z', 0)

    # Flatten all parameters for plotting
    flat_arrays = []
    flat_labels = []
    for k, v in samples_flat.items():
        arr = np.asarray(v)
        # Select per-object slice if needed
        if arr.ndim == 2 and arr.shape[1] == batch_data_len:
            arr = arr[:, i]
        elif arr.ndim == 3 and arr.shape[1] == batch_data_len:
            arr = arr[:, i, :]
        # Now flatten over bands
        if arr.ndim == 1:
            flat_arrays.append(arr)
            flat_labels.append(k)
        elif arr.ndim == 2:
            for j in range(arr.shape[1]):
                flat_arrays.append(arr[:, j])
                flat_labels.append(f"{k}_{j}")

    if not flat_arrays:
        print("No parameters to plot.")
        return None

    corner_data = np.vstack(flat_arrays).T

    ranges = []
    for i in range(corner_data.shape[1]):
        lo, hi = corner_data[:, i].min(), corner_data[:, i].max()
        if lo == hi:  # constant parameter
            # add jitter so corner doesn't fail
            print("Constant param: ", flat_labels[i])
            corner_data[:, i] += np.random.normal(0, 1e-6, size=corner_data.shape[0])
    
    fig = corner.corner(corner_data, labels=flat_labels, show_titles=True, 
                        quantiles=[0.16, 0.5, 0.84], bins=bins)

    # Save plot
    output_dir = f"posterior_plots/{prefix}_{suffix}/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{z:.1f}_{object_id}_posterior.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    logging.info(f"Saved posterior corner plot to {save_path}")
    return fig


def save_combined_plot(samples, model, X, y, yerr, band_idx, data, fit_bestP=False, psd_results=None):
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
        if fit_bestP:
            result = model.pred(samples, (t_test, jnp.full_like(t_test, n, dtype=int)))
            #print(mu, std, '!!!!!!!!!!!!!!')
        else:
            posterior_median = {k: np.median(v, axis=0) for k, v in samples.items()}
            result = model.pred(posterior_median, (t_test, jnp.full_like(t_test, n, dtype=int)))
            #print(mu, std, '!!!!!!!!!!!!!!')

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


        # --- PSD calculation and plotting ---
        # Remove offset for PSD calculation
        y_band = y[m]
        t_band = t[m]
        # Remove NaNs
        mask = np.isfinite(y_band) & np.isfinite(t_band)
        if np.sum(mask) > 5:
            # Data PSD
            from astropy.timeseries import LombScargle
            freqs = np.logspace(-4, -1, 250)
            psd = LombScargle(t_band[mask], y_band[mask]).power(freqs, normalization='psd')
            # Estimate the noise level (mean of error bars squared)
            if fit_bestP:
                noise_var = np.mean(yerr[m][mask] ** 2) #+ np.exp(np.median(2 * samples['log_jitter'], axis=0)) #[m]
            else:
                noise_var = np.mean(yerr[m][mask] ** 2) #+ np.exp(2 * np.median(samples['log_jitter'], axis=0))[n]
            # The Lomb-Scargle normalization is in mag^2/days, so subtract noise variance
            psd = np.clip(psd - noise_var, a_min=1e-10, a_max=None)
            # Bin the PSD in log-frequency space and plot it
            num_bins = 15
            log_freq_bins = np.logspace(np.log10(freqs.min()), np.log10(freqs.max()), num_bins + 1)
            bin_centers = 10 ** (0.5 * (np.log10(log_freq_bins[:-1]) + np.log10(log_freq_bins[1:])))
            psd_binned = np.zeros(num_bins)
            for i in range(num_bins):
                in_bin = (freqs >= log_freq_bins[i]) & (freqs < log_freq_bins[i + 1])
                if np.any(in_bin):
                    psd_binned[i] = np.median(psd[in_bin])
                else:
                    psd_binned[i] = np.nan
            ax_psd.plot(bin_centers, psd_binned, marker='o', ls='', color=colors[band_idx_map[n]], label=f"{band_idx_map[n]}-band (binned)", markersize=6, alpha=0.8)
            
        
        # Model PSD
        if psd_results is not None:
            logging.info('Plotting Model PSD')
            band = band_idx_map[n]
            result = psd_results[band].items()
            freqs = psd_results[band]["freqs"]
            psd = psd_results[band]["psd"]
            ax_psd.plot(freqs, psd, label=f"{band}-band", color=colors.get(band, "black"), lw=2)
    
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
    median_psd = np.interp(median_freq, freqs, psd)
    ref_psd2 *= median_psd / np.interp(median_freq, ref_freqs, ref_psd2)
    ref_psd4 *= median_psd / np.interp(median_freq, ref_freqs, ref_psd4)
    ax_psd.plot(ref_freqs, 10*ref_psd2, 'k--', label="-2")
    ax_psd.plot(ref_freqs, 10*ref_psd4, 'k:', label="-4")
    ax_psd.set_ylim(1e-5, 1e2)
    ax_psd.set_xlim(1e-4, 1e-1)

    plt.tight_layout()

    # Save the plot as a PNG file
    output_dir = f"light_curves_fits/{prefix}_{suffix}"
    os.makedirs(output_dir, exist_ok=True)
    if fit_bestP:
        fpath = os.path.join(output_dir, f'{data["z"]:.1f}_{object_id}_combined_plot_MLE.png')
    else:
        fpath = os.path.join(output_dir, f'{data["z"]:.1f}_{object_id}_combined_plot.png')
    logging.info(f"Saving figure to {fpath}")
    plt.savefig(fpath, dpi=120)
    plt.close(fig)
    

def plot_mcmc_traces(samples_dict, data, include=None, exclude=['gp', 'f', 'log_amp_delta_blr', 'raw'], figsize=(12, 2.5), alpha=0.7):
    """
    Generalized MCMC trace plotter for any set of parameters.

    Parameters:
    - samples_dict: dict with keys as parameter names and values as arrays of shape (n_samples, ...) or (n_samples, n_dim)
    - data: dict, must contain 'object_id'
    - include: list of parameter names to include (default: all)
    - exclude: list of parameter names to exclude (default: none)
    - figsize: tuple, base figure size (width, height per subplot)
    - alpha: float, line transparency
    """
    logging.info("Plotting MCMC Traces")
    trace_data = {}

    # Flatten and filter parameters
    for name, val in samples_dict.items():
        arr = np.asarray(val)
        # Per-object or global param
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        # Split vector-valued params
        if arr.ndim == 2 and arr.shape[1] > 1:
            for j in range(arr.shape[1]):
                trace_data[f"{name}_{j}"] = arr[:, j]
        elif arr.ndim == 1:
            trace_data[name] = arr


    # Filter included/excluded parameters
    keys = list(trace_data.keys())
    if include is not None:
        keys = [k for k in keys if k.split('_')[0] in include]
    if exclude is not None:
        keys = [k for k in keys if k.split('_')[0] not in exclude]

    total_traces = len(keys)
    fig, axes = plt.subplots(total_traces, 1, figsize=(figsize[0], figsize[1] * total_traces), sharex=True)
    if total_traces == 1:
        axes = [axes]

    for idx, key in enumerate(keys):
        axes[idx].plot(trace_data[key], alpha=alpha)
        axes[idx].set_ylabel(key)
        axes[idx].grid(True)

    axes[-1].set_xlabel("Sample index")
    plt.tight_layout()

    output_dir = f"mcmc_traces/{prefix}_{suffix}/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{data['z']:.1f}_{data['object_id']}_mcmc_traces.png")
    plt.savefig(save_path, dpi=150)
    logging.info(f"Saved trace plot to {save_path}")

    # Plot eta_A1 vs. log_tau trace if both are present
    if 'eta_A1' in trace_data and 'log_tau_drw0' in trace_data:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.scatter(trace_data['log_tau_drw0'], trace_data['eta_A1'], alpha=alpha, lw=0.7)
        ax2.set_xlabel('log_tau_drw0')
        ax2.set_ylabel('eta_A1')
        ax2.set_title('Trace: eta_A1 vs. log_tau_drw0')
        ax2.grid(True)
        save_path2 = os.path.join(output_dir, f"{data['z']:.1f}_{data['object_id']}_etaA1_vs_logtau.png")
        plt.tight_layout()
        plt.savefig(save_path2, dpi=150)
        plt.close(fig2)
        print("Saved eta_A1 vs. log_tau trace plot to", save_path2)

    # Plot eta_A1 vs. log_sigma_hat0 trace if both are present
    if 'eta_A1' in trace_data and 'log_sigma_hat0' in trace_data:
        fig_eta_sigma, ax_eta_sigma = plt.subplots(figsize=(6, 5))
        ax_eta_sigma.scatter(trace_data['log_sigma_hat0'], trace_data['eta_A1'], alpha=alpha, lw=0.7)
        ax_eta_sigma.set_xlabel('log_sigma_hat0')
        ax_eta_sigma.set_ylabel('eta_A1')
        ax_eta_sigma.set_title('Trace: eta_A1 vs. log_sigma_hat0')
        ax_eta_sigma.grid(True)
        save_path_eta_sigma = os.path.join(output_dir, f"{data['z']:.1f}_{data['object_id']}_etaA1_vs_logsigma.png")
        plt.tight_layout()
        plt.savefig(save_path_eta_sigma, dpi=150)
        plt.close(fig_eta_sigma)
        logging.info(f"Saved eta_A1 vs. log_sigma_hat0 trace plot to {save_path_eta_sigma}")

    # Plot log_tau_drw0 vs. log_sigma_hat0 trace if both are present
    if 'log_tau_drw0' in trace_data and 'log_sigma_hat0' in trace_data:
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        ax3.scatter(trace_data['log_tau_drw0'], trace_data['log_sigma_hat0'], alpha=alpha, lw=0.7)
        ax3.set_xlabel('log_tau_drw0')
        ax3.set_ylabel('log_sigma_hat0')
        ax3.set_title('Trace: log_tau_drw0 vs. log_sigma_hat0')
        ax3.grid(True)
        save_path3 = os.path.join(output_dir, f"{data['z']:.1f}_{data['object_id']}_logtau_vs_logsigma.png")
        plt.tight_layout()
        plt.savefig(save_path3, dpi=150)
        plt.close(fig3)
        logging.info(f"Saved log_tau_drw0 vs. log_sigma_hat0 trace plot to {save_path3}")