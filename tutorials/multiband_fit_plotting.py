import matplotlib.pyplot as plt
plt.style.use("style.mplstyle")
import corner
import numpy as np
import os
import jax.numpy as jnp

suffix = os.environ.get('SUFFIX', None)

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

def plot_posterior(samples, data, clean_bands=None):
    """
    Plot a corner plot of all posterior parameters found in the samples dict.
    """
    object_id = data['object_id']

    # Only use 1D or 2D arrays (with shape [n_samples, ...])
    posterior_samples = {}
    for key, val in samples.items():
        arr = np.asarray(val)
        if arr.ndim == 1:
            posterior_samples[key] = arr
        elif arr.ndim == 2:
            for i in range(arr.shape[1]):
                posterior_samples[f"{key}_{i}"] = arr[:, i]

    # Stack for corner plot
    corner_data = np.vstack([posterior_samples[k] for k in posterior_samples]).T
    fig = corner.corner(corner_data, labels=list(posterior_samples.keys()), show_titles=True)
    output_dir = "posterior_plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{object_id}_posterior.png"), dpi=200)
    plt.close(fig)
    return fig

def save_combined_plot(samples, model, X, y, yerr, band_idx, data):
    clean_bands = data['clean_bands']
    object_id = data['object_id']
    band_idx_map = {i: b for i, b in enumerate(clean_bands)}

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
    offsets = np.arange(len(clean_bands)) * 0.25

    t = X[0]# + data['times'][0]
    for n in np.unique(band_idx):
        m = band_idx == n
        # Plot the observed data
        ax.errorbar(t[m], y[m]+offsets[n], yerr=yerr[m], fmt='o', 
                label=f'{band_idx_map[n]}-band', alpha=0.7, color=colors[band_idx_map[n]], lw=1.0, capsize=1, markersize=1)
        # Generate test times for predictions
        t_test = np.linspace(t.min(), t.max(), 1000)
        # Compute predictions using the model
        posterior_median = {k: np.median(v, axis=0) for k, v in samples.items()}
        mu, std = model.pred(posterior_median, (t_test, jnp.full_like(t_test, n, dtype=int)))
        # Plot the predictions
        ax.plot(t_test, mu+offsets[n], alpha=0.8, color=colors[band_idx_map[n]], lw=1.0)
        ax.fill_between(t_test, mu+offsets[n]-std, mu+offsets[n]+std, alpha=0.3, 
                lw=0.5, color=colors[band_idx_map[n]])
    ax.set_xlabel('MJD')
    ax.set_ylabel('Magnitude + arbitrary offset')
    ax.invert_yaxis()  # Magnitudes are brighter when lower
    #ax.legend(loc='lower right')
    
    log_sigma_band = [f"{a:.2f}" for a in data['log_sigma_band']]
    log_sigma_band = ",".join(log_sigma_band)
    log_sigma_band_err = [f"{a:.2f}" for a in data['log_sigma_band_err']]
    log_sigma_band_err = ",".join(log_sigma_band_err)

    log_tau_band_RF = [f"{a:.2f}" for a in data['log_tau_band_RF']]
    log_tau_band_RF = ",".join(log_tau_band_RF)
    log_tau_band_RF_err = [f"{a:.2f}" for a in data['log_tau_band_RF_err']]
    log_tau_band_RF_err = ",".join(log_tau_band_RF_err)
    # Annotate tau_RF and sigma_RF with their values and errors
    ax.annotate(
        f"Object ID: {object_id} (z={data['z']:.2f})\n"
        f"$\\log_{{10}}(\\tau_{{UV RF}})$: {data['log_tau_UV_RF']:.2f} ± {data['log_tau_UV_RF_err']:.2f}\n"
        f"$\\log_{{10}}(\\tau_{{RF}})$: {log_tau_band_RF}\n                 ± {log_tau_band_RF_err}\n"
        f"$\\log_{{10}}(\\tau_{{blr}})$: {data['log_tau_blr']:.2f} ± {data['log_tau_blr_err']:.2f}\n"
        f"$\\log_{{10}}(\\sigma_{{UV}})$: {data['log_sigma_hat_UV']:.2f} ± {data['log_sigma_hat_UV_err']:.2f}\n"
        f"$\\log_{{10}}(\\sigma)$: {log_sigma_band}\n                 ± {log_sigma_band_err}\n"
        f"$\\eta_{{A_1}}$: {data['eta_A1']:.2f} ± {data['eta_A1_err']:.2f}\n"
        f"$\\eta_{{A_2}}$: {data['eta_A2']:.2f} ± {data['eta_A2_err']:.2f}\n"
        f"$\\eta_{{\\tau_1}}$: {data['eta_tau1']:.2f} ± {data['eta_tau1_err']:.2f}\n"
        f"$\\eta_{{\\tau_2}}$: {data['eta_tau2']:.2f} ± {data['eta_tau2_err']:.2f}\n"
        f"$\\eta_{{\\mathrm{{break}}}}$: {data['eta_break']:.2f} ± {data['eta_break_err']:.2f}\n"
        f"$\\eta_{{\\lambda_s}}$: {data['lam_s']:.2f} ± {data['lam_s_err']:.2f}\n"        
        f"$\\mathrm{{poly_1}}$: {data['poly1']:.2f} ± {data['poly1_err']:.2f}\n"
        f"$\\log_{{10}}(w)$: {data['log_w']:.2f} ± {data['log_w_err']:.2f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.4),
    )
    # Annotate the legend with each letter in the same color
    for i, band in enumerate(np.flip(clean_bands)):
        ax.annotate(
            band,
            xy=(0.95 - i * 0.05, 0.95),  # Adjust horizontal spacing
            xycoords="axes fraction",
            color=colors[band],
            fontsize=18,
            fontweight="bold",
            ha="right",
            va="top",
        )
    #ax.set_title(f'Light Curve for AGN {object_id}')

    plt.tight_layout()

    # Save the plot as a PNG file
    output_dir = f"light_curves_fits_{suffix}" if suffix else "light_curves_fits"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{data['z']:.1f}_{object_id}_combined_plot.png'), dpi=120)
    print(f"Saving figure to ", os.path.join(output_dir, f'{data['z']:.1f}_{object_id}_combined_plot.png'))
    plt.close(fig)
    return fig


def plot_mcmc_traces(samples, data):
    """
    Plot the MCMC traces for all parameters in the samples.

    Parameters:
    - samples: dict of arrays, where each key corresponds to a parameter name and the value is an array of MCMC samples.
    - object_id: identifier for the object being analyzed.
    """
    param_names = list(set(samples.keys()) - set(['log_kernel_param']))
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params+2, 1, figsize=(10, 2 * n_params), sharex=True)
    object_id = data['object_id']
    for i, param in enumerate(param_names):
        ax = axes[i] if n_params > 1 else axes
        ax.plot(samples[param], alpha=0.7, lw=0.5)
        ax.set_ylabel(param, fontsize=12)
        ax.grid(True)

    axes[-1].plot(np.log10(np.exp(samples['log_kernel_param'][:, 0])), alpha=0.7, lw=0.5)
    axes[-1].set_ylabel("log10_tau", fontsize=12)
    axes[-2].plot(np.log10(np.exp(samples['log_kernel_param'][:, 1])), alpha=0.7, lw=0.5)
    axes[-2].set_ylabel("log10_sigma", fontsize=12)

    axes[-1].set_xlabel("Step", fontsize=12)
    plt.tight_layout()

    # Save the plot as a PNG file
    output_dir = "mcmc_traces"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{object_id}_mcmc_traces.png"), dpi=120)
    plt.close(fig)
    return fig



def plot_psd(psd_results, object_id):
    """
    Plot the Power Spectral Density (PSD) for each band.

    Args:
        psd_results (dict): A dictionary containing frequencies and PSD for each band.
                            Format: {band: {"freqs": np.ndarray, "psd": np.ndarray}}
        object_id (int): Identifier for the object being analyzed.

    Returns:
        None
    """
    # Create a figure for the PSD plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the PSD for each band
    for band, result in psd_results.items():
        freqs = result["freqs"]
        psd = result["psd"]
        ax.plot(freqs, psd, label=f"{band}-band", color=colors.get(band, "black"), lw=2)

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
    ax.plot(ref_freqs, 10*ref_psd2, 'k--', label="-2")
    ax.plot(ref_freqs, 10*ref_psd4, 'k:', label="-4")

    # Set plot labels and title
    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    ax.set_ylabel("Power Spectral Density", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    ax.set_ylim(1e-5, np.max(psd)*5)

    # Save the plot as a PNG file
    output_dir = "psd_plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{object_id}_psd.png"), dpi=200)
    plt.close(fig)
    return fig
