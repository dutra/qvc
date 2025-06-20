import matplotlib.pyplot as plt
plt.style.use("style.mplstyle")
import corner
import numpy as np
import os
import jax.numpy as jnp
import arviz as az
import numpyro
from numpyro.diagnostics import print_summary

prefix = os.environ.get('PREFIX', "test")
suffix = os.environ.get('SUFFIX', "test")

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


def plot_trace_numpyro_for_object(mcmc, data, i, batch_data_len):
    """
    Plot trace plots for object-specific parameters from NumPyro MCMC samples.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        A completed NumPyro MCMC sampler.
    data : dict
        Object metadata (must include 'object_id').
    i : int
        Index of the object in the batch.
    batch_data_len : int
        Total number of objects in the batch.
    prefix : str
        Prefix for output directory.
    suffix : str
        Suffix for output directory.
    """
    object_id = data['object_id']
    samples = mcmc.get_samples(group_by_chain=True)

    # Extract per-object samples
    obj_samples = {
        k: v[..., i] if v.ndim == 3 and v.shape[-1] == batch_data_len else v
        for k, v in samples.items()
    }

    # Clean parameter names like param_3 → param
    obj_samples_clean = {
        k[:-(len(f"_{i}"))] if k.endswith(f"_{i}") else k: v
        for k, v in obj_samples.items()
    }

    # Convert to ArviZ InferenceData
    idata = az.from_dict(posterior=obj_samples_clean)

    # Get parameter names
    var_names = list(idata.posterior.data_vars)
        
    var_names = [k for k in var_names if k in ['eta_A1', 'eta_A2', 'eta_tau1', 'eta_tau2', 'log_sigma_hat0', 'log_tau_drw0', 'poly1']]

    print(var_names)
    n_vars = len(var_names)

    # Create trace plots
    fig, axes = plt.subplots(n_vars, 2, figsize=(14, 2.8 * n_vars), constrained_layout=True)
    if n_vars == 1:
        axes = axes.reshape(1, 2)

    az.plot_trace(idata, var_names=var_names, compact=True, axes=axes, show=False)

    # Save plot
    output_dir = f"mcmc_traces/{prefix}_{suffix}/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{data['z']:.1f}_{object_id}_mcmc_traces.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved trace plot to {save_path}")


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

def plot_posterior_for_object(mcmc, data, i, batch_data_len):
    """
    Plot a corner plot of posterior parameters for a specific object from NumPyro MCMC output.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        Completed MCMC object.
    data : dict
        Object metadata (must contain 'object_id' and 'z').
    i : int
        Index of the object in the batch.
    batch_data_len : int
        Total number of objects in the batch.
    prefix : str
        Output directory prefix.
    suffix : str
        Output directory suffix.
    """
    object_id = data['object_id']

    # Get flat samples
    samples_flat = mcmc.get_samples(group_by_chain=False)

    # Select per-object parameters
    obj_samples = {
        k: v[:, i] if v.ndim == 2 and v.shape[1] == batch_data_len else v
        for k, v in samples_flat.items()
    }

    # Clean names and flatten vector-valued parameters
    obj_samples_flattened = {}
    for k, v in obj_samples.items():
        print(k)
        if k not in ['eta_A1', 'eta_A2', 'eta_tau1', 'eta_tau2', 'log_sigma_hat0', 'log_tau_drw0', 'poly1']:
            continue
        v = np.asarray(v)
        base_name = k[:-(len(f"_{i}"))] if k.endswith(f"_{i}") else k
        if v.ndim == 1:
            obj_samples_flattened[base_name] = v
        elif v.ndim == 2:
            for j in range(v.shape[1]):
                obj_samples_flattened[f"{base_name}_{j}"] = v[:, j]
        else:
            print(f"Skipping {k} with shape {v.shape}")

    # Stack into matrix for corner plot
    corner_data = np.vstack([obj_samples_flattened[k] for k in obj_samples_flattened]).T
    labels = list(obj_samples_flattened.keys())

    fig = corner.corner(corner_data, labels=labels, show_titles=True)

    # Save plot
    output_dir = f"posterior_plots/{prefix}_{suffix}/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{data['z']:.1f}_{object_id}_posterior.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"Saved posterior corner plot to {save_path}")
    return fig


def save_combined_plot(samples, model, X, y, yerr, band_idx, data, fit_bestP=False, psd_results=None):
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
            mu, std = model.pred(samples, (t_test, jnp.full_like(t_test, n, dtype=int)))
            #print(mu, std, '!!!!!!!!!!!!!!')
        else:
            posterior_median = {k: np.median(v, axis=0) for k, v in samples.items()}
            mu, std = model.pred(posterior_median, (t_test, jnp.full_like(t_test, n, dtype=int)))

        # Plot the predictions
        ax_lc.plot(t_test, mu+offsets[n], alpha=0.8, color=colors[band_idx_map[n]], lw=1.0)
        ax_lc.fill_between(t_test, mu+offsets[n]-std, mu+offsets[n]+std, alpha=0.3, 
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
                noise_var = np.mean(yerr[m][mask] ** 2) + np.exp(np.median(2 * samples['log_jitter'], axis=0)) #[m]
            else:
                noise_var = np.mean(yerr[m][mask] ** 2) + np.exp(2 * np.median(samples['log_jitter'], axis=0))[n]
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
            print('model PSD')
            band = band_idx_map[n]
            result = psd_results[band].items()
            freqs = psd_results[band]["freqs"]
            psd = psd_results[band]["psd"]
            ax_psd.plot(freqs, psd, label=f"{band}-band", color=colors.get(band, "black"), lw=2)
    
    ax_lc.set_xlabel('MJD')
    ax_lc.set_ylabel('Magnitude + arbitrary offset')
    ax_lc.invert_yaxis()
    #ax_lc.legend(loc='best')

    if not fit_bestP:

        log_sigma_band = [f"{a:.2f}" for a in data['log_sigma_band']]
        log_sigma_band = ",".join(log_sigma_band)
        log_sigma_band_err = [f"{a:.2f}" for a in data['log_sigma_band_err']]
        log_sigma_band_err = ",".join(log_sigma_band_err)

        log_tau_band_RF = [f"{a:.2f}" for a in data['log_tau_band_RF']]
        log_tau_band_RF = ",".join(log_tau_band_RF)
        log_tau_band_RF_err = [f"{a:.2f}" for a in data['log_tau_band_RF_err']]
        log_tau_band_RF_err = ",".join(log_tau_band_RF_err)
        # Annotate tau_RF and sigma_RF with their values and errors
        ax_lc.annotate(
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
            f"$\\mathrm{{poly_1}}$: {data['poly1']:.2f} ± {data['poly1_err']:.2f}",
            #f"$\\log_{{10}}(w)$: {data['log_w']:.2f} ± {data['log_w_err']:.2f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.4),
        )
        # Annotate the legend with each letter in the same color
        for i, band in enumerate(np.flip(clean_bands)):
            ax_lc.annotate(
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
    print(f"Saving figure to ", fpath)
    plt.savefig(fpath, dpi=120)
    plt.close(fig)
    

def plot_mcmc_traces(mcmc, data):
    samples = mcmc.get_samples(group_by_chain=True)
    num_warmup = mcmc.num_warmup
    object_id = data['object_id']

    param_names = [k for k in samples if k != 'log_kernel_param']
    print(param_names)
    param_names = param_names[:20]
    n_params = len(param_names)
    num_chains, total_samples = samples[param_names[0]].shape

    # Thinning indices (along sample axis)
    thinned_idx = np.arange(total_samples)[::thinning]
    thinned_warmup = num_warmup // thinning

    # Limit total figure height for display
    height_per_plot = 2.0
    max_height = 50
    fig_height = min(n_params * height_per_plot, max_height)

    fig, axes = plt.subplots(n_params, 1, figsize=(12, fig_height), sharex=True)

    if n_params == 1:
        axes = [axes]  # ensure axes is iterable

    for i, param in enumerate(param_names):
        ax = axes[i]
        for c in range(num_chains):
            trace = samples[param][c, thinned_idx]
            ax.plot(thinned_idx, trace, alpha=0.6, lw=0.5, label=f'Chain {c+1}')
        ax.axvline(thinned_warmup, color='k', ls='--', lw=0.8)
        ax.set_ylabel(param, fontsize=10)
        ax.grid(True, alpha=0.3)
        #if i == 0:
            #ax.legend(loc='upper right', fontsize=7)

    axes[-1].set_xlabel("Step (thinned)", fontsize=11)
    plt.tight_layout()

    output_dir = f"mcmc_traces/{prefix}_{suffix}/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{object_id}_mcmc_traces.png")
    plt.savefig(save_path, dpi=150)
    print("Saved trace plot to", save_path)
    plt.close(fig)
    return fig
