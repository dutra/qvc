import matplotlib.pyplot as plt
plt.style.use("style.mplstyle")
import corner
import numpy as np
import os


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
    # Extract the posterior samples
    object_id = data['object_id']
    lambda_ref = 2500 # Any reference wavelength
    lambda_pivot_RF = lambda_pivot[clean_bands[0]]/(1 + data['z'])
    log_sigma_RF = np.log10(np.exp(samples['log_kernel_param'][:, 1] + samples['beta']*np.log(lambda_ref/lambda_pivot_RF)))
    log_tau_RF = np.log10(np.exp(samples['log_kernel_param'][:, 0])/(1+data['z']))

    posterior_samples = {
        r'$\beta$': samples['beta'],
        r'$\log(\tau_\mathrm{RF})$': log_tau_RF,
        r'$\log(\sigma_\mathrm{RF})$': log_sigma_RF,
        r'poly1': samples['poly1'],
    }
    for i, band in enumerate(clean_bands):
        posterior_samples[f'mean_{band}'] = samples['mean'][:, i]
        posterior_samples[f'lag_{band}'] = samples['lag'][:, i]
        posterior_samples[f'log_lag_blr_{band}'] = samples['log_lag_blr'][:, i]
        posterior_samples[f'log_amp_delta_blr_{band}'] = samples['log_amp_delta_blr'][:, i]
        posterior_samples[f'log_jitter_{band}'] = samples['log_jitter'][:, i]

    # Convert the samples to a 2D array for corner
    corner_data = np.vstack([posterior_samples[key] for key in posterior_samples.keys()]).T
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
                label=f'{band_idx_map[n]}-band', alpha=0.7, color=colors[band_idx_map[n]], lw=2.0, capsize=3)
        # Generate test times for predictions
        t_test = np.linspace(t.min(), t.max(), 1000)
        # Compute predictions using the model
        posterior_median = {k: np.median(v, axis=0) for k, v in samples.items()}
        mu, std = model.pred(posterior_median, (t_test, np.full_like(t_test, n, dtype=int)))
        # Plot the predictions
        ax.plot(t_test, mu+offsets[n], alpha=0.8, color=colors[band_idx_map[n]], lw=2.5)
        ax.fill_between(t_test, mu+offsets[n]-std, mu+offsets[n]+std, alpha=0.3, 
                lw=0.5, color=colors[band_idx_map[n]])
    ax.set_xlabel('MJD')
    ax.set_ylabel('Magnitude + arbitrary offset')
    ax.invert_yaxis()  # Magnitudes are brighter when lower
    #ax.legend(loc='lower right')

    # Annotate tau_RF and sigma_RF with their values and errors
    ax.annotate(
        f"$\\log_{{10}}(\\tau_{{RF}})$: {data['log_tau_RF']:.2f} ± {data['log_tau_RF_err']:.2f}\n"
        f"$\\log_{{10}}(\\sigma_{{RF}})$: {data['log_sigma_RF']:.2f} ± {data['log_sigma_RF_err']:.2f}\n"
        f"$\\beta$: {data['beta']:.2f} ± {data['beta_err']:.2f}\n"
        f"$\\mathrm{{poly1}}$: {data['poly1']:.2f} ± {data['poly1_err']:.2f}\n",
        #f"$\\mathrm{{poly2}}$: {data['poly2']:.2f} ± {data['poly2_err']:.2f}\n"
        #f"$\\mathrm{{mean}}$: {data['mean']:.2f} ± {data['mean_err']:.2f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.8),
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
    output_dir = "light_curves_fits"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{object_id}_combined_plot.png'))
    plt.close(fig)
    return fig

def save_combined_plot_bestp(bestP, model, X, y, yerr, band_idx, data):
    clean_bands = data['clean_bands']
    object_id = data['object_id']
    band_idx_map = {i: b for i, b in enumerate(clean_bands)}

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
    offsets = np.arange(len(clean_bands)) * 0.25

    t = X[0]    
    for n in np.unique(band_idx):
        m = band_idx == n
        # Plot the observed data
        ax.errorbar(t[m], y[m]+offsets[n], yerr=yerr[m], fmt='o', 
                label=f'{band_idx_map[n]}-band', alpha=0.7, color=colors[band_idx_map[n]], lw=2.0, capsize=3)
        # Generate test times for predictions
        t_test = np.linspace(t.min(), t.max(), 1000)
        # Compute predictions using the model
        posterior_median = bestP #{k: np.median(v, axis=0) for k, v in samples.items()}
        mu, std = model.pred(posterior_median, (t_test, np.full_like(t_test, n, dtype=int)))
        # Plot the predictions
        ax.plot(t_test, mu+offsets[n], alpha=0.8, color=colors[band_idx_map[n]], lw=2.5)
        ax.fill_between(t_test, mu+offsets[n]-std, mu+offsets[n]+std, alpha=0.3, 
                lw=0.5, color=colors[band_idx_map[n]])
    ax.set_xlabel('Days')
    ax.set_ylabel('Magnitude + arbitrary offset')
    ax.invert_yaxis()  # Magnitudes are brighter when lower
    ax.legend(loc='lower right')

    plt.tight_layout()

    # Save the plot as a PNG file
    output_dir = "light_curves_fits"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{object_id}_combined_plot_bestp.png'))
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
    plt.savefig(os.path.join(output_dir, f"{object_id}_mcmc_traces.png"))
    plt.close(fig)
    return fig


