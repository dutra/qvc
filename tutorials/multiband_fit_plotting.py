import matplotlib.pyplot as plt
plt.style.use("style.mplstyle")
import corner

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
        r'$\log(\sigma_RF)$': log_sigma_RF,
        r'poly1': samples['poly1'],
        r'mean': samples['mean'][:,0],
    }
    # Convert the samples to a 2D array for corner
    corner_data = np.vstack([posterior_samples[key] for key in posterior_samples.keys()]).T
    fig = corner.corner(corner_data, labels=list(posterior_samples.keys()), show_titles=True)
    output_dir = "posterior_plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{object_id}_posterior.png"))
    plt.close(fig)
    return fig

def save_combined_plot(samples, model, X, y, yerr, band_idx, object_id, clean_bands):
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
        posterior_median = {k: jnp.median(v, axis=0) for k, v in samples.items()}
        mu, std = model.pred(posterior_median, (t_test, np.full_like(t_test, n, dtype=int)))
        # Plot the predictions
        ax.plot(t_test, mu+offsets[n], alpha=0.8, color=colors[band_idx_map[n]], lw=2.5)
        ax.fill_between(t_test, mu+offsets[n]-std, mu+offsets[n]+std, alpha=0.3, 
                lw=0.5, color=colors[band_idx_map[n]])
    ax.set_xlabel('Days')
    ax.set_ylabel('Magnitude + arbitrary offset')
    ax.invert_yaxis()  # Magnitudes are brighter when lower
    #ax.legend(loc='upper right')
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


