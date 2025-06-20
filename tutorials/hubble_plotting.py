import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from tqdm import tqdm
import math
import corner
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.cosmology import FlatwCDM, Flatw0waCDM, FlatLambdaCDM
import matplotlib.pyplot as plt
import os
import copy

from hubble_model import K_corr, M_model_agn, M_model_agn_err, M_model_SN, get_model_params, log_tau_UV_RF_pivot
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import interp1d
from dynesty.utils import resample_equal
from tqdm import tqdm
from dynesty import plotting as dyplot

def plot_dynesty(results, cosmo_model, basename="plots/hubble/dynesty", show=False):
    """
    Plot dynesty diagnostics: runplot, traceplot, and cornerpoints using dyplot.
    Saves figures to files with the given basename.
    """

    os.makedirs(os.path.dirname(basename), exist_ok=True)
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)

    # Traceplot
    fig_trace, axes_trace = dyplot.traceplot(
        results,
        labels=model_labels_latex,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_quantiles=[0.16, 0.5, 0.84],
    )
    fig_trace.tight_layout(pad=2.0, h_pad=1)

    fig_trace.savefig(f"{basename}_traceplot.png", dpi=100)
    if show:
        fig_trace.show()
    plt.close(fig_trace)

    # Cornerplot
    fig_corner, axes_corner = dyplot.cornerplot(results, labels=model_labels_latex, quantiles=[0.16, 0.5, 0.84],
                                                 quantiles_2d = [0.393, 0.865, 0.989],
                                                 show_titles=True, title_quantiles=[0.16, 0.5, 0.84],
                                                 color='blue',
                                                 #fig=plt.subplots(1, 1, figsize=(10, 2.5 * len(model_labels))))
    )
    fig_corner.savefig(f"{basename}_cornerplots.png", dpi=100)
    if show:
        fig_corner.show()    
    plt.close(fig_corner)


    # Cornerpoints
    fig_corner, axes_corner = dyplot.cornerpoints(results, labels=model_labels_latex, cmap='plasma')
    fig_corner.savefig(f"{basename}_cornerpoints.png", dpi=100)
    if show:
        fig_corner.show()    
    plt.close(fig_corner)

    # Make a shallow copy of results to avoid touching the real object
    results_plot = copy.deepcopy(results)
    try:
        if results_plot.logz[-1] > 700:
            results_plot.logz[-1] = 700  # Safe maximum for exp
            print("🔧 Clipped logz[-1] to prevent overflow in runplot")

        fig_run, axes_run = dyplot.runplot(results_plot)
        fig_run.savefig(f"{basename}_runplot.png", dpi=100)
        if show:
            fig_run.show()
        plt.close(fig_run)
    except Exception as e:
        print(f"Error in runplot: {e}")
        
def plot_traces(sampler, only_sna=False, cosmo_model='Flatw0waCDM', show=True, use_dynesty=False):
    """
    Plot parameter traces from dynesty nested sampling results.
    
    Parameters
    ----------
    results : dynesty.results.Results
        The result object returned by `sampler.results`.
    labels : list of str, optional
        Parameter names to label each subplot. If None, uses param index.
    figsize : tuple
        Base size for each subplot (width, height).
    """
    if use_dynesty:
        results = sampler.results
        samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
        samples = resample_equal(samples, weights)
    else:
        samples = sampler.get_chain()

    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    ndim = len(model_labels)

    fig, axes = plt.subplots(ndim, 1, figsize=(10, ndim*2.5), sharex=True)
    if ndim == 1:
        axes = [axes]
    print("Plotting traces for cosmological model:", cosmo_model)
    print("Number of parameters:", ndim)
    print("Parameter labels:", model_labels)
    print("Priors: ", priors)
    print("Number of samples:", samples.shape[0])
    print("Number of iterations:", samples.shape[1])
    print("Shape of samples array:", samples.shape)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], color="black", alpha=0.6, lw=0.8)
        ax.set_ylabel(model_labels[i])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Iteration")
    plt.tight_layout()
    if show:
        plt.show()
    if only_sna:
        file_path = f"plots/hubble/traces_{cosmo_model}_sna.png"
    else:
        file_path = f"plots/hubble/traces_{cosmo_model}_agn.png"
    plt.savefig(file_path, dpi=200)

    return fig

def plot_posterior_corner(flat_samples, only_sna=False, cosmo_model='Flatw0waCDM', show=False):
    # Select cosmological parameters based on model
    if cosmo_model == 'FlatwCDM':
        cosmo_params = ['H0', 'Om0', 'w0']
    elif cosmo_model == 'Flatw0waCDM':
        cosmo_params = ['H0', 'Om0', 'w0', 'wa']
    elif cosmo_model == 'FlatLambdaCDM':
        cosmo_params = ['H0', 'Om0']
    else:
        raise ValueError("cosmo_model must be 'FlatwCDM', 'Flatw0waCDM', or 'FlatLambdaCDM'")

    # Model parameters: AGN correlation + SN calibration + cosmology
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)

    fig = corner.corner(
        flat_samples,
        labels=model_labels,
        truths=None,
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12}
    )

    os.makedirs("plots/hubble", exist_ok=True)
    if only_sna:
        fig.suptitle("SNIa only", fontsize=16)
        plt.savefig(f"plots/hubble/posterior_{cosmo_model}_sna.png", dpi=200)
    else:
        fig.suptitle("SNIa + AGN", fontsize=28)
        plt.savefig(f"plots/hubble/posterior_{cosmo_model}_agn.png", dpi=200)

    if show:
        plt.show()
    plt.close()

def plot_cosmo_corner(sampler_sna, sampler_agn, cosmo_model='Flatw0waCDM', show=False, sna_data=None, agn_data=None):
# === Parameter setup ===
    if cosmo_model == 'FlatwCDM':
        param_names = ["H0", "Om0", "w0"]
        labels = [r"$H_0$", r"$\Omega_M$", r"$w_0$"]
    elif cosmo_model == 'Flatw0waCDM':
        param_names = ["H0", "Om0", "w0", "wa"]
        labels = [r"$H_0$", r"$\Omega_M$", r"$w_0$", r"$w_a$"]
    elif cosmo_model == 'FlatLambdaCDM':
        param_names = ["H0", "Om0"]
        labels = [r"$H_0$", r"$\Omega_M$"]
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    n_params = len(labels)
    param_indices = [list(priors.keys()).index(p) for p in param_names]

    # === Get flattened MCMC chains ===
    if sna_data is None and agn_data is None:
        sna_data = sampler_sna.get_chain(flat=True)[:, param_indices]
        agn_data = sampler_agn.get_chain(flat=True)[:, param_indices]

    # === Fast KDE-level calculator ===
    def get_density_levels(values, probs=[0.393, 0.865]):
        z = values.ravel()
        z_sorted = np.sort(z)
        cdf = np.cumsum(z_sorted)
        cdf /= cdf[-1]
        levels = [z_sorted[np.searchsorted(cdf, 1 - p)] for p in probs]
        return np.unique(np.sort(levels))  # ensure strictly increasing

    # === Fast 2D KDE plot with filled contours ===
    def fast_filled_kde(ax, x, y, color, base_alpha=0.4, levels=[0.393, 0.865]):
        data = np.vstack([x, y])
        kde = gaussian_kde(data)

        # Low-resolution grid
        xmin, xmax = np.percentile(x, [0.5, 99.5])
        ymin, ymax = np.percentile(y, [0.5, 99.5])
        xgrid = np.linspace(xmin, xmax, 100)
        ygrid = np.linspace(ymin, ymax, 100)
        xx, yy = np.meshgrid(xgrid, ygrid)
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

        contour_levels = get_density_levels(zz, levels)
        for i in range(len(contour_levels)-1, -1, -1):
            ax.contourf(xx, yy, zz,
                        levels=[contour_levels[i], zz.max()],
                        colors=[color],
                        alpha=base_alpha * (i + 1) / len(contour_levels))
        ax.contour(xx, yy, zz, levels=contour_levels, colors=[color], linewidths=1.2)

    # === Triangle plot ===
    fig, axes = plt.subplots(n_params, n_params, figsize=(10, 10))

    for i in tqdm(range(n_params), desc="Generating corner plot"):
        for j in range(n_params):
            ax = axes[i, j]
            ax.tick_params(direction='in')

            if i < j:
                ax.axis("off")
            elif i == j:
                for data, color in [(sna_data, "blue"), (agn_data, "red")]:
                    kde = gaussian_kde(data[:, i], bw_method=0.1)
                    xmin = data[:, i].min()
                    xmax = data[:, i].max()
                    margin = 0.1 * (xmax - xmin)
                    x_vals = np.linspace(xmin - margin, xmax + margin, 300)
                    ax.plot(x_vals, kde(x_vals), color=color, lw=1.8)
            else:
                fast_filled_kde(ax, sna_data[:, j], sna_data[:, i], "blue", base_alpha=0.4)
                fast_filled_kde(ax, agn_data[:, j], agn_data[:, i], "red", base_alpha=0.4)

            if j == 0:
                ax.set_ylabel(labels[i])
            else:
                ax.set_yticklabels([])

            if i == n_params - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])

    # === Legend ===
    legend_elements = [
        Line2D([0], [0], color="blue", lw=4, label="SN Ia"),
        Line2D([0], [0], color="red", lw=4, label="SN Ia + AGN"),
    ]
    fig.legend(handles=legend_elements, bbox_to_anchor=(0.5, 0.92), loc="upper left", fontsize=26, frameon=False, markerscale=1.5)

    # === Layout & Save ===
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)

    os.makedirs("plots/hubble", exist_ok=True)
    fig.savefig(f"plots/hubble/corner_kde_{cosmo_model}.pdf", bbox_inches="tight", transparent=True)
    fig.savefig(f"plots/hubble/corner_kde_{cosmo_model}.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_hubble(flat_samples, df_agn, df_pantheon, cosmo_model, show=False, completeness=True, show_uncorrected=False, show_true=False):
    """Plot Hubble diagram + residuals, classic Pantheon+ style."""
    # Define cosmological parameter labels
    if cosmo_model == 'FlatwCDM':
        label = r"Flat$w$CDM Model"
    elif cosmo_model == 'Flatw0waCDM':
        label = r"Flat$w_0w_a$CDM Model"
    elif cosmo_model == 'FlatLambdaCDM':
        label = r"Flat$\Lambda$CDM Model"
    else:
        raise ValueError("Invalid cosmology model.")
    
    n_samples = flat_samples.shape[0]
    thin_factor = max(1, n_samples // 200)
    flat_samples = flat_samples[::thin_factor]
    
    z_grid = np.linspace(0.0001, 5, 200)

    # Build model_labels and get indices for cosmological parameters
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    param_indices = {name: model_labels.index(name) for name in model_labels}

    # Compute mu_models using parameter indices by label
    if cosmo_model == 'FlatwCDM':
        mu_models = np.array([
            FlatwCDM(
                H0=s[param_indices['H0']],
                Om0=s[param_indices['Om0']],
                w0=s[param_indices['w0']]
            ).distmod(z_grid).value
            for s in flat_samples
        ])
    elif cosmo_model == 'Flatw0waCDM':
        mu_models = np.array([
            Flatw0waCDM(
                H0=s[param_indices['H0']],
                Om0=s[param_indices['Om0']],
                w0=s[param_indices['w0']],
                wa=s[param_indices['wa']]
            ).distmod(z_grid).value
            for s in flat_samples
        ])
    elif cosmo_model == 'FlatLambdaCDM':
        mu_models = np.array([
            FlatLambdaCDM(
                H0=s[param_indices['H0']],
                Om0=s[param_indices['Om0']]
            ).distmod(z_grid).value
            for s in flat_samples
        ])
    else:
        raise ValueError("Invalid cosmology model.")
        

    results = {key: np.percentile(flat_samples[:, i], [16, 50, 84]) for i, key in enumerate(model_labels)}

    # --- Cosmology model ---    
    mu_model_median = np.percentile(mu_models, 50, axis=0)
    mu_model_16th = np.percentile(mu_models, 16, axis=0)
    mu_model_84th = np.percentile(mu_models, 84, axis=0)

    # Correct the apparent magnitude first
    #completeness2d, mag_centers, z_centers, dm, dz = get_completeness_function_2d(df_agn)
    #delta_mag_arr, delta_mag_arr_errs = compute_delta_mag_bias_2d_zbins(df_agn, completeness2d, mag_centers, z_centers, dm)
    if completeness:
        print("Applying completeness correction in hubble diagram...")
        corrected_apparent_mag = df_agn['apparent_mag_i_corr']
    else:
        print("NOT applying completeness correction in hubble diagram...")
        corrected_apparent_mag = df_agn['apparent_mag_i']

    # Then re-compute the distance modulus
    mu_pred = np.array([
        corrected_apparent_mag - (K_corr(df_agn['z'].values) - K_corr(2)) -
            (M_model_agn(s[param_indices['M0_sn']]+s[param_indices['delta_M0_agn']], 
                         s[param_indices['log_sigma_hat_sq_break']], 
                         s[param_indices['eta_A1_agn']], s[param_indices['eta_A2_agn']], 
                         s[param_indices['eta_break_agn']],
                        s[param_indices['beta_agn']],
                        df_agn['log_sigma_hat_UV'].values, df_agn['log_tau_UV_RF'].values))
        for s in flat_samples
    ])

    mu_pred_16th = np.percentile(mu_pred, 16, axis=0)
    mu_pred_84th = np.percentile(mu_pred, 84, axis=0)
    mu_pred_std = np.sqrt(df_agn['apparent_mag_i_err']**2 +
                 (-2.5 * 0.3 * np.log10(1 + df_agn["z"]))**2 +
                 (0.055 * df_agn["z"])**2 +
                #(results["alpha_agn"][1] * 2*df_agn['log_sigma_hat_UV_err']))**2
                M_model_agn_err(results['M0_sn'][1] + results['delta_M0_agn'][1],
                                  results['log_sigma_hat_sq_break'][1],
                                  results['eta_A1_agn'][1], results['eta_A2_agn'][1], 
                                  results['eta_break_agn'][1],
                                  results['beta_agn'][1],
                                  df_agn['log_sigma_hat_UV'].values,
                                  df_agn['log_sigma_hat_UV_err'].values,
                                  df_agn['log_tau_UV_RF_err'].values)**2)
    # Now take the median again:
    mu_pred_median = np.percentile(mu_pred, 50, axis=0)

    #--- Residuals ---
    mu_interp = np.interp(df_agn["z"], z_grid, mu_model_median)
    residuals = mu_pred_median - mu_interp

    # --- Binning ---
    bins = np.linspace(df_agn["z"].min(), df_agn["z"].max(), 27)
    bin_indices = np.digitize(df_agn["z"], bins)
    # Compute counts per bin
    bin_counts = [np.sum(bin_indices == i) for i in range(1, len(bins))]
    # Mask: True if bin has at least 3 items
    valid_mask = np.array(bin_counts) >= 5

    binned_mu_pred_mean = [np.mean(mu_pred_median[bin_indices == i]) for i in range(1, len(bins))]
    binned_mu_pred_std = [np.std(mu_pred_median[bin_indices == i])/np.sqrt(len(mu_pred_median[bin_indices == i])) for i in range(1, len(bins))]
    binned_z = [np.median(df_agn["z"][bin_indices == i]) for i in range(1, len(bins))]

    # Apply mask
    binned_mu_pred_mean = np.array(binned_mu_pred_mean)[valid_mask]
    binned_mu_pred_std = np.array(binned_mu_pred_std)[valid_mask]
    binned_z = np.array(binned_z)[valid_mask]

    if show_uncorrected:
        # --- Also compute uncorrected mu_pred for plotting ---
        mu_pred_uncorrected = np.array([
            df_agn['apparent_mag_i'] - K_corr(df_agn['z']) - (
                (M_model_agn(s[param_indices['M0_sn']] + s[param_indices['delta_M0_agn']],
                             s[param_indices['log_sigma_hat_sq_break']],
                             s[param_indices['eta_A1_agn']], s[param_indices['eta_A2_agn']], 
                             s[param_indices['eta_break_agn']], 
                            s[param_indices['beta_agn']],
                            df_agn['log_sigma_hat_UV'], df_agn['log_tau_UV_RF'])) - K_corr(2.0))
            for s in flat_samples
        ]) 
        mu_pred_uncorrected_median = np.percentile(mu_pred_uncorrected, 50, axis=0)
        # --- Binning ---
        bin_indices = np.digitize(df_agn["z"], bins)
        binned_mu_pred_uncorrected_mean = [np.mean(mu_pred_uncorrected_median[bin_indices == i]) for i in range(1, len(bins))]
        binned_mu_pred_uncorrected_std = [np.std(mu_pred_uncorrected_median[bin_indices == i])/np.sqrt(len(mu_pred_uncorrected_median[bin_indices == i])) for i in range(1, len(bins))]
        binned_z_uncorrected = [np.median(df_agn["z"][bin_indices == i]) for i in range(1, len(bins))]
        # Apply mask to remove bins with less than 3 objects
        binned_mu_pred_uncorrected_mean = np.array(binned_mu_pred_uncorrected_mean)[valid_mask]
        binned_mu_pred_uncorrected_std = np.array(binned_mu_pred_uncorrected_std)[valid_mask]
        binned_z_uncorrected = np.array(binned_z_uncorrected)[valid_mask]


    # --- Plot setup ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    inset_ax = inset_axes(ax, width="40%", height="40%", loc="lower right", borderpad=1.5)

    # --- Inset plot ---
    inset_ax.scatter(df_agn["z"], mu_pred_median, s=2, label="AGN", alpha=0.5, color="black",zorder=1)
    inset_ax.scatter(df_pantheon["zHD"], df_pantheon["MU_SH0ES"], color="dodgerblue", s=2, label="SN Ia", alpha=0.5, zorder=-10)
    inset_ax.plot(z_grid, mu_model_median, alpha=0.9, color="purple", zorder=10, lw=0.5, label="ΛCDM Model")
    inset_ax.fill_between(z_grid, mu_model_16th, mu_model_84th, color="purple", alpha=0.9, zorder=9)
    inset_ax.set_xscale('log')
    inset_ax.set_xlim(0.02, 4.2)
    inset_ax.set_ylim(32, 51)
    inset_ax.set_xlabel(r"$z$", fontsize=12, labelpad=-10)
    inset_ax.set_ylabel(r"$\mu$ (mag)", fontsize=12)
    inset_ax.tick_params(axis='both', which='major', labelsize=10)
#    inset_ax.legend(frameon=False, fontsize=6)

    # --- Main Hubble diagram ---
    # Original AGNs
    ax.errorbar(df_agn["z"], mu_pred_median, yerr=mu_pred_std, fmt='o', linestyle='none',
                markersize=2, alpha=0.2, color='k', lw=1.5, zorder=-7, label="AGN")
    
    # Binned AGNs
    ax.errorbar(binned_z, binned_mu_pred_mean, yerr=binned_mu_pred_std, label="Binned AGN",
                fmt='o', markersize=4, capsize=3, lw=1.5,alpha=0.9, color="red", zorder=-1)

    # Uncorrected AGNs
    if show_uncorrected:
        inset_ax.scatter(df_agn["z"], mu_pred_uncorrected_median, s=2, label="AGN (uncorrected)", alpha=0.5, color="green",zorder=-16)
        ax.errorbar(df_agn["z"], mu_pred_uncorrected_median, yerr=mu_pred_std, fmt='o', linestyle='none',
                    markersize=2, alpha=0.2, color='green', lw=1.5, zorder=-9, label="AGN (uncorrected)")
        ax.errorbar(binned_z, binned_mu_pred_uncorrected_mean, yerr=binned_mu_pred_uncorrected_std, label="Binned AGN (uncorrected)", 
                    fmt='o', markerfacecolor='none', markeredgecolor='red', markersize=4, capsize=3, lw=1.5, alpha=0.9, color="red", zorder=-6)
    
    if show_true:
        ax.scatter(df_agn['z'], df_agn['apparent_mag_i'] - df_agn['M_i'], alpha=0.7, edgecolor='k')
    
    # SNIa points
    ax.errorbar(df_pantheon["zHD"], df_pantheon["MU_SH0ES"], yerr=df_pantheon["MU_SH0ES_ERR_DIAG"], 
                fmt='s', markersize=3, color="dodgerblue", linestyle='none', lw=1, label="SN Ia", alpha=0.7, zorder=-9)

    # Cosmo model band
    ax.plot(z_grid, mu_model_median, alpha=0.9, color="m", zorder=-5, lw=2, label=label)
    ax.fill_between(z_grid, mu_model_16th, mu_model_84th, color="purple", alpha=0.9, zorder=-5)

    ax.set_ylabel(r"$\mu$ (mag)")
    ax.set_xlabel(r"$z$")
    ax.set_xlim(-0.2, np.round(df_agn['z'].max())+0.1)
    ax.set_ylim(26, 51)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.3, 0.05), fontsize=16)

    # Ticks styling
    for axi in [ax, inset_ax]:
        axi.minorticks_on()
        axi.tick_params(axis='both', which='minor', direction='in', length=4, top=True, right=True, width=2)
        axi.tick_params(axis='both', which='major', direction='in', length=8, top=True, right=True)

    fig.tight_layout()
    os.makedirs("plots/hubble", exist_ok=True)
    show_uncorrected_label = "_uncorrected" if show_uncorrected else ""
    #plt.savefig(f"plots/hubble/hubble_diagram_{cosmo_model}{show_uncorrected_label}.pdf", dpi=300)
    plt.savefig(f"plots/hubble/hubble_diagram_{cosmo_model}{show_uncorrected_label}.png")

    ax.set_yscale('log')
    ax.set_ylim(39, 50)
    plt.savefig(f"plots/hubble/hubble_diagram_{cosmo_model}{show_uncorrected_label}_ylog.png")

    if show:
        plt.show()
    plt.close()
    return residuals, mu_pred_std

def plot_predicted_vs_actual_Mi(flat_samples, df_agn, cosmo_model, show=False):
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    results = {key: np.percentile(flat_samples[:, i], [16, 50, 84]) for i, key in enumerate(model_labels)}

    M_i_pred = M_model_agn(
        results['M0_sn'][1] + results['delta_M0_agn'][1], 
        results['log_sigma_hat_sq_break'][1],
        results['eta_A1_agn'][1], results['eta_A2_agn'][1], 
        results['eta_break_agn'][1],
        results['beta_agn'][1], 
        df_agn['log_sigma_hat_UV'].values,
        df_agn['log_tau_UV_RF'].values
    ) + K_corr(2.0) # TODO: check this


    # Calculate prediction errors
    M_i_pred_err = np.sqrt(
        M_model_agn_err(
            results['M0_sn'][1] + results['delta_M0_agn'][1],
            results['log_sigma_hat_sq_break'][1],
            results['eta_A1_agn'][1], results['eta_A2_agn'][1], 
            results['eta_break_agn'][1],
            results['beta_agn'][1], 
            df_agn['log_sigma_hat_UV'].values, 
            df_agn['log_sigma_hat_UV_err'].values, 
            df_agn['log_tau_UV_RF_err'].values
        )**2
    )

    # Bin and color by redshift
    #z_bins = np.linspace(df_agn['z'].min(), df_agn['z'].max(), 10)  # Define redshift bins
    z_bins = np.linspace(0, 4, 10)  # Define redshift bins
    z_bin_indices = np.digitize(df_agn['z'], bins=z_bins)  # Assign each redshift to a bin

    # Define the number of redshift bins and their labels
    num_bins = len(z_bins) - 1
    bin_labels = [f"{z_bins[i]:.1f} < z < {z_bins[i+1]:.1f}" for i in range(num_bins)]

    # Calculate the number of rows and columns for the grid
    num_cols = 3  # Number of columns
    num_rows = math.ceil(num_bins / num_cols)  # Number of rows

    # Create subplots with reduced height
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), sharey=True, sharex=True)
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

    # Loop through each redshift bin and plot
    for i, ax in enumerate(axes):
        ax.set_xlim(df_agn['M_i'].min(), df_agn['M_i'].max())
        ax.set_ylim(M_i_pred.min(), M_i_pred.max())
        #ax.set_xlim(-29.8, -21.2)
        #ax.set_ylim(-29.8, -21.2)

        if i < num_bins:
            # Filter data for the current redshift bin
            bin_mask = z_bin_indices == (i + 1)
            actual_M_i = df_agn['M_i']
            predicted_M_i_bin = M_i_pred[bin_mask]
            predicted_M_i_err_bin = M_i_pred_err[bin_mask]
            M_i_axis = np.linspace(actual_M_i.min(), actual_M_i.max(), 100)
            ax.plot(M_i_axis, M_i_axis, color='m', alpha=0.7, label='y = x (Perfect Prediction)', lw=3, linestyle='--')
            # Scatter plot for the current bin with error bars
            scatter = ax.errorbar(
                actual_M_i[bin_mask], predicted_M_i_bin, xerr=0.25, yerr=predicted_M_i_err_bin, 
                fmt='o', markerfacecolor='k', markeredgecolor='k', alpha=0.4, lw=1.5, capsize=3, capthick=1, color='k'
            )
            # Invert x and y axes
            ax.invert_xaxis()
            ax.invert_yaxis()
            # Annotate bin label
            ax.annotate(bin_labels[i], xy=(0.05, 0.95), xycoords='axes fraction', 
                    fontsize=18, color='k', alpha=0.7, ha='left', va='top')  # Annotate bin label in grey
            # Annotate number of objects in the bin
            n_in_bin = np.sum(bin_mask)
            ax.annotate(f"N = {n_in_bin}", xy=(0.95, 0.05), xycoords='axes fraction',
                        fontsize=14, color='gray', ha='right', va='bottom')
            if i >= (num_rows - 1) * num_cols:  # Add xlabel only for the bottom row
                ax.set_xlabel('Actual $M_i$')
            if i % num_cols == 0:  # Add ylabel only for the first column
                ax.set_ylabel('Predicted $M_i$')
        else:
            # Hide unused subplots
            ax.axis('off')

    # Adjust layout to remove whitespace
    plt.subplots_adjust(wspace=0, hspace=0)

    os.makedirs("plots/hubble", exist_ok=True)
    plt.savefig(f"plots/hubble/predicted_vs_actual_Mi_{cosmo_model}.png", dpi=300)
    #plt.savefig(f"plots/hubble/predicted_vs_actual_Mi_{cosmo_model}.pdf", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_completeness_vs_mag_at_redshifts(p_detect, mag_centers, z_centers, 
                                          redshifts=[0.5, 1.0, 2.0, 3.0, 4.0], show=False):
    """
    Plot p(I=1 | m, z) vs apparent magnitude for several fixed redshifts.

    Parameters:
        p_detect : callable
            A function p_detect(mag, z) returning the completeness probability.
        mag_centers : ndarray
            1D array of magnitude bin centers used for evaluation.
        z_centers : ndarray
            1D array of redshift bin centers (not used in plotting directly).
        redshifts : list of floats
            Redshift values at which to evaluate the completeness curves.
    """
    mag_eval = np.linspace(np.min(mag_centers), np.max(mag_centers), 1000)

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    # Choose a colormap
    cmap = cm.get_cmap('viridis', len(redshifts))
    norm = mcolors.Normalize(vmin=min(redshifts), vmax=max(redshifts))

    # Define a list of line styles to cycle through
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
    style_cycle = iter(line_styles)

    plt.figure(figsize=(7, 5))
    for i, z in enumerate(redshifts):
        p_vals = p_detect(mag_eval, np.full_like(mag_eval, z))
        color = cmap(norm(z))
        plt.plot(mag_eval, p_vals, label=fr"$z = {z}$", color=color, linestyle=line_styles[i % len(line_styles)])

    #sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    #sm.set_array([])
    #plt.colorbar(sm, label="Redshift", ticks=redshifts)

    plt.xlabel(r"$m$ ($i$ mag)")
    plt.ylabel(r"$p(I{=}1|m, z)$")
    plt.legend(fontsize=16, loc="upper right", frameon=False)
    #plt.ylim(0, .02)
    plt.xlim(17, 25)
    plt.grid(False)
    plt.tight_layout()
    if show:
        plt.show()
    os.makedirs("plots/hubble", exist_ok=True)
    plt.savefig("plots/hubble/completeness_vs_mag_at_redshifts.png", dpi=300)
    #plt.savefig("plots/hubble/completeness_vs_mag_at_redshifts.pdf", dpi=300)
    plt.close()



def plot_predicted_sigma_hat_vs_luminosity(sampler, df_agn, cosmo_model, show=False, flat_samples=None):
    # Load samples
    if flat_samples is None:
        flat_samples = sampler.get_chain(flat=True)
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    param_indices = {name: model_labels.index(name) for name in model_labels}

    # Extract arrays of model parameters
    M0_sn_samples = flat_samples[:, param_indices['M0_sn']]
    delta_M_agn_samples = flat_samples[:, param_indices['delta_M_agn']]
    alpha_samples = flat_samples[:, param_indices['alpha_agn']]
    beta_samples = flat_samples[:, param_indices['beta_agn']]

    # Define grid in log L_bol
    log_lbol_grid = np.linspace(43, 49, 200)
    lbol_grid = 10 ** log_lbol_grid

    # Precompute terms
    # slopes = -2.5 / alpha_samples
    # intercepts = (90 - M0_samples) / alpha_samples + 2 * log_sigma_hat_pivot

    slopes = -2.5 / alpha_samples
    intercepts = 1/alpha_samples * (90 - M0_sn_samples + M0_agn_offset) #+ 2 * log_sigma_hat_pivot -2 - beta_samples * (df_agn['log_tau_UV_RF'].mean() - log_tau_UV_RF_pivot))


    #print("Intercepts:", intercepts)

    # Evaluate log_sigma_hat_sq across all posterior lines
    ys = np.outer(slopes, log_lbol_grid) + intercepts[:, None]  # shape: (n_samples, n_grid)

    # Compute 1σ band and median in log space
    y_low, y_high = np.percentile(ys, [16, 84], axis=0)
    y_median = np.median(ys, axis=0)

    # Convert to linear space
    y_low_lin = 10 ** y_low
    y_high_lin = 10 ** y_high
    y_median_lin = 10 ** y_median

    # --- Observed AGN data ---
    log_lbol = df_agn['log_lbol'].values
    log_sigma_hat_sq = 2 * df_agn['log_sigma_hat_UV'].values

    lbol = 10 ** log_lbol
    sigma_hat_sq = 10 ** log_sigma_hat_sq

    sigma_hat_sq_err = np.log(10) * sigma_hat_sq * 2 * df_agn['log_sigma_hat_UV_err']
    lbol_err = np.log(10) * lbol * df_agn['log_lbol_err']

    # --- Plot ---
    plt.figure(figsize=(8, 6))

    # Posterior 1σ band and median line
    plt.fill_between(
        lbol_grid,
        y_low_lin,
        y_high_lin,
        color='m',
        alpha=0.4)
    plt.plot(lbol_grid, y_median_lin, color='m', lw=2, label='Best fit model')

    plt.errorbar(
        lbol,
        sigma_hat_sq,
        yerr=sigma_hat_sq_err,
        xerr=lbol_err,
        fmt='o',
        linestyle='none',
        color='k',
        alpha=0.2,
        markersize=4, 
        lw=1,
        zorder=2,
        label='AGN'
    )

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$L_{\mathrm{bol}}$ (erg $\mathrm{s}^{-1}$)')
    plt.ylabel(r'$\hat{\sigma}_{\mathrm{UV}}^2$ $(\mathrm{mag}^2\ \mathrm{day}^{-1})$')
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.xlim(2e43, 9e47)
    plt.ylim(0.4e-3, 0.4e1)
    os.makedirs("plots/hubble", exist_ok=True)
    plt.savefig(f"plots/hubble/predicted_sigma_hat_sq_{cosmo_model}.png", dpi=300)
    #plt.savefig(f"plots/hubble/predicted_sigma_hat_sq_{cosmo_model}.pdf", dpi=300)
    if show:
        plt.show()
    plt.close()

    return slopes, intercepts


def plot_predicted_sigma_hat_vs_luminosity_pl(sampler, df_agn, cosmo_model, show=False):
    # Load samples
    if hasattr(sampler, "results"):  # dynesty
        results = sampler.results
        samples = results.samples
        weights = np.exp(results.logwt - results.logz[-1])
        samples = resample_equal(samples, weights)
    else:
        samples = sampler.get_chain(flat=True, thin=200)
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    param_indices = {name: model_labels.index(name) for name in model_labels}

    # Define grid in log L_bol
    log_lbol_grid = np.linspace(43, 49, 200)
    lbol_grid = 10 ** log_lbol_grid


    predicted_M_i_samples = []
    zipped_samples = []
    for log_sigma_hat_UV, log_tau_UV_RF in tqdm(zip(df_agn['log_sigma_hat_UV'], df_agn['log_tau_UV_RF']), total=len(df_agn), desc="Predicting M_i"):
        predicted_M_i_sample = M_model_agn(
                samples[:, param_indices['M0_sn']] + samples[:, param_indices['delta_M_agn']], 
                samples[:, param_indices['log_sigma_hat_sq_break']],
                samples[:, param_indices['eta_A1_agn']], samples[:, param_indices['eta_A2_agn']], 
                samples[:, param_indices['eta_break_agn']],
                samples[:, param_indices['beta_agn']],
                log_sigma_hat_UV, log_tau_UV_RF
            )
        predicted_M_i_samples.extend(predicted_M_i_sample)
        zipped_samples.append((log_sigma_hat_UV, log_tau_UV_RF, predicted_M_i_sample))
    predicted_M_i_samples = np.array(predicted_M_i_samples)

    # Convert predicted M_i to luminosity
    predicted_log_lbol_samples = (90 - predicted_M_i_samples) / 2.5

    # from scipy.optimize import brentq
    
    # log_sigma_hat_sq_samples = []
    # for predicted_M_i_sample in tqdm(range(len()), desc="Computing log_sigma_hat_sq samples"):
    #     log_sigma_hat_brentq = brentq(lambda log_sigma_hat: M_model_agn(
    #         M0_sn_samples, delta_M_agn_samples, alpha_samples, beta_samples,
    #         eta_A1_samples, eta_A2_samples, log_sigma_hat_break_samples,
    #         log_sigma_hat, mean_log_tau_UV_RF) - predicted_M_i_sample, -3.0, 0)
    #     log_sigma_hat_sq_samples.append(2 * log_sigma_hat_brentq)

    # log_sigma_hat_sq_samples = np.array(log_sigma_hat_sq_samples)

    # # Compute percentile bands
    # y_low, y_high = np.percentile(log_sigma_hat_sq_samples, [16, 84], axis=0)
    # y_median = np.median(log_sigma_hat_sq_samples, axis=0)

    # # Convert to linear space
    # y_low_lin = 10 ** y_low
    # y_high_lin = 10 ** y_high
    # y_median_lin = 10 ** y_median

    # Observed AGN data
    log_lbol = df_agn['log_lbol'].values
    log_sigma_hat_sq = 2 * df_agn['log_sigma_hat_UV'].values

    lbol = 10 ** log_lbol
    sigma_hat_sq = 10 ** log_sigma_hat_sq

    sigma_hat_sq_err = np.log(10) * sigma_hat_sq * 2 * df_agn['log_sigma_hat_UV_err']
    lbol_err = np.log(10) * lbol * df_agn['log_lbol_err']

    # Plotting
    plt.figure(figsize=(8, 6))

    # Plot the posterior band
    # plt.fill_between(lbol_grid, y_low_lin, y_high_lin, color='m', alpha=0.4)
    # plt.plot(lbol_grid, y_median_lin, color='m', lw=2, label='Model')

    # Observed data
    plt.errorbar(
        lbol,
        sigma_hat_sq,
        yerr=sigma_hat_sq_err,
        xerr=lbol_err,
        fmt='o', linestyle='none', color='k', alpha=0.2, markersize=4, lw=1, label='AGN'
    )

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$L_{\mathrm{bol}}$ (erg $\mathrm{s}^{-1}$)')
    plt.ylabel(r'$\hat{\sigma}_{\mathrm{UV}}^2$ $(\mathrm{mag}^2\ \mathrm{day}^{-1})$')
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.xlim(2e43, 9e47)
    plt.ylim(0.4e-3, 0.4e1)

    os.makedirs("plots/hubble", exist_ok=True)
    plt.savefig(f"plots/hubble/predicted_sigma_hat_sq_{cosmo_model}.png", dpi=300)
    #plt.savefig(f"plots/hubble/predicted_sigma_hat_sq_{cosmo_model}.pdf", dpi=300)

    if show:
        plt.show()
    plt.close()

    #return log_sigma_hat_sq_samples

def plot_Mi_vs_log_sigma_hat_sq(samples, df_agn, cosmo_model, show=False):

    # Observed values (already in log base 10)
    log_sigma_hat_sq = 2 * df_agn['log_sigma_hat_UV'].values
    log_sigma_hat_sq_err = 2 * df_agn['log_sigma_hat_UV_err'].values
    M_i = df_agn['M_i'].values

    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    param_indices = {name: model_labels.index(name) for name in model_labels}
    results = {key: np.percentile(samples[:, i], [16, 50, 84]) for i, key in enumerate(model_labels)}

    # Fit model: reparametrize alpha to log_alpha for fitting
    x_fit = np.linspace(-2.5, 0, 200)
    y_fit = M_model_agn(
        results['M0_sn'][1] + results['delta_M0_agn'][1],
        results['log_sigma_hat_sq_break'][1],
        results['eta_A1_agn'][1], results['eta_A2_agn'][1], 
        results['eta_break_agn'][1],
        results['beta_agn'][1],
        x_fit/2, df_agn['log_tau_UV_RF'].mean()
    ) 
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(2*df_agn['log_sigma_hat_UV'], df_agn['M_i'], alpha=0.7, edgecolor='k', zorder=-2)
    ax.plot(x_fit, y_fit, color='red', zorder=-1)

    ax.set_xlabel(r'$\log_{10}(\hat{\sigma}_{\mathrm{UV}}^2)$ (mag$^2$ day$^{-1}$)')
    ax.set_ylabel(r'$M_i$')
    #ax.invert_yaxis()
    #ax.grid(True, which='both', ls='--', lw=0.3)
    ax.set_xlim(log_sigma_hat_sq.min() - 0.1, log_sigma_hat_sq.max() + 0.1)
    ax.legend(fontsize=12)
    fig.tight_layout()
    if show:
        plt.show()

    os.makedirs("plots/hubble", exist_ok=True)
    fig.savefig("plots/hubble/observed_Mi_vs_log_sigma_hat_sq_fit.png", dpi=300)
    #fig.savefig("plots/hubble/observed_Mi_vs_log_sigma_hat_sq_fit.pdf", dpi=300)

    return fig, ax

def plot_Mi_vs_sigma_hat_sq_linearx(df_agn, show=False):
    # Base-10 log quantities
    log_sigma_hat_sq = 2 * df_agn['log_sigma_hat_UV'].values
    log_sigma_hat_sq_err = 2 * df_agn['log_sigma_hat_UV_err'].values
    M_i = df_agn['M_i'].values
    M_i_err = df_agn['M_i_err'].values if 'M_i_err' in df_agn.columns else np.zeros_like(M_i)

    # Fit model: parameterize with log_alpha
    def fit_model(logx, log_alpha, d1, d2):
        return (
            log_alpha
            + d1 * logx
            - (d2 - d1) * np.log10(1 + 10**(logx))  # ds = 1.0
        )

    # Initial guess
    p0 = [np.median(M_i), -1.0, -0.2]

    # Fit the curve
    popt, pcov = curve_fit(
        fit_model,
        log_sigma_hat_sq,
        M_i,
        sigma=M_i_err,
        absolute_sigma=True,
        p0=p0,
        maxfev=10000
    )

    # Fit curve in log-space, then convert to linear x
    x_fit_log = np.linspace(log_sigma_hat_sq.min(), log_sigma_hat_sq.max(), 300)
    y_fit = fit_model(x_fit_log, *popt)
    x_fit_lin = 10**x_fit_log

    # Plot in linear scale
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        10**log_sigma_hat_sq,  # linear x
        M_i,
        xerr=(np.log(10) * 10**log_sigma_hat_sq * log_sigma_hat_sq_err),  # propagate log err
        yerr=M_i_err,
        fmt='o', color='k', alpha=0.3, markersize=4, label='AGN'
    )
    #plt.plot(x_fit_lin, y_fit, color='crimson', lw=2, label='Broken PL Fit')
    plt.axvline(x=(10**(-0.638))**2, color='blue', linestyle='--', lw=2, label=r'pivot')
    plt.xscale('log')
    plt.xlabel(r'$\hat{\sigma}_{\mathrm{UV}}^2$ (mag$^2$ day$^{-1}$)')
    plt.ylabel(r'$M_i$')
    #plt.gca().invert_yaxis()
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.grid(True, which='both', ls='--', lw=0.3)
    plt.xlim(1e-3, 1e1)

    os.makedirs("plots/hubble", exist_ok=True)
    plt.savefig("plots/hubble/observed_Mi_vs_sigma_hat_sq_fit_linearx.png", dpi=300)
    #plt.savefig("plots/hubble/observed_Mi_vs_sigma_hat_sq_fit_linearx.pdf", dpi=300)

    if show:
        plt.show()
    plt.close()

    # Return parameters
    log_alpha, d1, d2 = popt
    return {"alpha": 10**log_alpha, "d1": d1, "d2": d2, "log_alpha": log_alpha}, pcov


def plot_completeness_diagnostics(df_agn, completeness2d, mag_centers, z_centers, output_prefix="plots/hubble/completeness_diag"):
    """
    Plot and save diagnostic figures for completeness:
    - Empirical P(detect) vs. redshift
    - 2D completeness map
    - Inverse completeness correction map (log10[1 / P])
    
    Parameters:
        df_agn           : DataFrame with 'z' and 'apparent_mag_i'
        completeness2d   : callable completeness function (mag, z)
        mag_centers      : 1D array of magnitude bin centers
        z_centers        : 1D array of redshift bin centers
        output_prefix    : filename prefix for saved figures (default: "completeness_diag")
    """
    z = df_agn['z'].values
    m = df_agn['apparent_mag_i'].values
    completeness = completeness2d(m, z)

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    # --- 1. P(detect) vs z ---
    plt.figure(figsize=(5, 5))
    plt.scatter(z, completeness, s=3, alpha=1)
    plt.xlabel('z')
    plt.ylabel('P(detect)')
    plt.title('Empirical Completeness')
    plt.grid(True)
    plt.tight_layout()
    plt.yscale('log')
    plt.savefig(f"{output_prefix}_vs_redshift.png", dpi=200)
    plt.close()

    # --- 2. 2D completeness map ---
    M_grid, Z_grid = np.meshgrid(mag_centers, z_centers, indexing='ij')
    P_grid = completeness2d(M_grid, Z_grid)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        P_grid,
        extent=[z_centers[0], z_centers[-1], mag_centers[-1], mag_centers[0]],
        aspect='auto',
        cmap='viridis',
        interpolation='nearest'
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$P(\mathrm{detect} \mid m, z)$")
    ax.set_xlabel(r"Redshift $z$")
    ax.set_ylabel(r"Apparent Magnitude $m$")
    ax.set_title("2D Completeness Map")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_2d_map.png", dpi=200)
    plt.close()

    # --- 3. log10(1/P) correction map ---
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        np.log10(1.0 / np.clip(P_grid, 1e-3, 1.0)),
        extent=[z_centers[0], z_centers[-1], mag_centers[-1], mag_centers[0]],
        origin='lower',
        aspect='auto',
        cmap='viridis',
        interpolation='nearest'
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Effective Correction log10(1 / P_detect)")
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Apparent Magnitude m")
    ax.set_title("Inverse Completeness Correction")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_inverse_correction.png", dpi=200)
    plt.close()
