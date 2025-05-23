import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from tqdm import tqdm
import math
import corner
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.cosmology import FlatwCDM, Flatw0waCDM
import matplotlib.pyplot as plt
import os

from hubble_fit import log_likelihood
from hubble_utils import get_completeness_function_2d, compute_delta_mag_bias_2d_zbins
from hubble_model import *
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import interp1d

def plot_corner(sampler, only_sna=False, cosmo_model='Flatw0waCDM'):
    # Select cosmological parameters based on model
    if cosmo_model == 'FlatwCDM':
        cosmo_params = ['H0', 'Om0', 'w0']
    elif cosmo_model == 'Flatw0waCDM':
        cosmo_params = ['H0', 'Om0', 'w0', 'wa']
    else:
        raise ValueError("cosmo_model must be 'FlatwCDM' or 'Flatw0waCDM'")
    # Model parameters: AGN correlation + SN calibration + cosmology
    priors, model_labels = get_model_params(cosmo_model)
    flat_samples = sampler.get_chain(flat=True)
    
    fig = corner.corner(
        flat_samples,
        labels=model_labels,
        truths=None,
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12}  # Reduce title font size
    )

    os.makedirs("plots/hubble", exist_ok=True)
    if only_sna:
        fig.suptitle("SNIa only", fontsize=16)
        plt.savefig(f"plots/hubble/posterior_{cosmo_model}_sna.png", dpi=200)
    else:
        fig.suptitle("SNIa + AGN", fontsize=28)
        plt.savefig(f"plots/hubble/posterior_{cosmo_model}_agn.png", dpi=200)
    #plt.show()
    plt.close()

def plot_cosmo_corner(sampler_sna, sampler_agn, cosmo_model='Flatw0waCDM', show=False):
# === Parameter setup ===
    if cosmo_model == 'FlatwCDM':
        param_names = ["H0", "Om0", "w0"]
        labels = [r"$H_0$", r"$\Omega_M$", r"$w_0$"]
    elif cosmo_model == 'Flatw0waCDM':
        param_names = ["H0", "Om0", "w0", "wa"]
        labels = [r"$H_0$", r"$\Omega_M$", r"$w_0$", r"$w_a$"]
    priors, model_labels = get_model_params(cosmo_model)
    n_params = len(labels)
    param_indices = [list(priors.keys()).index(p) for p in param_names]

    # === Get flattened MCMC chains ===
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
    fig.legend(handles=legend_elements, loc="upper right", fontsize=26, frameon=False, markerscale=1.5)

    # === Layout & Save ===
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)

    os.makedirs("plots/hubble", exist_ok=True)
    fig.savefig(f"plots/hubble/corner_kde_{cosmo_model}.pdf", bbox_inches="tight", transparent=True)
    fig.savefig(f"plots/hubble/corner_kde_{cosmo_model}.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_hubble(sampler, df_agn, df_pantheon, cosmo_model, show=False, completeness=True):
    """Plot Hubble diagram + residuals, classic Pantheon+ style."""
    # Define cosmological parameter labels
    if cosmo_model == 'FlatwCDM':
        label = r"Flat$w$CDM Model"
    elif cosmo_model == 'Flatw0waCDM':
        label = r"Flat$w_0w_a$CDM Model"
    else:
        raise ValueError("Invalid cosmology model.")
    
    flat_samples = sampler.get_chain(thin=15, flat=True)
    
    z_grid = np.linspace(0.0001, df_agn['z'].max(), len(df_agn))

    # Build model_labels and get indices for cosmological parameters
    priors, model_labels = get_model_params(cosmo_model)
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

    results = {key: np.percentile(flat_samples[:, i], [16, 50, 84]) for i, key in enumerate(model_labels)}

    # --- Cosmology model ---    
    mu_model_median = np.percentile(mu_models, 50, axis=0)
    mu_model_16th = np.percentile(mu_models, 16, axis=0)
    mu_model_84th = np.percentile(mu_models, 84, axis=0)

    # Correct the apparent magnitude first
    completeness2d, mag_centers, z_centers, dm, dz = get_completeness_function_2d(df_agn)
    delta_mag_arr, delta_mag_arr_errs = compute_delta_mag_bias_2d_zbins(df_agn, completeness2d, mag_centers, z_centers, dm)
    if completeness:
        corrected_apparent_mag = df_agn['apparent_mag_i'] - delta_mag_arr
    else:
        corrected_apparent_mag = df_agn['apparent_mag_i']

    # delta_mag_fit = -0.11 * df_agn['z'].values + 0.11
    # corrected_apparent_mag = df_agn['apparent_mag_i'] - delta_mag_fit
    
    # Then re-compute the distance modulus
    mu_pred = np.array([
        corrected_apparent_mag - (K_corr(df_agn['z']) - K_corr(2)) -
            M_model_agn(s[param_indices['M0_agn']], s[param_indices['alpha_agn']], 
                        df_agn['log_sigma_hat_UV'])
        for s in flat_samples
    ])

    # Now take the median again:
    mu_pred_median = np.percentile(mu_pred, 50, axis=0)

    # --- Also compute uncorrected mu_pred for plotting ---
    mu_pred_uncorrected = np.array([
        df_agn['apparent_mag_i'] - K_corr(df_agn['z']) - (
            M_model_agn(s[param_indices['M0_agn']], s[param_indices['alpha_agn']], 
                        df_agn['log_sigma_hat_UV']) - K_corr(2))
        for s in flat_samples
    ])
    mu_pred_uncorrected_median = np.percentile(mu_pred_uncorrected, 50, axis=0)

    mu_pred_16th = np.percentile(mu_pred, 16, axis=0)
    mu_pred_84th = np.percentile(mu_pred, 84, axis=0)
    mu_pred_std = np.sqrt(df_agn['apparent_mag_i_err']**2 +
                 (-2.5 * 0.3 * np.log10(1 + df_agn["z"]))**2 +
                 (0.055 * df_agn["z"])**2 +
                (results["alpha_agn"][1] * 2*df_agn['log_sigma_hat_UV_err']))**2

    #--- Residuals ---
    mu_interp = np.interp(df_agn["z"], z_grid, mu_model_median)
    residuals = mu_pred_median - mu_interp

    # --- Binning ---
    bins = np.linspace(df_agn["z"].min(), df_agn["z"].max(), 25)
    bin_indices = np.digitize(df_agn["z"], bins)
    binned_mu_pred_median = [np.mean(mu_pred_median[bin_indices == i]) for i in range(1, len(bins))]
    binned_mu_pred_std = [np.std(mu_pred_median[bin_indices == i])/np.sqrt(len(mu_pred_median[bin_indices == i])) for i in range(1, len(bins))]
    binned_z = [np.median(df_agn["z"][bin_indices == i]) for i in range(1, len(bins))]


    # --- Plot setup ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    inset_ax = inset_axes(ax, width="40%", height="40%", loc="lower right", borderpad=1.5)

    # --- Inset plot ---
    inset_ax.scatter(df_agn["z"], mu_pred_median, s=2, label="AGN", alpha=0.5, color="black",zorder=-9)
    inset_ax.scatter(df_pantheon["zHD"], df_pantheon["MU_SH0ES"], color="dodgerblue", s=2, label="SN Ia", alpha=0.5, zorder=-10)
    inset_ax.plot(z_grid, mu_model_median, alpha=0.9, color="purple", zorder=-9, lw=0.5, label="ΛCDM Model")
    inset_ax.fill_between(z_grid, mu_model_16th, mu_model_84th, color="purple", alpha=0.9)
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
                markersize=2, alpha=0.2, color='k', lw=1.5, zorder=-10, label="AGN")
    
    ax.errorbar(df_agn["z"], mu_pred_uncorrected_median, yerr=mu_pred_std, fmt='o', linestyle='none',
                markersize=2, alpha=0.2, color='green', lw=1.5, zorder=-10, label="Uncorrected AGN")
    # Binned AGNs
    ax.errorbar(binned_z, binned_mu_pred_median, yerr=binned_mu_pred_std, label="Binned AGN",
                fmt='o', markersize=4, capsize=3, lw=1.5,alpha=0.9, color="red", zorder=-7)

    # SNIa points
    ax.errorbar(df_pantheon["zHD"], df_pantheon["MU_SH0ES"], yerr=df_pantheon["MU_SH0ES_ERR_DIAG"], 
                fmt='s', markersize=3, color="dodgerblue", linestyle='none', lw=1, label="SN Ia", alpha=0.7, zorder=-8)

    # Cosmo model band
    ax.plot(z_grid, mu_model_median, alpha=0.9, color="m", zorder=-5, lw=2, label=label)
    ax.fill_between(z_grid, mu_model_16th, mu_model_84th, color="purple", alpha=0.9, zorder=-5)

    ax.set_ylabel(r"$\mu$ (mag)")
    ax.set_xlabel(r"$z$")
    ax.set_xlim(-0.2, df_agn['z'].max())
    #ax.set_ylim(26, 51)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.3, 0.05))

    # Ticks styling
    for axi in [ax, inset_ax]:
        axi.minorticks_on()
        axi.tick_params(axis='both', which='minor', direction='in', length=4, top=True, right=True, width=2)
        axi.tick_params(axis='both', which='major', direction='in', length=8, top=True, right=True)

    fig.tight_layout()
    os.makedirs("plots/hubble", exist_ok=True)
    plt.savefig(f"plots/hubble/hubble_diagram_{cosmo_model}.pdf", dpi=300)
    plt.savefig(f"plots/hubble/hubble_diagram_{cosmo_model}.png")
    if show:
        plt.show()
    plt.close()
    return residuals, mu_pred_std

def plot_predicted_vs_actual_Mi(sampler, df_agn, cosmo_model, show=False):
    flat_samples = sampler.get_chain(flat=True, thin=15)
    priors, model_labels = get_model_params(cosmo_model)
    results = {key: np.percentile(flat_samples[:, i], [16, 50, 84]) for i, key in enumerate(model_labels)}

    M_i_pred = M_model_agn(
        results['M0_agn'][1], 
        results['alpha_agn'][1], 
        df_agn['log_sigma_hat_UV']
    ) #+ K_corr(2) # TODO: check this


    # Calculate prediction errors
    M_i_pred_err = np.sqrt(
        #df_agn['M_i_err']**2 +
        (results['alpha_agn'][1] * 2*df_agn["log_sigma_hat_UV_err"])**2
        # (2.5 * 0.3 * np.log10(1 + df_agn['z']))**2 +
        # (0.055 * df_agn['z'])**2 +
        # np.exp(2 * results['log_f'][1])
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
        ax.set_ylim(df_agn['M_i'].min(), df_agn['M_i'].max())

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
                fmt='o', alpha=0.4, lw=1.5, capsize=3, capthick=1, color='k'
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
    plt.savefig(f"plots/hubble/predicted_vs_actual_Mi_{cosmo_model}.pdf", dpi=300)
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
    #plt.xlim(17, 25)
    plt.grid(False)
    plt.tight_layout()
    if show:
        plt.show()
    os.makedirs("plots/hubble", exist_ok=True)
    plt.savefig("plots/hubble/completeness_vs_mag_at_redshifts.png", dpi=300)
    plt.savefig("plots/hubble/completeness_vs_mag_at_redshifts.pdf", dpi=300)
    plt.close()


def plot_predicted_sigma_hat_vs_luminosity(sampler, df_agn, cosmo_model, show=False):

    flat_samples = sampler.get_chain(flat=True, thin=20)
    priors, model_labels = get_model_params(cosmo_model)
    results = {key: np.percentile(flat_samples[:, i], [16, 50, 84]) for i, key in enumerate(model_labels)}

    if cosmo_model == 'FlatwCDM':
        param_names = ["H0", "Om0", "w0"]
        labels = [r"$H_0$", r"$\Omega_M$", r"$w_0$"]
    elif cosmo_model == 'Flatw0waCDM':
        param_names = ["H0", "Om0", "w0", "wa"]
        labels = [r"$H_0$", r"$\Omega_M$", r"$w_0$", r"$w_a$"]
    priors, model_labels = get_model_params(cosmo_model)
    
    plt.figure(figsize=(7, 5))

    param_indices = {name: model_labels.index(name) for name in model_labels}

    sigma_hat_sq = 10**(2 * df_agn['log_sigma_hat_UV'])
    lbol = 10**(df_agn['log_lbol'])


    log_sigma_hat_pred = []
    for s in flat_samples:
        log_sigma_hat_pivot = -2.2
        M0_agn = s[param_indices['M0_agn']]
        alpha_agn = s[param_indices['alpha_agn']]
        M_i_pred = M_model_agn(M0_agn, alpha_agn, df_agn['log_sigma_hat_UV'])
        log_sigma_hat_pred.append((M_i_pred - M0_agn) / (2 * alpha_agn) + log_sigma_hat_pivot)
    log_sigma_hat_pred = np.array(log_sigma_hat_pred)

    log_sigma_hat_sq_pred = 2 * log_sigma_hat_pred
    sigma_hat_sq_pred = 10**log_sigma_hat_sq_pred

    # Work entirely in log-log space for both fitting and plotting
    log_lbol = np.log10(lbol)
    log_sigma_hat_sq = np.log10(sigma_hat_sq)
    log_sigma_hat_sq_pred = np.log10(sigma_hat_sq_pred)
    log_sigma_hat_sq_pred_median = np.median(log_sigma_hat_sq_pred, axis=0)

    # Fit a polynomial (degree 1) in log-log space
    coeffs = np.polyfit(log_lbol, log_sigma_hat_sq_pred_median, 1)
    fit_line = np.poly1d(coeffs)

    # Evaluate fit for plotting
    log_lbol_fit = np.linspace(log_lbol.min(), log_lbol.max(), 200)
    log_sigma_hat_sq_fit = fit_line(log_lbol_fit)

    # Calculate a single std value (in log space) between predicted and fit
    log_sigma_hat_sq_pred_fit = fit_line(log_lbol)
    log_sigma_hat_sq_fit_std = np.std(log_sigma_hat_sq_pred_median - log_sigma_hat_sq_pred_fit)


    plt.plot(log_lbol_fit, log_sigma_hat_sq_fit, color='purple', lw=2, label='Fit (median)')
    plt.fill_between(
        log_lbol_fit,
        log_sigma_hat_sq_fit - log_sigma_hat_sq_fit_std,
        log_sigma_hat_sq_fit + log_sigma_hat_sq_fit_std,
        color='purple', alpha=0.3, label='Fit ± std'
    )

    z_bins = np.linspace(z.min(), z.max(), 4 + 1)
    z_bin_indices = np.digitize(z, z_bins) - 1  # bin index for each object

    # Normalize for colormap
    norm = plt.Normalize(vmin=0, vmax=n_bins - 1)
    cmap = plt.cm.viridis  # or any other colormap
    scatter = plt.scatter(
        log_lbol, log_sigma_hat_sq,
        c=z_bin_indices,
        cmap=cmap,
        norm=norm,
        s=10,
        alpha=0.5,
        label='Observed'
    )

    # Colorbar with proper bin labels
    cbar = plt.colorbar(scatter, ticks=range(n_bins))
    bin_labels = [f"{z_bins[i]:.2f}-{z_bins[i+1]:.2f}" for i in range(n_bins)]
    cbar.ax.set_yticklabels(bin_labels)
    cbar.set_label('Redshift bin (z)')
    #plt.scatter(log_lbol, log_sigma_hat_sq, s=10, alpha=0.5, color='navy', label='Observed')
    plt.xlabel(r'$\log L_{\mathrm{bol}}$')
    plt.ylabel(r'$\log \hat{\sigma}_{\mathrm{UV}}^2$')
    plt.title(r'Predicted $\log \hat{\sigma}_{\mathrm{UV}}^2$ vs $\log L_{\mathrm{bol}}$')
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    os.makedirs("plots/hubble", exist_ok=True)
    plt.savefig(f"plots/hubble/predicted_sigma_hat_sq_{cosmo_model}_loglog.png", dpi=300)
    plt.savefig(f"plots/hubble/predicted_sigma_hat_sq_{cosmo_model}_loglog.pdf", dpi=300)
    if show:
        plt.show()
    plt.close()



#def plot_predicted_sigma_hat_vs_luminosity_linear_fit(sampler, df_agn, cosmo_model, show=False):
def plot_predicted_sigma_hat_vs_luminosity(sampler, df_agn, cosmo_model, show=False, log_sigma_hat_pivot=-2.2):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # Load samples
    flat_samples = sampler.get_chain(flat=True, thin=20)
    priors, model_labels = get_model_params(cosmo_model)
    param_indices = {name: model_labels.index(name) for name in model_labels}

    # Extract arrays of model parameters
    M0_samples = flat_samples[:, param_indices['M0_agn']]
    alpha_samples = flat_samples[:, param_indices['alpha_agn']]

    # Define grid in log L_bol
    log_lbol_grid = np.linspace(44, 47, 200)
    lbol_grid = 10 ** log_lbol_grid

    # Precompute terms
    pivot_term = 2 * log_sigma_hat_pivot
    slopes = -2.5 / alpha_samples
    intercepts = (90 - M0_samples) / alpha_samples + pivot_term

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
    plt.figure(figsize=(7, 5))

    # Posterior 1σ band and median line
    plt.fill_between(
        lbol_grid,
        y_low_lin,
        y_high_lin,
        color='orange',
        alpha=0.4,
        label=r'Fit $\pm 1\sigma$'
    )
    plt.plot(lbol_grid, y_median_lin, color='orange', lw=2, label='Median fit')

    # Observed data points with error bars
    plt.errorbar(
        lbol,
        sigma_hat_sq,
        yerr=sigma_hat_sq_err,
        xerr=lbol_err,
        fmt='o',
        color='black',
        markerfacecolor='black',
        markersize=3,
        capsize=3,
        elinewidth=1,
        linestyle='none',
        alpha=0.7,
        zorder=3,
        label='Observed'
    )
    z = df_agn['z'].values
    n_bins = 4
    z_bins = np.linspace(z.min(), z.max(), n_bins + 1)
    z_bin_indices = np.digitize(z, z_bins) - 1  # bin index for each object

    # Normalize for colormap
    norm = plt.Normalize(vmin=0, vmax=n_bins - 1)
    cmap = plt.cm.viridis  # or any other colormap
    # scatter = plt.scatter(
    #     log_lbol, log_sigma_hat_sq,
    #     c=z_bin_indices,
    #     cmap=cmap,
    #     norm=norm,
    #     s=10,
    #     alpha=0.5,
    #     label='Observed'
    # )

    # Colorbar with proper bin labels
    # cbar = plt.colorbar(scatter, ticks=range(n_bins))
    # bin_labels = [f"{z_bins[i]:.2f}-{z_bins[i+1]:.2f}" for i in range(n_bins)]
    # cbar.ax.set_yticklabels(bin_labels)
    # cbar.set_label('Redshift bin (z)')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$L_{\mathrm{bol}}$ [erg/s]')
    plt.ylabel(r'$\hat{\sigma}_{\mathrm{UV}}^2$')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(frameon=False, fontsize=10)
    plt.tight_layout()

    os.makedirs("plots/hubble", exist_ok=True)
    plt.savefig(f"plots/hubble/predicted_sigma_hat_sq_{cosmo_model}_analyticalband_withdata.png", dpi=300)
    plt.savefig(f"plots/hubble/predicted_sigma_hat_sq_{cosmo_model}_analyticalband_withdata.pdf", dpi=300)
    if show:
        plt.show()
    plt.close()

    return slopes, intercepts


def plot_sigma_hat_vs_log_lbol(df_agn, show=False):
    """
    Plot 2 * log_sigma_hat_UV vs log_lbol for AGN sample.

    Parameters:
        df_agn : pandas.DataFrame
            DataFrame containing 'log_sigma_hat_UV' and 'log_lbol' columns.
        show : bool
            Whether to display the plot interactively.
    """
    x = df_agn['log_lbol']
    y = 2 * df_agn['log_sigma_hat_UV']

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=10, alpha=0.5, color='navy')
    plt.xlabel(r'$\log L_{\mathrm{bol}}$')
    plt.ylabel(r'$2 \log \hat{\sigma}_{\mathrm{UV}}$')
    plt.title(r'$2 \log \hat{\sigma}_{\mathrm{UV}}$ vs $\log L_{\mathrm{bol}}$')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs("plots/hubble", exist_ok=True)
    plt.savefig("plots/hubble/2log_sigma_hat_vs_log_lbol.png", dpi=300)
    plt.savefig("plots/hubble/2log_sigma_hat_vs_log_lbol.pdf", dpi=300)
    if show:
        plt.show()
    plt.close()