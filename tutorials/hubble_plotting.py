import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from tqdm import tqdm
import math
import corner
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.cosmology import FlatwCDM, Flatw0waCDM
import matplotlib.pyplot as plt
from hubble_model import *

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
    fig = corner.corner(flat_samples, labels=model_labels, truths=None, show_titles=True, title_fmt=".2f")
    if only_sna:
        fig.suptitle("SNIa only", fontsize=16)
        plt.savefig(f"plots/posterior_{cosmo_model}_sna.png", dpi=200)
    else:
        fig.suptitle("SNIa + AGN", fontsize=16)
        plt.savefig(f"plots/posterior_{cosmo_model}_agn.png", dpi=200)
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
    fig.savefig(f"plots/corner_kde_{cosmo_model}.pdf", bbox_inches="tight", transparent=True)
    fig.savefig(f"plots/corner_kde_{cosmo_model}.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_hubble(sampler, df_agn, df_pantheon, cosmo_model, show=False):
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

    # --- AGN distance modulus ---
    mu_pred = np.array([
        df_agn['apparent_mag_i'] - K_corr(df_agn['z']) - (
            M_model_agn(s[param_indices['M0_agn']], s[param_indices['alpha_agn']], df_agn['log_sigma_UV'], df_agn['log_tau_UV_RF']) - K_corr(2))
        for s in flat_samples
    ])
    mu_pred_median = np.percentile(mu_pred, 50, axis=0)
    mu_pred_16th = np.percentile(mu_pred, 16, axis=0)
    mu_pred_84th = np.percentile(mu_pred, 84, axis=0)
    mu_pred_std = np.sqrt(df_agn['apparent_mag_i_err']**2 +
            np.abs(0.5 * (mu_pred_84th - mu_pred_16th))**2 +
                 (-2.5 * 0.3 * np.log10(1 + df_agn["z"]))**2 +
                 #(Kcorr * 0.05)**2 +
                (results["alpha"][1] * np.sqrt((2*df_agn['log_sigma_UV_err'])**2+df_agn['log_tau_UV_RF_err']**2))**2)

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
    ax.set_ylim(26, 51)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.3, 0.05))

    # Ticks styling
    for axi in [ax, inset_ax]:
        axi.minorticks_on()
        axi.tick_params(axis='both', which='minor', direction='in', length=4, top=True, right=True, width=2)
        axi.tick_params(axis='both', which='major', direction='in', length=8, top=True, right=True)

    fig.tight_layout()
    plt.savefig(f"plots/hubble_diagram_{cosmo_model}.pdf", dpi=300)
    plt.savefig(f"plots/hubble_diagram_{cosmo_model}.png")
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
        df_agn['log_sigma_UV'], 
        df_agn['log_tau_UV_RF']
    ) #+ K_corr(2) # TODO: check this


    # Calculate prediction errors
    M_i_pred_err = np.sqrt(
        #df_agn['M_i_err']**2 +
        (results['alpha_agn'][1] * np.sqrt((2*df_agn["log_sigma_UV_err"])**2+df_agn["log_tau_UV_RF_err"]**2))**2
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

    plt.savefig(f"plots/predicted_vs_actual_Mi_{cosmo_model}.png", dpi=300)
    plt.savefig(f"plots/predicted_vs_actual_Mi_{cosmo_model}.pdf", dpi=300)
    if show:
        plt.show()
    plt.close()


