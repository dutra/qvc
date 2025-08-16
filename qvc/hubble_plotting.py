import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from tqdm import tqdm
import math
import corner
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.cosmology import FlatwCDM, FlatwpwaCDM, FlatLambdaCDM
import matplotlib.pyplot as plt
import os
import copy

from hubble_model import M_model_agn, M_model_agn_err, M_model_SN, get_model_params, log_tau_UV_RF_pivot
from hubble_utils import calc_Mi_from_M2500
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import interp1d
from dynesty.utils import resample_equal
from tqdm import tqdm
from dynesty import plotting as dyplot

def plot_dynesty(results, cosmo_model, basename="plots/hubble/", show=False):
    """
    Plot dynesty diagnostics: runplot, traceplot, and cornerpoints using dyplot.
    Saves figures to files with the given basename.
    """

    os.makedirs(os.path.dirname(basename), exist_ok=True)
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)

    # Cornerplot
    fig_corner, axes_corner = dyplot.cornerplot(results, labels=model_labels_latex, quantiles=[0.16, 0.5, 0.84],
                                                 quantiles_2d = [0.393, 0.865, 0.989],
                                                 show_titles=True, title_quantiles=[0.16, 0.5, 0.84],
                                                 color='blue',
                                                 #fig=plt.subplots(1, 1, figsize=(10, 2.5 * len(model_labels))))
    )
    fig_corner.savefig(f"{basename}_cornerplots.png", dpi=100)
    if show:
        plt.show()    
    plt.close(fig_corner)

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
        plt.show()
    plt.close(fig_trace)


    # # Cornerpoints
    # fig_corner, axes_corner = dyplot.cornerpoints(results, labels=model_labels_latex, cmap='plasma')
    # fig_corner.savefig(f"{basename}_cornerpoints.png", dpi=100)
    # if show:
    #     fig_corner.show()    
    # plt.close(fig_corner)

    # # Make a shallow copy of results to avoid touching the real object
    # results_plot = copy.deepcopy(results)
    # try:
    #     if results_plot.logz[-1] > 700:
    #         results_plot.logz[-1] = 700  # Safe maximum for exp
    #         print("🔧 Clipped logz[-1] to prevent overflow in runplot")

    #     fig_run, axes_run = dyplot.runplot(results_plot)
    #     fig_run.savefig(f"{basename}_runplot.png", dpi=100)
    #     if show:
    #         fig_run.show()
    #     plt.close(fig_run)
    # except Exception as e:
    #     print(f"Error in runplot: {e}")
        
def plot_traces(sampler, only_sna=False, cosmo_model='Flatw0waCDM', show=False, use_dynesty=False):
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

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from tqdm import tqdm

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

def plot_cosmo_corner(
    flat_samples_sn,
    flat_samples_agn,
    cosmo_model,
    z_agn_pivot,
    show=False,
):
    """
    Corner-style plot (custom) of key cosmology params with diagonal stats labels.
      - Flatw0waCDM: H0, Om0, w0(= w(a=1) derived from wp,wa at z_agn_pivot), wa
      - FlatwCDM:    H0, Om0, w0
    If flat_samples_sn is None or empty, only the SN+AGN (red) set is plotted.
    """
    # --- pull model labels from your config ---
    _, model_labels, _ = get_model_params(cosmo_model)

    # ---------- helpers ----------
    def _find(labels, *cands):
        for c in cands:
            if c in labels:
                return labels.index(c)
        raise KeyError(f"Could not find any of {cands} in labels {labels}")

    def _subset(samples, labels):
        """Return reduced samples (N,k) and latex labels for the chosen model."""
        X = np.asarray(samples)
        i_H0  = _find(labels, "H0", "H_0")
        i_Om0 = _find(labels, "Om0", "OmegaM", "Omega_m")

        if cosmo_model == "Flatw0waCDM":
            i_wp = _find(labels, "wp", "w_p")
            i_wa = _find(labels, "wa", "w_a")
            a_p  = 1.0 / (1.0 + float(z_agn_pivot))
            wp, wa = X[:, i_wp], X[:, i_wa]
            w0 = wp - (1.0 - a_p) * wa
            Y = np.column_stack([X[:, i_H0], X[:, i_Om0], w0, wa])
            lab_latex = [r"$H_0$", r"$\Omega_m$", r"$w_0$", r"$w_a$"]
        elif cosmo_model == "FlatwCDM":
            i_w0 = _find(labels, "w0", "w_0", "w")
            Y = np.column_stack([X[:, i_H0], X[:, i_Om0], X[:, i_w0]])
            lab_latex = [r"$H_0$", r"$\Omega_m$", r"$w_0$"]
        else:
            raise ValueError(f"Unsupported cosmo_model '{cosmo_model}' for this plot.")
        return Y, lab_latex

    def _fmt_err(m, lo, hi):
        # a small heuristic for decimals
        nd = 3 if abs(m) < 1 else 2
        return f"{m:.{nd}f}", f"{hi - m:.{nd}f}", f"{m - lo:.{nd}f}"

    def _get_density_levels(values, probs=[0.393, 0.865]):
        z = values.ravel()
        z_sorted = np.sort(z)
        cdf = np.cumsum(z_sorted); cdf /= max(cdf[-1], 1e-300)
        levels = [z_sorted[np.searchsorted(cdf, 1 - p)] for p in probs]
        return np.unique(np.sort(levels))

    def _filled_kde(ax, x, y, color, base_alpha=0.4):
        kde = gaussian_kde(np.vstack([x, y]))
        xmin, xmax = np.percentile(x, [0.5, 99.5])
        ymin, ymax = np.percentile(y, [0.5, 99.5])
        xgrid = np.linspace(xmin, xmax, 120)
        ygrid = np.linspace(ymin, ymax, 120)
        xx, yy = np.meshgrid(xgrid, ygrid)
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        levels = _get_density_levels(zz, [0.393, 0.865])
        for i in range(len(levels)-1, -1, -1):
            ax.contourf(xx, yy, zz,
                        levels=[levels[i], zz.max()],
                        colors=[color], alpha=base_alpha*(i+1)/len(levels))
        ax.contour(xx, yy, zz, levels=levels, colors=[color], linewidths=1.2)

    # --- reduce to the parameters we actually plot ---
    agn_data, labels_latex = _subset(flat_samples_agn, model_labels)
    sna_data = None
    if flat_samples_sn is not None and len(flat_samples_sn) > 0:
        sna_data, _ = _subset(flat_samples_sn, model_labels)

    n_params = agn_data.shape[1]
    fig, axes = plt.subplots(n_params, n_params, figsize=(2.3*n_params, 2.3*n_params))

    # --- build the grid ---
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            ax.tick_params(direction='in')

            if i < j:
                ax.axis("off")
                continue

            if i == j:
                # 1D KDEs
                xs = np.linspace(np.min(agn_data[:, i]), np.max(agn_data[:, i]), 400)
                kde_r = gaussian_kde(agn_data[:, i])
                ax.plot(xs, kde_r(xs), color="red", lw=1.8)

                if sna_data is not None:
                    xs_b = np.linspace(np.min(sna_data[:, i]), np.max(sna_data[:, i]), 400)
                    kde_b = gaussian_kde(sna_data[:, i])
                    ax.plot(xs_b, kde_b(xs_b), color="blue", lw=1.8)

                # diagonal titles: AGN (red) on top; SN (blue) below if present
                m, lo, hi = np.median(agn_data[:, i]), np.percentile(agn_data[:, i],16), np.percentile(agn_data[:, i],84)
                ms, ps, ns = _fmt_err(m, lo, hi)
                ax.set_title(rf"{labels_latex[i]} = {ms}" + rf"$^{{+{ps}}}_{{-{ns}}}$",
                             color="red", fontsize=11, loc="left", pad=2)

                if sna_data is not None:
                    mb, lob, hib = np.median(sna_data[:, i]), np.percentile(sna_data[:, i],16), np.percentile(sna_data[:, i],84)
                    msb, psb, nsb = _fmt_err(mb, lob, hib)
                    ax.text(0.02, 0.86,
                            rf"{labels_latex[i]} = {msb}" + rf"$^{{+{psb}}}_{{-{nsb}}}$",
                            transform=ax.transAxes, ha="left", va="top",
                            color="blue", fontsize=11)
            else:
                # 2D KDEs
                if sna_data is not None:
                    _filled_kde(ax, sna_data[:, j], sna_data[:, i], "blue", base_alpha=0.4)
                _filled_kde(ax, agn_data[:, j], agn_data[:, i], "red", base_alpha=0.4)

            # tidy labels
            if j == 0:
                ax.set_ylabel(labels_latex[i])
            else:
                ax.set_yticklabels([])

            if i == n_params - 1:
                ax.set_xlabel(labels_latex[j])
            else:
                ax.set_xticklabels([])

    # legend
    legend = []
    if sna_data is not None:
        legend.append(Line2D([0],[0], color="blue", lw=4, label="SN Ia"))
    legend.append(Line2D([0],[0], color="red",  lw=4, label="SN Ia + AGN"))
    fig.legend(handles=legend, bbox_to_anchor=(0.5, 0.92), loc="upper left",
               fontsize=12, frameon=False, markerscale=1.5)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.9,
                        wspace=0.05, hspace=0.05)

    os.makedirs("plots/hubble", exist_ok=True)
    fig.savefig(f"plots/hubble/corner_kde_{cosmo_model}.png", bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)



def plot_hubble(flat_samples, df_agn, df_pantheon, cosmo_model, z_agn_pivot, show=False, completeness=True, show_true=False, fake_params=None):
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
    
    z_grid = np.linspace(0.0001, 5.2, 1000)

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
            FlatwpwaCDM(
                H0=s[param_indices['H0']],
                Om0=s[param_indices['Om0']],
                wp=s[param_indices['wp']],
                wa=s[param_indices['wa']],
                zp=z_agn_pivot
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

    apparent_mag = df_agn['apparent_mag_2500']

    # compute M_actual
    if cosmo_model == 'FlatwCDM':
        cosmo = FlatwCDM(
            H0=results['H0'][1],
            Om0=results['Om0'][1],
            w0=results['w0'][1]
        )
    elif cosmo_model == 'Flatw0waCDM':
        cosmo = FlatwpwaCDM(
            H0=results['H0'][1],
            Om0=results['Om0'][1],
            wp=results['wp'][1],
            wa=results['wa'][1],
            zp=z_agn_pivot
        )
    elif cosmo_model == 'FlatLambdaCDM':
        cosmo = FlatLambdaCDM(
            H0=results['H0'][1],
            Om0=results['Om0'][1]
        )
    else:
        raise ValueError("Invalid cosmology model.")
    
    #M_actual = M_2500_from_logL_2500_recosmo_pivot(df_agn['log_l2500'].values, df_agn['z'].values, cosmo_target=cosmo, z0=2, alpha_nu=df_agn['alpha_nu'].values)
    #print(M_actual.min(), M_actual.max())

    # Then re-compute the distance modulus       
    mu_pred = np.array([
        apparent_mag -
            (M_model_agn(
                s[param_indices['M0_agn']], 
                        s[param_indices['alpha_agn']],
                        s[param_indices['beta_agn']],
                        df_agn['log_sigma0'].values, df_agn['log_tau_UV_RF'].values,
                        df_agn['f_host'].values
                        ))
        for s in flat_samples
    ])
    if fake_params is not None:
        mu_pred = np.array([
            apparent_mag -
            (M_model_agn(
                fake_params['M0_agn'], 
                fake_params['alpha_agn'],
                fake_params['beta_agn'], 
                df_agn['log_sigma0'].values, df_agn['log_tau_UV_RF'].values,
            ))
            for _ in range(len(flat_samples))
        ])
    mu_pred_16th = np.percentile(mu_pred, 16, axis=0)
    mu_pred_84th = np.percentile(mu_pred, 84, axis=0)
    mu_pred_std = np.sqrt(df_agn['apparent_mag_2500_err'].values**2 +
                 #(-2.5 * 0.3 * np.log10(1 + df_agn["z"]))**2 +
                 #(-2.5 * df_agn['alpha_nu_err'].values * np.log10(1 + df_agn["z"]))**2 +
                 (0.055 * df_agn["z"])**2 +
                #(results["alpha_agn"][1] * df_agn['log_sigma0_err']))**2
                M_model_agn_err(results['M0_agn'][1],
                                  results['alpha_agn'][1],
                                  results['beta_agn'][1],
                                  df_agn['log_sigma0'].values,
                                  df_agn['log_sigma0_err'].values,
                                  df_agn['log_tau_UV_RF_err'].values,
                                  df_agn['f_host_err'].values)**2)
    # Now take the median again:
    mu_pred_median = np.percentile(mu_pred, 50, axis=0)

    #--- Residuals ---
    mu_interp = np.interp(df_agn["z"], z_grid, mu_model_median)
    residuals = mu_pred_median - mu_interp

    # --- Binning ---
    bins = np.linspace(df_agn["z"].min(), df_agn["z"].max(), 24)
    bin_indices = np.digitize(df_agn["z"], bins)
    # Compute counts per bin
    bin_counts = [np.sum(bin_indices == i) for i in range(1, len(bins))]
    # Mask: True if bin has at least 3 items
    valid_mask = np.array(bin_counts) >= 3

    binned_mu_pred_mean = [np.mean(mu_pred_median[bin_indices == i]) for i in range(1, len(bins))]
    binned_mu_pred_std = [np.std(mu_pred_median[bin_indices == i])/np.sqrt(len(mu_pred_median[bin_indices == i])) for i in range(1, len(bins))]
    binned_z = [np.median(df_agn["z"][bin_indices == i]) for i in range(1, len(bins))]

    # Apply mask
    binned_mu_pred_mean = np.array(binned_mu_pred_mean)[valid_mask]
    binned_mu_pred_std = np.array(binned_mu_pred_std)[valid_mask]
    binned_z = np.array(binned_z)[valid_mask]

    # --- Plot setup ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    inset_ax = inset_axes(ax, width="40%", height="40%", loc="lower right", borderpad=1.5)

    # --- Inset plot ---
    inset_ax.scatter(df_agn["z"], mu_pred_median, s=2, label="AGN", alpha=0.5, color="black",zorder=1)
    inset_ax.scatter(df_pantheon["zHD"], df_pantheon["MU_SH0ES"], color="dodgerblue", s=2, label="SN Ia", alpha=0.5, zorder=-10)
    inset_ax.plot(z_grid, mu_model_median, alpha=0.9, color="purple", zorder=10, lw=0.5, label="ΛCDM Model")
    inset_ax.fill_between(z_grid, mu_model_16th, mu_model_84th, color="purple", alpha=0.9, zorder=9)
    inset_ax.set_xscale('log')
    inset_ax.set_xlim(0.02, 5.2)
    inset_ax.set_ylim(32, 51)
    inset_ax.set_xlabel(r"$z$", fontsize=12, labelpad=-10)
    inset_ax.set_ylabel(r"$\mu$ (mag)", fontsize=12)
    inset_ax.tick_params(axis='both', which='major', labelsize=10)
#    inset_ax.legend(frameon=False, fontsize=6)

    # --- Main Hubble diagram ---
    # Original AGNs
    # ax.errorbar(df_agn["z"], mu_pred_median, yerr=mu_pred_std, fmt='o', linestyle='none',
    #             markersize=2, alpha=0.2, color='k', lw=1.5, zorder=-7, label="AGN")
    # Color AGN points by M_i
    sc = ax.scatter(
        df_agn["z"], mu_pred_median, c=df_agn["LOGLBOL"], cmap="viridis",
        s=20, alpha=0.7, edgecolor='k', lw=0.3, zorder=-7, label="AGN"
    )
    ax.errorbar(
        df_agn["z"], mu_pred_median, yerr=mu_pred_std, fmt='none',
        ecolor='gray', alpha=0.3, lw=1, zorder=-8
    )
    cbar = plt.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label(r"LOGLBOL", fontsize=12)

    # Binned AGNs
    ax.errorbar(binned_z, binned_mu_pred_mean, yerr=binned_mu_pred_std, label="Binned AGN",
                fmt='o', markersize=4, capsize=3, lw=1.5,alpha=0.9, color="red", zorder=-1)

    
    if show_true:
        ax.scatter(df_agn['z'], df_agn['apparent_mag_2500'] - df_agn['M_2500'], alpha=0.7, edgecolor='k')
        ax.scatter(df_agn['z'], df_agn['apparent_mag_i'] - df_agn['M_i'], alpha=0.7, edgecolor='g')
    
    # SNIa points
    ax.errorbar(df_pantheon["zHD"], df_pantheon["MU_SH0ES"], yerr=df_pantheon["MU_SH0ES_ERR_DIAG"], 
                fmt='s', markersize=3, color="dodgerblue", linestyle='none', lw=1, label="SN Ia", alpha=0.7, zorder=-9)

    # Cosmo model band
    ax.plot(z_grid, mu_model_median, alpha=0.9, color="m", zorder=-5, lw=2, label=label)
    ax.fill_between(z_grid, mu_model_16th, mu_model_84th, color="purple", alpha=0.9, zorder=-5)

    # Use median values for the other parameters, but evaluate at z_grid
    log_sigma0_med = np.median(df_agn['log_sigma0'].values)
    log_tau_UV_RF_med = np.median(df_agn['log_tau_UV_RF'].values)
    f_host_med = np.median(df_agn['f_host'].values)

    M_med_grid = np.median([
        M_model_agn(
            s[param_indices['M0_agn']],
            s[param_indices['alpha_agn']],
            s[param_indices['beta_agn']],
            log_sigma0_med * np.ones_like(z_grid),
            log_tau_UV_RF_med * np.ones_like(z_grid),
            f_host_med * np.ones_like(z_grid)
        )
        for s in flat_samples
    ], axis=0)

    mu_med = 24.0 - M_med_grid
    ax.fill_between(z_grid, np.full_like(mu_med, 55), mu_med, color="k", lw=0, alpha=0.25)

    # Plot concordance FlatLambdaCDM as dashed line
    concordance_cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    mu_concordance = concordance_cosmo.distmod(z_grid).value
    ax.plot(z_grid, mu_concordance, color="gray", lw=2, ls="--", label="Concordance ΛCDM")

    ax.set_ylabel(r"$\mu$ (mag)")
    ax.set_xlabel(r"$z$")
    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(26, 51)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.3, 0.05), fontsize=16)

    # Ticks styling
    for axi in [ax, inset_ax]:
        axi.minorticks_on()
        axi.tick_params(axis='both', which='minor', direction='in', length=4, top=True, right=True, width=2)
        axi.tick_params(axis='both', which='major', direction='in', length=8, top=True, right=True)

    fig.tight_layout()
    os.makedirs("plots/hubble", exist_ok=True)
    #plt.savefig(f"plots/hubble/hubble_diagram_{cosmo_model}{show_uncorrected_label}.pdf", dpi=300)
    plt.savefig(f"plots/hubble/hubble_diagram_{cosmo_model}.png")
    # ax.set_yscale('log')
    ax.set_ylim(26, 51)
    ax.set_xlim(-0.2, 4.2)
    # plt.savefig(f"plots/hubble/hubble_diagram_{cosmo_model}_ylog.png")

    if show:
        plt.show()
    plt.close()
    return residuals, mu_pred_median, mu_pred_std


def plot_predicted_vs_actual_M2500(flat_samples, df_agn, cosmo_model, z_agn_pivot, show=False):
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    results = {key: np.percentile(flat_samples[:, i], [16, 50, 84]) for i, key in enumerate(model_labels)}
    # compute M_actual
    if cosmo_model == 'FlatwCDM':
        cosmo = FlatwCDM(
            H0=results['H0'][1],
            Om0=results['Om0'][1],
            w0=results['w0'][1]
        )
    elif cosmo_model == 'Flatw0waCDM':
        cosmo = FlatwpwaCDM(
            H0=results['H0'][1],
            Om0=results['Om0'][1],
            wp=results['wp'][1],
            wa=results['wa'][1],
            zp=z_agn_pivot
        )
    elif cosmo_model == 'FlatLambdaCDM':
        cosmo = FlatLambdaCDM(
            H0=results['H0'][1],
            Om0=results['Om0'][1]
        )
    else:
        raise ValueError("Invalid cosmology model.")
        
    actual_M_2500 = df_agn['apparent_mag_2500'].values - np.array([cosmo.distmod(z).value for z in df_agn['z'].values])

    M_2500_pred = M_model_agn(
        results['M0_agn'][1], 
        results['alpha_agn'][1],
        results['beta_agn'][1], 
        df_agn['log_sigma0'].values,
        df_agn['log_tau_UV_RF'].values,
        df_agn['f_host'].values
    )

    M_2500_pred_err = np.sqrt(
        M_model_agn_err(
            results['M0_agn'][1],
            results['alpha_agn'][1], 
            results['beta_agn'][1], 
            df_agn['log_sigma0'].values, 
            df_agn['log_sigma0_err'].values, 
            df_agn['log_tau_UV_RF_err'].values,
            df_agn['f_host_err'].values
        )**2
    )

    z_bins = np.linspace(0.5, 3.5, 10)
    z_bin_indices = np.digitize(df_agn['z'], bins=z_bins)
    num_bins = len(z_bins) - 1
    bin_labels = [f"{z_bins[i]:.1f} < z < {z_bins[i+1]:.1f}" for i in range(num_bins)]
    num_cols = 3
    num_rows = math.ceil(num_bins / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), sharey=True, sharex=True)
    axes = axes.flatten()

    apparent_mag_2500 = df_agn['apparent_mag_2500'].values
    vmin = np.nanmin(apparent_mag_2500)
    vmax = np.nanmax(apparent_mag_2500)

    for i, ax in enumerate(axes):
        ax.set_xlim(-25.8, -18.2)
        ax.set_ylim(-25.8, -18.2)

        if i < num_bins:
            bin_mask = z_bin_indices == (i + 1)
            predicted_M_2500_bin = M_2500_pred[bin_mask]
            actual_M_2500_bin = actual_M_2500[bin_mask]
            apparent_mag_2500_bin = apparent_mag_2500[bin_mask]
            M_i_axis = np.linspace(actual_M_2500.min(), actual_M_2500.max(), 100)
            ax.plot(M_i_axis, M_i_axis, color='m', alpha=0.7, label='y = x (Perfect Prediction)', lw=3, linestyle='--')
            sc = ax.scatter(
                actual_M_2500_bin, predicted_M_2500_bin, 
                c=apparent_mag_2500_bin, cmap='viridis', s=20, alpha=0.7, edgecolor='k', lw=0.5, vmin=vmin, vmax=vmax
            )
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.annotate(bin_labels[i], xy=(0.05, 0.95), xycoords='axes fraction', 
                        fontsize=18, color='k', alpha=0.7, ha='left', va='top')
            n_in_bin = np.sum(bin_mask)
            ax.annotate(f"N = {n_in_bin}", xy=(0.95, 0.05), xycoords='axes fraction',
                        fontsize=14, color='gray', ha='right', va='bottom')
            if i >= (num_rows - 1) * num_cols:
                ax.set_xlabel('Actual $M_{2500}$')
            if i % num_cols == 0:
                ax.set_ylabel('Predicted $M_{2500}$')
        else:
            ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)

    # Add a colorbar for f_host
    cbar = fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label(r'm$_{2500}$', fontsize=14)

    os.makedirs("plots/hubble", exist_ok=True)
    plt.savefig(f"plots/hubble/predicted_vs_actual_M2500_{cosmo_model}.png", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_predicted_vs_actual_Mi(flat_samples, df_agn, cosmo_model, z_agn_pivot, show=False):
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    results = {key: np.percentile(flat_samples[:, i], [16, 50, 84]) for i, key in enumerate(model_labels)}
    # compute M_actual
    if cosmo_model == 'FlatwCDM':
        cosmo = FlatwCDM(
            H0=results['H0'][1],
            Om0=results['Om0'][1],
            w0=results['w0'][1]
        )
    elif cosmo_model == 'Flatw0waCDM':
        cosmo = FlatwpwaCDM(
            H0=results['H0'][1],
            Om0=results['Om0'][1],
            wp=results['wp'][1],
            wa=results['wa'][1],
            zp=z_agn_pivot
        )
    elif cosmo_model == 'FlatLambdaCDM':
        cosmo = FlatLambdaCDM(
            H0=results['H0'][1],
            Om0=results['Om0'][1]
        )
    else:
        raise ValueError("Invalid cosmology model.")
        
    #M_actual = M_2500_from_logL_2500_recosmo_pivot(df_agn['log_l2500'].values, df_agn['z'].values, cosmo_target=cosmo, z0=2, alpha_nu=df_agn['alpha_nu'].values)
    actual_M_i = df_agn['M_i'].values

    M_2500_pred = M_model_agn(
        results['M0_agn'][1], 
        results['alpha_agn'][1],
        results['beta_agn'][1], 
        df_agn['log_sigma0'].values,
        df_agn['log_tau_UV_RF'].values,
        df_agn['f_host'].values,
    ) #- (K_corr(df_agn['z'], df_agn['alpha_nu'].values) - K_corr(2.0, df_agn['alpha_nu'].values)) # TODO: check this
    #) + K_corr(2.0, df_agn['alpha_nu'].values)

    # Calculate prediction errors
    M_2500_pred_err = np.sqrt(
        M_model_agn_err(
            results['M0_agn'][1],
            results['alpha_agn'][1],
            results['beta_agn'][1], 
            
            df_agn['log_sigma0'].values, 
            df_agn['log_sigma0_err'].values, 
            df_agn['log_tau_UV_RF_err'].values,
            df_agn['f_host_err'].values,
        )**2
    )

    # M_i_pred = convert_M2500_to_MI(M_2500_pred, df_agn['alpha_nu'].values)# - 2
    M_i_pred = calc_Mi_from_M2500(M_2500_pred, df_agn['alpha_nu'].values, np.full_like(df_agn['alpha_nu'].values, 2))
    M_i_pred_err = np.zeros_like(M_i_pred)

    # Bin and color by redshift
    #z_bins = np.linspace(df_agn['z'].min(), df_agn['z'].max(), 10)  # Define redshift bins
    z_bins = np.linspace(0.5, 3.5, 10)  # Define redshift bins
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
        ax.set_xlim(actual_M_i.min(), actual_M_i.max())
        ax.set_ylim(M_i_pred.min(), M_i_pred.max())
        # ax.set_xlim(-28.8, -21.2)
        # ax.set_ylim(-28.8, -21.2)

        if i < num_bins:
            # Filter data for the current redshift bin
            bin_mask = z_bin_indices == (i + 1)
            predicted_M_i_bin = M_i_pred[bin_mask]
            predicted_M_i_err_bin = M_i_pred_err[bin_mask]
            M_i_axis = np.linspace(actual_M_i.min(), actual_M_i.max(), 100)
            ax.plot(M_i_axis, M_i_axis, color='m', alpha=0.7, label='y = x (Perfect Prediction)', lw=3, linestyle='--')
            # Scatter plot for the current bin with error bars
            # scatter = ax.errorbar(
            #     actual_M_i[bin_mask], predicted_M_i_bin, xerr=0.25, yerr=predicted_M_i_err_bin, 
            #     fmt='o', markerfacecolor='k', markeredgecolor='k', alpha=0.4, lw=1.5, capsize=3, capthick=1, color='k'
            # )
            scatter = ax.scatter(
                actual_M_i[bin_mask], predicted_M_i_bin, 
                s=20, alpha=0.5, edgecolor='k', facecolor='k', lw=0.5,
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
                ax.set_xlabel('Actual $M_{i}$')
            if i % num_cols == 0:  # Add ylabel only for the first column
                ax.set_ylabel('Predicted $M_{i}$')
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

def plot_Mi_vs_sigmahat(df_agn, cosmo_model, show=False):
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    param_indices = {name: model_labels.index(name) for name in model_labels}

    log_sigma0 = df_agn['log_sigma0']
    M_i = df_agn['M_i']

    plt.figure(figsize=(8, 6))
    plt.scatter(log_sigma0, M_i, label='Data', color='k', alpha=0.5)
    plt.xlabel(r'$\log \hat{\sigma}^2$')
    plt.ylabel(r'$M_{i}$')
    plt.legend()
    if show:
        plt.show()
    plt.savefig(f"plots/hubble/Mi_vs_sigmahat_{cosmo_model}.png", dpi=300)
    plt.close()



def plot_completeness_diagnostics(df_agn, completeness2d, mag_centers, z_centers, output_prefix="plots/hubble/completeness_diag"):
    """
    Plot and save diagnostic figures for completeness:
    - Empirical P(detect) vs. redshift
    - 2D completeness map
    - Inverse completeness correction map (log10[1 / P])
    
    Parameters:
        df_agn           : DataFrame with 'z' and 'apparent_mag_2500'
        completeness2d   : callable completeness function (mag, z)
        mag_centers      : 1D array of magnitude bin centers
        z_centers        : 1D array of redshift bin centers
        output_prefix    : filename prefix for saved figures (default: "completeness_diag")
    """
    z = df_agn['z'].values
    m = df_agn['apparent_mag_2500'].values
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


def plot_full_residuals(df_agn, residuals, flat_samples, cosmo_model, z_agn_pivot, show=False):
    import math

    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    param_indices = {name: model_labels.index(name) for name in model_labels}

    results = {key: np.percentile(flat_samples[:, i], [16, 50, 84]) for i, key in enumerate(model_labels)}

    if cosmo_model == 'FlatwCDM':
        cosmo = FlatwCDM(
            H0=results['H0'][1],
            Om0=results['Om0'][1],
            w0=results['w0'][1]
        )
    elif cosmo_model == 'Flatw0waCDM':
        cosmo = FlatwpwaCDM(
            H0=results['H0'][1],
            Om0=results['Om0'][1],
            wp=results['wp'][1],
            wa=results['wa'][1],
            zp=z_agn_pivot,
        )
    else:
        raise ValueError("Invalid cosmology model.")

    df_agn['MY_M_2500'] = df_agn['apparent_mag_2500'].values - np.array([cosmo.distmod(z).value for z in df_agn['z'].values])


    # Start with z < 10
    mask = df_agn['z'] < 10

    # Exclude non-numeric columns from -1e9 check
    num_cols = df_agn.select_dtypes(include=[np.number]).columns
    mask &= ~(df_agn[num_cols] == -1e9).any(axis=1)

    # Select only the keys in your specified list (order preserved by np.flip)
    keys = [col for col in np.flip([
        'apparent_mag_2500', 'MY_M_2500', 'z', 'log_lbol', 'log_ledd_ratio', 
        'log_sigma0', 'log_sigma_hat_UV', 'log_tau_UV_RF', 'chi_sq_g',
        'f_host', 'sn_median_all', 'bwb_alpha', 'bwb_beta', 'redchi', 'f_host_4200',
        'alpha_lambda',
        'eta_A1', 'eta_A2', 'eta_tau1', 'eta_tau2'
    ]) if col in df_agn.columns]

    n_keys = len(keys)
    n_cols = 4
    n_rows = math.ceil(n_keys / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    # Collect scatter plots for colorbar
    scatters = []

    for idx, key in enumerate(keys):
        ax = axes[idx]
        try:
            y = df_agn.loc[mask, key]
            if np.issubdtype(y.dtype, np.number) and len(y) == np.sum(mask):
                sc = ax.scatter(y, residuals[mask], c=df_agn.loc[mask, 'z'], cmap='viridis', s=10, alpha=0.5)
                scatters.append(sc)
                ax.axhline(0, color='red', linestyle='--', lw=1)
                ax.set_xlabel(key)
                ax.set_ylabel('Residuals')
                # Add colorbar to the right of each pane
                cbar = fig.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
                cbar.set_label('Redshift', fontsize=12)
                if key.upper() == 'LOGL2500':
                    ax.set_xlim(left=0)
                if key == 'f_host_4200':
                    ax.set_xlim(left=-1.1, right=2.5)
            else:
                print(f"Skipping non-numeric or mismatched data for key: {key}")
                ax.axis('off')
        except Exception as e:
            print(f"Error processing key {key}: {e}")
            ax.axis('off')
        ax.set_title(key)
        ax.grid(True)

    for j in range(n_keys, len(axes)):
        axes[j].axis('off')
    # Add a horizontal colorbar for redshift at the top

    plt.tight_layout()
    if show:
        plt.show()

    os.makedirs("plots/hubble", exist_ok=True)
    plt.savefig("plots/hubble/full_residuals.png", dpi=300)
    plt.close()



def plot_predicted_L2500_vs_sigmahat(flat_samples, df_agn, cosmo_model, z_agn_pivot, show=False):
    d = df_agn.copy()
    #d = d[d['LOGL2500_ERR'] > 0]

    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    param_indices = {name: model_labels.index(name) for name in model_labels}

    predicted_logL2500_samples = []
    predicted_logL2500_err_samples = []

    log_sigma0_grid = np.linspace(d['log_sigma0'].min()-0.5, d['log_sigma0'].max()+0.5, 100)
    for log_sigma0 in log_sigma0_grid:
        for s in flat_samples:
            sample_params = {
                'M0_agn': s[param_indices['M0_agn']],
                'alpha_agn': s[param_indices['alpha_agn']],
                'beta_agn': s[param_indices['beta_agn']],
            }

            predicted_M2500 = M_model_agn(
                sample_params['M0_agn'],
                sample_params['alpha_agn'], sample_params['beta_agn'],
                log_sigma0, d['log_tau_UV_RF'].mean(),
                d['f_host'].mean()
            )
            predicted_logL2500 = -0.4 * (predicted_M2500 - 90) #* np.log10(np.e)  # log10(L)
            predicted_logL2500_samples.append(predicted_logL2500)

            predicted_M2500_err = M_model_agn_err(
                sample_params['M0_agn'],
                sample_params['alpha_agn'], sample_params['beta_agn'],
                log_sigma0, d['log_tau_UV_RF_err'].mean(), d['log_sigma0_err'].mean(),
                d['f_host_err'].mean()
            )
            predicted_logL2500_err = -0.4 * predicted_M2500_err #* np.log10(np.e)  # log10(L)
            predicted_logL2500_err_samples.append(predicted_logL2500_err)

    predicted_logL2500_samples = np.array(predicted_logL2500_samples)
    predicted_logL2500_samples = predicted_logL2500_samples.reshape(len(log_sigma0_grid), -1)

    predicted_logL2500_median = np.median(predicted_logL2500_samples, axis=1)
    predicted_logL2500_low = np.percentile(predicted_logL2500_samples, 16, axis=1)
    predicted_logL2500_high = np.percentile(predicted_logL2500_samples, 84, axis=1)

    results = {key: np.percentile(flat_samples[:, i], [16, 50, 84]) for i, key in enumerate(model_labels)}
    if cosmo_model == 'FlatwCDM':
        cosmo = FlatwCDM(H0=results['H0'][1], Om0=results['Om0'][1], w0=results['w0'][1])
    elif cosmo_model == 'Flatw0waCDM':
        cosmo = FlatwpwaCDM(H0=results['H0'][1], Om0=results['Om0'][1], wp=results['wp'][1], wa=results['wa'][1], zp=z_agn_pivot)
    else:
        raise ValueError(f"Unknown cosmological model: {cosmo_model}")

    actual_M2500 = d['apparent_mag_2500'] - cosmo.distmod(d['z']).value
    actual_logL2500 = -0.4 * (actual_M2500 - 90) #* np.log10(np.e)  # log10(L)

    log_sigma0 = d['log_sigma0']

    # Interpolate model at data points for residuals (in log space)
    interp_model = interp1d(log_sigma0_grid, predicted_logL2500_median, bounds_error=False, fill_value='extrapolate')
    model_logL2500_at_data = interp_model(d['log_sigma0'])
    residuals = actual_logL2500 - model_logL2500_at_data

    if False and 'LOGL2500_ERR' in d.columns:
        obs_err = d['LOGL2500_ERR']
    else:
        interp_low = interp1d(log_sigma0_grid, predicted_logL2500_low, bounds_error=False, fill_value='extrapolate')
        interp_high = interp1d(log_sigma0_grid, predicted_logL2500_high, bounds_error=False, fill_value='extrapolate')
        obs_err = 0.5 * (interp_high(d['log_sigma0']) - interp_low(d['log_sigma0']))
    
    sigma0 = 10**(d['log_sigma0'])
    sigma0_err = np.log(10) * sigma0 * d['log_sigma0_err']
    dlogL_dlog_sigma0 = np.gradient(model_logL2500_at_data, d['log_sigma0'])
    propagated_err = np.abs(dlogL_dlog_sigma0) * d['log_sigma0_err']
    total_err = np.sqrt(obs_err**2 + propagated_err**2)
    
    chi2 = np.sum((residuals / total_err)**2)
    dof = len(actual_logL2500) - len(model_labels)
    reduced_chi2 = chi2 / dof
    print(f"Reduced chi^2: {reduced_chi2:.3f}")

    color = 'm'
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax)

    # Plot in linear space
    ax.scatter(sigma0, 10**actual_logL2500, label='Data',
               s=10, alpha=0.7, edgecolor='k', facecolor='k', lw=0.5, 
               zorder=-8)
    ax.errorbar(
        sigma0,
        10**actual_logL2500,
        xerr=sigma0_err,
        #yerr=10**actual_logL2500 * d['LOGL2500_ERR'] * np.log(10),
        fmt='none', alpha=0.2, lw=1.5, capsize=3, capthick=1, color='k',
        zorder=-9)
    ax.plot(10**(log_sigma0_grid), 10**predicted_logL2500_median, color=color, zorder=-4)
    ax.fill_between(10**(log_sigma0_grid), 10**predicted_logL2500_low, 10**predicted_logL2500_high, color=color, alpha=0.3, label='Model', zorder=-5)
    ax.set_ylabel(r'$L_{2500}$ (erg s$^{-1})$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    #ax.set_xlim(4e-4, 2)
    #ax.set_ylim(6e43, 6e46)
    ax.legend()
    ax.text(
        0.05, 0.95,
        fr"Reduced $\chi^2$ = {reduced_chi2:.2f}",
        transform=ax.transAxes,
        fontsize=16,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=1)
    )
    ax.annotate(
        "Model: " + cosmo_model,
        xy=(0.5, -0.13), xycoords='axes fraction',
        ha='center', va='top', fontsize=14
    )

    # Residuals plot (still in log space)
    ax_res.scatter(sigma0, residuals, 
                   s=10, alpha=0.7, edgecolor='k', facecolor='k', lw=0.5, 
                   zorder=-10)
    ax_res.errorbar(sigma0, residuals, yerr=total_err, 
                    fmt='none', alpha=0.2, lw=1.5, capsize=3, capthick=1, color='k',
                    zorder=-11)
    ax_res.axhline(0, color=color, linestyle='--', zorder=-9)
    ax_res.set_ylabel('Residuals (log)')
    ax_res.set_xlabel(r'$\sigma_0$ (mag)')
    ax_res.set_xscale('log')
    ax_res.set_ylim(-2.2, 2.2)

    plt.setp(ax.get_xticklabels(), visible=False)
    plt.savefig(f"plots/hubble/predicted_L2500_vs_sigmahat_{cosmo_model}.png", dpi=300)
    if show:
        plt.show()
    plt.close()