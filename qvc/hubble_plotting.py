import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from tqdm import tqdm
import math
import corner
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.cosmology import FlatwCDM, FlatwpwaCDM, FlatLambdaCDM, Flatw0waCDM
import matplotlib.pyplot as plt
import os
import copy
import matplotlib.transforms as mtransforms

from hubble_model import (M_model_agn, M_model_agn_err, get_model_params, agn_model_pack_params,
    agn_model_pack_obs, agn_model_eidx, agn_model_oidx, agn_model_pidx)
from hubble_utils import *
from hubble_completeness import make_dm_function
from dynesty.utils import resample_equal
from tqdm import tqdm
from dynesty import plotting as dyplot

def plot_dynesty(results, cosmo_model, plot_path="plots/hubble", show=False):
    """
    Plot dynesty diagnostics: runplot, traceplot, and cornerpoints using dyplot.
    Saves figures to files with the given basename.
    """

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)

    # Cornerplot
    fig_corner, axes_corner = dyplot.cornerplot(results, labels=model_labels_latex, quantiles=[0.16, 0.5, 0.84],
                                                 quantiles_2d = [0.393, 0.865, 0.989],
                                                 show_titles=True, title_quantiles=[0.16, 0.5, 0.84],
                                                 color='blue',
                                                 #fig=plt.subplots(1, 1, figsize=(10, 2.5 * len(model_labels))))
    )
    fig_corner.savefig(f"{plot_path}/cornerplot.png", dpi=100)
    if show:
        plt.show()    
    plt.close(fig_corner)

    # Traceplot
    fig_trace, axes_trace = dyplot.traceplot(
        results,
        labels=model_labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_quantiles=[0.16, 0.5, 0.84],
    )
    fig_trace.tight_layout(pad=2.0, h_pad=1)

    fig_trace.savefig(f"{plot_path}/traceplot.png", dpi=100)
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


def plot_cosmo_corner(
    flat_samples_sn,
    flat_samples_agn,
    cosmo_model,
    z_pivot_sna,
    z_pivot_agn,
    plot_path='plots/hubble',
    show=False,
):
    """
    Corner-style plot (custom) of key cosmology params with diagonal stats labels.
      - Flatw0waCDM: H0, Om0, w0(= w(a=1) derived from wp,wa at z_pivot_agn), wa
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

    def _subset(samples, labels, z_pivot):
        """Return reduced samples (N,k) and latex labels for the chosen model."""
        X = np.asarray(samples)
        i_H0  = _find(labels, "H0", "H_0")
        i_Om0 = _find(labels, "Om0", "OmegaM", "Omega_m")

        if cosmo_model == "FlatwpwaCDM":
            i_wp = _find(labels, "wp", "w_p")
            i_wa = _find(labels, "wa", "w_a")
            a_p  = 1.0 / (1.0 + float(z_pivot))
            wp, wa = X[:, i_wp], X[:, i_wa]
            w0 = wp - (1.0 - a_p) * wa
            Y = np.column_stack([X[:, i_H0], X[:, i_Om0], w0, wa])
            lab_latex = [r"$H_0$", r"$\Omega_m$", r"$w_0$", r"$w_a$"]
        elif cosmo_model == "Flatw0waCDM":
            i_w0 = _find(labels, "w0", "w_0")
            i_wa = _find(labels, "wa", "w_a")
            w0, wa = X[:, i_w0], X[:, i_wa]
            Y = np.column_stack([X[:, i_H0], X[:, i_Om0], w0, wa])
            lab_latex = [r"$H_0$", r"$\Omega_m$", r"$w_0$", r"$w_a$"]
        elif cosmo_model == "FlatwCDM":
            i_w0 = _find(labels, "w0", "w_0", "w")
            Y = np.column_stack([X[:, i_H0], X[:, i_Om0], X[:, i_w0]])
            lab_latex = [r"$H_0$", r"$\Omega_m$", r"$w_0$"]
        elif cosmo_model == 'FlatLambdaCDM':
            Y = np.column_stack([X[:, i_H0], X[:, i_Om0]])
            lab_latex = [r"$H_0$", r"$\Omega_m$"]
        else:
            raise ValueError(f"Unsupported cosmo_model '{cosmo_model}' for this plot.")
        return Y, lab_latex

    def _fmt_err(m, lo, hi, latex_label=""):
        nd = 2
        if latex_label == r"$H_0$":
            nd = 1
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
    agn_data, labels_latex = _subset(flat_samples_agn, model_labels, z_pivot_agn)
    sna_data = None
    if flat_samples_sn is not None and len(flat_samples_sn) > 0:
        sna_data, _ = _subset(flat_samples_sn, model_labels, z_pivot_sna)

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
                ax.plot(xs, kde_r(xs), color="k", lw=1.8)

                if sna_data is not None:
                    xs_b = np.linspace(np.min(sna_data[:, i]), np.max(sna_data[:, i]), 400)
                    kde_b = gaussian_kde(sna_data[:, i])
                    ax.plot(xs_b, kde_b(xs_b), color="dodgerblue", lw=1.8)

                # diagonal titles: AGN (red) on top; SN (blue) below if present
                m, lo, hi = np.median(agn_data[:, i]), np.percentile(agn_data[:, i],16), np.percentile(agn_data[:, i],84)
                ms, ps, ns = _fmt_err(m, lo, hi, latex_label=labels_latex[i])
                txt_red = rf"{labels_latex[i]} = {ms}" + rf"$^{{+{ps}}}_{{-{ns}}}$"
                # draw both labels inside the axes so layout/tight_layout won't move/clip them
                # place text ABOVE the axes, offset in points (device-independent)
                fig = ax.figure
                off_blue = mtransforms.ScaledTranslation(0,  2/72., fig.dpi_scale_trans)   # ~2 pt above top edge
                off_red  = mtransforms.ScaledTranslation(0, 15/72., fig.dpi_scale_trans)   # red line above blue

                ax.text(0.02, 1.0, txt_red,
                        transform=ax.transAxes + off_red,
                        ha="left", va="bottom", color="k", fontsize=11, clip_on=False)                # ax.set_title(rf"{labels_latex[i]} = {ms}" + rf"$^{{+{ps}}}_{{-{ns}}}$",
                #              color="red", fontsize=11, loc="left", pad=2)

                if sna_data is not None:
                    mb, lob, hib = np.median(sna_data[:, i]), np.percentile(sna_data[:, i],16), np.percentile(sna_data[:, i],84)
                    msb, psb, nsb = _fmt_err(mb, lob, hib, latex_label=labels_latex[i])
                    txt_blue = rf"{labels_latex[i]} = {msb}" + rf"$^{{+{psb}}}_{{-{nsb}}}$"
                    ax.text(0.02, 1.0, txt_blue,
                                transform=ax.transAxes + off_blue,
                                ha="left", va="bottom", color="dodgerblue", fontsize=11, clip_on=False)

            else:
                # 2D KDEs
                if sna_data is not None:
                    _filled_kde(ax, sna_data[:, j], sna_data[:, i], "dodgerblue", base_alpha=0.4)
                _filled_kde(ax, agn_data[:, j], agn_data[:, i], "k", base_alpha=0.4)

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
        legend.append(Line2D([0],[0], color="dodgerblue", lw=4, label="SN Ia"))
    legend.append(Line2D([0],[0], color="k",  lw=4, label="SN Ia + AGN"))
    fig.legend(handles=legend, bbox_to_anchor=(0.5, 0.92), loc="upper left",
               fontsize=12, frameon=False, markerscale=1.5)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.9,
                        wspace=0.05, hspace=0.05)

    os.makedirs(plot_path, exist_ok=True)

    fig.savefig(os.path.join(plot_path, f"cosmo_corner_{cosmo_model}.png"), bbox_inches="tight", dpi=150)
    fig.savefig(os.path.join(plot_path, f"cosmo_corner_{cosmo_model}.pdf"), bbox_inches="tight", dpi=600)
    if show:
        plt.show()
    plt.close(fig)


def plot_hubble(flat_samples, df_agn, df_pantheon, cosmo_model, z_pivot_agn, plot_path="plots/hubble/",
                show_binned_agn=True,
                debias=False, dms=None, show=False, completeness=True, show_true=False, verbose=True,
                cosmo_model_samples={}):
    """
    Hubble diagram (Pantheon+-style):
      • Model line + 68% band in magenta
      • Concordance ΛCDM in black
      • SN Ia in blue
      • AGN points + error bars (solid if 0.44<=z<=3.16 else open)
      • Main: AGN binned in linear z
      • Inset: AGN binned in log z (matches inset x-scale)
    If residuals_2 is provided, the residuals panel overlays a solid line of (residuals - residuals_2).
    Returns: residuals, mu_pred_median, mu_pred_std
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from astropy.cosmology import FlatLambdaCDM, FlatwCDM, Flatw0waCDM
    from scipy.ndimage import uniform_filter1d
    # Ensure your project provides these:
    # from your_module import FlatwpwaCDM, M_model_agn, M_model_agn_err, get_model_params, make_dm_function
    # (FlatwpwaCDM expected if using 'FlatwpwaCDM')

    # --- Labels ---
    if   cosmo_model == 'FlatwCDM':      label = r"Flat$w$CDM model"
    elif cosmo_model == 'Flatw0waCDM':   label = r"Flat$w_0w_a$CDM model"
    elif cosmo_model == 'FlatLambdaCDM': label = r"Flat$\Lambda$CDM model"
    elif cosmo_model == 'FlatwpwaCDM':   label = r"Flat$w_p\!-\!w_a$CDM model"
    else:
        raise ValueError("Invalid cosmology model.")

    # --- Thinning for speed (cap to ~500 samples) ---
    n_samples = int(flat_samples.shape[0])
    thin_factor = max(1, n_samples // 500)
    flat_samples = flat_samples[::thin_factor]

    z_grid = np.linspace(1e-4, 5.2, 250)

    # --- Parameter bookkeeping ---
    _, model_labels, _ = get_model_params(cosmo_model)
    param_indices = {name: model_labels.index(name) for name in model_labels}

    # --- Small helper: μ_model(z | params) ---
    def _mu_model(model_name, params_dict, z, zp):
        if model_name == 'FlatwCDM':
            return FlatwCDM(H0=params_dict['H0'], Om0=params_dict['Om0'], w0=params_dict['w0']).distmod(z).value
        elif model_name == 'Flatw0waCDM':
            return Flatw0waCDM(H0=params_dict['H0'], Om0=params_dict['Om0'],
                               w0=params_dict['w0'], wa=params_dict['wa']).distmod(z).value
        elif model_name == 'FlatLambdaCDM':
            return FlatLambdaCDM(H0=params_dict['H0'], Om0=params_dict['Om0']).distmod(z).value
        elif model_name == 'FlatwpwaCDM':
            return FlatwpwaCDM(H0=params_dict['H0'], Om0=params_dict['Om0'],
                               wp=params_dict['wp'], wa=params_dict['wa'], zp=zp).distmod(z).value
        else:
            raise ValueError("Invalid cosmology model for _mu_model().")

    # --- Cosmology band on grid from posterior samples ---
    mu_models = np.array([
        _mu_model(
            cosmo_model,
            {k: s[param_indices[k]] for k in model_labels},
            z_grid, z_pivot_agn
        )
        for s in flat_samples
    ])
    mu_model_16th   = np.percentile(mu_models, 16, axis=0)
    mu_model_median = np.percentile(mu_models, 50, axis=0)
    mu_model_84th   = np.percentile(mu_models, 84, axis=0)

    # Median params (also used later)
    results = {key: np.median(flat_samples[:, i]) for i, key in enumerate(model_labels)}

    # --- Predicted AGN μ per object ---
    m_obs = df_agn['apparent_mag_2500'].values
    mu_pred_samples = []
    for s in flat_samples:
        sample_params = {k: s[param_indices[k]] for k in model_labels}
        agn_params_arr = agn_model_pack_params(sample_params)
        agn_obs_arr, agn_err_arr, agn_pivot_arr = agn_model_pack_obs(df_agn)

        predicted_M2500 = M_model_agn(agn_params_arr, agn_obs_arr, agn_pivot_arr)
        predicted_M2500_err = M_model_agn_err(agn_params_arr, agn_obs_arr, agn_err_arr, agn_pivot_arr)
        mu_pred_samples.append(m_obs - predicted_M2500)
    mu_pred_samples = np.array(mu_pred_samples)

    # De-bias (assumes your make_dm_function clips to grid, no extrapolation)
    if debias:
        dm_interp = make_dm_function(m_obs, df_agn['z'], dms, method='linear')
        pts = np.column_stack([df_agn['z'].values, m_obs])
        mu_pred_samples -= dm_interp(pts)

    mu_pred_median = np.percentile(mu_pred_samples, 50, axis=0)
    mu_pred_16th   = np.percentile(mu_pred_samples, 16, axis=0)
    mu_pred_84th   = np.percentile(mu_pred_samples, 84, axis=0)

    # Per-object uncertainty (for yerr)
    agn_params_arr = agn_model_pack_params(results)
    agn_obs_arr, agn_err_arr, agn_pivot_arr = agn_model_pack_obs(df_agn)
    predicted_M2500 = M_model_agn(agn_params_arr, agn_obs_arr, agn_pivot_arr)
    predicted_M2500_err = M_model_agn_err(agn_params_arr, agn_obs_arr, agn_err_arr, agn_pivot_arr)

    mu_pred_std = np.sqrt(
        df_agn['apparent_mag_2500_err'].values**2 +
        (0.055 * df_agn["z"].values)**2 +
        predicted_M2500_err**2
    )

    # Residuals (vs. median μ_model)
    mu_interp = np.interp(df_agn["z"].values, z_grid, mu_model_median)
    residuals = mu_pred_median - mu_interp

    # ----------------- BINNING -----------------
    def _weighted_bin_stats(z, y, w, bins, *, min_count=3, center='weighted'):
        idx = np.digitize(z, bins)  # (bins[i-1], bins[i]]
        zs, means, sems = [], [], []
        for i in range(1, len(bins)):
            m = (idx == i)
            if m.sum() >= min_count:
                wi, yi, zi = w[m], y[m], z[m]
                wsum = wi.sum()
                means.append(np.sum(wi * yi) / wsum)
                sems.append(np.sqrt(1.0 / wsum))  # SE of weighted mean
                if center == 'weighted':
                    zs.append(np.average(zi, weights=wi))
                elif center == 'geom':
                    zs.append(np.sqrt(bins[i-1] * bins[i]))
                else:
                    zs.append(0.5 * (bins[i-1] + bins[i]))
        return np.asarray(zs), np.asarray(means), np.asarray(sems)

    # Weights for AGN
    w = 1.0 / np.square(mu_pred_std)

    # Linear-z bins for MAIN & RESIDUALS panel
    dz   = 0.2
    zmax_main = np.max(df_agn["z"].values)
    bins_linear = np.arange(0.05, zmax_main + dz + 1e-9, dz)
    z_lin, mu_lin_mean, mu_lin_sem = _weighted_bin_stats(
        df_agn["z"].values, mu_pred_median, w, bins_linear, min_count=3, center='weighted'
    )

    # NEW: binned residuals (linear-z), used in residual panel
    z_res_lin, resid_lin_mean, resid_lin_sem = _weighted_bin_stats(
        df_agn["z"].values, residuals, w, bins_linear, min_count=3, center='weighted'
    )

    # Log-z bins for INSET (match inset xscale='log')
    zpos = df_agn["z"].values[df_agn["z"].values > 0]
    zmin_inset = max(0.02, float(np.min(zpos))) if zpos.size else 0.02
    zmax_inset = 3.8
    bins_per_decade = 6
    decades = np.log10(zmax_inset) - np.log10(zmin_inset)
    n_bins_log = max(1, int(np.ceil(decades * bins_per_decade)))
    bins_log = np.logspace(np.log10(zmin_inset), np.log10(zmax_inset), n_bins_log + 1)
    z_log, mu_log_mean, mu_log_sem = _weighted_bin_stats(
        df_agn["z"].values, mu_pred_median, w, bins_log, min_count=3, center='weighted'
    )

    # ======== Plot ========
    fig = plt.figure(figsize=(9, 7))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    ax.set_ylim(26, 51)
    ax.set_xlim(-0.2, np.max(df_agn["z"].values) + 0.3)
    inset_ax = inset_axes(ax, width="40%", height="40%", loc="lower right", borderpad=1.5)
    ax_resid = fig.add_subplot(gs[1], sharex=ax)

    # Solid vs open AGN markers by z (both main and inset)
    mask_in  = df_agn["z"].between(0.44, 3.16)
    mask_out = ~mask_in

    # ---------- Inset (log-z) ----------
    inset_ax.set_xscale('log')

    # AGN (inside)
    inset_ax.errorbar(
        df_agn["z"][mask_in], mu_pred_median[mask_in], yerr=mu_pred_std[mask_in],
        fmt='o', linestyle='none', markersize=2,
        mfc="black", mec="none",
        ecolor="#666666", elinewidth=0.8,
        alpha=0.7, zorder=1, label="AGN"
    )
    # AGN (outside, open)
    inset_ax.errorbar(
        df_agn["z"][mask_out], mu_pred_median[mask_out], yerr=mu_pred_std[mask_out],
        fmt='o', linestyle='none', markersize=2, mfc='none', mec="k", alpha=0.70,
        ecolor="#666666", elinewidth=0.8, zorder=1
    )

    # INSET: log-binned AGN
    if show_binned_agn:
        inset_ax.errorbar(
            z_log, mu_log_mean, yerr=mu_log_sem,
            fmt='o', linestyle='none',
            markersize=3, mfc='red', mec='none',
            ecolor='red', elinewidth=2.2, capsize=3.5,
            alpha=0.98, zorder=14, label="AGN (z-binned, log)"
        )

    # SN Ia
    inset_ax.errorbar(
        df_pantheon["zHD"], df_pantheon["MU_SH0ES"], yerr=df_pantheon["MU_SH0ES_ERR_DIAG"],
        fmt='s', markersize=2, color="#0A84FF", linestyle='none', lw=0.8, alpha=0.7, zorder=1, label="SN Ia"
    )

    # Model + band
    inset_ax.plot(z_grid, mu_model_median, color="m", lw=1.4, alpha=1.0, zorder=5, label=label)
    inset_ax.fill_between(z_grid, mu_model_16th, mu_model_84th, color="m", alpha=0.22, zorder=4)

    # Flat Lambda CDM
    # mu_conc = FlatLambdaCDM(H0=70, Om0=0.3).distmod(z_grid).value
    # inset_ax.plot(z_grid, mu_conc, color="#F0B000", lw=1.2, ls='--', zorder=5, alpha=1.0, label=r"Concordance $\Lambda$CDM")


    inset_ax.set_xlim(0.02, 5.2)
    inset_ax.set_ylim(32, 51)
    inset_ax.set_xlabel(r"$z$", fontsize=12, labelpad=-10)
    inset_ax.set_ylabel(r"$\mu$ (mag)", fontsize=12)
    inset_ax.tick_params(axis='both', which='major', labelsize=10)

    # ---------- Main plot ----------
    # AGN (inside)
    ax.errorbar(
        df_agn["z"][mask_in], mu_pred_median[mask_in], yerr=mu_pred_std[mask_in],
        fmt='o', linestyle='none', markersize=3,
        mfc="black", mec="none",
        ecolor="#666666", elinewidth=1.1,
        alpha=0.3, zorder=0, label="AGN"
    )
    # AGN (outside, open)
    ax.errorbar(
        df_agn["z"][mask_out], mu_pred_median[mask_out], yerr=mu_pred_std[mask_out],
        fmt='o', linestyle='none', markersize=3, mfc='none', mec="k", alpha=0.3,
        ecolor="#666666", elinewidth=1.1, zorder=0
    )

    # MAIN: linear-binned AGN
    if show_binned_agn:
        ax.errorbar(
            z_lin, mu_lin_mean, yerr=mu_lin_sem,
            fmt='o', linestyle='none',
            markersize=5, mfc='red', mec='none',
            ecolor='red', elinewidth=2.2, capsize=3.5,
            alpha=0.98, zorder=14, label="AGN (z-binned)"
        )

    # SN Ia
    ax.errorbar(
        df_pantheon["zHD"], df_pantheon["MU_SH0ES"], yerr=df_pantheon["MU_SH0ES_ERR_DIAG"],
        fmt='s', markersize=2, color="#0A84FF", linestyle='none', lw=0.8, alpha=0.7, zorder=1, label="SN Ia"
    )

    # Model + 68% band
    ax.plot(z_grid, mu_model_median, color="m", lw=2.4, alpha=1.0, zorder=5, label=label)
    ax.fill_between(z_grid, mu_model_16th, mu_model_84th, color="m", alpha=0.25, zorder=4)

    # Survey magnitude limit (shade above)
    if completeness and not debias:
        agn_params_arr = agn_model_pack_params(results)
        agn_obs_med = dict(
            log_sigma_UV    = float(np.median(df_agn['log_sigma_UV'].values)) * np.ones_like(z_grid),
            log_sigma_UV_err = float(np.median(df_agn['log_sigma_UV_err'].values)) * np.ones_like(z_grid),
            log_tau_UV_RF = float(np.median(df_agn['log_tau_UV_RF'].values)) * np.ones_like(z_grid),
            log_tau_UV_RF_err = float(np.median(df_agn['log_tau_UV_RF_err'].values)) * np.ones_like(z_grid),
            alpha_nu     = float(np.median(df_agn['alpha_nu'].values)) * np.ones_like(z_grid),
            alpha_nu_err = float(np.median(df_agn['alpha_nu_err'].values)) * np.ones_like(z_grid),
            cov_log_sigma_UV_log_tau_UV_RF = float(np.median(df_agn['cov_log_sigma_UV_log_tau_UV_RF'].values)) * np.ones_like(z_grid),

            log_tau_UV_RF_std_psd = float(np.median(df_agn['log_tau_UV_RF_std_psd'].values)) * np.ones_like(z_grid),
            log_sigma_UV_std_psd = float(np.median(df_agn['log_sigma_UV_std_psd'].values)) * np.ones_like(z_grid),
            log_sigma_UV_log_tau_UV_RF_cov_psd = float(np.median(df_agn['log_sigma_UV_log_tau_UV_RF_cov_psd'].values)) * np.ones_like(z_grid),
        )
        agn_obs_arr, agn_err_arr, agn_pivot_arr = agn_model_pack_obs(agn_obs_med)

        M_med_grid = np.median([
            M_model_agn(agn_params_arr, agn_obs_arr, agn_pivot_arr)
            for s in flat_samples
        ], axis=0)

        m_lim = 24.0
        mu_lim = m_lim - M_med_grid
        ax.fill_between(z_grid, mu_lim, 60, color="red", alpha=0.15, zorder=2, label="< 50% complete")
        inset_ax.fill_between(z_grid, mu_lim, 60, color="red", alpha=0.12, zorder=2, label="< 50% complete")

    # Flat ΛCDM
    if 'FlatLambdaCDM' in cosmo_model_samples:
        flat_samples_lambda = cosmo_model_samples['FlatLambdaCDM']
        flat_samples_lambda = flat_samples_lambda[::thin_factor]

        _, model_labels_lambda, _ = get_model_params('FlatLambdaCDM')
        param_indices_lambda = {name: model_labels_lambda.index(name) for name in model_labels_lambda}
        mu_model_lambda = np.array([
            _mu_model(
                'FlatLambdaCDM',
                {k: s[param_indices_lambda[k]] for k in model_labels_lambda},
                z_grid, z_pivot_agn)
        for s in flat_samples_lambda
        ])
        mu_model_16th_lambda   = np.percentile(mu_model_lambda, 16, axis=0)
        mu_model_median_lambda = np.percentile(mu_model_lambda, 50, axis=0)
        mu_model_84th_lambda   = np.percentile(mu_model_lambda, 84, axis=0)

        ax.plot(z_grid, mu_model_median_lambda, color="#F0B000", lw=1.6, ls='--', alpha=1.0, zorder=5, label=r"Flat $\Lambda$CDM")
        #ax.fill_between(z_grid, mu_model_16th_lambda, mu_model_84th_lambda, color="#F0B000", alpha=0.25, zorder=4)
        inset_ax.plot(z_grid, mu_model_median_lambda, color="#F0B000", lw=1.2, ls='--', alpha=1.0, zorder=5)

    # Flatw0waCDM
    if 'Flatw0waCDM' in cosmo_model_samples:
        flat_samples_w0wa = cosmo_model_samples['Flatw0waCDM']
        flat_samples_w0wa = flat_samples_w0wa[::thin_factor]

        _, model_labels_w0wa, _ = get_model_params('Flatw0waCDM')
        param_indices_w0wa = {name: model_labels_w0wa.index(name) for name in model_labels_w0wa}
        mu_model_w0wa = np.array([
            _mu_model(
                'Flatw0waCDM',
                {k: s[param_indices_w0wa[k]] for k in model_labels_w0wa},
                z_grid, z_pivot_agn)
        for s in flat_samples_w0wa
        ])
        mu_model_16th_w0wa   = np.percentile(mu_model_w0wa, 16, axis=0)
        mu_model_median_w0wa = np.percentile(mu_model_w0wa, 50, axis=0)
        mu_model_84th_w0wa   = np.percentile(mu_model_w0wa, 84, axis=0)
        
        ax.plot(z_grid, mu_model_median_w0wa, color="c", ls='dotted', alpha=1.0, lw=1.6, zorder=5, label=r"Flatw$_0$w$_a$CDM")
        #ax.fill_between(z_grid, mu_model_16th_w0wa, mu_model_84th_w0wa, color="c", alpha=0.25, zorder=5)
        inset_ax.plot(z_grid, mu_model_median_w0wa, color="c", ls='dotted', lw=1.2, alpha=1.0, zorder=5)
    
    # Labels
    ax.set_ylabel(r"$\mu$ (mag)")
    ax.set_xlabel(r"$z$")

    # ---------- Residuals panel ----------
    # AGN residuals (inside)
    ax_resid.plot(
        df_agn["z"][mask_in], residuals[mask_in],
        'o', markersize=3, mfc="black", mec="none", alpha=0.3, zorder=0, label="AGN residuals"
    )
    # AGN residuals (outside)
    ax_resid.plot(
        df_agn["z"][mask_out], residuals[mask_out],
        'o', markersize=3, mfc='none', mec="k", alpha=0.3, zorder=0, label="AGN residuals (out)"
    )
    # Zero line
    ax_resid.axhline(0.0, color="m", lw=2.2, zorder=1)

    # NEW: binned residuals in red (points + thin connecting line)
    if z_res_lin.size:
        ax_resid.errorbar(
            z_res_lin, resid_lin_mean, yerr=resid_lin_sem,
            fmt='o', linestyle='none', markersize=5,
            mfc='red', mec='none', ecolor='red', elinewidth=2.0, capsize=3.0,
            alpha=0.98, zorder=15, label="Binned AGN residuals"
        )
        #ax_resid.plot(z_res_lin, resid_lin_mean, lw=1.2, color='red', alpha=0.9, zorder=14)

    # Optional: line for (residuals - residuals_2) using median params of each model
    if 'Flatw0waCDM' in cosmo_model_samples:
        z_grid_fine = np.linspace(1e-4, 5.2, 500)
        results_w0wa = {key: np.median(flat_samples_w0wa[:, i]) for i, key in enumerate(model_labels_w0wa)}

        mu_model_1 = _mu_model(cosmo_model, results,   z_grid_fine, z_pivot_agn)
        mu_model_w0wa = _mu_model('Flatw0waCDM', results_w0wa, z_grid_fine, z_pivot_agn)
        ax_resid.plot(z_grid_fine, mu_model_w0wa - mu_model_1, lw=2.2, color="c", ls='dotted', alpha=1.0, label=r"Flatw$_0$w$_a$CDM $\Delta$μ")
    if 'FlatLambdaCDM' in cosmo_model_samples:
        z_grid_fine = np.linspace(1e-4, 5.2, 500)
        results_lambda = {key: np.median(flat_samples_lambda[:, i]) for i, key in enumerate(model_labels_lambda)}

        mu_model_1 = _mu_model(cosmo_model, results,   z_grid_fine, z_pivot_agn)
        mu_model_lambda = _mu_model('FlatLambdaCDM', results_lambda, z_grid_fine, z_pivot_agn)
        ax_resid.plot(z_grid_fine, mu_model_lambda - mu_model_1, lw=2.2, color="#F0B000", ls='--', alpha=1.0,)

    ax_resid.set_ylabel(r"$\Delta\mu$ (mag)")
    ax_resid.set_xlabel(r"$z$")
    ax_resid.set_ylim(-.5, .5)
    #ax_resid.legend(frameon=True, loc="upper left", fontsize=10)

    for axi in (ax, inset_ax, ax_resid):
        axi.minorticks_on()
        axi.tick_params(axis='both', which='minor', direction='in', length=4, top=True, right=True, width=2)
        axi.tick_params(axis='both', which='major', direction='in', length=8, top=True, right=True)

    # Hide the main panel's x-axis labels, numbers, and ticks (leave residuals' x-axis intact)
    ax.set_xlabel("")  # remove main x-axis label
    ax.tick_params(axis='x', which='both', labelbottom=False, bottom=False, top=False)
    ax.xaxis.offsetText.set_visible(False)  # hide any scientific-notation offset text

    # Legend
    # Combine legends from ax and ax_resid
    # handles_main, labels_main = ax.get_legend_handles_labels()
    # handles_resid, labels_resid = ax_resid.get_legend_handles_labels()
    # handles = handles_main + handles_resid
    # labels = labels_main + labels_resid
    # ax.legend(handles, labels, frameon=False, loc="lower center", bbox_to_anchor=(0.22, 0.06), fontsize=12)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.22, 0.06), fontsize=12)

    # Save/show
    fig.tight_layout()
    os.makedirs(plot_path, exist_ok=True)
    filename = "hubble_diagram_debiased.png" if debias else "hubble_diagram.png"
    plt.savefig(os.path.join(plot_path, filename), dpi=300)
    filename = "hubble_diagram_debiased.pdf" if debias else "hubble_diagram.pdf"
    os.makedirs(os.path.join(plot_path, "pdf"), exist_ok=True)
    plt.savefig(os.path.join(plot_path, "pdf", filename), dpi=600)
    if show:
        plt.show()
    plt.close(fig)

    # Residual Outlier report
    outlier_mask = np.abs(residuals) > 4
    if np.any(outlier_mask) and verbose:
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Outliers with residuals > 4 (sorted by residual, largest last):")
        outlier_indices = np.where(outlier_mask)[0]
        # Sort indices by residual value (ascending)
        sorted_indices = outlier_indices[np.argsort(residuals[outlier_indices])]
        for idx in sorted_indices:
            sdss_name = df_agn.iloc[idx].get('sdss_name', 'Unknown')
            object_id = df_agn.iloc[idx].get('object_id', 'Unknown')
            ra  = df_agn.iloc[idx].get('ra',  np.nan)
            dec = df_agn.iloc[idx].get('dec', np.nan)
            z   = df_agn.iloc[idx]['z']
            npca_qso = df_agn.iloc[idx].get('npca_qso', 'N/A')
            print(f"\tz: {z:.2f} | object_id: {object_id} | npca_qso: {npca_qso} | SDSS: {sdss_name} | RA: {ra:.5f} | DEC: {dec:.5f} | Residual: {residuals[idx]:.1f}")
    # Save residuals to CSV under plot_path
    residuals_df = df_agn.copy()
    residuals_df["residuals"] = residuals
    residuals_df["mu_pred_median"] = mu_pred_median
    residuals_df["mu_pred_std"] = mu_pred_std
    fields = ['object_id', 'ra', 'dec', 'mu_pred_median', 'mu_pred_std', 'z', 'redchi', 'sdss_name', 'npca_qso', 'residuals']
    residuals_df = residuals_df[fields]
    residuals_df = residuals_df.sort_values(by="residuals", ascending=False)
    csv_path = os.path.join(plot_path, "residuals.csv")
    residuals_df.to_csv(csv_path, index=False)
    print(f"Residuals saved to {csv_path}")

    # Standard deviation Outlier report
    outlier_mask = mu_pred_std > 4
    if np.any(outlier_mask) and verbose:
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Outliers with mu_pred_std > 4:")
        for idx in np.where(outlier_mask)[0]:
            sdss_name = df_agn.iloc[idx].get('sdss_name', 'Unknown')
            object_id = df_agn.iloc[idx].get('object_id', 'Unknown')
            ra  = df_agn.iloc[idx].get('ra',  np.nan)
            dec = df_agn.iloc[idx].get('dec', np.nan)
            z   = df_agn.iloc[idx]['z']
            npca_qso = df_agn.iloc[idx].get('npca_qso', 'N/A')
            print(f"\tz: {z:.2f} | object_id: {object_id} | npca_qso: {npca_qso} | SDSS: {sdss_name} | RA: {ra:.5f} | DEC: {dec:.5f} | Residual: {residuals[idx]:.1f}")

    return residuals, mu_pred_median, mu_pred_std

def plot_predicted_vs_actual_M2500(
    flat_samples,
    df_agn,
    cosmo_model,
    z_pivot_agn,
    plot_path="plots/hubble",
    dms=None,  # de-biasing function (optional)
    debias=False,
    show=False,
    cmap="inferno",       # colormap for points
    box_alpha=0.7,        # transparency of white annotation boxes
    show_sigma_band=True,
    completeness=True,    # add "<50% complete" red region
    m_lim=24.0,           # survey apparent-magnitude limit for completeness shading
    n_cosmo_draws=50,    # posterior draws to propagate cosmology errors (for xerr)
    random_state=42,      # RNG seed for reproducibility of draws
):
    """
    Predicted vs Actual M_2500, with:
      • y-error bars from M_model_agn_err(...)
      • x-error bars = sqrt(apparent_mag_2500_err^2 + sigma_mu_cosmo(z)^2)
      • ±1σ band from intrinsic scatter sigma_int = exp(log_f) (magenta)
      • Points colored by alpha_nu with ONE global colorbar scale (vmin/vmax from full sample)
      • Optional "<50% complete" red region by bin.
    """
    import os, math
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.cosmology import FlatLambdaCDM, FlatwCDM, FlatwpwaCDM, Flatw0waCDM

    # --- model parameters from samples ---
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    results = {key: np.median(flat_samples[:, i])
               for i, key in enumerate(model_labels)}
    label_to_idx = {k: i for i, k in enumerate(model_labels)}

    # --- intrinsic scatter: sigma_int = exp(log_f) from posterior median ---
    sigma_intrinsic = float(np.exp(results['log_f']))

    # --- helpers to build cosmology objects ---
    def _cosmo_from_params(H0, Om0, **kw):
        if cosmo_model == "FlatwCDM":
            return FlatwCDM(H0=H0, Om0=Om0, w0=kw["w0"])
        elif cosmo_model == "FlatwpwaCDM":
            return FlatwpwaCDM(H0=H0, Om0=Om0, wp=kw["wp"], wa=kw["wa"], zp=z_pivot_agn)
        elif cosmo_model == "Flatw0waCDM":
            return Flatw0waCDM(H0=H0, Om0=Om0, w0=kw["w0"], wa=kw["wa"])
        elif cosmo_model == "FlatLambdaCDM":
            return FlatLambdaCDM(H0=H0, Om0=Om0)
        else:
            raise ValueError("Invalid cosmology model.")

    # Median cosmology for best-estimate distances
    if cosmo_model == "FlatwCDM":
        cosmo_med = _cosmo_from_params(results["H0"], results["Om0"], w0=results["w0"])
    elif cosmo_model == "FlatwpwaCDM":
        cosmo_med = _cosmo_from_params(results["H0"], results["Om0"],
                                       wp=results["wp"], wa=results["wa"], zp=z_pivot_agn)
    elif cosmo_model == "Flatw0waCDM":
        cosmo_med = _cosmo_from_params(results["H0"], results["Om0"],
                                       w0=results["w0"], wa=results["wa"])
    else:
        cosmo_med = _cosmo_from_params(results["H0"], results["Om0"])

    # --- data & predictions ---
    z = df_agn["z"].values
    m_app = df_agn["apparent_mag_2500"].values
    if "apparent_mag_2500_err" not in df_agn.columns:
        raise KeyError("df_agn must contain 'apparent_mag_2500_err' for x-error bars.")
    m_app_err = df_agn["apparent_mag_2500_err"].values

    distmod_med = np.array([cosmo_med.distmod(zi).value for zi in z])
    actual_M_2500 = m_app - distmod_med

    agn_params_arr = agn_model_pack_params(results)
    agn_obs_arr, agn_err_arr, agn_pivot_arr = agn_model_pack_obs(df_agn)

    M_2500_pred = M_model_agn(agn_params_arr, agn_obs_arr, agn_pivot_arr)
    M_2500_pred_err = M_model_agn_err(agn_params_arr, agn_obs_arr, agn_err_arr, agn_pivot_arr)
    M_2500_pred_err[~np.isfinite(M_2500_pred_err) | (M_2500_pred_err < 0)] = np.nan

    # --- x-errors: propagate cosmology posterior into distance modulus ---
    rng = np.random.default_rng(random_state)
    n_samp = flat_samples.shape[0]
    n_draws = min(n_cosmo_draws, n_samp)
    draw_idxs = rng.choice(n_samp, size=n_draws, replace=False) if n_draws < n_samp else np.arange(n_samp)

    def _cosmo_from_draw(row):
        if cosmo_model == "FlatwCDM":
            return FlatwCDM(H0=row[label_to_idx["H0"]], Om0=row[label_to_idx["Om0"]], w0=row[label_to_idx["w0"]])
        elif cosmo_model == "FlatwpwaCDM":
            return FlatwpwaCDM(H0=row[label_to_idx["H0"]],
                               Om0=row[label_to_idx["Om0"]],
                               wp=row[label_to_idx["wp"]],
                               wa=row[label_to_idx["wa"]],
                               zp=z_pivot_agn)
        elif cosmo_model == "Flatw0waCDM":
            return Flatw0waCDM(H0=row[label_to_idx["H0"]],
                               Om0=row[label_to_idx["Om0"]],
                               w0=row[label_to_idx["w0"]],
                               wa=row[label_to_idx["wa"]])
        elif cosmo_model == "FlatLambdaCDM":
            return FlatLambdaCDM(H0=row[label_to_idx["H0"]], Om0=row[label_to_idx["Om0"]])
        else:
            return FlatLambdaCDM(H0=row[label_to_idx["H0"]], Om0=row[label_to_idx["Om0"]])

    mu_draws = np.empty((n_draws, z.size), dtype=float)
    for j, idx in enumerate(draw_idxs):
        row = flat_samples[idx, :]
        cosmo_j = _cosmo_from_draw(row)
        mu_draws[j, :] = np.array([cosmo_j.distmod(zi).value for zi in z])
    sigma_mu_cosmo = np.nanstd(mu_draws, axis=0, ddof=1)  # per-object DM uncertainty
    xerr = np.sqrt(m_app_err**2 + sigma_mu_cosmo**2)

    if debias:
        dm_interp = make_dm_function(np.array(df_agn["apparent_mag_2500"].values), np.array(df_agn['z'].values), dms)

    # --- binning in redshift ---
    num_cols = 5
    num_rows = 7
    n_bins = num_cols * num_rows  # 30

    # Fixed edges for first and last
    first_edge = 0.0
    second_edge = 0.3
    last_edge = 3.7

    # How many interior bins are needed?
    n_interior = n_bins - 2  # exclude first [0–0.3] and last [4–∞]

    # Generate evenly spaced edges between 0.3 and 4.0
    interior_edges = np.linspace(second_edge, last_edge, n_interior + 1)
    interior_edges = np.arange(second_edge, last_edge, 0.1)

    # Concatenate
    z_bins = np.concatenate(([first_edge, second_edge], interior_edges[1:], [10]))

    print(len(z_bins) - 1, "bins")  # should be 30
    print(z_bins)
    z_bin_indices = np.digitize(z, bins=z_bins)
    num_bins = len(z_bins) - 1
    # Use open intervals for first and last bins
    bin_labels = []
    for i in range(num_bins):
        if i == 0:
            label = f"$z < {z_bins[i+1]:.1f}$"
        elif i == num_bins - 1:
            label = f"$z > {z_bins[i]:.1f}$"
        else:
            label = f"${z_bins[i]:.1f} < z < {z_bins[i+1]:.1f}$"
        bin_labels.append(label)

    # --- figure with full-height colorbar (dedicated column) ---
    fig = plt.figure(figsize=(5 * num_cols, 4 * num_rows))
    gs = fig.add_gridspec(num_rows, num_cols + 1,
                          width_ratios=[1]*num_cols + [0.06],
                          wspace=0.0, hspace=0.0)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(num_cols)] for r in range(num_rows)]).flatten()
    

    # --- axis helpers ---
    xlo, xhi = -25.8, -18.2
    ylo, yhi = -25.8, -18.2
    xx = np.linspace(min(xlo, ylo), max(xhi, yhi), 400)

    for i, ax in enumerate(axes):
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)

        if i >= num_bins:
            ax.axis("off"); continue

        bin_mask = z_bin_indices == (i + 1)
        if not np.any(bin_mask):
            ax.axis("off"); continue

        actual_M_2500_bin = actual_M_2500[bin_mask].copy()
        if debias:
            pts = np.column_stack([df_agn['z'][bin_mask], df_agn['apparent_mag_2500'][bin_mask]])
            actual_M_2500_bin -= dm_interp(pts)

        x = actual_M_2500_bin
        y = M_2500_pred[bin_mask]
        xerr_bin = xerr[bin_mask]
        yerr_bin = M_2500_pred_err[bin_mask]


        # residuals & coverage vs intrinsic sigma
        resid = y - x
        frac1 = 100.0 * np.mean(np.abs(resid) <= 1.0 * sigma_intrinsic)
        frac2 = 100.0 * np.mean(np.abs(resid) <= 2.0 * sigma_intrinsic)
        frac3 = 100.0 * np.mean(np.abs(resid) <= 3.0 * sigma_intrinsic)
        rms_resid = float(np.sqrt(np.nanmean(resid**2))) if resid.size else np.nan

        mask_in  = df_agn[bin_mask]["z"].between(0.44, 3.16)
        mask_out = ~mask_in

        
        ax.errorbar(
            x, y, xerr=xerr_bin, yerr=yerr_bin,
            fmt="none", ecolor="#666666", elinewidth=0.7, alpha=0.4, zorder=2
        )

        # AGN (inside)
        ax.scatter(x[mask_in], y[mask_in],
                  c='k', s=20, alpha=0.9, edgecolors="k", zorder=3
        )
        # AGN (outside)
        ax.scatter(x[mask_out], y[mask_out],
                  c='none', s=20, alpha=0.9, edgecolors="k", zorder=3
        )

        # y = x reference and ±1σ intrinsic band
        ax.plot(xx, xx, color="m", alpha=0.9, lw=2.2, zorder=9)
        if show_sigma_band:
            ax.plot(xx, xx - sigma_intrinsic, color="m", alpha=0.7, lw=1.5, linestyle="--", zorder=9, label=r"$y = x \pm 1\sigma_{\rm int}$" if i == 0 else None)
            ax.plot(xx, xx + sigma_intrinsic, color="m", alpha=0.7, lw=1.5, linestyle="--", zorder=9)

        # "< 50% complete" region
        if completeness and not debias:
            z_center = 0.5 * (z_bins[i] + z_bins[i+1])
            mu_center = cosmo_med.distmod(z_center).value
            M_lim = m_lim - mu_center
            xmin = max(M_lim, xlo)
            xmax = xhi
            if xmin < xmax:
                ax.axvspan(xmin, xmax, facecolor="red", alpha=0.15, zorder=0, label="< 50% complete")

        # cosmetics
        ax.invert_xaxis()
        ax.invert_yaxis()

        # annotations
        boxprops = dict(boxstyle="round,pad=0.2", facecolor="white", alpha=box_alpha, edgecolor="none")
        ax.annotate(
            bin_labels[i], xy=(0.03, 0.97), xycoords="axes fraction",
            fontsize=14, color="k", ha="left", va="top", bbox=boxprops,
        )
        n_in_bin = int(np.sum(bin_mask))
        ax.annotate(
            f"N = {n_in_bin}", xy=(0.97, 0.03), xycoords="axes fraction",
            fontsize=11, color="gray", ha="right", va="bottom", bbox=boxprops,
        )

        # labels only on bottom row / left col
        num_panels = num_rows * num_cols
        if i >= (num_rows - 1) * num_cols:
            ax.set_xlabel("Actual $M_{2500}$", fontsize=12)
        if i % num_cols == 0:
            ax.set_ylabel("Predicted $M_{2500}$", fontsize=12)
        ax.tick_params(axis="both", labelsize=10, length=3)

        # show legend once
        if (show_sigma_band or completeness) and i == num_cols-1:
            leg = ax.legend(loc="upper right", fontsize=14, frameon=True)
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_alpha(box_alpha)
            leg.get_frame().set_edgecolor("none")

    # full-height colorbar (global scale)
    # if sc_for_cbar is not None:
    # cax = fig.add_subplot(gs[:, -1])
    #     cbar = fig.colorbar(sc_for_cbar, cax=cax, orientation="vertical")
    #     cbar.set_label(r"$\alpha_{\nu}$", fontsize=12)
    #     cbar.ax.tick_params(labelsize=10)

    for ax in axes:
        if ax.has_data():
            ax.label_outer()

    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(os.path.join(plot_path, "pdf"), exist_ok=True)

    plt.savefig(os.path.join(plot_path, f"predicted_vs_actual_M2500{'_debias' if debias else ''}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(plot_path, "pdf", f"predicted_vs_actual_M2500{'_debias' if debias else ''}.pdf"), dpi=600, bbox_inches="tight")
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
    os.makedirs("plots/completeness", exist_ok=True)
    plt.savefig("plots/completeness/completeness_vs_mag_at_redshifts.png", dpi=300)
    #plt.savefig("plots/hubble/completeness_vs_mag_at_redshifts.pdf", dpi=300)
    plt.close()



def plot_full_residuals(df_agn, residuals, flat_samples, cosmo_model, z_pivot_agn, debias=False, dms=None, plot_path='plots/hubble', show=False):
    import math
    df_agn = df_agn.copy()
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    param_indices = {name: model_labels.index(name) for name in model_labels}

    results = {key: np.percentile(flat_samples[:, i], [16, 50, 84]) for i, key in enumerate(model_labels)}

    if cosmo_model == 'FlatwCDM':
        cosmo = FlatwCDM(
            H0=results['H0'][1],
            Om0=results['Om0'][1],
            w0=results['w0'][1]
        )
    elif cosmo_model == 'FlatwpwaCDM':
        cosmo = FlatwpwaCDM(
            H0=results['H0'][1],
            Om0=results['Om0'][1],
            wp=results['wp'][1],
            wa=results['wa'][1],
            zp=z_pivot_agn,
        )
    elif cosmo_model == 'Flatw0waCDM':
        cosmo = Flatw0waCDM(
            H0=results['H0'][1],
            Om0=results['Om0'][1],
            w0=results['w0'][1],
            wa=results['wa'][1]
        )
    elif cosmo_model == 'FlatLambdaCDM':
        cosmo = FlatLambdaCDM(
            H0=results['H0'][1],
            Om0=results['Om0'][1]
        )
    else:
        raise ValueError("Invalid cosmology model.")
    
    df_agn = df_agn.copy().reset_index(drop=True)

    df_agn['MY_M_2500'] = df_agn['apparent_mag_2500'].values - cosmo.distmod(df_agn['z'].values).value

    if debias:
        print(dms)
        dm_interp = make_dm_function(df_agn["apparent_mag_2500"].values, df_agn['z'].values, dms)
        pts = np.column_stack([df_agn['z'], df_agn['apparent_mag_2500']])
        df_agn['MY_M_2500'] -= dm_interp(pts)
        df_agn['apparent_mag_2500'] -= dm_interp(pts)


    # Select only the keys in your specified list (order preserved by np.flip)
    keys = [col for col in np.flip([
        'apparent_mag_2500', 'MY_M_2500', 'z', 'log_lbol', 'log_ledd_ratio', 
        'log_sigma_UV', 'log_sigma_hat0', 'log_sigma_hat_UV', 'log_tau_UV_RF', 'chi_sq_g',
        'bwb_beta', 'sn_median_all', 'bwb_alpha', 'bwb_beta',
        'redchi', 'bwb_beta_4200', 'alpha_lambda', 'alpha_nu', 
        'log_f_host_5100', 'log_f_host_4200',
        'zWarning', 'sameZ', 'class_code', 'subClass_code',
        'log_rho', 't_rf_length', 'tau_band_RF_mean',
        'log_tau_band_RF_mean', 'log_t_rf_length',
        'eta_A1', 'eta_A2', 'eta_tau1', 'eta_tau2',
    ]) if col in df_agn.columns]

    keys_masks = {
        'f_host_5100': (0, np.inf),
        'f_host_2500': (0, np.inf),
        'f_host_4200': (0, 2),
        'log_lbol': (1, np.inf),
    }

    keys_yx_line = ['MY_M_2500', 'apparent_mag_2500']

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
            # Exclude non-numeric columns from -1e9 check
            if np.issubdtype(df_agn[key].dtype, np.number):
                mask = (df_agn[key] > -1e9) & (~df_agn[key].isna())
            else:
                mask = np.ones(len(df_agn), dtype=bool)
            if key in keys_masks:
                low, high = keys_masks[key]
                mask &= df_agn[key].between(low, high)
            y = df_agn.loc[mask, key]
            if np.issubdtype(y.dtype, np.number) and len(y) == np.sum(mask):
                sc = ax.scatter(y, residuals[mask], c=df_agn.loc[mask, 'z'], cmap='viridis', s=10, alpha=0.5)
                scatters.append(sc)
                if key in keys_yx_line:
                    ax.plot([np.nanmin(y), np.nanmax(y)], [np.nanmin(y)-np.nanmean(y), np.nanmax(y)-np.nanmean(y)], color='red', linestyle='--', lw=1)
                ax.axhline(0, color='red', linestyle='--', lw=1)
                ax.set_xlabel(key)
                ax.set_ylabel('Residuals')
                # Add colorbar to the right of each pane
                cbar = fig.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
                cbar.set_label('Redshift', fontsize=12)
                if key.upper() == 'LOGL2500':
                    ax.set_xlim(left=0)
                if key == 'bwb_beta_4200':
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

    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path, f"full_residuals_{'debiased' if debias else 'biased'}.png"), dpi=300)
    plt.close()


from scipy.interpolate import interp1d
from matplotlib.ticker import LogLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec

def plot_predicted_L2500_vs_sigmahat(
    flat_samples, df_agn, cosmo_model, z_pivot_agn,
    plot_path='plots/hubble', show=False, debias=True, dms=None,
    show_residuals=False
):
    d = df_agn.copy()

    # --- Thinning for speed ---
    n_samples = int(flat_samples.shape[0])
    thin_factor = max(1, n_samples // 500)
    flat_samples = flat_samples[::thin_factor]

    # --- Indices & parameter names ---
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    param_indices = {name: model_labels.index(name) for name in model_labels}

    # --- Pack obs/errs/pivots once ---
    agn_obs_arr, agn_err_arr, agn_pivot_arr = agn_model_pack_obs(d)

    # Helper: posterior median dict
    med_params = {k: np.median(flat_samples[:, param_indices[k]]) for k in model_labels}

    # --- Cosmology from medians (only for placing the *data* on y) ---
    if cosmo_model == 'FlatwCDM':
        cosmo = FlatwCDM(H0=med_params['H0'], Om0=med_params['Om0'], w0=med_params['w0'])
    elif cosmo_model == 'FlatwpwaCDM':
        cosmo = FlatwpwaCDM(H0=med_params['H0'], Om0=med_params['Om0'],
                            wp=med_params['wp'], wa=med_params['wa'], zp=z_pivot_agn)
    elif cosmo_model == 'Flatw0waCDM':
        cosmo = Flatw0waCDM(H0=med_params['H0'], Om0=med_params['Om0'],
                            w0=med_params['w0'], wa=med_params['wa'])
    elif cosmo_model == 'FlatLambdaCDM':
        cosmo = FlatLambdaCDM(H0=med_params['H0'], Om0=med_params['Om0'])
    else:
        raise ValueError(f"Unknown cosmological model: {cosmo_model}")

    # --- y-data: log10 L_2500 (with optional de-bias of apparent magnitudes) ---
    if debias:
        dm_interp = make_dm_function(d["apparent_mag_2500"].values, d['z'].values, dms)
        pts = np.column_stack([d['z'], d['apparent_mag_2500']])
        actual_M2500 = (d['apparent_mag_2500'] - dm_interp(pts)) - cosmo.distmod(d['z']).value
    else:
        actual_M2500 = d['apparent_mag_2500'] - cosmo.distmod(d['z']).value
    actual_logL2500 = convert_M2500_to_logL2500(actual_M2500)

    # y measurement uncertainty propagated to log10 L
    y_log_meas_err = 0.4 * np.asarray(d['apparent_mag_2500_err'].fillna(0.0))

    # --- Reference x (built at POSTERIOR-MEDIAN params) ---
    # This is the "common" axis we’ll plot against. It fixes the x definition,
    # then each posterior draw is *refit* onto this axis (slope+intercept).
    med_arr = agn_model_pack_params(med_params)
    M0_med = med_arr[agn_model_pidx["M0_agn"]]
    # predicted_M_ref = (M_model - M0) evaluated at median params → this is the "x in M-space"
    x_log_ref = M_model_agn(med_arr, agn_obs_arr, agn_pivot_arr) - M0_med
    x_ref = 10.0 ** x_log_ref

    # --- x for the plotted points (computed at med params for consistency with x_ref) ---
    x_data = x_ref
    # Propagate x errors from median params (same as before)
    pred_M_err_med = M_model_agn_err(med_arr, agn_obs_arr, agn_err_arr, agn_pivot_arr)
    x_lower = 10.0 ** (x_log_ref - pred_M_err_med)
    x_upper = 10.0 ** (x_log_ref + pred_M_err_med)
    xerr_asym = np.vstack((x_data - np.maximum(x_lower, 1e-300),
                           np.maximum(x_upper, x_data) - x_data))

    # Center of mass in the *reference* x (we’ll use this for plotting & band widening)
    xcm = np.mean(x_log_ref)
    var_x = np.var(x_log_ref, ddof=1)
    if not np.isfinite(var_x) or var_x <= 0:
        # Fallback (shouldn't normally happen)
        var_x = np.var(x_log_ref) + 1e-8

    # --- Grid in reference x (log space), padded a bit beyond the data ---
    # include lowest left error and highest right error
    x_min_err = np.min(x_log_ref - pred_M_err_med)
    x_max_err = np.max(x_log_ref + pred_M_err_med)

    x_lo = x_min_err - 0.20
    x_hi = x_max_err + 0.20

    x_log_grid = np.linspace(x_lo, x_hi, 250)
    x_grid = 10.0 ** x_log_grid

    # ---------- KEY CHANGE: build distribution in M first, *per sample*, then map to L ----------
    # For each posterior sample:
    #   1) Compute the sample's predicted M_i for every object:  M_i^s = M_model_agn(s)  (full model)
    #   2) Regress M_i^s against *reference* x_log_ref: M_i^s ≈ c_s + k_s * x_log_ref   (OLS)
    #   3) Predict M on the grid: M_grid^s = c_s + k_s * x_log_grid
    #   4) Convert to L: logL_grid^s = -0.4*(M_grid^s - 90)
    #
    # Because k_s varies across samples, the spread grows as |x - x_cm| increases.
    ylog_grid_by_sample = []
    for s in flat_samples:
        sample_params = {k: s[param_indices[k]] for k in model_labels}
        s_arr = agn_model_pack_params(sample_params)

        # Full model (M) for each object at this draw
        M_i = M_model_agn(s_arr, agn_obs_arr, agn_pivot_arr)     # shape (N,)
        # OLS onto the *fixed* reference x
        xc = xcm
        Mc = np.mean(M_i)
        cov_Mx = np.mean((x_log_ref - xc) * (M_i - Mc))
        k_s = cov_Mx / var_x                     # slope in M-space against x_ref
        c_s = Mc - k_s * xc                      # intercept

        # Predict M on grid, then map to L
        M_grid_s = c_s + k_s * x_log_grid
        ylog_grid_s = -0.4 * (M_grid_s - 90.0)   # convert slope & intercept to L-space
        ylog_grid_by_sample.append(ylog_grid_s)

    ylog_grid_by_sample = np.asarray(ylog_grid_by_sample)  # (nsamp, ngrid)

    # Pointwise posterior summaries for the band
    ylog_med  = np.median(ylog_grid_by_sample, axis=0)
    ylog_low  = np.percentile(ylog_grid_by_sample, 16, axis=0)
    ylog_high = np.percentile(ylog_grid_by_sample, 84, axis=0)
    y_hi = np.max(ylog_high) + 0.05
    y_lo = np.min(ylog_low)  - 0.05

    # --- x for the plotted points (computed at med params for consistency with x_ref) ---
    x_data = x_ref
    # Propagate x errors from median params (same as before)
    pred_M_err_med = M_model_agn_err(med_arr, agn_obs_arr, agn_err_arr, agn_pivot_arr)
    x_lower = 10.0 ** (x_log_ref - pred_M_err_med)
    x_upper = 10.0 ** (x_log_ref + pred_M_err_med)
    xerr_asym = np.vstack((x_data - np.maximum(x_lower, 1e-300),
                           np.maximum(x_upper, x_data) - x_data))

    # --- Residuals vs median line (log space) for optional panel ---
    f_med  = interp1d(x_log_grid, ylog_med,  bounds_error=False, fill_value='extrapolate')
    f_low  = interp1d(x_log_grid, ylog_low,  bounds_error=False, fill_value='extrapolate')
    f_high = interp1d(x_log_grid, ylog_high, bounds_error=False, fill_value='extrapolate')

    model_logL_at_data = f_med(x_log_ref)
    residuals = actual_logL2500 - model_logL_at_data

    # Error budget in log space for residuals: model spread + x-prop + mag error
    obs_err = 0.5 * (f_high(x_log_ref) - f_low(x_log_ref))
    propagated_err = 0.4 * np.abs(pred_M_err_med)
    total_err = np.sqrt(obs_err**2 + propagated_err**2 + y_log_meas_err**2)

    # --- Figure scaffold ---
    color = 'm'
    if show_residuals:
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax = fig.add_subplot(gs[0])
        ax_res = fig.add_subplot(gs[1], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax_res = None

    # --- Data with errors (y errors converted to linear space) ---
    key = 'alpha_lambda'
    color_key = d[key] if key in d.columns else np.zeros(len(d))
    yerr_linear = 10**actual_logL2500 * np.log(10) * y_log_meas_err

    # Solid vs open AGN markers by z (both main and inset)
    mask_in  = d["z"].between(0.44, 3.16)
    mask_out = ~mask_in

    # # AGN (inside)
    # ax.errorbar(
    #     x_data[mask_in], 10**actual_logL2500[mask_in], xerr=xerr_asym[:, mask_in], yerr=yerr_linear[mask_in],
    #     fmt='o', linestyle='none', markersize=3,
    #     mfc="black", mec="none",
    #     ecolor="#666666", elinewidth=1.1,
    #     alpha=0.3, zorder=0, label="AGN"
    # )
    # # AGN (outside, open)
    # ax.errorbar(
    #     x_data[mask_out], 10**actual_logL2500[mask_out], xerr=xerr_asym[:, mask_out], yerr=yerr_linear[mask_out],
    #     fmt='o', linestyle='none', markersize=3, mfc='none', mec="k", alpha=0.3,
    #     ecolor="#666666", elinewidth=1.1, zorder=0
    # )

    ax.errorbar(
        x_data, 10**actual_logL2500, xerr=xerr_asym, yerr=yerr_linear,
        fmt="none", ecolor="#666666", elinewidth=0.7, alpha=0.4, zorder=2
    )

    # AGN (inside)
    ax.scatter(x_data[mask_in], 10**actual_logL2500[mask_in],
                c='k', s=10, alpha=0.2, edgecolors="k", zorder=3,
                label="AGN"
                #label="AGN (0.44 < z < 3.16)"
    )
    # AGN (outside)
    ax.scatter(x_data[mask_out], 10**actual_logL2500[mask_out],
                c='none', s=10, alpha=0.2, edgecolors="k", zorder=3,
                #label="AGN (z < 0.44 or z > 3.16)"
    )

    # --- Model: heteroscedastic ribbon (from varying slopes) + median line ---
    ax.fill_between(x_grid, 10**ylog_low, 10**ylog_high, color=color, alpha=0.5, label='best-fit model', zorder=9)
    #ax.plot(x_grid, 10**ylog_med, color=color, lw=2.0, zorder=10, label='best-fit model')

    # --- Comparison: Suberlak+2021 L2500 relation converted to our x ---
    A = 2.515 - 0.543
    B = 0.17 - 0.479
    C = 0.042 + 0.125
    D = 0.127 + 0.104
    log_MBH = 0
    log_lam = 0
    # For now, we only care about the slope in x, so we can ignore the intercept.
    L_scale = np.mean(10**actual_logL2500) # /(A + C*(23+90) + D*log_MBH + B*log_lam) * Mi -> M2500
    L_suberlak = L_scale * x_grid**(-2.5*C) * (2.5*C) 
    ax.plot(x_grid, L_suberlak, color='c', lw=2.0, zorder=10, label='Suberlak+2021 relation', linestyle='--')
 
 
 
    # --- Axes & labels ---
    ax.set_ylabel(r'$L_{2500}$ (erg s$^{-1})$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim((10**x_lo, 10**x_hi))
    ax.set_ylim((10**y_lo, 10**y_hi))
    # ax.set_xlim((1e-9, 1e7))
    # ax.set_ylim((1e41, 1e49))
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.legend(loc='upper right')

    if show_residuals and ax_res is not None:
        sc = ax_res.scatter(x_data, residuals, s=10, alpha=0.7, c=color_key, cmap='bwr',
                            edgecolor='k', lw=0.5, zorder=5)
        cbar = plt.colorbar(sc, ax=ax_res, orientation='vertical'); cbar.set_label(key)
        ax_res.errorbar(x_data, residuals, yerr=total_err,
                        fmt='none', alpha=0.25, lw=1.2, capsize=2.5, capthick=1, color='k', zorder=4)
        ax_res.axhline(0, color=color, linestyle='--', zorder=3)
        ax_res.set_ylabel('Residuals (log)')
        ax_res.set_xlabel(r'$(\sigma/\sigma_{\mathrm{p}})^{\alpha}(\tau/\tau_{\mathrm{p}})^{\beta}$')
        ax_res.set_xscale('log')
        ax_res.set_ylim(-2.2, 2.2)
        plt.setp(ax.get_xticklabels(), visible=False)
    else:
        ax.set_xlabel(r'$(\sigma/\sigma_{\mathrm{p}})^{\alpha}(\tau/\tau_{\mathrm{p}})^{\beta}$')
        plt.setp(ax.get_xticklabels(), visible=True)

    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(os.path.join(plot_path, "pdf"), exist_ok=True)
    if debias:
        plt.savefig(os.path.join(plot_path, "predicted_L2500_vs_fullcorr_band_debiased.png"), dpi=300)
        plt.savefig(os.path.join(plot_path, "pdf", "predicted_L2500_vs_fullcorr_band_debiased.pdf"), dpi=600)
    else:
        plt.savefig(os.path.join(plot_path, "predicted_L2500_vs_fullcorr_band.png"), dpi=300)
        plt.savefig(os.path.join(plot_path, "pdf", "predicted_L2500_vs_fullcorr_band.pdf"), dpi=600)

    if show:
        plt.show()
    plt.close()

def dmi_from_pdet_only(m_obs, m_obs_err, p_det, m_grid, sigma_completeness, z, tiny=1e-12):
    """
    m_obs: (N,)
    m_obs_err: (N,)
    p_det: (N, G) completeness vs magnitude for each object
    m_grid: (G,)
    """
    # variance term
    sigma2 = m_obs_err**2 + float(sigma_completeness)**2  # (N,)
    # safe log p_det and its slope w.r.t. magnitude
    logp = np.log(np.clip(p_det, tiny, 1.0))              # (N,G)
    dlogp_dm = np.gradient(logp, m_grid, axis=1)          # (N,G)
    # interpolate slope at m_obs
    idx = np.searchsorted(m_grid, m_obs) - 1
    idx = np.clip(idx, 0, len(m_grid) - 2)
    t = (m_obs - m_grid[idx]) / (m_grid[idx+1] - m_grid[idx])
    slope = (1 - t) * dlogp_dm[np.arange(len(m_obs)), idx] + t * dlogp_dm[np.arange(len(m_obs)), idx+1]
    # Δm ≈ σ² * d ln p_det / dm

    plt.figure(figsize=(7, 5))
    plt.scatter(m_obs, sigma2 * slope, c=m_obs_err, cmap='viridis', s=20, alpha=0.7, label='Objects')
    plt.xlabel('Observed Magnitude (m_obs)')
    plt.ylabel(r'$\Delta m = \sigma^2 \, \frac{d \ln p_{\rm det}}{dm}$')
    plt.title('Completeness Correction vs Observed Magnitude')
    plt.colorbar(label='Magnitude Error (m_obs_err)')
    plt.tight_layout()
    plt.ylim(-1, 0.5)
    os.makedirs("plots/completeness", exist_ok=True)
    plt.savefig("plots/completeness/dmi_vs_mag.png", dpi=300)
    plt.close()


    # Plot vs redshift (assuming you have z array)
    plt.figure(figsize=(7, 5))
    plt.scatter(z, sigma2 * slope, c=m_obs_err, cmap='viridis', s=20, alpha=0.7)
    plt.xlabel('Redshift (z)')
    plt.ylabel(r'$\Delta m = \sigma^2 \, \frac{d \ln p_{\rm det}}{dm}$')
    plt.title('Completeness Correction vs Redshift')
    plt.colorbar(label='Magnitude Error (m_obs_err)')
    plt.tight_layout()
    plt.ylim(-1, 0.5)
    plt.savefig("plots/completeness/dmi_vs_redshift.png", dpi=300)
    plt.close()
    return sigma2 * slope


def dmi_corr(m_obs, z_obs, m_obs_err,
                   H_obs_s, mag_centers, z_centers,
                   sigma_completeness, tiny=1e-12):
    """
    Δm ≈ σ^2 * ∂/∂m [ln n_obs(m|z)] evaluated at (m_obs, z_obs),
    where n_obs ∝ H_obs_s (smoothed counts per (mag,z) bin).

    Inputs
    ------
    m_obs, z_obs : (N,) arrays
    m_obs_err    : (N,) array (per-object photometric σ in mag)
    H_obs_s      : (Gm, Gz) smoothed 2D counts on (mag_centers, z_centers)
                   NOTE: H_obs_s axis 0 = mag, axis 1 = z
    mag_centers, z_centers : 1D grid centers used for H_obs_s
    sigma_completeness : extra magnitude scatter to include in σ (default 0)
    tiny : floor to avoid log(0)

    Returns
    -------
    dmi : (N,) array of magnitude shifts
    """
    # variance term
    sigma2 = m_obs_err**2 + float(sigma_completeness)**2

    # derivative of log counts along magnitude axis (units: 1/mag)
    dm = float(mag_centers[1] - mag_centers[0])
    logH = np.log(np.clip(H_obs_s, tiny, None))
    dlog_dm_grid = np.gradient(logH, dm, axis=0)  # axis 0 = mag

    # interpolate slope to object positions
    interp = RegularGridInterpolator(
        (mag_centers, z_centers), dlog_dm_grid,
        bounds_error=False, fill_value=0.0
    )
    slope = interp(np.column_stack([m_obs, z_obs]))

    # Teerikorpi-style first-order shift
    dmi = sigma2 * slope      # use "-sigma2 * slope" if following the minus-sign convention

    plt.figure(figsize=(7, 5))
    plt.scatter(m_obs, sigma2 * slope, c=m_obs_err, cmap='viridis', s=20, alpha=0.7, label='Objects')
    plt.xlabel('Observed Magnitude (m_obs)')
    plt.ylabel(r'$\Delta m = \sigma^2 \, \frac{d \ln p_{\rm det}}{dm}$')
    plt.title('Completeness Correction vs Observed Magnitude')
    plt.colorbar(label='Magnitude Error (m_obs_err)')
    plt.tight_layout()
    plt.ylim(-1, 0.5)
    os.makedirs("plots/completeness", exist_ok=True)
    plt.savefig("plots/completeness/dmi_vs_mag.png", dpi=300)
    plt.close()


    # Plot vs redshift (assuming you have z array)
    plt.figure(figsize=(7, 5))
    plt.scatter(z_obs, sigma2 * slope, c=m_obs_err, cmap='viridis', s=20, alpha=0.7)
    plt.xlabel('Redshift (z)')
    plt.ylabel(r'$\Delta m = \sigma^2 \, \frac{d \ln p_{\rm det}}{dm}$')
    plt.title('Completeness Correction vs Redshift')
    plt.colorbar(label='Magnitude Error (m_obs_err)')
    plt.tight_layout()
    plt.ylim(-1, 0.5)
    plt.savefig("plots/completeness/dmi_vs_redshift.png", dpi=300)
    plt.close()
    return dmi


from scipy.special import logsumexp
from hubble_likelihood import log_likelihood
def _highest_weight_theta(results):
    """
    Dynesty utils: pick the sample with the largest posterior weight.
    """
    w = np.exp(results.logwt - logsumexp(results.logwt))
    idx = int(np.argmax(w))
    return results.samples[idx]
def _blob_for_theta(theta, *, df_agn, df_pantheon, cosmo_model,
                    completeness_params, _sna_L, _sna_Lower, _sna_LogdetCov,
                    use_full_cov=True, use_mu_sh0es=False):
    """
    Re-evaluate the likelihood exactly once at 'theta' to get the selection blob.
    Returns: blob (2, N) and the AGN arrays z, m_obs needed for plotting.
    """
    ll, blob = log_likelihood(
        theta,
        agn_data=df_agn,
        pantheon_data=df_pantheon,
        _sna_L=_sna_L, _sna_Lower=_sna_Lower, _sna_LogdetCov=_sna_LogdetCov,
        cosmo_model=cosmo_model,
        completeness_params=completeness_params,
        only_sna=False, use_full_cov=use_full_cov,
        use_mu_sh0es=use_mu_sh0es,
    )
    z = df_agn['z'].values
    m_obs = df_agn['apparent_mag_2500'].values
    return blob, z, m_obs


def plot_Z_vs_z(z, Z, outdir, title_suffix=""):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8,5.2))
    plt.scatter(z, Z, s=12, alpha=0.55)
    plt.xlabel("Redshift (z)")
    plt.ylabel("integral (completeness)  Z = Φ(...) or ∫N×C")
    plt.title(f"Completeness integrals vs z {title_suffix}")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "completeness_integrals_vs_z.png"), dpi=200)
    plt.close()

def plot_dmi_vs_z(z, dmi, outdir, title_suffix=""):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8,5.2))
    # scatter shows multiplicity; line shows structure when sorted
    order = np.argsort(z)
    plt.plot(z[order], dmi[order], lw=1.4, alpha=0.9)
    plt.xlabel("Redshift (z)")
    plt.ylabel("dmi (mag)  = E[m|det] - m_obs")
    plt.title(f"Interpolated dmi vs z {title_suffix}")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dmi_vs_z.png"), dpi=200)
    plt.close()

def _hard_limit_m50_per_object(completeness2d, mag_centers, z):
    """
    Robust m50(z) (hard limit) for plotting:
    - clip z into the map's valid range,
    - find the first crossing of C=0.5 and linearly interpolate.
    """
    mgrid = np.asarray(mag_centers)
    z_in  = np.asarray(z, dtype=float)
    # Clip z to map bounds (avoids all-zero rows from the interpolator)
    zc = np.clip(z_in, getattr(completeness2d, "z_min", z_in.min()),
                        getattr(completeness2d, "z_max", z_in.max()))
    C = completeness2d(mgrid[None, :], zc[:, None])   # (N, G)

    m50 = np.empty(len(zc), dtype=float)
    for i, row in enumerate(C):
        target = 0.5
        if np.all(row <= target):
            m50[i] = mgrid[-1]
            continue
        if np.all(row >= target):
            m50[i] = mgrid[0]
            continue
        j = np.where((row[:-1] - target) * (row[1:] - target) <= 0)[0]
        j = j[0] if j.size else int(np.argmin(np.abs(row - target)))
        x0, x1 = mgrid[j], mgrid[j+1]
        y0, y1 = row[j], row[j+1]
        m50[i] = x0 + (target - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x0
    return m50

def plot_completeness_map_with_m50(completeness2d, mag_centers, z_centers,
                                   df_agn, outdir, title="Completeness map with hard m50(z)"):
    os.makedirs(outdir, exist_ok=True)
    # sample the map
    C = completeness2d(mag_centers[None, :], z_centers[:, None])  # (Z, M)
    # overlay m50(z) evaluated at the object's z, then rebin to z_centers for a smooth curve
    z_obj = df_agn['z'].values
    m50_obj = _hard_limit_m50_per_object(completeness2d, mag_centers, z_obj)
    # Bin m50(z) onto the z_centers grid for a single curve
    z_bins = np.r_[z_centers[0] - (z_centers[1]-z_centers[0])/2,
                   0.5*(z_centers[1:]+z_centers[:-1]),
                   z_centers[-1] + (z_centers[-1]-z_centers[-2])/2]
    inds = np.digitize(z_obj, z_bins) - 1
    m50_curve = np.array([np.median(m50_obj[inds==i]) if np.any(inds==i) else np.nan
                          for i in range(len(z_centers))])

    plt.figure(figsize=(7.6,5.6))
    im = plt.imshow(C.T, origin="lower", aspect="auto",
                    extent=[mag_centers[0], mag_centers[-1], z_centers[0], z_centers[-1]],
                    vmin=0.0, vmax=1.0)
    plt.xlabel("Apparent Magnitude")
    plt.ylabel("Redshift")
    plt.title(title)
    cbar = plt.colorbar(im); cbar.set_label("p(detect)")
    # overlay m50 curve
    ok = np.isfinite(m50_curve)
    if np.any(ok):
        plt.plot(m50_curve[ok], z_centers[ok], lw=2.2)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "completeness_map_with_m50.png"), dpi=200)
    plt.close()

def run_completeness_diagnostics(sampler_results, df_agn, df_pantheon,
                                 completeness_params, cosmo_model,
                                 _sna_L, _sna_Lower, _sna_LogdetCov,
                                 outdir="plots/completeness",
                                 use_full_cov=True, use_mu_sh0es=False,
                                 title_note="— highest posterior weight sample"):
    """
    One-call orchestration:
      - choose highest-posterior θ,
      - recompute selection blob via the SAME likelihood path (IMR or grid),
      - make Z(z), dmi(z), and map+m50 plots.
    """
    theta_star = _highest_weight_theta(sampler_results)
    blob, z, _ = _blob_for_theta(theta_star,
                                 df_agn=df_agn, df_pantheon=df_pantheon, cosmo_model=cosmo_model,
                                 completeness_params=completeness_params,
                                 _sna_L=_sna_L, _sna_Lower=_sna_Lower, _sna_LogdetCov=_sna_LogdetCov,
                                 use_full_cov=use_full_cov, use_mu_sh0es=use_mu_sh0es)
    Z   = np.asarray(blob[0], dtype=float)
    dmi = np.asarray(blob[1], dtype=float)
    plot_Z_vs_z(z, Z, outdir, title_suffix=title_note)
    plot_dmi_vs_z(z, dmi, outdir, title_suffix=title_note)

    completeness2d, mag_centers, z_centers, *_ = completeness_params
    plot_completeness_map_with_m50(completeness2d, mag_centers, z_centers, df_agn, outdir)
