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
import matplotlib.transforms as mtransforms

from hubble_model import M_model_agn, M_model_agn_err, M_model_SN, get_model_params, log_tau_UV_RF_pivot
from hubble_utils import calc_Mi_from_M2500
from hubble_completeness import make_dm_function
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import interp1d
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
        labels=model_labels_latex,
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

        if cosmo_model == "Flatw0waCDM":
            i_wp = _find(labels, "wp", "w_p")
            i_wa = _find(labels, "wa", "w_a")
            a_p  = 1.0 / (1.0 + float(z_pivot))
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
                ms, ps, ns = _fmt_err(m, lo, hi)
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
                    msb, psb, nsb = _fmt_err(mb, lob, hib)
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
    if show:
        plt.show()
    plt.close(fig)


def plot_hubble(flat_samples, df_agn, df_pantheon, cosmo_model, z_pivot_agn, plot_path="plots/hubble/",
                show_binned_agn=True,
                debias=False, dms=None, show=False, completeness=True, show_true=False, verbose=True):
    """
    Hubble diagram (Pantheon+-style):
      • Model line + 68% band in magenta
      • Concordance ΛCDM in black
      • SN Ia in blue
      • AGN points + error bars (solid if 0.44<=z<=3.16 else open)
      • Main: AGN binned in linear z
      • Inset: AGN binned in log z (matches inset x-scale)
    Returns: residuals, mu_pred_median, mu_pred_std
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from astropy.cosmology import FlatLambdaCDM, FlatwCDM
    # Ensure your project provides these:
    # from your_module import FlatwpwaCDM, M_model_agn, M_model_agn_err, get_model_params, make_dm_function

    # --- Labels ---
    if   cosmo_model == 'FlatwCDM':      label = r"flat $w$CDM model"
    elif cosmo_model == 'Flatw0waCDM':   label = r"flat $w_0w_a$CDM model"
    elif cosmo_model == 'FlatLambdaCDM': label = r"flat $\Lambda$CDM model"
    else:
        raise ValueError("Invalid cosmology model.")

    # --- Thinning for speed ---
    n_samples = int(flat_samples.shape[0])
    thin_factor = max(1, n_samples // 500)
    flat_samples = flat_samples[::thin_factor]

    z_grid = np.linspace(1e-4, 5.2, 250)

    # --- Parameter bookkeeping ---
    _, model_labels, _ = get_model_params(cosmo_model)
    param_indices = {name: model_labels.index(name) for name in model_labels}

    # --- Cosmology band on grid ---
    if cosmo_model == 'FlatwCDM':
        mu_models = np.array([
            FlatwCDM(H0=s[param_indices['H0']], Om0=s[param_indices['Om0']], w0=s[param_indices['w0']]).distmod(z_grid).value
            for s in flat_samples
        ])
    elif cosmo_model == 'Flatw0waCDM':
        mu_models = np.array([
            FlatwpwaCDM(H0=s[param_indices['H0']], Om0=s[param_indices['Om0']],
                        wp=s[param_indices['wp']], wa=s[param_indices['wa']], zp=z_pivot_agn
                        ).distmod(z_grid).value
            for s in flat_samples
        ])
    else:  # FlatLambdaCDM
        mu_models = np.array([
            FlatLambdaCDM(H0=s[param_indices['H0']], Om0=s[param_indices['Om0']]).distmod(z_grid).value
            for s in flat_samples
        ])

    mu_model_16th   = np.percentile(mu_models, 16, axis=0)
    mu_model_median = np.percentile(mu_models, 50, axis=0)
    mu_model_84th   = np.percentile(mu_models, 84, axis=0)

    # Median params for uncertainties
    results = {key: np.percentile(flat_samples[:, i], [16, 50, 84])
               for i, key in enumerate(model_labels)}

    # --- Predicted AGN μ per object ---
    m_obs = df_agn['apparent_mag_2500'].values
    mu_pred_samples = np.array([
        m_obs - M_model_agn(
            s[param_indices['M0_agn']],
            s[param_indices['alpha_agn']],
            s[param_indices['beta_agn']],
            s[param_indices['gamma_agn']],
            df_agn['log_sigma_UV'].values,
            df_agn['log_tau_UV_RF'].values,
            df_agn['bwb_beta'].values
        ) for s in flat_samples
    ])

    # De-bias (assumes your make_dm_function clips to grid, no extrapolation)
    if debias:
        dm_interp = make_dm_function(m_obs, df_agn['z'], dms, method='linear')
        pts = np.column_stack([df_agn['z'].values, m_obs])
        mu_pred_samples -= dm_interp(pts)

    mu_pred_median = np.percentile(mu_pred_samples, 50, axis=0)
    mu_pred_16th   = np.percentile(mu_pred_samples, 16, axis=0)
    mu_pred_84th   = np.percentile(mu_pred_samples, 84, axis=0)

    # Per-object uncertainty (for yerr)
    mu_pred_std = np.sqrt(
        df_agn['apparent_mag_2500_err'].values**2 +
        (0.055 * df_agn["z"].values)**2 +
        M_model_agn_err(
            results['M0_agn'][1],
            results['alpha_agn'][1],
            results['beta_agn'][1],
            results['gamma_agn'][1],
            df_agn['log_sigma_UV'].values,
            df_agn['log_sigma_UV_err'].values,
            df_agn['log_tau_UV_RF_err'].values,
            df_agn['bwb_beta_err'].values
        )**2
    )

    # Residuals (for outlier print only)
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

    # Linear-z bins for MAIN panel
    dz   = 0.2
    zmax_main = 3.8
    bins_linear = np.arange(0.05, zmax_main + dz + 1e-9, dz)
    z_lin, mu_lin_mean, mu_lin_sem = _weighted_bin_stats(
        df_agn["z"].values, mu_pred_median, w, bins_linear, min_count=3, center='weighted'
    )

    # Log-z bins for INSET (match inset xscale='log')
    zmin_pos = df_agn["z"].values[df_agn["z"].values > 0]
    zmin_inset = max(0.02, float(np.min(zmin_pos))) if zmin_pos.size else 0.02
    zmax_inset = 3.8
    bins_per_decade = 6  # tune resolution here
    decades = np.log10(zmax_inset) - np.log10(zmin_inset)
    
    n_bins_log = max(1, int(np.ceil(decades * bins_per_decade)))
    bins_log = np.logspace(np.log10(zmin_inset), np.log10(zmax_inset), n_bins_log + 1)
    z_log, mu_log_mean, mu_log_sem = _weighted_bin_stats(
        df_agn["z"].values, mu_pred_median, w, bins_log, min_count=3, center='weighted'
    )

    # ======== Plot ========
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.set_ylim(26, 51)
    ax.set_xlim(-0.2, 4.2)
    inset_ax = inset_axes(ax, width="40%", height="40%", loc="lower right", borderpad=1.5)

    # Solid vs open AGN markers by z (both main and inset)
    mask_in  = df_agn["z"].between(0.44, 3.16)
    mask_out = ~mask_in

    # ---------- Inset (log-z) with ALL elements ----------
    inset_ax.set_xscale('log')

    # AGN (inside)
    inset_ax.errorbar(
        df_agn["z"][mask_in], mu_pred_median[mask_in], yerr=mu_pred_std[mask_in],
        fmt='o', linestyle='none', markersize=2,
        mfc="black", mec="black",
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
            markersize=3, mfc='red', mec='red',
            ecolor='red', elinewidth=2.2, capsize=3.5,
            alpha=0.98, zorder=14, label="AGN (z-binned, log)"
        )

    # SN Ia
    inset_ax.errorbar(
        df_pantheon["zHD"], df_pantheon["MU_SH0ES"], yerr=df_pantheon["MU_SH0ES_ERR_DIAG"],
        fmt='s', markersize=2, color="#0A84FF", linestyle='none', lw=0.8, alpha=0.95, zorder=0, label="SN Ia"
    )

    # Model + band
    inset_ax.plot(z_grid, mu_model_median, color="m", lw=1.4, alpha=1.0, zorder=5, label=label)
    inset_ax.fill_between(z_grid, mu_model_16th, mu_model_84th, color="m", alpha=0.22, zorder=4)

    # Survey magnitude limit (shade above) — use positive z_grid on log axis
    if completeness and not debias:
        log_sigma0_med    = float(np.median(df_agn['log_sigma0'].values))
        log_tau_UV_RF_med = float(np.median(df_agn['log_tau_UV_RF'].values))
        bwb_med           = float(np.median(df_agn['bwb_beta'].values))
        M_med_grid = np.median([
            M_model_agn(
                s[param_indices['M0_agn']],
                s[param_indices['alpha_agn']],
                s[param_indices['beta_agn']],
                s[param_indices['gamma_agn']],
                log_sigma0_med * np.ones_like(z_grid),
                log_tau_UV_RF_med * np.ones_like(z_grid),
                bwb_med * np.ones_like(z_grid)
            ) for s in flat_samples
        ], axis=0)
        m_lim = 24.0
        mu_lim = m_lim - M_med_grid
        inset_ax.fill_between(z_grid, mu_lim, 60, color="red", alpha=0.12, zorder=2, label="< 50% complete")

    # Concordance
    mu_conc = FlatLambdaCDM(H0=70, Om0=0.3).distmod(z_grid).value
    inset_ax.plot(z_grid, mu_conc, color="#F0B000", lw=1.2, ls='--', alpha=0.9, label=r"Concordance $\Lambda$CDM")

    inset_ax.set_xlim(0.02, 5.2)
    inset_ax.set_ylim(32, 51)
    inset_ax.set_xlabel(r"$z$", fontsize=12, labelpad=-10)
    inset_ax.set_ylabel(r"$\mu$ (mag)", fontsize=12)
    inset_ax.tick_params(axis='both', which='major', labelsize=10)

    # ---------- Main plot with ALL elements ----------
    # AGN (inside)
    ax.errorbar(
        df_agn["z"][mask_in], mu_pred_median[mask_in], yerr=mu_pred_std[mask_in],
        fmt='o', linestyle='none', markersize=3,
        mfc="black", mec="black",
        ecolor="#666666", elinewidth=1.1,
        alpha=0.3, zorder=1, label="AGN"
    )
    # AGN (outside, open)
    ax.errorbar(
        df_agn["z"][mask_out], mu_pred_median[mask_out], yerr=mu_pred_std[mask_out],
        fmt='o', linestyle='none', markersize=3, mfc='none', mec="k", alpha=0.3,
        ecolor="#666666", elinewidth=1.1, zorder=1
    )

    # MAIN: linear-binned AGN
    if show_binned_agn:
        ax.errorbar(
            z_lin, mu_lin_mean, yerr=mu_lin_sem,
            fmt='o', linestyle='none',
            markersize=5, mfc='red', mec='red',
            ecolor='red', elinewidth=2.2, capsize=3.5,
            alpha=0.98, zorder=14, label="AGN (z-binned)"
        )

    # SN Ia
    ax.errorbar(
        df_pantheon["zHD"], df_pantheon["MU_SH0ES"], yerr=df_pantheon["MU_SH0ES_ERR_DIAG"],
        fmt='s', markersize=3, color="#0A84FF", linestyle='none', lw=1, alpha=0.95, zorder=0, label="SN Ia"
    )

    # Model + 68% band
    ax.plot(z_grid, mu_model_median, color="m", lw=2.4, alpha=1.0, zorder=5, label=label)
    ax.fill_between(z_grid, mu_model_16th, mu_model_84th, color="m", alpha=0.25, zorder=4)

    # Survey magnitude limit (shade above)
    if completeness and not debias:
        log_sigma0_med    = float(np.median(df_agn['log_sigma0'].values))
        log_tau_UV_RF_med = float(np.median(df_agn['log_tau_UV_RF'].values))
        bwb_med           = float(np.median(df_agn['bwb_beta'].values))
        M_med_grid = np.median([
            M_model_agn(
                s[param_indices['M0_agn']],
                s[param_indices['alpha_agn']],
                s[param_indices['beta_agn']],
                s[param_indices['gamma_agn']],
                log_sigma0_med * np.ones_like(z_grid),
                log_tau_UV_RF_med * np.ones_like(z_grid),
                bwb_med * np.ones_like(z_grid)
            ) for s in flat_samples
        ], axis=0)
        m_lim = 24.0
        mu_lim = m_lim - M_med_grid
        ax.fill_between(z_grid, mu_lim, 60, color="red", alpha=0.15, zorder=2, label="< 50% complete")

    # Concordance ΛCDM
    mu_conc = FlatLambdaCDM(H0=70, Om0=0.3).distmod(z_grid).value
    ax.plot(z_grid, mu_conc, color="#F0B000", lw=1.6, ls='--', alpha=0.9, label=r"Concordance $\Lambda$CDM")

    # Labels, ticks, legend
    ax.set_ylabel(r"$\mu$ (mag)")
    ax.set_xlabel(r"$z$")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.22, 0.06), fontsize=12)

    for axi in (ax, inset_ax):
        axi.minorticks_on()
        axi.tick_params(axis='both', which='minor', direction='in', length=4, top=True, right=True, width=2)
        axi.tick_params(axis='both', which='major', direction='in', length=8, top=True, right=True)

    # Save/show
    fig.tight_layout()
    os.makedirs(plot_path, exist_ok=True)
    filename = f"hubble_diagram.png"
    if debias:
        filename = "hubble_diagram_debiased.png"
    plt.savefig(os.path.join(plot_path, filename))
    if show:
        plt.show()
    plt.close(fig)

    # Outlier report
    outlier_mask = np.abs(residuals) > 4
    if np.any(outlier_mask) and verbose:
        print("Outliers with residuals > 4:")
        for idx in np.where(outlier_mask)[0]:
            sdss_name = df_agn.iloc[idx].get('sdss_name', 'Unknown')
            object_id = df_agn.iloc[idx].get('object_id', 'Unknown')
            ra  = df_agn.iloc[idx].get('ra',  np.nan)
            dec = df_agn.iloc[idx].get('dec', np.nan)
            z   = df_agn.iloc[idx]['z']
            print(f"z: {z:.2f} | object_id: {object_id} | SDSS: {sdss_name} | RA: {ra:.5f} | DEC: {dec:.5f} | Residual: {residuals[idx]:.1f}")

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
    cmap="bone_r",       # colormap for points
    box_alpha=0.7,        # transparency of white annotation boxes
    show_sigma_band=True,
    completeness=True,    # add "<50% complete" red region
    m_lim=24.0,           # survey apparent-magnitude limit for completeness shading
    n_cosmo_draws=50,    # posterior draws to propagate cosmology errors (for xerr)
    random_state=42,      # RNG seed for reproducibility of draws
):
    """
    Predicted vs Actual M_2500 with:
      • y-error bars from M_model_agn_err(...)
      • x-error bars = sqrt(apparent_mag_2500_err^2 + sigma_mu_cosmo(z)^2)
      • ±1σ band from intrinsic scatter sigma_int = exp(log_f) (magenta)
      • Per-bin red shaded region "< 50% complete" using bin-center z; shaded area fills
        the entire y-range to the LEFT of the vertical limit (with inverted x-axis).
      • Points colored by m_2500 (no marker outlines), full-height colorbar,
        white translucent annotation boxes.

    Layout: 12 redshift bins arranged as 4 columns × 3 rows.
    """
    import os, math
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.cosmology import FlatLambdaCDM, FlatwCDM, FlatwpwaCDM

    # --- model parameters from samples ---
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    results = {key: np.percentile(flat_samples[:, i], [16, 50, 84])
               for i, key in enumerate(model_labels)}
    label_to_idx = {k: i for i, k in enumerate(model_labels)}

    # --- intrinsic scatter: sigma_int = exp(log_f) from posterior median ---
    def _find_logf_key(keys):
        lowers = {k.lower(): k for k in keys}
        for cand in ("log_f", "logf", "lnf", "log_sigma_int", "ln_sigma_int"):
            if cand in lowers:
                return lowers[cand]
        for k in keys:
            if "log" in k.lower() and "f" in k.lower():
                return k
        return None
    logf_key = _find_logf_key(results.keys())
    if logf_key is None:
        raise KeyError("Could not find 'log_f' (or variant) among model parameters.")
    sigma_intrinsic = float(np.exp(results[logf_key][1]))

    # --- helpers to build cosmology objects ---
    def _cosmo_from_params(H0, Om0, **kw):
        if cosmo_model == "FlatwCDM":
            return FlatwCDM(H0=H0, Om0=Om0, w0=kw["w0"])
        elif cosmo_model == "Flatw0waCDM":
            return FlatwpwaCDM(H0=H0, Om0=Om0, wp=kw["wp"], wa=kw["wa"], zp=z_pivot_agn)
        elif cosmo_model == "FlatLambdaCDM":
            return FlatLambdaCDM(H0=H0, Om0=Om0)
        else:
            raise ValueError("Invalid cosmology model.")

    # Median cosmology for best-estimate distances
    if cosmo_model == "FlatwCDM":
        cosmo_med = _cosmo_from_params(results["H0"][1], results["Om0"][1], w0=results["w0"][1])
    elif cosmo_model == "Flatw0waCDM":
        cosmo_med = _cosmo_from_params(results["H0"][1], results["Om0"][1],
                                       wp=results["wp"][1], wa=results["wa"][1])
    else:
        cosmo_med = _cosmo_from_params(results["H0"][1], results["Om0"][1])

    # --- data & predictions ---
    z = df_agn["z"].values
    m_app = df_agn["apparent_mag_2500"].values
    if "apparent_mag_2500_err" not in df_agn.columns:
        raise KeyError("df_agn must contain 'apparent_mag_2500_err' for x-error bars.")
    m_app_err = df_agn["apparent_mag_2500_err"].values

    distmod_med = np.array([cosmo_med.distmod(zi).value for zi in z])
    actual_M_2500 = m_app - distmod_med

    M_2500_pred = M_model_agn(
        results["M0_agn"][1],
        results["alpha_agn"][1],
        results["beta_agn"][1],
        results["gamma_agn"][1],
        df_agn["log_sigma_UV"].values,
        df_agn["log_tau_UV_RF"].values,
        df_agn["bwb_beta"].values,
    )

    # --- y-errors from model error propagation ---
    M_2500_pred_err = M_model_agn_err(
        results["M0_agn"][1],
        results["alpha_agn"][1],
        results["beta_agn"][1],
        results["gamma_agn"][1],
        df_agn["log_sigma_UV"].values,
        df_agn["log_sigma_UV_err"].values,
        df_agn["log_tau_UV_RF_err"].values,
        df_agn["bwb_beta_err"].values,
    )
    M_2500_pred_err = np.asarray(M_2500_pred_err, dtype=float)
    M_2500_pred_err[~np.isfinite(M_2500_pred_err) | (M_2500_pred_err < 0)] = np.nan

    # --- x-errors: propagate cosmology posterior into distance modulus ---
    rng = np.random.default_rng(random_state)
    n_samp = flat_samples.shape[0]
    n_draws = min(n_cosmo_draws, n_samp)
    draw_idxs = rng.choice(n_samp, size=n_draws, replace=False) if n_draws < n_samp else np.arange(n_samp)

    def _cosmo_from_draw(row):
        if cosmo_model == "FlatwCDM":
            return FlatwCDM(H0=row[label_to_idx["H0"]], Om0=row[label_to_idx["Om0"]], w0=row[label_to_idx["w0"]])
        elif cosmo_model == "Flatw0waCDM":
            return FlatwpwaCDM(H0=row[label_to_idx["H0"]],
                               Om0=row[label_to_idx["Om0"]],
                               wp=row[label_to_idx["wp"]],
                               wa=row[label_to_idx["wa"]],
                               zp=z_pivot_agn)
        else:
            return FlatLambdaCDM(H0=row[label_to_idx["H0"]], Om0=row[label_to_idx["Om0"]])

    mu_draws = np.empty((n_draws, z.size), dtype=float)
    for j, idx in enumerate(draw_idxs):
        row = flat_samples[idx, :]
        cosmo_j = _cosmo_from_draw(row)
        mu_draws[j, :] = np.array([cosmo_j.distmod(zi).value for zi in z])
    sigma_mu_cosmo = np.nanstd(mu_draws, axis=0, ddof=1)  # per-object DM uncertainty
    xerr = np.sqrt(m_app_err**2 + sigma_mu_cosmo**2)

    # --- binning in redshift ---
    num_cols = 4
    num_rows = 4
    z_bins = np.linspace(0, 3.5, num_cols*num_rows+1)
    z_bin_indices = np.digitize(z, bins=z_bins)
    num_bins = len(z_bins) - 1  # = 12
    bin_labels = [f"{z_bins[i]:.1f} < z < {z_bins[i+1]:.1f}" for i in range(num_bins)]

    # --- figure with full-height colorbar (dedicated column) ---
    fig = plt.figure(figsize=(5 * num_cols + 1.4, 4 * num_rows))
    gs = fig.add_gridspec(num_rows, num_cols + 1,
                          width_ratios=[1]*num_cols + [0.06],
                          wspace=0.0, hspace=0.0)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(num_cols)] for r in range(num_rows)]).flatten()
    cax = fig.add_subplot(gs[:, -1])

    # --- color scaling by m_2500 ---
    vmin, vmax = np.nanmin(m_app), np.nanmax(m_app)

    # --- axis helpers ---
    xlo, xhi = -25.8, -18.2
    ylo, yhi = -25.8, -18.2
    xx = np.linspace(min(xlo, ylo), max(xhi, yhi), 400)

    sc_for_cbar = None

    if debias:
        apparent_mag_2500 = df_agn['apparent_mag_2500'].values
        dm_interp = make_dm_function(apparent_mag_2500, df_agn['z'].values, dms)
        dm_interp = make_dm_function(apparent_mag_2500, df_agn['z'], dms, method='linear')
        pts = np.column_stack([df_agn['z'].values, apparent_mag_2500])
        M_2500_pred += dm_interp(pts)

    for i, ax in enumerate(axes):
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)

        if i >= num_bins:
            ax.axis("off"); continue

        bin_mask = z_bin_indices == (i + 1)
        if not np.any(bin_mask):
            ax.axis("off"); continue

        x = actual_M_2500[bin_mask]
        y = M_2500_pred[bin_mask]
        xerr_bin = xerr[bin_mask]
        yerr_bin = M_2500_pred_err[bin_mask]
        cvals = m_app[bin_mask]

        # residuals & coverage vs intrinsic sigma
        resid = y - x
        frac1 = 100.0 * np.mean(np.abs(resid) <= 1.0 * sigma_intrinsic)
        frac2 = 100.0 * np.mean(np.abs(resid) <= 2.0 * sigma_intrinsic)
        frac3 = 100.0 * np.mean(np.abs(resid) <= 3.0 * sigma_intrinsic)
        rms_resid = float(np.sqrt(np.nanmean(resid**2))) if resid.size else np.nan

        # error bars (lighter & more transparent for visibility)
        ax.errorbar(
            x, y, xerr=xerr_bin, yerr=yerr_bin,
            fmt="none",
            ecolor="#777777",          # lighter gray
            elinewidth=0.7,            # thinner lines
            alpha=0.4,                 # more transparent
            zorder=2,
        )

        # colored points, no edge (label 'AGN' only once)
        scatter_kwargs = dict(
            c=cvals, cmap=cmap, vmin=vmin, vmax=vmax,
            s=20, alpha=0.9, edgecolors="none", zorder=3,
        )
        if i == 0:
            sc = ax.scatter(x, y, label="AGN", **scatter_kwargs)
        else:
            sc = ax.scatter(x, y, **scatter_kwargs)
        sc_for_cbar = sc

        # y = x reference and ±1σ intrinsic band (MAGENTA)
        ax.plot(xx, xx, color="m", alpha=0.9, lw=2.2, linestyle="--", zorder=1)
        if show_sigma_band:
            ax.fill_between(xx, xx - sigma_intrinsic, xx + sigma_intrinsic,
                            facecolor="m", alpha=0.12, edgecolor="m",
                            lw=0.8, zorder=1, label=r"$\pm 1\sigma_{\rm int}$")

        # "< 50% complete" region in RED — bin-center z; fill entire y-span to the LEFT
        if completeness:
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
        stats_text = (
            f"σ_int = {sigma_intrinsic:.2f} mag   |   RMS(resid) = {rms_resid:.2f}\n"
            f"within ±σ_int: {frac1:4.1f}%  •  ±2σ_int: {frac2:4.1f}%  •  ±3σ_int: {frac3:4.1f}%"
        )
        ax.annotate(
            stats_text, xy=(0.03, 0.03), xycoords="axes fraction",
            fontsize=10.5, color="dimgray", ha="left", va="bottom", bbox=boxprops,
        )

        n_in_bin = int(np.sum(bin_mask))
        ax.annotate(
            f"N = {n_in_bin}", xy=(0.97, 0.03), xycoords="axes fraction",
            fontsize=11, color="gray", ha="right", va="bottom", bbox=boxprops,
        )

        # labels only on bottom row / left col
        if i >= (num_rows - 1) * num_cols:
            ax.set_xlabel("Actual $M_{2500}$", fontsize=12)
        if i % num_cols == 0:
            ax.set_ylabel("Predicted $M_{2500}$", fontsize=12)
        ax.tick_params(axis="both", labelsize=10, length=3)

        # show legend once (will include AGN + <50% complete + band label)
        if (show_sigma_band or completeness) and i == num_cols-1:
            leg = ax.legend(loc="upper right", fontsize=14, frameon=True)
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_alpha(box_alpha)
            leg.get_frame().set_edgecolor("none")

    # full-height colorbar
    if sc_for_cbar is not None:
        cbar = fig.colorbar(sc_for_cbar, cax=cax, orientation="vertical")
        cbar.set_label(r"m$_{2500}$", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    for ax in axes:
        if ax.has_data():
            ax.label_outer()

    os.makedirs(plot_path, exist_ok=True)
    out_path = os.path.join(plot_path, f"predicted_vs_actual_M2500{'_debias' if debias else ''}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_predicted_vs_actual_Mi(flat_samples, df_agn, cosmo_model, z_pivot_agn, show=False):
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
            zp=z_pivot_agn
        )
    elif cosmo_model == 'FlatLambdaCDM':
        cosmo = FlatLambdaCDM(
            H0=results['H0'][1],
            Om0=results['Om0'][1]
        )
    else:
        raise ValueError("Invalid cosmology model.")
        
    actual_M_i = df_agn['M_i'].values

    M_2500_pred = M_model_agn(
        results['M0_agn'][1], 
        results['alpha_agn'][1],
        results['beta_agn'][1], 
        results['gamma_agn'][1], 
        df_agn['log_sigma_UV'].values,
        df_agn['log_tau_UV_RF'].values,
        df_agn['bwb_beta'].values,
    )

    # Calculate prediction errors
    M_2500_pred_err = np.sqrt(
        M_model_agn_err(
            results['M0_agn'][1],
            results['alpha_agn'][1],
            results['beta_agn'][1], 
            results['gamma_agn'][1], 
            df_agn['log_sigma_UV'].values, 
            df_agn['log_sigma_UV_err'].values, 
            df_agn['log_tau_UV_RF_err'].values,
            df_agn['bwb_beta_err'].values,
        )**2
    )

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

    os.makedirs(plot_path, exist_ok=True)
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

    log_sigma_UV = df_agn['log_sigma_UV']
    M_i = df_agn['M_i']

    plt.figure(figsize=(8, 6))
    plt.scatter(log_sigma_UV, M_i, label='Data', color='k', alpha=0.5)
    plt.xlabel(r'$\log \hat{\sigma}^2$')
    plt.ylabel(r'$M_{i}$')
    plt.legend()
    if show:
        plt.show()
    plt.savefig(f"plots/hubble/Mi_vs_sigmahat_{cosmo_model}.png", dpi=300)
    plt.close()


def plot_full_residuals(df_agn, residuals, flat_samples, cosmo_model, z_pivot_agn, plot_path='plots/hubble', show=False):
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
            zp=z_pivot_agn,
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
        'log_sigma_UV', 'log_sigma_hat_UV', 'log_tau_UV_RF', 'chi_sq_g',
        'bwb_beta', 'sn_median_all', 'bwb_alpha', 'bwb_beta',
        'redchi', 'bwb_beta_4200', 'alpha_lambda', 'f_host_4200',
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
    plt.savefig(os.path.join(plot_path, "full_residuals.png"), dpi=300)
    plt.close()



def plot_predicted_L2500_vs_sigmahat(flat_samples, df_agn, cosmo_model, z_pivot_agn, plot_path='plots/hubble', show=False):
    d = df_agn.copy()
    #d = d[d['LOGL2500_ERR'] > 0]

    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    param_indices = {name: model_labels.index(name) for name in model_labels}

    predicted_logL2500_samples = []
    predicted_logL2500_err_samples = []

    log_sigma_UV_grid = np.linspace(d['log_sigma_UV'].min()-0.5, d['log_sigma_UV'].max()+0.5, 100)
    for log_sigma_UV in log_sigma_UV_grid:
        for s in flat_samples:
            sample_params = {
                'M0_agn': s[param_indices['M0_agn']],
                'alpha_agn': s[param_indices['alpha_agn']],
                'beta_agn': s[param_indices['beta_agn']],
                'gamma_agn': s[param_indices['gamma_agn']],
            }

            predicted_M2500 = M_model_agn(
                sample_params['M0_agn'],
                sample_params['alpha_agn'], 
                sample_params['beta_agn'],
                sample_params['gamma_agn'],
                log_sigma_UV, d['log_tau_UV_RF'].mean(),
                d['bwb_beta'].mean()
            )
            predicted_logL2500 = -0.4 * (predicted_M2500 - 90) #* np.log10(np.e)  # log10(L)
            predicted_logL2500_samples.append(predicted_logL2500)

            predicted_M2500_err = M_model_agn_err(
                sample_params['M0_agn'],
                sample_params['alpha_agn'], 
                sample_params['beta_agn'],
                sample_params['gamma_agn'],
                log_sigma_UV, d['log_tau_UV_RF_err'].mean(), d['log_sigma_UV_err'].mean(),
                d['bwb_beta_err'].mean()
            )
            predicted_logL2500_err = -0.4 * predicted_M2500_err #* np.log10(np.e)  # log10(L)
            predicted_logL2500_err_samples.append(predicted_logL2500_err)

    predicted_logL2500_samples = np.array(predicted_logL2500_samples)
    predicted_logL2500_samples = predicted_logL2500_samples.reshape(len(log_sigma_UV_grid), -1)

    predicted_logL2500_median = np.median(predicted_logL2500_samples, axis=1)
    predicted_logL2500_low = np.percentile(predicted_logL2500_samples, 16, axis=1)
    predicted_logL2500_high = np.percentile(predicted_logL2500_samples, 84, axis=1)

    results = {key: np.percentile(flat_samples[:, i], [16, 50, 84]) for i, key in enumerate(model_labels)}
    if cosmo_model == 'FlatwCDM':
        cosmo = FlatwCDM(H0=results['H0'][1], Om0=results['Om0'][1], w0=results['w0'][1])
    elif cosmo_model == 'Flatw0waCDM':
        cosmo = FlatwpwaCDM(H0=results['H0'][1], Om0=results['Om0'][1], wp=results['wp'][1], wa=results['wa'][1], zp=z_pivot_agn)
    else:
        raise ValueError(f"Unknown cosmological model: {cosmo_model}")

    actual_M2500 = d['apparent_mag_2500'] - cosmo.distmod(d['z']).value
    actual_logL2500 = -0.4 * (actual_M2500 - 90) #* np.log10(np.e)  # log10(L)

    log_sigma_UV = d['log_sigma_UV']

    # Interpolate model at data points for residuals (in log space)
    interp_model = interp1d(log_sigma_UV_grid, predicted_logL2500_median, bounds_error=False, fill_value='extrapolate')
    model_logL2500_at_data = interp_model(d['log_sigma_UV'])
    residuals = actual_logL2500 - model_logL2500_at_data

    if False and 'LOGL2500_ERR' in d.columns:
        obs_err = d['LOGL2500_ERR']
    else:
        interp_low = interp1d(log_sigma_UV_grid, predicted_logL2500_low, bounds_error=False, fill_value='extrapolate')
        interp_high = interp1d(log_sigma_UV_grid, predicted_logL2500_high, bounds_error=False, fill_value='extrapolate')
        obs_err = 0.5 * (interp_high(d['log_sigma_UV']) - interp_low(d['log_sigma_UV']))
    
    sigma0 = 10**(d['log_sigma_UV'])
    sigma0_err = np.log(10) * sigma0 * d['log_sigma_UV_err']
    dlogL_dlog_sigma_UV = np.gradient(model_logL2500_at_data, d['log_sigma_UV'])
    propagated_err = np.abs(dlogL_dlog_sigma_UV) * d['log_sigma_UV_err']
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
    ax.plot(10**(log_sigma_UV_grid), 10**predicted_logL2500_median, color=color, zorder=-4)
    ax.fill_between(10**(log_sigma_UV_grid), 10**predicted_logL2500_low, 10**predicted_logL2500_high, color=color, alpha=0.3, label='Model', zorder=-5)
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
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path, "predicted_L2500_vs_sigmahat.png"), dpi=300)
    if show:
        plt.show()
    plt.close()