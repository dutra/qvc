import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from tqdm import tqdm
import math
import corner
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.cosmology import FlatwCDM, FlatwpwaCDM, FlatLambdaCDM, Flatw0waCDM
from astropy.cosmology.realizations import Planck18
import matplotlib.pyplot as plt
import os

prefix = os.environ.get("PREFIX", "")

import copy
import matplotlib.transforms as mtransforms

from hubble_model import (M_model_agn, M_model_agn_err, get_model_params, agn_model_pack_params,
    agn_model_pack_obs, agn_model_eidx, agn_model_oidx, agn_model_pidx)
from hubble_likelihood import sigma_lens_from_dc
from hubble_utils import *
from hubble_model import agn_model_req_obs, agn_model_req_errs
#from hubble_completeness import make_dm_function
from dynesty.utils import resample_equal
from tqdm import tqdm
from dynesty import plotting as dyplot

def plot_dynesty(results, cosmo_model, plot_path="plots/hubble", only_sna="", speed="", show=False):
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
    fig_corner.savefig(f"{plot_path}/cornerplot_{cosmo_model}_{'sna' if only_sna else 'joint'}_{speed}.png", dpi=100)
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

    fig_trace.savefig(f"{plot_path}/traceplot_{cosmo_model}_{'sna' if only_sna else 'joint'}_{speed}.png", dpi=100)
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
    speed='',
    smooth=160,
    gauss_sigma=1.2,
    kde_bw_scale=1.0,
    grid_q=(0.0005, 0.9995),
    pad_frac=0.25
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
    from matplotlib.lines import Line2D
    from scipy.stats import gaussian_kde
    from scipy.ndimage import gaussian_filter

    # --- pull model labels from your config ---
    _, model_labels, _ = get_model_params(cosmo_model)

    # ---------- helpers ----------
    def _find(labels, *cands):
        for c in cands:
            if c in labels:
                return labels.index(c)
        raise KeyError(f"Could not find any of {cands} in labels {labels}")

    def _subset(samples, labels, z_pivot):
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
            units_latex = ["(km s$^{-1}$ Mpc$^{-1}$)", "", "", ""]
        elif cosmo_model == "Flatw0waCDM":
            i_w0 = _find(labels, "w0", "w_0")
            i_wa = _find(labels, "wa", "w_a")
            w0, wa = X[:, i_w0], X[:, i_wa]
            Y = np.column_stack([X[:, i_H0], X[:, i_Om0], w0, wa])
            lab_latex = [r"$H_0$", r"$\Omega_m$", r"$w_0$", r"$w_a$"]
            units_latex = ["(km s$^{-1}$ Mpc$^{-1}$)", "", "", ""]
        elif cosmo_model == "FlatwCDM":
            i_w0 = _find(labels, "w0", "w_0", "w")
            Y = np.column_stack([X[:, i_H0], X[:, i_Om0], X[:, i_w0]])
            lab_latex = [r"$H_0$", r"$\Omega_m$", r"$w_0$"]
            units_latex = ["(km s$^{-1}$ Mpc$^{-1}$)", "", ""]
        elif cosmo_model == 'FlatLambdaCDM':
            Y = np.column_stack([X[:, i_H0], X[:, i_Om0]])
            lab_latex = [r"$H_0$", r"$\Omega_m$"]
            units_latex = ["(km s$^{-1}$ Mpc$^{-1}$)", ""]
        else:
            raise ValueError(f"Unsupported cosmo_model '{cosmo_model}' for this plot.")
        return Y, lab_latex, units_latex

    def _fmt_err(m, lo, hi, latex_label=""):
        nd = 2
        if latex_label == r"$H_0$":
            nd = 1
        return f"{m:.{nd}f}", f"{hi - m:.{nd}f}", f"{m - lo:.{nd}f}"

    def _get_density_levels(values, probs=[0.393, 0.865]):
        z = values.ravel()
        z_sorted = np.sort(z)
        cdf = np.cumsum(z_sorted)
        cdf /= max(cdf[-1], 1e-300)
        levels = [z_sorted[np.searchsorted(cdf, 1 - p)] for p in probs]
        return np.unique(np.sort(levels))

    def _kde2d(x, y):
        data = np.vstack([x, y])
        kde = gaussian_kde(data) if kde_bw_scale == 1.0 else gaussian_kde(
            data, bw_method=gaussian_kde(data).scotts_factor() * float(kde_bw_scale)
        )
        return kde

    def _grid_limits(x, q, pad):
        qlo, qhi = np.clip(q[0], 0, 1), np.clip(q[1], 0, 1)
        xmin, xmax = np.quantile(x, [qlo, qhi])
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
            xmin, xmax = np.min(x), np.max(x)
        rng = xmax - xmin
        pad_abs = pad * (rng if rng > 0 else (abs(xmax) + 1.0))
        return xmin - pad_abs, xmax + pad_abs

    def _kde2d_grid(x, y, ngrid, q=grid_q, pad=pad_frac):
        kde = _kde2d(x, y)
        xmin, xmax = _grid_limits(x, q, pad)
        ymin, ymax = _grid_limits(y, q, pad)
        xgrid = np.linspace(xmin, xmax, ngrid)
        ygrid = np.linspace(ymin, ymax, ngrid)
        xx, yy = np.meshgrid(xgrid, ygrid)
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        return xx, yy, zz, (xmin, xmax, ymin, ymax)

    def _kde1d_grid(x, n=400, q=grid_q, pad=pad_frac):
        xmin, xmax = _grid_limits(x, q, pad)
        xs = np.linspace(xmin, xmax, n)
        kde = gaussian_kde(x) if kde_bw_scale == 1.0 else gaussian_kde(
            x, bw_method=gaussian_kde(x).scotts_factor() * float(kde_bw_scale)
        )
        return xs, kde(xs), (xmin, xmax)

    def _filled_kde_with_3sigma(ax, x, y, color, base_alpha=0.4, *, set_limits=True):
        xx, yy, zz, (xmin, xmax, ymin, ymax) = _kde2d_grid(x, y, smooth)
        if gauss_sigma and gauss_sigma > 0:
            zz = gaussian_filter(zz, sigma=float(gauss_sigma), mode='reflect')

        levels_12 = _get_density_levels(zz, [0.393, 0.865])
        for i in range(len(levels_12)-1, -1, -1):
            ax.contourf(
                xx, yy, zz,
                levels=[levels_12[i], zz.max()],
                colors=[color],
                alpha=base_alpha * (i+1) / len(levels_12)
            )
        ax.contour(xx, yy, zz, levels=levels_12, colors=[color], linewidths=1.2)
        level_3 = _get_density_levels(zz, [0.989])[0]
        ax.contour(xx, yy, zz, levels=[level_3], colors=[color], linewidths=1.4, linestyles='--')

        if set_limits:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
        return xmin, xmax, ymin, ymax

    # --- reduce to plotted params ---
    agn_data, labels_latex, units_latex = _subset(flat_samples_agn, model_labels, z_pivot_agn)
    sna_data = None
    if flat_samples_sn is not None and len(flat_samples_sn) > 0:
        sna_data, _, _ = _subset(flat_samples_sn, model_labels, z_pivot_sna)

    n_params = agn_data.shape[1]
    fig, axes = plt.subplots(n_params, n_params, figsize=(2.3*n_params, 2.3*n_params))

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            ax.tick_params(direction='in')

            if i < j:
                ax.axis("off"); continue

            if i == j:
                xs, ys, (xmin, xmax) = _kde1d_grid(agn_data[:, i])
                ax.plot(xs, ys, color="k", lw=1.8)
                if sna_data is not None:
                    xs_b, ys_b, (xmin_b, xmax_b) = _kde1d_grid(sna_data[:, i])
                    ax.plot(xs_b, ys_b, color="dodgerblue", lw=1.8)
                    ax.set_xlim(min(xmin, xmin_b), max(xmax, xmax_b))
                else:
                    ax.set_xlim(xmin, xmax)

                m, lo, hi = np.median(agn_data[:, i]), np.percentile(agn_data[:, i],16), np.percentile(agn_data[:, i],84)
                ms, ps, ns = _fmt_err(m, lo, hi, latex_label=labels_latex[i])
                txt_black = rf"{labels_latex[i]} = {ms}" + rf"$^{{+{ps}}}_{{-{ns}}}$" + f" {units_latex[i]}"
                figt = ax.figure
                off_blue = mtransforms.ScaledTranslation(0,  2/72., figt.dpi_scale_trans)
                off_blk  = mtransforms.ScaledTranslation(0, 15/72., figt.dpi_scale_trans)
                ax.text(0.02, 1.0, txt_black,
                        transform=ax.transAxes + off_blk,
                        ha="left", va="bottom", color="k", fontsize=11, clip_on=False)

                if sna_data is not None:
                    mb, lob, hib = np.median(sna_data[:, i]), np.percentile(sna_data[:, i],16), np.percentile(sna_data[:, i],84)
                    msb, psb, nsb = _fmt_err(mb, lob, hib, latex_label=labels_latex[i])
                    txt_blue = rf"{labels_latex[i]} = {msb}" + rf"$^{{+{psb}}}_{{-{nsb}}}$" + f" {units_latex[i]}"
                    ax.text(0.02, 1.0, txt_blue,
                            transform=ax.transAxes + off_blue,
                            ha="left", va="bottom", color="dodgerblue", fontsize=11, clip_on=False)
            else:
                # -------- FIX: draw both without setting limits, then union of ranges --------
                lims = []
                if sna_data is not None:
                    lims.append(_filled_kde_with_3sigma(
                        ax, sna_data[:, j], sna_data[:, i], "dodgerblue", base_alpha=0.4, set_limits=False
                    ))
                lims.append(_filled_kde_with_3sigma(
                    ax, agn_data[:, j], agn_data[:, i], "k", base_alpha=0.4, set_limits=False
                ))
                xmin = min(l[0] for l in lims); xmax = max(l[1] for l in lims)
                ymin = min(l[2] for l in lims); ymax = max(l[3] for l in lims)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                # ---------------------------------------------------------------------------

            if j == 0:
                ax.set_ylabel(f"{labels_latex[i]} {units_latex[i]}")
            else:
                ax.set_yticklabels([])

            if i == n_params - 1:
                ax.set_xlabel(f"{labels_latex[j]} {units_latex[j]}")
            else:
                ax.set_xticklabels([])

    legend = []
    if sna_data is not None:
        legend.append(Line2D([0],[0], color="dodgerblue", lw=4, label="SN Ia"))
    legend.append(Line2D([0],[0], color="k",  lw=4, label="SN Ia + AGN"))
    fig.legend(handles=legend, bbox_to_anchor=(0.5, 0.92), loc="upper left",
               fontsize=12, frameon=False, markerscale=1.5)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.9,
                        wspace=0.05, hspace=0.05)

    os.makedirs(plot_path, exist_ok=True)
    fig.savefig(os.path.join(plot_path, f"cosmo_corner_{cosmo_model}_{speed}.png"),
                bbox_inches="tight", dpi=150)
    fig.savefig(os.path.join(plot_path, f"cosmo_corner_{cosmo_model}_{speed}.pdf"),
                bbox_inches="tight", dpi=600)
    if show:
        plt.show()
    plt.close(fig)


def _weighted_bin_stats(z, y, yerr, bins, *, min_count=3, center='mid'):
    """
    Simplest weighted binning:
    - weights w = 1 / yerr^2
    - mean = (∑ w y) / (∑ w)
    - SEM  = sqrt(1 / ∑ w)
    center: 'weighted' (default), 'mid', or 'geom'
    Returns zc, mean, sem, n for bins meeting min_count.
    """
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    e = np.asarray(yerr, float)

    m = np.isfinite(z) & np.isfinite(y) & np.isfinite(e) & (e > 0)
    if not np.any(m):
        return np.array([]), np.array([]), np.array([]), np.array([])

    z, y, e = z[m], y[m], e[m]
    w = 1.0 / (e * e)

    B = len(bins) - 1
    k = np.digitize(z, bins, right=True) - 1          # 0..B-1
    inr = (k >= 0) & (k < B)
    if not np.any(inr):
        return np.array([]), np.array([]), np.array([]), np.array([])

    z, y, w, k = z[inr], y[inr], w[inr], k[inr]

    sw  = np.bincount(k, weights=w,    minlength=B)
    swy = np.bincount(k, weights=w*y,  minlength=B)
    swz = np.bincount(k, weights=w*z,  minlength=B)
    n   = np.bincount(k,               minlength=B)

    mean = np.divide(swy, sw, out=np.full(B, np.nan), where=sw > 0)
    if center == 'weighted':
        zc = np.divide(swz, sw, out=np.full(B, np.nan), where=sw > 0)
    elif center == 'geom':
        zc = np.sqrt(bins[:-1] * bins[1:])
    else:  # 'mid'
        zc = 0.5 * (bins[:-1] + bins[1:])

    sem = np.sqrt(np.divide(1.0, sw, out=np.full(B, np.nan), where=sw > 0))

    keep = (n >= min_count) & np.isfinite(mean) & np.isfinite(sem) & np.isfinite(zc)
    return zc[keep], mean[keep], sem[keep], n[keep]



def plot_hubble(flat_samples, df_agn, df_pantheon, cosmo_model, z_pivot_agn, plot_path="plots/hubble/",
                show_binned_agn=True, show_residuals=True,
                debias=False, dm_interp=None, show=False, completeness=True, show_true=False, verbose=True,
                cosmo_model_samples={}, residuals_sigma_clip=None, df_calibrators=None,):
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
    label = cosmo_model_label_latex(cosmo_model)

    # --- Thinning for speed (cap to ~500 samples) ---
    n_samples = int(flat_samples.shape[0])
    thin_factor = max(1, n_samples // 100)
    flat_samples = flat_samples[::thin_factor]

    z_grid = np.linspace(1e-4, 5.2, 500)

    # --- Parameter bookkeeping ---
    _, model_labels, _ = get_model_params(cosmo_model)
    param_indices = {name: model_labels.index(name) for name in model_labels}

    # --- Small helper: μ_model(z | params) ---
    def get_cosmo(model_name, params_dict, zp):
        if model_name == 'FlatwCDM':
            return FlatwCDM(H0=params_dict['H0'], Om0=params_dict['Om0'], w0=params_dict['w0'])
        elif model_name == 'Flatw0waCDM':
            return Flatw0waCDM(H0=params_dict['H0'], Om0=params_dict['Om0'],
                               w0=params_dict['w0'], wa=params_dict['wa'])
        elif model_name == 'FlatLambdaCDM':
            return FlatLambdaCDM(H0=params_dict['H0'], Om0=params_dict['Om0'])
        elif model_name == 'FlatwpwaCDM':
            return FlatwpwaCDM(H0=params_dict['H0'], Om0=params_dict['Om0'],
                               wp=params_dict['wp'], wa=params_dict['wa'], zp=zp)
        else:
            raise ValueError("Invalid cosmology model for _mu_model().")
    def _mu_model(model_name, params_dict, z, zp):
        return get_cosmo(model_name, params_dict, zp).distmod(z).value

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
        #dm_interp = make_dm_function(m_obs, df_agn['z'], dms, method='linear')
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

    cosmo = get_cosmo(cosmo_model, results, z_pivot_agn)
    sigma_lens = sigma_lens_from_dc(df_agn['z'].values, cosmo)

    mu_pred_std = np.sqrt(
        df_agn['apparent_mag_2500_err'].values**2 +
        #(0.055 * df_agn["z"].values)**2 +
        sigma_lens**2 +
        df_agn['z_err'].values**2 +
        predicted_M2500_err**2
    )

    mu_pred_std_with_scatter = np.sqrt(
        mu_pred_std**2 +
        np.exp(results['log_f'])**2
        #(np.exp(results['log_f']) + results['sigma_b'] * (1+df_agn["z"].values))**2
    )
    
    # Residuals (vs. median μ_model)
    mu_interp = np.interp(df_agn["z"].values, z_grid, mu_model_median)
    residuals = mu_pred_median - mu_interp
    residuals_err = mu_pred_std_with_scatter

    mu_zscore = np.abs(residuals) / mu_pred_std_with_scatter

    # ----------------- BINNING -----------------
    # Linear-z bins for MAIN & RESIDUALS panel
    bins_linear = np.arange(0.4, np.max(df_agn["z"].values)+0.05, 0.2)
    print("Using linear-z bins:", bins_linear)
    z_lin_scatter, mu_lin_mean_scatter, mu_lin_sem_scatter, n_lin = _weighted_bin_stats(
        df_agn["z"].values, mu_pred_median, mu_pred_std_with_scatter, bins_linear
    )
    

    # NEW: binned residuals (linear-z), used in residual panel
    # z_res_lin_scatter, resid_lin_mean_scatter, resid_lin_sem_scatter, n_res = _weighted_bin_stats(
    #     df_agn["z"].values, residuals, mu_pred_std_with_scatter, bins_linear
    # )
    z_res_lin_scatter = z_lin_scatter  # same bins
    mu_res_interp = np.interp(z_res_lin_scatter, z_grid, mu_model_median)
    resid_lin_mean_scatter = mu_lin_mean_scatter - mu_res_interp
    resid_lin_sem_scatter = mu_lin_sem_scatter

    # Log-z bins for INSET (match inset xscale='log')
    zpos = df_agn["z"].values[df_agn["z"].values > 0]
    zmin_inset = max(0.02, float(np.min(zpos))) if zpos.size else 0.02
    zmax_inset = 3.8
    bins_per_decade = 6
    decades = np.log10(zmax_inset) - np.log10(zmin_inset)
    n_bins_log = max(1, int(np.ceil(decades * bins_per_decade)))
    bins_log = np.logspace(np.log10(bins_linear[0]), np.log10(bins_linear[-1]), n_bins_log + 1)
    #bins_log = bins_linear
    z_log, mu_log_mean, mu_log_sem, n_log = _weighted_bin_stats(
        df_agn["z"].values, mu_pred_median, mu_pred_std_with_scatter, bins_log)

    # ======== Plot ========
    fig = plt.figure(figsize=(9, 7))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    ax.set_ylim(26, 51)
    ax.set_xlim(-0.2, np.max(df_agn["z"].values) + 0.3)
    inset_ax = inset_axes(ax, width="40%", height="40%", loc="lower right", borderpad=1.5)
    if show_residuals:
        if df_calibrators is not None:
            ax_resid = fig.add_subplot(gs[1])
        else:
            ax_resid = fig.add_subplot(gs[1], sharex=ax)

    else:
        ax_resid = ax  # dummy, not used


    # ---------- Inset (log-z) ----------
    inset_ax.set_xscale('log')

    # Solid vs open AGN markers by z (both main and inset)
    mask_in  = df_agn["z"].between(0.44, 3.16)
    mask_out = ~mask_in

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
        mask_in  = (0.44 < z_log) & (z_log < 3.16)
        mask_out = ~mask_in
        # binned (inside)
        inset_ax.errorbar(
            z_log[mask_in], mu_log_mean[mask_in], yerr=mu_log_sem[mask_in],
            fmt='o', linestyle='none',
            markersize=4, mfc='red', mec='none',
            ecolor='red', elinewidth=2.2, capsize=3.5,
            alpha=0.98, zorder=14, label="AGN (z-binned, log)"
        )
        # binned (outside, open)
        inset_ax.errorbar(
            z_log[mask_out], mu_log_mean[mask_out], yerr=mu_log_sem[mask_out],
            fmt='o', linestyle='none',
            markersize=4, mfc='none', mec='red',
            ecolor='red', elinewidth=2.2, capsize=3.5,
            alpha=0.98, zorder=14, label="AGN (z-binned, log)",
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


    if df_calibrators is not None:
        inset_ax.set_xlim(df_calibrators['z'].min()/2, 5.2)
        inset_ax.set_ylim(26, 51)
    else:
        inset_ax.set_xlim(0.02, 5.2)
        inset_ax.set_ylim(32, 51)

    inset_ax.set_xlabel(r"$z$", fontsize=12, labelpad=-10)
    inset_ax.set_ylabel(r"$\mu$ (mag)", fontsize=12)
    inset_ax.tick_params(axis='both', which='major', labelsize=10)

    # ---------- Main plot ----------

    # Color AGN points: clipped (mu_zscore > 3) as blue, others as black        
    clipped = mu_zscore > 3
    # if residuals_sigma_clip is None:
    #     colors = np.where(clipped, 'b', 'k')
    # else:
    colors = ['black'] * len(df_agn)
    if verbose:
        n_clipped = np.sum(clipped)
        print(f"Note: {n_clipped} / {len(df_agn)} AGN clipped in residuals panel (> 3σ)")
    mask_in  = df_agn["z"].between(0.44, 3.16)
    mask_out = ~mask_in
    # AGN (inside)
    for i in np.where(mask_in)[0]:
        ax.errorbar(
            df_agn["z"].iloc[i], mu_pred_median[i], yerr=mu_pred_std[i],
            fmt='o', linestyle='none', markersize=3,
            mec="none",
            mfc=(0, 0, 0, 0.3),
            ecolor=(0.2, 0.2, 0.2, 0.1), elinewidth=0.8,
            capsize=2, capthick=0.8,
            zorder=0, label="AGN" if i == np.where(mask_in)[0][0] else None
        )

    # AGN (outside, open)
    for i in np.where(mask_out)[0]:
        ax.errorbar(
            df_agn["z"].iloc[i], mu_pred_median[i], yerr=mu_pred_std[i],
            fmt='o', linestyle='none', markersize=3, mfc='none',
            mec=(0, 0, 0, 0.4),
            capsize=2, capthick=0.8,
            ecolor=(0.2, 0.2, 0.2, 0.1), elinewidth=0.8, zorder=0, label=None
        )

    # inset_ax.errorbar(
    #     df_agn["z"][mask_in], mu_pred_median[mask_in], yerr=mu_pred_std[mask_in],
    #     fmt='o', linestyle='none', markersize=4,
    #     #mfc="black",
    #     mec="none",
    #     mfc=(0, 0, 0, 0.5),   # RGBA: black with alpha=0.3
    #     #mec=(0, 0, 0, 0.3),   # optional: semi-transparent edge
    #     ecolor=(0.2, 0.2, 0.2, 0.1), elinewidth=0.8,
    #     zorder=1, capsize=2, capthick=0.8, label="AGN"
    # )
    # # AGN (outside, open)
    # inset_ax.errorbar(
    #     df_agn["z"][mask_out], mu_pred_median[mask_out], yerr=mu_pred_std[mask_out],
    #     fmt='o', linestyle='none', markersize=3, 
    #     #mfc='none', mec="k",
    #     mfc='none',
    #     #mfc=(0, 0, 0, 0.3),   # RGBA: black with alpha=0.3
    #     mec=(0, 0, 0, 0.5),   # optional: semi-transparent edge
    #     ecolor=(0.2, 0.2, 0.2, 0.1), elinewidth=0.8, zorder=1, capsize=2, capthick=0.8,
    # )


    # MAIN: linear-binned AGN
    if show_binned_agn:
        mask_in  = (0.44 < z_lin_scatter) & (z_lin_scatter < 3.16)
        mask_out = ~mask_in
        # with scatter
        # binned (inside)
        print("Plotting binned AGN (linear z) at:", z_lin_scatter)
        print("\tmask_out:", mask_out)
        ax.errorbar(
            z_lin_scatter[mask_in], mu_lin_mean_scatter[mask_in], yerr=mu_lin_sem_scatter[mask_in],
            fmt='o', linestyle='none',
            markersize=5, mfc='red', mec='none',
            ecolor='red', elinewidth=2.2, capsize=3.5,
            alpha=0.98, zorder=14, label="AGN (z-binned)"
        )
        # binned (outside, open)
        ax.errorbar(
            z_lin_scatter[mask_out], mu_lin_mean_scatter[mask_out], yerr=mu_lin_sem_scatter[mask_out],
            fmt='o', linestyle='none',
            markersize=5, mfc='none', mec='red',
            ecolor='red', elinewidth=2.2, capsize=3.5,
            alpha=0.98, zorder=14
        )

    # SN Ia
    if debias:
        sna_mu = df_pantheon["MU_SH0ES"]
    else:
        sna_mu = df_pantheon["MU_SH0ES"] + df_pantheon['biasCor_m_b']
    ax.errorbar(
        df_pantheon["zHD"], sna_mu, yerr=df_pantheon["MU_SH0ES_ERR_DIAG"],
        fmt='s', markersize=2, color="#0A84FF", linestyle='none', lw=0.8, alpha=0.7, zorder=1, label="SN Ia"
    )


    # Model + 68% band
    ax.plot(z_grid, mu_model_median, color="m", lw=2.4, alpha=1.0, zorder=5, label=label)
    ax.fill_between(z_grid, mu_model_16th, mu_model_84th, color="m", alpha=0.25, zorder=4)

    # Survey magnitude limit (shade above)
    if completeness and not debias:
        agn_params_arr = agn_model_pack_params(results)
        agn_obs_med = {key: float(np.median(df_agn[key].values)) * np.ones_like(z_grid) for key in agn_model_req_obs + agn_model_req_errs}
        agn_obs_arr, agn_err_arr, agn_pivot_arr = agn_model_pack_obs(agn_obs_med)

        M_med_grid = np.median([
            M_model_agn(agn_params_arr, agn_obs_arr, agn_pivot_arr)
            for s in flat_samples
        ], axis=0)

        m_lim = 24.0
        mu_lim = m_lim - M_med_grid
        ax.fill_between(z_grid, mu_lim, 60, color="red", alpha=0.15, zorder=2, label="< 50% complete")
        inset_ax.fill_between(z_grid, mu_lim, 60, color="red", alpha=0.12, zorder=2, label="< 50% complete")

        # Optional: line for (residuals - residuals_2) using median params of each model
    colors = {'Flatw0waCDM': 'tab:red', 'FlatLambdaCDM': "tab:blue", 'FlatwCDM': 'tab:green'}
    line_styles = {'Flatw0waCDM': 'dotted', 'FlatLambdaCDM': "dotted", 'FlatwCDM': 'dashdot'}

    for cosmo_model_other, cosmo_model_samples_other in cosmo_model_samples.items():
        _, model_labels_other, _ = get_model_params(cosmo_model_other)
        model_label_latex_other = cosmo_model_label_latex(cosmo_model_other)
        results_other = {key: np.median(cosmo_model_samples_other[:, i]) for i, key in enumerate(model_labels_other)}

        mu_model_other = _mu_model(cosmo_model_other, results_other,   z_grid, z_pivot_agn)
        ax.plot(z_grid, mu_model_other, lw=1.2, color=colors[cosmo_model_other], ls=line_styles[cosmo_model_other], alpha=1.0, 
                        label=model_label_latex_other, zorder=6)
        inset_ax.plot(z_grid, mu_model_other, lw=1.2, color=colors[cosmo_model_other], ls=line_styles[cosmo_model_other], alpha=1.0, zorder=6)

    # Flat ΛCDM
    mu_conc = Planck18.distmod(z_grid).value
    ax.plot(z_grid, mu_conc, color="#F0B000", lw=1.2, ls='--', zorder=5, alpha=1.0, label="flat $\Lambda$CDM (Planck 2018)")

    # Labels
    ax.set_ylabel(r"$\mu$ (mag)")
    ax.set_xlabel(r"$z$")

    # ---------- Residuals panel ----------
    if show_residuals:
        # Zero line
        ax_resid.axhline(0.0, color="m", lw=2.2, zorder=1)

        # NEW: binned residuals in red (points + thin connecting line)
        if z_res_lin_scatter.size:
            mask_in  = (0.44 < z_res_lin_scatter) & (z_res_lin_scatter < 3.16)
            mask_out = ~mask_in
            ax_resid.errorbar(
                z_res_lin_scatter[mask_in], resid_lin_mean_scatter[mask_in], yerr=resid_lin_sem_scatter[mask_in],
                fmt='o', linestyle='none', markersize=6,
                mfc='red', mec='none', ecolor='red', elinewidth=2.0, capsize=3.0,
                alpha=0.98, zorder=15, label="Binned AGN residuals (w/ scatter)"
            )
            ax_resid.errorbar(
                z_res_lin_scatter[mask_out], resid_lin_mean_scatter[mask_out], yerr=resid_lin_sem_scatter[mask_out],
                fmt='o', linestyle='none', markersize=6,
                mfc='white', mec='red', ecolor='red', elinewidth=2.0, capsize=3.0,
                alpha=0.98, zorder=15
            )


        for cosmo_model_other, cosmo_model_samples_other in cosmo_model_samples.items():
            _, model_labels_other, _ = get_model_params(cosmo_model_other)
            z_grid_fine = np.linspace(1e-4, 5.2, 500)
            results_other = {key: np.median(cosmo_model_samples_other[:, i]) for i, key in enumerate(model_labels_other)}

            mu_model_other = _mu_model(cosmo_model_other, results_other,   z_grid, z_pivot_agn)
            mu_model = _mu_model(cosmo_model, results, z_grid, z_pivot_agn)
            ax_resid.plot(z_grid_fine, mu_model_other - mu_model, lw=2.2, color=colors[cosmo_model_other], ls=line_styles[cosmo_model_other], 
                          alpha=1.0, label=f"{cosmo_model_other} $\Delta$μ")
            
        # Planck 2018 ΛCDM
        mu_model_1 = _mu_model(cosmo_model, results, z_grid, z_pivot_agn)

        mu_conc = Planck18.distmod(z_grid).value
        #ax.plot(z_grid, mu_conc, color="#F0B000", lw=1.2, ls='--', zorder=5, alpha=1.0, label="flat $\Lambda$CDM (Planck 2018)")
        ax_resid.plot(z_grid, mu_conc - mu_model_1, lw=2.2, color="#F0B000", ls='--', alpha=1.0,)


        ax_resid.set_ylabel(r"$\Delta\mu$ (mag)")
        ax_resid.set_xlabel(r"$z$")
        if df_calibrators is not None:
            ax_resid.set_ylim(-0.5, 0.5)
            ax_resid.set_xlim(df_calibrators['z'].min()*0.2, df_calibrators['z'].max()*1.1)
        else:
            ax_resid.set_ylim(-0.5, 0.5)
        #ax_resid.legend(frameon=True, loc="upper left", fontsize=10)

    for axi in (ax, inset_ax, ax_resid):
        axi.minorticks_on()
        axi.tick_params(axis='both', which='minor', direction='in', length=4, top=True, right=True, width=2)
        axi.tick_params(axis='both', which='major', direction='in', length=8, top=True, right=True)

    if show_residuals:
        # Hide the main panel's x-axis labels, numbers, and ticks (leave residuals' x-axis intact)
        ax.set_xlabel("")  # remove main x-axis label
        ax.tick_params(axis='x', which='minor', direction='in', labelbottom=False, length=4, top=True, right=True, width=2)
        ax.tick_params(axis='x', which='major', direction='in', labelbottom=False, length=8, top=True, right=True)
        ax.xaxis.offsetText.set_visible(False)  # hide any scientific-notation offset text

    # ========= HIGHLIGHT: df_calibrators on Hubble diagram (MAIN + INSET) =========
    if df_calibrators is not None and len(df_calibrators) > 0:
        ds = df_calibrators.copy()

        # Build predicted M_2500 for SHOW objects at median params (results)
        agn_params_arr_show = agn_model_pack_params(results)
        obs_show, err_show, piv_show = agn_model_pack_obs(ds)
        pred_M_show      = M_model_agn(agn_params_arr_show, obs_show, piv_show)
        pred_M_err_show  = M_model_agn_err(agn_params_arr_show, obs_show, err_show, piv_show)

        # Distance modulus prediction: mu = m_2500 - M_2500
        m_show = ds['apparent_mag_2500'].values
        mu_show = ds['mu'].values
        # Uncertainties for SHOW (match main formula)
        z_show     = ds['z'].values

        mu_show_std = ds['mu_err'].values
        # Optionally include intrinsic scatter (used in residuals if desired)
        mu_show_std_with_scatter = np.sqrt(mu_show_std**2 + np.exp(results['log_f'])**2)

        # Distinct colors per object
        cmap = plt.get_cmap("Set1")  # 10 distinct colors

        # --- Plot in INSET (z vs mu) ---
        for i in range(len(ds)):
            #c = cmap(i)
            c = 'darkorange'
            inset_ax.errorbar(
                z_show[i], mu_show[i], yerr=mu_show_std[i],
                fmt='*', linestyle='none', markersize=10,
                mfc=c, mec='k', mew=0.6,
                ecolor=c, elinewidth=1.4, alpha=0.9, zorder=22,
                #label=str(ds.iloc[i]['object_id'])
            )

            ax.errorbar(
                z_show[i], mu_show[i], yerr=mu_show_std[i],
                fmt='*', linestyle='none', markersize=12,
                mfc=c, mec='k', mew=0.7,
                ecolor=c, elinewidth=1.6, alpha=0.9, zorder=22,
                label='Calibrators' if i == 0 else None
                #label=str(ds.iloc[i]['object_id'])
            )

        # --- Residuals overlay for SHOW (optional) ---
        if show_residuals:
            mu_model_at_show = np.interp(z_show, z_grid, mu_model_median)
            resid_show = mu_show - mu_model_at_show
            print(resid_show)
            
            for i in range(len(ds)):
                print(f"Showing residual for object_id={ds.iloc[i]['object_id']}: z={z_show[i]:.3f}, resid={resid_show[i]:.3f} mag")
                c = 'darkorange'

                ax_resid.errorbar(
                    z_show[i], resid_show[i], yerr=mu_show_std_with_scatter[i],
                    fmt='*', linestyle='none', markersize=15,
                    mfc=c, mec='k', mew=0.7,
                    ecolor=c, elinewidth=1.6, alpha=0.9, zorder=22,
                    #label=str(ds.iloc[i]['object_id'])
                )

    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.24, 0.06), fontsize=10)

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
    if debias:
        residuals_df = df_agn.copy()
        residuals_df["residuals"] = residuals
        residuals_df["mu_pred_median"] = mu_pred_median
        residuals_df["mu_pred_std"] = mu_pred_std
        residuals_df["mu_pred_std_with_scatter"] = mu_pred_std_with_scatter
        residuals_df["mu_zscore"] = mu_zscore
        fields = ['object_id', 'apparent_mag_2500', 'f_host_2500', 'ra', 'dec', 
                  'mu_pred_median', 'mu_pred_std', 'mu_pred_std_with_scatter',
                    'z', 'redchi', 'sdss_name', 'npca_qso', 'poly', 'redchi2_conti_full', 'residuals', 'mu_zscore']
        residuals_df = residuals_df[fields]
        residuals_df = residuals_df.sort_values(by="residuals", ascending=False)
        csv_path = os.path.join(plot_path, "residuals.csv")
        residuals_df.to_csv(csv_path, index=False)
        print(f"Residuals saved to {csv_path}")

        # Save outliers with residuals > 4 to outliers.csv
        if np.any(outlier_mask):
            outliers_df = residuals_df[outlier_mask]
            outliers_csv_path = os.path.join(plot_path, "outliers.csv")
            outliers_df.to_csv(outliers_csv_path, index=False)
            print(f"Outliers (|residuals| > 5) saved to {outliers_csv_path}")


        # Standard deviation Outlier report
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

    return residuals, residuals_err, mu_pred_median, mu_pred_std, mu_pred_std_with_scatter

def plot_predicted_vs_actual_M2500(
    flat_samples,
    df_agn,
    cosmo_model,
    z_pivot_agn,
    plot_path="plots/hubble",
    dm_interp=None,  # de-biasing function (optional)
    debias=False,
    show=False,
    cmap="inferno",       # (unused for discrete bins now, kept for API compatibility)
    box_alpha=0.7,        # transparency of white annotation boxes
    show_sigma_band=True,
    completeness=True,    # add "<50% complete" red region
    m_lim=24.0,           # survey apparent-magnitude limit for completeness shading
    n_cosmo_draws=50,     # posterior draws to propagate cosmology errors (for xerr)
    random_state=42,      # RNG seed for reproducibility of draws
):
    """
    Predicted vs Actual M_2500, with:
      • y-error bars from M_model_agn_err(...)
      • x-error bars = sqrt(apparent_mag_2500_err^2 + sigma_mu_cosmo(z)^2)
      • ±1σ band from intrinsic scatter sigma_int = exp(log_f) (magenta)
      • Points colored by delta error = predicted_M2500_err / |predicted_M2500|
        with discrete bins: <0.2, 0.2–0.3, 0.3–0.4, 0.4–0.5, >0.5.
      • Optional "<50% complete" red region by bin.
    """
    import os, math
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
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

   # if debias:
        #dm_interp = make_dm_function(np.array(df_agn["apparent_mag_2500"].values), np.array(df_agn['z'].values), dms)

    # --- actual minus optional debias ---
    if debias:
        pts_all = np.column_stack([df_agn['z'].values, df_agn['apparent_mag_2500'].values])
        actual_M_2500_eff = actual_M_2500 - dm_interp(pts_all)
    else:
        actual_M_2500_eff = actual_M_2500

    residuals_all = M_2500_pred - actual_M_2500_eff               # mag
    sigma_all     = np.sqrt(M_2500_pred_err**2 + xerr**2)          # mag

    # Safety mask for nan/inf on global vectors (used for overall outputs only)
    m_global = np.isfinite(residuals_all) & np.isfinite(sigma_all) & (sigma_all > 0)

    # ===================== NEW: delta-error categories ===================== #
    denom = np.maximum(np.abs(M_2500_pred), 1e-6)  # avoid div-by-zero; magnitude can be negative
    delta_err = M_2500_pred_err / denom

    # Define bins and labels
    q = np.quantile(delta_err, [0.4, 0.6])
    _bins = np.array([0.0, q[0], q[1], np.inf])
    lo, hi = (np.round(q * 100)).astype(int)
    _labels = [f"< {lo}%", f"{lo}–{hi}%", f"> {hi}%"]

    # Discretize: cats in {0..4}; NaN -> -1 (unclassified)
    cats = np.full(delta_err.shape, -1, dtype=int)
    good = np.isfinite(delta_err)
    cats[good] = np.digitize(delta_err[good], _bins, right=False) - 1

    # Choose a 5-color, high-contrast palette (categorical)
    palette = np.array(["blue", "orange", "red"])  # blue→red
    # ======================================================================= #

    # --- binning in redshift ---
    num_cols, num_rows = 4, 8
    n_bins = num_cols * num_rows  # 32

    first_edge = 0.0
    second_edge = 0.3              # keeps the special low-z bin [0.0, 0.3)
    last_finite_edge = 3.3         # 30 bins of width 0.1 from 0.3 to 3.3, plus final open bin

    # Core edges: 0.4, 0.5, ..., 3.3  (these define [0.3,0.4), [0.4,0.5), ..., [3.2,3.3))
    edges_core = np.round(np.arange(0.4, last_finite_edge + 1e-9, 0.1), 1)

    # Final edges array: [0.0, 0.3, 0.4, ..., 3.3, inf]
    z_bins = np.concatenate(([first_edge, second_edge], edges_core, [np.inf]))

    assert (len(z_bins) - 1) == n_bins, f"Expected {n_bins} bins, got {len(z_bins)-1}"

    z_bin_indices = np.digitize(z, bins=z_bins, right=False)
    num_bins = len(z_bins) - 1

    bin_labels = []
    for i in range(num_bins):
        lo, hi = z_bins[i], z_bins[i+1]
        if i == 0:
            label = rf"$z < {hi:.1f}$"
        elif np.isfinite(hi):
            label = rf"${lo:.1f} \leq z < {hi:.1f}$"
        else:
            label = rf"$z \geq {lo:.1f}$"
        bin_labels.append(label)

    # --- figure with full-height (unused) colorbar column kept for layout symmetry ---
    fig = plt.figure(figsize=(5 * num_cols, 4 * num_rows))
    gs = fig.add_gridspec(num_rows, num_cols + 1,
                          width_ratios=[1]*num_cols + [0.06],
                          wspace=0.0, hspace=0.0)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(num_cols)] for r in range(num_rows)]).flatten()

    xlo, xhi = -25.8, -18.2
    ylo, yhi = -25.8, -18.2
    xx = np.linspace(min(xlo, ylo), max(xhi, yhi), 400)

    resid_bybin_aligned = np.full(len(df_agn), np.nan, dtype=float)

    # Pre-build legend handles (once)
    legend_handles = [Line2D([0], [0], marker='o', linestyle='',
                             markerfacecolor=palette[i], markeredgecolor='k', label=_labels[i])
                      for i in range(len(_labels))]

    legend_added = False

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

        # residuals (for CSV/diagnostics)
        resid = y - x
        resid_bybin_aligned[bin_mask] = resid

        # pick colors by category
        cats_bin = cats[bin_mask]
        colors_bin = np.where(cats_bin >= 0, palette[np.clip(cats_bin, 0, 4)], "#999999")  # gray for NaN

        # plot errorbars
        ax.errorbar(
            x, y, xerr=xerr_bin, yerr=yerr_bin,
            fmt="none", ecolor="#666666", elinewidth=0.7, alpha=0.4, zorder=2
        )

        # scatter with discrete colors: closed if 0.44 < z < 3.16, open otherwise
        z_bin = z[bin_mask]
        mask_closed = (z_bin > 0.44) & (z_bin < 3.16)
        mask_open   = ~mask_closed

        # filled markers (keep black edges like before)
        ax.scatter(
            x[mask_closed], y[mask_closed],
            facecolors="k", edgecolors='k', #c=colors_bin[mask_closed], 
            s=20, alpha=1.0,
            linewidths=0.8, zorder=3,
        )

        # open markers (no fill; colored edges)
        ax.scatter(
            x[mask_open], y[mask_open],
            facecolors="none", edgecolors='k', #edgecolors=colors_bin[mask_open],
            s=20, alpha=1.0, linewidths=1, zorder=3,
        )

        # y = x reference and ±1σ intrinsic band
        ax.plot(xx, xx, color="m", alpha=0.9, lw=2.2, zorder=9)
        if show_sigma_band:
            ax.plot(xx, xx - sigma_intrinsic, color="m", alpha=0.7, lw=1.5, linestyle="--", zorder=9,
                    label=r"$y = x \pm 1\sigma_{\rm int}$" if i == 0 else None)
            ax.plot(xx, xx + sigma_intrinsic, color="m", alpha=0.7, lw=1.5, linestyle="--", zorder=9)

        # "< 50% complete" region (skip open-ended last bin)
        if completeness and not debias and np.isfinite(z_bins[i+1]):
            z_center = 0.5 * (z_bins[i] + z_bins[i+1])
            mu_center = cosmo_med.distmod(z_center).value
            M_lim = m_lim - mu_center
            xmin = max(M_lim, xlo)
            xmax = xhi
            if xmin < xmax:
                ax.axvspan(xmin, xmax, facecolor="red", alpha=0.15, zorder=0, label="< 50% complete" if i == 0 else None)

        ax.invert_xaxis()
        ax.invert_yaxis()

        # annotations
        boxprops = dict(boxstyle="round,pad=0.2", facecolor="white", alpha=box_alpha, edgecolor="none")
        ax.annotate(
            bin_labels[i], xy=(0.03, 0.97), xycoords="axes fraction",
            fontsize=22, color="k", ha="left", va="top", bbox=boxprops,
        )
        n_in_bin = int(np.sum(bin_mask))
        ax.annotate(
            f"N = {n_in_bin}", xy=(0.97, 0.03), xycoords="axes fraction",
            fontsize=22, color="k", ha="right", va="bottom", bbox=boxprops,
        )

        # labels only on bottom row / left col
        if i >= (num_rows - 1) * num_cols:
            ax.set_xlabel("Actual $M_{2500}$", fontsize=22)
        if i % num_cols == 0:
            ax.set_ylabel("Predicted $M_{2500}$", fontsize=22)
        ax.tick_params(axis="both", labelsize=10, length=3)

        # Add the categorical legend only once (top-right panel of row 1, or first panel with data)
        # if not legend_added:
        #     leg1 = ax.legend(
        #         handles=legend_handles,
        #         title=r"$\Delta \equiv \sigma(M_{2500})/|M_{2500}|$",
        #         loc="upper left", 
        #         bbox_to_anchor=(0.0, 0.92),
        #         alignment="left",          # left-justify markers + labels (mpl ≥ 3.8)
        #         fontsize=14,               # larger label text
        #         title_fontsize=14,         # larger title
        #         markerscale=1.6,           # make the points in the legend bigger
        #         handlelength=1.2,          # length of marker/line sample
        #         handletextpad=0.6,         # space between marker and text
        #         labelspacing=0.35,         # vertical spacing between entries
        #         frameon=False,
        #     )
        #     leg1.get_frame().set_facecolor("white")
        #     leg1.get_frame().set_alpha(box_alpha)
        #     leg1.get_frame().set_edgecolor("none")
        #     legend_added = True

        # Add band/completeness legend once as well (if present)
        if (show_sigma_band or completeness) and i == num_cols-1:
            leg = ax.legend(loc="lower right", fontsize=12, frameon=True)
            leg.get_frame().set_facecolor("none")
            leg.get_frame().set_alpha(box_alpha)
            leg.get_frame().set_edgecolor("none")

    for ax in axes:
        if ax.has_data():
            ax.label_outer()

    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(os.path.join(plot_path, "pdf"), exist_ok=True)

    # Save object_id, z, and resid_bybin_aligned to CSV
    m2500_residuals_df = df_agn.loc[:, ["object_id", "z"]].copy()
    m2500_residuals_df["residual"] = resid_bybin_aligned
    m2500_residuals_df.to_csv(os.path.join(plot_path, "m2500_residuals.csv"), index=False)

    plt.savefig(os.path.join(plot_path, f"predicted_vs_actual_M2500{'_debias' if debias else ''}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(plot_path, "pdf", f"predicted_vs_actual_M2500{'_debias' if debias else ''}.pdf"), dpi=600, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    return residuals_all[m_global], sigma_all[m_global], resid_bybin_aligned, z_bin_indices

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
    os.makedirs(f"plots/hubble/{prefix}/completeness", exist_ok=True)
    plt.savefig(f"plots/hubble/{prefix}/completeness/completeness_vs_mag_at_redshifts.png", dpi=300)
    #plt.savefig("plots/hubble/completeness_vs_mag_at_redshifts.pdf", dpi=300)
    plt.close()



def plot_full_residuals(
    df_agn, residuals, residuals_err, flat_samples, cosmo_model, z_pivot_agn,
    debias=False, dm_interp=None, plot_path='plots/hubble', show=False,
    *, nbins=10, min_count=5, z_cut=None, key_y='residuals', key_color='z',
):

    # --- Cosmology from posterior summaries ---
    df_agn = df_agn.copy()
    df_agn['residuals'] = residuals

    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    param_indices = {name: model_labels.index(name) for name in model_labels}
    results = {key: np.percentile(flat_samples[:, i], [16, 50, 84]) for i, key in enumerate(model_labels)}

    if cosmo_model == 'FlatwCDM':
        cosmo = FlatwCDM(H0=results['H0'][1], Om0=results['Om0'][1], w0=results['w0'][1])
    elif cosmo_model == 'FlatwpwaCDM':
        cosmo = FlatwpwaCDM(H0=results['H0'][1], Om0=results['Om0'][1],
                            wp=results['wp'][1], wa=results['wa'][1], zp=z_pivot_agn)
    elif cosmo_model == 'Flatw0waCDM':
        cosmo = Flatw0waCDM(H0=results['H0'][1], Om0=results['Om0'][1],
                            w0=results['w0'][1], wa=results['wa'][1])
    elif cosmo_model == 'FlatLambdaCDM':
        cosmo = FlatLambdaCDM(H0=results['H0'][1], Om0=results['Om0'][1])
    else:
        raise ValueError("Invalid cosmology model.")

    df_agn = df_agn.reset_index(drop=True)

    # Absolute M_2500 estimate used in plots
    df_agn['MY_M_2500'] = df_agn['apparent_mag_2500'].values - cosmo.distmod(df_agn['z'].values).value

    # Optional de-biasing via your dm surface
    if debias:
        #dm_interp = make_dm_function(df_agn["apparent_mag_2500"].values, df_agn['z'].values, dms)
        pts = np.column_stack([df_agn['z'].values, df_agn['apparent_mag_2500'].values])
        delta = dm_interp(pts)
        df_agn['MY_M_2500'] -= delta
        df_agn['apparent_mag_2500'] -= delta
        if 'apparent_mag_2500_reddened' in df_agn.columns:
            df_agn['apparent_mag_2500_reddened'] -= delta

    def _safelog(a):
        return np.log10(np.abs(a) + 1e-10)
    if 'dm_red' in df_agn: df_agn['log_dm_red'] = _safelog(df_agn['dm_red'])
    if 'reddening_integral' in df_agn: df_agn['log_reddening_integral'] = _safelog(df_agn['reddening_integral'])
    if 'reddening_proxy' in df_agn: df_agn['log_reddening_proxy'] = _safelog(df_agn['reddening_proxy'])
    if 'redchi' in df_agn: df_agn['log_redchi'] = _safelog(df_agn['redchi'])
    if 'redchi2_conti_full' in df_agn: df_agn['log_redchi2_conti_full'] = _safelog(df_agn['redchi2_conti_full'])
    if 'apparent_mag_2500_err' in df_agn: df_agn['log_apparent_mag_2500_err'] = _safelog(df_agn['apparent_mag_2500_err'])
    if 'log_sigma_UV_err' in df_agn: df_agn['log_log_sigma_UV_err'] = _safelog(df_agn['log_sigma_UV_err'])
    if 'log_tau_UV_RF_err' in df_agn: df_agn['log_log_tau_UV_RF_err'] = _safelog(df_agn['log_tau_UV_RF_err'])
    if 'psf_minus_fiber_r' in df_agn: df_agn['log_psf_minus_fiber_r'] = _safelog(df_agn['psf_minus_fiber_r'])
    if 'petroRad_r' in df_agn: df_agn['log_petroRad_r'] = _safelog(df_agn['petroRad_r'])
    if 'log_tau_drw0_rhat' in df_agn: df_agn['log_log_tau_drw0_rhat'] = _safelog(df_agn['log_tau_drw0_rhat'])
    for col in ['BC', 'decomp_host', 'poly']:
        if col in df_agn:
            df_agn[col] = df_agn[col].replace(
                {True: 1, False: 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0}
            )

    # ---- Which x-keys to show (keep your order) ----
    keys = [col for col in np.flip([
        'apparent_mag_2500_err', 'log_apparent_mag_2500_err', 
        'log_sigma_UV_err', 'log_log_sigma_UV_err',
        'log_tau_UV_RF_err', 'log_log_tau_UV_RF_err',
        'apparent_mag_2500', 'apparent_mag_2500_reddened', 'dm_red', 'log_dm_red', 
        'reddening_integral', 'log_reddening_integral',
        'ebv_wu',  'BC', 'decomp_host', 'poly', 'npca_qso',
        'conti_a_0', 'PL_slope_red', 'PL_slope_blue', 'iron_frac',
        'MY_M_2500', 'z', 'log_lbol', 'log_ledd_ratio', 
        'log_sigma_UV', 'log_sigma_hat0', 'log_sigma_hat_UV', 'log_tau_UV_RF',
        'log_tau_fast0',
        'chi_sq_g',
        'sn_median_all', 'redchi', 'log_redchi', 'alpha_lambda',
        'redchi2_conti_full', 'log_redchi2_conti_full',
        'bwb_alpha', 'bwb_beta', 
        'log_f_host_5100','f_host_5100', 'log_f_host_2500', 'f_host_2500',
        'log_rho', 't_rf_length', 'tau_band_RF_mean',
        'log_tau_band_RF_mean', 'log_t_rf_length', 
        'alphaOX', 'alphaOX_int',
        'bwb_alpha_u', 'bwb_alpha_g', 'bwb_alpha_r', 'bwb_alpha_i', 'bwb_alpha_z',
        
        'eta_A1', 'eta_A2', 'eta_tau1', 'eta_tau2',
        'PL_slope_blue', 'PL_slope_red', 'PL_break_wave_inbounds', 'lam_min', 'lam_max', 'lam_range', 
        'poly1', 'psf_minus_fiber_r', 'log_psf_minus_fiber_r', 'petroRad_r', 'log_petroRad_r',
        'eta_A1_rhat', 'eta_A1_ess', 'eta_A1_acf', 'eta_A1_stuck_any', 
        'cadence', 'cadence_err', 'number_points', 'dm_psf_correction',
        'log_jitter_total', 'log_amp_delta_blr_total',
        'log_amp_delta_blr_u', 'log_amp_delta_blr_g', 'log_amp_delta_blr_r', 'log_amp_delta_blr_i', 'log_amp_delta_blr_z',
        'log_jitter_u', 'log_jitter_g', 'log_jitter_r', 'log_jitter_i', 'log_jitter_z',

    ]) if col in df_agn.columns]

    keys_masks = {
        'dm_red': (-5, 5),
        'log_dm_red': (-np.inf, 1),
        'f_host_2500': (-2, 1),
        'log_lbol': (1, np.inf),
    }

    keys_yx_line = ['MY_M_2500', 'apparent_mag_2500']

    n_keys = len(keys)
    n_cols = 4
    n_rows = math.ceil(n_keys / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    # --- Panels ---
    for idx, key in enumerate(keys):
        ax = axes[idx]
        try:
            # Base mask (finite & optional clipping per-key)
            if np.issubdtype(df_agn[key].dtype, np.number):
                mask = (df_agn[key] > -1e9) & np.isfinite(df_agn[key])
            else:
                mask = np.ones(len(df_agn), dtype=bool)
            if z_cut is not None:
                mask &= df_agn['z'] < z_cut

            if key in keys_masks:
                low, high = keys_masks[key]
                mask &= df_agn[key].between(low, high)

            # Also require finite residuals (+ err for the overlay later)
            mask &= np.isfinite(residuals)

            
            if key_y == 'residuals':
                y = residuals[mask]
                x = df_agn.loc[mask, key].to_numpy()
                ax.set_xlabel(key)
                ax.set_ylabel(key_y)
                norm = mpl.colors.Normalize(vmin=np.nanmin(df_agn.loc[mask, key_color]), vmax=np.nanmax(df_agn.loc[mask, key_color]))
                cmap = 'viridis'
            else:
                x = df_agn.loc[mask, key_y].to_numpy()
                y = df_agn.loc[mask, key].to_numpy()
                ax.set_ylabel(key)
                ax.set_xlabel(key_y)
                norm = mpl.colors.Normalize(vmin=-4, vmax=4)
                cmap = 'bwr_r'


            sc = ax.scatter(x, y, c=df_agn.loc[mask, key_color].to_numpy(), cmap=cmap, norm=norm, s=10, alpha=0.5)

            # Optional diagonal guide for a couple of variables
            if key in keys_yx_line:
                xmin, xmax = np.nanmin(x), np.nanmax(x)
                xm = np.nanmean(x)
                if key_y == 'residuals':
                    ax.plot([xmin, xmax], [xmin - xm, xmax - xm], color='red', linestyle='--', lw=1)

            # Zero line
            if key_y == 'residuals':
                ax.axhline(0, color='red', linestyle='--', lw=1)

            # Per-panel colorbar
            cbar = fig.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label(key_color, fontsize=12)

            # ===========================
            # Binned overlay (weighted)
            # ===========================
            # Build residual errors for the masked points
            if residuals_err is None:
                err = np.full_like(y, np.nan, dtype=float)
            else:
                err = np.asarray(residuals_err)[mask]

            mfin = np.isfinite(x) & np.isfinite(y) & np.isfinite(err) & (err > 0)
            if np.any(mfin):
                xb, yb, eb = x[mfin], y[mfin], err[mfin]

                # Robust bin edges (linear in x). If you prefer log for positive x:
                # use np.geomspace(lo, hi, nbins+1) when xb>0 and spans orders of magnitude.
                lo, hi = np.nanpercentile(xb, [1, 99])
                if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                    lo, hi = np.nanmin(xb), np.nanmax(xb)
                bins = np.linspace(lo, hi, nbins + 1)

                # Use your provided binning function
                zc, mean, sem, n = _weighted_bin_stats(xb, yb, eb, bins, min_count=min_count, center='weighted')

                if len(zc):
                    # trend + SEM error bars (no double-counting)
                    ax.errorbar(
                        zc, mean, yerr=sem,
                        fmt='o', color='red', markersize=4,
                        elinewidth=1.0, capsize=2,
                        alpha=0.9, zorder=10
                    )
        except Exception as e:
            print(f"Error processing key {key}: {e}")
            ax.axis('off')

        ax.set_title(key)
        ax.grid(True)

    # Hide any extra axes
    for j in range(n_keys, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if show:
        plt.show()

    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path, f"full_residuals_{'debiased' if debias else 'biased'}_y{key_y}_c{key_color}_zcut{z_cut}.png"), dpi=150)
    plt.close()

from scipy.interpolate import interp1d
from matplotlib.ticker import LogLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec

from scipy.stats import gaussian_kde

def _kde_conf_levels(Z, conf=(0.954, 0.683)):
    """
    Return strictly-increasing density thresholds so that regions Z >= level
    enclose each conf fraction. Uses ascending order (95% then 68%).
    """
    Zflat = Z.ravel()
    Zsort = np.sort(Zflat)             # ascending densities
    cdf   = np.cumsum(Zsort)
    cdf  /= cdf[-1]

    # threshold density so that mass above it is 'conf'
    thr = [Zsort[np.searchsorted(cdf, 1.0 - c)] for c in conf]
    levels = np.array(thr, dtype=float)
    levels.sort()                      # ensure increasing for contour()

    # nudge if any ties remain (can happen on coarse grids)
    for i in range(1, len(levels)):
        if levels[i] <= levels[i-1]:
            levels[i] = np.nextafter(levels[i-1], np.inf)

    # also clamp inside (min,max) just in case
    zmin, zmax = float(np.min(Z)), float(np.max(Z))
    eps = np.finfo(float).eps * (zmax - zmin + 1.0)
    levels = np.clip(levels, zmin + eps, zmax - eps)
    return levels


def plot_predicted_L2500_vs_sigmahat(
    flat_samples, df_agn, cosmo_model, z_pivot_agn,
    plot_path='plots/hubble', show=False, debias=True, dm_interp=None,
    show_residuals=False, df_calibrators=None
):
    d = df_agn.copy()

    # --- Thinning for speed ---
    n_samples = int(flat_samples.shape[0])
    thin_factor = max(1, n_samples // 500)
    flat_samples = flat_samples[::thin_factor]

    # --- Indices & parameter names ---
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model)
    param_indices = {name: model_labels.index(name) for name in model_labels}

    # --- Pack obs/errs/pivots once (MAIN sample) ---
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

    # --- y-data for MAIN: log10 L_2500 ---
    if debias:
        #dm_interp = make_dm_function(d["apparent_mag_2500"].values, d['z'].values, dms)
        pts = np.column_stack([d['z'], d['apparent_mag_2500']])
        actual_M2500 = (d['apparent_mag_2500'] - dm_interp(pts)) - cosmo.distmod(d['z']).value
    else:
        actual_M2500 = d['apparent_mag_2500'] - cosmo.distmod(d['z']).value
    actual_logL2500 = convert_M2500_to_logL2500(actual_M2500)
    y_log_meas_err = 0.4 * np.asarray(d['apparent_mag_2500_err'].fillna(0.0))

    # --- Reference x (built at POSTERIOR-MEDIAN params) ---
    med_arr = agn_model_pack_params(med_params)
    M0_med = med_arr[agn_model_pidx["M0_agn"]]
    x_log_ref = M_model_agn(med_arr, agn_obs_arr, agn_pivot_arr) - M0_med
    x_ref = 10.0 ** x_log_ref

    # x errors for MAIN at median params
    pred_M_err_med = M_model_agn_err(med_arr, agn_obs_arr, agn_err_arr, agn_pivot_arr)
    x_lower = 10.0 ** (x_log_ref - pred_M_err_med)
    x_upper = 10.0 ** (x_log_ref + pred_M_err_med)
    xerr_asym = np.vstack((x_ref - np.maximum(x_lower, 1e-300),
                           np.maximum(x_upper, x_ref) - x_ref))

    # Compute calibrators x,y range for plotting band
    if df_calibrators is not None and len(df_calibrators) > 0:
        ds = df_calibrators.copy()

        M2500_show = ds['apparent_mag_2500'].values - cosmo.distmod(ds['z'].values).value
        # y for SHOW: actual_logL2500_show (use the same dm_interp built for MAIN)
        # if debias:
        #     pts_show = np.column_stack([ds['z'].values, ds['apparent_mag_2500'].values])
        #     M2500_show = (ds['apparent_mag_2500'].values - dm_interp(pts_show)) - cosmo.distmod(ds['z'].values).value
        actual_logL2500_show = convert_M2500_to_logL2500(M2500_show)
        y_log_meas_err_show = 0.4 * np.asarray(ds['apparent_mag_2500_err'].fillna(0.0), dtype=float)
        yerr_linear_show = (10.0**actual_logL2500_show) * np.log(10.0) * y_log_meas_err_show

        # x for SHOW at median params (using ONLY df_calibrators fields)
        obs_show, err_show, piv_show = agn_model_pack_obs(ds)
        x_log_ref_show = M_model_agn(med_arr, obs_show, piv_show) - M0_med
        x_show = 10.0 ** x_log_ref_show

        pred_M_err_show = M_model_agn_err(med_arr, obs_show, err_show, piv_show)
        x_log_lower_show = np.min(np.ravel(x_log_ref_show - pred_M_err_show))
        x_log_upper_show = np.max(np.ravel(x_log_ref_show + pred_M_err_show))
        x_lower_show = 10.0 ** (x_log_ref_show - pred_M_err_show)
        x_upper_show = 10.0 ** (x_log_ref_show + pred_M_err_show)
    else:
        x_log_lower_show = 0
        x_log_upper_show = 0
        x_log_ref_show = x_log_ref
        pred_M_err_show = 0

    # --- Grid and band (unchanged) ---
    xcm = np.mean(x_log_ref)
    var_x = np.var(x_log_ref, ddof=1) if np.isfinite(np.var(x_log_ref, ddof=1)) else np.var(x_log_ref) + 1e-8
    # x_min_err = np.min([np.min(x_log_ref - pred_M_err_med), np.min(x_log_lower_show)])
    # x_max_err = np.max([np.max(x_log_ref + pred_M_err_med), np.max(x_log_upper_show)])
    x_min_err = np.min([np.min(x_log_ref), np.min(x_log_lower_show)])
    x_max_err = np.max([np.max(x_log_ref), np.max(x_log_upper_show)])
    print(f"x_log_ref range with errors: {x_min_err:.3f} to {x_max_err:.3f}")
    x_lo = x_min_err - 1.8
    x_hi = x_max_err + 3
    x_log_grid = np.linspace(x_lo, x_hi, 250)
    x_grid = 10.0 ** x_log_grid

    ylog_grid_by_sample = []
    for s in flat_samples:
        sample_params = {k: s[param_indices[k]] for k in model_labels}
        s_arr = agn_model_pack_params(sample_params)
        M_i = M_model_agn(s_arr, agn_obs_arr, agn_pivot_arr)
        Mc = np.mean(M_i)
        cov_Mx = np.mean((x_log_ref - xcm) * (M_i - Mc))
        k_s = cov_Mx / var_x
        c_s = Mc - k_s * xcm
        M_grid_s = c_s + k_s * x_log_grid
        ylog_grid_by_sample.append(convert_M2500_to_logL2500(M_grid_s))
    ylog_grid_by_sample = np.asarray(ylog_grid_by_sample)
    ylog_med  = np.median(ylog_grid_by_sample, axis=0)
    ylog_low  = np.percentile(ylog_grid_by_sample, 16, axis=0)
    ylog_high = np.percentile(ylog_grid_by_sample, 84, axis=0)

    # For residuals vs median (MAIN)
    f_med  = interp1d(x_log_grid, ylog_med,  bounds_error=False, fill_value='extrapolate')
    model_logL_at_data = f_med(x_log_ref)
    residuals = actual_logL2500 - model_logL_at_data

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

    # --- Baseline data (MAIN) ---
    yerr_linear = 10**actual_logL2500 * np.log(10) * y_log_meas_err
    mask_in  = d["z"].between(0.44, 3.16)
    mask_out = ~mask_in

    # inside redshift range: filled markers
    ax.errorbar(
        x_ref[mask_in], 10**actual_logL2500[mask_in], xerr=xerr_asym[:, mask_in], yerr=yerr_linear[mask_in],
        fmt='o', linestyle='none', markersize=4, mfc=(0,0,0,0.4), mec="none",
        #markeredgewidth=0,
        ecolor=(0.2, 0.2, 0.2, 0.1), elinewidth=0.8, capsize=2, capthick=0.8,
        zorder=1, label="AGN"
    )
    # outside redshift range: open markers
    ax.errorbar(
        x_ref[mask_out], 10**actual_logL2500[mask_out], xerr=xerr_asym[:, mask_out], yerr=yerr_linear[mask_out],
        fmt='o', linestyle='none', markersize=3, mfc='none', mec=(0,0,0,0.4),
        ecolor=(0.2, 0.2, 0.2, 0.1), elinewidth=0.8, capsize=2, capthick=0.8,
        zorder=1
    )

    # --- 68% / 95% KDE contours (outlines only) ---
    try:
        finite  = np.isfinite(x_log_ref) & np.isfinite(actual_logL2500)
        in_use  = finite & mask_in.values
        xlog    = x_log_ref[in_use]
        ylog    = actual_logL2500[in_use]

        if xlog.size > 50:
            kde = gaussian_kde(np.vstack([xlog, ylog]), bw_method='scott')

            xq = np.quantile(xlog, [0.01, 0.99]); rx = xq[1] - xq[0]
            yq = np.quantile(ylog, [0.01, 0.99]); ry = yq[1] - yq[0]
            Xg, Yg = np.meshgrid(
                np.linspace(xq[0] - 0.10*rx, xq[1] + 0.10*rx, 220),
                np.linspace(yq[0] - 0.10*ry, yq[1] + 0.10*ry, 220),
            )
            Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

            # Ascending levels: [95%, 68%]
            levels = _kde_conf_levels(Z, conf=(0.954, 0.683))

            CS = ax.contour(10.0**Xg, 10.0**Yg, Z,
                            levels=levels,
                            colors='darkgray',
                            alpha=1.0,
                            linestyles=('solid', 'solid'),   # 95% dashed, 68% solid
                            linewidths=(1.6, 2.0),
                            zorder=4)

            from matplotlib.lines import Line2D
            _extra_contour_handles = [
                Line2D([0],[0], color='k', lw=1.2, ls='--', label='95% contour'),
                Line2D([0],[0], color='k', lw=1.8, ls='-',  label='68% contour'),
            ]
        else:
            _extra_contour_handles = []
    except Exception as e:
        print(f"[KDE contours] skipped: {e}")
        _extra_contour_handles = []



    # --- Model ribbon + median ---
    ax.fill_between(x_grid, 10**ylog_low, 10**ylog_high, color=color, alpha=0.5, zorder=9)
    ax.plot(x_grid, 10**ylog_med, color=color, lw=2.0, zorder=10, label='best-fit model')

    # --- Suberlak+2021 comparison (unchanged) ---
    C = 0.035 + 0.118; C_err = np.sqrt(0.007**2 + 0.003**2)
    ylog_anchor = np.interp(xcm, x_log_grid, ylog_med)
    L_anchor = 10.0**ylog_anchor
    x_anchor = 10.0**xcm
    L_scale = L_anchor * (x_anchor ** (2.5 * C))
    y_central = L_scale * (x_grid ** (-2.5 * C))
    rng = np.random.default_rng(42)
    C_samps = rng.normal(loc=C, scale=C_err, size=100)
    curves = L_scale * (x_grid[None, :] ** (-2.5 * C_samps[:, None]))
    sub_lo, sub_hi = np.percentile(curves, [16, 84], axis=0)
    ax.plot(x_grid, y_central, color='c', lw=2.0, zorder=10,
            label='Suberlak+2021 relation', linestyle='--')
    ax.fill_between(x_grid, sub_lo, sub_hi, color='c', alpha=0.3, zorder=8)

    # ========= HIGHLIGHT: compute EVERYTHING from df_calibrators =========
    if df_calibrators is not None and len(df_calibrators) > 0:
        ds = df_calibrators.copy()

        M2500_show = ds['apparent_mag_2500'].values - cosmo.distmod(ds['z'].values).value
        # y for SHOW: actual_logL2500_show (use the same dm_interp built for MAIN)
        # if debias:
        #     pts_show = np.column_stack([ds['z'].values, ds['apparent_mag_2500'].values])
        #     M2500_show = (ds['apparent_mag_2500'].values - dm_interp(pts_show)) - cosmo.distmod(ds['z'].values).value
        actual_logL2500_show = convert_M2500_to_logL2500(M2500_show)
        y_log_meas_err_show = 0.4 * np.asarray(ds['apparent_mag_2500_err'].fillna(0.0), dtype=float)
        yerr_linear_show = (10.0**actual_logL2500_show) * np.log(10.0) * y_log_meas_err_show

        # x for SHOW at median params (using ONLY df_calibrators fields)
        obs_show, err_show, piv_show = agn_model_pack_obs(ds)
        x_log_ref_show = M_model_agn(med_arr, obs_show, piv_show) - M0_med
        x_show = 10.0 ** x_log_ref_show

        pred_M_err_show = M_model_agn_err(med_arr, obs_show, err_show, piv_show)
        x_lower_show = 10.0 ** (x_log_ref_show - pred_M_err_show)
        x_upper_show = 10.0 ** (x_log_ref_show + pred_M_err_show)

        # Finite mask for safety
        m_show = (
            np.isfinite(x_show) & np.isfinite(x_lower_show) & np.isfinite(x_upper_show) &
            np.isfinite(actual_logL2500_show) & np.isfinite(yerr_linear_show)
        )

        # Distinct color map per object
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(ds))]

        # Plot each SHOW point with its own error bars and legend label = object_id
        # Smaller than the main AGN sample
        
        for idx in np.where(m_show)[0]:
            xi = float(x_show[idx])
            yi = float(10.0**actual_logL2500_show[idx])
            xerr_lo = max(xi - float(x_lower_show[idx]), 0.0)
            xerr_hi = max(float(x_upper_show[idx]) - xi, 0.0)
            yerr_i  = float(yerr_linear_show[idx])

            # Error bars (asymmetric in x)
            ax.errorbar(
                xi, yi,
                xerr=np.array([[xerr_lo], [xerr_hi]]),
                yerr=yerr_i,
                fmt='none',
                ecolor='k',
                elinewidth=1.2,
                alpha=0.95,
                zorder=29,
            )

            ax.scatter(
                xi, yi,
                s=140, facecolors='darkorange', alpha=0.9,
                edgecolors='k', linewidths=0.9, zorder=31,
                marker='*', label='Calibrator' if idx == np.where(m_show)[0][0] else None
            )

    # --- Axes & labels ---
    ax.set_ylabel(r'$L_{2500\,\mathrm{\AA}}$ (erg s$^{-1}$)')
    ax.set_xscale('log'); ax.set_yscale('log')
    if df_calibrators is not None and len(df_calibrators) > 0:
        # ax.set_xlim((2e-9, 6e13))
        # ax.set_ylim((5e39, 2e48))
        ax.set_xlim((np.min(x_grid), np.max(x_grid)))
        ax.set_ylim((np.min(10**ylog_med), np.max(10**ylog_med)))
    else:
        # ax.set_xlim((7e-8, 9e5))
        # ax.set_ylim((3e42, 2e47))
        ax.set_xlim((np.min(x_grid), np.max(x_grid)))
        ax.set_ylim((np.min(10**ylog_med), np.max(10**ylog_med)))

    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))

    # x label (from MAIN pivots; just a label)
    obs_arr, err_arr, pivots_arr = agn_model_pack_obs(df_agn)
    log_sigma_UV_pivot  = pivots_arr[agn_model_oidx["log_sigma_UV"]]
    log_tau_UV_RF_pivot = pivots_arr[agn_model_oidx["log_tau_UV_RF"]]
    sigma_UV_pivot  = 10.0 ** log_sigma_UV_pivot
    tau_UV_RF_pivot = 10.0 ** log_tau_UV_RF_pivot
    alpha_agn_L = med_params['alpha_agn'] * (-1/2.5)
    beta_agn_L  = med_params['beta_agn']  * (-1/2.5)
    xlabel = rf"$({{\sigma}}_\mathrm{{UV}} \, / \, {sigma_UV_pivot:.1f}\,\mathrm{{mag}})^{{{alpha_agn_L:.2f}}} \, ({{\tau}}_\mathrm{{UV,RF}} \, / \, {tau_UV_RF_pivot:.0f}\,\mathrm{{days}})^{{{beta_agn_L:.2f}}}$"
    ax.set_xlabel(xlabel)
    ax.legend(loc='upper right')

    # --- Residuals panel (MAIN) ---
    sigma_meas = np.asarray(y_log_meas_err, dtype=float)
    slope_grid = np.gradient(ylog_med, x_log_grid)
    f_slope = interp1d(x_log_grid, slope_grid, bounds_error=False, fill_value='extrapolate')
    slope_at_data = f_slope(x_log_ref)
    sigma_x = np.asarray(pred_M_err_med, dtype=float)
    sigma_xy = np.abs(slope_at_data) * np.abs(sigma_x)
    sigma_mu_log = 0.0
    sigma_chi = np.sqrt(sigma_meas**2 + sigma_xy**2 + sigma_mu_log**2)
    good = np.isfinite(residuals) & np.isfinite(sigma_chi) & (sigma_chi > 0)

    if show_residuals and ax_res is not None:
        sc = ax_res.scatter(x_ref[good], residuals[good], s=5, alpha=0.4, c=np.zeros(np.sum(good)),
                            cmap='viridis', lw=0.5, zorder=5)
        plt.colorbar(sc, ax=ax_res, orientation='vertical').set_label('alpha_lambda (main)')

        ax_res.axhline(0, color='m', linestyle='--', zorder=3)
        ax_res.set_ylabel('Residuals (log)')
        ax_res.set_xlabel(xlabel)
        ax_res.set_xscale('log')
        ax_res.set_ylim(-2.2, 2.2)


    # Save & return
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(os.path.join(plot_path, "pdf"), exist_ok=True)
    out_png = "predicted_L2500_vs_fullcorr_band_debiased.png" if debias else "predicted_L2500_vs_fullcorr_band.png"
    out_pdf = os.path.splitext(out_png)[0] + ".pdf"
    plt.savefig(os.path.join(plot_path, out_png), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(plot_path, "pdf", out_pdf), dpi=600, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # Return MAIN residuals; show residuals can be computed externally if needed
    return residuals, sigma_chi

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
    os.makedirs(f"plots/hubble/{prefix}/completeness", exist_ok=True)
    plt.savefig(f"plots/hubble/{prefix}/completeness/dmi_vs_mag.png", dpi=300)
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
    plt.savefig(f"plots/hubble/{prefix}/completeness/dmi_vs_redshift.png", dpi=300)
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
    os.makedirs(f"plots/hubble/{prefix}/completeness", exist_ok=True)
    plt.savefig(f"plots/hubble/{prefix}/completeness/dmi_vs_mag.png", dpi=300)
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
    plt.savefig(f"plots/hubble/{prefix}/completeness/dmi_vs_redshift.png", dpi=300)
    plt.close()
    return dmi


from scipy.special import logsumexp
from hubble_likelihood import log_likelihood
from matplotlib.colors import SymLogNorm
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
    plt.figure(figsize=(8,5.2))
    plt.scatter(z, Z, s=12, alpha=0.55)
    plt.xlabel("Redshift (z)")
    plt.ylabel("integral (completeness)  Z = Φ(...) or ∫N×C")
    plt.title(f"Completeness integrals vs z {title_suffix}")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    os.makedirs(f"plots/hubble/{prefix}/completeness", exist_ok=True)
    plt.savefig(f"plots/hubble/{prefix}/completeness/completeness_integrals_vs_z.png", dpi=200)
    plt.close()

def plot_dmi_vs_z(z, dmi, outdir, title_suffix=""):
    plt.figure(figsize=(8,5.2))
    # scatter shows multiplicity; line shows structure when sorted
    order = np.argsort(z)
    plt.plot(z[order], dmi[order], lw=1.4, alpha=0.9)
    plt.xlabel("Redshift (z)")
    plt.ylabel("dmi (mag)  = E[m|det] - m_obs")
    plt.title(f"Interpolated dmi vs z {title_suffix}")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    os.makedirs(f"plots/hubble/{prefix}/completeness", exist_ok=True)
    plt.savefig(f"plots/hubble/{prefix}/completeness/dmi_vs_z.png", dpi=200)
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
    os.makedirs(f"plots/hubble/{prefix}/completeness", exist_ok=True)
    plt.savefig(f"plots/hubble/{prefix}/completeness/completeness_map_with_m50.png", dpi=200)
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


def plot_residuals_vs_alphaOX(
    df_agn,
    residuals,
    residuals_err,
    show=False,
    plot_path=f"plots/hubble/{prefix}/appendix/",
    nbins=6,
    binning="uniform",     # "quantile", "uniform", or pass explicit edges via nbins=array_like
    min_per_bin=4           # hide bins with too few points
):
    """
    Plot residuals vs delta_alphaOX, colored by redshift, with binned means.

    Binning:
      - binning="quantile": edges chosen by quantiles (equal counts)
      - binning="uniform": edges uniformly spaced in delta_alphaOX
      - nbins can be an array-like of explicit edges to override both behaviors
    The binned mean is inverse-variance weighted by residuals_err.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # --- Extract and sanitize inputs ---
    x = np.asarray(df_agn["delta_alphaOX"])
    xerr = np.asarray(df_agn.get("delta_alphaOX_err", np.full_like(x, np.nan)))
    z = np.asarray(df_agn["z"])
    y = np.asarray(residuals)
    yerr = np.asarray(residuals_err)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
    if np.isfinite(xerr).any():
        m &= np.isfinite(xerr) | np.isnan(xerr)
    x, xerr, y, yerr, z = x[m], xerr[m], y[m], yerr[m], z[m]

    # --- Figure/axes ---
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2))

    vmin, vmax = np.nanmin(z), np.nanmax(z)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.get_cmap('viridis')
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    # --- Masks for filled vs open ---
    mask_in = (z > 0.44) & (z < 3.16)   # filled
    mask_out = ~mask_in                 # open (hollow)

    # --- Plot each point with its own error bar ---
    n_pts = len(x)
    for i in range(n_pts):
        zi = z[i]
        ci = cmap(norm(zi))

        # x-error for this point (None if not finite)
        xi_err = xerr[i] if np.isfinite(xerr[i]) else None

        # Filled vs hollow styling
        if mask_in[i]:
            mfc = ci
            mec = 'none'
        else:
            mfc = 'none'
            mec = ci

        label = "AGN" if i == n_pts - 1 else None  # only last point gets the legend label

        ax.errorbar(
            x[i], y[i],
            xerr=xi_err,
            yerr=yerr[i],
            fmt='o',
            markersize=6,
            mfc=mfc,
            mec=mec,
            mew=0.9,
            ecolor=(0.5, 0.5, 0.5, 0.7), elinewidth=0.8, capsize=2, capthick=0.8,
            zorder=2,
            label=label,
        )

    # Zero reference
    ax.axhline(0.0, color='magenta', linewidth=2, zorder=0)

    # --- Binning setup ---
    if np.ndim(nbins) > 0:  # explicit edges provided
        edges = np.asarray(nbins, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("Explicit 'nbins' must be a 1D array of bin edges with size >= 2.")
    else:
        if binning == "quantile":
            qs = np.linspace(0, 1, nbins + 1)
            edges = np.quantile(x, qs)
            edges = np.unique(edges)
            if edges.size < 2:
                raise ValueError("Not enough unique quantile edges for binning.")
        elif binning == "uniform":
            edges = np.linspace(x.min(), x.max(), nbins + 1)
        else:
            raise ValueError("binning must be 'quantile', 'uniform', or provide explicit edges via nbins.")

    # --- Compute binned stats using _weighted_bin_stats ---
    bx, by, by_sem, bN = _weighted_bin_stats(
        x, y, yerr,
        bins=edges,
        min_count=min_per_bin,
        center='mid',  # display at bin midpoints; change to 'weighted' if preferred
    )

    if len(bx):
        ax.errorbar(
            bx, by, yerr=by_sem,
            fmt='o', ms=6, lw=2, color='red', mfc='red', mew=1.2,
            zorder=3, label="Binned mean"
        )

    # --- Labels, colorbar, cosmetics ---
    ax.set_xlabel(r'$\Delta\, \alpha_{\mathrm{OX}}$')
    ax.set_ylabel('Residuals (mag)')

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r'$z$')

    # Legend with frame; last AGN point + binned mean will be picked up by labels
    ax.legend(
        loc='lower right',
        frameon=True,
        framealpha=0.8,
    )

    plt.tight_layout()
    os.makedirs(plot_path, exist_ok=True)
    out_pdf = os.path.join(plot_path, "pdf/alphaOx_int_residuals.pdf")
    out_png = os.path.join(plot_path, "alphaOx_int_residuals.png")
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=220)

    plt.ylim(-4.6, 3.9)

    if show:
        plt.show()
    plt.close()

def plot_Mi_relation(df_agn):
    cosmo   = FlatLambdaCDM(H0=70, Om0=0.3)

    DL = cosmo.luminosity_distance(df_agn['z'].values).to(u.parsec).value
    M_i_my = df_agn['apparent_mag_2500'].values - 5.0 * (np.log10(DL) - 1)
    M_i_Wu_z2 = 91 - 2.5 * df_agn['log_lbol']

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(M_i_my, M_i_Wu_z2, c=df_agn['z'], cmap='viridis', alpha=0.6, s=10)
    plt.plot([min(M_i_my), max(M_i_my)], [min(M_i_my), max(M_i_my)], color='red', linestyle='--', label='y=x')
    plt.xlabel('M_i_my')
    plt.ylabel('M_i_Wu_z2')
    plt.title('M_i_my vs M_i_Wu_z2')
    plt.colorbar(scatter, label='Redshift (z)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(f"plots/hubble/{prefix}/diagnostics", exist_ok=True)
    plt.savefig(f"plots/hubble/{prefix}/diagnostics/Mi_relation_comparison.png", dpi=200)
    plt.close()


def plot_completeness_diagnostics(dmi_max_w, z, integrals_max_w):

    # Plot dmi_interp vs z for the highest-weight sample
    dmi_interp = interp1d(z, dmi_max_w, kind='nearest', bounds_error=False, fill_value='extrapolate')
    
    # Plot dmi_interp vs z for the highest-weight sample
    z_plot = np.linspace(0, 4, 200)
    plt.figure(figsize=(8, 5))
    plt.plot(z_plot, dmi_interp(z_plot), label="dmi_interp(z)")
    plt.xlabel("Redshift (z)")
    plt.ylabel("dmi (mag)")
    plt.title("Interpolated dmi vs z — highest posterior weight sample")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(f"plots/hubble/{prefix}/completeness", exist_ok=True)
    plt.savefig(f"plots/hubble/{prefix}/completeness/dmi_interp_vs_z_highest_weight.png", dpi=150)
    plt.close()

    # Plot log(integrals) vs redshift for highest-weight sample
    plt.figure(figsize=(8, 5))
    plt.scatter(z, integrals_max_w, s=16, alpha=0.3)
    plt.xlabel("Redshift (z)")
    plt.ylabel("integral  (completeness)")
    plt.title("Completeness integrals vs z — highest posterior weight sample")
    plt.grid(True)
    plt.tight_layout()
    # Optional: save to disk
    plt.savefig(f"plots/hubble/{prefix}/completeness/integrals_vs_z_highest_weight.png", dpi=150)
    #plt.show()
    plt.close()