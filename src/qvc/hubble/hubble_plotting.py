import numpy as np
import os
import math
import re
import warnings
from ast import literal_eval

import corner
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.cosmology import FlatwCDM, FlatwpwaCDM, FlatLambdaCDM, Flatw0waCDM
from astropy.cosmology.realizations import Planck18
from astropy import units as u
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import minimize_scalar
from scipy.stats import gaussian_kde, kurtosis, norm, normaltest, probplot, skew
from tqdm import tqdm

from qvc.hubble.hubble_model import (M_model_agn, M_model_agn_err, get_model_params, agn_model_pack_params,
    agn_model_pack_obs, agn_model_oidx, agn_model_pidx, agn_model_req_obs, agn_model_req_errs,
    evaluate_log_f, resolve_model_option_flags, get_agn_model_spec, AGN_ALPHA_LAMBDA_PARAM, AGN_ALPHA_LAMBDA_ERR)
from qvc.hubble.hubble_likelihood import sigma_lens_from_dc, sigma_mu_from_z_err
from qvc.hubble.hubble_utils import (
    convert_M2500_to_logL2500,
    cosmo_model_label_latex,
    format_result_errors,
    reduced_chi_squared,
    sym_percentile,
)
from qvc.hubble.hubble_completeness_refactored import (
    apparent_mag_to_logL2500,
    build_smooth_trend_1d,
    evaluate_dm_interp,
    fit_fhost_2500_l2500_model,
    predict_fhost_2500_from_logL2500,
)
from dynesty.utils import resample_equal
from dynesty import plotting as dyplot

warnings.filterwarnings(
    "ignore",
    message=r"This figure includes Axes that are not compatible with tight_layout.*",
    category=UserWarning,
)

_FULL_RESIDUAL_YLIM = (-0.5, 0.5)


_SDSS_FILTER_EDGES_OBS = {
    "u": (3055.11, 4030.64),
    "g": (3797.64, 5553.04),
    "r": (5418.23, 6994.42),
    "i": (6692.41, 8400.32),
    "z": (7964.70, 10873.33),
}

_BLR_LINE_MODELS = {
    "C IV": {"lambda_rest": 1549.0, "mu0": 1.05, "slope": 0.50, "sigma": 0.28},
    "Mg II": {"lambda_rest": 2798.0, "mu0": 1.35, "slope": 0.50, "sigma": 0.25},
    "Hβ": {"lambda_rest": 4861.0, "mu0": 1.55, "slope": 0.50, "sigma": 0.23},
    "Hα": {"lambda_rest": 6563.0, "mu0": 1.72, "slope": 0.50, "sigma": 0.22},
}

_BLR_LINE_LUMINOSITY_SPECS = {
    "C IV": {
        "value_col": "log_lambda_Llambda_1350_agn",
        "err_col": "log_lambda_Llambda_1350_agn_err",
        "axis_label": r"$\log L_{1350}$",
    },
    "Mg II": {
        "value_col": "log_lambda_Llambda_3000_agn",
        "err_col": "log_lambda_Llambda_3000_agn_err",
        "axis_label": r"$\log L_{3000}$",
    },
    "Hβ": {
        "value_col": "log_lambda_Llambda_5100_agn",
        "err_col": "log_lambda_Llambda_5100_agn_err",
        "axis_label": r"$\log L_{5100}$",
    },
    "Hα": {
        "value_col": "log_lambda_Llambda_5100_agn",
        "err_col": "log_lambda_Llambda_5100_agn_err",
        "axis_label": r"$\log L_{5100}$",
    },
}

_BAND_COLORS = {
    "u": "tab:blue",
    "g": "tab:green",
    "r": "tab:red",
    "i": "tab:orange",
    "z": "tab:purple",
}

_BLR_LAG_KL_MIN = 0.05


def _pdf_path(path):
    """Normalize any requested output path to a PDF path."""
    root, _ = os.path.splitext(path)
    return f"{root}.pdf"


def _save_figure(fig, path, *, dpi=300, bbox_inches="tight", show=False):
    """Save a Matplotlib figure as PDF, then optionally show and close it."""
    pdf_path = _pdf_path(path)
    directory = os.path.dirname(pdf_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(pdf_path, dpi=dpi, bbox_inches=bbox_inches)
    if show:
        plt.show()
    plt.close(fig)
    return pdf_path


def _nanmedian_stacked(rows):
    """Return row-wise nanmedian for a sequence of equal-length arrays."""

    if not rows:
        return None
    stacked = np.vstack(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(stacked, axis=0)


def _derive_log_sigma_bc(df_agn):
    """Infer a scalar log sigma_BC from the saved light-curve summary columns."""

    if "log_sigma_bc" in df_agn.columns:
        return pd.to_numeric(df_agn["log_sigma_bc"], errors="coerce").to_numpy(dtype=float)

    if {"log_sigma_uv", "log_amp_delta_bc"}.issubset(df_agn.columns):
        log_sigma_uv = pd.to_numeric(df_agn["log_sigma_uv"], errors="coerce").to_numpy(dtype=float)
        log_amp_delta_bc = pd.to_numeric(df_agn["log_amp_delta_bc"], errors="coerce").to_numpy(dtype=float)
        return log_sigma_uv + log_amp_delta_bc

    rows = []
    for band in ("u", "g", "r", "i", "z"):
        amp_col = f"amp_bc_{band}"
        weight_col = f"bc_weight_{band}"
        if amp_col not in df_agn.columns or weight_col not in df_agn.columns:
            continue
        amp_bc = pd.to_numeric(df_agn[amp_col], errors="coerce").to_numpy(dtype=float)
        bc_weight = pd.to_numeric(df_agn[weight_col], errors="coerce").to_numpy(dtype=float)
        log_sigma_bc = np.full(len(df_agn), np.nan, dtype=float)
        mask = np.isfinite(amp_bc) & np.isfinite(bc_weight) & (amp_bc > 0.0) & (bc_weight > 0.0)
        log_sigma_bc[mask] = np.log10(amp_bc[mask]) - np.log10(bc_weight[mask])
        rows.append(log_sigma_bc)
    return _nanmedian_stacked(rows)


def _derive_log_lag_bc_rf(df_agn):
    """Infer a scalar rest-frame BC lag from saved per-band BC lag columns."""

    rows_rf = []
    for band in ("u", "g", "r", "i", "z"):
        col = f"log_lag_bc_{band}_RF"
        if col in df_agn.columns:
            rows_rf.append(pd.to_numeric(df_agn[col], errors="coerce").to_numpy(dtype=float))
    aggregated_rf = _nanmedian_stacked(rows_rf)
    if aggregated_rf is not None:
        return aggregated_rf

    if "z" not in df_agn.columns:
        return None

    z = pd.to_numeric(df_agn["z"], errors="coerce").to_numpy(dtype=float)
    rows_obs = []
    for band in ("u", "g", "r", "i", "z"):
        col = f"lag_bc_{band}"
        if col not in df_agn.columns:
            continue
        lag_bc = pd.to_numeric(df_agn[col], errors="coerce").to_numpy(dtype=float)
        log_lag_rf = np.full(len(df_agn), np.nan, dtype=float)
        mask = np.isfinite(lag_bc) & (lag_bc > 0.0) & np.isfinite(z) & (z > -1.0)
        log_lag_rf[mask] = np.log10(lag_bc[mask]) - np.log10(1.0 + z[mask])
        rows_obs.append(log_lag_rf)
    return _nanmedian_stacked(rows_obs)


def _get_cosmo_from_params(model_name, params_dict, zp):
    if model_name == "FlatwCDM":
        return FlatwCDM(H0=params_dict["H0"], Om0=params_dict["Om0"], w0=params_dict["w0"])
    if model_name == "Flatw0waCDM":
        return Flatw0waCDM(
            H0=params_dict["H0"],
            Om0=params_dict["Om0"],
            w0=params_dict["w0"],
            wa=params_dict["wa"],
        )
    if model_name == "FlatLambdaCDM":
        return FlatLambdaCDM(H0=params_dict["H0"], Om0=params_dict["Om0"])
    if model_name == "FlatwpwaCDM":
        return FlatwpwaCDM(
            H0=params_dict["H0"],
            Om0=params_dict["Om0"],
            wp=params_dict["wp"],
            wa=params_dict["wa"],
            zp=zp,
        )
    raise ValueError(f"Invalid cosmology model: {model_name}")


def _resolve_debias_values(
    df_agn,
    *,
    dm_interp=None,
    dmi_values=None,
):
    """Use direct per-object dmi where available and fall back to dm_interp."""
    dmi = None
    if dmi_values is not None:
        dmi = np.asarray(dmi_values, dtype=float)
        if dmi.shape != (len(df_agn),):
            raise ValueError(
                f"dmi_values has shape {dmi.shape}, but expected {(len(df_agn),)}."
            )
    if dm_interp is None:
        if dmi is None:
            raise ValueError("Need either dm_interp or dmi_values for debias=True.")
        return dmi

    dmi_interp = evaluate_dm_interp(
        dm_interp,
        df_agn["z"].values,
        df_agn["apparent_mag_2500"].values,
        f_host_2500=df_agn.get("f_host_2500"),
        alpha_lambda=df_agn.get("alpha_lambda"),
    )
    if dmi is None:
        return dmi_interp
    return np.where(np.isfinite(dmi), dmi, dmi_interp)


def _coerce_dropped_bands(value):
    if isinstance(value, (list, tuple, set)):
        return set(value)
    if isinstance(value, str):
        try:
            parsed = literal_eval(value)
            if isinstance(parsed, (list, tuple, set)):
                return set(parsed)
        except (SyntaxError, ValueError):
            pass
        return {value}
    return set()


def _soft_band_overlap_weight(line_lambda_rest, z, band, edge_softness=120.0):
    if band not in _SDSS_FILTER_EDGES_OBS or not np.isfinite(z):
        return 0.0
    lo_obs, hi_obs = _SDSS_FILTER_EDGES_OBS[band]
    lam_obs = line_lambda_rest * (1.0 + z)
    sigmoid_left = 1.0 / (1.0 + np.exp(-(lam_obs - lo_obs) / edge_softness))
    sigmoid_right = 1.0 / (1.0 + np.exp(-(hi_obs - lam_obs) / edge_softness))
    return float(sigmoid_left * sigmoid_right)


def _coerce_numeric_vector(values, n_rows, *, fill_value=np.nan):
    arr = np.full(n_rows, fill_value, dtype=float)
    if values is None:
        return arr
    values_arr = np.asarray(values, dtype=float)
    n_copy = min(n_rows, values_arr.size)
    if n_copy > 0:
        arr[:n_copy] = values_arr[:n_copy]
    return arr


def _build_blr_line_luminosity_maps(df, fallback_log_luminosity, log_luminosity_shift):
    n_rows = len(df)
    fallback_log_luminosity = _coerce_numeric_vector(
        fallback_log_luminosity,
        n_rows,
    )
    log_luminosity_shift = _coerce_numeric_vector(
        log_luminosity_shift,
        n_rows,
        fill_value=0.0,
    )

    luminosity_maps = {}
    for line_name, spec in _BLR_LINE_LUMINOSITY_SPECS.items():
        values = (
            pd.to_numeric(df[spec["value_col"]], errors="coerce").to_numpy(dtype=float)
            if spec["value_col"] in df.columns
            else np.full(n_rows, np.nan, dtype=float)
        )
        errs = (
            pd.to_numeric(df[spec["err_col"]], errors="coerce").to_numpy(dtype=float)
            if spec["err_col"] in df.columns
            else np.full(n_rows, np.nan, dtype=float)
        )
        shifted_values = values.copy()
        valid = np.isfinite(shifted_values) & np.isfinite(log_luminosity_shift)
        shifted_values[valid] = shifted_values[valid] + log_luminosity_shift[valid]
        use_fallback = ~np.isfinite(shifted_values)
        shifted_values[use_fallback] = fallback_log_luminosity[use_fallback]
        luminosity_maps[line_name] = {
            "values": shifted_values,
            "errs": np.where(np.isfinite(errs) & (errs >= 0.0), errs, np.nan),
            "value_col": spec["value_col"],
            "err_col": spec["err_col"],
            "axis_label": spec["axis_label"],
        }

    return fallback_log_luminosity, luminosity_maps


def _blr_line_assignment_longform(
    df,
    logL2500_debiased,
    *,
    log_luminosity_shift=None,
    lag_err_max=0.25,
    null_score=0.05,
):
    rows = []
    continuum_ref_col = "log_sigma_uv"
    logL2500_arr, luminosity_maps = _build_blr_line_luminosity_maps(
        df,
        logL2500_debiased,
        log_luminosity_shift,
    )
    for suffix in ("", "2"):
        component = 1 if suffix == "" else 2
        for band in ("u", "g", "r", "i", "z"):
            amp_col = f"log_amp_delta_blr{suffix}_{band}"
            lag_col = f"log_lag_blr{suffix}_{band}_RF"
            lag_err_col = f"{lag_col}_err"
            if amp_col not in df.columns or lag_col not in df.columns:
                continue

            continuum_col = (
                f"log_sigma_band_{band}"
                if f"log_sigma_band_{band}" in df.columns
                else continuum_ref_col
            )
            log_amp_blr = (
                pd.to_numeric(df[continuum_col], errors="coerce").to_numpy(dtype=float)
                + pd.to_numeric(df[amp_col], errors="coerce").to_numpy(dtype=float)
            )
            log_lag_rf = pd.to_numeric(df[lag_col], errors="coerce").to_numpy(dtype=float)
            lag_err = (
                pd.to_numeric(df[lag_err_col], errors="coerce").to_numpy(dtype=float)
                if lag_err_col in df.columns
                else np.full(len(df), np.nan, dtype=float)
            )
            lag_kl_col = f"log_lag_blr{suffix}_{band}_kl"
            lag_kl = (
                pd.to_numeric(df[lag_kl_col], errors="coerce").to_numpy(dtype=float)
                if lag_kl_col in df.columns
                else np.full(len(df), np.nan, dtype=float)
            )
            z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
            object_ids = df["object_id"].astype(str).to_numpy() if "object_id" in df.columns else np.arange(len(df)).astype(str)
            dropped_sets = (
                df["dropped_bands"].apply(_coerce_dropped_bands).tolist()
                if "dropped_bands" in df.columns
                else [set() for _ in range(len(df))]
            )

            for i in range(len(df)):
                if band in dropped_sets[i]:
                    continue
                if not (
                    np.isfinite(log_amp_blr[i])
                    and np.isfinite(log_lag_rf[i])
                    and np.isfinite(z[i])
                ):
                    continue

                line_scores = {}
                for line_name, cfg in _BLR_LINE_MODELS.items():
                    overlap = _soft_band_overlap_weight(cfg["lambda_rest"], z[i], band)
                    line_scores[line_name] = overlap

                line_score_sum = float(sum(line_scores.values()))
                total_score = null_score + line_score_sum
                probs = {line_name: score / total_score for line_name, score in line_scores.items()}
                p_null = null_score / total_score if np.isfinite(total_score) and total_score > 0.0 else np.nan
                if (not np.isfinite(line_score_sum)) or line_score_sum <= 0.0:
                    assigned_line = "Unassigned"
                    assigned_prob = 0.0
                else:
                    assigned_line = max(probs, key=probs.get)
                    assigned_prob = probs[assigned_line]
                luminosity_spec = luminosity_maps.get(assigned_line)
                if luminosity_spec is None:
                    log_luminosity = logL2500_arr[i] if i < len(logL2500_arr) else np.nan
                    log_luminosity_err = np.nan
                    luminosity_col = "logL2500_debiased"
                    luminosity_err_col = None
                    luminosity_axis_label = r"$\log L_{2500}$"
                else:
                    log_luminosity = luminosity_spec["values"][i]
                    log_luminosity_err = luminosity_spec["errs"][i]
                    luminosity_col = luminosity_spec["value_col"]
                    luminosity_err_col = luminosity_spec["err_col"]
                    luminosity_axis_label = luminosity_spec["axis_label"]
                rows.append(
                    {
                        "object_id": object_ids[i],
                        "band": band,
                        "component": component,
                        "z": z[i],
                        "logL2500_debiased": logL2500_arr[i] if i < len(logL2500_arr) else np.nan,
                        "log_line_luminosity": log_luminosity,
                        "log_line_luminosity_err": log_luminosity_err,
                        "line_luminosity_col": luminosity_col,
                        "line_luminosity_err_col": luminosity_err_col,
                        "line_luminosity_axis_label": luminosity_axis_label,
                        "log_amp_blr": log_amp_blr[i],
                        "log_lag_rf": log_lag_rf[i],
                        "log_lag_rf_err": lag_err[i],
                        "log_lag_kl": lag_kl[i],
                        "well_constrained": bool(np.isfinite(lag_err[i]) and lag_err[i] <= lag_err_max),
                        "assigned_line": assigned_line,
                        "assigned_prob": assigned_prob,
                        "p_null": p_null,
                        **{f"p_{name.replace(' ', '_').replace('β', 'b').replace('α', 'a')}": prob for name, prob in probs.items()},
                    }
                )

    return pd.DataFrame(rows)


def _plot_blr_lag_line_panel(ax, line_df, line_name, *, x_suffix=""):
    if line_df.empty:
        ax.text(
            0.5,
            0.5,
            "No assignments after filters",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        for component, marker in ((1, "o"), (2, "s")):
            comp_df = line_df[line_df["component"] == component]
            if comp_df.empty:
                continue

            x = pd.to_numeric(comp_df["log_line_luminosity"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(comp_df["log_lag_rf"], errors="coerce").to_numpy(dtype=float)
            xerr = np.abs(
                pd.to_numeric(comp_df["log_line_luminosity_err"], errors="coerce").to_numpy(dtype=float)
            )
            yerr = np.abs(
                pd.to_numeric(comp_df["log_lag_rf_err"], errors="coerce").to_numpy(dtype=float)
            )
            keep = np.isfinite(x) & np.isfinite(y)
            if not np.any(keep):
                continue
            xerr = np.where(np.isfinite(xerr[keep]), xerr[keep], 0.0)
            yerr = np.where(np.isfinite(yerr[keep]), yerr[keep], 0.0)
            ax.errorbar(
                x[keep],
                y[keep],
                xerr=xerr,
                yerr=yerr,
                fmt=marker,
                linestyle="none",
                color="black",
                ecolor="black",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=4.0,
                elinewidth=0.8,
                capsize=2.0,
                alpha=0.8,
            )

    x_label = _BLR_LINE_LUMINOSITY_SPECS.get(line_name, {}).get(
        "axis_label",
        r"$\log L_{2500}$",
    )
    if x_suffix:
        x_label = f"{x_label} {x_suffix}"
    ax.set_title(f"{line_name} (N={len(line_df)})")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\log \tau_{\rm BLR,RF}$")


def plot_blr_line_lags_vs_l2500(
    flat_samples,
    df_agn,
    cosmo_model,
    z_pivot_agn,
    dm_interp,
    *,
    plot_path="plots/hubble",
    show=False,
    prob_thresh=0.9,
    lag_err_max=0.25,
    use_alpha_lambda_term=None,
    use_redshift_log_f_term=None,
):
    """Plot BLR lag against line-matched debiased continuum luminosity."""
    if df_agn.empty or dm_interp is None:
        return None
    required = {"z", "apparent_mag_2500"}
    if not required.issubset(df_agn.columns):
        return None

    option_flags = resolve_model_option_flags(
        cosmo_model,
        np.asarray(flat_samples).shape[1],
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    _, model_labels, _ = get_model_params(
        cosmo_model,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
    )
    param_indices = {name: model_labels.index(name) for name in model_labels}
    med_params = {key: np.median(flat_samples[:, idx]) for key, idx in param_indices.items()}
    cosmo = _get_cosmo_from_params(cosmo_model, med_params, z_pivot_agn)

    z = pd.to_numeric(df_agn["z"], errors="coerce").to_numpy(dtype=float)
    m2500 = pd.to_numeric(df_agn["apparent_mag_2500"], errors="coerce").to_numpy(dtype=float)
    dmi = evaluate_dm_interp(
        dm_interp,
        z,
        m2500,
        f_host_2500=df_agn.get("f_host_2500"),
        alpha_lambda=df_agn.get("alpha_lambda"),
    )
    actual_M2500 = (m2500 - dmi) - cosmo.distmod(z).value
    logL2500_debiased = convert_M2500_to_logL2500(actual_M2500)

    assignments = _blr_line_assignment_longform(
        df_agn,
        logL2500_debiased,
        log_luminosity_shift=0.4 * dmi,
        lag_err_max=lag_err_max,
    )
    if assignments.empty:
        return None

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    os.makedirs(diagnostics_path, exist_ok=True)
    assignments.to_csv(
        os.path.join(diagnostics_path, "blr_line_assignment_probabilities.csv"),
        index=False,
    )

    keep = assignments["assigned_prob"] > prob_thresh
    selected = assignments.loc[keep].copy()
    selected.to_csv(
        os.path.join(diagnostics_path, "blr_line_assignment_selected.csv"),
        index=False,
    )

    line_order = ["C IV", "Mg II", "Hβ", "Hα"]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.5), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, line_name in zip(axes, line_order):
        line_df = selected[selected["assigned_line"] == line_name]
        _plot_blr_lag_line_panel(
            ax,
            line_df,
            line_name,
            x_suffix="(debiased)",
        )

    component_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color="k", label="BLR 1", markersize=6),
        Line2D([0], [0], marker="s", linestyle="none", color="k", label="BLR 2", markersize=6),
    ]
    fig.legend(
        handles=component_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        rf"Assigned BLR lags vs line-matched debiased continuum luminosity ($p \geq {prob_thresh:.1f}$)",
        y=1.05,
    )
    fig.tight_layout()
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, "blr_line_lags_vs_l2500_debiased.pdf"),
        dpi=200,
        show=show,
    )


def plot_blr_assignment_probabilities(
    assignments,
    *,
    plot_path="plots/hubble",
    show=False,
    filename="blr_line_assignment_probabilities.pdf",
    title_suffix="",
):
    """Plot assigned-line probability distributions overall and by line."""
    if assignments is None or len(assignments) == 0:
        return None

    df = pd.DataFrame(assignments).copy()
    if "assigned_prob" not in df.columns or "assigned_line" not in df.columns:
        return None

    df["assigned_prob"] = pd.to_numeric(df["assigned_prob"], errors="coerce")
    df = df[np.isfinite(df["assigned_prob"])].copy()
    if df.empty:
        return None

    line_order = ["C IV", "Mg II", "Hβ", "Hα"]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.0), sharex=True, sharey=True)
    axes = axes.ravel()
    panel_specs = [("All", df)] + [(line, df[df["assigned_line"] == line]) for line in line_order]
    bins = np.linspace(0.0, 1.0, 21)

    for ax, (label, sub) in zip(axes, panel_specs):
        if sub.empty:
            ax.text(0.5, 0.5, "No assignments", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.hist(
                sub["assigned_prob"].to_numpy(dtype=float),
                bins=bins,
                color="black",
                alpha=0.85,
                edgecolor="white",
            )
        ax.set_title(f"{label} (N={len(sub)})")
        ax.set_xlabel("assigned probability")
        ax.set_ylabel("Count")
        ax.set_xlim(0.0, 1.0)

    for ax in axes[len(panel_specs):]:
        ax.set_axis_off()

    suffix = f" {title_suffix}".rstrip() if title_suffix else ""
    fig.suptitle(f"BLR assigned-line probabilities{suffix}", y=1.01)
    fig.tight_layout()

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    os.makedirs(diagnostics_path, exist_ok=True)
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_blr_line_lags_vs_l2500_fiducial(
    df_agn,
    *,
    plot_path="plots/hubble",
    show=False,
    prob_thresh=0.6,
    lag_err_max=0.25,
    lag_kl_min=_BLR_LAG_KL_MIN,
    cosmo=None,
    filename="blr_line_lags_vs_l2500_fiducial.pdf",
    assignment_probabilities_filename="blr_line_assignment_probabilities_fiducial.pdf",
):
    """Plot BLR lag against line-matched fit_spectra continuum luminosity."""
    if df_agn.empty:
        return None
    required = {"z", "apparent_mag_2500"}
    if not required.issubset(df_agn.columns):
        return None

    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)

    z = pd.to_numeric(df_agn["z"], errors="coerce").to_numpy(dtype=float)
    m2500 = pd.to_numeric(df_agn["apparent_mag_2500"], errors="coerce").to_numpy(dtype=float)
    actual_M2500 = m2500 - cosmo.distmod(z).value
    logL2500_fid = convert_M2500_to_logL2500(actual_M2500)

    assignments = _blr_line_assignment_longform(
        df_agn,
        logL2500_fid,
        lag_err_max=lag_err_max,
    )
    if assignments.empty:
        return None

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    os.makedirs(diagnostics_path, exist_ok=True)
    assignments.to_csv(
        os.path.join(diagnostics_path, "blr_line_assignment_probabilities_fiducial.csv"),
        index=False,
    )
    plot_blr_assignment_probabilities(
        assignments,
        plot_path=plot_path,
        show=show,
        filename=assignment_probabilities_filename,
        title_suffix="(fiducial cosmology)",
    )

    keep = assignments["assigned_prob"] >= prob_thresh
    keep &= assignments["well_constrained"].to_numpy(dtype=bool)
    keep &= pd.to_numeric(assignments["log_lag_rf"], errors="coerce").to_numpy(dtype=float) > 0.0
    lag_kl = pd.to_numeric(assignments.get("log_lag_kl"), errors="coerce").to_numpy(dtype=float)
    if np.any(np.isfinite(lag_kl)):
        keep &= np.isfinite(lag_kl) & (lag_kl >= lag_kl_min)
    selected = assignments.loc[keep].copy()
    selected.to_csv(
        os.path.join(diagnostics_path, "blr_line_assignment_selected_fiducial.csv"),
        index=False,
    )

    line_order = ["C IV", "Mg II", "Hβ", "Hα"]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.5), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, line_name in zip(axes, line_order):
        line_df = selected[selected["assigned_line"] == line_name]
        _plot_blr_lag_line_panel(
            ax,
            line_df,
            line_name,
            x_suffix="(fit_spectra)",
        )

    component_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color="k", label="BLR 1", markersize=6),
        Line2D([0], [0], marker="s", linestyle="none", color="k", label="BLR 2", markersize=6),
    ]
    fig.legend(
        handles=component_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        rf"Assigned BLR lags vs line-matched fit_spectra continuum luminosity ($p > {prob_thresh:.1f}$)",
        y=1.05,
    )
    fig.tight_layout()
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_l2500_vs_uv_variability_fiducial(
    df_agn,
    *,
    plot_path="plots/hubble",
    show=False,
    cosmo=None,
    thin_step=5,
    filename="l2500_vs_uv_variability_fiducial.pdf",
    dynamic_axes=False,
):
    """Plot fiducial-cosmology L_2500 against log_sigma_uv and log_tau_uv_rf."""
    required = {"z", "apparent_mag_2500", "log_sigma_uv", "log_tau_uv_rf"}
    if not required.issubset(df_agn.columns):
        missing = ", ".join(sorted(required - set(df_agn.columns)))
        raise KeyError(f"Missing required columns for L2500 UV-variability plot: {missing}")

    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)

    z = pd.to_numeric(df_agn["z"], errors="coerce").to_numpy(dtype=float)
    m2500 = pd.to_numeric(df_agn["apparent_mag_2500"], errors="coerce").to_numpy(dtype=float)
    log_sigma = pd.to_numeric(df_agn["log_sigma_uv"], errors="coerce").to_numpy(dtype=float)
    log_tau_rf = pd.to_numeric(df_agn["log_tau_uv_rf"], errors="coerce").to_numpy(dtype=float)

    actual_M2500 = m2500 - cosmo.distmod(z).value
    logL2500_fid = convert_M2500_to_logL2500(actual_M2500)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=True)
    panels = [
        (log_sigma, r"$\log \sigma_{\rm UV}$", "No finite log_sigma_uv values"),
        (log_tau_rf, r"$\log \tau_{\rm UV,RF}$", "No finite log_tau_uv_rf values"),
    ]

    for ax, (y, ylabel, empty_text) in zip(axes, panels):
        mask = np.isfinite(logL2500_fid) & np.isfinite(y) & np.isfinite(z)
        if np.any(mask):
            idx = np.flatnonzero(mask)
            if thin_step is not None and thin_step > 1:
                idx = idx[::thin_step]
            sc = ax.scatter(
                y[idx],
                logL2500_fid[idx],
                c=z[idx],
                cmap="viridis",
                s=10,
                alpha=0.7,
                linewidths=0,
                rasterized=True,
            )
            if dynamic_axes and idx.size > 0:
                x = y[idx]
                yplot = logL2500_fid[idx]
                xpad = 0.05 * max(np.nanmax(x) - np.nanmin(x), 1e-6)
                ypad = 0.05 * max(np.nanmax(yplot) - np.nanmin(yplot), 1e-6)
                ax.set_xlim(np.nanmin(x) - xpad, np.nanmax(x) + xpad)
                ax.set_ylim(np.nanmin(yplot) - ypad, np.nanmax(yplot) + ypad)
        else:
            sc = None
            ax.text(0.5, 0.5, empty_text, ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(r"$\log L_{2500}$ (fiducial cosmology)")

    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.tolist())
        cbar.set_label("Redshift z")

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_l2500_vs_eta_sigma_fiducial(
    df_agn,
    *,
    plot_path="plots/hubble",
    show=False,
    cosmo=None,
    thin_step=5,
    filename="l2500_vs_eta_sigma_fiducial.pdf",
    dynamic_axes=False,
):
    """Plot fiducial-cosmology L_2500 against eta_sigma."""
    required = {"z", "apparent_mag_2500", "eta_sigma"}
    if not required.issubset(df_agn.columns):
        missing = ", ".join(sorted(required - set(df_agn.columns)))
        raise KeyError(f"Missing required columns for L2500 eta_sigma plot: {missing}")

    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)

    z = pd.to_numeric(df_agn["z"], errors="coerce").to_numpy(dtype=float)
    m2500 = pd.to_numeric(df_agn["apparent_mag_2500"], errors="coerce").to_numpy(dtype=float)
    eta_sigma = pd.to_numeric(df_agn["eta_sigma"], errors="coerce").to_numpy(dtype=float)

    actual_M2500 = m2500 - cosmo.distmod(z).value
    logL2500_fid = convert_M2500_to_logL2500(actual_M2500)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.8))
    mask = np.isfinite(logL2500_fid) & np.isfinite(eta_sigma) & np.isfinite(z)
    if np.any(mask):
        idx = np.flatnonzero(mask)
        if thin_step is not None and thin_step > 1:
            idx = idx[::thin_step]
        sc = ax.scatter(
            eta_sigma[idx],
            logL2500_fid[idx],
            c=z[idx],
            cmap="viridis",
            s=10,
            alpha=0.7,
            linewidths=0,
            rasterized=True,
        )
        if dynamic_axes and idx.size > 0:
            x = eta_sigma[idx]
            yplot = logL2500_fid[idx]
            xpad = 0.05 * max(np.nanmax(x) - np.nanmin(x), 1e-6)
            ypad = 0.05 * max(np.nanmax(yplot) - np.nanmin(yplot), 1e-6)
            ax.set_xlim(np.nanmin(x) - xpad, np.nanmax(x) + xpad)
            ax.set_ylim(np.nanmin(yplot) - ypad, np.nanmax(yplot) + ypad)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Redshift z")
    else:
        ax.text(0.5, 0.5, "No finite eta_sigma values", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel(r"$\eta_{\sigma}$")
    ax.set_ylabel(r"$\log L_{2500}$ (fiducial cosmology)")

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_cut_diagnostics(df_before, df_after, bins=30, cut_info="", save_path="plots/hubble/cuts/"):
    """
    Plot a combined cut diagnostic with:
      - top panel: m_2500 vs redshift
      - bottom panel: redshift histogram

    Both panels show the kept and removed populations against the full sample.
    """
    if len(df_before) == len(df_after):
        return

    def _cut_slug(text):
        """Build a stable filename token from cut text without numeric thresholds."""
        if not text:
            return "generic"
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(text))
        stop = {"or", "and", "nan"}
        tokens = [tok for tok in tokens if tok.lower() not in stop]
        if not tokens:
            return "generic"
        return "_".join(tokens)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    before_ids = set(df_before["object_id"].astype(str))
    after_ids = set(df_after["object_id"].astype(str))
    removed_ids = before_ids - after_ids
    df_removed = df_before[df_before["object_id"].astype(str).isin(removed_ids)]

    def _finite(df):
        z = df["z"].to_numpy(dtype=float)
        m = df["apparent_mag_2500"].to_numpy(dtype=float)
        ok = np.isfinite(z) & np.isfinite(m)
        return z[ok], m[ok]

    z_all, m_all = _finite(df_before)
    z_kept, m_kept = _finite(df_after)
    z_removed, m_removed = _finite(df_removed)

    if z_all.size:
        x_min, x_max = np.nanmin(z_all), np.nanmax(z_all)
    else:
        x_min, x_max = 0.0, 1.0

    # Use a stacked layout so the photometry and redshift views stay aligned.
    fig = plt.figure(figsize=(11, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.2, 1.0], hspace=0.12)

    ax1 = fig.add_subplot(gs[0])
    if z_all.size:
        ax1.scatter(z_all, m_all, s=6, alpha=0.18, c="0.4", linewidths=0, label="All", rasterized=True)
    if z_kept.size:
        ax1.scatter(z_kept, m_kept, s=10, alpha=0.8, c="tab:orange", linewidths=0, label="Kept", rasterized=True)
    if z_removed.size:
        ax1.scatter(z_removed, m_removed, s=12, alpha=0.75, c="tab:red", linewidths=0, label="Removed", rasterized=True)
    ax1.set_xlabel("Redshift $z$")
    ax1.set_ylabel(r"$m_{2500}$ (AB)")
    ax1.set_xlim(x_min, x_max)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", frameon=False)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    _, bin_edges = np.histogram(df_before["z"].dropna(), bins=bins)
    ax2.hist(df_before["z"].dropna(), bins=bin_edges, histtype="step", linewidth=1.8, color="0.35", label="All")
    ax2.hist(df_after["z"].dropna(), bins=bin_edges, color="tab:orange", alpha=0.55, edgecolor="none", label="Kept")
    if len(df_removed) > 0:
        ax2.hist(df_removed["z"].dropna(), bins=bin_edges, color="tab:red", alpha=0.45, edgecolor="none", label="Removed")
    ax2.set_xlabel("Redshift $z$")
    ax2.set_ylabel("Count")
    ax2.set_xlim(x_min, x_max)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", frameon=False)

    if cut_info:
        fig.text(0.5, 0.01, f"Cut info: {cut_info}", ha="center", va="bottom", fontsize=11, color="k")

    filename = f"cut_diagnostic_{_cut_slug(cut_info)}.pdf"
    plot_path = os.path.join(os.path.dirname(save_path), filename)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save_figure(fig, plot_path, dpi=150)


def plot_sigma_uv_host_correction(df, plot_path="plots/hubble", show=False, filename="sigma_uv_host_correction_comparison.pdf"):
    """Compare corrected and uncorrected UV variability amplitudes, colored by redshift."""
    required = {"log_sigma_uv", "log_sigma_uv_uncorrected", "z", "frac_host_psf_2500"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise KeyError(f"Missing required columns for sigma_uv host-correction plot: {missing}")

    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
    frac_host_psf = pd.to_numeric(df["frac_host_psf_2500"], errors="coerce").to_numpy(dtype=float)
    delta_log_sigma = (
        pd.to_numeric(df["log_sigma_uv"], errors="coerce").to_numpy(dtype=float)
        - pd.to_numeric(df["log_sigma_uv_uncorrected"], errors="coerce").to_numpy(dtype=float)
    )

    mask_left = np.isfinite(delta_log_sigma) & np.isfinite(z)
    if not np.any(mask_left):
        raise ValueError("No finite rows available for sigma_uv host-correction diagnostics.")

    x_left = z[mask_left]
    y_left = delta_log_sigma[mask_left]
    z_left = z[mask_left]

    mask_right = (
        np.isfinite(delta_log_sigma)
        & np.isfinite(frac_host_psf)
        & np.isfinite(z)
        & (frac_host_psf != -1.0)
        & (frac_host_psf > 0.0)
    )
    log_frac_host_psf = np.log10(frac_host_psf[mask_right]) if np.any(mask_right) else np.array([])
    delta_right = delta_log_sigma[mask_right]
    z_right = z[mask_right]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    sc_left = axes[0].scatter(
        x_left,
        y_left,
        c=z_left,
        cmap="viridis",
        s=10,
        alpha=0.65,
        linewidths=0,
        rasterized=True,
    )
    axes[0].axhline(0.0, color="k", ls="--", lw=1, alpha=0.8)
    axes[0].set_xlabel("Redshift z")
    axes[0].set_ylabel(r"$\Delta \log \sigma_{\rm UV}$")
    axes[0].grid(True, alpha=0.25)

    if np.any(mask_right):
        sc_right = axes[1].scatter(
            log_frac_host_psf,
            delta_right,
            c=z_right,
            cmap="viridis",
            s=10,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )
        cbar = fig.colorbar(sc_right, ax=axes.tolist())
    else:
        axes[1].text(0.5, 0.5, "No valid frac_host_psf_2500 values", ha="center", va="center", transform=axes[1].transAxes)
        cbar = fig.colorbar(sc_left, ax=axes.tolist())
    axes[1].axhline(0.0, color="k", ls="--", lw=1, alpha=0.8)
    axes[1].set_xlabel(r"$\log_{10}(\mathrm{frac\_host\_psf\_2500})$")
    axes[1].set_ylabel(r"$\Delta \log \sigma_{\rm UV}$")
    axes[1].grid(True, alpha=0.25)
    cbar.set_label("Redshift z")

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_tau_sigma_vs_redshift(df, plot_path="plots/hubble", show=False, filename="tau_sigma_vs_redshift.pdf"):
    """Plot log_tau_uv_rf and log_sigma_uv against redshift for AGN diagnostics."""
    required = {"z", "log_tau_uv_rf", "log_sigma_uv"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise KeyError(f"Missing required columns for tau/sigma vs redshift plot: {missing}")

    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
    log_tau = pd.to_numeric(df["log_tau_uv_rf"], errors="coerce").to_numpy(dtype=float)
    log_sigma = pd.to_numeric(df["log_sigma_uv"], errors="coerce").to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 10.0), sharex=True)

    mask_tau = np.isfinite(z) & np.isfinite(log_tau)
    if np.any(mask_tau):
        axes[0].scatter(
            z[mask_tau],
            log_tau[mask_tau],
            s=5,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )
    else:
        axes[0].text(0.5, 0.5, "No finite log_tau_uv_rf values", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set_xlabel("Redshift z")
    axes[0].set_ylabel(r"$\log \tau_{\rm UV,RF}$")
    axes[0].grid(True, alpha=0.25)

    mask_sigma = np.isfinite(z) & np.isfinite(log_sigma)
    if np.any(mask_sigma):
        axes[1].scatter(
            z[mask_sigma],
            log_sigma[mask_sigma],
            s=5,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )
    else:
        axes[1].text(0.5, 0.5, "No finite log_sigma_uv values", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_xlabel("Redshift z")
    axes[1].set_ylabel(r"$\log \sigma_{\rm UV}$")
    axes[1].grid(True, alpha=0.25)

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_sigma_uv_vs_tau_uv_rf(
    df_agn,
    *,
    plot_path="plots/hubble",
    show=False,
    thin_step=5,
    filename="sigma_uv_vs_tau_uv_rf.pdf",
    dynamic_axes=False,
):
    """Plot log_sigma_uv against log_tau_uv_rf."""
    required = {"log_sigma_uv", "log_tau_uv_rf"}
    if not required.issubset(df_agn.columns):
        missing = ", ".join(sorted(required - set(df_agn.columns)))
        raise KeyError(f"Missing required columns for sigma/tau plot: {missing}")

    log_sigma = pd.to_numeric(df_agn["log_sigma_uv"], errors="coerce").to_numpy(dtype=float)
    log_tau_rf = pd.to_numeric(df_agn["log_tau_uv_rf"], errors="coerce").to_numpy(dtype=float)
    log_sigma_err = (
        pd.to_numeric(df_agn["log_sigma_uv_err"], errors="coerce").to_numpy(dtype=float)
        if "log_sigma_uv_err" in df_agn.columns
        else np.full(len(df_agn), np.nan, dtype=float)
    )
    log_tau_rf_err = (
        pd.to_numeric(df_agn["log_tau_uv_rf_err"], errors="coerce").to_numpy(dtype=float)
        if "log_tau_uv_rf_err" in df_agn.columns
        else np.full(len(df_agn), np.nan, dtype=float)
    )

    mask = np.isfinite(log_sigma) & np.isfinite(log_tau_rf)
    if np.any(mask):
        idx = np.flatnonzero(mask)
        if thin_step is not None and thin_step > 1:
            idx = idx[::thin_step]
    else:
        idx = np.array([], dtype=int)

    fig = plt.figure(figsize=(7.8, 7.0))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(4.0, 1.2),
        height_ratios=(1.2, 4.0),
        hspace=0.05,
        wspace=0.05,
    )
    ax_hist_x = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_hist_x)
    ax_hist_y = fig.add_subplot(gs[1, 1], sharey=ax)
    if idx.size > 0:
        ax.errorbar(
            log_sigma[idx],
            log_tau_rf[idx],
            xerr=log_sigma_err[idx] if np.any(np.isfinite(log_sigma_err[idx])) else None,
            yerr=log_tau_rf_err[idx] if np.any(np.isfinite(log_tau_rf_err[idx])) else None,
            fmt="o",
            color="0.3",
            ecolor="0.3",
            markersize=2.2,
            alpha=0.7,
            elinewidth=0.4,
            capsize=0,
            rasterized=True,
        )
        if dynamic_axes:
            xpad = 0.05 * max(np.nanmax(log_sigma[idx]) - np.nanmin(log_sigma[idx]), 1e-6)
            ypad = 0.05 * max(np.nanmax(log_tau_rf[idx]) - np.nanmin(log_tau_rf[idx]), 1e-6)
            ax.set_xlim(np.nanmin(log_sigma[idx]) - xpad, np.nanmax(log_sigma[idx]) + xpad)
            ax.set_ylim(np.nanmin(log_tau_rf[idx]) - ypad, np.nanmax(log_tau_rf[idx]) + ypad)

        sigma_use = log_sigma[idx]
        tau_use = log_tau_rf[idx]
        sigma_med = np.nanmedian(sigma_use)
        tau_med = np.nanmedian(tau_use)
        sigma_med_linear = 10.0 ** sigma_med
        tau_med_linear = 10.0 ** tau_med
        ax_hist_x.hist(sigma_use, bins=30, color="0.35", histtype="stepfilled", alpha=0.25)
        ax_hist_x.axvline(sigma_med, color="k", ls="--", lw=1.5)
        ax_hist_x.text(
            0.98,
            0.90,
            f"{sigma_med_linear:.1f} mag",
            ha="right",
            va="top",
            transform=ax_hist_x.transAxes,
        )
        ax_hist_y.hist(
            tau_use,
            bins=30,
            orientation="horizontal",
            color="0.35",
            histtype="stepfilled",
            alpha=0.25,
        )
        ax_hist_y.axhline(tau_med, color="k", ls="--", lw=1.5)
        ax_hist_y.text(
            0.95,
            0.98,
            f"{100.0 * np.round(tau_med_linear / 100.0):.0f} days",
            ha="right",
            va="top",
            transform=ax_hist_y.transAxes,
        )
    else:
        ax.text(0.5, 0.5, "No finite log_sigma_uv/log_tau_uv_rf values", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel(r"$\log \sigma_{\rm UV}$")
    ax.set_ylabel(r"$\log \tau_{\rm UV,RF}$")
    ax_hist_x.tick_params(labelbottom=False)
    ax_hist_y.tick_params(labelleft=False)
    ax_hist_x.set_ylabel("N")
    ax_hist_y.set_xlabel("N")

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_sigma_tau_err_std_psd_comparison(
    df,
    *,
    plot_path="plots/hubble",
    show=False,
    filename="sigma_tau_err_std_psd_comparison.pdf",
):
    """Compare raw percentile errors against regularized PSD std terms."""
    specs = [
        (
            "log_sigma_uv_err",
            "log_sigma_uv_std_psd",
            r"$\log \sigma_{\rm UV}$",
        ),
        (
            "log_tau_uv_rf_err",
            "log_tau_uv_rf_std_psd",
            r"$\log \tau_{\rm UV,RF}$",
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), sharey=False)
    for ax, (raw_col, psd_col, axis_label) in zip(axes, specs):
        if {raw_col, psd_col}.issubset(df.columns):
            raw_err = pd.to_numeric(df[raw_col], errors="coerce").to_numpy(dtype=float)
            psd_std = pd.to_numeric(df[psd_col], errors="coerce").to_numpy(dtype=float)
            mask = (
                np.isfinite(raw_err)
                & np.isfinite(psd_std)
                & (raw_err > 0.0)
                & (psd_std > 0.0)
            )
        else:
            mask = np.array([], dtype=bool)

        if np.any(mask):
            x = raw_err[mask]
            y = psd_std[mask]
            ax.scatter(
                x,
                y,
                s=6,
                alpha=0.45,
                linewidths=0,
                color="0.25",
                rasterized=True,
            )
            lo = float(np.nanmin(np.concatenate([x, y])))
            hi = float(np.nanmax(np.concatenate([x, y])))
            pad = 0.05 * max(hi - lo, 1e-6)
            ax.plot(
                [lo - pad, hi + pad],
                [lo - pad, hi + pad],
                color="tab:red",
                lw=1.8,
                zorder=3,
            )
            ratio = y / x
            ax.text(
                0.04,
                0.96,
                "\n".join(
                    [
                        f"N={np.count_nonzero(mask)}",
                        f"median ratio={np.nanmedian(ratio):.3f}",
                    ]
                ),
                ha="left",
                va="top",
                transform=ax.transAxes,
            )
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
        else:
            ax.text(
                0.5,
                0.5,
                f"No finite {raw_col}/{psd_col} rows",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_xlabel(f"{raw_col} (raw percentile)")
        ax.set_ylabel(f"{psd_col} (regularized PSD)")
        ax.set_title(axis_label)

    fig.tight_layout()
    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_sigma_uv_vs_variability_chi_sq_red_g(
    df,
    plot_path="plots/hubble",
    show=False,
    filename="sigma_uv_vs_variability_chi_sq_red_g.pdf",
):
    """Plot log_sigma_uv against log10 reduced g-band variability chi-squared."""
    required = {"log_sigma_uv", "variability_chi_sq_red_g"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise KeyError(f"Missing required columns for sigma_uv vs reduced chi-squared plot: {missing}")

    log_sigma = pd.to_numeric(df["log_sigma_uv"], errors="coerce").to_numpy(dtype=float)
    chi_sq_red = pd.to_numeric(df["variability_chi_sq_red_g"], errors="coerce").to_numpy(dtype=float)
    log_chi_sq_red = np.full(len(df), np.nan, dtype=float)
    positive = np.isfinite(chi_sq_red) & (chi_sq_red > 0.0)
    log_chi_sq_red[positive] = np.log10(chi_sq_red[positive])
    mask = np.isfinite(log_sigma) & np.isfinite(log_chi_sq_red)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.0))
    if np.any(mask):
        ax.scatter(
            log_chi_sq_red[mask],
            log_sigma[mask],
            s=8,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No finite log_sigma_uv/log_variability_chi_sq_red_g values",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel(r"$\log \chi^2_{\rm red,g}$")
    ax.set_ylabel(r"$\log \sigma_{\rm UV}$")
    ax.grid(True, alpha=0.25)

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_tau_sigma_vs_wu_catalog(df, plot_path="plots/hubble", show=False, filename="tau_sigma_vs_wu_catalog.pdf"):
    """Plot UV variability diagnostics against Wu-catalog BH mass and Eddington ratio."""
    required = {"log_tau_uv_rf", "log_sigma_uv", "LOGMBH", "LOGLEDD_RATIO"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise KeyError(f"Missing required columns for Wu-catalog diagnostic plot: {missing}")

    log_tau = pd.to_numeric(df["log_tau_uv_rf"], errors="coerce").to_numpy(dtype=float)
    log_sigma = pd.to_numeric(df["log_sigma_uv"], errors="coerce").to_numpy(dtype=float)
    log_mbh = pd.to_numeric(df["LOGMBH"], errors="coerce").to_numpy(dtype=float)
    log_edd = pd.to_numeric(df["LOGLEDD_RATIO"], errors="coerce").to_numpy(dtype=float)
    mask_tau = np.isfinite(log_mbh) & np.isfinite(log_tau)
    mask_sigma = np.isfinite(log_edd) & np.isfinite(log_sigma)
    if (not np.any(mask_tau)) and (not np.any(mask_sigma)):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), sharey=False)

    if np.any(mask_tau):
        axes[0].scatter(
            log_mbh[mask_tau],
            log_tau[mask_tau],
            s=5,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )
    else:
        axes[0].text(0.5, 0.5, "No finite LOGMBH/log_tau_uv_rf values", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set_xlabel(r"$\log M_{\rm BH}$ (Wu catalog)")
    axes[0].set_ylabel(r"$\log \tau_{\rm UV,RF}$")
    axes[0].grid(True, alpha=0.25)

    if np.any(mask_sigma):
        axes[1].scatter(
            log_edd[mask_sigma],
            log_sigma[mask_sigma],
            s=5,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )
    else:
        axes[1].text(0.5, 0.5, "No finite LOGLEDD_RATIO/log_sigma_uv values", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_xlabel(r"$\log (L/L_{\rm Edd})$ (Wu catalog)")
    axes[1].set_ylabel(r"$\log \sigma_{\rm UV}$")
    axes[1].grid(True, alpha=0.25)

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def _fit_suberlak_relation_fixed_lambda(
    y,
    yerr,
    M_abs,
    M_abs_err,
    log_mbh,
    log_mbh_err,
    *,
    b_lambda,
    lambda_rf=2500.0,
    nwarm=500,
    nsamp=500,
    nuts_seed=0,
):
    """Fit Suberlak's relation with fixed B and x/y measurement errors."""
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    M_abs = np.asarray(M_abs, dtype=float)
    M_abs_err = np.asarray(M_abs_err, dtype=float)
    log_mbh = np.asarray(log_mbh, dtype=float)
    log_mbh_err = np.asarray(log_mbh_err, dtype=float)
    x_lambda = np.log10(float(lambda_rf) / 4000.0)
    mask = (
        np.isfinite(y)
        & np.isfinite(M_abs)
        & np.isfinite(M_abs_err)
        & np.isfinite(log_mbh)
        & np.isfinite(log_mbh_err)
        & np.isfinite(yerr)
        & (yerr > 0.0)
        & (M_abs_err > 0.0)
        & (log_mbh_err > 0.0)
    )
    if np.count_nonzero(mask) < 4:
        return None

    y_fit = y[mask]
    yerr_fit = yerr[mask]
    M_abs_fit = M_abs[mask]
    M_abs_err_fit = M_abs_err[mask]
    log_mbh_fit = log_mbh[mask]
    log_mbh_err_fit = log_mbh_err[mask]

    x_mat = np.column_stack(
        [
            np.ones(np.count_nonzero(mask), dtype=float),
            M_abs_fit + 23.0,
            log_mbh_fit - 9.0,
        ]
    )
    y_no_lambda = y_fit - float(b_lambda) * x_lambda
    beta_init, *_ = np.linalg.lstsq(x_mat, y_no_lambda, rcond=None)

    try:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
    except Exception as exc:
        print(
            "[WARNING] NumPyro unavailable for Suberlak-style NUTS regression; "
            f"falling back to weighted least squares. Error: {exc}"
        )

        def _solve_beta_and_nll(log_sigma_int):
            sigma_int = np.exp(float(log_sigma_int))
            sigma_eff = np.sqrt(
                np.maximum(yerr_fit, 1e-6) ** 2
                + sigma_int**2
                + (beta_init[1] * M_abs_err_fit) ** 2
                + (beta_init[2] * log_mbh_err_fit) ** 2
            )
            w_sqrt = 1.0 / sigma_eff
            beta_trial, *_ = np.linalg.lstsq(
                x_mat * w_sqrt[:, None],
                y_no_lambda * w_sqrt,
                rcond=None,
            )
            sigma_eff = np.sqrt(
                np.maximum(yerr_fit, 1e-6) ** 2
                + sigma_int**2
                + (beta_trial[1] * M_abs_err_fit) ** 2
                + (beta_trial[2] * log_mbh_err_fit) ** 2
            )
            resid_trial = y_no_lambda - x_mat @ beta_trial
            nll_trial = 0.5 * np.sum(
                (resid_trial / sigma_eff) ** 2 + np.log(2.0 * np.pi * sigma_eff**2)
            )
            return beta_trial, sigma_int, resid_trial, float(nll_trial)

        opt = minimize_scalar(
            lambda log_sigma_int: _solve_beta_and_nll(log_sigma_int)[3],
            bounds=(-8.0, 2.0),
            method="bounded",
        )
        beta, sigma_int, resid_no_lambda, nll = _solve_beta_and_nll(opt.x)
        samples = {
            "A": np.array([beta[0]], dtype=float),
            "C": np.array([beta[1]], dtype=float),
            "D": np.array([beta[2]], dtype=float),
            "sigma_int": np.array([sigma_int], dtype=float),
        }
    else:
        y_j = jnp.asarray(y_fit)
        yerr_j = jnp.asarray(np.maximum(yerr_fit, 1e-6))
        m_j = jnp.asarray(M_abs_fit + 23.0)
        merr_j = jnp.asarray(np.maximum(M_abs_err_fit, 1e-6))
        bh_j = jnp.asarray(log_mbh_fit - 9.0)
        bherr_j = jnp.asarray(np.maximum(log_mbh_err_fit, 1e-6))
        xlam_j = jnp.asarray(x_lambda)

        def _model():
            A = numpyro.sample("A", dist.Normal(float(beta_init[0]), 1.0))
            C = numpyro.sample("C", dist.Normal(float(beta_init[1]), 0.5))
            D = numpyro.sample("D", dist.Normal(float(beta_init[2]), 0.5))
            sigma_int = numpyro.sample("sigma_int", dist.HalfNormal(1.0))
            mu = A + float(b_lambda) * xlam_j + C * m_j + D * bh_j
            sigma_eff = jnp.sqrt(
                yerr_j**2
                + sigma_int**2
                + (C * merr_j) ** 2
                + (D * bherr_j) ** 2
            )
            numpyro.sample("y_obs", dist.Normal(mu, sigma_eff), obs=y_j)

        kernel = NUTS(_model)
        mcmc = MCMC(
            kernel,
            num_warmup=int(nwarm),
            num_samples=int(nsamp),
            num_chains=1,
            progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(int(nuts_seed)))
        samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
        beta = np.array(
            [
                np.median(samples["A"]),
                np.median(samples["C"]),
                np.median(samples["D"]),
            ],
            dtype=float,
        )
        sigma_int = float(np.median(samples["sigma_int"]))
        resid_no_lambda = y_no_lambda - x_mat @ beta
        sigma_eff = np.sqrt(
            np.maximum(yerr_fit, 1e-6) ** 2
            + sigma_int**2
            + (beta[1] * M_abs_err_fit) ** 2
            + (beta[2] * log_mbh_err_fit) ** 2
        )
        nll = float(
            0.5
            * np.sum(
                (resid_no_lambda / sigma_eff) ** 2
                + np.log(2.0 * np.pi * sigma_eff**2)
            )
        )
    y_model = x_mat @ beta + float(b_lambda) * x_lambda
    resid = y[mask] - y_model
    sigma_eff = np.sqrt(np.maximum(yerr_fit, 1e-6) ** 2 + sigma_int**2)
    sigma_eff = np.sqrt(
        np.maximum(yerr_fit, 1e-6) ** 2
        + sigma_int**2
        + (beta[1] * M_abs_err_fit) ** 2
        + (beta[2] * log_mbh_err_fit) ** 2
    )
    chi2 = np.sum((resid_no_lambda / sigma_eff) ** 2)
    dof = max(1, np.count_nonzero(mask) - 4)
    return {
        "A": float(beta[0]),
        "B": float(b_lambda),
        "C": float(beta[1]),
        "D": float(beta[2]),
        "sigma_int": float(sigma_int),
        "A_err": float(0.5 * (np.percentile(samples["A"], 84) - np.percentile(samples["A"], 16))),
        "C_err": float(0.5 * (np.percentile(samples["C"], 84) - np.percentile(samples["C"], 16))),
        "D_err": float(0.5 * (np.percentile(samples["D"], 84) - np.percentile(samples["D"], 16))),
        "sigma_int_err": float(
            0.5
            * (
                np.percentile(samples["sigma_int"], 84)
                - np.percentile(samples["sigma_int"], 16)
            )
        ),
        "mask": mask,
        "y_model": y_model,
        "resid": resid,
        "chi2_red": float(chi2 / dof),
        "nll": float(nll),
        "n_used": int(np.count_nonzero(mask)),
        "lambda_rf": float(lambda_rf),
    }


def plot_suberlak_style_sigma_tau_fits(
    df,
    plot_path="plots/hubble",
    show=False,
    filename="suberlak_style_sigma_tau_fits.pdf",
    lambda_rf=2500.0,
    abs_mag_column="M_2500",
    sample_label=None,
):
    """Fit Suberlak-form regressions to log_sigma_uv and log_tau_uv_rf."""
    required = {"LOGMBH", "log_sigma_uv", "log_tau_uv_rf"}
    if abs_mag_column == "M_2500":
        required |= {"z", "apparent_mag_2500"}
    elif abs_mag_column == "M_i_Wu_z2":
        required.add("LOGLBOL_CORRECTED")
    else:
        required.add(abs_mag_column)
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise KeyError(f"Missing required columns for Suberlak-style regression plot: {missing}")

    if abs_mag_column == "M_2500":
        cosmo_fid = FlatLambdaCDM(H0=70.0, Om0=0.3)
        z_vals = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
        M_abs = (
            pd.to_numeric(df["apparent_mag_2500"], errors="coerce").to_numpy(dtype=float)
            - cosmo_fid.distmod(z_vals).value
        )
        M_abs_err = np.sqrt(
            pd.to_numeric(df["apparent_mag_2500_err"], errors="coerce").fillna(np.nan).to_numpy(dtype=float) ** 2
            + sigma_mu_from_z_err(
                z_vals,
                pd.to_numeric(df["z_err"], errors="coerce").fillna(np.nan).to_numpy(dtype=float),
                cosmo_fid,
            ) ** 2
        )
        abs_mag_label = r"$M_{2500}$"
    elif abs_mag_column == "M_i_Wu_z2":
        loglbol_corrected = pd.to_numeric(
            df["LOGLBOL_CORRECTED"],
            errors="coerce",
        ).to_numpy(dtype=float)
        M_abs = 91.0 - 2.5 * loglbol_corrected
        M_abs = np.where(np.isfinite(M_abs) & (M_abs <= 0.0), M_abs, np.nan)
        if "LOGLBOL_CORRECTED_ERR" in df.columns:
            M_abs_err = 2.5 * pd.to_numeric(
                df["LOGLBOL_CORRECTED_ERR"],
                errors="coerce",
            ).to_numpy(dtype=float)
        else:
            M_abs_err = np.full(len(df), np.nan, dtype=float)
        abs_mag_label = r"$M_{i,\rm Wu}$"
    else:
        M_abs = pd.to_numeric(df[abs_mag_column], errors="coerce").to_numpy(dtype=float)
        err_col = f"{abs_mag_column}_err"
        if err_col in df.columns:
            M_abs_err = pd.to_numeric(df[err_col], errors="coerce").to_numpy(dtype=float)
        else:
            M_abs_err = np.full(len(df), np.nan, dtype=float)
        abs_mag_label = rf"${abs_mag_column}$"
    log_mbh = pd.to_numeric(df["LOGMBH"], errors="coerce").to_numpy(dtype=float)
    if "LOGMBH_ERR" in df.columns:
        log_mbh_err = pd.to_numeric(df["LOGMBH_ERR"], errors="coerce").to_numpy(dtype=float)
    else:
        log_mbh_err = np.full(len(df), np.nan, dtype=float)

    fit_specs = [
        {
            "y_col": "log_sigma_uv",
            "yerr_col": "log_sigma_uv_std_psd" if "log_sigma_uv_std_psd" in df.columns else "log_sigma_uv_err",
            "b_lambda": -0.479,
            "sub_c": 0.118,
            "sub_d": 0.118,
            "ylabel": r"$\log \sigma_{\rm UV}$",
            "name": "sigma_uv",
        },
        {
            "y_col": "log_tau_uv_rf",
            "yerr_col": "log_tau_uv_rf_std_psd" if "log_tau_uv_rf_std_psd" in df.columns else "log_tau_uv_rf_err",
            "b_lambda": 0.17,
            "sub_c": 0.035,
            "sub_d": 0.141,
            "ylabel": r"$\log \tau_{\rm UV,RF}$",
            "name": "tau_uv_rf",
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=False)
    for ax, spec in zip(axes, fit_specs):
        if spec["yerr_col"] not in df.columns:
            ax.text(
                0.5,
                0.5,
                f"No {spec['yerr_col']} column",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xlabel(f"Suberlak-form prediction for {spec['name']}")
            ax.set_ylabel(spec["ylabel"])
            continue

        y = pd.to_numeric(df[spec["y_col"]], errors="coerce").to_numpy(dtype=float)
        yerr = pd.to_numeric(df[spec["yerr_col"]], errors="coerce").to_numpy(dtype=float)
        fit = _fit_suberlak_relation_fixed_lambda(
            y,
            yerr,
            M_abs,
            M_abs_err,
            log_mbh,
            log_mbh_err,
            b_lambda=spec["b_lambda"],
            lambda_rf=lambda_rf,
        )
        if fit is None:
            ax.text(
                0.5,
                0.5,
                f"Not enough finite rows for {spec['name']}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xlabel(f"Suberlak-form prediction for {spec['name']}")
            ax.set_ylabel(spec["ylabel"])
            continue

        y_obs = y[fit["mask"]]
        y_model = fit["y_model"]
        yerr_fit = yerr[fit["mask"]]
        print(
            f"Suberlak-style fit"
            f"{'' if sample_label is None else f' [{sample_label}]'} "
            f"for {spec['name']} using {abs_mag_column} "
            f"at lambda_RF={fit['lambda_rf']:.0f} A "
            f"(B fixed={fit['B']:.3f}, N={fit['n_used']}, chi2_red={fit['chi2_red']:.3f}): "
            f"A={fit['A']:.4f}±{fit['A_err']:.4f}, "
            f"C={fit['C']:.4f}±{fit['C_err']:.4f} (Suberlak {spec['sub_c']:.4f}), "
            f"D={fit['D']:.4f}±{fit['D_err']:.4f} (Suberlak {spec['sub_d']:.4f}), "
            f"sigma_int={fit['sigma_int']:.4f}±{fit['sigma_int_err']:.4f} dex"
        )
        ax.errorbar(
            y_model,
            y_obs,
            yerr=yerr_fit,
            fmt="o",
            linestyle="none",
            markersize=3.0,
            mfc=(0, 0, 0, 0.35),
            mec="none",
            ecolor=(0.2, 0.2, 0.2, 0.15),
            elinewidth=0.8,
            capsize=0,
            zorder=2,
            rasterized=True,
        )
        if y_obs.size > 50:
            try:
                kde = gaussian_kde(np.vstack([y_model, y_obs]), bw_method="scott")
                xq = np.quantile(y_model, [0.01, 0.99])
                yq = np.quantile(y_obs, [0.01, 0.99])
                rx = xq[1] - xq[0]
                ry = yq[1] - yq[0]
                x_grid_kde, y_grid_kde = np.meshgrid(
                    np.linspace(xq[0] - 0.1 * rx, xq[1] + 0.1 * rx, 220),
                    np.linspace(yq[0] - 0.1 * ry, yq[1] + 0.1 * ry, 220),
                )
                z_kde = kde(
                    np.vstack([x_grid_kde.ravel(), y_grid_kde.ravel()])
                ).reshape(x_grid_kde.shape)
                levels = _kde_conf_levels(z_kde, conf=(0.954, 0.683))
                ax.contour(
                    x_grid_kde,
                    y_grid_kde,
                    z_kde,
                    levels=levels,
                    colors=["0.5", "0.2"],
                    linestyles=["--", "-"],
                    linewidths=[1.6, 2.0],
                    zorder=4,
                )
            except Exception as exc:
                print(f"[Suberlak-style KDE contours] skipped for {spec['name']}: {exc}")
        lo = float(np.nanmin(np.concatenate([y_obs, y_model])))
        hi = float(np.nanmax(np.concatenate([y_obs, y_model])))
        pad = 0.05 * max(hi - lo, 1e-3)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="tab:red", lw=1.8, zorder=3)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"Suberlak-form prediction for {spec['name']} using {abs_mag_column}")
        ax.set_ylabel(spec["ylabel"])
        ax.text(
            0.04,
            0.96,
            "\n".join(
                [
                    rf"$B={fit['B']:.3f}$ fixed",
                    rf"$C={fit['C']:.3f}\pm{fit['C_err']:.3f}$, "
                    rf"$D={fit['D']:.3f}\pm{fit['D_err']:.3f}$",
                    rf"$\sigma_{{\rm int}}={fit['sigma_int']:.3f}\pm{fit['sigma_int_err']:.3f}$ dex",
                    rf"$\chi^2_\nu={fit['chi2_red']:.2f}$",
                ]
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
        )
        ax.legend(
            handles=[
                Line2D([0], [0], color="0.2", lw=2.0, ls="-", label=r"1$\sigma$"),
                Line2D([0], [0], color="0.5", lw=1.6, ls="--", label=r"2$\sigma$"),
            ],
            loc="lower right",
            frameon=False,
            fontsize=11,
        )

    fig.suptitle(
        rf"Suberlak-form fits at fixed $\lambda_{{\rm RF}}={lambda_rf:.0f}\,\AA$ "
        rf"using {abs_mag_label} (fit $A,C,D$; $B$ fixed)",
        y=0.99,
    )
    fig.tight_layout()
    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_eta_tau_sigma_vs_redshift(
    df,
    plot_path="plots/hubble",
    show=False,
    filename="eta_tau_sigma_vs_redshift.pdf",
):
    """Plot eta_tau and eta_sigma against redshift for AGN diagnostics."""
    required = {"z", "eta_tau", "eta_sigma"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise KeyError(f"Missing required columns for eta-vs-redshift plot: {missing}")

    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
    eta_tau = pd.to_numeric(df["eta_tau"], errors="coerce").to_numpy(dtype=float)
    eta_sigma = pd.to_numeric(df["eta_sigma"], errors="coerce").to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 10.0), sharex=True)

    mask_tau = np.isfinite(z) & np.isfinite(eta_tau)
    if np.any(mask_tau):
        axes[0].scatter(
            z[mask_tau],
            eta_tau[mask_tau],
            s=5,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )
    else:
        axes[0].text(0.5, 0.5, "No finite eta_tau values", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set_xlabel("Redshift z")
    axes[0].set_ylabel(r"$\eta_{\tau}$")
    axes[0].grid(True, alpha=0.25)

    mask_sigma = np.isfinite(z) & np.isfinite(eta_sigma)
    if np.any(mask_sigma):
        axes[1].scatter(
            z[mask_sigma],
            eta_sigma[mask_sigma],
            s=5,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )
    else:
        axes[1].text(0.5, 0.5, "No finite eta_sigma values", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_xlabel("Redshift z")
    axes[1].set_ylabel(r"$\eta_{\sigma}$")
    axes[1].grid(True, alpha=0.25)

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_fast_vs_uv_variability(df, plot_path="plots/hubble", show=False, filename="fast_vs_uv_variability.pdf"):
    """Plot fast-vs-UV variability timescales and amplitudes on log-log axes."""
    tau_fast_col = "log_tau_fast_uv" if "log_tau_fast_uv" in df.columns else None
    tau_uv_col = "log_tau_uv_rf" if "log_tau_uv_rf" in df.columns else ("log_tau_uv" if "log_tau_uv" in df.columns else None)
    sigma_fast_col = 'log_sigma_fast_uv' if "log_sigma_fast_uv" in df.columns else None
    sigma_uv_col = "log_sigma_uv" if "log_sigma_uv" in df.columns else None

    if tau_fast_col is None or tau_uv_col is None:
        missing = []
        if tau_fast_col is None:
            missing.append("log_tau_fast_uv")
        if tau_uv_col is None:
            missing.append("log_tau_uv_rf or log_tau_uv")
        raise KeyError(f"Missing required columns for fast-vs-UV diagnostic plot: {', '.join(missing)}")

    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float) if "z" in df.columns else np.full(len(df), np.nan)
    log_tau_fast = pd.to_numeric(df[tau_fast_col], errors="coerce").to_numpy(dtype=float)
    log_tau_uv = pd.to_numeric(df[tau_uv_col], errors="coerce").to_numpy(dtype=float)
    log_sigma_fast = (
        pd.to_numeric(df[sigma_fast_col], errors="coerce").to_numpy(dtype=float)
        if sigma_fast_col is not None else np.full(len(df), np.nan)
    )
    log_sigma_uv = (
        pd.to_numeric(df[sigma_uv_col], errors="coerce").to_numpy(dtype=float)
        if sigma_uv_col is not None else np.full(len(df), np.nan)
    )

    if tau_uv_col == "log_tau_uv" and "z" in df.columns:
        log_tau_uv = log_tau_uv - np.log10(1.0 + z)
    if tau_fast_col == "log_tau_fast_uv" and "z" in df.columns:
        log_tau_fast = log_tau_fast - np.log10(1.0 + z)

    tau_fast = np.power(10.0, log_tau_fast)
    tau_uv = np.power(10.0, log_tau_uv)
    sigma_fast = np.power(10.0, log_sigma_fast)
    sigma_uv = np.power(10.0, log_sigma_uv)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    panels = [
        (
            axes[0],
            tau_uv,
            tau_fast,
            r"$\tau_{\rm UV,RF}$ [days]",
            r"$\tau_{\rm fast,UV,RF}$ [days]",
            "No finite tau_fast/tau_uv values",
        ),
        (
            axes[1],
            sigma_uv,
            sigma_fast,
            r"$\sigma_{\rm UV}$ [mag]",
            r"$\sigma_{\rm fast}$ [mag]",
            "No distinct sigma_fast/sigma_uv values",
        ),
    ]

    last_scatter = None
    for ax, x, y, xlabel, ylabel, empty_label in panels:
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        color_mask = mask & np.isfinite(z)
        if np.any(mask):
            cvals = z[mask] if np.any(color_mask) else None
            last_scatter = ax.scatter(
                x[mask],
                y[mask],
                c=cvals,
                cmap="viridis" if cvals is not None else None,
                s=10,
                alpha=0.65,
                linewidths=0,
                rasterized=True,
            )
            lo = min(np.nanmin(x[mask]), np.nanmin(y[mask]))
            hi = max(np.nanmax(x[mask]), np.nanmax(y[mask]))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                ax.plot([lo, hi], [lo, hi], color="k", ls="--", lw=1.0, alpha=0.8)
        else:
            ax.text(0.5, 0.5, empty_label, ha="center", va="center", transform=ax.transAxes)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    if last_scatter is not None and last_scatter.get_array() is not None:
        cbar = fig.colorbar(last_scatter, ax=axes.tolist())
        cbar.set_label("Redshift z")

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_sf_vs_uv_variability(
    df,
    plot_path="plots/hubble",
    show=False,
    filename="sf_vs_uv_variability.pdf",
):
    """Compare UV-converted SF summaries from the g-band fit against the main UV variability fit."""
    required = {"log_sigma_uv", "log_sigma_uv_sf", "log_tau_uv_rf_sf"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise KeyError(f"Missing required columns for SF-vs-UV diagnostic plot: {missing}")

    tau_uv_col = "log_tau_uv_rf" if "log_tau_uv_rf" in df.columns else ("log_tau_uv" if "log_tau_uv" in df.columns else None)
    if tau_uv_col is None:
        missing = []
        if tau_uv_col is None:
            missing.append("log_tau_uv_rf or log_tau_uv")
        raise KeyError(f"Missing required columns for SF-vs-UV diagnostic plot: {', '.join(missing)}")

    variability_chi_sq_g = (
        pd.to_numeric(df["variability_chi_sq_g"], errors="coerce").to_numpy(dtype=float)
        if "variability_chi_sq_g" in df.columns else np.full(len(df), np.nan)
    )
    log_variability_chi_sq_g = np.full(len(df), np.nan, dtype=float)
    positive_chi_sq = np.isfinite(variability_chi_sq_g) & (variability_chi_sq_g > 0.0)
    log_variability_chi_sq_g[positive_chi_sq] = np.log10(variability_chi_sq_g[positive_chi_sq])
    log_sigma_uv = pd.to_numeric(df["log_sigma_uv"], errors="coerce").to_numpy(dtype=float)
    log_sigma_sf = pd.to_numeric(df["log_sigma_uv_sf"], errors="coerce").to_numpy(dtype=float)
    log_tau_uv = pd.to_numeric(df[tau_uv_col], errors="coerce").to_numpy(dtype=float)
    log_tau_sf = pd.to_numeric(df["log_tau_uv_rf_sf"], errors="coerce").to_numpy(dtype=float)
    sf_valid = (
        pd.Series(df["sf_valid"]).fillna(False).astype(bool).to_numpy()
        if "sf_valid" in df.columns else np.ones(len(df), dtype=bool)
    )

    if tau_uv_col == "log_tau_uv" and "z" in df.columns:
        z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
        log_tau_uv = log_tau_uv - np.log10(1.0 + z)

    sigma_uv = np.power(10.0, log_sigma_uv)
    sigma_sf = np.power(10.0, log_sigma_sf)
    tau_uv = np.power(10.0, log_tau_uv)
    tau_sf = np.power(10.0, log_tau_sf)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    panels = [
        (
            axes[0],
            sigma_uv,
            sigma_sf,
            r"$\sigma_{\rm UV}$ [mag]",
            r"$\sigma_{\rm SF}$ [mag]",
            "No finite sigma_sf/sigma_uv values",
        ),
        (
            axes[1],
            tau_uv,
            tau_sf,
            r"$\tau_{\rm UV,RF}$ [days]",
            r"$\tau_{\rm SF,UV,RF}$ [days]",
            "No finite tau_sf/tau_uv values",
        ),
    ]

    last_scatter = None
    for ax, x, y, xlabel, ylabel, empty_label in panels:
        mask = sf_valid & np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        color_mask = mask & np.isfinite(log_variability_chi_sq_g)
        if np.any(mask):
            if np.any(~color_mask & mask):
                ax.scatter(
                    x[mask & ~color_mask],
                    y[mask & ~color_mask],
                    color="0.75",
                    s=10,
                    alpha=0.35,
                    linewidths=0,
                    rasterized=True,
                )
            if np.any(color_mask):
                last_scatter = ax.scatter(
                    x[color_mask],
                    y[color_mask],
                    c=log_variability_chi_sq_g[color_mask],
                    cmap="viridis",
                    s=10,
                    alpha=0.65,
                    linewidths=0,
                    rasterized=True,
                )
            lo = min(np.nanmin(x[mask]), np.nanmin(y[mask]))
            hi = max(np.nanmax(x[mask]), np.nanmax(y[mask]))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                ax.plot([lo, hi], [lo, hi], color="k", ls="--", lw=1.0, alpha=0.8)
        else:
            ax.text(0.5, 0.5, empty_label, ha="center", va="center", transform=ax.transAxes)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    if last_scatter is not None and last_scatter.get_array() is not None:
        cbar = fig.colorbar(last_scatter, ax=axes.tolist())
        cbar.set_label(r"$\log_{10}(\chi^2_g)$")

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_sf_ref_band_vs_model_g(
    df,
    plot_path="plots/hubble",
    show=False,
    filename="sf_ref_band_vs_model_g.pdf",
):
    """Compare empirical SF summaries against the closest model-equivalent g-band quantities."""
    required = {"log_sigma_sf_ref_band", "log_tau_sf_ref_band"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise KeyError(f"Missing required columns for SF-ref-vs-model-g diagnostic plot: {missing}")

    sigma_col = "log_sigma_rms_band_g" if "log_sigma_rms_band_g" in df.columns else "log_sigma_band_g"
    tau_col = "log_tau_sf_model_ref_band" if "log_tau_sf_model_ref_band" in df.columns else "log_tau_band_g_RF"
    if sigma_col not in df.columns or tau_col not in df.columns:
        missing = [col for col in (sigma_col, tau_col) if col not in df.columns]
        raise KeyError(
            "Missing required model columns for SF-ref-vs-model-g diagnostic plot: "
            + ", ".join(sorted(missing))
        )

    log_sigma_model_g = pd.to_numeric(df[sigma_col], errors="coerce").to_numpy(dtype=float)
    log_sigma_sf_ref = pd.to_numeric(df["log_sigma_sf_ref_band"], errors="coerce").to_numpy(dtype=float)
    log_tau_model_g = pd.to_numeric(df[tau_col], errors="coerce").to_numpy(dtype=float)
    log_tau_sf_ref = pd.to_numeric(df["log_tau_sf_ref_band"], errors="coerce").to_numpy(dtype=float)
    sf_valid = (
        pd.Series(df["sf_valid"]).fillna(False).astype(bool).to_numpy()
        if "sf_valid" in df.columns else np.ones(len(df), dtype=bool)
    )
    ref_band_ok = (
        pd.Series(df["sf_ref_band"]).fillna("").astype(str).eq("g").to_numpy()
        if "sf_ref_band" in df.columns else np.ones(len(df), dtype=bool)
    )
    variability_chi_sq_g = (
        pd.to_numeric(df["variability_chi_sq_g"], errors="coerce").to_numpy(dtype=float)
        if "variability_chi_sq_g" in df.columns else np.full(len(df), np.nan)
    )
    log_variability_chi_sq_g = np.full(len(df), np.nan, dtype=float)
    positive_chi_sq = np.isfinite(variability_chi_sq_g) & (variability_chi_sq_g > 0.0)
    log_variability_chi_sq_g[positive_chi_sq] = np.log10(variability_chi_sq_g[positive_chi_sq])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    panels = [
        (
            axes[0],
            log_sigma_model_g,
            log_sigma_sf_ref,
            r"$\log \sigma_{g,\mathrm{model\ RMS}}$" if sigma_col == "log_sigma_rms_band_g" else r"$\log \sigma_{g,\mathrm{model}}$",
            r"$\log \sigma_{\mathrm{SF,ref}}$",
            "No finite SF/model-g sigma values",
        ),
        (
            axes[1],
            log_tau_model_g,
            log_tau_sf_ref,
            r"$\log \tau_{\mathrm{SF,model\ equiv}}$" if tau_col == "log_tau_sf_model_ref_band" else r"$\log \tau_{g,\mathrm{model,RF}}$",
            r"$\log \tau_{\mathrm{SF,ref}}$",
            "No finite SF/model-g tau values",
        ),
    ]

    last_scatter = None
    for ax, x, y, xlabel, ylabel, empty_label in panels:
        mask = sf_valid & ref_band_ok & np.isfinite(x) & np.isfinite(y)
        color_mask = mask & np.isfinite(log_variability_chi_sq_g)
        if np.any(mask):
            if np.any(mask & ~color_mask):
                ax.scatter(
                    x[mask & ~color_mask],
                    y[mask & ~color_mask],
                    color="0.75",
                    s=10,
                    alpha=0.35,
                    linewidths=0,
                    rasterized=True,
                )
            if np.any(color_mask):
                last_scatter = ax.scatter(
                    x[color_mask],
                    y[color_mask],
                    c=log_variability_chi_sq_g[color_mask],
                    cmap="viridis",
                    s=10,
                    alpha=0.65,
                    linewidths=0,
                    rasterized=True,
                )
            lo = min(np.nanmin(x[mask]), np.nanmin(y[mask]))
            hi = max(np.nanmax(x[mask]), np.nanmax(y[mask]))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                pad = 0.05 * max(hi - lo, 1e-6)
                ax.plot([lo, hi], [lo, hi], color="k", ls="--", lw=1.0, alpha=0.8)
                ax.set_xlim(lo - pad, hi + pad)
                ax.set_ylim(lo - pad, hi + pad)
        else:
            ax.text(0.5, 0.5, empty_label, ha="center", va="center", transform=ax.transAxes)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    if last_scatter is not None and last_scatter.get_array() is not None:
        cbar = fig.colorbar(last_scatter, ax=axes.tolist())
        cbar.set_label(r"$\log_{10}(\chi^2_g)$")

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_f_host_2500_vs_l2500(
    df,
    plot_path="plots/hubble",
    show=False,
    nbins=15,
    min_bin_count=5,
    fit_logL_max=45.5,
    filename="f_host_2500_vs_l2500.pdf",
):
    """Plot host fraction against AGN-only log L_2500 with median and sigmoid trends."""
    required = {"z", "apparent_mag_2500", "f_host_2500"}
    if not required.issubset(df.columns):
        return None

    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
    m2500 = pd.to_numeric(df["apparent_mag_2500"], errors="coerce").to_numpy(dtype=float)
    f_host = pd.to_numeric(df["f_host_2500"], errors="coerce").to_numpy(dtype=float)

    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    logL2500 = apparent_mag_to_logL2500(m2500, z, cosmo)

    mask = (
        np.isfinite(logL2500)
        & np.isfinite(f_host)
        & np.isfinite(z)
        & (z > 0.0)
        & (f_host >= 0.0)
        & (f_host <= 1.0)
    )

    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    if np.any(mask):
        x = logL2500[mask]
        y = f_host[mask]
        ax.scatter(
            x,
            y,
            s=10,
            alpha=0.3,
            color="tab:blue",
            linewidths=0,
            rasterized=True,
            label="Objects",
        )

        if np.nanmax(x) > np.nanmin(x):
            bin_edges = np.linspace(np.nanmin(x), np.nanmax(x), nbins + 1)
            xmid = []
            ymed = []
            for i in range(len(bin_edges) - 1):
                lo = bin_edges[i]
                hi = bin_edges[i + 1]
                in_bin = (x >= lo) & (x < hi)
                if i == len(bin_edges) - 2:
                    in_bin = (x >= lo) & (x <= hi)
                if np.count_nonzero(in_bin) >= min_bin_count:
                    xmid.append(np.nanmedian(x[in_bin]))
                    ymed.append(np.nanmedian(y[in_bin]))
            if xmid:
                ax.plot(xmid, ymed, color="k", lw=2, label="Binned median")

        fit_mask = np.isfinite(x) & np.isfinite(y) & (x <= fit_logL_max)
        if np.count_nonzero(fit_mask) >= 8 and np.nanmax(x[fit_mask]) > np.nanmin(x[fit_mask]):
            try:
                fit_model = fit_fhost_2500_l2500_model(
                    df.loc[mask].copy(),
                    fit_logL_max=fit_logL_max,
                    cosmo=cosmo,
                )
                x_grid = np.linspace(np.nanmin(x), np.nanmax(x), 400)
                y_grid = predict_fhost_2500_from_logL2500(x_grid, fit_model)
                ax.plot(
                    x_grid,
                    y_grid,
                    color="tab:red",
                    lw=2,
                    label=rf"Generalized sigmoid fit ($f\to1$ low-$L$, $f\to0$ high-$L$)",
                )
                ax.text(
                    0.03,
                    0.03,
                    (
                        rf"$x_0={fit_model['x0']:.2f}$" "\n"
                        rf"$k={fit_model['k']:.2f}$" "\n"
                        rf"$\nu={fit_model['nu']:.2f}$" "\n"
                        rf"$\sigma_{{\rm logit}}={fit_model['sigma_host_logit']:.2f}$"
                    ),
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none"),
                )
            except Exception:
                pass
    else:
        ax.text(
            0.5,
            0.5,
            "No finite log L_2500 / f_host_2500 values",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_xlabel(r"$\log L_{2500}$")
    ax.set_ylabel(r"$f_{\rm host,2500}$")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False)
    fig.tight_layout()

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(fig, os.path.join(diagnostics_path, filename), dpi=200, show=show)


def plot_alpha_lambda_vs_l2500_by_redshift(
    df,
    plot_path="plots/hubble",
    show=False,
    z_bin_width=0.5,
    nbins_l2500=12,
    min_bin_count=5,
):
    """Plot alpha_lambda against AGN-only log L_2500 in redshift bins."""
    required = {"z", "apparent_mag_2500", "alpha_lambda"}
    if not required.issubset(df.columns):
        return None

    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
    m2500 = pd.to_numeric(df["apparent_mag_2500"], errors="coerce").to_numpy(dtype=float)
    alpha_lambda = pd.to_numeric(df["alpha_lambda"], errors="coerce").to_numpy(dtype=float)
    logL2500 = apparent_mag_to_logL2500(m2500, z, FlatLambdaCDM(H0=70.0, Om0=0.3))

    mask = np.isfinite(z) & np.isfinite(m2500) & np.isfinite(alpha_lambda) & np.isfinite(logL2500) & (z > 0.0)
    if not np.any(mask):
        return None

    z_use = z[mask]
    logL_use = logL2500[mask]
    alpha_use = alpha_lambda[mask]

    z_lo = float(np.floor(np.nanmin(z_use) / z_bin_width) * z_bin_width)
    z_hi = float(np.ceil(np.nanmax(z_use) / z_bin_width) * z_bin_width)
    z_edges = np.arange(z_lo, z_hi + z_bin_width, z_bin_width)
    if z_edges.size < 2:
        z_edges = np.array([z_lo, z_lo + z_bin_width], dtype=float)

    panel_masks = []
    panel_labels = []
    for i in range(len(z_edges) - 1):
        lo = z_edges[i]
        hi = z_edges[i + 1]
        in_bin = (z_use >= lo) & (z_use < hi)
        if i == len(z_edges) - 2:
            in_bin = (z_use >= lo) & (z_use <= hi)
        if np.count_nonzero(in_bin) == 0:
            continue
        panel_masks.append(in_bin)
        panel_labels.append((lo, hi))

    if not panel_masks:
        return None

    n_panels = len(panel_masks)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.3 * n_cols, 4.8 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes = axes.ravel()

    x_min = float(np.nanmin(logL_use))
    x_max = float(np.nanmax(logL_use))
    y_min = float(np.nanmin(alpha_use))
    y_max = float(np.nanmax(alpha_use))
    y_pad = 0.08 * max(y_max - y_min, 1.0)

    for ax, in_bin, (z0, z1) in zip(axes, panel_masks, panel_labels):
        x = logL_use[in_bin]
        y = alpha_use[in_bin]
        ax.scatter(
            x,
            y,
            s=10,
            alpha=0.32,
            color="tab:blue",
            linewidths=0,
            rasterized=True,
        )

        if np.nanmax(x) > np.nanmin(x):
            l_edges = np.linspace(np.nanmin(x), np.nanmax(x), nbins_l2500 + 1)
            xmid = []
            ymed = []
            for i in range(len(l_edges) - 1):
                lo = l_edges[i]
                hi = l_edges[i + 1]
                keep = (x >= lo) & (x < hi)
                if i == len(l_edges) - 2:
                    keep = (x >= lo) & (x <= hi)
                if np.count_nonzero(keep) >= min_bin_count:
                    xmid.append(np.nanmedian(x[keep]))
                    ymed.append(np.nanmedian(y[keep]))
            if xmid:
                ax.plot(xmid, ymed, color="k", lw=2)

        if z1 < z_edges[-1]:
            ax.set_title(f"{z0:.1f} <= z < {z1:.1f}")
        else:
            ax.set_title(f"{z0:.1f} <= z <= {z1:.1f}")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    for ax in axes[-n_cols:]:
        if ax.get_visible():
            ax.set_xlabel(r"$\log L_{2500}$")
    for row in range(n_rows):
        ax = axes[row * n_cols]
        if ax.get_visible():
            ax.set_ylabel(r"$\alpha_{\lambda}$")

    fig.tight_layout()
    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, "alpha_lambda_vs_l2500_by_redshift.pdf"),
        dpi=200,
        show=show,
    )


def plot_alpha_lambda_vs_l2500(
    df,
    plot_path="plots/hubble",
    show=False,
    nbins_l2500=16,
    min_bin_count=8,
    filename="alpha_lambda_vs_l2500.pdf",
):
    """Plot alpha_lambda against AGN-only log L_2500 in a single diagnostic panel."""
    required = {"z", "apparent_mag_2500", "alpha_lambda"}
    if not required.issubset(df.columns):
        return None

    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
    m2500 = pd.to_numeric(df["apparent_mag_2500"], errors="coerce").to_numpy(dtype=float)
    alpha_lambda = pd.to_numeric(df["alpha_lambda"], errors="coerce").to_numpy(dtype=float)
    logL2500 = apparent_mag_to_logL2500(m2500, z, FlatLambdaCDM(H0=70.0, Om0=0.3))

    mask = np.isfinite(z) & np.isfinite(m2500) & np.isfinite(alpha_lambda) & np.isfinite(logL2500) & (z > 0.0)
    if not np.any(mask):
        return None

    x = logL2500[mask]
    y = alpha_lambda[mask]
    z_use = z[mask]

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    sc = ax.scatter(
        x,
        y,
        c=z_use,
        cmap="viridis",
        s=12,
        alpha=0.4,
        linewidths=0,
        rasterized=True,
    )

    if np.nanmax(x) > np.nanmin(x):
        l_edges = np.linspace(np.nanmin(x), np.nanmax(x), nbins_l2500 + 1)
        xmid = []
        ymed = []
        for i in range(len(l_edges) - 1):
            lo = l_edges[i]
            hi = l_edges[i + 1]
            keep = (x >= lo) & (x < hi)
            if i == len(l_edges) - 2:
                keep = (x >= lo) & (x <= hi)
            if np.count_nonzero(keep) >= min_bin_count:
                xmid.append(np.nanmedian(x[keep]))
                ymed.append(np.nanmedian(y[keep]))
        if xmid:
            ax.plot(xmid, ymed, color="k", lw=2, label="Binned median")

    ax.set_xlabel(r"$\log L_{2500}$")
    ax.set_ylabel(r"$\alpha_{\lambda}$")
    ax.grid(True, alpha=0.2)
    cbar = fig.colorbar(sc, ax=ax, orientation="vertical")
    cbar.set_label("Redshift z")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False)
    fig.tight_layout()

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_alpha_lambda_histogram(df, plot_path="plots/hubble", show=False, nbins=40, filename="alpha_lambda_histogram.pdf"):
    """Plot a simple alpha_lambda histogram and estimate its 1 sigma width."""
    if "alpha_lambda" not in df.columns:
        raise KeyError("Missing required column: 'alpha_lambda'.")

    alpha = pd.to_numeric(df["alpha_lambda"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(alpha)
    alpha = alpha[mask]
    if alpha.size == 0:
        raise ValueError("No finite alpha_lambda values available for histogram.")

    p16, p50, p84 = np.nanpercentile(alpha, [16, 50, 84])
    sigma_1 = 0.5 * (p84 - p16)
    std = float(np.nanstd(alpha))

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.hist(alpha, bins=nbins, color="tab:blue", alpha=0.75, edgecolor="white")
    ax.axvline(p50, color="k", lw=2, label=fr"median = {p50:.2f}")
    ax.axvline(p16, color="k", lw=1.5, ls="--")
    ax.axvline(p84, color="k", lw=1.5, ls="--", label=fr"$\sigma_{{68}}$ = {sigma_1:.2f}")
    ax.set_xlabel(r"$\alpha_{\lambda}$")
    ax.set_ylabel("Count")
    ax.set_title(
        f"alpha_lambda distribution\nN={alpha.size}, sigma_68={sigma_1:.2f}, std={std:.2f}"
    )
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, loc="best")

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_alpha_lambda_vs_redshift(
    df,
    plot_path="plots/hubble",
    show=False,
    nbins_z=14,
    min_bin_count=6,
    filename="alpha_lambda_vs_redshift.pdf",
):
    """Plot alpha_lambda against redshift with a binned median trend."""
    required = {"z", "alpha_lambda"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise KeyError(f"Missing required columns for alpha_lambda vs redshift plot: {missing}")

    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
    alpha_lambda = pd.to_numeric(df["alpha_lambda"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(z) & np.isfinite(alpha_lambda) & (z > 0.0)
    if not np.any(mask):
        return None

    z_use = z[mask]
    alpha_use = alpha_lambda[mask]

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(
        z_use,
        alpha_use,
        s=12,
        alpha=0.35,
        color="tab:blue",
        linewidths=0,
        rasterized=True,
    )

    if np.nanmax(z_use) > np.nanmin(z_use):
        z_edges = np.linspace(np.nanmin(z_use), np.nanmax(z_use), nbins_z + 1)
        zmid = []
        ymed = []
        for i in range(len(z_edges) - 1):
            lo = z_edges[i]
            hi = z_edges[i + 1]
            keep = (z_use >= lo) & (z_use < hi)
            if i == len(z_edges) - 2:
                keep = (z_use >= lo) & (z_use <= hi)
            if np.count_nonzero(keep) >= min_bin_count:
                zmid.append(np.nanmedian(z_use[keep]))
                ymed.append(np.nanmedian(alpha_use[keep]))
        if zmid:
            ax.plot(zmid, ymed, color="k", lw=2, label="Binned median")

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\alpha_{\lambda}$")
    ax.grid(True, alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False)
    fig.tight_layout()

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_alpha_lambda_vs_eta_sigma(
    df,
    plot_path="plots/hubble",
    show=False,
    nbins_eta=14,
    min_bin_count=6,
    filename="alpha_lambda_vs_eta_sigma.pdf",
):
    """Plot alpha_lambda against eta_sigma with a binned median trend."""
    required = {"alpha_lambda", "eta_sigma"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise KeyError(f"Missing required columns for alpha_lambda vs eta_sigma plot: {missing}")

    eta_sigma = pd.to_numeric(df["eta_sigma"], errors="coerce").to_numpy(dtype=float)
    alpha_lambda = pd.to_numeric(df["alpha_lambda"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(eta_sigma) & np.isfinite(alpha_lambda)
    if not np.any(mask):
        return None

    x = eta_sigma[mask]
    y = alpha_lambda[mask]

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(
        x,
        y,
        s=12,
        alpha=0.35,
        color="tab:blue",
        linewidths=0,
        rasterized=True,
    )

    if np.nanmax(x) > np.nanmin(x):
        x_edges = np.linspace(np.nanmin(x), np.nanmax(x), nbins_eta + 1)
        xmid = []
        ymed = []
        for i in range(len(x_edges) - 1):
            lo = x_edges[i]
            hi = x_edges[i + 1]
            keep = (x >= lo) & (x < hi)
            if i == len(x_edges) - 2:
                keep = (x >= lo) & (x <= hi)
            if np.count_nonzero(keep) >= min_bin_count:
                xmid.append(np.nanmedian(x[keep]))
                ymed.append(np.nanmedian(y[keep]))
        if xmid:
            ax.plot(xmid, ymed, color="k", lw=2, label="Binned median")

    ax.set_xlabel(r"$\eta_{\sigma}$")
    ax.set_ylabel(r"$\alpha_{\lambda}$")
    ax.grid(True, alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False)
    fig.tight_layout()

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_blr_lag_vs_amp_by_band(df, plot_path="plots/hubble", show=False, lag_suffix="", filename=None):
    """Plot BLR lag against inferred BLR amplitude in each band.

    The plotted BLR amplitude is constructed from the continuum variability
    amplitude plus the band-specific BLR amplitude offset. If a band-specific
    continuum amplitude column is available, use it; otherwise fall back to the
    UV-reference continuum amplitude.
    """
    suffix = str(lag_suffix or "")
    amp_delta_prefix = f"log_amp_delta_blr{suffix}_"
    lag_prefix = f"log_lag_blr{suffix}_"
    lag_rf_prefix = f"log_lag_blr{suffix}_"

    bands = [
        band
        for band in ("u", "g", "r", "i", "z")
        if f"{amp_delta_prefix}{band}" in df.columns
    ]
    if not bands:
        raise KeyError(f"No {amp_delta_prefix}<band> columns found in the dataframe.")

    continuum_ref_col = None
    for candidate in ("log_sigma_uv",):
        if candidate in df.columns:
            continuum_ref_col = candidate
            break
    if continuum_ref_col is None:
        raise KeyError("Missing continuum amplitude column: need 'log_sigma_uv'.")

    n_panels = len(bands)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.3 * n_cols, 4.8 * n_rows),
        squeeze=False,
    )
    axes = axes.ravel()

    dropped_bands = df["dropped_bands"] if "dropped_bands" in df.columns else None
    title_label = "BLR 2" if suffix == "2" else "BLR"
    log_kl_by_band = {}
    finite_log_kl_values = []
    for band in bands:
        lag_kl_col = f"{lag_prefix}{band}_kl"
        if lag_kl_col not in df.columns:
            log_kl_by_band[band] = np.full(len(df), np.nan, dtype=float)
            continue
        lag_kl = pd.to_numeric(df[lag_kl_col], errors="coerce").to_numpy(dtype=float)
        log_kl = np.full_like(lag_kl, np.nan, dtype=float)
        finite_positive = np.isfinite(lag_kl) & (lag_kl > 0.0)
        log_kl[finite_positive] = np.clip(np.log10(lag_kl[finite_positive]), -3.0, None)
        log_kl_by_band[band] = log_kl
        finite_log_kl_values.append(log_kl[finite_positive])

    kl_norm = None
    if finite_log_kl_values:
        finite_log_kl = np.concatenate(finite_log_kl_values)
        if finite_log_kl.size:
            kl_vmin = float(np.nanmin(finite_log_kl))
            kl_vmax = float(np.nanmax(finite_log_kl))
            if np.isclose(kl_vmin, kl_vmax):
                kl_vmin -= 0.5
                kl_vmax += 0.5
            kl_norm = colors.Normalize(vmin=kl_vmin, vmax=kl_vmax)
    kl_cmap = mpl.colormaps["viridis"]

    for ax, band in zip(axes, bands):
        continuum_band_col = f"log_sigma_band_{band}"
        continuum_col = continuum_band_col if continuum_band_col in df.columns else continuum_ref_col
        amp_col = f"{amp_delta_prefix}{band}"
        lag_candidates = (
            f"{lag_rf_prefix}{band}_RF",
            f"{lag_prefix}{band}",
        )
        lag_col = next((candidate for candidate in lag_candidates if candidate in df.columns), None)

        if lag_col is None:
            ax.text(
                0.5,
                0.5,
                f"No lag column for {band}-band",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        log_amp_blr = (
            pd.to_numeric(df[continuum_col], errors="coerce").to_numpy(dtype=float)
            + pd.to_numeric(df[amp_col], errors="coerce").to_numpy(dtype=float)
        )
        log_lag = pd.to_numeric(df[lag_col], errors="coerce").to_numpy(dtype=float)
        log_kl = log_kl_by_band[band]
        mask = np.isfinite(log_amp_blr) & np.isfinite(log_lag)
        if dropped_bands is not None:
            mask &= ~dropped_bands.apply(
                lambda s: band in s if isinstance(s, (list, tuple, set, str)) else False
            ).to_numpy(dtype=bool)

        if np.any(mask):
            finite_kl_mask = mask & np.isfinite(log_kl)
            missing_kl_mask = mask & ~np.isfinite(log_kl)
            if np.any(missing_kl_mask):
                ax.scatter(
                    log_amp_blr[missing_kl_mask],
                    log_lag[missing_kl_mask],
                    s=7,
                    color="0.75",
                    alpha=0.35,
                    linewidths=0,
                    rasterized=True,
                )
            ax.scatter(
                log_amp_blr[finite_kl_mask] if np.any(finite_kl_mask) else log_amp_blr[mask],
                log_lag[finite_kl_mask] if np.any(finite_kl_mask) else log_lag[mask],
                c=log_kl[finite_kl_mask] if np.any(finite_kl_mask) else "0.75",
                cmap=kl_cmap if np.any(finite_kl_mask) else None,
                norm=kl_norm if np.any(finite_kl_mask) else None,
                s=7,
                alpha=0.6,
                linewidths=0,
                rasterized=True,
            )
        else:
            ax.text(
                0.5,
                0.5,
                f"No finite {band}-band values",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        lag_label = r"$\log \tau_{\rm BLR,RF}$" if lag_col.endswith("_RF") else r"$\log \tau_{\rm BLR}$"
        if suffix == "2":
            lag_label = lag_label.replace("BLR", "BLR,2")
        ax.set_xlabel(r"$\log A_{\rm BLR}$")
        ax.set_ylabel(lag_label)
        ax.set_title(f"{title_label} {band}-band")
        ax.grid(True, alpha=0.25)

    for ax in axes[n_panels:]:
        ax.set_axis_off()

    if kl_norm is not None:
        sm = mpl.cm.ScalarMappable(norm=kl_norm, cmap=kl_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[:n_panels], pad=0.02, fraction=0.05)
        cbar.set_label(r"$\log_{10}\,\mathrm{KL}_\tau$")

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    output_name = filename if filename is not None else ("blr_lag2_vs_amp_by_band.pdf" if suffix == "2" else "blr_lag_vs_amp_by_band.pdf")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, output_name),
        dpi=200,
        show=show,
    )


def plot_blr_lag_vs_redshift_by_band(df, plot_path="plots/hubble", show=False, lag_suffix="", filename=None):
    """Plot BLR lag against redshift in each band."""
    suffix = str(lag_suffix or "")
    lag_prefix = f"log_lag_blr{suffix}_"

    bands = [
        band
        for band in ("u", "g", "r", "i", "z")
        if (
            f"{lag_prefix}{band}_RF" in df.columns
            or f"{lag_prefix}{band}" in df.columns
        )
    ]
    if not bands:
        raise KeyError(f"No {lag_prefix}<band> lag columns found in the dataframe.")
    if "z" not in df.columns:
        raise KeyError("Missing required 'z' column for BLR lag vs redshift plot.")

    n_panels = len(bands)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.3 * n_cols, 4.8 * n_rows),
        squeeze=False,
    )
    axes = axes.ravel()

    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
    dropped_bands = df["dropped_bands"] if "dropped_bands" in df.columns else None
    title_label = "BLR 2" if suffix == "2" else "BLR"

    for ax, band in zip(axes, bands):
        lag_candidates = (
            f"{lag_prefix}{band}_RF",
            f"{lag_prefix}{band}",
        )
        lag_col = next((candidate for candidate in lag_candidates if candidate in df.columns), None)
        if lag_col is None:
            ax.text(
                0.5,
                0.5,
                f"No lag column for {band}-band",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        log_lag = pd.to_numeric(df[lag_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(z) & np.isfinite(log_lag)
        if dropped_bands is not None:
            mask &= ~dropped_bands.apply(
                lambda s: band in s if isinstance(s, (list, tuple, set, str)) else False
            ).to_numpy(dtype=bool)

        if np.any(mask):
            ax.scatter(
                z[mask],
                log_lag[mask],
                s=7,
                alpha=0.6,
                linewidths=0,
                rasterized=True,
            )
        else:
            ax.text(
                0.5,
                0.5,
                f"No finite {band}-band values",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        lag_label = r"$\log \tau_{\rm BLR,RF}$" if lag_col.endswith("_RF") else r"$\log \tau_{\rm BLR}$"
        if suffix == "2":
            lag_label = lag_label.replace("BLR", "BLR,2")
        ax.set_xlabel("Redshift z")
        ax.set_ylabel(lag_label)
        ax.set_title(f"{title_label} {band}-band")
        ax.grid(True, alpha=0.25)

    for ax in axes[n_panels:]:
        ax.set_axis_off()

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    output_name = filename if filename is not None else (
        "blr_lag2_vs_redshift_by_band.pdf"
        if suffix == "2"
        else "blr_lag_vs_redshift_by_band.pdf"
    )
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, output_name),
        dpi=200,
        show=show,
    )


def plot_blr_amp_vs_redshift_by_band(df, plot_path="plots/hubble", show=False, lag_suffix="", filename=None):
    """Plot inferred BLR amplitude against redshift in each band."""
    suffix = str(lag_suffix or "")
    amp_delta_prefix = f"log_amp_delta_blr{suffix}_"

    bands = [
        band
        for band in ("u", "g", "r", "i", "z")
        if f"{amp_delta_prefix}{band}" in df.columns
    ]
    if not bands:
        raise KeyError(f"No {amp_delta_prefix}<band> columns found in the dataframe.")

    continuum_ref_col = None
    for candidate in ("log_sigma_uv",):
        if candidate in df.columns:
            continuum_ref_col = candidate
            break
    if continuum_ref_col is None:
        raise KeyError("Missing continuum amplitude column: need 'log_sigma_uv'.")

    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
    dropped_bands = df["dropped_bands"] if "dropped_bands" in df.columns else None
    title_label = "BLR 2" if suffix == "2" else "BLR"

    n_panels = len(bands)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.3 * n_cols, 4.8 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes = axes.ravel()

    for ax, band in zip(axes, bands):
        continuum_band_col = f"log_sigma_band_{band}"
        continuum_col = continuum_band_col if continuum_band_col in df.columns else continuum_ref_col
        amp_col = f"{amp_delta_prefix}{band}"

        log_amp_blr = (
            pd.to_numeric(df[continuum_col], errors="coerce").to_numpy(dtype=float)
            + pd.to_numeric(df[amp_col], errors="coerce").to_numpy(dtype=float)
        )
        mask = np.isfinite(z) & np.isfinite(log_amp_blr)
        if dropped_bands is not None:
            mask &= ~dropped_bands.apply(lambda bands_set: band in bands_set).to_numpy()

        if np.any(mask):
            ax.scatter(
                z[mask],
                log_amp_blr[mask],
                s=10,
                alpha=0.4,
                color="tab:blue",
                linewidths=0,
                rasterized=True,
            )
        else:
            ax.text(
                0.5,
                0.5,
                f"No finite {title_label} amplitudes for {band}-band",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_xlabel(r"$z$")
        ax.set_ylabel(rf"$\log A_{{\rm {title_label.lower().replace(' ', '')},{band}}}$")
        ax.set_title(f"{title_label} amplitude vs z ({band}-band)")
        ax.grid(True, alpha=0.2)

    for ax in axes[n_panels:]:
        ax.set_axis_off()

    fig.tight_layout()
    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    output_name = filename if filename is not None else ("blr2_amp_vs_redshift_by_band.pdf" if suffix == "2" else "blr_amp_vs_redshift_by_band.pdf")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, output_name),
        dpi=200,
        show=show,
    )


def _plot_dm_by_band(
    df,
    *,
    x_getter,
    x_label,
    output_name,
    df_keep=None,
    bands=("u", "g", "r", "i", "z"),
    z_range=(0.44, 3.16),
    show=False,
    alpha=0.5,
    s=6,
    rolling_window=501,
    plot_path="plots/hubble/diagnostics",
):
    """Plot PSF-minus-fiber offsets by band against a chosen x-axis quantity."""
    band_cols = [(band, f"psf_ps1_minus_fiber_sdss_{band}") for band in bands if f"psf_ps1_minus_fiber_sdss_{band}" in df.columns]
    if not band_cols:
        raise KeyError("No psf_ps1_minus_fiber_sdss_{band} columns found in the dataframe.")

    n_panels = len(band_cols)
    n_rows = min(4, n_panels)
    n_cols = int(np.ceil(n_panels / n_rows))

    # Lay out one panel per band so band-dependent PSF-fiber trends are easy to compare.
    id_col = "object_id" if "object_id" in df.columns else None
    keep_ids = None
    if df_keep is not None:
        if id_col is not None and id_col in df_keep.columns:
            keep_ids = set(df_keep[id_col].astype(str))
        else:
            keep_ids = set(df_keep.index.tolist())

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 4.5 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    z = np.asarray(df["z"], dtype=float)
    for ax, (band, col) in zip(axes, band_cols):
        x = np.asarray(x_getter(df, band), dtype=float)
        y = np.asarray(df[col], dtype=float)
        petro_col = f"petroRad_{band}_sdss"
        petro = np.asarray(df[petro_col], dtype=float) if petro_col in df.columns else np.full(len(df), np.nan)
        mask = np.isfinite(x) & np.isfinite(z) & np.isfinite(y) & np.isfinite(petro) & (petro > 0)
        petro_plot = np.log10(petro[mask])
        if petro_plot.size > 0:
            vmin, vmax = np.nanpercentile(petro_plot, [1, 99])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin = np.nanmin(petro_plot)
                vmax = np.nanmax(petro_plot)
            petro_plot = np.clip(petro_plot, vmin, vmax)
        else:
            vmin, vmax = None, None
        idx = np.flatnonzero(mask)
        if keep_ids is not None:
            if id_col is not None and id_col in df.columns:
                keep_mask = df.iloc[idx][id_col].astype(str).isin(keep_ids).to_numpy(dtype=bool)
            else:
                keep_mask = np.array([i in keep_ids for i in idx], dtype=bool)
        else:
            keep_mask = np.ones(len(idx), dtype=bool)
        x_masked = x[mask]
        z_masked = z[mask]
        y_masked = y[mask]
        in_z = (z_masked >= z_range[0]) & (z_masked <= z_range[1])

        keep_in_z = keep_mask & in_z
        keep_out_z = keep_mask & (~in_z)
        cut_in_z = (~keep_mask) & in_z
        cut_out_z = (~keep_mask) & (~in_z)

        cmap_obj = mpl.cm.get_cmap("viridis")
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        color_keep_out_z = cmap_obj(norm(petro_plot[keep_out_z])) if np.any(keep_out_z) else None
        color_cut_out_z = cmap_obj(norm(petro_plot[cut_out_z])) if np.any(cut_out_z) else None

        sc = ax.scatter(
            x_masked[keep_in_z],
            y_masked[keep_in_z],
            c=petro_plot[keep_in_z],
            s=s,
            alpha=alpha,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
            label=f"{band}-band",
        )
        ax.scatter(
            x_masked[cut_in_z],
            y_masked[cut_in_z],
            c=petro_plot[cut_in_z],
            s=s,
            alpha=alpha,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
            marker="D",
        )
        ax.scatter(
            x_masked[keep_out_z],
            y_masked[keep_out_z],
            c=color_keep_out_z,
            s=s,
            alpha=1.0,
            marker="D",
            linewidths=1.5,
            rasterized=True,
        )
        ax.scatter(
            x_masked[cut_out_z],
            y_masked[cut_out_z],
            c=color_cut_out_z,
            s=s,
            alpha=1.0,
            marker="D",
            linewidths=1.5,
            rasterized=True,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(rf"$\log_{{10}}(\mathrm{{petroRad}}_{{{band}}})$")

        # Overlay a rolling median in redshift to highlight broad trends by band.
        if np.count_nonzero(mask) >= 5:
            order = np.argsort(x[mask])
            x_sorted = x[mask][order]
            y_sorted = y[mask][order]

            window = min(int(rolling_window), len(x_sorted))
            if window % 2 == 0:
                window = max(1, window - 1)
            window = max(21, window)

            y_med = (
                pd.Series(y_sorted)
                .rolling(window=window, center=True, min_periods=max(3, window // 5))
                .median()
                .to_numpy()
            )
            med_mask = np.isfinite(y_med)
            ax.plot(x_sorted[med_mask], y_med[med_mask], color="darkorange", lw=2.0, zorder=3, label="rolling median")

        ax.axhline(0.0, lw=1.0, color="k", alpha=0.6)
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"$m_{\rm PS1,PSF} - m_{\rm SDSS,fiber}$")
        ax.set_ylim(-2, 1)
        ax.legend(loc="upper right", frameon=False)

    # Hide any unused subplot slots in the grid.
    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.tight_layout()
    os.makedirs(plot_path, exist_ok=True)
    _save_figure(fig, os.path.join(plot_path, output_name), dpi=200, show=show)


def plot_df_psf_fiber(
    df,
    df_keep=None,
    bands=("u", "g", "r", "i", "z"),
    z_range=(0.44, 3.16),
    show=False,
    alpha=0.5,
    s=6,
    rolling_window=501,
    plot_path="plots/hubble/diagnostics",
):
    """Plot PS1 PSF minus SDSS fiber magnitude offsets versus redshift for each band."""
    return _plot_dm_by_band(
        df,
        x_getter=lambda frame, band: frame["z"].to_numpy(dtype=float),
        x_label="Redshift (z)",
        output_name="psf_ps1_minus_fiber_sdss_by_band.pdf",
        df_keep=df_keep,
        bands=bands,
        z_range=z_range,
        show=show,
        alpha=alpha,
        s=s,
        rolling_window=rolling_window,
        plot_path=plot_path,
    )


def plot_df_psf_fiber_vs_petro(
    df,
    df_keep=None,
    bands=("u", "g", "r", "i", "z"),
    z_range=(0.44, 3.16),
    show=False,
    alpha=0.5,
    s=6,
    rolling_window=501,
    plot_path="plots/hubble/diagnostics",
):
    """Plot PS1 PSF minus SDSS fiber magnitude offsets versus Petrosian radius for each band."""
    return _plot_dm_by_band(
        df,
        x_getter=lambda frame, band: np.log10(np.asarray(frame[f"petroRad_{band}_sdss"], dtype=float)),
        x_label=r"$\log_{10}(\mathrm{petroRad})$",
        output_name="psf_ps1_minus_fiber_sdss_vs_petrorad_by_band.pdf",
        df_keep=df_keep,
        bands=bands,
        z_range=z_range,
        show=show,
        alpha=alpha,
        s=s,
        rolling_window=rolling_window,
        plot_path=plot_path,
    )


def plot_df_psf_fiber_vs_fhost(
    df,
    df_keep=None,
    bands=("u", "g", "r", "i", "z"),
    z_range=(0.44, 3.16),
    show=False,
    alpha=0.5,
    s=6,
    rolling_window=501,
    plot_path="plots/hubble/diagnostics",
):
    """Plot PS1 PSF minus SDSS fiber magnitude offsets versus log10(f_host_2500) for each band."""
    return _plot_dm_by_band(
        df,
        x_getter=lambda frame, band: np.where(
            np.asarray(frame["f_host_2500"], dtype=float) > 0,
            np.log10(np.asarray(frame["f_host_2500"], dtype=float)),
            np.nan,
        ),
        x_label=r"$\log_{10}(f_{\mathrm{host,2500}})$",
        output_name="psf_ps1_minus_fiber_sdss_vs_fhost_by_band.pdf",
        df_keep=df_keep,
        bands=bands,
        z_range=z_range,
        show=show,
        alpha=alpha,
        s=s,
        rolling_window=rolling_window,
        plot_path=plot_path,
    )


def plot_log_fhost_vs_petrorad_by_band(
    df,
    df_keep=None,
    bands=("u", "g", "r", "i", "z"),
    z_range=(0.44, 3.16),
    show=False,
    alpha=0.5,
    s=6,
    plot_path="plots/hubble/diagnostics",
):
    """Plot log10(f_host_2500) against log10(Petrosian radius) in each band."""
    if "f_host_2500" not in df.columns:
        return None

    id_col = "object_id" if "object_id" in df.columns else None
    keep_ids = None
    if df_keep is not None:
        if id_col is not None and id_col in df_keep.columns:
            keep_ids = set(df_keep[id_col].astype(str))
        else:
            keep_ids = set(df_keep.index.tolist())

    band_cols = [(band, f"petroRad_{band}_sdss") for band in bands if f"petroRad_{band}_sdss" in df.columns]
    if not band_cols:
        return None

    n_panels = len(band_cols)
    n_rows = min(4, n_panels)
    n_cols = int(np.ceil(n_panels / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 4.5 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    z = np.asarray(df["z"], dtype=float)
    log_fhost = np.where(
        np.asarray(df["f_host_2500"], dtype=float) > 0,
        np.log10(np.asarray(df["f_host_2500"], dtype=float)),
        np.nan,
    )

    for ax, (band, petro_col) in zip(axes, band_cols):
        petro = np.asarray(df[petro_col], dtype=float)
        log_petro = np.where(petro > 0, np.log10(petro), np.nan)
        mask = np.isfinite(z) & np.isfinite(log_fhost) & np.isfinite(log_petro)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            ax.axis("off")
            continue

        if keep_ids is not None:
            if id_col is not None and id_col in df.columns:
                keep_mask = df.iloc[idx][id_col].astype(str).isin(keep_ids).to_numpy(dtype=bool)
            else:
                keep_mask = np.array([i in keep_ids for i in idx], dtype=bool)
        else:
            keep_mask = np.ones(len(idx), dtype=bool)

        x = log_petro[mask]
        y = log_fhost[mask]
        z_masked = z[mask]
        in_z = (z_masked >= z_range[0]) & (z_masked <= z_range[1])

        keep_in_z = keep_mask & in_z
        keep_out_z = keep_mask & (~in_z)
        cut_in_z = (~keep_mask) & in_z
        cut_out_z = (~keep_mask) & (~in_z)

        ax.scatter(x[keep_in_z], y[keep_in_z], s=s, alpha=alpha, color="tab:blue", marker="o", rasterized=True, label=f"{band}-band")
        ax.scatter(x[cut_in_z], y[cut_in_z], s=s, alpha=alpha, color="tab:orange", marker="D", rasterized=True)
        ax.scatter(
            x[keep_out_z],
            y[keep_out_z],
            s=s,
            alpha=1.0,
            color="tab:blue",
            marker="D",
            linewidths=1.5,
            rasterized=True,
        )
        ax.scatter(
            x[cut_out_z],
            y[cut_out_z],
            s=s,
            alpha=1.0,
            color="tab:orange",
            marker="D",
            linewidths=1.5,
            rasterized=True,
        )

        ax.set_xlabel(r"$\log_{10}(\mathrm{petroRad})$")
        ax.set_ylabel(r"$\log_{10}(f_{\mathrm{host,2500}})$")
        ax.legend(loc="upper right", frameon=False)

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.tight_layout()
    os.makedirs(plot_path, exist_ok=True)
    _save_figure(fig, os.path.join(plot_path, "log_fhost_vs_petrorad_by_band.pdf"), dpi=200, show=show)
    return fig


def plot_dynesty(
    results,
    cosmo_model,
    plot_path="plots/hubble",
    only_sna="",
    speed="",
    show=False,
    use_alpha_lambda_term=None,
    use_redshift_log_f_term=None,
):
    """
    Plot dynesty diagnostics: runplot, traceplot, and cornerpoints using dyplot.
    Saves figures to files with the given basename.
    """

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    option_flags = resolve_model_option_flags(
        cosmo_model,
        np.asarray(results.samples).shape[1],
        only_sna=bool(only_sna),
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    priors, model_labels, model_labels_latex = get_model_params(
        cosmo_model,
        only_sna=bool(only_sna),
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
    )

    # Cornerplot
    fig_corner, axes_corner = dyplot.cornerplot(results, labels=model_labels_latex, quantiles=[0.16, 0.5, 0.84],
                                                 quantiles_2d = [0.393, 0.865, 0.989],
                                                 show_titles=True, title_quantiles=[0.16, 0.5, 0.84],
                                                 color='blue',
                                                 #fig=plt.subplots(1, 1, figsize=(10, 2.5 * len(model_labels))))
    )
    _save_figure(fig_corner, f"{plot_path}/cornerplot_{cosmo_model}_{'sna' if only_sna else 'joint'}_{speed}.pdf", dpi=100, show=show)

    # Traceplot
    fig_trace, axes_trace = dyplot.traceplot(
        results,
        labels=model_labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_quantiles=[0.16, 0.5, 0.84],
    )
    fig_trace.tight_layout(pad=2.0, h_pad=1)

    _save_figure(fig_trace, f"{plot_path}/traceplot_{cosmo_model}_{'sna' if only_sna else 'joint'}_{speed}.pdf", dpi=100, show=show)


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
        
def plot_traces(
    sampler,
    only_sna=False,
    cosmo_model='Flatw0waCDM',
    show=False,
    use_dynesty=False,
    plot_path="plots/hubble",
    use_alpha_lambda_term=None,
    use_redshift_log_f_term=None,
):
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

    option_flags = resolve_model_option_flags(
        cosmo_model,
        samples.shape[-1],
        only_sna=only_sna,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    priors, model_labels, model_labels_latex = get_model_params(
        cosmo_model,
        only_sna=only_sna,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
    )
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
    fig.tight_layout()
    os.makedirs(plot_path, exist_ok=True)
    if only_sna:
        file_path = os.path.join(plot_path, f"traces_{cosmo_model}_sna.pdf")
    else:
        file_path = os.path.join(plot_path, f"traces_{cosmo_model}_agn.pdf")
    _save_figure(fig, file_path, dpi=200, show=show)

    return fig

def plot_posterior_corner(
    flat_samples,
    only_sna=False,
    cosmo_model='Flatw0waCDM',
    show=False,
    plot_path="plots/hubble",
    use_alpha_lambda_term=None,
    use_redshift_log_f_term=None,
):
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
    option_flags = resolve_model_option_flags(
        cosmo_model,
        np.asarray(flat_samples).shape[1],
        only_sna=only_sna,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    priors, model_labels, model_labels_latex = get_model_params(
        cosmo_model,
        only_sna=only_sna,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
    )

    fig = corner.corner(
        flat_samples,
        labels=model_labels,
        truths=None,
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12}
    )

    os.makedirs(plot_path, exist_ok=True)
    if only_sna:
        fig.suptitle("SNIa only", fontsize=16)
        _save_figure(fig, os.path.join(plot_path, f"posterior_{cosmo_model}_sna.pdf"), dpi=200, show=show)
    else:
        fig.suptitle("SNIa + AGN", fontsize=28)
        _save_figure(fig, os.path.join(plot_path, f"posterior_{cosmo_model}_agn.pdf"), dpi=200, show=show)


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
    pad_frac=0.25,
    include_alpha_beta=False,
    use_alpha_lambda_term=None,
    use_redshift_log_f_term=None,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
    from matplotlib.lines import Line2D
    from scipy.stats import gaussian_kde
    from scipy.ndimage import gaussian_filter
    # --- pull model labels from your config ---
    ref_samples = flat_samples_agn if flat_samples_agn is not None else flat_samples_sn
    option_flags = resolve_model_option_flags(
        cosmo_model,
        np.asarray(ref_samples).shape[1],
        only_sna=flat_samples_agn is None,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    _, model_labels, model_labels_latex = get_model_params(
        cosmo_model,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
    )
    idx = {k: i for i, k in enumerate(model_labels)}
    latex = dict(zip(model_labels, model_labels_latex))

    # ---------- helpers ----------
    def _subset(samples, z_pivot, include_alpha_beta=False, include_m0_agn=False):
        X = np.asarray(samples)
        cols = []
        names = []
        lab_latex = []
        units_latex = []

        if include_m0_agn:
            cols.append(X[:, idx["M0_agn"]])
            names.append("M0_agn")
            lab_latex.append(latex["M0_agn"])
            units_latex.append("")

        if include_alpha_beta:
            cols.append(X[:, idx["alpha_agn"]])
            cols.append(X[:, idx["beta_agn"]])
            names += ["alpha_agn", "beta_agn"]
            lab_latex += [latex["alpha_agn"], latex["beta_agn"]]
            units_latex += ["", ""]

        cols.append(X[:, idx["H0"]])
        cols.append(X[:, idx["Om0"]])
        names += ["H0", "Om0"]
        lab_latex += [latex["H0"], latex["Om0"]]
        units_latex += ["(km s$^{-1}$ Mpc$^{-1}$)", ""]

        if cosmo_model == "FlatwpwaCDM":
            wp = X[:, idx["wp"]]
            wa = X[:, idx["wa"]]
            a_p = 1.0 / (1.0 + float(z_pivot))
            w0 = wp - (1.0 - a_p) * wa
            cols += [w0, wa]
            names += ["w0", "wa"]
            lab_latex += [r"$w_0$", latex["wa"]]
            units_latex += ["", ""]
        elif cosmo_model == "Flatw0waCDM":
            cols += [X[:, idx["w0"]], X[:, idx["wa"]]]
            names += ["w0", "wa"]
            lab_latex += [latex["w0"], latex["wa"]]
            units_latex += ["", ""]
        elif cosmo_model == "FlatwCDM":
            cols += [X[:, idx["w0"]]]
            names += ["w0"]
            lab_latex += [latex["w0"]]
            units_latex += [""]
        elif cosmo_model == "FlatLambdaCDM":
            pass
        else:
            raise ValueError(f"Unsupported cosmo_model '{cosmo_model}' for this plot.")

        Y = np.column_stack(cols)
        return Y, names, lab_latex, units_latex
    def _fmt_err(m, lo, hi, latex_label=""):
        nd = 1 if latex_label == latex["H0"] else 2
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
        for i in range(len(levels_12) - 1, -1, -1):
            ax.contourf(
                xx, yy, zz,
                levels=[levels_12[i], zz.max()],
                colors=[color],
                alpha=base_alpha * (i + 1) / len(levels_12)
            )
        ax.contour(xx, yy, zz, levels=levels_12, colors=[color], linewidths=1.2)
        level_3 = _get_density_levels(zz, [0.989])[0]
        ax.contour(xx, yy, zz, levels=[level_3], colors=[color], linewidths=1.4, linestyles='--')

        if set_limits:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
        return xmin, xmax, ymin, ymax

    # --- reduce to plotted params ---
    agn_data, agn_names, labels_latex, units_latex = _subset(
        flat_samples_agn,
        z_pivot_agn,
        include_alpha_beta=include_alpha_beta,
        include_m0_agn=True,
    )

    sna_data = None
    sna_names = None
    if flat_samples_sn is not None and len(flat_samples_sn) > 0:
        sna_data, sna_names, _, _ = _subset(
            flat_samples_sn,
            z_pivot_sna,
            include_alpha_beta=False,
            include_m0_agn=False,
        )

    n_extra = 1 + (2 if include_alpha_beta else 0)   # M0_agn + optional alpha,beta
    n_params = agn_data.shape[1]
    fig, axes = plt.subplots(n_params, n_params, figsize=(2.3 * n_params, 2.3 * n_params))
    if n_params == 1:
        axes = np.array([[axes]])

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            ax.tick_params(direction='in')

            if i < j:
                ax.axis("off")
                continue

            i_sn = i - n_extra
            j_sn = j - n_extra
            has_sn_here = (sna_data is not None) and (i_sn >= 0) and (j_sn >= 0)

            if i == j:
                xs, ys, (xmin, xmax) = _kde1d_grid(agn_data[:, i])
                ax.plot(xs, ys, color="k", lw=1.8)

                if has_sn_here:
                    xs_b, ys_b, (xmin_b, xmax_b) = _kde1d_grid(sna_data[:, i_sn])
                    ax.plot(xs_b, ys_b, color="dodgerblue", lw=1.8)
                    ax.set_xlim(min(xmin, xmin_b), max(xmax, xmax_b))
                else:
                    ax.set_xlim(xmin, xmax)

                figt = ax.figure

                # SN Ia on top
                if has_sn_here:
                    median, err, err_lower, err_upper = sym_percentile(sna_data[:, i_sn])
                    if sna_names[i_sn] in ['w0']:
                        txt_value = format_result_errors(
                            median, err_lower=err_lower, err_upper=err_upper, nd=2
                        )
                    elif sna_names[i_sn] in ['wa']:
                        txt_value = format_result_errors(
                            median, err_lower=err_lower, err_upper=err_upper, nd=1
                        )
                    else:
                        txt_value = format_result_errors(median, err=err)

                    txt_blue = rf"{labels_latex[i]} = ${txt_value}$ {units_latex[i]}"
                    off_blue = mtransforms.ScaledTranslation(0, 15 / 72., figt.dpi_scale_trans)
                    ax.text(
                        0.02, 1.0, txt_blue,
                        transform=ax.transAxes + off_blue,
                        ha="left", va="bottom", color="dodgerblue",
                        fontsize=11, clip_on=False
                    )

                # SN Ia + AGN on bottom
                median, err, err_lower, err_upper = sym_percentile(agn_data[:, i])
                if agn_names[i] in ['w0']:
                    txt_value = format_result_errors(
                        median, err_lower=err_lower, err_upper=err_upper, nd=2
                    )
                elif agn_names[i] in ['wa']:
                    txt_value = format_result_errors(
                        median, err_lower=err_lower, err_upper=err_upper, nd=1
                    )
                else:
                    txt_value = format_result_errors(median, err=err)

                txt_black = rf"{labels_latex[i]} = ${txt_value}$ {units_latex[i]}"
                off_blk = mtransforms.ScaledTranslation(0, 2 / 72., figt.dpi_scale_trans)
                ax.text(
                    0.02, 1.0, txt_black,
                    transform=ax.transAxes + off_blk,
                    ha="left", va="bottom", color="k",
                    fontsize=11, clip_on=False
                )

            else:
                lims = []
                if has_sn_here:
                    lims.append(_filled_kde_with_3sigma(
                        ax, sna_data[:, j_sn], sna_data[:, i_sn], "dodgerblue",
                        base_alpha=0.4, set_limits=False
                    ))

                lims.append(_filled_kde_with_3sigma(
                    ax, agn_data[:, j], agn_data[:, i], "k",
                    base_alpha=0.4, set_limits=False
                ))

                xmin = min(l[0] for l in lims)
                xmax = max(l[1] for l in lims)
                ymin = min(l[2] for l in lims)
                ymax = max(l[3] for l in lims)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
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
        legend.append(Line2D([0], [0], color="dodgerblue", lw=6, label="SN Ia"))
    legend.append(Line2D([0], [0], color="k", lw=6, label="SN Ia + AGN"))
    fig.legend(handles=legend, bbox_to_anchor=(0.99, 0.92), loc="upper right",
               fontsize=18, frameon=False, markerscale=1.5)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.9,
                        wspace=0.05, hspace=0.05)

    os.makedirs(plot_path, exist_ok=True)
    _save_figure(
        fig,
        os.path.join(plot_path, f"cosmo_corner_{cosmo_model}_{'alphabeta' if include_alpha_beta else 'noalphabeta'}.pdf"),
        dpi=600,
        show=show,
    )

def _weighted_bin_stats(z, y, yerr, bins, *, min_count=3, center='mid', plot_path=None):
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
                cosmo_model_samples={}, residuals_sigma_clip=None, df_calibrators=None, z_range=(0.44, 3.16),
                dmi_values=None, dmi_sigma=None, dmi_selection_sigma=None,
                use_alpha_lambda_term=None, use_redshift_log_f_term=None):
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
    option_flags = resolve_model_option_flags(
        cosmo_model,
        np.asarray(flat_samples).shape[1],
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    _, model_labels, _ = get_model_params(
        cosmo_model,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
    )
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
        agn_params_arr = agn_model_pack_params(sample_params, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"])
        agn_obs_arr, agn_err_arr, agn_pivot_arr = agn_model_pack_obs(
            df_agn, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"]
        )

        predicted_M2500 = M_model_agn(
            agn_params_arr, agn_obs_arr, agn_pivot_arr, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"]
        )
        predicted_M2500_err = M_model_agn_err(
            agn_params_arr, agn_obs_arr, agn_err_arr, agn_pivot_arr, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"]
        )
        mu_pred_samples.append(m_obs - predicted_M2500)
    mu_pred_samples = np.array(mu_pred_samples)

    # De-bias (assumes your make_dm_function clips to grid, no extrapolation)
    if debias:
        mu_pred_samples -= _resolve_debias_values(
            df_agn,
            dm_interp=dm_interp,
            dmi_values=dmi_values,
        )

    mu_pred_median = np.percentile(mu_pred_samples, 50, axis=0)
    mu_pred_16th   = np.percentile(mu_pred_samples, 16, axis=0)
    mu_pred_84th   = np.percentile(mu_pred_samples, 84, axis=0)

    # Per-object uncertainty (for yerr)
    agn_params_arr = agn_model_pack_params(results, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"])
    agn_obs_arr, agn_err_arr, agn_pivot_arr = agn_model_pack_obs(
        df_agn, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"]
    )
    predicted_M2500 = M_model_agn(
        agn_params_arr, agn_obs_arr, agn_pivot_arr, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"]
    )
    predicted_M2500_err = M_model_agn_err(
        agn_params_arr, agn_obs_arr, agn_err_arr, agn_pivot_arr, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"]
    )
    req_params_local, _, req_errs_local = get_agn_model_spec(
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"]
    )
    pidx_local = {k: i for i, k in enumerate(req_params_local)}
    eidx_local = {k: i for i, k in enumerate(req_errs_local)}
    alpha_agn = agn_params_arr[pidx_local["alpha_agn"]]
    beta_agn = agn_params_arr[pidx_local["beta_agn"]]
    log_sigma_uv_std_psd = agn_err_arr[eidx_local["log_sigma_uv_std_psd"]]
    log_tau_uv_rf_std_psd = agn_err_arr[eidx_local["log_tau_uv_rf_std_psd"]]
    log_sigma_uv_log_tau_uv_rf_cov_psd = agn_err_arr[eidx_local["log_sigma_uv_log_tau_uv_rf_cov_psd"]]
    pred_m2500_sigma_var = (alpha_agn * log_sigma_uv_std_psd) ** 2
    pred_m2500_tau_var = (beta_agn * log_tau_uv_rf_std_psd) ** 2
    pred_m2500_cov_var = 2 * alpha_agn * beta_agn * log_sigma_uv_log_tau_uv_rf_cov_psd
    pred_m2500_alpha_lambda_var = np.zeros_like(pred_m2500_sigma_var)
    if option_flags["use_alpha_lambda_term"]:
        gamma_alpha_lambda = agn_params_arr[pidx_local[AGN_ALPHA_LAMBDA_PARAM]]
        alpha_lambda_err = agn_err_arr[eidx_local[AGN_ALPHA_LAMBDA_ERR]]
        pred_m2500_alpha_lambda_var = (gamma_alpha_lambda * alpha_lambda_err) ** 2

    cosmo = get_cosmo(cosmo_model, results, z_pivot_agn)
    sigma_lens = sigma_lens_from_dc(df_agn['z'].values, cosmo)

    apparent_mag_err = df_agn['apparent_mag_2500_err'].values
    z_err = sigma_mu_from_z_err(df_agn["z"].values, df_agn["z_err"].values, cosmo)
    m_app_var = apparent_mag_err**2
    lens_var = sigma_lens**2
    z_var = z_err**2
    pred_m2500_var = predicted_M2500_err**2

    mu_pred_std = np.sqrt(
        m_app_var +
        #(0.055 * df_agn["z"].values)**2 +
        lens_var +
        z_var +
        pred_m2500_var
    )

    log_f_eff = evaluate_log_f(
        results,
        df_agn["z"].values,
        z_pivot=z_pivot_agn,
        use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
    )
    intrinsic_scatter = np.exp(log_f_eff)
    intrinsic_var = intrinsic_scatter**2
    mu_pred_std_with_scatter = np.sqrt(mu_pred_std**2 + intrinsic_var)
    
    sigma_dmi = None
    if dmi_sigma is not None:
        sigma_dmi = np.asarray(dmi_sigma, dtype=float)
        if sigma_dmi.shape != mu_pred_std.shape:
            raise ValueError(
                f"dmi_sigma has shape {sigma_dmi.shape}, but expected {mu_pred_std.shape}."
            )

    sigma_sel = None
    sigma_sel_floor_mag = 0.05
    n_sigma_sel_floored = 0
    if dmi_selection_sigma is not None:
        sigma_sel = np.asarray(dmi_selection_sigma, dtype=float)
        if sigma_sel.shape != mu_pred_std.shape:
            raise ValueError(
                f"dmi_selection_sigma has shape {sigma_sel.shape}, but expected {mu_pred_std.shape}."
            )
        sigma_sel_valid = np.isfinite(sigma_sel) & (sigma_sel > 0.0)
        n_sigma_sel_floored = int(
            np.count_nonzero(sigma_sel_valid & (sigma_sel < sigma_sel_floor_mag))
        )
        sigma_sel = np.where(
            sigma_sel_valid,
            np.maximum(sigma_sel, sigma_sel_floor_mag),
            np.nan,
        )

    # Residuals (vs. median μ_model)
    mu_interp = np.interp(df_agn["z"].values, z_grid, mu_model_median)
    residuals = mu_pred_median - mu_interp
    residuals_err = mu_pred_std_with_scatter
    if debias and sigma_sel is not None:
        use_sigma_sel = np.isfinite(sigma_sel) & (sigma_sel > 0.0)
        residuals_err = np.where(use_sigma_sel, sigma_sel, residuals_err)

    mu_zscore = np.abs(residuals) / residuals_err

    # ----------------- BINNING -----------------
    # Linear-z bins for MAIN & RESIDUALS panel
    bins_linear = np.arange(0.32, np.max(df_agn["z"].values), 0.2)
    print("Using linear-z bins:", bins_linear)
    z_lin_scatter, mu_lin_mean_scatter, mu_lin_sem_scatter, n_lin = _weighted_bin_stats(
        df_agn["z"].values, mu_pred_median, residuals_err, bins_linear
    )
    

    # NEW: binned residuals (linear-z), used in residual panel
    # z_res_lin_scatter, resid_lin_mean_scatter, resid_lin_sem_scatter, n_res = _weighted_bin_stats(
    #     df_agn["z"].values, residuals, residuals_err, bins_linear
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
        df_agn["z"].values, mu_pred_median, residuals_err, bins_log)

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

    # Filled circles vs filled diamonds by z (both main and inset)
    mask_in  = df_agn["z"].between(z_range[0], z_range[1])
    mask_out = ~mask_in

    # AGN (inside)
    inset_ax.errorbar(
        df_agn["z"][mask_in], mu_pred_median[mask_in], yerr=residuals_err[mask_in],
        fmt='o', linestyle='none', markersize=2,
        mfc="black", mec="none",
        ecolor="#666666", elinewidth=0.8,
        alpha=0.7, zorder=1, label="AGN"
    )
    # AGN (outside, filled diamond)
    inset_ax.errorbar(
        df_agn["z"][mask_out], mu_pred_median[mask_out], yerr=residuals_err[mask_out],
        fmt='D', linestyle='none', markersize=2, mfc="black", mec="none", alpha=0.70,
        ecolor="#666666", elinewidth=0.8, zorder=1
    )

    # INSET: log-binned AGN
    if show_binned_agn:
        mask_in  = (z_range[0] < z_log) & (z_log < z_range[1])
        mask_out = ~mask_in
        # binned (inside)
        inset_ax.errorbar(
            z_log[mask_in], mu_log_mean[mask_in], yerr=mu_log_sem[mask_in],
            fmt='o', linestyle='none',
            markersize=4, mfc='red', mec='none',
            ecolor='red', elinewidth=2.2, capsize=3.5,
            alpha=0.98, zorder=14, label="AGN (z-binned, log)"
        )
        # binned (outside, filled diamond)
        inset_ax.errorbar(
            z_log[mask_out], mu_log_mean[mask_out], yerr=mu_log_sem[mask_out],
            fmt='D', linestyle='none',
            markersize=4, mfc='red', mec='none',
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
    mask_in  = df_agn["z"].between(z_range[0], z_range[1])
    mask_out = ~mask_in
    # AGN (inside)
    for i in np.where(mask_in)[0]:
        ax.errorbar(
            df_agn["z"].iloc[i], mu_pred_median[i], yerr=residuals_err[i],
            fmt='o', linestyle='none', markersize=3,
            mec="none",
            mfc=(0, 0, 0, 0.3),
            ecolor=(0.2, 0.2, 0.2, 0.1), elinewidth=0.8,
            capsize=2, capthick=0.8,
            zorder=0, label="AGN" if i == np.where(mask_in)[0][0] else None
        )

    # AGN (outside, filled diamond)
    for i in np.where(mask_out)[0]:
        ax.errorbar(
            df_agn["z"].iloc[i], mu_pred_median[i], yerr=residuals_err[i],
            fmt='D', linestyle='none', markersize=3, mfc=(0, 0, 0, 0.4),
            mec="none",
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
        mask_in  = (z_range[0] < z_lin_scatter) & (z_lin_scatter < z_range[1])
        print("Plotting binned AGN (linear z) at:", z_lin_scatter)
        print("\tmask_in:", mask_in)
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
        # binned (outside, filled diamond)
        ax.errorbar(
            z_lin_scatter[mask_out], mu_lin_mean_scatter[mask_out], yerr=mu_lin_sem_scatter[mask_out],
            fmt='D', linestyle='none',
            markersize=5, mfc='red', mec='none',
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
        agn_params_arr = agn_model_pack_params(results, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"])
        agn_obs_med = {key: float(np.median(df_agn[key].values)) * np.ones_like(z_grid) for key in agn_model_req_obs + agn_model_req_errs}
        if option_flags["use_alpha_lambda_term"]:
            agn_obs_med["alpha_lambda"] = float(np.median(df_agn["alpha_lambda"].values)) * np.ones_like(z_grid)
            agn_obs_med["alpha_lambda_err"] = float(np.median(df_agn["alpha_lambda_err"].values)) * np.ones_like(z_grid)
        agn_obs_arr, agn_err_arr, agn_pivot_arr = agn_model_pack_obs(
            agn_obs_med, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"]
        )

        M_med_grid = np.median([
            M_model_agn(agn_params_arr, agn_obs_arr, agn_pivot_arr, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"])
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
        option_flags_other = resolve_model_option_flags(
            cosmo_model_other, np.asarray(cosmo_model_samples_other).shape[1]
        )
        _, model_labels_other, _ = get_model_params(
            cosmo_model_other,
            use_alpha_lambda_term=option_flags_other["use_alpha_lambda_term"],
            use_redshift_log_f_term=option_flags_other["use_redshift_log_f_term"],
        )
        model_label_latex_other = cosmo_model_label_latex(cosmo_model_other)
        results_other = {key: np.median(cosmo_model_samples_other[:, i]) for i, key in enumerate(model_labels_other)}

        mu_model_other = _mu_model(cosmo_model_other, results_other,   z_grid, z_pivot_agn)
        ax.plot(z_grid, mu_model_other, lw=1.2, color=colors[cosmo_model_other], ls=line_styles[cosmo_model_other], alpha=1.0, 
                        label=model_label_latex_other, zorder=6)
        inset_ax.plot(z_grid, mu_model_other, lw=1.2, color=colors[cosmo_model_other], ls=line_styles[cosmo_model_other], alpha=1.0, zorder=6)

    # Flat ΛCDM
    mu_conc = Planck18.distmod(z_grid).value
    ax.plot(z_grid, mu_conc, color="#F0B000", lw=1.2, ls='--', zorder=5, alpha=1.0, label=r"flat $\Lambda$CDM (Planck 2018)")

    # Labels
    ax.set_ylabel(r"$\mu$ (mag)")
    ax.set_xlabel(r"$z$")

    # ---------- Residuals panel ----------
    if show_residuals:
        # Zero line
        ax_resid.axhline(0.0, color="m", lw=2.2, zorder=1)

        # NEW: binned residuals in red (points + thin connecting line)
        if z_res_lin_scatter.size:
            mask_in  = (z_range[0] < z_res_lin_scatter) & (z_res_lin_scatter < z_range[1])
            mask_out = ~mask_in
            ax_resid.errorbar(
                z_res_lin_scatter[mask_in], resid_lin_mean_scatter[mask_in], yerr=resid_lin_sem_scatter[mask_in],
                fmt='o', linestyle='none', markersize=6,
                mfc='red', mec='none', ecolor='red', elinewidth=2.0, capsize=3.0,
                alpha=0.98, zorder=15, label="Binned AGN residuals (w/ scatter)"
            )
            ax_resid.errorbar(
                z_res_lin_scatter[mask_out], resid_lin_mean_scatter[mask_out], yerr=resid_lin_sem_scatter[mask_out],
                fmt='D', linestyle='none', markersize=6,
                mfc='red', mec='none', ecolor='red', elinewidth=2.0, capsize=3.0,
                alpha=0.98, zorder=15
            )


        for cosmo_model_other, cosmo_model_samples_other in cosmo_model_samples.items():
            option_flags_other = resolve_model_option_flags(
                cosmo_model_other, np.asarray(cosmo_model_samples_other).shape[1]
            )
            _, model_labels_other, _ = get_model_params(
                cosmo_model_other,
                use_alpha_lambda_term=option_flags_other["use_alpha_lambda_term"],
                use_redshift_log_f_term=option_flags_other["use_redshift_log_f_term"],
            )
            z_grid_fine = np.linspace(1e-4, 5.2, 500)
            results_other = {key: np.median(cosmo_model_samples_other[:, i]) for i, key in enumerate(model_labels_other)}

            mu_model_other = _mu_model(cosmo_model_other, results_other,   z_grid, z_pivot_agn)
            mu_model = _mu_model(cosmo_model, results, z_grid, z_pivot_agn)
            ax_resid.plot(z_grid_fine, mu_model_other - mu_model, lw=2.2, color=colors[cosmo_model_other], ls=line_styles[cosmo_model_other], 
                          alpha=1.0, label=fr"{cosmo_model_other} $\Delta$μ")
            
        # Planck 2018 ΛCDM
        mu_model_1 = _mu_model(cosmo_model, results, z_grid, z_pivot_agn)

        mu_conc = Planck18.distmod(z_grid).value
        #ax.plot(z_grid, mu_conc, color="#F0B000", lw=1.2, ls='--', zorder=5, alpha=1.0, label="flat $\Lambda$CDM (Planck 2018)")
        ax_resid.plot(z_grid, mu_conc - mu_model_1, lw=2.2, color="#F0B000", ls='--', alpha=1.0,)


        ax_resid.set_ylabel(r"$\Delta\mu$ (mag)")
        ax_resid.set_xlabel(r"$z$")
        chi2_red = np.nan
        chi2_red_no_logf = np.nan
        chi2_red_full_plus_sigma_dmi = np.nan
        chi2_red_no_logf_plus_sigma_dmi = np.nan
        if residuals.size:
            chi2_red, _ = reduced_chi_squared(
                residuals,
                residuals_err,
                n_params=len(model_labels) - 1,
            )
            if debias:
                chi2_red_no_logf, _ = reduced_chi_squared(
                    residuals,
                    mu_pred_std,
                    n_params=len(model_labels) - 1,
                )
                if sigma_dmi is not None:
                    chi2_red_full_plus_sigma_dmi, _ = reduced_chi_squared(
                        residuals,
                        residuals_err,
                        extra_err=sigma_dmi,
                        n_params=len(model_labels) - 1,
                    )
                    chi2_red_no_logf_plus_sigma_dmi, _ = reduced_chi_squared(
                        residuals,
                        mu_pred_std,
                        extra_err=sigma_dmi,
                        n_params=len(model_labels) - 1,
                    )
        if debias and np.isfinite(chi2_red):
            chi2_lines = [
                rf"full: {chi2_red:.2f}",
                rf"no log $f$: {chi2_red_no_logf:.2f}" if np.isfinite(chi2_red_no_logf) else r"no log $f$: nan",
                rf"full $+\sigma_{{\rm dmi}}$: {chi2_red_full_plus_sigma_dmi:.2f}" if np.isfinite(chi2_red_full_plus_sigma_dmi) else r"full $+\sigma_{\rm dmi}$: nan",
                rf"no log $f$ $+\sigma_{{\rm dmi}}$: {chi2_red_no_logf_plus_sigma_dmi:.2f} [rec]" if np.isfinite(chi2_red_no_logf_plus_sigma_dmi) else r"no log $f$ $+\sigma_{\rm dmi}$: nan [rec]",
            ]
            ax_resid.text(
                0.98,
                0.96,
                "$\\chi^2_\\nu$ debiased\n" + "\n".join(chi2_lines),
                transform=ax_resid.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                linespacing=1.15,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
            )
        elif np.isfinite(chi2_red):
            ax_resid.text(
                0.98,
                0.96,
                rf"$\chi^2_\nu = {chi2_red:.2f}$",
                transform=ax_resid.transAxes,
                ha="right",
                va="top",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
            )
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
        agn_params_arr_show = agn_model_pack_params(results, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"])
        obs_show, err_show, piv_show = agn_model_pack_obs(ds, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"])
        pred_M_show = M_model_agn(
            agn_params_arr_show, obs_show, piv_show, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"]
        )
        pred_M_err_show = M_model_agn_err(
            agn_params_arr_show, obs_show, err_show, piv_show, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"]
        )

        # Distance modulus prediction: mu = m_2500 - M_2500
        m_show = ds['apparent_mag_2500'].values
        mu_show = ds['mu'].values
        # Uncertainties for SHOW (match main formula)
        z_show     = ds['z'].values

        mu_show_std = ds['mu_err'].values
        # Optionally include intrinsic scatter (used in residuals if desired)
        log_f_eff_show = evaluate_log_f(
            results,
            z_show,
            z_pivot=z_pivot_agn,
            use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
        )
        mu_show_std_with_scatter = np.sqrt(mu_show_std**2 + np.exp(log_f_eff_show) ** 2)

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
    filename = "hubble_diagram_debiased.pdf" if debias else "hubble_diagram.pdf"
    _save_figure(fig, os.path.join(plot_path, filename), dpi=600, show=show)

    diagnostics_path = os.path.join(plot_path, "diagnostics")
    os.makedirs(diagnostics_path, exist_ok=True)

    sigma_dmi_var = np.zeros_like(mu_pred_std)
    if sigma_dmi is not None:
        sigma_dmi_var = np.square(sigma_dmi)
    sigma_sel_var = np.full_like(mu_pred_std, np.nan, dtype=float)
    if sigma_sel is not None:
        sigma_sel_var = np.square(sigma_sel)

    total_var = m_app_var + lens_var + z_var + pred_m2500_var + intrinsic_var
    total_var_plus_sigma_dmi = total_var + sigma_dmi_var
    total_var_no_logf = m_app_var + lens_var + z_var + pred_m2500_var
    total_var_no_logf_plus_sigma_dmi = total_var_no_logf + sigma_dmi_var
    error_budget_mask = np.isfinite(total_var) & np.isfinite(residuals) & (total_var > 0)
    sigma_dmi_mask = error_budget_mask & np.isfinite(sigma_dmi) if sigma_dmi is not None else None
    sigma_sel_mask = (
        error_budget_mask & np.isfinite(sigma_sel) & (sigma_sel > 0.0)
        if sigma_sel is not None
        else None
    )
    if np.any(error_budget_mask):
        def _chi2_red_from_var(var_term):
            mask = error_budget_mask & np.isfinite(var_term) & (var_term > 0)
            if not np.any(mask):
                return np.nan
            dof_local = max(int(np.count_nonzero(mask)) - 1, 1)
            return float(np.sum((residuals[mask] ** 2) / var_term[mask]) / dof_local)

        def _median_fraction(component_var):
            mask = error_budget_mask & np.isfinite(component_var)
            if not np.any(mask):
                return np.nan
            return float(np.median(component_var[mask] / total_var[mask]))

        residual_rms = float(np.sqrt(np.mean(residuals[error_budget_mask] ** 2)))
        budget_rows = [
            {"metric": "n_objects", "value": float(np.count_nonzero(error_budget_mask))},
            {"metric": "residual_rms_mag", "value": residual_rms},
            {"metric": "median_abs_residual_mag", "value": float(np.median(np.abs(residuals[error_budget_mask])))},
            {"metric": "chi2_red_full", "value": _chi2_red_from_var(total_var)},
            {"metric": "chi2_red_no_intrinsic_scatter", "value": _chi2_red_from_var(total_var_no_logf)},
            {"metric": "chi2_red_full_plus_sigma_dmi", "value": _chi2_red_from_var(total_var_plus_sigma_dmi)},
            {"metric": "chi2_red_no_intrinsic_scatter_plus_sigma_dmi", "value": _chi2_red_from_var(total_var_no_logf_plus_sigma_dmi)},
            {"metric": "chi2_red_no_predicted_M2500_err", "value": _chi2_red_from_var(m_app_var + lens_var + z_var + intrinsic_var)},
            {"metric": "chi2_red_no_sigma_lens", "value": _chi2_red_from_var(m_app_var + z_var + pred_m2500_var + intrinsic_var)},
            {"metric": "chi2_red_no_apparent_mag_err", "value": _chi2_red_from_var(lens_var + z_var + pred_m2500_var + intrinsic_var)},
            {"metric": "chi2_red_no_z_err", "value": _chi2_red_from_var(m_app_var + lens_var + pred_m2500_var + intrinsic_var)},
            {"metric": "chi2_red_sigma_sel", "value": _chi2_red_from_var(sigma_sel_var)},
            {"metric": "median_apparent_mag_err_mag", "value": float(np.median(apparent_mag_err[error_budget_mask]))},
            {"metric": "median_sigma_lens_mag", "value": float(np.median(sigma_lens[error_budget_mask]))},
            {"metric": "median_z_err_mag", "value": float(np.median(z_err[error_budget_mask]))},
            {"metric": "median_predicted_M2500_err_mag", "value": float(np.median(predicted_M2500_err[error_budget_mask]))},
            {"metric": "median_predicted_M2500_sigma_term_mag", "value": float(np.median(np.sqrt(np.clip(pred_m2500_sigma_var[error_budget_mask], 0.0, None))))},
            {"metric": "median_predicted_M2500_tau_term_mag", "value": float(np.median(np.sqrt(np.clip(pred_m2500_tau_var[error_budget_mask], 0.0, None))))},
            {"metric": "median_predicted_M2500_cov_term_mag_signed", "value": float(np.median(np.sign(pred_m2500_cov_var[error_budget_mask]) * np.sqrt(np.abs(pred_m2500_cov_var[error_budget_mask]))))},
            {"metric": "median_predicted_M2500_alpha_lambda_term_mag", "value": float(np.median(np.sqrt(np.clip(pred_m2500_alpha_lambda_var[error_budget_mask], 0.0, None))))},
            {"metric": "median_mu_pred_std_mag", "value": float(np.median(mu_pred_std[error_budget_mask]))},
            {"metric": "median_intrinsic_scatter_mag", "value": float(np.median(intrinsic_scatter[error_budget_mask]))},
            {"metric": "median_mu_pred_std_with_scatter_mag", "value": float(np.median(mu_pred_std_with_scatter[error_budget_mask]))},
            {"metric": "median_sigma_dmi_mag", "value": float(np.median(sigma_dmi[sigma_dmi_mask])) if sigma_dmi_mask is not None and np.any(sigma_dmi_mask) else np.nan},
            {"metric": "median_sigma_sel_mag", "value": float(np.median(sigma_sel[sigma_sel_mask])) if sigma_sel_mask is not None and np.any(sigma_sel_mask) else np.nan},
            {"metric": "sigma_sel_floor_mag", "value": float(sigma_sel_floor_mag)},
            {"metric": "n_sigma_sel_floored", "value": float(n_sigma_sel_floored)},
            {"metric": "median_var_fraction_apparent_mag_err", "value": _median_fraction(m_app_var)},
            {"metric": "median_var_fraction_sigma_lens", "value": _median_fraction(lens_var)},
            {"metric": "median_var_fraction_z_err", "value": _median_fraction(z_var)},
            {"metric": "median_var_fraction_predicted_M2500_err", "value": _median_fraction(pred_m2500_var)},
            {"metric": "median_var_fraction_intrinsic_scatter", "value": _median_fraction(intrinsic_var)},
            {"metric": "median_var_fraction_sigma_dmi", "value": _median_fraction(sigma_dmi_var)},
            {"metric": "median_var_fraction_predicted_M2500_sigma_term", "value": _median_fraction(pred_m2500_sigma_var)},
            {"metric": "median_var_fraction_predicted_M2500_tau_term", "value": _median_fraction(pred_m2500_tau_var)},
            {"metric": "median_var_fraction_predicted_M2500_cov_term", "value": _median_fraction(pred_m2500_cov_var)},
            {"metric": "median_var_fraction_predicted_M2500_alpha_lambda_term", "value": _median_fraction(pred_m2500_alpha_lambda_var)},
        ]
        budget_suffix = "_debiased" if debias else ""
        budget_summary_path = os.path.join(diagnostics_path, f"hubble_error_budget_summary{budget_suffix}.csv")
        pd.DataFrame(budget_rows).to_csv(budget_summary_path, index=False)

        per_object_budget_df = df_agn.copy()
        per_object_budget_df["residuals"] = residuals
        per_object_budget_df["residuals_err"] = residuals_err
        per_object_budget_df["apparent_mag_2500_err_term"] = apparent_mag_err
        per_object_budget_df["sigma_lens_term"] = sigma_lens
        per_object_budget_df["z_err_term"] = z_err
        per_object_budget_df["predicted_M2500_err_term"] = predicted_M2500_err
        per_object_budget_df["predicted_M2500_sigma_term"] = np.sqrt(np.clip(pred_m2500_sigma_var, 0.0, None))
        per_object_budget_df["predicted_M2500_tau_term"] = np.sqrt(np.clip(pred_m2500_tau_var, 0.0, None))
        per_object_budget_df["predicted_M2500_cov_term_signed"] = np.sign(pred_m2500_cov_var) * np.sqrt(np.abs(pred_m2500_cov_var))
        per_object_budget_df["predicted_M2500_alpha_lambda_term"] = np.sqrt(np.clip(pred_m2500_alpha_lambda_var, 0.0, None))
        per_object_budget_df["intrinsic_scatter_term"] = intrinsic_scatter
        per_object_budget_df["sigma_dmi_term"] = sigma_dmi if sigma_dmi is not None else np.nan
        per_object_budget_df["sigma_sel_term"] = sigma_sel if sigma_sel is not None else np.nan
        per_object_budget_df["mu_pred_std_no_intrinsic"] = mu_pred_std
        per_object_budget_df["mu_pred_std_with_intrinsic"] = mu_pred_std_with_scatter
        per_object_budget_df["mu_pred_std_with_intrinsic_and_sigma_dmi"] = np.sqrt(total_var_plus_sigma_dmi)
        per_object_budget_df["mu_pred_std_no_intrinsic_and_sigma_dmi"] = np.sqrt(total_var_no_logf_plus_sigma_dmi)
        per_object_budget_fields = [
            "object_id",
            "z",
            "residuals",
            "residuals_err",
            "apparent_mag_2500_err_term",
            "sigma_lens_term",
            "z_err_term",
            "predicted_M2500_err_term",
            "predicted_M2500_sigma_term",
            "predicted_M2500_tau_term",
            "predicted_M2500_cov_term_signed",
            "predicted_M2500_alpha_lambda_term",
            "intrinsic_scatter_term",
            "sigma_dmi_term",
            "sigma_sel_term",
            "mu_pred_std_no_intrinsic",
            "mu_pred_std_with_intrinsic",
            "mu_pred_std_no_intrinsic_and_sigma_dmi",
            "mu_pred_std_with_intrinsic_and_sigma_dmi",
        ]
        per_object_budget_fields = [field for field in per_object_budget_fields if field in per_object_budget_df.columns]
        budget_per_object_path = os.path.join(diagnostics_path, f"hubble_error_budget_per_object{budget_suffix}.csv")
        per_object_budget_df[per_object_budget_fields].to_csv(budget_per_object_path, index=False)

        print(f"Saved Hubble error-budget summary to {budget_summary_path}")
        print(f"Saved per-object Hubble error budget to {budget_per_object_path}")
        budget_print = (
            "Hubble error budget:"
            f" chi2_full={_chi2_red_from_var(total_var):.3f},"
            f" chi2_no_intrinsic={_chi2_red_from_var(total_var_no_logf):.3f},"
            f" chi2_full_plus_sigma_dmi={_chi2_red_from_var(total_var_plus_sigma_dmi):.3f},"
            f" chi2_no_intrinsic_plus_sigma_dmi={_chi2_red_from_var(total_var_no_logf_plus_sigma_dmi):.3f},"
            f" chi2_no_predM={_chi2_red_from_var(m_app_var + lens_var + z_var + intrinsic_var):.3f},"
            f" median_predM_err={float(np.median(predicted_M2500_err[error_budget_mask])):.3f} mag,"
            f" median_predM_sigma={float(np.median(np.sqrt(np.clip(pred_m2500_sigma_var[error_budget_mask], 0.0, None)))):.3f} mag,"
            f" median_predM_tau={float(np.median(np.sqrt(np.clip(pred_m2500_tau_var[error_budget_mask], 0.0, None)))):.3f} mag,"
            f" median_intrinsic={float(np.median(intrinsic_scatter[error_budget_mask])):.3f} mag,"
        )
        if sigma_dmi_mask is not None and np.any(sigma_dmi_mask):
            budget_print += f" median_sigma_dmi={float(np.median(sigma_dmi[sigma_dmi_mask])):.3f} mag,"
        budget_print += f" residual_rms={residual_rms:.3f} mag"
        print(budget_print)

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
        residuals_df["sigma_dmi"] = sigma_dmi if sigma_dmi is not None else np.nan
        residuals_df["mu_pred_std_with_scatter_and_sigma_dmi"] = np.sqrt(total_var_plus_sigma_dmi)
        residuals_df["mu_pred_std_and_sigma_dmi"] = np.sqrt(total_var_no_logf_plus_sigma_dmi)
        residuals_df["mu_zscore"] = mu_zscore
        fields = ['object_id', 'apparent_mag_2500', 'f_host_2500', 'ra', 'dec', 
                  'mu_pred_median', 'mu_pred_std', 'mu_pred_std_with_scatter',
                  'sigma_dmi', 'mu_pred_std_with_scatter_and_sigma_dmi', 'mu_pred_std_and_sigma_dmi',
                  'z', 'wrms', 'sdss_name', 'residuals', 'mu_zscore']
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

    return residuals, residuals_err, mu_pred_median, mu_pred_std, residuals_err


def plot_hubble_residual_normality(
    residuals,
    residuals_err,
    *,
    plot_path="plots/hubble",
    show=False,
    filename="hubble_residual_normality.pdf",
):
    """Plot histogram and normal Q-Q diagnostics for normalized Hubble residuals."""

    residuals = np.asarray(residuals, dtype=float)
    residuals_err = np.asarray(residuals_err, dtype=float)
    mask = np.isfinite(residuals) & np.isfinite(residuals_err) & (residuals_err > 0.0)
    z_resid = residuals[mask] / residuals_err[mask]

    fig, (ax_hist, ax_qq) = plt.subplots(1, 2, figsize=(10.8, 4.8))
    if z_resid.size == 0:
        ax_hist.text(
            0.5,
            0.5,
            "No finite normalized residuals",
            ha="center",
            va="center",
            transform=ax_hist.transAxes,
        )
        ax_qq.set_axis_off()
    else:
        z_mean = float(np.mean(z_resid))
        z_std = float(np.std(z_resid, ddof=1)) if z_resid.size > 1 else np.nan
        z_skew = float(skew(z_resid, bias=False)) if z_resid.size > 2 else np.nan
        z_kurt = float(kurtosis(z_resid, fisher=True, bias=False)) if z_resid.size > 3 else np.nan
        k2_stat = np.nan
        p_norm = np.nan
        if z_resid.size >= 8:
            try:
                k2_stat, p_norm = normaltest(np.asarray(z_resid, dtype=np.float64))
                k2_stat = float(k2_stat)
                p_norm = float(p_norm)
            except Exception as exc:
                print(
                    "[WARNING] Skipping normaltest() in Hubble residual "
                    f"Gaussianity diagnostic: {exc}"
                )

        x_grid = np.linspace(-5.0, 5.0, 512)
        ax_hist.hist(
            z_resid,
            bins=40,
            density=True,
            color="0.35",
            alpha=0.35,
            edgecolor="white",
        )
        ax_hist.plot(
            x_grid,
            norm.pdf(x_grid, loc=0.0, scale=1.0),
            color="black",
            lw=1.8,
            label=r"$\mathcal{N}(0,1)$",
        )
        ax_hist.axvline(0.0, color="black", ls="--", lw=1.0, alpha=0.8)
        ax_hist.set_xlim(-5.0, 5.0)
        ax_hist.set_xlabel(r"$\Delta\mu / \sigma_{\Delta\mu}$")
        ax_hist.set_ylabel("Density")
        ax_hist.legend(loc="upper left", frameon=False)
        ax_hist.text(
            0.98,
            0.98,
            (
                f"N={z_resid.size}\n"
                f"mean={z_mean:.2f}\n"
                f"std={z_std:.2f}\n"
                f"skew={z_skew:.2f}\n"
                f"kurt={z_kurt:.2f}\n"
                f"K2 p={p_norm:.2g}"
            ),
            ha="right",
            va="top",
            transform=ax_hist.transAxes,
        )

        osm, osr = probplot(z_resid, dist="norm", fit=False)
        q_lo = min(float(np.min(osm)), float(np.min(osr)))
        q_hi = max(float(np.max(osm)), float(np.max(osr)))
        ax_qq.scatter(
            osm,
            osr,
            s=10,
            alpha=0.55,
            color="0.25",
            linewidths=0,
            rasterized=True,
        )
        ax_qq.plot([q_lo, q_hi], [q_lo, q_hi], color="tab:red", lw=1.5)
        ax_qq.set_xlabel("Normal quantiles")
        ax_qq.set_ylabel("Observed quantiles")

    fig.tight_layout()
    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


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
    show_cosmo_uncertainty_band=True,
    completeness=True,    # add "<50% complete" red region
    m_lim=24.0,           # survey apparent-magnitude limit for completeness shading
    n_cosmo_draws=50,     # posterior draws to propagate cosmology errors (for xerr)
    random_state=42,      # RNG seed for reproducibility of draws
    z_range=(0.44, 3.16),
    use_alpha_lambda_term=None,
    use_redshift_log_f_term=None,
    dmi_selection_sigma_interp=None,
    sigma_sel_floor_mag=0.05,
):
    """
    Predicted vs Actual M_2500, with:
      • y-error bars from M_model_agn_err(...)
      • x-error bars = apparent_mag_2500_err only
      • Optional cosmology-uncertainty band around y=x from sigma_mu_cosmo(z)
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
    option_flags = resolve_model_option_flags(
        cosmo_model,
        np.asarray(flat_samples).shape[1],
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    priors, model_labels, model_labels_latex = get_model_params(
        cosmo_model,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
    )
    results = {key: np.median(flat_samples[:, i])
               for i, key in enumerate(model_labels)}
    label_to_idx = {k: i for i, k in enumerate(model_labels)}

    # --- intrinsic scatter: sigma_int = exp(log_f) from posterior median ---
    sigma_intrinsic = float(
        np.exp(
            evaluate_log_f(
                results,
                np.array([z_pivot_agn]),
                z_pivot=z_pivot_agn,
                use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
            )[0]
        )
    )

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

    agn_params_arr = agn_model_pack_params(results, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"])
    agn_obs_arr, agn_err_arr, agn_pivot_arr = agn_model_pack_obs(
        df_agn, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"]
    )

    M_2500_pred = M_model_agn(
        agn_params_arr, agn_obs_arr, agn_pivot_arr, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"]
    )
    M_2500_pred_err = M_model_agn_err(
        agn_params_arr, agn_obs_arr, agn_err_arr, agn_pivot_arr, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"]
    )
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
    xerr = np.asarray(m_app_err, dtype=float)

   # if debias:
        #dm_interp = make_dm_function(np.array(df_agn["apparent_mag_2500"].values), np.array(df_agn['z'].values), dms)

    # --- actual minus optional debias ---
    if debias:
        actual_M_2500_eff = actual_M_2500 - evaluate_dm_interp(
            dm_interp,
            df_agn["z"].values,
            df_agn["apparent_mag_2500"].values,
            f_host_2500=df_agn.get("f_host_2500"),
            alpha_lambda=df_agn.get("alpha_lambda"),
        )
    else:
        actual_M_2500_eff = actual_M_2500

    residuals_all = M_2500_pred - actual_M_2500_eff               # mag
    sigma_sel = None
    if debias and dmi_selection_sigma_interp is not None:
        sigma_sel = evaluate_dm_interp(
            dmi_selection_sigma_interp,
            df_agn["z"].values,
            df_agn["apparent_mag_2500"].values,
            f_host_2500=df_agn.get("f_host_2500"),
            alpha_lambda=df_agn.get("alpha_lambda"),
        )
        sigma_sel = np.asarray(sigma_sel, dtype=float)
        sigma_sel_valid = np.isfinite(sigma_sel) & (sigma_sel > 0.0)
        sigma_sel = np.where(
            sigma_sel_valid,
            np.maximum(sigma_sel, float(sigma_sel_floor_mag)),
            np.nan,
        )
    sigma_all = np.sqrt(
        M_2500_pred_err**2 + xerr**2 + sigma_intrinsic**2
    )  # fallback full chi2 denominator; plotted bars still exclude sigma_int
    if sigma_sel is not None:
        sigma_all = np.where(np.isfinite(sigma_sel) & (sigma_sel > 0.0), sigma_sel, sigma_all)

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
            actual_M_2500_bin -= evaluate_dm_interp(
                dm_interp,
                df_agn["z"][bin_mask],
                df_agn["apparent_mag_2500"][bin_mask],
                f_host_2500=df_agn.get("f_host_2500", None)[bin_mask] if "f_host_2500" in df_agn.columns else None,
                alpha_lambda=df_agn.get("alpha_lambda", None)[bin_mask] if "alpha_lambda" in df_agn.columns else None,
            )

        x = actual_M_2500_bin
        y = M_2500_pred[bin_mask]
        xerr_bin = xerr[bin_mask]
        yerr_bin = M_2500_pred_err[bin_mask]
        sigma_bin_chi2 = np.sqrt(xerr_bin**2 + yerr_bin**2 + sigma_intrinsic**2)
        if sigma_sel is not None:
            sigma_sel_bin = sigma_sel[bin_mask]
            sigma_bin_chi2 = np.where(
                np.isfinite(sigma_sel_bin) & (sigma_sel_bin > 0.0),
                sigma_sel_bin,
                sigma_bin_chi2,
            )

        # residuals (for CSV/diagnostics)
        resid = y - x
        resid_bybin_aligned[bin_mask] = resid

        mask_chi2 = (
            np.isfinite(resid)
            & np.isfinite(sigma_bin_chi2)
            & (sigma_bin_chi2 > 0)
        )
        if np.any(mask_chi2):
            dof = max(int(np.count_nonzero(mask_chi2)) - 1, 1)
            chi2_red_bin = float(
                np.sum((resid[mask_chi2] / sigma_bin_chi2[mask_chi2]) ** 2) / dof
            )
        else:
            chi2_red_bin = np.nan

        # pick colors by category
        cats_bin = cats[bin_mask]
        colors_bin = np.where(cats_bin >= 0, palette[np.clip(cats_bin, 0, 4)], "#999999")  # gray for NaN

        # plot errorbars
        ax.errorbar(
            x, y, xerr=xerr_bin, yerr=yerr_bin,
            fmt="none", ecolor="#666666", elinewidth=0.7, alpha=0.4, zorder=2
        )

        # scatter with discrete colors: filled circles inside z-range, filled diamonds outside
        z_bin = z[bin_mask]
        mask_closed = (z_bin > z_range[0]) & (z_bin < z_range[1])
        mask_open   = ~mask_closed

        # filled markers (keep black edges like before)
        ax.scatter(
            x[mask_closed], y[mask_closed],
            facecolors="k", edgecolors='k', #c=colors_bin[mask_closed], 
            s=20, alpha=1.0,
            linewidths=0.8, zorder=3,
        )

        # filled diamonds outside z-range
        ax.scatter(
            x[mask_open], y[mask_open],
            facecolors="k", edgecolors='k', #edgecolors=colors_bin[mask_open],
            marker="D",
            s=20, alpha=1.0, linewidths=1, zorder=3,
        )

        # y = x reference and ±1σ intrinsic band
        ax.plot(xx, xx, color="m", alpha=0.9, lw=2.2, zorder=9)
        if show_cosmo_uncertainty_band:
            cosmo_sigma_bin = sigma_mu_cosmo[bin_mask]
            finite_cosmo_sigma = np.isfinite(cosmo_sigma_bin) & (cosmo_sigma_bin > 0)
            if np.any(finite_cosmo_sigma):
                cosmo_sigma_med = float(np.nanmedian(cosmo_sigma_bin[finite_cosmo_sigma]))
                ax.fill_between(
                    xx,
                    xx - cosmo_sigma_med,
                    xx + cosmo_sigma_med,
                    color="0.5",
                    alpha=0.12,
                    zorder=1,
                    label=r"$y = x \pm 1\sigma_{\rm cosmo}$" if i == 0 else None,
                )
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
        if np.isfinite(chi2_red_bin):
            ax.annotate(
                rf"$\chi^2_\nu = {chi2_red_bin:.2f}$", xy=(0.97, 0.11), xycoords="axes fraction",
                fontsize=20, color="k", ha="right", va="bottom", bbox=boxprops,
            )
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
        if (show_sigma_band or show_cosmo_uncertainty_band or completeness) and i == num_cols-1:
            leg = ax.legend(loc="lower right", fontsize=12, frameon=True)
            leg.get_frame().set_facecolor("none")
            leg.get_frame().set_alpha(box_alpha)
            leg.get_frame().set_edgecolor("none")

    for ax in axes:
        if ax.has_data():
            ax.label_outer()

    os.makedirs(plot_path, exist_ok=True)

    # Save object_id, z, and resid_bybin_aligned to CSV
    m2500_residuals_df = df_agn.loc[:, ["object_id", "z"]].copy()
    m2500_residuals_df["residual"] = resid_bybin_aligned
    m2500_residuals_df.to_csv(os.path.join(plot_path, "m2500_residuals.csv"), index=False)

    _save_figure(
        fig,
        os.path.join(plot_path, f"predicted_vs_actual_M2500{'_debias' if debias else ''}.pdf"),
        dpi=600,
        show=show,
    )

    return residuals_all[m_global], sigma_all[m_global], resid_bybin_aligned, z_bin_indices

def plot_completeness_vs_mag_at_redshifts(
    p_detect, mag_centers, z_centers,
    redshifts=[0.5, 1.0, 2.0, 3.0, 4.0], show=False, plot_path=None
):
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

    # Draw the completeness curves in a single axes for a compact diagnostic view.
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, z in enumerate(redshifts):
        p_vals = p_detect(mag_eval, np.full_like(mag_eval, z))
        color = cmap(norm(z))
        ax.plot(mag_eval, p_vals, label=fr"$z = {z}$", color=color, linestyle=line_styles[i % len(line_styles)])

    #sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    #sm.set_array([])
    #plt.colorbar(sm, label="Redshift", ticks=redshifts)

    ax.set_xlabel(r"$m$ ($i$ mag)")
    ax.set_ylabel(r"$p(I{=}1|m, z)$")
    ax.legend(fontsize=16, loc="upper right", frameon=False)
    ax.set_xlim(17, 25)
    ax.grid(False)
    fig.tight_layout()
    base_plot_path = plot_path or "plots/hubble"
    completeness_path = os.path.join(base_plot_path, "completeness")
    os.makedirs(completeness_path, exist_ok=True)
    _save_figure(fig, os.path.join(completeness_path, "completeness_vs_mag_at_redshifts.pdf"), dpi=300, show=show)



def plot_full_residuals(
    df_agn, residuals, residuals_err, flat_samples, cosmo_model, z_pivot_agn,
    debias=False, dm_interp=None, plot_path='plots/hubble', show=False,
    *, nbins=10, min_count=5, z_cut=None, key_y='residuals', key_color='z',
    z_range=(0.44, 3.16), residual_label='residuals', output_tag='full_residuals',
    max_categories=12, category_min_count=5, category_jitter=0.15,
    use_alpha_lambda_term=None, use_redshift_log_f_term=None,
):
    df_agn = df_agn.copy()
    df_agn[residual_label] = residuals
    if key_y == 'residuals':
        key_y = residual_label
    if key_color == 'residuals':
        key_color = residual_label

    df_agn = df_agn.reset_index(drop=True)

    def _median_param_dict(samples):
        option_flags = resolve_model_option_flags(
            cosmo_model,
            np.asarray(samples).shape[1],
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_redshift_log_f_term=use_redshift_log_f_term,
        )
        _, model_labels, _ = get_model_params(
            cosmo_model,
            use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
            use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
        )
        return {
            key: np.percentile(samples[:, i], [16, 50, 84])
            for i, key in enumerate(model_labels)
        }

    def _build_cosmology(results):
        if cosmo_model == 'FlatwCDM':
            return FlatwCDM(H0=results['H0'][1], Om0=results['Om0'][1], w0=results['w0'][1])
        if cosmo_model == 'FlatwpwaCDM':
            return FlatwpwaCDM(
                H0=results['H0'][1],
                Om0=results['Om0'][1],
                wp=results['wp'][1],
                wa=results['wa'][1],
                zp=z_pivot_agn,
            )
        if cosmo_model == 'Flatw0waCDM':
            return Flatw0waCDM(
                H0=results['H0'][1],
                Om0=results['Om0'][1],
                w0=results['w0'][1],
                wa=results['wa'][1],
            )
        if cosmo_model == 'FlatLambdaCDM':
            return FlatLambdaCDM(H0=results['H0'][1], Om0=results['Om0'][1])
        raise ValueError("Invalid cosmology model.")

    def _safelog(a):
        return np.log10(np.abs(a) + 1e-10)

    def _augment_plot_columns(frame, cosmo):
        frame['MY_M_2500'] = frame['apparent_mag_2500'].values - cosmo.distmod(frame['z'].values).value

        if debias:
            delta = evaluate_dm_interp(
                dm_interp,
                frame["z"].values,
                frame["apparent_mag_2500"].values,
                f_host_2500=frame.get("f_host_2500"),
                alpha_lambda=frame.get("alpha_lambda"),
            )
            frame['MY_M_2500'] -= delta
            frame['apparent_mag_2500'] -= delta
            if 'apparent_mag_2500_reddened' in frame.columns:
                frame['apparent_mag_2500_reddened'] -= delta

        if 'apparent_mag_2500_err' in frame.columns and 'apparent_mag_2500' in frame.columns:
            mag = np.asarray(frame['apparent_mag_2500'], dtype=float)
            mag_err = np.asarray(frame['apparent_mag_2500_err'], dtype=float)
            frame['rel_apparent_mag_2500_err'] = np.divide(
                mag_err,
                np.maximum(np.abs(mag), 1e-8),
                out=np.full_like(mag_err, np.nan, dtype=float),
                where=np.isfinite(mag_err) & np.isfinite(mag),
            )
        if 'frac_host_psf_2500' in frame.columns:
            frac_host = np.asarray(frame['frac_host_psf_2500'], dtype=float)
            log_frac_host = np.full(frac_host.shape, np.nan, dtype=float)
            valid_frac_host = np.isfinite(frac_host) & (frac_host > 0)
            log_frac_host[valid_frac_host] = np.log10(frac_host[valid_frac_host])
            frame['log_frac_host_psf_2500'] = log_frac_host

        if "f_na" in frame.columns or "f_br" in frame.columns:
            f_na = (
                pd.to_numeric(frame["f_na"], errors="coerce")
                if "f_na" in frame.columns
                else pd.Series(0.0, index=frame.index)
            )
            f_br = (
                pd.to_numeric(frame["f_br"], errors="coerce")
                if "f_br" in frame.columns
                else pd.Series(0.0, index=frame.index)
            )
            frame["f_lines"] = f_na.fillna(0.0) + f_br.fillna(0.0)
            frame["log_f_lines"] = _safelog(frame["f_lines"])

        log_columns = {
            'dm_red': 'log_dm_red',
            'reddening_integral': 'log_reddening_integral',
            'reddening_proxy': 'log_reddening_proxy',
            'BI': 'log_bi',
            'bi': 'log_bi',
            'redchi': 'log_redchi',
            'redchi2_conti_full': 'log_redchi2_conti_full',
            'apparent_mag_2500_err': 'log_apparent_mag_2500_err',
            'log_sigma_uv_err': 'log_log_sigma_uv_err',
            'log_tau_uv_rf_err': 'log_log_tau_uv_rf_err',
            'psf_minus_fiber_r': 'log_psf_minus_fiber_r',
            'petroRad_r': 'log_petroRad_r',
            'log_tau_uv_rhat': 'log_log_tau_uv_rhat',
            'f_host_2500': 'log_f_host_2500',
            'f_bc_3000': 'log_f_bc_3000',
            'f_fe_uv_3000': 'log_f_fe_uv_3000',
            'RCHI2': 'log_RCHI2',
            'RCHI2DIFF': 'log_RCHI2DIFF',
            'reddening_ebv': 'log_reddening_ebv',
            'ebv_mw': 'log_ebv_mw',
            'ebv_wu': 'log_ebv_wu',
            'frac_bc_2500': 'log_frac_bc_2500',
            'f_PL': 'log_f_PL',

        }
        for source_col, derived_col in log_columns.items():
            if source_col in frame.columns:
                frame[derived_col] = _safelog(frame[source_col])

        if {'log_tau_uv_rf', 'log_tau_fast_uv'}.issubset(frame.columns):
            z = np.asarray(frame['z'], dtype=float)
            log_tau_uv_rf = np.asarray(frame['log_tau_uv_rf'], dtype=float)
            log_tau_fast_uv = np.asarray(frame['log_tau_fast_uv'], dtype=float)
            log_tau_fast_uv_rf = log_tau_fast_uv - np.log10(1.0 + z)
            tau_uv_rf = np.power(10.0, log_tau_uv_rf)
            tau_fast_uv_rf = np.power(10.0, log_tau_fast_uv_rf)
            frame['delta_tau_uv_fast_rf'] = np.where(
                np.isfinite(tau_uv_rf) & np.isfinite(tau_fast_uv_rf),
                tau_uv_rf - tau_fast_uv_rf,
                np.nan,
            )
            frame['log_delta_tau_uv_fast_rf'] = np.where(
                np.isfinite(frame['delta_tau_uv_fast_rf']) & (frame['delta_tau_uv_fast_rf'] > 0.0),
                np.log10(frame['delta_tau_uv_fast_rf']),
                np.nan,
            )

        for col in ['BC', 'decomp_host', 'poly']:
            if col in frame.columns:
                frame[col] = frame[col].replace(
                    {True: 1, False: 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0}
                )

    results = _median_param_dict(flat_samples)
    cosmo = _build_cosmology(results)
    _augment_plot_columns(df_agn, cosmo)

    # ---- Which x-keys to show (keep your order) ----
    keys = [col for col in [
        'log_f_lines',
        'f_PL', 'log_f_PL',
        'log_amp_delta_bc', 
        'frac_bc_2500', 'log_frac_bc_2500',
        'log_reddening_ebv',
        'ebv_mw', 'log_ebv_mw',
        'ebv_wu', 'log_ebv_wu',
        'log_bi',
        'apparent_mag_2500_intrinsict',
        #'chi_sq_red_g_raw', 'log_chi_sq_red_g_raw', 'variability_chi_sq_g_raw', 'log_variability_chi_sq_g_raw',
        #'pvalue_g', 'log_pvalue_g',
        #'sdss_plate_count', 'RCHI2', 'log_RCHI2', 'RCHI2DIFF', 'log_RCHI2DIFF', 'VDISP', 'ZWARNING', 'RUN2D',
        'log_frac_host_psf_2500',
        'wrms', 'log_f_bc_3000', 'log_f_fe_uv_3000', 'log_f_host_2500',
        'rel_apparent_mag_2500_err',
        #'apparent_mag_2500_err', 'log_apparent_mag_2500_err', 
        #'log_sigma_uv_err', 'log_log_sigma_uv_err',
        #'log_tau_uv_rf_err', 'log_log_tau_uv_rf_err',
        #'apparent_mag_2500', 'apparent_mag_2500_reddened', 'dm_red', 'log_dm_red', 
        'ebv_wu',
        #'conti_a_0', 'PL_slope_blue', 
        #'MY_M_2500', 'z', 'log_lbol', 'log_ledd_ratio', 
        'delta_tau_uv_fast_rf', 'log_delta_tau_uv_fast_rf',
        'log_sigma_uv', 'log_tau_uv_rf',
        'log_tau_uv', 'log_tau_fast_uv',
        #'log_tau_fast_band_u_RF', 'log_tau_fast_band_g_RF', 'log_tau_fast_band_r_RF', 'log_tau_fast_band_i_RF', 'log_tau_fast_band_z_RF',
        'sn_median_all', 'redchi', 'log_redchi', 'alpha_lambda',
        #'redchi2_conti_full', 'log_redchi2_conti_full',
        'bwb_alpha', 'bwb_beta', 
        #'log_rho', 't_rf_length', 'tau_band_RF_mean',
        #'log_tau_band_RF_mean', 'log_t_rf_length', 
        #'alphaOX', 'alphaOX_int',
        #'bwb_alpha_u', 'bwb_alpha_g', 'bwb_alpha_r', 'bwb_alpha_i', 'bwb_alpha_z',
        'eta_sigma', 'eta_tau',
        'log_amp_delta_bc',
        #'PL_slope_blue', 'lam_min', 'lam_max', 'lam_range', 
        #'poly1', 'psf_minus_fiber_r', 'log_psf_minus_fiber_r', 'petroRad_r', 'log_petroRad_r',
        #'cadence', 'number_points',
        #'log_jitter_total', 'log_amp_delta_blr_total',
        'log_amp_delta_blr_u', 'log_amp_delta_blr_g', 'log_amp_delta_blr_r', 'log_amp_delta_blr_i', 'log_amp_delta_blr_z',
        'log_amp_delta_lya_band_u', 'log_amp_delta_lya_band_g', 'log_amp_delta_lya_band_r', 'log_amp_delta_lya_band_i', 'log_amp_delta_lya_band_z',
        #'log_jitter_u', 'log_jitter_g', 'log_jitter_r', 'log_jitter_i', 'log_jitter_z',

    ] if col in df_agn.columns]


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
    share_residual_y = key_y == residual_label

    fig, axes_grid = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        sharey="row" if share_residual_y else False,
        squeeze=False,
    )
    axes = axes_grid.flatten()

    global_color_norm = None
    global_color_cmap = 'viridis' if key_y == residual_label else 'bwr_r'
    if key_color in df_agn.columns and pd.api.types.is_numeric_dtype(df_agn[key_color]):
        color_all = pd.to_numeric(df_agn[key_color], errors='coerce').to_numpy(dtype=float)
        finite_color_all = np.isfinite(color_all)
        if np.any(finite_color_all):
            cmin = float(np.nanmin(color_all[finite_color_all]))
            cmax = float(np.nanmax(color_all[finite_color_all]))
            if cmin == cmax:
                cmin, cmax = cmin - 1e-6, cmax + 1e-6
            global_color_norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)

    def _panel_mask(key):
        if pd.api.types.is_numeric_dtype(df_agn[key]):
            mask = (df_agn[key] > -1e9) & np.isfinite(df_agn[key])
        else:
            mask = np.ones(len(df_agn), dtype=bool)
        if z_cut is not None:
            mask &= df_agn['z'] < z_cut
        if key in keys_masks:
            low, high = keys_masks[key]
            mask &= df_agn[key].between(low, high)
        if key in {"f_host_2500", "log_f_host_2500"} and "f_host_2500" in df_agn.columns:
            mask &= pd.to_numeric(df_agn["f_host_2500"], errors="coerce").to_numpy(dtype=float) > 0.0
        mask &= np.isfinite(residuals)
        if isinstance(mask, pd.Series):
            mask = mask.fillna(False).to_numpy(dtype=bool)
        elif hasattr(mask, "fillna"):
            mask = np.asarray(mask.fillna(False), dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
        return mask

    def _panel_xy_and_style(mask, key):
        color_values = df_agn.loc[mask, key_color].to_numpy()
        if key_y == residual_label:
            x = df_agn.loc[mask, key].to_numpy()
            y = residuals[mask]
            xlabel, ylabel = key, key_y
            color_num = pd.to_numeric(pd.Series(color_values), errors='coerce').to_numpy(dtype=float)
            finite_color = np.isfinite(color_num)
            if np.any(finite_color):
                norm = global_color_norm
                if norm is None:
                    cmin = float(np.nanmin(color_num[finite_color]))
                    cmax = float(np.nanmax(color_num[finite_color]))
                    if cmin == cmax:
                        cmin, cmax = cmin - 1e-6, cmax + 1e-6
                    norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
                color_values = color_num
            else:
                norm = None
            cmap = global_color_cmap
        else:
            x = df_agn.loc[mask, key_y].to_numpy()
            y = df_agn.loc[mask, key].to_numpy()
            xlabel, ylabel = key_y, key
            color_num = pd.to_numeric(pd.Series(color_values), errors='coerce').to_numpy(dtype=float)
            finite_color = np.isfinite(color_num)
            if np.any(finite_color):
                norm = global_color_norm
                if norm is None:
                    cmin = float(np.nanmin(color_num[finite_color]))
                    cmax = float(np.nanmax(color_num[finite_color]))
                    if cmin == cmax:
                        cmin, cmax = cmin - 1e-6, cmax + 1e-6
                    norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
                color_values = color_num
            else:
                norm = None
            cmap = global_color_cmap
        return x, y, color_values, xlabel, ylabel, cmap, norm

    def _normalize_category_value(value):
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        if text == "" or text.lower() in {"nan", "none", "null"}:
            return np.nan
        return text

    def _prepare_categories(x_raw):
        x_cat = pd.Series(x_raw).apply(_normalize_category_value)
        valid = x_cat.notna()
        if not np.any(valid):
            return None, None, None
        x_cat = x_cat.loc[valid].reset_index(drop=True)
        counts = x_cat.value_counts()
        eligible = counts[counts >= int(category_min_count)]
        if eligible.empty:
            keep = counts.index[: int(max_categories)]
        else:
            keep = eligible.index[: int(max_categories)]
        x_limited = x_cat.where(x_cat.isin(set(keep)), "OTHER")
        order = x_limited.value_counts().index.tolist()
        x_limited = pd.Categorical(x_limited, categories=order, ordered=True)
        positions = np.asarray(x_limited.codes, dtype=float)
        return positions, np.asarray(x_limited.astype(str)), np.asarray(valid)

    def _draw_reference_guides(ax, key, x):
        if key_y == residual_label:
            ax.axhline(0, color='red', linestyle='--', lw=1)
            if key in keys_yx_line and len(x):
                xmin, xmax = np.nanmin(x), np.nanmax(x)
                xmid = np.nanmean(x)
                ax.plot([xmin, xmax], [xmin - xmid, xmax - xmid], color='red', linestyle='--', lw=1)

    def _draw_binned_overlay(ax, x, y, err):
        _ = err
        mfin = np.isfinite(x) & np.isfinite(y)
        if not np.any(mfin):
            return

        xb, yb = np.asarray(x[mfin], dtype=float), np.asarray(y[mfin], dtype=float)
        lo, hi = np.nanpercentile(xb, [1, 99])
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            lo, hi = np.nanmin(xb), np.nanmax(xb)
        bins = np.linspace(lo, hi, nbins + 1)
        x_med, y_med, y_lo, y_hi, y_med_err = [], [], [], [], []
        for i_bin in range(len(bins) - 1):
            if i_bin == len(bins) - 2:
                in_bin = (xb >= bins[i_bin]) & (xb <= bins[i_bin + 1])
            else:
                in_bin = (xb >= bins[i_bin]) & (xb < bins[i_bin + 1])
            if np.count_nonzero(in_bin) < int(min_count):
                continue
            x_bin = xb[in_bin]
            y_bin = yb[in_bin]
            x_med.append(np.nanmedian(x_bin))
            y_bin_med = np.nanmedian(y_bin)
            y_med.append(y_bin_med)
            lo_i, hi_i = np.nanpercentile(y_bin, [16, 84])
            y_lo.append(lo_i)
            y_hi.append(hi_i)
            robust_sigma = max(0.5 * (hi_i - lo_i), 0.0)
            y_med_err.append(1.253 * robust_sigma / np.sqrt(np.count_nonzero(in_bin)))

        if x_med:
            x_med = np.asarray(x_med, dtype=float)
            y_med = np.asarray(y_med, dtype=float)
            y_lo = np.asarray(y_lo, dtype=float)
            y_hi = np.asarray(y_hi, dtype=float)
            y_med_err = np.asarray(y_med_err, dtype=float)
            order = np.argsort(x_med)
            x_med = x_med[order]
            y_med = y_med[order]
            y_lo = y_lo[order]
            y_hi = y_hi[order]
            y_med_err = y_med_err[order]
            ax.fill_between(
                x_med,
                y_lo,
                y_hi,
                color="0.6",
                alpha=0.12,
                linewidth=0,
                zorder=8,
            )
            ax.fill_between(
                x_med,
                y_med - y_med_err,
                y_med + y_med_err,
                color="red",
                alpha=0.18,
                linewidth=0,
                zorder=9,
            )
            ax.plot(
                x_med,
                y_med,
                color="red",
                lw=1.8,
                alpha=0.95,
                zorder=10,
            )

    def _set_robust_numeric_xlim(ax, x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            return
        x_lo, x_hi = np.nanpercentile(x, [1.0, 99.0])
        if not (np.isfinite(x_lo) and np.isfinite(x_hi) and x_hi > x_lo):
            return
        pad = 0.05 * (x_hi - x_lo)
        ax.set_xlim(x_lo - pad, x_hi + pad)

    for idx, key in enumerate(keys):
        ax = axes[idx]
        try:
            mask = _panel_mask(key)
            x, y, color_values, xlabel, ylabel, cmap, norm = _panel_xy_and_style(mask, key)
            ax.set_xlabel(xlabel)
            if share_residual_y:
                if idx % n_cols == 0:
                    ax.set_ylabel(ylabel)
                else:
                    ax.set_ylabel("")
                    ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(ylabel)
            z_masked = df_agn.loc[mask, 'z'].to_numpy(dtype=float)
            x_is_numeric = pd.api.types.is_numeric_dtype(pd.Series(x))
            if key_y == residual_label and not x_is_numeric:
                cat_pos, cat_vals, valid_cat_mask = _prepare_categories(x)
                if cat_pos is None:
                    raise ValueError(f"No finite categorical values for key '{key}'.")
                y = np.asarray(y)[valid_cat_mask]
                z_masked = z_masked[valid_cat_mask]
                color_values = np.asarray(color_values)[valid_cat_mask]
                in_z = (z_masked >= z_range[0]) & (z_masked <= z_range[1])
                out_z = ~in_z

                cats = list(dict.fromkeys(cat_vals.tolist()))
                box_data = [np.asarray(y)[cat_vals == cat] for cat in cats]
                ax.boxplot(
                    box_data,
                    positions=np.arange(len(cats), dtype=float),
                    widths=0.55,
                    showfliers=False,
                    patch_artist=True,
                    boxprops=dict(facecolor='white', edgecolor='0.35', linewidth=1.0),
                    medianprops=dict(color='tab:red', linewidth=1.2),
                    whiskerprops=dict(color='0.35', linewidth=1.0),
                    capprops=dict(color='0.35', linewidth=1.0),
                )

                rng = np.random.default_rng(1000 + idx)
                jitter = rng.uniform(-category_jitter, category_jitter, size=len(cat_pos))
                xj = cat_pos + jitter

                color_num = pd.to_numeric(pd.Series(color_values), errors='coerce').to_numpy(dtype=float)
                finite_color = np.isfinite(color_num)
                use_numeric_color = np.any(finite_color)
                sc = None
                if use_numeric_color:
                    cmin = float(np.nanmin(color_num[finite_color]))
                    cmax = float(np.nanmax(color_num[finite_color]))
                    if cmin == cmax:
                        cmin, cmax = cmin - 1e-6, cmax + 1e-6
                    cat_norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
                    if np.any(in_z):
                        sc = ax.scatter(
                            xj[in_z],
                            y[in_z],
                            c=color_num[in_z],
                            cmap=cmap,
                            norm=cat_norm,
                            s=10,
                            alpha=0.5,
                            rasterized=True,
                        )
                    if np.any(out_z):
                        edgecols = mpl.cm.get_cmap(cmap)(cat_norm(color_num[out_z]))
                        sc_out = ax.scatter(
                            xj[out_z],
                            y[out_z],
                            c=color_num[out_z],
                            cmap=cmap,
                            norm=cat_norm,
                            s=10,
                            alpha=0.8,
                            marker='D',
                            edgecolors=edgecols,
                            linewidths=0.8,
                            rasterized=True,
                        )
                        if sc is None:
                            sc = sc_out
                    if sc is not None and norm is not None and global_color_norm is None:
                        cbar = fig.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
                        cbar.set_label(key_color, fontsize=12)
                else:
                    if np.any(in_z):
                        ax.scatter(
                            xj[in_z],
                            y[in_z],
                            c='tab:blue',
                            s=10,
                            alpha=0.5,
                            rasterized=True,
                        )
                    if np.any(out_z):
                        ax.scatter(
                            xj[out_z],
                            y[out_z],
                            c='tab:blue',
                            s=10,
                            alpha=0.8,
                            marker='D',
                            edgecolors='tab:blue',
                            linewidths=0.8,
                            rasterized=True,
                        )

                if residuals_err is None:
                    err = np.full_like(y, np.nan, dtype=float)
                else:
                    err = np.asarray(residuals_err)[mask][valid_cat_mask]
                cat_x_med, cat_y_med, cat_y_lo, cat_y_hi, cat_y_med_err = [], [], [], [], []
                for ci, cat in enumerate(cats):
                    cat_mask = (cat_vals == cat)
                    _ = err
                    ww = np.isfinite(y[cat_mask])
                    if not np.any(ww):
                        continue
                    ycat = y[cat_mask][ww]
                    if ycat.size < int(min_count):
                        continue
                    cat_x_med.append(float(ci))
                    ycat_med = np.nanmedian(ycat)
                    cat_y_med.append(ycat_med)
                    lo_i, hi_i = np.nanpercentile(ycat, [16, 84])
                    cat_y_lo.append(lo_i)
                    cat_y_hi.append(hi_i)
                    robust_sigma = max(0.5 * (hi_i - lo_i), 0.0)
                    cat_y_med_err.append(1.253 * robust_sigma / np.sqrt(ycat.size))

                if cat_x_med:
                    cat_x_med = np.asarray(cat_x_med, dtype=float)
                    cat_y_med = np.asarray(cat_y_med, dtype=float)
                    cat_y_lo = np.asarray(cat_y_lo, dtype=float)
                    cat_y_hi = np.asarray(cat_y_hi, dtype=float)
                    cat_y_med_err = np.asarray(cat_y_med_err, dtype=float)
                    ax.fill_between(
                        cat_x_med,
                        cat_y_lo,
                        cat_y_hi,
                        color="0.6",
                        alpha=0.12,
                        linewidth=0,
                        zorder=8,
                    )
                    ax.fill_between(
                        cat_x_med,
                        cat_y_med - cat_y_med_err,
                        cat_y_med + cat_y_med_err,
                        color="red",
                        alpha=0.18,
                        linewidth=0,
                        zorder=9,
                    )
                    ax.plot(
                        cat_x_med,
                        cat_y_med,
                        color="red",
                        lw=1.8,
                        alpha=0.95,
                        zorder=10,
                    )

                ax.set_xticks(np.arange(len(cats), dtype=float))
                ax.set_xticklabels(cats, rotation=45, ha='right')
                ax.axhline(0, color='red', linestyle='--', lw=1)
            else:
                in_z = (z_masked >= z_range[0]) & (z_masked <= z_range[1])
                out_z = ~in_z

                sc = None
                if np.any(in_z):
                    if norm is None:
                        sc = ax.scatter(
                            x[in_z],
                            y[in_z],
                            c='tab:blue',
                            s=10,
                            alpha=0.5,
                            rasterized=True,
                        )
                    else:
                        sc = ax.scatter(
                            x[in_z],
                            y[in_z],
                            c=color_values[in_z],
                            cmap=cmap,
                            norm=norm,
                            s=10,
                            alpha=0.5,
                            rasterized=True,
                        )
                if np.any(out_z):
                    if norm is None:
                        sc_out = ax.scatter(
                            x[out_z],
                            y[out_z],
                            c='tab:blue',
                            s=10,
                            alpha=0.8,
                            marker='D',
                            edgecolors='tab:blue',
                            linewidths=0.8,
                            rasterized=True,
                        )
                    else:
                        edgecols = mpl.cm.get_cmap(cmap)(norm(color_values[out_z]))
                        sc_out = ax.scatter(
                            x[out_z],
                            y[out_z],
                            c=color_values[out_z],
                            cmap=cmap,
                            norm=norm,
                            s=10,
                            alpha=0.8,
                            marker='D',
                            edgecolors=edgecols,
                            linewidths=0.8,
                            rasterized=True,
                        )
                    if sc is None:
                        sc = sc_out
                _draw_reference_guides(ax, key, x)

                if sc is not None and norm is not None and global_color_norm is None:
                    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
                    cbar.set_label(key_color, fontsize=12)

                if residuals_err is None:
                    err = np.full_like(y, np.nan, dtype=float)
                else:
                    err = np.asarray(residuals_err)[mask]
                if pd.api.types.is_numeric_dtype(pd.Series(x)) and pd.api.types.is_numeric_dtype(pd.Series(y)):
                    _set_robust_numeric_xlim(ax, x)
                    _draw_binned_overlay(ax, x, y, err)
        except Exception as e:
            print(f"Error processing key {key}: {e}")
            ax.axis('off')

        if key_y == residual_label and ax.has_data():
            ax.set_ylim(*_FULL_RESIDUAL_YLIM)
        ax.grid(False)

    # Hide any extra axes
    for j in range(n_keys, len(axes)):
        axes[j].axis('off')

    if global_color_norm is not None:
        sm = mpl.cm.ScalarMappable(norm=global_color_norm, cmap=global_color_cmap)
        sm.set_array([])
        for i_row in range(n_rows):
            row_start = i_row * n_cols
            row_end = min((i_row + 1) * n_cols, len(axes))
            row_axes = [
                axes[j]
                for j in range(row_start, row_end)
                if j < n_keys and axes[j].axison
            ]
            if row_axes:
                cbar = fig.colorbar(
                    sm,
                    ax=row_axes,
                    orientation='vertical',
                    fraction=0.025,
                    pad=0.02,
                )
                cbar.set_label(key_color, fontsize=12)

    os.makedirs(plot_path, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 0.98, 1.0))
    return _save_figure(
        fig,
        os.path.join(plot_path, f"{output_tag}_{'debiased' if debias else 'biased'}_y{key_y}_c{key_color}_zcut{z_cut}.pdf"),
        dpi=150,
        show=show,
    )


def plot_full_residuals_rz(
    df_agn, residuals, residuals_err, flat_samples, cosmo_model, z_pivot_agn,
    debias=False, dm_interp=None, plot_path='plots/hubble', show=False,
    *, nbins=10, min_count=5, z_cut=None, key_y='r_z', key_color='z',
    z_range=(0.44, 3.16), nz_bins=12, z_min_count=8,
    lowess_frac=0.25, lowess_it=1, lowess_min_points=10,
    max_categories=12, category_min_count=5, category_jitter=0.15,
    use_alpha_lambda_term=None, use_redshift_log_f_term=None,
):
    """
    Plot redshift-detrended residual diagnostics where
    r_z = residual - E[residual | z].

    E[residual | z] is estimated with the shared 1D redshift smoother.
    """
    z = np.asarray(df_agn['z'], dtype=float)
    r = np.asarray(residuals, dtype=float)
    if residuals_err is None:
        rerr = np.full_like(r, np.nan, dtype=float)
    else:
        rerr = np.asarray(residuals_err, dtype=float)

    good = np.isfinite(z) & np.isfinite(r)
    good_weighted = good & np.isfinite(rerr) & (rerr > 0)

    if np.count_nonzero(good) < max(z_min_count, 3):
        r_z = np.asarray(r, dtype=float)
    else:
        use_weighted = np.count_nonzero(good_weighted) >= max(z_min_count, 3)
        fit_mask = good_weighted if use_weighted else good
        z_good = z[fit_mask]
        r_good = r[fit_mask]
        rerr_good = rerr[fit_mask] if use_weighted else None
        trend = build_smooth_trend_1d(
            z_good,
            r_good,
            yerr=rerr_good,
            frac=lowess_frac,
            it=lowess_it,
            min_points=max(int(lowess_min_points), int(z_min_count)),
            fallback_bins=nz_bins,
        )
        trend_at_z = np.asarray(trend(z), dtype=float)
        r_z = r - trend_at_z

    return plot_full_residuals(
        df_agn,
        r_z,
        residuals_err,
        flat_samples,
        cosmo_model,
        z_pivot_agn,
        debias=debias,
        dm_interp=dm_interp,
        plot_path=plot_path,
        show=show,
        nbins=nbins,
        min_count=min_count,
        z_cut=z_cut,
        key_y=key_y,
        key_color=key_color,
        z_range=z_range,
        residual_label='r_z',
        output_tag='full_residuals_rz',
        max_categories=max_categories,
        category_min_count=category_min_count,
        category_jitter=category_jitter,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )

def _kde_conf_levels(Z, conf=(0.954, 0.683), plot_path=None):
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
    show_residuals=False, df_calibrators=None, z_range=(0.44, 3.16),
    use_alpha_lambda_term=None, use_redshift_log_f_term=None,
    dmi_values=None,
    dmi_selection_sigma_interp=None,
    sigma_sel_floor_mag=0.05,
):
    d = df_agn.copy()

    # --- Thinning for speed ---
    n_samples = int(flat_samples.shape[0])
    thin_factor = max(1, n_samples // 500)
    flat_samples = flat_samples[::thin_factor]

    # --- Indices & parameter names ---
    option_flags = resolve_model_option_flags(
        cosmo_model,
        np.asarray(flat_samples).shape[1],
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    priors, model_labels, model_labels_latex = get_model_params(
        cosmo_model,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
    )
    param_indices = {name: model_labels.index(name) for name in model_labels}

    # --- Pack obs/errs/pivots once (MAIN sample) ---
    agn_obs_arr, agn_err_arr, agn_pivot_arr = agn_model_pack_obs(
        d, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"]
    )

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
        actual_M2500 = (
            d["apparent_mag_2500"]
            - _resolve_debias_values(
                d,
                dm_interp=dm_interp,
                dmi_values=dmi_values,
            )
        ) - cosmo.distmod(d["z"]).value
    else:
        actual_M2500 = d['apparent_mag_2500'] - cosmo.distmod(d['z']).value
    actual_logL2500 = convert_M2500_to_logL2500(actual_M2500)
    y_log_meas_err = 0.4 * np.asarray(d['apparent_mag_2500_err'].fillna(0.0))

    # --- Reference x (built at POSTERIOR-MEDIAN params) ---
    med_arr = agn_model_pack_params(med_params, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"])
    M0_med = med_arr[agn_model_pidx["M0_agn"]]
    x_log_ref = -0.4 * (
        M_model_agn(
        med_arr, agn_obs_arr, agn_pivot_arr, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"]
        ) - M0_med
    )
    x_ref = 10.0 ** x_log_ref

    # x errors for MAIN at median params
    pred_M_err_med = M_model_agn_err(
        med_arr, agn_obs_arr, agn_err_arr, agn_pivot_arr, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"]
    )
    x_log_err_med = 0.4 * pred_M_err_med
    x_lower = 10.0 ** (x_log_ref - x_log_err_med)
    x_upper = 10.0 ** (x_log_ref + x_log_err_med)
    xerr_asym = np.vstack((x_ref - np.maximum(x_lower, 1e-300),
                           np.maximum(x_upper, x_ref) - x_ref))

    # Compute calibrators x,y range for plotting band
    if df_calibrators is not None and len(df_calibrators) > 0:
        ds = df_calibrators.copy()

        M2500_show = ds['apparent_mag_2500'].values - cosmo.distmod(ds['z'].values).value
        if debias:
            M2500_show -= evaluate_dm_interp(
                dm_interp,
                ds["z"].values,
                ds["apparent_mag_2500"].values,
                f_host_2500=ds.get("f_host_2500"),
                alpha_lambda=ds.get("alpha_lambda"),
            )
        actual_logL2500_show = convert_M2500_to_logL2500(M2500_show)
        y_log_meas_err_show = 0.4 * np.asarray(ds['apparent_mag_2500_err'].fillna(0.0), dtype=float)
        yerr_linear_show = (10.0**actual_logL2500_show) * np.log(10.0) * y_log_meas_err_show

        # x for SHOW at median params, using the AGN fit pivots.
        obs_show, err_show, _ = agn_model_pack_obs(
            ds, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"]
        )
        x_log_ref_show = -0.4 * (
            M_model_agn(
                med_arr,
                obs_show,
                agn_pivot_arr,
                use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
            ) - M0_med
        )
        x_show = 10.0 ** x_log_ref_show

        pred_M_err_show = M_model_agn_err(
            med_arr,
            obs_show,
            err_show,
            agn_pivot_arr,
            use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        )
        x_log_err_show = 0.4 * pred_M_err_show
        x_log_lower_show = np.min(np.ravel(x_log_ref_show - x_log_err_show))
        x_log_upper_show = np.max(np.ravel(x_log_ref_show + x_log_err_show))
        x_lower_show = 10.0 ** (x_log_ref_show - x_log_err_show)
        x_upper_show = 10.0 ** (x_log_ref_show + x_log_err_show)
    else:
        x_log_lower_show = 0
        x_log_upper_show = 0
        x_log_ref_show = x_log_ref
        pred_M_err_show = 0

    # --- Grid and band (unchanged) ---
    # x_min_err = np.min([np.min(x_log_ref - x_log_err_med), np.min(x_log_lower_show)])
    # x_max_err = np.max([np.max(x_log_ref + x_log_err_med), np.max(x_log_upper_show)])
    x_min_err = np.min([np.min(x_log_ref), np.min(x_log_lower_show)])
    x_max_err = np.max([np.max(x_log_ref), np.max(x_log_upper_show)])
    print(f"x_log_ref range with errors: {x_min_err:.3f} to {x_max_err:.3f}")
    x_lo = x_min_err - 1.8
    x_hi = x_max_err + 3
    x_log_grid = np.linspace(x_lo, x_hi, 250)
    x_grid = 10.0 ** x_log_grid

    M0_samples = np.asarray(flat_samples[:, param_indices["M0_agn"]], dtype=float)
    ylog_grid_by_sample = x_log_grid[None, :] + convert_M2500_to_logL2500(M0_samples)[:, None]
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
    mask_in  = d["z"].between(z_range[0], z_range[1])
    mask_out = ~mask_in

    # inside redshift range: filled markers
    ax.errorbar(
        x_ref[mask_in], 10**actual_logL2500[mask_in], xerr=xerr_asym[:, mask_in], yerr=yerr_linear[mask_in],
        fmt='o', linestyle='none', markersize=4, mfc=(0,0,0,0.4), mec="none",
        #markeredgewidth=0,
        ecolor=(0.2, 0.2, 0.2, 0.1), elinewidth=0.8, capsize=2, capthick=0.8,
        zorder=1, label="AGN"
    )
    # outside redshift range: filled diamonds
    ax.errorbar(
        x_ref[mask_out], 10**actual_logL2500[mask_out], xerr=xerr_asym[:, mask_out], yerr=yerr_linear[mask_out],
        fmt='D', linestyle='none', markersize=3, mfc=(0,0,0,0.4), mec="none",
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

    # --- Suberlak+2021 comparison in the same luminosity-space x convention ---
    # Suberlak+2021 gives separate luminosity slopes for log SF_inf and log tau.
    # Convert those into d log L / d log x for
    # x = (sigma/sigma0)^alpha_L (tau/tau0)^beta_L.
    #
    # Also plot a second variant that assumes fixed Eddington ratio, so
    # log M_BH contributes as log M_BH ∝ log L ∝ -0.4 M_i.
    c_tau_suberlak = 0.035
    c_tau_suberlak_err = 0.007
    d_tau_suberlak = 0.141
    c_sigma_suberlak = 0.118
    c_sigma_suberlak_err = 0.003
    d_sigma_suberlak = 0.118
    alpha_agn_L = med_params["alpha_agn"] * (-1.0 / 2.5)
    beta_agn_L = med_params["beta_agn"] * (-1.0 / 2.5)
    xcm = float(np.nanmean(x_log_ref))
    ylog_anchor = np.interp(xcm, x_log_grid, ylog_med)
    L_anchor = 10.0**ylog_anchor
    x_anchor = 10.0**xcm

    def _plot_suberlak_projection(
        c_sigma,
        c_tau,
        *,
        c_sigma_err,
        c_tau_err,
        color,
        label,
    ):
        suberlak_denom = alpha_agn_L * c_sigma + beta_agn_L * c_tau
        if not np.isfinite(suberlak_denom) or np.abs(suberlak_denom) < 1e-8:
            print(
                "[WARNING] Skipping Suberlak+2021 overlay because the converted "
                f"slope denominator is invalid: {suberlak_denom}"
            )
            return
        sub_slope = -0.4 / suberlak_denom
        y_central = L_anchor * (x_grid / x_anchor) ** sub_slope

        rng = np.random.default_rng(42)
        c_sigma_samps = rng.normal(
            loc=c_sigma,
            scale=c_sigma_err,
            size=500,
        )
        c_tau_samps = rng.normal(
            loc=c_tau,
            scale=c_tau_err,
            size=500,
        )
        denom_samps = alpha_agn_L * c_sigma_samps + beta_agn_L * c_tau_samps
        valid_sub_samps = np.isfinite(denom_samps) & (np.abs(denom_samps) >= 1e-8)
        sub_slope_samps = -0.4 / denom_samps[valid_sub_samps]
        if sub_slope_samps.size == 0:
            return
        curves = L_anchor * (x_grid[None, :] / x_anchor) ** sub_slope_samps[:, None]
        sub_lo, sub_hi = np.percentile(curves, [16, 84], axis=0)
        ax.plot(x_grid, y_central, color=color, lw=2.0, zorder=10,
                label=label, linestyle='--')
        ax.fill_between(x_grid, sub_lo, sub_hi, color=color, alpha=0.25, zorder=8)

    _plot_suberlak_projection(
        c_sigma_suberlak,
        c_tau_suberlak,
        c_sigma_err=c_sigma_suberlak_err,
        c_tau_err=c_tau_suberlak_err,
        color='c',
        label='Suberlak+2021 relation',
    )
    # _plot_suberlak_projection(
    #     c_sigma_suberlak - 0.4 * d_sigma_suberlak,
    #     c_tau_suberlak - 0.4 * d_tau_suberlak,
    #     c_sigma_err=c_sigma_suberlak_err,
    #     c_tau_err=c_tau_suberlak_err,
    #     color='darkorange',
    #     label=r'Suberlak+2021, fixed $\lambda_{\rm Edd}$',
    # )

    # ========= HIGHLIGHT: compute EVERYTHING from df_calibrators =========
    if df_calibrators is not None and len(df_calibrators) > 0:
        ds = df_calibrators.copy()

        M2500_show = ds['apparent_mag_2500'].values - cosmo.distmod(ds['z'].values).value
        if debias:
            M2500_show -= evaluate_dm_interp(
                dm_interp,
                ds["z"].values,
                ds["apparent_mag_2500"].values,
                f_host_2500=ds.get("f_host_2500"),
                alpha_lambda=ds.get("alpha_lambda"),
            )
        actual_logL2500_show = convert_M2500_to_logL2500(M2500_show)
        y_log_meas_err_show = 0.4 * np.asarray(ds['apparent_mag_2500_err'].fillna(0.0), dtype=float)
        yerr_linear_show = (10.0**actual_logL2500_show) * np.log(10.0) * y_log_meas_err_show

        # x for SHOW at median params, using the same AGN-sample pivots as the fit.
        obs_show, err_show, _ = agn_model_pack_obs(ds, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"])
        x_log_ref_show = -0.4 * (
            M_model_agn(
                med_arr,
                obs_show,
                agn_pivot_arr,
                use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
            )
            - M0_med
        )
        x_show = 10.0 ** x_log_ref_show

        pred_M_err_show = M_model_agn_err(
            med_arr,
            obs_show,
            err_show,
            agn_pivot_arr,
            use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        )
        x_log_err_show = 0.4 * pred_M_err_show
        x_lower_show = 10.0 ** (x_log_ref_show - x_log_err_show)
        x_upper_show = 10.0 ** (x_log_ref_show + x_log_err_show)

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
        ax.set_xlim((1.2e-3, 1.25e2))
        ax.set_ylim((np.min(10**ylog_med), np.max(10**ylog_med)))
    else:
        # ax.set_xlim((7e-8, 9e5))
        # ax.set_ylim((3e42, 2e47))
        ax.set_xlim((1.2e-3, 1.25e2))
        ax.set_ylim((np.min(10**ylog_med), np.max(10**ylog_med)))

    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))

    # x label (from MAIN pivots; just a label)
    obs_arr, err_arr, pivots_arr = agn_model_pack_obs(
        df_agn, use_alpha_lambda_term=option_flags["use_alpha_lambda_term"]
    )
    log_sigma_uv_pivot  = pivots_arr[agn_model_oidx["log_sigma_uv"]]
    log_tau_uv_rf_pivot = pivots_arr[agn_model_oidx["log_tau_uv_rf"]]
    sigma_uv_pivot  = 10.0 ** log_sigma_uv_pivot
    tau_uv_rf_pivot = 10.0 ** log_tau_uv_rf_pivot
    xlabel = rf"$({{\sigma}}_\mathrm{{uv}} \, / \, {sigma_uv_pivot:.1f}\,\mathrm{{mag}})^{{{alpha_agn_L:.2f}}} \, ({{\tau}}_\mathrm{{uv,rf}} \, / \, {tau_uv_rf_pivot:.0f}\,\mathrm{{days}})^{{{beta_agn_L:.2f}}}$"
    ax.set_xlabel(xlabel)
    ax.legend(loc='upper left')

    # --- Residuals panel (MAIN) ---
    sigma_meas = np.asarray(y_log_meas_err, dtype=float)
    slope_grid = np.gradient(ylog_med, x_log_grid)
    f_slope = interp1d(x_log_grid, slope_grid, bounds_error=False, fill_value='extrapolate')
    slope_at_data = f_slope(x_log_ref)
    sigma_x = np.asarray(x_log_err_med, dtype=float)
    sigma_xy = np.abs(slope_at_data) * np.abs(sigma_x)
    sigma_int_log = (
        np.exp(
            evaluate_log_f(
                med_params,
                np.asarray(d["z"].values, dtype=float),
                z_pivot=z_pivot_agn,
                use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
            )
        )
        / 2.5
    )
    sigma_chi_plot = np.sqrt(sigma_meas**2 + sigma_xy**2)
    sigma_chi_full = np.sqrt(sigma_meas**2 + sigma_xy**2 + sigma_int_log**2)
    if debias and dmi_selection_sigma_interp is not None:
        sigma_sel_mag = evaluate_dm_interp(
            dmi_selection_sigma_interp,
            d["z"].values,
            d["apparent_mag_2500"].values,
            f_host_2500=d.get("f_host_2500"),
            alpha_lambda=d.get("alpha_lambda"),
        )
        sigma_sel_mag = np.asarray(sigma_sel_mag, dtype=float)
        sigma_sel_valid = np.isfinite(sigma_sel_mag) & (sigma_sel_mag > 0.0)
        sigma_sel_log = np.full_like(sigma_sel_mag, np.nan, dtype=float)
        sigma_sel_log[sigma_sel_valid] = (
            np.maximum(sigma_sel_mag[sigma_sel_valid], float(sigma_sel_floor_mag))
            / 2.5
        )
        sigma_chi_plot = np.where(sigma_sel_valid, sigma_sel_log, sigma_chi_plot)
        sigma_chi_full = np.where(sigma_sel_valid, sigma_sel_log, sigma_chi_full)
    good_plot = (
        np.isfinite(residuals)
        & np.isfinite(sigma_chi_plot)
        & (sigma_chi_plot > 0)
    )
    good = (
        np.isfinite(residuals)
        & np.isfinite(sigma_chi_full)
        & (sigma_chi_full > 0)
    )

    if show_residuals and ax_res is not None:
        good_in = good & d["z"].between(z_range[0], z_range[1]).to_numpy(dtype=bool)
        good_out = good & ~d["z"].between(z_range[0], z_range[1]).to_numpy(dtype=bool)
        if np.any(good_in):
            good_in_plot = good_plot & d["z"].between(z_range[0], z_range[1]).to_numpy(dtype=bool)
            ax_res.errorbar(
                x_ref[good_in_plot],
                residuals[good_in_plot],
                yerr=sigma_chi_plot[good_in_plot],
                fmt='o',
                linestyle='none',
                markersize=2.8,
                mfc=(0, 0, 0, 0.4),
                mec="none",
                ecolor=(0.2, 0.2, 0.2, 0.18),
                elinewidth=0.6,
                capsize=0,
                zorder=5,
                label="AGN",
            )
        if np.any(good_out):
            good_out_plot = good_plot & ~d["z"].between(z_range[0], z_range[1]).to_numpy(dtype=bool)
            ax_res.errorbar(
                x_ref[good_out_plot],
                residuals[good_out_plot],
                yerr=sigma_chi_plot[good_out_plot],
                fmt='D',
                linestyle='none',
                markersize=2.8,
                mfc=(0, 0, 0, 0.4),
                mec="none",
                ecolor=(0.2, 0.2, 0.2, 0.18),
                elinewidth=0.6,
                capsize=0,
                zorder=6,
                label="outside z range",
            )

        ax_res.axhline(0, color='m', linestyle='--', zorder=3)
        ax_res.set_ylabel('Residuals (log)')
        ax_res.set_xlabel(xlabel)
        ax_res.set_xscale('log')
        ax_res.set_ylim(-1.0, 1.0)
        chi2_red_in = np.nan
        if np.any(good_in):
            chi2_red_in, _ = reduced_chi_squared(
                residuals[good_in],
                sigma_chi_full[good_in],
                n_params=len(model_labels) - 1,
            )
        if np.isfinite(chi2_red_in):
            ax_res.text(
                0.98,
                0.95,
                rf"$\chi^2_\nu={chi2_red_in:.2f}$",
                color="red",
                ha="right",
                va="top",
                transform=ax_res.transAxes,
            )


    # Save & return
    os.makedirs(plot_path, exist_ok=True)
    out_pdf = "predicted_L2500_vs_fullcorr_band"
    if debias:
        out_pdf += "_debiased"
    if show_residuals:
        out_pdf += "_with_residuals"
    out_pdf += ".pdf"
    _save_figure(fig, os.path.join(plot_path, out_pdf), dpi=600, show=show)

    # Return MAIN residuals; show residuals can be computed externally if needed
    return residuals, sigma_chi_full

def dmi_from_pdet_only(m_obs, m_obs_err, p_det, m_grid, sigma_completeness, z, tiny=1e-12, plot_path=None):
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

    fig_mag, ax_mag = plt.subplots(figsize=(7, 5))
    sc_mag = ax_mag.scatter(m_obs, sigma2 * slope, c=m_obs_err, cmap='viridis', s=20, alpha=0.7, label='Objects')
    ax_mag.set_xlabel('Observed Magnitude (m_obs)')
    ax_mag.set_ylabel(r'$\Delta m = \sigma^2 \, \frac{d \ln p_{\rm det}}{dm}$')
    ax_mag.set_title('Completeness Correction vs Observed Magnitude')
    fig_mag.colorbar(sc_mag, ax=ax_mag, label='Magnitude Error (m_obs_err)')
    ax_mag.set_ylim(-1, 0.5)
    fig_mag.tight_layout()
    base_plot_path = plot_path or "plots/hubble"
    completeness_path = os.path.join(base_plot_path, "completeness")
    os.makedirs(completeness_path, exist_ok=True)
    _save_figure(fig_mag, os.path.join(completeness_path, "dmi_vs_mag.pdf"), dpi=300)


    # Plot vs redshift (assuming you have z array)
    fig_z, ax_z = plt.subplots(figsize=(7, 5))
    sc_z = ax_z.scatter(z, sigma2 * slope, c=m_obs_err, cmap='viridis', s=20, alpha=0.7)
    ax_z.set_xlabel('Redshift (z)')
    ax_z.set_ylabel(r'$\Delta m = \sigma^2 \, \frac{d \ln p_{\rm det}}{dm}$')
    ax_z.set_title('Completeness Correction vs Redshift')
    fig_z.colorbar(sc_z, ax=ax_z, label='Magnitude Error (m_obs_err)')
    ax_z.set_ylim(-1, 0.5)
    fig_z.tight_layout()
    _save_figure(fig_z, os.path.join(completeness_path, "dmi_vs_redshift.pdf"), dpi=300)
    return sigma2 * slope


def dmi_corr(
    m_obs, z_obs, m_obs_err,
    H_obs_s, mag_centers, z_centers,
    sigma_completeness, tiny=1e-12, plot_path=None
):
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

    fig_mag, ax_mag = plt.subplots(figsize=(7, 5))
    sc_mag = ax_mag.scatter(m_obs, sigma2 * slope, c=m_obs_err, cmap='viridis', s=20, alpha=0.7, label='Objects')
    ax_mag.set_xlabel('Observed Magnitude (m_obs)')
    ax_mag.set_ylabel(r'$\Delta m = \sigma^2 \, \frac{d \ln p_{\rm det}}{dm}$')
    ax_mag.set_title('Completeness Correction vs Observed Magnitude')
    fig_mag.colorbar(sc_mag, ax=ax_mag, label='Magnitude Error (m_obs_err)')
    ax_mag.set_ylim(-1, 0.5)
    fig_mag.tight_layout()
    base_plot_path = plot_path or "plots/hubble"
    completeness_path = os.path.join(base_plot_path, "completeness")
    os.makedirs(completeness_path, exist_ok=True)
    _save_figure(fig_mag, os.path.join(completeness_path, "dmi_vs_mag.pdf"), dpi=300)


    # Plot vs redshift (assuming you have z array)
    fig_z, ax_z = plt.subplots(figsize=(7, 5))
    sc_z = ax_z.scatter(z_obs, sigma2 * slope, c=m_obs_err, cmap='viridis', s=20, alpha=0.7)
    ax_z.set_xlabel('Redshift (z)')
    ax_z.set_ylabel(r'$\Delta m = \sigma^2 \, \frac{d \ln p_{\rm det}}{dm}$')
    ax_z.set_title('Completeness Correction vs Redshift')
    fig_z.colorbar(sc_z, ax=ax_z, label='Magnitude Error (m_obs_err)')
    ax_z.set_ylim(-1, 0.5)
    fig_z.tight_layout()
    _save_figure(fig_z, os.path.join(completeness_path, "dmi_vs_redshift.pdf"), dpi=300)
    return dmi


from scipy.special import logsumexp
from qvc.hubble.hubble_likelihood import log_likelihood
from matplotlib.colors import SymLogNorm
def _highest_weight_theta(results, plot_path=None):
    """
    Dynesty utils: pick the sample with the largest posterior weight.
    """
    w = np.exp(results.logwt - logsumexp(results.logwt))
    idx = int(np.argmax(w))
    return results.samples[idx]
def _blob_for_theta(theta, *, df_agn, df_pantheon, cosmo_model,
                    completeness_params, _sna_L, _sna_Lower, _sna_LogdetCov,
                    use_full_cov=True, plot_path=None):
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
    )
    z = df_agn['z'].values
    m_obs = df_agn['apparent_mag_2500'].values
    return blob, z, m_obs


def plot_Z_vs_z(z, Z, outdir=None, title_suffix="", plot_path=None):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.scatter(z, Z, s=12, alpha=0.55)
    ax.set_xlabel("Redshift (z)")
    ax.set_ylabel("integral (completeness)  Z = Φ(...) or ∫N×C")
    ax.set_title(f"Completeness integrals vs z {title_suffix}")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    base_plot_path = plot_path or outdir or "plots/hubble"
    completeness_path = os.path.join(base_plot_path, "completeness")
    os.makedirs(completeness_path, exist_ok=True)
    _save_figure(fig, os.path.join(completeness_path, "completeness_integrals_vs_z.pdf"), dpi=200)

def plot_dmi_vs_z(z, dmi, outdir=None, title_suffix="", plot_path=None):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    # Plot the sorted line so the redshift trend is readable.
    order = np.argsort(z)
    ax.plot(z[order], dmi[order], lw=1.4, alpha=0.9)
    ax.set_xlabel("Redshift (z)")
    ax.set_ylabel("dmi (mag)  = E[m|det] - m_obs")
    ax.set_title(f"Interpolated dmi vs z {title_suffix}")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    base_plot_path = plot_path or outdir or "plots/hubble"
    completeness_path = os.path.join(base_plot_path, "completeness")
    os.makedirs(completeness_path, exist_ok=True)
    _save_figure(fig, os.path.join(completeness_path, "dmi_vs_z.pdf"), dpi=200)

def _hard_limit_m50_per_object(completeness2d, mag_centers, z, plot_path=None):
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

def plot_completeness_map_with_m50(
    completeness2d, mag_centers, z_centers,
    df_agn, outdir=None, title="Completeness map with hard m50(z)", plot_path=None
):
    base_plot_path = plot_path or outdir or "plots/hubble"
    completeness_path = os.path.join(base_plot_path, "completeness")
    os.makedirs(completeness_path, exist_ok=True)
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

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    im = ax.imshow(C.T, origin="lower", aspect="auto",
                   extent=[mag_centers[0], mag_centers[-1], z_centers[0], z_centers[-1]],
                   vmin=0.0, vmax=1.0)
    ax.set_xlabel("Apparent Magnitude")
    ax.set_ylabel("Redshift")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("p(detect)")
    # overlay m50 curve
    ok = np.isfinite(m50_curve)
    if np.any(ok):
        ax.plot(m50_curve[ok], z_centers[ok], lw=2.2)
    fig.tight_layout()
    _save_figure(fig, os.path.join(completeness_path, "completeness_map_with_m50.pdf"), dpi=200)

def run_completeness_diagnostics(sampler_results, df_agn, df_pantheon,
                                 completeness_params, cosmo_model,
                                 _sna_L, _sna_Lower, _sna_LogdetCov,
                                 outdir="plots/completeness", plot_path=None,
                                 use_full_cov=True,
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
                                 use_full_cov=use_full_cov)
    Z   = np.asarray(blob[0], dtype=float)
    dmi = np.asarray(blob[1], dtype=float)
    _plot_path = plot_path or outdir
    plot_Z_vs_z(z, Z, outdir, title_suffix=title_note, plot_path=_plot_path)
    plot_dmi_vs_z(z, dmi, outdir, title_suffix=title_note, plot_path=_plot_path)

    completeness_model = completeness_params[0]
    if getattr(completeness_model, "mode", "2d") == "2d":
        completeness2d, mag_centers, z_centers, *_ = completeness_params
        plot_completeness_map_with_m50(completeness2d, mag_centers, z_centers, df_agn, outdir, plot_path=_plot_path)


def plot_residuals_vs_alphaOX(
    df_agn,
    residuals,
    residuals_err,
    show=False,
    plot_path="plots/hubble/appendix",
    nbins=6,
    binning="uniform",     # "quantile", "uniform", or pass explicit edges via nbins=array_like
    min_per_bin=4,           # hide bins with too few points
    z_range=(0.44, 3.16)
):
    """
    Plot residuals vs delta_alphaOX and alphaOX, colored by redshift, with binned means.

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
    z_all = np.asarray(df_agn["z"], dtype=float)
    y_all = np.asarray(residuals, dtype=float)
    yerr_all = np.asarray(residuals_err, dtype=float)

    def _plot_one(xcol, xerr_col, xlabel, filename):
        x = np.asarray(df_agn.get(xcol, np.full(len(df_agn), np.nan)), dtype=float)
        xerr = np.asarray(df_agn.get(xerr_col, np.full(len(df_agn), np.nan)), dtype=float)
        z = z_all.copy()
        y = y_all.copy()
        yerr = yerr_all.copy()

        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & np.isfinite(z)
        if np.isfinite(xerr).any():
            m &= np.isfinite(xerr) | np.isnan(xerr)
        x, xerr, y, yerr, z = x[m], xerr[m], y[m], yerr[m], z[m]

        fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Residuals (mag)")
        ax.axhline(0.0, color="magenta", linewidth=2, zorder=0)
        ax.set_ylim(-4.6, 3.9)
        ax.grid(True, alpha=0.25)

        if len(x) == 0:
            ax.text(
                0.5,
                0.5,
                f"No finite {xcol} values",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            fig.tight_layout()
            os.makedirs(plot_path, exist_ok=True)
            return _save_figure(fig, os.path.join(plot_path, filename), show=show)

        vmin, vmax = np.nanmin(z), np.nanmax(z)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
        elif vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = mpl.colormaps["viridis"]
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

        mask_in = (z > z_range[0]) & (z < z_range[1])
        n_pts = len(x)
        for i in range(n_pts):
            zi = z[i]
            ci = cmap(norm(zi))
            xi_err = xerr[i] if np.isfinite(xerr[i]) else None
            if mask_in[i]:
                mfc = ci
                mec = "none"
            else:
                mfc = "none"
                mec = ci

            label = "AGN" if i == n_pts - 1 else None
            ax.errorbar(
                x[i],
                y[i],
                xerr=xi_err,
                yerr=yerr[i],
                fmt="o",
                markersize=6,
                mfc=mfc,
                mec=mec,
                mew=0.9,
                ecolor=(0.5, 0.5, 0.5, 0.7),
                elinewidth=0.8,
                capsize=2,
                capthick=0.8,
                zorder=2,
                label=label,
            )

        edges = None
        if np.ndim(nbins) > 0:
            edges = np.asarray(nbins, dtype=float)
            if edges.ndim != 1 or edges.size < 2:
                raise ValueError("Explicit 'nbins' must be a 1D array of bin edges with size >= 2.")
        elif len(x) >= 2 and np.nanmax(x) > np.nanmin(x):
            if binning == "quantile":
                qs = np.linspace(0, 1, nbins + 1)
                edges = np.unique(np.quantile(x, qs))
            elif binning == "uniform":
                edges = np.linspace(x.min(), x.max(), nbins + 1)
            else:
                raise ValueError("binning must be 'quantile', 'uniform', or provide explicit edges via nbins.")

        if edges is not None and np.size(edges) >= 2:
            bx, by, by_sem, _ = _weighted_bin_stats(
                x,
                y,
                yerr,
                bins=edges,
                min_count=min_per_bin,
                center="mid",
            )
            if len(bx):
                ax.errorbar(
                    bx,
                    by,
                    yerr=by_sem,
                    fmt="o",
                    ms=6,
                    lw=2,
                    color="red",
                    mfc="red",
                    mew=1.2,
                    zorder=3,
                    label="Binned mean",
                )

        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(r"$z$")
        ax.legend(loc="lower right", frameon=True, framealpha=0.8)

        fig.tight_layout()
        os.makedirs(plot_path, exist_ok=True)
        return _save_figure(fig, os.path.join(plot_path, filename), show=show)

    delta_path = _plot_one(
        "delta_alphaOX",
        "delta_alphaOX_err",
        r"$\Delta\, \alpha_{\mathrm{OX}}$",
        "delta_alphaOX_residuals.pdf",
    )
    alpha_path = _plot_one(
        "alphaOX",
        "alphaOX_err",
        r"$\alpha_{\mathrm{OX}}$",
        "alphaOX_residuals.pdf",
    )
    return delta_path, alpha_path


def plot_debias_impact_diagnostics(
    df_agn,
    residuals_biased,
    residuals_debiased,
    *,
    plot_path="plots/hubble",
    show=False,
    nbins=10,
    min_count=6,
):
    """Plot the residual change induced by debiasing against key observables."""
    delta_residual = np.asarray(residuals_biased, dtype=float) - np.asarray(residuals_debiased, dtype=float)

    diagnostics = [
        ("z", "Redshift z"),
        ("apparent_mag_2500", r"$m_{2500}$"),
        ("f_host_2500", r"$f_{\rm host,2500}$"),
        ("alpha_lambda", r"$\alpha_{\lambda}$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), squeeze=False)
    axes = axes.ravel()

    for ax, (key, xlabel) in zip(axes, diagnostics):
        if key not in df_agn.columns:
            ax.axis("off")
            continue

        x = np.asarray(df_agn[key], dtype=float)
        m = np.isfinite(x) & np.isfinite(delta_residual)
        if np.count_nonzero(m) == 0:
            ax.axis("off")
            continue

        x_use = x[m]
        y_use = delta_residual[m]

        ax.scatter(
            x_use,
            y_use,
            s=12,
            alpha=0.35,
            color="tab:blue",
            linewidths=0,
            rasterized=True,
        )
        ax.axhline(0.0, color="magenta", lw=1.6, zorder=0)

        if np.nanmax(x_use) > np.nanmin(x_use):
            if key == "z":
                trend = build_smooth_trend_1d(
                    x_use,
                    y_use,
                    frac=0.25,
                    it=1,
                    min_points=max(int(min_count), 10),
                    fallback_bins=nbins,
                )
                x_grid = np.linspace(np.nanmin(x_use), np.nanmax(x_use), 300)
                ax.plot(x_grid, trend(x_grid), color="red", lw=2.0)
            else:
                # Horizontal banding is expected here because delta_residual remains
                # a redshift-only correction projected onto other observables.
                edges = np.linspace(np.nanmin(x_use), np.nanmax(x_use), nbins + 1)
                xmid = []
                ymed = []
                for i in range(len(edges) - 1):
                    lo = edges[i]
                    hi = edges[i + 1]
                    keep = (x_use >= lo) & (x_use < hi)
                    if i == len(edges) - 2:
                        keep = (x_use >= lo) & (x_use <= hi)
                    if np.count_nonzero(keep) >= min_count:
                        xmid.append(np.nanmedian(x_use[keep]))
                        ymed.append(np.nanmedian(y_use[keep]))
                if xmid:
                    ax.plot(xmid, ymed, color="red", lw=2.0)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\Delta$ residual = biased $-$ debiased (mag)")

    fig.tight_layout()
    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, "debias_impact_diagnostics.pdf"),
        dpi=200,
        show=show,
    )


def plot_redshift_bin_residual_summary(
    df_agn,
    residuals_biased,
    residuals_biased_err,
    residuals_debiased,
    residuals_debiased_err,
    *,
    plot_path="plots/hubble",
    show=False,
    z_bins=None,
):
    """Compare biased vs debiased residual summary statistics in redshift bins."""
    z = np.asarray(df_agn["z"], dtype=float)
    rb = np.asarray(residuals_biased, dtype=float)
    eb = np.asarray(residuals_biased_err, dtype=float)
    rd = np.asarray(residuals_debiased, dtype=float)
    ed = np.asarray(residuals_debiased_err, dtype=float)

    if z_bins is None:
        z_bins = np.array([0.3, 0.6, 0.9, 1.2, 1.6, 2.0, 2.5, 3.2, np.inf], dtype=float)
    else:
        z_bins = np.asarray(z_bins, dtype=float)

    rows = []
    for i in range(len(z_bins) - 1):
        lo = z_bins[i]
        hi = z_bins[i + 1]
        mask = (z >= lo) & (z < hi if np.isfinite(hi) else z >= lo)
        mask_b = mask & np.isfinite(rb) & np.isfinite(eb) & (eb > 0)
        mask_d = mask & np.isfinite(rd) & np.isfinite(ed) & (ed > 0)

        def _stats(resid, err, m):
            if np.count_nonzero(m) == 0:
                return (0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            r = resid[m]
            e = err[m]
            n = int(r.size)
            w = 1.0 / np.square(e)
            wsum = float(np.sum(w))
            mean = float(np.sum(w * r) / wsum) if wsum > 0 else np.nan
            mean_err = float(np.sqrt(1.0 / wsum)) if wsum > 0 else np.nan
            median = float(np.median(r))
            rms = float(np.sqrt(np.mean(r**2)))
            mad_sigma = float(1.4826 * np.median(np.abs(r - np.median(r))))
            dof = max(n - 1, 1)
            chi2_red = float(np.sum((r / e) ** 2) / dof)
            return (n, mean, mean_err, median, rms, mad_sigma, chi2_red)

        n_b, mean_b, mean_err_b, median_b, rms_b, mad_b, chi2_b = _stats(rb, eb, mask_b)
        n_d, mean_d, mean_err_d, median_d, rms_d, mad_d, chi2_d = _stats(rd, ed, mask_d)
        rows.append(
            {
                "z_lo": lo,
                "z_hi": hi,
                "z_mid": 0.5 * (lo + hi) if np.isfinite(hi) else lo + 0.1,
                "N_biased": n_b,
                "mean_biased": mean_b,
                "mean_err_biased": mean_err_b,
                "median_biased": median_b,
                "rms_biased": rms_b,
                "mad_sigma_biased": mad_b,
                "chi2_red_biased": chi2_b,
                "N_debiased": n_d,
                "mean_debiased": mean_d,
                "mean_err_debiased": mean_err_d,
                "median_debiased": median_d,
                "rms_debiased": rms_d,
                "mad_sigma_debiased": mad_d,
                "chi2_red_debiased": chi2_d,
            }
        )

    summary = pd.DataFrame(rows)
    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    os.makedirs(diagnostics_path, exist_ok=True)
    summary.to_csv(os.path.join(diagnostics_path, "redshift_bin_residual_summary.csv"), index=False)

    x = summary["z_mid"].to_numpy(dtype=float)
    fig, axes = plt.subplots(4, 1, figsize=(8.0, 12.0), sharex=True)

    axes[0].errorbar(
        x,
        summary["mean_biased"],
        yerr=summary["mean_err_biased"],
        color="tab:blue",
        marker="o",
        label="Biased",
    )
    axes[0].errorbar(
        x,
        summary["mean_debiased"],
        yerr=summary["mean_err_debiased"],
        color="tab:red",
        marker="o",
        label="Debiased",
    )
    axes[0].axhline(0.0, color="k", lw=1.0, alpha=0.7)
    axes[0].set_ylabel("Mean residual")
    axes[0].legend(frameon=False)

    axes[1].plot(x, summary["rms_biased"], color="tab:blue", marker="o", label="Biased")
    axes[1].plot(x, summary["rms_debiased"], color="tab:red", marker="o", label="Debiased")
    axes[1].set_ylabel("RMS residual")

    axes[2].plot(x, summary["mad_sigma_biased"], color="tab:blue", marker="o", label="Biased")
    axes[2].plot(x, summary["mad_sigma_debiased"], color="tab:red", marker="o", label="Debiased")
    axes[2].set_ylabel("1.4826 MAD")

    axes[3].plot(x, summary["chi2_red_biased"], color="tab:blue", marker="o", label="Biased")
    axes[3].plot(x, summary["chi2_red_debiased"], color="tab:red", marker="o", label="Debiased")
    axes[3].axhline(1.0, color="magenta", lw=1.5)
    axes[3].set_ylabel(r"$\chi^2_\nu$")
    axes[3].set_xlabel("Redshift bin midpoint")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, "redshift_bin_residual_summary.pdf"),
        dpi=200,
        show=show,
    )

def plot_Mi_relation(df_agn, plot_path=None):
    required_cols = {"z", "apparent_mag_2500", "LOGLBOL_CORRECTED"}
    if not required_cols.issubset(df_agn.columns):
        return None

    cosmo   = FlatLambdaCDM(H0=70, Om0=0.3)

    DL = cosmo.luminosity_distance(df_agn['z'].values).to(u.parsec).value
    M_i_my = df_agn['apparent_mag_2500'].values - 5.0 * (np.log10(DL) - 1)
    M_i_Wu_z2 = 91 - 2.5 * df_agn['LOGLBOL_CORRECTED']
    M_i_Wu_z2 = M_i_Wu_z2.mask(M_i_Wu_z2 > 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(M_i_my, M_i_Wu_z2, c=df_agn['z'], cmap='viridis', alpha=0.6, s=10)
    ax.plot([min(M_i_my), max(M_i_my)], [min(M_i_my), max(M_i_my)], color='red', linestyle='--', label='y=x')
    ax.set_xlabel(r'$M_i = m_{2500} - 5 \log_{10}(D_L/10 \text{ pc})$')
    ax.set_ylabel(r'$M_i$ (Wu & Shen 2022)')
    fig.colorbar(scatter, ax=ax, label='Redshift (z)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    base_plot_path = plot_path or "plots/hubble"
    diagnostics_path = os.path.join(base_plot_path, "diagnostics")
    os.makedirs(diagnostics_path, exist_ok=True)
    _save_figure(fig, os.path.join(diagnostics_path, "Mi_relation_comparison.pdf"), dpi=200)


def plot_adf_pvalue_g_diagnostic(
    df_agn,
    plot_path="plots/hubble",
    show=False,
    pvalue_col="adf_pvalue_g",
    alpha=0.05,
):
    """Plot g-band ADF p-value diagnostics against the non-stationary null."""
    if pvalue_col not in df_agn.columns:
        return None

    pvals = pd.to_numeric(df_agn[pvalue_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(pvals) & (pvals >= 0.0) & (pvals <= 1.0)
    pvals = pvals[mask]
    if pvals.size == 0:
        return None

    pvals = pvals[pvals > 0.0]
    if pvals.size == 0:
        return None

    n = pvals.size
    pmin = float(np.nanmin(pvals))
    left = 10 ** np.floor(np.log10(pmin))
    bins = np.logspace(np.log10(left), 0.0, 30)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(pvals, bins=bins, color="tab:green", alpha=0.85, edgecolor="white")
    if alpha > 0.0:
        ax.axvline(alpha, color="black", linestyle="--", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel(r"$g$-band ADF p-value")
    ax.set_ylabel("Count")
    ax.set_title(f"ADF p-value distribution in g band (N = {n})")
    fig.tight_layout()

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    os.makedirs(diagnostics_path, exist_ok=True)
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, "adf_pvalue_g_diagnostic.pdf"),
        dpi=200,
        show=show,
    )


def plot_spectral_fraction_vs_redshift(
    df_agn,
    plot_path="plots/hubble",
    show=False,
    nbins=12,
    min_bin_count=20,
    z_range=(0.44, 3.16),
    df_cut_sources=None,
    filename="spectral_fraction_vs_redshift.pdf",
    cut_thresholds=None,
):
    """Plot available spectral fractions against redshift."""
    required = {"z", "f_bc_3000", "f_fe_uv_3000", "f_host_2500"}
    if not required.issubset(df_agn.columns):
        return None

    def _prepare_spectral_fraction_frame(frame):
        frame = frame.copy()
        if "f_na" in frame.columns or "f_br" in frame.columns:
            f_na = (
                pd.to_numeric(frame["f_na"], errors="coerce")
                if "f_na" in frame.columns
                else pd.Series(0.0, index=frame.index)
            )
            f_br = (
                pd.to_numeric(frame["f_br"], errors="coerce")
                if "f_br" in frame.columns
                else pd.Series(0.0, index=frame.index)
            )
            frame["f_lines"] = f_na.fillna(0.0) + f_br.fillna(0.0)
            f_na_err = (
                pd.to_numeric(frame["f_na_err"], errors="coerce")
                if "f_na_err" in frame.columns
                else pd.Series(0.0, index=frame.index)
            )
            f_br_err = (
                pd.to_numeric(frame["f_br_err"], errors="coerce")
                if "f_br_err" in frame.columns
                else pd.Series(0.0, index=frame.index)
            )
            if "f_na_err" in frame.columns or "f_br_err" in frame.columns:
                frame["f_lines_err"] = np.hypot(
                    f_na_err.fillna(0.0).to_numpy(dtype=float),
                    f_br_err.fillna(0.0).to_numpy(dtype=float),
                )
        return frame

    df_plot = _prepare_spectral_fraction_frame(df_agn)
    df_cut_plot = (
        _prepare_spectral_fraction_frame(df_cut_sources)
        if df_cut_sources is not None
        else None
    )
    z = pd.to_numeric(df_plot["z"], errors="coerce").to_numpy(dtype=float)
    panel_specs = [
        ("f_bc_3000", r"$f_{\rm BC}$"),
        ("f_fe_uv_3000", r"$f_{\rm FeII}$"),
    ]
    if "f_lines" in df_plot.columns:
        panel_specs.append(("f_lines", r"$f_{\rm lines}$"))

    if "f_host_2500" in df_plot.columns:
        panel_specs.append(("f_host_2500", r"$f_{\rm host,2500\,\AA}$"))
    if len(panel_specs) == 2:
        return None

    fig, axes = plt.subplots(
        1,
        len(panel_specs),
        figsize=(5.6 * len(panel_specs), 4.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for i_ax, (ax, (col, ylabel)) in enumerate(zip(axes, panel_specs)):
        y = pd.to_numeric(df_plot[col], errors="coerce").to_numpy(dtype=float)
        yerr = None
        err_col = f"{col}_err"
        if err_col in df_plot.columns:
            yerr = pd.to_numeric(df_plot[err_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(z) & np.isfinite(y) & (y > 0.0)
        if yerr is not None:
            mask &= np.isfinite(yerr) & (yerr >= 0.0)
        if np.count_nonzero(mask) == 0:
            ax.text(0.5, 0.5, f"No finite {col}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        z_use = z[mask]
        y_use = y[mask]
        yerr_use = yerr[mask] if yerr is not None else None
        in_z = (z_use >= z_range[0]) & (z_use <= z_range[1])
        out_z = ~in_z

        if np.any(in_z):
            ax.errorbar(
                z_use[in_z],
                y_use[in_z],
                yerr=yerr_use[in_z] if yerr_use is not None else None,
                fmt="o",
                markersize=2.5,
                alpha=0.25,
                color="k",
                elinewidth=0.4,
                capsize=0,
                zorder=3,
                label=ylabel,
                rasterized=True,
            )
        if np.any(out_z):
            ax.errorbar(
                z_use[out_z],
                y_use[out_z],
                yerr=yerr_use[out_z] if yerr_use is not None else None,
                fmt="D",
                markersize=3.2,
                alpha=0.30,
                color="k",
                elinewidth=0.4,
                capsize=0,
                zorder=3,
                label=ylabel if not np.any(in_z) else None,
                rasterized=True,
            )

        if df_cut_plot is not None and col in df_cut_plot.columns:
            z_cut = pd.to_numeric(df_cut_plot["z"], errors="coerce").to_numpy(dtype=float)
            y_cut = pd.to_numeric(df_cut_plot[col], errors="coerce").to_numpy(dtype=float)
            yerr_cut = None
            err_col = f"{col}_err"
            if err_col in df_cut_plot.columns:
                yerr_cut = pd.to_numeric(df_cut_plot[err_col], errors="coerce").to_numpy(dtype=float)
            mask_cut = np.isfinite(z_cut) & np.isfinite(y_cut) & (y_cut > 0.0)
            if yerr_cut is not None:
                mask_cut &= np.isfinite(yerr_cut) & (yerr_cut >= 0.0)
            if np.any(mask_cut):
                yerr_plot = yerr_cut[mask_cut] if yerr_cut is not None else None
                ax.errorbar(
                    z_cut[mask_cut],
                    y_cut[mask_cut],
                    yerr=yerr_plot,
                    fmt="x",
                    markersize=4,
                    alpha=0.7,
                    color="#E74C3C",
                    elinewidth=0.7,
                    capsize=0,
                    zorder=2,
                    label="cut",
                    rasterized=True,
                )

        threshold = None if cut_thresholds is None else cut_thresholds.get(col)
        if threshold is not None and np.isfinite(threshold):
            ax.axhline(
                threshold,
                color="gray",
                linestyle="--",
                linewidth=3,
                alpha=1,
                zorder=1,
                label="cut threshold",
            )

        ax.set_xlabel(r"$z$")
        ax.set_ylabel("Component fraction" if i_ax == 0 else "")
        ax.set_yscale("log")
        ax.set_ylim(4e-3, 8e0)
        ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    os.makedirs(diagnostics_path, exist_ok=True)
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )

def plot_sigma_bc_vs_redshift(
    df_agn,
    plot_path="plots/hubble",
    show=False,
    nbins=12,
    min_bin_count=20,
    filename="sigma_bc_vs_redshift.pdf",
):
    """Plot inferred BC variability amplitude against redshift."""

    if "z" not in df_agn.columns:
        raise KeyError("Missing required 'z' column for sigma_BC vs redshift plot.")

    log_sigma_bc = _derive_log_sigma_bc(df_agn)
    if log_sigma_bc is None:
        raise KeyError(
            "Missing required BC amplitude columns: need "
            "'log_sigma_uv'+'log_amp_delta_bc' or per-band 'amp_bc_<band>' with 'bc_weight_<band>'."
        )

    z = pd.to_numeric(df_agn["z"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(z) & np.isfinite(log_sigma_bc)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
    if np.any(mask):
        z_use = z[mask]
        y_use = log_sigma_bc[mask]
        ax.scatter(
            z_use,
            y_use,
            s=10,
            alpha=0.25,
            color="k",
            linewidths=0,
            rasterized=True,
        )

        if np.nanmax(z_use) > np.nanmin(z_use):
            edges = np.linspace(np.nanmin(z_use), np.nanmax(z_use), nbins + 1)
            xmid = []
            ymed = []
            ylo = []
            yhi = []
            for i in range(len(edges) - 1):
                lo = edges[i]
                hi = edges[i + 1]
                keep = (z_use >= lo) & (z_use < hi)
                if i == len(edges) - 2:
                    keep = (z_use >= lo) & (z_use <= hi)
                if np.count_nonzero(keep) < min_bin_count:
                    continue
                y_bin = y_use[keep]
                xmid.append(np.nanmedian(z_use[keep]))
                ymed.append(np.nanmedian(y_bin))
                ylo.append(np.nanpercentile(y_bin, 16))
                yhi.append(np.nanpercentile(y_bin, 84))
            if xmid:
                xmid = np.asarray(xmid, dtype=float)
                ymed = np.asarray(ymed, dtype=float)
                ylo = np.asarray(ylo, dtype=float)
                yhi = np.asarray(yhi, dtype=float)
                ax.fill_between(xmid, ylo, yhi, color="tab:blue", alpha=0.18, linewidth=0)
                ax.plot(xmid, ymed, color="tab:blue", lw=2.0)
    else:
        ax.text(0.5, 0.5, "No finite BC-amplitude values", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Redshift z")
    ax.set_ylabel(r"$\log \sigma_{\rm BC}$")
    ax.grid(True, alpha=0.2)

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    os.makedirs(diagnostics_path, exist_ok=True)
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_f_host_2500_vs_redshift(
    df_agn,
    plot_path="plots/hubble",
    show=False,
    nbins=12,
    min_bin_count=20,
):
    """Plot f_host_2500 against redshift."""
    required = {"z", "f_host_2500"}
    if not required.issubset(df_agn.columns):
        return None

    z = pd.to_numeric(df_agn["z"], errors="coerce").to_numpy(dtype=float)
    f_host_2500 = pd.to_numeric(df_agn["f_host_2500"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(z) & np.isfinite(f_host_2500)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.8))
    if np.count_nonzero(mask) == 0:
        ax.text(0.5, 0.5, "No finite f_host_2500", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        z_use = z[mask]
        y_use = f_host_2500[mask]

        ax.scatter(
            z_use,
            y_use,
            s=10,
            alpha=0.25,
            color="k",
            linewidths=0,
            rasterized=True,
        )

        if np.nanmax(z_use) > np.nanmin(z_use):
            edges = np.linspace(np.nanmin(z_use), np.nanmax(z_use), nbins + 1)
            xmid = []
            ymed = []
            ylo = []
            yhi = []
            for i in range(len(edges) - 1):
                lo = edges[i]
                hi = edges[i + 1]
                keep = (z_use >= lo) & (z_use < hi)
                if i == len(edges) - 2:
                    keep = (z_use >= lo) & (z_use <= hi)
                if np.count_nonzero(keep) < min_bin_count:
                    continue
                y_bin = y_use[keep]
                xmid.append(np.nanmedian(z_use[keep]))
                ymed.append(np.nanmedian(y_bin))
                ylo.append(np.nanpercentile(y_bin, 16))
                yhi.append(np.nanpercentile(y_bin, 84))
            if xmid:
                xmid = np.asarray(xmid, dtype=float)
                ymed = np.asarray(ymed, dtype=float)
                ylo = np.asarray(ylo, dtype=float)
                yhi = np.asarray(yhi, dtype=float)
                ax.fill_between(xmid, ylo, yhi, color="tab:blue", alpha=0.18, linewidth=0)
                ax.plot(xmid, ymed, color="tab:blue", lw=2.0)

        ax.set_xlabel("Redshift z")
        ax.set_ylabel(r"$f_{\rm host,2500}$")
        ax.grid(True, alpha=0.2)

    fig.tight_layout()

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    os.makedirs(diagnostics_path, exist_ok=True)
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, "f_host_2500_vs_redshift.pdf"),
        dpi=200,
        show=show,
    )


def plot_sigma_bc_vs_frac_bc(
    df_agn,
    plot_path="plots/hubble",
    show=False,
    nbins=12,
    min_bin_count=20,
    filename="sigma_bc_vs_frac_bc.pdf",
):
    """Plot inferred BC variability amplitude against the spectral BC fraction."""

    if "f_bc_3000" not in df_agn.columns:
        raise KeyError("Missing required 'f_bc_3000' column for sigma_BC vs f_BC plot.")

    log_sigma_bc = _derive_log_sigma_bc(df_agn)
    if log_sigma_bc is None:
        raise KeyError(
            "Missing required BC amplitude columns: need "
            "'log_sigma_uv'+'log_amp_delta_bc' or per-band 'amp_bc_<band>' with 'bc_weight_<band>'."
        )

    f_bc = pd.to_numeric(df_agn["f_bc_3000"], errors="coerce").to_numpy(dtype=float)
    z = (
        pd.to_numeric(df_agn["z"], errors="coerce").to_numpy(dtype=float)
        if "z" in df_agn.columns
        else np.full(len(df_agn), np.nan, dtype=float)
    )
    mask = np.isfinite(f_bc) & (f_bc > 0.0) & np.isfinite(log_sigma_bc)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
    if np.any(mask):
        x = np.log10(f_bc[mask])
        y = log_sigma_bc[mask]
        z_use = z[mask]

        use_color = np.all(np.isfinite(z_use))
        sc = ax.scatter(
            x,
            y,
            c=z_use if use_color else "0.3",
            cmap="viridis" if use_color else None,
            s=12,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )
        if use_color:
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("Redshift z")

        if np.nanmax(x) > np.nanmin(x):
            edges = np.linspace(np.nanmin(x), np.nanmax(x), nbins + 1)
            xmid = []
            ymed = []
            ylo = []
            yhi = []
            for i in range(len(edges) - 1):
                lo = edges[i]
                hi = edges[i + 1]
                keep = (x >= lo) & (x < hi)
                if i == len(edges) - 2:
                    keep = (x >= lo) & (x <= hi)
                if np.count_nonzero(keep) < min_bin_count:
                    continue
                y_bin = y[keep]
                xmid.append(np.nanmedian(x[keep]))
                ymed.append(np.nanmedian(y_bin))
                ylo.append(np.nanpercentile(y_bin, 16))
                yhi.append(np.nanpercentile(y_bin, 84))
            if xmid:
                xmid = np.asarray(xmid, dtype=float)
                ymed = np.asarray(ymed, dtype=float)
                ylo = np.asarray(ylo, dtype=float)
                yhi = np.asarray(yhi, dtype=float)
                ax.fill_between(xmid, ylo, yhi, color="tab:blue", alpha=0.18, linewidth=0)
                ax.plot(xmid, ymed, color="tab:blue", lw=2.0)
    else:
        ax.text(0.5, 0.5, "No finite positive BC-amplitude / f_BC values", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel(r"$\log f_{\rm BC}$")
    ax.set_ylabel(r"$\log \sigma_{\rm BC}$")
    ax.grid(True, alpha=0.2)

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    os.makedirs(diagnostics_path, exist_ok=True)
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_bc_lag_vs_l2500(
    df_agn,
    plot_path="plots/hubble",
    show=False,
    nbins=12,
    min_bin_count=20,
    filename="bc_lag_vs_l2500.pdf",
):
    """Plot inferred rest-frame BC lag against fiducial 2500 A luminosity."""

    required = {"z", "apparent_mag_2500"}
    if not required.issubset(df_agn.columns):
        missing = ", ".join(sorted(required - set(df_agn.columns)))
        raise KeyError(f"Missing required columns for BC lag vs L2500 plot: {missing}")

    log_lag_bc_rf = _derive_log_lag_bc_rf(df_agn)
    if log_lag_bc_rf is None:
        raise KeyError("Missing required BC lag columns: need 'log_lag_bc_<band>_RF' or 'lag_bc_<band>'.")

    z = pd.to_numeric(df_agn["z"], errors="coerce").to_numpy(dtype=float)
    m2500 = pd.to_numeric(df_agn["apparent_mag_2500"], errors="coerce").to_numpy(dtype=float)
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    logL2500 = np.full(len(df_agn), np.nan, dtype=float)
    mask_lum = np.isfinite(z) & np.isfinite(m2500) & (z > 0.0)
    if np.any(mask_lum):
        M2500 = m2500[mask_lum] - cosmo.distmod(z[mask_lum]).value
        logL2500[mask_lum] = convert_M2500_to_logL2500(M2500)

    mask = np.isfinite(logL2500) & np.isfinite(log_lag_bc_rf)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
    if np.any(mask):
        x = logL2500[mask]
        y = log_lag_bc_rf[mask]
        z_use = z[mask]

        use_color = np.all(np.isfinite(z_use))
        sc = ax.scatter(
            x,
            y,
            c=z_use if use_color else "0.3",
            cmap="viridis" if use_color else None,
            s=12,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )
        if use_color:
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("Redshift z")

        if np.nanmax(x) > np.nanmin(x):
            edges = np.linspace(np.nanmin(x), np.nanmax(x), nbins + 1)
            xmid = []
            ymed = []
            ylo = []
            yhi = []
            for i in range(len(edges) - 1):
                lo = edges[i]
                hi = edges[i + 1]
                keep = (x >= lo) & (x < hi)
                if i == len(edges) - 2:
                    keep = (x >= lo) & (x <= hi)
                if np.count_nonzero(keep) < min_bin_count:
                    continue
                y_bin = y[keep]
                xmid.append(np.nanmedian(x[keep]))
                ymed.append(np.nanmedian(y_bin))
                ylo.append(np.nanpercentile(y_bin, 16))
                yhi.append(np.nanpercentile(y_bin, 84))
            if xmid:
                xmid = np.asarray(xmid, dtype=float)
                ymed = np.asarray(ymed, dtype=float)
                ylo = np.asarray(ylo, dtype=float)
                yhi = np.asarray(yhi, dtype=float)
                ax.fill_between(xmid, ylo, yhi, color="tab:blue", alpha=0.18, linewidth=0)
                ax.plot(xmid, ymed, color="tab:blue", lw=2.0)
    else:
        ax.text(0.5, 0.5, "No finite BC-lag/L2500 values", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel(r"$\log L_{2500}$")
    ax.set_ylabel(r"$\log \tau_{\rm BC,RF}$")
    ax.grid(True, alpha=0.2)

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    os.makedirs(diagnostics_path, exist_ok=True)
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_g_band_drift_slope_histograms(
    df_agn,
    *,
    slope_kind="mean",
    z_min=None,
    z_max=1.5,
    m2500_max=22.5,
    x_limit=8.0,
    plot_path="plots/hubble",
    show=False,
    filename=None,
):
    """Compare raw vs detrended g-band drift slopes in side-by-side histograms."""
    if slope_kind not in {"mean", "var"}:
        raise ValueError("slope_kind must be 'mean' or 'var'.")

    raw_col = f"g_raw_{slope_kind}_slope"
    resid_col = f"g_resid_{slope_kind}_slope"
    if raw_col not in df_agn.columns or resid_col not in df_agn.columns:
        return None

    mask = np.ones(len(df_agn), dtype=bool)
    z = pd.to_numeric(df_agn.get("z"), errors="coerce").to_numpy(dtype=float) if "z" in df_agn.columns else None
    if z is not None:
        mask &= np.isfinite(z)
        if z_min is not None and np.isfinite(z_min):
            mask &= z >= float(z_min)
        if np.isfinite(z_max):
            mask &= z < float(z_max)
    m2500 = (
        pd.to_numeric(df_agn.get("apparent_mag_2500"), errors="coerce").to_numpy(dtype=float)
        if "apparent_mag_2500" in df_agn.columns
        else None
    )
    if m2500 is not None and np.isfinite(m2500_max):
        mask &= np.isfinite(m2500) & (m2500 < float(m2500_max))

    raw = pd.to_numeric(df_agn.loc[mask, raw_col], errors="coerce").to_numpy(dtype=float)
    resid = pd.to_numeric(df_agn.loc[mask, resid_col], errors="coerce").to_numpy(dtype=float)
    raw = raw[np.isfinite(raw)]
    resid = resid[np.isfinite(resid)]
    if raw.size == 0 or resid.size == 0:
        return None

    scale = 1e-4
    raw_plot = raw / scale
    resid_plot = resid / scale

    combined = np.concatenate([raw_plot, resid_plot])
    if combined.size == 0:
        return None
    xmax = float(x_limit) if np.isfinite(x_limit) and x_limit > 0.0 else float(np.nanmax(np.abs(combined)))
    if (not np.isfinite(xmax)) or xmax <= 0.0:
        xmax = 1.0
    if np.nanmin(combined) == np.nanmax(combined) and not (np.isfinite(x_limit) and x_limit > 0.0):
        center = float(np.nanmin(combined))
        bins = np.linspace(center - 1.0, center + 1.0, 21)
    else:
        bins = np.linspace(-xmax, xmax, 31)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True)
    labels = [
        (axes[0], raw_plot, "no detrending"),
        (axes[1], resid_plot, "with detrending"),
    ]
    selection_label = []
    if z_min is not None and np.isfinite(z_min) and np.isfinite(z_max):
        selection_label.append(fr"${float(z_min):g} \leq z < {float(z_max):g}$")
    elif z_min is not None and np.isfinite(z_min):
        selection_label.append(fr"$z \geq {float(z_min):g}$")
    elif np.isfinite(z_max):
        selection_label.append(fr"$z < {float(z_max):g}$")
    if np.isfinite(m2500_max):
        selection_label.append(fr"$m_{{2500\,\mathrm{{\AA}}}} < {float(m2500_max):g}$")
    selection_text = "\n".join(selection_label)

    if slope_kind == "mean":
        xlabel = r"mean slope ($10^{-4}$ mag day$^{-1}$)"
    else:
        xlabel = r"variance slope ($10^{-4}$ mag$^2$ day$^{-1}$)"
    for ax, values, panel_label in labels:
        ax.hist(values, bins=bins, color="black", alpha=0.85, edgecolor="white")
        q16, q50, q84 = np.nanpercentile(values, [16.0, 50.0, 84.0])
        sigma = float(np.nanstd(values, ddof=1)) if values.size > 1 else np.nan
        ax.axvspan(q16, q84, color="0.5", alpha=0.2, zorder=0)
        ax.axvline(q50, color="0.25", linestyle="--", linewidth=1.4)
        ax.axvline(0.0, color="dodgerblue", linestyle="-", linewidth=2.0)
        ax.set_xlim(-xmax, xmax)
        ax.set_xlabel(xlabel)
        ax.text(
            0.03,
            0.95,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )
        if np.isfinite(sigma):
            ax.text(
                0.03,
                0.86,
                fr"$\sigma = {sigma:.2f}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )
        if selection_text:
            ax.text(
                0.97,
                0.95,
                selection_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.7", alpha=0.9),
            )
    axes[0].set_ylabel("Count")
    fig.tight_layout()

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    os.makedirs(diagnostics_path, exist_ok=True)
    if filename is None:
        filename = f"g_band_{slope_kind}_slope_histograms.pdf"
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_completeness_diagnostics(dmi_plot, z, m2500, integrals_max_w, plot_path="plots/hubble"):

    # Plot dmi vs z for the posterior-summary correction used in debiasing.
    dmi_interp = interp1d(z, dmi_plot, kind='nearest', bounds_error=False, fill_value='extrapolate')
    
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(z, -dmi_plot, marker="o", linestyle="none", label="AGN", color='k', alpha=0.5)

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\Delta m$ (mag)")
    
    ax.legend(frameon=True, loc="upper right", fontsize=12)
    fig.tight_layout()

    outdir = os.path.join(plot_path, "completeness")
    os.makedirs(outdir, exist_ok=True)

    fig.savefig(f"{outdir}/dmi_vs_z_posterior_median.pdf", dpi=300)
    plt.close(fig)

    # Plot dmi vs m2500 (apparent magnitude)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(m2500, -dmi_plot, alpha=0.5, s=20, color='k', label='AGN')

    ax.set_xlabel(r"Apparent magnitude $m_{2500}$ (mag)")
    ax.set_ylabel(r"$\Delta m$ (mag)")

    ax.legend(frameon=True, loc="upper right", fontsize=12)
    fig.tight_layout()

    fig.savefig(f"{outdir}/dmi_vs_m2500_posterior_median.pdf", dpi=300)
    plt.close(fig)

    # Plot log(integrals) vs redshift for highest-weight sample
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(z, integrals_max_w, s=16, alpha=0.3)
    ax.set_xlabel("Redshift (z)")
    ax.set_ylabel("integral  (completeness)")
    ax.set_title("Completeness integrals vs z — highest posterior weight sample")
    ax.grid(True)
    fig.tight_layout()
    _save_figure(fig, os.path.join(outdir, "integrals_vs_z_highest_weight.pdf"), dpi=150)

def plot_redshift_histograms(df_pantheon, df_agn,
                            plot_path="plots/hubble",
                            z_col_sn="zHD",
                            z_col_agn="z",
                            xscale="log",
                            bins=40,
                            z_range=(0.44, 3.16),
                            show=False):
    """
    Plot redshift histograms for SN (Pantheon) and AGN samples
    using a logarithmic redshift axis.
    """

    # --- SN ---
    z_sn = df_pantheon[z_col_sn].to_numpy()

    # --- AGN ---
    z_agn_all = df_agn[z_col_agn].to_numpy()
    z_agn_fid = df_agn[df_agn[z_col_agn].between(z_range[0], z_range[1])][z_col_agn].to_numpy()
    z_agn_restricted = df_agn[df_agn[z_col_agn].between(1.0, z_range[1])][z_col_agn].to_numpy()

    # Remove non-positive values
    z_all = np.concatenate([z_sn, z_agn_all])
    z_all = z_all[z_all > 0.01]

    # Log bins
    zmin = z_all.min()
    zmax = z_all.max()
    if xscale == "log":
        log_bins = np.logspace(np.log10(zmin), np.log10(zmax), bins)
    elif xscale == "linear":
        log_bins = np.linspace(zmin, zmax, bins)
    else:
        raise ValueError("xscale must be 'log' or 'linear'")

    def _decimal_log_tick(x, pos):
        if x <= 0:
            return ""
        return f"{x:g}"

    fig, ax = plt.subplots(figsize=(8,5))

    # SN
    ax.hist(
        z_sn,
        bins=log_bins,
        histtype="step",
        color="dodgerblue",
        linewidth=2.5,
        label="SN Ia (Pantheon+)"
    )

    # AGN full sample
    ax.hist(
        z_agn_all,
        bins=log_bins,
        histtype="step",
        linestyle="dotted",
        color="black",
        linewidth=2.8,
        label=r"AGN ($\mathit{plotted\ sample}$)"
    )

    # AGN fiducial sample
    ax.hist(
        z_agn_fid,
        bins=log_bins,
        histtype="step",
        linestyle="solid",
        color="0.4",
        linewidth=2.8,
        label=rf"AGN ($\mathit{{fiducial\ fitting\ sample}};\ {z_range[0]}<z<{z_range[1]}$)",
        zorder=-1
    )

    # AGN restricted sample
    ax.hist(
        z_agn_restricted,
        bins=log_bins,
        histtype="step",
        linestyle="--",
        color="0.7",
        linewidth=2.8,
        label=rf"AGN ($\mathit{{restricted\ fitting\ sample}};\ 1.0<z<{z_range[1]}$)",
        zorder=-2
    )

    ax.set_xscale(xscale)
    if xscale == "log":
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_major_formatter(FuncFormatter(_decimal_log_tick))
    ax.set_xlabel(r"$z$")
    ax.set_ylabel("Number")

    ax.legend(frameon=False, loc="upper left", fontsize=12)
    fig.tight_layout()
    os.makedirs(plot_path, exist_ok=True)
    _save_figure(fig, os.path.join(plot_path, "redshift_histograms.pdf"), dpi=600, show=show)


def plot_delta_m_flux_recal_vs_redshift(df_agn, plot_path="plots/hubble", show=False):
    """Plot the mean photometric flux-recalibration offset against redshift."""
    if "z" not in df_agn.columns or "delta_m_flux_recal" not in df_agn.columns:
        return None

    z = np.asarray(df_agn["z"], dtype=float)
    dm = np.asarray(df_agn["delta_m_flux_recal"], dtype=float)
    mask = np.isfinite(z) & np.isfinite(dm)
    if not np.any(mask):
        return None

    z = z[mask]
    dm = dm[mask]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(z, dm, s=10, alpha=0.35, color="tab:blue", linewidths=0, rasterized=True)

    if z.size >= 11:
        order = np.argsort(z)
        z_sorted = z[order]
        dm_sorted = dm[order]
        window = min(201, len(z_sorted))
        if window % 2 == 0:
            window -= 1
        window = max(11, window)
        dm_med = (
            pd.Series(dm_sorted)
            .rolling(window=window, center=True, min_periods=max(5, window // 5))
            .median()
            .to_numpy()
        )
        med_mask = np.isfinite(dm_med)
        ax.plot(z_sorted[med_mask], dm_med[med_mask], color="darkorange", lw=2.0, label="rolling median")
        ax.legend(loc="best", frameon=False)

    ax.axhline(0.0, color="k", lw=1.0, alpha=0.7)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\Delta m_{\mathrm{flux\ recal}}$")
    ax.grid(True, alpha=0.3)

    os.makedirs(plot_path, exist_ok=True)
    _save_figure(fig, os.path.join(plot_path, "delta_m_flux_recal_vs_redshift.pdf"), dpi=300, show=show)
    return fig


def plot_m2500_vs_z_colorpanels(
    df,
    df_keep=None,
    color_cols=("f_host_2500", "f_host_center", "f_bc_3000", "wrms"),
    xcol="z",
    ycol="apparent_mag_2500",
    z_range=None,
    cuts=None,
    log_color=True,
    color_clip=None,   # dict: {col: (vmin, vmax)} in displayed space (log space if log_color=True)
    cmap="viridis",
    figsize=(8, 8.5),
    s=12,
    alpha=0.7,
    thin=4,
):
    if thin and thin > 1:
        print(f"[m2500_vs_z] Warning: thinning displayed points by factor {thin}")

    cols = [xcol, ycol] + list(color_cols)
    id_col = "object_id" if "object_id" in df.columns else None
    if id_col is not None:
        cols = cols + [id_col]

    base = df[cols].copy().dropna(subset=[xcol, ycol])
    cuts = {} if cuts is None else cuts
    color_clip = {} if color_clip is None else color_clip

    keep_ids = None
    if df_keep is not None:
        if id_col is not None and id_col in df_keep.columns:
            keep_ids = set(df_keep[id_col].astype(str))
            base["_is_kept"] = base[id_col].astype(str).isin(keep_ids)
        else:
            keep_index = set(df_keep.index.tolist())
            base["_is_kept"] = base.index.isin(keep_index)
    else:
        base["_is_kept"] = True
    if z_range is not None:
        z_lo, z_hi = z_range
        base["_in_z_range"] = base[xcol].between(z_lo, z_hi)
    else:
        base["_in_z_range"] = True

    # Pretty colorbar labels
    label_map = {
        "f_host_2500": r"f_{\mathrm{host},2500}",
        "f_host_center": r"f_{\mathrm{host}}",
        "f_fe_uv_3000": r"f_{\mathrm{Fe\, II}}",
        "f_bc_3000": r"f_{\mathrm{BC}}",
        "wrms": r"\chi^2/\nu",
    }

    fig, axes = plt.subplots(len(color_cols), 1, figsize=figsize, sharex=True, sharey=True)
    if len(color_cols) == 1:
        axes = [axes]

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.4", markeredgecolor="0.4",
               markersize=6, linestyle="None", label="Kept in z-range"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="0.4", markeredgecolor="0.4",
               markersize=6, linestyle="None", label="Cut in z-range"),
    ]

    for ax, ccol in zip(axes, color_cols):
        d = base.dropna(subset=[ccol]).copy()
        if thin and thin > 1:
            d = d.iloc[::thin].copy()

        if log_color:
            d = d[d[ccol] > 0].copy()
            c_all = np.log10(d[ccol].to_numpy(dtype=float))
        else:
            c_all = d[ccol].to_numpy(dtype=float)

        keep = d["_is_kept"].to_numpy(dtype=bool)
        if ccol in cuts and cuts[ccol] is not None:
            lo, hi = cuts[ccol]
            if lo is not None:
                keep &= (d[ccol].to_numpy() >= lo)
            if hi is not None:
                keep &= (d[ccol].to_numpy() <= hi)

        in_z = d["_in_z_range"].to_numpy(dtype=bool)
        keep_in_z = keep & in_z
        keep_out_z = keep & (~in_z)
        cut_in_z = (~keep) & in_z
        cut_out_z = (~keep) & (~in_z)

        d_keep_in_z = d.iloc[keep_in_z]
        d_keep_out_z = d.iloc[keep_out_z]
        d_cut_in_z = d.iloc[cut_in_z]
        d_cut_out_z = d.iloc[cut_out_z]
        c_keep_in_z = c_all[keep_in_z]
        c_keep_out_z = c_all[keep_out_z]
        c_cut_in_z = c_all[cut_in_z]
        c_cut_out_z = c_all[cut_out_z]

        # Per-panel clipping
        clip_lo, clip_hi = color_clip.get(ccol, (None, None))
        c_keep_in_z_plot = c_keep_in_z.copy()
        c_keep_out_z_plot = c_keep_out_z.copy()
        c_cut_in_z_plot = c_cut_in_z.copy()
        c_cut_out_z_plot = c_cut_out_z.copy()
        if clip_lo is not None:
            c_keep_in_z_plot = np.clip(c_keep_in_z_plot, clip_lo, None)
            c_keep_out_z_plot = np.clip(c_keep_out_z_plot, clip_lo, None)
            c_cut_in_z_plot = np.clip(c_cut_in_z_plot, clip_lo, None)
            c_cut_out_z_plot = np.clip(c_cut_out_z_plot, clip_lo, None)
        if clip_hi is not None:
            c_keep_in_z_plot = np.clip(c_keep_in_z_plot, None, clip_hi)
            c_keep_out_z_plot = np.clip(c_keep_out_z_plot, None, clip_hi)
            c_cut_in_z_plot = np.clip(c_cut_in_z_plot, None, clip_hi)
            c_cut_out_z_plot = np.clip(c_cut_out_z_plot, None, clip_hi)

        # Colorbar limits from clipped all-points (keep+cut)
        c_all_plot = c_all.copy()
        if clip_lo is not None:
            c_all_plot = np.clip(c_all_plot, clip_lo, None)
        if clip_hi is not None:
            c_all_plot = np.clip(c_all_plot, None, clip_hi)

        vmin = clip_lo if clip_lo is not None else np.nanmin(c_all_plot)
        vmax = clip_hi if clip_hi is not None else np.nanmax(c_all_plot)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        color_keep_out_z = mpl.cm.get_cmap(cmap)(norm(c_keep_out_z_plot)) if len(c_keep_out_z_plot) else None
        color_cut_out_z = mpl.cm.get_cmap(cmap)(norm(c_cut_out_z_plot)) if len(c_cut_out_z_plot) else None

        print(
            f"[m2500_vs_z:{ccol}] kept_in_z={len(d_keep_in_z)} "
            f"cut_in_z={len(d_cut_in_z)} kept_out_z={len(d_keep_out_z)} cut_out_z={len(d_cut_out_z)}"
        )

        ax.scatter(
            d_keep_in_z[xcol], d_keep_in_z[ycol],
            c=c_keep_in_z_plot, cmap=cmap, norm=norm, s=s, alpha=alpha, marker="o", rasterized=True
        )
        ax.scatter(
            d_cut_in_z[xcol], d_cut_in_z[ycol],
            c=c_cut_in_z_plot, cmap=cmap, norm=norm, s=s, alpha=alpha, marker="D", rasterized=True
        )
        ax.scatter(
            d_keep_out_z[xcol],
            d_keep_out_z[ycol],
            c=color_keep_out_z,
            s=s,
            alpha=1.0,
            marker="D",
            linewidths=1.5,
            zorder=5,
            rasterized=True,
        )
        ax.scatter(
            d_cut_out_z[xcol],
            d_cut_out_z[ycol],
            c=color_cut_out_z,
            s=s,
            alpha=1.0,
            marker="D",
            linewidths=1.6,
            zorder=5,
            rasterized=True,
        )

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)

        base_label = label_map.get(ccol, ccol)
        cbar.set_label(rf"$\log_{{10}}({base_label})$" if log_color else base_label)

        ax.set_ylabel(r"$m_{2500\,\mathrm{\AA}}$")

    axes[0].legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=10)

    axes[-1].set_xlabel(xcol)
    axes[0].invert_yaxis()
    fig.tight_layout()
    return fig, axes
