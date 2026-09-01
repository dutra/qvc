import numpy as np
import os
import math
import re
import textwrap
import warnings
from ast import literal_eval
from dataclasses import dataclass

import corner
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.cosmology import FlatwCDM, FlatwpwaCDM, FlatLambdaCDM, Flatw0waCDM
from astropy.cosmology.realizations import Planck18
from astropy import units as u
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    AutoMinorLocator,
    FixedLocator,
    FormatStrFormatter,
    FuncFormatter,
    LogLocator,
    MultipleLocator,
    NullFormatter,
    NullLocator,
)
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import minimize_scalar
from scipy.stats import chi2 as chi2_distribution
from scipy.stats import gaussian_kde, kurtosis, norm, normaltest, probplot, skew, spearmanr
from tqdm import tqdm

from qvc.hubble.hubble_model import (
    AGN_ALPHA_LAMBDA_ERR,
    AGN_ALPHA_LAMBDA_PARAM,
    AgnPivotContext,
    M_model_agn,
    M_model_agn_err,
    M_model_agn_observable_variance_posterior,
    M_model_agn_posterior_samples,
    agn_model_oidx,
    agn_model_pack_obs,
    agn_model_pack_params,
    agn_model_pidx,
    agn_model_req_errs,
    agn_model_req_obs,
    evaluate_log_f,
    get_agn_model_spec,
    get_model_params,
    resolve_model_option_flags,
)
from qvc.hubble.hubble_likelihood import sigma_lens_from_dc, sigma_mu_from_z_err
from qvc.hubble.cuts import (
    AGN_TIER1_FIT_QUALITY_CUTS,
    LIGHT_CURVE_N_POINTS_EXCLUDED_BANDS,
    light_curve_point_count_series,
)
from qvc.light_curve.band_colors import BAND_COLORS as LIGHT_CURVE_BAND_COLORS
from qvc.hubble.sigma_tau_lambda_fit import (
    SDSS_LAMBDA_PIVOT,
    fit_sigma_tau_lambda_broken_pl,
    log_broken_pl,
    std_from_slope_cov,
)
from qvc.hubble.hubble_utils import (
    convert_M2500_to_logL2500,
    cosmo_model_label_latex,
    format_result_errors,
    reduced_chi_squared,
    sym_percentile,
)
from qvc.hubble.hubble_completeness_refactored import (
    COMPLETENESS_FHOST_COL,
    COMPLETENESS_MAG_COL,
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
_OUT_OF_RANGE_AGN_COLOR = "tab:green"
_OUT_OF_RANGE_AGN_MARKER_COLOR = mpl.colors.to_rgba(_OUT_OF_RANGE_AGN_COLOR, alpha=0.65)
_OUT_OF_RANGE_AGN_ERROR_COLOR = mpl.colors.to_rgba(_OUT_OF_RANGE_AGN_COLOR, alpha=0.3)
_COSMO_CORNER_LEGEND_FONTSIZE = 40


@dataclass(frozen=True)
class HubblePosteriorDrawSelection:
    """Posterior draws bound to their sample rows and AGN column order."""

    values: np.ndarray
    sample_indices: np.ndarray
    object_ids: tuple[str, ...]

    def __post_init__(self):
        values = np.asarray(self.values, dtype=float)
        sample_indices = np.asarray(self.sample_indices)
        object_ids = np.asarray(self.object_ids, dtype=object)
        if values.ndim != 2:
            raise ValueError(
                "Hubble posterior draw values must be two-dimensional; "
                f"got shape {values.shape}."
            )
        if (
            sample_indices.ndim != 1
            or np.issubdtype(sample_indices.dtype, np.bool_)
            or not np.issubdtype(sample_indices.dtype, np.integer)
        ):
            raise ValueError(
                "Hubble posterior draw sample_indices must be a "
                "one-dimensional integer array."
            )
        if object_ids.ndim != 1:
            raise ValueError(
                "Hubble posterior draw object_ids must be one-dimensional."
            )
        if values.shape != (sample_indices.size, object_ids.size):
            raise ValueError(
                "Hubble posterior draw values must have shape "
                f"({sample_indices.size}, {object_ids.size}); "
                f"got {values.shape}."
            )
        if np.any(~np.isfinite(values)):
            raise ValueError(
                "Hubble posterior draw values must contain only finite "
                "values."
            )
        if np.unique(sample_indices).size != sample_indices.size:
            raise ValueError(
                "Hubble posterior draw sample_indices must not contain "
                "duplicates."
            )

        values = values.copy()
        sample_indices = sample_indices.astype(int, copy=True)
        values.setflags(write=False)
        sample_indices.setflags(write=False)
        object_id_tuple = tuple(str(value) for value in object_ids)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "sample_indices", sample_indices)
        object.__setattr__(self, "object_ids", object_id_tuple)


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

_SHEN_2024_LAG_LUMINOSITY_RELATIONS = {
    "Hβ": {
        "intercept": 1.458,
        "slope": 0.41,
        "sigma_int": 0.32,
        "log_lum_unit": 44.0,
        "x_range": (42.87, 45.40),
        "label": "Shen et al. (2024)",
    },
    "Mg II": {
        "intercept": 2.086,
        "slope": 0.31,
        "sigma_int": 0.32,
        "log_lum_unit": 45.0,
        "x_range": (43.58, 46.28),
        "label": "Shen et al. (2024)",
    },
    "C IV": {
        "intercept": 1.840,
        "slope": 0.32,
        "sigma_int": 0.51,
        "log_lum_unit": 45.0,
        "x_range": (44.22, 46.95),
        "label": "Shen et al. (2024)",
    },
}

_BAND_COLORS = LIGHT_CURVE_BAND_COLORS.copy()

_BLR_LAG_KL_MIN = 0.05
_BLR_PDF_BANDS = ("u", "g", "r", "i")
_BLR_PDF_LINE_WAVELENGTHS = {
    "Lyα 1216": 1215.67,
    "C IV 1549": 1549.0,
    "C III] 1909": 1908.73,
    "Mg II 2798": 2798.0,
    "Hβ 4861": 4861.33,
    "Hα 6563": 6562.80,
    "BC": 3646.0,
}
_BLR_PDF_BAND_EDGES = {
    "u": (3000.0, 4000.0),
    "g": (4000.0, 5500.0),
    "r": (5500.0, 7000.0),
    "i": (7000.0, 8500.0),
}


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


def _add_blr_visibility_overlays(ax_top, ax_bottom, band, z_min, z_max):
    lo, hi = _BLR_PDF_BAND_EDGES[band]
    z_grid = np.linspace(0.0, 4.0, 600)
    y_text = ax_bottom.get_ylim()[0] + 0.3

    for line_name, lam0 in _BLR_PDF_LINE_WAVELENGTHS.items():
        lam_obs = lam0 * (1.0 + z_grid)
        in_band = (lam_obs >= lo + 200.0) & (lam_obs <= hi - 200.0)
        if not np.any(in_band):
            continue

        diff = np.diff(in_band.astype(int))
        entry_indices = np.where(diff == 1)[0] + 1
        exit_indices = np.where(diff == -1)[0] + 1
        if in_band[0]:
            entry_indices = np.insert(entry_indices, 0, 0)
        if in_band[-1]:
            exit_indices = np.append(exit_indices, len(z_grid) - 1)

        for entry, exit_idx in zip(entry_indices, exit_indices):
            z_start = float(z_grid[entry])
            z_end = float(z_grid[exit_idx])
            if not (z_min < z_end and z_start < z_max):
                continue
            ax_top.axvspan(z_start, z_end, color="m", alpha=0.2, zorder=-5)
            ax_bottom.axvspan(z_start, z_end, color="m", alpha=0.2, zorder=-5)
            ax_bottom.text(
                0.5 * (z_start + z_end),
                y_text,
                line_name,
                color="m",
                rotation=90,
                va="bottom",
                ha="center",
                fontsize=12,
            )


def plot_blr_diagnostics_summary(
    df_agn,
    *,
    plot_path="plots/hubble",
    show=False,
    filename="blr.pdf",
    z_range=(0.44, 3.16),
):
    """Plot the BLR and continuum amplitude summary versus redshift."""
    if df_agn is None or len(df_agn) == 0:
        return None

    required_columns = {"z", "log_sigma_uv"}
    for band in _BLR_PDF_BANDS:
        required_columns.update(
            {
                f"dlog_amp_blr_{band}",
                f"log_sigma_band_{band}",
            }
        )
    if not required_columns.issubset(df_agn.columns):
        return None

    z = pd.to_numeric(df_agn["z"], errors="coerce").to_numpy(dtype=float)
    z_finite = z[np.isfinite(z)]
    if z_finite.size == 0:
        return None

    fig, axes = plt.subplots(
        2,
        len(_BLR_PDF_BANDS),
        figsize=(4 * len(_BLR_PDF_BANDS), 7),
        sharex=True,
        sharey="row",
    )
    point_color = (0.0, 0.0, 0.0, 0.2)
    error_color = (0.2, 0.2, 0.2, 0.05)
    plotted_any = False
    z_in_range = (z >= z_range[0]) & (z <= z_range[1])

    def _plot_range_split_points(ax, y, yerr, valid_mask, *, label):
        in_mask = valid_mask & z_in_range
        out_mask = valid_mask & ~z_in_range
        if np.any(in_mask):
            ax.errorbar(
                z[in_mask],
                y[in_mask],
                yerr=yerr[in_mask] if yerr is not None else None,
                fmt="o",
                linestyle="none",
                markersize=4,
                mfc=point_color,
                mec="none",
                ecolor=error_color,
                elinewidth=0.8,
                capsize=2,
                capthick=0.8,
                zorder=1,
                label=label,
            )
        if np.any(out_mask):
            ax.errorbar(
                z[out_mask],
                y[out_mask],
                yerr=yerr[out_mask] if yerr is not None else None,
                fmt="D",
                linestyle="none",
                markersize=3,
                mfc=point_color,
                mec="none",
                ecolor=error_color,
                elinewidth=0.8,
                capsize=2,
                capthick=0.8,
                zorder=1,
            )

    for i, band in enumerate(_BLR_PDF_BANDS):
        ax_top = axes[0, i]
        ax_bottom = axes[1, i]

        blr = (
            pd.to_numeric(df_agn["log_sigma_uv"], errors="coerce")
            + pd.to_numeric(df_agn[f"dlog_amp_blr_{band}"], errors="coerce")
        ).to_numpy(dtype=float)
        blr_err = None
        blr_err_cols = ("log_sigma_uv_err", f"dlog_amp_blr_{band}_err")
        if all(col in df_agn.columns for col in blr_err_cols):
            blr_err = np.hypot(
                pd.to_numeric(df_agn["log_sigma_uv_err"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(df_agn[f"dlog_amp_blr_{band}_err"], errors="coerce").to_numpy(dtype=float),
            )
        cont = pd.to_numeric(df_agn[f"log_sigma_band_{band}"], errors="coerce").to_numpy(dtype=float)
        cont_err = None
        if f"log_sigma_band_{band}_err" in df_agn.columns:
            cont_err = pd.to_numeric(df_agn[f"log_sigma_band_{band}_err"], errors="coerce").to_numpy(dtype=float)

        blr_mask = np.isfinite(z) & np.isfinite(blr)
        if blr_err is not None:
            blr_mask &= np.isfinite(blr_err)
        cont_mask = np.isfinite(z) & np.isfinite(cont)
        if cont_err is not None:
            cont_mask &= np.isfinite(cont_err)
        plotted_any |= np.any(blr_mask) or np.any(cont_mask)

        if np.any(blr_mask):
            _plot_range_split_points(ax_top, blr, blr_err, blr_mask, label=f"AGN {band}")
        else:
            ax_top.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_top.transAxes)
        ax_top.legend(loc="upper right")
        if i == 0:
            ax_top.set_ylabel(r"$\sigma_\mathrm{BLR,\ band}$")

        if np.any(cont_mask):
            _plot_range_split_points(ax_bottom, cont, cont_err, cont_mask, label=f"AGN {band}")
        else:
            ax_bottom.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_bottom.transAxes)
        ax_bottom.legend(loc="upper right")
        ax_bottom.set_xlabel("z")
        ax_bottom.set_ylim(-2.5, 0.4)
        if i == 0:
            ax_bottom.set_ylabel(r"$\sigma_\mathrm{cont,\ band}$")

        _add_blr_visibility_overlays(
            ax_top,
            ax_bottom,
            band,
            float(np.nanmin(z_finite)),
            float(np.nanmax(z_finite)),
        )

    if not plotted_any:
        plt.close(fig)
        return None

    fig.tight_layout()
    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    os.makedirs(diagnostics_path, exist_ok=True)
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=600,
        show=show,
    )


def _population_scatter_offsets(scale, *, enabled=True, seed=1739):
    """Return deterministic display-only Normal(0, scale) offsets."""
    scale = np.asarray(scale, dtype=float)
    offsets = np.zeros_like(scale, dtype=float)
    if not enabled:
        return offsets
    valid = np.isfinite(scale) & (scale > 0.0)
    if np.any(valid):
        rng = np.random.default_rng(seed)
        offsets[valid] = rng.normal(0.0, scale[valid])
    return offsets


def _resolve_clipped_mask(df_like, clipped_mask):
    if clipped_mask is None:
        return None
    clipped_mask = np.asarray(clipped_mask, dtype=bool)
    if clipped_mask.ndim != 1 or clipped_mask.shape[0] != len(df_like):
        raise ValueError(
            f"clipped_mask must be a 1D boolean array of length {len(df_like)}, "
            f"got shape {clipped_mask.shape}."
        )
    return clipped_mask


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

    if {"log_sigma_uv", "dlog_amp_bc"}.issubset(df_agn.columns):
        log_sigma_uv = pd.to_numeric(df_agn["log_sigma_uv"], errors="coerce").to_numpy(dtype=float)
        dlog_amp_bc = pd.to_numeric(df_agn["dlog_amp_bc"], errors="coerce").to_numpy(dtype=float)
        return log_sigma_uv + dlog_amp_bc

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
    """Prefer an aligned direct dmi array; otherwise evaluate ``dm_interp``."""
    dmi = None
    if dmi_values is not None:
        dmi = np.asarray(dmi_values, dtype=float)
        if dmi.shape != (len(df_agn),):
            raise ValueError(
                f"dmi_values has shape {dmi.shape}, but expected {(len(df_agn),)}."
            )
        # Direct values are authoritative; do not silently replace them with a
        # separately interpolated correction.
        return dmi
    if dm_interp is None:
        raise ValueError("Need either dm_interp or dmi_values for debias=True.")

    dmi_interp = evaluate_dm_interp(
        dm_interp,
        df_agn["z"].values,
        df_agn[COMPLETENESS_MAG_COL].values,
        f_host_2500_psf=df_agn.get(COMPLETENESS_FHOST_COL),
        alpha_lambda=df_agn.get("alpha_lambda"),
    )
    return dmi_interp


def _resolve_selection_sigma_values(
    df_agn,
    *,
    dmi_selection_sigma=None,
    dmi_selection_sigma_interp=None,
    sigma_sel_floor_mag=0.05,
):
    sigma_sel = None
    if dmi_selection_sigma is not None:
        sigma_sel = np.asarray(dmi_selection_sigma, dtype=float)
        if sigma_sel.shape != (len(df_agn),):
            raise ValueError(
                "dmi_selection_sigma has shape "
                f"{sigma_sel.shape}, but expected {(len(df_agn),)}."
            )

    if sigma_sel is None and dmi_selection_sigma_interp is not None:
        sigma_sel_interp = evaluate_dm_interp(
            dmi_selection_sigma_interp,
            df_agn["z"].values,
            df_agn[COMPLETENESS_MAG_COL].values,
            f_host_2500_psf=df_agn.get(COMPLETENESS_FHOST_COL),
            alpha_lambda=df_agn.get("alpha_lambda"),
        )
        sigma_sel = sigma_sel_interp

    if sigma_sel is None:
        return None

    sigma_sel = np.asarray(sigma_sel, dtype=float)
    sigma_sel_valid = np.isfinite(sigma_sel) & (sigma_sel > 0.0)
    return np.where(
        sigma_sel_valid,
        np.maximum(sigma_sel, float(sigma_sel_floor_mag)),
        np.nan,
    )


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


def _build_blr_line_luminosity_maps(df, log_luminosity_shift):
    n_rows = len(df)
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
        luminosity_maps[line_name] = {
            "values": shifted_values,
            "errs": np.where(np.isfinite(errs) & (errs >= 0.0), errs, np.nan),
            "value_col": spec["value_col"],
            "err_col": spec["err_col"],
            "axis_label": spec["axis_label"],
        }

    return luminosity_maps


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
    logL2500_arr = _coerce_numeric_vector(logL2500_debiased, len(df))
    luminosity_maps = _build_blr_line_luminosity_maps(df, log_luminosity_shift)
    for suffix in ("", "2"):
        component = 1 if suffix == "" else 2
        for band in ("u", "g", "r", "i", "z"):
            amp_col = f"dlog_amp_blr{suffix}_{band}"
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
                    log_luminosity = np.nan
                    log_luminosity_err = np.nan
                    luminosity_col = None
                    luminosity_err_col = None
                    luminosity_axis_label = None
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
    line_df = line_df[line_df["component"] == 1].copy()
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
        x = pd.to_numeric(line_df["log_line_luminosity"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(line_df["log_lag_rf"], errors="coerce").to_numpy(dtype=float)
        xerr = np.abs(
            pd.to_numeric(line_df["log_line_luminosity_err"], errors="coerce").to_numpy(dtype=float)
        )
        yerr = np.abs(
            pd.to_numeric(line_df["log_lag_rf_err"], errors="coerce").to_numpy(dtype=float)
        )
        keep = np.isfinite(x) & np.isfinite(y)
        if np.any(keep):
            clipped_mask = None
            if "is_clipped" in line_df.columns:
                clipped_mask = np.asarray(line_df["is_clipped"], dtype=bool)[keep]
            x = x[keep]
            y = y[keep]
            xerr = np.where(np.isfinite(xerr[keep]), xerr[keep], 0.0)
            yerr = np.where(np.isfinite(yerr[keep]), yerr[keep], 0.0)
            ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                fmt="none",
                ecolor=(0.0, 0.0, 0.0, 0.10),
                elinewidth=0.35,
                capsize=0.0,
                zorder=2,
            )
            ax.scatter(
                x,
                y,
                s=10.0,
                marker="o",
                facecolors=(0.0, 0.0, 0.0, 0.18),
                edgecolors=(0.0, 0.0, 0.0, 0.26),
                linewidths=0.15,
                zorder=3,
            )
            if clipped_mask is not None and np.any(clipped_mask):
                ax.scatter(
                    x[clipped_mask],
                    y[clipped_mask],
                    s=20.0,
                    marker="o",
                    facecolors="tab:green",
                    edgecolors="tab:green",
                    linewidths=0.3,
                    zorder=4,
                    label="Clipped AGN" if line_name == "C IV" else None,
                )

    shen_relation = _SHEN_2024_LAG_LUMINOSITY_RELATIONS.get(line_name)
    if shen_relation is not None:
        x_grid = np.linspace(
            shen_relation["x_range"][0],
            shen_relation["x_range"][1],
            200,
            dtype=float,
        )
        y_grid = shen_relation["intercept"] + shen_relation["slope"] * (
            x_grid - shen_relation["log_lum_unit"]
        )
        sigma_int = float(shen_relation["sigma_int"])
        ax.fill_between(
            x_grid,
            y_grid - sigma_int,
            y_grid + sigma_int,
            color="tab:blue",
            alpha=0.12,
            linewidth=0.0,
            zorder=0,
        )
        ax.plot(
            x_grid,
            y_grid,
            color="tab:blue",
            linestyle="--",
            linewidth=1.6,
            label=shen_relation["label"],
            zorder=1,
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
    dm_interp=None,
    *,
    dmi_values=None,
    plot_path="plots/hubble",
    show=False,
    prob_thresh=0.9,
    lag_err_max=0.25,
    clipped_mask=None,
    use_alpha_lambda_term=None,
    use_eta_sigma_term=None,
    use_f_agn_psf_2500_sigmoid_term=None,
    use_f_agn_psf_2500_flux_fraction_term=None,
    use_redshift_log_f_term=None,
):
    """Plot BLR lag against line-matched debiased continuum luminosity."""
    if df_agn.empty or (dm_interp is None and dmi_values is None):
        return None
    required = {"z", "apparent_mag_2500"}
    if not required.issubset(df_agn.columns):
        return None

    option_flags = resolve_model_option_flags(
        cosmo_model,
        np.asarray(flat_samples).shape[1],
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_f_agn_psf_2500_sigmoid_term=use_f_agn_psf_2500_sigmoid_term,
        use_f_agn_psf_2500_flux_fraction_term=use_f_agn_psf_2500_flux_fraction_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    _, model_labels, _ = get_model_params(
        cosmo_model,
        only_agn=option_flags["only_agn"],
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
        use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
    )
    param_indices = {name: model_labels.index(name) for name in model_labels}
    med_params = {key: np.median(flat_samples[:, idx]) for key, idx in param_indices.items()}
    cosmo = _get_cosmo_from_params(cosmo_model, med_params, z_pivot_agn)

    z = pd.to_numeric(df_agn["z"], errors="coerce").to_numpy(dtype=float)
    m2500 = pd.to_numeric(df_agn["apparent_mag_2500"], errors="coerce").to_numpy(dtype=float)
    dmi = _resolve_debias_values(
        df_agn,
        dm_interp=dm_interp,
        dmi_values=dmi_values,
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
    clipped_mask = _resolve_clipped_mask(df_agn, clipped_mask)
    clipped_object_ids = set()
    if clipped_mask is not None and "object_id" in df_agn.columns:
        clipped_object_ids = set(df_agn.loc[clipped_mask, "object_id"].astype(str))

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
        if clipped_object_ids and "object_id" in line_df.columns:
            line_df = line_df.copy()
            line_df["is_clipped"] = line_df["object_id"].astype(str).isin(clipped_object_ids)
        _plot_blr_lag_line_panel(
            ax,
            line_df,
            line_name,
            x_suffix="(debiased)",
        )

    component_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color="k", label="BLR 1", markersize=6),
        Line2D(
            [0],
            [0],
            color="tab:blue",
            linestyle="--",
            linewidth=1.6,
            label="Shen et al. (2024)",
        ),
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
        Line2D(
            [0],
            [0],
            color="tab:blue",
            linestyle="--",
            linewidth=1.6,
            label="Shen et al. (2024)",
        ),
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


def plot_tier1_cuts_vs_redshift(
    df,
    plot_path="plots/hubble",
    show=False,
    filename="tier1_cuts_vs_redshift_precut.pdf",
):
    """Plot current spectra (top) and light-curve (bottom) Tier-1 cuts."""

    spectra_columns = (
        ("sed_reduced_chi2", r"SED $\chi^2_\nu$"),
        ("spectroscopy_reduced_chi2", r"Spectroscopy $\chi^2_\nu$"),
        ("joint_reduced_chi2", r"Joint $\chi^2_\nu$"),
        ("m_2500_dereddened_rhat", r"$m_{2500,\rm dered}$ R-hat"),
        ("m_2500_attenuated_model_rhat", r"$m_{2500,\rm atten}$ R-hat"),
    )
    light_curve_columns = (
        ("loo_chi2_eff", r"LC LOO $\chi^2_{\rm eff}$"),
        ("log_tau_uv_rf_rhat", r"$\log\,\tau_{\rm UV,RF}$ R-hat"),
        ("log_sigma_uv_rhat", r"$\log\,\sigma_{\rm UV}$ R-hat"),
    )
    thresholds = {
        column: upper
        for column, lower, upper in AGN_TIER1_FIT_QUALITY_CUTS
        if lower is None and upper is not None
    }

    fig = plt.figure(figsize=(18.5, 8.2))
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.46)
    top = outer[0].subgridspec(1, len(spectra_columns), wspace=0.34)
    bottom = outer[1].subgridspec(1, len(light_curve_columns), wspace=0.25)
    spectra_axes = [fig.add_subplot(top[0, index]) for index in range(len(spectra_columns))]
    light_curve_axes = [
        fig.add_subplot(bottom[0, index]) for index in range(len(light_curve_columns))
    ]

    z = (
        pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
        if "z" in df.columns
        else np.full(len(df), np.nan, dtype=float)
    )
    tick_candidates = np.array(
        [0.05, 0.1, 0.2, 0.5, 0.7, 1.0, 1.1, 1.2, 1.3, 1.5,
         2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0],
        dtype=float,
    )

    def _plot_panel(axis, column, title, color):
        threshold = thresholds.get(column)
        axis.set_title(title, fontsize=12, pad=7)
        axis.set_xlabel("Redshift", fontsize=11)
        axis.set_xlim(0.0, 5.0)
        axis.set_yscale("log")

        if column not in df.columns or threshold is None:
            axis.set_ylim(0.5, 5.0)
            message = "Column missing" if column not in df.columns else "Cut disabled"
            axis.text(0.5, 0.5, message, transform=axis.transAxes,
                      ha="center", va="center", color="0.4")
            return

        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(z) & np.isfinite(values) & (values > 0.0)
        passed = finite & (values <= threshold)
        failed = finite & (values > threshold)
        positive = values[finite]
        if positive.size:
            lower = min(float(np.min(positive)), float(threshold)) / 1.12
            upper = max(float(np.max(positive)), float(threshold)) * 1.12
        else:
            lower, upper = float(threshold) / 2.0, float(threshold) * 2.0
        axis.set_ylim(lower, upper)

        axis.scatter(z[passed], values[passed], s=9, color=color, alpha=0.34,
                     edgecolors="none", rasterized=True)
        axis.scatter(z[failed], values[failed], s=14, color="#A51C30", marker="x",
                     alpha=0.78, linewidths=0.7, rasterized=True)
        axis.axhline(threshold, color="#A51C30", lw=1.4, ls="--", zorder=1)
        axis.text(0.97, threshold, f"cut = {threshold:g}",
                  transform=axis.get_yaxis_transform(), ha="right", va="bottom",
                  fontsize=8.5, color="#A51C30")

        count = int(np.count_nonzero(finite))
        annotation = (
            f"N = {count:,}\npass = {100.0 * np.count_nonzero(passed) / count:.1f}%"
            if count else "No finite values"
        )
        axis.text(0.03, 0.96, annotation, transform=axis.transAxes,
                  ha="left", va="top", fontsize=8.5, color="0.25",
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72})

        ticks = tick_candidates[(tick_candidates >= lower) & (tick_candidates <= upper)]
        ticks = np.unique(np.append(ticks, float(threshold)))
        axis.yaxis.set_major_locator(FixedLocator(ticks))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.grid(axis="y", which="major", color="0.86", lw=0.65, ls=":")
        axis.tick_params(labelsize=9)

    for axis, (column, title) in zip(spectra_axes, spectra_columns):
        _plot_panel(axis, column, title, "#7A5195")
    for axis, (column, title) in zip(light_curve_axes, light_curve_columns):
        _plot_panel(axis, column, title, "#0072B2")

    fig.text(0.012, 0.705, "Spectra Tier 1 diagnostic (log scale)", rotation=90,
             va="center", ha="center", fontsize=12.5)
    fig.text(0.012, 0.275, "Light-curve Tier 1 diagnostic (log scale)", rotation=90,
             va="center", ha="center", fontsize=12.5)
    fig.suptitle("Tier 1 cuts versus redshift (pre-cut sample)", fontsize=14, y=0.985)
    fig.subplots_adjust(left=0.052, right=0.993, bottom=0.075, top=0.935)

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=220,
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
    required = {"log_sigma_uv", "log_sigma_uv_uncorrected", "z", "f_PL"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise KeyError(f"Missing required columns for sigma_uv host-correction plot: {missing}")

    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
    f_pl = pd.to_numeric(df["f_PL"], errors="coerce").to_numpy(dtype=float)
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
        & np.isfinite(f_pl)
        & np.isfinite(z)
        & (f_pl > 0.0)
    )
    log_f_pl = np.log10(f_pl[mask_right]) if np.any(mask_right) else np.array([])
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
            log_f_pl,
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
        axes[1].text(0.5, 0.5, "No valid f_PL values", ha="center", va="center", transform=axes[1].transAxes)
        cbar = fig.colorbar(sc_left, ax=axes.tolist())
    axes[1].axhline(0.0, color="k", ls="--", lw=1, alpha=0.8)
    axes[1].set_xlabel(r"$\log_{10}(f_{\rm PL})$")
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


def plot_sigma_uv_mpred_correction(
    df,
    alpha_agn,
    plot_path="plots/hubble",
    show=False,
    filename="sigma_uv_mpred_correction_postcut.pdf",
):
    """Plot the magnitude-level impact of the sigma_uv dilution correction."""

    required = {"log_sigma_uv", "log_sigma_uv_uncorrected", "z"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise KeyError(f"Missing required columns for sigma_uv M_pred correction plot: {missing}")

    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
    log_sigma_uv = pd.to_numeric(df["log_sigma_uv"], errors="coerce").to_numpy(dtype=float)
    log_sigma_uv_uncorrected = pd.to_numeric(
        df["log_sigma_uv_uncorrected"], errors="coerce"
    ).to_numpy(dtype=float)
    delta_log_sigma = log_sigma_uv - log_sigma_uv_uncorrected
    delta_m_pred = float(alpha_agn) * delta_log_sigma

    mask_main = np.isfinite(z) & np.isfinite(delta_m_pred)
    if not np.any(mask_main):
        raise ValueError("No finite rows available for sigma_uv M_pred correction diagnostics.")

    fig = plt.figure(figsize=(17, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.85], wspace=0.28)
    ax_z = fig.add_subplot(gs[0, 0])
    ax_fpl = fig.add_subplot(gs[0, 1], sharey=ax_z)
    ax_hist = fig.add_subplot(gs[0, 2])

    scatter = ax_z.scatter(
        z[mask_main],
        delta_m_pred[mask_main],
        c=z[mask_main],
        cmap="viridis",
        s=10,
        alpha=0.65,
        linewidths=0,
        rasterized=True,
    )
    ax_z.axhline(0.0, color="k", ls="--", lw=1, alpha=0.8)
    ax_z.set_xlabel("Redshift $z$")
    ax_z.set_ylabel(r"$\Delta M_{\rm pred}$ (mag)")
    ax_z.grid(True, alpha=0.25)

    if "f_PL" in df.columns:
        f_pl = pd.to_numeric(df["f_PL"], errors="coerce").to_numpy(dtype=float)
        mask_fpl = mask_main & np.isfinite(f_pl) & (f_pl > 0.0)
    else:
        f_pl = np.full(len(df), np.nan, dtype=float)
        mask_fpl = np.zeros(len(df), dtype=bool)

    if np.any(mask_fpl):
        ax_fpl.scatter(
            np.log10(f_pl[mask_fpl]),
            delta_m_pred[mask_fpl],
            c=z[mask_fpl],
            cmap="viridis",
            s=10,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )
        ax_fpl.grid(True, alpha=0.25)
    else:
        ax_fpl.text(
            0.5,
            0.5,
            "No valid $f_{\\rm PL}$ values",
            ha="center",
            va="center",
            transform=ax_fpl.transAxes,
        )
    ax_fpl.axhline(0.0, color="k", ls="--", lw=1, alpha=0.8)
    ax_fpl.set_xlabel(r"$\log_{10}(f_{\rm PL})$")

    delta_show = delta_m_pred[mask_main]
    ax_hist.hist(
        delta_show,
        bins=40,
        color="0.65",
        edgecolor="white",
        alpha=0.9,
    )
    median_delta = float(np.nanmedian(delta_show))
    p90_abs = float(np.nanpercentile(np.abs(delta_show), 90))
    ax_hist.axvline(0.0, color="k", ls="--", lw=1, alpha=0.8)
    ax_hist.axvline(median_delta, color="tab:red", lw=2.0, alpha=0.9)
    ax_hist.set_xlabel(r"$\Delta M_{\rm pred}$ (mag)")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(True, alpha=0.25)
    ax_hist.text(
        0.97,
        0.97,
        f"N={delta_show.size}\n"
        f"median={median_delta:.3f} mag\n"
        f"90% |Δ|={p90_abs:.3f} mag",
        ha="right",
        va="top",
        transform=ax_hist.transAxes,
        fontsize=11,
    )

    colorbar = fig.colorbar(scatter, ax=[ax_z, ax_fpl], pad=0.02)
    colorbar.set_label("Redshift z")

    fig.suptitle(
        r"$\Delta M_{\rm pred} = \alpha_{\rm agn}\,\Delta \log \sigma_{\rm UV}$"
        + f"   ({float(alpha_agn):.3f} × correction)",
        fontsize=16,
        y=1.02,
    )

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


def plot_linear_trend_vs_redshift(
    df,
    plot_path="plots/hubble",
    show=False,
    filename="linear_trend_vs_redshift.pdf",
):
    """Plot linear_trend against redshift for AGN diagnostics."""
    required = {"z", "linear_trend"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise KeyError(f"Missing required columns for linear_trend-vs-redshift plot: {missing}")

    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
    linear_trend = pd.to_numeric(df["linear_trend"], errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    mask = np.isfinite(z) & np.isfinite(linear_trend)
    if np.any(mask):
        ax.scatter(
            z[mask],
            linear_trend[mask],
            s=5,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No finite linear_trend values",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_xlabel("Redshift z")
    ax.set_ylabel("linear_trend")
    ax.grid(True, alpha=0.25)

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_mean_function_slope_vs_tau(
    df,
    plot_path="plots/hubble",
    show=False,
    filename="mean_function_slope_vs_tau.pdf",
    z_range=(0.44, 3.16),
):
    """Plot the fitted GP mean-function trend parameter against UV rest-frame tau."""
    if "linear_trend" not in df.columns:
        raise KeyError("Missing required column for mean-function-slope-vs-tau plot: linear_trend")
    tau_col = "log_tau_uv_rf" if "log_tau_uv_rf" in df.columns else ("log_tau_uv" if "log_tau_uv" in df.columns else None)
    if tau_col is None:
        raise KeyError("Missing required column for mean-function-slope-vs-tau plot: log_tau_uv_rf or log_tau_uv")
    if tau_col == "log_tau_uv" and "z" not in df.columns:
        raise KeyError("Missing required column for mean-function-slope-vs-tau plot: z")

    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float) if "z" in df.columns else np.full(len(df), np.nan)
    log_tau = pd.to_numeric(df[tau_col], errors="coerce").to_numpy(dtype=float)
    log_tau_err = (
        pd.to_numeric(df[f"{tau_col}_err"], errors="coerce").to_numpy(dtype=float)
        if f"{tau_col}_err" in df.columns else np.full(len(df), np.nan)
    )
    if tau_col == "log_tau_uv":
        log_tau = log_tau - np.log10(1.0 + z)

    tau = np.power(10.0, log_tau)
    tau_err = np.full((2, len(df)), np.nan, dtype=float)
    finite_tau_err = (
        np.isfinite(tau)
        & np.isfinite(log_tau)
        & np.isfinite(log_tau_err)
        & (tau > 0.0)
        & (log_tau_err >= 0.0)
    )
    tau_err[0, finite_tau_err] = np.clip(
        tau[finite_tau_err] - np.power(10.0, log_tau[finite_tau_err] - log_tau_err[finite_tau_err]),
        0.0,
        None,
    )
    tau_err[1, finite_tau_err] = np.clip(
        np.power(10.0, log_tau[finite_tau_err] + log_tau_err[finite_tau_err]) - tau[finite_tau_err],
        0.0,
        None,
    )

    slope = pd.to_numeric(df["linear_trend"], errors="coerce").to_numpy(dtype=float)
    slope_err = (
        pd.to_numeric(df["linear_trend_err"], errors="coerce").to_numpy(dtype=float)
        if "linear_trend_err" in df.columns else np.full(len(df), np.nan)
    )

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    finite = np.isfinite(tau) & (tau > 0.0) & np.isfinite(slope)
    if np.any(finite):
        err_mask = finite & np.all(np.isfinite(tau_err), axis=0) & np.isfinite(slope_err) & (slope_err >= 0.0)
        in_z = finite & np.isfinite(z) & (z >= z_range[0]) & (z <= z_range[1])
        out_z = finite & ~in_z
        for mask, marker, label in ((in_z, "o", "AGN"), (out_z, "D", None)):
            if not np.any(mask):
                continue
            marker_err = mask & err_mask
            if np.any(marker_err):
                ax.errorbar(
                    tau[marker_err],
                    slope[marker_err],
                    xerr=tau_err[:, marker_err],
                    yerr=slope_err[marker_err],
                    fmt=marker,
                    linestyle="none",
                    markersize=3,
                    mfc=(0, 0, 0, 0.4),
                    mec="none",
                    ecolor=(0.2, 0.2, 0.2, 0.1),
                    elinewidth=0.8,
                    capsize=2,
                    capthick=0.8,
                    rasterized=True,
                    zorder=1,
                    label=label,
                )
            marker_noerr = mask & ~err_mask
            if np.any(marker_noerr):
                ax.scatter(
                    tau[marker_noerr],
                    slope[marker_noerr],
                    s=10 if marker == "o" else 12,
                    marker=marker,
                    c="black",
                    alpha=0.4,
                    linewidths=0,
                    rasterized=True,
                    zorder=1,
                    label=label if not np.any(marker_err) else None,
                )

        log_tau_finite = np.log10(tau[finite])
        log_lo = np.nanmin(log_tau_finite)
        log_hi = np.nanmax(log_tau_finite)
        log_pad = 0.08 * max(log_hi - log_lo, 1e-6)
        ax.set_xlim(10.0 ** (log_lo - log_pad), 10.0 ** (log_hi + log_pad))

        y_lo = np.nanmin(slope[finite])
        y_hi = np.nanmax(slope[finite])
        if np.isfinite(y_lo) and np.isfinite(y_hi):
            y_span = max(y_hi - y_lo, 1e-8)
            ax.set_ylim(y_lo - 0.12 * y_span, y_hi + 0.12 * y_span)
    else:
        ax.text(0.5, 0.5, "No finite linear_trend/tau values", ha="center", va="center", transform=ax.transAxes)

    ax.axhline(0.0, color="0.45", lw=1.0, alpha=0.8, zorder=0)
    ax.set_xscale("log")
    ax.tick_params(axis="x", which="both", pad=8)
    ax.set_xlabel(r"$\tau_{\rm UV,RF}$ (days)")
    ax.set_ylabel("Mean-function slope (linear_trend)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, loc="best")

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


def plot_sigma_tau_vs_lambda_broken_pl_fit(
    df,
    *,
    plot_path="plots/hubble",
    show=False,
    filename="sigma_tau_vs_lambda_broken_pl_fit_postcut.pdf",
    bands=("u", "g", "r", "i", "z"),
    lam_s=2500.0,
    ds_fixed_sigma=0.1,
    ds_fixed_tau=0.1,
    min_points=3,
):
    """Plot UV-subtracted per-band sigma/tau versus rest wavelength with broken power-law fits."""
    try:
        fit_result = fit_sigma_tau_lambda_broken_pl(
            df,
            bands=bands,
            lam_s=lam_s,
            ds_fixed_sigma=ds_fixed_sigma,
            ds_fixed_tau=ds_fixed_tau,
            min_points=min_points,
            include_plot_payload=True,
        )
    except (KeyError, ValueError) as exc:
        print(
            "[WARNING] Skipping sigma/tau wavelength broken power-law diagnostic: "
            f"{exc}"
        )
        return None

    sigma_data = fit_result["sigma_data"]
    tau_data = fit_result["tau_data"]
    fit_sigma = fit_result["fit_sigma"]
    fit_tau = fit_result["fit_tau"]

    xgrid = np.linspace(2, 5, 600)
    xgrid = np.sort(np.unique(np.append(xgrid, np.log10(float(lam_s)))))
    lam_grid = 10.0**xgrid
    yfit_sigma = fit_sigma["intercept"] + log_broken_pl(
        lam_grid,
        lam_s,
        fit_sigma["d1"],
        fit_sigma["d2"],
        ds_fixed_sigma,
    )
    yfit_tau = fit_tau["intercept"] + log_broken_pl(
        lam_grid,
        lam_s,
        fit_tau["d1"],
        fit_tau["d2"],
        ds_fixed_tau,
    )
    std_sigma = std_from_slope_cov(fit_sigma, lam_grid, lam_s=lam_s, ds_fixed=ds_fixed_sigma)
    std_tau = std_from_slope_cov(fit_tau, lam_grid, lam_s=lam_s, ds_fixed=ds_fixed_tau)

    fig, (ax_sigma, ax_tau) = plt.subplots(
        2,
        1,
        figsize=(6.2, 6.2),
        sharex=True,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.02)

    plotted_bands = []
    for band in bands:
        if band not in SDSS_LAMBDA_PIVOT:
            continue
        sigma_mask = sigma_data["band"] == band
        tau_mask = tau_data["band"] == band
        if np.any(sigma_mask):
            plotted_bands.append(band)
            ax_sigma.scatter(
                sigma_data["x"][sigma_mask],
                sigma_data["y_res"][sigma_mask],
                s=14,
                alpha=0.5,
                color=_BAND_COLORS.get(band),
                edgecolor="none",
                rasterized=True,
                zorder=1,
            )
        if np.any(tau_mask):
            ax_tau.scatter(
                tau_data["x"][tau_mask],
                tau_data["y_res"][tau_mask],
                s=14,
                alpha=0.6,
                color=_BAND_COLORS.get(band),
                edgecolor="none",
                rasterized=True,
                zorder=1,
            )

    if std_sigma is not None:
        ax_sigma.fill_between(
            xgrid,
            yfit_sigma - std_sigma,
            yfit_sigma + std_sigma,
            color="m",
            alpha=0.30,
            linewidth=0,
            zorder=4,
        )
    if std_tau is not None:
        ax_tau.fill_between(
            xgrid,
            yfit_tau - std_tau,
            yfit_tau + std_tau,
            color="m",
            alpha=0.30,
            linewidth=0,
            zorder=4,
        )
    ax_sigma.plot(xgrid, yfit_sigma, color="m", lw=1.6, zorder=5)
    ax_tau.plot(xgrid, yfit_tau, color="m", lw=1.6, zorder=5)

    for ax in (ax_sigma, ax_tau):
        ax.tick_params(direction="in", which="both", top=True, right=True, length=3, pad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    ax_sigma.set_ylabel(r"$\log(\sigma_{\mathrm{band}}/\sigma_{\mathrm{UV}})$")
    ax_tau.set_ylabel(r"$\log(\tau_{\mathrm{band,RF}}/\tau_{\mathrm{UV,RF}})$")
    ax_tau.set_xlabel(r"$\log_{10}\,\lambda_{\mathrm{RF}}\;(\mathrm{\AA})$")

    def _loglam_to_angstrom(x):
        return np.power(10.0, x)

    def _angstrom_to_loglam(x):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(x)

    secax = ax_sigma.secondary_xaxis("top", functions=(_loglam_to_angstrom, _angstrom_to_loglam))
    secax.set_xlabel(r"$\lambda_{\mathrm{RF}}\;(\mathrm{\AA})$")
    secax.tick_params(direction="in", which="major", top=True)
    lam_min, lam_max = float(10.0**xgrid.min()), float(10.0**xgrid.max())
    span = lam_max - lam_min
    candidates = np.array([500.0, 1000.0, 2000.0])
    step = float(candidates[np.argmin(np.abs(span / candidates - 4.0))])
    ticks_angstrom = np.arange(
        np.ceil(lam_min / step) * step,
        np.floor(lam_max / step) * step + 0.5 * step,
        step,
    )
    ticks_angstrom = ticks_angstrom[(ticks_angstrom >= lam_min) & (ticks_angstrom <= lam_max)]
    secax.xaxis.set_major_locator(FixedLocator(ticks_angstrom))
    secax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    secax.xaxis.set_minor_locator(NullLocator())

    band_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=6,
            markerfacecolor=_BAND_COLORS.get(band),
            markeredgecolor="none",
            label=f"{band}-band",
        )
        for band in dict.fromkeys(plotted_bands)
    ]
    model_handle = [Line2D([0], [0], color="m", lw=1.6, label="Population fit")]
    if band_handles:
        ax_sigma.legend(handles=band_handles + model_handle, loc="upper right", frameon=False, ncol=2, fontsize=9)

    ax_sigma.set_ylim(-0.54, 0.64)
    ax_tau.set_xlim(2.81, 3.89)
    ax_tau.set_ylim(-0.69, 0.64)

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=600,
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


def plot_eta_sigma_vs_redshift_colored_by_kl(
    df,
    plot_path="plots/hubble",
    show=False,
    filename="eta_sigma_vs_redshift_colored_by_kl.pdf",
    *,
    kl_color_limits=None,
    sample_label=None,
):
    """Plot eta_sigma versus redshift, colored by its approximate KL divergence."""
    required = {"z", "eta_sigma", "eta_sigma_kl"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise KeyError(f"Missing required columns for eta_sigma KL plot: {missing}")

    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
    eta_sigma = pd.to_numeric(df["eta_sigma"], errors="coerce").to_numpy(dtype=float)
    eta_sigma_kl = pd.to_numeric(df["eta_sigma_kl"], errors="coerce").to_numpy(dtype=float)
    if "eta_sigma_err" in df.columns:
        eta_sigma_err = pd.to_numeric(
            df["eta_sigma_err"], errors="coerce"
        ).to_numpy(dtype=float)
    else:
        eta_sigma_err = np.full(len(df), np.nan, dtype=float)

    mask = np.isfinite(z) & np.isfinite(eta_sigma) & np.isfinite(eta_sigma_kl)
    if not np.any(mask):
        raise ValueError("No finite z, eta_sigma, and eta_sigma_kl rows to plot.")
    z = z[mask]
    eta_sigma = eta_sigma[mask]
    eta_sigma_kl = eta_sigma_kl[mask]
    eta_sigma_err = eta_sigma_err[mask]

    if kl_color_limits is None:
        kl_vmin, kl_vmax = np.nanpercentile(eta_sigma_kl, [1.0, 99.0])
    else:
        if len(kl_color_limits) != 2:
            raise ValueError("kl_color_limits must contain exactly (vmin, vmax).")
        kl_vmin, kl_vmax = map(float, kl_color_limits)
    if not np.isfinite(kl_vmin) or not np.isfinite(kl_vmax):
        raise ValueError("KL color limits must be finite.")
    if kl_vmax <= kl_vmin:
        padding = max(0.05 * abs(kl_vmin), 0.05)
        kl_vmin -= padding
        kl_vmax += padding

    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    finite_err = np.isfinite(eta_sigma_err) & (eta_sigma_err >= 0.0)
    if np.any(finite_err):
        ax.errorbar(
            z[finite_err],
            eta_sigma[finite_err],
            yerr=eta_sigma_err[finite_err],
            fmt="none",
            ecolor="0.55",
            elinewidth=0.45,
            alpha=0.10,
            rasterized=True,
            zorder=1,
        )

    # Draw high-KL points last so the most data-informative fits remain visible.
    order = np.argsort(eta_sigma_kl)
    kl_norm = colors.Normalize(vmin=kl_vmin, vmax=kl_vmax, clip=False)
    points = ax.scatter(
        z[order],
        eta_sigma[order],
        c=eta_sigma_kl[order],
        cmap="viridis",
        norm=kl_norm,
        s=17,
        linewidths=0,
        alpha=0.82,
        rasterized=True,
        zorder=2,
    )

    edges = np.linspace(np.min(z), np.max(z), 13)
    if np.unique(edges).size > 1:
        centers = 0.5 * (edges[:-1] + edges[1:])
        bin_id = np.digitize(z, edges[1:-1])
        median = np.array(
            [
                np.median(eta_sigma[bin_id == index])
                if np.count_nonzero(bin_id == index) >= 10
                else np.nan
                for index in range(len(centers))
            ]
        )
        if np.any(np.isfinite(median)):
            ax.plot(centers, median, color="white", linewidth=3.2, zorder=3)
            ax.plot(
                centers,
                median,
                color="black",
                linewidth=1.35,
                marker="o",
                markersize=3.5,
                label="Binned median",
                zorder=4,
            )

    prior_mean = None
    if "eta_prior_profile" in df.columns:
        profiles = {
            str(value).strip().lower()
            for value in df.loc[mask, "eta_prior_profile"]
            if pd.notna(value)
        }
        if profiles == {"modified"}:
            prior_mean = -1.0
        elif profiles == {"default"}:
            prior_mean = -0.5
    if prior_mean is not None:
        ax.axhline(
            prior_mean,
            color="tab:red",
            linestyle="--",
            linewidth=1.1,
            label=rf"Prior location (${prior_mean:g}$)",
            zorder=0,
        )

    ax.set_xlabel("Redshift")
    ax.set_ylabel(r"$\eta_\sigma$")
    ax.set_title(r"Wavelength-dependence slope versus redshift")
    ax.grid(alpha=0.18, linewidth=0.6)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, loc="upper right")
    cbar = fig.colorbar(points, ax=ax, extend="both", pad=0.02)
    cbar.set_label(r"Approximate $D_{\mathrm{KL}}(q\,\Vert\,p)$ for $\eta_\sigma$")
    if sample_label:
        ax.text(
            0.015,
            0.02,
            f"{sample_label}: N = {len(z):,}",
            transform=ax.transAxes,
            fontsize=9,
            ha="left",
            va="bottom",
        )

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=300,
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
        warnings.warn(
            "Skipping optional fast-vs-UV diagnostic plot because the "
            f"following column(s) are unavailable: {', '.join(missing)}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

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


def plot_bpl_psd_vs_uv_variability(
    df,
    plot_path="plots/hubble",
    show=False,
    filename="bpl_psd_vs_uv_variability.pdf",
    max_log_tau_bpl_err=0.5,
    min_log_chi_sq_red_g=None,
    z_range=(0.44, 3.16),
):
    """Compare the displayed LS bending-power-law PSD fit against the main UV fit."""
    required = {"log_sigma_uv", "log_sigma_ls", "log_tau_ls"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise KeyError(f"Missing required columns for BPL PSD-vs-UV diagnostic plot: {missing}")
    tau_uv_col = "log_tau_uv_rf" if "log_tau_uv_rf" in df.columns else ("log_tau_uv" if "log_tau_uv" in df.columns else None)
    if tau_uv_col is None:
        raise KeyError("Missing required column for BPL PSD-vs-UV diagnostic plot: log_tau_uv_rf or log_tau_uv")

    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float) if "z" in df.columns else np.full(len(df), np.nan)
    log_sigma_uv = pd.to_numeric(df["log_sigma_uv"], errors="coerce").to_numpy(dtype=float)
    log_sigma_uv_err = (
        pd.to_numeric(df["log_sigma_uv_err"], errors="coerce").to_numpy(dtype=float)
        if "log_sigma_uv_err" in df.columns else np.full(len(df), np.nan)
    )
    log_sigma_bpl = pd.to_numeric(df["log_sigma_ls"], errors="coerce").to_numpy(dtype=float)
    log_sigma_bpl_err = (
        pd.to_numeric(df["log_sigma_ls_err"], errors="coerce").to_numpy(dtype=float)
        if "log_sigma_ls_err" in df.columns else np.full(len(df), np.nan)
    )
    alpha_high = (
        pd.to_numeric(df["alpha_high_ls"], errors="coerce").to_numpy(dtype=float)
        if "alpha_high_ls" in df.columns else np.full(len(df), -2.0, dtype=float)
    )
    slope = -alpha_high
    valid_slope = np.isfinite(slope) & (slope > 1.0)
    rms_factor = np.full(len(df), 1.0 / np.sqrt(2.0), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rms_factor[valid_slope] = np.sqrt(
            1.0 / (slope[valid_slope] * np.sin(np.pi / slope[valid_slope]))
        )
    rms_factor = np.where(np.isfinite(rms_factor) & (rms_factor > 0.0), rms_factor, 1.0 / np.sqrt(2.0))
    log_sigma_bpl_comparable = log_sigma_bpl + np.log10(rms_factor / np.sqrt(2.0 * np.pi))
    log_tau_uv = pd.to_numeric(df[tau_uv_col], errors="coerce").to_numpy(dtype=float)
    log_tau_uv_err_col = f"{tau_uv_col}_err"
    log_tau_uv_err = (
        pd.to_numeric(df[log_tau_uv_err_col], errors="coerce").to_numpy(dtype=float)
        if log_tau_uv_err_col in df.columns else np.full(len(df), np.nan)
    )
    log_tau_bpl = pd.to_numeric(df["log_tau_ls"], errors="coerce").to_numpy(dtype=float)
    log_tau_bpl_err = (
        pd.to_numeric(df["log_tau_ls_err"], errors="coerce").to_numpy(dtype=float)
        if "log_tau_ls_err" in df.columns else np.full(len(df), np.nan)
    )
    if tau_uv_col == "log_tau_uv" and "z" in df.columns:
        log_tau_uv = log_tau_uv - np.log10(1.0 + z)

    if "log_tau_uv" in df.columns:
        log_tau_uv_obs = pd.to_numeric(df["log_tau_uv"], errors="coerce").to_numpy(dtype=float)
        log_tau_uv_obs_err = (
            pd.to_numeric(df["log_tau_uv_err"], errors="coerce").to_numpy(dtype=float)
            if "log_tau_uv_err" in df.columns else log_tau_uv_err
        )
    else:
        log_tau_uv_obs = log_tau_uv + np.log10(1.0 + z)
        log_tau_uv_obs_err = log_tau_uv_err
    log_tau_bpl_obs = (
        pd.to_numeric(df["log_tau_ls_obs"], errors="coerce").to_numpy(dtype=float)
        if "log_tau_ls_obs" in df.columns else log_tau_bpl + np.log10(1.0 + z)
    )
    psd_valid = (
        pd.Series(df["psd_ls_valid"]).fillna(False).astype(bool).to_numpy()
        if "psd_ls_valid" in df.columns else np.ones(len(df), dtype=bool)
    )
    tau_bpl_well_constrained = (
        psd_valid
        & np.isfinite(log_tau_bpl_err)
        & (log_tau_bpl_err >= 0.0)
        & (log_tau_bpl_err <= float(max_log_tau_bpl_err))
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))

    def _linear_error_from_log(value, log_value, log_err):
        value = np.asarray(value, dtype=float)
        log_value = np.asarray(log_value, dtype=float)
        log_err = np.asarray(log_err, dtype=float)
        finite = (
            np.isfinite(value)
            & np.isfinite(log_value)
            & np.isfinite(log_err)
            & (value > 0.0)
            & (log_err >= 0.0)
        )
        lower = np.full(value.shape, np.nan, dtype=float)
        upper = np.full(value.shape, np.nan, dtype=float)
        lower[finite] = np.clip(value[finite] - np.power(10.0, log_value[finite] - log_err[finite]), 0.0, None)
        upper[finite] = np.clip(np.power(10.0, log_value[finite] + log_err[finite]) - value[finite], 0.0, None)
        return np.vstack([lower, upper])

    panels = [
        (
            axes[0],
            np.power(10.0, log_sigma_uv),
            np.power(10.0, log_sigma_bpl_comparable),
            _linear_error_from_log(np.power(10.0, log_sigma_uv), log_sigma_uv, log_sigma_uv_err),
            _linear_error_from_log(np.power(10.0, log_sigma_bpl_comparable), log_sigma_bpl_comparable, log_sigma_bpl_err),
            r"$\sigma_{\rm UV}$ (mag)",
            r"$\sigma_{\rm LS}$ (mag)",
            "No valid BPL sigma values",
            psd_valid & (np.power(10.0, log_sigma_bpl_comparable) > 5e-2),
            (2e-2, 2e0),
        ),
        (
            axes[1],
            np.power(10.0, log_tau_uv),
            np.power(10.0, log_tau_bpl),
            _linear_error_from_log(np.power(10.0, log_tau_uv), log_tau_uv, log_tau_uv_err),
            _linear_error_from_log(np.power(10.0, log_tau_bpl), log_tau_bpl, log_tau_bpl_err),
            r"$\tau_{\rm UV,RF}$ (days)",
            r"$\tau_{\rm LS,RF}$ (days)",
            "No well-constrained BPL tau values",
            tau_bpl_well_constrained & (np.power(10.0, log_sigma_bpl_comparable) > 5e-2),
            None,
        ),
    ]
    for ax, x, y, xerr, yerr, xlabel, ylabel, empty_label, panel_filter, fixed_axis_limits in panels:
        finite_mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0) & panel_filter
        if np.any(finite_mask):
            err_mask = finite_mask & np.all(np.isfinite(xerr), axis=0) & np.all(np.isfinite(yerr), axis=0)
            in_z = finite_mask & np.isfinite(z) & (z >= z_range[0]) & (z <= z_range[1])
            out_z = finite_mask & ~in_z
            for mask, marker, label in ((in_z, "o", "AGN"), (out_z, "D", None)):
                if not np.any(mask):
                    continue
                marker_err = mask & err_mask
                if np.any(marker_err):
                    ax.errorbar(
                        x[marker_err],
                        y[marker_err],
                        xerr=xerr[:, marker_err],
                        yerr=yerr[:, marker_err],
                        fmt=marker,
                        linestyle="none",
                        markersize=3,
                        mfc=(0, 0, 0, 0.4),
                        mec="none",
                        ecolor=(0.2, 0.2, 0.2, 0.1),
                        elinewidth=0.8,
                        capsize=2,
                        capthick=0.8,
                        rasterized=True,
                        zorder=1,
                        label=label,
                    )
                marker_noerr = mask & ~err_mask
                if np.any(marker_noerr):
                    ax.scatter(
                        x[marker_noerr],
                        y[marker_noerr],
                        s=10 if marker == "o" else 12,
                        marker=marker,
                        c="black",
                        alpha=0.4,
                        linewidths=0,
                        rasterized=True,
                        zorder=1,
                        label=label if not np.any(marker_err) else None,
                    )
            lo = min(np.nanmin(x[finite_mask]), np.nanmin(y[finite_mask]))
            hi = max(np.nanmax(x[finite_mask]), np.nanmax(y[finite_mask]))
            log_delta = np.log10(y[finite_mask]) - np.log10(x[finite_mask])
            log_delta = log_delta[np.isfinite(log_delta)]
            if log_delta.size:
                bias = float(np.mean(log_delta))
                sigma = float(np.std(log_delta))
                ax.text(
                    0.97,
                    0.03,
                    (
                        f"N = {log_delta.size}\n"
                        f"bias = {bias:.2f} dex\n"
                        f"$\\sigma$ = {sigma:.2f} dex"
                    ),
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=10.5,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.9),
                    zorder=20,
                )
        else:
            ax.text(0.5, 0.5, empty_label, ha="center", va="center", transform=ax.transAxes)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if np.any(finite_mask) and np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            if fixed_axis_limits is not None:
                axis_limits = fixed_axis_limits
            else:
                log_lo = np.log10(lo)
                log_hi = np.log10(hi)
                pad = 0.12 * max(log_hi - log_lo, 1e-6)
                axis_limits = (10.0 ** (log_lo - pad), 10.0 ** (log_hi + pad))
            ax.set_xlim(*axis_limits)
            ax.set_ylim(*axis_limits)
            ax.plot(axis_limits, axis_limits, color="m", ls="-", lw=2.2, alpha=0.95, zorder=0)
        ax.tick_params(axis="x", which="both", pad=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, loc="best")

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=200,
        show=show,
    )


def plot_psd_uv_recovery_comparison(
    df,
    plot_path="plots/hubble",
    show=False,
    filename="sigma_tau_psd_free_vs_fixed.pdf",
):
    """Compare valid free-slope BPL and fixed-slope DRW PSD fits to the UV fit."""

    required = {
        "log_sigma_uv",
        "log_sigma_uv_err",
        "log_tau_uv_rf",
        "log_tau_uv_rf_err",
        "log_sigma_ls",
        "log_sigma_ls_err",
        "log_tau_ls",
        "log_tau_ls_err",
        "alpha_high_ls",
        "psd_ls_valid",
        "log_sigma_ls_fixed",
        "log_sigma_ls_fixed_err",
        "log_tau_ls_fixed",
        "log_tau_ls_fixed_err",
        "psd_ls_fixed_valid",
    }
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise KeyError(f"Missing required columns for PSD-vs-UV recovery plot: {missing}")

    def _numeric(column):
        return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)

    log_sigma_uv = _numeric("log_sigma_uv")
    log_sigma_uv_err = _numeric("log_sigma_uv_err")
    log_tau_uv_rf = _numeric("log_tau_uv_rf")
    log_tau_uv_rf_err = _numeric("log_tau_uv_rf_err")

    slope = -_numeric("alpha_high_ls")
    slope_err = (
        _numeric("alpha_high_ls_err")
        if "alpha_high_ls_err" in df.columns
        else np.zeros(len(df), dtype=float)
    )
    valid_slope = np.isfinite(slope) & (slope > 1.0)
    log_rms_factor = np.full(len(df), np.nan, dtype=float)
    normalization_err = np.zeros(len(df), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rms_factor[valid_slope] = 0.5 * np.log10(
            1.0
            / (
                slope[valid_slope]
                * np.sin(np.pi / slope[valid_slope])
            )
        )
        derivative = -0.5 / np.log(10.0) * (
            1.0 / slope
            - np.pi
            * np.cos(np.pi / slope)
            / (slope**2 * np.sin(np.pi / slope))
        )
    finite_slope_err = valid_slope & np.isfinite(slope_err) & (slope_err >= 0.0)
    normalization_err[finite_slope_err] = (
        np.abs(derivative[finite_slope_err]) * slope_err[finite_slope_err]
    )

    free_sigma = _numeric("log_sigma_ls") + log_rms_factor
    free_sigma_err = np.hypot(_numeric("log_sigma_ls_err"), normalization_err)
    free_valid = pd.Series(df["psd_ls_valid"]).fillna(False).astype(bool).to_numpy()
    fixed_valid = (
        pd.Series(df["psd_ls_fixed_valid"])
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )

    panel_inputs = [
        (
            log_sigma_uv,
            free_sigma,
            log_sigma_uv_err,
            free_sigma_err,
            free_valid,
            r"$\log\,\sigma_{\rm UV}\ ({\rm mag})$",
            r"$\log\,\sigma_{\rm PSD,RMS}\ ({\rm mag})$",
            "sigma",
        ),
        (
            log_sigma_uv,
            _numeric("log_sigma_ls_fixed"),
            log_sigma_uv_err,
            _numeric("log_sigma_ls_fixed_err"),
            fixed_valid,
            r"$\log\,\sigma_{\rm UV}\ ({\rm mag})$",
            r"$\log\,\sigma_{\rm PSD,RMS}\ ({\rm mag})$",
            "sigma",
        ),
        (
            log_tau_uv_rf,
            _numeric("log_tau_ls"),
            log_tau_uv_rf_err,
            _numeric("log_tau_ls_err"),
            free_valid,
            r"$\log\,\tau_{\rm UV,RF}\ ({\rm days})$",
            r"$\log\,\tau_{\rm PSD,RF}\ ({\rm days})$",
            "tau",
        ),
        (
            log_tau_uv_rf,
            _numeric("log_tau_ls_fixed"),
            log_tau_uv_rf_err,
            _numeric("log_tau_ls_fixed_err"),
            fixed_valid,
            r"$\log\,\tau_{\rm UV,RF}\ ({\rm days})$",
            r"$\log\,\tau_{\rm PSD,RF}\ ({\rm days})$",
            "tau",
        ),
    ]

    panels = []
    for x, y, xerr, yerr, valid, xlabel, ylabel, quantity in panel_inputs:
        mask = (
            valid
            & np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(xerr)
            & np.isfinite(yerr)
            & (xerr >= 0.0)
            & (yerr >= 0.0)
        )
        panels.append(
            {
                "x": x[mask],
                "y": y[mask],
                "xerr": xerr[mask],
                "yerr": yerr[mask],
                "xlabel": xlabel,
                "ylabel": ylabel,
                "quantity": quantity,
            }
        )

    def _shared_limits(selected_panels, *, step, margin_floor):
        values = [
            values
            for panel in selected_panels
            for values in (panel["x"], panel["y"])
            if values.size
        ]
        if not values:
            return None
        values = np.concatenate(values)
        span = float(np.max(values) - np.min(values))
        margin = max(0.04 * span, margin_floor)
        lower = step * np.floor((np.min(values) - margin) / step)
        upper = step * np.ceil((np.max(values) + margin) / step)
        return float(lower), float(upper)

    sigma_limits = _shared_limits(panels[:2], step=0.05, margin_floor=0.06)
    tau_limits = _shared_limits(panels[2:], step=0.05, margin_floor=0.08)
    if sigma_limits is None and tau_limits is None:
        raise ValueError("No finite valid free-slope or fixed-slope PSD fits to plot.")

    def _plot_kde_contours(ax, x, y):
        if x.size <= 50:
            return
        try:
            kde = gaussian_kde(np.vstack([x, y]), bw_method="scott")
            xq = np.quantile(x, [0.01, 0.99])
            yq = np.quantile(y, [0.01, 0.99])
            x_range = float(xq[1] - xq[0])
            y_range = float(yq[1] - yq[0])
            if x_range <= 0.0 or y_range <= 0.0:
                return
            x_grid, y_grid = np.meshgrid(
                np.linspace(xq[0] - 0.1 * x_range, xq[1] + 0.1 * x_range, 220),
                np.linspace(yq[0] - 0.1 * y_range, yq[1] + 0.1 * y_range, 220),
            )
            density = kde(
                np.vstack([x_grid.ravel(), y_grid.ravel()])
            ).reshape(x_grid.shape)
            levels = _kde_conf_levels(density, conf=(0.954, 0.683))
            ax.contour(
                x_grid,
                y_grid,
                density,
                levels=levels,
                colors="red",
                linestyles=("solid", "solid"),
                linewidths=(2.6, 3.2),
                alpha=1.0,
                zorder=3,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"[PSD-vs-UV KDE contours] skipped: {exc}")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.0, 9.0),
        sharex="row",
        sharey="row",
        constrained_layout=True,
    )
    fig.get_layout_engine().set(
        w_pad=0.04,
        h_pad=0.04,
        wspace=0.06,
        hspace=0.04,
    )

    for ax, panel in zip(axes.flat, panels):
        limits = sigma_limits if panel["quantity"] == "sigma" else tau_limits
        if limits is None or panel["x"].size == 0:
            ax.text(
                0.5,
                0.5,
                "No valid PSD fits",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            continue
        x = panel["x"]
        y = panel["y"]
        delta = y - x
        ax.plot(limits, limits, "--", color="m", lw=2.0, zorder=-4)
        ax.errorbar(
            x,
            y,
            xerr=panel["xerr"],
            yerr=panel["yerr"],
            fmt="none",
            color="0.4",
            alpha=0.15,
            lw=0.75,
            capsize=1.2,
            capthick=0.6,
            rasterized=True,
            zorder=-3,
        )
        ax.scatter(
            x,
            y,
            s=10,
            color="k",
            alpha=0.58,
            edgecolors="none",
            rasterized=True,
            zorder=-2,
        )
        _plot_kde_contours(ax, x, y)
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(panel["xlabel"])
        ax.set_ylabel(panel["ylabel"])
        ax.text(
            0.97,
            0.03,
            (
                f"N = {delta.size}\n"
                f"Bias = {np.mean(delta):+.2f} dex\n"
                f"$\\sigma$ = {np.std(delta):.2f} dex"
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10.5,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="0.65",
                alpha=0.92,
            ),
        )
        major_step = 0.5
        ax.xaxis.set_major_locator(MultipleLocator(major_step))
        ax.yaxis.set_major_locator(MultipleLocator(major_step))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(
            direction="in",
            top=True,
            right=True,
            which="major",
            length=4,
            width=1.0,
        )
        ax.tick_params(
            direction="in",
            top=True,
            right=True,
            which="minor",
            length=2.5,
            width=0.8,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

    axes[0, 0].set_title("Free-slope BPL", fontsize=14)
    axes[0, 1].set_title("Fixed-slope DRW", fontsize=14)
    axes[0, 1].tick_params(labelleft=False)
    axes[1, 1].tick_params(labelleft=False)

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    return _save_figure(
        fig,
        os.path.join(diagnostics_path, filename),
        dpi=300,
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
    f_host_col="f_host_2500",
    f_host_label=None,
):
    """Plot host fraction against AGN-only log L_2500 with median and sigmoid trends."""
    required = {"z", "apparent_mag_2500", f_host_col}
    if not required.issubset(df.columns):
        return None
    if f_host_label is None:
        f_host_label = r"$f_{\rm host,2500}$"

    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
    m2500 = pd.to_numeric(df["apparent_mag_2500"], errors="coerce").to_numpy(dtype=float)
    f_host = pd.to_numeric(df[f_host_col], errors="coerce").to_numpy(dtype=float)

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
                fit_df = df.loc[mask].copy()
                fit_df["f_host_2500"] = y
                fit_model = fit_fhost_2500_l2500_model(
                    fit_df,
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
            f"No finite log L_2500 / {f_host_col} values",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_xlabel(r"$\log L_{2500}$")
    ax.set_ylabel(f_host_label)
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
        ylo = []
        yhi = []
        for i in range(len(l_edges) - 1):
            lo = l_edges[i]
            hi = l_edges[i + 1]
            keep = (x >= lo) & (x < hi)
            if i == len(l_edges) - 2:
                keep = (x >= lo) & (x <= hi)
            if np.count_nonzero(keep) >= min_bin_count:
                xmid.append(np.nanmedian(x[keep]))
                ymed.append(np.nanmedian(y[keep]))
                ylo.append(np.nanpercentile(y[keep], 16))
                yhi.append(np.nanpercentile(y[keep], 84))
        if xmid:
            ax.fill_between(
                xmid,
                ylo,
                yhi,
                color="k",
                alpha=0.16,
                linewidth=0,
                label=r"Binned $1\sigma$",
            )
            ax.plot(xmid, ymed, color="k", lw=2, label="Binned median")

    ax.set_xlabel(r"$\log L_{2500}$")
    ax.set_ylabel(r"$\alpha_{\lambda}$")
    ax.grid(False)
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


def plot_light_curve_n_points_vs_apparent_mag(
    df,
    plot_path="plots/hubble",
    show=False,
    filename="light_curve_n_points_vs_apparent_mag.pdf",
    nbins_mag=12,
    min_bin_count=5,
    exclude_bands=LIGHT_CURVE_N_POINTS_EXCLUDED_BANDS,
):
    """Plot light-curve point count against apparent m_2500."""
    if "apparent_mag_2500" not in df.columns:
        raise KeyError("Missing required column for LC point-count diagnostic: 'apparent_mag_2500'.")

    m2500 = pd.to_numeric(df["apparent_mag_2500"], errors="coerce").to_numpy(dtype=float)
    n_points, count_cols = light_curve_point_count_series(df, exclude_bands=exclude_bands)
    if n_points is None or len(count_cols) == 0:
        return None

    mask = np.isfinite(m2500) & np.isfinite(n_points) & (n_points >= 0.0)
    if not np.any(mask):
        return None

    x = m2500[mask]
    y = n_points[mask]
    z = (
        pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)[mask]
        if "z" in df.columns
        else None
    )

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    if z is not None and np.any(np.isfinite(z)):
        sc = ax.scatter(
            x,
            y,
            c=z,
            cmap="viridis",
            s=18,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(r"$z$")
    else:
        ax.scatter(
            x,
            y,
            s=18,
            alpha=0.55,
            color="tab:blue",
            linewidths=0,
            rasterized=True,
        )

    if np.nanmax(x) > np.nanmin(x):
        edges = np.linspace(np.nanmin(x), np.nanmax(x), int(nbins_mag) + 1)
        x_mid, y_med, y_lo, y_hi = [], [], [], []
        for i in range(len(edges) - 1):
            lo = edges[i]
            hi = edges[i + 1]
            keep = (x >= lo) & (x < hi)
            if i == len(edges) - 2:
                keep = (x >= lo) & (x <= hi)
            if np.count_nonzero(keep) >= int(min_bin_count):
                x_mid.append(np.nanmedian(x[keep]))
                y_med.append(np.nanmedian(y[keep]))
                p16, p84 = np.nanpercentile(y[keep], [16.0, 84.0])
                y_lo.append(p16)
                y_hi.append(p84)
        if x_mid:
            ax.plot(x_mid, y_med, color="k", lw=2.0, label="Binned median")
            ax.fill_between(x_mid, y_lo, y_hi, color="k", alpha=0.12, linewidth=0, label="16-84%")

    ax.set_xlabel(r"$m_{2500\,\mathrm{\AA}}$ (mag)")
    excluded = tuple(str(b) for b in (exclude_bands or ()))
    excluded_text = f" excluding {', '.join(excluded)}" if excluded else ""
    ax.set_ylabel(f"Light-curve points{excluded_text}")
    ax.set_title(
        f"LC sampling vs apparent magnitude{excluded_text}\n"
        f"N={int(np.count_nonzero(mask))}, columns={len(count_cols)}"
    )
    ax.grid(True, alpha=0.2)
    handles, _ = ax.get_legend_handles_labels()
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
    amp_delta_prefix = f"dlog_amp_blr{suffix}_"
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
        ax.grid(False)

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
    amp_delta_prefix = f"dlog_amp_blr{suffix}_"

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
    only_agn=False,
    speed="",
    show=False,
    use_alpha_lambda_term=None,
    use_eta_sigma_term=None,
    use_f_agn_psf_2500_sigmoid_term=None,
    use_f_agn_psf_2500_flux_fraction_term=None,
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
        only_agn=bool(only_agn),
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_f_agn_psf_2500_sigmoid_term=use_f_agn_psf_2500_sigmoid_term,
        use_f_agn_psf_2500_flux_fraction_term=use_f_agn_psf_2500_flux_fraction_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    priors, model_labels, model_labels_latex = get_model_params(
        cosmo_model,
        only_sna=bool(only_sna),
        only_agn=option_flags["only_agn"],
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
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
    use_eta_sigma_term=None,
    use_f_agn_psf_2500_sigmoid_term=None,
    use_f_agn_psf_2500_flux_fraction_term=None,
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
        use_eta_sigma_term=use_eta_sigma_term,
        use_f_agn_psf_2500_sigmoid_term=use_f_agn_psf_2500_sigmoid_term,
        use_f_agn_psf_2500_flux_fraction_term=use_f_agn_psf_2500_flux_fraction_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    priors, model_labels, model_labels_latex = get_model_params(
        cosmo_model,
        only_sna=only_sna,
        only_agn=option_flags["only_agn"],
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
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
    use_eta_sigma_term=None,
    use_f_agn_psf_2500_sigmoid_term=None,
    use_f_agn_psf_2500_flux_fraction_term=None,
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
        use_eta_sigma_term=use_eta_sigma_term,
        use_f_agn_psf_2500_sigmoid_term=use_f_agn_psf_2500_sigmoid_term,
        use_f_agn_psf_2500_flux_fraction_term=use_f_agn_psf_2500_flux_fraction_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    priors, model_labels, model_labels_latex = get_model_params(
        cosmo_model,
        only_sna=only_sna,
        only_agn=option_flags["only_agn"],
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
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
    only_agn=False,
    use_alpha_lambda_term=None,
    use_eta_sigma_term=None,
    use_f_agn_psf_2500_sigmoid_term=None,
    use_f_agn_psf_2500_flux_fraction_term=None,
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
        only_agn=only_agn,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_f_agn_psf_2500_sigmoid_term=use_f_agn_psf_2500_sigmoid_term,
        use_f_agn_psf_2500_flux_fraction_term=use_f_agn_psf_2500_flux_fraction_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    _, model_labels, model_labels_latex = get_model_params(
        cosmo_model,
        only_agn=option_flags["only_agn"],
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
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
    agn_label = "AGN" if only_agn else "SN Ia + AGN"
    legend.append(Line2D([0], [0], color="k", lw=6, label=agn_label))
    fig.legend(handles=legend, bbox_to_anchor=(0.99, 0.92), loc="upper right",
               fontsize=_COSMO_CORNER_LEGEND_FONTSIZE, frameon=False, markerscale=1.5)

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
    - membership is [left, right), except the final bin includes its right edge
    center: 'weighted' (default), 'mid', or 'geom'
    Returns zc, mean, sem, n for bins meeting min_count.
    """
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    e = np.asarray(yerr, float)
    bins = np.asarray(bins, float)
    if (
        bins.ndim != 1
        or bins.size < 2
        or not np.all(np.isfinite(bins))
        or np.any(np.diff(bins) <= 0)
    ):
        raise ValueError("bins must be a finite, strictly increasing 1-D array")

    m = np.isfinite(z) & np.isfinite(y) & np.isfinite(e) & (e > 0)
    if not np.any(m):
        return np.array([]), np.array([]), np.array([]), np.array([])

    z, y, e = z[m], y[m], e[m]
    w = 1.0 / (e * e)

    B = len(bins) - 1
    k = np.searchsorted(bins, z, side="right") - 1
    # np.searchsorted assigns the final edge just above the final bin.  Match
    # np.histogram by closing that one outer boundary explicitly.
    k[z == bins[-1]] = B - 1
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


def compute_hubble_redshift_trend(
    redshift,
    residuals,
    sigma_sel,
    *,
    z_pivot,
):
    """Fit the selection-weighted mean residual trend in pivoted log(1+z).

    The delta chi-squared compares a constant residual with a constant plus
    one redshift-slope parameter. It targets coherent redshift structure
    rather than measuring the total object-to-object scatter.
    """
    z = np.asarray(redshift, dtype=float)
    r = np.asarray(residuals, dtype=float)
    sigma = np.asarray(sigma_sel, dtype=float)
    if z.shape != r.shape or z.shape != sigma.shape:
        raise ValueError(
            "redshift, residuals, and sigma_sel must have identical shapes"
        )
    if not np.isfinite(z_pivot) or z_pivot <= -1.0:
        raise ValueError("z_pivot must be finite and greater than -1")

    valid = (
        np.isfinite(z)
        & (z > -1.0)
        & np.isfinite(r)
        & np.isfinite(sigma)
        & (sigma > 0.0)
    )
    n_used = int(np.count_nonzero(valid))
    empty = {
        "n_used": n_used,
        "intercept_mag": np.nan,
        "intercept_err_mag": np.nan,
        "slope_mag_per_dex": np.nan,
        "slope_err_mag_per_dex": np.nan,
        "slope_significance_sigma": np.nan,
        "delta_chi2": np.nan,
        "p_value": np.nan,
        "weighted_correlation": np.nan,
    }
    if n_used < 3:
        return empty

    z = z[valid]
    r = r[valid]
    sigma = sigma[valid]
    x = np.log10((1.0 + z) / (1.0 + float(z_pivot)))
    weights = 1.0 / np.square(sigma)
    design_constant = np.ones((n_used, 1), dtype=float)
    design_trend = np.column_stack((np.ones(n_used, dtype=float), x))

    def _fit(design):
        normal_matrix = design.T @ (weights[:, None] * design)
        try:
            covariance = np.linalg.inv(normal_matrix)
        except np.linalg.LinAlgError:
            return None
        coefficients = covariance @ (design.T @ (weights * r))
        fit_residuals = r - design @ coefficients
        chi2_value = float(np.sum(weights * np.square(fit_residuals)))
        return coefficients, covariance, chi2_value

    constant_fit = _fit(design_constant)
    trend_fit = _fit(design_trend)
    if constant_fit is None or trend_fit is None:
        return empty

    coefficients, covariance, trend_chi2 = trend_fit
    slope_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    delta_chi2 = max(float(constant_fit[2] - trend_chi2), 0.0)
    x_mean = float(np.sum(weights * x) / np.sum(weights))
    r_mean = float(np.sum(weights * r) / np.sum(weights))
    covariance_xr = float(np.sum(weights * (x - x_mean) * (r - r_mean)))
    variance_x = float(np.sum(weights * np.square(x - x_mean)))
    variance_r = float(np.sum(weights * np.square(r - r_mean)))
    correlation_denom = np.sqrt(variance_x * variance_r)

    return {
        "n_used": n_used,
        "intercept_mag": float(coefficients[0]),
        "intercept_err_mag": float(np.sqrt(max(covariance[0, 0], 0.0))),
        "slope_mag_per_dex": float(coefficients[1]),
        "slope_err_mag_per_dex": slope_error,
        "slope_significance_sigma": (
            float(coefficients[1] / slope_error) if slope_error > 0.0 else np.nan
        ),
        "delta_chi2": delta_chi2,
        "p_value": float(chi2_distribution.sf(delta_chi2, 1)),
        "weighted_correlation": (
            covariance_xr / correlation_denom
            if correlation_denom > 0.0
            else np.nan
        ),
    }


def _interval_bin_edges(bins, lower, upper):
    """Return ``bins`` clipped to one non-empty interval."""
    bins = np.asarray(bins, dtype=float)
    lower = max(float(lower), float(bins[0]))
    upper = min(float(upper), float(bins[-1]))
    if upper <= lower:
        return None
    return np.concatenate(
        (
            np.array([lower]),
            bins[(bins > lower) & (bins < upper)],
            np.array([upper]),
        )
    )


def _concatenate_weighted_bin_stats(parts):
    nonempty = [part for part in parts if part[0].size]
    if not nonempty:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=int),
        )
    combined = tuple(
        np.concatenate([part[index] for part in nonempty])
        for index in range(4)
    )
    order = np.argsort(combined[0], kind="stable")
    return tuple(values[order] for values in combined)


def _range_partitioned_weighted_bin_stats(
    z,
    y,
    yerr,
    bins,
    z_range,
    *,
    min_count=3,
    center="mid",
    fit_membership_mask=None,
):
    """Bin fit-range and out-of-range objects without mixed boundary bins.

    The fit interval is inclusive at both ends.  Its endpoints are inserted as
    bin edges, and objects below, inside, and above the interval are binned
    independently.  The return value is ``(in_range_stats, out_of_range_stats)``,
    where each stats tuple has the same layout as :func:`_weighted_bin_stats`.
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    bins = np.asarray(bins, dtype=float)
    if z.shape != y.shape or z.shape != yerr.shape:
        raise ValueError(
            "z, y, and yerr must have identical shapes for range-partitioned binning"
        )
    if fit_membership_mask is None:
        fit_membership = np.ones(z.shape, dtype=bool)
    else:
        fit_membership = np.asarray(fit_membership_mask, dtype=bool)
        if fit_membership.shape != z.shape:
            raise ValueError("fit_membership_mask must have the same shape as z")
    if len(z_range) != 2:
        raise ValueError("z_range must contain exactly two endpoints")
    z_lo, z_hi = map(float, z_range)
    if not np.isfinite(z_lo) or not np.isfinite(z_hi) or z_hi <= z_lo:
        raise ValueError("z_range must be finite and strictly increasing")

    def summarize(mask, lower, upper):
        edges = _interval_bin_edges(bins, lower, upper)
        if edges is None:
            return (
                np.empty(0, dtype=float),
                np.empty(0, dtype=float),
                np.empty(0, dtype=float),
                np.empty(0, dtype=int),
            )
        return _weighted_bin_stats(
            z[mask],
            y[mask],
            yerr[mask],
            edges,
            min_count=min_count,
            center=center,
        )

    below = summarize(z < z_lo, bins[0], z_lo)
    inside = summarize(
        (z >= z_lo) & (z <= z_hi) & fit_membership,
        z_lo,
        z_hi,
    )
    above = summarize(z > z_hi, z_hi, bins[-1])
    return inside, _concatenate_weighted_bin_stats((below, above))


def get_hubble_posterior_sample_indices(n_samples, target_samples=100):
    """Return the deterministic posterior rows used by ``plot_hubble``."""
    if (
        isinstance(n_samples, (bool, np.bool_))
        or not isinstance(n_samples, (int, np.integer))
        or n_samples <= 0
    ):
        raise ValueError(
            f"n_samples must be a positive integer; got {n_samples!r}."
        )
    if (
        isinstance(target_samples, (bool, np.bool_))
        or not isinstance(target_samples, (int, np.integer))
        or target_samples <= 0
    ):
        raise ValueError(
            "target_samples must be a positive integer; "
            f"got {target_samples!r}."
        )
    thin_factor = max(1, int(n_samples) // int(target_samples))
    return np.arange(int(n_samples), dtype=int)[::thin_factor]


def _validate_hubble_posterior_sample_indices(indices, n_samples):
    values = np.asarray(indices)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            "posterior_sample_indices must be a nonempty one-dimensional "
            f"array; got shape {values.shape}."
        )
    if np.issubdtype(values.dtype, np.bool_) or not np.issubdtype(
        values.dtype, np.integer
    ):
        raise ValueError("posterior_sample_indices must contain integers.")
    values = values.astype(int, copy=False)
    if np.any(values < 0) or np.any(values >= n_samples):
        raise ValueError(
            "posterior_sample_indices contains a row outside "
            f"[0, {n_samples})."
        )
    if np.unique(values).size != values.size:
        raise ValueError("posterior_sample_indices must not contain duplicates.")
    return values



def plot_hubble(flat_samples, df_agn, df_pantheon, cosmo_model, z_pivot_agn, plot_path="plots/hubble/",
                show_binned_agn=True, show_residuals=True,
                debias=False, dm_interp=None, show=False, completeness=True, show_true=False, verbose=True,
                cosmo_model_samples={}, residuals_sigma_clip=None, df_calibrators=None, z_range=(0.44, 3.16),
                dmi_values=None, dmi_sigma=None, dmi_selection_sigma=None, clipped_mask=None,
                filename=None, sigma_clip_threshold=None,
                use_alpha_lambda_term=None, use_eta_sigma_term=None,
                use_f_agn_psf_2500_sigmoid_term=None,
                use_f_agn_psf_2500_flux_fraction_term=None,
                use_redshift_log_f_term=None,
                only_agn=False,
                use_intrinsic_scatter_in_residual_sigma=True,
                diagnostics_suffix=None,
                residuals_csv_filename="residuals.csv",
                compute_only=False,
                *,
                dmi_posterior_draws=None,
                posterior_sample_indices=None,
                agn_pivot_context: AgnPivotContext):
    """
    Hubble diagram (Pantheon+-style):
      • Model line + 68% band in magenta
      • Concordance ΛCDM in black
      • SN Ia in blue
      • AGN points + error bars (solid if 0.44<=z<=3.16 else open)
      • Main: AGN binned in linear z
      • Inset: AGN binned in log z (matches inset x-scale)
      • Residuals: median of matched posterior M, cosmology, and debias draws

    Posterior debias draws must be passed as
    ``HubblePosteriorDrawSelection`` so their posterior rows and AGN column
    order can be verified.
    Returns:
      residuals,
      clipping_sigma,
      mu_pred_median,
      mu_pred_std,
      mu_pred_std_with_scatter
    """
    import os
    import numpy as np

    out_of_range_color = _OUT_OF_RANGE_AGN_COLOR
    out_of_range_main_marker_color = _OUT_OF_RANGE_AGN_MARKER_COLOR
    out_of_range_main_error_color = _OUT_OF_RANGE_AGN_ERROR_COLOR
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from astropy.cosmology import FlatLambdaCDM, FlatwCDM, Flatw0waCDM
    from scipy.ndimage import uniform_filter1d
    # Ensure your project provides these:
    # from your_module import FlatwpwaCDM, M_model_agn, M_model_agn_err, get_model_params, make_dm_function
    # (FlatwpwaCDM expected if using 'FlatwpwaCDM')

    # --- Labels ---
    label = cosmo_model_label_latex(cosmo_model)
    clipped_mask = _resolve_clipped_mask(df_agn, clipped_mask)

    # --- Deterministic posterior thinning shared by M, cosmology, and dmi ---
    flat_samples_all = np.asarray(flat_samples, dtype=float)
    if flat_samples_all.ndim != 2 or flat_samples_all.shape[0] == 0:
        raise ValueError(
            "flat_samples must be a nonempty two-dimensional array; "
            f"got shape {flat_samples_all.shape}."
        )
    n_samples = int(flat_samples_all.shape[0])
    draw_selection = (
        dmi_posterior_draws
        if isinstance(dmi_posterior_draws, HubblePosteriorDrawSelection)
        else None
    )
    if dmi_posterior_draws is not None and draw_selection is None:
        raise TypeError(
            "dmi_posterior_draws must be a "
            "HubblePosteriorDrawSelection so posterior-row and object-column "
            "alignment can be verified."
        )
    if draw_selection is not None and posterior_sample_indices is None:
        posterior_sample_indices = draw_selection.sample_indices
    explicit_sample_indices = posterior_sample_indices is not None
    if explicit_sample_indices:
        posterior_sample_indices = _validate_hubble_posterior_sample_indices(
            posterior_sample_indices,
            n_samples,
        )
    else:
        posterior_sample_indices = get_hubble_posterior_sample_indices(
            n_samples
        )

    selected_dmi_posterior_draws = None
    if draw_selection is not None:
        if not np.array_equal(
            draw_selection.sample_indices,
            posterior_sample_indices,
        ):
            raise ValueError(
                "dmi_posterior_draws sample indices do not match "
                "posterior_sample_indices."
            )
        expected_object_ids = tuple(
            str(value) for value in df_agn["object_id"].to_numpy()
        )
        if draw_selection.object_ids != expected_object_ids:
            raise ValueError(
                "dmi_posterior_draws object_id order does not match "
                "df_agn."
            )
        selected_dmi_posterior_draws = draw_selection.values
    flat_samples = flat_samples_all[posterior_sample_indices]

    z_grid = np.linspace(1e-4, 5.2, 500)

    # --- Parameter bookkeeping ---
    option_flags = resolve_model_option_flags(
        cosmo_model,
        np.asarray(flat_samples).shape[1],
        only_agn=only_agn,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_f_agn_psf_2500_sigmoid_term=use_f_agn_psf_2500_sigmoid_term,
        use_f_agn_psf_2500_flux_fraction_term=use_f_agn_psf_2500_flux_fraction_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    _, model_labels, _ = get_model_params(
        cosmo_model,
        only_agn=option_flags["only_agn"],
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
        use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
    )
    n_agn_params = sum(label != "M0_sn" for label in model_labels)
    show_sne = (
        not option_flags["only_agn"]
        and df_pantheon is not None
        and {"zHD", "MU_SH0ES", "MU_SH0ES_ERR_DIAG"}.issubset(df_pantheon.columns)
        and len(df_pantheon) > 0
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
    sample_parameter_dicts = [
        {key: sample[param_indices[key]] for key in model_labels}
        for sample in flat_samples
    ]
    sample_cosmologies = [
        get_cosmo(cosmo_model, params, z_pivot_agn)
        for params in sample_parameter_dicts
    ]
    mu_models = np.asarray(
        [cosmo.distmod(z_grid).value for cosmo in sample_cosmologies],
        dtype=float,
    )
    mu_model_16th   = np.percentile(mu_models, 16, axis=0)
    mu_model_median = np.percentile(mu_models, 50, axis=0)
    mu_model_84th   = np.percentile(mu_models, 84, axis=0)

    # Median params (also used later)
    results = {key: np.median(flat_samples[:, i]) for i, key in enumerate(model_labels)}

    # --- Predicted AGN μ per object ---
    m_obs = df_agn['apparent_mag_2500'].values
    agn_obs_arr, agn_err_arr, agn_pivot_arr = agn_model_pack_obs(
        df_agn,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
        pivot_context=agn_pivot_context,
    )
    agn_parameter_names, _, _ = get_agn_model_spec(
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
    )
    agn_parameter_samples = np.column_stack(
        [flat_samples[:, param_indices[name]] for name in agn_parameter_names]
    )
    predicted_M2500_samples = M_model_agn_posterior_samples(
        agn_parameter_samples,
        agn_obs_arr,
        agn_pivot_arr,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
    )
    mu_pred_samples = m_obs[None, :] - predicted_M2500_samples

    # De-bias (assumes your make_dm_function clips to grid, no extrapolation)
    if debias:
        if selected_dmi_posterior_draws is not None:
            mu_pred_samples -= selected_dmi_posterior_draws
        else:
            mu_pred_samples -= _resolve_debias_values(
                df_agn,
                dm_interp=dm_interp,
                dmi_values=dmi_values,
            )

    mu_pred_median = np.percentile(mu_pred_samples, 50, axis=0)
    mu_pred_16th   = np.percentile(mu_pred_samples, 16, axis=0)
    mu_pred_84th   = np.percentile(mu_pred_samples, 84, axis=0)

    # Average the observable-error variance over the posterior coefficients.
    # Global M0/slope posterior variance is correlated across objects and is
    # deliberately not copied into independent data error bars.
    pred_m2500_var, pred_m2500_var_components = (
        M_model_agn_observable_variance_posterior(
            agn_parameter_samples,
            agn_err_arr,
            obs_arr=agn_obs_arr,
            pivots_array=agn_pivot_arr,
            use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
            use_eta_sigma_term=option_flags["use_eta_sigma_term"],
            use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
        )
    )
    predicted_M2500_err = np.sqrt(pred_m2500_var)
    pred_m2500_sigma_var = pred_m2500_var_components["sigma"]
    pred_m2500_tau_var = pred_m2500_var_components["tau"]
    pred_m2500_cov_var = pred_m2500_var_components["covariance"]
    pred_m2500_alpha_lambda_var = pred_m2500_var_components.get(
        "alpha_lambda",
        np.zeros_like(pred_m2500_var),
    )
    pred_m2500_eta_sigma_var = pred_m2500_var_components.get(
        "eta_sigma",
        np.zeros_like(pred_m2500_var),
    )
    pred_m2500_fagn_sigmoid_var = pred_m2500_var_components.get(
        "f_agn_psf_2500_sigmoid",
        np.zeros_like(pred_m2500_var),
    )
    pred_m2500_fagn_flux_fraction_var = pred_m2500_var_components.get(
        "f_agn_psf_2500_flux_fraction",
        np.zeros_like(pred_m2500_var),
    )

    cosmo = get_cosmo(cosmo_model, results, z_pivot_agn)
    sigma_lens = sigma_lens_from_dc(df_agn['z'].values, cosmo)

    apparent_mag_err = df_agn['apparent_mag_2500_err'].values
    z_err = sigma_mu_from_z_err(df_agn["z"].values, df_agn["z_err"].values, cosmo)
    m_app_var = apparent_mag_err**2
    lens_var = sigma_lens**2
    z_var = z_err**2

    data_var_without_sigma_dmi = (
        m_app_var
        + lens_var
        + z_var
        + pred_m2500_var
    )

    sigma_dmi = None
    sigma_dmi_var = np.zeros_like(data_var_without_sigma_dmi)
    if dmi_sigma is not None:
        sigma_dmi = np.asarray(dmi_sigma, dtype=float)
        if sigma_dmi.shape != data_var_without_sigma_dmi.shape:
            raise ValueError(
                "dmi_sigma has shape "
                f"{sigma_dmi.shape}, but expected {data_var_without_sigma_dmi.shape}."
            )
        invalid_sigma_dmi = ~np.isfinite(sigma_dmi) | (sigma_dmi < 0.0)
        if np.any(invalid_sigma_dmi):
            invalid_rows = np.flatnonzero(invalid_sigma_dmi)[:10].tolist()
            raise ValueError(
                "dmi_sigma must contain finite, non-negative uncertainties; "
                f"invalid row indices: {invalid_rows}"
            )
        sigma_dmi_var = np.square(sigma_dmi)

    # The uncertainty of an applied debias correction is observational for
    # this diagram and therefore belongs in every debiased point error.
    data_var = data_var_without_sigma_dmi + sigma_dmi_var
    mu_pred_std_without_sigma_dmi = np.sqrt(data_var_without_sigma_dmi)
    mu_pred_std = np.sqrt(data_var)

    log_f_eff = evaluate_log_f(
        results,
        df_agn["z"].values,
        z_pivot=z_pivot_agn,
        use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
    )
    intrinsic_scatter = np.exp(log_f_eff)
    intrinsic_var = intrinsic_scatter**2
    total_var_without_sigma_dmi = data_var_without_sigma_dmi + intrinsic_var
    total_var = data_var + intrinsic_var
    mu_pred_std_with_scatter_without_sigma_dmi = np.sqrt(
        total_var_without_sigma_dmi
    )
    mu_pred_std_with_scatter = np.sqrt(total_var)

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

    z_values = df_agn["z"].to_numpy(dtype=float)
    mu_cosmo_samples = np.asarray(
        [cosmo.distmod(z_values).value for cosmo in sample_cosmologies],
        dtype=float,
    )
    # Preserve posterior covariance by taking the median of matched
    # sample-wise residuals, not a difference of separate marginal medians.
    residuals = np.percentile(
        mu_pred_samples - mu_cosmo_samples,
        50,
        axis=0,
    )
    mu_cosmo_posterior_median = np.percentile(
        mu_cosmo_samples,
        50,
        axis=0,
    )
    mu_pred_joint_consistent = (
        mu_cosmo_posterior_median + residuals
    )
    fit_membership_mask = (
        df_agn["is_fit_selection"].to_numpy(dtype=bool)
        if "is_fit_selection" in df_agn.columns
        else (
            np.isfinite(z_values)
            & (z_values >= z_range[0])
            & (z_values <= z_range[1])
        )
    )
    chi2_redshift_mask = (
        np.isfinite(z_values)
        & np.isfinite(residuals)
        & np.isfinite(data_var)
        & np.isfinite(total_var)
        & (data_var > 0.0)
        & (total_var > 0.0)
        & fit_membership_mask
    )
    if clipped_mask is not None:
        chi2_redshift_mask &= ~clipped_mask

    redshift_trend = None
    if debias and sigma_sel is not None:
        trend_mask = (
            np.isfinite(z_values)
            & fit_membership_mask
        )
        if clipped_mask is not None:
            trend_mask &= ~clipped_mask
        redshift_trend = compute_hubble_redshift_trend(
            z_values[trend_mask],
            residuals[trend_mask],
            sigma_sel[trend_mask],
            z_pivot=z_pivot_agn,
        )

    # Plot the inferred distance moduli directly.  The observed population
    # already contains its real scatter; adding a synthetic intrinsic-scatter
    # realization would move both individual points and statistical summaries
    # by an arbitrary, seed-dependent amount.
    mu_pred_plot = mu_pred_median
    clipping_sigma = mu_pred_std_with_scatter if use_intrinsic_scatter_in_residual_sigma else mu_pred_std
    display_residuals_err = mu_pred_std if debias else clipping_sigma
    binning_sigma = clipping_sigma
    chi2_sigma = mu_pred_std_with_scatter

    mu_zscore = np.abs(residuals) / clipping_sigma

    # ----------------- BINNING -----------------
    # Linear-z bins for MAIN & RESIDUALS panel
    #bins_linear = np.arange(0.4, 3.36, 0.1)
    bins_linear = np.arange(0.4, 3.41, 0.2)

    print("Using linear-z bins:", bins_linear)
    linear_main_in, linear_main_out = _range_partitioned_weighted_bin_stats(
        df_agn["z"].values,
        mu_pred_plot,
        binning_sigma,
        bins_linear,
        z_range,
        min_count=5,
        center="mid",
        fit_membership_mask=fit_membership_mask,
    )

    # Residual-panel bins use the same point-level fit-range partition as the
    # main panel, applied to the actual residuals.
    linear_residual_in, linear_residual_out = _range_partitioned_weighted_bin_stats(
        df_agn["z"].values,
        residuals,
        clipping_sigma,
        bins_linear,
        z_range,
        min_count=5,
        center="mid",
        fit_membership_mask=fit_membership_mask,
    )

    # Log-z bins for INSET (match inset xscale='log')
    zpos = df_agn["z"].values[df_agn["z"].values > 0]
    zmin_inset = max(0.02, float(np.min(zpos))) if zpos.size else 0.02
    zmax_inset = 3.8
    bins_per_decade = 6
    decades = np.log10(zmax_inset) - np.log10(zmin_inset)
    n_bins_log = max(1, int(np.ceil(decades * bins_per_decade)))
    bins_log = np.logspace(np.log10(bins_linear[0]), np.log10(bins_linear[-1]), n_bins_log + 1)
    #bins_log = bins_linear
    log_main_in, log_main_out = _range_partitioned_weighted_bin_stats(
        df_agn["z"].values,
        mu_pred_plot,
        binning_sigma,
        bins_log,
        z_range,
        fit_membership_mask=fit_membership_mask,
    )

    if compute_only:
        return (
            residuals,
            clipping_sigma,
            mu_pred_median,
            mu_pred_std,
            mu_pred_std_with_scatter,
        )

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
    clipped_in = clipped_mask & mask_in.to_numpy(dtype=bool) if clipped_mask is not None else None
    clipped_out = clipped_mask & mask_out.to_numpy(dtype=bool) if clipped_mask is not None else None

    # Background AGN point cloud
    inset_ax.scatter(
        df_agn["z"][mask_in], mu_pred_plot[mask_in],
        s=10, marker='o', c="black", alpha=0.18,
        linewidths=0, zorder=0
    )
    inset_ax.scatter(
        df_agn["z"][mask_out], mu_pred_plot[mask_out],
        s=12, marker='D', c=out_of_range_color, alpha=0.35,
        linewidths=0, zorder=0
    )

    # AGN (inside)
    inset_ax.errorbar(
        df_agn["z"][mask_in], mu_pred_plot[mask_in], yerr=display_residuals_err[mask_in],
        fmt='o', linestyle='none', markersize=2,
        mfc="black", mec="none",
        ecolor="#666666", elinewidth=0.8,
        alpha=0.7, zorder=1, label="AGN"
    )
    # AGN (outside, filled diamond)
    inset_ax.errorbar(
        df_agn["z"][mask_out], mu_pred_plot[mask_out], yerr=display_residuals_err[mask_out],
        fmt='D', linestyle='none', markersize=2, mfc=out_of_range_color, mec="none", alpha=0.85,
        ecolor=out_of_range_color, elinewidth=0.8, zorder=1
    )
    if clipped_in is not None and np.any(clipped_in):
        inset_ax.scatter(
            df_agn["z"][clipped_in],
            mu_pred_plot[clipped_in],
            s=16,
            marker="o",
            c="tab:green",
            alpha=0.95,
            linewidths=0,
            zorder=3,
            label="Clipped AGN",
        )
    if clipped_out is not None and np.any(clipped_out):
        inset_ax.scatter(
            df_agn["z"][clipped_out],
            mu_pred_plot[clipped_out],
            s=18,
            marker="D",
            c="tab:green",
            alpha=0.95,
            linewidths=0,
            zorder=3,
        )

    # INSET: log-binned AGN
    if show_binned_agn:
        z_log_in, mu_log_mean_in, mu_log_sem_in, _ = log_main_in
        z_log_out, mu_log_mean_out, mu_log_sem_out, _ = log_main_out
        # binned (inside)
        inset_ax.errorbar(
            z_log_in, mu_log_mean_in, yerr=mu_log_sem_in,
            fmt='o', linestyle='none',
            markersize=4, mfc='red', mec='none',
            ecolor='red', elinewidth=2.2, capsize=3.5,
            alpha=0.98, zorder=14, label="AGN (z-binned, log)"
        )
        # binned (outside, filled diamond)
        inset_ax.errorbar(
            z_log_out, mu_log_mean_out, yerr=mu_log_sem_out,
            fmt='D', linestyle='none',
            markersize=4, mfc='red', mec='none',
            ecolor='red', elinewidth=2.2, capsize=3.5,
            alpha=0.98, zorder=14, label="AGN (z-binned, log)",
        )

    # SN Ia
    if show_sne:
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

    if verbose:
        if sigma_clip_threshold is not None:
            threshold = float(sigma_clip_threshold)
            n_above_threshold = int(np.count_nonzero(mu_zscore > threshold))
            print(
                "Note: "
                f"{n_above_threshold} / {len(df_agn)} AGN exceed the residuals threshold "
                f"(|mu_zscore| > {threshold:.2f}) in this panel."
            )
    mask_in  = df_agn["z"].between(z_range[0], z_range[1])
    mask_out = ~mask_in
    ax.scatter(
        df_agn["z"][mask_in], mu_pred_plot[mask_in],
        s=12, marker='o', c="black", alpha=0.18,
        linewidths=0, zorder=-1
    )
    ax.scatter(
        df_agn["z"][mask_out], mu_pred_plot[mask_out],
        s=14, marker='D', c=out_of_range_color, alpha=0.35,
        linewidths=0, zorder=-1
    )
    # AGN (inside)
    for i in np.where(mask_in)[0]:
        ax.errorbar(
            df_agn["z"].iloc[i], mu_pred_plot[i], yerr=display_residuals_err[i],
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
            df_agn["z"].iloc[i], mu_pred_plot[i], yerr=display_residuals_err[i],
            fmt='D', linestyle='none', markersize=3, mfc=out_of_range_main_marker_color,
            mec="none",
            capsize=2, capthick=0.8,
            ecolor=out_of_range_main_error_color, elinewidth=0.8, zorder=0, label=None
        )
    if clipped_in is not None and np.any(clipped_in):
        ax.scatter(
            df_agn["z"][clipped_in],
            mu_pred_plot[clipped_in],
            s=22,
            marker="o",
            c="tab:green",
            alpha=0.95,
            linewidths=0,
            zorder=2,
            label="Clipped AGN",
        )
    if clipped_out is not None and np.any(clipped_out):
        ax.scatter(
            df_agn["z"][clipped_out],
            mu_pred_plot[clipped_out],
            s=24,
            marker="D",
            c="tab:green",
            alpha=0.95,
            linewidths=0,
            zorder=2,
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
        z_lin_in, mu_lin_mean_in, mu_lin_sem_in, _ = linear_main_in
        z_lin_out, mu_lin_mean_out, mu_lin_sem_out, _ = linear_main_out
        # binned (inside)
        print("Plotting in-range binned AGN (linear z) at:", z_lin_in)
        print("Plotting out-of-range binned AGN (linear z) at:", z_lin_out)
        ax.errorbar(
            z_lin_in, mu_lin_mean_in, yerr=mu_lin_sem_in,
            fmt='o', linestyle='none',
            markersize=5, mfc='red', mec='none',
            ecolor='red', elinewidth=2.2, capsize=3.5,
            alpha=0.98, zorder=14, label="AGN (z-binned)"
        )
        # binned (outside, filled diamond)
        ax.errorbar(
            z_lin_out, mu_lin_mean_out, yerr=mu_lin_sem_out,
            fmt='D', linestyle='none',
            markersize=5, mfc='red', mec='none',
            ecolor='red', elinewidth=2.2, capsize=3.5,
            alpha=0.98, zorder=14
        )

    # SN Ia
    if show_sne:
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
        agn_params_arr = agn_model_pack_params(
            results,
            use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
            use_eta_sigma_term=option_flags["use_eta_sigma_term"],
            use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
        )
        agn_obs_med = {key: float(np.median(df_agn[key].values)) * np.ones_like(z_grid) for key in agn_model_req_obs + agn_model_req_errs}
        if option_flags["use_alpha_lambda_term"]:
            agn_obs_med["alpha_lambda"] = float(np.median(df_agn["alpha_lambda"].values)) * np.ones_like(z_grid)
            agn_obs_med["alpha_lambda_err"] = float(np.median(df_agn["alpha_lambda_err"].values)) * np.ones_like(z_grid)
        if option_flags["use_eta_sigma_term"]:
            agn_obs_med["eta_sigma"] = float(np.median(df_agn["eta_sigma"].values)) * np.ones_like(z_grid)
            agn_obs_med["eta_sigma_err"] = float(np.median(df_agn["eta_sigma_err"].values)) * np.ones_like(z_grid)
        if (
            option_flags["use_f_agn_psf_2500_sigmoid_term"]
            or option_flags["use_f_agn_psf_2500_flux_fraction_term"]
        ):
            agn_obs_med["f_AGN_psf_2500"] = float(
                np.median(df_agn["f_AGN_psf_2500"].values)
            ) * np.ones_like(z_grid)
            agn_obs_med["f_AGN_psf_2500_err"] = float(
                np.median(df_agn["f_AGN_psf_2500_err"].values)
            ) * np.ones_like(z_grid)
        agn_obs_arr, agn_err_arr, agn_pivot_arr = agn_model_pack_obs(
            agn_obs_med,
            use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
            use_eta_sigma_term=option_flags["use_eta_sigma_term"],
            use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
            pivot_context=agn_pivot_context,
        )

        M_med_grid = np.median([
            M_model_agn(
                agn_params_arr,
                agn_obs_arr,
                agn_pivot_arr,
                use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
                use_eta_sigma_term=option_flags["use_eta_sigma_term"],
                use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
            )
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
            cosmo_model_other,
            np.asarray(cosmo_model_samples_other).shape[1],
            use_f_agn_psf_2500_sigmoid_term=option_flags[
                "use_f_agn_psf_2500_sigmoid_term"
            ],
            use_f_agn_psf_2500_flux_fraction_term=option_flags[
                "use_f_agn_psf_2500_flux_fraction_term"
            ],
        )
        _, model_labels_other, _ = get_model_params(
            cosmo_model_other,
            only_agn=option_flags_other["only_agn"],
            use_alpha_lambda_term=option_flags_other["use_alpha_lambda_term"],
            use_eta_sigma_term=option_flags_other["use_eta_sigma_term"],
            use_f_agn_psf_2500_sigmoid_term=option_flags_other["use_f_agn_psf_2500_sigmoid_term"],
            use_f_agn_psf_2500_flux_fraction_term=option_flags_other["use_f_agn_psf_2500_flux_fraction_term"],
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
        ax_resid.axhline(0.0, color="m", lw=3.0, zorder=1)

        # NEW: binned residuals in red (points + thin connecting line)
        z_res_in, resid_lin_mean_in, resid_lin_sem_in, _ = linear_residual_in
        z_res_out, resid_lin_mean_out, resid_lin_sem_out, _ = linear_residual_out
        if z_res_in.size or z_res_out.size:
            ax_resid.errorbar(
                z_res_in, resid_lin_mean_in, yerr=resid_lin_sem_in,
                fmt='o', linestyle='none', markersize=6,
                mfc='red', mec='none', ecolor='red', elinewidth=2.0, capsize=3.0,
                alpha=0.98, zorder=15, label="Binned AGN residuals"
            )
            ax_resid.errorbar(
                z_res_out, resid_lin_mean_out, yerr=resid_lin_sem_out,
                fmt='D', linestyle='none', markersize=6,
                mfc='red', mec='none', ecolor='red', elinewidth=2.0, capsize=3.0,
                alpha=0.98, zorder=15
            )


        for cosmo_model_other, cosmo_model_samples_other in cosmo_model_samples.items():
            option_flags_other = resolve_model_option_flags(
                cosmo_model_other,
                np.asarray(cosmo_model_samples_other).shape[1],
                use_f_agn_psf_2500_sigmoid_term=option_flags[
                    "use_f_agn_psf_2500_sigmoid_term"
                ],
                use_f_agn_psf_2500_flux_fraction_term=option_flags[
                    "use_f_agn_psf_2500_flux_fraction_term"
                ],
            )
            _, model_labels_other, _ = get_model_params(
                cosmo_model_other,
                only_agn=option_flags_other["only_agn"],
                use_alpha_lambda_term=option_flags_other["use_alpha_lambda_term"],
                use_eta_sigma_term=option_flags_other["use_eta_sigma_term"],
                use_f_agn_psf_2500_sigmoid_term=option_flags_other["use_f_agn_psf_2500_sigmoid_term"],
                use_f_agn_psf_2500_flux_fraction_term=option_flags_other["use_f_agn_psf_2500_flux_fraction_term"],
                use_redshift_log_f_term=option_flags_other["use_redshift_log_f_term"],
            )
            z_grid_fine = np.linspace(1e-4, 5.2, 500)
            param_indices_other = {name: model_labels_other.index(name) for name in model_labels_other}
            mu_models_other_fine = np.array([
                _mu_model(
                    cosmo_model_other,
                    {k: s[param_indices_other[k]] for k in model_labels_other},
                    z_grid_fine,
                    z_pivot_agn,
                )
                for s in np.asarray(cosmo_model_samples_other)
            ])
            mu_model_other_fine = np.percentile(mu_models_other_fine, 50, axis=0)
            mu_model_current_fine = np.interp(z_grid_fine, z_grid, mu_model_median)
            ax_resid.plot(z_grid_fine, mu_model_other_fine - mu_model_current_fine, lw=2.2, color=colors[cosmo_model_other], ls=line_styles[cosmo_model_other],
                          alpha=1.0, label=fr"{cosmo_model_other} $\Delta$μ")
            
        # Planck 2018 ΛCDM
        mu_conc = Planck18.distmod(z_grid).value
        #ax.plot(z_grid, mu_conc, color="#F0B000", lw=1.2, ls='--', zorder=5, alpha=1.0, label="flat $\Lambda$CDM (Planck 2018)")
        ax_resid.plot(z_grid, mu_conc - mu_model_median, lw=2.2, color="#F0B000", ls='--', alpha=1.0,)


        ax_resid.set_ylabel(r"$\Delta\mu$ (mag)")
        ax_resid.set_xlabel(r"$z$")
        def _paired_reduced_chi2(mask):
            if np.count_nonzero(mask) <= n_agn_params:
                return np.nan, np.nan
            chi2_full, _ = reduced_chi_squared(
                residuals[mask],
                mu_pred_std_with_scatter[mask],
                n_params=n_agn_params,
            )
            chi2_data_only, _ = reduced_chi_squared(
                residuals[mask],
                mu_pred_std[mask],
                n_params=n_agn_params,
            )
            return chi2_full, chi2_data_only

        chi2_full, chi2_data_only = _paired_reduced_chi2(
            chi2_redshift_mask
        )
        high_z_chi2_mask = chi2_redshift_mask & (z_values > 1.0)
        chi2_full_zgt1, chi2_data_only_zgt1 = _paired_reduced_chi2(
            high_z_chi2_mask
        )

        if np.isfinite(chi2_full) and np.isfinite(chi2_data_only):
            chi2_kind = "Debiased" if debias else "Biased"
            chi2_annotation_lines = [
                rf"{chi2_kind} $\chi^2_\nu$ (full / data only)",
                (
                    rf"${z_range[0]:.2f}\leq z\leq{z_range[1]:.2f}$: "
                    f"{chi2_full:.2f} / {chi2_data_only:.2f}"
                ),
            ]
            if (
                np.isfinite(chi2_full_zgt1)
                and np.isfinite(chi2_data_only_zgt1)
            ):
                chi2_annotation_lines.append(
                    (
                        rf"$1.00<z\leq{z_range[1]:.2f}$: "
                        f"{chi2_full_zgt1:.2f} / "
                        f"{chi2_data_only_zgt1:.2f}"
                    )
                )
            if (
                redshift_trend is not None
                and np.isfinite(redshift_trend["slope_mag_per_dex"])
            ):
                chi2_annotation_lines.append(
                    (
                        r"Selection-weighted $z$ trend: "
                        rf"$\gamma_z={redshift_trend['slope_mag_per_dex']:+.2f}"
                        rf"\pm{redshift_trend['slope_err_mag_per_dex']:.2f}$ "
                        rf"mag dex$^{{-1}}$ "
                        rf"$({redshift_trend['slope_significance_sigma']:+.1f}\sigma, "
                        rf"\Delta\chi^2={redshift_trend['delta_chi2']:.1f})$"
                    )
                )
            ax_resid.text(
                0.02,
                0.08,
                "\n".join(chi2_annotation_lines),
                transform=ax_resid.transAxes,
                ha="left",
                va="bottom",
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.02", facecolor="white", alpha=0.8, edgecolor="none"),
                zorder=20,                
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
        agn_params_arr_show = agn_model_pack_params(
            results,
            use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
            use_eta_sigma_term=option_flags["use_eta_sigma_term"],
            use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
        )
        obs_show, err_show, piv_show = agn_model_pack_obs(
            ds,
            use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
            use_eta_sigma_term=option_flags["use_eta_sigma_term"],
            use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
            pivot_context=agn_pivot_context,
        )
        pred_M_show = M_model_agn(
            agn_params_arr_show,
            obs_show,
            piv_show,
            use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
            use_eta_sigma_term=option_flags["use_eta_sigma_term"],
            use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
        )
        pred_M_err_show = M_model_agn_err(
            agn_params_arr_show,
            obs_show,
            err_show,
            piv_show,
            use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
            use_eta_sigma_term=option_flags["use_eta_sigma_term"],
            use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
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
            mu_model_at_show = np.percentile(
                np.asarray(
                    [
                        cosmo.distmod(z_show).value
                        for cosmo in sample_cosmologies
                    ],
                    dtype=float,
                ),
                50,
                axis=0,
            )
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
    filename = filename or ("hubble_diagram_debiased.pdf" if debias else "hubble_diagram.pdf")
    _save_figure(fig, os.path.join(plot_path, filename), dpi=600, show=show)

    diagnostics_path = os.path.join(plot_path, "diagnostics")
    os.makedirs(diagnostics_path, exist_ok=True)

    sigma_sel_var = np.full_like(mu_pred_std, np.nan, dtype=float)
    if sigma_sel is not None:
        sigma_sel_var = np.square(sigma_sel)

    # Retain the historical aliases in diagnostic files while making their
    # primary counterparts the complete debiased uncertainty model.
    total_var_plus_sigma_dmi = total_var
    total_var_no_logf = data_var
    total_var_no_logf_plus_sigma_dmi = data_var
    error_budget_mask = np.isfinite(total_var) & np.isfinite(residuals) & (total_var > 0) & chi2_redshift_mask
    sigma_dmi_mask = error_budget_mask & np.isfinite(sigma_dmi) if sigma_dmi is not None else None
    sigma_sel_mask = (
        error_budget_mask & np.isfinite(sigma_sel) & (sigma_sel > 0.0)
        if sigma_sel is not None
        else None
    )
    if np.any(error_budget_mask):
        def _chi2_red_from_var(var_term):
            mask = error_budget_mask & np.isfinite(var_term) & (var_term > 0)
            if np.count_nonzero(mask) <= n_agn_params:
                return np.nan
            value, _ = reduced_chi_squared(
                residuals[mask],
                np.sqrt(var_term[mask]),
                n_params=n_agn_params,
            )
            return float(value)

        def _median_fraction(component_var):
            mask = error_budget_mask & np.isfinite(component_var)
            if not np.any(mask):
                return np.nan
            return float(np.median(component_var[mask] / total_var[mask]))

        residual_rms = float(np.sqrt(np.mean(residuals[error_budget_mask] ** 2)))
        budget_rows = [
            {"metric": "n_objects", "value": float(np.count_nonzero(error_budget_mask))},
            {"metric": "n_agn_parameters", "value": float(n_agn_params)},
            {
                "metric": "chi2_degrees_of_freedom",
                "value": float(
                    np.count_nonzero(error_budget_mask) - n_agn_params
                ),
            },
            {"metric": "residual_rms_mag", "value": residual_rms},
            {"metric": "median_abs_residual_mag", "value": float(np.median(np.abs(residuals[error_budget_mask])))},
            {"metric": "chi2_red_full", "value": _chi2_red_from_var(total_var)},
            {"metric": "chi2_red_no_intrinsic_scatter", "value": _chi2_red_from_var(total_var_no_logf)},
            {"metric": "chi2_red_full_plus_sigma_dmi", "value": _chi2_red_from_var(total_var_plus_sigma_dmi)},
            {"metric": "chi2_red_no_intrinsic_scatter_plus_sigma_dmi", "value": _chi2_red_from_var(total_var_no_logf_plus_sigma_dmi)},
            {"metric": "chi2_red_full_without_sigma_dmi", "value": _chi2_red_from_var(total_var_without_sigma_dmi)},
            {"metric": "chi2_red_no_intrinsic_scatter_without_sigma_dmi", "value": _chi2_red_from_var(data_var_without_sigma_dmi)},
            {"metric": "chi2_red_no_predicted_M2500_err", "value": _chi2_red_from_var(m_app_var + lens_var + z_var + sigma_dmi_var + intrinsic_var)},
            {"metric": "chi2_red_no_sigma_lens", "value": _chi2_red_from_var(m_app_var + z_var + pred_m2500_var + sigma_dmi_var + intrinsic_var)},
            {"metric": "chi2_red_no_apparent_mag_err", "value": _chi2_red_from_var(lens_var + z_var + pred_m2500_var + sigma_dmi_var + intrinsic_var)},
            {"metric": "chi2_red_no_z_err", "value": _chi2_red_from_var(m_app_var + lens_var + pred_m2500_var + sigma_dmi_var + intrinsic_var)},
            {"metric": "chi2_red_sigma_sel", "value": _chi2_red_from_var(sigma_sel_var)},
            {"metric": "median_apparent_mag_err_mag", "value": float(np.median(apparent_mag_err[error_budget_mask]))},
            {"metric": "median_sigma_lens_mag", "value": float(np.median(sigma_lens[error_budget_mask]))},
            {"metric": "median_z_err_mag", "value": float(np.median(z_err[error_budget_mask]))},
            {"metric": "median_predicted_M2500_err_mag", "value": float(np.median(predicted_M2500_err[error_budget_mask]))},
            {"metric": "median_predicted_M2500_sigma_term_mag", "value": float(np.median(np.sqrt(np.clip(pred_m2500_sigma_var[error_budget_mask], 0.0, None))))},
            {"metric": "median_predicted_M2500_tau_term_mag", "value": float(np.median(np.sqrt(np.clip(pred_m2500_tau_var[error_budget_mask], 0.0, None))))},
            {"metric": "median_predicted_M2500_cov_term_mag_signed", "value": float(np.median(np.sign(pred_m2500_cov_var[error_budget_mask]) * np.sqrt(np.abs(pred_m2500_cov_var[error_budget_mask]))))},
            {"metric": "median_predicted_M2500_alpha_lambda_term_mag", "value": float(np.median(np.sqrt(np.clip(pred_m2500_alpha_lambda_var[error_budget_mask], 0.0, None))))},
            {"metric": "median_predicted_M2500_eta_sigma_term_mag", "value": float(np.median(np.sqrt(np.clip(pred_m2500_eta_sigma_var[error_budget_mask], 0.0, None))))},
            {"metric": "median_predicted_M2500_f_agn_psf_2500_sigmoid_term_mag", "value": float(np.median(np.sqrt(np.clip(pred_m2500_fagn_sigmoid_var[error_budget_mask], 0.0, None))))},
            {"metric": "median_predicted_M2500_f_agn_psf_2500_flux_fraction_term_mag", "value": float(np.median(np.sqrt(np.clip(pred_m2500_fagn_flux_fraction_var[error_budget_mask], 0.0, None))))},
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
            {"metric": "median_var_fraction_predicted_M2500_eta_sigma_term", "value": _median_fraction(pred_m2500_eta_sigma_var)},
            {"metric": "median_var_fraction_predicted_M2500_f_agn_psf_2500_sigmoid_term", "value": _median_fraction(pred_m2500_fagn_sigmoid_var)},
            {"metric": "median_var_fraction_predicted_M2500_f_agn_psf_2500_flux_fraction_term", "value": _median_fraction(pred_m2500_fagn_flux_fraction_var)},
        ]
        if redshift_trend is not None:
            budget_rows.extend(
                {
                    "metric": f"redshift_trend_{metric}",
                    "value": float(value),
                }
                for metric, value in redshift_trend.items()
            )
        budget_suffix = diagnostics_suffix if diagnostics_suffix is not None else ("_debiased" if debias else "")
        budget_summary_path = os.path.join(diagnostics_path, f"hubble_error_budget_summary{budget_suffix}.csv")
        pd.DataFrame(budget_rows).to_csv(budget_summary_path, index=False)

        per_object_budget_df = df_agn.copy()
        per_object_budget_df["residuals"] = residuals
        per_object_budget_df["residuals_err"] = clipping_sigma
        per_object_budget_df["clipping_sigma"] = clipping_sigma
        per_object_budget_df["chi2_sigma"] = chi2_sigma
        per_object_budget_df["apparent_mag_2500_err_term"] = apparent_mag_err
        per_object_budget_df["sigma_lens_term"] = sigma_lens
        per_object_budget_df["z_err_term"] = z_err
        per_object_budget_df["predicted_M2500_err_term"] = predicted_M2500_err
        per_object_budget_df["predicted_M2500_sigma_term"] = np.sqrt(np.clip(pred_m2500_sigma_var, 0.0, None))
        per_object_budget_df["predicted_M2500_tau_term"] = np.sqrt(np.clip(pred_m2500_tau_var, 0.0, None))
        per_object_budget_df["predicted_M2500_cov_term_signed"] = np.sign(pred_m2500_cov_var) * np.sqrt(np.abs(pred_m2500_cov_var))
        per_object_budget_df["predicted_M2500_alpha_lambda_term"] = np.sqrt(np.clip(pred_m2500_alpha_lambda_var, 0.0, None))
        per_object_budget_df["predicted_M2500_eta_sigma_term"] = np.sqrt(np.clip(pred_m2500_eta_sigma_var, 0.0, None))
        per_object_budget_df["predicted_M2500_f_agn_psf_2500_sigmoid_term"] = np.sqrt(np.clip(pred_m2500_fagn_sigmoid_var, 0.0, None))
        per_object_budget_df["predicted_M2500_f_agn_psf_2500_flux_fraction_term"] = np.sqrt(np.clip(pred_m2500_fagn_flux_fraction_var, 0.0, None))
        per_object_budget_df["intrinsic_scatter_term"] = intrinsic_scatter
        per_object_budget_df["sigma_dmi_term"] = sigma_dmi if sigma_dmi is not None else np.nan
        per_object_budget_df["sigma_sel_term"] = sigma_sel if sigma_sel is not None else np.nan
        per_object_budget_df["mu_pred_std_no_intrinsic_without_sigma_dmi"] = mu_pred_std_without_sigma_dmi
        per_object_budget_df["mu_pred_std_with_intrinsic_without_sigma_dmi"] = mu_pred_std_with_scatter_without_sigma_dmi
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
            "predicted_M2500_eta_sigma_term",
            "predicted_M2500_f_agn_psf_2500_sigmoid_term",
            "predicted_M2500_f_agn_psf_2500_flux_fraction_term",
            "intrinsic_scatter_term",
            "sigma_dmi_term",
            "sigma_sel_term",
            "mu_pred_std_no_intrinsic_without_sigma_dmi",
            "mu_pred_std_with_intrinsic_without_sigma_dmi",
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
            f" chi2_full_without_sigma_dmi={_chi2_red_from_var(total_var_without_sigma_dmi):.3f},"
            f" chi2_no_intrinsic_without_sigma_dmi={_chi2_red_from_var(data_var_without_sigma_dmi):.3f},"
            f" chi2_no_predM={_chi2_red_from_var(m_app_var + lens_var + z_var + sigma_dmi_var + intrinsic_var):.3f},"
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
    # Save raw per-object plot payload to CSV under plot_path.
    if debias:
        residuals_df = df_agn.copy()
        residuals_df["residuals"] = residuals
        residuals_df["mu_pred_median"] = mu_pred_median
        residuals_df["mu_pred_joint_consistent"] = (
            mu_pred_joint_consistent
        )
        residuals_df["mu_cosmo_posterior_median"] = (
            mu_cosmo_posterior_median
        )
        residuals_df["residual_estimator"] = (
            "median_matched_posterior_draws"
        )
        residuals_df["residual_posterior_sample_count"] = len(
            posterior_sample_indices
        )
        residuals_df["mu_pred_std_without_sigma_dmi"] = mu_pred_std_without_sigma_dmi
        residuals_df["mu_pred_std_with_scatter_without_sigma_dmi"] = mu_pred_std_with_scatter_without_sigma_dmi
        residuals_df["mu_pred_std"] = mu_pred_std
        residuals_df["mu_pred_std_with_scatter"] = mu_pred_std_with_scatter
        residuals_df["sigma_dmi"] = sigma_dmi if sigma_dmi is not None else np.nan
        residuals_df["mu_pred_std_with_scatter_and_sigma_dmi"] = np.sqrt(total_var_plus_sigma_dmi)
        residuals_df["mu_pred_std_and_sigma_dmi"] = np.sqrt(total_var_no_logf_plus_sigma_dmi)
        residuals_df["clipping_sigma"] = clipping_sigma
        residuals_df["chi2_sigma"] = chi2_sigma
        residuals_df["sigma_sel"] = sigma_sel if sigma_sel is not None else np.nan
        residuals_df["mu_zscore"] = mu_zscore
        fields = [
            "object_id",
            "apparent_mag_2500",
            "ra",
            "dec",
            "mu_pred_median",
            "mu_pred_joint_consistent",
            "mu_cosmo_posterior_median",
            "residual_estimator",
            "residual_posterior_sample_count",
            "mu_pred_std_without_sigma_dmi",
            "mu_pred_std_with_scatter_without_sigma_dmi",
            "mu_pred_std",
            "mu_pred_std_with_scatter",
            "clipping_sigma",
            "chi2_sigma",
            "sigma_sel",
            "sigma_dmi",
            "mu_pred_std_with_scatter_and_sigma_dmi",
            "mu_pred_std_and_sigma_dmi",
            "z",
            "sdss_name",
            "residuals",
            "mu_zscore",
        ]
        for membership_col in ("in_fit_z_range", "is_fit_selection"):
            if membership_col in residuals_df.columns:
                fields.insert(1, membership_col)
        sedfit_fields = [
            "fit_backend",
            "fracAGN_5100_fit",
            "fracAGN_5100_fit_err",
            "pl_slope",
            "pl_slope_err",
            "uv_slope",
            "uv_slope_err",
            "ebv_agn",
            "ebv_agn_err",
            "ebv_gal",
            "ebv_gal_err",
            "m_2500_dereddened",
            "m_2500_dereddened_err",
            "m_2500_attenuated_model",
            "m_2500_attenuated_model_err",
        ]
        fields.extend(field for field in sedfit_fields if field in residuals_df.columns)
        residuals_df = residuals_df[fields]
        residuals_df = residuals_df.sort_values(by="residuals", ascending=False)
        if residuals_csv_filename is not None:
            csv_path = os.path.join(plot_path, residuals_csv_filename)
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

    return residuals, clipping_sigma, mu_pred_median, mu_pred_std, mu_pred_std_with_scatter


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


def _residual_tail_bin_stats(x, z_resid, *, nbins=10, min_count=25):
    """Compute binned medians and negative-tail fractions for standardized residuals."""

    x = np.asarray(x, dtype=float)
    z_resid = np.asarray(z_resid, dtype=float)
    mask = np.isfinite(x) & np.isfinite(z_resid)
    if np.count_nonzero(mask) < max(min_count, 3):
        return None

    x_use = x[mask]
    z_use = z_resid[mask]
    if np.nanmax(x_use) <= np.nanmin(x_use):
        return None

    edges = np.linspace(np.nanmin(x_use), np.nanmax(x_use), int(nbins) + 1)
    rows = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        keep = (x_use >= lo) & (x_use < hi)
        if i == len(edges) - 2:
            keep = (x_use >= lo) & (x_use <= hi)
        n_bin = int(np.count_nonzero(keep))
        if n_bin < int(min_count):
            continue
        z_bin = z_use[keep]
        rows.append(
            {
                "x_mid": float(np.nanmedian(x_use[keep])),
                "n_bin": n_bin,
                "z_median": float(np.nanmedian(z_bin)),
                "z_p16": float(np.nanpercentile(z_bin, 16)),
                "z_p84": float(np.nanpercentile(z_bin, 84)),
                "frac_lt_m2": float(np.mean(z_bin < -2.0)),
                "frac_lt_m3": float(np.mean(z_bin < -3.0)),
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def plot_hubble_residual_tail_diagnostics(
    df_agn,
    residuals,
    residuals_err,
    *,
    sigma_dmi=None,
    sigma_sel=None,
    plot_path="plots/hubble",
    show=False,
    summary_filename="hubble_residual_tail_summary.csv",
    overview_filename="hubble_residual_tail_overview.pdf",
    fractions_filename="hubble_residual_tail_fractions.pdf",
    nbins=10,
    min_count=25,
    n_worst=15,
):
    """Localize negative standardized-residual tails in the debiased Hubble sample."""

    residuals = np.asarray(residuals, dtype=float)
    residuals_err = np.asarray(residuals_err, dtype=float)
    if residuals.shape != residuals_err.shape:
        raise ValueError(
            f"residuals shape {residuals.shape} does not match residuals_err shape {residuals_err.shape}."
        )
    if len(df_agn) != residuals.size:
        raise ValueError(
            f"df_agn has length {len(df_agn)}, but residual arrays have length {residuals.size}."
        )

    z_resid = np.full_like(residuals, np.nan, dtype=float)
    valid = np.isfinite(residuals) & np.isfinite(residuals_err) & (residuals_err > 0.0)
    z_resid[valid] = residuals[valid] / residuals_err[valid]

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    os.makedirs(diagnostics_path, exist_ok=True)

    summary = df_agn.copy()
    summary["residuals"] = residuals
    summary["residuals_err"] = residuals_err
    summary["z_resid"] = z_resid
    if sigma_dmi is not None:
        sigma_dmi = np.asarray(sigma_dmi, dtype=float)
        if sigma_dmi.shape != residuals.shape:
            raise ValueError(
                f"sigma_dmi shape {sigma_dmi.shape} does not match residual shape {residuals.shape}."
            )
        summary["sigma_dmi"] = sigma_dmi
    else:
        summary["sigma_dmi"] = np.nan
    if sigma_sel is not None:
        sigma_sel = np.asarray(sigma_sel, dtype=float)
        if sigma_sel.shape != residuals.shape:
            raise ValueError(
                f"sigma_sel shape {sigma_sel.shape} does not match residual shape {residuals.shape}."
            )
        summary["sigma_sel"] = sigma_sel
    else:
        summary["sigma_sel"] = np.nan
    summary["is_tail_lt_m2"] = summary["z_resid"] < -2.0
    summary["is_tail_lt_m3"] = summary["z_resid"] < -3.0

    preferred_cols = [
        "object_id",
        "sdss_name",
        "z",
        "residuals",
        "residuals_err",
        "z_resid",
        "sigma_dmi",
        "sigma_sel",
        "f_host_2500",
        "apparent_mag_2500",
        "alpha_lambda",
        "wrms",
        "ra",
        "dec",
        "is_tail_lt_m2",
        "is_tail_lt_m3",
    ]
    summary_cols = [col for col in preferred_cols if col in summary.columns]
    summary_csv_path = os.path.join(diagnostics_path, summary_filename)
    summary[summary_cols].sort_values("z_resid", ascending=True).to_csv(summary_csv_path, index=False)

    diagnostics = [
        ("z", "Redshift z"),
        ("sigma_dmi", r"$\sigma_{\rm dmi}$"),
        ("f_host_2500", r"$f_{\rm host,2500}$"),
        ("apparent_mag_2500", r"$m_{2500}$"),
    ]

    fig_overview, axes_overview = plt.subplots(2, 2, figsize=(11.0, 8.0), squeeze=False)
    axes_overview = axes_overview.ravel()
    fig_frac, axes_frac = plt.subplots(2, 2, figsize=(11.0, 8.0), squeeze=False)
    axes_frac = axes_frac.ravel()

    gaussian_ref_lt_m2 = float(norm.cdf(-2.0))
    gaussian_ref_lt_m3 = float(norm.cdf(-3.0))

    for ax_overview, ax_frac, (key, xlabel) in zip(axes_overview, axes_frac, diagnostics):
        if key not in summary.columns:
            ax_overview.axis("off")
            ax_frac.axis("off")
            continue

        x = np.asarray(summary[key], dtype=float)
        mask = np.isfinite(x) & np.isfinite(z_resid)
        if np.count_nonzero(mask) == 0:
            ax_overview.axis("off")
            ax_frac.axis("off")
            continue

        x_use = x[mask]
        z_use = z_resid[mask]
        ax_overview.scatter(
            x_use,
            z_use,
            s=12,
            alpha=0.35,
            color="tab:blue",
            linewidths=0,
            rasterized=True,
        )
        for yref, color, linestyle in [
            (0.0, "black", "-"),
            (-2.0, "tab:orange", "--"),
            (2.0, "tab:orange", "--"),
            (-3.0, "tab:red", ":"),
            (3.0, "tab:red", ":"),
        ]:
            ax_overview.axhline(yref, color=color, lw=1.2, ls=linestyle, alpha=0.9)

        stats_df = _residual_tail_bin_stats(x_use, z_use, nbins=nbins, min_count=min_count)
        if stats_df is not None:
            x_mid = stats_df["x_mid"].to_numpy(dtype=float)
            z_mid = stats_df["z_median"].to_numpy(dtype=float)
            z_lo = stats_df["z_p16"].to_numpy(dtype=float)
            z_hi = stats_df["z_p84"].to_numpy(dtype=float)
            ax_overview.fill_between(
                x_mid,
                z_lo,
                z_hi,
                color="0.6",
                alpha=0.15,
                linewidth=0,
                zorder=8,
            )
            ax_overview.plot(x_mid, z_mid, color="red", lw=2.0, zorder=9)

            ax_frac.plot(
                x_mid,
                stats_df["frac_lt_m2"].to_numpy(dtype=float),
                color="tab:orange",
                marker="o",
                lw=1.8,
                label=r"$P(z_{\rm resid} < -2)$",
            )
            ax_frac.plot(
                x_mid,
                stats_df["frac_lt_m3"].to_numpy(dtype=float),
                color="tab:red",
                marker="o",
                lw=1.8,
                label=r"$P(z_{\rm resid} < -3)$",
            )

        ax_overview.set_xlabel(xlabel)
        ax_overview.set_ylabel(r"$\Delta\mu / \sigma_{\Delta\mu}$")

        ax_frac.axhline(gaussian_ref_lt_m2, color="tab:orange", lw=1.0, ls="--", alpha=0.8)
        ax_frac.axhline(gaussian_ref_lt_m3, color="tab:red", lw=1.0, ls="--", alpha=0.8)
        ax_frac.set_ylim(0.0, 1.0)
        ax_frac.set_xlabel(xlabel)
        ax_frac.set_ylabel("Negative-tail fraction")
        ax_frac.legend(frameon=False, fontsize=9, loc="upper right")
        ax_frac.grid(True, alpha=0.25)

    fig_overview.tight_layout()
    fig_frac.tight_layout()
    overview_path = _save_figure(
        fig_overview,
        os.path.join(diagnostics_path, overview_filename),
        dpi=200,
        show=show,
    )
    fractions_path = _save_figure(
        fig_frac,
        os.path.join(diagnostics_path, fractions_filename),
        dpi=200,
        show=show,
    )
    return overview_path, fractions_path, summary_csv_path


def plot_predicted_vs_actual_M2500(
    flat_samples,
    df_agn,
    cosmo_model,
    z_pivot_agn,
    plot_path="plots/hubble",
    dm_interp=None,  # de-biasing function (optional)
    dmi_values=None,
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
    clipped_mask=None,
    use_alpha_lambda_term=None,
    use_eta_sigma_term=None,
    use_f_agn_psf_2500_sigmoid_term=None,
    use_f_agn_psf_2500_flux_fraction_term=None,
    use_redshift_log_f_term=None,
    dmi_selection_sigma=None,
    dmi_selection_sigma_interp=None,
    sigma_sel_floor_mag=0.05,
    *,
    agn_pivot_context: AgnPivotContext,
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
    clipped_mask = _resolve_clipped_mask(df_agn, clipped_mask)

    # --- model parameters from samples ---
    option_flags = resolve_model_option_flags(
        cosmo_model,
        np.asarray(flat_samples).shape[1],
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_f_agn_psf_2500_sigmoid_term=use_f_agn_psf_2500_sigmoid_term,
        use_f_agn_psf_2500_flux_fraction_term=use_f_agn_psf_2500_flux_fraction_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    priors, model_labels, model_labels_latex = get_model_params(
        cosmo_model,
        only_agn=option_flags["only_agn"],
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
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

    agn_params_arr = agn_model_pack_params(
        results,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
    )
    agn_obs_arr, agn_err_arr, agn_pivot_arr = agn_model_pack_obs(
        df_agn,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
        pivot_context=agn_pivot_context,
    )

    M_2500_pred = M_model_agn(
        agn_params_arr,
        agn_obs_arr,
        agn_pivot_arr,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
    )
    M_2500_pred_err = M_model_agn_err(
        agn_params_arr,
        agn_obs_arr,
        agn_err_arr,
        agn_pivot_arr,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
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
        actual_M_2500_eff = actual_M_2500 - _resolve_debias_values(
            df_agn,
            dm_interp=dm_interp,
            dmi_values=dmi_values,
        )
    else:
        actual_M_2500_eff = actual_M_2500

    residuals_all = M_2500_pred - actual_M_2500_eff               # mag
    sigma_sel = None
    if debias:
        sigma_sel = _resolve_selection_sigma_values(
            df_agn,
            dmi_selection_sigma=dmi_selection_sigma,
            dmi_selection_sigma_interp=dmi_selection_sigma_interp,
            sigma_sel_floor_mag=sigma_sel_floor_mag,
        )
    sigma_all = np.sqrt(
        M_2500_pred_err**2 + xerr**2 + sigma_intrinsic**2
    )  # full chi2 denominator
    if sigma_sel is not None:
        sigma_all = np.where(np.isfinite(sigma_sel) & (sigma_sel > 0.0), sigma_sel, sigma_all)

    # Display the population-level intrinsic scatter as point scatter, not as
    # enlarged error bars. Keep chi-squared on the current sigma_sel path.
    point_scatter_m2500 = _population_scatter_offsets(
        np.full_like(M_2500_pred, sigma_intrinsic, dtype=float),
        enabled=debias,
        seed=1742,
    )
    M_2500_pred_plot = M_2500_pred + point_scatter_m2500
    yerr_plot_all = np.asarray(M_2500_pred_err, dtype=float)

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

        x = actual_M_2500_eff[bin_mask]
        y = M_2500_pred[bin_mask]
        y_plot = M_2500_pred_plot[bin_mask]
        xerr_bin = xerr[bin_mask]
        yerr_bin = M_2500_pred_err[bin_mask]
        yerr_plot_bin = yerr_plot_all[bin_mask]
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

        # Filled circles are in range; out-of-range objects follow the Hubble
        # diagram's dark blue-gray diamond styling.
        z_bin = z[bin_mask]
        mask_closed = (z_bin >= z_range[0]) & (z_bin <= z_range[1])
        mask_open = ~mask_closed
        clipped_bin = clipped_mask[bin_mask] if clipped_mask is not None else None

        if np.any(mask_closed):
            ax.errorbar(
                x[mask_closed], y_plot[mask_closed],
                xerr=xerr_bin[mask_closed], yerr=yerr_plot_bin[mask_closed],
                fmt="none", ecolor="#666666", elinewidth=0.7, alpha=0.4, zorder=2,
            )
        if np.any(mask_open):
            ax.errorbar(
                x[mask_open], y_plot[mask_open],
                xerr=xerr_bin[mask_open], yerr=yerr_plot_bin[mask_open],
                fmt="none", ecolor=_OUT_OF_RANGE_AGN_ERROR_COLOR,
                elinewidth=0.7, zorder=2,
            )

        # Filled markers. Green clipped markers are overlaid below.
        ax.scatter(
            x[mask_closed], y_plot[mask_closed],
            facecolors="k", edgecolors='k', #c=colors_bin[mask_closed],
            s=20, alpha=1.0,
            linewidths=0.8, zorder=3,
        )

        # filled diamonds outside z-range
        ax.scatter(
            x[mask_open], y_plot[mask_open],
            facecolors=_OUT_OF_RANGE_AGN_MARKER_COLOR, edgecolors="none",
            marker="D",
            s=20, linewidths=0, zorder=3,
        )
        if clipped_bin is not None and np.any(clipped_bin):
            clipped_closed = mask_closed & clipped_bin
            clipped_open = mask_open & clipped_bin
            if np.any(clipped_closed):
                ax.scatter(
                    x[clipped_closed],
                    y_plot[clipped_closed],
                    facecolors="tab:green",
                    edgecolors="tab:green",
                    s=28,
                    alpha=0.95,
                    linewidths=0.6,
                    zorder=4,
                    label="Clipped AGN" if i == 0 else None,
                )
            if np.any(clipped_open):
                ax.scatter(
                    x[clipped_open],
                    y_plot[clipped_open],
                    facecolors="tab:green",
                    edgecolors="tab:green",
                    marker="D",
                    s=28,
                    alpha=0.95,
                    linewidths=0.6,
                    zorder=4,
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
    redshifts=(0.5, 1.0, 2.0, 3.0), show=False, plot_path=None
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


def plot_completeness_pre_post_cut_audit(
    p_detect,
    mag_centers,
    z_centers,
    before_cuts,
    after_cuts,
    *,
    magnitude_col=COMPLETENESS_MAG_COL,
    plot_path="plots/hubble",
    filename="completeness_audit_pre_post_cuts.pdf",
    map_label=r"$C(m,z)$",
    magnitude_label=r"$m_{2500}$",
    show=False,
):
    """Plot the frozen completeness map against samples before and after cuts.

    The two rows use one identical map and color normalization.  The left
    panels show the catalog locations over the map; the right panels show the
    completeness evaluated at each object's observed ``(m, z)``.  Counts
    outside the finite map-center rectangle are reported rather than silently
    hidden by the axes limits.
    """

    required = {magnitude_col, "z"}
    for sample_name, sample in (
        ("before cuts", before_cuts),
        ("after cuts", after_cuts),
    ):
        missing = required - set(sample.columns)
        if missing:
            raise KeyError(
                f"Completeness audit {sample_name} sample is missing "
                f"{sorted(missing)}."
            )

    mag_centers = np.asarray(mag_centers, dtype=float)
    z_centers = np.asarray(z_centers, dtype=float)
    if (
        mag_centers.ndim != 1
        or z_centers.ndim != 1
        or len(mag_centers) < 2
        or len(z_centers) < 2
        or np.any(~np.isfinite(mag_centers))
        or np.any(~np.isfinite(z_centers))
    ):
        raise ValueError("Completeness audit requires finite one-dimensional grids.")

    mesh_mag, mesh_z = np.meshgrid(mag_centers, z_centers, indexing="ij")
    mesh_completeness = np.asarray(p_detect(mesh_mag, mesh_z), dtype=float)
    if mesh_completeness.shape != mesh_mag.shape:
        raise ValueError(
            "Completeness map returned an incompatible audit-grid shape: "
            f"{mesh_completeness.shape}, expected {mesh_mag.shape}."
        )

    dm = float(np.median(np.diff(mag_centers)))
    dz = float(np.median(np.diff(z_centers)))
    mag_edges = np.concatenate(
        ([mag_centers[0] - 0.5 * dm], mag_centers + 0.5 * dm)
    )
    z_edges = np.concatenate(([z_centers[0] - 0.5 * dz], z_centers + 0.5 * dz))
    norm = colors.LogNorm(vmin=1e-5, vmax=1.0, clip=True)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10), constrained_layout=True)
    samples = (("Before Hubble-quality cuts", before_cuts),
               ("After Hubble-quality cuts", after_cuts))
    image = None
    for row, (sample_label, sample) in enumerate(samples):
        mag = pd.to_numeric(sample[magnitude_col], errors="coerce").to_numpy(dtype=float)
        redshift = pd.to_numeric(sample["z"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(mag) & np.isfinite(redshift)
        completeness = np.full(len(sample), np.nan, dtype=float)
        if np.any(finite):
            completeness[finite] = np.asarray(
                p_detect(mag[finite], redshift[finite]), dtype=float
            )
        finite_value = finite & np.isfinite(completeness)
        outside = finite & (
            (mag < mag_edges[0])
            | (mag > mag_edges[-1])
            | (redshift < z_edges[0])
            | (redshift > z_edges[-1])
        )
        below_floor = finite_value & (completeness < norm.vmin)

        ax_map = axes[row, 0]
        image = ax_map.pcolormesh(
            mag_edges,
            z_edges,
            np.clip(mesh_completeness.T, norm.vmin, norm.vmax),
            shading="auto",
            cmap="viridis",
            norm=norm,
            rasterized=True,
        )
        ax_map.scatter(
            mag[finite], redshift[finite], s=5, color="white", alpha=0.32,
            linewidths=0, rasterized=True,
        )
        ax_map.set_xlim(mag_edges[0], mag_edges[-1])
        ax_map.set_ylim(z_edges[0], z_edges[-1])
        ax_map.set_xlabel(magnitude_label)
        ax_map.set_ylabel("z")
        ax_map.set_title(f"{sample_label}: {np.count_nonzero(finite):,} finite objects")
        ax_map.text(
            0.02, 0.02,
            f"outside grid: {np.count_nonzero(outside):,}\n"
            f"nonfinite: {len(sample) - np.count_nonzero(finite):,}",
            transform=ax_map.transAxes, ha="left", va="bottom", fontsize=10,
            color="white", bbox={"facecolor": "black", "alpha": 0.45, "pad": 3},
        )

        ax_value = axes[row, 1]
        if np.any(finite_value):
            ax_value.scatter(
                redshift[finite_value],
                np.clip(completeness[finite_value], norm.vmin, norm.vmax),
                c=np.clip(completeness[finite_value], norm.vmin, norm.vmax),
                cmap="viridis", norm=norm, s=8, alpha=0.38, linewidths=0,
                rasterized=True,
            )
            minimum = float(np.min(completeness[finite_value]))
            minimum_text = f"{minimum:.3g}"
        else:
            minimum_text = "n/a"
        ax_value.set_yscale("log")
        ax_value.set_ylim(norm.vmin, 1.05)
        ax_value.set_xlim(z_edges[0], z_edges[-1])
        ax_value.set_xlabel("z")
        ax_value.set_ylabel(f"{map_label} at observed $(m,z)$")
        ax_value.set_title(
            f"{sample_label}: minimum={minimum_text}; "
            f"below $10^{{-5}}$={np.count_nonzero(below_floor):,}"
        )

    if image is not None:
        colorbar = fig.colorbar(image, ax=axes[:, 0], pad=0.015)
        colorbar.set_label(map_label)
    fig.suptitle("Completeness audit before and after Hubble-quality cuts", fontsize=17)
    output_path = os.path.join(plot_path or "plots/hubble", filename)
    return _save_figure(fig, output_path, dpi=250, show=show)


def _residual_axis_label(residual_label):
    if residual_label == "residuals":
        return "Residuals (mag)"
    if residual_label == "L2500_sigma_tau_residuals":
        return r"$\Delta \log L_{2500}$ (dex)"
    if residual_label == "r_z":
        return r"$r_z$ (mag)"
    return residual_label



_PARTIAL_CONTROL_PRIORITY_FIELDS = (
    "m_2500_dereddened",
    "m_2500_attenuated_model",
    "a_2500_total",
    "a_2500_internal",
    "a_2500_galaxy",
    "ebv_agn",
    "ebv_gal",
    "log_ebv_agn",
    "log_ebv_gal",
    "alpha_nu_attenuated_1450_2500",
    "alpha_nu_intrinsic_1450_2500",
    "delta_alpha_nu",
    "uv_slope",
)
_PARTIAL_CONTROL_EXCLUDED_SUFFIXES = (
    "_err",
    "_err_lower",
    "_err_upper",
    "_std",
    "_rhat",
)


def _partial_control_add_derived_fields(frame):
    """Add the derived scalar quantities shown in the partial-control atlas."""
    if {
        "alpha_nu_attenuated_1450_2500",
        "alpha_nu_intrinsic_1450_2500",
    }.issubset(frame.columns):
        frame["delta_alpha_nu"] = (
            pd.to_numeric(
                frame["alpha_nu_attenuated_1450_2500"], errors="coerce"
            )
            - pd.to_numeric(
                frame["alpha_nu_intrinsic_1450_2500"], errors="coerce"
            )
        )
    if "ebv_wu" in frame.columns:
        frame["log_ebv_wu"] = np.log10(
            np.abs(pd.to_numeric(frame["ebv_wu"], errors="coerce")) + 1e-10
        )
    if "log_sigma_uv_uncorrected" in frame.columns:
        frame["log_sigma_uv_diluted"] = pd.to_numeric(
            frame["log_sigma_uv_uncorrected"], errors="coerce"
        )
    if {"log_tau_uv_rf", "log_tau_fast_uv", "z"}.issubset(frame.columns):
        tau = np.power(
            10.0, pd.to_numeric(frame["log_tau_uv_rf"], errors="coerce")
        )
        tau_fast_rf = np.power(
            10.0,
            pd.to_numeric(frame["log_tau_fast_uv"], errors="coerce")
            - np.log10(
                1.0 + pd.to_numeric(frame["z"], errors="coerce")
            ),
        )
        delta = tau - tau_fast_rf
        frame["log_delta_tau_uv_fast_rf"] = np.where(
            delta > 0.0, np.log10(delta), np.nan
        )


def _partial_control_parameter_groups(frame, *, min_points):
    """Return fitted predictors followed by scalar spectra-H5 parameters."""
    fitted_predictors = [
        field
        for field in ("log_sigma_uv", "log_tau_uv_rf")
        if field in frame.columns
    ]
    spectra_fields = set(frame.attrs.get("spectra_fit_columns", ()))
    if "delta_alpha_nu" in frame.columns:
        spectra_fields.add("delta_alpha_nu")
    excluded = {
        "object_id",
        "sdss_name",
        "z",
        *fitted_predictors,
    }
    sed_fields = []
    for field in spectra_fields:
        if field in excluded or field not in frame.columns:
            continue
        if field.startswith("SDSS_") or field.endswith(
            _PARTIAL_CONTROL_EXCLUDED_SUFFIXES
        ):
            continue
        series = frame[field]
        if not isinstance(series, pd.Series) or not (
            pd.api.types.is_numeric_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
        ):
            continue
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size < int(min_points) or np.unique(finite).size < 2:
            continue
        sed_fields.append(field)

    sed_unique = sorted(set(sed_fields))
    sed_ordered = [
        field for field in _PARTIAL_CONTROL_PRIORITY_FIELDS if field in sed_unique
    ]
    sed_ordered.extend(
        field for field in sed_unique if field not in sed_ordered
    )
    return [
        ("Hubble-fit predictors", fitted_predictors),
        ("SED / spectral-fit parameters", sed_ordered),
    ]


def _partial_control_residualize(values, controls):
    values = np.asarray(values, dtype=float)
    controls = np.asarray(controls, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values) & np.all(np.isfinite(controls), axis=1)
    if np.count_nonzero(finite) <= controls.shape[1]:
        return result
    coefficients = np.linalg.lstsq(
        controls[finite], values[finite], rcond=None
    )[0]
    result[finite] = values[finite] - controls[finite] @ coefficients
    return result


def _partial_control_binned_trend(x, y, z, *, bins=14, min_count=8):
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[finite], y[finite], z[finite]
    if x.size < max(3 * int(min_count), 10) or np.unique(x).size < 3:
        return (np.array([]),) * 5
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, int(bins) + 1)))
    centers, medians, lower, upper, mean_z = [], [], [], [], []
    for bin_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        use = (x >= left) & (
            (x <= right) if bin_index == len(edges) - 2 else (x < right)
        )
        if np.count_nonzero(use) < int(min_count):
            continue
        centers.append(np.median(x[use]))
        medians.append(np.median(y[use]))
        lower.append(np.quantile(y[use], 0.16))
        upper.append(np.quantile(y[use], 0.84))
        mean_z.append(np.mean(z[use]))
    return tuple(
        np.asarray(value)
        for value in (centers, medians, lower, upper, mean_z)
    )


def plot_full_residuals_debiased_partial_controls(
    df_agn,
    residuals,
    *,
    plot_path="plots/hubble",
    z_range=(0.44, 3.16),
    show=False,
    redshift_bin_width=0.3,
    panels_per_page=20,
    min_points=10,
    trend_bins=14,
):
    """Plot post-cut Hubble residual associations after shared controls.

    Each fitted predictor is residualized against redshift-bin fixed effects
    and the other fitted predictor. Each spectra-H5 parameter is residualized
    against the same redshift effects and both fitted predictors. The Hubble
    residual is residualized against the identical controls in each panel.
    """
    if "z" not in df_agn.columns:
        raise ValueError("df_agn must contain a 'z' column.")
    if int(min_points) < 3:
        raise ValueError("min_points must be at least 3.")
    if int(panels_per_page) < 1:
        raise ValueError("panels_per_page must be positive.")
    if not np.isfinite(redshift_bin_width) or redshift_bin_width <= 0.0:
        raise ValueError("redshift_bin_width must be positive.")

    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 1 or residuals.size != len(df_agn):
        raise ValueError(
            f"residuals length {residuals.size} does not match dataframe "
            f"length {len(df_agn)}."
        )
    z_min, z_max = map(float, z_range)
    if not (np.isfinite(z_min) and np.isfinite(z_max) and z_min < z_max):
        raise ValueError("z_range must contain two finite, increasing values.")

    source_attrs = dict(df_agn.attrs)
    table = df_agn.copy().reset_index(drop=True)
    table.attrs.update(source_attrs)
    table["residuals"] = residuals
    z_all = pd.to_numeric(table["z"], errors="coerce").to_numpy(dtype=float)
    fit_selection = (
        table["is_fit_selection"].fillna(False).to_numpy(dtype=bool)
        if "is_fit_selection" in table.columns
        else np.ones(len(table), dtype=bool)
    )
    postcut = (
        fit_selection
        & np.isfinite(z_all)
        & np.isfinite(residuals)
        & (z_all >= z_min)
        & (z_all <= z_max)
    )
    table = table.loc[postcut].copy().reset_index(drop=True)
    table.attrs.update(source_attrs)
    if len(table) < int(min_points):
        raise ValueError(
            "Too few finite post-cut residuals for partial-control "
            f"diagnostics: {len(table)} < {int(min_points)}."
        )
    _partial_control_add_derived_fields(table)

    groups = _partial_control_parameter_groups(table, min_points=min_points)
    fields = [field for _, section_fields in groups for field in section_fields]
    if not fields:
        raise ValueError("No eligible fitted or spectra-H5 parameters found.")

    z = pd.to_numeric(table["z"], errors="coerce").to_numpy(dtype=float)
    redshift_edges = np.arange(
        z_min, z_max + float(redshift_bin_width), float(redshift_bin_width)
    )
    if redshift_edges[-1] < z_max:
        redshift_edges = np.append(redshift_edges, z_max)
    else:
        redshift_edges[-1] = z_max
    z_bin = pd.cut(
        table["z"], redshift_edges, right=True, include_lowest=True
    )
    z_dummies = pd.get_dummies(z_bin, drop_first=True, dtype=float).to_numpy()
    y = table["residuals"].to_numpy(dtype=float)

    partials = {}
    for field in fields:
        if field == "log_sigma_uv":
            continuous = [name for name in ("log_tau_uv_rf",) if name in table]
            control_label = r"$z$ bins + $\log\tau_{\rm UV,rf}$"
        elif field == "log_tau_uv_rf":
            continuous = [name for name in ("log_sigma_uv",) if name in table]
            control_label = r"$z$ bins + $\log\sigma_{\rm UV}$"
        else:
            continuous = [
                name
                for name in ("log_sigma_uv", "log_tau_uv_rf")
                if name in table
            ]
            control_label = (
                r"$z$ bins + $\log\sigma_{\rm UV}$ + $\log\tau_{\rm UV,rf}$"
                if len(continuous) == 2
                else "$z$ bins" + " + " + " + ".join(continuous)
            )
        continuous_values = np.column_stack(
            [
                pd.to_numeric(table[name], errors="coerce").to_numpy(float)
                for name in continuous
            ]
        ) if continuous else np.empty((len(table), 0), dtype=float)
        controls = np.column_stack(
            [np.ones(len(table)), continuous_values, z_dummies]
        )
        x = pd.to_numeric(table[field], errors="coerce").to_numpy(float)
        common = np.isfinite(x) & np.isfinite(y) & np.all(
            np.isfinite(controls), axis=1
        )
        partials[field] = (
            _partial_control_residualize(np.where(common, x, np.nan), controls),
            _partial_control_residualize(np.where(common, y, np.nan), controls),
            control_label,
        )

    output_dir = plot_path or "plots/hubble"
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(
        output_dir, "full_residuals_debiased_partial_controls.pdf"
    )
    table_path = os.path.join(output_dir, "partial_control_residuals.csv")
    index_path = os.path.join(
        output_dir, "partial_control_parameter_index.csv"
    )
    color_norm = mpl.colors.Normalize(vmin=z_min, vmax=z_max)
    color_map = mpl.colormaps["viridis"]
    index_rows = []
    absolute_page = 0

    with PdfPages(pdf_path) as pdf:
        for section, section_fields in groups:
            if not section_fields:
                continue
            section_pages = int(math.ceil(len(section_fields) / panels_per_page))
            for section_page, start in enumerate(
                range(0, len(section_fields), panels_per_page), start=1
            ):
                absolute_page += 1
                page_fields = section_fields[start : start + panels_per_page]
                compact = len(page_fields) <= 2
                n_columns = 2 if compact else 4
                n_rows = int(math.ceil(len(page_fields) / n_columns))
                fig, axes_grid = plt.subplots(
                    n_rows,
                    n_columns,
                    figsize=(
                        (8.0 * n_columns, 6.4)
                        if compact
                        else (5.0 * n_columns, 3.3 * n_rows)
                    ),
                    sharey="row",
                    squeeze=False,
                )
                axes = axes_grid.ravel()
                for panel_index, (ax, field) in enumerate(
                    zip(axes, page_fields)
                ):
                    x_partial, r_partial, control_label = partials[field]
                    finite = (
                        np.isfinite(x_partial)
                        & np.isfinite(r_partial)
                        & np.isfinite(z)
                    )
                    xf, rf, zf = x_partial[finite], r_partial[finite], z[finite]
                    ax.scatter(
                        xf,
                        rf,
                        c=zf,
                        cmap=color_map,
                        norm=color_norm,
                        s=10,
                        alpha=0.43,
                        linewidths=0,
                        rasterized=True,
                        zorder=2,
                    )
                    xb, rb, rlo, rhi, zb = _partial_control_binned_trend(
                        xf,
                        rf,
                        zf,
                        bins=trend_bins,
                        min_count=max(3, min_points // 2),
                    )
                    if xb.size:
                        ax.fill_between(
                            xb,
                            rlo,
                            rhi,
                            color="#64748B",
                            alpha=0.10,
                            linewidth=0,
                            zorder=3,
                        )
                        ax.plot(xb, rb, color="white", lw=4.2, zorder=4)
                        ax.plot(xb, rb, color="#111827", lw=2.0, zorder=5)
                        ax.scatter(
                            xb,
                            rb,
                            c=zb,
                            cmap=color_map,
                            norm=color_norm,
                            s=38,
                            edgecolors="#111827",
                            linewidths=0.65,
                            zorder=6,
                        )
                    if xf.size >= 3 and np.nanmax(xf) > np.nanmin(xf):
                        x_low, x_high = np.quantile(xf, [0.01, 0.99])
                        padding = 0.05 * (x_high - x_low)
                        ax.set_xlim(x_low - padding, x_high + padding)
                    ax.axhline(
                        0.0, color="#64748B", lw=0.8, ls=(0, (3, 3)), zorder=1
                    )
                    ax.set_ylim(*_FULL_RESIDUAL_YLIM)
                    ax.set_title(
                        textwrap.fill(field, 34),
                        fontsize=9.5,
                        fontweight="bold",
                        pad=5,
                    )
                    ax.set_xlabel(f"{field} residual after controls", fontsize=8.2)
                    if panel_index % n_columns == 0:
                        ax.set_ylabel(r"Hubble $R$ residual after controls [mag]")
                    else:
                        ax.tick_params(labelleft=False)
                    ax.text(
                        0.97,
                        0.04,
                        f"Controls: {control_label}",
                        transform=ax.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=7.2,
                        color="#334155",
                        bbox={
                            "boxstyle": "round,pad=0.25",
                            "facecolor": "white",
                            "edgecolor": "#CBD5E1",
                            "alpha": 0.86,
                        },
                        zorder=8,
                    )
                    ax.grid(True, color="#E2E8F0", lw=0.55, alpha=0.7, zorder=0)
                    ax.spines[["top", "right"]].set_visible(False)
                for ax in axes[len(page_fields) :]:
                    ax.axis("off")

                fig.suptitle(
                    f"{section} — partial-regression Hubble residuals",
                    fontsize=17,
                    fontweight="bold",
                    y=0.992,
                )
                if section.startswith("Hubble-fit"):
                    explanation = (
                        r"For each fitted predictor, both $X$ and $R$ are "
                        r"residualized against redshift-bin fixed effects and "
                        r"the other fitted predictor."
                    )
                else:
                    explanation = (
                        r"For each SED field, both $X$ and $R$ are residualized "
                        r"against redshift-bin fixed effects, $\log\sigma_{\rm UV}$, "
                        r"and $\log\tau_{\rm UV,rf}$."
                    )
                fig.text(
                    0.5,
                    0.865 if compact else 0.950,
                    explanation
                    + "\n"
                    + r"A non-flat black trend indicates an association beyond "
                    + r"the listed controls; color shows remaining redshift structure."
                    + "\n"
                    + f"Section page {section_page}/{section_pages}; "
                    + f"{len(table):,} post-cut quasars.",
                    ha="center",
                    va="center",
                    fontsize=11.2,
                    linespacing=1.3,
                    color="#334155",
                )
                layout_top = 0.76 if compact else 0.895
                fig.tight_layout(
                    rect=(0.025, 0.02, 0.93, layout_top),
                    h_pad=1.55,
                    w_pad=1.1,
                )
                colorbar_ax = fig.add_axes([0.95, 0.12, 0.012, 0.76])
                colorbar = fig.colorbar(
                    mpl.cm.ScalarMappable(norm=color_norm, cmap=color_map),
                    cax=colorbar_ax,
                )
                colorbar.set_label("Redshift  $z$", fontsize=10)
                pdf.savefig(fig, dpi=180)
                if show:
                    plt.show()
                plt.close(fig)
                index_rows.extend(
                    {
                        "pdf_page": absolute_page,
                        "section": section,
                        "section_page": section_page,
                        "panel": panel + 1,
                        "field": field,
                        "controls": partials[field][2],
                    }
                    for panel, field in enumerate(page_fields)
                )

    export = {
        "object_id": (
            table["object_id"].to_numpy()
            if "object_id" in table
            else table.index.to_numpy()
        ),
        "z": z,
        "residuals": y,
        "z_control_bin": z_bin.astype(str).to_numpy(),
    }
    for field in fields:
        export[field] = pd.to_numeric(table[field], errors="coerce").to_numpy()
        export[f"{field}_partial"] = partials[field][0]
        export[f"R_partial_for_{field}"] = partials[field][1]
    pd.DataFrame(export).to_csv(table_path, index=False)
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    return {
        "pdf": pdf_path,
        "residuals_csv": table_path,
        "parameter_index_csv": index_path,
    }


def plot_parameter_residual_diagnostics(
    df_agn,
    residuals,
    residuals_err=None,
    *,
    plot_path="plots/hubble",
    z_range=(0.44, 3.16),
    show=False,
    min_points=10,
    panels_per_page=6,
    top_n=40,
    nbins=10,
):
    """
    Rank and plot scalar numeric parameters against final Hubble residuals.

    The ranking emphasizes parameters that reduce the residual-redshift
    Spearman correlation when conditioned upon. This is a diagnostic priority,
    not a causal attribution.
    """
    if "z" not in df_agn.columns:
        raise ValueError("df_agn must contain a 'z' column.")

    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 1 or residuals.size != len(df_agn):
        raise ValueError(
            f"residuals length {residuals.size} does not match dataframe length "
            f"{len(df_agn)}."
        )
    if residuals_err is None:
        residuals_err = np.full(residuals.shape, np.nan, dtype=float)
    else:
        residuals_err = np.asarray(residuals_err, dtype=float)
        if residuals_err.ndim != 1 or residuals_err.size != len(df_agn):
            raise ValueError(
                f"residuals_err length {residuals_err.size} does not match "
                f"dataframe length {len(df_agn)}."
            )

    if int(min_points) < 3:
        raise ValueError("min_points must be at least 3.")
    if int(panels_per_page) < 1:
        raise ValueError("panels_per_page must be positive.")
    if int(top_n) < 1:
        raise ValueError("top_n must be positive.")

    z_min, z_max = (float(z_range[0]), float(z_range[1]))
    if not (np.isfinite(z_min) and np.isfinite(z_max) and z_min < z_max):
        raise ValueError("z_range must contain two finite, increasing values.")

    z = pd.to_numeric(df_agn["z"], errors="coerce").to_numpy(dtype=float)
    analysis_mask = (
        np.isfinite(z)
        & np.isfinite(residuals)
        & (z >= z_min)
        & (z <= z_max)
    )
    if np.count_nonzero(analysis_mask) < int(min_points):
        raise ValueError(
            "Too few finite in-range residuals for parameter diagnostics: "
            f"{np.count_nonzero(analysis_mask)} < {int(min_points)}."
        )

    weighted_mask = (
        analysis_mask
        & np.isfinite(residuals_err)
        & (residuals_err > 0.0)
    )
    use_weighted = np.count_nonzero(weighted_mask) >= int(min_points)
    trend_mask = weighted_mask if use_weighted else analysis_mask
    redshift_trend = build_smooth_trend_1d(
        z[trend_mask],
        residuals[trend_mask],
        yerr=residuals_err[trend_mask] if use_weighted else None,
        frac=0.25,
        it=1,
        min_points=int(min_points),
        fallback_bins=max(int(nbins), 3),
    )
    residuals_detrended = residuals - np.asarray(redshift_trend(z), dtype=float)

    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    os.makedirs(diagnostics_path, exist_ok=True)
    summary_path = os.path.join(
        diagnostics_path,
        "hubble_parameter_residual_summary.pdf",
    )
    atlas_path = os.path.join(
        diagnostics_path,
        "hubble_parameter_residual_atlas.pdf",
    )
    rankings_path = os.path.join(
        diagnostics_path,
        "hubble_parameter_residual_rankings.csv",
    )
    skipped_path = os.path.join(
        diagnostics_path,
        "hubble_parameter_residual_skipped.csv",
    )

    spectra_fit_columns = set(df_agn.attrs.get("spectra_fit_columns", ()))
    identifier_columns = {"object_id", "sdss_name"}
    skipped_rows = []
    ranking_rows = []
    parameter_values = {}

    def _record_skip(parameter, reason, *, finite_count=0, unique_count=0):
        column = df_agn[parameter]
        column_dtype = (
            str(column.dtype)
            if isinstance(column, pd.Series)
            else "duplicate_column_name"
        )
        skipped_rows.append(
            {
                "parameter": parameter,
                "reason": reason,
                "dtype": column_dtype,
                "finite_in_range": int(finite_count),
                "unique_finite_in_range": int(unique_count),
                "source": (
                    "spectra_fit_csv"
                    if parameter in spectra_fit_columns
                    else "hdf5_or_derived"
                ),
            }
        )

    def _spearman(a, b):
        if len(a) < 3:
            return np.nan, np.nan
        if np.unique(a).size < 2 or np.unique(b).size < 2:
            return np.nan, np.nan
        result = spearmanr(a, b, nan_policy="omit")
        statistic = float(result.statistic)
        pvalue = float(result.pvalue)
        return statistic, pvalue

    for parameter in df_agn.columns:
        if parameter in identifier_columns:
            _record_skip(parameter, "identifier")
            continue
        if parameter == "z":
            _record_skip(parameter, "analysis_axis")
            continue

        series = df_agn[parameter]
        if (
            not isinstance(series, pd.Series)
            or not (
                pd.api.types.is_numeric_dtype(series.dtype)
                or pd.api.types.is_bool_dtype(series.dtype)
            )
        ):
            _record_skip(parameter, "non_numeric_or_non_scalar")
            continue

        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        finite_mask = analysis_mask & np.isfinite(values)
        finite_count = int(np.count_nonzero(finite_mask))
        if finite_count < int(min_points):
            _record_skip(
                parameter,
                "insufficient_finite_values",
                finite_count=finite_count,
            )
            continue

        unique_count = int(np.unique(values[finite_mask]).size)
        if unique_count < 2:
            _record_skip(
                parameter,
                "constant",
                finite_count=finite_count,
                unique_count=unique_count,
            )
            continue

        p = values[finite_mask]
        z_use = z[finite_mask]
        r_use = residuals[finite_mask]
        rz_use = residuals_detrended[finite_mask]
        rho_pz, p_pz = _spearman(p, z_use)
        rho_pr, p_pr = _spearman(p, r_use)
        rho_prz, p_prz = _spearman(p, rz_use)
        rho_rz, p_rz = _spearman(r_use, z_use)

        partial_denominator = np.sqrt(
            max(0.0, 1.0 - rho_pr**2)
            * max(0.0, 1.0 - rho_pz**2)
        )
        if np.isfinite(partial_denominator) and partial_denominator > 1e-12:
            partial_rho = (rho_rz - rho_pr * rho_pz) / partial_denominator
            partial_rho = float(np.clip(partial_rho, -1.0, 1.0))
        else:
            partial_rho = np.nan

        attenuation = (
            float(abs(rho_rz) - abs(partial_rho))
            if np.isfinite(rho_rz) and np.isfinite(partial_rho)
            else np.nan
        )
        ranking_rows.append(
            {
                "parameter": parameter,
                "source": (
                    "spectra_fit_csv"
                    if parameter in spectra_fit_columns
                    else "hdf5_or_derived"
                ),
                "finite_in_range": finite_count,
                "coverage_in_range": finite_count / int(np.count_nonzero(analysis_mask)),
                "unique_finite_in_range": unique_count,
                "rho_parameter_redshift": rho_pz,
                "p_parameter_redshift": p_pz,
                "rho_parameter_residual": rho_pr,
                "p_parameter_residual": p_pr,
                "rho_parameter_detrended_residual": rho_prz,
                "p_parameter_detrended_residual": p_prz,
                "rho_residual_redshift": rho_rz,
                "p_residual_redshift": p_rz,
                "partial_rho_residual_redshift_given_parameter": partial_rho,
                "redshift_correlation_attenuation": attenuation,
                "abs_rho_parameter_detrended_residual": abs(rho_prz),
            }
        )
        parameter_values[parameter] = values

    ranking_columns = [
        "parameter",
        "source",
        "finite_in_range",
        "coverage_in_range",
        "unique_finite_in_range",
        "rho_parameter_redshift",
        "p_parameter_redshift",
        "q_parameter_redshift",
        "rho_parameter_residual",
        "p_parameter_residual",
        "q_parameter_residual",
        "rho_parameter_detrended_residual",
        "p_parameter_detrended_residual",
        "q_parameter_detrended_residual",
        "rho_residual_redshift",
        "p_residual_redshift",
        "partial_rho_residual_redshift_given_parameter",
        "redshift_correlation_attenuation",
        "abs_rho_parameter_detrended_residual",
    ]

    rankings = pd.DataFrame(ranking_rows)

    def _benjamini_hochberg(pvalues):
        pvalues = np.asarray(pvalues, dtype=float)
        adjusted = np.full(pvalues.shape, np.nan, dtype=float)
        finite = np.isfinite(pvalues)
        if not np.any(finite):
            return adjusted
        p_finite = pvalues[finite]
        order = np.argsort(p_finite)
        ranked = p_finite[order]
        n_tests = len(ranked)
        q_ranked = ranked * n_tests / np.arange(1, n_tests + 1, dtype=float)
        q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
        q_finite = np.empty_like(q_ranked)
        q_finite[order] = np.clip(q_ranked, 0.0, 1.0)
        adjusted[finite] = q_finite
        return adjusted

    if rankings.empty:
        rankings = pd.DataFrame(columns=ranking_columns)
    else:
        rankings["q_parameter_redshift"] = _benjamini_hochberg(
            rankings["p_parameter_redshift"]
        )
        rankings["q_parameter_residual"] = _benjamini_hochberg(
            rankings["p_parameter_residual"]
        )
        rankings["q_parameter_detrended_residual"] = _benjamini_hochberg(
            rankings["p_parameter_detrended_residual"]
        )
        rankings = rankings.sort_values(
            [
                "redshift_correlation_attenuation",
                "abs_rho_parameter_detrended_residual",
                "parameter",
            ],
            ascending=[False, False, True],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
        rankings = rankings.loc[:, ranking_columns]

    skipped_columns = [
        "parameter",
        "reason",
        "dtype",
        "finite_in_range",
        "unique_finite_in_range",
        "source",
    ]
    skipped = pd.DataFrame(skipped_rows, columns=skipped_columns)
    rankings.to_csv(rankings_path, index=False)
    skipped.to_csv(skipped_path, index=False)

    top = rankings.head(int(top_n))
    if top.empty:
        fig_summary, ax_summary = plt.subplots(figsize=(10.0, 4.0))
        ax_summary.axis("off")
        ax_summary.text(
            0.5,
            0.5,
            "No eligible numeric parameters.",
            ha="center",
            va="center",
        )
    else:
        heatmap_columns = [
            "rho_parameter_redshift",
            "rho_parameter_residual",
            "rho_parameter_detrended_residual",
            "partial_rho_residual_redshift_given_parameter",
        ]
        heatmap_labels = [
            r"$\rho(p,z)$",
            r"$\rho(p,r)$",
            r"$\rho(p,r_z)$",
            r"$\rho(r,z\mid p)$",
        ]
        heatmap = top[heatmap_columns].to_numpy(dtype=float)
        fig_height = max(4.5, 0.28 * len(top) + 2.4)
        fig_summary, ax_summary = plt.subplots(
            figsize=(10.5, fig_height),
        )
        image_handle = ax_summary.imshow(
            heatmap,
            aspect="auto",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
        )
        parameter_labels = [
            (
                f"[SED] {row.parameter} (N={row.finite_in_range})"
                if row.source == "spectra_fit_csv"
                else f"{row.parameter} (N={row.finite_in_range})"
            )
            for row in top.itertuples()
        ]
        ax_summary.set_yticks(np.arange(len(top)))
        ax_summary.set_yticklabels(parameter_labels, fontsize=8)
        ax_summary.set_xticks(np.arange(len(heatmap_labels)))
        ax_summary.set_xticklabels(heatmap_labels, fontsize=11)
        for row_idx in range(heatmap.shape[0]):
            for col_idx in range(heatmap.shape[1]):
                value = heatmap[row_idx, col_idx]
                if np.isfinite(value):
                    ax_summary.text(
                        col_idx,
                        row_idx,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if abs(value) > 0.55 else "black",
                    )
        cbar = fig_summary.colorbar(image_handle, ax=ax_summary, pad=0.02)
        cbar.set_label("Spearman correlation")
        ax_summary.set_title(
            "Hubble-residual parameter priorities\n"
            "ranked by reduction in residual–redshift correlation; "
            "diagnostic, not causal",
            fontsize=12,
        )
    fig_summary.tight_layout()
    summary_pdf = _save_figure(
        fig_summary,
        summary_path,
        dpi=200,
        show=show,
    )

    def _quantile_binned_medians(x, y, *, minimum_per_bin):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.size < minimum_per_bin:
            return np.array([]), np.array([])
        unique_x = np.unique(x)
        if unique_x.size <= int(nbins):
            groups = [x == value for value in unique_x]
        else:
            edges = np.unique(np.nanquantile(x, np.linspace(0.0, 1.0, int(nbins) + 1)))
            if edges.size < 2:
                return np.array([]), np.array([])
            groups = []
            for edge_idx in range(edges.size - 1):
                if edge_idx == edges.size - 2:
                    groups.append((x >= edges[edge_idx]) & (x <= edges[edge_idx + 1]))
                else:
                    groups.append((x >= edges[edge_idx]) & (x < edges[edge_idx + 1]))
        x_median = []
        y_median = []
        for group in groups:
            if np.count_nonzero(group) < minimum_per_bin:
                continue
            x_median.append(float(np.nanmedian(x[group])))
            y_median.append(float(np.nanmedian(y[group])))
        return np.asarray(x_median), np.asarray(y_median)

    with PdfPages(atlas_path) as pdf:
        if rankings.empty:
            fig_atlas, ax_atlas = plt.subplots(figsize=(11.0, 8.5))
            ax_atlas.axis("off")
            ax_atlas.text(
                0.5,
                0.5,
                "No eligible numeric parameters.",
                ha="center",
                va="center",
            )
            pdf.savefig(fig_atlas, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig_atlas)
        else:
            n_cols = min(3, int(panels_per_page))
            n_rows = int(math.ceil(int(panels_per_page) / n_cols))
            color_norm = mpl.colors.Normalize(vmin=z_min, vmax=z_max)
            color_map = "viridis"
            bin_minimum = max(3, int(min_points) // 2)

            for page_start in range(0, len(rankings), int(panels_per_page)):
                page = rankings.iloc[
                    page_start : page_start + int(panels_per_page)
                ]
                fig_atlas, axes_grid = plt.subplots(
                    n_rows,
                    n_cols,
                    figsize=(5.0 * n_cols, 3.8 * n_rows),
                    squeeze=False,
                )
                axes = axes_grid.ravel()
                for ax, row in zip(axes, page.itertuples()):
                    values = parameter_values[row.parameter]
                    mask = analysis_mask & np.isfinite(values)
                    x_use = values[mask]
                    r_use = residuals[mask]
                    rz_use = residuals_detrended[mask]
                    z_use = z[mask]

                    ax.scatter(
                        x_use,
                        r_use,
                        c=z_use,
                        cmap=color_map,
                        norm=color_norm,
                        s=8,
                        alpha=0.28,
                        linewidths=0,
                        rasterized=True,
                    )
                    x_raw, y_raw = _quantile_binned_medians(
                        x_use,
                        r_use,
                        minimum_per_bin=bin_minimum,
                    )
                    x_detrended, y_detrended = _quantile_binned_medians(
                        x_use,
                        rz_use,
                        minimum_per_bin=bin_minimum,
                    )
                    if x_raw.size:
                        ax.plot(
                            x_raw,
                            y_raw,
                            color="tab:red",
                            lw=2.0,
                            label="raw median",
                        )
                    if x_detrended.size:
                        ax.plot(
                            x_detrended,
                            y_detrended,
                            color="tab:orange",
                            lw=2.0,
                            linestyle="--",
                            label=r"$r_z$ median",
                        )
                    ax.axhline(0.0, color="black", lw=0.8, alpha=0.65)

                    if x_use.size >= 2:
                        x_lo, x_hi = np.nanpercentile(x_use, [1.0, 99.0])
                        if np.isfinite(x_lo) and np.isfinite(x_hi) and x_hi > x_lo:
                            padding = 0.04 * (x_hi - x_lo)
                            ax.set_xlim(x_lo - padding, x_hi + padding)

                    source_tag = " [SED]" if row.source == "spectra_fit_csv" else ""
                    ax.set_title(f"{row.parameter}{source_tag}", fontsize=10)
                    ax.set_xlabel(row.parameter, fontsize=9)
                    ax.set_ylabel("Debiased Hubble residual (mag)", fontsize=9)
                    ax.text(
                        0.02,
                        0.98,
                        (
                            f"N={row.finite_in_range}, "
                            f"coverage={row.coverage_in_range:.1%}\n"
                            rf"$\rho(p,z)$={row.rho_parameter_redshift:.2f}; "
                            rf"$\rho(p,r)$={row.rho_parameter_residual:.2f}"
                            "\n"
                            rf"$\rho(p,r_z)$="
                            f"{row.rho_parameter_detrended_residual:.2f}; "
                            rf"$\Delta|\rho_z|$="
                            f"{row.redshift_correlation_attenuation:.2f}"
                        ),
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        fontsize=7,
                        bbox={
                            "facecolor": "white",
                            "alpha": 0.75,
                            "edgecolor": "none",
                            "pad": 2.0,
                        },
                    )
                    ax.grid(True, alpha=0.15)

                for ax in axes[len(page) :]:
                    ax.axis("off")

                active_axes = [ax for ax in axes[: len(page)] if ax.axison]
                fig_atlas.suptitle(
                    "Exhaustive Hubble-residual parameter atlas "
                    f"({page_start + 1}–{page_start + len(page)} of "
                    f"{len(rankings)}; central 1–99% x-range)",
                    fontsize=13,
                    y=0.985,
                )
                fig_atlas.subplots_adjust(
                    left=0.07,
                    right=0.89,
                    bottom=0.08,
                    top=0.84,
                    wspace=0.30,
                    hspace=0.42,
                )
                if active_axes:
                    scalar_mappable = mpl.cm.ScalarMappable(
                        norm=color_norm,
                        cmap=color_map,
                    )
                    scalar_mappable.set_array([])
                    colorbar_ax = fig_atlas.add_axes([0.915, 0.16, 0.012, 0.66])
                    colorbar = fig_atlas.colorbar(
                        scalar_mappable,
                        cax=colorbar_ax,
                    )
                    colorbar.set_label("Redshift")
                    handles, labels = [], []
                    for legend_ax in active_axes:
                        handles, labels = legend_ax.get_legend_handles_labels()
                        if handles:
                            break
                    if handles:
                        fig_atlas.legend(
                            handles,
                            labels,
                            loc="upper center",
                            bbox_to_anchor=(0.5, 0.945),
                            ncol=2,
                            frameon=False,
                        )
                pdf.savefig(fig_atlas, dpi=100, bbox_inches="tight")
                if show:
                    plt.show()
                plt.close(fig_atlas)

    return {
        "summary_pdf": summary_pdf,
        "atlas_pdf": atlas_path,
        "rankings_csv": rankings_path,
        "skipped_csv": skipped_path,
    }


def plot_redshift_wiggle_diagnostics(
    df_agn,
    residuals_biased,
    residuals_biased_err,
    residuals_debiased,
    residuals_debiased_err,
    *,
    plot_path="plots/hubble",
    z_range=(0.44, 3.16),
    show=False,
    min_points=10,
    n_z_bins=12,
    n_bootstrap=300,
    n_permutations=500,
    cv_folds=5,
    atlas_top_n=None,
    panels_per_page=6,
    random_seed=93741,
):
    """
    Diagnose local redshift structure in final Hubble residuals.

    Unlike the parameter-first exhaustive audit, this view keeps redshift on
    the x axis. Parameter priorities are based on out-of-fold reduction of the
    RMS redshift wiggle and largest adjacent-bin jump. They are diagnostic
    priorities, not causal conclusions.
    """
    if "z" not in df_agn.columns:
        raise ValueError("df_agn must contain a 'z' column.")
    n_rows = len(df_agn)

    def _aligned(values, name):
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or array.size != n_rows:
            raise ValueError(
                f"{name} length {array.size} does not match dataframe length "
                f"{n_rows}."
            )
        return array

    rb = _aligned(residuals_biased, "residuals_biased")
    eb = _aligned(residuals_biased_err, "residuals_biased_err")
    rd = _aligned(residuals_debiased, "residuals_debiased")
    ed = _aligned(residuals_debiased_err, "residuals_debiased_err")
    z = pd.to_numeric(df_agn["z"], errors="coerce").to_numpy(dtype=float)
    z_min, z_max = map(float, z_range)
    if not (np.isfinite(z_min) and np.isfinite(z_max) and z_min < z_max):
        raise ValueError("z_range must contain two finite, increasing values.")
    if int(min_points) < 3:
        raise ValueError("min_points must be at least 3.")
    if int(n_z_bins) < 4:
        raise ValueError("n_z_bins must be at least 4.")
    if int(cv_folds) < 2:
        raise ValueError("cv_folds must be at least 2.")

    base_mask = (
        np.isfinite(z)
        & np.isfinite(rd)
        & (z >= z_min)
        & (z <= z_max)
    )
    if np.count_nonzero(base_mask) < int(min_points):
        raise ValueError("Too few finite in-range debiased residuals.")
    rng = np.random.default_rng(int(random_seed))
    diagnostics_path = os.path.join(plot_path or "plots/hubble", "diagnostics")
    os.makedirs(diagnostics_path, exist_ok=True)
    overview_path = os.path.join(
        diagnostics_path, "hubble_redshift_wiggle_overview.pdf"
    )
    change_path = os.path.join(
        diagnostics_path, "hubble_redshift_change_points.csv"
    )
    transitions_path = os.path.join(
        diagnostics_path, "hubble_redshift_physical_transitions.csv"
    )
    rankings_path = os.path.join(
        diagnostics_path, "hubble_redshift_parameter_wiggle_rankings.csv"
    )
    atlas_path = os.path.join(
        diagnostics_path, "hubble_redshift_parameter_wiggle_atlas.pdf"
    )

    z_base = z[base_mask]
    rd_base = rd[base_mask]
    # Equal-count bins make sparse ends of the redshift distribution visible
    # without allowing dense regions to dominate the wiggle metric.
    z_edges = np.unique(
        np.nanquantile(
            z_base,
            np.linspace(0.0, 1.0, min(int(n_z_bins), len(z_base)) + 1),
        )
    )
    if z_edges.size < 5:
        z_edges = np.linspace(z_min, z_max, int(n_z_bins) + 1)
    z_edges[0] = min(z_edges[0], z_min)
    z_edges[-1] = max(z_edges[-1], z_max)

    def _binned(z_values, y_values, edges=z_edges, minimum=3):
        z_values = np.asarray(z_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        centers, medians, counts = [], [], []
        for bin_index in range(len(edges) - 1):
            if bin_index == len(edges) - 2:
                keep = (
                    (z_values >= edges[bin_index])
                    & (z_values <= edges[bin_index + 1])
                )
            else:
                keep = (
                    (z_values >= edges[bin_index])
                    & (z_values < edges[bin_index + 1])
                )
            count = int(np.count_nonzero(keep))
            counts.append(count)
            centers.append(
                float(np.nanmedian(z_values[keep]))
                if count
                else 0.5 * (edges[bin_index] + edges[bin_index + 1])
            )
            medians.append(
                float(np.nanmedian(y_values[keep]))
                if count >= int(minimum)
                else np.nan
            )
        return (
            np.asarray(centers),
            np.asarray(medians),
            np.asarray(counts, dtype=int),
        )

    def _wiggle_metrics(z_values, y_values):
        _, medians, _ = _binned(z_values, y_values)
        finite = np.isfinite(medians)
        if np.count_nonzero(finite) < 3:
            return np.nan, np.nan
        centered = medians[finite] - np.nanmedian(medians[finite])
        rms = float(np.sqrt(np.nanmean(centered**2)))
        adjacent = np.abs(np.diff(medians))
        max_jump = (
            float(np.nanmax(adjacent))
            if np.any(np.isfinite(adjacent))
            else np.nan
        )
        return rms, max_jump

    def _bootstrap_binned(z_values, y_values):
        centers, medians, counts = _binned(z_values, y_values)
        draws = np.full((max(int(n_bootstrap), 0), len(medians)), np.nan)
        for bin_index in range(len(z_edges) - 1):
            right = (
                z_values <= z_edges[bin_index + 1]
                if bin_index == len(z_edges) - 2
                else z_values < z_edges[bin_index + 1]
            )
            values = y_values[
                (z_values >= z_edges[bin_index]) & right & np.isfinite(y_values)
            ]
            if values.size < 3:
                continue
            for draw_index in range(draws.shape[0]):
                draws[draw_index, bin_index] = np.median(
                    rng.choice(values, size=values.size, replace=True)
                )
        if draws.shape[0]:
            lower, upper = np.nanpercentile(draws, [16.0, 84.0], axis=0)
        else:
            lower = upper = np.full(len(medians), np.nan)
        return centers, medians, counts, lower, upper

    # Robust adjacent-window change-point scan. The permutation distribution
    # uses the largest statistic anywhere in z, providing a look-elsewhere
    # correction for the scan.
    order = np.argsort(z_base)
    z_sorted = z_base[order]
    r_sorted = rd_base[order]
    window = max(int(min_points), int(np.ceil(0.06 * len(z_sorted))))
    step = max(1, len(z_sorted) // 80)
    split_indices = np.arange(window, len(z_sorted) - window + 1, step)

    def _change_statistics(values):
        statistics = []
        jumps = []
        for split in split_indices:
            left = values[split - window : split]
            right = values[split : split + window]
            jump = float(np.nanmedian(right) - np.nanmedian(left))
            left_scale = 1.4826 * np.nanmedian(
                np.abs(left - np.nanmedian(left))
            )
            right_scale = 1.4826 * np.nanmedian(
                np.abs(right - np.nanmedian(right))
            )
            standard_error = np.sqrt(
                left_scale**2 / max(len(left), 1)
                + right_scale**2 / max(len(right), 1)
            )
            statistics.append(
                jump / max(float(standard_error), np.finfo(float).eps)
            )
            jumps.append(jump)
        return np.asarray(statistics), np.asarray(jumps)

    change_stat, change_jump = _change_statistics(r_sorted)
    null_max = np.full(max(int(n_permutations), 0), np.nan)
    for permutation_index in range(null_max.size):
        permuted_stat, _ = _change_statistics(rng.permutation(r_sorted))
        if permuted_stat.size:
            null_max[permutation_index] = np.nanmax(np.abs(permuted_stat))
    change_rows = []
    for row_index, split in enumerate(split_indices):
        abs_stat = abs(change_stat[row_index])
        global_p = (
            (1.0 + np.count_nonzero(null_max >= abs_stat))
            / (1.0 + np.count_nonzero(np.isfinite(null_max)))
            if null_max.size
            else np.nan
        )
        change_rows.append(
            {
                "z_split": 0.5 * (z_sorted[split - 1] + z_sorted[split]),
                "median_jump": change_jump[row_index],
                "robust_jump_statistic": change_stat[row_index],
                "look_elsewhere_pvalue": global_p,
                "window_size_each_side": window,
            }
        )
    change_points = pd.DataFrame(change_rows)
    if not change_points.empty:
        change_points = change_points.iloc[
            np.argsort(
                -np.abs(change_points["robust_jump_statistic"].to_numpy())
            )
        ].reset_index(drop=True)
    change_points.to_csv(change_path, index=False)

    transition_rows = []
    transition_features = {
        **{
            name: float(config["lambda_rest"])
            for name, config in _BLR_LINE_MODELS.items()
        },
        "2500A continuum": 2500.0,
    }
    for feature, wavelength in transition_features.items():
        for band, (blue_edge, red_edge) in _SDSS_FILTER_EDGES_OBS.items():
            for edge_name, observed_edge in (
                ("enters", blue_edge),
                ("leaves", red_edge),
            ):
                transition_z = observed_edge / wavelength - 1.0
                if z_min <= transition_z <= z_max:
                    transition_rows.append(
                        {
                            "z": transition_z,
                            "feature": feature,
                            "filter": band,
                            "edge": edge_name,
                            "label": f"{feature} {edge_name} {band}",
                        }
                    )
    transitions = pd.DataFrame(
        transition_rows,
        columns=["z", "feature", "filter", "edge", "label"],
    ).sort_values("z", ignore_index=True)
    transitions.to_csv(transitions_path, index=False)

    spectra_fit_columns = set(df_agn.attrs.get("spectra_fit_columns", ()))
    base_counts = _binned(z_base, rd_base)[2]
    parameter_values = {}
    ranking_rows = []
    identifiers = {"object_id", "sdss_name", "z"}
    for parameter in df_agn.columns:
        if parameter in identifiers:
            continue
        series = df_agn[parameter]
        if not isinstance(series, pd.Series) or not (
            pd.api.types.is_numeric_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
        ):
            continue
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        mask = base_mask & np.isfinite(values)
        finite_count = int(np.count_nonzero(mask))
        if finite_count < int(min_points) or np.unique(values[mask]).size < 2:
            continue
        z_use = z[mask]
        r_use = rd[mask]
        e_use = ed[mask]
        p_use = values[mask]
        # Rank scaling makes the simple cross-validated model robust to units
        # and extreme values while retaining monotonic parameter effects.
        p_rank = (
            pd.Series(p_use).rank(method="average").to_numpy(dtype=float)
            - 0.5
        ) / len(p_use)
        sorted_local = np.argsort(z_use, kind="stable")
        fold_id = np.empty(len(z_use), dtype=int)
        fold_id[sorted_local] = np.arange(len(z_use)) % int(cv_folds)
        predictions = np.full(len(z_use), np.nan)
        fold_reductions = []
        for fold in range(int(cv_folds)):
            test = fold_id == fold
            train = ~test
            if np.count_nonzero(train) < 3 or np.count_nonzero(test) < 3:
                continue
            design_train = np.column_stack(
                [np.ones(np.count_nonzero(train)), p_rank[train]]
            )
            weights = np.ones(np.count_nonzero(train))
            good_error = np.isfinite(e_use[train]) & (e_use[train] > 0)
            weights[good_error] = 1.0 / np.square(e_use[train][good_error])
            sqrt_weight = np.sqrt(weights)
            coefficients = np.linalg.lstsq(
                design_train * sqrt_weight[:, None],
                r_use[train] * sqrt_weight,
                rcond=None,
            )[0]
            predictions[test] = (
                coefficients[0] + coefficients[1] * p_rank[test]
            )
            before_fold = _wiggle_metrics(z_use[test], r_use[test])[0]
            after_fold = _wiggle_metrics(
                z_use[test], r_use[test] - predictions[test]
            )[0]
            if np.isfinite(before_fold) and np.isfinite(after_fold):
                fold_reductions.append(before_fold - after_fold)
        valid_prediction = np.isfinite(predictions)
        baseline_rms, baseline_jump = _wiggle_metrics(
            z_use[valid_prediction], r_use[valid_prediction]
        )
        adjusted_rms, adjusted_jump = _wiggle_metrics(
            z_use[valid_prediction],
            r_use[valid_prediction] - predictions[valid_prediction],
        )
        _, _, parameter_counts = _binned(z_use, r_use)
        with np.errstate(divide="ignore", invalid="ignore"):
            bin_coverage = np.divide(
                parameter_counts,
                base_counts,
                out=np.zeros_like(parameter_counts, dtype=float),
                where=base_counts > 0,
            )
        populated = base_counts >= 3
        locally_covered = populated & (parameter_counts >= 3) & (
            bin_coverage >= 0.2
        )
        covered_fraction = (
            float(np.count_nonzero(locally_covered) / np.count_nonzero(populated))
            if np.any(populated)
            else 0.0
        )
        min_bin_coverage = (
            float(np.min(bin_coverage[populated])) if np.any(populated) else 0.0
        )
        reliable = bool(
            finite_count >= max(30, int(min_points))
            and covered_fraction >= 0.8
            and min_bin_coverage >= 0.1
        )
        reduction_rms = (
            baseline_rms - adjusted_rms
            if np.isfinite(baseline_rms) and np.isfinite(adjusted_rms)
            else np.nan
        )
        reduction_jump = (
            baseline_jump - adjusted_jump
            if np.isfinite(baseline_jump) and np.isfinite(adjusted_jump)
            else np.nan
        )
        alias_group = re.sub(r"[^a-z0-9]+", "", parameter.lower())
        ranking_rows.append(
            {
                "parameter": parameter,
                "source": (
                    "spectra_fit_csv"
                    if parameter in spectra_fit_columns
                    else "hdf5_or_derived"
                ),
                "alias_group": alias_group,
                "finite_in_range": finite_count,
                "coverage_in_range": finite_count / np.count_nonzero(base_mask),
                "redshift_bins_covered_fraction": covered_fraction,
                "minimum_redshift_bin_coverage": min_bin_coverage,
                "reliable_redshift_coverage": reliable,
                "cv_baseline_wiggle_rms": baseline_rms,
                "cv_adjusted_wiggle_rms": adjusted_rms,
                "cv_wiggle_rms_reduction": reduction_rms,
                "cv_baseline_max_jump": baseline_jump,
                "cv_adjusted_max_jump": adjusted_jump,
                "cv_max_jump_reduction": reduction_jump,
                "fold_reduction_positive_fraction": (
                    float(np.mean(np.asarray(fold_reductions) > 0))
                    if fold_reductions
                    else np.nan
                ),
            }
        )
        parameter_values[parameter] = values

    rankings = pd.DataFrame(ranking_rows)
    if not rankings.empty:
        rankings = rankings.sort_values(
            [
                "reliable_redshift_coverage",
                "cv_wiggle_rms_reduction",
                "cv_max_jump_reduction",
                "fold_reduction_positive_fraction",
            ],
            ascending=[False, False, False, False],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
        rankings["is_alias_representative"] = ~rankings.duplicated("alias_group")
    else:
        rankings["is_alias_representative"] = pd.Series(dtype=bool)
    rankings.to_csv(rankings_path, index=False)

    rb_mask = base_mask & np.isfinite(rb)
    correction = rb - rd
    correction_mask = base_mask & np.isfinite(correction)
    zb, rb_med, rb_count, rb_lo, rb_hi = _bootstrap_binned(
        z[rb_mask], rb[rb_mask]
    )
    zd, rd_med, rd_count, rd_lo, rd_hi = _bootstrap_binned(z_base, rd_base)
    zc, correction_med, _, correction_lo, correction_hi = _bootstrap_binned(
        z[correction_mask], correction[correction_mask]
    )
    fig, axes = plt.subplots(
        4, 1, figsize=(11.0, 13.0), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.8, 1.1, 1.8]},
    )
    axes[0].scatter(
        z_base, rd_base, s=5, alpha=0.12, color="0.35",
        linewidths=0, rasterized=True,
    )
    axes[0].plot(zb, rb_med, color="tab:blue", lw=2, label="biased median")
    axes[0].fill_between(zb, rb_lo, rb_hi, color="tab:blue", alpha=0.15)
    axes[0].plot(zd, rd_med, color="tab:red", lw=2, label="debiased median")
    axes[0].fill_between(zd, rd_lo, rd_hi, color="tab:red", alpha=0.15)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("Hubble residual (mag)")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].set_title(
        "Redshift-first Hubble-residual diagnostic "
        "(bands: bootstrap 16–84%; diagnostic, not causal)"
    )
    axes[1].plot(zc, correction_med, color="tab:purple", lw=2)
    axes[1].fill_between(
        zc, correction_lo, correction_hi, color="tab:purple", alpha=0.18
    )
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("bias correction\n(biased − debiased)")
    width = np.diff(z_edges)
    axes[2].bar(
        0.5 * (z_edges[:-1] + z_edges[1:]),
        rd_count,
        width=0.88 * width,
        color="0.55",
    )
    axes[2].set_ylabel("N")
    if change_rows:
        scan = pd.DataFrame(change_rows).sort_values("z_split")
        axes[3].plot(
            scan["z_split"],
            scan["robust_jump_statistic"],
            color="tab:orange",
            lw=1.8,
        )
        axes[3].axhline(0, color="black", lw=0.8)
        best = change_points.iloc[0]
        axes[3].axvline(best["z_split"], color="tab:red", lw=1.6)
        axes[3].text(
            0.01,
            0.96,
            (
                f"strongest candidate z={best['z_split']:.3f}, "
                f"jump={best['median_jump']:+.3f} mag, "
                f"global p={best['look_elsewhere_pvalue']:.3g}"
            ),
            transform=axes[3].transAxes,
            va="top",
            fontsize=9,
        )
    transition_colors = {
        "C IV": "#4477AA",
        "Mg II": "#228833",
        "Hβ": "#CCBB44",
        "Hα": "#EE6677",
        "2500A continuum": "#AA3377",
    }
    for row in transitions.itertuples():
        axes[3].axvline(
            row.z,
            color=transition_colors.get(row.feature, "0.6"),
            lw=0.7,
            alpha=0.28,
        )
    transition_handles = [
        Line2D([0], [0], color=color, lw=1.5, label=feature)
        for feature, color in transition_colors.items()
        if feature in set(transitions["feature"])
    ]
    if transition_handles:
        axes[3].legend(
            handles=transition_handles,
            title="filter-edge crossings",
            loc="lower right",
            ncol=min(3, len(transition_handles)),
            fontsize=7,
            title_fontsize=7,
            frameon=True,
            framealpha=0.8,
        )
    axes[3].set_ylabel("robust local\njump statistic")
    axes[3].set_xlabel("Redshift")
    axes[3].set_xlim(z_min, z_max)
    for ax in axes:
        ax.grid(True, alpha=0.16)
    fig.tight_layout()
    overview_pdf = _save_figure(fig, overview_path, dpi=200, show=show)

    atlas_rows = rankings[
        rankings["is_alias_representative"].astype(bool)
    ] if not rankings.empty else rankings
    if atlas_top_n is not None:
        atlas_rows = atlas_rows.head(int(atlas_top_n))
    with PdfPages(atlas_path) as pdf:
        if atlas_rows.empty:
            empty_fig, empty_ax = plt.subplots(figsize=(11, 8.5))
            empty_ax.axis("off")
            empty_ax.text(
                0.5, 0.5, "No eligible numeric parameters.",
                ha="center", va="center",
            )
            pdf.savefig(empty_fig)
            plt.close(empty_fig)
        else:
            n_cols = min(3, int(panels_per_page))
            n_rows_page = int(np.ceil(int(panels_per_page) / n_cols))
            for page_start in range(0, len(atlas_rows), int(panels_per_page)):
                page = atlas_rows.iloc[
                    page_start : page_start + int(panels_per_page)
                ]
                page_fig, page_axes = plt.subplots(
                    n_rows_page,
                    n_cols,
                    figsize=(5.1 * n_cols, 3.9 * n_rows_page),
                    squeeze=False,
                    sharex=True,
                )
                axes_flat = page_axes.ravel()
                for ax, row in zip(axes_flat, page.itertuples()):
                    values = parameter_values[row.parameter]
                    mask = base_mask & np.isfinite(values)
                    p = values[mask]
                    z_use = z[mask]
                    r_use = rd[mask]
                    lo, hi = np.nanpercentile(p, [2, 98])
                    color_values = np.clip(p, lo, hi) if hi > lo else p
                    ax.scatter(
                        z_use, r_use, c=color_values, cmap="viridis",
                        s=6, alpha=0.22, linewidths=0, rasterized=True,
                    )
                    quantiles = np.unique(np.nanquantile(p, [0, 1/3, 2/3, 1]))
                    curve_colors = ["#3366AA", "#777777", "#CC3311"]
                    curve_labels = ["low p", "mid p", "high p"]
                    if quantiles.size == 4:
                        for group_index in range(3):
                            group = (p >= quantiles[group_index]) & (
                                p <= quantiles[group_index + 1]
                                if group_index == 2
                                else p < quantiles[group_index + 1]
                            )
                            x_curve, y_curve, _ = _binned(
                                z_use[group], r_use[group], minimum=3
                            )
                            ax.plot(
                                x_curve, y_curve,
                                color=curve_colors[group_index],
                                lw=1.8,
                                label=curve_labels[group_index],
                            )
                    _, _, coverage_counts = _binned(z_use, r_use)
                    coverage = np.divide(
                        coverage_counts,
                        base_counts,
                        out=np.zeros_like(coverage_counts, dtype=float),
                        where=base_counts > 0,
                    )
                    coverage_ax = ax.inset_axes([0.08, 0.03, 0.88, 0.09])
                    coverage_ax.bar(
                        0.5 * (z_edges[:-1] + z_edges[1:]),
                        coverage,
                        width=0.9 * np.diff(z_edges),
                        color="black",
                        alpha=0.18,
                    )
                    coverage_ax.set_ylim(0, 1)
                    coverage_ax.set_xlim(z_min, z_max)
                    coverage_ax.set_xticks([])
                    coverage_ax.set_yticks([0, 1])
                    coverage_ax.tick_params(labelsize=5, length=1)
                    coverage_ax.set_ylabel("cov", fontsize=5, labelpad=0)
                    ax.axhline(0, color="black", lw=0.7, alpha=0.6)
                    source = " [SED]" if row.source == "spectra_fit_csv" else ""
                    reliability = "reliable" if row.reliable_redshift_coverage else "LOW COVERAGE"
                    ax.set_title(
                        f"{row.parameter}{source}\n"
                        f"Δwiggle={row.cv_wiggle_rms_reduction:+.3f}, "
                        f"Δjump={row.cv_max_jump_reduction:+.3f}; {reliability}",
                        fontsize=9,
                    )
                    ax.text(
                        0.98,
                        0.97,
                        f"color: p2={lo:.3g} → p98={hi:.3g}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=6,
                        bbox={
                            "facecolor": "white",
                            "alpha": 0.7,
                            "edgecolor": "none",
                            "pad": 1.0,
                        },
                    )
                    ax.set_ylabel("Debiased residual (mag)")
                    ax.set_xlim(z_min, z_max)
                    ax.grid(True, alpha=0.12)
                for ax in axes_flat[len(page):]:
                    ax.axis("off")
                axes_flat[min(len(page), len(axes_flat)) - 1].set_xlabel(
                    "Redshift"
                )
                handles, labels = axes_flat[0].get_legend_handles_labels()
                if handles:
                    page_fig.legend(
                        handles, labels, frameon=False, ncol=3,
                        loc="upper center", bbox_to_anchor=(0.5, 0.96),
                    )
                page_fig.suptitle(
                    "Parameter-colored residual vs redshift atlas "
                    f"({page_start + 1}–{page_start + len(page)} of "
                    f"{len(atlas_rows)}; ranked by held-out wiggle reduction)",
                    y=0.995,
                    fontsize=12,
                )
                page_fig.subplots_adjust(
                    left=0.07, right=0.98, bottom=0.07, top=0.88,
                    wspace=0.24, hspace=0.42,
                )
                pdf.savefig(page_fig, dpi=100, bbox_inches="tight")
                if show:
                    plt.show()
                plt.close(page_fig)

    return {
        "overview_pdf": overview_pdf,
        "change_points_csv": change_path,
        "transitions_csv": transitions_path,
        "rankings_csv": rankings_path,
        "atlas_pdf": atlas_path,
    }


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
    use_alpha_lambda_term=None, use_eta_sigma_term=None,
    use_f_agn_psf_2500_sigmoid_term=None,
    use_f_agn_psf_2500_flux_fraction_term=None,
    use_redshift_log_f_term=None,
    dmi_values=None,
    dmi_selection_sigma=None,
    dmi_selection_sigma_interp=None,
    clipped_mask=None,
    sigma_sel_floor_mag=0.05,
    *,
    agn_pivot_context: AgnPivotContext,
):
    d = df_agn.copy()
    clipped_mask = _resolve_clipped_mask(d, clipped_mask)
    out_of_range_color = "#354B5B"
    out_of_range_marker_color = mpl.colors.to_rgba(out_of_range_color, alpha=0.4)
    out_of_range_error_color = mpl.colors.to_rgba(out_of_range_color, alpha=0.1)
    out_of_range_residual_error_color = mpl.colors.to_rgba(out_of_range_color, alpha=0.18)

    # --- Thinning for speed ---
    n_samples = int(flat_samples.shape[0])
    thin_factor = max(1, n_samples // 500)
    flat_samples = flat_samples[::thin_factor]

    # --- Indices & parameter names ---
    option_flags = resolve_model_option_flags(
        cosmo_model,
        np.asarray(flat_samples).shape[1],
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_f_agn_psf_2500_sigmoid_term=use_f_agn_psf_2500_sigmoid_term,
        use_f_agn_psf_2500_flux_fraction_term=use_f_agn_psf_2500_flux_fraction_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    priors, model_labels, model_labels_latex = get_model_params(
        cosmo_model,
        only_agn=option_flags["only_agn"],
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
        use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
    )
    param_indices = {name: model_labels.index(name) for name in model_labels}

    # --- Pack obs/errs/pivots once (MAIN sample) ---
    agn_obs_arr, agn_err_arr, agn_pivot_arr = agn_model_pack_obs(
        d,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
        pivot_context=agn_pivot_context,
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
    med_arr = agn_model_pack_params(
        med_params,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
    )
    M0_med = med_arr[agn_model_pidx["M0_agn"]]
    logL0_med = convert_M2500_to_logL2500(M0_med)
    L2500_offset = 10.0 ** logL0_med
    x_log_ref = -0.4 * (
        M_model_agn(
            med_arr,
            agn_obs_arr,
            agn_pivot_arr,
            use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
            use_eta_sigma_term=option_flags["use_eta_sigma_term"],
            use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
        ) - M0_med
    )
    x_ref = 10.0 ** x_log_ref

    # x errors for MAIN at median params
    pred_M_err_med = M_model_agn_err(
        med_arr,
        agn_obs_arr,
        agn_err_arr,
        agn_pivot_arr,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
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
                ds[COMPLETENESS_MAG_COL].values,
                f_host_2500_psf=ds.get(COMPLETENESS_FHOST_COL),
                alpha_lambda=ds.get("alpha_lambda"),
            )
        actual_logL2500_show = convert_M2500_to_logL2500(M2500_show)
        y_log_meas_err_show = 0.4 * np.asarray(ds['apparent_mag_2500_err'].fillna(0.0), dtype=float)
        yerr_linear_show = (10.0**actual_logL2500_show) * np.log(10.0) * y_log_meas_err_show

        # x for SHOW at median params, using the AGN fit pivots.
        obs_show, err_show, _ = agn_model_pack_obs(
            ds,
            use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
            use_eta_sigma_term=option_flags["use_eta_sigma_term"],
            use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
            pivot_context=agn_pivot_context,
        )
        x_log_ref_show = -0.4 * (
            M_model_agn(
                med_arr,
                obs_show,
                agn_pivot_arr,
                use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
                use_eta_sigma_term=option_flags["use_eta_sigma_term"],
                use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
            ) - M0_med
        )
        x_show = 10.0 ** x_log_ref_show

        pred_M_err_show = M_model_agn_err(
            med_arr,
            obs_show,
            err_show,
            agn_pivot_arr,
            use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
            use_eta_sigma_term=option_flags["use_eta_sigma_term"],
            use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
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

    model_band_mask = (
        np.isfinite(x_log_ref)
        & np.isfinite(np.asarray(d["z"].values, dtype=float))
        & (np.asarray(d["z"].values, dtype=float) >= z_range[0])
        & (np.asarray(d["z"].values, dtype=float) <= z_range[1])
    )
    if np.count_nonzero(model_band_mask) < 2:
        model_band_mask = np.isfinite(x_log_ref)

    ylog_grid_by_sample = []
    for sample in flat_samples:
        sample_params = {k: sample[param_indices[k]] for k in model_labels}
        sample_arr = agn_model_pack_params(
            sample_params,
            use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
            use_eta_sigma_term=option_flags["use_eta_sigma_term"],
            use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
        )
        sample_M0 = sample_arr[agn_model_pidx["M0_agn"]]
        sample_x_log = -0.4 * (
            M_model_agn(
                sample_arr,
                agn_obs_arr,
                agn_pivot_arr,
                use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
                use_eta_sigma_term=option_flags["use_eta_sigma_term"],
                use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
            )
            - sample_M0
        )
        sample_ylog = sample_x_log + convert_M2500_to_logL2500(sample_M0)
        fit_mask = model_band_mask & np.isfinite(sample_ylog)
        if np.count_nonzero(fit_mask) >= 2 and np.nanmax(x_log_ref[fit_mask]) > np.nanmin(x_log_ref[fit_mask]):
            slope, intercept = np.polyfit(x_log_ref[fit_mask], sample_ylog[fit_mask], 1)
            ylog_grid_by_sample.append(slope * x_log_grid + intercept)
        else:
            ylog_grid_by_sample.append(x_log_grid + convert_M2500_to_logL2500(sample_M0))
    ylog_grid_by_sample = np.asarray(ylog_grid_by_sample, dtype=float)
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
        fig, (ax, ax_res) = plt.subplots(
            2,
            1,
            figsize=(8, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax_res = None

    # --- Baseline data (MAIN) ---
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
    point_scatter_logL = _population_scatter_offsets(
        sigma_int_log,
        enabled=debias,
        seed=1743,
    )
    actual_logL2500_plot = actual_logL2500 + point_scatter_logL
    residuals_plot = residuals + point_scatter_logL
    y_log_display_err = np.asarray(y_log_meas_err, dtype=float)
    yerr_linear_display = (10.0**actual_logL2500_plot) * np.log(10.0) * y_log_display_err
    mask_in  = d["z"].between(z_range[0], z_range[1])
    mask_out = ~mask_in
    clipped_in = clipped_mask & mask_in.to_numpy(dtype=bool) if clipped_mask is not None else None
    clipped_out = clipped_mask & mask_out.to_numpy(dtype=bool) if clipped_mask is not None else None

    # inside redshift range: filled markers
    ax.errorbar(
        x_ref[mask_in], 10**actual_logL2500_plot[mask_in], xerr=xerr_asym[:, mask_in], yerr=yerr_linear_display[mask_in],
        fmt='o', linestyle='none', markersize=4, mfc=(0,0,0,0.4), mec="none",
        #markeredgewidth=0,
        ecolor=(0.2, 0.2, 0.2, 0.1), elinewidth=0.8, capsize=2, capthick=0.8,
        zorder=1, label="AGN"
    )
    # outside redshift range: filled diamonds
    ax.errorbar(
        x_ref[mask_out], 10**actual_logL2500_plot[mask_out], xerr=xerr_asym[:, mask_out], yerr=yerr_linear_display[mask_out],
        fmt='D', linestyle='none', markersize=3, mfc=out_of_range_marker_color, mec="none",
        ecolor=out_of_range_error_color, elinewidth=0.8, capsize=2, capthick=0.8,
        zorder=1
    )
    if clipped_in is not None and np.any(clipped_in):
        ax.scatter(
            x_ref[clipped_in],
            10**actual_logL2500_plot[clipped_in],
            s=26,
            marker="o",
            c="tab:green",
            alpha=0.95,
            linewidths=0,
            zorder=2,
            label="Clipped AGN",
        )
    if clipped_out is not None and np.any(clipped_out):
        ax.scatter(
            x_ref[clipped_out],
            10**actual_logL2500_plot[clipped_out],
            s=28,
            marker="D",
            c="tab:green",
            alpha=0.95,
            linewidths=0,
            zorder=2,
        )

    # --- 68% / 95% KDE contours (outlines only) ---
    try:
        finite  = np.isfinite(x_log_ref) & np.isfinite(actual_logL2500_plot)
        in_use  = finite & mask_in.values
        xlog    = x_log_ref[in_use]
        ylog    = actual_logL2500_plot[in_use]

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
            if show_residuals:
                contour_color = "red"
                contour_linewidths = (2.6, 3.2)
                contour_handles = [
                    Line2D([0],[0], color='red', lw=2.6, ls='-', label='95% contour'),
                    Line2D([0],[0], color='red', lw=3.2, ls='-',  label='68% contour'),
                ]
            else:
                contour_color = "darkgray"
                contour_linewidths = (1.6, 2.0)
                contour_handles = [
                    Line2D([0],[0], color='k', lw=1.2, ls='--', label='95% contour'),
                    Line2D([0],[0], color='k', lw=1.8, ls='-',  label='68% contour'),
                ]

            CS = ax.contour(10.0**Xg, 10.0**Yg, Z,
                            levels=levels,
                            colors=contour_color,
                            alpha=1.0,
                            linestyles=('solid', 'solid'),   # 95% dashed, 68% solid
                            linewidths=contour_linewidths,
                            zorder=4)

            _extra_contour_handles = contour_handles
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
                ds[COMPLETENESS_MAG_COL].values,
                f_host_2500_psf=ds.get(COMPLETENESS_FHOST_COL),
                alpha_lambda=ds.get("alpha_lambda"),
            )
        actual_logL2500_show = convert_M2500_to_logL2500(M2500_show)
        y_log_meas_err_show = 0.4 * np.asarray(ds['apparent_mag_2500_err'].fillna(0.0), dtype=float)
        yerr_linear_show = (10.0**actual_logL2500_show) * np.log(10.0) * y_log_meas_err_show

        # x for SHOW at median params, using the same AGN-sample pivots as the fit.
        obs_show, err_show, _ = agn_model_pack_obs(
            ds,
            use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
            use_eta_sigma_term=option_flags["use_eta_sigma_term"],
            use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
            pivot_context=agn_pivot_context,
        )
        x_log_ref_show = -0.4 * (
            M_model_agn(
                med_arr,
                obs_show,
                agn_pivot_arr,
                use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
                use_eta_sigma_term=option_flags["use_eta_sigma_term"],
                use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
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
            use_eta_sigma_term=option_flags["use_eta_sigma_term"],
            use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
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
    ax.set_ylabel(r'$L_{2500\,\mathrm{\AA}}$ (erg s$^{-1}$)', fontsize=14)
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
        df_agn,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
        pivot_context=agn_pivot_context,
    )
    log_sigma_uv_pivot  = pivots_arr[agn_model_oidx["log_sigma_uv"]]
    log_tau_uv_rf_pivot = pivots_arr[agn_model_oidx["log_tau_uv_rf"]]
    sigma_uv_pivot  = 10.0 ** log_sigma_uv_pivot
    tau_uv_rf_pivot = 10.0 ** log_tau_uv_rf_pivot
    offset_mantissa, offset_exponent = f"{L2500_offset:.2e}".split("e")
    offset_exponent = int(offset_exponent)
    xlabel = (
        rf"$({offset_mantissa}\times10^{{{offset_exponent}}}\,\mathrm{{erg\,s^{{-1}}}})\,"
        rf"({{\sigma}}_\mathrm{{uv}} \, / \, {sigma_uv_pivot:.1f}\,\mathrm{{mag}})^{{{alpha_agn_L:.2f}}}"
        rf"({{\tau}}_\mathrm{{uv,rf}} \, / \, {tau_uv_rf_pivot:.0f}\,\mathrm{{days}})^{{{beta_agn_L:.2f}}}$"
    )
    if not show_residuals:
        ax.set_xlabel(xlabel, fontsize=14)
    ax.legend(loc='upper left')

    # --- Residuals panel (MAIN) ---
    sigma_meas = np.asarray(y_log_meas_err, dtype=float)
    slope_grid = np.gradient(ylog_med, x_log_grid)
    f_slope = interp1d(x_log_grid, slope_grid, bounds_error=False, fill_value='extrapolate')
    slope_at_data = f_slope(x_log_ref)
    sigma_x = np.asarray(x_log_err_med, dtype=float)
    sigma_xy = np.abs(slope_at_data) * np.abs(sigma_x)
    sigma_chi_plot = np.sqrt(sigma_meas**2 + sigma_xy**2)
    sigma_chi_full = np.sqrt(sigma_meas**2 + sigma_xy**2 + sigma_int_log**2)
    if debias:
        sigma_sel_mag = _resolve_selection_sigma_values(
            d,
            dmi_selection_sigma=dmi_selection_sigma,
            dmi_selection_sigma_interp=dmi_selection_sigma_interp,
            sigma_sel_floor_mag=sigma_sel_floor_mag,
        )
        if sigma_sel_mag is not None:
            sigma_sel_valid = np.isfinite(sigma_sel_mag) & (sigma_sel_mag > 0.0)
            sigma_sel_log = np.full_like(sigma_sel_mag, np.nan, dtype=float)
            sigma_sel_log[sigma_sel_valid] = sigma_sel_mag[sigma_sel_valid] / 2.5
            sigma_chi_full = np.where(sigma_sel_valid, sigma_sel_log, sigma_chi_full)
    good_plot = (
        np.isfinite(residuals_plot)
        & np.isfinite(sigma_chi_plot)
        & (sigma_chi_plot > 0)
    )
    good = (
        np.isfinite(residuals)
        & np.isfinite(sigma_chi_full)
        & (sigma_chi_full > 0)
    )

    if show_residuals and ax_res is not None:
        fit_membership = (
            d["is_fit_selection"].to_numpy(dtype=bool)
            if "is_fit_selection" in d.columns
            else d["z"].between(z_range[0], z_range[1]).to_numpy(dtype=bool)
        )
        good_in = good & fit_membership
        good_in_plot = good_plot & d["z"].between(z_range[0], z_range[1]).to_numpy(dtype=bool)
        good_out_plot = good_plot & ~d["z"].between(z_range[0], z_range[1]).to_numpy(dtype=bool)
        if np.any(good_in_plot):
            ax_res.errorbar(
                x_ref[good_in_plot],
                residuals_plot[good_in_plot],
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
        if np.any(good_out_plot):
            ax_res.errorbar(
                x_ref[good_out_plot],
                residuals_plot[good_out_plot],
                yerr=sigma_chi_plot[good_out_plot],
                fmt='D',
                linestyle='none',
                markersize=2.8,
                mfc=out_of_range_marker_color,
                mec="none",
                ecolor=out_of_range_residual_error_color,
                elinewidth=0.6,
                capsize=0,
                zorder=6,
                label="outside z range",
            )
        if clipped_mask is not None:
            clipped_good_in = good_in_plot & clipped_mask
            clipped_good_out = good_out_plot & clipped_mask
            if np.any(clipped_good_in):
                ax_res.scatter(
                    x_ref[clipped_good_in],
                    residuals_plot[clipped_good_in],
                    s=24,
                    marker="o",
                    c="tab:green",
                    alpha=0.95,
                    linewidths=0,
                    zorder=7,
                    label="Clipped AGN",
                )
            if np.any(clipped_good_out):
                ax_res.scatter(
                    x_ref[clipped_good_out],
                    residuals_plot[clipped_good_out],
                    s=26,
                    marker="D",
                    c="tab:green",
                    alpha=0.95,
                    linewidths=0,
                    zorder=7,
                )

        ax_res.axhline(0, color='m', linestyle='--', zorder=3)
        ax_res.set_ylabel(_residual_axis_label("L2500_sigma_tau_residuals"))
        ax_res.set_xlabel(xlabel)
        ax_res.set_xscale('log')
        ax_res.set_ylim(-1.0, 1.0)
        rms_scatter_in = np.nan
        if np.any(good_in):
            rms_scatter_in = float(np.sqrt(np.nanmean(np.square(residuals[good_in]))))
        if np.isfinite(rms_scatter_in):
            ax_res.text(
                0.98,
                0.95,
                rf"$1\sigma\ \mathrm{{RMS}}={rms_scatter_in:.2f}$",
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


def plot_L2500_vs_sigma_tau_separate(
    flat_samples,
    df_agn,
    cosmo_model,
    z_pivot_agn,
    plot_path="plots/hubble",
    show=False,
    debias=True,
    dm_interp=None,
    dmi_values=None,
    dmi_selection_sigma=None,
    dmi_selection_sigma_interp=None,
    clipped_mask=None,
    show_residuals=True,
    z_range=(0.44, 3.16),
    use_alpha_lambda_term=None,
    use_eta_sigma_term=None,
    use_f_agn_psf_2500_sigmoid_term=None,
    use_f_agn_psf_2500_flux_fraction_term=None,
    use_redshift_log_f_term=None,
    sigma_sel_floor_mag=0.05,
    *,
    agn_pivot_context: AgnPivotContext,
):
    """Plot debiased L_2500 against sigma_UV and tau_UV,RF in separate panels."""

    d = df_agn.copy()
    clipped_mask = _resolve_clipped_mask(d, clipped_mask)

    n_samples = int(flat_samples.shape[0])
    thin_factor = max(1, n_samples // 500)
    flat_samples = flat_samples[::thin_factor]

    option_flags = resolve_model_option_flags(
        cosmo_model,
        np.asarray(flat_samples).shape[1],
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_f_agn_psf_2500_sigmoid_term=use_f_agn_psf_2500_sigmoid_term,
        use_f_agn_psf_2500_flux_fraction_term=use_f_agn_psf_2500_flux_fraction_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    _priors, model_labels, _model_labels_latex = get_model_params(
        cosmo_model,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
        use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
    )
    param_indices = {name: model_labels.index(name) for name in model_labels}
    med_params = {k: np.median(flat_samples[:, param_indices[k]]) for k in model_labels}

    if cosmo_model == "FlatwCDM":
        cosmo = FlatwCDM(H0=med_params["H0"], Om0=med_params["Om0"], w0=med_params["w0"])
    elif cosmo_model == "FlatwpwaCDM":
        cosmo = FlatwpwaCDM(
            H0=med_params["H0"],
            Om0=med_params["Om0"],
            wp=med_params["wp"],
            wa=med_params["wa"],
            zp=z_pivot_agn,
        )
    elif cosmo_model == "Flatw0waCDM":
        cosmo = Flatw0waCDM(H0=med_params["H0"], Om0=med_params["Om0"], w0=med_params["w0"], wa=med_params["wa"])
    elif cosmo_model == "FlatLambdaCDM":
        cosmo = FlatLambdaCDM(H0=med_params["H0"], Om0=med_params["Om0"])
    else:
        raise ValueError(f"Unknown cosmological model: {cosmo_model}")

    if debias:
        actual_M2500 = (
            d["apparent_mag_2500"]
            - _resolve_debias_values(d, dm_interp=dm_interp, dmi_values=dmi_values)
        ) - cosmo.distmod(d["z"]).value
    else:
        actual_M2500 = d["apparent_mag_2500"] - cosmo.distmod(d["z"]).value
    actual_logL2500 = convert_M2500_to_logL2500(actual_M2500)
    y_log_meas_err = 0.4 * np.asarray(d["apparent_mag_2500_err"].fillna(0.0), dtype=float)

    agn_obs_arr, agn_err_arr, agn_pivot_arr = agn_model_pack_obs(
        d,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
        pivot_context=agn_pivot_context,
    )
    med_arr = agn_model_pack_params(
        med_params,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
    )
    M0_med = med_arr[agn_model_pidx["M0_agn"]]
    logL0_med = convert_M2500_to_logL2500(M0_med)
    full_x_log_ref = -0.4 * (
        M_model_agn(
            med_arr,
            agn_obs_arr,
            agn_pivot_arr,
            use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
            use_eta_sigma_term=option_flags["use_eta_sigma_term"],
            use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
        )
        - M0_med
    )
    model_logL_at_data = full_x_log_ref + logL0_med
    residuals = actual_logL2500 - model_logL_at_data

    pred_M_err_med = M_model_agn_err(
        med_arr,
        agn_obs_arr,
        agn_err_arr,
        agn_pivot_arr,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_f_agn_psf_2500_sigmoid_term=option_flags["use_f_agn_psf_2500_sigmoid_term"],
        use_f_agn_psf_2500_flux_fraction_term=option_flags["use_f_agn_psf_2500_flux_fraction_term"],
    )
    sigma_meas = np.asarray(y_log_meas_err, dtype=float)
    sigma_model = 0.4 * np.asarray(pred_M_err_med, dtype=float)
    sigma_chi_plot = np.sqrt(sigma_meas**2 + sigma_model**2)

    sigma_int_log = np.exp(
        evaluate_log_f(
            med_params,
            np.asarray(d["z"].values, dtype=float),
            z_pivot=z_pivot_agn,
            use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
        )
    ) / 2.5
    sigma_chi_full = np.sqrt(sigma_chi_plot**2 + sigma_int_log**2)
    if debias:
        sigma_sel_mag = _resolve_selection_sigma_values(
            d,
            dmi_selection_sigma=dmi_selection_sigma,
            dmi_selection_sigma_interp=dmi_selection_sigma_interp,
            sigma_sel_floor_mag=sigma_sel_floor_mag,
        )
        if sigma_sel_mag is not None:
            sigma_sel_valid = np.isfinite(sigma_sel_mag) & (sigma_sel_mag > 0.0)
            sigma_sel_log = np.full_like(sigma_sel_mag, np.nan, dtype=float)
            sigma_sel_log[sigma_sel_valid] = sigma_sel_mag[sigma_sel_valid] / 2.5
            sigma_chi_full = np.where(sigma_sel_valid, sigma_sel_log, sigma_chi_full)

    point_scatter_logL = _population_scatter_offsets(sigma_int_log, enabled=debias, seed=1847)
    actual_logL2500_plot = actual_logL2500 + point_scatter_logL
    residuals_plot = residuals + point_scatter_logL
    yerr_linear_display = (10.0**actual_logL2500_plot) * np.log(10.0) * y_log_meas_err

    log_sigma = pd.to_numeric(d["log_sigma_uv"], errors="coerce").to_numpy(dtype=float)
    log_tau = pd.to_numeric(d["log_tau_uv_rf"], errors="coerce").to_numpy(dtype=float)
    log_sigma_err = (
        pd.to_numeric(d["log_sigma_uv_std_psd"], errors="coerce").to_numpy(dtype=float)
        if "log_sigma_uv_std_psd" in d.columns
        else pd.to_numeric(d.get("log_sigma_uv_err", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(dtype=float)
    )
    log_tau_err = (
        pd.to_numeric(d["log_tau_uv_rf_std_psd"], errors="coerce").to_numpy(dtype=float)
        if "log_tau_uv_rf_std_psd" in d.columns
        else pd.to_numeric(d.get("log_tau_uv_rf_err", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(dtype=float)
    )

    log_sigma_pivot = agn_pivot_arr[agn_model_oidx["log_sigma_uv"]]
    log_tau_pivot = agn_pivot_arr[agn_model_oidx["log_tau_uv_rf"]]
    alpha_L_med = -0.4 * med_params["alpha_agn"]
    beta_L_med = -0.4 * med_params["beta_agn"]

    def _grid_for(values):
        finite = values[np.isfinite(values)]
        if finite.size < 2:
            return np.linspace(-1.0, 1.0, 250)
        lo, hi = np.nanpercentile(finite, [1, 99])
        pad = 0.12 * max(hi - lo, 0.1)
        return np.linspace(lo - pad, hi + pad, 250)

    grids = {
        "sigma": _grid_for(log_sigma),
        "tau": _grid_for(log_tau),
    }

    def _posterior_band(which, grid):
        curves = []
        for sample in flat_samples:
            sample_params = {k: sample[param_indices[k]] for k in model_labels}
            sample_M0 = sample_params["M0_agn"]
            sample_logL0 = convert_M2500_to_logL2500(sample_M0)
            if which == "sigma":
                slope = -0.4 * sample_params["alpha_agn"]
                pivot = log_sigma_pivot
            else:
                slope = -0.4 * sample_params["beta_agn"]
                pivot = log_tau_pivot
            curves.append(sample_logL0 + slope * (grid - pivot))
        curves = np.asarray(curves, dtype=float)
        return (
            np.nanmedian(curves, axis=0),
            np.nanpercentile(curves, 16, axis=0),
            np.nanpercentile(curves, 84, axis=0),
        )

    bands = {
        "sigma": _posterior_band("sigma", grids["sigma"]),
        "tau": _posterior_band("tau", grids["tau"]),
    }

    if show_residuals:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12.5, 8.0),
            sharex="col",
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05, "wspace": 0.22},
        )
        top_axes = axes[0]
        residual_axes = axes[1]
    else:
        fig, top_axes = plt.subplots(1, 2, figsize=(12.5, 5.8), sharey=True)
        residual_axes = [None, None]
    specs = [
        {
            "key": "sigma",
            "xlog": log_sigma,
            "xlog_err": log_sigma_err,
            "xlabel": r"$\sigma_{\rm UV}$ (mag)",
            "model_label": rf"$L_{{2500}} \propto \sigma_{{\rm UV}}^{{{alpha_L_med:.2f}}}$",
        },
        {
            "key": "tau",
            "xlog": log_tau,
            "xlog_err": log_tau_err,
            "xlabel": r"$\tau_{\rm UV,RF}$ (days)",
            "model_label": rf"$L_{{2500}} \propto \tau_{{\rm UV,RF}}^{{{beta_L_med:.2f}}}$",
        },
    ]
    mask_in = d["z"].between(z_range[0], z_range[1]).to_numpy(dtype=bool)
    mask_out = ~mask_in
    good_residual = np.isfinite(residuals_plot) & np.isfinite(sigma_chi_plot) & (sigma_chi_plot > 0)
    good_in = np.isfinite(residuals) & np.isfinite(sigma_chi_full) & (sigma_chi_full > 0) & mask_in

    for col, spec in enumerate(specs):
        ax = top_axes[col]
        ax_res = residual_axes[col]
        x = np.power(10.0, spec["xlog"])
        x_lo = np.power(10.0, spec["xlog"] - spec["xlog_err"])
        x_hi = np.power(10.0, spec["xlog"] + spec["xlog_err"])
        xerr = np.vstack((x - np.maximum(x_lo, 1e-300), np.maximum(x_hi, x) - x))
        finite_x = np.isfinite(x) & (x > 0.0)
        for select, marker, size, label in (
            (mask_in, "o", 4, "AGN"),
            (mask_out, "D", 3, "outside z range"),
        ):
            mask = finite_x & select & np.isfinite(actual_logL2500_plot)
            if np.any(mask):
                ax.errorbar(
                    x[mask],
                    10.0**actual_logL2500_plot[mask],
                    xerr=xerr[:, mask],
                    yerr=yerr_linear_display[mask],
                    fmt=marker,
                    linestyle="none",
                    markersize=size,
                    mfc=(0, 0, 0, 0.4),
                    mec="none",
                    ecolor=(0.2, 0.2, 0.2, 0.12),
                    elinewidth=0.8,
                    capsize=2,
                    capthick=0.8,
                    zorder=2,
                    label=label if col == 0 else None,
                )
        if clipped_mask is not None:
            clipped = finite_x & clipped_mask & np.isfinite(actual_logL2500_plot)
            if np.any(clipped):
                ax.scatter(
                    x[clipped],
                    10.0**actual_logL2500_plot[clipped],
                    s=28,
                    marker="D",
                    c="tab:green",
                    alpha=0.95,
                    linewidths=0,
                    zorder=4,
                    label="Clipped AGN" if col == 0 else None,
                )

        grid = grids[spec["key"]]
        y_med, y_lo, y_hi = bands[spec["key"]]
        x_grid = np.power(10.0, grid)
        ax.fill_between(x_grid, 10.0**y_lo, 10.0**y_hi, color="m", alpha=0.35, zorder=8)
        ax.plot(x_grid, 10.0**y_med, color="m", lw=2.0, zorder=9, label=spec["model_label"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        if not show_residuals:
            y_limit_values = [
                np.asarray(10.0**actual_logL2500_plot[finite_x & np.isfinite(actual_logL2500_plot)], dtype=float),
                np.asarray(10.0**y_lo, dtype=float),
                np.asarray(10.0**y_hi, dtype=float),
            ]
            y_limit_values = np.concatenate([arr[np.isfinite(arr) & (arr > 0.0)] for arr in y_limit_values])
            if y_limit_values.size:
                y_lo_lim, y_hi_lim = np.nanpercentile(y_limit_values, [0.5, 99.5])
                if np.isfinite(y_lo_lim) and np.isfinite(y_hi_lim) and y_hi_lim > y_lo_lim:
                    pad_dex = 0.08 * (np.log10(y_hi_lim) - np.log10(y_lo_lim))
                    ax.set_ylim(
                        10.0 ** (np.log10(y_lo_lim) - pad_dex),
                        10.0 ** (np.log10(y_hi_lim) + pad_dex),
                    )
        ax.set_ylabel(r"$L_{2500\,\mathrm{\AA}}$ (erg s$^{-1}$)" if col == 0 else "")
        if not show_residuals:
            ax.set_xlabel(spec["xlabel"])
        ax.legend(loc="upper left", frameon=True, fontsize=10)

        if not show_residuals:
            continue

        res_in = finite_x & good_residual & mask_in
        res_out = finite_x & good_residual & mask_out
        if np.any(res_in):
            ax_res.errorbar(
                x[res_in],
                residuals_plot[res_in],
                yerr=sigma_chi_plot[res_in],
                fmt="o",
                linestyle="none",
                markersize=2.8,
                mfc=(0, 0, 0, 0.4),
                mec="none",
                ecolor=(0.2, 0.2, 0.2, 0.18),
                elinewidth=0.6,
                capsize=0,
                zorder=5,
            )
        if np.any(res_out):
            ax_res.errorbar(
                x[res_out],
                residuals_plot[res_out],
                yerr=sigma_chi_plot[res_out],
                fmt="D",
                linestyle="none",
                markersize=2.8,
                mfc=(0, 0, 0, 0.4),
                mec="none",
                ecolor=(0.2, 0.2, 0.2, 0.18),
                elinewidth=0.6,
                capsize=0,
                zorder=6,
            )
        ax_res.axhline(0.0, color="m", linestyle="--", zorder=3)
        ax_res.set_xscale("log")
        ax_res.set_ylim(-1.0, 1.0)
        ax_res.set_xlabel(spec["xlabel"])
        ax_res.set_ylabel(_residual_axis_label("L2500_sigma_tau_residuals") if col == 0 else "")
        if np.any(good_in) and col == 1:
            rms_scatter_in = float(np.sqrt(np.nanmean(np.square(residuals[good_in]))))
            ax_res.text(
                0.98,
                0.95,
                rf"$1\sigma\ \mathrm{{RMS}}={rms_scatter_in:.2f}$",
                color="red",
                ha="right",
                va="top",
                transform=ax_res.transAxes,
            )

    fig.tight_layout()
    os.makedirs(plot_path, exist_ok=True)
    out_pdf = "L2500_vs_sigma_tau_separate"
    if debias:
        out_pdf += "_debiased"
    if show_residuals:
        out_pdf += "_with_residuals"
    out_pdf += ".pdf"
    _save_figure(fig, os.path.join(plot_path, out_pdf), dpi=600, show=show)
    return residuals, sigma_chi_full


def plot_catalog_quantity_vs_sigma_tau_separate(
    df_agn,
    *,
    y_col,
    yerr_col=None,
    y_label,
    filename,
    plot_path="plots/hubble",
    show=False,
    clipped_mask=None,
    z_range=(0.44, 3.16),
):
    """Plot an external catalog quantity against sigma_UV and tau_UV,RF."""

    d = df_agn.copy()
    if (
        y_col == "LOGLEDD_RATIO"
        and y_col not in d.columns
        and {"LOGLBOL_CORRECTED", "LOGMBH"}.issubset(d.columns)
    ):
        d[y_col] = (
            pd.to_numeric(d["LOGLBOL_CORRECTED"], errors="coerce")
            - pd.to_numeric(d["LOGMBH"], errors="coerce")
            - np.log10(1.26e38)
        )
        if yerr_col is not None and yerr_col not in d.columns:
            lbol_err = pd.to_numeric(d.get("LOGLBOL_CORRECTED_ERR", pd.Series(np.nan, index=d.index)), errors="coerce")
            mbh_err = pd.to_numeric(d.get("LOGMBH_ERR", pd.Series(np.nan, index=d.index)), errors="coerce")
            d[yerr_col] = np.sqrt(np.square(lbol_err) + np.square(mbh_err))

    required = {"log_sigma_uv", "log_tau_uv_rf", "z", y_col}
    if not required.issubset(d.columns):
        missing = ", ".join(sorted(required - set(d.columns)))
        print(f"[WARNING] Skipping {filename}: missing required columns: {missing}")
        return None

    clipped_mask = _resolve_clipped_mask(d, clipped_mask)
    y = pd.to_numeric(d[y_col], errors="coerce").to_numpy(dtype=float)
    yerr = (
        pd.to_numeric(d[yerr_col], errors="coerce").to_numpy(dtype=float)
        if yerr_col is not None and yerr_col in d.columns
        else np.full(len(d), np.nan, dtype=float)
    )
    log_sigma = pd.to_numeric(d["log_sigma_uv"], errors="coerce").to_numpy(dtype=float)
    log_tau = pd.to_numeric(d["log_tau_uv_rf"], errors="coerce").to_numpy(dtype=float)
    log_sigma_err = (
        pd.to_numeric(d["log_sigma_uv_std_psd"], errors="coerce").to_numpy(dtype=float)
        if "log_sigma_uv_std_psd" in d.columns
        else pd.to_numeric(d.get("log_sigma_uv_err", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(dtype=float)
    )
    log_tau_err = (
        pd.to_numeric(d["log_tau_uv_rf_std_psd"], errors="coerce").to_numpy(dtype=float)
        if "log_tau_uv_rf_std_psd" in d.columns
        else pd.to_numeric(d.get("log_tau_uv_rf_err", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(dtype=float)
    )
    z = pd.to_numeric(d["z"], errors="coerce").to_numpy(dtype=float)
    mask_in = np.isfinite(z) & (z >= z_range[0]) & (z <= z_range[1])
    mask_out = ~mask_in
    if not np.any(np.isfinite(y)):
        print(f"[WARNING] Skipping {filename}: no finite values in {y_col}.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), sharey=True)
    specs = [
        {
            "xlog": log_sigma,
            "xlog_err": log_sigma_err,
            "xlabel": r"$\sigma_{\rm UV}$ (mag)",
        },
        {
            "xlog": log_tau,
            "xlog_err": log_tau_err,
            "xlabel": r"$\tau_{\rm UV,RF}$ (days)",
        },
    ]

    def _plot_binned_band(ax, xlog, yvals):
        finite = np.isfinite(xlog) & np.isfinite(yvals)
        if np.count_nonzero(finite) < 25:
            return
        x_use = xlog[finite]
        y_use = yvals[finite]
        n_bins = min(12, max(5, int(np.sqrt(x_use.size))))
        edges = np.nanquantile(x_use, np.linspace(0.0, 1.0, n_bins + 1))
        edges = np.unique(edges)
        if edges.size < 3:
            return
        x_mid, y_med, y_lo, y_hi = [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (x_use >= lo) & (x_use <= hi)
            if np.count_nonzero(sel) < 5:
                continue
            x_mid.append(float(np.nanmedian(x_use[sel])))
            y_med.append(float(np.nanmedian(y_use[sel])))
            y_lo.append(float(np.nanpercentile(y_use[sel], 16)))
            y_hi.append(float(np.nanpercentile(y_use[sel], 84)))
        if len(x_mid) < 2:
            return
        x_plot = np.power(10.0, np.asarray(x_mid))
        ax.fill_between(x_plot, y_lo, y_hi, color="m", alpha=0.25, zorder=8)
        ax.plot(x_plot, y_med, color="m", lw=2.0, zorder=9, label="binned median")

    for col, spec in enumerate(specs):
        ax = axes[col]
        x = np.power(10.0, spec["xlog"])
        x_lo = np.power(10.0, spec["xlog"] - spec["xlog_err"])
        x_hi = np.power(10.0, spec["xlog"] + spec["xlog_err"])
        xerr = np.vstack((x - np.maximum(x_lo, 1e-300), np.maximum(x_hi, x) - x))
        finite = np.isfinite(x) & (x > 0.0) & np.isfinite(y)
        yerr_plot = np.where(np.isfinite(yerr) & (yerr >= 0.0), yerr, np.nan)
        for select, marker, size, label in (
            (mask_in, "o", 4, "AGN"),
            (mask_out, "D", 3, "outside z range"),
        ):
            mask = finite & select
            if np.any(mask):
                ax.errorbar(
                    x[mask],
                    y[mask],
                    xerr=xerr[:, mask],
                    yerr=yerr_plot[mask],
                    fmt=marker,
                    linestyle="none",
                    markersize=size,
                    mfc=(0, 0, 0, 0.4),
                    mec="none",
                    ecolor=(0.2, 0.2, 0.2, 0.12),
                    elinewidth=0.8,
                    capsize=2,
                    capthick=0.8,
                    zorder=2,
                    label=label if col == 0 else None,
                )
        if clipped_mask is not None:
            clipped = finite & clipped_mask
            if np.any(clipped):
                ax.scatter(
                    x[clipped],
                    y[clipped],
                    s=28,
                    marker="D",
                    c="tab:green",
                    alpha=0.95,
                    linewidths=0,
                    zorder=4,
                    label="Clipped AGN" if col == 0 else None,
                )

        _plot_binned_band(ax, spec["xlog"], y)
        ax.set_xscale("log")
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylabel(y_label if col == 0 else "")
        y_visible = y[finite]
        if y_visible.size:
            lo, hi = np.nanpercentile(y_visible, [0.5, 99.5])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                pad = 0.08 * (hi - lo)
                ax.set_ylim(lo - pad, hi + pad)
        ax.legend(loc="best", frameon=True, fontsize=10)

    fig.tight_layout()
    os.makedirs(plot_path, exist_ok=True)
    return _save_figure(fig, os.path.join(plot_path, filename), dpi=600, show=show)

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
                    z_pivot_agn, agn_pivot_context,
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
        z_pivot_agn=z_pivot_agn,
        agn_pivot_context=agn_pivot_context,
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
    - use the completeness model's constant redshift-edge values,
    - find the first crossing of C=0.5 and linearly interpolate.
    """
    mgrid = np.asarray(mag_centers)
    z_in  = np.asarray(z, dtype=float)
    C = completeness2d(mgrid[None, :], z_in[:, None])   # (N, G)

    m50 = np.empty(len(z_in), dtype=float)
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
                                 title_note="— highest posterior weight sample",
                                 *,
                                 z_pivot_agn,
                                 agn_pivot_context: AgnPivotContext):
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
                                 z_pivot_agn=z_pivot_agn,
                                 agn_pivot_context=agn_pivot_context,
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
    z_range=(0.44, 3.16),
    clipped_mask=None,
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
    clipped_mask = _resolve_clipped_mask(df_agn, clipped_mask)
    z_all = np.asarray(df_agn["z"], dtype=float)
    y_all = np.asarray(residuals, dtype=float)
    yerr_all = np.asarray(residuals_err, dtype=float)

    def _plot_one(xcol, xerr_col, xlabel, filename, *, marker_alpha=1.0, show_grid=True):
        x = np.asarray(df_agn.get(xcol, np.full(len(df_agn), np.nan)), dtype=float)
        xerr = np.asarray(df_agn.get(xerr_col, np.full(len(df_agn), np.nan)), dtype=float)
        z = z_all.copy()
        y = y_all.copy()
        yerr = yerr_all.copy()
        clipped_local = clipped_mask.copy() if clipped_mask is not None else None

        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & np.isfinite(z)
        if np.isfinite(xerr).any():
            m &= np.isfinite(xerr) | np.isnan(xerr)
        x, xerr, y, yerr, z = x[m], xerr[m], y[m], yerr[m], z[m]
        if clipped_local is not None:
            clipped_local = clipped_local[m]

        fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Residuals (mag)")
        ax.axhline(0.0, color="magenta", linewidth=2, zorder=0)
        ax.set_ylim(-4.6, 3.9)
        if show_grid:
            ax.grid(True, alpha=0.25)
        else:
            ax.grid(False)

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
                alpha=marker_alpha,
                zorder=2,
                label=label,
            )
        if clipped_local is not None and np.any(clipped_local):
            in_clipped = mask_in & clipped_local
            out_clipped = (~mask_in) & clipped_local
            if np.any(in_clipped):
                ax.scatter(
                    x[in_clipped],
                    y[in_clipped],
                    s=26,
                    c="tab:green",
                    linewidths=0,
                    zorder=3,
                    label="Clipped AGN",
                )
            if np.any(out_clipped):
                ax.scatter(
                    x[out_clipped],
                    y[out_clipped],
                    s=28,
                    c="tab:green",
                    marker="D",
                    edgecolors="tab:green",
                    linewidths=0.8,
                    zorder=3,
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
        show_grid=False,
    )
    alpha_path = _plot_one(
        "alphaOX",
        "alphaOX_err",
        r"$\alpha_{\mathrm{OX}}$",
        "alphaOX_residuals.pdf",
        show_grid=False,
    )
    alpha_lambda_path = _plot_one(
        "alpha_lambda",
        "alpha_lambda_err",
        r"$\alpha_{\lambda}$",
        "alpha_lambda_residuals.pdf",
        marker_alpha=0.55,
        show_grid=False,
    )
    return delta_path, alpha_path, alpha_lambda_path


def plot_debias_impact_diagnostics(
    df_agn,
    residuals_biased,
    residuals_debiased,
    *,
    plot_path="plots/hubble",
    show=False,
    nbins=10,
    min_count=6,
    clipped_mask=None,
):
    """Plot the residual change induced by debiasing against key observables."""
    clipped_mask = _resolve_clipped_mask(df_agn, clipped_mask)
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
        clipped_use = clipped_mask[m] if clipped_mask is not None else None

        ax.scatter(
            x_use,
            y_use,
            s=12,
            alpha=0.35,
            color="tab:blue",
            linewidths=0,
            rasterized=True,
        )
        if clipped_use is not None and np.any(clipped_use):
            ax.scatter(
                x_use[clipped_use],
                y_use[clipped_use],
                s=18,
                alpha=0.9,
                color="tab:green",
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
    log_safe_errors=True,
    show_cut_source_errors=False,
    cap_cut_source_errors=False,
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

    def _plot_fraction_points(
        ax,
        x,
        y,
        *,
        yerr=None,
        fmt="o",
        markersize=2.5,
        alpha=0.25,
        color="k",
        elinewidth=0.4,
        error_alpha=None,
        error_color=None,
        capsize=1.5,
        max_yerr=None,
        zorder=3,
        label=None,
        rasterized=True,
    ):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if yerr is not None:
            yerr = np.asarray(yerr, dtype=float)
            valid_err = np.isfinite(yerr) & (yerr > 0.0)
            if np.any(valid_err):
                if not np.all(valid_err):
                    ax.plot(
                        x[~valid_err],
                        y[~valid_err],
                        linestyle="None",
                        marker=fmt,
                        markersize=markersize,
                        alpha=alpha,
                        color=color,
                        zorder=zorder,
                        label=None,
                        rasterized=rasterized,
                    )
                x_err = x[valid_err]
                y_err_center = y[valid_err]
                yerr_err = yerr[valid_err]
                if max_yerr is not None and np.isfinite(max_yerr) and max_yerr > 0.0:
                    yerr_err = np.minimum(yerr_err, float(max_yerr))
                lower_err = np.minimum(yerr_err, 0.999999 * y_err_center)
                if error_color is None:
                    error_color = color
                if error_alpha is None:
                    error_alpha = alpha
                error_container = ax.errorbar(
                    x_err,
                    y_err_center,
                    yerr=np.vstack([lower_err, yerr_err]),
                    linestyle="None",
                    marker=fmt,
                    markersize=markersize,
                    alpha=alpha,
                    color=color,
                    ecolor=error_color,
                    elinewidth=elinewidth,
                    capsize=capsize,
                    zorder=zorder,
                    label=label,
                    rasterized=rasterized,
                )
                _, caplines, barlinecols = error_container
                for bar_collection in barlinecols:
                    bar_collection.set_alpha(error_alpha)
                for capline in caplines:
                    capline.set_alpha(error_alpha)
                return
        ax.plot(
            x,
            y,
            linestyle="None",
            marker=fmt,
            markersize=markersize,
            alpha=alpha,
            color=color,
            zorder=zorder,
            label=label,
            rasterized=rasterized,
        )

    def _typical_errors(y, yerr):
        if yerr is None:
            return None
        y = np.asarray(y, dtype=float)
        yerr = np.asarray(yerr, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac_err = yerr / y
        valid = np.isfinite(y) & (y > 0.0) & np.isfinite(yerr) & (yerr > 0.0) & np.isfinite(frac_err)
        if not np.any(valid):
            return None
        return float(np.nanmedian(frac_err[valid])), float(np.nanmedian(yerr[valid]))

    def _set_panel_redshift_xlim(ax, redshift_values):
        redshift_values = np.asarray(redshift_values, dtype=float)
        redshift_values = redshift_values[np.isfinite(redshift_values)]
        if redshift_values.size == 0:
            return
        z_lo = float(np.nanmin(redshift_values))
        z_hi = float(np.nanmax(redshift_values))
        if z_hi > z_lo:
            pad = 0.05 * (z_hi - z_lo)
        else:
            pad = 0.1
        ax.set_xlim(z_lo - pad, z_hi + pad)

    df_plot = _prepare_spectral_fraction_frame(df_agn)
    df_cut_plot = (
        _prepare_spectral_fraction_frame(df_cut_sources)
        if df_cut_sources is not None
        else None
    )
    z = pd.to_numeric(df_plot["z"], errors="coerce").to_numpy(dtype=float)
    kept_color = "tab:red"
    cut_color = "gray"
    panel_specs = [
        ("f_bc_3000", r"$f_{\rm BC}$"),
        ("f_fe_uv_3000", r"$f_{\rm FeII}$"),
    ]
    if "f_host_2500" in df_plot.columns:
        panel_specs.append(("f_host_2500", r"$f_{\rm host,2500\,\AA}$"))
    if len(panel_specs) == 2:
        return None

    fig, axes = plt.subplots(
        1,
        len(panel_specs),
        figsize=(4 * len(panel_specs), 4),
        sharex=False,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()

    def _order_legend_entries(handles, labels):
        entries_by_label = {}
        for handle, label in zip(handles, labels):
            if label and not label.startswith("_") and label not in entries_by_label:
                entries_by_label[label] = handle

        def _priority(label):
            if label.endswith("(kept)"):
                return 0
            if label.endswith("(cut)"):
                return 1
            if label == "cut threshold":
                return 2
            return 3

        ordered_labels = sorted(
            entries_by_label,
            key=lambda label: (_priority(label), list(entries_by_label).index(label)),
        )
        return [entries_by_label[label] for label in ordered_labels], ordered_labels

    for i_ax, (ax, (col, ylabel)) in enumerate(zip(axes, panel_specs)):
        y = pd.to_numeric(df_plot[col], errors="coerce").to_numpy(dtype=float)
        yerr = None
        err_col = f"{col}_err"
        if err_col in df_plot.columns:
            yerr = pd.to_numeric(df_plot[err_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(z) & np.isfinite(y) & (y > 0.0)
        if np.count_nonzero(mask) == 0:
            ax.text(0.5, 0.5, f"No finite {col}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        z_use = z[mask]
        y_use = y[mask]
        yerr_use = yerr[mask] if yerr is not None else None
        panel_redshifts = [z_use]
        kept_typical_errors = _typical_errors(y_use, yerr_use)
        kept_label = f"{ylabel} (kept)"
        cut_label = f"{ylabel} (cut)"
        component_label = kept_label
        in_z = (z_use >= z_range[0]) & (z_use <= z_range[1])
        out_z = ~in_z

        if np.any(in_z):
            _plot_fraction_points(
                ax,
                z_use[in_z],
                y_use[in_z],
                yerr=None,
                fmt="o",
                markersize=2.5,
                alpha=0.1,
                color=kept_color,
                elinewidth=0.4,
                zorder=3,
                label=component_label,
                rasterized=True,
            )
        if np.any(out_z):
            _plot_fraction_points(
                ax,
                z_use[out_z],
                y_use[out_z],
                yerr=None,
                fmt="D",
                markersize=3.2,
                alpha=0.1,
                color=kept_color,
                elinewidth=0.4,
                zorder=6,
                label=component_label if not np.any(in_z) else None,
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
            if np.any(mask_cut):
                z_cut_use = z_cut[mask_cut]
                panel_redshifts.append(z_cut_use)
                yerr_plot = (
                    yerr_cut[mask_cut]
                    if show_cut_source_errors and yerr_cut is not None
                    else None
                )
                max_cut_yerr = (
                    kept_typical_errors[1]
                    if cap_cut_source_errors and kept_typical_errors is not None
                    else None
                )
                _plot_fraction_points(
                    ax,
                    z_cut_use,
                    y_cut[mask_cut],
                    yerr=yerr_plot,
                    fmt="x",
                    markersize=4,
                    alpha=0.7,
                    color=cut_color,
                    elinewidth=0.32,
                    error_alpha=0.26,
                    error_color=cut_color,
                    capsize=0,
                    max_yerr=max_cut_yerr,
                    zorder=2,
                    label=cut_label,
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

        _set_panel_redshift_xlim(ax, np.concatenate(panel_redshifts))
        ax.set_xlabel(r"$z$")
        ax.set_ylabel("Component fraction" if i_ax == 0 else "")
        ax.set_yscale("log")
        ax.set_ylim(2e-6, 5e1)
        handles, labels = _order_legend_entries(*ax.get_legend_handles_labels())
        if handles:
            ax.legend(
                handles,
                labels,
                fontsize=12,
                loc="upper right",
                frameon=True,
                facecolor="white",
                framealpha=0.75,
                edgecolor="none",
            )

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
            "'log_sigma_uv'+'dlog_amp_bc' or per-band 'amp_bc_<band>' with 'bc_weight_<band>'."
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
            "'log_sigma_uv'+'dlog_amp_bc' or per-band 'amp_bc_<band>' with 'bc_weight_<band>'."
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


def plot_completeness_diagnostics(
    dmi_plot,
    z,
    m2500,
    integrals_max_w=None,
    plot_path="plots/hubble",
    z_range=None,
):

    # Plot dmi vs z for the posterior-summary correction used in debiasing.
    z = np.asarray(z, dtype=float)
    m2500 = np.asarray(m2500, dtype=float)
    dmi_plot = np.asarray(dmi_plot, dtype=float)
    if z.shape != dmi_plot.shape or m2500.shape != dmi_plot.shape:
        raise ValueError(
            f"Expected z, m2500, and dmi_plot to have the same shape, got "
            f"{z.shape}, {m2500.shape}, {dmi_plot.shape}."
        )
    finite = np.isfinite(z) & np.isfinite(m2500) & np.isfinite(dmi_plot)
    if z_range is None:
        fit_mask = finite
        out_mask = np.zeros_like(finite, dtype=bool)
    else:
        fit_mask = finite & (z >= z_range[0]) & (z <= z_range[1])
        out_mask = finite & ~((z >= z_range[0]) & (z <= z_range[1]))
    
    fig, ax = plt.subplots(figsize=(8, 5))

    if np.any(fit_mask):
        ax.plot(
            z[fit_mask],
            -dmi_plot[fit_mask],
            marker="o",
            linestyle="none",
            label="in $z$ range",
            color="k",
            alpha=0.5,
        )
    if np.any(out_mask):
        ax.plot(
            z[out_mask],
            -dmi_plot[out_mask],
            marker="D",
            linestyle="none",
            label="outside $z$ range",
            color="k",
            alpha=0.5,
        )

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

    if np.any(fit_mask):
        ax.scatter(
            m2500[fit_mask],
            -dmi_plot[fit_mask],
            alpha=0.5,
            s=20,
            marker="o",
            color="k",
            label="in $z$ range",
        )
    if np.any(out_mask):
        ax.scatter(
            m2500[out_mask],
            -dmi_plot[out_mask],
            alpha=0.5,
            s=28,
            marker="D",
            color="k",
            label="outside $z$ range",
        )

    ax.set_xlabel(r"Apparent magnitude $m_{2500}$ (mag)")
    ax.set_ylabel(r"$\Delta m$ (mag)")

    ax.legend(frameon=True, loc="upper right", fontsize=12)
    fig.tight_layout()

    fig.savefig(f"{outdir}/dmi_vs_m2500_posterior_median.pdf", dpi=300)
    plt.close(fig)

    if integrals_max_w is not None:
        integrals_max_w = np.asarray(integrals_max_w, dtype=float)
        if integrals_max_w.shape == z.shape:
            mask_integrals = np.isfinite(z) & np.isfinite(integrals_max_w)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(z[mask_integrals], integrals_max_w[mask_integrals], s=16, alpha=0.3)
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
                            only_agn=False,
                            show=False):
    """
    Plot redshift histograms for SN (Pantheon) and AGN samples
    using a logarithmic redshift axis.
    """

    # --- SN ---
    show_sne = (
        not only_agn
        and df_pantheon is not None
        and z_col_sn in df_pantheon.columns
        and len(df_pantheon) > 0
    )
    z_sn = df_pantheon[z_col_sn].to_numpy() if show_sne else np.empty(0, dtype=float)

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
    if show_sne:
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
        if d.empty:
            ax.set_axis_off()
            continue

        if log_color:
            c_all = np.log10(d[ccol].to_numpy(dtype=float))
        else:
            c_all = d[ccol].to_numpy(dtype=float)
        if c_all.size == 0 or not np.any(np.isfinite(c_all)):
            ax.set_axis_off()
            continue

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
