"""Centralized parameter-cut thresholds for the QVC Hubble pipeline."""

from ast import literal_eval
import os
import re

import numpy as np
import pandas as pd


def _cut_env_float(name, default):
    """Return an optional numeric cut override from the environment.

    The normal pipeline continues to use the in-source science defaults.  This
    narrow override mechanism is for reproducible cut scans: each launched
    process receives its complete cut profile, without mutating the defaults
    or another scan member's configuration.
    """

    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return default
    if str(value).strip().lower() in {"none", "null", "off"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a floating-point value or 'none', got {value!r}") from exc

LOG_TAU_UV_RF_MIN = _cut_env_float("QVC_CUT_LOG_TAU_UV_RF_MIN", 1.5)
LOG_TAU_UV_RF_MAX = _cut_env_float("QVC_CUT_LOG_TAU_UV_RF_MAX", 4.0)
FRAC_AGN_5100_MIN = None
APPARENT_MAG_2500_ERR_MAX = _cut_env_float(
    "QVC_CUT_APPARENT_MAG_2500_ERR_MAX", 1.0
)

# JAXSEDFit goodness-of-fit and posterior-convergence diagnostics.
JAXSEDFIT_JOINT_REDUCED_CHI2_MAX = _cut_env_float(
    "QVC_CUT_JAXSEDFIT_JOINT_REDUCED_CHI2_MAX", 1.5
)
A_2500_TOTAL_MAX = _cut_env_float("QVC_CUT_A_2500_TOTAL_MAX", None)
SPECTRAL_RHAT_MAX = _cut_env_float("QVC_CUT_SPECTRAL_RHAT_MAX", 1.20)
LIGHT_CURVE_RHAT_MAX = _cut_env_float(
    "QVC_CUT_LIGHT_CURVE_RHAT_MAX", 1.10
)
# ESS is not used as a hard Hubble-sample selection criterion. The spectral
# catalog persists R-hat only.

# The completeness map is tabulated at histogram-bin centers, but the selected
# sample occupies the full histogram interval.  The likelihood extends the
# first and last bin values from their centers to these hard-cut edges.
COMPLETENESS_MAG_EDGE_MIN = 18.5
COMPLETENESS_MAG_EDGE_MAX = 24.0
COMPLETENESS_N_MAG_BINS = 30
COMPLETENESS_MAG_BIN_WIDTH = (
    COMPLETENESS_MAG_EDGE_MAX - COMPLETENESS_MAG_EDGE_MIN
) / COMPLETENESS_N_MAG_BINS
_DEFAULT_COMPLETENESS_MAG_2500_MIN = COMPLETENESS_MAG_EDGE_MIN
_DEFAULT_COMPLETENESS_MAG_2500_MAX = COMPLETENESS_MAG_EDGE_MAX
COMPLETENESS_MAG_2500_MIN = _cut_env_float(
    "QVC_CUT_COMPLETENESS_MAG_2500_MIN", _DEFAULT_COMPLETENESS_MAG_2500_MIN
)
COMPLETENESS_MAG_2500_MAX = _cut_env_float(
    "QVC_CUT_COMPLETENESS_MAG_2500_MAX", _DEFAULT_COMPLETENESS_MAG_2500_MAX
)

LIGHT_CURVE_N_POINTS_COLUMN = "light_curve_n_points"
LIGHT_CURVE_N_POINTS_EXCLUDED_BANDS = ("u",)

# Historical scalar/component defaults (disabled):
# WRMS_MAX=1.2; T_RF_LENGTH_MIN=1700; LIGHT_CURVE_N_POINTS_MIN=250
# F_HOST_2500_MAX=0.1; VARIABILITY_CHI_SQ_RED_G_MIN=30; LOO_CHI2_EFF_MAX=1.01
# LOG_SIGMA_UV=(-1.5, 0.2); LOG_AMP_DELTA_BLR_UPPER=-0.2
# LOG_AMP_DELTA_BC_UPPER=-0.2; LOG_F_BC_3000_MAX=log10(0.05)
# LOG_F_FE_UV_3000_MAX=log10(0.1); REL_APPARENT_MAG_2500_ERR_MAX=0.0025
# COMPLETENESS_MAG_2500_MAX=None; ALPHA_LAMBDA=(None, None)
WRMS_MAX = None
T_RF_LENGTH_MIN = None
LIGHT_CURVE_N_POINTS_MIN = None
ALPHA_LAMBDA_MIN = None
ALPHA_LAMBDA_MAX = None

LOG_SIGMA_UV_MIN = None
LOG_SIGMA_UV_MAX = None
REDDENING_EBV_MAX = None

VARIABILITY_CHI_SQ_RED_G_MIN = None
LOO_CHI2_EFF_MAX = None
F_HOST_2500_MAX = None
LOG_AMP_DELTA_BLR_UPPER = None
LOG_AMP_DELTA_BLR_UPPER_BY_BAND = {}
LOG_AMP_DELTA_BC_UPPER = None
LOG_F_BC_3000_MAX = None
LOG_F_FE_UV_3000_MAX = None
REL_APPARENT_MAG_2500_ERR_MAX = None


EXCLUDED_SDSS_NAMES = (
    "221120.38+010905.6",
    "024555.35+005332.6",
    "015802.36+002917.3",
    "014641.18+010815.9",
    "010151.08+002028.8",
    "025646.57+003858.3",
    "215013.64-001627.2",
    "221018.27+005832.1",
    "220311.37+005056.3"

)
AGN_SCALAR_PARAMETER_CUTS = (
    ("log_tau_uv_rf", LOG_TAU_UV_RF_MIN, LOG_TAU_UV_RF_MAX),
    ("fracAGN_5100_fit", FRAC_AGN_5100_MIN, None),
    ("apparent_mag_2500_err", None, APPARENT_MAG_2500_ERR_MAX),
    (
        "m_2500_dereddened",
        COMPLETENESS_MAG_2500_MIN,
        COMPLETENESS_MAG_2500_MAX,
    ),
    ("joint_reduced_chi2", None, JAXSEDFIT_JOINT_REDUCED_CHI2_MAX),
    ("m_2500_dereddened_rhat", None, SPECTRAL_RHAT_MAX),
    ("m_2500_attenuated_model_rhat", None, SPECTRAL_RHAT_MAX),
    ("log_tau_uv_rf_rhat", None, LIGHT_CURVE_RHAT_MAX),
    ("log_sigma_uv_rhat", None, LIGHT_CURVE_RHAT_MAX),
)

# SVI fits do not define chain-based R-hat/ESS diagnostics.  Missing values in
# these columns therefore mean "not applicable" rather than a failed fit.
# Finite diagnostics are still tested against the thresholds above.
ALLOW_MISSING_SCALAR_CUT_COLUMNS = frozenset(
    {
        "m_2500_dereddened_rhat",
        "m_2500_attenuated_model_rhat",
        "log_tau_uv_rf_rhat",
        "log_sigma_uv_rhat",
    }
)


def light_curve_point_count_series(df, *, exclude_bands=None):
    """Return total per-object light-curve point counts and the columns used."""

    excluded = set() if exclude_bands is None else {str(b) for b in exclude_bands}
    count_cols_by_band = {}
    for col in df.columns:
        match = re.match(r"^(variability_n_points|number_points)_([^_]+)$", str(col))
        if match is None:
            continue
        prefix, band = match.groups()
        if band in excluded:
            continue
        count_cols_by_band.setdefault(band, {})[prefix] = col

    selected_cols = []
    for band in sorted(count_cols_by_band):
        choices = count_cols_by_band[band]
        selected_cols.append(choices.get("variability_n_points", choices.get("number_points")))

    if selected_cols:
        total = np.zeros(len(df), dtype=float)
        has_count = np.zeros(len(df), dtype=bool)
        for col in selected_cols:
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            total[finite] += values[finite]
            has_count |= finite
        total[~has_count] = np.nan
        return total, selected_cols

    if "number_points" not in df.columns:
        return None, []

    total = np.full(len(df), np.nan, dtype=float)
    fixed_bands = ("u", "g", "r", "i", "z")
    for i, value in enumerate(df["number_points"]):
        parsed = value
        if isinstance(value, str):
            try:
                parsed = literal_eval(value)
            except Exception:
                parsed = value
        if isinstance(parsed, dict):
            vals = pd.to_numeric(
                pd.Series(
                    [v for k, v in parsed.items() if str(k) not in excluded]
                ),
                errors="coerce",
            ).to_numpy(dtype=float)
            if np.any(np.isfinite(vals)):
                total[i] = np.nansum(vals)
        elif isinstance(parsed, (list, tuple, np.ndarray)):
            values = list(parsed)
            if len(values) == len(fixed_bands):
                values = [v for b, v in zip(fixed_bands, values) if b not in excluded]
            vals = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
            if np.any(np.isfinite(vals)):
                total[i] = np.nansum(vals)
        else:
            scalar = pd.to_numeric(pd.Series([parsed]), errors="coerce").to_numpy(dtype=float)[0]
            if np.isfinite(scalar):
                total[i] = scalar
    return total, ["number_points"]


def add_light_curve_point_count_column(
    df,
    *,
    column=LIGHT_CURVE_N_POINTS_COLUMN,
    exclude_bands=LIGHT_CURVE_N_POINTS_EXCLUDED_BANDS,
):
    """Add the scalar light-curve point-count column used by AGN cuts."""

    counts, count_cols = light_curve_point_count_series(df, exclude_bands=exclude_bands)
    if counts is None:
        return df, count_cols
    df = df.copy()
    df[column] = counts
    return df, count_cols
