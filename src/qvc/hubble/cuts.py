"""Centralized parameter-cut thresholds for the QVC Hubble pipeline."""

from ast import literal_eval
import re

import numpy as np
import pandas as pd

LOG_TAU_UV_RF_MIN = 1.5
LOG_TAU_UV_RF_MAX = 4.0
WRMS_MAX = 1.2
T_RF_LENGTH_MIN = 1700.0
LIGHT_CURVE_N_POINTS_MIN = 250
LIGHT_CURVE_N_POINTS_COLUMN = "light_curve_n_points"
LIGHT_CURVE_N_POINTS_EXCLUDED_BANDS = ("u",)
F_HOST_2500_MAX = None
ALPHA_LAMBDA_MIN = None
ALPHA_LAMBDA_MAX = None
VARIABILITY_CHI_SQ_RED_G_MIN = 20.0
LOG_SIGMA_UV_MIN = -1.5
LOG_SIGMA_UV_MAX = 0.2
REDDENING_EBV_MAX = None

LOG_AMP_DELTA_BLR_UPPER = 0.0
LOG_AMP_DELTA_BLR_UPPER_BY_BAND = {
    "u": LOG_AMP_DELTA_BLR_UPPER,
    "g": LOG_AMP_DELTA_BLR_UPPER,
    "r": LOG_AMP_DELTA_BLR_UPPER,
    "i": LOG_AMP_DELTA_BLR_UPPER,
}

LOG_AMP_DELTA_BC_UPPER = 0.0
LOG_F_BC_3000_MAX = -1.0
LOG_F_FE_UV_3000_MAX = -1.0
REL_APPARENT_MAG_2500_ERR_MAX = 0.003
EXCLUDED_SDSS_NAMES = (
    "221120.38+010905.6",
    "024555.35+005332.6",
    "015802.36+002917.3",
)
AGN_SCALAR_PARAMETER_CUTS = (
    ("log_tau_uv_rf", LOG_TAU_UV_RF_MIN, LOG_TAU_UV_RF_MAX),
    ("wrms", None, WRMS_MAX),
    ("t_rf_length", T_RF_LENGTH_MIN, None),
    (LIGHT_CURVE_N_POINTS_COLUMN, LIGHT_CURVE_N_POINTS_MIN, None),
    ("f_host_2500", None, F_HOST_2500_MAX),
    ("alpha_lambda", ALPHA_LAMBDA_MIN, ALPHA_LAMBDA_MAX),
    ("variability_chi_sq_red_g", VARIABILITY_CHI_SQ_RED_G_MIN, None),
    ("log_sigma_uv", LOG_SIGMA_UV_MIN, LOG_SIGMA_UV_MAX),
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
