"""Centralized parameter-cut thresholds for the QVC Hubble pipeline."""

from ast import literal_eval
import os
import re

import numpy as np
import pandas as pd


SDSS_TARGET_SELECTION_CHOICES = (
    "all",
    "legacy-sdss",
    "boss",
    "eboss",
    "eboss-color-sensitivity",
    "eboss-var-s82-inclusive",
    "eboss-non-var-s82",
    "eboss-var-s82-only",
    "eboss-var-s82-core-only",
)
SDSS_EBOSS_QSO_REASON_BITS = tuple(
    list(range(9, 20)) + list(range(22, 32)) + list(range(35, 38))
)
SDSS_EBOSS_VAR_S82_BIT = 9
SDSS_EBOSS_CORE_BIT = 10
EBOSS_COLOR_BITS = {"target0": (10,), "target1": (10,), "target2": ()}
EBOSS_ALT_CHANNEL_BITS = {
    "var_s82": {"target0": (), "target1": (9,), "target2": ()},
    "ptf": {"target0": (11, 40), "target1": (11,), "target2": ()},
    "tdss": {"target0": (), "target1": (), "target2": (20, 26)},
    "radio": {"target0": (14,), "target1": (14,), "target2": ()},
    "xray_agn": {
        "target0": (20, 22),
        "target1": (19, 28, 29),
        "target2": (0, 2, 4),
    },
}
EBOSS_DISQUALIFY_BITS = {
    "target0": (12, 13, 15, 16, 17, 18, 21, 23),
    "target1": (12, 13, 15, 16, 17, 18, 20),
    "target2": (1, 3, 5),
}
SDSS_TARGET_SELECTION_REQUIRED_COLUMNS = (
    "SDSS_EBOSS_TARGET0",
    "SDSS_EBOSS_TARGET1",
    "SDSS_EBOSS_TARGET2",
)


def normalize_sdss_target_selection(value):
    """Normalize a named SDSS targeting-sample preset."""

    normalized = str(value).strip().lower().replace("_", "-")
    if normalized not in SDSS_TARGET_SELECTION_CHOICES:
        choices = ", ".join(SDSS_TARGET_SELECTION_CHOICES)
        raise ValueError(
            f"Unknown SDSS target selection {value!r}; choose one of: {choices}."
        )
    return normalized


def _nonnegative_integer_mask_values(series):
    """Return exact integer targeting masks plus a validity mask."""

    values = np.zeros(len(series), dtype=np.uint64)
    valid = np.zeros(len(series), dtype=bool)
    for index, raw_value in enumerate(series.to_numpy(dtype=object)):
        if pd.isna(raw_value):
            continue
        try:
            if isinstance(raw_value, (float, np.floating)):
                if not np.isfinite(raw_value) or not float(raw_value).is_integer():
                    continue
            parsed = int(raw_value)
        except (TypeError, ValueError, OverflowError):
            continue
        if parsed < 0 or parsed > np.iinfo(np.uint64).max:
            continue
        values[index] = parsed
        valid[index] = True
    return values, valid


def _normalized_survey_value(value):
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8", errors="replace")
    return str(value).strip().lower()


def _matched_sdss_rows(df):
    if "SDSS_SPECOBJ_MATCHED" not in df.columns:
        return np.ones(len(df), dtype=bool)
    truthy = {"1", "true", "t", "yes", "y"}
    return np.asarray(
        [
            False if pd.isna(value) else _normalized_survey_value(value) in truthy
            for value in df["SDSS_SPECOBJ_MATCHED"].to_numpy(dtype=object)
        ],
        dtype=bool,
    )


def _has_any_bits(values, bits):
    mask = np.uint64(sum(1 << int(bit) for bit in bits))
    return (values & mask) != 0


def build_sdss_target_selection_mask(df, selection="all"):
    """Build a targeting mask and human-readable criterion for an AGN table."""

    selection = normalize_sdss_target_selection(selection)
    if selection == "all":
        return np.ones(len(df), dtype=bool), "all SDSS targeting selections"

    if "SDSS_SURVEY" not in df.columns:
        raise ValueError(
            f"SDSS target selection {selection!r} requires spectra metadata column "
            "'SDSS_SURVEY'. Use the *_sdss_metadata.h5 spectra catalog and rerun."
        )

    normalized_survey = np.asarray(
        [_normalized_survey_value(value) for value in df["SDSS_SURVEY"]],
        dtype=object,
    )
    matched = _matched_sdss_rows(df)
    if selection == "legacy-sdss":
        criterion = "Legacy SDSS survey rows (SDSS_SURVEY in sdss/segue1/segue2)"
        return matched & np.isin(normalized_survey, ("sdss", "segue1", "segue2")), criterion
    if selection == "boss":
        return matched & (normalized_survey == "boss"), "BOSS survey rows"
    if selection == "eboss":
        return matched & (normalized_survey == "eboss"), "eBOSS survey rows"
    if selection == "eboss-color-sensitivity":
        if "SDSS_PROGRAMNAME" not in df.columns:
            raise ValueError(
                "eBOSS color sensitivity requires SDSS_PROGRAMNAME provenance."
            )
        program = np.asarray(
            [_normalized_survey_value(value) for value in df["SDSS_PROGRAMNAME"]],
            dtype=object,
        )
        criterion = (
            "all matched main-eBOSS program rows; CORE and alternative-channel "
            "bits are used only to train the offline color head"
        )
        return matched & (normalized_survey == "eboss") & (program == "eboss"), criterion

    missing = [
        column for column in SDSS_TARGET_SELECTION_REQUIRED_COLUMNS
        if column not in df.columns
    ]
    if missing:
        raise ValueError(
            f"SDSS target selection {selection!r} requires spectra metadata columns "
            f"{missing}. Use the *_sdss_metadata.h5 spectra catalog (or populate "
            "SDSS_EBOSS_TARGET0/1/2) and rerun."
        )

    survey_is_eboss = normalized_survey == "eboss"
    target0, target0_valid = _nonnegative_integer_mask_values(
        df["SDSS_EBOSS_TARGET0"]
    )
    target1, target1_valid = _nonnegative_integer_mask_values(
        df["SDSS_EBOSS_TARGET1"]
    )
    target2, target2_valid = _nonnegative_integer_mask_values(
        df["SDSS_EBOSS_TARGET2"]
    )
    valid = (
        survey_is_eboss
        & target0_valid
        & target1_valid
        & target2_valid
        & matched
    )

    var_s82_mask = 1 << SDSS_EBOSS_VAR_S82_BIT
    if selection == "eboss-var-s82-inclusive":
        criterion = (
            "eBOSS rows with EBOSS_TARGET1 bit 9 (VAR_S82) set; other targeting "
            "bits are allowed"
        )
        return valid & ((target1 & np.uint64(var_s82_mask)) != 0), criterion
    if selection == "eboss-non-var-s82":
        criterion = (
            "eBOSS rows without EBOSS_TARGET1 bit 9 (VAR_S82); all other "
            "targeting reasons are allowed"
        )
        return valid & ((target1 & np.uint64(var_s82_mask)) == 0), criterion

    qso_reason_mask = sum(1 << bit for bit in SDSS_EBOSS_QSO_REASON_BITS)
    allowed_reason_mask = var_s82_mask
    if selection == "eboss-var-s82-core-only":
        allowed_reason_mask |= 1 << SDSS_EBOSS_CORE_BIT
    criterion = (
        "eBOSS rows with QSO-reason bits exactly "
        f"{sorted(bit for bit in SDSS_EBOSS_QSO_REASON_BITS if allowed_reason_mask & (1 << bit))}, "
        "EBOSS_TARGET0 == 0, and EBOSS_TARGET2 == 0"
    )
    mask = (
        valid
        & (target0 == 0)
        & (target2 == 0)
        & ((target1 & np.uint64(qso_reason_mask)) == np.uint64(allowed_reason_mask))
    )
    return mask, criterion


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
T_RF_OVER_TAU_UV_RF_COLUMN = "t_rf_over_tau_uv_rf"
T_RF_OVER_TAU_UV_RF_MIN = _cut_env_float(
    "QVC_CUT_T_RF_OVER_TAU_UV_RF_MIN", 5.0
)
FRAC_AGN_5100_MIN = None
APPARENT_MAG_2500_ERR_MAX = _cut_env_float(
    "QVC_CUT_APPARENT_MAG_2500_ERR_MAX", 1.0
)

# JAXSEDFit goodness-of-fit and posterior-convergence diagnostics.
JAXSEDFIT_JOINT_REDUCED_CHI2_MAX = _cut_env_float(
    "QVC_CUT_JAXSEDFIT_JOINT_REDUCED_CHI2_MAX", 1.5
)
SED_REDUCED_CHI2_MAX = _cut_env_float("QVC_CUT_SED_REDUCED_CHI2_MAX", 2.0)
SPECTROSCOPY_REDUCED_CHI2_MAX = _cut_env_float(
    "QVC_CUT_SPECTROSCOPY_REDUCED_CHI2_MAX", 1.3
)
LOO_CHI2_EFF_MAX = _cut_env_float("QVC_CUT_LOO_CHI2_EFF_MAX", 1.01)
A_2500_TOTAL_MAX = _cut_env_float("QVC_CUT_A_2500_TOTAL_MAX", None)
SPECTRAL_RHAT_MAX = _cut_env_float("QVC_CUT_SPECTRAL_RHAT_MAX", 1.20)
LIGHT_CURVE_RHAT_MAX = _cut_env_float(
    "QVC_CUT_LIGHT_CURVE_RHAT_MAX", 1.10
)
NUM_DIVERGENCES_MAX = _cut_env_float("QVC_CUT_NUM_DIVERGENCES_MAX", None)
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
# F_HOST_2500_MAX=0.1; VARIABILITY_CHI_SQ_RED_G_MIN=30
# LOG_SIGMA_UV=(-1.5, 0.2); LOG_AMP_DELTA_BLR_UPPER=-0.2
# LOG_AMP_DELTA_BC_UPPER=-0.2; LOG_F_BC_3000_MAX=log10(0.05)
# LOG_F_FE_UV_3000_MAX=log10(0.1); REL_APPARENT_MAG_2500_ERR_MAX=0.0025
# COMPLETENESS_MAG_2500_MAX=None; ALPHA_LAMBDA=(None, None)
WRMS_MAX = None
T_RF_LENGTH_MIN = None
LIGHT_CURVE_N_POINTS_MIN = _cut_env_float(
    "QVC_CUT_LIGHT_CURVE_N_POINTS_MIN", None
)
ALPHA_LAMBDA_MIN = None
ALPHA_LAMBDA_MAX = None

LOG_SIGMA_UV_MIN = _cut_env_float("QVC_CUT_LOG_SIGMA_UV_MIN", None)
LOG_SIGMA_UV_MAX = _cut_env_float("QVC_CUT_LOG_SIGMA_UV_MAX", None)
REDDENING_EBV_MAX = None

VARIABILITY_CHI_SQ_RED_G_MIN = _cut_env_float(
    "QVC_CUT_VARIABILITY_CHI_SQ_RED_G_MIN", None
)
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
    *(
        ((T_RF_OVER_TAU_UV_RF_COLUMN, T_RF_OVER_TAU_UV_RF_MIN, None),)
        if T_RF_OVER_TAU_UV_RF_MIN is not None
        else ()
    ),
    *(
        (("log_sigma_uv", LOG_SIGMA_UV_MIN, LOG_SIGMA_UV_MAX),)
        if LOG_SIGMA_UV_MIN is not None or LOG_SIGMA_UV_MAX is not None
        else ()
    ),
    *(
        (("variability_chi_sq_red_g", VARIABILITY_CHI_SQ_RED_G_MIN, None),)
        if VARIABILITY_CHI_SQ_RED_G_MIN is not None
        else ()
    ),
    *(
        ((LIGHT_CURVE_N_POINTS_COLUMN, LIGHT_CURVE_N_POINTS_MIN, None),)
        if LIGHT_CURVE_N_POINTS_MIN is not None
        else ()
    ),
    ("fracAGN_5100_fit", FRAC_AGN_5100_MIN, None),
    ("apparent_mag_2500_err", None, APPARENT_MAG_2500_ERR_MAX),
    (
        "m_2500_dereddened",
        COMPLETENESS_MAG_2500_MIN,
        COMPLETENESS_MAG_2500_MAX,
    ),
    ("sed_reduced_chi2", None, SED_REDUCED_CHI2_MAX),
    ("spectroscopy_reduced_chi2", None, SPECTROSCOPY_REDUCED_CHI2_MAX),
    ("joint_reduced_chi2", None, JAXSEDFIT_JOINT_REDUCED_CHI2_MAX),
    ("loo_chi2_eff", None, LOO_CHI2_EFF_MAX),
    *(
        (("num_divergences", 0.0, NUM_DIVERGENCES_MAX),)
        if NUM_DIVERGENCES_MAX is not None
        else ()
    ),
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


def add_t_rf_over_tau_uv_rf_column(
    df,
    *,
    column=T_RF_OVER_TAU_UV_RF_COLUMN,
):
    """Add the rest-frame monitoring-baseline to UV-timescale ratio.

    ``t_rf_length`` and ``tau_uv_rf = 10**log_tau_uv_rf`` are both measured
    in rest-frame days. Invalid or nonpositive inputs remain NaN so an active
    scalar quality cut rejects them.
    """

    required = ("t_rf_length", "log_tau_uv_rf")
    if not set(required).issubset(df.columns):
        return df, []

    t_rf = pd.to_numeric(df["t_rf_length"], errors="coerce").to_numpy(dtype=float)
    log_tau = pd.to_numeric(df["log_tau_uv_rf"], errors="coerce").to_numpy(dtype=float)
    ratio = np.full(len(df), np.nan, dtype=float)
    valid = np.isfinite(t_rf) & (t_rf > 0.0) & np.isfinite(log_tau)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        tau = np.power(10.0, log_tau[valid])
        values = t_rf[valid] / tau
    valid_values = np.isfinite(values) & (values > 0.0)
    valid_indices = np.flatnonzero(valid)
    ratio[valid_indices[valid_values]] = values[valid_values]

    df = df.copy()
    df[column] = ratio
    return df, list(required)
