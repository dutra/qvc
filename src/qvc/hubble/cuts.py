"""Centralized parameter-cut thresholds for the QVC Hubble pipeline."""

from ast import literal_eval
import os
import re

import numpy as np
import pandas as pd

from .completeness_grid import grid_bin_count, MAG_WIDTH_ENV, Z_WIDTH_ENV


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
CUT_TIER_CHOICES = ("none", "0", "1", "2")


def normalize_cut_tier(value):
    """Normalize the maximum Hubble cut tier to ``none``, ``0``, ``1``, or ``2``."""

    normalized = str(value).strip().lower()
    if normalized not in CUT_TIER_CHOICES:
        raise ValueError(
            f"Unknown cut tier {value!r}; choose one of: "
            f"{', '.join(CUT_TIER_CHOICES)}."
        )
    return normalized


def cut_tier_level(value):
    """Return an ordered numeric level, using -1 for the uncut diagnostic mode."""

    normalized = normalize_cut_tier(value)
    return -1 if normalized == "none" else int(normalized)
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

LOG_TAU_UV_RF_MIN = _cut_env_float("QVC_CUT_LOG_TAU_UV_RF_MIN", 1.3)
LOG_TAU_UV_RF_MAX = _cut_env_float("QVC_CUT_LOG_TAU_UV_RF_MAX", None)
T_RF_OVER_TAU_UV_RF_COLUMN = "t_rf_over_tau_uv_rf"
T_RF_OVER_TAU_UV_RF_MIN = _cut_env_float(
    "QVC_CUT_T_RF_OVER_TAU_UV_RF_MIN", None
)
FRAC_AGN_5100_MIN = _cut_env_float("QVC_CUT_FRAC_AGN_5100_MIN", None)
APPARENT_MAG_2500_ERR_MAX = _cut_env_float(
    "QVC_CUT_APPARENT_MAG_2500_ERR_MAX", None
)

# JAXSEDFit goodness-of-fit and posterior-convergence diagnostics.
JAXSEDFIT_JOINT_REDUCED_CHI2_MAX = _cut_env_float(
    "QVC_CUT_JAXSEDFIT_JOINT_REDUCED_CHI2_MAX", 1.2
)
SED_REDUCED_CHI2_MAX = _cut_env_float("QVC_CUT_SED_REDUCED_CHI2_MAX", 1.5)
SPECTROSCOPY_REDUCED_CHI2_MAX = _cut_env_float(
    "QVC_CUT_SPECTROSCOPY_REDUCED_CHI2_MAX", 1.1
)
LOO_CHI2_EFF_MAX = _cut_env_float("QVC_CUT_LOO_CHI2_EFF_MAX", 1.0)
# Reject spectral solutions whose inferred rest-frame 2500 A attenuation is so
# large that the attenuated completeness coordinate admits extreme Hubble
# outliers.  This is exactly
# m_2500_attenuated_model - m_2500_dereddened.
A_2500_TOTAL_MAX = _cut_env_float("QVC_CUT_A_2500_TOTAL_MAX", None)
EBV_GAL_PLUS_EBV_AGN_COLUMN = "ebv_gal_plus_ebv_agn"
EBV_GAL_PLUS_EBV_AGN_MAX = _cut_env_float(
    "QVC_CUT_EBV_GAL_PLUS_EBV_AGN_MAX", None
)
SPECTRAL_RHAT_MAX = _cut_env_float("QVC_CUT_SPECTRAL_RHAT_MAX", 1.1)
LIGHT_CURVE_RHAT_MAX = _cut_env_float("QVC_CUT_LIGHT_CURVE_RHAT_MAX", 1.05)
NUM_DIVERGENCES_MAX = _cut_env_float("QVC_CUT_NUM_DIVERGENCES_MAX", 0.0)

# The completeness map is tabulated at histogram-bin centers on a deliberately
# padded physical grid. Science-selection coordinates lie inside that center
# hull, so no magnitude/redshift extrapolation or clipping is needed.
COMPLETENESS_MAP_MAG_EDGE_MIN = 16.5
COMPLETENESS_MAP_MAG_EDGE_MAX = 24.5
COMPLETENESS_N_MAG_BINS = grid_bin_count(
    COMPLETENESS_MAP_MAG_EDGE_MAX - COMPLETENESS_MAP_MAG_EDGE_MIN,
    os.environ.get(MAG_WIDTH_ENV, "0.1"),
)
COMPLETENESS_MAP_Z_EDGE_MIN = 0.0
COMPLETENESS_MAP_Z_EDGE_MAX = 4.5
COMPLETENESS_N_Z_BINS = grid_bin_count(
    COMPLETENESS_MAP_Z_EDGE_MAX - COMPLETENESS_MAP_Z_EDGE_MIN,
    os.environ.get(Z_WIDTH_ENV, "0.1"),
)
COMPLETENESS_Z_BIN_WIDTH = (
    COMPLETENESS_MAP_Z_EDGE_MAX - COMPLETENESS_MAP_Z_EDGE_MIN
) / COMPLETENESS_N_Z_BINS
COMPLETENESS_Z_MIN = (
    COMPLETENESS_MAP_Z_EDGE_MIN + 0.5 * COMPLETENESS_Z_BIN_WIDTH
)
COMPLETENESS_Z_MAX = (
    COMPLETENESS_MAP_Z_EDGE_MAX - 0.5 * COMPLETENESS_Z_BIN_WIDTH
)

# The Hubble sample selection is deliberately narrower than the padded map.
COMPLETENESS_MAG_EDGE_MIN = 17.0
COMPLETENESS_MAG_EDGE_MAX = 24.0
COMPLETENESS_MAG_BIN_WIDTH = (
    COMPLETENESS_MAP_MAG_EDGE_MAX - COMPLETENESS_MAP_MAG_EDGE_MIN
) / COMPLETENESS_N_MAG_BINS
COMPLETENESS_MAG_2500_MIN = _cut_env_float(
    "QVC_CUT_COMPLETENESS_MAG_2500_MIN", COMPLETENESS_MAG_EDGE_MIN
)
COMPLETENESS_MAG_2500_MAX = _cut_env_float(
    "QVC_CUT_COMPLETENESS_MAG_2500_MAX", COMPLETENESS_MAG_EDGE_MAX
)

# Magnitude-selection policies.  The completeness map is calibrated only over
# the finite interval above, so every ordinary Hubble fit defaults to that hard
# support.  Tail extrapolation remains available only as an explicit opt-in.
COMPLETENESS_MAGNITUDE_SUPPORT_MODES = ("tails", "hard-cut")
DEFAULT_COMPLETENESS_MAGNITUDE_SUPPORT_MODE = "hard-cut"
COMPLETENESS_TAIL_MAG_2500_MIN = 14.0
COMPLETENESS_TAIL_MAG_2500_MAX = 32.0


def normalize_completeness_magnitude_support_mode(mode):
    normalized = str(mode).strip().lower().replace("_", "-")
    if normalized not in COMPLETENESS_MAGNITUDE_SUPPORT_MODES:
        raise ValueError(
            "completeness_magnitude_support_mode must be one of "
            f"{COMPLETENESS_MAGNITUDE_SUPPORT_MODES}; got {mode!r}."
        )
    return normalized
if (
    COMPLETENESS_MAG_2500_MIN is None
    or COMPLETENESS_MAG_2500_MAX is None
    or COMPLETENESS_MAG_2500_MIN < COMPLETENESS_MAG_EDGE_MIN
    or COMPLETENESS_MAG_2500_MAX > COMPLETENESS_MAG_EDGE_MAX
    or COMPLETENESS_MAG_2500_MIN >= COMPLETENESS_MAG_2500_MAX
):
    raise ValueError(
        "Completeness magnitude support must be a finite increasing interval "
        f"within [{COMPLETENESS_MAG_EDGE_MIN}, {COMPLETENESS_MAG_EDGE_MAX}]; "
        f"got [{COMPLETENESS_MAG_2500_MIN}, {COMPLETENESS_MAG_2500_MAX}]."
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
T_RF_LENGTH_MIN = _cut_env_float("QVC_CUT_T_RF_LENGTH_MIN", None)
LIGHT_CURVE_N_POINTS_MIN = _cut_env_float("QVC_CUT_LIGHT_CURVE_N_POINTS_MIN", None)
SN_MEDIAN_ALL_MIN = _cut_env_float("QVC_CUT_SN_MEDIAN_ALL_MIN", 3.0)
ETA_SIGMA_KL_MIN = _cut_env_float("QVC_CUT_ETA_SIGMA_KL_MIN", None)
for _name, _threshold in (
    ("QVC_CUT_T_RF_LENGTH_MIN", T_RF_LENGTH_MIN),
    ("QVC_CUT_LIGHT_CURVE_N_POINTS_MIN", LIGHT_CURVE_N_POINTS_MIN),
    ("QVC_CUT_SN_MEDIAN_ALL_MIN", SN_MEDIAN_ALL_MIN),
    ("QVC_CUT_ETA_SIGMA_KL_MIN", ETA_SIGMA_KL_MIN),
):
    if _threshold is not None and not (np.isfinite(_threshold) and _threshold >= 0):
        raise ValueError(f"{_name} must be finite and nonnegative or none.")
ALPHA_LAMBDA_MIN = None
ALPHA_LAMBDA_MAX = None

LOG_SIGMA_UV_MIN = None
LOG_SIGMA_UV_MAX = None
REDDENING_EBV_MAX = None

VARIABILITY_CHI_SQ_RED_G_MIN = _cut_env_float(
    "QVC_CUT_VARIABILITY_CHI_SQ_RED_G_MIN", None
)
if VARIABILITY_CHI_SQ_RED_G_MIN is not None and not (
    np.isfinite(VARIABILITY_CHI_SQ_RED_G_MIN)
    and VARIABILITY_CHI_SQ_RED_G_MIN >= 0.0
):
    raise ValueError(
        "QVC_CUT_VARIABILITY_CHI_SQ_RED_G_MIN must be finite and nonnegative or none."
    )
F_HOST_2500_MAX = None
F_HOST_2500_PSF_MAX = _cut_env_float("QVC_CUT_F_HOST_2500_PSF_MAX", None)
if F_HOST_2500_PSF_MAX is not None and not 0.0 <= F_HOST_2500_PSF_MAX <= 1.0:
    raise ValueError("QVC_CUT_F_HOST_2500_PSF_MAX must be within [0, 1] or none.")
LOW_L2500_FHOST_LOG_L_MAX = _cut_env_float(
    "QVC_CUT_LOW_L2500_FHOST_LOG_L_MAX", None
)
LOW_L2500_FHOST_PSF_MAX = _cut_env_float(
    "QVC_CUT_LOW_L2500_FHOST_PSF_MAX", None
)
LOG_AMP_DELTA_BLR_UPPER = None
LOG_AMP_DELTA_BLR_UPPER_BY_BAND = {}
LOG_AMP_DELTA_BC_UPPER = None


def _fraction_cut(name, default):
    threshold = _cut_env_float(name, default)
    if threshold is None:
        return None
    if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError(f"{name} must be within (0, 1] or none.")
    return float(threshold)


F_BC_3000_MAX = _fraction_cut("QVC_CUT_F_BC_3000_MAX", None)
F_FE_UV_3000_MAX = _fraction_cut("QVC_CUT_F_FE_UV_3000_MAX", None)
LOG_F_BC_3000_MAX = (
    None if F_BC_3000_MAX is None else float(np.log10(F_BC_3000_MAX))
)
LOG_F_FE_UV_3000_MAX = (
    None if F_FE_UV_3000_MAX is None else float(np.log10(F_FE_UV_3000_MAX))
)
REL_APPARENT_MAG_2500_ERR_MAX = None


EXCLUDED_SPECTRAL_SDSS_NAMES = (
    "221120.38+010905.6",
    "024555.35+005332.6",
    "015802.36+002917.3",
    "014641.18+010815.9",
    "010151.08+002028.8",
    "025646.57+003858.3",
    "215013.64-001627.2",
    "221018.27+005832.1",
    "220311.37+005056.3",
   # "020950.71-000506.4",
)
EXCLUDED_LIGHT_CURVE_SDSS_NAMES = (
    "020618.17-003844.1",
    "213638.58+004154.2",
    "010922.91+005305.2",
    "233631.11+011027.4",
    "000701.32+002242.3",
    "213414.33-001342.6",
    # Very large light-curve outliers that our detectors did not capture.
    "000709.92+001854.2",
    "210952.72-004745.3",
    "000222.47-000443.5",
    "000121.86-000327.1",
)
EXCLUDED_SDSS_NAMES = (
    *EXCLUDED_SPECTRAL_SDSS_NAMES,
    *EXCLUDED_LIGHT_CURVE_SDSS_NAMES,
)
AGN_TIER0_ELIGIBILITY_CUTS = (
    ("z", COMPLETENESS_Z_MIN, COMPLETENESS_Z_MAX),
    (
        "m_2500_dereddened",
        COMPLETENESS_MAG_2500_MIN,
        COMPLETENESS_MAG_2500_MAX,
    ),
)

AGN_TIER0_EXTREME_MAGNITUDE_CUTS = (
    ("z", COMPLETENESS_Z_MIN, COMPLETENESS_Z_MAX),
    (
        "m_2500_dereddened",
        COMPLETENESS_TAIL_MAG_2500_MIN,
        COMPLETENESS_TAIL_MAG_2500_MAX,
    ),
)

AGN_TIER1_FIT_QUALITY_CUTS = (
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

AGN_TIER2_PARAMETER_CUTS = (
    ("t_rf_length", T_RF_LENGTH_MIN, None),
    (LIGHT_CURVE_N_POINTS_COLUMN, LIGHT_CURVE_N_POINTS_MIN, None),
    ("SN_MEDIAN_ALL", SN_MEDIAN_ALL_MIN, None),
    ("variability_chi_sq_red_g", VARIABILITY_CHI_SQ_RED_G_MIN, None),
    ("eta_sigma_kl", ETA_SIGMA_KL_MIN, None),
    ("log_tau_uv_rf", LOG_TAU_UV_RF_MIN, LOG_TAU_UV_RF_MAX),
    *(
        ((T_RF_OVER_TAU_UV_RF_COLUMN, T_RF_OVER_TAU_UV_RF_MIN, None),)
        if T_RF_OVER_TAU_UV_RF_MIN is not None
        else ()
    ),
    ("fracAGN_5100_fit", FRAC_AGN_5100_MIN, None),
    ("apparent_mag_2500_err", None, APPARENT_MAG_2500_ERR_MAX),
    (EBV_GAL_PLUS_EBV_AGN_COLUMN, None, EBV_GAL_PLUS_EBV_AGN_MAX),
    *(
        (("f_host_2500_psf", 0.0, F_HOST_2500_PSF_MAX),)
        if F_HOST_2500_PSF_MAX is not None
        else ()
    ),
)

# Tier-1 convergence diagnostics are mandatory when Tier 1 is active.
ALLOW_MISSING_SCALAR_CUT_COLUMNS = frozenset()

# Fixed-definition ("reference") chi-square columns written by the joint
# spectra wrapper (jaxsedfit ``*_chi2_ref`` sites). The stored columns use the
# likelihood's variance, which includes fitted noise terms; the reference
# columns use a fixed variance that does not depend on any fitted parameter:
# spectra err**2 + (0.08*model)**2 per pixel; photometry the likelihood's
# variance with the systematic and AGN-mismatch widths fixed to 0.10 (times
# the model band flux) and 0.15 (times the AGN band flux); joint the unweighted
# sum of the two over the summed counts.
#
# Only the spectral reference has the definition of an earlier statistic (the
# fixed 8% floor the spectroscopic threshold was set on). The SED reference is
# a new statistic (earlier fits used fitted widths, as the stored column does),
# and so is the joint reference (earlier joint values weighted the spectrum
# and used the fitted-width SED term). Their thresholds are inherited and have
# to be assessed on data: the per-cut statistic switches and the stored-versus-
# reference comparison in the cut report exist for that.
REFERENCE_CHI2_CUT_COLUMNS = {
    "sed_reduced_chi2": "sed_reduced_chi2_ref",
    "spectroscopy_reduced_chi2": "spectroscopy_reduced_chi2_ref",
    "joint_reduced_chi2": "joint_reduced_chi2_ref",
}
# Per-cut choice of statistic: "reference" (default; falls back to the stored
# column for catalogues that have no reference columns) or "stored".
CHI2_CUT_STATISTIC_ENV = {
    "sed_reduced_chi2": "QVC_CUT_SED_CHI2_STATISTIC",
    "spectroscopy_reduced_chi2": "QVC_CUT_SPECTROSCOPY_CHI2_STATISTIC",
    "joint_reduced_chi2": "QVC_CUT_JOINT_CHI2_STATISTIC",
}
CHI2_STATISTIC_CHOICES = ("reference", "stored")
# Stored statistics whose definition changed when jaxsedfit started fitting
# the spectral jitter: they depend on S/N, so cutting on them in a fitted-noise
# catalogue needs the explicit override. The stored SED definition (fitted
# systematic and AGN widths) did not change.
SNR_DEPENDENT_STORED_CHI2 = frozenset({"spectroscopy_reduced_chi2", "joint_reduced_chi2"})
# Columns present only in catalogues from jaxsedfit versions that fit the
# spectral noise (jitter).
FITTED_SPECTRAL_NOISE_COLUMNS = ("frac_jitter", "add_jitter", "frac_fe_jitter")
# Explicit per-row noise-model marker (minimal Hubble inputs carry it).
SPECTRAL_NOISE_MODEL_COLUMN = "spectral_noise_model"
SPECTRAL_NOISE_MODELS = ("fixed_floor", "fitted_jitter")
ALLOW_FITTED_NOISE_CHI2_ENV = "QVC_CUT_ALLOW_FITTED_NOISE_CHI2"
ALLOW_MISSING_REFERENCE_CHI2_ENV = "QVC_CUT_ALLOW_MISSING_REFERENCE_CHI2"
ALLOW_MIXED_PHOTOMETRY_EXCLUSION_ENV = "QVC_CUT_ALLOW_MIXED_PHOTOMETRY_EXCLUSION"
# Columns recording the sub-Lyman photometry exclusion of a row (NaN = off;
# catalogues from before the option lack them, which also means off), and the
# chi-square cuts that contain the photometric chi-square and so depend on it.
PHOTOMETRY_EXCLUSION_WAVELENGTH_COLUMN = "photometry_exclusion_rest_wavelength"
PHOTOMETRY_EXCLUSION_FRACTION_COLUMN = "photometry_exclusion_max_fraction"
PHOTOMETRY_EXCLUSION_DEPENDENT_CUTS = frozenset({"sed_reduced_chi2", "joint_reduced_chi2"})
# Reference variance floors the Tier-1 thresholds assume; the catalogue's
# recorded floors must match them.
REFERENCE_FLOOR_ENV = {
    "reference_spectrum_floor": ("QVC_CUT_REFERENCE_SPECTRUM_FLOOR", 0.08),
    "reference_photometry_floor": ("QVC_CUT_REFERENCE_PHOTOMETRY_FLOOR", 0.10),
    "reference_agn_floor": ("QVC_CUT_REFERENCE_AGN_FLOOR", 0.15),
}


def _env_flag(name):
    value = str(os.environ.get(name, "")).strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"{name} must be 0/1 (or true/false), got {value!r}")


def expected_reference_floors():
    """Return the reference floors the thresholds assume (env-overridable)."""
    floors = {}
    for field, (env_name, default) in REFERENCE_FLOOR_ENV.items():
        value = _cut_env_float(env_name, default)
        if value is None or not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{env_name} must be a finite non-negative number, got {value!r}")
        floors[field] = float(value)
    return floors


def chi2_cut_statistics(statistics=None):
    """Return the requested statistic per chi-square cut, validated.

    ``statistics`` overrides the ``QVC_CUT_*_CHI2_STATISTIC`` environment
    variables per cut; unset cuts default to ``"reference"``.
    """
    statistics = dict(statistics or {})
    unknown_cuts = sorted(set(statistics) - set(CHI2_CUT_STATISTIC_ENV))
    if unknown_cuts:
        raise ValueError(f"Unknown chi-square cut(s): {unknown_cuts}")
    resolved = {}
    for column, env_name in CHI2_CUT_STATISTIC_ENV.items():
        value = statistics.get(column, os.environ.get(env_name))
        value = "reference" if value is None or not str(value).strip() else str(value).strip().lower()
        if value not in CHI2_STATISTIC_CHOICES:
            raise ValueError(
                f"{env_name} must be one of {CHI2_STATISTIC_CHOICES}, got {value!r}"
            )
        resolved[column] = value
    return resolved


def _numeric(frame, column):
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def fitted_noise_markers(frame_or_columns):
    """Return the evidence that a catalogue comes from a fitted-noise fit."""
    columns = set(getattr(frame_or_columns, "columns", frame_or_columns))
    markers = [column for column in FITTED_SPECTRAL_NOISE_COLUMNS if column in columns]
    if SPECTRAL_NOISE_MODEL_COLUMN in columns and hasattr(frame_or_columns, "columns"):
        values = set(frame_or_columns[SPECTRAL_NOISE_MODEL_COLUMN].astype(str).str.strip())
        unknown = sorted(values - set(SPECTRAL_NOISE_MODELS))
        if unknown:
            raise ValueError(
                f"Unknown {SPECTRAL_NOISE_MODEL_COLUMN} value(s) {unknown}; "
                f"expected {SPECTRAL_NOISE_MODELS}."
            )
        if "fitted_jitter" in values:
            markers.append(f"{SPECTRAL_NOISE_MODEL_COLUMN}=fitted_jitter")
    return markers


def check_reference_floors(frame, floors=None):
    """Require every recorded, finite reference floor to equal the expected one.

    Returns the expected floors. Raises when a floor column is missing or a
    recorded value differs, since the thresholds assume these floors.
    """
    floors = expected_reference_floors() if floors is None else dict(floors)
    problems = []
    for field, expected in floors.items():
        if field not in frame.columns:
            problems.append(f"{field} column missing")
            continue
        values = _numeric(frame, field)
        finite = values[np.isfinite(values)]
        bad = finite[~np.isclose(finite, expected, rtol=1e-9, atol=1e-12)]
        if bad.size:
            problems.append(
                f"{field}: {bad.size} row(s) with {sorted(set(np.round(bad, 6)))[:5]}, expected {expected:g}"
            )
    if problems:
        raise ValueError(
            "Reference chi-square floors do not match the floors the Tier-1 thresholds "
            "assume (" + "; ".join(problems) + "). Re-predict with the expected floors, "
            "or set QVC_CUT_REFERENCE_*_FLOOR to the catalogue's floors and re-assess "
            "the thresholds."
        )
    return floors


def photometry_exclusion_settings(frame):
    """Return ``{setting: row count}`` of a catalogue's photometry exclusion.

    A setting is ``"off"`` or ``(rest wavelength, max fraction)``. A NaN or
    missing wavelength is off, whatever the fraction column says.
    """
    n_rows = len(frame)
    if PHOTOMETRY_EXCLUSION_WAVELENGTH_COLUMN not in frame.columns:
        return {"off": n_rows} if n_rows else {}
    wavelength = _numeric(frame, PHOTOMETRY_EXCLUSION_WAVELENGTH_COLUMN)
    fraction = (
        _numeric(frame, PHOTOMETRY_EXCLUSION_FRACTION_COLUMN)
        if PHOTOMETRY_EXCLUSION_FRACTION_COLUMN in frame.columns
        else np.full(n_rows, np.nan)
    )
    counts = {}
    for lam, frac in zip(wavelength, fraction):
        key = "off" if not np.isfinite(lam) else (round(float(lam), 6), round(float(frac), 6))
        counts[key] = counts.get(key, 0) + 1
    return counts


def check_photometry_exclusion_settings(frame, *, allow_mixed=None):
    """Require one photometry-exclusion setting for every row of a catalogue.

    The SED (and joint) chi-square of a row depends on which bands its fit left
    out, so a cut on it across rows with different settings compares different
    statistics. Raises unless all rows share one setting, or
    ``allow_mixed`` / ``QVC_CUT_ALLOW_MIXED_PHOTOMETRY_EXCLUSION=1``. Returns
    the settings with their row counts.
    """
    counts = photometry_exclusion_settings(frame)
    if len(counts) <= 1:
        return counts
    allow = _env_flag(ALLOW_MIXED_PHOTOMETRY_EXCLUSION_ENV) if allow_mixed is None else bool(allow_mixed)
    if allow:
        return counts
    detail = "; ".join(
        f"{'off' if key == 'off' else f'{key[0]:g} A, max fraction {key[1]:g}'}: {n} row(s)"
        for key, n in counts.items()
    )
    raise ValueError(
        "The SED/joint chi-square cut is applied to a catalogue whose rows use "
        f"different sub-Lyman photometry exclusion settings ({detail}); their "
        "photometric chi-squares count different bands. Re-predict all rows with one "
        "setting (the joint spectra wrapper with --resume ... --resume-only and the "
        f"same --drop-sub-lyman-photometry options), or set "
        f"{ALLOW_MIXED_PHOTOMETRY_EXCLUSION_ENV}=1 to cut anyway."
    )


def resolve_chi2_cut_columns(
    frame,
    cuts,
    *,
    statistics=None,
    allow_fitted_noise_chi2=None,
    allow_missing_reference_chi2=None,
    floors=None,
    allow_mixed_photometry_exclusion=None,
):
    """Return Tier-1 cuts with each chi-square column mapped to the one to use.

    ``frame`` is the catalogue (a DataFrame; a plain list of column names is
    accepted, then the value checks below are skipped). Per chi-square cut,
    with the requested statistic from :func:`chi2_cut_statistics`:

    - ``reference`` and the ``*_ref`` column present: cut on it, same threshold.
      Rows whose stored value is finite but reference value is not would be
      dropped silently (for example when spectra files of different versions
      are combined), so they raise unless ``allow_missing_reference_chi2`` or
      ``QVC_CUT_ALLOW_MISSING_REFERENCE_CHI2=1``; their count is always
      reported. The recorded reference floors must equal the expected ones.
    - ``reference`` without reference columns: a catalogue without fitted-noise
      markers (fixed spectral floor) uses the stored column, whose definition
      the thresholds were set on; a fitted-noise catalogue uses the stored SED
      column, and for spectroscopy/joint raises unless the override is set.
    - ``stored``: the stored column; for spectroscopy/joint in a fitted-noise
      catalogue only with the override (``allow_fitted_noise_chi2`` or
      ``QVC_CUT_ALLOW_FITTED_NOISE_CHI2=1``).

    When an SED or joint chi-square cut is present, every row must share one
    sub-Lyman photometry exclusion setting (:func:`check_photometry_exclusion_settings`;
    override ``allow_mixed_photometry_exclusion`` or
    ``QVC_CUT_ALLOW_MIXED_PHOTOMETRY_EXCLUSION=1``).

    Returns ``(cuts, choices)``; ``choices`` maps each chi-square cut to
    ``{"statistic", "column", "reason", "missing_reference"}`` and, when any
    reference column is used, ``choices["reference_floors"]`` holds the floors.
    """
    has_values = hasattr(frame, "columns")
    columns = set(frame.columns if has_values else frame)
    requested = chi2_cut_statistics(statistics)
    chi2_cuts = [column for column, _lower, _upper in cuts if column in REFERENCE_CHI2_CUT_COLUMNS]
    reference_present = [
        REFERENCE_CHI2_CUT_COLUMNS[column]
        for column in chi2_cuts
        if REFERENCE_CHI2_CUT_COLUMNS[column] in columns
    ]
    if reference_present and len(reference_present) != len(chi2_cuts):
        missing = sorted(
            {REFERENCE_CHI2_CUT_COLUMNS[c] for c in chi2_cuts} - set(reference_present)
        )
        raise ValueError(
            f"Input catalogue has only some reference chi-square columns; missing {missing}."
        )
    fitted_noise = fitted_noise_markers(frame)
    allow = None
    choices = {}
    blocked = []
    missing_reference = {}
    for column in chi2_cuts:
        reference = REFERENCE_CHI2_CUT_COLUMNS[column]
        statistic = requested[column]
        if statistic == "reference" and reference in columns:
            n_missing = 0
            if has_values and column in columns:
                stored_values = _numeric(frame, column)
                reference_values = _numeric(frame, reference)
                n_missing = int(
                    np.count_nonzero(np.isfinite(stored_values) & ~np.isfinite(reference_values))
                )
            if n_missing:
                missing_reference[column] = n_missing
            choices[column] = {
                "statistic": statistic,
                "column": reference,
                "reason": "reference column",
                "missing_reference": n_missing,
            }
            continue
        if not fitted_noise:
            reason = (
                "stored column (requested)"
                if statistic == "stored"
                else "no reference column; fixed-floor catalogue"
            )
        elif column not in SNR_DEPENDENT_STORED_CHI2:
            reason = (
                "stored column (requested); stored SED definition unchanged"
                if statistic == "stored"
                else "no reference column; stored SED definition unchanged"
            )
        else:
            if allow is None:
                allow = (
                    _env_flag(ALLOW_FITTED_NOISE_CHI2_ENV)
                    if allow_fitted_noise_chi2 is None
                    else bool(allow_fitted_noise_chi2)
                )
            if not allow:
                blocked.append((column, statistic))
                continue
            reason = f"stored fitted-noise column ({ALLOW_FITTED_NOISE_CHI2_ENV}=1)"
        choices[column] = {
            "statistic": statistic,
            "column": column,
            "reason": reason,
            "missing_reference": 0,
        }
    if blocked:
        detail = ", ".join(f"{column} ({statistic})" for column, statistic in blocked)
        raise ValueError(
            "Input catalogue comes from a fit with a fitted spectral noise model "
            f"({fitted_noise}); its stored {[c for c, _ in blocked]} use the "
            "fitted noise and depend on S/N, so the Tier-1 thresholds do not apply "
            f"to them. Blocked cuts: {detail}. Use the reference statistic (re-predict "
            "the saved posteriors with the joint spectra wrapper, --resume ... "
            "--resume-only, to add the *_ref columns), or set "
            f"{ALLOW_FITTED_NOISE_CHI2_ENV}=1 to cut on the fitted-noise columns."
        )
    if missing_reference:
        allow_missing = (
            _env_flag(ALLOW_MISSING_REFERENCE_CHI2_ENV)
            if allow_missing_reference_chi2 is None
            else bool(allow_missing_reference_chi2)
        )
        if not allow_missing:
            raise ValueError(
                "Rows with a finite stored chi-square but no reference value would be "
                f"removed by the reference cut: {missing_reference}. This happens when "
                "spectra catalogues with and without reference columns are combined. "
                "Re-predict the older catalogue, or set "
                f"{ALLOW_MISSING_REFERENCE_CHI2_ENV}=1 to remove those rows."
            )
    if any(choice["column"] != cut for cut, choice in choices.items()) and has_values:
        choices["reference_floors"] = check_reference_floors(frame, floors)
    if has_values and PHOTOMETRY_EXCLUSION_DEPENDENT_CUTS.intersection(chi2_cuts):
        check_photometry_exclusion_settings(frame, allow_mixed=allow_mixed_photometry_exclusion)
    resolved = [
        (choices[column]["column"] if column in choices else column, lower, upper)
        for column, lower, upper in cuts
    ]
    return resolved, choices


def summarize_chi2_cut_choices(choices):
    """Return a one-line description of the chi-square columns a cut used."""
    parts = []
    for column, choice in choices.items():
        if column == "reference_floors":
            parts.append("reference floors " + ", ".join(f"{k}={v:g}" for k, v in choice.items()))
            continue
        text = f"{column}: {choice['column']} [{choice['statistic']}: {choice['reason']}]"
        if choice.get("missing_reference"):
            text += f" ({choice['missing_reference']} row(s) without a reference value)"
        parts.append(text)
    return "; ".join(parts)


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
    """Add the rest-frame monitoring-baseline to UV-timescale ratio."""

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
