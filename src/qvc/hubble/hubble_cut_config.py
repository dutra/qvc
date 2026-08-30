"""Default AGN selection-cut configuration for the QVC pipeline."""

import math

from qvc.hubble.cuts import (
    A_2500_TOTAL_MAX,
    AGN_TIER0_ELIGIBILITY_CUTS,
    AGN_TIER1_FIT_QUALITY_CUTS,
    AGN_TIER2_PARAMETER_CUTS,
    ALPHA_LAMBDA_MAX,
    F_HOST_2500_MAX,
    LOG_AMP_DELTA_BLR_UPPER,
    LOG_AMP_DELTA_BLR_UPPER_BY_BAND,
    REDDENING_EBV_MAX,
    VARIABILITY_CHI_SQ_RED_G_MIN,
    WRMS_MAX,
)


DEFAULT_F_HOST_CUT = F_HOST_2500_MAX
DEFAULT_WRMS_CUT = WRMS_MAX
DEFAULT_CHI_SQ_CUT = VARIABILITY_CHI_SQ_RED_G_MIN
DEFAULT_REDDENING_EBV_CUT = REDDENING_EBV_MAX
DEFAULT_ALPHA_LAMBDA_UPPER_CUT = ALPHA_LAMBDA_MAX
DEFAULT_LOG_AMP_DELTA_BLR_UPPER_CUT = LOG_AMP_DELTA_BLR_UPPER

DEFAULT_LOG_AMP_DELTA_BLR_UPPER_CUTS = dict(LOG_AMP_DELTA_BLR_UPPER_BY_BAND)


def _with_completeness_magnitude(cuts, completeness_magnitude):
    completeness_columns = {
        "dereddened": "m_2500_dereddened",
        "attenuated": "m_2500_attenuated_model",
    }
    if completeness_magnitude not in completeness_columns:
        raise ValueError(
            "completeness_magnitude must be 'dereddened' or 'attenuated', "
            f"got {completeness_magnitude!r}."
        )
    return [
        (
            completeness_columns[completeness_magnitude]
            if column == "m_2500_dereddened"
            else column,
            lower,
            upper,
        )
        for column, lower, upper in cuts
    ]


def build_tier0_cuts(*, completeness_magnitude="dereddened"):
    return _with_completeness_magnitude(
        AGN_TIER0_ELIGIBILITY_CUTS, completeness_magnitude
    )


def build_tier1_cuts():
    cuts = list(AGN_TIER1_FIT_QUALITY_CUTS)
    invalid = [
        column
        for column, lower, upper in cuts
        if (lower is None and upper is None)
        or any(
            not math.isfinite(float(value))
            for value in (lower, upper)
            if value is not None
        )
    ]
    if invalid:
        raise ValueError(
            "Tier-1 fit-quality thresholds must be finite and cannot be disabled: "
            f"{invalid}. Configure finite chi-square/R-hat thresholds."
        )
    return cuts


def build_tier2_cuts(
    *,
    completeness_magnitude="dereddened",
    f_host_cut=DEFAULT_F_HOST_CUT,
    wrms_cut=DEFAULT_WRMS_CUT,
    variability_chi_sq_red_g_cut=DEFAULT_CHI_SQ_CUT,
    reddening_ebv_cut=DEFAULT_REDDENING_EBV_CUT,
):
    """
    Return the default AGN quality cuts as (column, lower, upper) tuples.

    Thresholds are kept here so they are easy to inspect and adjust without
    digging through the data-loading logic.
    """
    if f_host_cut is None:
        f_host_cut = DEFAULT_F_HOST_CUT
    if wrms_cut is None:
        wrms_cut = DEFAULT_WRMS_CUT
    if reddening_ebv_cut is None:
        reddening_ebv_cut = DEFAULT_REDDENING_EBV_CUT

    cut_overrides = {
        "wrms": (None, wrms_cut),
        "f_host_2500": (None, f_host_cut),
        "variability_chi_sq_red_g": (variability_chi_sq_red_g_cut, None),
    }
    cuts = []
    for column, lower, upper in AGN_TIER2_PARAMETER_CUTS:
        resolved_lower, resolved_upper = cut_overrides.get(column, (lower, upper))
        if resolved_lower is None and resolved_upper is None:
            continue
        cuts.append((column, resolved_lower, resolved_upper))
    if A_2500_TOTAL_MAX is not None:
        cuts.append(("a_2500_total", None, A_2500_TOTAL_MAX))
    if reddening_ebv_cut is not None:
        cuts.append(("reddening_ebv", None, reddening_ebv_cut))
    return cuts


def build_agn_cuts(**kwargs):
    """Return all staged scalar cuts in application order."""

    completeness_magnitude = kwargs.get("completeness_magnitude", "dereddened")
    return (
        build_tier0_cuts(completeness_magnitude=completeness_magnitude)
        + build_tier1_cuts()
        + build_tier2_cuts(**kwargs)
    )


def build_dlog_amp_blr_cuts(cuts=None):
    """
    Return the default per-band BLR amplitude cuts as (column, lower, upper) tuples.

    These are handled separately from the scalar AGN cuts because they are applied
    band by band and allow missing values to pass.
    """
    if cuts is None:
        cuts = DEFAULT_LOG_AMP_DELTA_BLR_UPPER_CUTS

    return [
        (f"dlog_amp_blr_{band}", None, upper)
        for band, upper in cuts.items()
    ]
