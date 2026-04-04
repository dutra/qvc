"""Default AGN selection-cut configuration for the QVC pipeline."""

from qvc.hubble.cuts import (
    AGN_SCALAR_PARAMETER_CUTS,
    ALPHA_LAMBDA_MAX,
    F_BC_3000_MAX,
    F_FE_UV_3000_MAX,
    F_HOST_2500_MAX,
    LOG_AMP_DELTA_BLR_UPPER,
    LOG_AMP_DELTA_BLR_UPPER_BY_BAND,
    REDDENING_EBV_MAX,
    VARIABILITY_CHI_SQ_RED_G_MIN,
    WRMS_MAX,
)


DEFAULT_F_HOST_CUT = F_HOST_2500_MAX
DEFAULT_WRMS_CUT = WRMS_MAX
DEFAULT_IRON_FRAC_CUT = F_FE_UV_3000_MAX
DEFAULT_BC_FRAC_CUT = F_BC_3000_MAX
DEFAULT_CHI_SQ_CUT = VARIABILITY_CHI_SQ_RED_G_MIN
DEFAULT_REDDENING_EBV_CUT = REDDENING_EBV_MAX
DEFAULT_ALPHA_LAMBDA_UPPER_CUT = ALPHA_LAMBDA_MAX
DEFAULT_LOG_AMP_DELTA_BLR_UPPER_CUT = LOG_AMP_DELTA_BLR_UPPER

DEFAULT_LOG_AMP_DELTA_BLR_UPPER_CUTS = dict(LOG_AMP_DELTA_BLR_UPPER_BY_BAND)


def build_agn_cuts(
    *,
    f_host_cut=DEFAULT_F_HOST_CUT,
    iron_frac_cut=DEFAULT_IRON_FRAC_CUT,
    bc_frac_cut=DEFAULT_BC_FRAC_CUT,
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
    if iron_frac_cut is None:
        iron_frac_cut = DEFAULT_IRON_FRAC_CUT
    if wrms_cut is None:
        wrms_cut = DEFAULT_WRMS_CUT
    if bc_frac_cut is None:
        bc_frac_cut = DEFAULT_BC_FRAC_CUT
    if reddening_ebv_cut is None:
        reddening_ebv_cut = DEFAULT_REDDENING_EBV_CUT

    cut_overrides = {
        "wrms": (None, wrms_cut),
        "f_host_2500": (None, f_host_cut),
        "frac_host_psf_2500": (None, f_host_cut),
        "f_fe_uv_3000": (None, iron_frac_cut),
        "f_bc_3000": (None, bc_frac_cut),
        "variability_chi_sq_red_g": (variability_chi_sq_red_g_cut, None),
    }
    cuts = [
        (column, *cut_overrides.get(column, (lower, upper)))
        for column, lower, upper in AGN_SCALAR_PARAMETER_CUTS
    ]
    if reddening_ebv_cut is not None:
        cuts.append(("reddening_ebv", None, reddening_ebv_cut))
    return cuts


def build_log_amp_delta_blr_cuts(cuts=None):
    """
    Return the default per-band BLR amplitude cuts as (column, lower, upper) tuples.

    These are handled separately from the scalar AGN cuts because they are applied
    band by band and allow missing values to pass.
    """
    if cuts is None:
        cuts = DEFAULT_LOG_AMP_DELTA_BLR_UPPER_CUTS

    return [
        (f"log_amp_delta_blr_{band}", None, upper)
        for band, upper in cuts.items()
    ]
