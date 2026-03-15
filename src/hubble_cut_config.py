"""Default AGN selection-cut configuration for the QVC pipeline."""

DEFAULT_F_HOST_CUT = 0.1
DEFAULT_WRMS_CUT = 1.2
DEFAULT_IRON_FRAC_CUT = 1.0 # Wide default cut that allows all values to pass
DEFAULT_BC_FRAC_CUT = 1.0
DEFAULT_LOG_AMP_DELTA_BLR_UPPER_CUT = -0.2
DEFAULT_CHI_SQ_CUT = 10.0
DEFAULT_LOG_AMP_DELTA_BLR_UPPER_CUTS = {
    "u": DEFAULT_LOG_AMP_DELTA_BLR_UPPER_CUT,
    "g": DEFAULT_LOG_AMP_DELTA_BLR_UPPER_CUT,
    "r": DEFAULT_LOG_AMP_DELTA_BLR_UPPER_CUT,
    "i": DEFAULT_LOG_AMP_DELTA_BLR_UPPER_CUT,
} # Wide default cuts that allow all values to pass


def build_agn_cuts(
    *,
    f_host_cut=DEFAULT_F_HOST_CUT,
    iron_frac_cut=DEFAULT_IRON_FRAC_CUT,
    bc_frac_cut=DEFAULT_BC_FRAC_CUT,
    wrms_cut=DEFAULT_WRMS_CUT,
    chi_sq_g_cut=DEFAULT_CHI_SQ_CUT,
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

    return [
        ("log_tau_UV_RF", 1.5, 4.0),
        ("wrms", None, wrms_cut),
        ("t_rf_length", 1700, None),
        ("log_tau_UV_RF_err", 0.0, 1.0),
        ("log_sigma_UV_err", 0.0, 0.3),
        ("f_host_center", None, f_host_cut),
        ("frac_host_psf_2500", None, f_host_cut),
        ('f_fe_uv_over_pl_3000', None, iron_frac_cut),
        ('f_bc_over_pl_3000', None, bc_frac_cut),
        ("chi_sq_g", chi_sq_g_cut, None),
    ]


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
