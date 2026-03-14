"""Default AGN selection-cut configuration for the QVC pipeline."""

DEFAULT_F_HOST_CUT = 0.1
DEFAULT_IRON_FRAC_CUT = 10.0
DEFAULT_REDCHI2_CONTI_FULL_CUT = 1.2
DEFAULT_LOG_AMP_DELTA_BLR_UPPER_CUTS = {
    "u": 0.0,
    "g": 0.0,
    "r": 0.0,
    "i": 0.0,
}


def build_agn_cuts(
    *,
    f_host_cut=DEFAULT_F_HOST_CUT,
    iron_frac_cut=DEFAULT_IRON_FRAC_CUT,
    redchi2_cut=DEFAULT_REDCHI2_CONTI_FULL_CUT,
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
    if redchi2_cut is None:
        redchi2_cut = DEFAULT_REDCHI2_CONTI_FULL_CUT

    return [
        ("log_tau_UV_RF", 1.5, 4.0),
        ("wrms", None, redchi2_cut),
        ("t_rf_length", 1700, None),
        ("iron_frac", None, iron_frac_cut),
        ("log_tau_UV_RF_err", 0.0, 1.0),
        ("log_sigma_UV_err", 0.0, 0.3),
        ("f_host_center", -1.0, f_host_cut),
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
