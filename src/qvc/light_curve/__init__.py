"""Light-curve modeling and result-merging code."""

from qvc.light_curve.variability_metrics import (
    compute_binned_chi_squared,
    compute_variability_metrics_for_cleaned_lc,
    compute_variability_metrics_for_flat_row,
    gap_aligned_bins,
)

__all__ = [
    "compute_binned_chi_squared",
    "compute_variability_metrics_for_cleaned_lc",
    "compute_variability_metrics_for_flat_row",
    "gap_aligned_bins",
]
