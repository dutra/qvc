import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.light_curve.variability_metrics import (
    compute_binned_chi_squared,
    compute_variability_metrics_for_bands,
    compute_variability_metrics_for_flat_row,
    gap_aligned_bins,
)


def _notebook_g_chi_sq_red(times, mags, errs):
    times = np.asarray(times, dtype=float)
    mags = np.asarray(mags, dtype=float)
    errs = np.asarray(errs, dtype=float)
    bin_edges = gap_aligned_bins(times)

    good = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs) & (errs > 0)
    times = times[good]
    mags = mags[good]
    errs = errs[good]

    bin_indices = np.digitize(times, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)

    n_bins = len(bin_edges) - 1
    bin_means = np.full(n_bins, np.nan)
    bin_errors = np.full(n_bins, np.nan)
    for idx in range(n_bins):
        mask = bin_indices == idx
        if np.sum(mask) == 0:
            continue
        weights = 1.0 / np.square(errs[mask])
        weights_sum = np.sum(weights)
        if weights_sum <= 0:
            continue
        bin_means[idx] = np.sum(mags[mask] * weights) / weights_sum
        bin_errors[idx] = np.sqrt(1.0 / weights_sum)

    valid_bins = np.isfinite(bin_means) & np.isfinite(bin_errors) & (bin_errors > 0)
    weighted_mean = np.average(bin_means[valid_bins], weights=1.0 / np.square(bin_errors[valid_bins]))
    chi_sq = np.sum(np.square((bin_means[valid_bins] - weighted_mean) / bin_errors[valid_bins]))
    return float(chi_sq / (np.sum(valid_bins) - 1))


def test_g_band_reduced_chi_squared_matches_notebook_logic():
    times = np.array([0.0, 20.0, 380.0, 395.0], dtype=float)
    mags = np.array([20.0, 20.2, 19.7, 19.9], dtype=float)
    errs = np.array([0.1, 0.1, 0.1, 0.1], dtype=float)

    result = compute_binned_chi_squared(times, mags, errs, gap_aligned_bins(times))

    assert np.isclose(result.chi_sq_red, _notebook_g_chi_sq_red(times, mags, errs))


def test_variability_metrics_keep_raw_reduced_pvalue_relationships():
    metrics = compute_variability_metrics_for_bands(
        {"g": np.array([0.0, 15.0, 400.0, 420.0], dtype=float)},
        {"g": np.array([20.0, 20.3, 19.6, 20.0], dtype=float)},
        {"g": np.full(4, 0.1, dtype=float)},
        bands=["g"],
    )

    dof = metrics["variability_dof_g"]
    chi_sq = metrics["variability_chi_sq_g"]
    chi_sq_red = metrics["variability_chi_sq_red_g"]
    pvalue = metrics["variability_pvalue_g"]

    assert np.isfinite(dof)
    assert np.isclose(chi_sq, chi_sq_red * dof)
    assert np.isfinite(pvalue)
    assert np.isclose(metrics["variability_neg_log10_pvalue_g"], -np.log10(pvalue))


def test_variability_metrics_handle_insufficient_data_and_missing_flattened_series():
    metrics = compute_variability_metrics_for_bands(
        {"g": np.array([10.0], dtype=float)},
        {"g": np.array([20.0], dtype=float)},
        {"g": np.array([0.1], dtype=float)},
        bands=["g"],
    )
    assert metrics["variability_n_points_g"] == 1
    assert np.isnan(metrics["variability_chi_sq_g"])
    assert np.isnan(metrics["variability_pvalue_g"])
    assert np.isnan(metrics["variability_neg_log10_pvalue_g"])

    flat_metrics = compute_variability_metrics_for_flat_row({"bands": "g"})
    assert flat_metrics["variability_n_points_g"] == 0
    assert np.isnan(flat_metrics["variability_chi_sq_red_g"])
