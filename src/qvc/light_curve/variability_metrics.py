"""Corrected per-band variability metrics for cleaned light curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.stats import chi2


DEFAULT_BANDS = ("u", "g", "r", "i", "z", "y")


@dataclass(frozen=True)
class BinnedChiSquaredResult:
    """Binned chi-squared summary for one band."""

    chi_sq_red: float
    chi_sq: float
    n_valid_bins: int
    bin_means: np.ndarray
    bin_errors: np.ndarray


def gap_aligned_bins(times, bin_width=365.0, gap_threshold=180.0):
    """Build annual bins, restarting after the first large seasonal gap."""

    times = np.sort(np.asarray(times, dtype=float))
    if len(times) == 0:
        return np.asarray([0.0, float(bin_width)], dtype=float)

    gaps = np.diff(times)
    large_gaps = np.where(gaps > gap_threshold)[0]

    bin_edges = []
    if len(large_gaps) > 0:
        first_cluster_end = times[large_gaps[0]]
        start_time = times[0] - bin_width / 2.0
        while start_time < first_cluster_end:
            bin_edges.append(start_time)
            start_time += bin_width
        start_time = times[large_gaps[0] + 1] - bin_width / 2.0
    else:
        start_time = times[0] - bin_width / 2.0

    while start_time <= times[-1] + bin_width:
        bin_edges.append(start_time)
        start_time += bin_width

    return np.asarray(bin_edges, dtype=float)


def compute_binned_chi_squared(times, mags, mag_errs, bin_edges):
    """Compute corrected raw/reduced chi-squared for one binned light curve."""

    times = np.asarray(times, dtype=float)
    mags = np.asarray(mags, dtype=float)
    mag_errs = np.asarray(mag_errs, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)

    if len(times) == 0:
        return BinnedChiSquaredResult(np.nan, np.nan, 0, np.asarray([]), np.asarray([]))

    good = np.isfinite(times) & np.isfinite(mags) & np.isfinite(mag_errs) & (mag_errs > 0)
    times = times[good]
    mags = mags[good]
    mag_errs = mag_errs[good]

    if len(times) == 0 or len(bin_edges) < 2:
        return BinnedChiSquaredResult(np.nan, np.nan, 0, np.asarray([]), np.asarray([]))

    bin_indices = np.digitize(times, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)

    n_bins = len(bin_edges) - 1
    bin_means = np.full(n_bins, np.nan)
    bin_errors = np.full(n_bins, np.nan)

    for idx in range(n_bins):
        mask = bin_indices == idx
        if np.sum(mask) == 0:
            continue
        weights = 1.0 / np.square(mag_errs[mask])
        weights_sum = np.sum(weights)
        if weights_sum <= 0:
            continue
        bin_means[idx] = np.sum(mags[mask] * weights) / weights_sum
        bin_errors[idx] = np.sqrt(1.0 / weights_sum)

    valid_bins = np.isfinite(bin_means) & np.isfinite(bin_errors) & (bin_errors > 0)
    n_valid_bins = int(np.sum(valid_bins))
    if n_valid_bins < 2:
        return BinnedChiSquaredResult(np.nan, np.nan, n_valid_bins, bin_means, bin_errors)

    weighted_mean = np.average(bin_means[valid_bins], weights=1.0 / np.square(bin_errors[valid_bins]))
    chi_sq = float(np.sum(np.square((bin_means[valid_bins] - weighted_mean) / bin_errors[valid_bins])))
    dof = n_valid_bins - 1
    chi_sq_red = chi_sq / dof
    return BinnedChiSquaredResult(chi_sq_red, chi_sq, n_valid_bins, bin_means, bin_errors)


def reconstruct_cleaned_by_band(lc):
    """Reconstruct per-band arrays from a cleaned `make_lc` payload."""

    times_rel, band_idx = lc["X"]
    times = np.asarray(times_rel, dtype=float) + float(lc["time0"])
    mags = np.asarray(lc["y"], dtype=float)
    errs = np.asarray(lc["yerr"], dtype=float)
    band_idx = np.asarray(band_idx)
    bands = list(lc["bands"])

    times_dict = {}
    mags_dict = {}
    errs_dict = {}
    for index, band in enumerate(bands):
        mask = band_idx == index
        times_dict[band] = times[mask]
        mags_dict[band] = mags[mask]
        errs_dict[band] = errs[mask]
    return times_dict, mags_dict, errs_dict


def compute_variability_metrics_for_bands(times_by_band, mags_by_band, errs_by_band, *, bands=None):
    """Compute corrected per-band variability metrics."""

    if bands is None:
        ordered = []
        seen = set()
        for band in list(DEFAULT_BANDS) + list(times_by_band.keys()):
            if band in seen:
                continue
            seen.add(band)
            ordered.append(band)
        bands = ordered

    metrics = {}
    for band in bands:
        times = np.asarray(times_by_band.get(band, []), dtype=float)
        mags = np.asarray(mags_by_band.get(band, []), dtype=float)
        errs = np.asarray(errs_by_band.get(band, []), dtype=float)

        good = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs) & (errs > 0)
        times = times[good]
        mags = mags[good]
        errs = errs[good]

        metrics[f"variability_n_points_{band}"] = int(len(times))
        metrics[f"variability_n_valid_bins_{band}"] = 0
        metrics[f"variability_dof_{band}"] = np.nan
        metrics[f"variability_chi_sq_{band}"] = np.nan
        metrics[f"variability_chi_sq_red_{band}"] = np.nan
        metrics[f"variability_pvalue_{band}"] = np.nan
        metrics[f"variability_neg_log10_pvalue_{band}"] = np.nan

        if len(times) < 2:
            continue

        chi_sq_result = compute_binned_chi_squared(times, mags, errs, gap_aligned_bins(times))
        dof = chi_sq_result.n_valid_bins - 1
        metrics[f"variability_n_valid_bins_{band}"] = int(chi_sq_result.n_valid_bins)
        if dof < 1 or not np.isfinite(chi_sq_result.chi_sq) or not np.isfinite(chi_sq_result.chi_sq_red):
            continue

        pvalue = float(chi2.sf(chi_sq_result.chi_sq, dof))
        metrics[f"variability_dof_{band}"] = float(dof)
        metrics[f"variability_chi_sq_{band}"] = float(chi_sq_result.chi_sq)
        metrics[f"variability_chi_sq_red_{band}"] = float(chi_sq_result.chi_sq_red)
        metrics[f"variability_pvalue_{band}"] = pvalue
        if np.isfinite(pvalue) and pvalue > 0.0:
            metrics[f"variability_neg_log10_pvalue_{band}"] = float(-np.log10(pvalue))

    return metrics


def compute_variability_metrics_for_cleaned_lc(lc):
    """Compute corrected per-band variability metrics from a cleaned light-curve payload."""

    times_by_band, mags_by_band, errs_by_band = reconstruct_cleaned_by_band(lc)
    return compute_variability_metrics_for_bands(
        times_by_band,
        mags_by_band,
        errs_by_band,
        bands=list(lc.get("bands", [])),
    )


def _extract_numbered_series(row: Mapping[str, object], prefix: str):
    values = []
    index = 0
    while True:
        key = f"{prefix}_{index}"
        if key not in row:
            break
        value = row.get(key)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(np.nan)
        index += 1
    return np.asarray(values, dtype=float)


def _normalize_bands_value(value):
    if value is None:
        return []
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return []
        if "," in txt:
            return [part.strip() for part in txt.split(",") if part.strip()]
        return [txt] if len(txt) > 1 and txt not in DEFAULT_BANDS else [txt] if txt else []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def infer_bands_from_flat_row(row):
    """Infer available band labels from a flattened row."""

    bands = []
    seen = set()
    for band in _normalize_bands_value(row.get("bands")):
        if band and band not in seen:
            seen.add(band)
            bands.append(band)
    for band in DEFAULT_BANDS:
        if any(f"{prefix}_{band}_" in key for prefix in ("times", "mags", "magerrs") for key in row.keys()):
            if band not in seen:
                seen.add(band)
                bands.append(band)
    return bands


def compute_variability_metrics_for_flat_row(row, *, bands=None):
    """Compute per-band variability metrics from flattened row columns when available."""

    if bands is None:
        bands = infer_bands_from_flat_row(row)

    times_by_band = {}
    mags_by_band = {}
    errs_by_band = {}
    for band in bands:
        times_by_band[band] = _extract_numbered_series(row, f"times_{band}")
        mags_by_band[band] = _extract_numbered_series(row, f"mags_{band}")
        errs_by_band[band] = _extract_numbered_series(row, f"magerrs_{band}")

    return compute_variability_metrics_for_bands(
        times_by_band,
        mags_by_band,
        errs_by_band,
        bands=bands,
    )
