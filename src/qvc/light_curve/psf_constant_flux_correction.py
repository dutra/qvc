#!/usr/bin/env python3
"""Apply spectra-informed constant-flux corrections to PSF light curves.

This module approximates AGN-only light curves by:

1. converting PSF magnitudes to relative flux,
2. estimating a constant contaminating flux from a spectra-derived PL/total
   bandpass fraction,
3. subtracting that constant contaminating flux in flux space, and
4. converting the corrected light curve back to magnitudes for the existing GP
   pipeline.

The required spectra inputs are per-bandpass PSF fractions saved as
``f_PL_psf_<band>``.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from tqdm import tqdm

from qvc.hubble.hubble_utils import resolve_qvc_data_path
from qvc.light_curve.multiband_generate_lc import (
    concat_light_curves,
    lambda_pivot,
    populate_sdss_fields,
)

MAG_TO_FLUX_DERIV = 0.4 * np.log(10.0)


def normalize_object_id(value) -> str:
    """Return a normalized object-id string for joins."""
    return str(value).strip()


def mag_to_relative_flux(mag):
    """Convert magnitudes to arbitrary relative flux units."""
    mag = np.asarray(mag, dtype=float)
    return 10.0 ** (-0.4 * mag)


def magerr_to_relative_fluxerr(mag, magerr):
    """Propagate magnitude errors to relative flux errors."""
    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)
    flux = mag_to_relative_flux(mag)
    return flux * MAG_TO_FLUX_DERIV * magerr


def relative_flux_to_mag(flux):
    """Convert arbitrary relative flux units back to magnitudes."""
    flux = np.asarray(flux, dtype=float)
    return -2.5 * np.log10(np.clip(flux, 1e-300, None))


def relative_fluxerr_to_magerr(flux, fluxerr):
    """Propagate relative-flux errors back to magnitude errors."""
    flux = np.asarray(flux, dtype=float)
    fluxerr = np.asarray(fluxerr, dtype=float)
    return (2.5 / np.log(10.0)) * fluxerr / np.clip(flux, 1e-300, None)


def load_spectra_pl_psf_fractions(spectra_fit_csvs) -> dict[str, dict]:
    """Load only the spectra columns needed for PSF constant-flux correction."""

    if not spectra_fit_csvs:
        return {}

    frames = []
    for csv_path in spectra_fit_csvs:
        resolved = resolve_qvc_data_path(csv_path)
        header = pd.read_csv(resolved, nrows=0)
        usecols = [
            col
            for col in header.columns
            if col == "object_id" or col.startswith("f_PL_psf_")
        ]
        if "object_id" not in usecols:
            raise ValueError(f"Spectra CSV {resolved} is missing required column 'object_id'.")
        if not any(col.startswith("f_PL_psf_") for col in usecols):
            raise ValueError(
                f"Spectra CSV {resolved} is missing required per-band columns 'f_PL_psf_<band>'."
            )
        frames.append(pd.read_csv(resolved, usecols=usecols))

    if not frames:
        return {}

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged["object_id_key"] = merged["object_id"].map(normalize_object_id)
    merged = merged.drop_duplicates("object_id_key", keep="last")
    return {
        row["object_id_key"]: row.drop(labels=["object_id_key"]).to_dict()
        for _, row in merged.iterrows()
    }


def get_bandpass_pl_fraction(source: Mapping[str, object], band: str) -> tuple[float, str | None]:
    """Return the preferred PL/total fraction for one photometric band."""

    band_key = f"f_PL_psf_{band}"
    val = source.get(band_key, np.nan)
    if np.isfinite(val):
        val = float(val)
        if 0.0 < val <= 1.0:
            return min(val, 0.999), band_key

    return np.nan, None


def subtract_constant_flux_from_band(
    mags,
    magerrs,
    pl_fraction,
    *,
    reference_stat: str = "median",
):
    """Return a PL-only magnitude light curve from total PSF magnitudes."""

    mags = np.asarray(mags, dtype=float)
    magerrs = np.asarray(magerrs, dtype=float)
    finite = np.isfinite(mags) & np.isfinite(magerrs)
    corrected_mags = mags.astype(float, copy=True)
    corrected_magerrs = magerrs.astype(float, copy=True)

    summary = {
        "pl_fraction": np.nan,
        "reference_total_flux": np.nan,
        "constant_contaminant_flux": np.nan,
        "n_points": int(np.sum(finite)),
        "n_nonpositive_after_subtraction": 0,
        "median_delta_mag": np.nan,
    }

    if (not np.isfinite(pl_fraction)) or pl_fraction <= 0.0 or np.sum(finite) == 0:
        return corrected_mags, corrected_magerrs, summary

    total_flux = mag_to_relative_flux(mags[finite])
    total_flux_err = magerr_to_relative_fluxerr(mags[finite], magerrs[finite])

    if reference_stat == "mean":
        reference_total_flux = float(np.nanmean(total_flux))
    else:
        reference_total_flux = float(np.nanmedian(total_flux))
    if (not np.isfinite(reference_total_flux)) or reference_total_flux <= 0.0:
        return corrected_mags, corrected_magerrs, summary

    contaminant_flux = (1.0 - float(pl_fraction)) * reference_total_flux
    pl_flux = total_flux - contaminant_flux
    valid = pl_flux > 0.0

    corrected = np.full(total_flux.shape, np.nan, dtype=float)
    corrected_err = np.full(total_flux_err.shape, np.nan, dtype=float)
    if np.any(valid):
        corrected[valid] = relative_flux_to_mag(pl_flux[valid])
        corrected_err[valid] = relative_fluxerr_to_magerr(pl_flux[valid], total_flux_err[valid])

    corrected_mags[finite] = corrected
    corrected_magerrs[finite] = corrected_err

    summary.update(
        {
            "pl_fraction": float(pl_fraction),
            "reference_total_flux": reference_total_flux,
            "constant_contaminant_flux": contaminant_flux,
            "n_nonpositive_after_subtraction": int(np.sum(~valid)),
            "median_delta_mag": float(np.nanmedian(corrected - mags[finite])) if np.any(valid) else np.nan,
        }
    )
    return corrected_mags, corrected_magerrs, summary


def apply_constant_flux_correction_to_object(
    obj: Mapping[str, object],
    *,
    bands: Iterable[str] | None = None,
    reference_stat: str = "median",
):
    """Return a corrected light-curve object plus a correction summary."""

    corrected_obj = dict(obj)
    corrected_obj["times"] = {
        band: np.asarray(values, dtype=float).copy()
        for band, values in obj["times"].items()
    }
    corrected_obj["mags"] = {
        band: np.asarray(values, dtype=float).copy()
        for band, values in obj["mags"].items()
    }
    corrected_obj["magerrs"] = {
        band: np.asarray(values, dtype=float).copy()
        for band, values in obj["magerrs"].items()
    }

    if bands is None:
        bands = list(corrected_obj["mags"].keys())

    band_summaries = {}
    n_corrected_bands = 0
    n_missing_fraction_bands = 0
    total_nonpositive = 0

    for band in bands:
        if band not in corrected_obj["mags"] or band not in lambda_pivot:
            continue
        pl_fraction, source_key = get_bandpass_pl_fraction(corrected_obj, str(band))
        if not np.isfinite(pl_fraction):
            n_missing_fraction_bands += 1
            continue
        mags_new, magerrs_new, summary = subtract_constant_flux_from_band(
            corrected_obj["mags"][band],
            corrected_obj["magerrs"][band],
            pl_fraction,
            reference_stat=reference_stat,
        )
        corrected_obj["mags"][band] = mags_new
        corrected_obj["magerrs"][band] = magerrs_new
        summary["source_key"] = source_key
        band_summaries[str(band)] = summary
        n_corrected_bands += 1
        total_nonpositive += int(summary["n_nonpositive_after_subtraction"])

    if "mags_mean" in corrected_obj:
        corrected_obj["mags_mean"] = [
            float(np.nanmean(corrected_obj["mags"][band]))
            if np.any(np.isfinite(corrected_obj["mags"][band]))
            else np.nan
            for band in corrected_obj["mags"].keys()
        ]

    corrected_obj["psf_constant_flux_corrected"] = bool(n_corrected_bands > 0)
    corrected_obj["psf_constant_flux_band_summaries"] = band_summaries

    object_summary = {
        "object_id": normalize_object_id(obj.get("object_id")),
        "n_corrected_bands": n_corrected_bands,
        "n_missing_fraction_bands": n_missing_fraction_bands,
        "n_nonpositive_after_subtraction": total_nonpositive,
    }
    return corrected_obj, object_summary


def apply_constant_flux_correction_to_objects(
    objs,
    *,
    spectra_fit_csvs,
    progress_bar: bool = False,
    reference_stat: str = "median",
):
    """Apply spectra-informed constant-flux subtraction to a list of objects."""

    spectra_rows = load_spectra_pl_psf_fractions(spectra_fit_csvs)
    corrected_objs = []
    summary = {
        "n_objects": len(objs),
        "n_objects_with_spectra": 0,
        "n_objects_corrected": 0,
        "n_bands_corrected": 0,
        "n_missing_spectra": 0,
        "n_nonpositive_after_subtraction": 0,
        "median_pl_fraction": np.nan,
        "median_abs_delta_mag": np.nan,
        "per_band": {},
    }

    pl_fractions = []
    delta_mags = []

    iterator = tqdm(objs, desc="Applying PSF flux correction", disable=not progress_bar)
    for obj in iterator:
        oid = normalize_object_id(obj.get("object_id"))
        spectra_row = spectra_rows.get(oid)
        if spectra_row is None:
            corrected_objs.append(obj)
            summary["n_missing_spectra"] += 1
            continue

        merged = dict(obj)
        merged.update(spectra_row)
        corrected_obj, object_summary = apply_constant_flux_correction_to_object(
            merged,
            reference_stat=reference_stat,
        )
        corrected_objs.append(corrected_obj)
        summary["n_objects_with_spectra"] += 1
        if object_summary["n_corrected_bands"] > 0:
            summary["n_objects_corrected"] += 1
            summary["n_bands_corrected"] += int(object_summary["n_corrected_bands"])
            summary["n_nonpositive_after_subtraction"] += int(
                object_summary["n_nonpositive_after_subtraction"]
            )
            for band_name, band_summary in corrected_obj["psf_constant_flux_band_summaries"].items():
                band_name = str(band_name)
                if np.isfinite(band_summary["pl_fraction"]):
                    pl_fractions.append(float(band_summary["pl_fraction"]))
                if np.isfinite(band_summary["median_delta_mag"]):
                    delta_mags.append(float(abs(band_summary["median_delta_mag"])))
                stats = summary["per_band"].setdefault(
                    band_name,
                    {
                        "n_corrected": 0,
                        "n_nonpositive_after_subtraction": 0,
                        "pl_fractions": [],
                        "abs_delta_mags": [],
                    },
                )
                stats["n_corrected"] += 1
                stats["n_nonpositive_after_subtraction"] += int(
                    band_summary["n_nonpositive_after_subtraction"]
                )
                if np.isfinite(band_summary["pl_fraction"]):
                    stats["pl_fractions"].append(float(band_summary["pl_fraction"]))
                if np.isfinite(band_summary["median_delta_mag"]):
                    stats["abs_delta_mags"].append(float(abs(band_summary["median_delta_mag"])))

    if pl_fractions:
        summary["median_pl_fraction"] = float(np.nanmedian(np.asarray(pl_fractions, dtype=float)))
    if delta_mags:
        summary["median_abs_delta_mag"] = float(np.nanmedian(np.asarray(delta_mags, dtype=float)))
    for stats in summary["per_band"].values():
        pl_vals = np.asarray(stats.pop("pl_fractions"), dtype=float)
        delta_vals = np.asarray(stats.pop("abs_delta_mags"), dtype=float)
        stats["median_pl_fraction"] = float(np.nanmedian(pl_vals)) if pl_vals.size > 0 else np.nan
        stats["median_abs_delta_mag"] = (
            float(np.nanmedian(delta_vals)) if delta_vals.size > 0 else np.nan
        )
    return corrected_objs, summary


def print_constant_flux_correction_summary(summary):
    """Print a concise correction summary."""

    print("PSF constant-flux correction summary:")
    print(f"  objects total: {summary['n_objects']}")
    print(f"  objects with spectra fractions: {summary['n_objects_with_spectra']}")
    print(f"  objects corrected: {summary['n_objects_corrected']}")
    print(f"  bands corrected: {summary['n_bands_corrected']}")
    print(f"  objects missing spectra fractions: {summary['n_missing_spectra']}")
    print(
        "  points driven non-positive after subtraction: "
        f"{summary['n_nonpositive_after_subtraction']}"
    )
    if np.isfinite(summary["median_pl_fraction"]):
        print(f"  median f_PL_psf: {summary['median_pl_fraction']:.3f}")
    if np.isfinite(summary["median_abs_delta_mag"]):
        print(f"  median |Δmag| after correction: {summary['median_abs_delta_mag']:.3f}")
    if summary["per_band"]:
        print("  per-band corrections:")
        for band in sorted(summary["per_band"]):
            stats = summary["per_band"][band]
            line = (
                f"    {band}: n={stats['n_corrected']}, "
                f"median f_PL_psf={stats['median_pl_fraction']:.3f}, "
                f"median |Δmag|={stats['median_abs_delta_mag']:.3f}, "
                f"nonpositive={stats['n_nonpositive_after_subtraction']}"
            )
            print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Apply spectra-driven constant-flux subtraction to PSF light curves.",
    )
    parser.add_argument("--spectra_fit_csv", nargs="+", required=True, help="Spectra-fit CSV file(s).")
    parser.add_argument("--filter_object_id", nargs="+", default=None, help="Optional object IDs to inspect.")
    parser.add_argument("--N", type=int, default=None, help="Optional object count limit.")
    parser.add_argument("--skip", type=int, default=None, help="Optional number of objects to skip.")
    parser.add_argument("--progress", action="store_true", help="Show a progress bar.")
    parser.add_argument(
        "--reference_stat",
        choices=("median", "mean"),
        default="median",
        help="Statistic used to anchor the constant contaminating flux.",
    )
    args = parser.parse_args()

    objs = concat_light_curves(
        filter_object_ids=args.filter_object_id,
        progress_bar=args.progress,
        skip=args.skip,
        N=args.N,
    )
    objs = populate_sdss_fields(objs, progress_bar=args.progress)
    corrected, summary = apply_constant_flux_correction_to_objects(
        objs,
        spectra_fit_csvs=args.spectra_fit_csv,
        progress_bar=args.progress,
        reference_stat=args.reference_stat,
    )
    print_constant_flux_correction_summary(summary)
    print(f"Loaded {len(objs)} objects, corrected {len(corrected)} objects.")


if __name__ == "__main__":
    main()
