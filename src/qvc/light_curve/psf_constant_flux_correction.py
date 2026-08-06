#!/usr/bin/env python3
"""Attach spectra-informed fixed PSF dilution factors to light curves.

The light-curve likelihood consumes the central ``f_AGN_psf_<band>`` values as
fixed per-band factors. Fraction uncertainties are retained as provenance
metadata but are not sampled. The observations remain in their original
total-PSF-flux space, avoiding nonlinear pointwise subtraction and preserving
every epoch.
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

def normalize_object_id(value) -> str:
    """Return a normalized object-id string for joins."""
    return str(value).strip()


def load_spectra_psf_fractions(spectra_fit_csvs) -> dict[str, dict]:
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
            if col == "object_id" or col.startswith("f_AGN_psf_")
        ]
        if "object_id" not in usecols:
            raise ValueError(f"Spectra CSV {resolved} is missing required column 'object_id'.")
        fraction_cols = [col for col in usecols if col.startswith("f_AGN_psf_")]
        value_cols = [col for col in fraction_cols if not col.endswith("_err")]
        if not value_cols:
            raise ValueError(
                f"Spectra CSV {resolved} is missing required per-band columns 'f_AGN_psf_<band>'."
            )
        missing_errors = [f"{col}_err" for col in value_cols if f"{col}_err" not in usecols]
        if missing_errors:
            raise ValueError(
                f"Spectra CSV {resolved} is missing fraction uncertainty column(s): "
                + ", ".join(missing_errors)
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


def get_bandpass_agn_fraction(source: Mapping[str, object], band: str) -> tuple[float, float, str | None]:
    """Return the variable-AGN/total fraction and its uncertainty for one photometric band."""

    band_key = f"f_AGN_psf_{band}"
    band_err_key = f"{band_key}_err"
    val = source.get(band_key, np.nan)
    err = source.get(band_err_key, np.nan)
    if np.isfinite(val) and np.isfinite(err):
        val = float(val)
        if 0.0 < val <= 1.0:
            err = float(err)
            err = float(np.clip(err, 0.0, 1.0))
            return val, err, band_key

    return np.nan, np.nan, None


def apply_constant_flux_correction_to_object(
    obj: Mapping[str, object],
    *,
    bands: Iterable[str] | None = None,
):
    """Return a light-curve object carrying fixed per-band dilution factors."""

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
    for band in bands:
        if band not in corrected_obj["mags"] or band not in lambda_pivot:
            continue
        agn_fraction, agn_fraction_err, source_key = get_bandpass_agn_fraction(corrected_obj, str(band))
        if not np.isfinite(agn_fraction):
            n_missing_fraction_bands += 1
            continue
        summary = {
            "agn_fraction": agn_fraction,
            "agn_fraction_err": agn_fraction_err,
            "source_key": source_key,
            "n_points": int(np.sum(np.isfinite(corrected_obj["mags"][band]))),
        }
        band_summaries[str(band)] = summary
        n_corrected_bands += 1

    if n_corrected_bands > 0:
        band_order = list(corrected_obj["mags"].keys())
        if "psf_fraction_reference_mags_by_band" not in corrected_obj:
            means = list(corrected_obj.get("mags_mean", []))
            if len(means) != len(band_order):
                raise ValueError(
                    "PSF dilution requires the native light-curve mean "
                    "magnitude for every band."
                )
            corrected_obj["psf_fraction_reference_mags_by_band"] = dict(
                zip(band_order, means)
            )
        if "psf_fraction_reference_magerrs_by_band" not in corrected_obj:
            mean_errors = list(corrected_obj.get("mags_mean_err", []))
            corrected_obj["psf_fraction_reference_magerrs_by_band"] = {
                band: mean_errors[index] if index < len(mean_errors) else np.nan
                for index, band in enumerate(band_order)
            }

    corrected_obj["psf_constant_flux_n_bands_corrected"] = int(n_corrected_bands)
    corrected_obj["psf_constant_flux_corrected"] = bool(n_corrected_bands > 0)
    corrected_obj["psf_constant_flux_band_summaries"] = band_summaries

    object_summary = {
        "object_id": normalize_object_id(obj.get("object_id")),
        "n_corrected_bands": n_corrected_bands,
        "n_missing_fraction_bands": n_missing_fraction_bands,
    }
    return corrected_obj, object_summary


def apply_constant_flux_correction_to_objects(
    objs,
    *,
    spectra_fit_csvs,
    progress_bar: bool = False,
):
    """Attach spectra-informed fixed dilution factors to a list of objects."""

    spectra_rows = load_spectra_psf_fractions(spectra_fit_csvs)
    corrected_objs = []
    summary = {
        "n_objects": len(objs),
        "n_objects_with_spectra": 0,
        "n_objects_corrected": 0,
        "n_bands_corrected": 0,
        "n_missing_spectra": 0,
        "median_agn_fraction": np.nan,
        "per_band": {},
    }

    agn_fractions = []
    missing_spectra_object_ids = []
    zero_corrected_object_ids = []

    iterator = tqdm(objs, desc="Applying PSF flux correction", disable=not progress_bar)
    for obj in iterator:
        oid = normalize_object_id(obj.get("object_id"))
        spectra_row = spectra_rows.get(oid)
        if spectra_row is None:
            corrected_objs.append(obj)
            summary["n_missing_spectra"] += 1
            missing_spectra_object_ids.append(oid)
            continue

        merged = dict(obj)
        merged.update(spectra_row)
        corrected_obj, object_summary = apply_constant_flux_correction_to_object(
            merged,
        )
        corrected_objs.append(corrected_obj)
        summary["n_objects_with_spectra"] += 1
        if object_summary["n_corrected_bands"] > 0:
            summary["n_objects_corrected"] += 1
            summary["n_bands_corrected"] += int(object_summary["n_corrected_bands"])
            for band_name, band_summary in corrected_obj["psf_constant_flux_band_summaries"].items():
                band_name = str(band_name)
                if np.isfinite(band_summary["agn_fraction"]):
                    agn_fractions.append(float(band_summary["agn_fraction"]))
                stats = summary["per_band"].setdefault(
                    band_name,
                    {
                        "n_corrected": 0,
                        "agn_fractions": [],
                    },
                )
                stats["n_corrected"] += 1
                if np.isfinite(band_summary["agn_fraction"]):
                    stats["agn_fractions"].append(float(band_summary["agn_fraction"]))
        else:
            zero_corrected_object_ids.append(oid)

    if agn_fractions:
        summary["median_agn_fraction"] = float(np.nanmedian(np.asarray(agn_fractions, dtype=float)))
    for stats in summary["per_band"].values():
        agn_vals = np.asarray(stats.pop("agn_fractions"), dtype=float)
        stats["median_agn_fraction"] = float(np.nanmedian(agn_vals)) if agn_vals.size > 0 else np.nan

    if missing_spectra_object_ids or zero_corrected_object_ids:
        msg_parts = [
            "--subtract_psf_constant_flux requires every object to have at least one corrected band."
        ]
        if missing_spectra_object_ids:
            msg_parts.append(
                "Missing spectra rows for object_id(s): "
                + ", ".join(missing_spectra_object_ids)
            )
        if zero_corrected_object_ids:
            msg_parts.append(
                "No valid PSF constant-flux correction band for object_id(s): "
                + ", ".join(zero_corrected_object_ids)
            )
        raise ValueError(" ".join(msg_parts))

    return corrected_objs, summary


def print_constant_flux_correction_summary(summary):
    """Print a concise correction summary."""

    print("PSF constant-flux correction summary:")
    print(f"  objects total: {summary['n_objects']}")
    print(f"  objects with spectra fractions: {summary['n_objects_with_spectra']}")
    print(f"  objects corrected: {summary['n_objects_corrected']}")
    print(f"  bands corrected: {summary['n_bands_corrected']}")
    print(f"  objects missing spectra fractions: {summary['n_missing_spectra']}")
    print("  observations modified before fitting: 0")
    if np.isfinite(summary["median_agn_fraction"]):
        print(f"  median retained AGN fraction: {summary['median_agn_fraction']:.3f}")
    if summary["per_band"]:
        print("  per-band corrections:")
        for band in sorted(summary["per_band"]):
            stats = summary["per_band"][band]
            line = (
                f"    {band}: n={stats['n_corrected']}, "
                f"median variable-AGN fraction={stats['median_agn_fraction']:.3f}"
            )
            print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Attach spectra-driven fixed PSF dilution factors to light curves.",
    )
    parser.add_argument("--spectra_fit_csv", nargs="+", required=True, help="Spectra-fit CSV file(s).")
    parser.add_argument("--filter_object_id", nargs="+", default=None, help="Optional object IDs to inspect.")
    parser.add_argument("--N", type=int, default=None, help="Optional object count limit.")
    parser.add_argument("--skip", type=int, default=None, help="Optional number of objects to skip.")
    parser.add_argument("--progress", action="store_true", help="Show a progress bar.")
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
    )
    print_constant_flux_correction_summary(summary)
    print(f"Loaded {len(objs)} objects; attached fixed factors to {len(corrected)} objects.")


if __name__ == "__main__":
    main()
