#!/usr/bin/env python3
"""Apply spectra-informed fixed PSF dilution factors to light curves.

The relative-flux Erlang likelihood consumes the central
``f_AGN_psf_<band>`` values as fixed per-band factors while leaving observations
unchanged. The legacy additive-magnitude model instead subtracts the implied
constant contaminating flux before fitting. Fraction uncertainties are retained
as provenance metadata but are not sampled.
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


def mag_to_relative_flux(mag):
    """Convert magnitudes to arbitrary relative-flux units."""

    return 10.0 ** (-0.4 * np.asarray(mag, dtype=float))


def magerr_to_relative_fluxerr(mag, magerr):
    """Propagate magnitude errors to relative-flux errors."""

    flux = mag_to_relative_flux(mag)
    return flux * MAG_TO_FLUX_DERIV * np.asarray(magerr, dtype=float)


def relative_flux_to_mag(flux):
    """Convert positive arbitrary relative flux to magnitudes."""

    return -2.5 * np.log10(np.asarray(flux, dtype=float))


def relative_fluxerr_to_magerr(flux, fluxerr):
    """Propagate relative-flux errors to magnitude errors."""

    flux = np.asarray(flux, dtype=float)
    fluxerr = np.asarray(fluxerr, dtype=float)
    return (2.5 / np.log(10.0)) * fluxerr / flux


def subtract_constant_flux_from_band(
    mags,
    magerrs,
    agn_fraction,
    *,
    reference_mag,
    agn_fraction_err=0.0,
):
    """Return AGN-only magnitudes after subtracting fixed contaminating flux.

    This restores the pre-likelihood correction used by the historical
    additive-magnitude model. ``reference_mag`` is the native total-PSF
    magnitude at which ``agn_fraction`` is defined. Fraction uncertainty is a
    shared systematic and is retained only as provenance metadata.
    """

    mags = np.asarray(mags, dtype=float)
    magerrs = np.asarray(magerrs, dtype=float)
    finite = np.isfinite(mags) & np.isfinite(magerrs) & (magerrs >= 0.0)
    corrected_mags = np.full_like(mags, np.nan, dtype=float)
    corrected_magerrs = np.full_like(magerrs, np.nan, dtype=float)
    summary = {
        "agn_fraction": float(agn_fraction),
        "agn_fraction_err": float(agn_fraction_err),
        "reference_total_mag": float(reference_mag),
        "reference_total_flux": np.nan,
        "reference_agn_mag": np.nan,
        "constant_contaminant_flux": np.nan,
        "n_points": int(np.sum(finite)),
        "n_nonpositive_after_subtraction": 0,
        "median_delta_mag": np.nan,
    }

    if (
        not np.isfinite(agn_fraction)
        or not 0.0 < float(agn_fraction) <= 1.0
        or not np.isfinite(reference_mag)
        or not np.any(finite)
    ):
        return corrected_mags, corrected_magerrs, summary

    reference_total_flux = float(mag_to_relative_flux(reference_mag))
    constant_flux = (1.0 - float(agn_fraction)) * reference_total_flux
    total_flux = mag_to_relative_flux(mags[finite])
    total_flux_err = magerr_to_relative_fluxerr(mags[finite], magerrs[finite])
    agn_flux = total_flux - constant_flux
    positive = agn_flux > 0.0

    corrected = np.full_like(total_flux, np.nan, dtype=float)
    corrected_err = np.full_like(total_flux_err, np.nan, dtype=float)
    corrected[positive] = relative_flux_to_mag(agn_flux[positive])
    corrected_err[positive] = relative_fluxerr_to_magerr(
        agn_flux[positive], total_flux_err[positive]
    )
    corrected_mags[finite] = corrected
    corrected_magerrs[finite] = corrected_err

    reference_agn_flux = float(agn_fraction) * reference_total_flux
    summary.update(
        {
            "reference_total_flux": reference_total_flux,
            "reference_agn_mag": float(relative_flux_to_mag(reference_agn_flux)),
            "constant_contaminant_flux": constant_flux,
            "n_nonpositive_after_subtraction": int(np.sum(~positive)),
            "median_delta_mag": (
                float(np.nanmedian(corrected - mags[finite]))
                if np.any(positive)
                else np.nan
            ),
        }
    )
    return corrected_mags, corrected_magerrs, summary

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
    subtract_observations: bool = False,
):
    """Attach dilution factors and optionally subtract constant flux."""

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

    band_order = list(corrected_obj["mags"].keys())

    def ensure_reference_metadata():
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

    if subtract_observations:
        ensure_reference_metadata()

    band_summaries = {}
    n_corrected_bands = 0
    n_missing_fraction_bands = 0
    n_nonpositive_after_subtraction = 0
    corrected_reference_mags = {}
    corrected_reference_magerrs = {}
    for band in bands:
        if band not in corrected_obj["mags"] or band not in lambda_pivot:
            continue
        agn_fraction, agn_fraction_err, source_key = get_bandpass_agn_fraction(corrected_obj, str(band))
        if not np.isfinite(agn_fraction):
            n_missing_fraction_bands += 1
            continue
        if subtract_observations:
            reference_mag = corrected_obj["psf_fraction_reference_mags_by_band"].get(
                str(band), np.nan
            )
            mags_new, magerrs_new, summary = subtract_constant_flux_from_band(
                corrected_obj["mags"][band],
                corrected_obj["magerrs"][band],
                agn_fraction,
                reference_mag=reference_mag,
                agn_fraction_err=agn_fraction_err,
            )
            corrected_obj["mags"][band] = mags_new
            corrected_obj["magerrs"][band] = magerrs_new
            corrected_reference_mags[str(band)] = summary["reference_agn_mag"]
            reference_magerr = corrected_obj[
                "psf_fraction_reference_magerrs_by_band"
            ].get(str(band), np.nan)
            corrected_reference_magerrs[str(band)] = (
                float(reference_magerr) / float(agn_fraction)
                if np.isfinite(reference_magerr)
                else np.nan
            )
            n_nonpositive_after_subtraction += int(
                summary["n_nonpositive_after_subtraction"]
            )
        else:
            summary = {
                "agn_fraction": agn_fraction,
                "agn_fraction_err": agn_fraction_err,
                "n_points": int(np.sum(np.isfinite(corrected_obj["mags"][band]))),
            }
        summary["source_key"] = source_key
        band_summaries[str(band)] = summary
        n_corrected_bands += 1

    if n_corrected_bands > 0 and not subtract_observations:
        ensure_reference_metadata()
    if subtract_observations and n_corrected_bands > 0:
        corrected_obj["psf_corrected_reference_mags_by_band"] = corrected_reference_mags
        corrected_obj[
            "psf_corrected_reference_magerrs_by_band"
        ] = corrected_reference_magerrs
        if "mags_mean" in corrected_obj:
            corrected_obj["mags_mean"] = [
                corrected_reference_mags.get(band, np.nan) for band in band_order
            ]
        if "mags_mean_err" in corrected_obj:
            corrected_obj["mags_mean_err"] = [
                corrected_reference_magerrs.get(band, np.nan) for band in band_order
            ]

    corrected_obj["psf_constant_flux_n_bands_corrected"] = int(n_corrected_bands)
    corrected_obj["psf_constant_flux_corrected"] = bool(n_corrected_bands > 0)
    corrected_obj["psf_constant_flux_mode"] = (
        "subtracted" if subtract_observations else "likelihood_dilution"
    )
    corrected_obj["psf_constant_flux_band_summaries"] = band_summaries

    object_summary = {
        "object_id": normalize_object_id(obj.get("object_id")),
        "n_corrected_bands": n_corrected_bands,
        "n_missing_fraction_bands": n_missing_fraction_bands,
        "n_nonpositive_after_subtraction": n_nonpositive_after_subtraction,
    }
    return corrected_obj, object_summary


def apply_constant_flux_correction_to_objects(
    objs,
    *,
    spectra_fit_csvs,
    progress_bar: bool = False,
    subtract_observations: bool = False,
):
    """Apply the selected fixed-dilution treatment to a list of objects."""

    spectra_rows = load_spectra_psf_fractions(spectra_fit_csvs)
    corrected_objs = []
    summary = {
        "n_objects": len(objs),
        "n_objects_with_spectra": 0,
        "n_objects_corrected": 0,
        "n_bands_corrected": 0,
        "n_missing_spectra": 0,
        "n_nonpositive_after_subtraction": 0,
        "median_agn_fraction": np.nan,
        "median_abs_delta_mag": np.nan,
        "correction_mode": (
            "subtracted" if subtract_observations else "likelihood_dilution"
        ),
        "per_band": {},
    }

    agn_fractions = []
    delta_mags = []
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
            subtract_observations=subtract_observations,
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
                if np.isfinite(band_summary["agn_fraction"]):
                    agn_fractions.append(float(band_summary["agn_fraction"]))
                if np.isfinite(band_summary.get("median_delta_mag", np.nan)):
                    delta_mags.append(float(abs(band_summary["median_delta_mag"])))
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
    if delta_mags:
        summary["median_abs_delta_mag"] = float(
            np.nanmedian(np.asarray(delta_mags, dtype=float))
        )
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
    print(f"  correction mode: {summary['correction_mode']}")
    print(
        "  observations modified before fitting: "
        f"{'yes' if summary['correction_mode'] == 'subtracted' else 'no'}"
    )
    if summary["correction_mode"] == "subtracted":
        print(
            "  points driven non-positive after subtraction: "
            f"{summary['n_nonpositive_after_subtraction']}"
        )
    if np.isfinite(summary["median_agn_fraction"]):
        print(f"  median retained AGN fraction: {summary['median_agn_fraction']:.3f}")
    if np.isfinite(summary["median_abs_delta_mag"]):
        print(
            "  median |delta mag| after correction: "
            f"{summary['median_abs_delta_mag']:.3f}"
        )
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
