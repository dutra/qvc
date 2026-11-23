"""Versioned minimal HDF5 inputs for the fiducial Hubble-paper workflow."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


LIGHT_CURVE_FORMAT = "qvc_hubble_paper_light_curve_v1"
# v2: carries the reference chi-square columns and the explicit per-row
# ``spectral_noise_model`` marker. v1 files are rejected: written before the
# marker existed, they cannot tell a fitted-noise catalogue from a fixed-floor
# one, so the Tier-1 chi-square cuts could silently use the wrong statistic.
SPECTRA_FORMAT = "qvc_hubble_paper_spectra_v2"
PREVIOUS_SPECTRA_FORMATS = ("qvc_hubble_paper_spectra_v1",)
NOISE_MODEL_FIELD = "spectral_noise_model"
FITTED_NOISE_SOURCE_FIELDS = ("frac_jitter", "add_jitter", "frac_fe_jitter")
FORMAT_ATTR = "qvc_hubble_paper_input_format"
MANIFEST_ATTR = "field_manifest_json"

_BANDS = ("u", "g", "r", "i")
_JITTER_SURVEYS = ("ps1", "sdss", "ztf")

LIGHT_CURVE_FIELDS = (
    "object_id",
    "sdss_name",
    "ra",
    "dec",
    "z",
    "z_err",
    *(f"mags_mean_{band}" for band in _BANDS),
    *(f"dlog_amp_blr_{band}" for band in _BANDS),
    *(f"dlog_amp_blr_{band}_err" for band in _BANDS),
    *(
        f"log_jitter_{band}_{survey}"
        for band in _BANDS
        for survey in _JITTER_SURVEYS
    ),
    "log_sigma_uv",
    "log_sigma_uv_err",
    "log_sigma_uv_std_psd",
    "log_tau_uv_rf",
    "log_tau_uv_rf_err",
    "log_tau_uv_rf_std_psd",
    "log_sigma_uv_log_tau_uv_rf_cov_psd",
    "log_sigma_uv_rhat",
    "log_tau_uv_rf_rhat",
    "loo_chi2_eff",
    "num_divergences",
    "t_rf_length",
    "SN_MEDIAN_ALL",
    "variability_chi_sq_red_g",
    "eta_sigma",
    "eta_sigma_err",
    *(f"log_sigma_band_{band}" for band in _BANDS),
    *(f"log_sigma_band_{band}_err" for band in _BANDS),
    "tau_fast_driver",
    "tau_slow_driver",
    "lag0",
    "lambda_center_rf",
    *(f"lag_disk_{band}" for band in _BANDS),
    "log_sigma_ls",
    "log_sigma_ls_err",
    "log_tau_ls",
    "log_tau_ls_err",
    "alpha_high_ls",
    "alpha_high_ls_err",
    "psd_ls_valid",
    "psd_ls_fmax",
    "log_sigma_ls_fixed",
    "log_sigma_ls_fixed_err",
    "log_tau_ls_fixed",
    "log_tau_ls_fixed_err",
    "psd_ls_fixed_valid",
    "psd_ls_fixed_fmax",
    "psd_bpl_ref_band",
    "psd_bpl_ref_lambda_rf",
    *(f"log_tau_band_{band}_RF" for band in _BANDS),
    *(f"log_tau_band_{band}_RF_err" for band in _BANDS),
    *(f"log_sigma_total_rms_band_{band}" for band in _BANDS),
    *(f"log_sigma_total_rms_band_{band}_err" for band in _BANDS),
)

SPECTRA_FIELDS = (
    "object_id",
    "fit_ok",
    "fit_backend",
    "m_2500_dereddened",
    "m_2500_dereddened_err",
    "m_2500_attenuated_model",
    "m_2500_attenuated_model_err",
    "pl_slope",
    "pl_slope_err",
    "sed_reduced_chi2",
    "spectroscopy_reduced_chi2",
    "joint_reduced_chi2",
    "m_2500_dereddened_rhat",
    "m_2500_attenuated_model_rhat",
    # Reference chi-square and the fitted-noise marker used to choose the
    # Tier-1 chi-square columns (see qvc.hubble.cuts.resolve_chi2_cut_columns).
    "sed_reduced_chi2_ref",
    "spectroscopy_reduced_chi2_ref",
    "joint_reduced_chi2_ref",
    "reference_spectrum_floor",
    "reference_photometry_floor",
    "reference_agn_floor",
    "frac_jitter",
    "spectrum_snr_median",
)
OPTIONAL_SPECTRA_FIELDS = (
    "pl_slope",
    "pl_slope_err",
    "sed_reduced_chi2_ref",
    "spectroscopy_reduced_chi2_ref",
    "joint_reduced_chi2_ref",
    "reference_spectrum_floor",
    "reference_photometry_floor",
    "reference_agn_floor",
    "frac_jitter",
    "spectrum_snr_median",
)
REQUIRED_SPECTRA_FIELDS = tuple(
    field for field in SPECTRA_FIELDS if field not in OPTIONAL_SPECTRA_FIELDS
)


def _decode(values):
    values = np.asarray(values)
    if values.dtype.kind == "S":
        return values.astype(str)
    if values.dtype.kind == "O":
        return np.asarray(
            [
                value.decode("utf-8", errors="replace")
                if isinstance(value, bytes)
                else value
                for value in values
            ],
            dtype=object,
        )
    return values


def _normalized_ids(values):
    return np.asarray([str(value).strip() for value in _decode(values)], dtype=object)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(dataset, indices):
    # Loading before arbitrary reordering avoids h5py's increasing-index rule.
    return np.asarray(dataset[...])[np.asarray(indices, dtype=np.int64)]


def _write_dataset(group, name, values):
    values = np.asarray(values)
    if values.dtype.kind in {"O", "S", "U"}:
        strings = [
            value.decode("utf-8", errors="replace")
            if isinstance(value, bytes)
            else str(value)
            for value in values
        ]
        group.create_dataset(
            name,
            data=strings,
            dtype=h5py.string_dtype("utf-8"),
            compression="gzip",
            shuffle=True,
        )
    else:
        group.create_dataset(
            name,
            data=values,
            compression="gzip",
            shuffle=True,
        )


def _validate_source_fields(group, fields, label):
    missing = sorted(set(fields).difference(group.keys()))
    if missing:
        raise ValueError(f"{label} source is missing required fields: {missing}")


def _set_provenance(handle, *, catalog_format, fields, source_path, source_rows):
    source_path = Path(source_path).resolve()
    stat = source_path.stat()
    handle.attrs[FORMAT_ATTR] = catalog_format
    handle.attrs[MANIFEST_ATTR] = json.dumps(list(fields), separators=(",", ":"))
    handle.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
    handle.attrs["source_path"] = str(source_path)
    handle.attrs["source_sha256"] = _sha256(source_path)
    handle.attrs["source_size_bytes"] = stat.st_size
    handle.attrs["source_mtime_ns"] = stat.st_mtime_ns
    handle.attrs["source_row_count"] = int(source_rows)
    handle.attrs["row_count"] = len(handle["object_id"] if "object_id" in handle else handle["catalog/object_id"])
    repo = None
    for parent in (source_path.parent, *source_path.parents):
        probe = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=parent,
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            repo = Path(probe.stdout.strip())
            break
    if repo is not None:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        handle.attrs["source_git_revision"] = revision
        handle.attrs["source_git_tracked_dirty"] = bool(dirty.strip())


def export_minimal_hubble_inputs(
    light_curve_source,
    spectra_source,
    light_curve_output,
    spectra_output,
):
    """Export matched successful rows for the current Hubble-paper profile."""

    light_curve_source = Path(light_curve_source)
    spectra_source = Path(spectra_source)
    light_curve_output = Path(light_curve_output)
    spectra_output = Path(spectra_output)

    with h5py.File(light_curve_source, "r") as light, h5py.File(
        spectra_source, "r"
    ) as spectra:
        if "catalog" not in spectra:
            raise ValueError("Spectra source has no /catalog group.")
        spectra_catalog = spectra["catalog"]
        _validate_source_fields(light, LIGHT_CURVE_FIELDS, "Light-curve")
        _validate_source_fields(spectra_catalog, REQUIRED_SPECTRA_FIELDS, "Spectra")
        spectra_fields = tuple(
            field for field in SPECTRA_FIELDS if field in spectra_catalog
        )

        light_ids = _normalized_ids(light["object_id"][...])
        spectra_ids = _normalized_ids(spectra_catalog["object_id"][...])
        if len(set(light_ids)) != len(light_ids):
            raise ValueError("Light-curve source contains duplicate object_id values.")
        if len(set(spectra_ids)) != len(spectra_ids):
            raise ValueError("Spectra source contains duplicate object_id values.")

        fit_ok_raw = _decode(spectra_catalog["fit_ok"][...])
        fit_ok = (
            fit_ok_raw.astype(bool)
            if fit_ok_raw.dtype.kind == "b"
            else np.asarray(
                [str(value).strip().lower() == "true" for value in fit_ok_raw]
            )
        )
        successful = {
            object_id: index
            for index, object_id in enumerate(spectra_ids)
            if fit_ok[index]
        }
        light_indices = np.asarray(
            [index for index, object_id in enumerate(light_ids) if object_id in successful],
            dtype=np.int64,
        )
        matched_ids = light_ids[light_indices]
        spectra_indices = np.asarray(
            [successful[object_id] for object_id in matched_ids], dtype=np.int64
        )
        if not len(matched_ids):
            raise ValueError("No successful spectra rows match the light-curve source.")

        light_curve_output.parent.mkdir(parents=True, exist_ok=True)
        spectra_output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(light_curve_output, "w") as output:
            for field in LIGHT_CURVE_FIELDS:
                _write_dataset(output, field, _read_rows(light[field], light_indices))
            _set_provenance(
                output,
                catalog_format=LIGHT_CURVE_FORMAT,
                fields=LIGHT_CURVE_FIELDS,
                source_path=light_curve_source,
                source_rows=len(light_ids),
            )

        with h5py.File(spectra_output, "w") as output:
            catalog = output.create_group("catalog")
            for field in spectra_fields:
                _write_dataset(
                    catalog,
                    field,
                    _read_rows(spectra_catalog[field], spectra_indices),
                )
            noise_model = (
                "fitted_jitter"
                if any(field in spectra_catalog for field in FITTED_NOISE_SOURCE_FIELDS)
                else "fixed_floor"
            )
            _write_dataset(
                catalog,
                NOISE_MODEL_FIELD,
                np.asarray([noise_model] * len(spectra_indices), dtype=object),
            )
            _set_provenance(
                output,
                catalog_format=SPECTRA_FORMAT,
                fields=(*spectra_fields, NOISE_MODEL_FIELD),
                source_path=spectra_source,
                source_rows=len(spectra_ids),
            )

    return len(matched_ids)


def read_minimal_spectra_hdf5(path):
    """Read and strictly validate a minimal Hubble-paper spectra catalog."""

    with h5py.File(path, "r") as handle:
        actual_format = str(handle.attrs.get(FORMAT_ATTR, ""))
        if actual_format in PREVIOUS_SPECTRA_FORMATS:
            raise ValueError(
                f"Minimal spectra file {path} uses {actual_format!r}, written before "
                "the spectral noise-model marker; it cannot tell whether its "
                "chi-square uses a fitted spectral jitter. Re-export it from the "
                f"spectra catalogue ({SPECTRA_FORMAT!r})."
            )
        if actual_format != SPECTRA_FORMAT:
            raise ValueError(
                f"Unsupported minimal spectra format {actual_format!r}; "
                f"expected {SPECTRA_FORMAT!r}."
            )
        if "catalog" not in handle:
            raise ValueError("Minimal spectra HDF5 has no /catalog group.")
        catalog = handle["catalog"]
        actual_fields = tuple(catalog.keys())
        required = (*REQUIRED_SPECTRA_FIELDS, NOISE_MODEL_FIELD)
        allowed = (*SPECTRA_FIELDS, NOISE_MODEL_FIELD)
        if not set(required).issubset(actual_fields) or not set(actual_fields).issubset(allowed):
            missing = sorted(set(required) - set(actual_fields))
            extra = sorted(set(actual_fields) - set(allowed))
            raise ValueError(
                f"Minimal spectra field manifest mismatch; missing={missing}, extra={extra}."
            )
        encoded_manifest = json.loads(str(handle.attrs.get(MANIFEST_ATTR, "[]")))
        expected_manifest = [field for field in SPECTRA_FIELDS if field in catalog] + [
            NOISE_MODEL_FIELD
        ]
        if encoded_manifest != expected_manifest:
            raise ValueError("Minimal spectra manifest attribute is invalid.")
        values = {name: _decode(catalog[name][...]) for name in expected_manifest}

    frame = pd.DataFrame(values)
    frame["object_id"] = frame["object_id"].astype(str).str.strip()
    if frame["object_id"].duplicated().any():
        raise ValueError("Minimal spectra HDF5 contains duplicate object_id values.")
    return frame


def validate_minimal_light_curve_hdf5(handle):
    """Validate an already-open minimal light-curve HDF5 catalog."""

    actual_format = str(handle.attrs.get(FORMAT_ATTR, ""))
    if actual_format != LIGHT_CURVE_FORMAT:
        raise ValueError(
            f"Unsupported minimal light-curve format {actual_format!r}; "
            f"expected {LIGHT_CURVE_FORMAT!r}."
        )
    actual_fields = tuple(
        name for name, value in handle.items() if isinstance(value, h5py.Dataset)
    )
    if set(actual_fields) != set(LIGHT_CURVE_FIELDS):
        missing = sorted(set(LIGHT_CURVE_FIELDS) - set(actual_fields))
        extra = sorted(set(actual_fields) - set(LIGHT_CURVE_FIELDS))
        raise ValueError(
            f"Minimal light-curve field manifest mismatch; missing={missing}, extra={extra}."
        )
    encoded_manifest = json.loads(str(handle.attrs.get(MANIFEST_ATTR, "[]")))
    if encoded_manifest != list(LIGHT_CURVE_FIELDS):
        raise ValueError("Minimal light-curve manifest attribute is invalid.")
