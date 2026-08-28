"""Versioned HDF5 I/O for merged JAXSEDFit spectral catalogs."""

from __future__ import annotations

import os
import json
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np
import pandas as pd

from qvc.provenance import write_hdf5_provenance


SPECTRA_CATALOG_FORMAT = "qvc_spectra_catalog_v3"
SPECTRA_CATALOG_FORMAT_V2 = "qvc_spectra_catalog_v2"
SPECTRA_CATALOG_FORMAT_V1 = "qvc_spectra_catalog_v1"
PSF_AGN_FRACTION_BANDS = ("u", "g", "r", "i", "z")
PSF_AGN_FRACTION_DRAW_COUNT = 64
F_HOST_2500_PSF_DRAW_COUNT = 64
JOINT_POSTERIOR_DRAW_COUNT = 64
JOINT_POSTERIOR_DRAW_FIELDS = (
    "f_host_2500_psf",
    "alpha_nu_intrinsic_1450_2500",
    "alpha_nu_attenuated_1450_2500",
    "m_2500_dereddened",
    "m_2500_attenuated_model",
    "a_2500_galaxy",
    "a_2500_internal",
    "a_2500_total",
)
JOINT_POSTERIOR_SCALAR_SUMMARY_FIELDS = tuple(
    key
    for name in JOINT_POSTERIOR_DRAW_FIELDS
    for key in (
        name,
        f"{name}_err",
        f"{name}_err_lower",
        f"{name}_err_upper",
    )
)
JOINT_POSTERIOR_DRAW_SELECTION = (
    "sha256_seed_object_id_uniform_without_replacement_v1"
)
JOINT_PSF_PHOTOMETRY_FORMAT = "qvc_joint_psf_photometry_v1"
JOINT_PSF_PHOTOMETRY_GROUP = "joint_psf_photometry_draws"
JOINT_PSF_PHOTOMETRY_BANDS = tuple(
    f"{band}_sdss" for band in PSF_AGN_FRACTION_BANDS
)
JOINT_PSF_PHOTOMETRY_DRAW_COUNT = JOINT_POSTERIOR_DRAW_COUNT
JOINT_PSF_PHOTOMETRY_UNIT = "mJy"
JOINT_PSF_PHOTOMETRY_COMPONENT = "total_captured_psf_model_flux"
JOINT_PSF_PHOTOMETRY_MW_STATE = "dereddened"
JOINT_PSF_PHOTOMETRY_ALIGNMENT = (
    "/joint_posterior_draws/posterior_index"
)
ALPHA_NU_BLUE_WAVELENGTH_ANGSTROM = 1450.0
ALPHA_NU_RED_WAVELENGTH_ANGSTROM = 2500.0
GRAHSP_ATTENUATION_NORMALIZATION = 1.2
GRAHSP_ATTENUATION_BREAK_ANGSTROM = 11_000.0
GRAHSP_ATTENUATION_OPTICAL_INDEX = -1.2
F_HOST_2500_PSF_CAPTURE_MODEL = "sdss_typical_fwhm_with_fitted_host_scale"
F_HOST_2500_PSF_FWHM_ARCSEC = 1.4


@dataclass(frozen=True)
class SpectraCatalog:
    frame: pd.DataFrame
    fraction_draws: np.ndarray
    valid_count: np.ndarray
    bands: tuple[str, ...]
    # Compatibility accessors. In v3 these point to the corresponding joint
    # field; in v2 they retain independently sampled legacy host draws.
    f_host_2500_psf_draws: np.ndarray
    f_host_2500_psf_valid_count: np.ndarray
    catalog_format: str = SPECTRA_CATALOG_FORMAT
    joint_posterior_draws: Mapping[str, np.ndarray] = field(default_factory=dict)
    joint_posterior_valid_count: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int16)
    )
    joint_posterior_index: np.ndarray = field(
        default_factory=lambda: np.empty(
            (0, JOINT_POSTERIOR_DRAW_COUNT), dtype=np.int32
        )
    )
    joint_posterior_source_draw_count: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    joint_posterior_selection_seed: int | None = None
    # Required v3 product. Values are total fitted PSF model fluxes at the
    # exact v3 posterior indices above. Legacy v1/v2 catalogs expose None.
    joint_psf_photometry_draws: np.ndarray | None = None
    joint_psf_photometry_bands: tuple[str, ...] = ()
    joint_psf_photometry_provenance: Mapping[str, object] = field(
        default_factory=dict
    )


def _decode_strings(values):
    arr = np.asarray(values)
    if arr.dtype.kind == "S":
        return arr.astype(str)
    if arr.dtype == object:
        return np.asarray(
            [
                value.decode("utf-8", errors="replace")
                if isinstance(value, bytes)
                else value
                for value in arr
            ],
            dtype=object,
        )
    return arr


def _decode_attr(value):
    return (
        value.decode("utf-8", errors="replace")
        if isinstance(value, bytes)
        else value
    )


def _write_catalog_column(group, name, series):
    if pd.api.types.is_bool_dtype(series.dtype):
        group.create_dataset(name, data=series.to_numpy(dtype=bool))
        return
    if pd.api.types.is_numeric_dtype(series.dtype):
        values = pd.to_numeric(series, errors="coerce").to_numpy()
        group.create_dataset(name, data=values, compression="gzip", shuffle=True)
        return
    string_dtype = h5py.string_dtype(encoding="utf-8")
    values = [
        ""
        if value is None
        or value is pd.NA
        or (isinstance(value, float) and np.isnan(value))
        else value.decode("utf-8", errors="replace")
        if isinstance(value, bytes)
        else str(value)
        for value in series.tolist()
    ]
    group.create_dataset(
        name,
        data=values,
        dtype=string_dtype,
        compression="gzip",
        shuffle=True,
    )


def _validate_nan_padded_draws(values, counts, *, name, bounds=None):
    for row_index, count_value in enumerate(counts):
        count = int(count_value)
        valid = values[row_index, :count]
        if not np.all(np.isfinite(valid)):
            raise ValueError(f"{name} row {row_index} is nonfinite within valid_count.")
        if bounds is not None:
            lower, upper = bounds
            if np.any((valid < lower) | (valid > upper)):
                raise ValueError(
                    f"{name} row {row_index} is outside [{lower}, {upper}]."
                )
        if not np.all(np.isnan(values[row_index, count:])):
            raise ValueError(
                f"{name} row {row_index} must be NaN-padded beyond valid_count."
            )


def _coerce_exact_integer_array(values, *, dtype, name):
    """Convert integer metadata without silently truncating or wrapping it."""

    raw = np.asarray(values)
    try:
        numeric = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain exact integers.") from exc
    info = np.iinfo(dtype)
    if (
        not np.all(np.isfinite(numeric))
        or np.any(numeric != np.floor(numeric))
        or np.any(numeric < info.min)
        or np.any(numeric > info.max)
    ):
        raise ValueError(f"{name} must contain exact {np.dtype(dtype).name} integers.")
    return numeric.astype(dtype)


def _validate_psf_draw_payload(frame, fraction_draws, valid_count):
    draws = np.asarray(fraction_draws, dtype=np.float32)
    counts = _coerce_exact_integer_array(
        valid_count, dtype=np.int16, name="valid_count"
    )
    expected_shape = (
        len(frame),
        PSF_AGN_FRACTION_DRAW_COUNT,
        len(PSF_AGN_FRACTION_BANDS),
    )
    if draws.shape != expected_shape:
        raise ValueError(
            f"fraction_draws has shape {draws.shape}; expected {expected_shape}."
        )
    if counts.shape != (len(frame),):
        raise ValueError(
            f"valid_count has shape {counts.shape}; expected {(len(frame),)}."
        )
    if np.any((counts < 0) | (counts > PSF_AGN_FRACTION_DRAW_COUNT)):
        raise ValueError("valid_count must be between 0 and 64.")
    _validate_nan_padded_draws(
        draws,
        counts,
        name="fraction_draws",
        bounds=(0.0, 1.0),
    )
    return draws, counts


def _successful_row_mask(frame):
    if "fit_ok" not in frame:
        return None
    values = frame["fit_ok"]
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.to_numpy(dtype=bool)
    return values.astype(str).str.strip().str.lower().eq("true").to_numpy()


def _validate_v3_catalog_frame(frame):
    """Require the scalar/provenance half of the v3 posterior contract."""

    required = {
        "fit_ok",
        "mw_deredden_applied",
        "joint_posterior_draw_source",
        *JOINT_POSTERIOR_SCALAR_SUMMARY_FIELDS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "qvc_spectra_catalog_v3 lacks required scalar/provenance columns: "
            f"{missing}."
        )
    for name in ("fit_ok", "mw_deredden_applied"):
        if not pd.api.types.is_bool_dtype(frame[name].dtype):
            raise ValueError(
                f"qvc_spectra_catalog_v3 column {name!r} must be boolean."
            )
        if frame[name].isna().any():
            raise ValueError(
                f"qvc_spectra_catalog_v3 column {name!r} cannot contain nulls."
            )
    success = _successful_row_mask(frame)
    if not np.any(success):
        return

    numeric = frame.loc[success, JOINT_POSTERIOR_SCALAR_SUMMARY_FIELDS].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.all(np.isfinite(numeric.to_numpy(dtype=float))):
        raise ValueError(
            "Successful qvc_spectra_catalog_v3 rows require finite joint "
            "posterior scalar summaries."
        )
    uncertainty_columns = [
        name
        for name in JOINT_POSTERIOR_SCALAR_SUMMARY_FIELDS
        if name.endswith(("_err", "_err_lower", "_err_upper"))
    ]
    if np.any(numeric[uncertainty_columns].to_numpy(dtype=float) < 0.0):
        raise ValueError("v3 posterior scalar uncertainties cannot be negative.")
    f_host = numeric["f_host_2500_psf"].to_numpy(dtype=float)
    if np.any((f_host < 0.0) | (f_host > 1.0)):
        raise ValueError("v3 f_host_2500_psf scalar summaries are outside [0, 1].")
    attenuation = numeric[
        ["a_2500_galaxy", "a_2500_internal", "a_2500_total"]
    ].to_numpy(dtype=float)
    if np.any(attenuation < 0.0):
        raise ValueError("v3 A_2500 scalar summaries cannot be negative.")
    if np.any(
        numeric["m_2500_attenuated_model"].to_numpy(dtype=float)
        < numeric["m_2500_dereddened"].to_numpy(dtype=float)
    ):
        raise ValueError(
            "v3 attenuated m_2500 scalar summaries cannot be brighter than "
            "their dereddened summaries."
        )
    if np.any(
        numeric["alpha_nu_attenuated_1450_2500"].to_numpy(dtype=float)
        > numeric["alpha_nu_intrinsic_1450_2500"].to_numpy(dtype=float)
    ):
        raise ValueError(
            "v3 attenuated alpha_nu scalar summaries cannot be bluer than "
            "their intrinsic summaries."
        )
    sources = (
        frame.loc[success, "joint_posterior_draw_source"]
        .astype(str)
        .str.strip()
    )
    if (sources == "").any():
        raise ValueError(
            "Successful v3 rows require nonempty joint_posterior_draw_source."
        )


def _validate_joint_posterior_payload(
    frame,
    joint_posterior_draws,
    joint_posterior_valid_count,
    joint_posterior_index,
    joint_posterior_source_draw_count,
):
    _validate_v3_catalog_frame(frame)
    if joint_posterior_draws is None:
        raise ValueError(
            "qvc_spectra_catalog_v3 requires joint_posterior_draws."
        )
    fields = tuple(joint_posterior_draws.keys())
    if set(fields) != set(JOINT_POSTERIOR_DRAW_FIELDS):
        missing = sorted(set(JOINT_POSTERIOR_DRAW_FIELDS) - set(fields))
        extra = sorted(set(fields) - set(JOINT_POSTERIOR_DRAW_FIELDS))
        raise ValueError(
            "joint_posterior_draws has an incompatible field set; "
            f"missing={missing}, extra={extra}."
        )

    counts = _coerce_exact_integer_array(
        joint_posterior_valid_count,
        dtype=np.int16,
        name="joint_posterior_valid_count",
    )
    indices = _coerce_exact_integer_array(
        joint_posterior_index,
        dtype=np.int32,
        name="joint_posterior_index",
    )
    source_counts = _coerce_exact_integer_array(
        joint_posterior_source_draw_count,
        dtype=np.int32,
        name="joint_posterior_source_draw_count",
    )
    expected_shape = (len(frame), JOINT_POSTERIOR_DRAW_COUNT)
    if counts.shape != (len(frame),):
        raise ValueError(
            "joint_posterior_valid_count has shape "
            f"{counts.shape}; expected {(len(frame),)}."
        )
    if indices.shape != expected_shape:
        raise ValueError(
            f"joint_posterior_index has shape {indices.shape}; expected {expected_shape}."
        )
    if source_counts.shape != (len(frame),):
        raise ValueError(
            "joint_posterior_source_draw_count has shape "
            f"{source_counts.shape}; expected {(len(frame),)}."
        )
    if np.any((counts < 0) | (counts > JOINT_POSTERIOR_DRAW_COUNT)):
        raise ValueError("joint_posterior_valid_count must be between 0 and 64.")
    if np.any(source_counts < counts):
        raise ValueError(
            "joint_posterior_source_draw_count cannot be smaller than valid_count."
        )

    draws = {
        name: np.asarray(joint_posterior_draws[name], dtype=np.float32)
        for name in JOINT_POSTERIOR_DRAW_FIELDS
    }
    for name, values in draws.items():
        if values.shape != expected_shape:
            raise ValueError(
                f"joint posterior field {name!r} has shape {values.shape}; "
                f"expected {expected_shape}."
            )
        bounds = (0.0, 1.0) if name == "f_host_2500_psf" else None
        _validate_nan_padded_draws(
            values,
            counts,
            name=f"joint posterior field {name!r}",
            bounds=bounds,
        )

    success = _successful_row_mask(frame)
    if success is not None:
        if np.any(success & (counts == 0)):
            rows = np.flatnonzero(success & (counts == 0))[:10].tolist()
            raise ValueError(
                "Successful spectral rows require joint posterior draws; "
                f"missing at rows {rows}."
            )
        if np.any(~success & (counts != 0)):
            rows = np.flatnonzero(~success & (counts != 0))[:10].tolist()
            raise ValueError(
                "Unsuccessful spectral rows must not contain joint posterior draws; "
                f"found them at rows {rows}."
            )

    alpha_denominator = np.log10(
        ALPHA_NU_RED_WAVELENGTH_ANGSTROM
        / ALPHA_NU_BLUE_WAVELENGTH_ANGSTROM
    )
    attenuation_ratio = (
        ALPHA_NU_BLUE_WAVELENGTH_ANGSTROM
        / ALPHA_NU_RED_WAVELENGTH_ANGSTROM
    ) ** GRAHSP_ATTENUATION_OPTICAL_INDEX
    for row_index, count_value in enumerate(counts):
        count = int(count_value)
        valid_indices = indices[row_index, :count]
        if count:
            if np.any(valid_indices < 0) or np.any(
                valid_indices >= source_counts[row_index]
            ):
                raise ValueError(
                    f"joint_posterior_index row {row_index} is outside its source axis."
                )
            if count > 1 and np.any(np.diff(valid_indices) <= 0):
                raise ValueError(
                    f"joint_posterior_index row {row_index} must be strictly increasing."
                )
        if not np.all(indices[row_index, count:] == -1):
            raise ValueError(
                f"joint_posterior_index row {row_index} must be -1-padded."
            )
        if count == 0:
            if source_counts[row_index] != 0:
                raise ValueError(
                    "Rows without joint posterior draws must have source_draw_count=0."
                )
            continue

        slc = np.s_[row_index, :count]
        a_galaxy = draws["a_2500_galaxy"][slc]
        a_internal = draws["a_2500_internal"][slc]
        a_total = draws["a_2500_total"][slc]
        if np.any(a_galaxy < 0.0) or np.any(a_internal < 0.0) or np.any(
            a_total < 0.0
        ):
            raise ValueError(
                f"joint posterior attenuation is negative for row {row_index}."
            )
        if not np.allclose(
            a_total,
            a_galaxy + a_internal,
            rtol=2e-5,
            atol=2e-6,
        ):
            raise ValueError(
                "joint posterior violates A_2500,total = "
                f"A_2500,galaxy + A_2500,internal for row {row_index}."
            )
        if not np.allclose(
            draws["m_2500_attenuated_model"][slc]
            - draws["m_2500_dereddened"][slc],
            a_total,
            rtol=2e-5,
            atol=2e-6,
        ):
            raise ValueError(
                "joint posterior violates m_2500 attenuation identity for row "
                f"{row_index}."
            )
        expected_attenuated_alpha = (
            draws["alpha_nu_intrinsic_1450_2500"][slc]
            - 0.4 * a_total * (attenuation_ratio - 1.0) / alpha_denominator
        )
        if not np.allclose(
            draws["alpha_nu_attenuated_1450_2500"][slc],
            expected_attenuated_alpha,
            rtol=2e-5,
            atol=2e-6,
        ):
            raise ValueError(
                "joint posterior violates the attenuated alpha_nu identity for row "
                f"{row_index}."
            )
    return draws, counts, indices, source_counts


def _validate_joint_psf_photometry_payload(
    frame,
    values_mjy,
    joint_posterior_valid_count,
):
    """Validate required total-PSF photometry at the v3 joint indices."""

    if values_mjy is None:
        raise ValueError(
            "qvc_spectra_catalog_v3 requires joint_psf_photometry_draws."
        )
    values = np.asarray(values_mjy, dtype=np.float32)
    expected_shape = (
        len(frame),
        JOINT_PSF_PHOTOMETRY_DRAW_COUNT,
        len(JOINT_PSF_PHOTOMETRY_BANDS),
    )
    if values.shape != expected_shape:
        raise ValueError(
            "joint_psf_photometry_draws has shape "
            f"{values.shape}; expected {expected_shape}."
        )
    counts = np.asarray(joint_posterior_valid_count, dtype=np.int16)
    _validate_nan_padded_draws(
        values,
        counts,
        name="joint PSF photometry values_mjy",
    )
    for row_index, count_value in enumerate(counts):
        count = int(count_value)
        if np.any(values[row_index, :count] <= 0.0):
            raise ValueError(
                "joint PSF photometry values_mjy must be strictly positive "
                f"within valid_count (row {row_index})."
            )
    has_draws = counts > 0
    if np.any(has_draws):
        if "mw_deredden_applied" not in frame:
            raise ValueError(
                "Joint PSF photometry requires mw_deredden_applied provenance."
            )
        mw_dereddened = frame["mw_deredden_applied"].to_numpy(dtype=bool)
        if np.any(has_draws & ~mw_dereddened):
            rows = np.flatnonzero(has_draws & ~mw_dereddened)[:10].tolist()
            raise ValueError(
                "Joint PSF photometry is defined only for MW-dereddened fits; "
                f"invalid rows {rows}."
            )
    return values


def _normalize_joint_psf_photometry_provenance(provenance):
    """Return a JSON-safe, minimally complete extension provenance record."""

    if not isinstance(provenance, Mapping):
        raise ValueError(
            "joint_psf_photometry_provenance must be a mapping for the "
            "required v3 photometry payload."
        )
    normalized = json.loads(json.dumps(dict(provenance), default=str))
    required = {"prediction_source", "jaxsedfit_git_commit"}
    missing = sorted(
        key for key in required if not str(normalized.get(key, "")).strip()
    )
    if missing:
        raise ValueError(
            "joint_psf_photometry_provenance lacks required nonempty fields "
            f"{missing}."
        )
    return normalized


def write_spectra_catalog_hdf5(
    path,
    frame,
    fraction_draws,
    valid_count,
    *,
    joint_posterior_draws,
    joint_posterior_valid_count,
    joint_posterior_index,
    joint_posterior_source_draw_count,
    joint_posterior_selection_seed,
    joint_psf_photometry_draws=None,
    joint_psf_photometry_provenance: Mapping | None = None,
    f_host_2500_psf_draws=None,
    f_host_2500_psf_valid_count=None,
    provenance: Mapping | None = None,
):
    """Atomically write scalar fields and a validated v3 posterior payload."""

    path = Path(path)
    frame = pd.DataFrame(frame).reset_index(drop=True)
    draws, counts = _validate_psf_draw_payload(frame, fraction_draws, valid_count)
    (
        joint_draws,
        joint_counts,
        joint_indices,
        source_counts,
    ) = _validate_joint_posterior_payload(
        frame,
        joint_posterior_draws,
        joint_posterior_valid_count,
        joint_posterior_index,
        joint_posterior_source_draw_count,
    )
    joint_psf_fluxes = _validate_joint_psf_photometry_payload(
        frame,
        joint_psf_photometry_draws,
        joint_counts,
    )
    joint_psf_provenance = _normalize_joint_psf_photometry_provenance(
        joint_psf_photometry_provenance
    )
    try:
        selection_seed = int(joint_posterior_selection_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("joint_posterior_selection_seed must be an integer.") from exc

    # Transitional callers may pass old host arguments. Validate that they are
    # merely aliases of the v3 field; never write an independently drifting group.
    if f_host_2500_psf_draws is not None:
        legacy_host = np.asarray(f_host_2500_psf_draws, dtype=np.float32)
        if legacy_host.shape != joint_draws["f_host_2500_psf"].shape or not np.allclose(
            legacy_host,
            joint_draws["f_host_2500_psf"],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ):
            raise ValueError(
                "Legacy f_host_2500_psf_draws does not match the v3 joint field."
            )
    if f_host_2500_psf_valid_count is not None and not np.array_equal(
        np.asarray(f_host_2500_psf_valid_count, dtype=np.int16), joint_counts
    ):
        raise ValueError(
            "Legacy f_host_2500_psf_valid_count does not match the v3 joint count."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        with h5py.File(tmp_name, "w") as handle:
            handle.attrs["qvc_spectra_catalog_format"] = SPECTRA_CATALOG_FORMAT
            handle.attrs["psf_agn_fraction_draw_count"] = PSF_AGN_FRACTION_DRAW_COUNT
            handle.attrs["psf_agn_fraction_draw_selection"] = (
                "deterministic_uniform_without_replacement"
            )
            handle.attrs["psf_agn_fraction_alignment_scope"] = (
                "ugriz_only_not_joint_posterior"
            )
            handle.attrs["joint_posterior_draw_count"] = JOINT_POSTERIOR_DRAW_COUNT
            handle.attrs["joint_posterior_draw_selection"] = (
                JOINT_POSTERIOR_DRAW_SELECTION
            )
            handle.attrs["joint_posterior_selection_seed"] = selection_seed
            handle.attrs["joint_posterior_index_semantics"] = (
                "zero_based_saved_bundle_leading_axis"
            )
            handle.attrs["alpha_nu_blue_wavelength_angstrom"] = (
                ALPHA_NU_BLUE_WAVELENGTH_ANGSTROM
            )
            handle.attrs["alpha_nu_red_wavelength_angstrom"] = (
                ALPHA_NU_RED_WAVELENGTH_ANGSTROM
            )
            handle.attrs["alpha_nu_definition"] = (
                "log10(Lnu_disk_1450/Lnu_disk_2500)/log10(2500/1450)"
            )
            handle.attrs["alpha_nu_convention"] = "Lnu_proportional_to_nu_power_alpha_nu"
            handle.attrs["alpha_lambda_relation"] = "alpha_lambda=-2-alpha_nu"
            handle.attrs["alpha_nu_intrinsic_component"] = (
                "unattenuated_bent_agn_disk_continuum_only"
            )
            handle.attrs["alpha_nu_attenuated_component"] = (
                "same_disk_after_ebv_gal_plus_ebv_agn_biattenuation"
            )
            handle.attrs["alpha_nu_excluded_components"] = (
                "host,host_dust,torus,feii,balmer_continuum,broad_lines,narrow_lines"
            )
            handle.attrs["m_2500_dereddened_definition"] = (
                "unattenuated_disk_only_rest_convention_apparent_ab"
            )
            handle.attrs["m_2500_attenuated_model_definition"] = (
                "m_2500_dereddened_plus_a_2500_total"
            )
            handle.attrs["a_2500_galaxy_definition"] = (
                "ebv_gal_times_grahsp_biattenuation_curve_at_2500"
            )
            handle.attrs["a_2500_internal_definition"] = (
                "ebv_agn_times_grahsp_biattenuation_curve_at_2500"
            )
            handle.attrs["a_2500_total_definition"] = (
                "a_2500_galaxy_plus_a_2500_internal"
            )
            handle.attrs["grahsp_attenuation_normalization"] = (
                GRAHSP_ATTENUATION_NORMALIZATION
            )
            handle.attrs["grahsp_attenuation_break_angstrom"] = (
                GRAHSP_ATTENUATION_BREAK_ANGSTROM
            )
            handle.attrs["grahsp_attenuation_optical_index"] = (
                GRAHSP_ATTENUATION_OPTICAL_INDEX
            )
            handle.attrs["milky_way_extinction_state"] = (
                "per_row_catalog_column:mw_deredden_applied"
            )
            handle.attrs["f_host_2500_psf_definition"] = (
                "captured_attenuated_host_over_captured_host_plus_attenuated_agn"
            )
            handle.attrs["f_host_2500_psf_capture_model"] = (
                F_HOST_2500_PSF_CAPTURE_MODEL
            )
            handle.attrs["f_host_2500_psf_fwhm_arcsec"] = (
                F_HOST_2500_PSF_FWHM_ARCSEC
            )

            catalog_group = handle.create_group("catalog", track_order=True)
            for column in frame.columns:
                _write_catalog_column(catalog_group, str(column), frame[column])

            draw_group = handle.create_group("psf_agn_fraction_draws")
            string_dtype = h5py.string_dtype(encoding="utf-8")
            draw_group.create_dataset(
                "bands", data=list(PSF_AGN_FRACTION_BANDS), dtype=string_dtype
            )
            draw_group.create_dataset(
                "values", data=draws, dtype=np.float32, compression="gzip", shuffle=True
            )
            draw_group.create_dataset(
                "valid_count", data=counts, dtype=np.int16, compression="gzip", shuffle=True
            )

            joint_group = handle.create_group("joint_posterior_draws")
            joint_group.create_dataset(
                "posterior_index", data=joint_indices, dtype=np.int32, compression="gzip", shuffle=True
            )
            joint_group.create_dataset(
                "source_draw_count", data=source_counts, dtype=np.int32, compression="gzip", shuffle=True
            )
            joint_group.create_dataset(
                "valid_count", data=joint_counts, dtype=np.int16, compression="gzip", shuffle=True
            )
            for name in JOINT_POSTERIOR_DRAW_FIELDS:
                joint_group.create_dataset(
                    name,
                    data=joint_draws[name],
                    dtype=np.float32,
                    compression="gzip",
                    shuffle=True,
                )
            psf_group = handle.create_group(JOINT_PSF_PHOTOMETRY_GROUP)
            psf_group.attrs["format"] = JOINT_PSF_PHOTOMETRY_FORMAT
            psf_group.attrs["unit"] = JOINT_PSF_PHOTOMETRY_UNIT
            psf_group.attrs["component"] = JOINT_PSF_PHOTOMETRY_COMPONENT
            psf_group.attrs["milky_way_extinction_state"] = (
                JOINT_PSF_PHOTOMETRY_MW_STATE
            )
            psf_group.attrs["source_frame_attenuation_state"] = "retained"
            psf_group.attrs["posterior_alignment"] = (
                JOINT_PSF_PHOTOMETRY_ALIGNMENT
            )
            psf_group.attrs["included_components"] = (
                "attenuated_agn_disk,host,emission_lines,feii,"
                "balmer_continuum,igm,torus"
            )
            psf_group.attrs["provenance_json"] = json.dumps(
                joint_psf_provenance,
                sort_keys=True,
                separators=(",", ":"),
            )
            string_dtype = h5py.string_dtype(encoding="utf-8")
            psf_group.create_dataset(
                "bands",
                data=list(JOINT_PSF_PHOTOMETRY_BANDS),
                dtype=string_dtype,
            )
            psf_group.create_dataset(
                "values_mjy",
                data=joint_psf_fluxes,
                dtype=np.float32,
                compression="gzip",
                shuffle=True,
            )
            if provenance is not None:
                write_hdf5_provenance(handle, provenance)
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _read_catalog_frame(handle, path):
    catalog_group = handle["catalog"]
    columns = {
        name: _decode_strings(dataset[...])
        for name, dataset in catalog_group.items()
    }
    lengths = {len(values) for values in columns.values()}
    if len(lengths) > 1:
        raise ValueError(f"Spectra catalog {path} has misaligned catalog columns.")
    return pd.DataFrame(columns)


def _read_psf_draws(handle, frame, path, include_draws):
    if int(handle.attrs.get("psf_agn_fraction_draw_count", -1)) != PSF_AGN_FRACTION_DRAW_COUNT:
        raise ValueError(f"Spectra catalog {path} has an incompatible draw width.")
    draw_group = handle["psf_agn_fraction_draws"]
    missing = {"bands", "values", "valid_count"}.difference(draw_group.keys())
    if missing:
        raise ValueError(
            f"Spectra catalog {path} is missing fraction-draw datasets {sorted(missing)}."
        )
    bands = tuple(str(value) for value in _decode_strings(draw_group["bands"][...]))
    if bands != PSF_AGN_FRACTION_BANDS:
        raise ValueError(f"Spectra catalog {path} has unsupported band order {bands}.")
    counts = _coerce_exact_integer_array(
        draw_group["valid_count"][...],
        dtype=np.int16,
        name=f"Spectra catalog {path} PSF valid_count",
    )
    if counts.shape != (len(frame),) or np.any(
        (counts < 0) | (counts > PSF_AGN_FRACTION_DRAW_COUNT)
    ):
        raise ValueError(f"Spectra catalog {path} has invalid valid_count values.")
    expected_shape = (len(frame), PSF_AGN_FRACTION_DRAW_COUNT, len(bands))
    if draw_group["values"].shape != expected_shape:
        raise ValueError(
            f"Spectra catalog {path} has invalid fraction-draw shape {draw_group['values'].shape}."
        )
    if not include_draws:
        return (
            np.empty((0, PSF_AGN_FRACTION_DRAW_COUNT, len(bands)), dtype=np.float32),
            counts,
            bands,
        )
    draws = np.asarray(draw_group["values"][...], dtype=np.float32)
    _validate_nan_padded_draws(
        draws, counts, name=f"Spectra catalog {path} fraction draws", bounds=(0.0, 1.0)
    )
    return draws, counts, bands


def _read_legacy_host_draws(handle, frame, path, include_draws, *, is_v1):
    if is_v1:
        counts = np.zeros(len(frame), dtype=np.int16)
        values = (
            np.full((len(frame), F_HOST_2500_PSF_DRAW_COUNT), np.nan, dtype=np.float32)
            if include_draws
            else np.empty((0, F_HOST_2500_PSF_DRAW_COUNT), dtype=np.float32)
        )
        return values, counts
    group = handle["f_host_2500_psf_draws"]
    missing = {"values", "valid_count"}.difference(group.keys())
    if missing:
        raise ValueError(
            f"Spectra catalog {path} is missing host-fraction draw datasets {sorted(missing)}."
        )
    if int(handle.attrs.get("f_host_2500_psf_draw_count", -1)) != F_HOST_2500_PSF_DRAW_COUNT:
        raise ValueError(
            f"Spectra catalog {path} has an incompatible host-fraction draw width."
        )
    counts = _coerce_exact_integer_array(
        group["valid_count"][...],
        dtype=np.int16,
        name=f"Spectra catalog {path} host-fraction valid_count",
    )
    if counts.shape != (len(frame),) or np.any(
        (counts < 0) | (counts > F_HOST_2500_PSF_DRAW_COUNT)
    ):
        raise ValueError(f"Spectra catalog {path} has invalid host-fraction valid counts.")
    expected_shape = (len(frame), F_HOST_2500_PSF_DRAW_COUNT)
    if group["values"].shape != expected_shape:
        raise ValueError(
            f"Spectra catalog {path} has invalid host-fraction draw shape {group['values'].shape}."
        )
    if not include_draws:
        return np.empty((0, F_HOST_2500_PSF_DRAW_COUNT), dtype=np.float32), counts
    values = np.asarray(group["values"][...], dtype=np.float32)
    _validate_nan_padded_draws(
        values,
        counts,
        name=f"Spectra catalog {path} host-fraction draws",
        bounds=(0.0, 1.0),
    )
    return values, counts


def _validate_v3_attrs(handle, path):
    expected = {
        "joint_posterior_draw_count": JOINT_POSTERIOR_DRAW_COUNT,
        "joint_posterior_draw_selection": JOINT_POSTERIOR_DRAW_SELECTION,
        "joint_posterior_index_semantics": "zero_based_saved_bundle_leading_axis",
        "alpha_nu_definition": (
            "log10(Lnu_disk_1450/Lnu_disk_2500)/log10(2500/1450)"
        ),
        "alpha_nu_convention": "Lnu_proportional_to_nu_power_alpha_nu",
        "alpha_lambda_relation": "alpha_lambda=-2-alpha_nu",
        "alpha_nu_intrinsic_component": (
            "unattenuated_bent_agn_disk_continuum_only"
        ),
        "alpha_nu_attenuated_component": (
            "same_disk_after_ebv_gal_plus_ebv_agn_biattenuation"
        ),
        "alpha_nu_excluded_components": (
            "host,host_dust,torus,feii,balmer_continuum,broad_lines,narrow_lines"
        ),
        "alpha_nu_blue_wavelength_angstrom": ALPHA_NU_BLUE_WAVELENGTH_ANGSTROM,
        "alpha_nu_red_wavelength_angstrom": ALPHA_NU_RED_WAVELENGTH_ANGSTROM,
        "m_2500_dereddened_definition": (
            "unattenuated_disk_only_rest_convention_apparent_ab"
        ),
        "m_2500_attenuated_model_definition": (
            "m_2500_dereddened_plus_a_2500_total"
        ),
        "a_2500_galaxy_definition": (
            "ebv_gal_times_grahsp_biattenuation_curve_at_2500"
        ),
        "a_2500_internal_definition": (
            "ebv_agn_times_grahsp_biattenuation_curve_at_2500"
        ),
        "a_2500_total_definition": (
            "a_2500_galaxy_plus_a_2500_internal"
        ),
        "grahsp_attenuation_normalization": GRAHSP_ATTENUATION_NORMALIZATION,
        "grahsp_attenuation_break_angstrom": GRAHSP_ATTENUATION_BREAK_ANGSTROM,
        "grahsp_attenuation_optical_index": GRAHSP_ATTENUATION_OPTICAL_INDEX,
        "milky_way_extinction_state": (
            "per_row_catalog_column:mw_deredden_applied"
        ),
        "f_host_2500_psf_definition": (
            "captured_attenuated_host_over_captured_host_plus_attenuated_agn"
        ),
        "f_host_2500_psf_capture_model": F_HOST_2500_PSF_CAPTURE_MODEL,
        "f_host_2500_psf_fwhm_arcsec": F_HOST_2500_PSF_FWHM_ARCSEC,
    }
    for name, expected_value in expected.items():
        actual = _decode_attr(handle.attrs.get(name))
        if actual != expected_value:
            raise ValueError(
                f"Spectra catalog {path} has incompatible {name}: {actual!r} != {expected_value!r}."
            )
    try:
        return int(handle.attrs["joint_posterior_selection_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Spectra catalog {path} lacks a valid joint posterior selection seed."
        ) from exc


def _read_v3_joint_draws(handle, frame, path, include_draws):
    selection_seed = _validate_v3_attrs(handle, path)
    group = handle["joint_posterior_draws"]
    required = {
        "posterior_index",
        "source_draw_count",
        "valid_count",
        *JOINT_POSTERIOR_DRAW_FIELDS,
    }
    missing = required.difference(group.keys())
    extra = set(group.keys()).difference(required)
    if missing or extra:
        raise ValueError(
            f"Spectra catalog {path} has incompatible joint posterior datasets; "
            f"missing={sorted(missing)}, extra={sorted(extra)}."
        )
    counts = _coerce_exact_integer_array(
        group["valid_count"][...],
        dtype=np.int16,
        name=f"Spectra catalog {path} joint valid_count",
    )
    indices = _coerce_exact_integer_array(
        group["posterior_index"][...],
        dtype=np.int32,
        name=f"Spectra catalog {path} posterior_index",
    )
    source_counts = _coerce_exact_integer_array(
        group["source_draw_count"][...],
        dtype=np.int32,
        name=f"Spectra catalog {path} source_draw_count",
    )
    if include_draws:
        values = {
            name: np.asarray(group[name][...], dtype=np.float32)
            for name in JOINT_POSTERIOR_DRAW_FIELDS
        }
        values, counts, indices, source_counts = _validate_joint_posterior_payload(
            frame, values, counts, indices, source_counts
        )
    else:
        expected_shape = (len(frame), JOINT_POSTERIOR_DRAW_COUNT)
        if (
            counts.shape != (len(frame),)
            or indices.shape != expected_shape
            or source_counts.shape != (len(frame),)
        ):
            raise ValueError(f"Spectra catalog {path} has misaligned joint metadata.")
        for name in JOINT_POSTERIOR_DRAW_FIELDS:
            if group[name].shape != expected_shape:
                raise ValueError(
                    f"Spectra catalog {path} has invalid joint field shape for {name!r}."
                )
        values = {
            name: np.empty((0, JOINT_POSTERIOR_DRAW_COUNT), dtype=np.float32)
            for name in JOINT_POSTERIOR_DRAW_FIELDS
        }
    return values, counts, indices, source_counts, selection_seed


def _read_v3_joint_psf_photometry(
    handle,
    frame,
    path,
    include_draws,
    joint_counts,
):
    """Read the required fitted total-PSF photometry product."""
    group = handle[JOINT_PSF_PHOTOMETRY_GROUP]
    required_datasets = {"bands", "values_mjy"}
    missing = required_datasets.difference(group.keys())
    extra = set(group.keys()).difference(required_datasets)
    if missing or extra:
        raise ValueError(
            f"Spectra catalog {path} has incompatible joint PSF photometry "
            f"datasets; missing={sorted(missing)}, extra={sorted(extra)}."
        )
    expected_attrs = {
        "format": JOINT_PSF_PHOTOMETRY_FORMAT,
        "unit": JOINT_PSF_PHOTOMETRY_UNIT,
        "component": JOINT_PSF_PHOTOMETRY_COMPONENT,
        "milky_way_extinction_state": JOINT_PSF_PHOTOMETRY_MW_STATE,
        "source_frame_attenuation_state": "retained",
        "posterior_alignment": JOINT_PSF_PHOTOMETRY_ALIGNMENT,
        "included_components": (
            "attenuated_agn_disk,host,emission_lines,feii,"
            "balmer_continuum,igm,torus"
        ),
    }
    for name, expected in expected_attrs.items():
        actual = _decode_attr(group.attrs.get(name))
        if actual != expected:
            raise ValueError(
                f"Spectra catalog {path} has incompatible joint PSF "
                f"photometry {name}: {actual!r} != {expected!r}."
            )
    bands = tuple(
        str(value) for value in _decode_strings(group["bands"][...])
    )
    if bands != JOINT_PSF_PHOTOMETRY_BANDS:
        raise ValueError(
            f"Spectra catalog {path} has unsupported joint PSF photometry "
            f"band order {bands}."
        )
    expected_shape = (
        len(frame),
        JOINT_PSF_PHOTOMETRY_DRAW_COUNT,
        len(bands),
    )
    if group["values_mjy"].shape != expected_shape:
        raise ValueError(
            f"Spectra catalog {path} has invalid joint PSF photometry shape "
            f"{group['values_mjy'].shape}; expected {expected_shape}."
        )
    raw_provenance = _decode_attr(group.attrs.get("provenance_json"))
    try:
        provenance = json.loads(str(raw_provenance))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Spectra catalog {path} has invalid joint PSF photometry provenance."
        ) from exc
    provenance = _normalize_joint_psf_photometry_provenance(provenance)
    if not include_draws:
        return (
            np.empty(
                (0, JOINT_PSF_PHOTOMETRY_DRAW_COUNT, len(bands)),
                dtype=np.float32,
            ),
            bands,
            provenance,
        )
    values = _validate_joint_psf_photometry_payload(
        frame,
        group["values_mjy"][...],
        joint_counts,
    )
    return values, bands, provenance


def read_spectra_catalog_hdf5(
    path,
    *,
    include_fraction_draws=True,
    allow_v2=False,
    allow_v1=False,
):
    """Read and strongly validate a versioned spectral catalog.

    V1 and v2 are rejected by default. Their explicit compatibility modes retain
    legacy products but never fabricate the v3 joint posterior payload.
    """

    path = Path(path)
    with h5py.File(path, "r") as handle:
        actual_format = str(_decode_attr(handle.attrs.get("qvc_spectra_catalog_format")))
        is_v3 = actual_format == SPECTRA_CATALOG_FORMAT
        is_v2 = actual_format == SPECTRA_CATALOG_FORMAT_V2
        is_v1 = actual_format == SPECTRA_CATALOG_FORMAT_V1
        allowed_legacy = (is_v2 and allow_v2) or (is_v1 and allow_v1)
        if not is_v3 and not allowed_legacy:
            hint = ""
            if is_v2:
                hint = " Pass allow_v2=True only for an explicitly approved legacy workflow."
            elif is_v1:
                hint = (
                    " Pass allow_v1=True only for an explicitly approved temporary "
                    "compatibility workflow."
                )
            raise ValueError(
                f"Spectra catalog {path} has format {actual_format!r}; "
                f"expected {SPECTRA_CATALOG_FORMAT!r}.{hint}"
            )
        if is_v2:
            warnings.warn(
                f"Loading legacy spectra catalog {path} with explicit v2 "
                "compatibility enabled. Its host draws are not jointly indexed "
                "with alpha_nu or A_2500 draws.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif is_v1:
            warnings.warn(
                f"Loading legacy spectra catalog {path} with explicit v1 "
                "compatibility enabled. It has no native f_host_2500_psf or "
                "joint physical posterior draws.",
                RuntimeWarning,
                stacklevel=2,
            )

        required_groups = {"catalog", "psf_agn_fraction_draws"}
        if is_v3:
            required_groups.update(
                {"joint_posterior_draws", JOINT_PSF_PHOTOMETRY_GROUP}
            )
        elif is_v2:
            required_groups.add("f_host_2500_psf_draws")
        if not required_groups.issubset(handle.keys()):
            raise ValueError(f"Spectra catalog {path} is missing required groups.")

        frame = _read_catalog_frame(handle, path)
        fraction_draws, counts, bands = _read_psf_draws(
            handle, frame, path, include_fraction_draws
        )
        if is_v3:
            (
                joint_draws,
                joint_counts,
                joint_indices,
                source_counts,
                selection_seed,
            ) = _read_v3_joint_draws(handle, frame, path, include_fraction_draws)
            host_draws = joint_draws["f_host_2500_psf"]
            host_counts = joint_counts
            (
                joint_psf_photometry_draws,
                joint_psf_photometry_bands,
                joint_psf_photometry_provenance,
            ) = _read_v3_joint_psf_photometry(
                handle,
                frame,
                path,
                include_fraction_draws,
                joint_counts,
            )
        else:
            host_draws, host_counts = _read_legacy_host_draws(
                handle, frame, path, include_fraction_draws, is_v1=is_v1
            )
            joint_draws = {}
            joint_counts = np.zeros(len(frame), dtype=np.int16)
            joint_indices = np.full(
                (len(frame), JOINT_POSTERIOR_DRAW_COUNT), -1, dtype=np.int32
            )
            source_counts = np.zeros(len(frame), dtype=np.int32)
            selection_seed = None
            joint_psf_photometry_draws = None
            joint_psf_photometry_bands = ()
            joint_psf_photometry_provenance = {}

    return SpectraCatalog(
        frame=frame,
        fraction_draws=fraction_draws,
        valid_count=counts,
        bands=bands,
        f_host_2500_psf_draws=host_draws,
        f_host_2500_psf_valid_count=host_counts,
        catalog_format=actual_format,
        joint_posterior_draws=joint_draws,
        joint_posterior_valid_count=joint_counts,
        joint_posterior_index=joint_indices,
        joint_posterior_source_draw_count=source_counts,
        joint_posterior_selection_seed=selection_seed,
        joint_psf_photometry_draws=joint_psf_photometry_draws,
        joint_psf_photometry_bands=joint_psf_photometry_bands,
        joint_psf_photometry_provenance=joint_psf_photometry_provenance,
    )
