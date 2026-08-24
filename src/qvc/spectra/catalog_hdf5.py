"""Versioned HDF5 I/O for merged JAXSEDFit spectral catalogs."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np
import pandas as pd

from qvc.provenance import write_hdf5_provenance


SPECTRA_CATALOG_FORMAT = "qvc_spectra_catalog_v2"
PSF_AGN_FRACTION_BANDS = ("u", "g", "r", "i", "z")
PSF_AGN_FRACTION_DRAW_COUNT = 64
F_HOST_2500_PSF_DRAW_COUNT = 64


@dataclass(frozen=True)
class SpectraCatalog:
    frame: pd.DataFrame
    fraction_draws: np.ndarray
    valid_count: np.ndarray
    bands: tuple[str, ...]
    f_host_2500_psf_draws: np.ndarray
    f_host_2500_psf_valid_count: np.ndarray


def _decode_strings(values):
    arr = np.asarray(values)
    if arr.dtype.kind == "S":
        return arr.astype(str)
    if arr.dtype == object:
        return np.asarray(
            [v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v for v in arr],
            dtype=object,
        )
    return arr


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
        "" if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value))
        else value.decode("utf-8", errors="replace") if isinstance(value, bytes)
        else str(value)
        for value in series.tolist()
    ]
    group.create_dataset(name, data=values, dtype=string_dtype, compression="gzip", shuffle=True)


def write_spectra_catalog_hdf5(
    path,
    frame,
    fraction_draws,
    valid_count,
    *,
    f_host_2500_psf_draws,
    f_host_2500_psf_valid_count,
    provenance: Mapping | None = None,
):
    """Atomically write scalar catalog fields and compact joint fraction draws."""

    path = Path(path)
    frame = pd.DataFrame(frame).reset_index(drop=True)
    draws = np.asarray(fraction_draws, dtype=np.float32)
    counts = np.asarray(valid_count, dtype=np.int16)
    expected_shape = (len(frame), PSF_AGN_FRACTION_DRAW_COUNT, len(PSF_AGN_FRACTION_BANDS))
    if draws.shape != expected_shape:
        raise ValueError(f"fraction_draws has shape {draws.shape}; expected {expected_shape}.")
    if counts.shape != (len(frame),):
        raise ValueError(f"valid_count has shape {counts.shape}; expected {(len(frame),)}.")
    if np.any((counts < 0) | (counts > PSF_AGN_FRACTION_DRAW_COUNT)):
        raise ValueError("valid_count must be between 0 and 64.")
    for row_index, count in enumerate(counts):
        if not np.all(np.isfinite(draws[row_index, :count])):
            raise ValueError(
                f"fraction_draws row {row_index} is nonfinite within valid_count."
            )
        if not np.all(np.isnan(draws[row_index, count:])):
            raise ValueError(
                f"fraction_draws row {row_index} must be NaN-padded beyond valid_count."
            )

    host_draws = np.asarray(f_host_2500_psf_draws, dtype=np.float32)
    host_counts = np.asarray(f_host_2500_psf_valid_count, dtype=np.int16)
    expected_host_shape = (len(frame), F_HOST_2500_PSF_DRAW_COUNT)
    if host_draws.shape != expected_host_shape:
        raise ValueError(
            f"f_host_2500_psf_draws has shape {host_draws.shape}; "
            f"expected {expected_host_shape}."
        )
    if host_counts.shape != (len(frame),):
        raise ValueError(
            "f_host_2500_psf_valid_count has shape "
            f"{host_counts.shape}; expected {(len(frame),)}."
        )
    if np.any((host_counts < 0) | (host_counts > F_HOST_2500_PSF_DRAW_COUNT)):
        raise ValueError("f_host_2500_psf_valid_count must be between 0 and 64.")
    for row_index, count in enumerate(host_counts):
        valid = host_draws[row_index, :count]
        if not np.all(np.isfinite(valid)):
            raise ValueError(
                f"f_host_2500_psf_draws row {row_index} is nonfinite within "
                "f_host_2500_psf_valid_count."
            )
        if np.any((valid < 0.0) | (valid > 1.0)):
            raise ValueError(
                f"f_host_2500_psf_draws row {row_index} is outside [0, 1]."
            )
        if not np.all(np.isnan(host_draws[row_index, count:])):
            raise ValueError(
                f"f_host_2500_psf_draws row {row_index} must be NaN-padded "
                "beyond f_host_2500_psf_valid_count."
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with h5py.File(tmp_name, "w") as handle:
            handle.attrs["qvc_spectra_catalog_format"] = SPECTRA_CATALOG_FORMAT
            handle.attrs["psf_agn_fraction_draw_count"] = PSF_AGN_FRACTION_DRAW_COUNT
            handle.attrs["psf_agn_fraction_draw_selection"] = (
                "deterministic_uniform_without_replacement"
            )
            handle.attrs["f_host_2500_psf_draw_count"] = F_HOST_2500_PSF_DRAW_COUNT
            handle.attrs["f_host_2500_psf_draw_selection"] = (
                "deterministic_uniform_without_replacement"
            )
            catalog_group = handle.create_group("catalog", track_order=True)
            for column in frame.columns:
                _write_catalog_column(catalog_group, str(column), frame[column])

            draw_group = handle.create_group("psf_agn_fraction_draws")
            string_dtype = h5py.string_dtype(encoding="utf-8")
            draw_group.create_dataset(
                "bands",
                data=list(PSF_AGN_FRACTION_BANDS),
                dtype=string_dtype,
            )
            draw_group.create_dataset(
                "values",
                data=draws,
                dtype=np.float32,
                compression="gzip",
                shuffle=True,
            )
            draw_group.create_dataset(
                "valid_count",
                data=counts,
                dtype=np.int16,
                compression="gzip",
                shuffle=True,
            )
            host_draw_group = handle.create_group("f_host_2500_psf_draws")
            host_draw_group.create_dataset(
                "values",
                data=host_draws,
                dtype=np.float32,
                compression="gzip",
                shuffle=True,
            )
            host_draw_group.create_dataset(
                "valid_count",
                data=host_counts,
                dtype=np.int16,
                compression="gzip",
                shuffle=True,
            )
            if provenance is not None:
                write_hdf5_provenance(handle, provenance)
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def read_spectra_catalog_hdf5(path, *, include_fraction_draws=True):
    """Read and validate a versioned spectral catalog."""

    path = Path(path)
    with h5py.File(path, "r") as handle:
        actual_format = handle.attrs.get("qvc_spectra_catalog_format")
        if isinstance(actual_format, bytes):
            actual_format = actual_format.decode("utf-8", errors="replace")
        if actual_format != SPECTRA_CATALOG_FORMAT:
            raise ValueError(
                f"Spectra catalog {path} has format {actual_format!r}; "
                f"expected {SPECTRA_CATALOG_FORMAT!r}."
            )
        required_groups = {
            "catalog",
            "psf_agn_fraction_draws",
            "f_host_2500_psf_draws",
        }
        if not required_groups.issubset(handle.keys()):
            raise ValueError(f"Spectra catalog {path} is missing required groups.")
        if int(handle.attrs.get("psf_agn_fraction_draw_count", -1)) != PSF_AGN_FRACTION_DRAW_COUNT:
            raise ValueError(f"Spectra catalog {path} has an incompatible draw width.")

        catalog_group = handle["catalog"]
        columns = {name: _decode_strings(dataset[...]) for name, dataset in catalog_group.items()}
        lengths = {len(values) for values in columns.values()}
        if len(lengths) > 1:
            raise ValueError(f"Spectra catalog {path} has misaligned catalog columns.")
        frame = pd.DataFrame(columns)

        draw_group = handle["psf_agn_fraction_draws"]
        required_draw_datasets = {"bands", "values", "valid_count"}
        missing_draw_datasets = required_draw_datasets.difference(draw_group.keys())
        if missing_draw_datasets:
            raise ValueError(
                f"Spectra catalog {path} is missing fraction-draw datasets "
                f"{sorted(missing_draw_datasets)}."
            )
        bands = tuple(str(value) for value in _decode_strings(draw_group["bands"][...]))
        if bands != PSF_AGN_FRACTION_BANDS:
            raise ValueError(f"Spectra catalog {path} has unsupported band order {bands}.")
        counts = np.asarray(draw_group["valid_count"][...], dtype=np.int16)
        if counts.shape != (len(frame),):
            raise ValueError(f"Spectra catalog {path} has misaligned valid_count.")
        if np.any((counts < 0) | (counts > PSF_AGN_FRACTION_DRAW_COUNT)):
            raise ValueError(f"Spectra catalog {path} has invalid valid_count values.")
        expected_draw_shape = (
            len(frame),
            PSF_AGN_FRACTION_DRAW_COUNT,
            len(bands),
        )
        if draw_group["values"].shape != expected_draw_shape:
            raise ValueError(
                f"Spectra catalog {path} has invalid fraction-draw shape "
                f"{draw_group['values'].shape}."
            )
        if include_fraction_draws:
            draws = np.asarray(draw_group["values"][...], dtype=np.float32)
        else:
            draws = np.empty((0, PSF_AGN_FRACTION_DRAW_COUNT, len(bands)), dtype=np.float32)
        if include_fraction_draws:
            for row_index, valid_count in enumerate(counts):
                if not np.all(np.isfinite(draws[row_index, :valid_count])):
                    raise ValueError(
                        f"Spectra catalog {path} has nonfinite values within valid draws "
                        f"for row {row_index}."
                    )
                if not np.all(np.isnan(draws[row_index, valid_count:])):
                    raise ValueError(
                        f"Spectra catalog {path} has non-NaN values beyond valid_count "
                        f"for row {row_index}."
                    )

        host_group = handle["f_host_2500_psf_draws"]
        missing_host_datasets = {"values", "valid_count"}.difference(
            host_group.keys()
        )
        if missing_host_datasets:
            raise ValueError(
                f"Spectra catalog {path} is missing host-fraction draw datasets "
                f"{sorted(missing_host_datasets)}."
            )
        if int(handle.attrs.get("f_host_2500_psf_draw_count", -1)) != F_HOST_2500_PSF_DRAW_COUNT:
            raise ValueError(
                f"Spectra catalog {path} has an incompatible host-fraction draw width."
            )
        host_counts = np.asarray(host_group["valid_count"][...], dtype=np.int16)
        if host_counts.shape != (len(frame),) or np.any(
            (host_counts < 0) | (host_counts > F_HOST_2500_PSF_DRAW_COUNT)
        ):
            raise ValueError(
                f"Spectra catalog {path} has invalid host-fraction valid counts."
            )
        expected_host_shape = (len(frame), F_HOST_2500_PSF_DRAW_COUNT)
        if host_group["values"].shape != expected_host_shape:
            raise ValueError(
                f"Spectra catalog {path} has invalid host-fraction draw shape "
                f"{host_group['values'].shape}."
            )
        if include_fraction_draws:
            host_draws = np.asarray(host_group["values"][...], dtype=np.float32)
            for row_index, valid_count in enumerate(host_counts):
                valid = host_draws[row_index, :valid_count]
                if not np.all(np.isfinite(valid)) or np.any(
                    (valid < 0.0) | (valid > 1.0)
                ):
                    raise ValueError(
                        f"Spectra catalog {path} has invalid host-fraction draws "
                        f"for row {row_index}."
                    )
                if not np.all(np.isnan(host_draws[row_index, valid_count:])):
                    raise ValueError(
                        f"Spectra catalog {path} has non-NaN host-fraction values "
                        f"beyond valid_count for row {row_index}."
                    )
        else:
            host_draws = np.empty(
                (0, F_HOST_2500_PSF_DRAW_COUNT), dtype=np.float32
            )
    return SpectraCatalog(
        frame=frame,
        fraction_draws=draws,
        valid_count=counts,
        bands=bands,
        f_host_2500_psf_draws=host_draws,
        f_host_2500_psf_valid_count=host_counts,
    )
