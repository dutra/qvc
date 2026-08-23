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


SPECTRA_CATALOG_FORMAT = "qvc_spectra_catalog_v1"
PSF_AGN_FRACTION_BANDS = ("u", "g", "r", "i", "z")
PSF_AGN_FRACTION_DRAW_COUNT = 64


@dataclass(frozen=True)
class SpectraCatalog:
    frame: pd.DataFrame
    fraction_draws: np.ndarray
    valid_count: np.ndarray
    bands: tuple[str, ...]


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
        if "catalog" not in handle or "psf_agn_fraction_draws" not in handle:
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
    return SpectraCatalog(frame=frame, fraction_draws=draws, valid_count=counts, bands=bands)
