"""SDSS Stripe 82 epochs taken on non-photometric nights (zero-point offsets).

A *zero-point cell* is one SDSS band x one night x 2 degrees of right
ascension. Its offset is the median, over every quasar observed in it, of the
epoch magnitude minus that object's median magnitude in the band; with enough
objects it measures the night's zero-point error at that position (cloud gives
positive offsets). An object-night is flagged when its g, r or i cell holds at
least ``ZERO_POINT_MIN_OBJECTS`` objects and has ``|offset| > threshold``;
then all its SDSS epochs (every band) are flagged. This is the rule of
plots/diagnostics/gold_sample/cloud_epochs (s2_cloud_flags.py), where the
cells were measured on the epochs before the outlier clip.

Cell table format (parquet or CSV), one row per cell:

    band      str    SDSS band, one of u g r i z
    night     int    floor(MJD) of the epochs
    rab       int    floor(RA_wrapped / 2 deg), RA_wrapped = RA - 360 for RA > 180
    cell_off  float  median (epoch - object median) magnitude offset, mag
    cell_n    int    number of epochs (objects) in the cell

(band, night, rab) must be unique. :func:`build_zero_point_cells` builds the
table from an epoch table; the diagnostic ``cloud_cells.parquet`` has this
format.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ZERO_POINT_CELL_COLUMNS = ("band", "night", "rab", "cell_off", "cell_n")
ZERO_POINT_CELL_KEYS = ("band", "night", "rab")
ZERO_POINT_RA_BIN_DEG = 2.0
ZERO_POINT_MIN_OBJECTS = 8
ZERO_POINT_FLAG_BANDS = ("g", "r", "i")
DEFAULT_ZERO_POINT_FLAG_THRESHOLD = 0.1
SDSS_BANDS = ("u", "g", "r", "i", "z")


def zero_point_ra_bin(ra_deg):
    """Integer 2-degree RA bin of the cell table (RA wrapped to (-180, 180])."""
    ra = np.asarray(ra_deg, dtype=float)
    ra = np.where(ra > 180.0, ra - 360.0, ra)
    return np.floor(ra / ZERO_POINT_RA_BIN_DEG).astype(np.int64)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_zero_point_cells(cells, *, source="zero-point cell table"):
    """Check the columns, types and key uniqueness of a cell table."""
    missing = [column for column in ZERO_POINT_CELL_COLUMNS if column not in cells.columns]
    if missing:
        raise ValueError(f"{source} is missing column(s) {missing}; expected {list(ZERO_POINT_CELL_COLUMNS)}.")
    cells = cells.loc[:, list(ZERO_POINT_CELL_COLUMNS)].copy()
    cells["band"] = cells["band"].astype(str).str.strip()
    bad_bands = sorted(set(cells["band"]) - set(SDSS_BANDS))
    if bad_bands:
        raise ValueError(f"{source} has unknown band(s) {bad_bands}; expected {list(SDSS_BANDS)}.")
    for column in ("night", "rab", "cell_n"):
        values = pd.to_numeric(cells[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(values == np.round(values)):
            raise ValueError(f"{source} column {column!r} must hold integers.")
        cells[column] = values.astype(np.int64)
    offsets = pd.to_numeric(cells["cell_off"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(offsets)):
        raise ValueError(f"{source} column 'cell_off' must be finite.")
    cells["cell_off"] = offsets
    if cells.duplicated(list(ZERO_POINT_CELL_KEYS)).any():
        raise ValueError(f"{source} has duplicate (band, night, rab) cells.")
    return cells.reset_index(drop=True)


def load_zero_point_cells(path):
    """Read and validate a cell table (``.parquet`` or ``.csv``)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Zero-point cell table not found: {path}")
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    return validate_zero_point_cells(frame, source=f"Zero-point cell table {path}")


def build_zero_point_cells(epochs, ra_by_object):
    """Cell table from SDSS epochs.

    ``epochs`` has columns object_id, band, time (MJD) and mag (one row per
    SDSS epoch, before any outlier clip); ``ra_by_object`` maps object_id to
    RA in degrees.
    """
    epochs = epochs.loc[:, ["object_id", "band", "time", "mag"]].copy()
    ra = epochs["object_id"].map(ra_by_object)
    if ra.isna().any():
        raise ValueError(f"No RA for {int(ra.isna().sum())} epoch(s) of the cell table.")
    epochs["res"] = epochs["mag"] - epochs.groupby(["object_id", "band"])["mag"].transform("median")
    epochs["night"] = np.floor(epochs["time"].to_numpy(dtype=float)).astype(np.int64)
    epochs["rab"] = zero_point_ra_bin(ra.to_numpy(dtype=float))
    cells = (
        epochs.groupby(list(ZERO_POINT_CELL_KEYS))["res"]
        .agg(cell_off="median", cell_n="size")
        .reset_index()
    )
    return validate_zero_point_cells(cells)


def flag_zero_point_epochs(epochs, ra_by_object, cells, threshold=DEFAULT_ZERO_POINT_FLAG_THRESHOLD):
    """Boolean mask of SDSS epochs on flagged object-nights.

    ``epochs`` has columns object_id, band and time (MJD); ``cells`` is a
    validated cell table.
    """
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"The zero-point flag threshold must be a positive magnitude, got {threshold!r}.")
    if len(epochs) == 0:
        return np.zeros(0, dtype=bool)
    ra = epochs["object_id"].map(ra_by_object)
    if ra.isna().any():
        raise ValueError(f"No RA for {int(ra.isna().sum())} SDSS epoch(s) to flag.")
    flagged = cells[
        (cells["cell_n"] >= ZERO_POINT_MIN_OBJECTS)
        & (cells["cell_off"].abs() > threshold)
        & cells["band"].isin(ZERO_POINT_FLAG_BANDS)
    ]
    keys = pd.DataFrame(
        {
            "object_id": epochs["object_id"].to_numpy(),
            "band": epochs["band"].astype(str).to_numpy(),
            "night": np.floor(epochs["time"].to_numpy(dtype=float)).astype(np.int64),
            "rab": zero_point_ra_bin(ra.to_numpy(dtype=float)),
        }
    )
    # Left merges keep the epoch order; a matched row has a non-missing marker.
    in_flagged_cell = (
        keys.merge(flagged[list(ZERO_POINT_CELL_KEYS)].assign(_hit=1), on=list(ZERO_POINT_CELL_KEYS), how="left")[
            "_hit"
        ]
        .notna()
        .to_numpy()
    )
    bad_nights = keys.loc[in_flagged_cell, ["object_id", "night"]].drop_duplicates()
    return (
        keys[["object_id", "night"]]
        .merge(bad_nights.assign(_bad=1), on=["object_id", "night"], how="left")["_bad"]
        .notna()
        .to_numpy()
    )
