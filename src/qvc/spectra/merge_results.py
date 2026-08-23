#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import sys

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

from qvc.hubble.hubble_utils import populate_sdss_fields
from qvc.hubble.hubble_utils import resolve_qvc_data_path
from qvc.provenance import build_run_record
from qvc.spectra.catalog_hdf5 import (
    PSF_AGN_FRACTION_BANDS,
    PSF_AGN_FRACTION_DRAW_COUNT,
    SpectraCatalog,
    read_spectra_catalog_hdf5,
    write_spectra_catalog_hdf5,
)


def read_quasars_from_csv(csv_path):
    return pd.read_csv(csv_path)


def write_quasars_to_csv(quasars, csv_path, fields=None):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    if not quasars:
        with open(csv_path, "w", newline="") as f:
            if fields:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
        print(f"Wrote empty CSV to {csv_path}")
        return

    if fields is None:
        seen = []
        seen_set = set()
        for q in quasars:
            for k in q.keys():
                if k not in seen_set:
                    seen_set.add(k)
                    seen.append(k)
        fields = seen

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for q in quasars:
            writer.writerow({k: q.get(k, "") for k in fields})

    print(f"Wrote {len(quasars)} rows to {csv_path}")


def enforce_expected_count(per_file_count, expected_n, file_path):
    if expected_n is None:
        return True
    if per_file_count == expected_n:
        return True

    print(
        f"WARNING: Skipping {file_path} with {per_file_count} objects "
        f"(expected {expected_n})."
    )
    return False


def is_missing_value(v):
    if v is None:
        return True
    try:
        return pd.isna(v)
    except Exception:
        return False


def deduplicate(quasars, keys):
    if not keys:
        return list(quasars)

    if isinstance(keys, (str, bytes)):
        keys = [keys]

    merged = {}
    order = []

    for q in quasars:
        vals = []
        for k in keys:
            v = q.get(k)
            if is_missing_value(v) or v == "":
                comp_key = ("__objid__", id(q))
                break
            vals.append(v)
        else:
            try:
                comp_key = tuple(vals)
                hash(comp_key)
            except TypeError:
                comp_key = tuple(repr(v) for v in vals)

        if comp_key not in merged:
            order.append(comp_key)

        # last occurrence wins
        merged[comp_key] = q

    return [merged[k] for k in order]


def load_and_merge_csv(file_list, expected_n):
    all_quasars = []

    for path in tqdm(file_list, desc="Merging CSV shards", unit="file"):
        try:
            qs_df = read_quasars_from_csv(path)

            if not enforce_expected_count(len(qs_df), expected_n, path):
                continue

            qs = qs_df.to_dict("records")
            all_quasars.extend(qs)

        except Exception as e:
            print(f"ERROR reading {path}: {e}")
            continue

    return all_quasars


def _dedup_row_indices(frame, keys):
    if not keys:
        return np.arange(len(frame), dtype=int)
    seen = {}
    order = []
    for index, row in frame.iterrows():
        values = []
        for key in keys:
            value = row.get(key)
            if is_missing_value(value) or value == "":
                comparison_key = ("__row__", int(index))
                break
            values.append(value)
        else:
            comparison_key = tuple(values)
        if comparison_key not in seen:
            order.append(comparison_key)
        seen[comparison_key] = int(index)
    return np.asarray([seen[key] for key in order], dtype=int)


_NUMERIC_DTYPE_KINDS = frozenset("iufc")


def _scalar_dtype_kinds_compatible(left, right):
    """Return whether two shard dtypes can share one scalar catalog column."""

    if left == right:
        return True
    return left in _NUMERIC_DTYPE_KINDS and right in _NUMERIC_DTYPE_KINDS


def load_and_merge_h5(file_list, expected_n=None, dedup_keys=None):
    """Load HDF5 shards and keep optional fields and fraction draws aligned."""

    frames = []
    draws = []
    counts = []
    column_order = []
    column_dtype_kinds = {}
    column_sources = {}
    for path in tqdm(file_list, desc="Merging HDF5 shards", unit="file"):
        catalog = read_spectra_catalog_hdf5(path)
        if not enforce_expected_count(len(catalog.frame), expected_n, path):
            continue
        if catalog.bands != PSF_AGN_FRACTION_BANDS:
            raise ValueError(f"Incompatible fraction bands in {path}: {catalog.bands}")
        for column in catalog.frame.columns:
            dtype_kind = catalog.frame[column].dtype.kind
            if column not in column_dtype_kinds:
                column_order.append(column)
                column_dtype_kinds[column] = dtype_kind
                column_sources[column] = path
                continue
            reference_kind = column_dtype_kinds[column]
            if not _scalar_dtype_kinds_compatible(reference_kind, dtype_kind):
                raise ValueError(
                    f"Incompatible scalar catalog dtype for column {column!r} in "
                    f"{path}: kind {dtype_kind!r} cannot be merged with kind "
                    f"{reference_kind!r} first seen in {column_sources[column]}."
                )
        frames.append(catalog.frame)
        draws.append(catalog.fraction_draws)
        counts.append(catalog.valid_count)

    if not frames:
        return SpectraCatalog(
            frame=pd.DataFrame(),
            fraction_draws=np.empty(
                (0, PSF_AGN_FRACTION_DRAW_COUNT, len(PSF_AGN_FRACTION_BANDS)),
                dtype=np.float32,
            ),
            valid_count=np.empty(0, dtype=np.int16),
            bands=PSF_AGN_FRACTION_BANDS,
        )

    # Posterior sites can legitimately vary by object because the fitted line
    # set depends on spectral coverage.  Concatenation forms their union and
    # fills fields absent from a shard with NaN.  Preserve deterministic
    # first-seen column order rather than requiring identical shard ordering.
    frame = pd.concat(frames, ignore_index=True, sort=False).reindex(
        columns=column_order
    )
    draw_array = np.concatenate(draws, axis=0)
    count_array = np.concatenate(counts, axis=0)
    keep = _dedup_row_indices(frame, dedup_keys or [])
    return SpectraCatalog(
        frame=frame.iloc[keep].reset_index(drop=True),
        fraction_draws=draw_array[keep],
        valid_count=count_array[keep],
        bands=PSF_AGN_FRACTION_BANDS,
    )


def enrich_h5_catalog_rows(catalog, enrichment):
    """Run record enrichment and restore the exact fraction-draw row order."""

    marker = "_qvc_fraction_draw_row_index"
    if marker in catalog.frame.columns:
        raise ValueError(f"Reserved alignment column {marker!r} is already present.")
    frame = catalog.frame.copy()
    frame[marker] = np.arange(len(frame), dtype=int)
    enriched = pd.DataFrame.from_records(enrichment(frame.to_dict("records")))
    if marker not in enriched.columns:
        raise ValueError("Spectral enrichment discarded the HDF5 row-alignment marker.")
    marker_values = pd.to_numeric(enriched[marker], errors="coerce").to_numpy()
    expected = np.arange(len(frame), dtype=int)
    if len(enriched) != len(frame) or set(marker_values.tolist()) != set(expected.tolist()):
        raise ValueError("Spectral enrichment added, removed, or duplicated HDF5 catalog rows.")
    enriched = (
        enriched.sort_values(marker, kind="stable")
        .drop(columns=[marker])
        .reset_index(drop=True)
    )
    return SpectraCatalog(
        frame=enriched,
        fraction_draws=catalog.fraction_draws,
        valid_count=catalog.valid_count,
        bands=catalog.bands,
    )


def _normalize_run2d(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if text.startswith("b'") and text.endswith("'"):
        text = text[2:-1]
    text = text.strip()
    if not text:
        return pd.NA
    return text


def populate_sdss_run2d_from_fits(quasars, fits_path):
    """Populate SDSS_RUN2D using DR17 specObj keys (plate, mjd, fiberid/fiber)."""
    if not quasars:
        return list(quasars)

    df = pd.DataFrame.from_records(quasars)
    out = df.copy()

    key_cols = ("plate", "mjd")
    has_fiberid = "fiberid" in out.columns
    has_fiber = "fiber" in out.columns
    if not all(c in out.columns for c in key_cols) or (not has_fiberid and not has_fiber):
        out["SDSS_RUN2D"] = pd.NA
        print(
            "[WARNING] Cannot populate SDSS_RUN2D: merged data is missing required key columns "
            "(need plate, mjd, and fiberid or fiber)."
        )
        return out.to_dict("records")

    if has_fiberid:
        fiber_col = "fiberid"
    else:
        fiber_col = "fiber"

    try:
        with fits.open(fits_path, memmap=True) as hdul:
            data = hdul[1].data
            table = pd.DataFrame(
                {
                    "plate": data["PLATE"],
                    "mjd": data["MJD"],
                    "fiberid": data["FIBERID"],
                    "SDSS_RUN2D": data["RUN2D"],
                }
            )
    except Exception as exc:
        out["SDSS_RUN2D"] = pd.NA
        print(f"[WARNING] Could not read SDSS RUN2D FITS file {fits_path}: {exc}")
        return out.to_dict("records")

    for col in ("plate", "mjd", "fiberid"):
        table[col] = pd.to_numeric(table[col], errors="coerce").astype("Int64")
    table["SDSS_RUN2D"] = table["SDSS_RUN2D"].apply(_normalize_run2d).astype("string")
    table = table.drop_duplicates(subset=["plate", "mjd", "fiberid"], keep="first")

    out["plate"] = pd.to_numeric(out["plate"], errors="coerce").astype("Int64")
    out["mjd"] = pd.to_numeric(out["mjd"], errors="coerce").astype("Int64")
    out["_fiber_merge_key"] = pd.to_numeric(out[fiber_col], errors="coerce").astype("Int64")

    merged = out.merge(
        table.rename(columns={"fiberid": "_fiber_merge_key"}),
        on=["plate", "mjd", "_fiber_merge_key"],
        how="left",
    )
    merged = merged.drop(columns=["_fiber_merge_key"])
    n_matched = int(pd.notna(merged["SDSS_RUN2D"]).sum())
    print(
        f"Populated SDSS_RUN2D from {fits_path}: matched {n_matched} / {len(merged)} rows."
    )
    return merged.to_dict("records")


def main():
    p = argparse.ArgumentParser(
        description=(
            "Merge JAXSED HDF5 shards (or legacy CSV shards) found in "
            "<base_dir>/<prefix>/."
        )
    )
    p.add_argument(
        "prefix",
        type=str,
        help=(
            "Subdirectory under --base-dir containing top-level shard files. "
            "Also used for the default output name."
        ),
    )
    p.add_argument(
        "--base-dir",
        "-b",
        type=str,
        default="results/data",
        help="Base directory that contains <prefix> shard files. Default: results/data",
    )
    p.add_argument(
        "--expected",
        "-N",
        type=int,
        default=None,
        help="Expected number of rows per input CSV shard. If set, non-matching shards are skipped.",
    )
    p.add_argument(
        "--skip-populate-sdss",
        action="store_true",
        default=False,
        help="Skip populate_sdss_fields before writing.",
    )
    p.add_argument(
        "--populate_sdss_run2d_file",
        nargs="?",
        const="data/SDSS_DR17/specObj-dr17.fits",
        default=None,
        help=(
            "Populate SDSS_RUN2D using RUN2D from the given SDSS DR17 specObj FITS file. "
            "If passed without a value, defaults to data/SDSS_DR17/specObj-dr17.fits."
        ),
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Explicit output path. Defaults to .h5 for HDF5 shards and .csv for legacy shards.",
    )
    p.add_argument(
        "--dedup-keys",
        type=str,
        nargs="*",
        default=["object_id", "run_label"],
        help=(
            "Keys to use for de-duplication across shards (last occurrence wins). "
            "Pass --dedup-keys with no values to disable."
        ),
    )
    args = p.parse_args()

    shard_dir = os.path.join(args.base_dir, args.prefix)
    h5_files = sorted(glob.glob(os.path.join(shard_dir, "*.h5"))) + sorted(
        glob.glob(os.path.join(shard_dir, "*.hdf5"))
    )
    csv_files = sorted(glob.glob(os.path.join(shard_dir, "*.csv")))
    if h5_files and csv_files:
        print(f"Mixed HDF5 and CSV shards are not supported in {shard_dir}.")
        sys.exit(1)
    file_list = h5_files or csv_files
    use_h5 = bool(h5_files)
    if not file_list:
        print(f"No input HDF5 or CSV shards found in {shard_dir}.")
        sys.exit(1)

    kind = "HDF5" if use_h5 else "legacy CSV"
    print(f"Discovered {len(file_list)} top-level {kind} shard(s) in {shard_dir}.")

    out_path = args.out
    if not out_path:
        suffix = ".h5" if use_h5 else ".csv"
        out_path = os.path.join(args.base_dir, f"{args.prefix}{suffix}")

    print(f"Output: {out_path}")

    if use_h5:
        before_catalog = load_and_merge_h5(file_list, expected_n=args.expected, dedup_keys=[])
        merged_catalog = load_and_merge_h5(
            file_list,
            expected_n=args.expected,
            dedup_keys=args.dedup_keys,
        )
        all_quasars = merged_catalog.frame.to_dict("records")
        print(f"Loaded total of {len(before_catalog.frame)} rows from {len(file_list)} shards.")
        if args.dedup_keys:
            print(
                f"De-duplicated by {args.dedup_keys}: "
                f"{len(before_catalog.frame)} -> {len(all_quasars)}"
            )
    else:
        all_quasars = load_and_merge_csv(file_list, expected_n=args.expected)
        print(f"Loaded total of {len(all_quasars)} rows from {len(file_list)} shards.")
        if args.dedup_keys:
            before = len(all_quasars)
            all_quasars = deduplicate(all_quasars, keys=args.dedup_keys)
            print(f"De-duplicated by {args.dedup_keys}: {before} -> {len(all_quasars)}")

    if not args.skip_populate_sdss and all_quasars:
        print("Populating SDSS fields...")
        if use_h5:
            merged_catalog = enrich_h5_catalog_rows(
                merged_catalog,
                populate_sdss_fields,
            )
            all_quasars = merged_catalog.frame.to_dict("records")
        else:
            all_quasars = populate_sdss_fields(all_quasars)

    if args.populate_sdss_run2d_file and all_quasars:
        try:
            fits_path = resolve_qvc_data_path(args.populate_sdss_run2d_file)
        except FileNotFoundError:
            fits_path = args.populate_sdss_run2d_file
        if not os.path.exists(fits_path):
            print(
                f"[WARNING] --populate_sdss_run2d_file requested, but file not found: {fits_path}. "
                "Continuing without SDSS_RUN2D enrichment."
            )
        else:
            if use_h5:
                merged_catalog = enrich_h5_catalog_rows(
                    merged_catalog,
                    lambda rows: populate_sdss_run2d_from_fits(rows, fits_path),
                )
                all_quasars = merged_catalog.frame.to_dict("records")
            else:
                all_quasars = populate_sdss_run2d_from_fits(all_quasars, fits_path)

    if use_h5:
        frame = pd.DataFrame.from_records(all_quasars)
        provenance = build_run_record(
            "qvc.spectra.merge_results",
            args,
            input_paths={f"shard_{index}": path for index, path in enumerate(file_list)},
            event_type="merge",
        )
        write_spectra_catalog_hdf5(
            out_path,
            frame,
            merged_catalog.fraction_draws,
            merged_catalog.valid_count,
            provenance=provenance,
        )
        print(f"Wrote {len(frame)} rows to merged HDF5 {out_path}")
    else:
        seen = []
        seen_set = set()
        for q in all_quasars:
            for k in q.keys():
                if k not in seen_set:
                    seen_set.add(k)
                    seen.append(k)
        write_quasars_to_csv(all_quasars, out_path, fields=seen)

    print("Done.")


if __name__ == "__main__":
    main()
