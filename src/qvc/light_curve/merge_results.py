#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import pickle
import sys

import h5py
import numpy as np
from tqdm import tqdm

from qvc.hubble.hubble_utils import read_quasars_from_hdf5_flat, populate_sdss_fields
from qvc.light_curve.fit_light_curves import make_lc
from qvc.light_curve.multiband_generate_lc import concat_light_curves
#from qvc.light_curve.multiband_generate_lc import populate_sdss_fields

def read_quasars_from_h5(h5_path):
    return read_quasars_from_hdf5_flat(h5_path)


def _decode_h5_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
    return value


def _read_h5_scalar_key(file_path, key):
    """
    Read a scalar/length-1 dataset from root and return string value.
    Returns empty string when missing/unreadable.
    """
    try:
        with h5py.File(file_path, "r") as hdf:
            if key not in hdf:
                print(f"WARNING: Missing metadata key '{key}' in {file_path}; using empty string.")
                return ""

            data = hdf[key][...]
            arr = np.asarray(data)
            if arr.ndim == 0:
                value = _decode_h5_scalar(arr.item())
                return "" if value is None else str(value)

            if arr.ndim == 1 and arr.shape[0] == 1:
                value = _decode_h5_scalar(arr[0])
                return "" if value is None else str(value)

            print(
                f"WARNING: Metadata key '{key}' in {file_path} is not scalar/length-1 "
                f"(shape={arr.shape}); using empty string."
            )
            return ""
    except Exception as exc:
        print(f"WARNING: Failed reading metadata key '{key}' from {file_path}: {exc}")
        return ""


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


def _is_string_like(v):
    return isinstance(v, (str, bytes))


def _is_missing(v):
    return v is None


def _build_flat_column(values, string_dt):
    has_string = any(_is_string_like(v) for v in values if v is not None)
    if has_string:
        out = []
        for v in values:
            if v is None:
                out.append("")
            elif isinstance(v, bytes):
                out.append(v.decode("utf-8", errors="replace"))
            else:
                out.append(str(v))
        return h5py.string_dtype(encoding="utf-8"), out

    out = []
    for v in values:
        if _is_missing(v):
            out.append(float("nan"))
        else:
            try:
                out.append(float(v))
            except Exception:
                out.append(float("nan"))
    return float, out


def write_quasars_to_h5_flat(quasars, h5_path):
    os.makedirs(os.path.dirname(h5_path) or ".", exist_ok=True)
    string_dt = h5py.string_dtype(encoding="utf-8")

    all_fields = []
    seen = set()
    for q in quasars:
        for k in q.keys():
            if k not in seen:
                seen.add(k)
                all_fields.append(k)

    with h5py.File(h5_path, "w") as hdf:
        for field in all_fields:
            values = [q.get(field, None) for q in quasars]
            dtype, col = _build_flat_column(values, string_dt)
            hdf.create_dataset(field, data=col, dtype=dtype)

    print(f"Wrote {len(quasars)} rows to flat HDF5 {h5_path}")


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
            if v is None or v == "":
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
        merged[comp_key] = q

    return [merged[k] for k in order]


def load_and_merge_h5(file_list, expected_n, load_n):
    all_quasars = []
    if load_n is not None:
        file_list = file_list[:load_n]
    for path in tqdm(file_list, desc="Merging HDF5 shards", unit="file"):
        try:
            source_git_commit = _read_h5_scalar_key(path, "git_commit")
            source_run_datetime = _read_h5_scalar_key(path, "run_datetime")
            qs_df = read_quasars_from_h5(path)
            if not enforce_expected_count(len(qs_df), expected_n, path):
                continue
            qs = qs_df.to_dict("records")
            for row in qs:
                row["git_commit"] = source_git_commit
                row["run_datetime"] = source_run_datetime
            all_quasars.extend(qs)
        except Exception as e:
            print(f"ERROR reading {path}: {e}")
            continue
    return all_quasars


def attach_variability_metrics(rows):
    """Reload raw light curves and attach authoritative corrected variability metrics."""

    object_ids = [str(row["object_id"]) for row in rows if row.get("object_id") not in (None, "")]
    unique_object_ids = list(dict.fromkeys(object_ids))
    reloaded = concat_light_curves(filter_object_ids=unique_object_ids, progress_bar=False)
    reloaded_by_object_id = {str(obj["object_id"]): obj for obj in reloaded}

    missing_object_ids = [oid for oid in unique_object_ids if oid not in reloaded_by_object_id]
    if missing_object_ids:
        raise ValueError(
            "Failed to recompute variability with concat_light_curves; "
            f"missing object_ids: {missing_object_ids}"
        )

    enriched_rows = []
    unusable_object_ids = []
    for row in rows:
        object_id = str(row["object_id"])
        lc = make_lc(
            reloaded_by_object_id[object_id],
            bands=["u", "g", "r", "i", "z"],
            inject_fake=False,
            drop_band_lyman_alpha=False,
            verbose=False,
        )
        if lc is None:
            unusable_object_ids.append(object_id)
            continue

        enriched = dict(row)
        for key, value in lc.items():
            if key.startswith("variability_"):
                enriched[key] = value
        enriched_rows.append(enriched)

    if unusable_object_ids:
        raise ValueError(
            "Failed to recompute variability because make_lc returned None for "
            f"object_ids: {unusable_object_ids}"
        )

    return enriched_rows


def main():
    p = argparse.ArgumentParser(
        description=(
            "Merge flat HDF5 shards (*.h5) found in <base_dir>/<prefix>/ "
            "and write merged output as flat HDF5 (or CSV)."
        )
    )
    p.add_argument(
        "prefix",
        type=str,
        help="Subdirectory under --base-dir containing top-level *.h5 shards. Also used for default output name.",
    )
    p.add_argument(
        "--base-dir",
        "-b",
        type=str,
        default="results/data",
        help="Base directory that contains <prefix>/*.h5. Default: results/data",
    )
    p.add_argument(
        "--expected",
        "-E",
        type=int,
        default=None,
        help="Expected number of rows per input HDF5 shard. If set, non-matching shards are skipped.",
    )
    p.add_argument(
        "--N",
        "-N",
        type=int,
        default=None,
        help="Total number of shards to load.",
    )
    p.add_argument(
        "--skip-populate-sdss",
        action="store_true",
        default=False,
        help="Skip populate_sdss_fields before writing.",
    )
    p.add_argument(
        "--compute-variability",
        action="store_true",
        default=False,
        help="Reload S82 light curves via concat_light_curves and recompute corrected per-band variability metrics.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Explicit output path. If omitted, defaults to <base_dir>/<prefix>.<ext>, where <ext> is derived from --out-format.",
    )
    p.add_argument(
        "--out-format",
        choices=["h5", "csv"],
        default=None,
        help="Output format. If omitted and --out is given, inferred from its extension. If both omitted, defaults to .h5.",
    )
    p.add_argument(
        "--dedup-keys",
        type=str,
        nargs="*",
        default=["object_id", "run_label"],
        help="Keys to use for de-duplication across shards (last occurrence wins). Set to '' to disable.",
    )
    
    args = p.parse_args()

    shard_dir = os.path.join(args.base_dir, args.prefix)
    file_list = sorted(glob.glob(os.path.join(shard_dir, "*.h5")))
    if not file_list:
        print(f"No input shards found in {shard_dir} matching *.h5.")
        sys.exit(1)

    print(f"Discovered {len(file_list)} top-level HDF5 shard(s) in {shard_dir}.")

    out_format = args.out_format
    out_path = args.out

    if out_path and not out_format:
        ext = os.path.splitext(out_path)[1].lower()
        if ext == ".h5":
            out_format = "h5"
        elif ext == ".csv":
            out_format = "csv"
        else:
            print("ERROR: Cannot infer --out-format from --out extension. Use .h5 or .csv or pass --out-format.")
            sys.exit(1)

    if not out_format:
        out_format = "h5"
    if not out_path:
        out_path = os.path.join(args.base_dir, f"{args.prefix}.{out_format}")

    print(f"Output: {out_path} (format: {out_format.upper()})")

    all_quasars = load_and_merge_h5(file_list, expected_n=args.expected, load_n=args.N)
    print(f"Loaded total of {len(all_quasars)} rows from {len(file_list)} shards.")

    dedup_keys = args.dedup_keys
    if dedup_keys:
        before = len(all_quasars)
        all_quasars = deduplicate(all_quasars, keys=dedup_keys)
        print(f"De-duplicated by '{dedup_keys}': {before} -> {len(all_quasars)}")

    if args.compute_variability and all_quasars:
        print("Computing corrected variability metrics from merged rows...")
        all_quasars = attach_variability_metrics(all_quasars)

    if not args.skip_populate_sdss and all_quasars:
        print("Populating SDSS fields...")
        all_quasars = populate_sdss_fields(all_quasars)
        if all_quasars and "plate" in all_quasars[0]:
            print(all_quasars[0]["plate"])
    if out_format == "csv":
        seen = []
        s = set()
        for q in all_quasars:
            for k in q.keys():
                if k not in s:
                    s.add(k)
                    seen.append(k)
        write_quasars_to_csv(all_quasars, out_path, fields=seen)
    elif out_format == "h5":
        write_quasars_to_h5_flat(all_quasars, out_path)
    else:
        print(f"ERROR: Unsupported out-format: {out_format}")
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
