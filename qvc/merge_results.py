#!/usr/bin/env python3
import argparse
import glob
import os
import re
import sys
import csv
from tqdm import tqdm

# External deps
import h5py  # required only if reading .h5 directly via hubble_utils
from hubble_utils import (
    populate_sdss_fields,
    read_quasars_from_hdf5,
    write_hdf5_file,
)

def parse_job_id_from_path(path):
    """Extract integer job id from filenames like 'job57.h5' or 'job57.csv'. Returns None if no match."""
    base = os.path.basename(path)
    m = re.search(r"^job(\d+)\.(h5|csv)$", base, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

# ----------------------------
# I/O helpers (uniform: list[dict])
# ----------------------------
def read_quasars_from_csv(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows

def read_quasars_from_h5(h5_path):
    # Your util returns list[dict]
    return read_quasars_from_hdf5(h5_path)

def write_quasars_to_csv(quasars, csv_path, fields=None):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    if not quasars:
        # No rows: write header only if fields provided, else empty file
        with open(csv_path, "w", newline="") as f:
            if fields:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
        print(f"Wrote empty CSV to {csv_path}")
        return

    # Determine columns
    if fields is None:
        # union of keys across rows, stable-ish order
        seen = []
        s = set()
        for q in quasars:
            for k in q.keys():
                if k not in s:
                    s.add(k); seen.append(k)
        fields = seen

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for q in quasars:
            writer.writerow({k: q.get(k, "") for k in fields})
    print(f"Wrote {len(quasars)} rows to {csv_path}")

def write_quasars_to_h5(quasars, h5_path):
    os.makedirs(os.path.dirname(h5_path) or ".", exist_ok=True)
    write_hdf5_file(quasars, h5_path)
    print(f"Wrote {len(quasars)} objects to {h5_path}")

# ----------------------------
# Merge logic
# ----------------------------
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
    """
    Deduplicate by the combination of `keys`.
    If any key is missing/empty in a record, that record is treated as unique (via id).
    Last occurrence wins for a given composite key.
    """
    if not keys:
        return list(quasars)
    if isinstance(keys, (str, bytes)):
        keys = [keys]

    merged = {}
    order = []

    for q in quasars:
        # Build composite key; if any component missing/empty -> unique by object id
        vals = []
        for k in keys:
            v = q.get(k)
            if v is None or v == "":
                comp_key = ("__objid__", id(q))
                break
            vals.append(v)
        else:
            # All present; ensure hashable composite key
            try:
                comp_key = tuple(vals)
                hash(comp_key)
            except TypeError:
                comp_key = tuple(repr(v) for v in vals)

        if comp_key not in merged:
            order.append(comp_key)
        merged[comp_key] = q  # change to: if comp_key not in merged: merged[comp_key] = q  -> keep first

    return [merged[k] for k in order]


def load_and_merge(file_list, in_format, expected_n):
    """
    Returns list[dict] of all quasars from the shards.
    """
    all_quasars = []
    for path in tqdm(file_list, desc="Merging shards", unit="file"):
        try:
            if in_format == "h5":
                qs = read_quasars_from_h5(path)
            elif in_format == "csv":
                qs = read_quasars_from_csv(path)
            else:
                raise ValueError("in_format must be 'h5' or 'csv'")

            if not enforce_expected_count(len(qs), expected_n, path):
                continue

            all_quasars.extend(qs)
        except Exception as e:
            print(f"ERROR reading {path}: {e}")
            continue
    return all_quasars

def detect_input_format(sample_files, forced):
    if forced in ("h5", "csv"):
        return forced
    # Auto-detect by extension of first file
    if not sample_files:
        raise RuntimeError("No input files to auto-detect input format.")
    ext = os.path.splitext(sample_files[0])[1].lower()
    if ext == ".h5":
        return "h5"
    if ext == ".csv":
        return "csv"
    raise RuntimeError(f"Unrecognized input extension: {ext}")

def main():
    p = argparse.ArgumentParser(
        description=(
            "Merge CSV or HDF5 shards (job*.{h5,csv}) found in <base_dir>/<prefix>/ "
            "and write a single merged output as either .h5 or .csv."
        )
    )
    p.add_argument("prefix", type=str,
                   help="Subdirectory under --base-dir containing job*.{h5,csv}. Also used for default output name.")
    p.add_argument("--base-dir", "-b", type=str, default="results/data",
                   help="Base directory that contains <prefix>/job*.{h5,csv}. Default: results/data")
    p.add_argument("--exclude-jobs", "-x", type=int, nargs="*", default=[],
                   help="Space-separated list of job IDs to exclude (from filenames like job57.h5 / job57.csv)")
    p.add_argument("--expected", "-N", type=int, default=None,
                   help="Expected number of objects per input shard (rows for CSV, top-level objects for H5). If set, non-matching shards are skipped.")
    p.add_argument("--skip-populate-sdss", action="store_true", default=False,
                   help="Skip populate_sdss_fields before writing.")
    p.add_argument("--in-format", choices=["auto", "h5", "csv"], default="auto",
                   help="Force input format; default auto-detect from file extension.")
    p.add_argument("--out", type=str, default=None,
                   help="Explicit output path. If omitted, defaults to <base_dir>/<prefix>.<ext>, "
                        "where <ext> is derived from --out-format.")
    p.add_argument("--out-format", choices=["h5", "csv"], default=None,
                   help="Output format. If omitted and --out is given, inferred from its extension. "
                        "If both omitted, defaults to .h5.")
    p.add_argument("--dedup-keys", type=str, nargs="*", default=["object_id", "run_label"],
                   help="Keys to use for de-duplication across shards (last occurrence wins). Set to '' to disable. Default: object_id")

    args = p.parse_args()

    shard_dir = os.path.join(args.base_dir, args.prefix)
    # Gather both CSV and H5; we'll filter by detected/forced input format
    files_h5  = sorted(glob.glob(os.path.join(shard_dir, "job*.h5")))
    files_csv = sorted(glob.glob(os.path.join(shard_dir, "job*.csv")))
    n_h5, n_csv = len(files_h5), len(files_csv)
    if n_h5 + n_csv == 0:
        print(f"No input shards found in {shard_dir} matching job*.h5 or job*.csv.")
        sys.exit(1)

    # Determine input format
    in_format = detect_input_format(files_h5 + files_csv, args.in_format if args.in_format != "auto" else None)

    file_list = files_h5 if in_format == "h5" else files_csv
    print(f"Detected input format: {in_format.upper()}  ({len(file_list)} candidate files)")

    # Exclude jobs
    if args.exclude_jobs:
        before = len(file_list)
        kept = []
        for f in file_list:
            jid = parse_job_id_from_path(f)
            if jid is not None and jid in args.exclude_jobs:
                continue
            kept.append(f)
        file_list = kept
        print(f"After excluding jobs {args.exclude_jobs}, {len(file_list)} files remain (filtered {before - len(file_list)}).")

    if not file_list:
        print("No input files remaining after filtering. Nothing to do.")
        sys.exit(1)

    # Decide output format & path
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
        out_format = "h5"  # default

    if not out_path:
        out_path = os.path.join(args.base_dir, f"{args.prefix}.{out_format}")

    print(f"Output: {out_path} (format: {out_format.upper()})")

    # Load + merge
    all_quasars = load_and_merge(file_list, in_format=in_format, expected_n=args.expected)
    print(f"Loaded total of {len(all_quasars)} rows/objects from {len(file_list)} shards.")

    # De-duplicate
    dedup_keys = args.dedup_keys
    if dedup_keys:
        before = len(all_quasars)
        all_quasars = deduplicate(all_quasars, keys=dedup_keys)
        print(f"De-duplicated by '{dedup_keys}': {before} -> {len(all_quasars)}")

    # Populate SDSS fields (if desired)
    if not args.skip_populate_sdss and all_quasars:
        print("Populating SDSS fields...")
        populate_sdss_fields(all_quasars)

    # Write output
    if out_format == "csv":
        # Choose columns: union of keys, stable order
        seen = []
        s = set()
        for q in all_quasars:
            for k in q.keys():
                if k not in s:
                    s.add(k); seen.append(k)
        write_quasars_to_csv(all_quasars, out_path, fields=seen)
    elif out_format == "h5":
        write_quasars_to_h5(all_quasars, out_path)
    else:
        print(f"ERROR: Unsupported out-format: {out_format}")
        sys.exit(1)

    print("Done.")

if __name__ == "__main__":
    main()
