#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import pickle
import sys

import h5py
from tqdm import tqdm

from qvc.hubble.hubble_utils import populate_sdss_fields, read_quasars_from_hdf5_flat


def read_quasars_from_h5(h5_path):
    return read_quasars_from_hdf5_flat(h5_path)


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


def load_and_merge_h5(file_list, expected_n):
    all_quasars = []
    for path in tqdm(file_list, desc="Merging HDF5 shards", unit="file"):
        try:
            qs = read_quasars_from_h5(path)
            if not enforce_expected_count(len(qs), expected_n, path):
                continue
            all_quasars.extend(qs)
        except Exception as e:
            print(f"ERROR reading {path}: {e}")
            continue
    return all_quasars


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
        "-N",
        type=int,
        default=None,
        help="Expected number of rows per input HDF5 shard. If set, non-matching shards are skipped.",
    )
    p.add_argument(
        "--skip-populate-sdss",
        action="store_true",
        default=False,
        help="Skip populate_sdss_fields before writing.",
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

    all_quasars = load_and_merge_h5(file_list, expected_n=args.expected)
    print(f"Loaded total of {len(all_quasars)} rows from {len(file_list)} shards.")

    dedup_keys = args.dedup_keys
    if dedup_keys:
        before = len(all_quasars)
        all_quasars = deduplicate(all_quasars, keys=dedup_keys)
        print(f"De-duplicated by '{dedup_keys}': {before} -> {len(all_quasars)}")

    if not args.skip_populate_sdss and all_quasars:
        print("Populating SDSS fields...")
        populate_sdss_fields(all_quasars)

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
        with open(out_path + ".pkl", "wb") as f:
            pickle.dump(all_quasars, f)
        print(f"Wrote {len(all_quasars)} rows to pickle file {out_path}.pkl")
    else:
        print(f"ERROR: Unsupported out-format: {out_format}")
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
