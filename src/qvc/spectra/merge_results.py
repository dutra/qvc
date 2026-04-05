#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import sys

import pandas as pd
from tqdm import tqdm

from qvc.hubble.hubble_utils import populate_sdss_fields


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


def main():
    p = argparse.ArgumentParser(
        description=(
            "Merge CSV shards (*.csv) found in <base_dir>/<prefix>/ "
            "and write merged output as CSV."
        )
    )
    p.add_argument(
        "prefix",
        type=str,
        help=(
            "Subdirectory under --base-dir containing top-level *.csv shards. "
            "Also used for the default output name."
        ),
    )
    p.add_argument(
        "--base-dir",
        "-b",
        type=str,
        default="results/data",
        help="Base directory that contains <prefix>/*.csv. Default: results/data",
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
        "--out",
        type=str,
        default=None,
        help="Explicit output CSV path. If omitted, defaults to <base_dir>/<prefix>.csv",
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
    file_list = sorted(glob.glob(os.path.join(shard_dir, "*.csv")))

    if not file_list:
        print(f"No input shards found in {shard_dir} matching *.csv.")
        sys.exit(1)

    print(f"Discovered {len(file_list)} top-level CSV shard(s) in {shard_dir}.")

    out_path = args.out
    if not out_path:
        out_path = os.path.join(args.base_dir, f"{args.prefix}.csv")

    print(f"Output: {out_path}")

    all_quasars = load_and_merge_csv(file_list, expected_n=args.expected)
    print(f"Loaded total of {len(all_quasars)} rows from {len(file_list)} shards.")

    dedup_keys = args.dedup_keys
    if dedup_keys:
        before = len(all_quasars)
        all_quasars = deduplicate(all_quasars, keys=dedup_keys)
        print(f"De-duplicated by {dedup_keys}: {before} -> {len(all_quasars)}")

    if not args.skip_populate_sdss and all_quasars:
        print("Populating SDSS fields...")
        all_quasars = populate_sdss_fields(all_quasars)

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