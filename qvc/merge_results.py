#!/usr/bin/env python3
import argparse
import glob
import os
import re
import sys
import h5py
import numpy as np
from tqdm import tqdm
import csv
from hubble_utils import populate_sdss_fields, read_quasars_from_hdf5, write_hdf5_file


def merge_hdf5_files(file_list, output_file, expected_n):
    """
    Merge multiple HDF5 files into a single output file.

    Each input file is expected to have exactly `expected_n` top-level groups.
    If a group already exists in the output, it will be overwritten.
    """
    object_ids = set()
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    with h5py.File(output_file, "w") as hdf_out:
        for file_path in file_list:
            try:
                with h5py.File(file_path, "r") as hdf_in:
                    nkeys = len(hdf_in.keys())
                    print(f"File {file_path}: {nkeys} top-level entries")

                    if expected_n is not None and nkeys != expected_n:
                        print(
                            f"WARNING. Skipping file {file_path} with {nkeys} objects "
                            f"(expected {expected_n})"
                        )
                        continue

                    object_ids.update(hdf_in.keys())

                    for group_name in hdf_in.keys():
                        if group_name in hdf_out:
                            del hdf_out[group_name]  # Overwrite existing group
                        hdf_in.copy(group_name, hdf_out)

            except Exception as e:
                print(f"For file: {file_path}, error merging file: {e}")
                continue

    print(f"Merged {len(file_list)} files into {output_file}")
    print(f"Unique object IDs: {len(object_ids)}")

def export_quasars_to_csv(quasars, csv_file, fields):
    """
    Reads quasars from an HDF5 file and writes selected fields to a CSV file.

    Args:
        h5_file (str): Path to the input HDF5 file.
        csv_file (str): Path to the output CSV file.
        fields (list of str): List of field names to write as columns.
    """

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for q in quasars:
            row = {field: q.get(field, "") for field in fields}
            writer.writerow(row)
    print(f"Wrote {len(quasars)} quasars to {csv_file}")

def write_sdss_fields(quasars, h5_file, skip_populate_sdss=False):
    populate_sdss_fields(quasars)
    write_hdf5_file(quasars, h5_file)
    return quasars

def parse_job_id_from_path(path):
    """Extract integer job id from filenames like 'job57.h5'. Returns None if no match."""
    base = os.path.basename(path)
    m = re.search(r"^job(\d+)\.h5$", base)
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge HDF5 shards found at <base_dir>/<prefix>/job*.h5 "
            "into <base_dir>/<prefix>.h5."
        )
    )
    parser.add_argument(
        "prefix",
        type=str,
        help="Subdirectory under --base-dir to search (and the name of the output file).",
    )

    parser.add_argument(
        "--base-dir",
        "-b",
        type=str,
        default="results/data",
        help="Base directory that contains <prefix>/job*.h5. Default: results/data",
    )
    parser.add_argument(
        "--exclude-jobs",
        "-x",
        type=int,
        nargs="*",
        default=[],
        help="Space-separated list of job IDs to exclude (from filenames like job57.h5).",
    )
    parser.add_argument(
        "--expected",
        "-N",
        type=int,
        default=None,
        help="Expected number of top-level objects per input file. Default: 20",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        default=False,
        help="If set, export merged quasars to CSV file.",
    )
    parser.add_argument(
        "--skip_populate_sdss",
        action="store_true",
        default=False,
        help="If set, skip populating SDSS fields.",
    )

    args = parser.parse_args()

    # Find inputs under <base_dir>/<prefix>/job*.h5
    shard_dir = os.path.join(args.base_dir, args.prefix)
    search_pattern = os.path.join(shard_dir, "job*.h5")
    file_list = sorted(glob.glob(search_pattern))
    print(f"Found {len(file_list)} candidate files in {shard_dir} matching job*.h5.")

    # Exclude jobs based on jobNNN.h5 pattern
    if args.exclude_jobs:
        before = len(file_list)
        kept = []
        for f in file_list:
            jid = parse_job_id_from_path(f)
            if jid is not None and jid in args.exclude_jobs:
                continue
            kept.append(f)
        file_list = kept
        print(
            f"After excluding jobs {args.exclude_jobs}, {len(file_list)} files remain "
            f"(filtered {before - len(file_list)})."
        )

    if not file_list:
        print("No input files remaining after filtering. Nothing to do.")
        sys.exit(1)

    # Output to <base_dir>/<prefix>.h5
    output_file = os.path.join(args.base_dir, f"{args.prefix}.h5")

    merge_hdf5_files(
        file_list=file_list,
        output_file=output_file,
        expected_n=args.expected,
    )

    quasars = read_quasars_from_hdf5(output_file)
    print(f"Read {len(quasars)} quasars from {output_file}")

    def filter_quasars(quasars):
        return [q for q in quasars if q.get("f_host_5100", -99) > -90]

    quasars = filter_quasars(quasars)
    print(f"After filtering, {len(quasars)} quasars remain.")

    if not args.skip_populate_sdss:
        print("Populating SDSS fields...")
        quasars = write_sdss_fields(quasars, output_file)
        print(f"Wrote SDSS fields to {output_file}")

    if args.csv:
        export_quasars_to_csv(
            quasars=quasars,
            csv_file=output_file.replace(".h5", ".csv"),
            fields=quasars[0].keys()
        )
        print(f"Wrote CSV file to {output_file.replace('.h5', '.csv')}")
    
if __name__ == "__main__":
    main()
