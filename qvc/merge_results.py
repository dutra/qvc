#!/usr/bin/env python3
import argparse
import glob
import os
import re
import sys
import h5py


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

                    if nkeys != expected_n:
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
        default=20,
        help="Expected number of top-level objects per input file. Default: 20",
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


if __name__ == "__main__":
    main()
