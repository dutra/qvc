#!/usr/bin/env python
import argparse
import shutil
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Copy files from a folder when their filename contains an object_id "
            "listed in a CSV."
        )
    )
    parser.add_argument("csv_path", help="CSV file containing an object_id column")
    parser.add_argument("source_folder", help="Folder to recursively scan for files")
    parser.add_argument("destination_folder", help="Folder to copy matched files into")
    parser.add_argument(
        "--object-id-column",
        default="object_id",
        help="Name of the CSV column containing object IDs (default: object_id)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matching copies without writing files",
    )
    return parser.parse_args()


def load_object_ids(csv_path, column):
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        columns = ", ".join(df.columns)
        raise ValueError(f"Column {column!r} not found in CSV. Available columns: {columns}")

    object_ids = (
        df[column]
        .dropna()
        .astype(str)
        .map(str.strip)
    )
    return sorted({object_id for object_id in object_ids if object_id})


def matching_object_id(filename, object_ids):
    for object_id in object_ids:
        if object_id in filename:
            return object_id
    return None


def copy_matching_files(csv_path, source_folder, destination_folder, object_id_column, dry_run):
    csv_path = Path(csv_path)
    source_folder = Path(source_folder)
    destination_folder = Path(destination_folder)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not source_folder.is_dir():
        raise NotADirectoryError(f"Source folder not found: {source_folder}")

    object_ids = load_object_ids(csv_path, object_id_column)
    files_scanned = 0
    files_copied = 0

    for source_path in source_folder.rglob("*"):
        if not source_path.is_file():
            continue

        files_scanned += 1
        object_id = matching_object_id(source_path.name, object_ids)
        if object_id is None:
            continue

        relative_path = source_path.relative_to(source_folder)
        destination_path = destination_folder / relative_path

        if dry_run:
            print(f"[dry-run] {source_path} -> {destination_path} (object_id={object_id})")
        else:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)

        files_copied += 1

    print(f"Object IDs loaded: {len(object_ids)}")
    print(f"Files scanned: {files_scanned}")
    print(f"Files copied: {files_copied}")
    print(f"Dry run: {dry_run}")


def main():
    args = parse_args()
    copy_matching_files(
        csv_path=args.csv_path,
        source_folder=args.source_folder,
        destination_folder=args.destination_folder,
        object_id_column=args.object_id_column,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
