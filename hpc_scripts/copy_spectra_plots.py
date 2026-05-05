#!/usr/bin/env python
import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd


DIAGNOSTIC_SUFFIXES = ("_corner", "_trace")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Copy main spectra plot PDFs from a folder for object_id rows listed "
            "in a CSV."
        )
    )
    parser.add_argument("csv_path", help="CSV file containing an object_id column")
    parser.add_argument("source_folder", help="Folder to recursively scan for spectra PDFs")
    parser.add_argument("destination_folder", help="Folder to copy matched PDFs into")
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
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Exit with an error if any requested object_id has no matching PDF",
    )
    return parser.parse_args()


def normalize_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_rows(csv_path, object_id_column):
    df = pd.read_csv(csv_path)
    if object_id_column not in df.columns:
        columns = ", ".join(df.columns)
        raise ValueError(
            f"Column {object_id_column!r} not found in CSV. Available columns: {columns}"
        )

    df = df.copy()
    df[object_id_column] = df[object_id_column].map(normalize_value)
    df = df[df[object_id_column] != ""]
    return df.drop_duplicates(subset=[object_id_column], keep="first").reset_index(drop=True)


def is_main_spectra_pdf(path):
    if path.suffix.lower() != ".pdf":
        return False
    return not any(suffix in path.stem for suffix in DIAGNOSTIC_SUFFIXES)


def candidate_stems_for_row(row):
    stems = []

    filename = normalize_value(row.get("filename"))
    if filename:
        stems.append(Path(filename).stem)

    sdss_name = normalize_value(row.get("sdss_name"))
    if sdss_name:
        for z_column in ("z", "redshift"):
            z = pd.to_numeric(row.get(z_column), errors="coerce")
            if pd.notna(z):
                stems.append(f"z{float(z):.3f}_{sdss_name}")
                break

    return list(dict.fromkeys(stems))


def build_pdf_index(source_folder):
    by_stem = defaultdict(list)
    pdf_paths = []
    pdfs_scanned = 0

    for path in source_folder.rglob("*.pdf"):
        if not path.is_file() or not is_main_spectra_pdf(path):
            continue

        pdfs_scanned += 1
        pdf_paths.append(path)
        by_stem[path.stem].append(path)

    for paths in by_stem.values():
        paths.sort()

    return dict(by_stem), sorted(pdf_paths), pdfs_scanned


def find_metadata_pdf_for_row(row, by_stem):
    for stem in candidate_stems_for_row(row):
        candidates = by_stem.get(stem, [])
        if candidates:
            return candidates[0], "metadata"

    return None, None


def copy_pdf(source_path, source_folder, destination_folder, object_id, match_source, dry_run):
    relative_path = source_path.relative_to(source_folder)
    destination_path = destination_folder / relative_path

    if dry_run:
        print(
            f"[dry-run] {source_path} -> {destination_path} "
            f"(object_id={object_id}, match={match_source})"
        )
    else:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def matching_object_id(filename, object_ids):
    for object_id in object_ids:
        if object_id in filename:
            return object_id
    return None


def build_fallback_matches(pdf_paths, object_ids):
    object_ids = list(dict.fromkeys(object_ids))
    matches = {}

    for path in pdf_paths:
        object_id = matching_object_id(path.name, object_ids)
        if object_id is not None and object_id not in matches:
            matches[object_id] = path

    return matches


def copy_spectra_plots(
    csv_path,
    source_folder,
    destination_folder,
    object_id_column,
    dry_run,
    require_all,
):
    csv_path = Path(csv_path)
    source_folder = Path(source_folder)
    destination_folder = Path(destination_folder)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not source_folder.is_dir():
        raise NotADirectoryError(f"Source folder not found: {source_folder}")

    rows = load_rows(csv_path, object_id_column)
    by_stem, pdf_paths, pdfs_scanned = build_pdf_index(source_folder)

    copied_paths = set()
    unresolved_rows = []
    missing_object_ids = []
    files_copied = 0

    for _, row in rows.iterrows():
        object_id = normalize_value(row[object_id_column])
        source_path, match_source = find_metadata_pdf_for_row(row, by_stem)
        if source_path is None:
            unresolved_rows.append(row)
            continue

        if source_path in copied_paths:
            continue

        copy_pdf(
            source_path=source_path,
            source_folder=source_folder,
            destination_folder=destination_folder,
            object_id=object_id,
            match_source=match_source,
            dry_run=dry_run,
        )
        copied_paths.add(source_path)
        files_copied += 1

    fallback_object_ids = [
        normalize_value(row[object_id_column])
        for row in unresolved_rows
    ]
    fallback_matches = build_fallback_matches(pdf_paths, fallback_object_ids)

    for row in unresolved_rows:
        object_id = normalize_value(row[object_id_column])
        source_path = fallback_matches.get(object_id)
        if source_path is None:
            missing_object_ids.append(object_id)
            continue

        if source_path in copied_paths:
            continue

        copy_pdf(
            source_path=source_path,
            source_folder=source_folder,
            destination_folder=destination_folder,
            object_id=object_id,
            match_source="object_id",
            dry_run=dry_run,
        )
        copied_paths.add(source_path)
        files_copied += 1

    print(f"Object IDs loaded: {len(rows)}")
    print(f"PDFs scanned: {pdfs_scanned}")
    print(f"PDFs copied: {files_copied}")
    print(f"Missing object IDs: {len(missing_object_ids)}")
    if missing_object_ids:
        preview = ", ".join(missing_object_ids[:20])
        print(f"First missing object IDs: {preview}")
    print(f"Dry run: {dry_run}")

    if require_all and missing_object_ids:
        raise SystemExit(1)


def main():
    args = parse_args()
    copy_spectra_plots(
        csv_path=args.csv_path,
        source_folder=args.source_folder,
        destination_folder=args.destination_folder,
        object_id_column=args.object_id_column,
        dry_run=args.dry_run,
        require_all=args.require_all,
    )


if __name__ == "__main__":
    main()
