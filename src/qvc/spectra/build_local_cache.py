#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm

from qvc.hubble.hubble_utils import resolve_qvc_data_path


def sdss_spec_filename(plate: int, mjd: int, fiber: int) -> str:
    """Return canonical SDSS spectrum filename: spec-PPPP-MMMMM-FFFF.fits."""
    return f"spec-{int(plate):04d}-{int(mjd):05d}-{int(fiber):04d}.fits"


def sdss_cache_file_path(cache_dir: str | Path, plate: int, mjd: int, fiber: int) -> Path:
    return Path(cache_dir) / sdss_spec_filename(plate=plate, mjd=mjd, fiber=fiber)


def load_sdss_spec_from_cache(cache_dir: str | Path, plate: int, mjd: int, fiber: int):
    cache_file = sdss_cache_file_path(cache_dir=cache_dir, plate=plate, mjd=mjd, fiber=fiber)
    if cache_file.exists():
        return fits.open(cache_file, memmap=False)
    return None


def _coerce_spec_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("plate", "mjd", "fiber"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    if "sdss_name" in df.columns:
        sdss = df["sdss_name"]
        mask = sdss.notna()
        df.loc[mask, "sdss_name"] = sdss.loc[mask].astype(str).str.strip()
    return df


def _has_complete_spec_fields(df: pd.DataFrame) -> bool:
    required = ("plate", "mjd", "fiber", "sdss_name")
    if not all(c in df.columns for c in required):
        return False
    mask = (
        df["plate"].notna()
        & df["mjd"].notna()
        & df["fiber"].notna()
        & df["sdss_name"].notna()
        & (df["sdss_name"].astype(str).str.strip() != "")
        & (df["sdss_name"].astype(str).str.lower().str.strip() != "nan")
    )
    return bool(mask.all())


def _resolve_spec_fields_from_object_id(
    df: pd.DataFrame,
    *,
    catalog_parquet: str,
    dr16q_fits: str,
    max_sep_arcsec: float,
) -> pd.DataFrame:
    if "object_id" not in df.columns:
        raise ValueError("Input CSV must contain an 'object_id' column.")

    out = df.copy()
    out["object_id"] = out["object_id"].astype(str).str.strip()

    cat = pd.read_parquet(resolve_qvc_data_path(catalog_parquet))
    cat_lookup = (
        cat.loc[:, ["objectId", "RA", "DEC"]]
        .dropna(subset=["objectId", "RA", "DEC"])
        .assign(object_id=lambda d: d["objectId"].astype(str))
        .drop_duplicates(subset=["object_id"], keep="first")
        .loc[:, ["object_id", "RA", "DEC"]]
    )
    out = out.merge(cat_lookup, on="object_id", how="left")

    dr_df = _load_dr16q_projection(dr16q_fits)
    dr_df = dr_df.dropna(subset=["RA", "DEC"])
    dr_coords = SkyCoord(
        ra=pd.to_numeric(dr_df["RA"], errors="coerce").to_numpy() * u.deg,
        dec=pd.to_numeric(dr_df["DEC"], errors="coerce").to_numpy() * u.deg,
    )

    valid = out["RA"].notna() & out["DEC"].notna()
    if np.any(valid):
        query = out.loc[valid, ["RA", "DEC"]]
        query_coords = SkyCoord(
            ra=pd.to_numeric(query["RA"], errors="coerce").to_numpy() * u.deg,
            dec=pd.to_numeric(query["DEC"], errors="coerce").to_numpy() * u.deg,
        )
        idx, d2d, _ = query_coords.match_to_catalog_sky(dr_coords)
        matched = d2d < (float(max_sep_arcsec) * u.arcsec)

        query_idx = out.index[valid]
        matched_query_idx = query_idx[np.asarray(matched)]
        matched_dr_idx = np.asarray(idx[np.asarray(matched)], dtype=int)

        if len(matched_query_idx) > 0:
            dr_match = dr_df.iloc[matched_dr_idx].reset_index(drop=True)
            matched_map = pd.DataFrame(
                {
                    "row_idx": np.asarray(matched_query_idx, dtype=int),
                    "plate_resolved": dr_match["PLATE"].to_numpy(),
                    "mjd_resolved": dr_match["MJD"].to_numpy(),
                    "fiber_resolved": dr_match["FIBERID"].to_numpy(),
                    "sdss_name_resolved": dr_match["SDSS_NAME"].astype(str).str.strip().to_numpy(),
                }
            )

            out = out.reset_index().rename(columns={"index": "row_idx"})
            out = out.merge(matched_map, on="row_idx", how="left")

            if "plate" not in out.columns:
                out["plate"] = np.nan
            if "mjd" not in out.columns:
                out["mjd"] = np.nan
            if "fiber" not in out.columns:
                out["fiber"] = np.nan
            if "sdss_name" not in out.columns:
                out["sdss_name"] = np.nan

            out["plate"] = out["plate"].where(out["plate"].notna(), out["plate_resolved"])
            out["mjd"] = out["mjd"].where(out["mjd"].notna(), out["mjd_resolved"])
            out["fiber"] = out["fiber"].where(out["fiber"].notna(), out["fiber_resolved"])
            out["sdss_name"] = out["sdss_name"].where(
                out["sdss_name"].notna() & (out["sdss_name"].astype(str).str.strip() != ""),
                out["sdss_name_resolved"],
            )

            out = out.drop(
                columns=[
                    "plate_resolved",
                    "mjd_resolved",
                    "fiber_resolved",
                    "sdss_name_resolved",
                ]
            )
            out = out.set_index("row_idx").sort_index()
            out.index.name = None

    out = _coerce_spec_columns(out)
    drop_cols = [c for c in ("RA", "DEC") if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out


def _load_dr16q_projection(dr16q_fits: str) -> pd.DataFrame:
    required_cols = ["RA", "DEC", "PLATE", "MJD", "FIBERID", "SDSS_NAME"]
    tbl = Table.read(resolve_qvc_data_path(dr16q_fits), hdu=1)
    missing = [c for c in required_cols if c not in tbl.colnames]
    if missing:
        raise ValueError(
            "DR16Q FITS is missing required columns for cache mapping: "
            f"{missing}. Required columns: {required_cols}"
        )
    return tbl[required_cols].to_pandas()


def build_cache(
    *,
    input_csv: str,
    source_root: str,
    dest_cache_dir: str,
    catalog_parquet: str,
    dr16q_fits: str,
    max_sep_arcsec: float,
    overwrite: bool,
    strict: bool,
    report_csv: str | None,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df = _coerce_spec_columns(df)

    if not _has_complete_spec_fields(df):
        df = _resolve_spec_fields_from_object_id(
            df,
            catalog_parquet=catalog_parquet,
            dr16q_fits=dr16q_fits,
            max_sep_arcsec=max_sep_arcsec,
        )

    source_root_path = Path(source_root)
    dest_root = Path(dest_cache_dir)
    dest_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Copying SDSS spectra"):
        status = "unknown"
        error = ""
        src_path = ""
        dest_path = ""

        plate = row.get("plate")
        mjd = row.get("mjd")
        fiber = row.get("fiber")
        object_id = str(row.get("object_id", ""))

        if pd.isna(plate) or pd.isna(mjd) or pd.isna(fiber):
            status = "missing_mapping"
            error = "plate/mjd/fiber not available"
        else:
            fname = sdss_spec_filename(int(plate), int(mjd), int(fiber))
            src = source_root_path / f"{int(plate)}" / fname
            dst = dest_root / fname
            src_path = str(src)
            dest_path = str(dst)

            if not src.exists():
                status = "missing_source"
                error = "source FITS does not exist"
            elif dst.exists() and not overwrite:
                status = "skipped_existing"
            else:
                shutil.copy2(src, dst)
                status = "copied"

        rows.append(
            {
                "row_index": int(i),
                "object_id": object_id,
                "plate": None if pd.isna(plate) else int(plate),
                "mjd": None if pd.isna(mjd) else int(mjd),
                "fiber": None if pd.isna(fiber) else int(fiber),
                "sdss_name": row.get("sdss_name", None),
                "status": status,
                "error": error,
                "source_path": src_path,
                "dest_path": dest_path,
            }
        )

    report = pd.DataFrame(rows)
    counts = report["status"].value_counts(dropna=False).to_dict()
    total = len(report)
    copied = int(counts.get("copied", 0))
    skipped_existing = int(counts.get("skipped_existing", 0))
    missing_mapping = int(counts.get("missing_mapping", 0))
    missing_source = int(counts.get("missing_source", 0))

    print(
        "Summary: "
        f"total={total}, copied={copied}, skipped_existing={skipped_existing}, "
        f"missing_mapping={missing_mapping}, missing_source={missing_source}"
    )

    if report_csv is not None:
        Path(report_csv).parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_csv, index=False)
        print(f"Wrote report CSV: {report_csv}")

    failures = missing_mapping + missing_source
    if strict and failures > 0:
        raise RuntimeError(f"Strict mode enabled with {failures} unresolved rows.")

    return report


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Copy local DR17 spectra into a cache using native SDSS filenames "
            "(spec-plate-mjd-fiber.fits)."
        )
    )
    p.add_argument("--input-csv", required=True, help="CSV containing at least object_id.")
    p.add_argument("--source-root", required=True, help="Path to local DR17 spectra/lite root.")
    p.add_argument("--dest-cache-dir", required=True, help="Destination cache directory.")
    p.add_argument("--catalog-parquet", default="data/S82/Catalog.parquet")
    p.add_argument("--dr16q-fits", default="data/dr16q_prop_May01_2024.fits")
    p.add_argument("--max-sep-arcsec", type=float, default=1.0)
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files.")
    p.add_argument("--strict", action="store_true", help="Raise if any row cannot be copied.")
    p.add_argument("--report-csv", default=None, help="Optional output CSV report path.")
    return p.parse_args()


def main():
    args = parse_args()
    build_cache(
        input_csv=args.input_csv,
        source_root=args.source_root,
        dest_cache_dir=args.dest_cache_dir,
        catalog_parquet=args.catalog_parquet,
        dr16q_fits=args.dr16q_fits,
        max_sep_arcsec=args.max_sep_arcsec,
        overwrite=args.overwrite,
        strict=args.strict,
        report_csv=args.report_csv,
    )


if __name__ == "__main__":
    main()
