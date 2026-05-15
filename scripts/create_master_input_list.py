#!/usr/bin/env python3
"""Build a master CSV of S82 light curves joined to DR17 spectroscopy metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from astropy.table import Table
from tqdm import tqdm

from qvc.hubble.hubble_utils import resolve_qvc_data_path
from qvc.light_curve.fit_light_curves import make_lc
from qvc.light_curve.multiband_generate_lc import concat_light_curves, populate_sdss_fields
from qvc.light_curve.variability_metrics import compute_variability_metrics_for_cleaned_lc


OUTPUT_COLUMNS = [
    "object_id",
    "sdss_name",
    "ra",
    "dec",
    "plate",
    "fiberid",
    "mjd",
    "variability_chi_sq_red_g",
    "SDSSS_RUN2D",
]

SPEC_COLUMNS = [
    "PLATE",
    "PLUG_RA",
    "PLUG_DEC",
    "SPECOBJID",
    "Z",
    "Z_ERR",
    "VDISP",
    "ZWARNING",
    "PLATEID",
    "FIBERID",
    "MJD",
    "RUN2D",
    "RCHI2",
    "RCHI2DIFF",
]


def _normalize_run2d(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if text.startswith("b'") and text.endswith("'"):
        text = text[2:-1]
    text = text.strip()
    return text or pd.NA


def _normalize_table_string(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _coerce_join_keys(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out


def build_run2d_allowed_keys(spec_table: Table, run2d_cut: str) -> set[tuple[int, int, int]]:
    normalized_cut = _normalize_run2d(run2d_cut)
    allowed = set()
    for row in spec_table:
        normalized_run2d = _normalize_run2d(_normalize_table_string(row["RUN2D"]))
        if normalized_run2d != normalized_cut:
            continue
        allowed.add((int(row["PLATE"]), int(row["FIBERID"]), int(row["MJD"])))
    return allowed


def compute_variability_table(*, allowed_spec_keys: set[tuple[int, int, int]] | None = None) -> tuple[pd.DataFrame, int, int]:
    objs = concat_light_curves(progress_bar=True)
    total_objects = len(objs)
    objs = populate_sdss_fields(objs, progress_bar=True)
    filtered_objects = len(objs)

    if allowed_spec_keys is not None:
        objs = [
            obj
            for obj in objs
            if not any(pd.isna(obj.get(name)) for name in ("plate", "fiberid", "mjd"))
            and (int(obj["plate"]), int(obj["fiberid"]), int(obj["mjd"])) in allowed_spec_keys
        ]
        filtered_objects = len(objs)

    rows = []
    for obj in tqdm(objs, desc="Computing g-band variability metrics"):
        cleaned_lc = make_lc(
            obj,
            bands=["g"],
            inject_fake=False,
            drop_band_lyman_alpha=False,
            verbose=False,
        )
        if cleaned_lc is None:
            continue

        metrics = compute_variability_metrics_for_cleaned_lc(cleaned_lc)
        rows.append(
            {
                "object_id": str(obj["object_id"]).strip(),
                "sdss_name": obj.get("sdss_name"),
                "ra": obj.get("ra"),
                "dec": obj.get("dec"),
                "plate": obj.get("plate"),
                "fiberid": obj.get("fiberid"),
                "mjd": obj.get("mjd"),
                "variability_chi_sq_red_g": metrics.get("variability_chi_sq_red_g"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No light-curve rows produced a valid g-band variability metric.")

    df["object_id"] = df["object_id"].astype(str).str.strip()
    df = _coerce_join_keys(df, ["plate", "fiberid", "mjd"])
    return df, total_objects, filtered_objects


def load_spec_table(spec_fits: str) -> Table:
    spec_path = resolve_qvc_data_path(spec_fits)
    spec = Table.read(spec_path, hdu=1)

    missing = [col for col in SPEC_COLUMNS if col not in spec.colnames]
    if missing:
        raise ValueError(
            f"DR17 spec table is missing required columns: {missing}. "
            f"Required columns: {SPEC_COLUMNS}"
        )

    return spec[SPEC_COLUMNS]


def join_variability_with_spec(df_variability: pd.DataFrame, spec_table: Table) -> pd.DataFrame:
    spec_lookup: dict[tuple[object, object, object], list[object]] = {}
    for row in spec_table:
        key = (row["PLATE"], row["FIBERID"], row["MJD"])
        spec_lookup.setdefault(key, []).append(row["RUN2D"])

    output_rows = []
    for row in df_variability.itertuples(index=False):
        if pd.isna(row.plate) or pd.isna(row.fiberid) or pd.isna(row.mjd):
            matches = [pd.NA]
        else:
            key = (int(row.plate), int(row.fiberid), int(row.mjd))
            matches = spec_lookup.get(key, [pd.NA])

        for run2d in matches:
            output_rows.append(
                {
                    "object_id": row.object_id,
                    "sdss_name": row.sdss_name,
                    "ra": row.ra,
                    "dec": row.dec,
                    "plate": row.plate,
                    "fiberid": row.fiberid,
                    "mjd": row.mjd,
                    "variability_chi_sq_red_g": row.variability_chi_sq_red_g,
                    "SDSSS_RUN2D": _normalize_run2d(_normalize_table_string(run2d)),
                }
            )

    out = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    out = out.sort_values("object_id", kind="mergesort").reset_index(drop=True)

    duplicate_mask = out.duplicated(subset=["object_id"], keep=False)
    if duplicate_mask.any():
        duplicate_rows = out.loc[duplicate_mask].copy()
        exact_duplicates = duplicate_rows.duplicated(keep="first")
        duplicate_rows = duplicate_rows.loc[~exact_duplicates].copy()

        counts = duplicate_rows.groupby("object_id").size()
        ambiguous_ids = counts[counts > 1].index.tolist()
        if ambiguous_ids:
            preview = ", ".join(ambiguous_ids[:10])
            raise RuntimeError(
                "Ambiguous DR17 join produced multiple distinct rows for object_id(s): "
                f"{preview}"
            )
        out = out.drop_duplicates(ignore_index=True)

    return out


def apply_filters(
    df: pd.DataFrame,
    *,
    variability_chisq_cut: float | None,
    run2d_cut: str | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    counts = {"after_join": len(df)}
    out = df.copy()

    if variability_chisq_cut is not None:
        out = out.loc[pd.to_numeric(out["variability_chi_sq_red_g"], errors="coerce") > variability_chisq_cut].copy()
    counts["after_variability_cut"] = len(out)

    if run2d_cut is not None:
        out = out.loc[out["SDSSS_RUN2D"] == run2d_cut].copy()
    counts["after_run2d_cut"] = len(out)
    return out, counts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a master input CSV for S82 light curves joined to DR17 spectroscopy."
    )
    parser.add_argument("--output-csv", required=True, help="Path to the output CSV.")
    parser.add_argument(
        "--variability_chisq_cut",
        type=float,
        default=None,
        help="Keep only rows with variability_chi_sq_red_g greater than this value.",
    )
    parser.add_argument(
        "--run2d_cut",
        type=str,
        default=None,
        help="Keep only rows with SDSSS_RUN2D equal to this value. Known DR17 values: 26, 103, 104, v5_13_2.",
    )
    parser.add_argument(
        "--spec-fits",
        default="data/SDSS_DR17/specObj-dr17.fits",
        help="Path to the DR17 specObj FITS table.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    spec_table = None
    allowed_spec_keys = None
    if args.run2d_cut is not None:
        spec_table = load_spec_table(args.spec_fits)
        allowed_spec_keys = build_run2d_allowed_keys(spec_table, args.run2d_cut)

    df_variability, total_objects, filtered_objects = compute_variability_table(
        allowed_spec_keys=allowed_spec_keys
    )
    print(f"Total light curves loaded: {total_objects}")
    if args.run2d_cut is not None:
        print(f"Objects remaining after RUN2D prefilter: {filtered_objects}")
    print(f"Rows with valid g-band variability metric: {len(df_variability)}")

    if spec_table is None:
        spec_table = load_spec_table(args.spec_fits)

    df_joined = join_variability_with_spec(df_variability, spec_table)
    print(f"Rows after DR17 join: {len(df_joined)}")

    filtered, counts = apply_filters(
        df_joined,
        variability_chisq_cut=args.variability_chisq_cut,
        run2d_cut=args.run2d_cut,
    )
    print(f"Rows after variability cut: {counts['after_variability_cut']}")
    print(f"Rows after RUN2D cut: {counts['after_run2d_cut']}")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)
    print(f"Final rows written: {len(filtered)}")
    print(f"Wrote CSV: {output_path}")


if __name__ == "__main__":
    main()
