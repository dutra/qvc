#!/usr/bin/env python3
import argparse
import csv
import glob
import multiprocessing
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from tqdm import tqdm

from qvc.hubble.cuts import LOG_SIGMA_UV_MAX, LOG_SIGMA_UV_MIN, LOG_TAU_UV_RF_MAX, LOG_TAU_UV_RF_MIN
from qvc.hubble.hubble_utils import resolve_qvc_data_path
from qvc.light_curve.fit_light_curves import make_lc
from qvc.light_curve.multiband_generate_lc import (
    MACLEOD_BANDS,
    MACLEOD_COLUMNS,
    concat_light_curves,
    populate_sdss_fields,
    read_macleod_band,
    resolve_stone_s82_matches,
)
from qvc.light_curve.plotting_appendix import plot_sigma_tau_identity_grid

MACLEOD_YEAR_DAYS = 365.25
MACLEOD_IDENTITY_BANDS = ("u", "g", "r", "i")
STONE_IDENTITY_BANDS = ("g", "r", "i")
STONE_SIGMA_LIMITS = (-1.6, 0.2)
STONE_TAU_LIMITS = (1.4, 4.6)

def _decode_h5_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
    return value


def _decode_h5_vector(values):
    arr = np.asarray(values)
    if arr.dtype.kind == "S":
        return arr.astype(str)
    if arr.dtype == object:
        out = []
        for value in arr:
            out.append(_decode_h5_scalar(value))
        return np.asarray(out, dtype=object)
    if arr.ndim > 1:
        return np.asarray([_decode_h5_scalar(v) for v in arr.tolist()], dtype=object)
    return arr


def _read_optional_h5_scalar(hdf, file_path, key):
    """
    Read a scalar/length-1 dataset from root and return string value.
    Returns empty string when missing/unreadable.
    """
    try:
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


def _load_h5_shard(path, expected_n):
    try:
        with h5py.File(path, "r") as hdf:
            source_git_commit = _read_optional_h5_scalar(hdf, path, "git_commit")
            source_run_datetime = _read_optional_h5_scalar(hdf, path, "run_datetime")

            row_columns = {}
            n_rows = None
            for key in hdf.keys():
                values = hdf[key][...]
                arr = np.asarray(values)
                if arr.ndim == 0:
                    continue
                if arr.ndim == 1 and arr.shape[0] == 1 and key in {"git_commit", "run_datetime"}:
                    continue
                if n_rows is None:
                    n_rows = int(arr.shape[0])
                elif int(arr.shape[0]) != n_rows:
                    print(
                        f"WARNING: Skipping dataset '{key}' in {path}: incompatible leading "
                        f"dimension {arr.shape[0]} (expected {n_rows})."
                    )
                    continue
                row_columns[key] = _decode_h5_vector(arr)

            if n_rows is None:
                n_rows = 0

            if not enforce_expected_count(n_rows, expected_n, path):
                return {"path": path, "ok": True, "skip": True, "rows": []}

            if not row_columns:
                rows = []
            else:
                fields = list(row_columns.keys())
                rows = []
                for idx in range(n_rows):
                    row = {field: row_columns[field][idx] for field in fields}
                    row["git_commit"] = source_git_commit
                    row["run_datetime"] = source_run_datetime
                    rows.append(row)

            return {"path": path, "ok": True, "skip": False, "rows": rows}
    except Exception as exc:
        return {"path": path, "ok": False, "skip": False, "error": str(exc), "rows": []}


def _load_h5_shard_worker(args):
    path, expected_n = args
    return _load_h5_shard(path, expected_n)


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


def load_and_merge_h5(file_list, expected_n, load_n, workers=1):
    all_quasars = []
    if load_n is not None:
        file_list = file_list[:load_n]

    workers = max(1, int(workers))
    if workers == 1:
        iterator = (
            _load_h5_shard(path, expected_n)
            for path in tqdm(file_list, desc="Merging HDF5 shards", unit="file")
        )
    else:
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(processes=workers)
        iterator = tqdm(
            pool.imap(_load_h5_shard_worker, ((path, expected_n) for path in file_list), chunksize=1),
            total=len(file_list),
            desc="Merging HDF5 shards",
            unit="file",
        )

    try:
        for result in iterator:
            if not result["ok"]:
                print(f"ERROR reading {result['path']}: {result['error']}")
                continue
            if result.get("skip"):
                continue
            all_quasars.extend(result["rows"])
    finally:
        if workers > 1:
            pool.close()
            pool.join()

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
        source_obj = reloaded_by_object_id[object_id]
        lc_input = dict(source_obj)
        lc_input.update(row)
        if "z" not in lc_input:
            raise ValueError(
                "Failed to recompute variability because required key 'z' is missing for "
                f"object_id: {object_id}"
            )
        lc = make_lc(
            lc_input,
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


def build_stone_identity_plot_path(prefix: str, base_dir: str) -> str:
    base_path = Path(base_dir)
    return str(base_path.parent / "plots" / prefix / "sigma_tau_identity_grid.pdf")


def build_stone_identity_plot_data(rows, stone_fits_path=None, s82_catalog_path=None, max_sep_arcsec=1.0):
    df_rows = pd.DataFrame(rows).copy()
    if df_rows.empty:
        return df_rows

    if stone_fits_path is None:
        stone_fits_path = resolve_qvc_data_path("data/Stone2021/TotalDat.fits")
    else:
        stone_fits_path = str(stone_fits_path)

    with fits.open(stone_fits_path, memmap=False) as hdul:
        stone_data = hdul[1].data

        stone_rows = []
        for i in range(len(stone_data)):
            row = {
                "stone_DBID": int(stone_data["DBID"][i]),
                "stone_RA": float(stone_data["RA"][i]),
                "stone_DEC": float(stone_data["DEC"][i]),
                "stone_Z": float(stone_data["Z"][i]),
            }
            for band in STONE_IDENTITY_BANDS:
                row |= {
                    f"stone_log_SIGMA_{band}": float(stone_data[f"log_SIGMA_{band}"][i]),
                    f"stone_log_SIGMA_{band}_ERR_L": float(stone_data[f"log_SIGMA_{band}_ERR_L"][i]),
                    f"stone_log_SIGMA_{band}_ERR_U": float(stone_data[f"log_SIGMA_{band}_ERR_U"][i]),
                    f"stone_log_TAU_REST_{band}": float(stone_data[f"log_TAU_REST_{band}"][i]),
                    f"stone_log_TAU_REST_{band}_ERR_L": float(stone_data[f"log_TAU_REST_{band}_ERR_L"][i]),
                    f"stone_log_TAU_REST_{band}_ERR_U": float(stone_data[f"log_TAU_REST_{band}_ERR_U"][i]),
                }
            stone_rows.append(row)

    stone_df = pd.DataFrame(stone_rows)
    matched, _ = resolve_stone_s82_matches(
        stone_fits_path=stone_fits_path,
        s82_catalog_path=s82_catalog_path,
        max_sep_arcsec=max_sep_arcsec,
    )
    if matched.empty:
        return matched.copy()

    matched = matched.copy()
    matched["stone_DBID"] = pd.to_numeric(matched["stone_DBID"], errors="coerce").astype("Int64")
    stone_df["stone_DBID"] = pd.to_numeric(stone_df["stone_DBID"], errors="coerce").astype("Int64")
    stone_df = stone_df.merge(
        matched.loc[:, ["stone_DBID", "object_id", "match_sep_arcsec"]],
        on="stone_DBID",
        how="inner",
    )
    if stone_df.empty:
        return stone_df

    df_rows["object_id"] = df_rows["object_id"].astype(str)
    stone_df["object_id"] = stone_df["object_id"].astype(str)
    merged = stone_df.merge(df_rows, on="object_id", how="inner", suffixes=("_stone", "_fit"))
    if merged.empty:
        return merged

    for band in STONE_IDENTITY_BANDS:
        merged[f"ours_sigma_{band}"] = pd.to_numeric(merged[f"log_sigma_band_{band}"], errors="coerce")
        merged[f"ours_sigma_{band}_err"] = pd.to_numeric(merged[f"log_sigma_band_{band}_err"], errors="coerce")
        merged[f"ours_tau_{band}"] = pd.to_numeric(merged[f"log_tau_band_{band}_RF"], errors="coerce")
        merged[f"ours_tau_{band}_err"] = pd.to_numeric(merged[f"log_tau_band_{band}_RF_err"], errors="coerce")
    return merged


def write_stone_sigma_tau_identity_grid(rows, output_path: str, stone_fits_path=None, s82_catalog_path=None):
    plot_df = build_stone_identity_plot_data(
        rows,
        stone_fits_path=stone_fits_path,
        s82_catalog_path=s82_catalog_path,
    )
    if plot_df.empty:
        print("WARNING: Skipping Stone sigma/tau identity grid because no matched Stone comparison rows were found.")
        return None

    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sigma_keys = {
        "x": "stone_log_SIGMA_{band}",
        "y": "ours_sigma_{band}",
        "xerr": ("stone_log_SIGMA_{band}_ERR_L", "stone_log_SIGMA_{band}_ERR_U"),
        "yerr": "ours_sigma_{band}_err",
        "xlabel": r"$\log\!\,\sigma_<<band>>$ (mag)" + "\n(Stone+21)",
        "ylabel": r"$\log\!\,\sigma_<<band>>$ (mag)" + "\n(this work)",
    }
    tau_keys = {
        "x": "stone_log_TAU_REST_{band}",
        "y": "ours_tau_{band}",
        "xerr": ("stone_log_TAU_REST_{band}_ERR_L", "stone_log_TAU_REST_{band}_ERR_U"),
        "yerr": "ours_tau_{band}_err",
        "xlabel": r"$\log\!\,\tau_{\mathrm{<<band>>},\,\mathrm{RF}}\,(\mathrm{days})$" + "\n(Stone+21)",
        "ylabel": r"$\log\!\,\tau_{\mathrm{<<band>>},\,\mathrm{RF}}\,(\mathrm{days})$" + "\n(this work)",
    }
    fig = plot_sigma_tau_identity_grid(
        plot_df,
        sigma_keys,
        tau_keys,
        bands=STONE_IDENTITY_BANDS,
        figsize=(12, 7.6),
        show=False,
        output_path=output_path,
        sigma_limits=STONE_SIGMA_LIMITS,
        tau_limits=STONE_TAU_LIMITS,
    )
    plt.close(fig)
    print(f"Wrote Stone sigma/tau identity grid to {output_path}")
    return output_path


def _match_rows_to_catalog(df_rows, df_catalog, max_sep_arcsec=1.0):
    row_coords = SkyCoord(
        ra=pd.to_numeric(df_rows["ra"], errors="coerce").to_numpy(dtype=float) * u.deg,
        dec=pd.to_numeric(df_rows["dec"], errors="coerce").to_numpy(dtype=float) * u.deg,
    )
    catalog_coords = SkyCoord(
        ra=pd.to_numeric(df_catalog["ra"], errors="coerce").to_numpy(dtype=float) * u.deg,
        dec=pd.to_numeric(df_catalog["dec"], errors="coerce").to_numpy(dtype=float) * u.deg,
    )
    idx, d2d, _ = row_coords.match_to_catalog_sky(catalog_coords)
    matched = df_rows.copy()
    matched["matched_idx_b"] = np.where(d2d < (max_sep_arcsec * u.arcsec), idx, -1)
    matched["matched_sep_arcsec"] = d2d.arcsec
    return matched


def build_macleod_identity_plot_path(prefix: str, base_dir: str) -> str:
    return build_stone_identity_plot_path(prefix, base_dir)


def build_macleod_identity_plot_data(rows, macleod_dir=None, max_sep_arcsec=1.0):
    if macleod_dir is None:
        macleod_dir = resolve_qvc_data_path("data/MacLeod2010")
    else:
        macleod_dir = str(macleod_dir)

    df_rows = pd.DataFrame(rows).copy()
    if df_rows.empty:
        return {}

    by_band = {}
    for band in MACLEOD_IDENTITY_BANDS:
        macleod = read_macleod_band(band, macleod_dir=macleod_dir)
        matched = _match_rows_to_catalog(df_rows, macleod, max_sep_arcsec=max_sep_arcsec)
        matched = matched[matched["matched_idx_b"] >= 0].copy()
        if matched.empty:
            by_band[band] = matched
            continue

        matched_idx = matched["matched_idx_b"].to_numpy(dtype=int)
        rename_map = {
            "SDR5ID": f"macleod_SDR5ID_{band}",
            "redshift": f"macleod_redshift_{band}",
            "log10_tau_days": f"macleod_tau_obs_{band}",
            "log10_tau_lim_lo": f"macleod_tau_obs_lo_{band}",
            "log10_tau_lim_hi": f"macleod_tau_obs_hi_{band}",
            "log10_sigma_mag_sqrt_yr": f"macleod_log_sigma_hat_catalog_{band}",
            "log10_sig_lim_lo": f"macleod_sigma_obs_lo_{band}",
            "log10_sig_lim_hi": f"macleod_sigma_obs_hi_{band}",
            "npts": f"macleod_npts_{band}",
            "edge_flag": f"macleod_edge_flag_{band}",
            "chi2_pdf": f"macleod_chi2_pdf_{band}",
            "Plike": f"macleod_Plike_{band}",
            "Pnoise": f"macleod_Pnoise_{band}",
            "Pinf": f"macleod_Pinf_{band}",
        }
        for source_col, target_col in rename_map.items():
            matched[target_col] = macleod.iloc[matched_idx][source_col].to_numpy()

        matched[f"macleod_sigma_{band}"] = (
            pd.to_numeric(matched[f"macleod_log_sigma_hat_catalog_{band}"], errors="coerce")
            - 0.5 * np.log10(2.0 * MACLEOD_YEAR_DAYS)
            + 0.5 * pd.to_numeric(matched[f"macleod_tau_obs_{band}"], errors="coerce")
        )
        matched[f"macleod_sigma_lo_{band}"] = (
            pd.to_numeric(matched[f"macleod_sigma_obs_lo_{band}"], errors="coerce")
            - 0.5 * np.log10(2.0 * MACLEOD_YEAR_DAYS)
            + 0.5 * pd.to_numeric(matched[f"macleod_tau_obs_{band}"], errors="coerce")
        )
        matched[f"macleod_sigma_hi_{band}"] = (
            pd.to_numeric(matched[f"macleod_sigma_obs_hi_{band}"], errors="coerce")
            - 0.5 * np.log10(2.0 * MACLEOD_YEAR_DAYS)
            + 0.5 * pd.to_numeric(matched[f"macleod_tau_obs_{band}"], errors="coerce")
        )
        matched[f"macleod_tau_{band}"] = pd.to_numeric(matched[f"macleod_tau_obs_{band}"], errors="coerce")
        matched[f"macleod_tau_lo_{band}"] = pd.to_numeric(matched[f"macleod_tau_obs_lo_{band}"], errors="coerce")
        matched[f"macleod_tau_hi_{band}"] = pd.to_numeric(matched[f"macleod_tau_obs_hi_{band}"], errors="coerce")
        matched[f"ours_sigma_{band}"] = pd.to_numeric(matched[f"log_sigma_band_{band}"], errors="coerce")
        matched[f"ours_sigma_{band}_err"] = pd.to_numeric(matched[f"log_sigma_band_{band}_err"], errors="coerce")
        matched[f"ours_tau_{band}"] = pd.to_numeric(matched[f"log_tau_band_{band}_RF"], errors="coerce")
        matched[f"ours_tau_{band}_err"] = pd.to_numeric(matched[f"log_tau_band_{band}_RF_err"], errors="coerce")

        m10_quality_mask = (
            (pd.to_numeric(matched[f"macleod_edge_flag_{band}"], errors="coerce") == 0)
            & (
                pd.to_numeric(matched[f"macleod_Plike_{band}"], errors="coerce")
                - pd.to_numeric(matched[f"macleod_Pnoise_{band}"], errors="coerce")
                > 2.0
            )
            & (
                pd.to_numeric(matched[f"macleod_Plike_{band}"], errors="coerce")
                - pd.to_numeric(matched[f"macleod_Pinf_{band}"], errors="coerce")
                > 0.05
            )
        )
        finite_range_mask = (
            matched[f"macleod_tau_{band}"].between(LOG_TAU_UV_RF_MIN, LOG_TAU_UV_RF_MAX)
            & matched[f"macleod_sigma_{band}"].between(LOG_SIGMA_UV_MIN, LOG_SIGMA_UV_MAX)
        )
        by_band[band] = matched.loc[m10_quality_mask & finite_range_mask].copy()
    return by_band


def write_macleod_sigma_tau_identity_grid(rows, output_path: str, macleod_dir=None):
    by_band = build_macleod_identity_plot_data(rows, macleod_dir=macleod_dir)
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sigma_keys = {
        "x": "macleod_sigma_{band}",
        "y": "ours_sigma_{band}",
        "xerr": ("macleod_sigma_lo_{band}", "macleod_sigma_hi_{band}"),
        "xerr_mode": "bounds",
        "yerr": "ours_sigma_{band}_err",
        "xlabel": r"$\log\!\,\sigma_<<band>>\,(\mathrm{mag})$" + "\n(MacLeod+2010)",
        "ylabel": r"$\log\!\,\sigma_<<band>>\,(\mathrm{mag})$" + "\n(this work)",
    }
    tau_keys = {
        "x": "macleod_tau_{band}",
        "y": "ours_tau_{band}",
        "xerr": ("macleod_tau_lo_{band}", "macleod_tau_hi_{band}"),
        "xerr_mode": "bounds",
        "yerr": "ours_tau_{band}_err",
        "xlabel": r"$\log\!\,\tau_{\mathrm{<<band>>},\,\mathrm{RF}}\,(\mathrm{days})$" + "\n(MacLeod+2010)",
        "ylabel": r"$\log\!\,\tau_{\mathrm{<<band>>},\,\mathrm{RF}}\,(\mathrm{days})$" + "\n(this work)",
    }
    fig = plot_sigma_tau_identity_grid(
        by_band,
        sigma_keys,
        tau_keys,
        bands=MACLEOD_IDENTITY_BANDS,
        show=False,
        output_path=output_path,
        style={"point_alpha": 0.25, "error_alpha": 0.08, "rasterized": False},
    )
    plt.close(fig)
    print(f"Wrote MacLeod sigma/tau identity grid to {output_path}")
    return output_path


def main(argv=None):
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
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use when reading HDF5 shards. Default: 1",
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
        "--plot-stone-sigma-tau-identity-grid",
        action="store_true",
        default=False,
        help="Generate a Stone raw-vs-fit sigma/tau identity grid from the merged rows.",
    )
    p.add_argument(
        "--stone-identity-plot-out",
        type=str,
        default=None,
        help="Optional output path for the Stone sigma/tau identity grid PDF.",
    )
    p.add_argument(
        "--plot-macleod-sigma-tau-identity-grid",
        action="store_true",
        default=False,
        help="Generate a MacLeod raw-vs-fit sigma/tau identity grid from the merged rows.",
    )
    p.add_argument(
        "--macleod-identity-plot-out",
        type=str,
        default=None,
        help="Optional output path for the MacLeod sigma/tau identity grid PDF.",
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
    
    args = p.parse_args(argv)

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

    if args.workers < 1:
        print("ERROR: --workers must be >= 1")
        sys.exit(1)

    all_quasars = load_and_merge_h5(
        file_list,
        expected_n=args.expected,
        load_n=args.N,
        workers=args.workers,
    )
    print(f"Loaded total of {len(all_quasars)} rows from {len(file_list)} shards.")

    dedup_keys = args.dedup_keys
    if dedup_keys:
        before = len(all_quasars)
        all_quasars = deduplicate(all_quasars, keys=dedup_keys)
        print(f"De-duplicated by '{dedup_keys}': {before} -> {len(all_quasars)}")

    if not args.skip_populate_sdss and all_quasars:
        print("Populating SDSS fields...")
        all_quasars = populate_sdss_fields(all_quasars)
        if all_quasars and "plate" in all_quasars[0]:
            print(all_quasars[0]["plate"])

    if args.compute_variability and all_quasars:
        print("Computing corrected variability metrics from merged rows...")
        all_quasars = attach_variability_metrics(all_quasars)

    if args.plot_stone_sigma_tau_identity_grid and all_quasars:
        plot_out = args.stone_identity_plot_out or build_stone_identity_plot_path(args.prefix, args.base_dir)
        write_stone_sigma_tau_identity_grid(all_quasars, plot_out)

    if args.plot_macleod_sigma_tau_identity_grid and all_quasars:
        plot_out = args.macleod_identity_plot_out or build_macleod_identity_plot_path(args.prefix, args.base_dir)
        write_macleod_sigma_tau_identity_grid(all_quasars, plot_out)

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
