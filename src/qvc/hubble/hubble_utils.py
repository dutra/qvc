"""Shared utility functions for the QVC/Hubble fitting workflow."""

import json
import math
import os
import warnings
from ast import literal_eval
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from typing import Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import FlatwCDM, Flatw0waCDM, FlatLambdaCDM, FlatwpwaCDM
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.votable import parse
from scipy.linalg import cho_factor
from scipy import stats
from tqdm import tqdm

from qvc.hubble.cuts import (
    EXCLUDED_SDSS_NAMES,
    F_HOST_2500_MAX,
    LIGHT_CURVE_N_POINTS_COLUMN,
    LIGHT_CURVE_N_POINTS_EXCLUDED_BANDS,
    LOG_AMP_DELTA_BC_UPPER,
    LOG_F_BC_3000_MAX,
    LOG_F_FE_UV_3000_MAX,
    REL_APPARENT_MAG_2500_ERR_MAX,
    add_light_curve_point_count_column,
)
from qvc.hubble.hubble_cut_config import (
    build_agn_cuts,
    build_dlog_amp_blr_cuts,
)
from qvc.hubble.hubble_model import (
    M_model_agn,
    M_model_agn_err,
    agn_model_oidx,
    agn_model_pack_obs,
    agn_model_pack_params,
    get_model_params,
    resolve_model_option_flags,
    infer_use_alpha_lambda_term,
)

PURPLE_ANSI = "\033[95m"
RESET_ANSI = "\033[0m"
HUBBLE_JITTER_SURVEYS = ("sdss", "ps1", "ztf")


def _count_redshift_bin_removals(frame):
    zero_counts = {
        "removed_z_lt_0p44": 0,
        "removed_z_0p44_to_1": 0,
        "removed_z_1_to_2": 0,
        "removed_z_2_to_3p16": 0,
        "removed_z_gt_3p16": 0,
    }
    if frame is None or len(frame) == 0 or "z" not in frame.columns:
        return zero_counts

    z = pd.to_numeric(frame["z"], errors="coerce").to_numpy(dtype=float)
    counts = {
        "removed_z_lt_0p44": int(np.count_nonzero(z < 0.44)),
        "removed_z_0p44_to_1": int(np.count_nonzero((z >= 0.44) & (z < 1.0))),
        "removed_z_1_to_2": int(np.count_nonzero((z >= 1.0) & (z < 2.0))),
        "removed_z_2_to_3p16": int(np.count_nonzero((z >= 2.0) & (z <= 3.16))),
        "removed_z_gt_3p16": int(np.count_nonzero(z > 3.16)),
    }
    return counts


def _append_cut_report_row(
    rows,
    *,
    step,
    criterion,
    before,
    kept,
    status,
    removed_frame=None,
):
    before_i = int(before)
    kept_i = int(kept)
    row = {
        "step": str(step),
        "criterion": str(criterion),
        "before": before_i,
        "removed": before_i - kept_i,
        "kept": kept_i,
        "status": str(status),
    }
    row.update(_count_redshift_bin_removals(removed_frame))
    rows.append(row)


def _render_cut_summary_table(rows):
    headers = ("step", "criterion", "before", "removed", "kept", "status")
    rendered_rows = []
    for row in rows:
        rendered_rows.append({key: str(row[key]) for key in headers})

    widths = {
        key: max(len(key), *(len(row[key]) for row in rendered_rows)) if rendered_rows else len(key)
        for key in headers
    }

    def _line(values):
        return "| " + " | ".join(values[key].ljust(widths[key]) for key in headers) + " |"

    border = "+-" + "-+-".join("-" * widths[key] for key in headers) + "-+"
    lines = [
        border,
        _line({key: key for key in headers}),
        border,
    ]
    lines.extend(_line(row) for row in rendered_rows)
    lines.append(border)
    return "\n".join(lines)


def _wrap_text_in_purple(text):
    return f"{PURPLE_ANSI}{text}{RESET_ANSI}"

def _discover_qvc_root() -> Path:
    """Find the repository root by walking upward to the pyproject file."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback to the historical layout assumption if the search fails.
    return here.parents[3]


QVC_ROOT = _discover_qvc_root()


def _resolve_qvc_repo_path(
    path_like: str | os.PathLike,
    *,
    env_var: str,
    repo_subdir: str,
) -> str:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"No such file or directory: '{path}'")

    candidates = []
    env_dir = os.environ.get(env_var)
    if env_dir:
        env_dir = Path(env_dir).expanduser()
        candidates.append(env_dir / path)
        if path.parts[:1] == (repo_subdir,):
            candidates.append(env_dir / Path(*path.parts[1:]))

    candidates.append(Path.cwd() / path)
    candidates.append(QVC_ROOT / path)
    if path.parts[:1] == (repo_subdir,):
        candidates.append(QVC_ROOT / repo_subdir / Path(*path.parts[1:]))
    else:
        candidates.append(QVC_ROOT / repo_subdir / path)

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        f"No such file or directory: '{path_like}'. "
        f"Tried relative to cwd='{Path.cwd()}', QVC root='{QVC_ROOT}', "
        f"and optional {env_var}."
    )


def resolve_qvc_data_path(path_like: str | os.PathLike) -> str:
    """Resolve a QVC data file independent of the current working directory."""
    return _resolve_qvc_repo_path(path_like, env_var="QVC_DATA_DIR", repo_subdir="data")


def resolve_qvc_result_path(path_like: str | os.PathLike) -> str:
    """Resolve a QVC results file independent of the current working directory."""
    return _resolve_qvc_repo_path(path_like, env_var="QVC_RESULT_DIR", repo_subdir="results")


def get_qvc_result_dir() -> Path:
    """Return the base directory used for generated result artifacts."""
    env_result_dir = os.environ.get("QVC_RESULT_DIR")
    if env_result_dir:
        return Path(env_result_dir).expanduser()
    return QVC_ROOT / "results"

def convert_M2500_to_logL2500(M2500):
    return -1/2.5 * (M2500 - 90.0)

def sym_percentile(x, p=[16, 50, 84], axis=0):
    lower, median, upper = np.percentile(x, p, axis=axis)
    err = 0.5 * (upper - lower)   # optional symmetric equivalent
    err_lower = median - lower
    err_upper = upper - median
    return median, err, err_lower, err_upper


def match_radec(df_a, df_b, populate_cols=[], ra_col_a='ra', dec_col_a='dec', ra_col_b='ra', dec_col_b='dec', max_sep_arcsec=1.0, add_prefix=False):
    """
    Match objects in df_a to df_b by sky coordinates within max_sep_arcsec.
    Returns a DataFrame with indices and separation for matches.
    """
    coords_a = SkyCoord(ra=df_a[ra_col_a].values * u.deg, dec=df_a[dec_col_a].values * u.deg)
    coords_b = SkyCoord(ra=df_b[ra_col_b].values * u.deg, dec=df_b[dec_col_b].values * u.deg)

    idx, d2d, _ = coords_a.match_to_catalog_sky(coords_b)
    match_mask = d2d < max_sep_arcsec * u.arcsec

    # Prepare result DataFrame
    result = df_a.copy()
    result['matched_idx_b'] = np.where(match_mask, idx, -1)
    result['matched_sep_arcsec'] = d2d.arcsec
    # Optionally, add matched object_id or similar column if present in df_b
    for col in populate_cols:
        matched_values = np.where(match_mask, df_b.iloc[idx][col].values, None)
        if add_prefix:
            result[f'matched_{col}'] = matched_values
        else:
            result[col] = matched_values

    # Print warnings for unmatched
    unmatched_object_ids = []
    for i, matched in enumerate(match_mask):
        if not matched:
            unmatched_object_ids.append(df_a.iloc[i]['object_id'])
    return result, unmatched_object_ids

def populate_xray(df, table_fpath="data/cscresults.vot"):
    xray_cols = [
        "flux_aper_b",
        "flux_aper_hilim_b",
        "flux_aper_lolim_b",
        "flux_aper_err_b",
        "log_Lxray",
        "log_Lxray_err",
        "alphaOX",
        "alphaOX_err",
        "alphaOX_exp",
        "alphaOX_exp_err",
        "delta_alphaOX",
        "delta_alphaOX_err",
    ]

    try:
        resolved_table_fpath = resolve_qvc_data_path(table_fpath)
    except FileNotFoundError:
        warnings.warn(
            f"CSC3 catalog not found at {table_fpath!r}; skipping X-ray enrichment.",
            RuntimeWarning,
            stacklevel=2,
        )
        df_out = df.copy()
        for col in xray_cols:
            if col not in df_out.columns:
                df_out[col] = np.nan
        return df_out

    # Read the CSC VOTable and normalize the column names.
    vo = parse(resolved_table_fpath)
    table = vo.get_first_table().to_table()

    fields = vo.get_first_table().fields
    new_names = [f.name for f in fields]
    for old, new in zip(table.colnames, new_names):
        table.rename_column(old, new)

    df_csc = table.to_pandas()

    # Match to the CSC catalog and copy the flux bounds.
    df_matched, unmatched_object_ids = match_radec(
        df, df_csc,
        populate_cols=['flux_aper_b', 'flux_aper_hilim_b', 'flux_aper_lolim_b'],
        max_sep_arcsec=1.0
    )
    print(f"Matched {len(df_matched) - len(unmatched_object_ids)} out of {len(df)} objects to CSC3 catalog.")

    # Ensure the matched flux columns are numeric.
    for c in ["flux_aper_b", "flux_aper_hilim_b", "flux_aper_lolim_b"]:
        df_matched[c] = pd.to_numeric(df_matched[c], errors="coerce")

    best = df_matched["flux_aper_b"]
    hi   = df_matched["flux_aper_hilim_b"]
    lo   = df_matched["flux_aper_lolim_b"]

    # Use the provided bounds to estimate a symmetric 1σ error.
    err = np.where(~hi.isna() & ~lo.isna(),
                   0.5 * (hi - lo),
                   np.where(~hi.isna(),
                            (hi - best),
                            np.where(~lo.isna(),
                                     (best - lo),
                                     np.nan)))
    df_matched["flux_aper_err_b"] = np.clip(err, a_min=0.0, a_max=None)

    # Build the luminosity distance in cm.
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    z = pd.to_numeric(df_matched["z"], errors="coerce")
    DL_cm = cosmo.luminosity_distance(z.values).to('cm').value

    # Adopt a fixed photon index and derive the nominal K-correction.
    GAMMA_X = 1.9
    Kcorr = np.power(1.0 + z.values, GAMMA_X - 2.0)  # = (1+z)^(-0.1) for Gamma=1.9
    # Keep the no-K-correction convention used elsewhere in this workflow.
    Kcorr = np.ones_like(Kcorr)

    # Pull the measured flux and error arrays.
    xray_flux     = df_matched["flux_aper_b"].replace(0, np.nan).values
    xray_flux_err = df_matched["flux_aper_err_b"].replace(0, np.nan).values

    # Compute 2-10 keV and monochromatic 2 keV luminosities.
    L_xray_210kev = 4.0 * np.pi * (DL_cm**2) * xray_flux * Kcorr
    L_xray_2 = L_xray_210kev * (2.0 - GAMMA_X) / (10.0**(2.0 - GAMMA_X) - 2.0**(2.0 - GAMMA_X)) * 2.0**(1.0 - GAMMA_X)

    df_matched["log_Lxray"] = np.log10(L_xray_2)

    # Propagate the X-ray flux error into log luminosity.
    df_matched["log_Lxray_err"] = (1.0 / np.log(10.0)) * (xray_flux_err / xray_flux)

    # Use the de-reddened UV luminosity for alpha_OX.
    df_matched["alphaOX"] = -(df_matched["log_Lxray"] - df_matched["log_L2500_int_fs"]) / 2.605
    df_matched["alphaOX_err"] = np.sqrt(
        df_matched["log_Lxray_err"]**2 + df_matched["log_L2500_int_fs_err"]**2
    ) / 2.605

    # Expected alpha_OX from Just et al. (2007), Equation 3.
    a = -0.140
    sigma_a = 0.007
    b = 2.705
    sigma_b = 0.212

    alphaOx_expected = a *  df_matched["log_L2500_int_fs"] + b

    x =  df_matched["log_L2500_int_fs"]
    sigma_x =  df_matched["log_L2500_int_fs_err"]

    alphaOx_expected_err = np.sqrt(
        (x**2) * (sigma_a**2) +
        (sigma_b**2) +
        (a**2) * (sigma_x**2)
    )

    df_matched["alphaOX_exp"] = alphaOx_expected
    df_matched["alphaOX_exp_err"] = alphaOx_expected_err

    # Compare the observed and expected alpha_OX values.
    df_matched["delta_alphaOX"] = df_matched["alphaOX"] - df_matched["alphaOX_exp"]
    df_matched["delta_alphaOX_err"] = np.sqrt(
        df_matched["alphaOX_err"]**2 + df_matched["alphaOX_exp_err"]**2
    )

    return df_matched


def populate_lc_info(df, lc_info_csv):
    fields = {
        # 'number_points': int,
        # 'cadence': float,
        # 'cadence_err': float,
        # 't_std': float,
        # 'variability_chi_sq_g_raw': float,
        # 'chi_sq_red_g_raw': float,
        # 'pvalue_g': float
        'variability_chi_sq_g': float,
    }
    # Load and concatenate two CSV files
    lc_info_csv = resolve_qvc_data_path(lc_info_csv)
    conv = _wrap_converters(fields)
    df_lcinfo = pd.read_csv(
        lc_info_csv,
        dtype={'object_id': str},
        converters=conv
    )
    print(f"Loaded lc info CSV: {df_lcinfo.keys()} with {len(df_lcinfo)} rows")
    merged = df.merge(df_lcinfo, on='object_id', how='left', suffixes=('', '_lcinfo'))

    print("Length of lc info merged DataFrame:", len(merged))
    missing_ids = set(df['object_id']) - set(df_lcinfo['object_id'])
    print("object_id not in merged:", list(missing_ids))

    for col in fields.keys():
        if f'{col}_lcinfo' in merged.columns:
            df[f'{col}'] = merged[f'{col}_lcinfo']
        else:
            df[col] = merged[col]
    return df

def _norm_name(s):
    s = (s or "").replace("\ufeff", "").strip()
    return s

def _wrap_converters(fields):
    """Wrap converters to report column+value; do NOT include 'object_id' here."""
    wrapped = {}
    for col, fn in fields.items():
        if col == "object_id":
            continue
        def make(fn, col):
            def _conv(x):
                x = None if x == "" else x
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return np.nan
                try:
                    return fn(x)
                except Exception as e:
                    raise ValueError(f"Converter failed for column {col!r} with value {x!r}") from e
            return _conv
        wrapped[col] = make(fn, col)
    return wrapped

def _ensure_object_id(df):
    # rescue an index named object_id
    if df.index.name and _norm_name(df.index.name).lower() == "object_id" and "object_id" not in df.columns:
        df = df.reset_index()
    if "object_id" not in df.columns:
        raise KeyError("DataFrame lacks 'object_id' column after normalization.")
    df["object_id"] = df["object_id"].astype(str)
    return df

def parse_list(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if not s:
        return []
    # Try Python literal syntax first.
    try:
        v = literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [str(t) for t in v]
    except Exception:
        pass
    # Fall back to JSON.
    try:
        v = json.loads(s)
        if isinstance(v, (list, tuple)):
            return [str(t) for t in v]
    except Exception:
        pass
    # Finally, treat the value as a comma-separated string.
    return [t.strip() for t in s.split(",") if t.strip()]

def populate_spectra_fit(df, spectra_fit_csvs):
    fields = {
        "object_id": str,
        "apparent_mag_2500": float,
        "apparent_mag_2500_err": float,
        "apparent_mag_2500_reddened": float,
        "apparent_mag_2500_reddened_err": float,
        "apparent_mag_2500_intrinsic": float,
        "apparent_mag_2500_intrinsic_err": float,
        "apparent_mag_i_rest": float,
        "apparent_mag_i_obs": float,
        "delta_m_flux_recal": float,
        "f_host_2500": float,
        "f_host_2500_err": float,
        "f_bc_3000": float,
        "f_bc_3000_err": float,
        "f_fe_uv_3000": float,
        "f_fe_uv_3000_err": float,
        "f_na": float,
        "f_na_err": float,
        "f_br": float,
        "f_br_err": float,
        "f_PL": float,
        "f_PL_err": float,
        "wrms": float,
        "f_host_center": float,
        "frac_host_psf_2500": float,
        "frac_host_psf_2500_err": float,
        "f_host_2500_psf": float,
        "f_host_2500_psf_err": float,
        "reddening_ebv": float,
        "bi": float,
        "ebv_fs": float,
        "euv_fs": float,
        "conti_a_0": float,
        "bands_used": parse_list,
        "PL_slope": float,
        "PL_slope_err": float,
    }

    required_cols = {
        "apparent_mag_2500",
        "apparent_mag_2500_err",
        "PL_slope",
        "PL_slope_err",
        "f_host_2500",
        "f_host_2500_err",
        "f_bc_3000",
        "f_bc_3000_err",
        "f_fe_uv_3000",
        "f_fe_uv_3000_err",
    }

    df = _ensure_object_id(df.copy())
    existing_to_drop = [col for col in fields if col != "object_id" and col in df.columns]
    if existing_to_drop:
        df = df.drop(columns=existing_to_drop)

    out = df
    wanted = set(fields) | {"object_id"}
    converters = _wrap_converters({k: v for k, v in fields.items() if k != "object_id"})

    for i, csv_path in enumerate(spectra_fit_csvs):
        csv_path = resolve_qvc_data_path(csv_path)
        print(f"\033[96mLoading spectra fit CSV ({i+1}/{len(spectra_fit_csvs)}): {csv_path}\033[0m")

        df_spectra = pd.read_csv(
            csv_path,
            usecols=lambda c: _norm_name(c) in wanted,
            converters=converters,
            encoding="utf-8-sig",
            skipinitialspace=True,
        )
        df_spectra.columns = [_norm_name(c) for c in df_spectra.columns]
        df_spectra = _ensure_object_id(df_spectra)

        missing_required = sorted(required_cols.difference(df_spectra.columns))
        if missing_required:
            raise ValueError(
                f"Spectra fit CSV '{csv_path}' is missing required columns {missing_required}. "
                "Regenerate the spectra-fit CSV with the current fit_spectra pipeline."
            )

        merged = out.merge(
            df_spectra,
            on="object_id",
            how="left",
            suffixes=("", "_spectralfit"),
            validate="one_to_one",
        )
        print("Length of merged DataFrame:", len(merged))

        for col in fields:
            if col == "object_id":
                continue
            if col in merged.columns:
                out[col] = merged[col].values

    if "ebv_fs" in out.columns:
        out["log_ebv_fs"] = np.log10(out["ebv_fs"].replace(0, np.nan))
    if "euv_fs" in out.columns:
        out["log_euv_fs"] = np.log10(out["euv_fs"].replace(0, np.nan))
    if {"apparent_mag_2500_reddened", "apparent_mag_2500"}.issubset(out.columns):
        out["dm_red"] = out["apparent_mag_2500_reddened"] - out["apparent_mag_2500"]
    if {"apparent_mag_2500_reddened_err", "apparent_mag_2500_err"}.issubset(out.columns):
        out["dm_red_err"] = np.sqrt(
            out["apparent_mag_2500_reddened_err"] ** 2 + out["apparent_mag_2500_err"] ** 2
        )
    out["alpha_lambda"] = out["PL_slope"]
    out["alpha_lambda_err"] = out["PL_slope_err"]
    out["alpha_nu"] = -out["alpha_lambda"] - 2
    out["alpha_nu_err"] = out["alpha_lambda_err"]
    out["log_L2500_int_fs"] = -0.4 * out["apparent_mag_2500"] + 36.0
    out["log_L2500_int_fs_err"] = 0.4 * out["apparent_mag_2500_err"]
    out["iron_frac"] = out["f_fe_uv_3000"]

    return out
def populate_sdss_fields(objs, progress_bar=True):
    """
    Populate SDSS/DR16Q-derived fields.

    Supported inputs:
    - pandas.DataFrame (returns pandas.DataFrame)
    - list[dict] (returns list[dict], for backward compatibility)
    """
    raise NotImplementedError("This function is currently disabled pending refactor.")
    input_is_df = isinstance(objs, pd.DataFrame)
    input_is_list = isinstance(objs, list)
    if not (input_is_df or input_is_list):
        raise TypeError(
            "populate_sdss_fields expects a pandas.DataFrame or list[dict], "
            f"got {type(objs).__name__}"
        )

    if input_is_df:
        df = objs.copy()
    else:
        if len(objs) == 0:
            return objs
        df = pd.DataFrame.from_records(objs)

    print(f"Populating SDSS fields: {len(df)}", flush=True)
    if df.empty:
        return df if input_is_df else []

    if "object_id" not in df.columns:
        raise ValueError("populate_sdss_fields requires an 'object_id' column.")

    # Keep this parameter for backward compatibility; matching is now vectorized.
    _ = progress_bar

    def _normalize_native_endian(values):
        """Normalize numeric arrays/scalars to native-endian for pandas assignment."""
        arr = np.asarray(values)
        if arr.dtype.kind in {"i", "u", "f", "c", "b"} and arr.dtype.byteorder not in {"=", "|"}:
            arr = arr.byteswap().view(arr.dtype.newbyteorder("="))
        return arr

    rows = pd.DataFrame(
        {
            "row_idx": np.arange(len(df), dtype=int),
            "object_id": df["object_id"].to_numpy(),
        }
    )
    rows["object_id_key"] = rows["object_id"].astype(str)

    cat = pd.read_parquet(resolve_qvc_data_path("data/S82/Catalog.parquet"))
    cat_lookup = (
        cat.loc[:, ["objectId", "RA", "DEC"]]
        .dropna(subset=["objectId", "RA", "DEC"])
        .assign(object_id_key=lambda d: d["objectId"].astype(str))
        .drop_duplicates(subset=["object_id_key"], keep="first")
        .loc[:, ["object_id_key", "RA", "DEC"]]
    )
    rows = rows.merge(cat_lookup, on="object_id_key", how="left")

    with fits.open(resolve_qvc_data_path("data/dr16q_prop_May01_2024.fits")) as hdul:
        fits_data = hdul[1].data
        fits_data_2 = hdul[2].data

        fits_coords = SkyCoord(ra=fits_data["RA"] * u.deg, dec=fits_data["DEC"] * u.deg)

        valid_coord_mask = rows["RA"].notna() & rows["DEC"].notna()
        if not np.any(valid_coord_mask):
            print(f"Skipped {len(objs)} objects: no S82 catalog coordinates found.")
            return df if input_is_df else df.to_dict("records")

        query_rows = rows.loc[valid_coord_mask, ["row_idx", "RA", "DEC"]].copy()
        query_coords = SkyCoord(
            ra=query_rows["RA"].to_numpy() * u.deg,
            dec=query_rows["DEC"].to_numpy() * u.deg,
        )
        nearest_idx, d2d, _ = query_coords.match_to_catalog_sky(fits_coords)
        match_mask = d2d < (1.0 * u.arcsec)

        matched_query = query_rows.loc[match_mask].copy()
        matched_query["fits_idx"] = np.asarray(nearest_idx[match_mask], dtype=int)

        if matched_query.empty:
            missing_catalog = int((~valid_coord_mask).sum())
            missing_dr16q = int(valid_coord_mask.sum())
            if missing_catalog:
                print(f"Skipped {missing_catalog} objects: no S82 catalog coordinates found.")
            print(f"Skipped {missing_dr16q} objects: no DR16Q match within 1 arcsec.")
            return df if input_is_df else df.to_dict("records")

        matched_row_idx = matched_query["row_idx"].to_numpy(dtype=int)
        matched_fits_idx = matched_query["fits_idx"].to_numpy(dtype=int)

        z_vals = np.asarray(fits_data["Z_SYS"])[matched_fits_idx]
        logl3000 = np.asarray(fits_data["LOGL3000"])[matched_fits_idx]
        logl3000_err = np.asarray(fits_data["LOGL3000_ERR"])[matched_fits_idx]
        loglbol = np.asarray(fits_data["LOGLBOL"])[matched_fits_idx]
        loglbol_err = np.asarray(fits_data["LOGLBOL_ERR"])[matched_fits_idx]

        # Preserve legacy z-dependent luminosity branch.
        log_lbol = np.where(z_vals < 0.7, np.log10(5.15) + logl3000, loglbol)
        log_lbol_err = np.where(z_vals < 0.7, logl3000_err, loglbol_err)

        logmbh = np.asarray(fits_data["LOGMBH"])[matched_fits_idx]
        logmbh_err = np.asarray(fits_data["LOGMBH_ERR"])[matched_fits_idx]
        valid_logmbh = np.isfinite(logmbh) & (logmbh != 0)
        logmbh_out = np.where(valid_logmbh, logmbh, np.nan)
        logmbh_err_out = np.where(valid_logmbh, logmbh_err, np.nan)

        band_to_idx = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4}
        psfmag = np.asarray(fits_data_2["PSFMAG"])[matched_fits_idx, :]
        psfflux = np.asarray(fits_data_2["PSFFLUX"])[matched_fits_idx, :]
        psfflux_ivar = np.asarray(fits_data_2["PSFFLUX_IVAR"])[matched_fits_idx, :]

        with np.errstate(divide="ignore", invalid="ignore"):
            apparent_mag = np.where(psfflux > 0, -2.5 * np.log10(psfflux) + 22.5, np.nan)
            apparent_mag_err = np.where(
                (psfflux > 0) & (psfflux_ivar > 0),
                (2.5 / np.log(10.0)) * np.sqrt(1.0 / psfflux_ivar) / psfflux,
                np.nan,
            )

        update_fields = {
            "plate": np.asarray(fits_data["PLATE"])[matched_fits_idx],
            "mjd": np.asarray(fits_data["MJD"])[matched_fits_idx],
            "fiberid": np.asarray(fits_data["FIBERID"])[matched_fits_idx],

            "ra": rows.loc[matched_row_idx, "RA"].to_numpy(),
            "dec": rows.loc[matched_row_idx, "DEC"].to_numpy(),
            "z": z_vals,
            "z_err": np.asarray(fits_data["Z_SYS_ERR"])[matched_fits_idx],
            "sdss_name": np.asarray(fits_data["SDSS_NAME"])[matched_fits_idx],
            "log_lbol": log_lbol,
            "log_lbol_err": log_lbol_err,
            "LOGMBH": logmbh_out,
            "LOGMBH_ERR": logmbh_err_out,
            "LOGLEDD_RATIO": np.asarray(fits_data["LOGLEDD_RATIO"])[matched_fits_idx],
            "LOGLEDD_RATIO_ERR": np.asarray(fits_data["LOGLEDD_RATIO_ERR"])[matched_fits_idx],
            "ebv_wu": np.asarray(fits_data["EBV"])[matched_fits_idx],
            "sn_median_all": np.asarray(fits_data["SN_MEDIAN_ALL"])[matched_fits_idx],
            "M_i": np.asarray(fits_data_2["M_I"])[matched_fits_idx],
            "LOGLBOL": loglbol,
            "LOGL1350": np.asarray(fits_data["LOGL1350"])[matched_fits_idx],
            "LOGL1700": np.asarray(fits_data["LOGL1700"])[matched_fits_idx],
            "LOGL2500_wu": np.asarray(fits_data["LOGL2500"])[matched_fits_idx],
            "LOGL3000": logl3000,
            "LOGL5100": np.asarray(fits_data["LOGL5100"])[matched_fits_idx],
            "LOGL1350_ERR": np.asarray(fits_data["LOGL1350_ERR"])[matched_fits_idx],
            "LOGL1700_ERR": np.asarray(fits_data["LOGL1700_ERR"])[matched_fits_idx],
            "LOGL2500_ERR_wu": np.asarray(fits_data["LOGL2500_ERR"])[matched_fits_idx],
            "LOGL3000_ERR": logl3000_err,
            "LOGL5100_ERR": np.asarray(fits_data["LOGL5100_ERR"])[matched_fits_idx],
            "FHOST_5100": np.asarray(fits_data["FHOST_5100"])[matched_fits_idx],
            "fhost": np.asarray(fits_data["FHOST_5100"])[matched_fits_idx],
            "EXTINCTION": np.asarray(fits_data_2["EXTINCTION"])[matched_fits_idx, band_to_idx["i"]],
        }
        for band, bidx in band_to_idx.items():
            update_fields[f"PSFMAG_{band}"] = psfmag[:, bidx]
            update_fields[f"apparent_mag_{band}"] = apparent_mag[:, bidx]
            update_fields[f"apparent_mag_{band}_err"] = apparent_mag_err[:, bidx]

        target_index = df.index.to_numpy()[matched_row_idx]
        for col, values in update_fields.items():
            normalized_values = _normalize_native_endian(values)
            if col not in df.columns:
                value_kind = normalized_values.dtype.kind
                if value_kind in {"U", "S", "O"}:
                    df[col] = pd.Series([np.nan] * len(df), index=df.index, dtype=object)
                else:
                    df[col] = np.nan
            df.loc[target_index, col] = normalized_values

        missing_catalog = int((~valid_coord_mask).sum())
        missing_dr16q = int(valid_coord_mask.sum() - len(matched_row_idx))
        if missing_catalog:
            print(f"Skipped {missing_catalog} objects: no S82 catalog coordinates found.")
        if missing_dr16q:
            print(f"Skipped {missing_dr16q} objects: no DR16Q match within 1 arcsec.")

    return df if input_is_df else df.to_dict("records")




def populate_sdss_rchi2_fields(df, csv_path="data/sdss/sdss_allspec_rchi2.csv"):
    """Populate SDSS allspec quality columns using (object_id, plate, fiberid, mjd)."""
    if "object_id" not in df.columns:
        raise ValueError("populate_sdss_rchi2_fields requires an 'object_id' column.")

    target_cols = ["RCHI2", "RCHI2DIFF", "VDISP", "ZWARNING", "RUN2D"]
    key_cols = ["object_id", "plate", "fiberid", "mjd"]
    csv_path = resolve_qvc_data_path(csv_path)

    usecols = key_cols + target_cols
    df_sdss = pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype={"object_id": str, "RUN2D": str},
    )
    missing_cols = [col for col in usecols if col not in df_sdss.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required SDSS RCHI2 columns in {csv_path}: {missing_cols}"
        )

    def _normalize_run2d(value):
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        if text.startswith("b'") and text.endswith("'"):
            text = text[2:-1]
        return text.strip()

    def _to_int64(series):
        return pd.to_numeric(series, errors="coerce").astype("Int64")

    df_sdss["object_id"] = df_sdss["object_id"].astype(str)
    for col in ["plate", "fiberid", "mjd"]:
        df_sdss[col] = _to_int64(df_sdss[col])
    df_sdss["RUN2D"] = df_sdss["RUN2D"].apply(_normalize_run2d)

    out = df.copy()
    out["object_id"] = out["object_id"].astype(str)

    def _ensure_target_col(frame, col):
        if col in frame.columns:
            return
        if col == "RUN2D":
            frame[col] = pd.Series([np.nan] * len(frame), index=frame.index, dtype=object)
        else:
            frame[col] = np.nan

    # Count all SDSS spectra rows per object_id, independent of composite-key match.
    sdss_plate_count = (
        df_sdss.groupby("object_id", as_index=False)
        .size()
        .rename(columns={"size": "sdss_plate_count"})
    )
    out = out.merge(sdss_plate_count, on="object_id", how="left")
    out["sdss_plate_count"] = out["sdss_plate_count"].fillna(0).astype("Int64")

    def _find_col(frame, preferred, fallback):
        if preferred in frame.columns:
            return preferred
        if fallback in frame.columns:
            return fallback
        return None

    plate_col = _find_col(out, "plate", "PLATEID")
    fiberid_col = _find_col(out, "fiberid", "FIBERID")
    mjd_col = _find_col(out, "mjd", "MJD")

    if plate_col is None or fiberid_col is None or mjd_col is None:
        for col in target_cols:
            _ensure_target_col(out, col)
        object_match_count = int(out["object_id"].isin(set(df_sdss["object_id"])).sum())
        print(
            "Populated SDSS RCHI2 fields "
            f"from {csv_path}: loaded_rows={len(df_sdss)}, "
            f"unique_object_id={df_sdss['object_id'].nunique()}, "
            f"object_id_matches={object_match_count}, exact_key_matches=0, "
            f"object_only_no_key_match={object_match_count} "
            "(missing plate/fiberid/mjd columns in AGN table)"
        )
        return out

    out["_sdss_plate_key"] = _to_int64(out[plate_col])
    out["_sdss_fiberid_key"] = _to_int64(out[fiberid_col])
    out["_sdss_mjd_key"] = _to_int64(out[mjd_col])

    df_sdss_keyed = df_sdss.rename(
        columns={
            "plate": "_sdss_plate_key",
            "fiberid": "_sdss_fiberid_key",
            "mjd": "_sdss_mjd_key",
        }
    )
    merged = out.merge(
        df_sdss_keyed,
        on=["object_id", "_sdss_plate_key", "_sdss_fiberid_key", "_sdss_mjd_key"],
        how="left",
        suffixes=("", "_sdss_rchi2"),
        indicator="_sdss_match",
    )

    object_id_matched_mask = out["object_id"].isin(set(df_sdss["object_id"]))
    exact_match_mask = merged["_sdss_match"] == "both"
    for col in target_cols:
        src_col = f"{col}_sdss_rchi2"
        _ensure_target_col(out, col)
        if src_col in merged.columns:
            assign_vals = merged.loc[exact_match_mask, src_col].values
        elif col in merged.columns:
            assign_vals = merged.loc[exact_match_mask, col].values
        else:
            continue
        if col == "RUN2D":
            assign_vals = pd.Series(assign_vals, dtype=object).values
        out.loc[exact_match_mask, col] = assign_vals

    out = out.drop(columns=["_sdss_plate_key", "_sdss_fiberid_key", "_sdss_mjd_key"], errors="ignore")

    object_id_match_count = int(object_id_matched_mask.sum())
    exact_match_count = int(exact_match_mask.sum())
    object_only_no_key_match = int(object_id_match_count - exact_match_count)
    print(
        "Populated SDSS RCHI2 fields "
        f"from {csv_path}: loaded_rows={len(df_sdss)}, "
        f"unique_object_id={df_sdss['object_id'].nunique()}, "
        f"object_id_matches={object_id_match_count}, exact_key_matches={exact_match_count}, "
        f"object_only_no_key_match={object_only_no_key_match}"
    )
    return out


def _mask_invalid_wu_bhmass(df):
    """Mask Wu-catalog BH masses that use zero as a missing-value sentinel."""
    if "LOGMBH" not in df.columns:
        return df

    logmbh = pd.to_numeric(df["LOGMBH"], errors="coerce")
    zero_mask = np.isfinite(logmbh) & (logmbh == 0.0)
    if not np.any(zero_mask):
        return df

    df = df.copy()
    df.loc[zero_mask, "LOGMBH"] = np.nan
    if "LOGMBH_ERR" in df.columns:
        df.loc[zero_mask, "LOGMBH_ERR"] = np.nan
    return df

def read_quasars_from_hdf5(file_path, N=None):
    quasar_list = []

    with h5py.File(file_path, "r") as hdf:
        for group_name in tqdm(list(hdf.keys()), desc="Reading quasars from HDF5"):
            group = hdf[group_name]
            quasar = {"object_id": group_name}
            for key, value in group.attrs.items():
                quasar[key] = value
            for sub_group_name in group.keys():
                sub_group = group[sub_group_name]
                quasar[sub_group_name] = {sub_key: sub_group[sub_key][...] for sub_key in sub_group.keys()}
            quasar_list.append(quasar)
            if (N is not None) and (N >= 0) and (len(quasar_list) >= N):
                break
    return quasar_list


def read_quasars_from_hdf5_flat(file_path, N=None):
    """
    Read a flat columnar HDF5 file (top-level datasets) into a DataFrame.
    """
    def _decode_scalar(x):
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="replace")
        if isinstance(x, np.generic):
            x = x.item()
            if isinstance(x, bytes):
                return x.decode("utf-8", errors="replace")
        return x

    def _decode_vector(arr):
        arr = np.asarray(arr)
        if arr.dtype.kind == "S":
            return arr.astype(str)
        if arr.dtype == object:
            out = []
            for v in arr:
                out.append(_decode_scalar(v))
            return np.asarray(out, dtype=object)
        if arr.ndim > 1:
            return np.asarray([_decode_scalar(v) for v in arr.tolist()], dtype=object)
        return arr

    with h5py.File(file_path, "r") as hdf:
        keys = list(hdf.keys())
        if not keys:
            return pd.DataFrame()

        row_columns = {}
        scalar_metadata = {}
        n_rows = None
        for key in keys:
            values = hdf[key][...]
            arr = np.asarray(values)
            if arr.ndim == 0:
                scalar_metadata[key] = _decode_scalar(arr.item())
                continue
            if n_rows is None:
                n_rows = int(arr.shape[0])
            elif int(arr.shape[0]) != n_rows:
                warnings.warn(
                    f"Skipping dataset '{key}' in {file_path}: incompatible leading "
                    f"dimension {arr.shape[0]} (expected {n_rows})."
                )
                continue
            row_columns[key] = _decode_vector(arr)

        if not row_columns:
            return pd.DataFrame()

        # Materialize through records so pandas normalizes numeric backing arrays
        # (e.g., big-endian inputs from FITS/HDF5) into native-compatible columns.
        df = pd.DataFrame.from_records(pd.DataFrame(row_columns).to_dict("records"))

        if N is not None and N >= 0:
            df = df.iloc[: int(N)].reset_index(drop=True)

        for meta_key, meta_value in scalar_metadata.items():
            df[meta_key] = meta_value
    return df

def load_agn_data(file_path, populate_sdss=False, apply_cut=True,
                  exclude_object_ids_csv=None,
                  residuals_sigma_clip=None, residuals_csv=None,
                  spectra_fit_csv=None, only_load=False,
                  correct_sigma_uv_host=False,
                  lc_info_csv="data/lc_chisq.csv",
                  z_range=(0.44, 3.16),
                  plot_path="plots/hubble",
                  cut_report_path=None):
    def _format_cut_bounds(lower, upper, *, upper_inclusive=True, allow_missing=False):
        left = "[" if lower is not None else "("
        right = "]" if upper_inclusive else ")"
        lower_text = f"{lower}" if lower is not None else "-inf"
        upper_text = f"{upper}" if upper is not None else "inf"
        suffix = " or NaN" if allow_missing else ""
        return f"{left}{lower_text}, {upper_text}{right}{suffix}"

    def _apply_column_compatibility_shim(frame):
        """Map known legacy/alias column names onto the schema expected downstream."""
        legacy_to_new = {
            "log_sigma_UV": "log_sigma_uv",
            "log_sigma_UV_err": "log_sigma_uv_err",
            "log_sigma_UV_uncorrected": "log_sigma_uv_uncorrected",
            "log_sigma_UV_std_psd": "log_sigma_uv_std_psd",
            "log_tau_UV_RF": "log_tau_uv_rf",
            "log_tau_UV_RF_err": "log_tau_uv_rf_err",
            "log_tau_UV_RF_std_psd": "log_tau_uv_rf_std_psd",
            "cov_log_sigma_UV_log_tau_UV_RF": "cov_log_sigma_uv_log_tau_uv_rf",
            "log_sigma_UV_log_tau_UV_RF_cov_psd": "log_sigma_uv_log_tau_uv_rf_cov_psd",
            "M_I": "M_i",
        }
        alias_to_existing = {
            "LOGMBH": "log_mbh",
            "LOGMBH_ERR": "log_mbh_err",
        }
        copied = []
        for legacy, new in legacy_to_new.items():
            if new not in frame.columns and legacy in frame.columns:
                frame[new] = frame[legacy]
                copied.append(f"{legacy}->{new}")
        for new, alias in alias_to_existing.items():
            if new not in frame.columns and alias in frame.columns:
                frame[new] = frame[alias]
                copied.append(f"{alias}->{new}")
        if copied:
            print("Applied column compatibility shim:", ", ".join(copied))
        return frame

    from qvc.hubble.hubble_plotting import (
        plot_adf_pvalue_g_diagnostic,
        plot_alpha_lambda_vs_l2500_by_redshift,
        plot_alpha_lambda_vs_l2500,
        plot_alpha_lambda_vs_eta_sigma,
        plot_alpha_lambda_histogram,
        plot_alpha_lambda_vs_redshift,
        plot_blr_amp_vs_redshift_by_band,
        plot_blr_diagnostics_summary,
        plot_bc_lag_vs_l2500,
        plot_blr_line_lags_vs_l2500_fiducial,
        plot_blr_lag_vs_amp_by_band,
        plot_blr_lag_vs_redshift_by_band,
        plot_eta_tau_sigma_vs_redshift,
        plot_fast_vs_uv_variability,
        plot_f_host_2500_vs_redshift,
        plot_f_host_2500_vs_l2500,
        plot_g_band_drift_slope_histograms,
        plot_l2500_vs_eta_sigma_fiducial,
        plot_l2500_vs_uv_variability_fiducial,
        plot_linear_trend_vs_redshift,
        plot_Mi_relation,
        plot_light_curve_n_points_vs_apparent_mag,
        plot_cut_diagnostics,
        plot_m2500_vs_z_colorpanels,
        plot_spectral_fraction_vs_redshift,
        plot_sf_ref_band_vs_model_g,
        plot_sf_vs_uv_variability,
        plot_sigma_bc_vs_frac_bc,
        plot_sigma_bc_vs_redshift,
        plot_sigma_tau_err_std_psd_comparison,
        plot_sigma_uv_vs_variability_chi_sq_red_g,
        plot_sigma_uv_vs_tau_uv_rf,
        plot_sigma_uv_host_correction,
        plot_suberlak_style_sigma_tau_fits,
        plot_tau_sigma_vs_wu_catalog,
        plot_tau_sigma_vs_redshift,
    )

    if exclude_object_ids_csv is None:
        exclude_object_ids_csv = []
    cut_rows = []

    def _record_cut(step, criterion, frame_before, mask, *, status="applied", reset_index=True):
        before = len(frame_before)
        kept = int(np.count_nonzero(mask))
        removed_frame = frame_before.loc[~np.asarray(mask, dtype=bool)].copy()
        _append_cut_report_row(
            cut_rows,
            step=step,
            criterion=criterion,
            before=before,
            kept=kept,
            status=status,
            removed_frame=removed_frame,
        )
        filtered = frame_before[mask]
        if reset_index:
            filtered = filtered.reset_index(drop=True)
        return filtered

    def _finalize_cut_report():
        if only_load or not cut_rows:
            return
        table_text = _render_cut_summary_table(cut_rows)
        print(_wrap_text_in_purple(table_text))
        if cut_report_path is None:
            return
        report_path = Path(cut_report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(table_text + "\n", encoding="utf-8")
        diagnostics_path = report_path.parent / "cut_diagnostics_by_z.csv"
        pd.DataFrame(cut_rows).to_csv(diagnostics_path, index=False)

    def _normalize_dropped_bands(value):
        if value is None:
            return []
        if isinstance(value, float) and np.isnan(value):
            return []
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            txt = value.strip()
            if txt == "" or txt.lower() == "nan":
                return []
            # Handle list-like serialized strings such as "['u', 'z']"
            if txt.startswith("[") and txt.endswith("]"):
                try:
                    parsed = literal_eval(txt)
                    if isinstance(parsed, (list, tuple, set, np.ndarray)):
                        return [str(x).strip() for x in parsed if str(x).strip()]
                except Exception:
                    pass
            # Handle compact strings such as "uz" or delimiters like "u,z"
            if "," in txt:
                return [x.strip() for x in txt.split(",") if x.strip()]
            if " " in txt:
                return [x.strip() for x in txt.split() if x.strip()]
            return [c for c in txt if c in {"u", "g", "r", "i", "z", "y"}]
        if isinstance(value, np.ndarray):
            return [str(x).strip() for x in value.tolist() if str(x).strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(x).strip() for x in value if str(x).strip()]
        return [str(value).strip()]

    if lc_info_csv is not None:
        lc_info_csv = resolve_qvc_data_path(lc_info_csv)
    if residuals_csv is not None:
        residuals_csv = resolve_qvc_data_path(residuals_csv)
    if spectra_fit_csv is not None:
        spectra_fit_csv = [resolve_qvc_data_path(path) for path in spectra_fit_csv]
    if exclude_object_ids_csv:
        exclude_object_ids_csv = [resolve_qvc_data_path(path) for path in exclude_object_ids_csv]

    file_path = resolve_qvc_data_path(file_path)
    df = read_quasars_from_hdf5_flat(file_path)
    df = _apply_column_compatibility_shim(df)
    print("Number of quasars loaded:", len(df))
    legacy_required = [f"mags_mean_{i}" for i in range(4)]
    if all(col in df.columns for col in legacy_required):
        for i, b in enumerate(['u', 'g', 'r', 'i', 'z']):
            legacy_col = f"mags_mean_{i}"
            if legacy_col in df.columns and f"mags_mean_{b}" not in df.columns:
                df[f"mags_mean_{b}"] = df[legacy_col]
            if legacy_col in df.columns:
                df = df.drop(columns=[legacy_col])

    # if populate_sdss:
    #     print("Populating SDSS fields...")
    #     df = populate_sdss_fields(df)

    # if ("ebv_wu" not in df.columns) or df["ebv_wu"].isna().all():
    #     print("Populating SDSS fields...")
    #     df = populate_sdss_fields(df)

    #df = populate_sdss_rchi2_fields(df)

    if "dropped_bands" in df.columns:
        df["dropped_bands"] = df["dropped_bands"].apply(_normalize_dropped_bands)
        df["len_dropped_bands"] = df["dropped_bands"].apply(len)
    else:
        df["dropped_bands"] = [[] for _ in range(len(df))]
        df["len_dropped_bands"] = 0

    df = _mask_invalid_wu_bhmass(df)

    required_mags_mean_cols = [f"mags_mean_{b}" for b in ("u", "g", "r", "i")]
    missing_mags_mean_cols = [col for col in required_mags_mean_cols if col not in df.columns]
    if missing_mags_mean_cols:
        raise ValueError(
            "Flat AGN input is missing required per-band magnitude means: "
            f"{missing_mags_mean_cols}. "
            "Only mags_mean_<band> columns are supported "
            "(legacy mags_mean arrays and mags_mean_0..N are unsupported)."
        )

    required_blr_cols = [f"dlog_amp_blr_{b}" for b in ("u", "g", "r", "i")]
    missing_var_cols = [col for col in required_blr_cols if col not in df.columns]
    missing_jitter_bands = []
    for b in ("u", "g", "r", "i"):
        legacy_col = f"log_jitter_{b}"
        survey_cols = [f"log_jitter_{b}_{survey}" for survey in HUBBLE_JITTER_SURVEYS]
        if legacy_col not in df.columns and not any(col in df.columns for col in survey_cols):
            missing_jitter_bands.append(b)
    if missing_var_cols:
        raise ValueError(
            "Flat AGN input is missing required variability columns: "
            f"{missing_var_cols}."
        )
    if missing_jitter_bands:
        raise ValueError(
            "Flat AGN input is missing jitter coverage for band(s): "
            f"{missing_jitter_bands}. Expected either legacy log_jitter_<band> "
            "columns or new log_jitter_<band>_<survey> columns."
        )

    if "dropped_bands" not in df.columns:
        raise ValueError(
            "Flat AGN input is missing required column 'dropped_bands'."
        )

    dropped_bands = df['dropped_bands']
    jitter_total_sq = np.zeros(len(df))
    amp_delta_blr_total_sq = np.zeros(len(df))

    for b in ['u', 'g', 'r', 'i']:
        dropped_band = dropped_bands.apply(lambda s: b in s).to_numpy(dtype=bool)
        survey_jitter_sq = np.zeros(len(df))
        survey_cols = [f"log_jitter_{b}_{survey}" for survey in HUBBLE_JITTER_SURVEYS]
        available_survey_cols = [col for col in survey_cols if col in df.columns]
        if available_survey_cols:
            for col in available_survey_cols:
                survey = col.removeprefix(f"log_jitter_{b}_")
                jitter = np.exp(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float))
                jitter[~np.isfinite(jitter)] = np.nan
                jitter[dropped_band] = 0.0
                df[f"jitter_{b}_{survey}"] = jitter
                survey_jitter_sq = survey_jitter_sq + np.nan_to_num(jitter, nan=0.0) ** 2
            jitter = np.sqrt(survey_jitter_sq)
        else:
            # Legacy Hubble inputs stored unsuffixed per-band jitter in base-10 log space.
            jitter = 10**pd.to_numeric(df[f'log_jitter_{b}'], errors="coerce").to_numpy(dtype=float)
            jitter[~np.isfinite(jitter)] = np.nan
            jitter[dropped_band] = 0.0
            survey_jitter_sq = np.nan_to_num(jitter, nan=0.0) ** 2
        df[f'jitter_{b}'] = jitter
        jitter_total_sq = jitter_total_sq + survey_jitter_sq

        #df.loc[dropped_bands.apply(lambda s: b in s), f'dlog_amp_blr_{b}'] = np.nan
        amp_delta_blr = 10**df[f'dlog_amp_blr_{b}'].values
        amp_delta_blr[dropped_band] = 0.0
        amp_delta_blr_total_sq = amp_delta_blr_total_sq + amp_delta_blr**2
        df[f'amp_delta_blr_{b}'] = amp_delta_blr
        blr2_col = f'dlog_amp_blr2_{b}'
        if blr2_col in df.columns:
            amp_delta_blr2 = 10**df[blr2_col].values
            amp_delta_blr2[dropped_band] = 0.0
            amp_delta_blr_total_sq = amp_delta_blr_total_sq + amp_delta_blr2**2
            df[f'amp_delta_blr2_{b}'] = amp_delta_blr2

    jitter_total = np.sqrt(jitter_total_sq)
    log_jitter_total = np.full(len(df), np.nan, dtype=float)
    valid_jitter_total = np.isfinite(jitter_total) & (jitter_total > 0.0)
    log_jitter_total[valid_jitter_total] = np.log10(jitter_total[valid_jitter_total])
    df['log_jitter_total'] = log_jitter_total
    df['dlog_amp_blr_total'] = np.log10(np.sqrt(amp_delta_blr_total_sq))

    # df['log_sigma_uv'] = df['log_sigma_uv'] + 1/2 * np.log10(1 + df['z'])

    if spectra_fit_csv is not None:
        print("Populating spectra fit data from:", spectra_fit_csv)
        df = populate_spectra_fit(df, spectra_fit_csv)
    else:
        print("[WARNING] spectra_fit_csv not provided, assuming spectral fit fields are in agn h5 file")
        if 'alpha_lambda' not in df.columns:
            raise ValueError("spectra_fit_csv not provided and spectral fields not found in agn h5 file")
            #raise ValueError("spectra_fit_csv must be provided if alpha_lambda not in agn h5 file")

    if "f_host_2500_psf" not in df.columns and "frac_host_psf_2500" in df.columns:
        df["f_host_2500_psf"] = pd.to_numeric(df["frac_host_psf_2500"], errors="coerce")
    if "f_host_2500_psf_err" not in df.columns and "frac_host_psf_2500_err" in df.columns:
        df["f_host_2500_psf_err"] = pd.to_numeric(df["frac_host_psf_2500_err"], errors="coerce")

    if "f_host_2500_psf" in df.columns:
        if "f_host_2500" in df.columns and "f_host_fiber_2500" not in df.columns:
            df["f_host_fiber_2500"] = pd.to_numeric(df["f_host_2500"], errors="coerce")
        if "f_host_2500_err" in df.columns and "f_host_fiber_2500_err" not in df.columns:
            df["f_host_fiber_2500_err"] = pd.to_numeric(df["f_host_2500_err"], errors="coerce")

        df["f_host_2500"] = pd.to_numeric(df["f_host_2500_psf"], errors="coerce")
        if "f_host_2500_psf_err" in df.columns:
            df["f_host_2500_err"] = pd.to_numeric(df["f_host_2500_psf_err"], errors="coerce")
        print(
            "Using f_host_2500_psf from PSF posterior reconstruction as the primary host fraction; "
            "completeness reads f_host_2500_psf explicitly."
        )

    if "f_host_2500" in df.columns:
        missing_f_host_2500 = pd.to_numeric(df["f_host_2500"], errors="coerce").isna()
        if np.any(missing_f_host_2500):
            df.loc[missing_f_host_2500, "f_host_2500"] = 0.0
            if "f_host_2500_err" in df.columns:
                df.loc[missing_f_host_2500, "f_host_2500_err"] = 0.0
            print(f"Filled {int(np.count_nonzero(missing_f_host_2500))} NaN f_host_2500 values with 0.0")

    if "log_sigma_uv" in df.columns:
        df["log_sigma_uv_uncorrected"] = pd.to_numeric(df["log_sigma_uv"], errors="coerce")

    df, lc_point_count_cols = add_light_curve_point_count_column(df)
    if lc_point_count_cols:
        excluded = ", ".join(LIGHT_CURVE_N_POINTS_EXCLUDED_BANDS)
        print(
            f"Computed {LIGHT_CURVE_N_POINTS_COLUMN} from: "
            f"{', '.join(lc_point_count_cols)}"
            f" excluding bands: {excluded}"
        )

    # Use the PL/total fraction at 2500 A for the sigma_uv dilution correction.
    if correct_sigma_uv_host:
        required_cols = {"log_sigma_uv_uncorrected", "f_PL", "log_sigma_uv_std_psd"}
        if required_cols.issubset(df.columns):
            f_pl = pd.to_numeric(df["f_PL"], errors="coerce")
            valid_dilution = np.isfinite(f_pl) & (f_pl > 0.0) & (f_pl <= 1.0)
            ln10 = np.log(10.0)

            log_sigma_uv_std_psd_uncorrected = pd.to_numeric(
                df["log_sigma_uv_std_psd"], errors="coerce"
            )
            df["log_sigma_uv_std_psd_uncorrected"] = log_sigma_uv_std_psd_uncorrected

            if "f_PL_err" in df.columns:
                f_pl_err = pd.to_numeric(df["f_PL_err"], errors="coerce").fillna(0.0)
            else:
                f_pl_err = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
            hostcorr_sigma_term = np.where(
                valid_dilution,
                f_pl_err.to_numpy(dtype=float) / (f_pl.to_numpy(dtype=float) * ln10),
                0.0,
            )
            df["log_sigma_uv_hostcorr_err"] = hostcorr_sigma_term

            df["sigma_uv_hostcorr_factor"] = np.where(valid_dilution, 1.0 / f_pl, np.nan)
            df["log_sigma_uv"] = np.where(
                valid_dilution,
                df["log_sigma_uv_uncorrected"] + np.log10(df["sigma_uv_hostcorr_factor"]),
                df["log_sigma_uv_uncorrected"],
            )
            corrected_sigma_var = (
                np.square(log_sigma_uv_std_psd_uncorrected.to_numpy(dtype=float))
                + np.square(hostcorr_sigma_term)
            )
            df["log_sigma_uv_std_psd_corrected"] = np.where(
                valid_dilution,
                np.sqrt(corrected_sigma_var),
                log_sigma_uv_std_psd_uncorrected,
            )
            df["log_sigma_uv_std_psd"] = df["log_sigma_uv_std_psd_corrected"]
            print(
                "Applied sigma_uv dilution correction using f_PL: "
                "sigma_uv_corrected = sigma_uv / f_PL"
            )
            delta_log_sigma = df["log_sigma_uv"] - df["log_sigma_uv_uncorrected"]
            valid_expected_increase = valid_dilution & np.isfinite(delta_log_sigma)
            if np.any(valid_expected_increase & (delta_log_sigma < 0.0)):
                bad_rows = df.loc[
                    valid_expected_increase & (delta_log_sigma < 0.0),
                    ["object_id", "f_PL", "log_sigma_uv_uncorrected", "log_sigma_uv"],
                ]
                raise ValueError(
                    "Dilution-corrected log_sigma_uv should be larger or equal than the uncorrected value "
                    "for all valid f_PL rows. Offending rows:\n"
                    f"{bad_rows.head(10).to_string(index=False)}"
                )
            if np.any(valid_expected_increase):
                print(
                    "Sigma_uv dilution correction sanity check: "
                    f"median delta={np.nanmedian(delta_log_sigma[valid_expected_increase]):.4f} dex, "
                    f"min delta={np.nanmin(delta_log_sigma[valid_expected_increase]):.4f} dex"
                )
            corrected_sigma_delta = (
                df["log_sigma_uv_std_psd"].to_numpy(dtype=float)
                - df["log_sigma_uv_std_psd_uncorrected"].to_numpy(dtype=float)
            )
            valid_sigma_delta = valid_dilution & np.isfinite(corrected_sigma_delta)
            if np.any(valid_sigma_delta):
                print(
                    "Sigma_uv dilution uncertainty propagation sanity check: "
                    f"median delta={np.nanmedian(corrected_sigma_delta[valid_sigma_delta]):.4f} dex, "
                    f"max delta={np.nanmax(corrected_sigma_delta[valid_sigma_delta]):.4f} dex"
                )
            plot_sigma_uv_host_correction(
                df,
                plot_path=plot_path,
                show=False,
                filename="sigma_uv_host_correction_comparison_precut.pdf",
            )
        else:
            missing_cols = sorted(required_cols - set(df.columns))
            raise KeyError(
                "correct_sigma_uv_host=True requires 'log_sigma_uv', 'f_PL', "
                f"and 'log_sigma_uv_std_psd'. Missing: {missing_cols}"
            )

    if {"z", "apparent_mag_2500", "f_host_2500_psf"}.issubset(df.columns):
        plot_f_host_2500_vs_l2500(
            df,
            plot_path=plot_path,
            show=False,
            filename="f_host_2500_vs_l2500_precut.pdf",
            f_host_col="f_host_2500_psf",
            f_host_label=r"$f_{\rm host,2500}^{\rm PSF}$",
        )
    plot_blr_diagnostics_summary(
        df,
        plot_path=plot_path,
        show=False,
        filename="blr_precut.pdf",
    )
    if "apparent_mag_2500" in df.columns:
        plot_light_curve_n_points_vs_apparent_mag(
            df,
            plot_path=plot_path,
            show=False,
            filename="light_curve_n_points_vs_apparent_mag_precut.pdf",
            exclude_bands=LIGHT_CURVE_N_POINTS_EXCLUDED_BANDS,
        )
    if {"z", "apparent_mag_2500", "alpha_lambda"}.issubset(df.columns):
        plot_alpha_lambda_vs_l2500(
            df,
            plot_path=plot_path,
            show=False,
            filename="alpha_lambda_vs_l2500_precut.pdf",
        )
    if "alpha_lambda" in df.columns:
        plot_alpha_lambda_vs_redshift(
            df,
            plot_path=plot_path,
            show=False,
            filename="alpha_lambda_vs_redshift_precut.pdf",
        )
        plot_alpha_lambda_histogram(
            df,
            plot_path=plot_path,
            show=False,
            filename="alpha_lambda_histogram_precut.pdf",
        )
    if {"alpha_lambda", "eta_sigma"}.issubset(df.columns):
        plot_alpha_lambda_vs_eta_sigma(
            df,
            plot_path=plot_path,
            show=False,
            filename="alpha_lambda_vs_eta_sigma_precut.pdf",
        )
    if {"z", "log_tau_uv_rf", "log_sigma_uv"}.issubset(df.columns):
        plot_tau_sigma_vs_redshift(
            df,
            plot_path=plot_path,
            show=False,
            filename="tau_sigma_vs_redshift_precut.pdf",
        )
        plot_sigma_uv_vs_tau_uv_rf(
            df,
            plot_path=plot_path,
            show=False,
            filename="sigma_uv_vs_tau_uv_rf_precut.pdf",
        )
    if {
        "log_sigma_uv_err",
        "log_sigma_uv_std_psd",
        "log_tau_uv_rf_err",
        "log_tau_uv_rf_std_psd",
    }.issubset(df.columns):
        plot_sigma_tau_err_std_psd_comparison(
            df,
            plot_path=plot_path,
            show=False,
            filename="sigma_tau_err_std_psd_comparison_precut.pdf",
        )
    if {"log_sigma_uv", "variability_chi_sq_red_g"}.issubset(df.columns):
        plot_sigma_uv_vs_variability_chi_sq_red_g(
            df,
            plot_path=plot_path,
            show=False,
            filename="sigma_uv_vs_variability_chi_sq_red_g_precut.pdf",
        )
    if {"z", "apparent_mag_2500", "log_tau_uv_rf", "log_sigma_uv"}.issubset(df.columns):
        plot_l2500_vs_uv_variability_fiducial(
            df,
            plot_path=plot_path,
            show=False,
            filename="l2500_vs_uv_variability_fiducial_precut.pdf",
        )
    if {"z", "apparent_mag_2500", "eta_sigma"}.issubset(df.columns):
        plot_l2500_vs_eta_sigma_fiducial(
            df,
            plot_path=plot_path,
            show=False,
            filename="l2500_vs_eta_sigma_fiducial_precut.pdf",
        )
    if {"z", "eta_tau", "eta_sigma"}.issubset(df.columns):
        plot_eta_tau_sigma_vs_redshift(
            df,
            plot_path=plot_path,
            show=False,
            filename="eta_tau_sigma_vs_redshift_precut.pdf",
        )
    if {"z", "linear_trend"}.issubset(df.columns):
        plot_linear_trend_vs_redshift(
            df,
            plot_path=plot_path,
            show=False,
            filename="linear_trend_vs_redshift_precut.pdf",
        )
    if (
        "z" in df.columns
        and (
            {"log_sigma_uv", "dlog_amp_bc"}.issubset(df.columns)
            or any(
                (f"amp_bc_{band}" in df.columns) and (f"bc_weight_{band}" in df.columns)
                for band in ("u", "g", "r", "i", "z")
            )
        )
    ):
        plot_sigma_bc_vs_redshift(
            df,
            plot_path=plot_path,
            show=False,
            filename="sigma_bc_vs_redshift_precut.pdf",
        )
    if (
        "f_bc_3000" in df.columns
        and (
            {"log_sigma_uv", "dlog_amp_bc"}.issubset(df.columns)
            or any(
                (f"amp_bc_{band}" in df.columns) and (f"bc_weight_{band}" in df.columns)
                for band in ("u", "g", "r", "i", "z")
            )
        )
    ):
        plot_sigma_bc_vs_frac_bc(
            df,
            plot_path=plot_path,
            show=False,
            filename="sigma_bc_vs_frac_bc_precut.pdf",
        )
    if {"z", "apparent_mag_2500"}.issubset(df.columns) and any(
        (f"log_lag_bc_{band}_RF" in df.columns) or (f"lag_bc_{band}" in df.columns)
        for band in ("u", "g", "r", "i", "z")
    ):
        plot_bc_lag_vs_l2500(
            df,
            plot_path=plot_path,
            show=False,
            filename="bc_lag_vs_l2500_precut.pdf",
        )
    if {"log_tau_uv_rf", "log_sigma_uv", "LOGMBH", "LOGLEDD_RATIO"}.issubset(df.columns):
        plot_tau_sigma_vs_wu_catalog(
            df,
            plot_path=plot_path,
            show=False,
            filename="tau_sigma_vs_wu_catalog_precut.pdf",
        )
    if {"log_tau_uv_rf", "log_sigma_uv", "LOGMBH", "z", "apparent_mag_2500"}.issubset(df.columns):
        plot_suberlak_style_sigma_tau_fits(
            df,
            plot_path=plot_path,
            show=False,
            filename="suberlak_style_sigma_tau_fits_precut.pdf",
            sample_label="precut M_2500",
        )
    if {"log_tau_uv_rf", "log_sigma_uv", "LOGMBH", "LOGLBOL_CORRECTED"}.issubset(df.columns):
        plot_suberlak_style_sigma_tau_fits(
            df,
            plot_path=plot_path,
            show=False,
            filename="suberlak_style_sigma_tau_fits_Mi_precut.pdf",
            abs_mag_column="M_i_Wu_z2",
            sample_label="precut M_i_Wu_z2",
        )
    if {"z", "log_tau_fast_uv", "log_tau_uv_rf", "log_sigma_uv"}.issubset(df.columns):
        plot_fast_vs_uv_variability(
            df,
            plot_path=plot_path,
            show=False,
            filename="fast_vs_uv_variability_precut.pdf",
        )
    if {"log_sigma_uv", "log_sigma_uv_sf", "log_tau_uv_rf_sf"}.issubset(df.columns) and (
        {"log_tau_uv_rf"}.issubset(df.columns)
        or {"log_tau_uv", "z"}.issubset(df.columns)
    ):
        plot_sf_vs_uv_variability(
            df,
            plot_path=plot_path,
            show=False,
            filename="sf_vs_uv_variability_precut.pdf",
        )
    if {"log_sigma_sf_ref_band", "log_tau_sf_ref_band", "log_sigma_band_g", "log_tau_band_g_RF"}.issubset(df.columns):
        plot_sf_ref_band_vs_model_g(
            df,
            plot_path=plot_path,
            show=False,
            filename="sf_ref_band_vs_model_g_precut.pdf",
        )
    if {"z", "apparent_mag_2500"}.issubset(df.columns) and any(
        (f"log_lag_blr_{band}_RF" in df.columns) or (f"log_lag_blr2_{band}_RF" in df.columns)
        for band in ("u", "g", "r", "i", "z")
    ):
        plot_blr_line_lags_vs_l2500_fiducial(
            df,
            plot_path=plot_path,
            show=False,
            prob_thresh=0.8,
            filename="blr_line_lags_vs_l2500_fiducial_precut.pdf",
            assignment_probabilities_filename="blr_line_assignment_probabilities_fiducial_precut.pdf",
        )
    if "log_sigma_uv" in df.columns:
        if any(f"dlog_amp_blr_{band}" in df.columns for band in ("u", "g", "r", "i", "z")):
            plot_blr_lag_vs_amp_by_band(
                df,
                plot_path=plot_path,
                show=False,
                lag_suffix="",
                filename="blr_lag_vs_amp_by_band_precut.pdf",
            )
            plot_blr_amp_vs_redshift_by_band(
                df,
                plot_path=plot_path,
                show=False,
                lag_suffix="",
                filename="blr_amp_vs_redshift_by_band_precut.pdf",
            )
        if any(
            (f"log_lag_blr_{band}_RF" in df.columns) or (f"log_lag_blr_{band}" in df.columns)
            for band in ("u", "g", "r", "i", "z")
        ):
            plot_blr_lag_vs_redshift_by_band(
                df,
                plot_path=plot_path,
                show=False,
                lag_suffix="",
                filename="blr_lag_vs_redshift_by_band_precut.pdf",
            )
        if any(f"dlog_amp_blr2_{band}" in df.columns for band in ("u", "g", "r", "i", "z")):
            plot_blr_lag_vs_amp_by_band(
                df,
                plot_path=plot_path,
                show=False,
                lag_suffix="2",
                filename="blr_lag2_vs_amp_by_band_precut.pdf",
            )
            plot_blr_amp_vs_redshift_by_band(
                df,
                plot_path=plot_path,
                show=False,
                lag_suffix="2",
                filename="blr2_amp_vs_redshift_by_band_precut.pdf",
            )
        if any(
            (f"log_lag_blr2_{band}_RF" in df.columns) or (f"log_lag_blr2_{band}" in df.columns)
            for band in ("u", "g", "r", "i", "z")
        ):
            plot_blr_lag_vs_redshift_by_band(
                df,
                plot_path=plot_path,
                show=False,
                lag_suffix="2",
                filename="blr_lag2_vs_redshift_by_band_precut.pdf",
            )

    df = populate_xray(df)
    
    # if lc_info_csv is not None:
    #     print("Populating LC info from:", lc_info_csv)
    #     df = populate_lc_info(df, lc_info_csv)
    # else:
    #     print("[WARNING] lc_info_csv not provided")

    num_quasars_z_0_1_before = len(df[(df['z'] > 0) & (df['z'] <= 1.0)])
    num_quasars_z_gt_3_before = len(df[df['z'] > 3])
    print("Number of quasars with 0 < z <= 1.0:", num_quasars_z_0_1_before)
    print("Number of quasars with z > 3:", num_quasars_z_gt_3_before)
    print("Highest redshift quasar:", df['z'].max())

    df_all = df.copy()

    if only_load:
        _finalize_cut_report()
        return df, df_all

    df['log_t_rf_length'] = np.log10(df['t_rf_length'])

    df['log_f_host_2500'] = np.where(df['f_host_2500'] > 0, np.log10(df['f_host_2500']), np.nan)
    if {"apparent_mag_2500", "apparent_mag_2500_err"}.issubset(df.columns):
        mag_2500 = pd.to_numeric(df["apparent_mag_2500"], errors="coerce").to_numpy(dtype=float)
        mag_2500_err = pd.to_numeric(df["apparent_mag_2500_err"], errors="coerce").to_numpy(dtype=float)
        df["rel_apparent_mag_2500_err"] = np.divide(
            mag_2500_err,
            np.maximum(np.abs(mag_2500), 1e-8),
            out=np.full_like(mag_2500_err, np.nan, dtype=float),
            where=np.isfinite(mag_2500_err) & np.isfinite(mag_2500),
        )

    df = df.reset_index(drop=True)
    
    # Apply a small hand-maintained exclusion list.
    exclusion_object_ids = []
    mask_exclude = ~df['object_id'].astype(str).isin(exclusion_object_ids)

    mask_exclude &= (~df["sdss_name"].astype(str).isin(EXCLUDED_SDSS_NAMES))
    df = _record_cut("exclusion_list", "sdss_name/object_id exclusion list", df, mask_exclude)


    # Remove outliers listed in external CSV files.
    for exclude_csv in exclude_object_ids_csv:
        if os.path.exists(exclude_csv):
            exclude_df = pd.read_csv(exclude_csv)
            exclude_ids = set(exclude_df['object_id'].astype(str))
            mask_exclude = ~df['object_id'].astype(str).isin(exclude_ids)
            plot_cut_diagnostics(df.copy(), df[mask_exclude], bins=30, cut_info="exclude csv")
            df = _record_cut(
                f"exclude_csv:{Path(exclude_csv).name}",
                f"object_id not in {Path(exclude_csv).name}",
                df,
                mask_exclude,
            )
        else:
            _append_cut_report_row(
                cut_rows,
                step=f"exclude_csv:{Path(exclude_csv).name}",
                criterion=f"missing file: {exclude_csv}",
                before=len(df),
                kept=len(df),
                status="warning",
            )
            print(f"[WARNING] Exclusion CSV not found: {exclude_csv}")

    blr_amp_cuts = build_dlog_amp_blr_cuts()
    for col, lower, upper in blr_amp_cuts:
        cut_desc = f"{col} in {_format_cut_bounds(lower, upper, upper_inclusive=False, allow_missing=True)}"
        if col not in df.columns:
            _append_cut_report_row(
                cut_rows,
                step=f"blr_amp:{col}",
                criterion=cut_desc,
                before=len(df),
                kept=len(df),
                status="skipped",
            )
            continue
        mask = np.ones(len(df), dtype=bool)
        if lower is not None:
            mask &= df[col] >= lower
        if upper is not None:
            mask &= df[col] < upper
        mask |= df[col].isna()
        plot_cut_diagnostics(df.copy(), df[mask], bins=30, cut_info=cut_desc)
        df = _record_cut(f"blr_amp:{col}", cut_desc, df, mask)

    cuts = build_agn_cuts()

    if apply_cut:
        mask = np.ones(len(df), dtype=bool)
        for col, lower, upper in cuts:
            cut_desc = f"{col} in {_format_cut_bounds(lower, upper, upper_inclusive=True)}"
            if col not in df.columns:
                _append_cut_report_row(
                    cut_rows,
                    step=f"agn_scalar:{col}",
                    criterion=cut_desc,
                    before=len(df),
                    kept=len(df),
                    status="skipped",
                )
                continue
            col_mask = np.ones(len(df), dtype=bool)
            if lower is not None:
                col_mask &= df[col] >= lower
            if upper is not None:
                col_mask &= df[col] <= upper
            plot_cut_diagnostics(df.copy(), df[col_mask], bins=30, cut_info=cut_desc)
            df = _record_cut(f"agn_scalar:{col}", cut_desc, df, col_mask)

        if "dlog_amp_bc" in df.columns:
            bc_amp_upper = LOG_AMP_DELTA_BC_UPPER
            bc_amp_mask = (
                pd.to_numeric(df["dlog_amp_bc"], errors="coerce").to_numpy(dtype=float)
                <= bc_amp_upper
            ) | df["dlog_amp_bc"].isna().to_numpy(dtype=bool)
            cut_desc = f"dlog_amp_bc in (-inf, {bc_amp_upper}] or NaN"
            plot_cut_diagnostics(df.copy(), df[bc_amp_mask], bins=30, cut_info=cut_desc)
            df = _record_cut("agn_scalar:dlog_amp_bc", cut_desc, df, bc_amp_mask)

        for frac_col, log_col, log_upper in (
            ("f_bc_3000", "log_f_bc_3000", LOG_F_BC_3000_MAX),
            ("f_fe_uv_3000", "log_f_fe_uv_3000", LOG_F_FE_UV_3000_MAX),
        ):
            if frac_col not in df.columns:
                _append_cut_report_row(
                    cut_rows,
                    step=f"agn_scalar:{log_col}",
                    criterion=f"{log_col} <= {log_upper} or NaN/non-positive",
                    before=len(df),
                    kept=len(df),
                    status="skipped",
                )
                continue
            frac_vals = pd.to_numeric(df[frac_col], errors="coerce").to_numpy(dtype=float)
            frac_upper = 10.0**log_upper
            frac_mask = (~np.isfinite(frac_vals)) | (frac_vals <= 0.0) | (frac_vals <= frac_upper)
            cut_desc = f"{log_col} <= {log_upper} or NaN/non-positive"
            plot_cut_diagnostics(df.copy(), df[frac_mask], bins=30, cut_info=cut_desc)
            df = _record_cut(f"agn_scalar:{log_col}", cut_desc, df, frac_mask)

        if "rel_apparent_mag_2500_err" in df.columns:
            rel_mag_err = pd.to_numeric(
                df["rel_apparent_mag_2500_err"],
                errors="coerce",
            ).to_numpy(dtype=float)
            rel_mag_err_mask = (
                (~np.isfinite(rel_mag_err))
                | (rel_mag_err < REL_APPARENT_MAG_2500_ERR_MAX)
            )
            cut_desc = (
                f"rel_apparent_mag_2500_err < {REL_APPARENT_MAG_2500_ERR_MAX} or NaN"
            )
            plot_cut_diagnostics(df.copy(), df[rel_mag_err_mask], bins=30, cut_info=cut_desc)
            df = _record_cut(
                "agn_scalar:rel_apparent_mag_2500_err",
                cut_desc,
                df,
                rel_mag_err_mask,
            )
    df = df.reset_index(drop=True)

    if residuals_sigma_clip is not None and residuals_csv is not None:
        if os.path.exists(residuals_csv):
            residual_df = pd.read_csv(residuals_csv)
            if 'residuals' not in residual_df.columns:
                raise ValueError(f"'residuals' column not found in {residuals_csv}")
            mu_zscore = dict(zip(residual_df['object_id'].astype(str), residual_df['mu_zscore']))
            df['mu_zscore'] = df['object_id'].astype(str).map(mu_zscore)
            mask_residual = df['mu_zscore'].abs() < residuals_sigma_clip
            plot_cut_diagnostics(df.copy(), df[mask_residual], bins=30, cut_info=f"|mu_zscore|<{residuals_sigma_clip}")
            df = _record_cut(
                "residual_sigma_clip",
                f"|mu_zscore| < {residuals_sigma_clip}",
                df,
                mask_residual,
            )
            df = df.drop(columns=['mu_zscore'])
        else:
            _append_cut_report_row(
                cut_rows,
                step="residual_sigma_clip",
                criterion=f"missing file: {residuals_csv}",
                before=len(df),
                kept=len(df),
                status="warning",
            )
            print(f"[WARNING] Residual CSV not found: {residuals_csv}")
            raise ValueError(f"Residual CSV not found: {residuals_csv}")

    num_quasars_z_0_1 = len(df[(df['z'] > 0) & (df['z'] <= 1.0)])
    num_quasars_z_gt_3 = len(df[df['z'] > 3])
    print("Number of quasars with 0 < z <= 1.0:", num_quasars_z_0_1)
    print("Number of dropped quasars with 0 < z <= 1.0:", num_quasars_z_0_1_before - num_quasars_z_0_1)
    print("Number of quasars with z > 3:", num_quasars_z_gt_3)
    print(f"\nTotal number of objects removed by all cuts: {len(df_all) - len(df)}")
    print("Final number of quasars:", len(df))
    if {"z", "apparent_mag_2500", "f_host_2500_psf"}.issubset(df.columns):
        plot_f_host_2500_vs_l2500(
            df,
            plot_path=plot_path,
            show=False,
            filename="f_host_2500_vs_l2500_postcut.pdf",
            f_host_col="f_host_2500_psf",
            f_host_label=r"$f_{\rm host,2500}^{\rm PSF}$",
        )
    plot_blr_diagnostics_summary(
        df,
        plot_path=plot_path,
        show=False,
        filename="blr_postcut.pdf",
    )
    if "apparent_mag_2500" in df.columns:
        plot_light_curve_n_points_vs_apparent_mag(
            df,
            plot_path=plot_path,
            show=False,
            filename="light_curve_n_points_vs_apparent_mag_postcut.pdf",
            exclude_bands=LIGHT_CURVE_N_POINTS_EXCLUDED_BANDS,
        )
    if "alpha_lambda" in df.columns:
        plot_alpha_lambda_vs_redshift(
            df,
            plot_path=plot_path,
            show=False,
            filename="alpha_lambda_vs_redshift_postcut.pdf",
        )
    if {"z", "f_bc_3000", "f_fe_uv_3000"}.issubset(df.columns) and (
        "f_host_center" in df.columns or "f_host_2500" in df.columns
    ):
        spectral_fraction_cut_thresholds = {
            "f_bc_3000": 10.0**LOG_F_BC_3000_MAX,
            "f_fe_uv_3000": 10.0**LOG_F_FE_UV_3000_MAX,
            "f_host_2500": F_HOST_2500_MAX,
        }
        plot_spectral_fraction_vs_redshift(
            df,
            plot_path=plot_path,
            show=False,
            filename="spectral_fraction_vs_redshift_postcut.pdf",
            cut_thresholds=spectral_fraction_cut_thresholds,
        )
        if "object_id" in df_all.columns and "object_id" in df.columns:
            cut_object_ids = set(df["object_id"].astype(str))
            df_cut_sources = df_all.loc[
                ~df_all["object_id"].astype(str).isin(cut_object_ids)
            ].copy()
            plot_spectral_fraction_vs_redshift(
                df,
                plot_path=plot_path,
                show=False,
                z_range=z_range,
                df_cut_sources=df_cut_sources,
                filename="spectral_fraction_vs_redshift_cuts.pdf",
                cut_thresholds=spectral_fraction_cut_thresholds,
            )
    if {"g_raw_mean_slope", "g_resid_mean_slope"}.issubset(df.columns):
        for m2500_cut in (23.0, 22.5, 22.0, 21.5):
            plot_g_band_drift_slope_histograms(
                df,
                slope_kind="mean",
                z_min=0.8,
                z_max=1.2,
                m2500_max=m2500_cut,
                plot_path=plot_path,
                show=False,
                filename=f"g_band_mean_slope_histograms_postcut_z0p8to1p2_m2500lt{str(m2500_cut).replace('.', 'p')}.pdf",
            )
    if {"g_raw_var_slope", "g_resid_var_slope"}.issubset(df.columns):
        plot_g_band_drift_slope_histograms(
            df,
            slope_kind="var",
            z_min=0.8,
            z_max=1.2,
            plot_path=plot_path,
            show=False,
            filename="g_band_var_slope_histograms_postcut_z0p8to1p2.pdf",
        )
    if {"log_tau_uv_rf", "log_sigma_uv"}.issubset(df.columns):
        plot_sigma_uv_vs_tau_uv_rf(
            df,
            plot_path=plot_path,
            show=False,
            filename="sigma_uv_vs_tau_uv_rf_postcut.pdf",
            dynamic_axes=True,
        )
    if {
        "log_sigma_uv_err",
        "log_sigma_uv_std_psd",
        "log_tau_uv_rf_err",
        "log_tau_uv_rf_std_psd",
    }.issubset(df.columns):
        plot_sigma_tau_err_std_psd_comparison(
            df,
            plot_path=plot_path,
            show=False,
            filename="sigma_tau_err_std_psd_comparison_postcut.pdf",
        )
    if {"log_sigma_uv", "variability_chi_sq_red_g"}.issubset(df.columns):
        plot_sigma_uv_vs_variability_chi_sq_red_g(
            df,
            plot_path=plot_path,
            show=False,
            filename="sigma_uv_vs_variability_chi_sq_red_g_postcut.pdf",
        )
    if {"z", "apparent_mag_2500", "log_tau_uv_rf", "log_sigma_uv"}.issubset(df.columns):
        plot_l2500_vs_uv_variability_fiducial(
            df,
            plot_path=plot_path,
            show=False,
            filename="l2500_vs_uv_variability_fiducial_postcut.pdf",
            dynamic_axes=True,
        )
    if {"z", "apparent_mag_2500", "eta_sigma"}.issubset(df.columns):
        plot_l2500_vs_eta_sigma_fiducial(
            df,
            plot_path=plot_path,
            show=False,
            filename="l2500_vs_eta_sigma_fiducial_postcut.pdf",
            dynamic_axes=True,
        )
    if {"z", "eta_tau", "eta_sigma"}.issubset(df.columns):
        plot_eta_tau_sigma_vs_redshift(
            df,
            plot_path=plot_path,
            show=False,
            filename="eta_tau_sigma_vs_redshift_postcut.pdf",
        )
    if {"z", "linear_trend"}.issubset(df.columns):
        plot_linear_trend_vs_redshift(
            df,
            plot_path=plot_path,
            show=False,
            filename="linear_trend_vs_redshift_postcut.pdf",
        )
    if (
        "z" in df.columns
        and (
            {"log_sigma_uv", "dlog_amp_bc"}.issubset(df.columns)
            or any(
                (f"amp_bc_{band}" in df.columns) and (f"bc_weight_{band}" in df.columns)
                for band in ("u", "g", "r", "i", "z")
            )
        )
    ):
        plot_sigma_bc_vs_redshift(
            df,
            plot_path=plot_path,
            show=False,
            filename="sigma_bc_vs_redshift_postcut.pdf",
        )
    if (
        "f_bc_3000" in df.columns
        and (
            {"log_sigma_uv", "dlog_amp_bc"}.issubset(df.columns)
            or any(
                (f"amp_bc_{band}" in df.columns) and (f"bc_weight_{band}" in df.columns)
                for band in ("u", "g", "r", "i", "z")
            )
        )
    ):
        plot_sigma_bc_vs_frac_bc(
            df,
            plot_path=plot_path,
            show=False,
            filename="sigma_bc_vs_frac_bc_postcut.pdf",
        )
    if {"z", "apparent_mag_2500"}.issubset(df.columns) and any(
        (f"log_lag_bc_{band}_RF" in df.columns) or (f"lag_bc_{band}" in df.columns)
        for band in ("u", "g", "r", "i", "z")
    ):
        plot_bc_lag_vs_l2500(
            df,
            plot_path=plot_path,
            show=False,
            filename="bc_lag_vs_l2500_postcut.pdf",
        )
    if {"log_sigma_uv", "log_sigma_uv_sf", "log_tau_uv_rf_sf"}.issubset(df.columns) and (
        {"log_tau_uv_rf"}.issubset(df.columns)
        or {"log_tau_uv", "z"}.issubset(df.columns)
    ):
        plot_sf_vs_uv_variability(
            df,
            plot_path=plot_path,
            show=False,
            filename="sf_vs_uv_variability_postcut.pdf",
        )
    if {"log_tau_uv_rf", "log_sigma_uv", "LOGMBH", "z", "apparent_mag_2500"}.issubset(df.columns):
        plot_suberlak_style_sigma_tau_fits(
            df,
            plot_path=plot_path,
            show=False,
            filename="suberlak_style_sigma_tau_fits_postcut.pdf",
            sample_label="postcut M_2500",
        )
    if {"log_tau_uv_rf", "log_sigma_uv", "LOGMBH", "LOGLBOL_CORRECTED"}.issubset(df.columns):
        plot_suberlak_style_sigma_tau_fits(
            df,
            plot_path=plot_path,
            show=False,
            filename="suberlak_style_sigma_tau_fits_Mi_postcut.pdf",
            abs_mag_column="M_i_Wu_z2",
            sample_label="postcut M_i_Wu_z2",
        )
    if {"log_sigma_sf_ref_band", "log_tau_sf_ref_band", "log_sigma_band_g", "log_tau_band_g_RF"}.issubset(df.columns):
        plot_sf_ref_band_vs_model_g(
            df,
            plot_path=plot_path,
            show=False,
            filename="sf_ref_band_vs_model_g_postcut.pdf",
        )
    plot_cut_diagnostics(df_all.copy(), df.copy(), bins=30, cut_info="all cuts")
    colorpanel_cols = [col for col in ("f_host_2500", "f_host_center", "f_bc_3000", "wrms") if col in df_all.columns]
    if len(colorpanel_cols) > 0 and "z" in df_all.columns and "apparent_mag_2500" in df_all.columns:
        cuts_plot_dir = os.path.join("plots", "hubble", "cuts")
        os.makedirs(cuts_plot_dir, exist_ok=True)
        colorpanel_result = plot_m2500_vs_z_colorpanels(
            df_all,
            df_keep=df,
            color_cols=tuple(colorpanel_cols),
            z_range=z_range,
        )
        fig_colorpanels = None
        if colorpanel_result:
            if isinstance(colorpanel_result, tuple):
                fig_colorpanels = colorpanel_result[0]
            else:
                fig_colorpanels = colorpanel_result
        if fig_colorpanels is not None:
            fig_colorpanels.savefig(
                os.path.join(cuts_plot_dir, "m2500_vs_z_colorpanels.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig_colorpanels)
    plot_Mi_relation(df_all.copy())
    _finalize_cut_report()
    return df, df_all

def load_pantheon_data():
    """
    Load Pantheon+SH0ES data and build the covariance *for the selected subset*:
      selection = (zHD > 0.01) OR (IS_CALIBRATOR == True)

    Returns:
        df_pantheon_sel : pandas.DataFrame (filtered to the selection)
        sna_logdetCov   : float, log|C_sel|
        sna_L           : ndarray, Cholesky factor of C_sel (lower)
        sna_lower       : bool (True) for cho_solve
    """

    # Load the Pantheon+ catalog.
    print("Loading Pantheon+ supernova data...")
    pantheon_data_path = resolve_qvc_data_path("data/Pantheon+SH0ES.dat")
    df_pantheon = pd.read_csv(
        # "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat",
        pantheon_data_path,
        sep=r"\s+"
    )
    
    # Select cosmological SNe plus calibrators.
    is_calib = np.asarray(df_pantheon["IS_CALIBRATOR"], dtype=bool)
    sel_mask = (df_pantheon["zHD"].values > 0.01) | is_calib

    # Load the full covariance and apply the same selection.
    print("Loading SN covariance matrix...")
    n_sn = len(df_pantheon)
    pantheon_cov_path = resolve_qvc_data_path("data/Pantheon+SH0ES_STAT+SYS.cov")
    cov_flat = np.loadtxt(
        # "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov",
        pantheon_cov_path,
        skiprows=1
    )
    cov_matrix = cov_flat.reshape((n_sn, n_sn))
    assert cov_matrix.shape == (n_sn, n_sn), f"Expected ({n_sn},{n_sn}), got {cov_matrix.shape}"

    cov_sel = cov_matrix[sel_mask][:, sel_mask]
    df_pantheon_sel = df_pantheon.loc[sel_mask].reset_index(drop=True)

    # Cholesky factorization of the selected covariance.
    try:
        sna_L, sna_lower = cho_factor(cov_sel, lower=True)
    except np.linalg.LinAlgError:
        raise ValueError("Selected covariance submatrix is not positive-definite!")

    # Compute log|C| from the Cholesky factor.
    sna_logdetCov = 2.0 * np.sum(np.log(np.diag(sna_L)))

    n_sel = cov_sel.shape[0]
    print(f"Cholesky factorization successful. "
          f"Selected SNe: {n_sel} / {n_sn} "
          f"(kept {(sel_mask).sum()}; dropped {n_sn - (sel_mask).sum()}).")

    return df_pantheon_sel, sna_logdetCov, sna_L, sna_lower

LN10 = math.log(10.0)
LN2  = math.log(2.0)
TWOPI = 2.0 * math.pi
HALF_LN_TWOPI = 0.5 * math.log(TWOPI)
Phi = NormalDist()

def _bayes_factor_repr_from_delta(delta, delta_err=None):
    log10K = delta / LN10
    s_main = f"10^{log10K:.2f}"
    ci_tuple = None
    if delta_err is not None:
        lo = (delta - delta_err) / LN10
        hi = (delta + delta_err) / LN10
        s_ci = f"[10^{lo:.2f}, 10^{hi:.2f}]"
        ci_tuple = (lo, hi, s_ci)
    return log10K, s_main, ci_tuple

def _jeffreys_strength(abs_delta, thresholds):
    t1, t2, t3 = thresholds
    if abs_delta < t1:
        return "barely worth mentioning"
    elif abs_delta < t2:
        return "substantial"
    elif abs_delta < t3:
        return "Strong"
    else:
        return "very strong"

def odds_sigmas_from_delta(delta):
    """Return Kass&Raftery 1995 sigma Z from |Δln Z| odds, stably."""
    absD = abs(float(delta))
    sigma = np.sqrt(2.0 * absD)
    return sigma


def _odds_sigma_summary_from_delta(delta, delta_err):
    """Return central odds sigma and asymmetric bounds derived from |Δln Z| ± σΔ."""
    abs_delta = abs(float(delta))
    delta_err = float(delta_err)
    sigma = odds_sigmas_from_delta(abs_delta)
    sigma_lo = odds_sigmas_from_delta(max(abs_delta - delta_err, 0.0))
    sigma_hi = odds_sigmas_from_delta(abs_delta + delta_err)
    return {
        "sigma": sigma,
        "sigma_lo": sigma_lo,
        "sigma_hi": sigma_hi,
        "sigma_err_lower": sigma - sigma_lo,
        "sigma_err_upper": sigma_hi - sigma,
    }


def _render_ascii_table(headers, rows):
    widths = [len(header) for header in headers]
    rendered_rows = []
    for row in rows:
        rendered = [str(value) for value in row]
        rendered_rows.append(rendered)
        widths = [max(width, len(value)) for width, value in zip(widths, rendered)]

    def _line(values):
        return "| " + " | ".join(value.ljust(width) for value, width in zip(values, widths)) + " |"

    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [border, _line(headers), border]
    lines.extend(_line(row) for row in rendered_rows)
    lines.append(border)
    return "\n".join(lines)


def compare_models_by_log_evidence_all(
        df_agn,
        cosmo_models_dict,
        jeffreys_thresholds=(1.0, 2.5, 5.0),   # |Δln Z| bands
        z_decisive=2.0,
        write_path="plots/hubble/",
        *,
        sample_label="AGNs",
        sample_count=None,
        output_filename="compare_all_models.txt",
):
    """
    Compare MANY models by log-evidence.
    Inputs
    ------
    cosmo_models_dict : dict
        { model_label: {"logZ": float, "logZerr": float}, ... }

    Returns
    -------
    result : dict with:
      - 'ranking': list of per-model dicts sorted by logZ (desc)
      - 'preferred_model': label of top model
      - 'top_vs_runnerup': dict with ΔlnZ, Δerr, z_mc, odds-based Z (1- & 2-sided),
                           Jeffreys strength, Bayes-factor strings
      - 'pairwise': dict of dicts with pairwise deltas and Z’s
      - 'text_path': path to saved human-readable summary
    """
    # Collect and validate the model evidence inputs.
    items = []
    for label, vals in cosmo_models_dict.items():
        try:
            z = float(vals["logZ"])
            e = float(vals["logZerr"])
        except Exception as exc:
            raise ValueError(f"Model '{label}' missing numeric 'logZ'/'logZerr'") from exc
        items.append((label, z, e))
    if len(items) < 2:
        raise ValueError("Need at least two models to compare.")

    # Sort models by descending log evidence.
    items.sort(key=lambda t: t[1], reverse=True)
    top_label, top_logZ, top_err = items[0]
    preferred_model = top_label

    # Compute per-model comparisons relative to the top-ranked model.
    ranking = []
    for (label, z, e) in items:
        d = z - top_logZ   # <= 0 for all except top (0)
        de = float(np.hypot(e, top_err))
        z_mc = np.inf if de == 0 else d / de
        sigma_summary = _odds_sigma_summary_from_delta(d, de)
        log10K, B_str, B_ci = _bayes_factor_repr_from_delta(d, de)
        strength = _jeffreys_strength(abs(d), jeffreys_thresholds)
        ranking.append({
            "model": label,
            "logZ": z,
            "logZerr": e,
            "delta_logZ_vs_top": d,
            "delta_logZ_err_vs_top": de,
            "z_mc_vs_top": z_mc,
            "sigma_two_sided_vs_top": sigma_summary["sigma"],
            "sigma_two_sided_err_lower_vs_top": sigma_summary["sigma_err_lower"],
            "sigma_two_sided_err_upper_vs_top": sigma_summary["sigma_err_upper"],
            "sigma_two_sided_ci_1sigma_vs_top": (
                sigma_summary["sigma_lo"],
                sigma_summary["sigma_hi"],
            ),
            "jeffreys_strength_vs_top": strength,
            "log10_Bayes_factor_vs_top": log10K,
            "Bayes_factor_str_vs_top": B_str,
            "Bayes_factor_ci_1sigma_vs_top": B_ci,
        })

    # Summarize the top-vs-runner-up comparison.
    if len(items) >= 2:
        ru_label, ru_logZ, ru_err = items[1]
        delta = top_logZ - ru_logZ
        delta_err = float(np.hypot(top_err, ru_err))
        z_mc_head = np.inf if delta_err == 0 else delta / delta_err
        sigma_summary = _odds_sigma_summary_from_delta(delta, delta_err)
        log10K, B_str, B_ci = _bayes_factor_repr_from_delta(delta, delta_err)
        strength = _jeffreys_strength(abs(delta), jeffreys_thresholds)
        decisive = abs(z_mc_head) >= z_decisive

        top_vs_runnerup = {
            "preferred_model": top_label,
            "runner_up": ru_label,
            "delta_logZ": delta,
            "delta_logZ_err": delta_err,
            "z_mc": z_mc_head,
            "sigma_from_odds_one_sided": 0,
            "sigma_from_odds_two_sided": sigma_summary["sigma"],
            "sigma_from_odds_two_sided_err_lower": sigma_summary["sigma_err_lower"],
            "sigma_from_odds_two_sided_err_upper": sigma_summary["sigma_err_upper"],
            "sigma_from_odds_two_sided_ci_1sigma": (
                sigma_summary["sigma_lo"],
                sigma_summary["sigma_hi"],
            ),
            "log10_Bayes_factor": log10K,
            "Bayes_factor_str": B_str,
            "Bayes_factor_ci_1sigma": B_ci,
            "jeffreys_strength": strength,
            "decisive_zmc_ge_thresh": decisive,
        }
    else:
        top_vs_runnerup = None

    # Build the full pairwise comparison matrix.
    pairwise = {}
    for i, (li, zi, ei) in enumerate(items):
        pairwise[li] = {}
        for j, (lj, zj, ej) in enumerate(items):
            if i == j:
                pairwise[li][lj] = {
                    "delta_logZ": 0.0,
                    "delta_logZ_err": float(np.hypot(ei, ej)),
                    "z_mc": np.nan,
                    "sigma_one_sided": np.nan,
                    "sigma_two_sided": np.nan,
                    "sigma_two_sided_err_lower": np.nan,
                    "sigma_two_sided_err_upper": np.nan,
                    "sigma_two_sided_ci_1sigma": (np.nan, np.nan),
                    "jeffreys_strength": "—",
                    "log10_Bayes_factor": 0.0,
                    "Bayes_factor_str": "1:1",
                    "Bayes_factor_ci_1sigma": None,
                }
            else:
                d = zi - zj
                de = float(np.hypot(ei, ej))
                zmc = np.inf if de == 0 else d / de
                sigma_summary = _odds_sigma_summary_from_delta(d, de)
                log10K, B_str, B_ci = _bayes_factor_repr_from_delta(d, de)
                strength = _jeffreys_strength(abs(d), jeffreys_thresholds)
                pairwise[li][lj] = {
                    "delta_logZ": d,
                    "delta_logZ_err": de,
                    "z_mc": zmc,
                    "sigma_two_sided": sigma_summary["sigma"],
                    "sigma_two_sided_err_lower": sigma_summary["sigma_err_lower"],
                    "sigma_two_sided_err_upper": sigma_summary["sigma_err_upper"],
                    "sigma_two_sided_ci_1sigma": (
                        sigma_summary["sigma_lo"],
                        sigma_summary["sigma_hi"],
                    ),
                    "jeffreys_strength": strength,
                    "log10_Bayes_factor": log10K,
                    "Bayes_factor_str": B_str,
                    "Bayes_factor_ci_1sigma": B_ci,
                }

    # Build a human-readable text summary.
    lines = []
    sample_total = len(df_agn) if sample_count is None else sample_count
    lines.append("Bayesian Model Comparison (multi-model)\n")
    lines.append(f"Sample: {sample_label} (N={sample_total})\n")
    lines.append(
        "Jeffreys thresholds on |Δln Z|: "
        f"{jeffreys_thresholds[0]:.1f}, {jeffreys_thresholds[1]:.1f}, {jeffreys_thresholds[2]:.1f}\n"
    )
    lines.append(f"Preferred model: {preferred_model}\n\n")

    ranking_headers = (
        "rank",
        "best",
        "model",
        "ln Z ± err",
        "Δln Z(top) ± err",
        "sigma_Z",
        "sigma_Z -err/+err",
        "Jeffreys",
        "log10(K)",
    )
    ranking_rows = []
    for idx, r in enumerate(ranking, start=1):
        ranking_rows.append(
            (
                str(idx),
                "*" if r["model"] == preferred_model else "",
                r["model"],
                f"{r['logZ']:.3f} ± {r['logZerr']:.3f}",
                f"{r['delta_logZ_vs_top']:.3f} ± {r['delta_logZ_err_vs_top']:.3f}",
                f"{r['sigma_two_sided_vs_top']:.3f}",
                (
                    f"-{r['sigma_two_sided_err_lower_vs_top']:.3f}"
                    f"/+{r['sigma_two_sided_err_upper_vs_top']:.3f}"
                ),
                r["jeffreys_strength_vs_top"],
                f"{r['log10_Bayes_factor_vs_top']:.3f}",
            )
        )
    lines.append("Ranking by log evidence:\n")
    lines.append(_render_ascii_table(ranking_headers, ranking_rows) + "\n")

    if top_vs_runnerup is not None:
        t = top_vs_runnerup
        lines.append("\nTop vs runner-up:\n")
        lines.append(
            f"{t['preferred_model']} vs {t['runner_up']}: "
            f"Δln Z = {t['delta_logZ']:.3f} ± {t['delta_logZ_err']:.3f} "
            f"(z_mc = {t['z_mc']:.2f})\n"
            f"sigma_Z = {t['sigma_from_odds_two_sided']:.4f}"
            f" -{t['sigma_from_odds_two_sided_err_lower']:.4f}"
            f"/+{t['sigma_from_odds_two_sided_err_upper']:.4f} "
            f"(CI: [{t['sigma_from_odds_two_sided_ci_1sigma'][0]:.4f}, "
            f"{t['sigma_from_odds_two_sided_ci_1sigma'][1]:.4f}])\n"
            f"log10(K) = {t['log10_Bayes_factor']:.3f}; "
            f"Bayes factor = {t['Bayes_factor_str']}"
            + (
                f" {t['Bayes_factor_ci_1sigma'][2]}"
                if t["Bayes_factor_ci_1sigma"] is not None else ""
            )
            + "\n"
            f"Jeffreys strength: {t['jeffreys_strength']}; "
            f"decisive (|z_mc|≥{z_decisive:.1f})? {'yes' if t['decisive_zmc_ge_thresh'] else 'no'}\n"
        )

    if len(items) > 2:
        pairwise_headers = (
            "models",
            "Δln Z ± err",
            "sigma_Z -err/+err",
            "Jeffreys",
            "log10(K)",
        )
        pairwise_rows = []
        for a, b in combinations([label for label, _, _ in items], 2):
            pair = pairwise[a][b]
            pairwise_rows.append(
                (
                    f"{a} vs {b}",
                    f"{pair['delta_logZ']:.3f} ± {pair['delta_logZ_err']:.3f}",
                    (
                        f"{pair['sigma_two_sided']:.3f} "
                        f"-{pair['sigma_two_sided_err_lower']:.3f}"
                        f"/+{pair['sigma_two_sided_err_upper']:.3f}"
                    ),
                    pair["jeffreys_strength"],
                    f"{pair['log10_Bayes_factor']:.3f}",
                )
            )
        lines.append("\nPairwise comparisons:\n")
        lines.append(_render_ascii_table(pairwise_headers, pairwise_rows) + "\n")

    # Print and save the text summary.
    for line in lines:
        print(line, end="")
    os.makedirs(write_path, exist_ok=True)
    text_path = os.path.join(write_path, output_filename)
    with open(text_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return {
        "ranking": ranking,
        "preferred_model": preferred_model,
        "top_vs_runnerup": top_vs_runnerup,
        "pairwise": pairwise,
        "text_path": text_path,
    }



def write_hdf5_file(quasar_list, file_path):
    print(f"Writing {len(quasar_list)} quasars to {file_path}", flush=True)
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    with h5py.File(file_path, "w") as hdf:
        for quasar in quasar_list:
            object_id = quasar["object_id"]

            group = hdf.create_group(object_id)
            for key, value in quasar.items():
                if isinstance(value, dict):
                    sub_group = group.create_group(key)
                    for sub_key, sub_value in value.items():
                        sub_group.create_dataset(sub_key, data=sub_value)
                else:
                    group.attrs[key] = value

def extract_cosmo_results_from_samples(
    samples,
    cosmo_model,
    only_sna,
    logZ_tuple=None,
    *,
    format_for_latex=False,
    value_fmt="{:.3f}",
    use_alpha_lambda_term=None,
    use_eta_sigma_term=None,
    use_redshift_log_f_term=None,
):
    """
    Extract summary stats for all cosmological parameters from posterior samples.

    Parameters
    ----------
    samples : (N, P) ndarray
        Posterior samples. Columns must align with `model_labels` from get_model_params(cosmo_model).
    cosmo_model : str
        'FlatwCDM' or 'Flatw0waCDM'
    only_sna : bool
        True if SN Ia only; False if SN Ia + AGN.
    logZ_tuple : (float, float) or None
        (logZ, logZerr) from dynesty; None for emcee.
    format_for_latex : bool, optional
        If True, values are strings like r"$x \\pm y$". Otherwise tuples (mean, std).
    value_fmt : str, optional
        Format for numbers when format_for_latex=True (e.g., "{:.2f}").

    Returns
    -------
    dict
        {
          "model": "<latex model name>",
          "data": "<latex data label>",
          "params": { "<param_name>": (mean, std) or "$x \\pm y$", ... },
          "param_order": [ "<param_name>", ... ],              # for consistent table columns
          "param_labels_latex": { "<param_name>": "<latex>", ... },
          "logZ": { "value": float, "err": float } or None
        }
    """
    option_flags = resolve_model_option_flags(
        cosmo_model,
        np.asarray(samples).shape[1],
        only_sna=only_sna,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    priors, model_labels, model_labels_latex = get_model_params(
        cosmo_model,
        only_sna=only_sna,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
    )

    # Validate the sample array shape.
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError("`samples` must be a 2D array of shape (n_samples, n_params).")
    if samples.shape[1] != len(model_labels):
        raise ValueError(
            f"`samples` has {samples.shape[1]} columns but model expects {len(model_labels)} "
            f"({model_labels}). Ensure column order matches `model_labels`."
        )

    data_label = "SN~Ia" if only_sna else "SN~Ia + AGN"
    if cosmo_model == "Flatw0waCDM":
        model_name_latex = "flat $w_0w_a$CDM"
    elif cosmo_model == "FlatwCDM":
        model_name_latex = "flat $w$CDM"
    elif cosmo_model == "FlatLambdaCDM":
        model_name_latex = r"flat $\Lambda$CDM"
    else:
        model_name_latex = cosmo_model

    # Compute mean and standard deviation for every sampled parameter.
    means = np.mean(samples, axis=0)
    stds  = np.std(samples, axis=0, ddof=0)

    def pack(m, s):
        if format_for_latex:
            return rf"${value_fmt.format(m)} \pm {value_fmt.format(s)}$"
        return (m, s)

    params = {name: pack(means[i], stds[i]) for i, name in enumerate(model_labels)}
    param_labels_latex = {name: model_labels_latex[i] for i, name in enumerate(model_labels)}

    logZ_out = None
    if logZ_tuple is not None:
        logZ_val, logZ_err = logZ_tuple
        logZ_out = {
            "value": float(logZ_val),
            "err": float(logZ_err),
        }

    return {
        "model": model_name_latex,
        "data": data_label,
        "params": params,                       # all params present
        "param_order": list(model_labels),      # preserve column order for LaTeX
        "param_labels_latex": param_labels_latex,
        "logZ": logZ_out,
    }

def display_results_summary(
    samples,
    cosmo_model,
    z_pivot_agn,
    use_alpha_lambda_term=None,
    use_eta_sigma_term=None,
    use_redshift_log_f_term=None,
    sigma_sel_posterior_median=None,
):
    """
    Print median and 16/84% intervals for sampled params, plus derived w0 (and wa)
    when applicable. If cosmo_model == 'Flatw0waCDM', w0 is computed from (wp, wa)
    at the supplied z_pivot_agn.
    """
    samples = np.asarray(samples)
    if (
        use_alpha_lambda_term is None
        or use_eta_sigma_term is None
        or use_redshift_log_f_term is None
    ):
        option_flags = resolve_model_option_flags(
            cosmo_model,
            samples.shape[1],
            only_sna=False,
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
            use_redshift_log_f_term=use_redshift_log_f_term,
        )
        if use_alpha_lambda_term is None:
            use_alpha_lambda_term = option_flags["use_alpha_lambda_term"]
        if use_eta_sigma_term is None:
            use_eta_sigma_term = option_flags["use_eta_sigma_term"]
        if use_redshift_log_f_term is None:
            use_redshift_log_f_term = option_flags["use_redshift_log_f_term"]
    _, model_labels, _ = get_model_params(
        cosmo_model,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )

    def _format_interval(median, lo, hi, ndigits=3):
        return (
            f"{median:.{ndigits}f} "
            f"(+{hi - median:.{ndigits}f}, -{median - lo:.{ndigits}f})"
        )

    # Print the sampled parameter summaries first.
    med = np.median(samples, axis=0)
    lo  = np.percentile(samples, 16, axis=0)
    hi  = np.percentile(samples, 84, axis=0)
    for name, m, l, h in zip(model_labels, med, lo, hi):
        print(f"{name:>15}: {m:.4f} (+{h - m:.4f}, -{m - l:.4f})")

    # Resolve parameter names across model variants.
    def _idx(labels, *cands):
        for c in cands:
            if c in labels:
                return labels.index(c)
        return None  # not found

    # Print derived summaries for the relevant cosmology.
    if cosmo_model == "Flatw0waCDM":
        i_wp = _idx(model_labels, "wp", "w_p")
        i_wa = _idx(model_labels, "wa", "w_a")
        if i_wp is None or i_wa is None:
            return  # nothing to do

        a_p = 1.0 / (1.0 + float(z_pivot_agn))
        wp  = samples[:, i_wp]
        wa  = samples[:, i_wa]
        w0  = wp - (1.0 - a_p) * wa

        for name, arr in (("w0", w0), ("wa", wa)):
            m = np.median(arr)
            l = np.percentile(arr, 16)
            h = np.percentile(arr, 84)
            print(f"{name:>15}: {m:.4f} (+{h - m:.4f}, -{m - l:.4f})")

    else:
        i_w0 = _idx(model_labels, "w0", "w_0", "w")
        if i_w0 is not None:
            arr = samples[:, i_w0]
            m = np.median(arr); l = np.percentile(arr, 16); h = np.percentile(arr, 84)
            print(f"{'w0':>15}: {m:.4f} (+{h - m:.4f}, -{m - l:.4f})")

        i_wa = _idx(model_labels, "wa", "w_a")
        if i_wa is not None:
            arr = samples[:, i_wa]
            m = np.median(arr); l = np.percentile(arr, 16); h = np.percentile(arr, 84)
            print(f"{'wa':>15}: {m:.4f} (+{h - m:.4f}, -{m - l:.4f})")

    if "log_f" in model_labels:
        idx_logf = model_labels.index("log_f")
        log_f_samples = np.asarray(samples[:, idx_logf], dtype=float)
        if use_redshift_log_f_term and (AGN_LOGF_Z_PARAM := "gamma_log_f_z") in model_labels:
            idx_gamma_f = model_labels.index(AGN_LOGF_Z_PARAM)
            gamma_f_samples = np.asarray(samples[:, idx_gamma_f], dtype=float)
            log_f_samples = log_f_samples + gamma_f_samples * np.log10(
                (1.0 + float(z_pivot_agn)) / (1.0 + float(z_pivot_agn))
            )
        sigma_mag_samples = np.exp(log_f_samples)
        sigma_dex_samples = sigma_mag_samples / 2.5

        mag_lo, mag_med, mag_hi = np.percentile(sigma_mag_samples, [16, 50, 84])
        dex_lo, dex_med, dex_hi = np.percentile(sigma_dex_samples, [16, 50, 84])

        title = f"Intrinsic AGN scatter at z_pivot={float(z_pivot_agn):.2f}"
        mag_text = _format_interval(mag_med, mag_lo, mag_hi, ndigits=3) + " mag"
        dex_text = _format_interval(dex_med, dex_lo, dex_hi, ndigits=3) + " dex"
        sigma_sel_text = None
        delta_text = None
        if sigma_sel_posterior_median is not None:
            sigma_sel = np.asarray(sigma_sel_posterior_median, dtype=float)
            valid_sel = np.isfinite(sigma_sel) & (sigma_sel > 0.0)
            if np.any(valid_sel):
                sel_med = float(np.nanmedian(sigma_sel[valid_sel]))
                sigma_sel_text = f"{sel_med:.3f} mag"
                delta_text = f"{mag_med - sel_med:+.3f} mag"
        width = max(
            len(title),
            len(mag_text) + 15,
            len(dex_text) + 15,
            len(sigma_sel_text or "") + 20,
            len(delta_text or "") + 20,
            44,
        )
        border = "+" + "-" * (width + 2) + "+"
        print(border)
        print(f"| {title:<{width}} |")
        print(border)
        print(f"| {'sigma_int (mag)':<15}{mag_text:<{width - 15}} |")
        print(f"| {'sigma_int (dex)':<15}{dex_text:<{width - 15}} |")
        if sigma_sel_text is not None:
            print(f"| {'median sigma_sel':<20}{sigma_sel_text:<{width - 20}} |")
            print(f"| {'sigma_int - sel':<20}{delta_text:<{width - 20}} |")
        print(border)
    return

def _weighted_quantile(x, q, w=None):
    """
    Weighted quantiles of x at probabilities q in [0,1].
    If w is None, falls back to np.quantile.
    """
    x = np.asarray(x)
    q = np.atleast_1d(q)
    if w is None:
        return np.quantile(x, q)
    w = np.asarray(w)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        raise ValueError("No finite values for weighted quantile.")
    x, w = x[m], w[m]
    order = np.argsort(x)
    x, w = x[order], w[order]
    cum_w = np.cumsum(w)
    cum_w /= cum_w[-1]
    return np.interp(q, cum_w, x)

def compute_age_universe_with_error(
    samples,
    cosmo_model,
    weights=None,
    ci=(0.68, 0.95),
    max_eval=None,
    random_seed=None,
    use_alpha_lambda_term=None,
    use_eta_sigma_term=None,
    use_redshift_log_f_term=None,
):
    """
    Compute the posterior distribution of the Universe age and summarize it.

    Parameters
    ----------
    samples : (N, P) array
        Posterior samples aligned with `model_labels` returned by get_model_params(cosmo_model).
    cosmo_model : str
        One of {"Flatw0waCDM","FlatwCDM","FlatLambdaCDM","FlatwpwaCDM"}.
    weights : (N,) array or None
        Optional sample weights (e.g., from nested sampling). If None, treats samples equally.
    ci : tuple
        Credible levels to report as central intervals, e.g., (0.68, 0.95).
    max_eval : int or None
        If set, randomly subsample to at most this many evaluations for speed.
    random_seed : int or None
        RNG seed for reproducible subsampling.

    Returns
    -------
    stats : dict
        {
          "mean": float, "std": float,
          "q16": float, "q50": float, "q84": float,
          "q2.5": float, "q97.5": float,
          "ages": np.ndarray (valid posterior ages in Gyr),
          "n_total": int, "n_valid": int, "n_invalid": int
        }
    """
    # Get parameter names & any needed priors (e.g., zp for FlatwpwaCDM)
    option_flags = resolve_model_option_flags(
        cosmo_model,
        np.asarray(samples).shape[1],
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    priors, model_labels, _ = get_model_params(
        cosmo_model,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
    )

    N = samples.shape[0]
    idx = np.arange(N)

    # Optionally subsample the posterior for speed.
    if (max_eval is not None) and (N > max_eval):
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(idx, size=max_eval, replace=False)
        samples = samples[idx]
        if weights is not None:
            weights = np.asarray(weights)[idx]

    ages = np.full(samples.shape[0], np.nan, dtype=float)

    # Evaluate the cosmological age sample by sample.
    for j in range(samples.shape[0]):
        params = {name: samples[j, i] for i, name in enumerate(model_labels)}
        try:
            if cosmo_model == "Flatw0waCDM":
                cosmo = Flatw0waCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'], wa=params['wa'])
            elif cosmo_model == "FlatwCDM":
                cosmo = FlatwCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'])
            elif cosmo_model == "FlatLambdaCDM":
                cosmo = FlatLambdaCDM(H0=params['H0'], Om0=params['Om0'])
            elif cosmo_model == "FlatwpwaCDM":
                cosmo = FlatwpwaCDM(H0=params['H0'], Om0=params['Om0'],
                                     wp=params['wp'], wa=params['wa'],
                                     zp=priors.get('zp', 0.0))
            else:
                raise ValueError(f"Unknown cosmology model: {cosmo_model}")

            age = cosmo.age(0).to(u.Gyr).value
            if np.isfinite(age) and (age > 0):
                ages[j] = age
        except Exception:
            continue

    m = np.isfinite(ages)
    n_valid = int(np.sum(m))
    n_total = ages.size
    n_invalid = n_total - n_valid
    if n_valid == 0:
        raise RuntimeError("All age evaluations failed; check parameter ranges.")

    a = ages[m]
    w = None if weights is None else np.asarray(weights)[m]

    # Summarize the valid age samples.
    mean = np.average(a, weights=w) if w is not None else float(np.mean(a))
    if w is not None:
        w_norm = w / np.sum(w)
        var = np.average((a - mean) ** 2, weights=w_norm)
        std = float(np.sqrt(var))
    else:
        std = float(np.std(a, ddof=1))

    q16, q50, q84 = _weighted_quantile(a, [0.16, 0.50, 0.84], w)
    q025, q975 = _weighted_quantile(a, [0.025, 0.975], w)

    print(f"Age of universe: {q50:.3f} (+{q84 - q50:.3f}/-{q50 - q16:.3f}) Gyr  "
          f"[mean={mean:.3f}±{std:.3f} Gyr; valid {n_valid}/{n_total}, skipped {n_invalid}]")

    return mean, std

def compute_pivot_redshift(flat_samples, cosmo_model, z_min=0.0, z_max=4.0):
    """
    Compute the optimal pivot redshift z_p* from MCMC samples of (wp, wa),
    with a constraint z_p* >= z_min (default 0).

    Returns
    -------
    z_p_star : float
        Constrained optimal pivot redshift (decorrelates w_p and w_a as much as
        possible under the z >= z_min constraint).
    """
    option_flags = resolve_model_option_flags(
        cosmo_model, np.asarray(flat_samples).shape[1]
    )
    priors, model_labels, model_labels_latex = get_model_params(
        cosmo_model,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
    )
    idx = {name: model_labels.index(name) for name in model_labels}

    # Samples
    wp = flat_samples[:, idx['wp']]
    wa = flat_samples[:, idx['wa']]

    # Read the current pivot from the priors if present.
    z_p_current = None
    for k in ('zp', 'z_p', 'z_pivot_agn', 'z_pivot'):
        if k in priors:
            z_p_current = priors[k] if np.isscalar(priors[k]) else float(priors[k])
            break
    if z_p_current is None:
        z_p_current = 0.0

    a_p = 1.0 / (1.0 + z_p_current)

    # Compute the covariance structure of the pivot parameters.
    C = np.cov(np.vstack([wp, wa]))
    cov_wp_wa = C[0, 1]
    var_wa    = C[1, 1]

    # Compute the unconstrained optimal pivot in scale factor.
    tiny = np.finfo(float).tiny
    a_p_star = a_p + cov_wp_wa / max(var_wa, tiny)

    # Enforce the requested redshift bounds.
    a_max_allowed = 1.0 / (1.0 + z_min)  # for z_min=0, this is 1
    a_min_allowed = 1.0 / (1.0 + (z_max if z_max is not None else np.inf))

    if z_max is not None:
        a_p_star = max(a_p_star, a_min_allowed)
    a_p_star = min(a_p_star, a_max_allowed)

    # Guard against non-physical scale factors.
    a_p_star = max(a_p_star, np.finfo(float).eps)
    z_p_star = 1.0 / a_p_star - 1.0
    return z_p_star

def posterior_corr(flat_samples, cosmo_model, z_pivot_agn):
    """
    Pearson correlation between posterior w0 and wa.

    If cosmo_model is 'Flatw0waCDM', w0 is derived from (wp, wa) at the given pivot redshift.
    Otherwise, tries to read w0 directly from samples.

    Parameters
    ----------
    flat_samples : (N, P) array
        Flattened MCMC samples: N total draws by P parameters.
    cosmo_model : str
        Cosmological model string.
    z_pivot_agn : float, optional
        Pivot redshift to compute w0 from wp, wa for Flatw0waCDM.

    Returns
    -------
    rho : float
        Posterior Pearson correlation coefficient corr(w0, wa).
    """
    option_flags = resolve_model_option_flags(
        cosmo_model, np.asarray(flat_samples).shape[1]
    )
    priors, model_labels, _ = get_model_params(
        cosmo_model,
        use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
        use_eta_sigma_term=option_flags["use_eta_sigma_term"],
        use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
    )
    flat_samples = np.asarray(flat_samples)

    def _idx(labels, *names):
        for name in names:
            if name in labels:
                return labels.index(name)
        raise ValueError(f"Parameter {names} not found in model labels.")

    if cosmo_model == "Flatw0waCDM":
        i_w0 = _idx(model_labels, "w0", "w_0")
        i_wa = _idx(model_labels, "wa", "w_a")
        w0 = flat_samples[:, i_w0]
        wa = flat_samples[:, i_wa]
    elif cosmo_model == "FlatwpwaCDM":
        i_wp = _idx(model_labels, "wp", "w_p")
        i_wa = _idx(model_labels, "wa", "w_a")
        a_p = 1.0 / (1.0 + float(z_pivot_agn))
        wp = flat_samples[:, i_wp]
        wa = flat_samples[:, i_wa]
        w0 = wp - (1.0 - a_p) * wa
    else:
        i_w0 = _idx(model_labels, "w0", "w_0", "w")
        i_wa = _idx(model_labels, "wa", "w_a")
        w0 = flat_samples[:, i_w0]
        wa = flat_samples[:, i_wa]

    mask = np.isfinite(w0) & np.isfinite(wa)
    if not np.any(mask):
        raise ValueError("No finite samples to compute correlation.")

    return np.corrcoef(w0[mask], wa[mask])[0, 1]

import math


def _to_finite_float(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def format_result_errors(
    val,
    err=None,
    err_lower=None,
    err_upper=None,
    unit=None,
    nd=2,
    sci_threshold=3,
):
    """
    Return a LaTeX-ready string with fixed decimal places.

    Examples
    --------
    12.35 \\pm 0.46
    (1.23 \\pm 0.05) \\times 10^{5}
    0.84^{+0.07}_{-0.05}
    (3.21^{+0.11}_{-0.09}) \\times 10^{-4}

    Notes
    -----
    - If either err_lower or err_upper is provided, asymmetric errors are used.
    - `unit` should already be LaTeX-ready, e.g. r"\\Msun" or r"\\mathrm{km\\,s^{-1}}".
    """

    valf = _to_finite_float(val)
    if valf is None:
        return "N/A"

    errf = _to_finite_float(err)
    errlf = _to_finite_float(err_lower)
    erruf = _to_finite_float(err_upper)

    # Use asymmetric mode if either asymmetric error is provided
    use_asym = (errlf is not None) or (erruf is not None)

    mags = [abs(valf)]
    if errf is not None:
        mags.append(abs(errf))
    if errlf is not None:
        mags.append(abs(errlf))
    if erruf is not None:
        mags.append(abs(erruf))

    max_mag = max(mags) if mags else 0.0
    exponent = int(math.floor(math.log10(max_mag))) if max_mag > 0 else 0

    use_sci = abs(exponent) >= sci_threshold
    scale = 10.0 ** exponent if use_sci else 1.0

    v = valf / scale
    fmt = f"{{:.{nd}f}}"

    if use_asym:
        el = abs(errlf) / scale if errlf is not None else 0.0
        eu = abs(erruf) / scale if erruf is not None else 0.0
        core = rf"{fmt.format(v)}^{{+{fmt.format(eu)}}}_{{-{fmt.format(el)}}}"
    elif errf is not None:
        e = abs(errf) / scale
        core = rf"{fmt.format(v)} \pm {fmt.format(e)}"
    else:
        core = fmt.format(v)

    if use_sci:
        out = rf"({core}) \times 10^{{{exponent}}}"
    else:
        out = core

    if unit:
        out += rf"\,\mathrm{{{unit}}}"

    return out

def sigma_distance_asym(x1, err1, x2, err2_lower, err2_upper):
    if err1 < 0 or err2_lower < 0 or err2_upper < 0:
        raise ValueError("Uncertainties must be non-negative.")

    if x1 < x2:
        err2 = err2_lower
    else:
        err2 = err2_upper

    denom = math.sqrt(err1**2 + err2**2)
    if denom == 0:
        raise ValueError("Combined uncertainty is zero.")

    return abs(x1 - x2) / denom


def write_results_tex_variables(
    df_agn,
    df_agn_all,
    df_pantheon,
    z_range,
    cosmo_model_joint_samples,
    cosmo_model_sna_samples,
    compare_r,
    write_path,
    result_prefix="",
    chisq_dict=None,
    cosmo_models_result_dict=None,
    cosmo_models_sna_result_dict=None,
    compare_r_sna=None,
):
    """
    Write key cosmological parameters AND model comparison results
    to a LaTeX file.
    """

    # Capitalize the first letter of result_prefix if present
    if result_prefix:
        result_prefix = result_prefix[0].upper() + result_prefix[1:]

    lines = []
    lines.append(r"% Auto-generated cosmological and evidence results")
    lines.append(r"% Do not edit manually; regenerated by write_results_tex_variables()")

    # Local formatting helpers.
    def _clean(name):
        return name.replace("0", "Zero").replace("Λ", "Lambda").replace("_", "")

    def _cmd(name, content, model_suffix=""):
        cmd_name = f"result{result_prefix}{_clean(model_suffix)}{name}"
        return f"\\newcommand{{\\{cmd_name}}}{{\\ensuremath{{{content}}}}}"

    def _sym_percentile(data, percentiles=[16, 50, 84]):
        if len(data) == 0: return np.nan, np.nan
        p = np.percentile(data, percentiles)
        return p[1], (p[2] - p[0]) / 2.0

    # Global AGN and SN summary counts.
    obs_arr, err_arr, pivots_arr = agn_model_pack_obs(df_agn)
    log_sigma_uv_pivot = pivots_arr[agn_model_oidx["log_sigma_uv"]]
    log_tau_uv_rf_pivot = pivots_arr[agn_model_oidx["log_tau_uv_rf"]]
    n_fitted = len(df_agn[df_agn['z'].between(z_range[0], z_range[1])])
    lines.append(_cmd("NumAGNInitial", len(df_agn_all)))
    lines.append(_cmd("NumAGNCut", len(df_agn_all)-len(df_agn)))

    lines.append(_cmd("NumAGNPlotted", len(df_agn)))
    lines.append(_cmd("NumAGNFitted", n_fitted))

    is_calib_bool = np.asarray(df_pantheon['IS_CALIBRATOR'], dtype=bool)
    mask = (df_pantheon['zHD'] > 0.01) | is_calib_bool
    lines.append(_cmd("NumSNaPlotted", len(df_pantheon)))
    lines.append(_cmd("NumSNaFitted", len(df_pantheon[mask])))
    lines.append(_cmd("SigmauvPivot", f"{10**log_sigma_uv_pivot:.1f}"))
    lines.append(_cmd("TauuvrfPivot", f"{10**log_tau_uv_rf_pivot:.0f}"))

    for model_name, flat_samples in cosmo_model_sna_samples.items():
        flat_samples = np.asarray(flat_samples)
        option_flags = resolve_model_option_flags(
            model_name, flat_samples.shape[1], only_sna=True
        )
        priors, model_labels, _ = get_model_params(
            model_name,
            only_sna=True,
            use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
            use_eta_sigma_term=option_flags["use_eta_sigma_term"],
            use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
        )
        results = {}
        for i, key in enumerate(model_labels):
            median, err, err_lower, err_upper = sym_percentile(flat_samples[:, i])
            results[key] = median
            results[f"{key}_err"] = err
            results[f"{key}_err_lower"] = err_lower
            results[f"{key}_err_upper"] = err_upper

        if 'M0_sn' in results:
            lines.append(_cmd("SNOnlyMZero", format_result_errors(results['M0_sn'], results['M0_sn_err']), model_suffix=model_name))
        if 'Om0' in results:
            formatted = format_result_errors(results['Om0'], results['Om0_err'])
            lines.append(_cmd("SNOnlyOmZero", formatted, model_suffix=model_name))
        if 'w0' in results:
            formatted = format_result_errors(results['w0'], err_lower=results['w0_err_lower'], err_upper=results['w0_err_upper'])
            lines.append(_cmd("SNOnlyWZero", formatted, model_suffix=model_name))
        if 'wa' in results:
            formatted = format_result_errors(results['wa'], err_lower=results['wa_err_lower'], err_upper=results['wa_err_upper'], nd=1)
            lines.append(_cmd("SNOnlyWa", formatted, model_suffix=model_name))
        if 'H0' in results:
            formatted = format_result_errors(results['H0'], results['H0_err'])
            lines.append(_cmd("SNOnlyHZero", formatted, model_suffix=model_name))

        sna_result = (cosmo_models_sna_result_dict or {}).get(model_name)
        if sna_result is not None:
            lines.append(
                _cmd(
                    "SNOnlyAgeUniverse",
                    format_result_errors(sna_result["age"], sna_result["age_err"], unit=r"Gyr"),
                    model_suffix=model_name,
                )
            )


    # Per-model summary parameters.
    for model_name, flat_samples in cosmo_model_joint_samples.items():
        flat_samples = np.asarray(flat_samples)
        option_flags = resolve_model_option_flags(
            model_name, flat_samples.shape[1]
        )
        priors, model_labels, _ = get_model_params(
            model_name,
            use_alpha_lambda_term=option_flags["use_alpha_lambda_term"],
            use_eta_sigma_term=option_flags["use_eta_sigma_term"],
            use_redshift_log_f_term=option_flags["use_redshift_log_f_term"],
        )
        results = {}
        for i, key in enumerate(model_labels):
            median, err, err_lower, err_upper = sym_percentile(flat_samples[:, i])
            results[key] = median
            results[f"{key}_err"] = err
            results[f"{key}_err_lower"] = err_lower
            results[f"{key}_err_upper"] = err_upper
        sigma_distance = sigma_distance_asym(results['Om0'], results['Om0_err'],  0.495, 0.043, 0.033)
        if 'Om0' in results:
            lines.append(_cmd("OmZero", format_result_errors(results['Om0'],results['Om0_err']), model_suffix=model_name))
            lines.append(_cmd("OmZeroSigmaDistanceDES", format_result_errors(sigma_distance, nd=1), model_suffix=model_name))
        if 'w0' in results:
            lines.append(_cmd("wZero", 
                            format_result_errors(results['w0'],err_lower=results['w0_err_lower'], err_upper=results['w0_err_upper']),
                            model_suffix=model_name))
        if 'wa' in results:
            lines.append(_cmd("wa", 
                              format_result_errors(results['wa'],err_lower=results['wa_err_lower'], err_upper=results['wa_err_upper'], nd=1),
                              model_suffix=model_name))
        if 'H0' in results:
             lines.append(_cmd("HZero", format_result_errors(results['H0'],results['H0_err']), model_suffix=model_name))
        if 'alpha_agn' in results:
            lines.append(_cmd("AlphaAGN", format_result_errors(results['alpha_agn'],results['alpha_agn_err']), model_suffix=model_name))
        if 'beta_agn' in results:
            lines.append(_cmd("BetaAGN", format_result_errors(results['beta_agn'],results['beta_agn_err']), model_suffix=model_name))
        if 'M0_agn' in results:
            lines.append(_cmd("MZeroAGN", format_result_errors(results['M0_agn'],results['M0_agn_err']), model_suffix=model_name))

        result = cosmo_models_result_dict[model_name]
        lines.append(_cmd("AgeUniverse", format_result_errors(result["age"], result["age_err"], unit=r"Gyr"), model_suffix=model_name))


        try:
            idx_M0 = model_labels.index('M0_agn')
            idx_alpha = model_labels.index('alpha_agn')
            idx_beta = model_labels.index('beta_agn')
            idx_logf = model_labels.index('log_f')

            L_intercept = np.power(10, (90 - flat_samples[:, idx_M0]) / 2.5)
            alpha_L = flat_samples[:, idx_alpha] * (-1/2.5)
            beta_L = flat_samples[:, idx_beta] * (-1/2.5)
            hd_scatter = np.exp(flat_samples[:, idx_logf])
            l_scatter = hd_scatter / 2.5

            lines.append(_cmd("LIntercept", format_result_errors(*_sym_percentile(L_intercept), unit=r"erg\,s^{-1}"), model_suffix=model_name))
            lines.append(_cmd("AlphaAGNL", format_result_errors(*_sym_percentile(alpha_L)), model_suffix=model_name))
            lines.append(_cmd("BetaAGNL", format_result_errors(*_sym_percentile(beta_L)), model_suffix=model_name))
            lines.append(_cmd("ScatterHD", format_result_errors(*_sym_percentile(hd_scatter), unit=r"mag"), model_suffix=model_name))
            lines.append(_cmd("ScatterL", format_result_errors(*_sym_percentile(l_scatter), unit=r"dex"), model_suffix=model_name))
        except ValueError:
            pass

        if chisq_dict and model_name in chisq_dict:
            lines.append(_cmd("ChiSqRed", format_result_errors(chisq_dict[model_name]), model_suffix=model_name))

    # Model-comparison summaries.
    if compare_r:
        lines.append(r"% --- Model Comparisons ---")
        if "preferred_model" in compare_r:
            lines.append(_cmd("PreferredModelOverall", compare_r["preferred_model"]))

        for r in compare_r.get("ranking", []):
            m_name = r["model"]
            lines.append(_cmd("LogZ", f"{r['logZ']:.1f}", model_suffix=m_name))
            lines.append(_cmd("LogZerr", f"{r['logZerr']:.1f}", model_suffix=m_name))
            lines.append(_cmd("DeltaLogZ", f"{r['delta_logZ_vs_top']:.1f}", model_suffix=m_name))
            lines.append(_cmd("Sigma", f"{r['sigma_two_sided_vs_top']:.1f}", model_suffix=m_name))
            lines.append(_cmd("JeffreysStrength", r['jeffreys_strength_vs_top'], model_suffix=m_name))

        pw = compare_r.get("pairwise", {})
        ranked_models = [r["model"] for r in compare_r.get("ranking", [])]
        
        for a, b in combinations(ranked_models, 2):
            pair = pw.get(a, {}).get(b) or pw.get(b, {}).get(a)
            pair_name = f"{_clean(a)}{_clean(b)}" 

            if pair:
                lines.append(_cmd(f"DeltaLogZ{pair_name}", f"{pair['delta_logZ']:.1f} \\pm {pair['delta_logZ_err']:.1f}"))
                lines.append(_cmd(f"Sigma{pair_name}", f"{pair['sigma_two_sided']:.1f}"))
                lines.append(_cmd(f"JeffreysStrength{pair_name}", pair['jeffreys_strength']))
            else:
                lines.append(_cmd(f"DeltaLogZ{pair_name}", "N/A"))

    if compare_r_sna:
        lines.append(r"% --- SN-only Model Comparisons ---")
        if "preferred_model" in compare_r_sna:
            lines.append(_cmd("SNOnlyPreferredModelOverall", compare_r_sna["preferred_model"]))

        for r in compare_r_sna.get("ranking", []):
            m_name = r["model"]
            lines.append(_cmd("SNOnlyLogZ", f"{r['logZ']:.1f}", model_suffix=m_name))
            lines.append(_cmd("SNOnlyLogZerr", f"{r['logZerr']:.1f}", model_suffix=m_name))
            lines.append(_cmd("SNOnlyDeltaLogZ", f"{r['delta_logZ_vs_top']:.1f}", model_suffix=m_name))
            lines.append(_cmd("SNOnlySigma", f"{r['sigma_two_sided_vs_top']:.1f}", model_suffix=m_name))
            lines.append(_cmd("SNOnlyJeffreysStrength", r['jeffreys_strength_vs_top'], model_suffix=m_name))

        pw = compare_r_sna.get("pairwise", {})
        ranked_models = [r["model"] for r in compare_r_sna.get("ranking", [])]

        for a, b in combinations(ranked_models, 2):
            pair = pw.get(a, {}).get(b) or pw.get(b, {}).get(a)
            pair_name = f"{_clean(a)}{_clean(b)}"

            if pair:
                lines.append(_cmd(f"SNOnlyDeltaLogZ{pair_name}", f"{pair['delta_logZ']:.1f} \\pm {pair['delta_logZ_err']:.1f}"))
                lines.append(_cmd(f"SNOnlySigma{pair_name}", f"{pair['sigma_two_sided']:.1f}"))
                lines.append(_cmd(f"SNOnlyJeffreysStrength{pair_name}", pair['jeffreys_strength']))
            else:
                lines.append(_cmd(f"SNOnlyDeltaLogZ{pair_name}", "N/A"))

    # Write the LaTeX command file.
    filename = f"param_results_{result_prefix.lower()}.tex" if result_prefix else "param_results.tex"
    tex_path = os.path.join(write_path, filename)
    os.makedirs(write_path, exist_ok=True)
    
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")
    
    print(f"Wrote result parameters LaTeX commands to {tex_path}")

def reduced_chi_squared(residuals,
                        model_err,
                        extra_err=None,
                        n_params=0,
                        min_err=1e-12):
    """
    Compute reduced chi^2 from residuals and per-point uncertainties.

    Parameters
    ----------
    residuals : array-like
        Data minus model, in the SAME units as the uncertainties
        (here: log10 L_2500).
    model_err : array-like
        1σ model spread at each data point (e.g., 0.5*(hi-lo) of the ribbon),
        in log10 units.
    extra_err : None or array-like, optional
        Optional additional 1σ errors per point to add in quadrature
        (e.g., measurement error in log10 L, propagated x-error in log space).
    n_params : int, optional
        Number of fitted parameters used to define the model (for DoF).
    min_err : float, optional
        Floor to avoid division by ~0.

    Returns
    -------
    chi2_red : float
        Reduced chi-squared.
    meta : dict
        {'chi2': float, 'dof': int, 'N_eff': int, 'n_params': int}
    """
    r = np.asarray(residuals, dtype=float)
    s = np.asarray(model_err,  dtype=float)

    if extra_err is not None:
        s = np.sqrt(s**2 + np.asarray(extra_err, dtype=float)**2)

    # Guard against non-finite/zero sigmas
    finite = np.isfinite(r) & np.isfinite(s) & (s > 0)
    r = r[finite]
    s = np.maximum(s[finite], min_err)

    N = r.size
    dof = max(1, N - int(n_params))

    chi2 = np.sum((r / s)**2)
    chi2_red = chi2 / dof

    return chi2_red, {'chi2': chi2, 'dof': dof, 'N_eff': N, 'n_params': int(n_params)}

def cosmo_model_label_latex(cosmo_model):
    if   cosmo_model == 'FlatwCDM':      label = r"flat $w$CDM model"
    elif cosmo_model == 'Flatw0waCDM':   label = r"flat $w_0w_a$CDM model"
    elif cosmo_model == 'FlatLambdaCDM': label = r"flat $\Lambda$CDM model"
    elif cosmo_model == 'FlatwpwaCDM':   label = r"flat $w_p\!-\!w_a$CDM model"
    else:
        raise ValueError("Invalid cosmology model.")
    
    return label


def _save_mapping_hdf5(filename, **kwargs):
    """Save keyword arrays/scalars into a flat HDF5 file."""
    with h5py.File(filename, 'w') as f:
        for name, data in kwargs.items():
            if data is None:
                data = np.nan
            if isinstance(data, str):
                f.create_dataset(name, data=data, dtype=h5py.string_dtype(encoding="utf-8"))
                continue
            if isinstance(data, (np.ndarray, list, tuple)):
                arr = np.asarray(data)
                if arr.dtype.kind in {"U", "O"}:
                    arr = arr.astype(h5py.string_dtype(encoding="utf-8"))
                    f.create_dataset(name, data=arr)
                else:
                    f.create_dataset(name, data=arr, compression="gzip")
            else:
                f.create_dataset(name, data=data)
    print(f"Saved: {list(kwargs.keys())} to {filename}")


def _load_mapping_hdf5(filename):
    """Load a flat HDF5 file into a dictionary."""
    results = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            value = f[key][()]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            elif isinstance(value, np.ndarray) and value.dtype.kind == "S":
                value = value.astype(str)
            results[key] = value
    return results


def save_chains(filename, **kwargs):
    """Backward-compatible wrapper for saving posterior chains to HDF5."""
    _save_mapping_hdf5(filename, **kwargs)


def load_chains(filename):
    """Backward-compatible wrapper for loading posterior chains from HDF5."""
    return _load_mapping_hdf5(filename)

def save_cosmo_results_hdf5(filename, models_dict):
    """
    Saves a dictionary of cosmological models to HDF5.
    Input structure: {'ModelName': {'param': val, 'param_err': val, ...}}
    """
    with h5py.File(filename, 'w') as f:
        for model_name, params in models_dict.items():
            # Create a dedicated Group for each model (e.g., 'FlatLambdaCDM')
            grp = f.create_group(model_name)
            
            for param_name, value in params.items():
                if value is None:
                    value = np.nan  # HDF5 doesn't support None, use NaN for missing values
                grp.create_dataset(param_name, data=value)
    
    print(f"Saved models: {list(models_dict.keys())} to {filename}")

def select_agn_subset_uniform_with_replacement(
    df_agn,
    z_range,
    N=None,
    subset_seed=42,
    id_col="object_id",
    z_uniform_min=0.44,
    kde_bw_method=None,
    max_weight_ratio=20.0,
):
    """
    Deterministic weighted sampling WITH replacement, using continuous
    inverse-density weights in redshift so the selected sample is
    approximately uniform in z between z_uniform_min and z_range[1].

    Same inputs -> same returned sample.
    Duplicates are allowed.
    """
    zmin, zmax = z_range
    z0 = max(zmin, z_uniform_min)

    df_sel = df_agn.copy()
    df_sel = df_sel[df_sel["z"].between(z0, zmax)].copy()

    n_avail = len(df_sel)
    print(f"AGN available after cuts: {n_avail}")

    if n_avail == 0:
        raise ValueError("No AGN available after z_range cut.")

    if N is None:
        df_sel = df_sel.sort_values([id_col, "z"]).reset_index(drop=True)
        print("Using all AGN in range.")
        print(f"+++ Length of selected AGNs: {len(df_sel)}")
        print(f"+++ Redshift range of selected AGNs: {df_sel['z'].min()} to {df_sel['z'].max()}")
        return df_sel

    z = df_sel["z"].to_numpy()

    # estimate parent density p(z)
    kde = gaussian_kde(z, bw_method=kde_bw_method)
    pz = kde(z)

    # target uniform in z => w(z) ∝ 1 / p(z)
    w = 1.0 / np.clip(pz, 1e-12, None)

    # cap extreme weights for stability
    w_med = np.median(w)
    w = np.minimum(w, max_weight_ratio * w_med)

    # normalize after sorting so results do not depend on input row order
    df_sel = df_sel.sort_values([id_col, "z"]).reset_index(drop=True)
    z = df_sel["z"].to_numpy()
    pz = kde(z)
    w = 1.0 / np.clip(pz, 1e-12, None)
    w_med = np.median(w)
    w = np.minimum(w, max_weight_ratio * w_med)
    probs = w / w.sum()

    rng = np.random.default_rng(subset_seed)
    idx = rng.choice(len(df_sel), size=N, replace=True, p=probs)

    df_out = df_sel.iloc[idx].reset_index(drop=True)

    print(f"Selected N={len(df_out)} AGN with subset_seed={subset_seed} (with replacement)")
    print(f"+++ Redshift range of selected AGNs: {df_out['z'].min()} to {df_out['z'].max()}")

    n_unique = df_out[id_col].nunique()
    print(f"+++ Unique AGNs in selected sample: {n_unique}/{len(df_out)}")

    return df_out



def report_pivots(df_agn):
    print("\nAGN pivot summary")
    _, _, pivots_arr = agn_model_pack_obs(df_agn)
    rows = [
        ("sigma_uv", "median", np.median(df_agn["log_sigma_uv"])),
        ("sigma_uv", "model pivot", pivots_arr[agn_model_oidx["log_sigma_uv"]]),
        ("tau_uv_rf", "median", np.median(df_agn["log_tau_uv_rf"])),
        ("tau_uv_rf", "model pivot", pivots_arr[agn_model_oidx["log_tau_uv_rf"]]),
    ]

    for name, kind, log_val in rows:
        lin_val = 10**log_val
        print(f"{name} {kind}: log10={log_val}, linear={lin_val}")
