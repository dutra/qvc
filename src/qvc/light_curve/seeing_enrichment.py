"""Bulk-fetch per-epoch image quality for QVC Stripe 82 light curves.

Remote responses are cached before compatible SDSS, PS1, and ZTF seeing
sidecars are assembled. Interrupted runs can therefore resume safely.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

from qvc.hubble.hubble_utils import resolve_qvc_data_path


SDSS_BANDS = ("u", "g", "r", "i", "z")
SURVEYS = ("sdss", "ps1", "ztf")
S82_DIR = "data/S82"
PS1_TAP_SYNC_URL = "https://mast.stsci.edu/vo-tap/api/v0.1/ps1dr2/sync"
ZTF_IBE_URL = "https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci"
# The dedicated DR7 Stripe 82 context contains all 303 repeat scans used by
# the light-motion curves. The default DR17 Field table omits many of them.
SDSS_SQL_URL = "https://skyserver.sdss.org/stripe82/en/tools/search/x_sql.asp"
SIDECAR_FILENAMES = {
    "sdss": "dr16s82_sdssSeeing.parquet",
    "ps1": "dr16s82_ps1Seeing.parquet",
    "ztf": "dr16s82_ztfSeeing.parquet",
}
SIDECAR_KEYS = {
    "sdss": ["objectId", "mjd", "filterID"],
    "ps1": ["detectID"],
    "ztf": ["ps1objID", "mjd", "fieldid", "rcidin", "filterID"],
}


def _read_csv_url(
    base_url, params, *, comment=None, method="GET", timeout=180, retries=4
):
    """Read a CSV endpoint with bounded exponential-backoff retries."""

    encoded = urllib.parse.urlencode(params).encode()
    if method.upper() == "POST":
        request = urllib.request.Request(base_url, data=encoded, method="POST")
        request.add_header("Content-Type", "application/x-www-form-urlencoded")
    else:
        request = urllib.request.Request(f"{base_url}?{encoded.decode()}")
    request.add_header("User-Agent", "qvc-seeing/2")
    last_error = None
    for attempt in range(int(retries) + 1):
        try:
            with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                text = response.read().decode("utf-8")
            if not text.strip():
                return pd.DataFrame()
            if text.lstrip().startswith("{"):
                payload = json.loads(text)
                if "ErrorMessage" in payload or "error" in payload:
                    raise RuntimeError(
                        payload.get("ErrorMessage", payload.get("error", text))
                    )
            lines = text.splitlines()
            if len(lines) > 1 and lines[1].strip().upper() == "ERROR":
                raise RuntimeError("\n".join(lines[2:8]).strip())
            return pd.read_csv(StringIO(text), comment=comment)
        except (
            TimeoutError,
            urllib.error.HTTPError,
            urllib.error.URLError,
            pd.errors.ParserError,
        ) as error:
            last_error = error
            if attempt >= int(retries):
                break
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Archive query failed after {retries + 1} attempts") from last_error


def _require_columns(table, columns, source):
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{source} response is missing columns {missing}.")


def _normalise_ps1_response(table):
    columns = ["detectID", "ps1objID", "psf_fwhm_arcsec"]
    if table.empty:
        return pd.DataFrame(columns=columns)
    names = {str(column).lower(): column for column in table.columns}
    required = ("detectid", "objid", "psfmajorfwhm", "psfminorfwhm")
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"PS1 TAP response is missing columns {missing}.")
    major = pd.to_numeric(table[names["psfmajorfwhm"]], errors="coerce")
    minor = pd.to_numeric(table[names["psfminorfwhm"]], errors="coerce")
    return pd.DataFrame(
        {
            "detectID": pd.to_numeric(table[names["detectid"]], errors="raise"),
            "ps1objID": pd.to_numeric(table[names["objid"]], errors="raise"),
            "psf_fwhm_arcsec": np.sqrt(major * minor),
        }
    )


def query_ps1_detection_seeing_bulk(ps1objids, *, max_records=200_000):
    """Fetch a batch of PS1 objects from the MAST DR2 TAP Detection table."""

    object_ids = list(dict.fromkeys(int(value) for value in ps1objids))
    if not object_ids:
        return _normalise_ps1_response(pd.DataFrame())
    id_sql = ",".join(str(value) for value in object_ids)
    query = (
        "SELECT detectid,objid,psfmajorfwhm,psfminorfwhm "
        f"FROM dbo.detection WHERE objid IN ({id_sql})"
    )
    table = _read_csv_url(
        PS1_TAP_SYNC_URL,
        {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "csv",
            "MAXREC": str(int(max_records)),
            "QUERY": query,
        },
    )
    result = _normalise_ps1_response(table)
    if len(result) >= int(max_records):
        raise RuntimeError(
            "PS1 TAP batch reached MAXREC; reduce --ps1-chunk-size."
        )
    return result


def query_ps1_detection_seeing(ps1objid):
    """Backward-compatible single-object PS1 query."""

    return query_ps1_detection_seeing_bulk([ps1objid]).drop(
        columns="ps1objID", errors="ignore"
    )


def _normalise_ztf_response(table):
    columns = ["mjd", "fieldid", "rcidin", "filterID", "psf_fwhm_arcsec"]
    if table.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(table, ["field", "rcid", "fid", "obsjd", "seeing"], "ZTF IBE")
    result = table.rename(
        columns={
            "field": "fieldid",
            "rcid": "rcidin",
            "fid": "filterID",
            "seeing": "psf_fwhm_arcsec",
        }
    ).copy()
    result["mjd"] = pd.to_numeric(result["obsjd"], errors="coerce") - 2400000.5
    for column in ("fieldid", "rcidin", "filterID", "psf_fwhm_arcsec"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result[columns]


def query_ztf_field_seeing(fieldid, mjd_start, mjd_end):
    """Fetch all ZTF quadrant metadata for one field and time interval."""

    where = (
        f"field={int(fieldid)} AND obsjd BETWEEN "
        f"{float(mjd_start) + 2400000.5:.8f} AND "
        f"{float(mjd_end) + 2400000.5:.8f}"
    )
    table = _read_csv_url(
        ZTF_IBE_URL,
        {
            "WHERE": where,
            "COLUMNS": "field,rcid,fid,obsjd,seeing",
            "ct": "csv",
        },
    )
    return _normalise_ztf_response(table)


def query_ztf_image_seeing(ra, dec):
    """Backward-compatible positional ZTF query."""

    table = _read_csv_url(
        ZTF_IBE_URL,
        {
            "POS": f"{float(ra):.10f},{float(dec):.10f}",
            "COLUMNS": "field,rcid,fid,obsjd,seeing",
            "ct": "csv",
        },
    )
    return _normalise_ztf_response(table)


def query_sdss_field_metadata(mjd_start, mjd_end, dec_min, dec_max):
    """Fetch SDSS field footprints, per-band times, and per-band PSF widths."""

    columns = ",".join(
        [
            "f.run",
            "f.rerun",
            "f.camcol",
            "f.field",
            "f.raMin",
            "f.raMax",
            "f.decMin",
            "f.decMax",
            *(f"f.mjd_{band}" for band in SDSS_BANDS),
            *(f"f.psfWidth_{band}" for band in SDSS_BANDS),
        ]
    )
    sql = (
        f"SELECT {columns} FROM Field f "
        f"WHERE f.mjd_r BETWEEN {float(mjd_start):.8f} AND {float(mjd_end):.8f} "
        f"AND f.decMin <= {float(dec_max):.10f} "
        f"AND f.decMax >= {float(dec_min):.10f}"
    )
    table = _read_csv_url(
        SDSS_SQL_URL, {"cmd": sql, "format": "csv"}, comment="#"
    )
    expected = [
        "raMin",
        "raMax",
        "decMin",
        "decMax",
        *(f"mjd_{band}" for band in SDSS_BANDS),
        *(f"psfWidth_{band}" for band in SDSS_BANDS),
    ]
    if table.empty:
        return pd.DataFrame(columns=expected)
    _require_columns(table, expected, "SDSS SkyServer")
    for column in expected:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    return table


def query_sdss_field_seeing(ra, dec, radius_arcsec=2.0):
    """Backward-compatible per-coordinate SDSS query."""

    half_width = float(radius_arcsec) / 3600.0
    columns = ",".join(f"f.psfWidth_{band}" for band in SDSS_BANDS)
    sql = (
        "SELECT p.objID,p.ra,p.dec,p.mjd,p.run,p.rerun,p.camcol,p.field,"
        f"{columns} FROM PhotoObjAll p JOIN Field f ON p.fieldID=f.fieldID "
        f"WHERE p.ra BETWEEN {float(ra) - half_width:.10f} "
        f"AND {float(ra) + half_width:.10f} "
        f"AND p.dec BETWEEN {float(dec) - half_width:.10f} "
        f"AND {float(dec) + half_width:.10f} ORDER BY p.mjd"
    )
    table = _read_csv_url(
        SDSS_SQL_URL, {"cmd": sql, "format": "csv"}, comment="#"
    )
    if table.empty:
        return table
    separation = np.hypot(
        (pd.to_numeric(table["ra"]) - float(ra)) * np.cos(np.deg2rad(float(dec))),
        pd.to_numeric(table["dec"]) - float(dec),
    )
    return (
        table.assign(_separation=separation)
        .sort_values("_separation")
        .drop_duplicates("mjd", keep="first")
    )


def _nearest_ztf_seeing(raw, archive, max_delta_days=0.01):
    """Vectorized nearest-time match within each ZTF image-key group."""

    values = np.full(len(raw), np.nan, dtype=float)
    if raw.empty or archive.empty:
        return values
    archive_groups = {
        tuple(int(value) for value in key): group.sort_values("mjd")
        for key, group in archive.dropna(
            subset=["fieldid", "rcidin", "filterID", "mjd"]
        ).groupby(["fieldid", "rcidin", "filterID"], sort=False)
    }
    raw_groups = raw.groupby(["fieldid", "rcidin", "filterID"], sort=False).indices
    all_times = pd.to_numeric(raw["mjd"], errors="coerce").to_numpy(dtype=float)
    for key, positions in raw_groups.items():
        candidates = archive_groups.get(tuple(int(value) for value in key))
        if candidates is None or candidates.empty:
            continue
        positions = np.asarray(positions, dtype=int)
        raw_times = all_times[positions]
        archive_times = candidates["mjd"].to_numpy(dtype=float)
        archive_seeing = candidates["psf_fwhm_arcsec"].to_numpy(dtype=float)
        insertion = np.searchsorted(archive_times, raw_times)
        right = np.clip(insertion, 0, len(archive_times) - 1)
        left = np.clip(insertion - 1, 0, len(archive_times) - 1)
        right_delta = np.abs(raw_times - archive_times[right])
        left_delta = np.abs(raw_times - archive_times[left])
        use_right = right_delta < left_delta
        nearest = np.where(use_right, right, left)
        delta = np.where(use_right, right_delta, left_delta)
        matched = np.isfinite(raw_times) & (delta <= float(max_delta_days))
        values[positions[matched]] = archive_seeing[nearest[matched]]
    return values


def _time_windows(start, end, width_days, *, padding=0.0):
    start, end, width_days = float(start), float(end), float(width_days)
    if not np.isfinite(start) or not np.isfinite(end) or end < start:
        return []
    if width_days <= 0:
        raise ValueError("Time-window width must be positive.")
    edges = np.arange(math.floor(start), math.ceil(end) + width_days, width_days)
    if len(edges) < 2:
        edges = np.array([math.floor(start), math.ceil(end) + width_days])
    return [
        (float(left - padding), float(min(right, math.ceil(end) + 1) + padding))
        for left, right in zip(edges[:-1], edges[1:])
        if left <= end
    ]


def _atomic_to_parquet(table, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _cached_query(path, query, *, resume=True):
    path = Path(path)
    if resume and path.exists():
        return pd.read_parquet(path)
    table = query()
    _atomic_to_parquet(table, path)
    return table


def _fetch_sdss_window_cached(
    cache_dir, start, end, dec_min, dec_max, *, resume=True, min_width_days=0.5
):
    """Fetch one SDSS interval, recursively splitting legacy buffer overflows."""

    cache_dir = Path(cache_dir)
    path = cache_dir / (
        f"fields-{start:.3f}-{end:.3f}-dec-{dec_min:.3f}-{dec_max:.3f}.parquet"
    )
    try:
        return _cached_query(
            path,
            lambda: query_sdss_field_metadata(start, end, dec_min, dec_max),
            resume=resume,
        )
    except RuntimeError as error:
        if (
            "Response Buffer Limit Exceeded" not in str(error)
            or float(end) - float(start) <= float(min_width_days)
        ):
            raise
        midpoint = (float(start) + float(end)) / 2.0
        left = _fetch_sdss_window_cached(
            cache_dir, start, midpoint, dec_min, dec_max,
            resume=resume, min_width_days=min_width_days,
        )
        right = _fetch_sdss_window_cached(
            cache_dir, midpoint, end, dec_min, dec_max,
            resume=resume, min_width_days=min_width_days,
        )
        combined = _concat_nonempty([left, right])
        _atomic_to_parquet(combined, path)
        return combined


def _run_batches(specifications, worker, *, max_workers=4, progress_bar=True, desc=None):
    results = [None] * len(specifications)
    if int(max_workers) <= 1:
        iterator = tqdm(
            enumerate(specifications), total=len(specifications),
            disable=not progress_bar, desc=desc, dynamic_ncols=True,
        )
        for index, specification in iterator:
            results[index] = worker(specification)
        return results
    with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
        futures = {
            executor.submit(worker, specification): index
            for index, specification in enumerate(specifications)
        }
        iterator = tqdm(
            as_completed(futures), total=len(futures),
            disable=not progress_bar, desc=desc, dynamic_ncols=True,
        )
        for future in iterator:
            results[futures[future]] = future.result()
    return results


def _stable_id_digest(values):
    return hashlib.sha256(",".join(map(str, values)).encode()).hexdigest()[:12]


def _load_catalog(object_ids=None, limit=None):
    catalog = pd.read_parquet(
        resolve_qvc_data_path(f"{S82_DIR}/Catalog.parquet"),
        columns=["objectId", "ps1objID", "RA", "DEC"],
    )
    if object_ids is not None:
        requested = {str(value) for value in object_ids}
        catalog = catalog[catalog["objectId"].astype(str).isin(requested)]
        missing = requested - set(catalog["objectId"].astype(str))
        if missing:
            raise ValueError(
                "Catalog does not contain requested objectId values: "
                + ", ".join(sorted(missing)[:5])
            )
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("--limit must be positive.")
        catalog = catalog.iloc[: int(limit)]
    if catalog.empty:
        raise ValueError("No catalog objects were selected.")
    return catalog.copy()


def _filter_raw_to_catalog(raw, catalog, id_column):
    # Catalog identifiers are stored as strings while survey tables commonly
    # use int64.  Never coerce the 18-digit PS1 IDs through float64: doing so
    # silently loses precision and drops valid matches.
    selected = set(catalog[id_column].dropna().astype(str))
    return raw[raw[id_column].astype(str).isin(selected)].copy()


def _clean_seeing(table):
    table = table.copy()
    values = pd.to_numeric(table["psf_fwhm_arcsec"], errors="coerce")
    invalid = (~np.isfinite(values)) | (values <= 0.0) | (values > 20.0)
    invalid_nonmissing = int((invalid & values.notna()).sum())
    table["psf_fwhm_arcsec"] = values.mask(invalid)
    return table, invalid_nonmissing


def _coverage_by_filter(raw, sidecar, keys):
    if "filterID" not in raw.columns:
        return {}
    # filterID is itself a key for SDSS/ZTF but not PS1.
    columns = list(dict.fromkeys([*keys, "filterID"]))
    joined = raw[columns].drop_duplicates(keys).merge(
        sidecar[[*keys, "psf_fwhm_arcsec"]], on=keys, how="left", validate="one_to_one"
    )
    result = {}
    for filter_id, group in joined.groupby("filterID", dropna=False):
        total = len(group)
        matched = int(group["psf_fwhm_arcsec"].notna().sum())
        result[str(filter_id)] = {
            "matched": matched,
            "total": total,
            "fraction": matched / total if total else 0.0,
        }
    return result


def _coverage_table(survey, raw, sidecar, keys):
    """Return match counts for every survey object and filter combination."""

    object_column = "objectId" if survey == "sdss" else "ps1objID"
    columns = list(dict.fromkeys([*keys, object_column, "filterID"]))
    joined = raw[columns].drop_duplicates(keys).merge(
        sidecar[[*keys, "psf_fwhm_arcsec"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    joined["matched"] = joined["psf_fwhm_arcsec"].notna().astype(np.int64)
    coverage = (
        joined.groupby([object_column, "filterID"], dropna=False, as_index=False)
        .agg(matched=("matched", "sum"), total=("matched", "size"))
    )
    coverage["fraction"] = coverage["matched"] / coverage["total"]
    return coverage


def _write_sidecar(survey, table, raw, *, output_dir, invalid_count=0):
    keys = SIDECAR_KEYS[survey]
    _require_columns(table, [*keys, "psf_fwhm_arcsec"], f"{survey} sidecar")
    table = table.drop_duplicates(keys, keep="last").reset_index(drop=True)
    output_dir = Path(output_dir)
    path = output_dir / SIDECAR_FILENAMES[survey]
    _atomic_to_parquet(table, path)
    coverage_path = output_dir / f"seeing_coverage_{survey}.parquet"
    _atomic_to_parquet(_coverage_table(survey, raw, table, keys), coverage_path)
    matched, total = int(table["psf_fwhm_arcsec"].notna().sum()), len(table)
    object_column = "objectId" if survey == "sdss" else "ps1objID"
    return {
        "path": str(path), "coverage_path": str(coverage_path),
        "matched": matched, "total": total,
        "fraction": matched / total if total else 0.0,
        "objects": int(raw[object_column].nunique()) if object_column in raw else None,
        "invalid_seeing_values": int(invalid_count),
        "coverage_by_filter": _coverage_by_filter(raw, table, keys),
    }


def _concat_nonempty(parts, columns=None):
    present = [part for part in parts if part is not None and not part.empty]
    return pd.concat(present, ignore_index=True) if present else pd.DataFrame(columns=columns)


def enrich_ps1_seeing(
    catalog, *, output_dir, cache_dir, resume=True, max_workers=4,
    chunk_size=250, progress_bar=True,
):
    raw = pd.read_parquet(
        resolve_qvc_data_path(f"{S82_DIR}/dr16s82_ps1LCRaw.parquet"),
        columns=["ps1objID", "detectID", "filterID"],
    )
    raw = _filter_raw_to_catalog(raw, catalog, "ps1objID")
    object_ids = sorted(int(value) for value in raw["ps1objID"].dropna().unique())
    chunks = [object_ids[i:i + int(chunk_size)] for i in range(0, len(object_ids), int(chunk_size))]
    cache_dir = Path(cache_dir) / "ps1-dr2"

    def worker(chunk):
        path = cache_dir / (
            f"objids-{chunk[0]}-{chunk[-1]}-{_stable_id_digest(chunk)}.parquet"
        )
        return _cached_query(path, lambda: query_ps1_detection_seeing_bulk(chunk), resume=resume)

    parts = _run_batches(
        chunks, worker, max_workers=max_workers, progress_bar=progress_bar,
        desc="PS1 TAP batches",
    )
    archive = _concat_nonempty(
        parts, ["detectID", "ps1objID", "psf_fwhm_arcsec"]
    ).drop_duplicates("detectID", keep="last")
    sidecar = raw[["detectID"]].drop_duplicates().merge(
        archive[["detectID", "psf_fwhm_arcsec"]], on="detectID",
        how="left", validate="one_to_one",
    )
    sidecar, invalid = _clean_seeing(sidecar)
    return _write_sidecar("ps1", sidecar, raw, output_dir=output_dir, invalid_count=invalid)


def enrich_ztf_seeing(
    catalog, *, output_dir, cache_dir, resume=True, max_workers=4,
    time_chunk_days=365, max_delta_days=0.01, progress_bar=True,
):
    keys = SIDECAR_KEYS["ztf"]
    raw = pd.read_parquet(
        resolve_qvc_data_path(f"{S82_DIR}/dr16s82_ZuberLCRaw.parquet"), columns=keys
    )
    raw = _filter_raw_to_catalog(raw, catalog, "ps1objID")
    raw = raw.drop_duplicates(keys).reset_index(drop=True)
    specifications = []
    for fieldid, group in raw.groupby("fieldid", sort=True):
        specifications.extend(
            (int(fieldid), start, end)
            for start, end in _time_windows(
                group["mjd"].min(), group["mjd"].max(), time_chunk_days,
                padding=max_delta_days,
            )
        )
    cache_dir = Path(cache_dir) / "ztf-public"

    def worker(specification):
        fieldid, start, end = specification
        path = cache_dir / f"field-{fieldid:06d}-{start:.3f}-{end:.3f}.parquet"
        return _cached_query(
            path, lambda: query_ztf_field_seeing(fieldid, start, end), resume=resume
        )

    parts = _run_batches(
        specifications, worker, max_workers=max_workers, progress_bar=progress_bar,
        desc="ZTF field windows",
    )
    archive = _concat_nonempty(
        parts, ["mjd", "fieldid", "rcidin", "filterID", "psf_fwhm_arcsec"]
    ).drop_duplicates(["mjd", "fieldid", "rcidin", "filterID"], keep="last")
    sidecar = raw[keys].copy()
    sidecar["psf_fwhm_arcsec"] = _nearest_ztf_seeing(
        raw, archive, max_delta_days=max_delta_days
    )
    sidecar, invalid = _clean_seeing(sidecar)
    return _write_sidecar("ztf", sidecar, raw, output_dir=output_dir, invalid_count=invalid)


def _ra_center(ra_min, ra_max):
    width = (ra_max - ra_min + 540.0) % 360.0 - 180.0
    return (ra_min + width / 2.0) % 360.0


def _unit_sphere(ra_degrees, dec_degrees):
    ra, dec = np.deg2rad(ra_degrees), np.deg2rad(dec_degrees)
    cos_dec = np.cos(dec)
    return np.column_stack((cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)))


def _match_sdss_field_seeing(raw, fields, *, max_delta_days=0.08, neighbors=12):
    """Match epochs to SDSS field footprints using sky position and band time."""

    output = np.full(len(raw), np.nan, dtype=float)
    if raw.empty or fields.empty:
        return output
    target_ra = pd.to_numeric(raw["RA"], errors="coerce").to_numpy(float) % 360.0
    target_dec = pd.to_numeric(raw["DEC"], errors="coerce").to_numpy(float)
    target_mjd = pd.to_numeric(raw["mjd"], errors="coerce").to_numpy(float)
    filters = pd.to_numeric(raw["filterID"], errors="coerce").to_numpy(float)
    for filter_id, band in enumerate(SDSS_BANDS):
        time_column, seeing_column = f"mjd_{band}", f"psfWidth_{band}"
        band_fields = fields.dropna(
            subset=[time_column, seeing_column, "raMin", "raMax", "decMin", "decMax"]
        ).copy()
        if band_fields.empty:
            continue
        band_fields["_day"] = np.floor(band_fields[time_column]).astype(int)
        by_day = {int(day): group for day, group in band_fields.groupby("_day")}
        band_positions = np.flatnonzero(filters == filter_id)
        raw_days = np.floor(target_mjd[band_positions]).astype(int)
        for day in np.unique(raw_days):
            positions = band_positions[raw_days == day]
            candidates = _concat_nonempty(
                [by_day.get(int(day + offset)) for offset in (-1, 0, 1)]
            )
            if candidates.empty:
                continue
            ra_min = candidates["raMin"].to_numpy(float) % 360.0
            ra_max = candidates["raMax"].to_numpy(float) % 360.0
            dec_min = candidates["decMin"].to_numpy(float)
            dec_max = candidates["decMax"].to_numpy(float)
            field_ra = _ra_center(ra_min, ra_max)
            field_dec = (dec_min + dec_max) / 2.0
            field_mjd = candidates[time_column].to_numpy(float)
            field_features = np.column_stack(
                (_unit_sphere(field_ra, field_dec), (field_mjd - day) * 0.5)
            )
            point_features = np.column_stack(
                (_unit_sphere(target_ra[positions], target_dec[positions]),
                 (target_mjd[positions] - day) * 0.5)
            )
            k = min(max(1, int(neighbors)), len(candidates))
            distances, indices = cKDTree(field_features).query(point_features, k=k)
            if k == 1:
                distances, indices = distances[:, None], indices[:, None]
            p_ra = target_ra[positions, None]
            c_ra_min, c_ra_max = ra_min[indices], ra_max[indices]
            inside_ra = np.where(
                c_ra_min <= c_ra_max,
                (p_ra >= c_ra_min) & (p_ra <= c_ra_max),
                (p_ra >= c_ra_min) | (p_ra <= c_ra_max),
            )
            p_dec = target_dec[positions, None]
            inside_dec = (p_dec >= dec_min[indices]) & (p_dec <= dec_max[indices])
            delta = np.abs(target_mjd[positions, None] - field_mjd[indices])
            scores = np.where(
                inside_ra & inside_dec & (delta <= float(max_delta_days)),
                distances, np.inf,
            )
            best = np.argmin(scores, axis=1)
            matched = np.isfinite(scores[np.arange(len(positions)), best])
            seeing = candidates[seeing_column].to_numpy(float)
            output[positions[matched]] = seeing[indices[matched, best[matched]]]
    return output


def enrich_sdss_seeing(
    catalog, *, output_dir, cache_dir, resume=True, max_workers=4,
    time_chunk_days=30, max_delta_days=0.08, progress_bar=True,
):
    keys = SIDECAR_KEYS["sdss"]
    raw = pd.read_parquet(
        resolve_qvc_data_path(f"{S82_DIR}/dr16s82_sdssLCRaw.parquet"), columns=keys
    )
    raw = _filter_raw_to_catalog(raw, catalog, "objectId")
    coordinates = catalog[["objectId", "RA", "DEC"]].copy()
    coordinates["_object_key"] = coordinates["objectId"].astype(str)
    raw["_object_key"] = raw["objectId"].astype(str)
    raw = raw.merge(
        coordinates[["_object_key", "RA", "DEC"]], on="_object_key",
        how="left", validate="many_to_one",
    ).drop(columns="_object_key")
    if raw[["RA", "DEC"]].isna().any().any():
        raise ValueError("Some SDSS light-curve objects lack catalog coordinates.")
    dec_min, dec_max = float(catalog["DEC"].min()) - 0.3, float(catalog["DEC"].max()) + 0.3
    specifications = _time_windows(
        raw["mjd"].min(), raw["mjd"].max(), time_chunk_days,
        padding=max_delta_days,
    )
    cache_dir = Path(cache_dir) / "sdss-stripe82-dr7"

    def worker(specification):
        start, end = specification
        return _fetch_sdss_window_cached(
            cache_dir, start, end, dec_min, dec_max, resume=resume
        )

    parts = _run_batches(
        specifications, worker, max_workers=max_workers, progress_bar=progress_bar,
        desc="SDSS field windows",
    )
    archive = _concat_nonempty(parts)
    if not archive.empty:
        archive = archive.drop_duplicates(["run", "rerun", "camcol", "field"], keep="last")
    sidecar = raw[keys].copy()
    sidecar["psf_fwhm_arcsec"] = _match_sdss_field_seeing(
        raw, archive, max_delta_days=max_delta_days
    )
    sidecar, invalid = _clean_seeing(sidecar)
    return _write_sidecar("sdss", sidecar, raw, output_dir=output_dir, invalid_count=invalid)


def _resolve_output_dir(output_dir):
    return Path(resolve_qvc_data_path(S82_DIR)) if output_dir is None else Path(output_dir)


def plan_bulk_queries(
    catalog, *, surveys=SURVEYS, ps1_chunk_size=250,
    ztf_time_chunk_days=365, sdss_time_chunk_days=30,
):
    plan = {"objects": len(catalog), "surveys": {}}
    if "ps1" in surveys:
        raw = pd.read_parquet(
            resolve_qvc_data_path(f"{S82_DIR}/dr16s82_ps1LCRaw.parquet"),
            columns=["ps1objID"],
        )
        raw = _filter_raw_to_catalog(raw, catalog, "ps1objID")
        count = raw["ps1objID"].nunique()
        plan["surveys"]["ps1"] = {
            "object_ids": int(count),
            "queries": int(math.ceil(count / int(ps1_chunk_size))),
        }
    if "ztf" in surveys:
        raw = pd.read_parquet(
            resolve_qvc_data_path(f"{S82_DIR}/dr16s82_ZuberLCRaw.parquet"),
            columns=["ps1objID", "fieldid", "mjd"],
        )
        raw = _filter_raw_to_catalog(raw, catalog, "ps1objID")
        count = sum(
            len(_time_windows(g["mjd"].min(), g["mjd"].max(), ztf_time_chunk_days))
            for _, g in raw.groupby("fieldid")
        )
        plan["surveys"]["ztf"] = {
            "epochs": len(raw), "fields": int(raw["fieldid"].nunique()),
            "queries": int(count),
        }
    if "sdss" in surveys:
        raw = pd.read_parquet(
            resolve_qvc_data_path(f"{S82_DIR}/dr16s82_sdssLCRaw.parquet"),
            columns=["objectId", "mjd"],
        )
        raw = _filter_raw_to_catalog(raw, catalog, "objectId")
        count = len(_time_windows(raw["mjd"].min(), raw["mjd"].max(), sdss_time_chunk_days))
        plan["surveys"]["sdss"] = {"epochs": len(raw), "queries": int(count)}
    return plan


def enrich_all_seeing(
    *, output_dir=None, cache_dir=None, surveys=SURVEYS, object_ids=None,
    limit=None, resume=True, max_workers=4, ps1_chunk_size=250,
    ztf_time_chunk_days=365, sdss_time_chunk_days=30,
    ztf_max_delta_days=0.01, sdss_max_delta_days=0.08, progress_bar=True,
):
    """Generate complete or selected sidecars with survey-level bulk queries."""

    surveys = tuple(dict.fromkeys(str(value).lower() for value in surveys))
    unknown = set(surveys) - set(SURVEYS)
    if unknown:
        raise ValueError(f"Unknown surveys: {sorted(unknown)}")
    catalog = _load_catalog(object_ids=object_ids, limit=limit)
    output_dir = _resolve_output_dir(output_dir)
    cache_dir = Path(cache_dir) if cache_dir else output_dir / ".seeing-cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    common = dict(
        output_dir=output_dir, cache_dir=cache_dir, resume=resume,
        max_workers=max_workers, progress_bar=progress_bar,
    )
    summaries = {}
    if "ps1" in surveys:
        summaries["ps1"] = enrich_ps1_seeing(
            catalog, chunk_size=ps1_chunk_size, **common
        )
    if "ztf" in surveys:
        summaries["ztf"] = enrich_ztf_seeing(
            catalog, time_chunk_days=ztf_time_chunk_days,
            max_delta_days=ztf_max_delta_days, **common,
        )
    if "sdss" in surveys:
        summaries["sdss"] = enrich_sdss_seeing(
            catalog, time_chunk_days=sdss_time_chunk_days,
            max_delta_days=sdss_max_delta_days, **common,
        )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive_releases": {
            "ps1": "dr2", "sdss": "stripe82-dr7", "ztf": "public"
        },
        "selected_objects": len(catalog), "surveys": list(surveys),
        "resume": bool(resume), "summaries": summaries,
        "cache": {
            "directory": str(cache_dir),
            "partitions": sorted(
                str(path.relative_to(cache_dir))
                for path in cache_dir.rglob("*.parquet")
            ),
        },
    }
    _atomic_write_json(manifest, output_dir / "seeing_enrichment_manifest.json")
    return summaries


def enrich_target_seeing(object_id, **kwargs):
    return enrich_all_seeing(object_ids=[object_id], **kwargs)


def _parse_object_list(path):
    values = []
    for line in Path(path).read_text().splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            values.append(value.split(",", 1)[0].strip())
    return values


def _print_summary(summary):
    for survey, values in summary.items():
        print(
            f"{survey}: matched {values['matched']}/{values['total']} epochs "
            f"({values['fraction']:.1%}); wrote {values['path']}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object_id", nargs="?", help="One QVC Catalog objectId.")
    parser.add_argument("--object-list", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--survey", choices=(*SURVEYS, "all"), default="all")
    parser.add_argument("--output-dir", "--output_dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--ps1-chunk-size", type=int, default=250)
    parser.add_argument("--ztf-time-chunk-days", type=float, default=365)
    parser.add_argument("--sdss-time-chunk-days", type=float, default=30)
    parser.add_argument("--ztf-max-delta-days", type=float, default=0.01)
    parser.add_argument("--sdss-max-delta-days", type=float, default=0.08)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Reuse completed cache partitions (default: enabled).",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.max_workers <= 0 or args.ps1_chunk_size <= 0:
        parser.error("--max-workers and --ps1-chunk-size must be positive")
    if args.object_id and args.object_list:
        parser.error("object_id and --object-list are mutually exclusive")
    object_ids = None
    if args.object_id:
        object_ids = [args.object_id]
    elif args.object_list:
        object_ids = _parse_object_list(args.object_list)
    surveys = SURVEYS if args.survey == "all" else (args.survey,)
    catalog = _load_catalog(object_ids=object_ids, limit=args.limit)
    if args.dry_run:
        plan = plan_bulk_queries(
            catalog, surveys=surveys, ps1_chunk_size=args.ps1_chunk_size,
            ztf_time_chunk_days=args.ztf_time_chunk_days,
            sdss_time_chunk_days=args.sdss_time_chunk_days,
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return plan
    summary = enrich_all_seeing(
        output_dir=args.output_dir, cache_dir=args.cache_dir, surveys=surveys,
        object_ids=object_ids, limit=args.limit, resume=args.resume,
        max_workers=args.max_workers, ps1_chunk_size=args.ps1_chunk_size,
        ztf_time_chunk_days=args.ztf_time_chunk_days,
        sdss_time_chunk_days=args.sdss_time_chunk_days,
        ztf_max_delta_days=args.ztf_max_delta_days,
        sdss_max_delta_days=args.sdss_max_delta_days,
        progress_bar=not args.no_progress,
    )
    _print_summary(summary)
    return summary


if __name__ == "__main__":
    main()
