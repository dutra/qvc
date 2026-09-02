#!/usr/bin/env python3
"""Download FIRST, cross-match it to the S82 catalog, and compute radio loudness.

The default invocation processes the complete S82 catalog::

    python scripts/download_first_radio.py \\
        --s82-catalog data/S82/Catalog.parquet \\
        --output data/S82/first_radio_matches.parquet

For a live one-object smoke test, add ``--max-objects 1`` and choose a temporary
output path.  Downloads are cached, resumable, validated, and atomically moved
into place.  Network access is never needed after the three source products and
the normalized FIRST Parquet cache have been verified.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import astropy
import numpy as np
import pandas as pd
import pyarrow
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS


FIRST_CATALOG_URL = "https://sundog.stsci.edu/first/catalogs/first_14dec17.fits.gz"
FIRST_NORTH_RMS_URL = (
    "https://sundog.stsci.edu/first/catalogs/coverage-north-3arcmin-14dec17.fits"
)
FIRST_SOUTH_RMS_URL = (
    "https://sundog.stsci.edu/first/catalogs/coverage-south-3arcmin-14dec17.fits"
)
NORMALIZED_SCHEMA_VERSION = 1
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class RemoteMetadata:
    url: str
    etag: str | None
    last_modified: str | None
    content_length: int | None
    accept_ranges: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _header(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", {})
    value = headers.get(name) if headers is not None else None
    return None if value is None else str(value)


def probe_remote(
    url: str,
    timeout: float,
    opener: Callable[..., Any] = urlopen,
) -> RemoteMetadata:
    """Read validators and size without downloading the source body."""
    request = Request(url, method="HEAD", headers={"User-Agent": "qvc-first/1"})
    try:
        with opener(request, timeout=timeout) as response:
            length = _header(response, "Content-Length")
            return RemoteMetadata(
                url=url,
                etag=_header(response, "ETag"),
                last_modified=_header(response, "Last-Modified"),
                content_length=int(length) if length and length.isdigit() else None,
                accept_ranges=(_header(response, "Accept-Ranges") or "").lower()
                == "bytes",
            )
    except HTTPError as exc:
        if exc.code not in (405, 501):
            raise

    # A few simple HTTP servers do not implement HEAD.  A one-byte range probe
    # still yields validators; Content-Range contains the full object length.
    request = Request(
        url,
        headers={"Range": "bytes=0-0", "User-Agent": "qvc-first/1"},
    )
    with opener(request, timeout=timeout) as response:
        content_range = _header(response, "Content-Range") or ""
        total = content_range.rsplit("/", 1)[-1]
        length = int(total) if total.isdigit() else None
        if length is None:
            raw_length = _header(response, "Content-Length")
            length = int(raw_length) if raw_length and raw_length.isdigit() else None
        return RemoteMetadata(
            url=url,
            etag=_header(response, "ETag"),
            last_modified=_header(response, "Last-Modified"),
            content_length=length,
            accept_ranges=getattr(response, "status", 200) == 206,
        )


def _metadata_matches(saved: Mapping[str, Any], remote: RemoteMetadata) -> bool:
    if saved.get("url") != remote.url:
        return False
    for name in ("etag", "last_modified", "content_length"):
        old, new = saved.get(name), getattr(remote, name)
        if old is not None and new is not None and old != new:
            return False
    return True


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gzipped_fits(path: Path) -> None:
    """Validate both the gzip trailer and the FITS table structure."""
    with gzip.open(path, "rb") as handle:
        for _ in iter(lambda: handle.read(CHUNK_SIZE), b""):
            pass
    with fits.open(path, memmap=False) as hdus:
        if not any(hdu.data is not None and getattr(hdu.data, "dtype", None).names for hdu in hdus):
            raise ValueError(f"No FITS table found in {path}")
        hdus.verify("exception")


def validate_image_fits(path: Path) -> None:
    with fits.open(path, memmap=False) as hdus:
        images = [hdu for hdu in hdus if hdu.data is not None]
        if not images or np.asarray(images[0].data).ndim != 2:
            raise ValueError(f"No two-dimensional FITS image found in {path}")
        if not WCS(images[0].header).has_celestial:
            raise ValueError(f"No celestial WCS found in {path}")
        hdus.verify("exception")


def download_with_resume(
    url: str,
    destination: Path,
    *,
    retries: int = 5,
    retry_delay: float = 2.0,
    timeout: float = 60.0,
    force: bool = False,
    validator: Callable[[Path], None] | None = None,
    opener: Callable[..., Any] = urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Download *url* using a resumable ``.part`` file and atomic rename."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    metadata_path = destination.with_name(destination.name + ".http.json")
    part_metadata_path = part.with_name(part.name + ".http.json")
    saved: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            saved = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            saved = {}
    partial_saved: dict[str, Any] = {}
    if part_metadata_path.exists():
        try:
            partial_saved = json.loads(part_metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            partial_saved = {}

    remote: RemoteMetadata | None = None
    probe_error: Exception | None = None
    for probe_attempt in range(retries + 1):
        try:
            remote = probe_remote(url, timeout, opener)
            break
        except (OSError, HTTPError, URLError) as exc:
            probe_error = exc
            if probe_attempt < retries:
                sleep(retry_delay * (2**probe_attempt))
    if remote is None:
        # A previously completed file remains useful on an offline machine.
        # Its FITS/gzip structure and hash are rechecked; provenance explicitly
        # records that its HTTP validators could not be refreshed.
        if not force and destination.exists():
            try:
                if validator:
                    validator(destination)
                result = {
                    "url": url,
                    "etag": saved.get("etag"),
                    "last_modified": saved.get("last_modified"),
                    "content_length": destination.stat().st_size,
                    "accept_ranges": saved.get("accept_ranges"),
                    "path": str(destination),
                    "reused": True,
                    "remote_validated": False,
                    "sha256": sha256_file(destination),
                }
                return result
            except Exception:
                pass
        raise RuntimeError(f"Could not inspect {url}") from probe_error

    replacement_required = force or bool(saved and not _metadata_matches(saved, remote))
    if force or (part.exists() and not _metadata_matches(partial_saved, remote)):
        part.unlink(missing_ok=True)
        part_metadata_path.unlink(missing_ok=True)

    in_progress_metadata = asdict(remote)
    in_progress_metadata["download_complete"] = False
    _atomic_json(part_metadata_path, in_progress_metadata)

    if destination.exists() and not replacement_required:
        size_ok = remote.content_length is None or destination.stat().st_size == remote.content_length
        if size_ok:
            try:
                if validator:
                    validator(destination)
                result = asdict(remote)
                result.update(
                    {
                        "path": str(destination),
                        "reused": True,
                        "remote_validated": True,
                        "download_complete": True,
                        "sha256": sha256_file(destination),
                    }
                )
                _atomic_json(metadata_path, result)
                part_metadata_path.unlink(missing_ok=True)
                return result
            except Exception:
                destination.unlink(missing_ok=True)

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if remote.content_length is not None and offset > remote.content_length:
                part.unlink()
                offset = 0
            headers = {"User-Agent": "qvc-first/1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
                if remote.etag:
                    headers["If-Range"] = remote.etag
                elif remote.last_modified:
                    headers["If-Range"] = remote.last_modified
            request = Request(url, headers=headers)
            with opener(request, timeout=timeout) as response:
                status = getattr(response, "status", response.getcode())
                if status not in (200, 206):
                    raise OSError(f"Unexpected HTTP status {status} for {url}")
                # A server may ignore Range.  Starting over avoids duplicating
                # the already-downloaded prefix.
                append = bool(offset and status == 206)
                mode = "ab" if append else "wb"
                with part.open(mode) as handle:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())

            actual = part.stat().st_size
            if remote.content_length is not None and actual != remote.content_length:
                raise OSError(
                    f"Truncated download for {url}: got {actual}, expected {remote.content_length} bytes"
                )
            if validator:
                validator(part)
            os.replace(part, destination)
            result = asdict(remote)
            result.update(
                {
                    "path": str(destination),
                    "reused": False,
                    "remote_validated": True,
                    "download_complete": True,
                    "sha256": sha256_file(destination),
                }
            )
            _atomic_json(metadata_path, result)
            part_metadata_path.unlink(missing_ok=True)
            return result
        except (OSError, ValueError, HTTPError, URLError, EOFError) as exc:
            last_error = exc
            # A complete but invalid partial cannot be repaired by appending.
            if part.exists() and remote.content_length is not None and part.stat().st_size >= remote.content_length:
                part.unlink()
            if attempt >= retries:
                break
            sleep(retry_delay * (2**attempt))
    raise RuntimeError(f"Failed to download {url} after {retries + 1} attempts") from last_error


FIRST_COLUMNS = {
    "RA": "first_ra_deg",
    "DEC": "first_dec_deg",
    "SIDEPROB": "first_sidelobe_probability",
    "FPEAK": "first_peak_flux_mjy",
    "FINT": "first_integrated_flux_mjy",
    "RMS": "first_catalog_rms_mjy",
    "MAJOR": "first_major_arcsec",
    "MINOR": "first_minor_arcsec",
    "POSANG": "first_pa_deg",
    "FITTED_MAJOR": "first_fitted_major_arcsec",
    "FITTED_MINOR": "first_fitted_minor_arcsec",
    "FITTED_POSANG": "first_fitted_pa_deg",
    "FLDNAME": "first_field",
    "YEAR": "first_epoch_year",
    "MJD": "first_epoch_jd",
    "MJDRMS": "first_epoch_rms_days",
}


def normalize_first_catalog(
    fits_path: Path,
    normalized_path: Path,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize the official FITS columns and cache only fields used here."""
    fits_hash = sha256_file(fits_path)
    manifest_path = normalized_path.with_suffix(normalized_path.suffix + ".json")
    if not force and normalized_path.exists() and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            if (
                manifest.get("schema_version") == NORMALIZED_SCHEMA_VERSION
                and manifest.get("source_sha256") == fits_hash
            ):
                return pd.read_parquet(normalized_path), manifest
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    with fits.open(fits_path, memmap=False) as hdus:
        table = next(
            hdu.data
            for hdu in hdus
            if hdu.data is not None and getattr(hdu.data, "dtype", None).names
        )
        missing = sorted(set(FIRST_COLUMNS) - set(table.dtype.names or ()))
        if missing:
            raise ValueError(f"FIRST FITS is missing required columns: {missing}")
        columns: dict[str, Any] = {}
        for source, target in FIRST_COLUMNS.items():
            values = np.asarray(table[source])
            if values.dtype.kind == "S":
                values = np.char.decode(values, "ascii", errors="replace")
            columns[target] = values
        normalized = pd.DataFrame(columns)

    normalized.insert(0, "first_catalog_row", np.arange(len(normalized), dtype=np.int64))
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    temp = normalized_path.with_name(normalized_path.name + ".tmp")
    normalized.to_parquet(temp, index=False)
    os.replace(temp, normalized_path)
    manifest = {
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "source_path": str(fits_path),
        "source_sha256": fits_hash,
        "row_count": len(normalized),
        "created_utc": _utc_now(),
    }
    _atomic_json(manifest_path, manifest)
    return normalized, manifest


def _image_and_wcs(path: Path) -> tuple[np.ndarray, WCS, float]:
    with fits.open(path, memmap=False) as hdus:
        hdu = next(hdu for hdu in hdus if hdu.data is not None)
        image = np.asarray(hdu.data, dtype=float).copy()
        wcs = WCS(hdu.header)
        reference_ra = float(hdu.header.get("CRVAL1", 0.0))
    return image, wcs, reference_ra


def lookup_rms_map(path: Path, ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Nearest-pixel RMS lookup, including linear-WCS coordinates across RA=0."""
    image, wcs, reference_ra = _image_and_wcs(path)
    ra_unwrapped = reference_ra + ((np.asarray(ra_deg) - reference_ra + 180.0) % 360.0 - 180.0)
    x, y = wcs.world_to_pixel_values(ra_unwrapped, np.asarray(dec_deg))
    ix = np.rint(x).astype(np.int64)
    iy = np.rint(y).astype(np.int64)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (ix >= 0)
        & (iy >= 0)
        & (ix < image.shape[1])
        & (iy < image.shape[0])
    )
    result = np.full(len(ra_unwrapped), np.nan, dtype=float)
    result[valid] = image[iy[valid], ix[valid]]
    result[~np.isfinite(result) | (result <= 0)] = np.nan
    return result


def lookup_coverage(
    north_path: Path,
    south_path: Path,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
) -> pd.DataFrame:
    north = lookup_rms_map(north_path, ra_deg, dec_deg)
    south = lookup_rms_map(south_path, ra_deg, dec_deg)
    choose_north = np.isfinite(north) & (~np.isfinite(south) | (north <= south))
    choose_south = np.isfinite(south) & ~choose_north
    rms = np.where(choose_north, north, np.where(choose_south, south, np.nan))
    region = np.full(len(rms), "outside", dtype=object)
    region[choose_north] = "north"
    region[choose_south] = "south"
    covered = np.isfinite(rms)
    return pd.DataFrame(
        {
            "first_coverage_region": region,
            "first_covered": covered,
            "first_local_rms_mjy": rms,
            "first_flux_limit_5sigma_mjy_approx": np.where(covered, 5.0 * rms + 0.25, np.nan),
        }
    )


def first_identifier(ra_deg: float, dec_deg: float) -> str:
    coordinate = SkyCoord(float(ra_deg) * u.deg, float(dec_deg) * u.deg)
    text = coordinate.to_string("hmsdms", sep="", precision=1, pad=True, alwayssign=True)
    return "FIRST J" + text.replace(" ", "")


def build_matches(
    s82: pd.DataFrame,
    first: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    core_radius_arcsec: float = 1.5,
    candidate_radius_arcsec: float = 30.0,
    sidelobe_threshold: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one-row-per-S82 primary matches and all nearby candidates."""
    if candidate_radius_arcsec < core_radius_arcsec:
        raise ValueError("candidate radius must be at least the core radius")
    if s82["objectId"].duplicated().any():
        raise ValueError("S82 objectId values must be unique")
    required = {"objectId", "RA", "DEC", "Z_DR16Q", "sdss_i_qg", "qg_fragal", "ebv"}
    missing = sorted(required - set(s82.columns))
    if missing:
        raise ValueError(f"S82 catalog is missing required columns: {missing}")

    primary = s82[list(required)].copy()
    # Preserve a stable, user-facing field order.
    primary = primary[["objectId", "RA", "DEC", "Z_DR16Q", "sdss_i_qg", "qg_fragal", "ebv"]]
    primary = pd.concat([primary.reset_index(drop=True), coverage.reset_index(drop=True)], axis=1)
    s82_coords = SkyCoord(primary["RA"].to_numpy() * u.deg, primary["DEC"].to_numpy() * u.deg)
    first_coords = SkyCoord(
        first["first_ra_deg"].to_numpy() * u.deg,
        first["first_dec_deg"].to_numpy() * u.deg,
    )
    s82_index, first_index, separation, _ = search_around_sky(
        s82_coords, first_coords, candidate_radius_arcsec * u.arcsec
    )
    candidates = first.iloc[first_index].reset_index(drop=True).copy()
    candidates.insert(0, "objectId", primary.iloc[s82_index]["objectId"].to_numpy())
    candidates.insert(1, "s82_row", np.asarray(s82_index, dtype=np.int64))
    candidates.insert(2, "first_match_separation_arcsec", separation.arcsec)
    candidates["first_reliable"] = (
        np.isfinite(candidates["first_sidelobe_probability"])
        & (candidates["first_sidelobe_probability"] <= sidelobe_threshold)
    )
    candidates["first_within_core_radius"] = (
        candidates["first_match_separation_arcsec"] <= core_radius_arcsec
    )
    candidates["first_core_selected"] = False
    if len(candidates):
        candidates["first_source_id"] = [
            first_identifier(ra, dec)
            for ra, dec in zip(candidates["first_ra_deg"], candidates["first_dec_deg"])
        ]
        candidates.sort_values(
            ["s82_row", "first_match_separation_arcsec", "first_catalog_row"],
            inplace=True,
            kind="stable",
        )

    all_counts = candidates.groupby("s82_row").size() if len(candidates) else pd.Series(dtype=int)
    reliable_counts = (
        candidates.loc[candidates["first_reliable"]].groupby("s82_row").size()
        if len(candidates)
        else pd.Series(dtype=int)
    )
    primary["first_candidate_count_30arcsec"] = (
        pd.Series(np.arange(len(primary))).map(all_counts).fillna(0).astype(np.int32)
    )
    primary["first_reliable_candidate_count_30arcsec"] = (
        pd.Series(np.arange(len(primary))).map(reliable_counts).fillna(0).astype(np.int32)
    )
    primary["first_possible_extended"] = primary["first_reliable_candidate_count_30arcsec"] > 1

    reliable_core = candidates[
        candidates["first_reliable"] & candidates["first_within_core_radius"]
    ]
    selected = reliable_core.drop_duplicates("s82_row", keep="first")
    if len(selected):
        candidates.loc[selected.index, "first_core_selected"] = True

    match_columns = [
        "first_match_separation_arcsec",
        "first_source_id",
        *FIRST_COLUMNS.values(),
    ]
    # Remove duplicate RA/DEC entries introduced through FIRST_COLUMNS.values().
    match_columns = list(dict.fromkeys(match_columns))
    primary["first_core_matched"] = False
    for column in match_columns:
        primary[column] = np.nan if column != "first_source_id" and column != "first_field" else None
    if len(selected):
        rows = selected["s82_row"].to_numpy(dtype=int)
        primary.loc[rows, "first_core_matched"] = True
        for column in match_columns:
            primary.loc[rows, column] = selected[column].to_numpy()

    primary["first_match_state"] = np.where(
        primary["first_core_matched"],
        "core_match",
        np.where(primary["first_covered"], "covered_no_core_match", "outside_coverage_no_match"),
    )
    peak = pd.to_numeric(primary["first_peak_flux_mjy"], errors="coerce").to_numpy()
    integrated = pd.to_numeric(primary["first_integrated_flux_mjy"], errors="coerce").to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        concentration = np.sqrt(integrated / peak)
        resolved = np.log10(integrated / peak) > 0.05
    concentration[~np.isfinite(concentration)] = np.nan
    primary["first_concentration_sqrt_fint_fpeak"] = concentration
    primary["first_catalog_resolved"] = np.where(primary["first_core_matched"], resolved, False)
    return primary, candidates.reset_index(drop=True)


def _ab_flux_mjy(magnitude: np.ndarray) -> np.ndarray:
    return 3631.0e3 * np.power(10.0, -0.4 * magnitude)


def add_radio_loudness(
    catalog: pd.DataFrame,
    *,
    alpha_radio: float = -0.7,
    alpha_optical: float = -0.5,
    i_wavelength_angstrom: float = 7625.0,
) -> pd.DataFrame:
    """Add observed R_i and a K-corrected 5 GHz/4400 A proxy."""
    result = catalog.copy()
    optical_mag = pd.to_numeric(result["sdss_i_qg"], errors="coerce").to_numpy(dtype=float)
    redshift = pd.to_numeric(result["Z_DR16Q"], errors="coerce").to_numpy(dtype=float)
    detected = result["first_core_matched"].to_numpy(dtype=bool)
    covered = result["first_covered"].to_numpy(dtype=bool)
    flux = pd.to_numeric(result["first_integrated_flux_mjy"], errors="coerce").to_numpy(dtype=float)
    limit = pd.to_numeric(
        result["first_flux_limit_5sigma_mjy_approx"], errors="coerce"
    ).to_numpy(dtype=float)
    optical_ok = np.isfinite(optical_mag)
    detection_ok = detected & np.isfinite(flux) & (flux > 0)
    limit_ok = ~detected & covered & np.isfinite(limit) & (limit > 0)

    radio_mag = np.full(len(result), np.nan)
    radio_mag[detection_ok] = -2.5 * np.log10((flux[detection_ok] / 1000.0) / 3631.0)
    ri = np.full(len(result), np.nan)
    ri[detection_ok & optical_ok] = 0.4 * (
        optical_mag[detection_ok & optical_ok] - radio_mag[detection_ok & optical_ok]
    )
    optical_flux = _ab_flux_mjy(optical_mag)
    ri_upper = np.full(len(result), np.nan)
    valid_limit = limit_ok & optical_ok & np.isfinite(optical_flux) & (optical_flux > 0)
    ri_upper[valid_limit] = np.log10(limit[valid_limit] / optical_flux[valid_limit])
    ri_state = np.full(len(result), "outside_coverage", dtype=object)
    ri_state[~optical_ok] = "missing_optical"
    ri_state[valid_limit & (ri_upper <= 1.0)] = "constrained_quiet"
    ri_state[valid_limit & (ri_upper > 1.0)] = "limit_ambiguous"
    ri_state[detection_ok & optical_ok & (ri <= 1.0)] = "detected_not_radio_loud"
    ri_state[detection_ok & optical_ok & (ri > 1.0)] = "radio_loud"
    result["first_radio_ab_mag"] = radio_mag
    result["radio_loudness_ri"] = ri
    result["radio_loudness_ri_upper_limit"] = ri_upper
    result["radio_loud_ri"] = pd.array(
        np.where(detection_ok & optical_ok, ri > 1.0, None), dtype="boolean"
    )
    result["radio_loudness_ri_state"] = ri_state

    z_ok = np.isfinite(redshift) & (redshift >= 0)
    base_flux = np.where(detection_ok, flux, limit)
    base_log_ratio = np.full(len(result), np.nan)
    base_ok = optical_ok & np.isfinite(base_flux) & (base_flux > 0) & z_ok
    base_log_ratio[base_ok] = np.log10(base_flux[base_ok] / optical_flux[base_ok])
    k_term = np.full(len(result), np.nan)
    k_term[z_ok] = (
        alpha_radio * np.log10(5.0 / (1.4 * (1.0 + redshift[z_ok])))
        - alpha_optical
        * np.log10(i_wavelength_angstrom / (4400.0 * (1.0 + redshift[z_ok])))
    )
    rest = np.where(detection_ok & base_ok, base_log_ratio + k_term, np.nan)
    rest_upper = np.where(valid_limit & z_ok, base_log_ratio + k_term, np.nan)
    rest_state = np.full(len(result), "outside_coverage", dtype=object)
    rest_state[~optical_ok] = "missing_optical"
    rest_state[optical_ok & ~z_ok] = "missing_redshift"
    rest_valid_limit = valid_limit & z_ok
    rest_state[rest_valid_limit & (rest_upper <= 1.0)] = "constrained_quiet"
    rest_state[rest_valid_limit & (rest_upper > 1.0)] = "limit_ambiguous"
    rest_detected = detection_ok & optical_ok & z_ok
    rest_state[rest_detected & (rest <= 1.0)] = "detected_not_radio_loud"
    rest_state[rest_detected & (rest > 1.0)] = "radio_loud"
    result["radio_loudness_log10_l5ghz_l4400"] = rest
    result["radio_loudness_log10_l5ghz_l4400_upper_limit"] = rest_upper
    result["radio_loud_rest_proxy"] = pd.array(
        np.where(rest_detected, rest > 1.0, None), dtype="boolean"
    )
    result["radio_loudness_rest_state"] = rest_state
    return result


def _candidate_output_path(output: Path) -> Path:
    return output.with_name(output.stem + "_candidates.parquet")


def _provenance_path(output: Path) -> Path:
    return output.with_name(output.stem + ".provenance.json")


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    cache = Path(args.cache_dir)
    products = [
        (FIRST_CATALOG_URL, cache / "first_14dec17.fits.gz", validate_gzipped_fits),
        (FIRST_NORTH_RMS_URL, cache / "coverage-north-3arcmin-14dec17.fits", validate_image_fits),
        (FIRST_SOUTH_RMS_URL, cache / "coverage-south-3arcmin-14dec17.fits", validate_image_fits),
    ]
    downloads: list[dict[str, Any]] = []
    for url, path, validator in products:
        print(f"Checking {url}", flush=True)
        downloads.append(
            download_with_resume(
                url,
                path,
                retries=args.retries,
                retry_delay=args.retry_delay,
                timeout=args.timeout,
                force=args.force_download,
                validator=validator,
            )
        )
    first, normalized_manifest = normalize_first_catalog(
        products[0][1],
        cache / "first_14dec17.normalized.parquet",
        force=args.force_rebuild,
    )
    s82 = pd.read_parquet(args.s82_catalog)
    input_count = len(s82)
    if args.max_objects is not None:
        s82 = s82.iloc[: args.max_objects].copy()
    coverage = lookup_coverage(
        products[1][1], products[2][1], s82["RA"].to_numpy(), s82["DEC"].to_numpy()
    )
    primary, candidates = build_matches(
        s82,
        first,
        coverage,
        core_radius_arcsec=args.core_radius,
        candidate_radius_arcsec=args.candidate_radius,
        sidelobe_threshold=args.sidelobe_threshold,
    )
    primary = add_radio_loudness(
        primary,
        alpha_radio=args.alpha_radio,
        alpha_optical=args.alpha_optical,
        i_wavelength_angstrom=args.i_wavelength,
    )
    output = Path(args.output)
    candidate_output = Path(args.candidates_output) if args.candidates_output else _candidate_output_path(output)
    provenance_output = _provenance_path(output)
    _atomic_parquet(primary, output)
    _atomic_parquet(candidates, candidate_output)
    provenance = {
        "created_utc": _utc_now(),
        "script": str(Path(__file__).resolve()),
        "sources": downloads,
        "normalized_catalog": normalized_manifest,
        "inputs": {
            "s82_catalog": str(Path(args.s82_catalog).resolve()),
            "s82_sha256": sha256_file(Path(args.s82_catalog)),
            "s82_source_rows": input_count,
            "s82_processed_rows": len(s82),
        },
        "outputs": {
            "primary": str(output.resolve()),
            "primary_sha256": sha256_file(output),
            "primary_rows": len(primary),
            "candidate_components": str(candidate_output.resolve()),
            "candidate_components_sha256": sha256_file(candidate_output),
            "candidate_rows": len(candidates),
            "core_matches": int(primary["first_core_matched"].sum()),
            "unique_object_ids": int(primary["objectId"].nunique()),
        },
        "settings": {
            "core_radius_arcsec": args.core_radius,
            "candidate_radius_arcsec": args.candidate_radius,
            "sidelobe_probability_max": args.sidelobe_threshold,
            "flux_limit_mjy": "5 * local_RMS + 0.25 (approximate unresolved-source limit)",
            "alpha_radio": args.alpha_radio,
            "alpha_optical": args.alpha_optical,
            "sdss_i_effective_wavelength_angstrom": args.i_wavelength,
            "observed_radio_loud_threshold_ri": 1.0,
            "rest_proxy_ratio_threshold": 10.0,
            "nearby_components_are_summed": False,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "astropy": astropy.__version__,
        },
    }
    _atomic_json(provenance_output, provenance)
    print(
        f"Wrote {len(primary):,} objects ({int(primary['first_core_matched'].sum()):,} core matches) "
        f"to {output}",
        flush=True,
    )
    print(f"Wrote {len(candidates):,} nearby components to {candidate_output}", flush=True)
    return output, candidate_output, provenance_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s82-catalog", default="data/S82/Catalog.parquet")
    parser.add_argument("--output", default="data/S82/first_radio_matches.parquet")
    parser.add_argument("--candidates-output", help="Default: <output stem>_candidates.parquet")
    parser.add_argument("--cache-dir", default="data/S82/.first_cache")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--core-radius", type=float, default=1.5, help="Arcsec (default: 1.5)")
    parser.add_argument("--candidate-radius", type=float, default=30.0, help="Arcsec (default: 30)")
    parser.add_argument("--sidelobe-threshold", type=float, default=0.1)
    parser.add_argument("--alpha-radio", type=float, default=-0.7)
    parser.add_argument("--alpha-optical", type=float, default=-0.5)
    parser.add_argument("--i-wavelength", type=float, default=7625.0, help="Angstrom")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument(
        "--max-objects",
        type=int,
        help="Process only the first N input rows (for a live smoke test)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.retries < 0 or args.retry_delay < 0 or args.timeout <= 0:
        parser.error("retry count/delay must be non-negative and timeout must be positive")
    if args.core_radius <= 0 or args.candidate_radius < args.core_radius:
        parser.error("radii must be positive and candidate radius must be at least core radius")
    if not (0 <= args.sidelobe_threshold <= 1):
        parser.error("sidelobe threshold must be between zero and one")
    if args.max_objects is not None and args.max_objects <= 0:
        parser.error("max-objects must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
