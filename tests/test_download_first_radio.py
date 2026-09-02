from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from scripts import download_first_radio as first_radio


class FakeResponse(io.BytesIO):
    def __init__(self, body=b"", *, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class FakeHTTP:
    def __init__(self, payload: bytes, *, ignore_range=False, truncate_once=False, etag='"fixture"'):
        self.payload = payload
        self.ignore_range = ignore_range
        self.truncate_once = truncate_once
        self.etag = etag
        self.get_calls = 0
        self.ranges = []

    def __call__(self, request, timeout):
        if request.get_method() == "HEAD":
            return FakeResponse(
                headers={
                    "Content-Length": str(len(self.payload)),
                    "ETag": self.etag,
                    "Accept-Ranges": "bytes",
                }
            )
        self.get_calls += 1
        range_header = request.headers.get("Range")
        self.ranges.append(range_header)
        offset = int(range_header.split("=")[1].split("-")[0]) if range_header else 0
        if self.ignore_range:
            offset, status = 0, 200
        else:
            status = 206 if range_header else 200
        body = self.payload[offset:]
        if self.truncate_once and self.get_calls == 1:
            body = body[: max(1, len(body) // 2)]
        return FakeResponse(body, status=status)


def test_download_resumes_part_and_atomically_replaces(tmp_path):
    payload = b"abcdefghijklmnopqrstuvwxyz"
    destination = tmp_path / "catalog.bin"
    destination.with_name(destination.name + ".part").write_bytes(payload[:8])
    destination.with_name(destination.name + ".part.http.json").write_text(
        json.dumps(
            {
                "url": "https://example.test/catalog",
                "etag": '"fixture"',
                "last_modified": None,
                "content_length": len(payload),
            }
        )
    )
    http = FakeHTTP(payload)

    metadata = first_radio.download_with_resume(
        "https://example.test/catalog",
        destination,
        opener=http,
        retry_delay=0,
        sleep=lambda _: None,
    )

    assert destination.read_bytes() == payload
    assert http.ranges == ["bytes=8-"]
    assert not destination.with_name(destination.name + ".part").exists()
    assert metadata["reused"] is False
    assert json.loads(destination.with_name(destination.name + ".http.json").read_text())["etag"] == '"fixture"'


def test_download_restarts_when_server_ignores_range(tmp_path):
    payload = b"complete file"
    destination = tmp_path / "catalog.bin"
    destination.with_name(destination.name + ".part").write_bytes(b"old")
    destination.with_name(destination.name + ".part.http.json").write_text(
        json.dumps(
            {
                "url": "https://example.test/catalog",
                "etag": '"fixture"',
                "last_modified": None,
                "content_length": len(payload),
            }
        )
    )
    http = FakeHTTP(payload, ignore_range=True)

    first_radio.download_with_resume(
        "https://example.test/catalog", destination, opener=http, sleep=lambda _: None
    )

    assert destination.read_bytes() == payload
    assert http.ranges == ["bytes=3-"]


def test_download_retries_truncation_then_resumes_and_reuses_cache(tmp_path):
    payload = b"a somewhat longer fixture"
    destination = tmp_path / "catalog.bin"
    http = FakeHTTP(payload, truncate_once=True)
    delays = []

    first_radio.download_with_resume(
        "https://example.test/catalog",
        destination,
        opener=http,
        retry_delay=0.25,
        sleep=delays.append,
    )
    assert destination.read_bytes() == payload
    assert http.get_calls == 2
    assert http.ranges[1] == f"bytes={len(payload) // 2}-"
    assert delays == [0.25]

    reused = first_radio.download_with_resume(
        "https://example.test/catalog", destination, opener=http, validator=lambda p: p.read_bytes()
    )
    assert reused["reused"] is True
    assert http.get_calls == 2


def test_failed_replacement_preserves_previous_atomic_file(tmp_path):
    destination = tmp_path / "catalog.bin"
    destination.write_bytes(b"previous valid catalog")
    destination.with_name(destination.name + ".http.json").write_text(
        json.dumps(
            {
                "url": "https://example.test/catalog",
                "etag": '"old"',
                "content_length": destination.stat().st_size,
            }
        )
    )
    http = FakeHTTP(b"new catalog contents", truncate_once=True, etag='"new"')

    with pytest.raises(RuntimeError, match="Failed to download"):
        first_radio.download_with_resume(
            "https://example.test/catalog",
            destination,
            opener=http,
            retries=0,
            sleep=lambda _: None,
        )

    assert destination.read_bytes() == b"previous valid catalog"


def _first_fixture(path: Path) -> None:
    columns = []
    values = {
        "RA": np.array([10.0, 11.0]),
        "DEC": np.array([0.0, 1.0]),
        "SIDEPROB": np.array([0.01, 0.2], dtype=np.float32),
        "FPEAK": np.array([2.0, 3.0], dtype=np.float32),
        "FINT": np.array([2.2, 3.2], dtype=np.float32),
        "RMS": np.array([0.15, 0.16], dtype=np.float32),
        "MAJOR": np.array([1.0, 2.0], dtype=np.float32),
        "MINOR": np.array([0.5, 1.0], dtype=np.float32),
        "POSANG": np.array([30.0, 40.0], dtype=np.float32),
        "FITTED_MAJOR": np.array([5.5, 6.0], dtype=np.float32),
        "FITTED_MINOR": np.array([5.0, 5.5], dtype=np.float32),
        "FITTED_POSANG": np.array([31.0, 41.0], dtype=np.float32),
        "FLDNAME": np.array(["field-a", "field-b"]),
        "YEAR": np.array([2001.0, 2002.0], dtype=np.float32),
        "MJD": np.array([2450001.0, 2450002.0]),
        "MJDRMS": np.array([0.1, 0.2]),
    }
    for name, value in values.items():
        if value.dtype.kind == "U":
            columns.append(fits.Column(name=name, format="12A", array=value))
        elif value.dtype.itemsize == 8:
            columns.append(fits.Column(name=name, format="D", array=value))
        else:
            columns.append(fits.Column(name=name, format="E", array=value))
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path)


def _rms_fixture(path: Path, *, crval_ra=0.0, value=0.2) -> None:
    image = np.zeros((5, 5), dtype=np.float32)
    image[2, 2] = value
    header = fits.Header(
        {
            "CTYPE1": "RA",
            "CTYPE2": "DEC",
            "CRVAL1": crval_ra,
            "CRVAL2": 0.0,
            "CRPIX1": 3.0,
            "CRPIX2": 3.0,
            "CDELT1": 1.0,
            "CDELT2": 1.0,
        }
    )
    fits.PrimaryHDU(image, header).writeto(path)


def test_fits_normalization_cache_and_rms_wcs_ra_wrap(tmp_path):
    source = tmp_path / "first.fits"
    normalized = tmp_path / "first.parquet"
    _first_fixture(source)

    frame, manifest = first_radio.normalize_first_catalog(source, normalized)
    assert list(frame["first_catalog_row"]) == [0, 1]
    assert frame.loc[0, "first_integrated_flux_mjy"] == pytest.approx(2.2)
    assert manifest["row_count"] == 2
    cached, _ = first_radio.normalize_first_catalog(source, normalized)
    assert len(cached) == 2

    rms_path = tmp_path / "rms.fits"
    _rms_fixture(rms_path, crval_ra=-1.0, value=0.23)
    # 359 degrees and -1 degree are the same location.
    rms = first_radio.lookup_rms_map(rms_path, np.array([359.0, 100.0]), np.array([0.0, 0.0]))
    assert rms[0] == pytest.approx(0.23)
    assert np.isnan(rms[1])


def _normalized_first(rows):
    defaults = {
        "first_sidelobe_probability": 0.01,
        "first_peak_flux_mjy": 2.0,
        "first_integrated_flux_mjy": 2.5,
        "first_catalog_rms_mjy": 0.15,
        "first_major_arcsec": 2.0,
        "first_minor_arcsec": 1.0,
        "first_pa_deg": 20.0,
        "first_fitted_major_arcsec": 6.0,
        "first_fitted_minor_arcsec": 5.0,
        "first_fitted_pa_deg": 21.0,
        "first_field": "field",
        "first_epoch_year": 2003.0,
        "first_epoch_jd": 2450000.0,
        "first_epoch_rms_days": 0.1,
    }
    records = []
    for index, row in enumerate(rows):
        record = {"first_catalog_row": index, **defaults, **row}
        records.append(record)
    return pd.DataFrame(records)


def test_match_selection_rejection_candidates_and_coverage_states():
    s82 = pd.DataFrame(
        {
            "objectId": ["a", "b", "c"],
            "RA": [10.0, 20.0, 30.0],
            "DEC": [0.0, 0.0, 0.0],
            "Z_DR16Q": [1.0, 1.0, 1.0],
            "sdss_i_qg": [19.0, 19.0, 19.0],
            "qg_fragal": [0.1, 0.2, 0.3],
            "ebv": [0.01, 0.02, 0.03],
        }
    )
    first = _normalized_first(
        [
            # Nearest is rejected; the second reliable component is selected.
            {"first_ra_deg": 10.0 + 0.2 / 3600, "first_dec_deg": 0.0, "first_sidelobe_probability": 0.5},
            {"first_ra_deg": 10.0 + 0.8 / 3600, "first_dec_deg": 0.0},
            # A second reliable nearby component produces the extended flag.
            {"first_ra_deg": 10.0 + 10.0 / 3600, "first_dec_deg": 0.0},
            # Candidate for b, but outside the core radius.
            {"first_ra_deg": 20.0 + 5.0 / 3600, "first_dec_deg": 0.0},
        ]
    )
    coverage = pd.DataFrame(
        {
            "first_coverage_region": ["north", "north", "outside"],
            "first_covered": [True, True, False],
            "first_local_rms_mjy": [0.15, 0.15, np.nan],
            "first_flux_limit_5sigma_mjy_approx": [1.0, 1.0, np.nan],
        }
    )

    primary, candidates = first_radio.build_matches(s82, first, coverage)

    assert len(primary) == len(s82)
    assert primary["objectId"].is_unique
    assert primary.loc[0, "first_match_separation_arcsec"] == pytest.approx(0.8, rel=1e-4)
    assert primary.loc[0, "first_candidate_count_30arcsec"] == 3
    assert primary.loc[0, "first_reliable_candidate_count_30arcsec"] == 2
    assert primary.loc[0, "first_possible_extended"]
    assert candidates.query("objectId == 'a'")["first_reliable"].tolist() == [False, True, True]
    assert candidates["first_core_selected"].sum() == 1
    assert primary.loc[1, "first_match_state"] == "covered_no_core_match"
    assert primary.loc[2, "first_match_state"] == "outside_coverage_no_match"


def test_observed_and_rest_loudness_formulas_and_states():
    # i=20 has Fnu=0.03631 mJy.  A 0.3631 mJy radio detection gives Ri=1.
    base = pd.DataFrame(
        {
            "sdss_i_qg": [20.0, 20.0, 20.0, np.nan, 20.0],
            "Z_DR16Q": [1.0, 1.0, 1.0, 1.0, np.nan],
            "first_core_matched": [True, False, False, False, True],
            "first_covered": [True, True, True, True, True],
            "first_integrated_flux_mjy": [0.3631, np.nan, np.nan, np.nan, 1.0],
            "first_flux_limit_5sigma_mjy_approx": [1.0, 0.1, 1.0, 0.1, 1.0],
        }
    )
    result = first_radio.add_radio_loudness(base)

    assert result.loc[0, "first_radio_ab_mag"] == pytest.approx(17.5)
    assert result.loc[0, "radio_loudness_ri"] == pytest.approx(1.0)
    assert result.loc[0, "radio_loudness_ri_state"] == "detected_not_radio_loud"
    assert result.loc[1, "radio_loudness_ri_state"] == "constrained_quiet"
    assert result.loc[2, "radio_loudness_ri_state"] == "limit_ambiguous"
    assert result.loc[3, "radio_loudness_ri_state"] == "missing_optical"
    expected = 1.0 + (-0.7) * np.log10(5.0 / 2.8) - (-0.5) * np.log10(7625.0 / 8800.0)
    assert result.loc[0, "radio_loudness_log10_l5ghz_l4400"] == pytest.approx(expected)
    assert result.loc[4, "radio_loudness_rest_state"] == "missing_redshift"


def test_run_is_idempotent_and_preserves_input_count(tmp_path, monkeypatch):
    source_catalog = tmp_path / "source_first.fits"
    north = tmp_path / "source_north.fits"
    south = tmp_path / "source_south.fits"
    _first_fixture(source_catalog)
    _rms_fixture(north, crval_ra=10.0, value=0.2)
    _rms_fixture(south, crval_ra=100.0, value=0.3)
    fixtures = {
        first_radio.FIRST_CATALOG_URL: source_catalog,
        first_radio.FIRST_NORTH_RMS_URL: north,
        first_radio.FIRST_SOUTH_RMS_URL: south,
    }

    def fake_download(url, destination, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        reused = destination.exists()
        if not reused:
            shutil.copyfile(fixtures[url], destination)
        return {
            "url": url,
            "etag": '"test"',
            "content_length": destination.stat().st_size,
            "path": str(destination),
            "reused": reused,
            "sha256": first_radio.sha256_file(destination),
        }

    monkeypatch.setattr(first_radio, "download_with_resume", fake_download)
    s82_path = tmp_path / "s82.parquet"
    pd.DataFrame(
        {
            "objectId": ["x", "y"],
            "RA": [10.0, 11.0],
            "DEC": [0.0, 1.0],
            "Z_DR16Q": [1.0, 2.0],
            "sdss_i_qg": [19.0, 20.0],
            "qg_fragal": [0.1, 0.2],
            "ebv": [0.01, 0.02],
        }
    ).to_parquet(s82_path, index=False)
    output = tmp_path / "matches.parquet"
    args = first_radio.build_parser().parse_args(
        [
            "--s82-catalog",
            str(s82_path),
            "--output",
            str(output),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    first_radio.run(args)
    first_hash = first_radio.sha256_file(output)
    first_radio.run(args)
    result = pd.read_parquet(output)

    assert len(result) == 2
    assert result["objectId"].nunique() == 2
    assert first_radio.sha256_file(output) == first_hash
    provenance = json.loads((tmp_path / "matches.provenance.json").read_text())
    assert provenance["outputs"]["primary_rows"] == 2
    assert provenance["outputs"]["unique_object_ids"] == 2
