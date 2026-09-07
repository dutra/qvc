import json

import numpy as np
import pandas as pd
import pytest

from qvc.light_curve import seeing_enrichment as seeing


def test_ps1_bulk_query_uses_tap_and_computes_geometric_fwhm(monkeypatch):
    captured = {}

    def fake_read(url, params, **kwargs):
        captured.update(url=url, params=params, kwargs=kwargs)
        return pd.DataFrame(
            {
                "detectid": [101, 102],
                "objid": [11, 12],
                "psfmajorfwhm": [4.0, 9.0],
                "psfminorfwhm": [1.0, 1.0],
            }
        )

    monkeypatch.setattr(seeing, "_read_csv_url", fake_read)
    result = seeing.query_ps1_detection_seeing_bulk([11, 12, 11])

    assert captured["url"] == seeing.PS1_TAP_SYNC_URL
    assert "method" not in captured["kwargs"]
    assert "objid IN (11,12)" in captured["params"]["QUERY"]
    assert result["detectID"].tolist() == [101, 102]
    np.testing.assert_allclose(result["psf_fwhm_arcsec"], [2.0, 3.0])


def test_ps1_bulk_query_rejects_possible_truncation(monkeypatch):
    monkeypatch.setattr(
        seeing,
        "_read_csv_url",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "detectid": [1],
                "objid": [2],
                "psfmajorfwhm": [1.0],
                "psfminorfwhm": [1.0],
            }
        ),
    )
    with pytest.raises(RuntimeError, match="MAXREC"):
        seeing.query_ps1_detection_seeing_bulk([2], max_records=1)


def test_ztf_field_query_uses_field_and_julian_date(monkeypatch):
    captured = {}

    def fake_read(url, params, **kwargs):
        captured.update(url=url, params=params)
        return pd.DataFrame(
            {
                "field": [441],
                "rcid": [41],
                "fid": [1],
                "obsjd": [2459480.796],
                "seeing": [2.1],
            }
        )

    monkeypatch.setattr(seeing, "_read_csv_url", fake_read)
    result = seeing.query_ztf_field_seeing(441, 59480.0, 59481.0)

    assert captured["url"] == seeing.ZTF_IBE_URL
    assert "field=441" in captured["params"]["WHERE"]
    assert "2459480.50000000" in captured["params"]["WHERE"]
    assert result.loc[0, "fieldid"] == 441
    assert result.loc[0, "mjd"] == pytest.approx(59480.296)


def test_sdss_query_requests_footprints_band_times_and_seeing(monkeypatch):
    captured = {}
    row = {
        "run": 1,
        "rerun": 301,
        "camcol": 2,
        "field": 3,
        "raMin": "359.9",
        "raMax": "0.1",
        "decMin": "-1.0",
        "decMax": "1.0",
    }
    row.update({f"mjd_{band}": "52200.1" for band in seeing.SDSS_BANDS})
    row.update({f"psfWidth_{band}": "1.2" for band in seeing.SDSS_BANDS})

    def fake_read(url, params, **kwargs):
        captured.update(url=url, params=params, kwargs=kwargs)
        return pd.DataFrame([row])

    monkeypatch.setattr(seeing, "_read_csv_url", fake_read)
    result = seeing.query_sdss_field_metadata(52199, 52201, -1.5, 1.5)

    sql = captured["params"]["cmd"]
    assert captured["url"] == seeing.SDSS_SQL_URL
    assert "/stripe82/" in captured["url"]
    assert "f.raMin" in sql and "f.mjd_u" in sql and "f.psfWidth_z" in sql
    assert "f.mjd_r BETWEEN 52199.00000000 AND 52201.00000000" in sql
    assert result.loc[0, "psfWidth_r"] == pytest.approx(1.2)


def test_nearest_ztf_seeing_is_grouped_and_tolerance_limited():
    raw = pd.DataFrame(
        {
            "fieldid": [1, 1, 1],
            "rcidin": [2, 2, 3],
            "filterID": [1, 1, 1],
            "mjd": [10.001, 11.0, 10.001],
        }
    )
    archive = pd.DataFrame(
        {
            "fieldid": [1, 1],
            "rcidin": [2, 3],
            "filterID": [1, 1],
            "mjd": [10.0, 10.0],
            "psf_fwhm_arcsec": [1.5, 2.5],
        }
    )
    result = seeing._nearest_ztf_seeing(raw, archive, max_delta_days=0.01)
    np.testing.assert_allclose(result[[0, 2]], [1.5, 2.5])
    assert np.isnan(result[1])


def test_catalog_filter_preserves_large_integer_identifier_precision():
    identifier = 107450000019989160
    catalog = pd.DataFrame({"ps1objID": [str(identifier)]})
    raw = pd.DataFrame({"ps1objID": np.array([identifier, identifier + 1], dtype=np.int64)})
    result = seeing._filter_raw_to_catalog(raw, catalog, "ps1objID")
    assert result["ps1objID"].tolist() == [identifier]


def _sdss_field(**updates):
    row = {
        "run": 1,
        "rerun": 301,
        "camcol": 1,
        "field": 1,
        "raMin": 359.8,
        "raMax": 0.2,
        "decMin": -0.2,
        "decMax": 0.2,
    }
    row.update({f"mjd_{band}": 100.01 for band in seeing.SDSS_BANDS})
    row.update({f"psfWidth_{band}": 1.0 + i for i, band in enumerate(seeing.SDSS_BANDS)})
    row.update(updates)
    return row


def test_sdss_local_match_handles_ra_wrap_band_and_tolerance():
    raw = pd.DataFrame(
        {
            "objectId": [1, 1, 2],
            "mjd": [100.011, 100.011, 100.5],
            "filterID": [0, 3, 0],
            "RA": [359.95, 0.05, 359.95],
            "DEC": [0.0, 0.0, 0.0],
        }
    )
    result = seeing._match_sdss_field_seeing(
        raw, pd.DataFrame([_sdss_field()]), max_delta_days=0.08
    )
    np.testing.assert_allclose(result[:2], [1.0, 4.0])
    assert np.isnan(result[2])


def test_cached_query_resumes_without_calling_network(tmp_path):
    path = tmp_path / "cache.parquet"
    expected = pd.DataFrame({"value": [1, 2]})
    first = seeing._cached_query(path, lambda: expected, resume=True)
    second = seeing._cached_query(
        path,
        lambda: (_ for _ in ()).throw(AssertionError("network called")),
        resume=True,
    )
    pd.testing.assert_frame_equal(first, second)


def test_sdss_cache_adaptively_splits_response_buffer_overflow(monkeypatch, tmp_path):
    calls = []

    def fake_query(start, end, dec_min, dec_max):
        calls.append((start, end))
        if end - start > 2:
            raise RuntimeError("Response Buffer Limit Exceeded")
        return pd.DataFrame({"start": [start], "end": [end]})

    monkeypatch.setattr(seeing, "query_sdss_field_metadata", fake_query)
    result = seeing._fetch_sdss_window_cached(
        tmp_path, 0, 4, -1, 1, resume=True
    )

    assert calls == [(0, 4), (0, 2), (2, 4)]
    assert result["start"].tolist() == [0, 2]
    # Successful child partitions are reused on a second attempt.
    calls.clear()
    seeing._fetch_sdss_window_cached(tmp_path, 0, 4, -1, 1, resume=True)
    assert calls == []


def test_write_sidecar_deduplicates_and_reports_filter_coverage(tmp_path):
    raw = pd.DataFrame(
        {"detectID": [1, 2], "ps1objID": [10, 10], "filterID": [1, 2]}
    )
    sidecar = pd.DataFrame(
        {
            "detectID": [1, 1, 2],
            "psf_fwhm_arcsec": [1.0, 1.1, np.nan],
        }
    )
    summary = seeing._write_sidecar("ps1", sidecar, raw, output_dir=tmp_path)
    written = pd.read_parquet(tmp_path / seeing.SIDECAR_FILENAMES["ps1"])

    assert len(written) == 2
    assert summary["matched"] == 1
    coverage = pd.read_parquet(summary["coverage_path"])
    assert coverage[["matched", "total"]].values.tolist() == [[1, 1], [0, 1]]
    assert summary["coverage_by_filter"]["1"]["matched"] == 1
    assert summary["coverage_by_filter"]["2"]["matched"] == 0


def test_enrich_all_dispatches_selected_surveys_and_writes_manifest(monkeypatch, tmp_path):
    catalog = pd.DataFrame(
        {"objectId": [1], "ps1objID": [10], "RA": [0.0], "DEC": [0.0]}
    )
    monkeypatch.setattr(seeing, "_load_catalog", lambda **kwargs: catalog)
    calls = []

    def fake_ps1(actual_catalog, **kwargs):
        calls.append(("ps1", actual_catalog, kwargs))
        return {"matched": 1, "total": 1, "fraction": 1.0, "path": "ps1"}

    monkeypatch.setattr(seeing, "enrich_ps1_seeing", fake_ps1)
    summary = seeing.enrich_all_seeing(
        output_dir=tmp_path,
        surveys=("ps1",),
        progress_bar=False,
    )

    assert list(summary) == ["ps1"]
    assert calls[0][0] == "ps1"
    manifest = json.loads((tmp_path / "seeing_enrichment_manifest.json").read_text())
    assert manifest["selected_objects"] == 1
    assert manifest["surveys"] == ["ps1"]
    assert manifest["cache"]["partitions"] == []


def test_time_windows_overlap_only_by_requested_padding():
    assert seeing._time_windows(10.2, 12.2, 2, padding=0.1) == [
        (9.9, 12.1),
        (11.9, 14.1),
    ]
