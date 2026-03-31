import os
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.light_curve.multiband_generate_lc import populate_sdss_fields


def _build_catalog_table():
    return Table(
        {
            "objectId": np.array(["obj_a", "obj_dup", "obj_dup"]),
            "RA": np.array([10.0, 20.0, 21.0]),
            "DEC": np.array([0.0, 1.0, 1.5]),
            "z": np.array([0.5, 1.2, 9.9]),
        }
    )


def _build_dr16q_table():
    return Table(
        {
            "RA": np.array([10.0 + 1.0e-4, 20.0 + 1.0e-4, 80.0]),
            "DEC": np.array([0.0, 1.0, 0.0]),
            "Z_SYS": np.array([0.5, 1.2, 3.0]),
            "Z_SYS_ERR": np.array([0.01, 0.02, 0.03]),
            "SDSS_NAME": np.array(["A", "B", "C"]),
            "EBV": np.array([0.03, 0.04, 0.05]),
            "LOGLBOL": np.array([45.1, 46.2, 47.3]),
            "LOGLBOL_ERR": np.array([0.1, 0.2, 0.3]),
            "LOGL3000": np.array([44.0, 45.0, 46.0]),
            "LOGL3000_ERR": np.array([0.4, 0.5, 0.6]),
            "LOGL5100": np.array([44.1, 45.2, 46.3]),
            "LOGL5100_ERR": np.array([0.01, 0.02, 0.03]),
            "LOGMBH": np.array([8.1, 8.2, 8.3]),
            "LOGMBH_ERR": np.array([0.11, 0.12, 0.13]),
            "LOGLEDD_RATIO": np.array([-0.5, -0.4, -0.3]),
            "LOGLEDD_RATIO_ERR": np.array([0.05, 0.04, 0.03]),
            "PLATE": np.array([1000, 1001, 1002]),
            "MJD": np.array([55000, 55001, 55002]),
            "FIBERID": np.array([10, 11, 12]),
        }
    )


def _mock_table_read(monkeypatch, cat, dr16q):
    calls = []

    monkeypatch.setattr(
        "qvc.light_curve.multiband_generate_lc.resolve_qvc_data_path",
        lambda path: str(path),
    )

    def fake_read(path, *args, **kwargs):
        path = str(path)
        calls.append((path, kwargs.copy()))
        if path.endswith("Catalog.parquet"):
            include_names = kwargs.get("include_names")
            if include_names is None:
                return cat.copy()
            missing = [name for name in include_names if name not in cat.colnames]
            if missing:
                raise KeyError(", ".join(missing))
            return cat[include_names].copy()
        if path.endswith("dr16q_prop_May01_2024.fits"):
            assert kwargs.get("hdu") == 1
            return dr16q.copy()
        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr("qvc.light_curve.multiband_generate_lc.Table.read", fake_read)
    return calls


def test_populate_sdss_fields_strict_success(monkeypatch):
    cat = _build_catalog_table()
    dr16q = _build_dr16q_table()
    calls = _mock_table_read(monkeypatch, cat, dr16q)

    objs = [
        {"object_id": "obj_a"},
        {"object_id": "obj_dup"},
    ]

    result = populate_sdss_fields(objs, progress_bar=True)

    assert result is objs
    assert result[0]["ra"] == 10.0
    assert result[0]["dec"] == 0.0
    assert result[0]["z"] == 0.5
    assert result[0]["z_err"] == 0.01
    assert result[0]["sdss_name"] == "A"
    assert result[0]["plate"] == 1000
    np.testing.assert_allclose(result[0]["LOGLBOL"], 45.1)
    np.testing.assert_allclose(result[0]["LOGLBOL_ERR"], 0.1)
    np.testing.assert_allclose(result[0]["LOGLBOL_CORRECTED"], np.log10(5.15) + 44.0)
    np.testing.assert_allclose(result[0]["LOGLBOL_CORRECTED_ERR"], 0.4)

    assert result[1]["ra"] == 20.0
    assert result[1]["dec"] == 1.0
    assert result[1]["z"] == 1.2
    assert result[1]["z_err"] == 0.02
    assert result[1]["sdss_name"] == "B"
    assert result[1]["LOGL5100"] == 45.2
    assert result[1]["fiberid"] == 11
    np.testing.assert_allclose(result[1]["LOGLBOL"], 46.2)
    np.testing.assert_allclose(result[1]["LOGLBOL_ERR"], 0.2)
    np.testing.assert_allclose(result[1]["LOGLBOL_CORRECTED"], 46.2)
    np.testing.assert_allclose(result[1]["LOGLBOL_CORRECTED_ERR"], 0.2)

    parquet_calls = [kwargs for path, kwargs in calls if path.endswith("Catalog.parquet")]
    assert len(parquet_calls) == 1
    assert parquet_calls[0]["include_names"] == ["objectId", "RA", "DEC", "z"]


def test_populate_sdss_fields_raises_on_missing_catalog_match(monkeypatch):
    cat = _build_catalog_table()
    dr16q = _build_dr16q_table()
    _mock_table_read(monkeypatch, cat, dr16q)

    with pytest.raises(ValueError, match=r"Missing S82 catalog match.*missing"):
        populate_sdss_fields([{"object_id": "obj_a"}, {"object_id": "missing"}], progress_bar=False)


def test_populate_sdss_fields_raises_on_missing_dr16q_match(monkeypatch):
    cat = _build_catalog_table()
    cat.add_row(("obj_far", 40.0, -5.0, 2.0))
    dr16q = _build_dr16q_table()
    _mock_table_read(monkeypatch, cat, dr16q)

    with pytest.raises(ValueError, match=r"Missing DR16Q match within 1 arcsec.*obj_far"):
        populate_sdss_fields([{"object_id": "obj_a"}, {"object_id": "obj_far"}], progress_bar=False)


def test_populate_sdss_fields_raises_on_missing_z_sys_err_column(monkeypatch):
    cat = _build_catalog_table()
    dr16q = _build_dr16q_table()
    dr16q.remove_column("Z_SYS_ERR")
    _mock_table_read(monkeypatch, cat, dr16q)

    with pytest.raises(ValueError, match=r"DR16Q table missing required column\(s\): Z_SYS_ERR"):
        populate_sdss_fields([{"object_id": "obj_a"}], progress_bar=False)


def test_populate_sdss_fields_raises_on_redshift_disagreement(monkeypatch):
    cat = _build_catalog_table()
    dr16q = _build_dr16q_table()
    dr16q["Z_SYS"][1] = 1.5
    _mock_table_read(monkeypatch, cat, dr16q)

    with pytest.raises(ValueError, match=r"Redshift disagreement exceeds tolerance .*obj_dup"):
        populate_sdss_fields([{"object_id": "obj_a"}, {"object_id": "obj_dup"}], progress_bar=False)


def test_populate_sdss_fields_raises_on_nonfinite_z_sys_err(monkeypatch):
    cat = _build_catalog_table()
    dr16q = _build_dr16q_table()
    dr16q["Z_SYS_ERR"][0] = np.nan
    _mock_table_read(monkeypatch, cat, dr16q)

    with pytest.raises(ValueError, match=r"Missing DR16Q Z_SYS_ERR.*obj_a"):
        populate_sdss_fields([{"object_id": "obj_a"}], progress_bar=False)


def test_populate_sdss_fields_falls_back_to_catalog_z_sys(monkeypatch):
    cat = _build_catalog_table()
    cat.rename_column("z", "Z_SYS")
    dr16q = _build_dr16q_table()
    calls = _mock_table_read(monkeypatch, cat, dr16q)

    result = populate_sdss_fields([{"object_id": "obj_a"}], progress_bar=False)

    assert result[0]["z"] == 0.5
    assert result[0]["z_err"] == 0.01

    parquet_calls = [kwargs for path, kwargs in calls if path.endswith("Catalog.parquet")]
    assert len(parquet_calls) == 2
    assert parquet_calls[0]["include_names"] == ["objectId", "RA", "DEC", "z"]
    assert parquet_calls[1]["include_names"] == ["objectId", "RA", "DEC", "Z_SYS"]
