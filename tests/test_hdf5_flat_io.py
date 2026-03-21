import sys
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble.hubble_utils import read_quasars_from_hdf5_flat
from qvc.light_curve import multiband_fit_utils as mfu


def _mock_quasars():
    return [
        {
            "object_id": "qso-a-1",
            "bands": ["u", "g", "r", "i", "z"],
            "z": 1.1,
            "mags_mean": [20.11, 19.91, 19.71, 19.61, 19.51],
            "mags_means": [20.1, 19.9, 19.7, 19.6, 19.5],
            "foo": [1.0, 2.0, 3.0, 4.0, 5.0],
            "other_vec": [10.0, 11.0],
            "nested": {
                "band_signal": [1.0, 2.0, 3.0, 4.0, 5.0],
                "generic": [7.0, 8.0],
            },
            "oversized": np.arange(2005),
        },
        {
            "object_id": "qso-2",
            "bands": ["g", "r", "i"],
            "z": 2.2,
            "mags_mean": [18.11, 17.91, 17.81, 17.71, 17.61],
            "mags_means": [18.0, 17.9, 17.8],
            "foo": [50.0, 51.0, 52.0, 53.0, 54.0],
            "other_vec": [99.0],
            "nested": {
                "band_signal": [9.0, 10.0, 11.0],
                "generic": [42.0],
            },
        },
    ]


def test_flat_hdf5_roundtrip_and_schema(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mfu.prefix = "flat_io_test"
    mfu.suffix = "roundtrip"

    quasars = _mock_quasars()
    mfu.save_quasar_list_hdf5(quasars, ignored_keys=["ignored_field"])

    out_path = tmp_path / "results" / "data" / "flat_io_test" / "roundtrip.h5"
    assert out_path.exists()

    with h5py.File(out_path, "r") as hdf:
        keys = set(hdf.keys())
        assert "object_id" in keys
        assert "mags_mean_u" in keys
        assert "mags_mean_z" in keys
        assert "mags_mean_0" not in keys
        assert "mags_means_u" in keys
        assert "mags_means_g" in keys
        assert "mags_means_z" in keys
        assert "foo_u" in keys
        assert "foo_z" in keys
        assert "foo_0" not in keys
        assert "other_vec_0" in keys
        assert "other_vec_1" in keys
        assert "nested_band_signal_u" in keys
        assert "nested_band_signal_z" in keys
        assert "nested_generic_0" in keys
        assert "nested_generic_1" in keys
        assert "oversized" not in keys
        assert hdf["object_id"].asstr()[0] == "qso-a-1"

    df = read_quasars_from_hdf5_flat(str(out_path))
    assert len(df) == 2

    r0 = df.iloc[0]
    r1 = df.iloc[1]
    assert r0["object_id"] == "qso-a-1"
    assert np.isclose(r0["mags_mean_u"], 20.11)
    assert np.isclose(r0["mags_mean_z"], 19.51)
    assert np.isclose(r0["mags_means_u"], 20.1)
    assert np.isclose(r0["mags_means_z"], 19.5)
    assert np.isclose(r0["foo_u"], 1.0)
    assert np.isclose(r0["foo_z"], 5.0)
    assert np.isclose(r0["other_vec_0"], 10.0)
    assert np.isclose(r0["other_vec_1"], 11.0)
    assert np.isclose(r0["nested_band_signal_r"], 3.0)
    assert np.isclose(r0["nested_generic_1"], 8.0)

    assert r1["object_id"] == "qso-2"
    assert np.isclose(r1["mags_mean_u"], 18.11)
    assert np.isclose(r1["mags_mean_z"], 17.61)
    assert np.isnan(r1["mags_means_u"])
    assert np.isnan(r1["mags_means_z"])
    assert np.isclose(r1["mags_means_g"], 18.0)
    assert np.isclose(r1["foo_u"], 50.0)
    assert np.isclose(r1["foo_z"], 54.0)
    assert np.isclose(r1["other_vec_0"], 99.0)
    assert np.isnan(r1["other_vec_1"])
    assert np.isnan(r1["nested_band_signal_u"])
    assert np.isnan(r1["nested_band_signal_z"])


def test_read_quasars_from_hdf5_flat_respects_n_limit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mfu.prefix = "flat_io_test"
    mfu.suffix = "n_limit"

    mfu.save_quasar_list_hdf5(_mock_quasars())
    out_path = tmp_path / "results" / "data" / "flat_io_test" / "n_limit.h5"

    df = read_quasars_from_hdf5_flat(str(out_path), N=1)
    assert len(df) == 1
    assert df.iloc[0]["object_id"] == "qso-a-1"
