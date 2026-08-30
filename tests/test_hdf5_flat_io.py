import sys
from pathlib import Path
import re
import os

import h5py
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble.hubble_utils import read_quasars_from_hdf5_flat
from qvc.light_curve import multiband_fit_utils as mfu
from qvc.light_curve.posterior_draws import (
    LIGHT_CURVE_POSTERIOR_DRAW_GROUP,
    LIGHT_CURVE_POSTERIOR_DRAW_PAYLOAD_KEY,
    compact_log_sigma_tau_posterior_draws,
)


def _write_sample_hdf5(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as hdf:
        hdf.create_dataset("theta", data=np.asarray([value], dtype=float))


def _mock_quasars():
    return [
        {
            "object_id": "qso-a-1",
            "bands": ["u", "g", "r", "i", "z"],
            "z": 1.1,
            "psf_constant_flux_corrected": False,
            "psf_constant_flux_n_bands_corrected": 0,
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
            "psf_constant_flux_corrected": True,
            "psf_constant_flux_n_bands_corrected": 2,
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


def test_load_obj_samples_from_hdf5_ignores_current_suffix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mfu.prefix = "sample_resume"
    mfu.suffix = "job1"

    sample_path = tmp_path / "results" / "samples" / "sample_resume" / "1424946_job21413.h5"
    _write_sample_hdf5(sample_path, 21413.0)

    samples = mfu.load_obj_samples_from_hdf5("1424946")

    assert np.allclose(samples["theta"], [21413.0])


def test_load_obj_samples_from_hdf5_uses_newest_object_id_match(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mfu.prefix = "sample_resume_newest"
    mfu.suffix = "job1"

    sample_dir = tmp_path / "results" / "samples" / "sample_resume_newest"
    old_path = sample_dir / "1424946_job1.h5"
    new_path = sample_dir / "1424946_job21413.h5"
    _write_sample_hdf5(old_path, 1.0)
    _write_sample_hdf5(new_path, 21413.0)
    os.utime(old_path, (1000, 1000))
    os.utime(new_path, (2000, 2000))

    samples = mfu.load_obj_samples_from_hdf5("1424946")

    assert np.allclose(samples["theta"], [21413.0])


def test_load_obj_samples_from_hdf5_missing_object_id_match(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mfu.prefix = "sample_resume_missing"
    mfu.suffix = "job1"

    sample_path = tmp_path / "results" / "samples" / "sample_resume_missing" / "14249460_job1.h5"
    _write_sample_hdf5(sample_path, 10.0)

    with pytest.raises(FileNotFoundError, match="object_id='1424946'"):
        mfu.load_obj_samples_from_hdf5("1424946")


def test_load_obj_samples_from_hdf5_explicit_file_path_is_exact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mfu.prefix = "sample_resume_exact"
    mfu.suffix = "job1"

    exact_path = tmp_path / "custom" / "not_the_object_suffix.h5"
    _write_sample_hdf5(exact_path, 99.0)

    samples = mfu.load_obj_samples_from_hdf5("1424946", file_path=str(exact_path))

    assert np.allclose(samples["theta"], [99.0])


def test_flat_hdf5_roundtrip_and_schema(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mfu.prefix = "flat_io_test"
    mfu.suffix = "roundtrip"

    quasars = _mock_quasars()
    mfu.save_quasar_list_hdf5(quasars, ignored_keys=["ignored_field"])

    out_files = sorted((tmp_path / "results" / "data" / "flat_io_test").glob("*.h5"))
    assert len(out_files) == 1
    out_path = out_files[0]
    assert re.fullmatch(r"\d{8}T\d{12}_[0-9a-f]{8}\.h5", out_path.name)
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
        assert "psf_constant_flux_corrected" in keys
        assert "psf_constant_flux_n_bands_corrected" in keys
        assert "oversized" not in keys
        assert hdf["object_id"].asstr()[0] == "qso-a-1"
        assert hdf["psf_constant_flux_corrected"].dtype == np.dtype(bool)
        assert hdf["psf_constant_flux_corrected"][0] == np.bool_(False)
        assert hdf["psf_constant_flux_corrected"][1] == np.bool_(True)
        assert hdf["psf_constant_flux_n_bands_corrected"][0] == 0
        assert hdf["psf_constant_flux_n_bands_corrected"][1] == 2

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
    assert bool(r0["psf_constant_flux_corrected"]) is False
    assert r0["psf_constant_flux_n_bands_corrected"] == 0

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
    assert bool(r1["psf_constant_flux_corrected"]) is True
    assert r1["psf_constant_flux_n_bands_corrected"] == 2


def test_read_quasars_from_hdf5_flat_respects_n_limit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mfu.prefix = "flat_io_test"
    mfu.suffix = "n_limit"

    mfu.save_quasar_list_hdf5(_mock_quasars())
    out_files = sorted((tmp_path / "results" / "data" / "flat_io_test").glob("*.h5"))
    assert len(out_files) == 1
    out_path = out_files[0]

    df = read_quasars_from_hdf5_flat(str(out_path), N=1)
    assert len(df) == 1
    assert df.iloc[0]["object_id"] == "qso-a-1"


def test_save_quasar_list_hdf5_uses_object_id_filename_for_single_object(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mfu.prefix = "flat_io_single"
    mfu.suffix = "job0"

    quasar = [_mock_quasars()[0]]
    mfu.save_quasar_list_hdf5(quasar)

    out_path = tmp_path / "results" / "data" / "flat_io_single" / "qso-a-1.h5"
    assert out_path.exists()


def test_save_quasar_list_hdf5_persists_total_light_curve_fit_runtime(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mfu.prefix = "flat_io_runtime"
    mfu.suffix = "job0"

    quasar = [_mock_quasars()[0]]
    quasar[0]["light_curve_fit_total_elapsed_sec"] = 123.5
    mfu.save_quasar_list_hdf5(quasar)

    out_path = tmp_path / "results" / "data" / "flat_io_runtime" / "qso-a-1.h5"
    with h5py.File(out_path, "r") as hdf:
        np.testing.assert_allclose(
            hdf["light_curve_fit_total_elapsed_sec"][:],
            np.array([123.5]),
        )


def test_save_quasar_list_hdf5_embeds_compact_sigma_tau_draws(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    mfu.prefix = "flat_io_draws"
    mfu.suffix = "job0"
    quasar = _mock_quasars()[0]
    quasar[LIGHT_CURVE_POSTERIOR_DRAW_PAYLOAD_KEY] = (
        compact_log_sigma_tau_posterior_draws(
            np.linspace(-2.0, 0.0, 100),
            np.linspace(3.0, 5.0, 100),
            redshift=quasar["z"],
            object_id=quasar["object_id"],
        )
    )

    mfu.save_quasar_list_hdf5([quasar])

    output = (
        tmp_path
        / "results"
        / "data"
        / "flat_io_draws"
        / "qso-a-1.h5"
    )
    with h5py.File(output, "r") as handle:
        group = handle[LIGHT_CURVE_POSTERIOR_DRAW_GROUP]
        assert group["log_sigma_uv"].shape == (1, 64)
        assert group["log_tau_uv_rf"].shape == (1, 64)
        assert group["valid_count"][0] == 64
        assert LIGHT_CURVE_POSTERIOR_DRAW_PAYLOAD_KEY not in handle

    frame = read_quasars_from_hdf5_flat(output)
    assert len(frame) == 1
    assert frame.iloc[0]["object_id"] == "qso-a-1"


def test_save_quasar_list_hdf5_uses_time_tied_random_filename_for_multiple_objects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mfu.prefix = "flat_io_multi"
    mfu.suffix = "job0"

    quasars = _mock_quasars()
    mfu.save_quasar_list_hdf5(quasars)
    first_files = sorted((tmp_path / "results" / "data" / "flat_io_multi").glob("*.h5"))
    assert len(first_files) == 1
    first_name = first_files[0].name
    assert first_name != "job0.h5"
    assert first_name != "qso-a-1.h5"
    assert re.fullmatch(r"\d{8}T\d{12}_[0-9a-f]{8}\.h5", first_name)

    mfu.save_quasar_list_hdf5(quasars)
    second_files = sorted((tmp_path / "results" / "data" / "flat_io_multi").glob("*.h5"))
    assert len(second_files) == 2
    second_name = second_files[-1].name
    assert second_name != first_name
    assert re.fullmatch(r"\d{8}T\d{12}_[0-9a-f]{8}\.h5", second_name)


def test_read_quasars_from_hdf5_flat_normalizes_endian_for_updates(tmp_path):
    out_path = tmp_path / "be_numeric.h5"
    with h5py.File(out_path, "w") as hdf:
        hdf.create_dataset("object_id", data=np.array([b"a", b"b"]))
        hdf.create_dataset("z", data=np.array([1.1, 2.2], dtype=">f8"))
        draw_group = hdf.create_group("light_curve_posterior_draws")
        draw_group.create_dataset("log_sigma_uv", data=np.ones((2, 64)))

    df = read_quasars_from_hdf5_flat(str(out_path))
    assert len(df) == 2

    update_series = pd.Series([3.3], index=[0])
    out = df.copy()
    out["z"] = update_series.combine_first(out["z"])
    assert np.isclose(out.loc[0, "z"], 3.3)
    assert np.isclose(out.loc[1, "z"], 2.2)


def test_populate_style_update_handles_big_endian_columns():
    df = pd.DataFrame(
        {
            "object_id": ["q1", "q2", "q3"],
            "z": np.array([0.1, 0.2, 0.3], dtype=">f8"),
            "keep": [1, 2, 3],
        }
    )

    matched_row_idx = np.array([0, 2], dtype=int)
    target_index = df.index.to_numpy()[matched_row_idx]
    update_fields = {
        "z": np.array([1.5, 3.5]),
        "log_lbol": np.array([45.1, 46.2]),
    }
    update_df = pd.DataFrame(update_fields, index=target_index)

    existing_cols = [col for col in update_df.columns if col in df.columns]
    if existing_cols:
        df.update(update_df.loc[:, existing_cols], overwrite=True)

    new_cols = [col for col in update_df.columns if col not in df.columns]
    for col in new_cols:
        df[col] = np.nan
        df.update(update_df.loc[:, [col]], overwrite=True)

    assert np.isclose(df.loc[0, "z"], 1.5)
    assert np.isclose(df.loc[1, "z"], 0.2)
    assert np.isclose(df.loc[2, "z"], 3.5)
    assert np.isclose(df.loc[0, "log_lbol"], 45.1)
    assert np.isnan(df.loc[1, "log_lbol"])
    assert np.isclose(df.loc[2, "log_lbol"], 46.2)
