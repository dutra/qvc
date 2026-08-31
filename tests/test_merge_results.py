import h5py
import numpy as np
import pandas as pd
import pytest

from qvc.light_curve import merge_results


def _write_light_curve_samples(path, log_sigma_uv, log_tau_uv):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("log_sigma_uv", data=np.asarray(log_sigma_uv))
        handle.create_dataset("log_tau_uv", data=np.asarray(log_tau_uv))


def _create_sample_dir(directory, filename, log_sigma_uv, log_tau_uv):
    _write_light_curve_samples(
        directory / filename,
        log_sigma_uv,
        log_tau_uv,
    )
    return directory


def _identity_row(**overrides):
    row = {
        "object_id": "123",
        "ra": 10.0,
        "dec": -2.0,
        "large_unrelated_posterior_field": object(),
    }
    for band in "ugri":
        row[f"log_sigma_band_{band}"] = -0.5
        row[f"log_sigma_band_{band}_err"] = 0.1
        row[f"log_tau_band_{band}_RF"] = 2.5
        row[f"log_tau_band_{band}_RF_err"] = 0.2
    row.update(overrides)
    return row


def test_identity_fit_frame_projects_only_required_columns():
    frame = merge_results._build_identity_fit_frame(
        [_identity_row()],
        merge_results.MACLEOD_IDENTITY_BANDS,
        include_coordinates=True,
    )

    assert "large_unrelated_posterior_field" not in frame.columns
    assert list(frame.columns) == merge_results._identity_fit_fields(
        merge_results.MACLEOD_IDENTITY_BANDS,
        include_coordinates=True,
    )


def test_identity_fit_frame_reports_missing_required_columns():
    row = _identity_row()
    del row["log_tau_band_r_RF"]

    with pytest.raises(KeyError, match="log_tau_band_r_RF"):
        merge_results._build_identity_fit_frame(
            [row],
            merge_results.STONE_IDENTITY_BANDS,
        )


@pytest.mark.parametrize(
    ("runtimes", "expected"),
    (
        ([10.0, 20.0], "Runtime: mean 15.0 s · p90 19.0 s"),
        ([60.0, 120.0], "Runtime: mean 1.5 min · p90 1.9 min"),
        ([3600.0, 7200.0], "Runtime: mean 1.5 h · p90 1.9 h"),
    ),
)
def test_format_light_curve_runtime_annotation_uses_shared_readable_unit(
    runtimes,
    expected,
):
    rows = [
        {"light_curve_fit_total_elapsed_sec": runtime}
        for runtime in runtimes
    ]

    assert merge_results._format_light_curve_runtime_annotation(rows) == expected


def test_format_light_curve_runtime_annotation_filters_invalid_values():
    rows = [
        {"light_curve_fit_total_elapsed_sec": 3600.0},
        {"light_curve_fit_total_elapsed_sec": 7200.0},
        {"light_curve_fit_total_elapsed_sec": float("nan")},
        {"light_curve_fit_total_elapsed_sec": float("inf")},
        {"light_curve_fit_total_elapsed_sec": -1.0},
        {"light_curve_fit_total_elapsed_sec": "invalid"},
        {},
    ]

    assert merge_results._format_light_curve_runtime_annotation(rows) == (
        "Runtime: mean 1.5 h · p90 1.9 h"
    )


def test_stone_plot_runtime_annotation_uses_all_merged_rows(monkeypatch, tmp_path):
    captured = {}
    rows = [
        _identity_row(object_id="matched", light_curve_fit_total_elapsed_sec=3600.0),
        _identity_row(object_id="unmatched", light_curve_fit_total_elapsed_sec=7200.0),
    ]
    monkeypatch.setattr(
        merge_results,
        "build_stone_identity_plot_data",
        lambda *_args, **_kwargs: pd.DataFrame({"object_id": ["matched"]}),
    )

    def fake_plot(*_args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(merge_results, "plot_sigma_tau_identity_grid", fake_plot)
    monkeypatch.setattr(merge_results.plt, "close", lambda _fig: None)

    merge_results.write_stone_sigma_tau_identity_grid(
        rows,
        tmp_path / "plots" / "stone.pdf",
    )

    assert captured["figure_annotation"] == "Runtime: mean 1.5 h · p90 1.9 h"


def test_stone_plot_omits_runtime_annotation_for_legacy_rows(
    monkeypatch,
    tmp_path,
    capsys,
):
    captured = {}
    monkeypatch.setattr(
        merge_results,
        "build_stone_identity_plot_data",
        lambda *_args, **_kwargs: pd.DataFrame({"object_id": ["matched"]}),
    )

    def fake_plot(*_args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(merge_results, "plot_sigma_tau_identity_grid", fake_plot)
    monkeypatch.setattr(merge_results.plt, "close", lambda _fig: None)

    merge_results.write_stone_sigma_tau_identity_grid(
        [_identity_row()],
        tmp_path / "plots" / "stone.pdf",
    )

    assert captured["figure_annotation"] is None
    assert "omitting runtime annotation" in capsys.readouterr().out


def test_collect_light_curve_posterior_draws_preserves_pairs_and_converts_units(
    tmp_path,
):
    samples_dir = tmp_path / "samples"
    sigma_raw = np.linspace(-2.0, 1.0, 100)
    tau_raw = np.linspace(3.0, 7.0, 100)
    _write_light_curve_samples(
        samples_dir / "101_run_101.h5",
        sigma_raw,
        tau_raw,
    )
    rows = [{"object_id": "101", "suffix": "run_101", "z": 1.0}]

    first = merge_results.collect_light_curve_posterior_draws(
        rows,
        samples_dir,
        selection_seed=17,
    )
    second = merge_results.collect_light_curve_posterior_draws(
        rows,
        samples_dir,
        selection_seed=17,
    )

    selected = first["posterior_index"][0]
    assert first["valid_count"].tolist() == [64]
    assert first["source_draw_count"].tolist() == [100]
    assert first["finite_source_draw_count"].tolist() == [100]
    assert np.all(np.diff(selected) > 0)
    np.testing.assert_array_equal(
        second["posterior_index"],
        first["posterior_index"],
    )
    np.testing.assert_allclose(
        first["log_sigma_uv"][0],
        sigma_raw[selected] / np.log(10.0),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        first["log_tau_uv_rf"][0],
        tau_raw[selected] / np.log(10.0) - np.log10(2.0),
        rtol=1e-6,
    )


def test_collect_light_curve_posterior_draws_filters_pairs_and_pads(tmp_path):
    samples_dir = tmp_path / "samples"
    _write_light_curve_samples(
        samples_dir / "202_run_202.h5",
        [1.0, np.nan, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    )

    payload = merge_results.collect_light_curve_posterior_draws(
        [{"object_id": "202", "suffix": "run_202", "z": 2.0}],
        samples_dir,
    )

    assert payload["valid_count"].tolist() == [3]
    assert payload["source_draw_count"].tolist() == [4]
    assert payload["finite_source_draw_count"].tolist() == [3]
    np.testing.assert_array_equal(
        payload["posterior_index"][0, :3],
        np.array([0, 2, 3]),
    )
    assert np.all(payload["posterior_index"][0, 3:] == -1)
    assert np.all(np.isnan(payload["log_sigma_uv"][0, 3:]))
    assert np.all(np.isnan(payload["log_tau_uv_rf"][0, 3:]))


def test_collect_light_curve_posterior_draws_missing_is_strict_by_default(
    tmp_path,
):
    rows = [{"object_id": "missing", "suffix": "run_missing", "z": 1.0}]

    with pytest.raises(FileNotFoundError, match="allow-missing-posterior-draws"):
        merge_results.collect_light_curve_posterior_draws(rows, tmp_path)

    payload = merge_results.collect_light_curve_posterior_draws(
        rows,
        tmp_path,
        allow_missing=True,
    )
    assert payload["missing_count"] == 1
    assert payload["valid_count"].tolist() == [0]
    assert np.all(np.isnan(payload["log_sigma_uv"]))
    assert np.all(payload["posterior_index"] == -1)


def test_write_quasars_to_h5_flat_writes_posterior_draw_group(tmp_path):
    samples_dir = tmp_path / "samples"
    sigma_raw = np.linspace(-1.0, 1.0, 80)
    tau_raw = np.linspace(2.0, 6.0, 80)
    _write_light_curve_samples(
        samples_dir / "303_run_303.h5",
        sigma_raw,
        tau_raw,
    )
    rows = [{"object_id": "303", "suffix": "run_303", "z": 0.5}]
    payload = merge_results.collect_light_curve_posterior_draws(
        rows,
        samples_dir,
        selection_seed=9,
    )
    output = tmp_path / "merged.h5"

    merge_results.write_quasars_to_h5_flat(
        rows,
        output,
        posterior_draw_payload=payload,
    )

    with h5py.File(output, "r") as handle:
        group = handle[merge_results.LIGHT_CURVE_POSTERIOR_DRAW_GROUP]
        # A raw sample file without the new definition attribute is legacy v1;
        # merging must preserve that semantic label rather than silently
        # relabeling its old log_tau_uv draws as the new 2500 A definition.
        assert (
            group.attrs["format"]
            == merge_results.LIGHT_CURVE_POSTERIOR_DRAW_FORMAT_V1
        )
        assert group.attrs["draw_count"] == 64
        assert group.attrs["selection_seed"] == 9
        assert group["log_sigma_uv"].shape == (1, 64)
        assert group["log_tau_uv_rf"].shape == (1, 64)
        assert group["posterior_index"].shape == (1, 64)
        assert group["valid_count"][0] == 64


def test_merge_prefers_embedded_chunk_draws_over_sample_file_fallback(tmp_path):
    rows = [{"object_id": "404", "suffix": "run_404", "z": 1.5}]
    embedded = merge_results.collect_light_curve_posterior_draws(
        rows,
        _create_sample_dir(
            tmp_path / "embedded_source",
            "404_run_404.h5",
            np.linspace(-2.0, -1.0, 80),
            np.linspace(4.0, 5.0, 80),
        ),
        selection_seed=0,
    )
    chunk = tmp_path / "chunk.h5"
    merge_results.write_quasars_to_h5_flat(
        rows,
        chunk,
        posterior_draw_payload=embedded,
    )
    loaded = merge_results._load_h5_shard(str(chunk), expected_n=None)
    assert loaded["ok"] is True

    fallback_dir = _create_sample_dir(
        tmp_path / "fallback",
        "404_run_404.h5",
        np.linspace(10.0, 11.0, 80),
        np.linspace(12.0, 13.0, 80),
    )
    merged = merge_results.collect_light_curve_posterior_draws(
        loaded["rows"],
        fallback_dir,
        selection_seed=0,
    )

    np.testing.assert_array_equal(
        merged["posterior_index"], embedded["posterior_index"]
    )
    np.testing.assert_allclose(
        merged["log_sigma_uv"], embedded["log_sigma_uv"]
    )
    np.testing.assert_allclose(
        merged["log_tau_uv_rf"], embedded["log_tau_uv_rf"]
    )


def test_main_merges_embedded_draws_without_sample_files(tmp_path):
    prefix = "embedded_only"
    base_dir = tmp_path / "data"
    shard_dir = base_dir / prefix
    shard_dir.mkdir(parents=True)
    rows = [{"object_id": "505", "suffix": "run_505", "z": 2.0}]
    source_dir = _create_sample_dir(
        tmp_path / "source_once",
        "505_run_505.h5",
        np.linspace(-2.0, -1.0, 80),
        np.linspace(4.0, 5.0, 80),
    )
    payload = merge_results.collect_light_curve_posterior_draws(
        rows,
        source_dir,
    )
    merge_results.write_quasars_to_h5_flat(
        rows,
        shard_dir / "505.h5",
        posterior_draw_payload=payload,
    )
    output = tmp_path / "merged.h5"

    merge_results.main(
        [
            prefix,
            "--base-dir",
            str(base_dir),
            "--skip-populate-sdss",
            "--out",
            str(output),
        ]
    )

    with h5py.File(output, "r") as handle:
        group = handle[merge_results.LIGHT_CURVE_POSTERIOR_DRAW_GROUP]
        np.testing.assert_allclose(
            group["log_sigma_uv"][...], payload["log_sigma_uv"]
        )
        np.testing.assert_allclose(
            group["log_tau_uv_rf"][...], payload["log_tau_uv_rf"]
        )
        np.testing.assert_array_equal(
            group["posterior_index"][...], payload["posterior_index"]
        )
