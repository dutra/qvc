import h5py
import numpy as np
import pandas as pd
import pytest

from qvc.light_curve import merge_results, plotting_appendix


def _write_light_curve_samples(path, log_sigma_uv, log_tau_uv):
    path.parent.mkdir(parents=True, exist_ok=True)
    n_draws = len(log_sigma_uv)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("log_sigma_uv", data=np.asarray(log_sigma_uv))
        handle.create_dataset("log_tau_uv", data=np.asarray(log_tau_uv))
        handle.create_dataset("eta_sigma", data=np.zeros(n_draws))
        handle.create_dataset("eta_tau", data=np.zeros(n_draws))
        handle.attrs["model_variant"] = "erlang_dho"
        handle.attrs["bands"] = "g,r"


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


def test_density_contour_grid_returns_68_and_84_percent_levels():
    rng = np.random.default_rng(123)
    x = rng.normal(size=2000)
    y = 0.6 * x + rng.normal(scale=0.5, size=x.size)

    contour_grid = plotting_appendix._density_contour_grid(
        x,
        y,
        xlim=(-4, 4),
        ylim=(-4, 4),
    )

    assert contour_grid is not None
    x_centers, y_centers, density, levels = contour_grid
    assert density.shape == (len(y_centers), len(x_centers))
    assert len(levels) == 2
    assert np.all(np.diff(levels) > 0)


def test_identity_panel_metrics_can_omit_count():
    text = plotting_appendix._format_identity_panel_metrics(
        {"N": 12, "Bias": 0.1, "sigma": 0.2},
        unit="dex",
        include_count=False,
    )

    assert "N =" not in text
    assert "Bias = 0.10 dex" in text


def test_identity_grid_can_expand_limits_for_errorbars():
    data = pd.DataFrame(
        {
            "sigma_x_g": [0.0],
            "sigma_y_g": [0.0],
            "sigma_xerr_g": [0.4],
            "sigma_yerr_g": [0.5],
            "tau_x_g": [2.0],
            "tau_y_g": [2.0],
            "tau_xerr_g": [0.6],
            "tau_yerr_g": [0.7],
        }
    )
    fig = plotting_appendix.plot_sigma_tau_identity_grid(
        data,
        {
            "x": "sigma_x_{band}",
            "y": "sigma_y_{band}",
            "xerr": "sigma_xerr_{band}",
            "yerr": "sigma_yerr_{band}",
        },
        {
            "x": "tau_x_{band}",
            "y": "tau_y_{band}",
            "xerr": "tau_xerr_{band}",
            "yerr": "tau_yerr_{band}",
        },
        bands=("g",),
        show=False,
        sigma_limits=(-0.1, 0.1),
        tau_limits=(1.9, 2.1),
        expand_limits_for_errorbars=True,
    )

    sigma_limits = fig.axes[0].get_xlim()
    tau_limits = fig.axes[1].get_xlim()
    merge_results.plt.close(fig)
    assert sigma_limits[0] < -0.5
    assert sigma_limits[1] > 0.5
    assert tau_limits[0] < 1.3
    assert tau_limits[1] > 2.7


def test_apply_macleod_hubble_light_curve_cuts_omits_spectral_fields(monkeypatch):
    monkeypatch.setattr(merge_results, "LOO_CHI2_EFF_MAX", 1.0)
    monkeypatch.setattr(merge_results, "NUM_DIVERGENCES_MAX", 0.0)
    monkeypatch.setattr(merge_results, "LIGHT_CURVE_RHAT_MAX", 1.05)
    monkeypatch.setattr(merge_results, "LOG_TAU_UV_RF_MIN", 1.3)
    monkeypatch.setattr(merge_results, "LOG_TAU_UV_RF_MAX", None)
    monkeypatch.setattr(merge_results, "T_RF_LENGTH_MIN", None)
    monkeypatch.setattr(merge_results, "LIGHT_CURVE_N_POINTS_MIN", None)
    monkeypatch.setattr(merge_results, "ETA_SIGMA_KL_MIN", None)
    monkeypatch.setattr(merge_results, "T_RF_OVER_TAU_UV_RF_MIN", None)
    base = {
        "sdss_name": "kept",
        "loo_chi2_eff": 0.9,
        "num_divergences": 0,
        "log_tau_uv_rf_rhat": 1.01,
        "log_sigma_uv_rhat": 1.01,
        "log_tau_uv_rf": 1.5,
        "spectroscopy_reduced_chi2": 99.0,
        "SN_MEDIAN_ALL": 0.0,
    }
    rows = [
        {**base, "object_id": "kept"},
        {**base, "object_id": "low_tau", "log_tau_uv_rf": 1.2},
        {
            **base,
            "object_id": "manual",
            "sdss_name": next(
                iter(merge_results.EXCLUDED_LIGHT_CURVE_SDSS_NAMES)
            ),
        },
    ]

    selected = merge_results.apply_macleod_hubble_light_curve_cuts(rows)

    assert [row["object_id"] for row in selected] == ["kept"]


def test_write_macleod_comparison_suite_writes_both_outputs(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        merge_results,
        "write_macleod_sigma_tau_identity_grid",
        lambda rows, output_path, macleod_dir=None: calls.append(
            (list(rows), str(output_path), macleod_dir)
        )
        or str(output_path),
    )
    monkeypatch.setattr(
        merge_results,
        "apply_macleod_hubble_light_curve_cuts",
        lambda rows: list(rows)[1:],
    )
    output = tmp_path / "sigma_tau_identity_grid_macleod.pdf"

    paths = merge_results.write_macleod_comparison_suite(
        [{"object_id": "1"}, {"object_id": "2"}],
        output,
    )

    expected_cut = tmp_path / "sigma_tau_identity_grid_macleod_hubble_lc_cuts.pdf"
    assert paths == {
        "all": str(output),
        "hubble_light_curve_cuts": str(expected_cut),
    }
    assert [call[1] for call in calls] == [str(output), str(expected_cut)]
    assert [row["object_id"] for row in calls[1][0]] == ["2"]


def test_write_macleod_identity_grid_enables_blue_sigma_contours(
    monkeypatch,
    tmp_path,
):
    captured = {}
    monkeypatch.setattr(
        merge_results,
        "build_macleod_identity_plot_data",
        lambda *_args, **_kwargs: {},
    )

    def fake_plot(*_args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(merge_results, "plot_sigma_tau_identity_grid", fake_plot)
    monkeypatch.setattr(merge_results.plt, "close", lambda _fig: None)

    merge_results.write_macleod_sigma_tau_identity_grid(
        [_identity_row()],
        tmp_path / "macleod.pdf",
    )

    assert captured["style"]["contour_color"] == "tab:blue"
    assert captured["style"]["contour_linewidth"] == 1.7
    assert captured["style"]["show_metric_count"] is False
    assert plotting_appendix.IDENTITY_CONTOUR_ENCLOSED_PROBABILITIES == (0.84, 0.68)


def test_write_samelength_identity_grid_enables_blue_contours_and_expands_limits(
    monkeypatch,
    tmp_path,
):
    captured = {}
    monkeypatch.setattr(
        merge_results,
        "build_samelength_identity_plot_data",
        lambda *_args, **_kwargs: pd.DataFrame({"object_id": ["1"]}),
    )

    def fake_plot(*_args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(merge_results, "plot_sigma_tau_identity_grid", fake_plot)
    monkeypatch.setattr(merge_results.plt, "close", lambda _fig: None)

    merge_results.write_samelength_sigma_tau_identity_grid(
        [_identity_row()],
        [_identity_row()],
        tmp_path / "samelength.pdf",
    )

    assert captured["style"]["contour_color"] == "tab:blue"
    assert captured["style"]["contour_smoothing"] == 4.0
    assert captured["style"]["show_metric_count"] is False
    assert captured["expand_limits_for_errorbars"] is True


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
    assert captured["style"]["contour_color"] == "tab:blue"
    assert captured["style"]["contour_smoothing"] == 4.0
    assert captured["style"]["show_metric_count"] is False
    assert captured["expand_limits_for_errorbars"] is True


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
    sigma_raw = np.linspace(-2.0, 1.0, 200)
    tau_raw = np.linspace(3.0, 7.0, 200)
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
    assert first["valid_count"].tolist() == [128]
    assert first["source_draw_count"].tolist() == [200]
    assert first["finite_source_draw_count"].tolist() == [200]
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
        assert group.attrs["format"] == merge_results.LIGHT_CURVE_POSTERIOR_DRAW_FORMAT
        assert group.attrs["draw_count"] == 128
        assert group.attrs["selection_seed"] == 9
        assert group["log_sigma_uv"].shape == (1, 128)
        assert group["log_tau_uv_rf"].shape == (1, 128)
        assert group["posterior_index"].shape == (1, 128)
        assert group["log_sigma_band_g"].shape == (1, 128)
        assert group["log_tau_cont_band_g_rf"].shape == (1, 128)
        assert group["valid_count"][0] == 80


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


def test_merge_rebuilds_old_embedded_group_from_sample_prefix(tmp_path):
    prefix = "old_shard"
    sample_prefix = "full_samples"
    base_dir = tmp_path / "data"
    shard_dir = base_dir / prefix
    shard_dir.mkdir(parents=True)
    with h5py.File(shard_dir / "606.h5", "w") as handle:
        handle.create_dataset("object_id", data=np.asarray([b"606"]))
        handle.create_dataset("suffix", data=np.asarray([b"run_606"]))
        handle.create_dataset("z", data=np.asarray([1.0]))
        handle.create_dataset("bands_kept", data=np.asarray([b"g,r"]))
        old = handle.create_group("light_curve_posterior_draws")
        old.attrs["format"] = "qvc_light_curve_posterior_draws_v2"
        old.attrs["draw_count"] = 64
        old.create_dataset("log_sigma_uv", data=np.ones((1, 64)))
    _write_light_curve_samples(
        tmp_path / "samples" / sample_prefix / "606_run_606.h5",
        np.linspace(-2.0, -1.0, 140),
        np.linspace(4.0, 5.0, 140),
    )
    output = tmp_path / "merged.h5"

    merge_results.main(
        [
            prefix,
            "--base-dir",
            str(base_dir),
            "--posterior-samples-prefix",
            sample_prefix,
            "--skip-populate-sdss",
            "--out",
            str(output),
        ]
    )

    with h5py.File(output, "r") as handle:
        group = handle["light_curve_posterior_draws"]
        assert group.attrs["format"] == "qvc_light_curve_posterior_draws_v3"
        assert group.attrs["draw_count"] == 128
        assert group["valid_count"][0] == 128
        assert group["log_sigma_band_g"].shape == (1, 128)


def test_slb_sample_fallback_computes_continuum_band_tau(tmp_path):
    samples_dir = tmp_path / "samples"
    path = samples_dir / "707_run_707.h5"
    path.parent.mkdir(parents=True)
    tau_fast = np.asarray([10.0, 15.0, 20.0])
    tau_slow = np.asarray([100.0, 150.0, 200.0])
    lag_disk = np.asarray([[2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("log_sigma_uv", data=np.log([0.1, 0.2, 0.3]))
        handle.create_dataset("log_tau_uv", data=np.log([110.0, 165.0, 220.0]))
        handle.create_dataset("eta_sigma", data=np.zeros(3))
        handle.create_dataset("tau_fast_driver", data=tau_fast)
        handle.create_dataset("tau_slow_driver", data=tau_slow)
        handle.create_dataset("lag_disk", data=lag_disk)
        handle.attrs["model_variant"] = "shared_latent_blr"
        handle.attrs["bands"] = "g,r"
        handle.attrs["disk_order"] = 3
        handle.attrs["log_tau_uv_definition"] = (
            "continuum_only_disk_convolved_integral_timescale_at_rest_"
            "2500A_observer_frame_natural_log"
        )

    payload = merge_results.collect_light_curve_posterior_draws(
        [{"object_id": "707", "suffix": "run_707", "z": 1.0}],
        samples_dir,
        posterior_disk_order=5,
    )

    expected_g = np.asarray(
        [
            np.log10(
                float(
                    merge_results.continuum_effective_timescale(
                        tf, ts, lag, disk_order=3
                    )
                )
            )
            - np.log10(2.0)
            for tf, ts, lag in zip(tau_fast, tau_slow, lag_disk[:, 0])
        ]
    )
    np.testing.assert_allclose(
        payload["log_tau_cont_band_g_rf"][0, :3], expected_g, rtol=2e-6
    )
    assert payload["disk_order"][0] == 3
