import pandas as pd
import pytest

from qvc.light_curve import merge_results


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
