import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble.cuts import (  # noqa: E402
    APPARENT_MAG_2500_MAX,
    LIGHT_CURVE_N_POINTS_COLUMN,
    LIGHT_CURVE_N_POINTS_EXCLUDED_BANDS,
    LIGHT_CURVE_N_POINTS_MIN,
    LOO_CHI2_EFF_MAX,
    add_light_curve_point_count_column,
    light_curve_point_count_series,
)
from qvc.hubble.hubble_cut_config import build_agn_cuts  # noqa: E402
from qvc.hubble.hubble_utils import (  # noqa: E402
    _append_cut_report_row,
    _count_redshift_bin_removals,
    _render_cut_summary_table,
    _scalar_cut_has_inclusive_upper,
    _scalar_parameter_cut_mask,
)


def test_build_agn_cuts_includes_light_curve_point_count_minimum():
    cuts = build_agn_cuts()
    cut_map = {column: (lower, upper) for column, lower, upper in cuts}

    assert cut_map[LIGHT_CURVE_N_POINTS_COLUMN] == (LIGHT_CURVE_N_POINTS_MIN, None)
    assert LIGHT_CURVE_N_POINTS_EXCLUDED_BANDS == ("u",)


def test_build_agn_cuts_includes_apparent_mag_2500_strict_maximum():
    cuts = build_agn_cuts()
    cut_map = {column: (lower, upper) for column, lower, upper in cuts}

    assert cut_map["apparent_mag_2500"] == (None, APPARENT_MAG_2500_MAX)
    assert not _scalar_cut_has_inclusive_upper("apparent_mag_2500")


def test_build_agn_cuts_includes_loo_chi2_eff_maximum():
    cuts = build_agn_cuts()
    cut_map = {column: (lower, upper) for column, lower, upper in cuts}

    assert cut_map["loo_chi2_eff"] == (None, LOO_CHI2_EFF_MAX)
    assert _scalar_cut_has_inclusive_upper("loo_chi2_eff")


def test_apparent_mag_2500_cut_is_disabled_when_threshold_is_none():
    df = pd.DataFrame(
        {
            "apparent_mag_2500": [
                20.0,
                22.5,
                23.0,
                np.nan,
                "not-a-number",
            ],
        }
    )

    mask = _scalar_parameter_cut_mask(
        df,
        "apparent_mag_2500",
        None,
        APPARENT_MAG_2500_MAX,
    )

    np.testing.assert_array_equal(mask, [True, True, True, True, True])


def test_light_curve_point_count_series_prefers_cleaned_per_band_counts():
    df = pd.DataFrame(
        {
            "variability_n_points_u": [1000, 1000, 1000],
            "variability_n_points_g": [300, 100, np.nan],
            "number_points_g": [999, 999, 999],
            "number_points_r": [250, 450, 10],
            "variability_n_points_i": [0, 20, 30],
        }
    )

    counts, count_cols = light_curve_point_count_series(df)

    assert "variability_n_points_u" in count_cols
    np.testing.assert_allclose(counts, [1550, 1570, 1040])


def test_light_curve_point_count_cut_excludes_u_band():
    df = pd.DataFrame(
        {
            "variability_n_points_u": [1000, 1000, 1000],
            "variability_n_points_g": [300, 100, np.nan],
            "number_points_g": [999, 999, 999],
            "number_points_r": [250, 450, 10],
            "variability_n_points_i": [0, 20, 30],
        }
    )

    counts, count_cols = light_curve_point_count_series(
        df,
        exclude_bands=LIGHT_CURVE_N_POINTS_EXCLUDED_BANDS,
    )

    assert "variability_n_points_u" not in count_cols
    assert "variability_n_points_g" in count_cols
    assert "number_points_g" not in count_cols
    np.testing.assert_allclose(counts, [550, 570, 40])


def test_add_light_curve_point_count_column_handles_legacy_number_points():
    df = pd.DataFrame(
        {
            "number_points": [
                "{'u': 1000, 'g': 250, 'r': 251}",
                [1000, 100, 200, 250, 50],
                499,
            ],
        }
    )

    out, count_cols = add_light_curve_point_count_column(df)

    assert count_cols == ["number_points"]
    np.testing.assert_allclose(out[LIGHT_CURVE_N_POINTS_COLUMN], [501, 600, 499])


def test_count_redshift_bin_removals_uses_requested_bins():
    df = pd.DataFrame(
        {
            "z": [0.1, 0.439, 0.44, 0.8, 1.0, 1.5, 2.0, 3.16, 3.17, np.nan],
        }
    )

    counts = _count_redshift_bin_removals(df)

    assert counts == {
        "removed_z_lt_0p44": 2,
        "removed_z_0p44_to_1": 2,
        "removed_z_1_to_2": 2,
        "removed_z_2_to_3p16": 2,
        "removed_z_gt_3p16": 1,
        "removed_z_lt_1p5": 5,
        "removed_z_ge_1p5": 4,
    }


def test_append_cut_report_row_includes_zero_filled_redshift_bins_without_removed_frame():
    rows = []

    _append_cut_report_row(
        rows,
        step="agn_scalar:dummy",
        criterion="dummy skipped",
        before=5,
        kept=5,
        status="skipped",
    )

    assert rows == [
        {
            "step": "agn_scalar:dummy",
            "criterion": "dummy skipped",
            "before": 5,
            "removed": 0,
            "kept": 5,
            "status": "skipped",
            "removed_z_lt_0p44": 0,
            "removed_z_0p44_to_1": 0,
            "removed_z_1_to_2": 0,
            "removed_z_2_to_3p16": 0,
            "removed_z_gt_3p16": 0,
            "removed_z_lt_1p5": 0,
            "removed_z_ge_1p5": 0,
        }
    ]


def test_render_cut_summary_table_shows_low_and_high_redshift_removals_after_total():
    rows = []
    _append_cut_report_row(
        rows,
        step="agn_scalar:dummy",
        criterion="dummy criterion",
        before=5,
        kept=1,
        status="applied",
        removed_frame=pd.DataFrame({"z": [0.2, 1.49, 1.5, 2.0]}),
    )

    table = _render_cut_summary_table(rows)
    lines = table.splitlines()
    headers = [cell.strip() for cell in lines[1].strip("|").split("|")]
    values = [cell.strip() for cell in lines[3].strip("|").split("|")]

    assert headers == [
        "step",
        "criterion",
        "before",
        "removed",
        "removed z < 1.5",
        "removed z >= 1.5",
        "kept",
        "status",
    ]
    assert values == [
        "agn_scalar:dummy",
        "dummy criterion",
        "5",
        "4",
        "2",
        "2",
        "1",
        "applied",
    ]
