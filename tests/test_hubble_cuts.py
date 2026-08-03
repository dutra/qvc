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
    AGN_SCALAR_PARAMETER_CUTS,
    APPARENT_MAG_2500_ERR_MAX,
    COMPLETENESS_MAG_2500_MIN,
    COMPLETENESS_MAG_2500_MAX,
    EXCLUDED_SDSS_NAMES,
    FRAC_AGN_5100_MIN,
    LIGHT_CURVE_N_POINTS_COLUMN,
    LIGHT_CURVE_N_POINTS_EXCLUDED_BANDS,
    REL_APPARENT_MAG_2500_ERR_MAX,
    add_light_curve_point_count_column,
    light_curve_point_count_series,
)
from qvc.hubble.hubble_cut_config import build_agn_cuts, build_dlog_amp_blr_cuts  # noqa: E402
from qvc.hubble.hubble_utils import (  # noqa: E402
    _append_cut_report_row,
    _count_redshift_bin_removals,
    _render_cut_summary_table,
    _scalar_cut_has_inclusive_upper,
    _scalar_parameter_cut_mask,
    populate_spectra_fit,
)


def test_build_agn_cuts_contains_only_fiducial_profile():
    cuts = build_agn_cuts()
    cut_map = {column: (lower, upper) for column, lower, upper in cuts}

    assert tuple(cuts) == AGN_SCALAR_PARAMETER_CUTS
    assert cut_map == {
        "log_tau_uv_rf": (1.5, 4.0),
        "fracAGN_5100_fit": (FRAC_AGN_5100_MIN, None),
        "apparent_mag_2500_err": (None, APPARENT_MAG_2500_ERR_MAX),
        "m_2500_attenuated_model": (
            COMPLETENESS_MAG_2500_MIN,
            COMPLETENESS_MAG_2500_MAX,
        ),
    }
    assert LIGHT_CURVE_N_POINTS_EXCLUDED_BANDS == ("u",)


def test_previous_scalar_and_component_defaults_are_disabled():
    active_columns = {column for column, _, _ in build_agn_cuts()}
    assert active_columns.isdisjoint(
        {
            "wrms",
            "t_rf_length",
            LIGHT_CURVE_N_POINTS_COLUMN,
            "f_host_2500",
            "alpha_lambda",
            "variability_chi_sq_red_g",
            "loo_chi2_eff",
            "log_sigma_uv",
        }
    )
    assert build_dlog_amp_blr_cuts() == []
    assert len(EXCLUDED_SDSS_NAMES) == 9
    assert REL_APPARENT_MAG_2500_ERR_MAX is None


def test_fiducial_cut_boundaries_are_inclusive_and_nonfinite_values_fail():
    cases = (
        ("log_tau_uv_rf", 1.5, 4.0),
        ("fracAGN_5100_fit", FRAC_AGN_5100_MIN, None),
        ("apparent_mag_2500_err", None, APPARENT_MAG_2500_ERR_MAX),
        (
            "m_2500_attenuated_model",
            COMPLETENESS_MAG_2500_MIN,
            COMPLETENESS_MAG_2500_MAX,
        ),
    )
    for column, lower, upper in cases:
        accepted = lower if lower is not None else upper
        outside = (
            np.nextafter(lower, -np.inf)
            if lower is not None
            else np.nextafter(upper, np.inf)
        )
        values = [accepted, outside, np.nan, np.inf, -np.inf]
        if lower is not None and upper is not None:
            values.insert(1, upper)
            expected = [True, True, False, False, False, False]
        else:
            expected = [True, False, False, False, False]
        mask = _scalar_parameter_cut_mask(
            pd.DataFrame({column: values}), column, lower, upper
        )
        np.testing.assert_array_equal(mask, expected)
        assert _scalar_cut_has_inclusive_upper(column)


def test_current_spectra_schema_requires_fracagn_5100_fit(tmp_path):
    csv_path = tmp_path / "spectra.csv"
    pd.DataFrame(
        {
            "object_id": ["obj"],
            "fit_ok": [True],
            "m_2500_dereddened": [20.0],
            "m_2500_dereddened_err": [0.1],
            "m_2500_attenuated_model": [20.1],
            "m_2500_attenuated_model_err": [0.1],
            "pl_slope": [-1.5],
            "pl_slope_err": [0.1],
        }
    ).to_csv(csv_path, index=False)

    with np.testing.assert_raises_regex(ValueError, "fracAGN_5100_fit"):
        populate_spectra_fit(pd.DataFrame({"object_id": ["obj"]}), [csv_path])


def test_current_spectra_schema_accepts_only_joint_sedfit_backend(tmp_path):
    csv_path = tmp_path / "spectra.csv"
    row = {
        "object_id": "obj",
        "fit_ok": True,
        "fit_backend": "jaxsedfit_joint",
        "fracAGN_5100_fit": 0.8,
        "fracAGN_5100_fit_err": 0.05,
        "m_2500_dereddened": 20.0,
        "m_2500_dereddened_err": 0.1,
        "m_2500_attenuated_model": 20.2,
        "m_2500_attenuated_model_err": 0.12,
        "pl_slope": -1.5,
        "pl_slope_err": 0.1,
    }
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    out = populate_spectra_fit(
        pd.DataFrame({"object_id": ["obj", "not-matched"]}),
        [csv_path],
    )

    assert out["object_id"].tolist() == ["obj"]
    assert out.loc[0, "fit_backend"] == "jaxsedfit_joint"
    assert out.loc[0, "alpha_lambda"] == row["pl_slope"]
    assert "PL_slope" not in out.columns
    assert "f_host_2500" not in out.columns

    row["fit_backend"] = "jaxqsofit"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    with np.testing.assert_raises_regex(ValueError, "unsupported fit_backend"):
        populate_spectra_fit(pd.DataFrame({"object_id": ["obj"]}), [csv_path])


def test_populate_spectra_fit_preserves_all_nonconflicting_columns_and_hdf5_wins(
    tmp_path,
):
    csv_path = tmp_path / "spectra.csv"
    row = {
        "object_id": "obj",
        "fit_ok": True,
        "fit_backend": "jaxsedfit_joint",
        "z": 1.25,
        "pl_slope": -1.5,
        "pl_slope_err": 0.1,
        "fracAGN_5100_fit": 0.8,
        "fracAGN_5100_fit_err": 0.05,
        "m_2500_dereddened": 20.0,
        "m_2500_dereddened_err": 0.1,
        "m_2500_attenuated_model": 20.2,
        "m_2500_attenuated_model_err": 0.12,
        "new_sed_parameter": 42.5,
        "new_sed_label": "well_constrained",
    }
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    source = pd.DataFrame(
        {
            "object_id": ["obj"],
            "z": [1.5],
            "pl_slope": [-0.8],
            "alpha_lambda": [-0.75],
            "alpha_lambda_err": [0.03],
            "alpha_nu": [-1.25],
            "alpha_nu_err": [0.03],
        }
    )

    out = populate_spectra_fit(source, [csv_path])

    assert out.loc[0, "z"] == 1.5
    assert out.loc[0, "pl_slope"] == -0.8
    assert out.loc[0, "alpha_lambda"] == -0.75
    assert out.loc[0, "alpha_lambda_err"] == 0.03
    assert out.loc[0, "alpha_nu"] == -1.25
    assert out.loc[0, "alpha_nu_err"] == 0.03
    assert out.loc[0, "new_sed_parameter"] == 42.5
    assert out.loc[0, "new_sed_label"] == "well_constrained"
    assert not any(column.endswith(("_x", "_y", "_sedfit")) for column in out.columns)
    assert "new_sed_parameter" in out.attrs["spectra_fit_columns"]
    assert "new_sed_label" in out.attrs["spectra_fit_columns"]
    assert "z" not in out.attrs["spectra_fit_columns"]
    assert "pl_slope" not in out.attrs["spectra_fit_columns"]


def test_populate_spectra_fit_derives_missing_slope_aliases_without_overwriting(
    tmp_path,
):
    csv_path = tmp_path / "spectra.csv"
    pd.DataFrame(
        [
            {
                "object_id": "obj",
                "fit_ok": True,
                "fit_backend": "jaxsedfit_joint",
                "fracAGN_5100_fit": 0.8,
                "fracAGN_5100_fit_err": 0.05,
                "m_2500_dereddened": 20.0,
                "m_2500_dereddened_err": 0.1,
                "m_2500_attenuated_model": 20.2,
                "m_2500_attenuated_model_err": 0.12,
                "pl_slope": -1.5,
                "pl_slope_err": 0.1,
            }
        ]
    ).to_csv(csv_path, index=False)

    out = populate_spectra_fit(pd.DataFrame({"object_id": ["obj"]}), [csv_path])

    assert out.loc[0, "alpha_lambda"] == -1.5
    assert out.loc[0, "alpha_lambda_err"] == 0.1
    assert out.loc[0, "alpha_nu"] == -0.5
    assert out.loc[0, "alpha_nu_err"] == 0.1


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
