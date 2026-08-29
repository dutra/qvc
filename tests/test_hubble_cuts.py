import os
from ast import literal_eval
import subprocess
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
    AGN_TIER0_ELIGIBILITY_CUTS,
    AGN_TIER1_FIT_QUALITY_CUTS,
    AGN_TIER2_PARAMETER_CUTS,
    ALLOW_MISSING_SCALAR_CUT_COLUMNS,
    APPARENT_MAG_2500_ERR_MAX,
    COMPLETENESS_MAG_2500_MIN,
    COMPLETENESS_MAG_2500_MAX,
    EXCLUDED_SDSS_NAMES,
    FRAC_AGN_5100_MIN,
    JAXSEDFIT_JOINT_REDUCED_CHI2_MAX,
    LIGHT_CURVE_N_POINTS_COLUMN,
    LIGHT_CURVE_N_POINTS_EXCLUDED_BANDS,
    LOO_CHI2_EFF_MAX,
    LIGHT_CURVE_RHAT_MAX,
    SED_REDUCED_CHI2_MAX,
    SPECTRAL_RHAT_MAX,
    SPECTROSCOPY_REDUCED_CHI2_MAX,
    T_RF_OVER_TAU_UV_RF_COLUMN,
    REL_APPARENT_MAG_2500_ERR_MAX,
    add_light_curve_point_count_column,
    light_curve_point_count_series,
    normalize_sdss_target_selection,
    normalize_cut_tier,
)
from qvc.hubble.hubble_cut_config import (  # noqa: E402
    build_agn_cuts,
    build_dlog_amp_blr_cuts,
    build_tier0_cuts,
    build_tier1_cuts,
    build_tier2_cuts,
)
from qvc.hubble.hubble_utils import (  # noqa: E402
    _append_cut_report_row,
    _count_redshift_bin_removals,
    _render_cut_summary_table,
    _scalar_cut_has_inclusive_upper,
    _scalar_parameter_cut_mask,
    populate_spectra_fit,
)


def test_build_agn_cuts_are_partitioned_in_tier_order():
    cuts = build_agn_cuts()
    cut_map = {column: (lower, upper) for column, lower, upper in cuts}

    assert cuts == build_tier0_cuts() + build_tier1_cuts() + build_tier2_cuts()
    assert tuple(build_tier0_cuts()) == AGN_TIER0_ELIGIBILITY_CUTS
    assert tuple(build_tier1_cuts()) == AGN_TIER1_FIT_QUALITY_CUTS
    assert cut_map == {
        "log_tau_uv_rf": (1.5, 4.0),
        T_RF_OVER_TAU_UV_RF_COLUMN: (5.0, None),
        "apparent_mag_2500_err": (None, APPARENT_MAG_2500_ERR_MAX),
        "m_2500_dereddened": (
            COMPLETENESS_MAG_2500_MIN,
            COMPLETENESS_MAG_2500_MAX,
        ),
        "sed_reduced_chi2": (None, SED_REDUCED_CHI2_MAX),
        "spectroscopy_reduced_chi2": (None, SPECTROSCOPY_REDUCED_CHI2_MAX),
        "joint_reduced_chi2": (None, JAXSEDFIT_JOINT_REDUCED_CHI2_MAX),
        "loo_chi2_eff": (None, LOO_CHI2_EFF_MAX),
        "m_2500_dereddened_rhat": (None, SPECTRAL_RHAT_MAX),
        "m_2500_attenuated_model_rhat": (None, SPECTRAL_RHAT_MAX),
        "log_tau_uv_rf_rhat": (None, LIGHT_CURVE_RHAT_MAX),
        "log_sigma_uv_rhat": (None, LIGHT_CURVE_RHAT_MAX),
    }
    assert LIGHT_CURVE_N_POINTS_EXCLUDED_BANDS == ("u",)


def test_normalize_cut_tier_accepts_exact_four_modes():
    assert [normalize_cut_tier(value) for value in ("none", 0, 1, 2)] == [
        "none", "0", "1", "2"
    ]
    with np.testing.assert_raises_regex(ValueError, "Unknown cut tier"):
        normalize_cut_tier("3")


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
            "m_2500_dereddened",
            COMPLETENESS_MAG_2500_MIN,
            COMPLETENESS_MAG_2500_MAX,
        ),
    )
    for column, lower, upper in cases:
        if lower is None and upper is None:
            mask = _scalar_parameter_cut_mask(
                pd.DataFrame({column: [0.5, np.nan, np.inf, -np.inf]}),
                column,
                lower,
                upper,
            )
            np.testing.assert_array_equal(mask, [True, False, False, False])
            continue
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


def test_completeness_magnitude_cut_follows_selected_definition():
    dereddened_columns = {column for column, _, _ in build_agn_cuts()}
    attenuated_columns = {
        column
        for column, _, _ in build_agn_cuts(completeness_magnitude="attenuated")
    }

    assert "m_2500_dereddened" in dereddened_columns
    assert "m_2500_attenuated_model" not in dereddened_columns
    assert "m_2500_attenuated_model" in attenuated_columns
    assert "m_2500_dereddened" not in attenuated_columns


def test_jaxsedfit_joint_reduced_chi2_cut_requires_finite_values():
    column = "joint_reduced_chi2"
    upper = JAXSEDFIT_JOINT_REDUCED_CHI2_MAX
    values = [upper, np.nextafter(upper, np.inf), np.nan, np.inf, -np.inf]
    mask = _scalar_parameter_cut_mask(
        pd.DataFrame({column: values}), column, None, upper
    )
    np.testing.assert_array_equal(mask, [True, False, False, False, False])


def test_fit_quality_cuts_reject_bad_or_missing_values():
    cases = (
        ("sed_reduced_chi2", SED_REDUCED_CHI2_MAX),
        ("spectroscopy_reduced_chi2", SPECTROSCOPY_REDUCED_CHI2_MAX),
        ("joint_reduced_chi2", JAXSEDFIT_JOINT_REDUCED_CHI2_MAX),
        ("loo_chi2_eff", LOO_CHI2_EFF_MAX),
    )
    for column, upper in cases:
        values = [upper, np.nextafter(upper, np.inf), np.nan, np.inf, -np.inf]
        mask = _scalar_parameter_cut_mask(
            pd.DataFrame({column: values}), column, None, upper
        )
        np.testing.assert_array_equal(mask, [True, False, False, False, False])


def test_explicit_zero_num_divergences_cut_requires_exactly_zero():
    values = [0, 1, -1, np.nan, np.inf, -np.inf]
    mask = _scalar_parameter_cut_mask(
        pd.DataFrame({"num_divergences": values}),
        "num_divergences",
        0.0,
        0.0,
    )
    np.testing.assert_array_equal(mask, [True, False, False, False, False, False])


def _divergence_cut_from_environment(value):
    env = os.environ.copy()
    env.pop("QVC_CUT_NUM_DIVERGENCES_MAX", None)
    if value is not None:
        env["QVC_CUT_NUM_DIVERGENCES_MAX"] = value
    script = (
        "from qvc.hubble.cuts import NUM_DIVERGENCES_MAX, AGN_TIER1_FIT_QUALITY_CUTS; "
        "print(repr((NUM_DIVERGENCES_MAX, [cut for cut in "
        "AGN_TIER1_FIT_QUALITY_CUTS if cut[0] == 'num_divergences'])))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=SRC,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return literal_eval(completed.stdout.strip())


def test_num_divergences_cut_defaults_off_and_can_be_enabled_explicitly():
    assert _divergence_cut_from_environment(None) == (None, [])
    assert _divergence_cut_from_environment("none") == (None, [])
    assert _divergence_cut_from_environment("0") == (
        0.0,
        [("num_divergences", 0.0, 0.0)],
    )


def test_rhat_cuts_reject_missing_and_bad_finite_values():
    cases = (
        ("m_2500_dereddened_rhat", None, SPECTRAL_RHAT_MAX),
        ("m_2500_attenuated_model_rhat", None, SPECTRAL_RHAT_MAX),
        ("log_tau_uv_rf_rhat", None, LIGHT_CURVE_RHAT_MAX),
        ("log_sigma_uv_rhat", None, LIGHT_CURVE_RHAT_MAX),
    )
    assert ALLOW_MISSING_SCALAR_CUT_COLUMNS == set()

    for column, lower, upper in cases:
        boundary = lower if lower is not None else upper
        rejected = (
            np.nextafter(lower, -np.inf)
            if lower is not None
            else np.nextafter(upper, np.inf)
        )
        values = [boundary, rejected, np.nan, np.inf, -np.inf]
        mask = _scalar_parameter_cut_mask(
            pd.DataFrame({column: values}), column, lower, upper
        )
        np.testing.assert_array_equal(mask, [True, False, False, False, False])


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


def test_populate_spectra_fit_preserves_nonconflicting_columns_and_spectra_hdf5_wins(
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

    assert out.loc[0, "z"] == row["z"]
    assert out.loc[0, "pl_slope"] == row["pl_slope"]
    assert out.loc[0, "alpha_lambda"] == -0.75
    assert out.loc[0, "alpha_lambda_err"] == 0.03
    assert out.loc[0, "alpha_nu"] == -1.25
    assert out.loc[0, "alpha_nu_err"] == 0.03
    assert out.loc[0, "new_sed_parameter"] == 42.5
    assert out.loc[0, "new_sed_label"] == "well_constrained"
    assert not any(column.endswith(("_x", "_y", "_sedfit")) for column in out.columns)
    assert "new_sed_parameter" in out.attrs["spectra_fit_columns"]
    assert "new_sed_label" in out.attrs["spectra_fit_columns"]
    assert "z" in out.attrs["spectra_fit_columns"]
    assert "pl_slope" in out.attrs["spectra_fit_columns"]
    assert out.attrs["spectra_fit_discarded_columns"] == ()
    assert out.attrs["light_curve_discarded_columns"] == ("pl_slope", "z")


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
            "tier": "assembly",
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
        "tier",
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
        "assembly",
        "agn_scalar:dummy",
        "dummy criterion",
        "5",
        "4",
        "2",
        "2",
        "1",
        "applied",
    ]
