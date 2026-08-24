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
    AGN_SCALAR_PARAMETER_CUTS,
    ALLOW_MISSING_SCALAR_CUT_COLUMNS,
    APPARENT_MAG_2500_ERR_MAX,
    COMPLETENESS_MAG_2500_MIN,
    COMPLETENESS_MAG_2500_MAX,
    EXCLUDED_SDSS_NAMES,
    FRAC_AGN_5100_MIN,
    JAXSEDFIT_JOINT_REDUCED_CHI2_MAX,
    LIGHT_CURVE_N_POINTS_COLUMN,
    LIGHT_CURVE_N_POINTS_EXCLUDED_BANDS,
    LIGHT_CURVE_RHAT_MAX,
    LOG_TAU_UV_RF_MAX,
    LOG_TAU_UV_RF_MIN,
    REL_APPARENT_MAG_2500_ERR_MAX,
    SPECTRAL_RHAT_MAX,
    add_light_curve_point_count_column,
    build_sdss_target_selection_mask,
    light_curve_point_count_series,
    normalize_sdss_target_selection,
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
from qvc.spectra.catalog_hdf5 import write_spectra_catalog_hdf5  # noqa: E402


def _write_spectra_h5(path, rows):
    frame = pd.DataFrame(rows)
    write_spectra_catalog_hdf5(
        path,
        frame,
        np.full((len(frame), 64, 5), np.nan, dtype=np.float32),
        np.zeros(len(frame), dtype=np.int16),
        f_host_2500_psf_draws=np.full(
            (len(frame), 64), np.nan, dtype=np.float32
        ),
        f_host_2500_psf_valid_count=np.zeros(len(frame), dtype=np.int16),
    )


def _sdss_target_selection_frame():
    bit = lambda index: 1 << index
    return pd.DataFrame(
        {
            "SDSS_SURVEY": [
                "eBOSS", b" EBOSS ", "eboss", "eboss",
                "boss", "eboss", "eboss", "eboss",
            ],
            "SDSS_EBOSS_TARGET0": [0, 0, 0, 0, 0, 0, 0, 0],
            "SDSS_EBOSS_TARGET1": [
                bit(9), bit(9) | bit(10), bit(9) | bit(14), bit(9),
                bit(9), -1, bit(9), 0,
            ],
            "SDSS_EBOSS_TARGET2": [0, 0, 0, bit(20), 0, 0, 0, 0],
            "SDSS_SPECOBJ_MATCHED": [True, True, True, True, True, True, False, True],
        }
    )


def test_sdss_target_selection_presets_handle_overlap_and_invalid_rows():
    frame = _sdss_target_selection_frame()

    all_mask, _ = build_sdss_target_selection_mask(frame, "all")
    inclusive, _ = build_sdss_target_selection_mask(
        frame, " EBOSS_VAR_S82_INCLUSIVE "
    )
    non_var, _ = build_sdss_target_selection_mask(frame, "eboss-non-var-s82")
    var_only, _ = build_sdss_target_selection_mask(frame, "eboss-var-s82-only")
    var_core_only, _ = build_sdss_target_selection_mask(
        frame, "eboss-var-s82-core-only"
    )

    assert np.flatnonzero(all_mask).tolist() == list(range(len(frame)))
    assert np.flatnonzero(inclusive).tolist() == [0, 1, 2, 3]
    assert np.flatnonzero(non_var).tolist() == [7]
    assert np.flatnonzero(var_only).tolist() == [0]
    assert np.flatnonzero(var_core_only).tolist() == [1]


def test_sdss_target_selection_survey_presets_are_disjoint_and_normalized():
    frame = pd.DataFrame(
        {
            "SDSS_SURVEY": [" SDSS ", "segue1", "BOSS", "eBOSS", "unknown"],
            "SDSS_SPECOBJ_MATCHED": [True, True, True, True, True],
        }
    )

    legacy, _ = build_sdss_target_selection_mask(frame, "legacy_sdss")
    boss, _ = build_sdss_target_selection_mask(frame, "boss")
    eboss, _ = build_sdss_target_selection_mask(frame, "eboss")

    assert np.flatnonzero(legacy).tolist() == [0, 1]
    assert np.flatnonzero(boss).tolist() == [2]
    assert np.flatnonzero(eboss).tolist() == [3]
    np.testing.assert_array_equal(
        legacy.astype(int) + boss.astype(int) + eboss.astype(int),
        [1, 1, 1, 1, 0],
    )


def test_sdss_target_selection_survey_presets_require_only_survey_metadata():
    frame = pd.DataFrame({"SDSS_SURVEY": ["boss", "eboss"]})
    boss, _ = build_sdss_target_selection_mask(frame, "boss")
    eboss, _ = build_sdss_target_selection_mask(frame, "eboss")
    np.testing.assert_array_equal(boss, [True, False])
    np.testing.assert_array_equal(eboss, [False, True])


def test_sdss_target_selection_requires_enriched_spectra_metadata():
    with np.testing.assert_raises_regex(ValueError, "\\*_sdss_metadata.h5"):
        build_sdss_target_selection_mask(
            pd.DataFrame({"SDSS_SURVEY": ["eboss"]}),
            "eboss-var-s82-only",
        )


def test_normalize_sdss_target_selection_rejects_unknown_name():
    assert normalize_sdss_target_selection(" EBOSS_VAR_S82_ONLY ") == "eboss-var-s82-only"
    with np.testing.assert_raises_regex(ValueError, "Unknown SDSS target selection"):
        normalize_sdss_target_selection("not-a-selection")


def test_build_agn_cuts_contains_only_fiducial_profile():
    cuts = build_agn_cuts()
    cut_map = {column: (lower, upper) for column, lower, upper in cuts}

    assert tuple(cuts) == AGN_SCALAR_PARAMETER_CUTS
    assert cut_map == {
        "log_tau_uv_rf": (1.5, 4.0),
        "fracAGN_5100_fit": (FRAC_AGN_5100_MIN, None),
        "apparent_mag_2500_err": (None, APPARENT_MAG_2500_ERR_MAX),
        "m_2500_dereddened": (
            COMPLETENESS_MAG_2500_MIN,
            COMPLETENESS_MAG_2500_MAX,
        ),
        "joint_reduced_chi2": (None, JAXSEDFIT_JOINT_REDUCED_CHI2_MAX),
        "m_2500_dereddened_rhat": (None, SPECTRAL_RHAT_MAX),
        "m_2500_attenuated_model_rhat": (None, SPECTRAL_RHAT_MAX),
        "log_tau_uv_rf_rhat": (None, LIGHT_CURVE_RHAT_MAX),
        "log_sigma_uv_rhat": (None, LIGHT_CURVE_RHAT_MAX),
    }
    assert JAXSEDFIT_JOINT_REDUCED_CHI2_MAX == 1.5
    assert SPECTRAL_RHAT_MAX == 1.20
    assert LIGHT_CURVE_RHAT_MAX == 1.10
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


def test_rhat_cuts_allow_missing_but_reject_bad_finite_values():
    cases = (
        ("m_2500_dereddened_rhat", None, SPECTRAL_RHAT_MAX),
        ("m_2500_attenuated_model_rhat", None, SPECTRAL_RHAT_MAX),
        ("log_tau_uv_rf_rhat", None, LIGHT_CURVE_RHAT_MAX),
        ("log_sigma_uv_rhat", None, LIGHT_CURVE_RHAT_MAX),
    )
    assert ALLOW_MISSING_SCALAR_CUT_COLUMNS == {column for column, _, _ in cases}

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
        np.testing.assert_array_equal(mask, [True, False, True, False, False])


def _rhat_thresholds_from_environment(overrides):
    env = os.environ.copy()
    for name in (
        "QVC_CUT_MCMC_RHAT_MAX",
        "QVC_CUT_SPECTRAL_RHAT_MAX",
        "QVC_CUT_LIGHT_CURVE_RHAT_MAX",
    ):
        env.pop(name, None)
    env.update(overrides)
    script = (
        "from qvc.hubble.cuts import LIGHT_CURVE_RHAT_MAX, SPECTRAL_RHAT_MAX; "
        "print(repr((SPECTRAL_RHAT_MAX, LIGHT_CURVE_RHAT_MAX)))"
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


def test_rhat_cut_environment_overrides_are_independent_and_allow_none():
    assert _rhat_thresholds_from_environment(
        {
            "QVC_CUT_SPECTRAL_RHAT_MAX": "1.15",
            "QVC_CUT_LIGHT_CURVE_RHAT_MAX": "1.07",
        }
    ) == (1.15, 1.07)
    assert _rhat_thresholds_from_environment(
        {
            "QVC_CUT_SPECTRAL_RHAT_MAX": "none",
            "QVC_CUT_LIGHT_CURVE_RHAT_MAX": "none",
        }
    ) == (None, None)


def test_legacy_shared_rhat_environment_override_is_ignored():
    assert _rhat_thresholds_from_environment(
        {"QVC_CUT_MCMC_RHAT_MAX": "1.01"}
    ) == (1.20, 1.10)


def _scalar_thresholds_from_environment(overrides):
    env = os.environ.copy()
    names = (
        "QVC_CUT_LOG_TAU_UV_RF_MIN",
        "QVC_CUT_LOG_TAU_UV_RF_MAX",
        "QVC_CUT_APPARENT_MAG_2500_ERR_MAX",
        "QVC_CUT_COMPLETENESS_MAG_2500_MIN",
        "QVC_CUT_COMPLETENESS_MAG_2500_MAX",
    )
    for name in names:
        env.pop(name, None)
    env.update(overrides)
    script = (
        "from qvc.hubble.cuts import ("
        "APPARENT_MAG_2500_ERR_MAX, COMPLETENESS_MAG_2500_MAX, "
        "COMPLETENESS_MAG_2500_MIN, LOG_TAU_UV_RF_MAX, LOG_TAU_UV_RF_MIN); "
        "print(repr((LOG_TAU_UV_RF_MIN, LOG_TAU_UV_RF_MAX, "
        "APPARENT_MAG_2500_ERR_MAX, COMPLETENESS_MAG_2500_MIN, "
        "COMPLETENESS_MAG_2500_MAX)))"
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


def test_scalar_cut_environment_overrides_allow_reproducible_cut_scans():
    assert _scalar_thresholds_from_environment(
        {
            "QVC_CUT_LOG_TAU_UV_RF_MIN": "1.7",
            "QVC_CUT_LOG_TAU_UV_RF_MAX": "3.2",
            "QVC_CUT_APPARENT_MAG_2500_ERR_MAX": "0.35",
            "QVC_CUT_COMPLETENESS_MAG_2500_MIN": "19.5",
            "QVC_CUT_COMPLETENESS_MAG_2500_MAX": "23.3",
        }
    ) == (1.7, 3.2, 0.35, 19.5, 23.3)
    assert _scalar_thresholds_from_environment(
        {
            "QVC_CUT_LOG_TAU_UV_RF_MIN": "none",
            "QVC_CUT_LOG_TAU_UV_RF_MAX": "none",
            "QVC_CUT_APPARENT_MAG_2500_ERR_MAX": "none",
            "QVC_CUT_COMPLETENESS_MAG_2500_MIN": "none",
            "QVC_CUT_COMPLETENESS_MAG_2500_MAX": "none",
        }
    ) == (None, None, None, None, None)


def test_total_a2500_cut_environment_override_is_inclusive():
    env = os.environ.copy()
    env["QVC_CUT_A_2500_TOTAL_MAX"] = "0.25"
    script = (
        "from qvc.hubble.hubble_cut_config import build_agn_cuts; "
        "print(repr([cut for cut in build_agn_cuts() "
        "if cut[0] == 'a_2500_total']))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=SRC,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert literal_eval(completed.stdout.strip()) == [
        ("a_2500_total", None, 0.25)
    ]
    mask = _scalar_parameter_cut_mask(
        pd.DataFrame({"a_2500_total": [0.25, np.nextafter(0.25, np.inf)]}),
        "a_2500_total",
        None,
        0.25,
    )
    np.testing.assert_array_equal(mask, [True, False])


def test_ess_diagnostics_are_not_hard_cuts():
    active_columns = {column for column, _, _ in build_agn_cuts()}
    assert "m_2500_dereddened_ess" not in active_columns
    assert "m_2500_attenuated_model_ess" not in active_columns


def test_current_spectra_schema_requires_fracagn_5100_fit(tmp_path):
    h5_path = tmp_path / "spectra.h5"
    _write_spectra_h5(h5_path, pd.DataFrame(
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
    ))

    with np.testing.assert_raises_regex(ValueError, "fracAGN_5100_fit"):
        populate_spectra_fit(pd.DataFrame({"object_id": ["obj"]}), [h5_path])


def test_current_spectra_schema_accepts_only_joint_sedfit_backend(tmp_path):
    h5_path = tmp_path / "spectra.h5"
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
    _write_spectra_h5(h5_path, [row])

    out = populate_spectra_fit(
        pd.DataFrame({"object_id": ["obj", "not-matched"]}),
        [h5_path],
    )

    assert out["object_id"].tolist() == ["obj"]
    assert out.loc[0, "fit_backend"] == "jaxsedfit_joint"
    assert out.loc[0, "alpha_lambda"] == row["pl_slope"]
    assert "PL_slope" not in out.columns
    assert "f_host_2500" not in out.columns

    row["fit_backend"] = "jaxqsofit"
    _write_spectra_h5(h5_path, [row])
    with np.testing.assert_raises_regex(ValueError, "unsupported fit_backend"):
        populate_spectra_fit(pd.DataFrame({"object_id": ["obj"]}), [h5_path])


def test_populate_spectra_fit_preserves_nonconflicting_columns_and_spectra_hdf5_wins(
    tmp_path,
):
    h5_path = tmp_path / "spectra.h5"
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
    _write_spectra_h5(h5_path, [row])
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

    out = populate_spectra_fit(source, [h5_path])

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
    h5_path = tmp_path / "spectra.h5"
    _write_spectra_h5(h5_path, pd.DataFrame(
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
    ))

    out = populate_spectra_fit(pd.DataFrame({"object_id": ["obj"]}), [h5_path])

    assert out.loc[0, "alpha_lambda"] == -1.5
    assert out.loc[0, "alpha_lambda_err"] == 0.1
    assert out.loc[0, "alpha_nu"] == -0.5
    assert out.loc[0, "alpha_nu_err"] == 0.1


def test_populate_spectra_fit_reconstructs_missing_total_a2500(tmp_path):
    h5_path = tmp_path / "spectra.h5"
    _write_spectra_h5(h5_path, pd.DataFrame(
        [
            {
                "object_id": "obj",
                "fit_ok": True,
                "fit_backend": "jaxsedfit_joint",
                "fracAGN_5100_fit": 0.8,
                "fracAGN_5100_fit_err": 0.05,
                "m_2500_dereddened": 20.0,
                "m_2500_dereddened_err": 0.1,
                "m_2500_attenuated_model": 20.25,
                "m_2500_attenuated_model_err": 0.12,
                "pl_slope": -1.5,
                "pl_slope_err": 0.1,
                "a_2500_galaxy": 0.10,
                "a_2500_internal": 0.15,
            },
            {
                "object_id": "obj2",
                "fit_ok": True,
                "fit_backend": "jaxsedfit_joint",
                "fracAGN_5100_fit": 0.9,
                "fracAGN_5100_fit_err": 0.04,
                "m_2500_dereddened": 20.5,
                "m_2500_dereddened_err": 0.1,
                "m_2500_attenuated_model": 20.57,
                "m_2500_attenuated_model_err": 0.11,
                "pl_slope": -1.7,
                "pl_slope_err": 0.08,
                "a_2500_galaxy": 0.03,
                "a_2500_internal": 0.04,
            },
        ]
    ))

    out = populate_spectra_fit(
        pd.DataFrame({"object_id": ["obj", "obj2"]}),
        [h5_path],
    )

    np.testing.assert_allclose(out["a_2500_total"], [0.25, 0.07])
    assert out.attrs["spectra_fit_columns"].count("a_2500_total") == 1


def test_populate_spectra_fit_preserves_saved_total_a2500(tmp_path):
    h5_path = tmp_path / "spectra.h5"
    _write_spectra_h5(h5_path, pd.DataFrame(
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
                "a_2500_galaxy": 0.10,
                "a_2500_internal": 0.15,
                "a_2500_total": 0.22,
            }
        ]
    ))

    out = populate_spectra_fit(pd.DataFrame({"object_id": ["obj"]}), [h5_path])

    assert out.loc[0, "a_2500_total"] == 0.22


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
