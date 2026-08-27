import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from qvc.hubble.completeness_strata import (
    COMPLETENESS_STRATUM_CODE_COL,
    COMPLETENESS_STRATUM_COL,
    StratifiedCompletenessBundle,
    assign_completeness_strata,
    get_completeness_stratification_preset,
    make_stratified_dm_function,
)
from qvc.hubble.hubble_completeness_refactored import (
    COMPLETENESS_MAG_COL,
    Completeness2D,
    evaluate_dm_interp,
)
from qvc.hubble.hubble_fit import (
    _completeness_stratification_checkpoint_payload,
    _validate_completeness_stratification_checkpoint,
    make_run_tag,
)
from qvc.hubble.hubble_fit_jax import (
    _completeness_loglike_jax,
    _prepare_completeness_for_jax,
)
from qvc.hubble.hubble_likelihood import (
    completeness_loglike,
    completeness_loglike_for_data,
)
from qvc.hubble import completeness_closure


def _target_frame():
    return pd.DataFrame(
        {
            "object_id": [
                "legacy",
                "boss",
                "var",
                "var_core",
                "core_only",
                "special",
                "invalid",
                "unmatched",
            ],
            "SDSS_SURVEY": [
                b" SDSS ",
                "BOSS",
                "eboss",
                b"EBOSS",
                "eboss",
                "eboss",
                "eboss",
                "eboss",
            ],
            "SDSS_SPECOBJ_MATCHED": [1, 1, 1, 1, 1, 1, 1, 0],
            "SDSS_EBOSS_TARGET0": [0, 0, 0, 0, 0, 1, -1, 0],
            "SDSS_EBOSS_TARGET1": [
                0,
                0,
                1 << 9,
                (1 << 9) | (1 << 10),
                1 << 10,
                1 << 9,
                1 << 9,
                1 << 9,
            ],
            "SDSS_EBOSS_TARGET2": 0,
        }
    )


def test_clean_var_core_assignment_is_exact_and_excludes_remainders():
    assignment = assign_completeness_strata(
        _target_frame(), "sdss_clean_var_core"
    )
    retained_ids = _target_frame().loc[assignment.retained_mask, "object_id"].tolist()
    assert retained_ids == ["legacy", "boss", "var", "var_core"]
    assert assignment.labels[assignment.retained_mask].tolist() == [
        "legacy-sdss",
        "boss",
        "eboss-var-not-core",
        "eboss-var-and-core",
    ]
    assert assignment.codes[assignment.retained_mask].tolist() == [0, 1, 2, 3]


def test_survey_var_assignment_partitions_all_valid_eboss_rows():
    assignment = assign_completeness_strata(_target_frame(), "sdss-survey-var")
    labels = dict(
        zip(
            _target_frame().loc[assignment.retained_mask, "object_id"],
            assignment.labels[assignment.retained_mask],
        )
    )
    assert labels["var"] == "eboss-var-s82-inclusive"
    assert labels["var_core"] == "eboss-var-s82-inclusive"
    assert labels["core_only"] == "eboss-non-var-s82"
    assert labels["special"] == "eboss-var-s82-inclusive"
    assert "invalid" not in labels
    assert "unmatched" not in labels


def test_survey_var_no_boss_assignment_excludes_boss_and_partitions_eboss():
    assignment = assign_completeness_strata(
        _target_frame(), "sdss_survey_var_no_boss"
    )
    labels = dict(
        zip(
            _target_frame().loc[assignment.retained_mask, "object_id"],
            assignment.labels[assignment.retained_mask],
        )
    )
    assert labels == {
        "legacy": "legacy-sdss",
        "var": "eboss-var-s82-inclusive",
        "var_core": "eboss-var-s82-inclusive",
        "core_only": "eboss-non-var-s82",
        "special": "eboss-var-s82-inclusive",
    }
    assert assignment.codes[assignment.retained_mask].tolist() == [0, 1, 1, 2, 1]
    assert "boss" not in labels
    assert assignment.stratum_names == (
        "legacy-sdss",
        "eboss-var-s82-inclusive",
        "eboss-non-var-s82",
    )


def test_assignment_requires_metadata_and_nonempty_strata():
    with pytest.raises(ValueError, match="SDSS_SURVEY"):
        assign_completeness_strata(pd.DataFrame({"object_id": ["x"]}), "sdss-survey")
    with pytest.raises(ValueError, match="empty stratum"):
        assign_completeness_strata(_target_frame().iloc[:1], "sdss-survey")


def _two_map_bundle():
    mag = np.linspace(18.5, 24.0, 12)
    redshift = np.linspace(0.4, 3.2, 8)
    model0 = Completeness2D(
        mag, redshift, np.full((len(mag), len(redshift)), 0.8), (18.5, 24.0)
    )
    model1 = Completeness2D(
        mag, redshift, np.full((len(mag), len(redshift)), 0.25), (18.5, 24.0)
    )
    preset = get_completeness_stratification_preset("sdss-survey")
    # The likelihood only depends on the model and magnitude grid; the tail
    # preserves the existing 2D completeness tuple contract.
    params0 = (model0, mag, redshift, 0.5, 0.2, 0.1)
    params1 = (model1, mag, redshift, 0.5, 0.2, 0.1)
    return StratifiedCompletenessBundle(
        preset_name="test-two-map",
        definition_json=preset.canonical_json(),
        stratum_names=("a", "b"),
        params_by_stratum=(params0, params1),
    )


def test_numpy_stratified_likelihood_matches_manual_groups_and_order():
    bundle = _two_map_bundle()
    codes = np.array([1, 0, 1, 0])
    arrays = {
        "m_obs": np.array([20.0, 20.2, 20.4, 20.6]),
        "m_obs_err": np.full(4, 0.1),
        "m_model": np.array([20.1, 20.1, 20.5, 20.5]),
        "mu_err": np.full(4, 0.3),
        "z": np.array([0.6, 1.0, 1.4, 2.0]),
    }
    total, blob = completeness_loglike_for_data(
        completeness_params=bundle,
        agn_data={COMPLETENESS_STRATUM_CODE_COL: codes},
        **arrays,
    )
    manual_total = 0.0
    manual_blob = np.zeros_like(blob)
    for code, params in enumerate(bundle.params_by_stratum):
        mask = codes == code
        group_total, group_blob = completeness_loglike(
            **{key: value[mask] for key, value in arrays.items()},
            completeness_model=params[0],
            m_grid=params[1],
            magnitude_support=params[0].magnitude_support,
            sigma_completeness=0.0,
        )
        manual_total += group_total
        manual_blob[:, mask] = group_blob
    assert total == pytest.approx(manual_total)
    np.testing.assert_allclose(blob, manual_blob)


def test_jax_stratified_2d_gathers_each_objects_map():
    pytest.importorskip("jax")
    bundle = _two_map_bundle()
    codes = np.array([1, 0, 1, 0], dtype=np.int16)
    m_model = np.array([20.1, 20.1, 20.5, 20.5])
    mu_err = np.full(4, 0.3)
    z = np.array([0.6, 1.0, 1.4, 2.0])
    prepared = _prepare_completeness_for_jax(
        bundle, selection_magnitude=np.full(4, 20.2)
    )
    actual = float(
        _completeness_loglike_jax(
            m_model, mu_err, z, prepared, codes
        )
    )
    expected = 0.0
    for code, params in enumerate(bundle.params_by_stratum):
        mask = codes == code
        single = _prepare_completeness_for_jax(params)
        expected += float(
            _completeness_loglike_jax(
                m_model[mask], mu_err[mask], z[mask], single
            )
        )
    assert actual == pytest.approx(expected, abs=1e-10)


def test_numpy_and_jax_likelihood_extend_center_values_to_support_edges():
    pytest.importorskip("jax")
    magnitude = np.array([19.0, 20.0, 21.0])
    redshift = np.array([0.5, 1.5])
    values = np.array(
        [
            [0.2, 0.3],
            [0.5, 0.6],
            [0.8, 0.9],
        ]
    )
    model = Completeness2D(
        magnitude,
        redshift,
        values,
        magnitude_support=(18.5, 21.5),
    )
    m_obs = np.array([18.5, 21.5])
    m_model = np.array([18.8, 21.2])
    mu_err = np.array([0.3, 0.4])
    z = np.array([0.5, 0.5])

    numpy_loglike, _ = completeness_loglike(
        m_obs=m_obs,
        m_obs_err=np.full(2, 0.1),
        m_model=m_model,
        mu_err=mu_err,
        z=z,
        completeness_model=model,
        m_grid=magnitude,
        magnitude_support=model.magnitude_support,
    )
    prepared = _prepare_completeness_for_jax(
        (model, magnitude),
        selection_magnitude=m_obs,
    )
    jax_loglike = float(
        _completeness_loglike_jax(m_model, mu_err, z, prepared)
    )

    assert jax_loglike == pytest.approx(numpy_loglike, rel=2e-6, abs=2e-6)


def test_stratified_interpolation_requires_labels_and_dispatches():
    frame = pd.DataFrame(
        {
            COMPLETENESS_MAG_COL: [20.0, 20.2, 20.0, 20.2],
            "z": [0.8, 1.0, 0.8, 1.0],
            COMPLETENESS_STRATUM_COL: ["a", "a", "b", "b"],
        }
    )
    interp = make_stratified_dm_function(frame, np.array([0.1, 0.1, 0.5, 0.5]))
    with pytest.raises(ValueError, match="requires completeness_stratum"):
        evaluate_dm_interp(interp, frame["z"], frame[COMPLETENESS_MAG_COL])
    actual = evaluate_dm_interp(
        interp,
        frame["z"],
        frame[COMPLETENESS_MAG_COL],
        completeness_stratum=frame[COMPLETENESS_STRATUM_COL],
    )
    np.testing.assert_allclose(actual, [0.1, 0.1, 0.5, 0.5])


def test_run_tag_and_checkpoint_metadata_are_stratification_safe():
    plain = make_run_tag(
        "FlatLambdaCDM", False, "quick", None, (0.44, 3.16)
    )
    active = make_run_tag(
        "FlatLambdaCDM",
        False,
        "quick",
        None,
        (0.44, 3.16),
        completeness_stratification="sdss-survey",
    )
    assert "_cstrat-" not in plain
    assert "_cstrat-sdss-survey" in active

    frame = pd.DataFrame(
        {
            COMPLETENESS_STRATUM_COL: ["legacy-sdss", "boss", "eboss"],
            COMPLETENESS_STRATUM_CODE_COL: [0, 1, 2],
        }
    )
    payload = _completeness_stratification_checkpoint_payload(
        "sdss-survey", frame
    )
    _validate_completeness_stratification_checkpoint(
        payload,
        checkpoint_file="memory.h5",
        expected_stratification="sdss-survey",
        expected_codes=np.array([0, 1, 2]),
    )
    with pytest.raises(RuntimeError, match="assignments do not match"):
        _validate_completeness_stratification_checkpoint(
            payload,
            checkpoint_file="memory.h5",
            expected_stratification="sdss-survey",
            expected_codes=np.array([0, 2, 1]),
        )
    # Legacy checkpoints remain valid only for the unstratified model.
    _validate_completeness_stratification_checkpoint(
        {}, checkpoint_file="legacy.h5", expected_stratification="none"
    )


def test_runner_forwards_stratification_and_sweep_is_controlled():
    root = Path(__file__).resolve().parents[1]
    runner = (root / "run_hubble.xonsh").read_text(encoding="utf-8")
    assert '"QVC_HUBBLE_COMPLETENESS_STRATIFICATION", "none"' in runner
    assert "--completeness-stratification @(completeness_stratification)" in runner

    sweep = (
        root / "run_hubble_completeness_stratification_sweep.xonsh"
    ).read_text(encoding="utf-8")
    for preset in (
        "none",
        "sdss-survey",
        "sdss-survey-var",
        "sdss-survey-var-no-boss",
        "sdss-clean-var-core",
    ):
        assert f'"{preset}"' in sweep
    assert '$QVC_HUBBLE_Z_MIN = "0.44"' in sweep
    assert '$QVC_HUBBLE_Z_MAX = "3.16"' in sweep
    assert '$QVC_HUBBLE_SDSS_TARGET_SELECTION = "all"' in sweep
    assert "$QVC_HUBBLE_COMPLETENESS_STRATIFICATION = preset" in sweep
    assert "rhat_reprocessed_sdss_metadata.h5" in sweep
    assert "QVC_HUBBLE_RESUME cannot be shared" in sweep
    assert "$RAISE_SUBPROC_ERROR = True" in sweep


def test_posterior_closure_runs_and_reports_each_stratum(monkeypatch, tmp_path):
    bundle = _two_map_bundle()
    agn_data = {
        "z": np.array([0.7, 1.0, 1.5, 2.0]),
        COMPLETENESS_STRATUM_CODE_COL: np.array([0, 0, 1, 1]),
    }

    monkeypatch.setattr(
        completeness_closure,
        "agn_selection_prediction",
        lambda *args, **kwargs: {
            "selection_model_magnitude": np.full(4, 20.5),
            "selection_total_error": np.full(4, 0.25),
        },
    )
    result = completeness_closure.simulate_hubble_posterior_closure(
        posterior_samples=np.ones((40, 1)),
        agn_data=agn_data,
        cosmo_model="FlatLambdaCDM",
        z_pivot_agn=1.5,
        agn_pivot_context=None,
        completeness_params=bundle,
        redshift_bins=np.array([0.4, 3.2]),
        max_posterior_draws=40,
        min_detected_per_bin=1,
        seed=7,
    )
    assert result.summary[COMPLETENESS_STRATUM_COL].tolist() == ["a", "b"]
    paths = completeness_closure.write_completeness_closure_diagnostics(
        result, tmp_path
    )
    assert paths["summary_csv"].is_file()
    assert set(paths["per_stratum_plot_pdfs"]) == {"a", "b"}
    assert all(path.is_file() for path in paths["per_stratum_plot_pdfs"].values())
