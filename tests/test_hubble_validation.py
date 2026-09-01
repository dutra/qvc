import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import h5py
import pytest
from astropy.cosmology import FlatLambdaCDM


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble import hubble_validation
from qvc.hubble.hubble_model import get_model_params
from qvc.hubble.hubble_validation import (
    ARM_NAMES,
    ValidationTruth,
    analytic_completeness_params,
    apply_sigmoid_selection,
    collect_recovery_fragments,
    derive_seed_ledger,
    ensemble_summary,
    generate_matched_fit_catalogs,
    inject_catalog_observables,
    incomplete_recovery_report,
    posterior_summary_row,
    project_absolute_magnitude_to_predictors,
    write_completeness_parent_hdf5,
)


def _load_plot_module():
    path = ROOT / "scripts" / "plot_hubble_validation.py"
    spec = importlib.util.spec_from_file_location("plot_hubble_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_runner_module():
    path = ROOT / "scripts" / "run_hubble_validation.py"
    spec = importlib.util.spec_from_file_location("run_hubble_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_ledger_is_reproducible_and_streams_are_distinct():
    first = derive_seed_ledger(1234, 7)
    second = derive_seed_ledger(1234, 7)
    different = derive_seed_ledger(1234, 8)
    assert first == second
    assert first != different
    assert len(set(first.values())) == len(first)


def test_projection_exactly_encodes_the_injected_relation():
    truth = ValidationTruth()
    absolute_magnitude = np.linspace(-27.0, -19.0, 101)
    orthogonal = np.random.default_rng(4).normal(size=absolute_magnitude.size)
    log_sigma, log_tau = project_absolute_magnitude_to_predictors(
        absolute_magnitude, orthogonal, truth
    )
    recovered = (
        truth.m0_agn
        + truth.alpha_agn * (log_sigma - truth.log_sigma_pivot)
        + truth.beta_agn * (log_tau - truth.log_tau_pivot)
    )
    np.testing.assert_allclose(recovered, absolute_magnitude, rtol=0.0, atol=2e-14)


def test_injected_catalog_scatter_terms_and_schema_are_consistent():
    truth = ValidationTruth()
    cosmology = FlatLambdaCDM(H0=truth.h0, Om0=truth.om0)
    redshift = np.linspace(0.2, 3.9, 64)
    absolute_magnitude = np.linspace(-26.0, -20.0, 64)
    frame = inject_catalog_observables(
        redshift,
        absolute_magnitude,
        truth=truth,
        rng=np.random.default_rng(10),
        cosmology=cosmology,
    )
    residual = (
        frame["apparent_mag_2500"].to_numpy()
        - absolute_magnitude
        - cosmology.distmod(redshift).value
    )
    expected = (
        frame["injected_intrinsic_residual_mag"].to_numpy()
        + frame["injected_lensing_residual_mag"].to_numpy()
    )
    np.testing.assert_allclose(residual, expected, rtol=0.0, atol=1e-13)
    assert np.all(frame["log_sigma_uv_std_psd"] == 0.0)
    assert np.all(frame["log_tau_uv_rf_std_psd"] == 0.0)
    assert frame["object_id"].is_unique


def test_sigmoid_selection_reuses_recorded_uniforms():
    truth = ValidationTruth()
    cosmology = FlatLambdaCDM(H0=truth.h0, Om0=truth.om0)
    frame = inject_catalog_observables(
        np.full(100, 1.0),
        np.full(100, -22.0),
        truth=truth,
        rng=np.random.default_rng(1),
        cosmology=cosmology,
    )
    annotated, selected = apply_sigmoid_selection(
        frame, m50=23.0, width=0.3, rng=np.random.default_rng(2)
    )
    expected = (
        annotated["injected_detection_uniform"]
        < annotated["injected_detection_probability"]
    )
    np.testing.assert_array_equal(annotated["injected_detected"], expected)
    assert selected["object_id"].tolist() == annotated.loc[expected, "object_id"].tolist()


def test_matched_catalogs_have_exact_sizes_and_selected_ids(monkeypatch):
    truth = ValidationTruth()
    cosmology = FlatLambdaCDM(H0=truth.h0, Om0=truth.om0)
    counter = {"value": 0}

    def fake_sample(*args, **kwargs):
        del args, kwargs
        start = counter["value"]
        counter["value"] += 60
        redshift = np.linspace(0.2, 3.8, 60)
        absolute = np.linspace(-26.0, -20.0, 60) + 0.0 * start
        return redshift, absolute

    monkeypatch.setattr(hubble_validation, "sample_lf_chunk", fake_sample)
    all_frame, selected = generate_matched_fit_catalogs(
        object(),
        cosmology,
        truth=truth,
        n_fit=25,
        m50=100.0,
        selection_width=0.3,
        population_rng=np.random.default_rng(3),
        scatter_rng=np.random.default_rng(4),
        selection_rng=np.random.default_rng(5),
        area_deg2=1.0,
    )
    assert len(all_frame) == len(selected) == 25
    assert all_frame["object_id"].tolist() == selected["object_id"].tolist()
    assert all_frame.attrs["n_parent_generated"] == 60
    assert all_frame.attrs["n_detected_generated"] == 60


def test_analytic_oracle_and_fixed_h0_are_exact():
    model, magnitude_grid, _, _, _ = analytic_completeness_params(23.0, 0.3)
    np.testing.assert_allclose(model(np.array([23.0])), 0.5, rtol=0.0, atol=1e-15)
    assert magnitude_grid[0] <= 10.0 and magnitude_grid[-1] >= 35.0
    priors, labels, _ = get_model_params(
        "Flatw0waCDM", only_agn=True, fixed_h0=70.0
    )
    assert "H0" in labels
    assert priors["H0"] == (70.0, 70.0)


def test_completeness_parent_sentinels_do_not_enter_map_magnitude_bins(tmp_path):
    frame = pd.DataFrame(
        {"apparent_mag_2500": [20.0, 23.0], "z": [0.5, 3.0]}
    )
    path = write_completeness_parent_hdf5(frame, tmp_path / "parent.h5")
    with h5py.File(path, "r") as handle:
        magnitude = handle["apparent_mag_2500"][:]
        redshift = handle["z"][:]
        assert handle.attrs["support_sentinels_outside_magnitude_map"] == 2
    assert set(magnitude[-2:]) == {17.0, 100.0}
    assert set(redshift[-2:]) == {0.0, 4.5}


def _synthetic_recovery(truth: ValidationTruth, n_runs=12):
    labels = ("M0_agn", "alpha_agn", "beta_agn", "log_f", "H0", "Om0", "w0", "wa")
    center = np.array([-23.0, 7.0, -1.0, np.log(0.5), 70.0, 0.3, -1.0, 0.0])
    rows = []
    for realization in range(n_runs):
        for arm_index, arm in enumerate(ARM_NAMES):
            rng = np.random.default_rng(1000 + 10 * realization + arm_index)
            samples = center + rng.normal(0.0, [0.1, 0.15, 0.08, 0.04, 0.0, 0.02, 0.1, 0.3], size=(200, len(center)))
            rows.append(
                posterior_summary_row(
                    samples,
                    labels,
                    arm=arm,
                    realization=realization,
                    checkpoint_file=Path(f"posterior_{arm}.h5"),
                    truth=truth,
                    n_fit=2000,
                    n_parent_generated=10000,
                    detection_fraction=0.2,
                )
            )
    return pd.DataFrame(rows)


def test_ensemble_summary_and_corner_use_one_median_per_fit(tmp_path):
    truth = ValidationTruth()
    recovery = _synthetic_recovery(truth)
    summary = ensemble_summary(recovery)
    alpha = summary.loc[
        (summary["arm"] == "all") & (summary["parameter"] == "alpha_agn")
    ].iloc[0]
    assert alpha["n_success"] == 12
    assert 0.0 <= alpha["coverage_68"] <= 1.0

    plot_module = _load_plot_module()
    output_pdf = tmp_path / "corner.pdf"
    output_png = tmp_path / "corner.png"
    plot_module.plot_median_recovery_corner(
        recovery,
        {
            "alpha_agn": truth.alpha_agn,
            "beta_agn": truth.beta_agn,
            "Om0": truth.om0,
            "w0": truth.w0,
            "wa": truth.wa,
        },
        output_pdf,
        output_png=output_png,
        min_contour_points=8,
    )
    assert output_pdf.is_file() and output_pdf.stat().st_size > 0
    assert output_png.is_file() and output_png.stat().st_size > 0


def test_plot_script_reads_persisted_campaign_without_posteriors(tmp_path):
    truth = ValidationTruth()
    recovery = _synthetic_recovery(truth, n_runs=9)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    recovery.to_csv(campaign / "recovery.csv", index=False)
    manifest = {"configuration": {"truth": {
        "alpha_agn": truth.alpha_agn,
        "beta_agn": truth.beta_agn,
        "om0": truth.om0,
        "w0": truth.w0,
        "wa": truth.wa,
    }}}
    (campaign / "manifest.json").write_text(json.dumps(manifest))
    plot_module = _load_plot_module()
    assert plot_module.main([str(campaign)]) == 0
    assert (campaign / "plots" / "median_recovery_corner.pdf").is_file()
    assert (campaign / "ensemble_summary.csv").is_file()


def test_campaign_resume_rejects_configuration_drift(tmp_path):
    runner = _load_runner_module()
    campaign = tmp_path / "campaign"
    configuration = {"schema_version": 1, "truth": {"alpha_agn": 7.0}}
    runner._write_or_validate_manifest(campaign, configuration, resume=False)
    runner._write_or_validate_manifest(campaign, configuration, resume=True)
    with pytest.raises(RuntimeError, match="does not match"):
        runner._write_or_validate_manifest(
            campaign,
            {"schema_version": 1, "truth": {"alpha_agn": 8.0}},
            resume=True,
        )


def test_runner_accepts_configurable_agn_count():
    runner = _load_runner_module()
    assert runner._parser().parse_args(["--n-agn", "123"]).n_agn == 123
    assert runner._parser().parse_args(["--num-agns", "456"]).n_agn == 456


def test_initialize_only_writes_manifest_and_complete_seed_ledger(tmp_path, monkeypatch):
    runner = _load_runner_module()
    monkeypatch.setattr(
        runner,
        "build_completeness_lf",
        lambda *args, **kwargs: pytest.fail("initialize-only must not build the LF"),
    )
    output_root = tmp_path / "results"
    assert runner.main([
        "--campaign", "init",
        "--output-root", str(output_root),
        "--n-runs", "3",
        "--seed-start", "4",
        "--initialize-only",
    ]) == 0
    campaign = output_root / "init"
    assert (campaign / "manifest.json").is_file()
    ledger = pd.read_csv(campaign / "seed_ledger.csv")
    assert ledger["realization"].tolist() == [4, 5, 6]
    assert not (campaign / "runs").exists()


def test_realization_selector_rejects_out_of_range_before_lf_build(tmp_path, monkeypatch):
    runner = _load_runner_module()
    monkeypatch.setattr(
        runner,
        "build_completeness_lf",
        lambda *args, **kwargs: pytest.fail("invalid realization must fail before LF construction"),
    )
    with pytest.raises(ValueError, match="outside the configured range 10-11"):
        runner.main([
            "--campaign", "invalid-realization",
            "--output-root", str(tmp_path),
            "--n-runs", "2",
            "--seed-start", "10",
            "--realization", "12",
        ])


def test_distributed_runner_writes_only_requested_seed_fragment(tmp_path, monkeypatch):
    runner = _load_runner_module()
    monkeypatch.setattr(runner, "build_completeness_lf", lambda *args, **kwargs: object())
    empty = pd.DataFrame({"object_id": ["object-1"]})
    monkeypatch.setattr(
        runner,
        "_load_or_generate_catalogs",
        lambda *args, **kwargs: (
            empty,
            empty,
            None,
            {"n_parent_generated": 1, "detection_fraction": 1.0},
        ),
    )

    def fake_fit(arm, *, realization, **kwargs):
        return {
            "realization": realization,
            "arm": arm,
            "status": "complete",
            "checkpoint_file": f"posterior_{arm}.h5",
            "posterior_sample_count": 20,
        }

    monkeypatch.setattr(runner, "_fit_arm", fake_fit)
    assert runner.main([
        "--campaign", "distributed",
        "--output-root", str(tmp_path),
        "--n-runs", "3",
        "--realization", "1",
        "--arms", "all", "selected_oracle",
    ]) == 0
    campaign = tmp_path / "distributed"
    fragment = pd.read_csv(campaign / "runs/seed_0001/recovery.csv")
    assert fragment[["realization", "arm"]].to_records(index=False).tolist() == [
        (1, "all"),
        (1, "selected_oracle"),
    ]
    assert not (campaign / "runs/seed_0000").exists()
    assert not (campaign / "runs/seed_0002").exists()
    assert not (campaign / "recovery.csv").exists()


def test_distributed_failure_is_recorded_skipped_and_retried(tmp_path, monkeypatch):
    runner = _load_runner_module()
    monkeypatch.setattr(runner, "build_completeness_lf", lambda *args, **kwargs: object())
    frame = pd.DataFrame({"object_id": ["object-1"]})
    monkeypatch.setattr(
        runner,
        "_load_or_generate_catalogs",
        lambda *args, **kwargs: (
            frame,
            frame,
            None,
            {"n_parent_generated": 1, "detection_fraction": 1.0},
        ),
    )
    attempts = []

    def fake_fit(arm, *, realization, **kwargs):
        attempts.append((realization, arm))
        if len(attempts) == 1:
            raise RuntimeError("intentional failure")
        return {
            "realization": realization,
            "arm": arm,
            "status": "complete",
            "checkpoint_file": f"posterior_{arm}.h5",
            "posterior_sample_count": 20,
        }

    monkeypatch.setattr(runner, "_fit_arm", fake_fit)
    base = [
        "--campaign", "retry",
        "--output-root", str(tmp_path),
        "--n-runs", "1",
        "--realization", "0",
        "--arms", "all",
    ]
    assert runner.main(base) == 0
    fragment_path = tmp_path / "retry/runs/seed_0000/recovery.csv"
    assert pd.read_csv(fragment_path).iloc[0]["status"] == "failed"
    assert runner.main([*base, "--resume"]) == 0
    assert attempts == [(0, "all")]
    assert runner.main([*base, "--resume", "--retry-failed"]) == 0
    assert attempts == [(0, "all"), (0, "all")]
    assert pd.read_csv(fragment_path).iloc[0]["status"] == "complete"


def test_fragment_aggregation_and_incomplete_report_are_deterministic(tmp_path):
    campaign = tmp_path / "campaign"
    seed0 = campaign / "runs/seed_0000"
    seed1 = campaign / "runs/seed_0001"
    seed0.mkdir(parents=True)
    seed1.mkdir(parents=True)
    pd.DataFrame([
        {"realization": 0, "arm": "all", "status": "complete"},
        {
            "realization": 0,
            "arm": "selected_oracle",
            "status": "failed",
            "error_type": "RuntimeError",
            "error_message": "fit failed",
        },
    ]).to_csv(seed0 / "recovery.csv", index=False)
    pd.DataFrame([
        {"realization": 1, "arm": "all", "status": "complete"},
    ]).to_csv(seed1 / "recovery.csv", index=False)

    recovery = collect_recovery_fragments(campaign)
    assert recovery[["realization", "arm"]].to_records(index=False).tolist() == [
        (0, "all"),
        (0, "selected_oracle"),
        (1, "all"),
    ]
    pd.testing.assert_frame_equal(recovery, pd.read_csv(campaign / "recovery.csv"))
    report = incomplete_recovery_report(
        recovery,
        {"seed_start": 0, "n_runs": 2, "arms": ["all", "selected_oracle"]},
    )
    assert report[["realization", "arm", "status"]].to_records(index=False).tolist() == [
        (0, "selected_oracle", "failed"),
        (1, "selected_oracle", "missing"),
    ]


def test_fragment_aggregation_preserves_unmigrated_legacy_rows(tmp_path):
    campaign = tmp_path / "campaign"
    fragment_dir = campaign / "runs/seed_0000"
    fragment_dir.mkdir(parents=True)
    pd.DataFrame([
        {"realization": 0, "arm": "all", "status": "complete", "marker": "new"},
    ]).to_csv(fragment_dir / "recovery.csv", index=False)
    pd.DataFrame([
        {"realization": 0, "arm": "all", "status": "failed", "marker": "old"},
        {"realization": 1, "arm": "all", "status": "complete", "marker": "legacy"},
    ]).to_csv(campaign / "recovery.csv", index=False)
    recovery = collect_recovery_fragments(campaign)
    assert recovery[["realization", "marker"]].to_records(index=False).tolist() == [
        (0, "new"),
        (1, "legacy"),
    ]


def test_plot_aggregates_partial_fragments_and_reports_missing_fits(tmp_path):
    truth = ValidationTruth()
    campaign = tmp_path / "campaign"
    fragment_dir = campaign / "runs/seed_0000"
    fragment_dir.mkdir(parents=True)
    recovery = _synthetic_recovery(truth, n_runs=2)
    recovery = recovery.loc[
        (recovery["realization"] == 0) & (recovery["arm"] == "all")
    ]
    recovery.to_csv(fragment_dir / "recovery.csv", index=False)
    manifest = {
        "configuration": {
            "truth": {
                "alpha_agn": truth.alpha_agn,
                "beta_agn": truth.beta_agn,
                "om0": truth.om0,
                "w0": truth.w0,
                "wa": truth.wa,
            },
            "seed_start": 0,
            "n_runs": 2,
            "arms": ["all", "selected_oracle"],
        }
    }
    (campaign / "manifest.json").write_text(json.dumps(manifest))
    plot_module = _load_plot_module()
    assert plot_module.main([str(campaign)]) == 0
    incomplete = pd.read_csv(campaign / "incomplete_fits.csv")
    assert len(incomplete) == 3
    assert set(incomplete["status"]) == {"missing"}
    assert (campaign / "recovery.csv").is_file()
    assert (campaign / "plots/median_recovery_corner.pdf").is_file()


def test_plot_reports_incomplete_campaign_before_failing_without_successes(tmp_path):
    truth = ValidationTruth()
    campaign = tmp_path / "campaign"
    fragment_dir = campaign / "runs/seed_0000"
    fragment_dir.mkdir(parents=True)
    pd.DataFrame([
        {
            "realization": 0,
            "arm": "all",
            "status": "failed",
            "error_type": "RuntimeError",
            "error_message": "fit failed",
        }
    ]).to_csv(fragment_dir / "recovery.csv", index=False)
    manifest = {
        "configuration": {
            "truth": {
                "alpha_agn": truth.alpha_agn,
                "beta_agn": truth.beta_agn,
                "om0": truth.om0,
                "w0": truth.w0,
                "wa": truth.wa,
            },
            "seed_start": 0,
            "n_runs": 1,
            "arms": ["all", "selected_oracle"],
        }
    }
    (campaign / "manifest.json").write_text(json.dumps(manifest))
    plot_module = _load_plot_module()
    with pytest.raises(ValueError, match="no successful fits"):
        plot_module.main([str(campaign)])
    report = pd.read_csv(campaign / "incomplete_fits.csv")
    assert report[["arm", "status"]].to_records(index=False).tolist() == [
        ("all", "failed"),
        ("selected_oracle", "missing"),
    ]
