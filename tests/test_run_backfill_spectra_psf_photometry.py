from pathlib import Path

import run_backfill_spectra_psf_photometry as backfill
import run_resume_spectra_local as direct


def test_defaults_target_original_run_and_canonical_v3():
    args = backfill.parse_args([])
    assert args.source_run.name == backfill.SOURCE_RUN_NAME
    assert args.output.name == f"{backfill.SOURCE_RUN_NAME}_resumed_m2500norm12_v3.h5"
    assert args.workers == 8
    assert args.max_tasks_per_worker == 1


def test_main_delegates_direct_builder_without_intermediate_catalog(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "merged_v3.h5"
    captured = {}

    def fake_run(args, repo_root):
        captured["args"] = args
        captured["repo_root"] = repo_root
        return 17

    monkeypatch.setattr(direct, "run_v3_build", fake_run)
    result = backfill.main(
        [
            "--source-run", str(source),
            "--output", str(output),
            "--workers", "3",
            "--max-tasks-per-worker", "4",
            "--dry-run",
        ]
    )

    assert result == 17
    args = captured["args"]
    assert Path(args.source_run) == source.resolve()
    assert Path(args.output_catalog) == output.resolve()
    assert args.parallel == 3
    assert args.max_tasks_per_worker == 4
    assert args.dry_run is True
    assert args.provenance_entrypoint == "run_backfill_spectra_psf_photometry.py"
    assert not hasattr(args, "input_catalog")
