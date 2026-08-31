import subprocess
import sys
import re
from datetime import datetime as real_datetime
from pathlib import Path
from types import SimpleNamespace

import h5py
import pandas as pd
import pytest

import hpc_scripts.sfitlc as sfitlc
from qvc.provenance import decode_record
from hpc_scripts.sfitlc import (
    JobConfig,
    build_job_configs,
    build_merge_sbatch_script,
    build_sbatch_script,
    parse_args,
    validate_chunking,
)


def _args(**overrides):
    values = {
        "fit": "chisq",
        "N": 3,
        "skip": 0,
        "ncores": 1,
        "mem": "12G",
        "partition": "day",
        "time": "2:00:00",
        "env": "jaxcpu2",
        "svi_steps": 4000,
        "svi_lr": 1e-3,
        "nwarm": 500,
        "nsamp": 250,
        "max_tree_depth": 12,
        "resume": None,
        "extra_fit_flags": (),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_run_stamp_zero_pads_day(monkeypatch):
    class FixedDatetime:
        @staticmethod
        def now():
            return real_datetime(2026, 8, 6, 15, 20)

    monkeypatch.setattr(sfitlc, "datetime", FixedDatetime)

    assert sfitlc.make_run_stamp() == "aug06_0320pm"


def test_fit_job_name_places_run_stamp_first():
    run_stamp = "aug06_0320pm"
    prefix = sfitlc.build_run_prefix(
        "chisq",
        run_stamp,
        "abc1234",
        resume_prefix_base=None,
        run_description="deep_run",
    )

    script = build_sbatch_script(
        prefix,
        JobConfig(description="chisq", object_ids=["1"]),
        _args(),
        "data/input.csv",
        None,
        object_ids_path=Path("/tmp/probe_chisq_object_ids.txt"),
    )

    assert "#SBATCH --job-name=aug06_0320pm_lcfit_deep_run_abc1234_chisq\n" in script
    assert f'export PREFIX="{prefix}"' in script


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        (
            "aug06_0320pm_abc1234_stone",
            "aug06_0320pm_lcfit_abc1234_stone",
        ),
        ("existing_resume_stone", "lcfit_existing_resume_stone"),
    ],
)
def test_fit_job_name_handles_no_description_and_opaque_resume_prefix(prefix, expected):
    assert sfitlc.build_fit_job_name(prefix) == expected


def test_sbatch_runs_each_chunk_object_in_a_fresh_process():
    script = build_sbatch_script(
        "probe_chisq",
        JobConfig(
            description="chisq",
            object_ids=["1", "2", "3"],
            use_psf_constant_flux=True,
        ),
        _args(),
        "data/input.csv",
        "results/data/spectra.h5",
        object_ids_path=Path("/tmp/probe_chisq_object_ids.txt"),
        exclusion_paths=[Path("/tmp/previous.h5")],
        original_object_count=5,
        excluded_object_count=2,
    )

    assert 'export N="3"' in script
    assert 'START=$(( TASK_ID * N ))' in script
    assert 'read -r -a OBJECT_IDS <<< "$IDS"' in script
    assert 'for OBJECT_ID in "${OBJECT_IDS[@]}"; do' in script
    assert 'export SUFFIX="job${TASK_ID}_obj${OBJECT_INDEX}"' in script
    assert '--filter_object_id "$OBJECT_ID"' in script
    assert "--filter_object_id $IDS" not in script
    assert 'export OBJECT_ID_FILE="/tmp/probe_chisq_object_ids.txt"' in script
    assert 'pd.read_csv(os.environ["FILTER_CSV"])' not in script
    assert "--subtract_psf_constant_flux" in script
    assert "--spectra_fit_h5" in script
    assert "results/data/spectra.h5" in script
    assert script.count("python -m qvc.light_curve.fit_light_curves") == 1
    encoded = re.search(
        r'^export QVC_SUBMISSION_PROVENANCE_B64="([^"]+)"$',
        script,
        flags=re.MULTILINE,
    ).group(1)
    submission = decode_record(encoded)
    assert submission["resolved"]["job"]["object_count"] == 3
    assert submission["resolved"]["job"]["original_object_count"] == 5
    assert submission["resolved"]["job"]["excluded_object_count"] == 2
    assert submission["resolved"]["inputs"]["exclude_object_ids"] == [
        "/tmp/previous.h5"
    ]
    assert submission["resolved"]["resources"]["memory"] == "12G"
    assert "--spectra_fit_h5" in submission["resolved"]["fit_flags"]
    assert "--svi_lr" in submission["resolved"]["fit_flags"]
    assert "0.001" in submission["resolved"]["fit_flags"]
    assert "--svi_steps" in submission["resolved"]["fit_flags"]
    assert "4000" in submission["resolved"]["fit_flags"]


def test_generated_multi_object_sbatch_is_valid_bash():
    script = build_sbatch_script(
        "probe_stone",
        JobConfig(description="stone", object_ids=["1", "2"]),
        _args(fit="stone", N=2),
        None,
        "results/data/spectra.h5",
        object_ids_path=Path("/tmp/probe_object_ids.txt"),
    )

    result = subprocess.run(
        ["bash", "-n"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_chunk_count_still_uses_objects_per_slurm_task():
    assert validate_chunking(total_objects=7, n_per_job=3, skip=0, num_jobs=-1) == (3, 0, 2)


def test_chisq_jobs_require_an_explicit_spectra_fit_h5(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["sfitlc.py", "--fit", "chisq", "--chisq-csv", "results/data/chisq.csv"],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2
    assert "--spectra-fit-h5 is required when --fit chisq is used." in capsys.readouterr().err


def test_non_chisq_jobs_do_not_require_a_spectra_fit_h5(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sfitlc.py", "--fit", "stone"])

    args = parse_args()

    assert args.spectra_fit_h5 is None
    assert args.stone_linear_mode == "both"
    assert args.svi_steps == 4000
    assert args.svi_lr == pytest.approx(1e-3)
    assert args.exclude_object_ids == []


def test_parse_args_accepts_multiple_exclusion_files(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sfitlc.py",
            "--fit",
            "stone",
            "--exclude-object-ids",
            "previous.csv",
            "previous.h5",
            "--num-jobs",
            "1",
        ],
    )

    args = parse_args()

    assert args.exclude_object_ids == ["previous.csv", "previous.h5"]
    assert args.num_jobs == 1


def test_load_exclusion_object_ids_unions_csv_and_hdf5_layouts(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(sfitlc, "REPO_ROOT", tmp_path)
    pd.DataFrame(
        {"object_id": [" 101 ", "102.0", None, "", "101"]}
    ).to_csv(tmp_path / "exclude.csv", index=False)
    with h5py.File(tmp_path / "flat.h5", "w") as handle:
        handle.create_dataset("object_id", data=[b"103", b"104", b"103"])
    with h5py.File(tmp_path / "spectra.hdf5", "w") as handle:
        catalog = handle.create_group("catalog")
        catalog.create_dataset("object_id", data=[b"104", b"105"])

    object_ids, paths = sfitlc.load_exclusion_object_ids(
        ["exclude.csv", "flat.h5", "spectra.hdf5"]
    )

    assert object_ids == {"101", "102", "103", "104", "105"}
    assert paths == [
        (tmp_path / "exclude.csv").resolve(),
        (tmp_path / "flat.h5").resolve(),
        (tmp_path / "spectra.hdf5").resolve(),
    ]


@pytest.mark.parametrize("filename", ["missing.csv", "missing.h5"])
def test_load_exclusion_object_ids_rejects_missing_files(
    tmp_path,
    monkeypatch,
    filename,
):
    monkeypatch.setattr(sfitlc, "REPO_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="exclusion file not found"):
        sfitlc.load_exclusion_object_ids([filename])


def test_load_exclusion_object_ids_rejects_invalid_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(sfitlc, "REPO_ROOT", tmp_path)
    pd.DataFrame({"wrong": ["101"]}).to_csv(tmp_path / "wrong.csv", index=False)
    (tmp_path / "wrong.txt").write_text("101\n", encoding="utf-8")
    with h5py.File(tmp_path / "wrong.h5", "w") as handle:
        handle.create_dataset("wrong", data=[b"101"])

    with pytest.raises(ValueError, match="missing required column 'object_id'"):
        sfitlc.load_object_ids_from_file("wrong.csv")
    with pytest.raises(ValueError, match="expected .csv, .h5, or .hdf5"):
        sfitlc.load_object_ids_from_file("wrong.txt")
    with pytest.raises(ValueError, match="missing dataset '/object_id'"):
        sfitlc.load_object_ids_from_file("wrong.h5")


@pytest.mark.parametrize("fit", ["chisq", "stone", "macleod", "samelength"])
def test_exclusions_apply_to_every_fit_mode_in_source_order(monkeypatch, fit):
    monkeypatch.setattr(sfitlc, "load_chisq_ids", lambda _path: ["1", "2", "3", "2"])
    monkeypatch.setattr(sfitlc, "load_stone_ids", lambda: ["1", "2", "3", "2"])
    monkeypatch.setattr(sfitlc, "load_macleod_ids", lambda: ["1", "2", "3", "2"])

    jobs = build_job_configs(fit, "input.csv")
    filtered = [sfitlc.exclude_job_object_ids(job, {"2"}) for job in jobs]

    assert all(job.object_ids == ["1", "3"] for job, _count in filtered)
    assert all(count == 2 for _job, count in filtered)


def test_nonmatching_exclusions_leave_job_unchanged():
    original = JobConfig(description="stone", object_ids=["1", "2", "1"])

    filtered, excluded_count = sfitlc.exclude_job_object_ids(original, {"999"})

    assert filtered.object_ids == original.object_ids
    assert excluded_count == 0


def test_main_stops_before_writing_when_exclusions_remove_every_object(monkeypatch):
    args = _args(
        fit="stone",
        chisq_csv=None,
        spectra_fit_h5=None,
        stone_linear_mode="linear",
        description=None,
        exclude_object_ids=["previous.csv"],
        num_jobs=-1,
    )
    monkeypatch.setattr(sfitlc, "parse_args", lambda: args)
    monkeypatch.setattr(sfitlc, "get_git_short_hash", lambda: "abc1234")
    monkeypatch.setattr(sfitlc, "make_run_stamp", lambda: "aug31_1200pm")
    monkeypatch.setattr(
        sfitlc,
        "load_exclusion_object_ids",
        lambda _paths: ({"1"}, [Path("/tmp/previous.csv")]),
    )
    monkeypatch.setattr(sfitlc, "load_stone_ids", lambda: ["1"])

    def fail_if_called(*_args, **_kwargs):
        pytest.fail("job files must not be written for an empty filtered sample")

    monkeypatch.setattr(sfitlc, "write_job_script", fail_if_called)

    with pytest.raises(ValueError, match="No object_ids remain for stone"):
        sfitlc.main()


@pytest.mark.parametrize("profile", ("modified", "modified_narrow"))
def test_eta_prior_profile_is_forwarded_to_light_curve_fitter(monkeypatch, profile):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sfitlc.py",
            "--fit",
            "stone",
            "--eta_prior_profile",
            profile,
        ],
    )

    args = parse_args()

    assert args.extra_fit_flags == ("--eta_prior_profile", profile)


@pytest.mark.parametrize(
    ("mode", "expected_descriptions", "expected_flags"),
    [
        ("both", ["stone", "stone_nolinear"], [(), ("--disable_linear_trend",)]),
        ("linear", ["stone"], [()]),
        ("nolinear", ["stone_nolinear"], [("--disable_linear_trend",)]),
    ],
)
def test_stone_linear_mode_selects_requested_jobs(
    monkeypatch,
    mode,
    expected_descriptions,
    expected_flags,
):
    monkeypatch.setattr(sfitlc, "load_stone_ids", lambda: ["stone-1"])

    jobs = build_job_configs("stone", None, stone_linear_mode=mode)

    assert [job.description for job in jobs] == expected_descriptions
    assert [job.extra_flags for job in jobs] == expected_flags
    assert all(job.object_ids == ["stone-1"] for job in jobs)


def test_stone_linear_mode_rejects_unknown_choice(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["sfitlc.py", "--fit", "stone", "--stone-linear-mode", "quadratic"],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2
    assert "invalid choice: 'quadratic'" in capsys.readouterr().err


def test_stone_linear_mode_rejects_non_stone_fit(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["sfitlc.py", "--fit", "macleod", "--stone-linear-mode", "linear"],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2
    assert "requires --fit stone" in capsys.readouterr().err


@pytest.mark.parametrize(
    "plot_flag",
    ("stone", "macleod"),
)
def test_comparison_merge_reuses_saved_shard_fields(plot_flag):
    script = build_merge_sbatch_script(
        "comparison_prefix",
        plot_flag,
        _args(fit=plot_flag),
        enable_stone_identity_plot=plot_flag == "stone",
        enable_macleod_identity_plot=plot_flag == "macleod",
        enable_suberlak_identity_plot=plot_flag == "macleod",
    )

    assert "--skip-populate-sdss" in script
    assert "--compute-variability" not in script
    assert "#SBATCH --mem=20G" in script


def test_regular_merge_keeps_variability_recomputation():
    script = build_merge_sbatch_script(
        "chisq_prefix",
        "chisq",
        _args(fit="chisq"),
    )

    assert "--compute-variability" in script
    assert "--skip-populate-sdss" not in script
    assert "#SBATCH --mem=40G" in script
