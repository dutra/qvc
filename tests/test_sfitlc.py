import subprocess
import sys
from datetime import datetime as real_datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import hpc_scripts.sfitlc as sfitlc
from hpc_scripts.sfitlc import JobConfig, build_sbatch_script, parse_args, validate_chunking


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
        "svi_steps": 1000,
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
        "results/data/spectra.csv",
    )

    assert 'export N="3"' in script
    assert 'START=$(( TASK_ID * N ))' in script
    assert 'read -r -a OBJECT_IDS <<< "$IDS"' in script
    assert 'for OBJECT_ID in "${OBJECT_IDS[@]}"; do' in script
    assert 'export SUFFIX="job${TASK_ID}_obj${OBJECT_INDEX}"' in script
    assert '--filter_object_id "$OBJECT_ID"' in script
    assert "--filter_object_id $IDS" not in script
    assert "--subtract_psf_constant_flux" in script
    assert "--spectra_fit_csv" in script
    assert "results/data/spectra.csv" in script
    assert script.count("python -m qvc.light_curve.fit_light_curves") == 1


def test_generated_multi_object_sbatch_is_valid_bash():
    script = build_sbatch_script(
        "probe_stone",
        JobConfig(description="stone", object_ids=["1", "2"]),
        _args(fit="stone", N=2),
        None,
        "results/data/spectra.csv",
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


def test_chisq_jobs_require_an_explicit_spectra_fit_csv(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["sfitlc.py", "--fit", "chisq", "--chisq-csv", "results/data/chisq.csv"],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2
    assert "--spectra-fit-csv is required when --fit chisq is used." in capsys.readouterr().err


def test_non_chisq_jobs_do_not_require_a_spectra_fit_csv(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sfitlc.py", "--fit", "stone"])

    args = parse_args()

    assert args.spectra_fit_csv is None
