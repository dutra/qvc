import subprocess
import sys
import re
from datetime import datetime as real_datetime
from pathlib import Path
from types import SimpleNamespace

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
    )

    assert 'export N="3"' in script
    assert 'START=$(( TASK_ID * N ))' in script
    assert 'read -r -a OBJECT_IDS <<< "$IDS"' in script
    assert 'for OBJECT_ID in "${OBJECT_IDS[@]}"; do' in script
    assert 'export SUFFIX="job${TASK_ID}_obj${OBJECT_INDEX}"' in script
    assert '--filter_object_id "$OBJECT_ID"' in script
    assert "--filter_object_id $IDS" not in script
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
