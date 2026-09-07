import sys
from types import SimpleNamespace

import pytest

from hpc_scripts import smanage


JOB_NAME = "sep01_0928pm_hval_fixed_truth_nagns2000_nruns64_quick_prior-centered_lcdm"


def _run_cli(monkeypatch, argv, squeue_output):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[0] == "squeue":
            return SimpleNamespace(stdout=squeue_output, stderr="", returncode=0)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(smanage.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["smanage.py", *argv])
    assert smanage.main() == 0
    return commands


def test_cancel_uses_array_parent_once_for_compact_and_running_rows(monkeypatch, capsys):
    squeue_output = "\n".join(
        (
            f"24489297|24489297_[56-63%64]|{JOB_NAME}|PENDING|0:00",
            f"24489297|24489297_54|{JOB_NAME}|RUNNING|1:18",
            f"24489297|24489297_55|{JOB_NAME}|RUNNING|1:18",
        )
    )

    commands = _run_cli(monkeypatch, ["cancel", JOB_NAME], squeue_output)

    assert commands[0] == [
        "squeue",
        "--noheader",
        "--format=%F|%i|%j|%T|%M",
        "--me",
    ]
    assert commands[1:] == [["scancel", "24489297"]]
    output = capsys.readouterr().out
    assert "24489297_[56-63%64]" in output
    assert "Cancelling job 24489297\n" in output


def test_cancel_deduplicates_each_array_and_preserves_scalar_jobs(monkeypatch):
    squeue_output = "\n".join(
        (
            "300|300_[0-3%2]|target_job|PENDING|0:00",
            "300|300_4|target_job|RUNNING|0:12",
            "301|301_0|target_job|RUNNING|0:10",
            "302|302|target_job|RUNNING|0:08",
        )
    )

    commands = _run_cli(monkeypatch, ["cancel", "target_job"], squeue_output)

    assert commands[1:] == [
        ["scancel", "300"],
        ["scancel", "301"],
        ["scancel", "302"],
    ]


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        ("hold", ["scontrol", "hold", "400"]),
        ("resume", ["scontrol", "release", "400"]),
    ),
)
def test_hold_and_resume_use_deduplicated_array_parent(monkeypatch, mode, expected):
    squeue_output = "\n".join(
        (
            "400|400_[1-2]|target_job|PENDING|0:00",
            "400|400_3|target_job|PENDING|0:00",
        )
    )

    commands = _run_cli(monkeypatch, [mode, "target_job"], squeue_output)

    assert commands[1:] == [expected]


def test_dry_run_displays_matches_without_scheduler_action(monkeypatch, capsys):
    commands = _run_cli(
        monkeypatch,
        ["cancel", "target_job", "--dry-run"],
        "500|500_[0-9%2]|target_job|PENDING|0:00",
    )

    assert len(commands) == 1
    assert "500_[0-9%2]" in capsys.readouterr().out
