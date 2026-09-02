import json
import os
from datetime import datetime
from pathlib import Path
import re
import runpy
import shutil
import stat
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "hpc_scripts/shubble_validation.xsh"
PREFIX_PATTERN = re.compile(r"^[a-z]{3}\d{2}_\d{4}(?:am|pm)_$")


def test_bouchet_resource_and_campaign_defaults():
    values = runpy.run_path(str(SCRIPT))
    assert values["cpus_per_task"] == 8
    assert values["memory"] == "32G"
    assert values["time_limit"] == "02:00:00"
    assert values["num_agns"] == 2000
    assert values["num_runs"] == 64
    assert values["prior_profile"] == "centered_lcdm"
    assert values["campaign"].endswith("_prior-centered_lcdm")


def test_submission_prefix_uses_requested_local_timestamp_format():
    values = runpy.run_path(str(SCRIPT))
    prefix = values["submission_prefix"]()
    assert PREFIX_PATTERN.fullmatch(prefix)
    assert values["submission_prefix"](datetime(2026, 9, 1, 17, 52)) == "sep01_0552pm_"


def test_configured_centered_prior_profile_is_forwarded():
    values = runpy.run_path(str(SCRIPT))
    centered_args = values["parse_args"]([])
    default_args = values["parse_args"](["--prior-profile", "default"])

    assert "--prior-profile" not in values["runner_arguments"](default_args)
    centered_arguments = values["runner_arguments"](centered_args)
    index = centered_arguments.index("--prior-profile")
    assert centered_arguments[index + 1] == "centered_lcdm"


def _write_executable(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _workspace(tmp_path):
    repo = tmp_path / "qvc"
    (repo / "hpc_scripts").mkdir(parents=True)
    (repo / "scripts").mkdir()
    driver = repo / "hpc_scripts/shubble_validation.xsh"
    shutil.copy2(SCRIPT, driver)
    init_calls = repo / "init_calls.json"
    _write_executable(
        repo / "scripts/run_hubble_validation.py",
        """#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--campaign")
parser.add_argument("--output-root")
args, unknown = parser.parse_known_args()
calls_path = Path(os.environ["INIT_CALLS"])
calls = json.loads(calls_path.read_text()) if calls_path.exists() else []
calls.append([*os.sys.argv[1:]])
calls_path.write_text(json.dumps(calls))
campaign = Path(args.output_root) / args.campaign
campaign.mkdir(parents=True, exist_ok=True)
""",
    )
    _write_executable(
        repo / "scripts/plot_hubble_validation.py",
        "#!/usr/bin/env python3\nraise SystemExit(0)\n",
    )
    return repo, driver, init_calls


def _base_command(repo, driver, output_root):
    return [
        sys.executable,
        str(driver),
        "--repo-dir", str(repo),
        "--python-bin", sys.executable,
        "--output-root", str(output_root),
        "--campaign", "test_campaign",
        "--num-runs", "3",
        "--num-agns", "123",
        "--seed-start", "5",
        "--speed", "quick",
        "--calibration-size", "456",
        "--arms", "all", "selected_oracle",
        "--cpus-per-task", "4",
        "--max-concurrent", "2",
    ]


def test_hubble_validation_submission_dry_run_writes_valid_scripts_without_submission(tmp_path):
    repo, driver, init_calls = _workspace(tmp_path)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    sbatch_marker = tmp_path / "sbatch-called"
    _write_executable(
        fake_bin / "sbatch",
        f"#!/usr/bin/env bash\ntouch {sbatch_marker}\nexit 99\n",
    )
    env = os.environ.copy()
    env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
    env["INIT_CALLS"] = str(init_calls)
    completed = subprocess.run(
        [*_base_command(repo, driver, tmp_path / "results"), "--dry-run"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert not sbatch_marker.exists()
    assert not init_calls.exists()
    assert not (tmp_path / "results/test_campaign").exists()

    submit_dir = repo / "hpc_scripts/submit/hubble_validation"
    fit_scripts = list(submit_dir.glob("*_test_campaign_fits.sbatch"))
    plot_scripts = list(submit_dir.glob("*_test_campaign_plot.sbatch"))
    assert len(fit_scripts) == len(plot_scripts) == 1
    fit_script = fit_scripts[0]
    plot_script = plot_scripts[0]
    prefix = fit_script.name.removesuffix("test_campaign_fits.sbatch")
    assert PREFIX_PATTERN.fullmatch(prefix)
    assert plot_script.name == f"{prefix}test_campaign_plot.sbatch"
    assert fit_script.is_file() and plot_script.is_file()
    assert subprocess.run(["bash", "-n", str(fit_script)]).returncode == 0
    assert subprocess.run(["bash", "-n", str(plot_script)]).returncode == 0
    fit_text = fit_script.read_text()
    plot_text = plot_script.read_text()
    assert f"#SBATCH --job-name={prefix}hval_test_campaign" in fit_text
    assert f"#SBATCH --job-name={prefix}hval_plot_test_campaign" in plot_text
    assert f"logs/hubble_validation/{prefix}test_campaign" in fit_text
    assert f"logs/hubble_validation/{prefix}test_campaign" in plot_text
    assert "#SBATCH --array=0-2%2" in fit_text
    assert "REALIZATION=$((5 + TASK_ID))" in fit_text
    assert "--num-agns 123" in fit_text
    assert "--n-runs 3" in fit_text
    assert "--arms all selected_oracle" in fit_text
    assert '--realization "$REALIZATION"' in fit_text
    assert "export NUM_CORES=\"${SLURM_CPUS_PER_TASK}\"" in fit_text


def test_hubble_validation_submission_uses_afterany_dependency_and_records_job_ids(tmp_path):
    repo, driver, init_calls = _workspace(tmp_path)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    sbatch_calls = tmp_path / "sbatch_calls.json"
    counter = tmp_path / "sbatch_counter.txt"
    _write_executable(
        fake_bin / "sbatch",
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

calls_path = Path(os.environ["SBATCH_CALLS"])
calls = json.loads(calls_path.read_text()) if calls_path.exists() else []
calls.append(sys.argv[1:])
calls_path.write_text(json.dumps(calls))
counter_path = Path(os.environ["SBATCH_COUNTER"])
value = int(counter_path.read_text()) + 1 if counter_path.exists() else 1001
counter_path.write_text(str(value))
print(f"{value};bouchet")
""",
    )
    env = os.environ.copy()
    env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
    env["INIT_CALLS"] = str(init_calls)
    env["SBATCH_CALLS"] = str(sbatch_calls)
    env["SBATCH_COUNTER"] = str(counter)
    output_root = tmp_path / "results"
    completed = subprocess.run(
        [*_base_command(repo, driver, output_root), "--resume", "--retry-failed"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    initialization = json.loads(init_calls.read_text())
    assert len(initialization) == 1
    assert "--initialize-only" in initialization[0]
    assert "--resume" in initialization[0]
    submit_dir = repo / "hpc_scripts/submit/hubble_validation"
    fit_scripts = list(submit_dir.glob("*_test_campaign_fits.sbatch"))
    assert len(fit_scripts) == 1
    fit_script = fit_scripts[0]
    prefix = fit_script.name.removesuffix("test_campaign_fits.sbatch")
    assert PREFIX_PATTERN.fullmatch(prefix)
    fit_text = fit_script.read_text()
    assert "--resume --retry-failed" in fit_text

    calls = json.loads(sbatch_calls.read_text())
    assert len(calls) == 2
    assert calls[0][0] == "--parsable"
    assert calls[1][0:2] == ["--parsable", "--dependency=afterany:1001"]
    assert calls[0][-1].endswith(f"{prefix}test_campaign_fits.sbatch")
    assert calls[1][-1].endswith(f"{prefix}test_campaign_plot.sbatch")

    metadata_files = list((output_root / "test_campaign").glob("*_hpc_submission.json"))
    assert len(metadata_files) == 1
    assert metadata_files[0].name == f"{prefix}hpc_submission.json"
    metadata = json.loads(metadata_files[0].read_text())
    assert metadata["job_ids"] == {"fit_array": "1001", "plot": "1002"}
    assert metadata["plot_dependency"] == "afterany:1001"
    assert metadata["settings"]["num_agns"] == 123
    assert metadata["settings"]["arms"] == ["all", "selected_oracle"]
    assert metadata["settings"]["retry_failed"] is True
    assert metadata["settings"]["submission_prefix"] == prefix
    assert metadata["settings"]["artifact_name"] == f"{prefix}test_campaign"
    assert metadata["settings"]["job_names"] == {
        "fit": f"{prefix}hval_test_campaign",
        "plot": f"{prefix}hval_plot_test_campaign",
    }
