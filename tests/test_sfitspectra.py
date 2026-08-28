import base64
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "hpc_scripts" / "sfitspectra.xsh"
RETRY_JOB_NAME = "aug03_1853_f3ea5a6_svi4000_N2000_PRS103to107"


def _write_executable(path, source):
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _copy_minimal_qvc_package(root):
    """Provide imports needed by a copied sfitspectra script."""
    qvc_dir = root / "src" / "qvc"
    qvc_dir.mkdir(parents=True)
    source_qvc_dir = SCRIPT.parents[1] / "src" / "qvc"
    shutil.copy2(source_qvc_dir / "__init__.py", qvc_dir / "__init__.py")
    shutil.copy2(source_qvc_dir / "provenance.py", qvc_dir / "provenance.py")


def _retry_workspace(
    tmp_path,
    accounting_rows,
    *,
    object_count=40,
    chunk_size=4,
    nproc=1,
):
    root = tmp_path / "repo"
    script_path = root / "hpc_scripts" / "sfitspectra.xsh"
    script_path.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT, script_path)
    (script_path.parent / "pandas.py").write_text("", encoding="utf-8")
    _copy_minimal_qvc_package(root)

    submit_dir = root / "hpc_scripts" / "submit" / "jaxqsofit"
    submit_dir.mkdir(parents=True)
    object_ids_path = submit_dir / f"{RETRY_JOB_NAME}_object_ids.txt"
    object_ids_path.write_text(
        "".join(f"{index}\n" for index in range(object_count)),
        encoding="utf-8",
    )
    saved_script = submit_dir / f"submit_{RETRY_JOB_NAME}.sbatch"
    relative_object_ids = object_ids_path.relative_to(root)
    saved_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f"#SBATCH --job-name={RETRY_JOB_NAME}",
                f'export PREFIX="{RETRY_JOB_NAME}"',
                f'export OBJECT_IDS_FILE="{relative_object_ids}"',
                f"export CHUNK_SIZE={chunk_size}",
                f"export NPROC={nproc}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    fake_bin = root / "fake-bin"
    fake_bin.mkdir()
    accounting_path = root / "sacct.txt"
    accounting_path.write_text("\n".join(accounting_rows) + "\n", encoding="utf-8")
    calls_path = root / "sbatch-calls.json"
    _write_executable(
        fake_bin / "sacct",
        """#!/usr/bin/env python3
import os
from pathlib import Path

print(Path(os.environ["FAKE_SACCT_OUTPUT"]).read_text(encoding="utf-8"), end="")
""",
    )
    _write_executable(
        fake_bin / "sbatch",
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

path = Path(os.environ["FAKE_SBATCH_CALLS"])
calls = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
calls.append(sys.argv[1:])
path.write_text(json.dumps(calls), encoding="utf-8")
""",
    )
    env = os.environ.copy()
    env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
    env["FAKE_SACCT_OUTPUT"] = str(accounting_path)
    env["FAKE_SBATCH_CALLS"] = str(calls_path)
    return root, script_path, saved_script, object_ids_path, calls_path, env


def _run_retry(root, script_path, env, *extra_args):
    return subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--retry",
            RETRY_JOB_NAME,
            *extra_args,
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _sbatch_calls(calls_path):
    if not calls_path.exists():
        return []
    return json.loads(calls_path.read_text(encoding="utf-8"))


def _fresh_workspace(tmp_path, csv_text="object_id\n1\n2\n"):
    root = tmp_path / "repo"
    script_path = root / "hpc_scripts" / "sfitspectra.xsh"
    script_path.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT, script_path)

    _copy_minimal_qvc_package(root)

    data_dir = root / "data"
    data_dir.mkdir()
    csv_path = data_dir / "objects.csv"
    csv_path.write_text(csv_text, encoding="utf-8")
    (data_dir / "jul14_master_input_file_chisqgt20_bandwagon_photometry.csv").write_text(
        "object_id\n", encoding="utf-8"
    )

    fake_bin = root / "fake-bin"
    fake_bin.mkdir()
    calls_path = root / "sbatch-calls.json"
    _write_executable(
        fake_bin / "git",
        """#!/usr/bin/env python3
print("abc1234")
""",
    )
    _write_executable(
        fake_bin / "sbatch",
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

path = Path(os.environ["FAKE_SBATCH_CALLS"])
calls = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
calls.append(sys.argv[1:])
path.write_text(json.dumps(calls), encoding="utf-8")
""",
    )
    env = os.environ.copy()
    env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
    env["FAKE_SBATCH_CALLS"] = str(calls_path)
    return root, script_path, csv_path, calls_path, env


def _run_fresh(root, script_path, env, *args):
    return subprocess.run(
        [sys.executable, str(script_path), *args],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_sfitspectra_uses_csv_object_ids_without_h5_membership_filtering():
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"--chisq-csv"' in source
    assert "chisq_path = Path(cli_args.chisq_csv).expanduser()" in source
    assert "submit_object_ids = requested_object_ids" in source
    assert "--filter_object_id" in source
    for legacy_text in (
        "read_quasars_from_hdf5_flat",
        "h5_file",
        "load_h5_object_ids",
        "missing_from_h5",
        "H5_FILE",
        "USE_H5",
    ):
        assert legacy_text not in source


def test_sfitspectra_supports_both_fit_backends():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'fit_script = "fit_spectra.py"' in source
    assert '"fit_spectra.py": "qvc.spectra.fit_spectra"' in source
    assert (
        '"fit_spectra_jaxsedfit_joint.py": '
        '"qvc.spectra.fit_spectra_jaxsedfit_joint"'
    ) in source
    assert '"-m", fit_module' in source
    assert "Unsupported fit_script" in source


def test_sfitspectra_uses_backend_specific_arguments():
    source = SCRIPT.read_text(encoding="utf-8")

    assert (
        'sed_photometry_path = '
        '"data/jul14_master_input_file_chisqgt20_bandwagon_photometry.csv"'
    ) in source
    assert 'if fit_script == "fit_spectra.py":' in source
    assert '"--plot_mcmc_diagnostics"' in source
    assert '"--save-fig"' in source
    assert 'elif fit_script == "fit_spectra_jaxsedfit_joint.py":' in source
    assert '"--sed-photometry-path", sed_photometry_path' in source
    assert '"--progress"' in source
    assert "SED photometry input not found" in source


def test_sfitspectra_accepts_cli_overrides_and_builds_timestamped_spectrafit_job_name():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'parser.add_argument(\n    "--description",' in source
    assert 'parser.add_argument(\n    "--chisq-csv",' in source
    assert 'parser.add_argument(\n    "--fit-script",' in source
    assert "fit_script = cli_args.fit_script" in source
    assert 'parser.add_argument(\n    "--sed-photometry-path",' not in source
    assert 'datetime.now().strftime("%b%d_%I%M%p").lower()' in source
    assert '["git", "rev-parse", "--short", "HEAD"]' in source
    assert 'r"[^A-Za-z0-9.-]+", "_", cli_args.description' in source
    assert 'job_name_parts = [date_hour, "spectrafit", git_commit]' in source
    assert 'job_name = "_".join(job_name_parts)' in source
    assert "prefix = job_name" in source
    assert 'output_dir = f"results/data/jaxqsofit/{prefix}"' in source
    assert 'fig_dir = f"plots/jaxqsofit/{prefix}"' in source
    assert "#SBATCH --job-name={job_name}" in source
    assert 'export {PROVENANCE_ENV}="{encoded_submission}"' in source


@pytest.mark.parametrize("path_mode", ["relative", "absolute"])
def test_sfitspectra_fresh_run_uses_explicit_csv_and_description(tmp_path, path_mode):
    root, script_path, csv_path, calls_path, env = _fresh_workspace(
        tmp_path,
        "object_id,unused\n1,a\n2.0,b\n1,duplicate\n 3 ,c\n4,d\n5,e\n",
    )
    csv_argument = str(csv_path if path_mode == "absolute" else csv_path.relative_to(root))

    result = _run_fresh(
        root,
        script_path,
        env,
        "--chisq-csv",
        csv_argument,
        "--description",
        "Nested N8000",
    )

    assert result.returncode == 0, result.stderr
    submit_dir = root / "hpc_scripts" / "submit" / "jaxqsofit"
    manifests = list(submit_dir.glob("*_object_ids.txt"))
    scripts = list(submit_dir.glob("submit_*.sbatch"))
    assert len(manifests) == len(scripts) == 1
    assert manifests[0].read_text(encoding="utf-8").splitlines() == ["1", "2", "3", "4", "5"]
    assert re.fullmatch(
        r"[a-z]{3}\d{2}_\d{4}(?:am|pm)_spectrafit_abc1234_Nested_N8000_object_ids\.txt",
        manifests[0].name,
    )
    generated = scripts[0].read_text(encoding="utf-8")
    assert "#SBATCH --array=0-1" in generated
    encoded = re.search(
        r'export QVC_SUBMISSION_PROVENANCE_B64="([^"]+)"', generated
    ).group(1)
    provenance = json.loads(base64.b64decode(encoded).decode("utf-8"))
    assert provenance["resolved"]["inputs"]["chisq_csv"] == str(csv_path.resolve())
    assert provenance["resolved"]["wrapper_args"]["description"] == "Nested N8000"
    assert provenance["resolved"]["description"] == "Nested_N8000"
    assert len(_sbatch_calls(calls_path)) == 1


@pytest.mark.parametrize(
    ("extra_args", "error_text"),
    [
        (("--description", "sample"), "--chisq-csv is required for fresh runs"),
        (("--chisq-csv", "data/objects.csv"), "--description is required for fresh runs"),
        (
            ("--chisq-csv", "data/objects.csv", "--description", "   "),
            "--description cannot be blank",
        ),
    ],
)
def test_sfitspectra_fresh_run_requires_csv_and_nonblank_description(
    tmp_path, extra_args, error_text
):
    root, script_path, _csv_path, calls_path, env = _fresh_workspace(tmp_path)

    result = _run_fresh(root, script_path, env, *extra_args)

    assert result.returncode == 2
    assert error_text in result.stderr
    assert not (root / "results").exists()
    assert _sbatch_calls(calls_path) == []


@pytest.mark.parametrize(
    ("csv_argument", "csv_text", "error_text"),
    [
        ("data/missing.csv", None, "Object-list CSV not found"),
        ("data/objects.csv", "wrong_column\n1\n", "missing required column 'object_id'"),
        (
            "data/objects.csv",
            "object_id\n\n",
            "No valid object_id values remain after normalization and exclusions",
        ),
    ],
)
def test_sfitspectra_rejects_invalid_object_csv_before_creating_outputs(
    tmp_path, csv_argument, csv_text, error_text
):
    root, script_path, csv_path, calls_path, env = _fresh_workspace(tmp_path)
    if csv_text is not None:
        csv_path.write_text(csv_text, encoding="utf-8")

    result = _run_fresh(
        root,
        script_path,
        env,
        "--chisq-csv",
        csv_argument,
        "--description",
        "sample",
    )

    assert result.returncode != 0
    assert error_text in result.stderr
    assert not (root / "results").exists()
    assert _sbatch_calls(calls_path) == []


def test_sfitspectra_named_resume_keeps_current_csv_selection_and_separate_outputs():
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"--resume",\n    metavar="OLD_RUN_NAME"' in source
    assert 'submit_object_ids = requested_object_ids' in source
    assert 'f"results/data/jaxqsofit/{resume_run_name}/all"' in source
    assert 'f"results/data/jaxqsofit/{prefix}"' in source
    assert 'resume_path.resolve() == (output_path / "all").resolve()' in source
    assert '"--resume is supported only with fit_spectra_jaxsedfit_joint.py"' in source
    assert 'export RESUME_DIR="{resume_dir}"' in source
    assert '"--resume", resume_dir' in source
    assert '"--resume-run-name", resume_run_name' in source


def test_sfitspectra_retry_resubmits_latest_unsuccessful_tasks_with_current_resources(
    tmp_path,
):
    rows = [
        f"100_0|{RETRY_JOB_NAME}|COMPLETED",
        f"100_1|{RETRY_JOB_NAME}|FAILED",
        f"101_1|{RETRY_JOB_NAME}|COMPLETED",
        f"100_2|{RETRY_JOB_NAME}|COMPLETED",
        f"101_2|{RETRY_JOB_NAME}|TIMEOUT",
        f"100_3|{RETRY_JOB_NAME}|RUNNING",
        f"100_4|{RETRY_JOB_NAME}|OUT_OF_MEMORY",
        f"100_5|{RETRY_JOB_NAME}|CANCELLED by 1234",
        f"100_6|{RETRY_JOB_NAME}|NODE_FAIL",
        f"100_7|{RETRY_JOB_NAME}|PREEMPTED",
        f"100_8|{RETRY_JOB_NAME}|FAILED",
        f"100_2.batch|{RETRY_JOB_NAME}|FAILED",
        "100_9|another_job|FAILED",
    ]
    root, script_path, saved_script, _object_ids, calls_path, env = (
        _retry_workspace(tmp_path, rows)
    )

    result = _run_retry(root, script_path, env)

    assert result.returncode == 0, result.stderr
    calls = _sbatch_calls(calls_path)
    assert len(calls) == 1
    assert calls[0][:5] == [
        "--array=2,4-8",
        "--partition=day",
        "--time=4:00:00",
        "--mem=40G",
        "--cpus-per-task=1",
    ]
    assert calls[0][5].startswith("--export=ALL,QVC_RETRY_PROVENANCE_B64=")
    assert Path(calls[0][-1]) == saved_script
    assert "task 2: TIMEOUT" in result.stdout
    assert "task 5: CANCELLED" in result.stdout


def test_sfitspectra_retry_preserves_scalar_task_zero(tmp_path):
    rows = [f"100|{RETRY_JOB_NAME}|FAILED"]
    root, script_path, saved_script, _object_ids, calls_path, env = (
        _retry_workspace(tmp_path, rows, object_count=1)
    )

    result = _run_retry(root, script_path, env)

    assert result.returncode == 0, result.stderr
    calls = _sbatch_calls(calls_path)
    assert calls[0][0] == "--array=0"
    assert Path(calls[0][-1]) == saved_script


def test_sfitspectra_retry_does_nothing_when_latest_tasks_are_not_failed(tmp_path):
    rows = [
        f"100_0|{RETRY_JOB_NAME}|FAILED",
        f"101_0|{RETRY_JOB_NAME}|COMPLETED",
        f"100_1|{RETRY_JOB_NAME}|PENDING",
    ]
    root, script_path, _saved_script, _object_ids, calls_path, env = (
        _retry_workspace(tmp_path, rows)
    )

    result = _run_retry(root, script_path, env)

    assert result.returncode == 0, result.stderr
    assert _sbatch_calls(calls_path) == []
    assert "No unsuccessful terminal tasks to retry" in result.stdout


@pytest.mark.parametrize(
    ("rows", "error_text"),
    [
        ([f"100_0|{RETRY_JOB_NAME}|MYSTERY"], "Unrecognized Slurm state"),
        ([f"100_10|{RETRY_JOB_NAME}|FAILED"], "outside the original task range"),
    ],
)
def test_sfitspectra_retry_rejects_unsafe_accounting_rows(
    tmp_path,
    rows,
    error_text,
):
    root, script_path, _saved_script, _object_ids, calls_path, env = (
        _retry_workspace(tmp_path, rows, object_count=8)
    )

    result = _run_retry(root, script_path, env)

    assert result.returncode != 0
    assert error_text in result.stderr
    assert _sbatch_calls(calls_path) == []


def test_sfitspectra_retry_requires_original_artifacts_and_sufficient_cpus(tmp_path):
    rows = [f"100_0|{RETRY_JOB_NAME}|FAILED"]
    root, script_path, _saved_script, object_ids, calls_path, env = (
        _retry_workspace(tmp_path, rows)
    )
    object_ids.unlink()

    missing_result = _run_retry(root, script_path, env)

    assert missing_result.returncode != 0
    assert "Original object-ID manifest not found" in missing_result.stderr
    assert _sbatch_calls(calls_path) == []

    root, script_path, _saved_script, _object_ids, calls_path, env = (
        _retry_workspace(tmp_path / "cpus", rows, nproc=2)
    )

    cpu_result = _run_retry(root, script_path, env)

    assert cpu_result.returncode != 0
    assert "fewer than the saved NPROC=2" in cpu_result.stderr
    assert _sbatch_calls(calls_path) == []


@pytest.mark.parametrize(
    "fresh_args",
    [
        ("--description", "new-name"),
        ("--description", ""),
        ("--chisq-csv", "data/new.csv"),
        ("--resume", ""),
        ("--fit-script", "fit_spectra.py"),
    ],
)
def test_sfitspectra_retry_rejects_fresh_run_options(tmp_path, fresh_args):
    rows = [f"100_0|{RETRY_JOB_NAME}|FAILED"]
    root, script_path, _saved_script, _object_ids, calls_path, env = (
        _retry_workspace(tmp_path, rows)
    )

    result = _run_retry(root, script_path, env, *fresh_args)

    assert result.returncode == 2
    assert "--retry cannot be combined with fresh-run options" in result.stderr
    assert _sbatch_calls(calls_path) == []
