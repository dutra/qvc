import base64
import hashlib
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
    assert '"--no-catalog-progress"' in source
    assert "SED photometry input not found" in source
    assert 'out_suffix = ".h5" if fit_script == "fit_spectra_jaxsedfit_joint.py" else ".csv"' in source


def test_sfitspectra_accepts_cli_overrides_and_builds_timestamped_spectrafit_job_name():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'parser.add_argument(\n    "--description",' in source
    assert 'parser.add_argument(\n    "--chisq-csv",' in source
    assert 'parser.add_argument(\n    "--fit-script",' in source
    assert 'parser.add_argument(\n    "--fit-bal",' in source
    assert "fit_script = cli_args.fit_script" in source
    assert 'parser.add_argument(\n    "--sed-photometry-path",' in source
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
    assert "#SBATCH --array=0-4" in generated
    encoded = re.search(
        r'export QVC_SUBMISSION_PROVENANCE_B64="([^"]+)"', generated
    ).group(1)
    provenance = json.loads(base64.b64decode(encoded).decode("utf-8"))
    assert provenance["resolved"]["inputs"]["chisq_csv"] == str(csv_path.resolve())
    assert provenance["resolved"]["wrapper_args"]["description"] == "Nested N8000"
    assert provenance["resolved"]["description"] == "Nested_N8000"
    assert len(_sbatch_calls(calls_path)) == 1


def test_sfitspectra_forwards_fit_bal_to_joint_backend(tmp_path):
    root, script_path, csv_path, _, env = _fresh_workspace(tmp_path)
    (root / "results/data/jaxqsofit/old_BAL/all").mkdir(parents=True)

    result = _run_fresh(
        root,
        script_path,
        env,
        "--chisq-csv", str(csv_path),
        "--description", "BAL",
        "--fit-bal",
        "--resume", "old_BAL",
        "--allow-resume-without-submission-record",
    )

    assert result.returncode == 0, result.stderr
    generated = next((root / "hpc_scripts/submit/jaxqsofit").glob("submit_*.sbatch"))
    source = generated.read_text(encoding="utf-8")
    assert 'export FIT_BAL="1"' in source
    assert 'cmd.append("--fit-bal")' in source
    assert 'export RESUME_RUN_NAME="old_BAL"' in source


def test_sfitspectra_rejects_fit_bal_with_standalone_backend(tmp_path):
    root, script_path, csv_path, _, env = _fresh_workspace(tmp_path)

    result = _run_fresh(
        root,
        script_path,
        env,
        "--chisq-csv", str(csv_path),
        "--description", "BAL",
        "--fit-script", "fit_spectra.py",
        "--fit-bal",
    )

    assert result.returncode != 0
    assert "--fit-bal is supported only" in result.stderr


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
        "--time=3:00:00",
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
        ("--fit-bal",),
        ("--line-extrapolation-anchor", "continuum_ew"),
        ("--line-extrapolation-ew-scope", "all"),
        ("--narrow-line-extrapolation", "optical_only"),
        ("--nuts-chains", "4"),
        ("--nuts-warmup=800",),
        ("--sdss-spectrum-selection-path", "data/selection.csv"),
        ("--cache-dir=data",),
        ("--allow-resume-without-submission-record",),
        ("--sed-photometry-path", "data/phot.csv"),
        ("--require-sed-photometry",),
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


def test_sfitspectra_forwards_resume_only(tmp_path, monkeypatch):
    root, script_path, csv_path, _, env = _fresh_workspace(tmp_path)
    (root / 'results/data/jaxqsofit/old/all').mkdir(parents=True)
    result = _run_fresh(root, script_path, env, '--chisq-csv', str(csv_path),
                        '--description', 'reconstruct', '--resume', 'old', '--resume-only',
                        '--allow-resume-without-submission-record')
    assert result.returncode == 0, result.stderr
    generated = next((root / 'hpc_scripts/submit/jaxqsofit').glob('submit_*.sbatch')).read_text()
    assert 'export RESUME_ONLY="1"' in generated
    # Execute the generated Python block with a stub subprocess to inspect
    # the actual command reaching the fitter, without submitting HPC work.
    block = generated.split('"$PYTHON_BIN" - <<\'PY\'\n', 1)[1].split('\nPY\n', 1)[0]
    for key, value in re.findall(r'^export (\w+)="([^"\n]*)"$', generated, re.M):
        monkeypatch.setenv(key, value)
    for key in ('CHUNK_SIZE', 'NPROC', 'CPUS_PER_TASK'):
        match = re.search(rf'^export {key}="?(\d+)"?$', generated, re.M)
        monkeypatch.setenv(key, match.group(1))
    monkeypatch.setenv('TASK_ID', '0')
    monkeypatch.chdir(root)
    commands = []
    monkeypatch.setattr(subprocess, 'run', lambda cmd, **kwargs: commands.append(cmd))
    exec(compile(block, str(script_path), 'exec'), {})
    assert len(commands) == 1
    assert '--resume-only' in commands[0]
    assert commands[0][commands[0].index('--resume') + 1].endswith('/old/all')


def test_sfitspectra_resume_only_requires_resume(tmp_path):
    root, script_path, csv_path, calls_path, env = _fresh_workspace(tmp_path)
    result = _run_fresh(root, script_path, env, '--chisq-csv', str(csv_path),
                        '--description', 'reconstruct', '--resume-only')
    assert result.returncode == 2
    assert '--resume-only requires --resume' in result.stderr
    assert _sbatch_calls(calls_path) == []


def _generated_fit_command(root, script_path, generated, monkeypatch):
    block = generated.split('"$PYTHON_BIN" - <<\'PY\'\n', 1)[1].split('\nPY\n', 1)[0]
    for key, value in re.findall(r'^export (\w+)="([^"\n]*)"$', generated, re.M):
        monkeypatch.setenv(key, value)
    for key in ('CHUNK_SIZE', 'NPROC', 'CPUS_PER_TASK'):
        match = re.search(rf'^export {key}="?(\d+)"?$', generated, re.M)
        monkeypatch.setenv(key, match.group(1))
    monkeypatch.setenv('TASK_ID', '0')
    monkeypatch.chdir(root)
    commands = []
    monkeypatch.setattr(subprocess, 'run', lambda cmd, **kwargs: commands.append(cmd))
    exec(compile(block, str(script_path), 'exec'), {})
    assert len(commands) == 1
    return commands[0]


@pytest.mark.parametrize(
    "extra",
    [
        (),
        ("--line-extrapolation-anchor", "fitted_lines"),
        ("--line-extrapolation-anchor", "continuum_ew", "--line-extrapolation-ew-scope", "all"),
        ("--narrow-line-extrapolation", "optical_only"),
    ],
)
def test_sfitspectra_forwards_prediction_options_only_when_given(tmp_path, monkeypatch, extra):
    root, script_path, csv_path, _, env = _fresh_workspace(tmp_path)
    (root / 'results/data/jaxqsofit/old/all').mkdir(parents=True)
    result = _run_fresh(root, script_path, env, '--chisq-csv', str(csv_path),
                        '--description', 'anchor', '--resume', 'old', '--resume-only',
                        '--allow-resume-without-submission-record', *extra)
    assert result.returncode == 0, result.stderr
    generated = next((root / 'hpc_scripts/submit/jaxqsofit').glob('submit_*.sbatch')).read_text()
    command = _generated_fit_command(root, script_path, generated, monkeypatch)
    expected = dict(zip(extra[0::2], extra[1::2]))
    for option in ('--line-extrapolation-anchor', '--line-extrapolation-ew-scope', '--narrow-line-extrapolation'):
        if option in expected:
            assert command[command.index(option) + 1] == expected[option]
        else:
            assert option not in command
    assert '--resume-only' in command


def test_sfitspectra_rejects_line_extrapolation_anchor_with_standalone_backend(tmp_path):
    root, script_path, csv_path, _, env = _fresh_workspace(tmp_path)

    result = _run_fresh(
        root,
        script_path,
        env,
        "--chisq-csv", str(csv_path),
        "--description", "anchor",
        "--fit-script", "fit_spectra.py",
        "--line-extrapolation-anchor", "continuum_ew",
    )

    assert result.returncode != 0
    assert "--line-extrapolation-anchor is supported only" in result.stderr


@pytest.mark.parametrize(
    "extra, expected",
    [
        ((), ("500", "500", "1")),
        (("--nuts-warmup", "800", "--nuts-samples", "300", "--nuts-chains", "4"), ("800", "300", "4")),
    ],
)
def test_sfitspectra_forwards_nuts_settings(tmp_path, monkeypatch, extra, expected):
    root, script_path, csv_path, _, env = _fresh_workspace(tmp_path)
    result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path),
                        "--description", "nuts", *extra)
    assert result.returncode == 0, result.stderr
    generated = next((root / "hpc_scripts/submit/jaxqsofit").glob("submit_*.sbatch")).read_text()
    for key, value in zip(("NUTS_WARMUP", "NUTS_SAMPLES", "NUTS_CHAINS"), expected):
        assert f'export {key}="{value}"\n' in generated
    command = _generated_fit_command(root, script_path, generated, monkeypatch)
    for option, value in zip(("--nuts-warmup", "--nuts-samples", "--nuts-chains"), expected):
        assert command.count(option) == 1
        assert command[command.index(option) + 1] == value


@pytest.mark.parametrize("bad", [("--nuts-chains", "0"), ("--nuts-warmup", "-1"), ("--nuts-samples", "x")])
def test_sfitspectra_rejects_invalid_nuts_settings(tmp_path, bad):
    root, script_path, csv_path, calls_path, env = _fresh_workspace(tmp_path)
    result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path),
                        "--description", "nuts", *bad)
    assert result.returncode == 2
    assert "expected a positive integer" in result.stderr
    assert _sbatch_calls(calls_path) == []


def test_sfitspectra_settings_block_is_unchanged():
    """Local edits to the settings block (memory, time) must keep applying cleanly."""
    source = SCRIPT.read_text(encoding="utf-8")
    block = source.split("# 1. Define your job settings here", 1)[1].split("# Retry helpers", 1)[0]
    assert 'time_limit = "3:00:00"' in block
    assert 'mem = "40G"' in block
    assert "nuts" not in block.lower()


def test_sfitspectra_resource_options_reach_sbatch_and_warn_on_sequential_chains(tmp_path):
    root, script_path, csv_path, _, env = _fresh_workspace(tmp_path)
    result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path),
                        "--description", "res", "--nuts-chains", "4", "--cpus-per-task", "2",
                        "--mem", "64G", "--time-limit", "6:00:00")
    assert result.returncode == 0, result.stderr
    assert "chains will run one after another" in result.stderr
    generated = next((root / "hpc_scripts/submit/jaxqsofit").glob("submit_*.sbatch")).read_text()
    assert "#SBATCH --cpus-per-task=2" in generated
    assert "#SBATCH --mem=64G" in generated
    assert "#SBATCH --time=6:00:00" in generated
    assert "export CPUS_PER_TASK=2" in generated
    quiet = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path),
                       "--description", "res2", "--nuts-chains", "2", "--cpus-per-task", "2")
    assert quiet.returncode == 0, quiet.stderr
    assert "one after another" not in quiet.stderr


@pytest.mark.parametrize("bad", [("--mem", "lots"), ("--time-limit", "3h"), ("--cpus-per-task", "0")])
def test_sfitspectra_rejects_invalid_resources(tmp_path, bad):
    root, script_path, csv_path, calls_path, env = _fresh_workspace(tmp_path)
    result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path), "--description", "r", *bad)
    assert result.returncode == 2
    assert _sbatch_calls(calls_path) == []


def test_sfitspectra_resume_only_passes_the_resumed_catalogue_for_comparison(tmp_path, monkeypatch):
    root, script_path, csv_path, _, env = _fresh_workspace(tmp_path)
    (root / "results/data/jaxqsofit/old/all").mkdir(parents=True)
    result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path),
                        "--description", "up", "--resume", "old", "--resume-only",
                        "--allow-resume-without-submission-record")
    assert result.returncode == 0, result.stderr
    assert "will not be compared" in result.stderr
    (root / "results/data/jaxqsofit/old.h5").write_bytes(b"")
    result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path),
                        "--description", "up2", "--resume", "old", "--resume-only",
                        "--allow-resume-without-submission-record")
    assert result.returncode == 0, result.stderr
    generated = sorted((root / "hpc_scripts/submit/jaxqsofit").glob("submit_*up2*.sbatch"))[-1].read_text()
    command = _generated_fit_command(root, script_path, generated, monkeypatch)
    assert command[command.index("--resume-compare-catalog") + 1] == "results/data/jaxqsofit/old.h5"


def test_sfitspectra_retry_accepts_memory_and_time_overrides(tmp_path):
    rows = [f"100_1|{RETRY_JOB_NAME}|OUT_OF_MEMORY"]
    root, script_path, _saved, _ids, calls_path, env = _retry_workspace(tmp_path, rows)
    result = _run_retry(root, script_path, env, "--mem", "80G", "--time-limit", "8:00:00")
    assert result.returncode == 0, result.stderr
    call = _sbatch_calls(calls_path)[0]
    assert "--mem=80G" in call and "--time=8:00:00" in call and "--cpus-per-task=1" in call
    rejected = _run_retry(root, script_path, env, "--cpus-per-task", "4")
    assert rejected.returncode == 2


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ((), []),
        (("--drop-sub-lyman-photometry",), ["--drop-sub-lyman-photometry"]),
        (
            ("--drop-sub-lyman-photometry", "--sub-lyman-max-fraction", "0.25"),
            ["--drop-sub-lyman-photometry", "--sub-lyman-max-fraction", "0.25"],
        ),
        (
            ("--drop-sub-lyman-photometry", "--sub-lyman-rest-wavelength", "1216"),
            ["--drop-sub-lyman-photometry", "--sub-lyman-rest-wavelength", "1216.0"],
        ),
    ],
)
def test_sfitspectra_forwards_sub_lyman_exclusion_only_when_given(tmp_path, monkeypatch, extra, expected):
    root, script_path, csv_path, _, env = _fresh_workspace(tmp_path)
    result = _run_fresh(root, script_path, env, '--chisq-csv', str(csv_path),
                        '--description', 'sublyman', *extra)
    assert result.returncode == 0, result.stderr
    generated = next((root / 'hpc_scripts/submit/jaxqsofit').glob('submit_*.sbatch')).read_text()
    command = _generated_fit_command(root, script_path, generated, monkeypatch)
    present = [arg for arg in command if 'sub-lyman' in arg]
    assert present == [arg for arg in expected if arg.startswith('--')]
    for option, value in zip(expected[1::2], expected[2::2]):
        assert command[command.index(option) + 1] == value


def test_sfitspectra_rejects_sub_lyman_parameters_without_the_flag(tmp_path):
    root, script_path, csv_path, calls_path, env = _fresh_workspace(tmp_path)
    result = _run_fresh(root, script_path, env, '--chisq-csv', str(csv_path),
                        '--description', 'sublyman', '--sub-lyman-max-fraction', '0.3')
    assert result.returncode == 2
    assert 'require --drop-sub-lyman-photometry' in result.stderr
    assert _sbatch_calls(calls_path) == []


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--drop-sub-lyman-photometry", "--sub-lyman-max-fraction", "1.0"), "must lie in [0, 1)"),
        (("--drop-sub-lyman-photometry", "--sub-lyman-max-fraction", "-0.1"), "must lie in [0, 1)"),
        (("--drop-sub-lyman-photometry", "--sub-lyman-rest-wavelength", "0"), "positive and finite"),
        (("--drop-sub-lyman-photometry", "--sub-lyman-rest-wavelength", "nan"), "positive and finite"),
    ],
)
def test_sfitspectra_validates_sub_lyman_ranges_before_submitting(tmp_path, extra, message):
    root, script_path, csv_path, calls_path, env = _fresh_workspace(tmp_path)
    result = _run_fresh(root, script_path, env, '--chisq-csv', str(csv_path),
                        '--description', 'sublyman', *extra)
    assert result.returncode == 2
    assert message in result.stderr
    assert _sbatch_calls(calls_path) == []
    assert not (root / 'hpc_scripts/submit/jaxqsofit').exists()



SELECTION_CSV = "object_id,plate,mjd,fiber,z\n1,1000,52000,10,2.1\n2,1001,52001,11,1.3\n"
SELECTED_SPECTRA = ("spec-1000-52000-0010.fits", "spec-1001-52001-0011.fits")


def _use_full_qvc_package(root):
    """The selection check imports the fitter's loader, so expose the real package."""
    shutil.rmtree(root / "src" / "qvc")
    (root / "src" / "qvc").symlink_to(SCRIPT.parents[1] / "src" / "qvc", target_is_directory=True)


def _selection_workspace(tmp_path, cache="data/spectra_cache_all"):
    root, script_path, csv_path, calls_path, env = _fresh_workspace(tmp_path)
    _use_full_qvc_package(root)
    (root / "data" / "selection.csv").write_text(SELECTION_CSV, encoding="utf-8")
    cache_path = root / cache
    cache_path.mkdir(parents=True, exist_ok=True)
    for name in SELECTED_SPECTRA:
        (cache_path / name).write_bytes(b"")
    return root, script_path, csv_path, calls_path, env


def _submission_inputs(generated):
    encoded = re.search(r'^export QVC_SUBMISSION_PROVENANCE_B64="([^"]*)"$', generated, re.M).group(1)
    return json.loads(base64.b64decode(encoded))["resolved"]["inputs"]


def _only_sbatch(root, pattern="submit_*.sbatch"):
    (path,) = sorted((root / "hpc_scripts/submit/jaxqsofit").glob(pattern))
    return path


def test_sfitspectra_spectrum_selection_and_cache_dir_default_unchanged(tmp_path, monkeypatch):
    root, script_path, csv_path, _, env = _fresh_workspace(tmp_path)
    result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path), "--description", "plain")
    assert result.returncode == 0, result.stderr
    generated = _only_sbatch(root).read_text()
    assert 'export CACHE_DIR="data/spectra_cache_all"' in generated
    command = _generated_fit_command(root, script_path, generated, monkeypatch)
    assert command[command.index("--cache-dir") + 1] == "data/spectra_cache_all"
    assert "--sdss-spectrum-selection-path" not in command
    inputs = _submission_inputs(generated)
    assert inputs["cache_dir"] == "data/spectra_cache_all"
    for key in ("sdss_spectrum_selection_path", "sdss_spectrum_selection_sha256",
                "sdss_spectrum_selection_frozen_path"):
        assert inputs[key] is None
    assert not list((root / "hpc_scripts/submit/jaxqsofit").glob("*_sdss_spectrum_selection*"))


def test_sfitspectra_forwards_a_frozen_spectrum_selection_and_cache_dir(tmp_path, monkeypatch):
    root, script_path, csv_path, calls_path, env = _selection_workspace(tmp_path, cache="data/other_cache")
    selection = root / "data" / "selection.csv"
    result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path), "--description", "sel",
                        "--sdss-spectrum-selection-path", "data/selection.csv",
                        "--cache-dir", "data/other_cache")
    assert result.returncode == 0, result.stderr
    assert len(_sbatch_calls(calls_path)) == 1
    saved = _only_sbatch(root)
    generated = saved.read_text()
    prefix = saved.name[len("submit_"):-len(".sbatch")]
    frozen = root / "hpc_scripts/submit/jaxqsofit" / f"{prefix}_sdss_spectrum_selection.csv"
    assert frozen.read_bytes() == selection.read_bytes()
    assert not os.access(frozen, os.W_OK)
    resolved_cache = str((root / "data" / "other_cache").resolve())
    assert f'export CACHE_DIR="{resolved_cache}"' in generated
    command = _generated_fit_command(root, script_path, generated, monkeypatch)
    assert command[command.index("--sdss-spectrum-selection-path") + 1] == str(frozen)
    assert command[command.index("--cache-dir") + 1] == resolved_cache
    inputs = _submission_inputs(generated)
    assert inputs["cache_dir"] == resolved_cache
    assert inputs["sdss_spectrum_selection_path"] == str(selection.resolve())
    assert inputs["sdss_spectrum_selection_frozen_path"] == str(frozen)
    assert inputs["sdss_spectrum_selection_sha256"] == hashlib.sha256(selection.read_bytes()).hexdigest()
    # Editing the original after submission does not reach the tasks.
    selection.write_text(SELECTION_CSV.replace("2.1", "2.2"), encoding="utf-8")
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == inputs["sdss_spectrum_selection_sha256"]


def _write_ecsv(path, columns):
    from astropy.table import Table

    Table(columns).write(path, format="ascii.ecsv", overwrite=True)


GOOD_COLUMNS = {"object_id": ["1", "2"], "plate": [1000, 1001], "mjd": [52000, 52001],
                "fiber": [10, 11], "z": [2.1, 1.3]}


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("no_z.csv", "object_id,plate,mjd,fiber\n1,1000,52000,10\n2,1001,52001,11\n",
         "lacks SDSS spectrum selection columns"),
        ("dup.csv", SELECTION_CSV + "2,1002,52002,12,1.3\n", "duplicate SDSS spectrum selections"),
        ("plate_text.csv", SELECTION_CSV.replace("1000,", "abc,"), "non-finite"),
        ("plate_float.csv", SELECTION_CSV.replace("1000,", "1000.5,"), "positive integer values for plate"),
        ("fiber_zero.csv", SELECTION_CSV.replace(",10,", ",0,"), "positive integer values for fiber"),
        ("z_zero.csv", SELECTION_CSV.replace("2.1", "0"), "strictly positive spectrum redshift"),
        ("z_empty.csv", SELECTION_CSV.replace("2.1", ""), "non-finite"),
        ("z_inf.csv", SELECTION_CSV.replace("2.1", "inf"), "non-finite"),
        ("partial.csv", SELECTION_CSV.rsplit("2,", 1)[0], "lacks 1 submitted object ID(s), e.g. ['2']"),
        ("no_z.ecsv", {k: v for k, v in GOOD_COLUMNS.items() if k != "z"},
         "lacks SDSS spectrum selection columns"),
        ("z_neg.ecsv", {**GOOD_COLUMNS, "z": [2.1, -1.0]}, "strictly positive spectrum redshift"),
        ("partial.ecsv", {k: v[:1] for k, v in GOOD_COLUMNS.items()}, "lacks 1 submitted object ID(s)"),
    ],
)
def test_sfitspectra_refuses_selection_files_the_fitter_rejects(tmp_path, name, content, message):
    root, script_path, csv_path, calls_path, env = _selection_workspace(tmp_path)
    path = root / "data" / name
    if isinstance(content, dict):
        _write_ecsv(path, content)
    else:
        path.write_text(content, encoding="utf-8")
    result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path), "--description", "bad",
                        "--sdss-spectrum-selection-path", f"data/{name}")
    assert result.returncode == 2, result.stderr
    assert message in result.stderr
    assert _sbatch_calls(calls_path) == []
    assert not (root / "hpc_scripts/submit/jaxqsofit").exists()


def test_sfitspectra_accepts_a_valid_ecsv_selection(tmp_path):
    root, script_path, csv_path, calls_path, env = _selection_workspace(tmp_path)
    _write_ecsv(root / "data" / "selection.ecsv", GOOD_COLUMNS)
    result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path), "--description", "ecsv",
                        "--sdss-spectrum-selection-path", "data/selection.ecsv")
    assert result.returncode == 0, result.stderr
    assert list((root / "hpc_scripts/submit/jaxqsofit").glob("*_sdss_spectrum_selection.ecsv"))


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--sdss-spectrum-selection-path", "data/missing.csv"), "not found or not readable"),
        (("--sdss-spectrum-selection-path", "data/selection.fits"), "must be a .csv, .txt or .ecsv"),
        (("--sdss-spectrum-selection-path", "data/selection.csv", "--fit-script", "fit_spectra.py"),
         "supported only with fit_spectra_jaxsedfit_joint.py"),
        (("--cache-dir", "data/missing_cache"), "--cache-dir not found or not a directory"),
        (("--cache-dir", "data/objects.csv"), "--cache-dir not found or not a directory"),
        (("--cache-dir", "data/has space"), "whitespace or shell characters"),
    ],
)
def test_sfitspectra_validates_spectrum_options_before_submitting(tmp_path, extra, message):
    root, script_path, csv_path, calls_path, env = _selection_workspace(tmp_path)
    (root / "data" / "selection.fits").write_text(SELECTION_CSV, encoding="utf-8")
    (root / "data" / "has space").mkdir()
    result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path),
                        "--description", "bad", *extra)
    assert result.returncode == 2
    assert message in result.stderr
    assert _sbatch_calls(calls_path) == []
    assert not (root / "hpc_scripts/submit/jaxqsofit").exists()


def test_sfitspectra_refuses_selected_spectra_missing_from_the_cache(tmp_path):
    root, script_path, csv_path, calls_path, env = _selection_workspace(tmp_path)
    (root / "data/spectra_cache_all" / SELECTED_SPECTRA[1]).unlink()
    (root / "data/empty_cache").mkdir()
    result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path), "--description", "nospec",
                        "--sdss-spectrum-selection-path", "data/selection.csv")
    assert result.returncode == 2
    assert "1 selected spectrum file(s) are not in the cache directory" in result.stderr
    assert SELECTED_SPECTRA[1] in result.stderr and SELECTED_SPECTRA[0] not in result.stderr
    result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path), "--description", "nospec2",
                        "--sdss-spectrum-selection-path", "data/selection.csv",
                        "--cache-dir", "data/empty_cache")
    assert result.returncode == 2
    assert all(name in result.stderr for name in SELECTED_SPECTRA)
    assert _sbatch_calls(calls_path) == []
    assert not (root / "hpc_scripts/submit/jaxqsofit").exists()


def _original_run(root, script_path, csv_path, env, description, *extra):
    result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path),
                        "--description", description, *extra)
    assert result.returncode == 0, result.stderr
    run_name = _only_sbatch(root, f"submit_*{description}.sbatch").name[len("submit_"):-len(".sbatch")]
    (root / "results/data/jaxqsofit" / run_name / "all").mkdir(parents=True)
    return run_name


def _frozen_copy(root, run_name):
    return root / "hpc_scripts/submit/jaxqsofit" / f"{run_name}_sdss_spectrum_selection.csv"


def test_sfitspectra_resume_requires_the_original_spectrum_selection(tmp_path):
    root, script_path, csv_path, calls_path, env = _selection_workspace(tmp_path)
    selection = root / "data" / "selection.csv"
    old = _original_run(root, script_path, csv_path, env, "orig",
                        "--sdss-spectrum-selection-path", "data/selection.csv")
    plain = _original_run(root, script_path, csv_path, env, "origplain")

    def resume(description, run_name, *extra):
        return _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path),
                          "--description", description, "--resume", run_name, *extra)

    same = resume("same", old, "--resume-only", "--sdss-spectrum-selection-path", str(selection))
    assert same.returncode == 0, same.stderr
    hybrid = resume("hybrid", old, "--sdss-spectrum-selection-path", "data/selection.csv")
    assert hybrid.returncode == 0, hybrid.stderr
    # The content, not the path, must match: a copy elsewhere is accepted.
    (root / "data" / "copy.csv").write_bytes(selection.read_bytes())
    copied = resume("copied", old, "--resume-only", "--sdss-spectrum-selection-path", "data/copy.csv")
    assert copied.returncode == 0, copied.stderr
    calls_before = len(_sbatch_calls(calls_path))

    dropped = resume("dropped", old, "--resume-only")
    assert dropped.returncode == 2
    assert "must not mix spectra" in dropped.stderr
    added = resume("added", plain, "--resume-only", "--sdss-spectrum-selection-path", "data/selection.csv")
    assert added.returncode == 2
    assert "none (DR16Q primary spectra)" in added.stderr
    selection.write_text(SELECTION_CSV.replace("2.1", "2.2"), encoding="utf-8")
    edited = resume("edited", old, "--resume-only", "--sdss-spectrum-selection-path", "data/selection.csv")
    assert edited.returncode == 2
    assert "same content" in edited.stderr
    assert len(_sbatch_calls(calls_path)) == calls_before


def test_sfitspectra_resume_refuses_an_altered_or_missing_frozen_selection(tmp_path):
    root, script_path, csv_path, calls_path, env = _selection_workspace(tmp_path)
    old = _original_run(root, script_path, csv_path, env, "orig",
                        "--sdss-spectrum-selection-path", "data/selection.csv")
    frozen = _frozen_copy(root, old)
    frozen.chmod(0o644)
    frozen.write_text(SELECTION_CSV.replace("2.1", "2.2"), encoding="utf-8")
    args = ("--chisq-csv", str(csv_path), "--resume", old, "--resume-only",
            "--sdss-spectrum-selection-path", "data/selection.csv")
    altered = _run_fresh(root, script_path, env, *args, "--description", "altered")
    assert altered.returncode == 2
    assert "changed after submission" in altered.stderr
    frozen.unlink()
    missing = _run_fresh(root, script_path, env, *args, "--description", "missing")
    assert missing.returncode == 2
    assert "is missing" in missing.stderr
    assert len(_sbatch_calls(calls_path)) == 1


def _write_fake_sacct(root, run_name):
    _write_executable(
        root / "fake-bin" / "sacct",
        "#!/usr/bin/env python3\n"
        f"print('100_0|{run_name}|FAILED')\nprint('100_1|{run_name}|COMPLETED')\n",
    )


def test_sfitspectra_retry_uses_the_frozen_selection_and_refuses_when_it_changed(tmp_path):
    root, script_path, csv_path, calls_path, env = _selection_workspace(tmp_path)
    old = _original_run(root, script_path, csv_path, env, "orig",
                        "--sdss-spectrum-selection-path", "data/selection.csv")
    _write_fake_sacct(root, old)
    saved = _only_sbatch(root, f"submit_{old}.sbatch")
    frozen = _frozen_copy(root, old)
    assert f"--sdss-spectrum-selection-path {frozen}" in saved.read_text()
    (root / "data" / "selection.csv").write_text(SELECTION_CSV.replace("2.1", "2.2"), encoding="utf-8")
    retried = _run_fresh(root, script_path, env, "--retry", old)
    assert retried.returncode == 0, retried.stderr
    assert _sbatch_calls(calls_path)[-1][-1] == str(saved)

    frozen.chmod(0o644)
    frozen.write_text(SELECTION_CSV.replace("2.1", "2.2"), encoding="utf-8")
    altered = _run_fresh(root, script_path, env, "--retry", old)
    assert altered.returncode != 0
    assert "changed after submission" in altered.stderr
    frozen.unlink()
    missing = _run_fresh(root, script_path, env, "--retry", old)
    assert missing.returncode != 0
    assert "is missing" in missing.stderr
    assert len(_sbatch_calls(calls_path)) == 2


def test_sfitspectra_resume_requires_the_original_cache_dir_for_fresh_fallbacks(tmp_path):
    root, script_path, csv_path, calls_path, env = _fresh_workspace(tmp_path)
    (root / "data" / "other_cache").mkdir()
    (root / "data" / "spectra_cache_all").mkdir()
    old = _original_run(root, script_path, csv_path, env, "orig")

    def resume(description, *extra):
        return _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path),
                          "--description", description, "--resume", old, *extra)

    assert resume("default").returncode == 0
    assert resume("explicitsame", "--cache-dir", "data/spectra_cache_all").returncode == 0
    assert resume("only", "--resume-only", "--cache-dir", "data/other_cache").returncode == 0
    calls_before = len(_sbatch_calls(calls_path))
    changed = resume("changed", "--cache-dir", "data/other_cache")
    assert changed.returncode == 2
    assert "fresh fallback fits would read spectra from another cache" in changed.stderr
    assert len(_sbatch_calls(calls_path)) == calls_before


def test_sfitspectra_resume_of_a_run_recorded_before_spectrum_options_works(tmp_path):
    """Records written before this feature lack the new keys and resume as before."""
    root, script_path, csv_path, calls_path, env = _fresh_workspace(tmp_path)
    old = _original_run(root, script_path, csv_path, env, "orig")
    saved = _only_sbatch(root, f"submit_{old}.sbatch")
    text = saved.read_text()
    encoded = re.search(r'^export QVC_SUBMISSION_PROVENANCE_B64="([^"]*)"$', text, re.M).group(1)
    record = json.loads(base64.b64decode(encoded))
    for key in ("sdss_spectrum_selection_path", "sdss_spectrum_selection_sha256",
                "sdss_spectrum_selection_frozen_path"):
        del record["resolved"]["inputs"][key]
    saved.write_text(text.replace(encoded, base64.b64encode(json.dumps(record).encode()).decode()))
    for extra in ((), ("--resume-only",)):
        result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path),
                            "--description", f"again{len(extra)}", "--resume", old, *extra)
        assert result.returncode == 0, result.stderr
        assert "submission record" not in result.stderr
    assert len(_sbatch_calls(calls_path)) == 3


def test_sfitspectra_resume_without_record_needs_the_explicit_flag(tmp_path):
    root, script_path, csv_path, calls_path, env = _selection_workspace(tmp_path)
    (root / "results/data/jaxqsofit/old/all").mkdir(parents=True)
    submit_dir = root / "hpc_scripts/submit/jaxqsofit"
    base = ("--chisq-csv", str(csv_path), "--resume", "old", "--resume-only")
    refused = _run_fresh(root, script_path, env, *base, "--description", "norec")
    assert refused.returncode == 2
    assert "no readable submission record" in refused.stderr
    assert "--allow-resume-without-submission-record" in refused.stderr
    submit_dir.mkdir(parents=True)
    (submit_dir / "submit_old.sbatch").write_text('#!/usr/bin/env bash\nexport PREFIX="old"\n')
    unreadable = _run_fresh(root, script_path, env, *base, "--description", "norec1")
    assert unreadable.returncode == 2
    assert "no readable submission record" in unreadable.stderr
    with_selection = _run_fresh(root, script_path, env, *base, "--description", "norec2",
                                "--allow-resume-without-submission-record",
                                "--sdss-spectrum-selection-path", "data/selection.csv")
    assert with_selection.returncode == 2
    assert "need the resumed run's submission record" in with_selection.stderr
    assert _sbatch_calls(calls_path) == []
    allowed = _run_fresh(root, script_path, env, *base, "--description", "norec3",
                         "--allow-resume-without-submission-record")
    assert allowed.returncode == 0, allowed.stderr
    assert "cannot check that the resumed run used the same SDSS spectrum selection" in allowed.stderr
    assert len(_sbatch_calls(calls_path)) == 1


PHOTOMETRY_CSV = (
    "object_id,filter_name,flux_mjy,flux_err_mjy\n"
    "1,galex_fuv,0.1,0.01\n1,u_sdss,0.2,0.02\n"
    "2,wise_w1,0.3,0.03\n"
)


def _photometry_workspace(tmp_path):
    root, script_path, csv_path, calls_path, env = _fresh_workspace(tmp_path)
    _use_full_qvc_package(root)
    (root / "data" / "phot.csv").write_text(PHOTOMETRY_CSV, encoding="utf-8")
    return root, script_path, csv_path, calls_path, env


def test_sfitspectra_default_sed_photometry_is_unchanged(tmp_path, monkeypatch):
    root, script_path, csv_path, _, env = _fresh_workspace(tmp_path)
    result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path), "--description", "plain")
    assert result.returncode == 0, result.stderr
    generated = _only_sbatch(root).read_text()
    default = "data/jul14_master_input_file_chisqgt20_bandwagon_photometry.csv"
    assert f'export SED_PHOTOMETRY_PATH="{default}"' in generated
    command = _generated_fit_command(root, script_path, generated, monkeypatch)
    assert command[command.index("--sed-photometry-path") + 1] == default
    inputs = _submission_inputs(generated)
    assert inputs["sed_photometry_path"] == default
    assert inputs["sed_photometry_sha256"] is None


def test_sfitspectra_forwards_and_records_sed_photometry_path(tmp_path, monkeypatch):
    root, script_path, csv_path, calls_path, env = _photometry_workspace(tmp_path)
    photometry = root / "data" / "phot.csv"
    result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path), "--description", "phot",
                        "--sed-photometry-path", "data/phot.csv")
    assert result.returncode == 0, result.stderr
    assert len(_sbatch_calls(calls_path)) == 1
    generated = _only_sbatch(root).read_text()
    resolved = str(photometry.resolve())
    assert f'export SED_PHOTOMETRY_PATH="{resolved}"' in generated
    command = _generated_fit_command(root, script_path, generated, monkeypatch)
    assert command[command.index("--sed-photometry-path") + 1] == resolved
    inputs = _submission_inputs(generated)
    assert inputs["sed_photometry_path"] == resolved
    assert inputs["sed_photometry_sha256"] == hashlib.sha256(photometry.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("extra", "content", "message"),
    [
        (("--sed-photometry-path", "data/missing.csv"), None, "not found or not readable"),
        (("--sed-photometry-path", "data/phot.csv", "--fit-script", "fit_spectra.py"), None,
         "supported only with fit_spectra_jaxsedfit_joint.py"),
        (("--sed-photometry-path", "data/phot.csv", "--require-sed-photometry"),
         PHOTOMETRY_CSV.replace("2,wise_w1,0.3,0.03\n", ""),
         "no usable non-SDSS photometry for 1 submitted object ID(s), e.g. ['2']"),
        (("--sed-photometry-path", "data/phot.csv", "--require-sed-photometry"),
         PHOTOMETRY_CSV.replace("1,galex_fuv,0.1,0.01\n", ""),
         "no usable non-SDSS photometry for 1 submitted object ID(s), e.g. ['1']"),
        (("--sed-photometry-path", "data/phot.csv", "--require-sed-photometry"),
         PHOTOMETRY_CSV.replace("0.3,0.03", "0.3,0"), "submitted object ID(s), e.g. ['2']"),
        (("--require-sed-photometry",), None, "submitted object ID(s), e.g. ['1', '2']"),
        (("--require-sed-photometry", "--fit-script", "fit_spectra.py"), None,
         "supported only with fit_spectra_jaxsedfit_joint.py"),
        (("--sed-photometry-path", "data/phot.csv"), "object_id,filter_name,flux_mjy\n1,galex_fuv,0.1\n",
         "lacks saved-SED columns"),
        (("--sed-photometry-path", "data/phot.json"), None, "Unsupported saved SED format"),
    ],
)
def test_sfitspectra_validates_sed_photometry_before_submitting(tmp_path, extra, content, message):
    root, script_path, csv_path, calls_path, env = _photometry_workspace(tmp_path)
    if content is not None:
        (root / "data" / "phot.csv").write_text(content, encoding="utf-8")
    (root / "data" / "phot.json").write_text("{}", encoding="utf-8")
    (root / "data/jul14_master_input_file_chisqgt20_bandwagon_photometry.csv").write_text(
        "object_id,filter_name,flux_mjy,flux_err_mjy\n", encoding="utf-8")
    result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path), "--description", "bad", *extra)
    assert result.returncode == 2, result.stderr
    assert message in result.stderr
    assert _sbatch_calls(calls_path) == []
    assert not (root / "hpc_scripts/submit/jaxqsofit").exists()


def test_sfitspectra_resume_with_sed_photometry_compares_content_for_fresh_fallbacks(tmp_path):
    root, script_path, csv_path, calls_path, env = _photometry_workspace(tmp_path)
    old = _original_run(root, script_path, csv_path, env, "orig", "--sed-photometry-path", "data/phot.csv")
    plain = _original_run(root, script_path, csv_path, env, "origplain")

    def resume(description, run_name, *extra):
        return _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path),
                          "--description", description, "--resume", run_name, *extra)

    (root / "data" / "copy.csv").write_text(PHOTOMETRY_CSV, encoding="utf-8")
    assert resume("same", old, "--sed-photometry-path", "data/copy.csv").returncode == 0
    # --resume-only reads the photometry stored in the bundles.
    assert resume("only", old, "--resume-only").returncode == 0
    (root / "data" / "partial.csv").write_text(PHOTOMETRY_CSV.replace("2,wise_w1,0.3,0.03\n", ""))
    assert resume("onlypartial", old, "--resume-only", "--sed-photometry-path", "data/partial.csv").returncode == 0
    # A plain run's default file, passed explicitly, is the same content.
    default = "data/jul14_master_input_file_chisqgt20_bandwagon_photometry.csv"
    (root / default).write_text(PHOTOMETRY_CSV, encoding="utf-8")
    assert resume("plainsame", plain, "--sed-photometry-path", "data/copy.csv").returncode == 0
    (root / default).write_text("object_id\n", encoding="utf-8")
    calls_before = len(_sbatch_calls(calls_path))

    dropped = resume("dropped", old)
    assert dropped.returncode == 2
    assert "fresh fallback fits would use other photometry" in dropped.stderr
    changed = resume("changed", plain, "--sed-photometry-path", "data/phot.csv")
    assert changed.returncode == 2
    assert "fresh fallback fits would use other photometry" in changed.stderr
    (root / "data" / "copy.csv").write_text(PHOTOMETRY_CSV + "1,wise_w2,0.4,0.04\n", encoding="utf-8")
    edited = resume("edited", old, "--sed-photometry-path", "data/copy.csv")
    assert edited.returncode == 2
    assert len(_sbatch_calls(calls_path)) == calls_before


def test_sfitspectra_retry_rechecks_the_sed_photometry_hash(tmp_path):
    root, script_path, csv_path, calls_path, env = _photometry_workspace(tmp_path)
    old = _original_run(root, script_path, csv_path, env, "orig", "--sed-photometry-path", "data/phot.csv")
    _write_fake_sacct(root, old)
    photometry = root / "data" / "phot.csv"
    retried = _run_fresh(root, script_path, env, "--retry", old)
    assert retried.returncode == 0, retried.stderr
    photometry.write_text(PHOTOMETRY_CSV + "1,wise_w2,0.4,0.04\n", encoding="utf-8")
    altered = _run_fresh(root, script_path, env, "--retry", old)
    assert altered.returncode != 0
    assert "SED photometry file" in altered.stderr and "changed after submission" in altered.stderr
    photometry.unlink()
    missing = _run_fresh(root, script_path, env, "--retry", old)
    assert missing.returncode != 0
    assert "is missing" in missing.stderr
    assert len(_sbatch_calls(calls_path)) == 2


def test_sfitspectra_sed_photometry_warns_lists_and_records_objects_without_rows(tmp_path):
    root, script_path, csv_path, calls_path, env = _photometry_workspace(tmp_path)
    (root / "data" / "phot.csv").write_text(PHOTOMETRY_CSV.replace("2,wise_w1,0.3,0.03\n", ""), encoding="utf-8")
    result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path), "--description", "warn",
                        "--sed-photometry-path", "data/phot.csv")
    assert result.returncode == 0, result.stderr
    assert "WARNING: 1 of 2 submitted object(s) have no usable non-SDSS photometry" in result.stderr
    assert len(_sbatch_calls(calls_path)) == 1
    saved = _only_sbatch(root)
    prefix = saved.name[len("submit_"):-len(".sbatch")]
    listed = f"hpc_scripts/submit/jaxqsofit/{prefix}_objects_without_sed_photometry.txt"
    assert (root / listed).read_text() == "2\n"
    assert listed in result.stderr
    check = json.loads(base64.b64decode(re.search(
        r'^export QVC_SUBMISSION_PROVENANCE_B64="([^"]*)"$', saved.read_text(), re.M).group(1)
    ))["resolved"]["sed_photometry_check"]
    assert check == {"required": False, "objects_without_photometry": 1,
                     "objects_without_photometry_file": listed}


def test_sfitspectra_require_sed_photometry_checks_the_default_table(tmp_path):
    root, script_path, csv_path, calls_path, env = _photometry_workspace(tmp_path)
    default = root / "data/jul14_master_input_file_chisqgt20_bandwagon_photometry.csv"
    default.write_text(PHOTOMETRY_CSV, encoding="utf-8")
    result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path), "--description", "req",
                        "--require-sed-photometry")
    assert result.returncode == 0, result.stderr
    assert "WARNING" not in result.stderr
    saved = _only_sbatch(root)
    generated = saved.read_text()
    assert 'export SED_PHOTOMETRY_PATH="data/jul14_master_input_file_chisqgt20_bandwagon_photometry.csv"' in generated
    record = json.loads(base64.b64decode(re.search(
        r'^export QVC_SUBMISSION_PROVENANCE_B64="([^"]*)"$', generated, re.M).group(1)))["resolved"]
    assert record["sed_photometry_check"] == {"required": True, "objects_without_photometry": 0,
                                              "objects_without_photometry_file": None}
    assert record["inputs"]["sed_photometry_sha256"] is None
    (root / "results/data/jaxqsofit/x/all").mkdir(parents=True)
    only = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path), "--description", "req2",
                      "--require-sed-photometry", "--resume", "x", "--resume-only")
    assert only.returncode == 2
    assert "no effect with --resume-only" in only.stderr


def test_sfitspectra_default_run_does_not_check_or_record_sed_photometry(tmp_path):
    root, script_path, csv_path, calls_path, env = _fresh_workspace(tmp_path)
    result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path), "--description", "plain")
    assert result.returncode == 0, result.stderr
    assert "WARNING" not in result.stderr
    generated = _only_sbatch(root).read_text()
    record = json.loads(base64.b64decode(re.search(
        r'^export QVC_SUBMISSION_PROVENANCE_B64="([^"]*)"$', generated, re.M).group(1)))["resolved"]
    assert record["sed_photometry_check"] is None
    assert not list((root / "hpc_scripts/submit/jaxqsofit").glob("*_objects_without_sed_photometry.txt"))


def test_sfitspectra_resume_warns_when_the_photometry_content_cannot_be_verified(tmp_path):
    root, script_path, csv_path, calls_path, env = _photometry_workspace(tmp_path)
    old = _original_run(root, script_path, csv_path, env, "orig")
    saved = _only_sbatch(root, f"submit_{old}.sbatch")
    text = saved.read_text()
    encoded = re.search(r'^export QVC_SUBMISSION_PROVENANCE_B64="([^"]*)"$', text, re.M).group(1)
    record = json.loads(base64.b64decode(encoded))
    del record["resolved"]["inputs"]["sed_photometry_sha256"]
    saved.write_text(text.replace(encoded, base64.b64encode(json.dumps(record).encode()).decode()))
    default = "data/jul14_master_input_file_chisqgt20_bandwagon_photometry.csv"
    (root / default).write_text(PHOTOMETRY_CSV, encoding="utf-8")
    result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path), "--description", "again",
                        "--resume", old, "--sed-photometry-path", default)
    assert result.returncode == 0, result.stderr
    assert "photometry content that run used cannot be verified" in result.stderr


def test_sfitspectra_fitter_import_leaves_the_sbatch_environment_unchanged(tmp_path):
    root, script_path, csv_path, calls_path, env = _photometry_workspace(tmp_path)
    for key in ("XLA_FLAGS", "JAX_PLATFORM_NAME", "NUM_CORES"):
        env.pop(key, None)
    env_dump = root / "sbatch-env.json"
    _write_executable(
        root / "fake-bin" / "sbatch",
        "#!/usr/bin/env python3\nimport json, os\n"
        f"open({str(env_dump)!r}, 'w').write(json.dumps("
        "{k: os.environ.get(k) for k in ('XLA_FLAGS', 'JAX_PLATFORM_NAME', 'NUM_CORES')}))\n",
    )
    result = _run_fresh(root, script_path, env, "--chisq-csv", str(csv_path), "--description", "env",
                        "--sed-photometry-path", "data/phot.csv")
    assert result.returncode == 0, result.stderr
    assert json.loads(env_dump.read_text()) == {"XLA_FLAGS": None, "JAX_PLATFORM_NAME": None, "NUM_CORES": None}
