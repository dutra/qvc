#!/usr/bin/env xonsh

import argparse
import hashlib
import math
import os
import re
import stat
import sys
import subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qvc.provenance import (
    PROVENANCE_ENV,
    RETRY_ENV,
    decode_record,
    encode_record,
    submission_record,
)

# ==========================================
# 1. Define your job settings here
# ==========================================
partition = "day"
time_limit = "3:00:00"

# Number of object_ids per array task
chunk_size = 1

# Number of CPUs used by fit_spectra.py
nproc = 1
cpus_per_task = 1
mem = "40G"

#fit_script = "fit_spectra.py"
fit_script = "fit_spectra_jaxsedfit_joint.py"

# Required only by fit_spectra_jaxsedfit_joint.py
sed_photometry_path = "data/jul14_master_input_file_chisqgt20_bandwagon_photometry.csv"

# Optional exclusion file
#exclude_csv = "results/data/jaxqsofit/aug28_0111pm_spectrafit_f6a63a3_chisqgt20_N8000_nested_fhostpsf_delayedburst_galexauto_successful_object_ids.csv"
exclude_csv = None

# Direct path to Python inside the Conda env
python_bin = "/home/id255/.conda/envs/jaxcpu2/bin/python"

# Optional: cache/data locations
qvc_data_dir = "/home/id255/project_pi_pn38/id255/qvc/data"
cache_dir = "data/spectra_cache_all"


# ==========================================
# Retry helpers
# ==========================================
RETRY_ACCOUNTING_STARTTIME = "now-7days"
RETRY_BATCH_LIMIT = 10_000
ARRAY_TASK_RE = re.compile(r"^(?P<parent_id>\d+)_(?P<task_id>\d+)$")
SCALAR_JOB_RE = re.compile(r"^\d+$")
ACTIVE_STATES = {
    "COMPLETING",
    "CONFIGURING",
    "PENDING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "RESIZING",
    "RUNNING",
    "SIGNALING",
    "STAGE_OUT",
    "STOPPED",
    "SUSPENDED",
}
TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}
KNOWN_STATES = ACTIVE_STATES | TERMINAL_STATES


def normalize_job_name(value, option):
    job_name = str(value).strip()
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", job_name)
    if not job_name or normalized != job_name or job_name in {".", ".."}:
        raise ValueError(f"{option} must be a full job name, not a path")
    return job_name


def normalize_slurm_state(value):
    state = str(value).strip().split()[0].rstrip("+") if str(value).strip() else ""
    if state not in KNOWN_STATES:
        raise ValueError(f"Unrecognized Slurm state {state or '<empty>'!r}")
    return state


def parse_saved_export(script_content, name):
    match = re.search(
        rf'^export {re.escape(name)}=(?:"([^"]*)"|([^\s#]+))\s*$',
        script_content,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"Saved submission script lacks export {name}")
    return match.group(1) if match.group(1) is not None else match.group(2)


def repo_relative_path(value):
    """Resolve a path as the jobs see it: relative paths are under REPO_ROOT."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def saved_submission_inputs(script_content):
    """Return resolved.inputs of the record embedded in a saved script, or None."""
    try:
        record = decode_record(parse_saved_export(script_content, PROVENANCE_ENV))
    except ValueError:
        return None
    inputs = ((record or {}).get("resolved") or {}).get("inputs")
    return inputs if isinstance(inputs, dict) else None


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recorded_photometry_problem(inputs):
    """Return why a recorded --sed-photometry-path file is unusable, or None."""
    recorded = (inputs or {}).get("sed_photometry_sha256")
    if not recorded:
        return None
    path = repo_relative_path(inputs.get("sed_photometry_path") or "")
    if not path.is_file():
        return f"its SED photometry file {path} is missing"
    actual = file_sha256(path)
    if actual != recorded:
        return (
            f"its SED photometry file {path} changed after submission "
            f"(sha256 {actual}, recorded {recorded})"
        )
    return None


def frozen_selection_problem(inputs):
    """Return why a recorded frozen SDSS spectrum selection is unusable, or None."""
    frozen = (inputs or {}).get("sdss_spectrum_selection_frozen_path")
    if not frozen:
        return None
    frozen = Path(frozen)
    if not frozen.is_file():
        return f"its frozen SDSS spectrum selection {frozen} is missing"
    recorded = inputs.get("sdss_spectrum_selection_sha256")
    actual = hashlib.sha256(frozen.read_bytes()).hexdigest()
    if actual != recorded:
        return (
            f"its frozen SDSS spectrum selection {frozen} changed after submission "
            f"(sha256 {actual}, recorded {recorded})"
        )
    return None


def load_retry_artifacts(job_name):
    submit_dir = REPO_ROOT / "hpc_scripts" / "submit" / "jaxqsofit"
    saved_script = submit_dir / f"submit_{job_name}.sbatch"
    object_ids_path = submit_dir / f"{job_name}_object_ids.txt"
    if not saved_script.is_file():
        raise FileNotFoundError(f"Original submission script not found: {saved_script}")
    if not object_ids_path.is_file():
        raise FileNotFoundError(
            f"Original object-ID manifest not found: {object_ids_path}"
        )

    script_content = saved_script.read_text(encoding="utf-8")
    job_name_match = re.search(
        r"^#SBATCH\s+--job-name=(\S+)\s*$",
        script_content,
        flags=re.MULTILINE,
    )
    saved_job_name = job_name_match.group(1) if job_name_match is not None else None
    saved_prefix = parse_saved_export(script_content, "PREFIX")
    saved_object_ids = parse_saved_export(script_content, "OBJECT_IDS_FILE")
    if saved_job_name != job_name or saved_prefix != job_name:
        raise ValueError(
            "Saved submission identity does not match --retry: "
            f"job_name={saved_job_name!r}, prefix={saved_prefix!r}"
        )

    referenced_object_ids = Path(saved_object_ids)
    if not referenced_object_ids.is_absolute():
        referenced_object_ids = REPO_ROOT / referenced_object_ids
    if referenced_object_ids.resolve() != object_ids_path.resolve():
        raise ValueError(
            "Saved OBJECT_IDS_FILE does not match the canonical retry manifest: "
            f"{saved_object_ids!r}"
        )

    try:
        saved_chunk_size = int(parse_saved_export(script_content, "CHUNK_SIZE"))
        saved_nproc = int(parse_saved_export(script_content, "NPROC"))
    except ValueError as exc:
        raise ValueError("Saved CHUNK_SIZE and NPROC must be integers") from exc
    if saved_chunk_size <= 0 or saved_nproc <= 0:
        raise ValueError("Saved CHUNK_SIZE and NPROC must be positive")
    if cpus_per_task < saved_nproc:
        raise ValueError(
            f"Current cpus_per_task={cpus_per_task} is fewer than the saved "
            f"NPROC={saved_nproc}"
        )

    object_count = sum(
        bool(line.strip())
        for line in object_ids_path.read_text(encoding="utf-8").splitlines()
    )
    if object_count == 0:
        raise ValueError(f"Original object-ID manifest is empty: {object_ids_path}")
    num_tasks = math.ceil(object_count / saved_chunk_size)
    return saved_script, num_tasks


def get_retry_accounting_rows(job_name):
    cmd = [
        "sacct",
        "--array",
        "--noheader",
        "--parsable2",
        f"--starttime={RETRY_ACCOUNTING_STARTTIME}",
        "--format=JobID%128,JobName%256,State%64",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    array_rows = []
    scalar_rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        job_id, row_job_name, raw_state = parts[:3]
        if row_job_name != job_name:
            continue
        array_match = ARRAY_TASK_RE.fullmatch(job_id)
        if array_match is not None:
            array_rows.append(
                {
                    "parent_id": int(array_match.group("parent_id")),
                    "task_id": int(array_match.group("task_id")),
                    "state": normalize_slurm_state(raw_state),
                }
            )
        elif SCALAR_JOB_RE.fullmatch(job_id):
            scalar_rows.append(
                {
                    "parent_id": int(job_id),
                    "task_id": 0,
                    "state": normalize_slurm_state(raw_state),
                }
            )

    rows = array_rows if array_rows else scalar_rows
    if not rows:
        raise RuntimeError(
            f"No allocation rows found for {job_name!r} since "
            f"{RETRY_ACCOUNTING_STARTTIME}"
        )
    return rows


def latest_task_attempts(rows):
    latest = {}
    for row in rows:
        task_id = row["task_id"]
        current = latest.get(task_id)
        if current is None or row["parent_id"] >= current["parent_id"]:
            latest[task_id] = row
    return latest


def format_array_spec(task_ids):
    ordered = sorted(set(task_ids))
    if not ordered:
        raise ValueError("Cannot format an empty Slurm array selection")
    ranges = []
    start = previous = ordered[0]
    for task_id in ordered[1:]:
        if task_id == previous + 1:
            previous = task_id
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = task_id
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def retry_unsuccessful_tasks(job_name):
    saved_script, num_tasks = load_retry_artifacts(job_name)
    saved_inputs = saved_submission_inputs(saved_script.read_text(encoding="utf-8"))
    problem = frozen_selection_problem(saved_inputs) or recorded_photometry_problem(saved_inputs)
    if problem:
        raise ValueError(
            f"Refusing to retry {job_name}: {problem}; the retried tasks would fit "
            "other inputs than the original run."
        )
    attempts = latest_task_attempts(get_retry_accounting_rows(job_name))
    invalid_task_ids = sorted(
        task_id for task_id in attempts if task_id < 0 or task_id >= num_tasks
    )
    if invalid_task_ids:
        raise ValueError(
            f"Accounting task IDs {invalid_task_ids} are outside the original task "
            f"range 0-{num_tasks - 1}"
        )

    retry_rows = sorted(
        (
            row
            for row in attempts.values()
            if row["state"] in TERMINAL_STATES and row["state"] != "COMPLETED"
        ),
        key=lambda row: row["task_id"],
    )
    if not retry_rows:
        print(f"No unsuccessful terminal tasks to retry for {job_name}.")
        return 0

    print(f"Retrying {len(retry_rows)} task(s) for {job_name}:")
    for row in retry_rows:
        print(f"  task {row['task_id']}: {row['state']}")

    task_ids = [row["task_id"] for row in retry_rows]
    retry_record = submission_record(
        "hpc_scripts/sfitspectra.xsh",
        sys.argv,
        {
            "retry_job_name": job_name,
            "task_ids": task_ids,
            "resources": {
                "partition": partition,
                "time": time_limit,
                "memory": mem,
                "cpus_per_task": cpus_per_task,
            },
        },
    )
    encoded_retry = encode_record(retry_record)
    for offset in range(0, len(task_ids), RETRY_BATCH_LIMIT):
        array_spec = format_array_spec(task_ids[offset : offset + RETRY_BATCH_LIMIT])
        cmd = [
            "sbatch",
            f"--array={array_spec}",
            f"--partition={partition}",
            f"--time={time_limit}",
            f"--mem={mem}",
            f"--cpus-per-task={cpus_per_task}",
            f"--export=ALL,{RETRY_ENV}={encoded_retry}",
            str(saved_script),
        ]
        print("Submitting retry:")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
    return 0


# ==========================================
# 2. Command-line overrides and job name
# ==========================================
def positive_int(value):
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def slurm_memory(value):
    text = str(value).strip()
    if not re.fullmatch(r"\d+[KMGT]?", text):
        raise argparse.ArgumentTypeError(f"expected SLURM memory such as 40G, got {value!r}")
    return text


def slurm_time(value):
    text = str(value).strip()
    if not re.fullmatch(r"(\d+-)?\d{1,2}(:\d{2}){0,2}", text):
        raise argparse.ArgumentTypeError(
            f"expected a SLURM time such as 3:00:00 or 1-00:00:00, got {value!r}"
        )
    return text


parser = argparse.ArgumentParser(description="Submit spectrum-fitting SLURM jobs.")
parser.add_argument(
    "--chisq-csv",
    default=None,
    metavar="PATH",
    help="CSV file containing the object_id values to fit (required for fresh runs).",
)
parser.add_argument(
    "--description",
    default=None,
    help="Short description appended to the generated SLURM job name (required for fresh runs).",
)
parser.add_argument(
    "--fit-script",
    choices=("fit_spectra.py", "fit_spectra_jaxsedfit_joint.py"),
    default=fit_script,
    help="Spectrum-fitting backend to run.",
)
parser.add_argument(
    "--fit-bal",
    action="store_true",
    help=(
        "Enable JAXSEDFit's built-in BAL absorption components. Supported only "
        "with fit_spectra_jaxsedfit_joint.py."
    ),
)
parser.add_argument(
    "--line-extrapolation-anchor",
    choices=("fitted_lines", "continuum_ew"),
    default=None,
    help=(
        "Anchor for broad lines outside the fitted spectrum, passed to "
        "fit_spectra_jaxsedfit_joint.py. 'continuum_ew' ties them to the fitted "
        "disc continuum through QSOGEN equivalent widths; with --resume it "
        "re-predicts the saved draws without resampling. Absent: not passed."
    ),
)
parser.add_argument(
    "--narrow-line-extrapolation",
    choices=("template", "optical_only", "none"),
    default=None,
    help="Passed through to the joint backend (absent: not passed).",
)
parser.add_argument(
    "--line-extrapolation-ew-scope",
    choices=("blueward", "all"),
    default=None,
    help="Lines anchored by continuum_ew (passed through; absent: not passed).",
)
parser.add_argument(
    "--drop-sub-lyman-photometry",
    action="store_true",
    help=(
        "Leave broadband photometry below the rest-frame Lyman limit out of the "
        "photometric likelihood and chi-squares (passed to "
        "fit_spectra_jaxsedfit_joint.py; off by default). With --resume it "
        "re-predicts the saved draws with those bands dropped; without it each "
        "resumed bundle keeps the setting it was sampled with."
    ),
)
parser.add_argument(
    "--sub-lyman-rest-wavelength",
    type=float,
    default=None,
    help="Rest-frame cut in Angstrom for --drop-sub-lyman-photometry (backend default 912).",
)
parser.add_argument(
    "--sub-lyman-max-fraction",
    type=float,
    default=None,
    help="Largest band response fraction below the cut that keeps a band (backend default 0.5).",
)
parser.add_argument(
    "--sdss-spectrum-selection-path",
    metavar="PATH",
    default=None,
    help=(
        "CSV/ECSV with object_id, plate, mjd, fiber and z for every submitted "
        "object, passed to fit_spectra_jaxsedfit_joint.py: each object is fitted "
        "with spec-<plate>-<mjd>-<fiber>.fits and that redshift instead of the "
        "DR16Q match. Absent: not passed. The file is checked with the fitter's "
        "loader, every selected spectrum must be in the cache, and the tasks "
        "read a read-only copy saved next to the submission script. With "
        "--resume the resumed run must have used the same file content."
    ),
)
parser.add_argument(
    "--cache-dir",
    metavar="DIR",
    default=None,
    help=(
        f"Spectrum cache directory passed to the fitter (default from the "
        f"settings block: {cache_dir}). With --resume (without --resume-only) "
        "the resumed run must have used the same directory."
    ),
)
parser.add_argument(
    "--sed-photometry-path",
    metavar="PATH",
    default=None,
    help=(
        "Long-form SED photometry table for fit_spectra_jaxsedfit_joint.py "
        f"(default from the settings block: {sed_photometry_path}). Before "
        "submission, objects without usable non-SDSS rows are reported (see "
        "--require-sed-photometry). With --resume (without --resume-only) the "
        "resumed run must have used the same file content."
    ),
)
parser.add_argument(
    "--require-sed-photometry",
    action="store_true",
    help=(
        "Refuse to submit when any submitted object has no usable non-SDSS row "
        "in the photometry table (--sed-photometry-path or the default); the "
        "fitter would fit such objects on the light-curve ugriz alone. Without "
        "it, --sed-photometry-path only warns and lists them."
    ),
)
parser.add_argument(
    "--cpus-per-task",
    type=positive_int,
    default=cpus_per_task,
    help=(
        f"SLURM CPUs per task (default from the settings block: {cpus_per_task}). "
        "The fitter gets one XLA CPU device per CPU, so NUTS chains run in "
        "parallel only up to this number; with fewer CPUs than chains they run "
        "one after another, about chains/CPUs times the wall time."
    ),
)
parser.add_argument(
    "--mem",
    type=slurm_memory,
    default=mem,
    help=(
        f"SLURM memory per task (default from the settings block: {mem}). "
        "Parallel chains hold one sampler state each and need more memory."
    ),
)
parser.add_argument(
    "--time-limit",
    type=slurm_time,
    default=time_limit,
    help=f"SLURM time limit (default from the settings block: {time_limit}).",
)
parser.add_argument(
    "--nuts-warmup",
    type=positive_int,
    default=500,
    help="NUTS warmup steps per chain passed to the fitter (default: 500).",
)
parser.add_argument(
    "--nuts-samples",
    type=positive_int,
    default=500,
    help="NUTS posterior draws per chain passed to the fitter (default: 500).",
)
parser.add_argument(
    "--nuts-chains",
    type=positive_int,
    default=1,
    help=(
        "Number of NUTS chains passed to the fitter (default: 1). Chains run in "
        "parallel only up to --cpus-per-task; set --cpus-per-task (and --mem) "
        "to the number of chains to run them in parallel."
    ),
)
parser.add_argument(
    "--resume",
    metavar="OLD_RUN_NAME",
    default="",
    help=(
        "For the joint JAXSEDFit backend, reuse usable posterior bundles from "
        "results/data/jaxqsofit/OLD_RUN_NAME/all and freshly fit the rest."
    ),
)
parser.add_argument(
    "--resume-only",
    action="store_true",
    help="Reconstruct saved posteriors only; record object failures without fresh fitting. Requires --resume.",
)
parser.add_argument(
    "--resume-compare-catalog",
    metavar="PATH",
    default=None,
    help=(
        "Spectral catalogue of the resumed run. Re-predicted values (chi-square, "
        "m_2500) are compared with it and an object fails when they differ, which "
        "catches a jaxsedfit version mismatch. Default with --resume-only: "
        "results/data/jaxqsofit/OLD_RUN_NAME.h5 when it exists."
    ),
)
parser.add_argument(
    "--allow-resume-without-submission-record",
    action="store_true",
    help=(
        "Resume even when the resumed run has no readable saved submission "
        "record (hpc_scripts/submit/jaxqsofit/submit_OLD_RUN_NAME.sbatch with "
        "provenance), so its spectrum selection and cache directory cannot be "
        "checked. Use only when that run used the DR16Q primary spectra and the "
        "default cache directory."
    ),
)
parser.add_argument(
    "--retry",
    metavar="FULL_JOB_NAME",
    default="",
    help=(
        "Resubmit only the latest unsuccessful terminal tasks for an existing "
        "spectrum-fit job while preserving its original task and chunk IDs."
    ),
)
cli_args = parser.parse_args()

# Resource overrides. --mem and --time-limit also apply to --retry; the CPU
# count of a retried task is fixed by its saved script (NUM_CORES).
mem = cli_args.mem
time_limit = cli_args.time_limit
cpus_per_task = cli_args.cpus_per_task
if cli_args.nuts_chains > cpus_per_task:
    print(
        f"Warning: --nuts-chains {cli_args.nuts_chains} > --cpus-per-task "
        f"{cpus_per_task}: chains will run one after another, about "
        f"{math.ceil(cli_args.nuts_chains / cpus_per_task)}x the sampling wall "
        f"time, within --time-limit {time_limit}.",
        file=sys.stderr,
    )

retry_job_name = cli_args.retry.strip()
fresh_run_options = (
    "--chisq-csv",
    "--description",
    "--fit-script",
    "--fit-bal",
    "--line-extrapolation-anchor",
    "--line-extrapolation-ew-scope",
    "--narrow-line-extrapolation",
    "--drop-sub-lyman-photometry",
    "--sub-lyman-rest-wavelength",
    "--sub-lyman-max-fraction",
    "--sdss-spectrum-selection-path",
    "--cache-dir",
    "--sed-photometry-path",
    "--require-sed-photometry",
    "--nuts-warmup",
    "--nuts-samples",
    "--nuts-chains",
    "--cpus-per-task",
    "--resume",
    "--resume-only",
    "--resume-compare-catalog",
    "--allow-resume-without-submission-record",
)
fresh_run_option_was_explicit = any(
    arg == option or arg.startswith(f"{option}=")
    for arg in sys.argv[1:]
    for option in fresh_run_options
)
if retry_job_name:
    if fresh_run_option_was_explicit:
        parser.error("--retry cannot be combined with fresh-run options")
    try:
        retry_job_name = normalize_job_name(retry_job_name, "--retry")
    except ValueError as exc:
        parser.error(str(exc))
    raise SystemExit(retry_unsuccessful_tasks(retry_job_name))

if cli_args.chisq_csv is None:
    parser.error("--chisq-csv is required for fresh runs")
if cli_args.description is None:
    parser.error("--description is required for fresh runs")

fit_script = cli_args.fit_script
resume_run_name = cli_args.resume.strip()
if cli_args.resume_only and not resume_run_name:
    parser.error("--resume-only requires --resume")
if cli_args.fit_bal and fit_script != "fit_spectra_jaxsedfit_joint.py":
    parser.error("--fit-bal is supported only with fit_spectra_jaxsedfit_joint.py")
joint_prediction_options = {
    "--line-extrapolation-anchor": cli_args.line_extrapolation_anchor,
    "--line-extrapolation-ew-scope": cli_args.line_extrapolation_ew_scope,
    "--narrow-line-extrapolation": cli_args.narrow_line_extrapolation,
}
joint_prediction_args = []
for option, value in joint_prediction_options.items():
    if value is None:
        continue
    if fit_script != "fit_spectra_jaxsedfit_joint.py":
        parser.error(f"{option} is supported only with fit_spectra_jaxsedfit_joint.py")
    if value is True:
        joint_prediction_args.append(option)
    elif value is False:
        joint_prediction_args.append(option.replace("--", "--no-", 1))
    else:
        joint_prediction_args.extend([option, value])
sub_lyman_options = {
    "--sub-lyman-rest-wavelength": cli_args.sub_lyman_rest_wavelength,
    "--sub-lyman-max-fraction": cli_args.sub_lyman_max_fraction,
}
if cli_args.sub_lyman_rest_wavelength is not None and not (
    math.isfinite(cli_args.sub_lyman_rest_wavelength) and cli_args.sub_lyman_rest_wavelength > 0.0
):
    parser.error("--sub-lyman-rest-wavelength must be positive and finite")
if cli_args.sub_lyman_max_fraction is not None and not (
    0.0 <= cli_args.sub_lyman_max_fraction < 1.0
):
    parser.error("--sub-lyman-max-fraction must lie in [0, 1)")
if cli_args.drop_sub_lyman_photometry:
    if fit_script != "fit_spectra_jaxsedfit_joint.py":
        parser.error("--drop-sub-lyman-photometry is supported only with fit_spectra_jaxsedfit_joint.py")
    joint_prediction_args.append("--drop-sub-lyman-photometry")
    for option, value in sub_lyman_options.items():
        if value is not None:
            joint_prediction_args.extend([option, repr(float(value))])
elif any(value is not None for value in sub_lyman_options.values()):
    parser.error(
        "--sub-lyman-rest-wavelength and --sub-lyman-max-fraction require "
        "--drop-sub-lyman-photometry"
    )


def import_fitter(option):
    """Import the fitter modules for a pre-submission check (they load JAX)."""
    saved_environ = dict(os.environ)
    # fit_spectra sizes its JAX host devices from NUM_CORES at import.
    os.environ["NUM_CORES"] = "1"
    try:
        from qvc.spectra import fit_spectra as legacy_fitter
        from qvc.spectra import fit_spectra_jaxsedfit_joint as joint_fitter
    except ImportError as exc:
        parser.error(
            f"{option} is checked with the fitter's loader, which this Python "
            f"cannot import ({exc}); run the launcher from the fitter's environment"
        )
    finally:
        # fit_spectra sets XLA/JAX variables at import; keep them (and the
        # NUM_CORES set above) out of the sbatch environment.
        os.environ.clear()
        os.environ.update(saved_environ)
    return legacy_fitter, joint_fitter


# The job script passes these through shell exports and a whitespace split.
UNSAFE_JOB_PATH_CHARS = re.compile(r'[\s"$`\\]')
sdss_spectrum_selection_path = None
sdss_spectrum_selection_sha256 = None
sdss_spectrum_selection_bytes = None
sdss_spectrum_selection_frame = None
sdss_spectrum_selection_frozen_path = None
if cli_args.sdss_spectrum_selection_path is not None:
    if fit_script != "fit_spectra_jaxsedfit_joint.py":
        parser.error("--sdss-spectrum-selection-path is supported only with fit_spectra_jaxsedfit_joint.py")
    selection_file = repo_relative_path(cli_args.sdss_spectrum_selection_path)
    if not selection_file.is_file() or not os.access(selection_file, os.R_OK):
        parser.error(f"--sdss-spectrum-selection-path not found or not readable: {selection_file}")
    if selection_file.suffix.lower() not in {".csv", ".txt", ".ecsv"}:
        parser.error("--sdss-spectrum-selection-path must be a .csv, .txt or .ecsv file")
    if UNSAFE_JOB_PATH_CHARS.search(str(selection_file)):
        parser.error(f"--sdss-spectrum-selection-path contains whitespace or shell characters: {selection_file}")
    # Check with the fitter's own loader so that a file every task would
    # reject is refused before submission.
    legacy_fitter, joint_fitter = import_fitter("--sdss-spectrum-selection-path")
    sdss_cache_file_path = legacy_fitter.sdss_cache_file_path
    sdss_spectrum_selection_bytes = selection_file.read_bytes()
    try:
        sdss_spectrum_selection_frame = joint_fitter.load_sdss_spectrum_selection_overrides(selection_file)
    except Exception as exc:
        parser.error(f"--sdss-spectrum-selection-path is rejected by the fitter: {exc}")
    sdss_spectrum_selection_path = str(selection_file)
    sdss_spectrum_selection_sha256 = hashlib.sha256(sdss_spectrum_selection_bytes).hexdigest()
if cli_args.cache_dir is not None:
    cache_dir_path = repo_relative_path(cli_args.cache_dir)
    if not cache_dir_path.is_dir():
        parser.error(f"--cache-dir not found or not a directory: {cache_dir_path}")
    if UNSAFE_JOB_PATH_CHARS.search(str(cache_dir_path)):
        parser.error(f"--cache-dir contains whitespace or shell characters: {cache_dir_path}")
    cache_dir = str(cache_dir_path)
sed_photometry_sha256 = None
if cli_args.sed_photometry_path is not None:
    if fit_script != "fit_spectra_jaxsedfit_joint.py":
        parser.error("--sed-photometry-path is supported only with fit_spectra_jaxsedfit_joint.py")
    photometry_file = repo_relative_path(cli_args.sed_photometry_path)
    if not photometry_file.is_file() or not os.access(photometry_file, os.R_OK):
        parser.error(f"--sed-photometry-path not found or not readable: {photometry_file}")
    if UNSAFE_JOB_PATH_CHARS.search(str(photometry_file)):
        parser.error(f"--sed-photometry-path contains whitespace or shell characters: {photometry_file}")
    sed_photometry_path = str(photometry_file)
    sed_photometry_sha256 = file_sha256(photometry_file)
if cli_args.require_sed_photometry and fit_script != "fit_spectra_jaxsedfit_joint.py":
    parser.error("--require-sed-photometry is supported only with fit_spectra_jaxsedfit_joint.py")
if cli_args.require_sed_photometry and cli_args.resume_only:
    parser.error(
        "--require-sed-photometry has no effect with --resume-only, which uses the "
        "photometry stored in the bundles"
    )
if resume_run_name and fit_script != "fit_spectra_jaxsedfit_joint.py":
    parser.error("--resume is supported only with fit_spectra_jaxsedfit_joint.py")
if resume_run_name:
    normalized_resume_name = re.sub(r"[^A-Za-z0-9._-]+", "_", resume_run_name)
    if normalized_resume_name != resume_run_name or resume_run_name in {".", ".."}:
        parser.error("--resume must be a run name, not a path")
description = re.sub(
    r"[^A-Za-z0-9.-]+", "_", cli_args.description.strip()
).strip("_.-")
if not description:
    parser.error("--description cannot be blank")
date_hour = datetime.now().strftime("%b%d_%I%M%p").lower()
git_commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    cwd=REPO_ROOT,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
job_name_parts = [date_hour, "spectrafit", git_commit]
if description:
    job_name_parts.append(description)
job_name = "_".join(job_name_parts)
prefix = job_name
if sdss_spectrum_selection_path:
    # Tasks and --retry read a frozen read-only copy saved next to the
    # submission script, so editing the original later cannot change spectra.
    sdss_spectrum_selection_frozen_path = str(
        REPO_ROOT / "hpc_scripts" / "submit" / "jaxqsofit"
        / f"{prefix}_sdss_spectrum_selection{Path(sdss_spectrum_selection_path).suffix}"
    )
    if UNSAFE_JOB_PATH_CHARS.search(sdss_spectrum_selection_frozen_path):
        parser.error(
            "the frozen SDSS spectrum selection path contains whitespace or shell "
            f"characters: {sdss_spectrum_selection_frozen_path}"
        )
    joint_prediction_args.extend(["--sdss-spectrum-selection-path", sdss_spectrum_selection_frozen_path])

# Use the generated run identifier for every run-specific folder and file.
output_dir = f"results/data/jaxqsofit/{prefix}"
fig_dir = f"plots/jaxqsofit/{prefix}"
resume_dir = (
    f"results/data/jaxqsofit/{resume_run_name}/all"
    if resume_run_name
    else ""
)

if cli_args.resume_compare_catalog and not resume_run_name:
    parser.error("--resume-compare-catalog requires --resume")
resume_compare_catalog = cli_args.resume_compare_catalog
if resume_compare_catalog is None and cli_args.resume_only:
    candidate = f"results/data/jaxqsofit/{resume_run_name}.h5"
    if (REPO_ROOT / candidate).is_file():
        resume_compare_catalog = candidate
    else:
        print(
            f"Warning: no {candidate}; re-predicted values will not be compared "
            "with the resumed run (pass --resume-compare-catalog).",
            file=sys.stderr,
        )
if resume_compare_catalog:
    if not (REPO_ROOT / resume_compare_catalog).is_file() and not Path(resume_compare_catalog).is_file():
        parser.error(f"--resume-compare-catalog not found: {resume_compare_catalog}")
    joint_prediction_args.extend(["--resume-compare-catalog", resume_compare_catalog])

if resume_dir:
    resume_path = REPO_ROOT / resume_dir
    output_path = REPO_ROOT / output_dir
    if not resume_path.is_dir():
        raise FileNotFoundError(f"Resume run sample directory not found: {resume_path}")
    if resume_path.resolve() == (output_path / "all").resolve():
        raise ValueError("Resume source and new sample destination must differ.")

    # A resume must not mix spectra: resumed rows keep the spectrum their bundle
    # was sampled on, while the fitter labels every row with the plate/MJD/fiber
    # of the current selection and fresh fallback fits read the current cache.
    resumed_script = REPO_ROOT / "hpc_scripts" / "submit" / "jaxqsofit" / f"submit_{resume_run_name}.sbatch"
    resumed_inputs = (
        saved_submission_inputs(resumed_script.read_text(encoding="utf-8"))
        if resumed_script.is_file()
        else None
    )
    spectrum_options_given = (
        cli_args.sdss_spectrum_selection_path is not None
        or cli_args.cache_dir is not None
        or (cli_args.sed_photometry_path is not None and not cli_args.resume_only)
    )
    if not isinstance(resumed_inputs, dict):
        if spectrum_options_given:
            parser.error(
                "--sdss-spectrum-selection-path/--cache-dir/--sed-photometry-path with --resume need the "
                f"resumed run's submission record ({resumed_script}) to confirm it "
                "used the same spectra; none was readable"
            )
        if not cli_args.allow_resume_without_submission_record:
            parser.error(
                f"--resume {resume_run_name}: no readable submission record "
                f"{resumed_script} (missing, or written before runs recorded "
                "provenance), so it cannot be checked that the resumed run used the "
                "same spectra as this one (DR16Q primary spectra, cache "
                f"{cache_dir}); a resume across different spectra would mix them in "
                "one catalogue. Pass --allow-resume-without-submission-record if "
                "you know it did."
            )
        print(
            f"Warning: no readable submission record {resumed_script}; cannot check "
            "that the resumed run used the same SDSS spectrum selection and cache "
            "directory.",
            file=sys.stderr,
        )
    else:
        problem = frozen_selection_problem(resumed_inputs)
        if problem:
            parser.error(
                f"--resume {resume_run_name}: {problem}; its bundles can no longer "
                "be matched to their spectra. Restore the copy or start a fresh run."
            )
        resumed_selection = resumed_inputs.get("sdss_spectrum_selection_path")
        resumed_selection_sha256 = resumed_inputs.get("sdss_spectrum_selection_sha256")
        if (resumed_selection_sha256 or None) != sdss_spectrum_selection_sha256:
            parser.error(
                f"--resume {resume_run_name} used SDSS spectrum selection "
                f"{resumed_selection or 'none (DR16Q primary spectra)'}"
                + (f" (sha256 {resumed_selection_sha256})" if resumed_selection_sha256 else "")
                + f", this run {sdss_spectrum_selection_path or 'none (DR16Q primary spectra)'}"
                + (f" (sha256 {sdss_spectrum_selection_sha256})" if sdss_spectrum_selection_sha256 else "")
                + "; one run must not mix spectra. Pass the same "
                "--sdss-spectrum-selection-path (same content) as the resumed run, "
                "or start a fresh run."
            )
        resumed_cache_dir = resumed_inputs.get("cache_dir")
        if not cli_args.resume_only and (
            not resumed_cache_dir
            or repo_relative_path(resumed_cache_dir) != repo_relative_path(cache_dir)
        ):
            parser.error(
                f"--resume {resume_run_name} used cache directory {resumed_cache_dir}, "
                f"this run {cache_dir}; fresh fallback fits would read spectra from "
                "another cache. Pass --cache-dir with the resumed run's directory, "
                "or --resume-only."
            )
        # Only fresh fallback fits read the photometry file; --resume-only
        # re-predicts from the photometry stored in each bundle.
        resumed_photometry_sha256 = resumed_inputs.get("sed_photometry_sha256")
        if not cli_args.resume_only and (
            cli_args.sed_photometry_path is not None or resumed_photometry_sha256
        ):
            resumed_photometry = resumed_inputs.get("sed_photometry_path")
            if not resumed_photometry_sha256 and resumed_photometry:
                # The record predates the hash: only the file as it is now can
                # be compared, not the content the resumed run used.
                resumed_file = repo_relative_path(resumed_photometry)
                resumed_photometry_sha256 = file_sha256(resumed_file) if resumed_file.is_file() else None
                print(
                    f"Warning: the submission record of --resume {resume_run_name} has no "
                    f"SED photometry hash, so the photometry content that run used cannot "
                    f"be verified; {resumed_photometry} is compared only as it is now.",
                    file=sys.stderr,
                )
            current_photometry_sha256 = sed_photometry_sha256 or file_sha256(
                repo_relative_path(sed_photometry_path)
            )
            if resumed_photometry_sha256 != current_photometry_sha256:
                parser.error(
                    f"--resume {resume_run_name} used SED photometry {resumed_photometry} "
                    f"(sha256 {resumed_photometry_sha256}), this run {sed_photometry_path} "
                    f"(sha256 {current_photometry_sha256}); fresh fallback fits would use "
                    "other photometry than the resumed bundles. Pass --sed-photometry-path "
                    "with the resumed run's file (same content), or --resume-only."
                )

# ==========================================
# 3. Helpers
# ==========================================
def normalize_object_id(value):
    text = str(value).strip()
    if not text:
        return ""
    return text[:-2] if text.endswith(".0") else text


def load_csv_object_ids(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Object-list CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype={"object_id": str}, low_memory=False)
    if "object_id" not in df.columns:
        raise ValueError(
            f"Object-list CSV {csv_path} is missing required column 'object_id'"
        )
    object_ids = (
        df["object_id"]
        .dropna()
        .map(normalize_object_id)
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    return object_ids


def submit_in_batches(script_filename, num_tasks, batch_limit=10_000):
    """
    Submit SLURM array jobs in chunks of at most batch_limit tasks.
    """
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")

    if num_tasks == 1:
        cmd = ["sbatch", script_filename]
        print("Submitting single non-array job:")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        return

    start = 0
    while start < num_tasks:
        end = min(start + batch_limit - 1, num_tasks - 1)
        cmd = ["sbatch", f"--array={start}-{end}", script_filename]
        print(f"Submitting array: {start}-{end}")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        start = end + 1


# ==========================================
# 4. Read object IDs and compute array size
# ==========================================
fit_modules = {
    "fit_spectra.py": "qvc.spectra.fit_spectra",
    "fit_spectra_jaxsedfit_joint.py": "qvc.spectra.fit_spectra_jaxsedfit_joint",
}
if fit_script not in fit_modules:
    supported = ", ".join(sorted(fit_modules))
    raise ValueError(f"Unsupported fit_script {fit_script!r}. Choose one of: {supported}")
fit_module = fit_modules[fit_script]

if fit_script == "fit_spectra_jaxsedfit_joint.py":
    if not sed_photometry_path:
        raise ValueError(
            "sed_photometry_path is required when using "
            "fit_spectra_jaxsedfit_joint.py"
        )
    sed_photometry_file = REPO_ROOT / sed_photometry_path
    if not sed_photometry_file.exists():
        raise FileNotFoundError(
            f"SED photometry input not found: {sed_photometry_file}"
        )

chisq_path = Path(cli_args.chisq_csv).expanduser()
if not chisq_path.is_absolute():
    chisq_path = REPO_ROOT / chisq_path
chisq_path = chisq_path.resolve()
chisq_csv = str(chisq_path)
exclude_path = REPO_ROOT / exclude_csv if exclude_csv else None

chisq_object_ids = load_csv_object_ids(chisq_path)
exclude_ids = []
if exclude_path is not None and exclude_path.exists():
    exclude_ids = load_csv_object_ids(exclude_path)
elif exclude_path is not None and not exclude_path.exists():
    print(f"[WARNING] EXCLUDE_CSV not found, ignoring exclusions: {exclude_path}")

exclude_set = set(exclude_ids)
requested_object_ids = [obj for obj in chisq_object_ids if obj not in exclude_set]
submit_object_ids = requested_object_ids

print(f"Length of chisq_object_ids: {len(chisq_object_ids)}")
print(f"Length of exclude_object_ids: {len(exclude_ids)}")
print(f"Length of requested_object_ids (chisq - exclude): {len(requested_object_ids)}")
print(f"Length of submit_object_ids: {len(submit_object_ids)}")

if len(submit_object_ids) == 0:
    raise ValueError(
        f"No valid object_id values remain after normalization and exclusions: {chisq_path}"
    )

# The fitter requires a selection row for every object it fits, and a fresh
# fit needs the selected spectrum in the cache; check both before submission.
if sdss_spectrum_selection_frame is not None:
    selections = sdss_spectrum_selection_frame.set_index("object_id")
    unselected_ids = []
    missing_spectra = []
    for obj in submit_object_ids:
        if obj not in selections.index:
            unselected_ids.append(obj)
            continue
        spectrum_file = sdss_cache_file_path(
            cache_dir=repo_relative_path(cache_dir),
            plate=int(selections.at[obj, "plate"]),
            mjd=int(selections.at[obj, "mjd"]),
            fiber=int(selections.at[obj, "fiber"]),
        )
        if not spectrum_file.is_file():
            missing_spectra.append(spectrum_file.name)
    if unselected_ids:
        parser.error(
            f"--sdss-spectrum-selection-path {sdss_spectrum_selection_path} lacks "
            f"{len(unselected_ids)} submitted object ID(s), e.g. {unselected_ids[:10]}"
        )
    if missing_spectra and not cli_args.resume_only:
        parser.error(
            f"{len(missing_spectra)} selected spectrum file(s) are not in the cache "
            f"directory {repo_relative_path(cache_dir)}: {missing_spectra}"
        )

num_tasks = math.ceil(len(submit_object_ids) / chunk_size)
max_array_id = num_tasks - 1

print(f"Loaded {len(submit_object_ids)} schedulable object_ids from {chisq_csv}")
print(f"Chunk size: {chunk_size}")
print(f"Number of array tasks: {num_tasks}")

# A joint fit of an object without rows in the photometry table does not fail:
# it silently runs on the light-curve ugriz only (the table's *_sdss rows are
# replaced by those). Report such objects; refuse with --require-sed-photometry.
unphotometered_ids = None
unphotometered_ids_file = None
if (
    cli_args.sed_photometry_path is not None or cli_args.require_sed_photometry
) and not cli_args.resume_only:
    photometry_option = (
        "--require-sed-photometry"
        if cli_args.sed_photometry_path is None
        else "--sed-photometry-path"
    )
    legacy_fitter, joint_fitter = import_fitter(photometry_option)
    try:
        photometry = joint_fitter.load_saved_sed_photometry(repo_relative_path(sed_photometry_path))
    except Exception as exc:
        parser.error(f"SED photometry {sed_photometry_path} is rejected by the fitter: {exc}")
    sdss_filters = {f"{band}_sdss" for band in legacy_fitter.SDSS_BANDS}
    photometry_ids = set(
        photometry.loc[~photometry["filter_name"].isin(sdss_filters), "source_id"]
    )
    unphotometered_ids = []
    for obj in submit_object_ids:
        if obj not in photometry_ids:
            unphotometered_ids.append(obj)
    if unphotometered_ids and cli_args.require_sed_photometry:
        parser.error(
            f"SED photometry {sed_photometry_path} has no usable non-SDSS "
            f"photometry for {len(unphotometered_ids)} submitted object ID(s), "
            f"e.g. {unphotometered_ids[:50]} (--require-sed-photometry)"
        )
    if unphotometered_ids:
        unphotometered_ids_file = (
            f"hpc_scripts/submit/jaxqsofit/{prefix}_objects_without_sed_photometry.txt"
        )
        print(
            f"WARNING: {len(unphotometered_ids)} of {len(submit_object_ids)} submitted "
            f"object(s) have no usable non-SDSS photometry in {sed_photometry_path}; "
            "the fitter fits them on the light-curve ugriz alone. List: "
            f"{unphotometered_ids_file} (e.g. {unphotometered_ids[:20]}). Pass "
            "--require-sed-photometry to refuse instead.",
            file=sys.stderr,
        )

# ==========================================
# 5. Directory setup
# ==========================================
log_dir = f"hpc_scripts/logs/jaxqsofit/{prefix}"
submit_dir = "hpc_scripts/submit/jaxqsofit"

existing_run_paths = [
    REPO_ROOT / output_dir,
    REPO_ROOT / fig_dir,
    REPO_ROOT / log_dir,
]
if sdss_spectrum_selection_frozen_path:
    existing_run_paths.append(Path(sdss_spectrum_selection_frozen_path))
if unphotometered_ids_file:
    existing_run_paths.append(REPO_ROOT / unphotometered_ids_file)
already_exist = [str(path) for path in existing_run_paths if path.exists()]
if already_exist:
    raise FileExistsError(
        "Refusing to reuse an existing new-run destination: "
        + ", ".join(already_exist)
    )

os.makedirs(log_dir, exist_ok=True)
print(f"Created log directory: {log_dir}")

os.makedirs(submit_dir, exist_ok=True)
print(f"Created submit directory: {submit_dir}")

os.makedirs(output_dir, exist_ok=True)
print(f"Created output directory: {output_dir}")

object_ids_file = f"{submit_dir}/{prefix}_object_ids.txt"
with open(object_ids_file, "w", encoding="utf-8") as f:
    for obj_id in submit_object_ids:
        f.write(f"{obj_id}\n")
print(f"Wrote schedulable object_ids file: {object_ids_file}")
if sdss_spectrum_selection_frozen_path:
    frozen_selection = Path(sdss_spectrum_selection_frozen_path)
    frozen_selection.write_bytes(sdss_spectrum_selection_bytes)
    frozen_selection.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(f"Froze SDSS spectrum selection: {frozen_selection}")
if unphotometered_ids_file:
    with open(unphotometered_ids_file, "w", encoding="utf-8") as f:
        for obj_id in unphotometered_ids:
            f.write(f"{obj_id}\n")
    print(f"Wrote objects without SED photometry: {unphotometered_ids_file}")

# ==========================================
# 6. Generate the SLURM bash script
# ==========================================
array_directive = f"#SBATCH --array=0-{max_array_id}" if num_tasks > 1 else ""

script_filename = f"{submit_dir}/submit_{prefix}.sbatch"

submission = submission_record(
    "hpc_scripts/sfitspectra.xsh",
    sys.argv,
    {
        "wrapper_args": vars(cli_args),
        "job_name": job_name,
        "prefix": prefix,
        "description": description,
        "fit_script": fit_script,
        "fit_module": fit_module,
        "inputs": {
            "chisq_csv": chisq_csv,
            "exclude_csv": exclude_csv,
            "sed_photometry_path": sed_photometry_path,
            "sed_photometry_sha256": sed_photometry_sha256,
            "dr16q_fits": "data/dr16q_prop_May01_2024.fits",
            "cache_dir": cache_dir,
            "sdss_spectrum_selection_path": sdss_spectrum_selection_path,
            "sdss_spectrum_selection_sha256": sdss_spectrum_selection_sha256,
            "sdss_spectrum_selection_frozen_path": sdss_spectrum_selection_frozen_path,
        },
        "sed_photometry_check": (
            None
            if unphotometered_ids is None
            else {
                "required": cli_args.require_sed_photometry,
                "objects_without_photometry": len(unphotometered_ids),
                "objects_without_photometry_file": unphotometered_ids_file,
            }
        ),
        "outputs": {"output_dir": output_dir, "fig_dir": fig_dir},
        "resume": {"directory": resume_dir, "run_name": resume_run_name, "only": cli_args.resume_only},
        "resources": {
            "partition": partition,
            "time": time_limit,
            "memory": mem,
            "cpus_per_task": cpus_per_task,
            "nproc": nproc,
            "chunk_size": chunk_size,
            "python_bin": python_bin,
        },
        "nuts": {
            "warmup": cli_args.nuts_warmup,
            "samples": cli_args.nuts_samples,
            "chains": cli_args.nuts_chains,
        },
        "object_count": len(submit_object_ids),
    },
)
encoded_submission = encode_record(submission)

script_content = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/fit_%A_%a.out
#SBATCH --error={log_dir}/fit_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --partition={partition}
{array_directive}

set -euo pipefail

export PREFIX="{prefix}"
export FIT_SCRIPT="{fit_script}"
export FIT_MODULE="{fit_module}"
export FIT_BAL="{int(cli_args.fit_bal)}"
export JOINT_PREDICTION_ARGS="{' '.join(joint_prediction_args)}"
export RESUME_ONLY="{int(cli_args.resume_only)}"
export SED_PHOTOMETRY_PATH="{sed_photometry_path}"
export OUTPUT_DIR="{output_dir}"
export OBJECT_IDS_FILE="{object_ids_file}"
export CHUNK_SIZE={chunk_size}
export NUTS_WARMUP="{cli_args.nuts_warmup}"
export NUTS_SAMPLES="{cli_args.nuts_samples}"
export NUTS_CHAINS="{cli_args.nuts_chains}"
export NPROC={nproc}
export CPUS_PER_TASK={cpus_per_task}
export PYTHON_BIN="{python_bin}"
export QVC_DATA_DIR="{qvc_data_dir}"
export CACHE_DIR="{cache_dir}"
export FIG_DIR="{fig_dir}"
export RESUME_DIR="{resume_dir}"
export RESUME_RUN_NAME="{resume_run_name}"
export {PROVENANCE_ENV}="{encoded_submission}"

export TASK_ID="${{SLURM_ARRAY_TASK_ID:-0}}"

export JAX_ENABLE_X64="True"
export QT_QPA_PLATFORM="offscreen"
export NUM_CORES="${{CPUS_PER_TASK}}"
export QVC_DATA_DIR="${{QVC_DATA_DIR}}"

mkdir -p "${{OUTPUT_DIR}}"
mkdir -p "${{OUTPUT_DIR}}/all"
mkdir -p "${{FIG_DIR}}"

echo "Job started on $(date)"
echo "Host: $(hostname)"
echo "TASK_ID=$TASK_ID"
echo "JOB_NAME={job_name}"
echo "PREFIX=$PREFIX"
echo "FIT_SCRIPT=$FIT_SCRIPT"
echo "FIT_MODULE=$FIT_MODULE"
echo "FIT_BAL=$FIT_BAL"
echo "JOINT_PREDICTION_ARGS=$JOINT_PREDICTION_ARGS"
echo "NUTS_WARMUP=$NUTS_WARMUP NUTS_SAMPLES=$NUTS_SAMPLES NUTS_CHAINS=$NUTS_CHAINS"
echo "SED_PHOTOMETRY_PATH=$SED_PHOTOMETRY_PATH"
echo "OBJECT_IDS_FILE=$OBJECT_IDS_FILE"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "FIG_DIR=$FIG_DIR"
echo "RESUME_DIR=$RESUME_DIR"
echo "RESUME_RUN_NAME=$RESUME_RUN_NAME"
echo "PYTHON_BIN=$PYTHON_BIN"

"$PYTHON_BIN" - <<'PY'
import os
import subprocess
from itertools import islice

prefix = os.environ["PREFIX"]
fit_script = os.environ["FIT_SCRIPT"]
fit_module = os.environ["FIT_MODULE"]
fit_bal = os.environ["FIT_BAL"] == "1"
joint_prediction_args = os.environ["JOINT_PREDICTION_ARGS"].split()
resume_only = os.environ["RESUME_ONLY"] == "1"
sed_photometry_path = os.environ["SED_PHOTOMETRY_PATH"]
output_dir = os.environ["OUTPUT_DIR"]
object_ids_file = os.environ["OBJECT_IDS_FILE"]
chunk_size = int(os.environ["CHUNK_SIZE"])
nuts_warmup = int(os.environ["NUTS_WARMUP"])
nuts_samples = int(os.environ["NUTS_SAMPLES"])
nuts_chains = int(os.environ["NUTS_CHAINS"])
nproc = int(os.environ["NPROC"])
cpus_per_task = int(os.environ["CPUS_PER_TASK"])
python_bin = os.environ["PYTHON_BIN"]
cache_dir = os.environ["CACHE_DIR"]
fig_dir = os.environ["FIG_DIR"]
resume_dir = os.environ["RESUME_DIR"]
resume_run_name = os.environ["RESUME_RUN_NAME"]
task_id = int(os.environ.get("TASK_ID", "0"))

start = task_id * chunk_size
stop = start + chunk_size

with open(object_ids_file, encoding="utf-8") as f:
    ids_this_task = [line.strip() for line in islice(f, start, stop) if line.strip()]
if len(ids_this_task) == 0:
    print(f"No schedulable object_ids for task_id={{task_id}} (slice {{start}}:{{stop}}). Exiting 0.")
    raise SystemExit(0)

chunk_tag = f"chunk{{task_id:04d}}"
out_suffix = ".h5" if fit_script == "fit_spectra_jaxsedfit_joint.py" else ".csv"
out_catalog = f"{{output_dir}}/{{prefix}}_{{chunk_tag}}{{out_suffix}}"
real_output_dir = f"{{output_dir}}/all"

print(f"Running task {{task_id}}")
print(f"Processing rows {{start}}:{{stop}}")
print(f"Number of object_ids in this task: {{len(ids_this_task)}}")
print(f"Output catalog: {{out_catalog}}")
print(f"object_ids: {{ids_this_task}}")

cmd = [
    python_bin,
    "-m", fit_module,
    "--mode", "fit",
    out_catalog,
    "--cache-dir", cache_dir,
    "--verbose",
    "--optax-steps", "5000",
    "--optax-lr", "0.005",
    "--nuts-warmup", str(nuts_warmup),
    "--nuts-samples", str(nuts_samples),
    "--nuts-chains", str(nuts_chains),
    "--output-dir", real_output_dir,
    "--fig-dir", fig_dir,
    "--filter_object_id", *ids_this_task,
    "--nproc", str(nproc),
]

if fit_script == "fit_spectra.py":
    cmd.extend([
        "--plot_mcmc_diagnostics",
        "--save-fig",
    ])
elif fit_script == "fit_spectra_jaxsedfit_joint.py":
    cmd.extend([
        "--sed-photometry-path", sed_photometry_path,
        "--progress",
        #"--no-catalog-progress",
    ])
    if fit_bal:
        cmd.append("--fit-bal")
    cmd.extend(joint_prediction_args)
    if resume_dir:
        cmd.extend([
            "--resume", resume_dir,
            "--resume-run-name", resume_run_name,
        ])
        if resume_only:
            cmd.append("--resume-only")
else:
    raise ValueError(f"Unsupported FIT_SCRIPT in generated job: {{fit_script!r}}")

print("\\nCommand:")
print(" ".join(cmd))
print()

subprocess.run(cmd, check=True)
PY

echo "Job finished on $(date)"
"""

with open(script_filename, "w") as f:
    f.write(script_content)

# Make it executable
st = os.stat(script_filename)
os.chmod(script_filename, st.st_mode | stat.S_IEXEC)

print(f"Generated SLURM script ({num_tasks} task(s)): {script_filename}")

# ==========================================
# 7. Submit
# ==========================================
submit_in_batches(script_filename, num_tasks)
