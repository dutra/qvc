#!/usr/bin/env xonsh

import argparse
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
    encode_record,
    submission_record,
)

# ==========================================
# 1. Define your job settings here
# ==========================================
partition = "day"
time_limit = "8:00:00"

# Number of object_ids per array task
chunk_size = 4

# Number of CPUs used by fit_spectra.py
nproc = 1
cpus_per_task = 1
mem = "40G"

#fit_script = "fit_spectra.py"
fit_script = "fit_spectra_jaxsedfit_joint.py"

# Required only by fit_spectra_jaxsedfit_joint.py
sed_photometry_path = "data/jul14_master_input_file_chisqgt20_bandwagon_photometry.csv"

# Optional exclusion file
exclude_csv = None #"results/data/jaxqsofit/jaxqsofit_apr20c_chisq20_apr18h.csv"

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
    "--resume",
    metavar="OLD_RUN_NAME",
    default="",
    help=(
        "For the joint JAXSEDFit backend, reuse usable posterior bundles from "
        "results/data/jaxqsofit/OLD_RUN_NAME/all and freshly fit the rest."
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

retry_job_name = cli_args.retry.strip()
fresh_run_options = ("--chisq-csv", "--description", "--fit-script", "--resume")
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

# Use the generated run identifier for every run-specific folder and file.
output_dir = f"results/data/jaxqsofit/{prefix}"
fig_dir = f"plots/jaxqsofit/{prefix}"
resume_dir = (
    f"results/data/jaxqsofit/{resume_run_name}/all"
    if resume_run_name
    else ""
)

if resume_dir:
    resume_path = REPO_ROOT / resume_dir
    output_path = REPO_ROOT / output_dir
    if not resume_path.is_dir():
        raise FileNotFoundError(f"Resume run sample directory not found: {resume_path}")
    if resume_path.resolve() == (output_path / "all").resolve():
        raise ValueError("Resume source and new sample destination must differ.")

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

num_tasks = math.ceil(len(submit_object_ids) / chunk_size)
max_array_id = num_tasks - 1

print(f"Loaded {len(submit_object_ids)} schedulable object_ids from {chisq_csv}")
print(f"Chunk size: {chunk_size}")
print(f"Number of array tasks: {num_tasks}")

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
            "dr16q_fits": "data/dr16q_prop_May01_2024.fits",
            "cache_dir": cache_dir,
        },
        "outputs": {"output_dir": output_dir, "fig_dir": fig_dir},
        "resume": {"directory": resume_dir, "run_name": resume_run_name},
        "resources": {
            "partition": partition,
            "time": time_limit,
            "memory": mem,
            "cpus_per_task": cpus_per_task,
            "nproc": nproc,
            "chunk_size": chunk_size,
            "python_bin": python_bin,
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
export SED_PHOTOMETRY_PATH="{sed_photometry_path}"
export OUTPUT_DIR="{output_dir}"
export OBJECT_IDS_FILE="{object_ids_file}"
export CHUNK_SIZE={chunk_size}
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
sed_photometry_path = os.environ["SED_PHOTOMETRY_PATH"]
output_dir = os.environ["OUTPUT_DIR"]
object_ids_file = os.environ["OBJECT_IDS_FILE"]
chunk_size = int(os.environ["CHUNK_SIZE"])
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
    "--optax-steps", "4000",
    "--optax-lr", "0.001",
    "--nuts-warmup", "500",
    "--nuts-samples", "250",
    "--nuts-chains", "1",
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
    ])
    if resume_dir:
        cmd.extend([
            "--resume", resume_dir,
            "--resume-run-name", resume_run_name,
        ])
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
