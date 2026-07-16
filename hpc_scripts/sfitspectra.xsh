#!/usr/bin/env xonsh

import math
import os
import stat
import sys
import subprocess
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# ==========================================
# 1. Define your job settings here
# ==========================================
#prefix = "jaxqsofit_apr1a_chisq20_mar31a"
prefix = "jaxqsofit_jul6b"
partition = "day"
time_limit = "10:00:00"

# Number of object_ids per array task
chunk_size = 12

# Number of CPUs used by fit_spectra.py
nproc = 3
cpus_per_task = 3
mem = "32G"

fit_script = "fit_spectra.py"

chisq_csv = "results/data/variability_chi_sq_red_g_gt_20.csv"

# Optional exclusion file
exclude_csv = None #"results/data/jaxqsofit/jaxqsofit_apr20c_chisq20_apr18h.csv"

# Each array task writes one CSV
output_dir = f"results/data/jaxqsofit/{prefix}"

# Direct path to Python inside the Conda env
python_bin = "/home/id255/.conda/envs/jaxcpu2/bin/python"

# Optional: cache/data locations
qvc_data_dir = "/home/id255/scratch_pi_pn38/id255/qvc/data"
cache_dir = "data/spectra_cache_all"
fig_dir = f"plots/jaxqsofit/{prefix}"

# ==========================================
# 2. Helpers
# ==========================================
def normalize_object_id(value):
    text = str(value).strip()
    if not text:
        return ""
    return text[:-2] if text.endswith(".0") else text


def load_csv_object_ids(csv_path):
    df = pd.read_csv(csv_path, dtype={"object_id": str}, low_memory=False)
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
# 3. Read object IDs and compute array size
# ==========================================
chisq_path = REPO_ROOT / chisq_csv
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
    raise ValueError(f"No valid object_id values found in {chisq_csv}")

num_tasks = math.ceil(len(submit_object_ids) / chunk_size)
max_array_id = num_tasks - 1

print(f"Loaded {len(submit_object_ids)} schedulable object_ids from {chisq_csv}")
print(f"Chunk size: {chunk_size}")
print(f"Number of array tasks: {num_tasks}")

# ==========================================
# 4. Directory setup
# ==========================================
log_dir = f"hpc_scripts/logs/jaxqsofit/{prefix}"
submit_dir = "hpc_scripts/submit/jaxqsofit"

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
# 5. Generate the SLURM bash script
# ==========================================
array_directive = f"#SBATCH --array=0-{max_array_id}" if num_tasks > 1 else ""

script_filename = f"{submit_dir}/submit_{prefix}.sbatch"

script_content = f"""#!/usr/bin/env bash
#SBATCH --job-name=jaxqsofit_{prefix}
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
export OUTPUT_DIR="{output_dir}"
export OBJECT_IDS_FILE="{object_ids_file}"
export CHUNK_SIZE={chunk_size}
export NPROC={nproc}
export CPUS_PER_TASK={cpus_per_task}
export PYTHON_BIN="{python_bin}"
export QVC_DATA_DIR="{qvc_data_dir}"
export CACHE_DIR="{cache_dir}"
export FIG_DIR="{fig_dir}"

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
echo "PREFIX=$PREFIX"
echo "OBJECT_IDS_FILE=$OBJECT_IDS_FILE"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "FIG_DIR=$FIG_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"

"$PYTHON_BIN" - <<'PY'
import os
import subprocess
from itertools import islice

prefix = os.environ["PREFIX"]
fit_script = os.environ["FIT_SCRIPT"]
output_dir = os.environ["OUTPUT_DIR"]
object_ids_file = os.environ["OBJECT_IDS_FILE"]
chunk_size = int(os.environ["CHUNK_SIZE"])
nproc = int(os.environ["NPROC"])
cpus_per_task = int(os.environ["CPUS_PER_TASK"])
python_bin = os.environ["PYTHON_BIN"]
cache_dir = os.environ["CACHE_DIR"]
fig_dir = os.environ["FIG_DIR"]
task_id = int(os.environ.get("TASK_ID", "0"))

start = task_id * chunk_size
stop = start + chunk_size

with open(object_ids_file, encoding="utf-8") as f:
    ids_this_task = [line.strip() for line in islice(f, start, stop) if line.strip()]
if len(ids_this_task) == 0:
    print(f"No schedulable object_ids for task_id={{task_id}} (slice {{start}}:{{stop}}). Exiting 0.")
    raise SystemExit(0)

chunk_tag = f"chunk{{task_id:04d}}"
out_csv = f"{{output_dir}}/{{prefix}}_{{chunk_tag}}.csv"
real_output_dir = f"{{output_dir}}/all"

print(f"Running task {{task_id}}")
print(f"Processing rows {{start}}:{{stop}}")
print(f"Number of object_ids in this task: {{len(ids_this_task)}}")
print(f"Output CSV: {{out_csv}}")
print(f"object_ids: {{ids_this_task}}")

cmd = [
    python_bin,
    "-m", "qvc.spectra.fit_spectra",
    "--mode", "fit",
    out_csv,
    "--plot_mcmc_diagnostics",
    "--cache-dir", cache_dir,
    "--verbose",
    "--save-fig",
    "--nuts-warmup", "250",
    "--nuts-samples", "250",
    "--nuts-chains", "1",
    "--output-dir", real_output_dir,
    "--fig-dir", fig_dir,
    "--filter_object_id", *ids_this_task,
    "--nproc", str(nproc),
]

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
# 6. Submit
# ==========================================
submit_in_batches(script_filename, num_tasks)
