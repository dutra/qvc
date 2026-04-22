#!/usr/bin/env python3
import argparse
import math
import os
import sys
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qvc.light_curve.multiband_generate_lc import resolve_macleod_object_ids, resolve_stone_object_ids

SCRIPT_DIR = REPO_ROOT / "hpc_scripts" / "jobs" / "multibandfit"
LOG_ROOT = REPO_ROOT / "hpc_scripts" / "logs" / "multibandfit"
DEFAULT_SPECTRA_FIT_CSV = "results/data/jaxqsofit_apr5d_chisq20_mar31a_good.csv"
MAX_ARRAY_SIZE = 10_000


@dataclass(frozen=True)
class JobConfig:
    description: str
    object_ids: list[str]
    extra_flags: tuple[str, ...] = ()
    use_psf_constant_flux: bool = False


def parse_args():
    parser = argparse.ArgumentParser(description="Submit multiband-fit SLURM jobs.")
    parser.add_argument("--fit", choices=("chisq", "stone", "macleod"), required=True, help="Sample to submit.")
    parser.add_argument("--chisq-csv", type=str, default=None, help="CSV file with object_id column for --fit chisq.")
    parser.add_argument("--num-jobs", type=int, default=-1, help="-1 means submit all chunks after skip.")
    parser.add_argument("--skip", type=int, default=0, help="Number of chunks to skip.")
    parser.add_argument("--N", type=int, default=1, help="Objects per array task.")
    parser.add_argument("--nwarm", type=int, default=500, help="Warmup steps.")
    parser.add_argument("--nsamp", type=int, default=200, help="Posterior samples per chain.")
    parser.add_argument("--ncores", type=int, default=1, help="CPUs per task.")
    parser.add_argument("--max-tree-depth", type=int, default=12, help="NUTS max tree depth.")
    parser.add_argument("--partition", default="day_amd", help="SLURM partition.")
    parser.add_argument("--time", default="2:00:00", help="SLURM time limit.")
    parser.add_argument("--mem", default="20G", help="SLURM memory request.")
    parser.add_argument("--env", default="jaxcpu2", help="Conda environment to activate inside submitted jobs.")
    args = parser.parse_args()
    if args.fit == "chisq" and not args.chisq_csv:
        parser.error("--chisq-csv is required when --fit chisq is used.")
    return args


def get_git_short_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        git_hash = result.stdout.strip()
        return git_hash or "nogit"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def make_run_stamp() -> str:
    now = datetime.now()
    return f"{now.strftime('%b').lower()}{now.day}_{now.strftime('%I%M%p').lower()}"


def load_chisq_ids(chisq_csv: str) -> list[str]:
    df = pd.read_csv(REPO_ROOT / chisq_csv)
    if "object_id" not in df.columns:
        raise KeyError(f"{chisq_csv} is missing an 'object_id' column.")
    return df["object_id"].astype(str).tolist()


def load_stone_ids() -> list[str]:
    return resolve_stone_object_ids()


def load_macleod_ids() -> list[str]:
    return resolve_macleod_object_ids()


def build_job_configs(fit: str, chisq_csv: str) -> list[JobConfig]:
    if fit == "chisq":
        return [
            JobConfig(
                description="chisq",
                object_ids=load_chisq_ids(chisq_csv),
                use_psf_constant_flux=True,
            )
        ]
    if fit == "stone":
        stone_object_ids = load_stone_ids()
        return [
            JobConfig(description="stone", object_ids=stone_object_ids),
            JobConfig(
                description="stone_nolinear",
                object_ids=stone_object_ids,
                extra_flags=("--disable_linear_trend",),
            ),
            JobConfig(
                description="stone_rf2400",
                object_ids=stone_object_ids,
                extra_flags=("--rf_length_cut", "2400"),
            ),
            JobConfig(
                description="stone_rf2400_nolinear",
                object_ids=stone_object_ids,
                extra_flags=("--disable_linear_trend", "--rf_length_cut", "2400"),
            ),
        ]
    if fit == "macleod":
        return [
            JobConfig(
                description="macleod",
                object_ids=load_macleod_ids(),
                use_psf_constant_flux=False,
            )
        ]
    raise ValueError(f"Unsupported fit mode: {fit}")


def validate_chunking(total_objects: int, n_per_job: int, skip: int, num_jobs: int) -> tuple[int, int, int]:
    if n_per_job <= 0:
        raise ValueError(f"N must be > 0, got {n_per_job}.")
    total_chunks = math.ceil(total_objects / n_per_job)
    if skip < 0 or skip > total_chunks:
        raise ValueError(f"skip={skip} is invalid for total_chunks={total_chunks}.")
    if num_jobs < 0:
        num_jobs = total_chunks - skip
    if num_jobs <= 0:
        raise ValueError(
            f"Nothing to submit: total_objects={total_objects}, total_chunks={total_chunks}, "
            f"skip={skip}, num_jobs={num_jobs}."
        )
    return total_chunks, skip, skip + num_jobs - 1


def build_flag_lines(flags: list[str]) -> str:
    return " \\\n ".join(flags)


def build_mail_lines() -> str:
    return "#SBATCH --mail-type=ALL\n"


def build_stone_identity_plot_path(prefix: str) -> str:
    return str(REPO_ROOT / "plots" / "lc_tests" / prefix / "sigma_tau_identity_grid.pdf")


def build_macleod_identity_plot_path(prefix: str) -> str:
    return str(REPO_ROOT / "plots" / "lc_tests" / prefix / "sigma_tau_identity_grid.pdf")


def build_object_ids_path(prefix: str, job: JobConfig) -> Path:
    return SCRIPT_DIR / f"{prefix}_{job.description}_object_ids.txt"


def build_sbatch_script(
    prefix: str,
    job: JobConfig,
    args,
    chisq_csv: str,
    spectra_fit_csv: str,
    object_ids_path: Path | None = None,
) -> str:
    log_dir = LOG_ROOT / prefix
    log_pattern = log_dir / f"{prefix}-%A_%a-%j.txt"
    if args.fit != "chisq" and object_ids_path is None:
        raise ValueError(f"object_ids_path is required for fit mode {args.fit!r}.")
    filter_csv = str(REPO_ROOT / chisq_csv) if chisq_csv is not None else ""
    object_id_file = str(object_ids_path) if object_ids_path is not None else ""
    base_flags = [
        "--plot",
        "--disable_trace_plot",
        "--disable_correlation_plot",
        "--disable_histogram_plot",
        "--disable_corner_plot",
        "--disable_sigma_tau_lambda_plot",
        "--disable_recovery_plot",
        "--fit_method",
        "svi+nuts",
        "--nwarm",
        str(args.nwarm),
        "--nsamp",
        str(args.nsamp),
        "--nchains",
        str(args.ncores),
        "--max_tree_depth",
        str(args.max_tree_depth),
    ]
    if job.use_psf_constant_flux:
        base_flags.extend(
            [
                "--subtract_psf_constant_flux",
                "--spectra_fit_csv",
                spectra_fit_csv,
            ]
        )
    base_flags.extend(job.extra_flags)
    return f"""#!/bin/bash
#SBATCH --job-name=multiband_{prefix}
#SBATCH --output={log_pattern}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={args.ncores}
#SBATCH --mem={args.mem}
#SBATCH --partition={args.partition}
#SBATCH --time={args.time}
{build_mail_lines()}\

set -euo pipefail

export JAX_ENABLE_X64=True
export PREFIX="{prefix}"
export SUFFIX="job${{SLURM_ARRAY_TASK_ID}}"
export NUM_CORES="{args.ncores}"
export N="{args.N}"
export SKIP="{args.skip}"
export TASK_FALLBACK="{args.skip}"
export FIT_MODE="{args.fit}"
export FILTER_CSV="{filter_csv}"
export OBJECT_ID_FILE="{object_id_file}"
export START=""
export END=""

module load miniconda
conda activate {args.env}

cd "{REPO_ROOT}"

start_epoch=$(date +%s)
echo "Start epoch: $start_epoch"
echo "SLURM_JOB_ID=${{SLURM_JOB_ID:-}} SLURM_ARRAY_JOB_ID=${{SLURM_ARRAY_JOB_ID:-}} SLURM_ARRAY_TASK_ID=${{SLURM_ARRAY_TASK_ID:-}}"

TASK_ID="${{SLURM_ARRAY_TASK_ID:-$TASK_FALLBACK}}"
START=$(( TASK_ID * N ))
END=$(( START + N ))
export START
export END
echo "Slice rows: [$START:$END)"

IDS=$(
python - <<'PY'
import os
import pandas as pd
from itertools import islice

fit_mode = os.environ["FIT_MODE"]
start = int(os.environ["START"])
end = int(os.environ["END"])

if fit_mode == "chisq":
    df = pd.read_csv(os.environ["FILTER_CSV"])
    ids = df["object_id"].astype(str).tolist()[start:end]
else:
    with open(os.environ["OBJECT_ID_FILE"], encoding="utf-8") as fh:
        ids = [line.strip() for line in islice(fh, start, end) if line.strip()]

print(" ".join(ids))
PY
)

if [ -z "${{IDS}}" ]; then
  echo "No object_ids for TASK_ID=${{TASK_ID}} (slice $START:$END). Exiting."
  end_epoch=$(date +%s)
  rt=$(( end_epoch - start_epoch ))
  echo "Total runtime: $((rt/3600))h $(((rt%3600)/60))m $((rt%60))s"
  exit 0
fi

echo "object_ids: $IDS"

python -m qvc.light_curve.fit_light_curves \\
 --filter_object_id $IDS \\
 {build_flag_lines(base_flags)}

end_epoch=$(date +%s)
rt=$(( end_epoch - start_epoch ))
echo "End epoch: $end_epoch"
echo "Total runtime: $((rt/3600))h $(((rt%3600)/60))m $((rt%60))s"
"""


def build_merge_sbatch_script(
    prefix: str,
    args,
    *,
    enable_stone_identity_plot: bool = False,
    enable_macleod_identity_plot: bool = False,
) -> str:
    log_dir = LOG_ROOT / prefix
    log_pattern = log_dir / f"{prefix}-merge-%j.txt"
    merge_cmd = f'python -m qvc.light_curve.merge_results "{prefix}" --compute-variability'
    if enable_stone_identity_plot:
        merge_cmd += (
            " --plot-stone-sigma-tau-identity-grid"
            f' --stone-identity-plot-out "{build_stone_identity_plot_path(prefix)}"'
        )
    if enable_macleod_identity_plot:
        merge_cmd += (
            " --plot-macleod-sigma-tau-identity-grid"
            f' --macleod-identity-plot-out "{build_macleod_identity_plot_path(prefix)}"'
        )
    return f"""#!/bin/bash
#SBATCH --job-name=merge_{prefix}
#SBATCH --output={log_pattern}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={args.mem}
#SBATCH --partition={args.partition}
#SBATCH --time={args.time}
{build_mail_lines()}\

set -euo pipefail

module load miniconda
conda activate {args.env}

cd "{REPO_ROOT}"

start_epoch=$(date +%s)
echo "Start epoch: $start_epoch"
echo "SLURM_JOB_ID=${{SLURM_JOB_ID:-}}"

{merge_cmd}

end_epoch=$(date +%s)
rt=$(( end_epoch - start_epoch ))
echo "End epoch: $end_epoch"
echo "Total runtime: $((rt/3600))h $(((rt%3600)/60))m $((rt%60))s"
"""


def write_job_script(prefix: str, sbatch_script: str) -> Path:
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_ROOT / prefix).mkdir(parents=True, exist_ok=True)
    sbatch_path = SCRIPT_DIR / f"{prefix}.sh"
    sbatch_path.write_text(sbatch_script)
    os.chmod(sbatch_path, 0o755)
    return sbatch_path


def write_object_ids_file(object_ids_path: Path, object_ids: list[str]) -> Path:
    object_ids_path.parent.mkdir(parents=True, exist_ok=True)
    object_ids_path.write_text("".join(f"{object_id}\n" for object_id in object_ids), encoding="utf-8")
    return object_ids_path


def parse_sbatch_job_id(stdout: str) -> str:
    parts = stdout.strip().split()
    for token in reversed(parts):
        if token.isdigit():
            return token
    raise ValueError(f"Could not parse sbatch job id from output: {stdout!r}")


def run_sbatch(cmd: list[str]) -> str:
    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or exc.output or "").strip()
        stderr = (exc.stderr or "").strip()
        details = [f"sbatch failed with exit code {exc.returncode}: {' '.join(cmd)}"]
        if stdout:
            details.append(f"stdout:\n{stdout}")
        if stderr:
            details.append(f"stderr:\n{stderr}")
        raise RuntimeError("\n".join(details)) from exc
    stdout = result.stdout.strip()
    if stdout:
        print(stdout)
    if result.stderr.strip():
        print(result.stderr.strip())
    return parse_sbatch_job_id(stdout)


def submit_merge_script(merge_sbatch_path: Path, dependency_job_id: str, prefix: str) -> str:
    cmd = [
        "sbatch",
        f"--dependency=afterany:{dependency_job_id}",
        str(merge_sbatch_path),
    ]
    print(
        f"Submitting merge job for {prefix}: {' '.join(cmd)} "
        f"(depends on {dependency_job_id})"
    )
    merge_job_id = run_sbatch(cmd)
    print(f"Submitted merge job {merge_job_id} for {prefix} after dependency {dependency_job_id}")
    return merge_job_id


def submit_script(
    sbatch_path: Path,
    merge_sbatch_path: Path,
    start_task: int,
    end_task: int,
    total_objects: int,
    fit_label: str,
    prefix: str,
) -> None:
    if total_objects == 0:
        raise ValueError(f"No object_ids found for {fit_label}.")

    print(
        f"Found {total_objects} objects for {fit_label} -> submitting tasks {start_task}-{end_task} "
        f"(chunk size N={args.N})"
    )
    if start_task == end_task:
        cmd = ["sbatch", str(sbatch_path)]
        print("Submitting:", " ".join(cmd))
        job_id = run_sbatch(cmd)
        print(f"Submitted light-curve job {job_id} for {prefix}")
        submit_merge_script(merge_sbatch_path, job_id, prefix)
        return

    for batch_start in range(start_task, end_task + 1, MAX_ARRAY_SIZE):
        batch_end = min(batch_start + MAX_ARRAY_SIZE - 1, end_task)
        cmd = ["sbatch", f"--array={batch_start}-{batch_end}", str(sbatch_path)]
        print("Submitting:", " ".join(cmd))
        job_id = run_sbatch(cmd)
        print(f"Submitted light-curve job {job_id} for {prefix} array range {batch_start}-{batch_end}")
        submit_merge_script(merge_sbatch_path, job_id, prefix)


def main():
    global args
    args = parse_args()
    git_hash = get_git_short_hash()
    run_stamp = make_run_stamp()
    chisq_csv = args.chisq_csv
    spectra_fit_csv = chisq_csv if chisq_csv is not None else DEFAULT_SPECTRA_FIT_CSV

    for job in build_job_configs(args.fit, chisq_csv):
        total_objects = len(job.object_ids)
        _, task_start, task_end = validate_chunking(total_objects, args.N, args.skip, args.num_jobs)
        prefix = f"{run_stamp}_{git_hash}_{job.description}"
        object_ids_path = None
        if args.fit != "chisq":
            object_ids_path = write_object_ids_file(build_object_ids_path(prefix, job), job.object_ids)
        sbatch_script = build_sbatch_script(
            prefix,
            job,
            args,
            chisq_csv,
            spectra_fit_csv,
            object_ids_path=object_ids_path,
        )
        merge_sbatch_script = build_merge_sbatch_script(
            prefix,
            args,
            enable_stone_identity_plot=job.description.startswith("stone"),
            enable_macleod_identity_plot=job.description == "macleod",
        )
        sbatch_path = write_job_script(prefix, sbatch_script)
        merge_sbatch_path = write_job_script(f"{prefix}_merge", merge_sbatch_script)
        submit_script(
            sbatch_path,
            merge_sbatch_path,
            task_start,
            task_end,
            total_objects,
            job.description,
            prefix,
        )


if __name__ == "__main__":
    main()
