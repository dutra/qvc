#!/usr/bin/env python3
import argparse
import math
import os
import re
import shlex
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
from qvc.provenance import PROVENANCE_ENV, encode_record, submission_record

SCRIPT_DIR = REPO_ROOT / "hpc_scripts" / "jobs" / "multibandfit"
LOG_ROOT = REPO_ROOT / "hpc_scripts" / "logs" / "multibandfit"
MAX_ARRAY_SIZE = 10_000


@dataclass(frozen=True)
class JobConfig:
    description: str
    object_ids: list[str]
    extra_flags: tuple[str, ...] = ()
    use_psf_constant_flux: bool = False


def parse_args():
    parser = argparse.ArgumentParser(description="Submit multiband-fit SLURM jobs.")
    parser.add_argument(
        "--fit",
        choices=("chisq", "stone", "macleod", "samelength"),
        required=True,
        help="Sample to submit.",
    )
    parser.add_argument(
        "--stone-linear-mode",
        choices=("both", "linear", "nolinear"),
        default="both",
        help=(
            "Stone linear-trend variants to submit: both (default), only the "
            "standard linear-trend fit, or only the no-linear-trend fit."
        ),
    )
    parser.add_argument("--chisq-csv", type=str, default=None, help="CSV file with object_id column for --fit chisq.")
    parser.add_argument(
        "--spectra-fit-h5",
        type=str,
        default=None,
        help="Spectra-fit HDF5 used for PSF-fraction treatment in --fit chisq jobs.",
    )
    parser.add_argument("--num-jobs", type=int, default=-1, help="-1 means submit all chunks after skip.")
    parser.add_argument("--skip", type=int, default=0, help="Number of chunks to skip.")
    parser.add_argument("--N", type=int, default=10, help="Objects per array task.")
    parser.add_argument("--nwarm", type=int, default=250, help="Warmup steps.")
    parser.add_argument("--nsamp", type=int, default=250, help="Posterior samples per chain.")
    parser.add_argument("--svi-steps", type=int, default=4000, help="SVI warm-start steps.")
    parser.add_argument("--svi-lr", type=float, default=1e-3, help="SVI learning rate.")
    parser.add_argument("--ncores", type=int, default=1, help="CPUs per task.")
    parser.add_argument("--max-tree-depth", type=int, default=12, help="NUTS max tree depth.")
    parser.add_argument("--partition", default="day", help="SLURM partition.")
    parser.add_argument("--time", default="12:00:00", help="SLURM time limit.")
    parser.add_argument("--mem", default="32G", help="SLURM memory request.")
    parser.add_argument("--env", default="jaxcpu2", help="Conda environment to activate inside submitted jobs.")
    parser.add_argument(
        "--description",
        default=None,
        help="Optional run description inserted after the run stamp in fresh run prefixes.",
    )
    parser.add_argument(
        "--resume",
        metavar="PREFIX_BASE",
        default=None,
        help=(
            "Resume an existing run by shared prefix base. The per-job prefix is built as "
            "PREFIX_BASE plus the job description, for example PREFIX_BASE_stone."
        ),
    )
    args, extra_fit_flags = parser.parse_known_args()
    args.extra_fit_flags = tuple(extra_fit_flags)
    if args.fit == "chisq" and not args.chisq_csv:
        parser.error("--chisq-csv is required when --fit chisq is used.")
    if args.fit == "chisq" and not args.spectra_fit_h5:
        parser.error("--spectra-fit-h5 is required when --fit chisq is used.")
    if args.fit != "stone" and args.stone_linear_mode != "both":
        parser.error(
            "--stone-linear-mode linear or nolinear requires --fit stone."
        )
    try:
        args.description = normalize_run_description(args.description)
    except ValueError as exc:
        parser.error(str(exc))
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
    return now.strftime("%b%d_%I%M%p").lower()


def normalize_run_description(description: str | None) -> str | None:
    if description is None:
        return None
    normalized = description.strip().lower().replace(" ", "_")
    if not normalized:
        raise ValueError("--description cannot be blank.")
    return normalized


def load_chisq_ids(chisq_csv: str) -> list[str]:
    df = pd.read_csv(REPO_ROOT / chisq_csv)
    if "object_id" not in df.columns:
        raise KeyError(f"{chisq_csv} is missing an 'object_id' column.")
    return df["object_id"].astype(str).tolist()


def load_stone_ids() -> list[str]:
    return resolve_stone_object_ids()


def load_macleod_ids() -> list[str]:
    return resolve_macleod_object_ids()


def build_job_configs(
    fit: str,
    chisq_csv: str,
    *,
    stone_linear_mode: str = "both",
) -> list[JobConfig]:
    if stone_linear_mode not in {"both", "linear", "nolinear"}:
        raise ValueError(
            "stone_linear_mode must be one of: both, linear, nolinear."
        )
    if fit != "stone" and stone_linear_mode != "both":
        raise ValueError(
            "stone_linear_mode linear or nolinear requires fit='stone'."
        )
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
        jobs = [
            JobConfig(description="stone", object_ids=stone_object_ids),
            JobConfig(
                description="stone_nolinear",
                object_ids=stone_object_ids,
                extra_flags=("--disable_linear_trend",),
            ),
        ]
        if stone_linear_mode == "linear":
            return jobs[:1]
        if stone_linear_mode == "nolinear":
            return jobs[1:]
        return jobs
    if fit == "samelength":
        stone_object_ids = load_stone_ids()
        return [
            JobConfig(description="samelength_fulllength", object_ids=stone_object_ids),
            JobConfig(
                description="samelength_fulllength_nolinear",
                object_ids=stone_object_ids,
                extra_flags=("--disable_linear_trend",),
            ),
            JobConfig(
                description="samelength_rf2400",
                object_ids=stone_object_ids,
                extra_flags=("--rf_length_cut", "2400"),
            ),
            JobConfig(
                description="samelength_rf2400_nolinear",
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
    return " \\\n ".join(shlex.quote(flag) for flag in flags)


def build_mail_lines() -> str:
    return "#SBATCH --mail-type=ALL\n"


def build_stone_identity_plot_path(prefix: str, job_description: str) -> str:
    filename = f"sigma_tau_identity_grid_{job_description}.pdf"
    return str(REPO_ROOT / "plots" / "lc_tests" / prefix / filename)


def build_macleod_identity_plot_path(prefix: str, job_description: str) -> str:
    filename = f"sigma_tau_identity_grid_{job_description}.pdf"
    return str(REPO_ROOT / "plots" / "lc_tests" / prefix / filename)


def build_suberlak_identity_plot_path(prefix: str, job_description: str) -> str:
    filename = f"sigma_tau_identity_grid_suberlak_{job_description}.pdf"
    return str(REPO_ROOT / "plots" / "lc_tests" / prefix / filename)


def build_samelength_comparison_plot_dir(run_prefix_base: str) -> Path:
    return REPO_ROOT / "plots" / "lc_tests" / f"{run_prefix_base}_samelength_comparison"


def build_samelength_comparison_plot_path(run_prefix_base: str, x_description: str, y_description: str) -> str:
    y_label = y_description.removeprefix("samelength_")
    filename = f"sigma_tau_identity_grid_{x_description}_vs_{y_label}.pdf"
    return str(build_samelength_comparison_plot_dir(run_prefix_base) / filename)


def build_object_ids_path(prefix: str, job: JobConfig) -> Path:
    return SCRIPT_DIR / f"{prefix}_{job.description}_object_ids.txt"


def build_run_prefix(
    job_description: str,
    run_stamp: str,
    git_hash: str,
    resume_prefix_base: str | None,
    run_description: str | None = None,
) -> str:
    if resume_prefix_base:
        return f"{resume_prefix_base}_{job_description}"
    if run_description:
        return f"{run_stamp}_{run_description}_{git_hash}_{job_description}"
    return f"{run_stamp}_{git_hash}_{job_description}"


def build_fit_job_name(prefix: str) -> str:
    """Build the scheduler name while keeping the result prefix unchanged."""

    match = re.fullmatch(
        r"(?P<date>[a-z]{3}\d{2})_(?P<time>\d{4}(?:am|pm))_(?P<identity>.+)",
        prefix,
    )
    if match is None:
        return f"lcfit_{prefix}"
    return (
        f"{match.group('date')}_{match.group('time')}_"
        f"lcfit_{match.group('identity')}"
    )


def build_sbatch_script(
    prefix: str,
    job: JobConfig,
    args,
    chisq_csv: str,
    spectra_fit_h5: str | None,
    object_ids_path: Path | None = None,
) -> str:
    log_dir = LOG_ROOT / prefix
    log_pattern = log_dir / f"{prefix}-%A_%a-%j.txt"
    job_name = build_fit_job_name(prefix)
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
        #"--plot_ls_broken_pl",
        "--disable_color_magnitude_plot",
        "--disable_recovery_plot",
        "--disable_sigma_tau_lambda_plot",
        "--disable_recovery_plot",
        "--fit_method",
        "svi+nuts",
        "--svi_steps",
        str(args.svi_steps),
        "--svi_lr",
        str(args.svi_lr),
        "--nwarm",
        str(args.nwarm),
        "--nsamp",
        str(args.nsamp),
        "--nchains",
        str(args.ncores),
        "--max_tree_depth",
        str(args.max_tree_depth),
    ]
    if getattr(args, "resume", None):
        base_flags.append("--resume")
    if job.use_psf_constant_flux:
        base_flags.extend(
            [
                "--subtract_psf_constant_flux",
                "--spectra_fit_h5",
                spectra_fit_h5,
            ]
        )
    base_flags.extend(job.extra_flags)
    base_flags.extend(getattr(args, "extra_fit_flags", ()))
    submission = submission_record(
        "hpc_scripts/sfitlc.py",
        sys.argv,
        {
            "wrapper_args": vars(args),
            "job": {
                "description": job.description,
                "prefix": prefix,
                "object_count": len(job.object_ids),
                "extra_flags": job.extra_flags,
                "use_psf_constant_flux": job.use_psf_constant_flux,
            },
            "inputs": {
                "chisq_csv": chisq_csv,
                "spectra_fit_h5": spectra_fit_h5,
                "object_ids_path": object_id_file,
            },
            "resources": {
                "cpus_per_task": args.ncores,
                "memory": args.mem,
                "partition": args.partition,
                "time": args.time,
                "environment": args.env,
            },
            "fit_flags": base_flags,
        },
    )
    encoded_submission = encode_record(submission)
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
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
export NUM_CORES="{args.ncores}"
export N="{args.N}"
export SKIP="{args.skip}"
export TASK_FALLBACK="{args.skip}"
export FIT_MODE="{args.fit}"
export FILTER_CSV="{filter_csv}"
export OBJECT_ID_FILE="{object_id_file}"
export START=""
export END=""
export {PROVENANCE_ENV}="{encoded_submission}"

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

read -r -a OBJECT_IDS <<< "$IDS"
OBJECT_INDEX=0
for OBJECT_ID in "${{OBJECT_IDS[@]}}"; do
  export SUFFIX="job${{TASK_ID}}_obj${{OBJECT_INDEX}}"
  object_start_epoch=$(date +%s)
  echo "Starting object $((OBJECT_INDEX + 1))/${{#OBJECT_IDS[@]}}: $OBJECT_ID (SUFFIX=$SUFFIX)"

  python -m qvc.light_curve.fit_light_curves \\
   --filter_object_id "$OBJECT_ID" \\
   {build_flag_lines(base_flags)}

  object_end_epoch=$(date +%s)
  object_rt=$(( object_end_epoch - object_start_epoch ))
  echo "Finished object $OBJECT_ID in $((object_rt/3600))h $(((object_rt%3600)/60))m $((object_rt%60))s"
  OBJECT_INDEX=$((OBJECT_INDEX + 1))
done

end_epoch=$(date +%s)
rt=$(( end_epoch - start_epoch ))
echo "End epoch: $end_epoch"
echo "Total runtime: $((rt/3600))h $(((rt%3600)/60))m $((rt%60))s"
"""


def build_merge_sbatch_script(
    prefix: str,
    job_description: str,
    args,
    *,
    enable_stone_identity_plot: bool = False,
    enable_macleod_identity_plot: bool = False,
    enable_suberlak_identity_plot: bool = False,
) -> str:
    log_dir = LOG_ROOT / prefix
    log_pattern = log_dir / f"{prefix}-merge-%j.txt"
    comparison_only = enable_stone_identity_plot or enable_macleod_identity_plot
    merge_mode_flag = (
        "--skip-populate-sdss"
        if comparison_only
        else "--compute-variability"
    )
    merge_memory = "20G" if comparison_only else "40G"
    merge_cmd = (
        f'python -m qvc.light_curve.merge_results "{prefix}" {merge_mode_flag}'
    )
    if enable_stone_identity_plot:
        merge_cmd += (
            " --plot-stone-sigma-tau-identity-grid"
            f' --stone-identity-plot-out "{build_stone_identity_plot_path(prefix, job_description)}"'
        )
    if enable_macleod_identity_plot:
        merge_cmd += (
            " --plot-macleod-sigma-tau-identity-grid"
            f' --macleod-identity-plot-out "{build_macleod_identity_plot_path(prefix, job_description)}"'
        )
    if enable_suberlak_identity_plot:
        merge_cmd += (
            " --plot-suberlak-sigma-tau-identity-grid"
            f' --suberlak-identity-plot-out "{build_suberlak_identity_plot_path(prefix, job_description)}"'
        )
    submission = submission_record(
        "hpc_scripts/sfitlc.py",
        sys.argv,
        {
            "wrapper_args": vars(args),
            "merge_prefix": prefix,
            "job_description": job_description,
            "merge_command": merge_cmd,
            "resources": {
                "memory": merge_memory,
                "partition": args.partition,
                "time": args.time,
                "environment": args.env,
            },
        },
    )
    encoded_submission = encode_record(submission)
    return f"""#!/bin/bash
#SBATCH --job-name=merge_{prefix}
#SBATCH --output={log_pattern}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={merge_memory}
#SBATCH --partition={args.partition}
#SBATCH --time={args.time}
{build_mail_lines()}\

set -euo pipefail

module load miniconda
conda activate {args.env}

export {PROVENANCE_ENV}="{encoded_submission}"

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


def build_samelength_comparison_sbatch_script(run_prefix_base: str, args) -> str:
    prefix = f"{run_prefix_base}_samelength_comparison"
    log_dir = LOG_ROOT / prefix
    log_pattern = log_dir / f"{prefix}-%j.txt"
    comparisons = [
        ("samelength_rf2400", "samelength_fulllength"),
        ("samelength_rf2400_nolinear", "samelength_fulllength_nolinear"),
    ]
    commands = []
    for x_description, y_description in comparisons:
        x_prefix = build_run_prefix(x_description, "", "", run_prefix_base)
        y_prefix = build_run_prefix(y_description, "", "", run_prefix_base)
        plot_out = build_samelength_comparison_plot_path(run_prefix_base, x_description, y_description)
        commands.append(
            "python -m qvc.light_curve.merge_results"
            " --plot-samelength-sigma-tau-identity-grid"
            f' --samelength-x-prefix "{x_prefix}"'
            f' --samelength-y-prefix "{y_prefix}"'
            f' --samelength-identity-plot-out "{plot_out}"'
        )
    comparison_cmds = "\n".join(commands)
    return f"""#!/bin/bash
#SBATCH --job-name=samelength_cmp_{run_prefix_base}
#SBATCH --output={log_pattern}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
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

{comparison_cmds}

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


def submit_merge_script(merge_sbatch_path: Path, dependency_job_ids: str | list[str], prefix: str) -> str:
    if isinstance(dependency_job_ids, str):
        dependency_job_ids = [dependency_job_ids]
    if not dependency_job_ids:
        raise ValueError(f"No dependency job IDs provided for merge job {prefix}.")
    dependency_text = ":".join(dependency_job_ids)
    cmd = [
        "sbatch",
        f"--dependency=afterany:{dependency_text}",
        str(merge_sbatch_path),
    ]
    print(
        f"Submitting merge job for {prefix}: {' '.join(cmd)} "
        f"(depends on {dependency_text})"
    )
    merge_job_id = run_sbatch(cmd)
    print(f"Submitted merge job {merge_job_id} for {prefix} after dependency {dependency_text}")
    return merge_job_id


def submit_script(
    sbatch_path: Path,
    merge_sbatch_path: Path,
    start_task: int,
    end_task: int,
    total_objects: int,
    fit_label: str,
    prefix: str,
) -> str:
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
        return submit_merge_script(merge_sbatch_path, job_id, prefix)

    job_ids = []
    for batch_start in range(start_task, end_task + 1, MAX_ARRAY_SIZE):
        batch_end = min(batch_start + MAX_ARRAY_SIZE - 1, end_task)
        cmd = ["sbatch", f"--array={batch_start}-{batch_end}", str(sbatch_path)]
        print("Submitting:", " ".join(cmd))
        job_id = run_sbatch(cmd)
        job_ids.append(job_id)
        print(f"Submitted light-curve job {job_id} for {prefix} array range {batch_start}-{batch_end}")
    return submit_merge_script(merge_sbatch_path, job_ids, prefix)


def submit_samelength_comparison_script(
    comparison_sbatch_path: Path,
    dependency_job_ids: list[str],
    run_prefix_base: str,
) -> str:
    if not dependency_job_ids:
        raise ValueError("No merge job IDs provided for samelength comparison job.")
    dependency_text = ":".join(dependency_job_ids)
    cmd = [
        "sbatch",
        f"--dependency=afterany:{dependency_text}",
        str(comparison_sbatch_path),
    ]
    print(
        f"Submitting samelength comparison job for {run_prefix_base}: {' '.join(cmd)} "
        f"(depends on {dependency_text})"
    )
    comparison_job_id = run_sbatch(cmd)
    print(f"Submitted samelength comparison job {comparison_job_id} for {run_prefix_base}")
    return comparison_job_id


def main():
    global args
    args = parse_args()
    git_hash = get_git_short_hash()
    run_stamp = make_run_stamp()
    run_prefix_base = args.resume or "_".join(
        part for part in (run_stamp, args.description, git_hash) if part
    )
    chisq_csv = args.chisq_csv
    spectra_fit_h5 = args.spectra_fit_h5
    samelength_merge_job_ids = []

    for job in build_job_configs(
        args.fit,
        chisq_csv,
        stone_linear_mode=args.stone_linear_mode,
    ):
        total_objects = len(job.object_ids)
        _, task_start, task_end = validate_chunking(total_objects, args.N, args.skip, args.num_jobs)
        prefix = build_run_prefix(job.description, run_stamp, git_hash, args.resume, args.description)
        object_ids_path = None
        if args.fit != "chisq":
            object_ids_path = write_object_ids_file(build_object_ids_path(prefix, job), job.object_ids)
        sbatch_script = build_sbatch_script(
            prefix,
            job,
            args,
            chisq_csv,
            spectra_fit_h5,
            object_ids_path=object_ids_path,
        )
        merge_sbatch_script = build_merge_sbatch_script(
            prefix,
            job.description,
            args,
            enable_stone_identity_plot=args.fit == "stone" and job.description.startswith("stone"),
            enable_macleod_identity_plot=job.description == "macleod",
            enable_suberlak_identity_plot=job.description == "macleod",
        )
        sbatch_path = write_job_script(prefix, sbatch_script)
        merge_sbatch_path = write_job_script(f"{prefix}_merge", merge_sbatch_script)
        merge_job_id = submit_script(
            sbatch_path,
            merge_sbatch_path,
            task_start,
            task_end,
            total_objects,
            job.description,
            prefix,
        )
        if args.fit == "samelength":
            samelength_merge_job_ids.append(merge_job_id)

    if args.fit == "samelength":
        comparison_sbatch_script = build_samelength_comparison_sbatch_script(run_prefix_base, args)
        comparison_sbatch_path = write_job_script(
            f"{run_prefix_base}_samelength_comparison",
            comparison_sbatch_script,
        )
        submit_samelength_comparison_script(
            comparison_sbatch_path,
            samelength_merge_job_ids,
            run_prefix_base,
        )


if __name__ == "__main__":
    main()
