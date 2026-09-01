#!/usr/bin/env xonsh
"""Generate and submit a Bouchet Slurm Hubble-validation campaign."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys


# ==========================================
# 1. Edit campaign and Bouchet settings here
# ==========================================
hpc_home = Path("/home/id255")
repo_dir = Path("/home/id255/project_pi_pn38/id255/qvc")
python_bin = hpc_home / ".conda/envs/jaxcpu2/bin/python"
output_root = repo_dir / "results/hubble_validation"

speed = "quick"
num_agns = 2000
num_runs = 64
campaign = f"fixed_truth_nagns{num_agns}_nruns{num_runs}_{speed}"
seed_start = 0
master_seed = 20260901
calibration_size = 200000
arms = ["all", "selected_uncorrected", "selected_oracle", "selected_estimated"]

# Fixed injected truth and catalog/selection settings.
h0 = 70.0
om0 = 0.30
w0 = -1.0
wa = 0.0
alpha = 7.0
beta = -1.0
m0 = -23.0
scatter_mag = 0.5
log_sigma_pivot = -0.8
log_sigma_scale = 0.2
log_tau_pivot = 2.7
log_tau_scale = 0.4
m50 = 23.0
selection_width = 0.3
lf_area_deg2 = 20.0
lf_model = "wang2026_type1_lade_a"
z_min = 0.1
z_max = 4.0
lf_mag_min = 14.0
lf_mag_max = 28.0

# Fit-array resources. One array task runs every enabled arm for one seed.
partition = "day"
time_limit = "02:00:00"
cpus_per_task = 8
memory = "32G"
max_concurrent = 9999

# Dependent plotting-job resources.
plot_partition = partition
plot_time_limit = "00:30:00"
plot_cpus = 1
plot_memory = "8G"

# Resume an existing compatible campaign and optionally retry recorded failures.
resume_campaign = False
retry_failed = False

# Additional run_hubble_validation.py flags, for uncommon overrides.
extra_runner_args = []


ARM_CHOICES = (
    "all",
    "selected_uncorrected",
    "selected_oracle",
    "selected_estimated",
)
SPEED_CHOICES = ("fastest", "quick", "standard", "production")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Write and validate batch scripts without initializing or submitting the campaign.")
    parser.add_argument("--resume", action="store_true", default=resume_campaign)
    parser.add_argument("--retry-failed", action="store_true", default=retry_failed)
    parser.add_argument("--campaign", default=campaign)
    parser.add_argument("--repo-dir", type=Path, default=repo_dir)
    parser.add_argument("--python-bin", type=Path, default=python_bin)
    parser.add_argument("--output-root", type=Path, default=output_root)
    parser.add_argument("--num-agns", type=int, default=num_agns)
    parser.add_argument("--num-runs", type=int, default=num_runs)
    parser.add_argument("--seed-start", type=int, default=seed_start)
    parser.add_argument("--master-seed", type=int, default=master_seed)
    parser.add_argument("--speed", choices=SPEED_CHOICES, default=speed)
    parser.add_argument("--calibration-size", type=int, default=calibration_size)
    parser.add_argument("--arms", nargs="+", choices=ARM_CHOICES, default=arms)
    parser.add_argument("--partition", default=partition)
    parser.add_argument("--time", default=time_limit)
    parser.add_argument("--cpus-per-task", type=int, default=cpus_per_task)
    parser.add_argument("--mem", default=memory)
    parser.add_argument("--max-concurrent", type=int, default=max_concurrent)
    parser.add_argument("--plot-partition", default=plot_partition)
    parser.add_argument("--plot-time", default=plot_time_limit)
    parser.add_argument("--plot-cpus", type=int, default=plot_cpus)
    parser.add_argument("--plot-mem", default=plot_memory)
    return parser.parse_args(argv)


def normalize_campaign(value):
    text = str(value).strip()
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError("campaign must be a relative path below output-root")
    return text


def slurm_name(value):
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    if not normalized:
        raise ValueError("campaign does not contain any Slurm-safe job-name characters")
    return normalized[:80]


def submission_prefix(now=None):
    """Return a local-time prefix such as ``sep01_0552pm_``."""

    current = datetime.now().astimezone() if now is None else now
    return current.strftime("%b%d_%I%M%p_").lower()


def runner_arguments(args):
    return [
        "--campaign", args.campaign,
        "--output-root", str(args.output_root),
        "--n-runs", str(args.num_runs),
        "--seed-start", str(args.seed_start),
        "--master-seed", str(args.master_seed),
        "--num-agns", str(args.num_agns),
        "--calibration-size", str(args.calibration_size),
        "--lf-area-deg2", str(lf_area_deg2),
        "--lf-model", str(lf_model),
        "--z-min", str(z_min),
        "--z-max", str(z_max),
        "--lf-mag-min", str(lf_mag_min),
        "--lf-mag-max", str(lf_mag_max),
        "--h0", str(h0),
        "--om0", str(om0),
        "--w0", str(w0),
        "--wa", str(wa),
        "--alpha", str(alpha),
        "--beta", str(beta),
        "--m0", str(m0),
        "--scatter-mag", str(scatter_mag),
        "--log-sigma-pivot", str(log_sigma_pivot),
        "--log-sigma-scale", str(log_sigma_scale),
        "--log-tau-pivot", str(log_tau_pivot),
        "--log-tau-scale", str(log_tau_scale),
        "--m50", str(m50),
        "--selection-width", str(selection_width),
        "--speed", args.speed,
        "--arms", *args.arms,
        *[str(value) for value in extra_runner_args],
    ]


def validate_settings(args):
    args.campaign = normalize_campaign(args.campaign)
    args.repo_dir = args.repo_dir.expanduser().resolve()
    args.python_bin = args.python_bin.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.num_agns <= 0 or args.num_runs <= 0 or args.calibration_size <= 0:
        raise ValueError("num-agns, num-runs, and calibration-size must be positive")
    if args.seed_start < 0 or args.cpus_per_task <= 0 or args.plot_cpus <= 0:
        raise ValueError("seed-start must be nonnegative and CPU counts must be positive")
    if args.max_concurrent <= 0:
        raise ValueError("max-concurrent must be positive")
    if not args.python_bin.is_file():
        raise FileNotFoundError(f"Configured Python does not exist: {args.python_bin}")
    runner = args.repo_dir / "scripts/run_hubble_validation.py"
    plotter = args.repo_dir / "scripts/plot_hubble_validation.py"
    for path in (runner, plotter):
        if not path.is_file():
            raise FileNotFoundError(f"Required validation script does not exist: {path}")
    return runner, plotter


def write_executable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def bash_command(arguments):
    return shlex.join([str(value) for value in arguments])


def build_fit_script(args, runner, log_dir, job_name):
    constant_command = bash_command(
        [
            args.python_bin,
            runner,
            *runner_arguments(args),
            "--resume",
            *(["--retry-failed"] if args.retry_failed else []),
        ]
    )
    max_running = min(args.max_concurrent, args.num_runs)
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/fit_%A_%a.out
#SBATCH --error={log_dir}/fit_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --mem={args.mem}
#SBATCH --partition={args.partition}
#SBATCH --time={args.time}
#SBATCH --array=0-{args.num_runs - 1}%{max_running}

set -euo pipefail

export QVC_REPO_DIR={shlex.quote(str(args.repo_dir))}
export PYTHONPATH={shlex.quote(str(args.repo_dir / 'src'))}
export MPLBACKEND=Agg
export MPLCONFIGDIR="${{SLURM_TMPDIR:-/tmp}}/qvc-matplotlib-${{SLURM_JOB_ID}}"
export QT_QPA_PLATFORM=offscreen
export NUM_CORES="${{SLURM_CPUS_PER_TASK}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$MPLCONFIGDIR"
cd "$QVC_REPO_DIR"

TASK_ID="${{SLURM_ARRAY_TASK_ID}}"
REALIZATION=$(({args.seed_start} + TASK_ID))
echo "Started $(date --iso-8601=seconds) on $(hostname)"
echo "SLURM_JOB_ID=${{SLURM_JOB_ID}} TASK_ID=$TASK_ID REALIZATION=$REALIZATION NUM_CORES=$NUM_CORES"

{constant_command} --realization "$REALIZATION"

echo "Finished $(date --iso-8601=seconds)"
"""


def build_plot_script(args, plotter, campaign_dir, log_dir, job_name):
    plot_command = bash_command([args.python_bin, plotter, campaign_dir])
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/plot_%j.out
#SBATCH --error={log_dir}/plot_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={args.plot_cpus}
#SBATCH --mem={args.plot_mem}
#SBATCH --partition={args.plot_partition}
#SBATCH --time={args.plot_time}

set -euo pipefail

export QVC_REPO_DIR={shlex.quote(str(args.repo_dir))}
export PYTHONPATH={shlex.quote(str(args.repo_dir / 'src'))}
export MPLBACKEND=Agg
export MPLCONFIGDIR="${{SLURM_TMPDIR:-/tmp}}/qvc-matplotlib-${{SLURM_JOB_ID}}"
export QT_QPA_PLATFORM=offscreen
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$MPLCONFIGDIR"
cd "$QVC_REPO_DIR"

echo "Plot job started $(date --iso-8601=seconds) on $(hostname)"
{plot_command}
echo "Plot job finished $(date --iso-8601=seconds)"
"""


def parse_job_id(stdout):
    token = stdout.strip().splitlines()[-1].split(";", 1)[0].strip() if stdout.strip() else ""
    if not token.isdigit():
        raise ValueError(f"Could not parse Slurm job ID from: {stdout!r}")
    return token


def submit(command, cwd):
    print("Submitting:", bash_command(command))
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stderr.strip():
        print(completed.stderr.strip())
    print(completed.stdout.strip())
    return parse_job_id(completed.stdout)


def write_metadata(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    args = parse_args(argv)
    runner, plotter = validate_settings(args)
    prefix = submission_prefix()
    artifact_name = slurm_name(f"{prefix}{args.campaign}")
    fit_job_name = slurm_name(f"{prefix}hval_{args.campaign}")
    plot_job_name = slurm_name(f"{prefix}hval_plot_{args.campaign}")
    submit_dir = args.repo_dir / "hpc_scripts/submit/hubble_validation"
    log_dir = args.repo_dir / "hpc_scripts/logs/hubble_validation" / artifact_name
    campaign_dir = args.output_root / args.campaign
    fit_script = submit_dir / f"{artifact_name}_fits.sbatch"
    plot_script = submit_dir / f"{artifact_name}_plot.sbatch"

    if not args.dry_run:
        init_command = [
            args.python_bin,
            runner,
            *runner_arguments(args),
            "--initialize-only",
            *(["--resume"] if args.resume else []),
        ]
        init_env = os.environ.copy()
        init_env.update(
            {
                "PYTHONPATH": str(args.repo_dir / "src"),
                "MPLBACKEND": "Agg",
                "MPLCONFIGDIR": "/tmp/qvc-hubble-validation-init",
                "QT_QPA_PLATFORM": "offscreen",
                "NUM_CORES": "1",
            }
        )
        print("Initializing:", bash_command(init_command))
        subprocess.run(init_command, cwd=args.repo_dir, env=init_env, check=True)

    log_dir.mkdir(parents=True, exist_ok=True)
    write_executable(
        fit_script,
        build_fit_script(args, runner, log_dir, fit_job_name),
    )
    write_executable(
        plot_script,
        build_plot_script(args, plotter, campaign_dir, log_dir, plot_job_name),
    )
    for path in (fit_script, plot_script):
        subprocess.run(["bash", "-n", str(path)], check=True)
    print(f"Fit batch script: {fit_script}")
    print(f"Plot batch script: {plot_script}")

    settings = {
        "campaign": args.campaign,
        "campaign_dir": str(campaign_dir),
        "submission_prefix": prefix,
        "artifact_name": artifact_name,
        "job_names": {"fit": fit_job_name, "plot": plot_job_name},
        "hpc_home": str(hpc_home),
        "python_bin": str(args.python_bin),
        "repo_dir": str(args.repo_dir),
        "num_agns": args.num_agns,
        "num_runs": args.num_runs,
        "seed_start": args.seed_start,
        "master_seed": args.master_seed,
        "speed": args.speed,
        "calibration_size": args.calibration_size,
        "arms": list(args.arms),
        "retry_failed": bool(args.retry_failed),
        "fit_resources": {
            "partition": args.partition,
            "time": args.time,
            "memory": args.mem,
            "cpus_per_task": args.cpus_per_task,
            "max_concurrent": args.max_concurrent,
        },
        "plot_resources": {
            "partition": args.plot_partition,
            "time": args.plot_time,
            "memory": args.plot_mem,
            "cpus_per_task": args.plot_cpus,
        },
        "scripts": {"fit": str(fit_script), "plot": str(plot_script)},
    }

    if args.dry_run:
        preview_path = submit_dir / f"{artifact_name}_submission_preview.json"
        write_metadata(
            preview_path,
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "dry_run": True,
                "settings": settings,
            },
        )
        print(f"Dry run only; no campaign was initialized and no jobs were submitted: {preview_path}")
        return 0

    metadata_path = campaign_dir / f"{prefix}hpc_submission.json"
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": False,
        "settings": settings,
        "job_ids": {},
    }
    fit_job_id = submit(["sbatch", "--parsable", str(fit_script)], args.repo_dir)
    metadata["job_ids"]["fit_array"] = fit_job_id
    write_metadata(metadata_path, metadata)
    plot_job_id = submit(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{fit_job_id}",
            str(plot_script),
        ],
        args.repo_dir,
    )
    metadata["job_ids"]["plot"] = plot_job_id
    metadata["plot_dependency"] = f"afterany:{fit_job_id}"
    write_metadata(metadata_path, metadata)
    print(f"Submitted fit array {fit_job_id} and dependent plot job {plot_job_id}")
    print(f"Submission metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
