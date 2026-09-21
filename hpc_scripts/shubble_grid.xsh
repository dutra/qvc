#!/usr/bin/env xonsh
"""Generate and submit the paper-profile Hubble N-by-redshift Slurm grid."""

import argparse
import base64
import itertools
import json
import math
import os
from datetime import datetime
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qvc.hubble.completeness_grid import grid_bin_count
PROVENANCE_ENV = "QVC_SUBMISSION_PROVENANCE_B64"


DEFAULT_N_VALUES = tuple(range(1000, 8001, 1000))
DEFAULT_ZMAX_VALUES = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
SPEED_CHOICES = ("fastest", "quicker", "quick", "medium", "standard", "production")

SPECTRA_FIT_CSV = "results/data/jaxqsofit/aug31_w500s500_d01c034.h5"
H5_FILE = (
    "results/data/"
    "sep09_svi10000lr0003w500s250_specaug31w500s250_83cb31d.h5"
)

PAPER_SETTING_DEFAULTS = {
    "QVC_HUBBLE_MAGNITUDE_CONVENTION": "dereddened",
    "QVC_HUBBLE_COMPLETENESS_MAGNITUDE": "attenuated",
    "QVC_HUBBLE_COMPLETENESS_MAGNITUDE_SUPPORT_MODE": "hard-cut",
    "QVC_HUBBLE_COMPLETENESS_LF_MODEL": "wang2026_type1_lade_a",
    "QVC_HUBBLE_CUT_TIER": "2",
    "QVC_HUBBLE_LIGHT_CURVE_UNCERTAINTY_MODE": "covariance",
    "QVC_HUBBLE_COMPLETENESS_MAG_BIN_WIDTH": "0.2",
    "QVC_HUBBLE_COMPLETENESS_Z_BIN_WIDTH": "0.2",
    "QVC_HUBBLE_COMPLETENESS_SMOOTH_SIGMA_MAG": "0.10",
    "QVC_HUBBLE_COMPLETENESS_SMOOTH_SIGMA_Z": "0.30",
}

CUT_THRESHOLD_DEFAULTS = {
    "QVC_CUT_COMPLETENESS_MAG_2500_MIN": "17",
    "QVC_CUT_COMPLETENESS_MAG_2500_MAX": "24",
    "QVC_CUT_JAXSEDFIT_JOINT_REDUCED_CHI2_MAX": "1.2",
    "QVC_CUT_SED_REDUCED_CHI2_MAX": "1.3",
    "QVC_CUT_SPECTROSCOPY_REDUCED_CHI2_MAX": "1.1",
    "QVC_CUT_LOO_CHI2_EFF_MAX": "1.05",
    "QVC_CUT_SPECTRAL_RHAT_MAX": "1.05",
    "QVC_CUT_LIGHT_CURVE_RHAT_MAX": "1.05",
    "QVC_CUT_NUM_DIVERGENCES_MAX": "0",
    "QVC_CUT_LOG_TAU_UV_RF_MIN": "1.3",
    "QVC_CUT_T_RF_LENGTH_MIN": "none",
    "QVC_CUT_LIGHT_CURVE_N_POINTS_MIN": "none",
    "QVC_CUT_SN_MEDIAN_ALL_MIN": "3",
    "QVC_CUT_VARIABILITY_CHI_SQ_RED_G_MIN": "20",
    "QVC_CUT_ETA_SIGMA_KL_MIN": "none",
    "QVC_CUT_LOG_TAU_UV_RF_MAX": "none",
    "QVC_CUT_T_RF_OVER_TAU_UV_RF_MIN": "none",
    "QVC_CUT_APPARENT_MAG_2500_ERR_MAX": "none",
    "QVC_CUT_F_BC_3000_MAX": "none",
    "QVC_CUT_F_FE_UV_3000_MAX": "none",
    "QVC_CUT_F_HOST_2500_PSF_MAX": "none",
    "QVC_CUT_FRAC_AGN_5100_MIN": "none",
    "QVC_CUT_A_2500_TOTAL_MAX": "none",
    "QVC_CUT_EBV_GAL_PLUS_EBV_AGN_MAX": "none",
    "QVC_CUT_LOW_L2500_FHOST_LOG_L_MAX": "none",
    "QVC_CUT_LOW_L2500_FHOST_PSF_MAX": "none",
}


def submission_record(entrypoint, argv, resolved):
    """Build the dependency-free submission envelope consumed by QVC runs."""

    return {
        "entrypoint": entrypoint,
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "argv": [str(value) for value in argv],
        "command": shlex.join([str(value) for value in argv]),
        "resolved": resolved,
    }


def encode_record(record):
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return base64.b64encode(payload.encode("utf-8")).decode("ascii")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--description", required=True, help="Short campaign description.")
    parser.add_argument("--dry-run", action="store_true", help="Write the batch script without submitting it.")
    parser.add_argument("--n-values", type=int, nargs="+", default=list(DEFAULT_N_VALUES))
    parser.add_argument("--zmax-values", type=float, nargs="+", default=list(DEFAULT_ZMAX_VALUES))
    parser.add_argument(
        "--speed",
        choices=SPEED_CHOICES,
        default=os.environ.get("QVC_HUBBLE_SPEED", "quick"),
    )
    parser.add_argument("--ncores", type=int, default=12)
    parser.add_argument("--partition", default="day")
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument("--mem", default="32G")
    parser.add_argument("--env", default="jaxcpu2", help="Conda environment activated in each task.")
    parser.add_argument("--skip", type=int, default=0, help="First absolute grid task to submit.")
    parser.add_argument("--num-jobs", type=int, default=-1, help="Number of grid tasks to submit; -1 submits the remainder.")
    parser.add_argument(
        "--completeness-mag-bin-width",
        type=float,
        default=float(os.environ.get("QVC_HUBBLE_COMPLETENESS_MAG_BIN_WIDTH", "0.2")),
    )
    parser.add_argument(
        "--completeness-z-bin-width",
        type=float,
        default=float(os.environ.get("QVC_HUBBLE_COMPLETENESS_Z_BIN_WIDTH", "0.2")),
    )
    parser.add_argument(
        "--completeness-smooth-sigma-mag",
        type=float,
        default=float(os.environ.get("QVC_HUBBLE_COMPLETENESS_SMOOTH_SIGMA_MAG", "0.10")),
    )
    parser.add_argument(
        "--completeness-smooth-sigma-z",
        type=float,
        default=float(os.environ.get("QVC_HUBBLE_COMPLETENESS_SMOOTH_SIGMA_Z", "0.30")),
    )
    return parser.parse_args(argv)


def normalize_description(value):
    description = re.sub(r"[^A-Za-z0-9.-]+", "_", str(value).strip()).strip("_.-")
    if not description:
        raise ValueError("--description cannot be blank")
    return description.lower()


def get_git_short_hash(repo_root=REPO_ROOT):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "nogit"
    return result.stdout.strip() or "nogit"


def make_run_stamp():
    return datetime.now().strftime("%b%d_%H%M").lower()


def build_campaign_name(args, run_stamp, git_hash):
    return "_".join(
        (run_stamp, "hubble_grid", args.speed, args.description, git_hash)
    )


def grid_cells(n_values, zmax_values):
    return list(itertools.product(n_values, zmax_values))


def validate_args(args):
    args.description = normalize_description(args.description)
    args.n_values = list(dict.fromkeys(args.n_values))
    args.zmax_values = list(dict.fromkeys(args.zmax_values))
    if not args.n_values or any(value <= 0 for value in args.n_values):
        raise ValueError("--n-values must contain positive integers")
    if not args.zmax_values or any(not math.isfinite(value) or value <= 0.44 for value in args.zmax_values):
        raise ValueError("--zmax-values must contain finite values greater than 0.44")
    if args.ncores <= 0:
        raise ValueError("--ncores must be positive")
    if args.skip < 0:
        raise ValueError("--skip must be nonnegative")
    if args.num_jobs == 0 or args.num_jobs < -1:
        raise ValueError("--num-jobs must be -1 or a positive integer")
    for value, label in (
        (args.completeness_smooth_sigma_mag, "--completeness-smooth-sigma-mag"),
        (args.completeness_smooth_sigma_z, "--completeness-smooth-sigma-z"),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{label} must be finite and nonnegative")
    grid_bin_count(8.0, args.completeness_mag_bin_width)
    grid_bin_count(4.5, args.completeness_z_bin_width)
    if args.completeness_mag_bin_width > 1:
        raise ValueError("--completeness-mag-bin-width must be at most 1 mag")

    total = len(args.n_values) * len(args.zmax_values)
    if args.skip >= total:
        raise ValueError(f"--skip={args.skip} is outside the grid of {total} tasks")
    stop = total if args.num_jobs == -1 else min(total, args.skip + args.num_jobs)
    return args.skip, stop - 1


def resolve_paper_settings(args, environ=None):
    environ = os.environ if environ is None else environ
    settings = {
        name: str(environ.get(name, default)).strip()
        for name, default in PAPER_SETTING_DEFAULTS.items()
    }
    settings.update(
        {
            "QVC_HUBBLE_COMPLETENESS_MAG_BIN_WIDTH": str(args.completeness_mag_bin_width),
            "QVC_HUBBLE_COMPLETENESS_Z_BIN_WIDTH": str(args.completeness_z_bin_width),
            "QVC_HUBBLE_COMPLETENESS_SMOOTH_SIGMA_MAG": str(args.completeness_smooth_sigma_mag),
            "QVC_HUBBLE_COMPLETENESS_SMOOTH_SIGMA_Z": str(args.completeness_smooth_sigma_z),
        }
    )
    cuts = {
        name: str(environ.get(name, default)).strip()
        for name, default in CUT_THRESHOLD_DEFAULTS.items()
    }
    allowed_values = {
        "QVC_HUBBLE_COMPLETENESS_MAGNITUDE_SUPPORT_MODE": {"tails", "hard-cut"},
        "QVC_HUBBLE_CUT_TIER": {"none", "0", "1", "2"},
        "QVC_HUBBLE_LIGHT_CURVE_UNCERTAINTY_MODE": {"covariance", "posterior-draws"},
    }
    for name, choices in allowed_values.items():
        if settings[name] not in choices:
            expected = ", ".join(sorted(choices))
            raise ValueError(f"{name} must be one of {expected}; got {settings[name]!r}")
    return settings, cuts


def hubble_arguments(*, n_value, zmax, prefix, speed, settings):
    return [
        "-m", "qvc.hubble.hubble_fit",
        "--bright-subsample-completeness-min", "0.1",
        "--bright-subsample-margin", "0.2",
        "--cosmo_models", "FlatLambdaCDM", "FlatwCDM", "Flatw0waCDM",
        "--run", "single",
        "--speed", str(speed),
        "--spectra_fit_csv", SPECTRA_FIT_CSV,
        "--magnitude-convention", settings["QVC_HUBBLE_MAGNITUDE_CONVENTION"],
        "--completeness_magnitude", settings["QVC_HUBBLE_COMPLETENESS_MAGNITUDE"],
        "--completeness-magnitude-support-mode", settings["QVC_HUBBLE_COMPLETENESS_MAGNITUDE_SUPPORT_MODE"],
        "--completeness_lf_model", settings["QVC_HUBBLE_COMPLETENESS_LF_MODEL"],
        "--completeness-mag-bin-width", settings["QVC_HUBBLE_COMPLETENESS_MAG_BIN_WIDTH"],
        "--completeness-z-bin-width", settings["QVC_HUBBLE_COMPLETENESS_Z_BIN_WIDTH"],
        "--completeness-smooth-sigma-mag", settings["QVC_HUBBLE_COMPLETENESS_SMOOTH_SIGMA_MAG"],
        "--completeness-smooth-sigma-z", settings["QVC_HUBBLE_COMPLETENESS_SMOOTH_SIGMA_Z"],
        "--light-curve-uncertainty-mode", settings["QVC_HUBBLE_LIGHT_CURVE_UNCERTAINTY_MODE"],
        "--cut-tier", settings["QVC_HUBBLE_CUT_TIER"],
        "--result_prefix", "fiducial",
        "--z_range", "0.44", str(zmax),
        "--prefix", str(prefix),
        "--disable_sigma_clip_pass",
        "--sigma_clip_threshold", "3.0",
        "--prior-profile", "centered_lcdm",
        "--early-de-guard",
        "--skip-debiased-residual-plot",
        "--N", str(n_value),
        "--uniform_redshift_distribution",
        "--minimal-plots",
        H5_FILE,
    ]


def shell_array(values):
    return " ".join(shlex.quote(str(value)) for value in values)


def shell_command_word(value):
    text = str(value)
    variable_words = {
        "${N}": '"$N"',
        "${ZMAX}": '"$ZMAX"',
        "${CURRENT_PREFIX}": '"$CURRENT_PREFIX"',
    }
    return variable_words.get(text, shlex.quote(text))


def build_sbatch_script(args, campaign, settings, cuts, provenance):
    log_dir = REPO_ROOT / "hpc_scripts" / "logs" / "hubble" / campaign
    template_args = hubble_arguments(
        n_value="${N}",
        zmax="${ZMAX}",
        prefix="${CURRENT_PREFIX}",
        speed=args.speed,
        settings=settings,
    )
    command = " \\\n    ".join(shell_command_word(value) for value in ["python", *template_args])
    exports = {
        **settings,
        **cuts,
        "SHEN_PUBTOOLS_PATH": os.environ.get(
            "SHEN_PUBTOOLS_PATH",
            "/home/id255/project_pi_pn38/id255/quasarlf/pubtools/",
        ),
    }
    export_lines = "\n".join(
        f"export {name}={shlex.quote(str(value))}" for name, value in exports.items()
    )
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={campaign[:120]}
#SBATCH --output={log_dir}/grid_%A_%a.out
#SBATCH --error={log_dir}/grid_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={args.ncores}
#SBATCH --mem={args.mem}
#SBATCH --partition={args.partition}
#SBATCH --time={args.time}

set -euo pipefail

export JAX_ENABLE_X64=True
export QT_QPA_PLATFORM=offscreen
export NUM_CORES="${{SLURM_CPUS_PER_TASK:-{args.ncores}}}"
export PYTHONPATH={shlex.quote(str(REPO_ROOT / 'src'))}
export MPLCONFIGDIR="${{SLURM_TMPDIR:-/tmp}}/qvc-matplotlib-${{SLURM_JOB_ID:-local}}"
export CAMPAIGN={shlex.quote(campaign)}
export {PROVENANCE_ENV}={shlex.quote(provenance)}
{export_lines}

N_VALUES=({shell_array(args.n_values)})
ZMAX_VALUES=({shell_array(args.zmax_values)})
TASK_ID="${{SLURM_ARRAY_TASK_ID:-0}}"
NUM_ZMAX="${{#ZMAX_VALUES[@]}}"
N_INDEX=$(( TASK_ID / NUM_ZMAX ))
ZMAX_INDEX=$(( TASK_ID % NUM_ZMAX ))

if (( N_INDEX >= ${{#N_VALUES[@]}} )); then
    echo "Task ID $TASK_ID is outside the configured grid" >&2
    exit 2
fi

N="${{N_VALUES[$N_INDEX]}}"
ZMAX="${{ZMAX_VALUES[$ZMAX_INDEX]}}"
CURRENT_PREFIX="${{CAMPAIGN}}/N${{N}}_zmax${{ZMAX}}"
export PREFIX="$CURRENT_PREFIX"

module load miniconda
conda activate {shlex.quote(args.env)}
mkdir -p "$MPLCONFIGDIR"
cd {shlex.quote(str(REPO_ROOT))}

echo "Started $(date --iso-8601=seconds) on $(hostname)"
echo "SLURM_JOB_ID=${{SLURM_JOB_ID:-}} TASK_ID=$TASK_ID N=$N ZMAX=$ZMAX"
echo "PREFIX=$PREFIX NUM_CORES=$NUM_CORES"

{command}

echo "Finished $(date --iso-8601=seconds)"
"""


def write_executable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def submit_sbatch(script_path, first_task, last_task):
    command = ["sbatch", f"--array={first_task}-{last_task}", str(script_path)]
    print("Submitting:", shlex.join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main(argv=None):
    args = parse_args(argv)
    first_task, last_task = validate_args(args)
    campaign = build_campaign_name(
        args, make_run_stamp(), get_git_short_hash()
    )
    settings, cuts = resolve_paper_settings(args)
    cells = grid_cells(args.n_values, args.zmax_values)
    provenance_record = submission_record(
        "hpc_scripts/shubble_grid.xsh",
        [str(Path(__file__)), *(sys.argv[1:] if argv is None else argv)],
        {
            "campaign": campaign,
            "grid": {"n_values": args.n_values, "zmax_values": args.zmax_values, "zmin": 0.44},
            "selected_task_range": [first_task, last_task],
            "resources": {
                "cpus_per_task": args.ncores,
                "memory": args.mem,
                "partition": args.partition,
                "time": args.time,
                "environment": args.env,
            },
            "science_settings": settings,
            "cut_thresholds": cuts,
            "inputs": {"h5_file": H5_FILE, "spectra_fit_csv": SPECTRA_FIT_CSV},
        },
    )
    script_path = REPO_ROOT / "hpc_scripts" / "submit" / "hubble" / f"submit_{campaign}.sbatch"
    log_dir = REPO_ROOT / "hpc_scripts" / "logs" / "hubble" / campaign
    if script_path.exists() or log_dir.exists():
        raise FileExistsError(f"Refusing to reuse campaign artifacts for {campaign!r}")
    log_dir.mkdir(parents=True)
    write_executable(
        script_path,
        build_sbatch_script(args, campaign, settings, cuts, encode_record(provenance_record)),
    )
    print(f"Campaign: {campaign}")
    print(f"Grid cells: {len(cells)}; submitting tasks {first_task}-{last_task}")
    print(f"Results root: {REPO_ROOT / 'results' / 'cosmo' / campaign}")
    print(f"Generated: {script_path}")
    if args.dry_run:
        print("Dry run: batch script was not submitted.")
        return script_path
    submit_sbatch(script_path, first_task, last_task)
    return script_path


if __name__ == "__main__":
    main()
