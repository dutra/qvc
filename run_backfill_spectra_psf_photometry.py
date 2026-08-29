#!/usr/bin/env python3
"""Build the canonical mandatory-color v3 catalog from chunks and bundles.

The original v2 chunks supply scalar metadata plus the already selected host
and ugriz AGN-fraction draws. Saved posterior bundles supply inexpensive
analytic v3 quantities and exactly one selected-64 ``pred_fluxes`` prediction.
No inference, plots, DR16Q join, or full-posterior prediction is performed.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

SOURCE_RUN_NAME = (
    "aug24_0152pm_spectrafit_e5d2897_chisqgt20_N8000_nested_fhostpsf"
)
DEFAULT_SOURCE_RUN = REPOSITORY_ROOT / "results/data/jaxqsofit" / SOURCE_RUN_NAME
DEFAULT_OUTPUT = DEFAULT_SOURCE_RUN.with_name(
    f"{SOURCE_RUN_NAME}_resumed_m2500norm12_v3.h5"
)
DEFAULT_COMPATIBLE_PYTHON = Path(
    "/home/dutra/.conda/envs/jaxcpu5_sdev/bin/python"
)
REEXEC_ENV = "QVC_DIRECT_COLOR_BACKFILL_REEXEC"


def _ensure_runtime(python_bin: Path) -> None:
    missing = [
        name
        for name in ("h5py", "jaxsedfit", "numpy", "pandas", "tqdm")
        if importlib.util.find_spec(name) is None
    ]
    if not missing:
        return
    if os.environ.get(REEXEC_ENV) == "1":
        raise RuntimeError(f"Compatible runtime still lacks {missing}.")
    if not python_bin.is_file():
        raise RuntimeError(
            f"Current Python lacks {missing}, and {python_bin} does not exist."
        )
    env = dict(os.environ)
    env[REEXEC_ENV] = "1"
    env.setdefault("JAX_ENABLE_X64", "True")
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    env.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.setdefault("MPLCONFIGDIR", f"/tmp/qvc-color-backfill-{os.getuid()}")
    os.execve(
        str(python_bin),
        [str(python_bin), str(Path(__file__).resolve()), *sys.argv[1:]],
        env,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8, metavar="N")
    parser.add_argument(
        "--max-tasks-per-worker",
        type=int,
        default=1,
        help=(
            "Source chunks processed before recycling a JAX worker (default: "
            "1). Keep this at 1 to bound XLA compilation-cache memory."
        ),
    )
    parser.add_argument("--selection-seed", type=int, default=3)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_COMPATIBLE_PYTHON)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--object-id", nargs="+")
    parser.add_argument("--keep-shards", action="store_true")
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--force-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_tasks_per_worker < 1:
        parser.error("--max-tasks-per-worker must be positive")
    if args.start < 0:
        parser.error("--start cannot be negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    args.source_run = args.source_run.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.python_bin = args.python_bin.expanduser().resolve()
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if argv is None:
        _ensure_runtime(args.python_bin)

    import run_resume_spectra_local as direct

    delegated = argparse.Namespace(
        source_run=str(args.source_run),
        output_catalog=str(args.output),
        joint_posterior_selection_seed=int(args.selection_seed),
        python_bin=str(args.python_bin),
        start=int(args.start),
        limit=args.limit,
        object_id=args.object_id,
        parallel=int(args.workers),
        max_tasks_per_worker=int(args.max_tasks_per_worker),
        dry_run=bool(args.dry_run),
        keep_v3_shards=bool(args.keep_shards),
        merge=not bool(args.no_merge),
        force_output=bool(args.force_output),
        provenance_entrypoint="run_backfill_spectra_psf_photometry.py",
        driver_path=str(Path(__file__).resolve()),
    )
    return direct.run_v3_build(delegated, REPOSITORY_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
