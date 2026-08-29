#!/usr/bin/env python3
"""Render SED and spectrum PNGs directly from saved JAXSEDFit samples.

This driver deliberately performs no fitting, catalog construction, posterior
bundle writes, or diagnostic generation.  Each selected posterior bundle is
loaded once and its predictive calculation is shared by the two plots.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import importlib.util
import multiprocessing as mp
import os
import sys
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parent
RUN_NAME = "aug24_0152pm_spectrafit_e5d2897_chisqgt20_N8000_nested_fhostpsf"
DEFAULT_SOURCE_DIR = REPOSITORY_ROOT / "results/data/jaxqsofit" / RUN_NAME / "all"
DEFAULT_PLOT_DIR = REPOSITORY_ROOT / "plots" / f"{RUN_NAME}_plots"
DEFAULT_PYTHON = Path("/home/dutra/.conda/envs/jaxcpu5_sdev/bin/python")
REEXEC_ENV = "QVC_PLOT_SPECTRA_SAMPLES_REEXEC"
BUNDLE_SUFFIX = "_samples.h5"


@dataclass(frozen=True)
class PlotTask:
    bundle_path: Path
    sed_path: Path
    spectrum_path: Path
    make_sed: bool
    make_spectrum: bool


@dataclass(frozen=True)
class PlotResult:
    bundle_path: Path
    generated: int
    skipped: int
    error: str = ""


def parse_posterior_draws(value: str):
    """Parse ``median``, ``all``, or a positive integer draw count."""
    normalized = str(value).strip().lower()
    if normalized in {"median", "all"}:
        return normalized
    try:
        count = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be 'median', 'all', or a positive integer"
        ) from exc
    if count < 1:
        raise argparse.ArgumentTypeError(
            "must be 'median', 'all', or a positive integer"
        )
    return count


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument(
        "--object-id-csv",
        type=Path,
        help=(
            "CSV containing object_id values to plot. IDs are resolved to "
            "sample bundles through the run's chunk catalogs."
        ),
    )
    parser.add_argument(
        "--object-id-column",
        default="object_id",
        help="Object-ID column in --object-id-csv (default: object_id).",
    )
    parser.add_argument(
        "--posterior-draws",
        required=True,
        type=parse_posterior_draws,
        metavar="{median,N,all}",
        help=(
            "Predict at per-site posterior medians, at N deterministically "
            "spaced draws, or at all saved draws."
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-tasks-per-worker",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Recycle each spawned JAX worker after N bundles to release native "
            "compiled state (default: 1, safest for varying spectrum shapes)."
        ),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Keep nonempty existing PNGs instead of regenerating them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report discovered work without loading JAXSEDFit or writing files.",
    )
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_tasks_per_worker < 1:
        parser.error("--max-tasks-per-worker must be positive")
    if args.start < 0:
        parser.error("--start cannot be negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    args.source_dir = args.source_dir.expanduser().resolve()
    args.plot_dir = args.plot_dir.expanduser().resolve()
    args.python_bin = args.python_bin.expanduser().resolve()
    if args.object_id_csv is not None:
        args.object_id_csv = args.object_id_csv.expanduser().resolve()
        if not args.object_id_csv.is_file():
            parser.error(f"--object-id-csv does not exist: {args.object_id_csv}")
    return args


def _configure_runtime_environment() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "True")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
    os.environ.setdefault(
        "MPLCONFIGDIR", f"/tmp/qvc-plot-spectra-samples-{os.getuid()}"
    )


def _ensure_runtime(python_bin: Path) -> None:
    missing = [
        name
        for name in ("h5py", "jax", "jaxsedfit", "matplotlib", "numpy")
        if importlib.util.find_spec(name) is None
    ]
    if not missing:
        return
    if os.environ.get(REEXEC_ENV) == "1":
        raise RuntimeError(f"Compatible plotting runtime still lacks {missing}.")
    if not python_bin.is_file():
        raise RuntimeError(
            f"Current Python lacks {missing}, and {python_bin} does not exist."
        )
    environment = dict(os.environ)
    environment[REEXEC_ENV] = "1"
    os.execve(
        str(python_bin),
        [str(python_bin), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


def observation_id_from_bundle(path: Path) -> str:
    if not path.name.endswith(BUNDLE_SUFFIX):
        raise ValueError(f"Not a JAXSEDFit sample bundle: {path}")
    observation_id = path.name[: -len(BUNDLE_SUFFIX)]
    if not observation_id:
        raise ValueError(f"Bundle has an empty observation ID: {path}")
    return observation_id


def plot_paths(bundle_path: Path, plot_dir: Path) -> tuple[Path, Path]:
    observation_id = observation_id_from_bundle(bundle_path)
    spectrum_stem = (
        observation_id[: -len("_joint")]
        if observation_id.endswith("_joint")
        else observation_id
    )
    return (
        plot_dir / f"{observation_id}.png",
        plot_dir / f"{spectrum_stem}_spectrum.png",
    )


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def normalize_object_id(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def load_requested_object_ids(csv_path: Path, column: str) -> list[str]:
    """Load unique, nonempty object IDs while preserving CSV row order."""
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames or []) or "none"
            raise ValueError(
                f"Column {column!r} is missing from {csv_path}; "
                f"available columns: {available}."
            )
        object_ids = []
        seen = set()
        for row_number, row in enumerate(reader, start=2):
            object_id = normalize_object_id(row.get(column))
            if not object_id:
                raise ValueError(
                    f"Empty {column!r} value in {csv_path} at row {row_number}."
                )
            if object_id not in seen:
                seen.add(object_id)
                object_ids.append(object_id)
    if not object_ids:
        raise ValueError(f"No object IDs found in {csv_path}.")
    return object_ids


def _decode_hdf5_strings(values) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def resolve_object_id_bundles(
    source_dir: Path,
    requested_object_ids: list[str],
    available_bundles: Mapping[str, Path],
) -> list[Path]:
    """Resolve catalog object IDs to existing bundle paths in CSV order."""
    import h5py

    requested = set(requested_object_ids)
    matches: dict[str, Path] = {}
    chunk_paths = sorted(source_dir.parent.glob("*_chunk*.h5"))
    if not chunk_paths:
        raise FileNotFoundError(
            f"No chunk catalogs found beside sample directory {source_dir}."
        )

    for chunk_path in chunk_paths:
        with h5py.File(chunk_path, "r") as handle:
            if "catalog/object_id" not in handle or "catalog/fit_result_path" not in handle:
                continue
            object_ids = _decode_hdf5_strings(handle["catalog/object_id"][()])
            result_paths = _decode_hdf5_strings(
                handle["catalog/fit_result_path"][()]
            )
        for object_id, result_path in zip(object_ids, result_paths, strict=True):
            normalized = normalize_object_id(object_id)
            if normalized not in requested or not result_path:
                continue
            bundle = available_bundles.get(Path(result_path).name)
            if bundle is None:
                continue
            previous = matches.get(normalized)
            if previous is not None and previous != bundle:
                raise ValueError(
                    f"Object ID {normalized!r} maps to multiple bundles: "
                    f"{previous.name} and {bundle.name}."
                )
            matches[normalized] = bundle
        if len(matches) == len(requested):
            break

    missing = [object_id for object_id in requested_object_ids if object_id not in matches]
    if missing:
        preview = ", ".join(missing[:20])
        raise ValueError(
            f"No saved sample bundle found for {len(missing)} requested object ID(s): "
            f"{preview}."
        )
    return [matches[object_id] for object_id in requested_object_ids]


def discover_tasks(args) -> list[PlotTask]:
    if not args.source_dir.is_dir():
        raise NotADirectoryError(f"Sample directory not found: {args.source_dir}")
    bundles = sorted(args.source_dir.glob(f"*{BUNDLE_SUFFIX}"))
    object_id_csv = getattr(args, "object_id_csv", None)
    if object_id_csv is not None:
        requested = load_requested_object_ids(
            object_id_csv,
            getattr(args, "object_id_column", "object_id"),
        )
        requested = requested[args.start :]
        if args.limit is not None:
            requested = requested[: args.limit]
        bundles = resolve_object_id_bundles(
            args.source_dir,
            requested,
            {bundle.name: bundle for bundle in bundles},
        )
    else:
        bundles = bundles[args.start :]
        if args.limit is not None:
            bundles = bundles[: args.limit]

    tasks = []
    for bundle in bundles:
        sed_path, spectrum_path = plot_paths(bundle, args.plot_dir)
        make_sed = not (
            args.skip_existing and _is_nonempty_file(sed_path)
        )
        make_spectrum = not (
            args.skip_existing and _is_nonempty_file(spectrum_path)
        )
        tasks.append(
            PlotTask(
                bundle_path=bundle,
                sed_path=sed_path,
                spectrum_path=spectrum_path,
                make_sed=make_sed,
                make_spectrum=make_spectrum,
            )
        )
    return tasks


def select_posterior_samples(
    samples: Mapping[str, object], selection
) -> dict[str, object]:
    """Select aligned draws without changing the bundle on disk."""
    import numpy as np

    arrays = {name: np.asarray(value) for name, value in samples.items()}
    draw_counts = {value.shape[0] for value in arrays.values() if value.ndim > 0}
    if not draw_counts:
        raise ValueError("Posterior bundle contains no sampled draws.")
    if len(draw_counts) != 1:
        raise ValueError(
            f"Posterior sites have inconsistent draw counts: {sorted(draw_counts)}"
        )
    draw_count = draw_counts.pop()
    if draw_count < 1:
        raise ValueError("Posterior bundle contains zero draws.")

    if selection == "all":
        return arrays
    if selection == "median":
        return {
            name: (
                np.expand_dims(np.nanmedian(value, axis=0), axis=0)
                if value.ndim > 0
                else value
            )
            for name, value in arrays.items()
        }

    count = min(int(selection), draw_count)
    indices = np.linspace(0, draw_count - 1, num=count, dtype=int)
    return {
        name: value[indices] if value.ndim > 0 else value
        for name, value in arrays.items()
    }


def _temporary_png_path(destination: Path) -> Path:
    return destination.with_name(
        f".{destination.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.png"
    )


def _atomic_plot(destination: Path, render) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_png_path(destination)
    try:
        render(temporary)
        if not _is_nonempty_file(temporary):
            raise RuntimeError(f"Plot renderer did not write a valid PNG: {temporary}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _load_fitter(bundle_path: Path):
    from jaxsedfit import JAXSEDFit

    return JAXSEDFit.load(bundle_path)


def _release_worker_memory(fitter=None) -> None:
    """Drop Python, JAX compilation, and glibc allocations after one object."""
    if fitter is not None:
        try:
            fitter.predictive = None
            fitter.samples = None
        except Exception:
            pass
    gc.collect()
    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass
    gc.collect()
    # JAX's CPU backend ultimately allocates through libc on Linux.  Clearing
    # Python/JAX references does not always return free arenas to the OS.
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)
    except (AttributeError, OSError):
        pass


def render_task(task: PlotTask, posterior_draws) -> PlotResult:
    skipped = 2 - int(task.make_sed) - int(task.make_spectrum)
    if not task.make_sed and not task.make_spectrum:
        return PlotResult(task.bundle_path, generated=0, skipped=skipped)

    from matplotlib import pyplot as plt

    generated = 0
    fitter = None
    try:
        fitter = _load_fitter(task.bundle_path)
        fitter.samples = select_posterior_samples(
            fitter.samples, posterior_draws
        )
        fitter.predictive = None
        # Both JAXSEDFit plot methods call predict().  Precomputing here fills
        # the fitter cache so the expensive model evaluation happens once.
        prediction_kind = "plot" if task.make_sed else "photometry"
        fitter.predictive = fitter.predict(kind=prediction_kind)

        if task.make_sed:
            def render_sed(path):
                figure = fitter.plot_sed(output_path=path, show=False)
                if figure is not None:
                    plt.close(figure)

            _atomic_plot(task.sed_path, render_sed)
            generated += 1

        if task.make_spectrum:
            def render_spectrum(path):
                figure = fitter.plot_spectrum(
                    show_plot=False,
                    plot_residual=False,
                )
                if figure is None:
                    raise RuntimeError("JAXSEDFit did not return a spectrum figure.")
                try:
                    figure.savefig(path, dpi=150, bbox_inches="tight", format="png")
                finally:
                    plt.close(figure)

            _atomic_plot(task.spectrum_path, render_spectrum)
            generated += 1

        return PlotResult(
            task.bundle_path,
            generated=generated,
            skipped=skipped,
        )
    except Exception:
        plt.close("all")
        return PlotResult(
            task.bundle_path,
            generated=generated,
            skipped=skipped,
            error=traceback.format_exc(),
        )
    finally:
        plt.close("all")
        _release_worker_memory(fitter)


_WORKER_POSTERIOR_DRAWS = None


def _worker_initializer(posterior_draws) -> None:
    global _WORKER_POSTERIOR_DRAWS
    _configure_runtime_environment()
    _WORKER_POSTERIOR_DRAWS = posterior_draws


def _worker(task: PlotTask) -> PlotResult:
    return render_task(task, _WORKER_POSTERIOR_DRAWS)


def _progress(results, *, total):
    try:
        from tqdm import tqdm
    except ImportError:
        return results
    return tqdm(results, total=total, desc="Plotting saved spectra")


def run(args) -> int:
    tasks = discover_tasks(args)
    plots_to_generate = sum(
        int(task.make_sed) + int(task.make_spectrum) for task in tasks
    )
    plots_to_skip = 2 * len(tasks) - plots_to_generate
    print(f"Posterior bundles selected: {len(tasks)}")
    print(f"Plots to generate: {plots_to_generate}")
    print(f"Plots already complete: {plots_to_skip}")
    if args.dry_run:
        print("Dry run: no bundles loaded and no files written.")
        return 0
    if not tasks:
        return 0

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    if args.workers == 1:
        results = (
            render_task(task, args.posterior_draws) for task in tasks
        )
        iterator = _progress(results, total=len(tasks))
        completed = list(iterator)
    else:
        context = mp.get_context("spawn")
        with context.Pool(
            args.workers,
            initializer=_worker_initializer,
            initargs=(args.posterior_draws,),
            maxtasksperchild=args.max_tasks_per_worker,
        ) as pool:
            iterator = _progress(
                pool.imap_unordered(_worker, tasks), total=len(tasks)
            )
            completed = list(iterator)

    failures = [result for result in completed if result.error]
    for result in failures:
        print(f"FAILED: {result.bundle_path}", file=sys.stderr)
        print(result.error.rstrip(), file=sys.stderr)
    print(f"Plots generated: {sum(result.generated for result in completed)}")
    print(f"Plots skipped: {sum(result.skipped for result in completed)}")
    print(f"Bundles failed: {len(failures)}")
    return 1 if failures else 0


def main(argv=None) -> int:
    args = parse_args(argv)
    _configure_runtime_environment()
    if argv is None and (not args.dry_run or args.object_id_csv is not None):
        _ensure_runtime(args.python_bin)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
