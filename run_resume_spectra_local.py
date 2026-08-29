#!/usr/bin/env python3
"""Build a v3 spectral catalog directly from one run's chunks and bundles.

The workflow uses the original per-job v2 chunks only as row metadata and for
their saved PSF-fraction draws.  Every chunk row with a matching immutable
posterior bundle is rebuilt directly into the v3 joint-posterior schema.  The
chunks already contain the deterministically selected host and ugriz draws;
the other jointly indexed quantities are cheap analytic transformations of
the saved latent samples. Work is restartable at one-object HDF5 shards and
the final v3 catalog is written atomically. No prediction, merged v2 catalog,
inference rerun, or plotting is involved.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


# The driver is intentionally runnable as ``python3 run_resume_spectra_local.py``
# from a source checkout.  Add the src-layout package before any delayed qvc
# imports, including the dry-run path which deliberately skips the JAXSedFit
# interpreter handoff.
_REPOSITORY_ROOT = Path(__file__).resolve().parent
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))


SOURCE_RUN_NAME = (
    "aug24_0152pm_spectrafit_e5d2897_chisqgt20_N8000_nested_fhostpsf"
)
DEFAULT_SOURCE_RUN = Path("results/data/jaxqsofit") / SOURCE_RUN_NAME
DEFAULT_V3_CATALOG = DEFAULT_SOURCE_RUN.with_name(
    f"{SOURCE_RUN_NAME}_resumed_m2500norm12_v3.h5"
)
LOCAL_COMPATIBLE_PYTHON = Path(
    "/home/dutra/.conda/envs/jaxcpu5_sdev/bin/python"
)
BUNDLE_SUFFIX = "_joint_samples.h5"
REEXEC_ENV = "QVC_LOCAL_SPECTRA_RESUME_REEXEC"
EXPECTED_JAXSEDFIT_COMMIT = "bc9da74735260bd33b3da2076fd7929fdd592e0d"


@dataclass(frozen=True)
class ResumeObject:
    ordinal: int
    object_id: str
    sdss_name: str
    bundle_path: Path


@dataclass(frozen=True)
class ResumeCommandResult:
    returncode: int
    stdout: str
    stderr: str
    execution_error: str = ""


@dataclass(frozen=True)
class RepairedCatalogRow:
    item: ResumeObject
    row: object
    fraction_draws: object
    fraction_valid_count: int
    host_draws: object
    host_valid_count: int
    source_catalog: Path
    source_catalog_commit: str


@dataclass(frozen=True)
class V3BuildTask:
    """One independently restartable chunk-and-bundle v3 build task."""

    ordinal: int
    object_id: str
    sdss_name: str
    bundle_path: Path
    source_chunk_path: Path
    row: dict[str, object]
    psf_fraction_draws: object
    psf_fraction_valid_count: int
    host_fraction_draws: object
    host_fraction_valid_count: int
    shard_path: Path
    seed: int


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def normalize_object_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def normalize_sdss_name(value: object) -> str:
    """Normalize the historical CSV's stringified ``b'...'`` names."""
    text = str(value).strip()
    if len(text) >= 3 and text[:2] in {"b'", 'b"'}:
        try:
            decoded = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            decoded = None
        if isinstance(decoded, bytes):
            return decoded.decode("utf-8").strip()
    return text


def bundle_sdss_name(path: Path) -> str:
    name = path.name
    if not name.endswith(BUNDLE_SUFFIX):
        raise ValueError(f"Not a joint posterior bundle: {path}")
    stem = name[: -len(BUNDLE_SUFFIX)]
    try:
        redshift_token, sdss_name = stem.split("_", 1)
        float(redshift_token.removeprefix("z"))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Malformed joint posterior bundle name: {path}") from exc
    if not redshift_token.startswith("z") or not sdss_name:
        raise ValueError(f"Malformed joint posterior bundle name: {path}")
    return sdss_name


def resolve_source_bundle_dir(source_run: Path) -> tuple[Path, Path]:
    source_run = source_run.expanduser().resolve()
    if source_run.name == "all" and source_run.is_dir():
        return source_run.parent, source_run
    bundle_dir = source_run / "all"
    if not bundle_dir.is_dir():
        raise FileNotFoundError(
            f"Source run has no posterior-bundle directory: {bundle_dir}"
        )
    return source_run, bundle_dir


def discover_resume_objects(
    source_bundle_dir: Path,
    input_csv: Path,
) -> tuple[list[ResumeObject], list[dict[str, str]]]:
    """Map every downloaded bundle one-to-one onto the original input CSV."""
    with input_csv.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"object_id", "sdss_name"}
        missing_columns = required - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                f"Input CSV {input_csv} is missing columns: {sorted(missing_columns)}"
            )
        rows = list(reader)

    by_sdss_name: dict[str, dict[str, str]] = {}
    for row in rows:
        sdss_name = normalize_sdss_name(row["sdss_name"])
        object_id = normalize_object_id(row["object_id"])
        if not sdss_name or not object_id:
            raise ValueError(f"Input CSV {input_csv} contains an empty identity.")
        if sdss_name in by_sdss_name:
            raise ValueError(
                f"Input CSV {input_csv} contains duplicate SDSS name {sdss_name!r}."
            )
        row = dict(row)
        row["sdss_name"] = sdss_name
        row["object_id"] = object_id
        by_sdss_name[sdss_name] = row

    bundle_by_name: dict[str, Path] = {}
    for bundle_path in sorted(source_bundle_dir.glob(f"*{BUNDLE_SUFFIX}")):
        sdss_name = bundle_sdss_name(bundle_path)
        if sdss_name in bundle_by_name:
            raise ValueError(
                f"Duplicate posterior bundles for SDSS name {sdss_name!r}."
            )
        if sdss_name not in by_sdss_name:
            raise ValueError(
                f"Posterior bundle {bundle_path} has no row in {input_csv}."
            )
        bundle_by_name[sdss_name] = bundle_path.resolve()
    if not bundle_by_name:
        raise RuntimeError(f"No *{BUNDLE_SUFFIX} files found in {source_bundle_dir}.")

    available: list[ResumeObject] = []
    missing: list[dict[str, str]] = []
    for row in rows:
        sdss_name = normalize_sdss_name(row["sdss_name"])
        bundle_path = bundle_by_name.get(sdss_name)
        if bundle_path is None:
            missing.append(
                {
                    "object_id": normalize_object_id(row["object_id"]),
                    "sdss_name": sdss_name,
                }
            )
            continue
        available.append(
            ResumeObject(
                ordinal=len(available),
                object_id=normalize_object_id(row["object_id"]),
                sdss_name=sdss_name,
                bundle_path=bundle_path,
            )
        )
    if len({item.object_id for item in available}) != len(available):
        raise ValueError("Downloaded bundles do not map to unique object IDs.")
    return available, missing


def chunked(values: list[ResumeObject], size: int) -> list[list[ResumeObject]]:
    if size <= 0:
        raise ValueError("Batch size must be positive.")
    return [values[index : index + size] for index in range(0, len(values), size)]


def source_manifest_hash(objects: list[ResumeObject]) -> str:
    digest = hashlib.sha256()
    for item in objects:
        stat = item.bundle_path.stat()
        digest.update(
            (
                f"{item.object_id}\t{item.sdss_name}\t{item.bundle_path.name}\t"
                f"{stat.st_size}\t{stat.st_mtime_ns}\n"
            ).encode()
        )
    return digest.hexdigest()


def batch_shard_name(batch: list[ResumeObject]) -> str:
    identity = ",".join(item.object_id for item in batch)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    return (
        f"resume_{batch[0].ordinal:05d}_{len(batch):03d}_"
        f"{batch[0].object_id}_{digest}.h5"
    )


def source_catalog_shards(source_run: Path) -> list[Path]:
    """Return only original per-job v2 catalogs, never posterior bundles."""
    paths = sorted(source_run.glob(f"{source_run.name}_chunk*.h5"))
    if not paths:
        print(
            "No original source catalog shards were found; every object will "
            "use strict JAX resume fallback."
        )
    return [path.resolve() for path in paths]


def load_direct_v3_sources(
    source_run: Path,
    *,
    selection_seed: int = 3,
) -> tuple[list[dict[str, object]], list[Path]]:
    """Load bundle-backed rows from the run's original chunk catalogs.

    The chunks define scalar metadata, row order, and PSF-fraction draws.  A
    row is selected exactly when ``source_run/all`` contains its posterior
    bundle.  This intentionally ignores the chunks' false ``fit_ok`` values:
    the Aug 24 jobs saved valid scientific products before a missing plotting
    method raised and marked those rows failed.
    """

    import h5py
    import numpy as np

    from qvc.spectra.catalog_hdf5 import (
        SPECTRA_CATALOG_FORMAT_V2,
        read_spectra_catalog_hdf5,
    )

    source_run, bundle_dir = resolve_source_bundle_dir(source_run)
    chunk_paths = source_catalog_shards(source_run)
    if not chunk_paths:
        raise RuntimeError(f"No source chunk catalogs found in {source_run}.")

    bundle_by_name: dict[str, Path] = {}
    for path in sorted(bundle_dir.glob(f"*{BUNDLE_SUFFIX}")):
        sdss_name = bundle_sdss_name(path)
        if sdss_name in bundle_by_name:
            raise ValueError(f"Duplicate posterior bundles for {sdss_name!r}.")
        bundle_by_name[sdss_name] = path.resolve()
    if not bundle_by_name:
        raise RuntimeError(f"No posterior bundles found in {bundle_dir}.")

    seen_object_ids: set[str] = set()
    seen_sdss_names: set[str] = set()
    selected: list[dict[str, object]] = []
    for chunk_path in chunk_paths:
        with h5py.File(chunk_path, "r") as handle:
            selection = str(
                handle.attrs.get("f_host_2500_psf_draw_selection", "")
            )
            if selection != "deterministic_uniform_without_replacement":
                raise ValueError(
                    f"Source chunk {chunk_path} has unsupported host-draw "
                    f"selection {selection!r}."
                )
            provenance_text = handle.attrs.get("qvc_provenance_json", "")
            if isinstance(provenance_text, bytes):
                provenance_text = provenance_text.decode("utf-8")
            provenance = json.loads(str(provenance_text))
            chunk_seed = int(
                provenance["module"]["parsed_args"]["seed"]
            )
            if chunk_seed != int(selection_seed):
                raise ValueError(
                    f"Source chunk {chunk_path} used selection seed "
                    f"{chunk_seed}, not requested seed {selection_seed}."
                )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Loading legacy spectra catalog .*",
                category=RuntimeWarning,
            )
            catalog = read_spectra_catalog_hdf5(chunk_path, allow_v2=True)
        if catalog.catalog_format != SPECTRA_CATALOG_FORMAT_V2:
            raise ValueError(
                f"Source chunk {chunk_path} has format "
                f"{catalog.catalog_format!r}, not "
                f"{SPECTRA_CATALOG_FORMAT_V2!r}."
            )
        required = {"object_id", "sdss_name", "z"}
        missing = required - set(catalog.frame.columns)
        if missing:
            raise ValueError(
                f"Source chunk {chunk_path} lacks columns {sorted(missing)}."
            )
        for row_index, source_row in catalog.frame.iterrows():
            row = source_row.to_dict()
            object_id = normalize_object_id(row["object_id"])
            sdss_name = normalize_sdss_name(row["sdss_name"])
            if object_id in seen_object_ids:
                raise ValueError(f"Duplicate chunk object_id {object_id!r}.")
            if sdss_name in seen_sdss_names:
                raise ValueError(f"Duplicate chunk SDSS name {sdss_name!r}.")
            seen_object_ids.add(object_id)
            seen_sdss_names.add(sdss_name)
            bundle_path = bundle_by_name.pop(sdss_name, None)
            if bundle_path is None:
                continue
            with h5py.File(bundle_path, "r") as bundle_handle:
                raw = bundle_handle.attrs.get("qvc_provenance_json", "")
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                try:
                    bundle_provenance = json.loads(str(raw))
                    bundle_jaxsedfit_commit = (
                        bundle_provenance["runtime"]["dependencies"]
                        ["JAXSEDFit"]["git"]["commit"]
                    )
                except (KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Posterior bundle lacks JAXSedFit provenance: {bundle_path}"
                    ) from exc
                if bundle_jaxsedfit_commit != EXPECTED_JAXSEDFIT_COMMIT:
                    raise ValueError(
                        f"Posterior bundle {bundle_path} records JAXSedFit "
                        f"{bundle_jaxsedfit_commit!r}; expected "
                        f"{EXPECTED_JAXSEDFIT_COMMIT!r}."
                    )
            fraction_count = int(catalog.valid_count[row_index])
            if fraction_count <= 0:
                raise ValueError(
                    f"Bundle-backed chunk row {object_id} has no PSF draws."
                )
            fraction_draws = np.asarray(
                catalog.fraction_draws[row_index], dtype=np.float32
            ).copy()
            host_count = int(catalog.f_host_2500_psf_valid_count[row_index])
            if host_count <= 0:
                raise ValueError(
                    f"Bundle-backed chunk row {object_id} has no host draws."
                )
            host_draws = np.asarray(
                catalog.f_host_2500_psf_draws[row_index], dtype=np.float32
            ).copy()
            selected.append(
                {
                    "object_id": object_id,
                    "sdss_name": sdss_name,
                    "bundle_path": bundle_path,
                    "source_chunk_path": chunk_path,
                    "row": row,
                    "psf_fraction_draws": fraction_draws,
                    "psf_fraction_valid_count": fraction_count,
                    "host_fraction_draws": host_draws,
                    "host_fraction_valid_count": host_count,
                }
            )

    if bundle_by_name:
        examples = sorted(bundle_by_name)[:10]
        raise ValueError(
            f"{len(bundle_by_name):,} posterior bundle(s) have no chunk row; "
            f"first unmatched SDSS names: {examples}."
        )
    if not selected:
        raise RuntimeError("No chunk rows have matching posterior bundles.")
    return selected, chunk_paths


def source_catalog_manifest_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(
            f"{path.name}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode()
        )
    return digest.hexdigest()


def source_catalog_row_count(paths: list[Path]) -> int:
    """Count source rows from HDF5 shapes without loading catalog columns."""
    import h5py

    total = 0
    for path in paths:
        with h5py.File(path, "r") as handle:
            catalog = handle["catalog"]
            first = next(iter(catalog.values()), None)
            total += 0 if first is None else int(first.shape[0])
    return total


def contiguous_repair_runs(
    rows: list[RepairedCatalogRow],
) -> list[list[RepairedCatalogRow]]:
    """Keep repaired shard rows in global input order across fallback gaps."""
    ordered = sorted(rows, key=lambda value: value.item.ordinal)
    runs: list[list[RepairedCatalogRow]] = []
    for value in ordered:
        if not runs or value.item.ordinal != runs[-1][-1].item.ordinal + 1:
            runs.append([value])
        else:
            runs[-1].append(value)
    return runs


def posterior_source_draw_count(samples: dict[str, object]) -> int:
    """Return and validate the common leading posterior axis."""

    import numpy as np

    if "log_agn_amp" not in samples:
        raise ValueError("Posterior bundle lacks required log_agn_amp draws.")
    anchor = np.asarray(samples["log_agn_amp"])
    if anchor.ndim < 1 or anchor.shape[0] < 1:
        raise ValueError("Posterior bundle has no saved posterior draws.")
    count = int(anchor.shape[0])
    incompatible = {
        name: np.asarray(values).shape
        for name, values in samples.items()
        if np.asarray(values).ndim > 0 and np.asarray(values).shape[0] != count
    }
    if incompatible:
        raise ValueError(
            "Posterior sites do not share one leading draw axis: "
            f"{incompatible}."
        )
    return count


def subset_posterior_samples(
    samples: dict[str, object], posterior_index: object
) -> dict[str, object]:
    """Select explicit original draw indices from every posterior site."""

    import numpy as np

    count = posterior_source_draw_count(samples)
    index = np.asarray(posterior_index, dtype=np.int64)
    if index.ndim != 1 or index.size < 1:
        raise ValueError("posterior_index must be a nonempty one-dimensional array.")
    if np.any(index < 0) or np.any(index >= count):
        raise ValueError("posterior_index is outside the saved posterior axis.")
    if np.any(np.diff(index) <= 0):
        raise ValueError("posterior_index must be strictly increasing and unique.")
    return {
        name: np.asarray(values)[index]
        for name, values in samples.items()
    }


def resolve_v3_bundle_path(
    row: dict[str, object], posterior_bundle_dir: Path | None
) -> Path:
    """Resolve a local immutable bundle without trusting stale HPC paths."""

    direct = Path(str(row.get("fit_result_path", ""))).expanduser()
    if direct.is_file():
        return direct.resolve()
    sdss_name = normalize_sdss_name(row.get("sdss_name", ""))
    if not sdss_name:
        raise ValueError("Source catalog row has no SDSS name.")
    if posterior_bundle_dir is None:
        raise FileNotFoundError(
            f"Posterior bundle path is unavailable for {sdss_name!r}: {direct}"
        )
    matches = sorted(
        posterior_bundle_dir.glob(f"*_{sdss_name}{BUNDLE_SUFFIX}")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one posterior bundle for {sdss_name!r} in "
            f"{posterior_bundle_dir}; found {len(matches)}."
        )
    return matches[0].resolve()


def build_v3_row(
    task: V3BuildTask, samples: dict[str, object]
) -> dict[str, object]:
    """Build one v3 row analytically from bundle samples and chunk draws."""

    import numpy as np

    from qvc.spectra.catalog_hdf5 import (
        JOINT_POSTERIOR_DRAW_COUNT,
        JOINT_POSTERIOR_DRAW_FIELDS,
    )
    from qvc.spectra.fit_spectra_jaxsedfit_joint import (
        deterministic_compact_posterior_indices,
        estimate_joint_hubble_posterior_draws,
        summarize_host_2500_psf,
        summarize_joint_hubble_posterior_draws,
    )

    full_samples = {
        name: np.asarray(values) for name, values in samples.items()
    }
    source_draw_count = posterior_source_draw_count(full_samples)
    selected_index = deterministic_compact_posterior_indices(
        source_draw_count,
        object_id=task.object_id,
        seed=task.seed,
        draw_count=JOINT_POSTERIOR_DRAW_COUNT,
    )
    full_physical_samples = dict(full_samples)
    full_physical_samples["ebv_gal"] = _physical_dust_draws(
        full_samples, "ebv_gal"
    )
    full_physical_samples["ebv_agn"] = _physical_dust_draws(
        full_samples, "ebv_agn"
    )
    full_derived = estimate_joint_hubble_posterior_draws(
        full_physical_samples,
        float(task.row["z"]),
    )
    valid_count = int(len(selected_index))
    if int(task.host_fraction_valid_count) != valid_count:
        raise ValueError(
            "Chunk host draws do not match deterministic v3 selection count."
        )
    host_draws = np.asarray(task.host_fraction_draws, dtype=np.float32)
    selected_host = host_draws[:valid_count]
    if not np.all(np.isfinite(selected_host)):
        raise ValueError("Chunk host draws are nonfinite within valid_count.")
    compact = {
        name: np.full(JOINT_POSTERIOR_DRAW_COUNT, np.nan, dtype=np.float32)
        for name in JOINT_POSTERIOR_DRAW_FIELDS
    }
    compact["f_host_2500_psf"][:valid_count] = selected_host
    for name in JOINT_POSTERIOR_DRAW_FIELDS:
        if name == "f_host_2500_psf":
            continue
        values = np.asarray(full_derived[f"{name}_draws"], dtype=float)
        compact[name][:valid_count] = values[selected_index].astype(np.float32)

    posterior_index = np.full(
        JOINT_POSTERIOR_DRAW_COUNT, -1, dtype=np.int32
    )
    posterior_index[:valid_count] = selected_index
    # Keep restart shards compact. The atomic final merge overlays these
    # authoritative v3 products onto the complete scalar row from the original
    # chunk, so hundreds of unchanged columns need not be repeated per object.
    result = {
        name: task.row[name]
        for name in ("object_id", "sdss_name", "z")
        if name in task.row
    }
    selected_host_summary = summarize_host_2500_psf(
        {"component_host_fraction": selected_host[:, None]}
    )
    result.update(selected_host_summary)
    # The chunk scalar median/symmetric error were summarized from the full
    # host posterior before compaction. Preserve those two authoritative values
    # when present; the selected draws supply the aligned v3 payload.
    for name in ("f_host_2500_psf", "f_host_2500_psf_err"):
        try:
            source_value = float(task.row.get(name, np.nan))
        except (TypeError, ValueError):
            source_value = np.nan
        if np.isfinite(source_value):
            result[name] = source_value
    result.update(summarize_joint_hubble_posterior_draws(full_derived))
    result.update(
        {
            "fit_ok": True,
            "error_message": "",
            "execution_mode": "resumed",
            "fit_result_path": str(task.bundle_path),
            "resumed_from_path": str(task.bundle_path),
            "joint_posterior_draw_source": (
                "chunk_host_plus_bundle_analytic_selected64"
            ),
            "catalog_build_mode": "chunks_and_bundles_analytic_to_v3",
            "catalog_build_source_chunk": str(task.source_chunk_path),
            "catalog_build_original_posterior_draw_count": source_draw_count,
            "catalog_build_analytic_posterior_draw_count": valid_count,
            "catalog_build_f_host_scalar_source": (
                "original_chunk_full_posterior_summary"
            ),
            "mw_deredden_applied": bool(
                task.row.get("mw_deredden_applied", True)
            ),
            "_psf_agn_fraction_draws": np.asarray(
                task.psf_fraction_draws, dtype=np.float32
            ),
            "_psf_agn_fraction_valid_count": int(
                task.psf_fraction_valid_count
            ),
            "_joint_posterior_draws": compact,
            "_joint_posterior_valid_count": valid_count,
            "_joint_posterior_index": posterior_index,
            "_joint_posterior_source_draw_count": source_draw_count,
            "_joint_posterior_selection_seed": int(task.seed),
        }
    )
    return result


def build_v3_color_row(task: V3BuildTask) -> dict[str, object]:
    """Copy/derive one row and predict only its selected-64 ugriz fluxes."""

    import h5py
    import numpy as np

    from qvc.spectra.fit_spectra_jaxsedfit_joint import (
        POSTERIOR_BUNDLE_FORMAT,
        QVC_PSF_HOST_CAPTURE_GROUP,
        joint_psf_photometry_prediction_provenance,
    )
    from jaxsedfit import JAXSEDFit

    bundle_identity = task.bundle_path.name[: -len(BUNDLE_SUFFIX)]
    if not bundle_identity.endswith(f"_{task.sdss_name}"):
        raise ValueError(
            f"Posterior bundle {task.bundle_path} does not match SDSS name "
            f"{task.sdss_name!r}."
        )
    with h5py.File(task.bundle_path, "r") as handle:
        bundle_format = handle.attrs.get("posterior_bundle_format", "")
        capture_group = handle.attrs.get("qvc_host_capture_group", "")
        if isinstance(bundle_format, bytes):
            bundle_format = bundle_format.decode("utf-8")
        if isinstance(capture_group, bytes):
            capture_group = capture_group.decode("utf-8")
        if bundle_format != POSTERIOR_BUNDLE_FORMAT:
            raise ValueError(f"Bundle format is {bundle_format!r}.")
        if capture_group != QVC_PSF_HOST_CAPTURE_GROUP:
            raise ValueError(f"Bundle host-capture group is {capture_group!r}.")
        if "samples" not in handle:
            raise ValueError("Posterior bundle lacks samples.")

    # Loading the immutable bundle is the only model reconstruction. Reuse the
    # same in-memory samples for the analytic v3 products and the one required
    # photometry prediction; never run inference or a plot/full prediction.
    fitter = JAXSEDFit.load(task.bundle_path)
    samples = {name: np.asarray(value) for name, value in fitter.samples.items()}
    row = build_v3_row(task, samples)
    count = int(row["_joint_posterior_valid_count"])
    indices = np.asarray(row["_joint_posterior_index"][:count], dtype=np.int64)
    fitter.samples = subset_posterior_samples(samples, indices)
    fitter.predictive = None
    fitter._predictive_return_sites = lambda kind, **kwargs: ["pred_fluxes"]
    prediction = fitter.predict(kind="photometry")
    total = np.asarray(prediction.get("pred_fluxes"), dtype=float)
    filter_names = [str(value) for value in fitter.config.photometry.filter_names]
    wanted = [f"{band}_sdss" for band in "ugriz"]
    positions = []
    for name in wanted:
        matches = [i for i, value in enumerate(filter_names) if value == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one fitted {name!r}; found {len(matches)}.")
        positions.append(matches[0])
    selected_fluxes = total[:, positions]
    if selected_fluxes.shape != (count, 5):
        raise ValueError(
            f"Selected fitted ugriz fluxes have shape {selected_fluxes.shape}; "
            f"expected {(count, 5)}."
        )
    if not np.all(np.isfinite(selected_fluxes)) or np.any(selected_fluxes <= 0):
        raise ValueError("Selected fitted ugriz fluxes must be finite and positive.")
    compact_fluxes = np.full((64, 5), np.nan, dtype=np.float32)
    compact_fluxes[:count] = selected_fluxes.astype(np.float32)
    row["_joint_psf_photometry_draws"] = compact_fluxes
    row["_joint_psf_photometry_provenance"] = (
        joint_psf_photometry_prediction_provenance(
            "saved_posterior_bundle_selected64_prediction"
        )
    )
    return row


def process_v3_task(task: V3BuildTask) -> str:
    """Write one backwards-compatible single-object restart shard."""
    from qvc.spectra.fit_spectra_jaxsedfit_joint import write_joint_fit_results_hdf5

    row = build_v3_color_row(task)
    write_joint_fit_results_hdf5(task.shard_path, [row])
    return str(task.shard_path)


def process_v3_chunk(tasks: tuple[V3BuildTask, ...], shard_path: str) -> str:
    """Write one source-chunk-sized restart shard (normally four objects)."""
    from qvc.spectra.fit_spectra_jaxsedfit_joint import write_joint_fit_results_hdf5

    rows = [build_v3_color_row(task) for task in tasks]
    write_joint_fit_results_hdf5(Path(shard_path), rows)
    return str(shard_path)


def validate_v3_catalog(
    path: Path,
    object_ids: list[str],
    *,
    selection_seed: int,
) -> None:
    """Strongly validate a completed v3 shard or merged catalog."""

    import numpy as np

    from qvc.spectra.catalog_hdf5 import (
        JOINT_POSTERIOR_DRAW_FIELDS,
        SPECTRA_CATALOG_FORMAT,
        read_spectra_catalog_hdf5,
    )

    catalog = read_spectra_catalog_hdf5(path)
    validate_loaded_v3_catalog(
        catalog,
        path=path,
        object_ids=object_ids,
        selection_seed=selection_seed,
    )


def validate_loaded_v3_catalog(
    catalog,
    *,
    path: Path,
    object_ids: list[str],
    selection_seed: int,
) -> None:
    """Validate an already loaded v3 catalog without reopening its HDF5 file."""

    import numpy as np

    from qvc.spectra.catalog_hdf5 import (
        JOINT_POSTERIOR_DRAW_FIELDS,
        SPECTRA_CATALOG_FORMAT,
    )

    if catalog.catalog_format != SPECTRA_CATALOG_FORMAT:
        raise ValueError(f"Upgrade output is not v3: {path}")
    actual = [normalize_object_id(value) for value in catalog.frame["object_id"]]
    if actual != object_ids:
        raise ValueError(
            f"Upgrade catalog {path} has object IDs {actual}; expected {object_ids}."
        )
    if catalog.joint_posterior_selection_seed != int(selection_seed):
        raise ValueError(f"Upgrade catalog {path} has the wrong selection seed.")
    if set(catalog.joint_posterior_draws) != set(JOINT_POSTERIOR_DRAW_FIELDS):
        raise ValueError(f"Upgrade catalog {path} has incomplete joint draws.")
    if not np.all(catalog.joint_posterior_valid_count > 0):
        raise ValueError(f"Upgrade catalog {path} has empty joint draws.")
    if not np.all(catalog.valid_count > 0):
        raise ValueError(f"Upgrade catalog {path} has empty ugriz PSF draws.")
    if catalog.joint_psf_photometry_draws is None:
        raise ValueError(f"Upgrade catalog {path} lacks fitted ugriz photometry.")


def merge_v3_shards(
    shard_paths: list[Path],
    output_path: Path,
    *,
    expected_object_ids: list[str],
    selection_seed: int,
    base_rows: list[dict[str, object]],
    source_run: Path,
    source_chunks: list[Path],
    args: argparse.Namespace,
) -> None:
    """Overlay v3 shard products on chunk rows and atomically write v3."""

    import numpy as np
    import pandas as pd

    from qvc.provenance import build_run_record
    from qvc.spectra.catalog_hdf5 import (
        JOINT_POSTERIOR_DRAW_FIELDS,
        read_spectra_catalog_hdf5,
        write_spectra_catalog_hdf5,
    )

    row_count = len(expected_object_ids)
    fraction_draws = np.empty((row_count, 64, 5), dtype=np.float32)
    valid_count = np.empty(row_count, dtype=np.int16)
    joint_posterior_draws = {
        name: np.empty((row_count, 64), dtype=np.float32)
        for name in JOINT_POSTERIOR_DRAW_FIELDS
    }
    joint_posterior_valid_count = np.empty(row_count, dtype=np.int16)
    joint_posterior_index = np.empty((row_count, 64), dtype=np.int32)
    joint_posterior_source_draw_count = np.empty(row_count, dtype=np.int32)
    joint_psf_photometry_draws = np.empty(
        (row_count, 64, 5), dtype=np.float32
    )
    derived_frames = []
    joint_psf_photometry_provenance = None
    offset = 0
    for path in shard_paths:
        catalog = read_spectra_catalog_hdf5(path)
        count = len(catalog.frame)
        shard_object_ids = expected_object_ids[offset : offset + count]
        validate_loaded_v3_catalog(
            catalog,
            path=path,
            object_ids=shard_object_ids,
            selection_seed=selection_seed,
        )
        stop = offset + count
        destination = slice(offset, stop)
        derived_frames.append(catalog.frame)
        fraction_draws[destination] = catalog.fraction_draws
        valid_count[destination] = catalog.valid_count
        for name in JOINT_POSTERIOR_DRAW_FIELDS:
            joint_posterior_draws[name][destination] = (
                catalog.joint_posterior_draws[name]
            )
        joint_posterior_valid_count[destination] = (
            catalog.joint_posterior_valid_count
        )
        joint_posterior_index[destination] = catalog.joint_posterior_index
        joint_posterior_source_draw_count[destination] = (
            catalog.joint_posterior_source_draw_count
        )
        joint_psf_photometry_draws[destination] = (
            catalog.joint_psf_photometry_draws
        )
        provenance = dict(catalog.joint_psf_photometry_provenance)
        if joint_psf_photometry_provenance is None:
            joint_psf_photometry_provenance = provenance
        elif provenance != joint_psf_photometry_provenance:
            raise ValueError(
                f"Upgrade catalog {path} has inconsistent fitted-photometry "
                "provenance."
            )
        offset += count
        # The final arrays above are fixed-size. Release each per-shard catalog
        # immediately instead of retaining thousands of HDF5 payloads.
        del catalog
    if offset != len(expected_object_ids):
        raise ValueError(
            f"Restart shards contain {offset} rows; expected "
            f"{len(expected_object_ids)}."
        )
    derived_frame = pd.concat(derived_frames, ignore_index=True, sort=False)
    del derived_frames
    frame = pd.DataFrame(base_rows).reset_index(drop=True)
    source_ids = [normalize_object_id(value) for value in frame["object_id"]]
    if source_ids != expected_object_ids:
        raise ValueError("Chunk row order changed before the atomic v3 merge.")
    for column in derived_frame.columns:
        frame[column] = derived_frame[column].to_numpy()
    provenance = build_run_record(
        str(getattr(args, "provenance_entrypoint", "run_resume_spectra_local.py")),
        args,
        argv=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        input_paths={"source_run": source_run},
        event_type="chunks_and_bundles_to_v3",
    )
    provenance["catalog_build"] = {
        "source": "original_chunk_catalogs_and_posterior_bundles",
        "target_format": "qvc_spectra_catalog_v3",
        "inference_rerun": False,
        "plots_generated": False,
        "prediction_scope": "64_selected_original_posterior_indices_only",
        "joint_posterior_selection_seed": int(selection_seed),
        "source_run": str(source_run),
        "source_chunk_count": len(source_chunks),
        "source_target_count": source_catalog_row_count(source_chunks),
        "omitted_target_count": (
            source_catalog_row_count(source_chunks) - len(expected_object_ids)
        ),
        "source_chunk_manifest_sha256": source_catalog_manifest_hash(
            source_chunks
        ),
        "resumable_shard_count": len(shard_paths),
        "resumable_shard_manifest_sha256": file_set_hash(shard_paths),
    }
    write_spectra_catalog_hdf5(
        output_path,
        frame,
        fraction_draws,
        valid_count,
        joint_posterior_draws=joint_posterior_draws,
        joint_posterior_valid_count=joint_posterior_valid_count,
        joint_posterior_index=joint_posterior_index,
        joint_posterior_source_draw_count=joint_posterior_source_draw_count,
        joint_posterior_selection_seed=selection_seed,
        joint_psf_photometry_draws=joint_psf_photometry_draws,
        joint_psf_photometry_provenance=joint_psf_photometry_provenance,
        provenance=provenance,
    )
    validate_v3_catalog(
        output_path,
        expected_object_ids,
        selection_seed=selection_seed,
    )


def _physical_dust_draws(samples: dict[str, object], name: str):
    import numpy as np

    if name in samples:
        return np.asarray(samples[name], dtype=float)
    log_name = f"log_{name}"
    if log_name not in samples:
        raise ValueError(f"Posterior bundle lacks {name!r} and {log_name!r}.")
    return np.exp(np.asarray(samples[log_name], dtype=float))


def repair_source_catalog_row(
    item: ResumeObject,
    source_catalog,
    row_index: int,
    *,
    source_catalog_path: Path,
    source_catalog_commit: str,
    source_run: Path,
) -> RepairedCatalogRow:
    """Repair m2500 exactly while preserving existing predictive products."""
    import h5py
    import numpy as np

    from qvc.spectra.fit_spectra_jaxsedfit_joint import (
        POSTERIOR_BUNDLE_FORMAT,
        QVC_PSF_HOST_CAPTURE_GROUP,
        estimate_m2500_dereddened,
        summarize_m2500_dereddened,
    )
    from qvc.mcmc_diagnostics import compute_numpyro_summary

    row = source_catalog.frame.iloc[row_index].copy()
    object_id = normalize_object_id(row.get("object_id", ""))
    if object_id != item.object_id:
        raise ValueError(
            f"source row object_id={object_id!r} does not match {item.object_id!r}"
        )
    if normalize_sdss_name(row.get("sdss_name", "")) != item.sdss_name:
        raise ValueError("source row has a mismatched SDSS name")
    fraction_count = int(source_catalog.valid_count[row_index])
    host_count = int(source_catalog.f_host_2500_psf_valid_count[row_index])
    if fraction_count <= 0 or host_count <= 0:
        raise ValueError("source catalog row lacks compact posterior draws")
    required_source_scalars = (
        "z",
        "m_2500_dereddened",
        "m_2500_attenuated_model",
        "a_2500_total",
        "f_host_2500_psf",
        "f_host_2500_psf_err",
        "joint_reduced_chi2",
        "f_AGN_psf_u",
        "f_AGN_psf_g",
        "f_AGN_psf_r",
        "f_AGN_psf_i",
        "f_AGN_psf_z",
    )
    missing_scalars = [name for name in required_source_scalars if name not in row]
    if missing_scalars:
        raise ValueError(f"source catalog row lacks fields {missing_scalars}")
    if not np.all(
        np.isfinite([float(row[name]) for name in required_source_scalars])
    ):
        raise ValueError("source catalog row has nonfinite required summaries")

    with h5py.File(item.bundle_path, "r") as handle:
        bundle_format = handle.attrs.get("posterior_bundle_format")
        bundle_group = handle.attrs.get("qvc_host_capture_group")
        bundle_commit = handle.attrs.get("qvc_git_commit", "")
        for name, value in (
            ("posterior_bundle_format", bundle_format),
            ("qvc_host_capture_group", bundle_group),
            ("qvc_git_commit", bundle_commit),
        ):
            if isinstance(value, bytes):
                if name == "posterior_bundle_format":
                    bundle_format = value.decode("utf-8")
                elif name == "qvc_host_capture_group":
                    bundle_group = value.decode("utf-8")
                else:
                    bundle_commit = value.decode("utf-8")
        if bundle_format != POSTERIOR_BUNDLE_FORMAT:
            raise ValueError(f"bundle format is {bundle_format!r}")
        if bundle_group not in (None, "", QVC_PSF_HOST_CAPTURE_GROUP):
            raise ValueError(f"bundle host-capture group is {bundle_group!r}")
        if not source_catalog_commit or bundle_commit != source_catalog_commit:
            raise ValueError(
                "source catalog and posterior bundle record different commits: "
                f"{source_catalog_commit!r} != {bundle_commit!r}"
            )
        if "samples" not in handle:
            raise ValueError("posterior bundle lacks samples")
        samples = {
            name: np.asarray(dataset)
            for name, dataset in handle["samples"].items()
        }
        host_group_draws = samples.get("host_capture_group_fraction")
        if (
            host_group_draws is None
            or np.asarray(host_group_draws).ndim != 2
            or np.asarray(host_group_draws).shape[1] != 1
        ):
            raise ValueError(
                "posterior bundle lacks one shared host-capture group parameter"
            )

    samples["ebv_gal"] = _physical_dust_draws(samples, "ebv_gal")
    samples["ebv_agn"] = _physical_dust_draws(samples, "ebv_agn")
    redshift = float(row["z"])
    corrected = summarize_m2500_dereddened(samples, redshift)
    corrected_draws = estimate_m2500_dereddened(samples, redshift)

    # This fast path is intentionally specific to the pre-5ad45da products.
    # Prove both assumptions per object before preserving any source values.
    old_dereddened = float(row["m_2500_dereddened"])
    old_attenuation = float(row["a_2500_total"])
    if not np.isclose(
        old_dereddened,
        corrected["m_2500_dereddened"],
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("source intrinsic m2500 does not match latent draws")
    if not np.isclose(
        1.2 * old_attenuation,
        corrected["a_2500_total"],
        rtol=1e-9,
        atol=1e-10,
    ):
        raise ValueError("source attenuation is not the expected 1.0-normalized value")

    row.update(corrected)
    attenuated_draws = np.asarray(
        corrected_draws["m_2500_attenuated_model_draws"], dtype=float
    )
    attenuation_summary = compute_numpyro_summary(
        {"m_2500_attenuated_model": attenuated_draws.reshape(1, -1)},
        group_by_chain=True,
        prob=0.90,
    )
    attenuation_stats = attenuation_summary.get("m_2500_attenuated_model", {})
    attenuation_rhat = np.asarray(
        attenuation_stats.get("r_hat", np.nan), dtype=float
    )
    if attenuation_rhat.size != 1 or not np.isfinite(attenuation_rhat).all():
        raise ValueError("corrected attenuated m2500 R-hat is unavailable")
    row["m_2500_attenuated_model_rhat"] = float(
        attenuation_rhat.reshape(-1)[0]
    )
    # Assert the summaries came from the same exact draws used above.
    if not np.isclose(
        np.median(corrected_draws["m_2500_attenuated_model_draws"]),
        row["m_2500_attenuated_model"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("corrected attenuated m2500 summary is inconsistent")

    row["catalog_repair_source_fit_ok"] = bool(row.get("fit_ok", False))
    row["catalog_repair_source_error_message"] = str(
        row.get("error_message", "")
    )
    row["fit_ok"] = True
    row["error_message"] = ""
    row["execution_mode"] = "resumed"
    row["resumed_from_run"] = source_run.name
    row["resumed_from_path"] = str(item.bundle_path)
    row["resume_error_message"] = ""
    row["fit_result_path"] = str(item.bundle_path)
    row["sed_fig_path"] = ""
    row["spectrum_fig_path"] = ""
    row["catalog_repair_mode"] = "analytic_m2500_norm12"
    row["catalog_repair_source_catalog"] = str(source_catalog_path)
    row["catalog_repair_source_commit"] = str(bundle_commit)

    return RepairedCatalogRow(
        item=item,
        row=row,
        fraction_draws=np.asarray(
            source_catalog.fraction_draws[row_index], dtype=np.float32
        ).copy(),
        fraction_valid_count=fraction_count,
        host_draws=np.asarray(
            source_catalog.f_host_2500_psf_draws[row_index], dtype=np.float32
        ).copy(),
        host_valid_count=host_count,
        source_catalog=source_catalog_path,
        source_catalog_commit=source_catalog_commit,
    )


def write_repaired_catalog_shard(
    path: Path,
    repaired: list[RepairedCatalogRow],
    *,
    args: argparse.Namespace,
    source_catalog_path: Path,
) -> None:
    import numpy as np
    import pandas as pd

    from qvc.provenance import build_run_record
    from qvc.spectra.catalog_hdf5 import write_spectra_catalog_hdf5

    provenance = build_run_record(
        "run_resume_spectra_local.py",
        args,
        argv=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        input_paths={
            "source_catalog": source_catalog_path,
            **{
                f"posterior_bundle_{index}": value.item.bundle_path
                for index, value in enumerate(repaired)
            },
        },
        event_type="analytic_catalog_repair",
    )
    provenance["catalog_repair"] = {
        "mode": "analytic_m2500_norm12",
        "source_catalog": str(source_catalog_path),
        "source_catalog_commits": sorted(
            {value.source_catalog_commit for value in repaired}
        ),
        "preserved_predictive_products": [
            "psf_agn_fraction_draws",
            "f_host_2500_psf_draws",
        ],
        "recomputed_products": [
            "m_2500_dereddened",
            "m_2500_attenuated_model",
            "a_2500_galaxy",
            "a_2500_internal",
            "a_2500_total",
            "m_2500_attenuated_model_rhat",
        ],
        "attenuation_normalization": 1.2,
    }
    write_spectra_catalog_hdf5(
        path,
        pd.DataFrame([value.row for value in repaired]),
        np.stack([value.fraction_draws for value in repaired]),
        np.asarray(
            [value.fraction_valid_count for value in repaired], dtype=np.int16
        ),
        f_host_2500_psf_draws=np.stack(
            [value.host_draws for value in repaired]
        ),
        f_host_2500_psf_valid_count=np.asarray(
            [value.host_valid_count for value in repaired], dtype=np.int16
        ),
        provenance=provenance,
    )


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def module_git_state(module_name: str) -> tuple[str, bool, str]:
    """Resolve an installed source module's exact Git revision."""
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return "", False, ""
    origin = Path(spec.origin).resolve()
    try:
        root = subprocess.run(
            ["git", "-C", str(origin.parent), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return "", False, str(origin)
    return commit, dirty, root


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_environment(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
    )
    env.setdefault("JAX_ENABLE_X64", "True")
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    env.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.setdefault("MPLCONFIGDIR", f"/tmp/qvc-spectra-resume-mpl-{os.getuid()}")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def reexec_with_fit_python(args: argparse.Namespace, repo_root: Path) -> None:
    python_bin = Path(args.python_bin).expanduser().resolve()
    if not python_bin.is_file():
        raise FileNotFoundError(f"Fit Python does not exist: {python_bin}")
    if Path(sys.executable).resolve() == python_bin:
        return
    if os.environ.get(REEXEC_ENV) == "1":
        raise RuntimeError(
            f"Could not re-execute this driver with requested Python {python_bin}; "
            f"still running {sys.executable}."
        )
    env = runtime_environment(repo_root)
    env[REEXEC_ENV] = "1"
    os.execve(
        python_bin,
        [str(python_bin), str(Path(__file__).resolve()), *sys.argv[1:]],
        env,
    )


def probe_resume_environment(
    python_bin: Path,
    repo_root: Path,
    sample_bundle: Path,
    env: dict[str, str],
) -> None:
    code = """
import json
import jaxsedfit
from jaxsedfit import JAXSEDFit
from qvc.spectra.fit_spectra_jaxsedfit_joint import validate_resume_host_capture_fitter
fitter = JAXSEDFit.load(__import__('sys').argv[1])
validate_resume_host_capture_fitter(fitter, __import__('sys').argv[1])
print('__QVC_RESUME_PROBE__' + json.dumps({
    'jaxsedfit': jaxsedfit.__file__,
    'observation': str(fitter.config.observation.object_id),
}))
"""
    result = subprocess.run(
        [str(python_bin), "-c", code, str(sample_bundle)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            "The selected local Python/JAXSedFit cannot load the downloaded "
            f"posterior bundles. Python: {python_bin}\n{detail}"
        )
    marker_lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("__QVC_RESUME_PROBE__")
    ]
    if len(marker_lines) != 1:
        raise RuntimeError(f"Resume environment probe returned unexpected output: {result.stdout}")
    payload = json.loads(marker_lines[0].removeprefix("__QVC_RESUME_PROBE__"))
    print(f"Compatible JAXSedFit: {payload['jaxsedfit']}")
    print(f"Loaded probe bundle: {payload['observation']}")


def prepare_resume_records(
    path: Path,
    available: list[ResumeObject],
    input_csv: Path,
    dr16q_fits: Path,
) -> None:
    """Resolve saved spectra by plate/MJD/fiber once for all local batches."""
    import numpy as np
    import pandas as pd
    from astropy.io import fits

    expected_ids = [item.object_id for item in available]
    if path.is_file():
        frame = pd.read_csv(path, dtype={"object_id": str})
        actual_ids = [normalize_object_id(value) for value in frame["object_id"]]
        if actual_ids != expected_ids:
            raise ValueError(
                f"Existing prepared record manifest does not match source bundles: {path}"
            )
        print(f"Reusing {len(frame):,} prepared resume records: {path}")
        return

    print(
        f"Preparing {len(available):,} records once by exact DR16Q "
        "plate/MJD/fiber lookup."
    )
    input_frame = pd.read_csv(input_csv, dtype={"object_id": str})
    required = {
        "object_id",
        "sdss_name",
        "plate",
        "fiberid",
        "mjd",
        "SDSSS_RUN2D",
    }
    missing_columns = required - set(input_frame.columns)
    if missing_columns:
        raise ValueError(
            f"Input CSV {input_csv} is missing resume columns: "
            f"{sorted(missing_columns)}"
        )
    input_frame = input_frame.copy()
    input_frame["object_id"] = input_frame["object_id"].map(normalize_object_id)
    input_frame["sdss_name"] = input_frame["sdss_name"].map(normalize_sdss_name)
    if input_frame["object_id"].duplicated().any():
        raise ValueError(f"Input CSV {input_csv} contains duplicate object IDs.")
    by_id = input_frame.set_index("object_id", drop=False)
    missing_ids = [object_id for object_id in expected_ids if object_id not in by_id.index]
    if missing_ids:
        raise RuntimeError(
            f"Input CSV lacks {len(missing_ids)} downloaded object(s); "
            f"first missing: {missing_ids[:10]}"
        )
    selected_input = by_id.loc[expected_ids].reset_index(drop=True)

    def encode_spectrum_key(plate, mjd, fiber):
        plate = np.asarray(plate, dtype=np.int64)
        mjd = np.asarray(mjd, dtype=np.int64)
        fiber = np.asarray(fiber, dtype=np.int64)
        return ((plate * 100_000 + mjd) * 10_000) + fiber

    target_keys = encode_spectrum_key(
        selected_input["plate"],
        selected_input["mjd"],
        selected_input["fiberid"],
    )
    with fits.open(dr16q_fits, memmap=True) as hdul:
        catalog = hdul[1].data
        catalog_keys = encode_spectrum_key(
            catalog["PLATE"], catalog["MJD"], catalog["FIBERID"]
        )
        order = np.argsort(catalog_keys)
        sorted_keys = catalog_keys[order]
        left = np.searchsorted(sorted_keys, target_keys, side="left")
        right = np.searchsorted(sorted_keys, target_keys, side="right")
        bad = np.flatnonzero((right - left) != 1)
        if len(bad):
            examples = [
                {
                    "object_id": expected_ids[index],
                    "matches": int(right[index] - left[index]),
                }
                for index in bad[:10]
            ]
            raise RuntimeError(
                "Exact DR16Q plate/MJD/fiber lookup did not return one row for "
                f"{len(bad)} downloaded object(s): {examples}"
            )
        indices = order[left]
        catalog_names = np.asarray(catalog["SDSS_NAME"])[indices].astype(str)
        expected_names = np.asarray([item.sdss_name for item in available])
        mismatched_names = np.flatnonzero(catalog_names != expected_names)
        if len(mismatched_names):
            index = int(mismatched_names[0])
            raise RuntimeError(
                "DR16Q spectrum-key lookup changed the SDSS identity for "
                f"object_id={expected_ids[index]}: expected "
                f"{expected_names[index]!r}, found {catalog_names[index]!r}."
            )
        frame = pd.DataFrame(
            {
                "object_id": expected_ids,
                "sdss_name": catalog_names,
                "plate": np.asarray(catalog["PLATE"])[indices].astype(np.int64),
                "fiber": np.asarray(catalog["FIBERID"])[indices].astype(np.int64),
                "mjd": np.asarray(catalog["MJD"])[indices].astype(np.int64),
                "z": np.asarray(catalog["Z_SYS"])[indices].astype(float),
                "ra": np.asarray(catalog["RA"])[indices].astype(float),
                "dec": np.asarray(catalog["DEC"])[indices].astype(float),
                "loglbol": np.asarray(catalog["LOGLBOL"])[indices].astype(float),
                "SDSS_RUN2D": selected_input["SDSSS_RUN2D"].astype(str).to_numpy(),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    print(f"Wrote prepared resume records: {path}")


def build_fit_command(
    *,
    python_bin: Path,
    shard_path: Path,
    batch: list[ResumeObject],
    source_run: Path,
    source_bundle_dir: Path,
    output_run: Path,
    input_csv: Path,
    sed_photometry: Path,
    dr16q_fits: Path,
    cache_dir: Path,
    prepared_records: Path,
    verbose: bool,
) -> list[str]:
    command = [
        str(python_bin),
        "-m",
        "qvc.spectra.fit_spectra_jaxsedfit_joint",
        "--mode",
        "fit",
        str(shard_path),
        "--fpath-in",
        str(input_csv),
        "--dr16q-fits",
        str(dr16q_fits),
        "--cache-dir",
        str(cache_dir),
        "--sed-photometry-path",
        str(sed_photometry),
        "--output-dir",
        str(output_run / "all"),
        "--fig-dir",
        str(output_run / "figures_disabled"),
        "--filter_object_id",
        *[item.object_id for item in batch],
        "--nproc",
        "1",
        "--resume",
        str(source_bundle_dir),
        "--resume-run-name",
        source_run.name,
        "--resume-records-path",
        str(prepared_records),
        "--resume-only",
        "--allow-unannotated-resume-bundle",
        "--no-save-fig",
        "--no-save-jaxsedfit-samples",
        "--no-catalog-progress",
        "--no-print-convergence-summary",
    ]
    if verbose:
        command.append("--verbose")
    return command


def completed_marker_is_valid(
    marker_path: Path,
    shard_path: Path,
    object_ids: list[str],
    qvc_head: str,
) -> bool:
    if not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid completion marker {marker_path}: {exc}") from exc
    if marker.get("object_ids") != object_ids:
        raise ValueError(f"Completion marker has the wrong objects: {marker_path}")
    if marker.get("qvc_git_head") != qvc_head:
        raise ValueError(f"Completion marker has the wrong QVC commit: {marker_path}")
    validate_resume_catalog(shard_path, object_ids, qvc_head)
    return True


def validate_resume_catalog(
    path: Path,
    object_ids: list[str],
    qvc_head: str,
) -> None:
    """Validate catalog identity, resume provenance, and compact draws."""
    import h5py
    import numpy as np

    from qvc.spectra.catalog_hdf5 import read_spectra_catalog_hdf5

    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Resume catalog is absent or empty: {path}")
    with h5py.File(path, "r") as handle:
        recorded_head = handle.attrs.get("qvc_git_commit")
        if isinstance(recorded_head, bytes):
            recorded_head = recorded_head.decode("utf-8")
    if str(recorded_head or "") != qvc_head:
        raise ValueError(
            f"Resume catalog {path} records QVC commit {recorded_head!r}; "
            f"expected {qvc_head!r}."
        )

    catalog = read_spectra_catalog_hdf5(path)
    frame = catalog.frame
    actual_ids = [normalize_object_id(value) for value in frame.get("object_id", [])]
    if actual_ids != object_ids:
        raise ValueError(
            f"Resume catalog {path} has object IDs {actual_ids}; expected {object_ids}."
        )
    required_columns = {
        "fit_ok",
        "execution_mode",
        "resumed_from_path",
        "m_2500_dereddened",
        "m_2500_attenuated_model",
        "a_2500_total",
    }
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Resume catalog {path} is missing columns: {sorted(missing_columns)}"
        )
    if not frame["fit_ok"].astype(bool).all():
        raise ValueError(f"Resume catalog {path} contains fit_ok=False rows.")
    if not (frame["execution_mode"].astype(str) == "resumed").all():
        raise ValueError(f"Resume catalog {path} contains a non-resumed row.")
    if (frame["resumed_from_path"].astype(str).str.strip() == "").any():
        raise ValueError(f"Resume catalog {path} lacks source-bundle provenance.")
    for column in (
        "m_2500_dereddened",
        "m_2500_attenuated_model",
        "a_2500_total",
    ):
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Resume catalog {path} contains non-finite {column}.")
    if not (catalog.valid_count > 0).all():
        raise ValueError(f"Resume catalog {path} lacks PSF-fraction posterior draws.")
    if not (catalog.f_host_2500_psf_valid_count > 0).all():
        raise ValueError(f"Resume catalog {path} lacks 2500A host posterior draws.")


def file_set_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        stat = path.stat()
        digest.update(f"{path.name}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def append_failure(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def run_resume_command(
    command: list[str],
    repo_root: Path,
    env: dict[str, str],
) -> ResumeCommandResult:
    """Run one isolated fit subprocess and retain output for orderly reporting."""
    try:
        result = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return ResumeCommandResult(
            returncode=-1,
            stdout="",
            stderr="",
            execution_error=f"{type(exc).__name__}: {exc}",
        )
    return ResumeCommandResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a v3 spectral catalog directly from one run's original "
            "chunk catalogs and posterior bundles."
        )
    )
    parser.add_argument(
        "--output-catalog",
        default=str(DEFAULT_V3_CATALOG),
        help="Atomic qvc_spectra_catalog_v3 output.",
    )
    parser.add_argument(
        "--joint-posterior-selection-seed",
        type=int,
        default=3,
        help="Seed for the deterministic common 64-draw posterior indices.",
    )
    parser.add_argument(
        "--source-run",
        default=str(DEFAULT_SOURCE_RUN),
        help="Run directory containing chunk*.h5 catalogs and all/ bundles.",
    )
    parser.add_argument(
        "--python-bin",
        default=str(
            LOCAL_COMPATIBLE_PYTHON
            if LOCAL_COMPATIBLE_PYTHON.is_file()
            else Path(sys.executable)
        ),
        help=(
            "Python containing the bundle-compatible JAXSedFit. The local "
            "jaxcpu5_sdev environment is selected automatically when present."
        ),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--object-id", nargs="+")
    parser.add_argument(
        "--parallel",
        type=int,
        default=8,
        metavar="N",
        help=(
            "Run up to N independent HDF5/analytic workers concurrently "
            "(default: 8)."
        ),
    )
    parser.add_argument(
        "--max-tasks-per-worker",
        type=int,
        default=1,
        help=(
            "Source chunks processed before recycling a JAX worker (default: "
            "1). Values above 1 trade lower startup overhead for growing XLA "
            "compilation-cache memory."
        ),
    )
    parser.add_argument("--force-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-v3-shards",
        action="store_true",
        help=(
            "Keep restart shards after the final sibling catalog validates. "
            "By default they are removed only after a successful atomic merge."
        ),
    )
    parser.set_defaults(merge=True)
    parser.add_argument(
        "--no-merge",
        dest="merge",
        action="store_false",
        help="Do not merge all completed shards after a full successful run.",
    )
    args = parser.parse_args(argv)
    if args.start < 0:
        parser.error("--start cannot be negative.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    if args.parallel <= 0:
        parser.error("--parallel must be positive.")
    if args.max_tasks_per_worker <= 0:
        parser.error("--max-tasks-per-worker must be positive.")
    return args


def v3_shard_path(
    shard_dir: Path, *, ordinal: int, object_id: str
) -> Path:
    digest = hashlib.sha256(f"{ordinal}:{object_id}".encode()).hexdigest()[:12]
    return shard_dir / f"v3_{ordinal:05d}_{object_id}_{digest}.h5"


def v3_state_directory(output_catalog_path: Path) -> Path:
    """Return the visible resumable-state sibling for a v3 output catalog."""

    return output_catalog_path.parent / (
        f"{output_catalog_path.stem}_fitted_color_chunk_v3_state"
    )


def run_v3_build(args: argparse.Namespace, repo_root: Path) -> int:
    """Build v3 directly from original chunks and matching bundles."""

    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    from tqdm import tqdm
    source_run = Path(args.source_run).expanduser().resolve()
    output_catalog_path = Path(args.output_catalog).expanduser().resolve()
    if not source_run.is_dir():
        raise FileNotFoundError(f"Source run does not exist: {source_run}")
    # Resolve the source layout before checking the prediction runtime so a
    # malformed invocation fails on its direct cause.
    resolve_source_bundle_dir(source_run)
    commit, dirty, module_root = module_git_state("jaxsedfit")
    if commit != EXPECTED_JAXSEDFIT_COMMIT:
        raise RuntimeError(
            f"Loaded JAXSedFit commit {commit!r}; expected "
            f"{EXPECTED_JAXSEDFIT_COMMIT!r}."
        )
    if dirty:
        raise RuntimeError(
            f"JAXSedFit checkout is dirty at {module_root}; "
            "refusing an unreproducible fitted-color catalog."
        )
    source_rows, source_chunks = load_direct_v3_sources(
        source_run,
        selection_seed=int(args.joint_posterior_selection_seed),
    )
    object_ids = [str(value["object_id"]) for value in source_rows]
    source_target_count = source_catalog_row_count(source_chunks)
    omitted_target_count = source_target_count - len(source_rows)
    if omitted_target_count < 0:
        raise ValueError("Bundle-backed rows exceed the source chunk row count.")

    selected_ordinals = list(range(len(source_rows)))
    if args.object_id:
        requested = {normalize_object_id(value) for value in args.object_id}
        unknown = sorted(requested - set(object_ids))
        if unknown:
            raise ValueError(f"Unknown requested object IDs: {unknown[:10]}")
        selected_ordinals = [
            index
            for index, object_id in enumerate(object_ids)
            if object_id in requested
        ]
    selected_ordinals = selected_ordinals[args.start :]
    if args.limit is not None:
        selected_ordinals = selected_ordinals[: args.limit]
    if not selected_ordinals:
        raise RuntimeError("No bundle-backed chunk rows remain after selection.")

    state_dir = v3_state_directory(output_catalog_path)
    shard_dir = state_dir / "shards"
    all_tasks: list[V3BuildTask] = []
    for ordinal, source in enumerate(source_rows):
        object_id = str(source["object_id"])
        all_tasks.append(
            V3BuildTask(
                ordinal=ordinal,
                object_id=object_id,
                sdss_name=str(source["sdss_name"]),
                bundle_path=Path(source["bundle_path"]),
                source_chunk_path=Path(source["source_chunk_path"]),
                row=dict(source["row"]),
                psf_fraction_draws=source["psf_fraction_draws"],
                psf_fraction_valid_count=int(
                    source["psf_fraction_valid_count"]
                ),
                host_fraction_draws=source["host_fraction_draws"],
                host_fraction_valid_count=int(
                    source["host_fraction_valid_count"]
                ),
                shard_path=v3_shard_path(
                    shard_dir, ordinal=ordinal, object_id=object_id
                ),
                seed=int(args.joint_posterior_selection_seed),
            )
        )
    selected_set = set(selected_ordinals)
    tasks = [task for task in all_tasks if task.ordinal in selected_set]
    task_groups: list[tuple[tuple[V3BuildTask, ...], Path]] = []
    for task in tasks:
        if (
            not task_groups
            or task_groups[-1][0][-1].source_chunk_path != task.source_chunk_path
        ):
            digest = hashlib.sha256(
                f"{task.source_chunk_path}:{task.object_id}".encode("utf-8")
            ).hexdigest()[:12]
            task_groups.append(
                ((task,), shard_dir / f"chunk_{task.source_chunk_path.stem}_{digest}.h5")
            )
        else:
            previous, _ = task_groups[-1]
            combined = previous + (task,)
            identity = ",".join(item.object_id for item in combined)
            digest = hashlib.sha256(
                f"{task.source_chunk_path}:{identity}".encode("utf-8")
            ).hexdigest()[:12]
            path = shard_dir / f"chunk_{task.source_chunk_path.stem}_{digest}.h5"
            task_groups[-1] = (combined, path)

    if args.dry_run:
        print(
            f"Dry run: found {len(source_chunks):,} source chunk(s), "
            f"{len(all_tasks):,} bundle-backed row(s), and would process "
            f"{len(tasks):,} object(s) with up to {args.parallel} worker(s)."
        )
        print(
            f"Source targets: {source_target_count:,}; omitted without bundle: "
            f"{omitted_target_count:,}."
        )
        print(f"Source run: {source_run}")
        print(f"Atomic v3 output: {output_catalog_path}")
        return 0

    shard_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = state_dir / "manifest.json"
    manifest = {
        "schema": "qvc.local_spectra_v3_fitted_color_chunk.v3",
        "source_run": str(source_run),
        "source_chunk_count": len(source_chunks),
        "source_chunk_manifest_sha256": source_catalog_manifest_hash(
            source_chunks
        ),
        "source_row_count": len(all_tasks),
        "source_target_count": source_target_count,
        "omitted_target_count": omitted_target_count,
        "posterior_bundle_dir": str(source_run / "all"),
        "posterior_manifest_sha256": source_manifest_hash(
            [
                ResumeObject(
                    ordinal=task.ordinal,
                    object_id=task.object_id,
                    sdss_name=task.sdss_name,
                    bundle_path=task.bundle_path,
                )
                for task in all_tasks
            ]
        ),
        "joint_posterior_selection_seed": int(
            args.joint_posterior_selection_seed
        ),
        "mandatory_joint_psf_photometry": True,
        "jaxsedfit_git_commit": EXPECTED_JAXSEDFIT_COMMIT,
        "driver_sha256": file_sha256(
            Path(getattr(args, "driver_path", __file__)).resolve()
        ),
        "fit_module_sha256": file_sha256(
            repo_root / "src/qvc/spectra/fit_spectra_jaxsedfit_joint.py"
        ),
    }
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous != manifest:
            raise RuntimeError(
                "Existing direct-v3 shards were created from different inputs "
                f"or code: {manifest_path}"
            )
    else:
        atomic_write_json(manifest_path, manifest)

    full_selection = selected_ordinals == list(range(len(all_tasks)))
    if args.merge and full_selection and output_catalog_path.is_file() and not args.force_output:
        raise FileExistsError(
            f"Output already exists: {output_catalog_path}. Pass --force-output "
            "to atomically replace it."
        )

    worker_environment = runtime_environment(repo_root)
    for name in (
        "PYTHONPATH",
        "MPLCONFIGDIR",
        "JAX_ENABLE_X64",
        "JAX_PLATFORM_NAME",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    ):
        if name == "XLA_PYTHON_CLIENT_PREALLOCATE":
            os.environ[name] = worker_environment[name]
        else:
            os.environ.setdefault(name, worker_environment[name])

    pending: list[tuple[tuple[V3BuildTask, ...], Path]] = []
    failures: list[tuple[tuple[V3BuildTask, ...], BaseException]] = []
    with tqdm(
        total=len(tasks),
        desc="Building v3 joint posterior catalog",
        unit="object",
        dynamic_ncols=True,
    ) as progress:
        for group_tasks, group_path in task_groups:
            group_ids = [task.object_id for task in group_tasks]
            if group_path.is_file():
                validate_v3_catalog(
                    group_path,
                    group_ids,
                    selection_seed=group_tasks[0].seed,
                )
                progress.set_postfix_str(f"reused {group_ids[0]}")
                progress.update(len(group_tasks))
            else:
                pending.append((group_tasks, group_path))

        if pending:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=args.parallel,
                mp_context=context,
                max_tasks_per_child=int(args.max_tasks_per_worker),
            ) as executor:
                task_iterator = iter(pending)
                future_tasks = {}

                def submit_next() -> bool:
                    try:
                        next_group = next(task_iterator)
                    except StopIteration:
                        return False
                    group_tasks, group_path = next_group
                    future_tasks[
                        executor.submit(
                            process_v3_chunk, group_tasks, str(group_path)
                        )
                    ] = next_group
                    return True

                for _ in range(min(len(pending), 2 * args.parallel)):
                    submit_next()
                while future_tasks:
                    completed, _ = wait(
                        future_tasks, return_when=FIRST_COMPLETED
                    )
                    for future in completed:
                        group_tasks, group_path = future_tasks.pop(future)
                        try:
                            future.result()
                            validate_v3_catalog(
                                group_path,
                                [task.object_id for task in group_tasks],
                                selection_seed=group_tasks[0].seed,
                            )
                            progress.set_postfix_str(
                                f"wrote {group_tasks[0].object_id}"
                            )
                        except BaseException as exc:
                            failures.append((group_tasks, exc))
                            progress.write(
                                f"ERROR source_chunk={group_tasks[0].source_chunk_path.name}: "
                                f"{type(exc).__name__}: {exc}",
                                file=sys.stderr,
                            )
                        progress.update(len(group_tasks))
                        submit_next()

    if failures:
        failure_path = state_dir / "failures.json"
        atomic_write_json(
            failure_path,
            {
                "schema": "qvc.local_spectra_v3_analytic.failures.v1",
                "recorded_at": _now(),
                "failures": [
                    {
                        "ordinals": [task.ordinal for task in group_tasks],
                        "object_ids": [task.object_id for task in group_tasks],
                        "source_chunk": str(group_tasks[0].source_chunk_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    for group_tasks, exc in failures
                ],
            },
        )
        print(
            f"Direct v3 build retained resumable shards but {len(failures):,} "
            f"object(s) failed; details: {failure_path}",
            file=sys.stderr,
        )
        return 1

    if args.merge and full_selection:
        ordered_shards = [path for _, path in task_groups]
        print(
            f"Atomically merging {len(ordered_shards):,} source-chunk shards into "
            f"{output_catalog_path}"
        )
        merge_v3_shards(
            ordered_shards,
            output_catalog_path,
            expected_object_ids=object_ids,
            selection_seed=int(args.joint_posterior_selection_seed),
            base_rows=[dict(source["row"]) for source in source_rows],
            source_run=source_run,
            source_chunks=source_chunks,
            args=args,
        )
        print(f"Wrote validated v3 catalog: {output_catalog_path}")
        if not args.keep_v3_shards:
            for path in ordered_shards:
                path.unlink()
            print(
                f"Removed {len(ordered_shards):,} validated restart shard(s); "
                "they can be regenerated from the immutable posterior bundles."
            )
    elif args.merge:
        print(
            "Partial v3 selection completed; automatic final merge was "
            "intentionally skipped."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    # A command-line dry run still imports and validates the real catalog
    # stack, so it needs the same bundle-compatible interpreter as execution.
    # Unit tests that call ``main([...])`` remain in-process and injectable.
    if argv is None:
        reexec_with_fit_python(args, repo_root)

    return run_v3_build(args, repo_root)

    # Historical resume implementation retained below for audit reference.
    source_run, source_bundle_dir = resolve_source_bundle_dir(Path(args.source_run))
    input_csv = Path(args.input_csv).expanduser().resolve()
    sed_photometry = Path(args.sed_photometry).expanduser().resolve()
    dr16q_fits = Path(args.dr16q_fits).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output_run = Path(args.output_run).expanduser().resolve()
    python_bin = Path(args.python_bin).expanduser().resolve()
    for description, path in (
        ("input CSV", input_csv),
        ("SED photometry", sed_photometry),
        ("DR16Q catalog", dr16q_fits),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required {description} does not exist: {path}")
    if output_run == source_run or output_run == source_bundle_dir:
        raise ValueError("--output-run must be separate from the immutable source run.")

    available, missing = discover_resume_objects(source_bundle_dir, input_csv)
    print(f"Requested input objects: {len(available) + len(missing):,}")
    print(f"Downloaded posterior bundles: {len(available):,}")
    print(f"Cannot resume (bundle absent): {len(missing):,}")

    selected = available
    if args.object_id:
        requested = {normalize_object_id(value) for value in args.object_id}
        known = {item.object_id for item in available}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(
                f"Requested object IDs have no downloaded bundle: {unknown[:10]}"
            )
        selected = [item for item in selected if item.object_id in requested]
    selected = selected[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise RuntimeError("No downloaded objects remain after selection.")
    fallback_batches = chunked(selected, 1)
    original_catalogs = source_catalog_shards(source_run)
    print(
        f"Selected {len(selected):,} objects. Valid original v2 rows will be "
        "repaired analytically; remaining objects use restartable JAX fallback."
    )

    state_dir = output_run / ".resume_state"
    prepared_records = state_dir / "prepared_resume_records.csv"
    if args.dry_run:
        sample_shard = output_run / batch_shard_name(fallback_batches[0])
        command = build_fit_command(
            python_bin=python_bin,
            shard_path=sample_shard,
            batch=fallback_batches[0],
            source_run=source_run,
            source_bundle_dir=source_bundle_dir,
            output_run=output_run,
            input_csv=input_csv,
            sed_photometry=sed_photometry,
            dr16q_fits=dr16q_fits,
            cache_dir=cache_dir,
            prepared_records=prepared_records,
            verbose=args.verbose_fit,
        )
        print(
            f"Dry run; {len(original_catalogs):,} source catalog shards are "
            "available for analytic repair. Example strict fallback command:"
        )
        print(shlex.join(command))
        return 0

    env = runtime_environment(repo_root)
    for name in (
        "PYTHONPATH",
        "JAX_ENABLE_X64",
        "JAX_PLATFORM_NAME",
        "XLA_FLAGS",
        "MPLCONFIGDIR",
    ):
        os.environ.setdefault(name, env[name])
    probe_resume_environment(
        python_bin,
        repo_root,
        selected[0].bundle_path,
        env,
    )
    output_run.mkdir(parents=True, exist_ok=True)
    current_head = git_head(repo_root)
    prepare_resume_records(prepared_records, available, input_csv, dr16q_fits)
    manifest_path = state_dir / "run_manifest.json"
    manifest = {
        "schema": "qvc.local_spectra_resume.v1",
        "source_run": str(source_run),
        "source_bundle_dir": str(source_bundle_dir),
        "source_manifest_sha256": source_manifest_hash(available),
        "source_catalog_manifest_sha256": source_catalog_manifest_hash(
            original_catalogs
        ),
        "source_catalog_count": len(original_catalogs),
        "source_bundle_count": len(available),
        "missing_bundle_count": len(missing),
        "input_csv": str(input_csv),
        "sed_photometry": str(sed_photometry),
        "dr16q_fits": str(dr16q_fits),
        "python_bin": str(python_bin),
        "qvc_git_head": current_head,
        "resume_driver_sha256": file_sha256(Path(__file__).resolve()),
        "fit_module_sha256": file_sha256(
            repo_root / "src/qvc/spectra/fit_spectra_jaxsedfit_joint.py"
        ),
        "prepared_records_sha256": file_sha256(prepared_records),
        "created_at": _now(),
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        stable_keys = (
            "schema",
            "source_run",
            "source_manifest_sha256",
            "source_catalog_manifest_sha256",
            "source_catalog_count",
            "source_bundle_count",
            "input_csv",
            "sed_photometry",
            "dr16q_fits",
            "python_bin",
            "qvc_git_head",
            "resume_driver_sha256",
            "fit_module_sha256",
            "prepared_records_sha256",
        )
        changed = [key for key in stable_keys if existing.get(key) != manifest.get(key)]
        if changed:
            raise RuntimeError(
                f"Existing output run was created with different settings ({changed}): "
                f"{manifest_path}"
            )
    else:
        atomic_write_json(manifest_path, manifest)

    completed_dir = state_dir / "completed"
    failures_path = state_dir / "failures.jsonl"
    failed_artifacts = state_dir / "failed_artifacts"
    completed_now = 0
    repaired_now = 0
    fallback_completed = 0
    skipped = 0
    failures = 0
    consecutive_failures = 0
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from tqdm import tqdm

    with tqdm(
        total=len(selected),
        desc="Repairing spectra catalogs",
        unit="object",
        dynamic_ncols=True,
    ) as progress:
        from qvc.spectra.catalog_hdf5 import read_spectra_catalog_hdf5

        selected_by_id = {item.object_id: item for item in selected}
        source_row_owner: dict[str, Path] = {}
        repaired_ids: set[str] = set()
        repair_fallbacks: dict[str, str] = {}
        expected_shards: list[Path] = []
        expected_shard_objects: dict[Path, list[str]] = {}

        for source_catalog_path in original_catalogs:
            try:
                source_catalog = read_spectra_catalog_hdf5(source_catalog_path)
                import h5py

                with h5py.File(source_catalog_path, "r") as handle:
                    source_catalog_commit = handle.attrs.get(
                        "qvc_git_commit", ""
                    )
                if isinstance(source_catalog_commit, bytes):
                    source_catalog_commit = source_catalog_commit.decode(
                        "utf-8"
                    )
                source_catalog_commit = str(source_catalog_commit)
            except Exception as exc:
                progress.write(
                    f"WARNING: cannot reuse source catalog {source_catalog_path}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                continue

            repaired_rows: list[RepairedCatalogRow] = []
            for row_index, raw_id in enumerate(source_catalog.frame["object_id"]):
                object_id = normalize_object_id(raw_id)
                previous_owner = source_row_owner.get(object_id)
                if previous_owner is not None:
                    raise RuntimeError(
                        f"Duplicate source catalog row for object_id={object_id}: "
                        f"{previous_owner} and {source_catalog_path}"
                    )
                source_row_owner[object_id] = source_catalog_path
                item = selected_by_id.get(object_id)
                if item is None:
                    continue
                try:
                    repaired_rows.append(
                        repair_source_catalog_row(
                            item,
                            source_catalog,
                            row_index,
                            source_catalog_path=source_catalog_path,
                            source_catalog_commit=source_catalog_commit,
                            source_run=source_run,
                        )
                    )
                except Exception as exc:
                    repair_fallbacks[object_id] = (
                        f"{type(exc).__name__}: {exc}"
                    )

            for repaired_run in contiguous_repair_runs(repaired_rows):
                batch = [value.item for value in repaired_run]
                object_ids = [item.object_id for item in batch]
                shard_path = output_run / batch_shard_name(batch)
                marker_path = completed_dir / f"{shard_path.stem}.json"
                expected_shards.append(shard_path)
                expected_shard_objects[shard_path] = object_ids
                if completed_marker_is_valid(
                    marker_path,
                    shard_path,
                    object_ids,
                    current_head,
                ):
                    skipped += len(batch)
                    repaired_ids.update(object_ids)
                    progress.set_postfix_str(
                        f"skipped repaired {object_ids[-1]}"
                    )
                    progress.update(len(batch))
                    continue
                if shard_path.exists():
                    validate_resume_catalog(
                        shard_path, object_ids, current_head
                    )
                    atomic_write_json(
                        marker_path,
                        {
                            "schema": "qvc.local_spectra_resume.completed.v1",
                            "completed_at": _now(),
                            "recovered_unmarked_shard": True,
                            "mode": "analytic_m2500_norm12",
                            "object_ids": object_ids,
                            "source_catalog": str(source_catalog_path),
                            "source_bundles": [
                                str(item.bundle_path) for item in batch
                            ],
                            "shard": str(shard_path),
                            "qvc_git_head": current_head,
                        },
                    )
                    skipped += len(batch)
                    repaired_ids.update(object_ids)
                    progress.set_postfix_str(
                        f"recovered repaired {object_ids[-1]}"
                    )
                    progress.update(len(batch))
                    continue
                try:
                    write_repaired_catalog_shard(
                        shard_path,
                        repaired_run,
                        args=args,
                        source_catalog_path=source_catalog_path,
                    )
                    validate_resume_catalog(
                        shard_path, object_ids, current_head
                    )
                except Exception as exc:
                    shard_path.unlink(missing_ok=True)
                    expected_shards.pop()
                    expected_shard_objects.pop(shard_path, None)
                    for object_id in object_ids:
                        repair_fallbacks[object_id] = (
                            f"repair shard write failed: {type(exc).__name__}: "
                            f"{exc}"
                        )
                    continue
                atomic_write_json(
                    marker_path,
                    {
                        "schema": "qvc.local_spectra_resume.completed.v1",
                        "completed_at": _now(),
                        "mode": "analytic_m2500_norm12",
                        "object_ids": object_ids,
                        "source_catalog": str(source_catalog_path),
                        "source_bundles": [
                            str(item.bundle_path) for item in batch
                        ],
                        "shard": str(shard_path),
                        "qvc_git_head": current_head,
                    },
                )
                repaired_ids.update(object_ids)
                repaired_now += len(batch)
                completed_now += len(batch)
                progress.set_postfix_str(f"repaired {object_ids[-1]}")
                progress.update(len(batch))

            # A targeted smoke run need not index thousands of irrelevant
            # source shards once every requested identity has been resolved.
            if len(selected) < len(available) and set(selected_by_id).issubset(
                source_row_owner
            ):
                break

        fallback_items = [
            item for item in selected if item.object_id not in repaired_ids
        ]
        absent_source_count = sum(
            item.object_id not in source_row_owner for item in fallback_items
        )
        rejected_source_count = len(fallback_items) - absent_source_count
        progress.write(
            f"Analytic repair eligible: {len(repaired_ids):,}; strict JAX "
            f"fallback: {len(fallback_items):,} "
            f"({absent_source_count:,} absent source rows, "
            f"{rejected_source_count:,} rejected source rows)."
        )
        repair_fallback_path = state_dir / "analytic_repair_fallbacks.json"
        atomic_write_json(
            repair_fallback_path,
            {
                "schema": "qvc.local_spectra_resume.repair_fallbacks.v1",
                "recorded_at": _now(),
                "absent_source_object_ids": [
                    item.object_id
                    for item in fallback_items
                    if item.object_id not in source_row_owner
                ],
                "rejected_source_rows": repair_fallbacks,
            },
        )

        pending_calls = []
        for batch in chunked(fallback_items, 1):
            shard_path = output_run / batch_shard_name(batch)
            marker_path = completed_dir / f"{shard_path.stem}.json"
            object_ids = [item.object_id for item in batch]
            object_label = ",".join(object_ids)
            expected_shards.append(shard_path)
            expected_shard_objects[shard_path] = object_ids
            if completed_marker_is_valid(
                marker_path,
                shard_path,
                object_ids,
                current_head,
            ):
                skipped += len(batch)
                progress.set_postfix_str(f"skipped {object_label}")
                progress.update(len(batch))
                continue
            if shard_path.exists():
                # Recover the narrow crash window after the atomic HDF5 write but
                # before the completion marker was persisted.
                validate_resume_catalog(shard_path, object_ids, current_head)
                atomic_write_json(
                    marker_path,
                    {
                        "schema": "qvc.local_spectra_resume.completed.v1",
                        "completed_at": _now(),
                        "recovered_unmarked_shard": True,
                        "object_ids": object_ids,
                        "source_bundles": [str(item.bundle_path) for item in batch],
                        "shard": str(shard_path),
                        "qvc_git_head": current_head,
                    },
                )
                skipped += len(batch)
                progress.write(f"Recovered valid unmarked shard: {shard_path.name}")
                progress.set_postfix_str(f"recovered {object_label}")
                progress.update(len(batch))
                continue
            command = build_fit_command(
                python_bin=python_bin,
                shard_path=shard_path,
                batch=batch,
                source_run=source_run,
                source_bundle_dir=source_bundle_dir,
                output_run=output_run,
                input_csv=input_csv,
                sed_photometry=sed_photometry,
                dr16q_fits=dr16q_fits,
                cache_dir=cache_dir,
                prepared_records=prepared_records,
                verbose=args.verbose_fit,
            )
            pending_calls.append(
                {
                    "batch": batch,
                    "shard_path": shard_path,
                    "marker_path": marker_path,
                    "object_ids": object_ids,
                    "object_label": object_label,
                    "command": command,
                }
            )

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            future_calls = {
                executor.submit(
                    run_resume_command,
                    call["command"],
                    repo_root,
                    env,
                ): call
                for call in pending_calls
            }
            stop_requested = False
            for future in as_completed(future_calls):
                call = future_calls[future]
                if future.cancelled():
                    continue
                batch = call["batch"]
                shard_path = call["shard_path"]
                marker_path = call["marker_path"]
                object_ids = call["object_ids"]
                object_label = call["object_label"]
                command = call["command"]
                result = future.result()
                if args.verbose_fit:
                    child_output = "\n".join(
                        value.rstrip()
                        for value in (result.stdout, result.stderr)
                        if value.strip()
                    )
                    if child_output:
                        progress.write(child_output)

                validation_error = ""
                if result.returncode == 0:
                    try:
                        validate_resume_catalog(
                            shard_path, object_ids, current_head
                        )
                    except Exception as exc:
                        validation_error = f"{type(exc).__name__}: {exc}"
                if result.returncode == 0 and not validation_error:
                    atomic_write_json(
                        marker_path,
                        {
                            "schema": "qvc.local_spectra_resume.completed.v1",
                            "completed_at": _now(),
                            "object_ids": object_ids,
                            "source_bundles": [
                                str(item.bundle_path) for item in batch
                            ],
                            "shard": str(shard_path),
                            "qvc_git_head": current_head,
                        },
                    )
                    completed_now += len(batch)
                    fallback_completed += len(batch)
                    consecutive_failures = 0
                    progress.set_postfix_str(f"completed {object_label}")
                    progress.update(len(batch))
                    continue

                failures += len(batch)
                consecutive_failures += 1
                if shard_path.exists():
                    failed_artifacts.mkdir(parents=True, exist_ok=True)
                    quarantined = failed_artifacts / (
                        f"{shard_path.stem}."
                        f"{datetime.now().strftime('%Y%m%dT%H%M%S')}.h5"
                    )
                    os.replace(shard_path, quarantined)
                failure_log = (
                    state_dir / "failure_logs" / f"{shard_path.stem}.log"
                )
                failure_log.parent.mkdir(parents=True, exist_ok=True)
                failure_log.write_text(
                    "\n".join(
                        value.rstrip()
                        for value in (
                            result.execution_error,
                            result.stdout,
                            result.stderr,
                            validation_error,
                        )
                        if value.strip()
                    )
                    + "\n"
                )
                append_failure(
                    failures_path,
                    {
                        "failed_at": _now(),
                        "object_ids": object_ids,
                        "returncode": result.returncode,
                        "execution_error": result.execution_error,
                        "validation_error": validation_error,
                        "failure_log": str(failure_log),
                        "command": command,
                    },
                )
                failure_detail = (
                    f"; output validation failed: {validation_error}"
                    if validation_error
                    else ""
                )
                progress.write(
                    "ERROR: resume call failed with return code "
                    f"{result.returncode}{failure_detail}; log: {failure_log}",
                    file=sys.stderr,
                )
                progress.set_postfix_str(f"failed {object_label}")
                progress.update(len(batch))
                if (
                    consecutive_failures >= args.max_consecutive_failures
                    and not stop_requested
                ):
                    stop_requested = True
                    cancelled = sum(
                        other_future.cancel()
                        for other_future in future_calls
                        if other_future is not future
                    )
                    progress.write(
                        f"Stopping after {consecutive_failures} consecutive "
                        f"failed resume calls; cancelled {cancelled} queued "
                        "calls. Already-running calls will finish.",
                        file=sys.stderr,
                    )

    print(
        f"Resume summary: completed now={completed_now:,} "
        f"(analytic={repaired_now:,}, JAX fallback={fallback_completed:,}), "
        f"skipped={skipped:,}, failed objects={failures:,}."
    )
    full_selection = [item.object_id for item in selected] == [
        item.object_id for item in available
    ]
    if failures:
        return 1
    if args.merge and full_selection:
        merged_path = output_run.parent / f"{output_run.name}.h5"
        expected_shards = sorted(expected_shards)
        actual_shards = sorted(output_run.glob("*.h5"))
        unexpected_shards = sorted(set(actual_shards) - set(expected_shards))
        missing_shards = sorted(set(expected_shards) - set(actual_shards))
        if unexpected_shards or missing_shards:
            raise RuntimeError(
                "Top-level resume shards do not exactly match the controlled "
                f"manifest. Unexpected: {unexpected_shards[:10]}; "
                f"missing: {missing_shards[:10]}."
            )
        for shard_path in expected_shards:
            validate_resume_catalog(
                shard_path,
                expected_shard_objects[shard_path],
                current_head,
            )
        shards_sha256 = file_set_hash(expected_shards)
        merged_marker_path = state_dir / "merged_catalog.json"
        if merged_path.exists():
            if not merged_marker_path.is_file():
                raise RuntimeError(
                    "Merged catalog exists without a matching completion marker; "
                    f"refusing to treat it as current: {merged_path}"
                )
            merged_marker = json.loads(merged_marker_path.read_text())
            if (
                merged_marker.get("shards_sha256") != shards_sha256
                or merged_marker.get("qvc_git_head") != current_head
            ):
                raise RuntimeError(
                    f"Merged catalog is stale relative to resumed shards: {merged_path}"
                )
            validate_resume_catalog(
                merged_path,
                [item.object_id for item in available],
                current_head,
            )
            print(f"Validated existing merged catalog: {merged_path}")
        else:
            merge_command = [
                str(python_bin),
                "-m",
                "qvc.spectra.merge_results",
                output_run.name,
                "--base-dir",
                str(output_run.parent),
                "--skip-populate-sdss",
                "--dedup-keys",
                "object_id",
                "--out",
                str(merged_path),
            ]
            print(f"Merging {len(available):,} resumed objects into {merged_path}")
            subprocess.run(merge_command, cwd=repo_root, env=env, check=True)
            validate_resume_catalog(
                merged_path,
                [item.object_id for item in available],
                current_head,
            )
            atomic_write_json(
                merged_marker_path,
                {
                    "schema": "qvc.local_spectra_resume.merged.v1",
                    "completed_at": _now(),
                    "merged_catalog": str(merged_path),
                    "object_count": len(available),
                    "shards_sha256": shards_sha256,
                    "qvc_git_head": current_head,
                },
            )
    elif args.merge:
        print("Partial selection completed; automatic merge was intentionally skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
