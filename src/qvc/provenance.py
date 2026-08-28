"""Small, dependency-light helpers for reproducible HDF5 run provenance."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py

PROVENANCE_SCHEMA = "qvc.run_provenance.v1"
PROVENANCE_ENV = "QVC_SUBMISSION_PROVENANCE_B64"
RETRY_ENV = "QVC_RETRY_PROVENANCE_B64"
HASH_LIMIT_BYTES = 64 * 1024 * 1024
SENSITIVE_FRAGMENTS = ("token", "password", "passwd", "secret", "api_key", "apikey", "credential")
ENV_ALLOWLIST = (
    "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
    "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU",
    "SLURM_JOB_PARTITION", "PREFIX", "SUFFIX", "QVC_DATA_DIR",
    "JAX_ENABLE_X64", "JAX_PLATFORM_NAME", "XLA_FLAGS", "NUM_CORES",
)
DEPENDENCIES = {
    "qvc": "qvc",
    "EzTaoX": "eztaox",
    "JAXQSOFit": "jaxqsofit",
    "JAXSEDFit": "jaxsedfit",
}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, argparse.Namespace):
        return _json_safe(vars(value))
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _sensitive(name: str) -> bool:
    normalized = name.lower().replace("-", "_")
    return any(fragment in normalized for fragment in SENSITIVE_FRAGMENTS)


def redact(value: Any, key: str = "") -> Any:
    if key and _sensitive(key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(k): redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return _json_safe(value)


def canonical_json(record: Mapping[str, Any]) -> str:
    return json.dumps(redact(record), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def encode_record(record: Mapping[str, Any]) -> str:
    return base64.b64encode(canonical_json(record).encode("utf-8")).decode("ascii")


def decode_record(encoded: str | None) -> dict[str, Any] | None:
    if not encoded:
        return None
    try:
        value = json.loads(base64.b64decode(encoded).decode("utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def submission_record(entrypoint: str, argv: Sequence[str], resolved: Mapping[str, Any]) -> dict[str, Any]:
    safe_argv = redact_argv(list(argv))
    return {
        "entrypoint": entrypoint,
        "recorded_at": _now(),
        "argv": safe_argv,
        "command": shlex.join(safe_argv),
        "resolved": redact(resolved),
    }


def redact_argv(argv: Sequence[str]) -> list[str]:
    result = [str(v) for v in argv]
    redact_next = False
    for index, token in enumerate(result):
        if redact_next:
            result[index] = "<redacted>"
            redact_next = False
            continue
        if token.startswith("--"):
            name, separator, _ = token.partition("=")
            if _sensitive(name):
                if separator:
                    result[index] = f"{name}=<redacted>"
                else:
                    redact_next = True
    return result


def fingerprint_path(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    text = "" if path is None else str(path)
    item: dict[str, Any] = {"path": text}
    if not text:
        item["exists"] = False
        return item
    resolved = Path(text).expanduser().resolve(strict=False)
    item["resolved_path"] = str(resolved)
    item["exists"] = resolved.exists()
    if not resolved.exists():
        return item
    stat = resolved.stat()
    item.update(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if resolved.is_file() and stat.st_size <= HASH_LIMIT_BYTES:
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        item["sha256"] = digest.hexdigest()
    elif resolved.is_file():
        item["sha256"] = None
        item["sha256_skipped_reason"] = f"larger_than_{HASH_LIMIT_BYTES}_bytes"
    return item


def _git_state(path: Path) -> dict[str, Any]:
    try:
        root = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
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
        return {"root": root, "commit": commit, "dirty": dirty}
    except Exception:
        return {"root": "", "commit": "", "dirty": None}


def dependency_state(display_name: str, module_name: str) -> dict[str, Any]:
    item: dict[str, Any] = {"module": module_name, "available": False}
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception:
        spec = None
    if spec is None:
        return item
    item["available"] = True
    item["module_path"] = str(spec.origin or "")
    try:
        item["version"] = importlib.metadata.version(module_name)
    except importlib.metadata.PackageNotFoundError:
        item["version"] = ""
    origin = Path(spec.origin).resolve() if spec.origin else Path.cwd()
    item["git"] = _git_state(origin.parent)
    return item


@lru_cache(maxsize=1)
def runtime_state() -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "working_directory": str(Path.cwd()),
        "environment": {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ},
        "dependencies": {
            name: dependency_state(name, module) for name, module in DEPENDENCIES.items()
        },
    }


def specialize_object_argv(argv: Sequence[str], object_id: str) -> list[str]:
    """Replace a possibly chunked object-id option with one exact object."""
    values = list(argv)
    for option in ("--filter_object_id", "--filter-object-id"):
        if option not in values:
            continue
        start = values.index(option) + 1
        stop = start
        while stop < len(values) and not values[stop].startswith("--"):
            stop += 1
        return values[:start] + [str(object_id)] + values[stop:]
    return values + ["--filter_object_id", str(object_id)]


def build_run_record(
    entrypoint: str,
    args: argparse.Namespace | Mapping[str, Any],
    *,
    argv: Sequence[str] | None = None,
    object_id: str | None = None,
    input_paths: Mapping[str, Any] | None = None,
    event_type: str = "run",
) -> dict[str, Any]:
    if argv is None and entrypoint.startswith("qvc."):
        actual_argv = [sys.executable, "-m", entrypoint, *sys.argv[1:]]
    else:
        actual_argv = list(sys.argv if argv is None else argv)
    rerun_argv = specialize_object_argv(actual_argv, object_id) if object_id is not None else actual_argv
    submission = decode_record(os.environ.get(PROVENANCE_ENV))
    retry = decode_record(os.environ.get(RETRY_ENV))
    event = {"type": event_type, "recorded_at": _now()}
    if retry:
        event["retry_submission"] = retry
    record = {
        "schema": PROVENANCE_SCHEMA,
        "entrypoint": entrypoint,
        "recorded_at": event["recorded_at"],
        "submission": submission,
        "module": {
            "argv": redact_argv(actual_argv),
            "parsed_args": redact(vars(args) if isinstance(args, argparse.Namespace) else args),
            "command": shlex.join(redact_argv(actual_argv)),
            "rerun_argv": redact_argv(rerun_argv),
            "rerun_command": shlex.join(redact_argv(rerun_argv)),
        },
        "runtime": runtime_state(),
        "inputs": {str(k): fingerprint_path(v) for k, v in (input_paths or {}).items() if v},
        "events": [event],
    }
    record["qvc_git"] = record["runtime"]["dependencies"]["qvc"]["git"]
    return record


def provenance_fingerprint(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()


def merge_history(current: Mapping[str, Any], previous: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(_json_safe(current))
    if previous:
        old_events = list(previous.get("events", []))
        result["events"] = old_events + list(result.get("events", []))
        result["source_provenance_fingerprint"] = provenance_fingerprint(previous)
    return result


def read_hdf5_provenance(path_or_handle: str | os.PathLike[str] | h5py.File) -> dict[str, Any] | None:
    def _read(handle: h5py.File) -> dict[str, Any] | None:
        raw = handle.attrs.get("qvc_provenance_json")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        if not raw:
            return None
        try:
            value = json.loads(str(raw))
        except (TypeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None
    if isinstance(path_or_handle, h5py.File):
        return _read(path_or_handle)
    with h5py.File(path_or_handle, "r") as handle:
        return _read(handle)


def write_hdf5_provenance(path_or_handle: str | os.PathLike[str] | h5py.File, record: Mapping[str, Any]) -> None:
    safe = redact(record)
    payload = canonical_json(safe)
    git_commit = str(safe.get("qvc_git", {}).get("commit", ""))
    module = safe.get("module", {})
    submission = safe.get("submission") or {}

    def _write(handle: h5py.File) -> None:
        handle.attrs["qvc_provenance_schema"] = PROVENANCE_SCHEMA
        handle.attrs["qvc_provenance_json"] = payload
        handle.attrs["qvc_rerun_command"] = str(module.get("rerun_command", module.get("command", "")))
        handle.attrs["qvc_submission_command"] = str(submission.get("command", ""))
        handle.attrs["qvc_git_commit"] = git_commit
        handle.attrs["qvc_recorded_at"] = str(safe.get("recorded_at", _now()))
        handle.attrs["qvc_entrypoint"] = str(safe.get("entrypoint", ""))
    if isinstance(path_or_handle, h5py.File):
        _write(path_or_handle)
    else:
        with h5py.File(path_or_handle, "r+") as handle:
            _write(handle)
