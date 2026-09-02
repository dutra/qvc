#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fnmatch
import getpass
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import log_ndtr, ndtri_exp

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from rich.markup import escape
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, ScrollableContainer, Vertical
    from textual.screen import ModalScreen
    from textual.widgets import Button, DataTable, Footer, Header, Input, Label, Static

    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False

    class _UnavailableTextual:
        """Import-time stand-in so the non-TUI CLI does not require Textual."""

        @classmethod
        def __class_getitem__(cls, _item):
            return cls

        def __init__(self, *_args, **_kwargs):
            raise ImportError("Textual is required for TUI mode.", name="textual")

    class _UnavailableButton(_UnavailableTextual):
        Pressed = object()

    class _UnavailableDataTable(_UnavailableTextual):
        RowHighlighted = object()
        RowSelected = object()

    class _UnavailableInput(_UnavailableTextual):
        Changed = object()

    def _unavailable_decorator(*_args, **_kwargs):
        return lambda function: function

    App = ModalScreen = Horizontal = ScrollableContainer = Vertical = _UnavailableTextual
    Footer = Header = Label = Static = _UnavailableTextual
    Button = _UnavailableButton
    DataTable = _UnavailableDataTable
    Input = _UnavailableInput
    Binding = lambda *_args, **_kwargs: None
    ComposeResult = Any
    escape = str
    on = work = _unavailable_decorator


STATE_STYLES = {
    "COMPLETED": "green",
    "RUNNING": "cyan",
    "PENDING": "yellow",
    "OUT_OF_MEMORY": "bold red",
    "FAILED": "red",
    "TIMEOUT": "magenta",
    "CANCELLED": "dim yellow",
}

ARRAY_TASK_RE = re.compile(r"^(?P<array_job_id>\d+)_(?P<task_id>\d+)$")
SCALAR_JOB_RE = re.compile(r"^\d+$")
MEMORY_RE = re.compile(
    r"^(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[KMGTPE]?)B?(?P<scope>[cn]?)$",
    re.IGNORECASE,
)
ACCOUNTING_FIELDS = (
    "JobID%128,JobName%256,State,ElapsedRaw,TotalCPU,"
    "AllocCPUS,NNodes,MaxRSS,ReqMem,TimelimitRaw,ExitCode,"
    "Partition,Reason,Submit,Start,StdOut,StdErr,SubmitLine,WorkDir"
)
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}
LIVE_QUEUE_FORMAT = "%i|%F|%K|%j|%T|%M|%l|%L|%C|%D|%P|%R|%V|%S"
HELD_REASONS = {"JobHeldUser", "JobHeldAdmin"}


def _integer(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_state(state):
    return state.strip().split()[0].rstrip("+") if state.strip() else "UNKNOWN"


def _parse_duration_seconds(value):
    """Parse Slurm's [days-]hours:minutes:seconds duration format."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    try:
        day_text, clock = text.split("-", 1) if "-" in text else ("0", text)
        pieces = clock.split(":")
        if len(pieces) == 3:
            hours, minutes, seconds = pieces
        elif len(pieces) == 2:
            hours, minutes, seconds = "0", pieces[0], pieces[1]
        else:
            return None
        return int(day_text) * 86400 + int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return None


def _parse_memory_bytes(value):
    """Parse Slurm K/M/G/T values, ignoring an optional c/n scope suffix."""
    if value is None:
        return None
    match = MEMORY_RE.fullmatch(str(value).strip())
    if match is None:
        return None
    exponent = "KMGTPE".find(match.group("unit").upper()) + 1 if match.group("unit") else 0
    return int(float(match.group("value")) * (1024**exponent))


def _requested_memory_bytes(req_mem, alloc_cpus=None, nodes=None):
    if req_mem is None:
        return None
    match = MEMORY_RE.fullmatch(str(req_mem).strip())
    if match is None:
        return None
    base = _parse_memory_bytes(req_mem)
    scope = match.group("scope").lower()
    if scope == "c":
        return base * alloc_cpus if alloc_cpus else None
    if scope == "n":
        return base * nodes if nodes else None
    return base


def _parse_sacct_output(output):
    """Parse array allocations and attach the largest .batch/.extern MaxRSS.

    ``JobID`` preserves the ``ARRAYID_TASKID`` identity, unlike ``JobIDRaw``
    on Bouchet, where each array task has a separate numeric job ID.
    """
    allocations = {}
    scalar_candidates = {}
    step_max_rss = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 11:
            continue
        (
            job_id,
            job_name,
            state,
            elapsed_raw,
            total_cpu,
            alloc_cpus,
            nodes,
            max_rss,
            req_mem,
            timelimit_raw,
            exit_code,
        ) = parts[:11]
        (
            partition,
            reason,
            submit_time,
            start_time,
            stdout_path,
            stderr_path,
            submit_line,
            work_dir,
        ) = (parts[11:19] + [""] * 8)[:8]

        allocation_match = ARRAY_TASK_RE.fullmatch(job_id)
        scalar_match = SCALAR_JOB_RE.fullmatch(job_id)
        if allocation_match is not None or scalar_match is not None:
            array_job_id = (
                allocation_match.group("array_job_id")
                if allocation_match is not None
                else job_id
            )
            task_id = (
                int(allocation_match.group("task_id"))
                if allocation_match is not None
                else 0
            )
            row = {
                "job_id": job_id,
                "array_job_id": array_job_id,
                "task_id": task_id,
                "job_name": job_name,
                "state": _normalize_state(state),
                "elapsed_seconds": _integer(elapsed_raw),
                "total_cpu_seconds": _parse_duration_seconds(total_cpu),
                "alloc_cpus": _integer(alloc_cpus),
                "nodes": _integer(nodes),
                "max_rss_bytes": _parse_memory_bytes(max_rss),
                "req_mem": req_mem or None,
                "timelimit_minutes": _integer(timelimit_raw),
                "exit_code": exit_code,
                "partition": partition,
                "reason": reason,
                "submit_time": submit_time,
                "start_time": start_time,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "submit_line": submit_line,
                "work_dir": work_dir,
                "time_left_seconds": None,
                "held": reason in HELD_REASONS,
                "job_elapsed_seconds": None,
                "job_state": None,
            }
            row["requested_memory_bytes"] = _requested_memory_bytes(
                row["req_mem"], row["alloc_cpus"], row["nodes"]
            )
            if allocation_match is not None:
                allocations[job_id] = row
            else:
                scalar_candidates[job_id] = row
            continue
        parent_id, separator, _step_name = job_id.rpartition(".")
        if separator:
            rss = _parse_memory_bytes(max_rss)
            if rss is not None:
                step_max_rss[parent_id] = max(step_max_rss.get(parent_id, 0), rss)

    array_parent_ids = {row["array_job_id"] for row in allocations.values()}
    for parent_id, parent in scalar_candidates.items():
        if parent_id in array_parent_ids:
            for row in allocations.values():
                if row["array_job_id"] == parent_id:
                    row["job_elapsed_seconds"] = parent["elapsed_seconds"]
                    row["job_state"] = parent["state"]
            continue
        parent["job_elapsed_seconds"] = parent["elapsed_seconds"]
        parent["job_state"] = parent["state"]
        allocations[parent_id] = parent

    for job_id, row in allocations.items():
        if job_id in step_max_rss:
            row["max_rss_bytes"] = max(row["max_rss_bytes"] or 0, step_max_rss[job_id])
    return list(allocations.values())


def get_accounting_rows(starttime="now-7days", endtime=None, job_ids=None):
    cmd = [
        "sacct",
        "--array",
        "--noheader",
        "--parsable2",
        f"--user={getpass.getuser()}",
        f"--starttime={starttime}",
        f"--format={ACCOUNTING_FIELDS}",
    ]
    if endtime is not None:
        cmd.append(f"--endtime={endtime}")
    if job_ids:
        cmd.append(f"--jobs={','.join(str(job_id) for job_id in job_ids)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return _parse_sacct_output(result.stdout)


def _parse_squeue_output(output):
    """Parse expanded live queue rows used by the interactive manager."""
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.rstrip().split("|", maxsplit=13)
        if len(parts) != 14:
            continue
        (
            job_id,
            array_job_id,
            array_task_id,
            job_name,
            state,
            elapsed,
            time_limit,
            time_left,
            alloc_cpus,
            nodes,
            partition,
            reason,
            submit_time,
            start_time,
        ) = parts
        array_match = ARRAY_TASK_RE.fullmatch(job_id)
        base_id = array_job_id if array_job_id.isdigit() else (
            array_match.group("array_job_id") if array_match else job_id
        )
        task_id = _integer(array_task_id)
        if task_id is None and array_match is not None:
            task_id = int(array_match.group("task_id"))
        normalized_reason = reason.strip().strip("()")
        rows.append(
            {
                "job_id": job_id,
                "array_job_id": base_id,
                "task_id": task_id,
                "job_name": job_name,
                "state": _normalize_state(state),
                "elapsed_seconds": _parse_duration_seconds(elapsed),
                "timelimit_seconds": _parse_duration_seconds(time_limit),
                "time_left_seconds": _parse_duration_seconds(time_left),
                "alloc_cpus": _integer(alloc_cpus),
                "nodes": _integer(nodes),
                "partition": partition,
                "reason": normalized_reason,
                "submit_time": submit_time,
                "start_time": start_time,
                "held": normalized_reason in HELD_REASONS,
            }
        )
    return rows


def get_live_queue_rows():
    cmd = ["squeue", "--array", "--noheader", f"--format={LIVE_QUEUE_FORMAT}", "--me"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return _parse_squeue_output(result.stdout)


def matching_arrays(rows, pattern):
    groups = {}
    for row in rows:
        if fnmatch.fnmatch(row["job_name"], pattern):
            groups.setdefault((row["array_job_id"], row["job_name"]), []).append(row)
    return groups


def normalize_glob(pattern):
    """Accept glob patterns copied through Markdown-rendering chat clients.

    Some clients insert a backslash before underscores or glob metacharacters
    when copying inline commands.  ``fnmatch`` treats that backslash literally,
    while Slurm job names do not contain it.  Only remove escapes that cannot
    express a useful literal job-name character here.
    """
    return re.sub(r"\\([*?_\[\]])", r"\1", pattern)


def _linear_percentile(values, percentile):
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def metric_summary(rows, value_getter):
    valued_rows = []
    for row in rows:
        value = value_getter(row)
        if value is not None and math.isfinite(value):
            valued_rows.append((float(value), row))
    if not valued_rows:
        return None
    valued_rows.sort(key=lambda pair: (pair[0], pair[1]["task_id"]))
    values = [value for value, _row in valued_rows]
    median = statistics.median(values)
    p90 = _linear_percentile(values, 0.9)

    def nearest(target):
        return min(valued_rows, key=lambda pair: (abs(pair[0] - target), pair[1]["task_id"]))[1]

    return {
        "count": len(values),
        "minimum": values[0],
        "minimum_row": valued_rows[0][1],
        "median": median,
        "median_row": nearest(median),
        "p90": p90,
        "p90_row": nearest(p90),
        "maximum": values[-1],
        "maximum_row": valued_rows[-1][1],
    }


def completed_rows(rows):
    return [row for row in rows if row["state"] == "COMPLETED"]


def terminal_rows(rows):
    return [row for row in rows if row["state"] not in ACTIVE_STATES]


def memory_efficiency(row):
    used, requested = row.get("max_rss_bytes"), row.get("requested_memory_bytes")
    return 100.0 * used / requested if used is not None and requested else None


def cpu_efficiency(row):
    cpu, elapsed, cpus = row.get("total_cpu_seconds"), row.get("elapsed_seconds"), row.get("alloc_cpus")
    return 100.0 * cpu / (elapsed * cpus) if cpu is not None and elapsed and cpus else None


def timelimit_utilization(row):
    elapsed, minutes = row.get("elapsed_seconds"), row.get("timelimit_minutes")
    return 100.0 * elapsed / (minutes * 60) if elapsed and minutes else None


def format_duration(seconds):
    if seconds is None:
        return "—"
    whole_seconds = int(seconds)
    days, remainder = divmod(whole_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    formatted = f"{days}-" if days else ""
    formatted += f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return formatted if seconds == whole_seconds else formatted + f".{int(round((seconds - whole_seconds) * 10))}"


def format_bytes(value):
    if value is None:
        return "—"
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    number = float(value)
    for unit in units:
        if abs(number) < 1024 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024


def format_percent(value):
    return "—" if value is None else f"{value:.1f}%"


def get_matching_jobs(pattern):
    """Return live squeue jobs whose names match the glob pattern."""
    cmd = ["squeue", "--noheader", "--format=%F|%i|%j|%T|%M", "--me"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    matching_jobs = []
    for line in result.stdout.splitlines():
        parts = line.strip().split("|", maxsplit=4)
        if len(parts) != 5:
            continue
        array_job_id, job_id, job_name, job_state, runtime = parts
        if fnmatch.fnmatch(job_name, pattern):
            matching_jobs.append(
                {
                    "array_job_id": array_job_id,
                    "job_id": job_id,
                    "job_name": job_name,
                    "job_state": job_state,
                    "runtime": runtime,
                }
            )
    return matching_jobs


def act_on_jobs(job_ids, mode):
    if mode == "cancel":
        command, verb = "scancel", "Cancelling"
    elif mode == "hold":
        command, verb = "scontrol", "Holding"
    elif mode == "resume":
        command, verb = "scontrol", "Releasing"
    elif mode == "status":
        return
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for job_id in job_ids:
        print(f"{verb} job {job_id}")
        if mode == "cancel":
            subprocess.run([command, job_id], check=False)
        else:
            action = "hold" if mode == "hold" else "release"
            subprocess.run([command, action, job_id], check=False)


def _state_counts(rows):
    return Counter(row["state"] for row in rows)


def _array_overview(rows):
    terminal = terminal_rows(rows)
    completed = completed_rows(rows)
    oom = [row for row in rows if row["state"] == "OUT_OF_MEMORY"]
    total_cpu_seconds = sum(
        row["total_cpu_seconds"]
        for row in rows
        if row.get("total_cpu_seconds") is not None
    )
    return {
        "total": len(rows),
        "terminal": len(terminal),
        "completed": len(completed),
        "completion_rate": 100.0 * len(completed) / len(terminal) if terminal else None,
        "oom": len(oom),
        "oom_rate": 100.0 * len(oom) / len(terminal) if terminal else None,
        "total_cpu_hours": total_cpu_seconds / 3600,
    }


COMPACT_BAR_WIDTH = 16
COMPACT_FAILURE_PRIORITY = (
    "OUT_OF_MEMORY",
    "FAILED",
    "TIMEOUT",
    "CANCELLED",
)
COMPACT_RUNNING_STATES = {"RUNNING", "CONFIGURING", "COMPLETING"}
ARRAY_CAUSE_LIMIT = 3
ARTIFACT_READ_LIMIT = 1024 * 1024
ETA_MIN_COMPLETED_SAMPLES = 3
ETA_LOG_SIGMA_MIN = 0.1
ETA_LOG_SIGMA_MAX = 2.0
ETA_LOG_RUNTIME_BOUND_MARGIN = 2.0
ETA_SIMULATION_DRAWS = 4096
ETA_SIMULATION_SEED = 0


def _fit_censored_lognormal(completed_runtimes, censored_elapsed):
    """Fit lognormal runtimes with running tasks as right-censored data."""
    completed_logs = np.log(np.asarray(completed_runtimes, dtype=float))
    censored_logs = np.log(np.asarray(censored_elapsed, dtype=float))
    observed_logs = (
        np.concatenate((completed_logs, censored_logs))
        if censored_logs.size
        else completed_logs
    )
    initial_mu = float(np.median(completed_logs))
    initial_sigma = float(np.std(completed_logs, ddof=1))
    initial_sigma = min(
        max(initial_sigma, 0.25, ETA_LOG_SIGMA_MIN),
        ETA_LOG_SIGMA_MAX,
    )

    def negative_log_likelihood(parameters):
        mu, log_sigma = parameters
        sigma = np.exp(log_sigma)
        completed_z = (completed_logs - mu) / sigma
        value = completed_logs.size * log_sigma + 0.5 * np.sum(completed_z**2)
        if censored_logs.size:
            censored_z = (censored_logs - mu) / sigma
            value -= np.sum(log_ndtr(-censored_z))
        return float(value)

    mu_bounds = (
        float(np.min(observed_logs) - ETA_LOG_RUNTIME_BOUND_MARGIN),
        float(np.max(observed_logs) + ETA_LOG_RUNTIME_BOUND_MARGIN),
    )
    fit = minimize(
        negative_log_likelihood,
        (initial_mu, math.log(initial_sigma)),
        method="L-BFGS-B",
        bounds=(
            mu_bounds,
            (math.log(ETA_LOG_SIGMA_MIN), math.log(ETA_LOG_SIGMA_MAX)),
        ),
    )
    if not fit.success or not np.all(np.isfinite(fit.x)) or not math.isfinite(float(fit.fun)):
        return None
    mu = float(fit.x[0])
    sigma = math.exp(float(fit.x[1]))
    median_runtime = math.exp(mu)
    if not all(math.isfinite(value) and value > 0 for value in (sigma, median_runtime)):
        return None
    return {
        "mu": mu,
        "sigma": sigma,
        "median_runtime_seconds": median_runtime,
    }


def _conditional_lognormal_median_remaining(elapsed, model):
    """Return median residual runtime conditional on surviving to ``elapsed``."""
    return _joint_running_makespan_quantile([elapsed], model, 0.5)


def _conditional_lognormal_completion_cdf(remaining, elapsed, model):
    """CDF of residual life conditional on a task surviving to ``elapsed``."""
    if remaining <= 0:
        return 0.0
    mu, sigma = model["mu"], model["sigma"]
    future = elapsed + remaining
    if not math.isfinite(future) or future <= 0:
        return None
    elapsed_log_survival = (
        0.0
        if elapsed <= 0
        else float(log_ndtr(-(math.log(elapsed) - mu) / sigma))
    )
    future_log_survival = float(log_ndtr(-(math.log(future) - mu) / sigma))
    log_survival_ratio = min(0.0, future_log_survival - elapsed_log_survival)
    probability = -math.expm1(log_survival_ratio)
    if not math.isfinite(probability):
        return None
    return min(max(probability, 0.0), 1.0)


def _joint_running_makespan_quantile(running_elapsed, model, probability):
    """Quantile of the maximum conditional residual time for running tasks."""
    if not running_elapsed or not 0 < probability < 1:
        return None

    def objective(remaining):
        log_probability = 0.0
        for elapsed in running_elapsed:
            task_probability = _conditional_lognormal_completion_cdf(
                remaining, elapsed, model
            )
            if task_probability is None:
                raise ValueError("nonfinite conditional completion probability")
            if task_probability <= 0:
                return -math.inf
            log_probability += math.log(task_probability)
        return log_probability - math.log(probability)

    upper = max(float(model["median_runtime_seconds"]), 1.0)
    try:
        for _ in range(100):
            value = objective(upper)
            if math.isfinite(value) and value >= 0:
                quantile = float(brentq(objective, 0.0, upper))
                return quantile if math.isfinite(quantile) and quantile > 0 else None
            upper *= 2.0
            if not math.isfinite(upper):
                return None
    except (ArithmeticError, RuntimeError, ValueError):
        return None
    return None


def _sample_conditional_lognormal_residuals(rng, elapsed, model, size):
    """Sample lognormal residual life conditional on survival to ``elapsed``."""
    mu, sigma = model["mu"], model["sigma"]
    elapsed_log_survival = (
        0.0
        if elapsed <= 0
        else float(log_ndtr(-(math.log(elapsed) - mu) / sigma))
    )
    uniforms = rng.random(size)
    target_log_survival = elapsed_log_survival + np.log1p(-uniforms)
    z_values = -ndtri_exp(target_log_survival)
    totals = np.exp(mu + sigma * z_values)
    residuals = totals - elapsed
    if not np.all(np.isfinite(residuals)) or np.any(residuals <= 0):
        return None
    return residuals


def _simulate_array_makespan_quantiles(
    running_elapsed,
    pending_count,
    model,
    *,
    draws=ETA_SIMULATION_DRAWS,
    seed=ETA_SIMULATION_SEED,
):
    """Simulate p50/p90 makespan while assigning pending tasks in waves."""
    if not running_elapsed or pending_count <= 0 or draws <= 0:
        return None
    try:
        rng = np.random.default_rng(seed)
        slots = []
        for elapsed in running_elapsed:
            residuals = _sample_conditional_lognormal_residuals(
                rng, elapsed, model, draws
            )
            if residuals is None:
                return None
            slots.append(residuals)
        slot_available = np.column_stack(slots)
        draw_indices = np.arange(draws)
        for _ in range(pending_count):
            earliest_slots = np.argmin(slot_available, axis=1)
            pending_durations = rng.lognormal(model["mu"], model["sigma"], size=draws)
            slot_available[draw_indices, earliest_slots] += pending_durations
        makespans = np.max(slot_available, axis=1)
        p50, p90 = (float(value) for value in np.quantile(makespans, (0.5, 0.9)))
    except (ArithmeticError, RuntimeError, ValueError):
        return None
    if not all(math.isfinite(value) and value > 0 for value in (p50, p90)):
        return None
    return p50, p90


def estimate_array_eta(rows):
    """Estimate remaining array wall time with a censored lognormal model.

    The estimate assumes that the array keeps its current number of running
    slots. Pending tasks are assigned to the earliest available slot in waves.
    """
    result = {
        "status": "unavailable",
        "remaining_seconds": None,
        "remaining_p50_seconds": None,
        "remaining_p90_seconds": None,
        "median_runtime_seconds": None,
        "sample_count": 0,
        "slot_count": 0,
        "model": None,
        "censored_count": 0,
    }
    if len(rows) <= 1:
        result["status"] = "scalar"
        return result

    active = [row for row in rows if row.get("state") in ACTIVE_STATES]
    if not active:
        result["status"] = "terminal"
        return result
    if any(row.get("held", False) or row.get("state") == "SUSPENDED" for row in active):
        result["status"] = "blocked"
        return result

    running = [row for row in active if row.get("state") in COMPACT_RUNNING_STATES]
    result["slot_count"] = len(running)
    if not running:
        result["status"] = "waiting"
        return result

    completed_runtimes = []
    for row in completed_rows(rows):
        elapsed = row.get("elapsed_seconds")
        if elapsed is None:
            continue
        try:
            elapsed = float(elapsed)
        except (TypeError, ValueError):
            continue
        if math.isfinite(elapsed) and elapsed > 0:
            completed_runtimes.append(elapsed)
    result["sample_count"] = len(completed_runtimes)
    if len(completed_runtimes) < ETA_MIN_COMPLETED_SAMPLES:
        result["status"] = "learning"
        return result

    censored_elapsed = []
    normalized_running_elapsed = []
    for row in running:
        elapsed = row.get("elapsed_seconds")
        try:
            elapsed = float(elapsed) if elapsed is not None else 0.0
        except (TypeError, ValueError):
            elapsed = 0.0
        if not math.isfinite(elapsed) or elapsed < 0:
            elapsed = 0.0
        normalized_running_elapsed.append(elapsed)
        if elapsed > 0:
            censored_elapsed.append(elapsed)
    result["censored_count"] = len(censored_elapsed)

    model = _fit_censored_lognormal(
        completed_runtimes,
        censored_elapsed,
    )
    model_values = (
        (model or {}).get("mu"),
        (model or {}).get("sigma"),
        (model or {}).get("median_runtime_seconds"),
    )
    if (
        model is None
        or not all(
            value is not None and math.isfinite(value) and value > 0
            for value in model_values[1:]
        )
        or model_values[0] is None
        or not math.isfinite(model_values[0])
    ):
        result["status"] = "model_unavailable"
        return result

    median_runtime = model["median_runtime_seconds"]
    result["median_runtime_seconds"] = median_runtime
    result["model"] = "censored_lognormal"
    pending_count = sum(row.get("state") == "PENDING" for row in active)
    if pending_count:
        quantiles = _simulate_array_makespan_quantiles(
            normalized_running_elapsed, pending_count, model
        )
    else:
        p50 = _joint_running_makespan_quantile(
            normalized_running_elapsed, model, 0.5
        )
        p90 = _joint_running_makespan_quantile(
            normalized_running_elapsed, model, 0.9
        )
        quantiles = (p50, p90) if p50 is not None and p90 is not None else None
    if quantiles is None:
        result["status"] = "model_unavailable"
        return result

    result["status"] = "estimated"
    result["remaining_p50_seconds"], result["remaining_p90_seconds"] = quantiles
    result["remaining_seconds"] = result["remaining_p50_seconds"]
    return result


def format_array_eta(eta, *, now=None, detailed=False):
    """Format p50/p90 ETA durations, optionally with local completion clocks."""
    status = (eta or {}).get("status", "unavailable")
    status_labels = {
        "blocked": "blocked",
        "learning": "learning",
        "model_unavailable": "unavailable",
        "waiting": "waiting",
    }
    if status != "estimated":
        return status_labels.get(status, "—")

    p50 = eta.get("remaining_p50_seconds", eta.get("remaining_seconds"))
    p90 = eta.get("remaining_p90_seconds")
    if not all(
        value is not None and math.isfinite(value) and value > 0
        for value in (p50, p90)
    ):
        return "—"
    if not detailed:
        return f"P50 {format_duration(p50)} / P90 {format_duration(p90)}"
    now = now or datetime.now().astimezone()
    if now.tzinfo is None:
        now = now.astimezone()

    def quantile_text(label, seconds):
        completion = now + timedelta(seconds=seconds)
        clock = (
            completion.strftime("%H:%M")
            if completion.date() == now.date()
            else completion.strftime("%b %d %H:%M")
        )
        return f"{label} ~{format_duration(seconds)} · {clock}"

    return f"{quantile_text('P50', p50)} / {quantile_text('P90', p90)}"


def format_array_eta_basis(eta):
    """Describe the runtime sample and concurrency behind an ETA."""
    if not eta or eta.get("status") != "estimated":
        return ""
    samples = eta["sample_count"]
    sample_label = "sample" if samples == 1 else "samples"
    slots = eta["slot_count"]
    slot_label = "slot" if slots == 1 else "slots"
    censored = eta.get("censored_count", 0)
    censored_label = "task" if censored == 1 else "tasks"
    return (
        f"censored lognormal, median {format_duration(eta['median_runtime_seconds'])}, "
        f"{samples} completed {sample_label}, {censored} running {censored_label}, "
        f"{slots} active {slot_label}"
    )


def recent_job_groups(rows, limit=20):
    """Return the newest scalar/array jobs, with one group per Slurm job ID."""
    groups = matching_arrays(rows, "*")
    ordered = sorted(groups.items(), key=lambda item: int(item[0][0]), reverse=True)
    return ordered[:limit]


def _compact_state_counts(rows):
    states = _state_counts(rows)
    completed = states.get("COMPLETED", 0)
    running = sum(states.get(state, 0) for state in COMPACT_RUNNING_STATES)
    pending = sum(states.get(state, 0) for state in ACTIVE_STATES - COMPACT_RUNNING_STATES)
    failed = len(rows) - completed - running - pending
    return {
        "completed": completed,
        "failed": failed,
        "running": running,
        "pending": pending,
    }


def _compact_overall_state(rows):
    states = _state_counts(rows)
    if states.get("RUNNING") or states.get("COMPLETING"):
        return "RUNNING"
    if states.get("CONFIGURING"):
        return "CONFIGURING"
    if states.get("PENDING"):
        return "PENDING"
    if states.get("SUSPENDED"):
        return "SUSPENDED"
    if states.get("COMPLETED") == len(rows):
        return "COMPLETED"
    for state in COMPACT_FAILURE_PRIORITY:
        if states.get(state):
            return state
    return next(iter(states), "UNKNOWN")


def _compact_bar_widths(counts, width=COMPACT_BAR_WIDTH):
    keys = ("completed", "failed", "running", "pending")
    total = sum(counts[key] for key in keys)
    if total <= 0:
        return {key: 0 for key in keys}
    active_keys = [key for key in keys if counts[key] > 0]
    if width < len(active_keys):
        return {key: int(key in active_keys[:width]) for key in keys}
    widths = {key: int(key in active_keys) for key in keys}
    distributable = width - len(active_keys)
    exact = {key: distributable * counts[key] / total for key in keys}
    for key in keys:
        widths[key] += int(math.floor(exact[key]))
    remaining = width - sum(widths.values())
    ranked = sorted(
        keys,
        key=lambda key: (
            exact[key] - math.floor(exact[key]),
            counts[key],
            -keys.index(key),
        ),
        reverse=True,
    )
    for key in ranked[:remaining]:
        widths[key] += 1
    return widths


def compact_job_summary(array_job_id, job_name, rows):
    counts = _compact_state_counts(rows)
    elapsed_candidates = [
        row.get("job_elapsed_seconds")
        for row in rows
        if row.get("job_elapsed_seconds") is not None
    ]
    if not elapsed_candidates:
        elapsed_candidates = [
            row.get("elapsed_seconds")
            for row in rows
            if row.get("elapsed_seconds") is not None
        ]
    finished = counts["completed"] + counts["failed"]
    return {
        "job_id": str(array_job_id),
        "job_name": job_name,
        "state": _compact_overall_state(rows),
        "total": len(rows),
        "finished": finished,
        "counts": counts,
        "bar_widths": _compact_bar_widths(counts),
        "elapsed_seconds": max(elapsed_candidates) if elapsed_candidates else None,
        "eta": estimate_array_eta(rows),
    }


def _cause_label(row):
    parts = [row.get("state") or "UNKNOWN"]
    reason = str(row.get("reason") or "").strip()
    if reason and reason not in {"None", "N/A", "Unknown"}:
        parts.append(reason)
    exit_code = str(row.get("exit_code") or "").strip()
    if exit_code and exit_code not in {"0:0", "N/A"}:
        parts.append(exit_code)
    return " / ".join(parts)


def _rank_causes(counter, limit=ARRAY_CAUSE_LIMIT):
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return {
        "items": ranked[:limit],
        "remaining": max(0, len(ranked) - limit),
    }


def array_overview_summary(rows, cause_limit=ARRAY_CAUSE_LIMIT, *, eta=None):
    """Build the compact status information shown above an array's tasks."""
    overview = _array_overview(rows)
    states = _state_counts(rows)
    failures = Counter(
        _cause_label(row)
        for row in terminal_rows(rows)
        if row.get("state") != "COMPLETED"
    )
    waiting = Counter(
        str(row.get("reason") or row.get("state") or "UNKNOWN")
        for row in rows
        if row.get("state") in (ACTIVE_STATES - COMPACT_RUNNING_STATES)
    )
    completed = completed_rows(rows)
    return {
        **overview,
        "states": sorted(states.items(), key=lambda item: (-item[1], item[0])),
        "failures": _rank_causes(failures, cause_limit),
        "waiting": _rank_causes(waiting, cause_limit),
        "runtime": metric_summary(completed, lambda row: row.get("elapsed_seconds")),
        "max_rss": metric_summary(completed, lambda row: row.get("max_rss_bytes")),
        "eta": eta if eta is not None else estimate_array_eta(rows),
    }


def _scheduler_value(value):
    text = str(value or "").strip().strip('"')
    return None if text in {"", "None", "N/A", "(null)", "Unknown"} else text


def _first_scheduler_value(*values):
    return next((value for value in values if _scheduler_value(value) is not None), None)


def _expand_slurm_path(value, task):
    text = _scheduler_value(value)
    if text is None:
        return None
    replacements = {
        "%A": str(task.get("array_job_id") or task.get("job_id") or ""),
        "%a": str(task.get("task_id") if task.get("task_id") is not None else 0),
        "%j": str(task.get("job_id") or ""),
        "%x": str(task.get("job_name") or ""),
    }
    for placeholder, replacement in replacements.items():
        text = text.replace(placeholder, replacement)
    return text


def _script_from_submit_line(submit_line):
    text = _scheduler_value(submit_line)
    if text is None:
        return None
    try:
        tokens = shlex.split(text)
    except ValueError:
        return None
    if not tokens or Path(tokens[0]).name != "sbatch" or len(tokens) < 2:
        return None
    candidate = tokens[-1]
    return None if candidate.startswith("-") else candidate


def _resolve_job_path(value, task, work_dir=None):
    expanded = _expand_slurm_path(value, task)
    if expanded is None:
        return None
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        base = Path(_expand_slurm_path(work_dir, task) or Path.cwd()).expanduser()
        path = base / path
    return path.resolve(strict=False)


def resolve_task_artifacts(task, details=None):
    """Resolve stdout, stderr, and submitted-script paths without a shell."""
    details = details or {}
    work_dir = _first_scheduler_value(details.get("WorkDir"), task.get("work_dir"))
    stdout = _first_scheduler_value(details.get("StdOut"), task.get("stdout_path"))
    stderr = _first_scheduler_value(details.get("StdErr"), task.get("stderr_path"))
    command = _first_scheduler_value(details.get("Command"), task.get("command"))
    if _scheduler_value(command) is None:
        command = _script_from_submit_line(details.get("SubmitLine") or task.get("submit_line"))
    return {
        "stdout": _resolve_job_path(stdout, task, work_dir),
        "stderr": _resolve_job_path(stderr, task, work_dir),
        "script": _resolve_job_path(command, task, work_dir),
    }


def prepare_local_task_execution(task, details=None):
    """Resolve and validate a selected task's local execution specification."""
    details = details or {}
    work_dir_value = _first_scheduler_value(
        details.get("WorkDir"), task.get("work_dir")
    )
    work_dir = (
        _resolve_job_path(work_dir_value, task)
        if work_dir_value is not None
        else Path.cwd().resolve()
    )
    script = resolve_task_artifacts(task, details)["script"]
    if script is None:
        raise ValueError("The submitted script path is unavailable for this task.")
    if not script.is_file():
        raise FileNotFoundError(f"Submitted script is not a readable file: {script}")
    if not os.access(script, os.X_OK):
        raise PermissionError(f"Submitted script is not executable: {script}")
    if work_dir is None or not work_dir.is_dir():
        raise NotADirectoryError(f"Task working directory is unavailable: {work_dir}")

    environment = os.environ.copy()
    for variable in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        environment.pop(variable, None)
    environment["SLURM_JOB_ID"] = str(task["job_id"])
    task_id = task.get("task_id")
    if task_id is not None:
        environment["SLURM_ARRAY_JOB_ID"] = str(
            task.get("array_job_id") or str(task["job_id"]).split("_", 1)[0]
        )
        environment["SLURM_ARRAY_TASK_ID"] = str(task_id)
    return {
        "task_id": str(task["job_id"]),
        "task_state": str(task.get("state") or "UNKNOWN"),
        "script": script,
        "work_dir": work_dir,
        "environment": environment,
    }


def run_local_task_script(execution):
    """Run a prepared task script and return its exit code and combined log."""
    safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", execution["task_id"])
    with tempfile.NamedTemporaryFile(
        prefix=f"smanage-{safe_task_id}-",
        suffix=".log",
        delete=False,
    ) as output:
        log_path = Path(output.name)
        result = subprocess.run(
            [str(execution["script"])],
            cwd=execution["work_dir"],
            env=execution["environment"],
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "task_id": execution["task_id"],
        "returncode": result.returncode,
        "log_path": log_path,
    }


def read_artifact(path, *, tail, limit=ARTIFACT_READ_LIMIT):
    """Read a bounded head or tail of an artifact with replacement decoding."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Artifact is not a readable file: {path}")
    size = path.stat().st_size
    with path.open("rb") as handle:
        if tail and size > limit:
            handle.seek(size - limit)
        data = handle.read(limit)
    truncated = size > limit
    return {
        "path": path,
        "size": size,
        "text": data.decode("utf-8", errors="replace"),
        "truncated": truncated,
        "tail": tail,
    }


def _compact_breakdown(summary):
    counts = summary["counts"]
    return (
        f"✓{counts['completed']} ✗{counts['failed']} "
        f"▶{counts['running']} …{counts['pending']}"
    )


def _plain_compact_bar(summary):
    widths = summary["bar_widths"]
    return "[" + "".join(
        (
            "#" * widths["completed"],
            "!" * widths["failed"],
            ">" * widths["running"],
            "." * widths["pending"],
        )
    ) + "]"


def _rich_compact_bar(summary):
    widths = summary["bar_widths"]
    bar = Text("[")
    for key, symbol, style in (
        ("completed", "█", "green"),
        ("failed", "█", "bold red"),
        ("running", "█", "cyan"),
        ("pending", "░", "yellow"),
    ):
        bar.append(symbol * widths[key], style=style)
    bar.append("]")
    return bar


def render_rich_compact_status(groups, starttime, endtime=None, console=None):
    console = console or Console()
    table = Table(
        title=f"Recent Slurm jobs · {starttime} → {endtime or 'now'}",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        pad_edge=False,
    )
    table.add_column("Job", justify="right", no_wrap=True)
    table.add_column("Name", max_width=38, overflow="ellipsis", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("Progress", no_wrap=True)
    table.add_column("Done", justify="right", no_wrap=True)
    table.add_column("Tasks", no_wrap=True)
    table.add_column("Elapsed", justify="right", no_wrap=True)
    table.add_column("ETA", justify="right", no_wrap=True)
    for (array_job_id, job_name), rows in groups:
        summary = compact_job_summary(array_job_id, job_name, rows)
        table.add_row(
            summary["job_id"],
            summary["job_name"],
            Text(summary["state"], style=STATE_STYLES.get(summary["state"], "white")),
            _rich_compact_bar(summary),
            f"{summary['finished']}/{summary['total']}",
            _compact_breakdown(summary),
            format_duration(summary["elapsed_seconds"]),
            format_array_eta(summary["eta"]),
        )
    console.print(table)
    console.print("[dim]█ completed  █ failed  █ running  ░ pending[/dim]")


def _shorten(text, width):
    text = str(text)
    return text if len(text) <= width else text[: width - 1] + "…"


def render_plain_compact_status(groups, starttime, endtime=None):
    print(f"Recent Slurm jobs ({starttime} -> {endtime or 'now'})")
    print(
        f"{'JOB':>10}  {'NAME':<30}  {'STATE':<13}  {'PROGRESS':<18}  "
        f"{'DONE':>7}  {'TASKS':<19}  {'ELAPSED':>10}  {'ETA':>22}"
    )
    for (array_job_id, job_name), rows in groups:
        summary = compact_job_summary(array_job_id, job_name, rows)
        print(
            f"{summary['job_id']:>10}  {_shorten(summary['job_name'], 30):<30}  "
            f"{summary['state']:<13}  {_plain_compact_bar(summary):<18}  "
            f"{summary['finished']:>3}/{summary['total']:<3}  "
            f"{_compact_breakdown(summary):<19}  "
            f"{format_duration(summary['elapsed_seconds']):>10}  "
            f"{format_array_eta(summary['eta']):>22}"
        )
    print("# completed  ! failed  > running  . pending")


def render_compact_status(groups, starttime, endtime=None, console=None):
    if RICH_AVAILABLE:
        render_rich_compact_status(groups, starttime, endtime, console=console)
    else:
        render_plain_compact_status(groups, starttime, endtime)


def build_tui_snapshot():
    """Return the current user's 20 newest jobs with live queue overlays."""
    live_rows = get_live_queue_rows()
    accounting_rows = get_accounting_rows(starttime="now-7days")
    tasks_by_id = {row["job_id"]: dict(row) for row in accounting_rows}

    for live in live_rows:
        task = tasks_by_id.setdefault(live["job_id"], {})
        task.update(live)

    for task in tasks_by_id.values():
        task.setdefault("partition", "")
        task.setdefault("reason", "")
        task.setdefault("submit_time", "")
        task.setdefault("start_time", "")
        task.setdefault("time_left_seconds", None)
        task.setdefault("held", False)
        if task.get("timelimit_seconds") is None:
            minutes = task.get("timelimit_minutes")
            task["timelimit_seconds"] = minutes * 60 if minutes is not None else None
        task.setdefault("total_cpu_seconds", None)
        task.setdefault("max_rss_bytes", None)
        task.setdefault("requested_memory_bytes", None)
        task.setdefault("req_mem", None)
        task.setdefault("exit_code", "")
        task.setdefault("job_elapsed_seconds", None)

    groups = recent_job_groups(list(tasks_by_id.values()), limit=20)
    jobs = []
    for (base_id, job_name), grouped_tasks in groups:
        tasks = sorted(
            grouped_tasks,
            key=lambda row: (
                row.get("task_id") is None,
                row.get("task_id") if row.get("task_id") is not None else -1,
            ),
        )
        live_for_job = [row for row in live_rows if row["array_job_id"] == base_id]
        summary = compact_job_summary(base_id, job_name, tasks)
        reasons = sorted({row.get("reason", "") for row in tasks if row.get("reason")})
        partitions = sorted({row.get("partition", "") for row in tasks if row.get("partition")})
        time_limits = [
            row.get("timelimit_seconds")
            for row in tasks
            if row.get("timelimit_seconds") is not None
        ]
        time_left = [
            row.get("time_left_seconds")
            for row in tasks
            if row.get("time_left_seconds") is not None
        ]
        jobs.append(
            {
                **summary,
                "tasks": tasks,
                "progress_bar": _plain_compact_bar(summary),
                "breakdown": _compact_breakdown(summary),
                "partition": ",".join(partitions),
                "reason": ", ".join(reasons),
                "timelimit_seconds": max(time_limits) if time_limits else None,
                "time_left_seconds": max(time_left) if time_left else None,
                "alloc_cpus": sum(row.get("alloc_cpus") or 0 for row in tasks),
                "nodes": sum(row.get("nodes") or 0 for row in tasks),
                "held": any(row.get("held", False) for row in tasks),
                "has_running": any(row["state"] in COMPACT_RUNNING_STATES for row in tasks),
                "has_pending": any(row["state"] == "PENDING" for row in tasks),
            }
        )
    return {"jobs": jobs, "refreshed_at": datetime.now().astimezone().isoformat()}


def parse_scontrol_job_details(output):
    """Parse ``scontrol show job -o`` while retaining its raw record."""
    text = output.strip()
    fields = {}
    for match in re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(.*?)(?=\s[A-Za-z][A-Za-z0-9_]*=|$)", text):
        fields[match.group(1)] = match.group(2).strip()
    fields["_raw"] = text
    return fields


def get_job_details(target):
    result = subprocess.run(
        ["scontrol", "show", "job", "-o", str(target)],
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_scontrol_job_details(result.stdout)


def build_scheduler_command(action, target):
    target = str(target)
    if action == "cancel":
        return ["scancel", target]
    if action == "hold":
        return ["scontrol", "hold", target]
    if action in {"resume", "release"}:
        return ["scontrol", "release", target]
    raise ValueError(f"Unsupported scheduler action: {action}")


def scheduler_action_allowed(action, selection):
    if not selection:
        return False, "No job or task is selected."
    state = selection.get("state", "UNKNOWN")
    if action == "cancel":
        return (state in ACTIVE_STATES, "Only active jobs and tasks can be cancelled.")
    if action == "hold":
        if selection.get("has_running") or state in COMPACT_RUNNING_STATES:
            return False, "Hold does not pause a running Slurm job."
        if selection.get("held"):
            return False, "The selected job or task is already held."
        return (state == "PENDING", "Only pending jobs and tasks can be held.")
    if action in {"resume", "release"}:
        return (bool(selection.get("held")), "The selected job or task is not held.")
    return False, f"Unsupported scheduler action: {action}"


def run_scheduler_action(action, target):
    return subprocess.run(
        build_scheduler_command(action, target),
        capture_output=True,
        text=True,
        check=False,
    )


def _metric_specs():
    return (
        ("Runtime", lambda row: row.get("elapsed_seconds"), format_duration),
        ("Max RSS", lambda row: row.get("max_rss_bytes"), format_bytes),
        ("Memory efficiency", memory_efficiency, format_percent),
        ("CPU efficiency", cpu_efficiency, format_percent),
        ("Time-limit utilization", timelimit_utilization, format_percent),
    )


def _rich_live_table(live_jobs):
    table = Table(title="Live queue", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Job ID", justify="right", no_wrap=True)
    table.add_column("Job name", overflow="fold")
    table.add_column("State", no_wrap=True)
    table.add_column("Runtime", justify="right", no_wrap=True)
    for job in live_jobs:
        state = job["job_state"]
        table.add_row(
            job["job_id"],
            job["job_name"],
            Text(state, style=STATE_STYLES.get(state, "white")),
            job["runtime"],
        )
    if not live_jobs:
        table.add_row("—", "No matching jobs currently in squeue", "—", "—")
    return table


def _rich_state_table(rows):
    counts = _state_counts(rows)
    table = Table(title="Task states", box=box.SIMPLE_HEAVY, header_style="bold")
    table.add_column("State")
    table.add_column("Count", justify="right")
    table.add_column("Percent", justify="right")
    total = len(rows)
    for state, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        table.add_row(
            Text(state, style=STATE_STYLES.get(state, "white")),
            str(count),
            format_percent(100.0 * count / total if total else None),
        )
    return table


def _rich_metrics_table(rows):
    completed = completed_rows(rows)
    table = Table(title="Completed-task metrics", box=box.ROUNDED, header_style="bold green")
    table.add_column("Metric")
    table.add_column("Minimum", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("P90", justify="right")
    table.add_column("Maximum", justify="right")
    table.add_column("Coverage", justify="right")
    for label, getter, formatter in _metric_specs():
        summary = metric_summary(completed, getter)
        if summary is None:
            table.add_row(label, "—", "—", "—", "—", f"0/{len(completed)}")
        else:
            table.add_row(
                label,
                formatter(summary["minimum"]),
                formatter(summary["median"]),
                formatter(summary["p90"]),
                formatter(summary["maximum"]),
                f"{summary['count']}/{len(completed)}",
            )
    return table


def _rich_extremes_table(rows, top=5):
    completed = completed_rows(rows)
    table = Table(title="Notable completed tasks", box=box.MINIMAL_DOUBLE_HEAD, header_style="bold")
    table.add_column("Category")
    table.add_column("Task", justify="right")
    table.add_column("Value", justify="right")

    runtime = metric_summary(completed, lambda row: row.get("elapsed_seconds"))
    if runtime is not None:
        for label, value_key, row_key in (
            ("Runtime minimum", "minimum", "minimum_row"),
            ("Runtime median (nearest)", "median", "median_row"),
            ("Runtime p90 (nearest)", "p90", "p90_row"),
            ("Runtime maximum", "maximum", "maximum_row"),
        ):
            table.add_row(label, runtime[row_key]["job_id"], format_duration(runtime[value_key]))

    for rank, row in enumerate(
        sorted(
            [row for row in completed if row.get("elapsed_seconds") is not None],
            key=lambda row: row["elapsed_seconds"],
            reverse=True,
        )[:top],
        start=1,
    ):
        table.add_row(
            f"Slowest #{rank}",
            row["job_id"],
            format_duration(row["elapsed_seconds"]),
        )

    for row in sorted(
        [row for row in completed if row.get("max_rss_bytes") is not None],
        key=lambda row: row["max_rss_bytes"],
        reverse=True,
    )[:top]:
        table.add_row("Highest memory", row["job_id"], format_bytes(row["max_rss_bytes"]))
    return table


def _rich_array_report(console, title, rows, *, show_eta=True):
    overview = _array_overview(rows)
    eta = estimate_array_eta(rows) if show_eta else None
    oom_style = "bold red" if overview["oom"] else "green"
    summary = Text()
    summary.append(f"Tasks {overview['total']}   ")
    summary.append(f"Completed {overview['completed']}/{overview['terminal']} terminal ")
    summary.append(f"({format_percent(overview['completion_rate'])})   ", style="green")
    summary.append(
        f"OOM {overview['oom']} ({format_percent(overview['oom_rate'])})   ",
        style=oom_style,
    )
    summary.append(f"CPU hours {overview['total_cpu_hours']:.2f}")
    if show_eta:
        summary.append(f"\nETA {format_array_eta(eta, detailed=True)}")
        eta_basis = format_array_eta_basis(eta)
        if eta_basis:
            summary.append(f"  ·  {eta_basis}", style="dim")
    console.print(Rule(title, style="bright_blue"))
    console.print(Panel(summary, border_style="blue"))
    console.print(_rich_state_table(rows))
    console.print(_rich_metrics_table(rows))
    console.print(_rich_extremes_table(rows))
    completed = completed_rows(rows)
    missing_rss = sum(row.get("max_rss_bytes") is None for row in completed)
    if missing_rss:
        console.print(
            f"[dim]MaxRSS unavailable for {missing_rss}/{len(completed)} completed tasks; "
            "missing values were excluded.[/dim]"
        )


def render_rich_status(pattern, live_jobs, groups, starttime, endtime=None, console=None):
    console = console or Console()
    window = f"{starttime} → {endtime or 'now'}"
    console.print(
        Panel.fit(
            f"[bold]{pattern}[/bold]\nAccounting window: {window}",
            title="Slurm status dashboard",
            border_style="bright_cyan",
        )
    )
    console.print(_rich_live_table(live_jobs))

    all_rows = []
    for (array_job_id, job_name), rows in sorted(
        groups.items(), key=lambda item: int(item[0][0])
    ):
        _rich_array_report(console, f"Array {array_job_id} · {job_name}", rows)
        all_rows.extend(rows)
    if len(groups) > 1:
        _rich_array_report(
            console,
            f"Combined · {len(groups)} arrays",
            all_rows,
            show_eta=False,
        )


def _plain_metric_line(rows, label, getter, formatter):
    summary = metric_summary(completed_rows(rows), getter)
    if summary is None:
        return f"    {label}: unavailable"
    return (
        f"    {label}: min={formatter(summary['minimum'])}, "
        f"median={formatter(summary['median'])}, p90={formatter(summary['p90'])}, "
        f"max={formatter(summary['maximum'])} ({summary['count']} values)"
    )


def render_plain_status(pattern, live_jobs, groups, starttime, endtime=None):
    print(f"Slurm status dashboard: {pattern} ({starttime} -> {endtime or 'now'})")
    print("Live jobs:")
    if live_jobs:
        for job in live_jobs:
            print(
                f"  {job['job_id']:>12}  {job['job_name']:<40}  "
                f"{job['job_state']:<12}  {job['runtime']:>12}"
            )
    else:
        print("  None")

    all_rows = []
    reports = list(sorted(groups.items(), key=lambda item: int(item[0][0])))
    for (array_job_id, job_name), rows in reports:
        overview = _array_overview(rows)
        eta = estimate_array_eta(rows)
        print(f"\nArray {array_job_id}  {job_name}")
        print("  states: " + ", ".join(f"{k}={v}" for k, v in sorted(_state_counts(rows).items())))
        print(
            f"  completed={overview['completed']}/{overview['terminal']} terminal "
            f"({format_percent(overview['completion_rate'])}), "
            f"OOM={overview['oom']} ({format_percent(overview['oom_rate'])}), "
            f"CPU hours={overview['total_cpu_hours']:.2f}"
        )
        eta_basis = format_array_eta_basis(eta)
        eta_text = format_array_eta(eta, detailed=True)
        print(f"  ETA={eta_text}" + (f" ({eta_basis})" if eta_basis else ""))
        for label, getter, formatter in _metric_specs():
            print(_plain_metric_line(rows, label, getter, formatter))
        all_rows.extend(rows)
    if len(reports) > 1:
        overview = _array_overview(all_rows)
        print("\nCombined matching arrays")
        print(f"  CPU hours={overview['total_cpu_hours']:.2f}")
        for label, getter, formatter in _metric_specs():
            print(_plain_metric_line(all_rows, label, getter, formatter))


def render_status(pattern, live_jobs, groups, starttime, endtime=None, console=None):
    if RICH_AVAILABLE:
        render_rich_status(pattern, live_jobs, groups, starttime, endtime, console=console)
    else:
        render_plain_status(pattern, live_jobs, groups, starttime, endtime)


REFRESH_INTERVAL_SECONDS = 2.0
TUI_STATE_COLORS = {
    "RUNNING": "bold cyan",
    "COMPLETING": "bold cyan",
    "CONFIGURING": "cyan",
    "PENDING": "bold yellow",
    "SUSPENDED": "yellow",
    "COMPLETED": "bold green",
    "FAILED": "bold red",
    "OUT_OF_MEMORY": "bold red",
    "TIMEOUT": "bold magenta",
    "CANCELLED": "dim yellow",
}


def styled_state(state: str) -> Text:
    return Text(state, style=TUI_STATE_COLORS.get(state, "white"))


def styled_progress(job: dict[str, Any]) -> Text:
    widths = job["bar_widths"]
    bar = Text("[")
    for key, symbol, style in (
        ("completed", "█", "green"),
        ("failed", "█", "bold red"),
        ("running", "█", "cyan"),
        ("pending", "░", "yellow"),
    ):
        bar.append(symbol * widths[key], style=style)
    bar.append("]")
    return bar


def _format_ranked_causes(ranked):
    if not ranked["items"]:
        return "none"
    text = " · ".join(f"{label} ×{count}" for label, count in ranked["items"])
    if ranked["remaining"]:
        text += f" · +{ranked['remaining']} more"
    return text


def format_array_overview(job: dict[str, Any] | None) -> Text:
    if not job:
        return Text("Select a job to view its array overview.", style="dim")
    summary = array_overview_summary(job["tasks"], eta=job.get("eta"))
    text = Text()
    text.append(f"{job['job_name']}  #{job['job_id']}  ", style="bold")
    text.append_text(styled_progress(job))
    text.append(f"  {job['breakdown']}  ·  {job['finished']}/{job['total']} finished\n")
    text.append(
        f"Success {format_percent(summary['completion_rate'])} of terminal  ·  "
        f"OOM {summary['oom']} ({format_percent(summary['oom_rate'])})  ·  "
        f"CPU {summary['total_cpu_hours']:.2f} h\n"
    )
    state_text = " · ".join(
        f"{state} {count} ({format_percent(100.0 * count / summary['total'])})"
        for state, count in summary["states"]
    ) if summary["total"] else "none"
    text.append(f"States: {state_text}\n")
    text.append(f"Failures: {_format_ranked_causes(summary['failures'])}\n", style="red")
    text.append(f"Waiting: {_format_ranked_causes(summary['waiting'])}\n", style="yellow")
    eta = summary["eta"]
    text.append(f"ETA {format_array_eta(eta, detailed=True)}")
    eta_basis = format_array_eta_basis(eta)
    if eta_basis:
        text.append(f"  ·  {eta_basis}", style="dim")
    text.append("\n")
    runtime = summary["runtime"]
    memory = summary["max_rss"]
    runtime_text = (
        f"median {format_duration(runtime['median'])}, p90 {format_duration(runtime['p90'])}"
        if runtime else "unavailable"
    )
    memory_text = (
        f"median {format_bytes(memory['median'])}, p90 {format_bytes(memory['p90'])}"
        if memory else "unavailable"
    )
    text.append(f"Completed runtime: {runtime_text}  ·  Max RSS: {memory_text}", style="dim")
    return text


def format_task_id(task: dict[str, Any]) -> str:
    task_id = task.get("task_id")
    return "—" if task_id is None else str(task_id)


def format_task_details(task: dict[str, Any] | None) -> str:
    if not task:
        return "[dim]Select a task to inspect it.[/dim]"
    state = task.get("state", "UNKNOWN")
    return "\n".join(
        (
            f"[b]{escape(task.get('job_name') or 'Slurm task')}[/b]",
            f"Task [b]#{escape(str(task.get('job_id') or '—'))}[/b]",
            f"[{TUI_STATE_COLORS.get(state, 'white')}]{state}[/]  "
            f"Reason [yellow]{escape(task.get('reason') or '—')}[/yellow]",
            "",
            f"Elapsed [b]{format_duration(task.get('elapsed_seconds'))}[/b]",
            f"Limit {format_duration(task.get('timelimit_seconds'))}",
            f"Remaining {format_duration(task.get('time_left_seconds'))}",
            "",
            f"CPUs {task.get('alloc_cpus') or '—'}  Nodes {task.get('nodes') or '—'}",
            f"Requested memory {format_bytes(task.get('requested_memory_bytes'))}",
            f"Maximum RSS {format_bytes(task.get('max_rss_bytes'))}",
            f"Total CPU {format_duration(task.get('total_cpu_seconds'))}",
            f"Exit status {escape(task.get('exit_code') or '—')}",
            "",
            f"Submitted {escape(task.get('submit_time') or '—')}",
            f"Started {escape(task.get('start_time') or '—')}",
        )
    )


def format_expanded_details(selection: dict[str, Any], details: dict[str, str]) -> str:
    labels = (
        ("Target", selection.get("job_id")),
        ("Name", selection.get("job_name")),
        ("State", selection.get("state")),
        ("Reason", selection.get("reason")),
        ("Elapsed", format_duration(selection.get("elapsed_seconds"))),
        ("Time limit", format_duration(selection.get("timelimit_seconds"))),
        ("Time left", format_duration(selection.get("time_left_seconds"))),
        ("CPUs", selection.get("alloc_cpus")),
        ("Nodes", selection.get("nodes")),
        ("Requested memory", format_bytes(selection.get("requested_memory_bytes"))),
        ("Max RSS", format_bytes(selection.get("max_rss_bytes"))),
        ("Total CPU", format_duration(selection.get("total_cpu_seconds"))),
        ("Exit code", selection.get("exit_code")),
        ("Submitted", selection.get("submit_time")),
        ("Started", selection.get("start_time")),
    )
    lines = [
        f"[b cyan]{escape(str(label)):<18}[/b cyan] {escape(str(value or '—'))}"
        for label, value in labels
    ]
    lines.append("\n[b]Scheduler record[/b]")
    preferred = (
        "Account", "QOS", "Partition", "Command", "WorkDir", "StdOut", "StdErr",
        "Dependency", "NodeList", "ReqNodeList", "NumCPUs", "ReqMem", "TRES",
    )
    for key in preferred:
        if details.get(key):
            lines.append(f"[b cyan]{key:<18}[/b cyan] {escape(details[key])}")
    if details.get("_unavailable"):
        lines.extend(
            (
                "\n[dim]Live scheduler details are no longer available for this task.[/dim]",
                f"[dim]{escape(details['_unavailable'])}[/dim]",
            )
        )
    return "\n".join(lines)


ModalResult = TypeVar("ModalResult")

BUTTON_CHOICE_BINDINGS = [
    Binding("left", "focus_previous_button", "Previous option", show=False, priority=True),
    Binding("up", "focus_previous_button", "Previous option", show=False, priority=True),
    Binding("right", "focus_next_button", "Next option", show=False, priority=True),
    Binding("down", "focus_next_button", "Next option", show=False, priority=True),
]


class ButtonChoiceScreen(ModalScreen[ModalResult]):
    """Modal screen with discoverable keyboard navigation between buttons."""

    AUTO_FOCUS = "Button"

    def action_focus_previous_button(self) -> None:
        self.focus_previous("Button")

    def action_focus_next_button(self) -> None:
        self.focus_next("Button")


class ConfirmCancelScreen(ButtonChoiceScreen[bool]):
    BINDINGS = [
        *BUTTON_CHOICE_BINDINGS,
        Binding("y", "confirm", "Yes"),
        Binding("n", "dismiss_no", "No"),
        Binding("escape", "dismiss_no", "No", show=False),
    ]

    CSS = """
    ConfirmCancelScreen { align: center middle; background: $background 70%; }
    #confirm-box { width: 64; height: auto; padding: 1 2; border: heavy $error; background: $surface; }
    #confirm-title { text-style: bold; color: $error; margin-bottom: 1; }
    #confirm-buttons { height: auto; align-horizontal: right; margin-top: 1; }
    #confirm-buttons Button { margin-left: 1; }
    #confirm-buttons Button:focus { border: heavy $accent; text-style: bold reverse; }
    """

    def __init__(self, target: str, description: str) -> None:
        super().__init__()
        self.target = target
        self.description = description

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Label("Cancel Slurm work?", id="confirm-title")
            yield Static(
                f"Cancel [b]{escape(self.description)}[/b]\n"
                f"Target: [yellow]{escape(self.target)}[/yellow]\n\nThis cannot be undone."
            )
            with Horizontal(id="confirm-buttons"):
                yield Button("Keep", id="keep", variant="default")
                yield Button("Cancel job", id="confirm", variant="error")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_dismiss_no(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")


class ConfirmExecuteScreen(ButtonChoiceScreen[bool]):
    BINDINGS = [
        *BUTTON_CHOICE_BINDINGS,
        Binding("y", "confirm", "Yes"),
        Binding("n", "dismiss_no", "No"),
        Binding("escape", "dismiss_no", "No", show=False),
    ]

    CSS = """
    ConfirmExecuteScreen { align: center middle; background: $background 70%; }
    #execute-confirm-box {
        width: 92%; height: auto; padding: 1 2;
        border: heavy $warning; background: $surface;
    }
    #execute-confirm-title { text-style: bold; color: $warning; margin-bottom: 1; }
    #execute-confirm-buttons { height: auto; align-horizontal: right; margin-top: 1; }
    #execute-confirm-buttons Button { margin-left: 1; }
    #execute-confirm-buttons Button:focus { border: heavy $accent; text-style: bold reverse; }
    """

    def __init__(self, execution: dict[str, Any]) -> None:
        super().__init__()
        self.execution = execution

    def compose(self) -> ComposeResult:
        environment = self.execution["environment"]
        variables = [
            f"SLURM_JOB_ID={environment['SLURM_JOB_ID']}",
            *(
                [
                    f"SLURM_ARRAY_JOB_ID={environment['SLURM_ARRAY_JOB_ID']}",
                    f"SLURM_ARRAY_TASK_ID={environment['SLURM_ARRAY_TASK_ID']}",
                ]
                if "SLURM_ARRAY_TASK_ID" in environment
                else []
            ),
        ]
        with Vertical(id="execute-confirm-box"):
            yield Label("Execute submitted script locally?", id="execute-confirm-title")
            yield Static(
                f"Task: [b]{escape(self.execution['task_id'])}[/b] "
                f"({escape(self.execution['task_state'])})\n"
                f"Script: [yellow]{escape(str(self.execution['script']))}[/yellow]\n"
                f"Working directory: {escape(str(self.execution['work_dir']))}\n"
                f"Environment: {escape('  '.join(variables))}\n\n"
                "This runs outside Slurm on the machine hosting this TUI."
            )
            with Horizontal(id="execute-confirm-buttons"):
                yield Button("Keep", id="keep", variant="default")
                yield Button("Execute locally", id="confirm", variant="warning")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_dismiss_no(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")


class InformationScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "dismiss_info", "Close"),
        Binding("q", "dismiss_info", "Close", show=False),
    ]

    CSS = """
    InformationScreen { align: center middle; background: $background 70%; }
    #info-box { width: 88%; height: 88%; border: heavy $accent; background: $surface; }
    #info-title { dock: top; height: 3; padding: 1 2; text-style: bold; background: $boost; }
    #info-scroll { padding: 1 2; }
    """

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self.title = title
        self.body = body

    def compose(self) -> ComposeResult:
        with Vertical(id="info-box"):
            yield Static(escape(self.title), id="info-title")
            with ScrollableContainer(id="info-scroll"):
                yield Static(self.body, id="info-body")

    def action_dismiss_info(self) -> None:
        self.dismiss(None)


class LogChoiceScreen(ButtonChoiceScreen[str | None]):
    BINDINGS = [
        *BUTTON_CHOICE_BINDINGS,
        Binding("o", "choose_stdout", "Stdout"),
        Binding("e", "choose_stderr", "Stderr"),
        Binding("escape", "dismiss_choice", "Close", show=False),
    ]

    CSS = """
    LogChoiceScreen { align: center middle; background: $background 70%; }
    #log-choice-box { width: 78; height: auto; padding: 1 2; border: heavy $accent; background: $surface; }
    #log-choice-buttons { height: auto; align-horizontal: right; margin-top: 1; }
    #log-choice-buttons Button { margin-left: 1; }
    #log-choice-buttons Button:focus { border: heavy $accent; text-style: bold reverse; }
    """

    def __init__(self, stdout_path: Path, stderr_path: Path) -> None:
        super().__init__()
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path

    def compose(self) -> ComposeResult:
        with Vertical(id="log-choice-box"):
            yield Label("Open task log", classes="pane-title")
            yield Static(
                f"[b]Stdout[/b]  {escape(str(self.stdout_path))}\n"
                f"[b]Stderr[/b]  {escape(str(self.stderr_path))}"
            )
            with Horizontal(id="log-choice-buttons"):
                yield Button("Stdout", id="stdout", variant="primary")
                yield Button("Stderr", id="stderr", variant="warning")
                yield Button("Cancel", id="cancel")

    def action_choose_stdout(self) -> None:
        self.dismiss("stdout")

    def action_choose_stderr(self) -> None:
        self.dismiss("stderr")

    def action_dismiss_choice(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id if event.button.id in {"stdout", "stderr"} else None)


class FileViewerScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("r", "reload", "Reload"),
        Binding("escape", "dismiss_viewer", "Close"),
        Binding("q", "dismiss_viewer", "Close", show=False),
    ]

    CSS = """
    FileViewerScreen { align: center middle; background: $background 75%; }
    #file-box { width: 94%; height: 92%; border: heavy $accent; background: $surface; }
    #file-title { height: 3; padding: 1 2; text-style: bold; background: $boost; }
    #file-meta { height: auto; max-height: 4; padding: 0 2 1 2; color: $text-muted; }
    #file-scroll { height: 1fr; padding: 0 2 1 2; }
    """

    def __init__(self, title: str, result: dict[str, Any]) -> None:
        super().__init__()
        self.viewer_title = title
        self.path = Path(result["path"])
        self.tail = bool(result["tail"])
        self.result = result

    def compose(self) -> ComposeResult:
        with Vertical(id="file-box"):
            yield Static(escape(self.viewer_title), id="file-title")
            yield Static("", id="file-meta")
            with ScrollableContainer(id="file-scroll"):
                yield Static(Text(), id="file-content")

    def on_mount(self) -> None:
        self.apply_file_result(self.result)

    def apply_file_result(self, result: dict[str, Any]) -> None:
        self.result = result
        direction = "tail" if result["tail"] else "head"
        truncation = (
            f" · truncated · showing {direction} {ARTIFACT_READ_LIMIT // 1024} KiB"
            if result["truncated"] else ""
        )
        self.query_one("#file-meta", Static).update(
            f"{escape(str(result['path']))} · {format_bytes(result['size'])}{truncation}"
        )
        self.query_one("#file-content", Static).update(Text(result["text"]))

    def action_reload(self) -> None:
        self.reload_file()

    @work(thread=True, exclusive=True, group="file-reload")
    def reload_file(self) -> None:
        try:
            result = read_artifact(self.path, tail=self.tail)
        except Exception as exc:
            self.app.call_from_thread(
                self.app.notify,
                f"Unable to reload {self.path}: {exc}",
                severity="error",
                timeout=8,
            )
        else:
            self.app.call_from_thread(self.apply_file_result, result)

    def action_dismiss_viewer(self) -> None:
        self.dismiss(None)


TUI_HELP_TEXT = """
[b cyan]Navigation[/b cyan]
  ↑/↓ or j/k    Move the current selection
  Enter         Open a job or show expanded task information
  Esc           Return to recent jobs or close a dialog
  /             Filter jobs by name or ID

[b cyan]Actions[/b cyan]
  c             Cancel the selected job/task (asks first)
  h             Hold a pending job/task
  r             Release a held pending job/task
  l             Open the selected task's stdout/stderr log
  s             Open the selected task's submitted script
  e             Execute the selected task's script locally
  Shift+R       Refresh immediately
  i             Show expanded scheduler information
  ?             Show this help
  q             Quit

[dim]Hold does not pause running work. Job actions target the whole scalar job or array;
task actions target only the selected array element.[/dim]
"""


class SlurmManagerApp(App[None]):
    TITLE = "QVC Slurm Manager"
    SUB_TITLE = "recent jobs and array tasks"
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("enter", "drill_down", "Open"),
        Binding("/", "filter_jobs", "Filter"),
        Binding("c", "cancel_selected", "Cancel"),
        Binding("h", "hold_selected", "Hold"),
        Binding("r", "resume_selected", "Release"),
        Binding("l", "open_log", "Log"),
        Binding("s", "open_script", "Script"),
        Binding("e", "execute_script", "Execute"),
        Binding("shift+r", "refresh", "Refresh"),
        Binding("i", "information", "Info"),
        Binding("?", "help", "Help"),
        Binding("escape", "escape_view", "Back", show=False),
    ]

    CSS = """
    Screen { background: #0b1020; color: #d9e2f2; }
    Header { background: #111b33; color: #8ed8ff; }
    Footer { background: #111b33; }
    #status-line { height: 3; padding: 1 2; background: #101a2e; color: #8ba3c7; }
    #filter { height: 3; margin: 0 1; border: tall #3a79a8; display: none; }
    #filter.visible { display: block; }
    #jobs-view { height: 1fr; border: round #245a83; margin: 0 1 1 1; }
    #drilldown-view { height: 1fr; display: none; }
    #drilldown-view.visible { display: block; }
    #jobs-view.hidden { display: none; }
    #array-overview {
        height: auto;
        margin: 0 1;
        padding: 1 2;
        border: round #245a83;
        background: #10182b;
    }
    #task-workspace { height: 1fr; }
    #tasks-pane { width: 64%; min-width: 58; border: round #245a83; margin: 0 1 1 1; }
    #details-pane { width: 36%; min-width: 34; border: round #245a83; margin: 0 1 1 0; }
    .pane-title { height: 3; padding: 1 1; text-style: bold; color: #8ed8ff; background: #101a2e; }
    DataTable { height: 1fr; background: #0d1426; }
    DataTable:focus { border: tall #47b5e8; }
    #task-detail-scroll { height: 1fr; padding: 1 2; background: #10182b; }
    #error-line { height: auto; max-height: 4; padding: 0 2; color: #ff7b89; display: none; }
    #error-line.visible { display: block; }
    #empty { layer: overlay; align: center middle; color: #7186a8; display: none; }
    #empty.visible { display: block; }
    Screen.narrow #task-workspace { layout: vertical; }
    Screen.narrow #tasks-pane { width: 100%; height: 58%; min-width: 0; }
    Screen.narrow #details-pane { width: 100%; height: 42%; min-width: 0; margin-left: 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.snapshot: dict[str, Any] = {"jobs": [], "refreshed_at": ""}
        self.jobs_by_id: dict[str, dict[str, Any]] = {}
        self.tasks_by_id: dict[str, dict[str, Any]] = {}
        self.selected_job_id: str | None = None
        self.selected_task_id: str | None = None
        self.filter_text = ""
        self.refresh_generation = 0
        self.refresh_in_progress = False
        self.view_mode = "jobs"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Connecting to Slurm…", id="status-line")
        yield Input(placeholder="Filter jobs by name or ID · Esc clears", id="filter")
        with Vertical(id="jobs-view"):
            yield Static("RECENT JOBS · newest first", classes="pane-title")
            yield DataTable(id="jobs", cursor_type="row", zebra_stripes=True)
            yield Static("No recent jobs in the accounting window", id="empty")
        with Vertical(id="drilldown-view"):
            yield Static(format_array_overview(None), id="array-overview")
            with Horizontal(id="task-workspace"):
                with Vertical(id="tasks-pane"):
                    yield Static("TASKS", id="tasks-title", classes="pane-title")
                    yield DataTable(id="tasks", cursor_type="row", zebra_stripes=True)
                with Vertical(id="details-pane"):
                    yield Static("TASK DETAILS", classes="pane-title")
                    with ScrollableContainer(id="task-detail-scroll"):
                        yield Static(format_task_details(None), id="task-detail")
        yield Static("", id="error-line")
        yield Footer()

    def on_mount(self) -> None:
        jobs = self.query_one("#jobs", DataTable)
        jobs.add_columns("Job", "Name", "State", "Progress", "Done", "Tasks", "Elapsed", "ETA")
        tasks = self.query_one("#tasks", DataTable)
        tasks.add_columns("Task", "State", "Elapsed", "Left", "CPU", "Memory", "Reason", "Exit")
        jobs.focus()
        self.set_interval(REFRESH_INTERVAL_SECONDS, self.action_refresh)
        self.action_refresh()

    def on_resize(self, event) -> None:
        self.screen.set_class(event.size.width < 100, "narrow")

    def action_refresh(self) -> None:
        if self.refresh_in_progress:
            return
        self.refresh_in_progress = True
        self.refresh_generation += 1
        generation = self.refresh_generation
        self.query_one("#status-line", Static).update("[cyan]●[/cyan] Refreshing my Slurm jobs…")
        self.refresh_snapshot(generation)

    @work(thread=True, exclusive=True, group="queue-refresh")
    def refresh_snapshot(self, generation: int) -> None:
        try:
            snapshot = build_tui_snapshot()
        except Exception as exc:
            self.call_from_thread(self.apply_refresh_error, generation, exc)
        else:
            self.call_from_thread(self.apply_snapshot, generation, snapshot)

    def apply_refresh_error(self, generation: int, exc: Exception) -> None:
        if generation != self.refresh_generation:
            return
        self.refresh_in_progress = False
        error = self.query_one("#error-line", Static)
        error.update(f"[b]Refresh failed:[/b] {escape(str(exc))} · showing the last good snapshot")
        error.add_class("visible")
        self.query_one("#status-line", Static).update("[red]●[/red] Slurm refresh failed")

    def apply_snapshot(self, generation: int, snapshot: dict[str, Any]) -> None:
        if generation != self.refresh_generation:
            return
        self.refresh_in_progress = False
        drilled_job_id = self.selected_job_id if self.view_mode == "tasks" else None
        self.snapshot = snapshot
        self.jobs_by_id = {job["job_id"]: job for job in snapshot["jobs"]}
        error = self.query_one("#error-line", Static)
        error.remove_class("visible")
        error.update("")
        if drilled_job_id and drilled_job_id not in self.jobs_by_id:
            self.show_jobs_view()
            self.notify(
                f"Job {drilled_job_id} is no longer among the 20 most recent jobs.",
                severity="information",
            )
        self.populate_jobs()
        if self.view_mode == "tasks" and drilled_job_id in self.jobs_by_id:
            self.selected_job_id = drilled_job_id
            self.prepare_job_tasks(drilled_job_id)
        active_tasks = sum(
            task.get("state") in ACTIVE_STATES
            for job in snapshot["jobs"]
            for task in job["tasks"]
        )
        refreshed = snapshot.get("refreshed_at", "")
        clock = refreshed[11:19] if len(refreshed) >= 19 else refreshed
        self.query_one("#status-line", Static).update(
            f"[green]●[/green] {len(snapshot['jobs'])} recent jobs · {active_tasks} active tasks · "
            f"my jobs · refreshed {escape(clock)}"
        )

    def filtered_jobs(self) -> list[dict[str, Any]]:
        query = self.filter_text.casefold().strip()
        if not query:
            return self.snapshot["jobs"]
        return [
            job for job in self.snapshot["jobs"]
            if query in job["job_id"].casefold() or query in job["job_name"].casefold()
        ]

    def populate_jobs(self) -> None:
        table = self.query_one("#jobs", DataTable)
        previous = self.selected_job_id
        table.clear(columns=False)
        visible = self.filtered_jobs()
        for job in visible:
            table.add_row(
                job["job_id"],
                job["job_name"],
                styled_state(job["state"]),
                styled_progress(job),
                f"{job['finished']}/{job['total']}",
                job["breakdown"],
                format_duration(job.get("elapsed_seconds")),
                format_array_eta(job.get("eta")),
                key=job["job_id"],
            )
        empty = self.query_one("#empty", Static)
        empty.set_class(not visible, "visible")
        if visible:
            ids = [job["job_id"] for job in visible]
            selected = previous if previous in ids else ids[0]
            self.selected_job_id = selected
            table.move_cursor(row=ids.index(selected), animate=False)
            self.prepare_job_tasks(selected)
        else:
            self.selected_job_id = None
            self.selected_task_id = None
            self.tasks_by_id = {}
            self.query_one("#tasks", DataTable).clear(columns=False)
            self.query_one("#task-detail", Static).update(format_task_details(None))
            self.query_one("#array-overview", Static).update(format_array_overview(None))

    def prepare_job_tasks(self, job_id: str) -> None:
        job = self.jobs_by_id.get(job_id)
        if not job:
            return
        self.selected_job_id = job_id
        self.query_one("#array-overview", Static).update(format_array_overview(job))
        self.query_one("#tasks-title", Static).update(
            f"TASKS · {escape(job['job_name'])} · #{job_id} · {job['breakdown']}"
        )
        tasks = self.query_one("#tasks", DataTable)
        previous = self.selected_task_id
        tasks.clear(columns=False)
        self.tasks_by_id = {task["job_id"]: task for task in job["tasks"]}
        task_ids = list(self.tasks_by_id)
        for task_id, task in self.tasks_by_id.items():
            tasks.add_row(
                format_task_id(task),
                styled_state(task.get("state", "UNKNOWN")),
                format_duration(task.get("elapsed_seconds")),
                format_duration(task.get("time_left_seconds")),
                str(task.get("alloc_cpus") or "—"),
                format_bytes(task.get("max_rss_bytes")),
                task.get("reason") or "—",
                task.get("exit_code") or "—",
                key=task_id,
            )
        if task_ids:
            selected = previous if previous in task_ids else task_ids[0]
            self.selected_task_id = selected
            tasks.move_cursor(row=task_ids.index(selected), animate=False)
            self.show_task_details(selected)
        else:
            self.selected_task_id = None
            self.query_one("#task-detail", Static).update(format_task_details(None))

    def show_task_details(self, task_id: str) -> None:
        task = self.tasks_by_id.get(task_id)
        if not task:
            return
        self.selected_task_id = task_id
        self.query_one("#task-detail", Static).update(format_task_details(task))

    def enter_job(self, job_id: str) -> None:
        if job_id not in self.jobs_by_id:
            return
        self.prepare_job_tasks(job_id)
        filter_widget = self.query_one("#filter", Input)
        filter_widget.remove_class("visible")
        self.query_one("#jobs-view", Vertical).add_class("hidden")
        self.query_one("#drilldown-view", Vertical).add_class("visible")
        self.view_mode = "tasks"
        self.query_one("#tasks", DataTable).focus()

    def show_jobs_view(self) -> None:
        self.query_one("#drilldown-view", Vertical).remove_class("visible")
        self.query_one("#jobs-view", Vertical).remove_class("hidden")
        self.view_mode = "jobs"
        self.query_one("#jobs", DataTable).focus()

    @on(DataTable.RowHighlighted, "#jobs")
    def job_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is None or event.row_key.value is None:
            return
        job_id = str(event.row_key.value)
        if job_id in self.jobs_by_id:
            self.prepare_job_tasks(job_id)

    @on(DataTable.RowSelected, "#jobs")
    def job_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key is None or event.row_key.value is None:
            return
        job_id = str(event.row_key.value)
        if job_id in self.jobs_by_id and self.tasks_by_id:
            self.enter_job(job_id)

    @on(DataTable.RowHighlighted, "#tasks")
    def task_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is None or event.row_key.value is None:
            return
        task_id = str(event.row_key.value)
        if task_id in self.tasks_by_id:
            self.show_task_details(task_id)

    @on(DataTable.RowSelected, "#tasks")
    def task_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key is None or event.row_key.value is None:
            return
        task_id = str(event.row_key.value)
        if task_id in self.tasks_by_id:
            self.selected_task_id = task_id
            self.action_information()

    @on(Input.Changed, "#filter")
    def filter_changed(self, event: Input.Changed) -> None:
        self.filter_text = event.value
        self.populate_jobs()

    def active_table(self) -> DataTable:
        if self.view_mode == "tasks":
            return self.query_one("#tasks", DataTable)
        return self.query_one("#jobs", DataTable)

    def current_selection(self) -> dict[str, Any] | None:
        if self.view_mode == "tasks" and self.selected_task_id:
            return self.tasks_by_id.get(self.selected_task_id)
        return self.jobs_by_id.get(self.selected_job_id or "")

    def action_cursor_down(self) -> None:
        self.active_table().action_cursor_down()

    def action_cursor_up(self) -> None:
        self.active_table().action_cursor_up()

    def action_drill_down(self) -> None:
        if self.view_mode == "jobs":
            if self.selected_job_id and self.tasks_by_id:
                self.enter_job(self.selected_job_id)
            return
        self.action_information()

    def action_filter_jobs(self) -> None:
        if self.view_mode != "jobs":
            self.notify("Return to recent jobs before filtering.", severity="information")
            return
        widget = self.query_one("#filter", Input)
        widget.add_class("visible")
        widget.focus()

    def action_escape_view(self) -> None:
        filter_widget = self.query_one("#filter", Input)
        if filter_widget.has_class("visible"):
            filter_widget.value = ""
            filter_widget.remove_class("visible")
            self.query_one("#jobs", DataTable).focus()
            return
        if self.view_mode == "tasks":
            self.show_jobs_view()
            return
        self.query_one("#jobs", DataTable).focus()

    def action_cancel_selected(self) -> None:
        selection = self.current_selection()
        allowed, reason = scheduler_action_allowed("cancel", selection)
        if not allowed:
            self.notify(reason, severity="warning")
            return
        target = str(selection["job_id"])
        description = selection.get("job_name") or target
        self.push_screen(
            ConfirmCancelScreen(target, description),
            lambda confirmed: self.perform_action("cancel", target) if confirmed else None,
        )

    def action_hold_selected(self) -> None:
        self.request_action("hold")

    def action_resume_selected(self) -> None:
        self.request_action("release")

    def request_action(self, action: str) -> None:
        selection = self.current_selection()
        allowed, reason = scheduler_action_allowed(action, selection)
        if not allowed:
            self.notify(reason, severity="warning")
            return
        self.perform_action(action, str(selection["job_id"]))

    @work(thread=True, group="scheduler-actions")
    def perform_action(self, action: str, target: str) -> None:
        try:
            result = run_scheduler_action(action, target)
        except Exception as exc:
            self.call_from_thread(
                self.notify,
                f"{action.title()} failed: {exc}",
                severity="error",
                timeout=8,
            )
        else:
            self.call_from_thread(self.finish_action, action, target, result)

    def finish_action(self, action: str, target: str, result: Any) -> None:
        if result.returncode == 0:
            verb = {"cancel": "Cancelled", "hold": "Held", "release": "Released"}[action]
            self.notify(f"{verb} {target}", severity="information")
            self.action_refresh()
        else:
            detail = (result.stderr or result.stdout or "scheduler command failed").strip()
            self.notify(f"{action.title()} failed: {detail}", severity="error", timeout=8)

    def action_information(self) -> None:
        selection = self.current_selection()
        if not selection:
            self.notify("No job or task is selected.", severity="warning")
            return
        self.load_information(dict(selection))

    def selected_artifact_task(self) -> dict[str, Any] | None:
        if self.view_mode != "tasks" or not self.selected_task_id:
            self.notify("Open a job and select a task first.", severity="warning")
            return None
        task = self.tasks_by_id.get(self.selected_task_id)
        if task is None:
            self.notify("No task is selected.", severity="warning")
        return task

    def action_open_log(self) -> None:
        task = self.selected_artifact_task()
        if task is not None:
            self.load_task_artifacts(dict(task), "log")

    def action_open_script(self) -> None:
        task = self.selected_artifact_task()
        if task is not None:
            self.load_task_artifacts(dict(task), "script")

    def action_execute_script(self) -> None:
        task = self.selected_artifact_task()
        if task is not None:
            self.load_task_execution(dict(task))

    @work(thread=True, exclusive=True, group="execution-metadata")
    def load_task_execution(self, task: dict[str, Any]) -> None:
        metadata_error = None
        try:
            details = get_job_details(task["job_id"])
        except Exception as exc:
            details = {}
            metadata_error = str(exc)
        try:
            execution = prepare_local_task_execution(task, details)
        except Exception as exc:
            message = f"Unable to execute task {task['job_id']}: {exc}"
            if metadata_error:
                message += f" Scheduler details: {metadata_error}"
            self.call_from_thread(
                self.notify,
                message,
                severity="error",
                timeout=8,
            )
        else:
            self.call_from_thread(self.confirm_task_execution, execution)

    def confirm_task_execution(self, execution: dict[str, Any]) -> None:
        self.push_screen(
            ConfirmExecuteScreen(execution),
            lambda confirmed: self.start_task_execution(execution) if confirmed else None,
        )

    def start_task_execution(self, execution: dict[str, Any]) -> None:
        self.notify(
            f"Executing task {execution['task_id']} locally…",
            severity="information",
        )
        self.execute_task_script(execution)

    @work(thread=True, group="local-script-execution")
    def execute_task_script(self, execution: dict[str, Any]) -> None:
        try:
            result = run_local_task_script(execution)
        except Exception as exc:
            self.call_from_thread(
                self.notify,
                f"Local execution of task {execution['task_id']} failed to start: {exc}",
                severity="error",
                timeout=8,
            )
        else:
            self.call_from_thread(self.finish_task_execution, result)

    def finish_task_execution(self, result: dict[str, Any]) -> None:
        returncode = result["returncode"]
        severity = "information" if returncode == 0 else "error"
        self.notify(
            f"Local task {result['task_id']} exited with code {returncode}. "
            f"Log: {result['log_path']}",
            severity=severity,
            timeout=8,
        )
        self.open_artifact_viewer(
            f"Local execution · {result['task_id']} · exit {returncode}",
            result["log_path"],
            tail=True,
        )

    @work(thread=True, exclusive=True, group="artifact-metadata")
    def load_task_artifacts(self, task: dict[str, Any], kind: str) -> None:
        metadata_error = None
        try:
            details = get_job_details(task["job_id"])
        except Exception as exc:
            details = {}
            metadata_error = str(exc)
        paths = resolve_task_artifacts(task, details)
        self.call_from_thread(self.present_task_artifacts, task, kind, paths, metadata_error)

    def present_task_artifacts(
        self,
        task: dict[str, Any],
        kind: str,
        paths: dict[str, Path | None],
        metadata_error: str | None,
    ) -> None:
        task_id = str(task["job_id"])
        if kind == "script":
            path = paths["script"]
            if path is None:
                message = "The submitted script path is unavailable for this task."
                if metadata_error:
                    message += f" Scheduler details: {metadata_error}"
                self.notify(message, severity="error", timeout=8)
                return
            self.open_artifact_viewer(f"Submitted script · {task_id}", path, tail=False)
            return

        stdout_path, stderr_path = paths["stdout"], paths["stderr"]
        available = [path for path in (stdout_path, stderr_path) if path is not None]
        if not available:
            message = "The stdout/stderr paths are unavailable for this task."
            if metadata_error:
                message += f" Scheduler details: {metadata_error}"
            self.notify(message, severity="error", timeout=8)
            return
        if stdout_path is None or stderr_path is None or stdout_path == stderr_path:
            self.open_artifact_viewer(f"Task log · {task_id}", available[0], tail=True)
            return

        def open_choice(choice: str | None) -> None:
            if choice == "stdout":
                self.open_artifact_viewer(f"Stdout · {task_id}", stdout_path, tail=True)
            elif choice == "stderr":
                self.open_artifact_viewer(f"Stderr · {task_id}", stderr_path, tail=True)

        self.push_screen(LogChoiceScreen(stdout_path, stderr_path), open_choice)

    @work(thread=True, group="artifact-read")
    def open_artifact_viewer(self, title: str, path: Path, *, tail: bool) -> None:
        try:
            result = read_artifact(path, tail=tail)
        except Exception as exc:
            self.call_from_thread(
                self.notify,
                f"Unable to open {path}: {exc}",
                severity="error",
                timeout=8,
            )
        else:
            self.call_from_thread(self.push_screen, FileViewerScreen(title, result))

    @work(thread=True, exclusive=True, group="job-information")
    def load_information(self, selection: dict[str, Any]) -> None:
        try:
            details = get_job_details(selection["job_id"])
        except Exception as exc:
            details = {"_unavailable": str(exc)}
        body = format_expanded_details(selection, details)
        self.call_from_thread(
            self.push_screen,
            InformationScreen(f"Job information · {selection['job_id']}", body),
        )

    def action_help(self) -> None:
        self.push_screen(InformationScreen("Keyboard commands", TUI_HELP_TEXT))


def run_tui() -> None:
    if not TEXTUAL_AVAILABLE:
        raise ImportError("Textual is required for TUI mode.", name="textual")
    SlurmManagerApp().run()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Manage Slurm jobs in an interactive TUI, inspect recent jobs, or "
            "inspect, cancel, hold, and resume jobs by job-name glob."
        )
    )
    parser.add_argument("mode", choices=["cancel", "hold", "resume", "status", "tui"])
    parser.add_argument(
        "pattern",
        nargs="?",
        help=(
            'Job name glob pattern, e.g. "train_*" or "mar19c*". '
            "Omit it with status to show the 20 most recent jobs; TUI mode "
            "does not accept a pattern."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--starttime",
        default="now-7days",
        help="sacct start time for status mode (default: now-7days).",
    )
    parser.add_argument("--endtime", default=None, help="Optional sacct end time for status mode.")
    args = parser.parse_args(argv)
    if args.mode not in {"status", "tui"} and args.pattern is None:
        parser.error(f"pattern is required for {args.mode}")
    if args.mode == "tui" and args.pattern is not None:
        parser.error("tui does not accept a job-name pattern")
    return args


def _run_compact_status(args):
    try:
        accounting_rows = get_accounting_rows(
            starttime=args.starttime,
            endtime=args.endtime,
        )
    except FileNotFoundError as exc:
        print(f"Error: required Slurm command not found: {exc.filename}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() if exc.stderr else str(exc)
        print(f"Error querying Slurm: {detail}", file=sys.stderr)
        return 1

    groups = recent_job_groups(accounting_rows, limit=20)
    if not groups:
        print("No recent jobs found in the selected accounting window.")
        return 0
    render_compact_status(groups, args.starttime, args.endtime)
    return 0


def _run_status(args):
    try:
        live_jobs = get_matching_jobs(args.pattern)
        accounting_rows = get_accounting_rows(
            starttime=args.starttime,
            endtime=args.endtime,
        )
    except FileNotFoundError as exc:
        print(f"Error: required Slurm command not found: {exc.filename}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() if exc.stderr else str(exc)
        print(f"Error querying Slurm: {detail}", file=sys.stderr)
        return 1

    groups = matching_arrays(accounting_rows, args.pattern)
    if not live_jobs and not groups:
        print(f"No jobs matched {args.pattern!r} in squeue or the selected accounting window.")
        return 0
    render_status(args.pattern, live_jobs, groups, args.starttime, args.endtime)
    return 0


def main():
    args = parse_args()
    if args.mode == "tui":
        try:
            run_tui()
        except ImportError as exc:
            if exc.name == "textual":
                print(
                    "Textual is required for TUI mode. Install the updated project dependencies.",
                    file=sys.stderr,
                )
                return 1
            raise
        return 0
    if args.mode == "status":
        if args.pattern is None:
            return _run_compact_status(args)
        args.pattern = normalize_glob(args.pattern)
        return _run_status(args)

    args.pattern = normalize_glob(args.pattern)

    try:
        matches = get_matching_jobs(args.pattern)
    except subprocess.CalledProcessError as exc:
        print(f"Error running squeue: {exc}", file=sys.stderr)
        return 1
    if not matches:
        print("No matching jobs found.")
        return 0

    print("Matching jobs:")
    print(f"  {'JOB ID':>12}  {'JOB NAME':<40}  {'STATE':<12}  {'RUNTIME':>12}")
    for job in matches:
        print(
            f"  {job['job_id']:>12}  {job['job_name']:<40}  "
            f"{job['job_state']:<12}  {job['runtime']:>12}"
        )
    if not args.dry_run:
        action_job_ids = list(dict.fromkeys(job["array_job_id"] for job in matches))
        act_on_jobs(action_job_ids, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
