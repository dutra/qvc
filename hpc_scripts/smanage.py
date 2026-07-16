#!/usr/bin/env python3

import argparse
import fnmatch
import math
import re
import statistics
import subprocess
import sys
from collections import Counter

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
MEMORY_RE = re.compile(
    r"^(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[KMGTPE]?)B?(?P<scope>[cn]?)$",
    re.IGNORECASE,
)
ACCOUNTING_FIELDS = (
    "JobID%128,JobName%256,State,ElapsedRaw,TotalCPU,"
    "AllocCPUS,NNodes,MaxRSS,ReqMem,TimelimitRaw,ExitCode"
)
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}


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

        allocation_match = ARRAY_TASK_RE.fullmatch(job_id)
        if allocation_match is not None:
            array_job_id = allocation_match.group("array_job_id")
            task_id = int(allocation_match.group("task_id"))
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
            }
            row["requested_memory_bytes"] = _requested_memory_bytes(
                row["req_mem"], row["alloc_cpus"], row["nodes"]
            )
            allocations[job_id] = row
            continue
        parent_id, separator, _step_name = job_id.rpartition(".")
        if separator:
            rss = _parse_memory_bytes(max_rss)
            if rss is not None:
                step_max_rss[parent_id] = max(step_max_rss.get(parent_id, 0), rss)

    for job_id, row in allocations.items():
        if job_id in step_max_rss:
            row["max_rss_bytes"] = max(row["max_rss_bytes"] or 0, step_max_rss[job_id])
    return list(allocations.values())


def get_accounting_rows(starttime="now-7days", endtime=None, all_users=False):
    cmd = [
        "sacct",
        "--array",
        "--noheader",
        "--parsable2",
        f"--starttime={starttime}",
        f"--format={ACCOUNTING_FIELDS}",
    ]
    if endtime is not None:
        cmd.append(f"--endtime={endtime}")
    if all_users:
        cmd.append("--allusers")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return _parse_sacct_output(result.stdout)


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


def get_matching_jobs(pattern, me_only=True):
    """Return live squeue jobs whose names match the glob pattern."""
    cmd = ["squeue", "--noheader", "--format=%i|%j|%T|%M"]
    if me_only:
        cmd.append("--me")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    matching_jobs = []
    for line in result.stdout.splitlines():
        parts = line.strip().split("|", maxsplit=3)
        if len(parts) != 4:
            continue
        job_id, job_name, job_state, runtime = parts
        if fnmatch.fnmatch(job_name, pattern):
            matching_jobs.append(
                {
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
    return {
        "total": len(rows),
        "terminal": len(terminal),
        "completed": len(completed),
        "completion_rate": 100.0 * len(completed) / len(terminal) if terminal else None,
        "oom": len(oom),
        "oom_rate": 100.0 * len(oom) / len(terminal) if terminal else None,
    }


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


def _rich_array_report(console, title, rows):
    overview = _array_overview(rows)
    oom_style = "bold red" if overview["oom"] else "green"
    summary = Text()
    summary.append(f"Tasks {overview['total']}   ")
    summary.append(f"Completed {overview['completed']}/{overview['terminal']} terminal ")
    summary.append(f"({format_percent(overview['completion_rate'])})   ", style="green")
    summary.append(
        f"OOM {overview['oom']} ({format_percent(overview['oom_rate'])})",
        style=oom_style,
    )
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
        _rich_array_report(console, f"Combined · {len(groups)} arrays", all_rows)


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
        print(f"\nArray {array_job_id}  {job_name}")
        print("  states: " + ", ".join(f"{k}={v}" for k, v in sorted(_state_counts(rows).items())))
        print(
            f"  completed={overview['completed']}/{overview['terminal']} terminal "
            f"({format_percent(overview['completion_rate'])}), "
            f"OOM={overview['oom']} ({format_percent(overview['oom_rate'])})"
        )
        for label, getter, formatter in _metric_specs():
            print(_plain_metric_line(rows, label, getter, formatter))
        all_rows.extend(rows)
    if len(reports) > 1:
        print("\nCombined matching arrays")
        for label, getter, formatter in _metric_specs():
            print(_plain_metric_line(all_rows, label, getter, formatter))


def render_status(pattern, live_jobs, groups, starttime, endtime=None, console=None):
    if RICH_AVAILABLE:
        render_rich_status(pattern, live_jobs, groups, starttime, endtime, console=console)
    else:
        render_plain_status(pattern, live_jobs, groups, starttime, endtime)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect, cancel, hold, or resume Slurm jobs by job-name glob."
    )
    parser.add_argument("mode", choices=["cancel", "hold", "resume", "status"])
    parser.add_argument("pattern", help='Job name glob pattern, e.g. "train_*" or "mar19c*"')
    parser.add_argument("--all-users", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--starttime",
        default="now-7days",
        help="sacct start time for status mode (default: now-7days).",
    )
    parser.add_argument("--endtime", default=None, help="Optional sacct end time for status mode.")
    return parser.parse_args()


def _run_status(args):
    try:
        live_jobs = get_matching_jobs(args.pattern, me_only=not args.all_users)
        accounting_rows = get_accounting_rows(
            starttime=args.starttime,
            endtime=args.endtime,
            all_users=args.all_users,
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
    args.pattern = normalize_glob(args.pattern)
    if args.mode == "status":
        return _run_status(args)

    try:
        matches = get_matching_jobs(args.pattern, me_only=not args.all_users)
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
        act_on_jobs([job["job_id"] for job in matches], args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
