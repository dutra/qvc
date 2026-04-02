#!/usr/bin/env python3

import argparse
import fnmatch
import subprocess
import sys


def get_matching_jobs(pattern, me_only=True):
    """
    Return a list of dicts for jobs whose names match the glob pattern.
    """
    # %i = job id, %j = job name, %T = job state
    cmd = ["squeue", "--noheader", "--format=%i|%j|%T"]
    if me_only:
        cmd.append("--me")

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    matching_jobs = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split("|", maxsplit=2)
        if len(parts) != 3:
            continue

        job_id, job_name, job_state = parts
        if fnmatch.fnmatch(job_name, pattern):
            matching_jobs.append(
                {
                    "job_id": job_id,
                    "job_name": job_name,
                    "job_state": job_state,
                }
            )

    return matching_jobs


def act_on_jobs(job_ids, mode):
    """
    Apply the requested Slurm action to each job ID.
    """
    if mode == "cancel":
        for job_id in job_ids:
            print(f"Cancelling job {job_id}")
            subprocess.run(["scancel", job_id], check=False)

    elif mode == "hold":
        for job_id in job_ids:
            print(f"Holding job {job_id}")
            subprocess.run(["scontrol", "hold", job_id], check=False)

    elif mode == "resume":
        for job_id in job_ids:
            print(f"Releasing job {job_id}")
            subprocess.run(["scontrol", "release", job_id], check=False)

    elif mode == "status":
        # Non-mutating mode; nothing to do here.
        return

    else:
        raise ValueError(f"Unknown mode: {mode}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect, cancel, hold, or resume Slurm jobs by job-name glob."
    )
    parser.add_argument(
        "mode",
        choices=["cancel", "hold", "resume", "status"],
        help="Action to apply to matching jobs.",
    )
    parser.add_argument(
        "pattern",
        help='Job name glob pattern, e.g. "train_*" or "mar19c*"',
    )
    parser.add_argument(
        "--all-users",
        action="store_true",
        help="Search all users' jobs instead of only your own.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show matching jobs without applying any action.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        matches = get_matching_jobs(args.pattern, me_only=not args.all_users)
    except subprocess.CalledProcessError as e:
        print(f"Error running squeue: {e}", file=sys.stderr)
        sys.exit(1)

    if not matches:
        print("No matching jobs found.")
        return

    print("Matching jobs:")
    for job in matches:
        print(f"  {job['job_id']:>12}  {job['job_name']:<40}  {job['job_state']}")

    if args.mode == "status":
        return

    if args.dry_run:
        return

    act_on_jobs([job["job_id"] for job in matches], args.mode)


if __name__ == "__main__":
    main()