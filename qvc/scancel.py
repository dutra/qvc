#!/usr/bin/env python3

import subprocess
import fnmatch
import sys

def get_matching_jobs(pattern):
    # Get all running jobs: JobID and JobName
    result = subprocess.run(
        ["squeue", "--noheader", "--format=%i %j"],
        capture_output=True, text=True, check=True
    )

    matching_jobs = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        job_id, job_name = line.strip().split(maxsplit=1)
        if fnmatch.fnmatch(job_name, pattern):
            matching_jobs.append(job_id)
    return matching_jobs

def cancel_jobs(job_ids):
    for job_id in job_ids:
        print(f"Cancelling job {job_id}")
        subprocess.run(["scancel", job_id], check=False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./cancel_matching_jobs.py <job_name_glob>")
        sys.exit(1)

    name_pattern = sys.argv[1]
    jobs_to_cancel = get_matching_jobs(name_pattern)

    if jobs_to_cancel:
        cancel_jobs(jobs_to_cancel)
    else:
        print("No matching jobs found.")
