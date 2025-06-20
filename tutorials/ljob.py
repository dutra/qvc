import os
import glob
import pandas as pd
import numpy as np
from subprocess import call

# Parameters
N = 20
nwarm = 1000
nsamp = 500
nchains = -1
script_path = "submit_jobs"

# Loop over each choose_N and submit a job
for job_id in np.flip(np.arange(25, 28)):
    prefix = f"june3_etab05_{N}w{nwarm}s{nsamp}"
    print("Job id ", job_id)
    suffix = f"job{job_id}"
    sbatch_filename = os.path.join(script_path, f"{prefix}_job_{suffix}.sh")
    print("Executing ", sbatch_filename)
    call(["bash", sbatch_filename])

