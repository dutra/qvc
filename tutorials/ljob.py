import os
import glob
import pandas as pd
import numpy as np
from subprocess import call

# Parameters
N = 20
nwarm = 500
nsamp = 100
nchains = -1
prefix = f"may23_joint_chi510_N{N}w{nwarm}s{nsamp}"
script_path = "submit_jobs"

# Loop over each choose_N and submit a job
for job_id in np.arange(217, 300):
    print("Job id ", job_id)
    suffix = f"N{N}_job{job_id}"
    sbatch_filename = os.path.join(script_path, f"{prefix}_job_{suffix}.sh")
    print("Executing ", sbatch_filename)
    call(["bash", sbatch_filename])

