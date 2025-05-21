import os
import glob
import pandas as pd
import numpy as np
from subprocess import call

# Parameters
choose_N = 20
nwarm = 500
nsamp = 250
nchains = -1
prefix = "may21_joint"
lc_file = "data/may8_lc_all.h5"
#filter_file = "data/df_quasars_filtered_apr29.csv"
#filter_file = "data/may19_quasars_filtered_ebv005.csv"
filter_file = "data/colin_object_ids_test_rearrangedN20.csv"
script_path = "submit_jobs"

# Loop over each choose_N and submit a job
for job_id in np.flip(np.arange(0, 19)):
    print("Job id ", job_id)
    suffix = f"N20_job{job_id}"
    sbatch_filename = os.path.join(script_path, f"{prefix}_job_{suffix}.sh")
    print("Executing ", sbatch_filename)
    call(["bash", sbatch_filename])

