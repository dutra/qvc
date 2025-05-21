import os
import glob
import pandas as pd
import numpy as np

# Parameters
choose_N = 20
nwarm = 500
nsamp = 250
nchains = -1
prefix = "may21_a100_joint"
lc_file = "data/may8_lc_all.h5"
#filter_file = "data/df_quasars_filtered_apr29.csv"
#filter_file = "data/may19_quasars_filtered_ebv005.csv"
filter_file = "data/colin_object_ids_test_rearrangedN20.csv"
script_path = "submit_jobs"

# Get total number of rows from CSV
df = pd.read_csv(filter_file)
total_objects = len(df)
print(f"Found {total_objects} objects in {filter_file}")

os.makedirs(script_path, exist_ok=True)
#SBATCH --time=2-00:00:00

# Loop over each choose_N and submit a job
for job_id in range(0, 27):
    suffix = f"N20_job{job_id}"
    sbatch_filename = os.path.join(script_path, f"{prefix}_job_{suffix}.sh")
    output_filename = os.path.join(script_path, f"{prefix}_gpu_job_{suffix}.txt")
    result_file = f"data/N20_w{nwarm}/{prefix}_fits_{suffix}.h5"

    print(f"Submitting job {prefix}_{suffix}")

    with open(sbatch_filename, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name={prefix}_fit_{suffix}
#SBATCH --output={output_filename}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --constraint="a100"

export JAX_ENABLE_X64=True
export PREFIX={prefix}
export SUFFIX={suffix}

module load miniconda
conda activate jaxgpu

nvidia-smi

start=`date +%s`
echo $start

python multiband_fit.py  --progress \\
--filter_file {filter_file} --file {result_file} --plot --nwarm {nwarm} --nsamp {nsamp} --nchains {nchains} --joint --job_id {job_id}

end=`date +%s`
echo $end

runtime=$( echo "$end - $start" | bc -l )
echo $runtime
""")

    os.system(f"sbatch {sbatch_filename}")
