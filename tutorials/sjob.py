import os
import glob
import pandas as pd
import numpy as np

# Parameters
choose_N = 20
#nwarm = 2000
nwarm_list = [250, 500, 1000, 1500, 2000, 3000, 5000]
nchains = -1
prefix = "may20_joint"
lc_file = "data/may8_lc_all.h5"
#filter_file = "data/df_quasars_filtered_apr29.csv"
filter_file = "data/may19_quasars_filtered_ebv005.csv"
script_path = "submit_jobs"

# Get total number of rows from CSV
df = pd.read_csv(filter_file)
total_objects = len(df)
print(f"Found {total_objects} objects in {filter_file}")

os.makedirs(script_path, exist_ok=True)

# Loop over each choose_N and submit a job
for nwarm in nwarm_list:
    nsamp = nwarm
    suffix = f"N{choose_N}_w{nwarm}"
    sbatch_filename = os.path.join(script_path, f"{prefix}_job_{suffix}.sh")
    output_filename = os.path.join(script_path, f"{prefix}_gpu_job_{suffix}.txt")
    result_file = f"data/N{choose_N}/{prefix}_objs_{suffix}.h5"

    print(f"Submitting job {prefix}_{suffix}")

    with open(sbatch_filename, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name={prefix}_fit_{suffix}
#SBATCH --output={output_filename}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --constraint="a100"

export JAX_ENABLE_X64=True
export PREFIX={prefix}
export SUFFIX={suffix}

module load miniconda
conda activate jaxgpu

start=`date +%s`
echo $start

python multiband_fit.py --choose_N {choose_N}  --progress \\
--filter_file {filter_file} --file {result_file} --plot --nwarm {nwarm} --nsamp {nsamp} --nchains {nchains} --joint

end=`date +%s`
echo $end

runtime=$( echo "$end - $start" | bc -l )
echo $runtime
""")

    os.system(f"sbatch {sbatch_filename}")
