import os
import glob
import pandas as pd

# Parameters
choose_N_list = [100, 50, 25, 150, 250, 500, 1000, 2000, 3800]
prefix = "may18_joint"
lc_file = "data/may8_lc_all.h5"
filter_file = "data/df_quasars_filtered_apr29.csv"
script_path = "submit_jobs"

# Get total number of rows from CSV
df = pd.read_csv(filter_file)
total_objects = len(df)
print(f"Found {total_objects} objects in {filter_file}")

os.makedirs(script_path, exist_ok=True)

# Loop over each choose_N and submit a job
for choose_N in choose_N_list:
    suffix = f"N{choose_N}"
    sbatch_filename = os.path.join(script_path, f"{prefix}_job_{suffix}.sh")
    output_filename = os.path.join(script_path, f"{prefix}_gpu_job_{suffix}.txt")
    result_file = f"data/{prefix}_objs_{suffix}.h5"

    with open(sbatch_filename, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name={prefix}_fit_{suffix}
#SBATCH --output={output_filename}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --constraint="a100-80g"

export JAX_ENABLE_X64=True
export PREFIX={prefix}
export SUFFIX={suffix}

module load miniconda
conda activate jaxgpu

start=`date +%s`
echo $start

python multiband_fit.py --choose_N {choose_N} --lc_file {lc_file} \\
    --filter_file {filter_file} --file {result_file} --plot --nwarm 500 --nsamp 100 --joint

end=`date +%s`
echo $end

runtime=$( echo "$end - $start" | bc -l )
echo $runtime
""")

    os.system(f"sbatch {sbatch_filename}")
