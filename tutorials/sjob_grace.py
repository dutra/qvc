import os
import glob
import pandas as pd
import numpy as np

# Parameters
N = 1
#nwarm = 2000
nsamp = 100
nchains = 4
max_tree_depth = 6
job_id = 0
#lc_file = "data/may8_lc_all.h5"
#lc_file = "data/s82_lc_rf2000days_allsame.h5"
#filter_file = "data/df_quasars_filtered_apr29.csv"
#filter_file = "data/may19_quasars_filtered_ebv005.csv"
#filter_file = "data/colin_object_ids_test_rearrangedN20.csv"
#filter_file = "data/may28_rf2000days_allsame_ranked.csv"
#filter_file = "data/may22_good_sources_chisqcut.csv"
filter_file = "data/july21_chisq2_ebv_sn.csv"
script_path = "submit_jobs"
log_path = "log_jobs"

# Get total number of rows from CSV
df = pd.read_csv(filter_file)
total_objects = len(df)
print(f"Found {total_objects} objects in {filter_file}")

os.makedirs(script_path, exist_ok=True)
#SBATCH --time=2-00:00:00

#for job_id in range(250, 260):
for nwarm in np.flip(np.array([2000, 4000, 8000])):
    prefix = f"july30_single_{N}w{nwarm}s{nsamp}"

    suffix = f"job{job_id}"
    sbatch_filename = os.path.join(script_path, f"{prefix}_job_{suffix}.sh")
    output_filename = os.path.join(log_path, f"{prefix}_job_{suffix}.txt")
    result_file = f"data/{prefix}/{prefix}_fits_{suffix}.h5"

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
#SBATCH --time=2-00:00:00
#SBATCH --constraint="a100"

export JAX_ENABLE_X64=True
export PREFIX={prefix}
export SUFFIX={suffix}

module load miniconda
conda activate jaxgpu

nvidia-smi

start=$(date +%s)
echo "Start time: $start"

python multiband_fit.py --progress \
--file {result_file} --plot \
--nwarm {nwarm} --nsamp {nsamp} --nchains {nchains} \
--max_tree_depth {max_tree_depth} \
--joint --filter_object_id 1411060
#--joint --filter_file {filter_file} --job_id {job_id} --job_N {N} 

end=$(date +%s)
echo "End time: $end"

runtime=$((end - start))

hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

echo "Total runtime: ${{hours}}h ${{minutes}}m ${{seconds}}s"

""")

    os.system(f"sbatch {sbatch_filename}")
