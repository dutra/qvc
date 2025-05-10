import os
import glob
import pandas as pd

# Parameters
chunk_size = 500         # Number of objects per job
suffix = "e5smoothpl"
lc_file = "data/may8_lc_all.h5"
filter_file = "data/df_quasars_filtered_apr29.csv"
script_path = "submit_jobs"
output_glob = f"data/may10_objs_tauwavelength_taublr_{suffix}_*.h5"
log_glob = f"{script_path}/gpu_job_*.txt"

def delete_existing_outputs():
    """Delete existing output .h5 files and SLURM logs."""
    for pattern in [output_glob, log_glob]:
        for filepath in glob.glob(pattern):
            print(f"Deleting {filepath}")
            os.remove(filepath)
    if os.path.exists(script_path):
        for f in os.listdir(script_path):
            os.remove(os.path.join(script_path, f))
    os.makedirs(script_path, exist_ok=True)

# Get total number of rows from CSV
df = pd.read_csv(filter_file)
total_objects = len(df)
print(f"Found {total_objects} objects in {filter_file}")

# Delete previous outputs before starting
delete_existing_outputs()

# Generate and submit new sbatch scripts
for i, skip in enumerate(range(0, total_objects, chunk_size)):
    sbatch_filename = os.path.join(script_path, f"job_{i}.sh")
    output_filename = f"{script_path}/gpu_job_{i}.txt"
    result_file = f"data/may10_objs_tauwavelength_taublr_{suffix}_{i}.h5"

    with open(sbatch_filename, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=multiband_fit_taublr_{suffix}_{i}
#SBATCH --output={output_filename}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --constraint="a100"

export JAX_ENABLE_X64=True
export SUFFIX={suffix}

module load miniconda
conda activate jaxgpu

python multiband_fit_blrkernel.py --skip {skip} --lc_file {lc_file} \\
    --filter_file {filter_file} --file {result_file} --plot
""")
# --N {chunk_size} 
    os.system(f"sbatch {sbatch_filename}")
