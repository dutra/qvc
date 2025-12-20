import os
import glob
import pandas as pd
import numpy as np

# Parameters
N = 1
nwarm = 8000
nsamp = 1000
ncores = 10
nchains = ncores
max_tree_depth = 14
script_path = "submit_jobs/multibandfit"
log_path = "log_jobs/multibandfit"


prefix = f"oct22a_single_nearbylcs_tighterpriors_{N}w{nwarm}s{nsamp}t{max_tree_depth}co{ncores}ch{nchains}"

# Get total number of rows from CSV

os.makedirs(script_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

nearby_lcs = ['ngc3227', 'ngc4051', 'ngc4151', 'ngc4395', 'ngc4593', 'ucg06728']
#nearby_lcs = ['ucg06728']

for lc_object in nearby_lcs:
    
    suffix = f"job{lc_object}"

    sbatch_filename = os.path.join(script_path, f"{prefix}_job_{suffix}.sh")

    output_filename = os.path.join(log_path, f"{prefix}_job_{suffix}.txt")

    print(f"Submitting job {lc_object} {prefix}_{suffix}")

    with open(sbatch_filename, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=multiband_{prefix}_{suffix}
#SBATCH --output={output_filename}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={ncores}
#SBATCH --mem=20G
#SBATCH --partition=day
#SBATCH --time=1-00:00:00

export JAX_ENABLE_X64=True
export PREFIX={prefix}
export SUFFIX={suffix}
export NUM_CORES={ncores}

module load miniconda
conda activate jaxcpu

start=$(date +%s)
echo "Start time: $start"


# --- Run ---
python multiband_fit_single.py --plot \\
  --progress --nwarm {nwarm} --nsamp {nsamp} --nchains {nchains} \\
  --max_tree_depth {max_tree_depth} \\
  --disable_fhost \\
  --lmc -6 \\
  --load_nearby_lc_csv {lc_object}

end=$(date +%s)
echo "End time: $end"

runtime=$((end - start))

hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

echo "Total runtime: ${{hours}}h ${{minutes}}m ${{seconds}}s"

""")

    os.system(f"sbatch {sbatch_filename}")
