# submit_array.py
import os, math
import pandas as pd

# ---------- Parameters ----------
num_jobs = -1
#num_jobs = 500
skip = 0
N = 1
nwarm = 2000
nsamp = 200
ncores = 4
nchains = ncores
max_tree_depth = 14
nwarm = 500
nsamp = 100
ncores = 1
max_tree_depth = 12
nchains = ncores

script_path = "hpc_scripts/jobs/multibandfit" 
log_path    = "hpc_scripts/logs/multibandfit" 
#chisq_csv = "data/aug4_sample_chisqg10_ebv005sn3.csv"
#chisq_csv = "data/all_object_id.csv"
chisq_csv = "results/data/variability_chi_sq_red_g_gt_20.csv"
#chisq_csv = "results/data/lc_chisq_corrected_good.csv"
#chisq_csv = "results/data/oct9b_missing_object_ids.csv"
#fake_flags = "--inject_random_fake_etas --inject_fake --disable_lag_blr --disable_fhost"

#fhost_csv = "results/data/sep5_chisq_fhost0_N1w4000s1000t8c4_sept6_newN1fit_frachost0_nc64_shuffled.csv"
#fhost_csv = "results/data/sept6_newN1fit_frachost0_nc8_sep26e_preview_newfhostdef_euv_shuffled.csv"
#fhost_csv = "results/data/sept6_newN1fit_frachost0_nc16_sep27a_preview.csv"
#fhost_csv = "results/data/sept6_newN1fit_frachost0_nc16_sep30a_preview.csv"
fhost_csv = ""

#sample = 'stoneyu'
sample = 'chisq'
bpl = False

date = "apr1a_redchisq20_fastrun"

flags = ""
other = ""
if bpl:
    other += "bpl"
    flags += " --broken_pl"
else:
    other += "spl"

# chisq
if sample == 'chisq':
    print("Running chisq sample")
    filter_file = chisq_csv
    flags += ""
elif sample == 'stonelcs':
    flags += " --disable_fhost --load_stone_lcs"
    filter_file = "data/aug8_stone_merged.csv"
elif sample == 'stoneyu':
    flags += ""
    filter_file = "data/aug8_stone_merged.csv"
elif sample == 'fake_chisq':
    flags += " --disable_fhost --inject_fake --inject_random_fake_eta"
    filter_file = fhost_csv
    other += "_fakechisq"
elif sample == 'fakerandom_stoneyu':
    flags += " --disable_fhost --inject_fake --inject_random_fake_eta"
    filter_file = "data/aug8_stone_merged.csv"
    other += "_fakerandomstoneyu"
elif sample == 'fake_stoneyu':
    flags += " --disable_fhost --inject_fake --beta_tau 0.2"
    filter_file = "data/aug8_stone_merged.csv"
    other += "_fakestoneyu_betatau02"
else:
    raise Exception("sample invalid")


flags += ""
other += ""


prefix = f"{date}_{sample}_N{N}w{nwarm}s{nsamp}t{max_tree_depth}ch{nchains}"

# --------------------------------
os.makedirs(script_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

df = pd.read_csv(filter_file)
#print("[WARN] Data cut f_host_2500")
#if sample == 'chisq':
#df = df[df['f_host_2500'].between(0, 0.1)]
#print("Length of df after cuts: ", len(df))

total_objects = len(df)
if num_jobs < 0:
    num_jobs = math.ceil(total_objects / N) - skip

print(f"Found {total_objects} objects in {filter_file} → array size {num_jobs} (chunks of {N})")

sbatch_filename = os.path.join(script_path, f"{prefix}.sh")
# %x = job name, %A = array parent ID, %a = array task ID, %j = job ID string (for arrays, often like 2554691_17)
log_pattern = os.path.join(log_path, f"{prefix}-%A_%a-%j.txt")

sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=multiband_{prefix}
#SBATCH --output={log_pattern}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={ncores}
#SBATCH --mem=20G
#SBATCH --partition=day_amd
#SBATCH --time=4:00:00

# --- Environment ---
export JAX_ENABLE_X64=True
export PREFIX={prefix}
export SUFFIX="job${{SLURM_ARRAY_TASK_ID}}"
export NUM_CORES={ncores}
export N={N}
export SKIP={skip}
export FILTER_CSV="{filter_file}"
export FHOST_CSV="{fhost_csv}"

module load miniconda
conda activate jaxcpu2

start=$(date +%s)
echo "Start time: $start"
echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "SLURM_JOB_ID=$SLURM_JOB_ID SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

# --- Compute slice for this task ---
export TASK=${{SLURM_ARRAY_TASK_ID}}
export START=$(( TASK * N ))
export END=$(( START + N ))
echo "Slice: [${{START}}:${{END}})"

# --- Build the space-separated object_id list for this slice using pandas ---
IDS=$(python - << 'PY'
import os, pandas as pd
csv = os.environ['FILTER_CSV']
start = int(os.environ['START'])
end   = int(os.environ['END'])
df = pd.read_csv(csv) 
#if '{sample}' == 'chisq':
#df = df[df['f_host_2500'].between(0, 0.1)]
ids = df["object_id"].astype(str).tolist()[start:end]
print(" ".join(ids))
PY
)

if [ -z "$IDS" ]; then
  echo "No object_ids for TASK $TASK (slice $START:$END). Exiting."
  end=$(date +%s); rt=$((end-start))
  echo "Total runtime: $((rt/3600))h $(((rt%3600)/60))m $((rt%60))s"
  exit 0
fi

echo "object_ids: $IDS"


# --- Run ---

python -m qvc.light_curve.fit_light_curves \
 --filter_object_id $IDS \
 --plot \
 --disable_trace_plot --disable_correlation_plot --disable_histogram_plot \
 --disable_corner_plot --disable_sigma_tau_lambda_plot --disable_recovery_plot \
 --progress \
 --nwarm {nwarm} \
 --nsamp {nsamp} \
 --nchains {nchains}

end=$(date +%s)
rt=$((end - start))
echo "End time: $end"
echo "Total runtime: $((rt/3600))h $(((rt%3600)/60))m $((rt%60))s"
"""

with open(sbatch_filename, "w") as f:
    f.write(sbatch_script)

MAX_ARRAY_SIZE = 10_000

if num_jobs > 1:
    start = skip
    end = skip + num_jobs - 1

    for batch_start in range(start, end + 1, MAX_ARRAY_SIZE):
        batch_end = min(batch_start + MAX_ARRAY_SIZE - 1, end)
        os.system(f"sbatch --array={batch_start}-{batch_end} {sbatch_filename}")
        print(f"Submitted array: {batch_start}-{batch_end} using {sbatch_filename}")

elif num_jobs == 1:
    os.system(f"sbatch {sbatch_filename}")
    print(f"Submitted job using {sbatch_filename}")
