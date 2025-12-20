# submit_array.py
import os, math
import pandas as pd

# ---------- Parameters ----------
#num_jobs = 500
num_jobs = -1
skip = 0
N = 1
nwarm = 1000
nsamp = 200
ncores = 4
nchains = ncores
max_tree_depth = 14

script_path = "submit_jobs/multibandfit" 
log_path    = "log_jobs/multibandfit" 
chisq_csv = "data/aug4_sample_chisqg10_ebv005sn3.csv"
#chisq_csv = "results/data/oct9b_missing_object_ids.csv"
#fake_flags = "--inject_random_fake_etas --inject_fake --disable_lag_blr --disable_fhost"

#fhost_csv = "results/data/sep5_chisq_fhost0_N1w4000s1000t8c4_sept6_newN1fit_frachost0_nc64_shuffled.csv"
#fhost_csv = "results/data/sept6_newN1fit_frachost0_nc8_sep26e_preview_newfhostdef_euv_shuffled.csv"
#fhost_csv = "results/data/sept6_newN1fit_frachost0_nc16_sep27a_preview.csv"
#fhost_csv = "results/data/sept6_newN1fit_frachost0_nc16_sep30a_preview.csv"
fhost_csv = ""
lmc = -6

#sample = 'stoneyu'
sample = 'chisq'
bpl = False

date = "nov10a_single_chisq_carma_mixscalar_nozband_highertaufastlim_removemix_fixband"

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
    #filter_file = fhost_csv
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


flags += " --disable_fhost --bwb"
other += "_nofhost_bwb"
#flags += " --disable_fhost --disable_poly1 --disable_band_drop"
#other +="_nofhost_nopoly1_nobwb_disablebandrop"
#flags += " --disable_fhost --bwb --rf_length_cut 2400 --disable_poly1"
#other += "_nofhost_bwb_rflengthcut2400_nopoly1"

if N > 1:
    flags += " --d_eta"
    other += "_deta"


prefix = f"{date}_lagblrband_{sample}_{other}_lmc{lmc}_N{N}w{nwarm}s{nsamp}t{max_tree_depth}ch{nchains}"

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
#SBATCH --partition=day
#SBATCH --time=1:00:00

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
conda activate jaxcpu

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
python multiband_fit_single.py --plot \\
  --progress --nwarm {nwarm} --nsamp {nsamp} --nchains {nchains} \\
  --max_tree_depth {max_tree_depth} \\
  --lmc {lmc} \\
  {flags} \\
  --filter_object_id $IDS
  #--fhost_csv "$FHOST_CSV"\\
  
# --plot 
# --rf_length_cut 2400
  #--N {N} --skip ${{START}} 

          
  #--d_eta --disable_poly1 \\
          # 
end=$(date +%s)
rt=$((end - start))
echo "End time: $end"
echo "Total runtime: $((rt/3600))h $(((rt%3600)/60))m $((rt%60))s"
"""

with open(sbatch_filename, "w") as f:
    f.write(sbatch_script)

# Submit one array covering all chunks
if num_jobs > 10_000:
    os.system(f"sbatch --array=0-9999 {sbatch_filename}")
    print(f"Submitted array: 0-9999 using {sbatch_filename}")
    os.system(f"sbatch --array=10000-{num_jobs} {sbatch_filename}")
    print(f"Submitted array: 10000-{num_jobs} using {sbatch_filename}")
elif num_jobs > 1:
    os.system(f"sbatch --array={skip}-{skip+num_jobs-1} {sbatch_filename}")
    print(f"Submitted array: {skip}-{skip+num_jobs-1} using {sbatch_filename}")
elif num_jobs == 1:
    os.system(f"sbatch {sbatch_filename}")
    print(f"Submitted array: using {sbatch_filename}")


