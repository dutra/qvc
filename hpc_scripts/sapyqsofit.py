#!/usr/bin/env python3
"""
spyqsofit.py  —  Submit a single SLURM *array* job for PyQSOFit batches.

This converts the previous per-job submission into one array submission,
mirroring the slice/array style used elsewhere.

Each array task i runs:
    python -m spectra.fit_spectra INPUT.h5 OUTPUT_DIR/job{i}.h5 --N N --skip i*N

Edit the parameters in the "Parameters" section below.
"""

import os
import math
from pathlib import Path

# ---------------- Parameters (EDIT ME) ----------------
ncores    = 4
mem_gb    = 48
partition = "day"
time_str  = "8:00:00"

# Total number of array tasks
num_jobs  = 300

mode = "single"

# Input table and run label
#input_file = "results/data/sep5_chisq_fhost0_N1w4000s1000t8c4_2.h5"
#estimated_total = 13194
#input_file = "results/data/oct3g_nofhost_etatau0505_preview_chisq_spl_sigma1_nofhost_fhost0_lmc-6_N1w1000s200t14co4ch4.h5"
#estimated_total = 9636
#input_file = "results/data/oct9b_lagblrband_chisq_spl_nofhost_bwb_lmc-6_N1w1000s200t14ch4.h5"
#input_file = "results/data/sep19b_sep8_chisq_yupriors_N20w4000s1000t8co4ch4_13k.h5"
#estimated_total = 13025

#input_file = "results/data/oct24a_single_chisq_lagblrband_chisq_spl_nofhost_lmc-6_N1w1000s200t14ch4.h5"
#input_file = "results/data/oct26a_carma_single_lagblrband_chisq_spl_nofhost_bwb_taufasttruncated_lmc-6_N1w1000s200t14ch4.h5"
input_file = "results/data/nov10a_single_chisq_carma_mixscalar_nozband_highertaufastlim_removemix_fixband_lagblrband_chisq_spl_nofhost_bwb_lmc-6_N1w1000s200t14ch4.h5"
estimated_total = 13189

#single_csv = "results/data/oct22a_scratch_sep10b_sep8_nofhost_huber_poly_bc_mc0_noscale_fixbrokenpl_PLbreakwave4k_fluxrescale_fixedPLbreak.csv"
#single_csv = "results/data/oct25a_collect_scratch_oct24a_nofhost_huber_poly_bc_fluxrescale_fixedPLbreak_mc0.csv"
#single_csv = "results/data/oct26a_collect_scratch_oct25a_nofhost_huber_poly_bc_fixedPLbreak_fixedscaleerr_mc0.csv"
#single_csv = "results/data/oct26d_collect_scratch_oct25a_nofhost_huber_poly_bc_fixedPLbreak_nofluxscale_mc0.csv"
#single_csv = "results/data/oct26d_collect_scratch_oct25a_nofhost_huber_poly_bc_fixedPLbreak_nofluxscale_mc0_poly150.csv"
#single_csv = "results/data/oct27b_collect_scratch_oct26a_carma_nofhost_huber_poly_bc_fixedPLbreak_nofluxscale_mc0_mc0.csv"
#single_csv = "results/data/oct27b_collect_scratch_oct26a_carma_nofhost_huber_poly_bc_fixedPLbreak_nofluxscale_mc0_bc120.csv"
#single_csv = "results/data/oct27b_collect_scratch_oct26a_carma_nofhost_huber_poly_bc_fixedPLbreak_nofluxscale_mc0_mc0_poly150.csv"
#single_csv = "results/data/oct27b_collect_scratch_oct26a_carma_nofhost_huber_poly_bc_fixedPLbreak_nofluxscale_mc0_mc0_polyz1_poly120.csv"
#single_csv = "results/data/nov1c_collect_scratch_oct27b_carma_nofhost_iron_best.csv"

single_csv = "results/data/nov11c_collect_scratch_nov10a_carma_removemix_fixmeanband_no1pluszflux2L_freeiron_mc0.csv"

prefix     = f"nov12a_11c_{mode}_scratch_nov10a_carma_removemix_fixmeanband_no1pluszflux2L_freeiron"

flags = "--enable_BC --enable_poly"

if mode == "single":
    flags += f" --MC_samples 50 --single_csv {single_csv}"
    prefix += "_mc50_best"
elif mode == "collect":
    flags += " --MC_samples 0"
    prefix += "_mc0"


# Estimated total rows to process (set if you want exact chunking).
# If unknown, just leave as None and set N explicitly below.

# Items per task (chunk size). If None, will be computed from 'estimated_total' and 'num_jobs'.
N = None
# -----------------------------------------------------


def main():
    os.makedirs("submit_jobs/pyqsofit", exist_ok=True)
    os.makedirs("log_jobs/pyqsofit", exist_ok=True)

    # Determine chunk size N
    if N is None:
        if estimated_total is None:
            raise SystemExit(
                "Please set either N explicitly, or provide estimated_total to compute it."
            )
        # Ceil so we cover the full set even if not divisible
        chunk = math.ceil(estimated_total / num_jobs)
    else:
        chunk = int(N)

    out_dir = Path(f"results/data/{prefix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compose single sbatch script for the entire array
    sbatch_path = Path("submit_jobs/pyqsofit") / f"pyqsofit_{prefix}.sh"
    log_pattern = Path("log_jobs/pyqsofit") / f"{prefix}-%A_%a.txt"

    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=pyqsofit_{prefix}
#SBATCH --output={log_pattern}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={ncores}
#SBATCH --mem={mem_gb}G
#SBATCH --partition={partition}
#SBATCH --time={time_str}

# --- Environment ---
export JAX_ENABLE_X64=True
export QT_QPA_PLATFORM=offscreen
export NUM_CORES={ncores}
export PREFIX={prefix}
export OUTDIR="{out_dir}"

module load miniconda
conda activate pyqsofit

start=$(date +%s)
echo "Start time: $start"
echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

# --- Derive slice for this array task ---
TASK=${{SLURM_ARRAY_TASK_ID}}
N={chunk}
START=$(( TASK * N ))
echo "Chunk size N=$N; slice START=$START"

# --- I/O paths ---
INPUT="{input_file}"
export SUFFIX="job${{TASK}}"
OUTFILE="${{OUTDIR}}/${{SUFFIX}}.csv"
mkdir -p "${{OUTDIR}}"

echo "Running: python -m spectra.fit_spectra $INPUT $OUTFILE --N $N --skip $START"

python -m spectra.fit_spectra "$INPUT" "$OUTFILE" --mode {mode} --N "$N" --skip "$START" {flags} 

end=$(date +%s)
echo "End time: $end"
rt=$((end - start))
echo "Total runtime: $((rt/3600))h $(((rt%3600)/60))m $((rt%60))s"
"""

    sbatch_path.write_text(sbatch_script)

    # Submit the array: 0..num_jobs-1
    array_span = f"0-{num_jobs-1}"
    submit_cmd = f"sbatch --array={array_span} {sbatch_path}"
    print(f"Submitting array with: {submit_cmd}")
    os.system(submit_cmd)
    print(f"Submitted array {array_span}; sbatch script at: {sbatch_path}")


if __name__ == "__main__":
    main()
