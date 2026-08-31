#!/usr/bin/env xonsh

# Default: xonsh run_fit_sed_spectra.xsh
# Multiple objects in parallel: xonsh run_fit_sed_spectra.xsh parallel
# Multiple objects sequentially: xonsh run_fit_sed_spectra.xsh sequential

import sys
from time import perf_counter

# z0.304_013453.20-001842.3_joint.png
object_id = 1414639 # bad fits?
#object_id = 1458203 # bal referee
#object_id = 1460028 # bal referee

multi_object_ids = [
    1458203, 1460028,
    #1414639,
    #1467746, 1428411, 1390061, 1443547, 1392347,
    #1417521, 1451057, 1457809, 1420763, 1454455,
]
sed_photometry_path = "data/jul14_master_input_file_chisqgt20_bandwagon_photometry_updated.csv"

mode = sys.argv[1] if len(sys.argv) > 1 else "single"
if mode == "single":
    object_ids = [object_id]
    run_tag = f"aug31_bal_{object_id}_notebook_sdss_spectrum"
    nproc = 1
elif mode == "parallel":
    object_ids = multi_object_ids
    run_tag = "parallel"
    nproc = len(object_ids)
elif mode == "sequential":
    object_ids = multi_object_ids
    run_tag = "sequential"
    nproc = 1
else:
    print("Usage: xonsh run_fit_sed_spectra.xsh [single|parallel|sequential]")
    raise SystemExit(2)

fpath_out = f"results/data/spectra/sed_spectra_{run_tag}.h5"
output_dir = f"results/jaxsedfit_joint/{run_tag}"
fig_dir = f"plots/jaxsedfit_joint/{run_tag}"

print(f"Mode: {mode}")
print(f"Object IDs: {object_ids}")
print(f"Worker processes: {nproc}")

start = perf_counter()
python -m qvc.spectra.fit_spectra_jaxsedfit_joint \
    --mode fit \
    --fit-bal \
    --cache-dir "data/spectra_cache_all" \
    --verbose \
    --optax-steps 4000 \
    --optax-lr 0.005 \
    --nuts-warmup 250 \
    --nuts-samples 250 \
    --nuts-chains 1 \
    --sed-photometry-path @(sed_photometry_path) \
    --output-dir @(output_dir) \
    --fig-dir @(fig_dir) \
    --filter_object_id @(object_ids) \
    --progress \
    --nproc @(nproc) \
    @(fpath_out)

print(f"Total run time: {perf_counter() - start:.1f} seconds")
