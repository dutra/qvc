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
    1458203, 1460028, #referee
    1410570,
    #1414639, # bad fit?
    # definitely bal
    #1410570, 1422228, 1400339, 1425653,
    #1467746, 1428411, 1390061, 1443547, 1392347,
    #1417521, 1451057, 1457809, 1420763, 1454455,
]
sed_photometry_path = "data/jul14_master_input_file_chisqgt20_bandwagon_photometry.csv"

# z~1.5 gap test: BOSS objects with no fit in aug31_w500s500_d01c034 (the
# cluster jobs left no chunk, bundle or error row), plus controls that the
# cluster did fit. Run with: xonsh run_fit_sed_spectra.xsh bouchet
bouchet_gap_object_ids = [
    1444555,  # missing, z=1.518
    1444087,  # missing, z=1.513
    1444296,  # missing, z=1.547 (upper edge of the gap)
    1444490,  # control: fitted, z=1.508 (inside the gap)
    1441189,  # control: fitted, z=1.556 (BAL on, Hd out of coverage)
    1443859,  # control: fitted, z=1.493 (BAL off, Hd in coverage)
]

mode = sys.argv[1] if len(sys.argv) > 1 else "single"
if mode == "bouchet":
    # Same fit settings as the generated job in hpc_scripts/sfitspectra.xsh
    # (--fit-bal, 5000 Optax steps, 1 chain, nproc 1) except NUTS, which is
    # shortened to 200 warmup / 100 samples from the cluster's 500/500, and one
    # process per object like the chunk_size=1 array tasks. The cluster limits
    # were 3:00:00 wall time and 40G; /usr/bin/time -l reports both here.
    import os
    import subprocess
    run_tag = "bouchet_gap_test"
    for oid in bouchet_gap_object_ids:
        log_path = f"results/jaxsedfit_joint/{run_tag}/{oid}.log"
        mkdir -p @(f"results/jaxsedfit_joint/{run_tag}")
        cmd = [
            "/usr/bin/time", "-l",
            "python", "-m", "qvc.spectra.fit_spectra_jaxsedfit_joint",
            "--mode", "fit",
            f"results/data/spectra/sed_spectra_{run_tag}_{oid}.h5",
            "--cache-dir", "data/spectra_cache_all",
            "--verbose",
            "--optax-steps", "5000",
            "--optax-lr", "0.005",
            "--nuts-warmup", "200",
            "--nuts-samples", "100",
            "--nuts-chains", "1",
            "--output-dir", f"results/jaxsedfit_joint/{run_tag}/all",
            "--fig-dir", f"plots/jaxsedfit_joint/{run_tag}",
            "--filter_object_id", str(oid),
            "--nproc", "1",
            "--sed-photometry-path", sed_photometry_path,
            "--progress",
            "--fit-bal",
        ]
        n_done = bouchet_gap_object_ids.index(oid) + 1
        print(f"[bouchet {n_done}/{len(bouchet_gap_object_ids)}] object_id={oid} -> {log_path}")
        t0 = perf_counter()
        # Show the fit's progress bars live and keep a copy in the log.
        fit = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        tee = subprocess.Popen(["tee", log_path], stdin=fit.stdout)
        fit.stdout.close()
        tee.wait()
        rc = fit.wait()
        print(f"[bouchet {n_done}/{len(bouchet_gap_object_ids)}] object_id={oid} exit={rc} wall={perf_counter() - t0:.0f}s")
    raise SystemExit(0)

if mode == "single":
    object_ids = [object_id]
    run_tag = f"aug31b_nobal_{object_id}_notebook_sdss_spectrum"
    nproc = 1
elif mode == "parallel":
    object_ids = multi_object_ids
    run_tag = "parallel"
    nproc = len(object_ids)
elif mode == "sequential":
    object_ids = multi_object_ids
    run_tag = "sep1a_referee"
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
    --plot-init \
    --cache-dir "data/spectra_cache_all" \
    --verbose \
    --fit-method optax+nuts \
    --optax-steps 4000 \
    --optax-lr 0.005 \
    --nuts-warmup 500 \
    --nuts-samples 500 \
    --nuts-chains 4 \
    --sed-photometry-path @(sed_photometry_path) \
    --output-dir @(output_dir) \
    --fig-dir @(fig_dir) \
    --filter_object_id @(object_ids) \
    --progress \
    --nproc @(nproc) \
    @(fpath_out)

print(f"Total run time: {perf_counter() - start:.1f} seconds")
