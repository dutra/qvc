#!/usr/bin/env xonsh

# Reproduce the Stone no-flux-guard HPC fit directly on this machine:
# xonsh run_lc.xsh

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

object_ids = [
    #1406458, good lr0003
    1467775,
    1385200,
    1437146

    #1443303,
    #1388730,
    # 1387321,
    # 1390600,
    # 1391093,
    # 1388129,
]
description = "aug28d_erlang_dho_iters3_svithennuts_psflogitnormal_svi10000w250s100lr0003_stonechisq_specaug24_v3"
nproc = 3
repo_root = Path(__file__).resolve().parent
spectra_fit_h5 = repo_root / (
    "results/data/jaxqsofit/"
    "aug24_0152pm_spectrafit_e5d2897_chisqgt20_N8000_nested_"
    "fhostpsf_resumed_m2500norm12_v3.h5"
)
if not spectra_fit_h5.is_file():
    raise FileNotFoundError(f"Required v3 spectra catalog not found: {spectra_fit_h5}")

print(f"Description: {description}")
print(f"Object IDs: {object_ids}")
print(f"Parallel processes: {nproc}")
print(f"Spectra catalog: {spectra_fit_h5}")
print("Model: flux-linearized Erlang BLR + legacy two-timescale continuum")

env = os.environ.copy()
env["PREFIX"] = description

start = perf_counter()


def fit_object(index, object_id):
    object_id = str(object_id)
    object_start = perf_counter()
    object_env = env.copy()
    object_env["SUFFIX"] = f"{description}_{object_id}"
    command = [
        sys.executable,
        "-m", "qvc.light_curve.fit_light_curves",
        "--dho_drw_parameterization",
        "--filter_object_id", object_id,
        "--progress",
        "--outlier_half_window_days", "60",
        "--svi_steps", "10000",
        "--svi_lr", "0.0003",
        "--nwarm", "250",
        "--nsamp", "100",
        "--nchains", "3",
        "--max_tree_depth", "8",
        "--target_accept", "0.7",
        "--model_variant", "mag_flux_linearized_erlang",
        "--flux_linearized_refinement_iters", "3",
        "--flux_linearized_refinement_strategy", "svi_then_nuts",
        "--fit_method", "svi+nuts",
        "--plot",
        "--disable_trace_plot",
        "--disable_correlation_plot",
        "--disable_histogram_plot",
        "--disable_corner_plot",
        "--disable_color_magnitude_plot",
        "--disable_recovery_plot",
        "--disable_sigma_tau_lambda_plot",
        #"--spectra_fit_h5", str(spectra_fit_h5),
        #"--psf-fraction-mode", "logit-normal",
        #"--subtract_psf_constant_flux",
#        "--fast_solver",

    ]

    print(f"[{index}/{len(object_ids)}] Fitting object {object_id}")
    result = subprocess.run(command, env=object_env, check=False)
    if result.returncode != 0:
        print(f"[{object_id}] Failed with exit code {result.returncode}")
        return object_id, result.returncode
    elapsed = perf_counter() - object_start
    print(f"[{object_id}] Finished in {elapsed:.1f} seconds")
    return object_id, 0


if nproc == 1:
    outcomes = [
        fit_object(index, object_id)
        for index, object_id in enumerate(object_ids, start=1)
    ]
else:
    outcomes = []
    with ThreadPoolExecutor(max_workers=nproc) as executor:
        futures = {
            executor.submit(fit_object, index, object_id): object_id
            for index, object_id in enumerate(object_ids, start=1)
        }
        for future in as_completed(futures):
            outcomes.append(future.result())

elapsed = perf_counter() - start
print(f"Total run time: {elapsed:.1f} seconds")
failures = [(object_id, code) for object_id, code in outcomes if code != 0]
if failures:
    print(f"Failed fits: {failures}")
    raise SystemExit(1)
