#!/usr/bin/env python3
"""Compare warmed SLB/band-pole kernels in isolated CPU processes; no fitting."""

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def worker(args):
    # Equal CPU affinity, thread settings and synthetic coordinates per model.
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[: args.cpus])
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    )
    import jax
    import jax.numpy as jnp
    import numpy as np
    from tinygp import GaussianProcess
    from qvc.light_curve.multiband_model_shared_latent_blr import SharedLatentDiskBLRQS
    from qvc.light_curve.multiband_model_shared_latent_band_poles_blr import (
        SharedLatentBandPolesBLRQS,
    )

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(42)
    t = jnp.asarray(np.sort(rng.uniform(0.0, 5000.0, args.n)))
    b = jnp.asarray(np.arange(args.n) % 4)
    y = jnp.asarray(rng.normal(0.0, 0.08, args.n))
    ratios = jnp.array([0.7, 1.0, 1.3, 1.6])

    def likelihood(theta):
        f, s, lag, blr, amp = jnp.exp(theta[:5])
        common = dict(
            tau_fast=jnp.atleast_1d(f),
            lag_disk=lag * ratios ** (4 / 3),
            lag_blr=blr * jnp.ones(4),
            amp_cont=amp * ratios ** (-0.8),
            amp_blr=0.02 * jnp.ones(4),
            disk_order=3,
            blr_order=3,
        )
        if args.worker == "slb":
            k = SharedLatentDiskBLRQS(tau_slow=jnp.atleast_1d(s), **common)
        else:
            k = SharedLatentBandPolesBLRQS(
                tau_slow_uv=jnp.atleast_1d(s),
                tau_slow_band=s * ratios ** theta[5],
                transition=args.worker,
                **common,
            )
        return GaussianProcess(k, (t, b), diag=0.02**2).log_probability(y)

    theta = jnp.array(
        [
            np.log(2.0),
            np.log(2.0002 if args.case == "fallback" else 300.0),
            np.log(3.0),
            np.log(100.0),
            np.log(0.1),
            0.0 if args.case == "fallback" else 0.5,
        ]
    )
    compiled = [jax.jit(likelihood), jax.jit(jax.value_and_grad(likelihood))]
    timings = {}
    for name, fn in zip(["likelihood", "value_and_gradient"], compiled):
        start = time.perf_counter()
        jax.block_until_ready(fn(theta))
        timings[name + "_compile_and_first_sec"] = time.perf_counter() - start
        jax.block_until_ready(fn(theta))
        runs = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            value = jax.block_until_ready(fn(theta))
            runs.append(time.perf_counter() - start)
        for leaf in jax.tree.leaves(value):
            if not np.all(np.isfinite(leaf)):
                raise RuntimeError("Nonfinite benchmark output")
        timings[name + "_median_sec"] = float(np.median(runs))
        timings[name + "_runs_sec"] = runs
    print(
        json.dumps(
            dict(
                model=args.worker,
                case=args.case,
                local_expm_chains=(
                    8 if args.case == "fallback" and args.worker == "analytic" else 0
                ),
                n=args.n,
                bands=4,
                disk_order=3,
                blr_order=3,
                cpus=args.cpus,
                jax=jax.__version__,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                **timings,
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--cpus", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--worker", choices=["slb", "analytic", "expm"], help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--case",
        choices=["normal", "fallback"],
        default="normal",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--n", type=int, default=128, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    results = []
    for n in args.counts:
        for model, case in [
            ("slb", "normal"),
            ("analytic", "normal"),
            ("expm", "normal"),
            ("analytic", "fallback"),
            ("expm", "fallback"),
        ]:
            command = [
                sys.executable,
                __file__,
                "--worker",
                model,
                "--case",
                case,
                "--n",
                str(n),
                "--cpus",
                str(args.cpus),
                "--repeats",
                str(args.repeats),
            ]
            output = subprocess.check_output(command, text=True)
            row = json.loads(output.strip().splitlines()[-1])
            results.append(row)
            print(json.dumps(row), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "note": "Peak process RSS includes JAX compilation. Kernel likelihood and gradient only; no posterior fitting.",
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
