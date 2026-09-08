#!/usr/bin/env python3
"""Isolated before/after performance worker; synthetic data, no scientific fits.

Run each case in a fresh process. Baseline methods come from --baseline-revision
without checking out or editing source. See scripts/shared_latent_performance.md.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import types


def load_baseline(path, name, revision):
    source = subprocess.check_output(["git", "show", f"{revision}:{path}"], text=True)
    module = types.ModuleType(name)
    module.__file__ = f"{revision}:{path}"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "row", "checkpoint", "optimized"], default="optimized")
    parser.add_argument("--variant", choices=["slb", "analytic", "expm"], default="analytic")
    parser.add_argument("--stage", choices=["likelihood", "training", "prediction", "plot", "loo", "posterior"], default="likelihood")
    parser.add_argument("--case", choices=["normal", "fallback"], default="normal")
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--queries", type=int, default=800)
    parser.add_argument("--draws", type=int, default=750)
    parser.add_argument("--cpus", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--baseline-revision", default="3904cdda87f33d92b728058214e7c865463c39ea")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode in {"row", "checkpoint"} and args.stage != "likelihood":
        parser.error("Individual kernel optimizations are benchmarked with --stage likelihood")
    if args.variant == "slb" and args.stage == "posterior":
        parser.error("Posterior batching changes apply only to band-pole moments")
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.cpus])
    os.environ.update(JAX_ENABLE_X64="true", OPENBLAS_NUM_THREADS="1",
                      OMP_NUM_THREADS="1", NUM_CORES=str(args.cpus))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from tinygp import GaussianProcess
    from qvc.light_curve import multiband_model_shared_latent_band_poles_blr as bp
    from qvc.light_curve.multiband_model_shared_latent_blr import make_multiband_shared_latent_blr_model as slb_factory
    from qvc.light_curve import band_poles_fit

    jax.config.update("jax_enable_x64", True)
    if args.mode != "optimized" and args.variant != "slb":
        old = load_baseline("src/qvc/light_curve/multiband_model_shared_latent_band_poles_blr.py",
            "qvc.light_curve._performance_baseline_kernel", args.baseline_revision)
        if args.mode in {"baseline", "checkpoint"}:
            bp.SharedLatentBandPolesBLRQS.stationary_covariance = old.SharedLatentBandPolesBLRQS.stationary_covariance
        if args.mode in {"baseline", "row"}:
            bp.SharedLatentBandPolesBLRQS._transition_matrices = old.SharedLatentBandPolesBLRQS._transition_matrices

    rng = np.random.default_rng(42)
    X = jnp.asarray(np.sort(rng.uniform(0, 5000, args.n))), jnp.arange(args.n) % 4
    query = jnp.asarray(rng.uniform(-100, 5100, args.queries)), jnp.arange(args.queries) % 4
    y = jnp.asarray(rng.normal(0, .08, args.n))
    factory = slb_factory if args.variant == "slb" else bp.make_multiband_shared_latent_band_poles_blr_model
    model = factory(X, y, jnp.full(args.n, .02), n_band=4, zero_mean=True,
        has_jitter=False, **({} if args.variant == "slb" else {"transition": args.variant}))
    ratios = jnp.array([.7, 1., 1.3, 1.6])
    theta = jnp.array([np.log(2.), np.log(2.0002 if args.case == "fallback" else 300.),
                      np.log(3.), np.log(100.), np.log(.1), 0. if args.case == "fallback" else .5])

    def parameters(p):
        f, s, lag, blr, amp = jnp.exp(p[:5])
        return dict(tau_fast_driver=f, tau_slow_driver=s, tau_slow_uv_driver=s,
            tau_slow_band=s * ratios**p[5], lag_disk=lag * ratios**(4/3),
            lag_blr=blr * jnp.ones(4), amp_cont_relflux=amp * ratios**(-.8),
            amp_blr_relflux=.02 * jnp.ones(4))

    def gp_at(p):
        return model._build_gp(parameters(p))[0]

    functions = {}
    if args.stage == "likelihood":
        likelihood = lambda p: gp_at(p).log_probability(y)
        functions = {"likelihood": jax.jit(likelihood),
                     "gradient": jax.jit(jax.value_and_grad(likelihood))}
    elif args.stage in {"training", "prediction"}:
        if args.mode == "baseline":
            def legacy(p):
                pred = gp_at(p).condition(y, X if args.stage == "training" else query).gp
                return pred.loc, jnp.sqrt(pred.variance)
            fn = jax.jit(legacy)
        elif args.stage == "training":
            fn = jax.jit(lambda p: model.pred_training_mean(parameters(p)))
        else:
            fn = jax.jit(lambda p: model.pred(parameters(p), query))
        functions = {args.stage: fn}
    elif args.stage == "loo":
        from qvc.light_curve import fit_light_curves as fit
        if args.mode == "baseline":
            from qvc.light_curve import quasisep_prediction
            quasisep_prediction.supports_shared_prediction = lambda _: False

        def fn(p):
            draws = {k: np.stack([v, v]) for k, v in parameters(p).items()}
            result = fit.compute_loo_short_lag_residual_diagnostics(
                model, draws, {"object_id": "benchmark", "z": 1.}, ["g", "r", "i", "z"])
            if not result["loo_resid_valid"]:
                raise ValueError(result.get("loo_resid_error"))
            return np.asarray([result["loo_chi2_eff"], result["loo_rms"]])
        functions = {"loo": fn}
    elif args.stage == "plot":
        from qvc.light_curve.multiband_fit_plotting import _predict_regular_band_grid
        if args.mode == "baseline":
            import equinox as eqx

            @eqx.filter_jit
            def legacy_pred(self, p, points):
                gp, inds = self._build_gp(p)
                pred = gp.condition(self._observed_y_sorted(p, inds), points).gp
                return pred.loc, jnp.sqrt(pred.variance)

            type(model).pred = legacy_pred
        functions = {"plot": lambda p: _predict_regular_band_grid(
            model, parameters(p), np.linspace(-100, 5100, args.queries), np.arange(4))}
    else:
        p = {k: np.broadcast_to(v, (args.draws,) + np.shape(v)) for k, v in parameters(theta).items()}
        p["lag0"] = np.full(args.draws, 3.)
        p["lambda_center_rf"] = np.full(args.draws, 2500.)
        bands = ["g", "r", "i", "z"]
        for key in ["tau_slow_band", "lag_disk", "lag_blr", "amp_cont_relflux", "amp_blr_relflux"]:
            for i, band in enumerate(bands):
                p[f"{key}_{band}"] = p[key][:, i]
        module = band_poles_fit if args.mode == "optimized" else load_baseline(
            "src/qvc/light_curve/band_poles_fit.py", "qvc.light_curve._performance_baseline_fit", args.baseline_revision)
        functions = {"posterior": lambda _: module.posterior_band_poles_moments(p, bands)}

    results = {}
    for name, fn in functions.items():
        start = time.perf_counter()
        value = jax.block_until_ready(fn(theta))
        first = time.perf_counter() - start
        jax.block_until_ready(fn(theta))
        runs = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            value = jax.block_until_ready(fn(theta))
            runs.append(time.perf_counter() - start)
        if not all(np.isfinite(x).all() for x in jax.tree.leaves(value)):
            raise ValueError("Nonfinite benchmark output")
        result = dict(first_call_sec=first, median_sec=float(np.median(runs)), runs_sec=runs,
                      output_sums=[float(np.sum(x)) for x in jax.tree.leaves(value)])
        if hasattr(fn, "lower"):
            memory = fn.lower(theta).compile().memory_analysis()
            result["compiler_temp_mib"] = memory.temp_size_in_bytes / 2**20
        results[name] = result
    protected = Path("src/qvc/light_curve/multiband_model_shared_latent_blr.py")
    result = {**vars(args), "output": str(args.output), "jax": jax.__version__,
        "slb_sha256": hashlib.sha256(protected.read_bytes()).hexdigest(),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "measurements": results}
    output = json.dumps(result, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    main()
