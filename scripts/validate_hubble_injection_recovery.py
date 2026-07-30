#!/usr/bin/env python
"""Run reproducible AGN Hubble sigma/tau injection-recovery comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble import hubble_injection_recovery as recovery


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        choices=recovery.SELECTION_MODELS,
        default=None,
    )
    parser.add_argument(
        "--candidate",
        choices=recovery.COMPLETENESS_CANDIDATES,
        nargs="+",
        default=["none"],
    )
    parser.add_argument(
        "--backend",
        choices=("fast", "production"),
        default="fast",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-parent", type=int, default=None)
    parser.add_argument(
        "--predictor-noise",
        choices=recovery.PREDICTOR_NOISE_MODES,
        default=None,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-injection", type=Path, default=None)
    parser.add_argument("--load-injection", type=Path, default=None)
    return parser


def _default_truth() -> recovery.HubbleInjectionTruth:
    import numpy as np

    return recovery.HubbleInjectionTruth(
        H0=70.0,
        Om0=0.3,
        M0_agn=-23.4,
        alpha_agn=-1.8,
        beta_agn=-0.9,
        log_f=np.log(0.35),
        reference_pivots=(np.log10(0.2), np.log10(300.0)),
    )


def _load_or_generate(args, parser):
    generation_values = (args.selection, args.seed, args.n_parent, args.predictor_noise)
    if args.load_injection is not None:
        if any(value is not None for value in generation_values):
            parser.error(
                "--load-injection cannot be combined with --selection, --seed, "
                "--n-parent, or --predictor-noise."
            )
        return recovery.load_injection_hdf5(args.load_injection)

    config = recovery.HubbleInjectionConfig(
        seed=20260728 if args.seed is None else args.seed,
        n_parent=2500 if args.n_parent is None else args.n_parent,
        z_range=(0.44, 3.16),
        selection_model="none" if args.selection is None else args.selection,
        predictor_noise=(
            "noiseless" if args.predictor_noise is None else args.predictor_noise
        ),
    )
    return recovery.generate_hubble_injection(config, _default_truth())


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    dataset = _load_or_generate(args, parser)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for candidate in args.candidate:
        completeness_params = recovery.build_completeness_candidate(
            candidate,
            dataset,
            args.output_dir / "candidate_work" / candidate,
        )
        if args.backend == "fast":
            result = recovery.fit_fixed_cosmology(dataset, completeness_params)
        else:
            result = recovery.run_joint_sampler_recovery(
                dataset,
                candidate,
                args.output_dir,
            )
        results.append(result)

    paths = recovery.write_recovery_artifacts(
        dataset,
        results,
        args.output_dir,
    )
    if args.save_injection is not None:
        args.save_injection.parent.mkdir(parents=True, exist_ok=True)
        if args.save_injection.resolve() != paths["injection"].resolve():
            shutil.copyfile(paths["injection"], args.save_injection)

    print(f"Injection dataset: {dataset.dataset_id}")
    print(f"Selected objects: {len(dataset.selected)}/{len(dataset.parent)}")
    for result in results:
        print(
            f"{result.candidate}: parameter_rmse={result.metrics['parameter_rmse']:.4f}, "
            f"residual_z_slope={result.metrics['residual_z_slope']:.4f}, "
            f"runtime={result.runtime_seconds:.2f}s"
        )
    print(f"Artifacts: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
