#!/usr/bin/env python3
"""Run the fixed-truth, four-arm AGN Hubble validation campaign."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble.completeness_mock_catalog import build_completeness_lf
from qvc.hubble.hubble_fit import (
    _validate_checkpoint_prior_metadata,
    canonical_prior_bounds_json,
    run_mcmc_pipeline,
)
from qvc.hubble.hubble_model import (
    DEFAULT_PRIOR_PROFILE,
    PRIOR_PROFILE_CHOICES,
    build_agn_pivot_context,
    get_model_params,
)
from qvc.hubble.cuts import (
    COMPLETENESS_MAG_2500_MAX,
    COMPLETENESS_MAG_2500_MIN,
)
from qvc.hubble.hubble_utils import load_chains
from qvc.hubble.hubble_validation import (
    ARM_NAMES,
    ValidationTruth,
    analytic_completeness_params,
    collect_recovery_fragments,
    config_fingerprint,
    derive_seed_ledger,
    ensemble_summary,
    generate_calibration_catalog,
    generate_matched_fit_catalogs,
    posterior_summary_row,
    write_completeness_parent_hdf5,
    write_dataframe_atomic,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        default="fixed_truth",
        help="Campaign name below results/hubble_validation.",
    )
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results" / "hubble_validation")
    parser.add_argument("--n-runs", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--master-seed", type=int, default=20260901)
    parser.add_argument(
        "--n-agn",
        "--num-agns",
        dest="n_agn",
        type=int,
        default=2000,
        help="Number of AGNs in each unselected and selected fit sample.",
    )
    parser.add_argument(
        "--calibration-size",
        type=int,
        default=200000,
        help="Number of independent parent objects used to estimate each 2D map.",
    )
    parser.add_argument("--lf-area-deg2", type=float, default=20.0)
    parser.add_argument("--lf-model", default="wang2026_type1_lade_a")
    parser.add_argument("--z-min", type=float, default=0.1)
    parser.add_argument("--z-max", type=float, default=4.0)
    parser.add_argument("--lf-mag-min", type=float, default=14.0)
    parser.add_argument("--lf-mag-max", type=float, default=28.0)
    parser.add_argument("--h0", type=float, default=70.0)
    parser.add_argument("--om0", type=float, default=0.30)
    parser.add_argument("--w0", type=float, default=-1.0)
    parser.add_argument("--wa", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=7.0)
    parser.add_argument("--beta", type=float, default=-1.0)
    parser.add_argument("--m0", type=float, default=-23.0)
    parser.add_argument("--scatter-mag", type=float, default=0.5)
    parser.add_argument("--log-sigma-pivot", type=float, default=-0.8)
    parser.add_argument("--log-sigma-scale", type=float, default=0.2)
    parser.add_argument("--log-tau-pivot", type=float, default=2.7)
    parser.add_argument("--log-tau-scale", type=float, default=0.4)
    parser.add_argument("--m50", type=float, default=23.0)
    parser.add_argument("--selection-width", type=float, default=0.3)
    parser.add_argument("--speed", default="production")
    parser.add_argument(
        "--prior-profile",
        choices=PRIOR_PROFILE_CHOICES,
        default=DEFAULT_PRIOR_PROFILE,
        help="Named Hubble-fit prior profile.",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ARM_NAMES,
        default=list(ARM_NAMES),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--initialize-only",
        action="store_true",
        help="Create or validate campaign metadata without generating catalogs or fitting.",
    )
    parser.add_argument(
        "--realization",
        type=int,
        help="Run exactly one realization from the configured campaign seed range.",
    )
    parser.add_argument(
        "--simulate-only",
        action="store_true",
        help="Generate and persist catalogs without starting Dynesty.",
    )
    return parser


def _git_provenance() -> dict:
    def run(*arguments):
        completed = subprocess.run(
            ["git", *arguments], cwd=REPO_ROOT, text=True, capture_output=True, check=False
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def _truth_from_args(args) -> ValidationTruth:
    truth = ValidationTruth(
        h0=args.h0,
        om0=args.om0,
        w0=args.w0,
        wa=args.wa,
        alpha_agn=args.alpha,
        beta_agn=args.beta,
        m0_agn=args.m0,
        intrinsic_scatter_mag=args.scatter_mag,
        log_sigma_pivot=args.log_sigma_pivot,
        log_sigma_scale=args.log_sigma_scale,
        log_tau_pivot=args.log_tau_pivot,
        log_tau_scale=args.log_tau_scale,
    )
    truth.validate()
    return truth


def _configuration(args, truth: ValidationTruth) -> dict:
    if args.n_runs <= 0 or args.n_agn <= 0 or args.calibration_size <= 0:
        raise ValueError("n-runs, n-agn, and calibration-size must be positive.")
    if args.seed_start < 0:
        raise ValueError("seed-start must be nonnegative.")
    if not args.z_min < args.z_max or not args.lf_mag_min < args.lf_mag_max:
        raise ValueError("Redshift and LF magnitude bounds must be increasing.")
    if args.selection_width <= 0.0 or args.lf_area_deg2 <= 0.0:
        raise ValueError("Selection width and LF area must be positive.")
    campaign_path = Path(args.campaign)
    if not args.campaign.strip() or campaign_path.is_absolute() or ".." in campaign_path.parts:
        raise ValueError("campaign must be a relative path below output-root.")
    configuration = {
        "schema_version": 2,
        "truth": asdict(truth),
        "n_runs": int(args.n_runs),
        "seed_start": int(args.seed_start),
        "master_seed": int(args.master_seed),
        "n_agn": int(args.n_agn),
        "calibration_size": int(args.calibration_size),
        "lf_area_deg2": float(args.lf_area_deg2),
        "lf_model": str(args.lf_model),
        "z_range": [float(args.z_min), float(args.z_max)],
        "lf_apparent_magnitude_support": [float(args.lf_mag_min), float(args.lf_mag_max)],
        "selection": {"m50": float(args.m50), "width": float(args.selection_width)},
        "fit": {
            "model": "Flatw0waCDM",
            "only_agn": True,
            "fixed_h0": float(args.h0),
            "speed": str(args.speed),
            "minimal_plots": True,
            "sigma_clipping": False,
            "completeness_magnitude_support_mode": "hard-cut",
            "completeness_magnitude_support": [
                float(COMPLETENESS_MAG_2500_MIN),
                float(COMPLETENESS_MAG_2500_MAX),
            ],
        },
        "arms": list(args.arms),
    }
    if args.prior_profile != DEFAULT_PRIOR_PROFILE:
        priors, _, _ = get_model_params(
            "Flatw0waCDM",
            only_agn=True,
            fixed_h0=args.h0,
            prior_profile=args.prior_profile,
        )
        configuration["fit"]["prior_profile"] = args.prior_profile
        configuration["fit"]["prior_bounds"] = {
            name: [float(bounds[0]), float(bounds[1])]
            for name, bounds in priors.items()
        }
    return configuration


def _write_or_validate_manifest(campaign_dir: Path, configuration: dict, resume: bool) -> None:
    manifest_path = campaign_dir / "manifest.json"
    fingerprint = config_fingerprint(configuration)
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text())
        if not resume:
            raise FileExistsError(
                f"Campaign already exists: {campaign_dir}. Pass --resume to continue it."
            )
        if stored.get("configuration_fingerprint") != fingerprint:
            raise RuntimeError(
                "Existing campaign configuration does not match the requested configuration."
            )
        return
    campaign_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_fingerprint": fingerprint,
        "configuration": configuration,
        "provenance": _git_provenance(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _campaign_design(args) -> pd.DataFrame:
    rows = [
        {
            "realization": realization,
            **derive_seed_ledger(args.master_seed, realization),
        }
        for realization in range(args.seed_start, args.seed_start + args.n_runs)
    ]
    return pd.DataFrame(rows)


def _write_or_validate_seed_ledger(campaign_dir: Path, design: pd.DataFrame) -> None:
    path = campaign_dir / "seed_ledger.csv"
    if path.is_file():
        stored = pd.read_csv(path)
        try:
            pd.testing.assert_frame_equal(
                stored.reset_index(drop=True),
                design.reset_index(drop=True),
                check_dtype=False,
            )
        except AssertionError as exc:
            raise RuntimeError(
                "Existing campaign seed ledger does not match the requested configuration."
            ) from exc
        return
    write_dataframe_atomic(design, path)


def _load_seed_recovery(campaign_dir: Path, realization_dir: Path, realization: int) -> pd.DataFrame:
    """Load a seed fragment, migrating matching legacy campaign rows if needed."""

    fragment_path = realization_dir / "recovery.csv"
    if fragment_path.is_file():
        return pd.read_csv(fragment_path)
    campaign_path = campaign_dir / "recovery.csv"
    if campaign_path.is_file():
        campaign_recovery = pd.read_csv(campaign_path)
        if "realization" in campaign_recovery:
            fragment = campaign_recovery.loc[
                campaign_recovery["realization"] == realization
            ].copy()
            if not fragment.empty:
                write_dataframe_atomic(fragment, fragment_path)
                return fragment
    return pd.DataFrame()


def _restore_catalog(path: Path, metadata: dict) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"object_id": str})
    frame.attrs.update(
        {
            "completeness_magnitude": "dereddened",
            "completeness_magnitude_source": "m_2500_dereddened",
            "completeness_magnitude_err_source": "m_2500_dereddened_err",
            "completeness_magnitude_support_mode": "hard-cut",
            **metadata,
        }
    )
    return frame


def _persist_generation(
    realization_dir: Path,
    all_frame: pd.DataFrame,
    selected_frame: pd.DataFrame,
    calibration_parent: pd.DataFrame | None,
    calibration_detected: pd.DataFrame | None,
    *,
    z_range,
    seed_ledger,
) -> dict:
    all_frame.to_csv(realization_dir / "all.csv", index=False)
    selected_frame.to_csv(realization_dir / "selected.csv", index=False)
    metadata = {
        "n_parent_generated": int(all_frame.attrs["n_parent_generated"]),
        "n_detected_generated": int(all_frame.attrs["n_detected_generated"]),
        "detection_fraction": float(all_frame.attrs["detection_fraction"]),
        "seed_ledger": seed_ledger,
    }
    if calibration_parent is not None and calibration_detected is not None:
        calibration_detected.to_csv(realization_dir / "calibration_detected.csv", index=False)
        write_completeness_parent_hdf5(
            calibration_parent,
            realization_dir / "calibration_parent.h5",
            z_range=z_range,
        )
        metadata.update(
            {
                "calibration_parent_count": int(len(calibration_parent)),
                "calibration_detected_count": int(len(calibration_detected)),
            }
        )
    (realization_dir / "generation.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def _load_or_generate_catalogs(
    realization_dir,
    *,
    args,
    truth,
    lf_grid,
    cosmology,
    seed_ledger,
):
    metadata_path = realization_dir / "generation.json"
    required = [realization_dir / "all.csv", realization_dir / "selected.csv"]
    need_calibration = "selected_estimated" in args.arms
    if need_calibration:
        required.extend(
            [realization_dir / "calibration_detected.csv", realization_dir / "calibration_parent.h5"]
        )
    if metadata_path.exists() and all(path.exists() for path in required):
        metadata = json.loads(metadata_path.read_text())
        all_frame = _restore_catalog(required[0], metadata)
        selected_frame = _restore_catalog(required[1], metadata)
        calibration_detected = (
            _restore_catalog(realization_dir / "calibration_detected.csv", {})
            if need_calibration
            else None
        )
        return all_frame, selected_frame, calibration_detected, metadata

    all_frame, selected_frame = generate_matched_fit_catalogs(
        lf_grid,
        cosmology,
        truth=truth,
        n_fit=args.n_agn,
        m50=args.m50,
        selection_width=args.selection_width,
        population_rng=np.random.default_rng(seed_ledger["population"]),
        scatter_rng=np.random.default_rng(seed_ledger["scatter"]),
        selection_rng=np.random.default_rng(seed_ledger["selection"]),
        area_deg2=args.lf_area_deg2,
        z_range=(args.z_min, args.z_max),
        apparent_magnitude_support=(args.lf_mag_min, args.lf_mag_max),
    )
    calibration_parent = calibration_detected = None
    if need_calibration:
        calibration_parent, calibration_detected = generate_calibration_catalog(
            lf_grid,
            cosmology,
            truth=truth,
            n_parent=args.calibration_size,
            m50=args.m50,
            selection_width=args.selection_width,
            population_rng=np.random.default_rng(seed_ledger["calibration_population"]),
            scatter_rng=np.random.default_rng(seed_ledger["calibration_scatter"]),
            selection_rng=np.random.default_rng(seed_ledger["calibration_selection"]),
            area_deg2=args.lf_area_deg2,
            z_range=(args.z_min, args.z_max),
            apparent_magnitude_support=(args.lf_mag_min, args.lf_mag_max),
        )
    metadata = _persist_generation(
        realization_dir,
        all_frame,
        selected_frame,
        calibration_parent,
        calibration_detected,
        z_range=(args.z_min, args.z_max),
        seed_ledger=seed_ledger,
    )
    return all_frame, selected_frame, calibration_detected, metadata


def _fit_arm(
    arm,
    *,
    args,
    truth,
    all_frame,
    selected_frame,
    calibration_detected,
    realization_dir,
    realization,
    generation_metadata,
    seed_ledger,
):
    source_frame = all_frame if arm == "all" else selected_frame
    fit_frame = source_frame.copy()
    fit_frame.attrs.update(source_frame.attrs)
    completeness = arm in {"selected_oracle", "selected_estimated"}
    oracle = (
        analytic_completeness_params(args.m50, args.selection_width)
        if arm == "selected_oracle"
        else None
    )
    if arm == "selected_oracle":
        fit_frame.attrs["completeness_magnitude_support_mode"] = "hard-cut"
    completeness_file = (
        realization_dir / "calibration_parent.h5"
        if arm == "selected_estimated"
        else None
    )
    completeness_frame = calibration_detected if arm == "selected_estimated" else fit_frame
    checkpoint_file = realization_dir / f"posterior_{arm}.h5"
    prefix = f"hubble_validation/{args.campaign}/seed_{realization:04d}/{arm}"
    pivot_context = build_agn_pivot_context(
        fit_frame, (args.z_min, args.z_max)
    )
    pivot_values = pivot_context.as_dict()
    parameter_truths = truth.parameter_truths()
    parameter_truths["M0_agn"] = (
        truth.m0_agn
        + truth.alpha_agn
        * (pivot_values["log_sigma_uv"] - truth.log_sigma_pivot)
        + truth.beta_agn
        * (pivot_values["log_tau_uv_rf"] - truth.log_tau_pivot)
    )
    priors, _, _ = get_model_params(
        "Flatw0waCDM",
        only_agn=True,
        fixed_h0=truth.h0,
        prior_profile=args.prior_profile,
    )
    prior_bounds_json = canonical_prior_bounds_json(priors)
    if args.resume and checkpoint_file.is_file():
        checkpoint = load_chains(checkpoint_file)
        _validate_checkpoint_prior_metadata(
            checkpoint,
            checkpoint_file,
            expected_prior_profile=args.prior_profile,
            expected_prior_bounds_json=prior_bounds_json,
            expected_early_de_guard=False,
        )
        if "model_labels" not in checkpoint:
            raise RuntimeError(
                f"Existing validation checkpoint lacks model_labels: {checkpoint_file}"
            )
        model_labels = tuple(
            value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in np.asarray(checkpoint["model_labels"]).tolist()
        )
        checkpoint_ids = tuple(
            value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in np.asarray(checkpoint["object_id_fit_selection"]).tolist()
        )
        if checkpoint_ids != tuple(fit_frame["object_id"].astype(str)):
            raise RuntimeError(
                f"Existing validation checkpoint object IDs do not match {arm}: {checkpoint_file}"
            )
        return posterior_summary_row(
            checkpoint["flat_samples"],
            model_labels,
            arm=arm,
            realization=realization,
            checkpoint_file=checkpoint_file,
            truth=truth,
            n_fit=len(fit_frame),
            n_parent_generated=generation_metadata["n_parent_generated"],
            detection_fraction=generation_metadata["detection_fraction"],
            parameter_truths=parameter_truths,
        )
    empty = np.empty((0, 0), dtype=float)
    result = run_mcmc_pipeline(
        fit_frame,
        all_frame,
        pd.DataFrame(),
        empty,
        empty,
        empty,
        agn_pivot_context=pivot_context,
        cosmo_model="Flatw0waCDM",
        only_agn=True,
        completeness=completeness,
        use_full_cov=False,
        resume=False,
        speed=args.speed,
        z_range=(args.z_min, args.z_max),
        prefix=prefix,
        checkpoint_file_override=checkpoint_file,
        completeness_sim_file=completeness_file,
        completeness_params_override=oracle,
        completeness_mode="2d",
        completeness_magnitude="dereddened",
        selection_attenuation_mode="fixed-offset",
        light_curve_uncertainty_mode="covariance",
        N=len(fit_frame),
        minimal_plots=True,
        disable_ceph_dist_calibration=False,
        use_planck_h0_prior=False,
        use_planck_om_prior=False,
        prior_profile=args.prior_profile,
        fixed_h0=truth.h0,
        rng_seed=seed_ledger[f"inference_{arm}"],
        df_agn_completeness=completeness_frame,
        completeness_z_range=(args.z_min, args.z_max),
    )
    flat_samples, model_labels = result[:2]
    return posterior_summary_row(
        flat_samples,
        model_labels,
        arm=arm,
        realization=realization,
        checkpoint_file=checkpoint_file,
        truth=truth,
        n_fit=len(fit_frame),
        n_parent_generated=generation_metadata["n_parent_generated"],
        detection_fraction=generation_metadata["detection_fraction"],
        parameter_truths=parameter_truths,
    )


def _upsert_recovery(path: Path, row: dict) -> pd.DataFrame:
    if path.exists():
        recovery = pd.read_csv(path)
        same = (recovery["realization"] == row["realization"]) & (recovery["arm"] == row["arm"])
        recovery = recovery.loc[~same].copy()
    else:
        recovery = pd.DataFrame()
    recovery = pd.concat([recovery, pd.DataFrame([row])], ignore_index=True)
    recovery = recovery.sort_values(["realization", "arm"], kind="stable")
    write_dataframe_atomic(recovery, path)
    return recovery


def _write_status(realization_dir: Path, arm: str, status: str, **details) -> None:
    payload = {
        "arm": arm,
        "status": status,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    path = realization_dir / f"status_{arm}.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    truth = _truth_from_args(args)
    configuration = _configuration(args, truth)
    campaign_dir = args.output_root.expanduser().resolve() / args.campaign
    _write_or_validate_manifest(campaign_dir, configuration, args.resume)
    design = _campaign_design(args)
    _write_or_validate_seed_ledger(campaign_dir, design)
    if args.initialize_only:
        print(f"Initialized campaign: {campaign_dir}")
        return 0

    if args.realization is not None:
        valid_realizations = set(design["realization"].astype(int))
        if args.realization not in valid_realizations:
            first = int(design["realization"].min())
            last = int(design["realization"].max())
            raise ValueError(
                f"realization {args.realization} is outside the configured range {first}-{last}."
            )
        design = design.loc[design["realization"] == args.realization].copy()

    cosmology = FlatLambdaCDM(H0=truth.h0, Om0=truth.om0)
    lf_grid = build_completeness_lf(
        args.lf_model,
        z_range=(args.z_min, args.z_max),
        target_cosmology=cosmology,
    )
    for design_row in design.to_dict("records"):
        realization = int(design_row["realization"])
        seed_ledger = {
            key: int(value) for key, value in design_row.items() if key != "realization"
        }
        realization_dir = campaign_dir / "runs" / f"seed_{realization:04d}"
        realization_dir.mkdir(parents=True, exist_ok=True)
        recovery_path = realization_dir / "recovery.csv"
        recovery = _load_seed_recovery(campaign_dir, realization_dir, realization)
        all_frame, selected_frame, calibration_detected, generation_metadata = (
            _load_or_generate_catalogs(
                realization_dir,
                args=args,
                truth=truth,
                lf_grid=lf_grid,
                cosmology=cosmology,
                seed_ledger=seed_ledger,
            )
        )
        if args.simulate_only:
            print(f"seed {realization}: generated catalogs only")
            continue
        for arm in args.arms:
            previous = recovery.loc[
                (recovery.get("realization", pd.Series(dtype=int)) == realization)
                & (recovery.get("arm", pd.Series(dtype=str)) == arm)
            ]
            if not previous.empty:
                status = str(previous.iloc[-1].get("status", ""))
                if status == "complete" or (status == "failed" and not args.retry_failed):
                    print(f"seed {realization} / {arm}: skipping existing {status} row")
                    continue
            print(f"seed {realization} / {arm}: fitting", flush=True)
            _write_status(realization_dir, arm, "running")
            try:
                row = _fit_arm(
                    arm,
                    args=args,
                    truth=truth,
                    all_frame=all_frame,
                    selected_frame=selected_frame,
                    calibration_detected=calibration_detected,
                    realization_dir=realization_dir,
                    realization=realization,
                    generation_metadata=generation_metadata,
                    seed_ledger=seed_ledger,
                )
            except Exception as exc:
                row = {
                    "realization": realization,
                    "arm": arm,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                (realization_dir / f"error_{arm}.txt").write_text(traceback.format_exc())
                _write_status(
                    realization_dir,
                    arm,
                    "failed",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                recovery = _upsert_recovery(recovery_path, row)
                if args.fail_fast:
                    raise
                continue
            recovery = _upsert_recovery(recovery_path, row)
            _write_status(
                realization_dir,
                arm,
                "complete",
                checkpoint_file=row["checkpoint_file"],
                posterior_sample_count=row["posterior_sample_count"],
            )

    if args.realization is None:
        recovery = collect_recovery_fragments(campaign_dir)
        if not recovery.empty:
            write_dataframe_atomic(
                ensemble_summary(recovery), campaign_dir / "ensemble_summary.csv"
            )
    print(f"Campaign artifacts: {campaign_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
