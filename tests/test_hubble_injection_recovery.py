import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble import hubble_fit_jax
from qvc.hubble.hubble_completeness_refactored import Completeness2D
from qvc.hubble.hubble_likelihood import completeness_loglike


def _small_injection(*, seed=41, selection_model="none", predictor_noise="realistic"):
    from qvc.hubble.hubble_injection_recovery import (
        HubbleInjectionConfig,
        HubbleInjectionTruth,
        generate_hubble_injection,
    )

    return generate_hubble_injection(
        HubbleInjectionConfig(
            seed=seed,
            n_parent=96,
            z_range=(0.44, 3.16),
            selection_model=selection_model,
            predictor_noise=predictor_noise,
        ),
        HubbleInjectionTruth(
            H0=70.0,
            Om0=0.3,
            M0_agn=-23.4,
            alpha_agn=-1.8,
            beta_agn=-0.9,
            log_f=np.log(0.35),
            reference_pivots=(np.log10(0.2), np.log10(300.0)),
        ),
    )


def _all_complete_2d(*, sigma_mag):
    mag_centers = np.linspace(18.5, 24.0, 60)
    z_centers = np.linspace(0.0, 4.0, 20)
    completeness = Completeness2D(
        mag_centers,
        z_centers,
        np.ones((mag_centers.size, z_centers.size)),
    )
    params = (
        completeness,
        mag_centers,
        z_centers,
        float(np.diff(mag_centers).mean()),
        float(np.diff(z_centers).mean()),
        float(sigma_mag),
    )
    return completeness, params


def test_jax_completeness_matches_cpu_for_bright_gaussian_tail():
    completeness, completeness_params = _all_complete_2d(sigma_mag=0.2)
    m_model = np.array([17.5])
    mu_err = np.array([0.3])
    z = np.array([1.0])

    cpu_log_z, _ = completeness_loglike(
        m_obs=m_model,
        m_obs_err=np.array([0.05]),
        m_model=m_model,
        mu_err=mu_err,
        z=z,
        completeness_model=completeness,
        m_grid=completeness_params[1],
        sigma_completeness=0.0,
    )
    jax_completeness = hubble_fit_jax._prepare_completeness_for_jax(
        completeness_params
    )
    jax_log_z = float(
        hubble_fit_jax._completeness_loglike_jax(
            m_model,
            mu_err,
            z,
            jax_completeness,
            None,
            None,
        )
    )

    np.testing.assert_allclose(jax_log_z, cpu_log_z, atol=1e-6, rtol=1e-6)


def test_jax_nonconstant_selection_matches_cpu_and_ignores_map_smoothing_metadata():
    mag_centers = np.linspace(18.5, 24.0, 121)
    z_centers = np.linspace(0.0, 4.0, 41)
    mm, zz = np.meshgrid(mag_centers, z_centers, indexing="ij")
    completeness_map = np.clip(
        0.90 - 0.11 * (mm - 18.5) - 0.04 * zz,
        0.02,
        0.95,
    )
    completeness = Completeness2D(
        mag_centers,
        z_centers,
        completeness_map,
    )
    m_model = np.array([17.7, 21.4, 23.7])
    mu_err = np.array([0.25, 0.35, 0.45])
    z = np.array([-0.5, 1.4, 4.5])

    cpu_log_z, _ = completeness_loglike(
        m_obs=m_model,
        m_obs_err=np.full(3, 0.05),
        m_model=m_model,
        mu_err=mu_err,
        z=z,
        completeness_model=completeness,
        m_grid=mag_centers,
        sigma_completeness=0.0,
    )
    jax_values = []
    for sigma_mag_metadata in (0.0, 99.0):
        params = (
            completeness,
            mag_centers,
            z_centers,
            float(np.diff(mag_centers).mean()),
            float(np.diff(z_centers).mean()),
            sigma_mag_metadata,
        )
        jax_values.append(
            float(
                hubble_fit_jax._completeness_loglike_jax(
                    m_model,
                    mu_err,
                    z,
                    hubble_fit_jax._prepare_completeness_for_jax(params),
                    None,
                    None,
                )
            )
        )

    assert abs(cpu_log_z) > 0.1
    np.testing.assert_allclose(jax_values, cpu_log_z, atol=1e-6, rtol=1e-6)


def test_hubble_injection_is_deterministic_and_transforms_truth_to_fit_pivot():
    from qvc.hubble.hubble_injection_recovery import (
        HubbleInjectionConfig,
        HubbleInjectionTruth,
        generate_hubble_injection,
    )

    truth = HubbleInjectionTruth(
        H0=70.0,
        Om0=0.3,
        M0_agn=-23.4,
        alpha_agn=-1.8,
        beta_agn=-0.9,
        log_f=np.log(0.35),
        reference_pivots=(np.log10(0.2), np.log10(300.0)),
    )
    config = HubbleInjectionConfig(
        seed=7,
        n_parent=256,
        z_range=(0.44, 3.16),
        selection_model="none",
        predictor_noise="noiseless",
    )

    first = generate_hubble_injection(config, truth)
    second = generate_hubble_injection(config, truth)

    pd.testing.assert_frame_equal(first.parent, second.parent, check_exact=True)
    pd.testing.assert_frame_equal(first.selected, second.selected, check_exact=True)
    np.testing.assert_array_equal(first.selected_mask, np.ones(config.n_parent, dtype=bool))
    np.testing.assert_array_equal(first.selection_probability, np.ones(config.n_parent))
    assert first.pivot_context == second.pivot_context
    assert first.dataset_id == second.dataset_id

    pivot_values = first.pivot_context.as_dict()
    expected_m0 = (
        truth.M0_agn
        + truth.alpha_agn
        * (pivot_values["log_sigma_uv"] - truth.reference_pivots[0])
        + truth.beta_agn
        * (pivot_values["log_tau_uv_rf"] - truth.reference_pivots[1])
    )
    assert first.truth_at_fit_pivot["M0_agn"] == expected_m0


def test_injection_pivot_ignores_extreme_objects_outside_fitted_redshift_range():
    from qvc.hubble.hubble_injection_recovery import (
        HubbleInjectionConfig,
        HubbleInjectionTruth,
        generate_hubble_injection,
    )
    from qvc.hubble.hubble_model import build_agn_pivot_context

    dataset = generate_hubble_injection(
        HubbleInjectionConfig(
            seed=29,
            n_parent=128,
            z_range=(0.44, 3.16),
            selection_model="none",
            predictor_noise="noiseless",
        ),
        HubbleInjectionTruth(
            H0=70.0,
            Om0=0.3,
            M0_agn=-23.4,
            alpha_agn=-1.8,
            beta_agn=-0.9,
            log_f=np.log(0.35),
            reference_pivots=(np.log10(0.2), np.log10(300.0)),
        ),
    )
    extremes = dataset.selected.iloc[:2].copy()
    extremes["object_id"] = ["outside_low", "outside_high"]
    extremes["z"] = [0.1, 4.0]
    extremes["log_sigma_uv"] = [-9.0, 9.0]
    extremes["log_tau_uv_rf"] = [-9.0, 9.0]
    extended = pd.concat([dataset.selected, extremes], ignore_index=True)

    extended_context = build_agn_pivot_context(
        extended,
        dataset.config.z_range,
    )

    assert extended_context == dataset.pivot_context


def test_injection_hdf5_round_trip_is_exact_and_missing_metadata_is_rejected(
    tmp_path,
):
    from qvc.hubble.hubble_injection_recovery import (
        HubbleInjectionConfig,
        HubbleInjectionTruth,
        generate_hubble_injection,
        load_injection_hdf5,
        save_injection_hdf5,
    )

    dataset = generate_hubble_injection(
        HubbleInjectionConfig(
            seed=11,
            n_parent=128,
            z_range=(0.44, 3.16),
            selection_model="logistic-magnitude",
            predictor_noise="realistic",
        ),
        HubbleInjectionTruth(
            H0=70.0,
            Om0=0.3,
            M0_agn=-23.4,
            alpha_agn=-1.8,
            beta_agn=-0.9,
            log_f=np.log(0.35),
            reference_pivots=(np.log10(0.2), np.log10(300.0)),
        ),
    )
    path = tmp_path / "injection.h5"

    save_injection_hdf5(dataset, path)
    loaded = load_injection_hdf5(path)

    pd.testing.assert_frame_equal(loaded.parent, dataset.parent, check_exact=True)
    pd.testing.assert_frame_equal(loaded.selected, dataset.selected, check_exact=True)
    np.testing.assert_array_equal(loaded.selected_mask, dataset.selected_mask)
    np.testing.assert_array_equal(
        loaded.selection_probability,
        dataset.selection_probability,
    )
    assert loaded.pivot_context == dataset.pivot_context
    assert loaded.truth_at_fit_pivot == dataset.truth_at_fit_pivot
    assert loaded.dataset_id == dataset.dataset_id
    assert loaded.config == dataset.config
    assert loaded.truth == dataset.truth

    with h5py.File(path, "r+") as handle:
        del handle["pivot_values"]
    with pytest.raises(ValueError, match="missing required"):
        load_injection_hdf5(path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_parent": 0}, "n_parent"),
        ({"z_range": (2.0, 1.0)}, "z_range"),
        ({"selection_model": "guess"}, "selection_model"),
        ({"predictor_noise": "guess"}, "predictor_noise"),
    ],
)
def test_injection_config_rejects_invalid_values(overrides, message):
    from qvc.hubble.hubble_injection_recovery import HubbleInjectionConfig

    values = {
        "seed": 1,
        "n_parent": 32,
        "z_range": (0.44, 3.16),
        "selection_model": "none",
        "predictor_noise": "noiseless",
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        HubbleInjectionConfig(**values)


def test_injection_truth_rejects_nonfinite_values():
    from qvc.hubble.hubble_injection_recovery import HubbleInjectionTruth

    with pytest.raises(ValueError, match="M0_agn"):
        HubbleInjectionTruth(
            H0=70.0,
            Om0=0.3,
            M0_agn=np.nan,
            alpha_agn=-1.8,
            beta_agn=-0.9,
            log_f=np.log(0.35),
            reference_pivots=(np.log10(0.2), np.log10(300.0)),
        )


def test_injection_hdf5_rejects_old_schema_and_mask_length(tmp_path):
    from qvc.hubble.hubble_injection_recovery import (
        HubbleInjectionConfig,
        HubbleInjectionTruth,
        generate_hubble_injection,
        load_injection_hdf5,
        save_injection_hdf5,
    )

    dataset = generate_hubble_injection(
        HubbleInjectionConfig(
            seed=17,
            n_parent=64,
            z_range=(0.44, 3.16),
            selection_model="none",
            predictor_noise="noiseless",
        ),
        HubbleInjectionTruth(
            H0=70.0,
            Om0=0.3,
            M0_agn=-23.4,
            alpha_agn=-1.8,
            beta_agn=-0.9,
            log_f=np.log(0.35),
            reference_pivots=(np.log10(0.2), np.log10(300.0)),
        ),
    )
    old_schema = tmp_path / "old_schema.h5"
    save_injection_hdf5(dataset, old_schema)
    with h5py.File(old_schema, "r+") as handle:
        handle.attrs["schema_version"] = 0
    with pytest.raises(ValueError, match="Unsupported injection schema"):
        load_injection_hdf5(old_schema)

    bad_mask = tmp_path / "bad_mask.h5"
    save_injection_hdf5(dataset, bad_mask)
    with h5py.File(bad_mask, "r+") as handle:
        del handle["selected_mask"]
        handle.create_dataset("selected_mask", data=np.ones(63, dtype=bool))
    with pytest.raises(ValueError, match="parent-length"):
        load_injection_hdf5(bad_mask)


def test_injection_validation_rejects_invalid_probability_empty_selection_and_covariance(
    tmp_path,
):
    from qvc.hubble.hubble_injection_recovery import save_injection_hdf5

    dataset = _small_injection()

    probability = dataset.selection_probability.copy()
    probability[0] = 1.1
    parent = dataset.parent.copy()
    parent["selection_probability"] = probability
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        save_injection_hdf5(
            replace(
                dataset,
                parent=parent,
                selection_probability=probability,
            ),
            tmp_path / "bad_probability.h5",
        )

    empty_mask = np.zeros(len(dataset.parent), dtype=bool)
    parent = dataset.parent.copy()
    parent["selected"] = empty_mask
    with pytest.raises(ValueError, match="at least one selected"):
        save_injection_hdf5(
            replace(
                dataset,
                parent=parent,
                selected=parent.loc[empty_mask].copy(),
                selected_mask=empty_mask,
            ),
            tmp_path / "empty.h5",
        )

    parent = dataset.parent.copy()
    parent.loc[0, "log_sigma_uv_log_tau_uv_rf_cov_psd"] = 1.0
    selected = parent.loc[dataset.selected_mask].copy()
    with pytest.raises(ValueError, match="positive semidefinite"):
        save_injection_hdf5(
            replace(dataset, parent=parent, selected=selected),
            tmp_path / "bad_covariance.h5",
        )


def test_injection_hdf5_rejects_incompatible_fit_truth_metadata(tmp_path):
    from qvc.hubble.hubble_injection_recovery import (
        load_injection_hdf5,
        save_injection_hdf5,
    )

    dataset = _small_injection(seed=43)
    path = tmp_path / "bad_truth_at_fit.h5"
    save_injection_hdf5(dataset, path)
    with h5py.File(path, "r+") as handle:
        values = handle["truth_at_fit_values"][:]
        values[2] += 0.5
        handle["truth_at_fit_values"][:] = values

    with pytest.raises(ValueError, match="truth_at_fit_pivot"):
        load_injection_hdf5(path)


@pytest.mark.parametrize(
    "corruption",
    ["reordered_names", "nonfinite_value", "duplicate_reference_id"],
)
def test_injection_hdf5_rejects_incompatible_pivot_metadata(
    tmp_path,
    corruption,
):
    from qvc.hubble.hubble_injection_recovery import (
        load_injection_hdf5,
        save_injection_hdf5,
    )

    dataset = _small_injection(seed=45)
    path = tmp_path / f"{corruption}.h5"
    save_injection_hdf5(dataset, path)
    with h5py.File(path, "r+") as handle:
        if corruption == "reordered_names":
            names = handle["pivot_observable_names"][:]
            handle["pivot_observable_names"][:] = names[::-1]
        elif corruption == "nonfinite_value":
            values = handle["pivot_values"][:]
            values[0] = np.nan
            handle["pivot_values"][:] = values
        else:
            object_ids = handle["pivot_reference_object_ids"][:]
            object_ids[1] = object_ids[0]
            handle["pivot_reference_object_ids"][:] = object_ids

    with pytest.raises(ValueError):
        load_injection_hdf5(path)


def test_fixed_recovery_rejects_optimizer_failure_and_unknown_candidates(
    tmp_path,
    monkeypatch,
):
    from qvc.hubble import hubble_injection_recovery as recovery

    dataset = _small_injection(seed=47, predictor_noise="noiseless")
    with pytest.raises(ValueError, match="Completeness candidate"):
        recovery.build_completeness_candidate("unknown", dataset, tmp_path)
    with pytest.raises(ValueError, match="supports only candidates"):
        recovery.run_joint_sampler_recovery(dataset, "oracle", tmp_path)

    monkeypatch.setattr(
        recovery,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            fun=np.inf,
            x=np.full(4, np.nan),
            message="injected optimizer failure",
        ),
    )
    with pytest.raises(RuntimeError, match="injected optimizer failure"):
        recovery.fit_fixed_cosmology(dataset, None)


def test_sigma_tau_selection_is_explicit_and_has_no_magnitude_only_oracle():
    from qvc.hubble.hubble_injection_recovery import (
        HubbleInjectionConfig,
        HubbleInjectionTruth,
        build_completeness_candidate,
        generate_hubble_injection,
    )

    truth = HubbleInjectionTruth(
        H0=70.0,
        Om0=0.3,
        M0_agn=-23.4,
        alpha_agn=-1.8,
        beta_agn=-0.9,
        log_f=np.log(0.35),
        reference_pivots=(np.log10(0.2), np.log10(300.0)),
    )
    dataset = generate_hubble_injection(
        HubbleInjectionConfig(
            seed=19,
            n_parent=256,
            z_range=(0.44, 3.16),
            selection_model="logistic-sigma-tau",
            predictor_noise="noiseless",
        ),
        truth,
    )
    parent = dataset.parent
    magnitude_limit = 22.35 - 0.12 * (parent["z"].to_numpy() - 1.5)
    magnitude_logit = (
        magnitude_limit - parent["apparent_mag_2500"].to_numpy()
    ) / 0.30
    observed_logit = np.log(
        dataset.selection_probability / (1.0 - dataset.selection_probability)
    )
    expected_extra = (
        1.20
        * (parent["log_sigma_uv"].to_numpy() - truth.reference_pivots[0])
        / 0.18
        - 0.80
        * (parent["log_tau_uv_rf"].to_numpy() - truth.reference_pivots[1])
        / 0.28
    )

    np.testing.assert_allclose(observed_logit - magnitude_logit, expected_extra)
    assert np.std(expected_extra) > 0.5
    with pytest.raises(ValueError, match="oracle"):
        build_completeness_candidate("oracle", dataset, Path("/tmp"))


@pytest.mark.parametrize("seed", [20260728, 20260729, 20260730])
def test_fixed_cosmology_no_selection_recovers_injected_standardization(
    tmp_path,
    seed,
):
    from qvc.hubble.hubble_injection_recovery import (
        HubbleInjectionConfig,
        HubbleInjectionTruth,
        build_completeness_candidate,
        fit_fixed_cosmology,
        generate_hubble_injection,
    )

    dataset = generate_hubble_injection(
        HubbleInjectionConfig(
            seed=seed,
            n_parent=1000,
            z_range=(0.44, 3.16),
            selection_model="none",
            predictor_noise="noiseless",
        ),
        HubbleInjectionTruth(
            H0=70.0,
            Om0=0.3,
            M0_agn=-23.4,
            alpha_agn=-1.8,
            beta_agn=-0.9,
            log_f=np.log(0.35),
            reference_pivots=(np.log10(0.2), np.log10(300.0)),
        ),
    )

    completeness = build_completeness_candidate("none", dataset, tmp_path)
    result = fit_fixed_cosmology(dataset, completeness)

    assert result.candidate == "none"
    assert result.backend == "fast"
    assert abs(result.estimates["M0_agn"] - result.truth["M0_agn"]) < 0.08
    assert abs(result.estimates["alpha_agn"] - result.truth["alpha_agn"]) < 0.20
    assert abs(result.estimates["beta_agn"] - result.truth["beta_agn"]) < 0.20
    assert abs(result.estimates["log_f"] - result.truth["log_f"]) < 0.12
    assert abs(result.metrics["residual_z_slope"]) < 0.08
    binned_means = [
        value
        for name, value in result.metrics.items()
        if name.startswith("debiased_residual_mean_z")
    ]
    assert binned_means
    assert all(abs(value) < 0.12 for value in binned_means)
    assert np.all(np.isfinite(result.residuals["raw_residual"]))
    np.testing.assert_allclose(
        result.residuals["raw_residual"],
        result.residuals["debiased_residual"],
    )


def test_realistic_correlated_predictor_errors_recover_with_looser_tolerance(
    tmp_path,
):
    from qvc.hubble.hubble_injection_recovery import (
        HubbleInjectionConfig,
        HubbleInjectionTruth,
        fit_fixed_cosmology,
        generate_hubble_injection,
    )

    dataset = generate_hubble_injection(
        HubbleInjectionConfig(
            seed=20260731,
            n_parent=1200,
            z_range=(0.44, 3.16),
            selection_model="none",
            predictor_noise="realistic",
        ),
        HubbleInjectionTruth(
            H0=70.0,
            Om0=0.3,
            M0_agn=-23.4,
            alpha_agn=-1.8,
            beta_agn=-0.9,
            log_f=np.log(0.35),
            reference_pivots=(np.log10(0.2), np.log10(300.0)),
        ),
    )
    result = fit_fixed_cosmology(dataset, None)

    covariance = dataset.parent[
        "log_sigma_uv_log_tau_uv_rf_cov_psd"
    ].to_numpy(dtype=float)
    sigma_variance = np.square(
        dataset.parent["log_sigma_uv_std_psd"].to_numpy(dtype=float)
    )
    tau_variance = np.square(
        dataset.parent["log_tau_uv_rf_std_psd"].to_numpy(dtype=float)
    )
    assert np.all(sigma_variance * tau_variance - covariance**2 >= 0.0)
    assert abs(result.estimates["M0_agn"] - result.truth["M0_agn"]) < 0.12
    assert abs(result.estimates["alpha_agn"] - result.truth["alpha_agn"]) < 0.30
    assert abs(result.estimates["beta_agn"] - result.truth["beta_agn"]) < 0.30
    assert abs(result.estimates["log_f"] - result.truth["log_f"]) < 0.20


def test_oracle_logistic_completeness_improves_recovery_on_same_selected_sample(
    tmp_path,
):
    from qvc.hubble.hubble_injection_recovery import (
        HubbleInjectionConfig,
        HubbleInjectionTruth,
        build_completeness_candidate,
        fit_fixed_cosmology,
        generate_hubble_injection,
    )

    dataset = generate_hubble_injection(
        HubbleInjectionConfig(
            seed=20260728,
            n_parent=1800,
            z_range=(0.44, 3.16),
            selection_model="logistic-magnitude",
            predictor_noise="noiseless",
        ),
        HubbleInjectionTruth(
            H0=70.0,
            Om0=0.3,
            M0_agn=-23.4,
            alpha_agn=-1.8,
            beta_agn=-0.9,
            log_f=np.log(0.35),
            reference_pivots=(np.log10(0.2), np.log10(300.0)),
        ),
    )

    no_correction = fit_fixed_cosmology(
        dataset,
        build_completeness_candidate("none", dataset, tmp_path),
    )
    oracle = fit_fixed_cosmology(
        dataset,
        build_completeness_candidate("oracle", dataset, tmp_path),
    )

    assert no_correction.residuals["object_id"].tolist() == oracle.residuals[
        "object_id"
    ].tolist()
    assert oracle.metrics["parameter_rmse"] < 0.75 * no_correction.metrics[
        "parameter_rmse"
    ]
    assert abs(oracle.metrics["residual_z_slope"]) < 0.75 * abs(
        no_correction.metrics["residual_z_slope"]
    )
    assert abs(oracle.estimates["M0_agn"] - oracle.truth["M0_agn"]) < 0.12
    assert abs(oracle.estimates["alpha_agn"] - oracle.truth["alpha_agn"]) < 0.25
    assert abs(oracle.estimates["beta_agn"] - oracle.truth["beta_agn"]) < 0.25
    assert abs(oracle.estimates["log_f"] - oracle.truth["log_f"]) < 0.15


def test_current_2d_candidate_is_a_finite_benchmark_on_frozen_selection(tmp_path):
    from qvc.hubble.hubble_injection_recovery import (
        HubbleInjectionConfig,
        HubbleInjectionTruth,
        build_completeness_candidate,
        fit_fixed_cosmology,
        generate_hubble_injection,
    )

    dataset = generate_hubble_injection(
        HubbleInjectionConfig(
            seed=91,
            n_parent=1200,
            z_range=(0.44, 3.16),
            selection_model="logistic-magnitude",
            predictor_noise="noiseless",
        ),
        HubbleInjectionTruth(
            H0=70.0,
            Om0=0.3,
            M0_agn=-23.4,
            alpha_agn=-1.8,
            beta_agn=-0.9,
            log_f=np.log(0.35),
            reference_pivots=(np.log10(0.2), np.log10(300.0)),
        ),
    )

    completeness_params = build_completeness_candidate(
        "current-2d",
        dataset,
        tmp_path,
    )
    model = completeness_params[0]
    pdet = model(
        dataset.selected["apparent_mag_2500"].to_numpy(dtype=float),
        dataset.selected["z"].to_numpy(dtype=float),
    )
    result = fit_fixed_cosmology(dataset, completeness_params)

    assert result.candidate == "current-2d"
    assert np.all(np.isfinite(pdet))
    assert np.all((pdet >= 0.0) & (pdet <= 1.0))
    assert result.residuals["object_id"].tolist() == dataset.selected[
        "object_id"
    ].tolist()
    assert all(np.isfinite(value) for value in result.estimates.values())
    assert all(
        np.isfinite(value)
        for value in result.metrics.values()
        if not np.isnan(value)
    )


def test_relative_2d_candidate_uses_frozen_parent_and_support_map(tmp_path):
    from qvc.hubble.hubble_injection_recovery import (
        HubbleInjectionConfig,
        HubbleInjectionTruth,
        build_completeness_candidate,
        generate_hubble_injection,
    )

    dataset = generate_hubble_injection(
        HubbleInjectionConfig(
            seed=91,
            n_parent=1200,
            z_range=(0.44, 3.16),
            selection_model="logistic-magnitude",
            predictor_noise="noiseless",
        ),
        HubbleInjectionTruth(
            H0=70.0,
            Om0=0.3,
            M0_agn=-23.4,
            alpha_agn=-1.8,
            beta_agn=-0.9,
            log_f=np.log(0.35),
            reference_pivots=(np.log10(0.2), np.log10(300.0)),
        ),
    )

    completeness_params = build_completeness_candidate(
        "relative-2d",
        dataset,
        tmp_path,
    )
    model = completeness_params[0]
    weights = np.asarray(model._interp.values)

    assert model.mode == "2d_relative_support"
    assert getattr(model, "_recovery_candidate_name") == "relative-2d"
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)
    assert np.any(weights > 1.0)
    assert model.relative_selection_metadata["mock_count_scale_used_in_weights"] is False


def test_relative_2d_improves_logistic_recovery_across_fixed_seeds(tmp_path):
    from qvc.hubble.hubble_injection_recovery import (
        HubbleInjectionConfig,
        HubbleInjectionTruth,
        build_completeness_candidate,
        fit_fixed_cosmology,
        generate_hubble_injection,
    )

    truth = HubbleInjectionTruth(
        H0=70.0,
        Om0=0.3,
        M0_agn=-23.4,
        alpha_agn=-1.8,
        beta_agn=-0.9,
        log_f=np.log(0.35),
        reference_pivots=(np.log10(0.2), np.log10(300.0)),
    )
    none_rmse = []
    relative_rmse = []
    none_slopes = []
    relative_slopes = []
    for seed in (20260728, 20260729, 20260730):
        dataset = generate_hubble_injection(
            HubbleInjectionConfig(
                seed=seed,
                n_parent=1800,
                z_range=(0.44, 3.16),
                selection_model="logistic-magnitude",
                predictor_noise="noiseless",
            ),
            truth,
        )
        none = fit_fixed_cosmology(dataset, None)
        relative = fit_fixed_cosmology(
            dataset,
            build_completeness_candidate(
                "relative-2d",
                dataset,
                tmp_path / str(seed),
            ),
        )
        none_rmse.append(none.metrics["parameter_rmse"])
        relative_rmse.append(relative.metrics["parameter_rmse"])
        none_slopes.append(abs(none.metrics["residual_z_slope"]))
        relative_slopes.append(abs(relative.metrics["residual_z_slope"]))

    assert np.mean(relative_rmse) <= 0.75 * np.mean(none_rmse)
    assert np.mean(relative_slopes) <= 0.75 * np.mean(none_slopes)


def test_injection_recovery_cli_writes_complete_artifact_set(tmp_path):
    output_dir = tmp_path / "artifacts"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "validate_hubble_injection_recovery.py"),
        "--selection",
        "none",
        "--candidate",
        "none",
        "--backend",
        "fast",
        "--seed",
        "123",
        "--n-parent",
        "256",
        "--output-dir",
        str(output_dir),
    ]

    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mplconfig")},
    )

    assert completed.returncode == 0, completed.stderr
    for filename in (
        "injection.h5",
        "manifest.json",
        "recovery.csv",
        "residuals.csv",
        "metrics.json",
        "recovery_diagnostics.pdf",
    ):
        assert (output_dir / filename).is_file(), filename


@pytest.mark.slow_hubble_recovery
@pytest.mark.skipif(
    os.environ.get("RUN_HUBBLE_INJECTION_SLOW") != "1",
    reason="Set RUN_HUBBLE_INJECTION_SLOW=1 to run the real Dynesty recovery.",
)
def test_production_joint_sampler_returns_finite_posterior_and_pivot(tmp_path):
    from qvc.hubble.hubble_injection_recovery import (
        HubbleInjectionConfig,
        HubbleInjectionTruth,
        generate_hubble_injection,
        run_joint_sampler_recovery,
    )

    dataset = generate_hubble_injection(
        HubbleInjectionConfig(
            seed=314,
            n_parent=96,
            z_range=(0.44, 3.16),
            selection_model="none",
            predictor_noise="noiseless",
        ),
        HubbleInjectionTruth(
            H0=70.0,
            Om0=0.3,
            M0_agn=-23.4,
            alpha_agn=-1.8,
            beta_agn=-0.9,
            log_f=np.log(0.35),
            reference_pivots=(np.log10(0.2), np.log10(300.0)),
        ),
    )

    result = run_joint_sampler_recovery(
        dataset,
        "none",
        tmp_path / "production",
    )

    assert result.backend == "production"
    assert result.samples is not None
    assert result.samples.ndim == 2
    assert result.samples.shape[1] == len(result.model_labels)
    assert np.all(np.isfinite(result.samples))
    assert {
        "M0_sn",
        "M0_agn",
        "alpha_agn",
        "beta_agn",
        "log_f",
        "H0",
        "Om0",
    } == set(result.model_labels)
    assert abs(result.estimates["M0_agn"] - result.truth["M0_agn"]) < 0.8
    assert abs(result.estimates["alpha_agn"] - result.truth["alpha_agn"]) < 2.0
    assert abs(result.estimates["beta_agn"] - result.truth["beta_agn"]) < 2.0
    assert abs(result.estimates["log_f"] - result.truth["log_f"]) < 0.6
    assert abs(result.estimates["H0"] - result.truth["H0"]) < 10.0
    assert abs(result.estimates["Om0"] - result.truth["Om0"]) < 0.5
