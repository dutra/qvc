import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_relative_selection_2d_preserves_weights_above_one():
    from qvc.hubble.hubble_completeness_refactored import RelativeSelection2D

    mag_centers = np.array([19.0, 20.0, 21.0])
    z_centers = np.array([0.5, 1.5])
    weights = np.array(
        [
            [0.5, 0.75],
            [1.0, 1.25],
            [2.0, 3.0],
        ]
    )

    model = RelativeSelection2D(mag_centers, z_centers, weights)

    np.testing.assert_allclose(
        model(np.array([19.0, 20.0, 21.0]), np.array([0.5, 1.5, 1.5])),
        np.array([0.5, 1.25, 3.0]),
    )
    assert np.max(model._interp.values) == 3.0
    assert model.mode == "2d_relative_support"
    assert model(18.0, 0.5) == model(19.0, 0.5)
    assert model(22.0, 0.5) == 0.0


def _write_mock(
    path,
    *,
    mock_count_scale,
    thinning_probability=0.125,
    area_deg2=1_000.0,
):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("apparent_mag_2500", data=[19.0, 22.0, 22.0, 22.0])
        handle.create_dataset("z", data=[0.5, 0.5, 0.5, 0.5])
        handle.attrs["mock_count_scale"] = mock_count_scale
        handle.attrs["thinning_probability"] = thinning_probability
        handle.attrs["area_deg2"] = area_deg2


def test_relative_selection_builder_uses_conditional_density_ratio(tmp_path):
    from qvc.hubble.hubble_completeness_refactored import (
        get_relative_selection_function_2d,
    )

    observed = pd.DataFrame(
        {
            "apparent_mag_2500": [19.0, 19.0, 22.0],
            "z": [0.5, 0.5, 0.5],
        }
    )
    mock_a = tmp_path / "mock_a.h5"
    mock_b = tmp_path / "mock_b.h5"
    _write_mock(
        mock_a,
        mock_count_scale=0.1,
        thinning_probability=0.01,
        area_deg2=100.0,
    )
    _write_mock(
        mock_b,
        mock_count_scale=100.0,
        thinning_probability=0.9,
        area_deg2=10_000.0,
    )

    params_a = get_relative_selection_function_2d(
        observed,
        sim_file=str(mock_a),
        n_mag_bins=2,
        n_z_bins=2,
        smooth_counts=False,
    )
    params_b = get_relative_selection_function_2d(
        observed,
        sim_file=str(mock_b),
        n_mag_bins=2,
        n_z_bins=2,
        smooth_counts=False,
    )
    model_a, mag_centers, z_centers, *_ = params_a
    model_b = params_b[0]
    mag_grid, z_grid = np.meshgrid(mag_centers, z_centers, indexing="ij")
    weights = model_a(mag_grid, z_grid)

    p_mock = np.array([1.5 / 5.0, 3.5 / 5.0])
    expected_first_slice = np.array([(2.5 / 4.0) / p_mock[0], (1.5 / 4.0) / p_mock[1]])
    np.testing.assert_allclose(weights[:, 0], expected_first_slice)
    np.testing.assert_allclose(np.sum(p_mock * weights[:, 0]), 1.0)
    np.testing.assert_allclose(weights[:, 1], 1.0)
    np.testing.assert_allclose(model_b(mag_grid, z_grid), weights)
    assert model_a.relative_selection_metadata["mock_count_scale"] == 0.1
    assert model_b.relative_selection_metadata["mock_count_scale"] == 100.0

    from qvc.hubble.hubble_likelihood import completeness_loglike

    likelihood_args = {
        "m_obs": np.array([20.0, 22.0]),
        "m_obs_err": np.array([0.05, 0.05]),
        "m_model": np.array([20.0, 22.0]),
        "mu_err": np.array([0.3, 0.4]),
        "z": np.array([0.5, 0.5]),
        "m_grid": mag_centers,
    }
    loglike_a, blob_a = completeness_loglike(
        **likelihood_args,
        completeness_model=model_a,
    )
    loglike_b, blob_b = completeness_loglike(
        **likelihood_args,
        completeness_model=model_b,
    )
    np.testing.assert_allclose(loglike_a, loglike_b)
    np.testing.assert_allclose(blob_a, blob_b)


def test_existing_absolute_2d_ratio_and_clipping_are_unchanged(tmp_path):
    from qvc.hubble.hubble_completeness_refactored import (
        get_completeness_function_2d,
    )

    observed = pd.DataFrame(
        {
            "apparent_mag_2500": [19.0, 19.0, 22.0],
            "z": [0.5, 0.5, 0.5],
        }
    )
    mock_path = tmp_path / "absolute_mock.h5"
    _write_mock(mock_path, mock_count_scale=2.0)
    model, mag_centers, z_centers, *_ = get_completeness_function_2d(
        observed,
        sim_file=str(mock_path),
        n_mag_bins=2,
        n_z_bins=2,
        smooth_counts=False,
    )
    mag_grid, z_grid = np.meshgrid(mag_centers, z_centers, indexing="ij")

    np.testing.assert_allclose(
        model(mag_grid, z_grid),
        np.array([[1.0, 0.0], [1.0 / 6.0, 0.0]]),
    )


def test_cpu_pipeline_routes_relative_completeness_mode(monkeypatch):
    from qvc.hubble import hubble_fit

    sentinel = object()
    calls = []

    def fake_builder(df, **kwargs):
        calls.append((df, kwargs))
        return sentinel

    monkeypatch.setattr(
        hubble_fit,
        "get_relative_selection_function_2d",
        fake_builder,
        raising=False,
    )
    frame = pd.DataFrame({"apparent_mag_2500": [20.0], "z": [1.0]})
    hubble_fit.validate_completeness_mode("2d_relative_support")
    result = hubble_fit._build_completeness_params(
        frame,
        frame,
        completeness=True,
        completeness_mode="2d_relative_support",
        completeness_sim_file="mock.h5",
        plot_path="plots/test",
    )

    assert result is sentinel
    assert calls == [
        (
            frame,
            {
                "sim_file": "mock.h5",
                "plot": False,
                "plot_path": "plots/test",
            },
        )
    ]
    tag = hubble_fit.make_run_tag(
        "FlatLambdaCDM",
        False,
        "fastest",
        None,
        (0.44, 3.16),
        completeness=True,
        completeness_mode="2d_relative_support",
    )
    assert tag.endswith("_2d_relative_support")


def test_jax_preparation_preserves_relative_mode_and_weights_above_one():
    from qvc.hubble.hubble_completeness_refactored import RelativeSelection2D
    from qvc.hubble.hubble_fit_jax import _prepare_completeness_for_jax

    model = RelativeSelection2D(
        np.array([19.0, 20.0, 21.0]),
        np.array([0.5, 1.5]),
        np.array([[0.5, 0.75], [1.0, 1.25], [2.0, 3.0]]),
    )
    prepared = _prepare_completeness_for_jax(
        (model, model.mag_centers, model.z_centers, 1.0, 1.0, 0.0)
    )

    assert prepared["mode"] == "2d_relative_support"
    np.testing.assert_allclose(np.asarray(prepared["cube"]), model._interp.values)
    assert float(np.max(np.asarray(prepared["cube"]))) == 3.0


def test_relative_checkpoint_metadata_is_strict(tmp_path):
    from qvc.hubble.hubble_completeness_refactored import (
        RELATIVE_SELECTION_PRIOR_CONCENTRATION,
        RELATIVE_SELECTION_RULE,
        RelativeSelection2D,
    )
    from qvc.hubble import hubble_fit
    from qvc.hubble.hubble_fit import (
        _relative_selection_checkpoint_payload,
        _validate_relative_selection_checkpoint_metadata,
    )

    model = RelativeSelection2D(
        np.array([19.0, 20.0, 21.0]),
        np.array([0.5, 1.5]),
        np.ones((3, 2)),
    )
    model.relative_selection_metadata = {
        "rule": RELATIVE_SELECTION_RULE,
        "n_mag_bins": 3,
        "n_z_bins": 2,
        "sigma_mag": 0.2,
        "sigma_z": 0.2,
        "prior_concentration": RELATIVE_SELECTION_PRIOR_CONCENTRATION,
        "mock_count_scale": 10.0,
        "thinning_probability": 0.1,
        "area_deg2": 100.0,
    }
    params = (model, model.mag_centers, model.z_centers, 1.0, 1.0, 0.2)
    payload = _relative_selection_checkpoint_payload(params)

    checkpoint_path = tmp_path / "relative_checkpoint.h5"
    hubble_fit.save_chains(checkpoint_path, **payload)
    round_tripped = hubble_fit.load_chains(checkpoint_path)
    _validate_relative_selection_checkpoint_metadata(
        round_tripped,
        checkpoint_file=str(checkpoint_path),
        completeness_mode="2d_relative_support",
        completeness_params=params,
    )
    for key in tuple(payload):
        incomplete = dict(payload)
        incomplete.pop(key)
        with pytest.raises(RuntimeError, match="relative"):
            _validate_relative_selection_checkpoint_metadata(
                incomplete,
                checkpoint_file="relative.h5",
                completeness_mode="2d_relative_support",
                completeness_params=params,
            )

    incompatible = dict(payload, relative_selection_rule="wrong")
    with pytest.raises(RuntimeError, match="incompatible"):
        _validate_relative_selection_checkpoint_metadata(
            incompatible,
            checkpoint_file="relative.h5",
            completeness_mode="2d_relative_support",
            completeness_params=params,
        )


def test_relative_builder_writes_support_diagnostics(tmp_path):
    from qvc.hubble.hubble_completeness_refactored import (
        get_relative_selection_function_2d,
    )

    observed = pd.DataFrame(
        {
            "object_id": ["a", "b", "c"],
            "apparent_mag_2500": [19.0, 19.0, 22.0],
            "z": [0.5, 0.5, 0.5],
        }
    )
    mock_path = tmp_path / "mock.h5"
    _write_mock(mock_path, mock_count_scale=7.0)
    get_relative_selection_function_2d(
        observed,
        sim_file=str(mock_path),
        n_mag_bins=2,
        n_z_bins=2,
        smooth_counts=False,
        plot=True,
        plot_path=str(tmp_path),
    )

    diagnostic_dir = tmp_path / "completeness"
    expected = {
        "relative_selection_raw_map.pdf",
        "relative_selection_regularized_map.pdf",
        "relative_selection_prior_fraction_map.pdf",
        "relative_selection_slice_diagnostics.csv",
        "relative_selection_prior_dominated_objects.csv",
        "relative_selection_metadata.json",
        "relative_selection_diagnostics.npz",
    }
    assert expected <= {path.name for path in diagnostic_dir.iterdir()}
    with open(diagnostic_dir / "relative_selection_metadata.json") as handle:
        metadata = json.load(handle)
    assert metadata["rule"] == "conditional_density_ratio_dirichlet_v1"
    assert metadata["mock_count_scale"] == 7.0
    assert metadata["mock_count_scale_used_in_weights"] is False
    with np.load(
        diagnostic_dir / "relative_selection_diagnostics.npz"
    ) as diagnostics:
        assert "obs_prior_fraction" in diagnostics
        assert "mock_prior_fraction" in diagnostics
        assert "obs_support_fraction" in diagnostics
        assert "mock_support_fraction" in diagnostics


def test_relative_selection_cpu_jax_integral_and_moments_match():
    from qvc.hubble.hubble_completeness_refactored import RelativeSelection2D
    from qvc.hubble.hubble_fit_jax import (
        _completeness_loglike_jax,
        _prepare_completeness_for_jax,
    )
    from qvc.hubble.hubble_likelihood import completeness_loglike

    mag_centers = np.linspace(18.5, 24.0, 121)
    z_centers = np.linspace(0.0, 4.0, 41)
    mm, zz = np.meshgrid(mag_centers, z_centers, indexing="ij")
    weights = 0.25 + np.exp(0.38 * (mm - 20.0) - 0.12 * zz)
    model = RelativeSelection2D(mag_centers, z_centers, weights)
    params = (
        model,
        mag_centers,
        z_centers,
        float(np.diff(mag_centers).mean()),
        float(np.diff(z_centers).mean()),
        99.0,
    )
    m_model = np.array([17.8, 21.2, 23.6])
    mu_err = np.array([0.25, 0.35, 0.45])
    z = np.array([-0.5, 1.4, 4.5])

    cpu_log_z, cpu_blob = completeness_loglike(
        m_obs=m_model,
        m_obs_err=np.full(3, 0.05),
        m_model=m_model,
        mu_err=mu_err,
        z=z,
        completeness_model=model,
        m_grid=mag_centers,
        sigma_completeness=0.0,
    )
    jax_log_z, jax_blob = _completeness_loglike_jax(
        m_model,
        mu_err,
        z,
        _prepare_completeness_for_jax(params),
        None,
        None,
        return_blob=True,
    )

    np.testing.assert_allclose(float(jax_log_z), cpu_log_z, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(jax_blob), cpu_blob, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "bad_weights",
    [
        [[1.0, 1.0], [1.0, 0.0]],
        [[1.0, 1.0], [1.0, -0.1]],
        [[1.0, 1.0], [1.0, np.nan]],
        [[1.0, 1.0], [1.0, np.inf]],
    ],
)
def test_relative_selection_rejects_nonpositive_or_nonfinite_weights(
    bad_weights,
):
    from qvc.hubble.hubble_completeness_refactored import RelativeSelection2D

    with pytest.raises(ValueError, match="finite, strictly positive"):
        RelativeSelection2D([19.0, 20.0], [0.5, 1.5], bad_weights)


def test_sparse_relative_map_is_regularized_without_max_normalization(tmp_path):
    from qvc.hubble.hubble_completeness_refactored import (
        get_relative_selection_function_2d,
    )

    observed = pd.DataFrame(
        {
            "apparent_mag_2500": np.repeat(23.8, 40),
            "z": np.repeat(0.5, 40),
        }
    )
    mock_path = tmp_path / "sparse_mock.h5"
    with h5py.File(mock_path, "w") as handle:
        handle.create_dataset("apparent_mag_2500", data=[18.6])
        handle.create_dataset("z", data=[0.5])
        handle.attrs["mock_count_scale"] = 1e-6

    model, *_ = get_relative_selection_function_2d(
        observed,
        sim_file=str(mock_path),
        n_mag_bins=30,
        n_z_bins=4,
        smooth_counts=False,
    )
    weights = np.asarray(model._interp.values)
    p_mock = model.relative_selection_diagnostics["P_mock"]

    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)
    assert np.max(weights[:, 0]) > 1.0
    assert np.max(weights[:, 0]) < 1e6
    np.testing.assert_allclose(np.sum(p_mock[:, 0] * weights[:, 0]), 1.0)
    np.testing.assert_allclose(weights[:, 1:], 1.0)
