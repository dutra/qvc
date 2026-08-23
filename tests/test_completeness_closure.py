import os
import sys
from pathlib import Path

import json

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble import completeness_closure
from qvc.hubble.completeness_closure import (
    simulate_completeness_closure,
    simulate_hubble_posterior_closure,
    write_completeness_closure_diagnostics,
)


class HardMagnitudeLimit:
    mode = "2d"

    def __init__(self, limit):
        self.limit = float(limit)

    def __call__(self, magnitude, redshift):
        magnitude, redshift = np.broadcast_arrays(
            np.asarray(magnitude, dtype=float),
            np.asarray(redshift, dtype=float),
        )
        return np.where(
            magnitude < self.limit,
            1.0,
            np.where(magnitude == self.limit, 0.5, 0.0),
        )


class AlwaysComplete:
    mode = "2d"

    def __call__(self, magnitude, redshift):
        magnitude, redshift = np.broadcast_arrays(
            np.asarray(magnitude, dtype=float),
            np.asarray(redshift, dtype=float),
        )
        return np.ones(magnitude.shape, dtype=float)


def _posterior_predictive_inputs(n_per_bin=300, n_draws=24):
    z_centers = np.array([0.6, 1.0, 1.4])
    redshift = np.repeat(z_centers, n_per_bin)
    base_model = np.repeat(np.array([21.7, 22.0, 22.3]), n_per_bin)
    draw_offsets = np.linspace(-0.04, 0.04, n_draws)[:, None]
    model_draws = base_model[None, :] + draw_offsets
    sigma_draws = np.full_like(model_draws, 0.4)
    return redshift, model_draws, sigma_draws


def test_posterior_predictive_closure_recovers_zero_in_every_redshift_bin():
    redshift, model_draws, sigma_draws = _posterior_predictive_inputs()

    result = simulate_completeness_closure(
        model_magnitude_draws=model_draws,
        sigma_draws=sigma_draws,
        redshift=redshift,
        completeness_model=HardMagnitudeLimit(22.0),
        magnitude_grid=np.linspace(18.0, 24.5, 1001),
        redshift_bins=np.array([0.4, 0.8, 1.2, 1.6]),
        seed=7721,
        max_abs_mean_zscore=4.0,
        min_detected_per_bin=100,
    )

    assert result.all_bins_pass
    assert result.summary["bin_pass"].all()
    assert np.all(result.summary["mean_raw_residual"] < -0.05)
    assert np.all(np.abs(result.summary["mean_corrected_residual"]) < 0.03)
    assert np.all(np.abs(result.summary["mean_corrected_zscore"]) < 4.0)
    assert np.all(np.abs(result.summary["reduced_chi2_corrected"] - 1.0) < 0.15)


def test_closure_rejects_a_mismatched_completeness_correction():
    redshift, model_draws, sigma_draws = _posterior_predictive_inputs(
        n_per_bin=500,
        n_draws=32,
    )

    result = simulate_completeness_closure(
        model_magnitude_draws=model_draws,
        sigma_draws=sigma_draws,
        redshift=redshift,
        completeness_model=HardMagnitudeLimit(22.0),
        correction_completeness_model=AlwaysComplete(),
        magnitude_grid=np.linspace(18.0, 24.5, 1001),
        redshift_bins=np.array([0.4, 0.8, 1.2, 1.6]),
        seed=7721,
        max_abs_mean_zscore=4.0,
        min_detected_per_bin=100,
    )

    assert not result.all_bins_pass
    assert (~result.summary["bin_pass"]).any()
    assert np.nanmax(np.abs(result.summary["mean_corrected_zscore"])) > 4.0


def test_hubble_posterior_closure_uses_selection_prediction_for_each_draw(monkeypatch):
    n_objects = 300
    redshift = np.linspace(0.5, 1.5, n_objects)
    agn_data = {
        "z": redshift,
        "f_host_2500_psf": np.full(n_objects, 0.2),
        "alpha_lambda": np.full(n_objects, -1.5),
    }
    posterior_samples = np.arange(20.0).reshape(5, 4)
    calls = []

    def fake_prediction(theta, **kwargs):
        calls.append(np.asarray(theta))
        return {
            "selection_model_magnitude": np.full(n_objects, 21.0 + theta[0] / 100.0),
            "selection_total_error": np.full(n_objects, 0.25),
        }

    monkeypatch.setattr(
        completeness_closure,
        "agn_selection_prediction",
        fake_prediction,
    )

    result = simulate_hubble_posterior_closure(
        posterior_samples=posterior_samples,
        agn_data=agn_data,
        cosmo_model="FlatLambdaCDM",
        z_pivot_agn=1.0,
        agn_pivot_context=object(),
        completeness_params=(AlwaysComplete(), np.linspace(18.0, 24.5, 301)),
        redshift_bins=np.array([0.4, 0.8, 1.2, 1.6]),
        seed=91,
        max_posterior_draws=3,
        min_detected_per_bin=10,
    )

    assert len(calls) == result.n_posterior_draws == 3
    np.testing.assert_array_equal(calls[0], posterior_samples[0])
    np.testing.assert_array_equal(calls[-1], posterior_samples[-1])
    assert result.all_bins_pass


def test_closure_diagnostics_write_machine_readable_summary_and_plot(tmp_path):
    redshift, model_draws, sigma_draws = _posterior_predictive_inputs()
    result = simulate_completeness_closure(
        model_magnitude_draws=model_draws,
        sigma_draws=sigma_draws,
        redshift=redshift,
        completeness_model=HardMagnitudeLimit(22.0),
        magnitude_grid=np.linspace(18.0, 24.5, 501),
        redshift_bins=np.array([0.4, 0.8, 1.2, 1.6]),
        seed=7721,
        min_detected_per_bin=100,
    )

    paths = write_completeness_closure_diagnostics(result, tmp_path)

    assert paths["summary_csv"].is_file()
    assert paths["metadata_json"].is_file()
    assert paths["plot_pdf"].is_file()
    metadata = json.loads(paths["metadata_json"].read_text())
    assert metadata["all_bins_pass"] is True
    assert metadata["n_posterior_draws"] == result.n_posterior_draws
    assert metadata["seed"] == 7721
