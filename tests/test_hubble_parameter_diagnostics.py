import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble import hubble_plotting  # noqa: E402


def _diagnostic_frame(n=80):
    rng = np.random.default_rng(125)
    z = np.linspace(0.5, 3.0, n)
    driver = z + rng.normal(0.0, 0.04, n)
    residuals = 0.35 * driver + rng.normal(0.0, 0.025, n)
    frame = pd.DataFrame(
        {
            "object_id": [f"obj_{i}" for i in range(n)],
            "z": z,
            "driver": driver,
            "noise": rng.normal(0.0, 1.0, n),
            "constant": np.ones(n),
            "sparse": np.where(np.arange(n) < 5, np.arange(n), np.nan),
            "label": [f"unique_{i}" for i in range(n)],
        }
    )
    frame["array_field"] = [np.array([1.0, 2.0]) for _ in range(n)]
    frame.attrs["spectra_fit_columns"] = ("driver",)
    return frame, residuals


def test_plot_parameter_residual_diagnostics_writes_ranked_summary_and_atlas(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    frame, residuals = _diagnostic_frame()

    outputs = hubble_plotting.plot_parameter_residual_diagnostics(
        frame,
        residuals,
        np.full(len(frame), 0.08),
        plot_path=str(tmp_path / "plots"),
        z_range=(0.44, 3.16),
        show=False,
        min_points=10,
        panels_per_page=2,
        top_n=10,
    )

    assert set(outputs) == {"summary_pdf", "atlas_pdf", "rankings_csv", "skipped_csv"}
    for path in outputs.values():
        assert Path(path).is_file()
        assert Path(path).stat().st_size > 0

    rankings = pd.read_csv(outputs["rankings_csv"])
    skipped = pd.read_csv(outputs["skipped_csv"])
    driver = rankings.set_index("parameter").loc["driver"]
    noise = rankings.set_index("parameter").loc["noise"]

    assert driver["source"] == "spectra_fit_csv"
    assert driver["redshift_correlation_attenuation"] > noise[
        "redshift_correlation_attenuation"
    ]
    assert driver["rho_parameter_redshift"] > 0.9
    assert driver["rho_parameter_residual"] > 0.9
    assert {
        "q_parameter_residual",
        "q_parameter_detrended_residual",
    }.issubset(rankings.columns)

    skipped_by_parameter = skipped.set_index("parameter")["reason"]
    assert skipped_by_parameter["object_id"] == "identifier"
    assert skipped_by_parameter["z"] == "analysis_axis"
    assert skipped_by_parameter["constant"] == "constant"
    assert skipped_by_parameter["sparse"] == "insufficient_finite_values"
    assert skipped_by_parameter["array_field"] == "non_numeric_or_non_scalar"


def test_plot_parameter_residual_diagnostics_rejects_misaligned_inputs(tmp_path):
    frame, residuals = _diagnostic_frame(n=20)

    with pytest.raises(ValueError, match="length"):
        hubble_plotting.plot_parameter_residual_diagnostics(
            frame,
            residuals[:-1],
            np.full(len(frame), 0.08),
            plot_path=str(tmp_path),
            z_range=(0.44, 3.16),
        )


def test_plot_partial_control_atlas_uses_postcut_rows_and_spectra_fields(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    rng = np.random.default_rng(912)
    n = 80
    z = np.linspace(0.5, 3.0, n)
    log_sigma = -0.7 + 0.15 * z + rng.normal(0.0, 0.04, n)
    log_tau = 2.2 - 0.08 * z + rng.normal(0.0, 0.05, n)
    sed_shape = 0.5 * log_sigma + rng.normal(0.0, 0.08, n)
    residuals = 0.3 * sed_shape + 0.15 * log_tau + rng.normal(0.0, 0.03, n)
    frame = pd.DataFrame(
        {
            "object_id": [f"obj_{index}" for index in range(n)],
            "z": z,
            "is_fit_selection": np.arange(n) % 2 == 0,
            "log_sigma_uv": log_sigma,
            "log_tau_uv_rf": log_tau,
            "sed_shape": sed_shape,
            "sed_shape_err": np.full(n, 0.1),
            "not_from_spectra": rng.normal(size=n),
        }
    )
    frame.attrs["spectra_fit_columns"] = (
        "sed_shape",
        "sed_shape_err",
    )

    outputs = hubble_plotting.plot_full_residuals_debiased_partial_controls(
        frame,
        residuals,
        plot_path=str(tmp_path / "plots"),
        z_range=(0.44, 3.16),
        panels_per_page=2,
        min_points=8,
    )

    assert set(outputs) == {"pdf", "residuals_csv", "parameter_index_csv"}
    for path in outputs.values():
        assert Path(path).is_file()
        assert Path(path).stat().st_size > 0

    exported = pd.read_csv(outputs["residuals_csv"])
    parameter_index = pd.read_csv(outputs["parameter_index_csv"])
    assert len(exported) == frame["is_fit_selection"].sum()
    assert set(parameter_index["field"]) == {
        "log_sigma_uv",
        "log_tau_uv_rf",
        "sed_shape",
    }
    assert "sed_shape_partial" in exported
    assert "R_partial_for_sed_shape" in exported
    assert "sed_shape_err" not in parameter_index["field"].tolist()
    assert "not_from_spectra" not in parameter_index["field"].tolist()


def test_plot_redshift_wiggle_diagnostics_finds_jump_and_driver(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    rng = np.random.default_rng(731)
    n = 240
    z = np.linspace(0.5, 3.0, n)
    driver = (z >= 1.72).astype(float) + rng.normal(0.0, 0.03, n)
    debiased = 0.24 * driver + 0.025 * np.sin(10.0 * z)
    debiased += rng.normal(0.0, 0.018, n)
    correction = 0.035 * np.cos(5.0 * z)
    biased = debiased + correction
    frame = pd.DataFrame(
        {
            "object_id": [f"obj_{i}" for i in range(n)],
            "z": z,
            "jump_driver": driver,
            "noise": rng.normal(0.0, 1.0, n),
            "sparse": np.where(np.arange(n) < 18, driver, np.nan),
        }
    )
    frame.attrs["spectra_fit_columns"] = ("jump_driver",)

    outputs = hubble_plotting.plot_redshift_wiggle_diagnostics(
        frame,
        biased,
        np.full(n, 0.06),
        debiased,
        np.full(n, 0.06),
        plot_path=str(tmp_path / "plots"),
        z_range=(0.44, 3.16),
        show=False,
        min_points=10,
        n_bootstrap=30,
        n_permutations=40,
        atlas_top_n=4,
        panels_per_page=2,
    )

    assert set(outputs) == {
        "overview_pdf",
        "change_points_csv",
        "transitions_csv",
        "rankings_csv",
        "atlas_pdf",
    }
    for path in outputs.values():
        assert Path(path).is_file()
        assert Path(path).stat().st_size > 0

    changes = pd.read_csv(outputs["change_points_csv"])
    best_change = changes.iloc[0]
    assert abs(best_change["z_split"] - 1.72) < 0.15
    assert best_change["median_jump"] > 0
    assert 0 <= best_change["look_elsewhere_pvalue"] <= 1

    rankings = pd.read_csv(outputs["rankings_csv"]).set_index("parameter")
    driver_row = rankings.loc["jump_driver"]
    noise_row = rankings.loc["noise"]
    assert driver_row["source"] == "spectra_fit_csv"
    assert driver_row["cv_wiggle_rms_reduction"] > noise_row[
        "cv_wiggle_rms_reduction"
    ]
    assert driver_row["cv_max_jump_reduction"] > 0
    assert bool(driver_row["reliable_redshift_coverage"])
    assert not bool(rankings.loc["sparse", "reliable_redshift_coverage"])


def test_plot_redshift_wiggle_diagnostics_rejects_misaligned_inputs(tmp_path):
    frame, residuals = _diagnostic_frame(n=20)

    with pytest.raises(ValueError, match="length"):
        hubble_plotting.plot_redshift_wiggle_diagnostics(
            frame,
            residuals[:-1],
            np.full(len(frame), 0.08),
            residuals,
            np.full(len(frame), 0.08),
            plot_path=str(tmp_path),
        )
