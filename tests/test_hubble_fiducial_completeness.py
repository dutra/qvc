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


def test_default_completeness_support_matches_histogram_edges(tmp_path):
    from qvc.hubble.cuts import (
        COMPLETENESS_MAG_EDGE_MAX,
        COMPLETENESS_MAG_EDGE_MIN,
        COMPLETENESS_MAG_2500_MAX,
        COMPLETENESS_MAG_2500_MIN,
    )
    from qvc.hubble.hubble_completeness_refactored import (
        COMPLETENESS_MAG_COL,
        get_completeness_function_2d,
    )

    mock_path = tmp_path / "mock.h5"
    with h5py.File(mock_path, "w") as handle:
        handle.create_dataset("apparent_mag_2500", data=[19.0, 21.0, 23.0])
        handle.create_dataset("z", data=[0.5, 1.5, 2.5])
        handle.attrs["mock_count_scale"] = 1.0

    observed = pd.DataFrame(
        {
            "apparent_mag_2500": [18.6, 20.6, 22.6],
            "m_2500_attenuated_model": [19.0, 21.0, 23.0],
            COMPLETENESS_MAG_COL: [19.0, 21.0, 23.0],
            "z": [0.5, 1.5, 2.5],
        }
    )
    completeness, mag_centers, *_ = get_completeness_function_2d(
        observed,
        sim_file=str(mock_path),
        smooth_counts=False,
    )

    assert COMPLETENESS_MAG_2500_MIN == pytest.approx(COMPLETENESS_MAG_EDGE_MIN)
    assert COMPLETENESS_MAG_2500_MAX == pytest.approx(COMPLETENESS_MAG_EDGE_MAX)
    assert completeness.magnitude_support == pytest.approx(
        (COMPLETENESS_MAG_EDGE_MIN, COMPLETENESS_MAG_EDGE_MAX)
    )
    assert mag_centers[0] > COMPLETENESS_MAG_2500_MIN
    assert mag_centers[-1] < COMPLETENESS_MAG_2500_MAX


def test_completeness_extends_nearest_magnitude_edge_indefinitely(capsys):
    from qvc.hubble.hubble_completeness_refactored import Completeness2D

    magnitude = np.array([19.0, 20.0, 21.0])
    redshift = np.array([0.5, 1.5])
    values = np.array(
        [
            [0.2, 0.3],
            [0.5, 0.6],
            [0.8, 0.9],
        ]
    )
    model = Completeness2D(
        magnitude,
        redshift,
        values,
        magnitude_support=(18.5, 21.5),
    )

    evaluated = model(
        np.array([-100.0, 18.5, 19.0, 21.0, 21.5, 24.0, np.nan]),
        0.5,
    )
    np.testing.assert_allclose(evaluated[:-1], [0.2, 0.2, 0.2, 0.8, 0.8, 0.8])
    assert evaluated[-1] == 0.0
    assert model(24.0, 0.5) == pytest.approx(0.8)

    broadcast = model(np.array([[18.0], [22.0]]), np.array([0.5, 1.5]))
    np.testing.assert_allclose(broadcast, [[0.2, 0.3], [0.8, 0.9]])

    warning = capsys.readouterr().out
    assert warning.count("[WARNING]") == 1
    assert "nearest magnitude-edge value is used" in warning


def test_2d_completeness_smoothing_defaults_and_runner_overrides(
    tmp_path, monkeypatch, capsys
):
    from qvc.hubble.hubble_completeness_refactored import (
        COMPLETENESS_MAG_COL,
        COMPLETENESS_SMOOTH_SIGMA_MAG_ENV,
        COMPLETENESS_SMOOTH_SIGMA_Z_ENV,
        DEFAULT_COMPLETENESS_SMOOTH_SIGMA_MAG,
        DEFAULT_COMPLETENESS_SMOOTH_SIGMA_Z,
        get_completeness_function_2d,
    )

    assert DEFAULT_COMPLETENESS_SMOOTH_SIGMA_MAG == pytest.approx(0.10)
    assert DEFAULT_COMPLETENESS_SMOOTH_SIGMA_Z == pytest.approx(0.30)

    mock_path = tmp_path / "mock.h5"
    with h5py.File(mock_path, "w") as handle:
        handle.create_dataset("apparent_mag_2500", data=[19.0, 21.0, 23.0])
        handle.create_dataset("z", data=[0.5, 1.5, 2.5])
        handle.attrs["mock_count_scale"] = 1.0
    observed = pd.DataFrame(
        {
            COMPLETENESS_MAG_COL: [19.0, 21.0, 23.0],
            "z": [0.5, 1.5, 2.5],
        }
    )

    get_completeness_function_2d(observed, sim_file=str(mock_path), plot=False)
    output = capsys.readouterr().out
    assert "sigma_mag=0.1 mag" in output
    assert "sigma_z=0.3 absolute z" in output

    monkeypatch.setenv(COMPLETENESS_SMOOTH_SIGMA_MAG_ENV, "0.20")
    monkeypatch.setenv(COMPLETENESS_SMOOTH_SIGMA_Z_ENV, "0.20")
    get_completeness_function_2d(observed, sim_file=str(mock_path), plot=False)
    output = capsys.readouterr().out
    assert "sigma_mag=0.2 mag" in output
    assert "sigma_z=0.2 absolute z" in output
