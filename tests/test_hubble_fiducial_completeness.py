import os
import sys
from pathlib import Path

import h5py
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
