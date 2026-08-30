import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble import hubble_fit


def test_fresh_empirical_lf_mock_is_wired_through_and_records_provenance(
    tmp_path, monkeypatch
):
    captured = {}

    def fake_sampler(lf_grid, area_deg2, *args, **kwargs):
        captured["model_id"] = lf_grid.model_id
        captured["area_deg2"] = area_deg2
        captured["z_range"] = kwargs["z_range"]
        return (
            [np.array([21.0])],
            np.array([1.0]),
            [np.array([21.2])],
            np.array([1]),
            np.array([0.72]),
            np.array([21.0]),
            np.array([21.2]),
            np.array([0]),
            np.array([-1.5]),
        )

    monkeypatch.setattr(hubble_fit, "mock_lf_grid_per_zbin", fake_sampler)
    output = hubble_fit.generate_fresh_completeness_sim_file(
        tmp_path,
        area_deg2=0.25,
        z_range=(0.68, 0.80),
        completeness_magnitude="attenuated",
        lf_model="kulkarni2019_type1_model2",
    )

    assert captured == {
        "model_id": "kulkarni2019_type1_model2",
        "area_deg2": 0.25,
        "z_range": (0.68, 0.80),
    }
    with h5py.File(output, "r") as handle:
        assert handle.attrs["lf_model"] == "kulkarni2019_type1_model2"
        assert handle.attrs["lf_native_magnitude_name"] == "M_1450_AB"
        assert handle.attrs["requested_redshift_min"] == 0.68
        assert handle.attrs["requested_redshift_max"] == 0.80


def test_empirical_lf_mock_rejects_dereddened_magnitude_semantics(tmp_path):
    with pytest.raises(ValueError, match="require completeness_magnitude='attenuated'"):
        hubble_fit.generate_fresh_completeness_sim_file(
            tmp_path,
            area_deg2=0.25,
            z_range=(0.68, 0.80),
            completeness_magnitude="dereddened",
            lf_model="wang2026_type1_lade_a",
        )
