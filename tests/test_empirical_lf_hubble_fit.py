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
from qvc.hubble.hubble_completeness_refactored import (
    _validate_mock_magnitude_coverage,
)


def test_fresh_empirical_lf_mock_is_wired_through_and_records_provenance(
    tmp_path, monkeypatch
):
    captured = {}

    def fake_sampler(lf_grid, area_deg2, *args, **kwargs):
        captured["model_id"] = lf_grid.model_id
        captured["area_deg2"] = area_deg2
        captured["z_range"] = kwargs["z_range"]
        captured["m2500_support"] = kwargs["m2500_support"]
        return (
            [np.array([16.5, 24.0])],
            np.array([2.0]),
            [np.array([16.5, 27.5])],
            np.array([2]),
            np.array([0.70, 0.72]),
            np.array([16.5, 24.0]),
            np.array([16.5, 27.5]),
            np.array([0, 0]),
            np.array([-1.5, -1.5]),
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
        "m2500_support": (16.5, 27.5),
    }
    with h5py.File(output, "r") as handle:
        _validate_mock_magnitude_coverage(handle["apparent_mag_2500"][:])
        assert handle.attrs["lf_model"] == "kulkarni2019_type1_model2"
        assert handle.attrs["lf_native_magnitude_name"] == "M_1450_AB"
        assert handle.attrs["m2500_support_min"] == 16.5
        assert handle.attrs["m2500_support_max"] == 27.5
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


def test_old_mock_magnitude_support_requires_regeneration():
    with pytest.raises(ValueError, match='Regenerate'):
        _validate_mock_magnitude_coverage(np.linspace(18., 24.5, 100))
    _validate_mock_magnitude_coverage(np.linspace(16.5, 27.5, 111))


def test_sparse_bright_mock_uses_generation_support(tmp_path):
    import pandas as pd
    from qvc.hubble.hubble_completeness_refactored import get_completeness_function_2d

    path = tmp_path / "sparse_bright_mock.h5"
    with h5py.File(path, "w") as handle:
        handle["apparent_mag_2500"] = np.linspace(16.6409, 27.49998, 100)
        handle["z"] = np.linspace(0.0, 4.5, 100)
        handle.attrs["m2500_support_min"] = 16.5
        handle.attrs["m2500_support_max"] = 27.5
        handle.attrs["mock_count_scale"] = 1.0
    observed = pd.DataFrame({"z": [1.0], "completeness_m_2500": [21.0]})
    model, *_ = get_completeness_function_2d(observed, sim_file=str(path))
    assert np.isfinite(model(21.0, 1.0))
    with h5py.File(path, "a") as handle:
        del handle.attrs["m2500_support_min"]
        del handle.attrs["m2500_support_max"]
        # Desired map bounds are not evidence of generator support.
        handle.attrs["completeness_map_magnitude_min"] = 16.5
        handle.attrs["completeness_map_magnitude_max"] = 27.5
    with pytest.raises(ValueError, match="Regenerate"):
        get_completeness_function_2d(observed, sim_file=str(path))


@pytest.mark.parametrize("support", [(17.0, 27.5), (16.5, 27.0), (np.nan, 27.5), (27.5, 16.5)])
def test_declared_mock_support_must_be_valid_and_cover_map(support):
    with pytest.raises(ValueError):
        _validate_mock_magnitude_coverage([17.5, 26.0], declared_support=support)


def test_mock_values_must_agree_with_declared_support():
    with pytest.raises(ValueError, match="outside declared"):
        _validate_mock_magnitude_coverage([16.0, 27.5], declared_support=(16.5, 27.5))
