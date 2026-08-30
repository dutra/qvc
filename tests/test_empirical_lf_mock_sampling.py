import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble.completeness_mock_catalog import (
    COSMO,
    _sample_piecewise_linear_grid,
    build_completeness_lf,
    mock_lf_grid_per_zbin,
    native_absolute_magnitude_to_m2500,
    save_mock_catalog,
)
from qvc.hubble.empirical_luminosity_functions import EMPIRICAL_LF_MODEL_IDS


class _FixedUniformRng:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def random(self, size):
        assert int(size) == self.values.size
        return self.values.copy()


def test_piecewise_linear_sampler_inverts_trapezoidal_density():
    # A density proportional to x on [0, 1] has CDF x**2.
    uniforms = np.array([0.01, 0.25, 0.81])
    samples = _sample_piecewise_linear_grid(
        _FixedUniformRng(uniforms),
        np.array([0.0, 1.0]),
        np.array([0.0, 2.0]),
        size=uniforms.size,
    )
    np.testing.assert_allclose(samples, np.sqrt(uniforms), atol=1e-14)


def test_m1450_to_m2500_color_sign_matches_fnu_power_law():
    shift = native_absolute_magnitude_to_m2500(-26.0, -0.5, 1450.0) + 26.0
    np.testing.assert_allclose(shift, -0.2957150080463284, atol=2e-14)


def test_palanque_native_reference_normalization_and_color_shift():
    grid = build_completeness_lf(
        "palanque2016_ple_lede", z_range=(0.68, 0.80)
    )
    shift = (
        native_absolute_magnitude_to_m2500(
            -26.0,
            -0.5,
            grid.reference_wavelength_angstrom,
            grid.native_to_monochromatic_ab_offset,
        )
        + 26.0
    )
    np.testing.assert_allclose(shift, 0.9356226582671712, atol=2e-14)


@pytest.mark.parametrize("model_id", EMPIRICAL_LF_MODEL_IDS)
def test_empirical_lf_sampler_is_finite_and_exactly_inside_support(model_id):
    grid = build_completeness_lf(model_id, z_range=(0.68, 0.80))
    result = mock_lf_grid_per_zbin(
        grid,
        0.1,
        -0.5,
        0.3,
        COSMO,
        z_range=(0.68, 0.80),
        m2500_support=(18.5, 24.0),
        z_res=32,
        rng=np.random.default_rng(1),
        return_global=True,
        return_alpha=True,
    )
    _, expected, _, selected, z, observed_i, observed_2500, _, alpha = result
    assert np.sum(expected) > 0.0
    assert np.sum(selected) == z.size
    assert z.size > 0
    assert all(
        np.all(np.isfinite(values))
        for values in (z, observed_i, observed_2500, alpha)
    )
    assert np.all((z >= 0.68) & (z <= 0.80))
    assert np.all((observed_2500 >= 18.5) & (observed_2500 <= 24.0))


def test_mock_catalog_persists_lf_coordinate_provenance(tmp_path):
    grid = build_completeness_lf(
        "kulkarni2019_type1_model2", z_range=(0.68, 0.80)
    )
    output = tmp_path / "mock.h5"
    save_mock_catalog(
        output,
        np.array([0.7]),
        np.array([21.0]),
        np.array([21.2]),
        alpha_lambda_all=np.array([-1.5]),
        lf_grid=grid,
        m2500_support=(18.5, 24.0),
        z_range=(0.68, 0.80),
    )
    with h5py.File(output, "r") as handle:
        assert handle.attrs["lf_model"] == "kulkarni2019_type1_model2"
        assert handle.attrs["lf_native_magnitude_name"] == "M_1450_AB"
        assert handle.attrs["lf_reference_wavelength_angstrom"] == 1450.0
        assert handle.attrs["m2500_support_min"] == 18.5
        assert handle.attrs["m2500_support_max"] == 24.0
