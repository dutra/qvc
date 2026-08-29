import numpy as np
import pandas as pd
import pytest

from qvc.hubble import hubble_fit
from qvc.hubble.hubble_likelihood import (
    _joint_attenuation_draw_arrays,
    joint_posterior_completeness_loglike_for_data,
)


class ConstantCompleteness:
    mode = "2d"

    def __call__(self, magnitude, redshift):
        return np.full(np.broadcast_shapes(np.shape(magnitude), np.shape(redshift)), 0.5)


def test_joint_attenuation_uses_only_declared_valid_draws():
    data = {
        "m_2500_dereddened_draws": np.array([[20.0, 20.1, np.nan]]),
        "m_2500_attenuated_model_draws": np.array([[20.3, 20.5, np.nan]]),
        "joint_posterior_valid_count": np.array([2], dtype=np.int16),
    }
    dereddened, attenuated, counts = _joint_attenuation_draw_arrays(
        data, np.array([20.0])
    )
    assert dereddened.shape == attenuated.shape == (1, 3)
    assert counts.tolist() == [2]

    loglike, blob = joint_posterior_completeness_loglike_for_data(
        completeness_params=(ConstantCompleteness(), np.linspace(17.0, 25.0, 401)),
        agn_data=data,
        hubble_magnitude=np.array([20.0]),
        hubble_magnitude_error=np.array([0.1]),
        hubble_model_magnitude=np.array([20.0]),
        hubble_total_error=np.array([0.25]),
        z=np.array([1.0]),
    )
    assert np.isfinite(loglike)
    assert blob.shape == (3, 1)


def test_joint_attenuation_configuration_requires_v3_and_attenuated_selection():
    frame = pd.DataFrame({"apparent_mag_2500": [20.0]})
    with pytest.raises(ValueError, match="attenuated"):
        hubble_fit.validate_selection_attenuation_configuration(
            frame,
            selection_attenuation_mode="joint-posterior",
            completeness=True,
            completeness_magnitude="dereddened",
        )
    with pytest.raises(KeyError, match="spectra v3"):
        hubble_fit.validate_selection_attenuation_configuration(
            frame,
            selection_attenuation_mode="joint-posterior",
            completeness=True,
            completeness_magnitude="attenuated",
        )
