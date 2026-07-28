from pathlib import Path
import sys

import numpy as np
import pytest


pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble.hubble_fit_jax import _agn_model_jax, _prepare_agn_arrays
from qvc.hubble.hubble_model import (
    M_model_agn,
    agn_model_pack_obs,
    build_agn_pivot_context,
)


def _skewed_agn_data():
    sigma_linear = np.array([0.24, 0.26, 8.0])
    tau_days = np.array([140.0, 260.0, 20_000.0])
    return {
        "object_id": np.array(["agn-a", "agn-b", "agn-c"]),
        "z": np.array([0.7, 0.8, 0.9]),
        "log_sigma_uv": np.log10(sigma_linear),
        "log_tau_uv_rf": np.log10(tau_days),
        "log_sigma_uv_std_psd": np.full(3, 0.04),
        "log_tau_uv_rf_std_psd": np.full(3, 0.05),
        "log_sigma_uv_log_tau_uv_rf_cov_psd": np.zeros(3),
    }


def _pivot_context(data):
    return build_agn_pivot_context(data, z_range=(0.5, 1.0))


def test_prepare_agn_arrays_uses_context_rounded_medians_not_means():
    data = _skewed_agn_data()
    context = _pivot_context(data)

    prepared = _prepare_agn_arrays(data, agn_pivot_context=context)
    actual = np.asarray(prepared["_pivot_arr"])
    expected = np.log10(np.array([0.3, 300.0]))
    arithmetic_means = np.array(
        [
            np.mean(data["log_sigma_uv"]),
            np.mean(data["log_tau_uv_rf"]),
        ]
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
    assert not np.allclose(actual, arithmetic_means, rtol=0.0, atol=1e-3)


def test_cpu_and_jax_agn_predictions_share_exact_pivot_context():
    data = _skewed_agn_data()
    context = _pivot_context(data)
    params = np.array([-23.1, -1.7, 0.8])

    obs_arr, _, pivot_arr = agn_model_pack_obs(
        data,
        pivot_context=context,
    )
    cpu_prediction = M_model_agn(params, obs_arr, pivot_arr)

    prepared = _prepare_agn_arrays(data, agn_pivot_context=context)
    jax_prediction = _agn_model_jax(
        jnp.asarray(params),
        prepared["_obs_arr"],
        prepared["_pivot_arr"],
    )

    np.testing.assert_allclose(
        np.asarray(jax_prediction),
        cpu_prediction,
        rtol=1e-12,
        atol=1e-12,
    )


def test_agn_array_preparation_rejects_missing_pivot_context():
    with pytest.raises(
        ValueError,
        match="requires an explicit AgnPivotContext",
    ):
        _prepare_agn_arrays(
            _skewed_agn_data(),
            agn_pivot_context=None,
        )
