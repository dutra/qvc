import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.light_curve.fit_light_curves import (
    build_explicit_model_params,
    bending_power_law_psd,
    compute_structure_function_diagnostics,
    compute_band_adf,
    compute_g_band_residual_drift_diagnostics,
    compute_g_band_raw_drift_diagnostics,
    compute_parameter_kls,
    compute_object_adf_diagnostics,
    compute_lambda_center_rf,
    empirical_structure_function,
    fit_bending_power_law_psd,
    lya_variability_weight,
    make_lc,
    posterior_median_mean_function,
)
from qvc.light_curve.multiband_fit_utils import lambda_pivot, log_single_pl, process_samples


def _make_raw_public(n_band):
    return {
        "log_sigma_uv": jnp.array(np.log(0.2)),
        "log_tau_uv": jnp.array(np.log(300.0)),
        "log_tau_fast_uv": jnp.array(np.log(30.0)),
        "eta_sigma": jnp.array(-0.55),
        "eta_tau": jnp.array(0.25),
        "log_amp_delta_lya": jnp.array(0.0),
        "log_amp_delta_blr": jnp.full((n_band,), -1.0),
        "log_amp_delta_blr2": jnp.full((n_band,), -1.5),
        "log_lag_blr": jnp.full((n_band,), np.log(20.0)),
        "log_lag_blr2": jnp.full((n_band,), np.log(60.0)),
        "lag0": jnp.array(5.0),
        "lag_beta": jnp.array(4.0 / 3.0),
    }


def _make_object(z=1.6):
    bands = ["u", "g", "r", "i", "z"]
    return {
        "object_id": "obj",
        "z": z,
        "times": {band: np.array([0.0, 50.0], dtype=float) for band in bands},
        "mags": {band: np.array([20.0, 20.2], dtype=float) for band in bands},
        "magerrs": {band: np.array([0.05, 0.05], dtype=float) for band in bands},
        "cadence": {band: 5.0 for band in bands},
        "cadence_err": {band: 0.5 for band in bands},
        "number_points": {band: 2 for band in bands},
    }




def test_compute_lambda_center_rf_matches_geometric_mean():
    lam_rf = jnp.array([1500.0, 2400.0, 3600.0])
    expected = float(np.exp(np.mean(np.log(np.asarray(lam_rf)))))
    got = float(compute_lambda_center_rf(lam_rf))
    assert np.isclose(got, expected)


def test_build_explicit_model_params_preserves_uv_intercepts_across_band_sets():
    lam_full = jnp.array([1700.0, 2300.0, 3000.0, 3800.0])
    lam_sub = jnp.array([2300.0, 3000.0, 3800.0])

    explicit_full = build_explicit_model_params(_make_raw_public(len(lam_full)), lam_full)
    explicit_sub = build_explicit_model_params(_make_raw_public(len(lam_sub)), lam_sub)

    assert np.isclose(float(explicit_full["log_sigma_uv"]), np.log(0.2))
    assert np.isclose(float(explicit_full["log_tau_uv"]), np.log(300.0))
    assert np.isclose(float(explicit_full["log_tau_fast_uv"]), np.log(30.0))
    assert np.isclose(float(explicit_sub["log_sigma_uv"]), np.log(0.2))
    assert np.isclose(float(explicit_sub["log_tau_uv"]), np.log(300.0))
    assert np.isclose(float(explicit_sub["log_tau_fast_uv"]), np.log(30.0))

    assert np.allclose(
        np.asarray(explicit_full["amp_cont"])[1:],
        np.asarray(explicit_sub["amp_cont"]),
    )
    assert np.allclose(
        np.asarray(explicit_full["tau_fast_band"])[1:],
        np.asarray(explicit_sub["tau_fast_band"]),
    )
    assert np.allclose(
        np.asarray(explicit_full["tau_slow_band"])[1:],
        np.asarray(explicit_sub["tau_slow_band"]),
    )


def test_build_explicit_model_params_internal_and_public_forms_are_equivalent():
    lam_rf = jnp.array([2100.0, 2900.0, 3600.0])
    explicit_public = build_explicit_model_params(_make_raw_public(len(lam_rf)), lam_rf)

    internal = {
        "log_sigma_center0": explicit_public["log_sigma_center0"],
        "log_tau_slow_center0": explicit_public["log_tau_slow_center0"],
        "log_tau_fast_center0": explicit_public["log_tau_fast_center0"],
        "lambda_center_rf": explicit_public["lambda_center_rf"],
        "eta_sigma": jnp.array(-0.55),
        "eta_tau": jnp.array(0.25),
        "log_amp_delta_blr": jnp.full((len(lam_rf),), -1.0),
        "log_amp_delta_blr2": jnp.full((len(lam_rf),), -1.5),
        "log_lag_blr": jnp.full((len(lam_rf),), np.log(20.0)),
        "log_lag_blr2": jnp.full((len(lam_rf),), np.log(60.0)),
        "lag0": jnp.array(5.0),
        "lag_beta": jnp.array(4.0 / 3.0),
    }
    explicit_internal = build_explicit_model_params(internal, lam_rf)

    assert np.isclose(float(explicit_internal["log_sigma_uv"]), float(explicit_public["log_sigma_uv"]))
    assert np.isclose(float(explicit_internal["log_tau_uv"]), float(explicit_public["log_tau_uv"]))
    assert np.isclose(float(explicit_internal["log_tau_fast_uv"]), float(explicit_public["log_tau_fast_uv"]))
    assert np.allclose(np.asarray(explicit_internal["amp_cont"]), np.asarray(explicit_public["amp_cont"]))
    assert np.allclose(np.asarray(explicit_internal["amp_blr"]), np.asarray(explicit_public["amp_blr"]))
    assert np.allclose(np.asarray(explicit_internal["amp_blr2"]), np.asarray(explicit_public["amp_blr2"]))
    assert np.allclose(np.asarray(explicit_internal["lag_blr"]), np.asarray(explicit_public["lag_blr"]))
    assert np.allclose(np.asarray(explicit_internal["lag_blr2"]), np.asarray(explicit_public["lag_blr2"]))
    assert np.allclose(np.asarray(explicit_internal["tau_fast_band"]), np.asarray(explicit_public["tau_fast_band"]))
    assert np.allclose(np.asarray(explicit_internal["tau_slow_band"]), np.asarray(explicit_public["tau_slow_band"]))


def test_build_explicit_model_params_centers_disk_lag_on_geometric_mean():
    lam_rf = jnp.array([1800.0, 2500.0, 3472.222222222222])
    raw = _make_raw_public(len(lam_rf))
    explicit = build_explicit_model_params(raw, lam_rf)

    lambda_center_rf = float(explicit["lambda_center_rf"])
    lag_disk = np.asarray(explicit["lag_disk"])
    expected_lag0 = float(raw["lag0"])

    center_idx = int(np.argmin(np.abs(np.asarray(lam_rf) - lambda_center_rf)))
    assert np.isclose(lambda_center_rf, np.exp(np.mean(np.log(np.asarray(lam_rf)))))
    assert np.isclose(np.asarray(lam_rf)[center_idx], lambda_center_rf)
    assert np.isclose(lag_disk[center_idx], expected_lag0)


def test_lya_variability_weight_is_stronger_blueward_of_lya():
    lam_rf = jnp.array([1050.0, 1216.0, 1600.0, 2500.0])
    weight = np.asarray(lya_variability_weight(lam_rf))
    assert weight[0] > weight[1] > weight[2] > weight[3]
    assert weight[0] > 0.5
    assert weight[-1] < 0.01


def test_build_explicit_model_params_smoothly_suppresses_blue_variability():
    lam_rf = jnp.array([1100.0, 1300.0, 2000.0])
    raw = _make_raw_public(len(lam_rf))
    raw["log_amp_delta_lya"] = jnp.array(-1.0)
    explicit = build_explicit_model_params(raw, lam_rf)

    baseline = build_explicit_model_params(_make_raw_public(len(lam_rf)), lam_rf)
    ratio = np.asarray(explicit["amp_cont"]) / np.asarray(baseline["amp_cont"])

    assert ratio[0] < ratio[1] < ratio[2]
    assert ratio[0] < 0.55
    assert ratio[2] > 0.9


def test_make_lc_drops_z_by_default_but_keeps_lya_bands():
    obj = _make_object(z=1.6)
    lc = make_lc(obj, bands=["u", "g", "r", "i", "z"], drop_band_lyman_alpha=False)

    assert lc is not None
    assert lc["bands"] == ["u", "g", "r", "i"]
    assert lc["dropped_bands"] == ["z"]


def test_make_lc_can_hard_drop_lya_affected_bands():
    obj = _make_object(z=1.6)
    lc = make_lc(obj, bands=["u", "g", "r", "i", "z"], drop_band_lyman_alpha=True)

    assert lc is not None
    assert lc["bands"] == ["g", "r", "i"]
    assert lc["dropped_bands"] == ["u", "z"]


def test_make_lc_adds_variability_fields_for_retained_bands():
    obj = {
        "object_id": "varobj",
        "z": 1.2,
        "times": {
            "u": np.array([0.0, 25.0, 380.0, 405.0], dtype=float),
            "g": np.array([0.0, 25.0, 380.0, 405.0], dtype=float),
            "r": np.array([5.0, 40.0, 385.0, 415.0], dtype=float),
            "i": np.array([10.0, 50.0, 390.0, 420.0], dtype=float),
            "z": np.array([0.0, 1.0], dtype=float),
        },
        "mags": {
            "u": np.array([20.0, 20.2, 19.8, 20.1], dtype=float),
            "g": np.array([20.0, 20.3, 19.7, 20.0], dtype=float),
            "r": np.array([19.8, 20.0, 19.6, 19.9], dtype=float),
            "i": np.array([19.6, 19.9, 19.5, 19.8], dtype=float),
            "z": np.array([19.5, 19.6], dtype=float),
        },
        "magerrs": {band: np.full(len(vals), 0.1, dtype=float) for band, vals in {
            "u": np.array([20.0, 20.2, 19.8, 20.1], dtype=float),
            "g": np.array([20.0, 20.3, 19.7, 20.0], dtype=float),
            "r": np.array([19.8, 20.0, 19.6, 19.9], dtype=float),
            "i": np.array([19.6, 19.9, 19.5, 19.8], dtype=float),
            "z": np.array([19.5, 19.6], dtype=float),
        }.items()},
        "cadence": {band: 5.0 for band in ["u", "g", "r", "i", "z"]},
        "cadence_err": {band: 0.5 for band in ["u", "g", "r", "i", "z"]},
        "number_points": {band: 4 for band in ["u", "g", "r", "i", "z"]},
    }
    obj["number_points"]["z"] = 2

    lc = make_lc(obj, bands=["u", "g", "r", "i", "z"], drop_band_lyman_alpha=False)

    assert lc is not None
    for band in lc["bands"]:
        assert f"variability_n_points_{band}" in lc
        assert f"variability_chi_sq_{band}" in lc
        assert f"variability_chi_sq_red_{band}" in lc
        assert f"variability_pvalue_{band}" in lc
        assert f"variability_neg_log10_pvalue_{band}" in lc
    assert np.isfinite(lc["variability_chi_sq_red_g"])


def test_make_lc_variability_uses_post_filtering_series():
    times = np.arange(13, dtype=float) * 30.0
    g_mags = np.array([20.0, 20.0, 20.1, 20.0, 20.1, 20.0, 23.5, 20.0, 20.1, 20.0, 20.1, 20.0, 20.1], dtype=float)
    obj = {
        "object_id": "outlier",
        "z": 1.0,
        "times": {
            "g": times,
            "r": np.array([0.0, 365.0], dtype=float),
            "i": np.array([20.0, 385.0], dtype=float),
            "z": np.array([0.0, 1.0], dtype=float),
        },
        "mags": {
            "g": g_mags,
            "r": np.array([19.8, 19.9], dtype=float),
            "i": np.array([19.6, 19.7], dtype=float),
            "z": np.array([19.5, 19.6], dtype=float),
        },
        "magerrs": {
            "g": np.full(13, 0.05, dtype=float),
            "r": np.full(2, 0.05, dtype=float),
            "i": np.full(2, 0.05, dtype=float),
            "z": np.full(2, 0.05, dtype=float),
        },
        "cadence": {band: 5.0 for band in ["g", "r", "i", "z"]},
        "cadence_err": {band: 0.5 for band in ["g", "r", "i", "z"]},
        "number_points": {"g": 13, "r": 2, "i": 2, "z": 2},
    }

    lc = make_lc(obj, bands=["g", "r", "i", "z"], drop_band_lyman_alpha=False)

    assert lc is not None
    assert lc["variability_n_points_g"] == 12


def test_compute_parameter_kls_ignores_nonfinite_conditioning_samples():
    flat_samples = {
        "eta_sigma": np.array([-0.4, np.nan, -0.2, -0.3]),
        "eta_tau": np.array([0.1, 0.2, np.nan, 0.25]),
        "log_sigma_center0": np.array([-1.0, -0.9, -0.8, -0.85]),
        "log_tau_slow_center0": np.array([5.0, 5.1, 5.2, 5.15]),
        "log_tau_fast_center0": np.array([2.3, 2.4, 2.5, 2.45]),
        "poly1": np.array([0.0, 0.01, -0.01, 0.02]),
        "lag0": np.array([5.0, 5.5, 4.5, 5.2]),
        "lag_beta": np.array([1.2, 1.4, 1.3, 1.35]),
        "log_amp_delta_lya": np.array([-0.1, -0.2, -0.15, -0.05]),
        "mean_g": np.array([0.0, 0.02, -0.01, 0.01]),
        "mean_r": np.array([0.0, 0.01, -0.02, 0.02]),
        "log_jitter_g": np.array([-3.0, -2.9, -3.1, -3.05]),
        "log_jitter_r": np.array([-3.1, -3.0, -3.2, -3.05]),
        "log_amp_delta_blr_g": np.array([-1.0, -0.9, -1.1, -1.05]),
        "log_amp_delta_blr_r": np.array([-1.2, -1.1, -1.3, -1.25]),
        "log_amp_delta_blr2_g": np.array([-1.5, -1.4, -1.6, -1.55]),
        "log_amp_delta_blr2_r": np.array([-1.7, -1.6, -1.8, -1.75]),
        "log_lag_blr_g": np.array([3.0, 3.1, 3.2, 3.15]),
        "log_lag_blr_r": np.array([3.1, 3.2, 3.3, 3.25]),
        "log_lag_blr2_g": np.array([4.0, 4.1, 4.2, 4.15]),
        "log_lag_blr2_r": np.array([4.1, 4.2, 4.3, 4.25]),
    }

    kls = compute_parameter_kls(
        flat_samples,
        bands=["g", "r"],
        z=1.5,
        lambda_center_rf=2500.0,
        log_jitter_mean=np.array([-3.0, -3.1]),
        disable_poly1=False,
        disable_lag_blr=False,
        drop_band_lyman_alpha=False,
        tau_fast_truncated=False,
        n_blr_terms=2,
    )

    assert "log_sigma_center0_kl" in kls
    assert "log_tau_slow_center0_kl" in kls
    assert "log_tau_fast_center0_kl" in kls
    assert np.isfinite(kls["log_sigma_center0_kl"])
    assert np.isfinite(kls["log_tau_slow_center0_kl"])
    assert np.isfinite(kls["log_tau_fast_center0_kl"])


def test_process_samples_keeps_uv_outputs_at_2500_and_stores_band_metadata():
    z = 1.5
    bands = ["g", "r", "i"]
    lam_rf_kept = np.asarray([lambda_pivot[b] / (1.0 + z) for b in bands], dtype=float)
    lambda_center_rf = float(np.exp(np.mean(np.log(lam_rf_kept))))

    eta_sigma = np.asarray([-0.6, -0.4, -0.5])
    eta_tau = np.asarray([0.1, 0.3, 0.2])
    log_sigma_center0 = np.asarray(np.log([0.18, 0.21, 0.24]))
    log_tau_slow_center0 = np.asarray(np.log([250.0, 310.0, 400.0]))
    log_tau_fast_center0 = np.asarray(np.log([25.0, 32.0, 40.0]))

    flat_samples = {
        "log_sigma_center0": log_sigma_center0,
        "log_tau_slow_center0": log_tau_slow_center0,
        "log_tau_fast_center0": log_tau_fast_center0,
        "log_sigma_uv": log_sigma_center0,
        "log_tau_uv": log_tau_slow_center0,
        "log_tau_fast_uv": log_tau_fast_center0,
        "eta_sigma": eta_sigma,
        "eta_tau": eta_tau,
        "log_lag_blr_g": np.asarray(np.log([30.0, 40.0, 50.0])),
        "log_lag_blr_r": np.asarray(np.log([35.0, 45.0, 55.0])),
        "log_lag_blr_i": np.asarray(np.log([40.0, 50.0, 60.0])),
        "log_lag_blr2_g": np.asarray(np.log([80.0, 90.0, 100.0])),
        "log_lag_blr2_r": np.asarray(np.log([85.0, 95.0, 105.0])),
        "log_lag_blr2_i": np.asarray(np.log([90.0, 100.0, 110.0])),
    }

    result = process_samples(
        flat_samples,
        {"object_id": "obj", "z": z},
        bands=bands,
    )

    expected_log_sigma_uv = np.percentile(log_sigma_center0 / np.log(10), 50)
    expected_log_tau_uv_rf = np.percentile(
        log_tau_slow_center0 / np.log(10) - np.log10(1.0 + z),
        50,
    )
    expected_log_tau_fast_uv_rf = np.percentile(
        log_tau_fast_center0 / np.log(10) - np.log10(1.0 + z),
        50,
    )

    assert np.isclose(result["lambda_center_rf"], lambda_center_rf)
    assert result["n_bands_kept"] == 3
    assert result["bands_kept"] == "g,r,i"
    assert np.isclose(result["log_sigma_uv"], expected_log_sigma_uv)
    assert np.isclose(result["log_tau_uv"], np.percentile(log_tau_slow_center0 / np.log(10), 50))
    assert np.isclose(result["log_tau_uv_rf"], expected_log_tau_uv_rf)
    assert np.isclose(result["log_tau_fast_uv_rf"], expected_log_tau_fast_uv_rf)
    assert np.isclose(
        result["log_lag_blr_r_RF"],
        np.percentile(np.log10([35.0, 45.0, 55.0]) - np.log10(1.0 + z), 50),
    )
    assert np.isclose(
        result["log_lag_blr2_r_RF"],
        np.percentile(np.log10([85.0, 95.0, 105.0]) - np.log10(1.0 + z), 50),
    )
    expected_sigma_rms_g = np.percentile(
        np.log10(np.array([0.18, 0.21, 0.24]) * np.sqrt((np.array([25.0, 32.0, 40.0]) ** 2 + np.array([250.0, 310.0, 400.0]) ** 2) / (np.array([250.0, 310.0, 400.0]) - np.array([25.0, 32.0, 40.0])) ** 2)),
        50,
    )
    assert np.isclose(result["log_sigma_rms_band_g"], expected_sigma_rms_g)


def test_compute_parameter_kls_returns_expected_keys():
    rng = np.random.default_rng(123)
    bands = ["g", "r"]
    z = 1.2
    lam_rf = np.asarray([lambda_pivot[b] / (1.0 + z) for b in bands], dtype=float)
    lambda_center_rf = float(np.exp(np.mean(np.log(lam_rf))))
    n = 64

    flat_samples = {
        "eta_sigma": rng.normal(-0.45, 0.08, size=n),
        "eta_tau": rng.normal(0.55, 0.10, size=n),
        "log_sigma_center0": rng.normal(np.log(0.2), 0.2, size=n),
        "log_tau_slow_center0": rng.normal(np.log(300.0), 0.3, size=n),
        "log_tau_fast_center0": rng.normal(np.log(30.0), 0.2, size=n),
        "poly1": rng.normal(0.0, 0.05, size=n),
        "lag0": np.abs(rng.normal(5.0, 1.0, size=n)),
        "lag_beta": np.abs(rng.normal(4.0 / 3.0, 0.1, size=n)),
        "log_amp_delta_lya": rng.normal(-0.5, 0.1, size=n),
        "mean_g": rng.normal(0.0, 0.05, size=n),
        "mean_r": rng.normal(0.0, 0.05, size=n),
        "log_jitter_g": rng.normal(np.log(0.03), 0.1, size=n),
        "log_jitter_r": rng.normal(np.log(0.03), 0.1, size=n),
        "log_amp_delta_blr_g": rng.normal(-1.0, 0.2, size=n),
        "log_amp_delta_blr_r": rng.normal(-1.0, 0.2, size=n),
        "log_amp_delta_blr2_g": rng.normal(-1.2, 0.2, size=n),
        "log_amp_delta_blr2_r": rng.normal(-1.2, 0.2, size=n),
        "log_lag_blr_g": rng.normal(np.log(20.0), 0.2, size=n),
        "log_lag_blr_r": rng.normal(np.log(25.0), 0.2, size=n),
        "log_lag_blr2_g": rng.normal(np.log(70.0), 0.2, size=n),
        "log_lag_blr2_r": rng.normal(np.log(80.0), 0.2, size=n),
    }

    kls = compute_parameter_kls(
        flat_samples,
        bands=bands,
        z=z,
        lambda_center_rf=lambda_center_rf,
        log_jitter_mean=np.asarray([np.log(0.03), np.log(0.03)]),
        n_blr_terms=2,
    )

    expected_keys = {
        "eta_sigma_kl",
        "eta_tau_kl",
        "log_sigma_center0_kl",
        "log_tau_slow_center0_kl",
        "log_tau_fast_center0_kl",
        "lag0_kl",
        "lag_beta_kl",
        "log_amp_delta_lya_kl",
        "mean_g_kl",
        "mean_r_kl",
        "log_jitter_g_kl",
        "log_jitter_r_kl",
        "log_amp_delta_blr_g_kl",
        "log_amp_delta_blr2_g_kl",
        "log_lag_blr_g_kl",
        "log_lag_blr2_g_kl",
        "kl_total",
    }
    assert expected_keys.issubset(kls.keys())
    assert all(np.isfinite(kls[key]) for key in expected_keys)


def test_posterior_median_mean_function_uses_global_time_normalization():
    t_eval = np.array([0.0, 20.0], dtype=float)
    t_ref = np.array([0.0, 10.0, 20.0], dtype=float)
    flat_samples = {
        "mean_g": np.array([0.8, 1.0, 1.2], dtype=float),
        "poly1": np.array([0.1, 0.2, 0.3], dtype=float),
    }

    got = posterior_median_mean_function(flat_samples, t_eval, "g", t_ref=t_ref)

    t_center = 10.0
    t_std = np.std(t_ref)
    expected = 1.0 + 0.2 * ((t_eval - t_center) / t_std)
    assert np.allclose(got, expected)


def test_compute_band_adf_returns_valid_output_for_stationary_series():
    rng = np.random.default_rng(7)
    noise = rng.normal(scale=0.3, size=256)
    x = np.empty_like(noise)
    x[0] = noise[0]
    for i in range(1, x.size):
        x[i] = 0.5 * x[i - 1] + noise[i]

    result = compute_band_adf(x)

    assert result["adf_valid"] is True
    assert np.isfinite(result["adf_stat"])
    assert np.isfinite(result["adf_pvalue"])
    assert result["adf_nobs"] > 0


def test_compute_band_adf_rejects_constant_or_short_series():
    short_result = compute_band_adf(np.array([1.0, 2.0, 3.0], dtype=float))
    const_result = compute_band_adf(np.ones(16, dtype=float))

    assert short_result["adf_valid"] is False
    assert np.isnan(short_result["adf_stat"])
    assert const_result["adf_valid"] is False
    assert np.isnan(const_result["adf_pvalue"])


def test_empirical_structure_function_matches_stone_definition():
    t = np.array([0.0, 1.0, 2.0], dtype=float)
    y = np.array([0.0, 1.0, 2.0], dtype=float)
    yerr = np.array([0.1, 0.2, 0.3], dtype=float)

    tau, sf, sf_lo, sf_hi = empirical_structure_function(
        t,
        y,
        yerr,
        bins_per_decade=1,
        min_pairs=3,
    )

    pair_terms = np.array(
        [
            1.0**2 - 0.1**2 - 0.2**2,
            2.0**2 - 0.1**2 - 0.3**2,
            1.0**2 - 0.2**2 - 0.3**2,
        ],
        dtype=float,
    )
    expected_tau = 10.0 ** np.mean(np.log10([1.0, 2.0, 1.0]))
    expected_sf = np.sqrt(np.mean(pair_terms))

    assert tau.shape == (1,)
    assert np.isclose(tau[0], expected_tau)
    assert np.isclose(sf[0], expected_sf)
    assert sf_lo[0] <= sf[0] <= sf_hi[0]


def test_compute_structure_function_diagnostics_returns_finite_sensible_g_band_fit():
    rng = np.random.default_rng(11)
    sigma_true = 0.18
    tau_true = 300.0
    t_g = np.linspace(0.0, 3000.0, 40, dtype=float)
    dt = np.abs(t_g[:, None] - t_g[None, :])
    cov = (sigma_true**2) * np.exp(-dt / tau_true)
    y_g = rng.multivariate_normal(np.zeros(t_g.size, dtype=float), cov)
    yerr_g = np.full(t_g.size, 0.01, dtype=float)

    t_r = np.linspace(0.0, 3000.0, 8, dtype=float)
    y_r = np.zeros(t_r.size, dtype=float)
    yerr_r = np.full(t_r.size, 0.02, dtype=float)

    obj = {
        "bands": ["g", "r"],
        "band_idx": np.array([0] * t_g.size + [1] * t_r.size, dtype=int),
        "X": (np.concatenate([t_g, t_r]), np.array([0] * t_g.size + [1] * t_r.size, dtype=int)),
        "y": np.concatenate([y_g, y_r]),
        "yerr": np.concatenate([yerr_g, yerr_r]),
    }
    samples = {
        "eta_sigma": np.array([0.0, 0.05, -0.05], dtype=float),
        "eta_tau": np.array([0.0, 0.05, -0.05], dtype=float),
        "amp_cont_g": np.array([0.18, 0.19, 0.20], dtype=float),
        "tau_fast_g": np.array([35.0, 30.0, 40.0], dtype=float),
        "tau_slow_g": np.array([320.0, 300.0, 340.0], dtype=float),
    }

    result = compute_structure_function_diagnostics(samples, obj, z=0.8)

    assert result["sf_ref_band"] == "g"
    assert np.isfinite(result["log_sigma_sf_ref_band"])
    assert np.isfinite(result["log_tau_sf_ref_band"])
    assert 1.0 < result["log_tau_sf_ref_band"] < 4.0
    assert -2.0 < result["log_sigma_sf_ref_band"] < 0.0
    assert np.isclose(result["log_sigma_sf_ref_band"], np.log10(sigma_true), atol=0.35)
    assert np.isfinite(result["log_sigma_sf_model_ref_band"])
    assert np.isfinite(result["log_tau_sf_model_ref_band"])
    assert result["sf_model_valid"] is True
    assert result["log_tau_sf_model_ref_band"] < np.log10(np.median(samples["tau_slow_g"]))


def test_compute_object_adf_diagnostics_returns_per_band_fields():
    obj = {
        "X": (np.tile(np.arange(24, dtype=float), 2), np.array([0] * 24 + [1] * 24, dtype=int)),
        "y": np.concatenate(
            [
                np.sin(np.arange(24, dtype=float) / 4.0),
                np.cos(np.arange(24, dtype=float) / 5.0),
            ]
        ),
        "band_idx": np.array([0] * 24 + [1] * 24, dtype=int),
    }
    flat_samples = {
        "mean_g": np.array([0.0, 0.1, 0.0], dtype=float),
        "mean_r": np.array([0.0, -0.1, 0.0], dtype=float),
        "poly1": np.array([0.0, 0.01, 0.0], dtype=float),
    }

    result = compute_object_adf_diagnostics(flat_samples, obj, ["g", "r"])

    for band in ("g", "r"):
        assert f"adf_stat_{band}" in result
        assert f"adf_pvalue_{band}" in result
        assert f"adf_usedlag_{band}" in result
        assert f"adf_nobs_{band}" in result
        assert f"adf_valid_{band}" in result
    assert "adf_min_pvalue" in result
    assert "adf_any_pvalue_lt_0p05" in result


def test_compute_g_band_residual_drift_diagnostics_returns_finite_positive_slopes():
    t_bins = [
        np.array([100.0, 250.0, 500.0, 750.0], dtype=float),
        np.array([1100.0, 1250.0, 1500.0, 1750.0], dtype=float),
        np.array([2100.0, 2250.0, 2500.0, 2750.0], dtype=float),
        np.array([3100.0, 3250.0, 3500.0, 3750.0], dtype=float),
    ]
    means = [0.0, 0.2, 0.4, 0.6]
    spreads = [0.05, 0.1, 0.2, 0.4]
    y_chunks = []
    for mu, spread in zip(means, spreads):
        y_chunks.append(mu + np.array([-1.5, -0.5, 0.5, 1.5], dtype=float) * spread)

    t_g = np.concatenate(t_bins)
    y_g = np.concatenate(y_chunks)
    yerr_g = np.full(t_g.size, 0.03, dtype=float)

    obj = {
        "z": 0.0,
        "X": (t_g, np.zeros(t_g.size, dtype=int)),
        "y": y_g,
        "yerr": yerr_g,
        "band_idx": np.zeros(t_g.size, dtype=int),
    }
    flat_samples = {
        "mean_g": np.array([0.0, 0.0, 0.0], dtype=float),
        "poly1": np.array([0.0, 0.0, 0.0], dtype=float),
    }

    result = compute_g_band_residual_drift_diagnostics(
        flat_samples,
        obj,
        ["g"],
        z=0.0,
        bin_width_rf_days=1000.0,
        min_count=4,
    )

    assert result["g_resid_n_bins"] == 4
    assert result["g_resid_mean_trend_valid"] is True
    assert result["g_resid_var_trend_valid"] is True
    assert np.isfinite(result["g_resid_mean_slope"])
    assert np.isfinite(result["g_resid_mean_slope_err"])
    assert np.isfinite(result["g_resid_var_slope"])
    assert np.isfinite(result["g_resid_var_slope_err"])
    assert result["g_resid_mean_slope"] > 0.0
    assert result["g_resid_var_slope"] > 0.0


def test_compute_g_band_raw_drift_diagnostics_returns_finite_positive_mean_slope():
    t_bins = [
        np.array([100.0, 250.0, 500.0, 750.0], dtype=float),
        np.array([1100.0, 1250.0, 1500.0, 1750.0], dtype=float),
        np.array([2100.0, 2250.0, 2500.0, 2750.0], dtype=float),
        np.array([3100.0, 3250.0, 3500.0, 3750.0], dtype=float),
    ]
    means = [19.0, 19.2, 19.4, 19.6]
    spreads = [0.05, 0.08, 0.1, 0.12]
    y_chunks = []
    for mu, spread in zip(means, spreads):
        y_chunks.append(mu + np.array([-1.5, -0.5, 0.5, 1.5], dtype=float) * spread)

    t_g = np.concatenate(t_bins)
    y_g = np.concatenate(y_chunks)
    yerr_g = np.full(t_g.size, 0.03, dtype=float)

    obj = {
        "z": 0.0,
        "X": (t_g, np.zeros(t_g.size, dtype=int)),
        "y": y_g,
        "yerr": yerr_g,
        "band_idx": np.zeros(t_g.size, dtype=int),
    }
    flat_samples = {
        "mean_g": np.array([19.0, 19.0, 19.0], dtype=float),
        "poly1": np.array([0.0, 0.0, 0.0], dtype=float),
    }

    result = compute_g_band_raw_drift_diagnostics(
        flat_samples,
        obj,
        ["g"],
        z=0.0,
        bin_width_rf_days=1000.0,
        min_count=4,
    )

    assert result["g_raw_n_bins"] == 4
    assert result["g_raw_mean_trend_valid"] is True
    assert np.isfinite(result["g_raw_mean_slope"])
    assert np.isfinite(result["g_raw_mean_slope_err"])
    assert result["g_raw_mean_slope"] > 0.0


def test_fit_bending_power_law_psd_recovers_tau_and_sigma():
    freq = np.logspace(-4.5, -1.5, 60)
    true_log_sigma = np.log10(0.2)
    true_log_tau = np.log10(300.0)
    power = bending_power_law_psd(freq, true_log_sigma, true_log_tau)
    power_lo = power * 0.9
    power_hi = power * 1.1

    result = fit_bending_power_law_psd(freq, power, power_lo, power_hi)

    assert result["psd_bpl_valid"] is True
    assert np.isclose(result["log_sigma_bpl"], true_log_sigma, atol=0.05)
    assert np.isclose(result["log_tau_bpl"], true_log_tau, atol=0.05)


def test_fit_bending_power_law_psd_handles_too_few_bins():
    result = fit_bending_power_law_psd(
        np.array([1e-3, 2e-3, 3e-3]),
        np.array([1.0, 2.0, 3.0]),
    )

    assert result["psd_bpl_valid"] is False
    assert np.isnan(result["log_sigma_bpl"])
