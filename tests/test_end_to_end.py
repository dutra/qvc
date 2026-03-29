import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import device_get, random
from jax.tree_util import tree_map
from numpyro.handlers import seed, trace
from numpyro.infer import MCMC, NUTS


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble import hubble_plotting, hubble_utils
from qvc.light_curve.fit_light_curves import (
    build_single_object_model,
    compute_g_band_residual_drift_diagnostics,
    compute_g_band_raw_drift_diagnostics,
    compute_object_adf_diagnostics,
    make_lc,
)
from qvc.light_curve.multiband_fit_utils import (
    flatten_flat_samples_per_band,
    lambda_pivot,
    process_samples,
)
from qvc.spectra.fit_spectra import effective_decompose_host_flag, effective_fit_bal_flag


def _write_test_quasars_hdf5(path, quasars):
    path.parent.mkdir(parents=True, exist_ok=True)
    string_dt = h5py.string_dtype(encoding="utf-8")

    def _to_scalar(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="replace")
        return x

    def _flatten_value(row, base_key, value):
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                _flatten_value(row, f"{base_key}_{sub_key}", sub_value)
            return

        arr = np.asarray(value)
        if arr.ndim == 0:
            row[base_key] = _to_scalar(arr.reshape(-1)[0])
            return

        flat = arr.reshape(-1)
        for i, item in enumerate(flat):
            row[f"{base_key}_{i}"] = _to_scalar(item)

    rows = []
    for quasar in quasars:
        row = {"object_id": str(quasar["object_id"])}
        for key, value in quasar.items():
            if key == "object_id":
                continue
            _flatten_value(row, str(key), value)
        rows.append(row)

    all_keys = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                all_keys.append(key)

    with h5py.File(path, "w") as hdf:
        for key in all_keys:
            values = [row.get(key, None) for row in rows]
            has_string = any(isinstance(v, (str, bytes)) for v in values if v is not None)
            if has_string:
                col = []
                for v in values:
                    if v is None:
                        col.append("")
                    elif isinstance(v, bytes):
                        col.append(v.decode("utf-8", errors="replace"))
                    else:
                        col.append(str(v))
                hdf.create_dataset(key, data=np.asarray(col, dtype=object).astype(string_dt))
            else:
                col = []
                for v in values:
                    if v is None:
                        col.append(np.nan)
                    else:
                        col.append(float(v))
                hdf.create_dataset(key, data=np.asarray(col, dtype=float))


def _make_fake_public_object():
    bands = ("g", "r", "i", "z")
    return {
        "object_id": "101",
        "z": 1.35,
        "times": {
            band: np.linspace(58000.0, 58540.0, 12, dtype=float)
            for band in bands
        },
        "mags": {
            "g": np.linspace(20.0, 20.25, 12, dtype=float),
            "r": np.linspace(19.7, 19.92, 12, dtype=float),
            "i": np.linspace(19.5, 19.72, 12, dtype=float),
            "z": np.linspace(19.3, 19.55, 12, dtype=float),
        },
        "magerrs": {band: np.full(12, 0.05, dtype=float) for band in bands},
        "cadence": {band: 7.0 for band in bands},
        "cadence_err": {band: 0.5 for band in bands},
        "number_points": {band: 12 for band in bands},
    }


def test_plot_adf_pvalue_g_diagnostic_writes_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    df = pd.DataFrame(
        {
            "adf_pvalue_g": [0.001, 0.004, 0.02, 0.03, 0.07, 0.11, 0.2, 0.5, 0.8, np.nan],
        }
    )

    out = hubble_plotting.plot_adf_pvalue_g_diagnostic(
        df,
        plot_path=str(tmp_path / "figures"),
        show=False,
    )

    assert out is not None
    assert os.path.exists(out)
    assert out.endswith("adf_pvalue_g_diagnostic.pdf")


def test_effective_decompose_host_flag_disables_host_above_z_1p5():
    assert effective_decompose_host_flag(0.8, requested=True) is True
    assert effective_decompose_host_flag(1.5, requested=True) is True
    assert effective_decompose_host_flag(1.5001, requested=True) is False
    assert effective_decompose_host_flag(3.0, requested=True) is False
    assert effective_decompose_host_flag(0.8, requested=False) is False


def test_effective_fit_bal_flag_enables_bal_only_above_z_2():
    assert effective_fit_bal_flag(1.5) is False
    assert effective_fit_bal_flag(2.0) is False
    assert effective_fit_bal_flag(2.0001) is True
    assert effective_fit_bal_flag(3.0) is True
    assert effective_fit_bal_flag(np.nan) is False


def test_plot_g_band_drift_slope_histograms_writes_pdfs(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    df = pd.DataFrame(
        {
            "g_raw_mean_slope": [-2e-4, -1e-4, 0.0, 1e-4, 2e-4],
            "g_resid_mean_slope": [-8e-5, -2e-5, 0.0, 2e-5, 8e-5],
            "g_raw_var_slope": [-5e-6, -1e-6, 0.0, 1e-6, 5e-6],
            "g_resid_var_slope": [-2e-6, -5e-7, 0.0, 5e-7, 2e-6],
        }
    )

    out_mean = hubble_plotting.plot_g_band_drift_slope_histograms(
        df,
        slope_kind="mean",
        plot_path=str(tmp_path / "figures"),
        show=False,
    )
    out_var = hubble_plotting.plot_g_band_drift_slope_histograms(
        df,
        slope_kind="var",
        plot_path=str(tmp_path / "figures"),
        show=False,
    )

    assert out_mean is not None
    assert out_var is not None
    assert os.path.exists(out_mean)
    assert os.path.exists(out_var)
    assert out_mean.endswith("g_band_mean_slope_histograms.pdf")
    assert out_var.endswith("g_band_var_slope_histograms.pdf")


def test_plot_spectral_fraction_vs_redshift_writes_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    df = pd.DataFrame(
        {
            "z": np.linspace(0.3, 2.2, 24),
            "f_bc_over_pl_3000": np.linspace(0.05, 0.25, 24),
            "f_fe_uv_over_pl_3000": np.linspace(0.1, 0.4, 24),
            "f_host_center": np.linspace(0.3, 0.02, 24),
        }
    )

    out = hubble_plotting.plot_spectral_fraction_vs_redshift(
        df,
        plot_path=str(tmp_path / "figures"),
        show=False,
        nbins=6,
        min_bin_count=3,
    )

    assert out is not None
    assert os.path.exists(out)
    assert out.endswith("spectral_fraction_vs_redshift.pdf")


def test_plot_blr_line_lags_vs_l2500_fiducial_writes_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    df = pd.DataFrame(
        {
            "object_id": ["a", "b", "c", "d"],
            "z": [1.2, 0.8, 0.5, 0.2],
            "apparent_mag_2500": [20.5, 20.0, 19.5, 19.0],
            "log_sigma_uv": [-0.7, -0.8, -0.9, -1.0],
            "dropped_bands": [[], [], [], []],
            "log_amp_delta_blr_g": [0.0, 0.0, 0.0, 0.0],
            "log_lag_blr_g_RF": [1.15, 1.32, 1.52, 1.70],
            "log_lag_blr_g_RF_err": [0.1, 0.1, 0.1, 0.1],
        }
    )

    out = hubble_plotting.plot_blr_line_lags_vs_l2500_fiducial(
        df,
        plot_path=str(tmp_path / "figures"),
        show=False,
        prob_thresh=0.0,
    )

    assert out is not None
    assert os.path.exists(out)
    assert out.endswith("blr_line_lags_vs_l2500_fiducial.pdf")


def test_plot_blr_assignment_probabilities_writes_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    assignments = pd.DataFrame(
        {
            "assigned_line": ["C IV", "Mg II", "Hβ", "Hα", "Mg II", "C IV"],
            "assigned_prob": [0.25, 0.42, 0.58, 0.71, 0.33, 0.61],
        }
    )

    out = hubble_plotting.plot_blr_assignment_probabilities(
        assignments,
        plot_path=str(tmp_path / "figures"),
        show=False,
        filename="blr_assignment_probabilities_test.pdf",
    )

    assert out is not None
    assert os.path.exists(out)
    assert out.endswith("blr_assignment_probabilities_test.pdf")


def test_blr_line_assignment_uses_visibility_only():
    df = pd.DataFrame(
        {
            "object_id": ["obj1"],
            "z": [1.0],
            "dropped_bands": [[]],
            "log_sigma_uv": [-1.0],
            "log_amp_delta_blr_g": [0.0],
            "log_lag_blr_g_RF": [100.0],
            "log_lag_blr_g_RF_err": [0.1],
        }
    )

    out = hubble_plotting._blr_line_assignment_longform(
        df,
        np.array([np.nan], dtype=float),
    )

    assert len(out) == 1
    assert out.iloc[0]["assigned_line"] == "Mg II"
    assert np.isfinite(out.iloc[0]["assigned_prob"])
    assert out.iloc[0]["assigned_prob"] > out.iloc[0]["p_C_IV"]
    assert out.iloc[0]["assigned_prob"] > out.iloc[0]["p_Hb"]


def test_build_single_object_model_disables_second_blr_term_by_default():
    obj = _make_fake_public_object()
    lc = make_lc(
        obj,
        ["g", "r"],
        inject_fake=False,
        drop_band_lyman_alpha=False,
    )
    obj = obj | lc

    bands = obj["bands"]
    lam_rf = jnp.array([lambda_pivot[b] for b in bands], dtype=float) / (1.0 + float(obj["z"]))
    bidx = np.asarray(obj["band_idx"])
    yerr = np.asarray(obj["yerr"])
    log_jitter_mean = np.array(
        [
            np.log(np.mean(yerr[(bidx == i) & np.isfinite(yerr) & (yerr < 10)]))
            for i in range(len(bands))
        ],
        dtype=float,
    )
    model = build_single_object_model(
        obj,
        lam_rf,
        log_jitter_mean=jnp.array(log_jitter_mean),
        disable_poly1=False,
        disable_lag_blr=False,
        drop_band_lyman_alpha=False,
        tau_fast_truncated=False,
        n_blr_terms=1,
    )

    model_trace = trace(seed(model, random.PRNGKey(0))).get_trace()
    assert "log_amp_delta_blr_raw" in model_trace
    assert "log_lag_blr_raw" in model_trace
    assert "log_amp_delta_blr2_raw" not in model_trace
    assert "log_lag_blr2_raw" not in model_trace
    assert np.allclose(np.asarray(model_trace["log_amp_delta_blr2"]["value"]), -9.0)
    assert np.allclose(np.asarray(model_trace["log_lag_blr2"]["value"]), -9.0)


def test_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    for name in (
        "plot_alpha_lambda_vs_l2500_by_redshift",
        "plot_adf_pvalue_g_diagnostic",
        "plot_alpha_lambda_histogram",
        "plot_blr_lag_vs_amp_by_band",
        "plot_blr_line_lags_vs_l2500_fiducial",
        "plot_blr_lag_vs_redshift_by_band",
        "plot_f_host_center_vs_l2500",
        "plot_g_band_drift_slope_histograms",
        "plot_Mi_relation",
        "plot_cut_diagnostics",
        "plot_m2500_vs_z_colorpanels",
        "plot_sigma_uv_host_correction",
        "plot_tau_sigma_vs_wu_catalog",
        "plot_tau_sigma_vs_redshift",
    ):
        monkeypatch.setattr(hubble_plotting, name, lambda *args, **kwargs: None)

    monkeypatch.setattr(hubble_utils, "populate_xray", lambda df: df)

    obj = _make_fake_public_object()
    lc = make_lc(obj, bands=["g", "r", "i", "z"], inject_fake=True, drop_band_lyman_alpha=False)

    assert lc is not None
    assert lc["bands"] == ["g", "r", "i"]
    obj = obj | lc

    bands = obj["bands"]
    lam_rf = jnp.array([lambda_pivot[b] for b in bands], dtype=float) / (1.0 + float(obj["z"]))
    bidx = np.asarray(obj["band_idx"])
    yerr = np.asarray(obj["yerr"])
    log_jitter_mean = np.array(
        [
            np.log(np.mean(yerr[(bidx == i) & np.isfinite(yerr) & (yerr < 10)]))
            for i in range(len(bands))
        ],
        dtype=float,
    )
    numpyro_model = build_single_object_model(
        obj,
        lam_rf,
        log_jitter_mean=jnp.array(log_jitter_mean),
        disable_poly1=False,
        disable_lag_blr=False,
        drop_band_lyman_alpha=False,
        tau_fast_truncated=False,
        n_blr_terms=1,
    )

    nuts = NUTS(
        numpyro_model,
        dense_mass=False,
        max_tree_depth=2,
        target_accept_prob=0.8,
    )
    mcmc = MCMC(
        nuts,
        num_warmup=5,
        num_samples=8,
        num_chains=1,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(0))

    samples_flat = mcmc.get_samples(group_by_chain=False)
    samples_flat = tree_map(lambda x: np.asarray(device_get(x)), samples_flat)
    flat_per_band = flatten_flat_samples_per_band(samples_flat, bands=bands)
    result = process_samples(flat_per_band, obj, bands=bands)
    adf_result = compute_object_adf_diagnostics(flat_per_band, obj, bands)
    drift_result = compute_g_band_residual_drift_diagnostics(flat_per_band, obj, bands, z=float(obj["z"]))
    raw_drift_result = compute_g_band_raw_drift_diagnostics(flat_per_band, obj, bands, z=float(obj["z"]))

    quasar = {
        "object_id": obj["object_id"],
        "z": float(obj["z"]),
        "mags_mean_u": np.nan,
        "mags_mean_g": float(obj["mags_means"][0]),
        "mags_mean_r": float(obj["mags_means"][1]),
        "mags_mean_i": float(obj["mags_means"][2]),
        "mags_mean_z": np.nan,
        "dropped_bands": ",".join(obj["dropped_bands"]),
        "t_rf_length": float(obj["t_rf_length"]),
        "t_obs_length": float(obj["t_obs_length"]),
        "ebv_wu": 0.01,
        "apparent_mag_2500": 20.2,
        "alpha_lambda": -1.45,
        "alpha_lambda_err": 0.08,
        "ra": 150.0,
        "dec": 2.0,
        "cadence": obj["cadence"],
        "cadence_err": obj["cadence_err"],
        "number_points": obj["number_points"],
        "log_sigma_uv": float(result["log_sigma_uv"]),
        "log_sigma_uv_err": float(result["log_sigma_uv_err"]),
        "log_tau_uv_rf": float(result["log_tau_uv_rf"]),
        "log_tau_uv_rf_err": float(result["log_tau_uv_rf_err"]),
        "log_sigma_uv_log_tau_uv_rf_cov_psd": float(result["log_sigma_uv_log_tau_uv_rf_cov_psd"]),
        "log_sigma_uv_std_psd": float(result["log_sigma_uv_std_psd"]),
        "log_tau_uv_rf_std_psd": float(result["log_tau_uv_rf_std_psd"]),
        "log_jitter_u": -9.0,
        "log_amp_delta_blr_u": -9.0,
        "log_jitter_g": float(np.percentile(flat_per_band["log_jitter_g"], 50)),
        "log_jitter_r": float(np.percentile(flat_per_band["log_jitter_r"], 50)),
        "log_jitter_i": float(np.percentile(flat_per_band["log_jitter_i"], 50)),
        "log_amp_delta_blr_g": float(np.percentile(flat_per_band["log_amp_delta_blr_g"], 50)),
        "log_amp_delta_blr_r": float(np.percentile(flat_per_band["log_amp_delta_blr_r"], 50)),
        "log_amp_delta_blr_i": float(np.percentile(flat_per_band["log_amp_delta_blr_i"], 50)),
    }
    quasar.update(adf_result)
    quasar.update(drift_result)
    quasar.update(raw_drift_result)

    h5_path = tmp_path / "data" / "fake_light_curve_end_to_end.h5"
    _write_test_quasars_hdf5(h5_path, [quasar])

    df = hubble_utils.load_agn_data(
        h5_path,
        spectra_fit_csv=None,
        lc_info_csv=None,
        only_load=True,
        apply_cut=False,
        plot_path=str(tmp_path / "figures"),
    )

    assert len(df) == 1
    row = df.iloc[0]
    assert row["object_id"] == obj["object_id"]
    assert row["len_dropped_bands"] == 1
    assert np.isclose(row["t_rf_length"], obj["t_rf_length"])
    assert np.isclose(row["log_sigma_uv"], result["log_sigma_uv"])
    assert np.isclose(row["log_tau_uv_rf"], result["log_tau_uv_rf"])
    assert np.isclose(row["mags_mean_g"], obj["mags_means"][0])
    assert np.isclose(row["mags_mean_r"], obj["mags_means"][1])
    assert np.isclose(row["mags_mean_i"], obj["mags_means"][2])
    assert "adf_min_pvalue" in row.index
    assert "adf_any_pvalue_lt_0p05" in row.index
    assert "adf_pvalue_g" in row.index
    assert "adf_pvalue_r" in row.index
    assert "adf_pvalue_i" in row.index
    assert "g_resid_mean_slope" in row.index
    assert "g_resid_mean_slope_err" in row.index
    assert "g_resid_var_slope" in row.index
    assert "g_resid_var_slope_err" in row.index
    assert "g_raw_mean_slope" in row.index
    assert "g_raw_mean_slope_err" in row.index
    assert "g_raw_var_slope" in row.index
    assert "g_raw_var_slope_err" in row.index
