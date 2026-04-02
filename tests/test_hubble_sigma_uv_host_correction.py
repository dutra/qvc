import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble import hubble_model, hubble_plotting, hubble_utils


def _make_loader_input():
    return pd.DataFrame(
        {
            "object_id": ["agn_valid", "agn_zero", "agn_hi", "agn_neg", "agn_nanerr"],
            "z": [1.2, 1.3, 1.4, 1.5, 1.6],
            "mags_mean_u": [20.1, 20.1, 20.1, 20.1, 20.1],
            "mags_mean_g": [19.8, 19.8, 19.8, 19.8, 19.8],
            "mags_mean_r": [19.5, 19.5, 19.5, 19.5, 19.5],
            "mags_mean_i": [19.2, 19.2, 19.2, 19.2, 19.2],
            "dropped_bands": ["", "", "", "", ""],
            "log_jitter_u": [-9.0, -9.0, -9.0, -9.0, -9.0],
            "log_jitter_g": [-2.0, -2.0, -2.0, -2.0, -2.0],
            "log_jitter_r": [-2.0, -2.0, -2.0, -2.0, -2.0],
            "log_jitter_i": [-2.0, -2.0, -2.0, -2.0, -2.0],
            "log_amp_delta_blr_u": [-9.0, -9.0, -9.0, -9.0, -9.0],
            "log_amp_delta_blr_g": [-2.5, -2.5, -2.5, -2.5, -2.5],
            "log_amp_delta_blr_r": [-2.5, -2.5, -2.5, -2.5, -2.5],
            "log_amp_delta_blr_i": [-2.5, -2.5, -2.5, -2.5, -2.5],
            "alpha_lambda": [-1.5, -1.5, -1.5, -1.5, -1.5],
            "log_sigma_hat0": [-2.1, -2.0, -2.2, -2.3, -2.1],
            "log_sigma_hat0_err": [0.05, 0.05, 0.05, 0.05, 0.05],
            "log_sigma_uv": [-1.0, -0.9, -1.1, -1.2, -0.95],
            "log_sigma_uv_std_psd": [0.10, 0.11, 0.12, 0.13, 0.14],
            "log_tau_uv_rf": [2.4, 2.5, 2.6, 2.7, 2.8],
            "log_tau_uv_rf_std_psd": [0.2, 0.2, 0.2, 0.2, 0.2],
            "log_sigma_uv_log_tau_uv_rf_cov_psd": [0.01, 0.01, 0.01, 0.01, 0.01],
            "apparent_mag_2500": [20.3, 20.4, 20.5, 20.6, 20.7],
            "f_host_2500": [0.2, 0.0, 1.1, -0.1, 0.3],
            "f_host_2500_err": [0.05, 0.0, 0.1, 0.2, np.nan],
        }
    )


def test_load_agn_data_propagates_host_error_into_sigma_uv(monkeypatch, tmp_path):
    df_in = _make_loader_input()
    input_path = tmp_path / "fake_input.h5"
    input_path.touch()

    monkeypatch.setattr(hubble_utils, "read_quasars_from_hdf5_flat", lambda *args, **kwargs: df_in.copy())
    monkeypatch.setattr(hubble_plotting, "plot_sigma_uv_host_correction", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_plotting, "plot_alpha_lambda_vs_redshift", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_plotting, "plot_alpha_lambda_histogram", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_plotting, "plot_tau_sigma_vs_redshift", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_plotting, "plot_sigma_uv_vs_tau_uv_rf", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_plotting, "plot_cut_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda df: df)

    df = hubble_utils.load_agn_data(
        input_path,
        spectra_fit_csv=None,
        lc_info_csv=None,
        only_load=True,
        apply_cut=False,
        correct_sigma_uv_host=True,
        plot_path=str(tmp_path / "figures"),
    )

    ln10 = np.log(10.0)
    orig_sigma = df_in["log_sigma_uv"].to_numpy(dtype=float)
    orig_err = df_in["log_sigma_uv_std_psd"].to_numpy(dtype=float)
    f_host = df_in["f_host_2500"].to_numpy(dtype=float)
    f_host_err = np.nan_to_num(df_in["f_host_2500_err"].to_numpy(dtype=float), nan=0.0)
    valid = np.isfinite(f_host) & (f_host >= 0.0) & (f_host < 1.0)
    expected_delta = np.zeros(len(df_in), dtype=float)
    expected_delta[valid] = -np.log10(1.0 - f_host[valid])
    expected_host_err = np.zeros(len(df_in), dtype=float)
    expected_host_err[valid] = f_host_err[valid] / ((1.0 - f_host[valid]) * ln10)
    expected_err = orig_err.copy()
    expected_err[valid] = np.sqrt(orig_err[valid] ** 2 + expected_host_err[valid] ** 2)

    np.testing.assert_allclose(df["log_sigma_uv_uncorrected"], orig_sigma)
    np.testing.assert_allclose(df["log_sigma_uv"], orig_sigma + expected_delta)
    np.testing.assert_allclose(df["log_sigma_uv_std_psd_uncorrected"], orig_err)
    np.testing.assert_allclose(df["log_sigma_uv_hostcorr_err"], expected_host_err)
    np.testing.assert_allclose(df["log_sigma_uv_std_psd_corrected"], expected_err)
    np.testing.assert_allclose(df["log_sigma_uv_std_psd"], expected_err)

    valid_factor = np.full(len(df_in), np.nan, dtype=float)
    valid_factor[valid] = 1.0 / (1.0 - f_host[valid])
    np.testing.assert_allclose(df["sigma_uv_hostcorr_factor"], valid_factor, equal_nan=True)

    obs_dict = {key: df[key].to_numpy(dtype=float) for key in hubble_model.agn_model_req_obs + hubble_model.agn_model_req_errs}
    obs_arr, err_arr, pivots = hubble_model.agn_model_pack_obs(obs_dict)
    assert obs_arr.shape[1] == len(df)
    assert err_arr.shape[1] == len(df)
    assert pivots.shape[0] == len(hubble_model.agn_model_req_obs)
