import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble import hubble_plotting, hubble_utils


def _patch_plotting(monkeypatch):
    plot_names = [
        "plot_alpha_lambda_histogram",
        "plot_alpha_lambda_vs_eta_sigma",
        "plot_alpha_lambda_vs_l2500",
        "plot_alpha_lambda_vs_redshift",
        "plot_blr_amp_vs_redshift_by_band",
        "plot_blr_lag_vs_amp_by_band",
        "plot_blr_lag_vs_redshift_by_band",
        "plot_cut_diagnostics",
        "plot_eta_tau_sigma_vs_redshift",
        "plot_f_host_2500_vs_l2500",
        "plot_f_host_2500_vs_redshift",
        "plot_fast_vs_uv_variability",
        "plot_g_band_drift_slope_histograms",
        "plot_l2500_vs_eta_sigma_fiducial",
        "plot_l2500_vs_uv_variability_fiducial",
        "plot_m2500_vs_z_colorpanels",
        "plot_sf_vs_uv_variability",
        "plot_sigma_uv_host_correction",
        "plot_sigma_uv_vs_tau_uv_rf",
        "plot_sigma_uv_vs_variability_chi_sq_red_g",
        "plot_spectral_fraction_vs_redshift",
        "plot_tau_sigma_vs_redshift",
        "plot_tau_sigma_vs_wu_catalog",
        "plot_blr_line_lags_vs_l2500_fiducial",
    ]
    for name in plot_names:
        monkeypatch.setattr(hubble_plotting, name, lambda *args, **kwargs: None)


def _make_cut_input():
    return pd.DataFrame(
        {
            "object_id": ["good", "bad_mag", "bad_tau", "bad_alpha_err"],
            "sdss_name": ["good", "bad_mag", "bad_tau", "bad_alpha_err"],
            "z": [1.2, 1.3, 1.4, 1.5],
            "mags_mean_u": [20.1, 20.1, 20.1, 20.1],
            "mags_mean_g": [19.8, 19.8, 19.8, 19.8],
            "mags_mean_r": [19.5, 19.5, 19.5, 19.5],
            "mags_mean_i": [19.2, 19.2, 19.2, 19.2],
            "dropped_bands": ["", "", "", ""],
            "log_jitter_u": [-9.0, -9.0, -9.0, -9.0],
            "log_jitter_g": [-2.0, -2.0, -2.0, -2.0],
            "log_jitter_r": [-2.0, -2.0, -2.0, -2.0],
            "log_jitter_i": [-2.0, -2.0, -2.0, -2.0],
            "log_amp_delta_blr_u": [-9.0, -9.0, -9.0, -9.0],
            "log_amp_delta_blr_g": [-2.5, -2.5, -2.5, -2.5],
            "log_amp_delta_blr_r": [-2.5, -2.5, -2.5, -2.5],
            "log_amp_delta_blr_i": [-2.5, -2.5, -2.5, -2.5],
            "alpha_lambda": [-1.5, -1.5, -1.5, -1.5],
            "alpha_lambda_err": [0.1, 0.1, 0.1, pd.NA],
            "log_sigma_uv": [-1.0, -1.0, -0.2, -1.0],
            "log_sigma_uv_err": [0.1, 0.1, 0.1, 0.1],
            "log_sigma_uv_std_psd": [0.1, 0.1, 0.1, 0.1],
            "log_tau_uv_rf": [3.0, 3.0, 2.7, 3.0],
            "log_tau_uv_rf_err": [0.2, 0.2, 0.2, 0.2],
            "log_tau_uv_rf_std_psd": [0.2, 0.2, 0.2, 0.2],
            "log_sigma_uv_log_tau_uv_rf_cov_psd": [0.01, 0.01, 0.01, 0.01],
            "apparent_mag_2500": [20.3, 25.0, 20.5, 20.6],
            "apparent_mag_2500_err": [0.1, 0.1, 0.1, 0.1],
            "f_host_2500": [0.2, 0.2, 0.2, 0.2],
            "f_host_5100": [0.3, 0.3, 0.3, 0.3],
            "wrms": [0.8, 0.8, 0.8, 0.8],
            "t_rf_length": [2000.0, 2000.0, 2000.0, 2000.0],
            "f_fe_uv_over_pl_3000": [0.2, 0.2, 0.2, 0.2],
            "f_bc_over_pl_3000": [0.2, 0.2, 0.2, 0.2],
            "variability_chi_sq_red_g": [25.0, 25.0, 25.0, 25.0],
        }
    )


def test_cut_summary_report_file_and_stdout(monkeypatch, tmp_path, capsys):
    df_in = _make_cut_input()
    input_path = tmp_path / "fake_input.h5"
    input_path.touch()
    report_path = tmp_path / "cut_summary.txt"

    _patch_plotting(monkeypatch)
    monkeypatch.setattr(hubble_utils, "read_quasars_from_hdf5_flat", lambda *args, **kwargs: df_in.copy())
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda df: df)

    df = hubble_utils.load_agn_data(
        input_path,
        spectra_fit_csv=None,
        lc_info_csv=None,
        apply_cut=True,
        plot_path=str(tmp_path / "figures"),
        cut_report_path=report_path,
    )

    out = capsys.readouterr().out
    report_text = report_path.read_text(encoding="utf-8")

    assert len(df) == 1
    assert df.iloc[0]["object_id"] == "good"
    assert report_path.exists()
    assert "step" in report_text
    assert "criterion" in report_text
    assert "apparent_mag_2500" in report_text
    assert "tau_sigma" in report_text
    assert "not_nan:alpha_lambda_err" in report_text
    assert "agn_scalar:frac_host_psf_2500" in report_text
    assert "skipped" in report_text
    assert hubble_utils.PURPLE_ANSI in out
    assert hubble_utils.RESET_ANSI in out


def test_cut_summary_records_optional_residual_sigma_clip(monkeypatch, tmp_path):
    df_in = _make_cut_input().iloc[[0, 3]].reset_index(drop=True)
    input_path = tmp_path / "fake_input.h5"
    input_path.touch()
    report_path = tmp_path / "cut_summary_residual.txt"
    residuals_path = tmp_path / "residuals.csv"
    pd.DataFrame(
        {
            "object_id": ["good", "bad_alpha_err"],
            "mu_zscore": [0.1, 9.0],
            "residuals": [0.01, 0.02],
        }
    ).to_csv(residuals_path, index=False)

    _patch_plotting(monkeypatch)
    monkeypatch.setattr(hubble_utils, "read_quasars_from_hdf5_flat", lambda *args, **kwargs: df_in.copy())
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda df: df)

    df = hubble_utils.load_agn_data(
        input_path,
        spectra_fit_csv=None,
        lc_info_csv=None,
        apply_cut=False,
        residuals_sigma_clip=3.0,
        residuals_csv=residuals_path,
        plot_path=str(tmp_path / "figures"),
        cut_report_path=report_path,
    )

    report_text = report_path.read_text(encoding="utf-8")

    assert len(df) == 1
    assert df.iloc[0]["object_id"] == "good"
    assert "residual_sigma_clip" in report_text
    assert "|mu_zscore| < 3.0" in report_text


def test_wrap_text_in_purple():
    wrapped = hubble_utils._wrap_text_in_purple("hello")
    assert wrapped == f"{hubble_utils.PURPLE_ANSI}hello{hubble_utils.RESET_ANSI}"
