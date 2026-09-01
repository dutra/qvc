import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble import hubble_plotting, hubble_utils  # noqa: E402


def _minimal_agn_frame(n=10):
    z = np.linspace(0.7, 2.0, n)
    return pd.DataFrame(
        {
            "object_id": [f"obj{i}" for i in range(n)],
            "sdss_name": [f"sdss{i}" for i in range(n)],
            "z": z,
            "dropped_bands": ["" for _ in range(n)],
            "mags_mean_u": np.full(n, 20.5),
            "mags_mean_g": np.full(n, 20.0),
            "mags_mean_r": np.full(n, 19.8),
            "mags_mean_i": np.full(n, 19.7),
            "log_jitter_u": np.full(n, -4.0),
            "log_jitter_g": np.full(n, -4.0),
            "log_jitter_r": np.full(n, -4.0),
            "log_jitter_i": np.full(n, -4.0),
            "dlog_amp_blr_u": np.full(n, -1.0),
            "dlog_amp_blr_g": np.full(n, -1.0),
            "dlog_amp_blr_r": np.full(n, -1.0),
            "dlog_amp_blr_i": np.full(n, -1.0),
            "log_tau_uv_rf": np.full(n, 2.5),
            "wrms": np.full(n, 0.5),
            "t_rf_length": np.full(n, 2000.0),
            "f_host_2500": np.full(n, 5e-4),
            "f_host_2500_psf": np.full(n, 0.2),
            "frac_host_psf_2500": np.full(n, 5e-4),
            "frac_host_psf_2500_err": np.full(n, 2e-4),
            "alpha_lambda": np.full(n, -1.5),
            "variability_chi_sq_red_g": np.full(n, 30.0),
            "log_sigma_uv": np.full(n, -0.6),
            "log_sigma_uv_std_psd": np.full(n, 0.05),
            "log_sigma_uv_bpl": np.full(n, -0.62),
            "log_sigma_uv_bpl_err": np.full(n, 0.06),
            "log_sigma_ls": np.full(n, -0.58),
            "log_sigma_ls_err": np.full(n, 0.05),
            "log_tau_uv_rf_bpl": np.full(n, 2.45),
            "log_tau_uv_rf_bpl_err": np.full(n, 0.09),
            "log_tau_ls": np.full(n, 2.52),
            "log_tau_ls_err": np.full(n, 0.08),
            "apparent_mag_2500": np.linspace(20.0, 21.0, n),
            "apparent_mag_2500_err": np.full(n, 0.01),
            "m_2500_dereddened": np.linspace(20.0, 21.0, n),
            "m_2500_dereddened_err": np.full(n, 0.01),
            "m_2500_attenuated_model": np.linspace(20.0, 21.0, n),
            "m_2500_attenuated_model_err": np.full(n, 0.01),
            "sed_reduced_chi2": np.full(n, 1.0),
            "spectroscopy_reduced_chi2": np.full(n, 1.0),
            "joint_reduced_chi2": np.full(n, 1.0),
            "loo_chi2_eff": np.full(n, 1.0),
            "m_2500_dereddened_rhat": np.full(n, 1.01),
            "m_2500_attenuated_model_rhat": np.full(n, 1.01),
            "log_tau_uv_rf_rhat": np.full(n, 1.01),
            "log_sigma_uv_rhat": np.full(n, 1.01),
            "SDSS_RUN2D": np.full(n, "v5_13_2"),
            "number_points_g": np.full(n, 300),
            "number_points_r": np.full(n, 300),
            "log_sigma0": np.full(n, -1.2),
            "log_sigma0_err": np.full(n, 0.04),
            "log_amp_delta_blr_u": np.full(n, -0.25),
            "log_amp_delta_blr_u_err": np.full(n, 0.03),
            "log_amp_delta_blr_g": np.full(n, -0.20),
            "log_amp_delta_blr_g_err": np.full(n, 0.03),
            "log_amp_delta_blr_r": np.full(n, -0.15),
            "log_amp_delta_blr_r_err": np.full(n, 0.03),
            "log_amp_delta_blr_i": np.full(n, -0.10),
            "log_amp_delta_blr_i_err": np.full(n, 0.03),
            "log_sigma_band_u": np.full(n, -1.0),
            "log_sigma_band_u_err": np.full(n, 0.04),
            "log_sigma_band_g": np.full(n, -0.95),
            "log_sigma_band_g_err": np.full(n, 0.04),
            "log_sigma_band_r": np.full(n, -0.90),
            "log_sigma_band_r_err": np.full(n, 0.04),
            "log_sigma_band_i": np.full(n, -0.85),
            "log_sigma_band_i_err": np.full(n, 0.04),
            "log_sigma_band_z": np.full(n, -0.80),
            "log_sigma_band_z_err": np.full(n, 0.04),
            "log_tau_band_u_RF": np.full(n, 2.40),
            "log_tau_band_u_RF_err": np.full(n, 0.05),
            "log_tau_band_g_RF": np.full(n, 2.45),
            "log_tau_band_g_RF_err": np.full(n, 0.05),
            "log_tau_band_r_RF": np.full(n, 2.50),
            "log_tau_band_r_RF_err": np.full(n, 0.05),
            "log_tau_band_i_RF": np.full(n, 2.55),
            "log_tau_band_i_RF_err": np.full(n, 0.05),
            "log_tau_band_z_RF": np.full(n, 2.60),
            "log_tau_band_z_RF_err": np.full(n, 0.05),
        }
    )


def _patch_load_agn_plotters(monkeypatch):
    plot_noops = (
        "plot_adf_pvalue_g_diagnostic",
        "plot_alpha_lambda_vs_l2500_by_redshift",
        "plot_alpha_lambda_vs_l2500",
        "plot_alpha_lambda_vs_eta_sigma",
        "plot_alpha_lambda_histogram",
        "plot_alpha_lambda_vs_redshift",
        "plot_blr_amp_vs_redshift_by_band",
        "plot_bc_lag_vs_l2500",
        "plot_blr_line_lags_vs_l2500_fiducial",
        "plot_blr_lag_vs_amp_by_band",
        "plot_blr_lag_vs_redshift_by_band",
        "plot_bpl_psd_vs_uv_variability",
        "plot_eta_tau_sigma_vs_redshift",
        "plot_fast_vs_uv_variability",
        "plot_f_host_2500_vs_redshift",
        "plot_g_band_drift_slope_histograms",
        "plot_l2500_vs_eta_sigma_fiducial",
        "plot_l2500_vs_uv_variability_fiducial",
        "plot_linear_trend_vs_redshift",
        "plot_Mi_relation",
        "plot_light_curve_n_points_vs_apparent_mag",
        "plot_cut_diagnostics",
        "plot_m2500_vs_z_colorpanels",
        "plot_spectral_fraction_vs_redshift",
        "plot_sf_ref_band_vs_model_g",
        "plot_sf_vs_uv_variability",
        "plot_sigma_bc_vs_frac_bc",
        "plot_sigma_bc_vs_redshift",
        "plot_sigma_tau_err_std_psd_comparison",
        "plot_sigma_tau_vs_lambda_broken_pl_fit",
        "plot_sigma_uv_vs_variability_chi_sq_red_g",
        "plot_sigma_uv_vs_tau_uv_rf",
        "plot_sigma_uv_host_correction",
        "plot_suberlak_style_sigma_tau_fits",
        "plot_tau_sigma_vs_wu_catalog",
        "plot_tau_sigma_vs_redshift",
        "plot_tier1_cuts_vs_redshift",
        "plot_f_host_2500_vs_l2500",
        "plot_blr_diagnostics_summary",
    )
    for name in plot_noops:
        monkeypatch.setattr(hubble_plotting, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hubble_utils, "plot_sigma_tau_identity_grid", lambda *_args, **_kwargs: None)


def test_load_agn_data_makes_default_combined_tier1_diagnostic(tmp_path, monkeypatch):
    source_path = tmp_path / "agn.h5"
    source_path.touch()
    frame = _minimal_agn_frame(n=4)
    monkeypatch.setattr(
        hubble_utils,
        "read_quasars_from_hdf5_flat",
        lambda *_args, **_kwargs: frame.copy(),
    )
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda value: value)
    _patch_load_agn_plotters(monkeypatch)

    calls = []
    monkeypatch.setattr(
        hubble_plotting,
        "plot_tier1_cuts_vs_redshift",
        lambda data, **kwargs: calls.append((data.copy(), kwargs)),
    )

    hubble_utils.load_agn_data(
        source_path,
        magnitude_convention="dereddened",
        spectra_fit_h5=None,
        cut_tier="none",
        plot_path=str(tmp_path / "plots"),
    )

    assert len(calls) == 1
    plotted, kwargs = calls[0]
    assert plotted["object_id"].tolist() == frame["object_id"].tolist()
    assert kwargs["filename"] == "tier1_cuts_vs_redshift_precut.pdf"
    assert kwargs["plot_path"] == str(tmp_path / "plots")


def test_load_agn_data_makes_pre_and_postcut_joint_sed_and_blr_plots(tmp_path, monkeypatch):
    source_path = tmp_path / "agn.h5"
    source_path.touch()
    frame = _minimal_agn_frame()
    frame["eta_sigma"] = np.linspace(-0.9, -0.6, len(frame))
    frame["eta_sigma_err"] = np.full(len(frame), 0.2)
    frame["eta_sigma_kl"] = np.linspace(0.0, 1.0, len(frame))
    frame["eta_tau"] = np.full(len(frame), 0.3)
    frame["eta_prior_profile"] = "modified"
    frame["a_2500_total"] = np.zeros(len(frame))
    monkeypatch.setattr(
        hubble_utils,
        "read_quasars_from_hdf5_flat",
        lambda *_args, **_kwargs: frame.copy(),
    )
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda df: df)

    captured_calls = []

    def capture_plot(*_args, **kwargs):
        captured_calls.append(kwargs)
        return None

    plot_noops = (
        "plot_adf_pvalue_g_diagnostic",
        "plot_alpha_lambda_vs_l2500_by_redshift",
        "plot_alpha_lambda_vs_l2500",
        "plot_alpha_lambda_vs_eta_sigma",
        "plot_alpha_lambda_histogram",
        "plot_alpha_lambda_vs_redshift",
        "plot_blr_amp_vs_redshift_by_band",
        "plot_bc_lag_vs_l2500",
        "plot_blr_line_lags_vs_l2500_fiducial",
        "plot_blr_lag_vs_amp_by_band",
        "plot_blr_lag_vs_redshift_by_band",
        "plot_bpl_psd_vs_uv_variability",
        "plot_eta_tau_sigma_vs_redshift",
        "plot_fast_vs_uv_variability",
        "plot_f_host_2500_vs_redshift",
        "plot_g_band_drift_slope_histograms",
        "plot_l2500_vs_eta_sigma_fiducial",
        "plot_l2500_vs_uv_variability_fiducial",
        "plot_linear_trend_vs_redshift",
        "plot_Mi_relation",
        "plot_light_curve_n_points_vs_apparent_mag",
        "plot_cut_diagnostics",
        "plot_m2500_vs_z_colorpanels",
        "plot_spectral_fraction_vs_redshift",
        "plot_sf_ref_band_vs_model_g",
        "plot_sf_vs_uv_variability",
        "plot_sigma_bc_vs_frac_bc",
        "plot_sigma_bc_vs_redshift",
        "plot_sigma_tau_err_std_psd_comparison",
        "plot_sigma_tau_vs_lambda_broken_pl_fit",
        "plot_sigma_uv_vs_variability_chi_sq_red_g",
        "plot_sigma_uv_vs_tau_uv_rf",
        "plot_sigma_uv_host_correction",
        "plot_suberlak_style_sigma_tau_fits",
        "plot_tau_sigma_vs_wu_catalog",
        "plot_tau_sigma_vs_redshift",
        "plot_tier1_cuts_vs_redshift",
    )
    for name in plot_noops:
        monkeypatch.setattr(hubble_plotting, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hubble_plotting, "plot_f_host_2500_vs_l2500", capture_plot)
    monkeypatch.setattr(hubble_plotting, "plot_alpha_lambda_vs_l2500", capture_plot)
    monkeypatch.setattr(hubble_plotting, "plot_blr_diagnostics_summary", capture_plot)
    monkeypatch.setattr(hubble_plotting, "plot_sigma_tau_vs_lambda_broken_pl_fit", capture_plot)
    monkeypatch.setattr(
        hubble_plotting,
        "plot_eta_sigma_vs_redshift_colored_by_kl",
        capture_plot,
    )

    hubble_utils.load_agn_data(
        source_path,
        magnitude_convention="dereddened",
        spectra_fit_csv=None,
        lc_info_csv=None,
        cut_tier="2",
        plot_path=str(tmp_path / "figures"),
        cut_report_path=tmp_path / "cut_summary.txt",
    )

    captured_by_filename = {call.get("filename"): call for call in captured_calls}
    assert captured_by_filename["f_host_2500_vs_l2500_precut.pdf"]["f_host_col"] == "f_host_2500_psf"
    assert captured_by_filename["f_host_2500_vs_l2500_postcut.pdf"]["f_host_col"] == "f_host_2500_psf"
    assert "alpha_lambda_vs_l2500_precut.pdf" in captured_by_filename
    assert "alpha_lambda_vs_l2500_postcut.pdf" in captured_by_filename
    assert "blr_precut.pdf" in captured_by_filename
    assert "blr_postcut.pdf" in captured_by_filename
    assert "sigma_tau_vs_lambda_broken_pl_fit_postcut.pdf" in captured_by_filename
    precut_eta = captured_by_filename[
        "eta_sigma_vs_redshift_colored_by_kl_precut.pdf"
    ]
    postcut_eta = captured_by_filename[
        "eta_sigma_vs_redshift_colored_by_kl_postcut.pdf"
    ]
    assert precut_eta["sample_label"] == "Pre-cut sample"
    assert postcut_eta["sample_label"] == "Post-cut sample"
    assert precut_eta["kl_color_limits"] == pytest.approx((0.01, 0.99))
    assert postcut_eta["kl_color_limits"] == pytest.approx((0.01, 0.99))


def test_load_agn_data_writes_sigma_tau_ls_identity_grids_to_diagnostics(tmp_path, monkeypatch):
    source_path = tmp_path / "agn.h5"
    source_path.touch()
    monkeypatch.setattr(
        hubble_utils,
        "read_quasars_from_hdf5_flat",
        lambda *_args, **_kwargs: _minimal_agn_frame(),
    )
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda df: df)

    plot_calls = []

    def capture_identity_grid(*_args, **kwargs):
        plot_calls.append(kwargs)
        return None

    monkeypatch.setattr(hubble_utils, "plot_sigma_tau_identity_grid", capture_identity_grid)

    hubble_utils.load_agn_data(
        source_path,
        magnitude_convention="dereddened",
        spectra_fit_csv=None,
        lc_info_csv=None,
        cut_tier="2",
        plot_path=str(tmp_path / "plots" / "hubble" / "prefix"),
        cut_report_path=tmp_path / "cut_summary.txt",
    )

    assert len(plot_calls) == 2
    by_suffix = {
        Path(call["output_path"]).name: call
        for call in plot_calls
    }
    assert "sigma_tau_ls_identity_precut.pdf" in by_suffix
    assert "sigma_tau_ls_identity_postcut.pdf" in by_suffix

    precut_output = Path(by_suffix["sigma_tau_ls_identity_precut.pdf"]["output_path"])
    postcut_output = Path(by_suffix["sigma_tau_ls_identity_postcut.pdf"]["output_path"])
    expected_diagnostics_dir = tmp_path / "plots" / "hubble" / "prefix" / "diagnostics"
    assert precut_output.parent == expected_diagnostics_dir
    assert postcut_output.parent == expected_diagnostics_dir

    postcut_kwargs = by_suffix["sigma_tau_ls_identity_postcut.pdf"]
    assert postcut_kwargs["sigma_limits"] == (-1.9, 1.2)
    assert postcut_kwargs["tau_limits"] == (-0.2, 4.9)


def test_load_agn_data_writes_psd_uv_recovery_comparisons(tmp_path, monkeypatch):
    source_path = tmp_path / "agn.h5"
    source_path.touch()
    frame = _minimal_agn_frame()
    frame["log_sigma_uv_err"] = 0.05
    frame["log_tau_uv_rf_err"] = 0.08
    frame["alpha_high_ls"] = -2.0
    frame["alpha_high_ls_err"] = 0.05
    frame["psd_ls_valid"] = True
    frame["log_sigma_ls_fixed"] = -0.65
    frame["log_sigma_ls_fixed_err"] = 0.05
    frame["log_tau_ls_fixed"] = 2.45
    frame["log_tau_ls_fixed_err"] = 0.08
    frame["psd_ls_fixed_valid"] = True
    frame["ebv_gal"] = 0.02
    frame["ebv_agn"] = 0.02

    monkeypatch.setattr(
        hubble_utils,
        "read_quasars_from_hdf5_flat",
        lambda *_args, **_kwargs: frame.copy(),
    )
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda value: value)
    _patch_load_agn_plotters(monkeypatch)

    calls = []
    monkeypatch.setattr(
        hubble_plotting,
        "plot_psd_uv_recovery_comparison",
        lambda data, **kwargs: calls.append((data.copy(), kwargs)),
    )

    hubble_utils.load_agn_data(
        source_path,
        magnitude_convention="dereddened",
        spectra_fit_h5=None,
        cut_tier="1",
        plot_path=str(tmp_path / "plots"),
    )

    assert [call[1]["filename"] for call in calls] == [
        "sigma_tau_psd_free_vs_fixed_precut.pdf",
        "sigma_tau_psd_free_vs_fixed_postcut.pdf",
    ]
    assert all(call[1]["plot_path"] == str(tmp_path / "plots") for call in calls)


def test_plot_f_host_2500_vs_l2500_accepts_psf_column(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    df = _minimal_agn_frame(n=12).drop(columns=["f_host_2500"])
    df["f_host_2500_psf"] = df["frac_host_psf_2500"]

    out = hubble_plotting.plot_f_host_2500_vs_l2500(
        df,
        plot_path=str(tmp_path / "figures"),
        show=False,
        fit_logL_max=99.0,
        filename="f_host_2500_psf_vs_l2500.pdf",
        f_host_col="f_host_2500_psf",
        f_host_label=r"$f_{\rm host,2500}^{\rm PSF}$",
    )

    assert out is not None
    assert os.path.exists(out)
    assert out.endswith("f_host_2500_psf_vs_l2500.pdf")


def test_plot_blr_diagnostics_summary_writes_default_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    old_schema_cols = [
        "log_sigma0",
        "log_sigma0_err",
        "log_amp_delta_blr_u",
        "log_amp_delta_blr_u_err",
        "log_amp_delta_blr_g",
        "log_amp_delta_blr_g_err",
        "log_amp_delta_blr_r",
        "log_amp_delta_blr_r_err",
        "log_amp_delta_blr_i",
        "log_amp_delta_blr_i_err",
    ]
    df = _minimal_agn_frame(n=12).drop(columns=old_schema_cols)

    out = hubble_plotting.plot_blr_diagnostics_summary(
        df,
        plot_path=str(tmp_path / "figures"),
        show=False,
    )

    assert out is not None
    assert os.path.exists(out)
    assert out.endswith(os.path.join("diagnostics", "blr.pdf"))


def test_plot_blr_diagnostics_summary_marks_out_of_range_redshifts(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    df = _minimal_agn_frame(n=6)
    df["z"] = [0.2, 0.7, 1.2, 2.0, 3.0, 3.8]
    errorbar_calls = []
    original_errorbar = hubble_plotting.mpl.axes.Axes.errorbar

    def capture_errorbar(self, *args, **kwargs):
        errorbar_calls.append(kwargs)
        return original_errorbar(self, *args, **kwargs)

    monkeypatch.setattr(hubble_plotting.mpl.axes.Axes, "errorbar", capture_errorbar)

    out = hubble_plotting.plot_blr_diagnostics_summary(
        df,
        plot_path=str(tmp_path / "figures"),
        show=False,
    )

    formats = [call.get("fmt") for call in errorbar_calls]
    assert out is not None
    assert "o" in formats
    assert "D" in formats
    assert any(call.get("markersize") == 4 for call in errorbar_calls if call.get("fmt") == "o")
    assert any(call.get("markersize") == 3 for call in errorbar_calls if call.get("fmt") == "D")


def test_plot_blr_diagnostics_summary_returns_none_when_columns_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    df = _minimal_agn_frame(n=8).drop(columns=["dlog_amp_blr_u"])

    out = hubble_plotting.plot_blr_diagnostics_summary(
        df,
        plot_path=str(tmp_path / "figures"),
        show=False,
    )

    assert out is None


def test_load_agn_data_run2d_filter_v5_13_2_and_drop_missing(tmp_path, monkeypatch):
    source_path = tmp_path / "agn.h5"
    source_path.touch()
    df = _minimal_agn_frame(n=5)
    df["object_id"] = ["a", "b", "c", "d", "e"]
    df["SDSS_RUN2D"] = ["v5_13_2", "26", "", None, "v5_13_2"]
    df["frac_host_psf_2500"] = np.full(len(df), 5e-4)

    monkeypatch.setattr(hubble_utils, "read_quasars_from_hdf5_flat", lambda *_args, **_kwargs: df.copy())
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda frame: frame)
    _patch_load_agn_plotters(monkeypatch)

    filtered, _ = hubble_utils.load_agn_data(
        source_path,
        magnitude_convention="dereddened",
        spectra_fit_csv=None,
        lc_info_csv=None,
        cut_tier="2",
        spectra_sdss_run2d="v5_13_2",
        plot_path=str(tmp_path / "figures"),
        cut_report_path=tmp_path / "cut_summary.txt",
    )

    assert filtered["object_id"].tolist() == ["a", "e"]


def test_load_agn_data_run2d_filter_26(tmp_path, monkeypatch):
    source_path = tmp_path / "agn.h5"
    source_path.touch()
    df = _minimal_agn_frame(n=4)
    df["object_id"] = ["a", "b", "c", "d"]
    df["SDSS_RUN2D"] = ["v5_13_2", "26", "26", "v5_13_2"]
    df["frac_host_psf_2500"] = np.full(len(df), 5e-4)

    monkeypatch.setattr(hubble_utils, "read_quasars_from_hdf5_flat", lambda *_args, **_kwargs: df.copy())
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda frame: frame)
    _patch_load_agn_plotters(monkeypatch)

    filtered, _ = hubble_utils.load_agn_data(
        source_path,
        magnitude_convention="dereddened",
        spectra_fit_csv=None,
        lc_info_csv=None,
        cut_tier="2",
        spectra_sdss_run2d="26",
        plot_path=str(tmp_path / "figures"),
        cut_report_path=tmp_path / "cut_summary.txt",
    )

    assert filtered["object_id"].tolist() == ["b", "c"]


def test_load_agn_data_run2d_filter_bypassed_at_cut_tier_none(tmp_path, monkeypatch):
    source_path = tmp_path / "agn.h5"
    source_path.touch()
    df = _minimal_agn_frame(n=4)
    df["object_id"] = ["a", "b", "c", "d"]
    df["SDSS_RUN2D"] = ["v5_13_2", "26", "", None]
    df["log_tau_uv_rf"] = 0.0
    df["fracAGN_5100_fit"] = 0.0
    df["apparent_mag_2500"] = 30.0
    df["apparent_mag_2500_err"] = 5.0

    monkeypatch.setattr(hubble_utils, "read_quasars_from_hdf5_flat", lambda *_args, **_kwargs: df.copy())
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda frame: frame)
    _patch_load_agn_plotters(monkeypatch)

    filtered, _ = hubble_utils.load_agn_data(
        source_path,
        magnitude_convention="dereddened",
        spectra_fit_csv=None,
        lc_info_csv=None,
        cut_tier="none",
        spectra_sdss_run2d="v5_13_2",
        plot_path=str(tmp_path / "figures"),
        cut_report_path=tmp_path / "cut_summary.txt",
    )

    assert filtered["object_id"].tolist() == ["a", "b", "c", "d"]


def test_completeness_support_is_structural_at_cut_tier_none(tmp_path, monkeypatch):
    source_path = tmp_path / "agn.h5"
    source_path.touch()
    frame = _minimal_agn_frame(n=4)
    frame["object_id"] = ["bright", "lower-edge", "upper-edge", "faint"]
    values = [18.49, 18.5, 24.0, 24.01]
    frame["m_2500_dereddened"] = values
    frame["m_2500_attenuated_model"] = values

    monkeypatch.setattr(
        hubble_utils,
        "read_quasars_from_hdf5_flat",
        lambda *_args, **_kwargs: frame.copy(),
    )
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda value: value)
    _patch_load_agn_plotters(monkeypatch)

    enabled, _ = hubble_utils.load_agn_data(
        source_path,
        magnitude_convention="dereddened",
        completeness_magnitude="dereddened",
        enforce_completeness_support=True,
        cut_tier="none",
        plot_path=str(tmp_path / "enabled"),
        cut_report_path=tmp_path / "enabled.txt",
    )
    disabled, _ = hubble_utils.load_agn_data(
        source_path,
        magnitude_convention="dereddened",
        completeness_magnitude="dereddened",
        enforce_completeness_support=False,
        cut_tier="none",
        plot_path=str(tmp_path / "disabled"),
        cut_report_path=tmp_path / "disabled.txt",
    )

    assert enabled["object_id"].tolist() == ["lower-edge", "upper-edge"]
    assert disabled["object_id"].tolist() == [
        "bright", "lower-edge", "upper-edge", "faint"
    ]
    config = json.loads(enabled.attrs["cut_configuration_json"])
    assert config["completeness_support_enforced"] is True
    assert config["completeness_interpolation_policy"] == "strict-padded-v1"


def test_tier0_applies_science_magnitude_support_before_completeness_parent(
    tmp_path, monkeypatch
):
    source_path = tmp_path / "agn.h5"
    source_path.touch()
    frame = _minimal_agn_frame(n=7)
    frame["object_id"] = ["map-bright-oob", "pad-bright", "lower", "upper", "pad-faint", "map-faint-oob", "middle"]
    values = [17.9, 18.2, 18.5, 24.0, 24.3, 24.6, 21.0]
    frame["m_2500_dereddened"] = values
    frame["m_2500_attenuated_model"] = values

    monkeypatch.setattr(hubble_utils, "read_quasars_from_hdf5_flat", lambda *_args, **_kwargs: frame.copy())
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda value: value)
    _patch_load_agn_plotters(monkeypatch)

    analysis, _all, parent = hubble_utils.load_agn_data(
        source_path,
        magnitude_convention="dereddened",
        completeness_magnitude="dereddened",
        enforce_completeness_support=True,
        return_completeness_parent=True,
        cut_tier="0",
        plot_path=str(tmp_path / "figures"),
        cut_report_path=tmp_path / "cuts.txt",
    )

    assert parent["object_id"].tolist() == ["lower", "upper", "middle"]
    assert analysis["object_id"].tolist() == ["lower", "upper", "middle"]


def test_load_agn_data_target_selection_is_tier0_eligibility(
    tmp_path, monkeypatch
):
    source_path = tmp_path / "agn.h5"
    source_path.touch()
    df = _minimal_agn_frame(n=4)
    df["object_id"] = ["var", "var-core", "var-other", "boss-var"]
    df["SDSS_SURVEY"] = ["eBOSS", "eboss", "eboss", "boss"]
    df["SDSS_EBOSS_TARGET0"] = [0, 0, 0, 0]
    df["SDSS_EBOSS_TARGET1"] = [1 << 9, (1 << 9) | (1 << 10), (1 << 9) | (1 << 14), 1 << 9]
    df["SDSS_EBOSS_TARGET2"] = [0, 0, 0, 0]
    df["SDSS_SPECOBJ_MATCHED"] = [True, True, True, True]

    monkeypatch.setattr(
        hubble_utils,
        "read_quasars_from_hdf5_flat",
        lambda *_args, **_kwargs: df.copy(),
    )
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda frame: frame)
    _patch_load_agn_plotters(monkeypatch)

    filtered, parent = hubble_utils.load_agn_data(
        source_path,
        magnitude_convention="dereddened",
        spectra_fit_h5=None,
        lc_info_csv=None,
        cut_tier="0",
        sdss_target_selection="eboss-var-s82-only",
        plot_path=str(tmp_path / "figures"),
        cut_report_path=tmp_path / "cut_summary.txt",
    )

    assert filtered["object_id"].tolist() == ["var"]
    assert parent["object_id"].tolist() == ["var"]
    summary = (tmp_path / "cut_summary.txt").read_text(encoding="utf-8")
    assert "sample:sdss_target_selection" in summary
    assert "eboss-var-s82-only" in summary

def test_load_agn_data_run2d_filter_requires_sdss_run2d_column(tmp_path, monkeypatch):
    source_path = tmp_path / "agn.h5"
    source_path.touch()
    df = _minimal_agn_frame(n=3).drop(columns=["SDSS_RUN2D"])

    monkeypatch.setattr(hubble_utils, "read_quasars_from_hdf5_flat", lambda *_args, **_kwargs: df.copy())
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda frame: frame)
    _patch_load_agn_plotters(monkeypatch)

    with pytest.raises(ValueError, match="SDSS_RUN2D is not available"):
        hubble_utils.load_agn_data(
            source_path,
            magnitude_convention="dereddened",
            spectra_fit_csv=None,
            lc_info_csv=None,
            cut_tier="2",
            spectra_sdss_run2d="v5_13_2",
            plot_path=str(tmp_path / "figures"),
            cut_report_path=tmp_path / "cut_summary.txt",
        )


def test_load_agn_data_cut_tiers_apply_cumulatively(tmp_path, monkeypatch):
    source_path = tmp_path / "agn.h5"
    source_path.touch()
    frame = _minimal_agn_frame(n=4)
    frame["object_id"] = ["keep", "tier0", "tier1", "tier2"]
    frame.loc[1, ["m_2500_dereddened", "m_2500_attenuated_model"]] = 24.5
    frame.loc[2, "joint_reduced_chi2"] = 10.0
    frame.loc[3, "log_tau_uv_rf"] = 4.5

    monkeypatch.setattr(
        hubble_utils,
        "read_quasars_from_hdf5_flat",
        lambda *_args, **_kwargs: frame.copy(),
    )
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda value: value)
    _patch_load_agn_plotters(monkeypatch)

    expected = {
        "none": ["keep", "tier0", "tier1", "tier2"],
        "0": ["keep", "tier1", "tier2"],
        "1": ["keep", "tier2"],
        "2": ["keep"],
    }
    for cut_tier, expected_ids in expected.items():
        selected, _parent = hubble_utils.load_agn_data(
            source_path,
            magnitude_convention="dereddened",
            spectra_fit_h5=None,
            cut_tier=cut_tier,
            plot_diagnostics=False,
            plot_path=str(tmp_path / cut_tier),
            cut_report_path=tmp_path / cut_tier / "cut_summary.txt",
        )
        assert selected["object_id"].tolist() == expected_ids
        assert selected.attrs["cut_tier"] == cut_tier


def test_load_agn_data_defers_z_range_to_fit_selection(tmp_path, monkeypatch):
    source_path = tmp_path / "agn.h5"
    source_path.touch()
    frame = _minimal_agn_frame(n=4)
    frame["object_id"] = ["below", "inside", "above", "nonfinite"]
    frame["z"] = [0.2, 1.5, 3.5, np.nan]

    monkeypatch.setattr(
        hubble_utils,
        "read_quasars_from_hdf5_flat",
        lambda *_args, **_kwargs: frame.copy(),
    )
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda value: value)
    _patch_load_agn_plotters(monkeypatch)

    selected, parent = hubble_utils.load_agn_data(
        source_path,
        magnitude_convention="dereddened",
        spectra_fit_h5=None,
        cut_tier="0",
        z_range=(1.0, 3.16),
        plot_diagnostics=False,
        plot_path=str(tmp_path / "plots"),
        cut_report_path=tmp_path / "cut_summary.txt",
    )

    assert selected["object_id"].tolist() == ["below", "inside", "above"]
    assert parent["object_id"].tolist() == ["below", "inside", "above"]
    assert selected.loc[
        selected["z"].between(1.0, 3.16), "object_id"
    ].tolist() == ["inside"]
    assert "deferred to fit selection" in (
        tmp_path / "cut_summary.txt"
    ).read_text(encoding="utf-8")
    assert '"z_range_semantics":"fit_only_v1"' in selected.attrs[
        "cut_configuration_json"
    ]


def test_tier1_fails_when_a_mandatory_diagnostic_column_is_missing(
    tmp_path, monkeypatch
):
    source_path = tmp_path / "agn.h5"
    source_path.touch()
    frame = _minimal_agn_frame(n=2).drop(columns=["joint_reduced_chi2"])
    monkeypatch.setattr(
        hubble_utils,
        "read_quasars_from_hdf5_flat",
        lambda *_args, **_kwargs: frame.copy(),
    )
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda value: value)
    _patch_load_agn_plotters(monkeypatch)

    with pytest.raises(ValueError, match="Tier 1.*joint_reduced_chi2"):
        hubble_utils.load_agn_data(
            source_path,
            magnitude_convention="dereddened",
            spectra_fit_h5=None,
            cut_tier="1",
            plot_diagnostics=False,
        )


def test_tier2_excludes_only_joint_low_l2500_low_psf_host_region(
    tmp_path, monkeypatch
):
    source_path = tmp_path / "agn.h5"
    source_path.touch()
    frame = _minimal_agn_frame(n=2)
    frame["object_id"] = ["low-l-low-host", "high-l-low-host"]
    frame["z"] = 0.7
    frame["m_2500_dereddened"] = [21.0, 20.0]
    frame["m_2500_attenuated_model"] = [21.0, 20.0]
    frame["f_host_2500_psf"] = 0.05
    monkeypatch.setattr(
        hubble_utils,
        "read_quasars_from_hdf5_flat",
        lambda *_args, **_kwargs: frame.copy(),
    )
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda value: value)
    _patch_load_agn_plotters(monkeypatch)

    selected, _parent = hubble_utils.load_agn_data(
        source_path,
        magnitude_convention="dereddened",
        spectra_fit_h5=None,
        cut_tier="2",
        plot_diagnostics=False,
    )

    assert selected["object_id"].tolist() == ["high-l-low-host"]


def test_fast_vs_uv_diagnostic_skips_catalog_without_fast_timescale(tmp_path):
    frame = pd.DataFrame(
        {
            "z": [1.0],
            "log_tau_uv_rf": [2.5],
            "log_sigma_uv": [-0.5],
        }
    )

    with pytest.warns(
        RuntimeWarning,
        match="Skipping optional fast-vs-UV diagnostic plot.*log_tau_fast_uv",
    ):
        result = hubble_plotting.plot_fast_vs_uv_variability(
            frame,
            plot_path=str(tmp_path),
            show=False,
        )

    assert result is None
    assert not (tmp_path / "diagnostics" / "fast_vs_uv_variability.pdf").exists()
