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
            "f_host_2500": np.linspace(0.08, 0.25, n),
            "frac_host_psf_2500": np.linspace(0.05, 0.18, n),
            "frac_host_psf_2500_err": np.full(n, 0.02),
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
        "plot_sigma_uv_vs_variability_chi_sq_red_g",
        "plot_sigma_uv_vs_tau_uv_rf",
        "plot_sigma_uv_host_correction",
        "plot_suberlak_style_sigma_tau_fits",
        "plot_tau_sigma_vs_wu_catalog",
        "plot_tau_sigma_vs_redshift",
        "plot_f_host_2500_vs_l2500",
        "plot_blr_diagnostics_summary",
    )
    for name in plot_noops:
        monkeypatch.setattr(hubble_plotting, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hubble_utils, "plot_sigma_tau_identity_grid", lambda *_args, **_kwargs: None)


def test_load_agn_data_makes_precut_and_postcut_fhost_and_blr_plots(tmp_path, monkeypatch):
    source_path = tmp_path / "agn.h5"
    source_path.touch()
    monkeypatch.setattr(
        hubble_utils,
        "read_quasars_from_hdf5_flat",
        lambda *_args, **_kwargs: _minimal_agn_frame(),
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
        "plot_sigma_uv_vs_variability_chi_sq_red_g",
        "plot_sigma_uv_vs_tau_uv_rf",
        "plot_sigma_uv_host_correction",
        "plot_suberlak_style_sigma_tau_fits",
        "plot_tau_sigma_vs_wu_catalog",
        "plot_tau_sigma_vs_redshift",
    )
    for name in plot_noops:
        monkeypatch.setattr(hubble_plotting, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hubble_plotting, "plot_f_host_2500_vs_l2500", capture_plot)
    monkeypatch.setattr(hubble_plotting, "plot_blr_diagnostics_summary", capture_plot)

    hubble_utils.load_agn_data(
        source_path,
        spectra_fit_csv=None,
        lc_info_csv=None,
        apply_cut=True,
        plot_path=str(tmp_path / "figures"),
        cut_report_path=tmp_path / "cut_summary.txt",
    )

    captured_by_filename = {call.get("filename"): call for call in captured_calls}
    assert "f_host_2500_vs_l2500_precut.pdf" in captured_by_filename
    assert "f_host_2500_vs_l2500_postcut.pdf" in captured_by_filename
    assert "blr_precut.pdf" in captured_by_filename
    assert "blr_postcut.pdf" in captured_by_filename
    assert captured_by_filename["f_host_2500_vs_l2500_precut.pdf"]["f_host_col"] == "f_host_2500_psf"
    assert captured_by_filename["f_host_2500_vs_l2500_postcut.pdf"]["f_host_col"] == "f_host_2500_psf"


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
        spectra_fit_csv=None,
        lc_info_csv=None,
        apply_cut=True,
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
    df = _minimal_agn_frame(n=12)

    out = hubble_plotting.plot_blr_diagnostics_summary(
        df,
        plot_path=str(tmp_path / "figures"),
        show=False,
    )

    assert out is not None
    assert os.path.exists(out)
    assert out.endswith(os.path.join("diagnostics", "blr.pdf"))


def test_plot_blr_diagnostics_summary_returns_none_when_columns_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    df = _minimal_agn_frame(n=8).drop(columns=["log_sigma0"])

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

    monkeypatch.setattr(hubble_utils, "read_quasars_from_hdf5_flat", lambda *_args, **_kwargs: df.copy())
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda frame: frame)
    _patch_load_agn_plotters(monkeypatch)

    filtered, _ = hubble_utils.load_agn_data(
        source_path,
        spectra_fit_csv=None,
        lc_info_csv=None,
        apply_cut=True,
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

    monkeypatch.setattr(hubble_utils, "read_quasars_from_hdf5_flat", lambda *_args, **_kwargs: df.copy())
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda frame: frame)
    _patch_load_agn_plotters(monkeypatch)

    filtered, _ = hubble_utils.load_agn_data(
        source_path,
        spectra_fit_csv=None,
        lc_info_csv=None,
        apply_cut=True,
        spectra_sdss_run2d="26",
        plot_path=str(tmp_path / "figures"),
        cut_report_path=tmp_path / "cut_summary.txt",
    )

    assert filtered["object_id"].tolist() == ["b", "c"]


def test_load_agn_data_run2d_filter_bypassed_when_no_cuts(tmp_path, monkeypatch):
    source_path = tmp_path / "agn.h5"
    source_path.touch()
    df = _minimal_agn_frame(n=4)
    df["object_id"] = ["a", "b", "c", "d"]
    df["SDSS_RUN2D"] = ["v5_13_2", "26", "", None]

    monkeypatch.setattr(hubble_utils, "read_quasars_from_hdf5_flat", lambda *_args, **_kwargs: df.copy())
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda frame: frame)
    _patch_load_agn_plotters(monkeypatch)

    filtered, _ = hubble_utils.load_agn_data(
        source_path,
        spectra_fit_csv=None,
        lc_info_csv=None,
        apply_cut=False,
        spectra_sdss_run2d="v5_13_2",
        plot_path=str(tmp_path / "figures"),
        cut_report_path=tmp_path / "cut_summary.txt",
    )

    assert filtered["object_id"].tolist() == ["a", "b", "c", "d"]


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
            spectra_fit_csv=None,
            lc_info_csv=None,
            apply_cut=True,
            spectra_sdss_run2d="v5_13_2",
            plot_path=str(tmp_path / "figures"),
            cut_report_path=tmp_path / "cut_summary.txt",
        )
