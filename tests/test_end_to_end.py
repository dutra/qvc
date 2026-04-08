import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import jax.numpy as jnp
import pytest
import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
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
    build_explicit_model_params_fluxmix_fast,
    build_single_object_model,
    build_single_object_model_continuum_only,
    build_single_object_model_mag_flux_linearized,
    build_single_object_model_mag_fluxmix_stage2,
    build_mag_fluxmix_fast_display_model,
    compute_flux_line_ratio_offsets,
    compute_lomb_scargle_break_diagnostics,
    compute_g_band_residual_drift_diagnostics,
    compute_g_band_raw_drift_diagnostics,
    compute_object_adf_diagnostics,
    _fluxmix_stage1_raw_median_params,
    make_lc,
    run_two_stage_fluxmix_fast_inference,
)
from qvc.light_curve.multiband_fit_plotting import (
    plot_all_histograms,
    save_combined_plot,
    save_dm_df_over_f_distribution_plot,
)
from qvc.light_curve.multiband_fit_utils import (
    flatten_flat_samples_per_band,
    flatten_per_chain_samples_per_band,
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


def _horizontal_dashed_levels(ax):
    levels = []
    for line in ax.lines:
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if line.get_linestyle() != "--" or ydata.size == 0 or not np.all(np.isfinite(ydata)):
            continue
        if np.allclose(ydata, ydata[0]):
            levels.append(float(ydata[0]))
    return levels


def _has_cut_errorbar_overlay(ax):
    return any(
        isinstance(container, ErrorbarContainer) and container.get_label() == "cut"
        for container in ax.containers
    )


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
            "z": [0.7, 0.9, 1.0, 1.1, 1.3],
            "apparent_mag_2500": [22.4, 22.4, 22.1, 21.7, 21.2],
            "g_raw_mean_slope": [-2e-4, -1e-4, 0.0, 1e-4, 2e-4],
            "g_resid_mean_slope": [-8e-5, -2e-5, 0.0, 2e-5, 8e-5],
            "g_raw_var_slope": [-5e-6, -1e-6, 0.0, 1e-6, 5e-6],
            "g_resid_var_slope": [-2e-6, -5e-7, 0.0, 5e-7, 2e-6],
        }
    )

    out_mean = hubble_plotting.plot_g_band_drift_slope_histograms(
        df,
        slope_kind="mean",
        z_min=0.8,
        z_max=1.2,
        m2500_max=22.5,
        plot_path=str(tmp_path / "figures"),
        show=False,
        filename="g_band_mean_slope_histograms_postcut_z0p8to1p2_m2500lt22p5.pdf",
    )
    out_var = hubble_plotting.plot_g_band_drift_slope_histograms(
        df,
        slope_kind="var",
        z_min=0.8,
        z_max=1.2,
        plot_path=str(tmp_path / "figures"),
        show=False,
        filename="g_band_var_slope_histograms_postcut_z0p8to1p2.pdf",
    )

    assert out_mean is not None
    assert out_var is not None
    assert os.path.exists(out_mean)
    assert os.path.exists(out_var)
    assert out_mean.endswith("g_band_mean_slope_histograms_postcut_z0p8to1p2_m2500lt22p5.pdf")
    assert out_var.endswith("g_band_var_slope_histograms_postcut_z0p8to1p2.pdf")


def test_plot_spectral_fraction_vs_redshift_requires_f_host_2500(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    df = pd.DataFrame(
        {
            "z": np.linspace(0.3, 2.2, 24),
            "f_bc_3000": np.linspace(0.05, 0.25, 24),
            "f_fe_uv_3000": np.linspace(0.1, 0.4, 24),
            "f_na": np.linspace(0.02, 0.12, 24),
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

    assert out is None


def test_plot_spectral_fraction_vs_redshift_writes_pdf_with_f_host_2500_only(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    df = pd.DataFrame(
        {
            "z": np.linspace(0.3, 2.2, 24),
            "f_bc_3000": np.linspace(0.05, 0.25, 24),
            "f_fe_uv_3000": np.linspace(0.1, 0.4, 24),
            "f_host_2500": np.linspace(0.25, 0.01, 24),
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


def test_plot_spectral_fraction_vs_redshift_writes_pdf_with_both_host_fractions(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    df = pd.DataFrame(
        {
            "z": np.linspace(0.3, 2.2, 24),
            "f_bc_3000": np.linspace(0.05, 0.25, 24),
            "f_fe_uv_3000": np.linspace(0.1, 0.4, 24),
            "f_host_center": np.linspace(0.3, 0.02, 24),
            "f_host_2500": np.linspace(0.25, 0.01, 24),
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


def test_plot_spectral_fraction_vs_redshift_draws_dashed_cut_thresholds(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    captured = {}
    original_save_figure = hubble_plotting._save_figure

    def _capture_save_figure(fig, path, **kwargs):
        captured["axes"] = list(fig.axes)
        captured["path"] = path
        return original_save_figure(fig, path, **kwargs)

    monkeypatch.setattr(hubble_plotting, "_save_figure", _capture_save_figure)

    df = pd.DataFrame(
        {
            "z": np.linspace(0.3, 2.2, 24),
            "f_bc_3000": np.linspace(0.05, 0.25, 24),
            "f_fe_uv_3000": np.linspace(0.1, 0.4, 24),
            "f_host_2500": np.linspace(0.25, 0.01, 24),
        }
    )
    cut_thresholds = {"f_bc_3000": 0.2, "f_fe_uv_3000": 0.3, "f_host_2500": 0.15}

    out = hubble_plotting.plot_spectral_fraction_vs_redshift(
        df,
        plot_path=str(tmp_path / "figures"),
        show=False,
        nbins=6,
        min_bin_count=3,
        cut_thresholds=cut_thresholds,
    )

    assert out is not None
    assert os.path.exists(out)
    assert captured["path"].endswith("spectral_fraction_vs_redshift.pdf")
    assert len(captured["axes"]) == 3
    for ax, expected in zip(captured["axes"], (0.2, 0.3, 0.15)):
        assert any(np.isclose(level, expected) for level in _horizontal_dashed_levels(ax))


def test_plot_spectral_fraction_vs_redshift_cut_overlay_keeps_thresholds_and_skips_f_lines(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    captured = {}
    original_save_figure = hubble_plotting._save_figure

    def _capture_save_figure(fig, path, **kwargs):
        captured["axes"] = list(fig.axes)
        captured["path"] = path
        return original_save_figure(fig, path, **kwargs)

    monkeypatch.setattr(hubble_plotting, "_save_figure", _capture_save_figure)

    df = pd.DataFrame(
        {
            "z": np.linspace(0.3, 2.2, 24),
            "f_bc_3000": np.linspace(0.05, 0.25, 24),
            "f_fe_uv_3000": np.linspace(0.1, 0.4, 24),
            "f_na": np.linspace(0.02, 0.12, 24),
            "f_host_2500": np.linspace(0.25, 0.01, 24),
        }
    )
    df_cut_sources = pd.DataFrame(
        {
            "z": np.linspace(0.4, 2.0, 6),
            "f_bc_3000": np.linspace(0.12, 0.22, 6),
            "f_fe_uv_3000": np.linspace(0.18, 0.35, 6),
            "f_na": np.linspace(0.03, 0.09, 6),
            "f_host_2500": np.linspace(0.2, 0.04, 6),
        }
    )
    cut_thresholds = {"f_bc_3000": 0.2, "f_fe_uv_3000": 0.3, "f_host_2500": 0.15}

    out = hubble_plotting.plot_spectral_fraction_vs_redshift(
        df,
        plot_path=str(tmp_path / "figures"),
        show=False,
        nbins=6,
        min_bin_count=3,
        df_cut_sources=df_cut_sources,
        filename="spectral_fraction_vs_redshift_cuts.pdf",
        cut_thresholds=cut_thresholds,
    )

    assert out is not None
    assert os.path.exists(out)
    assert captured["path"].endswith("spectral_fraction_vs_redshift_cuts.pdf")
    assert len(captured["axes"]) == 4
    assert any(np.isclose(level, 0.2) for level in _horizontal_dashed_levels(captured["axes"][0]))
    assert any(np.isclose(level, 0.3) for level in _horizontal_dashed_levels(captured["axes"][1]))
    assert _horizontal_dashed_levels(captured["axes"][2]) == []
    assert any(np.isclose(level, 0.15) for level in _horizontal_dashed_levels(captured["axes"][3]))
    assert all(_has_cut_errorbar_overlay(ax) for ax in captured["axes"])


def test_plot_spectral_fraction_vs_redshift_ignores_f_pl_panel(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    captured = {}
    original_save_figure = hubble_plotting._save_figure

    def _capture_save_figure(fig, path, **kwargs):
        captured["n_axes"] = len(fig.axes)
        captured["path"] = path
        return original_save_figure(fig, path, **kwargs)

    monkeypatch.setattr(hubble_plotting, "_save_figure", _capture_save_figure)

    df = pd.DataFrame(
        {
            "z": np.linspace(0.3, 2.2, 24),
            "f_bc_3000": np.linspace(0.05, 0.25, 24),
            "f_fe_uv_3000": np.linspace(0.1, 0.4, 24),
            "f_na": np.linspace(0.02, 0.12, 24),
            "f_host_2500": np.linspace(0.25, 0.01, 24),
            "f_PL": np.linspace(0.15, 0.35, 24),
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
    assert captured["path"].endswith("spectral_fraction_vs_redshift.pdf")
    assert captured["n_axes"] == 4


def test_plot_sigma_bc_vs_redshift_writes_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    df = pd.DataFrame(
        {
            "z": np.linspace(0.4, 2.4, 24),
            "log_sigma_uv": np.linspace(-1.0, -0.3, 24),
            "log_amp_delta_bc": np.linspace(-0.5, -0.2, 24),
        }
    )

    out = hubble_plotting.plot_sigma_bc_vs_redshift(
        df,
        plot_path=str(tmp_path / "figures"),
        show=False,
        filename="sigma_bc_vs_redshift_postcut.pdf",
    )

    assert out is not None
    assert os.path.exists(out)
    assert out.endswith("sigma_bc_vs_redshift_postcut.pdf")


def test_plot_f_host_2500_vs_redshift_writes_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    df = pd.DataFrame(
        {
            "z": np.linspace(0.4, 2.4, 20),
            "f_host_2500": np.linspace(0.22, 0.01, 20),
        }
    )

    out = hubble_plotting.plot_f_host_2500_vs_redshift(
        df,
        plot_path=str(tmp_path / "figures"),
        show=False,
        nbins=5,
        min_bin_count=3,
    )

    assert out is not None
    assert os.path.exists(out)
    assert out.endswith("f_host_2500_vs_redshift.pdf")


def test_plot_m2500_vs_z_colorpanels_supports_both_host_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    df = pd.DataFrame(
        {
            "object_id": [f"obj-{i}" for i in range(20)],
            "z": np.linspace(0.4, 2.4, 20),
            "apparent_mag_2500": np.linspace(20.5, 23.0, 20),
            "f_host_center": np.linspace(0.30, 0.02, 20),
            "f_host_2500": np.linspace(0.24, 0.01, 20),
            "f_bc_3000": np.linspace(0.05, 0.25, 20),
            "wrms": np.linspace(0.8, 1.4, 20),
        }
    )

    result = hubble_plotting.plot_m2500_vs_z_colorpanels(
        df,
        df_keep=df.iloc[:12].copy(),
        thin=1,
    )

    assert result is not None


def test_plot_sigma_bc_vs_frac_bc_writes_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    df = pd.DataFrame(
        {
            "z": np.linspace(0.4, 2.4, 24),
            "f_bc_3000": np.linspace(0.05, 0.35, 24),
            "log_sigma_uv": np.linspace(-1.0, -0.3, 24),
            "log_amp_delta_bc": np.linspace(-0.5, -0.2, 24),
        }
    )

    out = hubble_plotting.plot_sigma_bc_vs_frac_bc(
        df,
        plot_path=str(tmp_path / "figures"),
        show=False,
        filename="sigma_bc_vs_frac_bc_postcut.pdf",
    )

    assert out is not None
    assert os.path.exists(out)
    assert out.endswith("sigma_bc_vs_frac_bc_postcut.pdf")


def test_plot_bc_lag_vs_l2500_writes_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    df = pd.DataFrame(
        {
            "z": np.linspace(0.5, 2.0, 24),
            "apparent_mag_2500": np.linspace(20.0, 22.5, 24),
            "log_lag_bc_g_RF": np.linspace(0.8, 1.2, 24),
            "log_lag_bc_r_RF": np.linspace(0.82, 1.18, 24),
        }
    )

    out = hubble_plotting.plot_bc_lag_vs_l2500(
        df,
        plot_path=str(tmp_path / "figures"),
        show=False,
        filename="bc_lag_vs_l2500_postcut.pdf",
    )

    assert out is not None
    assert os.path.exists(out)
    assert out.endswith("bc_lag_vs_l2500_postcut.pdf")


def test_plot_residuals_vs_alphaOX_writes_both_pdfs(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    df = pd.DataFrame(
        {
            "z": [0.35, 0.6, 1.0, 1.8, 3.3],
            "delta_alphaOX": [-0.15, -0.08, 0.0, 0.06, 0.12],
            "delta_alphaOX_err": [0.03, 0.02, 0.02, 0.03, 0.04],
            "alphaOX": [-1.9, -1.8, -1.7, -1.6, -1.5],
            "alphaOX_err": [0.05, 0.04, 0.04, 0.05, 0.06],
        }
    )
    residuals = np.array([0.18, 0.07, 0.01, -0.05, -0.12], dtype=float)
    residuals_err = np.full_like(residuals, 0.08)

    delta_path, alpha_path = hubble_plotting.plot_residuals_vs_alphaOX(
        df,
        residuals,
        residuals_err,
        plot_path=str(tmp_path / "figures"),
        show=False,
    )

    assert delta_path is not None
    assert alpha_path is not None
    assert os.path.exists(delta_path)
    assert os.path.exists(alpha_path)
    assert delta_path.endswith("delta_alphaOX_residuals.pdf")
    assert alpha_path.endswith("alphaOX_residuals.pdf")


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


def test_plot_blr_line_lags_vs_l2500_fiducial_filters_negative_and_prior_like_lags(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    plot_path = str(tmp_path / "figures")
    df = pd.DataFrame(
        {
            "object_id": ["keep", "neg", "prior"],
            "z": [1.2, 1.2, 1.2],
            "apparent_mag_2500": [20.5, 20.2, 20.1],
            "log_sigma_uv": [-0.7, -0.8, -0.9],
            "dropped_bands": [[], [], []],
            "log_amp_delta_blr_g": [0.0, 0.0, 0.0],
            "log_lag_blr_g_RF": [1.15, -0.10, 1.20],
            "log_lag_blr_g_RF_err": [0.1, 0.1, 0.1],
            "log_lag_blr_g_kl": [0.20, 0.20, 0.01],
        }
    )

    out = hubble_plotting.plot_blr_line_lags_vs_l2500_fiducial(
        df,
        plot_path=plot_path,
        show=False,
        prob_thresh=0.0,
    )

    selected_csv = os.path.join(plot_path, "diagnostics", "blr_line_assignment_selected_fiducial.csv")
    selected = pd.read_csv(selected_csv)

    assert out is not None
    assert os.path.exists(out)
    assert selected["object_id"].tolist() == ["keep"]


def test_plot_blr_lag_line_panel_adds_shen_relation_for_supported_lines():
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    hb_df = pd.DataFrame(
        {
            "component": [1],
            "log_line_luminosity": [44.5],
            "log_line_luminosity_err": [0.1],
            "log_lag_rf": [1.6],
            "log_lag_rf_err": [0.1],
        }
    )

    hubble_plotting._plot_blr_lag_line_panel(axes[0], hb_df, "Hβ")
    hubble_plotting._plot_blr_lag_line_panel(axes[1], hb_df, "Hα")

    shen_lines_hb = [line for line in axes[0].lines if line.get_label() == "Shen et al. (2024)"]
    shen_lines_ha = [line for line in axes[1].lines if line.get_label() == "Shen et al. (2024)"]

    assert len(shen_lines_hb) == 1
    assert len(shen_lines_ha) == 0
    x = np.asarray(shen_lines_hb[0].get_xdata(), dtype=float)
    y = np.asarray(shen_lines_hb[0].get_ydata(), dtype=float)
    assert np.isclose(x[0], 42.87)
    assert np.isclose(x[-1], 45.40)
    np.testing.assert_allclose(y, 1.458 + 0.41 * (x - 44.0))

    plt.close(fig)


def test_plot_l2500_vs_uv_variability_fiducial_writes_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    df = pd.DataFrame(
        {
            "object_id": ["a", "b", "c", "d"],
            "z": [0.3, 0.7, 1.1, 1.8],
            "apparent_mag_2500": [19.1, 19.8, 20.6, 21.4],
            "log_sigma_uv": [-1.05, -0.92, -0.84, -0.73],
            "log_tau_uv_rf": [2.10, 2.35, 2.62, 2.88],
        }
    )

    out = hubble_plotting.plot_l2500_vs_uv_variability_fiducial(
        df,
        plot_path=str(tmp_path / "figures"),
        show=False,
    )

    assert out is not None
    assert os.path.exists(out)
    assert out.endswith("l2500_vs_uv_variability_fiducial.pdf")


def test_plot_sf_vs_uv_variability_writes_custom_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    df = pd.DataFrame(
        {
            "object_id": ["a", "b", "c", "d"],
            "z": [0.3, 0.7, 1.1, 1.8],
            "log_sigma_uv": [-1.05, -0.92, -0.84, -0.73],
            "log_sigma_uv_sf": [-1.00, -0.90, -0.80, -0.70],
            "log_tau_uv_rf": [2.10, 2.35, 2.62, 2.88],
            "log_tau_uv_rf_sf": [2.05, 2.30, 2.58, 2.80],
            "variability_chi_sq_g": [8.0, 12.0, 20.0, 35.0],
            "sf_valid": [True, True, True, True],
        }
    )

    out = hubble_plotting.plot_sf_vs_uv_variability(
        df,
        plot_path=str(tmp_path / "figures"),
        show=False,
        filename="sf_vs_uv_variability_precut.pdf",
    )

    assert out is not None
    assert os.path.exists(out)
    assert out.endswith("sf_vs_uv_variability_precut.pdf")


def test_plot_sf_ref_band_vs_model_g_writes_custom_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    df = pd.DataFrame(
        {
            "object_id": ["a", "b", "c", "d"],
            "log_sigma_band_g": [-1.05, -0.92, -0.84, -0.73],
            "log_sigma_rms_band_g": [-0.99, -0.88, -0.79, -0.69],
            "log_sigma_sf_ref_band": [-1.00, -0.90, -0.80, -0.70],
            "log_tau_band_g_RF": [2.10, 2.35, 2.62, 2.88],
            "log_tau_sf_model_ref_band": [2.02, 2.28, 2.55, 2.77],
            "log_tau_sf_ref_band": [2.05, 2.30, 2.58, 2.80],
            "variability_chi_sq_g": [8.0, 12.0, 20.0, 35.0],
            "sf_valid": [True, True, True, True],
            "sf_ref_band": ["g", "g", "g", "g"],
        }
    )

    out = hubble_plotting.plot_sf_ref_band_vs_model_g(
        df,
        plot_path=str(tmp_path / "figures"),
        show=False,
        filename="sf_ref_band_vs_model_g_precut.pdf",
    )

    assert out is not None
    assert os.path.exists(out)
    assert out.endswith("sf_ref_band_vs_model_g_precut.pdf")


def test_plot_redshift_bin_residual_summary_writes_pdf_and_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    df = pd.DataFrame({"z": [0.4, 0.55, 0.85, 1.0, 1.35, 1.7, 2.2, 2.7]})
    residuals_biased = np.array([0.08, 0.05, 0.04, 0.02, 0.01, -0.01, -0.03, -0.04], dtype=float)
    residuals_debiased = np.array([0.03, 0.02, 0.01, 0.00, -0.01, -0.01, -0.02, -0.02], dtype=float)
    residuals_biased_err = np.full_like(residuals_biased, 0.1)
    residuals_debiased_err = np.full_like(residuals_debiased, 0.1)

    out = hubble_plotting.plot_redshift_bin_residual_summary(
        df,
        residuals_biased,
        residuals_biased_err,
        residuals_debiased,
        residuals_debiased_err,
        plot_path=str(tmp_path / "figures"),
        show=False,
        z_bins=[0.3, 0.9, 1.6, 3.2],
    )

    csv_path = tmp_path / "figures" / "diagnostics" / "redshift_bin_residual_summary.csv"

    assert out is not None
    assert os.path.exists(out)
    assert out.endswith("redshift_bin_residual_summary.pdf")
    assert csv_path.exists()

    summary = pd.read_csv(csv_path)
    assert {"mean_biased", "mean_err_biased", "mean_debiased", "mean_err_debiased"}.issubset(summary.columns)
    assert np.isfinite(summary["mean_debiased"]).any()


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


def test_blr_line_assignment_uses_fit_spectra_line_specific_luminosities():
    df = pd.DataFrame(
        {
            "object_id": ["civ", "mgii", "hb", "ha"],
            "z": [2.0, 0.8, 0.0, 0.0],
            "dropped_bands": [[], [], [], []],
            "log_sigma_uv": [-1.0, -1.0, -1.0, -1.0],
            "log_amp_delta_blr_g": [0.0, 0.0, 0.0, np.nan],
            "log_lag_blr_g_RF": [1.0, 1.1, 1.2, np.nan],
            "log_lag_blr_g_RF_err": [0.1, 0.1, 0.1, np.nan],
            "log_amp_delta_blr_r": [np.nan, np.nan, np.nan, 0.0],
            "log_lag_blr_r_RF": [np.nan, np.nan, np.nan, 1.3],
            "log_lag_blr_r_RF_err": [np.nan, np.nan, np.nan, 0.2],
            "log_lambda_Llambda_1350_agn": [45.1, 45.2, 45.3, 45.4],
            "log_lambda_Llambda_1350_agn_err": [0.01, 0.02, 0.03, 0.04],
            "log_lambda_Llambda_3000_agn": [46.1, 46.2, 46.3, 46.4],
            "log_lambda_Llambda_3000_agn_err": [0.05, 0.06, 0.07, 0.08],
            "log_lambda_Llambda_5100_agn": [47.1, 47.2, 47.3, 47.4],
            "log_lambda_Llambda_5100_agn_err": [0.09, 0.10, 0.11, 0.12],
        }
    )

    out = hubble_plotting._blr_line_assignment_longform(
        df,
        np.array([99.0, 99.0, 99.0, 99.0], dtype=float),
    )

    expected = {
        "civ": ("C IV", "log_lambda_Llambda_1350_agn", 45.1, 0.01),
        "mgii": ("Mg II", "log_lambda_Llambda_3000_agn", 46.2, 0.06),
        "hb": ("Hβ", "log_lambda_Llambda_5100_agn", 47.3, 0.11),
        "ha": ("Hα", "log_lambda_Llambda_5100_agn", 47.4, 0.12),
    }

    assert len(out) == 4
    for row in out.itertuples():
        line_name, lum_col, log_lum, log_lum_err = expected[row.object_id]
        assert row.assigned_line == line_name
        assert row.line_luminosity_col == lum_col
        assert np.isclose(row.log_line_luminosity, log_lum)
        assert np.isclose(row.log_line_luminosity_err, log_lum_err)


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
    assert "log_amp_ratio_blr_raw" in model_trace
    assert "delta_log_lag_blr_raw" in model_trace
    assert "log_amp_delta_bc" in model_trace
    assert "log_lag_ratio_bc_to_blr" in model_trace
    assert "log_amp_ratio_blr2_raw" not in model_trace
    assert "delta_log_lag_blr2_raw" not in model_trace
    assert np.allclose(np.asarray(model_trace["log_amp_delta_blr2"]["value"]), -9.0)
    assert np.allclose(np.asarray(model_trace["log_lag_blr2"]["value"]), -9.0)


def test_build_single_object_model_mag_flux_linearized_smoke():
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
    numpyro_model = build_single_object_model_mag_flux_linearized(
        obj,
        lam_rf,
        log_jitter_mean=jnp.array(log_jitter_mean),
        disable_poly1=False,
        disable_lag_blr=False,
        disable_lag_bc=False,
        drop_band_lyman_alpha=False,
        tau_fast_truncated=False,
        n_blr_terms=1,
    )

    nuts = NUTS(
        numpyro_model,
        dense_mass=False,
        max_tree_depth=1,
        target_accept_prob=0.8,
    )
    mcmc = MCMC(
        nuts,
        num_warmup=1,
        num_samples=1,
        num_chains=1,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(4))

    samples_flat = mcmc.get_samples(group_by_chain=False)
    samples_flat = tree_map(lambda x: np.asarray(device_get(x)), samples_flat)
    assert "amp_cont" in samples_flat
    assert "amp_cont_relflux" in samples_flat
    assert "lag_blr" in samples_flat
    assert np.all(np.isfinite(samples_flat["log_sigma_uv"]))
    assert np.all(np.isfinite(samples_flat["log_sigma_uv_relflux"]))
    assert np.all(np.isfinite(samples_flat["F0_cont_band"]))
    assert np.all(np.isfinite(samples_flat["amp_cont_relflux"]))


def test_build_single_object_model_mag_flux_linearized_rejects_second_blr_term():
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

    with pytest.raises(ValueError, match=r"mag_flux_linearized.*n_blr_terms=1"):
        build_single_object_model_mag_flux_linearized(
            obj,
            lam_rf,
            log_jitter_mean=jnp.array(log_jitter_mean),
            disable_poly1=False,
            disable_lag_blr=False,
            disable_lag_bc=False,
            drop_band_lyman_alpha=False,
            tau_fast_truncated=False,
            n_blr_terms=2,
        )


def test_build_single_object_model_continuum_only_smoke():
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
    numpyro_model = build_single_object_model_continuum_only(
        obj,
        lam_rf,
        log_jitter_mean=jnp.array(log_jitter_mean),
        disable_poly1=False,
        drop_band_lyman_alpha=False,
        tau_fast_truncated=False,
    )

    model_trace = trace(seed(numpyro_model, random.PRNGKey(0))).get_trace()
    assert "log_amp_ratio_blr_raw" not in model_trace
    assert "log_amp_ratio_bc" not in model_trace

    nuts = NUTS(
        numpyro_model,
        dense_mass=False,
        max_tree_depth=1,
        target_accept_prob=0.8,
    )
    mcmc = MCMC(
        nuts,
        num_warmup=1,
        num_samples=1,
        num_chains=1,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(8))
    samples_flat = tree_map(lambda x: np.asarray(device_get(x)), mcmc.get_samples(group_by_chain=False))
    assert np.allclose(samples_flat["amp_blr"], 0.0)
    assert np.allclose(samples_flat["amp_bc"], 0.0)
    assert np.all(np.isfinite(samples_flat["amp_cont"]))


def test_run_two_stage_fluxmix_fast_inference_smoke():
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

    samples_flat, samples_per_chain, model, diagnostics = run_two_stage_fluxmix_fast_inference(
        obj,
        lam_rf,
        log_jitter_mean=jnp.array(log_jitter_mean),
        rng_key=random.PRNGKey(9),
        num_warmup=1,
        num_samples=1,
        num_chains=1,
        chain_method="sequential",
        progress_bar=False,
        dense_mass=False,
        max_tree_depth=1,
        disable_poly1=False,
        disable_lag_blr=False,
        disable_lag_bc=False,
        drop_band_lyman_alpha=False,
        tau_fast_truncated=False,
        n_blr_terms=1,
    )

    assert "log_cont_scale" in samples_flat
    assert "amp_cont" in samples_flat
    assert "amp_blr" in samples_flat
    assert "lag_blr" in samples_flat
    assert np.all(np.isfinite(samples_flat["log_sigma_uv"]))
    assert np.all(np.isfinite(samples_flat["amp_cont"]))
    assert diagnostics["stage2_conditioned_on_median_basis"] is True

    flat_per_band = flatten_flat_samples_per_band(samples_flat, bands)
    per_chain_per_band = flatten_per_chain_samples_per_band(samples_per_chain, bands)
    assert f"amp_cont_{bands[0]}" in flat_per_band
    assert f"amp_blr_{bands[0]}" in per_chain_per_band

    posterior_median = {k: np.median(v, axis=0) for k, v in samples_flat.items()}
    query_t = jnp.array([-5000.0, 0.0, float(np.max(np.asarray(obj["X"][0], dtype=float))) + 500.0], dtype=float)
    query_b = jnp.array([0, 0, 1], dtype=int)
    pred_mu, pred_std = model.pred(posterior_median, (query_t, query_b))
    assert np.all(np.isfinite(np.asarray(pred_mu)))
    assert np.all(np.isfinite(np.asarray(pred_std)))

    rebuilt_model = build_mag_fluxmix_fast_display_model(obj, lam_rf, samples_flat)
    rebuilt_mu, rebuilt_std = rebuilt_model.pred(posterior_median, (query_t, query_b))
    assert np.all(np.isfinite(np.asarray(rebuilt_mu)))
    assert np.all(np.isfinite(np.asarray(rebuilt_std)))

    psd_diag = compute_lomb_scargle_break_diagnostics(
        rebuilt_model,
        samples_flat,
        obj,
        float(obj["z"]),
        n_freq=32,
    )
    assert "log_sigma_uv_bpl" in psd_diag
    assert "log_sigma_bpl_ref_band" in psd_diag


def test_run_alternating_two_stage_fluxmix_fast_inference_smoke():
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

    samples_flat, samples_per_chain, model, diagnostics = run_two_stage_fluxmix_fast_inference(
        obj,
        lam_rf,
        log_jitter_mean=jnp.array(log_jitter_mean),
        rng_key=random.PRNGKey(11),
        num_warmup=1,
        num_samples=1,
        num_chains=1,
        chain_method="sequential",
        progress_bar=False,
        dense_mass=False,
        max_tree_depth=1,
        disable_poly1=False,
        disable_lag_blr=False,
        disable_lag_bc=False,
        drop_band_lyman_alpha=False,
        tau_fast_truncated=False,
        n_blr_terms=1,
        outer_iters=2,
    )

    assert np.all(np.isfinite(samples_flat["log_sigma_uv"]))
    assert np.all(np.isfinite(samples_flat["amp_blr"]))
    assert diagnostics["fluxmix_outer_iters"] == 2
    assert "stage1_iter2_accept_prob" in diagnostics
    assert "stage2_iter2_accept_prob" in diagnostics
    assert "stage1_iter2_line_subtracted_rms" in diagnostics

    posterior_median = {k: np.median(v, axis=0) for k, v in samples_flat.items()}
    pred_mu, pred_std = model.pred(posterior_median, obj["X"])
    assert np.all(np.isfinite(np.asarray(pred_mu)))
    assert np.all(np.isfinite(np.asarray(pred_std)))
    assert f"amp_cont_{bands[0]}" in flatten_per_chain_samples_per_band(samples_per_chain, bands)


def test_fluxmix_stage2_eta_sigma_recomputes_continuum_and_line_ratios():
    lam_rf = jnp.array([2000.0, 3500.0], dtype=float)
    lambda_center_rf = jnp.exp(jnp.mean(jnp.log(lam_rf)))
    raw_base = dict(
        log_tau_slow_center0=jnp.log(100.0),
        log_tau_fast_center0=jnp.log(10.0),
        log_sigma_center0=jnp.log(0.1),
        lambda_center_rf=lambda_center_rf,
        poly1=0.0,
        mean=jnp.zeros(2, dtype=float),
        log_jitter=jnp.full(2, -4.0, dtype=float),
        lag0=jnp.asarray(5.0),
        lag_beta=jnp.asarray(4.0 / 3.0),
        log_igm_transmission_band=jnp.zeros(2, dtype=float),
        eta_sigma=jnp.asarray(-0.5),
        eta_tau=jnp.asarray(0.2),
        log_amp_delta_blr=jnp.array([-0.3, -0.2], dtype=float),
        log_lag_blr=jnp.log(jnp.array([50.0, 60.0], dtype=float)),
        log_amp_delta_blr2=jnp.full(2, -9.0, dtype=float),
        log_lag_blr2=jnp.full(2, -9.0, dtype=float),
        log_amp_delta_bc=jnp.asarray(-0.8),
        log_lag_ratio_bc_to_blr=jnp.log(0.2),
    )
    raw_shifted = dict(raw_base)
    raw_shifted["eta_sigma"] = jnp.asarray(-1.2)

    explicit_base = build_explicit_model_params_fluxmix_fast(raw_base, lam_rf)
    explicit_shifted = build_explicit_model_params_fluxmix_fast(raw_shifted, lam_rf)
    ratio_base = compute_flux_line_ratio_offsets(
        lam_rf,
        lambda_center_rf=lambda_center_rf,
        eta_sigma=raw_base["eta_sigma"],
        log_igm_transmission_band=raw_base["log_igm_transmission_band"],
    )
    ratio_shifted = compute_flux_line_ratio_offsets(
        lam_rf,
        lambda_center_rf=lambda_center_rf,
        eta_sigma=raw_shifted["eta_sigma"],
        log_igm_transmission_band=raw_shifted["log_igm_transmission_band"],
    )

    assert not np.allclose(np.asarray(explicit_base["amp_cont"]), np.asarray(explicit_shifted["amp_cont"]))
    assert not np.allclose(np.asarray(explicit_base["amp_blr"]), np.asarray(explicit_shifted["amp_blr"]))
    assert not np.allclose(np.asarray(explicit_base["amp_bc"]), np.asarray(explicit_shifted["amp_bc"]))
    assert not np.allclose(np.asarray(ratio_base["blr_band"]), np.asarray(ratio_shifted["blr_band"]))
    assert not np.allclose(np.asarray(ratio_base["bc_ref"]), np.asarray(ratio_shifted["bc_ref"]))


def test_save_combined_plot_fluxmix_handles_singleton_sample_entries(monkeypatch):
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

    samples_flat, _, model, _ = run_two_stage_fluxmix_fast_inference(
        obj,
        lam_rf,
        log_jitter_mean=jnp.array(log_jitter_mean),
        rng_key=random.PRNGKey(10),
        num_warmup=1,
        num_samples=2,
        num_chains=1,
        chain_method="sequential",
        progress_bar=False,
        dense_mass=False,
        max_tree_depth=1,
        disable_poly1=False,
        disable_lag_blr=False,
        disable_lag_bc=False,
        drop_band_lyman_alpha=False,
        tau_fast_truncated=False,
        n_blr_terms=1,
    )

    plot_samples = dict(samples_flat)
    plot_samples["singleton_diag"] = np.asarray([1.0], dtype=float)
    monkeypatch.setattr("matplotlib.pyplot.savefig", lambda *args, **kwargs: None)

    save_combined_plot(
        plot_samples,
        model,
        obj["X"],
        obj["y"],
        obj["yerr"],
        obj["band_idx"],
        obj["mags_means"],
        obj.get("survey_times", {}),
        obj,
        time0=obj["time0"],
        bands=bands,
        plot_psd=True,
        filename_suffix="pytest_fluxmix_singleton",
    )


def test_fluxmix_saved_samples_preserve_stage1_basis_for_rebuild():
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

    samples_flat, _, _, _ = run_two_stage_fluxmix_fast_inference(
        obj,
        lam_rf,
        log_jitter_mean=jnp.array(log_jitter_mean),
        rng_key=random.PRNGKey(12),
        num_warmup=1,
        num_samples=2,
        num_chains=1,
        chain_method="sequential",
        progress_bar=False,
        dense_mass=False,
        max_tree_depth=1,
        disable_poly1=False,
        disable_lag_blr=False,
        disable_lag_bc=False,
        drop_band_lyman_alpha=False,
        tau_fast_truncated=False,
        n_blr_terms=1,
    )

    assert "stage1_basis_eta_sigma" in samples_flat
    assert "stage1_basis_log_sigma_center0" in samples_flat
    assert "delta_eta_sigma" in samples_flat
    assert "delta_log_sigma_center0" in samples_flat
    np.testing.assert_allclose(
        np.asarray(samples_flat["delta_eta_sigma"], dtype=float),
        np.asarray(samples_flat["eta_sigma"], dtype=float)
        - np.asarray(samples_flat["stage1_basis_eta_sigma"], dtype=float),
    )
    np.testing.assert_allclose(
        np.asarray(samples_flat["delta_log_sigma_center0"], dtype=float),
        np.asarray(samples_flat["log_sigma_center0"], dtype=float)
        - np.asarray(samples_flat["stage1_basis_log_sigma_center0"], dtype=float),
    )

    stage1_raw = _fluxmix_stage1_raw_median_params(samples_flat, lam_rf)
    np.testing.assert_allclose(
        np.asarray(stage1_raw["eta_sigma"], dtype=float),
        np.median(np.asarray(samples_flat["stage1_basis_eta_sigma"], dtype=float), axis=0),
    )
    np.testing.assert_allclose(
        np.asarray(stage1_raw["log_sigma_center0"], dtype=float),
        np.median(np.asarray(samples_flat["stage1_basis_log_sigma_center0"], dtype=float), axis=0),
    )

    rebuilt_model = build_mag_fluxmix_fast_display_model(obj, lam_rf, samples_flat)
    posterior_median = {k: np.median(v, axis=0) for k, v in samples_flat.items()}
    rebuilt_mu, rebuilt_std = rebuilt_model.pred(posterior_median, obj["X"])
    assert np.all(np.isfinite(np.asarray(rebuilt_mu)))
    assert np.all(np.isfinite(np.asarray(rebuilt_std)))


def test_build_single_object_model_mag_fluxmix_stage2_rejects_second_blr_term():
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
    stage1_raw_median = dict(
        log_tau_slow_center0=jnp.log(100.0),
        log_tau_fast_center0=jnp.log(10.0),
        log_sigma_center0=jnp.log(0.1),
        lambda_center_rf=jnp.exp(jnp.mean(jnp.log(lam_rf))),
        poly1=0.0,
        mean=jnp.zeros(len(bands)),
        log_jitter=jnp.full(len(bands), -4.0),
        lag0=jnp.asarray(5.0),
        lag_beta=jnp.asarray(4.0 / 3.0),
        log_igm_transmission_band=jnp.zeros(len(bands), dtype=float),
        eta_sigma=jnp.asarray(-0.5),
        eta_tau=jnp.asarray(0.2),
    )
    basis_grid_t = jnp.linspace(-100.0, 100.0, 64)
    basis_relflux_norm = jnp.ones((len(bands), 64), dtype=float) * 0.01
    stage1_params_median = {
        "lambda_center_rf": jnp.exp(jnp.mean(jnp.log(lam_rf))),
        "log_sigma_uv": jnp.log(0.1),
        "amp_cont": jnp.full(len(bands), 0.1),
    }

    with pytest.raises(ValueError, match=r"mag_fluxmix_fast.*n_blr_terms=1"):
        build_single_object_model_mag_fluxmix_stage2(
            obj,
            lam_rf,
            stage1_raw_median=stage1_raw_median,
            stage1_params_median=stage1_params_median,
            basis_grid_t=basis_grid_t,
            basis_relflux_norm=basis_relflux_norm,
            disable_lag_blr=False,
            disable_lag_bc=False,
            n_blr_terms=2,
        )


def test_plot_all_histograms_handles_constant_parameter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    samples_flat = {
        "F0_cont_band_g": np.ones(32, dtype=float),
        "amp_cont_g": np.linspace(0.1, 0.2, 32, dtype=float),
    }
    data = {
        "object_id": "12345",
        "z": 0.5,
    }

    _, _, save_path = plot_all_histograms(samples_flat, data)
    assert Path(save_path).exists()


def test_save_dm_df_over_f_distribution_plot_writes_pdf(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    obj = _make_fake_public_object()
    lc = make_lc(
        obj,
        ["g", "r"],
        inject_fake=False,
        drop_band_lyman_alpha=False,
    )
    obj = obj | lc

    save_dm_df_over_f_distribution_plot(obj | {"bands": obj["bands"]})

    save_path = (
        tmp_path
        / "plots"
        / "multiband"
        / "test"
        / "dm_df_over_f_distributions"
        / f"{float(obj['z']):.1f}_{obj['object_id']}_dm_df_over_f_test.pdf"
    )
    assert save_path.exists()


def test_flatten_flat_samples_per_band_skips_internal_log_kernel_param():
    flat_per_band = flatten_flat_samples_per_band(
        {
            "log_kernel_param": np.ones((20, 8), dtype=float),
            "amp_cont": np.ones((20, 4), dtype=float),
            "eta_sigma": np.ones(20, dtype=float),
        },
        bands=["u", "g", "r", "i"],
    )

    assert "log_kernel_param" not in flat_per_band
    assert "amp_cont_u" in flat_per_band
    assert "eta_sigma" in flat_per_band


def test_flatten_per_chain_samples_per_band_skips_internal_log_kernel_param():
    flat_per_band = flatten_per_chain_samples_per_band(
        {
            "log_kernel_param": np.ones((1, 20, 8), dtype=float),
            "amp_cont": np.ones((1, 20, 4), dtype=float),
            "eta_sigma": np.ones((1, 20), dtype=float),
        },
        bands=["u", "g", "r", "i"],
    )

    assert "log_kernel_param" not in flat_per_band
    assert "amp_cont_u" in flat_per_band
    assert "eta_sigma" in flat_per_band


def test_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    for name in (
        "plot_alpha_lambda_vs_l2500_by_redshift",
        "plot_adf_pvalue_g_diagnostic",
        "plot_alpha_lambda_histogram",
        "plot_blr_lag_vs_amp_by_band",
        "plot_bc_lag_vs_l2500",
        "plot_blr_line_lags_vs_l2500_fiducial",
        "plot_blr_lag_vs_redshift_by_band",
        "plot_f_host_2500_vs_l2500",
        "plot_f_host_2500_vs_redshift",
        "plot_g_band_drift_slope_histograms",
        "plot_Mi_relation",
        "plot_cut_diagnostics",
        "plot_m2500_vs_z_colorpanels",
        "plot_sf_ref_band_vs_model_g",
        "plot_sf_vs_uv_variability",
        "plot_sigma_bc_vs_frac_bc",
        "plot_sigma_bc_vs_redshift",
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

    df, df_all = hubble_utils.load_agn_data(
        h5_path,
        spectra_fit_csv=None,
        lc_info_csv=None,
        only_load=True,
        apply_cut=False,
        plot_path=str(tmp_path / "figures"),
    )
    assert df_all.equals(df)

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
