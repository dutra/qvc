import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import pytest

from qvc.hubble import hubble_plotting


def _psd_recovery_frame():
    log_sigma_uv = np.array([-1.0, -0.8, -0.6])
    log_tau_uv_rf = np.array([1.5, 2.0, 2.5])
    drw_log_rms_factor = -0.5 * np.log10(2.0)
    return pd.DataFrame(
        {
            "log_sigma_uv": log_sigma_uv,
            "log_sigma_uv_err": np.full(3, 0.04),
            "log_tau_uv_rf": log_tau_uv_rf,
            "log_tau_uv_rf_err": np.full(3, 0.08),
            "log_sigma_ls": log_sigma_uv + 0.10 - drw_log_rms_factor,
            "log_sigma_ls_err": np.full(3, 0.05),
            "log_tau_ls": log_tau_uv_rf + 0.20,
            "log_tau_ls_err": np.full(3, 0.10),
            "alpha_high_ls": np.full(3, -2.0),
            "alpha_high_ls_err": np.zeros(3),
            "psd_ls_valid": [True, False, True],
            "log_sigma_ls_fixed": log_sigma_uv + 0.05,
            "log_sigma_ls_fixed_err": np.full(3, 0.05),
            "log_tau_ls_fixed": log_tau_uv_rf + 0.12,
            "log_tau_ls_fixed_err": np.full(3, 0.10),
            "psd_ls_fixed_valid": [True, True, False],
        }
    )


def test_plot_psd_uv_recovery_comparison_normalizes_and_filters(monkeypatch):
    captured = {}

    def capture_figure(fig, path, **_kwargs):
        captured["fig"] = fig
        captured["path"] = path
        return path

    monkeypatch.setattr(hubble_plotting, "_save_figure", capture_figure)
    output = hubble_plotting.plot_psd_uv_recovery_comparison(
        _psd_recovery_frame(),
        plot_path="plots/example",
        filename="comparison.pdf",
    )

    fig = captured["fig"]
    try:
        axes = np.asarray(fig.axes).reshape(2, 2)
        assert axes[0, 0].get_title() == "Free-slope BPL"
        assert axes[0, 1].get_title() == "Fixed-slope DRW"
        assert axes[0, 0].get_xlim() == pytest.approx(axes[0, 1].get_xlim())
        assert axes[1, 0].get_xlim() == pytest.approx(axes[1, 1].get_xlim())
        assert "N = 2" in axes[0, 0].texts[0].get_text()
        assert "Bias = +0.10 dex" in axes[0, 0].texts[0].get_text()
        assert "N = 2" in axes[0, 1].texts[0].get_text()
        assert "Bias = +0.05 dex" in axes[0, 1].texts[0].get_text()
        assert fig._suptitle is None
    finally:
        plt.close(fig)

    assert output.endswith("plots/example/diagnostics/comparison.pdf")


def test_plot_psd_uv_recovery_comparison_draws_thick_red_contours(monkeypatch):
    n_objects = 80
    phase = np.linspace(0.0, 4.0 * np.pi, n_objects)
    log_sigma_uv = np.linspace(-1.3, -0.45, n_objects)
    log_tau_uv_rf = np.linspace(1.0, 3.3, n_objects)
    frame = pd.DataFrame(
        {
            "log_sigma_uv": log_sigma_uv,
            "log_sigma_uv_err": np.full(n_objects, 0.04),
            "log_tau_uv_rf": log_tau_uv_rf,
            "log_tau_uv_rf_err": np.full(n_objects, 0.08),
            "log_sigma_ls": (
                log_sigma_uv + 0.08 + 0.04 * np.sin(phase) + 0.5 * np.log10(2.0)
            ),
            "log_sigma_ls_err": np.full(n_objects, 0.05),
            "log_tau_ls": log_tau_uv_rf + 0.20 + 0.12 * np.sin(phase),
            "log_tau_ls_err": np.full(n_objects, 0.10),
            "alpha_high_ls": np.full(n_objects, -2.0),
            "alpha_high_ls_err": np.zeros(n_objects),
            "psd_ls_valid": np.ones(n_objects, dtype=bool),
            "log_sigma_ls_fixed": log_sigma_uv + 0.05 + 0.03 * np.cos(phase),
            "log_sigma_ls_fixed_err": np.full(n_objects, 0.05),
            "log_tau_ls_fixed": log_tau_uv_rf + 0.12 + 0.10 * np.cos(phase),
            "log_tau_ls_fixed_err": np.full(n_objects, 0.10),
            "psd_ls_fixed_valid": np.ones(n_objects, dtype=bool),
        }
    )

    contour_kwargs = []
    original_contour = Axes.contour

    def capture_contour(self, *args, **kwargs):
        contour_kwargs.append(kwargs)
        return original_contour(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "contour", capture_contour)
    monkeypatch.setattr(
        hubble_plotting,
        "_save_figure",
        lambda fig, path, **_kwargs: (plt.close(fig), path)[1],
    )

    hubble_plotting.plot_psd_uv_recovery_comparison(frame)

    assert len(contour_kwargs) == 4
    for kwargs in contour_kwargs:
        assert len(kwargs["levels"]) == 2
        assert kwargs["colors"] == "red"
        assert kwargs["linestyles"] == ("solid", "solid")
        assert kwargs["linewidths"] == (2.6, 3.2)
