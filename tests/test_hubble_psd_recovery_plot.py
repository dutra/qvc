import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import pytest

from qvc.hubble import hubble_plotting


def _psd_recovery_frame():
    log_sigma_uv = np.array([-1.0, -0.8, -0.6])
    log_tau_uv_rf = np.array([1.5, 2.0, 2.5])
    log_tau_total_band = log_tau_uv_rf + 0.25
    drw_log_rms_factor = -0.5 * np.log10(2.0)
    return pd.DataFrame(
        {
            "z": np.full(3, 1.0),
            "eta_sigma": np.full(3, -0.5),
            "eta_sigma_err": np.full(3, 0.05),
            "psd_bpl_ref_band": ["g", "g", "g"],
            "psd_bpl_ref_lambda_rf": np.full(3, 3000.0),
            "log_sigma_total_rms_band_g": log_sigma_uv,
            "log_sigma_total_rms_band_g_err": np.full(3, 0.04),
            "log_tau_band_g_RF": log_tau_total_band,
            "log_tau_band_g_RF_err": np.full(3, 0.06),
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
            "psd_ls_fmax": np.full(3, 2e-2),
            "log_sigma_ls_fixed": log_sigma_uv + 0.05,
            "log_sigma_ls_fixed_err": np.full(3, 0.05),
            "log_tau_ls_fixed": log_tau_uv_rf + 0.12,
            "log_tau_ls_fixed_err": np.full(3, 0.10),
            "psd_ls_fixed_valid": [True, True, False],
            "psd_ls_fixed_fmax": np.full(3, 2e-2),
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
        wavelength_shift = -0.5 * np.log10(2500.0 / 3000.0)
        free_sigma_points = next(
            collection.get_offsets()
            for collection in axes[0, 0].collections
            if collection.get_offsets().shape == (2, 2)
        )
        np.testing.assert_allclose(
            np.sort(np.asarray(free_sigma_points)[:, 0]),
            np.sort(np.asarray([-1.0, -0.6]) + wavelength_shift),
        )
        free_tau_points = next(
            collection.get_offsets()
            for collection in axes[1, 0].collections
            if collection.get_offsets().shape == (2, 2)
        )
        np.testing.assert_allclose(
            np.sort(np.asarray(free_tau_points)[:, 0]),
            [1.75, 2.75],
        )
        assert "tau_{\\rm model}" in axes[1, 0].get_xlabel()
        assert "(days)" in axes[1, 0].get_xlabel()
        assert "tau_{\\rm PSD}" in axes[1, 0].get_ylabel()
        assert "(days)" in axes[1, 0].get_ylabel()
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
            "z": np.full(n_objects, 1.0),
            "eta_sigma": np.full(n_objects, -0.5),
            "psd_bpl_ref_band": np.full(n_objects, "g"),
            "psd_bpl_ref_lambda_rf": np.full(n_objects, 3000.0),
            "log_sigma_total_rms_band_g": log_sigma_uv,
            "log_sigma_total_rms_band_g_err": np.full(n_objects, 0.04),
            "log_tau_band_g_RF": log_tau_uv_rf + 0.15,
            "log_tau_band_g_RF_err": np.full(n_objects, 0.06),
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
            "psd_ls_fmax": np.full(n_objects, 2e-2),
            "log_sigma_ls_fixed": log_sigma_uv + 0.05 + 0.03 * np.cos(phase),
            "log_sigma_ls_fixed_err": np.full(n_objects, 0.05),
            "log_tau_ls_fixed": log_tau_uv_rf + 0.12 + 0.10 * np.cos(phase),
            "log_tau_ls_fixed_err": np.full(n_objects, 0.10),
            "psd_ls_fixed_valid": np.ones(n_objects, dtype=bool),
            "psd_ls_fixed_fmax": np.full(n_objects, 2e-2),
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


def test_plot_psd_uv_recovery_marks_and_excludes_unresolved_tau(monkeypatch):
    frame = _psd_recovery_frame()
    frame["psd_ls_fmax"] = 1e-3
    captured = {}

    monkeypatch.setattr(
        hubble_plotting,
        "_save_figure",
        lambda fig, path, **_kwargs: captured.update(fig=fig) or path,
    )

    hubble_plotting.plot_psd_uv_recovery_comparison(frame)

    fig = captured["fig"]
    try:
        free_tau_ax = np.asarray(fig.axes).reshape(2, 2)[1, 0]
        summary = free_tau_ax.texts[0].get_text()
        assert "N = 1" in summary
        assert "Unresolved" not in summary
        assert any(
            collection.get_label().startswith("$\\tau_{\\rm model}<")
            for collection in free_tau_ax.collections
        )
    finally:
        plt.close(fig)


def test_plot_psd_uv_recovery_filters_unresolved_tau_points(monkeypatch):
    frame = _psd_recovery_frame()
    frame["psd_ls_fmax"] = 1e-3
    captured = {}

    monkeypatch.setattr(
        hubble_plotting,
        "_save_figure",
        lambda fig, path, **_kwargs: captured.update(fig=fig) or path,
    )

    hubble_plotting.plot_psd_uv_recovery_comparison(
        frame,
        tau_resolution_mode="filter",
    )

    fig = captured["fig"]
    try:
        free_tau_ax = np.asarray(fig.axes).reshape(2, 2)[1, 0]
        summary = free_tau_ax.texts[0].get_text()
        assert "N = 1" in summary
        assert "Unresolved" not in summary
        assert not any(
            collection.get_label().startswith("$\\tau_{\\rm model}<")
            for collection in free_tau_ax.collections
        )
    finally:
        plt.close(fig)


def test_plot_psd_uv_recovery_selects_current_total_quantities_by_ref_band(
    monkeypatch,
):
    frame = _psd_recovery_frame()
    frame["psd_bpl_ref_band"] = ["g", "r", "g"]
    frame["psd_ls_valid"] = True
    frame["psd_ls_fixed_valid"] = True
    frame["log_sigma_total_rms_band_r"] = [-9.0, -0.35, -9.0]
    frame["log_sigma_total_rms_band_r_err"] = [9.0, 0.07, 9.0]
    frame["log_tau_band_r_RF"] = [-9.0, 3.10, -9.0]
    frame["log_tau_band_r_RF_err"] = [9.0, 0.09, 9.0]
    captured = {}
    tau_errorbars = []
    original_errorbar = Axes.errorbar

    def capture_errorbar(self, x, y, *args, **kwargs):
        x_values = np.asarray(x)
        if x_values.shape == (3,) and np.allclose(
            np.sort(x_values), [1.75, 2.75, 3.10]
        ):
            tau_errorbars.append(np.asarray(kwargs["xerr"]))
        return original_errorbar(self, x, y, *args, **kwargs)

    monkeypatch.setattr(
        hubble_plotting,
        "_save_figure",
        lambda fig, path, **_kwargs: captured.update(fig=fig) or path,
    )
    monkeypatch.setattr(Axes, "errorbar", capture_errorbar)

    hubble_plotting.plot_psd_uv_recovery_comparison(frame)

    fig = captured["fig"]
    try:
        free_tau_ax = np.asarray(fig.axes).reshape(2, 2)[1, 0]
        free_tau_points = next(
            collection.get_offsets()
            for collection in free_tau_ax.collections
            if collection.get_offsets().shape == (3, 2)
        )
        np.testing.assert_allclose(
            np.sort(np.asarray(free_tau_points)[:, 0]),
            [1.75, 2.75, 3.10],
        )
        assert tau_errorbars
        np.testing.assert_allclose(tau_errorbars[0], [0.06, 0.09, 0.06])
    finally:
        plt.close(fig)


def test_plot_psd_uv_recovery_reconstructs_legacy_total_rms(monkeypatch):
    frame = _psd_recovery_frame().drop(
        columns=[
            "log_sigma_total_rms_band_g",
            "log_sigma_total_rms_band_g_err",
        ]
    )
    frame["psd_bpl_ref_lambda_rf"] = 2500.0
    legacy_values = {
        "tau_fast_driver": 30.0,
        "tau_slow_driver": 300.0,
        "lag_disk_g": 5.0,
        "lag_blr_g": 80.0,
        "amp_cont_relflux_g": 0.08,
        "amp_blr_relflux_g": 0.04,
    }
    for column, value in legacy_values.items():
        frame[column] = value
        frame[f"{column}_err"] = 0.05 * value
    captured = {}
    monkeypatch.setattr(
        hubble_plotting,
        "_save_figure",
        lambda fig, path, **_kwargs: captured.update(fig=fig) or path,
    )

    with pytest.warns(RuntimeWarning, match="Reconstructed legacy total-band RMS"):
        hubble_plotting.plot_psd_uv_recovery_comparison(frame)

    fig = captured["fig"]
    try:
        free_sigma_ax = np.asarray(fig.axes).reshape(2, 2)[0, 0]
        free_sigma_points = next(
            collection.get_offsets()
            for collection in free_sigma_ax.collections
            if collection.get_offsets().shape == (2, 2)
        )
        assert np.all(np.isfinite(np.asarray(free_sigma_points)[:, 0]))
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "missing_column",
    [
        "log_sigma_total_rms_band_g",
        "log_sigma_total_rms_band_g_err",
        "log_tau_band_g_RF",
        "log_tau_band_g_RF_err",
    ],
)
def test_plot_psd_uv_recovery_requires_current_total_band_pairs(
    monkeypatch,
    missing_column,
):
    frame = _psd_recovery_frame().drop(columns=missing_column)
    frame["tau_fast_driver"] = 1.0
    frame["tau_slow_driver"] = 100.0
    for stem in ("lag_disk", "lag_blr", "amp_cont", "amp_blr"):
        frame[f"{stem}_g"] = 1.0
    save_called = False

    def capture_save(*_args, **_kwargs):
        nonlocal save_called
        save_called = True

    monkeypatch.setattr(hubble_plotting, "_save_figure", capture_save)

    with pytest.raises(KeyError, match=missing_column):
        hubble_plotting.plot_psd_uv_recovery_comparison(frame)
    assert not save_called
