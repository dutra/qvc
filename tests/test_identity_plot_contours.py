import numpy as np
import pandas as pd

from qvc.light_curve import merge_results, plotting_appendix


def test_identity_density_contours_use_68_and_84_percent_levels():
    rng = np.random.default_rng(123)
    x = rng.normal(size=2000)
    y = 0.6 * x + rng.normal(scale=0.5, size=x.size)

    contour_grid = plotting_appendix._density_contour_grid(
        x,
        y,
        xlim=(-4, 4),
        ylim=(-4, 4),
    )

    assert contour_grid is not None
    x_centers, y_centers, density, levels = contour_grid
    assert density.shape == (len(y_centers), len(x_centers))
    assert len(levels) == 2
    assert np.all(np.diff(levels) > 0)
    assert plotting_appendix.IDENTITY_CONTOUR_ENCLOSED_PROBABILITIES == (
        0.84,
        0.68,
    )


def _capture_identity_plot_style(monkeypatch):
    captured = {}

    def fake_plot(*_args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(merge_results, "plot_sigma_tau_identity_grid", fake_plot)
    monkeypatch.setattr(merge_results.plt, "close", lambda _fig: None)
    return captured


def test_stone_identity_grid_uses_smooth_blue_contours(monkeypatch, tmp_path):
    captured = _capture_identity_plot_style(monkeypatch)
    monkeypatch.setattr(
        merge_results,
        "build_stone_identity_plot_data",
        lambda *_args, **_kwargs: pd.DataFrame({"object_id": ["1"]}),
    )

    merge_results.write_stone_sigma_tau_identity_grid(
        [],
        tmp_path / "plots" / "stone.pdf",
    )

    assert captured["style"]["contour_color"] == "tab:blue"
    assert captured["style"]["contour_linewidth"] == 1.7
    assert captured["style"]["contour_smoothing"] == 4.0


def test_samelength_identity_grid_uses_smooth_blue_contours(
    monkeypatch,
    tmp_path,
):
    captured = _capture_identity_plot_style(monkeypatch)
    monkeypatch.setattr(
        merge_results,
        "build_samelength_identity_plot_data",
        lambda *_args, **_kwargs: pd.DataFrame({"object_id": ["1"]}),
    )

    merge_results.write_samelength_sigma_tau_identity_grid(
        [],
        [],
        tmp_path / "samelength.pdf",
    )

    assert captured["style"]["contour_color"] == "tab:blue"
    assert captured["style"]["contour_linewidth"] == 1.7
    assert captured["style"]["contour_smoothing"] == 4.0
