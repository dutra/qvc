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

from qvc.hubble import hubble_plotting
from qvc.hubble.cuts import LIGHT_CURVE_N_POINTS_EXCLUDED_BANDS


def test_plot_light_curve_n_points_vs_apparent_mag_writes_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    df = pd.DataFrame(
        {
            "z": [0.7, 1.0, 1.4, 1.8, 2.2, 2.6],
            "apparent_mag_2500": [20.0, 20.5, 21.0, 21.5, 22.0, 22.5],
            "variability_n_points_g": [18, 16, 15, 12, 9, 7],
            "variability_n_points_r": [16, 15, 13, 11, 8, 6],
            "number_points_g": [20, 20, 20, 20, 20, 20],
        }
    )

    out = hubble_plotting.plot_light_curve_n_points_vs_apparent_mag(
        df,
        plot_path=str(tmp_path / "figures"),
        show=False,
    )

    assert out is not None
    assert os.path.exists(out)
    assert out.endswith("light_curve_n_points_vs_apparent_mag.pdf")


def test_plot_light_curve_n_points_vs_apparent_mag_excludes_u_by_default(tmp_path, monkeypatch):
    captured = {}

    def fake_point_count_series(_df, *, exclude_bands=None):
        captured["exclude_bands"] = exclude_bands
        return np.array([10.0, 20.0, 30.0]), ["variability_n_points_g"]

    def fake_save_figure(fig, path, **_kwargs):
        hubble_plotting.plt.close(fig)
        return path

    monkeypatch.setattr(hubble_plotting, "light_curve_point_count_series", fake_point_count_series)
    monkeypatch.setattr(hubble_plotting, "_save_figure", fake_save_figure)

    df = pd.DataFrame(
        {
            "z": [0.7, 1.0, 1.4],
            "apparent_mag_2500": [20.0, 20.5, 21.0],
            "variability_n_points_u": [1000, 1000, 1000],
            "variability_n_points_g": [10, 20, 30],
        }
    )

    hubble_plotting.plot_light_curve_n_points_vs_apparent_mag(
        df,
        plot_path=str(tmp_path / "figures"),
        show=False,
    )

    assert captured["exclude_bands"] == LIGHT_CURVE_N_POINTS_EXCLUDED_BANDS
