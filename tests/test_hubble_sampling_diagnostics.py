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

