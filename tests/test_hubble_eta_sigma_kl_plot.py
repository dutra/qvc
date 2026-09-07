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

from qvc.hubble import hubble_plotting  # noqa: E402


def test_plot_eta_sigma_vs_redshift_colored_by_kl_writes_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    n_objects = 24
    frame = pd.DataFrame(
        {
            "z": np.linspace(0.4, 3.1, n_objects),
            "eta_sigma": np.linspace(-0.95, -0.55, n_objects),
            "eta_sigma_err": np.full(n_objects, 0.2),
            # Include a small negative numerical estimate: the plot must use
            # linear KL colors rather than taking an invalid logarithm.
            "eta_sigma_kl": np.linspace(-0.01, 1.3, n_objects),
            "eta_prior_profile": np.full(n_objects, "modified"),
        }
    )

    output = hubble_plotting.plot_eta_sigma_vs_redshift_colored_by_kl(
        frame,
        plot_path=str(tmp_path / "figures"),
        filename="eta_sigma_kl.pdf",
        kl_color_limits=(-0.01, 1.3),
        sample_label="Pre-cut sample",
    )

    assert output.endswith(os.path.join("diagnostics", "eta_sigma_kl.pdf"))
    assert os.path.exists(output)
    assert os.path.getsize(output) > 0


def test_plot_eta_sigma_vs_redshift_colored_by_kl_requires_kl_column(tmp_path):
    frame = pd.DataFrame({"z": [1.0], "eta_sigma": [-0.7]})

    with pytest.raises(KeyError, match="eta_sigma_kl"):
        hubble_plotting.plot_eta_sigma_vs_redshift_colored_by_kl(
            frame,
            plot_path=str(tmp_path),
        )
