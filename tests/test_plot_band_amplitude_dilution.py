import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_band_amplitude_dilution.py"
spec = importlib.util.spec_from_file_location("plot_band_amplitude_dilution", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def test_excess_variance_recovers_injected_amplitude():
    rng = np.random.default_rng(0)
    rows = []
    for band_id, amp in ((0, 0.20), (1, 0.15), (2, 0.10), (3, 0.08)):
        mags = 20.0 + rng.normal(0.0, amp, 400) + rng.normal(0.0, 0.02, 400)
        rows.append(pd.DataFrame({"objectId": "1", "psMag": mags, "psMagErr": 0.02, "filterID": band_id}))
    table = script.excess_variance_table(pd.concat(rows), min_epochs=20)

    for band, amp in zip("ugri", (0.20, 0.15, 0.10, 0.08)):
        assert np.sqrt(table[f"xs_{band}"].iloc[0]) == pytest.approx(amp, rel=0.12)


def test_binned_ratios_use_median_excess_variance_ratio():
    table = pd.DataFrame({
        "z": np.full(100, 1.0),
        "xs_u": np.full(100, 4.0e-2),
        "xs_g": np.full(100, 1.0e-2),
        "xs_r": np.full(100, 1.0e-2),
        "xs_i": np.full(100, 0.25e-2),
    })
    ratios = script.binned_band_ratios(table, [0.95, 1.05], n_boot=10)

    assert ratios.loc[0, "u"] == pytest.approx(np.log10(2.0))
    assert ratios.loc[0, "g"] == pytest.approx(0.0)
    assert ratios.loc[0, "i"] == pytest.approx(np.log10(0.5))


def test_toy_lyman_alpha_dilutes_u_more_than_r_at_z_two():
    pytest.importorskip("jaxsedfit")
    toy = script.DilutionToy(script.jaxsedfit_resource_dir(), [1.0, 2.0])
    params = dict(ew_fe=0.0, f_bc=0.0, r_line=0.0, r_bc=0.0, r_fe=0.0)
    dilution = toy.log_dilution(params)

    assert np.all(dilution.to_numpy() <= 0.0)
    assert dilution.loc[2.0, "u"] < dilution.loc[2.0, "r"] - 0.05
    # Fully responsive lines and no Fe II or Balmer continuum give no dilution.
    no_dilution = toy.log_dilution({**params, "r_line": 1.0})
    assert np.allclose(no_dilution.to_numpy(), 0.0)
