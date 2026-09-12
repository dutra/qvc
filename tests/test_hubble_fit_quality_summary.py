import io

import pytest
from rich.console import Console

from qvc.hubble.hubble_fit import print_debiased_fit_quality


@pytest.fixture
def summary():
    return dict(n_fit=3374, n_selected=3373, z_range=(.44, 3.16),
                chi2_full=.37, chi2_data_only=1.7, chi2_selected=1.04,
                median_sigmas=[.97, .51, .58], residual_rms=.614,
                redshift_trend=dict(slope_mag_per_dex=-.19, slope_err_mag_per_dex=.07,
                                    slope_significance_sigma=-2.71, delta_chi2=7.35))


def render(monkeypatch, summary, *, width=100, terminal=False, prefix="unit"):
    stream = io.StringIO()
    console = Console(file=stream, width=width, force_terminal=terminal, color_system="standard" if terminal else None)
    monkeypatch.setattr("rich.console.Console", lambda: console)
    print_debiased_fit_quality(summary, cosmo_model="Flatw0waCDM", prefix=prefix)
    return stream.getvalue()


def test_console_fit_quality(monkeypatch, summary):
    text = render(monkeypatch, summary)
    for label in ("Quantity", "Value", "Units / interpretation", "Residual RMS",
                  "Reduced χ²: full", "Reduced χ²: data only", "Reduced χ²: selected",
                  "Median σ: full", "Median σ: data only", "Median σ: selected",
                  "Selection-weighted γ_z", "Trend significance", "Trend Δχ²",
                  "Run: unit", "Model: Flatw0waCDM", "selection-conditioned"):
        assert label in text
    for value in ("3,374", "3,373", "0.440", "3.160", "0.370", "1.700", "1.040",
                  "0.970", "0.510", "0.580", "0.614", "-0.190 ± 0.070", "-2.71", "7.350",
                  "mag / dex in log₁₀(1+z)", "individual AGNs"):
        assert value in text
    assert "\x1b[" not in text


@pytest.mark.parametrize("trend", [None, {}])
def test_unavailable_trend(monkeypatch, summary, trend):
    summary["redshift_trend"] = trend
    summary["chi2_selected"] = float("nan")
    summary["median_sigmas"][2] = float("inf")
    text = render(monkeypatch, summary)
    assert text.count("unavailable") == 5
    assert "nan" not in text
    assert "inf" not in text


def test_empty_summary_is_silent(capsys):
    print_debiased_fit_quality({}, cosmo_model="Flatw0waCDM", prefix="unit")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("terminal", [False, True])
def test_long_run_name_and_narrow_output(monkeypatch, summary, terminal):
    prefix = "sep11b_" + "long_run_name_" * 20 + "[literal]"
    text = render(monkeypatch, summary, width=80, terminal=terminal, prefix=prefix)
    import re
    plain = re.sub(r"\x1b\[[0-9;]*m", "", text)
    assert prefix in "".join(plain.splitlines()).replace("Run: ", "")
    assert all(len(line) <= 80 for line in plain.splitlines())
    assert "0.614" in plain
    assert "Trend Δχ²" in plain
