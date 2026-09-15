import numpy as np
import pandas as pd
from astropy.table import Table

from qvc.hubble import hubble_utils, hubble_plotting
from qvc.hubble.hubble_model import AgnPivotContext


def test_xray_match_flag_includes_counterpart_without_flux(tmp_path):
    catalog = tmp_path / "csc.vot"
    Table({"ra": [10.], "dec": [0.], "flux_aper_b": [np.nan],
           "flux_aper_lolim_b": [np.nan], "flux_aper_hilim_b": [np.nan]}).write(catalog, format="votable")
    frame = pd.DataFrame({"object_id": ["yes", "no"], "ra": [10., 20.], "dec": [0., 0.]})
    result = hubble_utils.populate_xray(frame, table_fpath=str(catalog))
    assert result.xray_matched.tolist() == [True, False]
    assert result.flux_aper_b.isna().all()


def test_alpha_ox_count_uses_plot_mask_and_keeps_missing_xerror(tmp_path, monkeypatch):
    frame = pd.DataFrame({"z": [1., 1., 1., 4.], "alphaOX": [1., np.nan, 1., 1.],
                          "alphaOX_err": [np.nan, .1, .1, .1],
                          "delta_alphaOX": [0., np.nan, 0., 0.],
                          "delta_alphaOX_err": [.1]*4})
    monkeypatch.setattr(hubble_plotting, "_save_figure", lambda *a, **k: None)
    counts = {}
    hubble_plotting.plot_residuals_vs_alphaOX(frame, np.zeros(4), [1., 1., np.nan, 1.],
                                             plot_path=str(tmp_path), sample_counts=counts)
    assert counts["NumAGNAlphaOXPlotted"] == 2
    import matplotlib.pyplot as plt
    plt.close("all")


def test_tex_xray_counts_reference_and_per_model(tmp_path, monkeypatch):
    monkeypatch.setattr(hubble_utils, "fit_sigma_tau_lambda_broken_pl", lambda *a, **k: None)
    frame = pd.DataFrame({"z": [1., 4.], "xray_matched": [True, True]})
    counts = {"FlatLambdaCDM": {"NumAGNXrayMatched": 2, "NumAGNAlphaOXPlotted": 1},
              "Flatw0waCDM": {"NumAGNXrayMatched": 2, "NumAGNAlphaOXPlotted": 2}}
    pivot = AgnPivotContext(observable_names=("log_sigma_uv", "log_tau_uv_rf"),
                            values=(-1., np.log10(200)), reference_object_ids=("obj1",), z_range=(.44,3.16))
    hubble_utils.write_results_tex_variables(
        frame, frame, pd.DataFrame({"IS_CALIBRATOR": [], "zHD": []}), (.44,3.16),
        {}, {}, None, str(tmp_path), result_prefix="fiducial", agn_pivot_context=pivot,
        agn_xray_counts=counts,
    )
    tex = (tmp_path/"param_results_fiducial.tex").read_text()
    assert r"\newcommand{\resultFiducialNumAGNXrayMatched}{\ensuremath{2}}" in tex
    assert r"\newcommand{\resultFiducialNumAGNAlphaOXPlotted}{\ensuremath{2}}" in tex
    assert r"\newcommand{\resultFiducialFlatLambdaCDMNumAGNAlphaOXPlotted}{\ensuremath{1}}" in tex
