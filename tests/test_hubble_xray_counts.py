import numpy as np
import pandas as pd
import pytest
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


def _write_ellipse_catalog(path, *, ra, dec, major, minor, angle, flux):
    Table(
        {
            "ra": ra,
            "dec": dec,
            "err_ellipse_r0": major,
            "err_ellipse_r1": minor,
            "err_ellipse_ang": angle,
            "flux_aper_b": flux,
            "flux_aper_lolim_b": np.asarray(flux) * 0.8,
            "flux_aper_hilim_b": np.asarray(flux) * 1.2,
        }
    ).write(path, format="votable")


def test_xray_match_uses_rotated_csc_error_ellipse(tmp_path):
    catalog = tmp_path / "csc_ellipse.vot"
    # PA=0 means that the 2-arcsec major axis points North.  The northern
    # object is inside while the eastern object lies outside the 0.5-arcsec
    # minor axis, despite both having the same 1.5-arcsec radial separation.
    _write_ellipse_catalog(
        catalog,
        ra=[10.0],
        dec=[0.0],
        major=[2.0],
        minor=[0.5],
        angle=[0.0],
        flux=[1.0e-14],
    )
    offset_deg = 1.5 / 3600.0
    frame = pd.DataFrame(
        {
            "object_id": ["north", "east"],
            "ra": [10.0, 10.0 + offset_deg],
            "dec": [offset_deg, 0.0],
        }
    )

    result = hubble_utils.populate_xray(frame, table_fpath=str(catalog))

    assert result.xray_matched.tolist() == [True, False]
    assert result.loc[0, "matched_ellipse_distance"] == pytest.approx(0.75, rel=1e-4)


def test_xray_ellipse_match_selects_smallest_normalized_distance(tmp_path):
    catalog = tmp_path / "csc_ellipse.vot"
    offset_deg = 0.4 / 3600.0
    _write_ellipse_catalog(
        catalog,
        ra=[10.0, 10.0 + offset_deg],
        dec=[0.0, 0.0],
        major=[2.0, 0.5],
        minor=[2.0, 0.5],
        angle=[0.0, 0.0],
        flux=[1.0e-14, 2.0e-14],
    )
    frame = pd.DataFrame(
        {
            "object_id": ["agn"],
            "ra": [10.0 + 0.1 / 3600.0],
            "dec": [0.0],
        }
    )

    result = hubble_utils.populate_xray(frame, table_fpath=str(catalog))

    # The first source is farther away on the sky but closer in units of its
    # positional uncertainty: 0.1/2 < 0.3/0.5.
    assert result.loc[0, "matched_idx_b"] == 0
    assert result.loc[0, "flux_aper_b"] == pytest.approx(1.0e-14)


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
