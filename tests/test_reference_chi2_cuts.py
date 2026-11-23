"""Tier-1 chi-square cuts on the fixed-definition reference chi-square."""

import numpy as np
import pytest

from qvc.hubble import hubble_utils
from qvc.hubble.cuts import (
    ALLOW_FITTED_NOISE_CHI2_ENV,
    AGN_TIER1_FIT_QUALITY_CUTS,
    REFERENCE_CHI2_CUT_COLUMNS,
    resolve_chi2_cut_columns,
)

from test_hubble_load_plots import _minimal_agn_frame, _patch_load_agn_plotters

LEGACY = list(REFERENCE_CHI2_CUT_COLUMNS)
REFERENCE = list(REFERENCE_CHI2_CUT_COLUMNS.values())
OTHER = ["loo_chi2_eff", "m_2500_dereddened_rhat"]


def _columns(cuts):
    return [column for column, _lower, _upper in cuts]


def test_reference_columns_replace_legacy_with_same_thresholds(monkeypatch):
    monkeypatch.delenv(ALLOW_FITTED_NOISE_CHI2_ENV, raising=False)
    cuts, source = resolve_chi2_cut_columns(
        LEGACY + REFERENCE + ["sed_reduced_chi2", "sed_reduced_chi2_ref", "frac_jitter"] + OTHER,
        AGN_TIER1_FIT_QUALITY_CUTS,
    )
    assert source == "reference"
    legacy_bounds = {c: (lo, hi) for c, lo, hi in AGN_TIER1_FIT_QUALITY_CUTS}
    for column, lower, upper in cuts:
        assert column not in LEGACY
        original = {v: k for k, v in REFERENCE_CHI2_CUT_COLUMNS.items()}.get(column, column)
        assert (lower, upper) == legacy_bounds[original]
    assert set(REFERENCE) <= set(_columns(cuts))
    # The SED cut keeps its version-independent fitted-width statistic.
    assert "sed_reduced_chi2" in _columns(cuts)
    assert "sed_reduced_chi2_ref" not in _columns(cuts)
    assert len(cuts) == len(AGN_TIER1_FIT_QUALITY_CUTS)


def test_catalogue_without_jitter_uses_legacy_columns_silently(monkeypatch):
    monkeypatch.delenv(ALLOW_FITTED_NOISE_CHI2_ENV, raising=False)
    cuts, source = resolve_chi2_cut_columns(LEGACY + OTHER, AGN_TIER1_FIT_QUALITY_CUTS)
    assert source == "legacy"
    assert cuts == list(AGN_TIER1_FIT_QUALITY_CUTS)


@pytest.mark.parametrize("marker", ["frac_jitter", "add_jitter", "frac_fe_jitter"])
def test_fitted_noise_catalogue_without_reference_raises(monkeypatch, marker):
    monkeypatch.delenv(ALLOW_FITTED_NOISE_CHI2_ENV, raising=False)
    with pytest.raises(ValueError, match="fitted spectral noise model"):
        resolve_chi2_cut_columns(LEGACY + [marker] + OTHER, AGN_TIER1_FIT_QUALITY_CUTS)


def test_fitted_noise_catalogue_can_be_allowed_explicitly(monkeypatch):
    columns = LEGACY + ["frac_jitter"] + OTHER
    cuts, source = resolve_chi2_cut_columns(
        columns, AGN_TIER1_FIT_QUALITY_CUTS, allow_fitted_noise_chi2=True
    )
    assert (source, cuts) == ("fitted-noise", list(AGN_TIER1_FIT_QUALITY_CUTS))
    monkeypatch.setenv(ALLOW_FITTED_NOISE_CHI2_ENV, "1")
    assert resolve_chi2_cut_columns(columns, AGN_TIER1_FIT_QUALITY_CUTS)[1] == "fitted-noise"
    monkeypatch.setenv(ALLOW_FITTED_NOISE_CHI2_ENV, "maybe")
    with pytest.raises(ValueError, match=ALLOW_FITTED_NOISE_CHI2_ENV):
        resolve_chi2_cut_columns(columns, AGN_TIER1_FIT_QUALITY_CUTS)


def test_partial_reference_columns_raise():
    with pytest.raises(ValueError, match="only some reference"):
        resolve_chi2_cut_columns(LEGACY + REFERENCE[:1], AGN_TIER1_FIT_QUALITY_CUTS)


def _load(tmp_path, monkeypatch, frame, tier="1"):
    source_path = tmp_path / "agn.h5"
    source_path.touch()
    monkeypatch.setattr(
        hubble_utils, "read_quasars_from_hdf5_flat", lambda *_a, **_k: frame.copy()
    )
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda value: value)
    _patch_load_agn_plotters(monkeypatch)
    return hubble_utils.load_agn_data(
        source_path,
        magnitude_convention="dereddened",
        spectra_fit_h5=None,
        cut_tier=tier,
        plot_diagnostics=False,
        plot_path=str(tmp_path / "plots"),
        cut_report_path=tmp_path / "cut_summary.txt",
    )


def test_load_agn_data_cuts_on_reference_columns(tmp_path, monkeypatch):
    monkeypatch.delenv(ALLOW_FITTED_NOISE_CHI2_ENV, raising=False)
    frame = _minimal_agn_frame(n=3)
    frame["object_id"] = ["keep", "legacy-high", "ref-high"]
    frame["frac_jitter"] = 0.004
    for legacy, reference in REFERENCE_CHI2_CUT_COLUMNS.items():
        frame[reference] = frame[legacy]
    frame["sed_reduced_chi2_ref"] = 50.0  # stored diagnostic, not cut on
    # A high fitted-noise value alone no longer removes an object ...
    frame.loc[1, "spectroscopy_reduced_chi2"] = 10.0
    # ... a high reference value does.
    frame.loc[2, "spectroscopy_reduced_chi2_ref"] = 10.0
    selected, _parent = _load(tmp_path, monkeypatch, frame)
    assert selected["object_id"].tolist() == ["keep", "legacy-high"]
    report = (tmp_path / "cut_summary.txt").read_text()
    assert "spectroscopy_reduced_chi2_ref" in report


def test_load_agn_data_rejects_fitted_noise_catalogue_without_reference(tmp_path, monkeypatch):
    monkeypatch.delenv(ALLOW_FITTED_NOISE_CHI2_ENV, raising=False)
    frame = _minimal_agn_frame(n=2)
    frame["frac_jitter"] = 0.004
    with pytest.raises(ValueError, match="reference chi-square"):
        _load(tmp_path, monkeypatch, frame)


def test_load_agn_data_keeps_legacy_columns_for_old_catalogues(tmp_path, monkeypatch):
    monkeypatch.delenv(ALLOW_FITTED_NOISE_CHI2_ENV, raising=False)
    frame = _minimal_agn_frame(n=2)
    frame["object_id"] = ["keep", "legacy-high"]
    frame.loc[1, "spectroscopy_reduced_chi2"] = 10.0
    selected, _parent = _load(tmp_path, monkeypatch, frame)
    assert selected["object_id"].tolist() == ["keep"]
