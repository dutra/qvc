"""Photometric coverage class of the SED fit and the optional minimum-coverage cut."""

import json

import numpy as np
import pandas as pd
import pytest

from qvc.hubble import hubble_utils
from qvc.hubble.cuts import (
    PHOTOMETRY_COVERAGE_CLASS_COLUMN,
    PHOTOMETRY_COVERAGE_FLAG_COLUMNS,
    PHOTOMETRY_COVERAGE_MIN_ENV,
    PHOTOMETRY_FILTER_GROUPS,
    add_photometry_coverage_columns,
    normalize_photometry_coverage_min,
    photometry_coverage_columns,
    photometry_coverage_mask,
)

from test_hubble_load_plots import _minimal_agn_frame, _patch_load_agn_plotters

UGRIZ = ",".join(f"{band}_sdss" for band in "ugriz")
# One filter list per class, in the order of PHOTOMETRY_COVERAGE_CLASSES.
ROWS = {
    "full": UGRIZ + ",NUV_galex,J_ukidss,W1",
    "no_galex": UGRIZ + ",Ks_vhs,W2,W3",
    "no_nir": UGRIZ + ",FUV_galex,W4",
    "no_wise": UGRIZ + ",NUV_galex,H_2mass",
    "nir_only": UGRIZ + ",Y_ukidss,K_ukidss",
    "wise_only": UGRIZ + ",W1",
    "galex_only": UGRIZ + ",FUV_galex,NUV_galex",
    "ugriz_only": UGRIZ,
    "akari_iras_only": UGRIZ + ",F60_iras",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(PHOTOMETRY_COVERAGE_MIN_ENV, raising=False)


def _frame(filters, excluded=None):
    frame = pd.DataFrame({"photometry_filters": list(filters)})
    if excluded is not None:
        frame["photometry_filters_excluded"] = list(excluded)
    return frame


def test_every_photometry_table_filter_name_has_a_group():
    # Filter names of data/jul14_master_input_file_chisqgt20_bandwagon_photometry.csv
    # (catalog -> filter_name) plus the ugriz light-curve bands of the fit rows.
    table_names = {
        "H_2mass", "J_2mass", "Ks_2mass", "WIDE-S_akari", "L18W_akari",
        "W1", "W2", "W3", "W4", "FUV_galex", "NUV_galex",
        "F100_iras", "F25_iras", "F60_iras",
        "H_ukidss", "J_ukidss", "K_ukidss", "Y_ukidss",
        "H_vhs", "J_vhs", "Ks_vhs",
    }
    assert table_names | {f"{b}_sdss" for b in "ugriz"} == set(PHOTOMETRY_FILTER_GROUPS)


def test_classes_and_flags():
    coverage = photometry_coverage_columns(_frame(ROWS.values()))
    assert coverage[PHOTOMETRY_COVERAGE_CLASS_COLUMN].tolist() == list(ROWS)
    assert coverage["photometry_has_galex"].tolist() == [1, 0, 1, 1, 0, 0, 1, 0, 0]
    assert coverage["photometry_has_nir"].tolist() == [1, 1, 0, 1, 1, 0, 0, 0, 0]
    assert coverage["photometry_has_wise"].tolist() == [1, 1, 1, 0, 0, 1, 0, 0, 0]


def test_bytes_and_excluded_filters():
    # GALEX left out of the likelihood (sub-Lyman exclusion) does not count.
    frame = _frame(
        [ROWS["full"].encode(), ROWS["full"], ROWS["galex_only"]],
        excluded=["", "NUV_galex", "FUV_galex,NUV_galex"],
    )
    labels = photometry_coverage_columns(frame)[PHOTOMETRY_COVERAGE_CLASS_COLUMN].tolist()
    assert labels == ["full", "no_galex", "ugriz_only"]


def test_unknown_filter_raises():
    with pytest.raises(ValueError, match=r"Unknown photometric filter name\(s\) \['X_new'\]"):
        photometry_coverage_columns(_frame([UGRIZ + ",X_new"]))


def test_missing_filter_list_is_unknown_and_blocks_the_cut():
    frame = add_photometry_coverage_columns(_frame([ROWS["full"], np.nan]))
    assert frame[PHOTOMETRY_COVERAGE_CLASS_COLUMN].tolist() == ["full", None]
    assert photometry_coverage_mask(frame, "none").all()
    with pytest.raises(ValueError, match="1 row\\(s\\) have no 'photometry_filters'"):
        photometry_coverage_mask(frame, "galex")


@pytest.mark.parametrize(
    ("level", "kept"),
    [
        ("none", list(ROWS)),
        ("beyond_ugriz", [c for c in ROWS if c != "ugriz_only"]),
        ("infrared", ["full", "no_galex", "no_nir", "no_wise", "nir_only", "wise_only"]),
        ("galex", ["full", "no_nir", "no_wise", "galex_only"]),
    ],
)
def test_cut_levels(level, kept):
    frame = add_photometry_coverage_columns(_frame(ROWS.values()))
    mask = photometry_coverage_mask(frame, level)
    assert frame.loc[mask, PHOTOMETRY_COVERAGE_CLASS_COLUMN].tolist() == kept


def test_level_setting(monkeypatch):
    assert normalize_photometry_coverage_min() == "none"
    monkeypatch.setenv(PHOTOMETRY_COVERAGE_MIN_ENV, "Beyond-UGRIZ")
    assert normalize_photometry_coverage_min() == "beyond_ugriz"
    monkeypatch.setenv(PHOTOMETRY_COVERAGE_MIN_ENV, "nir")
    with pytest.raises(ValueError, match=PHOTOMETRY_COVERAGE_MIN_ENV):
        normalize_photometry_coverage_min()


def test_missing_columns():
    frame = pd.DataFrame({"z": [1.0]})
    # Not requested: nothing added, nothing raised.
    assert add_photometry_coverage_columns(frame) is frame
    with pytest.raises(ValueError, match="'photometry_filters'"):
        add_photometry_coverage_columns(frame, required=True)
    with pytest.raises(ValueError, match="needs the photometric coverage columns"):
        photometry_coverage_mask(frame, "infrared")


def test_minimal_inputs_carry_the_filter_lists(tmp_path):
    import h5py

    from qvc.hubble.minimal_inputs import export_minimal_hubble_inputs, read_minimal_spectra_hdf5
    from test_hubble_minimal_inputs import _write_sources

    light_source, spectra_source = _write_sources(tmp_path)
    with h5py.File(spectra_source, "r+") as handle:
        catalog = handle["catalog"]
        for name in ("photometry_filters", "photometry_filters_excluded"):
            if name in catalog:
                del catalog[name]
        catalog.create_dataset(
            "photometry_filters",
            data=[ROWS["full"], ROWS["ugriz_only"], ROWS["wise_only"]],
            dtype=h5py.string_dtype("utf-8"),
        )
        catalog.create_dataset(
            "photometry_filters_excluded", data=["NUV_galex", "", ""], dtype=h5py.string_dtype("utf-8")
        )
    output = tmp_path / "minimal_spectra.h5"
    export_minimal_hubble_inputs(light_source, spectra_source, tmp_path / "minimal_light.h5", output)
    frame = add_photometry_coverage_columns(read_minimal_spectra_hdf5(output).set_index("object_id"))
    assert frame.loc[["1", "2"], PHOTOMETRY_COVERAGE_CLASS_COLUMN].tolist() == ["no_galex", "ugriz_only"]


# --- load_agn_data ------------------------------------------------------------


def _load(tmp_path, monkeypatch, frame, tier="1"):
    source_path = tmp_path / "agn.h5"
    source_path.touch()
    monkeypatch.setattr(hubble_utils, "read_quasars_from_hdf5_flat", lambda *_a, **_k: frame.copy())
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


def _coverage_frame():
    frame = _minimal_agn_frame(n=4)
    frame["photometry_filters"] = [ROWS["full"], ROWS["nir_only"], ROWS["galex_only"], ROWS["ugriz_only"]]
    return frame


def _strip_coverage(frame):
    return frame.drop(
        columns=[*PHOTOMETRY_COVERAGE_FLAG_COLUMNS.values(), PHOTOMETRY_COVERAGE_CLASS_COLUMN],
        errors="ignore",
    )


def test_load_agn_data_off_keeps_every_object_and_the_configuration(tmp_path, monkeypatch):
    frame = _coverage_frame()
    selected, _ = _load(tmp_path, monkeypatch, frame)
    baseline, _ = _load(tmp_path, monkeypatch, frame.drop(columns="photometry_filters"))
    # Off: only the descriptive columns differ; values and the cut fingerprint are identical.
    assert selected[PHOTOMETRY_COVERAGE_CLASS_COLUMN].tolist() == ["full", "nir_only", "galex_only", "ugriz_only"]
    pd.testing.assert_frame_equal(
        _strip_coverage(selected).drop(columns="photometry_filters"), baseline
    )
    assert selected.attrs["cut_configuration_json"] == baseline.attrs["cut_configuration_json"]
    assert "photometry_coverage" not in selected.attrs["cut_configuration_json"]


@pytest.mark.parametrize(
    ("level", "kept"),
    [("beyond_ugriz", ["obj0", "obj1", "obj2"]), ("infrared", ["obj0", "obj1"]), ("galex", ["obj0", "obj2"])],
)
def test_load_agn_data_applies_the_cut(tmp_path, monkeypatch, level, kept):
    monkeypatch.setenv(PHOTOMETRY_COVERAGE_MIN_ENV, level)
    selected, _ = _load(tmp_path, monkeypatch, _coverage_frame())
    assert selected["object_id"].tolist() == kept
    configuration = json.loads(selected.attrs["cut_configuration_json"])
    assert configuration["tier1_photometry_coverage_min"] == level
    assert "tier1:photometry_coverage" in (tmp_path / "cut_summary.txt").read_text()


def test_load_agn_data_cut_needs_the_filter_column(tmp_path, monkeypatch):
    monkeypatch.setenv(PHOTOMETRY_COVERAGE_MIN_ENV, "galex")
    with pytest.raises(ValueError, match="'photometry_filters'"):
        _load(tmp_path, monkeypatch, _minimal_agn_frame(n=4))


def test_load_agn_data_unknown_filter_raises_only_when_requested(tmp_path, monkeypatch, capsys):
    frame = _coverage_frame()
    frame.loc[0, "photometry_filters"] = UGRIZ + ",X_new"
    selected, _ = _load(tmp_path, monkeypatch, frame)
    assert PHOTOMETRY_COVERAGE_CLASS_COLUMN not in selected.columns
    assert "Photometric coverage columns not added" in capsys.readouterr().out
    monkeypatch.setenv(PHOTOMETRY_COVERAGE_MIN_ENV, "infrared")
    with pytest.raises(ValueError, match="Unknown photometric filter"):
        _load(tmp_path, monkeypatch, frame)


def test_load_agn_data_cut_is_tier1(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(PHOTOMETRY_COVERAGE_MIN_ENV, "galex")
    selected, _ = _load(tmp_path, monkeypatch, _coverage_frame(), tier="0")
    assert len(selected) == 4
    assert "not applied with --cut-tier 0" in capsys.readouterr().out
    assert "photometry_coverage" not in selected.attrs["cut_configuration_json"]
