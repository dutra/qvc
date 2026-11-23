"""Sub-Lyman photometry exclusion: one setting per catalogue for SED chi-square cuts."""

import h5py
import numpy as np
import pandas as pd
import pytest

from qvc.hubble.cuts import (
    AGN_TIER1_FIT_QUALITY_CUTS,
    ALLOW_FITTED_NOISE_CHI2_ENV,
    ALLOW_MISSING_REFERENCE_CHI2_ENV,
    ALLOW_MIXED_PHOTOMETRY_EXCLUSION_ENV,
    CHI2_CUT_STATISTIC_ENV,
    REFERENCE_CHI2_CUT_COLUMNS,
    REFERENCE_FLOOR_ENV,
    check_photometry_exclusion_settings,
    expected_reference_floors,
    photometry_exclusion_settings,
    resolve_chi2_cut_columns,
)
from qvc.hubble.minimal_inputs import export_minimal_hubble_inputs, read_minimal_spectra_hdf5
from qvc.spectra.merge_results import load_and_merge_h5

from test_hubble_minimal_inputs import _write_sources
from test_spectra_merge_results import _write_v3_catalog

WAVE = "photometry_exclusion_rest_wavelength"
FRAC = "photometry_exclusion_max_fraction"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        *CHI2_CUT_STATISTIC_ENV.values(),
        ALLOW_FITTED_NOISE_CHI2_ENV,
        ALLOW_MISSING_REFERENCE_CHI2_ENV,
        ALLOW_MIXED_PHOTOMETRY_EXCLUSION_ENV,
        *(env for env, _default in REFERENCE_FLOOR_ENV.values()),
    ):
        monkeypatch.delenv(name, raising=False)


def _catalog(wavelengths, fractions=None, n_old=2):
    n = n_old if wavelengths is None else len(wavelengths)
    frame = {column: np.ones(n) for column in (*REFERENCE_CHI2_CUT_COLUMNS, *REFERENCE_CHI2_CUT_COLUMNS.values())}
    frame.update({field: np.full(n, value) for field, value in expected_reference_floors().items()})
    frame["frac_jitter"] = np.full(n, 0.02)
    if wavelengths is not None:
        frame[WAVE] = np.asarray(wavelengths, dtype=float)
        frame[FRAC] = np.asarray(fractions if fractions is not None else [0.5] * n, dtype=float)
    return pd.DataFrame(frame)


def test_settings_treat_nan_and_missing_columns_as_off():
    assert photometry_exclusion_settings(_catalog(None).iloc[:0]) == {}
    assert photometry_exclusion_settings(_catalog(None)) == {"off": 2}
    # NaN wavelength is off whatever the fraction column holds.
    assert photometry_exclusion_settings(_catalog([np.nan, np.nan], [np.nan, 0.5])) == {"off": 2}
    assert photometry_exclusion_settings(_catalog([912.0, 912.0])) == {(912.0, 0.5): 2}
    assert photometry_exclusion_settings(_catalog([912.0, np.nan])) == {(912.0, 0.5): 1, "off": 1}


@pytest.mark.parametrize("wavelengths", [None, [np.nan, np.nan, np.nan], [912.0, 912.0, 912.0]])
def test_uniform_catalogues_pass(wavelengths):
    cuts, _ = resolve_chi2_cut_columns(_catalog(wavelengths), AGN_TIER1_FIT_QUALITY_CUTS)
    assert len(cuts) == len(AGN_TIER1_FIT_QUALITY_CUTS)


@pytest.mark.parametrize(
    ("wavelengths", "fractions"),
    [([912.0, np.nan, np.nan], None), ([912.0, 912.0, 912.0], [0.5, 0.25, 0.5]), ([912.0, 1216.0, 912.0], None)],
)
def test_mixed_settings_block_the_sed_cut(wavelengths, fractions, monkeypatch):
    frame = _catalog(wavelengths, fractions)
    with pytest.raises(ValueError, match="different sub-Lyman photometry exclusion settings"):
        resolve_chi2_cut_columns(frame, AGN_TIER1_FIT_QUALITY_CUTS)
    # The explicit overrides let it through.
    resolve_chi2_cut_columns(frame, AGN_TIER1_FIT_QUALITY_CUTS, allow_mixed_photometry_exclusion=True)
    monkeypatch.setenv(ALLOW_MIXED_PHOTOMETRY_EXCLUSION_ENV, "1")
    assert len(check_photometry_exclusion_settings(frame)) == 2


def test_mixed_settings_do_not_block_cuts_without_photometric_chi2():
    frame = _catalog([912.0, np.nan])
    spectroscopy_only = [cut for cut in AGN_TIER1_FIT_QUALITY_CUTS if cut[0] == "spectroscopy_reduced_chi2"]
    assert spectroscopy_only
    resolve_chi2_cut_columns(frame, spectroscopy_only)
    # A list of column names (no values) cannot be checked and is not blocked.
    resolve_chi2_cut_columns(list(frame.columns), AGN_TIER1_FIT_QUALITY_CUTS)


def _shard_frame(object_id, **extra):
    return pd.DataFrame(
        {
            "object_id": [object_id],
            "fit_ok": [True],
            "fit_backend": ["jaxsedfit_joint"],
            "z": [2.5],
            "n_photometry": [16],
            "sed_reduced_chi2_ref": [1.5],
            **{key: [value] for key, value in extra.items()},
        }
    )


def test_merge_of_old_and_new_schema_shards(tmp_path):
    old, new_on, new_off = tmp_path / "chunk0000.h5", tmp_path / "chunk0001.h5", tmp_path / "chunk0002.h5"
    _write_v3_catalog(old, _shard_frame("1"), [0.2])
    _write_v3_catalog(
        new_on,
        _shard_frame("2", **{WAVE: 912.0, FRAC: 0.5, "photometry_filters_excluded": "NUV_galex", "n_photometry_excluded": 1}),
        [0.2],
    )
    _write_v3_catalog(
        new_off,
        _shard_frame("3", **{WAVE: np.nan, FRAC: np.nan, "photometry_filters_excluded": "", "n_photometry_excluded": 0}),
        [0.2],
    )
    old_and_off = load_and_merge_h5([str(old), str(new_off)]).frame
    assert np.isnan(old_and_off[WAVE]).all()
    assert photometry_exclusion_settings(old_and_off) == {"off": 2}
    check_photometry_exclusion_settings(old_and_off)

    mixed = load_and_merge_h5([str(old), str(new_on), str(new_off)]).frame
    assert photometry_exclusion_settings(mixed) == {"off": 2, (912.0, 0.5): 1}
    with pytest.raises(ValueError, match="QVC_CUT_ALLOW_MIXED_PHOTOMETRY_EXCLUSION"):
        check_photometry_exclusion_settings(mixed)


def test_minimal_inputs_carry_the_setting(tmp_path):
    light_source, spectra_source = _write_sources(tmp_path)
    with h5py.File(spectra_source, "r+") as handle:
        catalog = handle["catalog"]
        for name in (WAVE, FRAC):
            if name in catalog:
                del catalog[name]
        catalog.create_dataset(WAVE, data=np.asarray([912.0, np.nan, 912.0]))
        catalog.create_dataset(FRAC, data=np.asarray([0.5, np.nan, 0.5]))
    spectra_output = tmp_path / "minimal_spectra.h5"
    export_minimal_hubble_inputs(light_source, spectra_source, tmp_path / "minimal_light.h5", spectra_output)
    frame = read_minimal_spectra_hdf5(spectra_output).set_index("object_id")
    assert frame.loc["1", WAVE] == 912.0
    assert np.isnan(frame.loc["2", WAVE])
    with pytest.raises(ValueError, match="different sub-Lyman"):
        check_photometry_exclusion_settings(frame)

    # Older spectra catalogues without the columns still export (and count as off).
    (tmp_path / "old").mkdir()
    light_source, spectra_source = _write_sources(tmp_path / "old")
    with h5py.File(spectra_source, "r+") as handle:
        for name in (WAVE, FRAC):
            if name in handle["catalog"]:
                del handle["catalog"][name]
    old_output = tmp_path / "old_minimal_spectra.h5"
    export_minimal_hubble_inputs(light_source, spectra_source, tmp_path / "old_light.h5", old_output)
    old = read_minimal_spectra_hdf5(old_output)
    assert WAVE not in old
    assert photometry_exclusion_settings(old) == {"off": len(old)}
