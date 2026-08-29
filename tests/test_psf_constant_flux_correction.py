import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.light_curve.psf_constant_flux_correction import (
    apply_constant_flux_correction_to_object,
    apply_constant_flux_correction_to_objects,
    subtract_constant_flux_from_band,
)
import qvc.light_curve.psf_constant_flux_correction as correction_module


def test_subtract_constant_flux_from_band_makes_curve_fainter_and_more_variable():
    mags = np.array([20.0, 20.1, 19.9, 20.2], dtype=float)
    magerrs = np.full_like(mags, 0.02)

    corrected_mags, corrected_magerrs, summary = subtract_constant_flux_from_band(
        mags,
        magerrs,
        agn_fraction=0.5,
    )

    assert np.all(np.isfinite(corrected_mags))
    assert np.all(np.isfinite(corrected_magerrs))
    assert summary["constant_contaminant_flux"] > 0.0
    assert np.nanmedian(corrected_mags) > np.nanmedian(mags)
    assert np.nanstd(corrected_mags) > np.nanstd(mags)


def test_subtract_constant_flux_from_band_keeps_agn_fraction_error_out_of_epoch_errors():
    mags = np.array([20.0, 20.1, 19.9, 20.2], dtype=float)
    magerrs = np.full_like(mags, 0.02)

    _, corrected_magerrs_no_frac_err, _ = subtract_constant_flux_from_band(
        mags,
        magerrs,
        agn_fraction=0.5,
        agn_fraction_err=0.0,
    )
    _, corrected_magerrs_with_frac_err, summary = subtract_constant_flux_from_band(
        mags,
        magerrs,
        agn_fraction=0.5,
        agn_fraction_err=0.05,
    )

    assert np.all(np.isfinite(corrected_magerrs_with_frac_err))
    np.testing.assert_allclose(
        corrected_magerrs_with_frac_err,
        corrected_magerrs_no_frac_err,
    )
    assert np.isclose(summary["agn_fraction_err"], 0.05)


def test_subtract_constant_flux_from_band_uses_error_weighted_mean_flux_by_default():
    mags = np.array([20.0, 20.1, 19.4, 20.2], dtype=float)
    magerrs = np.array([0.01, 0.02, 0.20, 0.04], dtype=float)

    _, _, summary = subtract_constant_flux_from_band(
        mags,
        magerrs,
        agn_fraction=0.7,
    )

    flux = 10.0 ** (-0.4 * mags)
    fluxerr = flux * (0.4 * np.log(10.0)) * magerrs
    weights = 1.0 / fluxerr**2
    expected_reference_flux = np.sum(weights * flux) / np.sum(weights)
    assert np.isclose(summary["reference_total_flux"], expected_reference_flux)
    assert not np.isclose(
        summary["reference_total_flux"],
        np.mean(flux),
        rtol=1e-6,
        atol=0.0,
    )


def test_apply_constant_flux_correction_to_object_requires_bandpass_fraction():
    obj = {
        "object_id": "123",
        "z": 1.0,
        "times": {"g": np.array([0.0, 10.0, 20.0], dtype=float)},
        "mags": {"g": np.array([20.0, 20.1, 19.9], dtype=float)},
        "magerrs": {"g": np.full(3, 0.03, dtype=float)},
        "mags_mean": [20.0],
        "f_AGN_psf_g": 0.5,
        "f_AGN_psf_g_err": 0.04,
    }

    corrected_obj, summary = apply_constant_flux_correction_to_object(obj)

    assert corrected_obj["psf_constant_flux_corrected"] is True
    assert corrected_obj["psf_constant_flux_n_bands_corrected"] == 1
    assert summary["n_corrected_bands"] == 1
    band_summary = corrected_obj["psf_constant_flux_band_summaries"]["g"]
    assert band_summary["source_key"] == "f_AGN_psf_g"
    assert np.isclose(band_summary["agn_fraction"], 0.5)
    assert np.isclose(band_summary["agn_fraction_err"], 0.04)
    assert np.nanmedian(corrected_obj["mags"]["g"]) > np.nanmedian(obj["mags"]["g"])


def test_apply_constant_flux_correction_to_object_skips_band_without_bandpass_fraction():
    obj = {
        "object_id": "123",
        "z": 1.0,
        "times": {"g": np.array([0.0, 10.0, 20.0], dtype=float)},
        "mags": {"g": np.array([20.0, 20.1, 19.9], dtype=float)},
        "magerrs": {"g": np.full(3, 0.03, dtype=float)},
        "mags_mean": [20.0],
    }

    corrected_obj, summary = apply_constant_flux_correction_to_object(obj)

    assert corrected_obj["psf_constant_flux_corrected"] is False
    assert corrected_obj["psf_constant_flux_n_bands_corrected"] == 0
    assert summary["n_corrected_bands"] == 0
    assert summary["n_missing_fraction_bands"] == 1
    assert corrected_obj["psf_constant_flux_band_summaries"] == {}
    assert np.allclose(corrected_obj["mags"]["g"], obj["mags"]["g"])


def _make_light_curve_object(object_id, *, include_fraction=False):
    obj = {
        "object_id": object_id,
        "z": 1.0,
        "times": {"g": np.array([0.0, 10.0, 20.0], dtype=float)},
        "mags": {"g": np.array([20.0, 20.1, 19.9], dtype=float)},
        "magerrs": {"g": np.full(3, 0.03, dtype=float)},
        "mags_mean": [20.0],
    }
    if include_fraction:
        obj["f_AGN_psf_g"] = 0.5
    return obj


def _mock_spectra_h5(monkeypatch, rows):
    calls = []

    def fake_reader(path, *, include_fraction_draws):
        calls.append((path, include_fraction_draws))
        return SimpleNamespace(frame=pd.DataFrame(rows))

    monkeypatch.setattr(correction_module, "read_spectra_catalog_hdf5", fake_reader)
    monkeypatch.setattr(correction_module, "resolve_qvc_data_path", lambda path: path)
    return "spectra_fit.h5", calls


def test_apply_constant_flux_correction_to_objects_raises_when_spectra_row_missing(monkeypatch):
    objs = [_make_light_curve_object("123")]
    spectra_h5, _ = _mock_spectra_h5(
        monkeypatch,
        [{"object_id": "999", "f_AGN_psf_g": 0.5}],
    )

    with pytest.raises(ValueError, match="Missing spectra rows.*123"):
        apply_constant_flux_correction_to_objects(
            objs,
            spectra_fit_h5s=[spectra_h5],
        )


def test_apply_constant_flux_correction_to_objects_raises_when_no_band_is_corrected(monkeypatch):
    objs = [_make_light_curve_object("123")]
    spectra_h5, _ = _mock_spectra_h5(
        monkeypatch,
        [{"object_id": "123", "f_AGN_psf_g": np.nan}],
    )

    with pytest.raises(ValueError, match="No valid PSF constant-flux correction band.*123"):
        apply_constant_flux_correction_to_objects(
            objs,
            spectra_fit_h5s=[spectra_h5],
        )


def test_apply_constant_flux_correction_to_objects_succeeds_when_all_objects_have_a_band(monkeypatch):
    objs = [_make_light_curve_object("123"), _make_light_curve_object("456")]
    spectra_h5, calls = _mock_spectra_h5(
        monkeypatch,
        [
            {"object_id": "123", "f_AGN_psf_g": 0.5},
            {"object_id": "456", "f_AGN_psf_g": 0.6},
        ],
    )

    corrected_objs, summary = apply_constant_flux_correction_to_objects(
        objs,
        spectra_fit_h5s=[spectra_h5],
    )

    assert [str(obj["object_id"]) for obj in corrected_objs] == ["123", "456"]
    assert all(obj["psf_constant_flux_corrected"] is True for obj in corrected_objs)
    assert [obj["psf_constant_flux_n_bands_corrected"] for obj in corrected_objs] == [1, 1]
    assert summary["n_objects_corrected"] == 2
    assert summary["n_bands_corrected"] == 2
    assert calls == [(spectra_h5, False)]


def test_apply_constant_flux_correction_to_object_does_not_accept_pl_fraction_only():
    obj = {
        "object_id": "123",
        "z": 1.0,
        "times": {"g": np.array([0.0, 10.0, 20.0], dtype=float)},
        "mags": {"g": np.array([20.0, 20.1, 19.9], dtype=float)},
        "magerrs": {"g": np.full(3, 0.03, dtype=float)},
        "mags_mean": [20.0],
        "f_PL_psf_g": 0.5,
    }

    corrected_obj, summary = apply_constant_flux_correction_to_object(obj)

    assert corrected_obj["psf_constant_flux_band_summaries"] == {}
    assert corrected_obj["psf_constant_flux_corrected"] is False
    assert corrected_obj["psf_constant_flux_n_bands_corrected"] == 0
    assert summary["n_corrected_bands"] == 0
    assert summary["n_missing_fraction_bands"] == 1


def test_apply_constant_flux_correction_to_objects_raises_when_h5_has_only_pl_columns(monkeypatch):
    objs = [_make_light_curve_object("123")]
    spectra_h5, _ = _mock_spectra_h5(
        monkeypatch,
        [{"object_id": "123", "f_PL_psf_g": 0.5}],
    )

    with pytest.raises(ValueError, match="missing required per-band columns 'f_AGN_psf_<band>'"):
        apply_constant_flux_correction_to_objects(
            objs,
            spectra_fit_h5s=[spectra_h5],
        )
