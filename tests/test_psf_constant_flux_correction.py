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

from qvc.light_curve.psf_constant_flux_correction import (
    apply_constant_flux_correction_to_object,
    apply_constant_flux_correction_to_objects,
    get_bandpass_agn_fraction,
    subtract_constant_flux_from_band,
)


def test_apply_constant_flux_correction_attaches_fixed_factor_without_modifying_observations():
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
    np.testing.assert_array_equal(corrected_obj["mags"]["g"], obj["mags"]["g"])
    np.testing.assert_array_equal(corrected_obj["magerrs"]["g"], obj["magerrs"]["g"])


def test_native_unit_agn_fraction_is_not_artificially_capped():
    value, error, key = get_bandpass_agn_fraction(
        {"f_AGN_psf_g": 1.0, "f_AGN_psf_g_err": 0.01},
        "g",
    )

    assert value == 1.0
    assert error == 0.01
    assert key == "f_AGN_psf_g"


def test_subtract_constant_flux_from_band_uses_native_reference_and_propagates_errors():
    reference_mag = 20.0
    mags = np.array([20.0, 20.1, 19.9], dtype=float)
    magerrs = np.full(3, 0.02, dtype=float)
    fraction = 0.5

    corrected_mags, corrected_magerrs, summary = subtract_constant_flux_from_band(
        mags,
        magerrs,
        fraction,
        reference_mag=reference_mag,
        agn_fraction_err=0.04,
    )

    reference_flux = 10.0 ** (-0.4 * reference_mag)
    total_flux = 10.0 ** (-0.4 * mags)
    agn_flux = total_flux - (1.0 - fraction) * reference_flux
    expected_mags = -2.5 * np.log10(agn_flux)
    expected_magerrs = total_flux / agn_flux * magerrs
    np.testing.assert_allclose(corrected_mags, expected_mags, rtol=1e-13)
    np.testing.assert_allclose(corrected_magerrs, expected_magerrs, rtol=1e-13)
    assert summary["reference_agn_mag"] == pytest.approx(
        reference_mag - 2.5 * np.log10(fraction)
    )
    assert summary["agn_fraction_err"] == pytest.approx(0.04)


def test_subtract_constant_flux_marks_nonpositive_agn_epochs_nonfinite():
    corrected_mags, corrected_magerrs, summary = subtract_constant_flux_from_band(
        np.array([20.0, 21.0]),
        np.array([0.02, 0.02]),
        0.2,
        reference_mag=20.0,
    )

    assert np.isfinite(corrected_mags[0])
    assert np.isfinite(corrected_magerrs[0])
    assert np.isnan(corrected_mags[1])
    assert np.isnan(corrected_magerrs[1])
    assert summary["n_nonpositive_after_subtraction"] == 1


def test_apply_constant_flux_correction_can_subtract_for_mag_linear():
    obj = {
        "object_id": "123",
        "z": 1.0,
        "times": {"g": np.array([0.0, 10.0, 20.0], dtype=float)},
        "mags": {"g": np.array([20.0, 20.1, 19.9], dtype=float)},
        "magerrs": {"g": np.full(3, 0.03, dtype=float)},
        "mags_mean": [20.0],
        "mags_mean_err": [0.01],
        "f_AGN_psf_g": 0.5,
        "f_AGN_psf_g_err": 0.04,
    }

    corrected_obj, summary = apply_constant_flux_correction_to_object(
        obj, subtract_observations=True
    )

    assert corrected_obj["psf_constant_flux_mode"] == "subtracted"
    assert not np.array_equal(corrected_obj["mags"]["g"], obj["mags"]["g"])
    assert np.all(corrected_obj["magerrs"]["g"] > obj["magerrs"]["g"])
    assert corrected_obj["psf_corrected_reference_mags_by_band"]["g"] == pytest.approx(
        20.0 - 2.5 * np.log10(0.5)
    )
    assert corrected_obj["psf_corrected_reference_magerrs_by_band"]["g"] == pytest.approx(
        0.02
    )
    assert summary["n_nonpositive_after_subtraction"] == 0


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
        obj["f_AGN_psf_g_err"] = 0.05
    return obj


def _write_spectra_csv(tmp_path, rows):
    csv_path = tmp_path / "spectra_fit.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return str(csv_path)


def test_apply_constant_flux_correction_to_objects_raises_when_spectra_row_missing(tmp_path):
    objs = [_make_light_curve_object("123")]
    spectra_csv = _write_spectra_csv(
        tmp_path,
        [{"object_id": "999", "f_AGN_psf_g": 0.5, "f_AGN_psf_g_err": 0.05}],
    )

    with pytest.raises(ValueError, match="Missing spectra rows.*123"):
        apply_constant_flux_correction_to_objects(
            objs,
            spectra_fit_csvs=[spectra_csv],
        )


def test_apply_constant_flux_correction_to_objects_raises_when_no_band_is_corrected(tmp_path):
    objs = [_make_light_curve_object("123")]
    spectra_csv = _write_spectra_csv(
        tmp_path,
        [{"object_id": "123", "f_AGN_psf_g": np.nan, "f_AGN_psf_g_err": 0.05}],
    )

    with pytest.raises(ValueError, match="No valid PSF constant-flux correction band.*123"):
        apply_constant_flux_correction_to_objects(
            objs,
            spectra_fit_csvs=[spectra_csv],
        )


def test_apply_constant_flux_correction_to_objects_succeeds_when_all_objects_have_a_band(tmp_path):
    objs = [_make_light_curve_object("123"), _make_light_curve_object("456")]
    spectra_csv = _write_spectra_csv(
        tmp_path,
        [
            {"object_id": "123", "f_AGN_psf_g": 0.5, "f_AGN_psf_g_err": 0.05},
            {"object_id": "456", "f_AGN_psf_g": 0.6, "f_AGN_psf_g_err": 0.06},
        ],
    )

    corrected_objs, summary = apply_constant_flux_correction_to_objects(
        objs,
        spectra_fit_csvs=[spectra_csv],
    )

    assert [str(obj["object_id"]) for obj in corrected_objs] == ["123", "456"]
    assert all(obj["psf_constant_flux_corrected"] is True for obj in corrected_objs)
    assert [obj["psf_constant_flux_n_bands_corrected"] for obj in corrected_objs] == [1, 1]
    assert summary["n_objects_corrected"] == 2
    assert summary["n_bands_corrected"] == 2


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


def test_apply_constant_flux_correction_to_objects_raises_when_csv_has_only_pl_columns(tmp_path):
    objs = [_make_light_curve_object("123")]
    spectra_csv = _write_spectra_csv(
        tmp_path,
        [{"object_id": "123", "f_PL_psf_g": 0.5}],
    )

    with pytest.raises(ValueError, match="missing required per-band columns 'f_AGN_psf_<band>'"):
        apply_constant_flux_correction_to_objects(
            objs,
            spectra_fit_csvs=[spectra_csv],
        )


def test_loader_rejects_fraction_without_uncertainty(tmp_path):
    objs = [_make_light_curve_object("123")]
    spectra_csv = _write_spectra_csv(
        tmp_path,
        [{"object_id": "123", "f_AGN_psf_g": 0.5}],
    )

    with pytest.raises(ValueError, match="missing fraction uncertainty.*f_AGN_psf_g_err"):
        apply_constant_flux_correction_to_objects(
            objs,
            spectra_fit_csvs=[spectra_csv],
        )
