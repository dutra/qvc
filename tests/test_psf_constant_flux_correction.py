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
    load_spectra_psf_fractions,
    subtract_constant_flux_from_band,
)
from qvc.spectra.catalog_hdf5 import (
    JOINT_POSTERIOR_DRAW_COUNT,
    JOINT_POSTERIOR_DRAW_FIELDS,
    write_spectra_catalog_hdf5,
)


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


def _write_v3_spectra_catalog(path, frame, fraction_draws, valid_count):
    """Write the PSF fixture inside a physically valid spectra-v3 envelope."""

    frame = frame.copy()
    n_rows = len(frame)
    frame["fit_ok"] = True
    frame["mw_deredden_applied"] = True
    frame["joint_posterior_draw_source"] = "synthetic_test_posterior"
    medians = {
        "f_host_2500_psf": np.full(n_rows, 0.2),
        "alpha_nu_intrinsic_1450_2500": np.full(n_rows, -0.5),
        "alpha_nu_attenuated_1450_2500": np.full(n_rows, -0.5),
        "m_2500_dereddened": np.full(n_rows, 20.0),
        "m_2500_attenuated_model": np.full(n_rows, 20.0),
        "a_2500_galaxy": np.zeros(n_rows),
        "a_2500_internal": np.zeros(n_rows),
        "a_2500_total": np.zeros(n_rows),
    }
    for name, values in medians.items():
        frame[name] = values
        frame[f"{name}_err"] = 0.0
        frame[f"{name}_err_lower"] = 0.0
        frame[f"{name}_err_upper"] = 0.0
    joint_draws = {
        name: np.repeat(
            np.asarray(medians[name], dtype=np.float32)[:, None],
            JOINT_POSTERIOR_DRAW_COUNT,
            axis=1,
        )
        for name in JOINT_POSTERIOR_DRAW_FIELDS
    }
    fitted_fluxes = np.ones(
        (n_rows, JOINT_POSTERIOR_DRAW_COUNT, 5), dtype=np.float32
    )
    write_spectra_catalog_hdf5(
        path,
        frame,
        fraction_draws,
        valid_count,
        joint_posterior_draws=joint_draws,
        joint_posterior_valid_count=np.full(
            n_rows, JOINT_POSTERIOR_DRAW_COUNT, dtype=np.int16
        ),
        joint_posterior_index=np.tile(
            np.arange(JOINT_POSTERIOR_DRAW_COUNT, dtype=np.int32),
            (n_rows, 1),
        ),
        joint_psf_photometry_draws=fitted_fluxes,
        joint_psf_photometry_provenance={
            "prediction_source": "synthetic_test",
            "jaxsedfit_git_commit": "a" * 40,
        },
        joint_posterior_source_draw_count=np.full(
            n_rows, JOINT_POSTERIOR_DRAW_COUNT, dtype=np.int32
        ),
        joint_posterior_selection_seed=12345,
    )


def _write_spectra_h5(tmp_path, rows):
    h5_path = tmp_path / "spectra_fit.h5"
    frame = pd.DataFrame(rows)
    _write_v3_spectra_catalog(
        h5_path,
        frame,
        np.full((len(frame), 64, 5), np.nan, dtype=np.float32),
        np.zeros(len(frame), dtype=np.int16),
    )
    return str(h5_path)


def test_load_spectra_psf_fractions_reads_joint_draws_from_hdf5(tmp_path):
    path = tmp_path / "spectra_fit.h5"
    frame = pd.DataFrame(
        [{"object_id": "123", "f_AGN_psf_g": 0.7, "f_AGN_psf_g_err": 0.05}]
    )
    draws = np.full((1, 64, 5), np.nan, dtype=np.float32)
    draws[0, :2] = [[0.6, 0.7, 0.8, 0.9, 1.0], [0.5, 0.6, 0.7, 0.8, 0.9]]
    _write_v3_spectra_catalog(
        path,
        frame,
        draws,
        np.array([2]),
    )

    result = load_spectra_psf_fractions([path])

    assert result["123"]["f_AGN_psf_g"] == pytest.approx(0.7)
    assert result["123"]["psf_agn_fraction_valid_count"] == 2
    np.testing.assert_allclose(
        result["123"]["psf_agn_fraction_draws"],
        draws[0, :2],
    )
    assert result["123"]["psf_agn_fraction_bands"] == ("u", "g", "r", "i", "z")


def test_apply_constant_flux_correction_to_objects_raises_when_spectra_row_missing(tmp_path):
    objs = [_make_light_curve_object("123")]
    spectra_h5 = _write_spectra_h5(
        tmp_path,
        [{"object_id": "999", "f_AGN_psf_g": 0.5}],
    )

    with pytest.raises(ValueError, match="Missing spectra rows.*123"):
        apply_constant_flux_correction_to_objects(
            objs,
            spectra_fit_h5s=[spectra_h5],
        )


def test_apply_constant_flux_correction_to_objects_raises_when_no_band_is_corrected(tmp_path):
    objs = [_make_light_curve_object("123")]
    spectra_h5 = _write_spectra_h5(
        tmp_path,
        [{"object_id": "123", "f_AGN_psf_g": np.nan}],
    )

    with pytest.raises(ValueError, match="No valid PSF constant-flux correction band.*123"):
        apply_constant_flux_correction_to_objects(
            objs,
            spectra_fit_h5s=[spectra_h5],
        )


def test_apply_constant_flux_correction_to_objects_succeeds_when_all_objects_have_a_band(tmp_path):
    objs = [_make_light_curve_object("123"), _make_light_curve_object("456")]
    spectra_h5 = _write_spectra_h5(
        tmp_path,
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


def test_apply_constant_flux_correction_to_objects_raises_when_h5_has_only_pl_columns(tmp_path):
    objs = [_make_light_curve_object("123")]
    spectra_h5 = _write_spectra_h5(
        tmp_path,
        [{"object_id": "123", "f_PL_psf_g": 0.5}],
    )

    with pytest.raises(ValueError, match="missing required per-band columns 'f_AGN_psf_<band>'"):
        apply_constant_flux_correction_to_objects(
            objs,
            spectra_fit_h5s=[spectra_h5],
        )
