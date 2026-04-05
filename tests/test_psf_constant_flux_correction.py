import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.light_curve.psf_constant_flux_correction import (
    apply_constant_flux_correction_to_object,
    subtract_constant_flux_from_band,
)


def test_subtract_constant_flux_from_band_makes_curve_fainter_and_more_variable():
    mags = np.array([20.0, 20.1, 19.9, 20.2], dtype=float)
    magerrs = np.full_like(mags, 0.02)

    corrected_mags, corrected_magerrs, summary = subtract_constant_flux_from_band(
        mags,
        magerrs,
        pl_fraction=0.5,
    )

    assert np.all(np.isfinite(corrected_mags))
    assert np.all(np.isfinite(corrected_magerrs))
    assert summary["constant_contaminant_flux"] > 0.0
    assert np.nanmedian(corrected_mags) > np.nanmedian(mags)
    assert np.nanstd(corrected_mags) > np.nanstd(mags)


def test_apply_constant_flux_correction_to_object_requires_bandpass_fraction():
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

    assert corrected_obj["psf_constant_flux_corrected"] is True
    assert summary["n_corrected_bands"] == 1
    band_summary = corrected_obj["psf_constant_flux_band_summaries"]["g"]
    assert band_summary["source_key"] == "f_PL_psf_g"
    assert np.isclose(band_summary["pl_fraction"], 0.5)
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
    assert summary["n_corrected_bands"] == 0
    assert summary["n_missing_fraction_bands"] == 1
    assert corrected_obj["psf_constant_flux_band_summaries"] == {}
    assert np.allclose(corrected_obj["mags"]["g"], obj["mags"]["g"])
