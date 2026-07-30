import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.constants import h
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble.hubble_utils import (
    compute_alpha_ox,
    integrated_xray_flux_to_log_lnu_2kev,
    populate_spectra_fit,
    populate_xray,
    rest_frame_ab_magnitude_to_log_lnu,
    xray_band_integral,
)


REFERENCE_LOG_L2500_NU = 30.278006016683868
REFERENCE_LOG_L2KEV_NU = 25.613712869321603
REFERENCE_COSMOLOGY = FlatLambdaCDM(H0=70, Om0=0.3)


def _write_csc_catalog(path, fluxes, flux_fractional_errors=None):
    fluxes = np.asarray(fluxes, dtype=float)
    if flux_fractional_errors is None:
        flux_fractional_errors = np.full(fluxes.shape, 0.2)
    errors = fluxes * np.asarray(flux_fractional_errors, dtype=float)
    Table(
        {
            "ra": 150.0 + np.arange(len(fluxes)) * 0.01,
            "dec": 2.0 + np.arange(len(fluxes)) * 0.01,
            "flux_aper_b": fluxes,
            "flux_aper_hilim_b": fluxes + errors,
            "flux_aper_lolim_b": fluxes - errors,
        }
    ).write(path, format="votable")


def _spectra_row(**updates):
    row = {
        "object_id": "agn_1",
        "fit_ok": True,
        "fracAGN_5100_fit": 0.5,
        "m_2500_dereddened": 20.0,
        "m_2500_dereddened_err": 0.1,
        "m_2500_attenuated_model": 20.4,
        "m_2500_attenuated_model_err": 0.2,
        "pl_slope": -1.5,
        "pl_slope_err": 0.1,
    }
    row.update(updates)
    return row


def test_luminosity_helpers_require_an_explicit_cosmology():
    with pytest.raises(TypeError, match="cosmology"):
        rest_frame_ab_magnitude_to_log_lnu(20.0, 1.0)
    with pytest.raises(TypeError, match="cosmology"):
        integrated_xray_flux_to_log_lnu_2kev(1.0e-14, 1.0)


def test_reference_uv_luminosity_uses_intrinsic_rest_frame_ab_magnitude():
    got = rest_frame_ab_magnitude_to_log_lnu(
        20.0,
        1.0,
        cosmology=REFERENCE_COSMOLOGY,
    )

    assert got == pytest.approx(REFERENCE_LOG_L2500_NU, abs=1e-12)


def test_reference_xray_luminosity_uses_csc_band_planck_without_k_correction():
    gamma = 1.9
    flux = 1.0e-14
    z = 1.0
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    dl_cm = cosmo.luminosity_distance(z).to_value(u.cm)

    band_integral = xray_band_integral(gamma, 0.5, 7.0)
    xray_norm = flux / band_integral
    l_e_2kev = (
        4.0
        * np.pi
        * dl_cm**2
        * xray_norm
        * 2.0 ** (1.0 - gamma)
    )
    expected_l_nu = l_e_2kev * h.to_value(u.keV * u.s)

    assert band_integral == pytest.approx(
        (7.0 ** (2.0 - gamma) - 0.5 ** (2.0 - gamma)) / (2.0 - gamma)
    )
    assert expected_l_nu / l_e_2kev == pytest.approx(h.to_value(u.keV * u.s))
    assert integrated_xray_flux_to_log_lnu_2kev(
        flux,
        z,
        cosmology=cosmo,
    ) == pytest.approx(
        REFERENCE_LOG_L2KEV_NU,
        abs=1e-12,
    )


def test_xray_band_integral_handles_gamma_two_logarithmic_limit():
    assert xray_band_integral(2.0, 0.5, 7.0) == pytest.approx(np.log(14.0))


@pytest.mark.parametrize(
    ("flux", "redshift"),
    [
        (0.0, 1.0),
        (-1.0e-14, 1.0),
        (np.nan, 1.0),
        (1.0e-14, 0.0),
        (1.0e-14, -1.0),
        (1.0e-14, np.nan),
    ],
)
def test_invalid_xray_inputs_produce_nan(flux, redshift):
    assert np.isnan(
        integrated_xray_flux_to_log_lnu_2kev(
            flux,
            redshift,
            cosmology=REFERENCE_COSMOLOGY,
        )
    )


@pytest.mark.parametrize(
    ("magnitude", "redshift"),
    [
        (np.nan, 1.0),
        (20.0, 0.0),
        (20.0, -1.0),
        (20.0, np.nan),
    ],
)
def test_invalid_uv_inputs_produce_nan(magnitude, redshift):
    assert np.isnan(
        rest_frame_ab_magnitude_to_log_lnu(
            magnitude,
            redshift,
            cosmology=REFERENCE_COSMOLOGY,
        )
    )


def test_luminosity_helpers_are_array_compatible_and_preserve_invalid_masks():
    uv = rest_frame_ab_magnitude_to_log_lnu(
        np.array([20.0, np.nan]),
        np.array([1.0, 1.0]),
        cosmology=REFERENCE_COSMOLOGY,
    )
    xray = integrated_xray_flux_to_log_lnu_2kev(
        np.array([1.0e-14, 0.0]),
        np.array([1.0, 1.0]),
        cosmology=REFERENCE_COSMOLOGY,
    )

    np.testing.assert_allclose(uv[0], REFERENCE_LOG_L2500_NU, atol=1e-12)
    np.testing.assert_allclose(xray[0], REFERENCE_LOG_L2KEV_NU, atol=1e-12)
    assert np.isnan(uv[1])
    assert np.isnan(xray[1])


def test_populate_spectra_fit_does_not_compute_cosmology_dependent_fields(tmp_path):
    spectra_path = tmp_path / "spectra.csv"
    pd.DataFrame([_spectra_row()]).to_csv(spectra_path, index=False)
    source = pd.DataFrame({"object_id": ["agn_1"], "z": [1.0]})

    out = populate_spectra_fit(source, [spectra_path])

    assert out.loc[0, "m_2500_dereddened"] == pytest.approx(20.0)
    assert out.loc[0, "m_2500_dereddened_err"] == pytest.approx(0.1)
    assert "log_L2500_nu" not in out.columns
    assert "log_L2500_nu_err" not in out.columns
    assert "log_L2500_int_fs" not in out.columns
    assert "log_L2500_int_fs_err" not in out.columns


def test_populate_spectra_fit_requires_intrinsic_magnitude_fields(tmp_path):
    spectra_path = tmp_path / "spectra_without_intrinsic.csv"
    row = _spectra_row()
    del row["m_2500_dereddened"]
    del row["m_2500_dereddened_err"]
    pd.DataFrame([row]).to_csv(spectra_path, index=False)
    source = pd.DataFrame({"object_id": ["agn_1"], "z": [1.0]})

    with pytest.raises(
        ValueError,
        match=r"m_2500_dereddened.*m_2500_dereddened_err",
    ):
        populate_spectra_fit(source, [spectra_path])


def test_populate_xray_only_adds_observed_flux_measurements(tmp_path):
    catalog_path = tmp_path / "csc.vot"
    _write_csc_catalog(catalog_path, [1.0e-14])
    source = pd.DataFrame(
        {
            "object_id": ["agn_1"],
            "ra": [150.0],
            "dec": [2.0],
            "z": [1.0],
        }
    )

    out = populate_xray(source, catalog_path)
    row = out.iloc[0]

    assert row["flux_aper_b"] == pytest.approx(1.0e-14)
    assert row["flux_aper_err_b"] == pytest.approx(2.0e-15)
    for column in (
        "log_L2500_nu",
        "log_L2keV_nu",
        "alphaOX",
        "alphaOX_exp",
        "delta_alphaOX",
        "log_L2500_int_fs",
        "log_Lxray",
    ):
        assert column not in out.columns


def test_compute_alpha_ox_reference_values_and_measurement_errors():
    source = pd.DataFrame(
        {
            "z": [1.0],
            "apparent_mag_2500": [24.0],
            "apparent_mag_2500_err": [0.5],
            "m_2500_dereddened": [20.0],
            "m_2500_dereddened_err": [0.1],
            "flux_aper_b": [1.0e-14],
            "flux_aper_err_b": [2.0e-15],
        }
    )

    out = compute_alpha_ox(source, cosmology=REFERENCE_COSMOLOGY)
    row = out.iloc[0]

    assert row["log_L2500_nu"] == pytest.approx(REFERENCE_LOG_L2500_NU, abs=1e-12)
    assert row["log_L2keV_nu"] == pytest.approx(REFERENCE_LOG_L2KEV_NU, abs=1e-12)
    assert row["log_L2keV_nu_err"] == pytest.approx(0.08685889638065036)
    assert row["alphaOX"] == pytest.approx(1.7905156035939598, abs=1e-12)
    assert row["alphaOX_exp"] == pytest.approx(1.486812926569315, abs=1e-12)
    assert row["delta_alphaOX"] == pytest.approx(0.3037026770246447, abs=1e-12)
    assert row["alphaOX_err"] == pytest.approx(0.03670891022048923, abs=1e-12)
    assert row["alphaOX_exp_err"] == pytest.approx(0.006160000000000001, abs=1e-12)
    assert row["delta_alphaOX_err"] == pytest.approx(
        0.034587787230010145,
        abs=1e-12,
    )
    observed_log_l2500 = rest_frame_ab_magnitude_to_log_lnu(
        source.loc[0, "apparent_mag_2500"],
        source.loc[0, "z"],
        cosmology=REFERENCE_COSMOLOGY,
    )
    assert row["log_L2500_nu"] != pytest.approx(observed_log_l2500)
    assert "log_L2500_int_fs" not in out.columns
    assert "log_Lxray" not in out.columns


def test_compute_alpha_ox_uses_supplied_cosmology():
    source = pd.DataFrame(
        {
            "z": [1.0],
            "m_2500_dereddened": [20.0],
            "m_2500_dereddened_err": [0.1],
            "flux_aper_b": [1.0e-14],
            "flux_aper_err_b": [2.0e-15],
        }
    )
    alternate_cosmology = FlatLambdaCDM(H0=73, Om0=0.4)

    reference = compute_alpha_ox(source, cosmology=REFERENCE_COSMOLOGY).iloc[0]
    alternate = compute_alpha_ox(source, cosmology=alternate_cosmology).iloc[0]

    assert alternate["log_L2500_nu"] != pytest.approx(reference["log_L2500_nu"])
    assert alternate["log_L2keV_nu"] != pytest.approx(reference["log_L2keV_nu"])
    assert alternate["alphaOX"] == pytest.approx(reference["alphaOX"], abs=1e-12)
    assert alternate["delta_alphaOX"] != pytest.approx(reference["delta_alphaOX"])


def test_zero_xray_flux_remains_excluded_without_infinite_values():
    source = pd.DataFrame(
        {
            "z": [1.0],
            "m_2500_dereddened": [20.0],
            "m_2500_dereddened_err": [0.1],
            "flux_aper_b": [0.0],
            "flux_aper_err_b": [0.0],
        }
    )

    out = compute_alpha_ox(source, cosmology=REFERENCE_COSMOLOGY)

    assert out["log_L2keV_nu"].isna().all()
    assert out["alphaOX"].isna().all()
    assert not np.isinf(
        out.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    ).any()


def test_compute_alpha_ox_requires_intrinsic_uv_measurements():
    source = pd.DataFrame(
        {
            "z": [1.0],
            "apparent_mag_2500": [20.0],
            "apparent_mag_2500_err": [0.1],
            "flux_aper_b": [1.0e-14],
            "flux_aper_err_b": [2.0e-15],
        }
    )

    with pytest.raises(ValueError, match="m_2500_dereddened"):
        compute_alpha_ox(source, cosmology=REFERENCE_COSMOLOGY)
