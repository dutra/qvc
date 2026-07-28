import numpy as np
import pandas as pd
import pytest

from qvc.spectra.fit_spectra_jaxsedfit_joint import (
    ab_mag_to_mjy,
    add_qvc_psf_photometry,
    estimate_m2500_dereddened,
    load_saved_sed_photometry,
)


def _record_with_ugriz():
    record = {"object_id": "1452887"}
    for band in "ugriz":
        record[f"psf_mag_{band}"] = 20.0
        record[f"psf_mag_err_{band}"] = 0.1
    return record


def test_saved_sed_loader_normalizes_object_id_and_upper_limits(tmp_path):
    path = tmp_path / "sed.csv"
    pd.DataFrame(
        {
            "object_id": [1452887],
            "filter_name": ["W3"],
            "flux_mjy": [0.5],
            "flux_err_mjy": [0.1],
            "is_upper_limit": ["true"],
        }
    ).to_csv(path, index=False)

    result = load_saved_sed_photometry(path)

    assert result["source_id"].tolist() == ["1452887"]
    assert result["is_upper_limit"].tolist() == [True]


def test_qvc_psf_photometry_replaces_saved_sdss_and_includes_z():
    saved = pd.DataFrame(
        {
            "filter_name": ["g_sdss", "W2"],
            "flux_mjy": [99.0, 0.2],
            "flux_err_mjy": [1.0, 0.02],
            "is_upper_limit": [False, False],
        }
    )

    result = add_qvc_psf_photometry(_record_with_ugriz(), saved)

    assert set(result["filter_name"]) == {
        "u_sdss", "g_sdss", "r_sdss", "i_sdss", "z_sdss", "W2"
    }
    g_flux = result.loc[result["filter_name"] == "g_sdss", "flux_mjy"].item()
    assert np.isclose(g_flux, ab_mag_to_mjy(20.0))


def test_qvc_psf_photometry_requires_z_even_if_variability_fit_dropped_it():
    record = _record_with_ugriz()
    record["psf_mag_z"] = np.nan

    with pytest.raises(ValueError, match=r"missing/invalid: \['z'\]"):
        add_qvc_psf_photometry(record, pd.DataFrame())


def test_dereddened_m2500_uses_intrinsic_disk_and_both_attenuation_terms():
    samples = {
        "log_agn_amp": np.log(np.array([1.0e38, 1.1e38])),
        "pl_slope": np.array([-1.8, -1.8]),
        "pl_bend_loc": np.array([1000.0, 1000.0]),
        "pl_bend_width": np.array([10.0, 10.0]),
        "ebv_gal": np.array([0.02, 0.02]),
        "ebv_agn": np.array([0.03, 0.03]),
    }

    result = estimate_m2500_dereddened(samples, redshift=1.0)

    intrinsic = result["m_2500_dereddened_draws"]
    attenuated = result["m_2500_attenuated_model_draws"]
    a_gal = result["a_2500_galaxy_draws"]
    a_internal = result["a_2500_internal_draws"]
    assert np.all(np.isfinite(intrinsic))
    assert np.allclose(attenuated - intrinsic, a_gal + a_internal)
    assert np.all(a_gal > 0)
    assert np.all(a_internal > 0)
