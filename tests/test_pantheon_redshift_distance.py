import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from qvc.hubble.hubble_likelihood import (
    log_likelihood_pantheon_cephdist,
    pantheon_distance_modulus,
)


def test_pantheon_distance_modulus_matches_astropy_when_redshifts_match():
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    z = np.array([0.015, 0.1, 0.7, 1.5])

    actual = pantheon_distance_modulus(cosmo, z, z)

    np.testing.assert_allclose(
        actual,
        cosmo.distmod(z).value,
        rtol=0.0,
        atol=1e-12,
    )


def test_pantheon_distance_modulus_has_expected_heliocentric_shift():
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    z_hd = np.array([0.015, 0.1, 0.7])
    z_hel = np.array([0.014, 0.102, 0.69])

    actual = pantheon_distance_modulus(cosmo, z_hd, z_hel)
    standard = cosmo.distmod(z_hd).value
    expected_shift = 5.0 * np.log10((1.0 + z_hel) / (1.0 + z_hd))

    np.testing.assert_allclose(
        actual - standard,
        expected_shift,
        rtol=0.0,
        atol=1e-12,
    )


def test_numpy_pantheon_likelihood_requires_zhel():
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    pantheon_data = {
        "zHD": np.array([0.1]),
        "m_b_corr": np.array([19.0]),
        "IS_CALIBRATOR": np.array([False]),
        "CEPH_DIST": np.array([-9.0]),
        "MU_SH0ES_ERR_DIAG": np.array([0.1]),
    }

    with pytest.raises(KeyError, match="requires the zHEL field"):
        log_likelihood_pantheon_cephdist(
            {"M0_sn": -19.2},
            pantheon_data,
            None,
            True,
            None,
            cosmo,
            False,
        )


def test_numpy_pantheon_likelihood_uses_mixed_redshift_distance():
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    z_hd = np.array([0.02, 0.08, 0.2])
    z_hel = np.array([0.018, 0.083, 0.19])
    m0_sn = -19.2
    sigma = np.full(z_hd.shape, 0.1)
    mixed_mu = pantheon_distance_modulus(cosmo, z_hd, z_hel)
    pantheon_data = {
        "zHD": z_hd,
        "zHEL": z_hel,
        "m_b_corr": mixed_mu + m0_sn,
        "IS_CALIBRATOR": np.zeros(z_hd.shape, dtype=bool),
        "CEPH_DIST": np.full(z_hd.shape, -9.0),
        "MU_SH0ES_ERR_DIAG": sigma,
    }

    actual = log_likelihood_pantheon_cephdist(
        {"M0_sn": m0_sn},
        pantheon_data,
        None,
        True,
        None,
        cosmo,
        False,
    )
    expected = np.sum(-np.log(sigma) - 0.5 * np.log(2.0 * np.pi))

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_jax_pantheon_distance_matches_numpy_mixed_redshifts():
    pytest.importorskip("jax")
    from qvc.hubble.hubble_fit_jax import (
        _distance_modulus_from_redshifts_jax,
    )

    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    z_hd = np.array([0.015, 0.1, 0.7, 1.5])
    z_hel = np.array([0.014, 0.102, 0.69, 1.49])
    params = {"H0": 70.0, "Om0": 0.3}

    actual, _ = _distance_modulus_from_redshifts_jax(
        z_hd,
        z_hel,
        params,
        "FlatLambdaCDM",
        1.0,
    )
    expected = pantheon_distance_modulus(cosmo, z_hd, z_hel)

    np.testing.assert_allclose(
        np.asarray(actual),
        expected,
        rtol=0.0,
        atol=2e-5,
    )


def test_jax_pantheon_array_packing_requires_zhel():
    pytest.importorskip("jax")
    from qvc.hubble.hubble_fit_jax import _prepare_pantheon_arrays

    pantheon_data = {
        "zHD": np.array([0.05, 0.1]),
        "m_b_corr": np.array([16.0, 17.0]),
        "IS_CALIBRATOR": np.array([False, False]),
        "CEPH_DIST": np.array([-9.0, -9.0]),
        "MU_SH0ES_ERR_DIAG": np.array([0.1, 0.1]),
    }

    with pytest.raises(KeyError, match="zHEL"):
        _prepare_pantheon_arrays(
            pantheon_data,
            np.eye(2),
            True,
            0.0,
        )
