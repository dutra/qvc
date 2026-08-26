from dataclasses import FrozenInstanceError

import h5py
import numpy as np
import pytest

from qvc.hubble.fitted_color_completeness import (
    COLOR_STRENGTH_PARAMETER,
    ColorParentSupportError,
    FittedColorConfig,
    QsogenColorParentCache,
    aligned_fitted_color_percentiles_2d,
    aligned_fitted_color_percentiles_3d,
    bounded_color_completeness,
    bounded_color_completeness_xp,
    color_parameter_prior_spec,
    color_relative_selection_factor,
    color_relative_selection_factor_xp,
    deterministic_color_draw_indices,
    fitted_color_config_hash,
    fitted_color_provenance,
    fitted_psf_g_minus_i,
    fixed_reference_host_fraction_quadrature,
    log_mean_color_relative_factor,
    read_qsogen_color_parent_cache,
    select_deterministic_color_draws,
    write_qsogen_color_parent_cache,
)


def _provenance():
    return {
        "construction": "synthetic_qsogen_test",
        "qsogen_commit": "d2f9abf1ad23c489da8857f7e3c1bca862105d22",
        "qsogen_source_url": "https://github.com/MJTemple/qsogen",
        "qsogen_license": "MIT",
        "jaxsedfit_filter_commit": "bc9da74735260bd33b3da2076fd7929fdd592e0d",
        "asset_sha256": {"synthetic": "0" * 64},
        "filter_names": ["g_sdss", "i_sdss"],
        "reference_cosmology": {"H0_km_s_Mpc": 70.0, "Omega_m": 0.3},
        "magnitude_state": "attenuation_retaining",
        "source_internal_attenuation": "qsogen mean EBV zero",
        "host_scaling": "exact synthetic fhost at 2500",
        "m2500_to_qsogen_luminosity": "synthetic explicit conversion",
        "parent_population_interpretation": (
            "selected DR16Q median template, not an unbiased parent"
        ),
        "residual_scatter_limitation": (
            "symmetric Gaussian does not model an asymmetric dust-red tail"
        ),
    }


def _cache(*, host_shift=0.8, fhost_step=0.002):
    magnitude = np.array([18.5, 21.25, 24.0])
    redshift = np.array([0.44, 1.80, 3.16])
    f_host = np.linspace(0.0, 1.0, int(round(1.0 / fhost_step)) + 1)
    m_mesh, z_mesh = np.meshgrid(magnitude, redshift, indexing="ij")
    agn = 0.04 * (m_mesh - 21.25) + 0.1 * (z_mesh - 1.8)
    total = agn[..., None] + host_shift * f_host
    return QsogenColorParentCache(
        magnitude,
        redshift,
        f_host,
        agn,
        total,
        _provenance(),
    )


def test_draw_indices_are_fixed_midpoints_and_selection_is_aligned():
    indices = deterministic_color_draw_indices()
    np.testing.assert_array_equal(indices, np.arange(2, 64, 4))
    values = np.arange(2 * 64).reshape(2, 64)
    np.testing.assert_array_equal(
        select_deterministic_color_draws(values), values[:, indices]
    )
    with pytest.raises(ValueError, match="Expected 64 draws"):
        select_deterministic_color_draws(values[:, :-1])


def test_response_is_bounded_marginal_preserving_and_has_clear_sign():
    base = np.linspace(0.0, 1.0, 19)[:, None]
    percentile = np.linspace(0.0, 1.0, 1001)[None, :]
    response = bounded_color_completeness(base, percentile, 1.0)
    assert np.min(response) >= 0.0
    assert np.max(response) <= 1.0
    np.testing.assert_allclose(np.mean(response, axis=1), base[:, 0], atol=2e-16)

    blue, red = bounded_color_completeness(0.6, [0.0, 1.0], 0.8)
    assert blue > 0.6 > red
    np.testing.assert_array_equal(
        bounded_color_completeness(base, percentile, 0.0),
        np.broadcast_to(base, (19, 1001)),
    )


def test_relative_factor_rejects_selected_zero_base_and_handles_one_exactly():
    with pytest.raises(ValueError, match="strictly positive"):
        color_relative_selection_factor(0.0, 0.5, 0.2)
    np.testing.assert_array_equal(
        color_relative_selection_factor(np.ones(3), [0.0, 0.5, 1.0], 1.0),
        np.ones(3),
    )
    assert log_mean_color_relative_factor(
        [0.5, 0.5], [0.25, 0.75], 0.9
    ) == pytest.approx(0.0, abs=1e-15)


def test_single_parameter_prior_config_hash_and_provenance_are_stable(tmp_path):
    prior = color_parameter_prior_spec()
    assert prior == {
        COLOR_STRENGTH_PARAMETER: {
            "distribution": "uniform",
            "low": -1.0,
            "high": 1.0,
        }
    }
    digest = "a" * 64
    first = FittedColorConfig(str(tmp_path / "a.h5"), digest)
    second = FittedColorConfig(str(tmp_path / "moved.h5"), digest)
    assert fitted_color_config_hash(first) == fitted_color_config_hash(second)
    provenance = fitted_color_provenance(first)
    assert provenance["response"]["positive_s_color"] == (
        "redder_objects_are_less_complete"
    )
    assert "selected_DR16Q" in provenance["parent_interpretation"][
        "calibration_caveat"
    ]
    assert "dust_red_tail" in provenance["parent_interpretation"][
        "dust_tail_caveat"
    ]
    with pytest.raises(FrozenInstanceError):
        first.parent_sigma = 0.3
    with pytest.raises(ValueError, match="positive"):
        FittedColorConfig("parent.h5", digest, parent_sigma=0.0)


def test_numpy_and_jax_response_primitives_match():
    jnp = pytest.importorskip("jax.numpy")
    base = np.array([0.1, 0.5, 0.9])
    percentile = np.array([0.05, 0.4, 0.95])
    strength = -0.37
    np.testing.assert_allclose(
        np.asarray(
            bounded_color_completeness_xp(
                base, percentile, strength, xp=jnp
            )
        ),
        bounded_color_completeness(base, percentile, strength),
        rtol=2e-7,
    )
    np.testing.assert_allclose(
        np.asarray(
            color_relative_selection_factor_xp(
                base, percentile, strength, xp=jnp
            )
        ),
        color_relative_selection_factor(base, percentile, strength),
        rtol=2e-7,
    )


def test_parent_interpolation_and_conditional_percentiles():
    cache = _cache()
    m, z, host = 20.0, 1.2, 0.25
    mean = cache.total_mean_color(m, z, host)
    assert cache.percentile_3d(mean, m, z, host) == pytest.approx(0.5)
    assert cache.total_mean_color(m, z, 0.8) > cache.total_mean_color(m, z, 0.2)

    low_mean = cache.total_mean_color(m, z, 0.0)
    high_mean = cache.total_mean_color(m, z, 1.0)
    midpoint = 0.5 * (low_mean + high_mean)
    percentile = cache.percentile_2d(
        midpoint, m, z, [0.0, 1.0], [0.5, 0.5], sigma=0.2
    )
    assert percentile == pytest.approx(0.5, abs=1e-14)

    with pytest.raises(ColorParentSupportError, match="strict support"):
        cache.total_mean_color(24.01, z, host)
    with pytest.raises(ValueError, match="nonnegative"):
        cache.percentile_2d(0.0, m, z, [0.0, 1.0], [1.0, -1.0])


def test_cache_round_trip_and_tamper_detection(tmp_path):
    cache = _cache()
    path = tmp_path / "parent.h5"
    write_qsogen_color_parent_cache(path, cache)
    loaded = read_qsogen_color_parent_cache(
        path, expected_content_hash=cache.content_hash_sha256
    )
    assert loaded.content_hash_sha256 == cache.content_hash_sha256
    np.testing.assert_array_equal(
        loaded.total_mean_g_minus_i, cache.total_mean_g_minus_i
    )
    config = FittedColorConfig.from_parent_file(path, parent_sigma=0.25)
    assert config.parent_cache_sha256 == cache.content_hash_sha256
    assert config.parent_sigma == 0.25

    with h5py.File(path, "r+") as handle:
        handle["mean_color/total_g_minus_i"][1, 1, 1] += 0.1
    with pytest.raises(ValueError, match="content hash"):
        read_qsogen_color_parent_cache(path)


def test_active_config_rejects_a_coarse_parent_even_with_a_valid_hash(tmp_path):
    path = tmp_path / "coarse_parent.h5"
    write_qsogen_color_parent_cache(path, _cache(fhost_step=0.05))
    # General inspection remains possible, but an inference config cannot
    # silently reuse this scientifically unconverged parent surface.
    read_qsogen_color_parent_cache(path)
    with pytest.raises(ValueError, match="too coarse.*Regenerate the v2"):
        FittedColorConfig.from_parent_file(path)


def test_fitted_psf_color_uses_only_positive_g_and_i_fluxes():
    flux = np.array([[8.0, 4.0, 3.0, 1.0, 0.5]])
    assert fitted_psf_g_minus_i(flux) == pytest.approx(
        [-2.5 * np.log10(4.0)]
    )
    assert fitted_psf_g_minus_i(
        np.array([[4.0, 1.0]]), bands=["g_sdss", "i_sdss"]
    ) == pytest.approx([-2.5 * np.log10(4.0)])
    with pytest.raises(ValueError, match="positive"):
        fitted_psf_g_minus_i(np.array([[1.0, 0.0]]), bands=["g", "i"])


def test_fixed_host_quadrature_is_deterministic_and_fainter_means_more_host():
    model = {
        "x0": 45.0,
        "k": 2.0,
        "nu": 1.0,
        "sigma_host_logit": 0.4,
        "clip_eps": 1e-6,
    }
    nodes, weights = fixed_reference_host_fraction_quadrature(
        [19.0, 23.0], [1.0, 1.0], model
    )
    assert nodes.shape == (2, 12)
    assert weights.shape == (12,)
    assert np.sum(weights) == pytest.approx(1.0)
    assert np.sum(nodes[1] * weights) > np.sum(nodes[0] * weights)
    assert np.all((nodes > 0.0) & (nodes < 1.0))


def test_aligned_helpers_select_the_same_16_fitted_draws():
    cache = _cache(host_shift=0.0)
    flux = np.ones((2, 64, 5))
    magnitude = np.full((2, 64), 21.25)
    redshift = np.array([1.8, 1.8])
    host = np.full((2, 64), 0.3)
    q3 = aligned_fitted_color_percentiles_3d(
        cache, flux, magnitude, redshift, host
    )
    assert q3.shape == (2, 16)
    np.testing.assert_allclose(q3, 0.5)

    host_model = {
        "x0": 45.0,
        "k": 2.0,
        "nu": 1.0,
        "sigma_host_logit": 0.4,
    }
    q2 = aligned_fitted_color_percentiles_2d(
        cache, flux, magnitude, redshift, host_model
    )
    assert q2.shape == (2, 16)
    np.testing.assert_allclose(q2, 0.5)
