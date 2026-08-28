import numpy as np
import pytest

from qvc.light_curve.multiband_generate_lc import (
    inverse_variance_weighted_mean,
    rolling_photometric_outlier_mask,
)


def test_rolling_outlier_mask_flags_center_without_clipping_neighbors():
    mags = np.zeros(21)
    magerrs = np.full(21, 0.02)
    times = np.arange(21, dtype=float)
    mags[10] = 1.0

    rejected = rolling_photometric_outlier_mask(times, mags, magerrs)

    expected = np.zeros(21, dtype=bool)
    expected[10] = True
    np.testing.assert_array_equal(rejected, expected)


def test_rolling_outlier_mask_uses_truncated_edge_window_when_supported():
    mags = np.zeros(15)
    magerrs = np.full(15, 0.02)
    times = np.arange(15, dtype=float)
    mags[2] = 1.0
    mags[-3] = -1.0

    rejected = rolling_photometric_outlier_mask(times, mags, magerrs)

    assert rejected[2]
    assert rejected[-3]


def test_rolling_outlier_mask_keeps_under_supported_extreme_edges():
    mags = np.zeros(15)
    magerrs = np.full(15, 0.02)
    times = np.concatenate(([0.0], np.arange(100.0, 113.0), [200.0]))
    mags[0] = 1.0
    mags[-1] = -1.0

    rejected = rolling_photometric_outlier_mask(times, mags, magerrs)

    assert not rejected[0]
    assert not rejected[-1]


def test_rolling_outlier_mask_accepts_custom_half_window():
    times = np.concatenate(([0.0], np.arange(40.0, 48.0)))
    mags = np.zeros(times.size)
    mags[0] = 1.0
    magerrs = np.full(times.size, 0.02)

    default_rejected = rolling_photometric_outlier_mask(times, mags, magerrs)
    wider_rejected = rolling_photometric_outlier_mask(
        times,
        mags,
        magerrs,
        half_window_days=60.0,
    )

    assert not default_rejected[0]
    assert wider_rejected[0]


@pytest.mark.parametrize("half_window_days", [0.0, -1.0, np.nan])
def test_rolling_outlier_mask_rejects_invalid_half_window(half_window_days):
    with pytest.raises(ValueError, match="half_window_days must be positive"):
        rolling_photometric_outlier_mask(
            [0.0],
            [0.0],
            [0.1],
            half_window_days=half_window_days,
        )


def test_inverse_variance_weighted_mean_and_formal_error():
    mags = np.array([20.0, 21.0])
    magerrs = np.array([0.1, 0.2])

    mean, mean_err = inverse_variance_weighted_mean(mags, magerrs)

    assert np.isclose(mean, 20.2)
    assert np.isclose(mean_err, 1.0 / np.sqrt(125.0))


def test_inverse_variance_weighted_mean_ignores_invalid_and_nonpositive_errors():
    mean, mean_err = inverse_variance_weighted_mean(
        [20.0, 99.0, 88.0, np.nan],
        [0.1, 0.0, -1.0, 0.2],
    )

    assert mean == 20.0
    assert mean_err == 0.1
