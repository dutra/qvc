import numpy as np

from qvc.light_curve.fit_light_curves import rolling_photometric_outlier_mask


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
