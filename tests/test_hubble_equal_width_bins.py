import numpy as np
import pytest
from qvc.hubble.hubble_plotting import (
    _hubble_linear_bin_edges, _range_partitioned_weighted_bin_stats,
)


def test_sixteen_bins_and_outside_extension():
    edges = _hubble_linear_bin_edges([.01, .44, 3.16, 3.8, np.nan], (.44, 3.16))
    np.testing.assert_allclose(np.diff(edges), .17)
    fitted = edges[(edges >= .44) & (edges <= 3.16)]
    assert len(fitted) == 17
    assert fitted[0] == .44 and fitted[-1] == 3.16
    np.testing.assert_allclose((fitted[:-1] + fitted[1:]) / 2, .525 + .17*np.arange(16))
    assert edges[0] <= .01 and edges[-1] >= 3.8


def test_boundary_membership_and_fixed_midpoints():
    z = np.array([.43, .44, .45, 3.15, 3.16, 3.17])
    edges = _hubble_linear_bin_edges(z, (.44, 3.16))
    inside, outside = _range_partitioned_weighted_bin_stats(
        z, z, np.ones_like(z), edges, (.44, 3.16), min_count=1,
    )
    np.testing.assert_allclose(inside[0], [.525, 3.075])
    np.testing.assert_allclose(outside[0], [.355, 3.245])
    np.testing.assert_array_equal(inside[3], [2, 2])
    np.testing.assert_array_equal(outside[3], [1, 1])
    np.testing.assert_allclose(inside[1], [.445, 3.155])
    np.testing.assert_allclose(inside[2], 1/np.sqrt(2))


def test_other_range_and_no_finite_objects():
    edges = _hubble_linear_bin_edges([np.nan], (1., 5.))
    np.testing.assert_allclose(edges, np.linspace(1, 5, 17))
    for bounds in [(1, 1), (2, 1), (np.nan, 3), (1, np.inf)]:
        with pytest.raises(ValueError):
            _hubble_linear_bin_edges([], bounds)


def test_three_objects_displayed_but_two_omitted():
    z = np.array([.45,.46,.47,.62,.63,3.17,3.18,3.19])
    edges = _hubble_linear_bin_edges(z, (.44,3.16))
    inside, outside = _range_partitioned_weighted_bin_stats(
        z, np.ones(len(z)), np.ones(len(z)), edges, (.44,3.16), min_count=3,
        center='mid',
    )
    np.testing.assert_allclose(inside[0], [.525])
    np.testing.assert_allclose(outside[0], [3.245])
    np.testing.assert_array_equal(inside[3], [3])
    np.testing.assert_allclose(inside[2], [1/np.sqrt(3)])


def test_lowest_out_of_range_bin_can_be_omitted_without_shifting_grid():
    z = np.array([.11, .12, .13, .28, .29, .30, .45, .46, .47, 3.17, 3.18, 3.19])
    edges = _hubble_linear_bin_edges(z, (.44, 3.16))
    inside, outside = _range_partitioned_weighted_bin_stats(
        z,
        np.ones(len(z)),
        np.ones(len(z)),
        edges,
        (.44, 3.16),
        min_count=3,
        center="mid",
        drop_lowest_out_of_range_bin=True,
    )

    np.testing.assert_allclose(inside[0], [.525])
    np.testing.assert_allclose(outside[0], [.355, 3.245])
    np.testing.assert_array_equal(outside[3], [3, 3])
