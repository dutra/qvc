import numpy as np

from qvc.light_curve.multiband_fit_plotting import _predict_regular_band_grid


class _RecordingPredictionModel:
    def __init__(self):
        self.calls = []

    def pred(self, params, X_test):
        times, bands = (np.asarray(value) for value in X_test)
        self.calls.append((params, times, bands))
        return times + 10.0 * bands, np.ones_like(times)


def test_predict_regular_band_grid_uses_one_band_per_call():
    model = _RecordingPredictionModel()
    times = np.array([100.0, 102.0, 105.0])
    bands = np.array([0, 2, 4])

    mean, std = _predict_regular_band_grid(
        model,
        {"parameter": 1.0},
        times,
        bands,
        time0=100.0,
    )

    assert len(model.calls) == bands.size
    for call, band in zip(model.calls, bands):
        _, query_times, query_bands = call
        assert np.array_equal(query_times, times - 100.0)
        assert np.array_equal(query_bands, np.full(times.size, band))
    assert mean.shape == (times.size, bands.size)
    assert std.shape == mean.shape
    assert np.array_equal(mean[:, 0], times - 100.0)
    assert np.array_equal(mean[:, 1], times - 100.0 + 20.0)
    assert np.array_equal(mean[:, 2], times - 100.0 + 40.0)


def test_predict_regular_band_grid_splits_large_queries_without_changing_output():
    model = _RecordingPredictionModel()
    times = np.array([100.0, 102.0, 105.0])
    bands = np.array([0, 2, 4])

    mean, std = _predict_regular_band_grid(
        model,
        {"parameter": 1.0},
        times,
        bands,
        time0=100.0,
        max_query_points=2,
    )

    assert len(model.calls) == 6
    assert all(call[1].size <= 2 for call in model.calls)
    assert mean.shape == (times.size, bands.size)
    assert std.shape == mean.shape
    assert np.array_equal(mean[:, 0], times - 100.0)
    assert np.array_equal(mean[:, 1], times - 100.0 + 20.0)
    assert np.array_equal(mean[:, 2], times - 100.0 + 40.0)
