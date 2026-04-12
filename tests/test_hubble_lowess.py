import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble import hubble_completeness_refactored as hcr


def test_build_lowess_trend_1d_preserves_monotone_trend():
    z = np.linspace(0.4, 3.2, 80)
    y = -0.15 - 0.25 * z + 0.02 * np.sin(3.0 * z)
    trend = hcr.build_lowess_trend_1d(z, y, frac=0.25, it=1, min_points=10)
    y_fit = trend(z)

    assert np.all(np.isfinite(y_fit))
    assert np.nanmax(np.abs(y_fit - y)) < 0.08
    assert np.all(np.diff(y_fit) < 0.03)


def test_build_lowess_trend_1d_handles_duplicates_and_noise():
    rng = np.random.default_rng(123)
    z = np.repeat(np.linspace(0.5, 2.5, 15), 3)
    y = 0.2 * np.cos(z) + rng.normal(0.0, 0.05, size=z.size)
    yerr = np.full(z.size, 0.05)

    trend = hcr.build_lowess_trend_1d(z, y, yerr=yerr, frac=0.3, it=1, min_points=8)
    zq = np.linspace(0.45, 2.55, 60)
    yq = trend(zq)

    assert np.all(np.isfinite(yq))
    assert yq.shape == zq.shape


def test_build_lowess_trend_1d_constant_input_is_constant():
    z = np.linspace(0.2, 2.8, 30)
    y = np.full_like(z, -0.42)
    trend = hcr.build_lowess_trend_1d(z, y, frac=0.25, it=1, min_points=6)
    y_fit = trend(np.array([0.1, 0.8, 1.6, 3.5]))

    assert np.all(np.isfinite(y_fit))
    assert np.allclose(y_fit, -0.42, atol=1e-10)


def test_make_dm_function_returns_callable_and_clamps_edges():
    z = np.linspace(0.4, 3.0, 50)
    m = 20.0 + 0.5 * z
    dm = -0.1 - 0.3 * z + 0.02 * np.sin(2.5 * z)

    dm_interp = hcr.make_dm_function(m, z, dm, lowess_frac=0.25, lowess_it=1, lowess_min_points=8)

    pts = np.array([
        [0.2, 20.0],
        [0.8, 20.4],
        [1.7, 21.0],
        [3.5, 21.8],
    ])
    vals = dm_interp(pts)

    assert vals.shape == (4,)
    assert np.all(np.isfinite(vals))
    assert np.isclose(vals[0], dm_interp(np.array([[z.min(), 20.0]]))[0])
    assert np.isclose(vals[-1], dm_interp(np.array([[z.max(), 21.8]]))[0])


def test_make_dm_function_sparse_input_falls_back_cleanly():
    z = np.array([0.8, 1.6, 2.4], dtype=float)
    m = np.array([20.0, 20.5, 21.0], dtype=float)
    dm = np.array([-0.2, -0.45, -0.7], dtype=float)

    dm_interp = hcr.make_dm_function(m, z, dm, lowess_frac=0.25, lowess_it=1, lowess_min_points=10)
    vals = dm_interp(np.array([[1.0, 20.1], [2.0, 20.7]]))

    assert vals.shape == (2,)
    assert np.all(np.isfinite(vals))


def test_host_debias_interpolator_returns_nan_for_nonfinite_queries():
    z = np.linspace(0.5, 2.5, 12)
    m = 20.0 + 0.4 * z
    fhost = np.linspace(0.1, 0.6, z.size)
    dm = -0.2 - 0.15 * z + 0.05 * fhost

    dm_interp = hcr.make_dm_function(m, z, dm, f_host_2500_psf=fhost)
    pts = np.array(
        [
            [1.0, 20.4, 0.2],
            [np.nan, 20.5, 0.3],
            [1.8, np.inf, 0.4],
            [2.0, 20.8, np.nan],
        ]
    )

    vals = dm_interp(pts)
    vals_public = hcr.evaluate_dm_interp(
        dm_interp,
        pts[:, 0],
        pts[:, 1],
        f_host_2500_psf=pts[:, 2],
    )

    assert vals.shape == (4,)
    assert np.isfinite(vals[0])
    assert np.all(np.isnan(vals[1:]))
    assert np.allclose(vals_public, vals, equal_nan=True)
