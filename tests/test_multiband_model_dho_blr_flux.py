import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.light_curve.multiband_model_dho_blr_flux import (
    mag_to_relative_flux,
    magerr_to_relative_fluxerr,
    make_multiband_dho_blr_flux_model,
    relative_flux_to_mag,
)


def _make_params(n_band):
    return {
        "mean": jnp.zeros(n_band, dtype=float),
        "poly1": jnp.array(0.0, dtype=float),
        "tau_fast_band": jnp.full(n_band, 8.0, dtype=float),
        "tau_slow_band": jnp.full(n_band, 60.0, dtype=float),
        "amp_cont": jnp.full(n_band, 0.2, dtype=float),
        "amp_blr": jnp.full(n_band, 0.05, dtype=float),
        "amp_blr2": jnp.full(n_band, 0.02, dtype=float),
        "amp_bc": jnp.full(n_band, 0.03, dtype=float),
        "lag_disk": jnp.full(n_band, 1.0, dtype=float),
        "lag_blr": jnp.full(n_band, 12.0, dtype=float),
        "lag_blr2": jnp.full(n_band, 20.0, dtype=float),
        "lag_bc": jnp.full(n_band, 4.0, dtype=float),
    }


def test_mag_flux_round_trip_and_error_propagation():
    mags = jnp.asarray([0.0, 0.2, -0.1], dtype=float)
    magerrs = jnp.asarray([0.01, 0.02, 0.03], dtype=float)

    flux = mag_to_relative_flux(mags)
    mags_round_trip = relative_flux_to_mag(flux)
    fluxerr = magerr_to_relative_fluxerr(mags, magerrs)

    np.testing.assert_allclose(np.asarray(mags_round_trip), np.asarray(mags), rtol=1e-10, atol=1e-10)
    assert np.all(np.isfinite(np.asarray(fluxerr)))
    assert np.all(np.asarray(fluxerr) > 0.0)


def test_flux_hybrid_zero_component_limit_returns_zero_residual_mag():
    t = jnp.asarray([0.0, 3.0, 7.0, 10.0], dtype=float)
    band = jnp.asarray([0, 0, 1, 1], dtype=jnp.int32)
    y = jnp.zeros_like(t)
    yerr = jnp.full_like(t, 0.03)

    model = make_multiband_dho_blr_flux_model((t, band), y, yerr, n_band=2, zero_mean=False, has_jitter=True)
    params = _make_params(2)
    params["amp_blr"] = jnp.zeros(2, dtype=float)
    params["amp_blr2"] = jnp.zeros(2, dtype=float)
    params["amp_bc"] = jnp.zeros(2, dtype=float)

    cov, x_aug, index = model.latent_covariance(params, include_bc=False, include_blr2=False)
    latent = jnp.zeros(cov.shape[0], dtype=float)
    out = model.total_flux_and_model_mag(
        params,
        latent,
        x_aug,
        index,
        include_bc=False,
        include_blr2=False,
        f_host_band=jnp.zeros(2, dtype=float),
    )

    np.testing.assert_allclose(np.asarray(out["model_mag"]), np.zeros(4, dtype=float), atol=1e-8)
    assert np.all(np.asarray(out["positive_flux"]))


def test_flux_hybrid_blr_flux_can_brighten_total_magnitude():
    t = jnp.asarray([0.0, 4.0, 8.0], dtype=float)
    band = jnp.asarray([0, 0, 0], dtype=jnp.int32)
    y = jnp.zeros_like(t)
    yerr = jnp.full_like(t, 0.03)

    model = make_multiband_dho_blr_flux_model((t, band), y, yerr, n_band=1, zero_mean=False, has_jitter=True)
    params = _make_params(1)
    params["amp_blr2"] = jnp.zeros(1, dtype=float)
    params["amp_bc"] = jnp.zeros(1, dtype=float)

    cov, x_aug, index = model.latent_covariance(params, include_bc=False, include_blr2=False)
    latent = jnp.zeros(cov.shape[0], dtype=float)
    latent = latent.at[index["blr"]].set(-0.25)

    out = model.total_flux_and_model_mag(
        params,
        latent,
        x_aug,
        index,
        include_bc=False,
        include_blr2=False,
        f_host_band=jnp.zeros(1, dtype=float),
    )

    assert np.all(np.isfinite(np.asarray(out["model_mag"])))
    assert np.any(np.asarray(out["model_mag"]) < 0.0)
