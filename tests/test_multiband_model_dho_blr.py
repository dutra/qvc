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

from multiband_model_dho_blr import (
    ContiBLR_SHO_Wrapper,
    OverdampedSHOBaseQS,
    make_multiband_dho_blr_model,
)


def test_make_multiband_dho_blr_model_random_data():
    rng = np.random.default_rng(42)
    n_band = 3
    n_obs = 36

    t = np.sort(rng.uniform(0.0, 100.0, size=n_obs))
    band = rng.integers(0, n_band, size=n_obs)
    yerr = rng.uniform(0.01, 0.05, size=n_obs)
    y = rng.normal(0.0, yerr, size=n_obs)

    model = make_multiband_dho_blr_model(
        X=(jnp.asarray(t), jnp.asarray(band)),
        y=jnp.asarray(y),
        yerr=jnp.asarray(yerr),
        n_band=n_band,
        zero_mean=True,
        has_jitter=True,
    )

    assert model is not None
    assert hasattr(model, "log_prob")
    assert hasattr(model, "pred")

    tau_fast = jnp.asarray(rng.uniform(5.0, 15.0, size=n_band))
    tau_slow = tau_fast + jnp.asarray(rng.uniform(20.0, 80.0, size=n_band))
    base_kernel = OverdampedSHOBaseQS(tau_fast=tau_fast, tau_slow=tau_slow)
    params = {
        "amp_cont": jnp.asarray(rng.uniform(0.1, 0.4, size=n_band)),
        "amp_blr": jnp.asarray(rng.uniform(0.01, 0.08, size=n_band)),
        "lag_disk": jnp.asarray(rng.uniform(0.5, 5.0, size=n_band)),
        "lag_blr": jnp.asarray(rng.uniform(5.0, 30.0, size=n_band)),
    }
    wrapper = ContiBLR_SHO_Wrapper(kernel=base_kernel, params=params)

    design = base_kernel.design_matrix()
    stationary = base_kernel.stationary_covariance()
    transition = wrapper.transition_matrix((jnp.array(0.0), jnp.array(1)), (jnp.array(3.0), jnp.array(1)))
    obs = wrapper.observation_model((jnp.array(0.0), jnp.array(1)))

    assert design.shape == (2 * n_band, 2 * n_band)
    assert stationary.shape == (2 * n_band, 2 * n_band)
    assert transition.shape == (2 * n_band, 2 * n_band)
    assert obs.shape == (2 * n_band,)
    assert np.all(np.isfinite(np.asarray(design)))
    assert np.all(np.isfinite(np.asarray(stationary)))
    assert np.all(np.isfinite(np.asarray(transition)))
    assert np.all(np.isfinite(np.asarray(obs)))
