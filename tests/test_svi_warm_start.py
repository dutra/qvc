import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from qvc.light_curve import fit_light_curves as fit_lc


def test_progress_svi_compiles_update_once(monkeypatch):
    python_update_calls = 0
    original_update = fit_lc.SVI.update

    def counted_update(self, *args, **kwargs):
        nonlocal python_update_calls
        python_update_calls += 1
        return original_update(self, *args, **kwargs)

    def model():
        numpyro.sample("location", dist.Normal(0.0, 1.0))

    monkeypatch.setattr(fit_lc.SVI, "update", counted_update)
    monkeypatch.setattr(fit_lc, "tqdm", lambda values, **_kwargs: values)

    init_values, final_loss = fit_lc.run_svi_warm_start(
        model,
        jax.random.PRNGKey(3),
        num_steps=4,
        learning_rate=1e-2,
        progress_bar=True,
    )

    assert python_update_calls == 1
    assert np.isfinite(final_loss)
    assert np.isfinite(np.asarray(init_values["location"])).all()


def test_progress_svi_matches_compiled_loop_result(monkeypatch):
    observations = jnp.asarray([-0.4, 0.1, 0.3, 0.8])

    def model():
        location = numpyro.sample("location", dist.Normal(0.0, 1.0))
        numpyro.sample("observations", dist.Normal(location, 0.5), obs=observations)

    monkeypatch.setattr(fit_lc, "tqdm", lambda values, **_kwargs: values)
    kwargs = {
        "num_steps": 8,
        "learning_rate": 1e-2,
    }
    progress_values, progress_loss = fit_lc.run_svi_warm_start(
        model,
        jax.random.PRNGKey(8),
        progress_bar=True,
        **kwargs,
    )
    loop_values, loop_loss = fit_lc.run_svi_warm_start(
        model,
        jax.random.PRNGKey(8),
        progress_bar=False,
        **kwargs,
    )

    np.testing.assert_allclose(progress_loss, loop_loss, rtol=1e-12, atol=1e-12)
    assert progress_values.keys() == loop_values.keys()
    for name in progress_values:
        np.testing.assert_allclose(
            np.asarray(progress_values[name]),
            np.asarray(loop_values[name]),
            rtol=1e-12,
            atol=1e-12,
        )
