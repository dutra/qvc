import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

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

    init_values, final_loss, summary_dict = fit_lc.run_svi_warm_start(
        model,
        jax.random.PRNGKey(3),
        num_steps=4,
        learning_rate=1e-2,
        progress_bar=True,
    )

    assert python_update_calls == 1
    assert np.isfinite(final_loss)
    assert np.isfinite(np.asarray(init_values["location"])).all()
    assert "location" in summary_dict


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
    progress_values, progress_loss, progress_summary = fit_lc.run_svi_warm_start(
        model,
        jax.random.PRNGKey(8),
        progress_bar=True,
        **kwargs,
    )
    loop_values, loop_loss, loop_summary = fit_lc.run_svi_warm_start(
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
    assert progress_summary.keys() == loop_summary.keys()


def test_svi_guide_summary_reports_all_statistics_and_nonfinite_fraction():
    summary = fit_lc.summarize_svi_guide_samples(
        {
            "theta": np.array([1.0, 2.0, np.nan, 3.0]),
            "vector": np.array([[1.0, np.nan], [3.0, np.nan]]),
        }
    )

    assert list(summary["theta"]) == [
        "mean",
        "std",
        "median",
        "5.0%",
        "95.0%",
        "finite%",
    ]
    assert summary["theta"]["mean"] == 2.0
    assert summary["theta"]["median"] == 2.0
    assert summary["theta"]["finite%"] == 75.0
    np.testing.assert_allclose(summary["vector"]["finite%"], [100.0, 0.0])
    assert np.isnan(summary["vector"]["mean"][1])


@pytest.mark.parametrize(
    ("final_loss", "init_values", "message"),
    [
        (np.nan, {"theta": np.array(1.0)}, "non-finite final loss"),
        (1.0, {"theta": np.array(np.nan)}, "non-finite guide median values for theta"),
    ],
)
def test_svi_report_refuses_nonfinite_nuts_initialization(
    final_loss,
    init_values,
    message,
    capsys,
):
    with pytest.raises(FloatingPointError, match=message):
        fit_lc.print_and_validate_svi_warm_start(
            "object-1",
            refinement_index=1,
            refinement_iters=2,
            final_loss=final_loss,
            init_values=init_values,
            summary_dict={
                "theta": {
                    "mean": np.array(np.nan),
                    "std": np.array(np.nan),
                    "median": np.array(np.nan),
                    "5.0%": np.array(np.nan),
                    "95.0%": np.array(np.nan),
                    "finite%": np.array(0.0),
                }
            },
        )

    output = capsys.readouterr().out
    assert "SVI warm-start summary for refinement 1/2" in output
    assert "final_loss:" in output
    assert "nonfinite_guide_medians:" in output
