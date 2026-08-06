import os
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.light_curve.fit_light_curves import (
    _weighted_quantile,
    _default_survey_labels_for_band,
    _compute_survey_offset_active_mask,
    balmer_continuum_weight,
    build_explicit_model_params,
    build_explicit_model_params_relflux,
    build_single_object_model_mag_flux_linearized,
    bending_power_law_psd,
    compute_band_igm_transmission,
    compute_flux_line_ratio_offsets,
    compute_igm_transmission_obs_wave,
    compute_igm_transmission_rest_wave,
    compute_lam_lya_suppression_rf,
    compute_structure_function_diagnostics,
    compute_band_adf,
    compute_g_band_residual_drift_diagnostics,
    compute_g_band_raw_drift_diagnostics,
    TAU_FAST_TO_SLOW_PRIOR_RATIO,
    log_tau_fast_center0_prior,
    log_tau_fast_separation_raw_prior,
    log_lag_blr_prior,
    relative_log_lag_blr_prior,
    ordered_log_tau_fast,
    linear_trend_prior,
    compute_parameter_kls,
    compute_object_adf_diagnostics,
    compute_lambda_center_rf,
    empirical_structure_function,
    fit_bending_power_law_psd,
    fit_fixed_slope_drw_psd,
    apply_resume_sample_save_policy,
    add_model_prediction_params,
    make_lc,
    posterior_median_mean_function,
    binned_loo_residual_pair_correlation,
    compute_loo_short_lag_residual_diagnostics,
    compute_hubble_chain_diagnostics,
    numeric_scalar_diagnostics,
    summarize_nuts_extra_fields,
)
from qvc.light_curve.multiband_model_dho_blr_erlang import (
    make_multiband_dho_blr_flux_linearized_erlang_model,
)
from qvc.light_curve.multiband_fit_plotting import (
    _corner_plot_labels,
    _trace_plot_labels,
    relative_to_2500_amplitude_scale,
)
from qvc.light_curve import multiband_fit_utils
from qvc.light_curve import fit_light_curves as fit_lc
from qvc.light_curve.multiband_fit_utils import lambda_pivot, log_single_pl, process_samples


def _make_raw_public(n_band):
    return {
        "log_sigma_uv": jnp.array(np.log(0.2)),
        "log_tau_uv": jnp.array(np.log(300.0)),
        "log_tau_fast_uv": jnp.array(np.log(30.0)),
        "eta_sigma": jnp.array(-0.55),
        "eta_tau": jnp.array(0.25),
        "log_igm_transmission_band": jnp.zeros((n_band,), dtype=float),
        "dlog_amp_blr": jnp.full((n_band,), -1.0),
        "dlog_amp_blr2": jnp.full((n_band,), -1.5),
        "log_lag_blr": jnp.full((n_band,), np.log(20.0)),
        "log_lag_blr2": jnp.full((n_band,), np.log(60.0)),
        "lag0": jnp.array(5.0),
        "lag_beta": jnp.array(4.0 / 3.0),
    }


def _make_object(z=1.6):
    bands = ["u", "g", "r", "i", "z"]
    return {
        "object_id": "obj",
        "z": z,
        "times": {band: np.array([0.0, 50.0], dtype=float) for band in bands},
        "mags": {band: np.array([20.0, 20.2], dtype=float) for band in bands},
        "magerrs": {band: np.array([0.05, 0.05], dtype=float) for band in bands},
        "cadence": {band: 5.0 for band in bands},
        "cadence_err": {band: 0.5 for band in bands},
        "number_points": {band: 2 for band in bands},
    }


@pytest.mark.parametrize(
    ("refinement_strategy", "expected_nuts_runs"),
    (("nuts_each", 3), ("svi_then_nuts", 1)),
)
def test_flux_linearized_refinement_strategy_controls_nuts_runs(
    monkeypatch,
    refinement_strategy,
    expected_nuts_runs,
):
    calls = {"svi": 0, "nuts": 0, "pseudo_params": []}
    obj = {
        "object_id": "test",
        "X": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        "y": np.array([0.0, 0.1], dtype=float),
        "yerr": np.array([0.05, 0.05], dtype=float),
        "survey_idx": np.array([0, 0], dtype=np.int32),
        "mags_means": np.array([20.0], dtype=float),
    }

    monkeypatch.setattr(
        fit_lc,
        "_build_mag_flux_linearized_model_for_fit",
        lambda fit_obj, *_args, **_kwargs: fit_obj,
    )

    def fake_svi(*_args, **_kwargs):
        calls["svi"] += 1
        return {"theta": np.array(float(10 + calls["svi"]))}, float(calls["svi"])

    def fake_nuts(*_args, **_kwargs):
        calls["nuts"] += 1
        samples = {"theta": np.array([100.0 + calls["nuts"], 102.0 + calls["nuts"]])}
        per_chain = {"theta": samples["theta"][None, :]}
        return samples, per_chain, {
            "accept_prob": 0.9,
            "num_divergences": 0,
            "elapsed_sec": 2.0,
        }

    monkeypatch.setattr(fit_lc, "run_svi_warm_start", fake_svi)
    monkeypatch.setattr(fit_lc, "_run_nuts_inference", fake_nuts)
    monkeypatch.setattr(
        fit_lc,
        "_model_params_at_values",
        lambda _model, _key, values: dict(values),
    )
    monkeypatch.setattr(
        fit_lc,
        "add_model_prediction_params",
        lambda samples, *_args, **_kwargs: dict(samples),
    )
    monkeypatch.setattr(
        fit_lc,
        "make_multiband_dho_blr_flux_linearized_erlang_model",
        lambda *_args, **_kwargs: object(),
    )

    def fake_pseudo(_obj, _model, params):
        calls["pseudo_params"].append(float(np.asarray(params["theta"])))
        step = len(calls["pseudo_params"])
        return np.full(2, step, dtype=float), np.full(2, 0.1, dtype=float)

    monkeypatch.setattr(fit_lc, "_flux_linearized_pseudo_data_from_prediction", fake_pseudo)

    samples, per_chain, _fit_obj, diagnostics = (
        fit_lc.run_iterated_mag_flux_linearized_inference(
            obj,
            np.array([2500.0]),
            np.array([np.log(0.01)]),
            rng_key=jax.random.PRNGKey(0),
            fit_method="svi+nuts",
            num_warmup=2,
            num_samples=2,
            num_chains=1,
            chain_method="sequential",
            progress_bar=False,
            dense_mass=True,
            max_tree_depth=4,
            svi_steps=2,
            svi_lr=1e-2,
                disable_lag_blr=True,
                refinement_strategy=refinement_strategy,
                refinement_iters=3,
            )
        )

    assert calls["svi"] == 3
    assert calls["nuts"] == expected_nuts_runs
    assert len(calls["pseudo_params"]) == 3
    assert diagnostics["flux_linearized_nuts_runs"] == expected_nuts_runs
    assert diagnostics["elapsed_sec"] == 2.0 * expected_nuts_runs
    assert np.isfinite(samples["theta"]).all()
    assert np.isfinite(per_chain["theta"]).all()
    if refinement_strategy == "svi_then_nuts":
        assert calls["pseudo_params"][:2] == [11.0, 12.0]
        assert diagnostics["flux_linearized_iter1_elapsed_sec"] == 0.0
        assert diagnostics["flux_linearized_iter2_elapsed_sec"] == 0.0


def test_svi_then_nuts_refinement_requires_svi_backend():
    with pytest.raises(ValueError, match="requires fit_method='svi\\+nuts'"):
        fit_lc.run_iterated_mag_flux_linearized_inference(
            {},
            np.array([2500.0]),
            np.array([np.log(0.01)]),
            rng_key=jax.random.PRNGKey(0),
            fit_method="nuts",
            num_warmup=2,
            num_samples=2,
            num_chains=1,
            chain_method="sequential",
            progress_bar=False,
            dense_mass=True,
            max_tree_depth=4,
            svi_steps=2,
            svi_lr=1e-2,
            refinement_strategy="svi_then_nuts",
        )


def test_hubble_chain_diagnostics_match_hubble_observable_transforms():
    rng = np.random.default_rng(8)
    log_sigma_uv = rng.normal(-1.3, 0.2, size=(4, 500))
    log_tau_uv = rng.normal(5.4, 0.3, size=(4, 500))

    diagnostics = compute_hubble_chain_diagnostics(
        {
            "log_sigma_uv": log_sigma_uv,
            "log_tau_uv": log_tau_uv,
        },
        np.array([1800.0, 2500.0, 3500.0]),
        z=1.7,
        model_variant="mag_flux_linearized_erlang",
    )

    expected_sigma = float(
        np.asarray(fit_lc.numpyro_split_gelman_rubin(log_sigma_uv / np.log(10.0)))
    )
    expected_tau = float(
        np.asarray(
            fit_lc.numpyro_split_gelman_rubin(
                log_tau_uv / np.log(10.0) - np.log10(2.7)
            )
        )
    )
    assert diagnostics["log_sigma_uv_rhat"] == expected_sigma
    assert diagnostics["log_tau_uv_rf_rhat"] == expected_tau
    assert set(diagnostics) == {
        "log_sigma_uv_rhat",
        "log_sigma_uv_ess",
        "log_tau_uv_rf_rhat",
        "log_tau_uv_rf_ess",
    }


def test_hubble_chain_diagnostics_remain_available_for_one_chain():
    rng = np.random.default_rng(9)
    log_sigma_uv = rng.normal(-1.3, 0.2, size=(1, 500))
    log_tau_uv = rng.normal(5.4, 0.3, size=(1, 500))

    diagnostics = compute_hubble_chain_diagnostics(
        {
            "log_sigma_uv": log_sigma_uv,
            "log_tau_uv": log_tau_uv,
        },
        np.array([1800.0, 2500.0, 3500.0]),
        z=1.7,
        model_variant="mag_flux_linearized_erlang",
    )

    assert np.isnan(diagnostics["log_sigma_uv_rhat"])
    assert np.isnan(diagnostics["log_tau_uv_rf_rhat"])
    assert diagnostics["log_sigma_uv_ess"] == float(
        np.asarray(fit_lc.numpyro_effective_sample_size(log_sigma_uv / np.log(10.0)))
    )
    assert np.isfinite(diagnostics["log_tau_uv_rf_ess"])


def test_print_hubble_convergence_diagnostics_includes_multichain_and_one_chain_health(
    capsys,
):
    assert hasattr(fit_lc, "print_hubble_convergence_diagnostics")

    fit_lc.print_hubble_convergence_diagnostics(
        "multi",
        {
            "log_sigma_uv_rhat": 1.026,
            "log_sigma_uv_ess": 236.9,
            "log_tau_uv_rf_rhat": 1.028,
            "log_tau_uv_rf_ess": 179.0,
            "sampler_nchains": 4,
            "sampler_nsamp": 250,
            "sampler_nwarm": 250,
            "sampler_target_accept": 0.7,
            "sampler_max_tree_depth": 8,
            "accept_prob": 0.724,
            "num_divergences": 24,
            "nuts_step_size_min": 0.135,
            "nuts_num_steps_median": 63.0,
            "nuts_max_tree_depth_fraction_worst_chain": 0.076,
        },
    )
    multi_output = capsys.readouterr().out
    assert "[multi] Hubble-input convergence diagnostics (4 chains x 250 draws, 250 warmup):" in multi_output
    assert "log_sigma_uv: R-hat=1.026, ESS=236.9" in multi_output
    assert "log_tau_uv_rf: R-hat=1.028, ESS=179.0" in multi_output
    assert (
        "NUTS health: accept=0.724 (target=0.700), divergences=24, "
        "min_step=0.135, median_steps=63.0, "
        "max_depth=7.6% worst chain (depth=8)"
    ) in multi_output

    fit_lc.print_hubble_convergence_diagnostics(
        "single",
        {
            "log_sigma_uv_rhat": np.nan,
            "log_sigma_uv_ess": 83.2,
            "log_tau_uv_rf_rhat": np.nan,
            "log_tau_uv_rf_ess": 71.4,
            "sampler_nchains": 1,
            "sampler_nsamp": 250,
            "sampler_nwarm": 250,
            "accept_prob": 0.81,
            "num_divergences": 0,
        },
    )
    single_output = capsys.readouterr().out
    assert "1 chain x 250 draws, 250 warmup; R-hat unavailable" in single_output
    assert "log_sigma_uv: R-hat=n/a, ESS=83.2" in single_output
    assert "log_tau_uv_rf: R-hat=n/a, ESS=71.4" in single_output
    assert "NUTS health: accept=0.810, divergences=0" in single_output
    assert "nan" not in single_output.lower()
    assert "converged=" not in single_output.lower()
    assert "pass=" not in single_output.lower()

    fit_lc.print_hubble_convergence_diagnostics("missing", {})
    missing_output = capsys.readouterr().out
    assert "[missing] Hubble-input convergence diagnostics: not available" in missing_output

    fit_lc.print_hubble_convergence_diagnostics(
        "short",
        {
            "sampler_nchains": 1,
            "sampler_nsamp": 2,
            "accept_prob": 0.8,
            "num_divergences": 1,
            "nuts_step_size_min": 0.02,
        },
    )
    short_output = capsys.readouterr().out
    assert "[short] Hubble-input convergence diagnostics: not available" in short_output
    assert "NUTS health: accept=0.800, divergences=1, min_step=0.02" in short_output


def test_summarize_nuts_extra_fields_tracks_step_size_and_tree_depth():
    diagnostics = summarize_nuts_extra_fields(
        {
            "accept_prob": np.array([[0.8, 0.9], [0.7, 0.8]]),
            "diverging": np.array([[False, True], [False, False]]),
            "adapt_state.step_size": np.array([[0.01, 0.01], [0.02, 0.02]]),
            "num_steps": np.array([[1, 255], [63, 128]]),
        },
        max_tree_depth=8,
    )

    assert diagnostics["nuts_step_size_min"] == 0.01
    assert diagnostics["nuts_num_steps_median"] == 95.5
    assert diagnostics["nuts_max_tree_depth_fraction_worst_chain"] == 0.5
    assert diagnostics["num_divergences"] == 1
    assert set(diagnostics) == {
        "accept_prob",
        "num_divergences",
        "nuts_step_size_min",
        "nuts_num_steps_median",
        "nuts_max_tree_depth_fraction_worst_chain",
    }


def test_summarize_nuts_extra_fields_supports_one_chain():
    diagnostics = summarize_nuts_extra_fields(
        {
            "accept_prob": np.array([[0.8, 0.9, 0.85, 0.75]]),
            "diverging": np.array([[False, False, True, False]]),
            "adapt_state.step_size": np.full((1, 4), 0.012),
            "num_steps": np.array([[63, 128, 255, 31]]),
        },
        max_tree_depth=8,
    )

    assert np.isclose(diagnostics.pop("accept_prob"), 0.825)
    assert diagnostics == {
        "num_divergences": 1,
        "nuts_step_size_min": 0.012,
        "nuts_num_steps_median": 95.5,
        "nuts_max_tree_depth_fraction_worst_chain": 0.5,
    }


def test_numeric_scalar_diagnostics_excludes_arrays_and_strings():
    assert numeric_scalar_diagnostics(
        {
            "log_sigma_uv_rhat": np.float64(1.002),
            "sampler_nchains": 1,
            "unselected_numeric_diagnostic": 123.0,
            "acf": np.array([1.0, 0.5]),
            "strategy": "svi_then_nuts",
        }
    ) == {"log_sigma_uv_rhat": 1.002, "sampler_nchains": 1.0}


def test_model_params_at_values_materializes_deterministic_sites():
    def model():
        latent = fit_lc.numpyro.sample("latent", fit_lc.dist.Normal(0.0, 1.0))
        fit_lc.numpyro.deterministic("twice_latent", 2.0 * latent)
        fit_lc.numpyro.sample("observed", fit_lc.dist.Normal(latent, 1.0), obs=0.0)

    params = fit_lc._model_params_at_values(
        model,
        jax.random.PRNGKey(1),
        {"latent": np.array(1.25)},
    )

    assert set(params) == {"latent", "twice_latent"}
    assert np.isclose(params["latent"], 1.25)
    assert np.isclose(params["twice_latent"], 2.5)




def test_compute_lambda_center_rf_matches_geometric_mean():
    lam_rf = jnp.array([1500.0, 2400.0, 3600.0])
    expected = float(np.exp(np.mean(np.log(np.asarray(lam_rf)))))
    got = float(compute_lambda_center_rf(lam_rf))
    assert np.isclose(got, expected)


def test_relative_log_lag_blr_prior_retains_shifted_interval_support():
    z = 1.3
    log_lag0 = np.log(7.0)
    absolute = log_lag_blr_prior(z=z)
    relative = relative_log_lag_blr_prior(z=z, log_lag0=log_lag0)

    assert np.isclose(
        float(relative.support.lower_bound),
        float(absolute.support.lower_bound) - log_lag0,
    )
    assert np.isclose(
        float(relative.support.upper_bound),
        float(absolute.support.upper_bound) - log_lag0,
    )
    assert np.isclose(
        float(relative.base_dist.loc),
        float(absolute.base_dist.loc) - log_lag0,
    )
    assert np.isclose(
        float(relative.base_dist.scale),
        float(absolute.base_dist.scale),
    )


def test_relative_log_lag_blr_prior_bounds_reconstruct_absolute_lag_bounds():
    z = 0.8
    log_lag0 = np.log(12.0)
    relative = relative_log_lag_blr_prior(z=z, log_lag0=log_lag0)

    reconstructed_low = np.exp(float(relative.support.lower_bound) + log_lag0)
    reconstructed_high = np.exp(float(relative.support.upper_bound) + log_lag0)
    assert np.isclose(reconstructed_low, 0.1 * (1.0 + z))
    assert np.isclose(reconstructed_high, 1000.0 * (1.0 + z))


def test_binned_loo_residual_pair_correlation_uses_within_band_rest_frame_pairs():
    times_rf = np.array([0.0, 5.0, 20.0, 0.0, 8.0, 40.0])
    band_idx = np.array([0, 0, 0, 1, 1, 1])
    residuals = np.array([1.0, 2.0, -1.0, -1.0, -2.0, 3.0])

    result = binned_loo_residual_pair_correlation(
        times_rf,
        band_idx,
        residuals,
        bin_edges=(0.0, 10.0, 30.0, 100.0),
        bands=("g", "r"),
    )

    # The 0--10 day pairs are (g0,g1) and (r0,r1), both with product +2.
    assert result["loo_resid_pair_count_rf_0_10d"] == 2
    assert np.isclose(result["loo_resid_corr_rf_0_10d"], 2.0)
    assert result["loo_resid_pair_count_rf_0_10d_g"] == 1
    assert result["loo_resid_pair_count_rf_0_10d_r"] == 1
    # Cross-band pairs at identical times must never enter the statistic.
    assert result["loo_resid_pair_count_rf_10_30d"] == 2


def test_loo_residual_diagnostic_accepts_numpy_posterior_samples_for_erlang_model():
    times = np.array([0.0, 2.0, 8.0, 20.0, 0.5, 4.0, 12.0, 25.0])
    band_idx = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    order = np.argsort(times + 1e-9 * band_idx)
    times, band_idx = times[order], band_idx[order]
    y = 0.03 * np.sin(times / 15.0)
    yerr = np.full(times.size, 0.02)
    model = make_multiband_dho_blr_flux_linearized_erlang_model(
        (jnp.asarray(times), jnp.asarray(band_idx)),
        jnp.asarray(y),
        jnp.asarray(yerr),
        n_band=2,
        survey_idx=jnp.zeros(times.size, dtype=jnp.int32),
        erlang_order=3,
    )
    median = {
        "tau_fast_band": np.array([10.0, 10.0]),
        "tau_slow_band": np.array([300.0, 300.0]),
        "amp_cont_relflux": np.array([0.15, 0.15]),
        "amp_blr_relflux": np.array([0.03, 0.03]),
        "lag_blr": np.array([50.0, 60.0]),
        "mean": np.zeros(2),
        "linear_trend": np.array(0.0),
        "log_jitter": np.log(np.full((2, 3), 0.01)),
        "survey_delta_mag": np.zeros((2, 3)),
    }
    samples = {key: np.stack([value, value]) for key, value in median.items()}

    result = compute_loo_short_lag_residual_diagnostics(
        model,
        samples,
        {"object_id": "mock", "z": 1.0, "X": (times, band_idx)},
        ["g", "r"],
    )

    assert result["loo_resid_valid"]
    assert result["loo_resid_nobs"] == times.size
    assert result["loo_resid_pair_count_rf_0_10d"] > 0
    assert np.isfinite(result["loo_chi2_eff"])
    assert np.isclose(result["loo_rms"] ** 2, result["loo_chi2_eff"])

    median_params = {key: jnp.asarray(np.median(value, axis=0)) for key, value in samples.items()}
    gp, inds = model._build_gp(median_params)
    y_sorted = np.asarray(model._observed_y_sorted(median_params, inds), dtype=float)
    covariance = np.asarray(gp.covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    covariance += np.eye(covariance.shape[0]) * (1e-10 * max(float(np.nanmedian(np.diag(covariance))), 1.0))
    chol = np.linalg.cholesky(covariance)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_sorted - np.asarray(gp.loc, dtype=float)))
    precision = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(chol.shape[0])))
    loo_standardized = alpha / np.sqrt(np.maximum(np.diag(precision), 1e-300))
    expected_chi2_eff = np.mean(np.square(loo_standardized[np.isfinite(loo_standardized)]))
    assert np.isclose(result["loo_chi2_eff"], expected_chi2_eff)


def test_save_obj_samples_to_hdf5_writes_loo_scalar_diagnostics(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(multiband_fit_utils, "prefix", "loo_test")
    monkeypatch.setattr(multiband_fit_utils, "suffix", "run")

    multiband_fit_utils.save_obj_samples_to_hdf5(
        {"lag_blr": np.array([1.0, 2.0])},
        "object",
        scalar_diagnostics={"loo_chi2_eff": 1.25, "loo_rms": np.sqrt(1.25)},
    )

    output_path = tmp_path / "results/samples/loo_test/object_run.h5"
    with h5py.File(output_path, "r") as hdf:
        np.testing.assert_array_equal(hdf["lag_blr"][:], np.array([1.0, 2.0]))
        assert hdf["_diagnostics/loo_chi2_eff"][()] == 1.25
        assert np.isclose(hdf["_diagnostics/loo_rms"][()], np.sqrt(1.25))

    loaded = multiband_fit_utils.load_obj_samples_from_hdf5(
        file_path=str(output_path)
    )
    assert set(loaded) == {"lag_blr"}


def test_apply_resume_sample_save_policy_disables_sample_saving_on_resume():
    args = SimpleNamespace(resume=True, save_sample_file=True)

    returned = apply_resume_sample_save_policy(args)

    assert returned is args
    assert args.save_sample_file is False


def test_apply_resume_sample_save_policy_preserves_fresh_sample_saving_choice():
    args = SimpleNamespace(resume=False, save_sample_file=True)

    returned = apply_resume_sample_save_policy(args)

    assert returned is args
    assert args.save_sample_file is True


def test_build_explicit_model_params_preserves_uv_intercepts_across_band_sets():
    lam_full = jnp.array([1700.0, 2300.0, 3000.0, 3800.0])
    lam_sub = jnp.array([2300.0, 3000.0, 3800.0])

    explicit_full = build_explicit_model_params(_make_raw_public(len(lam_full)), lam_full)
    explicit_sub = build_explicit_model_params(_make_raw_public(len(lam_sub)), lam_sub)

    assert np.isclose(float(explicit_full["log_sigma_uv"]), np.log(0.2))
    assert np.isclose(float(explicit_full["log_tau_uv"]), np.log(300.0))
    assert np.isclose(float(explicit_full["log_tau_fast_uv"]), np.log(30.0))
    assert np.isclose(float(explicit_sub["log_sigma_uv"]), np.log(0.2))
    assert np.isclose(float(explicit_sub["log_tau_uv"]), np.log(300.0))
    assert np.isclose(float(explicit_sub["log_tau_fast_uv"]), np.log(30.0))

    assert np.allclose(
        np.asarray(explicit_full["amp_cont"])[1:],
        np.asarray(explicit_sub["amp_cont"]),
    )
    assert np.allclose(
        np.asarray(explicit_full["tau_fast_band"])[1:],
        np.asarray(explicit_sub["tau_fast_band"]),
    )
    assert np.allclose(
        np.asarray(explicit_full["tau_slow_band"])[1:],
        np.asarray(explicit_sub["tau_slow_band"]),
    )


def test_build_explicit_model_params_internal_and_public_forms_are_equivalent():
    lam_rf = jnp.array([2100.0, 2900.0, 3600.0])
    explicit_public = build_explicit_model_params(_make_raw_public(len(lam_rf)), lam_rf)

    internal = {
        "log_sigma_center0": explicit_public["log_sigma_center0"],
        "log_tau_slow_center0": explicit_public["log_tau_slow_center0"],
        "log_tau_fast_center0": explicit_public["log_tau_fast_center0"],
        "lambda_center_rf": explicit_public["lambda_center_rf"],
        "eta_sigma": jnp.array(-0.55),
        "eta_tau": jnp.array(0.25),
        "dlog_amp_blr": jnp.full((len(lam_rf),), -1.0),
        "dlog_amp_blr2": jnp.full((len(lam_rf),), -1.5),
        "log_lag_blr": jnp.full((len(lam_rf),), np.log(20.0)),
        "log_lag_blr2": jnp.full((len(lam_rf),), np.log(60.0)),
        "lag0": jnp.array(5.0),
        "lag_beta": jnp.array(4.0 / 3.0),
    }
    explicit_internal = build_explicit_model_params(internal, lam_rf)

    assert np.isclose(float(explicit_internal["log_sigma_uv"]), float(explicit_public["log_sigma_uv"]))
    assert np.isclose(float(explicit_internal["log_tau_uv"]), float(explicit_public["log_tau_uv"]))
    assert np.isclose(float(explicit_internal["log_tau_fast_uv"]), float(explicit_public["log_tau_fast_uv"]))
    assert np.allclose(np.asarray(explicit_internal["amp_cont"]), np.asarray(explicit_public["amp_cont"]))
    assert np.allclose(np.asarray(explicit_internal["amp_blr"]), np.asarray(explicit_public["amp_blr"]))
    assert np.allclose(np.asarray(explicit_internal["amp_blr2"]), np.asarray(explicit_public["amp_blr2"]))
    assert np.allclose(np.asarray(explicit_internal["lag_blr"]), np.asarray(explicit_public["lag_blr"]))
    assert np.allclose(np.asarray(explicit_internal["lag_blr2"]), np.asarray(explicit_public["lag_blr2"]))
    assert np.allclose(np.asarray(explicit_internal["tau_fast_band"]), np.asarray(explicit_public["tau_fast_band"]))
    assert np.allclose(np.asarray(explicit_internal["tau_slow_band"]), np.asarray(explicit_public["tau_slow_band"]))


def test_build_explicit_model_params_centers_disk_lag_on_geometric_mean():
    lam_rf = jnp.array([1800.0, 2500.0, 3472.222222222222])
    raw = _make_raw_public(len(lam_rf))
    explicit = build_explicit_model_params(raw, lam_rf)

    lambda_center_rf = float(explicit["lambda_center_rf"])
    lag_disk = np.asarray(explicit["lag_disk"])
    expected_lag0 = float(raw["lag0"])

    center_idx = int(np.argmin(np.abs(np.asarray(lam_rf) - lambda_center_rf)))
    assert np.isclose(lambda_center_rf, np.exp(np.mean(np.log(np.asarray(lam_rf)))))
    assert np.isclose(np.asarray(lam_rf)[center_idx], lambda_center_rf)
    assert np.isclose(lag_disk[center_idx], expected_lag0)


def test_compute_band_igm_transmission_suppresses_u_more_than_redder_bands():
    transmission = np.asarray(compute_band_igm_transmission(["u", "g", "r", "i"], 3.0), dtype=float)
    assert transmission[0] < transmission[1] < transmission[2] <= transmission[3]
    assert transmission[0] < 0.6
    assert transmission[2] > 0.99


def test_build_explicit_model_params_keeps_igm_out_of_continuum_amplitude():
    lam_rf = jnp.array([1500.0, 2000.0, 2500.0])
    raw = _make_raw_public(len(lam_rf))
    transmission = jnp.array([0.25, 0.5, 0.9], dtype=float)
    raw["log_igm_transmission_band"] = jnp.log(transmission)
    explicit = build_explicit_model_params(raw, lam_rf)

    baseline = build_explicit_model_params(_make_raw_public(len(lam_rf)), lam_rf)

    assert np.allclose(np.asarray(explicit["amp_cont"]), np.asarray(baseline["amp_cont"]))
    assert np.allclose(np.asarray(explicit["igm_transmission_band"]), np.asarray(transmission))


def test_build_explicit_model_params_relflux_keeps_igm_out_of_continuum_amplitude():
    lam_rf = jnp.array([1500.0, 2000.0, 2500.0])
    raw = _make_raw_public(len(lam_rf))
    transmission = jnp.array([0.25, 0.5, 0.9], dtype=float)
    raw["log_igm_transmission_band"] = jnp.log(transmission)
    explicit = build_explicit_model_params_relflux(raw, lam_rf)

    baseline = build_explicit_model_params_relflux(_make_raw_public(len(lam_rf)), lam_rf)

    assert np.allclose(
        np.asarray(explicit["amp_cont_relflux"]),
        np.asarray(baseline["amp_cont_relflux"]),
    )
    assert np.allclose(np.asarray(explicit["igm_transmission_band"]), np.asarray(transmission))


def test_prediction_params_expand_drw_q_without_legacy_fast_uv_coordinate():
    lam_rf = jnp.array([1500.0, 2000.0, 2500.0])
    raw = _make_raw_public(len(lam_rf))
    raw.pop("log_tau_uv")
    raw.pop("log_tau_fast_uv")
    raw["log_tau_drw_center0"] = jnp.log(300.0)
    raw["log_quality_factor"] = jnp.log(2.0)
    raw["log_tau_perturb_ratio"] = jnp.log(0.02)

    explicit = add_model_prediction_params(
        raw,
        lam_rf,
        model_variant="mag_flux_linearized_erlang",
    )

    assert np.allclose(np.asarray(explicit["quality_factor"]), 2.0)
    assert np.asarray(explicit["tau_drw_band"]).shape == (len(lam_rf),)
    assert np.all(np.isfinite(np.asarray(explicit["tau_drw_band"])))
    assert np.allclose(
        np.asarray(explicit["tau_perturb_band"]),
        0.02 * np.asarray(explicit["tau_drw_band"]),
    )


def test_carma21_numpyro_model_trace_materializes_likelihood_and_uv_outputs():
    obj = {
        "object_id": "carma21-smoke",
        "z": 1.0,
        "X": (
            np.array([0.0, 5.0, 10.0, 15.0]),
            np.array([0, 1, 0, 1], dtype=np.int32),
        ),
        "y": np.array([0.0, 0.02, -0.01, 0.01]),
        "yerr": np.full(4, 0.03),
        "survey_idx": np.zeros(4, dtype=np.int32),
        "mags_means": np.array([20.0, 20.0]),
        "bands": ["g", "r"],
        "survey_names": ("sdss", "ps1", "ztf"),
        "agn_fraction_by_band": np.array([0.55, 0.75]),
        "agn_fraction_err_by_band": np.array([0.0, 0.08]),
    }
    model = build_single_object_model_mag_flux_linearized(
        obj,
        np.array([2000.0, 3000.0]),
        log_jitter_mean=np.full((2, 3), np.log(0.03)),
        drw_parameterization=True,
    )

    sites = fit_lc.trace(
        fit_lc.seed(model, jax.random.PRNGKey(0))
    ).get_trace()

    for key in (
        "log_quality_factor",
        "log_tau_perturb_ratio",
        "log_tau_uv",
        "log_tau_perturb_uv",
        "tau_perturb",
        "agn_fraction_by_band",
        "loglike",
    ):
        assert key in sites
        assert np.all(np.isfinite(np.asarray(sites[key]["value"])))
    assert "_agn_fraction_uncertain" not in sites
    assert sites["agn_fraction_by_band"]["type"] == "deterministic"
    np.testing.assert_allclose(
        np.asarray(sites["agn_fraction_by_band"]["value"]),
        [0.55, 0.75],
    )


def test_flux_line_ratio_offsets_include_static_igm_transmission():
    lam_rf = jnp.array([1500.0, 2000.0, 2500.0])
    lambda_center_rf = compute_lambda_center_rf(lam_rf)

    baseline = compute_flux_line_ratio_offsets(
        lam_rf,
        lambda_center_rf=lambda_center_rf,
        eta_sigma=jnp.array(-0.5),
        log_igm_transmission_band=jnp.zeros(len(lam_rf), dtype=float),
    )
    absorbed = compute_flux_line_ratio_offsets(
        lam_rf,
        lambda_center_rf=lambda_center_rf,
        eta_sigma=jnp.array(-0.5),
        log_igm_transmission_band=jnp.log(jnp.array([0.25, 0.5, 0.9], dtype=float)),
    )

    expected_blr = np.asarray(baseline["blr_band"]) - np.log(np.array([0.25, 0.5, 0.9], dtype=float))
    bc_weight = np.maximum(np.asarray(balmer_continuum_weight(lam_rf), dtype=float), 1e-6)
    expected_bc_ref = np.sum(
        bc_weight * (expected_blr + np.log(np.maximum(np.asarray(balmer_continuum_weight(lam_rf), dtype=float), 1e-12)))
    ) / np.sum(bc_weight)

    assert np.allclose(np.asarray(absorbed["blr_band"]), expected_blr)
    assert np.isclose(float(absorbed["bc_ref"]), expected_bc_ref)


def test_compute_igm_transmission_rest_wave_matches_observed_wave_conversion():
    rest_wave = np.array([900.0, 1216.0, 1500.0], dtype=float)
    z = 2.0

    rest_eval = np.asarray(compute_igm_transmission_rest_wave(rest_wave, z), dtype=float)
    obs_eval = np.asarray(compute_igm_transmission_obs_wave(rest_wave * (1.0 + z), z), dtype=float)

    assert np.allclose(rest_eval, obs_eval)


def test_compute_lam_lya_suppression_rf_turns_on_u_band_near_z_one_point_five():
    low_z = np.asarray(compute_lam_lya_suppression_rf(["u"], 1.4), dtype=float)
    high_z = np.asarray(compute_lam_lya_suppression_rf(["u"], 1.6), dtype=float)

    assert low_z.shape == (1,)
    assert high_z.shape == (1,)
    assert low_z[0] > 1216.0
    assert high_z[0] < 1216.0
    low_u = float(np.asarray(compute_band_igm_transmission(["u"], 1.4), dtype=float)[0])
    high_u = float(np.asarray(compute_band_igm_transmission(["u"], 1.6), dtype=float)[0])
    very_high_u = float(np.asarray(compute_band_igm_transmission(["u"], 3.0), dtype=float)[0])
    assert low_u > high_u > very_high_u
    assert low_u > 0.99
    assert high_u > 0.99
    assert very_high_u < 0.6


def test_log_tau_fast_center0_prior_is_centered_at_configured_tau_slow_ratio():
    log_tau_slow = jnp.array(np.log(1000.0))
    prior = log_tau_fast_center0_prior(log_tau_slow, tau_fast_truncated=False)

    assert np.isclose(float(prior.loc), np.log(1000.0 / TAU_FAST_TO_SLOW_PRIOR_RATIO))
    assert np.isclose(float(prior.scale), 0.4 * np.log(10.0))


def test_smooth_ordered_tau_parameterization_is_strict_and_centered():
    log_tau_slow = jnp.log(1500.0)
    raw_prior = log_tau_fast_separation_raw_prior()
    log_tau_fast = ordered_log_tau_fast(log_tau_slow, raw_prior.loc)

    assert float(log_tau_fast) < float(log_tau_slow)
    assert np.isclose(
        np.exp(float(log_tau_slow - log_tau_fast)),
        TAU_FAST_TO_SLOW_PRIOR_RATIO,
        rtol=1e-12,
    )


@pytest.mark.parametrize("raw", [-30.0, -5.0, 0.0, 5.0, 30.0])
def test_smooth_ordered_tau_parameterization_has_finite_gradient(raw):
    value, gradient = jax.value_and_grad(
        lambda x: ordered_log_tau_fast(jnp.log(1500.0), x)
    )(jnp.asarray(raw))

    assert np.isfinite(float(value))
    assert np.isfinite(float(gradient))
    assert float(value) < float(jnp.log(1500.0))


def test_linear_trend_prior_matches_1e_minus_4_mag_per_day_in_rest_frame():
    t_ref = np.array([0.0, 10.0, 20.0], dtype=float)
    prior = linear_trend_prior(t_ref=t_ref, z=1.0)

    expected_scale = 1e-4 * np.std(t_ref) / 2.0
    assert np.isclose(float(prior.loc), 0.0)
    assert np.isclose(float(prior.scale), expected_scale)


def test_corner_plot_labels_keep_only_curated_main_parameters():
    samples_flat = {
        "eta_sigma": np.array([0.1, 0.2]),
        "eta_tau": np.array([0.3, 0.4]),
        "log_sigma_center0": np.array([0.8, 0.9]),
        "log_sigma_uv": np.array([1.0, 1.1]),
        "log_tau_slow_center0": np.array([1.8, 1.9]),
        "log_tau_uv": np.array([2.0, 2.1]),
        "log_tau_fast_center0": np.array([0.2, 0.3]),
        "log_tau_fast_uv": np.array([0.5, 0.6]),
        "lag0": np.array([5.0, 6.0]),
        "lag_beta": np.array([1.1, 1.2]),
        "linear_trend": np.array([0.0, 0.01]),
        "dlog_amp_blr_g": np.array([-1.0, -0.9]),
        "dlog_amp_blr2_g": np.array([-1.3, -1.2]),
        "log_lag_blr_g": np.array([3.0, 3.1]),
        "log_lag_blr2_g": np.array([3.5, 3.6]),
        "mean_g": np.array([19.0, 19.1]),
        "log_jitter_g": np.array([-2.0, -2.1]),
        "amp_cont_g": np.array([0.2, 0.3]),
        "tau_fast_g": np.array([10.0, 11.0]),
        "tau_slow_g": np.array([100.0, 110.0]),
        "lag_disk_g": np.array([1.0, 1.1]),
    }

    all_labels, labels_for_corner = _corner_plot_labels(samples_flat)

    assert "eta_sigma" in labels_for_corner
    assert "log_sigma_uv" in labels_for_corner
    assert "dlog_amp_blr_g" in labels_for_corner
    assert "mean_g" in labels_for_corner
    assert "log_jitter_g" in labels_for_corner
    assert "log_sigma_center0" not in labels_for_corner
    assert "log_tau_slow_center0" not in labels_for_corner
    assert "log_tau_fast_center0" not in labels_for_corner
    assert "dlog_amp_blr2_g" not in labels_for_corner
    assert "log_lag_blr2_g" not in labels_for_corner
    assert "amp_cont_g" not in labels_for_corner
    assert "tau_fast_g" not in labels_for_corner
    assert "tau_slow_g" not in labels_for_corner
    assert "lag_disk_g" not in labels_for_corner
    assert set(labels_for_corner).issubset(set(all_labels))


def test_trace_plot_labels_include_fitted_survey_and_slope_offsets_only():
    samples_flat = {
        "eta_sigma": np.array([0.1, 0.2]),
        "linear_trend_band_offset_g": np.array([0.01, 0.015], dtype=float),
        "linear_trend_band_offset_r": np.zeros(2, dtype=float),
        "survey_delta_mag_g_sdss": np.zeros(2, dtype=float),
        "survey_delta_mag_g_ztf": np.array([0.01, 0.02], dtype=float),
        "survey_delta_mag_r_ps1": np.array([-0.01, -0.005], dtype=float),
    }

    all_labels, labels_for_trace = _trace_plot_labels(samples_flat)

    assert "eta_sigma" in labels_for_trace
    assert "linear_trend_band_offset_g" in labels_for_trace
    assert "linear_trend_band_offset_r" not in labels_for_trace
    assert "survey_delta_mag_g_ztf" in labels_for_trace
    assert "survey_delta_mag_r_ps1" in labels_for_trace
    assert "survey_delta_mag_g_sdss" not in labels_for_trace
    assert set(labels_for_trace).issubset(set(all_labels))


def test_balmer_continuum_weight_transitions_smoothly_across_3646():
    lam_rf = jnp.array([3200.0, 3646.0, 3900.0, 4500.0])
    weight = np.asarray(balmer_continuum_weight(lam_rf))

    assert weight[0] > weight[1] > weight[2] > weight[3]
    assert np.isclose(weight[1], 0.5)
    assert weight[0] > 0.8
    assert weight[3] < 0.05


def test_build_explicit_model_params_smoothly_weights_bc_amplitude_but_not_bc_lag():
    lam_rf = jnp.array([3200.0, 3646.0, 3900.0, 4500.0])
    raw = _make_raw_public(len(lam_rf))
    raw["dlog_amp_bc"] = jnp.array(-0.4)
    raw["log_lag_ratio_bc_to_blr"] = jnp.array(np.log(0.2))
    raw["log_lag_blr"] = jnp.log(jnp.array([20.0, 30.0, 40.0, 50.0]))

    explicit = build_explicit_model_params(raw, lam_rf)

    bc_weight = np.asarray(balmer_continuum_weight(lam_rf))
    amp_bc = np.asarray(explicit["amp_bc"])
    lag_bc = np.asarray(explicit["lag_bc"])
    expected_base_amp = np.exp(float(raw["log_sigma_uv"] + raw["dlog_amp_bc"]))
    expected_lag_bc = 0.2 * np.exp(np.mean(np.log(np.array([20.0, 30.0, 40.0, 50.0]))))

    assert np.allclose(amp_bc, expected_base_amp * bc_weight)
    assert amp_bc[0] > amp_bc[1] > amp_bc[2] > amp_bc[3] > 0.0
    assert np.allclose(lag_bc, expected_lag_bc)


def test_make_lc_drops_z_by_default_but_keeps_lya_bands():
    obj = _make_object(z=1.6)
    lc = make_lc(obj, bands=["u", "g", "r", "i", "z"], drop_band_lyman_alpha=False)

    assert lc is not None
    assert lc["bands"] == ["u", "g", "r", "i"]
    assert lc["dropped_bands"] == ["z"]


def test_make_lc_centers_with_inverse_variance_weighted_mean():
    obj = _make_object(z=1.0)
    obj["mags"]["g"] = np.array([20.0, 21.0])
    obj["magerrs"]["g"] = np.array([0.1, 0.2])

    lc = make_lc(obj, bands=["g", "r", "i", "z"], drop_band_lyman_alpha=False)

    assert lc is not None
    assert np.isclose(lc["mags_means"][0], 20.2)
    assert np.isclose(lc["mags_mean_errs"][0], 1.0 / np.sqrt(125.0))
    g_values = np.asarray(lc["y"])[np.asarray(lc["band_idx"]) == 0]
    np.testing.assert_allclose(g_values, [-0.2, 0.8])


def test_make_lc_preserves_aligned_psf_dilution_priors():
    obj = _make_object(z=1.0)
    obj["psf_constant_flux_corrected"] = True
    for band, fraction, error in zip("ugriz", [0.4, 0.5, 0.6, 0.7, 0.8], [0.04] * 5):
        obj[f"f_AGN_psf_{band}"] = fraction
        obj[f"f_AGN_psf_{band}_err"] = error

    lc = make_lc(obj, bands=["u", "g", "r", "i", "z"], drop_band_lyman_alpha=False)

    assert lc["bands"] == ["u", "g", "r", "i"]
    np.testing.assert_allclose(lc["agn_fraction_by_band"], [0.4, 0.5, 0.6, 0.7])
    np.testing.assert_allclose(lc["agn_fraction_err_by_band"], [0.04] * 4)


def test_make_lc_uses_the_sed_fraction_reference_after_time_cut():
    obj = _make_object(z=1.0)
    obj["psf_constant_flux_corrected"] = True
    obj["psf_fraction_reference_mags_by_band"] = {
        band: 19.5 for band in "ugriz"
    }
    obj["psf_fraction_reference_magerrs_by_band"] = {
        band: 0.02 for band in "ugriz"
    }
    for band in "ugriz":
        obj[f"f_AGN_psf_{band}"] = 0.6
        obj[f"f_AGN_psf_{band}_err"] = 0.05

    lc = make_lc(obj, bands=["g", "r", "i", "z"], drop_band_lyman_alpha=False)

    assert lc["mags_means"][0] == pytest.approx(19.5)
    assert lc["mags_mean_errs"][0] == pytest.approx(0.02)
    g_values = np.asarray(lc["y"])[np.asarray(lc["band_idx"]) == 0]
    np.testing.assert_allclose(g_values, np.asarray(obj["mags"]["g"]) - 19.5)


def test_make_lc_rejects_missing_dilution_prior_for_retained_band():
    obj = _make_object(z=1.0)
    obj["psf_constant_flux_corrected"] = True
    obj["f_AGN_psf_g"] = 0.5
    obj["f_AGN_psf_g_err"] = 0.05

    with pytest.raises(ValueError, match="fitted band.*u, r, i"):
        make_lc(obj, bands=["u", "g", "r", "i", "z"], drop_band_lyman_alpha=False)


def test_make_lc_can_hard_drop_lya_affected_bands():
    obj = _make_object(z=1.6)
    lc = make_lc(obj, bands=["u", "g", "r", "i", "z"], drop_band_lyman_alpha=True)

    assert lc is not None
    assert lc["bands"] == ["g", "r", "i"]
    assert lc["dropped_bands"] == ["u", "z"]


def test_make_lc_adds_variability_fields_for_retained_bands():
    obj = {
        "object_id": "varobj",
        "z": 1.2,
        "times": {
            "u": np.array([0.0, 25.0, 380.0, 405.0], dtype=float),
            "g": np.array([0.0, 25.0, 380.0, 405.0], dtype=float),
            "r": np.array([5.0, 40.0, 385.0, 415.0], dtype=float),
            "i": np.array([10.0, 50.0, 390.0, 420.0], dtype=float),
            "z": np.array([0.0, 1.0], dtype=float),
        },
        "mags": {
            "u": np.array([20.0, 20.2, 19.8, 20.1], dtype=float),
            "g": np.array([20.0, 20.3, 19.7, 20.0], dtype=float),
            "r": np.array([19.8, 20.0, 19.6, 19.9], dtype=float),
            "i": np.array([19.6, 19.9, 19.5, 19.8], dtype=float),
            "z": np.array([19.5, 19.6], dtype=float),
        },
        "magerrs": {band: np.full(len(vals), 0.1, dtype=float) for band, vals in {
            "u": np.array([20.0, 20.2, 19.8, 20.1], dtype=float),
            "g": np.array([20.0, 20.3, 19.7, 20.0], dtype=float),
            "r": np.array([19.8, 20.0, 19.6, 19.9], dtype=float),
            "i": np.array([19.6, 19.9, 19.5, 19.8], dtype=float),
            "z": np.array([19.5, 19.6], dtype=float),
        }.items()},
        "cadence": {band: 5.0 for band in ["u", "g", "r", "i", "z"]},
        "cadence_err": {band: 0.5 for band in ["u", "g", "r", "i", "z"]},
        "number_points": {band: 4 for band in ["u", "g", "r", "i", "z"]},
    }
    obj["number_points"]["z"] = 2

    lc = make_lc(obj, bands=["u", "g", "r", "i", "z"], drop_band_lyman_alpha=False)

    assert lc is not None
    for band in lc["bands"]:
        assert f"variability_n_points_{band}" in lc
        assert f"variability_chi_sq_{band}" in lc
        assert f"variability_chi_sq_red_{band}" in lc
        assert f"variability_pvalue_{band}" in lc
        assert f"variability_neg_log10_pvalue_{band}" in lc
    assert np.isfinite(lc["variability_chi_sq_red_g"])


def test_make_lc_does_not_repeat_loader_outlier_rejection():
    times = np.arange(13, dtype=float) * 5.0
    g_mags = np.array([20.0, 20.0, 20.1, 20.0, 20.1, 20.0, 23.5, 20.0, 20.1, 20.0, 20.1, 20.0, 20.1], dtype=float)
    obj = {
        "object_id": "outlier",
        "z": 1.0,
        "times": {
            "g": times,
            "r": np.array([0.0, 365.0], dtype=float),
            "i": np.array([20.0, 385.0], dtype=float),
            "z": np.array([0.0, 1.0], dtype=float),
        },
        "mags": {
            "g": g_mags,
            "r": np.array([19.8, 19.9], dtype=float),
            "i": np.array([19.6, 19.7], dtype=float),
            "z": np.array([19.5, 19.6], dtype=float),
        },
        "magerrs": {
            "g": np.full(13, 0.05, dtype=float),
            "r": np.full(2, 0.05, dtype=float),
            "i": np.full(2, 0.05, dtype=float),
            "z": np.full(2, 0.05, dtype=float),
        },
        "cadence": {band: 5.0 for band in ["g", "r", "i", "z"]},
        "cadence_err": {band: 0.5 for band in ["g", "r", "i", "z"]},
        "number_points": {"g": 13, "r": 2, "i": 2, "z": 2},
    }

    lc = make_lc(obj, bands=["g", "r", "i", "z"], drop_band_lyman_alpha=False)

    assert lc is not None
    assert lc["variability_n_points_g"] == 13


def test_compute_parameter_kls_ignores_nonfinite_conditioning_samples():
    flat_samples = {
        "eta_sigma": np.array([-0.4, np.nan, -0.2, -0.3]),
        "eta_tau": np.array([0.1, 0.2, np.nan, 0.25]),
        "log_sigma_center0": np.array([-1.0, -0.9, -0.8, -0.85]),
        "log_tau_slow_center0": np.array([5.0, 5.1, 5.2, 5.15]),
        "log_tau_fast_center0": np.array([2.3, 2.4, 2.5, 2.45]),
        "linear_trend": np.array([0.0, 0.01, -0.01, 0.02]),
        "lag0": np.array([5.0, 5.5, 4.5, 5.2]),
        "lag_beta": np.array([1.2, 1.4, 1.3, 1.35]),
        "mean_g": np.array([0.0, 0.02, -0.01, 0.01]),
        "mean_r": np.array([0.0, 0.01, -0.02, 0.02]),
        "log_jitter_g": np.array([-3.0, -2.9, -3.1, -3.05]),
        "log_jitter_r": np.array([-3.1, -3.0, -3.2, -3.05]),
        "dlog_amp_blr_g": np.array([-1.0, -0.9, -1.1, -1.05]),
        "dlog_amp_blr_r": np.array([-1.2, -1.1, -1.3, -1.25]),
        "dlog_amp_blr2_g": np.array([-1.5, -1.4, -1.6, -1.55]),
        "dlog_amp_blr2_r": np.array([-1.7, -1.6, -1.8, -1.75]),
        "log_lag_blr_g": np.array([3.0, 3.1, 3.2, 3.15]),
        "log_lag_blr_r": np.array([3.1, 3.2, 3.3, 3.25]),
        "log_lag_blr2_g": np.array([4.0, 4.1, 4.2, 4.15]),
        "log_lag_blr2_r": np.array([4.1, 4.2, 4.3, 4.25]),
        "dlog_amp_bc": np.array([-1.1, -1.0, -0.9, -1.05]),
        "log_lag_ratio_bc_to_blr": np.array([np.log(0.15), np.log(0.2), np.log(0.25), np.log(0.22)]),
    }

    kls = compute_parameter_kls(
        flat_samples,
        bands=["g", "r"],
        survey_names=("sdss", "ps1", "ztf"),
        t_ref=np.array([0.0, 10.0, 20.0], dtype=float),
        z=1.5,
        lambda_center_rf=2500.0,
        log_jitter_mean=np.array([-3.0, -3.1]),
        disable_linear_trend=False,
        disable_lag_blr=False,
        disable_lag_bc=False,
        drop_band_lyman_alpha=False,
        tau_fast_truncated=False,
        n_blr_terms=2,
    )

    assert "log_sigma_center0_kl" in kls
    assert "log_tau_slow_center0_kl" in kls
    assert "log_tau_fast_center0_kl" in kls
    assert np.isfinite(kls["log_sigma_center0_kl"])
    assert np.isfinite(kls["log_tau_slow_center0_kl"])
    assert np.isfinite(kls["log_tau_fast_center0_kl"])


def test_process_samples_keeps_uv_outputs_at_2500_and_stores_band_metadata():
    z = 1.5
    bands = ["g", "r", "i"]
    lam_rf_kept = np.asarray([lambda_pivot[b] / (1.0 + z) for b in bands], dtype=float)
    lambda_center_rf = float(np.exp(np.mean(np.log(lam_rf_kept))))

    eta_sigma = np.asarray([-0.6, -0.4, -0.5])
    eta_tau = np.asarray([0.1, 0.3, 0.2])
    log_sigma_center0 = np.asarray(np.log([0.18, 0.21, 0.24]))
    log_tau_slow_center0 = np.asarray(np.log([250.0, 310.0, 400.0]))
    log_tau_fast_center0 = np.asarray(np.log([25.0, 32.0, 40.0]))

    flat_samples = {
        "log_sigma_center0": log_sigma_center0,
        "log_tau_slow_center0": log_tau_slow_center0,
        "log_tau_fast_center0": log_tau_fast_center0,
        "log_sigma_uv": log_sigma_center0,
        "log_tau_uv": log_tau_slow_center0,
        "log_tau_fast_uv": log_tau_fast_center0,
        "eta_sigma": eta_sigma,
        "eta_tau": eta_tau,
        "log_lag_blr_g": np.asarray(np.log([30.0, 40.0, 50.0])),
        "log_lag_blr_r": np.asarray(np.log([35.0, 45.0, 55.0])),
        "log_lag_blr_i": np.asarray(np.log([40.0, 50.0, 60.0])),
        "log_lag_blr2_g": np.asarray(np.log([80.0, 90.0, 100.0])),
        "log_lag_blr2_r": np.asarray(np.log([85.0, 95.0, 105.0])),
        "log_lag_blr2_i": np.asarray(np.log([90.0, 100.0, 110.0])),
        "lag_bc_g": np.asarray([6.0, 8.0, 10.0]),
        "lag_bc_r": np.asarray([7.0, 9.0, 11.0]),
        "lag_bc_i": np.asarray([8.0, 10.0, 12.0]),
    }

    result = process_samples(
        flat_samples,
        {"object_id": "obj", "z": z},
        bands=bands,
    )

    expected_log_sigma_uv = np.percentile(log_sigma_center0 / np.log(10), 50)
    expected_log_tau_uv_rf = np.percentile(
        log_tau_slow_center0 / np.log(10) - np.log10(1.0 + z),
        50,
    )
    expected_log_tau_fast_uv_rf = np.percentile(
        log_tau_fast_center0 / np.log(10) - np.log10(1.0 + z),
        50,
    )

    assert np.isclose(result["lambda_center_rf"], lambda_center_rf)
    assert result["n_bands_kept"] == 3
    assert result["bands_kept"] == "g,r,i"
    assert np.isclose(result["log_sigma_uv"], expected_log_sigma_uv)
    assert np.isclose(result["log_tau_uv"], np.percentile(log_tau_slow_center0 / np.log(10), 50))
    assert np.isclose(result["log_tau_uv_rf"], expected_log_tau_uv_rf)
    assert np.isclose(result["log_tau_fast_uv_rf"], expected_log_tau_fast_uv_rf)
    assert np.isclose(
        result["log_lag_blr_r_RF"],
        np.percentile(np.log10([35.0, 45.0, 55.0]) - np.log10(1.0 + z), 50),
    )

    assert np.isclose(
        result["log_lag_blr2_r_RF"],
        np.percentile(np.log10([85.0, 95.0, 105.0]) - np.log10(1.0 + z), 50),
    )
    assert np.isclose(
        result["log_lag_bc_r_RF"],
        np.percentile(np.log10([7.0, 9.0, 11.0]) - np.log10(1.0 + z), 50),
    )
    assert result["bc_weight_g"] > result["bc_weight_r"] > result["bc_weight_i"] > 0.9
    lam_g_rf = lambda_pivot["g"] / (1.0 + z)
    log_sigma_band_g = (
        log_sigma_center0 / np.log(10)
        + log_single_pl(lam_g_rf, np.full_like(eta_sigma, 2500.0), eta_sigma)
    )
    log_tau_band_g_rf = (
        log_tau_slow_center0 / np.log(10)
        - np.log10(1.0 + z)
        + log_single_pl(lam_g_rf, np.full_like(eta_tau, 2500.0), eta_tau)
    )
    log_tau_fast_band_g_rf = (
        log_tau_fast_center0 / np.log(10)
        - np.log10(1.0 + z)
        + log_single_pl(lam_g_rf, np.full_like(eta_tau, 2500.0), eta_tau)
    )
    tau_band_g_rf = np.power(10.0, log_tau_band_g_rf)
    tau_fast_band_g_rf = np.power(10.0, log_tau_fast_band_g_rf)
    expected_sigma_rms_g = np.percentile(
        log_sigma_band_g,
        50,
    )
    assert np.isclose(result["log_sigma_rms_band_g"], expected_sigma_rms_g)


def test_process_samples_supports_drw_q_without_fast_pole_outputs():
    samples = {
        "log_sigma_uv": np.log(np.asarray([0.18, 0.20, 0.22])),
        "log_tau_uv": np.log(np.asarray([250.0, 300.0, 350.0])),
        "eta_sigma": np.asarray([-0.5, -0.45, -0.4]),
        "eta_tau": np.asarray([0.1, 0.2, 0.3]),
        "quality_factor": np.asarray([1.5, 2.0, 2.5]),
        "log_quality_factor": np.log(np.asarray([1.5, 2.0, 2.5])),
    }

    result = process_samples(
        samples,
        {"object_id": "qpo", "z": 1.0},
        bands=["g", "r"],
    )

    assert np.isfinite(result["log_tau_uv_rf"])
    assert np.isfinite(result["quality_factor"])
    assert "log_tau_fast_uv" not in result
    assert "log_tau_fast_band_g_RF" not in result
    assert np.isclose(
        result["log_sigma_rms_band_g"],
        result["log_sigma_band_g"],
    )


def test_process_samples_keeps_bc_lag_for_band_near_balmer_edge():
    z = lambda_pivot["r"] / 3900.0 - 1.0
    flat_samples = {
        "log_sigma_uv": np.asarray(np.log([0.18, 0.21, 0.24])),
        "log_tau_uv": np.asarray(np.log([250.0, 310.0, 400.0])),
        "log_tau_fast_uv": np.asarray(np.log([25.0, 32.0, 40.0])),
        "eta_sigma": np.asarray([-0.6, -0.4, -0.5]),
        "eta_tau": np.asarray([0.1, 0.3, 0.2]),
        "lag_bc_r": np.asarray([7.0, 9.0, 11.0]),
    }

    result = process_samples(
        flat_samples,
        {"object_id": "obj", "z": z},
        bands=["r"],
    )

    assert result["bc_weight_r"] > 0.2
    assert np.isclose(
        result["log_lag_bc_r_RF"],
        np.percentile(np.log10([7.0, 9.0, 11.0]) - np.log10(1.0 + z), 50),
    )


def test_compute_parameter_kls_returns_expected_keys():
    rng = np.random.default_rng(123)
    bands = ["g", "r"]
    z = 1.2
    lam_rf = np.asarray([lambda_pivot[b] / (1.0 + z) for b in bands], dtype=float)
    lambda_center_rf = float(np.exp(np.mean(np.log(lam_rf))))
    n = 64

    flat_samples = {
        "eta_sigma": rng.normal(-0.45, 0.08, size=n),
        "eta_tau": rng.normal(0.55, 0.10, size=n),
        "log_sigma_center0": rng.normal(np.log(0.2), 0.2, size=n),
        "log_tau_slow_center0": rng.normal(np.log(300.0), 0.3, size=n),
        "log_tau_fast_center0": rng.normal(np.log(30.0), 0.2, size=n),
        "linear_trend": rng.normal(0.0, 0.05, size=n),
        "lag0": np.abs(rng.normal(5.0, 1.0, size=n)),
        "lag_beta": np.abs(rng.normal(4.0 / 3.0, 0.1, size=n)),
        "mean_g": rng.normal(0.0, 0.05, size=n),
        "mean_r": rng.normal(0.0, 0.05, size=n),
        "log_jitter_g": rng.normal(np.log(0.03), 0.1, size=n),
        "log_jitter_r": rng.normal(np.log(0.03), 0.1, size=n),
        "dlog_amp_blr_g": rng.normal(-1.0, 0.2, size=n),
        "dlog_amp_blr_r": rng.normal(-1.0, 0.2, size=n),
        "dlog_amp_blr2_g": rng.normal(-1.2, 0.2, size=n),
        "dlog_amp_blr2_r": rng.normal(-1.2, 0.2, size=n),
        "log_lag_blr_g": rng.normal(np.log(20.0), 0.2, size=n),
        "log_lag_blr_r": rng.normal(np.log(25.0), 0.2, size=n),
        "log_lag_blr2_g": rng.normal(np.log(70.0), 0.2, size=n),
        "log_lag_blr2_r": rng.normal(np.log(80.0), 0.2, size=n),
        "dlog_amp_bc": rng.normal(-1.1, 0.2, size=n),
        "log_lag_ratio_bc_to_blr": rng.uniform(np.log(0.1), np.log(0.3), size=n),
    }

    kls = compute_parameter_kls(
        flat_samples,
        bands=bands,
        survey_names=("sdss", "ps1", "ztf"),
        t_ref=np.array([0.0, 10.0, 20.0], dtype=float),
        z=z,
        lambda_center_rf=lambda_center_rf,
        log_jitter_mean=np.asarray([np.log(0.03), np.log(0.03)]),
        disable_lag_bc=False,
        n_blr_terms=2,
    )

    expected_keys = {
        "eta_sigma_kl",
        "eta_tau_kl",
        "log_sigma_center0_kl",
        "log_tau_slow_center0_kl",
        "log_tau_fast_center0_kl",
        "lag0_kl",
        "lag_beta_kl",
        "mean_g_kl",
        "mean_r_kl",
        "log_jitter_g_kl",
        "log_jitter_r_kl",
        "dlog_amp_blr_g_kl",
        "dlog_amp_blr2_g_kl",
        "dlog_amp_bc_kl",
        "log_lag_blr_g_kl",
        "log_lag_blr2_g_kl",
        "log_lag_ratio_bc_to_blr_kl",
        "kl_total",
    }
    assert expected_keys.issubset(kls.keys())
    assert all(np.isfinite(kls[key]) for key in expected_keys)


def test_compute_parameter_kls_includes_band_slope_offset_terms():
    rng = np.random.default_rng(456)
    bands = ["g", "r"]
    n = 64
    flat_samples = {
        "eta_sigma": rng.normal(-0.45, 0.08, size=n),
        "eta_tau": rng.normal(0.55, 0.10, size=n),
        "log_sigma_center0": rng.normal(np.log(0.2), 0.2, size=n),
        "log_tau_slow_center0": rng.normal(np.log(300.0), 0.3, size=n),
        "log_tau_fast_center0": rng.normal(np.log(30.0), 0.2, size=n),
        "linear_trend": rng.normal(0.0, 0.05, size=n),
        "linear_trend_band_offset_g": rng.normal(0.005, 0.01, size=n),
        "linear_trend_band_offset_r": rng.normal(-0.005, 0.01, size=n),
        "lag0": np.abs(rng.normal(5.0, 1.0, size=n)),
        "lag_beta": np.abs(rng.normal(4.0 / 3.0, 0.1, size=n)),
        "mean_g": rng.normal(0.0, 0.05, size=n),
        "mean_r": rng.normal(0.0, 0.05, size=n),
        "log_jitter_g": rng.normal(np.log(0.03), 0.1, size=n),
        "log_jitter_r": rng.normal(np.log(0.03), 0.1, size=n),
        "dlog_amp_blr_g": rng.normal(-1.0, 0.2, size=n),
        "dlog_amp_blr_r": rng.normal(-1.0, 0.2, size=n),
        "log_lag_blr_g": rng.normal(np.log(20.0), 0.2, size=n),
        "log_lag_blr_r": rng.normal(np.log(25.0), 0.2, size=n),
    }

    kls = compute_parameter_kls(
        flat_samples,
        bands=bands,
        survey_names=("sdss", "ps1", "ztf"),
        t_ref=np.array([0.0, 10.0, 20.0], dtype=float),
        z=1.2,
        lambda_center_rf=2500.0,
        log_jitter_mean=np.asarray([np.log(0.03), np.log(0.03)]),
        disable_lag_bc=True,
        n_blr_terms=1,
    )

    assert "linear_trend_band_offset_g_kl" in kls
    assert "linear_trend_band_offset_r_kl" in kls
    assert np.isfinite(kls["linear_trend_band_offset_g_kl"])
    assert np.isfinite(kls["linear_trend_band_offset_r_kl"])


def test_compute_parameter_kls_uses_relflux_sigma_center0_when_requested():
    rng = np.random.default_rng(321)
    n = 32
    flat_samples = {
        "eta_sigma": rng.normal(-0.45, 0.08, size=n),
        "eta_tau": rng.normal(0.55, 0.10, size=n),
        "log_sigma_center0": rng.normal(-9.0, 0.05, size=n),
        "log_sigma_center0_relflux": rng.normal(np.log(0.15), 0.15, size=n),
        "log_tau_slow_center0": rng.normal(np.log(300.0), 0.3, size=n),
        "log_tau_fast_center0": rng.normal(np.log(30.0), 0.2, size=n),
        "linear_trend": rng.normal(0.0, 0.05, size=n),
        "lag0": np.abs(rng.normal(5.0, 1.0, size=n)),
        "lag_beta": np.abs(rng.normal(4.0 / 3.0, 0.1, size=n)),
        "mean_g": rng.normal(0.0, 0.05, size=n),
        "log_jitter_g": rng.normal(np.log(0.03), 0.1, size=n),
        "dlog_amp_blr_g": rng.normal(-1.0, 0.2, size=n),
        "log_lag_blr_g": rng.normal(np.log(20.0), 0.2, size=n),
        "dlog_amp_bc": rng.normal(-1.1, 0.2, size=n),
        "log_lag_ratio_bc_to_blr": rng.uniform(np.log(0.1), np.log(0.3), size=n),
    }

    kls = compute_parameter_kls(
        flat_samples,
        bands=["g"],
        survey_names=("sdss", "ps1", "ztf"),
        t_ref=np.array([0.0, 10.0, 20.0], dtype=float),
        z=1.2,
        lambda_center_rf=2500.0,
        log_jitter_mean=np.asarray([np.log(0.03)]),
        model_variant="mag_flux_linearized",
        disable_lag_bc=False,
        n_blr_terms=1,
    )

    assert "log_sigma_center0_kl" in kls
    assert np.isfinite(kls["log_sigma_center0_kl"])


def test_posterior_median_mean_function_uses_global_time_normalization():
    t_eval = np.array([0.0, 20.0], dtype=float)
    t_ref = np.array([0.0, 10.0, 20.0], dtype=float)
    flat_samples = {
        "mean_g": np.array([0.8, 1.0, 1.2], dtype=float),
        "linear_trend": np.array([0.1, 0.2, 0.3], dtype=float),
    }

    got = posterior_median_mean_function(flat_samples, t_eval, "g", t_ref=t_ref)

    t_center = 10.0
    t_std = np.std(t_ref)
    expected = 1.0 + 0.2 * ((t_eval - t_center) / t_std)
    assert np.allclose(got, expected)


def test_posterior_median_mean_function_uses_band_specific_slope_offsets():
    t_eval = np.array([0.0, 20.0], dtype=float)
    t_ref = np.array([0.0, 10.0, 20.0], dtype=float)
    flat_samples = {
        "mean_g": np.array([1.0, 1.0, 1.0], dtype=float),
        "linear_trend": np.array([0.2, 0.2, 0.2], dtype=float),
        "linear_trend_band_offset_g": np.array([0.05, 0.05, 0.05], dtype=float),
    }

    got = posterior_median_mean_function(flat_samples, t_eval, "g", t_ref=t_ref)

    t_center = 10.0
    t_std = np.std(t_ref)
    expected = 1.0 + 0.25 * ((t_eval - t_center) / t_std)
    assert np.allclose(got, expected)


def test_posterior_median_mean_function_applies_survey_offsets_in_mag_space():
    t_eval = np.array([0.0, 20.0], dtype=float)
    t_ref = np.array([0.0, 10.0, 20.0], dtype=float)
    survey_idx = np.array([0, 2], dtype=np.int32)
    flat_samples = {
        "mean_g": np.array([1.0, 1.0, 1.0], dtype=float),
        "linear_trend": np.array([0.0, 0.0, 0.0], dtype=float),
        "survey_delta_mag_g_sdss": np.zeros(3, dtype=float),
        "survey_delta_mag_g_ztf": np.full(3, 0.015, dtype=float),
    }

    got = posterior_median_mean_function(
        flat_samples,
        t_eval,
        "g",
        t_ref=t_ref,
        survey_idx=survey_idx,
        survey_names=("sdss", "ps1", "ztf"),
    )

    assert np.allclose(got, np.array([1.0, 1.015], dtype=float))


def test_compute_survey_offset_active_mask_uses_first_active_survey_as_reference():
    band_idx = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    survey_idx = np.array([1, 2, 2, 2, 2], dtype=np.int32)

    got = _compute_survey_offset_active_mask(band_idx, survey_idx, n_bands=2)

    expected = np.array(
        [
            [False, False, True],
            [False, False, False],
        ],
        dtype=bool,
    )
    assert np.array_equal(got, expected)


def test_default_survey_labels_keep_full_survey_name():
    assert np.array_equal(
        _default_survey_labels_for_band("g", 3),
        np.array(["ztf", "ztf", "ztf"], dtype=str),
    )
    assert np.array_equal(
        _default_survey_labels_for_band("u", 2),
        np.array(["sdss", "sdss"], dtype=str),
    )


def test_compute_band_adf_returns_valid_output_for_stationary_series():
    rng = np.random.default_rng(7)
    noise = rng.normal(scale=0.3, size=256)
    x = np.empty_like(noise)
    x[0] = noise[0]
    for i in range(1, x.size):
        x[i] = 0.5 * x[i - 1] + noise[i]

    result = compute_band_adf(x)

    assert result["adf_valid"] is True
    assert np.isfinite(result["adf_stat"])
    assert np.isfinite(result["adf_pvalue"])
    assert result["adf_nobs"] > 0


def test_compute_band_adf_rejects_constant_or_short_series():
    short_result = compute_band_adf(np.array([1.0, 2.0, 3.0], dtype=float))
    const_result = compute_band_adf(np.ones(16, dtype=float))

    assert short_result["adf_valid"] is False
    assert np.isnan(short_result["adf_stat"])
    assert const_result["adf_valid"] is False
    assert np.isnan(const_result["adf_pvalue"])


def test_empirical_structure_function_matches_stone_definition():
    t = np.array([0.0, 1.0, 2.0], dtype=float)
    y = np.array([0.0, 1.0, 2.0], dtype=float)
    yerr = np.array([0.1, 0.2, 0.3], dtype=float)

    tau, sf, sf_lo, sf_hi = empirical_structure_function(
        t,
        y,
        yerr,
        bins_per_decade=1,
        min_pairs=3,
    )

    pair_terms = np.array(
        [
            1.0**2 - 0.1**2 - 0.2**2,
            2.0**2 - 0.1**2 - 0.3**2,
            1.0**2 - 0.2**2 - 0.3**2,
        ],
        dtype=float,
    )
    pair_weights = 1.0 / np.square(
        np.array(
            [
                0.1**2 + 0.2**2,
                0.1**2 + 0.3**2,
                0.2**2 + 0.3**2,
            ],
            dtype=float,
        )
    )
    expected_tau = np.mean([1.0, 2.0, 1.0])
    expected_sf = np.sqrt(_weighted_quantile(pair_terms, pair_weights, [0.5])[0])

    assert tau.shape == (1,)
    assert np.isclose(tau[0], expected_tau)
    assert np.isclose(sf[0], expected_sf)
    assert sf_lo[0] <= sf[0] <= sf_hi[0]


def test_compute_structure_function_diagnostics_returns_finite_sensible_g_band_fit():
    rng = np.random.default_rng(11)
    sigma_true = 0.18
    tau_true = 700.0
    t_g = np.linspace(0.0, 3000.0, 40, dtype=float)
    dt = np.abs(t_g[:, None] - t_g[None, :])
    cov = (sigma_true**2) * np.exp(-dt / tau_true)
    y_g = rng.multivariate_normal(np.zeros(t_g.size, dtype=float), cov)
    yerr_g = np.full(t_g.size, 0.01, dtype=float)

    t_r = np.linspace(0.0, 3000.0, 8, dtype=float)
    y_r = np.zeros(t_r.size, dtype=float)
    yerr_r = np.full(t_r.size, 0.02, dtype=float)

    obj = {
        "bands": ["g", "r"],
        "band_idx": np.array([0] * t_g.size + [1] * t_r.size, dtype=int),
        "X": (np.concatenate([t_g, t_r]), np.array([0] * t_g.size + [1] * t_r.size, dtype=int)),
        "y": np.concatenate([y_g, y_r]),
        "yerr": np.concatenate([yerr_g, yerr_r]),
    }
    samples = {
        "eta_sigma": np.array([0.0, 0.05, -0.05], dtype=float),
        "eta_tau": np.array([0.0, 0.05, -0.05], dtype=float),
        "amp_cont_g": np.array([0.18, 0.19, 0.20], dtype=float),
        "tau_fast_g": np.array([75.0, 70.0, 80.0], dtype=float),
        "tau_slow_g": np.array([720.0, 700.0, 740.0], dtype=float),
    }

    result = compute_structure_function_diagnostics(samples, obj, z=0.8)

    assert result["sf_ref_band"] == "g"
    assert np.isfinite(result["log_sigma_sf_ref_band"])
    assert np.isfinite(result["log_tau_sf_ref_band"])
    assert 1.0 < result["log_tau_sf_ref_band"] < 4.0
    assert -2.0 < result["log_sigma_sf_ref_band"] < 0.5
    assert np.isfinite(result["log_sigma_sf_model_ref_band"])
    assert np.isfinite(result["log_tau_sf_model_ref_band"])
    assert result["log_sigma_sf_ref_band"] > np.log10(np.median(samples["amp_cont_g"]))
    assert np.isclose(
        result["log_sigma_sf_model_ref_band"],
        np.log10(np.median(samples["amp_cont_g"])),
        atol=0.15,
    )
    assert result["log_tau_sf_model_ref_band"] < np.log10(np.median(samples["tau_slow_g"]))


def test_compute_object_adf_diagnostics_returns_per_band_fields():
    obj = {
        "X": (np.tile(np.arange(24, dtype=float), 2), np.array([0] * 24 + [1] * 24, dtype=int)),
        "y": np.concatenate(
            [
                np.sin(np.arange(24, dtype=float) / 4.0),
                np.cos(np.arange(24, dtype=float) / 5.0),
            ]
        ),
        "band_idx": np.array([0] * 24 + [1] * 24, dtype=int),
    }
    flat_samples = {
        "mean_g": np.array([0.0, 0.1, 0.0], dtype=float),
        "mean_r": np.array([0.0, -0.1, 0.0], dtype=float),
        "linear_trend": np.array([0.0, 0.01, 0.0], dtype=float),
    }

    result = compute_object_adf_diagnostics(flat_samples, obj, ["g", "r"])

    for band in ("g", "r"):
        assert f"adf_stat_{band}" in result
        assert f"adf_pvalue_{band}" in result
        assert f"adf_usedlag_{band}" in result
        assert f"adf_nobs_{band}" in result
        assert f"adf_valid_{band}" in result
    assert "adf_min_pvalue" in result
    assert "adf_any_pvalue_lt_0p05" in result


def test_compute_g_band_residual_drift_diagnostics_returns_finite_positive_slopes():
    t_bins = [
        np.array([100.0, 250.0, 500.0, 750.0], dtype=float),
        np.array([1100.0, 1250.0, 1500.0, 1750.0], dtype=float),
        np.array([2100.0, 2250.0, 2500.0, 2750.0], dtype=float),
        np.array([3100.0, 3250.0, 3500.0, 3750.0], dtype=float),
    ]
    means = [0.0, 0.2, 0.4, 0.6]
    spreads = [0.05, 0.1, 0.2, 0.4]
    y_chunks = []
    for mu, spread in zip(means, spreads):
        y_chunks.append(mu + np.array([-1.5, -0.5, 0.5, 1.5], dtype=float) * spread)

    t_g = np.concatenate(t_bins)
    y_g = np.concatenate(y_chunks)
    yerr_g = np.full(t_g.size, 0.03, dtype=float)

    obj = {
        "z": 0.0,
        "X": (t_g, np.zeros(t_g.size, dtype=int)),
        "y": y_g,
        "yerr": yerr_g,
        "band_idx": np.zeros(t_g.size, dtype=int),
    }
    flat_samples = {
        "mean_g": np.array([0.0, 0.0, 0.0], dtype=float),
        "linear_trend": np.array([0.0, 0.0, 0.0], dtype=float),
    }

    result = compute_g_band_residual_drift_diagnostics(
        flat_samples,
        obj,
        ["g"],
        z=0.0,
        bin_width_rf_days=1000.0,
        min_count=4,
    )

    assert result["g_resid_n_bins"] == 4
    assert result["g_resid_mean_trend_valid"] is True
    assert result["g_resid_var_trend_valid"] is True
    assert np.isfinite(result["g_resid_mean_slope"])
    assert np.isfinite(result["g_resid_mean_slope_err"])
    assert np.isfinite(result["g_resid_var_slope"])
    assert np.isfinite(result["g_resid_var_slope_err"])
    assert result["g_resid_mean_slope"] > 0.0
    assert result["g_resid_var_slope"] > 0.0


def test_compute_g_band_raw_drift_diagnostics_returns_finite_positive_mean_slope():
    t_bins = [
        np.array([100.0, 250.0, 500.0, 750.0], dtype=float),
        np.array([1100.0, 1250.0, 1500.0, 1750.0], dtype=float),
        np.array([2100.0, 2250.0, 2500.0, 2750.0], dtype=float),
        np.array([3100.0, 3250.0, 3500.0, 3750.0], dtype=float),
    ]
    means = [19.0, 19.2, 19.4, 19.6]
    spreads = [0.05, 0.08, 0.1, 0.12]
    y_chunks = []
    for mu, spread in zip(means, spreads):
        y_chunks.append(mu + np.array([-1.5, -0.5, 0.5, 1.5], dtype=float) * spread)

    t_g = np.concatenate(t_bins)
    y_g = np.concatenate(y_chunks)
    yerr_g = np.full(t_g.size, 0.03, dtype=float)

    obj = {
        "z": 0.0,
        "X": (t_g, np.zeros(t_g.size, dtype=int)),
        "y": y_g,
        "yerr": yerr_g,
        "band_idx": np.zeros(t_g.size, dtype=int),
    }
    flat_samples = {
        "mean_g": np.array([19.0, 19.0, 19.0], dtype=float),
        "linear_trend": np.array([0.0, 0.0, 0.0], dtype=float),
    }

    result = compute_g_band_raw_drift_diagnostics(
        flat_samples,
        obj,
        ["g"],
        z=0.0,
        bin_width_rf_days=1000.0,
        min_count=4,
    )

    assert result["g_raw_n_bins"] == 4
    assert result["g_raw_mean_trend_valid"] is True
    assert np.isfinite(result["g_raw_mean_slope"])
    assert np.isfinite(result["g_raw_mean_slope_err"])
    assert result["g_raw_mean_slope"] > 0.0


def test_fit_bending_power_law_psd_recovers_tau_and_sigma():
    freq = np.logspace(-4.5, -1.5, 60)
    true_log_sigma = np.log10(0.2)
    true_log_tau = np.log10(300.0)
    true_alpha_high = -2.1
    power = bending_power_law_psd(freq, true_log_sigma, true_log_tau, alpha_high=true_alpha_high)
    power_lo = power * 0.9
    power_hi = power * 1.1

    result = fit_bending_power_law_psd(freq, power, power_lo, power_hi)

    assert result["psd_bpl_valid"] is True
    assert np.isclose(result["log_sigma_bpl"], true_log_sigma, atol=0.05)
    assert np.isclose(result["log_tau_bpl"], true_log_tau, atol=0.05)
    assert np.isclose(result["psd_bpl_alpha_high"], true_alpha_high, atol=0.15)


def test_fit_bending_power_law_psd_handles_too_few_bins():
    result = fit_bending_power_law_psd(
        np.array([1e-3, 2e-3, 3e-3]),
        np.array([1.0, 2.0, 3.0]),
    )

    assert result["psd_bpl_valid"] is False
    assert np.isnan(result["log_sigma_bpl"])
    assert np.isnan(result["psd_bpl_alpha_high"])


def test_fit_fixed_slope_drw_psd_recovers_sigma_and_tau():
    freq = np.logspace(-4.5, -1.5, 60)
    true_log_sigma = np.log10(0.2)
    true_log_tau = np.log10(300.0)
    power = bending_power_law_psd(freq, true_log_sigma, true_log_tau, alpha_high=-2.0)

    result = fit_fixed_slope_drw_psd(freq, power, power * 0.9, power * 1.1)

    assert result["valid"] is True
    assert np.isclose(result["log_sigma"], true_log_sigma, atol=0.02)
    assert np.isclose(result["log_tau"], true_log_tau, atol=0.02)


def test_relative_to_2500_amplitude_scale_uses_only_eta_sigma_and_rest_wavelength():
    lam_rf = np.array([1800.0, 2500.0, 4000.0], dtype=float)
    eta_sigma = -0.6

    got = relative_to_2500_amplitude_scale(lam_rf, eta_sigma)
    expected = np.array(
        [10.0 ** log_single_pl(2500.0, lam, eta_sigma) for lam in lam_rf],
        dtype=float,
    )

    assert np.allclose(got, expected)
    assert np.isclose(got[1], 1.0)


def test_relative_to_2500_amplitude_scale_is_independent_of_absolute_sigma():
    lam_rf = np.array([1900.0, 3200.0], dtype=float)
    eta_sigma = -0.4
    params_a = {"eta_sigma": eta_sigma, "log_sigma_uv": -2.0}
    params_b = {"eta_sigma": eta_sigma, "log_sigma_uv": 3.0}

    got_a = relative_to_2500_amplitude_scale(lam_rf, params_a["eta_sigma"])
    got_b = relative_to_2500_amplitude_scale(lam_rf, params_b["eta_sigma"])

    assert np.allclose(got_a, got_b)
