import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble import (
    hubble_fit,
    hubble_likelihood,
    hubble_model,
    hubble_plotting,
    hubble_utils,
)


FLAG = "use_f_agn_psf_2500_sigmoid_term"


def _sigmoid_inputs(*, alpha=False, eta=False):
    n = 5
    data = {
        "object_id": np.array([f"q{i}" for i in range(n)]),
        "z": np.linspace(0.5, 2.0, n),
        "log_sigma_uv": np.linspace(-1.0, -0.6, n),
        "log_tau_uv_rf": np.linspace(2.4, 2.8, n),
        "log_sigma_uv_std_psd": np.zeros(n),
        "log_tau_uv_rf_std_psd": np.zeros(n),
        "log_sigma_uv_log_tau_uv_rf_cov_psd": np.zeros(n),
        "f_AGN_psf_2500": np.array([0.1, 0.3, 0.5, 0.8, 0.95]),
        "f_AGN_psf_2500_err": np.full(n, 0.02),
    }
    params = {
        "M0_agn": -23.0,
        "alpha_agn": -1.2,
        "beta_agn": 0.7,
        "A_f_agn_psf_2500": 1.8,
        "log_k_f_agn_psf_2500": np.log(7.0),
        "x0_f_agn_psf_2500": 0.55,
    }
    if alpha:
        data["alpha_lambda"] = np.linspace(-2.0, -1.4, n)
        data["alpha_lambda_err"] = np.full(n, 0.03)
        params["gamma_alpha_lambda"] = 0.4
    if eta:
        data["eta_sigma"] = np.linspace(-0.6, -0.3, n)
        data["eta_sigma_err"] = np.full(n, 0.04)
        params["gamma_eta_sigma"] = -0.5
    context = hubble_model.build_agn_pivot_context(
        data,
        (0.5, 2.0),
        use_alpha_lambda_term=alpha,
        use_eta_sigma_term=eta,
        use_f_agn_psf_2500_sigmoid_term=True,
    )
    obs, err, pivots = hubble_model.agn_model_pack_obs(
        data,
        use_alpha_lambda_term=alpha,
        use_eta_sigma_term=eta,
        use_f_agn_psf_2500_sigmoid_term=True,
        pivot_context=context,
    )
    packed_params = hubble_model.agn_model_pack_params(
        params,
        use_alpha_lambda_term=alpha,
        use_eta_sigma_term=eta,
        use_f_agn_psf_2500_sigmoid_term=True,
    )
    return data, params, context, packed_params, obs, err, pivots


def test_source_columns_are_derived_without_schema_change():
    frame = pd.DataFrame(
        {
            "f_host_2500_psf": [0.0, 0.25, 1.0],
            "f_host_2500_psf_err": [0.01, 0.02, 0.03],
        }
    )
    result = hubble_utils.derive_f_agn_psf_2500_columns(frame)
    np.testing.assert_allclose(result["f_AGN_psf_2500"], [1.0, 0.75, 0.0])
    np.testing.assert_allclose(result["f_AGN_psf_2500_err"], [0.01, 0.02, 0.03])
    assert "f_host_2500_psf" in result


def test_anchored_formula_has_positive_steepness_and_is_zero_at_pivot():
    f = np.array([0.2, 0.5, 0.9])
    pivot = 0.5
    amplitude = 2.0
    log_k = np.log(8.0)
    x0 = 0.6
    actual = hubble_model.anchored_f_agn_psf_2500_sigmoid(
        f, pivot, amplitude, log_k, x0
    )
    expected = amplitude * (
        hubble_model.expit(8.0 * (f - x0))
        - hubble_model.expit(8.0 * (pivot - x0))
    )
    np.testing.assert_allclose(actual, expected)
    assert actual[0] < actual[1] < actual[2]
    assert actual[1] == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("alpha", [False, True])
@pytest.mark.parametrize("eta", [False, True])
def test_scalar_vectorized_model_agree_for_all_optional_linear_terms(alpha, eta):
    _, _, _, params, obs, _, pivots = _sigmoid_inputs(alpha=alpha, eta=eta)
    kwargs = {
        "use_alpha_lambda_term": alpha,
        "use_eta_sigma_term": eta,
        "use_f_agn_psf_2500_sigmoid_term": True,
    }
    scalar = hubble_model.M_model_agn(params, obs, pivots, **kwargs)
    vectorized = hubble_model.M_model_agn_posterior_samples(
        params[None, :], obs, pivots, **kwargs
    )[0]
    np.testing.assert_allclose(vectorized, scalar, rtol=1e-13, atol=1e-13)


def test_zero_amplitude_is_exactly_the_baseline_model():
    _, params_dict, context, _, _, _, _ = _sigmoid_inputs()
    params_dict["A_f_agn_psf_2500"] = 0.0
    data, _, _, _, _, _, _ = _sigmoid_inputs()
    base_context = hubble_model.build_agn_pivot_context(data, (0.5, 2.0))
    base_obs, _, base_pivots = hubble_model.agn_model_pack_obs(
        data, pivot_context=base_context
    )
    base_params = hubble_model.agn_model_pack_params(params_dict)
    baseline = hubble_model.M_model_agn(base_params, base_obs, base_pivots)
    sig_obs, _, sig_pivots = hubble_model.agn_model_pack_obs(
        data,
        use_f_agn_psf_2500_sigmoid_term=True,
        pivot_context=context,
    )
    sig_params = hubble_model.agn_model_pack_params(
        params_dict, use_f_agn_psf_2500_sigmoid_term=True
    )
    with_sigmoid = hubble_model.M_model_agn(
        sig_params,
        sig_obs,
        sig_pivots,
        use_f_agn_psf_2500_sigmoid_term=True,
    )
    np.testing.assert_array_equal(with_sigmoid, baseline)


def test_zero_amplitude_likelihood_equals_baseline():
    data, params, sigmoid_context, _, _, _, _ = _sigmoid_inputs()
    data["z_err"] = np.full(len(data["z"]), 0.001)
    data["apparent_mag_2500_err"] = np.full(len(data["z"]), 0.05)
    params.update({"log_f": np.log(0.5), "H0": 70.0, "Om0": 0.3})
    params["A_f_agn_psf_2500"] = 0.0
    base_context = hubble_model.build_agn_pivot_context(data, (0.5, 2.0))
    base_obs, _, base_pivots = hubble_model.agn_model_pack_obs(
        data, pivot_context=base_context
    )
    base_model_params = hubble_model.agn_model_pack_params(params)
    absolute_mag = hubble_model.M_model_agn(
        base_model_params, base_obs, base_pivots
    )
    from astropy.cosmology import FlatLambdaCDM

    data["apparent_mag_2500"] = (
        absolute_mag + FlatLambdaCDM(H0=70.0, Om0=0.3).distmod(data["z"]).value
    )

    def evaluate(enabled, context):
        _, labels, _ = hubble_model.get_model_params(
            "FlatLambdaCDM",
            only_agn=True,
            use_f_agn_psf_2500_sigmoid_term=enabled,
        )
        return hubble_likelihood.log_likelihood(
            np.asarray([params[name] for name in labels]),
            agn_data=data,
            pantheon_data={},
            _sna_L=None,
            _sna_Lower=None,
            _sna_LogdetCov=None,
            cosmo_model="FlatLambdaCDM",
            completeness_params=None,
            z_pivot_agn=1.5,
            agn_pivot_context=context,
            only_agn=True,
            use_f_agn_psf_2500_sigmoid_term=enabled,
        )[0]

    assert evaluate(True, sigmoid_context) == pytest.approx(
        evaluate(False, base_context), rel=0.0, abs=1e-12
    )


def test_analytic_error_matches_finite_difference():
    data, _, _, params, obs, err, pivots = _sigmoid_inputs()
    predicted_err = hubble_model.M_model_agn_err(
        params,
        obs,
        err,
        pivots,
        use_f_agn_psf_2500_sigmoid_term=True,
    )
    f_index = hubble_model.get_agn_model_spec(
        use_f_agn_psf_2500_sigmoid_term=True
    )[1].index("f_AGN_psf_2500")
    step = 1e-6
    obs_hi = obs.copy()
    obs_lo = obs.copy()
    obs_hi[f_index] += step
    obs_lo[f_index] -= step
    derivative = (
        hubble_model.M_model_agn(
            params, obs_hi, pivots, use_f_agn_psf_2500_sigmoid_term=True
        )
        - hubble_model.M_model_agn(
            params, obs_lo, pivots, use_f_agn_psf_2500_sigmoid_term=True
        )
    ) / (2.0 * step)
    np.testing.assert_allclose(
        predicted_err,
        np.abs(derivative) * data["f_AGN_psf_2500_err"],
        rtol=2e-9,
        atol=1e-11,
    )


def test_posterior_error_component_is_explicit_parameter_average():
    _, _, _, params, obs, err, pivots = _sigmoid_inputs()
    samples = np.vstack([params, params.copy()])
    samples[1, -3] *= -0.5
    samples[1, -2] = np.log(20.0)
    variance, components = hubble_model.M_model_agn_observable_variance_posterior(
        samples,
        err,
        use_f_agn_psf_2500_sigmoid_term=True,
        obs_arr=obs,
        pivots_array=pivots,
    )
    f = obs[-1][None, :]
    amplitude = samples[:, -3, None]
    k = np.exp(samples[:, -2, None])
    x0 = samples[:, -1, None]
    s = hubble_model.expit(k * (f - x0))
    expected = np.mean((amplitude * k * s * (1.0 - s)) ** 2, axis=0) * err[-1] ** 2
    np.testing.assert_allclose(components["f_agn_psf_2500_sigmoid"], expected)
    np.testing.assert_allclose(variance, expected)


def test_synthetic_sigmoid_amplitude_is_recovered():
    f = np.linspace(0.05, 0.98, 80)
    pivot = np.median(f)
    true_amplitude = -2.3
    basis = hubble_model.anchored_f_agn_psf_2500_sigmoid(
        f, pivot, 1.0, np.log(12.0), 0.62
    )
    synthetic_delta_m = true_amplitude * basis
    fitted_amplitude = np.dot(basis, synthetic_delta_m) / np.dot(basis, basis)
    assert fitted_amplitude == pytest.approx(true_amplitude, rel=1e-12)


@pytest.mark.parametrize("bad_fraction", [-0.01, 1.01, np.nan])
def test_fraction_bounds_are_rejected(bad_fraction):
    data, _, context, _, _, _, _ = _sigmoid_inputs()
    data["f_AGN_psf_2500"][2] = bad_fraction
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        hubble_model.agn_model_pack_obs(
            data,
            use_f_agn_psf_2500_sigmoid_term=True,
            pivot_context=context,
        )


@pytest.mark.parametrize("bad_error", [-0.01, np.nan])
def test_fraction_errors_are_rejected(bad_error):
    data, _, context, _, _, _, _ = _sigmoid_inputs()
    data["f_AGN_psf_2500_err"][1] = bad_error
    with pytest.raises(ValueError, match="finite and nonnegative"):
        hubble_model.agn_model_pack_obs(
            data,
            use_f_agn_psf_2500_sigmoid_term=True,
            pivot_context=context,
        )


def test_parameter_order_priors_labels_tag_and_legacy_inference():
    priors, labels, latex = hubble_model.get_model_params(
        "FlatLambdaCDM", use_f_agn_psf_2500_sigmoid_term=True
    )
    sigmoid_names = [
        "A_f_agn_psf_2500",
        "log_k_f_agn_psf_2500",
        "x0_f_agn_psf_2500",
    ]
    assert [name for name in labels if name in sigmoid_names] == sigmoid_names
    assert priors[sigmoid_names[0]] == (-10.0, 10.0)
    np.testing.assert_allclose(priors[sigmoid_names[1]], np.log([0.5, 100.0]))
    assert priors[sigmoid_names[2]] == (-0.5, 1.5)
    assert all(latex[labels.index(name)].startswith("$") for name in sigmoid_names)
    assert "_fagnPsf2500Sigmoid" in hubble_fit.make_run_tag(
        "FlatLambdaCDM",
        False,
        "fastest",
        None,
        (0.44, 3.16),
        completeness=False,
        use_f_agn_psf_2500_sigmoid_term=True,
    )

    legacy = hubble_model.resolve_model_option_flags(
        "FlatLambdaCDM", len(labels), only_agn=False
    )
    assert legacy[FLAG] is False
    explicit = hubble_model.resolve_model_option_flags(
        "FlatLambdaCDM",
        len(labels),
        only_agn=False,
        use_alpha_lambda_term=False,
        use_eta_sigma_term=False,
        use_redshift_log_f_term=False,
        use_f_agn_psf_2500_sigmoid_term=True,
    )
    assert explicit[FLAG] is True


def test_cli_declares_option_and_rejects_it_with_jax():
    source = inspect.getsource(hubble_fit)
    assert '"--fit_f_agn_psf_2500_sigmoid_term"' in source
    assert "args.use_jax and args.fit_f_agn_psf_2500_sigmoid_term" in source
    assert "CPU/Dynesty" in source


def test_hubble_prediction_writes_sigmoid_error_budget(tmp_path):
    data, params, context, _, _, _, _ = _sigmoid_inputs()
    frame = pd.DataFrame(data)
    frame["apparent_mag_2500"] = np.linspace(20.5, 22.0, len(frame))
    frame["apparent_mag_2500_err"] = 0.05
    frame["z_err"] = 0.002
    frame["is_fit_selection"] = True
    priors, labels, _ = hubble_model.get_model_params(
        "FlatLambdaCDM", use_f_agn_psf_2500_sigmoid_term=True
    )
    center = {
        "M0_sn": -19.3,
        **params,
        "log_f": np.log(0.5),
        "H0": 70.0,
        "Om0": 0.3,
    }
    samples = np.asarray([[center[name] for name in labels]] * 12, dtype=float)
    hubble_plotting.plot_hubble(
        samples,
        frame,
        pd.DataFrame(),
        "FlatLambdaCDM",
        1.5,
        plot_path=str(tmp_path),
        show=False,
        completeness=False,
        z_range=(0.5, 2.0),
        use_f_agn_psf_2500_sigmoid_term=True,
        agn_pivot_context=context,
    )
    budget = pd.read_csv(
        tmp_path / "diagnostics" / "hubble_error_budget_per_object.csv"
    )
    field = "predicted_M2500_f_agn_psf_2500_sigmoid_term"
    assert field in budget
    assert np.all(np.isfinite(budget[field]))
    assert np.any(budget[field] > 0.0)
