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


FLAG = "use_f_agn_psf_2500_flux_fraction_term"
PARAM = "gamma_f_agn_psf_2500_flux_fraction"


def _flux_inputs(*, alpha=False, eta=False):
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
        PARAM: 1.4,
    }
    if alpha:
        data["alpha_lambda"] = np.linspace(-2.0, -1.4, n)
        data["alpha_lambda_err"] = np.full(n, 0.03)
        params["gamma_alpha_lambda"] = 0.4
    if eta:
        data["eta_sigma"] = np.linspace(-0.6, -0.3, n)
        data["eta_sigma_err"] = np.full(n, 0.04)
        params["gamma_eta_sigma"] = -0.5
    kwargs = {
        "use_alpha_lambda_term": alpha,
        "use_eta_sigma_term": eta,
        FLAG: True,
    }
    context = hubble_model.build_agn_pivot_context(data, (0.5, 2.0), **kwargs)
    obs, err, pivots = hubble_model.agn_model_pack_obs(
        data, pivot_context=context, **kwargs
    )
    packed_params = hubble_model.agn_model_pack_params(params, **kwargs)
    return data, params, context, packed_params, obs, err, pivots


def test_flux_fraction_formula_is_physical_and_pivot_anchored():
    f = np.array([0.25, 0.5, 1.0])
    pivot = 0.5
    actual = hubble_model.anchored_f_agn_psf_2500_flux_fraction(
        f, pivot, 1.0
    )
    np.testing.assert_allclose(actual, -2.5 * np.log10(f / pivot))
    assert actual[1] == pytest.approx(0.0, abs=0.0)


@pytest.mark.parametrize("alpha", [False, True])
@pytest.mark.parametrize("eta", [False, True])
def test_scalar_and_vectorized_models_agree(alpha, eta):
    _, _, _, params, obs, _, pivots = _flux_inputs(alpha=alpha, eta=eta)
    kwargs = {
        "use_alpha_lambda_term": alpha,
        "use_eta_sigma_term": eta,
        FLAG: True,
    }
    scalar = hubble_model.M_model_agn(params, obs, pivots, **kwargs)
    vectorized = hubble_model.M_model_agn_posterior_samples(
        params[None, :], obs, pivots, **kwargs
    )[0]
    np.testing.assert_allclose(vectorized, scalar, rtol=1e-13, atol=1e-13)


def test_zero_gamma_is_exactly_the_baseline_model_and_likelihood():
    data, params, context, _, _, _, _ = _flux_inputs()
    params[PARAM] = 0.0
    data["z_err"] = np.full(len(data["z"]), 0.001)
    data["apparent_mag_2500_err"] = np.full(len(data["z"]), 0.05)
    params.update({"log_f": np.log(0.5), "H0": 70.0, "Om0": 0.3})
    base_context = hubble_model.build_agn_pivot_context(data, (0.5, 2.0))
    base_obs, _, base_pivots = hubble_model.agn_model_pack_obs(
        data, pivot_context=base_context
    )
    base_params = hubble_model.agn_model_pack_params(params)
    baseline = hubble_model.M_model_agn(base_params, base_obs, base_pivots)
    flux_obs, _, flux_pivots = hubble_model.agn_model_pack_obs(
        data, pivot_context=context, **{FLAG: True}
    )
    flux_params = hubble_model.agn_model_pack_params(params, **{FLAG: True})
    np.testing.assert_array_equal(
        hubble_model.M_model_agn(
            flux_params, flux_obs, flux_pivots, **{FLAG: True}
        ),
        baseline,
    )

    from astropy.cosmology import FlatLambdaCDM

    data["apparent_mag_2500"] = (
        baseline + FlatLambdaCDM(H0=70.0, Om0=0.3).distmod(data["z"]).value
    )

    def evaluate(enabled, pivot_context):
        _, labels, _ = hubble_model.get_model_params(
            "FlatLambdaCDM", only_agn=True, **{FLAG: enabled}
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
            agn_pivot_context=pivot_context,
            only_agn=True,
            **{FLAG: enabled},
        )[0]

    assert evaluate(True, context) == pytest.approx(
        evaluate(False, base_context), rel=0.0, abs=1e-12
    )


def test_analytic_error_matches_finite_difference_and_posterior_average():
    data, _, _, params, obs, err, pivots = _flux_inputs()
    predicted_err = hubble_model.M_model_agn_err(
        params, obs, err, pivots, **{FLAG: True}
    )
    f_index = hubble_model.get_agn_model_spec(**{FLAG: True})[1].index(
        "f_AGN_psf_2500"
    )
    step = 1e-7
    obs_hi, obs_lo = obs.copy(), obs.copy()
    obs_hi[f_index] += step
    obs_lo[f_index] -= step
    derivative = (
        hubble_model.M_model_agn(params, obs_hi, pivots, **{FLAG: True})
        - hubble_model.M_model_agn(params, obs_lo, pivots, **{FLAG: True})
    ) / (2.0 * step)
    np.testing.assert_allclose(
        predicted_err,
        np.abs(derivative) * data["f_AGN_psf_2500_err"],
        rtol=2e-8,
        atol=1e-11,
    )

    samples = np.vstack([params, params.copy()])
    samples[1, -1] = -0.6
    variance, components = hubble_model.M_model_agn_observable_variance_posterior(
        samples, err, obs_arr=obs, pivots_array=pivots, **{FLAG: True}
    )
    f = obs[f_index]
    expected = (
        np.mean(samples[:, -1] ** 2)
        * (2.5 / (np.log(10.0) * f)) ** 2
        * err[-1] ** 2
    )
    np.testing.assert_allclose(
        components["f_agn_psf_2500_flux_fraction"], expected
    )
    np.testing.assert_allclose(variance, expected)


def test_synthetic_gamma_is_recovered():
    f = np.geomspace(0.05, 1.0, 100)
    pivot = np.median(f)
    true_gamma = 1.7
    basis = hubble_model.anchored_f_agn_psf_2500_flux_fraction(f, pivot, 1.0)
    synthetic_delta_m = true_gamma * basis
    fitted_gamma = np.dot(basis, synthetic_delta_m) / np.dot(basis, basis)
    assert fitted_gamma == pytest.approx(true_gamma, rel=1e-12)


@pytest.mark.parametrize("bad_fraction", [0.0, -0.01, 1.01, np.nan])
def test_flux_fraction_bounds_are_rejected(bad_fraction):
    data, _, context, _, _, _, _ = _flux_inputs()
    data["f_AGN_psf_2500"][2] = bad_fraction
    with pytest.raises(ValueError, match=r"within \(0, 1\]"):
        hubble_model.agn_model_pack_obs(
            data, pivot_context=context, **{FLAG: True}
        )


@pytest.mark.parametrize("bad_error", [-0.01, np.nan])
def test_fraction_errors_are_rejected(bad_error):
    data, _, context, _, _, _, _ = _flux_inputs()
    data["f_AGN_psf_2500_err"][1] = bad_error
    with pytest.raises(ValueError, match="finite and nonnegative"):
        hubble_model.agn_model_pack_obs(
            data, pivot_context=context, **{FLAG: True}
        )


def test_parameter_prior_label_tag_and_explicit_only_inference():
    priors, labels, latex = hubble_model.get_model_params(
        "FlatLambdaCDM", **{FLAG: True}
    )
    assert priors[PARAM] == (-5.0, 5.0)
    assert PARAM in labels
    assert latex[labels.index(PARAM)] == r"$\gamma_{f,2500}^{\rm flux}$"
    assert "_fagnPsf2500FluxFraction" in hubble_fit.make_run_tag(
        "FlatLambdaCDM",
        False,
        "fastest",
        None,
        (0.44, 3.16),
        completeness=False,
        **{FLAG: True},
    )

    with pytest.raises(ValueError, match="Ambiguous model option flags"):
        hubble_model.resolve_model_option_flags(
            "FlatLambdaCDM", len(labels), only_agn=False
        )
    explicit = hubble_model.resolve_model_option_flags(
        "FlatLambdaCDM",
        len(labels),
        only_agn=False,
        use_alpha_lambda_term=False,
        use_eta_sigma_term=False,
        use_redshift_log_f_term=False,
        **{FLAG: True},
    )
    assert explicit[FLAG] is True

    _, combined_labels, _ = hubble_model.get_model_params(
        "FlatLambdaCDM",
        use_alpha_lambda_term=True,
        use_eta_sigma_term=True,
        use_redshift_log_f_term=True,
        **{FLAG: True},
    )
    combined = hubble_model.resolve_model_option_flags(
        "FlatLambdaCDM",
        len(combined_labels),
        only_agn=False,
        use_alpha_lambda_term=True,
        use_eta_sigma_term=True,
        use_redshift_log_f_term=True,
        **{FLAG: True},
    )
    assert all(
        combined[name]
        for name in (
            "use_alpha_lambda_term",
            "use_eta_sigma_term",
            "use_redshift_log_f_term",
            FLAG,
        )
    )


def test_sigmoid_and_flux_fraction_are_mutually_exclusive():
    kwargs = {
        "use_f_agn_psf_2500_sigmoid_term": True,
        FLAG: True,
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        hubble_model.get_agn_model_spec(**kwargs)
    with pytest.raises(ValueError, match="mutually exclusive"):
        hubble_model.get_model_params("FlatLambdaCDM", **kwargs)


def test_cli_declares_option_and_rejects_jax_and_sigmoid_combination():
    source = inspect.getsource(hubble_fit)
    assert '"--fit_f_agn_psf_2500_flux_fraction_term"' in source
    assert "args.use_jax and args.fit_f_agn_psf_2500_flux_fraction_term" in source
    assert "args.fit_f_agn_psf_2500_sigmoid_term" in source
    assert "mutually exclusive" in source


def test_resume_metadata_rejects_flux_fraction_mismatch():
    payload = {
        "flat_samples": np.zeros((3, 4)),
        "dmi_max_w": np.zeros(2),
        "dmi_posterior_sigma": np.ones(2),
        "integrals_max_w": np.zeros(2),
        "logZ": 0.0,
        "logZerr": 0.1,
        FLAG: True,
    }
    with pytest.raises(RuntimeError, match="flux-fraction model option"):
        hubble_fit.validate_resume_checkpoint(
            payload, "checkpoint.h5", ndim=4, n_agn=2
        )
    hubble_fit.validate_resume_checkpoint(
        payload,
        "checkpoint.h5",
        ndim=4,
        n_agn=2,
        expected_use_f_agn_psf_2500_flux_fraction_term=True,
    )


def test_result_summary_preserves_flux_fraction_parameter():
    _, labels, _ = hubble_model.get_model_params(
        "FlatLambdaCDM", **{FLAG: True}
    )
    samples = np.zeros((4, len(labels)), dtype=float)
    summary = hubble_utils.extract_cosmo_results_from_samples(
        samples,
        "FlatLambdaCDM",
        only_sna=False,
        use_alpha_lambda_term=False,
        use_eta_sigma_term=False,
        use_redshift_log_f_term=False,
        **{FLAG: True},
    )
    assert PARAM in summary["params"]
    assert summary["param_order"] == labels


def test_hubble_prediction_writes_flux_fraction_error_budget(tmp_path):
    data, params, context, _, _, _, _ = _flux_inputs()
    frame = pd.DataFrame(data)
    frame["apparent_mag_2500"] = np.linspace(20.5, 22.0, len(frame))
    frame["apparent_mag_2500_err"] = 0.05
    frame["z_err"] = 0.002
    frame["is_fit_selection"] = True
    _, labels, _ = hubble_model.get_model_params(
        "FlatLambdaCDM", **{FLAG: True}
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
        agn_pivot_context=context,
        **{FLAG: True},
    )
    budget = pd.read_csv(
        tmp_path / "diagnostics" / "hubble_error_budget_per_object.csv"
    )
    field = "predicted_M2500_f_agn_psf_2500_flux_fraction_term"
    assert field in budget
    assert np.all(np.isfinite(budget[field]))
    assert np.any(budget[field] > 0.0)
