"""Fixed or bounded latent-lf magnitude scatter sigma_M = exp(log_f)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qvc.hubble import hubble_eiv_lf as E
from qvc.hubble import hubble_fit
from qvc.hubble.dynesty_settings import prior_transform_dynesty
from qvc.hubble.hubble_likelihood import log_likelihood
from qvc.hubble.hubble_model import get_model_params

from test_hubble_eiv_lf import TRUTH, _mock, _pivot_context, _theta
from test_hubble_fit_smoke import _agn_pivot_context, _make_fake_agn_sample, _make_fake_pantheon_sample

REPO = Path(__file__).resolve().parents[1]
LOW = E.EIV_LOG_F_PRIOR[0]


def test_log_f_prior_settings():
    assert E.eiv_log_f_prior() == E.EIV_LOG_F_PRIOR
    assert E.eiv_log_f_prior(sigma_m_max=0.1) == (LOW, float(np.log(0.1)))
    assert E.eiv_log_f_prior(sigma_m_fixed=0.05) == (float(np.log(0.05)),) * 2
    # Consistent pair: the fixed value wins.
    assert E.eiv_log_f_prior(sigma_m_fixed=0.05, sigma_m_max=0.1) == (float(np.log(0.05)),) * 2
    with pytest.raises(ValueError, match="Inconsistent sigma_M settings"):
        E.eiv_log_f_prior(sigma_m_fixed=0.2, sigma_m_max=0.1)
    for bad in (0.01, 0.005, 2.0, np.nan):
        with pytest.raises(ValueError, match="upper bound"):
            E.eiv_log_f_prior(sigma_m_max=bad)
    for bad in (0.0, -0.1, np.inf):
        with pytest.raises(ValueError, match="fixed sigma_M"):
            E.eiv_log_f_prior(sigma_m_fixed=bad)


def test_model_params_default_unchanged_and_settings_applied():
    kwargs = dict(only_agn=True, prior_profile="centered_lcdm_om01", eiv_mode="latent-lf")
    default = get_model_params("FlatLambdaCDM", **kwargs)
    explicit_none = get_model_params("FlatLambdaCDM", eiv_sigma_m_fixed=None, eiv_sigma_m_max=None, **kwargs)
    assert default == explicit_none
    assert default[0]["log_f"] == E.EIV_LOG_F_PRIOR
    bounded = get_model_params("FlatLambdaCDM", eiv_sigma_m_max=0.1, **kwargs)
    # Same parameter vector; only the log_f box changes (the bound the diagnostic
    # MAP fits imposed from outside as HI[log_f] = ln 0.1).
    assert bounded[1] == default[1]
    assert {k: v for k, v in bounded[0].items() if k != "log_f"} == {
        k: v for k, v in default[0].items() if k != "log_f"
    }
    assert bounded[0]["log_f"] == (LOW, float(np.log(0.1)))
    with pytest.raises(ValueError, match="require eiv_mode='latent-lf'"):
        get_model_params("FlatLambdaCDM", only_agn=True, eiv_sigma_m_max=0.1)


def test_prior_transform_and_likelihood_bounds():
    data = _mock(n=200)
    common = dict(
        agn_data=data, pantheon_data=None, _sna_L=None, _sna_Lower=None, _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM", z_pivot_agn=None, agn_pivot_context=_pivot_context(), only_agn=True,
        eiv_mode="latent-lf", prior_profile="centered_lcdm_om01", completeness_params=None,
    )
    priors, labels, _ = get_model_params(
        "FlatLambdaCDM", only_agn=True, prior_profile="centered_lcdm_om01", eiv_mode="latent-lf"
    )
    i = labels.index("log_f")

    def ll(sigma_m, **settings):
        theta = _theta(dict(TRUTH, log_f=np.log(sigma_m)), {"H0": 70.0, "Om0": 0.3}, labels)
        return log_likelihood(theta, **common, **settings)[0]

    reference = ll(0.05)
    assert np.isfinite(reference)
    # Upper bound: values below it are unchanged, values above it are outside the prior.
    assert ll(0.05, eiv_sigma_m_max=0.1) == reference
    assert ll(0.2, eiv_sigma_m_max=0.1) == -np.inf
    # Fixed: the prior transform is constant and only that value is accepted.
    fixed, _, _ = get_model_params(
        "FlatLambdaCDM", only_agn=True, prior_profile="centered_lcdm_om01", eiv_mode="latent-lf",
        eiv_sigma_m_fixed=0.05,
    )
    rng = np.random.default_rng(0)
    draws = [prior_transform_dynesty(rng.random(len(labels)), fixed, labels)[i] for _ in range(5)]
    assert draws == [float(np.log(0.05))] * 5
    assert ll(0.05, eiv_sigma_m_fixed=0.05) == reference
    assert ll(0.06, eiv_sigma_m_fixed=0.05) == -np.inf
    # The default box still allows both.
    assert np.isfinite(ll(0.2)) and priors["log_f"] == E.EIV_LOG_F_PRIOR


class _Stop(Exception):
    pass


@pytest.mark.parametrize(
    ("settings", "log_f_bounds"),
    [
        ({}, E.EIV_LOG_F_PRIOR),
        ({"eiv_sigma_m_max": 0.1}, (LOW, float(np.log(0.1)))),
        ({"eiv_sigma_m_fixed": 0.05}, (float(np.log(0.05)),) * 2),
    ],
)
def test_run_mcmc_pipeline_passes_the_settings_to_the_sampler(monkeypatch, tmp_path, settings, log_f_bounds):
    """The sampler (and so a capture for the MAP/profile scripts) gets the log_f box."""
    df_fit = _make_fake_agn_sample(n_agn=6)
    captured = {}

    class FakeSampler:
        def __init__(self, loglike, ptform, ndim, logl_kwargs=None, ptform_kwargs=None, **kwargs):
            captured.update(logl_kwargs=logl_kwargs, ptform_kwargs=ptform_kwargs, ndim=ndim)

        def run_nested(self, *args, **kwargs):
            raise _Stop

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hubble_fit, "get_qvc_result_dir", lambda: Path.cwd() / "results")
    monkeypatch.setattr(hubble_fit, "DynamicNestedSampler", FakeSampler)
    monkeypatch.setattr(
        hubble_fit,
        "build_eiv_lf_table",
        lambda z, **kwargs: {E.EIV_LNPSI_COL: np.zeros((2, 3), np.float32), E.EIV_Z0_COL: 0.0,
                             E.EIV_DZ_COL: 5.0, E.EIV_U0_COL: 10.0, E.EIV_DU_COL: 10.0},
    )
    monkeypatch.setattr(
        hubble_fit,
        "get_completeness_function_2d",
        lambda *args, **kwargs: (
            lambda m, z: np.ones(np.broadcast_shapes(np.shape(m), np.shape(z))),
            np.array([19.0, 20.0]), np.array([0.5, 1.0]), 0.5, 0.1, 0.0,
        ),
    )
    with pytest.raises(_Stop):
        hubble_fit.run_mcmc_pipeline(
            df_agn=df_fit,
            df_agn_all=df_fit.copy(),
            df_pantheon=_make_fake_pantheon_sample(),
            _sna_L=None,
            _sna_Lower=True,
            _sna_LogdetCov=None,
            agn_pivot_context=_agn_pivot_context(df_fit, (0.44, 3.16)),
            cosmo_model="FlatLambdaCDM",
            only_agn=True,
            completeness=True,
            use_full_cov=False,
            speed="fastest",
            prefix="unit",
            prior_profile="centered_lcdm_om01",
            completeness_sim_file="dummy_completeness.h5",
            df_agn_completeness=df_fit,
            eiv_mode="latent-lf",
            **settings,
        )
    assert captured["ptform_kwargs"]["priors"]["log_f"] == log_f_bounds
    logl = captured["logl_kwargs"]
    for key in ("eiv_sigma_m_fixed", "eiv_sigma_m_max"):
        if key in settings:
            assert logl[key] == settings[key]
        else:
            # Off: the likelihood keyword arguments are those of the base commit.
            assert key not in logl


def _cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "qvc.hubble.hubble_fit", "unused.h5", "--magnitude-convention", "dereddened",
         "--spectra_fit_h5", "unused_spectra.h5", *args],
        cwd=REPO, capture_output=True, text=True, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin",
                                                         "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1"},
    )


def test_cli_validates_the_settings_before_loading_data():
    help_text = _cli("--help").stdout
    assert "--eiv-sigma-m-fixed" in help_text and "--eiv-sigma-m-max" in help_text
    result = _cli("--eiv-sigma-m-max", "0.1")
    assert result.returncode != 0 and "require --eiv-mode latent-lf" in result.stderr
    result = _cli("--eiv-mode", "latent-lf", "--run", "single", "--eiv-sigma-m-max", "0.1",
                  "--eiv-sigma-m-fixed", "0.2")
    assert result.returncode != 0 and "Inconsistent sigma_M settings" in result.stderr
