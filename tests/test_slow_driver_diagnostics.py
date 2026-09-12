"""Slow-driver convergence survives sample serialization without changing payloads."""
import h5py
import numpy as np
import pytest
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from qvc.light_curve import multiband_fit_utils as utils
from qvc.light_curve.slow_driver_diagnostics import slow_pole_convergence, slow_pole_draws

@pytest.mark.parametrize('chains', [1, 2])
def test_chain_diagnostics_match_numpyro(chains):
    draws = np.random.default_rng(42).normal(2., .2, (chains, 100))
    samples = {'tau_slow_driver': 10**draws * 2}
    np.testing.assert_allclose(slow_pole_draws(samples, 'shared_latent_blr', 1.), draws)
    result = slow_pole_convergence(samples, 'shared_latent_blr', 1.)
    assert result['log_tau_slow_rf_ess'] == pytest.approx(float(effective_sample_size(draws)))
    assert result['log_tau_slow_rf_rhat'] == pytest.approx(float(split_gelman_rubin(draws)))

@pytest.mark.parametrize('samples', [None, {'tau_slow_driver': np.ones((2, 10))},
                                    {'tau_slow_driver': np.ones((2, 3))},
                                    {'tau_slow_driver': np.arange(1., 11.)}])
def test_unavailable_chain_diagnostics_are_nan(samples):
    assert all(np.isnan(v) for v in slow_pole_convergence(samples, 'shared_latent_blr', 1.).values())

def test_sample_round_trip_and_legacy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(utils, 'prefix', 'diagnostics')
    monkeypatch.setattr(utils, 'suffix', 'test')
    samples = {'tau_slow_driver': np.arange(1., 11.), 'tau_fast_driver': np.ones(10)}
    diagnostics = {'log_tau_slow_rf_ess': 42., 'log_tau_slow_rf_rhat': 1.03}
    utils.save_obj_samples_to_hdf5(samples, 'object', scalar_diagnostics={**diagnostics, 'loo_rms': 1.2})
    loaded, attrs = utils.load_obj_samples_from_hdf5('object', return_metadata=True)
    default = utils.load_obj_samples_from_hdf5('object')
    assert isinstance(default, dict)
    assert set(default) == set(samples) | {'loo_rms'}
    for k in samples:
        np.testing.assert_array_equal(loaded[k], samples[k])
    assert slow_pole_convergence(None, 'shared_latent_blr', 1., saved_diagnostics=attrs) == diagnostics
    with h5py.File('legacy.h5', 'w') as h:
        h['tau_slow_driver'] = samples['tau_slow_driver']
    _, attrs = utils.load_obj_samples_from_hdf5(file_path='legacy.h5', return_metadata=True)
    assert all(np.isnan(v) for v in slow_pole_convergence(None, 'shared_latent_blr', 1., saved_diagnostics=attrs).values())

def test_invalid_inputs():
    with pytest.raises(ValueError, match='scalar'):
        slow_pole_convergence(None, 'shared_latent_blr', 1., saved_diagnostics={'log_tau_slow_rf_ess': [42.]})
    with pytest.raises(ValueError, match='require a refit'):
        slow_pole_draws({'tau_slow_driver': [1.], 'eta_tau': [.1]}, 'shared_latent_blr', 1.)
    with pytest.raises(ValueError, match='finite and positive'):
        slow_pole_draws({'tau_slow_driver': [0.]}, 'shared_latent_blr', 1.)
