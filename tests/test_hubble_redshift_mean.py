"""Independent numerical checks of mean redshift evolution and its plumbing."""
import numpy as np
import pandas as pd
import pytest
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import minimize_scalar
from test_hubble_fagn_flux_fraction import _flux_inputs
from qvc.hubble import hubble_model as model, hubble_likelihood as likelihood
from qvc.hubble import hubble_fit as fit, hubble_utils as utils, hubble_plotting as plotting

FLAG = 'use_redshift_mean_term'
PARAM = 'gamma_mean_z'


def inputs(gamma=2.0):
    data, params, *_ = _flux_inputs()
    params.update(gamma_mean_z=gamma, gamma_log_f_z=0.0, log_f=np.log(.5), H0=70., Om0=.3)
    data.update(z_err=np.zeros(5), apparent_mag_2500_err=np.full(5, .02))
    context = model.build_agn_pivot_context(data, (.5, 2.), **{FLAG: True})
    obs, err, piv = model.agn_model_pack_obs(data, pivot_context=context, **{FLAG: True})
    packed = model.agn_model_pack_params(params, **{FLAG: True})
    return data, params, context, packed, obs, err, piv


def test_mean_formula_and_vectorized_predictions():
    data, params, context, packed, obs, err, piv = inputs()
    baseline_context = model.build_agn_pivot_context(data, (.5, 2.))
    bobs, _, bpiv = model.agn_model_pack_obs(data, pivot_context=baseline_context)
    baseline = model.M_model_agn(model.agn_model_pack_params(params), bobs, bpiv)
    expected = baseline + 2 * np.log10((1 + data['z']) / (1 + np.median(data['z'])))
    scalar = model.M_model_agn(packed, obs, piv, **{FLAG: True})
    vector = model.M_model_agn_posterior_samples(np.vstack([packed, packed]), obs, piv, **{FLAG: True})
    np.testing.assert_allclose(scalar, expected)
    np.testing.assert_allclose(vector, np.vstack([expected, expected]))
    assert scalar[2] == baseline[2]
    # No spurious independent error for z: it is propagated jointly with distance.
    np.testing.assert_allclose(model.M_model_agn_err(packed, obs, err, piv, **{FLAG: True}), 0)


def evaluate(data, params, context, enabled=True, scatter=False):
    _, labels, _ = model.get_model_params('FlatLambdaCDM', only_agn=True,
        use_redshift_mean_term=enabled, use_redshift_log_f_term=scatter)
    return likelihood.log_likelihood(np.array([params[k] for k in labels]),
        agn_data=data, pantheon_data={}, _sna_L=None, _sna_Lower=None,
        _sna_LogdetCov=None, cosmo_model='FlatLambdaCDM', completeness_params=None,
        z_pivot_agn=1.5, agn_pivot_context=context, only_agn=True,
        use_redshift_mean_term=enabled, use_redshift_log_f_term=scatter)[0]


def test_zero_gamma_baseline_and_nonzero_likelihood_recovery():
    data, params, context, packed, obs, _, piv = inputs(0.)
    base = model.build_agn_pivot_context(data, (.5, 2.))
    data['apparent_mag_2500'] = model.M_model_agn(packed, obs, piv, **{FLAG: True}) + FlatLambdaCDM(H0=70, Om0=.3).distmod(data['z']).value
    assert evaluate(data, params, context) == pytest.approx(evaluate(data, params, base, False), abs=1e-12)
    data['apparent_mag_2500'] += 2.3 * np.log10((1 + data['z'])/(1 + np.median(data['z'])))
    result = minimize_scalar(lambda gamma: -evaluate(data, dict(params, gamma_mean_z=gamma), context, scatter=True), bounds=(-20,20), method='bounded')
    assert result.success
    assert result.x == pytest.approx(2.3, abs=1e-6)


def test_shared_redshift_error_includes_cross_term():
    z=np.array([.5, 1., 2.]); dz=np.full(3, 1e-4); cosmo=FlatLambdaCDM(H0=70, Om0=.3)
    gamma=-3.
    expected=.5*np.abs(cosmo.distmod(z+dz).value-cosmo.distmod(z-dz).value + gamma*np.log10((1+z+dz)/(1+z-dz)))
    np.testing.assert_allclose(likelihood.sigma_mu_from_z_err(z,dz,cosmo,gamma), expected)


def test_priors_summary_and_checkpoint_guard():
    priors, labels, _ = model.get_model_params('FlatLambdaCDM', only_agn=True, use_redshift_mean_term=True, use_redshift_log_f_term=True)
    assert priors[PARAM] == (-20.,20.)
    assert priors['gamma_log_f_z'] == (-20.,20.)
    summary = utils.extract_cosmo_results_from_samples(np.zeros((4,len(labels))), 'FlatLambdaCDM', only_sna=False, only_agn=True,
        use_alpha_lambda_term=False, use_eta_sigma_term=False, use_redshift_mean_term=True, use_redshift_log_f_term=True)
    assert PARAM in summary['params']
    assert '_meanz' in fit.make_run_tag('FlatLambdaCDM',False,'quick',None,(.44,3.16),use_redshift_mean_term=True)
    payload=dict(prior_profile='centered_lcdm', prior_bounds_json=fit.canonical_prior_bounds_json(model.get_model_params('FlatLambdaCDM')[0]),
        early_de_guard=False, flat_samples=np.zeros((3,4)), dmi_max_w=np.zeros(2), dmi_posterior_sigma=np.ones(2), integrals_max_w=np.zeros(2), logZ=0., logZerr=.1, use_redshift_mean_term=True)
    with pytest.raises(RuntimeError,match='mean-redshift'):
        fit.validate_resume_checkpoint(payload,'test.h5',ndim=4,n_agn=2)
    fit.validate_resume_checkpoint(payload,'test.h5',ndim=4,n_agn=2,expected_use_redshift_mean_term=True)


def test_prediction_plot_path_with_both_evolution_terms(tmp_path):
    data, params, context, packed, obs, _, piv = inputs()
    data['apparent_mag_2500'] = model.M_model_agn(packed,obs,piv,**{FLAG:True}) + FlatLambdaCDM(H0=70,Om0=.3).distmod(data['z']).value
    frame=pd.DataFrame(data);frame['is_fit_selection']=True
    _, labels, _ = model.get_model_params('FlatLambdaCDM', only_agn=True, use_redshift_mean_term=True, use_redshift_log_f_term=True)
    samples=np.array([[params[k] for k in labels]]*12)
    plotting.plot_hubble(samples, frame, pd.DataFrame(), 'FlatLambdaCDM', 1.5,
        plot_path=str(tmp_path), show=False, compute_only=True, completeness=False,
        z_range=(.5,2.), agn_pivot_context=context, only_agn=True,
        use_alpha_lambda_term=False, use_eta_sigma_term=False, use_redshift_mean_term=True, use_redshift_log_f_term=True)


def test_completeness_and_posterior_draws_keep_mean_evolution():
    from qvc.hubble.hubble_completeness_refactored import (
        Completeness2D, COMPLETENESS_MAG_COL, COMPLETENESS_MAG_ERR_COL,
    )
    data, params, context, packed, obs, _, piv = inputs(1.4)
    data['apparent_mag_2500'] = model.M_model_agn(packed,obs,piv,**{FLAG:True}) + FlatLambdaCDM(H0=70,Om0=.3).distmod(data['z']).value
    data[COMPLETENESS_MAG_COL]=data['apparent_mag_2500'].copy()
    data[COMPLETENESS_MAG_ERR_COL]=data['apparent_mag_2500_err'].copy()
    for col, source in [(likelihood.LIGHT_CURVE_LOG_SIGMA_DRAW_COL,'log_sigma_uv'),
                        (likelihood.LIGHT_CURVE_LOG_TAU_RF_DRAW_COL,'log_tau_uv_rf')]:
        data[col]=np.repeat(data[source][:,None],4,axis=1)
    data[likelihood.LIGHT_CURVE_POSTERIOR_VALID_COUNT_COL]=np.full(5,4)
    grid=np.linspace(14,32,121); zg=np.linspace(0,4,21)
    selection=Completeness2D(grid,zg,np.repeat(np.linspace(.99,.1,len(grid))[:,None],len(zg),axis=1),magnitude_support=(14,32))
    _,labels,_=model.get_model_params('FlatLambdaCDM',only_agn=True,use_redshift_mean_term=True,use_redshift_log_f_term=True)
    common=dict(agn_data=data,pantheon_data={},_sna_L=None,_sna_Lower=None,_sna_LogdetCov=None,
        cosmo_model='FlatLambdaCDM',completeness_params=(selection,grid),z_pivot_agn=1.5,
        agn_pivot_context=context,only_agn=True,use_redshift_mean_term=True,use_redshift_log_f_term=True)
    theta=np.array([params[k] for k in labels])
    covariance=likelihood.log_likelihood(theta,**common)
    empirical=likelihood.log_likelihood(theta,light_curve_uncertainty_mode='posterior-draws',**common)
    assert np.isfinite(covariance[0])
    np.testing.assert_allclose(covariance[0],empirical[0],atol=1e-10)
    np.testing.assert_allclose(covariance[1],empirical[1],atol=1e-10)
    theta[labels.index(PARAM)]=0
    assert likelihood.log_likelihood(theta,**common)[0] != pytest.approx(covariance[0])


def test_pivot_is_frozen_for_subset_and_roundtrips_checkpoint(tmp_path):
    data, params, context, packed, obs, _, piv = inputs()
    subset={k:np.asarray(v)[:2] for k,v in data.items()}
    sobs,_,spiv=model.agn_model_pack_obs(subset,pivot_context=context,**{FLAG:True})
    np.testing.assert_array_equal(spiv,piv)
    np.testing.assert_allclose(model.M_model_agn(packed,sobs,spiv,**{FLAG:True}),model.M_model_agn(packed,obs,piv,**{FLAG:True})[:2])
    payload=fit._agn_pivot_checkpoint_payload(context)
    checkpoint = tmp_path / 'mean_evolution.h5'
    fit.save_chains(str(checkpoint), **payload, use_redshift_mean_term=True)
    loaded = fit.load_chains(str(checkpoint))
    assert bool(loaded['use_redshift_mean_term'])
    restored=fit._load_agn_pivot_context_from_checkpoint(loaded,checkpoint_file=str(checkpoint),use_redshift_mean_term=True)
    assert restored == context


@pytest.mark.parametrize('cosmology', ['FlatLambdaCDM', 'FlatwCDM', 'Flatw0waCDM'])
def test_om01_profile_changes_only_matter_density(cosmology):
    base, labels, _ = model.get_model_params(cosmology, only_agn=True, use_redshift_mean_term=True)
    wide, wide_labels, _ = model.get_model_params(cosmology, only_agn=True, use_redshift_mean_term=True, prior_profile='centered_lcdm_om01')
    assert wide_labels == labels
    assert wide['Om0'] == (0., 1.)
    assert base['Om0'] == (.2, .8)
    assert {k:v for k,v in wide.items() if k != 'Om0'} == {k:v for k,v in base.items() if k != 'Om0'}
