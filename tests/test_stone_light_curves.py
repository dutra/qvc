import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from qvc.light_curve import multiband_generate_lc as loader
from qvc.light_curve import fit_light_curves as fit


@pytest.fixture
def stone_file(tmp_path, monkeypatch):
    cols = []
    for key in ['DBID', 'RA', 'DEC', 'Z', 'LOG_M_BH', 'LOG_M_BH_ERR', 'LOG_LBOL', 'LOG_LBOL_ERR']:
        values = [1, 2, 3] if key == 'DBID' else [0.5, 0.7, 0.9]
        cols.append(fits.Column(name=key, format='D', array=values))
    for band in 'gri':
        for key, values in [('MJD', [1, 2, 3, 4, 5, 6, 7]), ('MAG', [20, 21, 22, 23, np.nan, 25, 26]), ('MAG_ERR', [.1, .2, .3, .4, .1, 0, np.inf])]:
            cols.append(fits.Column(name=f'{key}_{band}', format='7D', array=[values]*3))
        cols.append(fits.Column(name=f'SURVEY_{band}', format='35A', dim='(5,7)', array=[['SDSS', 'PS1', 'DES', 'DECam', '', '', '']]*3))
        for key in ['log_SIGMA', 'log_TAU_REST']:
            for suffix in ['', '_ERR_L', '_ERR_U']:
                cols.append(fits.Column(name=f'{key}_{band}{suffix}', format='D', array=[.1]*3))
    path = tmp_path / 'stone.fits'
    fits.BinTableHDU.from_columns(cols).writeto(path)
    cat = tmp_path / 'catalog.parquet'
    pd.DataFrame({'objectId': ['101', '102'], 'RA': [.5, .7], 'DEC': [.5, .7]}).to_parquet(cat)
    monkeypatch.setattr(loader, 'resolve_qvc_data_path', lambda p: str(path if 'Stone2021' in p else cat))
    return path


def test_stone_loader_and_preparation(stone_file):
    with pytest.warns(UserWarning, match='Skipping 1 Stone'):
        objects = loader.load_stone_lcs()
    assert len(objects) == 2
    obj = objects[0]
    assert obj['number_points'] == 12
    assert obj['cadence'] == 1
    assert obj['cadence_err'] == 0
    assert obj['survey_names'] == ('sdss', 'ps1', 'des', 'decam')
    for band in 'gri':
        np.testing.assert_array_equal(obj['times'][band], [1, 2, 3, 4])
        np.testing.assert_array_equal(obj['surveys'][band], obj['survey_names'])
    lc = fit.make_lc(obj, list('gri'), verbose=False)
    assert lc['z'] == obj['stone_Z']
    assert lc['seeing_active_mask'].shape == (3, 4)
    assert not lc['seeing_active_mask'].any()
    assert np.isnan(lc['psf_fwhm_arcsec']).all()
    for i, band in enumerate('gri'):
        m = lc['band_idx'] == i
        np.testing.assert_allclose(np.asarray(lc['y'])[m] + lc['mags_means'][i], obj['mags'][band])
        np.testing.assert_array_equal(np.asarray(lc['X'][0])[m] + lc['time0'], obj['times'][band])
        np.testing.assert_array_equal(np.asarray(lc['yerr'])[m], obj['magerrs'][band])
        np.testing.assert_array_equal(lc['survey_idx'][m], [0, 1, 2, 3])
    jitter, offsets = fit._get_object_active_noise_calibration_masks(lc, 3)
    assert jitter.shape == offsets.shape == (3, 4)
    assert jitter.all()
    np.testing.assert_array_equal(offsets[0], [False, True, True, True])


def test_stone_filter_before_slice(stone_file):
    with pytest.warns(UserWarning):
        objects = loader.load_stone_lcs(filter_object_ids=[102], N=1)
    assert [o['object_id'] for o in objects] == ['102']
    with pytest.warns(UserWarning):
        objects = loader.load_stone_lcs(filter_object_ids=[101, 102], skip=1, N=1)
    assert [o['object_id'] for o in objects] == ['102']
    with pytest.warns(UserWarning):
        assert loader.load_stone_lcs(N=0) == []


@pytest.mark.parametrize('variant', ['erlang', 'shared_latent', 'band_poles'])
def test_stone_four_surveys_reach_model_and_output(stone_file, variant):
    from numpyro.handlers import seed, trace
    from qvc.light_curve.multiband_fit_utils import flatten_flat_samples_per_band
    with pytest.warns(UserWarning):
        obj = loader.load_stone_lcs(N=1)[0]
    obj.update(fit.make_lc(obj, list('gri'), verbose=False))
    lam = np.array([loader.lambda_pivot[b] / (1 + obj['z']) for b in 'gri'])
    if variant == 'band_poles':
        from qvc.light_curve.band_poles_fit import build_single_object_model_band_poles
        model = build_single_object_model_band_poles(obj, lam, None)
    else:
        model = fit.build_single_object_model_mag_flux_linearized(obj, lam, None, shared_latent=variant == 'shared_latent')
    sites = trace(seed(model, 4)).get_trace()
    observed = [site for site in sites.values() if site['type'] == 'sample' and site.get('is_observed')]
    assert observed
    assert all(np.isfinite(np.asarray(site['fn'].log_prob(site['value']))).all() for site in observed)
    samples = {k: np.asarray(sites[k]['value'])[None, ...] for k in ['log_jitter', 'survey_delta_mag']}
    assert samples['log_jitter'].shape == (1, 3, 4)
    assert np.isfinite(samples['log_jitter']).all()
    flattened = flatten_flat_samples_per_band(samples, list('gri'), survey_names=obj['survey_names'])
    assert 'log_jitter_g_decam' in flattened
    assert 'survey_delta_mag_i_des' in flattened
    assert not any('ztf' in k for k in flattened)
