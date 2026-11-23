"""Checks for the isolated published-data benchmark (optional CAMB dependency)."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import cho_solve

from qvc.hubble.pantheon_reproduction import BOUNDS, Likelihood, load_data
from qvc.hubble.dynesty_settings import prior_transform_dynesty


def test_dynesty_transform_preserves_benchmark_bounds():
    from qvc.hubble.pantheon_reproduction import LABELS
    priors = dict(zip(LABELS, BOUNDS))
    for coordinate in (0., .5, 1.):
        transformed = prior_transform_dynesty(np.full(5, coordinate), priors, LABELS)
        np.testing.assert_allclose(transformed, BOUNDS[:, 0]+coordinate*np.diff(BOUNDS, axis=1).ravel())


def test_likelihood_checkpoint_roundtrip(released):
    import pickle
    _, likelihood = released
    restored = pickle.loads(pickle.dumps(likelihood))
    theta = [.3, .7, -.9, .2, -19.3]
    assert restored(theta) == likelihood(theta)


def test_redshift_mask_subsets_both_covariance_axes(tmp_path):
    pd.DataFrame(dict(zHD=[.005, .01, .02, .4], IS_CALIBRATOR=[1, 1, 1, 0],
                      zHEL=[.005, .01, .02, .4], m_b_corr=[10., 11., 12., 13.])).to_csv(
        tmp_path/'Pantheon+SH0ES.dat', sep=' ', index=False)
    covariance = np.eye(4) + .1*np.ones((4, 4))
    with open(tmp_path/'Pantheon+SH0ES_STAT+SYS.cov', 'w') as stream:
        stream.write('4\n')
        np.savetxt(stream, covariance.ravel())
    data, selected = load_data(tmp_path)
    assert list(data.index) == [2, 3]  # Calibrator flag must not override z cut.
    np.testing.assert_array_equal(selected, covariance[2:, 2:])


@pytest.fixture
def released():
    pytest.importorskip('camb')
    reference = Path('results/pantheon_reproduction/reference')
    if not (reference/'likelihood.py').exists():
        pytest.skip('Download the public benchmark inputs first')
    return reference, Likelihood(reference)


def test_released_theory_method_and_precision_agree(released):
    reference, likelihood = released
    # Execute the actual released method, without importing/installing Cosmosis.
    tree = ast.parse((reference/'likelihood.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'extract_theory_points')
    namespace = dict(np=np, names=SimpleNamespace(supernova_params='supernova_params'))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(reference/'likelihood.py'), 'exec'), namespace)
    model = SimpleNamespace(zCMB=likelihood.z, zHEL=likelihood.zhel,
                            x_section='distances', y_section='distances', x_name='z', y_name='D_A', kind='linear')
    theta = [.3, .7, -.9, .2, -19.3]
    prediction = likelihood.theory(theta)
    # Unique nodes avoid repeated catalogue redshifts in scipy interpolation.
    z, indices = np.unique(likelihood.z, return_index=True)
    da = 10**((prediction-theta[-1]-25)/5)/((1+likelihood.z)*(1+likelihood.zhel))
    block = {('distances', 'z'): z, ('distances', 'D_A'): da[indices], ('supernova_params', 'M'): theta[-1]}
    np.testing.assert_allclose(namespace['extract_theory_points'](model, block), prediction, atol=1e-12)
    residual = likelihood.observed-prediction
    precision = cho_solve(likelihood.factor, np.eye(len(residual)))
    np.testing.assert_allclose(likelihood(theta), -.5*residual@precision@residual, atol=1e-8)
    assert len(residual) == 1590


def test_bounds_and_dark_energy_crossing(released):
    reference, likelihood = released
    theta = np.array([.3, .7, -.9, .2, -19.3])
    for i in range(5):
        outside = theta.copy()
        outside[i] = BOUNDS[i, 1]+.001
        assert likelihood(outside) == -np.inf
    crossing = [.3, .7, -.9, -.5, -19.3]
    assert likelihood(crossing) == -np.inf
    assert np.isfinite(Likelihood(reference, 'ppf')(crossing))
