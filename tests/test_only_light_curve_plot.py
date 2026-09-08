"""Exercise the production plot dispatch without running scientific inference."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from qvc.light_curve import fit_light_curves as fit


@pytest.mark.parametrize('flag', ['--only-light-curve-plot', '--only_light_curve_plot'])
def test_cli_alias_implies_plot(monkeypatch, flag):
    import sys
    monkeypatch.setattr(sys, 'argv', ['fit_light_curves', flag, '--disable_plot_psd'])
    with pytest.raises(SystemExit) as error:
        fit.main()
    assert error.value.code == 2


def test_policy_defaults_and_conflicts():
    args = SimpleNamespace(plot=False)
    assert fit.apply_only_light_curve_plot_policy(args) is args
    assert vars(args) == {'plot': False}
    for flag in ['disable_combined_plot', 'disable_plot_psd']:
        with pytest.raises(ValueError, match=flag):
            fit.apply_only_light_curve_plot_policy(SimpleNamespace(only_light_curve_plot=True, **{flag: True}))


@pytest.mark.parametrize('only', [False, True])
def test_actual_plot_dispatch(only):
    tree = ast.parse(Path(fit.__file__).read_text())
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == 'args.plot')
    args = SimpleNamespace(plot=True, only_light_curve_plot=only,
        disable_plot_psd=False, disable_combined_plot=False,
        plot_ls_broken_pl=True, show_combined_light_curve_component_overlay=True,
        corner_plot_mode='fast', model_variant='shared_latent_blr')
    for name in ['trace','color_magnitude','correlation','histogram','corner','sigma_tau_lambda','recovery']:
        setattr(args, f'disable_{name}_plot', False)
    fit.apply_only_light_curve_plot_policy(args)
    calls = {n.func.id for n in ast.walk(block) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id not in {'float', 'dict'}}
    mocks = {name: Mock(return_value={}) for name in calls}
    class Data(dict):
        def __missing__(self, key): return 0
    context = dict(mocks, args=args, result={}, psd_break_result={},
                   obj=Data(z=1.), obj_flat_samples_flatten_per_band={},
                   plot_samples={}, m=None, bands=['g'], prefix='test', suffix='test',
                   BAND_POLES_BLR_VARIANT='shared_latent_band_poles_blr', oid='test')
    # Execute the try body directly so exceptions cannot be swallowed by logging.
    code = ast.Module(body=block.body[0].body, type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), '<production plotting>', 'exec'), context)
    mocks['save_combined_plot'].assert_called_once()
    assert mocks['save_combined_plot'].call_args.kwargs['plot_psd'] is True
    assert mocks['save_combined_plot'].call_args.kwargs['plot_bpl_fit'] is True
    for name, mock in mocks.items():
        if name == 'save_combined_plot': continue
        if only: mock.assert_not_called()
        else: mock.assert_called_once()
    assert args.disable_sigma_tau_lambda_plot is only
    assert args.disable_recovery_plot is only


def test_synthetic_combined_figure_with_psd(tmp_path, monkeypatch):
    import numpy as np
    import matplotlib.pyplot as plt
    from qvc.light_curve import multiband_fit_plotting as plotting
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plotting, 'prefix', 'only_plot_smoke')
    class Model:
        def pred(self, params, X):
            return np.zeros(len(X[0])), np.full(len(X[0]), .01)
        def psd(self, params, omega, **kwargs):
            return 100 / (1 + (omega * 100)**2)
    f = np.logspace(-5, -2, 12)
    power = 100 / (1 + (2*np.pi*f*100)**2)
    monkeypatch.setattr(plotting, 'subtract_gp_mean_for_psd', lambda m,p,X,y,**kw: y)
    monkeypatch.setattr(plotting, 'combined_raw_band_lomb_scargle',
        lambda *a,**k: (f,power,power,.8*power,1.2*power,np.full(12,10),np.zeros(12)))
    monkeypatch.setattr(plotting, 'estimate_model_window_response', lambda *a,**k: (f,np.ones(12)))
    captured = []
    save = plt.savefig
    def capture(*args, **kwargs):
        captured.append([ax.get_ylabel() for ax in plt.gcf().axes])
        save(*args, **kwargs)
    monkeypatch.setattr(plt, 'savefig', capture)
    t = np.linspace(0,1000,40); band = np.tile([0,1],20)
    plotting.save_combined_plot({'log_tau_uv':np.log([90.,100.,110.])},Model(),
        (t,band),.1*np.sin(t/100),np.full(40,.03),band,np.array([20.,20.]),{},
        {'object_id':'synthetic','z':1.},bands=['g','r'],plot_psd=True)
    files = list(tmp_path.rglob('*.pdf'))
    assert len(files) == 1 and files[0].parent.name == 'light_curves_fits'
    assert files[0].read_bytes().startswith(b'%PDF')
    assert len(captured[0]) == 2 and 'PSD' in captured[0][1]
