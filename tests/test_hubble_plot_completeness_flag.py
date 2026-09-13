"""The full suite includes completeness; minimal mode can explicitly enable it."""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from qvc.hubble import hubble_fit


def tree():
    return ast.parse(Path(hubble_fit.__file__).read_text())


def test_cli_accepts_explicit_plot_completeness_request():
    parser = argparse.ArgumentParser()
    declaration = next(n for n in ast.walk(tree()) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
                       and n.args and isinstance(n.args[0], ast.Constant)
                       and n.args[0].value == '--plot-completeness')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(declaration)], type_ignores=[])), '<cli>', 'exec'), {'parser': parser})
    assert parser.parse_args([]).plot_completeness is False
    assert parser.parse_args(['--plot-completeness']).plot_completeness is True


@pytest.mark.parametrize('minimal,explicit,expected', [
    (False, False, True), (False, True, True),
    (True, False, False), (True, True, True),
])
@pytest.mark.parametrize('skip_residual', [False, True])
def test_effective_completeness_plot_mode(minimal, explicit, expected, skip_residual):
    args = SimpleNamespace(
        minimal_plots=minimal, plot_completeness=explicit, skip_plots=False,
        compare_sigma_only=False, only_sna=False, use_jax=False,
        disable_completeness=False, skip_debiased_residual_plot=skip_residual,
    )
    hubble_fit.validate_plot_mode_args(args)
    assert args.plot_completeness is expected
    assert args.skip_debiased_residual_plot is skip_residual


@pytest.mark.parametrize('disabled_mode', [
    'skip_plots', 'compare_sigma_only', 'only_sna', 'disable_completeness',
])
def test_automatic_completeness_respects_disabled_modes(disabled_mode):
    args = SimpleNamespace(
        minimal_plots=False, plot_completeness=False, skip_plots=False,
        compare_sigma_only=False, only_sna=False, use_jax=False,
        disable_completeness=False,
    )
    setattr(args, disabled_mode, True)
    hubble_fit.validate_plot_mode_args(args)
    assert args.plot_completeness is False


@pytest.mark.parametrize('minimal', [False, True])
def test_plot_completeness_compatible_with_minimal(minimal):
    hubble_fit.validate_plot_mode_args(SimpleNamespace(
        plot_completeness=True, minimal_plots=minimal, skip_plots=False,
        compare_sigma_only=False, only_sna=False, use_jax=False))


@pytest.mark.parametrize('conflict', ['skip_plots', 'compare_sigma_only'])
def test_conflicting_plot_contracts(conflict):
    args = dict(plot_completeness=True, minimal_plots=False, skip_plots=False,
                compare_sigma_only=False, only_sna=False, use_jax=False)
    args[conflict] = True
    with pytest.raises(ValueError, match='--plot-completeness'):
        hubble_fit.validate_plot_mode_args(SimpleNamespace(**args))


def test_all_orchestration_layers_forward_flag():
    names = {'run_single', 'run_all', 'run_mcmc_pipeline', '_run_hubble_fit_stage', 'run_single_jax'}
    calls = [n for n in ast.walk(tree()) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id in names]
    assert calls
    for call in calls:
        assert 'plot_completeness' in {k.arg for k in call.keywords}, call.func.id


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('completeness', [False, True])
def test_map_suite_guard_executes_only_when_requested(enabled, completeness):
    fn = next(n for n in tree().body if isinstance(n, ast.FunctionDef) and n.name == 'run_single')
    block = next(n for n in fn.body if isinstance(n, ast.If)
                 and ast.unparse(n.test) == 'plot_completeness and completeness')
    map_call, slices = Mock(return_value=(None,) * 6), Mock()
    from contextlib import nullcontext
    scope = dict(plot_completeness=enabled, completeness=completeness,
                 df_agn_completeness_parent=object(), df_agn_all=object(), completeness_mode='2d',
                 completeness_sim_file='unused', plot_path='unused', completeness_z_range=(0, 4.5),
                 _build_completeness_params=map_call,
                 plot_completeness_vs_mag_at_redshifts=slices,
                 trace_completeness_step=lambda *a: nullcontext())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), '<map suite>', 'exec'), scope)
    assert map_call.call_count == int(enabled and completeness)
    assert slices.call_count == int(enabled and completeness)
    if enabled and completeness:
        assert map_call.call_args.args[0] is scope['df_agn_completeness_parent']
        assert map_call.call_args.kwargs['plot'] is True
    # This suite must run before the minimal branch returns.
    assert fn.body.index(block) < next(i for i, n in enumerate(fn.body)
        if isinstance(n, ast.If) and ast.unparse(n.test) == 'minimal_plots')


def test_skip_debiased_residual_plot_cli_aliases_and_default():
    parser = argparse.ArgumentParser()
    declaration = next(n for n in ast.walk(tree()) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
                       and n.args and isinstance(n.args[0], ast.Constant)
                       and n.args[0].value == '--skip-debiased-residual-plot')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(declaration)], type_ignores=[])), '<cli>', 'exec'), {'parser': parser})
    assert parser.parse_args([]).skip_debiased_residual_plot is False
    for spelling in ('--skip-debiased-residual-plot', '--skip_debiased_residual_plot'):
        assert parser.parse_args([spelling]).skip_debiased_residual_plot is True


@pytest.mark.parametrize('skip', [False, True])
def test_skip_debiased_residual_plot_leaves_next_diagnostic_running(skip, capsys):
    fn = next(n for n in tree().body if isinstance(n, ast.FunctionDef) and n.name == 'run_single')
    block = next(n for n in fn.body if isinstance(n, ast.If)
                 and ast.unparse(n.test) == 'skip_debiased_residual_plot')
    following = fn.body[fn.body.index(block) + 1]
    assert following.value.func.id == 'plot_debias_impact_diagnostics'
    atlas, impact = Mock(), Mock()
    scope = dict(skip_debiased_residual_plot=skip,
                 plot_full_residuals_debiased_partial_controls=atlas,
                 plot_debias_impact_diagnostics=impact,
                 df_agn_pass2_plot_sample=object(), debiased_residuals=object(),
                 biased_residuals=object(), plot_path='unused', z_range=(.44, 3.16))
    exec(compile(ast.Module(body=[block, following], type_ignores=[]), '<residual plot guard>', 'exec'), scope)
    assert atlas.call_count == int(not skip)
    impact.assert_called_once()
    message = capsys.readouterr().out
    assert ('Plotting debiased residuals...' in message) == (not skip)
    assert ('Skipping debiased residual' in message) == skip


@pytest.mark.parametrize('skip', [False, True])
def test_skip_debiased_residual_plot_guards_redshift_wiggle_diagnostics(skip, capsys):
    fn = next(n for n in tree().body if isinstance(n, ast.FunctionDef) and n.name == 'run_single')
    block = next(n for n in fn.body if isinstance(n, ast.If)
                 and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                         and c.func.id == 'plot_redshift_wiggle_diagnostics'
                         for c in ast.walk(n)))
    diagnostic = Mock()
    scope = dict(skip_debiased_residual_plot=skip,
                 plot_redshift_wiggle_diagnostics=diagnostic,
                 df_agn_pass2_plot_sample=object(), biased_residuals=object(),
                 biased_residuals_err=object(), debiased_residuals=object(),
                 debiased_clipping_sigma=object(), plot_path='unused',
                 z_range=(.44, 3.16))
    exec(compile(ast.Module(body=[block], type_ignores=[]), '<wiggle guard>', 'exec'), scope)
    assert diagnostic.call_count == int(not skip)
    assert ('Skipping redshift-wiggle diagnostics' in capsys.readouterr().out) == skip


def test_skip_debiased_residual_plot_forwarded_from_cli_and_full_dispatch():
    for node in ast.walk(tree()):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'run_single', 'run_all'}:
            kwargs = {k.arg: k.value for k in node.keywords}
            assert 'skip_debiased_residual_plot' in kwargs
            assert ast.unparse(kwargs['skip_debiased_residual_plot']) in {
                'args.skip_debiased_residual_plot', 'skip_debiased_residual_plot'}
