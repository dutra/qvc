"""Completeness plotting is opt-in, independently of the minimal plot suite."""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from qvc.hubble import hubble_fit


def tree():
    return ast.parse(Path(hubble_fit.__file__).read_text())


def test_cli_accepts_plot_completeness_and_defaults_off():
    parser = argparse.ArgumentParser()
    declaration = next(n for n in ast.walk(tree()) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
                       and n.args and isinstance(n.args[0], ast.Constant)
                       and n.args[0].value == '--plot-completeness')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(declaration)], type_ignores=[])), '<cli>', 'exec'), {'parser': parser})
    assert parser.parse_args([]).plot_completeness is False
    assert parser.parse_args(['--plot-completeness']).plot_completeness is True


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
