import os
import subprocess
import sys

import pytest
from qvc.hubble.completeness_grid import launcher_completeness_options, grid_bin_count, MAG_WIDTH_ENV, Z_WIDTH_ENV


def test_defaults_and_cli_precedence():
    values = launcher_completeness_options([], {})
    assert values[MAG_WIDTH_ENV] == values[Z_WIDTH_ENV] == '0.1'
    values = launcher_completeness_options(['--completeness-mag-bin-width', '0.2', '--completeness-z-bin-width', '0.3', '--completeness-smooth-sigma-mag', '0', '--completeness-smooth-sigma-z', '0.5'], {MAG_WIDTH_ENV: '0.5'})
    assert values[MAG_WIDTH_ENV] == '0.2'
    assert values[Z_WIDTH_ENV] == '0.3'
    assert values['QVC_HUBBLE_COMPLETENESS_SMOOTH_SIGMA_MAG'] == '0.0'
    assert values['QVC_HUBBLE_COMPLETENESS_SMOOTH_SIGMA_Z'] == '0.5'
    assert grid_bin_count(8, .2) == 40
    assert grid_bin_count(4.5, .3) == 15
    assert grid_bin_count(4.5, .2) == 23


@pytest.mark.parametrize('option,value', [('mag-bin-width','0'), ('z-bin-width','nan'), ('z-bin-width','-1'), ('smooth-sigma-mag','inf'), ('smooth-sigma-z','-0.1'), ('mag-bin-width','2')])
def test_invalid_options(option, value):
    with pytest.raises(SystemExit):
        launcher_completeness_options(['--completeness-'+option, value], {})


def test_subprocess_grid_defaults_and_overrides():
    code = '''from qvc.hubble import cuts
from qvc.hubble.hubble_completeness_refactored import get_completeness_function_2d, get_completeness_function_3d_fhost, get_completeness_function_4d_fhost_alpha
from qvc.hubble.hubble_fit import completeness_map_variant_tag
import inspect
assert cuts.COMPLETENESS_N_MAG_BINS == 40
assert cuts.COMPLETENESS_N_Z_BINS == 15
for f in [get_completeness_function_2d, get_completeness_function_3d_fhost, get_completeness_function_4d_fhost_alpha]:
    assert inspect.signature(f).parameters['n_z_bins'].default == 15
    assert inspect.signature(f).parameters['n_mag_bins'].default == 40
assert '_compgrid40x15' in completeness_map_variant_tag()
'''
    env = dict(os.environ, **{MAG_WIDTH_ENV:'0.2', Z_WIDTH_ENV:'0.3'})
    subprocess.run([sys.executable, '-c', code], env=env, check=True, capture_output=True, text=True)


def test_direct_scientific_cli_grid_override_precedes_imports():
    code = '''from qvc.hubble.completeness_grid import configure_grid_from_argv
import os
configure_grid_from_argv(['--completeness-mag-bin-width', '0.2', '--completeness-z-bin-width', '0.2', 'catalog.h5'], os.environ)
from qvc.hubble import cuts
from qvc.hubble.hubble_completeness_refactored import get_completeness_function_2d
import inspect
assert cuts.COMPLETENESS_N_MAG_BINS == 40
assert cuts.COMPLETENESS_N_Z_BINS == 23
assert inspect.signature(get_completeness_function_2d).parameters['n_z_bins'].default == 23
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)


def test_direct_cli_registers_grid_arguments():
    import argparse
    from qvc.hubble.completeness_grid import add_grid_arguments
    parser = argparse.ArgumentParser()
    add_grid_arguments(parser)
    args = parser.parse_args(['--completeness-mag-bin-width', '0.2', '--completeness-z-bin-width', '0.2'])
    assert args.completeness_mag_bin_width == args.completeness_z_bin_width == .2
