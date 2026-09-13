"""Exercise source selection and preflight without touching real paper assets."""
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/copy_paper_assets.sh'


def setup_sources(tmp_path, legacy=False):
    (tmp_path / 'src').mkdir()
    base = tmp_path / 'plots' / 'run'
    run = base / ('Flatw0waCDM_joint_quicker_all_z0p44_3p16_2d_compmag-attenuated' if legacy else 'Flatw0waCDM_joint')
    compare = base / ('model_compare_joint_quicker_all_z0p44_3p16_2d' if legacy else 'model_compare')
    files = [run / name for name in (
        'predicted_L2500_vs_fullcorr_band_debiased.pdf', 'redshift_histograms.pdf',
        'hubble_diagram_debiased.pdf', 'hubble_diagram.pdf', 'agn_table.csv',
        'agn_table.tex', 'predicted_vs_actual_M2500_debias.pdf',
        'alphaOX_residuals.pdf', 'delta_alphaOX_residuals.pdf',
        'completeness/completeness_map_with_relative_percent_contours.pdf')]
    files += [base / 'diagnostics' / name for name in (
        'tier1_cuts_vs_redshift_precut.pdf', 'blr_postcut.pdf',
        'sigma_tau_vs_lambda_broken_pl_fit_postcut.pdf',
        'bpl_psd_vs_uv_variability_precut.pdf')]
    files += [compare / f'cosmo_corner_{model}_{kind}.pdf'
              for model in ('FlatLambdaCDM', 'FlatwCDM', 'Flatw0waCDM')
              for kind in ('alphabeta',)]
    files += [compare / 'param_results_fiducial.tex']
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(path.relative_to(base)))
    return base, run, compare


def execute(tmp_path, *args):
    return subprocess.run(['bash', str(SCRIPT), '--speed', 'quicker',
                           '--fiducial-dir', 'plots/run', '--only', 'hubble', *args],
                          cwd=tmp_path, text=True, capture_output=True)


@pytest.mark.parametrize('legacy', [False, True])
def test_exact_corner_sources_and_added_draft_assets(tmp_path, legacy):
    _, run, compare = setup_sources(tmp_path, legacy)
    result = execute(tmp_path, '--dry-run')
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / 'plots/run/paper').exists()
    result = execute(tmp_path)
    assert result.returncode == 0, result.stderr
    dest = tmp_path / 'plots/run/paper/hubble'
    for model in ('FlatLambdaCDM', 'FlatwCDM', 'Flatw0waCDM'):
        name = f'cosmo_corner_{model}_alphabeta.pdf'
        assert (dest / name).read_bytes() == (compare / name).read_bytes()
    assert not list(dest.glob('*_noalphabeta.pdf'))
    assert (dest / 'agn_table.tex').read_bytes() == (run / 'agn_table.tex').read_bytes()
    assert (dest / 'bpl_psd_vs_uv_variability_precut.pdf').exists()
    assert (dest / 'completeness_map_with_relative_percent_contours.pdf').exists()
    assert (dest / 'tier1_cuts_vs_redshift_precut.pdf').exists()
    assert not (dest / 'completeness_map.pdf').exists()
    assert not (dest / 'spectral_fraction_vs_redshift_cuts.pdf').exists()


def test_missing_sources_leave_existing_assets_untouched(tmp_path):
    _, run, _ = setup_sources(tmp_path)
    (run / 'completeness/completeness_map_with_relative_percent_contours.pdf').unlink()
    dest = tmp_path / 'plots/run/paper/hubble'
    dest.mkdir(parents=True)
    (dest / 'hubble_diagram.pdf').write_text('previous paper figure')
    result = execute(tmp_path)
    assert result.returncode != 0
    assert 'completeness_map_with_relative_percent_contours.pdf' in result.stderr
    assert (dest / 'hubble_diagram.pdf').read_text() == 'previous paper figure'
    assert len(list(dest.iterdir())) == 1


def test_ambiguous_run_directories_are_rejected(tmp_path):
    base, _, _ = setup_sources(tmp_path)
    (base / 'Flatw0waCDM_joint_quicker_all_z0p44_3p16_2d_other').mkdir()
    result = execute(tmp_path)
    assert result.returncode != 0
    assert 'expected exactly one' in result.stderr
    assert not (tmp_path / 'plots/run/paper').exists()
