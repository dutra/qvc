"""Check AGN preset arguments without launching scientific inference."""

import pytest

from test_run_hubble_paper_launcher import XONSH, launch, options

pytestmark = pytest.mark.skipif(XONSH is None, reason="xonsh is required")


def test_agn_preset_and_default_have_separate_outputs(tmp_path):
    result, calls = launch(tmp_path / "agn", "run_hubble.xonsh", {
        "QVC_HUBBLE_MODE": "agn-only-no-bright-cut",
    })
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    agn = options(calls[0])
    assert "--only_agn" in agn
    assert "--fit_redshift_log_f_term" not in agn
    assert "--bright-subsample-completeness-min" not in agn
    assert agn["--completeness-magnitude-support-mode"] == ["tails"]
    assert agn["--prior-profile"] == ["centered_lcdm"]
    for flag in ("--disable_completeness", "--use_planck_h0_prior",
                 "--disable_ceph_dist_calibration"):
        assert flag not in agn
    result, calls = launch(tmp_path / "joint", "run_hubble.xonsh")
    assert result.returncode == 0, result.stderr
    joint = options(calls[0])
    assert "--only_agn" not in joint
    assert joint["--bright-subsample-completeness-min"] == ["0.1"]
    assert joint["--completeness-magnitude-support-mode"] == ["hard-cut"]
    assert agn["--prefix"] != joint["--prefix"]
    assert agn["--result_prefix"] != joint["--result_prefix"]


def test_invalid_fit_mode_stops_before_inference(tmp_path):
    result, calls = launch(tmp_path, "run_hubble.xonsh", {
        "QVC_HUBBLE_MODE": "typo",
    })
    assert result.returncode != 0
    assert not calls


@pytest.mark.parametrize("flag", ["--fit_redshift_log_f_term", "--fit-redshift-log-f-term"])
@pytest.mark.parametrize("mode", ["joint", "agn-only-no-bright-cut"])
def test_redshift_scatter_flag_reaches_fitter(tmp_path, flag, mode):
    result, calls = launch(tmp_path, "run_hubble.xonsh", {
        "QVC_HUBBLE_MODE": mode,
    }, args=[flag, "--completeness-mag-bin-width", "0.3"])
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    args = options(calls[0])
    assert calls[0]["args"].count("--fit_redshift_log_f_term") == 1
    assert args["--completeness-mag-bin-width"] == ["0.3"]
    assert args["--prefix"][0].endswith("_logfz")
    assert args["--result_prefix"][0].endswith("_logfz")
    assert ("--only_agn" in args) == (mode == "agn-only-no-bright-cut")


@pytest.mark.parametrize('flag', ['--fit_redshift_mean_term', '--fit-redshift-mean-term'])
@pytest.mark.parametrize('scatter', [False, True])
def test_mean_redshift_flag_independent_of_scatter(tmp_path, flag, scatter):
    flags = [flag] + (['--fit_redshift_log_f_term'] if scatter else [])
    result, calls = launch(tmp_path, 'run_hubble.xonsh', {
        'QVC_HUBBLE_MODE': 'agn-only-no-bright-cut',
    }, args=flags)
    assert result.returncode == 0, result.stderr
    args = options(calls[0])
    assert '--fit_redshift_mean_term' in args
    assert ('--fit_redshift_log_f_term' in args) == scatter
    assert args['--prefix'][0].endswith('_meanz')
    assert args['--result_prefix'][0].endswith('_meanz')


def test_wide_matter_prior_reaches_fitter(tmp_path):
    result, calls = launch(tmp_path, 'run_hubble.xonsh', {
        'QVC_HUBBLE_MODE': 'agn-only-no-bright-cut',
        'QVC_HUBBLE_PRIOR_PROFILE': 'centered_lcdm_om01',
    }, args=['--fit_redshift_mean_term'])
    assert result.returncode == 0, result.stderr
    args = options(calls[0])
    assert args['--prior-profile'] == ['centered_lcdm_om01']
    assert '--fit_redshift_mean_term' in args
