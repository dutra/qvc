import os
from pathlib import Path
import shutil
import subprocess
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "hpc_scripts" / "shubble_grid.xsh"
XONSH = shutil.which("xonsh")


def load_launcher():
    module = ModuleType("shubble_grid_test")
    module.__file__ = str(SCRIPT)
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), module.__dict__)
    return module


def option_map(argv):
    parsed = {}
    current = None
    for value in argv[2:-1]:
        if value.startswith("--"):
            current = value
            parsed[current] = []
        else:
            parsed[current].append(value)
    return parsed


def test_defaults_define_dense_48_cell_quick_grid():
    launcher = load_launcher()
    args = launcher.parse_args(["--description", "paper grid"])
    first, last = launcher.validate_args(args)

    assert args.description == "paper_grid"
    assert args.n_values == list(range(1000, 8001, 1000))
    assert args.zmax_values == [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    assert args.speed == "quick"
    assert args.ncores == 12
    assert args.mem == "32G"
    assert launcher.grid_cells(args.n_values, args.zmax_values)[:7] == [
        (1000, 1.0), (1000, 1.5), (1000, 2.0),
        (1000, 2.5), (1000, 3.0), (1000, 3.5), (2000, 1.0),
    ]
    assert (first, last) == (0, 47)
    assert launcher.build_campaign_name(
        args, "sep15_1312", "abc1234"
    ) == "sep15_1312_hubble_grid_quick_paper_grid_abc1234"


@pytest.mark.skipif(XONSH is None, reason="xonsh is required")
def test_launcher_help_runs_before_compute_environment_is_activated():
    result = subprocess.run(
        [XONSH, "--no-rc", str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--description" in result.stdout


def test_partial_array_selection_and_validation():
    launcher = load_launcher()
    args = launcher.parse_args([
        "--description", "partial", "--skip", "2", "--num-jobs", "4",
        "--n-values", "1000", "2000", "--zmax-values", "1", "2", "3",
    ])
    assert launcher.validate_args(args) == (2, 5)

    outside = launcher.parse_args([
        "--description", "bad", "--skip", "6",
        "--n-values", "1000", "2000", "--zmax-values", "1", "2", "3",
    ])
    with pytest.raises(ValueError, match="outside the grid"):
        launcher.validate_args(outside)


@pytest.mark.parametrize(
    "name",
    [
        "QVC_HUBBLE_COMPLETENESS_MAGNITUDE_SUPPORT_MODE",
        "QVC_HUBBLE_CUT_TIER",
        "QVC_HUBBLE_LIGHT_CURVE_UNCERTAINTY_MODE",
    ],
)
def test_invalid_paper_setting_fails_before_submission(name):
    launcher = load_launcher()
    args = launcher.parse_args(["--description", "bad_setting"])

    with pytest.raises(ValueError, match=name):
        launcher.resolve_paper_settings(args, {name: "invalid"})


def test_submission_snapshots_paper_and_cut_environment_overrides(tmp_path, monkeypatch):
    launcher = load_launcher()
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)
    args = launcher.parse_args(["--description", "overrides"])
    settings, cuts = launcher.resolve_paper_settings(
        args,
        {
            "QVC_HUBBLE_COMPLETENESS_MAGNITUDE_SUPPORT_MODE": "tails",
            "QVC_HUBBLE_CUT_TIER": "1",
            "QVC_HUBBLE_LIGHT_CURVE_UNCERTAINTY_MODE": "posterior-draws",
            "QVC_CUT_LOG_TAU_UV_RF_MIN": "1.5",
            "QVC_CUT_NUM_DIVERGENCES_MAX": "none",
        },
    )
    script = launcher.build_sbatch_script(args, "campaign", settings, cuts, "encoded")

    assert "export QVC_HUBBLE_COMPLETENESS_MAGNITUDE_SUPPORT_MODE=tails" in script
    assert "export QVC_HUBBLE_CUT_TIER=1" in script
    assert "export QVC_HUBBLE_LIGHT_CURVE_UNCERTAINTY_MODE=posterior-draws" in script
    assert "export QVC_CUT_LOG_TAU_UV_RF_MIN=1.5" in script
    assert "export QVC_CUT_NUM_DIVERGENCES_MAX=none" in script
    assert "--completeness-magnitude-support-mode \\\n    tails" in script
    assert "--cut-tier \\\n    1" in script


def test_generated_sbatch_has_resources_grid_mapping_and_paper_profile(tmp_path, monkeypatch):
    launcher = load_launcher()
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)
    args = launcher.parse_args(["--description", "paper_grid"])
    launcher.validate_args(args)
    settings, cuts = launcher.resolve_paper_settings(args, {})
    grid_argv = launcher.hubble_arguments(
        n_value=1000,
        zmax=3.16,
        prefix="campaign/N1000_zmax3.16",
        speed=args.speed,
        settings=settings,
    )
    script = launcher.build_sbatch_script(args, "campaign", settings, cuts, "encoded")

    syntax = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr
    assert "#SBATCH --cpus-per-task=12" in script
    assert "#SBATCH --mem=32G" in script
    assert "#SBATCH --partition=day" in script
    assert "#SBATCH --time=24:00:00" in script
    assert "N_VALUES=(1000 2000 3000 4000 5000 6000 7000 8000)" in script
    assert "ZMAX_VALUES=(1.0 1.5 2.0 2.5 3.0 3.5)" in script
    assert 'CURRENT_PREFIX="${CAMPAIGN}/N${N}_zmax${ZMAX}"' in script
    assert grid_argv[grid_argv.index("--run") + 1] == "single"
    assert "--speed \\\n    quick" in script
    for flag in ("--minimal-plots", "--uniform_redshift_distribution", "--skip-debiased-residual-plot"):
        assert flag in script
    for flag in ("--skip_plots", "--compare_sigma_only", "--plot-completeness"):
        assert flag not in script
    assert "wang2026_type1_lade_a" in script
    assert "QVC_CUT_LOG_TAU_UV_RF_MIN=1.3" in script
    assert "QVC_CUT_NUM_DIVERGENCES_MAX=0" in script


def test_default_cuts_match_sep19b_standard_profile():
    launcher = load_launcher()
    args = launcher.parse_args(["--description", "sep19b_cuts"])
    _, cuts = launcher.resolve_paper_settings(args, {})

    assert cuts == {
        "QVC_CUT_COMPLETENESS_MAG_2500_MIN": "17",
        "QVC_CUT_COMPLETENESS_MAG_2500_MAX": "24",
        "QVC_CUT_JAXSEDFIT_JOINT_REDUCED_CHI2_MAX": "1.2",
        "QVC_CUT_SED_REDUCED_CHI2_MAX": "1.3",
        "QVC_CUT_SPECTROSCOPY_REDUCED_CHI2_MAX": "1.1",
        "QVC_CUT_LOO_CHI2_EFF_MAX": "1.05",
        "QVC_CUT_SPECTRAL_RHAT_MAX": "1.05",
        "QVC_CUT_LIGHT_CURVE_RHAT_MAX": "1.05",
        "QVC_CUT_NUM_DIVERGENCES_MAX": "0",
        "QVC_CUT_LOG_TAU_UV_RF_MIN": "1.3",
        "QVC_CUT_T_RF_LENGTH_MIN": "none",
        "QVC_CUT_LIGHT_CURVE_N_POINTS_MIN": "none",
        "QVC_CUT_SN_MEDIAN_ALL_MIN": "3",
        "QVC_CUT_VARIABILITY_CHI_SQ_RED_G_MIN": "20",
        "QVC_CUT_ETA_SIGMA_KL_MIN": "none",
        "QVC_CUT_LOG_TAU_UV_RF_MAX": "none",
        "QVC_CUT_T_RF_OVER_TAU_UV_RF_MIN": "none",
        "QVC_CUT_APPARENT_MAG_2500_ERR_MAX": "none",
        "QVC_CUT_F_BC_3000_MAX": "none",
        "QVC_CUT_F_FE_UV_3000_MAX": "none",
        "QVC_CUT_F_HOST_2500_PSF_MAX": "none",
        "QVC_CUT_FRAC_AGN_5100_MIN": "none",
        "QVC_CUT_A_2500_TOTAL_MAX": "none",
        "QVC_CUT_EBV_GAL_PLUS_EBV_AGN_MAX": "none",
        "QVC_CUT_LOW_L2500_FHOST_LOG_L_MAX": "none",
        "QVC_CUT_LOW_L2500_FHOST_PSF_MAX": "none",
    }


@pytest.mark.skipif(XONSH is None, reason="xonsh is required")
def test_hubble_arguments_match_current_paper_fiducial_profile(monkeypatch, tmp_path):
    launcher = load_launcher()
    args = launcher.parse_args(["--description", "compare"])
    settings, _ = launcher.resolve_paper_settings(args, {})
    grid_argv = launcher.hubble_arguments(
        n_value=1000,
        zmax=3.16,
        prefix="paper_test",
        speed="quick",
        settings=settings,
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "calls.txt"
    stub = "#!/usr/bin/env python3\nimport os,sys\nfrom pathlib import Path\np=Path(os.environ['CALLS'])\nwith p.open('a') as f: f.write('\\0'.join(sys.argv[1:])+'\\n')\n"
    for name in ("python", "bash"):
        path = fake_bin / name
        path.write_text(stub)
        path.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith("QVC_")}
    env.update(PATH=f"{fake_bin}{os.pathsep}{env['PATH']}", CALLS=str(calls), QVC_HUBBLE_SPEED="quick", QVC_HUBBLE_PREFIX="paper_test")
    result = subprocess.run(
        [XONSH, "--no-rc", str(ROOT / "run_hubble_paper.xonsh")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    paper_argv = calls.read_text().splitlines()[0].split("\0")

    paper = option_map(paper_argv)
    grid = option_map(grid_argv)
    intended_differences = {
        "--run", "--plot-completeness", "--N",
        "--uniform_redshift_distribution", "--minimal-plots",
    }
    assert paper["--run"] == ["full"]
    assert grid["--run"] == ["single"]
    assert {k: v for k, v in paper.items() if k not in intended_differences} == {
        k: v for k, v in grid.items() if k not in intended_differences
    }


def test_main_dry_run_writes_without_submitting(tmp_path, monkeypatch):
    launcher = load_launcher()
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "make_run_stamp", lambda: "sep15_1312")
    monkeypatch.setattr(launcher, "get_git_short_hash", lambda: "abc1234")
    submitted = []
    monkeypatch.setattr(launcher, "submit_sbatch", lambda *values: submitted.append(values))

    path = launcher.main(["--description", "paper_grid", "--dry-run"])

    assert path.is_file()
    campaign = "sep15_1312_hubble_grid_quick_paper_grid_abc1234"
    assert path.name == f"submit_{campaign}.sbatch"
    script = path.read_text()
    assert f"#SBATCH --job-name={campaign}" in script
    assert f"/logs/hubble/{campaign}/grid_%A_%a.out" in script
    assert f"export CAMPAIGN={campaign}" in script
    assert 'CURRENT_PREFIX="${CAMPAIGN}/N${N}_zmax${ZMAX}"' in script
    assert submitted == []


def test_main_submits_selected_absolute_array_range(tmp_path, monkeypatch):
    launcher = load_launcher()
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "make_run_stamp", lambda: "sep15_1313")
    monkeypatch.setattr(launcher, "get_git_short_hash", lambda: "abc1234")
    submitted = []
    monkeypatch.setattr(launcher, "submit_sbatch", lambda *values: submitted.append(values))

    path = launcher.main([
        "--description", "subset", "--skip", "7", "--num-jobs", "3"
    ])

    assert submitted == [(path, 7, 9)]
