import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_hubble_fitted_color_models.xonsh"


def test_fitted_color_runner_has_exact_order_and_apples_to_apples_inputs():
    source = RUNNER.read_text(encoding="utf-8")
    ordered = (
        '"baseline_2d"',
        '"fitted_color_2d"',
        '"baseline_3d_fhost"',
        '"fitted_color_3d_fhost"',
    )
    experiment_block = source.split("experiments = (", 1)[1].split(
        "# Apples-to-apples", 1
    )[0]
    positions = [experiment_block.index(name) for name in ordered]
    assert positions == sorted(positions)
    assert experiment_block.count('"baseline_') == 2
    assert experiment_block.count('"fitted_color_') == 2
    assert "m2500norm12_v3.h5" in source
    assert "m2500norm12_v3_psfcolor.h5" not in source
    assert "run_backfill_spectra_psf_photometry.py --workers N" in source
    assert '$QVC_HUBBLE_COMPLETENESS_LF_MODEL = "wang2026_type1_lade_a"' in source
    assert '$QVC_HUBBLE_COMPLETENESS_MAGNITUDE = "attenuated"' in source
    assert '$QVC_HUBBLE_MAGNITUDE_CONVENTION = "dereddened"' in source
    assert '$QVC_HUBBLE_COMPLETENESS_CLOSURE_TEST = "false"' in source
    assert '$QVC_HUBBLE_MINIMAL_PLOTS = "false"' in source
    assert '$QVC_HUBBLE_COMPLETENESS_COLOR_MODEL = color_model' in source
    assert '$QVC_HUBBLE_COMPLETENESS_COLOR_PARENT_FILE = (' in source
    assert '$XONSH_SUBPROC_CMD_RAISE_ERROR = True' in source


def test_run_hubble_forwards_exact_color_environment_interface():
    source = (ROOT / "run_hubble.xonsh").read_text(encoding="utf-8")
    assert "QVC_HUBBLE_COMPLETENESS_COLOR_MODEL" in source
    assert "QVC_HUBBLE_COMPLETENESS_COLOR_PARENT_FILE" in source
    assert "QVC_HUBBLE_COMPLETENESS_COLOR_PARENT_SIGMA" in source
    assert "--completeness-color-model" in source
    assert "--completeness-color-parent-file" in source
    assert "--completeness-color-parent-sigma" in source


def test_fitted_color_runner_dry_run_lists_four_jobs_without_catalog():
    env = os.environ.copy()
    env.pop("QVC_HUBBLE_RESUME", None)
    env["QVC_HUBBLE_FITTED_COLOR_DRY_RUN"] = "true"
    result = subprocess.run(
        ["xonsh", str(RUNNER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for index, name in enumerate(
        (
            "baseline_2d",
            "fitted_color_2d",
            "baseline_3d_fhost",
            "fitted_color_3d_fhost",
        ),
        start=1,
    ):
        assert f"[{index}/4] {name}" in result.stdout
    assert "Running hubble_fit.py" not in result.stdout
