import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_hubble_alpha_nu_completeness_models.xonsh"


def test_alpha_nu_runner_orders_models_by_cost_and_pins_wang_v3_semantics():
    source = RUNNER.read_text(encoding="utf-8")
    ordered = (
        '"baseline_3d_fhost"',
        '"latent_off"',
        '"latent_fixed0"',
        '"latent_fixed_beta_negative"',
        '"latent_fixed_beta_positive"',
        '"latent_joint_beta"',
        '"latent_off_maginteraction"',
        '"latent_joint_beta_maginteraction"',
    )
    default_block = source.split("default_experiment_names = (", 1)[1].split(
        ")", 1
    )[0]
    positions = [default_block.index(name) for name in ordered]
    assert positions == sorted(positions)
    assert "m2500norm12_v3.h5" in source
    assert '$QVC_HUBBLE_COMPLETENESS_LF_MODEL = "wang2026_type1_lade_a"' in source
    assert '$QVC_HUBBLE_COMPLETENESS_MAGNITUDE = "attenuated"' in source
    assert '$QVC_HUBBLE_MAGNITUDE_CONVENTION = "dereddened"' in source
    assert '$QVC_HUBBLE_MINIMAL_PLOTS = "false"' in source
    assert '$XONSH_SUBPROC_CMD_RAISE_ERROR = True' in source


def test_alpha_nu_runner_dry_run_supports_an_explicit_ordered_subset():
    env = os.environ.copy()
    env.pop("QVC_HUBBLE_RESUME", None)
    env["QVC_HUBBLE_ALPHA_COMPLETENESS_DRY_RUN"] = "true"
    env["QVC_HUBBLE_ALPHA_COMPLETENESS_EXPERIMENTS"] = (
        "baseline_3d_fhost,latent_joint_beta_maginteraction"
    )
    result = subprocess.run(
        ["xonsh", str(RUNNER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "jobs (2): baseline_3d_fhost, latent_joint_beta_maginteraction" in result.stdout
    first = result.stdout.index("[1/2] baseline_3d_fhost")
    last = result.stdout.index("[2/2] latent_joint_beta_maginteraction")
    assert first < last
    assert "Running hubble_fit.py" not in result.stdout
