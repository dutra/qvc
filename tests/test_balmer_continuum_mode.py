"""Tests of the --balmer-continuum-mode option of the joint spectral fit."""

import re
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qvc.spectra import fit_spectra_jaxsedfit_joint as joint

from test_fit_spectra_jaxsedfit_joint import _record_with_ugriz
from test_sfitspectra import _fresh_workspace, _run_fresh


COVERED_REST = np.array([2900.0, 3500.0, 4100.0])
UNCOVERED_REST = np.array([2000.0, 2900.0, 3500.0])


def _args(tmp_path=None, **overrides):
    values = dict(
        no_deredden=False,
        wave_min=1250.0,
        wave_max=8000.0,
        dsps_ssp_fn="unused-in-config-construction.h5",
        sed_n_wave=512,
        fit_lines=True,
        fit_fe=True,
        fit_bc=True,
        fit_bal=True,
        line_flux_scale_mjy=0.1,
        photometry_systematics=0.08,
        spectrum_student_t_df=5.0,
        spectrum_scale_prior_sigma_dex=0.1,
        fit_method="optax",
        optax_steps=10,
        optax_lr=5.0e-3,
        plot_init=False,
        seed=3,
        nuts_warmup=10,
        nuts_samples=10,
        nuts_chains=1,
        nuts_target_accept=0.9,
        dense_mass="blocks",
        output_dir="results",
        fig_dir="figures",
        save_fig=False,
        save_jaxsedfit_samples=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _build(args, rest_wave, z=1.0):
    record = {
        **_record_with_ugriz(),
        "sdss_name": "000000.00+000000.0",
        "z": z,
        "ra": 0.0,
        "dec": 0.0,
        "mjd": 55000.0,
    }
    lam = np.asarray(rest_wave) * (1.0 + z)
    config, _ = joint.build_joint_config(
        record,
        pd.DataFrame(),
        lam=lam,
        flux=np.ones_like(lam),
        err=np.full_like(lam, 0.1),
        resolving_power=2000.0,
        args=args,
        aperture_diameter_arcsec=2.0,
    )
    return config


@pytest.mark.parametrize(
    ("mode", "fit_bc", "rest", "expected"),
    [
        (None, True, COVERED_REST, "data"),
        (None, True, UNCOVERED_REST, "off"),
        ("coverage", True, UNCOVERED_REST, "off"),
        ("always", True, COVERED_REST, "data"),
        ("always", True, UNCOVERED_REST, "prior"),
        ("always", False, COVERED_REST, "off"),
        ("always", False, UNCOVERED_REST, "off"),
    ],
)
def test_balmer_continuum_constraint(mode, fit_bc, rest, expected):
    args = SimpleNamespace(fit_bc=fit_bc)
    if mode is not None:
        args.balmer_continuum_mode = mode
    assert joint.balmer_continuum_constraint(args, rest) == expected


def test_balmer_continuum_constraint_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown Balmer-continuum mode"):
        joint.balmer_continuum_constraint(
            SimpleNamespace(fit_bc=True, balmer_continuum_mode="sometimes"),
            UNCOVERED_REST,
        )


def test_default_config_is_unchanged_coverage_policy():
    pytest.importorskip("jaxsedfit")
    args = _args()  # no balmer_continuum_mode attribute: today's behaviour
    covered = _build(args, COVERED_REST)
    uncovered = _build(args, UNCOVERED_REST)
    explicit = _build(_args(balmer_continuum_mode="coverage"), UNCOVERED_REST)

    assert covered.agn.fit_balmer_continuum is True
    assert uncovered.agn.fit_balmer_continuum is False
    assert explicit.agn.fit_balmer_continuum is False
    for config in (covered, uncovered, explicit):
        assert getattr(config.agn, "balmer_ratio_prior", None) is None
        assert getattr(config.agn, "balmer_tau_prior", None) is None
        assert getattr(config.agn, "balmer_vel_prior", None) is None
    assert joint.config_balmer_continuum_constraint(covered) == "data"
    assert joint.config_balmer_continuum_constraint(uncovered) == "off"


def _require_jaxsedfit_balmer_priors():
    jaxsedfit = pytest.importorskip("jaxsedfit")
    if not hasattr(jaxsedfit.AGNConfig(), "balmer_ratio_prior"):
        pytest.skip("jaxsedfit lacks the opt-in Balmer-continuum ratio prior")


def test_always_mode_applies_prior_only_when_edge_is_uncovered():
    _require_jaxsedfit_balmer_priors()
    args = _args(balmer_continuum_mode="always")
    covered = _build(args, COVERED_REST)
    uncovered = _build(args, UNCOVERED_REST)

    assert covered.agn.fit_balmer_continuum is True
    assert covered.agn.balmer_ratio_prior is None
    assert covered.agn.balmer_tau_prior is None
    assert covered.agn.balmer_vel_prior is None
    assert joint.config_balmer_continuum_constraint(covered) == "data"

    assert uncovered.agn.fit_balmer_continuum is True
    assert uncovered.agn.balmer_ratio_prior == (
        joint.DEFAULT_BALMER_RATIO_PRIOR_MEDIAN,
        joint.DEFAULT_BALMER_RATIO_PRIOR_SIGMA_LN,
    )
    assert uncovered.agn.balmer_ratio_reference_rest == pytest.approx(3000.0)
    assert uncovered.agn.balmer_tau_prior == joint.BALMER_PRIOR_HELD_TAU
    assert uncovered.agn.balmer_vel_prior == joint.BALMER_PRIOR_HELD_VEL_KMS
    assert joint.config_balmer_continuum_constraint(uncovered) == "prior"

    custom = _build(
        _args(
            balmer_continuum_mode="always",
            balmer_ratio_prior_median=0.12,
            balmer_ratio_prior_sigma_ln=0.4,
        ),
        UNCOVERED_REST,
    )
    assert custom.agn.balmer_ratio_prior == (0.12, 0.4)

    disabled = _build(
        _args(balmer_continuum_mode="always", fit_bc=False), UNCOVERED_REST
    )
    assert disabled.agn.fit_balmer_continuum is False
    assert getattr(disabled.agn, "balmer_ratio_prior", None) is None


def test_apply_prior_refuses_jaxsedfit_without_prior_fields():
    with pytest.raises(RuntimeError, match="update jaxsedfit"):
        joint.apply_balmer_continuum_prior(SimpleNamespace(), SimpleNamespace())


def _resume_fitter(rest_wave, *, enabled, prior=None, redshift=1.0):
    agn = SimpleNamespace(fit_balmer_continuum=enabled)
    if prior is not None:
        for name, value in prior.items():
            setattr(agn, name, value)
    wave_obs = list(np.asarray(rest_wave) * (1.0 + redshift))
    return SimpleNamespace(
        config=SimpleNamespace(
            agn=agn,
            spectroscopy=SimpleNamespace(wave_obs=wave_obs, mask=[True] * len(wave_obs)),
        )
    )


def _validate(fitter, tmp_path, **args):
    return joint.validate_resume_balmer_continuum_fitter(
        fitter,
        tmp_path / "samples.h5",
        SimpleNamespace(fit_bc=True, **args),
        redshift=1.0,
    )


def test_resume_returns_bundle_constraint(tmp_path):
    prior = joint.balmer_continuum_prior_settings(SimpleNamespace())
    # Saved configurations come back from HDF5 with lists instead of tuples.
    saved_prior = {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in prior.items()
    }
    assert _validate(_resume_fitter(COVERED_REST, enabled=True), tmp_path) == "data"
    assert _validate(_resume_fitter(UNCOVERED_REST, enabled=False), tmp_path) == "off"
    assert (
        _validate(
            _resume_fitter(COVERED_REST, enabled=True),
            tmp_path,
            balmer_continuum_mode="always",
        )
        == "data"
    )
    assert (
        _validate(
            _resume_fitter(UNCOVERED_REST, enabled=True, prior=saved_prior),
            tmp_path,
            balmer_continuum_mode="always",
        )
        == "prior"
    )


def test_resume_refuses_bundle_without_bc_in_always_mode(tmp_path):
    with pytest.raises(
        joint.IncompatibleBalmerContinuumResumeError, match="cannot gain one post hoc"
    ):
        _validate(
            _resume_fitter(UNCOVERED_REST, enabled=False),
            tmp_path,
            balmer_continuum_mode="always",
        )


def test_resume_refuses_prior_held_bundle_in_coverage_mode(tmp_path):
    prior = joint.balmer_continuum_prior_settings(SimpleNamespace())
    with pytest.raises(
        joint.IncompatibleBalmerContinuumResumeError,
        match="current coverage policy requires False",
    ):
        _validate(_resume_fitter(UNCOVERED_REST, enabled=True, prior=prior), tmp_path)


def test_resume_refuses_uncovered_bc_without_prior(tmp_path):
    # An uncovered bundle with BC but without the ratio prior was not sampled
    # under the prior-held model.
    with pytest.raises(
        joint.IncompatibleBalmerContinuumResumeError,
        match="data-constrained BC, but the current always policy requires a prior",
    ):
        _validate(
            _resume_fitter(UNCOVERED_REST, enabled=True),
            tmp_path,
            balmer_continuum_mode="always",
        )


def test_resume_refuses_different_prior(tmp_path):
    prior = joint.balmer_continuum_prior_settings(SimpleNamespace())
    with pytest.raises(
        joint.IncompatibleBalmerContinuumResumeError, match="balmer_ratio_prior"
    ):
        _validate(
            _resume_fitter(UNCOVERED_REST, enabled=True, prior=prior),
            tmp_path,
            balmer_continuum_mode="always",
            balmer_ratio_prior_median=0.10,
        )


def test_base_result_records_mode_and_empty_constraint():
    rec = {"object_id": "1", "z": 1.7}
    default = joint._base_result(rec, SimpleNamespace(seed=0), execution_mode="fresh")
    always = joint._base_result(
        rec,
        SimpleNamespace(seed=0, balmer_continuum_mode="always"),
        execution_mode="fresh",
    )
    assert default["balmer_continuum_mode"] == "coverage"
    assert always["balmer_continuum_mode"] == "always"
    assert default["balmer_continuum_constraint"] == ""


def test_parse_args_balmer_continuum_mode(tmp_path):
    common = [
        "--mode", "fit",
        str(tmp_path / "out.h5"),
        "--sed-photometry-path", str(tmp_path / "phot.csv"),
        "--filter_object_id", "1",
    ]
    args = joint.parse_args(common)
    assert args.balmer_continuum_mode == "coverage"
    assert args.balmer_ratio_prior_median == pytest.approx(0.16)
    assert args.balmer_ratio_prior_sigma_ln == pytest.approx(0.7)
    args = joint.parse_args(
        [*common, "--balmer-continuum-mode", "always", "--balmer-ratio-prior-sigma-ln", "0.5"]
    )
    assert args.balmer_continuum_mode == "always"
    assert args.balmer_ratio_prior_sigma_ln == pytest.approx(0.5)
    with pytest.raises(SystemExit):
        joint.parse_args([*common, "--balmer-continuum-mode", "sometimes"])
    with pytest.raises(SystemExit):
        joint.parse_args([*common, "--balmer-ratio-prior-sigma-ln", "0"])


def _generated_command(root, script_path, generated, monkeypatch):
    block = generated.split("\"$PYTHON_BIN\" - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    for key, value in re.findall(r'^export (\w+)="([^"\n]*)"$', generated, re.M):
        monkeypatch.setenv(key, value)
    for key in ("CHUNK_SIZE", "NPROC", "CPUS_PER_TASK"):
        match = re.search(rf'^export {key}="?(\d+)"?$', generated, re.M)
        monkeypatch.setenv(key, match.group(1))
    monkeypatch.setenv("TASK_ID", "0")
    monkeypatch.chdir(root)
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: commands.append(cmd))
    exec(compile(block, str(script_path), "exec"), {})
    assert len(commands) == 1
    return commands[0]


@pytest.mark.parametrize("mode", [None, "coverage", "always"])
def test_sfitspectra_forwards_balmer_continuum_mode(tmp_path, monkeypatch, mode):
    root, script_path, csv_path, _, env = _fresh_workspace(tmp_path)
    extra = () if mode is None else ("--balmer-continuum-mode", mode)
    result = _run_fresh(
        root, script_path, env,
        "--chisq-csv", str(csv_path),
        "--description", "bcmode",
        *extra,
    )
    assert result.returncode == 0, result.stderr
    generated = next(
        (root / "hpc_scripts/submit/jaxqsofit").glob("submit_*.sbatch")
    ).read_text()
    expected = mode or "coverage"
    assert f'export BALMER_CONTINUUM_MODE="{expected}"' in generated
    command = _generated_command(root, script_path, generated, monkeypatch)
    assert command[command.index("--balmer-continuum-mode") + 1] == expected


def test_sfitspectra_rejects_always_mode_with_standalone_backend(tmp_path):
    root, script_path, csv_path, _, env = _fresh_workspace(tmp_path)
    result = _run_fresh(
        root, script_path, env,
        "--chisq-csv", str(csv_path),
        "--description", "bcmode",
        "--fit-script", "fit_spectra.py",
        "--balmer-continuum-mode", "always",
    )
    assert result.returncode != 0
    assert "--balmer-continuum-mode is supported only" in result.stderr
