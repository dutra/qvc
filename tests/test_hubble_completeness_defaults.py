import argparse
import inspect
from pathlib import Path

import pandas as pd

from qvc.hubble import hubble_fit, hubble_fit_jax


ROOT = Path(__file__).resolve().parents[1]


def test_cpu_jax_and_cli_share_fresh_completeness_defaults():
    assert hubble_fit.DEFAULT_COMPLETENESS is True
    assert hubble_fit.DEFAULT_COMPLETENESS_SIM_FILE is None

    for function in (
        hubble_fit.run_mcmc_pipeline,
        hubble_fit.run_single,
        hubble_fit.run_all,
        hubble_fit_jax.run_single_jax,
    ):
        parameters = inspect.signature(function).parameters
        assert parameters["completeness"].default is hubble_fit.DEFAULT_COMPLETENESS
        assert (
            parameters["completeness_sim_file"].default
            is hubble_fit.DEFAULT_COMPLETENESS_SIM_FILE
        )
        assert (
            parameters["completeness_mode"].default
            == "2d_relative_support"
        )

    parser = argparse.ArgumentParser()
    hubble_fit.add_completeness_cli_arguments(parser)
    args = parser.parse_args([])
    assert args.disable_completeness is False
    assert args.completeness_sim_file is None
    assert args.completeness_mode == "2d_relative_support"
    assert hubble_fit.make_run_tag(
        "FlatLambdaCDM",
        False,
        "fastest",
        None,
        (0.44, 3.16),
    ).endswith("_2d_relative_support")


def test_active_runner_does_not_override_fresh_completeness_defaults():
    runner = (ROOT / "run_hubble.xonsh").read_text()
    assert "--disable_completeness" not in runner
    assert "--completeness_sim_file" not in runner
    normalized_runner = " ".join(runner.split())
    assert "--completeness_mode 2d_relative_support" in normalized_runner


def test_default_completeness_resolves_to_a_fresh_mock(monkeypatch, tmp_path):
    generated = tmp_path / "fresh.h5"
    calls = []
    monkeypatch.setattr(
        hubble_fit,
        "estimate_sky_box_area_deg2",
        lambda dataframe: 7.5,
    )
    monkeypatch.setattr(
        hubble_fit,
        "generate_fresh_completeness_sim_file",
        lambda plot_path, *, area_deg2, seed: (
            calls.append((plot_path, area_deg2, seed)),
            str(generated),
        )[1],
    )

    resolved = hubble_fit.resolve_completeness_sim_file(
        completeness=hubble_fit.DEFAULT_COMPLETENESS,
        completeness_sim_file=hubble_fit.DEFAULT_COMPLETENESS_SIM_FILE,
        plot_path=tmp_path,
        df_agn_all=pd.DataFrame(),
        seed=19,
    )

    assert resolved == str(generated)
    assert calls == [(tmp_path, 7.5, 19)]


def test_explicit_completeness_mock_bypasses_fresh_generation(monkeypatch, tmp_path):
    explicit = tmp_path / "precomputed.h5"
    monkeypatch.setattr(
        hubble_fit,
        "generate_fresh_completeness_sim_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fresh generation must be bypassed")
        ),
    )

    resolved = hubble_fit.resolve_completeness_sim_file(
        completeness=True,
        completeness_sim_file=str(explicit),
        plot_path=tmp_path,
        df_agn_all=pd.DataFrame(),
    )

    assert resolved == str(explicit)
