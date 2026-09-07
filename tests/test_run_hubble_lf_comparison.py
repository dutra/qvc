import importlib.util
import os
import re
import shutil
import subprocess
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader

from qvc.hubble import lf_comparison_diagnostics
from qvc.hubble.completeness_mock_catalog import COMPLETENESS_LF_MODELS
from qvc.hubble.hubble_model import get_model_params


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_hubble_lf_comparison.py"
SPEC = importlib.util.spec_from_file_location("run_hubble_lf_comparison", SCRIPT)
comparison = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(comparison)


def _write_pdf(path: Path, text: str) -> None:
    figure = plt.figure(figsize=(4, 3))
    figure.text(0.5, 0.5, text, ha="center", va="center")
    figure.savefig(path, format="pdf")
    plt.close(figure)


def test_models_and_labels_follow_the_canonical_supported_order():
    assert tuple(COMPLETENESS_LF_MODELS) == (
        "shen",
        "wang2026_type1_lade_a",
        "palanque2016_ple_lede",
        "kulkarni2019_type1_model1",
        "kulkarni2019_type1_model2",
        "kulkarni2019_type1_model3",
    )
    assert comparison.LF_RUN_IDS == (
        "shen",
        "shen_type1_intrinsic",
        "shen_type1_attenuated",
        "wang2026_type1_lade_a",
        "palanque2016_ple_lede",
        "kulkarni2019_type1_model1",
        "kulkarni2019_type1_model2",
        "kulkarni2019_type1_model3",
    )
    assert tuple(comparison.LF_LABELS) == comparison.LF_RUN_IDS


def test_sweep_runs_every_model_with_only_the_intended_environment_changes(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(comparison, "REPO_ROOT", tmp_path)
    baseline = {
        "PATH": os.environ["PATH"],
        "QVC_HUBBLE_SPEED": "fastest",
        "UNCHANGED_SETTING": "sentinel",
        "QVC_HUBBLE_MINIMAL_PLOTS": "false",
        "QVC_HUBBLE_SHEN_LF_MODE": "stale",
    }
    calls = []

    def fake_run(command, *, cwd, env, check):
        calls.append((command, cwd, env.copy(), check))
        destination = (
            tmp_path
            / "plots"
            / "hubble"
            / env["QVC_HUBBLE_PREFIX"]
            / "run-tag"
            / "hubble_diagram_debiased.pdf"
        )
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"%PDF-1.4\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(comparison.subprocess, "run", fake_run)
    diagrams = comparison.run_luminosity_function_sweep(
        "comparison", xonsh_path="/usr/bin/xonsh", base_environment=baseline
    )

    assert [model for model, _ in diagrams] == list(comparison.LF_RUN_IDS)
    assert len(calls) == len(comparison.LF_RUNS)
    for (command, cwd, environment, check), run in zip(
        calls, comparison.LF_RUNS, strict=True
    ):
        model_prefix = f"comparison_{run.key}"
        expected_environment = baseline | {
            "QVC_HUBBLE_COMPLETENESS_LF_MODEL": run.lf_model,
            "QVC_HUBBLE_MINIMAL_PLOTS": "true",
            "QVC_HUBBLE_COMPLETENESS_MAGNITUDE": run.completeness_magnitude,
            "QVC_HUBBLE_PREFIX": model_prefix,
        }
        if run.shen_lf_mode is not None:
            expected_environment["QVC_HUBBLE_SHEN_LF_MODE"] = run.shen_lf_mode
        else:
            expected_environment.pop("QVC_HUBBLE_SHEN_LF_MODE")
        assert command == ["/usr/bin/xonsh", str(comparison.RUN_HUBBLE)]
        assert cwd == tmp_path
        assert environment == expected_environment
        assert check is True


def test_sweep_stops_on_the_first_failed_run(monkeypatch):
    calls = []

    def fail_run(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(comparison.subprocess, "run", fail_run)
    with pytest.raises(RuntimeError, match="failed.*shen.*exit code 7"):
        comparison.run_luminosity_function_sweep(
            "comparison", xonsh_path="xonsh", base_environment={}
        )
    assert len(calls) == 1


@pytest.mark.parametrize("count", [0, 2])
def test_diagram_discovery_requires_exactly_one_match(monkeypatch, tmp_path, count):
    monkeypatch.setattr(comparison, "REPO_ROOT", tmp_path)
    for index in range(count):
        path = (
            tmp_path
            / "plots"
            / "hubble"
            / "model-prefix"
            / f"run-{index}"
            / "hubble_diagram_debiased.pdf"
        )
        path.parent.mkdir(parents=True)
        path.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(RuntimeError, match=f"found {count}"):
        comparison.find_debiased_hubble_diagram("model-prefix")


def test_assembly_creates_one_page_with_all_eight_labels(tmp_path):
    diagrams = []
    for index, model in enumerate(comparison.LF_RUN_IDS):
        source = tmp_path / f"source-{index}.pdf"
        _write_pdf(source, f"Panel {index + 1}")
        diagrams.append((model, source))

    output = tmp_path / "comparison.pdf"
    result = comparison.assemble_comparison_pdf(diagrams, output)

    assert result == output.resolve()
    assert output.is_file()
    reader = PdfReader(output)
    assert len(reader.pages) == 1
    page = reader.pages[0]
    source_page = PdfReader(diagrams[0][1]).pages[0]
    expected_source_height = (
        float(source_page.cropbox.height)
        * comparison.COMPARISON_PANEL_WIDTH_PT
        / float(source_page.cropbox.width)
    )
    expected_page_height = (
        4 * (expected_source_height + comparison.COMPARISON_LABEL_HEIGHT_PT)
        + 3 * comparison.COMPARISON_ROW_GAP_PT
    )
    assert float(page.mediabox.width) == pytest.approx(
        2 * comparison.COMPARISON_PANEL_WIDTH_PT
        + comparison.COMPARISON_COLUMN_GAP_PT
    )
    assert float(page.mediabox.height) == pytest.approx(expected_page_height)
    assert page.cropbox == page.mediabox

    extracted = re.sub(r"\s+", " ", page.extract_text())
    for label in comparison.LF_LABELS.values():
        assert label in extracted


@pytest.mark.skipif(
    shutil.which("pdftoppm") is None,
    reason="PNG rendering test requires pdftoppm",
)
def test_comparison_png_is_rendered_from_pdf(tmp_path):
    source = tmp_path / "comparison.pdf"
    _write_pdf(source, "Comparison")
    output = tmp_path / "luminosity_function_hubble_comparison.png"

    result = comparison.render_comparison_png(
        source,
        output,
        pdftoppm_path=shutil.which("pdftoppm"),
        dpi=72,
    )

    assert result == output.resolve()
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def _write_diagnostic_inputs(
    tmp_path: Path, base_prefix: str, models=COMPLETENESS_LF_MODELS
):
    diagrams = []
    object_ids = np.asarray([f"object-{index:02d}" for index in range(18)])
    redshift = np.repeat([0.6, 1.0, 1.4, 1.8, 2.2, 2.8], 3)
    for model_index, model in enumerate(models):
        run_directory = (
            tmp_path
            / "plots"
            / "hubble"
            / f"{base_prefix}_{model}"
            / "run-tag"
        )
        run_directory.mkdir(parents=True)
        diagram = run_directory / "hubble_diagram_debiased.pdf"
        diagram.write_bytes(b"%PDF-1.4\n")
        diagrams.append((model, diagram))
        pd.DataFrame(
            {
                "object_id": object_ids,
                "is_fit_selection": True,
                "z": redshift,
                "residuals": 0.03 * np.sin(redshift) + 0.005 * model_index,
            }
        ).to_csv(run_directory / "hubble_plot_residuals.csv", index=False)

        posterior_directory = (
            tmp_path
            / "results"
            / "hubble_posteriors"
            / f"{base_prefix}_{model}"
        )
        posterior_directory.mkdir(parents=True)
        with h5py.File(posterior_directory / "posterior.h5", "w") as handle:
            handle.create_dataset(
                "object_id_fit_selection",
                data=object_ids.astype(h5py.string_dtype("utf-8")),
            )
            handle.create_dataset(
                "dmi_posterior_median",
                data=(
                    -0.1 * model_index
                    + 0.01 * redshift
                    + np.tile([-0.02, 0.0, 0.02], 6)
                ),
            )
            samples = np.zeros((20, 9), dtype=float)
            samples[:, 1] = 0.1 * model_index
            handle.create_dataset("flat_samples", data=samples)
    return diagrams


def _write_parameter_posteriors(
    tmp_path: Path, base_prefix: str, models=COMPLETENESS_LF_MODELS
):
    _, parameter_names, _ = get_model_params("Flatw0waCDM")
    for model_index, model in enumerate(models):
        posterior_directory = (
            tmp_path
            / "results"
            / "hubble_posteriors"
            / f"{base_prefix}_{model}"
        )
        posterior_directory.mkdir(parents=True)
        samples = np.empty((5, len(parameter_names)), dtype=float)
        for parameter_index in range(len(parameter_names)):
            samples[:, parameter_index] = (
                100.0 * model_index
                + 10.0 * parameter_index
                + np.arange(5, dtype=float)
            )
        with h5py.File(posterior_directory / "posterior.h5", "w") as handle:
            handle.create_dataset("flat_samples", data=samples)
    return parameter_names


def test_parameter_summaries_use_named_columns_and_canonical_model_order(tmp_path):
    parameter_names = _write_parameter_posteriors(tmp_path, "comparison")

    summaries = lf_comparison_diagnostics.load_lf_parameter_summaries(
        "comparison", repo_root=tmp_path
    )

    expected_pairs = [
        (model, parameter)
        for model in COMPLETENESS_LF_MODELS
        for parameter in lf_comparison_diagnostics.PARAMETER_COMPARISON_NAMES
    ]
    assert list(zip(summaries["model"], summaries["parameter"])) == expected_pairs
    for model_index, model in enumerate(COMPLETENESS_LF_MODELS):
        for parameter in lf_comparison_diagnostics.PARAMETER_COMPARISON_NAMES:
            parameter_index = parameter_names.index(parameter)
            row = summaries.loc[
                summaries["model"].eq(model)
                & summaries["parameter"].eq(parameter)
            ].iloc[0]
            values = (
                100.0 * model_index
                + 10.0 * parameter_index
                + np.arange(5, dtype=float)
            )
            expected = np.quantile(values, [0.16, 0.50, 0.84])
            assert row[["interval_16", "median", "interval_84"]].to_numpy(
                dtype=float
            ) == pytest.approx(expected)


def test_parameter_comparison_writes_single_page_pdf_and_png(tmp_path):
    _write_parameter_posteriors(
        tmp_path, "comparison", models=comparison.LF_RUN_IDS
    )
    output_directory = tmp_path / "plots" / "hubble" / "comparison"

    outputs = comparison.generate_lf_parameter_comparison(
        "comparison",
        output_directory,
        repo_root=tmp_path,
        models=comparison.LF_RUN_IDS,
    )

    assert set(outputs) == {"parameter_pdf", "parameter_png"}
    assert all(
        path.is_file() and path.stat().st_size > 0 for path in outputs.values()
    )
    assert len(PdfReader(outputs["parameter_pdf"]).pages) == 1
    assert outputs["parameter_png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_parameter_summaries_require_exactly_one_posterior(tmp_path):
    _write_parameter_posteriors(tmp_path, "comparison")
    shen_directory = (
        tmp_path / "results" / "hubble_posteriors" / "comparison_shen"
    )
    (shen_directory / "second.h5").write_bytes(b"")

    with pytest.raises(RuntimeError, match="found 2"):
        lf_comparison_diagnostics.load_lf_parameter_summaries(
            "comparison", repo_root=tmp_path
        )


def test_parameter_summaries_reject_missing_posterior(tmp_path):
    with pytest.raises(RuntimeError, match="found 0"):
        lf_comparison_diagnostics.load_lf_parameter_summaries(
            "comparison", repo_root=tmp_path
        )


def test_parameter_summaries_reject_incompatible_sample_shape(tmp_path):
    _write_parameter_posteriors(tmp_path, "comparison")
    shen_path = (
        tmp_path
        / "results"
        / "hubble_posteriors"
        / "comparison_shen"
        / "posterior.h5"
    )
    with h5py.File(shen_path, "w") as handle:
        handle.create_dataset("flat_samples", data=np.zeros((5, 2)))

    with pytest.raises(ValueError, match=r"expected \(\*, 9\)"):
        lf_comparison_diagnostics.load_lf_parameter_summaries(
            "comparison", repo_root=tmp_path
        )


def test_diagnostics_generate_paired_figures_and_tables(tmp_path):
    diagrams = _write_diagnostic_inputs(
        tmp_path, "comparison", models=comparison.LF_RUN_IDS
    )
    output_directory = tmp_path / "plots" / "hubble" / "comparison"

    outputs = comparison.generate_lf_comparison_diagnostics(
        "comparison",
        diagrams,
        output_directory,
        repo_root=tmp_path,
        bootstrap_draws=20,
    )

    assert set(outputs) == {
        "diagnostic_pdf",
        "diagnostic_png",
        "summary_csv",
        "binned_csv",
        "readme",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
    summary = pd.read_csv(outputs["summary_csv"])
    assert summary["model"].tolist() == list(comparison.LF_RUN_IDS)
    assert set(summary["n_paired"]) == {18}
    assert np.allclose(summary["median_delta_dmi_plus_delta_M0_mag"], 0.0)
    binned = pd.read_csv(outputs["binned_csv"])
    delta_rows = binned["quantity"].isin(
        [
            "delta_residual_vs_shen",
            "delta_dmi_vs_shen",
            "delta_dmi_plus_delta_M0_vs_shen",
        ]
    )
    assert set(binned.loc[delta_rows, "interval_kind"]) == {"paired_distribution"}
    absolute_rows = binned["quantity"].eq("dmi")
    assert set(binned.loc[absolute_rows, "interval_kind"]) == {
        "object_distribution"
    }
    assert np.all(
        binned.loc[absolute_rows, "interval_16"]
        < binned.loc[absolute_rows, "median"]
    )
    assert np.all(
        binned.loc[absolute_rows, "median"]
        < binned.loc[absolute_rows, "interval_84"]
    )
    readme = outputs["readme"].read_text()
    assert "same 18 fit-selection object IDs" in readme
    assert "actual object-level dmi values" in readme
    assert "actual paired object differences" in readme
    assert "style.mplstyle" in readme


def test_main_generates_diagnostics_after_comparison_pdf(monkeypatch, tmp_path):
    monkeypatch.setattr(comparison, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(comparison, "_required_executable", lambda name: name)
    diagrams = [
        (model, tmp_path / f"{model}.pdf") for model in comparison.LF_RUN_IDS
    ]
    monkeypatch.setattr(
        comparison,
        "run_luminosity_function_sweep",
        lambda prefix, *, xonsh_path: diagrams,
    )

    def fake_assemble(received, output_path):
        assert received == diagrams
        output_path.parent.mkdir(parents=True)
        output_path.write_bytes(b"%PDF-1.4\n")
        return output_path.resolve()

    rendered = []

    def fake_render(pdf_path, output_path, *, pdftoppm_path):
        rendered.append((pdf_path, output_path, pdftoppm_path))
        output_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        return output_path.resolve()

    calls = []

    def fake_diagnostics(prefix, received, output_directory):
        calls.append((prefix, received, output_directory))
        return {
            "diagnostic_pdf": (
                output_directory / "lf_selection_correction_and_hubble_residuals.pdf"
            )
        }

    monkeypatch.setattr(comparison, "assemble_comparison_pdf", fake_assemble)
    monkeypatch.setattr(comparison, "render_comparison_png", fake_render)
    monkeypatch.setattr(
        comparison, "generate_lf_comparison_diagnostics", fake_diagnostics
    )
    parameter_calls = []

    def fake_parameter_comparison(prefix, output_directory, *, models):
        parameter_calls.append((prefix, output_directory, models))
        return {
            "parameter_pdf": output_directory / "lf_parameter_comparison.pdf",
            "parameter_png": output_directory / "lf_parameter_comparison.png",
        }

    monkeypatch.setattr(
        comparison, "generate_lf_parameter_comparison", fake_parameter_comparison
    )

    assert comparison.main(["--prefix", "comparison"]) == 0
    expected_directory = tmp_path / "plots" / "hubble" / "comparison"
    assert calls == [("comparison", diagrams, expected_directory)]
    assert parameter_calls == [
        ("comparison", expected_directory, comparison.LF_RUN_IDS)
    ]
    expected_pdf = expected_directory / comparison.COMPARISON_FILENAME
    assert rendered == [
        (expected_pdf.resolve(), expected_pdf.with_suffix(".png"), "pdftoppm")
    ]
