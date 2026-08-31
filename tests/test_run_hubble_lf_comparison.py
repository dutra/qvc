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

from qvc.hubble.completeness_mock_catalog import COMPLETENESS_LF_MODELS


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
    assert tuple(comparison.LF_LABELS) == tuple(COMPLETENESS_LF_MODELS)


def test_sweep_runs_every_model_with_only_the_intended_environment_changes(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(comparison, "REPO_ROOT", tmp_path)
    baseline = {
        "PATH": os.environ["PATH"],
        "QVC_HUBBLE_SPEED": "fastest",
        "UNCHANGED_SETTING": "sentinel",
        "QVC_HUBBLE_MINIMAL_PLOTS": "false",
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

    assert [model for model, _ in diagrams] == list(COMPLETENESS_LF_MODELS)
    assert len(calls) == len(COMPLETENESS_LF_MODELS)
    for (command, cwd, environment, check), model in zip(
        calls, COMPLETENESS_LF_MODELS, strict=True
    ):
        model_prefix = f"comparison_{model}"
        expected_environment = baseline | {
            "QVC_HUBBLE_COMPLETENESS_LF_MODEL": model,
            "QVC_HUBBLE_MINIMAL_PLOTS": "true",
            "QVC_HUBBLE_COMPLETENESS_MAGNITUDE": "attenuated",
            "QVC_HUBBLE_PREFIX": model_prefix,
        }
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


def test_assembly_creates_one_page_with_all_six_labels(tmp_path):
    diagrams = []
    for index, model in enumerate(COMPLETENESS_LF_MODELS):
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
        3 * (expected_source_height + comparison.COMPARISON_LABEL_HEIGHT_PT)
        + 2 * comparison.COMPARISON_ROW_GAP_PT
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


def _write_diagnostic_inputs(tmp_path: Path, base_prefix: str):
    diagrams = []
    object_ids = np.asarray([f"object-{index:02d}" for index in range(18)])
    redshift = np.repeat([0.6, 1.0, 1.4, 1.8, 2.2, 2.8], 3)
    for model_index, model in enumerate(COMPLETENESS_LF_MODELS):
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
                data=-0.1 * model_index + 0.01 * redshift,
            )
            samples = np.zeros((20, 9), dtype=float)
            samples[:, 1] = 0.1 * model_index
            handle.create_dataset("flat_samples", data=samples)
    return diagrams


def test_diagnostics_generate_paired_figures_and_tables(tmp_path):
    diagrams = _write_diagnostic_inputs(tmp_path, "comparison")
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
    assert summary["model"].tolist() == list(COMPLETENESS_LF_MODELS)
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
    assert set(binned.loc[~delta_rows, "interval_kind"]) == {"bootstrap_median"}
    readme = outputs["readme"].read_text()
    assert "same 18 fit-selection object IDs" in readme
    assert "actual paired object differences" in readme
    assert "style.mplstyle" in readme


def test_main_generates_diagnostics_after_comparison_pdf(monkeypatch, tmp_path):
    monkeypatch.setattr(comparison, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(comparison, "_required_executable", lambda name: name)
    diagrams = [(model, tmp_path / f"{model}.pdf") for model in COMPLETENESS_LF_MODELS]
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

    assert comparison.main(["--prefix", "comparison"]) == 0
    expected_directory = tmp_path / "plots" / "hubble" / "comparison"
    assert calls == [("comparison", diagrams, expected_directory)]
    expected_pdf = expected_directory / comparison.COMPARISON_FILENAME
    assert rendered == [
        (expected_pdf.resolve(), expected_pdf.with_suffix(".png"), "pdftoppm")
    ]
