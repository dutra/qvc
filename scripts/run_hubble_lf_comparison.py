#!/usr/bin/env python3
"""Run the Hubble workflow for every supported luminosity function.

Each child run delegates to ``run_hubble.xonsh`` with the minimal plot set.
The resulting debiased Hubble diagrams are assembled into a labeled, single-
tightly cropped, single-page PDF without rasterizing the source figures, with
a matching PNG rendering written alongside it. Paired residual and selection-
correction diagnostics are then generated from the six residual tables and
posterior checkpoints.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from pypdf import PageObject, PdfReader, PdfWriter, Transformation


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble.completeness_mock_catalog import COMPLETENESS_LF_MODELS
from qvc.hubble.lf_comparison_diagnostics import (
    generate_lf_comparison_diagnostics,
)


RUN_HUBBLE = REPO_ROOT / "run_hubble.xonsh"
DEFAULT_SPEED = "quick"
COMPARISON_FILENAME = "luminosity_function_hubble_comparison.pdf"
COMPARISON_PNG_FILENAME = "luminosity_function_hubble_comparison.png"
COMPARISON_PNG_DPI = 200
COMPARISON_PANEL_WIDTH_PT = 280.0
COMPARISON_LABEL_HEIGHT_PT = 17.0
COMPARISON_COLUMN_GAP_PT = 6.0
COMPARISON_ROW_GAP_PT = 4.0
LF_LABELS = {
    "shen": "Shen et al. (2020)",
    "wang2026_type1_lade_a": "Wang et al. (2026), LADE-A",
    "palanque2016_ple_lede": "Palanque-Delabrouille et al. (2016), PLE+LEDE",
    "kulkarni2019_type1_model1": "Kulkarni et al. (2019), Model 1",
    "kulkarni2019_type1_model2": "Kulkarni et al. (2019), Model 2",
    "kulkarni2019_type1_model3": "Kulkarni et al. (2019), Model 3",
}


def _validate_model_labels(models: Sequence[str]) -> None:
    missing = [model for model in models if model not in LF_LABELS]
    if missing:
        raise RuntimeError(
            "Missing comparison labels for supported luminosity functions: "
            + ", ".join(missing)
        )


def _validate_prefix(prefix: str) -> str:
    prefix = prefix.strip()
    prefix_path = Path(prefix)
    if not prefix or prefix_path.is_absolute() or ".." in prefix_path.parts:
        raise ValueError("--prefix must be a non-empty path below plots/hubble/.")
    return prefix


def build_child_environment(
    base_environment: Mapping[str, str], model: str, model_prefix: str
) -> dict[str, str]:
    """Return a child environment differing only in sweep-specific settings."""
    environment = dict(base_environment)
    environment.update(
        {
            "QVC_HUBBLE_COMPLETENESS_LF_MODEL": model,
            "QVC_HUBBLE_MINIMAL_PLOTS": "true",
            "QVC_HUBBLE_COMPLETENESS_MAGNITUDE": "attenuated",
            "QVC_HUBBLE_PREFIX": model_prefix,
        }
    )
    return environment


def find_debiased_hubble_diagram(model_prefix: str) -> Path:
    """Find the sole final debiased diagram produced by one child run."""
    run_directory = REPO_ROOT / "plots" / "hubble" / model_prefix
    matches = sorted(run_directory.rglob("hubble_diagram_debiased.pdf"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one hubble_diagram_debiased.pdf for prefix "
            f"{model_prefix!r} under {run_directory}, found {len(matches)}."
        )
    return matches[0]


def run_luminosity_function_sweep(
    base_prefix: str,
    *,
    xonsh_path: str,
    base_environment: Mapping[str, str] | None = None,
) -> list[tuple[str, Path]]:
    """Run ``run_hubble.xonsh`` once for every canonical LF model."""
    models = tuple(COMPLETENESS_LF_MODELS)
    _validate_model_labels(models)
    base_environment = os.environ if base_environment is None else base_environment
    diagrams: list[tuple[str, Path]] = []

    for index, model in enumerate(models, start=1):
        model_prefix = f"{base_prefix}_{model}"
        environment = build_child_environment(
            base_environment, model=model, model_prefix=model_prefix
        )
        print(f"[{index}/{len(models)}] Running luminosity function: {model}", flush=True)
        try:
            subprocess.run(
                [xonsh_path, str(RUN_HUBBLE)],
                cwd=REPO_ROOT,
                env=environment,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"run_hubble.xonsh failed for luminosity function {model!r} "
                f"with exit code {error.returncode}."
            ) from error
        diagrams.append((model, find_debiased_hubble_diagram(model_prefix)))

    return diagrams


def assemble_comparison_pdf(
    diagrams: Sequence[tuple[str, Path]],
    output_path: Path,
) -> Path:
    """Compose six source PDFs onto one zero-margin vector 2-by-3 page."""
    models = [model for model, _ in diagrams]
    if tuple(models) != tuple(COMPLETENESS_LF_MODELS):
        raise ValueError(
            "Diagrams must contain every supported luminosity function exactly "
            "once and in canonical order."
        )
    _validate_model_labels(models)

    missing = [str(path) for _, path in diagrams if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source Hubble diagram(s): " + ", ".join(missing))

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    readers = [PdfReader(str(path)) for _, path in diagrams]
    if any(len(reader.pages) != 1 for reader in readers):
        raise ValueError("Every source Hubble diagram must contain exactly one page.")

    source_pages = [reader.pages[0] for reader in readers]
    scales = [
        COMPARISON_PANEL_WIDTH_PT / float(page.cropbox.width)
        for page in source_pages
    ]
    source_heights = [
        float(page.cropbox.height) * scale
        for page, scale in zip(source_pages, scales, strict=True)
    ]
    source_region_height = max(source_heights)
    panel_height = source_region_height + COMPARISON_LABEL_HEIGHT_PT
    page_width = 2 * COMPARISON_PANEL_WIDTH_PT + COMPARISON_COLUMN_GAP_PT
    page_height = 3 * panel_height + 2 * COMPARISON_ROW_GAP_PT
    composite = PageObject.create_blank_page(width=page_width, height=page_height)

    label_buffer = BytesIO()
    with plt.rc_context(
        {
            "font.family": "serif",
            "savefig.bbox": None,
            "savefig.pad_inches": 0,
        }
    ):
        label_figure = plt.figure(figsize=(page_width / 72, page_height / 72))
        label_figure.patch.set_alpha(0)
        for index, (model, source_page) in enumerate(
            zip(models, source_pages, strict=True)
        ):
            row, column = divmod(index, 2)
            source_height = source_heights[index]
            x = column * (COMPARISON_PANEL_WIDTH_PT + COMPARISON_COLUMN_GAP_PT)
            source_y = (
                page_height
                - (row + 1) * panel_height
                - row * COMPARISON_ROW_GAP_PT
            )
            source_y += 0.5 * (source_region_height - source_height)
            composite.merge_transformed_page(
                source_page,
                Transformation().scale(scales[index]).translate(x, source_y),
            )
            label_y = (
                page_height
                - row * (panel_height + COMPARISON_ROW_GAP_PT)
                - 0.5 * COMPARISON_LABEL_HEIGHT_PT
            )
            label_figure.text(
                (x + 0.5 * COMPARISON_PANEL_WIDTH_PT) / page_width,
                label_y / page_height,
                LF_LABELS[model],
                ha="center",
                va="center",
                fontsize=9.5,
                fontweight="bold",
            )
        label_figure.savefig(
            label_buffer,
            format="pdf",
            transparent=True,
            bbox_inches=None,
            pad_inches=0,
        )
        plt.close(label_figure)

    label_buffer.seek(0)
    composite.merge_page(PdfReader(label_buffer).pages[0])
    writer = PdfWriter()
    writer.add_page(composite)
    writer.pages[0].compress_content_streams()
    writer.compress_identical_objects(remove_identicals=True, remove_orphans=True)
    with output_path.open("wb") as handle:
        writer.write(handle)

    return output_path


def render_comparison_png(
    pdf_path: Path,
    output_path: Path,
    *,
    pdftoppm_path: str,
    dpi: int = COMPARISON_PNG_DPI,
) -> Path:
    """Render the single-page comparison PDF to a PNG companion."""
    if dpi < 1:
        raise ValueError("PNG resolution must be positive.")
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Comparison PDF does not exist: {pdf_path}")

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_prefix = output_path.with_suffix("")
    completed = subprocess.run(
        [
            pdftoppm_path,
            "-png",
            "-singlefile",
            "-r",
            str(dpi),
            str(pdf_path),
            str(output_prefix),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            "pdftoppm failed while rendering the luminosity-function "
            f"comparison PNG: {diagnostic}"
        )
    if not output_path.is_file():
        raise RuntimeError("pdftoppm completed without producing the comparison PNG.")
    return output_path


def _required_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"Required executable {name!r} was not found on PATH. "
            f"Install {name} before running this comparison."
        )
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    speed = os.environ.get("QVC_HUBBLE_SPEED", DEFAULT_SPEED)
    default_prefix = os.environ.get(
        "QVC_HUBBLE_LF_COMPARISON_PREFIX", f"hubble_lf_comparison_{speed}"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        default=default_prefix,
        help="Base directory name below plots/hubble/ (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Comparison PDF path. A same-stem PNG is also generated. Relative "
            "paths are resolved from the repository root; the default is "
            f"plots/hubble/<prefix>/{COMPARISON_FILENAME}."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prefix = _validate_prefix(args.prefix)
        xonsh_path = _required_executable("xonsh")
        pdftoppm_path = _required_executable("pdftoppm")
        output_path = args.output
        if output_path is None:
            output_directory = REPO_ROOT / "plots" / "hubble" / prefix
            output_path = output_directory / COMPARISON_FILENAME
            png_output_path = output_directory / COMPARISON_PNG_FILENAME
        elif not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
            png_output_path = output_path.with_suffix(".png")
        else:
            png_output_path = output_path.with_suffix(".png")

        diagrams = run_luminosity_function_sweep(prefix, xonsh_path=xonsh_path)
        result = assemble_comparison_pdf(diagrams, output_path)
        png_result = render_comparison_png(
            result,
            png_output_path,
            pdftoppm_path=pdftoppm_path,
        )
        diagnostic_paths = generate_lf_comparison_diagnostics(
            prefix,
            diagrams,
            output_path.parent,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Wrote luminosity-function comparison PDF: {result}")
    print(f"Wrote luminosity-function comparison PNG: {png_result}")
    for name, path in diagnostic_paths.items():
        print(f"Wrote LF diagnostic {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
