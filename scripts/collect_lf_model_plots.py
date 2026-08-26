#!/usr/bin/env python3
"""Collect an all-LF Hubble sweep and stamp every PDF with its LF model."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont


DEFAULT_PREFIX = "aug25_all_lf_models_m2500norm12_quick"
LF_LABELS = {
    "shen_type1_attenuated": "Shen et al. - Type 1, attenuated",
    "wang2026_type1_lade_a": "Wang et al. (2026) - Type 1 LADE-A",
    "palanque2016_ple_lede": "Palanque-Delabrouille et al. (2016) - PLE+LEDE",
    "kulkarni2019_type1_model1": "Kulkarni et al. (2019) - Type 1, Model 1",
    "kulkarni2019_type1_model2": "Kulkarni et al. (2019) - Type 1, Model 2",
    "kulkarni2019_type1_model3": "Kulkarni et al. (2019) - Type 1, Model 3",
}
MODES = ("3d_fhost", "2d")
OUTPUT_DPI = 300
HEADER_HEIGHT_PX = 275


def parse_run_name(name: str, prefix: str) -> tuple[str, str]:
    stem = f"{prefix}_"
    if not name.startswith(stem):
        raise ValueError(f"Run directory does not start with {stem!r}: {name}")
    descriptor = name[len(stem) :]
    for mode in MODES:
        suffix = f"_{mode}"
        if descriptor.endswith(suffix):
            lf_model = descriptor[: -len(suffix)]
            if lf_model not in LF_LABELS:
                raise ValueError(f"Unknown LF model in run directory: {name}")
            return lf_model, mode
    raise ValueError(f"Cannot determine completeness mode from: {name}")


def fitted_font(font_path: str, text: str, max_size: int, max_width: int) -> ImageFont.FreeTypeFont:
    size = max_size
    while size > 36:
        font = ImageFont.truetype(font_path, size)
        if font.getlength(text) <= max_width:
            return font
        size -= 2
    return ImageFont.truetype(font_path, size)


def stamp_pdf(source: Path, destination: Path, title: str, subtitle: str) -> int:
    bold_font_path = font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans", weight="bold")
    )
    with tempfile.TemporaryDirectory(prefix="qvc-lf-plot-") as temporary_dir:
        raster_stem = Path(temporary_dir) / "plot"
        subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-singlefile",
                "-r",
                str(OUTPUT_DPI),
                str(source),
                str(raster_stem),
            ],
            check=True,
        )
        with Image.open(raster_stem.with_suffix(".png")) as original:
            original = original.convert("RGB")
            canvas = Image.new(
                "RGB", (original.width, original.height + HEADER_HEIGHT_PX), "#f0f3f7"
            )
            canvas.paste(original, (0, HEADER_HEIGHT_PX))
            draw = ImageDraw.Draw(canvas)
            draw.line(
                (0, HEADER_HEIGHT_PX - 2, original.width, HEADER_HEIGHT_PX - 2),
                fill="#26364a",
                width=5,
            )
            title_font = fitted_font(bold_font_path, title, 76, original.width - 100)
            subtitle_font = fitted_font(bold_font_path, subtitle, 36, original.width - 100)
            title_box = draw.textbbox((0, 0), title, font=title_font)
            subtitle_box = draw.textbbox((0, 0), subtitle, font=subtitle_font)
            draw.text(
                ((original.width - (title_box[2] - title_box[0])) / 2, 58),
                title,
                font=title_font,
                fill="#172235",
            )
            draw.text(
                ((original.width - (subtitle_box[2] - subtitle_box[0])) / 2, 175),
                subtitle,
                font=subtitle_font,
                fill="#43546a",
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(destination, "PDF", resolution=OUTPUT_DPI, quality=95)
    return 1


def collect(source_root: Path, output_dir: Path, prefix: str) -> list[tuple[Path, Path, str, str]]:
    run_dirs = sorted(path for path in source_root.glob(f"{prefix}_*") if path.is_dir())
    records: list[tuple[Path, Path, str, str]] = []
    for run_dir in run_dirs:
        lf_model, mode = parse_run_name(run_dir.name, prefix)
        for source in sorted(run_dir.rglob("hubble_diagram_debiased.pdf")):
            destination = output_dir / f"{lf_model}__{mode}__hubble_diagram_debiased.pdf"
            if any(existing[1] == destination for existing in records):
                raise ValueError(f"Output filename collision: {destination}")
            records.append((source, destination, lf_model, mode))

    if not records:
        raise FileNotFoundError(f"No PDF plots found for {prefix!r} below {source_root}")

    for source, destination, lf_model, mode in records:
        title = f"LF: {LF_LABELS[lf_model]}"
        mode_label = "3D host-fraction completeness" if mode == "3d_fhost" else "2D completeness"
        stamp_pdf(source, destination, title, f"{mode_label}  |  {source.name}")
    return records


def write_manifest(
    output_dir: Path,
    source_root: Path,
    prefix: str,
    records: list[tuple[Path, Path, str, str]],
) -> None:
    lines = [
        "# Labeled luminosity-function plots",
        "",
        f"Source sweep: `{source_root / (prefix + '_*')}`",
        f"Collected Hubble-diagram PDFs: **{len(records)}**",
        "",
        "Each PDF retains the original plot at its original size and adds a title band",
        "stamped with the LF model and completeness mode.",
        "",
        "| LF model | Mode | Plot |",
        "|---|---|---|",
    ]
    for source, destination, lf_model, mode in records:
        lines.append(f"| {LF_LABELS[lf_model]} | {mode} | `{destination.name}` |")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("plots/hubble"))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/hubble/all_lf_models_labeled"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = collect(args.source_root, args.output_dir, args.prefix)
    write_manifest(args.output_dir, args.source_root, args.prefix, records)
    print(f"Wrote {len(records)} labeled PDFs to {args.output_dir}")


if __name__ == "__main__":
    main()
