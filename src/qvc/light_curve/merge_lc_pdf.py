#!/usr/bin/env python3
"""Merge rendered light-curve figure PDFs into one stamped PDF.

Workflow:
1. Load all rows from an input CSV.
2. Resolve one rendered light-curve figure PDF per row using ``object_id``.
3. Stamp the first page with row metadata.
4. Append one page per CSV row (in CSV order), skipping missing figures.
"""

from __future__ import annotations

import argparse
import io
import math
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    print("ERROR: Please install 'pypdf' (pip install pypdf)", file=sys.stderr)
    sys.exit(1)

try:
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
except ImportError:
    print("ERROR: Please install 'reportlab' (pip install reportlab)", file=sys.stderr)
    sys.exit(1)


REQUIRED_COLS = {"object_id"}
STAMP_FIELD_SPECS = (
    ("z", "numeric"),
    ("sdss_name", "string"),
    ("object_id", "string"),
    ("apparent_mag_2500", "numeric"),
    ("PL_slope", "numeric"),
    ("f_host_2500", "numeric"),
    ("f_bc_over_pl_3000", "numeric"),
    ("f_fe_uv_over_pl_3000", "numeric"),
    ("log_sigma_uv", "numeric"),
    ("log_tau_uv_rf", "numeric"),
    ("wrms", "numeric"),
    ("variability_chi_sq_red_g", "numeric"),
)


def parse_args():
    p = argparse.ArgumentParser(description="Build a stamped combined PDF from rendered light-curve figures.")
    p.add_argument("csv", type=Path, help="Input CSV with one row per entry to include.")
    p.add_argument("output", type=Path, help="Output merged PDF path.")
    p.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("plots/multiband"),
        help="Directory containing rendered light-curve PDFs. Default: plots/multiband",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of worker processes. Default: CPU-1",
    )
    p.add_argument("--stamp-font-size", type=int, default=10, help="Overlay stamp font size.")
    p.add_argument("--stamp-margin-mm", type=float, default=2.0, help="Overlay stamp margin (mm).")
    p.add_argument("--dry-run", action="store_true", help="Resolve and stamp in memory but do not write output.")
    return p.parse_args()


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"object_id": "string"})
    missing = [c for c in sorted(REQUIRED_COLS) if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    return df.copy()


def normalize_object_id_for_lookup(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return text
    if math.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text


def _extract_object_id_from_stem(stem: str) -> str | None:
    marker = "_light_curve_"
    if marker in stem:
        prefix = stem.split(marker, 1)[0]
        if "_" in prefix:
            return prefix.rsplit("_", 1)[1].strip() or None
    return None


def build_figure_index(fig_dir: Path) -> dict[str, object]:
    by_object_id: dict[str, list[Path]] = defaultdict(list)
    all_paths: list[Path] = []
    for path in fig_dir.rglob("*.pdf"):
        if not path.is_file():
            continue
        all_paths.append(path)
        object_id = _extract_object_id_from_stem(path.stem)
        if object_id is None:
            continue
        by_object_id[object_id].append(path)

    for object_id, paths in by_object_id.items():
        by_object_id[object_id] = sorted(paths)

    return {
        "by_object_id": dict(by_object_id),
        "all_paths": sorted(all_paths),
    }


def find_pdf_for_row(row: pd.Series, fig_index: dict[str, object]) -> Path | None:
    object_id = normalize_object_id_for_lookup(row.get("object_id"))
    if not object_id:
        return None

    candidates = list(fig_index["by_object_id"].get(object_id, []))
    if not candidates:
        candidates = [p for p in fig_index["all_paths"] if object_id in p.stem]
        candidates = sorted(candidates)
    if not candidates:
        return None

    z = pd.to_numeric(row.get("z", np.nan), errors="coerce")
    if pd.notna(z):
        pref_prefix = f"{float(z):.1f}_{object_id}_light_curve_"
        preferred = [p for p in candidates if p.stem.startswith(pref_prefix)]
        if preferred:
            return preferred[0]

    return candidates[0]


def _format_stamp_value(value) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.notna(x):
        xf = float(x)
        if math.isfinite(xf):
            return f"{xf:.4g}"
    if value is None or pd.isna(value):
        return "NA"
    text = str(value).strip()
    return text or "NA"


def _format_stamp_string(value) -> str:
    if value is None or pd.isna(value):
        return "NA"
    text = str(value).strip()
    return text or "NA"


def make_stamp_text(row: pd.Series) -> str:
    parts = []
    for field, value_type in STAMP_FIELD_SPECS:
        formatter = _format_stamp_string if value_type == "string" else _format_stamp_value
        parts.append(f"{field}={formatter(row.get(field))}")
    return " | ".join(parts)


def _wrap_text_to_width(text: str, max_width: float, measure_text) -> list[str]:
    text = str(text).strip()
    if not text:
        return [""]

    max_width = max(float(max_width), 1.0)
    lines: list[str] = []

    def append_wrapped_chunk(chunk: str):
        chunk = chunk.strip()
        if not chunk:
            return
        if measure_text(chunk) <= max_width:
            lines.append(chunk)
            return

        words = chunk.split()
        if len(words) > 1:
            current = words[0]
            for word in words[1:]:
                candidate = f"{current} {word}"
                if measure_text(candidate) <= max_width:
                    current = candidate
                else:
                    lines.append(current)
                    current = word
            append_wrapped_chunk(current)
            return

        current = ""
        for char in chunk:
            candidate = f"{current}{char}"
            if current and measure_text(candidate) > max_width:
                lines.append(current)
                current = char
            else:
                current = candidate
        if current:
            lines.append(current)

    current_line = ""
    for segment in text.split(" | "):
        segment = segment.strip()
        if not segment:
            continue
        candidate = segment if not current_line else f"{current_line} | {segment}"
        if measure_text(candidate) <= max_width:
            current_line = candidate
            continue
        if current_line:
            lines.append(current_line)
            current_line = ""
        append_wrapped_chunk(segment)

    if current_line:
        lines.append(current_line)

    return lines or [text]


def _make_overlay(page_width, page_height, text, margin_pts, font_size):
    buf = io.BytesIO()
    try:
        pdfmetrics.registerFont(TTFont("DejaVuSans", "DejaVuSans.ttf"))
        font_name = "DejaVuSans"
    except Exception:
        font_name = "Helvetica"

    c = canvas.Canvas(buf, pagesize=(page_width, page_height))
    c.setFont(font_name, font_size)

    pad = 1.0
    x = margin_pts
    y = margin_pts
    max_text_width = max(float(page_width) - 2.0 * float(margin_pts) - 2.0 * pad, 1.0)
    lines = _wrap_text_to_width(
        text,
        max_text_width,
        lambda line: c.stringWidth(line, font_name, font_size),
    )
    text_width = max(c.stringWidth(line, font_name, font_size) for line in lines)
    line_height = font_size * 1.2
    box_height = len(lines) * line_height + 2 * pad

    c.setFillGray(1.0)
    c.rect(x - pad, y - pad, text_width + 2 * pad, box_height, fill=1, stroke=0)
    c.setFillGray(0.0)
    top_y = y + pad + (len(lines) - 1) * line_height
    for idx, line in enumerate(lines):
        c.drawString(x, top_y - idx * line_height, line)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf


def stamp_first_page_to_bytes(src_pdf: Path, text: str, font_size: int, margin_mm: float) -> bytes:
    reader = PdfReader(str(src_pdf))
    if len(reader.pages) < 1:
        raise ValueError(f"PDF has no pages: {src_pdf}")

    page = reader.pages[0]
    box = getattr(page, "cropbox", None) or page.mediabox
    page_width, page_height = float(box.width), float(box.height)
    if page_width <= 0 or page_height <= 0:
        raise ValueError(f"Invalid page size in {src_pdf}")

    overlay_buf = _make_overlay(page_width, page_height, text, float(margin_mm) * mm, font_size)
    overlay = PdfReader(overlay_buf).pages[0]
    page.merge_page(overlay)

    writer = PdfWriter()
    writer.add_page(page)

    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out.getvalue()


def _worker_stamp(task):
    idx, src_pdf_str, stamp_text, font_size, margin_mm = task
    if src_pdf_str is None:
        return idx, None, "missing"
    try:
        out = stamp_first_page_to_bytes(Path(src_pdf_str), stamp_text, font_size, margin_mm)
        return idx, out, None
    except Exception as exc:
        return idx, None, str(exc)


def stamp_tasks(tasks: list[tuple], workers: int) -> tuple[list[bytes | None], int, int]:
    stamped_pages: list[bytes | None] = [None] * len(tasks)
    skipped_missing = 0
    stamp_failures = 0

    workers = max(1, int(workers))
    if workers == 1:
        iterator = map(_worker_stamp, tasks)
    else:
        pool = mp.Pool(processes=workers)
        iterator = pool.imap_unordered(_worker_stamp, tasks, chunksize=16)

    try:
        for idx, stamped_bytes, err in tqdm(iterator, total=len(tasks), desc="Stamping pages"):
            if stamped_bytes is None:
                if err == "missing":
                    skipped_missing += 1
                else:
                    stamp_failures += 1
            stamped_pages[idx] = stamped_bytes
    finally:
        if workers != 1:
            pool.close()
            pool.join()

    return stamped_pages, skipped_missing, stamp_failures


def write_merged_pdf(output: Path, stamped_pages: list[bytes | None]) -> int:
    kept = [buf for buf in stamped_pages if buf is not None]
    if not kept:
        print("[ERROR] No output pages were produced.", file=sys.stderr)
        return 1

    writer = PdfWriter()
    for buf in kept:
        reader = PdfReader(io.BytesIO(buf))
        writer.add_page(reader.pages[0])

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        writer.write(f)
    print(f"[OK] Wrote combined PDF: {output}")
    return 0


def main():
    args = parse_args()

    if not args.csv.exists():
        print(f"[ERROR] CSV not found: {args.csv}", file=sys.stderr)
        return 2
    if not args.fig_dir.exists():
        print(f"[ERROR] Figure directory not found: {args.fig_dir}", file=sys.stderr)
        return 2

    try:
        df = load_csv(args.csv)
    except Exception as exc:
        print(f"[ERROR] Failed to load CSV: {exc}", file=sys.stderr)
        return 2

    if df.empty:
        print("[WARN] CSV has zero rows. Nothing to do.", file=sys.stderr)
        return 1

    fig_index = build_figure_index(args.fig_dir)
    if not fig_index["all_paths"]:
        print(f"[WARN] No PDFs indexed under {args.fig_dir}", file=sys.stderr)

    tasks = []
    missing_pre = 0
    missing_object_ids = []
    for idx, (_, row) in enumerate(df.iterrows()):
        src_pdf = find_pdf_for_row(row, fig_index)
        object_id = _format_stamp_string(row.get("object_id"))
        if src_pdf is None:
            missing_pre += 1
            missing_object_ids.append(object_id)
        tasks.append(
            (
                idx,
                str(src_pdf) if src_pdf is not None else None,
                make_stamp_text(row),
                int(args.stamp_font_size),
                float(args.stamp_margin_mm),
            )
        )

    if missing_pre:
        print(f"[WARN] Missing figure for {missing_pre} row(s) before stamping.", file=sys.stderr)
        preview = ", ".join(missing_object_ids[:10])
        if preview:
            print(f"[WARN] First missing object_id values: {preview}", file=sys.stderr)

    stamped_pages, skipped_missing, stamp_failures = stamp_tasks(tasks, workers=args.workers)
    kept = sum(1 for page in stamped_pages if page is not None)
    print(
        f"[INFO] Rows={len(df)} kept={kept} skipped={len(df) - kept} "
        f"(missing={skipped_missing}, stamp_failures={stamp_failures})"
    )

    if args.dry_run:
        print("[DRY] Dry run complete; output PDF not written.")
        return 0

    return write_merged_pdf(args.output, stamped_pages)


if __name__ == "__main__":
    raise SystemExit(main())
