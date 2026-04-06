#!/usr/bin/env python3
"""Merge jaxqsofit figure PDFs into one stamped PDF.

Workflow:
1. Load all rows from an input CSV.
2. Resolve one figure PDF per row using `sdss_name` (prefer z-matched filename).
3. Stamp the figure with host/BC/iron fractions.
4. Append one page per CSV row (in CSV order), skipping missing figures.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    print("ERROR: Please install 'pypdf' (pip install pypdf)", file=sys.stderr)
    sys.exit(1)

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
except ImportError:
    print("ERROR: Please install 'reportlab' (pip install reportlab)", file=sys.stderr)
    sys.exit(1)


REQUIRED_COLS = {
    "sdss_name",
    #"f_host_center",
    #"f_bc_3000",
    #"f_fe_uv_3000",
}

SDSS_TOKEN_RE = re.compile(r"\d{6}\.\d{2}[+-]\d{6}\.\d")


def parse_args():
    p = argparse.ArgumentParser(description="Build a stamped combined PDF from jaxqsofit figures.")
    p.add_argument("csv", type=Path, help="Input CSV with one row per entry to include.")
    p.add_argument("output", type=Path, help="Output merged PDF path.")
    p.add_argument("--fig-dir", type=Path, default=Path("plots/jaxqsofit"),
                   help="Directory containing figure PDFs. Default: plots/jaxqsofit")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="Number of worker processes. Default: CPU-1")
    p.add_argument("--stamp-font-size", type=int, default=10, help="Overlay stamp font size.")
    p.add_argument("--stamp-margin-mm", type=float, default=2.0, help="Overlay stamp margin (mm).")
    p.add_argument("--dry-run", action="store_true", help="Resolve and stamp in memory but do not write output.")
    return p.parse_args()


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in sorted(REQUIRED_COLS) if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df.copy()
    df["sdss_name"] = df["sdss_name"].astype(str).str.strip()
    return df


def _extract_sdss_name_from_stem(stem: str) -> str | None:
    # Typical fit_spectra names: z0.907_212805.25-005145.7
    if stem.startswith("z") and "_" in stem:
        tail = stem.split("_", 1)[1].strip()
        # Handle accidental copies like "... (Copy)"
        if tail.endswith(" (Copy)"):
            tail = tail[:-7].strip()
        token_match = SDSS_TOKEN_RE.search(tail)
        if token_match:
            return token_match.group(0)
        return tail or None

    token_match = SDSS_TOKEN_RE.search(stem)
    if token_match:
        return token_match.group(0)
    return None


def build_figure_index(fig_dir: Path) -> dict[str, list[Path]]:
    by_sdss: dict[str, list[Path]] = defaultdict(list)
    for p in fig_dir.rglob("*.pdf"):
        if not p.is_file():
            continue
        sdss = _extract_sdss_name_from_stem(p.stem)
        if sdss is None:
            continue
        by_sdss[sdss].append(p)

    for sdss, paths in by_sdss.items():
        by_sdss[sdss] = sorted(paths)
    return dict(by_sdss)


def find_pdf_for_row(row: pd.Series, by_sdss: dict[str, list[Path]]) -> Path | None:
    sdss = str(row["sdss_name"]).strip()
    candidates = by_sdss.get(sdss, [])
    if not candidates:
        return None

    z = pd.to_numeric(row.get("z", np.nan), errors="coerce")
    if pd.notna(z):
        pref_prefix = f"z{float(z):.3f}_{sdss}"
        preferred = [p for p in candidates if p.stem.startswith(pref_prefix)]
        if preferred:
            return preferred[0]

    return candidates[0]


def _format_fraction(v) -> str:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x) or not math.isfinite(float(x)):
        return "NA"
    return f"{float(x):.4g}"


def make_stamp_text(row: pd.Series) -> str:
    z = _format_fraction(row.get('z'))
    sdss_name = str(row.get("sdss_name", "unknown")).strip()
    object_id = str(row.get("object_id", "unknown")).strip()
    m2500 = _format_fraction(row.get("apparent_mag_2500"))
    pl_slope = _format_fraction(row.get("PL_slope"))
    host = _format_fraction(row.get("f_host_center", ""))
    bc = _format_fraction(row.get("f_bc_3000", ""))
    iron = _format_fraction(row.get("f_fe_uv_3000", ""))
    host_psf_2500 = _format_fraction(row.get("frac_host_psf_2500", ""))
    psf_parts = [
        f"f_PL_psf_{band}={_format_fraction(row.get(f'f_PL_psf_{band}', ''))}"
        for band in ["u", "g", "r", "i", "z"]
    ]
    lines = [
        f"z={z} | sdss_name={sdss_name} | object_id={object_id}",
        f"m2500={m2500} | pl_slope={pl_slope} | host={host} | bc={bc} | iron={iron}",
        f"f_host_psf_2500={host_psf_2500}",
        " | ".join(psf_parts),
    ]
    return "\n".join(lines)


def _make_overlay(page_width, page_height, text, margin_pts, font_size):
    buf = io.BytesIO()
    try:
        pdfmetrics.registerFont(TTFont("DejaVuSans", "DejaVuSans.ttf"))
        font_name = "DejaVuSans"
    except Exception:
        font_name = "Helvetica"

    c = canvas.Canvas(buf, pagesize=(page_width, page_height))
    c.setFont(font_name, font_size)

    x = margin_pts
    lines = text.splitlines() or [text]
    line_height = font_size * 1.2
    text_width = max(c.stringWidth(line, font_name, font_size) for line in lines)
    text_height = line_height * len(lines)
    y = margin_pts + text_height - line_height

    pad = 1.0
    c.setFillGray(1.0)
    c.rect(
        x - pad,
        margin_pts - pad,
        text_width + 2 * pad,
        text_height + 2 * pad,
        fill=1,
        stroke=0,
    )
    c.setFillGray(0.0)
    for line in lines:
        c.drawString(x, y, line)
        y -= line_height

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
    pw, ph = float(box.width), float(box.height)
    if pw <= 0 or ph <= 0:
        raise ValueError(f"Invalid page size in {src_pdf}")

    overlay_buf = _make_overlay(pw, ph, text, float(margin_mm) * mm, font_size)
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
    if not fig_index:
        print(f"[WARN] No PDFs indexed under {args.fig_dir}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    tasks = []
    missing_pre = 0
    missing_sdss = []
    for idx, (_, row) in enumerate(df.iterrows()):
        src_pdf = find_pdf_for_row(row, fig_index)
        if src_pdf is None:
            missing_pre += 1
            missing_sdss.append(str(row["sdss_name"]))
        tasks.append((
            idx,
            str(src_pdf) if src_pdf is not None else None,
            make_stamp_text(row),
            int(args.stamp_font_size),
            float(args.stamp_margin_mm),
        ))

    if missing_pre:
        print(f"[WARN] Missing figure for {missing_pre} row(s) before stamping.", file=sys.stderr)
        preview = ", ".join(missing_sdss[:10])
        if preview:
            print(f"[WARN] First missing sdss_name values: {preview}", file=sys.stderr)

    stamped_pages = [None] * len(tasks)
    skipped_missing = 0
    stamp_failures = 0

    workers = max(1, int(args.workers))
    with mp.Pool(processes=workers) as pool:
        for idx, stamped_bytes, err in tqdm(
            pool.imap_unordered(_worker_stamp, tasks, chunksize=16),
            total=len(tasks),
            desc="Stamping pages",
        ):
            if stamped_bytes is None:
                if err == "missing":
                    skipped_missing += 1
                else:
                    stamp_failures += 1
            stamped_pages[idx] = stamped_bytes

    kept = sum(1 for x in stamped_pages if x is not None)
    print(
        f"[INFO] Rows={len(df)} kept={kept} skipped={len(df) - kept} "
        f"(missing={skipped_missing}, stamp_failures={stamp_failures})"
    )

    if args.dry_run:
        print("[DRY] Dry run complete; output PDF not written.")
        return 0

    if kept == 0:
        print("[ERROR] No output pages were produced.", file=sys.stderr)
        return 1

    writer = PdfWriter()
    for buf in stamped_pages:
        if buf is None:
            continue
        reader = PdfReader(io.BytesIO(buf))
        writer.add_page(reader.pages[0])

    with open(args.output, "wb") as f:
        writer.write(f)
    print(f"[OK] Wrote combined PDF: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
