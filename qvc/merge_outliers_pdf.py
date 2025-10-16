#!/usr/bin/env python3
"""
Concatenate PDFs for filtered rows based on filename patterns, and stamp each page
with per-row metadata (residual, npca_qso, z, sdss_name).

CSV must have columns (at least):
  object_id, ra, dec, mu_pred_median, mu_pred_std, z, redchi,
  sdss_name, npca_qso, residuals

For each filtered row (residuals > threshold), this looks for PDFs under:
  {plots_root}/npca_qso_{npca_qso}/
matching either:
  1) "{z:.2f}_{sdss_name}_*.pdf"  (preferred)
  2) "*{sdss_name}*.pdf"          (fallback)

Usage:
  python concat_npca_pdfs.py input.csv out.pdf
  python concat_npca_pdfs.py input.csv out.pdf --threshold 5
  python concat_npca_pdfs.py input.csv out.pdf --plots-root plots/pyqsofit
  python concat_npca_pdfs.py input.csv out.pdf --dry-run

Notes:
- Requires 'pandas', 'PyPDF2', and 'reportlab'.
- Deduplicates files across rows while preserving first-seen order.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Iterable, List, Set, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from PyPDF2 import PdfMerger, PdfReader, PdfWriter
except ImportError:
    print("ERROR: Please install 'PyPDF2'.\n  pip install PyPDF2", file=sys.stderr)
    sys.exit(1)

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
except ImportError:
    print("ERROR: Please install 'reportlab'.\n  pip install reportlab", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Concatenate and annotate PDFs for high-residual rows.")
    p.add_argument("csv", type=Path, help="Input CSV path.")
    p.add_argument("output", type=Path, help="Output concatenated PDF path.")
    p.add_argument("--low", type=float, default=0.0, 
                     help="Minimum absolute value of the key column to include (default: 0.0).")
    p.add_argument("--high", type=float, default=np.inf,
                     help="Maximum absolute value of the key column to include (default: no upper limit).")
    #p.add_argument("--threshold", type=float, default=3.0,
    p.add_argument("--plots-root", type=Path, default=Path("plots/pyqsofit"),
                   help="Root folder where npca_qso_* folders live (default: plots/pyqsofit).")
    p.add_argument("--recursive", action="store_true",
                   help="Search subdirectories recursively inside each npca_qso_* folder.")
    p.add_argument("--dry-run", action="store_true",
                   help="Only list matched files; do not write the output PDF.")
    p.add_argument("--sort-by", choices=["csv", "desc", "asc"], default="desc",
                   help="Order rows before matching files (default: residuals_desc order).")
    p.add_argument("--stamp-pos", choices=["top-right", "top-left", "bottom-left", "bottom-right"],
                   default="top-right", help="Where to place the annotation text on each page.")
    p.add_argument("--stamp-margin-mm", type=float, default=8.0,
                   help="Margin from edges for the stamp (in mm). Default: 8.")
    p.add_argument("--stamp-font-size", type=int, default=9,
                   help="Font size for the stamp text. Default: 9.")
    p.add_argument("--key", type=str, default="residuals",
                   help="Column name to use for filtering and sorting (default: residuals).")
    p.add_argument("--N", type=int, default=None,
                   help="Only process the first N matched PDFs (default: all).")
    return p.parse_args()


def load_and_filter(csv_path: Path, low: float, high: float, sort_by: str, key: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "z" not in df.columns and "Z_SYS" in df.columns:
        df["z"] = df["Z_SYS"]
    required_cols = {"sdss_name", "npca_qso", "z", key}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    mask = np.ones(len(df), dtype=bool)
    mask = (low <= np.abs(df[key])) & (np.abs(df[key]) < high)
    #mask = df[key].between(0, 1)
    #mask = df['sdss_name'] == '230022.05+004300.2'
    df_f = df.loc[mask].copy()

    if sort_by == "desc":
        df_f = df_f.reindex(df_f[key].abs().sort_values(ascending=False).index)
    elif sort_by == "asc":
        df_f = df_f.reindex(df_f[key].abs().sort_values(ascending=True).index)
    # else keep CSV order
    return df_f


def iter_pdf_matches(row: pd.Series, plots_root: Path, recursive: bool) -> Iterable[Path]:
    """
    Yield PDF matches for a single CSV row.

    Preferred pattern: "{z:.2f}_{sdss_name}_*.pdf"
    Fallback pattern: "*{sdss_name}*.pdf"
    """
    npca = int(row["npca_qso"])
    z = float(row["z"])
    sdss_name = str(row["sdss_name"])

    base_dir = plots_root / f"npca_qso_{npca}"
    if not base_dir.exists():
        print(f"WARNING: Directory does not exist: {base_dir}", file=sys.stderr)
        return

    preferred = f"{z:.2f}_{sdss_name}_*.pdf"
    fallback = f"*{sdss_name}*.pdf"

    if recursive:
        pref_hits = sorted(base_dir.rglob(preferred))
    else:
        pref_hits = sorted(base_dir.glob(preferred))

    if pref_hits:
        for p in pref_hits:
            if p.is_file():
                yield p
        return

    if recursive:
        fb_hits = sorted(base_dir.rglob(fallback))
    else:
        fb_hits = sorted(base_dir.glob(fallback))

    for p in fb_hits:
        if p.is_file():
            yield p


def dedup_preserve_order(records: Iterable[Tuple[Path, float, int, float, str, str, float, float]]) -> \
    List[Tuple[Path, float, int, float, str, str, float]]:
    """Deduplicate by path; preserve first occurrence’s metadata."""
    seen: Set[Path] = set()
    out: List[Tuple[Path, float, int, float, str, str, float]] = []
    for rec in records:
        p = rec[0]
        if p not in seen:
            seen.add(p)
            out.append(rec)
    return out


def _make_overlay(page_width: float,
                  page_height: float,
                  text: str,
                  pos: str,
                  margin_pts: float,
                  font_size: int) -> io.BytesIO:
    """
    Create a single-page PDF overlay with the provided text at a chosen corner.
    Dimensions in points. Returns an in-memory PDF buffer.
    """
    buf = io.BytesIO()

    # Register a simple font (try to ensure it exists).
    # Falling back to built-in Helvetica if TrueType registration fails.
    try:
        pdfmetrics.registerFont(TTFont("DejaVuSans", "DejaVuSans.ttf"))
        font_name = "DejaVuSans"
    except Exception:
        font_name = "Helvetica"

    c = canvas.Canvas(buf, pagesize=(page_width, page_height))
    c.setFont(font_name, font_size)

    # Measure text width to align right-side placements neatly
    text_width = c.stringWidth(text, font_name, font_size)

    if pos == "top-right":
        x = page_width - margin_pts - text_width
        y = page_height - margin_pts - font_size
    elif pos == "top-left":
        x = margin_pts
        y = page_height - margin_pts - font_size
    elif pos == "bottom-left":
        x = margin_pts
        y = margin_pts
    else:  # bottom-right
        x = page_width - margin_pts - text_width
        y = margin_pts

    # Draw a subtle white "knockout" box behind text for legibility
    pad = 1
    c.setFillGray(1.0)
    c.rect(x - pad, y - pad, text_width + 2 * pad, font_size + 2 * pad, fill=1, stroke=0)

    # Draw text in black
    c.setFillGray(0.0)
    c.drawString(x, y, text)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf


def _stamp_pdf_bytes(src_path: Path,
                     stamp_text: str,
                     pos: str,
                     margin_mm: float,
                     font_size: int) -> io.BytesIO:
    """
    Read a PDF, stamp every page with 'stamp_text', and return stamped bytes.
    """
    reader = PdfReader(str(src_path))
    writer = PdfWriter()

    margin_pts = float(margin_mm) * mm

    for page in reader.pages:
        # Determine page size from media box
        media = page.mediabox
        pw = float(media.width)
        ph = float(media.height)

        overlay_buf = _make_overlay(
            page_width=pw,
            page_height=ph,
            text=stamp_text,
            pos=pos,
            margin_pts=margin_pts,
            font_size=font_size,
        )

        overlay_pdf = PdfReader(overlay_buf)
        overlay_page = overlay_pdf.pages[0]
        # Merge overlay onto the current page
        page.merge_page(overlay_page)
        writer.add_page(page)

    out_buf = io.BytesIO()
    writer.write(out_buf)
    out_buf.seek(0)
    return out_buf


def main() -> int:
    args = parse_args()

    try:
        df = load_and_filter(args.csv, args.low, args.high, args.sort_by, args.key)
    except Exception as e:
        print(f"ERROR reading/filtering CSV: {e}", file=sys.stderr)
        return 2

    if df.empty:
        print(f"No rows. Nothing to do.")
        return 0

    # Collect (path, residual, npca_qso, z, sdss_name, redchi) tuples in chosen row order.
    collected: List[Tuple[Path, float, int, float, str, float]] = []
    missing_rows = 0

    for _, rec in df.iterrows():
        residual = float(rec.get("residuals", np.nan))
        npca = int(rec["npca_qso"])
        z = float(rec["z"])
        sdss_name = str(rec["sdss_name"])
        key = float(rec[args.key])
        redchi = float(rec["redchi"])
        object_id = str(rec.get("object_id", "?"))
        loglbol = float(rec.get("loglbol", np.nan))
        m_2500_err = float(rec.get("apparent_mag_2500_err", np.nan))
        f_host_2500 = float(rec.get("f_host_2500", np.nan))
        print(f"Row: sdss_name={sdss_name}  npca_qso={npca}  z={z:.3f}  {args.key}={key:.3f}  redchi={redchi:.2f} object_id:{object_id}")
        hits = list(iter_pdf_matches(rec, args.plots_root, args.recursive))
        if not hits:
            missing_rows += 1
        for p in hits:
            collected.append((p, residual, npca, z, sdss_name, redchi, object_id, loglbol, m_2500_err, f_host_2500))

    merged_list = dedup_preserve_order(collected)
    if args.N is not None:
        merged_list = merged_list[:args.N]

    # Report
    print(f"Filtered rows: {len(df)}  |  Rows with no matches: {missing_rows}")
    print(f"Unique PDFs matched: {len(merged_list)}")
    for p, r, n, z, s, redchi, object_id, loglbol, m_2500_err, f_host_2500 in merged_list:
        print(f"  + residual={r:.2f}  | npca_qso={n}  z={z:.2f}  SDSS={s} redchi={redchi:.3f} object_id:{object_id} loglbol={loglbol}")

    if args.dry_run:
        print("\nDry-run: not writing output.")
        return 0

    if not merged_list:
        print("No PDFs matched; aborting without writing an empty file.", file=sys.stderr)
        return 1

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Stamp each matched PDF, then append stamped bytes to merger
    try:
        merger = PdfMerger()
        for pdf_path, residual, npca, z, sdss, redchi, object_id, loglbol, m2500_err, f_host_2500 in tqdm(merged_list, desc="Stamping PDFs"):
            stamp_text = f"residual={residual:.1f}  |  npca_qso={npca} | redchi={redchi:.1f} |  z={z:.1f}  |  SDSS={sdss} | object_id:{object_id} | loglbol={loglbol:.1f} | m2500_err={m2500_err:.1f} |f_host_2500={f_host_2500:.1f}"
            stamped = _stamp_pdf_bytes(
            pdf_path,
            stamp_text=stamp_text,
            pos=args.stamp_pos,
            margin_mm=args.stamp_margin_mm,
            font_size=args.stamp_font_size,
            )
            merger.append(stamped)  # fileobj (BytesIO) is accepted
        with open(args.output, "wb") as f:
            merger.write(f)
        merger.close()
    except Exception as e:
        print(f"ERROR while stamping/merging PDFs: {e}", file=sys.stderr)
        return 3

    print(f"\nWrote merged PDF: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
