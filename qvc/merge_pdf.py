#!/usr/bin/env python3
"""
Parallel QSOFit plot combiner (multiprocessing.Pool).

- all-per-object: one worker per object builds a full grid page and returns its bytes.
- best-only: one worker per BEST row stamps a single page and returns its bytes.

Parent process assembles pages deterministically.

Requires:
  pip install pypdf reportlab pandas tqdm
"""

from __future__ import annotations
import os
import argparse
import io
import sys
import math
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm
import zipfile
import traceback

# ---- pypdf ----
try:
    from pypdf import PdfReader, PdfWriter, Transformation
    from pypdf.generic import IndirectObject, ArrayObject
except ImportError:
    print("ERROR: Please install 'pypdf' (pip install pypdf)", file=sys.stderr)
    sys.exit(1)

# ---- reportlab ----
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
except ImportError:
    print("ERROR: Please install 'reportlab' (pip install reportlab)", file=sys.stderr)
    sys.exit(1)


# ----------------------------- CLI -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Build a combined PDF of QSOFit plots (parallel).")
    p.add_argument("csv", type=Path, help="Input CSV with all runs (must include a 'best' column).")
    p.add_argument("output", type=Path, help="Output PDF path.")
    p.add_argument("--mode", choices=["all-per-object", "best-only"], required=True,
                   help="Select layout mode.")
    p.add_argument("--root", type=Path, default=Path("plots/pyqsofit"),
                   help="Root directory to search for PDFs (recursive). Default: plots/pyqsofit")
    p.add_argument("--prefix", type=str, default="pyqsofit",
                   help="Prefix under root (e.g. plots/pyqsofit/<prefix>/...). Default: pyqsofit")
    p.add_argument("--cols", type=int, default=2,
                   help="[all-per-object] Number of columns per grid page.")
    p.add_argument("--max-per-object", type=int, default=None,
                   help="[all-per-object] Optional cap on number of fits to place on a single object page.")
    p.add_argument("--stamp-font-size", type=int, default=10, help="Font size for overlay stamp.")
    p.add_argument("--stamp-margin-mm", type=float, default=2.0, help="Margin for stamp box.")
    p.add_argument("--dry-run", action="store_true", help="Only report what would be done; don't write PDF.")
    p.add_argument("--debug", action="store_true", help="Verbose debugging.")
    p.add_argument("--N", type=int, default=None,
                   help="Optional: Only process the first N rows/objects.")
    p.add_argument("--seed", type=int, default=42,
                   help="[all-per-object] Randomization seed for object ordering.")
    p.add_argument("--workers", type=int, default=max(28, (os.cpu_count() or 2) - 1),
                   help="Number of worker processes. Default: CPU-1")
    p.add_argument("--key", type=str, default=None, help="Column name to use as a key for filtering or sorting.")
    p.add_argument("--high", type=float, default=None, help="High threshold for the key column (inclusive).")
    p.add_argument("--low", type=float, default=None, help="Low threshold for the key column (inclusive).")
    p.add_argument("--order", choices=["asc", "desc"], default="asc",
                   help="Order of sorting by z_obj (asc or desc). Default: asc")
    p.add_argument("--filter-csv", type=Path, default=None,
                   help="Optional CSV file containing 'object_id' column to filter which objects to include.")
    return p.parse_args()


# ----------------------------- CSV Loading -----------------------------
REQUIRED_COLS = {
    "object_id", "sdss_name", "z",
    "npca_qso", "decomp_host", "BC", "poly",
    "apparent_mag_2500_err", "redchi2_conti_full",
    "best"
}

def load_csv(csv_path):
    df = pd.read_csv(csv_path)

    # allow z recovery
    if "z" not in df.columns and "Z_SYS" in df.columns:
        df["z"] = df["Z_SYS"]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # normalize types a bit
    df["object_id"] = df["object_id"].astype(str)
    df["sdss_name"] = df["sdss_name"].astype(str)
    # Coerce booleans for consistency
    for b in ["decomp_host", "BC", "poly", "best"]:
        if df[b].dtype != bool:
            df[b] = df[b].astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])
    return df


# ----------------------------- Helpers -----------------------------
def _page_contents_len(page):
    try:
        contents = page.get("/Contents", None)
        if contents is None:
            return 0
        if isinstance(contents, IndirectObject):
            contents = contents.get_object()
        if hasattr(contents, "get_data"):
            d = contents.get_data()
            return len(d) if d is not None else 0
        if isinstance(contents, ArrayObject):
            total = 0
            for obj in contents:
                if isinstance(obj, IndirectObject):
                    obj = obj.get_object()
                if hasattr(obj, "get_data"):
                    d = obj.get_data()
                    total += (len(d) if d is not None else 0)
            return total
    except Exception:
        pass
    return -1

def _bool_token(v):
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    s = str(v).strip().lower()
    return "True" if s in ("1", "true", "t", "yes", "y") else "False"

def find_pdf_for_row(row, root, prefix, debug=False):
    # Accept str or Path for robustness (works for both parent and workers)
    root = Path(root)                 # <— NEW
    search_root = (root / prefix) if prefix else root
    if not search_root.exists():
        if debug:
            print(f"[DEBUG] Search root does not exist: {search_root}", file=sys.stderr)
        return None

    sdss = str(row["sdss_name"])
    z = float(row["z"]) if pd.notna(row["z"]) else np.nan
    preferred = f"{z:.2f}_{sdss}_*.pdf" if np.isfinite(z) else None
    fallback  = f"*{sdss}*.pdf"

    npca = int(row["npca_qso"])
    decomp = _bool_token(row["decomp_host"])
    BCtok  = _bool_token(row["BC"])
    poly = _bool_token(row["poly"])


    cfg_dir = search_root / f"npca_qso_{npca}_decomp_host_{decomp}_BC_{BCtok}_poly_{poly}"
    best_dir = search_root / "best"

    search_dirs = []
    if cfg_dir.exists():
        search_dirs.append(cfg_dir)
    if bool(row.get("best", False)) and best_dir.exists():
        search_dirs.append(best_dir)

    for d in search_dirs:
        if preferred:
            hits = sorted(d.glob(preferred))
            if hits:
                return hits[0]
        hits = sorted(d.glob(fallback))
        if hits:
            return hits[0]

    if preferred:
        hits = sorted(search_root.rglob(preferred))
        for p in hits:
            if p.is_file() and (cfg_dir in p.parents or best_dir in p.parents):
                return p
        if hits:
            return hits[0]

    hits = sorted(search_root.rglob(fallback))
    if hits:
        return hits[0]

    return None


def _make_overlay(page_width, page_height, text, margin_pts, font_size):
    buf = io.BytesIO()
    try:
        pdfmetrics.registerFont(TTFont("DejaVuSans", "DejaVuSans.ttf"))
        font_name = "DejaVuSans"
    except Exception:
        font_name = "Helvetica"

    c = canvas.Canvas(buf, pagesize=(page_width, page_height))
    c.setFont(font_name, font_size)

    text_width = c.stringWidth(text, font_name, font_size)
    x = margin_pts
    y = margin_pts

    pad = 1.0
    c.setFillGray(1.0)
    c.rect(x - pad, y - pad, text_width + 2*pad, font_size + 2*pad, fill=1, stroke=0)
    c.setFillGray(0.0)
    c.drawString(x, y, text)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf

def format_stamp(row, star_best):
    parts = []
    if star_best:
        parts.append("★ best")

    def b2i(x):
        if isinstance(x, (bool, np.bool_)):
            return 1 if x else 0
        try:
            return 1 if str(x).lower() in ("1", "true", "t", "yes", "y") else 0
        except Exception:
            return 0

    parts.append(f"BC={b2i(row.get('BC', False))}")
    parts.append(f"decomp_host={b2i(row.get('decomp_host', False))}")

    npca = row.get("npca_qso", None)
    parts.append(f"npca_qso={int(npca) if pd.notna(npca) else 'NA'}")

    poly = row.get("poly", None)
    parts.append(f"poly={int(poly) if pd.notna(poly) else 'NA'}")

    ame = row.get("apparent_mag_2500_err", np.nan)
    try:
        parts.append(f"apparent_mag_2500_err={float(ame):.3f}")
    except Exception:
        parts.append("apparent_mag_2500_err=?")

    ame = row.get("apparent_mag_2500", np.nan)
    try:
        parts.append(f"apparent_mag_2500={float(ame):.2f}")
    except Exception:
        parts.append("apparent_mag_2500=?")

    try:
        log_lbol = row.get("log_lbol", np.nan)
        if pd.notna(log_lbol):
            parts.append(f"log_lbol={float(log_lbol):.1f}")
    except Exception:
        pass


    try:
        fhost = row.get("f_host_2500", np.nan)
        if pd.notna(fhost):
            parts.append(f"fhost={float(fhost):.5f}")
    except Exception:
        pass

    chi = row.get("redchi2_conti_full", np.nan)
    try:
        parts.append(f"redchi2_conti_full={float(chi):.3f}")
    except Exception:
        parts.append("redchi2_conti_full=?")
    try:
        parts.append(f"resid={float(row['resid']):.1f}")
    except Exception:
        parts.append("resid=?")

    return " | ".join(parts)

def stamp_pdf(src_pdf, text, font_size, margin_mm, debug=False):
    reader = PdfReader(str(src_pdf))
    assert len(reader.pages) >= 1, f"PDF has no pages: {src_pdf}"

    src_len0 = _page_contents_len(reader.pages[0])
    assert src_len0 != 0, f"Source PDF appears to have empty /Contents: {src_pdf}"

    writer = PdfWriter()
    margin_pts = float(margin_mm) * mm

    for page in reader.pages:
        box = getattr(page, "cropbox", None) or page.mediabox
        pw, ph = float(box.width), float(box.height)
        assert pw > 0 and ph > 0, f"Zero-sized page in {src_pdf}"

        overlay_buf = _make_overlay(pw, ph, text, margin_pts, font_size)
        overlay = PdfReader(overlay_buf).pages[0]
        page.merge_page(overlay)

        merged_len = _page_contents_len(page)
        assert merged_len != 0, f"Merged stamped page has empty /Contents for {src_pdf}"

        writer.add_page(page)

    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    # quick read check
    PdfReader(out)
    return out


def _blank_reader_page(width, height):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(width, height))
    c.setFillGray(1.0)
    c.rect(0, 0, width, height, fill=1, stroke=0)
    c.showPage()
    c.save()
    buf.seek(0)
    return PdfReader(buf)

def compose_grid_to_bytes(paged_buffers, cols, debug=False):
    if not paged_buffers:
        raise ValueError("No pages to compose in grid.")

    r0 = PdfReader(paged_buffers[0])
    p0 = r0.pages[0]
    cw = float((getattr(p0, "cropbox", None) or p0.mediabox).width)
    ch = float((getattr(p0, "cropbox", None) or p0.mediabox).height)
    assert cw > 0 and ch > 0, "Cell page has zero size"

    n = len(paged_buffers)
    cols = max(1, int(cols))
    rows = math.ceil(n / cols)
    W, H = cols * cw, rows * ch

    book = PdfWriter()
    base_page = book.add_blank_page(width=W, height=H)

    bg_reader = _blank_reader_page(W, H)
    base_page.merge_page(bg_reader.pages[0])

    # place top->bottom, left->right
    for i, buf in enumerate(paged_buffers):
        rr = PdfReader(buf)
        src = rr.pages[0]
        sbox = getattr(src, "cropbox", None) or src.mediabox
        sw, sh = float(sbox.width), float(sbox.height)
        assert sw > 0 and sh > 0, "Stamped tile has zero size"

        r = rows - 1 - (i // cols)
        c = i % cols
        tx = c * cw
        ty = r * ch
        sx = cw / sw
        sy = ch / sh

        T = Transformation().scale(sx, sy).translate(tx, ty)
        base_page.merge_transformed_page(src, T, expand=True)

    out = io.BytesIO()
    book.write(out)
    out.seek(0)
    PdfReader(out)  # validate
    return out


# ----------------------------- Worker Tasks -----------------------------
def _worker_best_only(task):
    """Stamp one BEST row -> return (index, bytes) or (index, None) on miss."""
    (idx, row_dict, root, prefix, font_size, margin_mm) = task
    row = pd.Series(row_dict)
    pdfp = find_pdf_for_row(row, root, prefix, debug=False)
    if pdfp is None:
        return (idx, None)
    text = format_stamp(row, star_best=True)
    try:
        stamped = stamp_pdf(pdfp, text, font_size=font_size, margin_mm=margin_mm, debug=False)
        return (idx, stamped.getvalue())
    except Exception:
        return (idx, None)


def _worker_all_per_object(task):
    """
    Process one object: sort its rows, find+stamp up to max_per_object, compose grid page.
    Return (order_key, object_id, bytes or None).
    """
    (order_key, object_id, rows_dicts, root, prefix, cols, max_per_object, font_size, margin_mm) = task

    sub = pd.DataFrame(rows_dicts).copy()
    sub["__dr"] = (sub["redchi2_conti_full"] - 1.0).abs()
    sort_keys = [k for k in ["best", "__dr", "aic", "bic"] if k in sub.columns]
    sort_asc  = [False, True, True, True][:len(sort_keys)]
    if sort_keys:
        sub = sub.sort_values(by=sort_keys, ascending=sort_asc)

    stamped_pages = []
    for _, row in sub.iterrows():
        if max_per_object is not None and len(stamped_pages) >= max_per_object:
            break
        pdfp = find_pdf_for_row(row, root, prefix, debug=False)
        if pdfp is None:
            continue
        text = format_stamp(row, star_best=bool(row.get("best", False)))
        try:
            stamped = stamp_pdf(pdfp, text, font_size=font_size, margin_mm=margin_mm, debug=False)
            stamped_pages.append(stamped)
        except Exception:
            continue

    if not stamped_pages:
        return (order_key, object_id, None)

    try:
        composed = compose_grid_to_bytes(stamped_pages, cols=cols, debug=False)
        return (order_key, object_id, composed.getvalue())
    except Exception:
        return (order_key, object_id, None)

def zip_file(args):
    # Optionally, create a zip file containing the output PDF with maximum compression
    zip_path = args.output.with_suffix('.zip')
    try:
        if zip_path.exists():
            if args.debug:
                print(f"[INFO] Overwriting existing zip: {zip_path}", file=sys.stderr)
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
            zipf.write(args.output, arcname=args.output.name)
        print(f"[OK] Wrote zip archive → {zip_path}")
    except Exception as e:
        print(f"[WARN] Failed to write zip {zip_path}: {e}", file=sys.stderr)

# ----------------------------- Main -----------------------------
def main():
    args = parse_args()
    df = load_csv(args.csv)

    
        # --- Filter object_id using outlier_agn.csv ---
    if args.filter_csv is not None:
        outlier_csv = Path(args.filter_csv)
        try:
            df_outlier = pd.read_csv(outlier_csv)
        except Exception as e:
            print(f"[WARN] Failed to read {outlier_csv}: {e}; no filtering applied.", file=sys.stderr)
        else:
            if 'object_id' not in df_outlier.columns:
                print(f"[WARN] 'object_id' column not found in {outlier_csv}; no filtering/merge applied.", file=sys.stderr)
            else:
                # Normalize keys to string for a clean join
                df['object_id'] = df['object_id'].astype(str)
                df_outlier['object_id'] = df_outlier['object_id'].astype(str)

                # Optional: keep only objects present in the outlier file
                outlier_ids = set(df_outlier['object_id'])
                before = df['object_id'].nunique()
                df = df[df['object_id'].isin(outlier_ids)].copy()
                after = df['object_id'].nunique()
                print(f"[INFO] Kept {after}/{before} objects present in {outlier_csv}.", file=sys.stderr)

                # Merge in log_lbol if available
                if 'log_lbol' in df_outlier.columns:
                    df = df.merge(
                        df_outlier[['object_id', 'log_lbol']].drop_duplicates('object_id'),
                        on='object_id', how='left', suffixes=('', '_outlier')
                    )

                    # If df already had log_lbol, prefer the outlier value when present
                    if 'log_lbol_outlier' in df.columns:
                        if 'log_lbol' in df.columns:
                            # prefer outlier value; fall back to existing df value
                            df['log_lbol'] = df['log_lbol_outlier'].combine_first(df['log_lbol'])
                        else:
                            # df did not have log_lbol originally; just rename
                            df = df.rename(columns={'log_lbol_outlier': 'log_lbol'})
                        df = df.drop(columns=['log_lbol_outlier'])
                else:
                    print(f"[WARN] 'log_lbol' column not found in {outlier_csv}; not merged.", file=sys.stderr)
                    traceback.print_exc()

    #df = df[df['sdss_name'].isin(['010012.97+000021.1', '021605.09-001019.0'])]

    # --- Optional filtering by key/low/high ---
    if args.key is not None:
        if args.low is not None:
            df = df[df[args.key] >= args.low]
        if args.high is not None:
            df = df[df[args.key] <= args.high]
    # --- Order by args.key according to args.order ---
    if args.key is not None:
        ascending = args.order == "asc"
        df = df.sort_values(args.key, ascending=ascending, kind="mergesort")


    # (Optional but robust) ensure 'best' is boolean even if CSV has strings like "True"/"False"
    if df["best"].dtype != bool:
        df["best"] = (
            df["best"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
            .fillna(False)
        )

    # # --- Select objects by conditions applied to their BEST row only ---
    # cond_best = df["best"] & (df["f_host_2500"] > 0.5) & (df["z"] > 2)
    # eligible_objids = set(df.loc[cond_best, "object_id"].unique())

    # if args.debug:
    #     total_objs = df["object_id"].nunique()
    #     elig_best_rows = cond_best.sum()
    #     print(f"[DEBUG] Loaded CSV: {args.csv}  rows={len(df)}  unique objects={total_objs}")
    #     print(f"[DEBUG] Eligible BEST rows (f_host_2500>0.5 & z>2): {elig_best_rows}")
    #     print(f"[DEBUG] Eligible objects: {len(eligible_objids)}")
    #     print(f"[DEBUG] Searching under: {args.root}  prefix={args.prefix}")

    # # Keep ALL rows for those eligible objects (so we can show *all fits*)
    # df = df[df["object_id"].isin(eligible_objids)].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("[DRY] Not writing output.")

    # ----- best-only: parallel per row -----
    if args.mode == "best-only":
        df_best = df[df["best"]].copy()
        if df_best.empty:
            print("No rows with best==True. Nothing to do.", file=sys.stderr)
            return 1
        if args.N is not None:
            df_best = df_best.iloc[:args.N]

        tasks = []
        for i, (_, row) in enumerate(df_best.iterrows()):
            tasks.append((
                i,                              # index to preserve order
                row.to_dict(),
                args.root,
                args.prefix,
                args.stamp_font_size,
                args.stamp_margin_mm
            ))

        pages_in_order = [None] * len(tasks)
        with mp.Pool(processes=args.workers) as pool:
            for idx, buf in tqdm(pool.imap_unordered(_worker_best_only, tasks, chunksize=8),
                                 total=len(tasks), desc="Stamping BEST rows"):
                pages_in_order[idx] = buf

        if not args.dry_run:
            book = PdfWriter()
            kept = 0
            for buf in pages_in_order:
                if buf is None:
                    continue
                r = PdfReader(io.BytesIO(buf))
                book.add_page(r.pages[0])
                kept += 1
            if kept == 0:
                print("No stamped pages were produced.", file=sys.stderr)
                return 1
            print(f"[INFO] Writing combined PDF to {args.output} ...")
            with open(args.output, "wb") as f:
                book.write(f)
            print(f"[OK] Wrote combined PDF → {args.output}")
            zip_file(args)
        return 0

    # ----- all-per-object: parallel per object -----
    # Build representative z per object for deterministic ordering
    z_by_obj = (
        df.groupby("object_id", sort=False)["z"]
        .apply(lambda s: np.nanmedian(s.values) if len(s) else np.nan)
        .rename("z_obj")
        .reset_index()
    )

    # shuffle reproducibly, then optional head(N), then sort by z (NaN last)
    z_by_obj = z_by_obj.sample(frac=1.0, random_state=args.seed)
    if args.N is not None:
        z_by_obj = z_by_obj.head(min(args.N, len(z_by_obj)))
    z_by_obj = z_by_obj.sort_values("z_obj", kind="mergesort")
    ordered_ids = z_by_obj["object_id"].tolist()

    # Prepare per-object rows now to avoid interprocess pandas slicing work
    rows_by_obj = {oid: df[df["object_id"] == oid].copy() for oid in ordered_ids}

    tasks = []
    for order_key, oid in enumerate(ordered_ids):
        tasks.append((
            order_key,
            oid,
            rows_by_obj[oid].to_dict(orient="records"),
            str(args.root),
            args.prefix,
            args.cols,
            args.max_per_object,
            args.stamp_font_size,
            args.stamp_margin_mm
        ))

    results = [None] * len(tasks)
    with mp.Pool(processes=args.workers) as pool:
        for order_key, oid, buf in tqdm(pool.imap_unordered(_worker_all_per_object, tasks, chunksize=1),
                                        total=len(tasks), desc="Building object pages"):
            results[order_key] = (oid, buf)

    if args.dry_run:
        return 0

    book = PdfWriter()
    kept_pages = 0
    for oid, buf in results:
        if buf is None:
            continue
        r = PdfReader(io.BytesIO(buf))
        # composed page is a single page, add as-is
        book.add_page(r.pages[0])
        kept_pages += 1

    if kept_pages == 0:
        print("No object pages were produced.", file=sys.stderr)
        return 1

    print(f"[INFO] Writing combined PDF to {args.output} ...")
    with open(args.output, "wb") as f:
        book.write(f)
    print(f"[OK] Wrote combined PDF → {args.output}")

    zip_file(args)
    return 0


if __name__ == "__main__":
    # On macOS/Linux this works out of the box.
    # On Windows, Pool requires the __main__ guard (present).
    raise SystemExit(main())
