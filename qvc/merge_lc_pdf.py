#!/usr/bin/env python3
"""
Concatenate light-curve PNGs into a single PDF with optional overlays and filtering.

Image path template (per row):
    plots/multiband/{prefix}/light_curve_fits/{z:.1f}_{object_id}_light_curve_{image_id}.png

Requirements:
    - Python 3.8+
    - pandas, pillow

Example:
    python make_lightcurve_pdf.py \
        --csv results/data/objects.csv \
        --prefix oct14a \
        --output plots/multiband/oct14a/light_curve_fits/lightcurves_oct14a.pdf \
        --overlay z,sdss_name,redchi2_conti_full \
        --key z --low 0.5 --high 2.0 \
        --max 100
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Concat light-curve PNGs into a PDF with overlays.")
    p.add_argument("--csv", required=True, help="Input CSV containing at least object_id and z.")
    p.add_argument("--prefix", required=True, help="Prefix used in the plots directory.")
    p.add_argument("--output", required=False, default=None,
                   help="Output PDF path. Defaults to plots/multiband/{prefix}/light_curve_fits/lightcurves_{prefix}.pdf")
    p.add_argument("--object-id-col", default="object_id", help="Column name for object id (default: object_id).")
    p.add_argument("--z-col", default="z", help="Column name for redshift z (default: z).")
    p.add_argument("--overlay", default="redchi2_conti_full,m2500_residuals",
                   help="Comma-separated list of CSV fields to overlay on images (e.g., 'z,sdss_name,redchi2_conti_full').")
    p.add_argument("--key", default=None, help="Numeric CSV column to filter by (inclusive).")
    p.add_argument("--low", type=float, default=None, help="Low bound for --key (inclusive).")
    p.add_argument("--high", type=float, default=None, help="High bound for --key (inclusive).")
    p.add_argument("--max", type=int, default=None, help="Maximum number of images to include.")
    p.add_argument("--font-size", type=int, default=24, help="Overlay font size (default: 24).")
    p.add_argument("--box-alpha", type=float, default=0.8, help="Overlay box opacity in [0,1] (default: 0.6).")
    p.add_argument("--box-pad", type=int, default=8, help="Padding for overlay box (default: 8).")
    p.add_argument("--skip-missing", action="store_true",
                   help="Skip rows whose PNG is missing (instead of failing).")
    p.add_argument("--sort-by", default=None, help="Column name to sort the CSV by (default: None).")
    return p.parse_args()


def build_image_path(prefix: str, z_val: float, object_id: str) -> Optional[Path]:
    """
    Find the first PNG whose filename contains the object_id.
    Preference order:
      1) '{z:.1f}_*{object_id}*.png'
      2) '*{object_id}*.png'
    Returns None if no file is found.
    """
    base = Path("plots") / "multiband" / prefix / "light_curves_fits"
    z_str = f"{float(z_val):.1f}"

    # Prefer exact z-prefixed matches if present
    preferred = sorted(base.glob(f"{z_str}_*{object_id}*.png"))
    if preferred:
        return preferred[0]

    # Fallback: any file containing object_id
    generic = sorted(base.glob(f"*{object_id}*.png"))
    return generic[0] if generic else None


def format_overlay_lines(row: pd.Series, object_id_col: str, z_col: str, fields: list) -> list:
    lines = [f"object_id: {row[object_id_col]}"]
    # Include z (rounded) if present, even if not in fields
    if z_col in row and pd.notna(row[z_col]):
        try:
            lines.append(f"{z_col}: {float(row[z_col]):.3f}")
        except Exception:
            lines.append(f"{z_col}: {row[z_col]}")
    for f in fields:
        if f == "":
            continue
        if f not in row:
            continue
        val = row[f]
        if pd.isna(val):
            continue
        # Pretty format floats
        if isinstance(val, float):
            # Short float representation
            lines.append(f"{f}: {val:.5g}")
        else:
            lines.append(f"{f}: {val}")
    return lines


def draw_overlay(im: Image.Image, lines: list, font_size: int = 24, box_alpha: float = 0.6, box_pad: int = 8) -> Image.Image:
    if not lines:
        return im

    # Ensure RGBA for alpha compositing
    if im.mode != "RGBA":
        im_rgba = im.convert("RGBA")
    else:
        im_rgba = im.copy()

    draw = ImageDraw.Draw(im_rgba)

    # Try default PIL font; fallback sizing
    try:
        font = ImageFont.truetype(font="DejaVuSansMono.ttf", size=font_size)  # none uses default
    except Exception:
        print("[WARN] Failed to load truetype font; using default.", file=sys.stderr)
        font = ImageFont.load_default()

    # Measure text block
    line_w = 0
    line_h_total = 0
    line_heights = []
    for line in lines:
        w, h = draw.textbbox((0, 0), line, font=font)[2:]
        line_w = max(line_w, w)
        line_heights.append(h)
        line_h_total += h

    box_w = line_w + 2 * box_pad
    box_h = line_h_total + (len(lines) - 1) * 2 + 2 * box_pad  # small inter-line gap

    # Position top-left
    x0, y0 = box_pad, box_pad
    x1, y1 = x0 + box_w, y0 + box_h

    # Semi-transparent rectangle
    overlay = Image.new("RGBA", im_rgba.size, (0, 0, 0, 0))
    rect = Image.new("RGBA", (int(box_w), int(box_h)), (0, 0, 0, int(255 * box_alpha)))
    overlay.paste(rect, (int(x0), int(y0)))
    im_rgba = Image.alpha_composite(im_rgba, overlay)
    draw = ImageDraw.Draw(im_rgba)

    # Draw text (white)
    y = y0 + box_pad
    for i, line in enumerate(lines):
        draw.text((x0 + box_pad, y), line, fill=(255, 255, 255, 255), font=font)
        y += line_heights[i] + 2

    return im_rgba


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        return 2

    df = pd.read_csv(csv_path)
    for col in (args.object_id_col, args.z_col):
        if col not in df.columns:
            print(f"[ERROR] Missing required column '{col}' in CSV.", file=sys.stderr)
            return 2

    # Filter by numeric key if requested
    if args.key is not None:  # hyphen isn't valid in Python attribute; fix below
        pass
    # Work around argparse name with hyphen:
    key = getattr(args, "key")
    if key:
        if key not in df.columns:
            print(f"[ERROR] --key '{key}' not found in CSV columns.", file=sys.stderr)
            return 2
        if args.low is not None:
            df = df[df[key] >= args.low]
        if args.high is not None:
            df = df[df[key] <= args.high]


    df = df.sort_values(by=args.sort_by, ascending=False)

    if df.empty:
        print("[WARN] No rows after filtering. Nothing to do.", file=sys.stderr)
        return 1

    # Derive default output if needed
    out_pdf = Path(args.output)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    overlay_fields = [s.strip() for s in args.overlay.split(",")] if args.overlay else []

    images_for_pdf = []
    missing = 0
    taken = 0

    # Iterate rows in CSV order
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Build path to PNG
        try:
            obj_id = str(int(row[args.object_id_col]))
            z_val = float(row[args.z_col])
        except Exception:
            print(f"[WARN] Row {idx}: invalid {args.object_id_col} or {args.z_col}; skipping.", file=sys.stderr)
            continue

        png_path = build_image_path(args.prefix, z_val, obj_id)
        if png_path is None:
            missing += 1
            msg = f"[WARN] No PNG found for object_id={obj_id} (z≈{z_val:.1f})"
            if args.skip_missing:
                print(msg, file=sys.stderr)
                continue
            else:
                print(msg, file=sys.stderr)
                return 2
            
        # Open, overlay, and store
        try:
            im = Image.open(png_path)
        except Exception as e:
            print(f"[WARN] Failed to open {png_path}: {e}", file=sys.stderr)
            if args.skip_missing:
                continue
            else:
                return 2

        lines = format_overlay_lines(row, args.object_id_col, args.z_col, overlay_fields)
        im = draw_overlay(im, lines, font_size=args.font_size, box_alpha=args.box_alpha, box_pad=args.box_pad)

        # Convert to RGB for PDF
        im_rgb = im.convert("RGB")
        if not images_for_pdf:
            images_for_pdf.append(im_rgb)
        else:
            images_for_pdf.append(im_rgb)

        taken += 1
        if args.max is not None and taken >= args.max:
            break

    if not images_for_pdf:
        print("[WARN] No images to save. Exiting.", file=sys.stderr)
        return 1

    # Save multi-page PDF
    first = images_for_pdf[0]
    rest = images_for_pdf[1:]
    try:
        first.save(out_pdf, "PDF", resolution=150.0, save_all=True, append_images=rest)
    except Exception as e:
        print(f"[ERROR] Failed to write PDF {out_pdf}: {e}", file=sys.stderr)
        return 2

    print(f"[OK] Wrote {len(images_for_pdf)} page(s) to: {out_pdf}")
    if missing:
        print(f"[NOTE] Missing PNGs skipped: {missing}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
