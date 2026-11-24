#!/usr/bin/env python3
"""Build a fixed reference-flux table for the PSF constant-flux subtraction.

The light-curve fitter scales the constant flux it removes from each band
from a reference total flux; by default that is the inverse-variance-weighted
mean flux of the epochs it is given. When an object is fitted on a subset of
its epochs, ``fit_light_curves --constant_flux_reference_path`` can instead
use the reference of the full light curve, written by this tool.

The table is built with the fitter's own path: ``concat_light_curves`` (same
outlier rejection and SDSS source), ``populate_sdss_fields``, the optional
rest-frame window cut, and ``reference_total_flux_from_band`` on each band,
so a table built from the full light curve reproduces the default
subtraction of the full light curve exactly. Pass the same loader options
here as to the fit, including ``--zero_point_flag_table`` /
``--zero_point_flag_threshold``: the reference is the full light curve as the
standard fit sees it, so when the fit drops the epochs of non-photometric
nights, the reference must be built without them too. The epoch tables are read through ``QVC_DATA_DIR`` as in
the fitter, so point it at the full tables when building the reference.

Usage::

    python -m qvc.light_curve.build_constant_flux_reference \\
        --filter_object_id 1467268 1436799 --output reference_flux.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qvc.light_curve.multiband_generate_lc import (
    concat_light_curves,
    cut_light_curve_restframe_window,
    populate_sdss_fields,
)
from qvc.light_curve.zero_point_flags import (
    DEFAULT_ZERO_POINT_FLAG_THRESHOLD,
    file_sha256,
    load_zero_point_cells,
)
from qvc.light_curve.psf_constant_flux_correction import (
    normalize_object_id,
    reference_total_flux_from_band,
)

REFERENCE_FLOAT_FORMAT = "%.17g"  # round-trips float64 exactly


def reference_rows_for_objects(objs, *, reference_stat: str = "mean") -> pd.DataFrame:
    """Return one row per object and band with the epochs' reference total flux."""

    rows = []
    for obj in objs:
        oid = normalize_object_id(obj.get("object_id"))
        for band in obj["mags"]:
            mags = np.asarray(obj["mags"][band], dtype=float)
            magerrs = np.asarray(obj["magerrs"][band], dtype=float)
            rows.append(
                {
                    "object_id": oid,
                    "band": str(band),
                    "reference_total_flux": reference_total_flux_from_band(
                        mags, magerrs, reference_stat=reference_stat
                    ),
                    "n_epochs": int(np.sum(np.isfinite(mags) & np.isfinite(magerrs))),
                    "reference_stat": reference_stat,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["object_id", "band", "reference_total_flux", "n_epochs", "reference_stat"],
    )


def build_constant_flux_reference_table(
    object_ids,
    *,
    outlier_half_window_days: float = 30.0,
    sdss_source: str = "dr16s82",
    rf_length_cut: int = -1,
    exact_same_length: bool = False,
    progress_bar: bool = False,
    zero_point_cells=None,
    zero_point_flag_threshold: float = DEFAULT_ZERO_POINT_FLAG_THRESHOLD,
) -> pd.DataFrame:
    """Load objects exactly as the fitter does and return their reference table.

    ``zero_point_cells`` / ``zero_point_flag_threshold``: the fitter's
    non-photometric-night exclusion (off when ``None``), passed to
    ``concat_light_curves`` as the fitter passes them.
    """

    zero_point_kwargs = (
        {}
        if zero_point_cells is None
        else dict(zero_point_cells=zero_point_cells, zero_point_flag_threshold=zero_point_flag_threshold)
    )
    objs = concat_light_curves(
        filter_object_ids=list(object_ids),
        progress_bar=progress_bar,
        outlier_half_window_days=outlier_half_window_days,
        sdss_source=sdss_source,
        **zero_point_kwargs,
    )
    objs = populate_sdss_fields(objs, progress_bar=progress_bar)
    if rf_length_cut > 0:
        objs = cut_light_curve_restframe_window(
            objs, n_days=rf_length_cut, same_length=exact_same_length
        )
    table = reference_rows_for_objects(objs)
    found = set(table["object_id"])
    missing = [normalize_object_id(o) for o in object_ids if normalize_object_id(o) not in found]
    if missing:
        raise ValueError(f"No light curve loaded for object_id(s): {', '.join(missing)}")
    return table


def write_constant_flux_reference_table(table: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format=REFERENCE_FLOAT_FORMAT)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--filter_object_id", nargs="+", default=None, help="Object IDs.")
    parser.add_argument("--filter_file", default=None, help="Text file with one object ID per line.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--outlier_half_window_days", type=float, default=30.0,
                        help="Same as the fitter's option (default: 30).")
    parser.add_argument("--sdss_source", choices=("dr16s82", "macleod"), default="dr16s82",
                        help="Same as the fitter's option (default: dr16s82).")
    parser.add_argument("--rf_length_cut", type=int, default=-1,
                        help="Same as the fitter's option (default: -1, no cut).")
    parser.add_argument("--exact_same_length", action="store_true",
                        help="Same as the fitter's option.")
    parser.add_argument("--zero_point_flag_table", default=None,
                        help="Same as the fitter's option (default: off).")
    parser.add_argument("--zero_point_flag_threshold", type=float, default=None,
                        help="Same as the fitter's option (default 0.1; requires --zero_point_flag_table).")
    parser.add_argument("--progress", action="store_true", help="Show progress bars.")
    args = parser.parse_args(argv)
    zero_point_cells = None
    if args.zero_point_flag_table is None:
        if args.zero_point_flag_threshold is not None:
            parser.error("--zero_point_flag_threshold requires --zero_point_flag_table")
    else:
        if args.zero_point_flag_threshold is None:
            args.zero_point_flag_threshold = DEFAULT_ZERO_POINT_FLAG_THRESHOLD
        if not np.isfinite(args.zero_point_flag_threshold) or args.zero_point_flag_threshold <= 0:
            parser.error("--zero_point_flag_threshold must be a positive magnitude")
        try:
            zero_point_cells = load_zero_point_cells(args.zero_point_flag_table)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(
            f"Zero-point epoch flags from {args.zero_point_flag_table} "
            f"(sha256 {file_sha256(args.zero_point_flag_table)}, "
            f"threshold {args.zero_point_flag_threshold:g} mag)."
        )

    object_ids = list(args.filter_object_id or [])
    if args.filter_file:
        with open(args.filter_file, encoding="utf-8") as handle:
            object_ids += [line.strip() for line in handle if line.strip()]
    if not object_ids:
        parser.error("give --filter_object_id and/or --filter_file")

    table = build_constant_flux_reference_table(
        object_ids,
        outlier_half_window_days=args.outlier_half_window_days,
        sdss_source=args.sdss_source,
        rf_length_cut=args.rf_length_cut,
        exact_same_length=args.exact_same_length,
        progress_bar=args.progress,
        **(
            {}
            if zero_point_cells is None
            else dict(zero_point_cells=zero_point_cells,
                      zero_point_flag_threshold=args.zero_point_flag_threshold)
        ),
    )
    path = write_constant_flux_reference_table(table, args.output)
    print(f"Wrote {len(table)} rows for {table['object_id'].nunique()} objects to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
