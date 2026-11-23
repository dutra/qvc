"""Compare Tier-1 chi-square cuts on the stored and the reference statistic.

The stored chi-square columns use the fit's likelihood variance; the reference
(``*_ref``) columns use a fixed variance (see
:data:`qvc.hubble.cuts.REFERENCE_CHI2_CUT_COLUMNS`). Switching a cut from one
statistic to the other is meant to be assessed on data, one cut at a time.
This module counts, at the same threshold, how many objects pass each cut on
either statistic, and for the whole Tier-1 selection how many pass with all
cuts on the stored statistic, all on the reference one, and the configuration
actually used. It also reports the rank correlation of the two statistics and
the reference threshold that keeps the stored pass fraction.

Command line::

    python -m qvc.hubble.chi2_statistic_comparison CATALOG.h5 --output-dir DIR

``CATALOG`` is a spectral catalogue (HDF5 with a ``catalog`` group, or CSV).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qvc.hubble.cuts import (
    JAXSEDFIT_JOINT_REDUCED_CHI2_MAX,
    REFERENCE_CHI2_CUT_COLUMNS,
    SED_REDUCED_CHI2_MAX,
    SPECTROSCOPY_REDUCED_CHI2_MAX,
)

CUT_COLUMNS = tuple(REFERENCE_CHI2_CUT_COLUMNS)
DEFAULT_THRESHOLDS = {
    "sed_reduced_chi2": SED_REDUCED_CHI2_MAX,
    "spectroscopy_reduced_chi2": SPECTROSCOPY_REDUCED_CHI2_MAX,
    "joint_reduced_chi2": JAXSEDFIT_JOINT_REDUCED_CHI2_MAX,
}
DEFAULT_Z_EDGES = (0.0, 1.0, 1.5, 2.0, 2.5, 7.0)
SNR_COLUMN_CANDIDATES = ("spectrum_snr_median", "SN_MEDIAN_ALL")
# Same grouping rule as the per-spectrograph completeness maps.
SPECTROGRAPH_MJD_BOUNDARY = 55000.0
CUT_COMPARISON_COLUMNS = (
    ("cut", "cut"),
    ("threshold", "threshold"),
    ("n_both_finite", "N (both finite)"),
    ("pass_stored", "pass stored"),
    ("pass_reference", "pass ref"),
    ("pass_both", "pass both"),
    ("only_stored", "only stored"),
    ("only_reference", "only ref"),
    ("spearman", "rank corr"),
    ("matched_reference_threshold", "ref thr @ stored pass frac"),
)
SELECTION_COMPARISON_COLUMNS = (
    ("configuration", "Tier-1 selection"),
    ("chi2_columns", "chi-square columns"),
    ("before", "before"),
    ("kept", "kept"),
)


def _values(frame, column):
    if column not in frame.columns:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _passes(values, lower, upper):
    mask = np.isfinite(values)
    if lower is not None:
        mask &= values >= lower
    if upper is not None:
        mask &= values <= upper
    return mask


def spearman(a, b):
    """Spearman rank correlation of the pairs where both are finite."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(finite) < 3:
        return np.nan
    ra = pd.Series(a[finite]).rank().to_numpy()
    rb = pd.Series(b[finite]).rank().to_numpy()
    if np.std(ra) == 0.0 or np.std(rb) == 0.0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def matched_reference_threshold(stored, reference, threshold):
    """Reference threshold that passes the same fraction as ``threshold`` on stored.

    Over the objects where both are finite, ``k`` pass ``stored <= threshold``;
    the returned value is the ``k``-th smallest reference value, so the same
    number ``k`` passes ``reference <= value`` (ties aside). NaN when ``k`` is 0.
    """
    stored = np.asarray(stored, dtype=float)
    reference = np.asarray(reference, dtype=float)
    finite = np.isfinite(stored) & np.isfinite(reference)
    k = int(np.count_nonzero(stored[finite] <= threshold))
    if k == 0:
        return np.nan
    return float(np.sort(reference[finite])[k - 1])


def compare_cut_statistics(frame, thresholds=None):
    """Per-cut pass counts on the stored and the reference statistic.

    Only cuts whose stored and reference columns are both present are
    reported. Counts are over all rows; a non-finite value fails the cut, as
    in the Hubble selection.
    """
    thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    rows = []
    for column in CUT_COLUMNS:
        reference_column = REFERENCE_CHI2_CUT_COLUMNS[column]
        if column not in frame.columns or reference_column not in frame.columns:
            continue
        threshold = thresholds.get(column)
        if threshold is None:
            continue
        stored = _values(frame, column)
        reference = _values(frame, reference_column)
        pass_stored = _passes(stored, None, threshold)
        pass_reference = _passes(reference, None, threshold)
        rows.append(
            {
                "cut": column,
                "threshold": float(threshold),
                "n": int(len(frame)),
                "n_both_finite": int(np.count_nonzero(np.isfinite(stored) & np.isfinite(reference))),
                "pass_stored": int(np.count_nonzero(pass_stored)),
                "pass_reference": int(np.count_nonzero(pass_reference)),
                "pass_both": int(np.count_nonzero(pass_stored & pass_reference)),
                "only_stored": int(np.count_nonzero(pass_stored & ~pass_reference)),
                "only_reference": int(np.count_nonzero(~pass_stored & pass_reference)),
                "spearman": spearman(stored, reference),
                "matched_reference_threshold": matched_reference_threshold(
                    stored, reference, threshold
                ),
            }
        )
    return rows


def _selection_mask(frame, cuts):
    mask = np.ones(len(frame), dtype=bool)
    for column, lower, upper in cuts:
        mask &= _passes(_values(frame, column), lower, upper)
    return mask


def compare_selection(frame, tier1_cuts, used_cuts=None):
    """Count objects passing every Tier-1 cut under each chi-square choice.

    ``tier1_cuts`` are the configured cuts with stored column names. Rows:
    all chi-square cuts on the stored statistic, all on the reference one
    (only when every reference column exists), and ``used_cuts`` if given.
    Non-chi-square cuts are applied identically in every row.
    """
    tier1_cuts = list(tier1_cuts)
    configurations = [("all stored", tier1_cuts)]
    chi2_in_cuts = [c for c, _lo, _hi in tier1_cuts if c in REFERENCE_CHI2_CUT_COLUMNS]
    if chi2_in_cuts and all(REFERENCE_CHI2_CUT_COLUMNS[c] in frame.columns for c in chi2_in_cuts):
        configurations.append(
            (
                "all reference",
                [(REFERENCE_CHI2_CUT_COLUMNS.get(c, c), lo, hi) for c, lo, hi in tier1_cuts],
            )
        )
    if used_cuts is not None:
        configurations.append(("used", list(used_cuts)))
    chi2_names = set(REFERENCE_CHI2_CUT_COLUMNS) | set(REFERENCE_CHI2_CUT_COLUMNS.values())
    rows = []
    for name, cuts in configurations:
        rows.append(
            {
                "configuration": name,
                "chi2_columns": ", ".join(c for c, _lo, _hi in cuts if c in chi2_names),
                "before": int(len(frame)),
                "kept": int(np.count_nonzero(_selection_mask(frame, cuts))),
            }
        )
    return rows


def _format_rows(rows, columns):
    from qvc.hubble.hubble_utils import render_ascii_table

    formatted = []
    for row in rows:
        out = {}
        for key, _label in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = "nan" if not np.isfinite(value) else f"{value:.4g}"
            out[key] = value
        formatted.append(out)
    return render_ascii_table(formatted, columns)


def render_comparison(cut_rows, selection_rows=None):
    """Return the comparison as plain-text tables."""
    parts = []
    if cut_rows:
        parts.append(
            "Tier-1 chi-square: stored vs reference statistic at the same threshold\n"
            + _format_rows(cut_rows, CUT_COMPARISON_COLUMNS)
        )
    if selection_rows:
        parts.append(
            "Tier-1 selection under each chi-square choice\n"
            + _format_rows(selection_rows, SELECTION_COMPARISON_COLUMNS)
        )
    return "\n\n".join(parts)


def spectrograph_groups(frame):
    """Spectrograph label per row: ``spectrograph`` column, else from ``mjd``."""
    if "spectrograph" in frame.columns:
        return frame["spectrograph"].astype(str).str.strip().str.lower().to_numpy()
    if "mjd" in frame.columns:
        mjd = _values(frame, "mjd")
        return np.where(
            np.isfinite(mjd), np.where(mjd < SPECTROGRAPH_MJD_BOUNDARY, "sdss", "boss"), "unknown"
        )
    return None


def snr_terciles(frame, column):
    """Tercile label (``low``/``mid``/``high``) of a S/N column; NaN rows ``unknown``."""
    values = _values(frame, column)
    labels = np.full(len(frame), "unknown", dtype=object)
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 3:
        return labels
    lo, hi = np.quantile(values[finite], [1.0 / 3.0, 2.0 / 3.0])
    labels[finite & (values <= lo)] = "low"
    labels[finite & (values > lo) & (values <= hi)] = "mid"
    labels[finite & (values > hi)] = "high"
    return labels


def redshift_bins(frame, edges=DEFAULT_Z_EDGES):
    z = _values(frame, "z")
    labels = np.full(len(frame), "unknown", dtype=object)
    for lo, hi in zip(edges[:-1], edges[1:]):
        labels[np.isfinite(z) & (z >= lo) & (z < hi)] = f"[{lo:g}, {hi:g})"
    return labels


def split_comparison(frame, thresholds=None, *, z_edges=DEFAULT_Z_EDGES, snr_column=None):
    """Per-cut comparison for all rows and split by redshift, spectrograph and S/N."""
    splits = [("all", np.full(len(frame), "all", dtype=object))]
    splits.append(("redshift", redshift_bins(frame, z_edges)))
    spectrograph = spectrograph_groups(frame)
    if spectrograph is not None:
        splits.append(("spectrograph", spectrograph))
    if snr_column is not None and snr_column in frame.columns:
        splits.append((f"{snr_column} tercile", snr_terciles(frame, snr_column)))
    rows = []
    for split, labels in splits:
        labels = np.asarray(labels, dtype=object)
        for label in pd.unique(labels):
            subset = frame.loc[labels == label]
            for row in compare_cut_statistics(subset, thresholds):
                rows.append({"split": split, "group": str(label), **row})
    return pd.DataFrame(rows)


def read_catalog(path):
    """Read a flat spectral catalogue (HDF5 ``catalog`` group or CSV)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    import h5py

    columns = {}
    with h5py.File(path, "r") as handle:
        group = handle["catalog"] if "catalog" in handle else handle
        for name, dataset in group.items():
            if not hasattr(dataset, "shape") or len(dataset.shape) != 1:
                continue
            values = dataset[...]
            if values.dtype.kind in "SO":
                values = np.asarray(
                    [v.decode("utf-8", "replace") if isinstance(v, bytes) else v for v in values],
                    dtype=object,
                )
            columns[name] = values
    lengths = {len(v) for v in columns.values()}
    if len(lengths) > 1:
        size = max(lengths, key=lambda n: sum(len(v) == n for v in columns.values()))
        columns = {k: v for k, v in columns.items() if len(v) == size}
    return pd.DataFrame(columns)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("catalog", help="Spectral catalogue (.h5 with a 'catalog' group, or .csv).")
    parser.add_argument("--output-dir", default=None, help="Write CSV tables here.")
    parser.add_argument(
        "--info-csv",
        default=None,
        help="Optional CSV merged on object_id (for example with spectrograph or SN_MEDIAN_ALL).",
    )
    parser.add_argument(
        "--snr-column",
        default=None,
        help=f"S/N column for the tercile split (default: first of {SNR_COLUMN_CANDIDATES} present).",
    )
    parser.add_argument(
        "--z-edges",
        default=",".join(f"{edge:g}" for edge in DEFAULT_Z_EDGES),
        help="Comma-separated redshift bin edges.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    frame = read_catalog(args.catalog)
    if args.info_csv:
        info = pd.read_csv(args.info_csv)
        frame["object_id"] = frame["object_id"].astype(str).str.strip()
        info["object_id"] = info["object_id"].astype(str).str.strip()
        extra = ["object_id"] + [c for c in info.columns if c not in frame.columns]
        frame = frame.merge(info[extra], on="object_id", how="left")
    snr_column = args.snr_column or next(
        (c for c in SNR_COLUMN_CANDIDATES if c in frame.columns), None
    )
    z_edges = tuple(float(edge) for edge in args.z_edges.split(","))
    cut_rows = compare_cut_statistics(frame)
    if not cut_rows:
        raise SystemExit(
            "The catalogue has no pair of stored and reference (*_ref) chi-square columns."
        )
    print(render_comparison(cut_rows))
    split = split_comparison(frame, z_edges=z_edges, snr_column=snr_column)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print()
        print(split.drop(columns=["n"]).to_string(index=False))
    if snr_column is None:
        print("\nNo S/N column found; S/N tercile split skipped.")
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cut_rows).to_csv(out / "chi2_statistic_comparison.csv", index=False)
        split.to_csv(out / "chi2_statistic_comparison_by_split.csv", index=False)
        print(f"\nWrote {out / 'chi2_statistic_comparison.csv'} and the split table.")
    return cut_rows, split


if __name__ == "__main__":
    main()
