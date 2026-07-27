"""Exhaustively screen cut profiles against a broad debiased Hubble fit.

The input residual table must come from an envelope fit whose AGN sample
contains every profile in the grid.  This script deliberately does not refit:
it evaluates the exact retained subset for every discrete profile using the
same envelope posterior, then writes a transparent ranking for follow-up
minimal-plot reruns and fresh fits.
"""

from itertools import product
import os
from pathlib import Path

import numpy as np
import pandas as pd

from qvc.hubble.hubble_utils import load_agn_data


H5_FILE = "results/data/jul22_1134pm_erlang_independentblrlags_fast_jul14chisq_spectrajul16call_n1_7fa5882_chisq.h5"
SPECTRA_CSV = "results/data/jaxqsofit_jul16c_chisqnew_all.csv"
ENVELOPE_RESIDUALS = (
    "plots/hubble/jul27cut_envelope_fastest/"
    "Flatw0waCDM_joint_fastest_all_z0p44_3p16_2d/hubble_plot_residuals.csv"
)
OUTPUT_CSV = "docs/hubble_cut_search_screen.csv"

# All profiles are subsets of the envelope fit.  The values form a finite,
# scientifically motivated grid around the current/default and historical
# relaxed cuts; continuous threshold optimization is intentionally not claimed.
GRID = {
    "blr_upper": (-0.10, -0.20, -0.30),
    "f_host_max": (0.10, 0.05, 0.02, 0.01),
    "variability_chi_sq_red_g_min": (20.0, 30.0, 50.0),
    "loo_chi2_eff_max": (None, 1.20, 1.10, 1.05, 1.01),
    "f_bc_3000_max": (0.10, 0.05, 0.02),
    "f_fe_uv_3000_max": (0.20, 0.10, 0.05),
    "rel_apparent_mag_2500_err_max": (0.0025, 0.0020, 0.0015),
}

Z_MIN, Z_MAX = 0.44, 3.16
MIN_FIT_AGN = 4000
MIN_LOW_Z_AGN = 50
MIN_HIGH_Z_AGN = 300


def _allow_missing_upper(values, upper, *, strict=False):
    values = np.asarray(values, dtype=float)
    if upper is None:
        return np.ones(values.size, dtype=bool)
    compare = values < upper if strict else values <= upper
    return ~np.isfinite(values) | compare


def _profile_mask(frame, profile):
    mask = np.ones(len(frame), dtype=bool)
    for band in "ugri":
        mask &= _allow_missing_upper(
            frame[f"dlog_amp_blr_{band}"], profile["blr_upper"], strict=True
        )
    mask &= _allow_missing_upper(frame["f_host_2500"], profile["f_host_max"])
    mask &= pd.to_numeric(
        frame["variability_chi_sq_red_g"], errors="coerce"
    ).to_numpy(dtype=float) >= profile["variability_chi_sq_red_g_min"]
    mask &= _allow_missing_upper(frame["loo_chi2_eff"], profile["loo_chi2_eff_max"])

    for column, key in (
        ("f_bc_3000", "f_bc_3000_max"),
        ("f_fe_uv_3000", "f_fe_uv_3000_max"),
    ):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        # This exactly mirrors the Hubble loader's fraction-cut semantics.
        mask &= ~np.isfinite(values) | (values <= 0.0) | (values <= profile[key])

    values = frame["rel_apparent_mag_2500_err"].to_numpy(dtype=float)
    mask &= ~np.isfinite(values) | (values < profile["rel_apparent_mag_2500_err_max"])
    return mask


def _binned_residual_metrics(z, residuals):
    bins = np.linspace(Z_MIN, Z_MAX, 10)
    indices = np.digitize(z, bins) - 1
    means = []
    for index in range(len(bins) - 1):
        values = residuals[indices == index]
        if len(values) >= 50:
            means.append(float(np.mean(values)))
    if not means:
        return np.nan, np.nan
    means = np.asarray(means, dtype=float)
    return float(np.sqrt(np.mean(means**2))), float(np.max(np.abs(means)))


def main():
    magnitude_convention = os.environ.get("QVC_HUBBLE_MAGNITUDE_CONVENTION")
    if magnitude_convention is None:
        raise RuntimeError(
            "Set QVC_HUBBLE_MAGNITUDE_CONVENTION explicitly to 'intrinsic' or "
            "'observed' before running the cut screen."
        )
    residuals_path = Path(ENVELOPE_RESIDUALS)
    if not residuals_path.exists():
        raise FileNotFoundError(f"Envelope residual table is missing: {residuals_path}")

    raw, _ = load_agn_data(
        H5_FILE,
        spectra_fit_csv=[SPECTRA_CSV],
        magnitude_convention=magnitude_convention,
        only_load=True,
        plot_diagnostics=False,
    )
    residuals = pd.read_csv(residuals_path)[
        ["object_id", "residuals", "mu_zscore"]
    ]
    raw["object_id"] = raw["object_id"].astype(str)
    residuals["object_id"] = residuals["object_id"].astype(str)
    frame = residuals.merge(raw, on="object_id", how="left", validate="one_to_one")
    if len(frame) != len(residuals):
        raise RuntimeError("Could not recover the source fields for every envelope residual row.")
    if frame["dlog_amp_blr_u"].isna().all():
        raise RuntimeError("The source-field merge failed; required BLR columns are absent.")

    mag = pd.to_numeric(frame["apparent_mag_2500"], errors="coerce").to_numpy(dtype=float)
    mag_err = pd.to_numeric(frame["apparent_mag_2500_err"], errors="coerce").to_numpy(dtype=float)
    frame["rel_apparent_mag_2500_err"] = np.divide(
        mag_err,
        np.maximum(np.abs(mag), 1e-8),
        out=np.full_like(mag_err, np.nan),
        where=np.isfinite(mag) & np.isfinite(mag_err),
    )

    residual = pd.to_numeric(frame["residuals"], errors="coerce").to_numpy(dtype=float)
    zscore = pd.to_numeric(frame["mu_zscore"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(frame["z"], errors="coerce").to_numpy(dtype=float)
    fit_range = np.isfinite(residual) & np.isfinite(zscore) & (z >= Z_MIN) & (z <= Z_MAX)

    rows = []
    names = tuple(GRID)
    for values in product(*(GRID[name] for name in names)):
        profile = dict(zip(names, values))
        selected = _profile_mask(frame, profile)
        in_fit = selected & fit_range
        n_fit = int(np.count_nonzero(in_fit))
        n_low = int(np.count_nonzero(in_fit & (z < 0.7)))
        n_high = int(np.count_nonzero(in_fit & (z > 2.5)))
        eligible = n_fit >= MIN_FIT_AGN and n_low >= MIN_LOW_Z_AGN and n_high >= MIN_HIGH_Z_AGN
        row = {
            **profile,
            "n_plot": int(np.count_nonzero(selected)),
            "n_fit": n_fit,
            "n_low_z_0p44_0p7": n_low,
            "n_high_z_2p5_3p16": n_high,
            "eligible": eligible,
        }
        if n_fit:
            selected_residual = residual[in_fit]
            row["residual_mean"] = float(np.mean(selected_residual))
            row["residual_rms"] = float(np.sqrt(np.mean(selected_residual**2)))
            row["residual_std"] = float(np.std(selected_residual))
            row["zscore_rms"] = float(np.sqrt(np.mean(zscore[in_fit] ** 2)))
            row["binned_mean_rms"], row["max_abs_binned_mean"] = _binned_residual_metrics(
                z[in_fit], selected_residual
            )
        rows.append(row)

    output = pd.DataFrame(rows)
    output["rank"] = np.nan
    eligible_index = output.index[output["eligible"]]
    ranked = output.loc[eligible_index].sort_values(
        ["residual_rms", "binned_mean_rms", "zscore_rms", "n_fit"],
        ascending=[True, True, True, False],
        kind="stable",
    )
    output.loc[ranked.index, "rank"] = np.arange(1, len(ranked) + 1)
    output = output.sort_values(["rank", "residual_rms"], na_position="last").reset_index(drop=True)

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_CSV, index=False)
    print(f"Screened {len(output)} profiles; {len(ranked)} meet the coverage floor.")
    print(output.head(20).to_string(index=False))
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
