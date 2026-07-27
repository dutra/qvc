#!/usr/bin/env python
"""Validate multiband Lomb--Scargle recovery of the GP variability amplitude.

The cadence and reported errors are taken from a real QVC object.  Each
realization is a shared latent DRW observed in all requested bands with the
same wavelength-amplitude law used by QVC.  The script compares the current
display-floor LS estimator with a direct fit to positive, noise-subtracted
periodogram bins and with the latter corrected for the PSD normalization
convention.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize

from qvc.light_curve.fit_light_curves import (
    bending_power_law_psd,
    fit_bending_power_law_psd,
)
from qvc.light_curve.multiband_fit_plotting import (
    combined_raw_band_lomb_scargle,
    fit_bending_power_law_to_display_points,
)
from validate_erlang_lag_recovery import load_real_cadence


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter_object_id", default="1452887")
    parser.add_argument("--bands", nargs="+", default=["u", "g", "r", "i"])
    parser.add_argument("--n-realizations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--tau-drw", type=float, default=300.0,
                        help="Observer-frame DRW damping time [days].")
    parser.add_argument("--eta-sigma", type=float, default=-0.2)
    parser.add_argument("--sigma-min", type=float, default=0.025)
    parser.add_argument("--sigma-max", type=float, default=0.8)
    parser.add_argument("--n-freq", type=int, default=500)
    parser.add_argument("--n-noise-sim", type=int, default=200)
    parser.add_argument("--output", type=Path,
                        default=Path("plots/multiband/ls_sigma_validation_1452887_ugri.png"))
    parser.add_argument("--results-csv", type=Path, default=None)
    return parser.parse_args()


def simulate_multiband_drw(rng, times, bands, errors, lam_rf, sigma_uv, tau, eta_sigma):
    """Draw one shared DRW and apply the QVC wavelength-amplitude law."""
    cov = np.exp(-np.abs(times[:, None] - times[None, :]) / tau)
    cov.flat[:: cov.shape[0] + 1] += 1e-10
    latent = rng.multivariate_normal(np.zeros(times.size), cov)
    band_scale = np.power(lam_rf / 2500.0, eta_sigma)
    signal = sigma_uv * band_scale[bands] * latent
    return signal + rng.normal(0.0, errors)


def current_display_fit(freq, power, lo, hi, sigma_uv, tau):
    """Reproduce the positive plotting-floor transformation used in production."""
    finite = np.isfinite(freq) & np.isfinite(power) & np.isfinite(lo) & np.isfinite(hi)
    display = np.asarray(power, float).copy()
    model = bending_power_law_psd(freq, np.log10(sigma_uv), np.log10(tau), alpha_high=-2.0)
    floor = np.maximum(1.35 * 2e-2, 0.35 * model)
    below = finite & (display <= 0.0)
    crossing = finite & (display > 0.0) & (lo <= 0.0)
    lower_span = np.clip(display - lo, 1e-300, None)
    crossing_fraction = np.clip(-lo / lower_span, 0.0, 1.0)
    shrink = np.clip(0.25 + 1.75 * crossing_fraction, 0.0, 1.0)
    display[crossing] = ((1.0 - shrink[crossing]) * display[crossing]
                         + shrink[crossing] * floor[crossing])
    display[below] = floor[below]
    good = finite & (display > 0.0) & (freq >= 8e-6) & (freq <= 2e-3)
    err_lo = np.clip(display - np.clip(lo, 1e-300, None), 0.0, None)
    err_hi = np.clip(hi - display, 0.0, None)
    err_lo[below] = 0.0
    return fit_bending_power_law_to_display_points(
        freq[good], display[good], err_lo[good], err_hi[good]
    )


def direct_fit(freq, power, lo, hi):
    good = (np.isfinite(freq) & np.isfinite(power) & np.isfinite(lo) & np.isfinite(hi)
            & (power > 0.0) & (freq >= 8e-6) & (freq <= 2e-3))
    return fit_bending_power_law_psd(freq[good], power[good], lo[good], hi[good])


def free_tau_drw_fit(freq, power, lo, hi):
    """Fit sigma and tau with the low/high DRW slopes fixed to 0 and -2."""
    good = (np.isfinite(freq) & np.isfinite(power) & np.isfinite(lo) & np.isfinite(hi)
            & (power > 0.0) & (freq >= 8e-6) & (freq <= 2e-3))
    if np.count_nonzero(good) < 4:
        return {"sigma": np.nan, "sigma_err": np.nan, "tau": np.nan, "valid": False}
    f, p = freq[good], power[good]
    logp = np.log10(p)
    safe_lo = np.clip(lo[good], 1e-300, None)
    elo = np.where((lo[good] > 0.0) & (lo[good] < p), logp - np.log10(safe_lo), np.nan)
    ehi = np.where(hi[good] > p, np.log10(hi[good]) - logp, np.nan)
    err = np.nanmean(np.column_stack((elo, ehi)), axis=1)
    err = np.where(np.isfinite(err), np.clip(err, 0.05, 0.60), 0.25)

    def model_log10(frequency, log_sigma, log_tau):
        return np.log10(bending_power_law_psd(
            frequency, log_sigma, log_tau, alpha_high=-2.0
        ))

    tau_init = 1.0 / (2.0 * np.pi * np.exp(np.mean(np.log(f))))
    sigma_init = np.sqrt(np.median(p) / max(2.0 * tau_init, 1e-30))
    try:
        popt, pcov = curve_fit(
            model_log10, f, logp,
            p0=(np.log10(max(sigma_init, 1e-6)), np.log10(tau_init)),
            sigma=err, absolute_sigma=True,
            bounds=([-6.0, -1.0], [3.0, 6.0]), maxfev=20000,
        )
        perr = np.sqrt(np.diag(pcov))
    except Exception:
        return {"sigma": np.nan, "sigma_err": np.nan, "tau": np.nan, "valid": False}
    tau = 10.0 ** popt[1]
    tau_min = np.min(1.0 / (2.0 * np.pi * f))
    tau_max = np.max(1.0 / (2.0 * np.pi * f))
    valid = bool(np.all(np.isfinite(popt)) and tau_min < tau < tau_max)
    sigma = 10.0 ** popt[0]
    return {
        "sigma": sigma if valid else np.nan,
        "sigma_err": np.log(10.0) * sigma * perr[0] if valid else np.nan,
        "tau": tau if valid else np.nan,
        "valid": valid,
    }


def gamma_noise_drw_fit(freq, raw_power, noise_power, counts):
    """Fit raw binned power as Gamma-distributed signal plus known noise."""
    good = (np.isfinite(freq) & np.isfinite(raw_power) & np.isfinite(noise_power)
            & np.isfinite(counts) & (freq >= 8e-6) & (freq <= 2e-3)
            & (raw_power > 0.0) & (noise_power >= 0.0) & (counts > 0))
    if np.count_nonzero(good) < 4:
        return {"sigma": np.nan, "tau": np.nan, "valid": False}
    f, observed, noise = freq[good], raw_power[good], noise_power[good]
    # Frequencies and bands within a bin are correlated. Counts still provide
    # useful relative weights, but cap them to avoid claiming excessive precision.
    shape = np.clip(np.asarray(counts[good], dtype=float), 1.0, 20.0)

    def nll(theta):
        signal = bending_power_law_psd(f, theta[0], theta[1], alpha_high=-2.0)
        mean = np.clip(signal + noise, 1e-300, None)
        return float(np.sum(shape * (np.log(mean) + observed / mean)))

    tau_init = 1.0 / (2.0 * np.pi * np.exp(np.mean(np.log(f))))
    excess = np.maximum(np.median(observed - noise), 1e-10)
    sigma_init = np.sqrt(excess / max(2.0 * tau_init, 1e-30))
    result = minimize(
        nll,
        x0=np.array([np.log10(max(sigma_init, 1e-6)), np.log10(tau_init)]),
        method="L-BFGS-B",
        bounds=((-6.0, 3.0), (-1.0, 6.0)),
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return {"sigma": np.nan, "tau": np.nan, "valid": False}
    tau = 10.0 ** result.x[1]
    tau_min = np.min(1.0 / (2.0 * np.pi * f))
    tau_max = np.max(1.0 / (2.0 * np.pi * f))
    valid = bool(tau_min < tau < tau_max)
    return {
        "sigma": 10.0 ** result.x[0] if valid else np.nan,
        "tau": tau if valid else np.nan,
        "valid": valid,
    }


def main():
    args = parse_args()
    times, bands, errors, band_names, redshift = load_real_cadence(
        args.filter_object_id, args.bands
    )
    times, bands, errors = map(np.asarray, (times, bands, errors))
    pivot = {"u": 3551.0, "g": 4686.0, "r": 6165.0, "i": 7481.0, "z": 8931.0}
    lam_rf = np.asarray([pivot[name] / (1.0 + redshift) for name in band_names])
    ref_idx = int(np.argmin(np.abs(lam_rf - 2500.0)))
    freqs = np.logspace(-6, 2, args.n_freq)
    params = {"eta_sigma": args.eta_sigma}
    rng = np.random.default_rng(args.seed)
    true_sigma = np.exp(rng.uniform(np.log(args.sigma_min), np.log(args.sigma_max),
                                    args.n_realizations))
    rows = []
    for index, sigma_uv in enumerate(true_sigma):
        y = simulate_multiband_drw(
            rng, times, bands, errors, lam_rf, sigma_uv, args.tau_drw, args.eta_sigma
        )
        fbin, raw_power, power, lo, hi, counts, noise_power = combined_raw_band_lomb_scargle(
            (times, bands), y, errors, params, 2.0 * np.pi * freqs,
            ref_band_idx=ref_idx, bins_per_decade=3, min_per_bin=5,
            band_wavelength_rf=lam_rf, n_noise_sim=args.n_noise_sim,
            random_state=args.seed + index,
        )
        display = current_display_fit(fbin, power, lo, hi, sigma_uv, args.tau_drw)
        direct = direct_fit(fbin, power, lo, hi)
        display_valid = display is not None and display["valid"]
        direct_valid = direct["psd_bpl_valid"]
        display_log = display["log_sigma"] if display_valid else np.nan
        display_err = display["log_sigma_err"] if display_valid else np.nan
        direct_log = direct["log_sigma_bpl"] if direct_valid else np.nan
        direct_err = direct["log_sigma_bpl_err"] if direct_valid else np.nan
        drw_fit = free_tau_drw_fit(fbin, power, lo, hi)
        gamma_fit = gamma_noise_drw_fit(fbin, raw_power, noise_power, counts)
        rows.append({
            "realization": index, "sigma_uv": sigma_uv,
            "sigma_ls_display": 10.0 ** display_log,
            "sigma_ls_display_err": np.log(10.0) * 10.0 ** display_log * display_err,
            "sigma_ls_direct": 10.0 ** direct_log,
            "sigma_ls_direct_err": np.log(10.0) * 10.0 ** direct_log * direct_err,
            "sigma_ls_direct_norm_corrected": 10.0 ** direct_log / np.sqrt(2.0),
            "sigma_ls_direct_norm_corrected_err": (
                np.log(10.0) * 10.0 ** direct_log * direct_err / np.sqrt(2.0)
            ),
            "sigma_ls_fixed_slopes": drw_fit["sigma"] / np.sqrt(2.0),
            "sigma_ls_fixed_slopes_err": drw_fit["sigma_err"] / np.sqrt(2.0),
            "tau_ls_fixed_slopes": drw_fit["tau"],
            "sigma_ls_gamma_noise": gamma_fit["sigma"] / np.sqrt(2.0),
            "sigma_ls_gamma_noise_err": np.nan,
            "tau_ls_gamma_noise": gamma_fit["tau"],
        })
        print(f"[{index + 1:02d}/{args.n_realizations}] sigma_UV={sigma_uv:.4f} "
              f"display={rows[-1]['sigma_ls_display']:.4f} "
              f"direct={rows[-1]['sigma_ls_direct']:.4f} "
              f"fixed-slopes={rows[-1]['sigma_ls_fixed_slopes']:.4f} "
              f"gamma-noise={rows[-1]['sigma_ls_gamma_noise']:.4f}")

    output_csv = args.results_csv or args.output.with_suffix(".csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    methods = [
        ("sigma_ls_display", "sigma_ls_display_err", "display-floor fit"),
        ("sigma_ls_direct", "sigma_ls_direct_err", "direct positive-bin fit"),
        ("sigma_ls_fixed_slopes", "sigma_ls_fixed_slopes_err",
         r"fixed slopes, free $\tau$, / $\sqrt{2}$"),
        ("sigma_ls_gamma_noise", "sigma_ls_gamma_noise_err",
         r"Gamma raw-power + noise"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.1), sharex=True, sharey=True)
    bounds = (args.sigma_min * 0.7, args.sigma_max * 1.8)
    for ax, (key, err_key, label) in zip(axes, methods, strict=True):
        x = np.asarray([row["sigma_uv"] for row in rows])
        y = np.asarray([row[key] for row in rows])
        yerr = np.asarray([row[err_key] for row in rows])
        good = np.isfinite(x) & np.isfinite(y)
        plot_err = np.where(np.isfinite(yerr[good]), yerr[good], 0.0)
        ax.errorbar(x[good], y[good], yerr=plot_err, fmt="o", ms=3.5,
                    color="k", ecolor="0.75", elinewidth=0.7, alpha=0.75)
        ax.plot(bounds, bounds, color="m", lw=2)
        ax.set(xscale="log", yscale="log", xlim=bounds, ylim=bounds,
               xlabel=r"true $\sigma_{\rm UV}$ (mag)")
        ax.text(0.05, 0.94, label, transform=ax.transAxes, va="top")
        bias = np.nanmedian(np.log10(y[good] / x[good]))
        scatter = 1.4826 * np.nanmedian(np.abs(np.log10(y[good] / x[good]) - bias))
        ax.text(0.95, 0.05, f"bias = {bias:+.2f} dex\nscatter = {scatter:.2f} dex",
                transform=ax.transAxes, ha="right", va="bottom")
    axes[0].set_ylabel(r"measured $\sigma_{\rm LS}$ (mag)")
    fig.tight_layout()
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"Saved {args.output}")
    print(f"Saved {output_csv}")


if __name__ == "__main__":
    main()
