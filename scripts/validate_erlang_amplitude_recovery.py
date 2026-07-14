#!/usr/bin/env python
"""Validate continuum and BLR amplitude recovery for the fast Erlang GP.

Synthetic light curves use the actual cadence and photometric uncertainties of
one QVC object.  A shared continuum is present in every selected band, while a
causal Erlang BLR response is added in flux only to the selected BLR band.  The
lag and continuum timescales are fixed to their injected values during fitting,
isolating recovery of the two amplitude coordinates.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tinygp import GaussianProcess

from validate_erlang_lag_recovery import load_real_cadence
from qvc.light_curve.multiband_model_dho_blr import (
    mag_residual_to_relative_flux,
    magerr_residual_to_relative_fluxerr,
    relative_flux_to_mag_residual,
)
from qvc.light_curve.multiband_model_dho_blr_erlang import ErlangResponseDHOQS


jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter_object_id", default="1452887")
    parser.add_argument("--bands", nargs="+", default=["u", "g", "r", "i"])
    parser.add_argument("--blr-band", default="r")
    parser.add_argument("--n-realizations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--erlang-order", type=int, default=3)
    parser.add_argument("--tau-drw", type=float, default=300.0)
    parser.add_argument("--continuum-amp-min", type=float, default=0.05)
    parser.add_argument("--continuum-amp-max", type=float, default=0.30)
    parser.add_argument("--blr-fraction-min", type=float, default=0.03)
    parser.add_argument("--blr-fraction-max", type=float, default=0.30)
    parser.add_argument("--lag-rest-min", type=float, default=10.0)
    parser.add_argument("--lag-rest-max", type=float, default=1000.0)
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument(
        "--observation-domain",
        choices=("magnitude", "flux"),
        default="magnitude",
        help="Inject exact magnitudes or use a matched relative-flux control.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/multiband/erlang_amplitude_recovery_1452887_ugri.png"),
    )
    parser.add_argument("--results-csv", type=Path, default=None)
    return parser.parse_args()


def make_kernel(
    n_band,
    lag_observed,
    continuum_amp,
    blr_amp,
    *,
    blr_band_index,
    tau_fast,
    tau_drw,
    erlang_order,
):
    amp_cont = jnp.full(n_band, continuum_amp)
    amp_blr = jnp.zeros(n_band).at[blr_band_index].set(blr_amp)
    return ErlangResponseDHOQS(
        tau_fast=jnp.full(n_band, tau_fast),
        tau_slow=jnp.full(n_band, tau_drw),
        lag_blr=jnp.full(n_band, lag_observed),
        amp_cont=amp_cont,
        amp_blr=amp_blr,
        order=erlang_order,
    )


def make_amplitude_log_likelihood_fn(
    times,
    bands,
    *,
    n_band,
    blr_band_index,
    tau_fast,
    tau_drw,
    erlang_order,
):
    def one(theta, y, noise_var, lag_observed):
        continuum_amp, blr_amp = jnp.exp(theta)
        kernel = make_kernel(
            n_band,
            lag_observed,
            continuum_amp,
            blr_amp,
            blr_band_index=blr_band_index,
            tau_fast=tau_fast,
            tau_drw=tau_drw,
            erlang_order=erlang_order,
        )
        gp = GaussianProcess(
            kernel,
            (times, bands),
            diag=noise_var,
            assume_sorted=True,
        )
        return gp.log_probability(y)

    return jax.jit(jax.vmap(one, in_axes=(0, None, None, None)))


def weighted_quantiles(grid, weights, quantiles=(0.16, 0.5, 0.84)):
    order = np.argsort(grid)
    grid = np.asarray(grid, dtype=float)[order]
    weights = np.asarray(weights, dtype=float)[order]
    weights = weights / np.sum(weights)
    cdf = np.concatenate([[0.0], np.cumsum(weights)])
    values = np.concatenate([[grid[0]], grid])
    return np.interp(np.asarray(quantiles), cdf, values)


def main() -> None:
    args = parse_args()
    times, bands, errors, band_names, redshift = load_real_cadence(
        args.filter_object_id, args.bands
    )
    if args.blr_band not in band_names:
        raise ValueError(f"BLR band {args.blr_band!r} is not in {band_names}")
    n_band = len(band_names)
    blr_band_index = band_names.index(args.blr_band)
    reference_noise_var = errors**2
    magnitude_errors = errors / (0.4 * np.log(10.0))
    tau_fast = 0.5

    # The absolute BLR grid covers every product allowed by the injected
    # continuum-amplitude and BLR-fraction ranges.
    continuum_grid = np.exp(
        np.linspace(
            np.log(args.continuum_amp_min),
            np.log(args.continuum_amp_max),
            args.grid_size,
        )
    )
    blr_amp_min = args.continuum_amp_min * args.blr_fraction_min
    blr_amp_max = args.continuum_amp_max * args.blr_fraction_max
    blr_grid = np.exp(
        np.linspace(np.log(blr_amp_min), np.log(blr_amp_max), args.grid_size)
    )
    log_cont_mesh, log_blr_mesh = np.meshgrid(
        np.log(continuum_grid), np.log(blr_grid), indexing="ij"
    )
    theta_grid = np.column_stack([log_cont_mesh.ravel(), log_blr_mesh.ravel()])
    likelihood_grid = make_amplitude_log_likelihood_fn(
        times,
        bands,
        n_band=n_band,
        blr_band_index=blr_band_index,
        tau_fast=tau_fast,
        tau_drw=args.tau_drw,
        erlang_order=args.erlang_order,
    )

    rng = np.random.default_rng(args.seed)
    true_continuum = np.exp(
        rng.uniform(
            np.log(args.continuum_amp_min),
            np.log(args.continuum_amp_max),
            args.n_realizations,
        )
    )
    true_fraction = np.exp(
        rng.uniform(
            np.log(args.blr_fraction_min),
            np.log(args.blr_fraction_max),
            args.n_realizations,
        )
    )
    true_blr = true_continuum * true_fraction
    true_lag_rest = np.exp(
        rng.uniform(
            np.log(args.lag_rest_min),
            np.log(args.lag_rest_max),
            args.n_realizations,
        )
    )
    true_lag_observed = true_lag_rest * (1.0 + redshift)

    rows = []
    for index in range(args.n_realizations):
        kernel = make_kernel(
            n_band,
            jnp.asarray(true_lag_observed[index]),
            jnp.asarray(true_continuum[index]),
            jnp.asarray(true_blr[index]),
            blr_band_index=blr_band_index,
            tau_fast=tau_fast,
            tau_drw=args.tau_drw,
            erlang_order=args.erlang_order,
        )
        key = jax.random.PRNGKey(args.seed + index)
        if args.observation_domain == "magnitude":
            latent_gp = GaussianProcess(
                kernel,
                (times, bands),
                # Numerical nugget only; many bands observe the same latent
                # driver at nearly identical epochs, making the noiseless
                # covariance effectively singular in finite precision.
                diag=jnp.full(times.shape, 1e-8),
                assume_sorted=True,
            )
            latent_relflux = latent_gp.sample(key)
            if float(jnp.min(1.0 + latent_relflux)) <= 0.0:
                # An exact magnitude is undefined for non-positive total flux;
                # deterministically try additional realizations rather than clip.
                for retry in range(1, 100):
                    latent_relflux = latent_gp.sample(jax.random.fold_in(key, retry))
                    if float(jnp.min(1.0 + latent_relflux)) > 0.0:
                        break
                else:
                    raise RuntimeError("Could not draw a positive-flux realization")
            exact_magnitude = relative_flux_to_mag_residual(latent_relflux)
            noise_key = jax.random.fold_in(key, 10_000)
            noisy_magnitude = exact_magnitude + jnp.asarray(magnitude_errors) * jax.random.normal(
                noise_key, shape=exact_magnitude.shape, dtype=exact_magnitude.dtype
            )
            y = mag_residual_to_relative_flux(noisy_magnitude)
            fit_errors = magerr_residual_to_relative_fluxerr(
                noisy_magnitude, jnp.asarray(magnitude_errors)
            )
            fit_noise_var = fit_errors**2
        else:
            gp = GaussianProcess(
                kernel,
                (times, bands),
                diag=reference_noise_var,
                assume_sorted=True,
            )
            y = gp.sample(key)
            fit_noise_var = reference_noise_var
        log_likelihood = np.asarray(
            likelihood_grid(
                jnp.asarray(theta_grid),
                y,
                jnp.asarray(fit_noise_var),
                jnp.asarray(true_lag_observed[index]),
            ),
            dtype=float,
        ).reshape((args.grid_size, args.grid_size))
        if not np.any(np.isfinite(log_likelihood)):
            debug_gp = GaussianProcess(
                kernel,
                (times, bands),
                diag=fit_noise_var,
                assume_sorted=True,
            )
            debug_loglike = float(debug_gp.log_probability(y))
            raise RuntimeError(
                "All amplitude-grid likelihoods are non-finite; "
                f"y range=({float(jnp.nanmin(y))}, {float(jnp.nanmax(y))}), "
                "noise variance range="
                f"({float(jnp.nanmin(fit_noise_var))}, {float(jnp.nanmax(fit_noise_var))}), "
                f"finite y={int(jnp.sum(jnp.isfinite(y)))}/{y.size}, "
                f"finite noise={int(jnp.sum(jnp.isfinite(fit_noise_var)))}/{fit_noise_var.size}, "
                f"direct true-parameter loglike={debug_loglike}"
            )
        weights = np.exp(log_likelihood - np.max(log_likelihood))
        weights /= np.sum(weights)
        cont_weights = np.sum(weights, axis=1)
        blr_weights = np.sum(weights, axis=0)
        cont_low, cont_hat, cont_high = weighted_quantiles(
            continuum_grid, cont_weights
        )
        blr_low, blr_hat, blr_high = weighted_quantiles(blr_grid, blr_weights)
        row = {
            "realization": index,
            "true_continuum_amp": true_continuum[index],
            "recovered_continuum_amp": cont_hat,
            "continuum_amp_low": cont_low,
            "continuum_amp_high": cont_high,
            "true_blr_amp": true_blr[index],
            "recovered_blr_amp": blr_hat,
            "blr_amp_low": blr_low,
            "blr_amp_high": blr_high,
            "true_blr_fraction": true_fraction[index],
            "true_lag_rest": true_lag_rest[index],
            "observation_domain": args.observation_domain,
        }
        rows.append(row)
        print(
            f"[{index + 1:02d}/{args.n_realizations}] "
            f"cont={true_continuum[index]:.3f}->{cont_hat:.3f}, "
            f"BLR={true_blr[index]:.4f}->{blr_hat:.4f}, "
            f"BLR/cont={true_fraction[index]:.3f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_csv or args.output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def values(name):
        return np.asarray([row[name] for row in rows], dtype=float)

    true_cont = values("true_continuum_amp")
    fit_cont = values("recovered_continuum_amp")
    cont_low = values("continuum_amp_low")
    cont_high = values("continuum_amp_high")
    true_blr_amp = values("true_blr_amp")
    fit_blr = values("recovered_blr_amp")
    blr_low = values("blr_amp_low")
    blr_high = values("blr_amp_high")
    fractions = values("true_blr_fraction")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    norm = plt.matplotlib.colors.LogNorm(
        args.blr_fraction_min, args.blr_fraction_max
    )
    panels = (
        (true_cont, fit_cont, cont_low, cont_high, "Continuum amplitude"),
        (true_blr_amp, fit_blr, blr_low, blr_high, "Absolute BLR amplitude"),
    )
    points = None
    for ax, (truth, fitted, low, high, label) in zip(axes, panels, strict=True):
        ax.errorbar(
            truth,
            fitted,
            yerr=np.vstack([fitted - low, high - fitted]),
            fmt="none",
            ecolor="0.55",
            elinewidth=1.0,
            capsize=2,
            alpha=0.8,
        )
        points = ax.scatter(
            truth,
            fitted,
            c=fractions,
            cmap="viridis",
            norm=norm,
            s=52,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )
        limits = [min(np.min(truth), np.min(low)), max(np.max(truth), np.max(high))]
        ax.plot(limits, limits, "k--", linewidth=1.1)
        ax.set(xscale="log", yscale="log", xlim=limits, ylim=limits)
        ax.set_xlabel(f"Injected {label.lower()}")
        ax.grid(alpha=0.18, which="both")
    axes[0].set_ylabel("Recovered amplitude")
    colorbar = fig.colorbar(points, ax=axes, pad=0.015)
    colorbar.set_label("Injected BLR / continuum amplitude")
    fig.savefig(args.output, dpi=220)
    plt.close(fig)

    cont_error = (fit_cont - true_cont) / true_cont
    blr_error = (fit_blr - true_blr_amp) / true_blr_amp
    print(f"Saved plot: {args.output}")
    print(f"Saved results: {csv_path}")
    print(
        "Median absolute fractional error: "
        f"continuum={np.median(np.abs(cont_error)):.3f}, "
        f"BLR={np.median(np.abs(blr_error)):.3f}"
    )


if __name__ == "__main__":
    main()
