#!/usr/bin/env python
"""Validate Erlang BLR lag recovery at the cadence of a real QVC light curve.

The simulation uses the timestamps, band sampling, and reported magnitude
errors of one object, but replaces its measurements with a fixed DRW-like
continuum plus a causal Erlang-filtered copy of that continuum.  Continuum
parameters are held fixed during inference so this test specifically measures
lag recovery as a function of BLR strength.
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

from qvc.light_curve.fit_light_curves import make_lc
from qvc.light_curve.multiband_generate_lc import (
    concat_light_curves,
    populate_sdss_fields,
)
from qvc.light_curve.multiband_model_dho_blr import (
    magerr_residual_to_relative_fluxerr,
)
from qvc.light_curve.multiband_model_dho_blr_erlang import ErlangResponseDHOQS


jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter_object_id", default="1452887")
    parser.add_argument("--bands", nargs="+", default=["u", "g", "r", "i"])
    parser.add_argument("--blr-band", default="r")
    parser.add_argument("--n-realizations", type=int, default=20)
    parser.add_argument("--n-lag-grid", type=int, default=241)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--erlang-order", type=int, default=3)
    parser.add_argument(
        "--injection-response",
        choices=("erlang", "delta"),
        default="erlang",
        help="Transfer function used to generate the mock data; fitting always uses Erlang.",
    )
    parser.add_argument("--tau-drw", type=float, default=300.0)
    parser.add_argument("--sigma-cont", type=float, default=0.15)
    parser.add_argument("--lag-min", type=float, default=10.0, help="Minimum rest-frame lag [days]")
    parser.add_argument("--lag-max", type=float, default=1000.0, help="Maximum rest-frame lag [days]")
    parser.add_argument("--amp-min", type=float, default=0.03)
    parser.add_argument("--amp-max", type=float, default=0.30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/multiband/erlang_lag_recovery_1452887.png"),
    )
    parser.add_argument("--results-csv", type=Path, default=None)
    return parser.parse_args()


def load_real_cadence(object_id: str, selected_bands: list[str]):
    objects = concat_light_curves(
        filter_object_ids=[str(object_id)], progress_bar=False
    )
    objects = populate_sdss_fields(objects, progress_bar=False)
    if not objects:
        raise ValueError(f"Object {object_id!r} was not found")
    raw = objects[0]
    light_curve = make_lc(
        raw,
        list(raw["times"]),
        inject_fake=False,
        drop_band_lyman_alpha=False,
        verbose=False,
    )
    if light_curve is None:
        raise ValueError(f"Object {object_id!r} has no usable observations")

    missing = sorted(set(selected_bands) - set(light_curve["bands"]))
    if missing:
        raise ValueError(f"Bands {missing} are unavailable; choices are {light_curve['bands']}")
    old_to_new = {
        light_curve["bands"].index(name): index
        for index, name in enumerate(selected_bands)
    }
    all_bands = np.asarray(light_curve["band_idx"], dtype=np.int32)
    selected = np.isin(all_bands, list(old_to_new))
    times = np.asarray(light_curve["X"][0], dtype=float)[selected]
    bands = np.asarray([old_to_new[int(value)] for value in all_bands[selected]], dtype=np.int32)
    # Linearized relative-flux uncertainty around zero magnitude residual.
    errors = np.asarray(
        magerr_residual_to_relative_fluxerr(
            jnp.zeros_like(jnp.asarray(light_curve["y"], dtype=float)[selected]),
            jnp.asarray(light_curve["yerr"], dtype=float)[selected],
        ),
        dtype=float,
    )
    valid = np.isfinite(times) & np.isfinite(errors) & (errors > 0.0)
    times, bands, errors = times[valid], bands[valid], errors[valid]
    order = np.lexsort((bands, times))
    return (
        jnp.asarray(times[order]),
        jnp.asarray(bands[order]),
        jnp.asarray(errors[order]),
        tuple(selected_bands),
        float(raw["z"]),
    )


def make_kernel(
    n_band: int,
    lag: jax.Array,
    fraction: jax.Array,
    *,
    tau_fast: float,
    tau_drw: float,
    sigma_cont: float,
    erlang_order: int,
    blr_band_index: int,
) -> ErlangResponseDHOQS:
    amp_cont = jnp.full(n_band, sigma_cont)
    amp_blr = jnp.zeros(n_band).at[blr_band_index].set(sigma_cont * fraction)
    return ErlangResponseDHOQS(
        tau_fast=jnp.full(n_band, tau_fast),
        tau_slow=jnp.full(n_band, tau_drw),
        lag_blr=jnp.full(n_band, lag),
        amp_cont=amp_cont,
        amp_blr=amp_blr,
        order=erlang_order,
    )


def make_log_likelihood_grid_fn(
    times: jax.Array,
    bands: jax.Array,
    noise_var: jax.Array,
    *,
    n_band: int,
    tau_fast: float,
    tau_drw: float,
    sigma_cont: float,
    erlang_order: int,
    blr_band_index: int,
):
    def one_log_likelihood(log_lag, y, fraction):
        kernel = make_kernel(
            n_band,
            jnp.exp(log_lag),
            fraction,
            tau_fast=tau_fast,
            tau_drw=tau_drw,
            sigma_cont=sigma_cont,
            erlang_order=erlang_order,
            blr_band_index=blr_band_index,
        )
        gp = GaussianProcess(
            kernel,
            (times, bands),
            diag=noise_var,
            assume_sorted=True,
        )
        return gp.log_probability(y)

    return jax.jit(
        jax.vmap(one_log_likelihood, in_axes=(0, None, None))
    )


def posterior_quantiles(log_lag_grid, log_likelihood, quantiles=(0.16, 0.5, 0.84)):
    """Quantiles for a posterior uniform in log lag on an even log grid."""

    log_likelihood = np.asarray(log_likelihood, dtype=float)
    weights = np.exp(log_likelihood - np.nanmax(log_likelihood))
    weights /= np.sum(weights)
    cdf = np.cumsum(weights)
    cdf = np.concatenate([[0.0], cdf])
    grid = np.concatenate([[log_lag_grid[0]], log_lag_grid])
    return np.exp(np.interp(np.asarray(quantiles), cdf, grid))


def sample_delta_response_drw(
    rng,
    times,
    bands,
    errors,
    *,
    lag,
    fraction,
    tau_drw,
    sigma_cont,
    blr_band_index,
):
    """Sample continuum plus an exact delayed copy in the selected BLR band."""
    times = np.asarray(times, dtype=float)
    bands = np.asarray(bands, dtype=np.int32)
    errors = np.asarray(errors, dtype=float)
    response = (bands == int(blr_band_index)).astype(float) * float(fraction)

    def ou_cov(left, right):
        return np.exp(-np.abs(left[:, None] - right[None, :]) / float(tau_drw))

    delayed_times = times - float(lag)
    covariance = float(sigma_cont) ** 2 * (
        ou_cov(times, times)
        + response[:, None] * ou_cov(delayed_times, times)
        + response[None, :] * ou_cov(times, delayed_times)
        + response[:, None] * response[None, :] * ou_cov(delayed_times, delayed_times)
    )
    covariance.flat[:: covariance.shape[0] + 1] += errors**2 + 1e-10
    return rng.multivariate_normal(np.zeros(times.size), covariance)


def main() -> None:
    args = parse_args()
    times, bands, errors, band_names, redshift = load_real_cadence(
        args.filter_object_id, args.bands
    )
    n_band = len(band_names)
    if args.blr_band not in band_names:
        raise ValueError(f"BLR band {args.blr_band!r} is not in {band_names}")
    blr_band_index = band_names.index(args.blr_band)
    noise_var = errors**2
    tau_fast = 0.5  # Negligible relative to this cadence: DRW-like SHO limit.
    one_plus_z = 1.0 + redshift
    log_lag_grid_rf = np.linspace(
        np.log(args.lag_min), np.log(args.lag_max), args.n_lag_grid
    )
    log_lag_grid_obs = log_lag_grid_rf + np.log(one_plus_z)
    log_likelihood_grid = make_log_likelihood_grid_fn(
        times,
        bands,
        noise_var,
        n_band=n_band,
        tau_fast=tau_fast,
        tau_drw=args.tau_drw,
        sigma_cont=args.sigma_cont,
        erlang_order=args.erlang_order,
        blr_band_index=blr_band_index,
    )

    rng = np.random.default_rng(args.seed)
    true_lags_rf = np.exp(
        rng.uniform(np.log(args.lag_min), np.log(args.lag_max), args.n_realizations)
    )
    true_fractions = np.exp(
        rng.uniform(np.log(args.amp_min), np.log(args.amp_max), args.n_realizations)
    )

    rows = []
    for index, (true_lag, true_fraction) in enumerate(
        zip(true_lags_rf, true_fractions, strict=True)
    ):
        true_lag_obs = true_lag * one_plus_z
        if args.injection_response == "delta":
            simulated = jnp.asarray(sample_delta_response_drw(
                rng,
                times,
                bands,
                errors,
                lag=true_lag_obs,
                fraction=true_fraction,
                tau_drw=args.tau_drw,
                sigma_cont=args.sigma_cont,
                blr_band_index=blr_band_index,
            ))
        else:
            kernel = make_kernel(
                n_band,
                jnp.asarray(true_lag_obs),
                jnp.asarray(true_fraction),
                tau_fast=tau_fast,
                tau_drw=args.tau_drw,
                sigma_cont=args.sigma_cont,
                erlang_order=args.erlang_order,
                blr_band_index=blr_band_index,
            )
            gp = GaussianProcess(
                kernel,
                (times, bands),
                diag=noise_var,
                assume_sorted=True,
            )
            key = jax.random.PRNGKey(args.seed + index)
            simulated = gp.sample(key)
        log_likelihood = np.asarray(
            log_likelihood_grid(
                jnp.asarray(log_lag_grid_obs), simulated, jnp.asarray(true_fraction)
            )
        )
        lag_low, lag_hat, lag_high = posterior_quantiles(
            log_lag_grid_rf, log_likelihood
        )
        row = {
            "realization": index,
            "true_lag_rest": true_lag,
            "true_lag_observed": true_lag_obs,
            "recovered_lag_rest": lag_hat,
            "lag_low_rest": lag_low,
            "lag_high_rest": lag_high,
            "true_blr_fraction": true_fraction,
            "injection_response": args.injection_response,
        }
        rows.append(row)
        print(
            f"[{index + 1:02d}/{args.n_realizations}] "
            f"lag={true_lag:6.1f} -> {lag_hat:6.1f} "
            f"[{lag_low:5.1f}, {lag_high:5.1f}] d, "
            f"BLR/cont={true_fraction:.3f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_csv or args.output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    true_lag = np.asarray([row["true_lag_rest"] for row in rows])
    lag_hat = np.asarray([row["recovered_lag_rest"] for row in rows])
    lag_low = np.asarray([row["lag_low_rest"] for row in rows])
    lag_high = np.asarray([row["lag_high_rest"] for row in rows])
    fraction = np.asarray([row["true_blr_fraction"] for row in rows])
    yerr = np.vstack([lag_hat - lag_low, lag_high - lag_hat])

    fig, axes = plt.subplots(
        1, 3, figsize=(14.5, 5.2), sharex=True, sharey=True, constrained_layout=True
    )
    finite_error = np.all(np.isfinite(yerr), axis=0) & np.all(yerr >= 0.0, axis=0)
    amp_edges = np.exp(np.linspace(np.log(args.amp_min), np.log(args.amp_max), 4))
    amp_labels = ("Low", "Medium", "High")
    norm = plt.matplotlib.colors.LogNorm(args.amp_min, args.amp_max)
    limits = [args.lag_min, args.lag_max]
    points = None
    for index, (ax, label) in enumerate(zip(axes, amp_labels, strict=True)):
        upper_comparison = fraction <= amp_edges[index + 1] if index == 2 else fraction < amp_edges[index + 1]
        selected = (fraction >= amp_edges[index]) & upper_comparison
        selected_error = selected & finite_error
        ax.errorbar(
            true_lag[selected_error],
            lag_hat[selected_error],
            yerr=yerr[:, selected_error],
            fmt="none",
            ecolor="0.55",
            elinewidth=1.0,
            capsize=2,
            alpha=0.8,
            zorder=1,
        )
        points = ax.scatter(
            true_lag[selected],
            lag_hat[selected],
            c=fraction[selected],
            cmap="viridis",
            norm=norm,
            s=55,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )
        ax.plot(limits, limits, color="black", linestyle="--", linewidth=1.1)
        ax.set(xscale="log", yscale="log", xlim=limits, ylim=limits)
        ax.text(
            0.04,
            0.96,
            f"{label}\n{amp_edges[index]:.3f}–{amp_edges[index + 1]:.3f}\n"
            f"N={np.count_nonzero(selected)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )
        ax.grid(alpha=0.18, which="both")
    axes[0].set_ylabel("Recovered rest-frame BLR lag [days]")
    fig.supxlabel("Injected rest-frame BLR lag [days]")
    colorbar = fig.colorbar(points, ax=axes, pad=0.015)
    colorbar.set_label("Injected BLR / continuum amplitude")
    fig.savefig(args.output, dpi=220)
    plt.close(fig)

    fractional_error = (lag_hat - true_lag) / true_lag
    print(f"Saved plot: {args.output}")
    print(f"Saved results: {csv_path}")
    print(
        "Lag recovery: median fractional error="
        f"{np.median(fractional_error):+.3f}, "
        f"median absolute fractional error={np.median(np.abs(fractional_error)):.3f}"
    )


if __name__ == "__main__":
    main()
