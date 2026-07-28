#!/usr/bin/env python
"""End-to-end recovery of BLR lag and amplitudes at a real QVC cadence.

This validation replaces the measurements of one real object with draws from
the same causal Erlang BLR model used for inference.  It preserves the real
timestamps, band sampling, redshift, and reported photometric uncertainties.
Unlike the conditional lag-grid validation, the continuum amplitude and the
BLR fraction, absolute BLR amplitude, and lag in every band are inferred
jointly.  The injection contains BLR signal in only one band.  By default the
fit uses the same independent per-band BLR lags as QVC. A short AutoNormal SVI
run initializes each NUTS fit.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
from tinygp import GaussianProcess

from qvc.light_curve.fit_light_curves import (
    compute_lambda_center_rf,
    eta_sigma_prior,
    lambda_pivot,
    run_svi_warm_start,
)
from qvc.light_curve.multiband_model_dho_blr_erlang import ErlangResponseDHOQS
from validate_erlang_lag_recovery import load_real_cadence


jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter-object-id", default="1452887")
    parser.add_argument("--bands", nargs="+", default=["u", "g", "r", "i"])
    parser.add_argument("--blr-band", default="r")
    parser.add_argument("--n-realizations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--erlang-order", type=int, default=3)
    parser.add_argument("--tau-drw", type=float, default=300.0)
    parser.add_argument("--lag-min", type=float, default=10.0)
    parser.add_argument("--lag-max", type=float, default=1000.0)
    parser.add_argument("--continuum-amp-min", type=float, default=0.08)
    parser.add_argument("--continuum-amp-max", type=float, default=0.25)
    parser.add_argument(
        "--injected-eta-sigma",
        type=float,
        default=-0.5,
        help="Injected power-law slope of continuum amplitude versus rest wavelength.",
    )
    parser.add_argument("--blr-fraction-min", type=float, default=0.05)
    parser.add_argument("--blr-fraction-max", type=float, default=0.30)
    parser.add_argument("--blr-fraction-prior-median", type=float, default=0.10)
    parser.add_argument("--blr-fraction-prior-log-sigma", type=float, default=1.0)
    parser.add_argument("--svi-steps", type=int, default=300)
    parser.add_argument("--svi-lr", type=float, default=1e-2)
    parser.add_argument("--num-warmup", type=int, default=150)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--max-tree-depth", type=int, default=7)
    parser.add_argument("--target-accept", type=float, default=0.85)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/multiband/erlang_end_to_end_recovery_1452887.png"),
    )
    parser.add_argument("--results-csv", type=Path, default=None)
    return parser.parse_args()


def build_joint_recovery_model(
    times,
    bands,
    errors,
    *,
    n_band,
    lam_rf,
    lambda_center_rf,
    redshift,
    tau_fast,
    tau_drw,
    erlang_order,
    lag_bounds,
    continuum_amp_bounds,
    blr_fraction_prior_median,
    blr_fraction_prior_log_sigma,
):
    """Return a model fitting BLR amplitudes and lags in every retained band."""

    log_lag_bounds = np.log(np.asarray(lag_bounds, dtype=float))
    log_cont_bounds = np.log(np.asarray(continuum_amp_bounds, dtype=float))

    def model(y=None):
        log_lag_rest = numpyro.sample(
            "log_lag_rest",
            dist.Uniform(*log_lag_bounds).expand((n_band,)).to_event(1),
        )
        log_continuum_amp = numpyro.sample(
            "log_continuum_amp", dist.Uniform(*log_cont_bounds)
        )
        eta_sigma = numpyro.sample("eta_sigma", eta_sigma_prior())
        log_blr_fraction = numpyro.sample(
            "log_blr_fraction",
            dist.Normal(
                np.log(blr_fraction_prior_median),
                blr_fraction_prior_log_sigma,
            ).expand((n_band,)).to_event(1),
        )
        lag_rest = numpyro.deterministic("lag_rest", jnp.exp(log_lag_rest))
        continuum_amp = numpyro.deterministic(
            "continuum_amp", jnp.exp(log_continuum_amp)
        )
        continuum_amp_band = numpyro.deterministic(
            "continuum_amp_band",
            continuum_amp * (lam_rf / lambda_center_rf) ** eta_sigma,
        )
        blr_fraction = numpyro.deterministic(
            "blr_fraction", jnp.exp(log_blr_fraction)
        )
        blr_amp = numpyro.deterministic(
            "blr_amp", continuum_amp_band * blr_fraction
        )
        kernel = ErlangResponseDHOQS(
            tau_fast=jnp.full(n_band, tau_fast),
            tau_slow=jnp.full(n_band, tau_drw),
            lag_blr=lag_rest * (1.0 + redshift),
            amp_cont=continuum_amp_band,
            amp_blr=continuum_amp_band * blr_fraction,
            order=erlang_order,
        )
        gp = GaussianProcess(
            kernel,
            (times, bands),
            diag=errors**2,
            assume_sorted=True,
        )
        numpyro.factor("light_curve_log_likelihood", gp.log_probability(y))

    return model


def posterior_interval(samples, name):
    values = np.asarray(samples[name], dtype=float)
    return tuple(np.quantile(values, [0.16, 0.5, 0.84]))


def fit_one_mock(model, simulated, *, seed, args):
    conditioned_model = lambda: model(simulated)
    svi_key, nuts_key = jax.random.split(jax.random.PRNGKey(seed))
    init_values, svi_loss = run_svi_warm_start(
        conditioned_model,
        svi_key,
        num_steps=args.svi_steps,
        learning_rate=args.svi_lr,
        progress_bar=False,
    )
    kernel = NUTS(
        conditioned_model,
        init_strategy=init_to_value(values=init_values),
        dense_mass=True,
        max_tree_depth=args.max_tree_depth,
        target_accept_prob=args.target_accept,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=1,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(nuts_key, extra_fields=("diverging", "accept_prob"))
    samples = {name: np.asarray(value) for name, value in mcmc.get_samples().items()}
    extra = {name: np.asarray(value) for name, value in mcmc.get_extra_fields().items()}
    diagnostics = {
        "svi_final_loss": float(svi_loss),
        "num_divergences": int(np.sum(extra.get("diverging", 0))),
        "mean_accept_prob": float(np.mean(extra.get("accept_prob", np.nan))),
    }
    return samples, diagnostics


def _add_interval(row, samples, name):
    low, median, high = posterior_interval(samples, name)
    row[f"{name}_low"] = low
    row[f"{name}_median"] = median
    row[f"{name}_high"] = high


def _coverage(rows, truth_name, posterior_name):
    return np.mean(
        [
            row[f"{posterior_name}_low"] <= row[truth_name] <= row[f"{posterior_name}_high"]
            for row in rows
        ]
    )


def main() -> None:
    args = parse_args()
    times, bands, errors, band_names, redshift = load_real_cadence(
        args.filter_object_id, args.bands
    )
    if args.blr_band not in band_names:
        raise ValueError(f"BLR band {args.blr_band!r} is not in {band_names}")
    n_band = len(band_names)
    blr_band_index = band_names.index(args.blr_band)
    lam_rf = jnp.asarray(
        [lambda_pivot[band] for band in band_names], dtype=float
    ) / (1.0 + redshift)
    lambda_center_rf = float(compute_lambda_center_rf(lam_rf))
    tau_fast = 0.5
    rng = np.random.default_rng(args.seed)

    true_lag = np.exp(
        rng.uniform(np.log(args.lag_min), np.log(args.lag_max), args.n_realizations)
    )
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

    model = build_joint_recovery_model(
        times,
        bands,
        errors,
        n_band=n_band,
        lam_rf=lam_rf,
        lambda_center_rf=lambda_center_rf,
        redshift=redshift,
        tau_fast=tau_fast,
        tau_drw=args.tau_drw,
        erlang_order=args.erlang_order,
        lag_bounds=(args.lag_min, args.lag_max),
        continuum_amp_bounds=(args.continuum_amp_min / 2, args.continuum_amp_max * 2),
        blr_fraction_prior_median=args.blr_fraction_prior_median,
        blr_fraction_prior_log_sigma=args.blr_fraction_prior_log_sigma,
    )

    rows = []
    for index in range(args.n_realizations):
        true_continuum_band = true_continuum[index] * np.asarray(
            lam_rf / lambda_center_rf
        ) ** args.injected_eta_sigma
        true_blr_amp_band = np.zeros(n_band, dtype=float)
        true_blr_amp_band[blr_band_index] = (
            true_continuum_band[blr_band_index] * true_fraction[index]
        )
        injection_kernel = ErlangResponseDHOQS(
            tau_fast=jnp.full(n_band, tau_fast),
            tau_slow=jnp.full(n_band, args.tau_drw),
            lag_blr=jnp.full(n_band, true_lag[index] * (1.0 + redshift)),
            amp_cont=jnp.asarray(true_continuum_band),
            amp_blr=jnp.asarray(true_blr_amp_band),
            order=args.erlang_order,
        )
        injection_gp = GaussianProcess(
            injection_kernel,
            (times, bands),
            diag=errors**2,
            assume_sorted=True,
        )
        simulated = injection_gp.sample(jax.random.PRNGKey(args.seed + index))
        samples, diagnostics = fit_one_mock(
            model, simulated, seed=args.seed + 10_000 + index, args=args
        )
        row = {
            "realization": index,
            "lag_model": "independent",
            "true_lag_rest": true_lag[index],
            "true_continuum_amp": true_continuum[index],
            "true_continuum_amp_active_band": true_continuum_band[blr_band_index],
            "true_eta_sigma": args.injected_eta_sigma,
            "true_blr_fraction": true_fraction[index],
            "true_blr_amp": true_blr_amp_band[blr_band_index],
            **diagnostics,
        }
        _add_interval(row, samples, "continuum_amp")
        _add_interval(row, samples, "eta_sigma")
        for name in ("lag_rest", "blr_fraction", "blr_amp"):
            active_samples = {name: np.asarray(samples[name])[:, blr_band_index]}
            _add_interval(row, active_samples, name)
        for band_index, band_name in enumerate(band_names):
            for name in ("lag_rest", "blr_fraction", "blr_amp"):
                band_samples = {name: np.asarray(samples[name])[:, band_index]}
                low, median, high = posterior_interval(band_samples, name)
                row[f"{name}_{band_name}_low"] = low
                row[f"{name}_{band_name}_median"] = median
                row[f"{name}_{band_name}_high"] = high
        inactive_indices = [i for i in range(n_band) if i != blr_band_index]
        inactive_fraction_medians = [
            row[f"blr_fraction_{band_names[i]}_median"] for i in inactive_indices
        ]
        row["max_inactive_blr_fraction_median"] = float(
            np.max(inactive_fraction_medians)
        )
        if "log_lag_rest_shared" in samples:
            shared_samples = {
                "lag_rest_shared": np.exp(np.asarray(samples["log_lag_rest_shared"]))
            }
            _add_interval(row, shared_samples, "lag_rest_shared")
        rows.append(row)
        print(
            f"[{index + 1}/{args.n_realizations}] lag {true_lag[index]:.1f} -> "
            f"{row['lag_rest_median']:.1f} d; cont {true_continuum[index]:.3f} -> "
            f"{row['continuum_amp_median']:.3f}; BLR {row['true_blr_amp']:.4f} -> "
            f"{row['blr_amp_median']:.4f}; max inactive fraction="
            f"{row['max_inactive_blr_fraction_median']:.3f}; "
            f"divergences={row['num_divergences']}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_csv or args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    panels = (
        ("true_lag_rest", "lag_rest", "BLR lag [rest-frame days]"),
        ("true_continuum_amp", "continuum_amp", "Continuum amplitude"),
        ("true_blr_amp", "blr_amp", "BLR amplitude"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    log10_blr_fraction = np.log10(
        np.asarray([row["true_blr_fraction"] for row in rows])
    )
    points = None
    for ax, (truth_name, posterior_name, label) in zip(axes, panels, strict=True):
        truth = np.asarray([row[truth_name] for row in rows])
        median = np.asarray([row[f"{posterior_name}_median"] for row in rows])
        low = np.asarray([row[f"{posterior_name}_low"] for row in rows])
        high = np.asarray([row[f"{posterior_name}_high"] for row in rows])
        limits = [min(truth.min(), low.min()) * 0.8, max(truth.max(), high.max()) * 1.2]
        ax.errorbar(
            truth,
            median,
            yerr=np.vstack((median - low, high - median)),
            fmt="none",
            ecolor="0.6",
            elinewidth=1,
            alpha=0.75,
            zorder=1,
        )
        points = ax.scatter(
            truth,
            median,
            c=log10_blr_fraction,
            cmap="viridis",
            vmin=np.log10(args.blr_fraction_min),
            vmax=np.log10(args.blr_fraction_max),
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        ax.plot(limits, limits, "k--", linewidth=1)
        ax.set(xscale="log", yscale="log", xlim=limits, ylim=limits)
        ax.set_xlabel(f"Injected {label}")
        ax.set_ylabel(f"Recovered {label}")
        ax.grid(alpha=0.2, which="both")
    colorbar = fig.colorbar(points, ax=axes, pad=0.015)
    colorbar.set_label(r"$\log_{10}(\mathrm{BLR}/\mathrm{continuum})$")
    fig.savefig(args.output, dpi=220)
    plt.close(fig)

    for truth_name, posterior_name, label in panels:
        truth = np.asarray([row[truth_name] for row in rows])
        estimate = np.asarray([row[f"{posterior_name}_median"] for row in rows])
        log_error = np.log(estimate / truth)
        print(
            f"{label}: 68% coverage={_coverage(rows, truth_name, posterior_name):.2f}, "
            f"median fractional bias={np.median(estimate / truth - 1):+.3f}, "
            f"median |log error|={np.median(np.abs(log_error)):.3f}"
        )
    inactive_fraction = np.asarray(
        [row["max_inactive_blr_fraction_median"] for row in rows], dtype=float
    )
    print(
        "Continuum-only bands: median maximum inferred BLR/continuum="
        f"{np.median(inactive_fraction):.3f}; fractions with maximum >0.1="
        f"{np.mean(inactive_fraction > 0.1):.2f} and >0.3="
        f"{np.mean(inactive_fraction > 0.3):.2f}"
    )
    print(f"Total divergences: {sum(row['num_divergences'] for row in rows)}")
    print(f"Saved plot: {args.output}")
    print(f"Saved results: {csv_path}")


if __name__ == "__main__":
    main()
