#!/usr/bin/env python
"""End-to-end recovery of BLR lag and amplitudes at a real QVC cadence.

This validation replaces the measurements of one real object with draws using
a causal Erlang BLR response.  Injection and recovery continuum kernels can be
selected independently, including exact DRW injection with CARMA(2,1)
recovery. It preserves the real timestamps, band sampling, redshift, and
reported photometric uncertainties.
Unlike the conditional lag-grid validation, the DRW timescale, continuum
amplitude, and the BLR fraction, absolute BLR amplitude, and lag in every band
are inferred jointly.  The injection contains BLR signal in only one band. By
default the fit uses the same independent per-band BLR lags as QVC. A short
AutoNormal SVI run initializes each NUTS fit.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam
from tinygp import GaussianProcess

from qvc.light_curve.fit_light_curves import (
    compute_lambda_center_rf,
    eta_sigma_prior,
    lambda_pivot,
)
from qvc.light_curve.dho_drw_parameterization import IntegratedTimescaleDHOBaseQS
from qvc.light_curve.multiband_dho_core import (
    mag_residual_to_relative_flux,
    magerr_residual_to_relative_fluxerr,
    relative_flux_to_mag_residual,
)
from qvc.light_curve.multiband_model_dho_blr_erlang_drw import (
    ErlangResponseIntegratedDHOQS,
    POSITIVE_FLUX_MARGIN_SOFTNESS,
    POSITIVE_FLUX_N_SIGMA,
)
from qvc.light_curve.multiband_model_dho_blr_erlang import (
    ErlangResponseDHOQS,
    ErlangResponseDRWQS,
)
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
    parser.add_argument(
        "--tau-drw",
        type=float,
        default=None,
        help=(
            "Fixed injected rest-frame DRW timescale in days. By default, "
            "each realization draws tau from --tau-drw-min/--tau-drw-max."
        ),
    )
    parser.add_argument(
        "--tau-drw-grid",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Explicit injected rest-frame tau grid in days. With --lag-grid, "
            "forms a Cartesian product with lag and BLR fraction; otherwise "
            "--n-realizations mocks are generated for each tau."
        ),
    )
    parser.add_argument(
        "--tau-drw-min",
        type=float,
        default=100.0,
        help=(
            "Minimum randomly injected rest-frame tau in days "
            "(default: 100; approximately the lower Stone range)."
        ),
    )
    parser.add_argument(
        "--tau-drw-max",
        type=float,
        default=6_000.0,
        help=(
            "Maximum randomly injected rest-frame tau in days "
            "(default: 6000; approximately the Stone 99th percentile)."
        ),
    )
    parser.add_argument(
        "--tau-drw-fit-min",
        type=float,
        default=10.0,
        help="Lower bound of the log-uniform rest-frame tau prior in days.",
    )
    parser.add_argument(
        "--tau-drw-fit-max",
        type=float,
        default=10_000.0,
        help="Upper bound of the log-uniform rest-frame tau prior in days.",
    )
    parser.add_argument(
        "--infer-tau-drw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Infer the DRW timescale (default). Use --no-infer-tau-drw for "
            "the previous conditional recovery with tau fixed to its injected value."
        ),
    )
    parser.add_argument("--quality-factor", type=float, default=0.1)
    parser.add_argument("--perturbation-ratio", type=float, default=0.02)
    parser.add_argument(
        "--kernel-model",
        choices=("carma21", "legacy"),
        default=None,
        help=(
            "Deprecated compatibility option that sets both injection and "
            "recovery kernels. Prefer the separate kernel options."
        ),
    )
    parser.add_argument(
        "--injection-kernel-model",
        choices=("drw", "carma21", "legacy"),
        default=None,
        help=(
            "Kernel used to generate mocks (default: carma21). Use drw for "
            "an exact OU/DRW continuum with the same causal Erlang response."
        ),
    )
    parser.add_argument(
        "--recovery-kernel-model",
        choices=("carma21", "legacy"),
        default=None,
        help="Kernel used for inference (default: carma21).",
    )
    parser.add_argument("--lag-min", type=float, default=10.0)
    parser.add_argument("--lag-max", type=float, default=1000.0)
    parser.add_argument(
        "--lag-grid",
        type=float,
        nargs="+",
        default=None,
        help="Explicit rest-frame lag grid; forms a Cartesian product with --blr-fraction-grid.",
    )
    parser.add_argument(
        "--continuum-amp-min",
        type=float,
        default=0.04,
        help=(
            "Minimum injected continuum sigma in relative-flux RMS "
            "(default: 0.04; converted from the Stone magnitude range)."
        ),
    )
    parser.add_argument(
        "--continuum-amp-max",
        type=float,
        default=0.41,
        help=(
            "Maximum injected continuum sigma in relative-flux RMS "
            "(default: 0.41; converted from the Stone magnitude range)."
        ),
    )
    parser.add_argument(
        "--injected-eta-sigma",
        type=float,
        default=-0.5,
        help="Injected power-law slope of continuum amplitude versus rest wavelength.",
    )
    parser.add_argument("--blr-fraction-min", type=float, default=0.05)
    parser.add_argument("--blr-fraction-max", type=float, default=0.30)
    parser.add_argument(
        "--blr-fraction-grid",
        type=float,
        nargs="+",
        default=None,
        help="Explicit BLR/continuum ratio grid; forms a Cartesian product with --lag-grid.",
    )
    parser.add_argument("--blr-fraction-prior-median", type=float, default=0.10)
    parser.add_argument("--blr-fraction-prior-log-sigma", type=float, default=1.0)
    parser.add_argument("--svi-steps", type=int, default=4000)
    parser.add_argument("--svi-lr", type=float, default=1e-3)
    parser.add_argument("--num-warmup", type=int, default=150)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=1,
        help="Number of independent realizations to fit concurrently (default: 1).",
    )
    parser.add_argument("--max-tree-depth", type=int, default=7)
    parser.add_argument("--target-accept", type=float, default=0.85)
    parser.add_argument(
        "--enforce-positive-flux-guard",
        action="store_true",
        default=False,
        help="Apply the historical four-sigma positive-total-flux penalty during recovery.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show SVI and NUTS progress bars (default: enabled).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/multiband/erlang_end_to_end_recovery_1452887.png"),
    )
    parser.add_argument("--results-csv", type=Path, default=None)
    return parser.parse_args()


def resolve_kernel_models(args: argparse.Namespace) -> None:
    """Resolve new split kernel flags while preserving --kernel-model."""

    compatibility_model = args.kernel_model
    for option_name in ("injection_kernel_model", "recovery_kernel_model"):
        explicit_model = getattr(args, option_name)
        if (
            compatibility_model is not None
            and explicit_model is not None
            and explicit_model != compatibility_model
        ):
            flag_name = option_name.replace("_", "-")
            raise ValueError(
                f"--kernel-model {compatibility_model} conflicts with "
                f"--{flag_name} {explicit_model}"
            )
        setattr(
            args,
            option_name,
            explicit_model or compatibility_model or "carma21",
        )


def make_erlang_kernel(
    tau_drw,
    quality_factor,
    perturbation_ratio,
    lag_observed,
    amp_cont,
    amp_blr_rms,
    erlang_order,
    kernel_model,
):
    """Build an exact DRW, CARMA(2,1), or legacy Erlang kernel."""

    tau_drw = jnp.asarray(tau_drw)
    amp_cont = jnp.asarray(amp_cont)
    amp_blr_rms = jnp.asarray(amp_blr_rms)
    lag_observed = jnp.asarray(lag_observed)
    if kernel_model == "legacy":
        return ErlangResponseDHOQS(
            tau_fast=jnp.asarray(perturbation_ratio) * tau_drw,
            tau_slow=tau_drw,
            lag_blr=lag_observed,
            amp_cont=amp_cont,
            amp_blr=amp_blr_rms,
            order=erlang_order,
        )
    if kernel_model == "drw":
        unit_response = ErlangResponseDRWQS(
            tau_drw=tau_drw,
            lag_blr=lag_observed,
            amp_cont=jnp.zeros_like(amp_cont),
            amp_blr=jnp.ones_like(amp_blr_rms),
            order=erlang_order,
        )
        band_indices = jnp.arange(tau_drw.size, dtype=jnp.int32)
        response_var = jax.vmap(
            lambda band: unit_response.evaluate((0.0, band), (0.0, band))
        )(band_indices)
        return ErlangResponseDRWQS(
            tau_drw=tau_drw,
            lag_blr=lag_observed,
            amp_cont=amp_cont,
            amp_blr=amp_blr_rms / jnp.sqrt(jnp.maximum(response_var, 1e-12)),
            order=erlang_order,
        )
    if kernel_model != "carma21":
        raise ValueError(f"Unsupported Erlang kernel model: {kernel_model!r}")
    base = IntegratedTimescaleDHOBaseQS.from_drw(
        tau_drw,
        jnp.asarray(quality_factor),
        jnp.asarray(perturbation_ratio) * tau_drw,
    )
    unit_response = ErlangResponseIntegratedDHOQS(
        tau_fast=jnp.full_like(tau_drw, 0.5),
        tau_slow=jnp.full_like(tau_drw, 0.5),
        lag_blr=lag_observed,
        amp_cont=jnp.zeros_like(amp_cont),
        amp_blr=jnp.ones_like(amp_blr_rms),
        order=erlang_order,
        carma_omega0=base.omega0,
        carma_damping=base.damping,
        carma_obs_position=base.obs_position,
        carma_obs_velocity=base.obs_velocity,
    )
    band_indices = jnp.arange(tau_drw.size, dtype=jnp.int32)
    response_var = jax.vmap(
        lambda band: unit_response.evaluate((0.0, band), (0.0, band))
    )(band_indices)
    return ErlangResponseIntegratedDHOQS(
        tau_fast=jnp.full_like(tau_drw, 0.5),
        tau_slow=jnp.full_like(tau_drw, 0.5),
        lag_blr=lag_observed,
        amp_cont=amp_cont,
        amp_blr=amp_blr_rms / jnp.sqrt(jnp.maximum(response_var, 1e-12)),
        order=erlang_order,
        carma_omega0=base.omega0,
        carma_damping=base.damping,
        carma_obs_position=base.obs_position,
        carma_obs_velocity=base.obs_velocity,
    )


def build_joint_recovery_model(
    times,
    bands,
    errors,
    *,
    n_band,
    lam_rf,
    lambda_center_rf,
    redshift,
    tau_drw_rest,
    infer_tau_drw,
    tau_drw_fit_bounds,
    quality_factor,
    perturbation_ratio,
    erlang_order,
    lag_bounds,
    continuum_amp_bounds,
    blr_fraction_prior_median,
    blr_fraction_prior_log_sigma,
    kernel_model,
    enforce_positive_flux_guard,
):
    """Return a model fitting BLR amplitudes and lags in every retained band."""

    log_lag_bounds = np.log(np.asarray(lag_bounds, dtype=float))
    log_cont_bounds = np.log(np.asarray(continuum_amp_bounds, dtype=float))
    log_tau_bounds = np.log(np.asarray(tau_drw_fit_bounds, dtype=float))

    def model(y=None):
        if infer_tau_drw:
            log_tau_drw_rest = numpyro.sample(
                "log_tau_drw_rest", dist.Uniform(*log_tau_bounds)
            )
            fitted_tau_drw_rest = numpyro.deterministic(
                "tau_drw_rest", jnp.exp(log_tau_drw_rest)
            )
        else:
            fitted_tau_drw_rest = numpyro.deterministic(
                "tau_drw_rest", jnp.asarray(tau_drw_rest)
            )
        tau_drw_observed = fitted_tau_drw_rest * (1.0 + redshift)
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
            dist.TruncatedNormal(
                np.log(blr_fraction_prior_median),
                blr_fraction_prior_log_sigma,
                low=np.log(5e-3),
                high=0.0,
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
        kernel = make_erlang_kernel(
            jnp.full(n_band, tau_drw_observed),
            quality_factor,
            perturbation_ratio,
            lag_rest * (1.0 + redshift),
            continuum_amp_band,
            continuum_amp_band * blr_fraction,
            erlang_order,
            kernel_model,
        )
        gp = GaussianProcess(
            kernel,
            (times, bands),
            diag=errors**2,
            assume_sorted=True,
        )
        band_indices = jnp.arange(n_band, dtype=jnp.int32)
        stationary_variance = jax.vmap(
            lambda band: kernel.evaluate((0.0, band), (0.0, band))
        )(band_indices)
        stationary_std = jnp.sqrt(jnp.maximum(stationary_variance, 1e-24))
        positive_flux_margin = 1.0 - POSITIVE_FLUX_N_SIGMA * stationary_std
        negative_flux_probability = jax.scipy.special.ndtr(-1.0 / stationary_std)
        numpyro.deterministic(
            "positive_flux_margin_min",
            jnp.min(positive_flux_margin),
        )
        numpyro.deterministic(
            "negative_total_flux_probability_max",
            jnp.max(negative_flux_probability),
        )
        if enforce_positive_flux_guard:
            scaled_violation = -positive_flux_margin / POSITIVE_FLUX_MARGIN_SOFTNESS
            numpyro.factor(
                "positive_flux_guard",
                -0.5 * jnp.sum(jax.nn.softplus(scaled_violation) ** 2),
            )
        numpyro.factor("light_curve_log_likelihood", gp.log_probability(y))

    return model


def posterior_interval(samples, name):
    values = np.asarray(samples[name], dtype=float)
    return tuple(np.quantile(values, [0.16, 0.5, 0.84]))


def fit_one_mock(model, simulated, *, seed, args):
    conditioned_model = lambda: model(simulated)
    svi_key, nuts_key = jax.random.split(jax.random.PRNGKey(seed))
    init_values = {
        "log_lag_rest": np.full(
            len(args.bands),
            0.5 * (np.log(args.lag_min) + np.log(args.lag_max)),
        ),
        "log_continuum_amp": 0.5
        * (
            np.log(args.continuum_amp_min / 2)
            + np.log(args.continuum_amp_max * 2)
        ),
        "eta_sigma": args.injected_eta_sigma,
        "log_blr_fraction": np.full(
            len(args.bands),
            np.log(args.blr_fraction_prior_median),
        ),
    }
    if args.infer_tau_drw:
        init_values["log_tau_drw_rest"] = 0.5 * (
            np.log(args.tau_drw_fit_min) + np.log(args.tau_drw_fit_max)
        )
    guide = AutoNormal(
        conditioned_model,
        init_loc_fn=init_to_value(values=init_values),
    )
    svi = SVI(
        conditioned_model,
        guide,
        Adam(args.svi_lr),
        Trace_ELBO(),
    )
    svi_result = svi.run(svi_key, args.svi_steps, progress_bar=args.progress)
    init_values = guide.median(svi_result.params)
    svi_loss = np.asarray(svi_result.losses)[-1]
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
        progress_bar=args.progress,
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


def run_one_realization(payload):
    """Inject and fit one mock; kept top-level for spawn-based process pools."""

    index = payload["index"]
    args = payload["args"]
    times = jnp.asarray(payload["times"])
    bands = jnp.asarray(payload["bands"])
    errors = jnp.asarray(payload["errors"])
    band_names = payload["band_names"]
    redshift = payload["redshift"]
    lam_rf = jnp.asarray(payload["lam_rf"])
    lambda_center_rf = payload["lambda_center_rf"]
    n_band = len(band_names)
    blr_band_index = band_names.index(args.blr_band)
    true_tau_drw_rest = payload["true_tau_drw_rest"]
    true_lag = payload["true_lag"]
    true_continuum = payload["true_continuum"]
    true_fraction = payload["true_fraction"]
    time_baseline_observed = payload["time_baseline_observed"]
    time_baseline_rest = payload["time_baseline_rest"]

    true_continuum_band = true_continuum * np.asarray(
        lam_rf / lambda_center_rf
    ) ** args.injected_eta_sigma
    true_blr_amp_band = np.zeros(n_band, dtype=float)
    true_blr_amp_band[blr_band_index] = (
        true_continuum_band[blr_band_index] * true_fraction
    )
    injection_kernel = make_erlang_kernel(
        jnp.full(n_band, true_tau_drw_rest * (1.0 + redshift)),
        args.quality_factor,
        args.perturbation_ratio,
        jnp.full(n_band, true_lag * (1.0 + redshift)),
        jnp.asarray(true_continuum_band),
        jnp.asarray(true_blr_amp_band),
        args.erlang_order,
        args.injection_kernel_model,
    )
    coordinates = (times, bands)
    dense_covariance = jax.vmap(
        lambda t1, b1: jax.vmap(
            lambda t2, b2: injection_kernel.evaluate((t1, b1), (t2, b2))
        )(*coordinates)
    )(*coordinates)
    dense_covariance = np.asarray(dense_covariance, dtype=float)
    dense_covariance = 0.5 * (dense_covariance + dense_covariance.T)
    eig_min = float(np.linalg.eigvalsh(dense_covariance)[0])
    dense_covariance += np.eye(dense_covariance.shape[0]) * max(
        1e-10,
        -eig_min + 1e-10,
    )
    latent_rng = np.random.default_rng(args.seed + 100_000 + index)
    latent_relflux = latent_rng.multivariate_normal(
        np.zeros(dense_covariance.shape[0]),
        dense_covariance,
    )
    mag_error = errors / (0.4 * np.log(10.0))
    noise_rng = np.random.default_rng(args.seed + 50_000 + index)
    simulated_mag = np.array(
        relative_flux_to_mag_residual(latent_relflux),
        dtype=float,
        copy=True,
    )
    simulated_mag += noise_rng.normal(0.0, np.asarray(mag_error))
    simulated = np.asarray(mag_residual_to_relative_flux(simulated_mag))
    simulated_errors = np.asarray(
        magerr_residual_to_relative_fluxerr(simulated_mag, mag_error)
    )
    fit_model = build_joint_recovery_model(
        times,
        bands,
        simulated_errors,
        n_band=n_band,
        lam_rf=lam_rf,
        lambda_center_rf=lambda_center_rf,
        redshift=redshift,
        tau_drw_rest=true_tau_drw_rest,
        infer_tau_drw=args.infer_tau_drw,
        tau_drw_fit_bounds=(args.tau_drw_fit_min, args.tau_drw_fit_max),
        quality_factor=args.quality_factor,
        perturbation_ratio=args.perturbation_ratio,
        erlang_order=args.erlang_order,
        lag_bounds=(args.lag_min, args.lag_max),
        continuum_amp_bounds=(
            args.continuum_amp_min / 2,
            args.continuum_amp_max * 2,
        ),
        blr_fraction_prior_median=args.blr_fraction_prior_median,
        blr_fraction_prior_log_sigma=args.blr_fraction_prior_log_sigma,
        kernel_model=args.recovery_kernel_model,
        enforce_positive_flux_guard=args.enforce_positive_flux_guard,
    )
    center_continuum = np.sqrt(
        (args.continuum_amp_min / 2) * (args.continuum_amp_max * 2)
    )
    center_continuum_band = center_continuum * np.asarray(
        lam_rf / lambda_center_rf
    ) ** args.injected_eta_sigma
    center_tau_rest = (
        np.sqrt(args.tau_drw_fit_min * args.tau_drw_fit_max)
        if args.infer_tau_drw
        else true_tau_drw_rest
    )
    center_kernel = make_erlang_kernel(
        jnp.full(n_band, center_tau_rest * (1.0 + redshift)),
        args.quality_factor,
        args.perturbation_ratio,
        jnp.full(
            n_band,
            np.sqrt(args.lag_min * args.lag_max) * (1.0 + redshift),
        ),
        jnp.asarray(center_continuum_band),
        jnp.asarray(center_continuum_band * args.blr_fraction_prior_median),
        args.erlang_order,
        args.recovery_kernel_model,
    )
    center_loglike = GaussianProcess(
        center_kernel,
        (times, bands),
        diag=simulated_errors**2,
        assume_sorted=True,
    ).log_probability(simulated)
    if not np.isfinite(float(center_loglike)):
        raise RuntimeError(
            "Recovery likelihood is non-finite at the deterministic prior center; "
            f"data finite={np.all(np.isfinite(simulated))}, "
            f"errors finite={np.all(np.isfinite(simulated_errors))}, "
            f"min error={np.min(simulated_errors):.3e}."
        )
    samples, diagnostics = fit_one_mock(
        fit_model,
        simulated,
        seed=args.seed + 10_000 + index,
        args=args,
    )
    row = {
        "realization": index,
        "lag_model": "independent",
        "kernel_model": args.recovery_kernel_model,
        "injection_kernel_model": args.injection_kernel_model,
        "recovery_kernel_model": args.recovery_kernel_model,
        "num_parallel_requested": args.num_parallel,
        "tau_drw_injection_mode": (
            "grid"
            if args.tau_drw_grid is not None
            else "fixed"
            if args.tau_drw is not None
            else "log_uniform"
        ),
        "tau_drw_injection_min_rest": args.tau_drw_min,
        "tau_drw_injection_max_rest": args.tau_drw_max,
        "continuum_sigma_injection_min": args.continuum_amp_min,
        "continuum_sigma_injection_max": args.continuum_amp_max,
        "positive_flux_guard_enforced": args.enforce_positive_flux_guard,
        "tau_drw_inferred": args.infer_tau_drw,
        "tau_drw_fit_min_rest": args.tau_drw_fit_min,
        "tau_drw_fit_max_rest": args.tau_drw_fit_max,
        "true_tau_drw_rest": true_tau_drw_rest,
        "time_baseline_observed": time_baseline_observed,
        "time_baseline_rest": time_baseline_rest,
        "true_tau_to_baseline_rest": true_tau_drw_rest / time_baseline_rest,
        "true_lag_rest": true_lag,
        "true_continuum_amp": true_continuum,
        "true_continuum_amp_active_band": true_continuum_band[blr_band_index],
        "true_eta_sigma": args.injected_eta_sigma,
        "true_blr_fraction": true_fraction,
        "true_blr_amp": true_blr_amp_band[blr_band_index],
        **diagnostics,
    }
    _add_interval(row, samples, "continuum_amp")
    _add_interval(row, samples, "tau_drw_rest")
    row["recovered_tau_to_baseline_rest"] = (
        row["tau_drw_rest_median"] / time_baseline_rest
    )
    if args.infer_tau_drw:
        log_sigma_samples = np.log(
            np.asarray(samples["continuum_amp"], dtype=float)
        )
        log_tau_samples = np.log(
            np.asarray(samples["tau_drw_rest"], dtype=float)
        )
        if (
            log_sigma_samples.size > 1
            and np.std(log_sigma_samples) > 0
            and np.std(log_tau_samples) > 0
        ):
            row["posterior_corr_log_sigma_log_tau"] = float(
                np.corrcoef(log_sigma_samples, log_tau_samples)[0, 1]
            )
        else:
            row["posterior_corr_log_sigma_log_tau"] = np.nan
    else:
        row["posterior_corr_log_sigma_log_tau"] = np.nan
    _add_interval(row, samples, "eta_sigma")
    _add_interval(row, samples, "positive_flux_margin_min")
    _add_interval(row, samples, "negative_total_flux_probability_max")
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
    return row


def write_results_csv(rows, csv_path):
    """Atomically rewrite the accumulated result table."""

    ordered_rows = sorted(rows, key=lambda row: row["realization"])
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered_rows[0]))
        writer.writeheader()
        writer.writerows(ordered_rows)


def format_recovery_result(row, completed, total, infer_tau_drw):
    fixed_label = " (fixed)" if not infer_tau_drw else ""
    return (
        f"[{completed}/{total}] realization {row['realization']}: "
        f"lag {row['true_lag_rest']:.1f} -> {row['lag_rest_median']:.1f} d; "
        f"sigma {row['true_continuum_amp']:.3f} -> "
        f"{row['continuum_amp_median']:.3f}; "
        f"tau {row['true_tau_drw_rest']:.1f} -> "
        f"{row['tau_drw_rest_median']:.1f} rest-frame d{fixed_label}; "
        f"BLR {row['true_blr_amp']:.4f} -> {row['blr_amp_median']:.4f}; "
        f"max inactive fraction={row['max_inactive_blr_fraction_median']:.3f}; "
        f"divergences={row['num_divergences']}"
    )


def main() -> None:
    args = parse_args()
    resolve_kernel_models(args)
    print(
        "Kernel models: "
        f"injection={args.injection_kernel_model}, "
        f"recovery={args.recovery_kernel_model}",
        flush=True,
    )
    if args.tau_drw_fit_min <= 0 or args.tau_drw_fit_max <= args.tau_drw_fit_min:
        raise ValueError(
            "Require 0 < --tau-drw-fit-min < --tau-drw-fit-max"
        )
    if args.tau_drw is not None and args.tau_drw_grid is not None:
        raise ValueError("Use either --tau-drw or --tau-drw-grid, not both")
    if args.tau_drw_min <= 0 or args.tau_drw_max <= args.tau_drw_min:
        raise ValueError("Require 0 < --tau-drw-min < --tau-drw-max")
    if (
        args.continuum_amp_min <= 0
        or args.continuum_amp_max <= args.continuum_amp_min
    ):
        raise ValueError(
            "Require 0 < --continuum-amp-min < --continuum-amp-max"
        )
    tau_grid = (
        np.asarray(args.tau_drw_grid, dtype=float)
        if args.tau_drw_grid is not None
        else None
    )
    explicit_tau = (
        tau_grid
        if tau_grid is not None
        else np.asarray([args.tau_drw], dtype=float)
        if args.tau_drw is not None
        else None
    )
    if explicit_tau is not None and (
        np.any(~np.isfinite(explicit_tau)) or np.any(explicit_tau <= 0)
    ):
        raise ValueError("Injected DRW timescales must be finite and positive")
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
    time_baseline_observed = float(np.max(times) - np.min(times))
    time_baseline_rest = time_baseline_observed / (1.0 + redshift)
    rng = np.random.default_rng(args.seed)

    if (args.lag_grid is None) != (args.blr_fraction_grid is None):
        raise ValueError("--lag-grid and --blr-fraction-grid must be specified together")
    if args.lag_grid is not None:
        if tau_grid is not None:
            true_tau_drw_rest, true_lag, true_fraction = (
                np.asarray(values, dtype=float).ravel()
                for values in np.meshgrid(
                    tau_grid,
                    np.asarray(args.lag_grid, dtype=float),
                    np.asarray(args.blr_fraction_grid, dtype=float),
                    indexing="ij",
                )
            )
        else:
            true_lag, true_fraction = (
                np.asarray(values, dtype=float).ravel()
                for values in np.meshgrid(
                    np.asarray(args.lag_grid, dtype=float),
                    np.asarray(args.blr_fraction_grid, dtype=float),
                    indexing="ij",
                )
            )
            if args.tau_drw is not None:
                true_tau_drw_rest = np.full(true_lag.size, args.tau_drw)
            else:
                true_tau_drw_rest = np.exp(
                    rng.uniform(
                        np.log(args.tau_drw_min),
                        np.log(args.tau_drw_max),
                        true_lag.size,
                    )
                )
        n_realizations = true_lag.size
        true_continuum = np.full(
            n_realizations,
            np.sqrt(args.continuum_amp_min * args.continuum_amp_max),
        )
    else:
        if tau_grid is not None:
            true_tau_drw_rest = np.repeat(tau_grid, args.n_realizations)
        elif args.tau_drw is not None:
            true_tau_drw_rest = np.full(args.n_realizations, args.tau_drw)
        else:
            true_tau_drw_rest = np.exp(
                rng.uniform(
                    np.log(args.tau_drw_min),
                    np.log(args.tau_drw_max),
                    args.n_realizations,
                )
            )
        n_realizations = true_tau_drw_rest.size
        true_lag = np.exp(
            rng.uniform(np.log(args.lag_min), np.log(args.lag_max), n_realizations)
        )
        true_continuum = np.exp(
            rng.uniform(
                np.log(args.continuum_amp_min),
                np.log(args.continuum_amp_max),
                n_realizations,
            )
        )
        true_fraction = np.exp(
            rng.uniform(
                np.log(args.blr_fraction_min),
                np.log(args.blr_fraction_max),
                n_realizations,
            )
        )

    if args.infer_tau_drw and (
        np.any(true_tau_drw_rest < args.tau_drw_fit_min)
        or np.any(true_tau_drw_rest > args.tau_drw_fit_max)
    ):
        raise ValueError(
            "Injected DRW timescales must lie within the fitted tau prior bounds"
        )

    if args.num_parallel < 1:
        raise ValueError("--num-parallel must be at least 1")
    worker_args = argparse.Namespace(**vars(args))
    if args.num_parallel > 1 and args.progress:
        print(
            "Parallel mode: disabling per-worker SVI/NUTS progress bars.",
            flush=True,
        )
        worker_args.progress = False

    common_payload = {
        "args": worker_args,
        "times": np.asarray(times),
        "bands": np.asarray(bands),
        "errors": np.asarray(errors),
        "band_names": band_names,
        "redshift": redshift,
        "lam_rf": np.asarray(lam_rf),
        "lambda_center_rf": lambda_center_rf,
        "time_baseline_observed": time_baseline_observed,
        "time_baseline_rest": time_baseline_rest,
    }
    tasks = [
        {
            **common_payload,
            "index": index,
            "true_tau_drw_rest": float(true_tau_drw_rest[index]),
            "true_lag": float(true_lag[index]),
            "true_continuum": float(true_continuum[index]),
            "true_fraction": float(true_fraction[index]),
        }
        for index in range(n_realizations)
    ]

    rows = []
    csv_path = args.results_csv or args.output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    def record_result(row):
        rows.append(row)
        write_results_csv(rows, csv_path)
        print(
            format_recovery_result(
                row,
                completed=len(rows),
                total=n_realizations,
                infer_tau_drw=args.infer_tau_drw,
            ),
            flush=True,
        )

    if args.num_parallel == 1:
        for task in tasks:
            print(
                f"[{len(rows) + 1}/{n_realizations}] Preparing injection and fit",
                flush=True,
            )
            record_result(run_one_realization(task))
    else:
        max_workers = min(args.num_parallel, n_realizations)
        print(
            f"Fitting {n_realizations} realizations with {max_workers} workers",
            flush=True,
        )
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=context,
        ) as executor:
            futures = [executor.submit(run_one_realization, task) for task in tasks]
            for future in as_completed(futures):
                record_result(future.result())
        rows.sort(key=lambda row: row["realization"])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    panels = (
        ("true_lag_rest", "lag_rest", "BLR lag\n[rest-frame days]"),
        (
            "true_continuum_amp",
            "continuum_amp",
            "Continuum $\\sigma$\n[stationary relative-flux RMS]",
        ),
        (
            "true_tau_drw_rest",
            "tau_drw_rest",
            "Continuum $\\tau_{\\rm DRW}$\n[rest-frame days]",
        ),
        ("true_blr_amp", "blr_amp", "BLR amplitude"),
    )
    fig, axes_grid = plt.subplots(
        2,
        2,
        figsize=(12.0, 9.5),
        constrained_layout=True,
    )
    axes = axes_grid.ravel()
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
            vmin=np.log10(np.min(true_fraction)),
            vmax=np.log10(np.max(true_fraction)),
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        ax.plot(limits, limits, "k--", linewidth=1)
        ax.set(xscale="log", yscale="log", xlim=limits, ylim=limits)
        display_ticks = np.geomspace(limits[0], limits[1], 3)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(mticker.FixedLocator(display_ticks))
            axis.set_major_formatter(
                mticker.FuncFormatter(lambda value, _: f"{value:.2g}")
            )
            axis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel(f"Injected {label}", fontsize=12)
        ax.set_ylabel(f"Recovered {label}", fontsize=12)
        ax.tick_params(axis="both", which="both", labelsize=10)
        ax.grid(alpha=0.2, which="both")
    colorbar = fig.colorbar(points, ax=axes, pad=0.015)
    colorbar.set_label(r"$\log_{10}(\mathrm{BLR}/\mathrm{continuum})$", fontsize=12)
    colorbar.ax.tick_params(labelsize=10)
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
    if np.unique(true_tau_drw_rest).size > 1:
        print("Tau recovery by injected rest-frame timescale:")
        for injected_tau in np.unique(true_tau_drw_rest):
            selected_rows = [
                row
                for row in rows
                if row["true_tau_drw_rest"] == injected_tau
            ]
            recovered_tau = np.asarray(
                [row["tau_drw_rest_median"] for row in selected_rows]
            )
            print(
                f"  {injected_tau:g} d "
                f"(tau/baseline={injected_tau / time_baseline_rest:.3f}, "
                f"N={len(selected_rows)}): median fractional bias="
                f"{np.median(recovered_tau / injected_tau - 1):+.3f}, "
                f"68% coverage="
                f"{_coverage(selected_rows, 'true_tau_drw_rest', 'tau_drw_rest'):.2f}"
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
