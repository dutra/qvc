#!/usr/bin/env python3
"""
Single-object multiband light-curve fitter using the DHO+BLR model.

This follows the structure of `multiband_fit.py`, but swaps in the new
`make_multiband_dho_blr_model(...)` path where the latent driver is a shared
overdamped SHO with sigma fixed to 1 and band structure handled in the wrapper.
"""

import os
import sys
import argparse
import logging
import traceback

import numpy as np
from tqdm import tqdm

# ---------- CPU & threading hygiene ----------
num_cores = os.environ.get("NUM_CORES", os.cpu_count() - 2)
try:
    num_cores = int(num_cores)
except ValueError:
    print(f"Invalid NUM_CORES value '{num_cores}', ignoring.")
    num_cores = os.cpu_count() - 2

if __name__ == "__main__" and (os.environ.get("PYTHON_EXECUTION_CONTEXT") != "worker"):
    print(f"CPU Num Cores: {num_cores}")

os.environ["XLA_FLAGS"] = (
    f"--xla_force_host_platform_device_count={num_cores} "
    f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=1"
)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.pop("NUMEXPR_MAX_THREADS", None)
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

prefix = os.environ.get("PREFIX", "test")
suffix = os.environ.get("SUFFIX", "test")

# ---------- JAX/NumPyro ----------
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)
import jax.numpy as jnp
from jax import random, device_get
from jax.tree_util import tree_map

import numpyro

numpyro.set_host_device_count(num_cores)
numpyro.enable_x64()
numpyro.enable_validation(True)
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from multiband_fit import has_jitter, make_lc, zero_mean
from multiband_fit_plotting import *
from multiband_fit_utils import *
from multiband_generate_lc import *
from multiband_model_dho_blr import make_multiband_dho_blr_model


def build_explicit_model_params(raw_params, lam_rf):
    """Convert sampled high-level parameters into explicit model arrays."""

    lam_rf = jnp.asarray(lam_rf)
    lambda_ref = jnp.array(2500.0)

    log_sigma0 = jnp.asarray(raw_params["log_sigma0"])
    eta_A1 = jnp.asarray(raw_params["eta_A1"])
    eta_tau1 = jnp.asarray(raw_params["eta_tau1"])
    log_amp_delta_blr = jnp.asarray(raw_params["log_amp_delta_blr"])
    log_lag_blr = jnp.asarray(raw_params["log_lag_blr"])
    lag0 = jnp.asarray(raw_params["lag0"])
    lag_beta = jnp.asarray(raw_params["lag_beta"])
    log_tau_fast0 = jnp.asarray(raw_params["log_tau_fast0"])
    log_tau_drw0 = jnp.asarray(raw_params["log_tau_drw0"])

    log_sigma0_exp = jnp.expand_dims(log_sigma0, axis=-1)
    eta_A1_exp = jnp.expand_dims(eta_A1, axis=-1)
    eta_tau1_exp = jnp.expand_dims(eta_tau1, axis=-1)
    lag0_exp = jnp.expand_dims(lag0, axis=-1)
    lag_beta_exp = jnp.expand_dims(lag_beta, axis=-1)
    log_tau_fast0_exp = jnp.expand_dims(log_tau_fast0, axis=-1)
    log_tau_drw0_exp = jnp.expand_dims(log_tau_drw0, axis=-1)

    log_sigma_band = log_sigma0_exp + jnp.log(10.0) * log_single_pl(
        lam_rf,
        lambda_ref,
        eta_A1_exp,
    )

    amp_cont = jnp.exp(log_sigma_band)
    amp_blr = jnp.exp(log_sigma0_exp + log_amp_delta_blr)
    lag_disk = lag0_exp * (lam_rf / 2500.0) ** lag_beta_exp
    lag_blr = jnp.exp(log_lag_blr)
    log_tau_scale = jnp.log(10.0) * log_single_pl(
        lam_rf,
        lambda_ref,
        eta_tau1_exp,
    )
    log_tau_fast_band = log_tau_fast0_exp + log_tau_scale
    log_tau_slow_band = log_tau_drw0_exp + log_tau_scale
    log_kernel_param = jnp.concatenate([log_tau_fast_band, log_tau_slow_band], axis=-1)

    explicit = dict(raw_params)
    explicit["amp_cont"] = amp_cont
    explicit["amp_blr"] = amp_blr
    explicit["lag_disk"] = lag_disk
    explicit["lag_blr"] = lag_blr
    explicit["tau_fast_band"] = jnp.exp(log_tau_fast_band)
    explicit["tau_slow_band"] = jnp.exp(log_tau_slow_band)
    explicit["log_kernel_param"] = log_kernel_param
    return explicit


def add_model_prediction_params(samples, lam_rf):
    """Add explicit model parameters needed for prediction/plotting."""

    out = dict(samples)
    if all(key in out for key in ("log_kernel_param", "amp_cont", "amp_blr", "lag_disk", "lag_blr", "tau_fast_band", "tau_slow_band")):
        return out

    explicit = build_explicit_model_params(
        out,
        lam_rf,
    )
    out["log_kernel_param"] = np.asarray(explicit["log_kernel_param"])
    out["amp_cont"] = np.asarray(explicit["amp_cont"])
    out["amp_blr"] = np.asarray(explicit["amp_blr"])
    out["lag_disk"] = np.asarray(explicit["lag_disk"])
    out["lag_blr"] = np.asarray(explicit["lag_blr"])
    out["tau_fast_band"] = np.asarray(explicit["tau_fast_band"])
    out["tau_slow_band"] = np.asarray(explicit["tau_slow_band"])
    return out


def build_single_object_model(
    obj_dict,
    lam_rf,
    log_jitter_mean,
    *,
    disable_poly1=False,
    disable_lag_blr=False,
    sigma_tau_uniform=False,
    tau_fast_truncated=False,
):
    """Return the NumPyro model for one object."""

    (t, bidx) = obj_dict["X"]
    y = obj_dict["y"]
    yerr = obj_dict["yerr"]
    z = float(obj_dict["z"])
    B = int(len(lam_rf))

    log_tau_drw0_c = jnp.log(10**2.5 * (1.0 + z))

    def model():
        eta_A1 = numpyro.sample(
            "eta_A1",
            dist.TruncatedNormal(-0.5, 1.0, high=0.0),
        )

        eta_tau1 = numpyro.sample("eta_tau1", dist.Normal(0.5, 0.5))

        log_tau_drw0_high = jnp.log(10**4.0 * (1.0 + z))
        log_tau_drw0_low = 0.0
        if sigma_tau_uniform:
            log_tau_drw0 = numpyro.sample(
                "log_tau_drw0",
                dist.Uniform(log_tau_drw0_low, log_tau_drw0_high),
            )
        else:
            log_tau_drw0 = numpyro.sample(
                "log_tau_drw0",
                dist.TruncatedNormal(
                    log_tau_drw0_c,
                    1.2 * jnp.log(10),
                    low=log_tau_drw0_low,
                    high=log_tau_drw0_high,
                ),
            )

        log_tau_fast0_low = 0.0
        log_tau_fast0_high = jnp.log(100.0 * (1.0 + z))
        log_tau_fast0_c = jnp.log(10.0 * (1.0 + z))
        if tau_fast_truncated:
            log_tau_fast0 = numpyro.sample(
                "log_tau_fast0",
                dist.TruncatedNormal(
                    log_tau_fast0_c,
                    jnp.log(25.0),
                    low=log_tau_fast0_low,
                    high=log_tau_fast0_high,
                ),
            )
        else:
            log_tau_fast0 = numpyro.sample(
                "log_tau_fast0",
                dist.Normal(log_tau_fast0_c, jnp.log(25.0)),
            )

        if sigma_tau_uniform:
            log_sigma0 = numpyro.sample(
                "log_sigma0",
                dist.Uniform(-2.0 * jnp.log(10), 0.2 * jnp.log(10)),
            )
        else:
            log_sigma0 = numpyro.sample(
                "log_sigma0",
                dist.Normal(-0.6 * jnp.log(10), 1.0 * jnp.log(10)),
            )
        log_sigma_hat0 = numpyro.deterministic(
            "log_sigma_hat0",
            log_sigma0 - 0.5 * log_tau_drw0,
        )

        if disable_poly1:
            poly1 = numpyro.deterministic("poly1", 0.0)
        else:
            poly1 = numpyro.sample(
                "poly1",
                dist.Normal(0.0, 0.1),
            )

        lag0 = numpyro.sample("lag0", dist.TruncatedNormal(5.0, 5.0, low=0.0))
        lag_beta = numpyro.sample(
            "lag_beta",
            dist.TruncatedNormal(4.0 / 3.0, 0.2, low=0.0),
        )

        with numpyro.plate("band", B):
            mean = numpyro.sample("mean", dist.Normal(jnp.zeros(B), 0.2))

            if disable_lag_blr:
                log_amp_delta_blr = numpyro.deterministic(
                    "log_amp_delta_blr",
                    jnp.full(B, -1e9),
                )
                log_lag_blr = numpyro.deterministic(
                    "log_lag_blr",
                    jnp.full(B, -9.0),
                )
            else:
                log_amp_delta_blr = numpyro.sample(
                    "log_amp_delta_blr",
                    dist.Normal(jnp.full(B, -1.0), 1.0),
                )
                log_lag_blr = numpyro.sample(
                    "log_lag_blr",
                    dist.Uniform(jnp.log(2.0), jnp.log(5000.0)),
                )

            log_jitter = numpyro.sample(
                "log_jitter",
                dist.Normal(log_jitter_mean, 1.0),
            )

        _ = numpyro.deterministic("log_tau_fake", float(obj_dict.get("log_tau_fake", -99.0)))
        _ = numpyro.deterministic("log_sigma_fake", float(obj_dict.get("log_sigma_fake", -99.0)))

        raw_params = dict(
            log_tau_drw0=log_tau_drw0,
            log_tau_fast0=log_tau_fast0,
            log_sigma0=log_sigma0,
            poly1=poly1,
            mean=mean,
            log_amp_delta_blr=log_amp_delta_blr,
            log_lag_blr=log_lag_blr,
            log_jitter=log_jitter,
            lag0=lag0,
            lag_beta=lag_beta,
            eta_A1=eta_A1,
            eta_tau1=eta_tau1,
        )

        params = build_explicit_model_params(
            raw_params,
            lam_rf,
        )

        numpyro.deterministic("tau_fast", params["tau_fast_band"])
        numpyro.deterministic("tau_slow", params["tau_slow_band"])
        numpyro.deterministic("amp_cont", params["amp_cont"])
        numpyro.deterministic("amp_blr", params["amp_blr"])
        numpyro.deterministic("lag_disk", params["lag_disk"])
        numpyro.deterministic("lag_blr", params["lag_blr"])

        m = make_multiband_dho_blr_model(
            X=(t, bidx),
            y=y,
            yerr=yerr,
            n_band=B,
            zero_mean=zero_mean,
            has_jitter=has_jitter,
        )
        numpyro.factor("loglike", m.log_prob(params))

    return model


def main():
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Fit quasars one-by-one with the DHO+BLR light-curve model.")
    parser.add_argument("--filter_object_id", nargs="+", help="List of object IDs to filter.")
    parser.add_argument("--N", type=int, help="Number of objects to process.")
    parser.add_argument("--skip", type=int, help="Number of objects to skip.")
    parser.add_argument("--filter_file", type=str, help="Path to file containing object IDs.")
    parser.add_argument("--plot", action="store_true", help="Enable plotting of results.")
    parser.add_argument("--progress", action="store_true", help="Show progress bar.")
    parser.add_argument("--nwarm", type=int, default=500, help="Warmup steps for MCMC.")
    parser.add_argument("--nsamp", type=int, default=250, help="Samples per chain for MCMC.")
    parser.add_argument("--nchains", type=int, default=2, help="Number of chains (>=1).")
    parser.add_argument("--inject_fake", action="store_true", help="Inject fake light curves.")
    parser.add_argument("--bwb", action="store_true", help="Accepted for compatibility; ignored in the DHO model.")
    parser.add_argument("--max_tree_depth", type=int, default=8, help="NUTS max tree depth.")
    parser.add_argument("--load_sample_file", action="store_true", help="Load saved samples (debug).")
    parser.add_argument("--disable_poly1", action="store_true", help="Disable trend.")
    parser.add_argument("--rf_length_cut", type=int, default=-1, help="Rest-frame cut (days).")
    parser.add_argument("--exact_same_length", action="store_true", help="Exact same RF length cut.")
    parser.add_argument("--load_stone_lcs", action="store_true", default=False, help="Use Stone LCs.")
    parser.add_argument("--disable_corner_plot", action="store_true", default=False, help="Disable corner plot.")
    parser.add_argument("--disable_lag_blr", action="store_true", default=False, help="Disable BLR lag model.")
    parser.add_argument("--sigma_tau_uniform", action="store_true", default=False, help="Uniform priors for sigma/tau.")
    parser.add_argument("--disable_plot_psd", action="store_true", default=False, help="Accepted for compatibility; PSD plotting is disabled for this model.")
    parser.add_argument("--inject_random_fake_etas", action="store_true", default=False, help="Randomize fake etas.")
    parser.add_argument("--beta_tau", type=float, default=0.2, help="beta_tau for fake curves.")
    parser.add_argument("--disable_band_drop", action="store_true", default=False, help="Disable Lya band drop.")
    parser.add_argument("--load_nearby_lc_csv", type=str, default=None, help="CSV listing nearby LCs to load.")
    parser.add_argument("--tau_fast_truncated", action="store_true", default=False, help="Truncated prior for tau_fast0.")
    args = parser.parse_args()
    print("Args:", args)

    if args.bwb:
        logging.info("Ignoring --bwb for fit_light_curves.py; the DHO wrapper does not use the old BWB mixture.")

    if args.load_stone_lcs:
        objs = load_stone_lcs(filter_object_ids=args.filter_object_id)
        print(f"Loaded {len(objs)} Stone light curves.")
    elif args.load_nearby_lc_csv is not None:
        objs = load_nearby_lcs(args.load_nearby_lc_csv)
        print(f"Loaded {len(objs)} nearby light curves from {args.load_nearby_lc_csv}.")
    else:
        objs = concat_light_curves(
            filter_object_ids=args.filter_object_id,
            progress_bar=args.progress,
            N=args.N,
            skip=args.skip,
        )
    print(f"Loaded {len(objs)} objects.")

    objs = populate_sdss_fields(objs, progress_bar=args.progress)
    if args.rf_length_cut > 0:
        objs = cut_light_curve_restframe_window(
            objs,
            n_days=args.rf_length_cut,
            same_length=args.exact_same_length,
        )
        print(f"After restframe cut, {len(objs)} objects remain.")

    if args.inject_random_fake_etas:
        rng = np.random.default_rng()
        alpha_sigma = float(rng.uniform(-1.0, 0.0))
        beta_tau = float(rng.uniform(-0.5, 2.0))
        print(f"Randomized alpha_sigma={alpha_sigma:.3f}, beta_tau={beta_tau:.3f}")
    else:
        alpha_sigma = -0.5
        beta_tau = float(args.beta_tau)
        print(f"Using fixed alpha_sigma={alpha_sigma:.3f}, beta_tau={beta_tau:.3f}")

    results = []
    chain_method = "parallel" if args.nchains and args.nchains > 1 else "sequential"

    iterator = tqdm(objs, desc="Fitting", disable=not args.progress)
    for idx, obj in enumerate(iterator):
        oid = str(obj["object_id"])
        try:
            default_bands = ["u", "g", "r", "i", "z"]
            if args.load_stone_lcs or args.load_nearby_lc_csv is not None:
                default_bands = ["g", "r", "i"]

            lc = make_lc(
                None,
                obj,
                bands=default_bands,
                inject_fake=args.inject_fake,
                alpha_sigma=alpha_sigma,
                beta_tau=beta_tau,
                disable_band_drop=args.disable_band_drop,
            )
            if lc is None:
                continue

            obj |= lc

            bands = obj["bands"]
            lam_rf = jnp.array([lambda_pivot[b] for b in bands], dtype=float) / (1.0 + float(obj["z"]))
            print(f"[{oid}] Using bands: {bands}")
            print(f"[{oid}] lam_rf = {lam_rf}")

            bidx = obj["band_idx"]
            yerr = np.asarray(obj["yerr"])
            B = len(bands)
            ljm = np.empty(B)
            for i in range(B):
                m = (bidx == i) & np.isfinite(yerr) & (yerr < 10)
                ljm[i] = np.log(np.mean(yerr[m])) if np.any(m) else np.log(1e-3)
            log_jitter_mean = jnp.array(ljm)

            numpyro_model = build_single_object_model(
                obj,
                lam_rf,
                log_jitter_mean=log_jitter_mean,
                disable_poly1=args.disable_poly1,
                disable_lag_blr=args.disable_lag_blr,
                sigma_tau_uniform=args.sigma_tau_uniform,
                tau_fast_truncated=args.tau_fast_truncated,
            )

            init_strategy = numpyro.infer.init_to_median()
            nuts = NUTS(
                numpyro_model,
                init_strategy=init_strategy,
                dense_mass=True,
                max_tree_depth=args.max_tree_depth,
                target_accept_prob=0.9,
            )
            mcmc = MCMC(
                nuts,
                num_warmup=args.nwarm,
                num_samples=args.nsamp,
                num_chains=max(1, args.nchains),
                chain_method=chain_method,
                progress_bar=args.progress,
            )

            if args.load_sample_file:
                logging.warning("[DEBUG] Loading saved samples (flat) — developer mode.")
                obj_flat_samples = load_obj_samples_from_hdf5(oid)
                samples_per_chain = None
            else:
                key = random.PRNGKey(0)
                key = random.fold_in(key, idx)
                mcmc.run(key)
                samples_flat = mcmc.get_samples(group_by_chain=False)
                samples_per_chain = mcmc.get_samples(group_by_chain=True)
                samples_flat = tree_map(lambda x: np.asarray(device_get(x)), samples_flat)
                samples_per_chain = tree_map(lambda x: np.asarray(device_get(x)), samples_per_chain)
                obj_flat_samples = samples_flat
                save_obj_samples_to_hdf5(obj_flat_samples, oid)

            obj_flat_samples_flatten_per_band = flatten_flat_samples_per_band(
                obj_flat_samples,
                bands=bands,
            )

            diagnostics = {}
            if samples_per_chain is not None:
                obj_samples_per_chain_flatten_per_band = flatten_per_chain_samples_per_band(
                    samples_per_chain,
                    bands=bands,
                )
                diagnostics = diagnostics_for_per_chain_samples(obj_samples_per_chain_flatten_per_band)

            result = process_samples(
                obj_flat_samples_flatten_per_band,
                obj,
                broken_pl=False,
                bands=bands,
            )

            if args.plot:
                try:
                    plot_mcmc_traces(obj_flat_samples_flatten_per_band, obj)
                    m = make_multiband_dho_blr_model(
                        obj["X"],
                        obj["y"],
                        obj["yerr"],
                        n_band=B,
                        zero_mean=zero_mean,
                        has_jitter=has_jitter,
                    )
                    plot_samples = add_model_prediction_params(
                        obj_flat_samples,
                        lam_rf,
                    )
                    save_combined_plot(
                        plot_samples,
                        m,
                        obj["X"],
                        obj["y"],
                        obj["yerr"],
                        obj["band_idx"],
                        obj["mags_means"],
                        obj["survey_times"],
                        result,
                        time0=obj["time0"],
                        bands=bands,
                        plot_psd=False,
                    )
                    plot_correlation_matrix(obj_flat_samples_flatten_per_band, obj)
                    plot_all_histograms(obj_flat_samples_flatten_per_band, obj)
                    if not args.disable_corner_plot:
                        plot_posterior_fast(obj_flat_samples_flatten_per_band, obj)
                except Exception as e:
                    logging.error(f"[{oid}] Plotting error: {e}")
                    logging.error(traceback.format_exc())

            final_result = obj | diagnostics | result | dict(prefix=prefix, suffix=suffix)
            log_sigma_UV = final_result.get("log_sigma_UV")
            log_sigma_UV_err = final_result.get("log_sigma_UV_err")
            log_tau_UV_RF = final_result.get("log_tau_UV_RF")
            log_tau_UV_RF_err = final_result.get("log_tau_UV_RF_err")
            print(
                f"[{oid}] log_sigma_UV = {log_sigma_UV} ± {log_sigma_UV_err} ; "
                f"log_tau_UV_RF = {log_tau_UV_RF} ± {log_tau_UV_RF_err}"
            )

            results.append(final_result)

            if args.inject_fake:
                compare_pairs = [
                    ("log_tau_fake", "log_tau_drw0", "log10_tau"),
                    ("log_sigma_fake", "log_sigma0", "log10_sigma"),
                ]
                summarize_fake_true_vs_recovered(final_result, diagnostics, compare_pairs=compare_pairs)

        except Exception as e:
            logging.error(f"[{oid}] Error during fit: {e}")
            logging.error(traceback.format_exc())
            continue

    save_quasar_list_hdf5(results, ignored_keys=["X", "y", "yerr", "band_idx"])

    try:
        plot_sigma_tau_vs_lambda_with_model(
            results,
            inject_fake=args.inject_fake,
            broken_pl=False,
        )
    except Exception as e:
        logging.error(f"plot_sigma_tau_vs_lambda_with_model error: {e}")
        logging.error(traceback.format_exc())

    if args.inject_fake:
        try:
            plot_recovery(results)
        except Exception as e:
            logging.error(f"plot_recovery error: {e}")
            logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
