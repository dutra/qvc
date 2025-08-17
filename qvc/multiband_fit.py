import os

env_cores = os.environ.get("NUM_CORES")

if env_cores is not None:
        try:
            num_cores = int(env_cores)
            print(f"CPU Num Cores: {num_cores}")
            os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_cores}"
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
        except ValueError:
            print(f"Invalid NUM_CORES value '{env_cores}', ignoring.")
else:
    print("NUM_CORES not set, leaving defaults.")


import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import numpyro

if env_cores is not None:
    num_cores = int(env_cores)
    numpyro.set_host_device_count(num_cores)  # Tell NumPyro how many to use

numpyro.enable_x64()

from numpyro import infer
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist

from tinygp import kernels

import warnings
#warnings.filterwarnings("ignore", category=RuntimeWarning)

import logging

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

from multiband_fit_utils import *
from multiband_fit_plotting import *
from multiband_generate_lc import *
from multiband_models import *

# define params
zero_mean = False
has_jitter = True
has_lag = True

universal_params = ['eta_A1_mean', 'eta_A2_mean', 'eta_tau1_mean', 'eta_tau2_mean', 'eta_break', 'lam_s', 'sigma_eta_A1', 'sigma_eta_A2', 'sigma_eta_tau1', 'sigma_eta_tau2', 'log_sigma_eta_A1', 'log_sigma_eta_A2', 'log_sigma_eta_tau1', 'log_sigma_eta_tau2']

def build_model(batch_data, zs, f_host_value, lam_rfs, log_jitter_mean, f_host_shen11=True, latent=False, bwb=True, disable_poly1=False, d_eta=True):
    # Precompute and capture constants in the closure so they are treated as
    # static by JAX/NumPyro. This prevents unnecessary retracing/recompilation
    # when running MCMC, as these values do not change between runs.
    batch_size = len(batch_data)
    nBands = 5  # or use from config

    log_tau_drw0_c = jnp.log(10**2.5 * (1 + zs))
    log_lab_blr_c = jnp.log(10**1.5 * (1 + zs))

    def numpyro_joint_model():
        # Initialize parameters
        # Global "universal" means for eta
        eta_A1_mean = numpyro.sample("eta_A1_mean", dist.TruncatedNormal(-0.5, 0.2, high=0.0))
        eta_A2_mean = numpyro.sample("eta_A2_mean", dist.TruncatedNormal(-0.5, 0.2, high=0.0))
        eta_tau1_mean = numpyro.sample("eta_tau1_mean", dist.TruncatedNormal(-0.5, 0.2))
        eta_tau2_mean = numpyro.sample("eta_tau2_mean", dist.TruncatedNormal(0.1, 0.2, low=0.0))
        eta_break = numpyro.deterministic("eta_break", 0.1)
        #lam_s = numpyro.sample("lam_s", dist.Normal(2500.0, 100.0)) # Hard to constrain
        lam_s = numpyro.deterministic("lam_s", 2500.0)

        # Population-level scatter (how much objects can deviate) 
        log_sigma_eta_A1 = numpyro.sample("log_sigma_eta_A1", dist.Normal(jnp.log(0.1), 0.1))
        log_sigma_eta_A2 = numpyro.sample("log_sigma_eta_A2", dist.Normal(jnp.log(0.1), 0.1))
        log_sigma_eta_tau1 = numpyro.sample("log_sigma_eta_tau1", dist.Normal(jnp.log(0.1), 0.1))
        log_sigma_eta_tau2 = numpyro.sample("log_sigma_eta_tau2", dist.Normal(jnp.log(0.1), 0.1))

        sigma_eta_A1 = numpyro.deterministic("sigma_eta_A1", jnp.exp(log_sigma_eta_A1))
        sigma_eta_A2 = numpyro.deterministic("sigma_eta_A2", jnp.exp(log_sigma_eta_A2))
        sigma_eta_tau1 = numpyro.deterministic("sigma_eta_tau1", jnp.exp(log_sigma_eta_tau1))
        sigma_eta_tau2 = numpyro.deterministic("sigma_eta_tau2", jnp.exp(log_sigma_eta_tau2))

        with numpyro.plate("objects", batch_size):
            # Object-level parameters (shape: [B])
            # Variability k-corrections
            if d_eta:
                eta_A1 = numpyro.sample("eta_A1", dist.Normal(eta_A1_mean, sigma_eta_A1))
                eta_A2 = numpyro.sample("eta_A2", dist.Normal(eta_A2_mean, sigma_eta_A2))
                eta_tau1 = numpyro.sample("eta_tau1", dist.Normal(eta_tau1_mean, sigma_eta_tau1))
                eta_tau2 = numpyro.sample("eta_tau2", dist.Normal(eta_tau2_mean, sigma_eta_tau2))
            # Or, use deterministic to set them to the universal means
            else:
                eta_A1 = numpyro.deterministic("eta_A1", jnp.full(batch_size, eta_A1_mean))
                eta_A2 = numpyro.deterministic("eta_A2", jnp.full(batch_size, eta_A2_mean))
                eta_tau1 = numpyro.deterministic("eta_tau1", jnp.full(batch_size, eta_tau1_mean))
                eta_tau2 = numpyro.deterministic("eta_tau2", jnp.full(batch_size, eta_tau2_mean))

            # Core kernel parameters
            log_tau_drw0 = numpyro.sample("log_tau_drw0", dist.Normal(log_tau_drw0_c, 2.0))
            log_sigma0 = numpyro.sample("log_sigma0", dist.Normal(-0.8, 1.0))
            log_sigma_hat0 = numpyro.deterministic("log_sigma_hat0", log_sigma0 - 0.5 * log_tau_drw0)

            # Host galaxy dilution
            alpha_host = numpyro.sample("alpha_host", dist.Normal(1.0, 0.1))
            f_host = numpyro.deterministic("f_host", f_host_value)

            # Mean function detrending
            if disable_poly1:
                poly1 = numpyro.deterministic("poly1", jnp.zeros_like(log_sigma0))                
            else:
                poly1 = numpyro.sample("poly1", dist.Normal(0.0, 0.1))

            # Disk lags
            lag0 = numpyro.sample("lag0", dist.TruncatedNormal(10.0, 5.0, low=0))
            lag_beta = numpyro.sample("lag_beta", dist.TruncatedNormal(4/3, 0.2, low=0))

            # Bluer when brighter (BWB) strength
            if bwb:
                bwb_alpha = numpyro.sample("bwb_alpha", dist.Normal(0.2, 0.2))
                bwb_beta = numpyro.sample("bwb_beta", dist.TruncatedNormal(0.2, 0.4, low=0))
                gamma = numpyro.sample("gamma", dist.Normal(2.0, 1.0))
            else:
                bwb_alpha = numpyro.deterministic("bwb_alpha", jnp.zeros(batch_size))
                bwb_beta = numpyro.deterministic("bwb_beta", jnp.zeros(batch_size))
                gamma = numpyro.deterministic("gamma", jnp.ones(batch_size) * 2.0)

        with numpyro.plate("objects", batch_size, dim=-2):
            with numpyro.plate("band", nBands, dim=-1):
                # Parameters with shape [B, nBands]
                # Means in each band
                mean = numpyro.sample("mean", dist.Normal(jnp.full(nBands, 0.0), 0.2))

                # BLR amplitudes and lags
                log_amp_delta_blr = numpyro.sample("log_amp_delta_blr", dist.Normal(jnp.full(nBands, -1.0), 1.0))
                log_lag_blr = numpyro.sample("log_lag_blr", dist.Normal(log_lab_blr_c[..., None], 3.0))
                #log_lag_blr = numpyro.deterministic("log_lag_blr", jnp.zeros_like(mean))

                # Jitter
                log_jitter = numpyro.sample("log_jitter", dist.Normal(log_jitter_mean, 1.0))

        def run_batch(obj, i):
            # Collect params for object i
            params = {
                "log_tau_drw0": log_tau_drw0[i],
                "log_sigma0": log_sigma0[i],
                "alpha_host": alpha_host[i],
                "f_host": f_host[i],
                "poly1": poly1[i],
                "mean": mean[i],
                "log_amp_delta_blr": log_amp_delta_blr[i],
                "log_lag_blr": log_lag_blr[i],
                "log_jitter": log_jitter[i],
                "lag0": lag0[i],
                "lag_beta": lag_beta[i],
                "bwb_alpha": bwb_alpha[i],
                "bwb_beta": bwb_beta[i],
                "gamma": gamma[i],
                # power law
                "eta_A1": eta_A1[i],
                "eta_A2": eta_A2[i],
                "eta_tau1": eta_tau1[i],
                "eta_tau2": eta_tau2[i],
                "eta_break": eta_break,
                "lam_s": lam_s
            }

            m = Model(
                X=(obj[:,0], obj[:,1]), y=obj[:,2], yerr=obj[:,3],
                kernel=kernels.quasisep.Exp(jnp.array([1, 1])),
                zero_mean=zero_mean, has_jitter=has_jitter, has_lag=has_lag,
                lam_rf=lam_rfs[i], z=zs[i]
            )

            return m.log_prob(params)

        log_probs = jax.vmap(run_batch, in_axes=(0, 0))(batch_data, jnp.arange(batch_size))
        log_probs = jnp.where(jnp.isfinite(log_probs), log_probs, -1e20)
        numpyro.factor("loglike", log_probs.sum())
    
    return numpyro_joint_model


def make_lc(Model, data, filter_red_bands=False):
    times = data['times']
    mags = data['mags']
    magerrs = data['magerrs']

    blue_bands = bands_bluer_than_lyman_alpha(data['z'])

    if filter_red_bands:
        red_bands = bands_redder_than(data['z'], threshold=4000)
        clean_bands = list(set(bands) - set(blue_bands) - set(red_bands))
        logging.info(
            f"Filtering out red bands (wavelength > 4000 Å) {red_bands} "
            f"for quasar {data['object_id']} at z={data['z']}"
        )
    else:
        logging.info(
            f"Excluding only blue bands {blue_bands} "
            f"for quasar {data['object_id']} at z={data['z']}"
        )
        clean_bands = list(set(bands) - set(blue_bands))

    # Sort clean_bands in desired photometric order
    clean_bands = list(sorted(clean_bands, key=lambda b: ['u', 'g', 'r', 'i', 'z', 'y'].index(b)))
    data['clean_bands'] = clean_bands

    if len(clean_bands) == 0:
        print(f"No clean bands for quasar {data['object_id']}, skipping.", flush=True)
        return None

    # Combine data across bands
    all_times = np.concatenate([times[b] for b in clean_bands])
    all_mags = np.concatenate([mags[b] for b in clean_bands])
    all_magerrs = np.concatenate([magerrs[b] for b in clean_bands])
    band_idx = np.concatenate([np.full(len(times[b]), i) for i, b in enumerate(clean_bands)])

    if len(all_times) == 0:
        print(f"No magnitudes for quasar {data['object_id']}, skipping.", flush=True)
        return None

    # Sort by time
    sort_idx = np.argsort(all_times)
    all_times, all_mags, all_magerrs, band_idx = (
        all_times[sort_idx],
        all_mags[sort_idx],
        all_magerrs[sort_idx],
        band_idx[sort_idx]
    )

    # Remove NaNs
    mask = np.isfinite(all_mags) & np.isfinite(all_magerrs) & np.isfinite(all_times)
    all_times, all_mags, all_magerrs, band_idx = (
        all_times[mask],
        all_mags[mask],
        all_magerrs[mask],
        band_idx[mask]
    )

    if len(all_times) == 0:
        print(f"No finite magnitudes for quasar {data['object_id']}, skipping.", flush=True)
        return None

    # --- Outlier rejection ---
    window_size = 6
    mask_outlier = np.ones(len(all_times), dtype=bool)

    from numpy.lib.stride_tricks import sliding_window_view
    from scipy.stats import median_abs_deviation

    for band in np.unique(band_idx):
        band_mask = band_idx == band
        idx_band = np.where(band_mask)[0]
        band_y = all_mags[band_mask]

        if len(band_y) < 2 * window_size + 1:
            continue

        windows = sliding_window_view(band_y, 2 * window_size + 1)
        centers = band_y[window_size:-window_size]
        medians = np.nanmean(windows, axis=1)
        mads = median_abs_deviation(windows, axis=1)

        is_outlier = np.abs(centers - medians) > 2.5 * mads
        mask_outlier[idx_band[window_size:-window_size][is_outlier]] = False

    # Apply outlier mask
    all_times = all_times[mask_outlier]
    all_mags = all_mags[mask_outlier]
    all_magerrs = all_magerrs[mask_outlier]
    band_idx = band_idx[mask_outlier]

    # --- Center magnitudes per band AFTER outlier rejection ---
    mags_means = np.array([
        np.nanmean(all_mags[band_idx == i]) for i in range(len(clean_bands))
    ])
    mags_stds = np.array([
        np.nanstd(all_mags[band_idx == i]) for i in range(len(clean_bands))
    ])
    
    for i in range(len(clean_bands)):
        band_mask = band_idx == i
        all_mags[band_mask] -= np.nanmean(all_mags[band_mask])

    # Define arrays for model
    X = (
        jnp.array(all_times) - jnp.min(all_times),
        jnp.array(band_idx)
    )
    y = jnp.array(all_mags)
    yerr = jnp.array(all_magerrs)
    t = jnp.array(all_times)

    batch_dict = {
        'X': X,
        'y': y,
        'yerr': yerr,
        'clean_bands': clean_bands,
        'z': data['z'],
        'band_idx': band_idx,
        'mags_means': mags_means,
        'mags_stds': mags_stds
    }
    return batch_dict

                    
if __name__ == '__main__': 
    
    logging.info("Starting multiband fit")

    parser = argparse.ArgumentParser(description="Process quasars with optional filtering.")
    parser.add_argument("--filter_object_id", nargs="+", help="List of object IDs to filter.")
    parser.add_argument("--N", type=int, help="Number of objects to process.")
    parser.add_argument("--skip", type=int, help="Number of objects to skip.")
    parser.add_argument("--chunk_size", type=int, default=500, help="Chunk size for processing objects.")
    parser.add_argument("--lc_file", type=str, help="Path to the light curve file.")
    parser.add_argument("--filter_file", type=str, help="Path to the file containing object IDs to filter.")
    parser.add_argument("--plot", action="store_true", help="Enable plotting of results.")
    parser.add_argument("--svi", action="store_true", help="Use stochastic variation inference (SVI).")
    parser.add_argument("--ignore_existing", action="store_true", help="Ignore sources already in the HDF5 file.")
    parser.add_argument("--create_lc", action="store_true", help="Only create LC file and exit.")
    parser.add_argument("--progress", action="store_true", help="Show progress bar.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU.")
    parser.add_argument("--nwarm", type=int, default=500, help="Number of warmup steps for MCMC.")
    parser.add_argument("--nsamp", type=int, default=250, help="Number of samples for MCMC.")
    parser.add_argument("--nchains", type=int, default=-1, help="Number of chains for MCMC.")
    parser.add_argument("--latent", action="store_true", help="Use latent variable model.")
    parser.add_argument("--bwb", action="store_true", help="Use BWB model.")
    parser.add_argument("--d_eta", action="store_true", help="Vary eta for each quasar with prior.")
    parser.add_argument("--choose_N", type=int, default=-1, help="Sample choose_N objects.")
    parser.add_argument("--job_id", type=int, default=-1, help="Job Index for parallel processing.")
    parser.add_argument("--job_N", type=int, default=-1, help="Number of objects to divide.")
    parser.add_argument("--max_tree_depth", type=int, default=8, help="Max tree depth param for NUTS sampler.")
    parser.add_argument("--f_host_shen11", action="store_true", help="Use host flux empirical relation from Shen et al. 2011.")
    parser.add_argument("--load_sample_file", action="store_true", help="Load samples from previously ran job.")
    parser.add_argument("--disable_poly1", action="store_true", help="Disable Mean function detrending.")
    parser.add_argument("--jax_trace", action="store_true", help="Enable jax tracing.")


    args = parser.parse_args()
    print("Args: ", args)

    check_64bit(gpu=not bool(args.cpu))

    if args.create_lc:
        objs = concat_light_curves(save_file_path=args.lc_file, progress_bar=args.progress)
        sys.exit("Created LC file. Exiting the program as requested.")


    filter_object_ids = args.filter_object_id if args.filter_object_id else []
    print(f"filter_object_ids: {filter_object_ids}")
    filter_object_ids = pd.read_csv(args.filter_file, dtype={"object_id": str})["object_id"].values if (args.filter_file and (not args.filter_object_id)) else filter_object_ids
    print(f"Loaded {len(filter_object_ids)=}")
    if args.choose_N > 0:
        filter_object_ids = np.random.choice(filter_object_ids, size=args.choose_N, replace=False)
        print(f"After choosing, total of {len(filter_object_ids)=}")

    elif args.job_id > -1:
        subarrays = [filter_object_ids[i:i + args.job_N] for i in range(0, len(filter_object_ids), args.job_N)]
        filter_object_ids = subarrays[args.job_id]
        print(f"Job ID {args.job_id} processing {filter_object_ids=}")

    if len(filter_object_ids) > 0:
        print(f"Filtering object IDs: {len(filter_object_ids)}")

    objs = concat_light_curves(filter_object_ids=filter_object_ids, N=args.N, skip=args.skip, save_file_path=args.lc_file, progress_bar=args.progress)
    if args.create_lc:
        sys.exit("Created LC file. Exiting the program as requested.")
    print(f"Loaded {len(objs)} objects from concat_light_curves")

    #objs = populate_sdss_fields(objs)

    Model = MyMultiVarModel
    if args.latent:
        print("Using latent model (with BLR contribution)")
        Model = MyMultiVarModelLatent

    # After loading objs
    logging.info("--- Joint fitting ---")
    batch_data = []
    for i, obj in enumerate(objs):
        # Prepare each object's data for the joint model
        result = make_lc(Model, obj, filter_red_bands=(not args.f_host_shen11))
        if result is None:
            continue
        obj['i'] = i
        obj |= result
        # Run bestP for each object
        n_bands = len(obj['clean_bands'])
        lam_rf = np.full(5, 2500.0)
        lam_rf[:len(obj['clean_bands'])] = np.array([lambda_pivot[band] for band in obj['clean_bands']]) / (1 + obj['z'])
        lam_rf = jnp.array(lam_rf)

        batch_data.append({
            'object_id': obj['object_id'],
            'X': obj['X'],
            'y': obj['y'],
            'yerr': obj['yerr'],
            'clean_bands': obj['clean_bands'],
            'band_idx': obj['band_idx'],
            'z': obj['z'],
            # add any other fields needed by your model
            'LOGLBOL': obj['LOGLBOL'],
            'mags_means': obj['mags_means'],
            'mags_stds': obj['mags_stds'],
            'lam_rf': lam_rf
        })

    num_objects = len(batch_data)
    logging.info(f"Running joint fit on {len(batch_data)} objects...")


    padded_batch_data = pad_batch(batch_data, nBands=5)

    # Set up
    estimated_nchains = 4
    if args.nchains < 1:
        nchains = estimated_nchains
    else:
        nchains = args.nchains
    print(f"{args.max_tree_depth=}, {args.nwarm=}, {args.nsamp=}, {args.nchains=}, default num_chains: {estimated_nchains}")
    
    init_strategy = numpyro.infer.init_to_sample()
    #init_strategy = numpyro.infer.init_to_median()
    logging.info("Done with numpyro.infer.init_to_sample")

    # --- Precompute log_jitter prior means ---
    zs = jnp.array([obj['z'] for obj in batch_data])
    lam_rfs = jnp.array([obj['lam_rf'] for obj in batch_data])
    log_jitter_mean = jnp.stack([
        jnp.array(jnp.full(5, 1e-6) + jnp.log(jnp.mean(jnp.array(obj[:,3][obj[:,3] < 10])))) for obj in padded_batch_data
    ])  # shape (B, nBands)

    # --- Precompute f_host_shen11 prior means ---
    if args.f_host_shen11:
        # Host flux empirical relation
        #logl5100 = jnp.array([obj['LOGL5100'] for obj in batch_data])
        logl5100 = jnp.array([obj['LOGLBOL'] - jnp.log10(9.26) for obj in batch_data])

        x = logl5100 - 44.0
        f_host = 0.8052 - 1.5502 * x + 0.9121 * jnp.power(x, 2) - 0.1577 * jnp.power(x, 3)
        f_host = jnp.clip(f_host, 0.0, None)
        f_host_value = jnp.where(logl5100 < 45.053, f_host, 0.0)
    else:
        batch_size = len(batch_data)
        f_host_value = jnp.zeros(batch_size)

    numpyro_joint_model = build_model(padded_batch_data, zs, f_host_value, lam_rfs, log_jitter_mean, args.f_host_shen11, args.latent, args.bwb, args.disable_poly1, args.d_eta)


    nuts_kernel = NUTS(numpyro_joint_model, init_strategy=init_strategy, dense_mass=True, max_tree_depth=args.max_tree_depth)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=args.nwarm,
        num_samples=args.nsamp,
        num_chains=nchains,
        progress_bar=args.progress,
        chain_method="parallel",
    )

    if args.load_sample_file:
        logging.warning(f"Loading samples from saved file")
        samples_flat = load_all_samples_from_hdf5()
    else: # run MCMC sampler
        if args.jax_trace:
            from jax.profiler import StepTraceAnnotation, start_trace, stop_trace
            # 1) Tiny compile pass OFF-trace (same chain_method/nchains to match shapes)
            compile_mcmc = MCMC(
                NUTS(numpyro_joint_model, init_strategy=init_strategy,
                    dense_mass=True, max_tree_depth=args.max_tree_depth),
                num_warmup=5,
                num_samples=1,
                num_chains=1,
                progress_bar=False,
                chain_method="vectorized",
            )
            compile_mcmc.run(jax.random.PRNGKey(0))  # triggers JIT compile

            # 2) Real run ON-trace
            trace_dir = getattr(args, "jax_trace_dir", "./jax_tb")
            start_trace(trace_dir)
            with StepTraceAnnotation("MCMC_run"):
                mcmc.run(jax.random.PRNGKey(1))
            stop_trace()
        else:
            # Plain run, no tracing
            mcmc.run(jax.random.PRNGKey(0))
        samples_flat = mcmc.get_samples(group_by_chain=False)
        save_all_samples_to_hdf5(samples_flat)
    
    #ns = NestedSampler(numpyro_joint_model)
    #ns.run(jax.random.PRNGKey(0), Model, batch_data)
    #samples_flat = ns.get_samples(jax.random.PRNGKey(0), num_samples=1000)

    logging.info("Done with MCMC run")

    # Save and plot the results
    results = []
    for i, obj in enumerate(batch_data):
        logging.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        logging.info(f"Quasar {i+1}/{len(batch_data)} Object ID: {obj['object_id']}")

        obj_flat_samples = select_samples_for_object(samples_flat, i, universal_params=universal_params)
        obj_flat_samples_flatten_per_band = flatten_flat_samples_per_band(obj_flat_samples, obj['clean_bands'])

        save_obj_samples_to_hdf5(obj_flat_samples_flatten_per_band, obj['object_id'])
        
        # Add the object-specific parameters
        result = process_samples(obj_flat_samples_flatten_per_band, obj)

        #samples_grouped = mcmc.get_samples(group_by_chain=True)
        #samples_grouped_cleaned = clean_grouped_samples(samples_grouped)
        #rhat_ess = compute_rhat_ess_dict(samples_grouped_cleaned)

        # Plotting
        if args.plot:
            plot_mcmc_traces(obj_flat_samples_flatten_per_band, obj)
            plot_posterior(obj_flat_samples_flatten_per_band, obj)

            m = Model(
                obj['X'], obj['y'], obj['yerr'], 
                kernels.quasisep.Exp(jnp.array([1, 1])),
                zero_mean=zero_mean, has_jitter=has_jitter, has_lag=has_lag,
                lam_rf=obj['lam_rf'], z=obj['z']
            )
            psd_results = compute_psd_from_samples(obj_flat_samples, obj["clean_bands"])
            save_combined_plot(obj_flat_samples, m, obj['X'], obj['y'], obj['yerr'], obj['band_idx'], result, psd_results=psd_results)
            plot_broken_power_law(obj_flat_samples, obj)
            #dump_mcmc_diagnostics(mcmc, obj, i, len(batch_data))
            
        final_result_obj = obj | result #| rhat_ess
        results.append(final_result_obj)
        logging.info("--------------------------------------------------------------")
    
    save_quasar_list_hdf5(results, ignored_keys=['X', 'y', 'yerr', 'band_idx'])
        
    sys.exit("Exiting the program as requested.")
