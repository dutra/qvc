import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import numpyro
from numpyro import infer
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist

from tinygp import kernels

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
has_jitter = False
has_lag = True

def mle(Model, nBand, X, y, yerr, clean_bands, z, latent=False, fixed=True):
    print('Starting MLE...')

    from jax.tree_util import tree_map
    import jaxopt

    # Initial parameters (convert all to jnp.array for consistency)
    init_params = tree_map(
        jnp.array,
        {
            "log_tau_drw0": 6.0,
            "log_sigma0": -0.2,
            "alpha_host": 0.3,
            "f_host": 0.0,
            "poly1": 0.0,
            "mean": jnp.full(nBand, 0.0),
            "log_amp_delta_blr": jnp.full(nBand, -1.0),
            "lag0": 2.0,
            "lag_beta": 4.0 / 3.0,
            "eta_A1": 0.0,
            "eta_A2": 0.0,
            "eta_tau1": 0.0,
            "eta_tau2": 0.0,
        }
    )
    if latent:
        init_params["log_lag_blr"] = jnp.full(nBand, jnp.log(1e2))

    if not fixed:
        m = Model(
                X, y, yerr, kernels.quasisep.Exp(jnp.array([1.0, 1.0])),
                zero_mean=zero_mean,
                has_jitter=False,
                has_lag=has_lag,
                clean_bands=clean_bands,
                z=z
            )

        def loss(params):
            return -m.log_prob(params)

        loss_jit = jax.jit(loss)

        # Run LBFGS (JAX-native and GPU-capable)
        opt = jaxopt.LBFGS(fun=loss_jit) #, maxiter=500)
        soln = opt.run(init_params)
        best_param = soln.params

        print("Log prob at best fit:", m.log_prob(best_param))

        print('Done MLE')
        print("Best params:")
        for k, v in best_param.items():
            print(f"  {k}: {v}")
    else:
        best_param = init_params

    return best_param


def numpyro_joint_model(Model, batch_data, latent=False, bwb=False):
    batch_size = len(batch_data)
    nBands = 5  # or use from config

    # Shared across all objects
    powerlaw_samples = {
        k: numpyro.sample(k, dist.Normal(loc, scale))
        for k, (loc, scale) in {
            "eta_A1": (0.0, 1.0),
            "eta_A2": (0.0, 1.0),
            "eta_tau1": (0.0, 1.0),
            "eta_tau2": (0.0, 1.0),
        }.items()
    }

    # Extract object-level prior means
    log_tau_drw0_mean = jnp.array([obj['bestP']['log_tau_drw0'] for obj in batch_data])
    log_sigma0_mean = jnp.array([obj['bestP']['log_sigma0'] for obj in batch_data])
    log_amp_delta_blr_mean = jnp.stack([jnp.array(obj['bestP']['log_amp_delta_blr']) for obj in batch_data])  # (B, 5)
    mean_mean = jnp.stack([jnp.array(obj['bestP']['mean']) for obj in batch_data])                            # (B, 5)
    lag0_mean = jnp.stack([jnp.array(obj['bestP']['lag0']) for obj in batch_data])                            # (B, 5)
    lag_beta_mean = jnp.stack([jnp.array(obj['bestP']['lag_beta']) for obj in batch_data])                    # (B, 5)
    if latent:
        log_lag_blr_mean = jnp.stack([jnp.array(obj['bestP']['log_lag_blr']) for obj in batch_data])
    #log_jitter_mean = jnp.stack([jnp.array(obj['bestP']['log_jitter']) + jnp.mean(obj['yerr']) for obj in batch_data]) # (B, 5)
    
    with numpyro.plate("objects", batch_size):
        # Object-level parameters (shape: [B])
        log_tau_drw0 = numpyro.sample("log_tau_drw0", dist.Normal(log_tau_drw0_mean, 1.0))
        log_sigma0 = numpyro.sample("log_sigma0", dist.Normal(log_sigma0_mean, 1.0))
        log_sigma_hat0 = numpyro.deterministic("log_sigma_hat0", 2.0 * log_sigma0 - log_tau_drw0)
        alpha_host = numpyro.sample("alpha_host", dist.Normal(0.5, 1.0))
        f_host = numpyro.sample("f_host", dist.Uniform(0.0, 1.0))
        poly1 = numpyro.sample("poly1", dist.Normal(0.0, 0.1))
        lag0 = numpyro.sample("lag0", dist.TruncatedNormal(2.0, 10.0, low=0))
        lag_beta = numpyro.sample("lag_beta", dist.TruncatedNormal(4/3, 0.2, low=0))
        if bwb:
            bwb_A = numpyro.sample("bwb_A", dist.TruncatedNormal(0.0, 2.0, low=0))

    with numpyro.plate("objects", batch_size, dim=-2):
        with numpyro.plate("band", nBands, dim=-1):
            # Parameters with shape [B, nBands]
            mean = numpyro.sample("mean", dist.Normal(mean_mean, 0.2))
            log_amp_delta_blr = numpyro.sample("log_amp_delta_blr", dist.Normal(log_amp_delta_blr_mean, 2.0))
            if latent:
                log_lag_blr = numpyro.sample("log_lag_blr", dist.Normal(log_lag_blr_mean, 4.0))
            #log_jitter = numpyro.sample("log_jitter", dist.Normal(log_jitter_mean + 1e-6, 1.0))

    for i, data in enumerate(batch_data):
        # Collect params for object i
        params = {
            "log_tau_drw0": log_tau_drw0[i],
            "log_sigma_hat0": log_sigma_hat0[i],
            #"log_tau_drw_blr": log_tau_drw_blr[i],
            "alpha_host": alpha_host[i],
            "f_host": f_host[i],
            "poly1": poly1[i],
            "mean": mean[i],
            "log_amp_delta_blr": log_amp_delta_blr[i],
            #"log_jitter": log_jitter[i],
            "lag0": lag0[i],
            "lag_beta": lag_beta[i],
            **powerlaw_samples,
        }
        if latent:
            params["log_lag_blr"] = log_lag_blr[i]
        if bwb:
            params["bwb_A"] = bwb_A[i]

        m = Model(
            data['X'], data['y'], data['yerr'],
            kernels.quasisep.Exp(jnp.array([1, 1])),  # Placeholder, your Model will build the kernel
            zero_mean=zero_mean, has_jitter=has_jitter, has_lag=has_lag,
            clean_bands=data['clean_bands'], z=data['z']
        )
        # Test BWB
        if bwb:
            m.sample(params, i)
        else:
            log_prob = m.log_prob(params)
            #jax.debug.print("log_prob: {lp} {i}", lp=log_prob, i=i)
            log_prob = jnp.where(jnp.isfinite(log_prob), log_prob, -1e20)
            numpyro.factor(f"loglike_{i}", log_prob)

        def log_prob_single(params, model):
            return model.log_prob(params)

        log_prob_batch = jax.vmap(log_prob_single, in_axes=(0, 0))

def make_lc(Model, data):
    times = data['times']
    mags = data['mags']
    data['mags_means'] = np.array([np.nanmean(mags[band]) for band in mags.keys()])
    for band in mags.keys():
       mags[band] = mags[band] - np.nanmean(mags[band])  # Center the magnitudes
    magerrs = data['magerrs']
    
    #red_bands = bands_redder_than_5000(data['z'])
    blue_bands = bands_bluer_than_lyman_alpha(data['z'])

    clean_bands = list(set(bands) - set(blue_bands))
    # Reorder clean_bands to match the desired order
    clean_bands = list(sorted(clean_bands, key=lambda band: ['u', 'g', 'r', 'i', 'z', 'y'].index(band)))
    #clean_bands = bands
    data['clean_bands'] = clean_bands
    print(f"Bands: {bands}, Clean Bands: {clean_bands}")
    if len(clean_bands) == 0:
        print(f"No clean bands for quasar {data['object_id']}, skipping.", flush=True)
        return None
    # Combine
    print(times.keys())
    all_times = np.concatenate([times[b] for b in clean_bands])
    all_mags = np.concatenate([mags[b] for b in clean_bands]) 
    all_magerrs = np.concatenate([magerrs[b] for b in clean_bands])
    band_idx = np.concatenate([np.full(len(times[b]), i) for i, b in enumerate(clean_bands)])

    if len(all_times) == 0 or len(all_mags) == 0 or len(all_magerrs) == 0:
        print(f"No magnitudes or errors for quasar {data['object_id']}, skipping.", flush=True)
        return None
    # Check for NaNs
    if np.all(~np.isfinite(all_times)) or np.all(~np.isfinite(all_mags)) or np.all(~np.isfinite(all_magerrs)):
        print(f"NaN values ({len(~np.isfinite(all_mags))}/{len(all_mags)}) found in data for quasar {data['object_id']}, skipping.", flush=True)
        return None

    # Sort in time
    sort_idx = np.argsort(all_times)
    all_times = all_times[sort_idx]
    all_mags = all_mags[sort_idx]
    all_magerrs = all_magerrs[sort_idx]
    band_idx = band_idx[sort_idx]

    # Mask NaNs
    mask = np.isfinite(all_mags)
    all_times = all_times[mask]
    all_mags = all_mags[mask]
    all_magerrs = all_magerrs[mask]
    band_idx = band_idx[mask]

    # Define X, y, yerr, t
    # X = (all_times, band_idx)
    X = (jnp.array(all_times)-jnp.min(all_times), jnp.array(band_idx))
    y = np.array(all_mags)
    yerr = np.array(all_magerrs)
    t = np.array(all_times)

    # Reject outliers in moving window per band
    window_size = 6
    mask_outlier = np.ones(len(y), dtype=bool)

    from numpy.lib.stride_tricks import sliding_window_view
    from scipy.stats import median_abs_deviation

    for band in np.unique(band_idx):
        band_mask = band_idx == band
        idx_band = np.where(band_mask)[0]
        band_y = y[band_mask]

        # Generate sliding windows: shape (N - 2*window, 2*window + 1)
        if len(band_y) < 2 * window_size + 1:
            continue  # Skip small bands

        windows = sliding_window_view(band_y, 2 * window_size + 1)
        centers = band_y[window_size:-window_size]
        medians = np.nanmean(windows, axis=1)
        mads = median_abs_deviation(windows, axis=1)

        # Compute boolean mask for which center points are outliers
        is_outlier = np.abs(centers - medians) > 2.5 * mads

        # Translate back to original indices
        mask_outlier[idx_band[window_size:-window_size][is_outlier]] = False

    # Apply final mask
    X = (jnp.array(all_times[mask_outlier]) - jnp.min(all_times[mask_outlier]), jnp.array(band_idx[mask_outlier]))
    y = jnp.array(y[mask_outlier])
    yerr = jnp.array(yerr[mask_outlier])
    t = jnp.array(t[mask_outlier])

    batch_dict = {'X': X, 'y': y, 'yerr': yerr, 'clean_bands': clean_bands, 'z': data['z'], 'band_idx': band_idx[mask_outlier]}
    return batch_dict

                    
if __name__ == '__main__': 
    
    logging.info("Starting multiband fit")

    parser = argparse.ArgumentParser(description="Process quasars with optional filtering.")
    parser.add_argument("--filter_object_id", nargs="+", help="List of object IDs to filter.")
    parser.add_argument("--N", type=int, help="Number of objects to process.")
    parser.add_argument("--skip", type=int, help="Number of objects to skip.")
    parser.add_argument("--chunk_size", type=int, default=500, help="Chunk size for processing objects.")
    parser.add_argument("-f", "--file", type=str, help="Path to the file to append (read and write) objects.") 
    parser.add_argument("--lc_file", type=str, help="Path to the light curve file.")
    parser.add_argument("--filter_file", type=str, help="Path to the file containing object IDs to filter.")
    parser.add_argument("--plot", action="store_true", help="Enable plotting of results.")
    parser.add_argument("--svi", action="store_true", help="Use stochastic variation inference (SVI).")
    parser.add_argument("--ignore_existing", action="store_true", help="Ignore sources already in the HDF5 file.")
    parser.add_argument("--create_lc", action="store_true", help="Only create LC file and exit.")
    parser.add_argument("--progress", action="store_true", help="Show progress bar.")
    parser.add_argument("--joint", action="store_true", help="Use joint model fitting.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU.")
    parser.add_argument("--nwarm", type=int, default=500, help="Number of warmup steps for MCMC.")
    parser.add_argument("--nsamp", type=int, default=250, help="Number of samples for MCMC.")
    parser.add_argument("--nchains", type=int, default=-1, help="Number of chains for MCMC.")
    parser.add_argument("--latent", action="store_true", help="Use latent variable model.")
    parser.add_argument("--bwb", action="store_true", help="Use BWB model.")
    parser.add_argument("--choose_N", type=int, default=-1, help="Sample choose_N objects.")
    parser.add_argument("--job_id", type=int, default=-1, help="Job Index for parallel processing.")
    parser.add_argument("--job_N", type=int, default=-1, help="Number of objects to divide.")
    parser.add_argument("--max_tree_depth", type=int, default=8, help="Max tree depth param for NUTS sampler.")

    args = parser.parse_args()
    print("Args: ", args)

    check_64bit(gpu=not bool(args.cpu))

    if args.create_lc:
        objs = concat_light_curves(save_file_path=args.lc_file, progress_bar=args.progress)
        sys.exit("Created LC file. Exiting the program as requested.")

    # Filter objects by object_id that exist in the HDF5 file
    existing_object_ids = set()
    if args.ignore_existing:
        if os.path.exists(args.file):
            with h5py.File(args.file, "r") as hdf:
                existing_object_ids = set(hdf.keys())
                print(f"Found {len(existing_object_ids)} existing object IDs in {args.file}")
        else:
            print("WARNING! --ignore_existing flag but no existing file")

    filter_object_ids = args.filter_object_id if args.filter_object_id else []
    filter_object_ids = pd.read_csv(args.filter_file, dtype={"object_id": str})["object_id"].values if args.filter_file else filter_object_ids
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

    objs = concat_light_curves(filter_object_ids=filter_object_ids, existing_object_ids=existing_object_ids, N=args.N, skip=args.skip, save_file_path=args.lc_file, progress_bar=args.progress)
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
        result = make_lc(Model, obj)
        if result is None:
            continue
        obj['i'] = i
        obj |= result
        # Run bestP for each object
        n_bands = len(obj['clean_bands'])
        bestP = mle(Model, 5, obj['X'], obj['y'], obj['yerr'], obj['clean_bands'], obj['z'], latent=args.latent)
        m = Model(
            obj['X'], obj['y'], obj['yerr'], 
            kernels.quasisep.Exp(jnp.array([1, 1])),
            zero_mean=zero_mean, has_jitter=False, has_lag=has_lag,
            clean_bands=obj['clean_bands'], z=obj['z']
        )
        #save_combined_plot(bestP, m, obj['X'], obj['y'], obj['yerr'], obj['band_idx'], obj, fit_bestP=True)

        num_params = sum(p.size for p in bestP.values())
        batch_data.append({
            'object_id': obj['object_id'],
            'X': obj['X'],
            'y': obj['y'],
            'yerr': obj['yerr'],
            'clean_bands': obj['clean_bands'],
            'band_idx': obj['band_idx'],
            'z': obj['z'],
            'bestP': bestP,
            # add any other fields needed by your model
        })
    num_params = sum(p.size for p in batch_data[0]['bestP'].values())
    num_objects = len(batch_data)
    logging.info(f"Running joint fit on {len(batch_data)} objects...")

    estimated_nchains = 4
    if args.nchains < 1:
        nchains = estimated_nchains
    else:
        nchains = args.nchains
    print(f"{args.max_tree_depth=}, {args.nwarm=}, {args.nsamp=}, {args.nchains=}, default num_chains: {estimated_nchains}, {num_params=}, {len(batch_data)=}")
    
    init_strategy = numpyro.infer.init_to_sample()
    #init_strategy = numpyro.infer.init_to_median()
    logging.info("Done with numpyro.infer.init_to_sample")

    nuts_kernel = NUTS(numpyro_joint_model, init_strategy=init_strategy, dense_mass=True, max_tree_depth=args.max_tree_depth)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=args.nwarm,
        num_samples=args.nsamp,
        num_chains=nchains,
        progress_bar=args.progress,
        chain_method="vectorized",
    )
    mcmc.run(jax.random.PRNGKey(0), Model, batch_data, args.latent, args.bwb)
    samples_flat = mcmc.get_samples(group_by_chain=False)
    diagnostics = mcmc.get_extra_fields()

    #ns = NestedSampler(numpyro_joint_model)
    #ns.run(jax.random.PRNGKey(0), Model, batch_data)
    #samples_flat = ns.get_samples(jax.random.PRNGKey(0), num_samples=1000)

    logging.info("Done with MCMC run")

    # Save and plot the results
    results = []
    for i, obj in enumerate(batch_data):

        for k, v in samples_flat.items():
            print(v.shape, k)
            
        # The universal parameters are 1D
        obj_samples_clean = {
            k: v[:, i] if k not in ['eta_A1', 'eta_A2', 'eta_tau1', 'eta_tau2'] else v
            for k, v in samples_flat.items()
        }

        # Add the object-specific parameters
        result = process_samples(obj_samples_clean, obj)

        # Plotting
        if args.plot:
            plot_mcmc_traces(obj_samples_clean, obj)
            m = Model(
                obj['X'], obj['y'], obj['yerr'], 
                kernels.quasisep.Exp(jnp.array([1, 1])),
                zero_mean=zero_mean, has_jitter=has_jitter, has_lag=has_lag,
                clean_bands=obj['clean_bands'], z=obj['z']
            )
            psd_results = compute_psd_from_samples(obj_samples_clean, obj["clean_bands"])
            save_combined_plot(obj_samples_clean, m, obj['X'], obj['y'], obj['yerr'], obj['band_idx'], result, fit_bestP=False, psd_results=psd_results)
            #dump_mcmc_diagnostics(mcmc, obj, i, len(batch_data))
            plot_posterior_for_object(samples_flat, obj, i, len(batch_data))
        results.append(obj | result)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++", flush=True)
        print(f"Quasar {i+1}/{len(batch_data)} Object ID: {obj['object_id']}", flush=True)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ Done fitting all objects")

        # Save results to HDF5 file
        if args.file:
            print("Saving results to ", args.file)
            append_hdf5_file(results, args.file)
        else:
            print("Warning!! Not saving results to file.")

    sys.exit("Exiting the program as requested.")
