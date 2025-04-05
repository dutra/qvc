import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["NUMEXPR_NUM_THREADS"] = "32"
os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["VECLIB_MAXIMUM_THREADS"] = "32"
os.environ["NUMBA_NUM_THREADS"] = "32"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm, trange

import scipy.stats as st

import jax
import jax.numpy as jnp


import numpyro
from numpyro import infer
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist

from tinygp import kernels, solvers

#print("Total device count:", jax.local_device_count())
numpyro.set_host_device_count(30)
jax.config.update("jax_enable_x64", True)

import warnings

import matplotlib.pyplot as plt
import numpy as np
import optax
from eztaox.fitter import fit
from eztaox.initializers import DRWInit, UniformInit
from eztaox.models import MultiVarModel
from eztaox.utils import formatlc
from tinygp import kernels

from astropy.coordinates import SkyCoord
from astropy import units as u

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

warnings.filterwarnings("ignore", category=RuntimeWarning)

from astropy.io import fits
from multiprocessing import Pool, get_context
import h5py
import sys

from tinygp.helpers import JAXArray


# define params
zero_mean = False
has_jitter = True
has_lag = True

def initSampler(key, nSample, nBand=3):
    # split keys
    subkeys = jax.random.split(key, 10)

    # uniform sampler
    lagSampler = UniformInit(nBand-1, [-10, 10])
    meanSampler = UniformInit(nBand, [-1, 1])
    logAmpDeltaSampler = UniformInit(nBand-1, [-2, 0.0])
    logJitterSampler = UniformInit(nBand, [-20, -5])
    betaSampler = UniformInit(1, [0, 1])

    # kernel init
    #kernelSampler = DRWInit([jnp.log(1 / 1000), jnp.log(1)], [jnp.log(0.05), 0.0])
    kernelSampler = DRWInit([jnp.log(100), jnp.log(0.1)], [jnp.log(0.01), 0.0])

    return {
        "log_kernel_param": kernelSampler(subkeys[0], nSample),
        "log_amp_delta": logAmpDeltaSampler(subkeys[1], nSample),
        "mean": meanSampler(subkeys[2], nSample),
        "lag": lagSampler(subkeys[3], nSample),
        "log_jitter": logJitterSampler(subkeys[4], nSample),
        "beta": betaSampler(subkeys[5], nSample),
    }
def numpyro_model(X, yerr, y=None, bestP=None):
    # kernel param
    flat_normal = dist.Normal(bestP["log_kernel_param"], jnp.array([5.0, 5.0]))
    diag_normal = dist.Independent(flat_normal, 1)
    log_kernel_param = numpyro.sample("log_kernel_param", diag_normal)

    # log amp delta
    log_amp_delta = numpyro.sample(
        "log_amp_delta", dist.Normal(bestP["log_amp_delta"], 2.0)
    )

    # lag
    lag = numpyro.sample("lag", dist.Normal(bestP['lag'], 10.0))
    
    # log jitter, mean => the prior for these two should be set small, otherwise
    # it is hard to converge
    log_jitter = numpyro.sample("log_jitter", dist.Normal(bestP["log_jitter"], 0.1))
    
    mean = numpyro.sample("mean", dist.Normal(bestP['mean'], 0.1))

    beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))

    # kernel
    k = kernels.quasisep.Exp(*jnp.exp(log_kernel_param))
    m1 = MultiVarModel(X, y, yerr, k, zero_mean=zero_mean, has_jitter=has_jitter, has_lag=has_lag)

    sample_params = {
        "log_kernel_param": log_kernel_param,
        "log_amp_delta": log_amp_delta,
        "lag": lag,
        "mean": mean,
        "log_jitter": log_jitter,
        "beta": beta
    }
    m1.sample(sample_params)


def fit_multiband(data):
    bands = ['g', 'r', 'i']
    times = data['times']
    mags = data['mags']
    magerrs = data['magerrs']

    # Combine
    all_times = np.concatenate([times[b] for b in bands])
    all_mags = np.concatenate([mags[b] for b in bands]) 
    all_magerrs = np.concatenate([magerrs[b] for b in bands])
    band_idx = np.concatenate([np.full(len(times[b]), i) for i, b in enumerate(bands)])

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

    # Reject outliers in moving window
    window_size = 20
    mask_outlier = np.ones(len(y), dtype=bool)
    for i in range(len(y)):
        if i < window_size or i >= len(y) - window_size:
            continue
        window = y[i - window_size:i + window_size + 1]
        if np.abs(y[i] - np.nanmean(window)) > 3 * st.median_abs_deviation(window):
            mask_outlier[i] = False

    X = (all_times[mask_outlier], band_idx[mask_outlier])
    y = jnp.array(y[mask_outlier])
    yerr = jnp.array(yerr[mask_outlier])
    t = jnp.array(t[mask_outlier])

    # define kernel
    initial_drw_params = {"log_kernel_param": jnp.log(np.array([100.0, 0.35]))}
    k = kernels.quasisep.Exp(*jnp.exp(initial_drw_params["log_kernel_param"]))

    # Override MultiVarModel
    class MyMultiVarModel(MultiVarModel):
        def amp_transform(self, params: dict[str, JAXArray]) -> JAXArray:
            # gri central wavelengths
            lamdba_RF = np.array([4770, 6231, 7625])/(1 + data['z'])
            return jnp.array([params["log_kernel_param"][1] - params["beta"]*np.log10(lamdba_RF[b]/4000) for b in range(3)])

    # define model
    m1 = MyMultiVarModel(
        X, y, yerr, k, zero_mean=zero_mean, has_jitter=has_jitter, has_lag=has_lag
    )
    bestP, logProb = fit(model=m1, 
                        optimizer=optax.adam(learning_rate=0.1),
                        initSampler=initSampler,
                        prng_key=jax.random.PRNGKey(0),
                        nSample=10_000, nIter=2, nBest=5)

    for k in bestP.keys():
        bestP[k] += 1e-4 * np.random.randn(*bestP[k].shape) 

    nuts_kernel = NUTS(
        partial(numpyro_model, bestP=bestP),
        dense_mass=True,
        target_accept_prob=0.9,
        # adapt_step_size=True,
    )

    mcmc = MCMC(
        nuts_kernel,
        num_warmup=250,
        num_samples=250,
        num_chains=1,
        progress_bar=False,
    )

    mcmc.run(jax.random.PRNGKey(1), X, yerr, y=y)
    samples = mcmc.get_samples(group_by_chain=False)
    diagnostics = mcmc.get_extra_fields()
    if np.all(diagnostics['diverging']):
        return None
    
    save_combined_plot(samples, m1, X, y, yerr, band_idx[mask_outlier], {0: 'g', 1: 'r', 2: 'i'}, data['object_id'])

    log_tau_rest = np.log10(np.exp(samples['log_kernel_param'][:, 0])/(1+data['z']))
    lower, median, upper = np.percentile(log_tau_rest, [16, 50, 84], axis=0)
    log_tau_rest_err = 0.5 * (upper - lower) # symmetric uncertainties
    log_tau_rest = median

    log_sigma = np.log10(np.exp(samples['log_kernel_param'][:, 1]))
    lower, median, upper = np.percentile(log_sigma, [16, 50, 84], axis=0)
    log_sigma_err = 0.5 * (upper - lower) # symmetric uncertainties
    log_sigma = median
    
    log_amp_delta = np.log10(np.exp(samples['log_amp_delta']))
    lower, median, upper = np.percentile(log_amp_delta, [16, 50, 84], axis=0)
    log_amp_delta_err = 0.5 * (upper - lower) # symmetric uncertainties
    log_amp_delta = median

    log_sigma = np.array([log_sigma, *(log_amp_delta+log_sigma)])
    log_sigma_err = np.array([log_sigma_err, *(np.sqrt(log_amp_delta_err**2+log_sigma_err**2))])

    log_jitter = np.percentile(np.log10(np.exp(2*samples['log_jitter'])), 50, axis=0)

    return dict(log_tau_rest=log_tau_rest,
                log_tau_rest_err=log_tau_rest_err,
                log_sigma=log_sigma,
                log_sigma_err=log_sigma_err,
                log_jitter=log_jitter)


def process_quasar(i_data, n=0):
    i, data = i_data
    print(f"Processing quasar {i}/{n} ({data['object_id']})", flush=True)

    # Load the quasar data
    result = fit_multiband(data)
    if result is None:
        print(f"Skipping quasar {data['object_id']} due to diverging MCMC.")
        return None
    data['i'] = i
    data |= result

    print(f"Quasar {i}/{n} ({data['object_id']}): log_tau_rest={data['log_tau_rest']:.3f}±{data['log_tau_rest_err']:.3f}, log_sigma={data['log_sigma']}±{data['log_sigma_err']}", flush=True)
    return data

def save_lc_plot(bands, times, mags, magerrs, object_id):
    # Plot and save the light curves
    fig, ax = plt.subplots(figsize=(10, 6))
    for band in bands:
        if len(times[band]) > 0:
            ax.errorbar(times[band], mags[band], yerr=magerrs[band], fmt='o', label=f'{band}-band', alpha=0.7)

    ax.set_xlabel('Time (MJD)', fontsize=14)
    ax.set_ylabel('Magnitude', fontsize=14)
    ax.invert_yaxis()  # Magnitudes are brighter when lower
    ax.legend()
    ax.set_title(f'Light Curve for Object {object_id}', fontsize=16)
    plt.tight_layout()

    # Save the plot as a PNG file
    output_dir = "light_curves"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join("light_curves", f'{object_id}_light_curve.png'))
    plt.close(fig)


def save_combined_plot(samples, model, X, y, yerr, band_idx, band_idx_map, object_id):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), sharex=True)

    t = X[0]
    colors = ['b', 'g', 'r']
    for n in np.unique(band_idx):
        m = band_idx == n
        # Plot the observed data
        ax.errorbar(t[m], y[m], yerr=yerr[m], fmt='o', label=f'{band_idx_map[n]}-band', alpha=0.7, color=colors[n])
        # Generate test times for predictions
        t_test = np.linspace(t.min(), t.max(), 1000)
        # Compute predictions using the model
        posterior_median = {k: jnp.median(v, axis=0) for k, v in samples.items()}
        mu, std = model.pred(posterior_median, (t_test, np.full_like(t_test, n, dtype=int)))
        # Plot the predictions
        ax.plot(t_test, mu, label=f'{band_idx_map[n]}-band', alpha=0.7, color=colors[n])
        ax.fill_between(t_test, mu - std, mu + std, alpha=0.3, label=f'{band_idx_map[n]}-band', color=colors[n])

    ax.set_ylabel('Magnitude (Observed)', fontsize=14)
    ax.invert_yaxis()  # Magnitudes are brighter when lower
    ax.legend(loc='upper right')
    ax.set_title(f'Light Curve for Object {object_id}', fontsize=16)

    plt.tight_layout()

    # Save the plot as a PNG file
    output_dir = "light_curves_fits"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{object_id}_combined_plot.png'))
    plt.close(fig)


def save_s82(file_path):

    s82_objs = []

    filters = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5} # harcoded filter order for SDSS
    bands = ['g','r','i']
    # Load the S82 data from the FITS file
    hdul = fits.open('data/dr16q_prop_May01_2024.fits')
    fits_data = hdul[1].data  # Assuming the data is in the first extension    

    cat = pd.read_parquet(f"data/S82/Catalog.parquet").set_index('idx')
    sdss = pd.read_parquet(f"data/S82/dr16s82_sdssLCRaw.parquet")
    sdss = sdss[sdss.mjd.notna() & (len(sdss.mjd) > 0)]
    ps1 = pd.read_parquet(f"data/S82/dr16s82_ps1LCRaw.parquet")
    ztf = pd.read_parquet(f"data/S82/dr16s82_ZuberLCRaw.parquet")

    # Find elements in cat where objectId exists in the list of objectId of sdss
    sdss_object_ids = set(sdss.objectId)
    sdss_object_ids = [
    # log10_sigma > 1    
    1384141, 1384142, 1384145, 1384146, 1384147, 1384148, 1384151, 1384153, 1384156, 1384157, 1384160, 1384165, 1384166, 1384171, 1384172,
    # random sample
    1385090, 1385694, 1384550, 1384780, 1385083, 1384786, 1385298, 1384985, 1384894, 1385218, 1384922, 1384773, 1385567, 1385607, 1384291]
    sdss_object_ids = [str(obj_id) for obj_id in sdss_object_ids]
    matching_indices = cat[cat.objectId.isin(sdss_object_ids)].index
    cat = cat.loc[matching_indices]
    #cat = cat[:1000]

    print("Len cat: ", len(cat))

    # Loop through the data and extract the relevant information        
    for idx, row in tqdm(cat.iterrows(), total=len(cat), desc="Processing quasars"):
        object_id = row['objectId']

        # Filter light curves for the current object_id
        sdss_lc = sdss[sdss.objectId == row['objectId']].copy()
        ps1_lc = ps1[ps1.ps1objID == int(row['ps1objID'])].copy() if row['ps1objID'] is not None else pd.DataFrame()
        ztf_lc = ztf[ztf.ps1objID == int(row['ps1objID'])].copy() if row['ps1objID'] is not None else pd.DataFrame()

        #print("LC len: ", len(sdss_lc), len(ps1_lc), len(ztf_lc))

        # Combine light curves from different catalogs
        times = {}
        mags = {}
        magerrs = {}
        magerrs_mean = []

        for band in bands:
            sdss_ps1_offset = {
                'g': row.sdss_g_qg - row.ps1_g_qg ,
                'r': row.sdss_r_qg - row.ps1_r_qg,
                'i': row.sdss_i_qg - row.ps1_i_qg
            }
            offset = sdss_ps1_offset[band] if band in sdss_ps1_offset else 0.0

            times[band] = np.concatenate([
                sdss_lc[sdss_lc.filterID == filters[band]].mjd.values if not sdss_lc.empty else [],
                ps1_lc[ps1_lc.filterID == filters[band]].obsTime.values if not ps1_lc.empty else [],
                ztf_lc[ztf_lc.filterID == filters[band]].mjd.values if not ztf_lc.empty else []
            ])

            mags[band] = np.concatenate([
                sdss_lc[sdss_lc.filterID == filters[band]].psMag.values if not sdss_lc.empty else [],
                ps1_lc[ps1_lc.filterID == filters[band]].psfMag.values + offset if not ps1_lc.empty else [],
                ztf_lc[ztf_lc.filterID == filters[band]].mag.values + offset if not ztf_lc.empty else []
            ])
            mags[band] = mags[band] - np.mean(mags[band])  # Center the magnitudes

            magerrs[band] = np.concatenate([
                sdss_lc[sdss_lc.filterID == filters[band]].psMagErr_p3.values if not sdss_lc.empty else [],
                ps1_lc[ps1_lc.filterID == filters[band]].psfMagErr_p3.values if not ps1_lc.empty else [],
                ztf_lc[ztf_lc.filterID == filters[band]].magerr_p3.values if not ztf_lc.empty else []
            ])
            # Select NaNs from mags and magerrs
            nan_mask = np.isnan(mags[band]) | np.isnan(magerrs[band])

            # Drop NaNs from mags and magerrs using the same indexes
            times[band] = times[band][~nan_mask]
            mags[band] = mags[band][~nan_mask]
            magerrs[band] = magerrs[band][~nan_mask]

            # Ensure magerrs_mean calculation handles empty arrays
            if len(magerrs[band]) == 0:
                magerrs_mean.append(np.nan)  # Default to 0.0 if no data is available
            else:
                magerrs_mean.append(np.mean(magerrs[band]))

        # Skip if no data is available for the object
        if any(len(times[band]) == 0 or len(mags[band]) == 0 or len(magerrs[band]) == 0 for band in bands):
            continue

        s82_objs.append({
            'object_id': object_id,
            'times': times,
            'mags': mags,
            'magerrs': magerrs,
            'magerrs_mean': magerrs_mean
        })

        save_lc_plot(bands, times, mags, magerrs, object_id)

    hdul.close()

    
    s82_objs = populate_sdss_fields(s82_objs)
    
    # Write s82_objs to an HDF5 file
    with h5py.File(file_path, "w") as hdf:
        for idx, obj in enumerate(s82_objs):
            group = hdf.create_group(obj['object_id'])  # Use object_id as the group name
            group.attrs['object_id'] = obj['object_id']  # Store the objectId for reference
            group.attrs['magerrs_mean'] = obj['magerrs_mean']
            for field in ['z', 'ra', 'dec', 'sdss_name', 'log_lbol', 'log_lbol_err',
                          'log_mbh', 'log_mbh_err', 'log_ledd_ratio', 'log_ledd_ratio_err']:
                group.attrs[field] = obj[field]

            for key in ['times', 'mags', 'magerrs']:
                sub_group = group.create_group(key)
                for band, values in obj[key].items():
                    sub_group.create_dataset(band, data=values)


    return s82_objs

def load_s82_from_hdf5(file_path="s82_objs.h5"):
    s82_objs = []

    with h5py.File(file_path, "r") as hdf:
        for object_id in hdf.keys():
            group = hdf[object_id]
            data = {
                "object_id": object_id,
                "magerrs_mean": group.attrs["magerrs_mean"],
                "times": {},
                "mags": {},
                "magerrs": {},
            }

            for key in ["times", "mags", "magerrs"]:
                sub_group = group[key]
                for band in sub_group.keys():
                    data[key][band] = sub_group[band][...]

            s82_objs.append(data)

    return s82_objs

def populate_sdss_fields(s82_objs):
    cat = pd.read_parquet(f"data/S82/Catalog.parquet").set_index('idx')
    hdul = fits.open('data/dr16q_prop_May01_2024.fits')
    fits_data = hdul[1].data  # Assuming the data is in the first extension    
    fits_data_2 = hdul[2].data  # Assuming the data is in the first extension    
    for d in tqdm(s82_objs, desc="Populating SDSS fields"):
        obj = cat.loc[cat['objectId'] == d['object_id']].iloc[0]
        c1 = SkyCoord(fits_data['RA'], fits_data['DEC'], unit='deg')
        c2 = SkyCoord(obj['RA'], obj['DEC'], unit='deg')
        sep = c1.separation(c2).to(u.arcsec)
        j = np.argwhere(sep < 1*u.arcsec).flatten()
        if len(j) == 0:
            print(f"Skipping entry {d['object_id']} as it does not exist in the fits data.")
            continue
        
        j = j[0]  # Get the first index if there are multiple matches
        d['ra'] = obj['RA']
        d['dec'] = obj['DEC']
        d['z'] = obj['Z_SYS']
        d['sdss_name'] = fits_data['SDSS_NAME'][j]  # Extract SDSS_NAME
        d['log_lbol'] = fits_data['LOGLBOL'][j]  # Extract log Lbol values
        d["log_lbol_err"] = fits_data['LOGLBOL_ERR'][j]  # Extract log Lbol error values
        d['log_mbh'] = fits_data['LOGMBH'][j]  # Extract log MBH values
        d['log_mbh_err'] = fits_data['LOGMBH_ERR'][j]  # Extract log MBH error values
        d['log_ledd_ratio'] = fits_data['LOGLEDD_RATIO'][j]  # Extract log L/edd values
        d['log_ledd_ratio_err'] = fits_data['LOGLEDD_RATIO_ERR'][j]  # Extract log L/edd error values
        #d['z_sys_lines'] = fits_data['ZSYS_LINES'][j]  # Extract redshift (Z_SYS)
        #d['z_sys_lines_err'] = fits_data['ZSYS_LINES_ERR'][j]  # Extract error in redshift (Z_SYS)

    return s82_objs

if __name__ == '__main__':

    #objs = save_s82(file_path="s82_objs_sdss_selected.h5")
    #sys.exit("Exiting the program as requested.")

    objs = load_s82_from_hdf5(file_path="s82_objs_sdss_selected.h5")
    filtered_objs = [obj for obj in objs if int(obj['object_id']) in [
    # log10_sigma > 1    
    1384141, 1384142, 1384145, 1384146, 1384147, 1384148, 1384151, 1384153, 1384156, 1384157, 1384160, 1384165, 1384166, 1384171, 1384172,
    # random sample
    1385090, 1385694, 1384550, 1384780, 1385083, 1384786, 1385298, 1384985, 1384894, 1385218, 1384922, 1384773, 1385567, 1385607, 1384291
    ]]
    #objs = objs[:1000]
    print(f"Loaded {len(objs)} quasars from the dataset.")
    objs = populate_sdss_fields(objs)
    print(f"Populated {len(objs)} quasars with SDSS data.")

    #r = process_quasar(objs[0])
    #sys.exit("Exiting the program as requested.")

    chunk_size = 500
    for start_idx in range(0, len(objs), chunk_size):
        print("========================================================================")
        print(f"Processing chunk {start_idx // chunk_size + 1}/{(len(objs) + chunk_size - 1) // chunk_size}...")
        chunk = objs[start_idx:start_idx + chunk_size]
        
        ctx = get_context("spawn")  # Safer for JAX when using multiprocessing
        results = []

        with ctx.Pool(processes=30) as pool:
            results = pool.map(partial(process_quasar, n=len(chunk)), enumerate(chunk))

        sys.exit("LC and fits were saved into folder light_curves_fits.")

        quasar_list = [q for q in results if q is not None]

        fields_to_filter = ['times', 'mags', 'magerrs']
        filtered_quasar_list = [{k: v for k, v in q.items() if k not in fields_to_filter} for q in quasar_list]
        #filtered_quasar_list = [{field: q[field] for !(field in fields_to_filter)} for q in quasar_list]

        df = pd.DataFrame.from_records(filtered_quasar_list)
        output_file = 's82_multiband_fitted_sdss.csv'
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            merged_df = pd.concat([existing_df, df], ignore_index=True)
            merged_df.to_csv(output_file, index=False)
        else:
            df.to_csv(output_file, index=False)

    # ctx = get_context("spawn")  # Safer for JAX when using multiprocessing
    # results = []
    # with ctx.Pool(processes=15) as pool:
    #     results = pool.map(partial(process_quasar, n=len(objs[0:10])), enumerate(objs[0:10]))


    # quasar_list = [q for q in results if q is not None]

    # fields_to_save = ['i', 'object_id', 'sdss_name', 'z', 'log_lbol', 'log_lbol_err', 'log_tau_rest', 'log_tau_rest_err', 'log_sigma', 'log_sigma_err', 'log_jitter']
    # filtered_quasar_list = [{field: q[field] for field in fields_to_save} for q in quasar_list]

    # df = pd.DataFrame.from_records(filtered_quasar_list)
    # df.to_csv('s82_multiband_fitted_0_10.csv', index=False)