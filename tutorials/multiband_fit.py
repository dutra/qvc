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

def initSampler(key, nSample, nBand=3):
    # split keys
    subkeys = jax.random.split(key, 10)

    # uniform sampler
    lagSampler = UniformInit(nBand-1, [-10, 10])
    meanSampler = UniformInit(nBand, [-1, 1])
    logAmpDeltaSampler = UniformInit(nBand-1, [-2, 0.0])
    logJitterSampler = UniformInit(nBand, [-20, -5])

    # kernel init
    #kernelSampler = DRWInit([jnp.log(1 / 1000), jnp.log(1)], [jnp.log(0.05), 0.0])
    kernelSampler = DRWInit([jnp.log(10), jnp.log(1e4)], [jnp.log(0.01), 0.0])

    return {
        "log_kernel_param": kernelSampler(subkeys[0], nSample),
        "log_amp_delta": logAmpDeltaSampler(subkeys[1], nSample),
        "mean": meanSampler(subkeys[2], nSample),
        "lag": lagSampler(subkeys[3], nSample),
        "log_jitter": logJitterSampler(subkeys[4], nSample),
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

    # kernel
    k = kernels.quasisep.Exp(*jnp.exp(log_kernel_param))
    m1 = MultiVarModel(X, y, yerr, k, zero_mean=False, has_jitter=True, has_lag=True)

    sample_params = {
        "log_kernel_param": log_kernel_param,
        "log_amp_delta": log_amp_delta,
        "lag": lag,
        "mean": mean,
        "log_jitter": log_jitter,
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
    y = jnp.array(all_mags)
    yerr = jnp.array(all_magerrs)
    t = jnp.array(all_times)

    # define params
    zero_mean = False
    has_jitter = True
    has_lag = True
    test_drw_params = {"log_kernel_param": jnp.log(np.array([100.0, 0.35]))}

    # define kernel
    k = kernels.quasisep.Exp(*jnp.exp(test_drw_params["log_kernel_param"]))

    # define model
    m1 = MultiVarModel(
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
        num_chains=2,
        progress_bar=False,
    )

    mcmc.run(jax.random.PRNGKey(1), X, yerr, y=y)
    samples = mcmc.get_samples(group_by_chain=False)
    diagnostics = mcmc.get_extra_fields()
    if np.all(diagnostics['diverging']):
        return None
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
    #print(f"Processing quasar {i}/{n} ({data['object_id']})", flush=True)

    # Load the quasar data
    result = fit_multiband(data)
    if result is None:
        print(f"Skipping quasar {data['object_id']} due to diverging MCMC.")
        return None
    data['i'] = i
    data |= result

    print(f"Quasar {i}/{n} ({data['object_id']}): log_tau_rest={data['log_tau_rest']:.3f}±{data['log_tau_rest_err']:.3f}, log_sigma={data['log_sigma']}±{data['log_sigma_err']}", flush=True)
    return data

def load_s82():

    s82_objs = []

    filters = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5} # harcoded filter order for SDSS
    bands = ['g','r','i']
    # Load the S82 data from the FITS file
    hdul = fits.open('data/dr16q_prop_May01_2024.fits')
    fits_data = hdul[1].data  # Assuming the data is in the first extension    

    cat = pd.read_parquet(f"data/S82/Catalog.parquet").set_index('idx')
    sdss = pd.read_parquet(f"data/S82/dr16s82_sdssLCRaw.parquet")
    ps1 = pd.read_parquet(f"data/S82/dr16s82_ps1LCRaw.parquet")
    ztf = pd.read_parquet(f"data/S82/dr16s82_ZuberLCRaw.parquet")

    for i in trange(len(cat)):
        data = {}  # Store the index for reference
        obj = cat.loc[i]
        data['object_id'] = obj['objectId']  # Store the objectId for reference

        c1 = SkyCoord(fits_data['RA'], fits_data['DEC'], unit='deg')
        c2 = SkyCoord(obj['RA'], obj['DEC'], unit='deg')
        sep = c1.separation(c2).to(u.arcsec)
        j = np.argwhere(sep < 1*u.arcsec).flatten()
        if len(j) == 0:
            print(f"Skipping entry {i} as it does not exist in the fits data.")
            continue
        j = j[0]  # Get the first index if there are multiple matches

        data['sdss_name'] = fits_data['SDSS_NAME'][j]  # Extract SDSS_NAME
        data['log_lbol'] = fits_data['LOGLBOL'][j]  # Extract log Lbol values
        data["log_lbol_err"] = fits_data['LOGLBOL_ERR'][j]  # Extract log Lbol error values
        data['log_mbh'] = fits_data['LOGMBH'][j]  # Extract log MBH values
        data['log_mbh_err'] = fits_data['LOGMBH_ERR'][j]  # Extract log MBH error values
        data['log_ledd_ratio'] = fits_data['LOGLEDD_RATIO'][j]  # Extract log L/edd values
        data['log_ledd_ratio_err'] = fits_data['LOGLEDD_RATIO_ERR'][j]  # Extract log L/edd error values
        data['z_sys_lines'] = fits_data['ZSYS_LINES'][j]  # Extract redshift (Z_SYS)
        data['z_sys_lines_err'] = fits_data['ZSYS_LINES_ERR'][j]  # Extract error in redshift (Z_SYS)
        try:
            sdss_lc = sdss[sdss.objectId == int(obj.objectId)].copy()
            ps1_lc = ps1[ps1.ps1objID == int(obj.ps1objID)].copy()
            ztf_lc = ztf[ztf.ps1objID == int(obj.ps1objID)].copy()
        except:
            continue

        data['z'] = obj['Z_SYS']  # Redshift

        times = {}
        mags = {}
        magerrs = {}

        for band in bands:
            times[band] = np.concatenate([sdss_lc.mjd[sdss_lc.filterID == filters[band]],
            ps1_lc.obsTime[ps1_lc.filterID == filters[band]],
            ztf_lc.mjd[ztf_lc.filterID == filters[band]]])

            sdss_ps1_offset = obj.sdss_g_qg - obj.ps1_g_qg 
            mags[band] = np.concatenate([sdss_lc.psMag[sdss_lc.filterID == filters[band]], 
            ps1_lc.psfMag[ps1_lc.filterID == filters[band]]+ sdss_ps1_offset,
            ztf_lc.mag[ztf_lc.filterID == filters[band]] + sdss_ps1_offset])
            mags[band] = mags[band] - np.nanmedian(mags[band])

            magerrs[band] = np.concatenate([sdss_lc.psMagErr_p3[sdss_lc.filterID == filters[band]],
            ps1_lc.psfMagErr_p3[ps1_lc.filterID == filters[band]],
            ztf_lc.magerr_p3[ztf_lc.filterID == filters[band]]])

        data['times'] = times
        data['mags'] = mags
        data['magerrs'] = magerrs

        s82_objs.append(data)
        

    hdul.close()
    

    # Write s82_objs to an HDF5 file
    with h5py.File("s82_objs.h5", "w") as hdf:
        for idx, obj in enumerate(s82_objs):
            group = hdf.create_group(obj['object_id'])  # Use object_id as the group name
            group.attrs['z'] = obj['z']
            group.attrs['object_id'] = obj['object_id']  # Store the objectId for reference
            group.attrs['sdss_name'] = obj['sdss_name']
            group.attrs['log_lbol'] = obj['log_lbol']
            group.attrs['log_lbol_err'] = obj['log_lbol_err']

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
                "z": group.attrs["z"],
                "sdss_name": group.attrs["sdss_name"],
                "log_lbol": group.attrs["log_lbol"],
                "log_lbol_err": group.attrs["log_lbol_err"],
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

if __name__ == '__main__':

    #objs = load_s82()
    objs = load_s82_from_hdf5()
    print(f"Loaded {len(objs)} quasars from the dataset.")
    # r = process_quasar(objs[0])
    chunk_size = 1000
    for start_idx in range(0, len(objs), chunk_size):
        chunk = objs[start_idx:start_idx + chunk_size]
        ctx = get_context("spawn")  # Safer for JAX when using multiprocessing
        results = []
        with ctx.Pool(processes=15) as pool:
            results = pool.map(partial(process_quasar, n=len(chunk)), enumerate(chunk))

        quasar_list = [q for q in results if q is not None]

        fields_to_save = ['i', 'object_id', 'sdss_name', 'z', 'log_lbol', 'log_lbol_err', 'log_tau_rest', 'log_tau_rest_err', 'log_sigma', 'log_sigma_err', 'log_jitter']
        filtered_quasar_list = [{field: q[field] for field in fields_to_save} for q in quasar_list]

        df = pd.DataFrame.from_records(filtered_quasar_list)
        output_file = 's82_multiband_fitted.csv'
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            merged_df = pd.concat([existing_df, df], ignore_index=True)
            merged_df.to_csv(output_file, index=False)
        else:
            df.to_csv(output_file, index=False)

    # ctx = get_context("spawn")  # Safer for JAX when using multiprocessing
    # results = []
    # with ctx.Pool(processes=15) as pool:
    #     results = pool.map(partial(process_quasar, n=len(objs[500:1000])), enumerate(objs[500:1000]))


    # quasar_list = [q for q in results if q is not None]

    # fields_to_save = ['i', 'object_id', 'sdss_name', 'z', 'log_lbol', 'log_lbol_err', 'log_tau_rest', 'log_tau_rest_err', 'log_sigma', 'log_sigma_err', 'log_jitter']
    # filtered_quasar_list = [{field: q[field] for field in fields_to_save} for q in quasar_list]

    # df = pd.DataFrame.from_records(filtered_quasar_list)
    # df.to_csv('s82_multiband_fitted_0_1000.csv', index=False)