import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["NUMEXPR_NUM_THREADS"] = "32"
os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["VECLIB_MAXIMUM_THREADS"] = "32"
os.environ["NUMBA_NUM_THREADS"] = "32"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

from collections.abc import Callable
import equinox as eqx
from tinygp.helpers import JAXArray
from numpy.typing import NDArray
import tinygp
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
plt.style.use("style.mplstyle")
import numpy as np
import optax
from eztaox.fitter import fit
from eztaox.initializers import DRWInit, UniformInit
from eztaox.models import MultiVarModel, MultiVarModelFFT
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
import corner
import argparse

num_samples = 500

# define params
zero_mean = False
has_jitter = True
has_lag = True

lambda_pivot = {
    'u': 3543,  # SDSS u-band
    'g': 4770,  # SDSS g-band
    'r': 6231,  # SDSS r-band
    'i': 7625,  # SDSS i-band
    'z': 9134,  # SDSS z-band
    'y': 9633,  # PS1 y-band
}


filters = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5} # harcoded filter order for SDSS
bands = ['u', 'g', 'r', 'i', 'z']#, 'y']

colors = {'u': 'tab:blue',
          'g': 'tab:green', 
          'r': 'tab:orange', 
          'i': 'tab:red', 
          'z': 'tab:brown', 
          'y': 'tab:gray'}

# Override MultiVarModel
class MyMultiVarModel(MultiVarModel):
    filtered_bands: JAXArray

    def __init__(
        self,
        X: JAXArray,
        y: JAXArray | NDArray,
        yerr: JAXArray | NDArray,
        kernel: tinygp.kernels.quasisep.Quasisep,
        **kwargs,
    ) -> None:
        super().__init__(X, y, yerr, kernel, **kwargs)
        self.filtered_bands = kwargs.get("filtered_bands", None)

    def amp_transform(self, params: dict[str, JAXArray]) -> JAXArray:
        b = params["beta"]
        params["log_amp_delta"] = jnp.array([b*np.log(lambda_pivot[band]/lambda_pivot[self.filtered_bands[0]]) for band in self.filtered_bands[1:]]) # comment this out for old version
        r = jnp.insert(jnp.atleast_1d(params["log_amp_delta"]), 0, 0.0)
        return r
    pass
    
def initSampler(key, nSample, nBand=len(bands)):
    # split keys
    subkeys = jax.random.split(key, 10)

    # uniform sampler
    lagSampler = UniformInit(nBand-1, [-10, 10])
    meanSampler = UniformInit(nBand, [-1, 1])
    logAmpDeltaSampler = UniformInit(nBand-1, [-2, 0.0])
    logJitterSampler = UniformInit(nBand, [-20, -5])
    betaSampler = UniformInit(1, [-2.0, 0.0])

    # kernel init
    kernelSampler = DRWInit([jnp.log(100), jnp.log(0.1)], [jnp.log(0.01), 0.0])

    return {
        "log_kernel_param": kernelSampler(subkeys[0], nSample),
        "log_amp_delta": logAmpDeltaSampler(subkeys[1], nSample),
        "mean": meanSampler(subkeys[2], nSample),
        "lag": lagSampler(subkeys[3], nSample),
        "log_jitter": logJitterSampler(subkeys[4], nSample),
        "beta": betaSampler(subkeys[5], nSample),
    }
def numpyro_model(X, yerr, y=None, bestP=None, filtered_bands=None):
    # kernel param
    flat_normal = dist.Normal(bestP["log_kernel_param"], jnp.array([5.0, 5.0]))
    diag_normal = dist.Independent(flat_normal, 1)
    log_kernel_param = numpyro.sample("log_kernel_param", diag_normal)

    # log amp delta
    #log_amp_delta = numpyro.sample(
    #   "log_amp_delta", dist.Normal(bestP["log_amp_delta"], 2.0)
    #) # comment this out when using beta

    # lag
    lag = numpyro.sample("lag", dist.Normal(bestP['lag'], 10.0))
    
    # log jitter, mean => the prior for these two should be set small, otherwise
    # it is hard to converge
    log_jitter = numpyro.sample("log_jitter", dist.Normal(bestP["log_jitter"], 0.1))
    
    mean = numpyro.sample("mean", dist.Normal(bestP['mean'], 0.1))

    beta = numpyro.sample("beta", dist.Normal(-0.5, 0.25))

    # kernel
    k = kernels.quasisep.Exp(*jnp.exp(log_kernel_param))
    m1 = MyMultiVarModel(X, y, yerr, k, zero_mean=zero_mean, has_jitter=has_jitter, has_lag=has_lag, filtered_bands=filtered_bands)

    sample_params = {
        "log_kernel_param": log_kernel_param,
        #"log_amp_delta": log_amp_delta, # comment this out when using beta
        "lag": lag,
        "mean": mean,
        "log_jitter": log_jitter,
        "beta": beta,
    }
    m1.sample(sample_params)


def fit_multiband(data, progress_bar=False):
    times = data['times']
    mags = data['mags']
    magerrs = data['magerrs']

    # Drop bands that cross the Lyman break
    lyman_break_wavelength = 912  # in Angstroms
    rest_frame_wavelengths = {band: lambda_pivot[band] / (1 + data['z']) for band in bands}
    filtered_bands = [band for band in bands if rest_frame_wavelengths[band] > lyman_break_wavelength]

    # Combine
    all_times = np.concatenate([times[b] for b in filtered_bands])
    all_mags = np.concatenate([mags[b] for b in filtered_bands]) 
    all_magerrs = np.concatenate([magerrs[b] for b in filtered_bands])
    band_idx = np.concatenate([np.full(len(times[b]), i) for i, b in enumerate(filtered_bands)])

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
        if jnp.abs(y[i] - jnp.nanmean(window)) > 3 * st.median_abs_deviation(window):
            mask_outlier[i] = False

    X = (jnp.array(all_times[mask_outlier]) - jnp.min(all_times[mask_outlier]), jnp.array(band_idx[mask_outlier]))
    y = jnp.array(y[mask_outlier])
    yerr = jnp.array(yerr[mask_outlier])
    t = jnp.array(t[mask_outlier])

    # define kernel
    initial_drw_params = {"log_kernel_param": jnp.log(np.array([100.0, 0.35]))}
    k = kernels.quasisep.Exp(*jnp.exp(initial_drw_params["log_kernel_param"]))


    # define model
    m1 = MyMultiVarModel(
        X, y, yerr, k, zero_mean=zero_mean, has_jitter=has_jitter, has_lag=has_lag, filtered_bands=filtered_bands
    )
    bestP, logProb = fit(model=m1, 
                        optimizer=optax.adam(learning_rate=0.1),
                        initSampler=initSampler,
                        prng_key=jax.random.PRNGKey(0),
                        nSample=10_000, nIter=2, nBest=5)

    for k in bestP.keys():
        bestP[k] += 1e-4 * np.random.randn(*bestP[k].shape) 

    nuts_kernel = NUTS(
        partial(numpyro_model, bestP=bestP, filtered_bands=filtered_bands),
        dense_mass=True,
        target_accept_prob=0.9,
        # adapt_step_size=True,
    )

    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_samples,
        num_samples=num_samples,
        num_chains=2,
        progress_bar=progress_bar,
    )

    mcmc.run(jax.random.PRNGKey(1), X, yerr, y=y)
    samples = mcmc.get_samples(group_by_chain=False)
    diagnostics = mcmc.get_extra_fields()
    if np.all(diagnostics['diverging']):
        print(f"Diverging MCMC for quasar {data['object_id']}, skipping.", flush=True)
        return None
    
    save_combined_plot(samples, m1, X, y, yerr, band_idx[mask_outlier], 
                       data['object_id'], filtered_bands=filtered_bands)
    plot_posterior(samples, data['object_id'])
    
    log_tau_RF = np.log10(np.exp(samples['log_kernel_param'][:, 0])/(1+data['z']))
    lower, median, upper = np.percentile(log_tau_RF, [16, 50, 84])
    log_tau_RF_err = 0.5 * (upper - lower) # symmetric uncertainties
    log_tau_RF = median


    lambda_ref = 2500 # Any reference wavelength
    lambda_pivot_RF = lambda_pivot[filtered_bands[0]]/(1 + data['z'])
    
    log_sigma_RF = np.log10(np.exp(samples['log_kernel_param'][:, 1] + samples['beta']*np.log(lambda_ref/lambda_pivot_RF)))
    lower, median, upper = np.percentile(log_sigma_RF, [16, 50, 84])
    log_sigma_RF_err = 0.5 * (upper - lower) # symmetric uncertainties
    log_sigma_RF = median


    log_sigma = np.log10(np.exp(samples['log_kernel_param'][:, 1]))
    beta = samples['beta']
    log_amp_delta = np.array([beta*np.log(lambda_pivot[band]/lambda_pivot[filtered_bands[0]]) for band in filtered_bands])
    #log_amp_delta = samples["log_amp_delta"].T # comment out this line when using beta
    #log_amp_delta = jnp.insert(log_amp_delta, 0, np.zeros(log_amp_delta.shape[1]), axis=0) # comment out this line when using beta

    log_sigma_band = log_sigma+log_amp_delta

    # beta
    lower, median, upper = np.percentile(beta, [16, 50, 84])
    beta_err = 0.5 * (upper - lower) # symmetric uncertainties
    beta = median

    # log_amp_delta
    lower, median, upper = np.percentile(log_amp_delta, [16, 50, 84], axis=1)
    log_amp_delta_err = 0.5 * (upper - lower) # symmetric uncertainties
    log_amp_delta = median

    # log_sigma_band
    lower, median, upper = np.percentile(log_sigma_band, [16, 50, 84], axis=1)    
    log_sigma_band_err = 0.5 * (upper - lower) # symmetric uncertainties
    log_sigma_band = median

    # log_sigma
    lower, median, upper = np.percentile(log_sigma, [16, 50, 84], axis=0)
    log_sigma_err = 0.5 * (upper - lower) # symmetric uncertainties
    log_sigma = median

    log_jitter = np.percentile(np.log10(np.exp(2*samples['log_jitter'])), 50, axis=0)


    d = dict(log_tau_RF=log_tau_RF,
            log_tau_RF_err=log_tau_RF_err,
            beta=beta,
            beta_err=beta_err,
            log_sigma_RF=log_sigma_RF,
            log_sigma_RF_err=log_sigma_RF_err,
            log_sigma_band=log_sigma_band,
            log_sigma_band_err=log_sigma_band_err,
            log_amp_delta=log_amp_delta,
            log_amp_delta_err=log_amp_delta_err,
            log_sigma=log_sigma,
            log_sigma_err=log_sigma_err,
            log_jitter=log_jitter)
    return d


def process_quasar(i_data, n=0, progress_bar=False):
    i, data = i_data
    print(f"Processing quasar {i}/{n} ({data['object_id']})", flush=True)

    # Load the quasar data
    result = fit_multiband(data, progress_bar=progress_bar)
    if result is None:
        print(f"Skipping quasar {data['object_id']} due to diverging MCMC.")
        return None
    data['i'] = i
    data |= result


    print(f"Quasar {i}/{n} ({data['object_id']}): log_tau_RF={data['log_tau_RF']:.3f}±{data['log_tau_RF_err']:.3f}, log_sigma_RF={data['log_sigma_RF']}±{data['log_sigma_RF_err']}", 
          flush=True)
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

def plot_posterior(samples, object_id):
    # Extract the posterior samples
    posterior_samples = {
        r'$\beta$': samples['beta'],
        r'$\log(\tau_\mathrm{RF})$': np.log10(np.exp(samples['log_kernel_param'][:, 0])),
        r'$\log(\sigma)$': np.log10(np.exp(samples['log_kernel_param'][:, 1]))
    }
    # Convert the samples to a 2D array for corner
    data = np.vstack([posterior_samples[key] for key in posterior_samples.keys()]).T
    fig = corner.corner(data, labels=list(posterior_samples.keys()), show_titles=True)
    output_dir = "posterior_plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{object_id}_posterior.png"))
    return fig
    plt.close(fig)

def save_combined_plot(samples, model, X, y, yerr, band_idx, object_id, filtered_bands):
    band_idx_map = {i: b for i, b in enumerate(filtered_bands)}

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
    offsets = np.arange(len(bands)) * 0.25

    t = X[0]    
    for n in np.unique(band_idx):
        m = band_idx == n
        # Plot the observed data
        ax.errorbar(t[m], y[m]+offsets[n], yerr=yerr[m], fmt='o', 
                    label=f'{band_idx_map[n]}-band', alpha=0.7, color=colors[band_idx_map[n]])
        # Generate test times for predictions
        t_test = np.linspace(t.min(), t.max(), 1000)
        # Compute predictions using the model
        posterior_median = {k: jnp.median(v, axis=0) for k, v in samples.items()}
        mu, std = model.pred(posterior_median, (t_test, np.full_like(t_test, n, dtype=int)))
        # Plot the predictions
        ax.plot(t_test, mu+offsets[n], alpha=0.7, color=colors[band_idx_map[n]])
        ax.fill_between(t_test, mu+offsets[n]-std, mu+offsets[n]+std, alpha=0.3, 
                        color=colors[band_idx_map[n]])
    ax.set_ylabel('Magnitude + arbitrary offset')
    ax.invert_yaxis()  # Magnitudes are brighter when lower
    #ax.legend(loc='upper right')
    # Annotate the legend with each letter in the same color
    for i, band in enumerate(np.flip(filtered_bands)):
        ax.annotate(
            band.upper(),
            xy=(0.95 - i * 0.05, 0.95),  # Adjust horizontal spacing
            xycoords="axes fraction",
            color=colors[band],
            fontsize=18,
            fontweight="bold",
            ha="right",
            va="top",
        )
    ax.set_title(f'Light Curve for AGN {object_id}')

    plt.tight_layout()

    # Save the plot as a PNG file
    output_dir = "light_curves_fits"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{object_id}_combined_plot.png'))
    plt.show()
    return fig
    plt.close(fig)


def concat_light_curves(filter_object_ids=None, save_file_path=None):
    s82_objs = []

    # Load the S82 data from the FITS file
    hdul = fits.open('data/dr16q_prop_May01_2024.fits')
    fits_data = hdul[1].data  # Assuming the data is in the first extension    

    cat = pd.read_parquet(f"data/S82/Catalog.parquet").set_index('idx')
    sdss = pd.read_parquet(f"data/S82/dr16s82_sdssLCRaw.parquet")
    sdss = sdss[sdss.mjd.notna() & (len(sdss.mjd) > 0)]
    ps1 = pd.read_parquet(f"data/S82/dr16s82_ps1LCRaw.parquet")
    ztf = pd.read_parquet(f"data/S82/dr16s82_ZuberLCRaw.parquet")

    # Find elements in cat where objectId exists in the list of objectId of sdss
    match_object_ids = set(sdss.objectId) if filter_object_ids is None else filter_object_ids
    matching_indices = cat[cat.objectId.isin(match_object_ids)].index
    cat = cat.loc[matching_indices]
    #cat = cat[:5000]

    print(f"Found {len(cat)} matching objects", len(cat))

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
                'g': row.sdss_g_qg - row.ps1_g_qg,
                'r': row.sdss_r_qg - row.ps1_r_qg,
                'i': row.sdss_i_qg - row.ps1_i_qg,
                'z': row.sdss_z_qg - row.ps1_z_qg,
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
            mags[band] = mags[band] - np.nanmean(mags[band])  # Center the magnitudes

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
    
    if save_file_path is not None:
        # Write s82_objs to an HDF5 file
        with h5py.File(save_file_path, "w") as hdf:
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
    print(f"Populating SDSS fields: {len(s82_objs)}", flush=True)
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

    return s82_objs

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Process quasars with optional filtering.")
    parser.add_argument("--filter_object_id", nargs="+", help="List of object IDs to filter.")
    args = parser.parse_args()

    filter_object_ids = set(args.filter_object_id) if args.filter_object_id else None
    if filter_object_ids is not None:
        print(f"Filtering object IDs: {filter_object_ids}")
        s82_objs = concat_light_curves(filter_object_ids=filter_object_ids)
        for i, obj in enumerate(s82_objs):
            q = process_quasar((i, obj), n=len(s82_objs), progress_bar=True)
            fields_to_filter = ['times', 'mags', 'magerrs']
            q = {k: v for k, v in q.items() if k not in fields_to_filter}
            print(q)
    
        sys.exit("Exiting the program as requested.")

    filter_df = pd.read_csv("data/object_ids_test.csv", dtype={"object_id": str})
    filter_object_ids = set(filter_df["object_id"])
    print(f"Loaded {len(filter_object_ids)} object IDs from filter_object_ids.csv")
    objs = concat_light_curves(filter_object_ids=filter_object_ids)

    chunk_size = 50
    for start_idx in range(0, len(objs), chunk_size):
        print("========================================================================")
        print(f"Processing chunk {start_idx // chunk_size + 1}/{(len(objs) + chunk_size - 1) // chunk_size}...")
        chunk = objs[start_idx:start_idx + chunk_size]
        
        ctx = get_context("spawn")  # Safer for JAX when using multiprocessing
        results = []

        with ctx.Pool(processes=14) as pool:
            results = pool.map(partial(process_quasar, n=len(chunk)), enumerate(chunk))

        quasar_list = [q for q in results if q is not None]

        fields_to_filter = ['times', 'mags', 'magerrs']
        filtered_quasar_list = [{k: v for k, v in q.items() if k not in fields_to_filter} for q in quasar_list]

        df = pd.DataFrame.from_records(filtered_quasar_list)
        output_file = 'data/s82_objects_ids_test.csv'
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            merged_df = pd.concat([existing_df, df], ignore_index=True)
            merged_df.to_csv(output_file, index=False)
        else:
            df.to_csv(output_file, index=False)