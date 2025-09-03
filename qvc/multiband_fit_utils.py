import h5py
import os
import numpy as np
import jax.numpy as jnp

prefix = os.environ.get('PREFIX', "test")
suffix = os.environ.get('SUFFIX', "test")

import logging

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

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
#bands = ['g', 'r', 'i']


import os
import jax.numpy as jnp
from datetime import datetime

import numpy as np

def pad_batch(batch_data, nBands):
    """
    Pads each object's time, band, y, and yerr arrays to the max length in the batch.
    Returns a jax array of shape (batch_size, N_max, 4) with columns:
    0 = t, 1 = b, 2 = y, 3 = yerr
    """
    import jax.numpy as jnp

    lengths = np.array([len(obj["X"][0]) for obj in batch_data], dtype=int)
    N_max = int(lengths.max())
    B = len(batch_data)

    arr = np.zeros((B, N_max, 4), dtype=float)
    arr[..., 1] = 0      # band default
    arr[..., 3] = 999.0    # yerr default

    for i, obj in enumerate(batch_data):
        t = np.asarray(obj["X"][0])
        b = np.asarray(obj["X"][1])
        y = np.asarray(obj["y"])
        yerr = np.asarray(obj["yerr"])
        n = len(t)
        arr[i, :n, 0] = t
        arr[i, :n, 1] = b
        arr[i, :n, 2] = y
        arr[i, :n, 3] = yerr

    return jnp.array(arr)

def select_samples_for_object(samples_flat, obj_index, universal_params):
    """
    Select samples for a specific object from flat samples.
    
    Parameters
    ----------
    samples_flat : dict
        Dictionary of flat samples with keys as parameter names.
    obj_index : int
        Index of the object to select samples for.
    universal_params : list
        List of universal parameter names to keep.

    Returns
    -------
    dict
        Dictionary with selected samples for the object.
    """
    obj_samples = {
            k: v[:, obj_index] if k not in universal_params else v
            for k, v in samples_flat.items()
        }    

    # Print shapes for inspection
    logging.debug("Selected object samples: " + ", ".join(f"{k}={v.shape}" for k, v in obj_samples.items()))

    return obj_samples

def flatten_flat_samples_per_band(samples_flat, bands=['u', 'g', 'r', 'i', 'z']):
    """
    Flatten flat samples for each band.
    
    Parameters
    ----------
    samples_flat : dict
        Dictionary of flat samples with keys as parameter names.
    bands : list
        List of bands to flatten.

    Returns
    -------
    dict
        Dictionary with flattened samples for each band.
    """
    flattened_samples = {}
    for k, v in samples_flat.items():
        if v.ndim == 1:
            flattened_samples[k] = v
        elif v.ndim == 2:
            # Flatten over bands
            for i, band in enumerate(bands):
                flattened_samples[f"{k}_{band}"] = v[:, i]
        else:
            raise ValueError(f"Unexpected shape for {k}: {v.shape}") 
    return flattened_samples


def clean_grouped_samples(samples_grouped, obj_index, batch_data_len):
    raise NotImplementedError("clean_grouped_samples is not implemented yet.")
    """
    Clean grouped samples (group_by_chain=True) in your style:
    - Universal params kept as-is (flattened over chains)
    - Object-specific params indexed [:, :, i]
    """
    universal_keys = ['eta_A1', 'eta_A2', 'eta_tau1', 'eta_tau2', 'eta_break', 'lam_s']

    # Print shapes for inspection
    for k, v in samples_grouped.items():
        print(f"{k}={v.shape}", end='; ')

    obj_samples_clean = dict()
    for k, v in samples_grouped.items():
        arr = np.asarray(v)  # shape: (n_chains, n_samples[, ...])

        # Flatten chains into single axis first: (n_samples_total, ...)
        arr = arr.reshape(-1, *arr.shape[2:])

        # Select per-object slice if needed
        if arr.ndim == 2 and arr.shape[1] == batch_data_len:
            arr = arr[:, obj_index]
        elif arr.ndim == 3 and arr.shape[1] == batch_data_len:
            arr = arr[:, obj_index, :]

        # Flatten dimensions for plotting
        if arr.ndim == 1:
            obj_samples_clean[k] = arr
        elif arr.ndim == 2:
            print("Flattening 2D array for parameter:", k)
            print("Shape before flattening:", arr.shape)
            for j in range(arr.shape[1]):
                obj_samples_clean[f"{k}_{j}"] = arr[:, j]

    return obj_samples_clean



import numpy as np
import logging

def compute_rhat_ess_dict(samples_dict):
    """
    Compute R-hat and ESS for a dict of MCMC chains without ArviZ.
    
    Parameters
    ----------
    samples_dict : dict[str, np.ndarray]
        Keys are parameter names.
        Values are arrays of shape:
        - (n_samples, n_chains) for scalar/global params
        - (n_samples, n_chains) for per-object params (already split)

    Returns
    -------
    dict[str, float]
        Dictionary mapping "<param>_rhat" and "<param>_ess" to values.
    """
    logging.info("Computing Rhat and ESS for dictionary of parameters")
    diagnostics = {}

    for k, chains in samples_dict.items():
        chains = np.asarray(chains)

        # Ensure shape: (n_samples, n_chains, n_params)
        if chains.ndim == 1:
            raise ValueError(f"Parameter {k} is 1D; need multiple chains for R-hat.")
        elif chains.ndim == 2:  # (n_samples, n_chains)
            chains = chains[..., None]
        elif chains.ndim != 3:
            raise ValueError(f"Parameter {k} has invalid shape {chains.shape}")

        n_samples, n_chains, n_params = chains.shape

        # Compute chain means and variances
        chain_means = chains.mean(axis=0)             # (n_chains, n_params)
        chain_variances = chains.var(axis=0, ddof=1)  # (n_chains, n_params)

        # Between-chain variance B
        B = n_samples * np.var(chain_means, axis=0, ddof=1)

        # Within-chain variance W
        W = chain_variances.mean(axis=0)

        # Marginal posterior variance estimate
        var_hat = (n_samples - 1)/n_samples * W + B/n_samples

        # R-hat (Gelman-Rubin)
        rhat = np.sqrt(var_hat / W)

        # Effective Sample Size (naive approximation)
        ess = (n_chains * n_samples) / rhat**2

        # Store in flat dict
        if n_params == 1:
            diagnostics[f"{k}_rhat"] = float(rhat.squeeze())
            diagnostics[f"{k}_ess"] = float(ess.squeeze())
        else:
            for i in range(n_params):
                diagnostics[f"{k}_{i}_rhat"] = float(rhat[i])
                diagnostics[f"{k}_{i}_ess"] = float(ess[i])

    return diagnostics

def modify_h5_file(save_file_path, s82_objs):
    with h5py.File(save_file_path, "a") as hdf:  # Open in append mode to modify
        for object_id in hdf.keys():  # Iterate through every object in the HDF5 file
            group = hdf[object_id]

            # Delete specified keys if they exist
            for key in ["mags", "times", "magerrs"]:
                if key in group:
                    del group[key]


def bands_redder_than(z, threshold=4000):
    """
    Returns a list of bands with rest-frame effective wavelength > 4000 Å.

    Args:
        z (float): Redshift.

    Returns:
        list: Bands redder than 4000 Å at rest frame.
    """
    bands = {
        'u': {'lambda_eff': 3551},
        'g': {'lambda_eff': 4686},
        'r': {'lambda_eff': 6165},
        'i': {'lambda_eff': 7481},
        'z': {'lambda_eff': 8931},
        'y': {'lambda_eff': 9700}
    }

    redder_bands = []
    for band, props in bands.items():
        rest_lambda_eff = props['lambda_eff'] / (1 + z)
        if rest_lambda_eff > threshold:
            redder_bands.append(band)

    return redder_bands

def bands_bluer_than_lyman_alpha(z, buffer=100):
    """
    Returns a list of bands with rest-frame effective wavelength < Lyman-alpha (1216 Å).

    Args:
        z (float): Redshift.

    Returns:
        list: Bands bluer than Lyman-alpha at rest frame.
    """
    lyman_alpha = 1216
    bands = {
        'u': {'lambda_eff': 3551},
        'g': {'lambda_eff': 4686},
        'r': {'lambda_eff': 6165},
        'i': {'lambda_eff': 7481},
        'z': {'lambda_eff': 8931},
        'y': {'lambda_eff': 9700}
    }

    bluer_bands = []
    for band, props in bands.items():
        rest_lambda_eff = props['lambda_eff'] / (1 + z)
        if rest_lambda_eff < (lyman_alpha - buffer):
            bluer_bands.append(band)

    return bluer_bands


def load_all_samples_from_hdf5():
    """
    Load all samples from an HDF5 file.
    
    Args:
        prefix (str): Prefix used for the samples directory and filename.
        suffix (str): Suffix used in the filename.
    
    Returns:
        dict: Dictionary containing all loaded samples.
    """

    output_dir=f"results/samples/{prefix}/"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"all_{suffix}.h5")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")

    logging.info(f"Loading all samples from {file_path}")

    samples = {}
    with h5py.File(file_path, "r") as hdf:
        for key in hdf.keys():
            samples[key] = np.array(hdf[key])

    logging.info(f"Loaded {len(samples)} datasets from {file_path}")
    return samples

def save_all_samples_to_hdf5(samples):
    """
    Save all samples to an HDF5 file
    Args:
        samples (dict): Dictionary containing MCMC samples.
    """
    output_dir=f"results/samples/{prefix}/"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"all_{suffix}.h5")

    logging.info(f"Saving all samples to {file_path}")

    with h5py.File(file_path, "w") as hdf:
        for key, value in samples.items():
            hdf.create_dataset(key, data=value)
    logging.info(f"Saved all samples to {file_path}")

def save_obj_samples_to_hdf5(samples, object_id):
    """
    Save all samples to an HDF5 file, one file per object_id.

    Args:
        samples (dict): Dictionary containing MCMC samples.
        object_id (str): The object ID for which the samples belong.
        output_dir (str): Directory where the HDF5 files will be saved.
    """
    output_dir=f"results/samples/{prefix}"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{object_id}_{suffix}.h5")

    logging.info(f"Saving samples for object_id {object_id} to {file_path}")

    with h5py.File(file_path, "w") as hdf:
        for key, value in samples.items():
            hdf.create_dataset(key, data=value)
    logging.info(f"Saved samples for object_id {object_id} to {file_path}")

def delete_file(file_path):
    """
    Delete a file if it exists.
    """
    if os.path.exists(file_path):
        os.remove(file_path)
        logging.info(f"Deleted existing file: {file_path}")
    else:
        logging.info(f"File does not exist; not deleting: {file_path}")

def save_quasar_list_hdf5(quasars, ignored_keys=None, size_threshold=1024):
    """
    Save a list of quasar dictionaries to an HDF5 file, overwriting the file if it exists.
    
    - The file is always truncated to start fresh.
    - Each quasar is stored under a group named by its object_id.
    - Nested dicts become sub-groups with datasets.
    - Simple values are stored as attributes.
    - Keys in ignored_keys or arrays larger than size_threshold are skipped.
    - Prints progress in the form 'i/N: Saved quasar <object_id>'.
    """
    ignored_keys = set(ignored_keys or [])
    
    string_dt = h5py.string_dtype(encoding="utf-8")

    total = len(quasars)
    output_dir=f"results/data/{prefix}"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{suffix}.h5")
    logging.info(f"Saving {total} quasars to {file_path}")

    with h5py.File(file_path, "w") as hdf:
        for i, quasar in enumerate(quasars):
            object_id = str(quasar["object_id"])
            logging.info(f"Writing quasar {object_id} to {file_path}")
            
            group = hdf.create_group(object_id)
            for key, value in quasar.items():
                # Skip ignored keys
                if key in ignored_keys:
                    continue

                if isinstance(value, dict):
                    # Nested dict → sub-group with datasets
                    sub_group = group.create_group(key)
                    for sub_key, sub_value in value.items():
                        arr = np.asarray(sub_value)
                        if arr.size > size_threshold:
                            logging.warning(f"Warning: Skipping sub-key '{key}/{sub_key}' (too large: {arr.size})")
                            continue
                        if arr.dtype.kind in {'U', 'S', 'O'}:
                            arr = arr.astype(string_dt)
                        sub_group.create_dataset(sub_key, data=arr)
                else:
                    # Attributes for simple values
                    arr = np.asarray(value)
                    if arr.size > size_threshold:
                        logging.warning(f"Warning: Skipping key '{key}' (too large: {arr.size})")
                        continue
                    if arr.dtype.kind in {'U', 'S', 'O'}:
                        arr = arr.astype(string_dt)
                    group.attrs[key] = arr
            
            # Print progress
            logging.info(f"{i+1}/{total}: Saved quasar {object_id}")

    logging.info("All quasars saved successfully.")
    

def log_broken_pl(lam, lam_s, d1, d2, ds=0.1):
    """
    Log10 of a smooth broken power-law, normalized to 0 at lam_s.
    Slopes approach d1 for lam << lam_s and d2 for lam >> lam_s.
    
    ds: smoothness control — larger ds = smoother transition,
        smaller ds = sharper transition.
    """
    x = lam / lam_s
    delta = d2 - d1

    # Use exponent 1/ds so larger ds => smoother
    smooth_exp = 1.0 / ds
    log10_1px = jnp.log1p(x**smooth_exp) / jnp.log(10.0)

    log_f = d1 * jnp.log10(x) + (delta / smooth_exp) * log10_1px
    log_f -= (delta / smooth_exp) * jnp.log10(2.0)  # normalize to 0 at lam_s

    return log_f

def process_samples(flat_samples, data, percentiles=[16, 50, 84], bands=['u', 'g', 'r', 'i', 'z']):
    """
    Generalized processing of MCMC samples for arbitrary parameters and bands.

    Args:
        flat_samples (dict): Dictionary of flat MCMC samples, each value is (n_samples,).
        data (dict): Data dictionary, must contain 'object_id', 'z', and 'clean_bands'.
        percentiles (list): Percentiles for summary statistics.

    Returns:
        dict: Summary statistics for all parameters and bands.
    """

    def sym_percentile(x, p=percentiles, axis=0):
        lower, median, upper = np.percentile(x, p, axis=axis)
        return median, 0.5 * (upper - lower)

    result = dict(object_id=data['object_id'], z=data['z'])

    # per flat param computation
    for k, v in flat_samples.items():
        if v.ndim > 1:
            logging.warning(f"Warning: {k} has shape {v.shape}, expected flat samples")
        # Convert to base 10
        if 'log_' in k:
            v = v / np.log(10)
        # Calculate median and 1-sigma error
        median, err = sym_percentile(v)
        result[k] = median
        result[f"{k}_err"] = err

    # generalized per-band computation
    # Power Law Params
    log_sigma_hat0 = np.asarray(flat_samples["log_sigma_hat0"])
    log_sigma0 = np.asarray(flat_samples["log_sigma0"])
    log_tau_drw0 = np.asarray(flat_samples["log_tau_drw0"])
    eta_A1 = np.asarray(flat_samples["eta_A1"])
    eta_A2 = np.asarray(flat_samples["eta_A2"])
    eta_tau1 = np.asarray(flat_samples["eta_tau1"])
    eta_tau2 = np.asarray(flat_samples["eta_tau2"])
    eta_break = np.asarray(flat_samples["eta_break"])
    lambda_ref = 2500
    lam_s = np.asarray(flat_samples["lam_s"])

    log_sigma_band = []
    for band in bands:
        lam_eff = lambda_pivot[band] / (1 + data['z'])
        val = log_sigma0 / np.log(10) + log_broken_pl(lam_eff, lam_s, eta_A1, eta_A2, eta_break)
        log_sigma_band.append(val)
    log_sigma_band = np.array(log_sigma_band).T

    log_tau_band = []
    for band in bands:
        lam_eff = lambda_pivot[band] / (1 + data['z'])
        val = log_tau_drw0 / np.log(10) - np.log10(1 + data['z']) + log_broken_pl(lam_eff, lam_s, eta_A1, eta_A2, eta_break)
        log_tau_band.append(val)
    log_tau_band = np.array(log_tau_band).T

    for i, band in enumerate(bands):
        median, err = sym_percentile(log_sigma_band[:, i])
        result[f"log_sigma_band_{band}"] = median
        result[f"log_sigma_band_{band}_err"] = err
        median, err = sym_percentile(log_tau_band[:, i])
        result[f"log_tau_band_{band}_RF"] = median
        result[f"log_tau_band_{band}_RF_err"] = err

    # Other special params
    host_frac = flat_samples["f_host"] * (lambda_ref / 5100.0) ** flat_samples["alpha_host"]
    dilution_factor = 1.0 / (1.0 + host_frac)
    log_dilution = jnp.log(dilution_factor)

    # log_sigma_UV_diluted
    samples_log_sigma_UV_diluted = (flat_samples["log_sigma0"] - log_dilution) / np.log(10) + log_broken_pl(lambda_ref, lam_s, eta_A1, eta_A2, eta_break)
    result['log_sigma_UV_diluted'], result['log_sigma_UV_diluted_err'] = sym_percentile(samples_log_sigma_UV_diluted)

    # log_sigma_UV
    samples_log_sigma_UV = flat_samples["log_sigma0"] / np.log(10) + log_broken_pl(lambda_ref, lam_s, eta_tau1, eta_tau2, eta_break)
    result['log_sigma_UV'], result['log_sigma_UV_err'] = sym_percentile(samples_log_sigma_UV)

    # log_tau_UV_RF
    samples_log_tau_UV_RF = flat_samples["log_tau_drw0"] / np.log(10) - np.log10(1 + data['z']) + log_broken_pl(lambda_ref, lam_s, eta_tau1, eta_tau2, eta_break)
    result['log_tau_UV_RF'], result['log_tau_UV_RF_err'] = sym_percentile(samples_log_tau_UV_RF)

    # Compute covariance between log_sigma_UV and log_tau_UV_RF
    cov_matrix = np.cov(samples_log_sigma_UV, samples_log_tau_UV_RF)
    cov_log_sigma_tau = cov_matrix[0, 1]
    result['cov_log_sigma_UV_log_tau_UV_RF'] = cov_log_sigma_tau

    return result

def drw_equiv(amp_cont, tau_drw, bwb_alpha, bwb_beta):
    A = amp_cont
    q2 = (bwb_alpha * A**2)**2  # q_b^2

    sigma2_eq = A**2 + 2.0 * q2
    tau_eq = tau_drw * (A**2 + (2.0/bwb_beta)*q2) / (A**2 + 2.0*q2)
    return tau_eq, sigma2_eq