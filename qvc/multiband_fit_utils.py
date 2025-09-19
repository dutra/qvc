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

import logging
import numpy as np

def select_samples_for_object_per_chain(samples_per_chain, obj_index, universal_params):
    """
    Select samples for a specific object from per-chain samples.
    
    Parameters
    ----------
    samples_per_chain : dict
        Dictionary of samples grouped by chain. 
        Keys are parameter names, values have shape (n_chains, n_draws, ...) 
    obj_index : int
        Index of the object to select samples for.
    universal_params : list
        List of universal parameter names to keep (same across objects).

    Returns
    -------
    dict
        Dictionary with selected samples for the object, preserving chain structure.
    """
    obj_samples = {
        k: v[:, :, obj_index] if (k not in universal_params and v.ndim > 2) else v
        for k, v in samples_per_chain.items()
    }

    # Print shapes for inspection
    logging.debug(
        "Selected object samples per chain: " 
        + ", ".join(f"{k}={v.shape}" for k, v in obj_samples.items())
    )

    return obj_samples


def flatten_per_chain_samples_per_band(samples_per_chain, bands=['u', 'g', 'r', 'i', 'z']):
    """
    Flatten per-chain samples for each band.
    
    Parameters
    ----------
    samples_per_chain : dict
        Dictionary of samples grouped by chain. 
        Keys are parameter names, values have shape (n_chains, n_draws, ...).
    bands : list
        List of bands to flatten.

    Returns
    -------
    dict
        Dictionary with flattened samples for each band, preserving chain structure.
    """
    flattened_samples = {}
    for k, v in samples_per_chain.items():
        logging.debug(f"flatten_per_chain: {k} shape={getattr(v, 'shape', None)}")
        if v.ndim == 2:
            flattened_samples[k] = v
        elif v.ndim == 3:
            if v.shape[-1] != len(bands):
                raise ValueError(f"Unexpected band dimension for {k}: {v.shape} vs bands={len(bands)}")
            for i, band in enumerate(bands):
                flattened_samples[f"{k}_{band}"] = v[:, :, i]
        else:
            raise ValueError(f"Unexpected shape for {k}: {v.shape}")
    return flattened_samples

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

def regularize_cov_from_percentiles(x16, x84, y16, y84, cov_xy, eps=1e-8):
    # 1) variance estimates from central 68% interval
    sx = 0.5 * (x84 - x16)
    sy = 0.5 * (y84 - y16)
    vx, vy = max(sx*sx, 0.0), max(sy*sy, 0.0)
    # early exit if any variance is ~0: set cov to 0
    if vx <= 0 or vy <= 0:
        return max(vx, eps), max(vy, eps), 0.0
    # 2) clip covariance to be within [-sqrt(vx*vy), +sqrt(vx*vy)]
    rho_raw = cov_xy / (sx * sy)
    rho = max(min(rho_raw, 1.0 - eps), -1.0 + eps)
    cov_xy_reg = rho * sx * sy
    return vx, vy, cov_xy_reg

def psd_cov_from_samples(X, Y, eps=1e-12, shrink_rho=0.0):
    X = np.asarray(X); Y = np.asarray(Y)
    C = np.cov(np.vstack([X, Y]), bias=False)  # moments in the correct units
    # shrink correlation if desired
    sx = np.sqrt(max(C[0,0], eps)); sy = np.sqrt(max(C[1,1], eps))
    rho = C[0,1] / max(sx*sy, eps)
    rho = (1.0 - shrink_rho) * rho
    rho = np.clip(rho, -1.0 + eps, 1.0 - eps)
    C = np.array([[sx*sx, rho*sx*sy],[rho*sx*sy, sy*sy]])
    return C

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
        val = log_tau_drw0 / np.log(10) - np.log10(1 + data['z']) + log_broken_pl(lam_eff, lam_s, eta_tau1, eta_tau2, eta_break)
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
    samples_log_sigma_UV = flat_samples["log_sigma0"] / np.log(10) + log_broken_pl(lambda_ref, lam_s, eta_A1, eta_A2, eta_break)
    result['log_sigma_UV'], result['log_sigma_UV_err'] = sym_percentile(samples_log_sigma_UV)

    # log_tau_UV_RF
    samples_log_tau_UV_RF = flat_samples["log_tau_drw0"] / np.log(10) - np.log10(1 + data['z']) + log_broken_pl(lambda_ref, lam_s, eta_tau1, eta_tau2, eta_break)
    result['log_tau_UV_RF'], result['log_tau_UV_RF_err'] = sym_percentile(samples_log_tau_UV_RF)

    # Compute covariance between log_sigma_UV and log_tau_UV_RF
    cov_matrix = np.cov(samples_log_sigma_UV, samples_log_tau_UV_RF)
    cov_log_sigma_tau = cov_matrix[0, 1]
    result['cov_log_sigma_UV_log_tau_UV_RF'] = cov_log_sigma_tau

    vx, vy, cov_log_sigma_tau_reg = regularize_cov_from_percentiles(
        np.percentile(samples_log_sigma_UV, 16),
        np.percentile(samples_log_sigma_UV, 84),
        np.percentile(samples_log_tau_UV_RF, 16),
        np.percentile(samples_log_tau_UV_RF, 84),
        cov_log_sigma_tau
    )
    result['cov_log_sigma_UV_log_tau_UV_RF'] = cov_log_sigma_tau_reg
    print("Regularized covariance: ", cov_log_sigma_tau_reg)

    C = psd_cov_from_samples(samples_log_sigma_UV, samples_log_tau_UV_RF, shrink_rho=0.05)
    sx2, sy2, sxy = C[0,0], C[1,1], C[0,1]
    result['log_sigma_UV_log_tau_UV_RF_cov_psd'] = sxy
    result['log_sigma_UV_std_psd'] = np.sqrt(sx2)
    result['log_tau_UV_RF_std_psd'] = np.sqrt(sy2)
    print("PSD covariance: ", sxy)

    return result

def drw_equiv(amp_cont, tau_drw, bwb_alpha, bwb_beta):
    A = amp_cont
    q2 = (bwb_alpha * A**2)**2  # q_b^2

    sigma2_eq = A**2 + 2.0 * q2
    tau_eq = tau_drw * (A**2 + (2.0/bwb_beta)*q2) / (A**2 + 2.0*q2)
    return tau_eq, sigma2_eq


import numpy as np
import logging

def _acf_1d_fft(x, max_lag=None):
    """
    Fast autocorrelation for a 1D array x.
    Returns lags 0..L where L = min(max_lag, n-1) (or n-1 if max_lag=None).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return np.array([1.0])

    # Clamp lag
    L = n - 1 if (max_lag is None) else min(max_lag, n - 1)

    x = x - x.mean()
    var = np.dot(x, x) / n
    if var == 0 or not np.isfinite(var):
        acf = np.zeros(L + 1, dtype=float); acf[0] = 1.0
        return acf

    # Next power of two >= 2n
    nfft = 1
    while nfft < 2 * n:
        nfft <<= 1

    fx = np.fft.rfft(x, nfft)
    acov = np.fft.irfft(fx * np.conjugate(fx), nfft)[:n] / n
    acf_full = acov / var
    return acf_full[:L + 1]


def _avg_acf_over_chains(arr_mc, max_lag=None):
    """
    Average ACF over chains for arr_mc with shape (m, n).
    Returns mean ACF over chains with length L+1, where L is clamped to n-1.
    """
    m, n = arr_mc.shape
    L = n - 1 if (max_lag is None) else min(max_lag, n - 1)
    acfs = np.empty((m, L + 1), dtype=float)
    for j in range(m):
        acfs[j] = _acf_1d_fft(arr_mc[j], max_lag=L)
    if max_lag is not None and L < max_lag:
        logging.debug(f"_avg_acf_over_chains: clipped max_lag to {L} (n_draws={n})")
    return acfs.mean(axis=0)


def _ess_from_avg_acf(avg_acf):
    """Geyer IPS on averaged acf (avg_acf[0]=1)."""
    rho = np.asarray(avg_acf[1:], dtype=float)
    s = 0.0
    for i in range(0, rho.size, 2):
        pair_sum = rho[i] if i+1 >= rho.size else (rho[i] + rho[i+1])
        if not np.isfinite(pair_sum) or pair_sum <= 0:
            break
        s += pair_sum
    tau = 1.0 + 2.0 * s
    return 1.0 if (not np.isfinite(tau) or tau <= 0) else tau


def _split_rhat(arr_mc):
    """Classic split-Rhat using true chains."""
    m, n = arr_mc.shape
    if m < 2 or n < 4:
        return np.nan
    if n % 2 == 1:
        arr_mc = arr_mc[:, :-1]; n -= 1
    half = n // 2
    if half < 2:
        return np.nan
    split = np.concatenate([arr_mc[:, :half], arr_mc[:, half:]], axis=0)
    means = split.mean(axis=1)
    W = split.var(axis=1, ddof=1).mean()
    B = half * np.var(means, ddof=1)
    if not np.isfinite(W) or W <= 0:
        return np.nan
    var_hat = (half - 1) / half * W + B / half
    return float(np.sqrt(var_hat / W))


def _longest_near_constant_run(x, tol=0.0):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return n
    diffs = np.abs(np.diff(x))
    mask = diffs <= tol
    longest = 0; cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    return longest + 1 if longest > 0 else 1

def _ess_single_chain(x, max_lag):
    acf = _acf_1d_fft(x, max_lag)
    rho = acf[1:]
    s = 0.0
    for i in range(0, rho.size, 2):
        pair_sum = rho[i] if i+1 >= rho.size else (rho[i] + rho[i+1])
        if not np.isfinite(pair_sum) or pair_sum <= 0:
            break
        s += pair_sum
    tau = 1.0 + 2.0 * s
    N = float(len(x))
    if not np.isfinite(tau) or tau <= 0:
        return N
    ess = N / tau
    return float(min(max(ess, 1.0), N))

# --- main ---
def diagnostics_for_per_chain_samples(
    flattened_per_chain,
    max_lag=100,
    rhat_thresh=1.05,
    ess_thresh=100,
    acf_longlag_thresh=0.1,
    tiny_var_rel_thresh=1e-6,
    tiny_var_abs_thresh=1e-12,
    stuck_tol="auto",          # 'auto' -> 1e-12 * scale, else float
    stuck_run_frac_thresh=0.5, # mark stuck if longest run ≥ this fraction
    per_chain_ess_thresh=50
):
    """
    Returns, per field k:
      k_acf : np.ndarray (max_lag+1,)
      k_rhat : float
      k_ess : float
      k_per_chain_variance : np.ndarray (n_chains,)
      k_stuck_chains : np.ndarray(bool) (n_chains,)
      k_stuck_any : bool
    Logs only the single worst offender per category.
    """
    out = {}

    # Worst (for compact logging)
    worst_rhat = (None, -np.inf)
    worst_ess  = (None,  np.inf)
    worst_acf  = (None, -np.inf, None)
    worst_var  = (None,  np.inf, None, None)
    worst_stuck = (None, -np.inf, None, None)
    worst_pcess = (None,  np.inf, None)

    for k, v in flattened_per_chain.items():
        v = np.asarray(v)
        if v.ndim != 2:
            raise ValueError(f"{k}: expected (n_chains, n_draws), got {v.shape}")
        m, n = v.shape

        if n < 2:
            acf = np.array([1.0]); rhat = np.nan; ess = float(m * n)
        else:
            acf = _avg_acf_over_chains(v, max_lag=max_lag)  # length is L+1 with L<=n-1
            tau = _ess_from_avg_acf(acf)
            Ntot = float(m * n)
            ess = Ntot / tau
            ess = 1.0 if (not np.isfinite(ess) or ess < 1) else min(ess, Ntot)
            rhat = _split_rhat(v)

            
            L = len(acf) - 1  # actual max lag used
            # use L instead of max_lag
            long_lags = acf[L // 2 :]    # last ~half of what we actually computed            
            if long_lags.size:
                idx = int(np.argmax(np.abs(long_lags)))
                max_long = float(np.abs(long_lags[idx]))
                lag_at = idx + max_lag // 2
                if max_long > worst_acf[1]:
                    worst_acf = (k, max_long, lag_at)

        # Per-chain variance
        per_var = v.var(axis=1, ddof=1) if v.size else np.array([])
        mean_var = float(np.mean(per_var)) if per_var.size and np.all(np.isfinite(per_var)) else np.nan

        # Stuck detection settings
        x_range = float(np.nanmax(v) - np.nanmin(v)) if v.size and np.isfinite(v).all() else 0.0
        tol = (1e-12 * max(1.0, x_range)) if stuck_tol == "auto" else float(stuck_tol)

        # Per-chain stuck + per-chain ESS
        longest_runs = np.zeros(m, dtype=int)
        stuck_frac = np.zeros(m, dtype=float)
        stuck_flags = np.zeros(m, dtype=bool)
        per_chain_ess = np.zeros(m, dtype=float)

        for j in range(m):
            # variance flags
            varj = float(per_var[j]) if per_var.size else np.nan
            rel = (varj / mean_var) if (np.isfinite(varj) and np.isfinite(mean_var) and mean_var > 0) else np.inf
            tiny_rel = (rel < tiny_var_rel_thresh)
            tiny_abs = (varj < tiny_var_abs_thresh) if np.isfinite(varj) else False
            if (tiny_rel or tiny_abs) and varj < worst_var[1]:
                worst_var = (k, varj, j, rel)

            # stuck run flags
            run_len = _longest_near_constant_run(v[j], tol=tol)
            longest_runs[j] = run_len
            stuck_frac[j] = run_len / n if n > 0 else 0.0
            # mark stuck if long run OR near-zero variance
            stuck_flags[j] = (stuck_frac[j] >= stuck_run_frac_thresh) or tiny_rel or tiny_abs
            if stuck_frac[j] > worst_stuck[1]:
                worst_stuck = (k, float(stuck_frac[j]), j, int(run_len))

            # per-chain ESS
            per_chain_ess[j] = _ess_single_chain(v[j], max_lag=max_lag)
            if per_chain_ess[j] < worst_pcess[1]:
                worst_pcess = (k, float(per_chain_ess[j]), j)

        # Save outputs
        out[f"{k}_acf"] = acf
        out[f"{k}_rhat"] = float(rhat) if np.isscalar(rhat) else rhat
        out[f"{k}_ess"] = float(ess)
        out[f"{k}_per_chain_variance"] = per_var
        out[f"{k}_stuck_chains"] = stuck_flags
        out[f"{k}_stuck_any"] = bool(stuck_flags.any())

        # Track worst global Rhat/ESS
        if np.isfinite(rhat) and rhat > worst_rhat[1]:
            worst_rhat = (k, float(rhat))
        if np.isfinite(ess) and ess < worst_ess[1]:
            worst_ess = (k, float(ess))

    # --- concise logging of only worst offenders (if exceeding thresholds) ---
    if worst_rhat[0] is not None and worst_rhat[1] > rhat_thresh:
        logging.warning(f"Worst Rhat: {worst_rhat[0]} = {worst_rhat[1]:.3f} (> {rhat_thresh})")
    if worst_ess[0] is not None and worst_ess[1] < ess_thresh:
        logging.warning(f"Worst ESS: {worst_ess[0]} = {worst_ess[1]:.1f} (< {ess_thresh})")
    if worst_acf[0] is not None and worst_acf[1] > acf_longlag_thresh:
        logging.warning(f"Worst ACF: {worst_acf[0]} lag={worst_acf[2]}, |acf|={worst_acf[1]:.3f} (> {acf_longlag_thresh})")
    if worst_var[0] is not None and (worst_var[1] < tiny_var_abs_thresh or (worst_var[3] is not None and worst_var[3] < tiny_var_rel_thresh)):
        logging.warning(
            f"Near-zero per-chain variance: {worst_var[0]} chain={worst_var[2]} "
            f"var={worst_var[1]:.3e}, rel_to_mean={worst_var[3]:.3e} "
            f"(abs<th={tiny_var_abs_thresh}, rel<th={tiny_var_rel_thresh})"
        )
    if worst_stuck[0] is not None and worst_stuck[1] >= stuck_run_frac_thresh:
        logging.warning(
            f"Stuck/flat run: {worst_stuck[0]} chain={worst_stuck[2]} "
            f"longest_run={worst_stuck[3]} draws ({worst_stuck[1]*100:.1f}% of chain) "
            f"(>= {int(stuck_run_frac_thresh*100)}%)"
        )
    if worst_pcess[0] is not None and worst_pcess[1] < per_chain_ess_thresh:
        logging.warning(
            f"Per-chain ESS low: {worst_pcess[0]} chain={worst_pcess[2]} "
            f"ESS={worst_pcess[1]:.1f} (< {per_chain_ess_thresh})"
        )

    return out

def safe_log_jitter_mean(obj_row):
    # obj_row[:,3] is yerr after padding
    yerr = jnp.array(obj_row[:, 3])
    good = yerr < 10.0
    # fallback prior mean jitter if no "good" points exist
    fallback = jnp.log(0.05)  # pick something conservative in log space
    m = jnp.where(good.any(), jnp.log(jnp.mean(yerr[good])), fallback)
    # return a length-5 vector (one per band)
    return jnp.full(5, 1e-6) + m

def resort_by_kernel_key(obj_matrix_np: np.ndarray) -> np.ndarray:
    """Resort a padded object's matrix by the kernel's coord_to_sortable key."""
    # obj_matrix columns: [time, band, y, yerr]
    t = obj_matrix_np[:, 0]
    b = obj_matrix_np[:, 1].astype(t.dtype, copy=False)
    eps = 10.0 * np.finfo(t.dtype).eps
    key = t + b * eps
    order = np.argsort(key, kind="mergesort")  # stable
    return obj_matrix_np[order]



def summarize_fake_true_vs_recovered(
    obj,
    diagnostics,
    compare_pairs=None,
):
    """
    Parameters
    ----------
    obj : dict
        Holds true (ground-truth) values, e.g. obj['alpha_sigma'], obj['log_tau_fake'] (natural log), etc.
    samples : Mapping[str, array_like]
        Posterior draws for recovered params, e.g. samples['eta_A1'], samples['log_tau_drw0'], ...
    diagnostics : Mapping[str, float]
        Diagnostics such as rhat stored under f"{param}_rhat".
    compare_pairs : list of tuples, optional
        Each tuple is (inj_key, rec_key) or (inj_key, rec_key, transform, label).
        - inj_key: key in obj for true value.
        - rec_key: key in samples for posterior draws.
        - transform: optional function to apply to both true and recovered values for display.
        - label: optional label for display (defaults to inj_key if not provided).
    """
    # ANSI colors
    _GREEN = "\033[92m"
    _YELLOW = "\033[93m"
    _RED = "\033[91m"
    _RESET = "\033[0m"


    def _in_bounds(val, p16, p84):
        center = 0.5 * (p16 + p84)
        half_1sigma = 0.5 * (p84 - p16)
        half_2sigma = 2.0 * half_1sigma
        in_1sigma = (center - half_1sigma) <= val <= (center + half_1sigma)
        in_2sigma = (center - half_2sigma) <= val <= (center + half_2sigma)
        return in_1sigma, in_2sigma, center, half_1sigma

    def _colorize(text, in_1sigma, in_2sigma):
        if in_1sigma:
            return f"{_GREEN}{text}{_RESET}"
        elif in_2sigma:
            return f"{_YELLOW}{text}{_RESET}"
        else:
            return f"{_RED}{text}{_RESET}"

    # Header
    print(f"[COMPARE RECOVERY] Object {obj.get('object_id', '<?>')}:")
    # One line per requested comparison
    for r in compare_pairs:
        if len(r) == 2:
            inj_key, rec_key = r
            transform = lambda x: x  # identity
            label = inj_key
        elif len(r) == 3:
            inj_key, rec_key, label = r
        else:
            raise ValueError("compare_pairs entries must be (inj_key, rec_key) or (inj_key, rec_key, label)")
            
        # true value (from obj)
        if inj_key not in obj:
            print(f"  {label}: true=<?> (missing '{inj_key}'), recovered=<?> (key '{rec_key}')")
            continue
        true_val = obj[inj_key]

        # recovered: posterior draws in samples[rec_key]
        if rec_key not in obj:
            print(f"  {label}: true=<?> (key '{inj_key}'), recovered=<?> (missing '{rec_key}')")
            continue
            
        # optional transform for *display* (applies to true, median, bounds, and ±)
        median = obj[rec_key]
        half_1sigma_disp = obj[rec_key + "_err"]
        p16_disp, p84_disp = median - half_1sigma_disp, median + half_1sigma_disp

        in_1sigma, in_2sigma, _, _ = _in_bounds(true_val, p16_disp, p84_disp)
        colored_med = _colorize(f"{median:.3f}", in_1sigma, in_2sigma)

        rhat = diagnostics.get(f"{rec_key}_rhat", np.nan)
        print(
            f"   {label}: true = {true_val:.3f}, "
            f"recovered = {colored_med} ± {half_1sigma_disp:.3f} "
            f"(in 1σ: {in_1sigma}, in 2σ: {in_2sigma}, rhat={rhat:.3f})"
        )
