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
            k: v[:, obj_index] if k not in ['eta_A1', 'eta_A2', 'eta_tau1', 'eta_tau2'] else v
            for k, v in samples_flat.items()
        }    

    # Print shapes for inspection
    print("Selected object samples: ", end='')
    for k, v in obj_samples.items():
        print(f"{k}={v.shape}", end='; ')
    
    return obj_samples

def flatten_flat_samples_per_band(samples_flat, clean_bands):
    """
    Flatten flat samples for each band in clean_bands.
    
    Parameters
    ----------
    samples_flat : dict
        Dictionary of flat samples with keys as parameter names.
    clean_bands : list
        List of clean bands to flatten.

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
            for i, band in enumerate(clean_bands):
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
    universal_keys = ['eta_A1', 'eta_A2', 'eta_tau1', 'eta_tau2']

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

def check_64bit(gpu=True):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    # JAX 64-bit mode check
    print("All JAX devices:", jax.devices())
    print("Default JAX device:", jax.devices()[0])
    print("JAX 64-bit enabled:", jax.config.read("jax_enable_x64"))

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("JAX 64-bit mode is not enabled. Please set jax_enable_x64 = True.")

    # Create a float64 array
    arr = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    assert arr.dtype == jnp.float64, "Array is not float64"

    @jax.jit
    def dot64(x, y):
        return jnp.dot(x, y)

    x = jnp.ones((1000,), dtype=jnp.float64)
    y = jnp.ones((1000,), dtype=jnp.float64)
    print("JAX 64-bit Dot product (should be 1000.0):", dot64(x, y))  # Should run without error and return 1000.0

    def assert_jax_gpu_64bit_ok():
        all_ok = True

        # 1. Check 64-bit mode
        x64_enabled = jax.config.read("jax_enable_x64")
        print(f"JAX 64-bit mode enabled: {x64_enabled}")
        if not x64_enabled:
            all_ok = False

        # 2. Create matrix and check dtype
        key = jax.random.PRNGKey(0)
        A = jax.random.normal(key, (3, 3), dtype=jnp.float64)
        print(f"Matrix dtype: {A.dtype}")
        if A.dtype != jnp.float64:
            all_ok = False

        # 3. Check GPU availability
        gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
        if gpu_devices:
            gpu = gpu_devices[0]
            print(f"Using GPU device: {gpu}")
        else:
            print("❌ No GPU found by JAX.")
            all_ok = False
            gpu = None

        # 4. Cholesky comparison
        try:
            K = A @ A.T + 1e-6 * jnp.eye(3)
            K_cpu = jax.device_put(K, jax.devices("cpu")[0])
            K_gpu = jax.device_put(K, gpu)

            L_cpu = jnp.linalg.cholesky(K_cpu)
            L_gpu = jnp.linalg.cholesky(K_gpu)
            L_gpu_cpu = jax.device_get(L_gpu)

            close = jnp.allclose(L_gpu_cpu, L_cpu, atol=1e-12)
            print(f"Cholesky GPU vs CPU match: {close}")
            if not close:
                all_ok = False
        except Exception as e:
            print(f"❌ Cholesky comparison failed: {e}")
            all_ok = False

        if all_ok:
            print("✅ All checks passed: 64-bit enabled, GPU available, and numerics consistent.")
        else:
            raise RuntimeError("❌ One or more JAX diagnostic checks failed.")

    if gpu:
        print(gpu)
        print('GPU')
        assert_jax_gpu_64bit_ok()

    return 1


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

def bands_bluer_than_lyman_alpha(z):
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
        if rest_lambda_eff < lyman_alpha:
            bluer_bands.append(band)

    return bluer_bands

def bands_with_any_contamination_annotated(z):
    """
    Returns ugrizy bands contaminated by Hα, Lyα, Lyman break, C IV, Mg II, or Hβ at redshift z,
    annotated by severity: 'severe', 'moderate', or not included.

    Uses λ_eff and Gaussian-derived 90% throughput ranges.

    Sources:
    - SDSS: Fukugita et al. 1996, Doi et al. 2010
    - LSST y-band: LSST Science Book
    - Severity classification: based on proximity of line to filter center

    Returns:
        dict with structure:
        {
            'Hα': {band: severity, ...},
            'Lyα': {band: severity, ...},
            'Lyman break': {band: severity, ...},
            'C IV': {band: severity, ...},
            'Mg II': {band: severity, ...},
            'Hβ': {band: severity, ...},
            'combined': {band: max severity across contaminants}
        }
    """
    lines = {
        #'Hα': 6563,
        #'Lyα': 1216,
        #'Lyman break': 912,  # Lyman break is not a line but a continuum drop
        'Lyman break': 1216,  # This is the Lyα line, not the break
        #'C IV': 1549,
        #'Mg II': 2798,
        #'Hβ': 4861
    }

    bands = {
        'u': {'lambda_eff': 3551, 'lambda_lo_90': 3152.4, 'lambda_hi_90': 3949.6},
        'g': {'lambda_eff': 4686, 'lambda_lo_90': 3715.4, 'lambda_hi_90': 5656.6},
        'r': {'lambda_eff': 6165, 'lambda_lo_90': 5207.0, 'lambda_hi_90': 7123.0},
        'i': {'lambda_eff': 7481, 'lambda_lo_90': 6407.6, 'lambda_hi_90': 8554.4},
        'z': {'lambda_eff': 8931, 'lambda_lo_90': 8266.7, 'lambda_hi_90': 9595.3},
        'y': {'lambda_eff': 9700, 'lambda_lo_90': 8900.0, 'lambda_hi_90': 10500.0}
    }

    def severity(lambda_obs, band_props):
        lo, hi = band_props['lambda_lo_90'], band_props['lambda_hi_90']
        center = band_props['lambda_eff']
        mid_25_lo = center - (hi - lo) * 0.25
        mid_25_hi = center + (hi - lo) * 0.25
        if mid_25_lo <= lambda_obs <= mid_25_hi:
            return 'severe'
        elif lo <= lambda_obs <= hi:
            return 'moderate'
        else:
            return None

    results = {line: {} for line in lines}

    for line, rest_wavelength in lines.items():
        lambda_obs = rest_wavelength * (1 + z)
        for band, props in bands.items():
            level = severity(lambda_obs, props)
            if level:
                results[line][band] = level

    # Combined: keep max severity for each band across all contaminants
    severity_order = {'moderate': 1, 'severe': 2}
    combined = {}
    for band in bands:
        levels = [results[line].get(band) for line in lines if band in results[line]]
        if levels:
            combined[band] = max(levels, key=lambda x: severity_order[x])
    results['combined'] = combined

    return results['combined']

def save_all_samples_to_hdf5(samples):
    """
    Save all samples to an HDF5 file
    Args:
        samples (dict): Dictionary containing MCMC samples.
    """
    output_dir=f"samples/{prefix}_{suffix}"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{prefix}_{suffix}_all.h5")

    logging.info(f"Saving all samples to {file_path}")

    with h5py.File(file_path, "w") as hdf:
        for key, value in samples.items():
            hdf.create_dataset(key, data=value)
    print(f"Saved all samples to {file_path}")

def save_obj_samples_to_hdf5(samples, object_id):
    """
    Save all samples to an HDF5 file, one file per object_id.

    Args:
        samples (dict): Dictionary containing MCMC samples.
        object_id (str): The object ID for which the samples belong.
        output_dir (str): Directory where the HDF5 files will be saved.
    """
    output_dir=f"samples/{prefix}_{suffix}"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{object_id}.h5")

    logging.info(f"Saving samples for object_id {object_id} to {file_path}")

    with h5py.File(file_path, "w") as hdf:
        for key, value in samples.items():
            hdf.create_dataset(key, data=value)
    print(f"Saved samples for object_id {object_id} to {file_path}")

def delete_file(file_path):
    """
    Delete a file if it exists.
    """
    if os.path.exists(file_path):
        os.remove(file_path)
        logging.info(f"Deleted existing file: {file_path}")
    else:
        logging.info(f"File does not exist; not deleting: {file_path}")

def save_quasar_list_hdf5(quasars, file_path, ignored_keys=None, size_threshold=1024):
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
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    string_dt = h5py.string_dtype(encoding="utf-8")

    total = len(quasars)
    logging.info(f"Saving {total} quasars to {file_path}")
    with h5py.File(file_path, "w") as hdf:
        for i, quasar in enumerate(quasars, start=1):
            object_id = str(quasar["object_id"])
            logging.info(f"Saving quasar {object_id} to {file_path}")
            
            group = hdf.create_group(object_id)
            for key, value in quasar.items():
                # Skip ignored keys
                if key in ignored_keys:
                    print(f"Warning: Skipping key '{key}' (ignored key)")
                    continue

                if isinstance(value, dict):
                    # Nested dict → sub-group with datasets
                    sub_group = group.create_group(key)
                    for sub_key, sub_value in value.items():
                        arr = np.asarray(sub_value)
                        if arr.size > size_threshold:
                            print(f"Warning: Skipping sub-key '{key}/{sub_key}' (too large: {arr.size})")
                            continue
                        if arr.dtype.kind in {'U', 'S', 'O'}:
                            arr = arr.astype(string_dt)
                        sub_group.create_dataset(sub_key, data=arr)
                else:
                    # Attributes for simple values
                    arr = np.asarray(value)
                    if arr.size > size_threshold:
                        print(f"Warning: Skipping key '{key}' (too large: {arr.size})")
                        continue
                    if arr.dtype.kind in {'U', 'S', 'O'}:
                        arr = arr.astype(string_dt)
                    group.attrs[key] = arr
            
            # Print progress
            print(f"{i}/{total}: Saved quasar {object_id}")

    logging.info("All quasars saved successfully.")


def log_broken_pl(lam, lam_s, d1, d2, ds=4.0):
    x = lam / lam_s
    log_f = -jnp.log10(
        jnp.power(x, -d1) * jnp.power(1.0 + jnp.power(x, ds), -(d2 - d1) / ds)
    )
    return log_f

def process_samples(flat_samples, data, percentiles=[16, 50, 84]):
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
            print(f"Warning: {k} has shape {v.shape}, expected flat samples")
        median, err = sym_percentile(v)
        result[k] = median
        result[f"{k}_err"] = err

    # generalized per-band computation
    # Power Law Params
    log_sigma_hat0 = np.asarray(flat_samples["log_sigma_hat0"])
    eta_A1 = np.asarray(flat_samples["eta_A1"])
    eta_A2 = np.asarray(flat_samples["eta_A2"])
    eta_tau1 = np.asarray(flat_samples["eta_tau1"])
    eta_tau2 = np.asarray(flat_samples["eta_tau2"])
    eta_break = 1
    lambda_ref = 2500
    lam_s = 2500

    log_sigma_band = []
    for band in data['clean_bands']:
        lam_eff = lambda_pivot[band]
        val = log_sigma_hat0 / np.log(10) + log_broken_pl(lam_eff, lam_s, eta_A1, eta_A2, eta_break)
        log_sigma_band.append(val)
    log_sigma_band = np.array(log_sigma_band).T

    for i, band in enumerate(data['clean_bands']):
        median, err = sym_percentile(log_sigma_band[:, i])
        result[f"log_sigma_band_{band}"] = median
        result[f"log_sigma_band_{band}_err"] = err

    # Other special params
    # log_sigma_hat_UV
    samples_log_sigma_hat_UV = flat_samples["log_sigma_hat0"] / np.log(10) + log_broken_pl(lambda_ref, lam_s, eta_A1, eta_A2, eta_break)
    result['log_sigma_hat_UV'], result['log_sigma_hat_UV_err'] = sym_percentile(samples_log_sigma_hat_UV)
    # log_tau_UV_RF
    samples_log_tau_UV_RF = flat_samples["log_tau_drw0"] / np.log(10) - np.log10(1 + data['z']) + log_broken_pl(lambda_ref, lam_s, eta_tau1, eta_tau2, eta_break)
    result['log_tau_UV_RF'], result['log_tau_UV_RF_err'] = sym_percentile(samples_log_tau_UV_RF)

    result["clean_bands"] = data['clean_bands']
    return result

def compute_psd_from_samples(samples, clean_bands, num_points=1000, time_range=(1.0, 365*20)):
    """
    Compute the Power Spectral Density (PSD) for each band using the median of the posterior samples.

    Args:
        samples (dict): MCMC samples containing kernel and power-law parameters.
        clean_bands (list): List of clean bands used in the model.
        num_points (int): Number of frequency points.
        time_range (tuple): Range of time lags (min_time, max_time) in days.

    Returns:
        dict: {band: {"freqs": ..., "psd": ...}} for each band in clean_bands.
    """

    # Reference wavelength (rest-frame)
    lambda_ref = 2500.0

    # Median parameters
    log_sigma_hat0 = np.median(samples["log_sigma_hat0"])
    log_tau_drw0 = np.median(samples["log_tau_drw0"])
    eta_A1 = np.median(samples["eta_A1"])
    eta_A2 = np.median(samples["eta_A2"])
    eta_break = 1.0 #np.median(samples["eta_break"])
    lam_s = 2500 #np.median(samples["lam_s"])
    eta_tau1 = np.median(samples["eta_tau1"])
    eta_tau2 = np.median(samples["eta_tau2"])

    # Helper: broken power law scaling
    def log_broken_pl(lam, lam_s, d1, d2, ds):
        x = lam / lam_s
        log_f = -np.log10(
            np.power(x, -d1) * np.power(1.0 + np.power(x, ds), -(d2 - d1) / ds)
        )
        return log_f

    # Frequency grid (cycles/day)
    min_time, max_time = time_range
    t_span = max_time - min_time
    freqs = np.logspace(-4, 0, num_points)  # 1/10,000 to 1 cycles/day

    psd_results = {}
    for band in clean_bands:
        # Get pivot wavelength for this band (rest-frame)
        lam_eff = lambda_pivot[band]

        # Compute log_sigma and log_tau for this band (rest-frame)
        log_sigma = log_sigma_hat0 / np.log(10) + log_broken_pl(lam_eff, lam_s, eta_A1, eta_A2, eta_break)
        log_tau = log_tau_drw0 / np.log(10) + log_broken_pl(lam_eff, lam_s, eta_tau1, eta_tau2, eta_break)

        sigma = 10 ** log_sigma
        tau = 10 ** log_tau

        # DRW PSD: S(f) = 2 sigma^2 tau^2 / [1 + (2 pi tau f)^2]
        S_f = 2 * sigma**2 * tau**2 / (1 + (2 * np.pi * tau * freqs) ** 2)

        psd_results[band] = {"freqs": freqs, "psd": S_f}

    return psd_results
