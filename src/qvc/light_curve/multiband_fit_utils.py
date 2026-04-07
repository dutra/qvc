import h5py
import os
import numpy as np
import jax.numpy as jnp
import secrets
import subprocess

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
BALMER_EDGE_REST_WAVELENGTH = 3646.0
BALMER_EDGE_ATTENUATION_WIDTH = 250.0
BALMER_EDGE_SUMMARY_WEIGHT_MIN = 0.05


import os
import jax.numpy as jnp
from datetime import datetime

import numpy as np


_GIT_COMMIT_SENTINEL = object()
_GIT_COMMIT_CACHE = _GIT_COMMIT_SENTINEL
_RUN_METADATA_KEYS = {"git_commit", "run_datetime"}


def _balmer_continuum_weight(
    lam_rf,
    transition=BALMER_EDGE_REST_WAVELENGTH,
    width=BALMER_EDGE_ATTENUATION_WIDTH,
):
    """Match the smooth Balmer-edge attenuation used during fitting."""

    lam_rf = np.asarray(lam_rf, dtype=float)
    return 1.0 / (1.0 + np.exp((lam_rf - transition) / width))


def _get_current_git_commit():
    """
    Resolve current git commit hash once per process.
    Returns an empty string when unavailable.
    """
    global _GIT_COMMIT_CACHE
    if _GIT_COMMIT_CACHE is not _GIT_COMMIT_SENTINEL:
        return _GIT_COMMIT_CACHE

    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(__file__),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _GIT_COMMIT_CACHE = proc.stdout.strip()
    except Exception as exc:
        logging.warning("Could not resolve git commit for HDF5 metadata: %s", exc)
        _GIT_COMMIT_CACHE = ""
    return _GIT_COMMIT_CACHE


def _current_local_datetime_iso8601():
    """
    Return local datetime with timezone offset in ISO8601 format.
    Example: 2026-03-19T16:14:33-04:00
    """
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_hdf5_run_metadata(hdf):
    """
    Write run metadata as root-level scalar datasets.
    """
    string_dt = h5py.string_dtype(encoding="utf-8")
    metadata = {
        "git_commit": _get_current_git_commit(),
        "run_datetime": _current_local_datetime_iso8601(),
    }

    for key, value in metadata.items():
        if key in hdf:
            del hdf[key]
        hdf.create_dataset(key, data=np.asarray(value, dtype=string_dt), dtype=string_dt)

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
        k: v[:, :, obj_index] if (k not in universal_params and v.ndim > 2 and not k.startswith('_')) else v
        for k, v in samples_per_chain.items()
    }

    # Print shapes for inspection
    logging.debug(
        "Selected object samples per chain: " 
        + ", ".join(f"{k}={v.shape}" for k, v in obj_samples.items())
    )

    return obj_samples


def flatten_per_chain_samples_per_band(samples_per_chain, bands):
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
    internal_skip_keys = {
        "log_kernel_param",
    }
    flattened_samples = {}
    for k, v in samples_per_chain.items():
        if k.startswith('_') or k in internal_skip_keys:
            continue  # Skip metadata keys
        logging.debug(f"flatten_per_chain: {k} shape={getattr(v, 'shape', None)}")
        if v.ndim == 0:
            continue
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
    obj_samples = {}
    for k, v in samples_flat.items():
        if k.startswith('_'):
            continue  # Skip metadata keys
        try:
            if k in universal_params:
                obj_samples[k] = v
            else:
                obj_samples[k] = v[:, obj_index]
        except Exception as e:
            logging.error(f"Error selecting samples for {k} with shape {getattr(v, 'shape', None)}: {e}")
            raise

    # Print shapes for inspection
    logging.debug("Selected object samples: " + ", ".join(f"{k}={v.shape}" for k, v in obj_samples.items()))

    return obj_samples

def flatten_flat_samples_per_band(samples_flat, bands):
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
    internal_skip_keys = {
        "log_kernel_param",
    }
    flattened_samples = {}
    for k, v in samples_flat.items():
        if k.startswith('_') or k in internal_skip_keys:
            continue  # Skip metadata keys
        if v.ndim == 0:
            continue
        if v.ndim == 1:
            flattened_samples[k] = v
        elif v.ndim == 2:
            # Flatten over bands
            if v.shape[-1] != len(bands):
                raise ValueError(f"Unexpected band dimension for {k}: {v.shape} vs bands={len(bands)}")
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
    universal_keys = ['eta_sigma', 'eta_tau']

    # Print shapes for inspection
    for k, v in samples_grouped.items():
        print(f"{k}={v.shape}", end='; ')

    obj_samples_clean = dict()
    for k, v in samples_grouped.items():
        if k.startswith('_'):
            continue  # Skip metadata keys
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


def sdss_bands_affected_by_lya(z, buffer=0.0):
    """
    SDSS ugriz bands whose *rest-frame blue edge* falls below 1216+buffer Å,
    i.e., likely contaminated by Lyα line/forest.

    Args:
        z (float): source redshift
        buffer (float): safety margin in Å (typ. 50–200)

    Returns:
        list[str]: bands to drop (contaminated)
    """
    # SDSS (full transmission) edges in observed Å (λ_min, λ_max)
    edges_obs = {
        "u": (3055.11, 4030.64),
        "g": (3797.64, 5553.04),
        "r": (5418.23, 6994.42),
        "i": (6692.41, 8400.32),
        "z": (7964.70, 10873.33),
    }

    cutoff = 1216.0 + buffer
    affected = []
    for b, (lo_obs, _hi_obs) in edges_obs.items():
        lo_rest = lo_obs / (1.0 + z)
        if lo_rest < cutoff:
            affected.append(b)
    return affected


def sdss_bands_safe_from_lya(z, buffer=100.0):
    """Complement: bands whose rest-frame blue edge is ≥ 1216+buffer Å."""
    cutoff = 1216.0 + buffer
    edges_obs = {
        "u": (3055.11, 4030.64),
        "g": (3797.64, 5553.04),
        "r": (5418.23, 6994.42),
        "i": (6692.41, 8400.32),
        "z": (7964.70, 10873.33),
    }
    return [b for b, (lo_obs, _) in edges_obs.items() if (lo_obs / (1.0 + z)) >= cutoff]


def load_all_samples_from_hdf5(file_path=None):
    """
    Load all samples from an HDF5 file.
    
    Args:
        prefix (str): Prefix used for the samples directory and filename.
        suffix (str): Suffix used in the filename.
    
    Returns:
        dict: Dictionary containing all loaded samples.
    """
    if file_path is None:
        output_dir=f"results/samples/{prefix}/"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"all_{suffix}.h5")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")

    logging.info(f"Loading all samples from {file_path}")

    samples = {}
    with h5py.File(file_path, "r") as hdf:
        for key in hdf.keys():
            if key in _RUN_METADATA_KEYS:
                continue
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
        _write_hdf5_run_metadata(hdf)
        for key, value in samples.items():
            hdf.create_dataset(key, data=value)
    logging.info(f"Saved all samples to {file_path}")
    print(f"Saved all samples to {file_path}")

def load_obj_samples_from_hdf5(object_id=None, file_path=None):
    """
    """
    if file_path is None:
        output_dir=f"results/samples/{prefix}/"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{object_id}_{suffix}.h5")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")

    logging.info(f"Loading all samples from {file_path}")

    samples = {}
    with h5py.File(file_path, "r") as hdf:
        for key in hdf.keys():
            if key in _RUN_METADATA_KEYS:
                continue
            samples[key] = np.array(hdf[key])

    logging.info(f"Loaded {len(samples)} datasets from {file_path}")
    return samples

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
        _write_hdf5_run_metadata(hdf)
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
    Save a list of quasar dictionaries to a *flat columnar* HDF5 file.

    - The file is always truncated to start fresh.
    - Every field is written as a top-level dataset of length N.
    - Band-related vectors are expanded to *_u, *_g, *_r, *_i, *_z.
    - Other vectors are expanded to *_0, *_1, ...
    - Nested dicts are flattened recursively using underscore-joined keys.
    - Keys in ignored_keys or arrays larger than size_threshold are skipped.
    """
    ignored_keys = set(ignored_keys or [])
    fixed_bands = ("u", "g", "r", "i", "z")
    string_dt = h5py.string_dtype(encoding="utf-8")

    def _to_scalar(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="replace")
        return x

    def _is_string_like(v):
        return isinstance(v, (str, bytes, np.str_, np.bytes_))

    def _is_bool_like(v):
        return isinstance(v, (bool, np.bool_))

    def _is_numeric_like(v):
        return isinstance(v, (int, float, np.integer, np.floating, np.bool_))

    def _string_fill_for(values):
        for v in values:
            if _is_string_like(v):
                return ""
        return np.nan

    def _should_expand_as_bands(base_key, arr_1d, obj_bands):
        if obj_bands is None or len(arr_1d) != len(obj_bands):
            return False
        if any(band not in fixed_bands for band in obj_bands):
            return False
        if base_key.endswith("bands") or "band" in base_key:
            return True
        if base_key in {"mags_mean", "mags_means", "mean", "cadence", "cadence_err", "number_points"}:
            return True
        if base_key.startswith(("mags_mean_", "mags_means_", "mean_", "cadence_", "number_points_")):
            return True
        return False

    def _flatten_value(row, base_key, value, obj_bands):
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                child_key = f"{base_key}_{sub_key}" if base_key else str(sub_key)
                _flatten_value(row, child_key, sub_value, obj_bands)
            return

        arr = np.asarray(value) if value is not None else np.asarray(np.nan)
        if arr.size > size_threshold:
            logging.warning(
                "Warning: Skipping key '%s' (too large: %s)",
                base_key,
                arr.size,
            )
            return

        if arr.ndim == 0:
            row[base_key] = _to_scalar(arr.reshape(-1)[0])
            return

        flat = arr.reshape(-1)
        if flat.size > 5:
            logging.warning(
                "Warning: key '%s' has vector length %d (>5); not saving.",
                base_key,
                flat.size,
            )
            return
        if flat.size == 5:
            for i, band in enumerate(fixed_bands):
                row[f"{base_key}_{band}"] = _to_scalar(flat[i])
            return
        if _should_expand_as_bands(base_key, flat, obj_bands):
            band_to_value = {band: _to_scalar(flat[i]) for i, band in enumerate(obj_bands)}
            fill_value = _string_fill_for(band_to_value.values())
            for band in fixed_bands:
                row[f"{base_key}_{band}"] = band_to_value.get(band, fill_value)
            return

        for i, v in enumerate(flat):
            row[f"{base_key}_{i}"] = _to_scalar(v)

    def _build_column(values):
        has_string = any(_is_string_like(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v)))
        if has_string:
            out = []
            for v in values:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    out.append("")
                elif isinstance(v, bytes):
                    out.append(v.decode("utf-8", errors="replace"))
                else:
                    out.append(str(v))
            return np.asarray(out, dtype=object).astype(string_dt)

        non_missing = [v for v in values if v is not None]
        if non_missing and all(_is_bool_like(v) for v in non_missing):
            out = []
            for v in values:
                if v is None:
                    out.append(False)
                else:
                    out.append(bool(v))
            return np.asarray(out, dtype=bool)

        has_numeric = any(_is_numeric_like(v) for v in values if v is not None)
        if has_numeric:
            out = []
            for v in values:
                if v is None:
                    out.append(np.nan)
                else:
                    out.append(float(v))
            return np.asarray(out, dtype=float)

        out = ["" if v is None else str(v) for v in values]
        return np.asarray(out, dtype=object).astype(string_dt)

    def _output_basename(quasars_list):
        if len(quasars_list) == 1:
            return f"{str(quasars_list[0]['object_id'])}.h5"
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f")
        token = secrets.token_hex(4)
        return f"{timestamp}_{token}.h5"

    rows = []
    total = len(quasars)
    output_dir = f"results/data/{prefix}"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, _output_basename(quasars))
    logging.info(f"Saving {total} quasars to {file_path}")

    for i, quasar in enumerate(quasars):
        object_id = str(quasar["object_id"])
        obj_bands = quasar.get("bands")
        if isinstance(obj_bands, np.ndarray):
            obj_bands = [str(x) for x in obj_bands.tolist()]
        elif isinstance(obj_bands, (list, tuple)):
            obj_bands = [str(x) for x in obj_bands]
        else:
            obj_bands = None

        row = {"object_id": object_id}
        for key, value in quasar.items():
            if key in ignored_keys or key == "object_id":
                continue
            _flatten_value(row, str(key), value, obj_bands)
        rows.append(row)
        logging.info(f"{i+1}/{total}: Flattened quasar {object_id}")

    all_columns = {"object_id"}
    for row in rows:
        all_columns.update(row.keys())

    with h5py.File(file_path, "w") as hdf:
        _write_hdf5_run_metadata(hdf)
        for col in sorted(all_columns):
            values = [row.get(col, None) for row in rows]
            arr = _build_column(values)
            hdf.create_dataset(col, data=arr)

    logging.info("All quasars saved successfully.")

def log_broken_pl(lam, lam_s, d1, d2, ds):
    """
    Log10 of a smooth broken power-law, normalized to 0 at lam_s.
    ds is a smoothness (larger ds => smoother transition).
    This version is numerically stable and AD-friendly.
    """
    # Preconditions: lam>0, lam_s>0, ds>0 (enforce in your priors/transforms)
    ln10 = jnp.log(10.0)

    # Work in log-space: log10(x) with x=lam/lam_s
    log10x = (jnp.log(lam) - jnp.log(lam_s)) / ln10
    # a = log10(x)/ds; we need log10(1 + 10^a) stably
    a = log10x / ds
    # log10(1 + 10^a) = log(1 + exp(a*ln 10)) / ln 10, computed stably:
    log10_1p10a = jnp.logaddexp(0.0, a * ln10) / ln10

    delta = d2 - d1
    # Your original: d1*log10(x) + (delta/smooth_exp)*log10_1px - (delta/smooth_exp)*log10(2)
    # with smooth_exp=1/ds  => (delta*ds)*(...)
    log_f = d1 * log10x + (delta * ds) * (log10_1p10a - jnp.log10(2.0))
    return log_f

def log_single_pl(lam, lam_s, d, *_, **__):
    """
    Log10 of a simple power-law, normalized to 0 at lam_s.
    Slope is d everywhere.

    Notes:
    - Works for array lam/lam_s; requires lam>0, lam_s>0 (enforce upstream).
    - Uses log-space to avoid overflow in lam/lam_s.
    - Swallows extra args (*_, **__) so you can call it with the same
      signature as your broken-PL without recompiles.
    """
    ln10 = jnp.log(jnp.array(10.0, dtype=jnp.result_type(lam, lam_s, d)))
    # log10(lam/lam_s) = (log(lam) - log(lam_s)) / ln(10)
    return d * (jnp.log(lam) - jnp.log(lam_s)) / ln10

def ordered_dho_taus(tau_fast, tau_slow, *, eps=1e-12):
    """Return numerically safe fast/slow DHO timescales with fast <= slow."""

    tau_fast = np.asarray(tau_fast, dtype=float)
    tau_slow = np.asarray(tau_slow, dtype=float)
    fast = np.maximum(np.minimum(tau_fast, tau_slow), eps)
    slow = np.maximum(np.maximum(tau_fast, tau_slow), fast * (1.0 + 1e-6))
    return fast, slow


def dho_stationary_variance_factor(tau_fast, tau_slow, *, eps=1e-12):
    """Variance factor of the observed overdamped-SHO process at zero lag."""

    fast, slow = ordered_dho_taus(tau_fast, tau_slow, eps=eps)
    denom = np.maximum(slow - fast, eps)
    c_fast = -fast / denom
    c_slow = slow / denom
    return np.square(c_fast) + np.square(c_slow)


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

def process_samples(flat_samples, data, bands, percentiles=[16, 50, 84]):
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

    lam_rf_kept = np.asarray([lambda_pivot[band] / (1.0 + data['z']) for band in bands], dtype=float)
    lambda_center_rf = float(np.exp(np.mean(np.log(lam_rf_kept)))) if len(lam_rf_kept) > 0 else np.nan

    result = dict(
        object_id=data['object_id'],
        z=data['z'],
        lambda_center_rf=lambda_center_rf,
        n_bands_kept=len(bands),
        bands_kept=",".join(bands),
    )

    internal_skip_keys = {
        "log_sigma_center0",
        "log_tau_slow_center0",
        "log_tau_fast_center0",
    }
    internal_skip_prefixes = (
        "log_amp_delta_blr_raw",
        "log_amp_delta_blr2_raw",
        "log_lag_blr_raw",
        "log_lag_blr2_raw",
    )

    # per flat param computation
    for k, v in flat_samples.items():
        if k in internal_skip_keys or any(k.startswith(prefix) for prefix in internal_skip_prefixes):
            continue
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
    log_sigma_uv = np.asarray(flat_samples["log_sigma_uv"]) if "log_sigma_uv" in flat_samples else None
    log_tau_uv = np.asarray(flat_samples["log_tau_uv"]) if "log_tau_uv" in flat_samples else None
    log_tau_fast_uv = np.asarray(flat_samples["log_tau_fast_uv"]) if "log_tau_fast_uv" in flat_samples else None
    eta_sigma = np.asarray(flat_samples["eta_sigma"])
    eta_tau = np.asarray(flat_samples["eta_tau"])
    lambda_ref = 2500
    lam_ref_arr = np.full_like(eta_sigma, lambda_ref, dtype=float)

    if "eta_sigma" not in result:
        result["eta_sigma"], result["eta_sigma_err"] = sym_percentile(eta_sigma)
    if "eta_tau" not in result:
        result["eta_tau"], result["eta_tau_err"] = sym_percentile(eta_tau)

    if "log_sigma_uv" not in result:
        result["log_sigma_uv"], result["log_sigma_uv_err"] = sym_percentile(log_sigma_uv / np.log(10))
    if "log_sigma_fast_uv" not in result:
        result["log_sigma_fast_uv"], result["log_sigma_fast_uv_err"] = sym_percentile(log_sigma_uv / np.log(10))
    if "log_tau_uv" not in result:
        result["log_tau_uv"], result["log_tau_uv_err"] = sym_percentile(log_tau_uv / np.log(10))
    if "log_tau_fast_uv" not in result:
        result["log_tau_fast_uv"], result["log_tau_fast_uv_err"] = sym_percentile(log_tau_fast_uv / np.log(10))
    samples_log_tau_uv_rf = log_tau_uv / np.log(10) - np.log10(1 + data['z']) + log_single_pl(lambda_ref, lam_ref_arr, eta_tau)
    result["log_tau_uv_rf"], result["log_tau_uv_rf_err"] = sym_percentile(samples_log_tau_uv_rf)
    samples_log_tau_fast_uv_rf = log_tau_fast_uv / np.log(10) - np.log10(1 + data['z']) + log_single_pl(lambda_ref, lam_ref_arr, eta_tau)
    result["log_tau_fast_uv_rf"], result["log_tau_fast_uv_rf_err"] = sym_percentile(samples_log_tau_fast_uv_rf)

    log_sigma_band = []
    for band in bands:
        lam_eff = lambda_pivot[band] / (1 + data['z'])
        val = log_sigma_uv / np.log(10) + log_single_pl(lam_eff, lam_ref_arr, eta_sigma)
        log_sigma_band.append(val)
    log_sigma_band = np.array(log_sigma_band).T

    log_tau_band = []
    for band in bands:
        lam_eff = lambda_pivot[band] / (1 + data['z'])
        val = log_tau_uv / np.log(10) - np.log10(1 + data['z']) + log_single_pl(lam_eff, lam_ref_arr, eta_tau)
        log_tau_band.append(val)
    log_tau_band = np.array(log_tau_band).T

    log_tau_fast_band = []
    for band in bands:
        lam_eff = lambda_pivot[band] / (1 + data['z'])
        val = log_tau_fast_uv / np.log(10) - np.log10(1 + data['z']) + log_single_pl(lam_eff, lam_ref_arr, eta_tau)
        log_tau_fast_band.append(val)
    log_tau_fast_band = np.array(log_tau_fast_band).T

    sigma_rms_band = []
    for i, _band in enumerate(bands):
        amp = np.power(10.0, log_sigma_band[:, i])
        tau_fast = np.power(10.0, log_tau_fast_band[:, i])
        tau_slow = np.power(10.0, log_tau_band[:, i])
        variance_factor = dho_stationary_variance_factor(tau_fast, tau_slow)
        sigma_rms_band.append(np.log10(amp * np.sqrt(np.maximum(variance_factor, 1e-300))))
    sigma_rms_band = np.array(sigma_rms_band).T

    for i, band in enumerate(bands):
        median, err = sym_percentile(log_sigma_band[:, i])
        result[f"log_sigma_band_{band}"] = median
        result[f"log_sigma_band_{band}_err"] = err
        median, err = sym_percentile(sigma_rms_band[:, i])
        result[f"log_sigma_rms_band_{band}"] = median
        result[f"log_sigma_rms_band_{band}_err"] = err
        median, err = sym_percentile(log_tau_band[:, i])
        result[f"log_tau_band_{band}_RF"] = median
        result[f"log_tau_band_{band}_RF_err"] = err
        median, err = sym_percentile(log_tau_fast_band[:, i])
        result[f"log_tau_fast_band_{band}_RF"] = median
        result[f"log_tau_fast_band_{band}_RF_err"] = err
        for lag_suffix in ("", "2"):
            log_lag_blr_key = f"log_lag_blr{lag_suffix}_{band}"
            if log_lag_blr_key in flat_samples:
                samples_log_lag_blr_rf = (
                    np.asarray(flat_samples[log_lag_blr_key]) / np.log(10) - np.log10(1 + data['z'])
                )
                median, err = sym_percentile(samples_log_lag_blr_rf)
                result[f"log_lag_blr{lag_suffix}_{band}_RF"] = median
                result[f"log_lag_blr{lag_suffix}_{band}_RF_err"] = err
        lag_bc_key = f"lag_bc_{band}"
        lam_eff = lambda_pivot[band] / (1.0 + data['z'])
        bc_weight = float(_balmer_continuum_weight(lam_eff))
        result[f"bc_weight_{band}"] = bc_weight
        if lag_bc_key in flat_samples and bc_weight > BALMER_EDGE_SUMMARY_WEIGHT_MIN:
            samples_lag_bc = np.asarray(flat_samples[lag_bc_key], dtype=float)
            mask_bc = np.isfinite(samples_lag_bc) & (samples_lag_bc > 0.0)
            if np.any(mask_bc):
                samples_log_lag_bc_rf = np.log10(samples_lag_bc[mask_bc]) - np.log10(1 + data['z'])
                median, err = sym_percentile(samples_log_lag_bc_rf)
                result[f"log_lag_bc_{band}_RF"] = median
                result[f"log_lag_bc_{band}_RF_err"] = err

    samples_log_sigma_uv = log_sigma_uv / np.log(10)
    cov_matrix = np.cov(samples_log_sigma_uv, samples_log_tau_uv_rf)
    cov_log_sigma_tau = cov_matrix[0, 1]
    
    vx, vy, cov_log_sigma_tau_reg = regularize_cov_from_percentiles(
        np.percentile(samples_log_sigma_uv, 16),
        np.percentile(samples_log_sigma_uv, 84),
        np.percentile(samples_log_tau_uv_rf, 16),
        np.percentile(samples_log_tau_uv_rf, 84),
        cov_log_sigma_tau
    )
    result['cov_log_sigma_uv_log_tau_uv_rf'] = cov_log_sigma_tau_reg
    print("Regularized covariance: ", cov_log_sigma_tau_reg)

    C_old = psd_cov_from_samples(samples_log_sigma_uv, samples_log_tau_uv_rf, shrink_rho=0.05)
    _, _, sxy_old = C_old[0,0], C_old[1,1], C_old[0,1]
    result['log_sigma_uv_log_tau_uv_rf_cov_psd_old'] = sxy_old
    print("Old PSD covariance: ", sxy_old)

    result['log_sigma_uv_log_tau_uv_rf_cov_psd'] = cov_log_sigma_tau_reg
    result['log_sigma_uv_std_psd'] = np.sqrt(vx)
    result['log_tau_uv_rf_std_psd'] = np.sqrt(vy)
    print("Hubble covariance term: ", cov_log_sigma_tau_reg)
    print("Hubble std terms: ", np.sqrt(vx), np.sqrt(vy))

    return result


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

    ignore_keys = ('bwb_beta', 'f_host', 'log_tau_fake', 'log_sigma_fake',
                   'gate_log_temp', 'lmc_sep_raw', 'lmc_sep_left_raw', 'lmc_sep_right_raw', 'lmc_span_raw',
                    'lmc_mu_raw', 'lmc_delta_raw', 'lmc_sep', 'lmc_sep_left', 'lmc_sep_right', 'lmc_span',)

    for k, v in flattened_per_chain.items():
        if k in ignore_keys:
            continue
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
        Posterior draws for recovered params, e.g. samples['eta_sigma'], samples['log_tau_uv'], ...
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
    print(f"[INJECT FAKE] Object {obj.get('object_id', '<?>')}:")
    # One line per requested comparison
    for r in compare_pairs:
        if len(r) == 2:
            inj_key, rec_key = r
            transform = lambda x: x  # identity
            label = rec_key
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
