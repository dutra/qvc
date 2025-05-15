import h5py
import os
import numpy as np
import jax.numpy as jnp

suffix = os.environ.get('SUFFIX', None)


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

def get_unique_times(t, band, lag_blr):
    # Step 1: build full latent time grid (t, t - lag_blr)
    t_direct = t
    t_lagged = t - lag_blr

    t_latent = jnp.concatenate([t_direct, t_lagged])
    band_latent = jnp.concatenate([band, band])

    # Step 2: sort the combined time array (to make GP matrix construction more stable)
    sort_idx = jnp.argsort(t_latent)
    t_latent_sorted = t_latent[sort_idx]
    band_latent_sorted = band_latent[sort_idx]

    # Step 3: construct index maps back to sorted array
    def find_index(t_query, t_sorted):
        return jnp.argmin(jnp.abs(t_query[:, None] - t_sorted[None, :]), axis=1)

    inv_direct = find_index(t_direct, t_latent_sorted)
    inv_lagged = find_index(t_lagged, t_latent_sorted)

    return t_latent_sorted, band_latent_sorted, inv_direct, inv_lagged

def build_H(t, band, inv_direct, inv_lagged, A_c, A_b, M):
    # Observation operator H: shape (N, M)
    N = len(t)
    rows = jnp.arange(N)
    H = jnp.zeros((N, M))

    # Add direct term: A_c[band] * f(t)
    H = H.at[rows, inv_direct].add(A_c[band])

    # Add lagged term: A_b[band] * f(t - tau[band])
    H = H.at[rows, inv_lagged].add(A_b[band])
    return H

def modify_h5_file(save_file_path, s82_objs):
    with h5py.File(save_file_path, "a") as hdf:  # Open in append mode to modify
        for object_id in hdf.keys():  # Iterate through every object in the HDF5 file
            group = hdf[object_id]

            # Delete specified keys if they exist
            for key in ["mags", "times", "magerrs"]:
                if key in group:
                    del group[key]


def bands_redder_than_5000(z):
    """
    Returns a list of bands with rest-frame effective wavelength > 5000 Å.

    Args:
        z (float): Redshift.

    Returns:
        list: Bands redder than 5000 Å at rest frame.
    """
    threshold = 5000
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

def bands_with_host_contamination(z):
    """
    Returns a list of bands contaminated by the host galaxy, defined as rest-frame wavelength > 6000 Å.

    Returns:
        list of contaminated bands
    """
    host_contamination_threshold = 6000
    bands = {
        'u': {'lambda_eff': 3551, 'lambda_lo_90': 3152.4, 'lambda_hi_90': 3949.6},
        'g': {'lambda_eff': 4686, 'lambda_lo_90': 3715.4, 'lambda_hi_90': 5656.6},
        'r': {'lambda_eff': 6165, 'lambda_lo_90': 5207.0, 'lambda_hi_90': 7123.0},
        'i': {'lambda_eff': 7481, 'lambda_lo_90': 6407.6, 'lambda_hi_90': 8554.4},
        'z': {'lambda_eff': 8931, 'lambda_lo_90': 8266.7, 'lambda_hi_90': 9595.3},
        'y': {'lambda_eff': 9700, 'lambda_lo_90': 8900.0, 'lambda_hi_90': 10500.0}
    }

    contaminated_bands = []
    for band, props in bands.items():
        rest_lo = props['lambda_lo_90'] / (1 + z)
        rest_hi = props['lambda_hi_90'] / (1 + z)
        if rest_lo > host_contamination_threshold or rest_hi > host_contamination_threshold:
            contaminated_bands.append(band)

    return contaminated_bands

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

def save_samples_to_hdf5(samples, object_id):
    """
    Save all samples to an HDF5 file, one file per object_id.

    Args:
        samples (dict): Dictionary containing MCMC samples.
        object_id (str): The object ID for which the samples belong.
        output_dir (str): Directory where the HDF5 files will be saved.
    """
    output_dir=f"samples_{suffix}"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{object_id}.h5")

    with h5py.File(file_path, "w") as hdf:
        for key, value in samples.items():
            hdf.create_dataset(key, data=value)
    print(f"Saved samples for object_id {object_id} to {file_path}")

def append_hdf5_file(quasar_list, file_path):
    # Append to HDF5 file if it exists, otherwise create a new one
    print(f"Appending {len(quasar_list)} quasars to {file_path}", flush=True)
    with h5py.File(file_path, "a") as hdf:
        for quasar in quasar_list:
            object_id = quasar["object_id"]
            if object_id in hdf:
                continue

            group = hdf.create_group(object_id)
            for key, value in quasar.items():
                if isinstance(value, dict):
                    sub_group = group.create_group(key)
                    for sub_key, sub_value in value.items():
                        sub_group.create_dataset(sub_key, data=sub_value)
                else:
                    group.attrs[key] = value

# def log_broken_pl(lam, lam_s, d1, d2):
#     return -jnp.log10(jnp.power(lam/lam_s, d1) + jnp.power(lam/lam_s, d2))

def log_broken_pl(lam, lam_s, d1, d2, ds=4.0):
    x = lam / lam_s
    log_f = -jnp.log10(
        jnp.power(x, -d1) * jnp.power(1.0 + jnp.power(x, ds), -(d2 - d1) / ds)
    )
    return log_f

def process_samples(samples, data):
    clean_bands = data['clean_bands']
    object_id = data['object_id']
    save_samples_to_hdf5(samples, data['object_id'])

    # power laws
    eta_A1 = samples["eta_A1"]
    eta_A2 = samples["eta_A2"]
    eta_tau1 = samples["eta_tau1"]
    eta_tau2 = samples["eta_tau2"]
    eta_break = samples["eta_break"]
    
    lambda_ref = 2500 # Any reference wavelength
    lambda_s_RF = samples["lam_s"]
    
    samples_log_sigma_UV = samples['log_kernel_param'][:, 1] / np.log(10) + log_broken_pl(lambda_ref, lambda_s_RF, eta_A1, eta_A2, eta_break)
    samples_log_amp_delta = np.array([log_broken_pl(lambda_pivot[band], samples["lam_s"], eta_A1, eta_A2) if band in clean_bands else np.full_like(samples["lam_s"], -9999) for band in bands])                                                                               
    samples_log_sigma_band = samples['log_kernel_param'][:, 1:2] / np.log(10)  + samples_log_amp_delta.T

    samples_log_tau_UV_RF = samples['log_kernel_param'][:, 0] / np.log(10) - np.log10(1 + data['z']) + log_broken_pl(lambda_ref, lambda_s_RF, eta_tau1, eta_tau2, eta_break)
    samples_log_tau_delta = np.array([log_broken_pl(lambda_pivot[band], samples["lam_s"], eta_tau1, eta_tau2) if band in clean_bands else np.full_like(samples["lam_s"], -9999) for band in bands])
    samples_log_tau_band_RF = samples['log_kernel_param'][:, 0:1] / np.log(10) - np.log10(1 + data['z']) + samples_log_tau_delta.T

    def sym_percentile(x, p=[16, 50, 84], axis=0):
        lower, median, upper = np.percentile(x, p, axis=axis)
        return median, 0.5 * (upper - lower)

    # parameter estimates
    log_jitter, log_jitter_err = sym_percentile(np.log10(np.exp(2*samples['log_jitter'])))
    poly1, poly1_err = sym_percentile(samples['poly1'])
    mean, mean_err = sym_percentile(samples['mean'])
    log_lag_blr, log_lag_blr_err = sym_percentile(np.log10(np.exp(samples['log_lag_blr'])))
    lag, lag_err = sym_percentile(samples['lag'])
    eta_A1, eta_A1_err = sym_percentile(eta_A1)
    eta_A2, eta_A2_err = sym_percentile(eta_A2)
    eta_tau1, eta_tau1_err = sym_percentile(eta_tau1)
    eta_tau2, eta_tau2_err = sym_percentile(eta_tau2)
    eta_break, eta_break_err = sym_percentile(eta_break)

    log_tau, log_tau_err = sym_percentile(np.log10(np.exp(samples['log_kernel_param'][:, 0])))
    log_sigma, log_sigma_err = sym_percentile(np.log10(np.exp(samples['log_kernel_param'][:, 1])))
    log_w, log_w_err = sym_percentile(np.log10(np.exp(samples.get("dlog_w", np.array([0])) + samples['log_kernel_param'][:, 0])))

    log_tau_UV_RF, log_tau_UV_RF_err = sym_percentile(samples_log_tau_UV_RF)
    log_tau_band_RF, log_tau_band_RF_err = sym_percentile(samples_log_tau_band_RF)
    log_sigma_UV, log_sigma_UV_err = sym_percentile(samples_log_sigma_UV)
    log_sigma_band, log_sigma_band_err = sym_percentile(samples_log_sigma_band)

    # BLR
    log_tau_blr, log_tau_blr_err = sym_percentile(np.log10(np.exp(samples['log_tau_drw_blr'])))
    log_sigma_blr, log_sigma_blr_err = sym_percentile((samples['log_kernel_param'][:, 1:2] + samples['log_amp_delta_blr']) / np.log(10), axis=0)

    lambda_s_RF, lambda_s_RF_err = sym_percentile(lambda_s_RF)

    # Construct the result dictionary
    d = dict(object_id=data['object_id'],
            z=data['z'],
            # kernel params latent
            log_tau_UV_RF=log_tau_UV_RF,
            log_tau_UV_RF_err=log_tau_UV_RF_err,
            log_tau_band_RF=log_tau_band_RF,
            log_tau_band_RF_err=log_tau_band_RF_err,
            log_sigma_UV=log_sigma_UV,
            log_sigma_UV_err=log_sigma_UV_err,
            log_sigma_band=log_sigma_band,
            log_sigma_band_err=log_sigma_band_err,
            # broken power law params
            eta_A1=eta_A1,
            eta_A1_err=eta_A1_err,
            eta_A2=eta_A2,
            eta_A2_err=eta_A2_err,
            eta_tau1=eta_tau1,
            eta_tau1_err=eta_tau1_err,
            eta_tau2=eta_tau2,
            eta_tau2_err=eta_tau2_err,
            eta_break=eta_break,
            eta_break_err=eta_break_err,
            lam_s=lambda_s_RF,
            lam_s_err=lambda_s_RF_err,
            # kernel params
            log_sigma=log_sigma,
            log_sigma_err=log_sigma_err,
            log_tau=log_tau,
            log_tau_err=log_tau_err,
            log_w=log_w,
            log_w_err=log_w_err,
            # BLR
            log_tau_blr=log_tau_blr,
            log_tau_blr_err=log_tau_blr_err,
            log_sigma_blr=log_sigma_blr,
            log_sigma_blr_err=log_sigma_blr_err,
            # other
            log_jitter=log_jitter,
            poly1=poly1,
            poly1_err=poly1_err,
            mean=mean,
            mean_err=mean_err,
            clean_bands=clean_bands,
            log_lag_blr=log_lag_blr,
            log_lag_blr_err=log_lag_blr_err,
            lag=lag,
            lag_err=lag_err,
            )
    return d
