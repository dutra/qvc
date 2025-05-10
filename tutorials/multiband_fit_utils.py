import h5py
import os

suffix = os.environ.get('SUFFIX', None)

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