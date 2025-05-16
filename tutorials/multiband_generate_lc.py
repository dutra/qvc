import os
import numpy as np
import pandas as pd
import h5py
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from tqdm import tqdm

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


def concat_light_curves(N=None, skip=None, filter_object_ids=[], existing_object_ids=set(), save_file_path=None, progress_bar=False):
    print(f"DEBUG concat_light_curves args: {N=}, {skip=}, {len(filter_object_ids)=}, {len(existing_object_ids)=}, {save_file_path=}")

    if skip:
        filter_object_ids = filter_object_ids[skip:]
    if N:
        filter_object_ids = filter_object_ids[:N]

    filter_object_ids = set(filter_object_ids)

    if save_file_path and os.path.exists(save_file_path):
        print(f"DEBUG concat_light_curves Loading LC data from {save_file_path}")
        s82_objs = load_s82_from_hdf5(save_file_path)
        print(f"DEBUG Loaded {len(s82_objs)} objs from {save_file_path}")
        s82_objs = [obj for obj in s82_objs if obj['object_id'] not in (filter_object_ids | existing_object_ids)]
        return s82_objs
    else: 
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
    match_object_ids = set(sdss.objectId) - set(filter_object_ids) - set(existing_object_ids)
    matching_indices = cat[cat.objectId.isin(match_object_ids)].index

    cat = cat.loc[matching_indices]
    cat = cat[skip:] if skip else cat
    cat = cat[:N] if N else cat

    print(f"Found {len(cat)} matching objects in concat_light_curves", len(cat))

    # Loop through the data and extract the relevant information        
    for idx, row in tqdm(cat.iterrows(), total=len(cat), desc="Processing quasars", disable=(not progress_bar)):
        object_id = row['objectId']
        if object_id in [obj['object_id'] for obj in s82_objs]:
            continue

        # Filter light curves for the current object_id
        sdss_lc = sdss[sdss.objectId == row['objectId']].copy()
        ps1_lc = ps1[ps1.ps1objID == int(row['ps1objID'])].copy() if row['ps1objID'] is not None else pd.DataFrame()
        ztf_lc = ztf[ztf.ps1objID == int(row['ps1objID'])].copy() if row['ps1objID'] is not None else pd.DataFrame()

        #print("LC len: ", len(sdss_lc), len(ps1_lc), len(ztf_lc))

        # Combine light curves from different catalogs
        times = {}
        mags = {}
        magerrs = {}

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
            mags_means = [np.nanmean(mags[band]) for band in mags.keys()]
            #mags[band] = mags[band] - np.nanmean(mags[band])  # Center the magnitudes

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

            if len(times[band]) == 0 or len(mags[band]) == 0 or len(magerrs[band]) == 0:
                continue

            # Sort times, mags, magerrs by time
            sort_idx = np.argsort(times[band])
            times[band] = times[band][sort_idx]
            mags[band] = mags[band][sort_idx]
            magerrs[band] = magerrs[band][sort_idx]

        # Skip if no data is available for the object
        if all((len(times[band]) == 0 or len(mags[band]) == 0 or len(magerrs[band]) == 0) for band in bands):
            print(f"No data available for object {object_id}, skipping.", flush=True)
            continue


        s82_objs.append({
            'object_id': object_id,
            'times': times,
            'mags': mags,
            'mags_mean': mags_means,
            'magerrs': magerrs,
        })

        #save_lc_plot(times, mags, magerrs, object_id, bands=bands)

    hdul.close()

    print(f"Found {len(s82_objs)} objects in concat_light_curves after time cut", len(s82_objs))

    s82_objs = populate_sdss_fields(s82_objs, progress_bar=progress_bar)

    if save_file_path:
        with h5py.File(save_file_path, "w") as hdf:
            for obj in s82_objs:
                object_id = obj["object_id"]
                group = hdf.create_group(object_id)

                # Save all attributes
                for key, value in obj.items():
                    if isinstance(value, dict):
                        sub_group = group.create_group(key)
                        for sub_key, sub_value in value.items():
                            sub_group.create_dataset(sub_key, data=sub_value)
                    else:
                        group.attrs[key] = value
        print(f"Saved {len(s82_objs)} LCs to {save_file_path}")

    return s82_objs

def load_s82_from_hdf5(file_path="s82_objs.h5"):
    s82_objs = []

    with h5py.File(file_path, "r") as hdf:
        for object_id in hdf.keys():
            group = hdf[object_id]
            data = {"object_id": object_id}

            # Load attributes
            for attr_key in group.attrs.keys():
                data[attr_key] = group.attrs[attr_key]

            # Load datasets
            for key in group.keys():
                if isinstance(group[key], h5py.Group):
                    data[key] = {}
                    for sub_key in group[key].keys():
                        data[key][sub_key] = group[key][sub_key][...]
                else:
                    data[key] = group[key][...]

            s82_objs.append(data)

    return s82_objs

def populate_sdss_fields(s82_objs, progress_bar=False):
    #print(f"Populating SDSS fields: {len(s82_objs)}", flush=True)
    cat = pd.read_parquet(f"data/S82/Catalog.parquet").set_index('idx')
    hdul = fits.open('data/dr16q_prop_May01_2024.fits')
    fits_data = hdul[1].data  # Assuming the data is in the first extension    
    fits_data_2 = hdul[2].data  # Assuming the data is in the first extension    
    for d in tqdm(s82_objs, desc="Populating SDSS fields", disable=(not progress_bar)):
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
