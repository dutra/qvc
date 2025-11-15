from astropy.coordinates import SkyCoord
import astropy.units as u
import pyvo
import pandas as pd
from tqdm import tqdm
import requests
import os
import h5py
import numpy as np
import time

# Function to perform the query and save results incrementally to HDF5
def query_ssa_service(coords, radius=0.1/60, tap_url="http://wfaudata.roe.ac.uk/ssa-dsa/TAP"):
    service = pyvo.dal.TAPService(tap_url)
    output_file = "incremental_results.hdf5"
    checkpoint_file = "checkpoint.txt"
    failed_coords_file = "failed_coords.txt"

    # Determine where to resume from
    resume_index = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            line = f.readline()
            if line.strip().isdigit():
                resume_index = int(line.strip())

    # Open or create HDF5 file
    with h5py.File(output_file, "a") as h5f:
        for i in tqdm(range(resume_index, len(coords)), desc="Querying SSA service"):
            coord = coords[i]
            key = f"{coord.ra.deg:.6f}_{coord.dec.deg:.6f}"
            if key in h5f:
                continue

            adql_query = f"""
            SELECT objID, surveyID, plateID, parentID, sourceID, ra, dec, sMag FROM Detection
            WHERE ra BETWEEN ({coord.ra.deg} - {radius}/2.0) AND ({coord.ra.deg} + {radius}/2.0)
              AND dec BETWEEN ({coord.dec.deg} - {radius}/2.0) AND ({coord.dec.deg} + {radius}/2.0)
            """

            attempt = 0
            max_attempts = 5
            backoff = 10

            success = False
            while attempt < max_attempts:
                try:
                    results = service.search(adql_query)
                    results_table = results.to_table()
                    if len(results_table) > 0:
                        data_array = results_table.as_array()
                        h5f.create_dataset(key, data=data_array)
                    success = True
                    break
                except Exception as e:
                    attempt += 1
                    wait_time = backoff * (2 ** (attempt - 1))
                    print(f"Attempt {attempt}/{max_attempts} failed at index {i} with error: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

            if not success:
                print(f"Failed permanently at index {i}. Skipping this coordinate.")
                with open(failed_coords_file, "a") as f:
                    f.write(f"{coord.ra.deg:.6f},{coord.dec.deg:.6f}\n")

            # Save progress checkpoint
            with open(checkpoint_file, 'w') as f:
                f.write(str(i + 1))

# Download catalog if not already present
catalog_file = "stripe82calibStars_v4.2.dat"
if not os.path.exists(catalog_file):
    url = "http://faculty.washington.edu/ivezic/sdss/calib82/dataV2/stripe82calibStars_v4.2.dat"
    response = requests.get(url)
    if response.status_code == 200:
        with open(catalog_file, "wb") as file:
            file.write(response.content)
        print("Catalog downloaded successfully.")
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

# Load the catalog
catalog = pd.read_csv(catalog_file, delim_whitespace=True, comment='#')
catalog_coords = SkyCoord(ra=catalog.iloc[:, 1].to_numpy() * u.degree, dec=catalog.iloc[:, 2].to_numpy() * u.degree, frame='icrs')

# Run query with incremental saving to HDF5
query_ssa_service(catalog_coords)
