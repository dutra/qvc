from astropy.io import fits, ascii
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord, concatenate
from astropy import table
from astropy.table import QTable, Table, Column

import requests

import pyvo
from astropy.coordinates import SkyCoord
import astropy.units as u

from astropy.table import vstack

import pandas as pd
from tqdm import tqdm

def query_ssa_service(coords, radius=0.1/60, tap_url="http://wfaudata.roe.ac.uk/ssa-dsa/TAP"):
  """
  Query the SSA TAP service for a list of coordinates.

  Parameters:
  coords (SkyCoord): A SkyCoord object containing the coordinates to query.
  radius (float): The search radius in degrees. Default is 0.1 arcminutes.
  tap_url (str): The TAP service URL. Default is the SSA TAP service.

  Returns:
  list: A list of Astropy tables containing the query results for each coordinate.
  """
  service = pyvo.dal.TAPService(tap_url)
  results_list = []

  for coord in tqdm(coords, desc="Querying SSA service"):
    central_ra = coord.ra.deg
    central_dec = coord.dec.deg

    adql_query = f"""
    SELECT OBJID, SURVEYID, PLATEID, PARENTID, SOURCEID, RA, DEC, SMAG
    FROM Detection
    WHERE ra BETWEEN ({central_ra} - {radius} / 2.0) AND ({central_ra} + {radius} / 2.0)
      AND dec BETWEEN ({central_dec} - {radius} / 2.0) AND ({central_dec} + {radius} / 2.0)
    """
    results = service.search(adql_query)
    results_table = results.to_table()
    results_list.append(results_table)

  return results_list

print("Querying std catalog...")

# URL of the catalog
url = "http://faculty.washington.edu/ivezic/sdss/calib82/dataV2/stripe82calibStars_v4.2.dat"

# Download the file
response = requests.get(url)
if response.status_code == 200:
    with open("stripe82calibStars_v4.2.dat", "wb") as file:
        file.write(response.content)
    print("File downloaded successfully.")
else:
    print(f"Failed to download file. Status code: {response.status_code}")

# Read the catalog into a pandas DataFrame
# Assuming the file is space-delimited and has a header
catalog = pd.read_csv("stripe82calibStars_v4.2.dat", delim_whitespace=True, comment='#')

# Create SkyCoord objects for the catalog
catalog_coords = SkyCoord(ra=catalog.iloc[:, 1].to_numpy() * u.degree, dec=catalog.iloc[:, 2].to_numpy() * u.degree, frame='icrs')

print(len(catalog_coords))

table_std = query_ssa_service(catalog_coords[:10])

# Concatenate all tables in table_lc
concatenated_table = vstack(table_std)

# Save the concatenated table to a file
concatenated_table.write("concatenated_table_std.fits", format="fits", overwrite=True)