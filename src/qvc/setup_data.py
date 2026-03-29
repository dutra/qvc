# pip install gdown tqdm colorama

import gdown
import os
import zipfile
import shutil
from tqdm import tqdm
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

def download_and_extract(url, folder, filename=None):
    """
    Downloads files from Google Drive and extracts ZIPs.
    Skips download if filename is provided and exists in folder.
    """
    # 1. Setup Directory
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"{Fore.CYAN}DIR  {Fore.RESET} Created: {Style.DIM}{folder}")

    # 2. Check if file already exists
    if filename:
        target_path = os.path.join(folder, filename)
        if os.path.exists(target_path):
            print(f"{Fore.YELLOW}SKIP {Fore.RESET} {filename} already exists in {folder}. Skipping.")
            return

    try:
        print(f"{Fore.BLUE}GET  {Fore.RESET} Fetching from Drive...")
        
        # Download to a temporary path first
        path = gdown.download(url, quiet=False, fuzzy=True)
        
        if not path:
            print(f"{Fore.RED}FAIL {Fore.RESET} Download returned no path for {url}")
            return

        # 3. Handle ZIP Extraction
        if path.lower().endswith('.zip'):
            print(f"{Fore.YELLOW}ZIP  {Fore.RESET} Extracting {path}...")
            
            with zipfile.ZipFile(path, 'r') as zip_ref:
                files = zip_ref.namelist()
                for file in tqdm(files, desc="      Unzipping", unit="file", leave=False):
                    zip_ref.extract(member=file, path=folder)
            
            os.remove(path)
            print(f"{Fore.GREEN}OK   {Fore.RESET} Extracted to {folder} and cleaned up ZIP.")
        
        # 4. Handle Regular Files
        else:
            # If filename wasn't provided, use the name gdown gave us
            final_name = filename if filename else os.path.basename(path)
            dest_path = os.path.join(folder, final_name)
            
            # Move and rename if necessary
            shutil.move(path, dest_path)
            print(f"{Fore.GREEN}OK   {Fore.RESET} File saved: {Style.BRIGHT}{dest_path}")

    except Exception as e:
        print(f"{Fore.RED}ERR  {Fore.RESET} Processing failed: {e}")

if __name__ == "__main__":
    # Format: (URL, Folder, Optional Filename)
    items = [
        # AGN data
        ("https://drive.google.com/file/d/1SOsMIjgxnPsS7OKtkTRDaAWy3bwlJhBr/view?usp=sharing", "results/data", 
         "nov10a_single_chisq_carma_mixscalar_nozband_highertaufastlim_removemix_fixband_lagblrband_chisq_spl_nofhost_bwb_lmc-6_N1w1000s200t14ch4.h5"),
        # other data
        ("https://drive.google.com/file/d/1sYr-N-DMpuWpbfdPg6zQ-IryP8InY5TK/view?usp=sharing", "./", "data/dr16q_prop_May01_2024.fits"),
        # nov2_sdss_mags.csv
        ("https://drive.google.com/file/d/1TbwmKDeBRDMPIGtQm8GcZhpRhcmsGUBO/view?usp=sharing", 'results/data', "nov2_sdss_mags.csv"),
        # spectra csv results #jaxqsofit_mar15c.csv
        ("https://drive.google.com/file/d/17ahDyRx_Lb7KZoIlj4tGoK5k_mrMv96n/view?usp=sharing", "results/data", "jaxqsofit_mar15c.csv"),
    ]

    print(f"{Style.BRIGHT}{Fore.MAGENTA}=== Google Drive Batch Downloader ===\n")
    
    for item in tqdm(items, desc="Overall Progress", unit="item"):
        # This unpacks (url, folder) or (url, folder, filename) safely
        download_and_extract(*item)
        print("-" * 50)

    print(f"\n{Fore.GREEN}{Style.BRIGHT}All tasks completed successfully!")
