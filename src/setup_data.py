# pip install gdown tqdm colorama

import gdown
import os
import zipfile
import shutil
from tqdm import tqdm
from colorama import Fore, Style, init

# Initialize colorama for Windows/Linux/Mac compatibility
init(autoreset=True)

def download_and_extract(url, folder):
    """
    Downloads files/folders from Google Drive and extracts ZIPs with status updates.
    """
    # 1. Setup Directory
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"{Fore.CYAN}DIR  {Fore.RESET} Created: {Style.DIM}{folder}")

    try:
        # 2. Download (gdown's built-in progress bar will show here)
        print(f"{Fore.BLUE}GET  {Fore.RESET} Fetching from Drive...")
        
        # fuzzy=True extracts ID from URL automatically
        # quiet=False keeps the gdown progress bar visible
        path = gdown.download(url, quiet=False, fuzzy=True)
        
        if not path:
            print(f"{Fore.RED}FAIL {Fore.RESET} Download returned no path for {url}")
            return

        # 3. Handle ZIP Extraction
        if path.lower().endswith('.zip'):
            print(f"{Fore.YELLOW}ZIP  {Fore.RESET} Extracting {path}...")
            
            with zipfile.ZipFile(path, 'r') as zip_ref:
                # Get list of files to show a mini-progress for extraction if large
                files = zip_ref.namelist()
                for file in tqdm(files, desc="      Unzipping", unit="file", leave=False):
                    zip_ref.extract(member=file, path=folder)
            
            os.remove(path)
            print(f"{Fore.GREEN}OK   {Fore.RESET} Extracted to {folder} and cleaned up ZIP.")
        
        # 4. Handle Regular Files
        else:
            dest_path = os.path.join(folder, os.path.basename(path))
            # Move only if the file isn't already in the target folder
            if os.path.abspath(path) != os.path.abspath(dest_path):
                shutil.move(path, dest_path)
            print(f"{Fore.GREEN}OK   {Fore.RESET} File saved: {Style.BRIGHT}{dest_path}")

    except Exception as e:
        print(f"{Fore.RED}ERR  {Fore.RESET} Processing failed: {e}")

if __name__ == "__main__":
    items = [
        # production hubble posteriors
        ("https://drive.google.com/file/d/1WWLtKjTlVQr_qs1Q6VtnJQZc25g3XZ-6/view?usp=sharing", "results/hubble_posteriors"),
        # test hubble posteriors
        ("https://drive.google.com/file/d/1PuQTpsEP_S-6rJF6VddOOk-RghbuLOci/view?usp=sharing", "results/hubble_posteriors"),
        # AGN data
        ("https://drive.google.com/file/d/1SOsMIjgxnPsS7OKtkTRDaAWy3bwlJhBr/view?usp=sharing", "results/data"),
        # other data
        ("https://drive.google.com/file/d/1sYr-N-DMpuWpbfdPg6zQ-IryP8InY5TK/view?usp=sharing", "./"),
    ]

    print(f"{Style.BRIGHT}{Fore.MAGENTA}=== Google Drive Batch Downloader ===\n")
    
    # Global progress bar for the entire list of items
    for url, folder in tqdm(items, desc="Overall Progress", unit="item"):
        download_and_extract(url, folder)
        print("-" * 50)

    print(f"\n{Fore.GREEN}{Style.BRIGHT}All tasks completed successfully!")