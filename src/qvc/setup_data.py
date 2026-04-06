import os
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import gdown
from colorama import Fore, Style, init
from tqdm import tqdm


init(autoreset=True)


def fetch_dustmaps():
    from dustmaps.config import config
    import dustmaps.sfd

    print(f"{Fore.BLUE}RUN  {Fore.RESET} Resetting dustmaps config...")
    config.reset()
    print(f"{Fore.BLUE}RUN  {Fore.RESET} Fetching dustmaps SFD data...")
    dustmaps.sfd.fetch()
    print(f"{Fore.GREEN}OK   {Fore.RESET} dustmaps SFD data fetched.")


COMMANDS = {
    "fetch_dustmaps": fetch_dustmaps,
}


def is_google_drive_url(url):
    parsed = urllib.parse.urlparse(url)
    return "drive.google.com" in parsed.netloc.lower()


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{Fore.CYAN}DIR  {Fore.RESET} Created: {Style.DIM}{path}")


def resolve_target_path(folder, filename):
    if filename is None:
        return None
    return os.path.join(folder, filename)


def ensure_parent_directory(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def infer_filename_from_url(url):
    path = urllib.parse.urlparse(url).path
    name = Path(path).name
    return name or "downloaded_file"


def infer_filename_from_response(response):
    headers = response.info()
    filename = headers.get_filename()
    if filename:
        return filename
    return None


def download_from_google_drive(url):
    print(f"{Fore.BLUE}GET  {Fore.RESET} Fetching from Drive...")
    return gdown.download(url, quiet=False, fuzzy=True)


def download_from_http(url, filename=None):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme for non-Google download: {parsed.scheme or '<missing>'}")

    print(f"{Fore.BLUE}GET  {Fore.RESET} Fetching via HTTP...")
    with urllib.request.urlopen(url) as response:
        resolved_name = filename or infer_filename_from_response(response) or infer_filename_from_url(url)
        suffix = Path(resolved_name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or "") as handle:
            shutil.copyfileobj(response, handle)
            temp_path = handle.name

    return temp_path, resolved_name


def extract_zip(path, folder):
    print(f"{Fore.YELLOW}ZIP  {Fore.RESET} Extracting {path}...")
    with zipfile.ZipFile(path, "r") as zip_ref:
        files = zip_ref.namelist()
        for file in tqdm(files, desc="      Unzipping", unit="file", leave=False):
            zip_ref.extract(member=file, path=folder)
    os.remove(path)
    print(f"{Fore.GREEN}OK   {Fore.RESET} Extracted to {folder} and cleaned up ZIP.")


def save_regular_file(source_path, folder, filename=None):
    final_name = filename if filename else os.path.basename(source_path)
    dest_path = os.path.join(folder, final_name)
    ensure_parent_directory(dest_path)
    shutil.move(source_path, dest_path)
    print(f"{Fore.GREEN}OK   {Fore.RESET} File saved: {Style.BRIGHT}{dest_path}")


def normalize_step(step):
    if isinstance(step, dict):
        return step

    if isinstance(step, (tuple, list)):
        if len(step) == 2:
            url, folder = step
            filename = None
        elif len(step) == 3:
            url, folder, filename = step
        else:
            raise ValueError(f"Unsupported download step shape: {step!r}")
        return {"type": "download", "url": url, "folder": folder, "filename": filename}

    raise TypeError(f"Unsupported step type: {type(step).__name__}")


def download_and_extract(url, folder, filename=None):
    """
    Download from Google Drive or generic HTTP(S) and extract ZIP archives.
    Skips download if filename is provided and already exists in folder.
    """
    ensure_directory(folder)

    target_path = resolve_target_path(folder, filename)
    if target_path and os.path.exists(target_path):
        print(f"{Fore.YELLOW}SKIP {Fore.RESET} {filename} already exists in {folder}. Skipping.")
        return

    try:
        if is_google_drive_url(url):
            path = download_from_google_drive(url)
            resolved_name = filename or (os.path.basename(path) if path else None)
        else:
            path, resolved_name = download_from_http(url, filename=filename)

        if not path:
            print(f"{Fore.RED}FAIL {Fore.RESET} Download returned no path for {url}")
            return

        zip_name = resolved_name or os.path.basename(path)
        if zip_name.lower().endswith(".zip") or path.lower().endswith(".zip"):
            extract_zip(path, folder)
        else:
            save_regular_file(path, folder, filename=filename or resolved_name)

    except Exception as e:
        print(f"{Fore.RED}ERR  {Fore.RESET} Processing failed: {e}")


def run_command_step(name):
    try:
        command = COMMANDS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown setup command: {name}") from exc

    print(f"{Fore.MAGENTA}CMD  {Fore.RESET} Running setup command: {Style.BRIGHT}{name}")
    command()


def run_step(step):
    normalized = normalize_step(step)
    step_type = normalized.get("type", "download")

    if step_type == "download":
        download_and_extract(
            normalized["url"],
            normalized["folder"],
            normalized.get("filename"),
        )
        return

    if step_type == "command":
        run_command_step(normalized["name"])
        return

    raise ValueError(f"Unknown setup step type: {step_type}")


if __name__ == "__main__":
    steps = [
        {
            "type": "download",
            "url": "https://drive.google.com/file/d/1sYr-N-DMpuWpbfdPg6zQ-IryP8InY5TK/view?usp=sharing",
            "folder": "./",
            "filename": "data/dr16q_prop_May01_2024.fits",
        },
        {
            "type": "download",
            "url": "https://drive.google.com/file/d/1bq1GEBEApSgJz0ezyxIx6ORz79epmQuU/view?usp=sharing",
            "folder": "results/data",
            "filename": "light_curves.h5",
        },
        {
            "type": "download",
            "url": "https://drive.google.com/file/d/11py-CEJuszTn12eMTEn4cMLD4vbJJ15e/view?usp=sharing",
            "folder": "results/data",
            "filename": "spectra.csv",
        },
        {
            "type": "download",
            "url": "https://drive.google.com/file/d/1j39Tc1vy3nnCdVayWIKu-6IlOeC_moke/view?usp=sharing",
            "folder": "data/spectra_cache",
            "filename": "spec-9180-57693-0463.fits",
        },
        {
            "type": "download",
            "url": "https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/ssp_data_fsps_v3.2_lgmet_age.h5",
            "folder": "data/",
            "filename": "ssp_data_fsps_v3.2_lgmet_age.h5",
        },
        {"type": "command", "name": "fetch_dustmaps"},
    ]

    print(f"{Style.BRIGHT}{Fore.MAGENTA}=== Setup Runner ===\n")

    for step in tqdm(steps, desc="Overall Progress", unit="step"):
        run_step(step)
        print("-" * 50)

    print(f"\n{Fore.GREEN}{Style.BRIGHT}All tasks completed successfully!")
