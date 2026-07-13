import argparse
import gzip
import os
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import gdown
from colorama import Fore, Style, init
from tqdm import tqdm


init(autoreset=True)

DUSTMAPS_SENTINEL = ("sfd", "SFD_dust_4096_ngp.fits")


def fetch_dustmaps():
    from dustmaps.config import config
    import dustmaps.sfd

    configured_root = os.environ.get("QVC_DUSTMAPS_DIR") or "results/dustmaps"
    configured_dir = Path(configured_root).expanduser()
    config["data_dir"] = str(configured_dir)
    sentinel = configured_dir.joinpath(*DUSTMAPS_SENTINEL)
    if sentinel.exists():
        print(
            f"{Fore.YELLOW}SKIP {Fore.RESET} dustmaps SFD data already exists at "
            f"{Style.BRIGHT}{sentinel}"
        )
        return

    print(f"{Fore.BLUE}RUN  {Fore.RESET} Fetching dustmaps SFD data...")
    dustmaps.sfd.fetch()
    print(f"{Fore.GREEN}OK   {Fore.RESET} dustmaps SFD data fetched.")


COMMANDS = {
    "fetch_dustmaps": fetch_dustmaps,
}

DEFAULT_STEPS = [
    {
        "type": "download",
        "url": "https://drive.google.com/file/d/1LRTJOGOWTPnQZQMsKBpOje8sZbf3T4bn/view?usp=sharing",
        "folder": "./",
        "filename": "data/dr16q_prop_May01_2024.fits",
    },
    {
        "type": "download",
        "url": "https://drive.google.com/file/d/1PUVRL7AlyG_15wKpkpF5wR-uYpCS8q11/view?usp=sharing",
        "folder": "results/data",
        "filename": "lc_data_all.h5",
    },
    {
        "type": "download",
        "url": "https://drive.google.com/file/d/1VtccnFl5WIan4pZHfyia6KjhgkrCAHrt/view?usp=sharing",
        "folder": "results/data",
        "filename": "spectra_data_all.csv",
    },
    {
        "type": "download",
        "url": "https://drive.google.com/file/d/1UTGwSZXfLm8kSAKTDHeS9r685isycLeC/view?usp=sharing",
        "folder": "data/spectra_cache_all",
        "filename": "spec-9152-58041-0926.fits",
    },
    {
        "type": "download",
        "url": "https://drive.google.com/file/d/1QNOzH3_gmM1mCQezQJdW557MpUQdc2UW/view?usp=sharing",
        "folder": "results/",
    },
    {
        "type": "download",
        "url": "https://drive.google.com/file/d/1KDnXK3pSWD3ZtFSIHIoDMjYJBviRdnoF/view?usp=sharing",
        "folder": "results/data/",
        "filename": "mock_completeness_catalog_fresh.h5",
    },
    {
        "type": "download",
        "url": "https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/ssp_data_continuum_fsps_v3.2_lgmet_age.h5",
        "folder": "data/",
        "filename": "ssp_data_continuum_fsps_v3.2_lgmet_age.h5",
    },
    {
        "type": "download",
        "url": "https://drive.google.com/file/d/1zg2-T8Y5C4iEiPpWy3UUc21dUYjAttlU/view?usp=sharing",
        "folder": "results/cosmo/",
    },
    {
        "type": "download",
        "url": "https://github.com/PantheonPlusSH0ES/DataRelease/raw/c447f0fea703fcd0fff57de5000947b5ca81286b/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov",
        "folder": "data/",
        "filename": "Pantheon+SH0ES_STAT+SYS.cov",
    },
    {
        "type": "download",
        "url": "https://github.com/PantheonPlusSH0ES/DataRelease/raw/c447f0fea703fcd0fff57de5000947b5ca81286b/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat",
        "folder": "data/",
        "filename": "Pantheon+SH0ES.dat",
    },
    
    {"type": "command", "name": "fetch_dustmaps"},
]

APPENDIX_STEPS = [
    {
        "type": "download",
        "url": "https://faculty.washington.edu/ivezic/macleod/qso_dr7/s82drw.tar.gz",
        "folder": "data/MacLeod2010",
        "filename": "s82drw.tar.gz",
        "skip_if_exists": "s82drw",
    },
    {
        "type": "download",
        "url": "https://zenodo.org/records/7624056/files/TotalDat.fits.gz?download=1",
        "folder": "data/Stone2021",
        "filename": "TotalDat.fits.gz",
        "skip_if_exists": "TotalDat.fits",
    },
    # {
    #     "type": "download",
    #     "url": "https://data.sdss.org/sas/dr17/sdss/spectro/redux/specObj-dr17.fits",
    #     "folder": "data/SDSS_DR17",
    #     "filename": "specObj-dr17.fits",
    # }
]


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


def resolve_skip_path(folder, skip_if_exists):
    if skip_if_exists is None:
        return None
    return os.path.join(folder, skip_if_exists)


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

    with urllib.request.urlopen(url) as response:
        resolved_name = filename or infer_filename_from_response(response) or infer_filename_from_url(url)
        print(f"{Fore.BLUE}GET  {Fore.RESET} Fetching via HTTP: {resolved_name}...")
        headers = getattr(response, "headers", None)
        if headers is None:
            headers = response.info()
        content_length = headers.get("Content-Length")
        total_bytes = None
        if content_length:
            try:
                parsed_length = int(content_length)
            except ValueError:
                parsed_length = None
            if parsed_length is not None and parsed_length >= 0:
                total_bytes = parsed_length
        suffix = Path(resolved_name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or "") as handle:
            with tqdm(
                total=total_bytes,
                desc=f"      Downloading {Path(resolved_name).name}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=False,
            ) as progress:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    progress.update(len(chunk))
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


def extract_tar(path, folder):
    print(f"{Fore.YELLOW}TAR  {Fore.RESET} Extracting {path}...")
    with tarfile.open(path, "r:*") as tar_ref:
        members = tar_ref.getmembers()
        for member in tqdm(members, desc="      Untarring", unit="file", leave=False):
            tar_ref.extract(member=member, path=folder, filter="data")
    os.remove(path)
    print(f"{Fore.GREEN}OK   {Fore.RESET} Extracted to {folder} and cleaned up TAR.")


def extract_gzip(path, folder, archive_name):
    output_name = Path(archive_name).name.removesuffix(".gz")
    dest_path = Path(folder) / output_name
    ensure_parent_directory(str(dest_path))
    print(f"{Fore.YELLOW}GZIP {Fore.RESET} Extracting {path}...")
    with gzip.open(path, "rb") as src, open(dest_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove(path)
    print(f"{Fore.GREEN}OK   {Fore.RESET} Extracted to {dest_path} and cleaned up GZIP.")


def is_zip_archive(name):
    return name.lower().endswith(".zip")


def is_tar_archive(name):
    lowered = name.lower()
    return lowered.endswith(".tar") or lowered.endswith(".tar.gz") or lowered.endswith(".tgz")


def is_gzip_archive(name):
    lowered = name.lower()
    return lowered.endswith(".gz") and not is_tar_archive(lowered)


def extract_archive(path, folder, archive_name):
    if is_zip_archive(archive_name) or is_zip_archive(path):
        extract_zip(path, folder)
        return True

    if is_tar_archive(archive_name) or is_tar_archive(path):
        extract_tar(path, folder)
        return True

    if is_gzip_archive(archive_name) or is_gzip_archive(path):
        extract_gzip(path, folder, archive_name)
        return True

    return False


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
        return {
            "type": "download",
            "url": url,
            "folder": folder,
            "filename": filename,
            "skip_if_exists": None,
        }

    raise TypeError(f"Unsupported step type: {type(step).__name__}")


def download_and_extract(url, folder, filename=None, skip_if_exists=None):
    """
    Download from Google Drive or generic HTTP(S) and extract supported archives.
    Skips download if filename or an extracted sentinel already exists.
    """
    ensure_directory(folder)

    target_path = resolve_target_path(folder, filename)
    skip_path = resolve_skip_path(folder, skip_if_exists)
    if target_path and os.path.exists(target_path):
        print(f"{Fore.YELLOW}SKIP {Fore.RESET} {filename} already exists in {folder}. Skipping.")
        return
    if skip_path and os.path.exists(skip_path):
        print(f"{Fore.YELLOW}SKIP {Fore.RESET} {skip_if_exists} already exists in {folder}. Skipping.")
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

        archive_name = resolved_name or os.path.basename(path)
        if not extract_archive(path, folder, archive_name):
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
            normalized.get("skip_if_exists"),
        )
        return

    if step_type == "command":
        run_command_step(normalized["name"])
        return

    raise ValueError(f"Unknown setup step type: {step_type}")


def build_parser():
    parser = argparse.ArgumentParser(description="Download demo or appendix datasets used by QVC.")
    parser.add_argument(
        "--appendix",
        action="store_true",
        help="Download only the appendix MacLeod 2010 dataset under data/MacLeod2010.",
    )
    return parser


def get_steps(args):
    if args.appendix:
        return APPENDIX_STEPS
    return DEFAULT_STEPS


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    steps = get_steps(args)

    print(f"{Style.BRIGHT}{Fore.MAGENTA}=== Setup Runner ===\n")

    for step in tqdm(steps, desc="Overall Progress", unit="step"):
        run_step(step)
        print("-" * 50)

    print(f"\n{Fore.GREEN}{Style.BRIGHT}All tasks completed successfully!")


if __name__ == "__main__":
    main()
