import argparse
import gzip
import hashlib
from importlib.resources import files as package_files
import http.client
import os
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.error
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import gdown
from colorama import Fore, Style, init
from tqdm import tqdm


init(autoreset=True)

DUSTMAPS_SENTINEL = ("sfd", "SFD_dust_4096_ngp.fits")
CSC_ENDPOINT = "https://cda.cfa.harvard.edu/csc21tap/sync"
CSC_CATALOG = Path("results/diagnostics/chandra_precut_sep13/csc21_region_download.vot")
CSC_COLUMNS = (
    "name", "ra", "dec", "err_ellipse_r0", "err_ellipse_r1", "err_ellipse_ang",
    "significance", "likelihood_class", "conf_flag", "sat_src_flag", "streak_src_flag",
    "flux_aper_b", "flux_aper_lolim_b", "flux_aper_hilim_b",
    "flux_aper_w", "flux_aper_lolim_w", "flux_aper_hilim_w",
    "flux_aper_avg_b", "flux_aper_avg_lolim_b", "flux_aper_avg_hilim_b",
    "flux_aper_avg_w", "flux_aper_avg_lolim_w", "flux_aper_avg_hilim_w",
)
# Preserve the paper's full regional catalog; do not prefilter using the older
# one-arcsecond matching rule. Hubble analysis uses the CSC error ellipses.
CSC_QUERY = (
    "SELECT " + ", ".join(f"m.{column}" for column in CSC_COLUMNS)
    + " FROM csc21.master_source m WHERE m.dec BETWEEN -1.251 AND 1.251"
)


def validate_csc_catalog(path):
    from astropy.io.votable import parse
    import numpy as np

    statuses = [
        item.attrib.get("value") for item in ET.parse(path).iter()
        if item.tag.rsplit("}", 1)[-1] == "INFO"
        and item.attrib.get("name") == "QUERY_STATUS"
    ]
    if not statuses or any(status != "OK" for status in statuses):
        raise ValueError(f"CSC TAP query failed or was truncated: {statuses}")
    table = parse(path).get_first_table().to_table(use_names_over_ids=True)
    missing = set(CSC_COLUMNS) - set(table.colnames)
    if missing or not len(table):
        raise ValueError(f"Invalid CSC catalog: {len(table)} rows, missing columns {sorted(missing)}")
    for column in ("ra", "dec"):
        values = np.ma.asarray(table[column])
        if np.any(np.ma.getmaskarray(values)) or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid CSC coordinates in {column}")
    return len(table)


def fetch_csc():
    """Download the paper's CSC 2.1 region as a validated, ready-to-use VOTable."""
    destination = CSC_CATALOG
    if destination.exists():
        count = validate_csc_catalog(destination)
        print(f"{Fore.YELLOW}SKIP {Fore.RESET} Valid CSC catalog: {destination} ({count} sources)")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    query = urllib.parse.urlencode({
        "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable",
        "MAXREC": "1000000", "QUERY": CSC_QUERY,
    })
    path, _ = download_from_http(
        f"{CSC_ENDPOINT}?{query}", filename=destination.name, timeout=60,
    )
    try:
        count = validate_csc_catalog(path)
        # Stage beside the destination so installation is atomic across filesystems.
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            staged = Path(handle.name)
        try:
            shutil.copyfile(path, staged)
            staged.replace(destination)
        finally:
            staged.unlink(missing_ok=True)
        destination.with_suffix(".adql").write_text(CSC_QUERY + "\n")
        print(f"{Fore.GREEN}OK   {Fore.RESET} CSC catalog: {destination} ({count} sources)")
    finally:
        Path(path).unlink(missing_ok=True)


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


LIGHT_CURVE_DEMO_URL = "https://drive.google.com/file/d/1MghLBBfCaS-7e8jHyI7yWzKja3vpVP85/view?usp=sharing"
LIGHT_CURVE_DEMO_MD5 = "520a3a323a60d65bbb177f45bd8a20cd"
LIGHT_CURVE_DEMO_FILES = (
    "S82/Catalog.parquet", "S82/dr16s82_sdssLCRaw.parquet",
    "S82/dr16s82_ps1LCRaw.parquet", "S82/dr16s82_ZuberLCRaw.parquet",
    "S82/dr16s82_sdssSeeing.parquet", "S82/dr16s82_ps1Seeing.parquet",
    "S82/dr16s82_ztfSeeing.parquet", "dr16q_prop_May01_2024.fits",
)


SPECTRUM_DEMO_URL = "https://drive.google.com/file/d/1RRRLpcsvros4K-AKW8dAwHBEhVK8y2FT/view?usp=sharing"
SPECTRUM_DEMO_MD5 = "3bfb733c29a47308c21b7ea2937e27ac"
SPECTRUM_DEMO_FILES = (
    "spectra_cache_all/spec-9202-57713-0009.fits",
    "demo_J233151.91+003258.3_sed.csv",
)


def _fetch_single_object_bundle(url, checksum, files, *, allow_matching_partial=False):
    """Install a checked bundle without replacing existing catalogs or spectra."""
    destination = Path(os.environ.get("QVC_DATA_DIR", "data")).expanduser()
    present = [(destination / name).is_file() for name in files]
    if all(present):
        print(f"SKIP Single-object inputs already exist in {destination}.")
        return
    if not allow_matching_partial and any((destination / name).exists() for name in files):
        raise FileExistsError(
            f"Incomplete existing single-object inputs in {destination}; "
            "use a separate empty data directory to avoid mixing catalogs."
        )
    with tempfile.TemporaryDirectory() as work:
        archive = Path(work) / "single_object_demo.zip"
        downloaded = gdown.download(url, output=str(archive), quiet=False, fuzzy=True)
        if not downloaded:
            raise RuntimeError("Single-object download returned no file")
        verify_md5(archive, checksum)
        with zipfile.ZipFile(archive) as bundle:
            expected = {f"data/{name}" for name in files}
            actual = {entry.filename for entry in bundle.infolist() if not entry.is_dir()}
            if actual != expected:
                raise ValueError("Unexpected files in the single-object archive")
            if bundle.testzip() is not None:
                raise ValueError("Corrupt single-object archive")
            # Check every existing member before installing any missing member.
            # This permits upgrading the old spectrum-only bundle safely.
            for name in files:
                target = destination / name
                if target.exists() and (
                    not target.is_file() or target.read_bytes() != bundle.read(f"data/{name}")
                ):
                    raise FileExistsError(f"Different existing demo data at {target}; refusing to overwrite")
            for name in files:
                target = destination / name
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(f"data/{name}") as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    print(f"OK Installed SDSS J233151.91+003258.3 (object 1448770) in {destination}.")


def fetch_light_curve_demo():
    _fetch_single_object_bundle(LIGHT_CURVE_DEMO_URL, LIGHT_CURVE_DEMO_MD5, LIGHT_CURVE_DEMO_FILES)


def fetch_spectrum_demo():
    _fetch_single_object_bundle(
        SPECTRUM_DEMO_URL, SPECTRUM_DEMO_MD5, SPECTRUM_DEMO_FILES,
        allow_matching_partial=True,
    )


def install_demo_sed_photometry():
    """Install the measured broadband SED rows shipped with the single-object demo."""
    destination = Path(os.environ.get("QVC_DATA_DIR", "data")).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    resources = package_files("qvc.spectra").joinpath("demo_inputs")
    for name in ("demo_J233151.91+003258.3_sed.csv", "demo_J233151.91+003258.3_sed.provenance.json"):
        target = destination / name
        content = resources.joinpath(name).read_bytes()
        if target.exists():
            if target.read_bytes() != content:
                raise FileExistsError(f"Different existing demo SED data at {target}; refusing to overwrite")
            continue
        target.write_bytes(content)
    print(f"OK Single-object broadband SED photometry installed in {destination}.")


COMMANDS = {
    "install_demo_sed_photometry": install_demo_sed_photometry,
    "fetch_spectrum_demo": fetch_spectrum_demo,
    "fetch_light_curve_demo": fetch_light_curve_demo,
    "fetch_dustmaps": fetch_dustmaps,
    "fetch_csc": fetch_csc,
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
        "url": "https://zenodo.org/records/23021880/files/light_curve_data.h5?download=1",
        "folder": "results/data",
        "filename": "light_curve_data.h5",
        "md5": "92eaa1c649e8380cdf1190b11db6cfcb",
    },
    {
        "type": "download",
        "url": "https://zenodo.org/records/23021880/files/spectra_data.h5?download=1",
        "folder": "results/data",
        "filename": "spectra_data.h5",
        "md5": "6ce97a192dc077929a1de73059e7a8a9",
    },
    {"type": "command", "name": "fetch_spectrum_demo"},
    {"type": "command", "name": "install_demo_sed_photometry"},
    {
        "type": "download",
        "url": "https://zenodo.org/records/23021880/files/hubble_posteriors.zip?download=1",
        "folder": "results/",
        "filename": "hubble_posteriors.zip",
        "md5": "772a6904baeb23dbf3eef16a406563e8",
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
    
    {"type": "command", "name": "fetch_csc"},
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
        "url": "https://faculty.washington.edu/ivezic/macleod/qso_dr7/QSO_S82.tar.gz",
        "folder": "data/MacLeod2010",
        "filename": "QSO_S82.tar.gz",
        "skip_if_exists": "QSO_S82",
    },
    {
        "type": "download",
        "url": "https://faculty.washington.edu/ivezic/macleod/qso_dr7/DB_QSO_S82.dat.gz",
        "folder": "data/MacLeod2010",
        "filename": "DB_QSO_S82.dat.gz",
        "skip_if_exists": "DB_QSO_S82.dat",
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

# The paper resume needs the published fit summaries/checkpoints and the two
# external Pantheon/CSC inputs, not the raw fitting catalogs or the old Shen mock.
PAPER_STEPS = [
    step for step in DEFAULT_STEPS
    if "zenodo.org/records/23021880/" in step.get("url", "")
    or step.get("filename", "").startswith("Pantheon+SH0ES")
    or step.get("name") == "fetch_csc"
]


def verify_md5(path, expected):
    if expected is None:
        return
    with open(path, "rb") as stream:
        actual = hashlib.file_digest(stream, "md5").hexdigest()
    if actual != expected:
        raise ValueError(f"MD5 mismatch for {path}: expected {expected}, got {actual}")


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


def download_from_http(url, filename=None, *, timeout=None):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme for non-Google download: {parsed.scheme or '<missing>'}")

    request_options = {} if timeout is None else {"timeout": timeout}
    with urllib.request.urlopen(url, **request_options) as response:
        if response.info().get_content_type() == "text/html":
            raise ValueError(f"Expected a data file but received an HTML page from {url}")
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
            temp_path = handle.name
            try:
                downloaded_bytes = 0
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
                        downloaded_bytes += len(chunk)
                        progress.update(len(chunk))
                if total_bytes is not None and downloaded_bytes < total_bytes:
                    raise http.client.IncompleteRead(b"", total_bytes - downloaded_bytes)
                handle.flush()
            except BaseException:
                Path(temp_path).unlink(missing_ok=True)
                raise

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


def download_and_extract(url, folder, filename=None, skip_if_exists=None, md5=None):
    """
    Download from Google Drive or generic HTTP(S) and extract supported archives.
    Skips download if filename or an extracted sentinel already exists.
    """
    ensure_directory(folder)

    target_path = resolve_target_path(folder, filename)
    skip_path = resolve_skip_path(folder, skip_if_exists)
    if target_path and os.path.exists(target_path):
        verify_md5(target_path, md5)
        if is_zip_archive(filename) or is_tar_archive(filename) or is_gzip_archive(filename):
            # A cached archive is not evidence that its contents were installed.
            # Extract a temporary copy so the user's cached archive is retained.
            with tempfile.NamedTemporaryFile(delete=False) as handle:
                cached_copy = handle.name
            shutil.copyfile(target_path, cached_copy)
            extract_archive(cached_copy, folder, filename)
            return
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
        verify_md5(path, md5)
        if not extract_archive(path, folder, archive_name):
            save_regular_file(path, folder, filename=filename or resolved_name)

    except Exception as e:
        print(f"{Fore.RED}ERR  {Fore.RESET} Processing failed: {e}")
        raise


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
            normalized.get("md5"),
        )
        return

    if step_type == "command":
        run_command_step(normalized["name"])
        return

    raise ValueError(f"Unknown setup step type: {step_type}")


def build_parser():
    parser = argparse.ArgumentParser(description="Download demo or appendix datasets used by QVC.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--spectra", action="store_true",
        help="Download single-object photometry, cached spectrum, stellar templates and dust maps.",
    )
    selection.add_argument(
        "--light-curve", action="store_true",
        help="Download only the raw single-object inputs for SDSS J233151.91+003258.3.",
    )
    selection.add_argument(
        "--paper",
        action="store_true",
        help="Download the published Hubble resume inputs and external Pantheon/CSC data.",
    )
    selection.add_argument(
        "--appendix",
        action="store_true",
        help="Download only the appendix MacLeod 2010 dataset under data/MacLeod2010.",
    )
    return parser


def get_steps(args):
    if args.spectra:
        template = next(dict(step) for step in DEFAULT_STEPS
                        if step.get("filename") == "ssp_data_continuum_fsps_v3.2_lgmet_age.h5")
        template["folder"] = os.environ.get("QVC_DATA_DIR", "data")
        return [
            {"type": "command", "name": "fetch_light_curve_demo"},
            {"type": "command", "name": "fetch_spectrum_demo"},
            {"type": "command", "name": "install_demo_sed_photometry"},
            template,
            {"type": "command", "name": "fetch_dustmaps"},
        ]
    if args.light_curve:
        return [{"type": "command", "name": "fetch_light_curve_demo"}]
    if args.paper:
        return PAPER_STEPS
    if args.appendix:
        return APPENDIX_STEPS
    return DEFAULT_STEPS


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    steps = get_steps(args)

    print(f"{Style.BRIGHT}{Fore.MAGENTA}=== Setup Runner ===\n")

    csc_unavailable = False
    for step in tqdm(steps, desc="Overall Progress", unit="step"):
        try:
            run_step(step)
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.IncompleteRead) as exc:
            normalized = normalize_step(step)
            if normalized.get("type") != "command" or normalized.get("name") != "fetch_csc":
                raise
            csc_unavailable = True
            print(
                f"{Fore.YELLOW}WARN {Fore.RESET} CSC unavailable from {CSC_ENDPOINT}: {exc}. "
                "Continuing setup without CSC; Hubble will skip X-ray enrichment "
                "when the catalog is missing. The next setup run will retry."
            )
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
        print("-" * 50)

    if csc_unavailable:
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}Setup completed with CSC unavailable.")
    else:
        print(f"\n{Fore.GREEN}{Style.BRIGHT}All tasks completed successfully!")


if __name__ == "__main__":
    main()
