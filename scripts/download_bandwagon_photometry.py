#!/usr/bin/env python3
"""Download Bandwagon photometry for the July 14 chi-square-selected sample.

The output is a long-form CSV with one row per object/filter measurement.
It queries GALEX UV, VHS, UKIDSS, 2MASS, AllWISE, AKARI, and IRAS photometry.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import time
from typing import Callable, Mapping

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from tqdm import tqdm


DEFAULT_INPUT = Path("data/jul14_master_input_file_chisqgt20.csv")
DEFAULT_OUTPUT = Path(
    "data/jul14_master_input_file_chisqgt20_bandwagon_photometry.csv"
)

# The requested wavelength coverage intentionally excludes optical catalogs.
PHOTOMETRY_CATALOGS: dict[str, str] = {
    # CDS XMatch does not expose the newer GR6+7 AIS table in the shape
    # expected by Bandwagon's GALEX normalization. Query GR5 while retaining
    # the stable ``galex_ais`` output key.
    "galex_ais": "II/312/ais",
    "vhs_dr5": "II/367/vhs_dr5",
    "ukidss_las_dr9": "II/319/las9",
    "2mass": "II/246/out",
    "allwise": "II/328/allwise",
    "akari_irc": "II/297/irc",
    "akari_fis": "II/298/fis",
    "iras_psc": "II/125/main",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-match the July 14 object list against Bandwagon's "
            "GALEX, VHS, UKIDSS, 2MASS, AllWISE, AKARI, and IRAS catalogs "
            "and write one long-form CSV."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV.")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV."
    )
    parser.add_argument(
        "--catalog",
        action="append",
        choices=tuple(PHOTOMETRY_CATALOGS),
        help="Catalog to query; repeat to select several. The default is all eight.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2_000,
        help="Number of input objects per CDS XMatch request batch.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of attempts per failed batch.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=5.0,
        help=(
            "Initial retry delay in seconds; subsequent delays use exponential "
            "backoff."
        ),
    )
    parser.add_argument(
        "--max-mag-error",
        type=float,
        default=None,
        help="If set, discard magnitude measurements with larger uncertainties.",
    )
    parser.add_argument(
        "--min-quality",
        type=int,
        default=2,
        help="Minimum AKARI/IRAS numeric flux-quality flag.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable astroquery's HTTP response cache.",
    )
    return parser.parse_args()


def load_sources(path: Path) -> Table:
    """Read and validate object identifiers and ICRS coordinates."""
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    sources = Table.read(path, format="ascii.csv")
    required = {"object_id", "ra", "dec"}
    missing = sorted(required.difference(sources.colnames))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    if len(sources) == 0:
        raise ValueError(f"Input CSV contains no objects: {path}")

    object_ids = np.asarray(sources["object_id"], dtype=str)
    ra = np.asarray(sources["ra"], dtype=float)
    dec = np.asarray(sources["dec"], dtype=float)

    if np.any(np.char.strip(object_ids) == ""):
        raise ValueError("Input CSV contains a blank object_id.")
    unique_ids, counts = np.unique(object_ids, return_counts=True)
    duplicate_ids = unique_ids[counts > 1]
    if len(duplicate_ids):
        preview = ", ".join(duplicate_ids[:10])
        raise ValueError(f"Input CSV contains duplicate object_id values: {preview}")
    if not np.all(np.isfinite(ra)) or not np.all(np.isfinite(dec)):
        raise ValueError("Input CSV contains a non-finite ra or dec value.")
    if np.any((ra < 0.0) | (ra >= 360.0)):
        raise ValueError("Input CSV contains ra outside [0, 360) degrees.")
    if np.any((dec < -90.0) | (dec > 90.0)):
        raise ValueError("Input CSV contains dec outside [-90, 90] degrees.")

    return Table({"object_id": object_ids, "ra": ra, "dec": dec})


def selected_catalogs(names: list[str] | None) -> dict[str, str]:
    if names is None:
        return dict(PHOTOMETRY_CATALOGS)
    # argparse validates names. dict.fromkeys removes accidental repetitions
    # while retaining the order supplied on the command line.
    return {name: PHOTOMETRY_CATALOGS[name] for name in dict.fromkeys(names)}


def _load_bandwagon() -> tuple[Callable, Callable]:
    try:
        from bandwagon import matches_to_photometry, xmatch_catalogs
    except ImportError as exc:
        raise RuntimeError(
            "Bandwagon is not installed. Install it with:\n"
            '  python -m pip install "bandwagon @ '
            'git+https://github.com/burke86/bandwagon.git"'
        ) from exc
    return xmatch_catalogs, matches_to_photometry


def download_photometry(
    sources: Table,
    *,
    catalogs: Mapping[str, str],
    batch_size: int,
    retries: int,
    retry_delay: float,
    max_mag_error: float | None,
    min_quality: int,
    cache: bool,
    xmatch_catalogs: Callable,
    matches_to_photometry: Callable,
) -> Table:
    """Download and normalize catalog photometry in bounded request batches."""
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if retries <= 0:
        raise ValueError("--retries must be positive.")
    if retry_delay < 0:
        raise ValueError("--retry-delay cannot be negative.")
    if min_quality < 0:
        raise ValueError("--min-quality cannot be negative.")
    if max_mag_error is not None and max_mag_error <= 0:
        raise ValueError("--max-mag-error must be positive.")

    photometry_batches: list[Table] = []
    total = len(sources)
    downloaded_rows = 0
    with tqdm(total=total, desc="Bandwagon", unit="object") as progress:
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            batch = sources[start:stop]
            coords = SkyCoord(
                ra=np.asarray(batch["ra"], dtype=float) * u.deg,
                dec=np.asarray(batch["dec"], dtype=float) * u.deg,
                frame="icrs",
            )
            source_ids = np.asarray(batch["object_id"], dtype=str)

            for attempt in range(1, retries + 1):
                try:
                    matches = xmatch_catalogs(
                        coords,
                        catalogs=catalogs,
                        source_id=source_ids,
                        cache=cache,
                    )
                    photometry = matches_to_photometry(
                        matches,
                        max_mag_err=max_mag_error,
                        min_quality=min_quality,
                    )
                    break
                except Exception as exc:
                    if attempt == retries:
                        raise RuntimeError(
                            f"Bandwagon batch {start}:{stop} failed after "
                            f"{retries} attempts."
                        ) from exc
                    delay = retry_delay * (2 ** (attempt - 1))
                    progress.write(
                        f"Batch {start}:{stop} attempt {attempt}/{retries} failed: "
                        f"{exc}. Retrying in {delay:g} seconds."
                    )
                    time.sleep(delay)

            if len(photometry):
                photometry_batches.append(photometry)
            downloaded_rows += len(photometry)
            progress.update(stop - start)
            progress.set_postfix(rows=f"{downloaded_rows:,}")

    if photometry_batches:
        output = vstack(photometry_batches, metadata_conflicts="silent")
    else:
        # Ask Bandwagon for its canonical empty output schema.
        output = matches_to_photometry({})

    if "source_id" in output.colnames:
        output.rename_column("source_id", "object_id")
    if len(output):
        output.sort(["object_id", "filter_name", "catalog"])
    return output


def write_csv_atomic(table: Table, output_path: Path) -> None:
    """Write a CSV completely before replacing the requested output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        table.write(temporary_path, format="ascii.csv", overwrite=True)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    catalogs = selected_catalogs(args.catalog)
    sources = load_sources(args.input)
    xmatch_catalogs, matches_to_photometry = _load_bandwagon()

    print(f"Loaded {len(sources):,} objects from {args.input}")
    print(f"Querying catalogs: {', '.join(catalogs)}")
    photometry = download_photometry(
        sources,
        catalogs=catalogs,
        batch_size=args.batch_size,
        retries=args.retries,
        retry_delay=args.retry_delay,
        max_mag_error=args.max_mag_error,
        min_quality=args.min_quality,
        cache=args.cache,
        xmatch_catalogs=xmatch_catalogs,
        matches_to_photometry=matches_to_photometry,
    )
    write_csv_atomic(photometry, args.output)
    matched_objects = (
        len(set(np.asarray(photometry["object_id"], dtype=str)))
        if len(photometry)
        else 0
    )
    print(
        f"Wrote {len(photometry):,} rows for {matched_objects:,} objects "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
