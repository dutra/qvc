"""Fetch per-epoch image quality for QVC Stripe 82 light curves."""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from qvc.hubble.hubble_utils import resolve_qvc_data_path


SDSS_BANDS = ("u", "g", "r", "i", "z")
S82_DIR = "data/S82"


def _read_csv_url(base_url, params, *, comment=None):
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "qvc-seeing/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        text = response.read().decode("utf-8")
    return pd.read_csv(StringIO(text), comment=comment)


def query_ps1_detection_seeing(ps1objid):
    table = _read_csv_url(
        "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/detection.csv",
        {
            "objID": str(int(ps1objid)),
            "columns": "[detectID,psfMajorFWHM,psfMinorFWHM]",
        },
    )
    major = pd.to_numeric(table["psfMajorFWHM"], errors="coerce")
    minor = pd.to_numeric(table["psfMinorFWHM"], errors="coerce")
    table["psf_fwhm_arcsec"] = np.sqrt(major * minor)
    return table[["detectID", "psf_fwhm_arcsec"]]


def query_ztf_image_seeing(ra, dec):
    table = _read_csv_url(
        "https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci",
        {
            "POS": f"{float(ra):.10f},{float(dec):.10f}",
            "COLUMNS": "field,rcid,fid,obsjd,seeing",
            "ct": "csv",
        },
    )
    table["mjd"] = pd.to_numeric(table["obsjd"], errors="coerce") - 2400000.5
    return table.rename(
        columns={
            "field": "fieldid",
            "rcid": "rcidin",
            "fid": "filterID",
            "seeing": "psf_fwhm_arcsec",
        }
    )[["mjd", "fieldid", "rcidin", "filterID", "psf_fwhm_arcsec"]]


def query_sdss_field_seeing(ra, dec, radius_arcsec=2.0):
    half_width = float(radius_arcsec) / 3600.0
    columns = ",".join(f"f.psfWidth_{band}" for band in SDSS_BANDS)
    sql = (
        "SELECT p.objID,p.ra,p.dec,p.mjd,p.run,p.rerun,p.camcol,p.field,"
        f"{columns} FROM PhotoObjAll p JOIN Field f ON p.fieldID=f.fieldID "
        f"WHERE p.ra BETWEEN {float(ra) - half_width:.10f} "
        f"AND {float(ra) + half_width:.10f} "
        f"AND p.dec BETWEEN {float(dec) - half_width:.10f} "
        f"AND {float(dec) + half_width:.10f} ORDER BY p.mjd"
    )
    table = _read_csv_url(
        "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch",
        {"cmd": sql, "format": "csv"},
        comment="#",
    )
    if table.empty:
        return table
    separation = np.hypot(
        (pd.to_numeric(table["ra"]) - float(ra)) * np.cos(np.deg2rad(float(dec))),
        pd.to_numeric(table["dec"]) - float(dec),
    )
    table = table.assign(_separation=separation).sort_values("_separation")
    return table.drop_duplicates("mjd", keep="first")


def _nearest_ztf_seeing(raw, archive, max_delta_days=0.01):
    values = np.full(len(raw), np.nan, dtype=float)
    grouped = {
        tuple(map(int, key)): group.sort_values("mjd")
        for key, group in archive.groupby(["fieldid", "rcidin", "filterID"])
    }
    for out_index, (_, row) in enumerate(raw.iterrows()):
        key = (int(row["fieldid"]), int(row["rcidin"]), int(row["filterID"]))
        candidates = grouped.get(key)
        if candidates is None or candidates.empty:
            continue
        delta = np.abs(candidates["mjd"].to_numpy(dtype=float) - float(row["mjd"]))
        nearest = int(np.argmin(delta))
        if delta[nearest] <= float(max_delta_days):
            values[out_index] = float(candidates.iloc[nearest]["psf_fwhm_arcsec"])
    return values


def enrich_target_seeing(object_id, *, output_dir=None):
    catalog = pd.read_parquet(resolve_qvc_data_path(f"{S82_DIR}/Catalog.parquet"))
    matches = catalog[catalog["objectId"].astype(str) == str(object_id)]
    if len(matches) != 1:
        raise ValueError(f"Expected one Catalog row for objectId={object_id!r}; found {len(matches)}.")
    target = matches.iloc[0]
    ra, dec = float(target["RA"]), float(target["DEC"])
    ps1objid = int(target["ps1objID"])

    sdss_raw = pd.read_parquet(resolve_qvc_data_path(f"{S82_DIR}/dr16s82_sdssLCRaw.parquet"))
    ps1_raw = pd.read_parquet(resolve_qvc_data_path(f"{S82_DIR}/dr16s82_ps1LCRaw.parquet"))
    ztf_raw = pd.read_parquet(resolve_qvc_data_path(f"{S82_DIR}/dr16s82_ZuberLCRaw.parquet"))
    sdss_target = sdss_raw[sdss_raw["objectId"].astype(str) == str(object_id)].copy()
    ps1_target = ps1_raw[ps1_raw["ps1objID"] == ps1objid].copy()
    ztf_target = ztf_raw[ztf_raw["ps1objID"] == ps1objid].copy()

    ps1_archive = query_ps1_detection_seeing(ps1objid)
    ps1_sidecar = ps1_target[["detectID"]].merge(
        ps1_archive, on="detectID", how="left", validate="one_to_one"
    )

    ztf_archive = query_ztf_image_seeing(ra, dec)
    ztf_sidecar = ztf_target[
        ["ps1objID", "mjd", "fieldid", "rcidin", "filterID"]
    ].copy()
    ztf_sidecar["psf_fwhm_arcsec"] = _nearest_ztf_seeing(ztf_target, ztf_archive)

    sdss_archive = query_sdss_field_seeing(ra, dec)
    sdss_sidecar = sdss_target[["objectId", "mjd", "filterID"]].copy()
    sdss_values = np.full(len(sdss_sidecar), np.nan, dtype=float)
    if not sdss_archive.empty:
        by_day = sdss_archive.set_index(sdss_archive["mjd"].astype(int))
        for index, row in enumerate(sdss_sidecar.itertuples(index=False)):
            day = int(np.floor(float(row.mjd)))
            if day in by_day.index and 0 <= int(row.filterID) < len(SDSS_BANDS):
                match = by_day.loc[day]
                if isinstance(match, pd.DataFrame):
                    match = match.iloc[0]
                sdss_values[index] = float(match[f"psfWidth_{SDSS_BANDS[int(row.filterID)]}"])
    sdss_sidecar["psf_fwhm_arcsec"] = sdss_values

    if output_dir is None:
        output_dir = Path(resolve_qvc_data_path(S82_DIR))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "sdss": (output_dir / "dr16s82_sdssSeeing.parquet", sdss_sidecar),
        "ps1": (output_dir / "dr16s82_ps1Seeing.parquet", ps1_sidecar),
        "ztf": (output_dir / "dr16s82_ztfSeeing.parquet", ztf_sidecar),
    }
    summary = {}
    for survey, (path, table) in outputs.items():
        table.to_parquet(path, index=False)
        matched = int(np.isfinite(table["psf_fwhm_arcsec"]).sum())
        summary[survey] = {"path": str(path), "matched": matched, "total": len(table)}
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object_id", help="QVC Catalog objectId to enrich.")
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args(argv)
    summary = enrich_target_seeing(args.object_id, output_dir=args.output_dir)
    for survey, values in summary.items():
        print(
            f"{survey}: matched {values['matched']}/{values['total']} epochs; "
            f"wrote {values['path']}"
        )


if __name__ == "__main__":
    main()
