import os
import numpy as np
import pandas as pd
import h5py
import warnings
from pathlib import Path
from scipy.stats import median_abs_deviation
from astropy.table import Table, join
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from tqdm import tqdm
from collections import OrderedDict
from qvc.hubble.hubble_utils import resolve_qvc_data_path


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
#bands = ['g', 'r', 'i']
SURVEY_NAMES = ("sdss", "ps1", "ztf")
SEEING_SIDECARS = {
    "sdss": "data/S82/dr16s82_sdssSeeing.parquet",
    "ps1": "data/S82/dr16s82_ps1Seeing.parquet",
    "ztf": "data/S82/dr16s82_ztfSeeing.parquet",
}
STONE_FITS_RELATIVE_PATH = "data/Stone2021/TotalDat.fits"
STONE_S82_MAX_SEP_ARCSEC = 1.0
MACLEOD_DIR_RELATIVE_PATH = "data/MacLeod2010"
MACLEOD_BANDS = ("u", "g", "r", "i")
MACLEOD_S82_MAX_SEP_ARCSEC = 1.0
MACLEOD_COLUMNS = [
    "SDR5ID",
    "ra",
    "dec",
    "redshift",
    "M_i",
    "mass_BH_Msun",
    "chi2_pdf",
    "log10_tau_days",
    "log10_sigma_mag_sqrt_yr",
    "log10_tau_lim_lo",
    "log10_tau_lim_hi",
    "log10_sig_lim_lo",
    "log10_sig_lim_hi",
    "edge_flag",
    "Plike",
    "Pnoise",
    "Pinf",
    "mu",
    "npts",
]


def rolling_photometric_outlier_mask(
    times,
    mags,
    magerrs,
    *,
    half_window_days=30.0,
    min_neighbors=8,
    threshold=3.0,
):
    """Return a mask for isolated photometric outliers in one band."""
    times = np.asarray(times, dtype=float)
    mags = np.asarray(mags, dtype=float)
    magerrs = np.asarray(magerrs, dtype=float)
    if not (times.shape == mags.shape == magerrs.shape):
        raise ValueError("times, mags, and magerrs must have matching shapes")
    if half_window_days <= 0:
        raise ValueError("half_window_days must be positive")
    if min_neighbors < 1:
        raise ValueError("min_neighbors must be at least 1")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    rejected = np.zeros(times.shape, dtype=bool)
    for i in range(times.size):
        neighbors = (
            (np.arange(times.size) != i)
            & np.isfinite(times)
            & np.isfinite(mags)
            & np.isfinite(magerrs)
            & (magerrs > 0)
            & (np.abs(times - times[i]) <= half_window_days)
        )
        if np.count_nonzero(neighbors) < min_neighbors:
            continue
        if not (np.isfinite(times[i]) and np.isfinite(mags[i]) and magerrs[i] > 0):
            continue

        local = mags[neighbors]
        local_median = float(np.median(local))
        local_sigma = 1.4826 * float(median_abs_deviation(local))
        total_sigma = float(np.hypot(magerrs[i], local_sigma))
        if total_sigma > 0 and abs(mags[i] - local_median) > threshold * total_sigma:
            rejected[i] = True
    return rejected


def inverse_variance_weighted_mean(mags, magerrs):
    """Return the inverse-variance mean and its formal measurement error."""
    mags = np.asarray(mags, dtype=float)
    magerrs = np.asarray(magerrs, dtype=float)
    if mags.shape != magerrs.shape:
        raise ValueError("mags and magerrs must have matching shapes")
    valid = np.isfinite(mags) & np.isfinite(magerrs) & (magerrs > 0)
    if not np.any(valid):
        return np.nan, np.nan
    weights = np.square(1.0 / magerrs[valid])
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        return np.nan, np.nan
    mean = float(np.sum(weights * mags[valid]) / weight_sum)
    return mean, float(np.sqrt(1.0 / weight_sum))


def resolve_stone_s82_matches(
    stone_fits_path=None,
    s82_catalog_path=None,
    max_sep_arcsec=STONE_S82_MAX_SEP_ARCSEC,
):
    """Resolve Stone rows to S82 object IDs via nearest-neighbor sky matching."""
    if stone_fits_path is None:
        stone_fits_path = resolve_qvc_data_path(STONE_FITS_RELATIVE_PATH)
    else:
        stone_fits_path = str(stone_fits_path)

    if s82_catalog_path is None:
        s82_catalog_path = resolve_qvc_data_path("data/S82/Catalog.parquet")
    else:
        s82_catalog_path = str(s82_catalog_path)

    with fits.open(stone_fits_path) as hdul:
        stone_data = hdul[1].data
        stone_df = pd.DataFrame(
            {
                "stone_row_index": np.arange(len(stone_data), dtype=int),
                "stone_DBID": np.asarray(stone_data["DBID"]).astype("int64"),
                "stone_RA": np.asarray(stone_data["RA"]).astype("float64"),
                "stone_DEC": np.asarray(stone_data["DEC"]).astype("float64"),
            }
        )

    cat = pd.read_parquet(s82_catalog_path).reset_index()
    cat_lookup = cat.loc[:, ["objectId", "RA", "DEC"]].copy()
    cat_lookup["objectId"] = cat_lookup["objectId"].astype(str)

    stone_coords = SkyCoord(
        ra=stone_df["stone_RA"].to_numpy() * u.deg,
        dec=stone_df["stone_DEC"].to_numpy() * u.deg,
    )
    cat_coords = SkyCoord(
        ra=cat_lookup["RA"].to_numpy() * u.deg,
        dec=cat_lookup["DEC"].to_numpy() * u.deg,
    )

    idx, d2d, _ = stone_coords.match_to_catalog_sky(cat_coords)
    sep_arcsec = d2d.to(u.arcsec).value
    match_mask = d2d < (max_sep_arcsec * u.arcsec)

    matched = stone_df.loc[match_mask].copy()
    matched["object_id"] = cat_lookup["objectId"].to_numpy()[idx][match_mask]
    matched["match_sep_arcsec"] = sep_arcsec[match_mask]

    unmatched = stone_df.loc[~match_mask].copy()
    unmatched["match_sep_arcsec"] = sep_arcsec[~match_mask]
    if not unmatched.empty:
        details = ", ".join(
            f"{int(row.stone_DBID)} ({row.match_sep_arcsec:.3f} arcsec)"
            for row in unmatched.itertuples(index=False)
        )
        warnings.warn(
            f"Skipping {len(unmatched)} Stone rows without an S82 match within "
            f"{max_sep_arcsec:.3f} arcsec: {details}",
            stacklevel=2,
        )

    return matched.reset_index(drop=True), unmatched.reset_index(drop=True)


def resolve_stone_object_ids(
    stone_fits_path=None,
    s82_catalog_path=None,
    max_sep_arcsec=STONE_S82_MAX_SEP_ARCSEC,
):
    """Return matched S82 object IDs for the Stone sample in Stone row order."""
    matched, _ = resolve_stone_s82_matches(
        stone_fits_path=stone_fits_path,
        s82_catalog_path=s82_catalog_path,
        max_sep_arcsec=max_sep_arcsec,
    )
    return matched["object_id"].astype(str).tolist()


def read_macleod_band(band, macleod_dir=None):
    """Read one MacLeod DRW catalog band and apply the notebook's tau-error validity cut."""
    if macleod_dir is None:
        macleod_dir = resolve_qvc_data_path(MACLEOD_DIR_RELATIVE_PATH)
    path = os.path.join(str(macleod_dir), f"s82drw_{band}.dat")
    df = pd.read_csv(path, comment="#", sep=r"\s+", header=None)
    df.columns = MACLEOD_COLUMNS

    x_tau = pd.to_numeric(df["log10_tau_days"], errors="coerce").to_numpy(dtype=float)
    tau_lo = pd.to_numeric(df["log10_tau_lim_lo"], errors="coerce").to_numpy(dtype=float)
    tau_hi = pd.to_numeric(df["log10_tau_lim_hi"], errors="coerce").to_numpy(dtype=float)
    edge_flag = pd.to_numeric(df["edge_flag"], errors="coerce").to_numpy(dtype=float)
    xerr_low_tau = x_tau - tau_lo
    xerr_high_tau = tau_hi - x_tau
    valid_xerr_tau = (
        np.isfinite(xerr_low_tau)
        & np.isfinite(xerr_high_tau)
        & (xerr_low_tau >= 0.0)
        & (xerr_high_tau >= 0.0)
        & (tau_lo < tau_hi)
        & (tau_lo < x_tau)
        & (x_tau < tau_hi)
        & (edge_flag == 0)
    )
    valid_large_tau_hi = np.isfinite(xerr_high_tau) & np.isfinite(x_tau)
    df = df.loc[valid_xerr_tau & valid_large_tau_hi].copy()
    df["band"] = band
    return df


def resolve_macleod_s82_matches(
    bands=MACLEOD_BANDS,
    macleod_dir=None,
    s82_catalog_path=None,
    max_sep_arcsec=MACLEOD_S82_MAX_SEP_ARCSEC,
):
    """Resolve MacLeod rows to S82 object IDs via nearest-neighbor sky matching."""
    if macleod_dir is None:
        macleod_dir = resolve_qvc_data_path(MACLEOD_DIR_RELATIVE_PATH)
    else:
        macleod_dir = str(macleod_dir)

    if s82_catalog_path is None:
        s82_catalog_path = resolve_qvc_data_path("data/S82/Catalog.parquet")
    else:
        s82_catalog_path = str(s82_catalog_path)

    cat = pd.read_parquet(s82_catalog_path).reset_index()
    cat_lookup = cat.loc[:, ["objectId", "RA", "DEC"]].copy()
    cat_lookup["objectId"] = cat_lookup["objectId"].astype(str)
    cat_coords = SkyCoord(
        ra=cat_lookup["RA"].to_numpy() * u.deg,
        dec=cat_lookup["DEC"].to_numpy() * u.deg,
    )

    matched_frames = []
    unmatched_frames = []
    for band in bands:
        band_df = read_macleod_band(band, macleod_dir=macleod_dir)
        query = band_df.loc[:, ["SDR5ID", "ra", "dec"]].copy()
        coords = SkyCoord(ra=query["ra"].to_numpy() * u.deg, dec=query["dec"].to_numpy() * u.deg)
        idx, d2d, _ = coords.match_to_catalog_sky(cat_coords)
        sep_arcsec = d2d.to(u.arcsec).value
        match_mask = d2d < (max_sep_arcsec * u.arcsec)

        matched = query.loc[match_mask].copy()
        matched["band"] = band
        matched["object_id"] = cat_lookup["objectId"].to_numpy()[idx][match_mask]
        matched["match_sep_arcsec"] = sep_arcsec[match_mask]
        matched_frames.append(matched.reset_index(drop=True))

        unmatched = query.loc[~match_mask].copy()
        unmatched["band"] = band
        unmatched["match_sep_arcsec"] = sep_arcsec[~match_mask]
        unmatched_frames.append(unmatched.reset_index(drop=True))

    matched_all = pd.concat(matched_frames, ignore_index=True) if matched_frames else pd.DataFrame()
    unmatched_all = pd.concat(unmatched_frames, ignore_index=True) if unmatched_frames else pd.DataFrame()
    if not unmatched_all.empty:
        details = ", ".join(
            f"{row.band}:{int(row.SDR5ID)} ({row.match_sep_arcsec:.3f} arcsec)"
            for row in unmatched_all.itertuples(index=False)
        )
        warnings.warn(
            f"Skipping {len(unmatched_all)} MacLeod rows without an S82 match within "
            f"{max_sep_arcsec:.3f} arcsec: {details}",
            stacklevel=2,
        )
    return matched_all, unmatched_all


def resolve_macleod_object_ids(
    bands=MACLEOD_BANDS,
    macleod_dir=None,
    s82_catalog_path=None,
    max_sep_arcsec=MACLEOD_S82_MAX_SEP_ARCSEC,
):
    """Return matched S82 object IDs for the MacLeod sample, de-duplicated in first-seen order."""
    matched, _ = resolve_macleod_s82_matches(
        bands=bands,
        macleod_dir=macleod_dir,
        s82_catalog_path=s82_catalog_path,
        max_sep_arcsec=max_sep_arcsec,
    )
    if matched.empty:
        return []
    return matched["object_id"].astype(str).drop_duplicates(keep="first").tolist()

def cut_light_curve_restframe_window(lc_list, n_days=1800, same_length=False):
    """
    Applies a rest-frame time cut to a list of light curve objects.
    - Cuts to rest-frame window [0, n_days]
    - Optional: Keeps object only if final rest-frame time span >= n_days - min_n_days

    Parameters:
    - lc: dict with keys 'mags', 'magerrs', 'times' (per band), and 'z'
    - n_days: desired rest-frame duration
    - delta_n_days: tolerance margin

    Returns:
    - new_lc: same structure with data cut in time, or None if rejected
    """
    bands = ['u', 'g', 'r', 'i', 'z']
    max_cut_span = n_days
    new_lc_list = []

    for lc in lc_list:
        z = lc['z']
        obj_id = lc.get('object_id')

        # --- Get object-specific t0 from all bands ---
        t0_candidates = [
            np.min(lc['times'][band])
            for band in bands
            if band in lc['times'] and len(lc['times'][band]) > 0
        ]

        if not t0_candidates:
            print(f"Skipping {obj_id}: no valid times in any band")
            continue

        t0 = np.min(t0_candidates)
        new_mags = {}
        new_magerrs = {}
        new_times = {}
        new_surveys = {}
        cut_rest_times = []

        for band in bands:
            t_obs = np.asarray(lc['times'].get(band, []))
            m_band = np.asarray(lc['mags'].get(band, []))
            me_band = np.asarray(lc['magerrs'].get(band, []))
            survey_band = np.asarray(lc.get('surveys', {}).get(band, []))

            if len(t_obs) == 0:
                new_times[band] = []
                new_mags[band] = []
                new_magerrs[band] = []
                new_surveys[band] = []
                continue

            # Convert to rest-frame
            t_rest = (t_obs - t0) / (1 + z)
            mask = (t_rest >= 0) & (t_rest <= max_cut_span)

            new_times[band] = t_obs[mask]       # keep in observer frame
            new_mags[band] = m_band[mask]
            new_magerrs[band] = me_band[mask]
            new_surveys[band] = survey_band[mask] if survey_band.size else []
            cut_rest_times.append(t_rest[mask])

        # --- Final rest-frame span check ---
        cut_rest_flat = np.concatenate(cut_rest_times) if cut_rest_times else np.array([])
        if len(cut_rest_flat) == 0:
            print(f"Skipping {obj_id}: no data in rest-frame cut window")
            continue

        span_rf = np.max(cut_rest_flat) - np.min(cut_rest_flat)

        min_n_days = n_days * 0.9
        if same_length and span_rf < min_n_days:
            print(f"Skipping {obj_id}: span = {span_rf:.1f} < {min_n_days:.1f} rest-frame days")
            continue

        print(f"✅ Keeping {obj_id}: rest-frame span = {span_rf:.1f} days")
        
        mags_means = [np.nanmean(new_mags[band]) for band in new_mags.keys()]
        mags_stds = [np.nanstd(new_mags[band]) for band in new_mags.keys()]
        new_lc = lc.copy()
        new_lc.update({
            'object_id': obj_id,
            'z': z,
            'span_rf': span_rf,
            'mags': new_mags,
            'magerrs': new_magerrs,
            'surveys': new_surveys,
            'times': new_times,  # still in observer frame
            'mags_mean': mags_means,
            'mags_std': mags_stds,
        })
        new_lc_list.append(new_lc)
        
    return new_lc_list


def concat_light_curves(
    filter_object_ids=None,
    progress_bar=False,
    skip=None,
    N=None,
    *,
    load_seeing=True,
):
    """Vectorized light-curve concatenation.

    Missing seeing sidecars are optional. Set ``load_seeing=False`` for callers,
    such as spectral fitting, that do not use epoch-level seeing information.
    """
    print(
        f"[DEBUG] Loading concat_light_curves with filter_object_ids={filter_object_ids}, skip={skip}, N={N}"
    )

    cat = pd.read_parquet(resolve_qvc_data_path("data/S82/Catalog.parquet")).set_index("idx")
    sdss = pd.read_parquet(resolve_qvc_data_path("data/S82/dr16s82_sdssLCRaw.parquet"))
    sdss = sdss[sdss.mjd.notna()].copy()
    ps1 = pd.read_parquet(resolve_qvc_data_path("data/S82/dr16s82_ps1LCRaw.parquet"))
    ztf = pd.read_parquet(resolve_qvc_data_path("data/S82/dr16s82_ZuberLCRaw.parquet"))

    def _attach_seeing_sidecar(frame, survey, keys):
        try:
            sidecar_path = Path(resolve_qvc_data_path(SEEING_SIDECARS[survey]))
        except FileNotFoundError:
            return frame
        if not sidecar_path.exists():
            return frame
        sidecar = pd.read_parquet(sidecar_path)
        required = [*keys, "psf_fwhm_arcsec"]
        missing = [name for name in required if name not in sidecar.columns]
        if missing:
            raise ValueError(f"Seeing sidecar {sidecar_path} is missing columns {missing}.")
        sidecar = sidecar[required].drop_duplicates(keys, keep="last")
        return frame.drop(columns=["psf_fwhm_arcsec"], errors="ignore").merge(
            sidecar, on=keys, how="left", validate="many_to_one"
        )

    if load_seeing:
        sdss = _attach_seeing_sidecar(sdss, "sdss", ["objectId", "mjd", "filterID"])
        ps1 = _attach_seeing_sidecar(ps1, "ps1", ["detectID"])
        ztf = _attach_seeing_sidecar(
            ztf,
            "ztf",
            ["ps1objID", "mjd", "fieldid", "rcidin", "filterID"],
        )

    if filter_object_ids is not None:
        match_object_ids = set(sdss.objectId) & set(filter_object_ids)
    else:
        match_object_ids = set(sdss.objectId)

    matching_indices = cat[cat.objectId.isin(match_object_ids)].index
    cat = cat.loc[matching_indices]
    print(f"Found {len(cat)} matching objects in concat_light_curves_jax")

    if skip is not None:
        cat = cat.iloc[skip:]
    if N is not None:
        cat = cat.iloc[:N]

    # Preserve the same "first occurrence wins" duplicate-object semantics as concat_light_curves.
    cat = cat.loc[~cat.objectId.duplicated(keep="first")].copy()
    object_ids = cat["objectId"].tolist()
    if len(object_ids) == 0:
        print("Found 0 objects in concat_light_curves_jax")
        return []

    valid_filter_ids = [filters[b] for b in bands]
    filter_to_band = {filters[b]: b for b in bands}

    def _seeing_column(df, candidates):
        for name in candidates:
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        return np.full(len(df), np.nan, dtype=float)

    def _offset_from_band(df):
        return np.asarray(
            np.select(
                [
                    df["band"].eq("g"),
                    df["band"].eq("r"),
                    df["band"].eq("i"),
                    df["band"].eq("z"),
                ],
                [
                    df["sdss_g_qg"] - df["ps1_g_qg"],
                    df["sdss_r_qg"] - df["ps1_r_qg"],
                    df["sdss_i_qg"] - df["ps1_i_qg"],
                    df["sdss_z_qg"] - df["ps1_z_qg"],
                ],
                default=0.0,
            ),
            dtype=float,
        )

    def _cadence_stats(all_times):
        all_times = np.asarray(all_times, dtype=float)
        all_times = all_times[np.isfinite(all_times)]
        all_times = np.sort(all_times)

        tol_min = 30
        tol = tol_min / 1440.0
        if len(all_times) > 1:
            keep = np.ones_like(all_times, dtype=bool)
            keep[1:] = np.diff(all_times) > tol
            all_times = all_times[keep]

        gap_days = 30
        if len(all_times) > 1:
            dt = np.diff(all_times)
            dt = dt[(dt > 0) & (dt < gap_days)]
        else:
            dt = np.array([])

        if len(dt) > 0:
            cadence = float(np.median(dt))
            cadence_err = float(np.percentile(dt, 84) - np.percentile(dt, 16))
        else:
            cadence = np.nan
            cadence_err = np.nan
        return cadence, cadence_err

    cat_ps1 = cat.loc[pd.notna(cat["ps1objID"])].copy()
    if not cat_ps1.empty:
        cat_ps1 = cat_ps1[
            [
                "objectId",
                "ps1objID",
                "sdss_g_qg",
                "sdss_r_qg",
                "sdss_i_qg",
                "sdss_z_qg",
                "ps1_g_qg",
                "ps1_r_qg",
                "ps1_i_qg",
                "ps1_z_qg",
            ]
        ].rename(columns={"objectId": "object_id"})
        cat_ps1["ps1objID"] = cat_ps1["ps1objID"].astype(np.int64)
        ps1_ids = cat_ps1["ps1objID"].unique()
    else:
        cat_ps1 = pd.DataFrame(
            columns=[
                "object_id",
                "ps1objID",
                "sdss_g_qg",
                "sdss_r_qg",
                "sdss_i_qg",
                "sdss_z_qg",
                "ps1_g_qg",
                "ps1_r_qg",
                "ps1_i_qg",
                "ps1_z_qg",
            ]
        )
        ps1_ids = np.array([], dtype=np.int64)

    sdss_subset = sdss[
        sdss["objectId"].isin(object_ids) & sdss["filterID"].isin(valid_filter_ids)
    ].copy()
    sdss_subset["band"] = sdss_subset["filterID"].map(filter_to_band)
    sdss_obs = pd.DataFrame(
        {
            "object_id": sdss_subset["objectId"].to_numpy(),
            "band": sdss_subset["band"].to_numpy(),
            "band_idx": sdss_subset["filterID"].to_numpy(dtype=np.int32),
            "survey": "sdss",
            "time": sdss_subset["mjd"].to_numpy(dtype=float),
            "mag": sdss_subset["psMag"].to_numpy(dtype=float),
            "magerr": sdss_subset["psMagErr_p3"].to_numpy(dtype=float),
            "psf_fwhm_arcsec": _seeing_column(
                sdss_subset, ("psf_fwhm_arcsec", "psfWidth", "seeing")
            ),
        }
    )

    if len(ps1_ids) > 0:
        ps1_subset = ps1[
            ps1["ps1objID"].isin(ps1_ids) & ps1["filterID"].isin(valid_filter_ids)
        ].copy()
        ps1_merge = ps1_subset.merge(cat_ps1, on="ps1objID", how="inner")
        ps1_merge["band"] = ps1_merge["filterID"].map(filter_to_band)
        ps1_offset = _offset_from_band(ps1_merge)
        ps1_obs = pd.DataFrame(
            {
                "object_id": ps1_merge["object_id"].to_numpy(),
                "band": ps1_merge["band"].to_numpy(),
                "band_idx": ps1_merge["filterID"].to_numpy(dtype=np.int32),
                "survey": "ps1",
                "time": ps1_merge["obsTime"].to_numpy(dtype=float),
                "mag": ps1_merge["psfMag"].to_numpy(dtype=float) + ps1_offset,
                "magerr": ps1_merge["psfMagErr_p3"].to_numpy(dtype=float),
                "psf_fwhm_arcsec": _seeing_column(
                    ps1_merge,
                    ("psf_fwhm_arcsec", "psfMajorFWHM", "seeing"),
                ),
            }
        )

        ztf_subset = ztf[
            ztf["ps1objID"].isin(ps1_ids) & ztf["filterID"].isin(valid_filter_ids)
        ].copy()
        ztf_merge = ztf_subset.merge(cat_ps1, on="ps1objID", how="inner")
        ztf_merge["band"] = ztf_merge["filterID"].map(filter_to_band)
        ztf_offset = _offset_from_band(ztf_merge)
        ztf_obs = pd.DataFrame(
            {
                "object_id": ztf_merge["object_id"].to_numpy(),
                "band": ztf_merge["band"].to_numpy(),
                "band_idx": ztf_merge["filterID"].to_numpy(dtype=np.int32),
                "survey": "ztf",
                "time": ztf_merge["mjd"].to_numpy(dtype=float),
                "mag": ztf_merge["mag"].to_numpy(dtype=float) + ztf_offset,
                "magerr": ztf_merge["magerr_p3"].to_numpy(dtype=float),
                "psf_fwhm_arcsec": _seeing_column(
                    ztf_merge, ("psf_fwhm_arcsec", "seeing", "fwhm")
                ),
            }
        )
    else:
        ps1_obs = pd.DataFrame(
            columns=["object_id", "band", "band_idx", "survey", "time", "mag", "magerr", "psf_fwhm_arcsec"]
        )
        ztf_obs = pd.DataFrame(
            columns=["object_id", "band", "band_idx", "survey", "time", "mag", "magerr", "psf_fwhm_arcsec"]
        )

    obs = pd.concat([sdss_obs, ps1_obs, ztf_obs], ignore_index=True)
    if obs.empty:
        print("Found 0 objects in concat_light_curves_jax")
        return []

    finite_mask = (
        np.isfinite(obs["time"])
        & np.isfinite(obs["mag"])
        & np.isfinite(obs["magerr"])
        & (obs["magerr"] > 0)
    )
    obs = obs.loc[finite_mask].copy()
    obs = obs.sort_values(["object_id", "band_idx", "time"], kind="mergesort")

    rejected = pd.Series(False, index=obs.index)
    for _key, group in obs.groupby(["object_id", "band"], sort=False):
        rejected.loc[group.index] = rolling_photometric_outlier_mask(
            group["time"].to_numpy(dtype=float),
            group["mag"].to_numpy(dtype=float),
            group["magerr"].to_numpy(dtype=float),
        )
    obs = obs.loc[~rejected].copy()
    obs = obs.sort_values(["object_id", "band_idx", "time"], kind="mergesort")

    by_obj_band = obs.groupby(["object_id", "band"], sort=False)
    times_by_obj_band = by_obj_band["time"].apply(lambda x: x.to_numpy(dtype=float)).to_dict()
    mags_by_obj_band = by_obj_band["mag"].apply(lambda x: x.to_numpy(dtype=float)).to_dict()
    magerrs_by_obj_band = by_obj_band["magerr"].apply(lambda x: x.to_numpy(dtype=float)).to_dict()
    surveys_by_obj_band = by_obj_band["survey"].apply(lambda x: x.astype(str).to_numpy()).to_dict()
    seeing_by_obj_band = by_obj_band["psf_fwhm_arcsec"].apply(
        lambda x: x.to_numpy(dtype=float)
    ).to_dict()
    weighted_photometry_by_obj_band = {
        key: inverse_variance_weighted_mean(
            group["mag"].to_numpy(dtype=float),
            group["magerr"].to_numpy(dtype=float),
        )
        for key, group in by_obj_band
    }
    number_points_by_obj = obs.groupby("object_id", sort=False).size()
    all_times_by_obj = (
        obs.groupby("object_id", sort=False)["time"]
        .apply(lambda x: np.sort(x.to_numpy(dtype=float)))
        .to_dict()
    )

    survey_times_by_obj_survey = (
        obs.groupby(["object_id", "survey"], sort=False)["time"]
        .apply(lambda values: np.sort(values.to_numpy(dtype=float)))
        .to_dict()
    )

    s82_objs = []
    iterator = tqdm(
        object_ids,
        total=len(object_ids),
        desc="Packaging quasars",
        disable=(not progress_bar),
    )
    for object_id in iterator:
        number_points = int(number_points_by_obj.get(object_id, 0))
        if number_points == 0:
            print(f"No data available for object {object_id}, skipping.", flush=True)
            continue

        times = {}
        mags = {}
        magerrs = {}
        surveys = {}
        psf_fwhm_arcsec = {}
        for band in bands:
            key = (object_id, band)
            times[band] = np.asarray(times_by_obj_band.get(key, np.array([])), dtype=float)
            mags[band] = np.asarray(mags_by_obj_band.get(key, np.array([])), dtype=float)
            magerrs[band] = np.asarray(magerrs_by_obj_band.get(key, np.array([])), dtype=float)
            surveys[band] = np.asarray(surveys_by_obj_band.get(key, np.array([])), dtype=str)
            psf_fwhm_arcsec[band] = np.asarray(
                seeing_by_obj_band.get(key, np.array([])), dtype=float
            )

        mags_means = []
        mags_mean_errs = []
        for band in bands:
            mean, mean_err = weighted_photometry_by_obj_band.get(
                (object_id, band), (np.nan, np.nan)
            )
            mags_means.append(float(mean))
            mags_mean_errs.append(float(mean_err))
        survey_times = {
            survey: np.asarray(
                survey_times_by_obj_survey.get((object_id, survey), np.array([])),
                dtype=float,
            )
            for survey in SURVEY_NAMES
        }
        cadence, cadence_err = _cadence_stats(all_times_by_obj.get(object_id, np.array([])))

        s82_objs.append(
            {
                "object_id": object_id,
                "times": times,
                "surveys": surveys,
                "psf_fwhm_arcsec": psf_fwhm_arcsec,
                "survey_times": survey_times,
                "survey_names": SURVEY_NAMES,
                "mags": mags,
                "mags_mean": mags_means,
                "mags_mean_err": mags_mean_errs,
                "magerrs": magerrs,
                "cadence": cadence,
                "cadence_err": cadence_err,
                "number_points": number_points,
            }
        )

    print(f"Found {len(s82_objs)} objects in concat_light_curves_jax")
    return s82_objs

def load_nearby_lcs(name):
    # Load light curves from a CSV file
    df = pd.read_csv(f'data/nearby_lcs/{name}.csv')

    bands = ['g', 'r', 'i']
    mags = {band: [] for band in bands}
    magerrs = {band: [] for band in bands}
    times = {band: [] for band in bands}
    surveys = {band: [] for band in bands}

    for band in bands:
        filter_name = f'z{band}'
        mask = df['filter'] == filter_name
        times[band] = df.loc[mask, 'mjd'].values
        mags[band] = df.loc[mask, 'mag'].values
        magerrs[band] = df.loc[mask, 'magerr'].values
        surveys[band] = np.full(times[band].shape, "ztf", dtype=str)

    for band in bands:
        print(f"{band}: {len(mags[band])} mag points")
    return [{
        'z': 0,
        'LOGLBOL': 1,
        'object_id': name,
        'times': times,
        'surveys': surveys,
        'survey_names': SURVEY_NAMES,
        'mags': mags,
        'magerrs': magerrs,
    }]

def load_stone_lcs(filter_object_ids=[], skip=None, N=None):
    # Load Stone et al. (2022) data
    fits_file = resolve_qvc_data_path(STONE_FITS_RELATIVE_PATH)
    with fits.open(fits_file) as hdul:
        data = hdul[1].data
    bands = ['g', 'r', 'i']
    fields = [
        'MAG', 'MAG_ERR', 'MJD',
        'log_SIGMA', 
        'log_TAU_REST',
    ]
    
    stone_lcs = OrderedDict()

    for i in range(len(data)):
        if skip is not None and i < skip:
            continue
        if N is not None and i >= skip + N:
            break
        dbid = data['DBID'][i]
        stone_lcs[dbid] = {
            'stone_DBID': dbid,
            'z': data['Z'][i],
            'stone_Z': data['Z'][i],
            'stone_RA': data['RA'][i],
            'stone_DEC': data['DEC'][i],
            'stone_LOG_M_BH': data['LOG_M_BH'][i],
            'stone_LOG_M_BH_ERR': data['LOG_M_BH_ERR'][i],
            'stone_LOG_LBOL': data['LOG_LBOL'][i],
            'stone_LOG_LBOL_ERR': data['LOG_LBOL_ERR'][i],        
            'mags': {},
            'magerrs': {},
            'times': {},
        }
        for band in bands:
            # Mask NaNs for MJD, MAG, and MAG_ERR fields
            mjd = data[f'MJD_{band}'][i]
            mag = data[f'MAG_{band}'][i]
            mag_err = data[f'MAG_ERR_{band}'][i]
            mask = ~np.isnan(mjd) & ~np.isnan(mag) & ~np.isnan(mag_err)

            stone_lcs[dbid]['mags'][band] = mag[mask]
            stone_lcs[dbid]['magerrs'][band] = mag_err[mask]
            stone_lcs[dbid]['times'][band] = mjd[mask]

            stone_lcs[dbid] |= {
                # f'MJD_{band}': mjd[mask],
                # f'MAG_{band}': mag[mask],
                # f'MAG_{band}_ERR': mag_err[mask],

                f'stone_log_SIGMA_{band}': data[f'log_SIGMA_{band}'][i],
                f'stone_log_SIGMA_{band}_ERR_L': data[f'log_SIGMA_{band}_ERR_L'][i],
                f'stone_log_SIGMA_{band}_ERR_U': data[f'log_SIGMA_{band}_ERR_U'][i],
                f'stone_log_TAU_REST_{band}': data[f'log_TAU_REST_{band}'][i],
                f'stone_log_TAU_REST_{band}_ERR_L': data[f'log_TAU_REST_{band}_ERR_L'][i],
                f'stone_log_TAU_REST_{band}_ERR_U': data[f'log_TAU_REST_{band}_ERR_U'][i],
                f'stone_log_SIGMA_{band}_ERR': (data[f'log_SIGMA_{band}_ERR_L'][i] + data[f'log_SIGMA_{band}_ERR_U'][i]) / 2,
                f'stone_log_TAU_REST_{band}_ERR': (data[f'log_TAU_REST_{band}_ERR_L'][i] + data[f'log_TAU_REST_{band}_ERR_U'][i]) / 2,
            }

    matched_rows, _ = resolve_stone_s82_matches(
        stone_fits_path=fits_file,
        max_sep_arcsec=STONE_S82_MAX_SEP_ARCSEC,
    )
    matched_object_ids = {
        int(row.stone_DBID): str(row.object_id) for row in matched_rows.itertuples(index=False)
    }
    for stone_dbid in list(stone_lcs):
        object_id = matched_object_ids.get(int(stone_dbid))
        if object_id is None:
            del stone_lcs[stone_dbid]
            continue
        stone_lcs[stone_dbid]['object_id'] = object_id

    if filter_object_ids:
        stone_lcs = {k: v for k, v in stone_lcs.items() if v.get('object_id') in filter_object_ids}
        print(f"After filtering, {len(stone_lcs)} Stone objects remain.")
    return list(stone_lcs.values())

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
    redshift_tolerance = 1.0e-3
    _ = progress_bar
    if not s82_objs:
        return s82_objs

    def _read_catalog_with_redshift():
        cat_path = resolve_qvc_data_path("data/S82/Catalog.parquet")
        read_attempts = (
            ["objectId", "RA", "DEC", "z"],
            ["objectId", "RA", "DEC", "Z_SYS"],
        )
        last_exc = None
        for include_names in read_attempts:
            try:
                cat_local = Table.read(cat_path, include_names=include_names)
                if include_names[-1] == "z":
                    cat_local["catalog_z"] = cat_local["z"]
                else:
                    cat_local["catalog_z"] = cat_local["Z_SYS"]
                return cat_local
            except (KeyError, ValueError, OSError) as exc:
                last_exc = exc
        raise ValueError(
            "S82 catalog must include either a 'z' or 'Z_SYS' redshift column."
        ) from last_exc

    rows = Table(
        {
            "row_idx": np.arange(len(s82_objs), dtype=int),
            "object_id": [obj["object_id"] for obj in s82_objs],
        }
    )
    rows["object_id_key"] = np.asarray(rows["object_id"]).astype(str)

    cat = _read_catalog_with_redshift()
    cat = cat[~np.isnan(cat["RA"]) & ~np.isnan(cat["DEC"])]
    _, first_idx = np.unique(np.asarray(cat["objectId"]).astype(str), return_index=True)
    cat = cat[np.sort(first_idx)]
    cat["object_id_key"] = np.asarray(cat["objectId"]).astype(str)

    rows = join(rows, cat["object_id_key", "RA", "DEC", "catalog_z"], keys="object_id_key", join_type="left")
    ra_mask = np.ma.getmaskarray(rows["RA"]) if hasattr(rows["RA"], "mask") else np.zeros(len(rows), dtype=bool)
    dec_mask = np.ma.getmaskarray(rows["DEC"]) if hasattr(rows["DEC"], "mask") else np.zeros(len(rows), dtype=bool)
    valid_rows = ~(ra_mask | dec_mask)

    if not np.all(valid_rows):
        missing_catalog_ids = np.asarray(rows["object_id"])[~valid_rows]
        missing_ids = ", ".join(str(object_id) for object_id in missing_catalog_ids)
        raise ValueError(f"Missing S82 catalog match for object_id(s): {missing_ids}")

    dr16q = Table.read(resolve_qvc_data_path("data/dr16q_prop_May01_2024.fits"), hdu=1)
    required_dr16q_columns = (
        "RA",
        "DEC",
        "Z_SYS",
        "Z_SYS_ERR",
        "SDSS_NAME",
        "EBV",
        "LOGLBOL",
        "LOGLBOL_ERR",
        "LOGL3000",
        "LOGL3000_ERR",
        "LOGL5100",
        "LOGL5100_ERR",
        "LOGMBH",
        "LOGMBH_ERR",
        "LOGLEDD_RATIO",
        "LOGLEDD_RATIO_ERR",
        "PLATE",
        "MJD",
        "FIBERID",
        "SN_MEDIAN_ALL"
    )
    missing_dr16q_columns = [name for name in required_dr16q_columns if name not in dr16q.colnames]
    if missing_dr16q_columns:
        missing_cols = ", ".join(missing_dr16q_columns)
        raise ValueError(f"DR16Q table missing required column(s): {missing_cols}")
    fits_coords = SkyCoord(ra=np.asarray(dr16q["RA"]) * u.deg, dec=np.asarray(dr16q["DEC"]) * u.deg)

    query_rows = rows[valid_rows]
    query_coords = SkyCoord(
        ra=np.asarray(query_rows["RA"], dtype=float) * u.deg,
        dec=np.asarray(query_rows["DEC"], dtype=float) * u.deg,
    )
    nearest_idx, d2d, _ = query_coords.match_to_catalog_sky(fits_coords)
    match_mask = d2d < (1.0 * u.arcsec)

    if not np.all(match_mask):
        unmatched_dr16q_ids = np.asarray(query_rows["object_id"])[~match_mask]
        missing_ids = ", ".join(str(object_id) for object_id in unmatched_dr16q_ids)
        raise ValueError(f"Missing DR16Q match within 1 arcsec for object_id(s): {missing_ids}")

    matched_rows = query_rows[match_mask]
    matched_row_idx = np.asarray(matched_rows["row_idx"], dtype=int)
    matched_fits_idx = np.asarray(nearest_idx[match_mask], dtype=int)
    cat_z_vals = np.asarray(matched_rows["catalog_z"], dtype=float)
    dr16q_z_vals = np.asarray(dr16q["Z_SYS"][matched_fits_idx], dtype=float)
    z_err_vals = np.asarray(dr16q["Z_SYS_ERR"][matched_fits_idx], dtype=float)
    logl3000 = np.asarray(dr16q["LOGL3000"])[matched_fits_idx]
    logl3000_err = np.asarray(dr16q["LOGL3000_ERR"])[matched_fits_idx]
    loglbol = np.asarray(dr16q["LOGLBOL"])[matched_fits_idx]
    loglbol_err = np.asarray(dr16q["LOGLBOL_ERR"])[matched_fits_idx]
    invalid_cat_z = ~np.isfinite(cat_z_vals)
    if np.any(invalid_cat_z):
        missing_ids = ", ".join(str(object_id) for object_id in np.asarray(matched_rows["object_id"])[invalid_cat_z])
        raise ValueError(f"Missing S82 redshift for object_id(s): {missing_ids}")
    invalid_dr16q_z = ~np.isfinite(dr16q_z_vals)
    if np.any(invalid_dr16q_z):
        missing_ids = ", ".join(str(object_id) for object_id in np.asarray(matched_rows["object_id"])[invalid_dr16q_z])
        raise ValueError(f"Missing DR16Q Z_SYS for object_id(s): {missing_ids}")
    invalid_z_err = ~np.isfinite(z_err_vals)
    if np.any(invalid_z_err):
        missing_ids = ", ".join(str(object_id) for object_id in np.asarray(matched_rows["object_id"])[invalid_z_err])
        raise ValueError(f"Missing DR16Q Z_SYS_ERR for object_id(s): {missing_ids}")
    z_disagreement = np.abs(cat_z_vals - dr16q_z_vals) > redshift_tolerance
    if np.any(z_disagreement):
        bad_ids = ", ".join(str(object_id) for object_id in np.asarray(matched_rows["object_id"])[z_disagreement])
        raise ValueError(
            f"Redshift disagreement exceeds tolerance {redshift_tolerance:g} for object_id(s): {bad_ids}"
        )
    z_vals = dr16q_z_vals
    loglbol_corrected = np.where(z_vals < 0.7, np.log10(5.15) + logl3000, loglbol)
    loglbol_corrected_err = np.where(z_vals < 0.7, logl3000_err, loglbol_err)
    ebv_vals = np.asarray(dr16q["EBV"])[matched_fits_idx]
    sn_median_all = np.asarray(dr16q["SN_MEDIAN_ALL"])[matched_fits_idx]

    for field, values in (
        ("ra", np.asarray(matched_rows["RA"], dtype=float)),
        ("dec", np.asarray(matched_rows["DEC"], dtype=float)),
        ("z", z_vals),
        ("z_err", z_err_vals),
        ("sdss_name", np.asarray(dr16q["SDSS_NAME"])[matched_fits_idx]),
        ("ebv_wu", ebv_vals),
        ("LOGLBOL", loglbol),
        ("LOGLBOL_ERR", loglbol_err),
        ("LOGLBOL_CORRECTED", loglbol_corrected),
        ("LOGLBOL_CORRECTED_ERR", loglbol_corrected_err),
        ("LOGL5100", np.asarray(dr16q["LOGL5100"])[matched_fits_idx]),
        ("LOGL5100_ERR", np.asarray(dr16q["LOGL5100_ERR"])[matched_fits_idx]),
        ("log_mbh", np.asarray(dr16q["LOGMBH"])[matched_fits_idx]),
        ("log_mbh_err", np.asarray(dr16q["LOGMBH_ERR"])[matched_fits_idx]),
        ("log_ledd_ratio", np.asarray(dr16q["LOGLEDD_RATIO"])[matched_fits_idx]),
        ("log_ledd_ratio_err", np.asarray(dr16q["LOGLEDD_RATIO_ERR"])[matched_fits_idx]),
        ("plate", np.asarray(dr16q["PLATE"])[matched_fits_idx]),
        ("mjd", np.asarray(dr16q["MJD"])[matched_fits_idx]),
        ("fiberid", np.asarray(dr16q["FIBERID"])[matched_fits_idx]),
        ("SN_MEDIAN_ALL", sn_median_all),
    ):
        for row_idx, value in zip(matched_row_idx, values):
            s82_objs[row_idx][field] = value
    for obj in s82_objs:
        if "z" not in obj or not np.isfinite(obj["z"]):
            raise ValueError(f"populate_sdss_fields did not populate finite z for object_id={obj['object_id']}")
        if "z_err" not in obj or not np.isfinite(obj["z_err"]):
            raise ValueError(f"populate_sdss_fields did not populate finite z_err for object_id={obj['object_id']}")
    return s82_objs
