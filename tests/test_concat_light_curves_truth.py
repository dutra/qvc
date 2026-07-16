import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble.hubble_utils import resolve_qvc_data_path
from qvc.light_curve.multiband_generate_lc import (
    bands,
    concat_light_curves,
    filters,
    inverse_variance_weighted_mean,
    rolling_photometric_outlier_mask,
)


def _load_s82_inputs():
    try:
        cat = pd.read_parquet(resolve_qvc_data_path("data/S82/Catalog.parquet")).set_index("idx")
        sdss = pd.read_parquet(resolve_qvc_data_path("data/S82/dr16s82_sdssLCRaw.parquet"))
        sdss = sdss[sdss.mjd.notna()].copy()
        ps1 = pd.read_parquet(resolve_qvc_data_path("data/S82/dr16s82_ps1LCRaw.parquet"))
        ztf = pd.read_parquet(resolve_qvc_data_path("data/S82/dr16s82_ZuberLCRaw.parquet"))
    except FileNotFoundError as exc:
        pytest.skip(f"S82 data unavailable for truth tests: {exc}")
    return cat, sdss, ps1, ztf


def _find_object(objs, object_id):
    for obj in objs:
        if obj["object_id"] == object_id:
            return obj
    raise AssertionError(f"object_id {object_id} not found in function output")


def _assert_array_equalish(a, b, atol=1e-12, rtol=1e-12):
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    assert aa.shape == bb.shape
    np.testing.assert_allclose(aa, bb, atol=atol, rtol=rtol, equal_nan=True)


def _assert_mag_array_matches_truth(a, b):
    # concat_light_curves applies cross-survey offsets with small float noise at the ~1e-6 level.
    _assert_array_equalish(a, b, atol=1e-6, rtol=1e-12)


def _reconstruct_notebook_truth(row, sdss, ps1, ztf):
    object_id = row["objectId"]
    sdss_lc = sdss[sdss.objectId == object_id].copy()

    has_ps1 = pd.notna(row["ps1objID"])
    if has_ps1:
        ps1_id = int(row["ps1objID"])
        ps1_lc = ps1[ps1.ps1objID == ps1_id].copy()
        ztf_lc = ztf[ztf.ps1objID == ps1_id].copy()
    else:
        ps1_lc = pd.DataFrame()
        ztf_lc = pd.DataFrame()

    sdss_ps1_offset = {
        "g": row["sdss_g_qg"] - row["ps1_g_qg"],
        "r": row["sdss_r_qg"] - row["ps1_r_qg"],
        "i": row["sdss_i_qg"] - row["ps1_i_qg"],
        "z": row["sdss_z_qg"] - row["ps1_z_qg"],
    }

    times = {}
    mags = {}
    magerrs = {}
    surveys = {}
    mags_mean = []
    mags_mean_err = []
    for band in bands:
        filter_id = filters[band]
        offset = sdss_ps1_offset.get(band, 0.0)

        t_band = np.concatenate(
            [
                sdss_lc.loc[sdss_lc.filterID == filter_id, "mjd"].values if not sdss_lc.empty else np.array([]),
                ps1_lc.loc[ps1_lc.filterID == filter_id, "obsTime"].values if not ps1_lc.empty else np.array([]),
                ztf_lc.loc[ztf_lc.filterID == filter_id, "mjd"].values if not ztf_lc.empty else np.array([]),
            ]
        )
        m_band = np.concatenate(
            [
                sdss_lc.loc[sdss_lc.filterID == filter_id, "psMag"].values if not sdss_lc.empty else np.array([]),
                ps1_lc.loc[ps1_lc.filterID == filter_id, "psfMag"].values + offset if not ps1_lc.empty else np.array([]),
                ztf_lc.loc[ztf_lc.filterID == filter_id, "mag"].values + offset if not ztf_lc.empty else np.array([]),
            ]
        )
        me_band = np.concatenate(
            [
                sdss_lc.loc[sdss_lc.filterID == filter_id, "psMagErr_p3"].values if not sdss_lc.empty else np.array([]),
                ps1_lc.loc[ps1_lc.filterID == filter_id, "psfMagErr_p3"].values if not ps1_lc.empty else np.array([]),
                ztf_lc.loc[ztf_lc.filterID == filter_id, "magerr_p3"].values if not ztf_lc.empty else np.array([]),
            ]
        )
        survey_band = np.concatenate(
            [
                np.full(np.count_nonzero(sdss_lc.filterID == filter_id), "sdss", dtype="U4"),
                np.full(np.count_nonzero(ps1_lc.filterID == filter_id), "ps1", dtype="U4"),
                np.full(np.count_nonzero(ztf_lc.filterID == filter_id), "ztf", dtype="U4"),
            ]
        )

        keep = np.isfinite(m_band) & np.isfinite(me_band) & np.isfinite(t_band) & (me_band > 0)
        t_band = t_band[keep]
        m_band = m_band[keep]
        me_band = me_band[keep]
        survey_band = survey_band[keep]

        if len(t_band) > 0:
            order = np.argsort(t_band, kind="mergesort")
            t_band = t_band[order]
            m_band = m_band[order]
            me_band = me_band[order]
            survey_band = survey_band[order]

            retained = ~rolling_photometric_outlier_mask(t_band, m_band, me_band)
            t_band = t_band[retained]
            m_band = m_band[retained]
            me_band = me_band[retained]
            survey_band = survey_band[retained]

        times[band] = t_band
        mags[band] = m_band
        magerrs[band] = me_band
        surveys[band] = survey_band
        mean, mean_err = inverse_variance_weighted_mean(m_band, me_band)
        mags_mean.append(mean)
        mags_mean_err.append(mean_err)

    all_times = np.concatenate([times[band] for band in bands])
    all_surveys = np.concatenate([surveys[band] for band in bands])
    survey_times = {
        survey: np.sort(all_times[all_surveys == survey]) for survey in ("sdss", "ps1", "ztf")
    }
    cadence_times = np.sort(all_times)
    if cadence_times.size > 1:
        distinct = np.ones(cadence_times.size, dtype=bool)
        distinct[1:] = np.diff(cadence_times) > 30.0 / 1440.0
        cadence_times = cadence_times[distinct]
    dt = np.diff(cadence_times)
    dt = dt[(dt > 0) & (dt < 30)]
    cadence = float(np.median(dt)) if dt.size else np.nan
    cadence_err = (
        float(np.percentile(dt, 84) - np.percentile(dt, 16)) if dt.size else np.nan
    )
    return {
        "times": times,
        "mags": mags,
        "magerrs": magerrs,
        "surveys": surveys,
        "mags_mean": mags_mean,
        "mags_mean_err": mags_mean_err,
        "survey_times": survey_times,
        "number_points": int(all_times.size),
        "cadence": cadence,
        "cadence_err": cadence_err,
    }


def test_concat_light_curves_matches_notebook_truth_for_idx_10_g_band():
    cat, sdss, ps1, ztf = _load_s82_inputs()

    if 10 not in cat.index:
        pytest.skip("Catalog does not contain idx=10 used by notebook truth check")

    row = cat.loc[10]
    object_id = row["objectId"]
    if object_id not in set(sdss["objectId"].unique()):
        pytest.skip("Notebook idx=10 object is not present in SDSS raw table")

    truth = _reconstruct_notebook_truth(row, sdss, ps1, ztf)

    concat_objs = concat_light_curves(filter_object_ids=[object_id], progress_bar=False)
    assert len(concat_objs) >= 1
    got = _find_object(concat_objs, object_id)

    _assert_array_equalish(got["times"]["g"], truth["times"]["g"])
    _assert_mag_array_matches_truth(got["mags"]["g"], truth["mags"]["g"])
    _assert_array_equalish(got["magerrs"]["g"], truth["magerrs"]["g"])
    assert got["number_points"] == truth["number_points"]
    _assert_mag_array_matches_truth(got["mags_mean"], truth["mags_mean"])
    _assert_array_equalish(got["mags_mean_err"], truth["mags_mean_err"])


def test_concat_light_curves_matches_notebook_truth_on_fixed_random_sample():
    cat, sdss, ps1, ztf = _load_s82_inputs()
    sdss_ids = set(sdss["objectId"].unique())

    candidate_rows = cat.loc[cat["objectId"].isin(sdss_ids)].copy()
    candidate_rows = candidate_rows.loc[~candidate_rows["objectId"].duplicated(keep="first")]
    if len(candidate_rows) < 5:
        pytest.skip("Fewer than 5 overlapping unique object IDs found for notebook truth parity test")

    rng = np.random.default_rng(42)
    sample_pos = rng.choice(len(candidate_rows), size=5, replace=False)
    sample_rows = candidate_rows.iloc[sample_pos]
    sample_ids = sample_rows["objectId"].tolist()
    expected_concat_ids = candidate_rows.loc[candidate_rows["objectId"].isin(sample_ids), "objectId"].tolist()

    truth_by_object = {
        row["objectId"]: _reconstruct_notebook_truth(row, sdss, ps1, ztf)
        for _, row in sample_rows.iterrows()
    }

    concat_objs = concat_light_curves(filter_object_ids=sample_ids, progress_bar=False)
    concat_ids = [obj["object_id"] for obj in concat_objs]
    assert concat_ids == expected_concat_ids

    for object_id in sample_ids:
        truth = truth_by_object[object_id]
        got = _find_object(concat_objs, object_id)
        for band in bands:
            _assert_array_equalish(got["times"][band], truth["times"][band])
            _assert_mag_array_matches_truth(got["mags"][band], truth["mags"][band])
            _assert_array_equalish(got["magerrs"][band], truth["magerrs"][band])
            np.testing.assert_array_equal(got["surveys"][band], truth["surveys"][band])
        assert got["number_points"] == truth["number_points"]
        _assert_mag_array_matches_truth(got["mags_mean"], truth["mags_mean"])
        _assert_array_equalish(got["mags_mean_err"], truth["mags_mean_err"])
        assert np.isclose(got["cadence"], truth["cadence"], equal_nan=True)
        assert np.isclose(got["cadence_err"], truth["cadence_err"], equal_nan=True)
        for survey in ("sdss", "ps1", "ztf"):
            _assert_array_equalish(got["survey_times"][survey], truth["survey_times"][survey])
