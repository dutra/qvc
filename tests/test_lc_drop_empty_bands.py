"""--drop_empty_bands: bands without valid epochs are dropped like z."""

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import hpc_scripts.sfitlc as sfitlc
from qvc.light_curve.fit_light_curves import (
    DROP_EMPTY_BANDS_MIN_BANDS,
    compute_lambda_center_rf,
    make_lc,
)
from qvc.light_curve.multiband_fit_utils import lambda_pivot

ADDED_KEYS = {"drop_empty_bands", "dropped_empty_bands"}


def _object(z=1.2, empty=(), n=6):
    rng = np.random.default_rng(3)
    bands = ["u", "g", "r", "i", "z"]
    obj = {
        "object_id": "1234",
        "z": z,
        "times": {},
        "mags": {},
        "magerrs": {},
        "surveys": {},
        "psf_fwhm_arcsec": {},
        "survey_names": ("sdss", "ps1", "ztf"),
        "cadence": [5.0] * 5,
        "cadence_err": [0.5] * 5,
        "number_points": [n] * 5,
    }
    for k, band in enumerate(bands):
        m = 0 if band in empty else n
        obj["times"][band] = np.sort(rng.uniform(0.0, 3000.0, m)) + 0.1 * k
        obj["mags"][band] = 20.0 + 0.1 * rng.standard_normal(m)
        obj["magerrs"][band] = np.full(m, 0.03)
        obj["surveys"][band] = np.array(
            (["sdss", "ps1", "ztf"] * n)[:m], dtype=str
        )
        obj["psf_fwhm_arcsec"][band] = rng.uniform(0.8, 2.0, m)
    return obj


def _assert_same_prepared_inputs(a, b, skip=()):
    assert set(a) - set(skip) == set(b) - set(skip)
    for key in a:
        if key in skip:
            continue
        va, vb = a[key], b[key]
        if key == "X":
            for xa, xb in zip(va, vb):
                np.testing.assert_array_equal(np.asarray(xa), np.asarray(xb))
        elif isinstance(va, (np.ndarray, list, tuple)) or hasattr(va, "shape"):
            np.testing.assert_array_equal(np.asarray(va), np.asarray(vb), err_msg=key)
        elif isinstance(va, float) and np.isnan(va):
            assert isinstance(vb, float) and np.isnan(vb), key
        else:
            assert va == vb, key


@pytest.mark.parametrize("drop_lya", [False, True])
def test_option_on_with_all_bands_populated_matches_default(drop_lya):
    obj = _object(z=2.6)
    default = make_lc(copy.deepcopy(obj), list("ugriz"), drop_band_lyman_alpha=drop_lya, verbose=False)
    dropped = make_lc(
        copy.deepcopy(obj),
        list("ugriz"),
        drop_band_lyman_alpha=drop_lya,
        verbose=False,
        drop_empty_bands=True,
    )

    assert not (ADDED_KEYS & set(default))
    _assert_same_prepared_inputs(default, dropped, skip=ADDED_KEYS)
    assert dropped["drop_empty_bands"] is True
    assert dropped["dropped_empty_bands"] == ""


def test_default_keeps_an_empty_band_with_a_nan_mean():
    """Default behaviour is unchanged: the empty u band stays and its mean is NaN,
    which later makes the reference flux F0_cont_band non-finite."""

    lc = make_lc(_object(empty=("u",)), list("ugriz"), verbose=False)

    assert lc["bands"] == ["u", "g", "r", "i"]
    assert lc["dropped_bands"] == ["z"]
    assert np.isnan(lc["mags_means"][0])
    assert not np.isfinite(10.0 ** (-0.4 * lc["mags_means"][0]))


def test_empty_u_band_is_dropped_and_moves_the_pivot():
    obj = _object(z=1.2, empty=("u",))
    lc = make_lc(obj, list("ugriz"), verbose=False, drop_empty_bands=True)

    assert lc["bands"] == ["g", "r", "i"]
    assert lc["dropped_bands"] == ["z", "u"]
    assert lc["dropped_empty_bands"] == "u"
    assert np.all(np.isfinite(lc["mags_means"]))
    assert sorted(set(np.asarray(lc["band_idx"]).tolist())) == [0, 1, 2]

    lam_rf = np.array([lambda_pivot[b] for b in lc["bands"]]) / (1.0 + obj["z"])
    lam_all = np.array([lambda_pivot[b] for b in "ugri"]) / (1.0 + obj["z"])
    pivot = float(compute_lambda_center_rf(lam_rf))
    assert pivot == pytest.approx(np.exp(np.mean(np.log(lam_rf))))
    assert pivot > float(compute_lambda_center_rf(lam_all))

    # The remaining bands are prepared exactly as when only g, r, i are requested.
    reference = make_lc(_object(z=1.2, empty=("u",)), list("griz"), verbose=False)
    _assert_same_prepared_inputs(reference, lc, skip=ADDED_KEYS | {"dropped_bands"})


def test_band_with_only_invalid_epochs_counts_as_empty():
    obj = _object()
    obj["mags"]["u"][:] = np.nan
    obj["magerrs"]["g"][:] = 0.0

    lc = make_lc(obj, list("ugriz"), verbose=False, drop_empty_bands=True)

    assert lc["bands"] == ["r", "i"]
    assert lc["dropped_empty_bands"] == "u,g"


def test_band_missing_from_the_input_dicts_counts_as_empty():
    obj = _object()
    for key in ("times", "mags", "magerrs", "surveys", "psf_fwhm_arcsec"):
        obj[key].pop("u")

    lc = make_lc(obj, list("ugriz"), verbose=False, drop_empty_bands=True)

    assert lc["bands"] == ["g", "r", "i"]
    assert lc["dropped_empty_bands"] == "u"


def test_too_few_bands_left_fails_with_a_clear_message():
    assert DROP_EMPTY_BANDS_MIN_BANDS == 2
    obj = _object(empty=("u", "r", "i"))

    with pytest.raises(ValueError, match="at least 2 bands"):
        make_lc(obj, list("ugriz"), verbose=False, drop_empty_bands=True)


def test_two_bands_need_two_epochs_each():
    obj = _object(empty=("u", "i"))
    for key in ("times", "mags", "magerrs", "surveys", "psf_fwhm_arcsec"):
        obj[key]["r"] = obj[key]["r"][:1]

    with pytest.raises(ValueError, match=r"valid epoch counts \[6, 1\]"):
        make_lc(obj, list("ugriz"), verbose=False, drop_empty_bands=True)

    obj = _object(empty=("u", "i"), n=2)
    lc = make_lc(obj, list("ugriz"), verbose=False, drop_empty_bands=True)
    assert lc["bands"] == ["g", "r"]


def _script(args):
    return sfitlc.build_sbatch_script(
        "probe",
        sfitlc.JobConfig(description="stone", object_ids=["1"]),
        args,
        None,
        None,
        object_ids_path=Path("/tmp/ids.txt"),
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_launcher_forwards_drop_empty_bands_only_when_given(monkeypatch, enabled):
    argv = ["sfitlc.py", "--fit", "stone"] + (["--drop_empty_bands"] if enabled else [])
    monkeypatch.setattr(sys, "argv", argv)
    args = sfitlc.parse_args()

    assert args.extra_fit_flags == (("--drop_empty_bands",) if enabled else ())
    script = _script(args)
    assert script.count("--drop_empty_bands") == int(enabled)
    command = script.split("python -m qvc.light_curve.fit_light_curves", 1)[1]
    assert command.count("--drop_empty_bands") == int(enabled)
    assert subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True).returncode == 0
