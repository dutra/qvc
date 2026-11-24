"""--constant_flux_reference_path: fixed reference flux for the constant-flux subtraction."""

import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import hpc_scripts.sfitlc as sfitlc
import qvc.light_curve.build_constant_flux_reference as builder
import qvc.light_curve.psf_constant_flux_correction as correction
from qvc.light_curve.fit_light_curves import make_lc

BANDS = ["u", "g", "r", "i", "z"]
FRACTIONS = {"u": 0.5, "g": 0.6, "r": 0.55, "i": 0.45, "z": 0.4}


def _object(oid="123", n=40, seed=0):
    rng = np.random.default_rng(seed)
    obj = {"object_id": oid, "z": 1.1, "times": {}, "mags": {}, "magerrs": {}, "surveys": {}}
    for k, band in enumerate(BANDS):
        t = np.sort(rng.uniform(0.0, 6000.0, n))
        obj["times"][band] = t
        obj["mags"][band] = 19.5 + 0.2 * np.sin(t / 700.0 + k) + 0.03 * rng.standard_normal(n)
        obj["magerrs"][band] = rng.uniform(0.01, 0.05, n)
        # first half SDSS, then PS1, then ZTF
        obj["surveys"][band] = np.where(t < 3000, "sdss", np.where(t < 4500, "ps1", "ztf"))
    obj["survey_names"] = ("sdss", "ps1", "ztf")
    obj["cadence"] = [5.0] * 5
    obj["cadence_err"] = [0.5] * 5
    obj["number_points"] = [n] * 5
    return obj


def _subset(obj, surveys):
    out = copy.deepcopy(obj)
    for band in BANDS:
        keep = np.isin(obj["surveys"][band], surveys)
        for key in ("times", "mags", "magerrs", "surveys"):
            out[key][band] = np.asarray(obj[key][band])[keep]
    return out


def _spectra(monkeypatch, oids):
    rows = {
        oid: {"object_id": oid, **{f"f_AGN_psf_{b}": f for b, f in FRACTIONS.items()}}
        for oid in oids
    }
    monkeypatch.setattr(correction, "load_spectra_psf_fractions", lambda *a, **k: rows)


def _table(objs, tmp_path):
    path = builder.write_constant_flux_reference_table(
        builder.reference_rows_for_objects(objs), tmp_path / "reference.csv"
    )
    return path, correction.load_constant_flux_reference_table(path)


def test_reference_helper_is_the_default_subtraction_reference():
    obj = _object()
    for band in BANDS:
        mags, errs = obj["mags"][band], obj["magerrs"][band]
        default = correction.subtract_constant_flux_from_band(mags, errs, 0.6)
        reference = correction.reference_total_flux_from_band(mags, errs)
        fixed = correction.subtract_constant_flux_from_band(
            mags, errs, 0.6, reference_total_flux=reference
        )
        assert default[2]["reference_total_flux"] == reference
        np.testing.assert_array_equal(default[0], fixed[0])
        np.testing.assert_array_equal(default[1], fixed[1])
        assert default[2] == fixed[2]


def test_csv_round_trip_is_exact(tmp_path):
    objs = [_object("123"), _object("456", seed=1)]
    path, table = _table(objs, tmp_path)
    for obj in objs:
        for band in BANDS:
            expected = correction.reference_total_flux_from_band(
                obj["mags"][band], obj["magerrs"][band]
            )
            assert table[obj["object_id"]][band] == expected
    assert list(pd.read_csv(path).columns[:3]) == list(correction.CONSTANT_FLUX_REFERENCE_COLUMNS)


def test_full_light_curve_table_reproduces_the_default_prepared_inputs(monkeypatch, tmp_path):
    objs = [_object("123"), _object("456", seed=1)]
    _spectra(monkeypatch, ["123", "456"])
    _, table = _table(objs, tmp_path)

    default, _ = correction.apply_constant_flux_correction_to_objects(
        copy.deepcopy(objs), spectra_fit_h5s=["unused.h5"]
    )
    fixed, _ = correction.apply_constant_flux_correction_to_objects(
        copy.deepcopy(objs), spectra_fit_h5s=["unused.h5"], reference_table=table
    )
    for a, b in zip(default, fixed):
        for band in BANDS:
            np.testing.assert_array_equal(a["mags"][band], b["mags"][band])
            np.testing.assert_array_equal(a["magerrs"][band], b["magerrs"][band])
            sa = a["psf_constant_flux_band_summaries"][band]
            sb = dict(b["psf_constant_flux_band_summaries"][band])
            assert sb.pop("reference_source") == "table"
            assert sa == sb
        assert b["psf_constant_flux_reference_invalid_bands"] == ""
        correction.validate_constant_flux_reference(b, ["u", "g", "r", "i"])
        lc_a = make_lc(a, list(BANDS), verbose=False)
        lc_b = make_lc(b, list(BANDS), verbose=False)
        for key in ("y", "yerr", "mags_means", "band_idx"):
            np.testing.assert_array_equal(np.asarray(lc_a[key]), np.asarray(lc_b[key]))


def test_subset_with_full_reference_removes_the_full_light_curve_constant(monkeypatch, tmp_path):
    full = _object("123")
    subset = _subset(full, ["ps1", "ztf"])
    _spectra(monkeypatch, ["123"])
    _, table = _table([full], tmp_path)

    (full_corr,), _ = correction.apply_constant_flux_correction_to_objects(
        [copy.deepcopy(full)], spectra_fit_h5s=["unused.h5"]
    )
    (sub_default,), _ = correction.apply_constant_flux_correction_to_objects(
        [copy.deepcopy(subset)], spectra_fit_h5s=["unused.h5"]
    )
    (sub_fixed,), _ = correction.apply_constant_flux_correction_to_objects(
        [copy.deepcopy(subset)], spectra_fit_h5s=["unused.h5"], reference_table=table
    )
    for band in BANDS:
        full_c = full_corr["psf_constant_flux_band_summaries"][band]["constant_contaminant_flux"]
        fixed_c = sub_fixed["psf_constant_flux_band_summaries"][band]["constant_contaminant_flux"]
        default_c = sub_default["psf_constant_flux_band_summaries"][band]["constant_contaminant_flux"]
        assert fixed_c == full_c
        assert default_c != full_c


def test_object_missing_from_table_fails_only_that_object(monkeypatch, tmp_path):
    objs = [_object("123"), _object("456", seed=1)]
    _spectra(monkeypatch, ["123", "456"])
    _, table = _table(objs[:1], tmp_path)

    corrected, _ = correction.apply_constant_flux_correction_to_objects(
        objs, spectra_fit_h5s=["unused.h5"], reference_table=table
    )
    correction.validate_constant_flux_reference(corrected[0], ["g", "r"])
    with pytest.raises(ValueError, match="456 is missing from the constant-flux reference table"):
        correction.validate_constant_flux_reference(corrected[1], ["g", "r"])
    np.testing.assert_array_equal(corrected[1]["mags"]["g"], objs[1]["mags"]["g"])


@pytest.mark.parametrize("bad_value", [np.nan, 0.0, -1.0, None])
def test_invalid_value_fails_a_fitted_band_but_not_an_unfitted_one(monkeypatch, tmp_path, bad_value):
    obj = _object("123")
    _spectra(monkeypatch, ["123"])
    _, table = _table([obj], tmp_path)
    if bad_value is None:
        table["123"].pop("g")
        table["123"].pop("z")
    else:
        table["123"]["g"] = bad_value
        table["123"]["z"] = bad_value

    (corrected,), _ = correction.apply_constant_flux_correction_to_objects(
        [obj], spectra_fit_h5s=["unused.h5"], reference_table=table
    )

    assert corrected["psf_constant_flux_reference_invalid_bands"] == "g,z"
    correction.validate_constant_flux_reference(corrected, ["u", "r", "i"])  # z is not fitted
    with pytest.raises(ValueError, match=r"fitted band\(s\) \['g'\]"):
        correction.validate_constant_flux_reference(corrected, ["u", "g", "r", "i"])


def test_band_without_epochs_needs_no_reference(monkeypatch, tmp_path):
    obj = _subset(_object("123"), ["ps1", "ztf"])
    for key in ("times", "mags", "magerrs", "surveys"):
        obj[key]["u"] = obj[key]["u"][:0]
    _spectra(monkeypatch, ["123"])
    _, table = _table([_object("123")], tmp_path)
    table["123"].pop("u")

    (corrected,), _ = correction.apply_constant_flux_correction_to_objects(
        [obj], spectra_fit_h5s=["unused.h5"], reference_table=table
    )

    assert corrected["psf_constant_flux_reference_invalid_bands"] == ""
    lc = make_lc(corrected, list(BANDS), verbose=False, drop_empty_bands=True)
    assert lc["bands"] == ["g", "r", "i"]
    correction.validate_constant_flux_reference(corrected, lc["bands"])


def test_table_loader_rejects_bad_tables(tmp_path):
    missing = tmp_path / "missing.csv"
    pd.DataFrame({"object_id": ["1"], "band": ["g"]}).to_csv(missing, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        correction.load_constant_flux_reference_table(missing)

    duplicate = tmp_path / "duplicate.csv"
    pd.DataFrame(
        {"object_id": ["1", "1"], "band": ["g", "g"], "reference_total_flux": [1e-8, 2e-8]}
    ).to_csv(duplicate, index=False)
    with pytest.raises(ValueError, match="duplicate .* 1/g"):
        correction.load_constant_flux_reference_table(duplicate)


def test_builder_uses_the_fitter_loading_path(monkeypatch, tmp_path):
    calls = {}
    objs = [_object("123"), _object("456", seed=1)]

    def fake_concat(**kwargs):
        calls["concat"] = kwargs
        return copy.deepcopy(objs)

    def fake_populate(loaded, progress_bar=False):
        calls["populate"] = True
        return loaded

    def fake_cut(loaded, n_days, same_length):
        calls["cut"] = (n_days, same_length)
        return loaded

    monkeypatch.setattr(builder, "concat_light_curves", fake_concat)
    monkeypatch.setattr(builder, "populate_sdss_fields", fake_populate)
    monkeypatch.setattr(builder, "cut_light_curve_restframe_window", fake_cut)

    table = builder.build_constant_flux_reference_table(
        ["123", "456"], outlier_half_window_days=20.0, sdss_source="macleod", rf_length_cut=2400
    )

    assert calls["concat"]["filter_object_ids"] == ["123", "456"]
    assert calls["concat"]["outlier_half_window_days"] == 20.0
    assert calls["concat"]["sdss_source"] == "macleod"
    assert calls["populate"] and calls["cut"] == (2400, False)
    assert len(table) == 10
    assert set(table["n_epochs"]) == {40}

    with pytest.raises(ValueError, match="No light curve loaded for object_id\\(s\\): 789"):
        builder.build_constant_flux_reference_table(["123", "789"])

    out = tmp_path / "ref.csv"
    assert builder.main(["--filter_object_id", "123", "456", "--output", str(out)]) == 0
    assert set(correction.load_constant_flux_reference_table(out)) == {"123", "456"}


@pytest.mark.parametrize("enabled", [False, True])
def test_launcher_forwards_reference_path_only_when_given(monkeypatch, enabled):
    flags = ["--constant_flux_reference_path", "/abs/reference.csv"] if enabled else []
    monkeypatch.setattr(
        sys,
        "argv",
        ["sfitlc.py", "--fit", "chisq", "--chisq-csv", "c.csv", "--spectra-fit-h5", "s.h5", *flags],
    )
    args = sfitlc.parse_args()

    assert args.extra_fit_flags == tuple(flags)
    script = sfitlc.build_sbatch_script(
        "probe",
        sfitlc.JobConfig(description="chisq", object_ids=["1"], use_psf_constant_flux=True),
        args,
        "c.csv",
        "s.h5",
        object_ids_path=Path("/tmp/ids.txt"),
    )
    command = script.split("python -m qvc.light_curve.fit_light_curves", 1)[1]
    assert command.count("--constant_flux_reference_path") == int(enabled)
    assert command.count("/abs/reference.csv") == int(enabled)
    assert subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True).returncode == 0


def test_fitter_rejects_reference_path_without_median_subtraction():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "qvc.light_curve.fit_light_curves",
         "--constant_flux_reference_path", "reference.csv"],
        text=True,
        capture_output=True,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )

    assert result.returncode == 2
    assert "--constant_flux_reference_path requires --subtract_psf_constant_flux" in result.stderr
