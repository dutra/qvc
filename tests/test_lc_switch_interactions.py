"""Interactions of the zero-point night exclusion with --drop_empty_bands and the fixed constant-flux reference."""

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import hpc_scripts.sfitlc as sfitlc
import qvc.light_curve.build_constant_flux_reference as builder
import qvc.light_curve.psf_constant_flux_correction as correction
from qvc.light_curve import fit_light_curves, multiband_generate_lc
from qvc.light_curve.fit_light_curves import make_lc
from qvc.light_curve.zero_point_flags import ZeroPointResumeMismatch, file_sha256

from test_zero_point_flags import CLOUD_CELL, CLOUDY_NIGHT, _install

BANDS = list("ugriz")


def _rows():
    # Object A: every band on nights 51000 and 51002-51009; r has epochs only on
    # the cloudy night 51001, whose g cell is flagged (so all bands of that
    # object-night go).
    rows = []
    for k, band in enumerate("ugiz"):
        rows += [("A", band, night + 0.2 + 0.01 * k, 20.0 + 0.03 * np.sin(night + k))
                 for night in [51000, *range(51002, 51010)]]
    rows += [("A", "g", CLOUDY_NIGHT + 0.21, 20.0)]
    rows += [("A", "r", CLOUDY_NIGHT + 0.22, 20.0), ("A", "r", CLOUDY_NIGHT + 0.23, 20.01)]
    return rows


def _loaded(monkeypatch, cells):
    _install(monkeypatch, _rows())
    (obj,) = multiband_generate_lc.concat_light_curves(**({} if cells is None else {"zero_point_cells": cells}))
    obj["z"] = 1.2
    return obj


def test_night_exclusion_emptying_a_band_with_and_without_drop_empty_bands(monkeypatch, capsys):
    obj = _loaded(monkeypatch, CLOUD_CELL)
    assert len(obj["times"]["r"]) == 0 and obj["zero_point_flagged_epochs_dropped_r"] == 2
    assert "removed all 2 epochs of band r" in capsys.readouterr().out

    # --drop_empty_bands: r leaves the fit cleanly, like z.
    on = make_lc(copy.deepcopy(obj), BANDS, verbose=False, drop_empty_bands=True)
    assert on["bands"] == ["u", "g", "i"] and on["dropped_empty_bands"] == "r"
    assert np.all(np.isfinite(on["mags_means"]))
    # The remaining bands are prepared as when r is not requested at all.
    reference = make_lc(copy.deepcopy(obj), ["u", "g", "i", "z"], verbose=False)
    for key in ("y", "yerr", "band_idx", "mags_means"):
        np.testing.assert_array_equal(np.asarray(on[key]), np.asarray(reference[key]))

    # Off: what both branches do for any empty band -- r stays with a NaN mean
    # magnitude, which makes the object's fit fail later (non-finite reference flux).
    off = make_lc(copy.deepcopy(obj), BANDS, verbose=False)
    assert off["bands"] == ["u", "g", "r", "i"]
    assert np.isnan(off["mags_means"][2])
    assert "dropped_empty_bands" not in off


def test_without_flags_drop_empty_bands_sees_all_bands(monkeypatch):
    obj = _loaded(monkeypatch, None)
    on = make_lc(copy.deepcopy(obj), BANDS, verbose=False, drop_empty_bands=True)
    assert on["bands"] == ["u", "g", "r", "i"] and on["dropped_empty_bands"] == ""


def test_reference_builder_applies_the_fit_night_exclusion(monkeypatch, tmp_path):
    """The reference is the full light curve as the fit sees it, flags included."""
    monkeypatch.setattr(builder, "populate_sdss_fields", lambda objs, progress_bar=False: objs)
    calls = []
    real_concat = multiband_generate_lc.concat_light_curves

    def spy(**kwargs):
        calls.append(kwargs)
        return real_concat(**kwargs)

    monkeypatch.setattr(builder, "concat_light_curves", spy)
    _install(monkeypatch, _rows())
    flagged = builder.build_constant_flux_reference_table(["A"], zero_point_cells=CLOUD_CELL)
    plain = builder.build_constant_flux_reference_table(["A"])
    assert "zero_point_cells" in calls[0] and calls[0]["zero_point_flag_threshold"] == 0.1
    assert "zero_point_cells" not in calls[1]
    # Equal to the reference the fit's loader gives with the flags.
    fit_view = real_concat(zero_point_cells=CLOUD_CELL)
    expected = builder.reference_rows_for_objects(fit_view)
    pd.testing.assert_frame_equal(flagged, expected)
    g = flagged.set_index("band").loc["g"]
    assert g["n_epochs"] == plain.set_index("band").loc["g", "n_epochs"] - 1

    # Command line: the flag table and threshold are passed through.
    cells_path = tmp_path / "cells.csv"  # read_parquet is faked for the S82 inputs
    CLOUD_CELL.to_csv(cells_path, index=False)
    out = tmp_path / "ref.csv"
    assert builder.main(["--filter_object_id", "A", "--output", str(out),
                         "--zero_point_flag_table", str(cells_path), "--zero_point_flag_threshold", "0.2"]) == 0
    assert calls[-1]["zero_point_flag_threshold"] == 0.2
    pd.testing.assert_frame_equal(calls[-1]["zero_point_cells"].reset_index(drop=True),
                                  CLOUD_CELL.astype({"night": np.int64, "rab": np.int64}).reset_index(drop=True),
                                  check_dtype=False)
    with pytest.raises(SystemExit):
        builder.main(["--filter_object_id", "A", "--output", str(out), "--zero_point_flag_threshold", "0.2"])


def test_fit_records_both_features_and_keeps_the_resume_guard(monkeypatch, tmp_path):
    """Both options on: per-row columns of both, and --resume still refuses a changed flag setting."""
    _install(monkeypatch, _rows())
    loaded = multiband_generate_lc.concat_light_curves()
    objs = []
    monkeypatch.setattr(fit_light_curves, "concat_light_curves",
                        lambda **kw: objs.extend(multiband_generate_lc.concat_light_curves(**kw)) or objs)
    monkeypatch.setattr(fit_light_curves, "populate_sdss_fields", lambda o, **kw: [dict(x, z=1.2) for x in o])
    monkeypatch.setattr(correction, "load_spectra_psf_fractions",
                        lambda *a, **k: {"A": {"object_id": "A", **{f"f_AGN_psf_{b}": 0.5 for b in BANDS}}})
    seen = {}

    def fake_load(oid, return_metadata=False):
        return {}, {}  # saved without the flag table

    def capture_validate(obj, bands):
        seen["obj"] = obj
        return real_validate(obj, bands)

    real_validate = fit_light_curves.validate_constant_flux_reference
    monkeypatch.setattr(fit_light_curves, "validate_constant_flux_reference", capture_validate)
    monkeypatch.setattr(fit_light_curves, "load_obj_samples_from_hdf5", fake_load)
    reference = builder.reference_rows_for_objects(loaded)
    ref_path = builder.write_constant_flux_reference_table(reference, tmp_path / "ref.csv")
    cells_path = tmp_path / "cells.csv"
    CLOUD_CELL.to_csv(cells_path, index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "fit_light_curves", "--filter_object_id", "A", "--resume", "--drop_empty_bands",
        "--zero_point_flag_table", str(cells_path),
        "--subtract_psf_constant_flux", "--spectra_fit_h5", "unused.h5", "--psf-fraction-mode", "median",
        "--constant_flux_reference_path", str(ref_path),
    ])
    with pytest.raises(ZeroPointResumeMismatch):
        fit_light_curves.main()
    obj = seen["obj"]
    assert obj["zero_point_flag_table_sha256"] == file_sha256(cells_path)
    assert obj["zero_point_flagged_epochs_dropped_r"] == 2
    assert obj["constant_flux_reference_sha256"] == file_sha256(ref_path)
    assert obj["dropped_empty_bands"] == "r"


def test_job_scripts_carry_both_launcher_changes(monkeypatch, tmp_path):
    cells = tmp_path / "cells.parquet"
    CLOUD_CELL.to_parquet(cells)
    monkeypatch.setattr(sys, "argv", ["sfitlc.py", "--fit", "chisq", "--chisq-csv", "c.csv", "--spectra-fit-h5", "s.h5",
                                      "--zero-point-flag-table", str(cells), "--drop_empty_bands"])
    args = sfitlc.parse_args()
    script = sfitlc.build_sbatch_script(
        "probe", sfitlc.JobConfig(description="chisq", object_ids=["1"], use_psf_constant_flux=True),
        args, "c.csv", "s.h5", object_ids_path=Path("/tmp/ids.txt"),
    )
    assert f'export PYTHONPATH="{sfitlc.SRC_ROOT}' in script
    command = script.split("python -m qvc.light_curve.fit_light_curves", 1)[1]
    assert "--zero_point_flag_table" in command and str(cells.resolve()) in command
    assert "--drop_empty_bands" in command
    merge = sfitlc.build_merge_sbatch_script("probe", "chisq", args)
    assert f'export PYTHONPATH="{sfitlc.SRC_ROOT}' in merge
