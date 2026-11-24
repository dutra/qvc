"""Dropping SDSS epochs taken on non-photometric nights (zero-point cell flags)."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qvc.light_curve import multiband_generate_lc
from qvc.light_curve.zero_point_flags import (
    ZERO_POINT_CELL_COLUMNS,
    build_zero_point_cells,
    file_sha256,
    flag_zero_point_epochs,
    load_zero_point_cells,
    validate_zero_point_cells,
    zero_point_ra_bin,
)

REPO = Path(__file__).resolve().parents[1]
# Object A at RA 10.5 (bin 5), object B at RA 359 (wrapped to -1, bin -1).
RA = {"A": 10.5, "B": 359.0}


def _cells():
    return pd.DataFrame(
        {
            "band": ["g", "r", "i", "u", "g", "r"],
            "night": [51000, 51001, 51002, 51003, 51004, 51000],
            "rab": [5, 5, 5, 5, 5, -1],
            "cell_off": [0.30, 0.08, -0.15, 0.50, 0.40, -0.20],
            "cell_n": [20, 20, 20, 20, 7, 30],
        }
    )


def _epochs():
    rows = []
    for obj in ("A", "B"):
        for night in (51000, 51001, 51002, 51003, 51004, 51005):
            for band in "ugriz":
                rows.append((obj, band, night + 0.2 + 0.001 * "ugriz".index(band)))
    return pd.DataFrame(rows, columns=["object_id", "band", "time"])


def test_ra_bins_wrap_like_the_analysis():
    np.testing.assert_array_equal(zero_point_ra_bin([0.0, 1.99, 2.0, 10.5, 180.0, 181.0, 359.0]), [0, 0, 1, 5, 90, -90, -1])


def test_flag_rule():
    epochs = _epochs()
    mask = flag_zero_point_epochs(epochs, RA, validate_zero_point_cells(_cells()), 0.1)
    flagged = set(map(tuple, epochs.loc[mask, ["object_id", "time"]].assign(night=lambda d: d.time.astype(int))[["object_id", "night"]].values))
    # A: g on 51000 (0.30) and i on 51002 (|-0.15|) flag all five bands of those nights;
    # r on 51001 is below 0.1, u on 51003 is not a flag band, 51004 has n < 8.
    # B: its r cell on 51000 (bin -1) is flagged.
    assert flagged == {("A", 51000), ("A", 51002), ("B", 51000)}
    assert int(mask.sum()) == 15
    # At 0.2 only A's 0.30 night remains (|offset| must exceed the threshold).
    mask = flag_zero_point_epochs(epochs, RA, validate_zero_point_cells(_cells()), 0.2)
    assert int(mask.sum()) == 5
    with pytest.raises(ValueError, match="positive magnitude"):
        flag_zero_point_epochs(epochs, RA, _cells(), 0.0)
    with pytest.raises(ValueError, match="No RA"):
        flag_zero_point_epochs(epochs, {"A": 1.0}, _cells(), 0.1)


def test_cell_table_validation_and_io(tmp_path):
    for suffix in (".parquet", ".csv"):
        path = tmp_path / f"cells{suffix}"
        (_cells().to_parquet if suffix == ".parquet" else lambda p: _cells().to_csv(p, index=False))(path)
        pd.testing.assert_frame_equal(load_zero_point_cells(path), validate_zero_point_cells(_cells()))
    with pytest.raises(ValueError, match="missing column"):
        validate_zero_point_cells(_cells().drop(columns="cell_n"))
    with pytest.raises(ValueError, match="duplicate"):
        validate_zero_point_cells(pd.concat([_cells(), _cells().iloc[:1]]))
    with pytest.raises(ValueError, match="unknown band"):
        validate_zero_point_cells(_cells().assign(band="y"))
    with pytest.raises(FileNotFoundError):
        load_zero_point_cells(tmp_path / "missing.parquet")


def test_builder_matches_a_direct_computation():
    rng = np.random.default_rng(1)
    epochs = _epochs()
    epochs["mag"] = 20.0 + rng.normal(0.0, 0.1, len(epochs))
    cells = build_zero_point_cells(epochs, RA)
    assert list(cells.columns) == list(ZERO_POINT_CELL_COLUMNS)
    res = epochs.mag - epochs.groupby(["object_id", "band"]).mag.transform("median")
    expected = (
        pd.DataFrame({"band": epochs.band, "night": epochs.time.astype(int),
                      "rab": epochs.object_id.map({"A": 5, "B": -1}), "res": res})
        .groupby(["band", "night", "rab"]).res.agg(cell_off="median", cell_n="size").reset_index()
    )
    pd.testing.assert_frame_equal(cells, validate_zero_point_cells(expected))


# --- concat_light_curves ------------------------------------------------------


def _fake_inputs(monkeypatch):
    epochs = _epochs()
    sdss = pd.DataFrame(
        {
            "objectId": epochs.object_id,
            "mjd": epochs.time,
            "filterID": epochs.band.map(multiband_generate_lc.filters),
            "psMag": 20.0 + 0.01 * np.arange(len(epochs)) % 0.05,
            "psMagErr_p3": 0.05,
        }
    )
    frames = {
        "Catalog.parquet": pd.DataFrame(
            {"idx": [0, 1], "objectId": ["A", "B"], "RA": [RA["A"], RA["B"]], "DEC": [0.0, 0.0], "ps1objID": [np.nan, np.nan]}
        ),
        "dr16s82_sdssLCRaw.parquet": sdss,
        "dr16s82_ps1LCRaw.parquet": pd.DataFrame(columns=["ps1objID", "filterID", "obsTime", "psfMag", "psfMagErr_p3"]),
        "dr16s82_ZuberLCRaw.parquet": pd.DataFrame(columns=["ps1objID", "filterID", "mjd", "mag", "magerr_p3"]),
    }

    def fake_resolve(path):
        if path in multiband_generate_lc.SEEING_SIDECARS.values():
            raise FileNotFoundError(path)
        return path

    monkeypatch.setattr(multiband_generate_lc, "resolve_qvc_data_path", fake_resolve)
    monkeypatch.setattr(multiband_generate_lc.pd, "read_parquet", lambda path: frames[str(path).rsplit("/", 1)[-1]].copy())


def _assert_same(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for key in a:
            _assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)) and not np.isscalar(a):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_same(x, y)
    else:
        x, y = np.asarray(a), np.asarray(b)
        assert x.dtype == y.dtype and x.shape == y.shape
        assert x.tobytes() == y.tobytes()


def test_concat_light_curves_off_and_on(monkeypatch):
    _fake_inputs(monkeypatch)
    default = multiband_generate_lc.concat_light_curves()
    explicit_off = multiband_generate_lc.concat_light_curves(zero_point_cells=None, zero_point_flag_threshold=0.3)
    _assert_same(default, explicit_off)
    assert not any(k.startswith("zero_point") for obj in default for k in obj)

    flagged = {obj["object_id"]: obj for obj in multiband_generate_lc.concat_light_curves(zero_point_cells=_cells())}
    base = {obj["object_id"]: obj for obj in default}
    for band in "ugriz":
        assert flagged["A"][f"zero_point_flagged_epochs_dropped_{band}"] == 2
        assert flagged["B"][f"zero_point_flagged_epochs_dropped_{band}"] == 1
        assert len(flagged["A"]["times"][band]) == len(base["A"]["times"][band]) - 2
        assert not np.any(np.floor(flagged["A"]["times"][band]).astype(int) == 51000)
    assert flagged["A"]["number_points"] == base["A"]["number_points"] - 10
    with pytest.raises(ValueError, match="dr16s82"):
        multiband_generate_lc.concat_light_curves(zero_point_cells=_cells(), sdss_source="macleod")


# --- fit_light_curves and the launcher ---------------------------------------


class _Stop(Exception):
    pass


def _run_fitter_main(monkeypatch, argv):
    from qvc.light_curve import fit_light_curves

    captured = {}

    def fake_concat(**kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(fit_light_curves, "concat_light_curves", fake_concat)
    monkeypatch.setattr(sys, "argv", ["fit_light_curves", "--filter_object_id", "A", *argv])
    with pytest.raises(_Stop):
        fit_light_curves.main()
    return captured


def test_fitter_passes_the_table_only_when_given(monkeypatch, tmp_path):
    captured = _run_fitter_main(monkeypatch, [])
    assert "zero_point_cells" not in captured and "zero_point_flag_threshold" not in captured
    path = tmp_path / "cells.parquet"
    _cells().to_parquet(path)
    captured = _run_fitter_main(monkeypatch, ["--zero_point_flag_table", str(path)])
    pd.testing.assert_frame_equal(captured["zero_point_cells"], validate_zero_point_cells(_cells()))
    assert captured["zero_point_flag_threshold"] == 0.1
    captured = _run_fitter_main(monkeypatch, ["--zero_point_flag_table", str(path), "--zero_point_flag_threshold", "0.2"])
    assert captured["zero_point_flag_threshold"] == 0.2


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--zero_point_flag_threshold", "0.2"], "requires --zero_point_flag_table"),
        (["--zero_point_flag_table", "missing.parquet"], "not found"),
        (["--zero_point_flag_table", "CELLS", "--sdss_source", "macleod"], "dr16s82 SDSS epochs only"),
        (["--zero_point_flag_table", "CELLS", "--zero_point_flag_threshold", "-1"], "positive magnitude"),
    ],
)
def test_fitter_rejects_bad_settings(monkeypatch, tmp_path, capsys, argv, message):
    path = tmp_path / "cells.parquet"
    _cells().to_parquet(path)
    argv = [str(path) if value == "CELLS" else value for value in argv]
    from qvc.light_curve import fit_light_curves

    monkeypatch.setattr(sys, "argv", ["fit_light_curves", *argv])
    with pytest.raises(SystemExit):
        fit_light_curves.main()
    assert message in capsys.readouterr().err


def _launcher_args(**overrides):
    values = {
        "fit": "chisq", "N": 3, "skip": 0, "ncores": 1, "mem": "12G", "partition": "day", "time": "2:00:00",
        "env": "jaxcpu2", "svi_steps": 4000, "svi_lr": 1e-3, "nwarm": 500, "nsamp": 250, "max_tree_depth": 12,
        "resume": None, "extra_fit_flags": (),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _script(args, monkeypatch):
    import hpc_scripts.sfitlc as sfitlc

    monkeypatch.setattr(sfitlc, "submission_record", lambda entrypoint, argv, resolved: {"resolved": resolved})
    return sfitlc.build_sbatch_script(
        "probe", sfitlc.JobConfig(description="chisq", object_ids=["1"]), args, "data/input.csv", None,
        object_ids_path=Path("/tmp/probe_ids.txt"),
    )


def test_launcher_off_is_unchanged_and_on_passes_the_flags(monkeypatch, tmp_path):
    from qvc.provenance import decode_record
    import re

    def record(script):
        return decode_record(re.search(r'export QVC_\w+="([^"]+)"', script).group(1))

    without = _script(_launcher_args(), monkeypatch)
    off = _script(_launcher_args(zero_point_flag_table=None, zero_point_flag_threshold=None), monkeypatch)
    assert without == off
    assert "zero_point" not in without
    path = tmp_path / "cells.parquet"
    _cells().to_parquet(path)
    on = _script(_launcher_args(zero_point_flag_table=str(path), zero_point_flag_threshold=0.2), monkeypatch)
    assert "--zero_point_flag_table" in on and str(path) in on and "--zero_point_flag_threshold" in on
    resolved = record(on)["resolved"]
    assert resolved["inputs"]["zero_point_flag_table_sha256"] == file_sha256(path)
    assert resolved["fit_flags"][-4:] == ["--zero_point_flag_table", str(path), "--zero_point_flag_threshold", "0.2"]


def test_launcher_parse_args_validates(monkeypatch, tmp_path):
    import hpc_scripts.sfitlc as sfitlc

    path = tmp_path / "cells.parquet"
    _cells().to_parquet(path)
    base = ["sfitlc.py", "--fit", "stone"]
    monkeypatch.setattr(sys, "argv", [*base, "--zero-point-flag-table", str(path)])
    args = sfitlc.parse_args()
    assert args.zero_point_flag_table == str(path.resolve()) and args.zero_point_flag_threshold is None
    for extra, message in (
        (["--zero-point-flag-threshold", "0.2"], "requires --zero-point-flag-table"),
        (["--zero-point-flag-table", str(tmp_path / "missing")], "not found"),
        (["--zero-point-flag-table", str(path), "--load_stone_lcs"], "load_stone_lcs"),
        (["--zero_point_flag_table", str(path)], "instead of passing"),
    ):
        monkeypatch.setattr(sys, "argv", [*base, *extra])
        with pytest.raises(SystemExit):
            sfitlc.parse_args()


def test_merge_reloads_with_the_fits_table(tmp_path):
    from qvc.light_curve.merge_results import _zero_point_reload_kwargs

    assert _zero_point_reload_kwargs([{"object_id": "A"}, {"object_id": "B", "zero_point_flag_table": np.nan}]) == {}
    path = tmp_path / "cells.parquet"
    _cells().to_parquet(path)
    row = {"zero_point_flag_table": str(path).encode(), "zero_point_flag_threshold": 0.2,
           "zero_point_flag_table_sha256": file_sha256(path)}
    kwargs = _zero_point_reload_kwargs([dict(row, object_id="A"), dict(row, object_id="B")])
    assert kwargs["zero_point_flag_threshold"] == 0.2
    pd.testing.assert_frame_equal(kwargs["zero_point_cells"], validate_zero_point_cells(_cells()))
    with pytest.raises(ValueError, match="different zero-point"):
        _zero_point_reload_kwargs([dict(row, object_id="A"), {"object_id": "B"}])
    with pytest.raises(ValueError, match="sha256 differs"):
        _zero_point_reload_kwargs([dict(row, zero_point_flag_table_sha256="0" * 64)])
