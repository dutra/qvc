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


# --- review additions ------------------------------------------------------------


def _install(monkeypatch, sdss_rows, *, ps1_rows=(), ztf_rows=()):
    """Fake S82 inputs: sdss_rows / ps1_rows / ztf_rows are (object, band, time, mag)."""
    objects = sorted({row[0] for row in (*sdss_rows, *ps1_rows, *ztf_rows)})
    ps1_ids = {obj: 1000 + i for i, obj in enumerate(objects)}
    cat = pd.DataFrame({"idx": range(len(objects)), "objectId": objects, "RA": [RA.get(o, 10.5) for o in objects],
                        "DEC": 0.0, "ps1objID": [float(ps1_ids[o]) for o in objects]})
    for band in "griz":
        cat[f"sdss_{band}_qg"] = 0.0
        cat[f"ps1_{band}_qg"] = 0.0

    def frame(rows, columns):
        return pd.DataFrame(rows, columns=columns)

    sdss = frame([(o, multiband_generate_lc.filters[b], t, m, 0.03) for o, b, t, m in sdss_rows],
                 ["objectId", "filterID", "mjd", "psMag", "psMagErr_p3"])
    ps1 = frame([(ps1_ids[o], multiband_generate_lc.filters[b], t, m, 0.03) for o, b, t, m in ps1_rows],
                ["ps1objID", "filterID", "obsTime", "psfMag", "psfMagErr_p3"])
    ztf = frame([(ps1_ids[o], multiband_generate_lc.filters[b], t, m, 0.03) for o, b, t, m in ztf_rows],
                ["ps1objID", "filterID", "mjd", "mag", "magerr_p3"])
    frames = {"Catalog.parquet": cat, "dr16s82_sdssLCRaw.parquet": sdss,
              "dr16s82_ps1LCRaw.parquet": ps1, "dr16s82_ZuberLCRaw.parquet": ztf}

    def fake_resolve(path):
        if path in multiband_generate_lc.SEEING_SIDECARS.values():
            raise FileNotFoundError(path)
        return path

    monkeypatch.setattr(multiband_generate_lc, "resolve_qvc_data_path", fake_resolve)
    monkeypatch.setattr(multiband_generate_lc.pd, "read_parquet", lambda path: frames[str(path).rsplit("/", 1)[-1]].copy())


CLOUDY_NIGHT = 51001
CLOUD_CELL = pd.DataFrame({"band": ["g"], "night": [CLOUDY_NIGHT], "rab": [5], "cell_off": [0.3], "cell_n": [20]})


def _clip_order_rows(*, drop_cloudy=False):
    # Eight g epochs on nights 51001-51008 and an outlier X on 51009: X has
    # exactly the eight neighbours the clip needs within 30 days, one of them
    # on the cloudy night 51001.
    rows = [("A", "g", night + 0.2, 20.0 + 0.01 * (-1) ** night) for night in range(51001, 51009)]
    rows.append(("A", "g", 51009.2, 25.0))
    if drop_cloudy:
        rows = [row for row in rows if int(row[2]) != CLOUDY_NIGHT]
    return rows


def test_flagged_epochs_are_dropped_before_the_outlier_clip(monkeypatch):
    _install(monkeypatch, _clip_order_rows())
    (off,) = multiband_generate_lc.concat_light_curves()
    assert 51009.2 not in off["times"]["g"]  # off: the clip removes X
    (on,) = multiband_generate_lc.concat_light_curves(zero_point_cells=CLOUD_CELL)
    # On: the cloudy epoch goes first, X then has 7 neighbours and is kept by the
    # clip -- exactly what the analysis's hand-filtered input gives.
    assert 51009.2 in on["times"]["g"] and CLOUDY_NIGHT + 0.2 not in on["times"]["g"]
    _install(monkeypatch, _clip_order_rows(drop_cloudy=True))
    (hand,) = multiband_generate_lc.concat_light_curves()
    _assert_same({k: v for k, v in on.items() if not k.startswith("zero_point")}, hand)


def test_only_sdss_epochs_are_flagged(monkeypatch):
    sdss = [("A", "g", night + 0.2, 20.0) for night in range(51000, 51004)]
    ps1 = [("A", "g", CLOUDY_NIGHT + 0.3, 20.1)]
    ztf = [("A", "g", CLOUDY_NIGHT + 0.4, 20.2)]
    _install(monkeypatch, sdss, ps1_rows=ps1, ztf_rows=ztf)
    (on,) = multiband_generate_lc.concat_light_curves(zero_point_cells=CLOUD_CELL)
    kept = dict(zip(on["times"]["g"], on["surveys"]["g"]))
    assert kept[CLOUDY_NIGHT + 0.3] == "ps1" and kept[CLOUDY_NIGHT + 0.4] == "ztf"
    assert CLOUDY_NIGHT + 0.2 not in kept
    assert on["zero_point_flagged_epochs_dropped_g"] == 1


def test_epochs_without_a_cell_are_counted_and_band_losses_reported(monkeypatch, capsys):
    rows = [("A", "g", night + 0.2, 20.0) for night in range(51000, 51010)]
    rows += [("A", "r", CLOUDY_NIGHT + 0.25, 20.0)]  # r only on the cloudy night
    _install(monkeypatch, rows)
    cells = pd.concat([CLOUD_CELL, pd.DataFrame({"band": ["g"] * 3, "night": [51000, 51002, 51003],
                                                 "rab": [5] * 3, "cell_off": [0.0] * 3, "cell_n": [20] * 3})])
    (on,) = multiband_generate_lc.concat_light_curves(zero_point_cells=cells)
    out = capsys.readouterr().out
    # g on 51004-51009 and r on 51001 have no cell of their own.
    assert on["zero_point_epochs_without_cell"] == 7
    assert "7 of 11 SDSS epochs (63.6%) have no cell" in out
    assert on["zero_point_flagged_epochs_dropped_r"] == 1 and len(on["times"]["r"]) == 0
    assert "--zero_point_flag_table removed all 1 epochs of band r" in out
    # The same table covering every epoch: no warning.
    full = pd.concat([cells, pd.DataFrame({"band": ["g"] * 6 + ["r"], "night": [*range(51004, 51010), CLOUDY_NIGHT],
                                           "rab": 5, "cell_off": 0.0, "cell_n": 20})])
    (on,) = multiband_generate_lc.concat_light_curves(zero_point_cells=full)
    assert on["zero_point_epochs_without_cell"] == 0 and "have no cell" not in capsys.readouterr().out.split("SDSS epochs on")[0]


def test_object_losing_every_epoch_is_skipped_with_a_message(monkeypatch, capsys):
    _install(monkeypatch, [("A", "g", CLOUDY_NIGHT + 0.2, 20.0), ("B", "g", 51005.2, 20.0)])
    objs = multiband_generate_lc.concat_light_curves(zero_point_cells=CLOUD_CELL)
    assert [o["object_id"] for o in objs] == ["B"]
    assert "Object A: --zero_point_flag_table removed all its epochs" in capsys.readouterr().out


def test_merge_variability_reload_uses_the_fits_table(monkeypatch, tmp_path):
    from qvc.light_curve import fit_light_curves, merge_results

    path = tmp_path / "cells.parquet"
    _cells().to_parquet(path)
    calls = []

    def fake_concat(**kwargs):
        calls.append(kwargs)
        return [{"object_id": "A", "times": {}}]

    monkeypatch.setattr(multiband_generate_lc, "concat_light_curves", fake_concat)
    monkeypatch.setattr(fit_light_curves, "make_lc", lambda *a, **k: {"variability_x": 1.0})
    rows = [{"object_id": "A", "z": 1.0, "zero_point_flag_table": str(path), "zero_point_flag_threshold": 0.2,
             "zero_point_flag_table_sha256": file_sha256(path)}]
    merge_results.attach_variability_metrics(rows)
    pd.testing.assert_frame_equal(calls[-1]["zero_point_cells"], validate_zero_point_cells(_cells()))
    assert calls[-1]["zero_point_flag_threshold"] == 0.2
    merge_results.attach_variability_metrics([{"object_id": "A", "z": 1.0}])
    assert "zero_point_cells" not in calls[-1]


def test_launcher_makes_the_table_path_absolute(monkeypatch, tmp_path):
    import hpc_scripts.sfitlc as sfitlc

    _cells().to_parquet(tmp_path / "cells.parquet")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["sfitlc.py", "--fit", "stone", "--zero-point-flag-table", "cells.parquet"])
    args = sfitlc.parse_args()
    assert Path(args.zero_point_flag_table).is_absolute()
    assert args.zero_point_flag_table == str((tmp_path / "cells.parquet").resolve())


@pytest.mark.parametrize("extra", [["--sdss_source", "macleod"], ["--sdss_source=macleod"],
                                   ["--load_nearby_lc_csv", "x"]])
def test_launcher_refuses_sources_the_fitter_refuses(monkeypatch, tmp_path, capsys, extra):
    import hpc_scripts.sfitlc as sfitlc

    _cells().to_parquet(tmp_path / "cells.parquet")
    monkeypatch.setattr(sys, "argv", ["sfitlc.py", "--fit", "stone", "--zero-point-flag-table",
                                      str(tmp_path / "cells.parquet"), *extra])
    with pytest.raises(SystemExit):
        sfitlc.parse_args()
    assert "dr16s82 SDSS epochs" in capsys.readouterr().err


def test_launcher_resume_carries_the_setting(monkeypatch, tmp_path):
    path = tmp_path / "cells.parquet"
    _cells().to_parquet(path)
    script = _script(_launcher_args(resume="prefix", zero_point_flag_table=str(path), zero_point_flag_threshold=None),
                     monkeypatch)
    # The resumed fits get the same table, so the fitter's resume guard applies to them.
    assert "--resume" in script and "--zero_point_flag_table" in script


def test_fitter_records_the_setting_per_row(monkeypatch, tmp_path):
    from qvc.light_curve import fit_light_curves

    path = tmp_path / "cells.parquet"
    _cells().to_parquet(path)
    captured = {}

    def fake_populate(objs, **kwargs):
        captured["objs"] = objs
        raise _Stop

    monkeypatch.setattr(fit_light_curves, "concat_light_curves", lambda **kw: [{"object_id": "A"}])
    monkeypatch.setattr(fit_light_curves, "populate_sdss_fields", fake_populate)
    monkeypatch.setattr(sys, "argv", ["fit_light_curves", "--filter_object_id", "A",
                                      "--zero_point_flag_table", str(path), "--zero_point_flag_threshold", "0.2"])
    with pytest.raises(_Stop):
        fit_light_curves.main()
    (obj,) = captured["objs"]
    assert obj["zero_point_flag_table_sha256"] == file_sha256(path)
    assert obj["zero_point_flag_threshold"] == 0.2
    assert obj["zero_point_flag_table"] == str(path.resolve())


def test_saved_samples_record_the_setting_and_resume_checks_it(monkeypatch, tmp_path):
    import h5py

    from qvc.light_curve import multiband_fit_utils
    from qvc.light_curve.zero_point_flags import (
        ZERO_POINT_OFF, ZeroPointResumeMismatch, check_zero_point_resume, zero_point_setting,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(multiband_fit_utils, "prefix", "p", raising=False)
    monkeypatch.setattr(multiband_fit_utils, "suffix", "s", raising=False)
    on = zero_point_setting("a" * 64, 0.1)
    multiband_fit_utils.save_obj_samples_to_hdf5({"x": np.zeros(3)}, "A", zero_point_flags=on)
    multiband_fit_utils.save_obj_samples_to_hdf5({"x": np.zeros(3)}, "B")
    with h5py.File(tmp_path / "results/samples/p/B_s.h5", "r") as handle:
        assert not any(key.startswith("zero_point") for key in handle.attrs)  # off: file as before
    _, meta_on = multiband_fit_utils.load_obj_samples_from_hdf5("A", return_metadata=True)
    _, meta_off = multiband_fit_utils.load_obj_samples_from_hdf5("B", return_metadata=True)
    check_zero_point_resume(meta_on, on, "A")
    check_zero_point_resume(meta_off, ZERO_POINT_OFF, "B")  # existing off runs keep resuming
    for meta, current in ((meta_off, on), (meta_on, ZERO_POINT_OFF), (meta_on, zero_point_setting("a" * 64, 0.2)),
                          (meta_on, zero_point_setting("b" * 64, 0.1))):
        with pytest.raises(ZeroPointResumeMismatch, match="--resume of object"):
            check_zero_point_resume(meta, current, "A")


def test_fitter_resume_refuses_a_changed_setting(monkeypatch, tmp_path):
    """The guard runs in the fitter's --resume path and is not swallowed by the per-object handler."""
    from qvc.light_curve import fit_light_curves
    from qvc.light_curve.zero_point_flags import ZeroPointResumeMismatch

    _install(monkeypatch, [("A", band, 51000.2 + 3 * k, 20.0 + 0.05 * np.sin(k)) for band in "ugriz" for k in range(40)])
    monkeypatch.setattr(fit_light_curves, "populate_sdss_fields", lambda objs, **kw: [dict(o, z=1.0) for o in objs])
    monkeypatch.setattr(fit_light_curves, "load_obj_samples_from_hdf5", lambda oid, return_metadata=False: ({}, {}))
    path = tmp_path / "cells.csv"  # read_parquet is faked for the S82 inputs
    _cells().to_csv(path, index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["fit_light_curves", "--filter_object_id", "A", "--resume",
                                      "--zero_point_flag_table", str(path)])
    with pytest.raises(ZeroPointResumeMismatch):
        fit_light_curves.main()
