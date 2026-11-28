"""Flat-or-tilted subtraction test: estimator on a synthetic sample, file readers, never-fail hook, CLI flags,
and a regression test against the reference figure (skipped when its data are absent)."""
import argparse
import ast
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from qvc.hubble import hubble_fit
from qvc.hubble import subtraction_test as st

R_TRUE = 0.5
GRAD = dict(u=1.0, g=0.5, r=0.0, i=-0.3)   # how the line fraction and the nuisance change with z, per band


def synthetic(n=2000, seed=0, first_noise=0.6, second_noise=0.3, outlier_frac=0.03, same_model=False,
              line_trend=2.0, nuisance=1.0):
    """Light curves subtracted with the right responsivity (R_TRUE) but with a noisy spectral fit.

    * The amplitude follows the true kept flux f_cont + R_TRUE (f_AGN - f_cont), times a continuum law with a
      strong redshift-dependent colour; the line fraction also changes with redshift, differently per band, so
      without the fixed-redshift comparison (cells of band pair and z) the slope is biased.
    * The run's kept fraction and x come from a noisy first fit (errors of 60% in D), so its errors enter both
      axes: the plain regression is biased, the instrumental variable (independent second fit) is not.
    * A few per cent of bands, more often where lines are strong, have grossly wrong amplitudes; trimming and the
      robust weights keep them from inflating the error.
    """
    rng = np.random.default_rng(seed)
    ids = np.array([str(1000 + i) for i in range(n)])
    z = rng.uniform(0.45, 2.0, n)
    lc = {"object_id": ids, "z": z, "fit_ok": np.ones(n, bool)}
    amps, second = [], {"object_id": ids}
    for b in st.BANDS:
        lam = st.LAM_EFF[b] / (1 + z)
        host = rng.uniform(0.0, 0.1, n)
        line = rng.uniform(0.05, 0.45, n) * np.exp(line_trend * GRAD[b] * (z - 1.2))
        cont = (1 - host) / (1 + line)
        agn = 1 - host
        d_true = np.log10(agn / cont)
        cont1 = agn / 10 ** (d_true * (1 + rng.normal(0, first_noise, n)))
        cont2 = cont1 if same_model else agn / 10 ** (d_true * (1 + rng.normal(0, second_noise, n)))
        keep_true = cont + R_TRUE * (agn - cont)
        keep_run = cont1 + R_TRUE * (agn - cont1)
        log_a = -1.0 + 0.3 * np.log10(lam / 3000.0) + nuisance * GRAD[b] * (z - 1.2)
        amp = log_a + np.log10(keep_true) + rng.normal(0, 0.03, n)
        bad = rng.uniform(size=n) < outlier_frac * line / line.mean()
        amp = amp + bad * rng.uniform(0.4, 1.0, n) * (1 if b in "gi" else -1)
        nv = 1e-4
        amps.append(pd.DataFrame({"object_id": ids, "band": b, "Fbar": 1.0, "var_obs": 10 ** (2 * amp) + nv,
                                  "nv_cal": nv}))
        lc[f"psf_agn_fraction_lc_posterior_{b}"] = keep_run
        lc[f"f_AGN_psf_{b}"] = agn
        lc[f"f_cont_psf_{b}"] = cont1
        second[f"f_AGN_{b}"] = agn
        second[f"f_cont_{b}"] = cont2
    return lc, pd.concat(amps, ignore_index=True), pd.DataFrame(second)


def write_inputs(tmp_path, drop=(), **kw):
    lc, amps, second = synthetic(**kw)
    lc_path = tmp_path / "lc.h5"
    with h5py.File(lc_path, "w") as h:
        for k, v in lc.items():
            if k in drop:
                continue
            h[k] = v.astype("S") if v.dtype.kind == "U" else v
        h["psf_fraction"] = np.array([b"continuum"] * len(lc["z"]))
    amps_path, second_path = tmp_path / "amps.parquet", tmp_path / "second.parquet"
    amps.to_parquet(amps_path)
    second.assign(spec_src="new").to_parquet(second_path)
    return lc_path, amps_path, second_path


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("synthetic")
    lc_path, amps_path, second_path = write_inputs(tmp)
    logs = []
    res = st.run_diagnostic(lc_path, tmp / "diag", amplitudes=amps_path, second_fit=f"{second_path}#new",
                            log=logs.append)
    assert res is not None, logs
    return res, tmp, logs


def test_right_subtraction_is_flat(result):
    res, _, _ = result
    P = res["panels"]["run"]
    assert abs(P["slope"]) < 0.08, P          # plain regression, no or coarse z cells give > +0.1
    assert P["flat"] is True and P["verdict"].startswith("flat within")
    assert P["err"] < 0.04, P                 # without trimming and robust weights the error is ~0.065
    assert res["kept_line_share_median"] == pytest.approx(R_TRUE, abs=0.02)
    assert res["same_model"] is False


def test_limits_are_tilted_with_the_right_words(result):
    res, _, _ = result
    c, d = res["panels"]["constant"], res["panels"]["disc"]
    assert 0.45 < c["slope"] < 0.65 and c["flat"] is False and c["verdict"] == "tilted: too much removed"
    assert -0.55 < d["slope"] < -0.35 and d["flat"] is False and d["verdict"] == "tilted: too little removed"


def test_outputs_and_log(result):
    res, tmp, logs = result
    pdf = tmp / "diag" / "subtraction_test_flat_or_tilted_postcut.pdf"
    assert pdf.stat().st_size > 1000
    saved = json.loads((tmp / "diag" / "subtraction_test_flat_or_tilted_postcut.json").read_text())
    assert saved["panels"]["run"]["slope"] == pytest.approx(res["panels"]["run"]["slope"])
    assert len(saved["panels"]["run"]["bins"]["x"]) == st.N_BINS
    for key in ("light_curve_file", "amplitudes_file", "second_fit"):
        rec = saved[key]
        assert len(rec["sha256"]) == 64 and rec["size_bytes"] > 0 and rec["mtime"]
    assert saved["second_fit"]["spec_src"] == "new"
    assert any("this run (psf fraction: continuum; line light kept: 50%): slope" in line and "flat within" in line for line in logs)


def test_object_selection_and_hdf5_second_fit(tmp_path):
    lc_path, amps_path, second_path = write_inputs(tmp_path, n=800)
    sec = pd.read_parquet(second_path)
    h5_path = tmp_path / "second.h5"
    with h5py.File(h5_path, "w") as h:
        g = h.create_group("catalog")
        g["object_id"] = sec.object_id.to_numpy().astype("S")
        for b in st.BANDS:
            g[f"f_AGN_psf_{b}"] = sec[f"f_AGN_{b}"].to_numpy()
            g[f"f_cont_psf_{b}"] = sec[f"f_cont_{b}"].to_numpy()
    ids = [str(1000 + i) for i in range(0, 800, 2)]
    res = st.run_diagnostic(lc_path, tmp_path / "d", ids, amplitudes=amps_path, second_fit=h5_path, log=lambda s: None)
    assert res is not None
    assert res["n_quasars"] <= 400 and res["second_fit"]["spec_src"] is None
    logs = []
    assert st.run_diagnostic(lc_path, tmp_path / "e", amplitudes=amps_path, second_fit=f"{h5_path}#new",
                             log=logs.append) is None
    assert "applies only to parquet" in logs[0]


@pytest.mark.parametrize("how", ["identical_fractions", "light_curve_file"])
def test_same_model_is_not_tested(tmp_path, how):
    lc_path, amps_path, second_path = write_inputs(tmp_path, n=800, same_model=(how == "identical_fractions"))
    second = f"{second_path}#new" if how == "identical_fractions" else lc_path
    logs = []
    res = st.run_diagnostic(lc_path, tmp_path / "d", amplitudes=amps_path, second_fit=second, log=logs.append)
    assert res is not None and res["same_model"] is True
    for P in res["panels"].values():
        assert P["flat"] is None and P["verdict"] == st.NOT_TESTED
        assert P["slope_kind"].startswith("uncorrected")
    saved = json.loads((tmp_path / "d" / "subtraction_test_flat_or_tilted_postcut.json").read_text())
    assert saved["panels"]["run"]["flat"] is None and saved["panels"]["run"]["verdict"] == st.NOT_TESTED
    line = [s for s in logs if "this run" in s][0].split(" -> ")[0]
    assert "uncorrected slope" in line and st.NOT_TESTED in line and "tilted" not in line and "flat within" not in line


@pytest.mark.parametrize("case", ["no_amplitudes", "no_second_fit", "no_fractions", "too_few", "bad_source"])
def test_missing_inputs_warn_and_return_none(tmp_path, case):
    drop = tuple(f"f_cont_psf_{b}" for b in st.BANDS) if case == "no_fractions" else ()
    lc_path, amps_path, second_path = write_inputs(tmp_path, drop=drop, n=400)
    kw = dict(amplitudes=amps_path, second_fit=f"{second_path}#new")
    ids = None
    if case == "no_amplitudes":
        kw["amplitudes"] = tmp_path / "missing.parquet"
    elif case == "no_second_fit":
        kw["second_fit"] = tmp_path / "missing.parquet"
    elif case == "too_few":
        ids = ["1000", "1001", "1002"]
    elif case == "bad_source":
        kw["second_fit"] = f"{second_path}#nosuchsource"
    logs = []
    assert st.run_diagnostic(lc_path, tmp_path / "out", ids, log=logs.append, **kw) is None
    assert len(logs) == 1 and "WARNING: skipped" in logs[0] and "not affected" in logs[0]
    assert not (tmp_path / "out" / "subtraction_test_flat_or_tilted_postcut.pdf").exists()


def test_unexpected_error_is_caught(tmp_path):
    bad = tmp_path / "lc.h5"
    bad.write_text("not an hdf5 file")
    logs = []
    assert st.run_diagnostic(bad, tmp_path / "out", log=logs.append) is None
    assert "WARNING: skipped after an error" in logs[0]


def test_default_tables_resolve_from_the_data_directory(tmp_path, monkeypatch):
    lc_path, amps_path, second_path = write_inputs(tmp_path, n=600)
    data = tmp_path / "datadir" / "subtraction_test"
    data.mkdir(parents=True)
    os.replace(amps_path, data / Path(st.DEFAULT_AMPLITUDES).name)
    os.replace(second_path, data / Path(st.DEFAULT_SECOND_FIT.split("#")[0]).name)
    monkeypatch.setenv("QVC_DATA_DIR", str(tmp_path / "datadir"))
    monkeypatch.chdir(tmp_path)
    res = st.run_diagnostic(lc_path, tmp_path / "d", log=lambda s: None)
    assert res is not None
    assert Path(res["amplitudes_file"]["path"]).parent == data.resolve()
    assert Path(res["second_fit"]["path"]).parent == data.resolve() and res["second_fit"]["spec_src"] == "new"


def _hook_args(**kw):
    a = dict(skip_subtraction_test_plot=False, skip_plots=False, compare_sigma_only=False, only_sna=False,
             subtraction_test_amplitudes=None, subtraction_test_second_fit=None, subtraction_test_first_fit=None,
             subtraction_test_continuum="auto")
    a.update(kw)
    return SimpleNamespace(**a)


def test_hook_as_called_by_hubble_fit(tmp_path):
    lc_path, amps_path, second_path = write_inputs(tmp_path, n=800)
    df_agn = pd.DataFrame({"object_id": [1000 + i for i in range(0, 800, 3)], "z": 1.0})   # ints, as after loading
    before = df_agn.copy()
    logs = []
    res = hubble_fit.run_subtraction_test_stage(
        _hook_args(agn_data_filepath=str(lc_path), subtraction_test_amplitudes=str(amps_path),
                   subtraction_test_second_fit=f"{second_path}#new"),
        df_agn, tmp_path / "plots" / "run", log=logs.append)
    assert res is not None and res["n_objects_requested"] == len(df_agn)
    assert (tmp_path / "plots" / "run" / "diagnostics" / "subtraction_test_flat_or_tilted_postcut.pdf").exists()
    pd.testing.assert_frame_equal(df_agn, before)
    assert hubble_fit.run_subtraction_test_stage(
        _hook_args(agn_data_filepath=str(lc_path), skip_subtraction_test_plot=True), df_agn, tmp_path / "x") is None
    assert not (tmp_path / "x").exists()
    logs = []
    assert hubble_fit.run_subtraction_test_stage(
        _hook_args(agn_data_filepath=str(tmp_path / "nope.h5")), df_agn, tmp_path / "y", log=logs.append) is None
    assert "WARNING" in logs[0]


def test_cli_switches_declared():
    tree = ast.parse(Path(hubble_fit.__file__).read_text())
    wanted = {"--skip-subtraction-test-plot", "--subtraction-test-amplitudes", "--subtraction-test-second-fit",
              "--subtraction-test-first-fit", "--subtraction-test-continuum"}
    parser = argparse.ArgumentParser()
    found = set()
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"
                and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value in wanted):
            found.add(n.args[0].value)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(n)], type_ignores=[])), "<cli>", "exec"),
                 {"parser": parser})
    assert found == wanted
    a = parser.parse_args([])
    assert a.skip_subtraction_test_plot is False
    assert a.subtraction_test_amplitudes is None and a.subtraction_test_second_fit is None
    assert a.subtraction_test_first_fit is None and a.subtraction_test_continuum == "auto"
    assert parser.parse_args(["--subtraction-test-continuum", "rebuild"]).subtraction_test_continuum == "rebuild"
    assert parser.parse_args(["--skip_subtraction_test_plot"]).skip_subtraction_test_plot is True


# ------------------------------------------------------------------ regression against the reference figure
REF_ROOT = Path(os.environ.get("QVC_SUBTRACTION_TEST_REFERENCE_ROOT", st.REPO_ROOT))
GS = REF_ROOT / "plots/diagnostics/gold_sample"
REF_CSV = GS / "responsivity_plot_for_colin/figures/fig1_simple_v3_slopes.csv"
REF_SCRIPTS = GS / "cosmology_independent_z_diagnostics/independent_check/scripts"
REF_INPUTS = GS / "cosmology_independent_z_diagnostics/independent_check/tables/inputs.parquet"


@pytest.mark.skipif(not (REF_CSV.exists() and REF_INPUTS.exists() and (REF_SCRIPTS / "s6_species.py").exists()),
                    reason="reference data of 3_subtraction_test_flat_or_tilted not present")
def test_reproduces_reference_slopes():
    """Same inputs as the one-off figure (its own table builder, read-only): the three slopes to all digits."""
    sys.path.insert(0, str(REF_SCRIPTS))
    try:
        from s6_species import COMP, species_table
    finally:
        sys.path.remove(str(REF_SCRIPTS))
    lit = dict(MgII=0.45, CIV=0.40, CIII=0.40, Balmer=0.50, UVo=0.45, narrow=0.45, bc=0.75)
    S = species_table("std")
    xs = 1.0
    for k in COMP:
        R = np.where(S.lam < 3500, 0.20, 0.40) if k == "feii" else lit[k]
        xs = xs + R * S["x_" + k]
    T = pd.DataFrame(dict(object_id=S.object_id, band=S.band, z=S.z, f_cont=S.fB, f_AGN=S.fA,
                          f_keep=S.fB * np.clip(xs, 1.0, S.fA / S.fB), amp=S.y,
                          relnoise=np.sqrt(S.nv_cal) / S.Fbar, D_second=S.D_new, lam=S.lam))
    T["D"] = np.log10(T.f_AGN / T.f_cont)
    res = st.compute(T)
    ref = pd.read_csv(REF_CSV).set_index("R")
    for key, r in (("constant", "0.0"), ("run", "lit"), ("disc", "1.0")):
        assert res["panels"][key]["slope"] == pytest.approx(ref.loc[r, "slope_iv"], rel=1e-10, abs=1e-12)
        assert res["panels"][key]["err"] == pytest.approx(ref.loc[r, "err"], rel=1e-10)
    assert res["n_quasars"] == 1975 and res["n_pairs"] == 4447


# ------------------------------------------------------------------ continuum source of the light-curve file
def _variant_file(tmp_path, variant, n=800, r=0.5):
    """Synthetic light-curve file with components that close (f_cont + f_line + f_feii + f_bc = f_AGN), then:
    clean | orig (f_cont rewritten to the kept flux, true continuum in armX_f_cont_orig) | rewritten (f_cont and the
    kept fraction = cont + r (AGN - cont), no orig column) | lines_high (normal, f_line 30% too high)."""
    lc_path, amps_path, second_path = write_inputs(tmp_path, n=n)
    with h5py.File(lc_path, "r+") as h:
        for b in st.BANDS:
            agn, cont = h[f"f_AGN_psf_{b}"][()], h[f"f_cont_psf_{b}"][()]
            other = agn - cont
            h[f"f_line_psf_{b}"] = (1.3 if variant == "lines_high" else 1.0) * 0.7 * other
            h[f"f_feii_psf_{b}"] = 0.2 * other
            h[f"f_bc_psf_{b}"] = 0.1 * other
            if variant in ("orig", "rewritten"):
                kept = cont + r * other
                del h[f"f_cont_psf_{b}"], h[f"psf_agn_fraction_lc_posterior_{b}"]
                h[f"f_cont_psf_{b}"] = kept
                h[f"psf_agn_fraction_lc_posterior_{b}"] = kept
                if variant == "orig":
                    h[f"armX_f_cont_orig_psf_{b}"] = cont
    st._CONTINUUM_WARNED.clear()
    return lc_path, amps_path, second_path


@pytest.mark.parametrize("variant,r,source,expect_used,expect_warn,expect_share", [
    ("clean", 0.5, "auto", "stored", False, 0.5),
    ("orig", 0.3, "auto", "orig", False, 0.3),
    ("rewritten", 0.5, "auto", "stored", True, 0.0),
    ("rewritten", 0.07, "auto", "stored", True, 0.0),
    ("lines_high", 0.5, "auto", "stored", True, 0.5),
    ("rewritten", 0.07, "rebuild", "rebuild", False, 0.07),
    ("orig", 0.3, "stored", "stored", True, 0.0),
])
def test_continuum_source_is_explicit_and_warned(tmp_path, variant, r, source, expect_used, expect_warn, expect_share):
    lc_path, amps_path, second_path = _variant_file(tmp_path, variant, r=r)
    logs = []
    res = st.run_diagnostic(lc_path, tmp_path / "d", amplitudes=amps_path, second_fit=f"{second_path}#new",
                            continuum=source, log=logs.append)
    assert res is not None, logs
    c = res["continuum"]
    assert c["source_requested"] == source and c["source_used"] == expect_used
    assert set(c["closure_stored"]) == set(st.BANDS)
    assert bool(c["warning"]) is expect_warn
    assert res["kept_line_share_median"] == pytest.approx(expect_share, abs=0.01)
    warns = [x for x in logs if "does not close" in x]
    assert len(warns) == (1 if expect_warn else 0)
    if expect_warn:
        assert "--subtraction-test-first-fit" in warns[0] and "--subtraction-test-continuum" in warns[0]
        assert all(f"{b} " in warns[0] for b in st.BANDS)
        assert res["reduction"].endswith("(unverified)")
    else:
        assert "unverified" not in res["reduction"]
    saved = json.loads((tmp_path / "d" / "subtraction_test_flat_or_tilted_postcut.json").read_text())
    assert saved["continuum"]["source_used"] == expect_used and saved["reduction"] == res["reduction"]


def test_continuum_warning_printed_once_for_both_diagnostics(tmp_path):
    from qvc.hubble import variability_curve as vc
    lc_path, amps_path, second_path = _variant_file(tmp_path, "rewritten", r=0.4)
    with h5py.File(lc_path, "r+") as h:
        h["eta_sigma"] = np.full(len(h["z"]), -0.8)
    logs = []
    st.run_diagnostic(lc_path, tmp_path / "d", amplitudes=amps_path, second_fit=f"{second_path}#new", log=logs.append)
    r2 = vc.run_diagnostic(lc_path, tmp_path / "d", amplitudes=amps_path, log=logs.append)
    assert sum("does not close" in x for x in logs) == 1
    assert r2 is not None and r2["continuum"]["warning"] and r2["reduction"].endswith("(unverified)")


def test_orig_source_missing_and_first_fit(tmp_path):
    lc_path, amps_path, second_path = _variant_file(tmp_path, "rewritten", r=0.4)
    logs = []
    assert st.run_diagnostic(lc_path, tmp_path / "d", amplitudes=amps_path, second_fit=f"{second_path}#new",
                             continuum="orig", log=logs.append) is None
    assert "orig" in logs[0] and "WARNING: skipped" in logs[0]
    # the first fit (here: the true continuum) replaces the file's fractions; no closure warning
    sec = pd.read_parquet(second_path)
    with h5py.File(lc_path, "r") as h:
        first = pd.DataFrame({"object_id": sec.object_id})
        for b in st.BANDS:
            first[f"f_AGN_{b}"] = h[f"f_AGN_psf_{b}"][()]
            first[f"f_cont_{b}"] = h[f"f_AGN_psf_{b}"][()] - h[f"f_line_psf_{b}"][()] - h[f"f_feii_psf_{b}"][()] - h[f"f_bc_psf_{b}"][()]
    first.to_parquet(tmp_path / "first.parquet")
    logs = []
    res = st.run_diagnostic(lc_path, tmp_path / "e", amplitudes=amps_path, second_fit=f"{second_path}#new",
                            first_fit=tmp_path / "first.parquet", log=logs.append)
    assert res is not None and res["continuum"]["source_used"] == "first spectral fit"
    assert not any("does not close" in x for x in logs) and "unverified" not in res["reduction"]
    assert res["kept_line_share_median"] == pytest.approx(0.4, abs=0.01)
