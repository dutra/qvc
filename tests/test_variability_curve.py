"""Stacked variability curve and predicted eta: synthetic truth, inputs, never-fail hook, CLI switch, and a
regression test against the one-off figure (skipped when its data are absent)."""
import argparse
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from qvc.hubble import hubble_fit
from qvc.hubble import variability_curve as vc

ETA = -0.8
X3000 = np.log10(3000.0)


def truth(x, bump=0.0):
    """True stacked curve in log10 rest wavelength: a power law (slope ETA) plus an optional bump at 2000 A."""
    g = lambda v: bump * np.exp(-0.5 * ((v - np.log10(2000.0)) / 0.06) ** 2)  # noqa: E731
    return ETA * (x - X3000) + g(x) - g(X3000)


def synthetic(n=1500, seed=0, bump=0.0, eta_noise=0.15, z_fixed=None):
    """Light-curve file and amplitude table whose kept-flux amplitude follows truth(); the per-quasar level rises
    with redshift (so the per-quasar normalization matters) and the u band of 40% of quasars is noise dominated
    and biased high (so the noise cut matters)."""
    rng = np.random.default_rng(seed)
    ids = np.array([str(5000 + i) for i in range(n)])
    z = rng.uniform(0.45, 3.0, n) if z_fixed is None else np.full(n, float(z_fixed))
    level = -1.0 + 0.3 * (z - 1.5) + rng.normal(0, 0.1, n)
    lc = {"object_id": ids, "z": z, "fit_ok": np.ones(n, bool), "eta_sigma": ETA + rng.normal(0, eta_noise, n) + 1.0 * (np.random.default_rng(seed + 99).uniform(size=n) < 0.1),  # 10% outliers
          "eta_sigma_err": np.full(n, 0.28)}
    amps = []
    noisy_u = rng.uniform(size=n) < 0.4
    for b in vc.BANDS:
        x = np.log10(vc.LAM_EFF[b] / (1 + z))
        host = rng.uniform(0, 0.1, n)
        # line light with a feature at a fixed rest wavelength (1900 A): the kept fraction bends with wavelength,
        # so a wrong sign or a missing kept-flux term bends the stacked curve
        line = rng.uniform(0.05, 0.4, n) + 1.5 * np.exp(-0.5 * ((x - np.log10(1900.0)) / 0.04) ** 2)
        cont = (1 - host) / (1 + line)
        keep = cont + 0.5 * (1 - host - cont)
        amp = level + truth(x, bump) + np.log10(keep) + rng.normal(0, 0.03, n)
        noisy = noisy_u if b == "u" else np.zeros(n, bool)
        amp = amp + 0.3 * noisy
        nv = np.where(noisy, 0.2 ** 2, 1e-4)
        amps.append(pd.DataFrame({"object_id": ids, "band": b, "Fbar": 1.0, "var_obs": 10 ** (2 * amp) + nv, "nv_cal": nv}))
        lc[f"psf_agn_fraction_lc_posterior_{b}"] = keep
        lc[f"f_AGN_psf_{b}"] = 1 - host
        lc[f"f_cont_psf_{b}"] = cont
    return lc, pd.concat(amps, ignore_index=True)


def write_inputs(tmp_path, drop=(), **kw):
    lc, amps = synthetic(**kw)
    lc_path = tmp_path / "lc.h5"
    with h5py.File(lc_path, "w") as h:
        for k, v in lc.items():
            if k not in drop:
                h[k] = v.astype("S") if v.dtype.kind == "U" else v
        h["psf_fraction"] = np.array([b"continuum"] * len(lc["z"]))
    amps.to_parquet(tmp_path / "amps.parquet")
    return lc_path, tmp_path / "amps.parquet"


@pytest.fixture(scope="module")
def power_law(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("powerlaw")
    lc_path, amps_path = write_inputs(tmp)
    logs = []
    res = vc.run_diagnostic(lc_path, tmp / "diag", amplitudes=amps_path, log=logs.append)
    assert res is not None, logs
    return res, tmp, logs


def test_power_law_curve_is_straight(power_law):
    res, _, _ = power_law
    C = pd.DataFrame(res["curve"])
    assert np.max(np.abs(C.S - truth(C.log_lam))) < 0.03
    for d in res["local_slopes"]:
        if d["n_points"] > 2:
            assert d["slope"] == pytest.approx(ETA, abs=max(0.12, 3 * d["slope_err"])), d


def test_power_law_predicts_the_input_eta_and_a_flat_trend(power_law):
    res, _, _ = power_law
    pred = np.asarray(res["predicted_eta"]["eta_pred"])
    assert len(pred) > 100 and np.max(np.abs(pred - ETA)) < 0.03
    assert abs(res["eta_vs_z_slope"]) < 3 * res["eta_vs_z_slope_err"] + 0.01
    assert res["rms_pred_minus_fit"] < 0.04
    EB = pd.DataFrame(res["eta_bins"])
    assert np.allclose(EB.eta_posterior_width_med.dropna(), 0.28) and EB.n.sum() == res["n_quasars_eta"]
    big = EB[EB.n >= 50]
    assert len(big) >= 8 and np.max(np.abs(big.eta_fit - ETA)) < 0.06


def test_outputs_label_and_log(power_law):
    res, tmp, logs = power_law
    assert res["reduction"] == "psf fraction: continuum; line light kept: 50%"
    pdf = tmp / "diag" / "variability_curve_and_band_windows_postcut.pdf"
    assert pdf.stat().st_size > 1000
    saved = json.loads((tmp / "diag" / "variability_curve_and_band_windows_postcut.json").read_text())
    assert saved["reduction"] == res["reduction"]
    assert len(saved["amplitudes_file"]["sha256"]) == 64 and len(saved["light_curve_file"]["sha256"]) == 64
    assert saved["band_bar_redshifts"] == sorted(saved["band_bar_redshifts"])
    assert len(logs) == 1 and res["reduction"] in logs[0] and "eta vs z slope" in logs[0]


def test_bump_gives_the_expected_predicted_wiggle(tmp_path):
    lc_path, amps_path = write_inputs(tmp_path, bump=0.12)
    res = vc.run_diagnostic(lc_path, tmp_path / "d", amplitudes=amps_path, log=lambda s: None)
    assert res is not None
    P = pd.DataFrame(res["predicted_eta"])
    expect = np.array([np.polyfit(np.log10(vc.PIVOTS / (1 + z)), truth(np.log10(vc.PIVOTS / (1 + z)), 0.12), 1)[0]
                       for z in P.z])
    assert np.max(np.abs(P.eta_pred - expect)) < 0.05
    assert np.ptp(P.eta_pred) > 0.2 and np.ptp(expect) > 0.2
    assert res["rms_pred_minus_fit"] > 0.05


def test_object_selection(tmp_path):
    lc_path, amps_path = write_inputs(tmp_path, n=800)
    ids = [str(5000 + i) for i in range(0, 800, 2)]
    res = vc.run_diagnostic(lc_path, tmp_path / "d", ids, amplitudes=amps_path, log=lambda s: None)
    assert res is not None and res["n_quasars_eta"] == 400 and res["n_objects_requested"] == 400


@pytest.mark.parametrize("case", ["no_amplitudes", "no_eta", "no_kept_fraction", "too_few"])
def test_missing_inputs_warn_and_return_none(tmp_path, case):
    drop = {"no_eta": ("eta_sigma",), "no_kept_fraction": ("psf_agn_fraction_lc_posterior_g",)}.get(case, ())
    lc_path, amps_path = write_inputs(tmp_path, drop=drop, n=300)
    ids = ["5000", "5001", "5002"] if case == "too_few" else None
    amps = tmp_path / "missing.parquet" if case == "no_amplitudes" else amps_path
    logs = []
    assert vc.run_diagnostic(lc_path, tmp_path / "out", ids, amplitudes=amps, log=logs.append) is None
    assert len(logs) == 1 and "WARNING: skipped" in logs[0] and "not affected" in logs[0]
    assert not (tmp_path / "out" / "variability_curve_and_band_windows_postcut.pdf").exists()


def _hook_args(**kw):
    a = dict(skip_variability_curve_plot=False, skip_plots=False, compare_sigma_only=False, only_sna=False,
             subtraction_test_amplitudes=None, subtraction_test_first_fit=None, subtraction_test_continuum="auto")
    a.update(kw)
    return SimpleNamespace(**a)


def test_hook_as_called_by_hubble_fit(tmp_path):
    lc_path, amps_path = write_inputs(tmp_path, n=800)
    df_agn = pd.DataFrame({"object_id": [5000 + i for i in range(0, 800, 3)], "z": 1.0})
    before = df_agn.copy()
    res = hubble_fit.run_variability_curve_stage(
        _hook_args(agn_data_filepath=str(lc_path), subtraction_test_amplitudes=str(amps_path)),
        df_agn, tmp_path / "plots" / "run", log=lambda s: None)
    assert res is not None and res["n_objects_requested"] == len(df_agn)
    assert (tmp_path / "plots" / "run" / "diagnostics" / "variability_curve_and_band_windows_postcut.pdf").exists()
    pd.testing.assert_frame_equal(df_agn, before)
    assert hubble_fit.run_variability_curve_stage(
        _hook_args(agn_data_filepath=str(lc_path), skip_variability_curve_plot=True), df_agn, tmp_path / "x") is None
    assert not (tmp_path / "x").exists()
    logs = []
    assert hubble_fit.run_variability_curve_stage(
        _hook_args(agn_data_filepath=str(tmp_path / "nope.h5")), df_agn, tmp_path / "y", log=logs.append) is None
    assert "WARNING" in logs[0]


def test_cli_switch_declared():
    tree = ast.parse(Path(hubble_fit.__file__).read_text())
    parser = argparse.ArgumentParser()
    found = False
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"
                and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--skip-variability-curve-plot"):
            found = True
            exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(n)], type_ignores=[])), "<cli>", "exec"),
                 {"parser": parser})
    assert found
    assert parser.parse_args([]).skip_variability_curve_plot is False
    assert parser.parse_args(["--skip_variability_curve_plot"]).skip_variability_curve_plot is True


# ------------------------------------------------------------------ regression against the one-off figure
REF_ROOT = Path(os.environ.get("QVC_SUBTRACTION_TEST_REFERENCE_ROOT", vc.REPO_ROOT if hasattr(vc, "REPO_ROOT") else "."))
GS = REF_ROOT / "plots/diagnostics/gold_sample"
TB = GS / "drop_blue_bands_refit/tables"
LONG = GS / "cosmology_independent_z_diagnostics/tables/long.parquet"
NOV21 = REF_ROOT / ("results/data/nov21_1110am_chisq_slb_psffractioncontinuum_onlyspectrafiltered_iters3disk3_"
                    "svi10000lr0003w1000s500_specaug31w500s500_6ad2c7f_chisq.h5")


@pytest.mark.skipif(not (LONG.exists() and NOV21.exists() and (TB / "s14_predicted_vs_fitted.csv").exists()),
                    reason="reference data of variability_curve_and_band_windows_apj not present")
def test_reproduces_reference_curve_and_eta():
    """The one-off's inputs (as-fitted nov21 reduction, kept flux f_cont, its 4,747 quasars)."""
    L = pd.read_parquet(LONG)
    L["object_id"] = L.object_id.astype(str).str.strip()
    with h5py.File(NOV21, "r") as h:
        oid = np.array([x.decode().strip() for x in h["object_id"][:]])
        E = pd.DataFrame({"eta": h["eta_sigma"][:], "eta_err": h["eta_sigma_err"][:]}, index=oid)
    E = E[~E.index.duplicated(keep="last")]
    O = L.drop_duplicates("object_id")[["object_id", "z"]].join(E, on="object_id").reset_index(drop=True)
    Q = L.assign(amp=L.y, f_keep=L.f_cont, lam_rest=L.band.map(vc.LAM_EFF) / (1 + L.z))
    Q = Q.assign(q=np.where((Q.relnoise < 0.15) & np.isfinite(Q.amp), Q.amp - np.log10(Q.f_keep), np.nan))
    res = vc.compute(Q[np.isfinite(Q.q)], O)

    C = pd.DataFrame(res["curve"]).assign(k=lambda d: np.round(d.log_lam, 6))
    ref = pd.read_csv(TB / "s09_composite.csv")
    ref = ref[(ref.treatment == "B") & np.isfinite(ref.S)].assign(k=lambda d: np.round(d.log_lam, 6))
    m = C.merge(ref, on="k", suffixes=("", "_ref"), validate="one_to_one")
    assert len(m) == len(C) == len(ref) == 25
    np.testing.assert_allclose(m.S, m.S_ref, atol=1e-12)
    assert (m.n == m.n_ref).all()
    assert 0.8 < np.median(m.S_err / m.S_err_ref) < 1.2         # bootstrap errors: other random stream

    LS = pd.DataFrame(res["local_slopes"])
    LR = pd.read_csv(TB / "s14_local_slopes.csv")
    lm = LS.merge(LR[LR.treatment == "B"], on=["lam_lo", "lam_hi"], suffixes=("", "_ref"))
    assert len(lm) == 7
    main = lm[(lm.lam_lo == 1390) & (lm.lam_hi == 2110)].iloc[0]
    assert main.slope == pytest.approx(-1.05, abs=0.02) and main.slope_ref == pytest.approx(-1.0456, abs=1e-4)
    np.testing.assert_allclose(lm.slope, lm.slope_ref, atol=0.05)   # weights 1/S_err differ slightly

    EB = pd.DataFrame(res["eta_bins"])
    PV = pd.read_csv(TB / "s14_predicted_vs_fitted.csv")
    em = EB.merge(PV, on=["z_lo", "z_hi"], suffixes=("", "_ref"), validate="one_to_one")
    assert len(em) == 13
    np.testing.assert_allclose(em.n, em.n_ref)
    np.testing.assert_allclose(em.z_med, em.z_med_ref, atol=1e-12)
    np.testing.assert_allclose(em.eta_fit, em.eta_fit_ref, atol=1e-12)
    np.testing.assert_allclose(em.eta_pred, em.eta_pred_B, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(em.eta_fit_err, em.eta_fit_err_ref, rtol=0.25)


def test_narrow_redshift_range_gives_no_trend(tmp_path):
    lc_path, amps_path = write_inputs(tmp_path, n=600, z_fixed=1.5)
    logs = []
    res = vc.run_diagnostic(lc_path, tmp_path / "d", amplitudes=amps_path, log=logs.append)
    assert res is not None
    assert np.isnan(res["eta_vs_z_slope"]) and np.isnan(res["eta_vs_z_slope_err"])
    assert res["eta_vs_z_note"].startswith("not measured")
    assert "eta vs z not measured" in logs[-1]
    assert res["n_bins_compared"] <= 1
    if res["n_bins_compared"] == 1:
        assert "one bin only" in logs[-1]
    saved = json.loads((tmp_path / "d" / "variability_curve_and_band_windows_postcut.json").read_text())
    assert saved["eta_vs_z_note"] == res["eta_vs_z_note"]


def test_label_states_both_samples(power_law):
    res, _, _ = power_law
    assert res["n_quasars_curve"] <= res["n_quasars_eta"]


def test_diagnostics_leave_backend_rcparams_and_pyplot_alone(tmp_path):
    """Run both diagnostics in a fresh process whose backend is not Agg: the backend, the rcParams and the
    pyplot figure list are unchanged, and the modules never import pyplot (so no window can open)."""
    import subprocess
    import sys
    import test_subtraction_test as tst
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    lc1, amps1, sec1 = tst.write_inputs(tmp_path / "a", n=600)
    lc2, amps2 = write_inputs(tmp_path / "b", n=600)
    code = f"""
import sys, matplotlib
before_backend = matplotlib.get_backend()
before_rc = dict(matplotlib.rcParams)
from qvc.hubble import subtraction_test as st, variability_curve as vc
r1 = st.run_diagnostic({str(lc1)!r}, {str(tmp_path / 'o1')!r}, amplitudes={str(amps1)!r}, second_fit={str(sec1) + '#new'!r}, log=lambda s: None)
r2 = vc.run_diagnostic({str(lc2)!r}, {str(tmp_path / 'o2')!r}, amplitudes={str(amps2)!r}, log=lambda s: None)
assert r1 is not None and r2 is not None
assert matplotlib.get_backend() == before_backend, (matplotlib.get_backend(), before_backend)
changed = [k for k, v in before_rc.items() if str(matplotlib.rcParams[k]) != str(v)]
assert not changed, changed
assert "matplotlib.pyplot" not in sys.modules
print("OK", before_backend)
"""
    env = dict(os.environ, MPLBACKEND="svg", PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "OK svg"
