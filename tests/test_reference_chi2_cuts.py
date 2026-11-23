"""Tier-1 chi-square cuts on the stored or the fixed-definition reference chi-square."""

import numpy as np
import pandas as pd
import pytest

from qvc.hubble import chi2_statistic_comparison as comparison
from qvc.hubble import hubble_utils
from qvc.hubble.cuts import (
    ALLOW_FITTED_NOISE_CHI2_ENV,
    ALLOW_MISSING_REFERENCE_CHI2_ENV,
    REFERENCE_FLOOR_ENV,
    AGN_TIER1_FIT_QUALITY_CUTS,
    CHI2_CUT_STATISTIC_ENV,
    REFERENCE_CHI2_CUT_COLUMNS,
    chi2_cut_statistics,
    resolve_chi2_cut_columns,
)

from test_hubble_load_plots import _minimal_agn_frame, _patch_load_agn_plotters

STORED = list(REFERENCE_CHI2_CUT_COLUMNS)
REFERENCE = list(REFERENCE_CHI2_CUT_COLUMNS.values())
OTHER = ["loo_chi2_eff", "m_2500_dereddened_rhat"]
SED, SPEC, JOINT = STORED


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        *CHI2_CUT_STATISTIC_ENV.values(),
        ALLOW_FITTED_NOISE_CHI2_ENV,
        ALLOW_MISSING_REFERENCE_CHI2_ENV,
        *(env for env, _default in REFERENCE_FLOOR_ENV.values()),
    ):
        monkeypatch.delenv(name, raising=False)


def _columns(cuts):
    return [column for column, _lower, _upper in cuts]


def _used(choices):
    return {cut: choice["column"] for cut, choice in choices.items()}


def test_statistic_defaults_and_validation(monkeypatch):
    assert chi2_cut_statistics() == {cut: "reference" for cut in STORED}
    monkeypatch.setenv("QVC_CUT_SED_CHI2_STATISTIC", "Stored")
    assert chi2_cut_statistics()[SED] == "stored"
    monkeypatch.setenv("QVC_CUT_JOINT_CHI2_STATISTIC", "fitted")
    with pytest.raises(ValueError, match="QVC_CUT_JOINT_CHI2_STATISTIC"):
        chi2_cut_statistics()
    with pytest.raises(ValueError, match="Unknown chi-square cut"):
        chi2_cut_statistics({"loo_chi2_eff": "stored"})


def test_reference_columns_replace_all_three_with_same_thresholds():
    cuts, choices = resolve_chi2_cut_columns(
        STORED + REFERENCE + ["frac_jitter"] + OTHER, AGN_TIER1_FIT_QUALITY_CUTS
    )
    assert _used(choices) == REFERENCE_CHI2_CUT_COLUMNS
    stored_bounds = {c: (lo, hi) for c, lo, hi in AGN_TIER1_FIT_QUALITY_CUTS}
    inverse = {v: k for k, v in REFERENCE_CHI2_CUT_COLUMNS.items()}
    for column, lower, upper in cuts:
        assert (lower, upper) == stored_bounds[inverse.get(column, column)]
    assert not set(STORED) & set(_columns(cuts))
    assert len(cuts) == len(AGN_TIER1_FIT_QUALITY_CUTS)


def test_each_cut_can_use_either_statistic_independently():
    columns = STORED + REFERENCE + ["frac_jitter"] + OTHER
    cuts, choices = resolve_chi2_cut_columns(
        columns,
        AGN_TIER1_FIT_QUALITY_CUTS,
        statistics={SED: "stored"},
    )
    assert _used(choices) == {SED: SED, SPEC: REFERENCE_CHI2_CUT_COLUMNS[SPEC], JOINT: REFERENCE_CHI2_CUT_COLUMNS[JOINT]}
    # Stored SED in a fitted-noise catalogue needs no override (definition unchanged) ...
    assert choices[SED]["statistic"] == "stored"
    # ... stored spectroscopy/joint do.
    for cut in (SPEC, JOINT):
        with pytest.raises(ValueError, match="fitted spectral noise model"):
            resolve_chi2_cut_columns(columns, AGN_TIER1_FIT_QUALITY_CUTS, statistics={cut: "stored"})
        _cuts, allowed = resolve_chi2_cut_columns(
            columns, AGN_TIER1_FIT_QUALITY_CUTS, statistics={cut: "stored"}, allow_fitted_noise_chi2=True
        )
        assert allowed[cut]["column"] == cut


def test_catalogue_without_jitter_uses_stored_columns_silently():
    cuts, choices = resolve_chi2_cut_columns(STORED + OTHER, AGN_TIER1_FIT_QUALITY_CUTS)
    assert cuts == list(AGN_TIER1_FIT_QUALITY_CUTS)
    assert _used(choices) == {cut: cut for cut in STORED}
    _cuts, stored = resolve_chi2_cut_columns(
        STORED + OTHER, AGN_TIER1_FIT_QUALITY_CUTS, statistics={cut: "stored" for cut in STORED}
    )
    assert _used(stored) == {cut: cut for cut in STORED}


@pytest.mark.parametrize("marker", ["frac_jitter", "add_jitter", "frac_fe_jitter"])
def test_fitted_noise_catalogue_without_reference_raises_for_spectral_cuts(marker):
    with pytest.raises(ValueError, match="fitted spectral noise model") as error:
        resolve_chi2_cut_columns(STORED + [marker] + OTHER, AGN_TIER1_FIT_QUALITY_CUTS)
    assert SPEC in str(error.value) and JOINT in str(error.value)
    assert "sed_reduced_chi2 (" not in str(error.value)


def test_fitted_noise_catalogue_can_be_allowed_explicitly(monkeypatch):
    columns = STORED + ["frac_jitter"] + OTHER
    cuts, choices = resolve_chi2_cut_columns(
        columns, AGN_TIER1_FIT_QUALITY_CUTS, allow_fitted_noise_chi2=True
    )
    assert cuts == list(AGN_TIER1_FIT_QUALITY_CUTS)
    assert "fitted-noise" in choices[SPEC]["reason"]
    assert "SED definition unchanged" in choices[SED]["reason"]
    monkeypatch.setenv(ALLOW_FITTED_NOISE_CHI2_ENV, "1")
    resolve_chi2_cut_columns(columns, AGN_TIER1_FIT_QUALITY_CUTS)
    monkeypatch.setenv(ALLOW_FITTED_NOISE_CHI2_ENV, "maybe")
    with pytest.raises(ValueError, match=ALLOW_FITTED_NOISE_CHI2_ENV):
        resolve_chi2_cut_columns(columns, AGN_TIER1_FIT_QUALITY_CUTS)


def test_partial_reference_columns_raise():
    with pytest.raises(ValueError, match="only some reference"):
        resolve_chi2_cut_columns(STORED + REFERENCE[:1], AGN_TIER1_FIT_QUALITY_CUTS)


# --- comparison --------------------------------------------------------------


def _comparison_frame():
    stored = np.array([0.5, 0.9, 1.05, 1.15, 1.3, 2.0, np.nan, 0.8])
    reference = np.array([0.4, 1.2, 0.95, 1.0, 1.05, 1.5, 0.7, np.nan])
    frame = pd.DataFrame({"z": [0.5, 0.8, 1.2, 1.6, 2.1, 2.6, 3.0, 0.7]})
    for cut in STORED:
        frame[cut] = stored
        frame[REFERENCE_CHI2_CUT_COLUMNS[cut]] = reference
    return frame


def test_compare_cut_statistics_counts_and_threshold():
    frame = _comparison_frame()
    rows = comparison.compare_cut_statistics(frame, {cut: 1.1 for cut in STORED})
    assert [row["cut"] for row in rows] == STORED
    row = rows[1]
    stored_pass = np.array([1, 1, 1, 0, 0, 0, 0, 1], dtype=bool)  # NaN fails
    reference_pass = np.array([1, 0, 1, 1, 1, 0, 1, 0], dtype=bool)
    assert row["pass_stored"] == stored_pass.sum() == 4
    assert row["pass_reference"] == reference_pass.sum() == 5
    assert row["pass_both"] == 2
    assert row["only_stored"] == 2
    assert row["only_reference"] == 3
    assert row["n_both_finite"] == 6
    # Among the 6 both-finite rows, 3 pass stored <= 1.1; the 3rd-smallest
    # reference value there is 1.0.
    assert row["matched_reference_threshold"] == pytest.approx(1.0)
    assert -1.0 <= row["spearman"] <= 1.0


def test_matched_threshold_and_spearman_edge_cases():
    assert np.isnan(comparison.matched_reference_threshold([2.0, 3.0], [1.0, 1.0], 1.1))
    assert comparison.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert comparison.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(comparison.spearman([1, 2], [1, 2]))


def test_compare_selection_reports_all_stored_all_reference_and_used():
    frame = _comparison_frame()
    frame["loo_chi2_eff"] = 1.0
    tier1 = [(cut, None, 1.1) for cut in STORED] + [("loo_chi2_eff", None, 1.05)]
    used = [(SED, None, 1.1)] + [(REFERENCE_CHI2_CUT_COLUMNS[c], None, 1.1) for c in (SPEC, JOINT)] + [
        ("loo_chi2_eff", None, 1.05)
    ]
    rows = {row["configuration"]: row for row in comparison.compare_selection(frame, tier1, used)}
    assert rows["all stored"]["kept"] == 4
    assert rows["all reference"]["kept"] == 5
    assert rows["used"]["kept"] == 2
    assert "sed_reduced_chi2_ref" in rows["all reference"]["chi2_columns"]


def test_split_comparison_by_redshift_spectrograph_and_snr():
    frame = _comparison_frame()
    frame["mjd"] = [52000, 56000, 52000, 56000, 52000, 56000, 52000, 56000]
    frame["spectrum_snr_median"] = np.arange(8, dtype=float)
    split = comparison.split_comparison(frame, {c: 1.1 for c in STORED}, snr_column="spectrum_snr_median")
    assert set(split["split"]) == {"all", "redshift", "spectrograph", "spectrum_snr_median tercile"}
    assert set(split.loc[split["split"] == "spectrograph", "group"]) == {"sdss", "boss"}
    assert set(split.loc[split["split"].str.contains("tercile"), "group"]) == {"low", "mid", "high"}
    spec = split[(split["cut"] == SPEC)]
    by_split = spec.groupby("split")["pass_stored"].sum()
    assert by_split["redshift"] == by_split["all"] == by_split["spectrograph"] == 4


def test_comparison_command_writes_tables(tmp_path, capsys):
    import h5py

    frame = _comparison_frame()
    frame["object_id"] = [str(i) for i in range(len(frame))]
    path = tmp_path / "cat.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("catalog")
        for column in frame.columns:
            values = frame[column].to_numpy()
            group.create_dataset(column, data=values.astype("S") if values.dtype == object else values)
    info = tmp_path / "info.csv"
    pd.DataFrame({"object_id": frame["object_id"], "spectrograph": ["sdss", "boss"] * 4}).to_csv(info, index=False)
    out = tmp_path / "out"
    comparison.main([str(path), "--output-dir", str(out), "--info-csv", str(info)])
    printed = capsys.readouterr().out
    assert "stored vs reference" in printed
    assert "No S/N column found" in printed
    table = pd.read_csv(out / "chi2_statistic_comparison.csv")
    assert list(table["cut"]) == STORED
    split = pd.read_csv(out / "chi2_statistic_comparison_by_split.csv")
    assert {"sdss", "boss"} <= set(split["group"])


def test_comparison_command_requires_both_statistics(tmp_path):
    path = tmp_path / "cat.csv"
    pd.DataFrame({"z": [1.0], SPEC: [1.0]}).to_csv(path, index=False)
    with pytest.raises(SystemExit, match="no pair"):
        comparison.main([str(path)])


# --- load_agn_data ------------------------------------------------------------


def _load(tmp_path, monkeypatch, frame, tier="1"):
    source_path = tmp_path / "agn.h5"
    source_path.touch()
    monkeypatch.setattr(
        hubble_utils, "read_quasars_from_hdf5_flat", lambda *_a, **_k: frame.copy()
    )
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda value: value)
    _patch_load_agn_plotters(monkeypatch)
    return hubble_utils.load_agn_data(
        source_path,
        magnitude_convention="dereddened",
        spectra_fit_h5=None,
        cut_tier=tier,
        plot_diagnostics=False,
        plot_path=str(tmp_path / "plots"),
        cut_report_path=tmp_path / "cut_summary.txt",
    )


def _reference_frame():
    frame = _minimal_agn_frame(n=4)
    frame["object_id"] = ["keep", "stored-high", "ref-high", "sed-ref-high"]
    frame["frac_jitter"] = 0.004
    for stored, reference in REFERENCE_CHI2_CUT_COLUMNS.items():
        frame[reference] = frame[stored]
    frame.loc[1, SPEC] = 10.0
    frame.loc[2, REFERENCE_CHI2_CUT_COLUMNS[SPEC]] = 10.0
    frame.loc[3, REFERENCE_CHI2_CUT_COLUMNS[SED]] = 10.0
    _add_floors(frame)
    return frame


def _add_floors(frame, spectrum=0.08, photometry=0.10, agn=0.15):
    frame["reference_spectrum_floor"] = spectrum
    frame["reference_photometry_floor"] = photometry
    frame["reference_agn_floor"] = agn
    return frame


def test_load_agn_data_cuts_on_reference_columns_and_reports_comparison(tmp_path, monkeypatch):
    selected, _parent = _load(tmp_path, monkeypatch, _reference_frame())
    assert selected["object_id"].tolist() == ["keep", "stored-high"]
    report = (tmp_path / "cut_summary.txt").read_text()
    assert "spectroscopy_reduced_chi2_ref" in report
    assert "stored vs reference" in report
    assert "all stored" in report and "all reference" in report and "used" in report
    cut_table = pd.read_csv(tmp_path / "plots" / "cuts" / "tier1_chi2_statistic_comparison.csv")
    spec = cut_table.set_index("cut").loc[SPEC]
    assert (spec["pass_stored"], spec["pass_reference"], spec["pass_both"]) == (3, 3, 2)
    selection = pd.read_csv(tmp_path / "plots" / "cuts" / "tier1_chi2_selection_comparison.csv")
    kept = dict(zip(selection["configuration"], selection["kept"]))
    assert kept == {"all stored": 3, "all reference": 2, "used": 2}


def test_load_agn_data_stored_sed_statistic_needs_no_override(tmp_path, monkeypatch):
    monkeypatch.setenv("QVC_CUT_SED_CHI2_STATISTIC", "stored")
    selected, _parent = _load(tmp_path, monkeypatch, _reference_frame())
    assert selected["object_id"].tolist() == ["keep", "stored-high", "sed-ref-high"]
    kept = dict(
        pd.read_csv(tmp_path / "plots" / "cuts" / "tier1_chi2_selection_comparison.csv")[
            ["configuration", "kept"]
        ].values
    )
    assert kept["used"] == 3


def test_load_agn_data_stored_spectral_statistic_needs_override(tmp_path, monkeypatch):
    monkeypatch.setenv("QVC_CUT_SPECTROSCOPY_CHI2_STATISTIC", "stored")
    with pytest.raises(ValueError, match="fitted spectral noise model"):
        _load(tmp_path, monkeypatch, _reference_frame())
    monkeypatch.setenv(ALLOW_FITTED_NOISE_CHI2_ENV, "1")
    selected, _parent = _load(tmp_path, monkeypatch, _reference_frame())
    assert selected["object_id"].tolist() == ["keep", "ref-high"]


def test_load_agn_data_rejects_fitted_noise_catalogue_without_reference(tmp_path, monkeypatch):
    frame = _minimal_agn_frame(n=2)
    frame["frac_jitter"] = 0.004
    with pytest.raises(ValueError, match="reference"):
        _load(tmp_path, monkeypatch, frame)


def test_load_agn_data_keeps_stored_columns_for_old_catalogues(tmp_path, monkeypatch):
    frame = _minimal_agn_frame(n=2)
    frame["object_id"] = ["keep", "stored-high"]
    frame.loc[1, SPEC] = 10.0
    selected, _parent = _load(tmp_path, monkeypatch, frame)
    assert selected["object_id"].tolist() == ["keep"]
    assert not (tmp_path / "plots" / "cuts" / "tier1_chi2_statistic_comparison.csv").exists()


# --- review fixes ---------------------------------------------------------------


def _ref_values_frame(n=3):
    frame = pd.DataFrame({"z": np.linspace(0.5, 2.0, n), "frac_jitter": 0.004})
    for stored, reference in REFERENCE_CHI2_CUT_COLUMNS.items():
        frame[stored] = 1.0
        frame[reference] = 1.0
    return _add_floors(frame)


def test_noise_model_marker_alone_marks_a_fitted_noise_catalogue():
    frame = pd.DataFrame({c: [1.0] for c in STORED})
    frame["spectral_noise_model"] = ["fitted_jitter"]
    with pytest.raises(ValueError, match="fitted spectral noise model"):
        resolve_chi2_cut_columns(frame, AGN_TIER1_FIT_QUALITY_CUTS)
    frame["spectral_noise_model"] = ["fixed_floor"]
    _cuts, choices = resolve_chi2_cut_columns(frame, AGN_TIER1_FIT_QUALITY_CUTS)
    assert _used(choices) == {c: c for c in STORED}
    frame["spectral_noise_model"] = ["mystery"]
    with pytest.raises(ValueError, match="Unknown spectral_noise_model"):
        resolve_chi2_cut_columns(frame, AGN_TIER1_FIT_QUALITY_CUTS)


def test_missing_reference_values_with_finite_stored_raise_and_are_counted():
    frame = _ref_values_frame(3)
    frame.loc[1, REFERENCE_CHI2_CUT_COLUMNS[SPEC]] = np.nan
    with pytest.raises(ValueError, match="no reference value") as error:
        resolve_chi2_cut_columns(frame, AGN_TIER1_FIT_QUALITY_CUTS)
    assert "'spectroscopy_reduced_chi2': 1" in str(error.value)
    _cuts, choices = resolve_chi2_cut_columns(
        frame, AGN_TIER1_FIT_QUALITY_CUTS, allow_missing_reference_chi2=True
    )
    assert choices[SPEC]["missing_reference"] == 1
    assert choices[SED]["missing_reference"] == 0
    # A row where both are missing is a plain chi-square failure, not a mismatch.
    frame = _ref_values_frame(3)
    frame.loc[1, [SPEC, REFERENCE_CHI2_CUT_COLUMNS[SPEC]]] = np.nan
    resolve_chi2_cut_columns(frame, AGN_TIER1_FIT_QUALITY_CUTS)


def test_reference_floors_must_match_and_are_recorded(monkeypatch):
    frame = _ref_values_frame(3)
    _cuts, choices = resolve_chi2_cut_columns(frame, AGN_TIER1_FIT_QUALITY_CUTS)
    assert choices["reference_floors"] == {
        "reference_spectrum_floor": 0.08,
        "reference_photometry_floor": 0.10,
        "reference_agn_floor": 0.15,
    }
    frame.loc[2, "reference_spectrum_floor"] = 0.05
    with pytest.raises(ValueError, match="reference_spectrum_floor: 1 row"):
        resolve_chi2_cut_columns(frame, AGN_TIER1_FIT_QUALITY_CUTS)
    frame["reference_spectrum_floor"] = 0.05
    monkeypatch.setenv("QVC_CUT_REFERENCE_SPECTRUM_FLOOR", "0.05")
    _cuts, choices = resolve_chi2_cut_columns(frame, AGN_TIER1_FIT_QUALITY_CUTS)
    assert choices["reference_floors"]["reference_spectrum_floor"] == 0.05
    missing = _ref_values_frame(2).drop(columns=["reference_agn_floor"])
    with pytest.raises(ValueError, match="reference_agn_floor column missing"):
        resolve_chi2_cut_columns(missing, AGN_TIER1_FIT_QUALITY_CUTS)
    # Floors are not checked when every cut uses the stored statistic.
    resolve_chi2_cut_columns(
        missing.drop(columns=["frac_jitter"]),
        AGN_TIER1_FIT_QUALITY_CUTS,
        statistics={c: "stored" for c in STORED},
    )


def _spectra_csv(path, ids, *, reference):
    rows = {
        "object_id": ids,
        "fit_ok": [True] * len(ids),
        "fit_backend": ["jaxsedfit_joint"] * len(ids),
        "m_2500_dereddened": [20.0] * len(ids),
        "m_2500_dereddened_err": [0.1] * len(ids),
        "m_2500_attenuated_model": [20.1] * len(ids),
        "m_2500_attenuated_model_err": [0.1] * len(ids),
        "frac_jitter": [0.004] * len(ids),
    }
    for stored, ref in REFERENCE_CHI2_CUT_COLUMNS.items():
        rows[stored] = [1.0] * len(ids)
        if reference:
            rows[ref] = [1.0] * len(ids)
    if reference:
        rows.update(
            reference_spectrum_floor=[0.08] * len(ids),
            reference_photometry_floor=[0.10] * len(ids),
            reference_agn_floor=[0.15] * len(ids),
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_mixed_spectra_versions_are_not_dropped_silently(tmp_path):
    new = _spectra_csv(tmp_path / "new.csv", ["a", "b"], reference=True)
    old = _spectra_csv(tmp_path / "old.csv", ["c"], reference=False)
    merged = hubble_utils.populate_spectra_fit(pd.DataFrame({"object_id": ["a", "b", "c"]}), [new, old])
    with pytest.raises(ValueError, match="no reference value"):
        resolve_chi2_cut_columns(merged, AGN_TIER1_FIT_QUALITY_CUTS)


def test_stale_light_curve_copies_do_not_steer_the_choice(tmp_path, capsys):
    spectra = _spectra_csv(tmp_path / "old_fixed.csv", ["a"], reference=False)
    frame = pd.read_csv(spectra).drop(columns=["frac_jitter"])
    frame.to_csv(spectra, index=False)
    light_curve = pd.DataFrame(
        {
            "object_id": ["a"],
            "spectroscopy_reduced_chi2_ref": [0.5],
            "frac_jitter": [0.01],
            "reference_spectrum_floor": [0.08],
        }
    )
    merged = hubble_utils.populate_spectra_fit(light_curve, [spectra])
    assert "spectroscopy_reduced_chi2_ref" not in merged.columns
    assert "frac_jitter" not in merged.columns
    assert "Dropping spectral fit-quality columns" in capsys.readouterr().out
    _cuts, choices = resolve_chi2_cut_columns(merged, AGN_TIER1_FIT_QUALITY_CUTS)
    assert _used(choices) == {c: c for c in STORED}


def test_joint_definitions_compare_stored_reference_and_mixed():
    frame = pd.DataFrame(
        {
            "joint_reduced_chi2": [1.0, 1.3, 1.1, 2.0],
            "joint_reduced_chi2_ref": [0.9, 1.0, 1.25, 1.5],
            "sed_chi2": [20.0, 30.0, 25.0, 40.0],
            "sed_n_eff": [20.0, 20.0, 20.0, 20.0],
            "spectroscopy_chi2_ref": [80.0, 90.0, 100.0, 160.0],
            "spectroscopy_n_ref": [80.0, 80.0, 80.0, 80.0],
        }
    )
    mixed = comparison.mixed_joint_reduced_chi2(frame)
    np.testing.assert_allclose(mixed, [1.0, 1.2, 1.25, 2.0])
    rows = {r["definition"]: r for r in comparison.compare_joint_definitions(frame, 1.2)}
    assert rows["stored"]["passing"] == 2
    assert rows["reference (fixed-floor SED + reference spectrum)"]["passing"] == 2
    assert rows["stored SED + reference spectrum"]["passing"] == 2
    assert rows["stored"]["spearman_vs_stored"] == pytest.approx(1.0)
    assert comparison.mixed_joint_reduced_chi2(frame.drop(columns=["sed_n_eff"])) is None
