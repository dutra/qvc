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

from qvc.hubble.hubble_utils import (  # noqa: E402
    _load_spectra_fit_rows,
    load_completeness_parent_spectra_fit,
    populate_spectra_fit,
)


def _legacy_row(object_id="obj", **overrides):
    row = {
        "object_id": object_id,
        "fit_ok": True,
        "z": 1.2,
        "ra": 12.3,
        "dec": -0.4,
        "PL_slope": -1.6,
        "PL_slope_err": 0.12,
        "apparent_mag_2500": 20.8,
        "apparent_mag_2500_err": 0.08,
        "apparent_mag_2500_intrinsic": 20.3,
        "apparent_mag_2500_intrinsic_err": 0.10,
        "chi2_per_pixel": 1.15,
        "frac_host_5100": 0.25,
        "f_host_5100": 0.25,
        "result_dir": "results/first",
        "fig_dir": "plots/first",
        "output_path": "results/first",
        "f_PL": 0.8,
        "f_PL_err": 0.03,
    }
    row.update(overrides)
    return row


def _current_row(object_id="current"):
    return {
        "object_id": object_id,
        "fit_ok": True,
        "fit_backend": "jaxsedfit_joint",
        "fracAGN_5100_fit": 0.8,
        "fracAGN_5100_fit_err": 0.05,
        "m_2500_dereddened": 20.0,
        "m_2500_dereddened_err": 0.1,
        "m_2500_attenuated_model": 20.2,
        "m_2500_attenuated_model_err": 0.12,
        "pl_slope": -1.5,
        "pl_slope_err": 0.1,
    }


def test_populate_spectra_fit_normalizes_legacy_rows_and_skips_failed(tmp_path):
    path = tmp_path / "legacy.csv"
    failed = _legacy_row("failed", fit_ok=False, PL_slope="not-a-number")
    no_host = _legacy_row(
        "no-host",
        frac_host_5100=-1.0,
        f_host_5100=-1.0,
    )
    pd.DataFrame([_legacy_row(), no_host, failed]).to_csv(path, index=False)

    out = populate_spectra_fit(
        pd.DataFrame({"object_id": ["obj", "no-host", "not-matched"]}),
        [path],
    )

    assert out["object_id"].tolist() == ["obj", "no-host"]
    assert out["fit_backend"].tolist() == [
        "jaxqsofit_legacy",
        "jaxqsofit_legacy",
    ]
    np.testing.assert_allclose(out["fracAGN_5100_fit"], [0.75, 0.999])
    assert out["fracAGN_5100_fit_err"].isna().all()
    assert out.loc[0, "m_2500_dereddened"] == 20.3
    assert out.loc[0, "m_2500_dereddened_err"] == 0.10
    assert out.loc[0, "m_2500_attenuated_model"] == 20.8
    assert out.loc[0, "m_2500_attenuated_model_err"] == 0.08
    assert out.loc[0, "pl_slope"] == -1.6
    assert out.loc[0, "pl_slope_err"] == 0.12
    assert out.loc[0, "spectroscopy_reduced_chi2"] == 1.15
    assert out.loc[0, "alpha_lambda"] == -1.6
    assert out.loc[0, "alpha_lambda_err"] == 0.12
    assert np.isclose(out.loc[0, "alpha_nu"], -0.4)
    assert out.loc[0, "alpha_nu_err"] == 0.12
    assert out.loc[0, "f_PL"] == 0.8
    assert out.attrs["spectra_fit_schema"] == "jaxqsofit_legacy"
    assert LEGACY_ALIASES.isdisjoint(out.columns)


LEGACY_ALIASES = {
    "PL_slope",
    "PL_slope_err",
    "apparent_mag_2500",
    "apparent_mag_2500_err",
    "apparent_mag_2500_intrinsic",
    "apparent_mag_2500_intrinsic_err",
    "chi2_per_pixel",
}


def test_legacy_completeness_parent_uses_canonical_magnitudes(tmp_path):
    path = tmp_path / "legacy-parent.csv"
    pd.DataFrame([_legacy_row()]).to_csv(path, index=False)

    parent = load_completeness_parent_spectra_fit([path])

    assert parent["object_id"].tolist() == ["obj"]
    assert parent.loc[0, "m_2500_dereddened"] == 20.3
    assert parent.loc[0, "m_2500_attenuated_model"] == 20.8
    assert parent.attrs["spectra_fit_schema"] == "jaxqsofit_legacy"
    assert parent.attrs["completeness_parent_rule"] == (
        "all_fit_ok_legacy_jaxqsofit_rows_v1"
    )


def test_legacy_completeness_parent_rejects_negative_magnitude_error(tmp_path):
    path = tmp_path / "legacy-negative-error.csv"
    pd.DataFrame(
        [_legacy_row(apparent_mag_2500_intrinsic_err=-0.1)]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="must be non-negative"):
        load_completeness_parent_spectra_fit([path])


def test_legacy_provenance_only_duplicates_are_collapsed(tmp_path, capsys):
    path = tmp_path / "legacy-duplicates.csv"
    duplicate = _legacy_row(
        result_dir="results/second",
        fig_dir="plots/second",
        output_path="results/second",
    )
    pd.DataFrame([_legacy_row(), duplicate]).to_csv(path, index=False)

    rows, _ = _load_spectra_fit_rows([path])

    assert rows["object_id"].tolist() == ["obj"]
    assert "Collapsing 1 provenance-only duplicate" in capsys.readouterr().out


def test_legacy_scientifically_divergent_duplicates_are_rejected(tmp_path):
    path = tmp_path / "legacy-divergent.csv"
    pd.DataFrame([_legacy_row(), _legacy_row(PL_slope=-2.0)]).to_csv(
        path,
        index=False,
    )

    with pytest.raises(ValueError, match="scientifically divergent duplicate"):
        _load_spectra_fit_rows([path])


def test_current_and_legacy_csvs_cannot_be_mixed(tmp_path):
    current_path = tmp_path / "current.csv"
    legacy_path = tmp_path / "legacy.csv"
    pd.DataFrame([_current_row()]).to_csv(current_path, index=False)
    pd.DataFrame([_legacy_row()]).to_csv(legacy_path, index=False)

    with pytest.raises(ValueError, match="Cannot mix current.*legacy"):
        _load_spectra_fit_rows([current_path, legacy_path])


@pytest.mark.parametrize(
    "updates",
    [
        {"frac_host_5100": 1.1, "f_host_5100": 1.1},
        {"frac_host_5100": 0.2, "f_host_5100": 0.3},
    ],
)
def test_legacy_host_fraction_validation_is_strict(tmp_path, updates):
    path = tmp_path / "legacy-invalid-host.csv"
    pd.DataFrame([_legacy_row(**updates)]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="host fraction|inconsistent"):
        _load_spectra_fit_rows([path])


def test_incomplete_spectra_schema_reports_current_and_legacy_requirements(tmp_path):
    path = tmp_path / "incomplete.csv"
    pd.DataFrame([{"object_id": "obj", "fit_ok": True}]).to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Missing current joint-JAXSEDFit columns.*legacy JAXQSOFit columns",
    ):
        _load_spectra_fit_rows([path])
