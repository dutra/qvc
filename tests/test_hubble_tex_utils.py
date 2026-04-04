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

from qvc.hubble.tex_utils import make_agn_csv_table, make_agn_latex_table


def _make_table_df():
    return pd.DataFrame(
        {
            "sdss_name": ["123456.78+123456.7"],
            "ra": [123.4567],
            "dec": [12.3456],
            "z": [1.2345],
            "z_err": [0.0012],
            "apparent_mag_2500": [20.15],
            "apparent_mag_2500_err": [0.07],
            "PL_slope": [-1.37],
            "PL_slope_err": [0.08],
            "log_tau_uv_rf": [2.34],
            "log_tau_uv_rf_std_psd": [0.11],
            "log_sigma_uv": [-0.88],
            "log_sigma_uv_std_psd": [0.09],
            "log_sigma_uv_log_tau_uv_rf_cov_psd": [0.012],
            "f_host_2500": [0.21],
            "f_host_2500_err": [0.03],
            "f_bc_3000": [0.14],
            "f_bc_3000_err": [0.02],
            "f_fe_uv_3000": [0.31],
            "f_fe_uv_3000_err": [0.04],
            "f_na": [0.05],
            "f_na_err": [0.01],
            "f_br": [0.07],
            "f_br_err": [0.02],
        }
    )


def test_make_agn_latex_table_writes_expected_output(tmp_path):
    df = _make_table_df()

    def dm_interp(points):
        arr = np.asarray(points, dtype=float)
        return np.zeros(arr.shape[0], dtype=float)

    latex = make_agn_latex_table(
        df,
        mu=np.array([44.21]),
        mu_err=np.array([0.13]),
        dm_interp=dm_interp,
        sort_by="z",
        ascending=True,
        max_rows=30,
        write_path=str(tmp_path),
    )

    out_path = tmp_path / "agn_table.tex"
    assert out_path.exists()
    assert r"\textbf{SDSS Name}" in latex
    assert r"$m_{2500}$" in latex
    assert r"\texttt{PL\_slope}" in latex
    assert r"$\log\tau_{\mathrm{UV,RF}}$" in latex
    assert r"$\log\sigma_{\mathrm{UV}}$" in latex
    assert r"$f_{\rm{host,\,2500\,\text{\AA}}}$" in latex
    assert r"$f_{\rm{BC}}$" in latex
    assert r"$f_{\rm{lines}}$" in latex
    assert r"$f_{\rm{Fe\,II}}$" in latex
    assert r"\textbf{J123456.78+123456.7}" in latex
    assert "$123.4567$" in latex
    assert "$+12.3456$" in latex
    assert "$20.15 \\pm 0.07$" in latex
    assert "$-1.37 \\pm 0.08$" in latex
    assert "$44.21 \\pm 0.13$" in latex
    assert "$2.34 \\pm 0.11$" in latex
    assert "$-0.88 \\pm 0.09$" in latex
    assert "$0.21 \\pm 0.03$" in latex
    assert "$0.14 \\pm 0.02$" in latex
    assert "$0.12 \\pm 0.02$" in latex
    assert "$0.31 \\pm 0.04$" in latex


def test_make_agn_csv_table_writes_expected_output(tmp_path):
    df = pd.concat([_make_table_df(), _make_table_df()], ignore_index=True)
    df.loc[1, "sdss_name"] = "223456.78+123456.7"
    df.loc[1, "z"] = 2.3456
    df.loc[1, "apparent_mag_2500"] = 21.15
    df.loc[1, "f_na"] = 0.09
    df.loc[1, "f_br"] = 0.03

    csv_df = make_agn_csv_table(
        df,
        mu=np.array([44.21, 45.21]),
        mu_err=np.array([0.13, 0.23]),
        dm_interp=lambda points: np.full(np.asarray(points).shape[0], 0.5, dtype=float),
        sort_by="z",
        ascending=True,
        write_path=str(tmp_path),
    )

    out_path = tmp_path / "agn_table.csv"
    assert out_path.exists()
    loaded = pd.read_csv(out_path)
    assert len(loaded) == 2
    assert list(loaded["z"]) == sorted(df["z"].tolist())
    for col in ("mu", "mu_err", "apparent_mag_2500_corr", "apparent_mag_2500_corr_err", "f_lines", "f_lines_err"):
        assert col in loaded.columns
    np.testing.assert_allclose(loaded["apparent_mag_2500_corr"], loaded["apparent_mag_2500"] - 0.5)
    np.testing.assert_allclose(loaded["f_lines"], loaded["f_na"] + loaded["f_br"])
    assert list(csv_df["z"]) == sorted(df["z"].tolist())


def test_make_agn_latex_table_supports_2d_dm_interp_with_richer_inputs(tmp_path):
    df = _make_table_df()

    def dm_interp(points):
        arr = np.asarray(points, dtype=float)
        assert arr.shape == (1, 3)
        return np.full(arr.shape[0], 0.5, dtype=float)

    latex = make_agn_latex_table(
        df,
        mu=np.array([44.21]),
        mu_err=np.array([0.13]),
        dm_interp=dm_interp,
        sort_by="z",
        ascending=True,
        max_rows=30,
        write_path=str(tmp_path),
    )

    assert "$19.65 \\pm 0.07$" in latex


def test_make_agn_csv_table_supports_2d_dm_interp_with_richer_inputs(tmp_path):
    df = _make_table_df()

    def dm_interp(points):
        arr = np.asarray(points, dtype=float)
        assert arr.shape == (1, 3)
        return np.full(arr.shape[0], 0.5, dtype=float)

    csv_df = make_agn_csv_table(
        df,
        mu=np.array([44.21]),
        mu_err=np.array([0.13]),
        dm_interp=dm_interp,
        sort_by="z",
        ascending=True,
        write_path=str(tmp_path),
    )

    np.testing.assert_allclose(csv_df["apparent_mag_2500_corr"], [19.65])


def test_make_agn_latex_table_passes_f_host_to_3d_dm_interp(tmp_path):
    df = _make_table_df()
    seen = {}

    def dm_interp(points):
        seen["points"] = np.asarray(points, dtype=float)
        return np.full(seen["points"].shape[0], 0.5, dtype=float)

    make_agn_latex_table(
        df,
        mu=np.array([44.21]),
        mu_err=np.array([0.13]),
        dm_interp=dm_interp,
        sort_by="z",
        ascending=True,
        max_rows=30,
        write_path=str(tmp_path),
    )

    np.testing.assert_allclose(
        seen["points"],
        np.array([[1.2345, 20.15, 0.21]], dtype=float),
    )

    make_agn_csv_table(
        df,
        mu=np.array([44.21]),
        mu_err=np.array([0.13]),
        dm_interp=dm_interp,
        sort_by="z",
        ascending=True,
        write_path=str(tmp_path),
    )


def test_make_agn_latex_table_passes_alpha_lambda_to_4d_dm_interp(tmp_path):
    df = _make_table_df()
    df["alpha_lambda"] = [-1.37]
    seen = {}

    def dm_interp(points):
        seen["points"] = np.asarray(points, dtype=float)
        return np.full(seen["points"].shape[0], 0.5, dtype=float)

    make_agn_latex_table(
        df,
        mu=np.array([44.21]),
        mu_err=np.array([0.13]),
        dm_interp=dm_interp,
        sort_by="z",
        ascending=True,
        max_rows=30,
        write_path=str(tmp_path),
    )

    np.testing.assert_allclose(
        seen["points"],
        np.array([[1.2345, 20.15, 0.21, -1.37]], dtype=float),
    )

    make_agn_csv_table(
        df,
        mu=np.array([44.21]),
        mu_err=np.array([0.13]),
        dm_interp=dm_interp,
        sort_by="z",
        ascending=True,
        write_path=str(tmp_path),
    )


@pytest.mark.parametrize(
    "missing_col",
    [
        "sdss_name",
        "ra",
        "dec",
        "z",
        "z_err",
        "apparent_mag_2500",
        "apparent_mag_2500_err",
        "PL_slope",
        "PL_slope_err",
        "log_tau_uv_rf",
        "log_tau_uv_rf_std_psd",
        "log_sigma_uv",
        "log_sigma_uv_std_psd",
        "log_sigma_uv_log_tau_uv_rf_cov_psd",
        "f_host_2500",
        "f_host_2500_err",
        "f_bc_3000",
        "f_bc_3000_err",
        "f_fe_uv_3000",
        "f_fe_uv_3000_err",
        "f_na",
        "f_na_err",
        "f_br",
        "f_br_err",
    ],
)
def test_make_agn_latex_table_raises_for_missing_required_columns(tmp_path, missing_col):
    df = _make_table_df().drop(columns=[missing_col])

    with pytest.raises(KeyError, match=missing_col):
        make_agn_latex_table(
            df,
            mu=np.array([44.21]),
            mu_err=np.array([0.13]),
            dm_interp=lambda points: np.zeros(np.asarray(points).shape[0], dtype=float),
            sort_by="z",
            ascending=True,
            max_rows=30,
            write_path=str(tmp_path),
        )

    with pytest.raises(KeyError, match=missing_col):
        make_agn_csv_table(
            df,
            mu=np.array([44.21]),
            mu_err=np.array([0.13]),
            dm_interp=lambda points: np.zeros(np.asarray(points).shape[0], dtype=float),
            sort_by="z",
            ascending=True,
            write_path=str(tmp_path),
        )
