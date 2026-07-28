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

from qvc.hubble import hubble_model, hubble_plotting, hubble_utils


def _make_loader_input():
    return pd.DataFrame(
        {
            "object_id": ["agn_valid", "agn_zero", "agn_hi", "agn_neg", "agn_nanerr"],
            "z": [1.2, 1.3, 1.4, 1.5, 1.6],
            "mags_mean_u": [20.1, 20.1, 20.1, 20.1, 20.1],
            "mags_mean_g": [19.8, 19.8, 19.8, 19.8, 19.8],
            "mags_mean_r": [19.5, 19.5, 19.5, 19.5, 19.5],
            "mags_mean_i": [19.2, 19.2, 19.2, 19.2, 19.2],
            "dropped_bands": ["", "", "", "", ""],
            "log_jitter_u": [-9.0, -9.0, -9.0, -9.0, -9.0],
            "log_jitter_g": [-2.0, -2.0, -2.0, -2.0, -2.0],
            "log_jitter_r": [-2.0, -2.0, -2.0, -2.0, -2.0],
            "log_jitter_i": [-2.0, -2.0, -2.0, -2.0, -2.0],
            "dlog_amp_blr_u": [-9.0, -9.0, -9.0, -9.0, -9.0],
            "dlog_amp_blr_g": [-2.5, -2.5, -2.5, -2.5, -2.5],
            "dlog_amp_blr_r": [-2.5, -2.5, -2.5, -2.5, -2.5],
            "dlog_amp_blr_i": [-2.5, -2.5, -2.5, -2.5, -2.5],
            "alpha_lambda": [-1.5, -1.5, -1.5, -1.5, -1.5],
            "log_sigma_hat0": [-2.1, -2.0, -2.2, -2.3, -2.1],
            "log_sigma_hat0_err": [0.05, 0.05, 0.05, 0.05, 0.05],
            "log_sigma_uv": [-1.0, -0.9, -1.1, -1.2, -0.95],
            "log_sigma_uv_std_psd": [0.10, 0.11, 0.12, 0.13, 0.14],
            "log_tau_uv_rf": [2.4, 2.5, 2.6, 2.7, 2.8],
            "log_tau_uv_rf_std_psd": [0.2, 0.2, 0.2, 0.2, 0.2],
            "log_sigma_uv_log_tau_uv_rf_cov_psd": [0.01, 0.01, 0.01, 0.01, 0.01],
            "apparent_mag_2500": [20.3, 20.4, 20.5, 20.6, 20.7],
            "apparent_mag_2500_err": [0.1, 0.1, 0.1, 0.1, 0.1],
            "apparent_mag_2500_intrinsic": [20.3, 20.4, 20.5, 20.6, 20.7],
            "apparent_mag_2500_intrinsic_err": [0.1, 0.1, 0.1, 0.1, 0.1],
            "apparent_mag_2500_reddened": [20.3, 20.4, 20.5, 20.6, 20.7],
            "apparent_mag_2500_reddened_err": [0.1, 0.1, 0.1, 0.1, 0.1],
            "f_host_2500": [0.2, 0.0, 1.1, -0.1, 0.3],
            "f_host_2500_err": [0.05, 0.0, 0.1, 0.2, np.nan],
            "f_PL": [0.8, 1.0, 1.1, -0.1, 0.7],
            "f_PL_err": [0.05, 0.0, 0.1, 0.2, np.nan],
        }
    )


def test_load_agn_data_requires_an_explicit_exact_magnitude_convention():
    with pytest.raises(TypeError, match="magnitude_convention"):
        hubble_utils.load_agn_data("unused.h5")

    with pytest.raises(ValueError, match="exactly 'intrinsic' or 'observed'"):
        hubble_utils.load_agn_data(
            "unused.h5",
            magnitude_convention=" Intrinsic ",
        )


def test_load_agn_data_raises_when_selected_magnitude_columns_are_missing(
    monkeypatch,
    tmp_path,
):
    df_in = _make_loader_input().iloc[:2].copy()
    df_in = df_in.drop(
        columns=[
            "apparent_mag_2500_intrinsic",
            "apparent_mag_2500_intrinsic_err",
        ]
    )
    input_path = tmp_path / "fake_input.h5"
    input_path.touch()
    monkeypatch.setattr(
        hubble_utils,
        "read_quasars_from_hdf5_flat",
        lambda *args, **kwargs: df_in.copy(),
    )
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda df: df)

    with pytest.raises(
        ValueError,
        match="apparent_mag_2500_intrinsic.*apparent_mag_2500_intrinsic_err",
    ):
        hubble_utils.load_agn_data(
            input_path,
            spectra_fit_csv=None,
            magnitude_convention="intrinsic",
            lc_info_csv=None,
            only_load=True,
            apply_cut=False,
            plot_diagnostics=False,
        )


def _patch_minimal_loader(monkeypatch, df_in):
    monkeypatch.setattr(
        hubble_utils,
        "read_quasars_from_hdf5_flat",
        lambda *args, **kwargs: df_in.copy(),
    )
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda df: df)


def test_load_agn_data_observed_aliases_legacy_generic_columns(
    monkeypatch,
    tmp_path,
):
    df_in = _make_loader_input().iloc[:2].copy()
    df_in = df_in.drop(
        columns=[
            "apparent_mag_2500_reddened",
            "apparent_mag_2500_reddened_err",
        ]
    )
    expected_magnitude = df_in["apparent_mag_2500"].copy()
    expected_error = df_in["apparent_mag_2500_err"].copy()
    input_path = tmp_path / "fake_input.h5"
    input_path.touch()
    _patch_minimal_loader(monkeypatch, df_in)

    df, df_all = hubble_utils.load_agn_data(
        input_path,
        spectra_fit_csv=None,
        magnitude_convention="observed",
        lc_info_csv=None,
        only_load=True,
        apply_cut=False,
        plot_diagnostics=False,
    )

    for frame in (df, df_all):
        np.testing.assert_array_equal(
            frame["apparent_mag_2500_reddened"], expected_magnitude
        )
        np.testing.assert_array_equal(
            frame["apparent_mag_2500_reddened_err"], expected_error
        )
        np.testing.assert_array_equal(
            frame["apparent_mag_2500"], expected_magnitude
        )
        np.testing.assert_array_equal(
            frame["apparent_mag_2500_err"], expected_error
        )


@pytest.mark.parametrize(
    ("canonical_column", "legacy_column"),
    [
        ("apparent_mag_2500_reddened", "apparent_mag_2500"),
        ("apparent_mag_2500_reddened_err", "apparent_mag_2500_err"),
    ],
)
def test_load_agn_data_observed_raises_when_canonical_and_legacy_disagree(
    monkeypatch,
    tmp_path,
    canonical_column,
    legacy_column,
):
    df_in = _make_loader_input().iloc[:2].copy()
    df_in.loc[df_in.index[1], canonical_column] = (
        df_in.loc[df_in.index[1], legacy_column] + 0.01
    )
    input_path = tmp_path / "fake_input.h5"
    input_path.touch()
    _patch_minimal_loader(monkeypatch, df_in)

    with pytest.raises(
        ValueError,
        match=rf"Conflicting observed-magnitude columns.*{canonical_column!s}",
    ):
        hubble_utils.load_agn_data(
            input_path,
            spectra_fit_csv=None,
            magnitude_convention="observed",
            lc_info_csv=None,
            only_load=True,
            apply_cut=False,
            plot_diagnostics=False,
        )


@pytest.mark.parametrize(
    "columns_to_drop",
    [
        [
            "apparent_mag_2500_reddened",
            "apparent_mag_2500_reddened_err",
            "apparent_mag_2500_err",
        ],
        ["apparent_mag_2500_reddened_err"],
        ["apparent_mag_2500_err"],
    ],
)
def test_load_agn_data_observed_raises_for_incomplete_alias_pairs(
    monkeypatch,
    tmp_path,
    columns_to_drop,
):
    df_in = _make_loader_input().iloc[:2].drop(columns=columns_to_drop)
    input_path = tmp_path / "fake_input.h5"
    input_path.touch()
    _patch_minimal_loader(monkeypatch, df_in)

    with pytest.raises(ValueError, match="magnitude_convention='observed'"):
        hubble_utils.load_agn_data(
            input_path,
            spectra_fit_csv=None,
            magnitude_convention="observed",
            lc_info_csv=None,
            only_load=True,
            apply_cut=False,
            plot_diagnostics=False,
        )


def test_load_agn_data_observed_colocated_nans_are_not_alias_conflicts(
    monkeypatch,
    tmp_path,
):
    df_in = _make_loader_input().iloc[:2].copy()
    df_in.loc[df_in.index[1], "apparent_mag_2500"] = np.nan
    df_in.loc[df_in.index[1], "apparent_mag_2500_reddened"] = np.nan
    input_path = tmp_path / "fake_input.h5"
    input_path.touch()
    _patch_minimal_loader(monkeypatch, df_in)

    with pytest.raises(ValueError, match="non-finite"):
        hubble_utils.load_agn_data(
            input_path,
            spectra_fit_csv=None,
            magnitude_convention="observed",
            lc_info_csv=None,
            only_load=True,
            apply_cut=False,
            plot_diagnostics=False,
        )


@pytest.mark.parametrize(
    ("bad_magnitude", "bad_error", "message"),
    [
        (np.nan, 0.1, "non-finite"),
        ("not-a-number", 0.1, "numeric"),
        (20.0, np.nan, "non-finite"),
        (20.0, -0.1, "non-negative"),
    ],
)
def test_load_agn_data_raises_for_invalid_selected_magnitude_values(
    monkeypatch,
    tmp_path,
    bad_magnitude,
    bad_error,
    message,
):
    df_in = _make_loader_input().iloc[:2].copy()
    df_in["apparent_mag_2500_intrinsic"] = [20.0, bad_magnitude]
    df_in["apparent_mag_2500_intrinsic_err"] = [0.1, bad_error]
    input_path = tmp_path / "fake_input.h5"
    input_path.touch()
    monkeypatch.setattr(
        hubble_utils,
        "read_quasars_from_hdf5_flat",
        lambda *args, **kwargs: df_in.copy(),
    )
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda df: df)

    with pytest.raises(ValueError, match=message):
        hubble_utils.load_agn_data(
            input_path,
            spectra_fit_csv=None,
            magnitude_convention="intrinsic",
            lc_info_csv=None,
            only_load=True,
            apply_cut=False,
            plot_diagnostics=False,
        )


def test_load_agn_data_propagates_host_error_into_sigma_uv(monkeypatch, tmp_path):
    df_in = _make_loader_input()
    input_path = tmp_path / "fake_input.h5"
    input_path.touch()

    monkeypatch.setattr(hubble_utils, "read_quasars_from_hdf5_flat", lambda *args, **kwargs: df_in.copy())
    monkeypatch.setattr(hubble_plotting, "plot_sigma_uv_host_correction", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_plotting, "plot_alpha_lambda_vs_redshift", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_plotting, "plot_alpha_lambda_histogram", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_plotting, "plot_tau_sigma_vs_redshift", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_plotting, "plot_sigma_uv_vs_tau_uv_rf", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_plotting, "plot_cut_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda df: df)

    df, df_all = hubble_utils.load_agn_data(
        input_path,
        magnitude_convention="intrinsic",
        spectra_fit_csv=None,
        lc_info_csv=None,
        only_load=True,
        apply_cut=False,
        correct_sigma_uv_host=True,
        plot_path=str(tmp_path / "figures"),
    )
    assert df_all.equals(df)
    np.testing.assert_allclose(
        df["apparent_mag_2500"],
        df_in["apparent_mag_2500"],
    )

    ln10 = np.log(10.0)
    orig_sigma = df_in["log_sigma_uv"].to_numpy(dtype=float)
    orig_err = df_in["log_sigma_uv_std_psd"].to_numpy(dtype=float)
    f_pl = df_in["f_PL"].to_numpy(dtype=float)
    f_pl_err = np.nan_to_num(df_in["f_PL_err"].to_numpy(dtype=float), nan=0.0)
    valid = np.isfinite(f_pl) & (f_pl > 0.0) & (f_pl <= 1.0)
    expected_delta = np.zeros(len(df_in), dtype=float)
    expected_delta[valid] = -np.log10(f_pl[valid])
    expected_host_err = np.zeros(len(df_in), dtype=float)
    expected_host_err[valid] = f_pl_err[valid] / (f_pl[valid] * ln10)
    expected_err = orig_err.copy()
    expected_err[valid] = np.sqrt(orig_err[valid] ** 2 + expected_host_err[valid] ** 2)

    np.testing.assert_allclose(df["log_sigma_uv_uncorrected"], orig_sigma)
    np.testing.assert_allclose(df["log_sigma_uv"], orig_sigma + expected_delta)
    np.testing.assert_allclose(df["log_sigma_uv_std_psd_uncorrected"], orig_err)
    np.testing.assert_allclose(df["log_sigma_uv_hostcorr_err"], expected_host_err)
    np.testing.assert_allclose(df["log_sigma_uv_std_psd_corrected"], expected_err)
    np.testing.assert_allclose(df["log_sigma_uv_std_psd"], expected_err)

    valid_factor = np.full(len(df_in), np.nan, dtype=float)
    valid_factor[valid] = 1.0 / f_pl[valid]
    np.testing.assert_allclose(df["sigma_uv_hostcorr_factor"], valid_factor, equal_nan=True)
    np.testing.assert_allclose(df["jitter_g"], 10**df_in["log_jitter_g"].to_numpy(dtype=float))
    expected_legacy_jitter_total = np.sqrt(
        (10**df_in["log_jitter_u"].to_numpy(dtype=float)) ** 2
        + (10**df_in["log_jitter_g"].to_numpy(dtype=float)) ** 2
        + (10**df_in["log_jitter_r"].to_numpy(dtype=float)) ** 2
        + (10**df_in["log_jitter_i"].to_numpy(dtype=float)) ** 2
    )
    np.testing.assert_allclose(df["log_jitter_total"], np.log10(expected_legacy_jitter_total))

    obs_dict = {key: df[key].to_numpy(dtype=float) for key in hubble_model.agn_model_req_obs + hubble_model.agn_model_req_errs}
    pivot_context = hubble_model.build_agn_pivot_context(
        df,
        z_range=(float(df["z"].min()), float(df["z"].max())),
    )
    obs_arr, err_arr, pivots = hubble_model.agn_model_pack_obs(
        obs_dict,
        pivot_context=pivot_context,
    )
    assert obs_arr.shape[1] == len(df)
    assert err_arr.shape[1] == len(df)
    assert pivots.shape[0] == len(hubble_model.agn_model_req_obs)


def test_load_agn_data_aliases_intrinsic_spectral_magnitude(monkeypatch, tmp_path, capsys):
    df_in = _make_loader_input().iloc[:2].copy()
    observed_mag = np.array([20.4, 21.1])
    observed_err = np.array([0.20, 0.30])
    intrinsic_mag = np.array([20.0, 20.5])
    intrinsic_err = np.array([0.10, 0.15])
    df_in["apparent_mag_2500"] = observed_mag
    df_in["apparent_mag_2500_err"] = observed_err

    def fake_populate_spectra_fit(frame, _spectra_fit_csv):
        frame = frame.copy()
        frame["apparent_mag_2500_reddened"] = observed_mag
        frame["apparent_mag_2500_reddened_err"] = observed_err
        frame["apparent_mag_2500_intrinsic"] = intrinsic_mag
        frame["apparent_mag_2500_intrinsic_err"] = intrinsic_err
        return frame

    input_path = tmp_path / "fake_input.h5"
    input_path.touch()
    spectra_path = tmp_path / "spectra.csv"
    spectra_path.touch()
    monkeypatch.setattr(
        hubble_utils,
        "read_quasars_from_hdf5_flat",
        lambda *args, **kwargs: df_in.copy(),
    )
    monkeypatch.setattr(hubble_utils, "populate_spectra_fit", fake_populate_spectra_fit)
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda df: df)

    df, df_all = hubble_utils.load_agn_data(
        input_path,
        spectra_fit_csv=[str(spectra_path)],
        magnitude_convention="intrinsic",
        lc_info_csv=None,
        only_load=True,
        apply_cut=False,
        plot_diagnostics=False,
    )

    assert df_all.equals(df)
    np.testing.assert_allclose(df["apparent_mag_2500"], intrinsic_mag)
    np.testing.assert_allclose(df["apparent_mag_2500_err"], intrinsic_err)
    np.testing.assert_allclose(df["apparent_mag_2500_reddened"], observed_mag)
    np.testing.assert_allclose(df["apparent_mag_2500_reddened_err"], observed_err)
    np.testing.assert_allclose(df["dm_red"], observed_mag - intrinsic_mag)
    np.testing.assert_allclose(
        df["dm_red_err"],
        np.sqrt(observed_err**2 + intrinsic_err**2),
    )
    assert "Using apparent_mag_2500_intrinsic" in capsys.readouterr().out


def test_load_agn_data_can_use_observed_spectral_magnitude(monkeypatch, tmp_path, capsys):
    df_in = _make_loader_input().iloc[:2].copy()
    observed_mag = np.array([20.4, 21.1])
    observed_err = np.array([0.20, 0.30])
    intrinsic_mag = np.array([20.0, 20.5])
    intrinsic_err = np.array([0.10, 0.15])
    df_in["apparent_mag_2500"] = observed_mag
    df_in["apparent_mag_2500_err"] = observed_err

    def fake_populate_spectra_fit(frame, _spectra_fit_csv):
        frame = frame.copy()
        frame["apparent_mag_2500_reddened"] = observed_mag
        frame["apparent_mag_2500_reddened_err"] = observed_err
        frame["apparent_mag_2500_intrinsic"] = intrinsic_mag
        frame["apparent_mag_2500_intrinsic_err"] = intrinsic_err
        return frame

    input_path = tmp_path / "fake_input.h5"
    input_path.touch()
    spectra_path = tmp_path / "spectra.csv"
    spectra_path.touch()
    monkeypatch.setattr(
        hubble_utils,
        "read_quasars_from_hdf5_flat",
        lambda *args, **kwargs: df_in.copy(),
    )
    monkeypatch.setattr(hubble_utils, "populate_spectra_fit", fake_populate_spectra_fit)
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda df: df)

    df, df_all = hubble_utils.load_agn_data(
        input_path,
        spectra_fit_csv=[str(spectra_path)],
        magnitude_convention="observed",
        lc_info_csv=None,
        only_load=True,
        apply_cut=False,
        plot_diagnostics=False,
    )

    assert df_all.equals(df)
    np.testing.assert_allclose(df["apparent_mag_2500"], observed_mag)
    np.testing.assert_allclose(df["apparent_mag_2500_err"], observed_err)
    np.testing.assert_allclose(df["apparent_mag_2500_intrinsic"], intrinsic_mag)
    np.testing.assert_allclose(df["apparent_mag_2500_intrinsic_err"], intrinsic_err)
    np.testing.assert_allclose(df["apparent_mag_2500_reddened"], observed_mag)
    np.testing.assert_allclose(df["apparent_mag_2500_reddened_err"], observed_err)
    np.testing.assert_allclose(df["dm_red"], observed_mag - intrinsic_mag)
    np.testing.assert_allclose(
        df["dm_red_err"],
        np.sqrt(observed_err**2 + intrinsic_err**2),
    )
    assert "Using apparent_mag_2500_reddened" in capsys.readouterr().out


def test_load_agn_data_applies_observed_convention_to_hdf5_spectral_fields(
    monkeypatch,
    tmp_path,
):
    df_in = _make_loader_input().iloc[:2].copy()
    observed_mag = np.array([20.4, 21.1])
    observed_err = np.array([0.20, 0.30])
    intrinsic_mag = np.array([20.0, 20.5])
    intrinsic_err = np.array([0.10, 0.15])
    df_in["apparent_mag_2500"] = observed_mag
    df_in["apparent_mag_2500_err"] = observed_err
    df_in["apparent_mag_2500_reddened"] = observed_mag
    df_in["apparent_mag_2500_reddened_err"] = observed_err
    df_in["apparent_mag_2500_intrinsic"] = intrinsic_mag
    df_in["apparent_mag_2500_intrinsic_err"] = intrinsic_err

    input_path = tmp_path / "fake_input.h5"
    input_path.touch()
    monkeypatch.setattr(
        hubble_utils,
        "read_quasars_from_hdf5_flat",
        lambda *args, **kwargs: df_in.copy(),
    )
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda df: df)

    df, df_all = hubble_utils.load_agn_data(
        input_path,
        spectra_fit_csv=None,
        magnitude_convention="observed",
        lc_info_csv=None,
        only_load=True,
        apply_cut=False,
        plot_diagnostics=False,
    )

    assert df_all.equals(df)
    np.testing.assert_allclose(df["apparent_mag_2500"], observed_mag)
    np.testing.assert_allclose(df["apparent_mag_2500_err"], observed_err)
    np.testing.assert_allclose(df["dm_red"], observed_mag - intrinsic_mag)
    np.testing.assert_allclose(
        df["dm_red_err"],
        np.sqrt(observed_err**2 + intrinsic_err**2),
    )


def test_load_agn_data_uses_survey_band_log_jitter_grid(monkeypatch, tmp_path):
    df_in = _make_loader_input().iloc[:2].copy()
    df_in["dropped_bands"] = ["", "r"]
    for band in ("u", "g", "r", "i"):
        df_in = df_in.drop(columns=[f"log_jitter_{band}"])

    jitter_values = {
        "u": {"sdss": [0.010, 0.020], "ps1": [0.011, 0.021], "ztf": [0.012, 0.022]},
        "g": {"sdss": [0.020, 0.030], "ps1": [0.021, 0.031], "ztf": [0.022, 0.032]},
        "r": {"sdss": [0.030, 0.040], "ps1": [0.031, 0.041], "ztf": [0.032, 0.042]},
        "i": {"sdss": [0.040, 0.050], "ps1": [0.041, 0.051], "ztf": [0.042, 0.052]},
    }
    for band, by_survey in jitter_values.items():
        for survey, values in by_survey.items():
            df_in[f"log_jitter_{band}_{survey}"] = np.log(np.asarray(values, dtype=float))

    input_path = tmp_path / "fake_input.h5"
    input_path.touch()
    monkeypatch.setattr(hubble_utils, "read_quasars_from_hdf5_flat", lambda *args, **kwargs: df_in.copy())
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda df: df)

    df, df_all = hubble_utils.load_agn_data(
        input_path,
        magnitude_convention="intrinsic",
        spectra_fit_csv=None,
        lc_info_csv=None,
        only_load=True,
        apply_cut=False,
        plot_path=str(tmp_path / "figures"),
    )
    assert df_all.equals(df)

    row0 = df.iloc[0]
    row1 = df.iloc[1]
    assert np.isclose(row0["jitter_g_sdss"], jitter_values["g"]["sdss"][0])
    assert np.isclose(row0["jitter_g_ps1"], jitter_values["g"]["ps1"][0])
    expected_g0 = np.sqrt(sum(jitter_values["g"][survey][0] ** 2 for survey in ("sdss", "ps1", "ztf")))
    assert np.isclose(row0["jitter_g"], expected_g0)

    assert row1["jitter_r_sdss"] == 0.0
    assert row1["jitter_r_ps1"] == 0.0
    assert row1["jitter_r_ztf"] == 0.0
    assert row1["jitter_r"] == 0.0

    expected_total0 = np.sqrt(
        sum(
            jitter_values[band][survey][0] ** 2
            for band in ("u", "g", "r", "i")
            for survey in ("sdss", "ps1", "ztf")
        )
    )
    expected_total1 = np.sqrt(
        sum(
            jitter_values[band][survey][1] ** 2
            for band in ("u", "g", "i")
            for survey in ("sdss", "ps1", "ztf")
        )
    )
    assert np.isclose(row0["log_jitter_total"], np.log10(expected_total0))
    assert np.isclose(row1["log_jitter_total"], np.log10(expected_total1))
