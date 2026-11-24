"""Extra m_2500 uncertainty added in quadrature in place of the spectral R-hat cut."""

import json

import numpy as np
import pandas as pd
import pytest

from qvc.hubble import hubble_utils
from qvc.hubble.cuts import (
    MAGNITUDE_ERROR_EXTRA_ENV,
    MAGNITUDE_ERROR_EXTRA_Z_BREAK_ENV,
    add_magnitude_error_extra,
    magnitude_error_extra_settings,
)
from qvc.hubble.hubble_completeness_refactored import (
    COMPLETENESS_MAG_ERR_COL,
    prepare_completeness_magnitude_columns,
)

from test_hubble_load_plots import _minimal_agn_frame, _patch_load_agn_plotters

DERED_LOW, DERED_HIGH = MAGNITUDE_ERROR_EXTRA_ENV["m_2500_dereddened_err"]
ATT_LOW, ATT_HIGH = MAGNITUDE_ERROR_EXTRA_ENV["m_2500_attenuated_model_err"]
ALL_ENV = (DERED_LOW, DERED_HIGH, ATT_LOW, ATT_HIGH, MAGNITUDE_ERROR_EXTRA_Z_BREAK_ENV)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ALL_ENV:
        monkeypatch.delenv(name, raising=False)


def _proposed(monkeypatch):
    for name, value in ((DERED_LOW, "0.03"), (DERED_HIGH, "0.10"), (ATT_LOW, "0.02"), (ATT_HIGH, "0.04")):
        monkeypatch.setenv(name, value)


def test_settings_default_off_and_validation(monkeypatch):
    assert magnitude_error_extra_settings() is None
    monkeypatch.setenv(DERED_LOW, "0")
    monkeypatch.setenv(ATT_HIGH, "none")
    assert magnitude_error_extra_settings() is None
    monkeypatch.setenv(DERED_HIGH, "0.1")
    assert magnitude_error_extra_settings() == {"z_break": 2.0, "m_2500_dereddened_err": [0.0, 0.1]}
    monkeypatch.setenv(MAGNITUDE_ERROR_EXTRA_Z_BREAK_ENV, "1.8")
    assert magnitude_error_extra_settings()["z_break"] == 1.8
    monkeypatch.setenv(DERED_LOW, "-0.01")
    with pytest.raises(ValueError, match=DERED_LOW):
        magnitude_error_extra_settings()


def test_terms_are_added_in_quadrature_with_the_redshift_break(monkeypatch):
    _proposed(monkeypatch)
    frame = pd.DataFrame(
        {
            "z": [1.0, 1.999, 2.0, 3.0],
            "m_2500_dereddened_err": [0.04, 0.04, 0.0, 0.10],
            "m_2500_attenuated_model_err": [0.0, 0.03, 0.03, 0.0],
        }
    )
    out = add_magnitude_error_extra(frame, magnitude_error_extra_settings())
    np.testing.assert_allclose(out["m_2500_dereddened_err"], [0.05, 0.05, 0.10, np.hypot(0.1, 0.1)])
    np.testing.assert_allclose(out["m_2500_attenuated_model_err"], [0.02, np.hypot(0.03, 0.02), 0.05, 0.04])
    np.testing.assert_array_equal(out["m_2500_dereddened_err_extra"], [0.03, 0.03, 0.10, 0.10])
    # The input frame is not modified.
    assert frame["m_2500_dereddened_err"].tolist() == [0.04, 0.04, 0.0, 0.10]


def test_off_is_a_no_op_and_requested_columns_must_exist(monkeypatch):
    frame = pd.DataFrame({"z": [1.0], "m_2500_dereddened_err": [0.1]})
    assert add_magnitude_error_extra(frame, None) is frame
    monkeypatch.setenv(ATT_HIGH, "0.04")
    with pytest.raises(ValueError, match="m_2500_attenuated_model_err"):
        add_magnitude_error_extra(frame, magnitude_error_extra_settings())
    # A non-numeric error column is not coerced (the later magnitude check's rule).
    with pytest.raises(ValueError, match="numeric values without coercion"):
        add_magnitude_error_extra(
            pd.DataFrame({"z": [1.0], "m_2500_attenuated_model_err": ["0.1"]}),
            magnitude_error_extra_settings(),
        )
    # Rows without a finite redshift get no term (the Tier-0 z cut removes them).
    out = add_magnitude_error_extra(
        pd.DataFrame({"z": [np.nan, 2.5], "m_2500_attenuated_model_err": [0.1, 0.1]}),
        magnitude_error_extra_settings(),
    )
    np.testing.assert_allclose(out["m_2500_attenuated_model_err"], [0.1, np.hypot(0.1, 0.04)])
    assert np.isnan(out["m_2500_attenuated_model_err_extra"][0])
    np.testing.assert_array_equal(out["m_2500_attenuated_model_err_raw"], [0.1, 0.1])


# --- load_agn_data ------------------------------------------------------------


def _load(tmp_path, monkeypatch, frame, convention="dereddened"):
    source_path = tmp_path / "agn.h5"
    source_path.touch()
    monkeypatch.setattr(hubble_utils, "read_quasars_from_hdf5_flat", lambda *_a, **_k: frame.copy())
    monkeypatch.setattr(hubble_utils, "populate_xray", lambda value: value)
    _patch_load_agn_plotters(monkeypatch)
    return hubble_utils.load_agn_data(
        source_path,
        magnitude_convention=convention,
        spectra_fit_h5=None,
        cut_tier="none",
        plot_diagnostics=False,
        plot_path=str(tmp_path / "plots"),
        cut_report_path=None,
    )


def _frame():
    frame = _minimal_agn_frame(n=4)
    frame["z"] = [0.8, 1.5, 2.0, 2.6]
    frame["m_2500_dereddened_err"] = [0.04, 0.05, 0.06, 0.07]
    frame["m_2500_attenuated_model_err"] = [0.03, 0.03, 0.03, 0.03]
    return frame


def test_load_agn_data_off_is_identical(tmp_path, monkeypatch):
    monkeypatch.setenv(DERED_LOW, "0")
    selected, _ = _load(tmp_path, monkeypatch, _frame())
    assert selected["apparent_mag_2500_err"].tolist() == [0.04, 0.05, 0.06, 0.07]
    assert "m_2500_dereddened_err_extra" not in selected.columns
    assert "magnitude_error_extra" not in selected.attrs["cut_configuration_json"]


@pytest.mark.parametrize("convention", ["dereddened", "attenuated"])
def test_load_agn_data_inflates_every_consumer(tmp_path, monkeypatch, convention):
    _proposed(monkeypatch)
    selected, parent = _load(tmp_path, monkeypatch, _frame(), convention=convention)
    dered = np.hypot([0.04, 0.05, 0.06, 0.07], [0.03, 0.03, 0.10, 0.10])
    att = np.hypot([0.03] * 4, [0.02, 0.02, 0.04, 0.04])
    np.testing.assert_allclose(selected["m_2500_dereddened_err"], dered)
    np.testing.assert_allclose(selected["m_2500_attenuated_model_err"], att)
    # The likelihood's magnitude error is the selected convention's inflated error.
    np.testing.assert_allclose(
        selected["apparent_mag_2500_err"], dered if convention == "dereddened" else att
    )
    np.testing.assert_allclose(selected["dm_red_err"], np.hypot(dered, att))
    np.testing.assert_allclose(parent["m_2500_dereddened_err"], dered)
    # The (non-EIV) selection term reads the inflated attenuated error.
    prepared = prepare_completeness_magnitude_columns(selected, "attenuated")
    np.testing.assert_allclose(prepared[COMPLETENESS_MAG_ERR_COL], att)
    configuration = json.loads(selected.attrs["cut_configuration_json"])
    assert configuration["magnitude_error_extra"] == {
        "z_break": 2.0,
        "m_2500_dereddened_err": [0.03, 0.1],
        "m_2500_attenuated_model_err": [0.02, 0.04],
        "error_cuts_column": "apparent_mag_2500_err_raw",
    }
    # The relative-error column (for its optional cut) uses the uninflated error.
    np.testing.assert_allclose(
        selected["rel_apparent_mag_2500_err"],
        selected["apparent_mag_2500_err_raw"] / selected["apparent_mag_2500"].abs(),
    )
    # The originals are kept.
    np.testing.assert_array_equal(selected["m_2500_dereddened_err_raw"], [0.04, 0.05, 0.06, 0.07])
    np.testing.assert_array_equal(
        selected["apparent_mag_2500_err_raw"],
        [0.04, 0.05, 0.06, 0.07] if convention == "dereddened" else [0.03] * 4,
    )


def test_error_cut_acts_on_the_uninflated_error(tmp_path, monkeypatch):
    """QVC_CUT_APPARENT_MAG_2500_ERR_MAX=0.1 keeps the same objects with the term on (0.10 at z >= 2)."""
    from qvc.hubble import hubble_cut_config

    monkeypatch.setattr(hubble_cut_config, "AGN_TIER2_PARAMETER_CUTS", (("apparent_mag_2500_err", None, 0.1),))

    def load():
        source_path = tmp_path / "agn.h5"
        source_path.touch()
        monkeypatch.setattr(hubble_utils, "read_quasars_from_hdf5_flat", lambda *_a, **_k: _frame())
        monkeypatch.setattr(hubble_utils, "populate_xray", lambda value: value)
        _patch_load_agn_plotters(monkeypatch)
        return hubble_utils.load_agn_data(
            source_path, magnitude_convention="dereddened", spectra_fit_h5=None, cut_tier="2",
            plot_diagnostics=False, plot_path=str(tmp_path / "plots"), cut_report_path=None,
        )[0]

    off = load()
    monkeypatch.setenv(DERED_HIGH, "0.10")
    on = load()
    assert len(off) == 4
    assert on["object_id"].tolist() == off["object_id"].tolist()
    # Inflated errors at z >= 2 exceed the threshold; the cut did not see them.
    assert (on.loc[on["z"] >= 2.0, "apparent_mag_2500_err"] > 0.1).all()
    tier2 = json.loads(on.attrs["cut_configuration_json"])["tier2"]
    assert ["apparent_mag_2500_err_raw", None, 0.1] in tier2
    assert ["apparent_mag_2500_err", None, 0.1] in json.loads(off.attrs["cut_configuration_json"])["tier2"]


def test_eiv_likelihood_variance_contains_the_term(tmp_path, monkeypatch):
    from qvc.hubble.hubble_completeness_refactored import COMPLETENESS_MAG_COL
    from qvc.hubble.hubble_eiv_lf import (
        EIV_DU_COL, EIV_DZ_COL, EIV_LNPSI_COL, EIV_U0_COL, EIV_Z0_COL, _eiv_inputs,
    )

    def v_mag(frame):
        agn = {column: frame[column].to_numpy(dtype=float) for column in (
            "apparent_mag_2500", "apparent_mag_2500_err", "z", "log_sigma_uv", "log_tau_uv_rf",
            "log_sigma_uv_std_psd",
        )}
        n = len(frame)
        agn.update({
            COMPLETENESS_MAG_COL: agn["apparent_mag_2500"],
            "log_tau_uv_rf_std_psd": np.full(n, 0.1),
            "log_sigma_uv_log_tau_uv_rf_cov_psd": np.zeros(n),
            EIV_LNPSI_COL: np.zeros((2, 3)), EIV_Z0_COL: 0.0, EIV_DZ_COL: 5.0, EIV_U0_COL: 10.0, EIV_DU_COL: 10.0,
        })
        zeros = np.zeros(n)
        return _eiv_inputs({"log_f": np.log(0.01)}, agn, mu_cosmo=zeros, sigma_lens=zeros,
                           sigma_mu_z=zeros, pivots=(0.0, 0.0))["v_mag"]

    off, _ = _load(tmp_path, monkeypatch, _frame())
    _proposed(monkeypatch)
    on, _ = _load(tmp_path, monkeypatch, _frame())
    np.testing.assert_allclose(v_mag(on) - v_mag(off), [0.03**2, 0.03**2, 0.10**2, 0.10**2])
