from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from qvc.spectra import fit_spectra


def _patch_concat_light_curves(monkeypatch, lc_objects):
    import qvc.light_curve.multiband_generate_lc as multiband_generate_lc

    monkeypatch.setattr(
        multiband_generate_lc,
        "concat_light_curves",
        lambda *args, **kwargs: lc_objects,
    )


def _build_records_args(**overrides):
    values = {
        "fpath_in": None,
        "filter_object_id": ["1458203"],
        "filter_sdss_name": None,
        "N": None,
        "skip": None,
        "dr16q_fits": "dr16q.fits",
        "max_sep": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_spectrum_hdul(loglam, *, wdisp=None):
    loglam = np.asarray(loglam, dtype=float)
    columns = [
        fits.Column(name="loglam", format="D", array=loglam),
        fits.Column(name="flux", format="D", array=np.ones_like(loglam)),
        fits.Column(name="ivar", format="D", array=np.full_like(loglam, 100.0)),
    ]
    if wdisp is not None:
        columns.append(
            fits.Column(name="wdisp", format="D", array=np.asarray(wdisp, dtype=float))
        )
    return fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]
    )


def test_get_spectrum_arrays_estimates_sdss_resolving_power(capsys):
    loglam = 3.6 + np.arange(5, dtype=float) * 1.0e-4
    wdisp = np.array([0.8, 0.9, 1.0, 1.1, 1.2], dtype=float)

    with _make_spectrum_hdul(loglam, wdisp=wdisp) as hdul:
        lam, flux, err, resolving_power = fit_spectra.get_spectrum_arrays(hdul)

    dloglam = float(np.nanmedian(np.diff(loglam)))
    expected = float(
        np.nanmedian(1.0 / (np.log(10.0) * dloglam * 2.355 * wdisp))
    )
    assert len(lam) == len(flux) == len(err) == len(loglam)
    assert resolving_power == pytest.approx(expected)
    assert (
        f"SDSS resolving power estimate R = {resolving_power:.0f}"
        in capsys.readouterr().out
    )


@pytest.mark.parametrize(
    ("loglam", "wdisp"),
    [
        (3.6 + np.arange(3, dtype=float) * 1.0e-4, None),
        (3.6 + np.arange(3, dtype=float) * 1.0e-4, [0.0, np.nan, -1.0]),
        (np.full(3, 3.6, dtype=float), [1.0, 1.0, 1.0]),
    ],
)
def test_get_spectrum_arrays_uses_default_sdss_resolving_power(
    loglam, wdisp, capsys
):
    with _make_spectrum_hdul(loglam, wdisp=wdisp) as hdul:
        _, _, _, resolving_power = fit_spectra.get_spectrum_arrays(hdul)

    assert resolving_power == 2000.0
    assert "SDSS resolving power estimate R = 2000" in capsys.readouterr().out


def test_build_records_uses_s82_for_direct_object_ids(monkeypatch):
    expected = pd.DataFrame([{"object_id": "1458203", "ra": 1.0, "dec": 2.0}])
    calls = []

    def fake_build(object_ids):
        calls.append(object_ids)
        return expected.copy()

    monkeypatch.setattr(fit_spectra, "build_sample_df_from_object_ids", fake_build)
    monkeypatch.setattr(fit_spectra, "prepare_sample_df", lambda df, **kwargs: df)
    monkeypatch.setattr(fit_spectra, "match_to_dr16q", lambda df, *args: df)

    records = fit_spectra.build_records(_build_records_args())

    assert calls == [["1458203"]]
    assert records == [{"object_id": "1458203", "ra": 1.0, "dec": 2.0}]


def test_build_records_accepts_csv_catalog(monkeypatch, tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame([{"object_id": "1458203", "ra": 1.0, "dec": 2.0}]).to_csv(
        csv_path, index=False
    )
    monkeypatch.setattr(fit_spectra, "prepare_sample_df", lambda df, **kwargs: df)
    monkeypatch.setattr(fit_spectra, "match_to_dr16q", lambda df, *args: df)

    records = fit_spectra.build_records(
        _build_records_args(fpath_in=str(csv_path), filter_object_id=None)
    )

    assert records[0]["object_id"] == 1458203


@pytest.mark.parametrize("suffix", [".h5", ".parquet", ".fits"])
def test_build_records_rejects_non_csv_input_without_reading(monkeypatch, suffix):
    monkeypatch.setattr(
        fit_spectra.pd,
        "read_csv",
        lambda *_args, **_kwargs: pytest.fail("read_csv should not be called"),
    )

    with pytest.raises(ValueError, match="supports CSV catalogs only"):
        fit_spectra.build_records(
            _build_records_args(fpath_in=f"catalog{suffix}", filter_object_id=None)
        )


def test_prepare_sample_df_uses_only_concat_psf_photometry(monkeypatch):
    _patch_concat_light_curves(
        monkeypatch,
        [
            {
                "object_id": "1458203",
                "mags_mean": [20.0, 21.0, 19.5, 19.25, 18.9],
                "mags_mean_err": [0.07, 0.06, 0.05, 0.04, 0.03],
            }
        ],
    )
    sample_df = pd.DataFrame(
        [
            {
                "object_id": "1458203",
                "sdss_name": "221958.21+003709.1",
                "z": 3.090241,
                "mags_mean_r": 99.0,
                "mags_mean_i": 98.0,
                "mean_u": 0.10,
                "mean_u_err": 0.07,
                "mean_g": 0.20,
                "mean_g_err": 0.06,
                "mean_r": -0.20,
                "mean_r_err": 0.05,
                "mean_i": 0.10,
                "mean_i_err": 0.04,
            }
        ]
    )

    out = fit_spectra.prepare_sample_df(sample_df, filter_sdss_name=["221958.21+003709.1"])

    row = out.iloc[0]
    assert [row[f"psf_mag_{band}"] for band in "ugri"] == [20.0, 21.0, 19.5, 19.25]
    assert [row[f"psf_mag_err_{band}"] for band in "ugri"] == [0.07, 0.06, 0.05, 0.04]
    assert "mean_corrected_u" not in out.columns


def test_prepare_sample_df_requires_concat_lc_mean(monkeypatch):
    _patch_concat_light_curves(
        monkeypatch,
        [{"object_id": "1458203", "mags_mean": [20.0, 21.0], "mags_mean_err": [0.07, 0.06]}],
    )
    sample_df = pd.DataFrame(
        [
            {
                "object_id": "1458203",
                "sdss_name": "221958.21+003709.1",
                "z": 3.090241,
                "mean_u": 0.10,
                "mean_u_err": 0.07,
                "mean_g": 0.20,
                "mean_g_err": 0.06,
                "mean_r": -0.20,
                "mean_r_err": 0.05,
                "mean_i": 0.10,
                "mean_i_err": 0.04,
            }
        ]
    )

    with pytest.raises(ValueError, match="missing finite psf_mag_r"):
        fit_spectra.prepare_sample_df(sample_df, filter_sdss_name=["221958.21+003709.1"])


def test_prepare_sample_df_ignores_missing_h5_fit_mean(monkeypatch):
    _patch_concat_light_curves(
        monkeypatch,
        [{
            "object_id": "1458203",
            "mags_mean": [20.0, 21.0, 19.5, 19.25, 18.9],
            "mags_mean_err": [0.07, 0.06, 0.05, 0.04, 0.03],
        }],
    )
    sample_df = pd.DataFrame(
        [
            {
                "object_id": "1458203",
                "sdss_name": "221958.21+003709.1",
                "z": 3.090241,
                "mean_u": 0.10,
                "mean_u_err": 0.07,
                "mean_g": 0.20,
                "mean_g_err": 0.06,
                "mean_r_err": 0.05,
                "mean_i": 0.10,
                "mean_i_err": 0.04,
            }
        ]
    )

    out = fit_spectra.prepare_sample_df(sample_df, filter_sdss_name=["221958.21+003709.1"])
    assert np.isclose(out.iloc[0]["psf_mag_r"], 19.5)


def test_prepare_sample_df_requires_positive_concat_mean_error(monkeypatch):
    _patch_concat_light_curves(
        monkeypatch,
        [{
            "object_id": "1458203",
            "mags_mean": [20.0, 21.0, 19.5, 19.25, 18.9],
            "mags_mean_err": [0.07, 0.06, 0.0, 0.04, 0.03],
        }],
    )
    sample_df = pd.DataFrame(
        [
            {
                "object_id": "1458203",
                "sdss_name": "221958.21+003709.1",
                "z": 3.090241,
                "mean_u": 0.10,
                "mean_u_err": 0.07,
                "mean_g": 0.20,
                "mean_g_err": 0.06,
                "mean_r": -0.20,
                "mean_r_err": 0.0,
                "mean_i": 0.10,
                "mean_i_err": 0.04,
            }
        ]
    )

    with pytest.raises(ValueError, match="missing positive finite psf_mag_err_r"):
        fit_spectra.prepare_sample_df(sample_df, filter_sdss_name=["221958.21+003709.1"])


def test_build_psf_photometry_inputs_uses_concat_values_unchanged():
    rec = {
        "z": 3.090241,
        "psf_mag_u": 20.0,
        "psf_mag_err_u": 0.07,
        "psf_mag_g": 21.0,
        "psf_mag_err_g": 0.06,
        "psf_mag_r": 19.5,
        "psf_mag_err_r": 0.05,
        "psf_mag_i": 19.25,
        "psf_mag_err_i": 0.04,
    }

    bands, mags, errs = fit_spectra.build_psf_photometry_inputs(rec)

    assert bands == ["u", "g", "r", "i"]
    assert mags == [20.0, 21.0, 19.5, 19.25]
    assert errs == [0.07, 0.06, 0.05, 0.04]


def test_extract_fit_stats_uses_fitted_arrays_without_local_lya_cutoff():
    q = SimpleNamespace(
        wave=np.array([1100.0, 1300.0, 2000.0], dtype=float),
        flux=np.array([2.0, 4.0, 6.0], dtype=float),
        model_total=np.array([1.0, 2.0, 3.0], dtype=float),
        err=np.array([1.0, 1.0, 1.0], dtype=float),
        numpyro_samples={
            "frac_jitter": np.zeros(1, dtype=float),
            "add_jitter": np.zeros(1, dtype=float),
        },
    )

    out = fit_spectra.extract_fit_stats(q)

    assert out["n_pixels"] == 3
    assert np.isclose(out["chi2"], 14.0)
    assert np.isclose(out["chi2_per_pixel"], 14.0 / 3.0)
    assert np.isclose(out["wrms"], np.sqrt(14.0 / 3.0))
    assert out["wave_min_rf"] == 1100.0


def test_compute_derived_results_uses_host_continuum_draw_ratio(monkeypatch):
    monkeypatch.setattr(
        fit_spectra,
        "estimate_m2500_from_model",
        lambda q: (np.nan, np.nan, np.nan, np.nan),
    )
    monkeypatch.setattr(
        fit_spectra,
        "reconstruct_posterior_components",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected reconstruction")),
    )
    monkeypatch.setattr(fit_spectra, "estimate_pl_psf_bandpass_fractions", lambda *args, **kwargs: {})
    monkeypatch.setattr(fit_spectra, "estimate_agn_psf_bandpass_fractions", lambda *args, **kwargs: {})

    q = SimpleNamespace(
        numpyro_samples={"log_frac_host": np.zeros(3, dtype=float)},
        pred_out={
            "gal_model": np.array(
                [
                    [0.10, 0.10, 0.10],
                    [0.12, 0.12, 0.12],
                    [0.08, 0.08, 0.08],
                ],
                dtype=float,
            ),
            "continuum_model": np.ones((3, 3), dtype=float),
            "f_bc_model": np.zeros((3, 3), dtype=float),
            "f_pl_model": np.ones((3, 3), dtype=float),
            "f_fe_mgii_model": np.zeros((3, 3), dtype=float),
        },
        lam=np.array([4000.0, 6000.0, 8000.0], dtype=float),
        wave=np.array([2000.0, 3000.0, 4000.0], dtype=float),
        z=1.0,
        bi=np.nan,
        bi_err=np.nan,
        _fit_decompose_host=True,
    )

    result = {"z": 1.0}
    args = SimpleNamespace(decompose_host=True)
    fit_spectra.compute_derived_results(result, q, args)

    assert np.isclose(result["f_host_2500"], 0.1)
    assert np.isclose(result["f_host_2500_err"], 0.0136)


def test_estimate_host_2500_fraction_requires_native_2500_coverage(monkeypatch):
    monkeypatch.setattr(
        fit_spectra,
        "reconstruct_posterior_components",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected reconstruction")),
    )

    q = SimpleNamespace(
        wave=np.array([3000.0, 4000.0, 5000.0], dtype=float),
        flux=np.ones(3, dtype=float),
        numpyro_samples={"PL_norm": np.ones(3, dtype=float)},
        pred_out={"fsps_weights": np.ones((3, 1), dtype=float)},
        _fit_prior_config={"PL_pivot": 4000.0},
        _fit_fsps_age_grid=np.array([1.0], dtype=float),
        _fit_fsps_logzsol_grid=np.array([0.0], dtype=float),
        _fit_dsps_ssp_fn="tempdata.h5",
        _fit_fit_poly=True,
        _fit_fit_reddening=False,
        _fit_fit_poly_order=2,
        _fit_custom_components=(),
        fe_uv_wave=np.array([2000.0, 3000.0], dtype=float),
        fe_uv_flux=np.zeros(2, dtype=float),
        fe_op_wave=np.array([2000.0, 3000.0], dtype=float),
        fe_op_flux=np.zeros(2, dtype=float),
    )

    median, err = fit_spectra.estimate_host_2500_fraction(q)

    assert np.isnan(median)
    assert np.isnan(err)


def test_estimate_host_psf_2500_fraction_ignores_scalar_fallback_without_native_coverage():
    q = SimpleNamespace(
        wave=np.array([3000.0, 4000.0, 5000.0], dtype=float),
        pred_out={
            "gal_model_psf": np.ones((3, 3), dtype=float),
            "agn_model_psf": np.ones((3, 3), dtype=float),
        },
        frac_host_psf_2500=0.42,
    )

    median, err = fit_spectra.estimate_host_psf_2500_fraction(q)

    assert np.isnan(median)
    assert np.isnan(err)


def test_estimate_host_center_fraction_ignores_scalar_fallback_without_components():
    q = SimpleNamespace(
        wave=np.array([2000.0, 3000.0, 4000.0], dtype=float),
        lam=np.array([4000.0, 6000.0, 8000.0], dtype=float),
        z=1.0,
        pred_out={},
        frac_host_pivot=0.42,
    )

    median, err = fit_spectra.estimate_host_center_fraction(q)

    assert np.isnan(median)
    assert np.isnan(err)


def test_compute_derived_results_uses_bc_over_total_continuum(monkeypatch):
    monkeypatch.setattr(
        fit_spectra,
        "estimate_m2500_from_model",
        lambda q: (np.nan, np.nan, np.nan, np.nan),
    )
    monkeypatch.setattr(
        fit_spectra,
        "reconstruct_posterior_components",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected reconstruction")),
    )
    monkeypatch.setattr(fit_spectra, "estimate_pl_psf_bandpass_fractions", lambda *args, **kwargs: {})
    monkeypatch.setattr(fit_spectra, "estimate_agn_psf_bandpass_fractions", lambda *args, **kwargs: {})

    q = SimpleNamespace(
        numpyro_samples={"log_frac_host": np.zeros(3, dtype=float)},
        pred_out={
            "gal_model": np.zeros((3, 3), dtype=float),
            "continuum_model": np.array(
                [
                    [2.0, 2.0, 2.0],
                    [2.5, 2.5, 2.5],
                    [1.5, 1.5, 1.5],
                ],
                dtype=float,
            ),
            "f_bc_model": np.array(
                [
                    [0.20, 0.20, 0.20],
                    [0.25, 0.25, 0.25],
                    [0.15, 0.15, 0.15],
                ],
                dtype=float,
            ),
            "f_pl_model": np.ones((3, 3), dtype=float),
            "f_fe_mgii_model": np.array(
                [
                    [0.20, 0.20, 0.20],
                    [0.25, 0.25, 0.25],
                    [0.15, 0.15, 0.15],
                ],
                dtype=float,
            ),
        },
        lam=np.array([4000.0, 6000.0, 8000.0], dtype=float),
        wave=np.array([2000.0, 3000.0, 4000.0], dtype=float),
        z=1.0,
        bi=np.nan,
        bi_err=np.nan,
        _fit_decompose_host=True,
    )

    result = {"z": 1.0}
    args = SimpleNamespace(decompose_host=True)
    fit_spectra.compute_derived_results(result, q, args)

    bc_ratio_draws = q.pred_out["f_bc_model"][:, 1] / q.pred_out["continuum_model"][:, 1]
    fe_ratio_draws = q.pred_out["f_fe_mgii_model"][:, 1] / q.pred_out["continuum_model"][:, 1]
    bc_p16, bc_p50, bc_p84 = np.percentile(bc_ratio_draws, [16, 50, 84])
    fe_p16, fe_p50, fe_p84 = np.percentile(fe_ratio_draws, [16, 50, 84])

    assert np.isclose(result["f_bc_3000"], bc_p50)
    assert np.isclose(result["f_bc_3000_err"], 0.5 * (bc_p84 - bc_p16))
    assert np.isclose(result["f_fe_uv_3000"], fe_p50)
    assert np.isclose(result["f_fe_uv_3000_err"], 0.5 * (fe_p84 - fe_p16))


def test_compute_derived_results_saves_narrow_line_to_continuum_integrated_fraction(monkeypatch):
    monkeypatch.setattr(
        fit_spectra,
        "estimate_m2500_from_model",
        lambda q: (np.nan, np.nan, np.nan, np.nan),
    )
    monkeypatch.setattr(
        fit_spectra,
        "reconstruct_posterior_components",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected reconstruction")),
    )
    monkeypatch.setattr(fit_spectra, "estimate_pl_psf_bandpass_fractions", lambda *args, **kwargs: {})
    monkeypatch.setattr(fit_spectra, "estimate_agn_psf_bandpass_fractions", lambda *args, **kwargs: {})

    q = SimpleNamespace(
        numpyro_samples={"log_frac_host": np.zeros(3, dtype=float)},
        pred_out={
            "gal_model": np.array(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=float,
            ),
            "continuum_model": np.ones((3, 3), dtype=float),
            "line_model_narrow": np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 3.0, 0.0],
                ],
                dtype=float,
            ),
            "line_model_broad": np.array(
                [
                    [0.0, 2.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [0.0, 6.0, 0.0],
                ],
                dtype=float,
            ),
            "f_bc_model": np.zeros((3, 3), dtype=float),
            "f_pl_model": np.ones((3, 3), dtype=float),
            "f_fe_mgii_model": np.zeros((3, 3), dtype=float),
        },
        lam=np.array([4000.0, 6000.0, 8000.0], dtype=float),
        wave=np.array([2000.0, 3000.0, 4000.0], dtype=float),
        z=1.0,
        bi=np.nan,
        bi_err=np.nan,
        _fit_decompose_host=True,
    )

    result = {"z": 1.0}
    args = SimpleNamespace(decompose_host=True)
    fit_spectra.compute_derived_results(result, q, args)

    na_int = np.trapezoid(q.pred_out["line_model_narrow"], q.wave, axis=1)
    br_int = np.trapezoid(q.pred_out["line_model_broad"], q.wave, axis=1)
    cont_int = np.trapezoid(q.pred_out["continuum_model"], q.wave, axis=1)
    na_ratio_draws = na_int / cont_int
    br_ratio_draws = br_int / cont_int
    na_p16, na_p50, na_p84 = np.percentile(na_ratio_draws, [16, 50, 84])
    br_p16, br_p50, br_p84 = np.percentile(br_ratio_draws, [16, 50, 84])

    assert np.isclose(result["f_na"], na_p50)
    assert np.isclose(result["f_na_err"], 0.5 * (na_p84 - na_p16))
    assert np.isclose(result["f_br"], br_p50)
    assert np.isclose(result["f_br_err"], 0.5 * (br_p84 - br_p16))


def test_compute_derived_results_saves_narrow_line_fraction_without_host_decomposition(monkeypatch):
    monkeypatch.setattr(
        fit_spectra,
        "estimate_m2500_from_model",
        lambda q: (np.nan, np.nan, np.nan, np.nan),
    )
    monkeypatch.setattr(
        fit_spectra,
        "reconstruct_posterior_components",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected reconstruction")),
    )
    monkeypatch.setattr(fit_spectra, "estimate_pl_psf_bandpass_fractions", lambda *args, **kwargs: {})
    monkeypatch.setattr(fit_spectra, "estimate_agn_psf_bandpass_fractions", lambda *args, **kwargs: {})

    q = SimpleNamespace(
        numpyro_samples={"log_frac_host": np.zeros(3, dtype=float)},
        pred_out={
            "gal_model": np.zeros((3, 3), dtype=float),
            "continuum_model": np.array(
                [
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [4.0, 4.0, 4.0],
                ],
                dtype=float,
            ),
            "line_model_narrow": np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 4.0, 0.0],
                ],
                dtype=float,
            ),
            "line_model_broad": np.array(
                [
                    [0.0, 0.5, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 2.0, 0.0],
                ],
                dtype=float,
            ),
            "f_bc_model": np.zeros((3, 3), dtype=float),
            "f_pl_model": np.ones((3, 3), dtype=float),
            "f_fe_mgii_model": np.zeros((3, 3), dtype=float),
        },
        lam=np.array([4000.0, 6000.0, 8000.0], dtype=float),
        wave=np.array([2000.0, 3000.0, 4000.0], dtype=float),
        z=1.0,
        bi=np.nan,
        bi_err=np.nan,
        _fit_decompose_host=False,
    )

    result = {"z": 1.0}
    args = SimpleNamespace(decompose_host=False)
    fit_spectra.compute_derived_results(result, q, args)

    na_int = np.trapezoid(q.pred_out["line_model_narrow"], q.wave, axis=1)
    br_int = np.trapezoid(q.pred_out["line_model_broad"], q.wave, axis=1)
    cont_int = np.trapezoid(q.pred_out["continuum_model"], q.wave, axis=1)
    na_ratio_draws = na_int / cont_int
    br_ratio_draws = br_int / cont_int
    na_p16, na_p50, na_p84 = np.percentile(na_ratio_draws, [16, 50, 84])
    br_p16, br_p50, br_p84 = np.percentile(br_ratio_draws, [16, 50, 84])

    assert result["f_host_2500"] == 0.0
    assert result["f_host_center"] == 0.0
    assert np.isclose(result["f_na"], na_p50)
    assert np.isclose(result["f_na_err"], 0.5 * (na_p84 - na_p16))
    assert np.isclose(result["f_br"], br_p50)
    assert np.isclose(result["f_br_err"], 0.5 * (br_p84 - br_p16))


def test_estimate_pl_psf_bandpass_fractions_uses_reconstructed_draws():
    captured = {}

    filters = fit_spectra.get_sdss_filters()
    u_wave_obs = fit_spectra.get_filter_wavelength_angstrom(filters["u"])
    g_wave_obs = fit_spectra.get_filter_wavelength_angstrom(filters["g"])
    r_wave_obs = fit_spectra.get_filter_wavelength_angstrom(filters["r"])
    expected_wave_rf = np.unique(np.concatenate([u_wave_obs, g_wave_obs, r_wave_obs]) / (1.0 + 0.8))

    q = SimpleNamespace(
        z=0.8,
        wave=np.array([2000.0, 4000.0, 6000.0], dtype=float),
        pred_out={
            "f_pl_model": np.full((3, 3), 2.0, dtype=float),
            "scale_psf": np.ones(3, dtype=float),
            "eta_psf": np.ones(3, dtype=float),
            "line_model_psf": np.zeros((3, 3), dtype=float),
        },
        reconstruct_posterior_spectrum=lambda **kwargs: (
            captured.setdefault("wave_out", np.asarray(kwargs["wave_out"], dtype=float)),
            {
                "wave": np.asarray(kwargs["wave_out"], dtype=float),
                "draws": {
                    "PL": np.full((3, len(kwargs["wave_out"])), 2.0, dtype=float),
                    "host": np.full((3, len(kwargs["wave_out"])), 1.0, dtype=float),
                    "Fe_uv": np.zeros((3, len(kwargs["wave_out"])), dtype=float),
                    "Fe_op": np.zeros((3, len(kwargs["wave_out"])), dtype=float),
                    "Balmer_cont": np.zeros((3, len(kwargs["wave_out"])), dtype=float),
                    "continuum": np.full((3, len(kwargs["wave_out"])), 3.0, dtype=float),
                },
            },
        )[1],
    )

    out = fit_spectra.estimate_pl_psf_bandpass_fractions(q, bands=("u", "g", "r"))

    assert np.allclose(captured["wave_out"], expected_wave_rf)
    assert np.isclose(out["u"][0], 2.0 / 3.0)
    assert np.isclose(out["g"][0], 2.0 / 3.0)
    assert np.isclose(out["r"][0], 2.0 / 3.0)


def test_estimate_pl_psf_bandpass_fractions_ignores_nan_line_draws():
    q = SimpleNamespace(
        z=0.8,
        wave=np.array([2000.0, 4000.0, 6000.0], dtype=float),
        pred_out={
            "f_pl_model": np.full((3, 3), 2.0, dtype=float),
            "scale_psf": np.ones(3, dtype=float),
            "eta_psf": np.ones(3, dtype=float),
            "line_model_psf": np.array(
                [
                    [0.0, np.nan, 0.0],
                    [0.0, np.nan, 0.0],
                    [0.0, np.nan, 0.0],
                ],
                dtype=float,
            ),
        },
        reconstruct_posterior_spectrum=lambda **kwargs: {
            "wave": np.asarray([1800.0, 3000.0, 5000.0], dtype=float),
            "draws": {
                "PL": np.full((3, 3), 2.0, dtype=float),
                "host": np.full((3, 3), 1.0, dtype=float),
                "Fe_uv": np.zeros((3, 3), dtype=float),
                "Fe_op": np.zeros((3, 3), dtype=float),
                "Balmer_cont": np.zeros((3, 3), dtype=float),
                "continuum": np.full((3, 3), 3.0, dtype=float),
            },
        },
    )

    out = fit_spectra.estimate_pl_psf_bandpass_fractions(q, bands=("g", "r"))

    assert np.isfinite(out["g"][0])
    assert np.isfinite(out["r"][0])
    assert np.isclose(out["g"][0], 2.0 / 3.0)
    assert np.isclose(out["r"][0], 2.0 / 3.0)


def test_estimate_pl_psf_bandpass_fractions_uses_full_posterior_line_reconstruction(monkeypatch):
    monkeypatch.setattr(
        fit_spectra,
        "_reconstruct_line_psf_draws_on_wave",
        lambda q, wave_out, n_use: np.ones((n_use, len(wave_out)), dtype=float),
    )

    q = SimpleNamespace(
        z=0.8,
        wave=np.array([2000.0, 4000.0, 6000.0], dtype=float),
        pred_out={
            "f_pl_model": np.full((3, 3), 2.0, dtype=float),
            "scale_psf": np.ones(3, dtype=float),
            "eta_psf": np.ones(3, dtype=float),
            "line_model_psf": np.array([[np.nan], [np.nan], [np.nan]], dtype=float),
        },
        reconstruct_posterior_spectrum=lambda **kwargs: {
            "wave": np.asarray([1800.0, 3000.0, 5000.0], dtype=float),
            "draws": {
                "PL": np.full((3, 3), 2.0, dtype=float),
                "host": np.full((3, 3), 1.0, dtype=float),
                "Fe_uv": np.zeros((3, 3), dtype=float),
                "Fe_op": np.zeros((3, 3), dtype=float),
                "Balmer_cont": np.zeros((3, 3), dtype=float),
                "continuum": np.full((3, 3), 3.0, dtype=float),
            },
        },
    )

    out = fit_spectra.estimate_pl_psf_bandpass_fractions(q, bands=("g",))

    assert np.isfinite(out["g"][0])
    assert np.isclose(out["g"][0], 2.0 / 4.0)


def test_estimate_agn_psf_bandpass_fractions_includes_broad_lines_but_not_narrow(monkeypatch):
    monkeypatch.setattr(
        fit_spectra,
        "_reconstruct_line_psf_draws_on_wave",
        lambda q, wave_out, n_use, return_components=False: (
            {
                "broad": np.full((n_use, len(wave_out)), 2.0, dtype=float),
                "narrow": np.full((n_use, len(wave_out)), 1.0, dtype=float),
                "total": np.full((n_use, len(wave_out)), 3.0, dtype=float),
            }
            if return_components
            else np.full((n_use, len(wave_out)), 3.0, dtype=float)
        ),
    )

    q = SimpleNamespace(
        z=0.8,
        wave=np.array([2000.0, 4000.0, 6000.0], dtype=float),
        pred_out={
            "scale_psf": np.ones(3, dtype=float),
            "eta_psf": np.ones(3, dtype=float),
        },
        reconstruct_posterior_spectrum=lambda **kwargs: {
            "wave": np.asarray(kwargs["wave_out"], dtype=float),
            "draws": {
                "PL": np.full((3, len(kwargs["wave_out"])), 2.0, dtype=float),
                "host": np.full((3, len(kwargs["wave_out"])), 1.0, dtype=float),
                "Fe_uv": np.zeros((3, len(kwargs["wave_out"])), dtype=float),
                "Fe_op": np.zeros((3, len(kwargs["wave_out"])), dtype=float),
                "Balmer_cont": np.zeros((3, len(kwargs["wave_out"])), dtype=float),
                "continuum": np.full((3, len(kwargs["wave_out"])), 3.0, dtype=float),
            },
        },
    )

    out = fit_spectra.estimate_agn_psf_bandpass_fractions(q, bands=("g",))

    assert np.isfinite(out["g"][0])
    assert np.isclose(out["g"][0], 4.0 / 6.0)


def test_reconstruct_line_psf_draws_keeps_only_native_supported_lines(monkeypatch):
    full_meta = {
        "n_lines": 2,
        "vgroup": np.array([0, 1], dtype=int),
        "wgroup": np.array([0, 1], dtype=int),
        "fgroup": np.array([0, 1], dtype=int),
        "flux_ratio": np.array([1.0, 1.0], dtype=float),
        "ln_lambda0": np.log(np.array([3000.0, 7000.0], dtype=float)),
        "amp_init_group": np.array([2.0, 4.0], dtype=float),
        "names": ["narrow_in_1", "narrow_out_1"],
    }
    native_meta = {
        "n_lines": 1,
        "vgroup": np.array([0], dtype=int),
        "wgroup": np.array([0], dtype=int),
        "fgroup": np.array([0], dtype=int),
        "flux_ratio": np.array([1.0], dtype=float),
        "ln_lambda0": np.log(np.array([3000.0], dtype=float)),
        "amp_init_group": np.array([2.0], dtype=float),
        "names": ["narrow_in_1"],
    }
    calls = []

    def fake_build(_line_table, wave):
        wave = np.asarray(wave, dtype=float)
        return native_meta if np.nanmax(wave) <= 4000.0 else full_meta

    def fake_many_gauss(_lnwave, amps, mus, sigs):
        calls.append((np.asarray(amps, dtype=float), np.asarray(mus, dtype=float), np.asarray(sigs, dtype=float)))
        return np.zeros_like(_lnwave, dtype=float)

    monkeypatch.setattr(fit_spectra, "_extract_line_table_from_prior_config", lambda _cfg: object())
    monkeypatch.setattr(fit_spectra, "build_tied_line_meta_from_linelist", fake_build)
    monkeypatch.setattr(fit_spectra, "_broad_line_mask", lambda names: np.array([0.0 if "narrow" in name else 1.0 for name in names], dtype=float))
    monkeypatch.setattr(fit_spectra, "_many_gauss_lnlam", fake_many_gauss)

    q = SimpleNamespace(
        wave=np.array([2500.0, 3500.0, 4000.0], dtype=float),
        numpyro_samples={
            "line_dmu_group": np.array([[0.1]], dtype=float),
            "line_sig_group": np.array([[0.5]], dtype=float),
            "line_amp_group": np.array([[6.0]], dtype=float),
        },
        pred_out={"scale_psf": np.ones(1, dtype=float), "eta_psf": np.ones(1, dtype=float)},
        _fit_prior_config={},
        _fit_custom_line_components=(),
    )

    out = fit_spectra._reconstruct_line_psf_draws_on_wave(q, np.array([3000.0, 7000.0], dtype=float), n_use=1)

    assert out.shape == (1, 2)
    assert len(calls) == 2
    narrow_amps, narrow_mus, narrow_sigs = calls[1]
    assert np.allclose(narrow_amps, [6.0, 0.0])
    assert np.allclose(narrow_sigs, [0.5, 0.0])
    assert np.allclose(narrow_mus, np.log([3000.0, 7000.0]) + np.array([0.1, 0.0]))


def test_reconstruct_line_psf_draws_does_not_extrapolate_family_lines(monkeypatch):
    full_meta = {
        "n_lines": 3,
        "vgroup": np.array([0, 1, 2], dtype=int),
        "wgroup": np.array([0, 1, 2], dtype=int),
        "fgroup": np.array([0, 1, 2], dtype=int),
        "flux_ratio": np.ones(3, dtype=float),
        "ln_lambda0": np.log(np.array([3000.0, 3500.0, 8000.0], dtype=float)),
        "amp_init_group": np.array([2.0, 4.0, 6.0], dtype=float),
        "names": ["broad_a_1", "broad_b_1", "broad_out_1"],
    }
    native_meta = {
        "n_lines": 2,
        "vgroup": np.array([0, 1], dtype=int),
        "wgroup": np.array([0, 1], dtype=int),
        "fgroup": np.array([0, 1], dtype=int),
        "flux_ratio": np.ones(2, dtype=float),
        "ln_lambda0": np.log(np.array([3000.0, 3500.0], dtype=float)),
        "amp_init_group": np.array([2.0, 4.0], dtype=float),
        "names": ["broad_a_1", "broad_b_1"],
    }
    calls = []

    def fake_build(_line_table, wave):
        wave = np.asarray(wave, dtype=float)
        return native_meta if np.nanmax(wave) <= 4000.0 else full_meta

    def fake_many_gauss(_lnwave, amps, mus, sigs):
        calls.append((np.asarray(amps, dtype=float), np.asarray(mus, dtype=float), np.asarray(sigs, dtype=float)))
        return np.zeros_like(_lnwave, dtype=float)

    monkeypatch.setattr(fit_spectra, "_extract_line_table_from_prior_config", lambda _cfg: object())
    monkeypatch.setattr(fit_spectra, "build_tied_line_meta_from_linelist", fake_build)
    monkeypatch.setattr(fit_spectra, "_broad_line_mask", lambda names: np.array([1.0 if "broad" in name else 0.0 for name in names], dtype=float))
    monkeypatch.setattr(fit_spectra, "_many_gauss_lnlam", fake_many_gauss)

    q = SimpleNamespace(
        wave=np.array([2500.0, 3500.0, 4000.0], dtype=float),
        numpyro_samples={
            "line_dmu_group": np.array([[0.2, 0.6]], dtype=float),
            "line_sig_group": np.array([[1.0, 3.0]], dtype=float),
            "line_amp_group": np.array([[4.0, 8.0]], dtype=float),
        },
        pred_out={"scale_psf": np.ones(1, dtype=float), "eta_psf": np.ones(1, dtype=float)},
        _fit_prior_config={},
        _fit_custom_line_components=(),
    )

    fit_spectra._reconstruct_line_psf_draws_on_wave(q, np.array([3000.0, 3500.0, 8000.0], dtype=float), n_use=1)

    broad_amps, broad_mus, broad_sigs = calls[0]
    assert np.allclose(broad_amps, [4.0, 8.0, 0.0])
    assert np.allclose(broad_sigs, [1.0, 3.0, 0.0])
    assert np.allclose(broad_mus, np.log([3000.0, 3500.0, 8000.0]) + np.array([0.2, 0.6, 0.0]))


def test_reconstruct_line_psf_draws_does_not_use_family_median_when_draw_has_no_anchor(monkeypatch):
    full_meta = {
        "n_lines": 2,
        "vgroup": np.array([0, 1], dtype=int),
        "wgroup": np.array([0, 1], dtype=int),
        "fgroup": np.array([0, 1], dtype=int),
        "flux_ratio": np.ones(2, dtype=float),
        "ln_lambda0": np.log(np.array([3200.0, 8200.0], dtype=float)),
        "amp_init_group": np.array([2.0, 6.0], dtype=float),
        "names": ["broad_in_1", "broad_out_1"],
    }
    native_meta = {
        "n_lines": 1,
        "vgroup": np.array([0], dtype=int),
        "wgroup": np.array([0], dtype=int),
        "fgroup": np.array([0], dtype=int),
        "flux_ratio": np.ones(1, dtype=float),
        "ln_lambda0": np.log(np.array([3200.0], dtype=float)),
        "amp_init_group": np.array([2.0], dtype=float),
        "names": ["broad_in_1"],
    }
    calls = []

    def fake_build(_line_table, wave):
        wave = np.asarray(wave, dtype=float)
        return native_meta if np.nanmax(wave) <= 4000.0 else full_meta

    def fake_many_gauss(_lnwave, amps, mus, sigs):
        calls.append((np.asarray(amps, dtype=float), np.asarray(mus, dtype=float), np.asarray(sigs, dtype=float)))
        return np.zeros_like(_lnwave, dtype=float)

    monkeypatch.setattr(fit_spectra, "_extract_line_table_from_prior_config", lambda _cfg: object())
    monkeypatch.setattr(fit_spectra, "build_tied_line_meta_from_linelist", fake_build)
    monkeypatch.setattr(fit_spectra, "_broad_line_mask", lambda names: np.array([1.0 if "broad" in name else 0.0 for name in names], dtype=float))
    monkeypatch.setattr(fit_spectra, "_many_gauss_lnlam", fake_many_gauss)

    q = SimpleNamespace(
        wave=np.array([2500.0, 3500.0, 4000.0], dtype=float),
        numpyro_samples={
            "line_dmu_group": np.array([[0.0], [0.0]], dtype=float),
            "line_sig_group": np.array([[1.5], [1.5]], dtype=float),
            "line_amp_group": np.array([[0.0], [4.0]], dtype=float),
        },
        pred_out={"scale_psf": np.ones(2, dtype=float), "eta_psf": np.ones(2, dtype=float)},
        _fit_prior_config={},
        _fit_custom_line_components=(),
    )

    fit_spectra._reconstruct_line_psf_draws_on_wave(q, np.array([3200.0, 8200.0], dtype=float), n_use=2)

    draw0_broad_amps = calls[0][0]
    draw1_broad_amps = calls[2][0]
    assert np.allclose(draw0_broad_amps, [0.0, 0.0])
    assert np.allclose(draw1_broad_amps, [4.0, 0.0])


def test_reconstruct_line_psf_draws_requires_posterior_psf_scaling(monkeypatch):
    meta = {
        "n_lines": 1,
        "vgroup": np.array([0], dtype=int),
        "wgroup": np.array([0], dtype=int),
        "fgroup": np.array([0], dtype=int),
        "flux_ratio": np.array([1.0], dtype=float),
        "ln_lambda0": np.log(np.array([3000.0], dtype=float)),
        "amp_init_group": np.array([2.0], dtype=float),
        "names": ["broad_a_1"],
    }

    monkeypatch.setattr(fit_spectra, "_extract_line_table_from_prior_config", lambda _cfg: object())
    monkeypatch.setattr(fit_spectra, "build_tied_line_meta_from_linelist", lambda _line_table, _wave: meta)
    monkeypatch.setattr(fit_spectra, "_broad_line_mask", lambda names: np.ones(len(names), dtype=float))

    q = SimpleNamespace(
        wave=np.array([2500.0, 3500.0], dtype=float),
        numpyro_samples={
            "line_dmu_group": np.array([[0.0]], dtype=float),
            "line_sig_group": np.array([[1.0]], dtype=float),
            "line_amp_group": np.array([[4.0]], dtype=float),
        },
        pred_out={},
        scale_psf=1.0,
        eta_psf=1.0,
        _fit_prior_config={},
        _fit_custom_line_components=(),
    )

    out = fit_spectra._reconstruct_line_psf_draws_on_wave(q, np.array([3000.0], dtype=float), n_use=1)

    assert np.isnan(out).all()


def test_estimate_m2500_from_model_uses_predictive_scale_and_reddening():
    pl_norm = np.array([1.0, 2.0, 4.0], dtype=float)
    pl_slope = np.array([-1.0, 0.0, 1.0], dtype=float)
    scale_psf = np.array([0.8, 1.0, 1.2], dtype=float)
    reddening_a2500 = np.array([0.1, 0.2, 0.4], dtype=float)
    pl_pivot = 2000.0
    q = SimpleNamespace(
        numpyro_samples={
            "PL_norm": pl_norm,
            "PL_slope": pl_slope,
            "scale_psf": np.full(3, 99.0, dtype=float),
            "reddening_ebv": np.full(3, 99.0, dtype=float),
        },
        pred_out={
            "scale_psf": scale_psf,
            "reddening_a2500": reddening_a2500,
        },
        scale_psf=99.0,
        _fit_prior_config={
            "PL_pivot": pl_pivot,
            "reddening_uv_ref": 2500.0,
            "reddening_alpha": 1.2,
        },
    )

    out = fit_spectra.estimate_m2500_from_model(q)

    f_lambda_intrinsic = scale_psf * pl_norm * (2500.0 / pl_pivot) ** pl_slope
    f_lambda_reddened = f_lambda_intrinsic * 10.0 ** (-0.4 * reddening_a2500)
    f_nu_factor = 1e-17 * 2500.0**2 / 2.99792458e18
    intrinsic_samples = -2.5 * np.log10(f_lambda_intrinsic * f_nu_factor) - 48.60
    reddened_samples = -2.5 * np.log10(f_lambda_reddened * f_nu_factor) - 48.60
    expected_reddened = fit_spectra.sym_percentile(reddened_samples)[:2]
    expected_intrinsic = fit_spectra.sym_percentile(intrinsic_samples)[:2]

    assert np.allclose(out, (*expected_reddened, *expected_intrinsic))
    assert out[1] > 0.0
    assert out[3] > 0.0
    assert out[0] >= out[2]


def test_estimate_m2500_from_model_requires_predictive_scale_psf():
    q = SimpleNamespace(
        numpyro_samples={
            "PL_norm": np.ones(3, dtype=float),
            "PL_slope": np.zeros(3, dtype=float),
            "scale_psf": np.ones(3, dtype=float),
        },
        pred_out={"reddening_a2500": np.zeros(3, dtype=float)},
        scale_psf=2.0,
        _fit_prior_config={"PL_pivot": 2500.0},
    )

    out = fit_spectra.estimate_m2500_from_model(q)

    assert all(np.isnan(value) for value in out)


def test_estimate_m2500_from_model_requires_predictive_reddening():
    q = SimpleNamespace(
        numpyro_samples={
            "PL_norm": np.ones(3, dtype=float),
            "PL_slope": np.zeros(3, dtype=float),
            "reddening_a2500": np.zeros(3, dtype=float),
        },
        pred_out={"scale_psf": np.ones(3, dtype=float)},
        _fit_prior_config={"PL_pivot": 2500.0},
    )

    out = fit_spectra.estimate_m2500_from_model(q)

    assert all(np.isnan(value) for value in out)


def test_estimate_m2500_from_model_requires_fitted_pl_pivot():
    q = SimpleNamespace(
        numpyro_samples={
            "PL_norm": np.ones(3, dtype=float),
            "PL_slope": np.zeros(3, dtype=float),
        },
        pred_out={
            "scale_psf": np.ones(3, dtype=float),
            "reddening_a2500": np.zeros(3, dtype=float),
        },
        wave=np.array([2000.0, 3000.0], dtype=float),
        _fit_prior_config={},
    )

    out = fit_spectra.estimate_m2500_from_model(q)

    assert all(np.isnan(value) for value in out)


def test_print_spectrum_diagnostics_includes_psf_and_broader_metrics(capsys):
    result = {
        "object_id": "obj-1",
        "sdss_name": "J0000+0000",
        "z": 1.2345,
        "bands_used": "gri",
        "f_AGN_psf_g": 0.83,
        "f_AGN_psf_g_err": 0.03,
        "f_PL_psf_g": 0.61,
        "f_PL_psf_g_err": 0.04,
        "f_PL_psf_r": 0.72,
        "f_PL_psf_r_err": np.nan,
        "f_PL": 0.55,
        "f_PL_err": 0.02,
        "f_host_2500": 0.12,
        "f_host_2500_err": 0.03,
        "frac_host_psf_2500": 0.09,
        "frac_host_psf_2500_err": 0.01,
        "f_host_center": 0.20,
        "f_host_center_err": 0.02,
        "apparent_mag_2500": 18.1,
        "apparent_mag_2500_err": 0.3,
        "apparent_mag_2500_intrinsic": 17.8,
        "apparent_mag_2500_intrinsic_err": 0.2,
    }

    fit_spectra.print_spectrum_diagnostics(result)

    out = capsys.readouterr().out
    assert "Spectrum diagnostics for object_id=obj-1" in out
    assert "Context: sdss_name=J0000+0000, z=1.23450, bands_used=gri" in out
    assert "PSF variable-AGN fractions:" in out
    assert "g: 0.8300 +/- 0.0300" in out
    assert "PSF pure-PL fractions:" in out
    assert "g: 0.6100 +/- 0.0400" in out
    assert "r: 0.7200" in out
    assert "f_host_2500: 0.1200 +/- 0.0300" in out
    assert "apparent_mag_2500: 18.1000 +/- 0.3000" in out


def test_print_spectrum_diagnostics_omits_nan_rows_and_uses_not_available(capsys):
    result = {
        "object_id": "obj-2",
        "f_AGN_psf_g": np.nan,
        "f_AGN_psf_g_err": np.nan,
        "f_PL_psf_g": np.nan,
        "f_PL_psf_g_err": np.nan,
        "apparent_mag_2500": np.nan,
        "apparent_mag_2500_err": np.nan,
    }

    fit_spectra.print_spectrum_diagnostics(result)

    out = capsys.readouterr().out
    assert "Spectrum diagnostics for object_id=obj-2" in out
    assert "PSF variable-AGN fractions:\n    not available" in out
    assert "PSF pure-PL fractions:\n    not available" in out
    assert "Broader diagnostics:\n    not available" in out
    assert "apparent_mag_2500" not in out


def test_run_one_fit_prints_consolidated_diagnostics_and_not_old_m2500_lines(monkeypatch, capsys, tmp_path):
    diagnostics_calls = []
    config_calls = []
    fit_calls = []

    class DummyHDUL:
        def close(self):
            return None

    class DummyJAXQSOFit:
        def __init__(self, config):
            config_calls.append(config)
            self.config = config
            self.flux = np.asarray(config.spectroscopy.fluxes, dtype=float)
            self.err = np.asarray(config.spectroscopy.errors, dtype=float)
            self.wave = np.array([2000.0, 3000.0, 4000.0], dtype=float)
            self.model_total = np.array([1.0, 1.0, 1.0], dtype=float)
            self.numpyro_samples = {"frac_jitter": np.zeros(1, dtype=float), "add_jitter": np.zeros(1, dtype=float)}
            self.pred_out = {}
            self.ra = config.observation.ra
            self.dec = config.observation.dec
            self.z = config.observation.redshift

        def fit(self, **kwargs):
            fit_calls.append(kwargs)
            return None

        def plot_mcmc_diagnostics(self, **kwargs):
            diagnostics_calls.append(kwargs)

    monkeypatch.setattr(fit_spectra, "load_spec_from_cache", lambda *args, **kwargs: DummyHDUL())
    monkeypatch.setattr(fit_spectra, "get_spectrum_arrays", lambda _hdul: (
        np.array([4000.0, 5000.0, 6000.0], dtype=float),
        np.array([1.0, 1.0, 1.0], dtype=float),
        np.array([0.1, 0.1, 0.1], dtype=float),
        1950.0,
    ))
    monkeypatch.setattr(fit_spectra, "JAXQSOFit", DummyJAXQSOFit)
    monkeypatch.setattr(fit_spectra, "build_default_prior_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(fit_spectra, "build_psf_photometry_inputs", lambda rec: (["g", "r"], [19.0, 18.5], [0.1, 0.1]))
    monkeypatch.setattr(fit_spectra, "extract_named_results", lambda q: {})
    monkeypatch.setattr(fit_spectra, "extract_scalar_attrs", lambda q: {})
    monkeypatch.setattr(fit_spectra, "extract_fit_stats", lambda q: {})

    def fake_compute(result, q, args):
        result["f_AGN_psf_g"] = 0.81
        result["f_AGN_psf_g_err"] = 0.04
        result["f_PL_psf_g"] = 0.65
        result["f_PL_psf_g_err"] = 0.05
        result["f_host_2500"] = 0.11
        result["f_host_2500_err"] = 0.02
        result["apparent_mag_2500"] = 18.2
        result["apparent_mag_2500_err"] = 0.3
        result["apparent_mag_2500_intrinsic"] = 17.9
        result["apparent_mag_2500_intrinsic_err"] = 0.2

    monkeypatch.setattr(fit_spectra, "compute_derived_results", fake_compute)

    rec = {
        "object_id": "obj-3",
        "sdss_name": "J0001+0001",
        "plate": 1,
        "fiber": 2,
        "mjd": 3,
        "z": 1.1,
        "ra": 10.0,
        "dec": 20.0,
        "loglbol": 46.0,
        "psf_mag_g": 19.0,
        "psf_mag_r": 18.5,
        "psf_mag_err_g": 0.1,
        "psf_mag_err_r": 0.1,
    }
    args = SimpleNamespace(
        output_dir=str(tmp_path / "out"),
        fig_dir=str(tmp_path / "fig"),
        save_fig=False,
        cache_dir=str(tmp_path / "cache"),
        decompose_host=True,
        fit_bc=True,
        fit_lines=True,
        fit_pl=True,
        fit_fe=True,
        fit_poly=False,
        fit_method="optax",
        dsps_ssp_fn="tempdata.h5",
        nuts_warmup=1,
        nuts_samples=1,
        nuts_chains=1,
        nuts_target_accept=0.9,
        optax_steps=1,
        optax_lr=1e-2,
        save_jaxqsofit_samples=False,
        plot_residual=False,
        plot_mcmc_diagnostics=False,
        verbose=False,
        resume=False,
        no_deredden=False,
        wave_min=1250.0,
        wave_max=8000.0,
        fit_bal=False,
    )

    result = fit_spectra.run_one_fit(rec, args)

    out = capsys.readouterr().out
    assert result["fit_ok"] is True
    assert "numpyro_sample_count" not in result
    assert "Spectrum diagnostics for object_id=obj-3" in out
    assert "g: 0.6500 +/- 0.0500" in out
    assert "f_host_2500: 0.1100 +/- 0.0200" in out
    assert "apparent_mag_2500: 18.2000 +/- 0.3000" in out
    assert "Estimated m2500 from model" not in out
    assert diagnostics_calls == []
    assert len(config_calls) == 1
    config = config_calls[0]
    assert config.observation.object_id == "z1.100_J0001+0001"
    assert config.observation.apply_mw_deredden is True
    assert config.spectroscopy.resolving_power == 1950.0
    assert config.spectroscopy.apply_instrumental_resolution is True
    assert config.preprocessing.wave_range == (1250.0, 8000.0)
    assert config.continuum.fit_power_law is True
    assert config.continuum.fit_feii is True
    assert config.continuum.fit_balmer_continuum is True
    assert config.bal.enabled is False
    assert config.continuum.fit_polynomial_tilt is False
    assert config.continuum.fit_reddening is True
    assert config.host.enabled is True
    assert config.host.dsps_ssp_fn == "tempdata.h5"
    assert config.lines.enabled is True
    assert config.inference.method == "optax"
    assert config.inference.map_steps == 1
    assert config.inference.learning_rate == 1e-2
    assert config.output.output_path == str(tmp_path / "out")
    assert config.output.save_name == "z1.100_J0001+0001"
    assert config.output.save_result is False
    assert config.output.plot_fig is False
    assert config.output.save_fig is False
    assert config.psf_photometry.filter_names == ("g", "r")
    assert config.psf_photometry.magnitudes == [19.0, 18.5]
    assert config.psf_photometry.magnitude_errors == [0.1, 0.1]
    assert fit_calls == [{
        "verbose": False,
        "kwargs_plot": {
            "save_fig_path": args.fig_dir,
            "plot_residual": False,
            "show_plot": False,
        },
    }]


def test_run_one_fit_plots_mcmc_diagnostics_when_requested(monkeypatch, tmp_path):
    diagnostics_calls = []

    class DummyHDUL:
        def close(self):
            return None

    class DummyJAXQSOFit:
        def __init__(self, config):
            self.flux = np.asarray(config.spectroscopy.fluxes, dtype=float)
            self.err = np.asarray(config.spectroscopy.errors, dtype=float)
            self.wave = np.array([2000.0, 3000.0, 4000.0], dtype=float)
            self.model_total = np.array([1.0, 1.0, 1.0], dtype=float)
            self.numpyro_samples = {"frac_jitter": np.zeros(1, dtype=float), "add_jitter": np.zeros(1, dtype=float)}
            self.pred_out = {}
            self.ra = config.observation.ra
            self.dec = config.observation.dec
            self.z = config.observation.redshift

        def fit(self, **kwargs):
            return None

        def plot_mcmc_diagnostics(self, **kwargs):
            diagnostics_calls.append(kwargs)

    monkeypatch.setattr(fit_spectra, "load_spec_from_cache", lambda *args, **kwargs: DummyHDUL())
    monkeypatch.setattr(fit_spectra, "get_spectrum_arrays", lambda _hdul: (
        np.array([4000.0, 5000.0, 6000.0], dtype=float),
        np.array([1.0, 1.0, 1.0], dtype=float),
        np.array([0.1, 0.1, 0.1], dtype=float),
        1950.0,
    ))
    monkeypatch.setattr(fit_spectra, "JAXQSOFit", DummyJAXQSOFit)
    monkeypatch.setattr(fit_spectra, "build_default_prior_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(fit_spectra, "build_psf_photometry_inputs", lambda rec: (["g", "r"], [19.0, 18.5], [0.1, 0.1]))
    monkeypatch.setattr(fit_spectra, "extract_named_results", lambda q: {})
    monkeypatch.setattr(fit_spectra, "extract_scalar_attrs", lambda q: {})
    monkeypatch.setattr(fit_spectra, "extract_fit_stats", lambda q: {})
    monkeypatch.setattr(fit_spectra, "compute_derived_results", lambda result, q, args: None)

    rec = {
        "object_id": "obj-4",
        "sdss_name": "J0002+0002",
        "plate": 1,
        "fiber": 2,
        "mjd": 3,
        "z": 1.1,
        "ra": 10.0,
        "dec": 20.0,
        "loglbol": 46.0,
        "psf_mag_g": 19.0,
        "psf_mag_r": 18.5,
        "psf_mag_err_g": 0.1,
        "psf_mag_err_r": 0.1,
    }
    args = SimpleNamespace(
        output_dir=str(tmp_path / "out"),
        fig_dir=str(tmp_path / "fig"),
        save_fig=False,
        cache_dir=str(tmp_path / "cache"),
        decompose_host=True,
        fit_bc=True,
        fit_lines=True,
        fit_pl=True,
        fit_fe=True,
        fit_poly=False,
        fit_method="optax",
        dsps_ssp_fn="tempdata.h5",
        nuts_warmup=1,
        nuts_samples=1,
        nuts_chains=1,
        nuts_target_accept=0.9,
        optax_steps=1,
        optax_lr=1e-2,
        save_jaxqsofit_samples=False,
        plot_residual=False,
        plot_mcmc_diagnostics=True,
        verbose=False,
        resume=False,
        no_deredden=False,
        wave_min=1250.0,
        wave_max=8000.0,
        fit_bal=False,
    )

    result = fit_spectra.run_one_fit(rec, args)

    assert result["fit_ok"] is True
    assert diagnostics_calls == [{"save_fig_path": args.fig_dir}]


def test_run_one_fit_resume_forwards_plot_mcmc_diagnostics(monkeypatch, tmp_path):
    load_calls = []

    class DummyHDUL:
        def close(self):
            return None

    class DummyJAXQSOFit:
        def __init__(self, config=None):
            self.flux = np.array([1.0, 1.0, 1.0], dtype=float)
            self.err = np.array([0.1, 0.1, 0.1], dtype=float)
            self.wave = np.array([2000.0, 3000.0, 4000.0], dtype=float)
            self.model_total = np.array([1.0, 1.0, 1.0], dtype=float)
            self.numpyro_samples = {"frac_jitter": np.zeros(1, dtype=float), "add_jitter": np.zeros(1, dtype=float)}
            self.pred_out = {}
            self.ra = 10.0
            self.dec = 20.0
            self.z = 1.1
            self.use_psf_phot = True

        @classmethod
        def load_from_samples(cls, **kwargs):
            load_calls.append(kwargs)
            return cls()

    monkeypatch.setattr(fit_spectra, "load_spec_from_cache", lambda *args, **kwargs: DummyHDUL())
    monkeypatch.setattr(fit_spectra, "get_spectrum_arrays", lambda _hdul: (
        np.array([4000.0, 5000.0, 6000.0], dtype=float),
        np.array([1.0, 1.0, 1.0], dtype=float),
        np.array([0.1, 0.1, 0.1], dtype=float),
        1950.0,
    ))
    monkeypatch.setattr(fit_spectra, "JAXQSOFit", DummyJAXQSOFit)
    monkeypatch.setattr(fit_spectra, "build_default_prior_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(fit_spectra, "build_psf_photometry_inputs", lambda rec: (["g", "r"], [19.0, 18.5], [0.1, 0.1]))
    monkeypatch.setattr(fit_spectra, "extract_named_results", lambda q: {})
    monkeypatch.setattr(fit_spectra, "extract_scalar_attrs", lambda q: {})
    monkeypatch.setattr(fit_spectra, "extract_fit_stats", lambda q: {})
    monkeypatch.setattr(fit_spectra, "compute_derived_results", lambda result, q, args: None)

    rec = {
        "object_id": "obj-5",
        "sdss_name": "J0003+0003",
        "plate": 1,
        "fiber": 2,
        "mjd": 3,
        "z": 1.1,
        "ra": 10.0,
        "dec": 20.0,
        "loglbol": 46.0,
        "psf_mag_g": 19.0,
        "psf_mag_r": 18.5,
        "psf_mag_err_g": 0.1,
        "psf_mag_err_r": 0.1,
    }
    args = SimpleNamespace(
        output_dir=str(tmp_path / "out"),
        fig_dir=str(tmp_path / "fig"),
        save_fig=False,
        cache_dir=str(tmp_path / "cache"),
        decompose_host=True,
        fit_bc=True,
        fit_lines=True,
        fit_pl=True,
        fit_fe=True,
        fit_poly=False,
        fit_method="optax",
        dsps_ssp_fn="tempdata.h5",
        nuts_warmup=1,
        nuts_samples=1,
        nuts_chains=1,
        nuts_target_accept=0.9,
        optax_steps=1,
        optax_lr=1e-2,
        save_jaxqsofit_samples=False,
        plot_residual=False,
        plot_mcmc_diagnostics=True,
        verbose=False,
        resume=True,
        no_deredden=False,
        wave_min=1250.0,
        wave_max=8000.0,
        fit_bal=False,
    )

    result = fit_spectra.run_one_fit(rec, args)

    assert result["fit_ok"] is True
    assert len(load_calls) == 1
    assert load_calls[0]["kwargs_plot"] == {
        "save_fig_path": args.fig_dir,
        "show_plot": False,
    }
    assert load_calls[0]["plot_diagnostics"] is True
    assert load_calls[0]["diagnostics_kwargs"] == {"save_fig_path": args.fig_dir}


def test_run_one_fit_resume_fails_when_saved_samples_lack_psf_photometry(monkeypatch, tmp_path):
    class DummyHDUL:
        def close(self):
            return None

    class DummyJAXQSOFit:
        def __init__(self):
            self.flux = np.array([1.0, 1.0, 1.0], dtype=float)
            self.err = np.array([0.1, 0.1, 0.1], dtype=float)
            self.wave = np.array([2000.0, 3000.0, 4000.0], dtype=float)
            self.model_total = np.array([1.0, 1.0, 1.0], dtype=float)
            self.numpyro_samples = {
                "frac_jitter": np.zeros(1, dtype=float),
                "add_jitter": np.zeros(1, dtype=float),
            }
            self.pred_out = {}
            self.ra = 10.0
            self.dec = 20.0
            self.z = 1.1
            self.use_psf_phot = False

        @classmethod
        def load_from_samples(cls, **kwargs):
            return cls()

    monkeypatch.setattr(
        fit_spectra, "load_spec_from_cache", lambda *args, **kwargs: DummyHDUL()
    )
    monkeypatch.setattr(
        fit_spectra,
        "get_spectrum_arrays",
        lambda _hdul: (
            np.array([4000.0, 5000.0, 6000.0], dtype=float),
            np.array([1.0, 1.0, 1.0], dtype=float),
            np.array([0.1, 0.1, 0.1], dtype=float),
            1950.0,
        ),
    )
    monkeypatch.setattr(fit_spectra, "JAXQSOFit", DummyJAXQSOFit)
    monkeypatch.setattr(fit_spectra, "build_default_prior_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        fit_spectra,
        "build_psf_photometry_inputs",
        lambda rec: (["g", "r"], [19.0, 18.5], [0.1, 0.1]),
    )

    rec = {
        "object_id": "obj-6",
        "sdss_name": "J0004+0004",
        "plate": 1,
        "fiber": 2,
        "mjd": 3,
        "z": 1.1,
        "ra": 10.0,
        "dec": 20.0,
        "loglbol": 46.0,
        "psf_mag_g": 19.0,
        "psf_mag_r": 18.5,
        "psf_mag_err_g": 0.1,
        "psf_mag_err_r": 0.1,
    }
    args = SimpleNamespace(
        output_dir=str(tmp_path / "out"),
        fig_dir=str(tmp_path / "fig"),
        save_fig=False,
        cache_dir=str(tmp_path / "cache"),
        decompose_host=True,
        fit_bc=True,
        fit_lines=True,
        fit_pl=True,
        fit_fe=True,
        fit_poly=False,
        fit_method="optax",
        dsps_ssp_fn="tempdata.h5",
        nuts_warmup=1,
        nuts_samples=1,
        nuts_chains=1,
        nuts_target_accept=0.9,
        optax_steps=1,
        optax_lr=1e-2,
        save_jaxqsofit_samples=False,
        plot_residual=False,
        plot_mcmc_diagnostics=False,
        verbose=False,
        resume=True,
        no_deredden=False,
        wave_min=1250.0,
        wave_max=8000.0,
        fit_bal=False,
    )

    result = fit_spectra.run_one_fit(rec, args)

    assert result["fit_ok"] is False
    assert "were fit without PSF photometry" in result["error_message"]
    assert "Rerun without --resume" in result["error_message"]
