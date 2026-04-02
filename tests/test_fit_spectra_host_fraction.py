from types import SimpleNamespace

import numpy as np

from qvc.spectra import fit_spectra


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


def test_estimate_host_2500_fraction_reconstructs_outside_native_range_without_poly(monkeypatch):
    def fake_reconstruct(**kwargs):
        assert kwargs["fit_poly"] is False
        return {
            "wave": np.asarray(kwargs["wave_out"], dtype=float),
            "draws": {
                "host": np.array(
                    [
                        [0.20, 0.20, 0.20],
                        [0.25, 0.25, 0.25],
                        [0.30, 0.30, 0.30],
                    ],
                    dtype=float,
                ),
                "continuum": np.ones((3, 3), dtype=float),
            },
        }

    monkeypatch.setattr(fit_spectra, "reconstruct_posterior_components", fake_reconstruct)

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

    assert np.isclose(median, 0.25)
    assert np.isclose(err, 0.034)


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

    na_int = np.trapz(q.pred_out["line_model_narrow"], q.wave, axis=1)
    cont_int = np.trapz(q.pred_out["continuum_model"], q.wave, axis=1)
    ratio_draws = na_int / cont_int
    p16, p50, p84 = np.percentile(ratio_draws, [16, 50, 84])

    assert np.isclose(result["f_na"], p50)
    assert np.isclose(result["f_na_err"], 0.5 * (p84 - p16))


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

    na_int = np.trapz(q.pred_out["line_model_narrow"], q.wave, axis=1)
    cont_int = np.trapz(q.pred_out["continuum_model"], q.wave, axis=1)
    ratio_draws = na_int / cont_int
    p16, p50, p84 = np.percentile(ratio_draws, [16, 50, 84])

    assert result["f_host_2500"] == 0.0
    assert result["f_host_center"] == 0.0
    assert np.isclose(result["f_na"], p50)
    assert np.isclose(result["f_na_err"], 0.5 * (p84 - p16))
