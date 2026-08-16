"""Fast integration tests for SED-derived PSF dilution in LC fitting."""

import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpyro.handlers import seed, trace

from qvc.light_curve.fit_light_curves import (
    build_single_object_model,
    build_single_object_model_mag_flux_linearized,
    make_lc,
)
from qvc.light_curve.multiband_dho_core import mag_residual_to_relative_flux
from qvc.light_curve.multiband_generate_lc import cut_light_curve_restframe_window
from qvc.light_curve.multiband_model_dho_blr_erlang import (
    make_multiband_dho_blr_flux_linearized_erlang_model,
)
from qvc.light_curve.psf_constant_flux_correction import (
    apply_constant_flux_correction_to_objects,
)
from qvc.spectra import fit_spectra_jaxsedfit_joint as joint


def _joint_record():
    return {
        "object_id": "integration-object",
        "sdss_name": "J0000+0000",
        "plate": 1,
        "fiber": 2,
        "mjd": 3,
        "z": 1.0,
        "ra": 10.0,
        "dec": 20.0,
        "loglbol": 46.0,
        "_joint_photometry": [],
    }


def _light_curve_object():
    bands = list("ugriz")
    times = np.array([0.0, 20.0, 55.0, 100.0])
    mags = {
        band: np.array([20.0, 20.1, 19.9, 20.05]) + 0.1 * index
        for index, band in enumerate(bands)
    }
    magerrs = {band: np.full(times.size, 0.03) for band in bands}
    means = [float(np.mean(mags[band])) for band in bands]
    return {
        "object_id": "integration-object",
        "z": 1.0,
        "times": {band: times.copy() for band in bands},
        "mags": mags,
        "magerrs": magerrs,
        "surveys": {band: np.full(times.size, "sdss") for band in bands},
        "mags_mean": means,
        "mags_mean_err": [0.015] * len(bands),
        "cadence": 20.0,
        "cadence_err": 2.0,
        "number_points": times.size * len(bands),
        "psf_constant_flux_corrected": False,
        "psf_constant_flux_n_bands_corrected": 0,
    }


def test_joint_posterior_csv_reaches_lc_likelihood_without_flux_mutation(
    monkeypatch, tmp_path
):
    filters = [f"{band}_sdss" for band in "ugriz"]
    total = np.ones((5, 5))
    fractions = np.array([0.40, 0.45, 0.50, 0.55, 0.60])
    prediction = {
        "pred_fluxes": total,
        "variable_agn_fluxes": fractions[:, None] * total,
    }
    prediction.update(
        {
            name: np.ones(5, dtype=float)
            for name in joint.JOINT_CHI2_SITES
        }
    )

    class DummyHDUL:
        def close(self):
            pass

    class DummyFitResult:
        samples = {}
        path = None

        def predict(self, *, kind):
            assert kind == "photometry"
            return prediction

    class DummyFitter:
        def __init__(self, config):
            self.config = config

        def fit(self, *, progress_bar):
            return DummyFitResult()

    config = SimpleNamespace(
        galaxy=SimpleNamespace(cosmology_h0=70.0, cosmology_om0=0.3)
    )
    used_phot = pd.DataFrame({"filter_name": filters})
    monkeypatch.setattr(
        joint.legacy, "load_spec_from_cache", lambda *args, **kwargs: DummyHDUL()
    )
    monkeypatch.setattr(
        joint.legacy,
        "get_spectrum_arrays",
        lambda hdul: (np.array([4000.0]), np.array([1.0]), np.array([0.1]), 2000.0),
    )
    monkeypatch.setattr(
        joint, "build_joint_config", lambda *args, **kwargs: (config, used_phot)
    )
    monkeypatch.setattr(joint, "summarize_m2500_dereddened", lambda *args, **kwargs: {})
    monkeypatch.setitem(sys.modules, "jaxsedfit", SimpleNamespace(JAXSEDFit=DummyFitter))
    args = SimpleNamespace(
        cache_dir="cache",
        progress=False,
        save_fig=False,
        save_jaxsedfit_samples=False,
        verbose=False,
        fig_dir="figs",
    )

    sed_row = joint.run_one_fit(_joint_record(), args)
    assert sed_row["fit_ok"] is True
    sed_csv = tmp_path / "joint.csv"
    pd.DataFrame([sed_row]).to_csv(sed_csv, index=False)

    raw = _light_curve_object()
    original_mags = {band: values.copy() for band, values in raw["mags"].items()}
    attached, summary = apply_constant_flux_correction_to_objects(
        [raw], spectra_fit_csvs=[sed_csv]
    )
    attached = attached[0]
    assert summary["n_bands_corrected"] == 5
    for band in "ugriz":
        np.testing.assert_array_equal(attached["mags"][band], original_mags[band])
        assert attached[f"f_AGN_psf_{band}"] == pytest.approx(0.5)
        assert attached[f"f_AGN_psf_{band}_err"] > 0.0

    attached = cut_light_curve_restframe_window(
        [attached], n_days=30.0, same_length=False
    )[0]
    lc = make_lc(
        attached,
        bands=list("ugriz"),
        drop_band_lyman_alpha=False,
    )
    assert lc["bands"] == list("ugri")
    np.testing.assert_allclose(lc["mags_means"], raw["mags_mean"][:4])
    np.testing.assert_allclose(lc["agn_fraction_by_band"], 0.5)
    assert np.all(lc["agn_fraction_err_by_band"] > 0.0)

    model = build_single_object_model_mag_flux_linearized(
        lc,
        jnp.array([1800.0, 2300.0, 3100.0, 3800.0]),
        log_jitter_mean=np.full((4, 3), np.log(0.03)),
    )
    sites = trace(seed(model, jax.random.PRNGKey(0))).get_trace()
    assert "_agn_fraction_uncertain" not in sites
    assert sites["agn_fraction_by_band"]["type"] == "deterministic"
    np.testing.assert_allclose(
        np.asarray(sites["agn_fraction_by_band"]["value"]),
        lc["agn_fraction_by_band"],
    )
    loglike = sites["loglike"]
    assert np.isfinite(float(loglike["fn"].log_prob(loglike["value"])))

    subtracted, subtraction_summary = apply_constant_flux_correction_to_objects(
        [raw],
        spectra_fit_csvs=[sed_csv],
        subtract_observations=True,
    )
    subtracted = subtracted[0]
    assert subtraction_summary["correction_mode"] == "subtracted"
    assert subtracted["psf_constant_flux_mode"] == "subtracted"
    assert any(
        not np.array_equal(subtracted["mags"][band], original_mags[band])
        for band in "ugriz"
    )

    mag_lc = make_lc(
        subtracted,
        bands=list("ugriz"),
        drop_band_lyman_alpha=False,
    )
    mag_model = build_single_object_model(
        mag_lc,
        jnp.array([1800.0, 2300.0, 3100.0, 3800.0]),
        log_jitter_mean=np.full((4, 3), np.log(0.06)),
    )
    mag_sites = trace(seed(mag_model, jax.random.PRNGKey(1))).get_trace()
    mag_loglike = mag_sites["loglike"]
    assert np.isfinite(
        float(mag_loglike["fn"].log_prob(mag_loglike["value"]))
    )


def test_diluted_likelihood_is_exactly_constant_flux_subtraction_in_relflux():
    reference_mag = 20.0
    mags = np.array([20.10, 19.90, 20.05, 19.95])
    fraction = 0.4
    reference_total_flux = 10.0 ** (-0.4 * reference_mag)
    total_flux = 10.0 ** (-0.4 * mags)
    constant_flux = (1.0 - fraction) * reference_total_flux
    agn_reference_flux = fraction * reference_total_flux
    agn_flux = total_flux - constant_flux

    observed_relflux = np.asarray(
        mag_residual_to_relative_flux(mags - reference_mag)
    )
    subtracted_agn_relflux = (agn_flux - agn_reference_flux) / agn_reference_flux
    np.testing.assert_allclose(
        observed_relflux,
        fraction * subtracted_agn_relflux,
        rtol=2e-14,
        atol=2e-14,
    )

    times = jnp.array([0.0, 10.0, 25.0, 50.0])
    bands = jnp.zeros(4, dtype=jnp.int32)
    y = jnp.asarray(observed_relflux)
    yerr = jnp.full(4, 0.02)
    model = make_multiband_dho_blr_flux_linearized_erlang_model(
        (times, bands), y, yerr, n_band=1, survey_idx=jnp.zeros(4, dtype=jnp.int32)
    )
    intrinsic = {
        "tau_fast_band": jnp.array([20.0]),
        "tau_slow_band": jnp.array([150.0]),
        "lag_blr": jnp.array([60.0]),
        "amp_cont_relflux": jnp.array([0.12]),
        "amp_blr_relflux": jnp.array([0.03]),
        "agn_fraction_by_band": jnp.array([fraction]),
        "mean": jnp.array([0.02]),
        "linear_trend": jnp.asarray(0.01),
        "linear_trend_band_offset": jnp.zeros(1),
        "log_jitter": jnp.full((1, 3), -10.0),
        "survey_delta_mag": jnp.zeros((1, 3)),
    }
    manually_diluted = dict(intrinsic)
    manually_diluted.pop("agn_fraction_by_band")
    manually_diluted["amp_cont_relflux"] *= fraction
    manually_diluted["amp_blr_relflux"] *= fraction
    manually_diluted["mean"] *= fraction
    manually_diluted["linear_trend"] *= fraction
    manually_diluted["linear_trend_band_offset"] *= fraction

    assert float(model.log_prob(intrinsic)) == pytest.approx(
        float(model.log_prob(manually_diluted)), rel=2e-12, abs=2e-12
    )
