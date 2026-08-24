import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from qvc.spectra import fit_spectra_jaxsedfit_joint as joint
from qvc.spectra.fit_spectra_jaxsedfit_joint import (
    ab_mag_to_mjy,
    add_qvc_psf_photometry,
    empty_psf_agn_fraction_summary,
    estimate_m2500_dereddened,
    extract_compact_psf_agn_fraction_draws,
    load_saved_sed_photometry,
    predict_catalog_posterior,
    save_spectrum_figure,
    summarize_catalog_posterior,
    summarize_joint_chi2,
    summarize_psf_agn_fractions,
    verify_new_posterior_bundle,
    write_joint_fit_results_hdf5,
)


def _record_with_ugriz():
    record = {"object_id": "1452887"}
    for band in "ugriz":
        record[f"psf_mag_{band}"] = 20.0
        record[f"psf_mag_err_{band}"] = 0.1
    return record


def test_saved_sed_loader_normalizes_object_id_and_upper_limits(tmp_path):
    path = tmp_path / "sed.csv"
    pd.DataFrame(
        {
            "object_id": [1452887],
            "filter_name": ["W3"],
            "flux_mjy": [0.5],
            "flux_err_mjy": [0.1],
            "is_upper_limit": ["true"],
        }
    ).to_csv(path, index=False)

    result = load_saved_sed_photometry(path)

    assert result["source_id"].tolist() == ["1452887"]
    assert result["is_upper_limit"].tolist() == [True]


def test_qvc_psf_photometry_replaces_saved_sdss_and_includes_z():
    saved = pd.DataFrame(
        {
            "filter_name": ["g_sdss", "W2"],
            "flux_mjy": [99.0, 0.2],
            "flux_err_mjy": [1.0, 0.02],
            "is_upper_limit": [False, False],
        }
    )

    result = add_qvc_psf_photometry(_record_with_ugriz(), saved)

    assert set(result["filter_name"]) == {
        "u_sdss", "g_sdss", "r_sdss", "i_sdss", "z_sdss", "W2"
    }
    g_flux = result.loc[result["filter_name"] == "g_sdss", "flux_mjy"].item()
    assert np.isclose(g_flux, ab_mag_to_mjy(20.0))
    qvc_rows = result["filter_name"].isin(
        [f"{band}_sdss" for band in "ugriz"]
    )
    assert set(result.loc[qvc_rows, "host_capture_group"]) == {
        joint.QVC_PSF_HOST_CAPTURE_GROUP
    }
    assert pd.isna(
        result.loc[result["filter_name"] == "W2", "host_capture_group"].item()
    )


def test_qvc_psf_photometry_requires_z_even_if_variability_fit_dropped_it():
    record = _record_with_ugriz()
    record["psf_mag_z"] = np.nan

    with pytest.raises(ValueError, match=r"missing/invalid: \['z'\]"):
        add_qvc_psf_photometry(record, pd.DataFrame())


def test_dereddened_m2500_uses_intrinsic_disk_and_both_attenuation_terms():
    samples = {
        "log_agn_amp": np.log(np.array([1.0e38, 1.1e38])),
        "pl_slope": np.array([-1.8, -1.8]),
        "pl_bend_loc": np.array([1000.0, 1000.0]),
        "pl_bend_width": np.array([10.0, 10.0]),
        "ebv_gal": np.array([0.02, 0.02]),
        "ebv_agn": np.array([0.03, 0.03]),
    }

    result = estimate_m2500_dereddened(samples, redshift=1.0)

    intrinsic = result["m_2500_dereddened_draws"]
    attenuated = result["m_2500_attenuated_model_draws"]
    a_gal = result["a_2500_galaxy_draws"]
    a_internal = result["a_2500_internal_draws"]
    assert np.all(np.isfinite(intrinsic))
    assert np.allclose(attenuated - intrinsic, a_gal + a_internal)
    assert np.all(a_gal > 0)
    assert np.all(a_internal > 0)


def test_m2500_resume_regenerates_dust_and_overrides_stale_values():
    latent = {
        "log_agn_amp": np.log(np.array([1.0e38, 1.1e38])),
        "pl_slope": np.array([-1.8, -1.8]),
    }
    regenerated = {
        "ebv_gal": np.array([0.02, 0.03]),
        "ebv_agn": np.array([0.04, 0.05]),
    }
    stale = {
        **latent,
        "ebv_gal": np.zeros(2),
        "ebv_agn": np.zeros(2),
    }

    resumed = joint.posterior_samples_for_m2500(stale, regenerated)
    fresh = {**latent, **regenerated}

    np.testing.assert_array_equal(resumed["ebv_gal"], regenerated["ebv_gal"])
    np.testing.assert_array_equal(resumed["ebv_agn"], regenerated["ebv_agn"])
    resumed_draws = estimate_m2500_dereddened(resumed, redshift=1.0)
    fresh_draws = estimate_m2500_dereddened(fresh, redshift=1.0)
    assert set(resumed_draws) == set(fresh_draws)
    for name in resumed_draws:
        np.testing.assert_allclose(resumed_draws[name], fresh_draws[name])


def test_m2500_resume_rejects_missing_regenerated_dust_site():
    latent = {
        "log_agn_amp": np.log(np.array([1.0e38, 1.1e38])),
        "pl_slope": np.array([-1.8, -1.8]),
    }

    with pytest.raises(joint.M2500ReconstructionError, match="ebv_agn"):
        joint.posterior_samples_for_m2500(
            latent,
            {"ebv_gal": np.array([0.02, 0.03])},
        )


def test_m2500_rejects_all_zero_attenuation_draws():
    samples = {
        "log_agn_amp": np.log(np.array([1.0e38, 1.1e38])),
        "pl_slope": np.array([-1.8, -1.8]),
        "ebv_gal": np.zeros(2),
        "ebv_agn": np.zeros(2),
    }

    with pytest.raises(
        joint.M2500ReconstructionError,
        match="zero attenuation for every draw",
    ):
        estimate_m2500_dereddened(samples, redshift=1.0)


def test_total_a2500_summary_is_computed_from_joint_posterior_draws():
    samples = {
        "log_agn_amp": np.log(np.full(3, 1.0e38)),
        "pl_slope": np.full(3, -1.8),
        "pl_bend_loc": np.full(3, 1000.0),
        "pl_bend_width": np.full(3, 10.0),
        # Anticorrelated components make the median of the draw-wise sum
        # differ from the sum of the component medians.
        "ebv_gal": np.array([0.01, 1.00, 1.01]),
        "ebv_agn": np.array([1.00, 0.01, 1.00]),
    }

    draws = estimate_m2500_dereddened(samples, redshift=1.0)
    summary = joint.summarize_m2500_dereddened(samples, redshift=1.0)
    expected_draws = (
        draws["a_2500_galaxy_draws"]
        + draws["a_2500_internal_draws"]
    )
    expected = joint.legacy.sym_percentile(expected_draws)

    np.testing.assert_allclose(draws["a_2500_total_draws"], expected_draws)
    assert summary["a_2500_total"] == pytest.approx(expected[0])
    assert summary["a_2500_total_err"] == pytest.approx(expected[1])
    assert summary["a_2500_total_err_lower"] == pytest.approx(expected[2])
    assert summary["a_2500_total_err_upper"] == pytest.approx(expected[3])


def test_m2500_convergence_uses_one_summary_for_print_and_fields(monkeypatch, capsys):
    grouped = {
        "log_agn_amp": np.log(np.full((2, 4), 1.0e38)),
        "pl_slope": np.full((2, 4), -1.8),
        "pl_bend_loc": np.full((2, 4), 1000.0),
        "pl_bend_width": np.full((2, 4), 10.0),
        "ebv_gal": np.full((2, 4), 0.02),
        "ebv_agn": np.full((2, 4), 0.03),
    }
    calls = []

    def fake_summary(samples, *, group_by_chain, prob):
        calls.append(samples)
        assert group_by_chain is True
        assert prob == pytest.approx(0.90)
        assert all(value.shape == (2, 4) for value in samples.values())
        return {
            name: {
                "mean": np.array(20.0),
                "std": np.array(0.1),
                "median": np.array(20.0),
                "5.0%": np.array(19.8),
                "95.0%": np.array(20.2),
                "n_eff": np.array(80.0 + index),
                "r_hat": np.array(1.01 + 0.01 * index),
            }
            for index, name in enumerate(joint.HUBBLE_MAGNITUDE_SITES)
        }

    monkeypatch.setattr(joint, "compute_numpyro_summary", fake_summary)

    result = joint.summarize_m2500_convergence(
        grouped,
        redshift=1.0,
        heading="m2500 posterior",
    )

    assert len(calls) == 1
    assert result == {
        "m_2500_dereddened_rhat": pytest.approx(1.01),
        "m_2500_attenuated_model_rhat": pytest.approx(1.02),
    }
    assert "m2500 posterior" in capsys.readouterr().out


def test_spectral_convergence_saves_all_scalar_sites_and_skips_arrays(monkeypatch):
    grouped = {
        "log_agn_amp": np.arange(8.0).reshape(2, 4),
        "pl_slope": np.arange(8.0).reshape(2, 4),
        "log_ebv_gal": np.arange(8.0).reshape(2, 4),
        "ebv_gal": np.arange(8.0).reshape(2, 4),
        "log_ebv_agn": np.arange(8.0).reshape(2, 4),
        "ebv_agn": np.arange(8.0).reshape(2, 4),
        "singleton_site": np.arange(8.0).reshape(2, 4, 1),
        "vector_site": np.ones((2, 4, 3)),
    }
    captured = {}

    def fake_summary(samples, *, group_by_chain, prob):
        captured.update(samples)
        assert group_by_chain is True
        assert prob == pytest.approx(0.90)
        return {
            name: {
                "n_eff": np.array(50.0 + index),
                "r_hat": np.array(1.0 + 0.01 * index),
            }
            for index, name in enumerate(samples)
        }

    monkeypatch.setattr(joint, "compute_numpyro_summary", fake_summary)
    monkeypatch.setattr(
        joint, "print_numpyro_summary_dict", lambda *args, **kwargs: None
    )

    result = joint.summarize_spectral_convergence(grouped, redshift=1.0)

    expected_scalar_sites = {
        "log_agn_amp",
        "pl_slope",
        "log_ebv_gal",
        "ebv_gal",
        "log_ebv_agn",
        "ebv_agn",
        "singleton_site",
        *joint.HUBBLE_MAGNITUDE_SITES,
        "a_2500_total",
    }
    assert set(captured) == expected_scalar_sites
    assert "vector_site" not in captured
    for name in expected_scalar_sites:
        assert np.isfinite(result[f"{name}_rhat"])
        assert f"{name}_ess" not in result


def test_flat_samples_reconstruct_chain_major_order():
    samples = {
        "a": np.arange(12.0),
        "b": np.arange(24.0).reshape(12, 2),
    }

    grouped = joint._reshape_flat_samples_by_chain(samples, 3)

    assert grouped["a"].shape == (3, 4)
    assert grouped["b"].shape == (3, 4, 2)
    np.testing.assert_array_equal(grouped["a"].reshape(-1), samples["a"])
    np.testing.assert_array_equal(grouped["b"].reshape(12, 2), samples["b"])


def test_fresh_fit_preserves_grouped_scientific_nuts_samples():
    class DummyMCMC:
        def get_samples(self, *, group_by_chain):
            assert group_by_chain is True
            return {
                "physical": np.arange(8.0).reshape(2, 4),
                "internal_aux": np.ones((2, 4)),
            }

    fit_result = SimpleNamespace(
        samples={
            "physical": np.arange(8.0),
            "deterministic": np.arange(8.0) + 10.0,
        },
        fitter=SimpleNamespace(nuts_result={"mcmc": DummyMCMC()}),
    )

    grouped = joint._fresh_grouped_nuts_samples(fit_result)

    assert set(grouped) == {"physical", "deterministic"}
    assert grouped["physical"].shape == (2, 4)
    np.testing.assert_array_equal(
        grouped["deterministic"],
        fit_result.samples["deterministic"].reshape(2, 4),
    )


def test_flat_samples_reject_unreconstructable_chain_shape():
    with pytest.raises(ValueError, match="Cannot reconstruct"):
        joint._reshape_flat_samples_by_chain({"a": np.arange(10.0)}, 3)


def test_base_result_has_stable_nan_hubble_convergence_schema(tmp_path):
    result = joint._base_result(
        _run_record(),
        _hybrid_args(tmp_path),
        execution_mode="fresh",
    )

    expected = joint.empty_hubble_convergence_summary()
    assert set(expected) <= set(result)
    assert all(np.isnan(result[name]) for name in expected)
    assert np.isnan(result["f_host_2500_psf"])
    assert np.isnan(result["f_host_2500_psf_err"])


def test_save_spectrum_figure_uses_separate_spectrum_filename(tmp_path):
    class FakeFitter:
        def __init__(self):
            self.show_plot = None
            self.plot_residual = None

        def plot_jaxqsofit_spectrum(self, *, show_plot, plot_residual):
            self.show_plot = show_plot
            self.plot_residual = plot_residual
            return plt.figure()

    fitter = FakeFitter()
    path = save_spectrum_figure(
        fitter,
        {"z": 0.300041, "sdss_name": "205105.02-003302.7"},
        tmp_path,
    )

    assert fitter.show_plot is False
    assert fitter.plot_residual is False
    assert path == tmp_path / "z0.300_205105.02-003302.7_spectrum.png"
    assert path.is_file()


def _run_record():
    return {
        "object_id": "1452887",
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


def _component_prediction(filter_names):
    total = np.tile(np.arange(1.0, len(filter_names) + 1.0), (4, 1))
    prediction = {
        "pred_fluxes": total,
        "variable_agn_fluxes": 0.7 * total,
        "fracAGN_5100_fit": np.array([0.6, 0.8]),
        "formed_stellar_mass": np.array([1.0e10, 1.2e10]),
        "component_host_fraction": np.array([[0.2], [0.3], [0.4], [0.5]]),
    }
    prediction.update(
        {
            name: np.array([float(index), float(index + 2)])
            for index, name in enumerate(joint.JOINT_CHI2_SITES)
        }
    )
    return prediction


def test_compact_psf_fraction_draws_preserve_joint_rows_and_pad_to_64():
    filter_names = [f"{band}_sdss" for band in "ugriz"]
    total = np.ones((4, 5), dtype=float)
    fractions = np.arange(20, dtype=float).reshape(4, 5) / 100.0 + 0.5
    prediction = {
        "pred_fluxes": total,
        "variable_agn_fluxes": fractions,
    }

    compact, valid_count = extract_compact_psf_agn_fraction_draws(
        prediction,
        filter_names,
        object_id="1452887",
        seed=3,
    )

    assert compact.shape == (64, 5)
    assert compact.dtype == np.float32
    assert valid_count == 4
    np.testing.assert_allclose(compact[:4], fractions.astype(np.float32))
    assert np.all(np.isnan(compact[4:]))


def test_joint_fit_result_writer_moves_private_draw_payload_out_of_catalog(tmp_path):
    path = tmp_path / "chunk.h5"
    draws = np.full((64, 5), np.nan, dtype=np.float32)
    draws[:2] = 0.75
    rows = [
        {
            "object_id": "1452887",
            "fit_ok": True,
            "fit_backend": "jaxsedfit_joint",
            "fracAGN_5100_fit": 0.65,
            "fracAGN_5100_fit_err": 0.04,
            "formed_stellar_mass": 1.1e10,
            "f_AGN_psf_g": 0.75,
            "_psf_agn_fraction_draws": draws,
            "_psf_agn_fraction_valid_count": 2,
        }
    ]

    write_joint_fit_results_hdf5(path, rows)

    with h5py.File(path, "r") as handle:
        assert "_psf_agn_fraction_draws" not in handle["catalog"]
        assert handle["catalog/fracAGN_5100_fit"][0] == pytest.approx(0.65)
        assert handle["catalog/fracAGN_5100_fit_err"][0] == pytest.approx(0.04)
        assert handle["catalog/formed_stellar_mass"][0] == pytest.approx(1.1e10)
        assert handle["psf_agn_fraction_draws/values"].shape == (1, 64, 5)
        assert handle["psf_agn_fraction_draws/valid_count"][0] == 2


def test_m2500_catalog_validation_rejects_zeroed_resumed_attenuation():
    row = {
        "object_id": "1452887",
        "fit_ok": True,
        "execution_mode": "resumed",
        "ebv_gal": 0.02,
        "ebv_agn": 0.03,
        "m_2500_dereddened": 20.0,
        "m_2500_attenuated_model": 20.0,
        "a_2500_galaxy": 0.0,
        "a_2500_internal": 0.0,
        "a_2500_total": 0.0,
    }

    with pytest.raises(
        joint.M2500ReconstructionError,
        match="positive EBV posteriors but zero reconstructed A_2500",
    ):
        joint.validate_m2500_catalog_rows([row])


@pytest.mark.parametrize(
    ("a_2500_total", "m_2500_attenuated_model"),
    [(0.0, 20.25), (0.25, 20.0)],
)
def test_m2500_catalog_validation_rejects_partially_zeroed_resume_fields(
    a_2500_total,
    m_2500_attenuated_model,
):
    row = {
        "object_id": "1452887",
        "fit_ok": True,
        "execution_mode": "resumed",
        "ebv_gal": 0.02,
        "ebv_agn": 0.03,
        "m_2500_dereddened": 20.0,
        "m_2500_attenuated_model": m_2500_attenuated_model,
        "a_2500_galaxy": 0.10,
        "a_2500_internal": 0.15,
        "a_2500_total": a_2500_total,
    }

    with pytest.raises(joint.M2500ReconstructionError, match="object_ids"):
        joint.validate_m2500_catalog_rows([row])


def test_m2500_catalog_validation_uses_latent_log_ebv_as_dust_evidence():
    row = {
        "object_id": "1452887",
        "fit_ok": True,
        "execution_mode": "resumed",
        "ebv_gal": 0.0,
        "ebv_agn": 0.0,
        "log_ebv_gal": np.log(0.02),
        "log_ebv_agn": np.log(0.03),
        "m_2500_dereddened": 20.0,
        "m_2500_attenuated_model": 20.0,
        "a_2500_galaxy": 0.0,
        "a_2500_internal": 0.0,
        "a_2500_total": 0.0,
    }

    with pytest.raises(joint.M2500ReconstructionError, match="object_ids"):
        joint.validate_m2500_catalog_rows([row])


def test_m2500_catalog_validation_accepts_physical_rows_and_ignores_failures():
    valid = {
        "object_id": "1452887",
        "fit_ok": True,
        "execution_mode": "resumed",
        "ebv_gal": 0.02,
        "ebv_agn": 0.03,
        "m_2500_dereddened": 20.0,
        "m_2500_attenuated_model": 20.25,
        "a_2500_galaxy": 0.10,
        "a_2500_internal": 0.15,
        "a_2500_total": 0.25,
    }
    failed = {
        "object_id": "failed",
        "fit_ok": False,
        "execution_mode": "resumed",
    }

    joint.validate_m2500_catalog_rows([valid, failed])


def test_m2500_catalog_validation_requires_same_success_schema():
    incomplete = {
        "object_id": "1452887",
        "fit_ok": True,
        "execution_mode": "fresh",
    }

    with pytest.raises(
        joint.M2500ReconstructionError,
        match="lacks required m2500 catalog fields",
    ):
        joint.validate_m2500_catalog_rows([incomplete])


def _hybrid_args(tmp_path):
    return SimpleNamespace(
        resume=str(tmp_path / "old"),
        resume_run_name="old_run",
        output_dir=str(tmp_path / "new"),
        fig_dir=str(tmp_path / "figs"),
        save_fig=True,
        save_jaxsedfit_samples=True,
        seed=3,
        verbose=False,
    )


def test_joint_summaries_have_stable_native_schema():
    filter_names = [f"{band}_sdss" for band in "ugriz"]
    prediction = _component_prediction(filter_names)

    fractions = summarize_psf_agn_fractions(prediction, filter_names)
    host_2500 = joint.summarize_host_2500_psf(prediction)
    chi2 = summarize_joint_chi2(prediction)

    assert set(empty_psf_agn_fraction_summary()) <= set(fractions)
    for band in "ugriz":
        assert fractions[f"f_AGN_psf_{band}"] == pytest.approx(0.7)
        assert fractions[f"f_AGN_psf_{band}_err"] == pytest.approx(0.0)
    for name in joint.JOINT_CHI2_SITES:
        assert name in chi2
        assert f"{name}_err" in chi2
    assert host_2500["f_host_2500_psf"] == pytest.approx(0.35)
    assert host_2500["f_host_2500_psf_err"] == pytest.approx(0.102)


def test_catalog_summary_combines_latent_and_deterministic_scalar_sites():
    samples = {
        "pl_slope": np.array([-1.9, -1.7]),
        "latent_vector": np.ones((2, 3)),
    }
    prediction = {
        "fracAGN_5100_fit": np.array([0.6, 0.8]),
        "formed_stellar_mass": np.array([1.0e10, 1.2e10]),
        "pred_fluxes": np.ones((2, 5)),
    }

    result = summarize_catalog_posterior(samples, prediction)

    assert result["pl_slope"] == pytest.approx(-1.8)
    assert result["fracAGN_5100_fit"] == pytest.approx(0.7)
    assert result["formed_stellar_mass"] == pytest.approx(1.1e10)
    assert "pl_slope_err" in result
    assert "fracAGN_5100_fit_err" in result
    assert "formed_stellar_mass_err" in result
    assert "latent_vector" not in result
    assert "pred_fluxes" not in result


def test_catalog_prediction_temporarily_requests_legacy_csv_scalar_sites():
    class DummyFitter:
        @staticmethod
        def _predictive_return_sites(kind, **kwargs):
            assert kind == "photometry"
            return ["pred_fluxes", "fracAGN_5100_fit"]

        def predict(self, *, kind, **kwargs):
            self.requested_sites = self._predictive_return_sites(kind)
            self.prediction_kwargs = kwargs
            return {"fracAGN_5100_fit": np.array([0.6, 0.8])}

    fitter = DummyFitter()

    prediction = predict_catalog_posterior(fitter, kind="photometry")

    assert "fracAGN_5100_fit" in prediction
    assert set(joint.LEGACY_CSV_SCALAR_PREDICTION_SITES) <= set(
        fitter.requested_sites
    )
    assert fitter._predictive_return_sites("photometry") == [
        "pred_fluxes",
        "fracAGN_5100_fit",
    ]


def test_catalog_prediction_forwards_monochromatic_component_request():
    class DummyFitter:
        @staticmethod
        def _predictive_return_sites(kind, **kwargs):
            sites = ["fracAGN_5100_fit"]
            if kwargs.get("include_component_request"):
                sites.append("component_host_fraction")
            return sites

        def predict(self, **kwargs):
            self.kwargs = kwargs
            self.requested_sites = self._predictive_return_sites(
                kwargs["kind"], include_component_request=True
            )
            return {"component_host_fraction": np.array([[0.25]])}

    fitter = DummyFitter()
    prediction = predict_catalog_posterior(
        fitter,
        kind="photometry",
        component_rest_wavelengths=(2500.0,),
        component_host_capture_group=joint.QVC_PSF_HOST_CAPTURE_GROUP,
    )
    assert prediction["component_host_fraction"][0, 0] == pytest.approx(0.25)
    assert fitter.kwargs["component_rest_wavelengths"] == (2500.0,)
    assert "component_host_fraction" in fitter.requested_sites


def test_verify_new_posterior_bundle_requires_v2_schema(tmp_path):
    current = tmp_path / "current.h5"
    with h5py.File(current, "w") as handle:
        handle.attrs["posterior_bundle_format"] = joint.POSTERIOR_BUNDLE_FORMAT
    assert verify_new_posterior_bundle(current) == current

    old = tmp_path / "old.h5"
    with h5py.File(old, "w") as handle:
        handle.attrs["posterior_bundle_format"] = "jaxsedfit_samples_meta_v1"
    with pytest.raises(ValueError, match="expected 'jaxsedfit_samples_meta_v2'"):
        verify_new_posterior_bundle(old)


def test_resume_preflight_rejects_old_and_accepts_grouped_bundle(tmp_path):
    args = _hybrid_args(tmp_path)
    rec = _run_record()
    path = joint.posterior_bundle_path(args.resume, rec)
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("samples/log_agn_amp", data=np.ones(2))

    with pytest.raises(
        joint.IncompatibleHostCaptureResumeError,
        match="predates the shared qvc_sdss_psf",
    ):
        joint.preflight_resume_host_capture_bundles([rec], args)

    with h5py.File(path, "a") as handle:
        handle.attrs[joint.HOST_CAPTURE_BUNDLE_ATTR] = (
            joint.QVC_PSF_HOST_CAPTURE_GROUP
        )
        handle.create_dataset(
            "samples/host_capture_group_fraction", data=np.ones((2, 1))
        )
    joint.preflight_resume_host_capture_bundles([rec], args)


def test_parse_args_resume_keeps_current_inputs_and_separate_destinations(tmp_path):
    resume_dir = tmp_path / "old_run" / "all"
    resume_dir.mkdir(parents=True)

    args = joint.parse_args(
        [
            "--mode",
            "fit",
            str(tmp_path / "new" / "chunk.h5"),
            "--sed-photometry-path",
            str(tmp_path / "current_photometry.csv"),
            "--filter_object_id",
            "1452887",
            "--resume",
            str(resume_dir),
            "--output-dir",
            str(tmp_path / "new" / "all"),
            "--fig-dir",
            str(tmp_path / "new_plots"),
        ]
    )

    assert args.filter_object_id == ["1452887"]
    assert args.sed_photometry_path.endswith("current_photometry.csv")
    assert args.resume == str(resume_dir)
    assert args.resume_run_name == "old_run"
    assert args.output_dir != args.resume


def test_parse_args_rejects_resume_output_directory_alias(tmp_path):
    resume_dir = tmp_path / "same" / "all"
    resume_dir.mkdir(parents=True)

    with pytest.raises(SystemExit):
        joint.parse_args(
            [
                "--mode",
                "fit",
                str(tmp_path / "chunk.csv"),
                "--sed-photometry-path",
                str(tmp_path / "photometry.csv"),
                "--filter_object_id",
                "1452887",
                "--resume",
                str(resume_dir),
                "--output-dir",
                str(resume_dir),
            ]
        )


def test_hybrid_missing_bundle_requires_fresh_spectral_run(monkeypatch, tmp_path):
    args = _hybrid_args(tmp_path)
    with pytest.raises(
        joint.IncompatibleHostCaptureResumeError,
        match="Run fresh spectral inference",
    ):
        joint.run_hybrid_fit(_run_record(), args)


def test_hybrid_bad_bundle_cleans_new_artifacts_and_falls_back(monkeypatch, tmp_path):
    args = _hybrid_args(tmp_path)
    source = joint.posterior_bundle_path(args.resume, _run_record())
    source.parent.mkdir(parents=True)
    source.write_bytes(b"old posterior")

    def failing_resume(rec, received_args, source_path):
        new_bundle = joint.posterior_bundle_path(received_args.output_dir, rec)
        new_bundle.parent.mkdir(parents=True)
        new_bundle.write_bytes(b"partial")
        sed_path = joint.sed_figure_path(received_args.fig_dir, rec)
        sed_path.parent.mkdir(parents=True)
        sed_path.write_bytes(b"partial")
        raise ValueError("incompatible posterior")

    fresh_calls = []

    def fake_fresh(rec, received_args, **kwargs):
        fresh_calls.append(kwargs)
        return {"fit_ok": True, **kwargs}

    monkeypatch.setattr(joint, "_run_resumed_fit", failing_resume)
    monkeypatch.setattr(joint, "run_one_fit", fake_fresh)

    result = joint.run_hybrid_fit(_run_record(), args)

    assert result["execution_mode"] == "fresh_resume_failed"
    assert "incompatible posterior" in result["resume_error_message"]
    assert not joint.posterior_bundle_path(args.output_dir, _run_record()).exists()
    assert not joint.sed_figure_path(args.fig_dir, _run_record()).exists()
    assert source.read_bytes() == b"old posterior"
    assert len(fresh_calls) == 1


def test_hybrid_m2500_reconstruction_error_does_not_refit(monkeypatch, tmp_path):
    args = _hybrid_args(tmp_path)
    source = joint.posterior_bundle_path(args.resume, _run_record())
    source.parent.mkdir(parents=True)
    source.write_bytes(b"posterior")

    def fail_resume(*args, **kwargs):
        raise joint.M2500ReconstructionError("missing regenerated ebv_agn")

    monkeypatch.setattr(joint, "_run_resumed_fit", fail_resume)
    monkeypatch.setattr(
        joint,
        "run_one_fit",
        lambda *args, **kwargs: pytest.fail("must not launch a fresh refit"),
    )

    with pytest.raises(
        joint.M2500ReconstructionError,
        match="missing regenerated ebv_agn",
    ):
        joint.run_hybrid_fit(_run_record(), args)


def test_fresh_m2500_reconstruction_error_is_not_silenced(monkeypatch, tmp_path):
    args = _hybrid_args(tmp_path)
    args.cache_dir = str(tmp_path / "cache")

    def fail_load(*args, **kwargs):
        raise joint.M2500ReconstructionError("systemic m2500 failure")

    monkeypatch.setattr(joint.legacy, "load_spec_from_cache", fail_load)

    with pytest.raises(
        joint.M2500ReconstructionError,
        match="systemic m2500 failure",
    ):
        joint.run_one_fit(_run_record(), args)


def test_resumed_fit_recomputes_and_writes_new_schema(monkeypatch, tmp_path):
    rec = _run_record()
    args = _hybrid_args(tmp_path)
    source = joint.posterior_bundle_path(args.resume, rec)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"immutable old posterior")
    filter_names = [f"{band}_sdss" for band in "ugriz"]
    prediction = _component_prediction(filter_names)
    prediction.update(
        {
            "ebv_gal": np.array([0.02, 0.02]),
            "ebv_agn": np.array([0.03, 0.03]),
        }
    )

    class DummyFitter:
        predictive = {"stale": np.array([1.0])}
        samples = {
            "log_agn_amp": np.log(np.array([1.0e38, 1.1e38])),
            "pl_slope": np.array([-1.8, -1.8]),
            "pl_bend_loc": np.array([1000.0, 1000.0]),
            "pl_bend_width": np.array([10.0, 10.0]),
            "host_capture_group_fraction": np.array([[0.4], [0.5]]),
        }
        config = SimpleNamespace(
            observation=SimpleNamespace(
                object_id=joint.joint_saved_name(rec),
                redshift=rec["z"],
            ),
            photometry=SimpleNamespace(
                filter_names=filter_names,
                host_capture_group=[joint.QVC_PSF_HOST_CAPTURE_GROUP] * 5,
            ),
            galaxy=SimpleNamespace(cosmology_h0=70.0, cosmology_om0=0.3),
        )

        @staticmethod
        def _predictive_return_sites(kind, **kwargs):
            return ["pred_fluxes", "variable_agn_fluxes"]

        def predict(self, *, kind, **kwargs):
            assert kind == "plot"
            assert self.predictive is None
            assert kwargs["component_rest_wavelengths"] == (2500.0,)
            return prediction

        def save(self, output_dir):
            path = joint.posterior_bundle_path(output_dir, rec)
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w") as handle:
                handle.attrs["posterior_bundle_format"] = joint.POSTERIOR_BUNDLE_FORMAT
                handle.create_dataset(
                    "samples/log_agn_amp", data=self.samples["log_agn_amp"]
                )
                handle.create_dataset(
                    "samples/host_capture_group_fraction",
                    data=self.samples["host_capture_group_fraction"],
                )
            return path

        def plot_sed(self, *, output_path, show):
            assert show is False
            fig = plt.figure()
            fig.savefig(output_path)
            return fig

    monkeypatch.setitem(
        sys.modules,
        "jaxsedfit",
        SimpleNamespace(JAXSEDFit=SimpleNamespace(load=lambda path: DummyFitter())),
    )
    monkeypatch.setattr(
        joint,
        "save_spectrum_figure",
        lambda fitter, record, fig_dir: Path(fig_dir) / "spectrum.png",
    )

    result = joint.run_hybrid_fit(rec, args)

    assert result["fit_ok"] is True
    assert result["execution_mode"] == "resumed"
    assert result["object_id"] == rec["object_id"]
    assert result["resumed_from_run"] == "old_run"
    assert result["joint_reduced_chi2"] == pytest.approx(9.0)
    assert result["f_AGN_psf_g"] == pytest.approx(0.7)
    assert result["f_host_2500_psf"] == pytest.approx(0.35)
    assert result["fracAGN_5100_fit"] == pytest.approx(0.7)
    assert result["fracAGN_5100_fit_err"] == pytest.approx(0.068)
    assert result["formed_stellar_mass"] == pytest.approx(1.1e10)
    assert result["a_2500_galaxy"] > 0.0
    assert result["a_2500_internal"] > 0.0
    assert result["a_2500_total"] == pytest.approx(
        result["a_2500_galaxy"] + result["a_2500_internal"]
    )
    assert (
        result["m_2500_attenuated_model"]
        > result["m_2500_dereddened"]
    )
    assert verify_new_posterior_bundle(result["fit_result_path"]).is_file()
    assert source.read_bytes() == b"immutable old posterior"


def test_fresh_fit_writes_same_diagnostic_schema_and_v2_bundle(monkeypatch, tmp_path):
    rec = _run_record()
    args = _hybrid_args(tmp_path)
    args.cache_dir = str(tmp_path / "cache")
    args.progress = False
    args.save_fig = False
    filter_names = [f"{band}_sdss" for band in "ugriz"]
    prediction = _component_prediction(filter_names)
    prediction.update(
        {
            "ebv_gal": np.array([0.02, 0.02]),
            "ebv_agn": np.array([0.03, 0.03]),
        }
    )
    saved_path = joint.posterior_bundle_path(args.output_dir, rec)
    saved_path.parent.mkdir(parents=True)
    with h5py.File(saved_path, "w") as handle:
        handle.attrs["posterior_bundle_format"] = joint.POSTERIOR_BUNDLE_FORMAT
        handle.create_dataset(
            "samples/host_capture_group_fraction", data=np.array([[0.4], [0.5]])
        )

    class DummyHDUL:
        def close(self):
            pass

    class DummyFitResult:
        samples = {
            "log_agn_amp": np.log(np.array([1.0e38, 1.1e38])),
            "pl_slope": np.array([-1.8, -1.8]),
        }
        path = saved_path

        def predict(self, *, kind):
            assert kind == "photometry"
            return prediction

    class DummyFitter:
        def __init__(self, config):
            self.config = config

        @staticmethod
        def _predictive_return_sites(kind, **kwargs):
            return ["pred_fluxes", "variable_agn_fluxes"]

        def fit(self, *, progress_bar):
            assert progress_bar is False
            return DummyFitResult()

        def predict(self, *, kind, **kwargs):
            assert kind == "photometry"
            assert kwargs["component_host_capture_group"] == (
                joint.QVC_PSF_HOST_CAPTURE_GROUP
            )
            return prediction

    config = SimpleNamespace(
        galaxy=SimpleNamespace(cosmology_h0=70.0, cosmology_om0=0.3)
    )
    used_phot = pd.DataFrame({"filter_name": filter_names})
    monkeypatch.setattr(
        joint.legacy, "load_spec_from_cache", lambda *args, **kwargs: DummyHDUL()
    )
    monkeypatch.setattr(
        joint.legacy,
        "get_spectrum_arrays",
        lambda hdul: (
            np.array([4000.0]),
            np.array([1.0]),
            np.array([0.1]),
            2000.0,
        ),
    )
    monkeypatch.setattr(
        joint, "build_joint_config", lambda *args, **kwargs: (config, used_phot)
    )
    monkeypatch.setitem(sys.modules, "jaxsedfit", SimpleNamespace(JAXSEDFit=DummyFitter))

    result = joint.run_one_fit(
        rec,
        args,
        execution_mode="fresh_missing_bundle",
        resumed_from_path=tmp_path / "old" / "missing.h5",
    )

    assert result["fit_ok"] is True
    assert result["execution_mode"] == "fresh_missing_bundle"
    assert result["joint_reduced_chi2"] == pytest.approx(9.0)
    assert result["f_AGN_psf_i"] == pytest.approx(0.7)
    assert result["f_host_2500_psf"] == pytest.approx(0.35)
    assert result["fracAGN_5100_fit"] == pytest.approx(0.7)
    assert result["fracAGN_5100_fit_err"] == pytest.approx(0.068)
    assert result["formed_stellar_mass"] == pytest.approx(1.1e10)
    assert result["a_2500_galaxy"] > 0.0
    assert result["a_2500_internal"] > 0.0
    assert result["a_2500_total"] == pytest.approx(
        result["a_2500_galaxy"] + result["a_2500_internal"]
    )
    assert (
        result["m_2500_attenuated_model"]
        > result["m_2500_dereddened"]
    )
    assert result["fit_result_path"] == str(saved_path)
