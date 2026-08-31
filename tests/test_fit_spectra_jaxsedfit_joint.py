import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from qvc.spectra import fit_spectra_jaxsedfit_joint as joint
from qvc.spectra.catalog_hdf5 import (
    SPECTRA_CATALOG_FORMAT,
    read_spectra_catalog_hdf5,
)
from qvc.spectra.fit_spectra_jaxsedfit_joint import (
    ab_mag_to_mjy,
    add_qvc_psf_photometry,
    empty_psf_agn_fraction_summary,
    estimate_m2500_dereddened,
    extract_compact_host_2500_psf_draws,
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


REPO_ROOT = Path(__file__).resolve().parents[1]


def _record_with_ugriz():
    record = {"object_id": "1452887"}
    for band in "ugriz":
        record[f"psf_mag_{band}"] = 20.0
        record[f"psf_mag_err_{band}"] = 0.1
    return record


def _spectrum_hdul_with_metadata(**metadata):
    dtype = [(name, "U32") for name in metadata]
    row = tuple(str(value) for value in metadata.values())
    data = np.array([row], dtype=dtype)
    return [SimpleNamespace(header={}), SimpleNamespace(data=data)]


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"SURVEY": "sdss"}, 3.0),
        ({"SURVEY": "segue2"}, 3.0),
        ({"SURVEY": "boss"}, 2.0),
        ({"SURVEY": "eboss"}, 2.0),
        ({"RUN2D": "26"}, 3.0),
        ({"RUN2D": "104"}, 3.0),
        ({"RUN2D": "v5_13_2"}, 2.0),
    ],
)
def test_sdss_spectrum_aperture_uses_survey_metadata(metadata, expected):
    hdul = _spectrum_hdul_with_metadata(**metadata)

    assert joint.sdss_spectrum_aperture_diameter_arcsec(hdul) == expected


def test_sdss_spectrum_aperture_rejects_unknown_metadata():
    hdul = _spectrum_hdul_with_metadata(RUN2D="unknown")

    with pytest.raises(ValueError, match="Cannot determine"):
        joint.sdss_spectrum_aperture_diameter_arcsec(hdul)


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


def test_saved_sed_loader_preserves_numeric_aperture_diameter(tmp_path):
    path = tmp_path / "sed.csv"
    pd.DataFrame(
        {
            "object_id": [1452887],
            "filter_name": ["J_ukidss"],
            "flux_mjy": [0.5],
            "flux_err_mjy": [0.1],
            "psf_fwhm_arcsec": [""],
            "aperture_diameter_arcsec": ["2.0"],
            "photometry_method": ["aperture"],
        }
    ).to_csv(path, index=False)

    result = load_saved_sed_photometry(path)

    assert result["aperture_diameter_arcsec"].tolist() == [2.0]
    assert np.isnan(result["psf_fwhm_arcsec"].item())


def test_host_capture_spatial_metadata_rejects_partial_flux_without_scale():
    phot = pd.DataFrame(
        {
            "catalog": ["ukidss_las_dr9", "allwise"],
            "filter_name": ["J_ukidss", "W1"],
            "photometry_method": ["aperture", "profile"],
            "psf_fwhm_arcsec": [np.nan, np.nan],
            "aperture_diameter_arcsec": [np.nan, np.nan],
        }
    )

    with pytest.raises(ValueError, match="J_ukidss.*Regenerate"):
        joint.validate_host_capture_spatial_metadata(phot)

    phot.loc[0, "aperture_diameter_arcsec"] = 2.0
    joint.validate_host_capture_spatial_metadata(phot)


def _sdss_override_frame(object_id="1452887"):
    return pd.DataFrame(
        {
            "object_id": [object_id] * 5,
            "filter_name": [f"{band}_sdss" for band in "ugriz"],
            "flux_mjy": [0.021, 0.032, 0.042, 0.051, 0.072],
            "flux_err_mjy": [0.0012, 0.0008, 0.0009, 0.0017, 0.0042],
        }
    )


def test_sdss_override_loader_preserves_fluxes_and_adds_static_metadata(tmp_path):
    path = tmp_path / "sdss.csv"
    source = _sdss_override_frame()
    source.to_csv(path, index=False)

    result = joint.load_sdss_psf_photometry_overrides(path)

    np.testing.assert_allclose(result["flux_mjy"], source["flux_mjy"])
    np.testing.assert_allclose(result["flux_err_mjy"], source["flux_err_mjy"])
    assert result["photometry_method"].eq("psf").all()
    assert result["psf_fwhm_arcsec"].tolist() == [
        joint.SDSS_STATIC_PSF_FWHM_ARCSEC[band] for band in "ugriz"
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.iloc[:-1], "missing filters"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]]), "duplicate"),
        (
            lambda frame: frame.assign(
                flux_mjy=[0.021, 0.032, 0.0, 0.051, 0.072]
            ),
            "invalid photometry",
        ),
    ],
)
def test_sdss_override_loader_rejects_incomplete_duplicate_or_invalid_rows(
    tmp_path, mutation, message
):
    path = tmp_path / "bad_sdss.csv"
    mutation(_sdss_override_frame()).to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        joint.load_sdss_psf_photometry_overrides(path)


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
    assert "host_capture_group" not in result
    assert result.loc[qvc_rows, "photometry_method"].eq("psf").all()
    actual_fwhm = dict(
        zip(
            result.loc[qvc_rows, "band"],
            result.loc[qvc_rows, "psf_fwhm_arcsec"],
            strict=True,
        )
    )
    assert actual_fwhm == joint.SDSS_STATIC_PSF_FWHM_ARCSEC


def test_sdss_override_replaces_qvc_means_without_changing_values():
    override = joint.load_sdss_psf_photometry_overrides(
        REPO_ROOT / "experiments/j013453_sdss_dr16_psf_photometry.ecsv"
    )
    record = _record_with_ugriz()
    record["object_id"] = "1414639"
    record["_joint_sdss_psf_photometry"] = override.to_dict(orient="records")

    result = add_qvc_psf_photometry(record, pd.DataFrame())

    np.testing.assert_allclose(
        result["flux_mjy"],
        [
            0.021198359299526164,
            0.03168182924375816,
            0.042236726255460745,
            0.0511717475777403,
            0.07153823960134105,
        ],
    )
    np.testing.assert_allclose(
        result["flux_err_mjy"],
        [
            0.001214184622648176,
            0.0008103245443177369,
            0.0009022591254456196,
            0.0016541449886567983,
            0.004220261567172738,
        ],
    )
    assert result["catalog"].eq("sdss_dr16_notebook").all()


def test_qvc_psf_photometry_requires_z_even_if_variability_fit_dropped_it():
    record = _record_with_ugriz()
    record["psf_mag_z"] = np.nan

    with pytest.raises(ValueError, match=r"missing/invalid: \['z'\]"):
        add_qvc_psf_photometry(record, pd.DataFrame())


def test_build_joint_config_uses_current_jaxsedfit_spectral_api(tmp_path):
    pytest.importorskip("jaxsedfit")
    record = {
        **_record_with_ugriz(),
        "sdss_name": "000000.00+000000.0",
        "z": 1.0,
        "ra": 0.0,
        "dec": 0.0,
        "mjd": 55000.0,
    }
    args = SimpleNamespace(
        no_deredden=False,
        wave_min=1250.0,
        wave_max=8000.0,
        dsps_ssp_fn="unused-in-config-construction.h5",
        sed_n_wave=512,
        fit_lines=True,
        fit_fe=False,
        fit_bc=True,
        fit_bal=True,
        line_flux_scale_mjy=0.2,
        photometry_systematics=0.08,
        spectrum_systematics=0.06,
        spectrum_student_t_df=7.0,
        spectrum_scale_prior_sigma_dex=0.12,
        fit_method="optax",
        optax_steps=10,
        optax_lr=5.0e-3,
        plot_init=True,
        seed=3,
        nuts_warmup=10,
        nuts_samples=10,
        nuts_chains=1,
        nuts_target_accept=0.9,
        dense_mass="blocks",
        output_dir=tmp_path / "results",
        fig_dir=tmp_path / "figures",
        save_fig=False,
        save_jaxsedfit_samples=False,
    )

    config, used_phot = joint.build_joint_config(
        record,
        pd.DataFrame(),
        lam=np.array([3000.0, 4000.0, 5000.0]),
        flux=np.array([1.0, 1.1, 1.2]),
        err=np.array([0.1, 0.1, 0.1]),
        resolving_power=2000.0,
        args=args,
        aperture_diameter_arcsec=2.0,
    )

    assert len(used_phot) == 5
    assert config.spectroscopy.resolving_power == pytest.approx(2000.0)
    assert config.spectroscopy.aperture_diameter_arcsec == pytest.approx(2.0)
    assert config.agn.fit_lines is True
    assert config.agn.tied_lines is True
    assert config.agn.fit_feii is False
    assert config.agn.fit_balmer_continuum is True
    assert [component.name for component in config.agn.custom_components] == [
        "bal_nv",
        "bal_siiv",
        "bal_civ",
    ]
    assert all(
        component.metadata["component_type"] == "bal_absorption"
        for component in config.agn.custom_components
    )
    assert config.agn.line_flux_scale_mjy == pytest.approx(0.2)
    assert config.inference.plot_init is True
    assert config.likelihood.spectrum_systematics_width == pytest.approx(0.06)
    assert config.likelihood.spectrum_student_t_df == pytest.approx(7.0)
    assert config.likelihood.spectrum_weight_mode == "resolution_elements"
    assert config.likelihood.fit_spectrum_scale is True
    assert config.likelihood.spectrum_scale_prior_sigma_dex == pytest.approx(0.12)
    assert config.likelihood.use_host_capture_model is True
    assert config.galaxy.host_sfh_model == "delayed_burst"
    burst_age_prior = config.prior_config.host.log_sfh_burst_age_gyr
    burst_tau_prior = config.prior_config.host.log_sfh_burst_tau_gyr
    assert float(burst_age_prior.low) == pytest.approx(np.log(0.01))
    assert float(burst_age_prior.high) == pytest.approx(np.log(0.5))
    assert float(burst_tau_prior.low) == pytest.approx(np.log(0.01))
    assert float(burst_tau_prior.high) == pytest.approx(np.log(0.2))
    assert not hasattr(config.photometry, "host_capture_group")
    assert config.photometry.psf_fwhm_arcsec == [
        joint.SDSS_STATIC_PSF_FWHM_ARCSEC[band] for band in "ugriz"
    ]
    assert config.photometry.aperture_diameter_arcsec == [None] * 5

    nir_phot = pd.DataFrame(
        {
            "source_id": [record["object_id"]],
            "catalog": ["ukidss_las_dr9"],
            "filter_name": ["J_ukidss"],
            "flux_mjy": [0.2],
            "flux_err_mjy": [0.02],
            "is_upper_limit": [False],
            "psf_fwhm_arcsec": [np.nan],
            "aperture_diameter_arcsec": [2.0],
            "photometry_method": ["aperture"],
        }
    )
    config_with_aperture, _ = joint.build_joint_config(
        record,
        nir_phot,
        lam=np.array([3000.0, 4000.0, 5000.0]),
        flux=np.array([1.0, 1.1, 1.2]),
        err=np.array([0.1, 0.1, 0.1]),
        resolving_power=2000.0,
        args=args,
        aperture_diameter_arcsec=2.0,
    )
    aperture_by_filter = dict(
        zip(
            config_with_aperture.photometry.filter_names,
            config_with_aperture.photometry.aperture_diameter_arcsec,
            strict=True,
        )
    )
    assert aperture_by_filter["J_ukidss"] == pytest.approx(2.0)

    args.fit_bal = False
    config_without_bal, _ = joint.build_joint_config(
        record,
        pd.DataFrame(),
        lam=np.array([3000.0, 4000.0, 5000.0]),
        flux=np.array([1.0, 1.1, 1.2]),
        err=np.array([0.1, 0.1, 0.1]),
        resolving_power=2000.0,
        args=args,
        aperture_diameter_arcsec=2.0,
    )
    assert config_without_bal.agn.custom_components == ()


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


def test_m2500_attenuation_matches_jaxsedfit_normalized_curve():
    ebv_gal = np.array([0.02, 0.04])
    ebv_agn = np.array([0.03, 0.05])
    samples = {
        "log_agn_amp": np.log(np.array([1.0e38, 1.1e38])),
        "pl_slope": np.array([-1.8, -1.8]),
        "pl_bend_loc": np.array([1000.0, 1000.0]),
        "pl_bend_width": np.array([10.0, 10.0]),
        "ebv_gal": ebv_gal,
        "ebv_agn": ebv_agn,
    }

    result = estimate_m2500_dereddened(samples, redshift=1.0)

    # JAXSEDFit's GRAHSP optical attenuation branch is
    # 1.2 * (wave / 11000 Angstrom)^-1.2.
    curve_2500 = 1.2 * (2500.0 / 11_000.0) ** -1.2
    assert curve_2500 == pytest.approx(7.101080857438753)
    np.testing.assert_allclose(
        result["a_2500_galaxy_draws"], ebv_gal * curve_2500
    )
    np.testing.assert_allclose(
        result["a_2500_internal_draws"], ebv_agn * curve_2500
    )
    np.testing.assert_allclose(
        result["m_2500_attenuated_model_draws"]
        - result["m_2500_dereddened_draws"],
        (ebv_gal + ebv_agn) * curve_2500,
    )


def test_alpha_nu_1450_2500_is_exact_bent_disk_secant_with_paired_dust():
    pytest.importorskip("jaxsedfit")
    from jaxsedfit.model import _powerlaw_jax

    samples = {
        "log_agn_amp": np.log(np.array([1.0e38, 1.1e38])),
        "pl_slope": np.array([-1.8, -1.6]),
        "pl_bend_loc": np.array([1000.0, 1300.0]),
        "pl_bend_width": np.array([10.0, 7.0]),
        "uv_slope": np.array([0.0, -0.2]),
        "pl_cutoff": np.array([100_000.0, 80_000.0]),
        "ebv_gal": np.array([0.02, 0.03]),
        "ebv_agn": np.array([0.03, 0.04]),
    }

    draws = joint.estimate_joint_hubble_posterior_draws(samples, redshift=1.0)
    disk_args = (
        np.exp(samples["log_agn_amp"]) / 5100.0,
        samples["uv_slope"],
        samples["pl_slope"],
        5100.0,
        samples["pl_bend_loc"],
        samples["pl_bend_width"],
        samples["pl_cutoff"],
    )
    # Evaluate JAXSedFit's authoritative continuum directly so this is an
    # independent gold test, not a comparison of the QVC helper with itself.
    l1450_nu = np.asarray(_powerlaw_jax(1450.0, *disk_args)) * 1450.0**2
    l2500_nu = np.asarray(_powerlaw_jax(2500.0, *disk_args)) * 2500.0**2
    denominator = np.log10(2500.0 / 1450.0)
    expected_intrinsic = np.log10(l1450_nu / l2500_nu) / denominator
    np.testing.assert_allclose(
        expected_intrinsic,
        np.array([-1.04214077, -1.06190063]),
        rtol=2.0e-8,
        atol=2.0e-8,
    )
    attenuation_ratio = (1450.0 / 2500.0) ** -1.2
    expected_attenuated = (
        expected_intrinsic
        - 0.4
        * draws["a_2500_total_draws"]
        * (attenuation_ratio - 1.0)
        / denominator
    )

    np.testing.assert_allclose(
        draws["alpha_nu_intrinsic_1450_2500_draws"], expected_intrinsic
    )
    np.testing.assert_allclose(
        draws["alpha_nu_attenuated_1450_2500_draws"], expected_attenuated
    )
    assert np.all(expected_attenuated < expected_intrinsic)


def test_compact_joint_posterior_uses_one_original_index_axis_for_every_field():
    source_count = 100
    base = np.arange(source_count, dtype=float)
    derived = {
        "alpha_nu_intrinsic_1450_2500_draws": -0.5 + base / 1000.0,
        "alpha_nu_attenuated_1450_2500_draws": -1.0 + base / 1000.0,
        "m_2500_dereddened_draws": 20.0 + base / 1000.0,
        "m_2500_attenuated_model_draws": 20.5 + base / 1000.0,
        "a_2500_galaxy_draws": np.full(source_count, 0.2),
        "a_2500_internal_draws": np.full(source_count, 0.3),
        "a_2500_total_draws": np.full(source_count, 0.5),
    }
    prediction = {"component_host_fraction": (base / 200.0)[:, None]}

    compact, count, indices, original_count = (
        joint.extract_compact_joint_posterior_draws(
            prediction,
            derived,
            object_id="1452887",
            seed=3,
        )
    )

    assert count == 64
    assert original_count == source_count
    assert np.all(np.diff(indices[:count]) > 0)
    assert np.all(indices[count:] == -1)
    np.testing.assert_allclose(
        compact["f_host_2500_psf"][:count],
        prediction["component_host_fraction"][indices[:count], 0],
    )
    np.testing.assert_allclose(
        compact["m_2500_dereddened"][:count],
        derived["m_2500_dereddened_draws"][indices[:count]],
    )
    np.testing.assert_allclose(
        compact["alpha_nu_intrinsic_1450_2500"][:count],
        derived["alpha_nu_intrinsic_1450_2500_draws"][indices[:count]],
    )


def test_m2500_resume_regenerates_dust_and_overrides_stale_values():
    latent = {
        "log_agn_amp": np.log(np.array([1.0e38, 1.1e38])),
        "pl_slope": np.array([-1.8, -1.8]),
        "pl_bend_loc": np.full(2, 1000.0),
        "pl_bend_width": np.full(2, 10.0),
        "uv_slope": np.zeros(2),
        "pl_cutoff": np.full(2, 100_000.0),
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
        "pl_bend_loc": np.full(2, 1000.0),
        "pl_bend_width": np.full(2, 10.0),
        "uv_slope": np.zeros(2),
        "pl_cutoff": np.full(2, 100_000.0),
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
        *joint.ALPHA_NU_CATALOG_SITES,
    }
    assert set(captured) == expected_scalar_sites
    assert "vector_site" not in captured
    for name in expected_scalar_sites:
        assert np.isfinite(result[f"{name}_rhat"])
        assert f"{name}_ess" not in result

    monkeypatch.setattr(
        joint,
        "print_numpyro_summary_dict",
        lambda *args, **kwargs: pytest.fail("summary output should be suppressed"),
    )
    joint.summarize_spectral_convergence(
        grouped,
        redshift=1.0,
        print_summary=False,
    )


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
    for name in joint.DERIVED_HUBBLE_CATALOG_SITES:
        assert np.isnan(result[name])
        assert np.isnan(result[f"{name}_err_lower"])
        assert np.isnan(result[f"{name}_err_upper"])
    assert set(result["_joint_posterior_draws"]) == set(
        joint.JOINT_POSTERIOR_DRAW_FIELDS
    )
    assert result["_joint_posterior_valid_count"] == 0
    assert np.all(result["_joint_posterior_index"] == -1)


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"SDSSS_RUN2D": "26"}, "26"),
        ({"SDSS_RUN2D": "v5_13_2", "SDSSS_RUN2D": "26"}, "v5_13_2"),
        ({}, None),
    ],
)
def test_base_result_canonicalizes_input_run2d_field(tmp_path, record, expected):
    rec = _run_record()
    rec.update(record)

    result = joint._base_result(
        rec,
        _hybrid_args(tmp_path),
        execution_mode="fresh",
    )

    assert result["SDSS_RUN2D"] == expected


def test_save_spectrum_figure_uses_separate_spectrum_filename(tmp_path):
    class FakeFitter:
        def __init__(self):
            self.show_plot = None
            self.plot_residual = None

        def plot_spectrum(self, *, show_plot, plot_residual):
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


def test_release_jaxsedfit_memory_discards_fit_state_and_clears_jax_cache(
    monkeypatch,
):
    figure = plt.figure()
    figure_number = figure.number
    state = SimpleNamespace(
        map_result={"optimizer": np.ones(2)},
        nuts_result={"mcmc": object()},
        ns_result={"sampler": object()},
        samples={"x": np.ones(2)},
        predictive={"model": np.ones((2, 3))},
        predictive_cache={"plot": {"model": np.ones((2, 3))}},
        summary={"x": 1.0},
        figure=figure,
        plot_cache={"sed": object()},
    )
    fitter = SimpleNamespace(
        _fit_state=state,
        context=object(),
        config=object(),
    )
    fitter._reset_fit_state = lambda: setattr(
        fitter, "_fit_state", SimpleNamespace()
    )
    fit_result = SimpleNamespace(
        fitter=fitter,
        samples=state.samples,
        median={"x": 1.0},
        summary=state.summary,
        figure=figure,
        _state=state,
        _spectrum=object(),
    )
    clear_calls = []
    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(clear_caches=lambda: clear_calls.append(True)),
    )

    joint.release_jaxsedfit_memory(fitter, fit_result)

    assert clear_calls == [True]
    assert not plt.fignum_exists(figure_number)
    for name in (
        "map_result",
        "nuts_result",
        "ns_result",
        "samples",
        "predictive",
        "predictive_cache",
        "summary",
        "figure",
        "plot_cache",
    ):
        assert getattr(state, name) is None
    assert fitter.context is None
    assert fitter.config is None
    assert fit_result.fitter is None
    assert fit_result.samples is None
    assert fit_result._state is None


def test_plot_init_saves_each_stage_without_showing(tmp_path):
    calls = []

    class FakeFitter:
        def plot_sed(self, *, output_path=None, show=False, title=None):
            calls.append(
                {"output_path": output_path, "show": show, "title": title}
            )
            figure = plt.figure()
            if output_path is not None:
                figure.savefig(output_path)
            return figure

        def fit(self, *, progress_bar):
            assert progress_bar is True
            self.plot_sed(
                show=True,
                title="Stage 1 continuum/host MAP initialization",
            )
            self.plot_sed(
                show=True,
                title="Stage 2 smooth spectral-feature MAP initialization",
            )
            self.plot_sed(
                show=True,
                title="Stage 3 full MAP initialization",
            )
            return "fit-result"

    fitter = FakeFitter()
    rec = {"z": 0.304, "sdss_name": "013453.20-001842.3"}
    args = SimpleNamespace(plot_init=True, progress=True, fig_dir=tmp_path)

    result = joint.fit_with_saved_initialization_plots(fitter, rec, args)

    assert result == "fit-result"
    assert [call["show"] for call in calls] == [False, False, False]
    assert [Path(call["output_path"]).name for call in calls] == [
        "z0.304_013453.20-001842.3_joint_init_stage1.png",
        "z0.304_013453.20-001842.3_joint_init_stage2.png",
        "z0.304_013453.20-001842.3_joint_init_stage3.png",
    ]
    assert all(Path(call["output_path"]).is_file() for call in calls)
    assert fitter.plot_sed.__func__ is FakeFitter.plot_sed


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
    desired_host_fraction = np.array([0.2, 0.3, 0.4, 0.5])
    effective_radius = (
        np.sqrt(2.0) * joint.SDSS_TYPICAL_PSF_FWHM_ARCSEC / 2.354820045
    )
    capture = effective_radius**2 / (effective_radius**2 + 1.0)
    agn_2500 = capture * (1.0 - desired_host_fraction) / desired_host_fraction
    rest_wave = np.broadcast_to(np.array([2000.0, 3000.0]), (4, 2))
    prediction = {
        "pred_fluxes": total,
        "variable_agn_fluxes": 0.7 * total,
        "fracAGN_5100_fit": np.array([0.6, 0.8]),
        "formed_stellar_mass": np.array([1.0e10, 1.2e10]),
        "component_host_fraction": desired_host_fraction[:, None],
        "rest_wave": rest_wave,
        "host_total_rest_sed": np.ones((4, 2)),
        "dust_rest_sed": np.zeros((4, 2)),
        "agn_rest_sed": np.repeat(agn_2500[:, None], 2, axis=1),
        "log_host_capture_scale_arcsec_fit": np.zeros(4),
        "pl_bend_loc": np.full(4, 1000.0),
        "pl_bend_width": np.full(4, 10.0),
        "uv_slope": np.zeros(4),
        "pl_cutoff": np.full(4, 100_000.0),
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


def test_total_psf_photometry_uses_authoritative_joint_indices_and_band_order():
    filter_names = ["W2", "i_sdss", "g_sdss", "z_sdss", "u_sdss", "r_sdss"]
    total = np.arange(1.0, 37.0, dtype=float).reshape(6, 6)
    posterior_index = np.full(64, -1, dtype=np.int32)
    posterior_index[:2] = [1, 4]

    compact = joint.extract_aligned_joint_psf_photometry_draws(
        {"pred_fluxes": total},
        filter_names,
        posterior_index,
        2,
    )

    expected_filter_indices = [4, 2, 5, 1, 3]
    np.testing.assert_allclose(
        compact[:2],
        total[np.ix_([1, 4], expected_filter_indices)],
    )
    assert compact.shape == (64, 5)
    assert compact.dtype == np.float32
    assert np.all(np.isnan(compact[2:]))


def test_total_psf_photometry_rejects_nonpositive_selected_flux():
    total = np.ones((2, 5), dtype=float)
    total[1, 1] = 0.0
    posterior_index = np.full(64, -1, dtype=np.int32)
    posterior_index[:2] = [0, 1]

    with pytest.raises(ValueError, match="finite and positive"):
        joint.extract_aligned_joint_psf_photometry_draws(
            {"pred_fluxes": total},
            [f"{band}_sdss" for band in "ugriz"],
            posterior_index,
            2,
        )


def test_compact_host_2500_psf_draws_preserve_values_and_pad_to_64():
    fractions = np.array([[0.15], [0.25], [0.35], [0.45]], dtype=float)

    compact, valid_count = extract_compact_host_2500_psf_draws(
        {"component_host_fraction": fractions},
        object_id="1452887",
        seed=3,
    )

    assert compact.shape == (64,)
    assert compact.dtype == np.float32
    assert valid_count == 4
    np.testing.assert_allclose(compact[:4], fractions[:, 0].astype(np.float32))
    assert np.all(np.isnan(compact[4:]))


def test_joint_fit_result_writer_moves_private_draw_payload_out_of_catalog(tmp_path):
    path = tmp_path / "chunk.h5"
    draws = np.full((64, 5), np.nan, dtype=np.float32)
    draws[:2] = 0.75
    derived = joint.estimate_joint_hubble_posterior_draws(
        {
            "log_agn_amp": np.log(np.array([1.0e38, 1.1e38])),
            "pl_slope": np.array([-1.8, -1.7]),
            "ebv_gal": np.array([0.02, 0.03]),
            "ebv_agn": np.array([0.03, 0.04]),
        },
        redshift=1.0,
    )
    joint_draws, joint_count, posterior_index, source_count = (
        joint.extract_compact_joint_posterior_draws(
            {"component_host_fraction": np.array([[0.25], [0.20]])},
            derived,
            object_id="1452887",
            seed=3,
        )
    )
    scalar_summary = joint.summarize_joint_hubble_posterior_draws(derived)
    scalar_summary.update(
        joint.summarize_host_2500_psf(
            {"component_host_fraction": np.array([[0.25], [0.20]])}
        )
    )
    rows = [
        {
            "object_id": "1452887",
            "fit_ok": True,
            "fit_backend": "jaxsedfit_joint",
            "fracAGN_5100_fit": 0.65,
            "fracAGN_5100_fit_err": 0.04,
            "formed_stellar_mass": 1.1e10,
            "f_AGN_psf_g": 0.75,
            "mw_deredden_applied": True,
            "joint_posterior_draw_source": "synthetic_test",
            **scalar_summary,
            "_psf_agn_fraction_draws": draws,
            "_psf_agn_fraction_valid_count": 2,
            "_joint_posterior_draws": joint_draws,
            "_joint_posterior_valid_count": joint_count,
            "_joint_posterior_index": posterior_index,
            "_joint_posterior_source_draw_count": source_count,
            "_joint_posterior_selection_seed": 3,
            "_joint_psf_photometry_draws": np.vstack(
                [
                    np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0]] * 2),
                    np.full((62, 5), np.nan),
                ]
            ).astype(np.float32),
            "_joint_psf_photometry_provenance": {
                "prediction_source": "synthetic_test",
                "jaxsedfit_git_commit": "a" * 40,
            },
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
        assert handle.attrs["f_host_2500_psf_capture_model"] == (
            "sdss_typical_fwhm_with_fitted_host_scale"
        )
        assert handle.attrs["f_host_2500_psf_fwhm_arcsec"] == pytest.approx(1.4)
        np.testing.assert_allclose(
            handle["joint_posterior_draws/f_host_2500_psf"][0, :2],
            [0.25, 0.20],
        )
        assert handle["joint_posterior_draws/valid_count"][0] == 2
        np.testing.assert_array_equal(
            handle["joint_posterior_draws/posterior_index"][0, :2], [0, 1]
        )
        assert "_joint_psf_photometry_draws" not in handle["catalog"]
        np.testing.assert_allclose(
            handle["joint_psf_photometry_draws/values_mjy"][0, :2, 1],
            [2.0, 2.0],
        )

    catalog = read_spectra_catalog_hdf5(path)
    assert catalog.joint_psf_photometry_bands == tuple(
        f"{band}_sdss" for band in "ugriz"
    )


def test_joint_fit_result_writer_preserves_v3_schema_for_empty_shard(tmp_path):
    path = tmp_path / "empty.h5"

    write_joint_fit_results_hdf5(path, [])
    catalog = read_spectra_catalog_hdf5(path)

    assert catalog.catalog_format == SPECTRA_CATALOG_FORMAT
    assert catalog.frame.empty
    assert set(joint.JOINT_POSTERIOR_SCALAR_SUMMARY_FIELDS) <= set(
        catalog.frame.columns
    )
    assert catalog.joint_posterior_draws[
        "alpha_nu_intrinsic_1450_2500"
    ].shape == (0, 64)


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
        "alpha_nu_intrinsic_1450_2500": -0.5,
        "alpha_nu_attenuated_1450_2500": -1.0,
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
        "alpha_nu_intrinsic_1450_2500": -0.5,
        "alpha_nu_attenuated_1450_2500": -1.0,
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
        "alpha_nu_intrinsic_1450_2500": -0.5,
        "alpha_nu_attenuated_1450_2500": -1.0,
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
        "alpha_nu_intrinsic_1450_2500": -0.5,
        "alpha_nu_attenuated_1450_2500": -1.0,
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


def test_catalog_prediction_requests_legacy_csv_scalar_sites_via_public_api():
    class DummyFitter:
        def predict(self, *, kind, **kwargs):
            assert kind == "photometry"
            self.prediction_kwargs = kwargs
            return {"fracAGN_5100_fit": np.array([0.6, 0.8])}

    fitter = DummyFitter()

    prediction = predict_catalog_posterior(fitter, kind="photometry")

    assert "fracAGN_5100_fit" in prediction
    assert set(joint.LEGACY_CSV_SCALAR_PREDICTION_SITES) <= set(
        fitter.prediction_kwargs["extra_return_sites"]
    )
    assert set(joint.M2500_POSTERIOR_SITES) <= set(
        fitter.prediction_kwargs["required_return_sites"]
    )


def test_sdss_psf_host_fraction_uses_prediction_only_typical_fwhm():
    prediction = {
        "rest_wave": np.array([2000.0, 3000.0]),
        "host_total_rest_sed": np.array([[1.0, 3.0], [2.0, 4.0]]),
        "dust_rest_sed": np.array([[0.2, 0.4], [0.1, 0.3]]),
        "agn_rest_sed": np.array([[4.0, 6.0], [5.0, 7.0]]),
        "log_host_capture_scale_arcsec_fit": np.log(np.array([1.0, 2.0])),
    }

    result = joint.add_sdss_psf_host_fraction_prediction(prediction)

    effective_radius = (
        np.sqrt(2.0) * joint.SDSS_TYPICAL_PSF_FWHM_ARCSEC / 2.354820045
    )
    capture = effective_radius**2 / (
        effective_radius**2 + np.array([1.0, 2.0]) ** 2
    )
    host = np.array([2.3, 3.2])
    agn = np.array([5.0, 6.0])
    expected = capture * host / (agn + capture * host)
    np.testing.assert_allclose(
        result["component_host_capture_fraction"][:, 0], capture
    )
    np.testing.assert_allclose(result["component_host_fraction"][:, 0], expected)
    assert "component_host_fraction" not in prediction


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


def test_resume_preflight_rejects_old_and_accepts_main_bundle(tmp_path):
    args = _hybrid_args(tmp_path)
    rec = _run_record()
    path = joint.posterior_bundle_path(args.resume, rec)
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("samples/log_agn_amp", data=np.ones(2))

    with pytest.raises(
        joint.IncompatibleHostCaptureResumeError,
        match="static SDSS ugriz PSF FWHMs",
    ):
        joint.preflight_resume_host_capture_bundles([rec], args)

    with h5py.File(path, "a") as handle:
        handle.attrs[joint.HOST_CAPTURE_BUNDLE_ATTR] = (
            joint.HOST_CAPTURE_BUNDLE_MARKER
        )
        handle.attrs[joint.HOST_CAPTURE_PSF_FWHM_ATTR] = (
            joint.SDSS_TYPICAL_PSF_FWHM_ARCSEC
        )
        handle.attrs[joint.HOST_CAPTURE_QVC_FWHM_ATTR] = np.asarray(
            [joint.SDSS_STATIC_PSF_FWHM_ARCSEC[band] for band in "ugriz"]
        )
        handle.create_dataset("samples/log_host_capture_scale_arcsec", data=np.ones(2))
    joint.preflight_resume_host_capture_bundles([rec], args)


def test_resume_validation_rejects_missing_spatial_scale_capture_fractions(
    tmp_path,
):
    filter_names = [f"{band}_sdss" for band in "ugriz"]
    fitter = SimpleNamespace(
        config=SimpleNamespace(
            likelihood=SimpleNamespace(use_host_capture_model=True),
            photometry=SimpleNamespace(
                filter_names=filter_names,
                photometry_method=["psf"] * 5,
                psf_fwhm_arcsec=[
                    joint.SDSS_STATIC_PSF_FWHM_ARCSEC[band] for band in "ugriz"
                ],
            ),
        ),
        samples={
            "log_host_capture_scale_arcsec": np.zeros(4),
            "missing_psf_host_capture_fraction": np.full((4, 1), 0.5),
        },
    )

    with pytest.raises(
        joint.IncompatibleHostCaptureResumeError,
        match="without spatial metadata",
    ):
        joint.validate_resume_host_capture_fitter(fitter, tmp_path / "samples.h5")


def test_resume_bal_validation_requires_saved_bal_components(tmp_path):
    path = tmp_path / "bal_samples.h5"
    fitter = SimpleNamespace(
        config=SimpleNamespace(
            agn=SimpleNamespace(custom_components=())
        )
    )

    with pytest.raises(
        joint.IncompatibleBALResumeError,
        match="lacks BAL components",
    ):
        joint.validate_resume_bal_fitter(fitter, path)

    fitter.config.agn.custom_components = tuple(
        SimpleNamespace(
            name=name,
            metadata={"component_type": "bal_absorption"},
        )
        for name in ("bal_nv", "bal_siiv", "bal_civ")
    )
    joint.validate_resume_bal_fitter(fitter, path)


def test_resume_preflight_rejects_shared_group_bundle(tmp_path):
    args = _hybrid_args(tmp_path)
    rec = _run_record()
    path = joint.posterior_bundle_path(args.resume, rec)
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        handle.attrs[joint.HOST_CAPTURE_BUNDLE_ATTR] = "qvc_sdss_psf"
        handle.attrs[joint.HOST_CAPTURE_PSF_FWHM_ATTR] = 1.4
        handle.create_dataset("samples/host_capture_group_fraction", data=np.ones((2, 1)))
    with pytest.raises(joint.IncompatibleHostCaptureResumeError):
        joint.preflight_resume_host_capture_bundles([rec], args)


def test_resume_preflight_allows_only_explicitly_unannotated_main_bundle(
    tmp_path,
):
    args = _hybrid_args(tmp_path)
    args.allow_unannotated_resume_bundle = True
    rec = _run_record()
    path = joint.posterior_bundle_path(args.resume, rec)
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("samples/log_host_capture_scale_arcsec", data=np.ones(2))

    with pytest.warns(RuntimeWarning, match="Explicitly accepting"):
        joint.preflight_resume_host_capture_bundles([rec], args)

    with h5py.File(path, "a") as handle:
        handle.attrs[joint.HOST_CAPTURE_BUNDLE_ATTR] = "some_other_group"
    with pytest.raises(
        joint.IncompatibleHostCaptureResumeError,
        match="static SDSS ugriz PSF FWHMs",
    ):
        joint.preflight_resume_host_capture_bundles([rec], args)


def test_prepared_resume_records_filter_in_requested_order(tmp_path):
    path = tmp_path / "prepared.csv"
    pd.DataFrame(
        {
            "object_id": ["2", "1"],
            "sdss_name": ["second", "first"],
            "plate": [2, 1],
            "fiber": [20, 10],
            "mjd": [52_002, 52_001],
            "z": [2.0, 1.0],
            "ra": [20.0, 10.0],
            "dec": [0.2, 0.1],
        }
    ).to_csv(path, index=False)

    records = joint.load_prepared_resume_records(path, ["1", "2"])

    assert [record["object_id"] for record in records] == ["1", "2"]
    assert [record["sdss_name"] for record in records] == ["first", "second"]


def test_prepared_resume_records_reject_missing_requested_id(tmp_path):
    path = tmp_path / "prepared.csv"
    pd.DataFrame(
        {
            "object_id": ["1"],
            "sdss_name": ["first"],
            "plate": [1],
            "fiber": [10],
            "mjd": [52_001],
            "z": [1.0],
            "ra": [10.0],
            "dec": [0.1],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="lack 1 requested"):
        joint.load_prepared_resume_records(path, ["2"])


def test_resume_build_records_uses_prepared_manifest_without_sed_reload(tmp_path):
    path = tmp_path / "prepared.csv"
    pd.DataFrame(
        {
            "object_id": ["1"],
            "sdss_name": ["first"],
            "plate": [1],
            "fiber": [10],
            "mjd": [52_001],
            "z": [1.0],
            "ra": [10.0],
            "dec": [0.1],
        }
    ).to_csv(path, index=False)
    args = SimpleNamespace(
        resume="old_run",
        resume_only=True,
        resume_records_path=str(path),
        filter_object_id=["1"],
        sed_photometry_path=str(tmp_path / "deliberately_missing.csv"),
    )

    records = joint.build_records(args)

    assert [record["object_id"] for record in records] == ["1"]
    assert "_joint_photometry" not in records[0]


def test_hybrid_resume_build_records_retains_fresh_fallback_photometry(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        joint.legacy,
        "build_records",
        lambda args: [{"object_id": "1"}],
    )
    monkeypatch.setattr(
        joint,
        "load_saved_sed_photometry",
        lambda path: pd.DataFrame(
            {
                "source_id": ["1"],
                "filter_name": ["W1"],
                "flux_mjy": [1.0],
                "flux_err_mjy": [0.1],
            }
        ),
    )
    args = SimpleNamespace(
        resume="old_run",
        resume_only=False,
        resume_records_path=None,
        sed_photometry_path=str(tmp_path / "photometry.csv"),
    )

    records = joint.build_records(args)

    assert records[0]["_joint_photometry"][0]["filter_name"] == "W1"


def test_build_records_requires_override_for_every_selected_object(
    monkeypatch, tmp_path
):
    base_path = tmp_path / "base.csv"
    pd.DataFrame(
        {
            "object_id": ["1", "2"],
            "filter_name": ["W1", "W1"],
            "flux_mjy": [1.0, 1.0],
            "flux_err_mjy": [0.1, 0.1],
        }
    ).to_csv(base_path, index=False)
    override_path = tmp_path / "sdss.csv"
    _sdss_override_frame(object_id="1").to_csv(override_path, index=False)
    selected_records = [{"object_id": "1"}]
    monkeypatch.setattr(
        joint.legacy,
        "build_records",
        lambda args: [dict(record) for record in selected_records],
    )
    args = SimpleNamespace(
        resume=None,
        resume_only=False,
        resume_records_path=None,
        filter_object_id=["1"],
        sed_photometry_path=str(base_path),
        sdss_psf_photometry_path=str(override_path),
    )

    records = joint.build_records(args)

    assert len(records[0]["_joint_sdss_psf_photometry"]) == 5
    assert {
        row["filter_name"] for row in records[0]["_joint_sdss_psf_photometry"]
    } == {f"{band}_sdss" for band in "ugriz"}

    selected_records.append({"object_id": "2"})
    args.filter_object_id = ["1", "2"]

    with pytest.raises(ValueError, match=r"lacks selected object ID.*2"):
        joint.build_records(args)


def test_parse_args_rejects_no_deredden_for_mandatory_v3_colors(tmp_path):
    with pytest.raises(SystemExit):
        joint.parse_args(
            [
                "--mode", "fit",
                str(tmp_path / "out.h5"),
                "--sed-photometry-path", str(tmp_path / "phot.csv"),
                "--filter_object_id", "1",
                "--no-deredden",
            ]
        )


def test_parse_args_fit_bal_is_opt_in(tmp_path):
    common = [
        "--mode", "fit",
        str(tmp_path / "out.h5"),
        "--sed-photometry-path", str(tmp_path / "phot.csv"),
        "--filter_object_id", "1",
    ]

    assert joint.parse_args(common).fit_bal is False
    assert joint.parse_args([*common, "--fit-bal"]).fit_bal is True


def test_parse_args_accepts_fit_bal_with_resume(tmp_path):
    resume_dir = tmp_path / "old" / "all"
    resume_dir.mkdir(parents=True)

    args = joint.parse_args(
        [
            "--mode", "fit",
            str(tmp_path / "out.h5"),
            "--sed-photometry-path", str(tmp_path / "phot.csv"),
            "--filter_object_id", "1",
            "--fit-bal",
            "--resume", str(resume_dir),
        ]
    )

    assert args.fit_bal is True
    assert args.resume == str(resume_dir)


def test_parse_args_rejects_removed_no_host_capture_baseline_flag(tmp_path):
    common = [
        "--mode", "fit",
        str(tmp_path / "out.h5"),
        "--sed-photometry-path", str(tmp_path / "phot.csv"),
        "--filter_object_id", "1",
    ]

    with pytest.raises(SystemExit):
        joint.parse_args([*common, "--no-host-capture-model"])


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


def test_parse_args_accepts_experiment_sdss_psf_photometry_path(tmp_path):
    override = tmp_path / "sdss.csv"
    args = joint.parse_args(
        [
            "--mode",
            "fit",
            str(tmp_path / "out.h5"),
            "--sed-photometry-path",
            str(tmp_path / "phot.csv"),
            "--sdss-psf-photometry-path",
            str(override),
            "--filter_object_id",
            "1414639",
        ]
    )

    assert args.sdss_psf_photometry_path == str(override)


def test_load_sdss_spectrum_selection_overrides(tmp_path):
    path = tmp_path / "spectrum.csv"
    pd.DataFrame(
        {
            "object_id": ["1414639"],
            "plate": [698],
            "mjd": [52203],
            "fiber": [279],
            "z": [0.3036496],
        }
    ).to_csv(path, index=False)

    selection = joint.load_sdss_spectrum_selection_overrides(path).iloc[0]

    assert selection["object_id"] == "1414639"
    assert selection["plate"] == 698
    assert selection["mjd"] == 52203
    assert selection["fiber"] == 279
    assert selection["z"] == pytest.approx(0.3036496)


@pytest.mark.parametrize(
    "rows,match",
    [
        (
            [
                {"object_id": "1", "plate": 1, "mjd": 2, "fiber": 3, "z": 0.3},
                {"object_id": "1", "plate": 4, "mjd": 5, "fiber": 6, "z": 0.4},
            ],
            "duplicate",
        ),
        (
            [{"object_id": "1", "plate": 1, "mjd": 2, "fiber": 0, "z": 0.3}],
            "positive integer",
        ),
        (
            [{"object_id": "1", "plate": 1, "mjd": 2, "fiber": 3, "z": float("nan")}],
            "non-finite",
        ),
    ],
)
def test_load_sdss_spectrum_selection_overrides_rejects_invalid(
    tmp_path, rows, match
):
    path = tmp_path / "spectrum.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        joint.load_sdss_spectrum_selection_overrides(path)


def test_build_records_applies_exact_sdss_spectrum_selection(monkeypatch, tmp_path):
    selection_path = tmp_path / "spectrum.csv"
    pd.DataFrame(
        {
            "object_id": ["1414639"],
            "plate": [698],
            "mjd": [52203],
            "fiber": [279],
            "z": [0.3036496],
        }
    ).to_csv(selection_path, index=False)
    phot_path = tmp_path / "phot.csv"
    pd.DataFrame(
        columns=["source_id", "filter_name", "flux_mjy", "flux_err_mjy"]
    ).to_csv(phot_path, index=False)
    monkeypatch.setattr(
        joint.legacy,
        "build_records",
        lambda args: [
            {
                "object_id": "1414639",
                "plate": 4230,
                "mjd": 55483,
                "fiber": 232,
                "z": 0.3042515647,
            }
        ],
    )
    args = SimpleNamespace(
        resume=None,
        resume_only=False,
        resume_records_path=None,
        filter_object_id=["1414639"],
        sed_photometry_path=str(phot_path),
        sdss_psf_photometry_path=None,
        sdss_spectrum_selection_path=str(selection_path),
    )

    record = joint.build_records(args)[0]

    assert (record["plate"], record["mjd"], record["fiber"]) == (698, 52203, 279)
    assert record["z"] == pytest.approx(0.3036496)


def test_parse_args_accepts_experiment_sdss_spectrum_selection_path(tmp_path):
    selection = tmp_path / "spectrum.ecsv"
    args = joint.parse_args(
        [
            "--mode",
            "fit",
            str(tmp_path / "out.h5"),
            "--sed-photometry-path",
            str(tmp_path / "phot.csv"),
            "--sdss-spectrum-selection-path",
            str(selection),
            "--filter_object_id",
            "1414639",
        ]
    )

    assert args.sdss_spectrum_selection_path == str(selection)


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


@pytest.mark.parametrize(
    "flag",
    ["--resume-only", "--allow-unannotated-resume-bundle"],
)
def test_parse_args_requires_resume_for_resume_safety_flags(tmp_path, flag):
    with pytest.raises(SystemExit):
        joint.parse_args(
            [
                "--mode",
                "fit",
                str(tmp_path / "chunk.h5"),
                "--sed-photometry-path",
                str(tmp_path / "photometry.csv"),
                "--filter_object_id",
                "1452887",
                flag,
            ]
        )


def test_parse_args_requires_resume_for_prepared_records(tmp_path):
    records = tmp_path / "records.csv"
    records.write_text("object_id\n1452887\n")
    with pytest.raises(SystemExit):
        joint.parse_args(
            [
                "--mode",
                "fit",
                str(tmp_path / "chunk.h5"),
                "--sed-photometry-path",
                str(tmp_path / "photometry.csv"),
                "--filter_object_id",
                "1452887",
                "--resume-records-path",
                str(records),
            ]
        )


def test_parse_args_requires_resume_only_for_prepared_records(tmp_path):
    resume_dir = tmp_path / "old" / "all"
    resume_dir.mkdir(parents=True)
    records = tmp_path / "records.csv"
    records.write_text("object_id\n1452887\n")
    with pytest.raises(SystemExit):
        joint.parse_args(
            [
                "--mode",
                "fit",
                str(tmp_path / "chunk.h5"),
                "--sed-photometry-path",
                str(tmp_path / "photometry.csv"),
                "--filter_object_id",
                "1452887",
                "--resume",
                str(resume_dir),
                "--resume-records-path",
                str(records),
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


def test_hybrid_resume_only_cleans_artifacts_without_fresh_fit(
    monkeypatch, tmp_path
):
    args = _hybrid_args(tmp_path)
    args.resume_only = True
    rec = _run_record()
    source = joint.posterior_bundle_path(args.resume, rec)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"posterior")

    def failing_resume(rec, received_args, source_path):
        new_bundle = joint.posterior_bundle_path(received_args.output_dir, rec)
        new_bundle.parent.mkdir(parents=True)
        new_bundle.write_bytes(b"partial")
        raise ValueError("prediction failed")

    monkeypatch.setattr(joint, "_run_resumed_fit", failing_resume)
    monkeypatch.setattr(
        joint,
        "run_one_fit",
        lambda *args, **kwargs: pytest.fail("must not launch a fresh refit"),
    )

    with pytest.raises(RuntimeError, match="fresh Optax/NUTS fit was not started"):
        joint.run_hybrid_fit(rec, args)

    assert not joint.posterior_bundle_path(args.output_dir, rec).exists()
    assert source.read_bytes() == b"posterior"


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
            "ebv_gal": np.full(4, 0.02),
            "ebv_agn": np.full(4, 0.03),
        }
    )

    class DummyFitter:
        predictive = {"stale": np.array([1.0])}
        samples = {
            "log_agn_amp": np.log(np.array([1.0e38, 1.1e38, 1.2e38, 1.3e38])),
            "pl_slope": np.full(4, -1.8),
            "pl_bend_loc": np.full(4, 1000.0),
            "pl_bend_width": np.full(4, 10.0),
            "log_host_capture_scale_arcsec": np.zeros(4),
        }
        config = SimpleNamespace(
            observation=SimpleNamespace(
                object_id=joint.joint_saved_name(rec),
                redshift=rec["z"],
            ),
            photometry=SimpleNamespace(
                filter_names=filter_names,
                photometry_method=["psf"] * 5,
                psf_fwhm_arcsec=[
                    joint.SDSS_STATIC_PSF_FWHM_ARCSEC[band] for band in "ugriz"
                ],
            ),
            likelihood=SimpleNamespace(use_host_capture_model=True),
            galaxy=SimpleNamespace(cosmology_h0=70.0, cosmology_om0=0.3),
        )

        @staticmethod
        def _predictive_return_sites(kind, **kwargs):
            return ["pred_fluxes", "variable_agn_fluxes"]

        def predict(self, *, kind, **kwargs):
            assert kind == "plot"
            assert self.predictive is None
            assert set(joint.M2500_POSTERIOR_SITES) <= set(
                kwargs["required_return_sites"]
            )
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
                    "samples/log_host_capture_scale_arcsec",
                    data=self.samples["log_host_capture_scale_arcsec"],
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
    assert np.isfinite(result["alpha_nu_intrinsic_1450_2500"])
    assert (
        result["alpha_nu_attenuated_1450_2500"]
        < result["alpha_nu_intrinsic_1450_2500"]
    )
    assert result["mw_deredden_applied"] is True
    assert result["joint_posterior_draw_source"] == "resume_bundle_reprocess"
    assert result["_joint_posterior_valid_count"] == 4
    np.testing.assert_array_equal(result["_joint_posterior_index"][:4], [0, 1, 2, 3])
    np.testing.assert_allclose(
        result["_joint_psf_photometry_draws"][:4],
        prediction["pred_fluxes"],
    )
    assert result["_joint_psf_photometry_provenance"][
        "prediction_source"
    ] == "saved_posterior_bundle_prediction"
    assert verify_new_posterior_bundle(result["fit_result_path"]).is_file()
    assert source.read_bytes() == b"immutable old posterior"

    args.save_jaxsedfit_samples = False
    args.save_fig = False
    catalog_only = joint.run_hybrid_fit(rec, args)
    assert catalog_only["fit_result_path"] == str(source)
    assert catalog_only["resumed_from_path"] == str(source)


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
            "ebv_gal": np.full(4, 0.02),
            "ebv_agn": np.full(4, 0.03),
        }
    )
    saved_path = joint.posterior_bundle_path(args.output_dir, rec)
    saved_path.parent.mkdir(parents=True)
    with h5py.File(saved_path, "w") as handle:
        handle.attrs["posterior_bundle_format"] = joint.POSTERIOR_BUNDLE_FORMAT
        handle.create_dataset("samples/log_host_capture_scale_arcsec", data=np.zeros(4))

    class DummyHDUL:
        def close(self):
            pass

    class DummyFitResult:
        samples = {
            "log_agn_amp": np.log(np.array([1.0e38, 1.1e38, 1.2e38, 1.3e38])),
            "pl_slope": np.full(4, -1.8),
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
            assert kind == "plot"
            assert set(joint.M2500_POSTERIOR_SITES) <= set(
                kwargs["required_return_sites"]
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
        joint, "sdss_spectrum_aperture_diameter_arcsec", lambda hdul: 2.0
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
    assert np.isfinite(result["alpha_nu_intrinsic_1450_2500"])
    assert (
        result["alpha_nu_attenuated_1450_2500"]
        < result["alpha_nu_intrinsic_1450_2500"]
    )
    assert result["mw_deredden_applied"] is True
    assert result["joint_posterior_draw_source"] == "fresh_fit"
    assert result["_joint_posterior_valid_count"] == 4
    np.testing.assert_array_equal(result["_joint_posterior_index"][:4], [0, 1, 2, 3])
    np.testing.assert_allclose(
        result["_joint_psf_photometry_draws"][:4],
        prediction["pred_fluxes"],
    )
    assert result["_joint_psf_photometry_provenance"][
        "prediction_source"
    ] == "fresh_fit_prediction"
    assert result["fit_result_path"] == str(saved_path)
