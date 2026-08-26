from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from qvc.hubble.completeness_mock_catalog import (
    COMPLETENESS_MOCK_SCHEMA_VERSION,
    COMPLETENESS_MOCK_SEMANTICS_VERSION,
)
from qvc.hubble.hubble_completeness_refactored import Completeness3D
from qvc.hubble.hubble_fit import (
    LATENT_ALPHA_COMPLETENESS_MODE,
    build_latent_alpha_config_from_args,
    run_single,
    standardization_plot_posterior_view,
    validate_loaded_spectra_catalog_compatibility,
    validate_latent_alpha_mock_semantics,
)
from qvc.hubble.hubble_fit_jax import (
    _completeness_loglike_jax,
    _prepare_completeness_for_jax,
    run_single_jax,
)
from qvc.hubble.hubble_likelihood import completeness_loglike_for_data
from qvc.hubble.hubble_model import get_model_params
from qvc.hubble.latent_alpha_completeness import (
    BETA_ALPHA_L_PARAMETER,
    LatentAlphaConfig,
    deterministic_joint_draw_indices,
    response_coefficient_names,
)


def _synthetic_completeness_params():
    # The real map is stored at centers inside the exact [18.5, 24.0]
    # selection support.
    mag = np.array([18.6, 20.0, 22.0, 23.9])
    redshift = np.array([0.5, 1.8, 3.2])
    fhost = np.array([0.1, 0.5, 0.9])
    mm, zz, ff = np.meshgrid(mag, redshift, fhost, indexing="ij")
    cube = 1.0 / (1.0 + np.exp(-(
        2.1 - 0.55 * (mm - 20.5) - 0.25 * (zz - 1.5) - 1.1 * ff
    )))
    model = Completeness3D(
        mag,
        redshift,
        fhost,
        cube,
        magnitude_support=(18.5, 24.0),
    )
    return (
        model,
        mag,
        redshift,
        fhost,
        1.0,
        1.0,
        1.0,
        # Real maps carry a nonzero mock-smoothing bandwidth here.  It must
        # not be reused as physical scatter by the JAX likelihood.
        0.2,
        {"source": "synthetic_test"},
    )


def _joint_draw_data(n_objects=3):
    phase = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    host = np.stack(
        [np.clip(0.12 + 0.18 * index + 0.08 * np.sin(phase), 0.0, 1.0)
         for index in range(n_objects)]
    )
    intrinsic = np.stack(
        [-0.72 + 0.16 * index + 0.24 * np.cos(phase + 0.3 * index)
         for index in range(n_objects)]
    )
    dereddened_magnitude = np.stack(
        [19.0 + 1.5 * index + 0.08 * np.sin(phase + 0.2 * index)
         for index in range(n_objects)]
    )
    attenuated_magnitude = dereddened_magnitude + np.stack(
        [0.30 + 0.04 * np.cos(phase - 0.1 * index)
         for index in range(n_objects)]
    )
    return {
        "f_host_2500_psf": np.median(host, axis=1),
        "f_host_2500_psf_draws": host,
        "alpha_nu_intrinsic_1450_2500_draws": intrinsic,
        # Deliberately very different.  An attenuation-retaining LF must still
        # use the intrinsic slope as its latent variable.
        "alpha_nu_attenuated_1450_2500_draws": intrinsic - 4.0,
        "m_2500_dereddened_draws": dereddened_magnitude,
        "m_2500_attenuated_model_draws": attenuated_magnitude,
        "joint_posterior_valid_count": np.full(n_objects, 64),
    }


def _valid_v3_frame(n_objects=2):
    phase = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    host = np.stack(
        [0.25 + 0.03 * np.sin(phase + index) for index in range(n_objects)]
    )
    intrinsic = np.stack(
        [-0.6 + 0.04 * np.cos(phase + index) for index in range(n_objects)]
    )
    a_galaxy = np.stack(
        [0.08 + 0.005 * np.cos(phase + index) for index in range(n_objects)]
    )
    a_internal = np.stack(
        [0.12 + 0.008 * np.sin(phase + index) for index in range(n_objects)]
    )
    a_total = a_galaxy + a_internal
    dereddened = np.stack(
        [20.0 + index + 0.03 * np.sin(phase) for index in range(n_objects)]
    )
    values = {
        "f_host_2500_psf_draws": host,
        "alpha_nu_intrinsic_1450_2500_draws": intrinsic,
        "alpha_nu_attenuated_1450_2500_draws": intrinsic - 0.25,
        "m_2500_dereddened_draws": dereddened,
        "m_2500_attenuated_model_draws": dereddened + a_total,
        "a_2500_galaxy_draws": a_galaxy,
        "a_2500_internal_draws": a_internal,
        "a_2500_total_draws": a_total,
    }
    frame = pd.DataFrame(
        {
            "qvc_spectra_catalog_format": ["qvc_spectra_catalog_v3"]
            * n_objects,
            "fit_ok": np.ones(n_objects, dtype=bool),
            "mw_deredden_applied": np.ones(n_objects, dtype=bool),
            "joint_posterior_valid_count": np.full(n_objects, 64),
            "joint_posterior_source_draw_count": np.full(n_objects, 128),
            "joint_posterior_index": [
                np.arange(64).copy() for _ in range(n_objects)
            ],
        }
    )
    for column, matrix in values.items():
        frame[column] = list(matrix)
    return frame


def _response_parameters(config):
    values = {
        name: 0.0
        for name in response_coefficient_names(
            config.include_magnitude_interactions
        )
    }
    values.update(
        {
            "alpha_sel_z_p0_linear": 0.45,
            "alpha_sel_z_p0_quadratic": -0.18,
            "alpha_sel_z_p1_linear": 0.22,
        }
    )
    if config.include_magnitude_interactions:
        values["alpha_sel_mag_z_p0_linear"] = -0.25
    if config.mode == "joint":
        values[BETA_ALPHA_L_PARAMETER] = 0.12
    return values


@pytest.mark.parametrize("magnitude_interaction", [False, True])
def test_numpy_and_jax_latent_alpha_selection_terms_match(magnitude_interaction):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)

    completeness_params = _synthetic_completeness_params()
    agn_data = _joint_draw_data()
    config = LatentAlphaConfig.for_lf(
        lf_model="wang2026_type1_lade_a",
        requested_luminosity_state="attenuated",
        mode="joint",
        include_magnitude_interactions=magnitude_interaction,
    )
    parameters = _response_parameters(config)
    m_obs = np.array([19.1, 21.0, 23.1])
    m_model = np.array([19.3, 20.8, 22.7])
    errors = np.array([0.32, 0.45, 0.38])
    redshift = np.array([0.7, 1.7, 3.0])
    distance_modulus = np.array([42.4, 44.1, 46.0])
    selected = deterministic_joint_draw_indices()
    # Exercise both physical-support half-bin tails and draws just beyond the
    # hard cut.  Both backends must extend only the former edge planes.
    magnitude_draws = agn_data["m_2500_attenuated_model_draws"]
    magnitude_draws[0, selected[0:2]] = [18.49, 18.55]
    magnitude_draws[2, selected[-2:]] = [23.95, 24.01]

    numpy_value, _ = completeness_loglike_for_data(
        completeness_params=completeness_params,
        agn_data=agn_data,
        m_obs=m_obs,
        m_obs_err=np.full(3, 0.08),
        m_model=m_model,
        mu_err=errors,
        z=redshift,
        latent_alpha_config=config,
        latent_alpha_parameters=parameters,
        latent_alpha_distance_modulus=distance_modulus,
    )

    prepared = _prepare_completeness_for_jax(
        completeness_params,
        selection_magnitude=m_obs,
    )
    assert float(prepared["sigma"]) == 0.0
    jax_value = _completeness_loglike_jax(
        jnp.asarray(m_model),
        jnp.asarray(errors),
        jnp.asarray(redshift),
        prepared,
        jnp.asarray(agn_data["f_host_2500_psf_draws"][:, selected]),
        None,
        selection_magnitude=jnp.asarray(m_obs),
        latent_alpha_draws=jnp.asarray(
            agn_data["alpha_nu_intrinsic_1450_2500_draws"][:, selected]
        ),
        latent_alpha_magnitude_draws=jnp.asarray(
            agn_data["m_2500_attenuated_model_draws"][:, selected]
        ),
        latent_alpha_distance_modulus=jnp.asarray(distance_modulus),
        latent_alpha_config=config,
        params={name: jnp.asarray(value) for name, value in parameters.items()},
    )
    np.testing.assert_allclose(float(jax_value), numpy_value, rtol=2e-11, atol=2e-11)


def test_attenuation_retaining_lf_uses_intrinsic_slope_but_attenuated_luminosity():
    params = _synthetic_completeness_params()
    data = _joint_draw_data()
    config = LatentAlphaConfig.for_lf(
        lf_model="wang2026_type1_lade_a",
        requested_luminosity_state="attenuated",
    )
    coefficients = _response_parameters(config)
    common = dict(
        completeness_params=params,
        m_obs=np.array([19.1, 21.0, 23.1]),
        m_obs_err=np.full(3, 0.08),
        m_model=np.array([19.3, 20.8, 22.7]),
        mu_err=np.array([0.32, 0.45, 0.38]),
        z=np.array([0.7, 1.7, 3.0]),
        latent_alpha_config=config,
        latent_alpha_parameters=coefficients,
        latent_alpha_distance_modulus=np.array([42.4, 44.1, 46.0]),
    )
    first, _ = completeness_loglike_for_data(agn_data=data, **common)
    data["_latent_alpha_compact_draw_cache"] = None
    data["alpha_nu_attenuated_1450_2500_draws"] += 100.0
    second, _ = completeness_loglike_for_data(agn_data=data, **common)
    assert first == pytest.approx(second, rel=0.0, abs=0.0)


def test_off_and_fixed_zero_have_identical_full_selection_likelihood():
    params = _synthetic_completeness_params()
    common = dict(
        completeness_params=params,
        agn_data=_joint_draw_data(),
        m_obs=np.array([19.1, 21.0, 23.1]),
        m_obs_err=np.full(3, 0.08),
        m_model=np.array([19.3, 20.8, 22.7]),
        mu_err=np.array([0.32, 0.45, 0.38]),
        z=np.array([0.7, 1.7, 3.0]),
        latent_alpha_distance_modulus=np.array([42.4, 44.1, 46.0]),
    )
    off = LatentAlphaConfig(mode="off")
    fixed_zero = LatentAlphaConfig(mode="fixed", fixed_beta_l=0.0)
    coefficients = _response_parameters(off)
    value_off, _ = completeness_loglike_for_data(
        latent_alpha_config=off,
        latent_alpha_parameters=coefficients,
        **common,
    )
    value_off_cached, _ = completeness_loglike_for_data(
        latent_alpha_config=off,
        latent_alpha_parameters=coefficients,
        **common,
    )
    assert value_off_cached == pytest.approx(value_off, rel=0.0, abs=0.0)
    common["agn_data"].pop("_latent_alpha_compact_draw_cache", None)
    value_fixed, _ = completeness_loglike_for_data(
        latent_alpha_config=fixed_zero,
        latent_alpha_parameters=coefficients,
        **common,
    )
    assert value_off == pytest.approx(value_fixed, rel=0.0, abs=0.0)


def test_joint_likelihood_uses_aligned_magnitude_slope_host_covariance():
    config = LatentAlphaConfig.for_lf(
        lf_model="wang2026_type1_lade_a",
        mode="joint",
        include_magnitude_interactions=True,
    )
    parameters = _response_parameters(config)
    common = dict(
        completeness_params=_synthetic_completeness_params(),
        m_obs=np.array([19.1, 21.0, 23.1]),
        m_obs_err=np.full(3, 0.08),
        m_model=np.array([19.3, 20.8, 22.7]),
        mu_err=np.array([0.32, 0.45, 0.38]),
        z=np.array([0.7, 1.7, 3.0]),
        latent_alpha_config=config,
        latent_alpha_parameters=parameters,
        latent_alpha_distance_modulus=np.array([42.4, 44.1, 46.0]),
    )
    aligned = _joint_draw_data()
    aligned_value, _ = completeness_loglike_for_data(
        agn_data=aligned, **common
    )

    # Preserve every marginal draw distribution but destroy its joint
    # posterior indexing relative to host and intrinsic slope.
    misaligned = _joint_draw_data()
    misaligned["m_2500_attenuated_model_draws"] = np.roll(
        misaligned["m_2500_attenuated_model_draws"], 17, axis=1
    )
    misaligned_value, _ = completeness_loglike_for_data(
        agn_data=misaligned, **common
    )
    assert abs(aligned_value - misaligned_value) > 1.0e-6


def test_beta_off_changes_ordinary_3d_likelihood_only_by_cosmology_constant():
    completeness_params = _synthetic_completeness_params()
    data = _joint_draw_data()
    ordinary_data = dict(data)
    ordinary_data["f_host_2500_psf"] = data[
        "f_host_2500_psf_draws"
    ][:, deterministic_joint_draw_indices()]
    config = LatentAlphaConfig(mode="off")
    parameters = _response_parameters(config)
    common = dict(
        completeness_params=completeness_params,
        m_obs=np.array([19.1, 21.0, 23.1]),
        m_obs_err=np.full(3, 0.08),
        mu_err=np.array([0.32, 0.45, 0.38]),
        z=np.array([0.7, 1.7, 3.0]),
    )
    ordinary_a, _ = completeness_loglike_for_data(
        agn_data=ordinary_data,
        m_model=np.array([19.3, 20.8, 22.7]),
        **common,
    )
    ordinary_b, _ = completeness_loglike_for_data(
        agn_data=ordinary_data,
        m_model=np.array([19.6, 21.2, 23.0]),
        **common,
    )
    latent_a, _ = completeness_loglike_for_data(
        agn_data=data,
        m_model=np.array([19.3, 20.8, 22.7]),
        latent_alpha_config=config,
        latent_alpha_parameters=parameters,
        latent_alpha_distance_modulus=np.array([42.4, 44.1, 46.0]),
        **common,
    )
    latent_b, _ = completeness_loglike_for_data(
        agn_data=data,
        m_model=np.array([19.6, 21.2, 23.0]),
        latent_alpha_config=config,
        latent_alpha_parameters=parameters,
        # Deliberately change the cosmological distance modulus.  With beta
        # off it cannot affect the slope parent or response normalization.
        latent_alpha_distance_modulus=np.array([42.7, 44.5, 46.3]),
        **common,
    )
    assert (latent_a - latent_b) == pytest.approx(
        ordinary_a - ordinary_b, rel=2.0e-13, abs=2.0e-13
    )


def _args(**updates):
    defaults = dict(
        completeness_mode=LATENT_ALPHA_COMPLETENESS_MODE,
        disable_completeness=False,
        only_sna=False,
        fit_alpha_lambda_term=False,
        agn_calibrators=None,
        completeness_alpha_luminosity_mode="off",
        completeness_alpha_parent_beta_l=None,
        completeness_alpha_parent_beta_l_prior=(-0.5, 0.5),
        completeness_alpha_parent_mean=-0.5,
        completeness_alpha_parent_sigma=0.3,
        completeness_alpha_parent_logl_pivot=45.5,
        completeness_alpha_magnitude_interaction=False,
        completeness_lf_model="wang2026_type1_lade_a",
        completeness_magnitude="attenuated",
        z_range=(0.44, 3.16),
    )
    defaults.update(updates)
    return SimpleNamespace(**defaults)


def test_cli_config_rejects_lf_magnitude_and_unsupported_feature_mismatches():
    assert build_latent_alpha_config_from_args(_args()).luminosity_state == "attenuated"
    with pytest.raises(ValueError, match="requires attenuated"):
        build_latent_alpha_config_from_args(
            _args(completeness_magnitude="dereddened")
        )
    with pytest.raises(ValueError, match="fit_alpha_lambda_term"):
        build_latent_alpha_config_from_args(_args(fit_alpha_lambda_term=True))
    with pytest.raises(ValueError, match="requires --completeness-alpha-parent-beta-l"):
        build_latent_alpha_config_from_args(
            _args(completeness_alpha_luminosity_mode="fixed")
        )


def test_loaded_v3_latent_preflight_requires_complete_foreground_corrected_draws():
    frame = _valid_v3_frame()
    validate_loaded_spectra_catalog_compatibility(
        frame,
        completeness_enabled=True,
        completeness_mode=LATENT_ALPHA_COMPLETENESS_MODE,
        approximate_v1_fhost_2500_psf=False,
    )

    wrong_format = frame.copy()
    wrong_format["qvc_spectra_catalog_format"] = "qvc_spectra_catalog_v2"
    with pytest.raises(ValueError, match="requires only.*v3"):
        validate_loaded_spectra_catalog_compatibility(
            wrong_format,
            completeness_enabled=True,
            completeness_mode=LATENT_ALPHA_COMPLETENESS_MODE,
            approximate_v1_fhost_2500_psf=False,
        )

    with pytest.raises(ValueError, match="missing latent-alpha"):
        validate_loaded_spectra_catalog_compatibility(
            frame.drop(columns=["a_2500_total_draws"]),
            completeness_enabled=True,
            completeness_mode=LATENT_ALPHA_COMPLETENESS_MODE,
            approximate_v1_fhost_2500_psf=False,
        )

    short = frame.copy()
    short["joint_posterior_valid_count"] = [63, 64]
    with pytest.raises(ValueError, match="exactly 64"):
        validate_loaded_spectra_catalog_compatibility(
            short,
            completeness_enabled=True,
            completeness_mode=LATENT_ALPHA_COMPLETENESS_MODE,
            approximate_v1_fhost_2500_psf=False,
        )

    malformed = frame.copy()
    malformed["alpha_nu_intrinsic_1450_2500_draws"] = [
        np.full(63, -0.5),
        np.full(63, -0.5),
    ]
    with pytest.raises(ValueError, match=r"expected \(2, 64\)"):
        validate_loaded_spectra_catalog_compatibility(
            malformed,
            completeness_enabled=True,
            completeness_mode=LATENT_ALPHA_COMPLETENESS_MODE,
            approximate_v1_fhost_2500_psf=False,
        )

    nonfinite = frame.copy()
    nonfinite_alpha = np.stack(
        nonfinite["alpha_nu_intrinsic_1450_2500_draws"]
    )
    nonfinite_alpha[0, 4] = np.nan
    nonfinite["alpha_nu_intrinsic_1450_2500_draws"] = list(nonfinite_alpha)
    with pytest.raises(ValueError, match="must be finite"):
        validate_loaded_spectra_catalog_compatibility(
            nonfinite,
            completeness_enabled=True,
            completeness_mode=LATENT_ALPHA_COMPLETENESS_MODE,
            approximate_v1_fhost_2500_psf=False,
        )

    foreground = frame.copy()
    foreground.loc[0, "mw_deredden_applied"] = False
    with pytest.raises(ValueError, match="mw_deredden_applied=True"):
        validate_loaded_spectra_catalog_compatibility(
            foreground,
            completeness_enabled=True,
            completeness_mode=LATENT_ALPHA_COMPLETENESS_MODE,
            approximate_v1_fhost_2500_psf=False,
        )


def test_programmatic_run_entries_enforce_lf_state_and_calibrator_guards():
    config = LatentAlphaConfig.for_lf(
        lf_model="wang2026_type1_lade_a",
        requested_luminosity_state="attenuated",
    )
    empty = pd.DataFrame()
    with pytest.raises(ValueError, match="requires 'attenuated'"):
        run_single(
            empty,
            empty,
            empty,
            None,
            None,
            None,
            "FlatLambdaCDM",
            completeness_mode=LATENT_ALPHA_COMPLETENESS_MODE,
            completeness_magnitude="dereddened",
            latent_alpha_config=config,
        )
    with pytest.raises(ValueError, match="does not yet support AGN calibrators"):
        run_single(
            empty,
            empty,
            empty,
            None,
            None,
            None,
            "FlatLambdaCDM",
            completeness_mode=LATENT_ALPHA_COMPLETENESS_MODE,
            completeness_magnitude="attenuated",
            df_calibrators=pd.DataFrame(),
            latent_alpha_config=config,
        )
    with pytest.raises(ValueError, match="requires 'attenuated'"):
        run_single_jax(
            empty,
            empty,
            empty,
            None,
            None,
            None,
            completeness_mode=LATENT_ALPHA_COMPLETENESS_MODE,
            completeness_magnitude="dereddened",
            latent_alpha_config=config,
        )


def test_latent_mock_semantics_and_runner_wiring_are_explicit(tmp_path):
    config = LatentAlphaConfig.for_lf(
        lf_model="wang2026_type1_lade_a",
        requested_luminosity_state="attenuated",
    )
    mock_path = tmp_path / "mock.h5"
    with h5py.File(mock_path, "w") as handle:
        handle.attrs["lf_model"] = config.lf_model
        handle.attrs["completeness_magnitude_state"] = "attenuated"
        handle.attrs["completeness_mock_schema_version"] = (
            COMPLETENESS_MOCK_SCHEMA_VERSION
        )
        handle.attrs["lf_semantics_version"] = (
            COMPLETENESS_MOCK_SEMANTICS_VERSION
        )
    validate_latent_alpha_mock_semantics(mock_path, config)
    with h5py.File(mock_path, "r+") as handle:
        handle.attrs["completeness_magnitude_state"] = "dereddened"
    with pytest.raises(ValueError, match="mock magnitude state"):
        validate_latent_alpha_mock_semantics(mock_path, config)

    runner = (Path(__file__).resolve().parents[1] / "run_hubble.xonsh").read_text(
        encoding="utf-8"
    )
    for variable in (
        "QVC_HUBBLE_COMPLETENESS_ALPHA_PARENT_MEAN",
        "QVC_HUBBLE_COMPLETENESS_ALPHA_PARENT_SIGMA",
        "QVC_HUBBLE_COMPLETENESS_ALPHA_LUMINOSITY_MODE",
        "QVC_HUBBLE_COMPLETENESS_ALPHA_PARENT_BETA_L",
        "QVC_HUBBLE_COMPLETENESS_ALPHA_PARENT_BETA_L_PRIOR",
        "QVC_HUBBLE_COMPLETENESS_ALPHA_PARENT_LOGL_PIVOT",
        "QVC_HUBBLE_COMPLETENESS_ALPHA_MAGNITUDE_INTERACTION",
    ):
        assert variable in runner


def test_joint_mode_adds_only_named_beta_and_response_parameters():
    _, off_labels, _ = get_model_params(
        "FlatLambdaCDM",
        use_latent_alpha_completeness=True,
        latent_alpha_luminosity_mode="off",
    )
    _, fixed_labels, _ = get_model_params(
        "FlatLambdaCDM",
        use_latent_alpha_completeness=True,
        latent_alpha_luminosity_mode="fixed",
    )
    _, joint_labels, _ = get_model_params(
        "FlatLambdaCDM",
        use_latent_alpha_completeness=True,
        latent_alpha_luminosity_mode="joint",
    )
    assert off_labels == fixed_labels
    assert BETA_ALPHA_L_PARAMETER not in off_labels
    assert joint_labels.count(BETA_ALPHA_L_PARAMETER) == 1
    assert set(response_coefficient_names()).issubset(joint_labels)
    assert len(joint_labels) == len(off_labels) + 1


@pytest.mark.parametrize(
    ("mode", "magnitude_interaction"),
    [("off", False), ("fixed", True), ("joint", False), ("joint", True)],
)
def test_standardization_plot_view_removes_latent_columns_by_name(
    mode, magnitude_interaction
):
    config = LatentAlphaConfig(
        mode=mode,
        fixed_beta_l=0.0 if mode == "fixed" else None,
        include_magnitude_interactions=magnitude_interaction,
    )
    _, base_labels, _ = get_model_params("Flatw0waCDM")
    _, latent_labels, _ = get_model_params(
        "Flatw0waCDM",
        use_latent_alpha_completeness=True,
        latent_alpha_luminosity_mode=mode,
        latent_alpha_magnitude_interaction=magnitude_interaction,
    )
    samples = np.arange(3 * len(latent_labels), dtype=float).reshape(
        3, len(latent_labels)
    )
    view, view_labels = standardization_plot_posterior_view(
        samples,
        latent_labels,
        latent_alpha_config=config,
    )
    assert view_labels == base_labels
    base_indices = [latent_labels.index(label) for label in base_labels]
    np.testing.assert_array_equal(view, samples[:, base_indices])


def test_standardization_plot_view_rejects_label_config_mismatch():
    config = LatentAlphaConfig(mode="joint")
    _, labels, _ = get_model_params(
        "FlatLambdaCDM",
        use_latent_alpha_completeness=True,
        latent_alpha_luminosity_mode="joint",
    )
    samples = np.zeros((2, len(labels) - 1))
    with pytest.raises(ValueError, match="misaligned"):
        standardization_plot_posterior_view(
            samples,
            labels,
            latent_alpha_config=config,
        )

    samples = np.zeros((2, len(labels)))
    labels = [
        label for label in labels
        if label != response_coefficient_names()[0]
    ] + ["unrelated_extra_parameter"]
    with pytest.raises(ValueError, match="authoritative configuration"):
        standardization_plot_posterior_view(
            samples,
            labels,
            latent_alpha_config=config,
        )
