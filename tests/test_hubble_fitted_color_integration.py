import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import qvc.hubble.hubble_fit as hubble_fit_module
from qvc.hubble.fitted_color_completeness import (
    COLOR_STRENGTH_PARAMETER,
    FittedColorConfig,
    QsogenColorParentCache,
    write_qsogen_color_parent_cache,
)
from qvc.hubble.hubble_completeness_refactored import (
    COMPLETENESS_FHOST_COL,
    Completeness2D,
    Completeness3D,
)
from qvc.hubble.hubble_fit import (
    _validate_fitted_color_checkpoint_config,
    build_fitted_color_config_from_args,
    prepare_fitted_color_posterior_draws,
    validate_fitted_color_runtime_semantics,
    validate_fitted_color_v3_frame,
    write_fitted_color_run_diagnostics,
)
from qvc.hubble.hubble_fit_jax import (
    _completeness_loglike_jax,
    _prepare_completeness_for_jax,
)
from qvc.hubble.hubble_likelihood import completeness_loglike_for_data
from qvc.hubble.hubble_model import get_model_params


def _parent_provenance():
    return {
        "construction": "synthetic_hubble_integration_test",
        "qsogen_commit": "d2f9abf1ad23c489da8857f7e3c1bca862105d22",
        "qsogen_source_url": "https://github.com/MJTemple/qsogen",
        "qsogen_license": "MIT",
        "jaxsedfit_filter_commit": "bc9da74735260bd33b3da2076fd7929fdd592e0d",
        "asset_sha256": {"synthetic": "0" * 64},
        "filter_names": ["g_sdss", "i_sdss"],
        "reference_cosmology": {"H0_km_s_Mpc": 70.0, "Omega_m": 0.3},
        "magnitude_state": "attenuation_retaining",
        "source_internal_attenuation": "synthetic",
        "host_scaling": "synthetic",
        "m2500_to_qsogen_luminosity": "synthetic",
        "parent_population_interpretation": "synthetic sensitivity parent",
        "residual_scatter_limitation": "synthetic symmetric scatter",
    }


def _write_parent_cache(path):
    magnitude = np.array([18.5, 21.25, 24.0])
    redshift = np.array([0.44, 1.80, 3.16])
    f_host = np.linspace(0.0, 1.0, 501)
    agn = np.zeros((len(magnitude), len(redshift)))
    total = np.zeros((len(magnitude), len(redshift), len(f_host)))
    cache = QsogenColorParentCache(
        magnitude,
        redshift,
        f_host,
        agn,
        total,
        _parent_provenance(),
    )
    write_qsogen_color_parent_cache(path, cache)
    return FittedColorConfig.from_parent_file(path)


def _completeness_params(mode):
    magnitude = np.array([18.6, 20.0, 22.0, 23.9])
    redshift = np.array([0.5, 1.8, 3.2])
    mm, zz = np.meshgrid(magnitude, redshift, indexing="ij")
    if mode == "2d":
        cube = 1.0 / (1.0 + np.exp(-(2.0 - 0.45 * (mm - 20.0) - 0.2 * zz)))
        model = Completeness2D(
            magnitude,
            redshift,
            cube,
            magnitude_support=(18.5, 24.0),
        )
        return (model, magnitude, redshift)
    f_host = np.array([0.1, 0.5, 0.9])
    mm, zz, ff = np.meshgrid(magnitude, redshift, f_host, indexing="ij")
    cube = 1.0 / (
        1.0
        + np.exp(
            -(2.0 - 0.45 * (mm - 20.0) - 0.2 * zz - 1.15 * ff)
        )
    )
    model = Completeness3D(
        magnitude,
        redshift,
        f_host,
        cube,
        magnitude_support=(18.5, 24.0),
    )
    return (model, magnitude, redshift, f_host)


def _selection_inputs(*, q=None, all_unsupported=False):
    n_object, n_draw = 2, 16
    magnitude = np.vstack(
        (
            np.linspace(19.1, 19.7, n_draw),
            np.linspace(22.0, 22.6, n_draw),
        )
    )
    in_support = np.ones_like(magnitude, dtype=bool)
    if all_unsupported:
        magnitude[:] = 24.2
        in_support[:] = False
    else:
        magnitude[0, 0] = 18.4
        magnitude[1, -1] = 24.2
        in_support[0, 0] = False
        in_support[1, -1] = False
    if q is None:
        q = np.vstack(
            (np.linspace(0.05, 0.45, n_draw), np.linspace(0.55, 0.95, n_draw))
        )
    host_draws = np.vstack(
        (np.linspace(0.15, 0.45, n_draw), np.linspace(0.55, 0.85, n_draw))
    )
    agn_data = {
        COMPLETENESS_FHOST_COL: np.array([0.30, 0.70]),
        "fitted_color_parent_percentile_draws": np.asarray(q, dtype=float),
        "fitted_color_magnitude_draws": magnitude,
        "fitted_color_fhost_draws": host_draws,
        "fitted_color_in_support_draws": in_support,
    }
    common = {
        "m_obs": np.array([19.4, 22.3]),
        "m_obs_err": np.array([0.08, 0.09]),
        "m_model": np.array([19.5, 22.1]),
        "mu_err": np.array([0.32, 0.40]),
        "z": np.array([0.8, 2.6]),
    }
    return agn_data, common


def _numpy_selection_value(params, agn_data, common, config=None, strength=0.0):
    kwargs = {}
    if config is not None:
        kwargs.update(
            fitted_color_config=config,
            fitted_color_parameters={COLOR_STRENGTH_PARAMETER: strength},
        )
    return completeness_loglike_for_data(
        completeness_params=params,
        agn_data=agn_data,
        **common,
        **kwargs,
    )[0]


def test_color_parameter_is_one_uniform_parameter_after_log_f():
    priors, labels, _ = get_model_params(
        "FlatLambdaCDM", use_fitted_color_completeness=True
    )
    assert priors[COLOR_STRENGTH_PARAMETER] == (-1.0, 1.0)
    assert labels.count(COLOR_STRENGTH_PARAMETER) == 1
    assert labels.index(COLOR_STRENGTH_PARAMETER) == labels.index("log_f") + 1


@pytest.mark.parametrize("mode", ["2d", "3d_fhost"])
def test_s_color_zero_is_exact_baseline_with_unsupported_draws(mode):
    config = FittedColorConfig("unused.h5", "0" * 64)
    params = _completeness_params(mode)
    agn_data, common = _selection_inputs()
    baseline = _numpy_selection_value(params, agn_data, common)
    fitted = _numpy_selection_value(params, agn_data, common, config, 0.0)
    assert fitted == pytest.approx(baseline, abs=1e-14)

    all_unsupported, common = _selection_inputs(all_unsupported=True)
    baseline = _numpy_selection_value(params, all_unsupported, common)
    fitted = _numpy_selection_value(
        params, all_unsupported, common, config, 1.0
    )
    assert fitted == pytest.approx(baseline, abs=1e-14)


def test_positive_s_color_favors_blue_and_penalizes_red_draws():
    config = FittedColorConfig("unused.h5", "0" * 64)
    params = _completeness_params("2d")
    blue, common = _selection_inputs(q=np.full((2, 16), 0.1))
    red, _ = _selection_inputs(q=np.full((2, 16), 0.9))
    baseline = _numpy_selection_value(params, blue, common)
    blue_value = _numpy_selection_value(params, blue, common, config, 0.8)
    red_value = _numpy_selection_value(params, red, common, config, 0.8)
    # This function returns log-denominator minus log-selected-color-factor.
    assert blue_value < baseline < red_value


def test_in_support_zero_base_is_impossible_and_unit_base_is_neutral():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    config = FittedColorConfig("unused.h5", "0" * 64)
    agn_data, common = _selection_inputs()
    magnitude = np.array([18.6, 20.0, 22.0, 23.9])
    redshift = np.array([0.5, 1.8, 3.2])
    for level in (0.0, 1.0):
        model = Completeness2D(
            magnitude,
            redshift,
            np.full((4, 3), level),
            magnitude_support=(18.5, 24.0),
        )
        params = (model, magnitude, redshift)
        numpy_value = _numpy_selection_value(
            params, agn_data, common, config, 0.9
        )
        prepared = _prepare_completeness_for_jax(
            params, selection_magnitude=common["m_obs"]
        )
        jax_value = _completeness_loglike_jax(
            jnp.asarray(common["m_model"]),
            jnp.asarray(common["mu_err"]),
            jnp.asarray(common["z"]),
            prepared,
            jnp.asarray(agn_data[COMPLETENESS_FHOST_COL]),
            None,
            fitted_color_config=config,
            fitted_color_percentile_draws=jnp.asarray(
                agn_data["fitted_color_parent_percentile_draws"]
            ),
            fitted_color_magnitude_draws=jnp.asarray(
                agn_data["fitted_color_magnitude_draws"]
            ),
            fitted_color_fhost_draws=jnp.asarray(
                agn_data["fitted_color_fhost_draws"]
            ),
            fitted_color_in_support_draws=jnp.asarray(
                agn_data["fitted_color_in_support_draws"]
            ),
            params={COLOR_STRENGTH_PARAMETER: jnp.asarray(0.9)},
        )
        if level == 0.0:
            assert np.isposinf(numpy_value)
            assert np.isposinf(float(jax_value))
        else:
            baseline = _numpy_selection_value(params, agn_data, common)
            assert numpy_value == pytest.approx(baseline, abs=1e-14)
            np.testing.assert_allclose(float(jax_value), baseline, atol=1e-13)


@pytest.mark.parametrize("mode", ["2d", "3d_fhost"])
@pytest.mark.parametrize("strength", [0.0, 0.63])
def test_numpy_and_jax_color_likelihoods_match(mode, strength):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    config = FittedColorConfig("unused.h5", "0" * 64)
    params = _completeness_params(mode)
    agn_data, common = _selection_inputs()
    expected = _numpy_selection_value(
        params, agn_data, common, config, strength
    )
    prepared = _prepare_completeness_for_jax(
        params, selection_magnitude=common["m_obs"]
    )
    actual = _completeness_loglike_jax(
        jnp.asarray(common["m_model"]),
        jnp.asarray(common["mu_err"]),
        jnp.asarray(common["z"]),
        prepared,
        jnp.asarray(agn_data[COMPLETENESS_FHOST_COL]),
        None,
        fitted_color_config=config,
        fitted_color_percentile_draws=jnp.asarray(
            agn_data["fitted_color_parent_percentile_draws"]
        ),
        fitted_color_magnitude_draws=jnp.asarray(
            agn_data["fitted_color_magnitude_draws"]
        ),
        fitted_color_fhost_draws=jnp.asarray(
            agn_data["fitted_color_fhost_draws"]
        ),
        fitted_color_in_support_draws=jnp.asarray(
            agn_data["fitted_color_in_support_draws"]
        ),
        params={COLOR_STRENGTH_PARAMETER: jnp.asarray(strength)},
    )
    np.testing.assert_allclose(float(actual), expected, rtol=2e-10, atol=2e-10)

    if mode == "3d_fhost" and strength == 0.0:
        ordinary = _completeness_loglike_jax(
            jnp.asarray(common["m_model"]),
            jnp.asarray(common["mu_err"]),
            jnp.asarray(common["z"]),
            prepared,
            jnp.asarray(agn_data[COMPLETENESS_FHOST_COL]),
            None,
        )
        np.testing.assert_allclose(float(actual), float(ordinary), atol=1e-13)


def _valid_color_v3_frame(n_objects=2):
    provenance = json.dumps(
        {
            "prediction_source": "synthetic_test",
            "jaxsedfit_git_commit": "bc9da74735260bd33b3da2076fd7929fdd592e0d",
        },
        sort_keys=True,
    )
    phase = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    frame = pd.DataFrame(
        {
            "qvc_spectra_catalog_format": ["qvc_spectra_catalog_v3"] * n_objects,
            "fit_ok": np.ones(n_objects, dtype=bool),
            "z": np.linspace(0.8, 2.0, n_objects),
            "joint_posterior_valid_count": np.full(n_objects, 64),
            "joint_psf_photometry_valid_count": np.full(n_objects, 64),
            "joint_posterior_index": [np.arange(64) for _ in range(n_objects)],
            "joint_psf_photometry_provenance_json": [provenance] * n_objects,
            "joint_psf_total_g_flux_mjy_draws": [
                np.ones(64) for _ in range(n_objects)
            ],
            "joint_psf_total_i_flux_mjy_draws": [
                np.ones(64) for _ in range(n_objects)
            ],
            "m_2500_attenuated_model_draws": [
                np.full(64, 21.0) for _ in range(n_objects)
            ],
        }
    )
    frame["f_host_2500_psf_draws"] = [
        np.clip(0.2 + 0.2 * index + 0.03 * np.sin(phase), 0.0, 1.0)
        for index in range(n_objects)
    ]
    return frame


def test_v3_extension_validation_and_preparation_feed_both_backends(tmp_path):
    config = _write_parent_cache(tmp_path / "parent.h5")
    frame = _valid_color_v3_frame()
    validate_fitted_color_v3_frame(frame)
    prepared = prepare_fitted_color_posterior_draws(
        frame,
        frame,
        config=config,
        completeness_mode="3d_fhost",
        z_range=(0.44, 3.16),
    )
    draw_columns = (
        "fitted_color_parent_percentile_draws",
        "fitted_color_magnitude_draws",
        "fitted_color_fhost_draws",
        "fitted_color_g_minus_i_draws",
        "fitted_color_in_support_draws",
    )
    numpy_package = {
        name: np.stack(prepared[name].to_numpy()) for name in draw_columns
    }
    assert all(value.shape == (2, 16) for value in numpy_package.values())
    jnp = pytest.importorskip("jax.numpy")
    jax_package = {name: jnp.asarray(value) for name, value in numpy_package.items()}
    assert all(tuple(value.shape) == (2, 16) for value in jax_package.values())
    np.testing.assert_allclose(
        numpy_package["fitted_color_parent_percentile_draws"], 0.5
    )

    malformed = frame.drop(columns="joint_psf_total_i_flux_mjy_draws")
    with pytest.raises(ValueError, match="missing fitted-color aligned fields"):
        validate_fitted_color_v3_frame(malformed)


def test_cli_semantics_lf_matching_warning_and_checkpoint_provenance(
    tmp_path, monkeypatch
):
    config = _write_parent_cache(tmp_path / "parent.h5")
    args = SimpleNamespace(
        completeness_color_model="qsogen_delta_gi",
        completeness_color_parent_file=str(tmp_path / "parent.h5"),
        completeness_color_parent_sigma=0.2,
        completeness_lf_model="wang2026_type1_lade_a",
        completeness_magnitude="attenuated",
    )
    built = build_fitted_color_config_from_args(args)
    assert built == config
    monkeypatch.setattr(
        hubble_fit_module, "_FITTED_COLOR_SCIENCE_WARNING_EMITTED", False
    )
    with pytest.warns(RuntimeWarning, match="SENSITIVITY WARNING") as warning_list:
        validate_fitted_color_runtime_semantics(
            built,
            completeness=True,
            completeness_mode="3d_fhost",
            completeness_magnitude="attenuated",
        )
        validate_fitted_color_runtime_semantics(
            built,
            completeness=True,
            completeness_mode="3d_fhost",
            completeness_magnitude="attenuated",
        )
    assert len(warning_list) == 1
    with pytest.raises(ValueError, match="only atop"):
        validate_fitted_color_runtime_semantics(
            built,
            completeness=True,
            completeness_mode="4d_fhost_alpha",
            completeness_magnitude="attenuated",
        )
    with pytest.raises(ValueError, match="alpha_lambda"):
        validate_fitted_color_runtime_semantics(
            built,
            completeness=True,
            completeness_mode="2d",
            completeness_magnitude="attenuated",
            use_alpha_lambda_term=True,
        )

    monkeypatch.setenv("QVC_HUBBLE_SHEN_LF_MODE", "type1_intrinsic")
    args.completeness_lf_model = "shen"
    with pytest.raises(ValueError, match="requires dereddened luminosity"):
        build_fitted_color_config_from_args(args)

    config_json = json.dumps(
        config.to_dict(), sort_keys=True, separators=(",", ":")
    )
    prediction_json = json.dumps(
        {"prediction_source": "test", "jaxsedfit_git_commit": "abc"},
        sort_keys=True,
    )
    checkpoint = {
        "fitted_color_config_json": np.asarray(config_json),
        "fitted_color_photometry_provenance_json": np.asarray(prediction_json),
    }
    _validate_fitted_color_checkpoint_config(
        checkpoint,
        checkpoint_file="synthetic.h5",
        expected_fitted_color_config=config,
        expected_photometry_provenance_json=prediction_json,
    )
    with pytest.raises(RuntimeError, match="prediction provenance"):
        _validate_fitted_color_checkpoint_config(
            checkpoint,
            checkpoint_file="synthetic.h5",
            expected_fitted_color_config=config,
            expected_photometry_provenance_json="different",
        )


def test_diagnostics_record_host_luminosity_relative_factor_and_boundary(tmp_path):
    config = FittedColorConfig("unused.h5", "0" * 64)
    agn_data, common = _selection_inputs()
    agn_data.update(
        z=common["z"],
        fitted_color_g_minus_i_draws=np.vstack(
            (np.linspace(-0.2, 0.1, 16), np.linspace(0.2, 0.6, 16))
        ),
    )
    output = write_fitted_color_run_diagnostics(
        agn_data=agn_data,
        completeness_params=_completeness_params("3d_fhost"),
        flat_samples=np.full((32, 1), 0.98),
        model_labels=[COLOR_STRENGTH_PARAMETER],
        config=config,
        plot_path=tmp_path,
    )
    assert output["summary_json"].is_file()
    assert output["response_plot"].is_file()
    summary = json.loads(output["summary_json"].read_text())
    assert summary["s_color_boundary_piled"] is True
    assert summary["n_objects_with_neutral_unsupported_draws"] == 2
    assert summary["median_host_fraction"] is not None
    assert summary["median_base_completeness"] is not None
    assert summary["median_relative_completeness"] is not None
    assert summary["diagnostic_luminosity"].startswith("log10_nu_Lnu_2500")
