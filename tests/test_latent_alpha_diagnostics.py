import json

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from qvc.hubble.latent_alpha_completeness import (  # noqa: E402
    BETA_ALPHA_L_PARAMETER,
    LatentAlphaConfig,
    response_coefficient_names,
)
from qvc.hubble.latent_alpha_diagnostics import (  # noqa: E402
    LATENT_ALPHA_DIAGNOSTICS_SCHEMA_VERSION,
    _evaluate_c3,
    write_latent_alpha_diagnostics,
)


def test_diagnostic_c3_lookup_extends_edge_planes_only_inside_hard_support():
    from qvc.hubble.hubble_completeness_refactored import Completeness3D

    magnitude_centers = np.array([18.6, 21.25, 23.9])
    redshift_centers = np.array([0.5, 2.0])
    host_centers = np.array([0.1, 0.9])
    cube = np.broadcast_to(
        np.array([0.2, 0.5, 0.8])[:, None, None], (3, 2, 2)
    )
    model = Completeness3D(
        magnitude_centers,
        redshift_centers,
        host_centers,
        cube,
        magnitude_support=(18.5, 24.0),
    )

    actual = _evaluate_c3(
        model,
        np.array([18.49, 18.55, 23.95, 24.01]),
        np.full(4, 1.0),
        np.full(4, 0.5),
        None,
    )
    np.testing.assert_allclose(actual, [0.0, 0.2, 0.8, 0.0])


def _frame(n_objects=10):
    draw_coordinate = np.linspace(-1.0, 1.0, 64)
    redshift = np.linspace(0.5, 3.0, n_objects)
    intrinsic = np.stack(
        [
            -0.62 + 0.08 * z + 0.035 * draw_coordinate
            for z in redshift
        ]
    )
    a_galaxy = np.broadcast_to(
        0.08 + 0.01 * draw_coordinate, (n_objects, 64)
    ).copy()
    a_internal = np.broadcast_to(
        0.12 + 0.015 * draw_coordinate, (n_objects, 64)
    ).copy()
    a_total = a_galaxy + a_internal
    attenuated = intrinsic - 0.35 * a_total
    m_dereddened = np.stack(
        [20.0 + 0.7 * z + 0.03 * draw_coordinate for z in redshift]
    )
    m_attenuated = m_dereddened + a_total
    f_host = np.stack(
        [
            np.clip(0.28 - 0.04 * z + 0.015 * draw_coordinate, 0.0, 1.0)
            for z in redshift
        ]
    )

    arrays = {
        "f_host_2500_psf_draws": f_host,
        "alpha_nu_intrinsic_1450_2500_draws": intrinsic,
        "alpha_nu_attenuated_1450_2500_draws": attenuated,
        "m_2500_dereddened_draws": m_dereddened,
        "m_2500_attenuated_model_draws": m_attenuated,
        "a_2500_galaxy_draws": a_galaxy,
        "a_2500_internal_draws": a_internal,
        "a_2500_total_draws": a_total,
    }
    frame = pd.DataFrame(
        {
            "z": redshift,
            "joint_posterior_valid_count": np.full(n_objects, 64),
        }
    )
    for name, values in arrays.items():
        frame[name] = pd.Series(list(values), dtype=object)
    return frame


def _completeness(magnitude, redshift, f_host, *, scale=1.0):
    magnitude, redshift, f_host = np.broadcast_arrays(
        np.asarray(magnitude, dtype=float),
        np.asarray(redshift, dtype=float),
        np.asarray(f_host, dtype=float),
    )
    logit = scale * (2.8 - 0.18 * (magnitude - 20.0) - 0.25 * redshift - f_host)
    return 1.0 / (1.0 + np.exp(-logit))


def _parameters(config, *, first_linear=0.45):
    parameters = {
        name: 0.0
        for name in response_coefficient_names(
            config.include_magnitude_interactions
        )
    }
    parameters["alpha_sel_z_p0_linear"] = first_linear
    if config.mode == "joint":
        parameters[BETA_ALPHA_L_PARAMETER] = 0.18
    return parameters


def test_writes_complete_bundle_and_preserves_normalization(tmp_path):
    frame = _frame()
    config = LatentAlphaConfig.for_lf(
        lf_model="wang2026_type1_lade_a",
        mode="fixed",
        fixed_beta_l=0.12,
    )
    result = write_latent_alpha_diagnostics(
        frame,
        config=config,
        completeness_model=_completeness,
        completeness_kwargs={"scale": 0.9},
        output_dir=tmp_path,
        parameters=_parameters(config),
        distance_modulus=np.linspace(42.5, 46.0, len(frame)),
        plot_format="png",
    )

    assert result.json_path.exists()
    assert len(result.plot_paths) == 6
    assert all(path.exists() and path.stat().st_size > 0 for path in result.plot_paths)
    payload = json.loads(result.json_path.read_text())
    assert payload["schema_version"] == LATENT_ALPHA_DIAGNOSTICS_SCHEMA_VERSION
    assert payload["inputs"]["luminosity_state"] == "attenuated"
    assert (
        payload["inputs"]["alpha_state_column"]
        == "alpha_nu_intrinsic_1450_2500_draws"
    )
    assert payload["inputs"]["latent_slope_state"] == "intrinsic_disk_only"
    assert payload["inputs"]["lf_luminosity_state"] == "attenuated"
    assert (
        payload["inputs"]["magnitude_state_column"]
        == "m_2500_attenuated_model_draws"
    )
    assert payload["inputs"]["all_64_used_for_diagnostics"] is True
    assert payload["inputs"]["joint_magnitude_host_slope_covariance_used"] is True
    assert payload["inputs"]["joint_draw_likelihood_count"] == 16
    assert payload["parent_vs_selected"]["derived_slope_prior_correction"] == "none"
    assert (
        payload["normalization_and_saturation"]["maximum_absolute_residual"]
        < 2.0e-12
    )
    assert 0.0 < payload["inverse_weight_ess"]["ess"] <= len(frame)
    assert payload["beta_cosmology_correlations"]["available"] is False


def test_joint_posterior_median_and_beta_cosmology_correlations(tmp_path):
    frame = _frame(7)
    config = LatentAlphaConfig.for_lf(
        lf_model="shen",
        shen_lf_mode="type1_intrinsic",
        mode="joint",
        include_magnitude_interactions=True,
    )
    rng = np.random.default_rng(17)
    beta = np.linspace(-0.2, 0.3, 80)
    posterior = {
        name: rng.normal(0.0, 0.08, beta.size)
        for name in response_coefficient_names(True)
    }
    posterior.update(
        {
            BETA_ALPHA_L_PARAMETER: beta,
            "H0": 70.0 + 4.0 * beta,
            "Om0": 0.3 - 0.1 * beta + rng.normal(0.0, 0.005, beta.size),
            "M0_agn": rng.normal(-22.0, 0.1, beta.size),
        }
    )
    result = write_latent_alpha_diagnostics(
        frame,
        config=config,
        completeness_model=_completeness,
        output_dir=tmp_path,
        posterior_samples=posterior,
        distance_modulus=np.linspace(42.0, 46.0, len(frame)),
        plot_format="png",
        filename_prefix="joint",
    )

    summary = result.summary
    assert summary["inputs"]["luminosity_state"] == "dereddened"
    assert (
        summary["inputs"]["magnitude_state_column"]
        == "m_2500_dereddened_draws"
    )
    correlations = summary["beta_cosmology_correlations"]
    assert correlations["available"] is True
    assert correlations["posterior_draw_count"] == beta.size
    assert correlations["correlations"]["H0"] == pytest.approx(1.0)
    assert "Om0" in correlations["correlations"]
    assert "M0_agn" not in correlations["correlations"]
    assert len(result.plot_paths) == 7
    assert any("beta_cosmology_correlations" in path.name for path in result.plot_paths)
    assert summary["representative_parameters"][BETA_ALPHA_L_PARAMETER] == pytest.approx(
        np.median(beta)
    )


def test_array_posterior_requires_labels_and_can_supply_parameters(tmp_path):
    frame = _frame(4)
    config = LatentAlphaConfig.for_lf(
        lf_model="kulkarni2019_type1_model2", mode="off"
    )
    labels = list(response_coefficient_names(False)) + ["H0"]
    samples = np.zeros((12, len(labels)))
    samples[:, -1] = np.linspace(68.0, 72.0, len(samples))

    with pytest.raises(ValueError, match="model_labels is required"):
        write_latent_alpha_diagnostics(
            frame,
            config=config,
            completeness_model=_completeness,
            output_dir=tmp_path,
            posterior_samples=samples,
            log_luminosity=np.linspace(45.0, 46.0, len(frame)),
            plot_format="png",
        )

    result = write_latent_alpha_diagnostics(
        frame,
        config=config,
        completeness_model=_completeness,
        output_dir=tmp_path,
        posterior_samples=samples,
        model_labels=labels,
        log_luminosity=np.linspace(45.0, 46.0, len(frame)),
        plot_format="png",
        filename_prefix="array",
    )
    assert result.summary["luminosity_dependence"]["beta_alpha_L"] == 0.0


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda frame: frame.drop(columns=["a_2500_total_draws"]),
            "Missing required aligned v3 draw column",
        ),
        (
            lambda frame: frame.assign(joint_posterior_valid_count=63),
            "require 64 valid aligned",
        ),
        (
            lambda frame: frame.assign(
                f_host_2500_psf_draws=pd.Series(
                    [np.zeros(63) for _ in range(len(frame))], dtype=object
                )
            ),
            "expected",
        ),
        (
            lambda frame: frame.assign(
                f_host_2500_psf_draws=pd.Series(
                    [np.full(64, 1.1) for _ in range(len(frame))], dtype=object
                )
            ),
            "must lie in",
        ),
    ],
)
def test_rejects_malformed_v3_draws(tmp_path, mutation, match):
    config = LatentAlphaConfig.for_lf(
        lf_model="palanque2016_ple_lede", mode="off"
    )
    with pytest.raises((KeyError, ValueError), match=match):
        write_latent_alpha_diagnostics(
            mutation(_frame(3)),
            config=config,
            completeness_model=_completeness,
            output_dir=tmp_path,
            parameters=_parameters(config),
            log_luminosity=np.array([45.0, 45.5, 46.0]),
            plot_format="png",
        )


def test_rejects_missing_luminosity_and_unknown_completeness_output(tmp_path):
    frame = _frame(3)
    config = LatentAlphaConfig.for_lf(
        lf_model="wang2026_type1_lade_a", mode="off"
    )
    with pytest.raises(ValueError, match="Provide log_luminosity"):
        write_latent_alpha_diagnostics(
            frame,
            config=config,
            completeness_model=_completeness,
            output_dir=tmp_path,
            parameters=_parameters(config),
            plot_format="png",
        )

    def invalid_completeness(magnitude, redshift, f_host):
        return np.full(np.broadcast_shapes(np.shape(magnitude), np.shape(f_host)), 1.2)

    with pytest.raises(ValueError, match=r"values in \[0, 1\]"):
        write_latent_alpha_diagnostics(
            frame,
            config=config,
            completeness_model=invalid_completeness,
            output_dir=tmp_path,
            parameters=_parameters(config),
            log_luminosity=np.array([45.0, 45.5, 46.0]),
            plot_format="png",
        )
