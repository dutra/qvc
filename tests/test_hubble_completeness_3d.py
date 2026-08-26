import ast
import json
import os
import inspect
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from astropy.cosmology import FlatLambdaCDM
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble import cuts as hubble_cuts
from qvc.hubble import hubble_completeness_refactored as hcr
from qvc.hubble import hubble_fit, hubble_likelihood, hubble_model
from qvc.hubble.completeness_mock_catalog import (
    AB_ABSOLUTE_MAG_ZEROPOINT,
    FULL_SKY_AREA_DEG2,
    KULKARNI2019_TYPE1_MODEL1,
    KULKARNI2019_TYPE1_MODEL2,
    KULKARNI2019_TYPE1_MODEL3,
    LOG10_MAG_JACOBIAN,
    NU_2500_HZ,
    SHEN_DEFAULT_LF_MODE,
    SHEN_GLOBAL_FIT,
    SHEN_LF_MODES,
    _configure_shen_paths,
    _shen_type1_nh_bin_fractions,
    build_shen_lf,
    log_nu_lnu_to_ab_absolute_magnitude,
    normalize_shen_lf_mode,
    plan_area_scaled_mock_sampling,
    save_mock_catalog,
    shen_absorbed_fraction,
    shen_type1_fraction,
)
from qvc.hubble.hubble_likelihood import completeness_loglike


def test_completeness_loglike_requires_explicit_magnitude_support():
    parameter = inspect.signature(completeness_loglike).parameters.get(
        "magnitude_support"
    )

    assert parameter is not None
    assert parameter.default is inspect.Parameter.empty


def test_configure_shen_paths_overrides_checkout_config(tmp_path):
    shen_config = types.SimpleNamespace(
        homepath="/stale/quasarlf/pubtools/",
        datapath="/stale/quasarlf/pubtools/data/",
    )

    obdata_path = _configure_shen_paths(shen_config, tmp_path)
    expected_homepath = f"{tmp_path.resolve()}{os.sep}"

    assert shen_config.homepath == expected_homepath
    assert shen_config.datapath == f"{expected_homepath}data{os.sep}"
    assert obdata_path == f"{expected_homepath}obdata_copy{os.sep}"


def test_build_shen_lf_uses_global_fit_a_extinction_convolved_2500_channel(
    tmp_path, monkeypatch
):
    """Gold test that the mock parent is Shen global fit A at observed 2500 A."""
    calls = []
    log_nu_lnu = np.array([44.0, 45.0])
    log_phi_dex = np.array([-5.0, -6.0])

    def fake_return_qlf_in_band(redshift, nu, model):
        calls.append((redshift, nu, model))
        return log_nu_lnu, log_phi_dex

    fake_utilities = types.ModuleType("utilities")
    fake_utilities.return_qlf_in_band = fake_return_qlf_in_band
    monkeypatch.setitem(sys.modules, "utilities", fake_utilities)

    phi_log10, m_grid, z_bins = build_shen_lf(tmp_path)

    assert len(calls) == len(z_bins) == 40
    assert SHEN_GLOBAL_FIT == "A"
    assert all(
        np.isclose(nu, NU_2500_HZ) and model == SHEN_GLOBAL_FIT
        for _, nu, model in calls
    )
    np.testing.assert_allclose(
        phi_log10,
        np.tile(log_phi_dex + LOG10_MAG_JACOBIAN, (len(z_bins), 1)),
    )
    np.testing.assert_allclose(
        m_grid,
        AB_ABSOLUTE_MAG_ZEROPOINT
        - 2.5 * (log_nu_lnu - np.log10(NU_2500_HZ)),
    )


def test_shen_lf_modes_are_explicit_and_normalized():
    assert SHEN_LF_MODES == (
        "all_nh_attenuated",
        "type1_intrinsic",
        "type1_attenuated",
    )
    assert normalize_shen_lf_mode("type1-intrinsic") == "type1_intrinsic"
    with pytest.raises(ValueError, match="Unknown Shen LF mode"):
        normalize_shen_lf_mode("implicit")


def test_shen_lf_mode_runner_uses_run_hubble_footprint_for_each_mode():
    sweep = (ROOT / "run_hubble_shen_lf_modes.xonsh").read_text(encoding="utf-8")
    runner = (ROOT / "run_hubble.xonsh").read_text(encoding="utf-8")

    # The declarations before the first xonsh environment lookup are ordinary
    # Python.  Parse them instead of merely counting model-name strings so the
    # test guards the actual default Cartesian product executed by the sweep.
    declaration_source = sweep.split("requested_models =", maxsplit=1)[0]
    declarations = {}
    for node in ast.parse(declaration_source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                declarations[target.id] = ast.literal_eval(node.value)

    experiment_specs = declarations["experiment_specs"]
    default_experiments = declarations["default_experiments"]
    completeness_runs = declarations["completeness_runs"]
    kulkarni_models = (
        "kulkarni2019_type1_model1",
        "kulkarni2019_type1_model2",
        "kulkarni2019_type1_model3",
    )

    assert default_experiments == (
        "shen_type1_attenuated",
        "wang2026_type1_lade_a",
        "palanque2016_ple_lede",
        *kulkarni_models,
    )
    for model_id in kulkarni_models:
        assert experiment_specs[model_id] == (model_id, None)
    default_jobs = {
        (experiment, completeness_label)
        for experiment in default_experiments
        for _, _, completeness_label in completeness_runs
    }
    assert len(default_experiments) == 6
    assert len(completeness_runs) == 2
    assert len(default_jobs) == 12

    for mode in SHEN_LF_MODES:
        assert f'"{mode}"' in sweep
    assert '"shen_type1_attenuated"' in sweep
    assert '"wang2026_type1_lade_a"' in sweep
    assert '"palanque2016_ple_lede"' in sweep
    for model_id in kulkarni_models:
        assert f'"{model_id}"' in sweep
    assert 'planned jobs: {len(experiments) * len(completeness_runs)}' in sweep
    assert "$QVC_HUBBLE_COMPLETENESS_LF_MODEL = lf_model" in sweep
    assert "$QVC_HUBBLE_SHEN_LF_MODE = shen_mode" in sweep
    assert '__xonsh__.env.pop("QVC_HUBBLE_SHEN_LF_MODE", None)' in sweep
    assert '$QVC_HUBBLE_COMPLETENESS_SIM_FILE = ""' in sweep
    assert "--area-deg2" not in sweep
    assert "QVC_HUBBLE_SHEN_AREA_DEG2" not in sweep
    assert "$QVC_HUBBLE_PREFIX = run_prefix" in sweep
    assert "QVC_HUBBLE_RESUME cannot be shared" in sweep
    assert '"QVC_HUBBLE_COMPLETENESS_SIM_FILE", ""' in runner
    assert "@(completeness_sim_args)" in runner
    assert '$QVC_HUBBLE_ALLOW_SPECTRA_CATALOG_V1 = (' in sweep
    assert '"false" if exact_v2_spectra else "true"' in sweep
    assert '$QVC_HUBBLE_NO_CUTS = "false"' in sweep
    assert '$QVC_HUBBLE_COMPLETENESS_MAGNITUDE = "attenuated"' in sweep
    assert '$QVC_CUT_COMPLETENESS_MAG_2500_MIN = "18.5"' in sweep
    assert '$QVC_CUT_COMPLETENESS_MAG_2500_MAX = "24.0"' in sweep
    assert "completeness magnitude cut:" in sweep
    assert '("2d", False, "2d")' in sweep
    assert '("3d_fhost", True, "3d_fhost_v1proxy")' in sweep
    assert "$QVC_HUBBLE_COMPLETENESS_MODE = completeness_mode" in sweep
    assert "$QVC_HUBBLE_APPROXIMATE_V1_FHOST_2500_PSF = (" in sweep
    assert '$QVC_HUBBLE_COMPLETENESS_MOCK_OVERSAMPLE = "4"' in sweep
    assert '$QVC_HUBBLE_COMPLETENESS_MOCK_MAX_ROWS = "2000000"' in sweep
    assert '$QVC_HUBBLE_COMPLETENESS_MOCK_PROPOSAL_AREA = "full_sky"' in sweep
    assert '$QVC_HUBBLE_COMPLETENESS_MOCK_REQUIRE_FULL_OVERSAMPLE = "true"' in sweep
    assert '$QVC_HUBBLE_DYNESTY_SEED = "12345"' in sweep
    assert 'f"aug24c_lf_areascale_{speed}"' in sweep
    assert 'f"aug24_shen_lf_modes_{speed}"' not in sweep
    assert "--completeness-mock-oversample" in runner
    assert "--completeness-mock-max-rows" in runner
    assert "--completeness-mock-proposal-area" in runner
    assert "--dynesty-seed @(dynesty_seed)" in runner
    assert "--completeness-lf-model @(completeness_lf_model)" in runner
    assert "@(allow_spectra_catalog_v1_args)" in runner
    assert "@(approximate_v1_fhost_args)" in runner


def test_all_lf_runner_pins_the_complete_matched_twelve_job_suite():
    runner_path = ROOT / "run_hubble_all_lf_models.xonsh"
    runner = runner_path.read_text(encoding="utf-8")

    declaration_source = runner.split("repo_root =", maxsplit=1)[0]
    declarations = {}
    for node in ast.parse(declaration_source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                declarations[target.id] = ast.literal_eval(node.value)

    assert declarations["lf_experiments"] == (
        "shen_type1_attenuated",
        "wang2026_type1_lade_a",
        "palanque2016_ple_lede",
        "kulkarni2019_type1_model1",
        "kulkarni2019_type1_model2",
        "kulkarni2019_type1_model3",
    )
    assert len(set(declarations["lf_experiments"])) == 6
    assert "run_hubble_shen_lf_modes.xonsh" in runner
    assert '$QVC_HUBBLE_COMPLETENESS_LF_MODELS = ",".join(lf_experiments)' in runner
    assert '$QVC_HUBBLE_LF_PREFIX_STEM = prefix_stem' in runner
    assert '$QVC_HUBBLE_SPECTRA_FIT_H5 = str(spectra_catalog)' in runner
    assert '$QVC_HUBBLE_EXACT_V2_SPECTRA = "true"' in runner
    assert '$QVC_HUBBLE_ALLOW_SPECTRA_CATALOG_V1 = "false"' in runner
    assert '$QVC_HUBBLE_APPROXIMATE_V1_FHOST_2500_PSF = "false"' in runner
    assert '$QVC_HUBBLE_MAGNITUDE_CONVENTION = "dereddened"' in runner
    assert '$QVC_HUBBLE_COMPLETENESS_MAGNITUDE = "attenuated"' in runner
    assert 'fhostpsf_resumed_m2500norm12.h5' in runner
    assert '__xonsh__.env.pop("QVC_HUBBLE_SHEN_LF_MODES", None)' in runner
    assert 'f"aug25_all_lf_models_m2500norm12_{speed}"' in runner
    assert 'total Hubble jobs: {2 * len(lf_experiments)}' in runner
    assert "xonsh @(str(sweep_runner))" in runner


def test_controlled_lf_sweep_switches_from_v1_proxy_to_exact_v2_host_draws():
    sweep = (ROOT / "run_hubble_shen_lf_modes.xonsh").read_text(
        encoding="utf-8"
    )

    assert '"QVC_HUBBLE_EXACT_V2_SPECTRA", "false"' in sweep
    assert '("3d_fhost", False, "3d_fhost")' in sweep
    assert '"false" if exact_v2_spectra else "true"' in sweep
    assert '"native v2 posterior draws"' in sweep


def test_baseline_quick_standard_runner_runs_only_one_matched_model_twice():
    runner = (ROOT / "run_hubble_baseline_quick_standard.xonsh").read_text(
        encoding="utf-8"
    )

    assert 'speeds = ("quick", "standard")' in runner
    assert "for speed in speeds:" in runner
    assert "$QVC_HUBBLE_SPEED = speed" in runner
    assert '$QVC_HUBBLE_MINIMAL_PLOTS = "false"' in runner
    assert '$QVC_HUBBLE_COMPLETENESS_LF_MODEL = "shen"' in runner
    assert '$QVC_HUBBLE_SHEN_LF_MODE = "type1_attenuated"' in runner
    assert '$QVC_HUBBLE_COMPLETENESS_MODE = "2d"' in runner
    assert '$QVC_HUBBLE_MAGNITUDE_CONVENTION = "dereddened"' in runner
    assert '$QVC_HUBBLE_COMPLETENESS_MAGNITUDE = "attenuated"' in runner
    assert '$QVC_CUT_COMPLETENESS_MAG_2500_MIN = "18.5"' in runner
    assert '$QVC_CUT_COMPLETENESS_MAG_2500_MAX = "24.0"' in runner
    assert '$QVC_HUBBLE_SPECTRA_FIT_H5 = str(spectra_catalog)' in runner
    assert 'fhostpsf_resumed_m2500norm12.h5' in runner
    assert '$QVC_HUBBLE_PREFIX = run_prefix' in runner
    assert 'run_prefix = f"{prefix_stem}_{speed}"' in runner
    assert 'runner = repo_root / "run_hubble.xonsh"' in runner
    assert "xonsh @(str(runner))" in runner
    assert "QVC_HUBBLE_RESUME cannot be shared" in runner
    assert "total jobs: 2 (quick, then standard)" in runner
    assert "run_hubble_all_lf_models.xonsh" not in runner


def test_shen_type1_fraction_matches_fiducial_formula_and_caps_at_z2():
    log_lx = np.array([43.0, 44.0, 45.0])
    psi = shen_absorbed_fraction(log_lx, 1.0)

    np.testing.assert_allclose(
        shen_type1_fraction(log_lx, 1.0),
        (1.0 - psi) / (1.0 + psi),
    )
    np.testing.assert_allclose(
        shen_type1_fraction(log_lx, 2.0),
        shen_type1_fraction(log_lx, 5.0),
    )
    assert np.all((psi >= 0.20) & (psi <= 0.84))


def test_shen_type1_nh_bins_sum_to_total_population_fraction():
    log_lx = np.linspace(42.0, 47.0, 30)
    f_20_21, f_21_22 = _shen_type1_nh_bin_fractions(log_lx, 1.4)

    np.testing.assert_allclose(
        f_20_21 + f_21_22,
        shen_type1_fraction(log_lx, 1.4),
        rtol=1e-12,
        atol=1e-14,
    )
    assert np.all(f_20_21 >= 0.0)
    assert np.all(f_21_22 >= 0.0)


def test_build_shen_type1_modes_use_bolometric_lf_and_attenuation(
    tmp_path,
    monkeypatch,
):
    log_lbol = np.linspace(43.0, 49.0, 80)
    log_phi = -5.0 - 0.15 * (log_lbol - 45.0) ** 2
    calls = {"bolometric": 0, "tau": 0, "log_nh": []}

    def fake_return_bolometric_qlf(redshift, model):
        calls["bolometric"] += 1
        assert model == SHEN_GLOBAL_FIT
        return log_lbol, log_phi - 0.05 * redshift

    fake_utilities = types.ModuleType("utilities")
    fake_utilities.return_bolometric_qlf = fake_return_bolometric_qlf
    fake_utilities.return_dtg = lambda redshift: 1.0 + 0.1 * redshift
    monkeypatch.setitem(sys.modules, "utilities", fake_utilities)

    class FakeBackend:
        @staticmethod
        def l_band(log_lbol_lsun, nu):
            offset = 1.0 if nu == -4.0 else 0.5
            return 10.0 ** (log_lbol_lsun - offset)

        @staticmethod
        def l_band_dispersion(log_lbol_lsun, nu):
            return 0.20

        @staticmethod
        def return_tau(log_nh, nu, dust_to_gas):
            calls["tau"] += 1
            calls["log_nh"].append(log_nh)
            return np.log(10.0) * 0.15 * (log_nh - 20.0) * dust_to_gas

    monkeypatch.setattr(
        "qvc.hubble.completeness_mock_catalog._load_shen_c_backend",
        lambda _: FakeBackend(),
    )

    intrinsic, intrinsic_m, z_bins = build_shen_lf(
        tmp_path,
        mode="type1_intrinsic",
    )
    attenuated, attenuated_m, attenuated_z = build_shen_lf(
        tmp_path,
        mode="type1_attenuated",
    )

    assert intrinsic.shape == attenuated.shape == (40, log_lbol.size)
    np.testing.assert_allclose(intrinsic_m, attenuated_m)
    np.testing.assert_allclose(z_bins, attenuated_z)
    assert calls["bolometric"] == 80
    assert calls["tau"] > 0
    assert min(calls["log_nh"]) >= 20.0
    assert max(calls["log_nh"]) < 22.0
    assert np.all(np.isfinite(intrinsic))
    assert np.all(np.isfinite(attenuated))
    assert np.sum(10.0**attenuated) < np.sum(10.0**intrinsic)

    monkeypatch.setattr(
        FakeBackend,
        "return_tau",
        staticmethod(lambda log_nh, nu, dust_to_gas: 0.0),
    )
    zero_attenuation, zero_attenuation_m, zero_attenuation_z = build_shen_lf(
        tmp_path,
        mode="type1_attenuated",
    )
    np.testing.assert_allclose(zero_attenuation_m, intrinsic_m)
    np.testing.assert_allclose(zero_attenuation_z, z_bins)
    np.testing.assert_allclose(
        zero_attenuation,
        intrinsic,
        rtol=2e-12,
        atol=2e-12,
    )


def test_log_nu_lnu_to_ab_absolute_magnitude_gold_value():
    target_magnitude = -25.0
    log_lnu = (AB_ABSOLUTE_MAG_ZEROPOINT - target_magnitude) / 2.5
    log_nu_lnu = log_lnu + np.log10(NU_2500_HZ)
    np.testing.assert_allclose(
        log_nu_lnu_to_ab_absolute_magnitude(log_nu_lnu, NU_2500_HZ),
        target_magnitude,
        atol=1e-12,
    )


def _build_pivot_context(df_agn):
    z = df_agn["z"].to_numpy(dtype=float)
    return hubble_model.build_agn_pivot_context(
        df_agn,
        z_range=(float(np.min(z)), float(np.max(z))),
    )


def test_completeness_loglike_respects_hard_magnitude_support():
    mag_centers = np.linspace(18.5, 24.0, 5501)
    z_centers = np.linspace(0.0, 4.0, 20)
    completeness = hcr.Completeness2D(
        mag_centers,
        z_centers,
        np.ones((mag_centers.size, z_centers.size)),
    )

    _, blob = completeness_loglike(
        m_obs=np.array([mag_centers[0]]),
        m_obs_err=np.array([0.05]),
        m_model=np.array([17.5]),
        mu_err=np.array([0.3]),
        z=np.array([1.0]),
        completeness_model=completeness,
        m_grid=mag_centers,
        magnitude_support=(mag_centers[0], mag_centers[-1]),
    )

    sigma = 0.3
    lower = mag_centers[0]
    upper = mag_centers[-1]
    alpha = (lower - 17.5) / sigma
    beta = (upper - 17.5) / sigma
    phi_alpha = np.exp(-0.5 * alpha**2) / np.sqrt(2.0 * np.pi)
    phi_beta = np.exp(-0.5 * beta**2) / np.sqrt(2.0 * np.pi)
    expected_z = ndtr(beta) - ndtr(alpha)
    expected_bias = sigma * (phi_alpha - phi_beta) / expected_z
    expected_variance = sigma**2 * (
        1.0
        + (alpha * phi_alpha - beta * phi_beta) / expected_z
        - ((phi_alpha - phi_beta) / expected_z) ** 2
    )

    np.testing.assert_allclose(blob[0, 0], expected_z, rtol=2e-5)
    np.testing.assert_allclose(blob[1, 0], expected_bias, rtol=2e-5)
    np.testing.assert_allclose(blob[2, 0], np.sqrt(expected_variance), rtol=2e-5)


def test_hard_cut_normalization_extends_edge_bin_completeness_values():
    lower, upper = 18.5, 24.0
    n_mag_bins = 30
    bin_width = (upper - lower) / n_mag_bins
    mag_centers = lower + (np.arange(n_mag_bins) + 0.5) * bin_width
    z_centers = np.array([0.5, 1.5])
    completeness_by_mag = np.linspace(0.85, 0.15, n_mag_bins)
    completeness = hcr.Completeness2D(
        mag_centers,
        z_centers,
        np.repeat(completeness_by_mag[:, None], z_centers.size, axis=1),
        magnitude_support=(lower, upper),
    )
    m_model = np.array([lower, upper])
    sigma = np.array([0.15, 0.15])

    try:
        _, blob = completeness_loglike(
            m_obs=np.array([lower, upper]),
            m_obs_err=np.full(2, 0.05),
            m_model=m_model,
            mu_err=sigma,
            z=np.ones(2),
            completeness_model=completeness,
            m_grid=mag_centers,
            magnitude_support=(lower, upper),
        )
    except ValueError as exc:
        pytest.fail(f"histogram-edge support must be integrable: {exc}")

    integration_grid = np.concatenate(([lower], mag_centers, [upper]))
    extended_completeness = np.interp(
        integration_grid,
        mag_centers,
        completeness_by_mag,
        left=completeness_by_mag[0],
        right=completeness_by_mag[-1],
    )
    pdf = np.exp(
        -0.5
        * ((integration_grid[None, :] - m_model[:, None]) / sigma[:, None]) ** 2
    ) / (np.sqrt(2.0 * np.pi) * sigma[:, None])
    expected_z = np.trapezoid(
        pdf * extended_completeness[None, :],
        integration_grid,
        axis=1,
    )

    np.testing.assert_allclose(blob[0], expected_z, rtol=1e-13, atol=0.0)


def test_hard_cut_normalization_uses_explicit_off_grid_bounds():
    class UnityCompleteness:
        mode = "2d"

        def __call__(self, mag, z):
            mag, z = np.broadcast_arrays(
                np.asarray(mag, dtype=float),
                np.asarray(z, dtype=float),
            )
            return np.ones_like(mag)

    mag_grid = np.linspace(18.0, 25.0, 7001)
    magnitude_support = (20.2555, 22.7445)
    sigma = np.array([0.3, 0.65, 0.3])
    m_model = np.array(
        [magnitude_support[0] - 1.0, 21.4, magnitude_support[1] + 1.0]
    )

    log_z, blob = completeness_loglike(
        m_obs=np.array([magnitude_support[0], 21.4, magnitude_support[1]]),
        m_obs_err=np.full(3, 0.05),
        m_model=m_model,
        mu_err=sigma,
        z=np.array([0.5, 1.5, 2.5]),
        completeness_model=UnityCompleteness(),
        m_grid=mag_grid,
        magnitude_support=magnitude_support,
    )

    alpha = (magnitude_support[0] - m_model) / sigma
    beta = (magnitude_support[1] - m_model) / sigma
    phi_alpha = np.exp(-0.5 * alpha**2) / np.sqrt(2.0 * np.pi)
    phi_beta = np.exp(-0.5 * beta**2) / np.sqrt(2.0 * np.pi)
    expected_z = ndtr(beta) - ndtr(alpha)
    expected_bias = sigma * (phi_alpha - phi_beta) / expected_z
    expected_variance = sigma**2 * (
        1.0
        + (alpha * phi_alpha - beta * phi_beta) / expected_z
        - ((phi_alpha - phi_beta) / expected_z) ** 2
    )

    np.testing.assert_allclose(log_z, np.log(expected_z).sum(), rtol=2e-5)
    np.testing.assert_allclose(blob[0], expected_z, rtol=2e-5)
    np.testing.assert_allclose(blob[1], expected_bias, rtol=2e-5)
    np.testing.assert_allclose(blob[2], np.sqrt(expected_variance), rtol=2e-5)


def test_jax_completeness_preparation_carries_explicit_support():
    pytest.importorskip("jax")
    from qvc.hubble.hubble_fit_jax import _prepare_completeness_for_jax

    mag_centers = np.linspace(18.0, 25.0, 8)
    z_centers = np.linspace(0.0, 4.0, 5)
    magnitude_support = (18.25, 24.75)
    completeness = hcr.Completeness2D(
        mag_centers,
        z_centers,
        np.ones((mag_centers.size, z_centers.size)),
        magnitude_support=magnitude_support,
    )

    prepared = _prepare_completeness_for_jax(
        (completeness, mag_centers, z_centers, 1.0, 1.0, 0.0)
    )

    assert prepared.get("magnitude_support") is not None
    np.testing.assert_allclose(
        np.asarray(prepared["magnitude_support"]),
        magnitude_support,
    )
    integration_grid = np.asarray(prepared["integration_mag_grid"])
    assert integration_grid[0] == magnitude_support[0]
    assert integration_grid[-1] == magnitude_support[1]
    with pytest.raises(ValueError, match="observed selection magnitudes"):
        _prepare_completeness_for_jax(
            (completeness, mag_centers, z_centers, 1.0, 1.0, 0.0),
            selection_magnitude=np.array([magnitude_support[0] - 1e-8]),
        )


def test_cpu_and_jax_hard_cut_normalizations_match():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from qvc.hubble.hubble_fit_jax import (
        _completeness_loglike_jax,
        _prepare_completeness_for_jax,
    )

    jax.config.update("jax_enable_x64", True)
    magnitude_support = (18.5, 24.0)
    n_mag_bins = 1400
    bin_width = (magnitude_support[1] - magnitude_support[0]) / n_mag_bins
    mag_centers = magnitude_support[0] + (
        np.arange(n_mag_bins) + 0.5
    ) * bin_width
    z_centers = np.linspace(0.0, 4.0, 9)
    mm, zz = np.meshgrid(mag_centers, z_centers, indexing="ij")
    completeness_map = np.clip(
        0.95 - 0.04 * (mm - magnitude_support[0]) - 0.03 * zz,
        0.1,
        1.0,
    )
    completeness = hcr.Completeness2D(
        mag_centers,
        z_centers,
        completeness_map,
        magnitude_support=magnitude_support,
    )
    completeness_params = (
        completeness,
        mag_centers,
        z_centers,
        mag_centers[1] - mag_centers[0],
        z_centers[1] - z_centers[0],
        0.0,
    )
    m_model = np.array([17.5, 21.2, 25.5])
    mu_err = np.array([0.3, 0.5, 0.3])
    redshift = np.array([0.5, 1.5, 2.5])

    cpu_log_z, _ = completeness_loglike(
        m_obs=np.array([magnitude_support[0], 21.2, magnitude_support[1]]),
        m_obs_err=np.full(3, 0.05),
        m_model=m_model,
        mu_err=mu_err,
        z=redshift,
        completeness_model=completeness,
        m_grid=mag_centers,
        magnitude_support=magnitude_support,
    )
    jax_log_z = _completeness_loglike_jax(
        jnp.asarray(m_model),
        jnp.asarray(mu_err),
        jnp.asarray(redshift),
        _prepare_completeness_for_jax(completeness_params),
        None,
        None,
    )

    np.testing.assert_allclose(float(jax_log_z), cpu_log_z, rtol=1e-11)


@pytest.mark.parametrize(
    "magnitude_support",
    [
        (20.0, 20.0),
        (21.0, 20.0),
        (np.nan, 22.0),
        (20.0, np.inf),
        (17.9, 22.0),
        (20.0, 24.1),
    ],
)
def test_completeness_loglike_rejects_invalid_hard_support(magnitude_support):
    class UnityCompleteness:
        mode = "2d"

        def __call__(self, mag, z):
            mag, z = np.broadcast_arrays(mag, z)
            return np.ones_like(mag, dtype=float)

    with pytest.raises(ValueError, match="magnitude_support"):
        completeness_loglike(
            m_obs=np.array([21.0]),
            m_obs_err=np.array([0.05]),
            m_model=np.array([21.0]),
            mu_err=np.array([0.3]),
            z=np.array([1.0]),
            completeness_model=UnityCompleteness(),
            m_grid=np.linspace(18.0, 24.0, 61),
            magnitude_support=magnitude_support,
        )


def test_hard_support_accepts_boundaries_and_rejects_unselected_observations():
    class UnityCompleteness:
        mode = "2d"

        def __call__(self, mag, z):
            mag, z = np.broadcast_arrays(mag, z)
            return np.ones_like(mag, dtype=float)

    kwargs = {
        "m_obs_err": np.full(2, 0.05),
        "m_model": np.full(2, 21.0),
        "mu_err": np.full(2, 0.3),
        "z": np.array([1.0, 1.5]),
        "completeness_model": UnityCompleteness(),
        "m_grid": np.linspace(18.0, 24.0, 61),
        "magnitude_support": (20.0, 22.0),
    }

    log_z, _ = completeness_loglike(m_obs=np.array([20.0, 22.0]), **kwargs)
    assert np.isfinite(log_z)
    with pytest.raises(ValueError, match="observed selection magnitudes"):
        completeness_loglike(m_obs=np.array([20.0 - 1e-8, 22.0]), **kwargs)
    with pytest.raises(ValueError, match="observed selection magnitudes"):
        completeness_loglike(m_obs=np.array([20.0, 22.0 + 1e-8]), **kwargs)


def test_selection_correction_matches_truncated_normal_and_recovers_parent_mean():
    """Regression test the full correction against a known magnitude-limit solution."""

    class HardMagnitudeLimit:
        mode = "2d"

        def __init__(self, limit):
            self.limit = float(limit)

        def __call__(self, mag, z):
            mag, z = np.broadcast_arrays(
                np.asarray(mag, dtype=float),
                np.asarray(z, dtype=float),
            )
            # Half weight at the discontinuity is the trapezoid-rule convention.
            return np.where(mag < self.limit, 1.0, np.where(mag == self.limit, 0.5, 0.0))

    m_model = 22.3
    sigma = 0.45
    m_limit = 22.0
    mag_grid = np.linspace(18.5, 24.5, 6001)

    log_z, blob = completeness_loglike(
        m_obs=np.array([m_model]),
        m_obs_err=np.array([0.05]),
        m_model=np.array([m_model]),
        mu_err=np.array([sigma]),
        z=np.array([1.2]),
        completeness_model=HardMagnitudeLimit(m_limit),
        m_grid=mag_grid,
        magnitude_support=(mag_grid[0], mag_grid[-1]),
    )

    alpha = (m_limit - m_model) / sigma
    expected_z = ndtr(alpha)
    inverse_mills = np.exp(-0.5 * alpha**2) / (np.sqrt(2.0 * np.pi) * expected_z)
    expected_bias = -sigma * inverse_mills
    expected_sigma = sigma * np.sqrt(1.0 - alpha * inverse_mills - inverse_mills**2)

    np.testing.assert_allclose(np.exp(log_z), expected_z, rtol=2e-6)
    np.testing.assert_allclose(blob[0, 0], expected_z, rtol=2e-6)
    np.testing.assert_allclose(blob[1, 0], expected_bias, atol=2e-6)
    np.testing.assert_allclose(blob[2, 0], expected_sigma, atol=2e-6)

    selected_mean = m_model + blob[1, 0]
    corrected_mean = selected_mean - blob[1, 0]
    np.testing.assert_allclose(corrected_mean, m_model, atol=1e-12)


def _make_fake_fhost_df(n=200, seed=123):
    rng = np.random.default_rng(seed)
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    logL = np.linspace(42.3, 45.6, n)
    z = np.linspace(0.4, 2.8, n)
    f_mean = hcr.generalized_sigmoid_fhost(logL, 44.2, 2.8, 0.9)
    f_host = np.clip(
        hcr.expit(hcr.logit(np.clip(f_mean, 1e-3, 1.0 - 1e-3)) + rng.normal(0.0, 0.45, size=n)),
        0.0,
        1.0,
    )
    M2500 = 90.0 - 2.5 * logL
    m2500 = M2500 + cosmo.distmod(z).value
    return pd.DataFrame(
        {
            "object_id": [f"agn_{i:04d}" for i in range(n)],
            "z": z,
            "apparent_mag_2500": m2500,
            "m_2500_dereddened": m2500,
            "m_2500_attenuated_model": m2500,
            hcr.COMPLETENESS_MAG_COL: m2500,
            "f_host_2500": f_host,
            "f_host_2500_psf": f_host,
        }
    )


def _make_fake_agn_sample_with_fhost(n_agn=24, seed=123):
    rng = np.random.default_rng(seed)
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)

    z = np.linspace(0.5, 2.2, n_agn)
    log_sigma_uv = rng.normal(-0.8, 0.12, size=n_agn)
    log_tau_uv = 2.7 + 0.35 * (z - np.mean(z)) + rng.normal(0.0, 0.08, size=n_agn)
    log_sigma_hat0 = log_sigma_uv - 1.4 + rng.normal(0.0, 0.05, size=n_agn)
    logL = np.linspace(42.8, 45.4, n_agn)
    M2500_true = 90.0 - 2.5 * logL

    true_params = {
        "M0_agn": -23.4,
        "alpha_agn": -1.8,
        "beta_agn": -0.9,
        "M0_sn": -19.2,
        "H0": 70.0,
        "Om0": 0.3,
    }

    obs_dict = {
        "log_sigma_hat0": log_sigma_hat0,
        "log_sigma_uv": log_sigma_uv,
        "log_tau_uv_rf": log_tau_uv,
        "log_sigma_hat0_err": np.full(n_agn, 0.04),
        "log_sigma_uv_std_psd": np.full(n_agn, 0.05),
        "log_tau_uv_rf_std_psd": np.full(n_agn, 0.06),
        "log_sigma_uv_log_tau_uv_rf_cov_psd": np.full(n_agn, 0.001),
    }
    object_ids = [f"agn_{i:03d}" for i in range(n_agn)]
    pivot_df = pd.DataFrame(
        {
            "object_id": object_ids,
            "z": z,
            **obs_dict,
        }
    )
    pivot_context = _build_pivot_context(pivot_df)
    params_arr = hubble_model.agn_model_pack_params(true_params)
    obs_arr, _, pivots = hubble_model.agn_model_pack_obs(
        obs_dict,
        pivot_context=pivot_context,
    )
    absolute_mag = hubble_model.M_model_agn(params_arr, obs_arr, pivots)
    mu = cosmo.distmod(z).value
    apparent_mag = absolute_mag + mu + rng.normal(0.0, 0.04, size=n_agn)
    f_host = np.clip(hcr.generalized_sigmoid_fhost(logL, 44.2, 2.8, 0.9), 0.0, 1.0)

    return pd.DataFrame(
        {
            "object_id": object_ids,
            "z": z,
            "z_err": np.full(n_agn, 0.002),
            "apparent_mag_2500": apparent_mag,
            "apparent_mag_2500_err": np.full(n_agn, 0.04),
            "m_2500_dereddened": apparent_mag,
            "m_2500_dereddened_err": np.full(n_agn, 0.04),
            "m_2500_attenuated_model": apparent_mag + 0.35,
            "m_2500_attenuated_model_err": np.full(n_agn, 0.06),
            hcr.COMPLETENESS_MAG_COL: apparent_mag,
            hcr.COMPLETENESS_MAG_ERR_COL: np.full(n_agn, 0.04),
            "log_sigma_hat0": log_sigma_hat0,
            "log_sigma_uv": log_sigma_uv,
            "log_tau_uv_rf": log_tau_uv,
            "log_sigma_hat0_err": np.full(n_agn, 0.04),
            "log_sigma_uv_std_psd": np.full(n_agn, 0.05),
            "log_tau_uv_rf_std_psd": np.full(n_agn, 0.06),
            "log_sigma_uv_log_tau_uv_rf_cov_psd": np.full(n_agn, 0.001),
            "delta_m_flux_recal": rng.normal(0.0, 0.02, size=n_agn),
            "f_host_2500": f_host,
            "f_host_2500_psf": f_host,
        }
    )


def _make_fake_agn_sample_with_fhost_alpha(n_agn=32, seed=123, alpha_center=-1.25):
    rng = np.random.default_rng(seed)
    df = _make_fake_agn_sample_with_fhost(n_agn=n_agn, seed=seed)
    trend = 0.08 * (df["z"].to_numpy(dtype=float) - float(df["z"].mean()))
    df["alpha_lambda"] = alpha_center + trend + rng.normal(0.0, 0.06, size=n_agn)
    return df


def _write_fake_sim_file(
    path,
    n=2000,
    seed=321,
    include_alpha=False,
    alpha_center=-1.2,
    include_fhost=False,
):
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.1, 3.5, size=n)
    logL = rng.uniform(42.2, 46.0, size=n)
    M2500 = 90.0 - 2.5 * logL
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    m2500 = M2500 + cosmo.distmod(z).value
    with h5py.File(path, "w") as handle:
        handle.create_dataset("z", data=z)
        handle.create_dataset("apparent_mag_2500", data=m2500)
        if include_alpha:
            alpha_lambda = alpha_center + 0.12 * (z - np.mean(z)) + rng.normal(0.0, 0.08, size=n)
            handle.create_dataset(
                "alpha_nu_lf_conversion",
                data=-alpha_lambda - 2.0,
            )
        if include_fhost:
            f_host = np.clip(0.25 + 0.08 * (z - np.mean(z)) + rng.normal(0.0, 0.05, size=n), 0.01, 0.9)
            handle.create_dataset("f_host_2500_psf", data=f_host)


def test_completeness_2d_plot_smoothing_is_display_only(tmp_path):
    df_agn = _make_fake_agn_sample_with_fhost(n_agn=36)
    sim_file = tmp_path / "mock2d.h5"
    _write_fake_sim_file(sim_file, n=1200)

    comp_no_plot, mag_centers, z_centers, *_ = hcr.get_completeness_function_2d(
        df_agn,
        sim_file=str(sim_file),
        n_mag_bins=12,
        n_z_bins=10,
        plot=False,
    )
    comp_with_plot, mag_centers_plot, z_centers_plot, *_ = hcr.get_completeness_function_2d(
        df_agn,
        sim_file=str(sim_file),
        n_mag_bins=12,
        n_z_bins=10,
        plot=True,
        plot_path=str(tmp_path),
    )

    mag_grid, z_grid = np.meshgrid(mag_centers, z_centers, indexing="ij")
    np.testing.assert_allclose(mag_centers_plot, mag_centers)
    np.testing.assert_allclose(z_centers_plot, z_centers)
    np.testing.assert_allclose(comp_with_plot(mag_grid, z_grid), comp_no_plot(mag_grid, z_grid))
    assert getattr(comp_no_plot, "magnitude_support", None) == (
        hubble_cuts.COMPLETENESS_MAG_2500_MIN,
        hubble_cuts.COMPLETENESS_MAG_2500_MAX,
    )
    assert (tmp_path / "completeness" / "completeness_map.pdf").exists()


def test_fit_fhost_2500_model_monotonic_and_bounded():
    df = _make_fake_fhost_df()
    model = hcr.fit_fhost_2500_l2500_model(df)
    assert np.isfinite(model["x0"])
    assert np.isfinite(model["k"])
    assert np.isfinite(model["nu"])
    assert np.isfinite(model["sigma_host_logit"])

    grid = np.linspace(42.5, 45.5, 200)
    pred = hcr.predict_fhost_2500_from_logL2500(grid, model)
    assert np.all(np.isfinite(pred))
    assert np.all((pred > 0.0) & (pred < 1.0))
    assert np.all(np.diff(pred) <= 1e-8)


def test_sample_fhost_2500_from_model_is_bounded_and_luminosity_dependent():
    df = _make_fake_fhost_df()
    model = hcr.fit_fhost_2500_l2500_model(df)
    rng = np.random.default_rng(0)
    low = hcr.sample_fhost_2500_from_logL2500(np.full(5000, 43.0), model, rng)
    high = hcr.sample_fhost_2500_from_logL2500(np.full(5000, 45.2), model, rng)
    assert np.all((low > 0.0) & (low < 1.0))
    assert np.all((high > 0.0) & (high < 1.0))
    assert np.mean(low) > np.mean(high)


def test_completeness3d_shape_and_likelihood_matches_2d_when_host_independent():
    mag_centers = np.linspace(18.5, 24.0, 9)
    z_centers = np.linspace(0.0, 4.0, 7)
    fhost_centers = np.linspace(0.05, 0.95, 5)
    alpha_centers = np.linspace(-2.4, -0.8, 4)
    mm, zz = np.meshgrid(mag_centers, z_centers, indexing="ij")
    c2 = np.clip(np.exp(-0.12 * zz) / (1.0 + np.exp((mm - 22.0) / 0.35)), 0.0, 1.0)
    c3 = np.repeat(c2[:, :, None], len(fhost_centers), axis=2)
    c4 = np.repeat(c3[:, :, :, None], len(alpha_centers), axis=3)

    comp2 = hcr.Completeness2D(mag_centers, z_centers, c2)
    comp3 = hcr.Completeness3D(mag_centers, z_centers, fhost_centers, c3)
    comp4 = hcr.Completeness4D(mag_centers, z_centers, fhost_centers, alpha_centers, c4)

    q = comp3(np.array([[20.0, 21.0]]), np.array([[1.0, 2.0]]), np.array([[0.2, 0.8]]))
    assert q.shape == (1, 2)
    assert np.all((q >= 0.0) & (q <= 1.0))
    q4 = comp4(
        np.array([[20.0, 21.0]]),
        np.array([[1.0, 2.0]]),
        np.array([[0.2, 0.8]]),
        np.array([[-1.9, -1.2]]),
    )
    assert q4.shape == (1, 2)
    assert np.all((q4 >= 0.0) & (q4 <= 1.0))

    m_obs = np.array([20.5, 21.3, 22.1])
    m_model = np.array([20.4, 21.1, 22.0])
    mu_err = np.array([0.15, 0.18, 0.20])
    z = np.array([0.8, 1.5, 2.2])
    magnitude_support = (18.75, 23.75)
    f_host = np.array([0.2, 0.5, 0.8])
    alpha_lambda = np.array([-1.8, -1.5, -1.1])

    ll2, blob2 = hubble_likelihood.completeness_loglike(
        m_obs=m_obs,
        m_obs_err=np.full_like(m_obs, 0.05),
        m_model=m_model,
        mu_err=mu_err,
        z=z,
        completeness_model=comp2,
        m_grid=mag_centers,
        magnitude_support=magnitude_support,
    )
    ll3, blob3 = hubble_likelihood.completeness_loglike(
        m_obs=m_obs,
        m_obs_err=np.full_like(m_obs, 0.05),
        m_model=m_model,
        mu_err=mu_err,
        z=z,
        completeness_model=comp3,
        m_grid=mag_centers,
        magnitude_support=magnitude_support,
        f_host_2500_psf=f_host,
    )
    ll4, blob4 = hubble_likelihood.completeness_loglike(
        m_obs=m_obs,
        m_obs_err=np.full_like(m_obs, 0.05),
        m_model=m_model,
        mu_err=mu_err,
        z=z,
        completeness_model=comp4,
        m_grid=mag_centers,
        magnitude_support=magnitude_support,
        f_host_2500_psf=f_host,
        alpha_lambda=alpha_lambda,
    )

    assert np.allclose(ll2, ll3, rtol=1e-10, atol=1e-10)
    assert np.allclose(blob2, blob3, rtol=1e-10, atol=1e-10)
    assert np.allclose(ll2, ll4, rtol=1e-10, atol=1e-10)
    assert np.allclose(blob2, blob4, rtol=1e-10, atol=1e-10)


def test_completeness_loglike_caches_detection_grid_across_parameter_calls():
    mag_centers = np.linspace(18.5, 24.0, 9)
    z_centers = np.linspace(0.0, 4.0, 7)
    mm, zz = np.meshgrid(mag_centers, z_centers, indexing="ij")
    c2 = np.clip(np.exp(-0.12 * zz) / (1.0 + np.exp((mm - 22.0) / 0.35)), 0.0, 1.0)
    base = hcr.Completeness2D(mag_centers, z_centers, c2)

    class CountingCompleteness:
        mode = "2d"

        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            return self.wrapped(*args, **kwargs)

    comp = CountingCompleteness(base)
    m_obs = np.array([20.5, 21.3, 22.1])
    m_model = np.array([20.4, 21.1, 22.0])
    mu_err = np.array([0.15, 0.18, 0.20])
    z = np.array([0.8, 1.5, 2.2])
    magnitude_support = (18.75, 23.75)

    ll1, blob1 = hubble_likelihood.completeness_loglike(
        m_obs=m_obs,
        m_obs_err=np.full_like(m_obs, 0.05),
        m_model=m_model,
        mu_err=mu_err,
        z=z,
        completeness_model=comp,
        m_grid=mag_centers,
        magnitude_support=magnitude_support,
    )
    ll2, blob2 = hubble_likelihood.completeness_loglike(
        m_obs=m_obs,
        m_obs_err=np.full_like(m_obs, 0.05),
        m_model=m_model + 0.05,
        mu_err=mu_err,
        z=z,
        completeness_model=comp,
        m_grid=mag_centers,
        magnitude_support=magnitude_support,
    )

    assert comp.calls == 1
    assert len(comp._likelihood_magnitude_grid_cache) == 1
    assert len(comp._likelihood_pdet_cache) == 1
    assert np.isfinite(ll1)
    assert np.isfinite(ll2)
    assert blob1.shape == blob2.shape == (3, len(m_obs))


def test_log_likelihood_does_not_use_completeness_smoothing_as_extra_scatter(monkeypatch):
    df_agn = _make_fake_agn_sample_with_fhost(n_agn=4)
    df_pantheon = pd.DataFrame(
        {
            "zHD": np.linspace(0.02, 0.8, 8),
            "zHEL": np.linspace(0.02, 0.8, 8),
            "m_b_corr": np.linspace(15.0, 18.0, 8),
            "IS_CALIBRATOR": np.zeros(8, dtype=int),
            "CEPH_DIST": np.full(8, -9.0),
            "MU_SH0ES_ERR_DIAG": np.full(8, 0.08),
        }
    )
    mag_centers = np.linspace(18.5, 24.0, 9)
    z_centers = np.linspace(0.0, 4.0, 7)
    c2 = np.ones((len(mag_centers), len(z_centers)))
    magnitude_support = (19.0, 23.5)
    completeness_model = hcr.Completeness2D(mag_centers, z_centers, c2)
    completeness_model.magnitude_support = magnitude_support
    completeness_params = (
        completeness_model,
        mag_centers,
        z_centers,
        0.5,
        0.1,
        99.0,
    )
    captured = {}

    def fake_completeness_loglike(*args, **kwargs):
        captured["sigma_completeness"] = kwargs["sigma_completeness"]
        captured["magnitude_support"] = kwargs.get("magnitude_support")
        n = len(kwargs["z"])
        return 0.0, np.ones((3, n), dtype=float)

    monkeypatch.setattr(hubble_likelihood, "completeness_loglike", fake_completeness_loglike)

    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)
    agn_fields = hubble_model.agn_model_req_obs + hubble_model.agn_model_req_errs
    agn_fields += ("apparent_mag_2500", "apparent_mag_2500_err", "z", "z_err", "object_id")
    agn_fields += (hcr.COMPLETENESS_MAG_COL, hcr.COMPLETENESS_MAG_ERR_COL)
    agn_data = {col: df_agn[col].to_numpy() for col in agn_fields}
    pantheon_data = {col: df_pantheon[col].to_numpy() for col in df_pantheon.columns}
    pivot_context = _build_pivot_context(df_agn)

    logl, _ = hubble_likelihood.log_likelihood(
        theta,
        agn_data=agn_data,
        pantheon_data=pantheon_data,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM",
        completeness_params=completeness_params,
        z_pivot_agn=hubble_fit.z_pivot_agn,
        agn_pivot_context=pivot_context,
        agn_calibrators_data=None,
        only_sna=False,
        use_full_cov=False,
    )

    assert np.isfinite(logl)
    assert captured["sigma_completeness"] == 0.0
    assert captured["magnitude_support"] == magnitude_support


def test_completeness_callables_return_zero_for_nonfinite_queries():
    mag_centers = np.linspace(18.5, 24.0, 5)
    z_centers = np.linspace(0.1, 3.0, 4)
    fhost_centers = np.linspace(0.05, 0.95, 3)
    alpha_centers = np.linspace(-2.5, -0.5, 3)
    c2 = np.ones((len(mag_centers), len(z_centers)))
    c3 = np.ones((len(mag_centers), len(z_centers), len(fhost_centers)))
    c4 = np.ones((len(mag_centers), len(z_centers), len(fhost_centers), len(alpha_centers)))

    comp2 = hcr.Completeness2D(mag_centers, z_centers, c2)
    comp3 = hcr.Completeness3D(mag_centers, z_centers, fhost_centers, c3)
    comp4 = hcr.Completeness4D(mag_centers, z_centers, fhost_centers, alpha_centers, c4)

    np.testing.assert_allclose(comp2([20.0, np.nan], [1.0, 1.0]), [1.0, 0.0])
    np.testing.assert_allclose(comp3([20.0, np.nan], [1.0, 1.0], [0.2, 0.2]), [1.0, 0.0])
    np.testing.assert_allclose(
        comp4([20.0, np.nan], [1.0, 1.0], [0.2, 0.2], [-1.5, -1.5]),
        [1.0, 0.0],
    )


def test_completeness3d_warning_uses_physical_bin_edges_and_axis_counts(capsys):
    mag_edges = np.linspace(18.5, 24.0, 31)
    z_edges = np.linspace(0.0, 4.0, 41)
    fhost_edges = np.linspace(0.0, 1.0, 21)
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    fhost_centers = 0.5 * (fhost_edges[:-1] + fhost_edges[1:])
    cube = np.ones(
        (len(mag_centers), len(z_centers), len(fhost_centers)),
        dtype=float,
    )
    comp3 = hcr.Completeness3D(
        mag_centers,
        z_centers,
        fhost_centers,
        cube,
        magnitude_support=(18.5, 24.0),
    )

    comp3(
        np.array([18.5, 24.0]),
        np.array([0.0, 4.0]),
        np.array([0.0, 1.0]),
    )
    assert "[WARNING]" not in capsys.readouterr().out

    comp3(
        np.array([18.4, 20.0, 20.0, np.nan]),
        np.array([1.0, -0.1, 1.0, 1.0]),
        np.array([0.5, 0.5, 1.1, 0.5]),
    )
    warning = capsys.readouterr().out
    assert "calibrated physical support" in warning
    assert "m=[18.50, 24.00]" in warning
    assert "z=[0.00, 4.00]" in warning
    assert "f_host=[0.000, 1.000]" in warning
    assert "counts: m=1, z=1, f_host=1, nonfinite=1, any=4" in warning


def test_completeness_callables_linearly_extrapolate_redshift_without_edge_clipping():
    mag_centers = np.array([20.0, 21.0])
    z_centers = np.array([1.0, 2.0])
    fhost_centers = np.array([0.1, 0.9])
    alpha_centers = np.array([-2.0, -1.0])
    c2 = np.broadcast_to(np.array([0.4, 0.6]), (2, 2)).copy()
    c3 = np.broadcast_to(c2[:, :, None], (2, 2, 2)).copy()
    c4 = np.broadcast_to(c3[:, :, :, None], (2, 2, 2, 2)).copy()

    comp2 = hcr.Completeness2D(mag_centers, z_centers, c2)
    comp3 = hcr.Completeness3D(mag_centers, z_centers, fhost_centers, c3)
    comp4 = hcr.Completeness4D(mag_centers, z_centers, fhost_centers, alpha_centers, c4)
    z_query = np.array([0.5, 2.5])

    np.testing.assert_allclose(comp2(20.5, z_query), [0.3, 0.7])
    np.testing.assert_allclose(comp3(20.5, z_query, 0.5), [0.3, 0.7])
    np.testing.assert_allclose(comp4(20.5, z_query, 0.5, -1.5), [0.3, 0.7])
    np.testing.assert_allclose(comp2(22.0, z_query), [0.0, 0.0])


def test_get_completeness_function_4d_fhost_alpha_uses_mock_alpha_dataset(tmp_path):
    df_agn = _make_fake_agn_sample_with_fhost_alpha(alpha_center=-1.15)
    df_pre = _make_fake_agn_sample_with_fhost_alpha(n_agn=64, seed=456, alpha_center=-0.95)
    sim_file = tmp_path / "mock4d_alpha.h5"
    _write_fake_sim_file(sim_file, include_alpha=True, alpha_center=-0.95, include_fhost=True)

    completeness_params = hcr.get_completeness_function_4d_fhost_alpha(
        df_agn,
        sim_file=str(sim_file),
        plot=False,
        n_mag_bins=8,
        n_z_bins=8,
        n_fhost_bins=4,
        n_alpha_bins=5,
        sigma_mag=0.25,
        sigma_z_abs=0.25,
        sigma_fhost=0.2,
        sigma_alpha=0.4,
        fit_logL_max=99.0,
        df_agn_fhost_population=df_pre,
    )

    assert completeness_params[0].mode == "4d_fhost_alpha"
    alpha_model = completeness_params[-1]
    assert alpha_model["source"] == (
        "mock_h5_dataset:alpha_nu_lf_conversion_converted_to_alpha_lambda"
    )
    assert alpha_model["n_mock"] > 0
    assert abs(alpha_model["alpha_mean"] - (-0.95)) < 0.25
    host_model = completeness_params[-2]
    assert host_model["source"] == "mock_h5_dataset:f_host_2500_psf"
    assert host_model["observed_fit_source"] == "precut_f_host_2500_psf_vs_l2500"
    assert host_model["n_fit"] == len(df_pre)
    assert host_model["n_mock"] > 0


def test_get_completeness_function_4d_fhost_alpha_falls_back_to_observed_alpha(tmp_path):
    df_agn = _make_fake_agn_sample_with_fhost_alpha(alpha_center=-0.85)
    sim_file = tmp_path / "mock4d_no_alpha.h5"
    _write_fake_sim_file(sim_file, include_alpha=False)

    completeness_params = hcr.get_completeness_function_4d_fhost_alpha(
        df_agn,
        sim_file=str(sim_file),
        plot=False,
        n_mag_bins=8,
        n_z_bins=8,
        n_fhost_bins=4,
        n_alpha_bins=5,
        sigma_mag=0.25,
        sigma_z_abs=0.25,
        sigma_fhost=0.2,
        sigma_alpha=0.4,
    )

    alpha_model = completeness_params[-1]
    assert alpha_model["source"] == "observed_alpha_lambda"
    assert alpha_model["n_fit"] == len(df_agn)
    assert abs(alpha_model["alpha_mean"] - np.median(df_agn["alpha_lambda"])) < 0.05


@pytest.mark.parametrize("ambiguous_name", ("PL_slope", "alpha_nu"))
def test_mock_alpha_reader_rejects_ambiguous_legacy_slope_aliases(
    tmp_path,
    ambiguous_name,
):
    path = tmp_path / f"ambiguous_{ambiguous_name}.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(ambiguous_name, data=np.array([-1.5, -1.4]))

    with h5py.File(path, "r") as handle:
        values, source = hcr._read_mock_alpha_lambda(handle)

    assert values is None
    assert source is None


def test_mock_alpha_reader_retains_explicit_pre_schema4_alpha_lambda(tmp_path):
    path = tmp_path / "legacy_explicit_alpha_lambda.h5"
    expected = np.array([-1.5, -1.4])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("alpha_lambda", data=expected)

    with h5py.File(path, "r") as handle:
        values, source = hcr._read_mock_alpha_lambda(handle)

    np.testing.assert_array_equal(values, expected)
    assert source == "mock_h5_dataset:alpha_lambda"


def test_mock_alpha_attr_reader_uses_only_explicit_lf_conversion_alpha_nu():
    model = hcr._alpha_lambda_model_from_h5_attrs(
        {
            "alpha_nu_lf_conversion_parent_mean": -0.4,
            "alpha_nu_lf_conversion_parent_sigma": 0.25,
        }
    )

    assert model["alpha_mean"] == pytest.approx(-1.6)
    assert model["alpha_sigma"] == pytest.approx(0.25)
    assert model["source"] == (
        "mock_h5_attrs:alpha_nu_lf_conversion_converted_to_alpha_lambda"
    )


@pytest.mark.parametrize(
    "attrs",
    (
        {"alpha_nu_parent_mean": -0.4, "alpha_nu_parent_sigma": 0.25},
        {"alpha_nu_input_mean": -0.4, "alpha_nu_input_sigma": 0.25},
        {"alpha_nu_mean": -0.4, "alpha_nu_sigma": 0.25},
    ),
)
def test_mock_alpha_attr_reader_rejects_generic_alpha_nu_aliases(attrs):
    assert hcr._alpha_lambda_model_from_h5_attrs(attrs) is None


def test_save_mock_catalog_persists_explicit_lf_conversion_slope(tmp_path):
    out = tmp_path / "mock_with_alpha.h5"
    z = np.array([0.5, 1.0, 1.5])
    m_i = np.array([20.1, 21.2, 22.3])
    m2500 = np.array([19.9, 21.0, 22.1])
    alpha_nu = np.array([-0.8, -0.6, -0.4])

    save_mock_catalog(
        out,
        z,
        m_i,
        m2500,
        alpha_nu_lf_conversion_all=alpha_nu,
        alpha_nu_lf_conversion_parent_mean=-0.5,
        alpha_nu_lf_conversion_parent_sigma=0.3,
        lf_model="shen",
        shen_lf_mode="type1_attenuated",
    )

    with h5py.File(out, "r") as handle:
        np.testing.assert_allclose(
            handle["alpha_nu_lf_conversion"][:], alpha_nu
        )
        assert "alpha_lambda_lf_conversion" not in handle
        assert "alpha_lambda_lf_conversion_parent_mean" not in handle.attrs
        assert "alpha_lambda_lf_conversion_parent_sigma" not in handle.attrs
        assert handle.attrs["alpha_nu_lf_conversion_mean"] == pytest.approx(
            np.mean(alpha_nu)
        )
        assert handle.attrs["alpha_nu_lf_conversion_sigma"] == pytest.approx(
            np.std(alpha_nu, ddof=1)
        )
        assert handle.attrs["alpha_nu_lf_conversion_parent_mean"] == -0.5
        assert handle.attrs["alpha_nu_lf_conversion_parent_sigma"] == 0.3
        assert handle.attrs["thinning_probability"] == 1.0
        assert handle.attrs["mock_count_scale"] == 1.0
        assert handle.attrs["lf_model"] == "shen"
        assert handle.attrs["shen_lf_mode"] == "type1_attenuated"


def test_area_scaled_sampling_plan_and_histogram_normalization_are_invariant():
    plan = plan_area_scaled_mock_sampling(
        273.25,
        proposal_area_deg2=FULL_SKY_AREA_DEG2,
        oversample=8.0,
    )
    assert plan["effective_sampled_area_deg2"] == pytest.approx(8.0 * 273.25)
    assert plan["thinning_probability"] == pytest.approx(
        8.0 * 273.25 / FULL_SKY_AREA_DEG2
    )
    assert plan["mock_count_scale"] == pytest.approx(1.0 / 8.0)
    assert plan["realized_oversample"] == pytest.approx(8.0)

    observed = np.array([[4.0, 8.0], [3.0, 6.0]])
    target_mock = np.array([[8.0, 16.0], [6.0, 12.0]])
    mag_centers = np.array([19.0, 21.0])
    baseline = hcr._scaled_completeness_ratio(
        observed,
        target_mock,
        mag_centers,
        label="baseline",
        count_scale=1.0,
    )
    oversampled = hcr._scaled_completeness_ratio(
        observed,
        8.0 * target_mock,
        mag_centers,
        label="oversampled",
        count_scale=1.0 / 8.0,
    )
    np.testing.assert_allclose(oversampled, baseline)


def test_area_oversampling_reduces_parent_count_noise():
    rng = np.random.default_rng(123)
    expected_target_count = 100.0
    one_x = rng.poisson(expected_target_count, size=20_000)
    eight_x = rng.poisson(8.0 * expected_target_count, size=20_000) / 8.0

    assert np.std(eight_x) < 0.4 * np.std(one_x)


def test_area_scaled_mock_metadata_is_strictly_validated(tmp_path):
    path = tmp_path / "bad_area_metadata.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("z", data=np.array([1.0, 2.0]))
        handle.attrs["mock_count_scale"] = 0.5
        handle.attrs["target_area_deg2"] = 100.0
        handle.attrs["proposal_area_deg2"] = 1000.0
        handle.attrs["effective_sampled_area_deg2"] = 200.0
        handle.attrs["thinning_probability"] = 0.1
        handle.attrs["stored_object_count"] = 2

    with h5py.File(path, "r") as handle:
        with pytest.raises(ValueError, match="effective area inconsistent"):
            hcr._read_mock_count_scale(handle, path)


def test_generate_fresh_completeness_uses_scaled_full_sky_and_cache(
    tmp_path,
    monkeypatch,
):
    calls = {}
    z_all = np.array([0.5, 1.0])
    m_all = np.array([20.0, 21.0])
    m_2500_all = np.array([19.8, 20.8])
    alpha_nu_all = np.array([-0.5, -0.4])

    monkeypatch.setenv("QVC_HUBBLE_SHEN_LF_MODE", "type1_attenuated")

    def fake_build_completeness_lf(
        lf_model,
        *,
        shen_lf_mode,
        z_range,
        target_cosmology,
        progress=False,
    ):
        calls["lf_model"] = lf_model
        calls["shen_lf_mode"] = shen_lf_mode
        calls["lf_progress"] = progress
        return types.SimpleNamespace(
            model_id=lf_model,
            phi_log10=np.zeros((2, 2)),
            native_magnitude_grid=np.array([-24.0, -23.0]),
            redshift_grid=np.array([0.0, 1.0]),
            reference_wavelength_angstrom=2500.0,
            native_to_monochromatic_ab_offset=0.0,
        )

    monkeypatch.setattr(
        hubble_fit,
        "build_completeness_lf",
        fake_build_completeness_lf,
    )

    def fake_mock_m_per_zbin(
        phi_log10,
        m_grid,
        z_bins,
        area_deg2,
        *args,
        thinning_probability,
        **kwargs,
    ):
        calls["mock_calls"] = calls.get("mock_calls", 0) + 1
        calls["mock_area_deg2"] = area_deg2
        calls["mock_thinning_probability"] = thinning_probability
        calls["mock_progress"] = kwargs.get("progress")
        return (
            [],
            np.array([]),
            [],
            np.array([]),
            z_all,
            m_all,
            m_2500_all,
            np.array([0, 0]),
            alpha_nu_all,
        )

    monkeypatch.setattr(hubble_fit, "mock_m_per_zbin", fake_mock_m_per_zbin)

    output_path = hubble_fit.generate_fresh_completeness_sim_file(
        tmp_path,
        area_deg2=74.1,
        cache_dir=tmp_path / "cache",
    )

    assert Path(output_path).parent == tmp_path / "cache"
    assert calls["shen_lf_mode"] == "type1_attenuated"
    assert calls["lf_progress"] is True
    assert calls["mock_area_deg2"] == pytest.approx(hubble_fit.FULL_SKY_AREA_DEG2)
    assert calls["mock_thinning_probability"] == pytest.approx(
        4.0 * 74.1 / hubble_fit.FULL_SKY_AREA_DEG2
    )
    assert calls["mock_progress"] is True
    with h5py.File(output_path, "r") as handle:
        assert handle.attrs["target_area_deg2"] == pytest.approx(74.1)
        assert handle.attrs["proposal_area_deg2"] == pytest.approx(
            hubble_fit.FULL_SKY_AREA_DEG2
        )
        assert handle.attrs["effective_sampled_area_deg2"] == pytest.approx(
            4.0 * 74.1
        )
        assert handle.attrs["mock_count_scale"] == pytest.approx(1.0 / 4.0)
        assert handle.attrs["shen_lf_mode"] == "type1_attenuated"
        np.testing.assert_array_equal(
            handle["alpha_nu_lf_conversion"][:],
            alpha_nu_all,
        )
        assert "alpha_lambda_lf_conversion" not in handle

    calls_before_reuse = dict(calls)
    reused_path = hubble_fit.generate_fresh_completeness_sim_file(
        tmp_path,
        area_deg2=74.1,
        cache_dir=tmp_path / "cache",
    )
    assert reused_path == output_path
    assert calls == calls_before_reuse

    with h5py.File(output_path, "r+") as handle:
        handle.create_dataset(
            "alpha_lambda_lf_conversion",
            data=-handle["alpha_nu_lf_conversion"][:] - 2.0,
        )
    regenerated_path = hubble_fit.generate_fresh_completeness_sim_file(
        tmp_path,
        area_deg2=74.1,
        cache_dir=tmp_path / "cache",
    )
    assert regenerated_path == output_path
    assert calls["mock_calls"] == 2
    with h5py.File(regenerated_path, "r") as handle:
        assert "alpha_lambda_lf_conversion" not in handle

    with h5py.File(output_path, "r+") as handle:
        handle.attrs["mock_count_scale"] = 999.0
    regenerated_path = hubble_fit.generate_fresh_completeness_sim_file(
        tmp_path,
        area_deg2=74.1,
        cache_dir=tmp_path / "cache",
    )
    assert regenerated_path == output_path
    assert calls["mock_calls"] == 3
    with h5py.File(regenerated_path, "r") as handle:
        assert handle.attrs["mock_count_scale"] == pytest.approx(1.0 / 4.0)


@pytest.mark.parametrize(
    ("lf_model", "z_range", "expected_interpretation"),
    (
        (
            KULKARNI2019_TYPE1_MODEL1,
            (0.44, 3.9),
            {
                "low_redshift_extrapolation": True,
                "boss_excluded_interval_overlap": True,
                "model1_sharp_beta_feature_overlap": True,
            },
        ),
        (
            KULKARNI2019_TYPE1_MODEL1,
            (0.44, 2.2),
            {
                "low_redshift_extrapolation": True,
                "boss_excluded_interval_overlap": False,
                "model1_sharp_beta_feature_overlap": False,
            },
        ),
        (
            KULKARNI2019_TYPE1_MODEL1,
            (0.6, 3.5),
            {
                "low_redshift_extrapolation": False,
                "boss_excluded_interval_overlap": True,
                "model1_sharp_beta_feature_overlap": False,
            },
        ),
        (
            KULKARNI2019_TYPE1_MODEL1,
            (3.5, 4.0),
            {
                "low_redshift_extrapolation": False,
                "boss_excluded_interval_overlap": False,
                "model1_sharp_beta_feature_overlap": True,
            },
        ),
        (
            KULKARNI2019_TYPE1_MODEL2,
            (0.7, 3.16),
            {
                "low_redshift_extrapolation": False,
                "boss_excluded_interval_overlap": True,
                "model1_sharp_beta_feature_overlap": False,
            },
        ),
        (
            KULKARNI2019_TYPE1_MODEL3,
            (0.7, 2.0),
            {
                "low_redshift_extrapolation": False,
                "boss_excluded_interval_overlap": False,
                "model1_sharp_beta_feature_overlap": False,
            },
        ),
    ),
)
def test_kulkarni_requested_range_interpretation_is_cached(
    tmp_path,
    monkeypatch,
    lf_model,
    z_range,
    expected_interpretation,
):
    calls = {"build": 0, "mock": 0}
    monkeypatch.setenv(
        "QVC_HUBBLE_COMPLETENESS_MOCK_REQUIRE_FULL_OVERSAMPLE",
        "false",
    )

    def fake_build_completeness_lf(model_id, **kwargs):
        calls["build"] += 1
        return types.SimpleNamespace(
            model_id=model_id,
            phi_log10=np.zeros((2, 2)),
            native_magnitude_grid=np.array([-24.0, -23.0]),
            redshift_grid=np.asarray(z_range),
            reference_wavelength_angstrom=1450.0,
            native_to_monochromatic_ab_offset=0.0,
        )

    midpoint = 0.5 * (z_range[0] + z_range[1])

    def fake_mock_m_per_zbin(*args, **kwargs):
        calls["mock"] += 1
        return (
            [],
            np.array([2.0]),
            [],
            np.array([2]),
            np.array([midpoint, midpoint]),
            np.array([20.0, 21.0]),
            np.array([19.0, 23.0]),
            np.array([0, 0]),
            np.array([-1.5, -1.6]),
        )

    monkeypatch.setattr(
        hubble_fit,
        "build_completeness_lf",
        fake_build_completeness_lf,
    )
    monkeypatch.setattr(
        hubble_fit,
        "mock_m_per_zbin",
        fake_mock_m_per_zbin,
    )

    generation_kwargs = {
        "area_deg2": 1.0,
        "proposal_area_deg2": 10.0,
        "oversample": 4.0,
        "max_rows": 100,
        "cache_dir": tmp_path / lf_model,
        "lf_model": lf_model,
        "z_range": z_range,
    }
    output_path = hubble_fit.generate_fresh_completeness_sim_file(
        tmp_path,
        **generation_kwargs,
    )

    assert Path(output_path).name.startswith(f"{lf_model}_")
    with h5py.File(output_path, "r") as handle:
        metadata = json.loads(handle.attrs["lf_metadata_json"])
        config_hash = str(handle.attrs["config_hash"])
        assert handle.attrs["lf_model"] == lf_model
        assert Path(output_path).stem.endswith(config_hash[:16])
        assert metadata["requested_redshift_range"] == list(z_range)
        assert metadata["redshift_extrapolation"] is expected_interpretation[
            "low_redshift_extrapolation"
        ]
        assert metadata["sample_provenance"][
            "boss_dr9_excluded_redshift_interval"
        ] == [2.2, 3.5]
        assert (
            metadata["requested_range_interpretation"]
            == expected_interpretation
        )

    calls_after_first_generation = dict(calls)
    reused_path = hubble_fit.generate_fresh_completeness_sim_file(
        tmp_path,
        **generation_kwargs,
    )
    assert reused_path == output_path
    assert calls == calls_after_first_generation

    # The cache validator compares the complete serialized LF metadata, not
    # only the filename hash.  Corrupting one interpretation flag must force
    # regeneration and restore the authoritative requested-range provenance.
    with h5py.File(output_path, "r+") as handle:
        tampered = json.loads(handle.attrs["lf_metadata_json"])
        interpretation = tampered["requested_range_interpretation"]
        interpretation["boss_excluded_interval_overlap"] = not interpretation[
            "boss_excluded_interval_overlap"
        ]
        handle.attrs["lf_metadata_json"] = json.dumps(
            tampered,
            sort_keys=True,
            separators=(",", ":"),
        )

    regenerated_path = hubble_fit.generate_fresh_completeness_sim_file(
        tmp_path,
        **generation_kwargs,
    )
    assert regenerated_path == output_path
    assert calls["build"] == calls_after_first_generation["build"] + 1
    assert calls["mock"] == calls_after_first_generation["mock"] + 1
    with h5py.File(regenerated_path, "r") as handle:
        restored = json.loads(handle.attrs["lf_metadata_json"])
        assert (
            restored["requested_range_interpretation"]
            == expected_interpretation
        )


def test_generate_fresh_completeness_row_cap_updates_effective_area(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("QVC_HUBBLE_SHEN_LF_MODE", raising=False)
    monkeypatch.setattr(
        hubble_fit,
        "build_completeness_lf",
        lambda lf_model, **kwargs: types.SimpleNamespace(
            model_id=lf_model,
            phi_log10=np.zeros((2, 2)),
            native_magnitude_grid=np.array([-24.0, -23.0]),
            redshift_grid=np.array([0.0, 1.0]),
            reference_wavelength_angstrom=2500.0,
            native_to_monochromatic_ab_offset=0.0,
        ),
    )
    values = np.arange(10, dtype=float)

    def fake_mock(*args, thinning_probability, **kwargs):
        assert thinning_probability == pytest.approx(0.4)
        return (
            [],
            np.array([100.0]),
            [],
            np.array([10]),
            0.5 + values / 100.0,
            20.0 + values / 100.0,
            19.8 + values / 100.0,
            np.zeros(10, dtype=int),
            np.full(10, -1.5),
        )

    monkeypatch.setattr(hubble_fit, "mock_m_per_zbin", fake_mock)
    output_path = hubble_fit.generate_fresh_completeness_sim_file(
        tmp_path,
        area_deg2=100.0,
        proposal_area_deg2=1000.0,
        oversample=4.0,
        max_rows=4,
        cache_dir=tmp_path / "cache",
    )

    with h5py.File(output_path, "r") as handle:
        assert len(handle["z"]) == 4
        assert handle.attrs["stored_object_count"] == 4
        assert handle.attrs["thinning_probability"] == pytest.approx(0.16)
        assert handle.attrs["effective_sampled_area_deg2"] == pytest.approx(160.0)
        assert handle.attrs["realized_oversample"] == pytest.approx(1.6)
        assert handle.attrs["mock_count_scale"] == pytest.approx(0.625)


def test_strict_mock_oversampling_rejects_row_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("QVC_HUBBLE_COMPLETENESS_MOCK_REQUIRE_FULL_OVERSAMPLE", "true")
    monkeypatch.setattr(
        hubble_fit,
        "build_completeness_lf",
        lambda lf_model, **kwargs: types.SimpleNamespace(
            model_id=lf_model,
            phi_log10=np.zeros((2, 2)),
            native_magnitude_grid=np.array([-24.0, -23.0]),
            redshift_grid=np.array([0.0, 1.0]),
            reference_wavelength_angstrom=2500.0,
            native_to_monochromatic_ab_offset=0.0,
        ),
    )
    values = np.arange(10, dtype=float)
    monkeypatch.setattr(
        hubble_fit,
        "mock_m_per_zbin",
        lambda *args, **kwargs: (
            [],
            np.array([100.0]),
            [],
            np.array([10]),
            0.5 + values / 100.0,
            20.0 + values / 100.0,
            19.8 + values / 100.0,
            np.zeros(10, dtype=int),
            np.full(10, -1.5),
        ),
    )

    with pytest.raises(RuntimeError, match="strict oversampling is enabled"):
        hubble_fit.generate_fresh_completeness_sim_file(
            tmp_path,
            area_deg2=100.0,
            proposal_area_deg2=1000.0,
            oversample=4.0,
            max_rows=4,
            cache_dir=tmp_path / "cache",
        )


def test_get_completeness_function_3d_fhost_and_loglikelihood_smoke(tmp_path):
    df_agn = _make_fake_agn_sample_with_fhost()
    df_pantheon = pd.DataFrame(
        {
            "zHD": np.linspace(0.02, 0.8, 12),
            "zHEL": np.linspace(0.02, 0.8, 12),
            "m_b_corr": np.linspace(15.0, 18.0, 12),
            "IS_CALIBRATOR": np.zeros(12, dtype=int),
            "CEPH_DIST": np.full(12, -9.0),
            "MU_SH0ES_ERR_DIAG": np.full(12, 0.08),
        }
    )
    sim_file = tmp_path / "mock3d.h5"
    _write_fake_sim_file(sim_file)

    completeness_params = hcr.get_completeness_function_3d_fhost(df_agn, sim_file=str(sim_file), plot=False)
    assert completeness_params[0].mode == "3d_fhost"

    priors, model_labels, _ = hubble_model.get_model_params("FlatLambdaCDM", only_sna=False)
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in model_labels], dtype=float)

    agn_fields = hubble_model.agn_model_req_obs + hubble_model.agn_model_req_errs
    agn_fields += ("apparent_mag_2500", "apparent_mag_2500_err", "z", "z_err", "object_id", "f_host_2500_psf")
    agn_fields += (hcr.COMPLETENESS_MAG_COL, hcr.COMPLETENESS_MAG_ERR_COL)
    agn_data = {col: df_agn[col].to_numpy() for col in agn_fields}
    pantheon_data = {col: df_pantheon[col].to_numpy() for col in df_pantheon.columns}
    pivot_context = _build_pivot_context(df_agn)

    logl, blob = hubble_likelihood.log_likelihood(
        theta,
        agn_data=agn_data,
        pantheon_data=pantheon_data,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM",
        completeness_params=completeness_params,
        z_pivot_agn=hubble_fit.z_pivot_agn,
        agn_pivot_context=pivot_context,
        agn_calibrators_data=None,
        only_sna=False,
        use_full_cov=False,
    )

    assert np.isfinite(logl)
    assert blob.shape == (3, len(df_agn))


def test_get_completeness_function_3d_ignores_legacy_fhost_column(tmp_path):
    df_agn = _make_fake_agn_sample_with_fhost()
    df_agn["f_host_2500"] = np.nan
    sim_file = tmp_path / "mock3d_legacy_fhost_ignored.h5"
    _write_fake_sim_file(sim_file, include_fhost=False)

    completeness_params = hcr.get_completeness_function_3d_fhost(
        df_agn,
        sim_file=str(sim_file),
        plot=False,
        n_mag_bins=8,
        n_z_bins=8,
        n_fhost_bins=4,
        sigma_mag=0.25,
        sigma_z_abs=0.25,
        sigma_fhost=0.2,
        fit_logL_max=99.0,
    )

    assert completeness_params[0].mode == "3d_fhost"
    assert completeness_params[-1]["observed_fit_source"] == "fit_sample_f_host_2500_psf_vs_l2500"


def test_get_completeness_function_3d_fhost_fits_host_population_on_precut_sample(tmp_path):
    df_postcut = _make_fake_agn_sample_with_fhost(n_agn=12, seed=123)
    df_precut = _make_fake_agn_sample_with_fhost(n_agn=48, seed=456)
    sim_file = tmp_path / "mock3d_no_fhost.h5"
    _write_fake_sim_file(sim_file, include_fhost=False)

    completeness_params = hcr.get_completeness_function_3d_fhost(
        df_postcut,
        sim_file=str(sim_file),
        plot=False,
        n_mag_bins=8,
        n_z_bins=8,
        n_fhost_bins=4,
        sigma_mag=0.25,
        sigma_z_abs=0.25,
        sigma_fhost=0.2,
        fit_logL_max=99.0,
        df_agn_fhost_population=df_precut,
    )

    host_model = completeness_params[-1]
    assert host_model["source"] == "observed_fhost_model"
    assert host_model["observed_fit_source"] == "precut_f_host_2500_psf_vs_l2500"
    assert host_model["n_observed_population"] == len(df_precut)
    assert host_model["n_fit"] == len(df_precut)


def test_fhost_population_model_records_approximate_v1_provenance():
    df_postcut = _make_fake_agn_sample_with_fhost(n_agn=12, seed=123)
    df_precut = _make_fake_agn_sample_with_fhost(n_agn=48, seed=456)
    df_precut["f_host_2500_psf_is_approximate"] = True
    df_precut["f_host_2500_psf_proxy_edge_clamped"] = False
    df_precut.loc[df_precut.index[:7], "f_host_2500_psf_proxy_edge_clamped"] = True

    host_model = hcr._fit_fhost_population_model(
        df_postcut,
        df_precut,
        fit_logL_max=99.0,
    )

    assert host_model["f_host_2500_psf_is_approximate"] is True
    assert host_model["n_approximate_f_host_2500_psf"] == len(df_precut)
    assert host_model["n_edge_clamped_f_host_2500_psf"] == 7
    assert host_model["observed_fit_source"].endswith(
        "_approximate_v1_psf_band_interpolation"
    )
