import h5py
import inspect
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from qvc.hubble.hubble_completeness_refactored import Completeness2D
from qvc.hubble.hubble_likelihood import completeness_loglike_for_data
from qvc.hubble.program_color_completeness import (
    EBOSS_FEATURE_NAMES,
    PREPARED_CATALOG_SCHEMA,
    EmpiricalFeatureSupport,
    LogisticColorHead,
    apply_paired_flux_noise,
    assert_paired_nuclear_state,
    build_artifact_from_prepared_catalog,
    build_hubble_completeness_map,
    color_support_stencil_mask,
    decode_target_marks,
    eboss_target_features,
    hash_completeness_2d,
    leave_one_channel_out_closure,
    marginalize_paired_color_completeness,
    plot_matched_hubble_residual_change,
    read_color_completeness_artifact,
    validate_cut_manifest,
    write_color_completeness_artifact,
    signed_luptitude,
)
from qvc.hubble.prepare_program_color_catalog import (
    WISE_AB_MINUS_VEGA,
    draw_host_capture_fraction,
    load_host_capture_calibration,
    validate_mock_host_capture_fraction,
    wise_ab_to_vega_nanomaggy,
)
from qvc.hubble.cuts import build_sdss_target_selection_mask
from qvc.hubble.hubble_utils import populate_sdss_target_metadata


def _products(host=0.5, nohost=0.4):
    shape = (3, 3)
    return {
        "C_color_host": np.full(shape, host),
        "C_color_nohost": np.full(shape, nohost),
        "delta_C_host_color": np.full(shape, host - nohost),
        "uncertainty_host": np.full(shape, 0.01),
        "uncertainty_nohost": np.full(shape, 0.01),
        "n_eff_host": np.full(shape, 250.0),
        "n_eff_nohost": np.full(shape, 250.0),
        "support_host": np.ones(shape, bool),
        "support_nohost": np.ones(shape, bool),
        "out_of_support_fraction_host": np.zeros(shape),
        "out_of_support_fraction_nohost": np.zeros(shape),
    }


def test_cut_stages_and_target_marks():
    stages = {name: [] for name in (
        "intrinsic_support", "target_eligibility", "target_selection",
        "downstream_survival", "hd_analysis",
    )}
    stages["target_selection"] = ["legacy_good"]
    assert validate_cut_manifest(stages)["target_selection"] == ["legacy_good"]
    stages["hd_analysis"] = ["legacy_good"]
    with pytest.raises(ValueError, match="exactly one stage"):
        validate_cut_manifest(stages)
    marks = decode_target_marks(
        color_targeted=[1, 0, 1, 0],
        alternative_targeted=[0, 1, 1, 1],
        color_eligible=[1, 1, 1, 0],
    )
    assert marks.tolist() == ["color_only", "alt_only", "both", "unknown"]


def test_signed_eboss_features_wise_units_and_no_redshift_head():
    sdss_flux = np.array([[-2.0, 1.0, 2.0, 3.0, 4.0]])
    wise_flux = np.array([[-1.0, 2.0]])
    assert np.all(np.isfinite(signed_luptitude(sdss_flux, np.ones(5))))
    features = eboss_target_features(
        sdss_flux, np.ones_like(sdss_flux), wise_flux,
        np.ones_like(wise_flux), np.zeros_like(wise_flux, dtype=bool),
        np.ones(5), np.ones(2),
    )
    assert features.shape == (1, len(EBOSS_FEATURE_NAMES))
    assert np.all(np.isfinite(features))
    converted = wise_ab_to_vega_nanomaggy(np.ones((1, 2)))
    np.testing.assert_allclose(
        converted[0], np.power(10.0, 0.4 * WISE_AB_MINUS_VEGA)
    )
    with pytest.raises(ValueError, match="redshift"):
        LogisticColorHead("eboss", ("z",), np.zeros(1), np.ones(1), np.zeros(1), 0.0, 1.0)


def test_paired_state_and_noise_use_one_empirical_error_vector():
    value = np.array([20.0, 21.0])
    assert_paired_nuclear_state(value, value.copy(), value - 40, value - 40)
    with pytest.raises(ValueError, match="modified m_hd"):
        assert_paired_nuclear_state(value, value + 1e-6, value, value)

    host = np.array([[1.0, -0.5]])
    nohost = np.array([[0.2, -0.5]])
    eps = np.array([[1.0, -1.0]])

    error = np.array([[0.2, 0.3]])
    nh, nn = apply_paired_flux_noise(host, nohost, error, eps)
    np.testing.assert_allclose(nh - host, error * eps)
    np.testing.assert_allclose(nn - nohost, error * eps)
    np.testing.assert_allclose(nh - host, nn - nohost)


def test_host_capture_validation_uses_capture_parameter_not_agn_fraction(tmp_path):
    path = tmp_path / "jaxsedfit.h5"
    names = np.array([f"qso-{index:03d}" for index in range(240)])
    with h5py.File(path, "w") as handle:
        catalog = handle.create_group("catalog")
        catalog.create_dataset(
            "sdss_name", data=names.astype("S"),
        )
        catalog.create_dataset(
            "z", data=np.r_[np.full(120, 0.7), np.full(120, 1.4)]
        )
        catalog.create_dataset(
            "host_capture_group_fraction",
            data=np.r_[np.linspace(0.15, 0.65, 120), np.linspace(0.10, 0.55, 120)],
        )
        catalog.create_dataset(
            "host_capture_group_fraction_err", data=np.full(240, 0.04)
        )
        # Deliberately omit f_AGN_psf_*: it is variable-AGN/total PSF and is
        # not a valid host-fraction calibration field.
    calibration = load_host_capture_calibration(path, names)
    assert "observed_host_fraction" not in calibration["low"]

    redshift = np.r_[np.full(20_000, 0.7), np.full(20_000, 1.4)]
    capture = draw_host_capture_fraction(
        redshift, calibration, np.random.default_rng(31)
    )
    diagnostics = validate_mock_host_capture_fraction(
        redshift, capture, calibration
    )
    assert set(diagnostics) == {"low", "high"}
    assert diagnostics["low"]["max_abs_difference"] < 0.03

    shifted = np.clip(capture + 0.15, 0.0, 1.0)
    with pytest.raises(ValueError, match="distribution closure"):
        validate_mock_host_capture_fraction(redshift, shifted, calibration)


def test_head_support_weighted_marginalization():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(100, 3))
    y = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(float)
    head = LogisticColorHead.fit("eboss", ("f_u", "f_g", "err_u"), x, y)
    probability = head.predict(x)
    assert np.all((probability >= 0) & (probability <= 1))
    support = EmpiricalFeatureSupport.fit(x, ["all"] * len(x))
    assert np.mean(support.contains(x, ["all"] * len(x))) >= 0.99
    assert not support.contains([[100, 100, 100]], ["all"])[0]

    product = marginalize_paired_color_completeness(
        m_bin=np.zeros(300, int), z_bin=np.zeros(300, int), n_m=1, n_z=1,
        weights=np.ones(300), probability_host=np.full(300, 0.7),
        probability_nohost=np.full(300, 0.5), supported_host=np.ones(300, bool),
        supported_nohost=np.ones(300, bool),
    )
    assert product["C_color_host"][0, 0] == pytest.approx(0.7)
    assert product["n_eff_host"][0, 0] == pytest.approx(300)


def test_channel_closure_is_a_hard_failure_when_testable():
    rng = np.random.default_rng(13)
    features = rng.normal(size=(120, 2))
    channels = np.repeat(["radio", "variability", "xray"], 40)
    success = np.concatenate((np.r_[np.ones(35), np.zeros(5)],
                              np.r_[np.ones(5), np.zeros(35)],
                              np.tile([0.0, 1.0], 20)))
    with pytest.raises(ValueError, match="closure failed"):
        leave_one_channel_out_closure(
            "eboss", ("flux_g", "flux_i"), features, success, channels,
            strict=True,
        )


def test_artifact_hash_modes_and_support_preflight(tmp_path):
    grid = np.array([19.0, 20.0, 21.0])
    old = Completeness2D(grid, np.array([0.5, 1.0, 1.5]), np.full((3, 3), 0.6), (19, 21))
    path = tmp_path / "color.h5"
    write_color_completeness_artifact(
        path, magnitude_grid=grid, redshift_grid=np.array([0.5, 1.0, 1.5]),
        products=_products(), metadata={"old_completeness_hash": hash_completeness_2d(old), "heads_frozen": True},
    )
    artifact = read_color_completeness_artifact(path)
    assert build_hubble_completeness_map("old", old_completeness=old) is old
    point = (np.array([20.0]), np.array([1.0]))
    ratio = build_hubble_completeness_map("host-removal", old_completeness=old, artifact=artifact,
                                         hd_magnitude=point[0], hd_redshift=point[1])
    assert ratio(20.0, 1.0) == pytest.approx(0.6 * 0.4 / 0.5)
    with pytest.raises(ValueError, match="Unknown completeness mode"):
        build_hubble_completeness_map(
            "color-host", old_completeness=old, artifact=artifact
        )
    with h5py.File(path, "r+") as handle:
        handle["C_color_host"][0, 0] = 0.9
    with pytest.raises(ValueError, match="hash mismatch"):
        read_color_completeness_artifact(path)


def test_host_removal_uses_old_last_center_value_at_upper_boundary(tmp_path):
    old_magnitude = np.array([19.0, 20.0, 21.0])
    redshift = np.array([0.5, 1.0, 1.5])
    old_values = np.repeat(np.array([[0.2], [0.5], [0.8]]), 3, axis=1)
    old = Completeness2D(
        old_magnitude,
        redshift,
        old_values,
        magnitude_support=(18.5, 21.5),
    )
    artifact_magnitude = np.array([18.5, 20.0, 21.5])
    path = tmp_path / "color-edge.h5"
    write_color_completeness_artifact(
        path,
        magnitude_grid=artifact_magnitude,
        redshift_grid=redshift,
        products=_products(),
        metadata={
            "old_completeness_hash": hash_completeness_2d(old),
            "heads_frozen": True,
        },
    )
    artifact = read_color_completeness_artifact(path)

    result = build_hubble_completeness_map(
        "host-removal",
        old_completeness=old,
        artifact=artifact,
    )

    assert result(21.5, 1.0) == pytest.approx(0.8 * 0.4 / 0.5)


def test_shared_color_support_cut_uses_complete_interpolation_stencil(tmp_path):
    grid = np.array([19.0, 20.0, 21.0])
    old = Completeness2D(grid, np.array([0.5, 1.0, 1.5]), np.full((3, 3), 0.6), (19, 21))
    products = _products()
    products["support_host"][0, 0] = False
    path = tmp_path / "support.h5"
    write_color_completeness_artifact(
        path, magnitude_grid=grid, redshift_grid=np.array([0.5, 1.0, 1.5]),
        products=products, metadata={"old_completeness_hash": hash_completeness_2d(old)},
    )
    artifact = read_color_completeness_artifact(path)
    mask = color_support_stencil_mask(
        artifact, [19.5, 20.5, 30.0], [0.75, 1.25, 1.0]
    )
    assert mask.tolist() == [False, True, False]


def test_cosmology_likelihood_has_only_frozen_2d_completeness_contract():
    parameters = set(inspect.signature(completeness_loglike_for_data).parameters)
    forbidden = {"colors", "target_flags", "qsogen", "support", "color_head", "alpha_lambda", "f_host_2500_psf"}
    assert parameters.isdisjoint(forbidden)


def test_prepared_catalog_builder_trains_one_eboss_head(tmp_path):
    prepared = tmp_path / "prepared.h5"
    output = tmp_path / "artifact.h5"
    rng = np.random.default_rng(19)
    text_dtype = h5py.string_dtype("utf-8")
    stages = {name: [] for name in (
        "intrinsic_support", "target_eligibility", "target_selection",
        "downstream_survival", "hd_analysis",
    )}
    with h5py.File(prepared, "w") as handle:
        handle.attrs["schema"] = PREPARED_CATALOG_SCHEMA
        handle.attrs["hubble_cut_configuration_json"] = '{}'
        handle.attrs["cut_manifest_json"] = __import__("json").dumps(stages)
        handle.attrs["opportunity_rules_json"] = json.dumps({"program": "eboss"})
        handle.attrs["old_completeness_hash"] = "old-map-hash"
        handle.attrs["target_provenance_json"] = json.dumps(
            {"assignment": "target_flags_and_selection_provenance"}
        )
        handle.attrs["input_catalog_hashes_json"] = json.dumps({"dr16q": "abc"})
        handle.attrs["host_capture_calibration_json"] = json.dumps({"source": "test"})
        handle.attrs["feature_transform_json"] = json.dumps({"schema": "test"})
        handle.attrs["closure_diagnostics_json"] = json.dumps([])
        handle.create_dataset("magnitude_grid", data=[19.0, 20.0, 21.0])
        handle.create_dataset("redshift_grid", data=[0.5, 1.0, 1.5])
        train = handle.create_group("training/eboss")
        x = rng.normal(size=(240, len(EBOSS_FEATURE_NAMES)))
        y = (x[:, 0] + rng.normal(scale=0.5, size=240) > 0).astype(np.int8)
        train.create_dataset("features", data=x)
        train.create_dataset("success", data=y)
        train.create_dataset("marks", data=np.where(y, b"both", b"alt_only"), dtype=text_dtype)
        train.create_dataset("patterns", data=np.full(240, b"wise:00"), dtype=text_dtype)
        train.create_dataset("alternative_channel", data=np.full(240, b"var_s82"), dtype=text_dtype)
        train.attrs["feature_names_json"] = json.dumps(EBOSS_FEATURE_NAMES)

        mock = handle.create_group("mock/eboss")
        m_bin = np.repeat(np.arange(3), 3 * 225)
        z_bin = np.tile(np.repeat(np.arange(3), 225), 3)
        n = len(m_bin)
        base = rng.normal(scale=0.5, size=(n, len(EBOSS_FEATURE_NAMES)))
        shift = np.zeros(len(EBOSS_FEATURE_NAMES)); shift[0] = 0.1
        mock.create_dataset("features_host", data=base)
        mock.create_dataset("features_nohost", data=base + shift)
        mock.create_dataset("patterns_host", data=np.full(n, b"wise:00"), dtype=text_dtype)
        mock.create_dataset("patterns_nohost", data=np.full(n, b"wise:00"), dtype=text_dtype)
        mock.create_dataset("m_bin", data=m_bin)
        mock.create_dataset("z_bin", data=z_bin)
        mock.create_dataset("weights", data=np.ones(n))
        mock.create_dataset("m_hd_host", data=19.0 + m_bin)
        mock.create_dataset("m_hd_nohost", data=19.0 + m_bin)
        mock.create_dataset("luminosity_host", data=np.full(n, 45.0))
        mock.create_dataset("luminosity_nohost", data=np.full(n, 45.0))
        mock.create_dataset("observing_state_id", data=np.arange(n))
        mock.create_dataset("noise_normal", data=rng.normal(size=(n, 7)))
        mock.create_dataset("host_capture_fraction", data=np.full(n, 0.5))
    digest = build_artifact_from_prepared_catalog(prepared, output)
    artifact = read_color_completeness_artifact(output)
    assert artifact.content_hash == digest
    assert np.all(artifact.arrays["n_eff_host"] >= 200)
    assert set(artifact.metadata["heads_frozen"]) == {"eboss"}
    assert not any(name.startswith("program_") for name in artifact.arrays)
    assert artifact.metadata["host_counterfactual"].startswith("eBOSS photometry-only")


def test_matched_runner_builds_missing_artifact_before_two_modes():
    runner = (Path(__file__).resolve().parents[1] / "run_hubble_color_modes.xonsh").read_text()
    assert "QVC_COLOR_COMPLETENESS_PREPARED_CATALOG" in runner
    assert "prepare_program_color_catalog" in runner
    assert "wang2026_type1_lade_a" in runner
    assert "QVC_QSOGEN_PATH" in runner
    assert "QVC_HUBBLE_SDSS_TARGET_METADATA_H5" in runner
    assert '"QVC_HUBBLE_COLOR_SUPPORT_CUT": "true"' in runner
    assert "program_color_completeness build" in runner
    assert runner.index("prepare_program_color_catalog") < runner.index(
        "program_color_completeness build"
    )
    assert '--require-program eboss' in runner
    assert '"eboss-color-sensitivity"' in runner
    assert 'for mode in ("old", "host-removal")' in runner
    assert '"color-host"' not in runner
    assert "plot-residual-change" in runner


def test_matched_residual_change_plot_uses_identical_objects(tmp_path):
    old = pd.DataFrame({
        "object_id": ["a", "b", "c"],
        "z": [0.5, 1.0, 1.5],
        "residuals": [0.2, -0.1, 0.3],
    })
    host_removal = pd.DataFrame({
        "object_id": ["c", "a", "b"],
        "z": [1.5, 0.5, 1.0],
        "residuals": [0.25, 0.1, -0.05],
    })
    old_path = tmp_path / "old.csv"
    host_path = tmp_path / "host.csv"
    output = tmp_path / "residual-change.png"
    old.to_csv(old_path, index=False)
    host_removal.to_csv(host_path, index=False)
    assert plot_matched_hubble_residual_change(old_path, host_path, output) == output
    assert output.stat().st_size > 0
    summary = pd.read_csv(output.with_suffix(".csv"))
    assert summary["n"].sum() == 3

    host_removal.loc[0, "object_id"] = "different"
    host_removal.to_csv(host_path, index=False)
    with pytest.raises(ValueError, match="identical object IDs"):
        plot_matched_hubble_residual_change(old_path, host_path, output)


def test_shared_eboss_sensitivity_cut_uses_all_main_program_objects():
    bit = lambda value: np.uint64(1) << np.uint64(value)
    frame = pd.DataFrame({
        "SDSS_SURVEY": ["eboss"] * 6,
        "SDSS_PROGRAMNAME": [
            "eboss", "eboss", "ELG_SGC", "eboss", "eboss", "eboss"
        ],
        "SDSS_EBOSS_TARGET0": [bit(10), 0, bit(10), bit(10), bit(10), -1],
        "SDSS_EBOSS_TARGET1": [0, bit(9), 0, bit(9) | bit(14), bit(18), 0],
        "SDSS_EBOSS_TARGET2": [0] * 6,
    })
    mask, _ = build_sdss_target_selection_mask(
        frame, "eboss-color-sensitivity"
    )
    # CORE, alternative, ambiguous, disqualified, and missing-bit rows all
    # remain in the Hubble sample when their main-eBOSS provenance matches.
    assert mask.tolist() == [True, True, False, True, True, True]


def test_local_target_metadata_join_replaces_sentinels_without_replacing_fits(tmp_path):
    path = tmp_path / "target_metadata.h5"
    text_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as handle:
        catalog = handle.create_group("catalog")
        catalog.create_dataset("object_id", data=np.array(["a", "b"], object), dtype=text_dtype)
        catalog.create_dataset("SDSS_SURVEY", data=np.array(["eboss", "eboss"], object), dtype=text_dtype)
        catalog.create_dataset("SDSS_PROGRAMNAME", data=np.array(["eboss", "eboss"], object), dtype=text_dtype)
        catalog.create_dataset("SDSS_SPECOBJ_MATCHED", data=[True, True])
        catalog.create_dataset("SDSS_EBOSS_TARGET0", data=[1 << 10, 0])
        catalog.create_dataset("SDSS_EBOSS_TARGET1", data=[0, 1 << 9])
        catalog.create_dataset("SDSS_EBOSS_TARGET2", data=[0, 0])
    frame = pd.DataFrame({
        "object_id": ["a", "b"], "fit_value": [1.0, 2.0],
        "SDSS_SURVEY": ["missing", "missing"],
    })
    joined = populate_sdss_target_metadata(frame, path)
    assert joined["fit_value"].tolist() == [1.0, 2.0]
    assert joined["SDSS_SURVEY"].tolist() == ["eboss", "eboss"]
    assert joined["SDSS_EBOSS_TARGET1"].tolist() == [0, 1 << 9]
