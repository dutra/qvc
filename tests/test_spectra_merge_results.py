import sys

import h5py
import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

from qvc.spectra.catalog_hdf5 import (
    ALPHA_NU_BLUE_WAVELENGTH_ANGSTROM,
    ALPHA_NU_RED_WAVELENGTH_ANGSTROM,
    GRAHSP_ATTENUATION_OPTICAL_INDEX,
    JOINT_PSF_PHOTOMETRY_BANDS,
    JOINT_POSTERIOR_DRAW_FIELDS,
    JOINT_POSTERIOR_SCALAR_SUMMARY_FIELDS,
    SPECTRA_CATALOG_FORMAT_V2,
    write_spectra_catalog_hdf5,
)
from qvc.spectra import merge_results
from qvc.spectra.merge_results import (
    deduplicate_h5_catalog,
    enrich_h5_catalog_rows,
    load_and_merge_h5,
)


def _joint_row_values(value):
    value = float(value)
    a_galaxy = 0.01 + 0.01 * value
    a_internal = 0.02 + 0.01 * value
    a_total = a_galaxy + a_internal
    alpha_intrinsic = -0.7 + 0.2 * value
    alpha_denominator = np.log10(
        ALPHA_NU_RED_WAVELENGTH_ANGSTROM
        / ALPHA_NU_BLUE_WAVELENGTH_ANGSTROM
    )
    attenuation_ratio = (
        ALPHA_NU_BLUE_WAVELENGTH_ANGSTROM
        / ALPHA_NU_RED_WAVELENGTH_ANGSTROM
    ) ** GRAHSP_ATTENUATION_OPTICAL_INDEX
    m_dereddened = 20.0 + value
    return {
        "f_host_2500_psf": 1.0 - value,
        "alpha_nu_intrinsic_1450_2500": alpha_intrinsic,
        "alpha_nu_attenuated_1450_2500": (
            alpha_intrinsic
            - 0.4 * a_total * (attenuation_ratio - 1.0) / alpha_denominator
        ),
        "m_2500_dereddened": m_dereddened,
        "m_2500_attenuated_model": m_dereddened + a_total,
        "a_2500_galaxy": a_galaxy,
        "a_2500_internal": a_internal,
        "a_2500_total": a_total,
    }


def _write_v3_catalog(
    path,
    frame,
    draw_values,
    *,
    selection_seed=3,
    joint_psf_values=None,
    joint_psf_provenance=None,
):
    """Write a small physically consistent v3 shard for merge tests."""

    frame = pd.DataFrame(frame).reset_index(drop=True)
    if len(draw_values) != len(frame):
        raise ValueError("draw_values must have one entry per test-catalog row")
    if "fit_ok" not in frame:
        frame["fit_ok"] = [value is not None for value in draw_values]
    if "mw_deredden_applied" not in frame:
        frame["mw_deredden_applied"] = True
    if "joint_posterior_draw_source" not in frame:
        frame["joint_posterior_draw_source"] = [
            "test_saved_bundle" if value is not None else ""
            for value in draw_values
        ]
    fraction_draws = np.full((len(frame), 64, 5), np.nan, dtype=np.float32)
    joint_draws = {
        name: np.full((len(frame), 64), np.nan, dtype=np.float32)
        for name in JOINT_POSTERIOR_DRAW_FIELDS
    }
    valid_count = np.zeros(len(frame), dtype=np.int16)
    posterior_index = np.full((len(frame), 64), -1, dtype=np.int32)
    source_draw_count = np.zeros(len(frame), dtype=np.int32)
    for index, value in enumerate(draw_values):
        if value is None:
            continue
        value = float(value)
        fraction_draws[index, 0] = value
        valid_count[index] = 1
        posterior_index[index, 0] = int(round(10.0 * value))
        source_draw_count[index] = 250
        for name, field_value in _joint_row_values(value).items():
            joint_draws[name][index, 0] = field_value
    for name in JOINT_POSTERIOR_DRAW_FIELDS:
        summaries = [
            np.nan if value is None else _joint_row_values(value)[name]
            for value in draw_values
        ]
        if name not in frame:
            frame[name] = summaries
        for suffix in ("_err", "_err_lower", "_err_upper"):
            column = f"{name}{suffix}"
            if column not in frame:
                frame[column] = [
                    np.nan if value is None else 0.0 for value in draw_values
                ]
    assert set(JOINT_POSTERIOR_SCALAR_SUMMARY_FIELDS).issubset(frame.columns)
    if joint_psf_values is None:
        joint_psf_values = draw_values
    if len(joint_psf_values) != len(frame):
        raise ValueError("joint_psf_values must match the test frame")
    joint_psf_photometry = np.full(
        (len(frame), 64, len(JOINT_PSF_PHOTOMETRY_BANDS)),
        np.nan,
        dtype=np.float32,
    )
    for row_index, value in enumerate(joint_psf_values):
        if value is not None:
            joint_psf_photometry[row_index, 0] = float(value)
    if joint_psf_provenance is None:
        joint_psf_provenance = {
            "prediction_source": "synthetic_test",
            "jaxsedfit_git_commit": "a" * 40,
        }
    write_spectra_catalog_hdf5(
        path,
        frame,
        fraction_draws,
        valid_count,
        joint_posterior_draws=joint_draws,
        joint_posterior_valid_count=valid_count,
        joint_posterior_index=posterior_index,
        joint_posterior_source_draw_count=source_draw_count,
        joint_posterior_selection_seed=selection_seed,
        joint_psf_photometry_draws=joint_psf_photometry,
        joint_psf_photometry_provenance=joint_psf_provenance,
    )


def _write_shard(path, object_ids, draw_values, *, selection_seed=3):
    frame = pd.DataFrame(
        {
            "object_id": object_ids,
            "run_label": ["fiducial"] * len(object_ids),
            "fit_ok": [True] * len(object_ids),
            "fit_backend": ["jaxsedfit_joint"] * len(object_ids),
            "z": np.arange(len(object_ids), dtype=float),
        }
    )
    _write_v3_catalog(
        path,
        frame,
        draw_values,
        selection_seed=selection_seed,
    )


def test_h5_merge_deduplicates_last_row_with_its_matching_draws(tmp_path):
    first = tmp_path / "chunk0000.h5"
    second = tmp_path / "chunk0001.h5"
    _write_shard(first, ["1", "2"], [0.1, 0.2])
    _write_shard(second, ["2", "3"], [0.8, 0.3])

    merged = load_and_merge_h5(
        [str(first), str(second)],
        expected_n=2,
        dedup_keys=["object_id", "run_label"],
    )

    assert merged.frame["object_id"].tolist() == ["1", "2", "3"]
    np.testing.assert_allclose(merged.fraction_draws[:, 0, 0], [0.1, 0.8, 0.3])
    np.testing.assert_allclose(
        merged.f_host_2500_psf_draws[:, 0], [0.9, 0.2, 0.7]
    )
    np.testing.assert_array_equal(merged.valid_count, [1, 1, 1])
    np.testing.assert_array_equal(merged.joint_posterior_valid_count, [1, 1, 1])
    np.testing.assert_array_equal(merged.joint_posterior_index[:, 0], [1, 8, 3])
    np.testing.assert_array_equal(
        merged.joint_posterior_source_draw_count, [250, 250, 250]
    )
    assert merged.joint_posterior_selection_seed == 3
    expected_values = [0.1, 0.8, 0.3]
    for name in JOINT_POSTERIOR_DRAW_FIELDS:
        np.testing.assert_allclose(
            merged.joint_posterior_draws[name][:, 0],
            [_joint_row_values(value)[name] for value in expected_values],
        )
        assert np.all(np.isnan(merged.joint_posterior_draws[name][:, 1:]))


def test_h5_catalog_can_be_deduplicated_after_single_merge(tmp_path):
    first = tmp_path / "chunk0000.h5"
    second = tmp_path / "chunk0001.h5"
    _write_shard(first, ["1", "2"], [0.1, 0.2])
    _write_shard(second, ["2", "3"], [0.8, 0.3])

    before = load_and_merge_h5(
        [str(first), str(second)],
        expected_n=2,
        dedup_keys=[],
    )
    merged = deduplicate_h5_catalog(before, ["object_id", "run_label"])

    assert len(before.frame) == 4
    assert merged.frame["object_id"].tolist() == ["1", "2", "3"]
    np.testing.assert_allclose(merged.fraction_draws[:, 0, 0], [0.1, 0.8, 0.3])
    np.testing.assert_array_equal(merged.valid_count, [1, 1, 1])


def test_h5_main_loads_shards_once_then_deduplicates_in_memory(
    monkeypatch, tmp_path
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_shard(run_dir / "chunk0000.h5", ["1", "2"], [0.1, 0.2])
    _write_shard(run_dir / "chunk0001.h5", ["2", "3"], [0.8, 0.3])
    output = tmp_path / "merged.h5"
    calls = []
    real_load = merge_results.load_and_merge_h5

    def counting_load(*args, **kwargs):
        calls.append((args, kwargs))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(merge_results, "load_and_merge_h5", counting_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_results",
            "--base-dir",
            str(tmp_path),
            "--skip-populate-sdss",
            "--out",
            str(output),
            "run",
        ],
    )

    merge_results.main()

    assert len(calls) == 1
    assert calls[0][1]["dedup_keys"] == []
    merged = merge_results.read_spectra_catalog_hdf5(output)
    assert merged.frame["object_id"].tolist() == ["1", "2", "3"]
    np.testing.assert_allclose(merged.fraction_draws[:, 0, 0], [0.1, 0.8, 0.3])
    assert merged.joint_posterior_selection_seed == 3
    np.testing.assert_array_equal(merged.joint_posterior_index[:, 0], [1, 8, 3])
    for name in JOINT_POSTERIOR_DRAW_FIELDS:
        np.testing.assert_allclose(
            merged.joint_posterior_draws[name][:, 0],
            [
                _joint_row_values(value)[name]
                for value in (0.1, 0.8, 0.3)
            ],
        )


def test_h5_merge_unions_optional_columns_and_accepts_numeric_dtype_promotion(tmp_path):
    first = tmp_path / "chunk0000.h5"
    second = tmp_path / "chunk0001.h5"

    first_frame = pd.DataFrame(
        {
            "object_id": ["1"],
            "run_label": ["fiducial"],
            "fit_ok": [True],
            "fit_backend": ["jaxsedfit_joint"],
            "z": [1.0],
            "n_photometry": [5.0],
            "jqf_line_amp_Lya_std": [0.2],
        }
    )

    # Reorder shared fields, omit Lya, add CIV, and use integer n_photometry.
    second_frame = pd.DataFrame(
        {
            "object_id": ["2"],
            "fit_backend": ["jaxsedfit_joint"],
            "fit_ok": [True],
            "run_label": ["fiducial"],
            "z": [2.0],
            "n_photometry": [5],
            "jqf_line_amp_CIV_std": [0.3],
        }
    )
    _write_v3_catalog(first, first_frame, [0.2])
    _write_v3_catalog(second, second_frame, [0.2])

    merged = load_and_merge_h5([str(first), str(second)])

    assert merged.frame.columns.tolist()[: len(first_frame.columns)] == list(
        first_frame.columns
    )
    assert merged.frame.columns.tolist()[-1] == "jqf_line_amp_CIV_std"
    assert merged.frame["n_photometry"].tolist() == [5.0, 5.0]
    assert merged.frame["jqf_line_amp_Lya_std"].tolist()[0] == pytest.approx(0.2)
    assert np.isnan(merged.frame["jqf_line_amp_Lya_std"].tolist()[1])
    assert np.isnan(merged.frame["jqf_line_amp_CIV_std"].tolist()[0])
    assert merged.frame["jqf_line_amp_CIV_std"].tolist()[1] == pytest.approx(0.3)


def test_h5_merge_rejects_incompatible_shared_column_dtype(tmp_path):
    first = tmp_path / "chunk0000.h5"
    second = tmp_path / "chunk0001.h5"
    _write_shard(first, ["1"], [0.1])

    frame = pd.DataFrame(
        {
            "object_id": ["2"],
            "run_label": ["fiducial"],
            "fit_ok": [True],
            "fit_backend": ["jaxsedfit_joint"],
            "z": ["not-numeric"],
        }
    )
    _write_v3_catalog(second, frame, [0.2])

    with pytest.raises(
        ValueError,
        match="Incompatible scalar catalog dtype for column 'z'",
    ):
        load_and_merge_h5([str(first), str(second)])


def test_h5_merge_rejects_corrupt_shard(tmp_path):
    valid = tmp_path / "chunk0000.h5"
    corrupt = tmp_path / "chunk0001.h5"
    _write_shard(valid, ["1"], [0.1])
    corrupt.write_bytes(b"not an HDF5 file")

    with pytest.raises(OSError):
        load_and_merge_h5([str(valid), str(corrupt)])


def test_h5_merge_rejects_different_joint_selection_seeds(tmp_path):
    first = tmp_path / "chunk0000.h5"
    second = tmp_path / "chunk0001.h5"
    _write_shard(first, ["1"], [0.1], selection_seed=3)
    _write_shard(second, ["2"], [0.2], selection_seed=4)

    with pytest.raises(ValueError, match="different joint posterior seeds"):
        load_and_merge_h5([str(first), str(second)])


def test_h5_merge_rejects_mixed_catalog_formats(tmp_path):
    first = tmp_path / "chunk0000.h5"
    second = tmp_path / "chunk0001.h5"
    _write_shard(first, ["1"], [0.1])
    _write_shard(second, ["2"], [0.2])
    with h5py.File(second, "r+") as handle:
        handle.attrs["qvc_spectra_catalog_format"] = SPECTRA_CATALOG_FORMAT_V2

    with pytest.raises(ValueError, match="has format 'qvc_spectra_catalog_v2'"):
        load_and_merge_h5([str(first), str(second)])


def test_h5_merge_rejects_incompatible_joint_semantics(tmp_path):
    first = tmp_path / "chunk0000.h5"
    second = tmp_path / "chunk0001.h5"
    _write_shard(first, ["1"], [0.1])
    _write_shard(second, ["2"], [0.2])
    with h5py.File(second, "r+") as handle:
        handle.attrs["alpha_nu_definition"] = "incompatible-test-definition"

    with pytest.raises(ValueError, match="incompatible alpha_nu_definition"):
        load_and_merge_h5([str(first), str(second)])


def test_h5_merge_preserves_required_joint_psf_photometry_alignment(tmp_path):
    first = tmp_path / "chunk0000.h5"
    second = tmp_path / "chunk0001.h5"
    _write_v3_catalog(
        first,
        pd.DataFrame({"object_id": ["1"], "fit_ok": [True], "z": [1.0]}),
        [0.1],
        joint_psf_values=[1.5],
    )
    _write_v3_catalog(
        second,
        pd.DataFrame({"object_id": ["2"], "fit_ok": [True], "z": [2.0]}),
        [0.2],
        joint_psf_values=[2.5],
    )

    merged = load_and_merge_h5([str(first), str(second)])

    assert merged.joint_psf_photometry_bands == JOINT_PSF_PHOTOMETRY_BANDS
    np.testing.assert_allclose(
        merged.joint_psf_photometry_draws[:, 0, 1],
        [1.5, 2.5],
    )
    assert np.all(np.isnan(merged.joint_psf_photometry_draws[:, 1:]))


def test_h5_merge_ignores_and_reports_prediction_provenance_mismatch(
    tmp_path, capsys
):
    first = tmp_path / "chunk0000.h5"
    failed = tmp_path / "chunk0001.h5"
    reference_provenance = {
        "prediction_source": "fresh_fit_prediction",
        "jaxsedfit_git_commit": "a" * 40,
    }
    _write_v3_catalog(
        first,
        pd.DataFrame({"object_id": ["1"], "fit_ok": [True], "z": [1.0]}),
        [0.1],
        joint_psf_provenance=reference_provenance,
    )
    _write_v3_catalog(
        failed,
        pd.DataFrame({"object_id": ["2"], "fit_ok": [False], "z": [2.0]}),
        [None],
        joint_psf_provenance={
            "prediction_source": "fit_attempt_no_valid_draws",
            "jaxsedfit_git_commit": "a" * 40,
        },
    )

    merged = load_and_merge_h5([str(first), str(failed)])

    assert merged.frame["object_id"].tolist() == ["1", "2"]
    np.testing.assert_array_equal(merged.joint_posterior_valid_count, [1, 0])
    assert merged.joint_psf_photometry_provenance == reference_provenance
    report = capsys.readouterr().out
    assert "Ignored 1 joint PSF photometry prediction provenance mismatch" in report
    assert str(failed) in report
    assert "no valid draws" in report
    assert "fit_attempt_no_valid_draws" in report


def test_h5_merge_rejects_v3_shard_missing_mandatory_joint_psf_photometry(tmp_path):
    first = tmp_path / "chunk0000.h5"
    second = tmp_path / "chunk0001.h5"
    _write_v3_catalog(
        first,
        pd.DataFrame({"object_id": ["1"], "fit_ok": [True], "z": [1.0]}),
        [0.1],
        joint_psf_values=[1.5],
    )
    _write_v3_catalog(
        second,
        pd.DataFrame({"object_id": ["2"], "fit_ok": [True], "z": [2.0]}),
        [0.2],
    )
    with h5py.File(second, "r+") as handle:
        del handle["joint_psf_photometry_draws"]

    with pytest.raises(ValueError, match="missing required groups"):
        load_and_merge_h5([str(first), str(second)])


def test_h5_enrichment_restores_row_and_draw_alignment(tmp_path):
    shard = tmp_path / "chunk0000.h5"
    _write_shard(shard, ["1", "2"], [0.1, 0.2])
    catalog = load_and_merge_h5([str(shard)])

    def reverse_and_enrich(rows):
        return [dict(row, sdss_field="added") for row in reversed(rows)]

    enriched = enrich_h5_catalog_rows(catalog, reverse_and_enrich)

    assert enriched.frame["object_id"].tolist() == ["1", "2"]
    assert enriched.frame["sdss_field"].tolist() == ["added", "added"]
    np.testing.assert_allclose(enriched.fraction_draws[:, 0, 0], [0.1, 0.2])
    np.testing.assert_array_equal(enriched.joint_posterior_index[:, 0], [1, 2])
    assert enriched.joint_posterior_selection_seed == 3
    for name in JOINT_POSTERIOR_DRAW_FIELDS:
        np.testing.assert_allclose(
            enriched.joint_posterior_draws[name],
            catalog.joint_posterior_draws[name],
            equal_nan=True,
        )


def test_populate_sdss_run2d_from_fits_copies_survey_and_targeting_metadata(
    tmp_path,
):
    specobj_path = tmp_path / "specObj-test.fits"
    large_target_mask = np.int64(2**60 + 3)
    Table(
        {
            "PLATE": np.array([1, 4], dtype=np.int32),
            "MJD": np.array([3, 6], dtype=np.int32),
            "FIBERID": np.array([2, 5], dtype=np.int32),
            "RUN2D": ["v5_13_2", "v5_13_2"],
            "SURVEY": ["boss", "eboss"],
            "INSTRUMENT": ["BOSS", "BOSS"],
            "PROGRAMNAME": ["boss", "eboss"],
            "SOURCETYPE": ["QSO", "QSO1_VAR_S82"],
            "TARGETTYPE": ["SCIENCE", "SCIENCE"],
            "CHUNK": ["boss1", "eboss2"],
            "PLATERUN": ["boss-run", "eboss-run"],
            "TARGETOBJID": ["123", "456"],
            "THING_ID": np.array([10, 20], dtype=np.int32),
            "BOSS_TARGET1": np.array([large_target_mask, 0], dtype=np.int64),
            "EBOSS_TARGET1": np.array([0, 4096], dtype=np.int64),
            "ANCILLARY_TARGET1": np.array([8, 16], dtype=np.int64),
        }
    ).write(specobj_path)
    quasars = [
        {
            "object_id": "eboss-object",
            "plate": 4,
            "mjd": 6,
            "fiber": 5,
            "SDSS_SURVEY": "stale",
        },
        {"object_id": "boss-object", "plate": 1, "mjd": 3, "fiber": 2},
        {
            "object_id": "unmatched-object",
            "plate": 9,
            "mjd": 9,
            "fiber": 9,
            "SDSS_SURVEY": "preexisting",
            "SDSS_BOSS_TARGET1": 99,
        },
    ]

    result = pd.DataFrame.from_records(
        merge_results.populate_sdss_run2d_from_fits(quasars, specobj_path)
    )

    assert result["object_id"].tolist() == [
        "eboss-object",
        "boss-object",
        "unmatched-object",
    ]
    assert result["SDSS_SPECOBJ_MATCHED"].tolist() == [True, True, False]
    assert (
        result["SDSS_SURVEY"]
        .astype(object)
        .where(result["SDSS_SURVEY"].notna(), None)
        .tolist()
        == ["eboss", "boss", "preexisting"]
    )
    assert result["SDSS_PROGRAMNAME"].iloc[:2].tolist() == ["eboss", "boss"]
    assert result["SDSS_SOURCETYPE"].iloc[:2].tolist() == ["QSO1_VAR_S82", "QSO"]
    assert result["SDSS_TARGETOBJID"].iloc[:2].tolist() == ["456", "123"]
    assert result["SDSS_EBOSS_TARGET1"].tolist() == [4096, 0, -1]
    assert result["SDSS_BOSS_TARGET1"].tolist() == [0, large_target_mask, 99]
    assert result["SDSS_ANCILLARY_TARGET1"].tolist() == [16, 8, -1]
    # Optional columns absent from the small test FITS still receive a stable schema.
    assert result["SDSS_EBOSS_TARGET2"].tolist() == [-1, -1, -1]

    output_path = tmp_path / "enriched.h5"
    _write_v3_catalog(output_path, result, [None] * len(result))
    reloaded = merge_results.read_spectra_catalog_hdf5(output_path).frame
    assert reloaded["SDSS_SURVEY"].tolist() == ["eboss", "boss", "preexisting"]
    assert reloaded["SDSS_SPECOBJ_MATCHED"].tolist() == [True, True, False]
    assert reloaded["SDSS_BOSS_TARGET1"].tolist() == [0, large_target_mask, 99]
