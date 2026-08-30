import h5py
import numpy as np
import pandas as pd
import pytest

from qvc.spectra.catalog_hdf5 import (
    ALPHA_NU_BLUE_WAVELENGTH_ANGSTROM,
    ALPHA_NU_RED_WAVELENGTH_ANGSTROM,
    GRAHSP_ATTENUATION_OPTICAL_INDEX,
    JOINT_PSF_PHOTOMETRY_BANDS,
    JOINT_PSF_PHOTOMETRY_FORMAT,
    JOINT_POSTERIOR_DRAW_FIELDS,
    SPECTRA_CATALOG_FORMAT,
    SPECTRA_CATALOG_FORMAT_V1,
    SPECTRA_CATALOG_FORMAT_V2,
    read_spectra_catalog_hdf5,
    write_spectra_catalog_hdf5,
)


def _write_v1_catalog(path, *, include_draw_group=True):
    with h5py.File(path, "w") as handle:
        handle.attrs["qvc_spectra_catalog_format"] = SPECTRA_CATALOG_FORMAT_V1
        handle.attrs["psf_agn_fraction_draw_count"] = 64
        catalog = handle.create_group("catalog")
        catalog.create_dataset("object_id", data=[b"101", b"102"])
        catalog.create_dataset("z", data=[1.0, 3.0])
        if include_draw_group:
            draw_group = handle.create_group("psf_agn_fraction_draws")
            draw_group.create_dataset("bands", data=[b"u", b"g", b"r", b"i", b"z"])
            values = np.full((2, 64, 5), np.nan, dtype=np.float32)
            values[0, :2] = 0.7
            values[1, :1] = 0.8
            draw_group.create_dataset("values", data=values)
            draw_group.create_dataset("valid_count", data=[2, 1])


def _write_v2_catalog(path):
    with h5py.File(path, "w") as handle:
        handle.attrs["qvc_spectra_catalog_format"] = SPECTRA_CATALOG_FORMAT_V2
        handle.attrs["psf_agn_fraction_draw_count"] = 64
        handle.attrs["f_host_2500_psf_draw_count"] = 64
        catalog = handle.create_group("catalog")
        catalog.create_dataset("object_id", data=[b"101", b"102"])
        catalog.create_dataset("fit_ok", data=[True, False])
        psf = handle.create_group("psf_agn_fraction_draws")
        psf.create_dataset("bands", data=[b"u", b"g", b"r", b"i", b"z"])
        psf_values = np.full((2, 64, 5), np.nan, dtype=np.float32)
        psf_values[0, :2] = 0.7
        psf.create_dataset("values", data=psf_values)
        psf.create_dataset("valid_count", data=[2, 0])
        host = handle.create_group("f_host_2500_psf_draws")
        host_values = np.full((2, 64), np.nan, dtype=np.float32)
        host_values[0, :2] = [0.3, 0.2]
        host.create_dataset("values", data=host_values)
        host.create_dataset("valid_count", data=[2, 0])


def _valid_joint_payload(row_count=2):
    values = {
        name: np.full((row_count, 64), np.nan, dtype=np.float32)
        for name in JOINT_POSTERIOR_DRAW_FIELDS
    }
    counts = np.zeros(row_count, dtype=np.int16)
    indices = np.full((row_count, 64), -1, dtype=np.int32)
    source_counts = np.zeros(row_count, dtype=np.int32)
    if row_count:
        counts[0] = 2
        indices[0, :2] = [3, 7]
        source_counts[0] = 10
        values["f_host_2500_psf"][0, :2] = [0.3, 0.2]
        values["m_2500_dereddened"][0, :2] = [20.0, 20.1]
        values["a_2500_galaxy"][0, :2] = [0.1, 0.2]
        values["a_2500_internal"][0, :2] = [0.2, 0.3]
        values["a_2500_total"][0, :2] = [0.3, 0.5]
        values["m_2500_attenuated_model"][0, :2] = [20.3, 20.6]
        intrinsic = np.array([-0.5, -0.7], dtype=np.float32)
        values["alpha_nu_intrinsic_1450_2500"][0, :2] = intrinsic
        denominator = np.log10(
            ALPHA_NU_RED_WAVELENGTH_ANGSTROM
            / ALPHA_NU_BLUE_WAVELENGTH_ANGSTROM
        )
        ratio = (
            ALPHA_NU_BLUE_WAVELENGTH_ANGSTROM
            / ALPHA_NU_RED_WAVELENGTH_ANGSTROM
        ) ** GRAHSP_ATTENUATION_OPTICAL_INDEX
        values["alpha_nu_attenuated_1450_2500"][0, :2] = (
            intrinsic
            - 0.4 * values["a_2500_total"][0, :2] * (ratio - 1.0) / denominator
        )
    return values, counts, indices, source_counts


def _add_v3_scalar_schema(frame, joint=None, counts=None):
    frame = frame.copy()
    frame["mw_deredden_applied"] = True
    frame["joint_posterior_draw_source"] = "synthetic_test"
    defaults = {
        "f_host_2500_psf": 0.25,
        "alpha_nu_intrinsic_1450_2500": -0.5,
        "alpha_nu_attenuated_1450_2500": -1.0,
        "m_2500_dereddened": 20.0,
        "m_2500_attenuated_model": 20.3,
        "a_2500_galaxy": 0.1,
        "a_2500_internal": 0.2,
        "a_2500_total": 0.3,
    }
    success = frame["fit_ok"].to_numpy(dtype=bool)
    for name in JOINT_POSTERIOR_DRAW_FIELDS:
        summary = np.full(len(frame), np.nan, dtype=float)
        for row_index in np.flatnonzero(success):
            count = int(counts[row_index]) if counts is not None else 0
            summary[row_index] = (
                float(np.median(joint[name][row_index, :count]))
                if joint is not None and count
                else defaults[name]
            )
        frame[name] = summary
        for suffix in ("err", "err_lower", "err_upper"):
            frame[f"{name}_{suffix}"] = np.where(success, 0.0, np.nan)
    return frame


def _write_v3_catalog(path, *, include_joint_psf_photometry=True):
    frame = pd.DataFrame(
        {
            "object_id": ["101", "102"],
            "fit_ok": [True, False],
            "fit_backend": ["jaxsedfit_joint", "jaxsedfit_joint"],
            "z": [1.2, np.nan],
            "error_message": ["", "failed"],
        }
    )
    psf = np.full((2, 64, 5), np.nan, dtype=np.float32)
    psf[0, :2] = np.array([[0.7] * 5, [0.8] * 5], dtype=np.float32)
    joint, counts, indices, source_counts = _valid_joint_payload()
    frame = _add_v3_scalar_schema(frame, joint, counts)
    joint_psf_photometry = None
    joint_psf_provenance = None
    if include_joint_psf_photometry:
        joint_psf_photometry = np.full(
            (2, 64, len(JOINT_PSF_PHOTOMETRY_BANDS)),
            np.nan,
            dtype=np.float32,
        )
        joint_psf_photometry[0, :2] = np.asarray(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [1.1, 2.1, 3.1, 4.1, 5.1]],
            dtype=np.float32,
        )
        joint_psf_provenance = {
            "prediction_source": "synthetic_saved_bundle_prediction",
            "jaxsedfit_git_commit": "a" * 40,
        }
    write_spectra_catalog_hdf5(
        path,
        frame,
        psf,
        np.array([2, 0]),
        joint_posterior_draws=joint,
        joint_posterior_valid_count=counts,
        joint_posterior_index=indices,
        joint_posterior_source_draw_count=source_counts,
        joint_posterior_selection_seed=13,
        joint_psf_photometry_draws=joint_psf_photometry,
        joint_psf_photometry_provenance=joint_psf_provenance,
    )


def test_spectra_catalog_v3_round_trip_preserves_catalog_and_joint_draws(tmp_path):
    path = tmp_path / "spectra.h5"
    _write_v3_catalog(path)

    result = read_spectra_catalog_hdf5(path)

    assert result.catalog_format == SPECTRA_CATALOG_FORMAT
    assert result.frame["object_id"].tolist() == ["101", "102"]
    assert result.frame["fit_ok"].tolist() == [True, False]
    assert result.frame["error_message"].tolist() == ["", "failed"]
    assert np.isnan(result.frame.loc[1, "z"])
    np.testing.assert_array_equal(result.valid_count, [2, 0])
    np.testing.assert_allclose(result.fraction_draws[0, :2], [[0.7] * 5, [0.8] * 5])
    np.testing.assert_array_equal(result.joint_posterior_valid_count, [2, 0])
    np.testing.assert_array_equal(result.joint_posterior_index[0, :2], [3, 7])
    np.testing.assert_array_equal(result.joint_posterior_source_draw_count, [10, 0])
    assert result.joint_posterior_selection_seed == 13
    assert set(result.joint_posterior_draws) == set(JOINT_POSTERIOR_DRAW_FIELDS)
    np.testing.assert_allclose(result.f_host_2500_psf_draws[0, :2], [0.3, 0.2])
    assert result.f_host_2500_psf_draws is result.joint_posterior_draws["f_host_2500_psf"]
    assert result.bands == ("u", "g", "r", "i", "z")
    assert result.joint_psf_photometry_draws.shape == (2, 64, 5)
    with h5py.File(path, "r") as handle:
        assert handle.attrs["qvc_spectra_catalog_format"] == SPECTRA_CATALOG_FORMAT
        assert handle.attrs["joint_posterior_selection_seed"] == 13
        assert set(handle) == {
            "catalog",
            "psf_agn_fraction_draws",
            "joint_posterior_draws",
            "joint_psf_photometry_draws",
        }


def test_spectra_catalog_v3_legacy_host_capture_metadata_requires_opt_in(tmp_path):
    path = tmp_path / "legacy_host_capture_metadata.h5"
    _write_v3_catalog(path)
    with h5py.File(path, "r+") as handle:
        del handle.attrs["f_host_2500_psf_capture_model"]
        del handle.attrs["f_host_2500_psf_fwhm_arcsec"]

    with pytest.raises(ValueError, match="f_host_2500_psf_capture_model"):
        read_spectra_catalog_hdf5(path)

    with pytest.warns(
        RuntimeWarning,
        match="missing host-capture metadata",
    ):
        result = read_spectra_catalog_hdf5(
            path,
            allow_legacy_v3_host_capture_metadata=True,
        )

    assert len(result.frame) == 2
    assert result.joint_posterior_draws["m_2500_dereddened"].shape == (2, 64)


@pytest.mark.parametrize(
    ("attribute", "bad_value"),
    [
        ("f_host_2500_psf_capture_model", "different_model"),
        ("f_host_2500_psf_fwhm_arcsec", 2.0),
    ],
)
def test_spectra_catalog_v3_legacy_opt_in_rejects_conflicting_metadata(
    tmp_path,
    attribute,
    bad_value,
):
    path = tmp_path / f"conflicting_{attribute}.h5"
    _write_v3_catalog(path)
    with h5py.File(path, "r+") as handle:
        handle.attrs[attribute] = bad_value

    with pytest.raises(ValueError, match=attribute):
        read_spectra_catalog_hdf5(
            path,
            allow_legacy_v3_host_capture_metadata=True,
        )


def test_spectra_catalog_v3_writer_rejects_missing_required_fitted_photometry(tmp_path):
    with pytest.raises(ValueError, match="requires joint_psf_photometry_draws"):
        _write_v3_catalog(
            tmp_path / "missing_color.h5",
            include_joint_psf_photometry=False,
        )


def test_required_joint_psf_photometry_round_trip_and_alignment(tmp_path):
    path = tmp_path / "spectra_psf_photometry.h5"
    _write_v3_catalog(path, include_joint_psf_photometry=True)

    result = read_spectra_catalog_hdf5(path)

    assert result.joint_psf_photometry_bands == JOINT_PSF_PHOTOMETRY_BANDS
    assert result.joint_psf_photometry_draws.shape == (2, 64, 5)
    np.testing.assert_allclose(
        result.joint_psf_photometry_draws[0, :2, 1],
        [2.0, 2.1],
    )
    assert np.all(np.isnan(result.joint_psf_photometry_draws[0, 2:]))
    assert result.joint_psf_photometry_provenance[
        "jaxsedfit_git_commit"
    ] == "a" * 40
    metadata_only = read_spectra_catalog_hdf5(
        path,
        include_fraction_draws=False,
    )
    assert metadata_only.joint_psf_photometry_draws.shape == (0, 64, 5)
    with h5py.File(path, "r") as handle:
        group = handle["joint_psf_photometry_draws"]
        assert group.attrs["format"] == JOINT_PSF_PHOTOMETRY_FORMAT
        assert group.attrs["posterior_alignment"] == (
            "/joint_posterior_draws/posterior_index"
        )
        assert set(group) == {"bands", "values_mjy"}


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("negative", "strictly positive"),
        ("padding", "NaN-padded"),
        ("alignment", "posterior_alignment"),
    ],
)
def test_joint_psf_photometry_reader_rejects_malformed_extension(
    tmp_path,
    tamper,
    message,
):
    path = tmp_path / f"bad_psf_photometry_{tamper}.h5"
    _write_v3_catalog(path, include_joint_psf_photometry=True)
    with h5py.File(path, "r+") as handle:
        group = handle["joint_psf_photometry_draws"]
        if tamper == "negative":
            group["values_mjy"][0, 0, 1] = -1.0
        elif tamper == "padding":
            group["values_mjy"][0, 2, 1] = 1.0
        else:
            group.attrs["posterior_alignment"] = "independent"

    with pytest.raises(ValueError, match=message):
        read_spectra_catalog_hdf5(path)
        assert set(handle["joint_posterior_draws"]) == {
            *JOINT_POSTERIOR_DRAW_FIELDS,
            "posterior_index",
            "source_draw_count",
            "valid_count",
        }


def test_spectra_catalog_v3_reader_revalidates_paired_physical_identities(tmp_path):
    path = tmp_path / "tampered.h5"
    _write_v3_catalog(path)
    with h5py.File(path, "r+") as handle:
        handle["joint_posterior_draws/a_2500_total"][0, 0] += 0.1

    with pytest.raises(ValueError, match="A_2500,total"):
        read_spectra_catalog_hdf5(path)


def test_spectra_catalog_v3_rejects_misaligned_or_unindexed_draws(tmp_path):
    path = tmp_path / "bad_indices.h5"
    _write_v3_catalog(path)
    with h5py.File(path, "r+") as handle:
        handle["joint_posterior_draws/posterior_index"][0, :2] = [7, 3]

    with pytest.raises(ValueError, match="strictly increasing"):
        read_spectra_catalog_hdf5(path)


def test_spectra_catalog_v3_rejects_success_without_joint_draws(tmp_path):
    path = tmp_path / "missing_success_draws.h5"
    frame = _add_v3_scalar_schema(
        pd.DataFrame({"object_id": ["101"], "fit_ok": [True]})
    )
    psf = np.full((1, 64, 5), np.nan, dtype=np.float32)
    joint = {
        name: np.full((1, 64), np.nan, dtype=np.float32)
        for name in JOINT_POSTERIOR_DRAW_FIELDS
    }
    with pytest.raises(ValueError, match="Successful spectral rows require"):
        write_spectra_catalog_hdf5(
            path,
            frame,
            psf,
            [0],
            joint_posterior_draws=joint,
            joint_posterior_valid_count=[0],
            joint_posterior_index=np.full((1, 64), -1),
            joint_posterior_source_draw_count=[0],
            joint_posterior_selection_seed=13,
        )


def test_spectra_catalog_v3_rejects_missing_alpha_scalar_provenance(tmp_path):
    path = tmp_path / "missing_alpha_scalar.h5"
    frame = _add_v3_scalar_schema(
        pd.DataFrame({"object_id": ["101"], "fit_ok": [True]})
    ).drop(columns=["alpha_nu_intrinsic_1450_2500"])
    psf = np.full((1, 64, 5), np.nan, dtype=np.float32)
    joint, counts, indices, source_counts = _valid_joint_payload(row_count=1)

    with pytest.raises(ValueError, match="scalar/provenance columns"):
        write_spectra_catalog_hdf5(
            path,
            frame,
            psf,
            [0],
            joint_posterior_draws=joint,
            joint_posterior_valid_count=counts,
            joint_posterior_index=indices,
            joint_posterior_source_draw_count=source_counts,
            joint_posterior_selection_seed=13,
        )


def test_spectra_catalog_v2_requires_separate_explicit_opt_in(tmp_path):
    path = tmp_path / "spectra_v2.h5"
    _write_v2_catalog(path)

    with pytest.raises(ValueError, match="allow_v2=True"):
        read_spectra_catalog_hdf5(path)
    with pytest.warns(RuntimeWarning, match="explicit v2 compatibility"):
        result = read_spectra_catalog_hdf5(path, allow_v2=True)

    assert result.catalog_format == SPECTRA_CATALOG_FORMAT_V2
    np.testing.assert_allclose(result.f_host_2500_psf_draws[0, :2], [0.3, 0.2])
    np.testing.assert_array_equal(result.f_host_2500_psf_valid_count, [2, 0])
    assert result.joint_posterior_draws == {}
    np.testing.assert_array_equal(result.joint_posterior_valid_count, [0, 0])
    assert result.joint_posterior_selection_seed is None


def test_spectra_catalog_v1_requires_opt_in_and_has_no_native_host_draws(tmp_path):
    path = tmp_path / "spectra_v1.h5"
    _write_v1_catalog(path)

    with pytest.raises(ValueError, match="allow_v1=True"):
        read_spectra_catalog_hdf5(path)

    with pytest.warns(RuntimeWarning, match="explicit v1 compatibility"):
        result = read_spectra_catalog_hdf5(path, allow_v1=True)

    assert result.catalog_format == SPECTRA_CATALOG_FORMAT_V1
    np.testing.assert_array_equal(result.f_host_2500_psf_valid_count, [0, 0])
    assert result.f_host_2500_psf_draws.shape == (2, 64)
    assert np.all(np.isnan(result.f_host_2500_psf_draws))
    assert result.joint_posterior_draws == {}


def test_spectra_catalog_v1_still_validates_required_groups(tmp_path):
    path = tmp_path / "malformed_v1.h5"
    _write_v1_catalog(path, include_draw_group=False)

    with pytest.warns(RuntimeWarning, match="explicit v1 compatibility"):
        with pytest.raises(ValueError, match="missing required groups"):
            read_spectra_catalog_hdf5(path, allow_v1=True)


@pytest.mark.parametrize(
    ("malformation", "error_match"),
    [
        ("invalid_count", "invalid valid_count"),
        ("invalid_shape", "invalid fraction-draw shape"),
    ],
)
def test_spectra_catalog_v1_validates_draw_counts_and_shapes(
    tmp_path,
    malformation,
    error_match,
):
    path = tmp_path / f"malformed_v1_{malformation}.h5"
    _write_v1_catalog(path)
    with h5py.File(path, "r+") as handle:
        draw_group = handle["psf_agn_fraction_draws"]
        if malformation == "invalid_count":
            draw_group["valid_count"][0] = 65
        else:
            del draw_group["values"]
            draw_group.create_dataset(
                "values",
                data=np.full((2, 63, 5), np.nan, dtype=np.float32),
            )

    with pytest.warns(RuntimeWarning, match="explicit v1 compatibility"):
        with pytest.raises(ValueError, match=error_match):
            read_spectra_catalog_hdf5(path, allow_v1=True)
