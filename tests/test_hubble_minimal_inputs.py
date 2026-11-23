import json

import h5py
import numpy as np
import pytest

from qvc.hubble.minimal_inputs import (
    FORMAT_ATTR,
    LIGHT_CURVE_FIELDS,
    LIGHT_CURVE_FORMAT,
    MANIFEST_ATTR,
    NOISE_MODEL_FIELD,
    SPECTRA_FIELDS,
    SPECTRA_FORMAT,
    export_minimal_hubble_inputs,
    read_minimal_spectra_hdf5,
    validate_minimal_light_curve_hdf5,
)


STRING_FIELDS = {"object_id", "sdss_name", "fit_backend", "psd_bpl_ref_band"}
BOOL_FIELDS = {"fit_ok", "psd_ls_valid", "psd_ls_fixed_valid"}


def _write_field(group, name, values):
    if name in STRING_FIELDS:
        group.create_dataset(name, data=values, dtype=h5py.string_dtype("utf-8"))
    else:
        group.create_dataset(name, data=values)


def _source_values(name, ids):
    size = len(ids)
    if name == "object_id":
        return ids
    if name == "sdss_name":
        return [f"SDSS-{value}" for value in ids]
    if name == "fit_backend":
        return ["jaxsedfit_joint"] * size
    if name == "psd_bpl_ref_band":
        return ["g"] * size
    if name in BOOL_FIELDS:
        return np.ones(size, dtype=bool)
    return np.arange(size, dtype=float) + 0.25


def _write_sources(tmp_path, *, duplicate_light_ids=False):
    light_path = tmp_path / "light.h5"
    spectra_path = tmp_path / "spectra.h5"
    light_ids = ["2", "1", "2"] if duplicate_light_ids else ["2", "1", "3"]
    with h5py.File(light_path, "w") as handle:
        for name in LIGHT_CURVE_FIELDS:
            _write_field(handle, name, _source_values(name, light_ids))
    with h5py.File(spectra_path, "w") as handle:
        catalog = handle.create_group("catalog")
        spectra_ids = ["1", "2", "4"]
        for name in SPECTRA_FIELDS:
            values = _source_values(name, spectra_ids)
            if name == "fit_ok":
                values = np.asarray([True, True, False])
            _write_field(catalog, name, values)
    return light_path, spectra_path


def test_export_writes_exact_versioned_manifests_and_matched_order(tmp_path):
    light_source, spectra_source = _write_sources(tmp_path)
    light_output = tmp_path / "minimal_light.h5"
    spectra_output = tmp_path / "minimal_spectra.h5"

    count = export_minimal_hubble_inputs(
        light_source, spectra_source, light_output, spectra_output
    )

    assert count == 2
    with h5py.File(light_output, "r") as handle:
        assert handle.attrs[FORMAT_ATTR] == LIGHT_CURVE_FORMAT
        assert json.loads(handle.attrs[MANIFEST_ATTR]) == list(LIGHT_CURVE_FIELDS)
        assert set(handle) == set(LIGHT_CURVE_FIELDS)
        assert handle["object_id"].asstr()[...].tolist() == ["2", "1"]
        assert "light_curve_posterior_draws" not in handle
        validate_minimal_light_curve_hdf5(handle)
    with h5py.File(spectra_output, "r") as handle:
        assert handle.attrs[FORMAT_ATTR] == SPECTRA_FORMAT
        assert json.loads(handle.attrs[MANIFEST_ATTR]) == [*SPECTRA_FIELDS, NOISE_MODEL_FIELD]
        assert set(handle["catalog"]) == {*SPECTRA_FIELDS, NOISE_MODEL_FIELD}
        removed = {
            "fracAGN_5100_fit",
            "f_host_2500_psf",
            "f_bc_3000",
            "f_fe_uv_3000",
        }
        assert not (removed & set(handle["catalog"]))
        assert set(handle) == {"catalog"}
    frame = read_minimal_spectra_hdf5(spectra_output)
    assert frame["object_id"].tolist() == ["2", "1"]


def test_export_accepts_spectra_without_slopes(tmp_path):
    light_source, spectra_source = _write_sources(tmp_path)
    with h5py.File(spectra_source, "r+") as handle:
        del handle["catalog/pl_slope"]
        del handle["catalog/pl_slope_err"]
    spectra_output = tmp_path / "minimal_spectra.h5"
    assert export_minimal_hubble_inputs(
        light_source, spectra_source, tmp_path / "minimal_light.h5", spectra_output
    ) == 2
    frame = read_minimal_spectra_hdf5(spectra_output)
    assert "pl_slope" not in frame and "pl_slope_err" not in frame


def test_export_rejects_duplicate_source_ids(tmp_path):
    light_source, spectra_source = _write_sources(
        tmp_path, duplicate_light_ids=True
    )
    with pytest.raises(ValueError, match="duplicate object_id"):
        export_minimal_hubble_inputs(
            light_source,
            spectra_source,
            tmp_path / "minimal_light.h5",
            tmp_path / "minimal_spectra.h5",
        )


def test_minimal_spectra_reader_rejects_extra_field(tmp_path):
    light_source, spectra_source = _write_sources(tmp_path)
    light_output = tmp_path / "minimal_light.h5"
    spectra_output = tmp_path / "minimal_spectra.h5"
    export_minimal_hubble_inputs(
        light_source, spectra_source, light_output, spectra_output
    )
    with h5py.File(spectra_output, "a") as handle:
        handle["catalog"].create_dataset("fracAGN_5100_fit", data=[0.8, 0.9])
    with pytest.raises(ValueError, match="manifest mismatch"):
        read_minimal_spectra_hdf5(spectra_output)


def test_export_records_the_spectral_noise_model(tmp_path):
    light_source, spectra_source = _write_sources(tmp_path)
    fitted = tmp_path / "fitted.h5"
    export_minimal_hubble_inputs(light_source, spectra_source, tmp_path / "l1.h5", fitted)
    assert set(read_minimal_spectra_hdf5(fitted)[NOISE_MODEL_FIELD]) == {"fitted_jitter"}
    with h5py.File(spectra_source, "r+") as handle:
        del handle["catalog/frac_jitter"]
    fixed = tmp_path / "fixed.h5"
    export_minimal_hubble_inputs(light_source, spectra_source, tmp_path / "l2.h5", fixed)
    assert set(read_minimal_spectra_hdf5(fixed)[NOISE_MODEL_FIELD]) == {"fixed_floor"}


def test_previous_minimal_spectra_format_is_rejected(tmp_path):
    light_source, spectra_source = _write_sources(tmp_path)
    output = tmp_path / "minimal_spectra.h5"
    export_minimal_hubble_inputs(light_source, spectra_source, tmp_path / "l.h5", output)
    with h5py.File(output, "r+") as handle:
        handle.attrs[FORMAT_ATTR] = "qvc_hubble_paper_spectra_v1"
        del handle["catalog/" + NOISE_MODEL_FIELD]
    with pytest.raises(ValueError, match="Re-export"):
        read_minimal_spectra_hdf5(output)
