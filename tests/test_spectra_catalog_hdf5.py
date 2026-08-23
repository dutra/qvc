import h5py
import numpy as np
import pandas as pd

from qvc.spectra.catalog_hdf5 import (
    SPECTRA_CATALOG_FORMAT,
    read_spectra_catalog_hdf5,
    write_spectra_catalog_hdf5,
)


def test_spectra_catalog_hdf5_round_trip_preserves_catalog_and_draws(tmp_path):
    path = tmp_path / "spectra.h5"
    frame = pd.DataFrame(
        {
            "object_id": ["101", "102"],
            "fit_ok": [True, False],
            "fit_backend": ["jaxsedfit_joint", "jaxsedfit_joint"],
            "z": [1.2, np.nan],
            "error_message": ["", "failed"],
        }
    )
    draws = np.full((2, 64, 5), np.nan, dtype=np.float32)
    draws[0, :2] = np.array([[0.7] * 5, [0.8] * 5], dtype=np.float32)

    write_spectra_catalog_hdf5(path, frame, draws, np.array([2, 0]))
    result = read_spectra_catalog_hdf5(path)

    assert result.frame["object_id"].tolist() == ["101", "102"]
    assert result.frame["fit_ok"].tolist() == [True, False]
    assert result.frame["error_message"].tolist() == ["", "failed"]
    assert np.isnan(result.frame.loc[1, "z"])
    np.testing.assert_array_equal(result.valid_count, np.array([2, 0]))
    np.testing.assert_allclose(result.fraction_draws[0, :2], draws[0, :2])
    assert result.bands == ("u", "g", "r", "i", "z")
    with h5py.File(path, "r") as handle:
        assert handle.attrs["qvc_spectra_catalog_format"] == SPECTRA_CATALOG_FORMAT
        assert set(handle) == {"catalog", "psf_agn_fraction_draws"}
