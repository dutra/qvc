import numpy as np
import pandas as pd
import pytest

from qvc.spectra.catalog_hdf5 import write_spectra_catalog_hdf5
from qvc.spectra.merge_results import enrich_h5_catalog_rows, load_and_merge_h5


def _write_shard(path, object_ids, draw_values):
    frame = pd.DataFrame(
        {
            "object_id": object_ids,
            "run_label": ["fiducial"] * len(object_ids),
            "fit_ok": [True] * len(object_ids),
            "fit_backend": ["jaxsedfit_joint"] * len(object_ids),
            "z": np.arange(len(object_ids), dtype=float),
        }
    )
    draws = np.full((len(object_ids), 64, 5), np.nan, dtype=np.float32)
    for index, value in enumerate(draw_values):
        draws[index, 0] = value
    write_spectra_catalog_hdf5(path, frame, draws, np.ones(len(object_ids), dtype=int))


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
    np.testing.assert_array_equal(merged.valid_count, [1, 1, 1])


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
    draws = np.full((1, 64, 5), np.nan, dtype=np.float32)
    draws[0, 0] = 0.2
    write_spectra_catalog_hdf5(first, first_frame, draws, np.array([1]))
    write_spectra_catalog_hdf5(second, second_frame, draws, np.array([1]))

    merged = load_and_merge_h5([str(first), str(second)])

    assert merged.frame.columns.tolist() == [
        *first_frame.columns,
        "jqf_line_amp_CIV_std",
    ]
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
    draws = np.full((1, 64, 5), np.nan, dtype=np.float32)
    draws[0, 0] = 0.2
    write_spectra_catalog_hdf5(second, frame, draws, np.array([1]))

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
