from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "hpc_scripts" / "sfitspectra.xsh"


def test_sfitspectra_uses_csv_object_ids_without_h5_membership_filtering():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'chisq_csv = "results/data/variability_chi_sq_red_g_gt_20.csv"' in source
    assert "submit_object_ids = requested_object_ids" in source
    assert "--filter_object_id" in source
    for legacy_text in (
        "read_quasars_from_hdf5_flat",
        "h5_file",
        "load_h5_object_ids",
        "missing_from_h5",
        "H5_FILE",
        "USE_H5",
    ):
        assert legacy_text not in source
