import pytest

from qvc.light_curve import merge_results


def _identity_row(**overrides):
    row = {
        "object_id": "123",
        "ra": 10.0,
        "dec": -2.0,
        "large_unrelated_posterior_field": object(),
    }
    for band in "ugri":
        row[f"log_sigma_band_{band}"] = -0.5
        row[f"log_sigma_band_{band}_err"] = 0.1
        row[f"log_tau_band_{band}_RF"] = 2.5
        row[f"log_tau_band_{band}_RF_err"] = 0.2
    row.update(overrides)
    return row


def test_identity_fit_frame_projects_only_required_columns():
    frame = merge_results._build_identity_fit_frame(
        [_identity_row()],
        merge_results.MACLEOD_IDENTITY_BANDS,
        include_coordinates=True,
    )

    assert "large_unrelated_posterior_field" not in frame.columns
    assert list(frame.columns) == merge_results._identity_fit_fields(
        merge_results.MACLEOD_IDENTITY_BANDS,
        include_coordinates=True,
    )


def test_identity_fit_frame_reports_missing_required_columns():
    row = _identity_row()
    del row["log_tau_band_r_RF"]

    with pytest.raises(KeyError, match="log_tau_band_r_RF"):
        merge_results._build_identity_fit_frame(
            [row],
            merge_results.STONE_IDENTITY_BANDS,
        )
