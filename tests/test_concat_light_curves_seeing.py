import pandas as pd
import pytest

from qvc.light_curve import multiband_generate_lc


@pytest.mark.parametrize("half_window_days", [0.0, -1.0, float("nan")])
def test_concat_light_curves_rejects_invalid_outlier_half_window(half_window_days):
    with pytest.raises(ValueError, match="outlier_half_window_days must be positive"):
        multiband_generate_lc.concat_light_curves(
            outlier_half_window_days=half_window_days
        )


def test_concat_light_curves_tolerates_missing_seeing_sidecars(monkeypatch):
    frames = {
        "Catalog.parquet": pd.DataFrame(columns=["idx", "objectId"]),
        "dr16s82_sdssLCRaw.parquet": pd.DataFrame(columns=["objectId", "mjd"]),
        "dr16s82_ps1LCRaw.parquet": pd.DataFrame(),
        "dr16s82_ZuberLCRaw.parquet": pd.DataFrame(),
    }

    def fake_resolve(path):
        if path in multiband_generate_lc.SEEING_SIDECARS.values():
            raise FileNotFoundError(path)
        return path

    def fake_read_parquet(path):
        name = str(path).rsplit("/", 1)[-1]
        return frames[name].copy()

    monkeypatch.setattr(multiband_generate_lc, "resolve_qvc_data_path", fake_resolve)
    monkeypatch.setattr(multiband_generate_lc.pd, "read_parquet", fake_read_parquet)

    assert multiband_generate_lc.concat_light_curves(filter_object_ids=[]) == []
