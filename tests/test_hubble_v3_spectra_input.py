from types import SimpleNamespace

import numpy as np
import pandas as pd

from qvc.hubble import hubble_utils


def test_populate_spectra_fit_reads_v3_joint_draws(monkeypatch, tmp_path):
    spectra_path = tmp_path / "spectra_v3.h5"
    spectra_path.touch()
    frame = pd.DataFrame(
        {
            "object_id": ["101"],
            "fit_ok": [True],
            "fit_backend": ["jaxsedfit_joint"],
            "fracAGN_5100_fit": [0.8],
            "fracAGN_5100_fit_err": [0.05],
            "m_2500_dereddened": [20.0],
            "m_2500_dereddened_err": [0.1],
            "m_2500_attenuated_model": [20.4],
            "m_2500_attenuated_model_err": [0.12],
            "pl_slope": [-1.5],
            "pl_slope_err": [0.1],
        }
    )
    dereddened = np.full((1, 64), np.nan, dtype=np.float32)
    attenuated = np.full((1, 64), np.nan, dtype=np.float32)
    dereddened[0, :2] = [19.9, 20.1]
    attenuated[0, :2] = [20.2, 20.6]
    catalog = SimpleNamespace(
        frame=frame,
        catalog_format="qvc_spectra_catalog_v3",
        joint_posterior_draws={
            "m_2500_dereddened": dereddened,
            "m_2500_attenuated_model": attenuated,
        },
        joint_posterior_valid_count=np.array([2], dtype=np.int16),
    )
    reader_calls = []

    def fake_read_spectra_catalog_hdf5(
        path,
        *,
        include_fraction_draws,
        allow_legacy_v3_host_capture_metadata=False,
    ):
        reader_calls.append(
            {
                "path": path,
                "include_fraction_draws": include_fraction_draws,
                "allow_legacy_v3_host_capture_metadata": (
                    allow_legacy_v3_host_capture_metadata
                ),
            }
        )
        return catalog

    monkeypatch.setattr(
        hubble_utils,
        "read_spectra_catalog_hdf5",
        fake_read_spectra_catalog_hdf5,
    )

    result = hubble_utils.populate_spectra_fit(
        pd.DataFrame({"object_id": ["101"], "pl_slope": [-9.0]}),
        [spectra_path],
        allow_legacy_v3_host_capture_metadata=True,
    )

    assert result.loc[0, "pl_slope"] == -1.5
    assert result.loc[0, "joint_posterior_valid_count"] == 2
    np.testing.assert_allclose(
        result.loc[0, "m_2500_dereddened_draws"][:2],
        [19.9, 20.1],
    )
    np.testing.assert_allclose(
        result.loc[0, "m_2500_attenuated_model_draws"][:2],
        [20.2, 20.6],
    )
    assert reader_calls == [
        {
            "path": str(spectra_path),
            "include_fraction_draws": True,
            "allow_legacy_v3_host_capture_metadata": True,
        }
    ]
