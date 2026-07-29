import numpy as np

from qvc.hubble import hubble_fit


def test_fresh_completeness_mock_uses_full_area_without_thinning(
    monkeypatch,
    tmp_path,
):
    generator_calls = []
    save_calls = []
    mock_values = np.array([1.0, 2.0])

    monkeypatch.setattr(
        hubble_fit,
        "build_shen_lf",
        lambda _path: (mock_values, mock_values, mock_values),
    )

    def fake_mock_m_per_zbin(*args, **kwargs):
        generator_calls.append((args, kwargs))
        return (mock_values,) * 9

    def fake_save_mock_catalog(*args, **kwargs):
        save_calls.append((args, kwargs))

    monkeypatch.setattr(hubble_fit, "mock_m_per_zbin", fake_mock_m_per_zbin)
    monkeypatch.setattr(hubble_fit, "save_mock_catalog", fake_save_mock_catalog)

    hubble_fit.generate_fresh_completeness_sim_file(
        tmp_path,
        area_deg2=hubble_fit.DEFAULT_COMPLETENESS_MOCK_AREA_DEG2,
    )

    assert hubble_fit.DEFAULT_COMPLETENESS_MOCK_AREA_DEG2 == 274.085
    assert generator_calls[0][1]["thinning_probability"] == 1.0
    assert save_calls[0][1]["thinning_probability"] == 1.0
    assert save_calls[0][1]["area_deg2"] == 274.085
