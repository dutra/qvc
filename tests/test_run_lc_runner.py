from pathlib import Path


def test_run_lc_uses_aug24_v3_spectra_catalog():
    source = (Path(__file__).resolve().parents[1] / "run_lc.xsh").read_text()

    assert "aug24_0152pm_spectrafit_e5d2897_chisqgt20_N8000_nested_" in source
    assert "fhostpsf_resumed_m2500norm12_v3.h5" in source
    assert "aug23_0925am_spectrafit" not in source
    assert '"--spectra_fit_h5", str(spectra_fit_h5)' in source
    assert "if not spectra_fit_h5.is_file():" in source
    assert '"--outlier_half_window_days", "60"' in source
