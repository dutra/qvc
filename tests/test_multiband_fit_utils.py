import numpy as np

from qvc.light_curve.multiband_fit_utils import diagnostics_for_per_chain_samples


def test_sampler_diagnostics_ignore_deterministic_agn_fractions():
    draws = np.linspace(-1.0, 1.0, 16).reshape(2, 8)
    diagnostics = diagnostics_for_per_chain_samples(
        {
            "eta_tau": draws,
            "agn_fraction_by_band_g": np.full((2, 8), 0.8),
        },
        max_lag=4,
    )

    assert "eta_tau_ess" in diagnostics
    assert not any(key.startswith("agn_fraction_by_band_g_") for key in diagnostics)
