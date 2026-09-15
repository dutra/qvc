"""Tests for the counts-comparison diagnostic and smoothing controls."""
import os

import h5py
import numpy as np
import pandas as pd
import pytest

from qvc.hubble import hubble_completeness_refactored as hcr
from qvc.hubble.cuts import (
    COMPLETENESS_MAP_MAG_EDGE_MAX,
    COMPLETENESS_MAP_MAG_EDGE_MIN,
    COMPLETENESS_MAP_Z_EDGE_MAX,
    COMPLETENESS_MAP_Z_EDGE_MIN,
)


def _write_mock(path, rng, n=200000):
    # Mock magnitudes rise toward the faint end, like a luminosity function.
    mags = COMPLETENESS_MAP_MAG_EDGE_MIN + (COMPLETENESS_MAP_MAG_EDGE_MAX - COMPLETENESS_MAP_MAG_EDGE_MIN) * rng.power(2.0, n)
    z = rng.uniform(COMPLETENESS_MAP_Z_EDGE_MIN, COMPLETENESS_MAP_Z_EDGE_MAX, n)
    mags[:2] = (COMPLETENESS_MAP_MAG_EDGE_MIN, COMPLETENESS_MAP_MAG_EDGE_MAX)
    z[:2] = (COMPLETENESS_MAP_Z_EDGE_MIN, COMPLETENESS_MAP_Z_EDGE_MAX)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("apparent_mag_2500", data=mags)
        handle.create_dataset("z", data=z)
        handle.attrs["mock_redshift_min"] = COMPLETENESS_MAP_Z_EDGE_MIN
        handle.attrs["mock_redshift_max"] = COMPLETENESS_MAP_Z_EDGE_MAX
        handle.attrs["mock_count_scale"] = 0.05
    return mags, z


def _observed_frame(rng, mags, z, *, bright_deficit=True):
    # Detect with a sigmoid at 22.5; optionally remove half of the bright objects
    # to imitate the bright-side rise seen in the real map.
    p = 1.0 / (1.0 + np.exp((mags - 22.5) / 0.3))
    if bright_deficit:
        p = np.where(mags < 20.0, 0.3 * p, p)
    keep = rng.random(mags.size) < 0.05 * p
    frame = pd.DataFrame(
        {
            "z": z[keep],
            "m_2500_dereddened": mags[keep],
            "m_2500_dereddened_err": np.full(int(keep.sum()), 0.02),
        }
    )
    return hcr.prepare_completeness_magnitude_columns(frame, "dereddened")


def test_counts_comparison(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    mock_path = tmp_path / "mock.h5"
    mags, z = _write_mock(mock_path, rng)
    frame = _observed_frame(rng, mags, z)

    plain, mag_centers, z_centers, *_ = hcr.get_completeness_function_2d(
        frame, sim_file=str(mock_path), plot=True, plot_path=str(tmp_path / "plain")
    )
    csv_path = tmp_path / "plain" / "completeness" / "completeness_counts_comparison.csv"
    assert csv_path.is_file()
    assert (tmp_path / "plain" / "completeness" / "completeness_counts_comparison.pdf").stat().st_size > 1000
    table = pd.read_csv(csv_path)
    assert {"z_lo", "z_hi", "mag", "n_obs", "n_mock_scaled", "ratio_obs_over_mock", "completeness_map_raw"} <= set(table.columns)
    assert table["n_obs"].sum() == pytest.approx(len(frame))
    # The injected bright deficit shows up as a rising ratio on the bright side.
    middle = table[(table["z_lo"] == 1.5)]
    bright = middle[(middle["mag"] > 18.6) & (middle["mag"] < 19.7)]["ratio_obs_over_mock"].mean()
    plateau = middle[(middle["mag"] > 20.4) & (middle["mag"] < 21.8)]["ratio_obs_over_mock"].mean()
    assert bright < 0.7 * plateau
    grid = plain(mag_centers[None, :], z_centers[:, None])
    assert np.any(np.diff(grid, axis=1) > 1e-6)  # plain map rises with magnitude somewhere



def test_run_tag_and_mode_table_reflect_map_variants(monkeypatch):
    from types import SimpleNamespace
    from qvc.hubble import hubble_fit

    monkeypatch.delenv(hcr.COMPLETENESS_SMOOTH_SIGMA_MAG_ENV, raising=False)
    monkeypatch.delenv(hcr.COMPLETENESS_SMOOTH_SIGMA_Z_ENV, raising=False)
    base = hubble_fit.make_run_tag("Flatw0waCDM", False, "fastest", None, (0.44, 3.16))
    assert hubble_fit.completeness_map_variant_tag() == "_compgrid80x45"
    monkeypatch.setenv(hcr.COMPLETENESS_SMOOTH_SIGMA_MAG_ENV, "0.25")
    assert hubble_fit.completeness_map_variant_tag() == "_compsm0p25x0p3_compgrid80x45"
    assert hubble_fit.make_run_tag("Flatw0waCDM", False, "fastest", None, (0.44, 3.16)) == base.replace("_compgrid80x45", "_compsm0p25x0p3_compgrid80x45")
    assert hubble_fit.make_run_tag("Flatw0waCDM", False, "fastest", None, (0.44, 3.16), completeness=False, prior_profile="default").endswith("_disable_completeness")
    args = SimpleNamespace(
        only_sna=False, only_agn=True, light_curve_uncertainty_mode="covariance",
        correct_sigma_uv_host=False,
        fit_alpha_lambda_term=False, fit_eta_sigma_term=False,
        fit_f_agn_psf_2500_sigmoid_term=False, fit_f_agn_psf_2500_flux_fraction_term=False,
        fit_redshift_log_f_term=False, disable_pivot_rounding=True, disable_completeness=False,
        completeness_mode="2d", completeness_magnitude="attenuated", completeness_lf_model="shen",
        completeness_magnitude_support_mode="hard-cut", selection_attenuation_mode="fixed-offset",
        disable_sigma_clip_pass=True, disable_full_covariance=False, use_jax=False,
        cosmo_models=["Flatw0waCDM"], prior_profile="default", cut_tier="2",
        magnitude_convention="dereddened",
        bright_subsample_completeness_min=None,
    )
    table = hubble_fit.render_hubble_mode_table(args)
    assert "smooth=(0.25 mag" in table
