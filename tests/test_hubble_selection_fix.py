"""Selection integral across the hard bright cut, and per-target-selection maps."""
import argparse

import h5py
import numpy as np
import pandas as pd
import pytest

from qvc.hubble import hubble_completeness_refactored as hcr
from qvc.hubble import hubble_likelihood
from qvc.hubble.hubble_bright_subsample import (
    DEFAULT_COMPLETENESS_MAP_GROUPS,
    BrightSubsampleCut,
    GroupedBrightSubsampleCut,
    GroupedCompletenessModel,
    BrightCutCompletenessModel,
    apply_bright_subsample_cut,
    assign_completeness_groups,
    bright_subsample_cut_from_json,
    build_grouped_completeness_params,
    wrap_completeness_params,
)
from qvc.hubble.hubble_completeness_refactored import COMPLETENESS_GROUP_COL, Completeness2D
from qvc.hubble.cuts import (
    COMPLETENESS_MAP_MAG_EDGE_MAX,
    COMPLETENESS_MAP_MAG_EDGE_MIN,
    COMPLETENESS_MAP_Z_EDGE_MAX,
    COMPLETENESS_MAP_Z_EDGE_MIN,
)


def _linear_map(bin_width, *, slope=-0.1, level=0.5):
    """Map that is exactly linear in magnitude, so linear interpolation is exact."""
    edges = np.arange(16.0, 24.0 + 1e-9, bin_width)
    mag = 0.5 * (edges[:-1] + edges[1:])
    z = np.linspace(0.1, 3.9, 12)
    values = np.clip(level + slope * (mag[:, None] - 20.0) + 0.0 * z[None, :], 0.0, 1.0)
    model = Completeness2D(mag, z, values, magnitude_support=(edges[0], edges[-1]))
    return model, mag, values


def _cut(threshold_low, threshold_high):
    return BrightSubsampleCut(
        completeness_min=0.1, margin=0.2, z_grid=(0.0, 4.0),
        threshold_grid=(threshold_low, threshold_high),
    )


def _reference_moments(m_model, sigma, z, cut, slope=-0.1, level=0.5):
    """Dense quadrature of N(m; m_model, sigma) p(m) 1[m <= cut(z)]."""
    out = []
    for mean, sig, redshift in zip(m_model, sigma, z):
        grid = np.linspace(16.1, cut.threshold(redshift), 200001)
        p = np.clip(level + slope * (grid - 20.0), 0.0, 1.0)
        w = np.exp(-0.5 * ((grid - mean) / sig) ** 2) / (np.sqrt(2 * np.pi) * sig) * p
        Z = np.trapezoid(w, grid)
        out.append((Z, np.trapezoid(w * grid, grid) / Z - mean))
    return np.asarray(out)


def _loglike_blob(model, mag, m_model, sigma, z, group_index=None):
    return hubble_likelihood.completeness_loglike(
        m_obs=np.full(m_model.shape, 19.0), m_obs_err=np.full(m_model.shape, 0.02),
        m_model=m_model, mu_err=sigma, z=z, completeness_model=model, m_grid=mag,
        magnitude_support=model.magnitude_support, group_index=group_index,
    )


def test_selection_integral_is_exact_across_hard_cut_for_any_bin_size():
    rng = np.random.default_rng(1)
    n = 60
    z = rng.uniform(0.3, 3.5, n)
    # Keep the bright map edge (first centre, which moves with the bin size)
    # far in the Gaussian tail.
    m_model = rng.uniform(20.5, 23.0, n)
    sigma = rng.uniform(0.7, 1.0, n)
    # The cut moves continuously with z, so it falls at every phase between nodes.
    cut = _cut(21.43, 22.37)
    reference = _reference_moments(m_model, sigma, z, cut)
    dmi = {}
    for width in (0.4, 0.2, 0.1, 0.05):
        model, mag, _ = _linear_map(width)
        _, blob = _loglike_blob(BrightCutCompletenessModel(model, cut), mag, m_model, sigma, z)
        # Each interval is integrated in closed form: no node-spacing error.
        assert np.allclose(blob[0], reference[:, 0], rtol=1e-5)
        assert np.max(np.abs(blob[1] - reference[:, 1])) < 5e-5
        dmi[width] = blob[1]
    # Independent of the map bin size across the cut.
    assert np.max(np.abs(dmi[0.4] - dmi[0.05])) < 5e-5


def test_old_node_ramp_would_bias_dmi_but_cut_nodes_do_not():
    """Regression: a cut between nodes must not be ramped over a bin."""
    model, mag, _ = _linear_map(0.2)
    z = np.array([1.0])
    m_model, sigma = np.array([22.0]), np.array([1.0])
    cut = _cut(21.49, 21.49)  # just below a node at 21.5
    reference = _reference_moments(m_model, sigma, z, cut)
    _, blob = _loglike_blob(BrightCutCompletenessModel(model, cut), mag, m_model, sigma, z)
    assert abs(blob[1, 0] - reference[0, 1]) < 1e-5
    # Shared-grid trapezoid of the masked integrand (the previous behaviour).
    wrapped = BrightCutCompletenessModel(model, cut)
    p = wrapped(mag[None, :], z[:, None])[0]
    w = np.exp(-0.5 * ((mag - m_model[0]) / sigma[0]) ** 2) * p
    ramp_dmi = np.trapezoid(w * mag, mag) / np.trapezoid(w, mag) - m_model[0]
    assert abs(ramp_dmi - reference[0, 1]) > 0.02


def test_posterior_draw_and_joint_variants_use_the_cut_nodes():
    model, mag, _ = _linear_map(0.2)
    wrapped = BrightCutCompletenessModel(model, _cut(21.43, 22.37))
    z = np.array([0.6, 1.7, 2.9])
    m_model = np.array([21.0, 22.0, 22.5])
    sigma = np.array([0.9, 1.0, 1.1])
    _, scalar = _loglike_blob(wrapped, mag, m_model, sigma, z)
    _, draws = hubble_likelihood.completeness_loglike_posterior_draws(
        m_obs=np.full(3, 19.0), m_obs_err=np.full(3, 0.02),
        m_model_draws=np.repeat(m_model[:, None], 4, axis=1), reference_m_model=m_model,
        mu_err=sigma, draw_counts=np.full(3, 4), z=z, completeness_model=wrapped, m_grid=mag,
        magnitude_support=model.magnitude_support,
    )
    assert np.allclose(scalar, draws, rtol=1e-10, atol=1e-12)


def _two_group_model():
    model, mag, _ = _linear_map(0.2)
    flat = Completeness2D(
        mag, model.z_centers, np.full((mag.size, model.z_centers.size), 0.3),
        magnitude_support=model.map_magnitude_support,
    )
    cut = GroupedBrightSubsampleCut(
        "spectrograph", ("sdss", "boss"), (_cut(21.1, 21.3), _cut(22.2, 22.4))
    )
    grouped = GroupedCompletenessModel(
        [BrightCutCompletenessModel(flat, cut.cuts[0]), BrightCutCompletenessModel(model, cut.cuts[1])], cut
    )
    return grouped, flat, model, mag, cut


def test_each_object_uses_its_own_group_map_and_cut():
    grouped, flat, linear, mag, cut = _two_group_model()
    z = np.array([0.5, 1.5, 2.5, 3.0, 1.0])
    m_model = np.array([21.5, 22.0, 22.5, 21.0, 22.3])
    sigma = np.full(5, 1.0)
    group = np.array([0, 1, 0, 1, 1])
    _, blob = _loglike_blob(grouped, mag, m_model, sigma, z, group_index=group)
    for g, inner in ((0, flat), (1, linear)):
        rows = group == g
        _, single = _loglike_blob(
            BrightCutCompletenessModel(inner, cut.cuts[g]), mag, m_model[rows], sigma[rows], z[rows]
        )
        assert np.allclose(blob[:, rows], single, rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError):
        grouped.selection_cut_magnitude(z)
    # The full log-likelihood refuses data without the group column.
    with pytest.raises(KeyError):
        hubble_likelihood.completeness_group_index_for_data({"z": z}, grouped)


def test_grouped_cut_mask_json_and_frame_filter():
    *_, cut = _two_group_model()
    assert bright_subsample_cut_from_json(cut.to_json()) == cut
    assert bright_subsample_cut_from_json(cut.cuts[0].to_json()) == cut.cuts[0]
    assert "compgroups-spectrograph" in cut.run_tag()
    frame = pd.DataFrame({
        "z": [1.0, 1.0, 1.0, 1.0], "mjd": [52000, 52000, 56000, 56000],
        "m_2500_attenuated_model": [21.0, 21.8, 21.8, 22.6],
        "m_2500_attenuated_model_err": [0.02] * 4,
    })
    grouped = assign_completeness_groups(frame, "spectrograph")
    assert grouped[COMPLETENESS_GROUP_COL].tolist() == [0, 0, 1, 1]
    kept, keep = apply_bright_subsample_cut(grouped, cut, completeness_magnitude="attenuated")
    assert keep.tolist() == [True, False, True, False]
    assert set(cut.summary_frame()["completeness_group"]) == {"sdss", "boss"}


def _write_mock(path, rng, n=200000):
    mags = COMPLETENESS_MAP_MAG_EDGE_MIN + (COMPLETENESS_MAP_MAG_EDGE_MAX - COMPLETENESS_MAP_MAG_EDGE_MIN) * rng.power(2.0, n)
    z = rng.uniform(COMPLETENESS_MAP_Z_EDGE_MIN, COMPLETENESS_MAP_Z_EDGE_MAX, n)
    mags[:2] = (COMPLETENESS_MAP_MAG_EDGE_MIN, COMPLETENESS_MAP_MAG_EDGE_MAX)
    z[:2] = (COMPLETENESS_MAP_Z_EDGE_MIN, COMPLETENESS_MAP_Z_EDGE_MAX)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("apparent_mag_2500", data=mags)
        handle.create_dataset("z", data=z)
        handle.attrs["mock_redshift_min"] = COMPLETENESS_MAP_Z_EDGE_MIN
        handle.attrs["mock_redshift_max"] = COMPLETENESS_MAP_Z_EDGE_MAX
    return mags, z


def test_group_maps_share_the_mock_normalization_and_sum_to_the_single_map(tmp_path):
    rng = np.random.default_rng(7)
    mock = tmp_path / "mock.h5"
    mags, z = _write_mock(mock, rng)
    # Two selections: a shallow one (sdss) and a deep one that avoids bright objects (boss).
    p_sdss = 0.04 / (1.0 + np.exp((mags - 20.5) / 0.25))
    p_boss = 0.04 / (1.0 + np.exp((mags - 22.2) / 0.25)) * (mags > 19.5)
    u = rng.random(mags.size)
    group = np.where(u < p_sdss, 0, np.where(u < p_sdss + p_boss, 1, -1))
    sel = group >= 0
    frame = pd.DataFrame({
        "z": z[sel], "m_2500_dereddened": mags[sel], "m_2500_dereddened_err": 0.02,
        "mjd": np.where(group[sel] == 0, 53000, 56000),
    })
    frame = assign_completeness_groups(hcr.prepare_completeness_magnitude_columns(frame, "dereddened"), "spectrograph")

    def build_map(subset, *, count_scale, plot, plot_path):
        return hcr.get_completeness_function_2d(subset, sim_file=str(mock), plot=False, count_scale=count_scale)

    grouped, cut, union, groups = build_grouped_completeness_params(
        build_map, frame, completeness_min=0.1, margin=0.2
    )
    mag_c, z_c = union[1], union[2]
    # The mock has no count-scale metadata; the bright-end estimate of the union is reused.
    assert groups["sdss"][0].count_scale == groups["boss"][0].count_scale == union[0].count_scale
    total = groups["sdss"][0](mag_c[:, None], z_c[None, :]) + groups["boss"][0](mag_c[:, None], z_c[None, :])
    single = union[0](mag_c[:, None], z_c[None, :])
    unclipped = single < 0.999  # every map is clipped to [0, 1] separately
    assert unclipped.mean() > 0.5
    assert np.max(np.abs(total - single)[unclipped]) < 1e-12
    # The shallow group's cut is brighter than the deep group's at every redshift.
    assert np.all(np.asarray(cut.cut_for("sdss").threshold_grid) < np.asarray(cut.cut_for("boss").threshold_grid))
    # Without a group index the grouped model returns the union selection probability.
    union_selection = grouped[0](mag_c[:, None], z_c[None, :])
    expected = sum(
        np.where(cut.cuts[i].mask(mag_c[:, None], z_c[None, :]), groups[name][0](mag_c[:, None], z_c[None, :]), 0.0)
        for i, name in enumerate(cut.groups)
    )
    assert np.allclose(union_selection, expected)
    assert wrap_completeness_params(grouped, cut) is grouped
    with pytest.raises(ValueError):
        wrap_completeness_params(union, cut)


def _configure(**overrides):
    from qvc.hubble import hubble_fit

    values = dict(
        only_sna=False, bright_subsample_completeness_min=0.1, bright_subsample_margin=0.2,
        disable_completeness=False, completeness_mode="2d", run="single", use_jax=False,
        selection_attenuation_mode="fixed-offset", completeness_lf_model="shen",
        completeness_magnitude_support_mode="hard-cut",
    )
    values.update(overrides)
    args = argparse.Namespace(**values)
    hubble_fit.configure_completeness_run_args(args)
    return args


def test_default_is_spectrograph_maps_when_prerequisites_hold(capsys):
    assert DEFAULT_COMPLETENESS_MAP_GROUPS == "spectrograph"
    args = _configure()
    assert args.completeness_map_groups == "spectrograph"
    assert args.completeness_map_groups_explicit is False
    assert "NOTICE" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "overrides, reason",
    [
        (dict(bright_subsample_completeness_min=None), "no --bright-subsample-completeness-min"),
        (dict(bright_subsample_completeness_min=None, completeness_magnitude_support_mode="tails"), "tails"),
        (dict(bright_subsample_completeness_min=None, use_jax=True), "NumPy/Dynesty"),
        (dict(bright_subsample_completeness_min=None, disable_completeness=True), "completeness is disabled"),
    ],
)
def test_default_falls_back_to_single_with_notice(capsys, overrides, reason):
    args = _configure(**overrides)
    assert args.completeness_map_groups == "single"
    out = capsys.readouterr().out
    assert "[NOTICE] Completeness map groups: falling back" in out and reason in out


def test_explicit_spectrograph_without_prerequisites_errors():
    with pytest.raises(NotImplementedError):
        _configure(completeness_map_groups="spectrograph", bright_subsample_completeness_min=None)
    assert _configure(completeness_map_groups="spectrograph").completeness_map_groups_explicit


def test_explicit_single_is_the_previous_behaviour(capsys):
    args = _configure(completeness_map_groups="single")
    assert args.completeness_map_groups == "single" and args.completeness_map_groups_explicit
    assert "NOTICE" not in capsys.readouterr().out
    frame = pd.DataFrame({"z": [1.0], "mjd": [52000]})
    assert assign_completeness_groups(frame, "single") is frame
    # SN-only runs always use 'single'.
    assert _configure(only_sna=True, completeness_map_groups="spectrograph").completeness_map_groups == "single"
    # A plain map without a hard cut keeps the shared grid (no per-object nodes).
    model, mag, _ = _linear_map(0.2)
    grid = hubble_likelihood._cached_magnitude_integration_grid(model, mag, model.magnitude_support)
    nodes = hubble_likelihood._selection_integration_nodes(model, grid, np.array([1.0]))
    assert nodes.shape == (1, grid.size) and np.array_equal(nodes[0], grid)


def test_missing_mjd_is_detected_for_the_default_fallback():
    from qvc.hubble.hubble_bright_subsample import completeness_group_prerequisite_problem

    assert completeness_group_prerequisite_problem(pd.DataFrame({"mjd": [52000, 56000]})) is None
    assert "no spectrum 'mjd'" in completeness_group_prerequisite_problem(pd.DataFrame({"z": [1.0]}))
    assert "1 objects lack" in completeness_group_prerequisite_problem(pd.DataFrame({"mjd": [52000, np.nan]}))


def test_far_upper_tail_keeps_precision():
    nodes = np.linspace(20.0, 22.0, 11)[None, :]
    p = np.linspace(0.6, 0.2, 11)[None, :]
    for sigma in (0.13, 0.12, 0.1):
        Z, m1, _ = hubble_likelihood._piecewise_linear_gaussian_moments(
            nodes, p, np.array([[19.0]]), np.array([[sigma]])
        )
        grid = np.linspace(20.0, 22.0, 400001)
        w = np.exp(-0.5 * ((grid - 19.0) / sigma) ** 2) * np.interp(grid, nodes[0], p[0])
        w /= np.sqrt(2 * np.pi) * sigma
        assert Z[0] > 0 and Z[0] == pytest.approx(np.trapezoid(w, grid), rel=1e-3)
        assert m1[0] / Z[0] == pytest.approx(np.trapezoid(w * grid, grid) / np.trapezoid(w, grid), abs=1e-3)
