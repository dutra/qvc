"""Tests for the bright-subsample Hubble diagnostic."""
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qvc.hubble.hubble_bright_subsample import (
    BRIGHT_SUBSAMPLE_JSON_ATTR,
    BrightCutCompletenessModel,
    BrightSubsampleCut,
    apply_bright_subsample_cut,
    derive_bright_subsample_cut,
    summarize_bright_subsample_cut,
    wrap_completeness_params,
)
from qvc.hubble.hubble_completeness_refactored import (
    COMPLETENESS_MAG_COL,
    COMPLETENESS_MAG_ERR_COL,
)
from qvc.hubble.hubble_validation import analytic_completeness_params


def test_derive_threshold_from_sigmoid_completeness():
    model, mag_grid, z_grid, _, _ = analytic_completeness_params(23.0, 0.3)
    cut = derive_bright_subsample_cut(
        model, mag_grid, z_grid, completeness_min=0.9, margin=1.0
    )
    expected = 23.0 - 0.3 * np.log(9.0) - 1.0
    np.testing.assert_allclose(cut.threshold(np.array([0.5, 2.0, 3.5])), expected, atol=0.02)
    assert cut.run_tag() == "_brightsub-rel0p9-dm1"
    assert BrightSubsampleCut.from_json(cut.to_json()) == cut
    frame = cut.summary_frame()
    assert list(frame.columns) == [
        "z", "bright_cut_magnitude", "complete_to_magnitude", "peak_completeness",
    ]
    np.testing.assert_allclose(frame["peak_completeness"], 1.0, atol=1e-6)
    # Relative thresholds rescale with the map amplitude; absolute ones do not.
    scaled = derive_bright_subsample_cut(
        lambda m, z: 0.08 * model(m, z), mag_grid, z_grid, completeness_min=0.9, margin=1.0
    )
    np.testing.assert_allclose(scaled.threshold(1.0), expected, atol=0.02)
    absolute = derive_bright_subsample_cut(
        lambda m, z: 0.08 * model(m, z), mag_grid, z_grid,
        completeness_min=0.9, margin=1.0, relative=False,
    )
    assert absolute.run_tag() == "_brightsub-abs0p9-dm1"
    assert np.all(np.asarray(absolute.threshold_grid) == -99.0)
    # Redshifts with no complete magnitude get a very bright sentinel.
    empty = derive_bright_subsample_cut(
        lambda m, z: np.zeros(np.broadcast(m, z).shape), mag_grid, z_grid,
        completeness_min=0.9, margin=0.5,
    )
    assert np.all(np.asarray(empty.threshold_grid) == -99.0)
    with pytest.raises(ValueError, match="completeness_min"):
        BrightSubsampleCut(1.5, 1.0, (0.5,), (20.0,))


def test_wrapper_zeroes_faint_side_and_delegates_attributes():
    model, mag_grid, z_grid, dm, dz = analytic_completeness_params(23.0, 0.3)
    cut = derive_bright_subsample_cut(model, mag_grid, z_grid, completeness_min=0.9, margin=1.0)
    params = wrap_completeness_params((model, mag_grid, z_grid, dm, dz), cut)
    wrapped = params[0]
    assert isinstance(wrapped, BrightCutCompletenessModel)
    assert wrapped.magnitude_support == model.magnitude_support
    assert wrapped.magnitude_support_mode == "hard-cut"
    threshold = float(cut.threshold(1.0))
    magnitudes = np.array([threshold - 0.5, threshold - 1e-6, threshold + 1e-6, 22.9])
    values = wrapped(magnitudes, np.full(4, 1.0))
    np.testing.assert_array_equal(values[2:], 0.0)
    np.testing.assert_allclose(values[:2], model(magnitudes[:2]))
    wrapped._likelihood_pdet_cache = {"x": 1}
    assert not hasattr(model, "_likelihood_pdet_cache")
    restored = pickle.loads(pickle.dumps(wrapped))
    np.testing.assert_allclose(restored(magnitudes, np.full(4, 1.0)), values)
    assert "bright_subsample_cut" in wrapped.grid
    # Wrapping twice with the same cut is idempotent; a different cut is rejected.
    assert wrap_completeness_params(params, cut) is params
    other = derive_bright_subsample_cut(model, mag_grid, z_grid, completeness_min=0.9, margin=2.0)
    with pytest.raises(ValueError, match="different bright cut"):
        wrap_completeness_params(params, other)
    assert wrap_completeness_params(None, cut) is None
    assert wrap_completeness_params(params, None) is params


def test_apply_cut_filters_frame_and_summarizes():
    model, mag_grid, z_grid, _, _ = analytic_completeness_params(23.0, 0.3)
    cut = derive_bright_subsample_cut(model, mag_grid, z_grid, completeness_min=0.9, margin=1.0)
    threshold = float(cut.threshold(1.0))
    n = 40
    magnitudes = np.linspace(threshold - 2.0, threshold + 2.0, n)
    frame = pd.DataFrame(
        {
            "object_id": [f"o{i}" for i in range(n)],
            "z": np.linspace(0.5, 3.0, n),
            "m_2500_dereddened": magnitudes,
            "m_2500_dereddened_err": np.full(n, 0.02),
            "m_2500_attenuated_model": magnitudes + 0.3,
            "m_2500_attenuated_model_err": np.full(n, 0.03),
        }
    )
    frame.attrs["cut_tier"] = "2"
    kept, keep = apply_bright_subsample_cut(frame, cut, completeness_magnitude="dereddened")
    np.testing.assert_array_equal(keep, magnitudes <= cut.threshold(frame["z"].to_numpy()))
    assert len(kept) == int(keep.sum()) == 20
    assert kept.attrs["cut_tier"] == "2"
    assert BrightSubsampleCut.from_json(kept.attrs[BRIGHT_SUBSAMPLE_JSON_ATTR]) == cut
    assert COMPLETENESS_MAG_COL not in kept.columns  # the frame itself is untouched
    kept_att, keep_att = apply_bright_subsample_cut(frame, cut, completeness_magnitude="attenuated")
    assert keep_att.sum() < keep.sum()
    summary = summarize_bright_subsample_cut(frame, keep, cut, z_range=(0.44, 3.16), n_bins=4)
    assert summary["n_before"].sum() == n
    assert summary["n_kept"].sum() == 20
    np.testing.assert_allclose(summary["bright_cut_magnitude_at_center"], threshold)


def test_plot_bright_subsample_cut_writes_figure(tmp_path):
    from qvc.hubble.hubble_bright_subsample import plot_bright_subsample_cut

    model, mag_grid, z_grid, _, _ = analytic_completeness_params(23.0, 0.3)
    cut = derive_bright_subsample_cut(model, mag_grid, z_grid, completeness_min=0.75, margin=0.5)
    rng = np.random.default_rng(1)
    n = 300
    frame = pd.DataFrame(
        {
            "object_id": [f"o{i}" for i in range(n)],
            "z": rng.uniform(0.2, 3.5, n),
            "m_2500_dereddened": rng.uniform(19.0, 23.5, n),
            "m_2500_dereddened_err": np.full(n, 0.02),
        }
    )
    _, keep = apply_bright_subsample_cut(frame, cut, completeness_magnitude="dereddened")
    output = plot_bright_subsample_cut(
        model, mag_grid, z_grid, cut, frame, keep,
        z_range=(0.44, 3.16), completeness_magnitude="dereddened", plot_path=str(tmp_path),
    )
    assert output.endswith("bright_subsample_cut.pdf")
    assert (tmp_path / "bright_subsample_cut.pdf").stat().st_size > 5000


def test_likelihood_treats_bright_cut_as_hard_selection():
    """The Malmquist blob under the wrapped model matches a truncated-Gaussian expectation."""
    from astropy.cosmology import FlatLambdaCDM
    from qvc.hubble.hubble_likelihood import log_likelihood, sigma_lens_from_dc
    from qvc.hubble.hubble_model import build_agn_pivot_context, get_model_params

    model, mag_grid, z_grid, dm, dz = analytic_completeness_params(23.0, 0.3)
    cut = derive_bright_subsample_cut(model, mag_grid, z_grid, completeness_min=0.9, margin=1.0)
    params = wrap_completeness_params((model, mag_grid, z_grid, dm, dz), cut)
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    rng = np.random.default_rng(2)
    n = 8
    z = np.linspace(0.6, 2.4, n)
    x = rng.normal([-0.8, 2.6], [0.15, 0.3], size=(n, 2))
    threshold = cut.threshold(z)
    m = threshold - rng.uniform(0.05, 1.5, n)  # all brighter than the cut
    frame = pd.DataFrame(
        {
            "object_id": [f"o{i}" for i in range(n)],
            "z": z, "z_err": np.zeros(n),
            "apparent_mag_2500": m, "apparent_mag_2500_err": np.full(n, 0.03),
            "log_sigma_uv": x[:, 0], "log_tau_uv_rf": x[:, 1],
            "log_sigma_uv_std_psd": np.full(n, 0.05), "log_tau_uv_rf_std_psd": np.full(n, 0.08),
            "log_sigma_uv_log_tau_uv_rf_cov_psd": np.zeros(n),
            COMPLETENESS_MAG_COL: m, COMPLETENESS_MAG_ERR_COL: np.full(n, 0.03),
        }
    )
    data = {key: frame[key].to_numpy() for key in frame}
    context = build_agn_pivot_context(frame, (0.5, 2.5), round_pivots=False)
    pivots = context.as_dict()
    _, labels, _ = get_model_params("FlatLambdaCDM", only_agn=True)
    values = dict(M0_agn=-22.0, alpha_agn=5.0, beta_agn=-1.0, log_f=np.log(0.5), H0=70.0, Om0=0.3)
    theta = np.array([values[label] for label in labels])
    value, blob = log_likelihood(
        theta, agn_data=data, pantheon_data={}, _sna_L=None, _sna_Lower=True, _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM", completeness_params=params, z_pivot_agn=1.5,
        agn_pivot_context=context, only_agn=True, use_full_cov=False,
    )
    assert np.isfinite(value)
    center = (
        -22.0 + 5.0 * (x[:, 0] - pivots["log_sigma_uv"]) - 1.0 * (x[:, 1] - pivots["log_tau_uv_rf"])
        + cosmo.distmod(z).value
    )
    sigma = np.sqrt(
        0.03**2 + (5.0 * 0.05) ** 2 + (1.0 * 0.08) ** 2 + sigma_lens_from_dc(z, cosmo) ** 2 + 0.5**2
    )
    # The likelihood resolves the hard cut at the completeness-grid spacing
    # (0.011 mag here), so a fine-grid reference agrees to that level.
    grid = np.linspace(mag_grid[0], mag_grid[-1], 20001)
    for i in range(n):
        weight = params[0](grid, np.full(grid.size, z[i])) * np.exp(-0.5 * ((grid - center[i]) / sigma[i]) ** 2)
        expected = np.sum(weight * grid) / np.sum(weight) - center[i]
        assert abs(blob[1, i] - expected) < 1.5e-2
        assert np.all(params[0](grid[grid > threshold[i] + 0.02], np.full((grid > threshold[i] + 0.02).sum(), z[i])) == 0.0)
    # Without the cut the correction is smaller in magnitude (selection acts only near m50).
    _, blob_plain = log_likelihood(
        theta, agn_data=data, pantheon_data={}, _sna_L=None, _sna_Lower=True, _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM", completeness_params=(model, mag_grid, z_grid, dm, dz),
        z_pivot_agn=1.5, agn_pivot_context=context, only_agn=True, use_full_cov=False,
    )
    assert np.all(np.abs(blob[1]) >= np.abs(blob_plain[1]) - 1e-9)


def test_run_tag_checkpoint_validation_and_mode_table():
    from qvc.hubble import hubble_fit

    cut = BrightSubsampleCut(0.9, 1.0, (0.5, 1.0, 2.0), (21.0, 21.2, 21.5))
    base = hubble_fit.make_run_tag("Flatw0waCDM", False, "fastest", None, (0.44, 3.16))
    tagged = hubble_fit.make_run_tag(
        "Flatw0waCDM", False, "fastest", None, (0.44, 3.16), bright_subsample_cut=cut
    )
    assert tagged == base + "_brightsub-rel0p9-dm1"
    assert hubble_fit.make_run_tag(
        "Flatw0waCDM", True, "fastest", None, (0.44, 3.16), bright_subsample_cut=cut
    ) == hubble_fit.make_run_tag("Flatw0waCDM", True, "fastest", None, (0.44, 3.16))
    hubble_fit._validate_checkpoint_bright_subsample({"bright_subsample_json": cut.to_json()}, "ckpt", cut)
    hubble_fit._validate_checkpoint_bright_subsample({}, "ckpt", None)
    with pytest.raises(RuntimeError, match="bright-subsample"):
        hubble_fit._validate_checkpoint_bright_subsample({}, "ckpt", cut)
    with pytest.raises(RuntimeError, match="bright-subsample"):
        hubble_fit._validate_checkpoint_bright_subsample({"bright_subsample_json": cut.to_json()}, "ckpt", None)

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
        bright_subsample_completeness_min=0.9,
        bright_subsample_margin=1.0, bright_subsample_absolute=False,
    )
    table = hubble_fit.render_hubble_mode_table(args)
    assert "bright-subsample diagnostic" in table and "1 mag brighter" in table
    args.bright_subsample_completeness_min = None
    assert "off (full selected sample)" in hubble_fit.render_hubble_mode_table(args)


def test_run_single_wraps_completeness_and_records_cut(monkeypatch, tmp_path):
    from test_hubble_fit_smoke import (
        _make_fake_agn_sample,
        _make_fake_pantheon_sample,
        _patch_run_single_plot_stack,
        _write_fake_checkpoint,
    )
    from qvc.hubble import hubble_fit
    from qvc.hubble.hubble_model import get_model_params

    df_agn = _make_fake_agn_sample(n_agn=6)
    df_pantheon = _make_fake_pantheon_sample()
    model, mag_grid, z_grid, dm, dz = analytic_completeness_params(23.0, 0.3)
    cut = derive_bright_subsample_cut(model, mag_grid, z_grid, completeness_min=0.9, margin=1.0)
    priors, labels, _ = get_model_params("FlatLambdaCDM")
    theta = np.array([(priors[key][0] + priors[key][1]) / 2.0 for key in labels])
    flat_samples = np.tile(theta, (8, 1))
    seen = {}

    def fake_run_mcmc_pipeline(df_agn_arg, *args, **kwargs):
        seen["pipeline_cut"] = kwargs["bright_subsample_cut"]
        n = len(df_agn_arg)
        _write_fake_checkpoint(
            kwargs["checkpoint_file_override"], flat_samples, np.zeros(n), np.full(n, 0.05),
            agn_pivot_context=kwargs["agn_pivot_context"],
        )
        return (
            flat_samples, list(labels), lambda pts: np.zeros(len(np.atleast_2d(pts))), None,
            -21.0, 0.2, np.zeros(n), np.full(n, 0.05), None,
        )

    def fake_build(*args, **kwargs):
        return (model, mag_grid, z_grid, dm, dz)

    def fake_direct(samples_arg, *, completeness_params, df_agn_plot_sample, dmi_draw_indices=None, **kwargs):
        seen["direct_model"] = completeness_params[0]
        n = len(df_agn_plot_sample)
        base = (np.zeros(n), np.full(n, 0.02), np.full(n, 0.07))
        if dmi_draw_indices is None:
            return base
        from qvc.hubble.hubble_plotting import HubblePosteriorDrawSelection
        return base + (
            HubblePosteriorDrawSelection(
                values=np.zeros((len(dmi_draw_indices), n)), sample_indices=np.asarray(dmi_draw_indices),
                object_ids=tuple(df_agn_plot_sample["object_id"].astype(str)),
            ),
        )

    monkeypatch.chdir(tmp_path)
    _patch_run_single_plot_stack(monkeypatch)
    monkeypatch.setattr(hubble_fit, "plot_redshift_wiggle_diagnostics", lambda *a, **k: None)
    monkeypatch.setattr(hubble_fit, "plot_hubble", lambda *a, **k: (np.zeros(len(a[1])), np.ones(len(a[1])), np.full(len(a[1]), 44.0), np.full(len(a[1]), 0.1), np.full(len(a[1]), 0.2)))
    monkeypatch.setattr(hubble_fit, "run_mcmc_pipeline", fake_run_mcmc_pipeline)
    monkeypatch.setattr(hubble_fit, "_build_completeness_params", fake_build)
    monkeypatch.setattr(hubble_fit, "generate_fresh_completeness_sim_file", lambda *a, **k: str(tmp_path / "mock.h5"))
    monkeypatch.setattr(hubble_fit, "_compute_direct_full_sample_completeness_summaries", fake_direct)
    monkeypatch.setattr(hubble_fit, "_plot_completeness_cut_audit", lambda *a, **k: None)
    monkeypatch.setattr(hubble_fit, "plot_completeness_pre_post_cut_audit", lambda *a, **k: None)

    hubble_fit.run_single(
        df_agn=df_agn, df_agn_all=df_agn.copy(), df_pantheon=df_pantheon, _sna_L=None,
        _sna_Lower=True, _sna_LogdetCov=None, cosmo_model="FlatLambdaCDM", completeness=True,
        use_full_cov=False, only_sna=False, speed="fastest", z_range=(0.44, 3.16),
        disable_sigma_clip_pass=True, prefix="unit", skip_plots=True, bright_subsample_cut=cut,
    )
    assert seen["pipeline_cut"] == cut
    plot_dirs = list(tmp_path.rglob("bright_subsample_thresholds.csv"))
    assert len(plot_dirs) == 1 and plot_dirs[0].parent.name == "FlatLambdaCDM_joint"


def test_single_cli_forwards_population_before_bright_cut():
    """A bright-selected fit must not rebuild its map from the selected rows."""
    import ast
    from pathlib import Path
    from qvc.hubble import hubble_fit

    tree = ast.parse(Path(hubble_fit.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.If)
                and ast.unparse(n.test) == "__name__ == '__main__'"
                and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                        and child.func.id == 'run_single' for child in ast.walk(n)))
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'run_single']
    assert calls
    for call in calls:
        kwargs = {k.arg: k.value for k in call.keywords}
        assert ast.unparse(kwargs['df_agn_completeness_parent']) == 'df_agn_completeness_parent'
        assert ast.unparse(kwargs['bright_subsample_cut']) == 'bright_subsample_cut'
