import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble import hubble_fit
from qvc.hubble.hubble_utils import (
    get_qvc_result_dir,
    populate_xray,
    resolve_qvc_result_path,
)


def test_speed_names_are_ordered_and_do_not_accept_legacy_aliases():
    assert hubble_fit.SPEED_CHOICES == ("fastest", "quick", "standard", "production")
    assert hubble_fit.normalize_speed("fastest") == "fastest"
    assert hubble_fit.normalize_speed("quick") == "quick"
    assert hubble_fit.normalize_speed("standard") == "standard"
    assert hubble_fit.normalize_speed("production") == "production"

    with pytest.raises(ValueError, match="Invalid speed"):
        hubble_fit.normalize_speed("fast")


def test_resolve_qvc_result_path_absolute_existing(tmp_path):
    target = tmp_path / "artifact.h5"
    target.write_text("ok")
    assert resolve_qvc_result_path(target) == str(target)


def test_resolve_qvc_result_path_repo_relative_results_prefix(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    file_path = repo_root / "results" / "foo" / "bar.h5"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("ok")

    monkeypatch.setattr("qvc.hubble.hubble_utils.QVC_ROOT", repo_root)
    monkeypatch.delenv("QVC_RESULT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    assert resolve_qvc_result_path("results/foo/bar.h5") == str(file_path.resolve())


def test_resolve_qvc_result_path_repo_relative_without_results_prefix(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    file_path = repo_root / "results" / "foo" / "bar.h5"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("ok")

    monkeypatch.setattr("qvc.hubble.hubble_utils.QVC_ROOT", repo_root)
    monkeypatch.delenv("QVC_RESULT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    assert resolve_qvc_result_path("foo/bar.h5") == str(file_path.resolve())


def test_resolve_qvc_result_path_env_override(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    result_root = tmp_path / "custom_results"
    file_path = result_root / "foo" / "bar.h5"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("ok")

    monkeypatch.setattr("qvc.hubble.hubble_utils.QVC_ROOT", repo_root)
    monkeypatch.setenv("QVC_RESULT_DIR", str(result_root))
    monkeypatch.chdir(tmp_path)

    assert resolve_qvc_result_path("results/foo/bar.h5") == str(file_path.resolve())
    assert resolve_qvc_result_path("foo/bar.h5") == str(file_path.resolve())
    assert get_qvc_result_dir() == result_root


def test_resolve_qvc_result_path_missing_mentions_env_var(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    monkeypatch.setattr("qvc.hubble.hubble_utils.QVC_ROOT", repo_root)
    monkeypatch.delenv("QVC_RESULT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="QVC_RESULT_DIR"):
        resolve_qvc_result_path("results/missing/file.h5")


def test_populate_xray_missing_catalog_returns_nan_columns(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    (repo_root / "data").mkdir(parents=True)
    monkeypatch.setattr("qvc.hubble.hubble_utils.QVC_ROOT", repo_root)
    monkeypatch.delenv("QVC_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    df = pd.DataFrame({"object_id": ["agn_1"], "ra": [150.0], "dec": [2.0], "z": [1.0]})

    with pytest.warns(RuntimeWarning, match="skipping X-ray enrichment"):
        out = populate_xray(df)

    assert len(out) == len(df)
    np.testing.assert_array_equal(out["object_id"].to_numpy(), df["object_id"].to_numpy())
    for col in (
        "flux_aper_b",
        "flux_aper_hilim_b",
        "flux_aper_lolim_b",
        "flux_aper_err_b",
    ):
        assert col in out.columns
        assert out[col].isna().all()
    for col in (
        "log_L2500_nu",
        "log_L2keV_nu",
        "log_Lxray",
        "alphaOX",
        "delta_alphaOX",
    ):
        assert col not in out.columns


def _minimal_agn_df():
    return pd.DataFrame(
        {
            "object_id": ["agn_001", "agn_002"],
            "z": [0.6, 1.1],
            "z_err": [0.01, 0.01],
            "apparent_mag_2500": [20.1, 20.4],
            "apparent_mag_2500_err": [0.1, 0.1],
            "log_sigma_uv": [-0.60, -0.45],
            "log_sigma_uv_err": [0.03, 0.04],
            "log_tau_uv_rf": [2.05, 2.25],
            "log_tau_uv_rf_err": [0.05, 0.06],
        }
    )


def _agn_pivot_context(df_agn=None, z_range=(0.44, 3.16)):
    if df_agn is None:
        df_agn = _minimal_agn_df()
    return hubble_fit.build_agn_pivot_context(df_agn, z_range)


def _agn_pivot_checkpoint_payload_for_ids(object_ids, z_range=(0.44, 3.16)):
    object_ids = list(object_ids)
    count = len(object_ids)
    df_agn = pd.DataFrame(
        {
            "object_id": object_ids,
            "z": np.linspace(0.6, 1.1, count),
            "log_sigma_uv": np.linspace(-0.60, -0.45, count),
            "log_tau_uv_rf": np.linspace(2.05, 2.25, count),
        }
    )
    return {
        **hubble_fit._agn_pivot_checkpoint_payload(
            _agn_pivot_context(df_agn, z_range=z_range)
        ),
        "sigma_clip_pass_stage": "single",
    }


def _minimal_pantheon_df():
    return pd.DataFrame(
        {
            "zHD": [0.05, 0.1],
            "m_b_corr": [16.0, 17.0],
            "IS_CALIBRATOR": [0, 0],
            "CEPH_DIST": [-9.0, -9.0],
            "MU_SH0ES_ERR_DIAG": [0.1, 0.1],
        }
    )


def test_run_mcmc_pipeline_default_resume_uses_result_dir(monkeypatch, tmp_path):
    df_agn = _minimal_agn_df()
    agn_pivot_context = _agn_pivot_context(df_agn)
    pivot_payload = hubble_fit._agn_pivot_checkpoint_payload(agn_pivot_context)
    df_pantheon = _minimal_pantheon_df()
    result_root = tmp_path / "result_root"
    expected = result_root / "hubble_posteriors" / "unit" / "posteriors_FlatLambdaCDM_joint_fastest_all_z0p44_3p16_disable_completeness.h5"
    captured = {}

    monkeypatch.setattr(hubble_fit, "get_qvc_result_dir", lambda: result_root)
    monkeypatch.setattr(hubble_fit, "get_model_params", lambda *args, **kwargs: ({"H0": (60.0, 80.0)}, ["H0"], ["H0"]))
    monkeypatch.setattr(hubble_fit, "get_agn_model_spec", lambda *args, **kwargs: ((), (), ()))
    monkeypatch.setattr(hubble_fit, "make_dm_function", lambda *args, **kwargs: "interp")
    monkeypatch.setattr(hubble_fit, "plot_completeness_diagnostics", lambda *args, **kwargs: None)

    def fake_load_chains(path):
        captured["path"] = path
        return {
            "flat_samples": np.ones((3, 1)),
            "dmi_max_w": np.zeros(len(df_agn)),
            "dmi_posterior_median": np.zeros(len(df_agn)),
            "dmi_posterior_sigma": np.full(len(df_agn), 0.05),
            "integrals_max_w": np.ones(len(df_agn)),
            "logZ": -1.0,
            "logZerr": 0.2,
            **pivot_payload,
            "sigma_clip_pass_stage": "single",
            "object_id_fit_selection": np.asarray(
                agn_pivot_context.reference_object_ids,
                dtype=str,
            ),
        }

    monkeypatch.setattr(hubble_fit, "load_chains", fake_load_chains)
    monkeypatch.setattr(hubble_fit.os.path, "exists", lambda path: str(path) == str(expected))

    (
        flat_samples,
        model_labels,
        dm_interp,
        dmi_selection_sigma_interp,
        logz,
        logzerr,
        dmi_posterior_median,
        dmi_posterior_sigma,
        dmi_selection_sigma_posterior_median,
    ) = hubble_fit.run_mcmc_pipeline(
        df_agn=df_agn,
        df_agn_all=df_agn.copy(),
        df_pantheon=df_pantheon,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        agn_pivot_context=agn_pivot_context,
        cosmo_model="FlatLambdaCDM",
        completeness=False,
        use_full_cov=False,
        resume=True,
        speed="fastest",
        prefix="unit",
    )

    assert captured["path"] == str(expected)
    assert expected.parent.is_dir()
    assert flat_samples.shape == (3, 1)
    assert model_labels == ["H0"]
    assert dm_interp == "interp"
    assert dmi_selection_sigma_interp is None
    assert logz == -1.0
    assert logzerr == 0.2
    np.testing.assert_allclose(dmi_posterior_median, 0.0)
    np.testing.assert_allclose(dmi_posterior_sigma, 0.05)
    assert dmi_selection_sigma_posterior_median is None


def test_run_mcmc_pipeline_explicit_resume_path_bypasses_default(monkeypatch, tmp_path):
    df_agn = _minimal_agn_df()
    agn_pivot_context = _agn_pivot_context(df_agn)
    pivot_payload = hubble_fit._agn_pivot_checkpoint_payload(agn_pivot_context)
    df_pantheon = _minimal_pantheon_df()
    result_root = tmp_path / "result_root"
    explicit = tmp_path / "custom" / "resume_here.h5"
    explicit.parent.mkdir(parents=True)
    explicit.write_text("stub")
    captured = {}

    monkeypatch.setattr(hubble_fit, "get_qvc_result_dir", lambda: result_root)
    monkeypatch.setattr(hubble_fit, "get_model_params", lambda *args, **kwargs: ({"H0": (60.0, 80.0)}, ["H0"], ["H0"]))
    monkeypatch.setattr(hubble_fit, "get_agn_model_spec", lambda *args, **kwargs: ((), (), ()))
    monkeypatch.setattr(hubble_fit, "make_dm_function", lambda *args, **kwargs: "interp")
    monkeypatch.setattr(hubble_fit, "plot_completeness_diagnostics", lambda *args, **kwargs: None)

    def fake_load_chains(path):
        captured["path"] = path
        return {
            "flat_samples": np.ones((2, 1)),
            "dmi_max_w": np.zeros(len(df_agn)),
            "dmi_posterior_median": np.zeros(len(df_agn)),
            "dmi_posterior_sigma": np.full(len(df_agn), 0.05),
            "integrals_max_w": np.ones(len(df_agn)),
            "logZ": -2.0,
            "logZerr": 0.3,
            **pivot_payload,
            "sigma_clip_pass_stage": "single",
            "object_id_fit_selection": np.asarray(
                agn_pivot_context.reference_object_ids,
                dtype=str,
            ),
        }

    monkeypatch.setattr(hubble_fit, "load_chains", fake_load_chains)
    monkeypatch.setattr(hubble_fit.os.path, "exists", lambda path: str(path) == str(explicit))

    hubble_fit.run_mcmc_pipeline(
        df_agn=df_agn,
        df_agn_all=df_agn.copy(),
        df_pantheon=df_pantheon,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        agn_pivot_context=agn_pivot_context,
        cosmo_model="FlatLambdaCDM",
        completeness=False,
        use_full_cov=False,
        resume=str(explicit),
        speed="fastest",
        prefix="unit",
    )

    assert captured["path"] == str(explicit)


def test_run_mcmc_pipeline_new_checkpoint_writes_fit_object_ids(monkeypatch, tmp_path):
    df_agn = _minimal_agn_df()
    agn_pivot_context = _agn_pivot_context(df_agn)
    df_pantheon = _minimal_pantheon_df()
    captured = {}

    class DummyPool:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyContext:
        def Pool(self, processes):
            return DummyPool()

    class DummyResults:
        samples = np.array([[70.0], [71.0], [72.0]], dtype=float)
        logl = np.array([-3.0, -2.0, -1.0], dtype=float)
        logz = np.array([0.0], dtype=float)
        logzerr = np.array([0.1], dtype=float)
        logwt = np.log(np.array([1.0, 2.0, 3.0], dtype=float))
        blob = np.array(
            [
                [[1.0, 1.1], [0.01, 0.02], [0.1, 0.2]],
                [[2.0, 2.1], [0.03, 0.04], [0.3, 0.4]],
                [[3.0, 3.1], [0.05, 0.06], [0.5, 0.6]],
            ],
            dtype=float,
        )

    class DummySampler:
        def __init__(self, *args, **kwargs):
            self.results = DummyResults()

        def run_nested(self, *args, **kwargs):
            return None

    monkeypatch.setattr(hubble_fit, "DynamicNestedSampler", DummySampler)
    monkeypatch.setattr(hubble_fit.multiprocessing, "get_context", lambda *args, **kwargs: DummyContext())
    monkeypatch.setattr(hubble_fit, "get_model_params", lambda *args, **kwargs: ({"H0": (60.0, 80.0)}, ["H0"], ["H0"]))
    monkeypatch.setattr(hubble_fit, "get_agn_model_spec", lambda *args, **kwargs: ((), (), ()))
    monkeypatch.setattr(hubble_fit, "make_dm_function", lambda *args, **kwargs: "interp")
    monkeypatch.setattr(hubble_fit, "plot_dynesty", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "plot_completeness_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "evaluate_log_f", lambda *args, **kwargs: np.zeros(1, dtype=float))
    monkeypatch.setattr(hubble_fit.dyfunc, "resample_equal", lambda idx, weights: idx)
    monkeypatch.setattr(hubble_fit, "save_chains", lambda filename, **kwargs: captured.update(kwargs))

    hubble_fit.run_mcmc_pipeline(
        df_agn=df_agn,
        df_agn_all=df_agn.copy(),
        df_pantheon=df_pantheon,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        agn_pivot_context=agn_pivot_context,
        cosmo_model="FlatLambdaCDM",
        completeness=False,
        use_full_cov=False,
        speed="fastest",
        prefix="unit",
    )

    np.testing.assert_array_equal(captured["object_id_fit_selection"], df_agn["object_id"].astype(str).to_numpy())
    assert set(hubble_fit.AGN_PIVOT_CHECKPOINT_KEYS).issubset(captured)
    restored_context = hubble_fit._load_agn_pivot_context_from_checkpoint(
        captured,
        checkpoint_file="captured-test-checkpoint.h5",
    )
    assert restored_context == agn_pivot_context


def test_resume_replot_with_cuts_remaps_per_object_arrays_by_object_id(tmp_path):
    checkpoint = tmp_path / "posterior.h5"
    hubble_fit.save_chains(
        str(checkpoint),
        flat_samples=np.ones((3, 1)),
        object_id_fit_selection=np.array(["agn_a", "agn_b", "agn_c"]),
        dmi_max_w=np.array([10.0, 20.0, 30.0]),
        dmi_posterior_median=np.array([11.0, 21.0, 31.0]),
        dmi_posterior_sigma=np.array([0.1, 0.2, 0.3]),
        dmi_selection_sigma_posterior_median=np.array([1.1, 1.2, 1.3]),
        integrals_max_w=np.array([100.0, 200.0, 300.0]),
        logZ=-1.0,
        logZerr=0.1,
        **_agn_pivot_checkpoint_payload_for_ids(["agn_a", "agn_b", "agn_c"]),
    )
    current = pd.DataFrame({"object_id": ["agn_c", "agn_a"]})
    remapped = hubble_fit._remap_resume_replot_checkpoint(
        hubble_fit.load_chains(str(checkpoint)),
        str(checkpoint),
        current,
        ndim=1,
    )

    np.testing.assert_array_equal(remapped["object_id_fit_selection"], np.array(["agn_c", "agn_a"]))
    np.testing.assert_allclose(remapped["dmi_max_w"], [30.0, 10.0])
    np.testing.assert_allclose(remapped["dmi_posterior_median"], [31.0, 11.0])
    np.testing.assert_allclose(remapped["dmi_posterior_sigma"], [0.3, 0.1])
    np.testing.assert_allclose(remapped["dmi_selection_sigma_posterior_median"], [1.3, 1.1])
    np.testing.assert_allclose(remapped["integrals_max_w"], [300.0, 100.0])


def test_resume_replot_with_cuts_rejects_missing_current_object_id(tmp_path):
    checkpoint = tmp_path / "posterior.h5"
    hubble_fit.save_chains(
        str(checkpoint),
        flat_samples=np.ones((3, 1)),
        object_id_fit_selection=np.array(["agn_a", "agn_b"]),
        dmi_max_w=np.array([10.0, 20.0]),
        dmi_posterior_sigma=np.array([0.1, 0.2]),
        integrals_max_w=np.array([100.0, 200.0]),
        logZ=-1.0,
        logZerr=0.1,
        **_agn_pivot_checkpoint_payload_for_ids(["agn_a", "agn_b"]),
    )
    current = pd.DataFrame({"object_id": ["agn_a", "agn_missing"]})

    with pytest.raises(RuntimeError, match="Missing 1 / 2 current object IDs"):
        hubble_fit._remap_resume_replot_checkpoint(
            hubble_fit.load_chains(str(checkpoint)),
            str(checkpoint),
            current,
            ndim=1,
        )


def test_resume_replot_with_cuts_rejects_legacy_checkpoint_without_object_ids(tmp_path):
    checkpoint = tmp_path / "legacy.h5"
    hubble_fit.save_chains(
        str(checkpoint),
        flat_samples=np.ones((3, 1)),
        dmi_max_w=np.array([10.0, 20.0]),
        dmi_posterior_sigma=np.array([0.1, 0.2]),
        integrals_max_w=np.array([100.0, 200.0]),
        logZ=-1.0,
        logZerr=0.1,
        **_agn_pivot_checkpoint_payload_for_ids(["agn_a", "agn_b"]),
    )
    current = pd.DataFrame({"object_id": ["agn_a"]})

    with pytest.raises(RuntimeError, match="object_id_fit_selection"):
        hubble_fit._remap_resume_replot_checkpoint(
            hubble_fit.load_chains(str(checkpoint)),
            str(checkpoint),
            current,
            ndim=1,
        )


def test_restrict_agn_to_resume_replot_sample_applies_current_cuts_with_checkpoint_order(tmp_path):
    checkpoint = tmp_path / "posterior.h5"
    hubble_fit.save_chains(
        str(checkpoint),
        flat_samples=np.ones((3, 1)),
        object_id_fit_selection=np.array(["agn_c", "agn_a", "agn_b"]),
        dmi_max_w=np.array([30.0, 10.0, 20.0]),
        dmi_posterior_sigma=np.array([0.3, 0.1, 0.2]),
        integrals_max_w=np.array([300.0, 100.0, 200.0]),
        logZ=-1.0,
        logZerr=0.1,
        **_agn_pivot_checkpoint_payload_for_ids(["agn_c", "agn_a", "agn_b"]),
    )
    current_after_cuts = pd.DataFrame(
        {
            "object_id": ["agn_b", "agn_x", "agn_c"],
            "z": [0.5, 0.6, 0.7],
        }
    )

    restricted = hubble_fit.restrict_agn_to_resume_replot_sample(current_after_cuts, str(checkpoint))

    assert restricted["object_id"].tolist() == ["agn_c", "agn_b"]
    assert restricted["z"].tolist() == [0.7, 0.5]


def test_run_all_saves_cosmo_results_under_result_dir(monkeypatch, tmp_path):
    df_agn = _minimal_agn_df()
    df_pantheon = _minimal_pantheon_df()
    result_root = tmp_path / "result_root"
    captured = {}

    monkeypatch.setattr(hubble_fit, "get_qvc_result_dir", lambda: result_root)
    monkeypatch.setattr(
        hubble_fit,
        "run_single",
        lambda *args, **kwargs: (np.ones((4, 2)), ["H0", "Om0"], "interp", -3.0, 0.1, None, 13.8, 0.2),
    )
    monkeypatch.setattr(hubble_fit, "plot_cosmo_corner", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "compare_models_by_log_evidence_all", lambda *args, **kwargs: {})
    monkeypatch.setattr(hubble_fit, "write_results_tex_variables", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "extract_cosmo_results_from_samples", lambda *args, **kwargs: {})
    monkeypatch.setattr(hubble_fit, "sym_percentile", lambda *args, **kwargs: (70.0, 1.0, 1.0, 1.0))
    monkeypatch.setattr(
        hubble_fit,
        "save_cosmo_results_hdf5",
        lambda filename, models_dict: captured.update({"filename": filename, "models": models_dict}),
    )

    hubble_fit.run_all(
        df_agn=df_agn,
        df_agn_all=df_agn.copy(),
        df_pantheon=df_pantheon,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_models=["FlatLambdaCDM"],
        skip_plots=True,
        z_range=(0.44, 3.16),
        speed="fastest",
        prefix="unit",
    )

    expected = result_root / "cosmo" / "unit" / "cosmo_results_all_z0p44_3p16.hdf5"
    assert captured["filename"] == str(expected)
    assert expected.parent.is_dir()


def test_run_all_compare_sigma_only_still_compares_models_and_skips_corner_plots(monkeypatch, tmp_path):
    df_agn = _minimal_agn_df()
    df_pantheon = _minimal_pantheon_df()
    compare_calls = []
    corner_calls = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        hubble_fit,
        "run_single",
        lambda *args, **kwargs: (
            np.ones((4, 2)),
            ["H0", "Om0"],
            "interp",
            -3.0 if not kwargs.get("only_sna") else -2.0,
            0.1,
            None,
            13.8,
            0.2,
        ),
    )
    monkeypatch.setattr(
        hubble_fit,
        "plot_cosmo_corner",
        lambda *args, **kwargs: corner_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        hubble_fit,
        "compare_models_by_log_evidence_all",
        lambda *args, **kwargs: compare_calls.append((args, kwargs)) or {"ranking": [], "pairwise": {}},
    )
    monkeypatch.setattr(hubble_fit, "write_results_tex_variables", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "extract_cosmo_results_from_samples", lambda *args, **kwargs: {})
    monkeypatch.setattr(hubble_fit, "sym_percentile", lambda *args, **kwargs: (70.0, 1.0, 1.0, 1.0))
    monkeypatch.setattr(hubble_fit, "save_cosmo_results_hdf5", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "get_qvc_result_dir", lambda: tmp_path / "result_root")

    hubble_fit.run_all(
        df_agn=df_agn,
        df_agn_all=df_agn.copy(),
        df_pantheon=df_pantheon,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_models=["FlatLambdaCDM", "Flatw0waCDM"],
        skip_plots=False,
        compare_sigma_only=True,
        z_range=(0.44, 3.16),
        speed="fastest",
        prefix="unit",
    )

    assert len(compare_calls) == 2
    assert corner_calls == []


def test_run_all_minimal_plots_keeps_joint_and_sna_fits_and_skips_corners(monkeypatch, tmp_path):
    df_agn = _minimal_agn_df()
    df_pantheon = _minimal_pantheon_df()
    fit_calls = []
    compare_calls = []
    corner_calls = []

    monkeypatch.chdir(tmp_path)

    def fake_run_single(*args, **kwargs):
        fit_calls.append(
            {
                "cosmo_model": kwargs["cosmo_model"],
                "only_sna": kwargs["only_sna"],
                "minimal_plots": kwargs["minimal_plots"],
            }
        )
        return (
            np.ones((4, 2)),
            ["H0", "Om0"],
            "interp",
            -3.0 if not kwargs["only_sna"] else -2.0,
            0.1,
            None,
            13.8,
            0.2,
        )

    monkeypatch.setattr(hubble_fit, "run_single", fake_run_single)
    monkeypatch.setattr(
        hubble_fit,
        "plot_cosmo_corner",
        lambda *args, **kwargs: corner_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        hubble_fit,
        "compare_models_by_log_evidence_all",
        lambda *args, **kwargs: compare_calls.append((args, kwargs))
        or {"ranking": [], "pairwise": {}},
    )
    monkeypatch.setattr(hubble_fit, "write_results_tex_variables", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "extract_cosmo_results_from_samples", lambda *args, **kwargs: {})
    monkeypatch.setattr(hubble_fit, "sym_percentile", lambda *args, **kwargs: (70.0, 1.0, 1.0, 1.0))
    monkeypatch.setattr(hubble_fit, "save_cosmo_results_hdf5", lambda *args, **kwargs: None)
    monkeypatch.setattr(hubble_fit, "get_qvc_result_dir", lambda: tmp_path / "result_root")

    hubble_fit.run_all(
        df_agn=df_agn,
        df_agn_all=df_agn.copy(),
        df_pantheon=df_pantheon,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_models=["FlatLambdaCDM", "Flatw0waCDM"],
        minimal_plots=True,
        z_range=(0.44, 3.16),
        speed="fastest",
        prefix="unit",
    )

    assert fit_calls == [
        {"cosmo_model": "FlatLambdaCDM", "only_sna": False, "minimal_plots": True},
        {"cosmo_model": "FlatLambdaCDM", "only_sna": True, "minimal_plots": True},
        {"cosmo_model": "Flatw0waCDM", "only_sna": False, "minimal_plots": True},
        {"cosmo_model": "Flatw0waCDM", "only_sna": True, "minimal_plots": True},
    ]
    assert len(compare_calls) == 2
    assert corner_calls == []
