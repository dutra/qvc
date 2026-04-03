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
        "log_Lxray",
        "log_Lxray_err",
        "alphaOX",
        "alphaOX_err",
        "alphaOX_exp",
        "alphaOX_exp_err",
        "delta_alphaOX",
        "delta_alphaOX_err",
    ):
        assert col in out.columns
        assert out[col].isna().all()


def _minimal_agn_df():
    return pd.DataFrame(
        {
            "object_id": ["agn_001", "agn_002"],
            "z": [0.6, 1.1],
            "z_err": [0.01, 0.01],
            "apparent_mag_2500": [20.1, 20.4],
            "apparent_mag_2500_err": [0.1, 0.1],
        }
    )


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
    df_pantheon = _minimal_pantheon_df()
    result_root = tmp_path / "result_root"
    expected = result_root / "hubble_posteriors" / "unit" / "posteriors_FlatLambdaCDM_joint_fast_all_z0p44_3p16.h5"
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
        dmi_posterior_sigma,
        dmi_selection_sigma_posterior_median,
    ) = hubble_fit.run_mcmc_pipeline(
        df_agn=df_agn,
        df_agn_all=df_agn.copy(),
        df_pantheon=df_pantheon,
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM",
        completeness=False,
        use_full_cov=False,
        resume=True,
        speed="fast",
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
    np.testing.assert_allclose(dmi_posterior_sigma, 0.05)
    assert dmi_selection_sigma_posterior_median is None


def test_run_mcmc_pipeline_explicit_resume_path_bypasses_default(monkeypatch, tmp_path):
    df_agn = _minimal_agn_df()
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
        cosmo_model="FlatLambdaCDM",
        completeness=False,
        use_full_cov=False,
        resume=str(explicit),
        speed="fast",
        prefix="unit",
    )

    assert captured["path"] == str(explicit)


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
        speed="fast",
        prefix="unit",
    )

    expected = result_root / "cosmo" / "unit" / "cosmo_results_all_z0p44_3p16.hdf5"
    assert captured["filename"] == str(expected)
    assert expected.parent.is_dir()
