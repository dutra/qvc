import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

import run_resume_spectra_local as local
from qvc.spectra.catalog_hdf5 import (
    JOINT_POSTERIOR_DRAW_COUNT,
    JOINT_POSTERIOR_DRAW_FIELDS,
    PSF_AGN_FRACTION_DRAW_COUNT,
    write_spectra_catalog_hdf5,
)
from qvc.spectra.fit_spectra_jaxsedfit_joint import (
    GRAHSP_ATTENUATION_NORMALIZATION,
    POSTERIOR_BUNDLE_FORMAT,
    QVC_PSF_HOST_CAPTURE_GROUP,
    summarize_m2500_dereddened,
)


def test_direct_dry_run_bootstraps_src_layout_without_pythonpath(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(local.__file__).resolve()),
            "--dry-run",
            "--source-run",
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Source run has no posterior-bundle directory" in result.stderr
    assert "No module named 'qvc'" not in result.stderr


def test_parallel_argument_defaults_to_eight_and_accepts_positive_count():
    assert local.parse_args([]).parallel == 8
    assert local.parse_args([]).max_tasks_per_worker == 1
    assert local.parse_args(["--parallel", "4"]).parallel == 4


def test_worker_runtime_disables_xla_memory_preallocation(tmp_path):
    environment = local.runtime_environment(tmp_path)
    assert environment["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_direct_v3_is_the_only_workflow_and_needs_no_v2_catalog():
    args = local.parse_args([])
    assert not hasattr(args, "source_catalog")
    assert not hasattr(args, "allow_spectra_catalog_v2")
    assert args.output_catalog.endswith("_v3.h5")


def test_parallel_argument_rejects_nonpositive_count():
    with pytest.raises(SystemExit):
        local.parse_args(["--parallel", "0"])


def test_discover_resume_objects_uses_bundle_set_in_input_order(tmp_path):
    input_csv = tmp_path / "input.csv"
    input_csv.write_text(
        "object_id,sdss_name\n"
        "10,b'000000.00+000000.0'\n"
        "20,b'111111.11-111111.1'\n"
        "30,b'222222.22+222222.2'\n"
    )
    bundle_dir = tmp_path / "all"
    bundle_dir.mkdir()
    (bundle_dir / "z2.000_222222.22+222222.2_joint_samples.h5").touch()
    (bundle_dir / "z1.000_000000.00+000000.0_joint_samples.h5").touch()

    available, missing = local.discover_resume_objects(bundle_dir, input_csv)

    assert [item.object_id for item in available] == ["10", "30"]
    assert [item.sdss_name for item in available] == [
        "000000.00+000000.0",
        "222222.22+222222.2",
    ]
    assert missing == [
        {"object_id": "20", "sdss_name": "111111.11-111111.1"}
    ]


def test_discover_resume_objects_rejects_unmapped_bundle(tmp_path):
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("object_id,sdss_name\n10,known\n")
    bundle_dir = tmp_path / "all"
    bundle_dir.mkdir()
    (bundle_dir / "z1.000_unknown_joint_samples.h5").touch()

    with pytest.raises(ValueError, match="has no row"):
        local.discover_resume_objects(bundle_dir, input_csv)


def test_direct_v3_sources_select_false_fit_row_with_matching_bundle(tmp_path):
    source_run = tmp_path / "run"
    source_run.mkdir()
    bundle_dir = source_run / "all"
    bundle_dir.mkdir()
    bundle = bundle_dir / "z1.000_name_joint_samples.h5"
    with h5py.File(bundle, "w") as handle:
        handle.attrs["qvc_provenance_json"] = json.dumps(
            {
                "runtime": {
                    "dependencies": {
                            "JAXSEDFit": {
                            "git": {"commit": local.EXPECTED_JAXSEDFIT_COMMIT}
                        }
                    }
                }
            }
        )
    chunk = source_run / "run_chunk0000.h5"
    with h5py.File(chunk, "w") as handle:
        handle.attrs["qvc_spectra_catalog_format"] = "qvc_spectra_catalog_v2"
        handle.attrs["psf_agn_fraction_draw_count"] = 64
        handle.attrs["f_host_2500_psf_draw_count"] = 64
        handle.attrs["f_host_2500_psf_draw_selection"] = (
            "deterministic_uniform_without_replacement"
        )
        handle.attrs["qvc_provenance_json"] = (
            '{"module":{"parsed_args":{"seed":3}}}'
        )
        catalog = handle.create_group("catalog")
        catalog.create_dataset("object_id", data=[b"10"])
        catalog.create_dataset("sdss_name", data=[b"name"])
        catalog.create_dataset("z", data=[1.0])
        catalog.create_dataset("fit_ok", data=[False])
        catalog.create_dataset("legacy_scalar", data=[42.0])
        psf = handle.create_group("psf_agn_fraction_draws")
        psf.create_dataset("bands", data=[b"u", b"g", b"r", b"i", b"z"])
        psf_values = np.full((1, 64, 5), np.nan, dtype=np.float32)
        psf_values[:, :1] = 0.5
        psf.create_dataset("values", data=psf_values)
        psf.create_dataset("valid_count", data=[1])
        host = handle.create_group("f_host_2500_psf_draws")
        host_values = np.full((1, 64), np.nan, dtype=np.float32)
        host_values[:, :1] = 0.5
        host.create_dataset("values", data=host_values)
        host.create_dataset("valid_count", data=[1])

    sources, chunks = local.load_direct_v3_sources(source_run)

    assert chunks == [chunk.resolve()]
    assert len(sources) == 1
    assert sources[0]["object_id"] == "10"
    assert sources[0]["bundle_path"] == bundle.resolve()
    assert sources[0]["row"]["fit_ok"] is False


def test_fit_command_is_strict_catalog_only_resume(tmp_path):
    source_run = tmp_path / "old"
    source_bundles = source_run / "all"
    output_run = tmp_path / "new"
    batch = [
        local.ResumeObject(
            ordinal=0,
            object_id="10",
            sdss_name="name",
            bundle_path=source_bundles / "z1.000_name_joint_samples.h5",
        )
    ]

    command = local.build_fit_command(
        python_bin=Path("/fit/python"),
        shard_path=output_run / "shard.h5",
        batch=batch,
        source_run=source_run,
        source_bundle_dir=source_bundles,
        output_run=output_run,
        input_csv=tmp_path / "input.csv",
        sed_photometry=tmp_path / "phot.csv",
        dr16q_fits=tmp_path / "dr16q.fits",
        cache_dir=tmp_path / "cache",
        prepared_records=tmp_path / "prepared.csv",
        verbose=False,
    )

    assert command[:3] == [
        "/fit/python",
        "-m",
        "qvc.spectra.fit_spectra_jaxsedfit_joint",
    ]
    assert "--resume-only" in command
    assert "--allow-unannotated-resume-bundle" in command
    assert "--resume-records-path" in command
    assert "--no-save-fig" in command
    assert "--no-save-jaxsedfit-samples" in command
    assert "--no-catalog-progress" in command
    assert "--no-print-convergence-summary" in command
    assert "--progress" not in command
    assert command[command.index("--filter_object_id") + 1] == "10"


def test_batch_shard_name_is_deterministic_and_batch_specific(tmp_path):
    first = local.ResumeObject(0, "10", "a", tmp_path / "a.h5")
    second = local.ResumeObject(1, "20", "b", tmp_path / "b.h5")

    name = local.batch_shard_name([first, second])

    assert name == local.batch_shard_name([first, second])
    assert name != local.batch_shard_name([first])
    assert name.startswith("resume_00000_002_10_")
    assert name.endswith(".h5")


def test_contiguous_repair_runs_split_around_fallback_gap(tmp_path):
    values = [
        local.RepairedCatalogRow(
            item=local.ResumeObject(index, str(index), str(index), tmp_path / f"{index}.h5"),
            row={},
            fraction_draws=np.empty((64, 5)),
            fraction_valid_count=64,
            host_draws=np.empty(64),
            host_valid_count=64,
            source_catalog=tmp_path / "source.h5",
            source_catalog_commit="old",
        )
        for index in (3, 0, 1)
    ]

    runs = local.contiguous_repair_runs(values)

    assert [[value.item.ordinal for value in run] for run in runs] == [
        [0, 1],
        [3],
    ]


def _write_analytic_repair_bundle(path, samples):
    with h5py.File(path, "w") as handle:
        handle.attrs["posterior_bundle_format"] = POSTERIOR_BUNDLE_FORMAT
        handle.attrs["qvc_host_capture_group"] = QVC_PSF_HOST_CAPTURE_GROUP
        handle.attrs["qvc_git_commit"] = "source-head"
        group = handle.create_group("samples")
        for name, values in samples.items():
            group.create_dataset(name, data=values)


def test_analytic_repair_preserves_draws_and_corrects_m2500(tmp_path):
    samples = {
        "log_agn_amp": np.log(np.array([1.0e38, 1.1e38, 0.9e38, 1.2e38])),
        "pl_slope": np.array([-1.8, -1.7, -1.9, -1.6]),
        "pl_bend_loc": np.full(4, 1000.0),
        "pl_bend_width": np.full(4, 10.0),
        "log_ebv_gal": np.log(np.array([0.02, 0.03, 0.04, 0.05])),
        "log_ebv_agn": np.log(np.array([0.03, 0.04, 0.05, 0.06])),
        "host_capture_group_fraction": np.full((4, 1), 0.7),
    }
    bundle = tmp_path / "z1.000_name_joint_samples.h5"
    _write_analytic_repair_bundle(bundle, samples)
    physical = dict(samples)
    physical["ebv_gal"] = np.exp(samples["log_ebv_gal"])
    physical["ebv_agn"] = np.exp(samples["log_ebv_agn"])
    expected = summarize_m2500_dereddened(physical, 1.0)
    row = {
        "object_id": "10",
        "sdss_name": "name",
        "z": 1.0,
        "fit_ok": True,
        "m_2500_dereddened": expected["m_2500_dereddened"],
        "m_2500_attenuated_model": 99.0,
        "a_2500_total": (
            expected["a_2500_total"] / GRAHSP_ATTENUATION_NORMALIZATION
        ),
        "f_host_2500_psf": 0.2,
        "f_host_2500_psf_err": 0.03,
        "joint_reduced_chi2": 1.1,
        **{f"f_AGN_psf_{band}": 0.8 for band in "ugriz"},
    }
    fraction_draws = np.arange(64 * 5, dtype=np.float32).reshape(1, 64, 5)
    host_draws = np.linspace(0.0, 1.0, 64, dtype=np.float32)[None, :]
    catalog = SimpleNamespace(
        frame=pd.DataFrame([row]),
        fraction_draws=fraction_draws,
        valid_count=np.array([64]),
        f_host_2500_psf_draws=host_draws,
        f_host_2500_psf_valid_count=np.array([64]),
    )
    item = local.ResumeObject(0, "10", "name", bundle)

    repaired = local.repair_source_catalog_row(
        item,
        catalog,
        0,
        source_catalog_path=tmp_path / "source.h5",
        source_catalog_commit="source-head",
        source_run=tmp_path / "old-run",
    )

    assert repaired.row["m_2500_attenuated_model"] == pytest.approx(
        expected["m_2500_attenuated_model"]
    )
    assert repaired.row["a_2500_total"] == pytest.approx(
        expected["a_2500_total"]
    )
    assert repaired.row["execution_mode"] == "resumed"
    assert repaired.row["catalog_repair_mode"] == "analytic_m2500_norm12"
    np.testing.assert_array_equal(repaired.fraction_draws, fraction_draws[0])
    np.testing.assert_array_equal(repaired.host_draws, host_draws[0])


def test_analytic_repair_rejects_catalog_without_old_normalization(tmp_path):
    samples = {
        "log_agn_amp": np.log(np.array([1.0e38, 1.1e38])),
        "pl_slope": np.array([-1.8, -1.7]),
        "log_ebv_gal": np.log(np.array([0.02, 0.03])),
        "log_ebv_agn": np.log(np.array([0.03, 0.04])),
        "host_capture_group_fraction": np.full((2, 1), 0.7),
    }
    bundle = tmp_path / "z1.000_name_joint_samples.h5"
    _write_analytic_repair_bundle(bundle, samples)
    physical = dict(samples)
    physical["ebv_gal"] = np.exp(samples["log_ebv_gal"])
    physical["ebv_agn"] = np.exp(samples["log_ebv_agn"])
    expected = summarize_m2500_dereddened(physical, 1.0)
    catalog = SimpleNamespace(
        frame=pd.DataFrame(
            [
                {
                    "object_id": "10",
                    "sdss_name": "name",
                    "z": 1.0,
                    "fit_ok": True,
                        "m_2500_dereddened": expected["m_2500_dereddened"],
                        "m_2500_attenuated_model": expected[
                            "m_2500_attenuated_model"
                        ],
                        "a_2500_total": expected["a_2500_total"],
                        "f_host_2500_psf": 0.2,
                        "f_host_2500_psf_err": 0.03,
                        "joint_reduced_chi2": 1.1,
                        **{f"f_AGN_psf_{band}": 0.8 for band in "ugriz"},
                }
            ]
        ),
        fraction_draws=np.zeros((1, 64, 5), dtype=np.float32),
        valid_count=np.array([64]),
        f_host_2500_psf_draws=np.zeros((1, 64), dtype=np.float32),
        f_host_2500_psf_valid_count=np.array([64]),
    )

    with pytest.raises(ValueError, match="1.0-normalized"):
        local.repair_source_catalog_row(
            local.ResumeObject(0, "10", "name", bundle),
            catalog,
            0,
            source_catalog_path=tmp_path / "source.h5",
            source_catalog_commit="source-head",
            source_run=tmp_path / "old-run",
        )


def test_prepare_resume_records_uses_exact_spectrum_key(tmp_path):
    input_csv = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "object_id": ["10"],
            "sdss_name": ["b'000000.00+000000.0'"],
            "plate": [1234],
            "fiberid": [56],
            "mjd": [56789],
            "SDSSS_RUN2D": ["v5_13_2"],
        }
    ).to_csv(input_csv, index=False)
    dr16q = tmp_path / "dr16q.fits"
    Table(
        {
            "PLATE": [9999, 1234],
            "MJD": [50000, 56789],
            "FIBERID": [1, 56],
            "Z_SYS": [2.0, 1.25],
            "LOGLBOL": [46.0, 45.5],
            "RA": [20.0, 10.0],
            "DEC": [2.0, 1.0],
            "SDSS_NAME": ["other", "000000.00+000000.0"],
        }
    ).write(dr16q)
    available = [
        local.ResumeObject(
            ordinal=0,
            object_id="10",
            sdss_name="000000.00+000000.0",
            bundle_path=tmp_path / "z1.250_000000.00+000000.0_joint_samples.h5",
        )
    ]
    output = tmp_path / "prepared.csv"

    local.prepare_resume_records(output, available, input_csv, dr16q)

    row = pd.read_csv(output, dtype={"object_id": str}).iloc[0]
    assert row["object_id"] == "10"
    assert row["sdss_name"] == "000000.00+000000.0"
    assert row["z"] == pytest.approx(1.25)
    assert row["loglbol"] == pytest.approx(45.5)
    assert row["ra"] == pytest.approx(10.0)
    assert row["SDSS_RUN2D"] == "v5_13_2"


def _write_resume_catalog(path, *, fit_ok=True, execution_mode="resumed"):
    scalar_values = {
        "f_host_2500_psf": 0.5,
        "alpha_nu_intrinsic_1450_2500": -0.5,
        "alpha_nu_attenuated_1450_2500": -0.5,
        "m_2500_dereddened": 20.0,
        "m_2500_attenuated_model": 20.0,
        "a_2500_galaxy": 0.0,
        "a_2500_internal": 0.0,
        "a_2500_total": 0.0,
    }
    frame = pd.DataFrame(
        {
            "object_id": ["10"],
            "fit_ok": [fit_ok],
            "execution_mode": [execution_mode],
            "resumed_from_path": ["old/all/source_samples.h5"],
            "mw_deredden_applied": [True],
            "joint_posterior_draw_source": ["test"],
            **{
                name: [value]
                for name, value in scalar_values.items()
            },
            **{
                f"{name}_{suffix}": [0.0]
                for name in JOINT_POSTERIOR_DRAW_FIELDS
                for suffix in ("err", "err_lower", "err_upper")
            },
        }
    )
    fraction_draws = np.full(
        (1, PSF_AGN_FRACTION_DRAW_COUNT, 5), np.nan, dtype=np.float32
    )
    fraction_draws[:, :1, :] = 0.5
    joint_draws = {
        name: np.full(
            (1, JOINT_POSTERIOR_DRAW_COUNT), np.nan, dtype=np.float32
        )
        for name in JOINT_POSTERIOR_DRAW_FIELDS
    }
    for name, value in scalar_values.items():
        joint_draws[name][:, :1] = value
    posterior_index = np.full(
        (1, JOINT_POSTERIOR_DRAW_COUNT), -1, dtype=np.int32
    )
    posterior_index[:, 0] = 3
    fitted_fluxes = np.full((1, 64, 5), np.nan, dtype=np.float32)
    fitted_fluxes[:, 0, :] = 1.0
    write_spectra_catalog_hdf5(
        path,
        frame,
        fraction_draws,
        np.array([1]),
        joint_posterior_draws=joint_draws,
        joint_posterior_valid_count=np.array([1]),
        joint_posterior_index=posterior_index,
        joint_posterior_source_draw_count=np.array([8]),
        joint_posterior_selection_seed=3,
        joint_psf_photometry_draws=fitted_fluxes,
        joint_psf_photometry_provenance={
            "prediction_source": "synthetic_test",
            "jaxsedfit_git_commit": "a" * 40,
        },
        provenance={"qvc_git": {"commit": "expected-head"}},
    )


def test_resume_catalog_validation_checks_hdf_content(tmp_path):
    path = tmp_path / "valid.h5"
    _write_resume_catalog(path)

    local.validate_resume_catalog(path, ["10"], "expected-head")

    bad = tmp_path / "bad.h5"
    _write_resume_catalog(bad)
    with h5py.File(bad, "r+") as handle:
        handle["catalog"]["fit_ok"][0] = False
    with pytest.raises(ValueError, match="Unsuccessful spectral rows"):
        local.validate_resume_catalog(bad, ["10"], "expected-head")


def test_completed_marker_validation_reads_catalog(tmp_path):
    path = tmp_path / "valid.h5"
    _write_resume_catalog(path)
    marker = tmp_path / "valid.json"
    local.atomic_write_json(
        marker,
        {"object_ids": ["10"], "qvc_git_head": "expected-head"},
    )

    assert local.completed_marker_is_valid(
        marker, path, ["10"], "expected-head"
    )


def test_subset_posterior_samples_uses_exact_common_indices():
    samples = {
        "log_agn_amp": np.arange(80.0),
        "matrix": np.arange(160.0).reshape(80, 2),
    }
    index = np.array([0, 7, 23, 79])

    selected = local.subset_posterior_samples(samples, index)

    np.testing.assert_array_equal(selected["log_agn_amp"], index.astype(float))
    np.testing.assert_array_equal(selected["matrix"], samples["matrix"][index])
    with pytest.raises(ValueError, match="strictly increasing"):
        local.subset_posterior_samples(samples, [7, 7])
    with pytest.raises(ValueError, match="outside"):
        local.subset_posterior_samples(samples, [80])


def test_build_v3_row_uses_chunk_host_and_analytic_bundle_draws(tmp_path):
    import qvc.spectra.fit_spectra_jaxsedfit_joint as spectral

    draw_count = 80
    samples = {
        "log_agn_amp": np.log(np.linspace(0.8e38, 1.2e38, draw_count)),
        "pl_slope": np.linspace(-2.0, -1.5, draw_count),
        "pl_bend_loc": np.full(draw_count, 1000.0),
        "pl_bend_width": np.full(draw_count, 10.0),
        "uv_slope": np.zeros(draw_count),
        "pl_cutoff": np.full(draw_count, 100_000.0),
        "log_ebv_gal": np.log(np.linspace(0.01, 0.03, draw_count)),
        "log_ebv_agn": np.log(np.linspace(0.02, 0.04, draw_count)),
        "host_capture_group_fraction": np.full((draw_count, 1), 0.7),
    }
    bundle = tmp_path / "z1.000_name_joint_samples.h5"
    bundle.touch()

    psf_draws = np.full((64, 5), 0.8, dtype=np.float32)
    expected_index = spectral.deterministic_compact_posterior_indices(
        draw_count, object_id="10", seed=3
    )
    host_draws = np.linspace(0.1, 0.3, 64, dtype=np.float32)
    task = local.V3BuildTask(
        ordinal=0,
        object_id="10",
        sdss_name="name",
        bundle_path=bundle,
        source_chunk_path=tmp_path / "chunk0000.h5",
        row={
            "object_id": "10",
            "sdss_name": "name",
            "z": 1.0,
            "fit_ok": True,
            "mw_deredden_applied": True,
        },
        psf_fraction_draws=psf_draws,
        psf_fraction_valid_count=64,
        host_fraction_draws=host_draws,
        host_fraction_valid_count=64,
        shard_path=tmp_path / "shard.h5",
        seed=3,
    )

    row = local.build_v3_row(task, samples)

    assert row["_joint_posterior_valid_count"] == 64
    assert row["_joint_posterior_source_draw_count"] == draw_count
    np.testing.assert_array_equal(
        row["_joint_posterior_index"], expected_index
    )
    np.testing.assert_array_equal(row["_psf_agn_fraction_draws"], psf_draws)
    np.testing.assert_array_equal(
        row["_joint_posterior_draws"]["f_host_2500_psf"], host_draws
    )
    assert np.isfinite(row["alpha_nu_intrinsic_1450_2500"])
    assert np.isfinite(row["alpha_nu_attenuated_1450_2500"])
    assert row["joint_posterior_draw_source"] == (
        "chunk_host_plus_bundle_analytic_selected64"
    )


def test_v3_shard_path_is_stable_and_identity_specific(tmp_path):
    first = local.v3_shard_path(tmp_path, ordinal=2, object_id="10")
    assert first == local.v3_shard_path(
        tmp_path, ordinal=2, object_id="10"
    )
    assert first != local.v3_shard_path(
        tmp_path, ordinal=3, object_id="10"
    )
    assert first.name.startswith("v3_00002_10_")


def test_v3_state_directory_is_visible_sibling(tmp_path):
    output = tmp_path / "spectra_resumed_v3.h5"
    state = local.v3_state_directory(output)

    assert state.parent == tmp_path
    assert state.name == "spectra_resumed_v3_fitted_color_chunk_v3_state"
    assert not state.name.startswith(".")


def test_atomic_v3_merge_restores_unchanged_chunk_scalar_columns(tmp_path):
    shard = tmp_path / "shard.h5"
    _write_resume_catalog(shard)
    output = tmp_path / "merged_v3.h5"
    local.merge_v3_shards(
        [shard],
        output,
        expected_object_ids=["10"],
        selection_seed=3,
        base_rows=[{"object_id": "10", "legacy_scalar": 42.0}],
        source_run=tmp_path,
        source_chunks=[],
        args=SimpleNamespace(parallel=1),
    )

    from qvc.spectra.catalog_hdf5 import read_spectra_catalog_hdf5

    merged = read_spectra_catalog_hdf5(output)
    assert merged.frame.loc[0, "legacy_scalar"] == pytest.approx(42.0)
    assert merged.frame.loc[0, "joint_posterior_draw_source"] == "test"
