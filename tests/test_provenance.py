import argparse
import json
from pathlib import Path

import h5py

from qvc import provenance


def _minimal_runtime():
    return {
        "python_executable": "/python",
        "python_version": "3.test",
        "platform": "test",
        "hostname": "host",
        "working_directory": "/work",
        "environment": {},
        "dependencies": {"qvc": {"git": {"commit": "abc", "dirty": False}}},
    }


def test_canonical_serialization_and_sensitive_redaction():
    record = {
        "z": Path("input.csv"),
        "api_token": "do-not-store",
        "nested": {"password": "also-secret", "safe": 4},
    }
    payload = provenance.canonical_json(record)
    assert payload == provenance.canonical_json(record)
    decoded = json.loads(payload)
    assert decoded["api_token"] == "<redacted>"
    assert decoded["nested"]["password"] == "<redacted>"
    assert decoded["nested"]["safe"] == 4


def test_redact_argv_supports_separate_and_equals_values():
    argv = ["run", "--api-key", "one", "--password=two", "--safe", "three"]
    assert provenance.redact_argv(argv) == [
        "run", "--api-key", "<redacted>", "--password=<redacted>", "--safe", "three"
    ]


def test_fingerprint_hash_threshold(tmp_path, monkeypatch):
    small = tmp_path / "small.dat"
    small.write_bytes(b"abc")
    assert provenance.fingerprint_path(small)["sha256"]

    monkeypatch.setattr(provenance, "HASH_LIMIT_BYTES", 2)
    large = provenance.fingerprint_path(small)
    assert large["sha256"] is None
    assert "larger_than_2_bytes" == large["sha256_skipped_reason"]


def test_hdf5_round_trip_and_history(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "runtime_state", _minimal_runtime)
    args = argparse.Namespace(value=3, password="hidden")
    old = provenance.build_run_record("entry", args, argv=["python", "old"], event_type="fit")
    current = provenance.build_run_record("entry", args, argv=["python", "new"], event_type="resume")
    record = provenance.merge_history(current, old)
    path = tmp_path / "result.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=[1])
        provenance.write_hdf5_provenance(handle, record)

    loaded = provenance.read_hdf5_provenance(path)
    assert [event["type"] for event in loaded["events"]] == ["fit", "resume"]
    assert loaded["module"]["parsed_args"]["password"] == "<redacted>"
    with h5py.File(path, "r") as handle:
        assert handle.attrs["qvc_provenance_schema"] == provenance.PROVENANCE_SCHEMA
        assert handle.attrs["qvc_rerun_command"] == "python new"
        assert handle.attrs["qvc_git_commit"] == "abc"


def test_submission_base64_round_trip_and_object_specialization(monkeypatch):
    submission = provenance.submission_record(
        "wrapper", ["wrapper", "--token", "secret"], {"memory": "20G"}
    )
    monkeypatch.setenv(provenance.PROVENANCE_ENV, provenance.encode_record(submission))
    monkeypatch.setattr(provenance, "runtime_state", _minimal_runtime)
    record = provenance.build_run_record(
        "module",
        argparse.Namespace(filter_object_id=["1", "2"]),
        argv=["python", "-m", "module", "--filter_object_id", "1", "2", "--N", "2"],
        object_id="2",
    )
    assert record["submission"]["argv"][-1] == "<redacted>"
    assert record["module"]["rerun_argv"] == [
        "python", "-m", "module", "--filter_object_id", "2", "--N", "2"
    ]


def test_light_curve_result_is_annotated_but_sample_file_is_not(tmp_path, monkeypatch):
    from qvc.light_curve import multiband_fit_utils as fit_utils

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fit_utils, "prefix", "prov_run")
    monkeypatch.setattr(fit_utils, "suffix", "job0")
    record = {"schema": provenance.PROVENANCE_SCHEMA, "entrypoint": "test", "events": []}

    result_path = fit_utils.save_quasar_list_hdf5(
        [{"object_id": "42", "value": 1.0}], provenance=record
    )
    fit_utils.save_obj_samples_to_hdf5({"draw": [1.0]}, "42")

    with h5py.File(result_path, "r") as handle:
        assert handle.attrs["qvc_provenance_schema"] == provenance.PROVENANCE_SCHEMA
    sample_path = tmp_path / "results" / "samples" / "prov_run" / "42_job0.h5"
    with h5py.File(sample_path, "r") as handle:
        assert "qvc_provenance_schema" not in handle.attrs


def test_new_attributes_do_not_change_native_bundle_content(tmp_path):
    path = tmp_path / "native_samples.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["posterior_bundle_format"] = "native-v2"
        handle.create_dataset("samples/x", data=[1.0, 2.0])
    provenance.write_hdf5_provenance(
        path,
        {"schema": provenance.PROVENANCE_SCHEMA, "entrypoint": "spectra", "events": []},
    )
    with h5py.File(path, "r") as handle:
        assert handle.attrs["posterior_bundle_format"] == "native-v2"
        assert handle["samples/x"][:].tolist() == [1.0, 2.0]


def test_merge_source_summary_deduplicates_object_shards(tmp_path):
    from qvc.light_curve.merge_results import summarize_source_provenance

    paths = []
    for index in range(25):
        path = tmp_path / f"{index}.h5"
        record = {
            "schema": provenance.PROVENANCE_SCHEMA,
            "entrypoint": "qvc.light_curve.fit_light_curves",
            "submission": {"command": "sfitlc --fit stone"},
            "module": {"parsed_args": {"filter_object_id": [str(index)], "nsamp": 250}},
            "qvc_git": {"commit": "abc"},
            "events": [{"type": "fit", "recorded_at": str(index)}],
        }
        with h5py.File(path, "w") as handle:
            provenance.write_hdf5_provenance(handle, record)
        paths.append(path)

    summary = summarize_source_provenance(paths)
    assert summary["source_count"] == 25
    assert summary["unique_run_count"] == 1
    assert summary["runs"][0]["source_count"] == 25
    assert len(provenance.canonical_json(summary)) < 2_000
