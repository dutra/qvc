import h5py
import numpy as np
import pandas as pd
import pytest

from qvc.light_curve.compare_fit_catalogs import compare_light_curve_fit_catalogs


def _write_catalog(
    path,
    object_ids,
    *,
    offset=0.0,
    include_sampler_diagnostics=False,
):
    object_ids = np.asarray(object_ids, dtype=h5py.string_dtype("utf-8"))
    n_objects = len(object_ids)
    base = np.linspace(-1.0, 1.0, n_objects)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("object_id", data=object_ids)
        handle.create_dataset(
            "model_variant",
            data=np.asarray(
                ["new_model" if include_sampler_diagnostics else "old_model"]
                * n_objects,
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
        for parameter, scale in (
            ("log_sigma_uv", 1.0),
            ("log_tau_uv_rf", 2.0),
        ):
            handle.create_dataset(parameter, data=scale * base + offset)
            handle.create_dataset(
                f"{parameter}_err",
                data=np.full(n_objects, 0.1),
            )
        if include_sampler_diagnostics:
            handle.create_dataset(
                "accept_prob",
                data=np.linspace(0.65, 0.95, n_objects),
            )
            divergences = np.zeros(n_objects)
            divergences[::7] = 1
            handle.create_dataset("num_divergences", data=divergences)


def test_compare_light_curve_fit_catalogs_writes_honest_outputs(tmp_path):
    old_path = tmp_path / "old.h5"
    new_path = tmp_path / "new.h5"
    object_ids = [f"obj_{index:03d}" for index in range(30)]
    _write_catalog(old_path, object_ids, offset=0.0)
    _write_catalog(
        new_path,
        object_ids,
        offset=0.05,
        include_sampler_diagnostics=True,
    )

    outputs = compare_light_curve_fit_catalogs(
        new_path,
        old_path,
        output_dir=tmp_path / "plots",
        new_label="new fit",
        old_label="old fit",
        atlas_parameters=("log_sigma_uv", "log_tau_uv_rf"),
        min_matched=10,
    )

    assert outputs["matched_objects"] == 30
    assert outputs["eligible_parameters"] == 2
    for key in (
        "overview_pdf",
        "atlas_pdf",
        "sampler_summary_csv",
        "parameter_stability_csv",
    ):
        assert (tmp_path / "plots" / outputs[key].split("/")[-1]).exists()

    sampler = pd.read_csv(outputs["sampler_summary_csv"]).set_index("catalog")
    assert bool(sampler.loc["new fit", "acceptance_available"])
    assert bool(sampler.loc["new fit", "divergences_available"])
    assert not bool(sampler.loc["old fit", "acceptance_available"])
    assert not bool(sampler.loc["old fit", "divergences_available"])
    assert not bool(sampler.loc["new fit", "rhat_available"])
    assert not bool(sampler.loc["new fit", "ess_available"])
    assert np.isnan(sampler.loc["old fit", "objects_with_divergences"])
    assert np.isnan(sampler.loc["old fit", "total_divergences"])

    stability = pd.read_csv(outputs["parameter_stability_csv"]).set_index(
        "parameter"
    )
    assert stability.loc["log_sigma_uv", "matched_finite"] == 30
    assert stability.loc["log_sigma_uv", "spearman_rho"] == pytest.approx(1.0)
    assert stability.loc["log_sigma_uv", "median_new_minus_old"] == pytest.approx(
        0.05
    )


def test_compare_light_curve_fit_catalogs_rejects_duplicate_ids(tmp_path):
    old_path = tmp_path / "old.h5"
    new_path = tmp_path / "new.h5"
    _write_catalog(old_path, ["a", "b", "c"])
    _write_catalog(
        new_path,
        ["a", "a", "c"],
        include_sampler_diagnostics=True,
    )

    with pytest.raises(ValueError, match="duplicate object_id"):
        compare_light_curve_fit_catalogs(
            new_path,
            old_path,
            output_dir=tmp_path / "plots",
            min_matched=1,
        )
