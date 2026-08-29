from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import h5py
import numpy as np
import pytest

import run_plot_spectra_samples as plotting


def test_cli_requires_valid_posterior_draws(tmp_path):
    with pytest.raises(SystemExit):
        plotting.parse_args(["--source-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        plotting.parse_args(
            ["--source-dir", str(tmp_path), "--posterior-draws", "0"]
        )
    with pytest.raises(SystemExit):
        plotting.parse_args(
            ["--source-dir", str(tmp_path), "--posterior-draws", "nope"]
        )


def test_cli_uses_memory_bounded_worker_defaults(tmp_path):
    args = plotting.parse_args(
        ["--source-dir", str(tmp_path), "--posterior-draws", "median"]
    )
    assert args.workers == 4
    assert args.max_tasks_per_worker == 1


@pytest.mark.parametrize(
    ("selection", "expected_count"),
    [("median", 1), (3, 3), (99, 5), ("all", 5)],
)
def test_select_posterior_samples_preserves_aligned_dimensions(
    selection, expected_count
):
    samples = {
        "scalar": np.arange(5.0),
        "vector": np.arange(10.0).reshape(5, 2),
    }
    selected = plotting.select_posterior_samples(samples, selection)
    assert selected["scalar"].shape == (expected_count,)
    assert selected["vector"].shape == (expected_count, 2)
    if selection == 3:
        np.testing.assert_array_equal(selected["scalar"], [0.0, 2.0, 4.0])
    if selection == "median":
        np.testing.assert_array_equal(selected["scalar"], [2.0])


def test_plot_paths_match_joint_conventions(tmp_path):
    bundle = tmp_path / "z1.250_012345.67+012345.6_joint_samples.h5"
    sed, spectrum = plotting.plot_paths(bundle, tmp_path / "plots")
    assert sed.name == "z1.250_012345.67+012345.6_joint.png"
    assert spectrum.name == "z1.250_012345.67+012345.6_spectrum.png"


class DummyFitter:
    def __init__(self, *, fail_spectrum=False):
        self.samples = {"x": np.arange(8.0)}
        self.predictive = None
        self.predict_computations = 0
        self.prediction_kinds = []
        self.fail_spectrum = fail_spectrum

    def predict(self, kind="plot"):
        if self.predictive is None:
            self.predict_computations += 1
            self.prediction_kinds.append(kind)
            self.predictive = {"model": np.ones((1, 2))}
        return self.predictive

    def plot_sed(self, *, output_path, show):
        assert show is False
        self.predict()
        figure = plt.figure()
        figure.savefig(output_path, format="png")
        return figure

    def plot_spectrum(self, *, show_plot, plot_residual):
        assert show_plot is False
        assert plot_residual is False
        self.predict()
        if self.fail_spectrum:
            raise RuntimeError("broken spectrum")
        return plt.figure()


def _task(tmp_path, *, make_sed=True, make_spectrum=True):
    return plotting.PlotTask(
        bundle_path=tmp_path / "z1.000_name_joint_samples.h5",
        sed_path=tmp_path / "plots" / "z1.000_name_joint.png",
        spectrum_path=tmp_path / "plots" / "z1.000_name_spectrum.png",
        make_sed=make_sed,
        make_spectrum=make_spectrum,
    )


def test_render_reuses_one_prediction_and_writes_only_pngs(monkeypatch, tmp_path):
    fitter = DummyFitter()
    monkeypatch.setattr(plotting, "_load_fitter", lambda path: fitter)

    task = _task(tmp_path)
    result = plotting.render_task(task, "median")

    assert result.error == ""
    assert result.generated == 2
    assert fitter.predict_computations == 1
    assert fitter.prediction_kinds == ["plot"]
    assert task.sed_path.stat().st_size > 0
    assert task.spectrum_path.stat().st_size > 0
    assert sorted(path.suffix for path in task.sed_path.parent.iterdir()) == [
        ".png",
        ".png",
    ]


def test_spectrum_only_uses_lightweight_prediction(monkeypatch, tmp_path):
    fitter = DummyFitter()
    monkeypatch.setattr(plotting, "_load_fitter", lambda path: fitter)
    task = _task(tmp_path, make_sed=False, make_spectrum=True)

    result = plotting.render_task(task, "median")

    assert result.error == ""
    assert fitter.predict_computations == 1
    assert fitter.prediction_kinds == ["photometry"]
    assert task.spectrum_path.is_file()


def test_discovery_overwrites_by_default_and_can_skip_nonempty_files(tmp_path):
    source = tmp_path / "all"
    source.mkdir()
    bundle = source / "z1.000_name_joint_samples.h5"
    bundle.touch()
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir()
    sed_path, spectrum_path = plotting.plot_paths(bundle, plot_dir)
    sed_path.write_bytes(b"png")
    spectrum_path.touch()

    common = dict(
        source_dir=source,
        plot_dir=plot_dir,
        start=0,
        limit=None,
    )
    overwrite = plotting.discover_tasks(
        SimpleNamespace(skip_existing=False, **common)
    )[0]
    skipping = plotting.discover_tasks(
        SimpleNamespace(skip_existing=True, **common)
    )[0]

    assert overwrite.make_sed and overwrite.make_spectrum
    assert skipping.make_sed is False
    assert skipping.make_spectrum is True


def test_object_id_csv_selects_only_mapped_bundles_in_csv_order(tmp_path):
    run_dir = tmp_path / "run"
    source = run_dir / "all"
    source.mkdir(parents=True)
    first = source / "z1.000_first_joint_samples.h5"
    second = source / "z2.000_second_joint_samples.h5"
    unrequested = source / "z3.000_third_joint_samples.h5"
    for bundle in (first, second, unrequested):
        bundle.touch()

    chunk = run_dir / "run_chunk0001.h5"
    with h5py.File(chunk, "w") as handle:
        catalog = handle.create_group("catalog")
        catalog.create_dataset("object_id", data=[b"101", b"102", b"103"])
        catalog.create_dataset(
            "fit_result_path",
            data=[
                str(first).encode(),
                str(second).encode(),
                str(unrequested).encode(),
            ],
        )
    selection = tmp_path / "selection.csv"
    selection.write_text("object_id\n102\n101.0\n102\n")
    args = SimpleNamespace(
        source_dir=source,
        plot_dir=tmp_path / "plots",
        object_id_csv=selection,
        object_id_column="object_id",
        start=0,
        limit=None,
        skip_existing=False,
    )

    tasks = plotting.discover_tasks(args)

    assert [task.bundle_path for task in tasks] == [second, first]


def test_object_id_csv_rejects_missing_column_and_unmatched_id(tmp_path):
    wrong_column = tmp_path / "wrong.csv"
    wrong_column.write_text("source_id\n101\n")
    with pytest.raises(ValueError, match="Column 'object_id' is missing"):
        plotting.load_requested_object_ids(wrong_column, "object_id")

    run_dir = tmp_path / "run"
    source = run_dir / "all"
    source.mkdir(parents=True)
    with h5py.File(run_dir / "run_chunk0001.h5", "w") as handle:
        catalog = handle.create_group("catalog")
        catalog.create_dataset("object_id", data=[b"101"])
        catalog.create_dataset("fit_result_path", data=[b"missing_samples.h5"])
    with pytest.raises(ValueError, match="No saved sample bundle found"):
        plotting.resolve_object_id_bundles(source, ["101"], {})


def test_failure_removes_temporary_pngs(monkeypatch, tmp_path):
    monkeypatch.setattr(
        plotting,
        "_load_fitter",
        lambda path: DummyFitter(fail_spectrum=True),
    )
    task = _task(tmp_path, make_sed=False)

    result = plotting.render_task(task, "median")

    assert "broken spectrum" in result.error
    assert not task.spectrum_path.exists()
    assert not list(task.spectrum_path.parent.glob(".*.tmp.png"))


def test_dry_run_does_not_load_or_write(monkeypatch, tmp_path):
    source = tmp_path / "all"
    source.mkdir()
    (source / "z1.000_name_joint_samples.h5").touch()
    plot_dir = tmp_path / "plots"
    monkeypatch.setattr(
        plotting,
        "_load_fitter",
        lambda path: pytest.fail("dry-run must not load bundles"),
    )
    args = SimpleNamespace(
        source_dir=source,
        plot_dir=plot_dir,
        posterior_draws="median",
        workers=1,
        start=0,
        limit=None,
        skip_existing=False,
        dry_run=True,
    )

    assert plotting.run(args) == 0
    assert not plot_dir.exists()


def test_run_returns_nonzero_after_worker_failure(monkeypatch, tmp_path):
    source = tmp_path / "all"
    source.mkdir()
    (source / "z1.000_name_joint_samples.h5").touch()
    monkeypatch.setattr(
        plotting,
        "_load_fitter",
        lambda path: DummyFitter(fail_spectrum=True),
    )
    args = SimpleNamespace(
        source_dir=source,
        plot_dir=tmp_path / "plots",
        posterior_draws="median",
        workers=1,
        start=0,
        limit=None,
        skip_existing=False,
        dry_run=False,
    )

    assert plotting.run(args) == 1
