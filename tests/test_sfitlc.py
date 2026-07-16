import subprocess
from pathlib import Path
from types import SimpleNamespace

from hpc_scripts.sfitlc import JobConfig, build_sbatch_script, validate_chunking


def _args(**overrides):
    values = {
        "fit": "chisq",
        "N": 3,
        "skip": 0,
        "ncores": 1,
        "mem": "12G",
        "partition": "day",
        "time": "2:00:00",
        "env": "jaxcpu2",
        "svi_steps": 1000,
        "nwarm": 500,
        "nsamp": 250,
        "max_tree_depth": 12,
        "resume": None,
        "extra_fit_flags": (),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_sbatch_runs_each_chunk_object_in_a_fresh_process():
    script = build_sbatch_script(
        "probe_chisq",
        JobConfig(description="chisq", object_ids=["1", "2", "3"]),
        _args(),
        "data/input.csv",
        "results/data/spectra.csv",
    )

    assert 'export N="3"' in script
    assert 'START=$(( TASK_ID * N ))' in script
    assert 'read -r -a OBJECT_IDS <<< "$IDS"' in script
    assert 'for OBJECT_ID in "${OBJECT_IDS[@]}"; do' in script
    assert 'export SUFFIX="job${TASK_ID}_obj${OBJECT_INDEX}"' in script
    assert '--filter_object_id "$OBJECT_ID"' in script
    assert "--filter_object_id $IDS" not in script
    assert script.count("python -m qvc.light_curve.fit_light_curves") == 1


def test_generated_multi_object_sbatch_is_valid_bash():
    script = build_sbatch_script(
        "probe_stone",
        JobConfig(description="stone", object_ids=["1", "2"]),
        _args(fit="stone", N=2),
        None,
        "results/data/spectra.csv",
        object_ids_path=Path("/tmp/probe_object_ids.txt"),
    )

    result = subprocess.run(
        ["bash", "-n"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_chunk_count_still_uses_objects_per_slurm_task():
    assert validate_chunking(total_objects=7, n_per_job=3, skip=0, num_jobs=-1) == (3, 0, 2)
