import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from qvc.hubble.qsogen_color_parent import (
    DEFAULT_FHOST_GRID,
    EXPECTED_ASSET_SHA256,
    build_qsogen_color_parent_cache,
    main,
    qsogen_mean_colors,
    solve_effective_mi,
    validate_vendored_assets,
)
from qvc.hubble.fitted_color_completeness import read_qsogen_color_parent_cache


def test_vendored_qsogen_and_jaxsedfit_assets_are_exactly_pinned():
    assert validate_vendored_assets() == EXPECTED_ASSET_SHA256


def test_effective_mi_solution_and_host_rescaling_are_finite_and_redder():
    effective_mi, sed = solve_effective_mi(-25.0, 1.5)
    assert effective_mi == pytest.approx(-24.97204744, abs=2e-6)
    assert sed.M_i == pytest.approx(effective_mi)
    agn, total, second_mi = qsogen_mean_colors(
        -25.0, 1.5, [0.0, 0.5, 1.0]
    )
    assert second_mi == pytest.approx(effective_mi)
    assert total[0] == agn
    assert np.all(np.diff(total) > 0.0)


def test_default_dense_fhost_grid_converges_against_direct_qsogen_mixing():
    # The low-z/faint corner has the strongest curvature as the first small
    # red-host contribution is introduced.  Test every half-step against
    # direct qsogen component mixing, rather than another interpolated grid.
    assert DEFAULT_FHOST_GRID == (0.0, 1.0, 0.002)
    redshift = 0.44
    apparent_m2500 = 24.0
    cosmology = FlatLambdaCDM(H0=70.0, Om0=0.3)
    absolute_m2500 = apparent_m2500 - cosmology.distmod(redshift).value
    dense_grid = np.linspace(0.0, 1.0, 501)
    half_steps = 0.5 * (dense_grid[:-1] + dense_grid[1:])
    _, dense_colors, _ = qsogen_mean_colors(
        absolute_m2500, redshift, dense_grid
    )
    _, direct_colors, _ = qsogen_mean_colors(
        absolute_m2500, redshift, half_steps
    )
    dense_error = np.max(
        np.abs(np.interp(half_steps, dense_grid, dense_colors) - direct_colors)
    )
    assert dense_error < 0.002

    coarse_grid = np.linspace(0.0, 1.0, 21)
    _, coarse_colors, _ = qsogen_mean_colors(
        absolute_m2500, redshift, coarse_grid
    )
    coarse_error = np.max(
        np.abs(np.interp(half_steps, coarse_grid, coarse_colors) - direct_colors)
    )
    assert coarse_error > 0.20


def test_small_qsogen_cache_is_deterministic_and_records_semantics():
    kwargs = {
        "magnitude_grid": [20.0, 21.0],
        "redshift_grid": [1.0, 1.2],
        "f_host_grid": [0.0, 0.5, 1.0],
        "progress": False,
    }
    first = build_qsogen_color_parent_cache(**kwargs)
    second = build_qsogen_color_parent_cache(**kwargs)
    assert first.content_hash_sha256 == second.content_hash_sha256
    np.testing.assert_array_equal(
        first.total_mean_g_minus_i, second.total_mean_g_minus_i
    )
    assert first.total_mean_g_minus_i.shape == (2, 2, 3)
    assert first.provenance["qsogen_default_redshift_luminosity_relation_used"] is False
    assert first.provenance["magnitude_state"] == "attenuation_retaining"
    assert first.provenance["filter_names"] == ["g_sdss", "i_sdss"]
    assert "selected SDSS DR16Q" in first.provenance[
        "parent_population_interpretation"
    ]
    assert "asymmetric internal-dust red tail" in first.provenance[
        "residual_scatter_limitation"
    ]


def test_cli_generates_an_atomic_loadable_cache_and_refuses_overwrite(
    tmp_path, capsys
):
    output = tmp_path / "parent.h5"
    arguments = [
        "--output",
        str(output),
        "--m-min",
        "20",
        "--m-max",
        "21",
        "--m-step",
        "1",
        "--z-min",
        "1",
        "--z-max",
        "1.2",
        "--z-step",
        "0.2",
        "--fhost-min",
        "0",
        "--fhost-max",
        "1",
        "--fhost-step",
        "0.5",
        "--no-progress",
    ]
    assert main(arguments) == 0
    cache = read_qsogen_color_parent_cache(output)
    assert cache.total_mean_g_minus_i.shape == (2, 2, 3)
    assert cache.content_hash_sha256 in capsys.readouterr().out
    assert not list(tmp_path.glob(".*.tmp"))
    with pytest.raises(FileExistsError, match="--force"):
        main(arguments)
