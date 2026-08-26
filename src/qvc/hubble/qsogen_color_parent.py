"""Build deterministic qsogen parent-colour caches for QVC.

This is a deliberately small adapter around the pinned MIT-licensed qsogen
core.  It does not import qsogen's historical ``model_colours.py`` (which uses
the removed ``scipy.integrate.simps`` API).  Synthetic photometry instead uses
the exact photon-response SDSS g/i curves and projection convention from the
JAXSedFit revision used for the fitted spectra.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import resources
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from qvc.hubble._vendor.qsogen.qsosed import Quasar_sed
from qvc.hubble.fitted_color_completeness import (
    JAXSEDFIT_FILTER_COMMIT,
    QSOGEN_COMMIT,
    QSOGEN_LICENSE,
    QSOGEN_SOURCE_URL,
    REFERENCE_COSMOLOGY,
    QsogenColorParentCache,
    write_qsogen_color_parent_cache,
)


QSOGEN_RESOURCE_PACKAGE = "qvc.hubble._vendor.qsogen"
FILTER_RESOURCE_PACKAGE = "qvc.hubble._vendor.jaxsedfit_filters"
JAXSEDFIT_SOURCE_URL = "https://github.com/dutra/jaxsedfit"

EXPECTED_ASSET_SHA256 = {
    "qsosed.py": "f00041c21c5b286333d745bde152fd82f1a9de6124f36904800bcab7af241075",
    "qsosed_emlines_20210625.dat": "811f0c8ee2e57752fcb0fd4685fc5f57a6220f262a1abd7cf2fec0655cbb21a0",
    "S0_template_norm.sed": "3388dc5d26d7621a0547f476491871296734cfe8a3de8e8b45c40512f9795456",
    "pl_ext_comp_03.sph": "d7057949f3a8bc4da80513f67791da752a39daa4dbe1a1459e8f39cd339921bc",
    "qsogen_LICENSE.md": "fc3c7689c4bb0f50e100f4535663877aa2703cc95977e5f4f7050832d2881576",
    "JAXSedFit_SLOAN_SDSS.g.dat": "ddfb8a48475af034fd33d876fafa055935a0d493d3fea4a22d727c295cf3e313",
    "JAXSedFit_SLOAN_SDSS.i.dat": "79429199b1db32de7f41fcdadfab35aa474eb0b4be39e8e4f8cba282d9666b12",
}

AB_ABSOLUTE_MAG_ZEROPOINT = 51.59477721004232
C_ANGSTROM_PER_SECOND = 2.99792458e18
REST_2500_ANGSTROM = 2500.0
AB_FNU_ZERO_CGS = 3.631e-20

DEFAULT_MAGNITUDE_GRID = (18.5, 24.0, 0.25)
DEFAULT_REDSHIFT_GRID = (0.44, 3.16, 0.08)
# Host colour changes are highly nonlinear as the first small red host
# component is introduced.  The former 0.05 step produced interpolation
# errors up to 0.258 mag; 0.002 is frozen by a direct qsogen convergence test.
DEFAULT_FHOST_GRID = (0.0, 1.0, 0.002)


def _resource_bytes(package: str, name: str) -> bytes:
    return resources.files(package).joinpath(name).read_bytes()


def _resource_array(package: str, name: str) -> np.ndarray:
    resource = resources.files(package).joinpath(name)
    with resources.as_file(resource) as path:
        return np.genfromtxt(path, unpack=True)


def vendored_asset_hashes() -> dict[str, str]:
    paths = {
        "qsosed.py": (QSOGEN_RESOURCE_PACKAGE, "qsosed.py"),
        "qsosed_emlines_20210625.dat": (
            QSOGEN_RESOURCE_PACKAGE,
            "qsosed_emlines_20210625.dat",
        ),
        "S0_template_norm.sed": (QSOGEN_RESOURCE_PACKAGE, "S0_template_norm.sed"),
        "pl_ext_comp_03.sph": (QSOGEN_RESOURCE_PACKAGE, "pl_ext_comp_03.sph"),
        "qsogen_LICENSE.md": (QSOGEN_RESOURCE_PACKAGE, "LICENSE.md"),
        "JAXSedFit_SLOAN_SDSS.g.dat": (
            FILTER_RESOURCE_PACKAGE,
            "SLOAN_SDSS.g.dat",
        ),
        "JAXSedFit_SLOAN_SDSS.i.dat": (
            FILTER_RESOURCE_PACKAGE,
            "SLOAN_SDSS.i.dat",
        ),
    }
    return {
        label: hashlib.sha256(_resource_bytes(package, name)).hexdigest()
        for label, (package, name) in paths.items()
    }


def validate_vendored_assets() -> dict[str, str]:
    actual = vendored_asset_hashes()
    mismatches = {
        name: {"expected": EXPECTED_ASSET_SHA256[name], "actual": actual.get(name)}
        for name in EXPECTED_ASSET_SHA256
        if actual.get(name) != EXPECTED_ASSET_SHA256[name]
    }
    if mismatches:
        raise RuntimeError(f"Pinned qsogen/JAXSedFit asset hash mismatch: {mismatches}")
    return actual


def _qsogen_parameters() -> dict[str, Any]:
    return {
        "plslp1": -0.349,
        "plslp2": 0.593,
        "plstep": -1.0,
        "plbrk1": 3880.0,
        "tbb": 1243.6,
        "plbrk3": 1200.0,
        "bbnorm": 3.961,
        "scal_emline": -0.9936,
        "emline_type": None,
        "scal_halpha": 1.0,
        "scal_lya": 1.0,
        "scal_nlr": 1.0,
        "emline_template": _resource_array(
            QSOGEN_RESOURCE_PACKAGE, "qsosed_emlines_20210625.dat"
        ),
        "galaxy_template": _resource_array(
            QSOGEN_RESOURCE_PACKAGE, "S0_template_norm.sed"
        ),
        "reddening_curve": _resource_array(
            QSOGEN_RESOURCE_PACKAGE, "pl_ext_comp_03.sph"
        ),
        # M_i is always supplied explicitly.  This retained array is required
        # by the upstream object constructor but never interpolated.
        "zlum_lumval": np.array(
            [
                [0.23, 0.34, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.3, 3.7, 4.13, 4.5],
                [-21.76, -22.9, -24.1, -25.4, -26.0, -26.6, -27.1, -27.6, -27.9, -28.1, -28.4, -28.6, -28.9],
            ],
            dtype=float,
        ),
        "M_i": -27.0,
        "beslope": 0.183,
        "benorm": -27.0,
        "bcnorm": False,
        "lyForest": True,
        "lylim": 912.0,
        "gflag": True,
        "fragal": 0.244,
        "gplind": 0.684,
    }


def _load_jaxsedfit_filter(name: str) -> tuple[np.ndarray, np.ndarray, float]:
    raw = _resource_array(FILTER_RESOURCE_PACKAGE, f"SLOAN_SDSS.{name}.dat")
    if raw.shape[0] < 2:
        raise RuntimeError(f"Malformed vendored SDSS {name} filter.")
    wave = np.asarray(raw[0], dtype=float)
    # The vendored files are photon response curves.  JAXSedFit converts these
    # to energy weighting by multiplying transmission by wavelength.
    transmission = np.clip(np.asarray(raw[1], dtype=float), 0.0, None) * wave
    order = np.argsort(wave, kind="stable")
    wave, transmission = wave[order], transmission[order]
    unique = np.concatenate(([True], np.diff(wave) > 0.0))
    wave, transmission = wave[unique], transmission[unique]
    transmission = transmission.copy()
    transmission[0] = 0.0
    transmission[-1] = 0.0
    numerator = float(np.trapezoid(transmission, wave))
    denominator = float(np.trapezoid(transmission / wave**2, wave))
    if numerator <= 0.0 or denominator <= 0.0:
        raise RuntimeError(f"Vendored SDSS {name} filter has zero support.")
    pivot = math.sqrt(numerator / denominator)
    return wave, transmission, pivot


def _project_l_lambda_to_fnu_proxy(
    rest_wave: np.ndarray,
    rest_l_lambda: np.ndarray,
    *,
    redshift: float,
    filter_curve: tuple[np.ndarray, np.ndarray, float],
) -> float:
    observed_wave, transmission, pivot = filter_curve
    sampled = np.interp(observed_wave / (1.0 + redshift), rest_wave, rest_l_lambda)
    mean_l_lambda = float(
        np.trapezoid(sampled * transmission, observed_wave)
        / np.trapezoid(transmission, observed_wave)
    )
    # All omitted distance/redshift/unit factors are common between g and i.
    return pivot**2 * mean_l_lambda


def _project_rest_shifted_i_lnu(
    rest_wave: np.ndarray,
    rest_l_lambda: np.ndarray,
    i_filter: tuple[np.ndarray, np.ndarray, float],
) -> float:
    observed_wave, transmission, pivot = i_filter
    shifted_wave = observed_wave / 3.0
    sampled = np.interp(shifted_wave, rest_wave, rest_l_lambda)
    mean_l_lambda = float(
        np.trapezoid(sampled * transmission, observed_wave)
        / np.trapezoid(transmission, observed_wave)
    )
    pivot_rest = pivot / 3.0
    return mean_l_lambda * pivot_rest**2 / C_ANGSTROM_PER_SECOND


def _target_l_lambda_2500(absolute_m2500: float) -> float:
    log_l_nu = (AB_ABSOLUTE_MAG_ZEROPOINT - float(absolute_m2500)) / 2.5
    return 10.0**log_l_nu * C_ANGSTROM_PER_SECOND / REST_2500_ANGSTROM**2


def _base_wavelength_grid() -> np.ndarray:
    # Slightly extend the upstream default below 890 A so the z=3.16 SDSS-g
    # blue edge is explicitly zeroed by qsogen's Lyman-limit treatment.
    return np.logspace(np.log10(850.0), np.log10(30_200.0), 20_001)


def _make_sed(
    *,
    redshift: float,
    effective_mi: float,
    parameters: dict[str, Any],
    include_host: bool,
) -> Quasar_sed:
    values = parameters.copy()
    values["M_i"] = float(effective_mi)
    values["gflag"] = bool(include_host)
    return Quasar_sed(
        z=float(redshift),
        LogL3000=None,
        wavlen=_base_wavelength_grid(),
        ebv=0.0,
        params=values,
    )


def solve_effective_mi(
    absolute_m2500: float,
    redshift: float,
    *,
    parameters: dict[str, Any] | None = None,
    tolerance: float = 2.0e-7,
    max_iterations: int = 48,
) -> tuple[float, Quasar_sed]:
    """Normalize the AGN at 2500 A and solve its synthetic ``M_i(z=2)``.

    The fixed point is needed because qsogen's emission-line mixture depends on
    ``M_i``.  This explicitly replaces its default DR16Q redshift-luminosity
    track, which would otherwise make two objects with the same luminosity use
    different line/host scalings merely because their redshifts differ.
    """

    if not np.isfinite(absolute_m2500):
        raise ValueError("absolute_m2500 must be finite.")
    if not np.isfinite(redshift) or redshift <= 0.0:
        raise ValueError("redshift must be finite and positive.")
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("Invalid fixed-point controls.")
    params = _qsogen_parameters() if parameters is None else parameters
    target_l_lambda = _target_l_lambda_2500(float(absolute_m2500))
    i_filter = _load_jaxsedfit_filter("i")
    effective_mi = float(absolute_m2500 - 0.15)
    last_sed: Quasar_sed | None = None
    for _ in range(max_iterations):
        sed = _make_sed(
            redshift=redshift,
            effective_mi=effective_mi,
            parameters=params,
            include_host=False,
        )
        value_2500 = float(np.interp(REST_2500_ANGSTROM, sed.wavlen, sed.flux))
        if not np.isfinite(value_2500) or value_2500 <= 0.0:
            raise RuntimeError("qsogen AGN has nonpositive rest-frame 2500-A flux.")
        normalized = sed.flux * (target_l_lambda / value_2500)
        effective_lnu_i = _project_rest_shifted_i_lnu(
            sed.wavlen, normalized, i_filter
        )
        if not np.isfinite(effective_lnu_i) or effective_lnu_i <= 0.0:
            raise RuntimeError("qsogen produced a nonpositive effective i-band luminosity.")
        updated = AB_ABSOLUTE_MAG_ZEROPOINT - 2.5 * np.log10(effective_lnu_i)
        last_sed = sed
        if abs(updated - effective_mi) <= tolerance:
            effective_mi = float(updated)
            break
        # Damping avoids oscillation when a strong line crosses a filter edge.
        effective_mi = float(0.5 * (effective_mi + updated))
    else:
        raise RuntimeError(
            "qsogen M2500-to-effective-M_i fixed point did not converge."
        )
    # Re-evaluate at the converged value so returned SED and M_i are aligned.
    last_sed = _make_sed(
        redshift=redshift,
        effective_mi=effective_mi,
        parameters=params,
        include_host=True,
    )
    return effective_mi, last_sed


def qsogen_mean_colors(
    absolute_m2500: float,
    redshift: float,
    f_host_grid: Any,
    *,
    parameters: dict[str, Any] | None = None,
) -> tuple[float, np.ndarray, float]:
    """Return AGN-only and exact-host-fraction qsogen ``g-i`` colours."""

    hosts = np.asarray(f_host_grid, dtype=float)
    if hosts.ndim != 1 or not np.all(np.isfinite(hosts)) or np.any(
        (hosts < 0.0) | (hosts > 1.0)
    ):
        raise ValueError("f_host_grid must be one-dimensional within [0, 1].")
    params = _qsogen_parameters() if parameters is None else parameters
    effective_mi, sed = solve_effective_mi(
        float(absolute_m2500), float(redshift), parameters=params
    )
    host = np.asarray(sed.host_galaxy_flux, dtype=float)
    agn = np.asarray(sed.flux, dtype=float) - host
    agn_2500 = float(np.interp(REST_2500_ANGSTROM, sed.wavlen, agn))
    host_2500 = float(np.interp(REST_2500_ANGSTROM, sed.wavlen, host))
    if agn_2500 <= 0.0 or host_2500 <= 0.0:
        raise RuntimeError("qsogen AGN/host cannot be normalized at rest 2500 A.")
    agn_unit = agn / agn_2500
    host_unit = host / host_2500
    filters = {
        "g": _load_jaxsedfit_filter("g"),
        "i": _load_jaxsedfit_filter("i"),
    }

    # Projection is linear.  Project each component once, then mix its band
    # fluxes algebraically for every exact host fraction.  The 25x denser grid
    # therefore adds negligible generation time.
    agn_g = _project_l_lambda_to_fnu_proxy(
        sed.wavlen, agn_unit, redshift=redshift, filter_curve=filters["g"]
    )
    agn_i = _project_l_lambda_to_fnu_proxy(
        sed.wavlen, agn_unit, redshift=redshift, filter_curve=filters["i"]
    )
    host_g = _project_l_lambda_to_fnu_proxy(
        sed.wavlen, host_unit, redshift=redshift, filter_curve=filters["g"]
    )
    host_i = _project_l_lambda_to_fnu_proxy(
        sed.wavlen, host_unit, redshift=redshift, filter_curve=filters["i"]
    )
    component_fluxes = np.asarray([agn_g, agn_i, host_g, host_i], dtype=float)
    if np.any(~np.isfinite(component_fluxes)) or np.any(component_fluxes <= 0.0):
        raise RuntimeError("qsogen produced nonpositive SDSS g/i component flux.")
    total_g = (1.0 - hosts) * agn_g + hosts * host_g
    total_i = (1.0 - hosts) * agn_i + hosts * host_i
    agn_color = float(-2.5 * np.log10(agn_g / agn_i))
    total_colors = -2.5 * np.log10(total_g / total_i)
    total_colors[hosts == 0.0] = agn_color
    return agn_color, total_colors, effective_mi


def _grid_from_step(lower: float, upper: float, step: float, *, name: str) -> np.ndarray:
    if not all(np.isfinite(value) for value in (lower, upper, step)):
        raise ValueError(f"{name} grid controls must be finite.")
    if upper <= lower or step <= 0.0:
        raise ValueError(f"{name} grid requires upper>lower and step>0.")
    interval_count = int(round((upper - lower) / step))
    if interval_count < 1 or not np.isclose(
        lower + interval_count * step, upper, rtol=0.0, atol=1.0e-10
    ):
        raise ValueError(f"{name} step must divide its closed support exactly.")
    return np.linspace(lower, upper, interval_count + 1)


def build_qsogen_color_parent_cache(
    magnitude_grid: Any,
    redshift_grid: Any,
    f_host_grid: Any,
    *,
    progress: bool = True,
) -> QsogenColorParentCache:
    """Generate a deterministic qsogen parent cache on supplied regular axes."""

    asset_hashes = validate_vendored_assets()
    magnitude = np.asarray(magnitude_grid, dtype=float)
    redshift = np.asarray(redshift_grid, dtype=float)
    hosts = np.asarray(f_host_grid, dtype=float)
    # Let the cache constructor provide detailed axis errors before expensive work.
    if magnitude.ndim != 1 or redshift.ndim != 1 or hosts.ndim != 1:
        raise ValueError("All qsogen parent grids must be one-dimensional.")
    if not np.all(np.isfinite(redshift)) or np.any(redshift <= 0.0):
        raise ValueError("redshift_grid must be finite and positive.")
    if hosts.size < 2 or hosts[0] != 0.0 or hosts[-1] != 1.0:
        raise ValueError("f_host_grid must span [0,1].")
    cosmology = FlatLambdaCDM(
        H0=REFERENCE_COSMOLOGY["H0_km_s_Mpc"],
        Om0=REFERENCE_COSMOLOGY["Omega_m"],
    )
    agn_color = np.empty((magnitude.size, redshift.size), dtype=float)
    total_color = np.empty((magnitude.size, redshift.size, hosts.size), dtype=float)
    effective_mi = np.empty_like(agn_color)
    contexts = [(i, j) for i in range(magnitude.size) for j in range(redshift.size)]
    iterator: Any = contexts
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(contexts, desc="qsogen g-i parent", unit="cell")
    params = _qsogen_parameters()
    for magnitude_index, redshift_index in iterator:
        z = float(redshift[redshift_index])
        absolute_m2500 = float(
            magnitude[magnitude_index] - cosmology.distmod(z).value
        )
        agn_value, total_values, mi_value = qsogen_mean_colors(
            absolute_m2500,
            z,
            hosts,
            parameters=params,
        )
        agn_color[magnitude_index, redshift_index] = agn_value
        total_color[magnitude_index, redshift_index] = total_values
        effective_mi[magnitude_index, redshift_index] = mi_value
    provenance = {
        "construction": "qvc_qsogen_fitted_psf_delta_gi_parent_v2_dense_fhost",
        "qsogen_commit": QSOGEN_COMMIT,
        "qsogen_source_url": QSOGEN_SOURCE_URL,
        "qsogen_license": QSOGEN_LICENSE,
        "jaxsedfit_filter_commit": JAXSEDFIT_FILTER_COMMIT,
        "jaxsedfit_source_url": JAXSEDFIT_SOURCE_URL,
        "asset_sha256": asset_hashes,
        "filter_names": ["g_sdss", "i_sdss"],
        "filter_projection": (
            "JAXSedFit photon response multiplied by wavelength; trapezoidal "
            "energy-weighted mean f_lambda converted at the pivot wavelength"
        ),
        "reference_cosmology": dict(REFERENCE_COSMOLOGY),
        "magnitude_state": "attenuation_retaining",
        "source_internal_attenuation": (
            "qsogen mean uses E(B-V)=0; fitted selected-object colors retain "
            "JAXSedFit source attenuation"
        ),
        "parent_population_interpretation": (
            "qsogen is calibrated to median colors of selected SDSS DR16Q "
            "quasars; it is a sensitivity-analysis reference, not an unbiased "
            "pre-targeting color parent"
        ),
        "residual_scatter_limitation": (
            "the downstream symmetric Gaussian residual with default sigma=0.20 "
            "mag does not model an asymmetric internal-dust red tail"
        ),
        "host_scaling": (
            "qsogen S0 host shape rescaled algebraically so f_host_2500_psf "
            "equals each grid coordinate; AGN normalization remains the supplied M2500"
        ),
        "m2500_to_qsogen_luminosity": (
            "M2500=m2500-DM(H0=70,Om0=0.3); normalize AGN-only rest Llambda(2500) "
            "to exact monochromatic AB M2500; solve fixed point for synthetic "
            "SDSS-i band shifted to z=2 and pass that effective M_i explicitly"
        ),
        "qsogen_default_redshift_luminosity_relation_used": False,
        "qsogen_ebv": 0.0,
        "igm_model": "qsogen Becker2013 mean forest and 912A Lyman limit",
        "effective_mi_range": [float(np.min(effective_mi)), float(np.max(effective_mi))],
    }
    return QsogenColorParentCache(
        magnitude_grid=magnitude,
        redshift_grid=redshift,
        f_host_grid=hosts,
        agn_only_mean_g_minus_i=agn_color,
        total_mean_g_minus_i=total_color,
        provenance=provenance,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the pinned qsogen g-i parent cache used by QVC."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--m-min", type=float, default=DEFAULT_MAGNITUDE_GRID[0])
    parser.add_argument("--m-max", type=float, default=DEFAULT_MAGNITUDE_GRID[1])
    parser.add_argument("--m-step", type=float, default=DEFAULT_MAGNITUDE_GRID[2])
    parser.add_argument("--z-min", type=float, default=DEFAULT_REDSHIFT_GRID[0])
    parser.add_argument("--z-max", type=float, default=DEFAULT_REDSHIFT_GRID[1])
    parser.add_argument("--z-step", type=float, default=DEFAULT_REDSHIFT_GRID[2])
    parser.add_argument("--fhost-min", type=float, default=DEFAULT_FHOST_GRID[0])
    parser.add_argument("--fhost-max", type=float, default=DEFAULT_FHOST_GRID[1])
    parser.add_argument("--fhost-step", type=float, default=DEFAULT_FHOST_GRID[2])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"Refusing to replace existing parent cache {args.output}; pass --force."
        )
    magnitude = _grid_from_step(args.m_min, args.m_max, args.m_step, name="magnitude")
    redshift = _grid_from_step(args.z_min, args.z_max, args.z_step, name="redshift")
    hosts = _grid_from_step(
        args.fhost_min, args.fhost_max, args.fhost_step, name="f_host"
    )
    cache = build_qsogen_color_parent_cache(
        magnitude, redshift, hosts, progress=not args.no_progress
    )
    write_qsogen_color_parent_cache(args.output, cache)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "content_hash_sha256": cache.content_hash_sha256,
                "shape": list(cache.total_mean_g_minus_i.shape),
                "support": cache.support,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_FHOST_GRID",
    "DEFAULT_MAGNITUDE_GRID",
    "DEFAULT_REDSHIFT_GRID",
    "EXPECTED_ASSET_SHA256",
    "build_qsogen_color_parent_cache",
    "main",
    "qsogen_mean_colors",
    "solve_effective_mi",
    "validate_vendored_assets",
    "vendored_asset_hashes",
]
