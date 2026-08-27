"""Prepare the eBOSS photometry-only host-sensitivity training/mock catalog.

This command is intentionally an offline adapter.  It joins the local DR16Q
photometry to the spectra target-provenance catalog, decodes a small versioned
set of eBOSS QSO target bits, and evaluates paired qsogen host/no-host
photometry on the LF mock.  The output is consumed by
``program_color_completeness build``; none of these arrays enter cosmology.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits

from qvc.hubble.cuts import (
    COMPLETENESS_MAG_EDGE_MAX,
    COMPLETENESS_MAG_EDGE_MIN,
    COMPLETENESS_N_MAG_BINS,
    EBOSS_ALT_CHANNEL_BITS,
    EBOSS_COLOR_BITS,
    EBOSS_DISQUALIFY_BITS,
)
from qvc.hubble.hubble_completeness_refactored import (
    get_completeness_function_2d,
    prepare_completeness_magnitude_columns,
)
from qvc.hubble.hubble_utils import load_agn_data
from qvc.hubble.program_color_completeness import (
    EBOSS_FEATURE_NAMES,
    PREPARED_CATALOG_SCHEMA,
    apply_paired_flux_noise,
    assert_paired_nuclear_state,
    eboss_target_features,
    hash_completeness_2d,
    leave_one_channel_out_closure,
)


TARGET_BIT_DEFINITION_VERSION = "sdss-dr17-bitmasks-2026-08-26"
BANDS = ("u", "g", "r", "i", "z", "W1", "W2")
HUBBLE_CUT_ENV_KEYS = (
    "QVC_CUT_SPECTRAL_RHAT_MAX",
    "QVC_CUT_LIGHT_CURVE_RHAT_MAX",
    "QVC_CUT_LOG_TAU_UV_RF_MAX",
    "QVC_CUT_APPARENT_MAG_2500_ERR_MAX",
    "QVC_CUT_SED_REDUCED_CHI2_MAX",
    "QVC_CUT_SPECTROSCOPY_REDUCED_CHI2_MAX",
    "QVC_CUT_JAXSEDFIT_JOINT_REDUCED_CHI2_MAX",
    "QVC_CUT_LOO_CHI2_EFF_MAX",
    "QVC_HUBBLE_COMPLETENESS_SMOOTH_SIGMA_MAG",
    "QVC_HUBBLE_COMPLETENESS_SMOOTH_SIGMA_Z",
)
COSMOLOGY = FlatLambdaCDM(H0=70.0, Om0=0.3)
AB_ZEROPOINT_LNU = 51.59477721004232
C_ANGSTROM_S = 2.99792458e18
NANOMAGGY_CGS = 3.631e-29
WISE_AB_MINUS_VEGA = np.array([2.699, 3.339])
HOST_CAPTURE_QUANTILE_LEVELS = np.array([0.05, 0.16, 0.50, 0.84, 0.95])
HOST_CAPTURE_QUANTILE_TOLERANCE = 0.03
HOST_CAPTURE_REFERENCE_DRAWS = 100_000


TARGET_FIELDS = {
    "target0": "SDSS_EBOSS_TARGET0",
    "target1": "SDSS_EBOSS_TARGET1",
    "target2": "SDSS_EBOSS_TARGET2",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dataset.asstr()).astype(str)


def _unsigned(values: Any) -> np.ndarray:
    """Preserve all 64 target bits from signed HDF5 integer storage."""
    return np.asarray(values, dtype=np.int64).view(np.uint64)


def _has_bits(values: np.ndarray, bits: Sequence[int]) -> np.ndarray:
    mask = np.uint64(0)
    for bit in bits:
        mask |= np.uint64(1) << np.uint64(bit)
    return (_unsigned(values) & mask) != 0


def _decode_bits(group: h5py.Group, definition: dict[str, Sequence[int]]) -> np.ndarray:
    selected = np.zeros(len(group["sdss_name"]), dtype=bool)
    for field, bits in definition.items():
        name = TARGET_FIELDS[field]
        if name not in group:
            raise ValueError(f"Spectra metadata is missing target mask {name!r}.")
        selected |= _has_bits(group[name], bits)
    return selected


def decode_eboss_trials(
    group: h5py.Group,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return local eBOSS opportunity, clean alternative trials, CORE, channel."""

    survey = np.char.lower(np.char.strip(_text(group["SDSS_SURVEY"])))
    program = np.char.lower(np.char.strip(_text(group["SDSS_PROGRAMNAME"])))
    main = (survey == "eboss") & (program == "eboss")
    color = _decode_bits(group, EBOSS_COLOR_BITS)
    selected_channels = {
        name: _decode_bits(group, definition)
        for name, definition in EBOSS_ALT_CHANNEL_BITS.items()
    }
    channel_count = np.sum(
        np.column_stack(list(selected_channels.values())), axis=1
    )
    disqualified = _decode_bits(group, EBOSS_DISQUALIFY_BITS)
    clean_alternative = (channel_count == 1) & ~disqualified
    opportunity = main & ~disqualified & (channel_count <= 1) & (color | clean_alternative)
    trial = opportunity & clean_alternative
    channel = np.full(len(color), "unknown", dtype="U16")
    for name, selected in selected_channels.items():
        channel[trial & selected] = name
    return opportunity, trial, color, channel


def _read_observed_photometry(
    dr16q_path: Path, names: np.ndarray
) -> dict[str, np.ndarray]:
    """Join the selected metadata rows to DR16Q by exact trimmed SDSS_NAME."""
    with fits.open(dr16q_path, memmap=True) as hdul:
        primary_names = np.char.strip(np.asarray(hdul[1].data["SDSS_NAME"]).astype(str))
        order = np.argsort(primary_names)
        query = np.char.strip(np.asarray(names).astype(str))
        position = np.searchsorted(primary_names[order], query)
        if np.any(position >= len(order)):
            raise ValueError("At least one spectra row is absent from DR16Q.")
        index = order[np.minimum(position, len(order) - 1)]
        if not np.array_equal(primary_names[index], query):
            missing = query[primary_names[index] != query][:5]
            raise ValueError(f"Exact SDSS_NAME join to DR16Q failed: {missing.tolist()}.")
        hdu2 = hdul[2].data
        sdss_flux = np.asarray(hdu2["PSFFLUX"])[index].astype(float)
        sdss_ivar = np.asarray(hdu2["PSFFLUX_IVAR"])[index].astype(float)
        extinction = np.asarray(hdu2["EXTINCTION"])[index].astype(float)
        wise_flux = np.column_stack((hdu2["W1_FLUX"][index], hdu2["W2_FLUX"][index])).astype(float)
        wise_ivar = np.column_stack((hdu2["W1_FLUX_IVAR"][index], hdu2["W2_FLUX_IVAR"][index])).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sdss_error = np.where(sdss_ivar > 0, 1.0 / np.sqrt(sdss_ivar), np.nan)
        wise_error = np.where(wise_ivar > 0, 1.0 / np.sqrt(wise_ivar), np.nan)
    wise_missing = ~np.isfinite(wise_flux) | ~np.isfinite(wise_error) | (wise_error <= 0.0)
    return {
        "sdss_flux": sdss_flux,
        "sdss_error": sdss_error,
        "extinction": extinction,
        "wise_flux": wise_flux,
        "wise_error": wise_error,
        "wise_missing": wise_missing,
    }


def _finite_photometry(phot: dict[str, np.ndarray]) -> np.ndarray:
    good = np.all(np.isfinite(phot["sdss_flux"]) & np.isfinite(phot["sdss_error"]) &
                  (phot["sdss_error"] > 0), axis=1)
    missing = phot["wise_missing"]
    good &= np.all(missing | (np.isfinite(phot["wise_flux"]) &
                              np.isfinite(phot["wise_error"]) &
                              (phot["wise_error"] > 0)), axis=1)
    return good


def _feature_photometry(phot: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply DR16Q Galactic correction and finite WISE placeholders."""

    factor = np.power(10.0, 0.4 * np.asarray(phot["extinction"], float))
    missing = np.asarray(phot["wise_missing"], bool)
    wise_error = np.asarray(phot["wise_error"], float).copy()
    for band in range(2):
        valid = ~missing[:, band] & np.isfinite(wise_error[:, band]) & (wise_error[:, band] > 0)
        if np.count_nonzero(valid) < 20:
            raise ValueError("Too few observed WISE errors for the eBOSS feature transform.")
        wise_error[~valid, band] = np.median(wise_error[valid, band])
    return {
        "sdss_flux": np.asarray(phot["sdss_flux"], float) * factor,
        "sdss_error": np.asarray(phot["sdss_error"], float) * factor,
        "wise_flux": np.where(missing, 0.0, np.asarray(phot["wise_flux"], float)),
        "wise_error": wise_error,
        "wise_missing": missing,
    }


def program_features(
    phot: dict[str, np.ndarray],
    sdss_softening: np.ndarray,
    wise_softening: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    feature_phot = _feature_photometry(phot)
    missing = feature_phot["wise_missing"]
    x = eboss_target_features(
        feature_phot["sdss_flux"], feature_phot["sdss_error"],
        feature_phot["wise_flux"], feature_phot["wise_error"], missing,
        sdss_softening, wise_softening,
    )
    pattern = np.asarray([
        "wise:" + "".join(row.astype(int).astype(str)) for row in missing
    ])
    if not np.all(np.isfinite(x)):
        raise ValueError("Non-finite eBOSS target features survived eligibility cuts.")
    return x, EBOSS_FEATURE_NAMES, pattern


def empirical_observing_pool(
    phot: dict[str, np.ndarray], opportunity: np.ndarray
) -> dict[str, np.ndarray]:
    """Return paired local depth/extinction/missingness states without fitting."""

    use = np.asarray(opportunity, bool) & _finite_photometry(phot)
    if np.count_nonzero(use) < 100:
        raise ValueError("Too few local eBOSS observing states.")
    subset = {key: np.asarray(value)[use] for key, value in phot.items()}
    feature_phot = _feature_photometry(subset)
    return {
        "sdss_error_raw": subset["sdss_error"],
        "wise_error": feature_phot["wise_error"],
        "extinction": subset["extinction"],
        "wise_missing": subset["wise_missing"],
    }


def targeting_magnitude_eligible(phot: dict[str, np.ndarray]) -> np.ndarray:
    """Local eBOSS proxy: dereddened g<22 or r<22."""

    factor = np.power(10.0, 0.4 * np.asarray(phot["extinction"], float))
    corrected = np.asarray(phot["sdss_flux"], float) * factor
    limit_flux = 10.0 ** ((22.5 - 22.0) / 2.5)
    return (corrected[:, 1] > limit_flux) | (corrected[:, 2] > limit_flux)


def load_host_capture_calibration(
    spectra_fit_h5: Path, eligible_names: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    """Load empirical JAXSedFit PSF-capture posterior summaries in two z bins."""

    eligible = set(np.char.strip(np.asarray(eligible_names).astype(str)).tolist())
    output: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(spectra_fit_h5, "r") as handle:
        group = handle["catalog"]
        required = {
            "sdss_name", "z", "host_capture_group_fraction",
            "host_capture_group_fraction_err",
        }
        missing = required - set(group)
        if missing:
            raise ValueError(
                f"JAXSedFit catalog is missing host-capture fields {sorted(missing)}."
            )
        names = np.char.strip(_text(group["sdss_name"]))
        selected = np.isin(names, np.asarray(sorted(eligible)))
        z = np.asarray(group["z"], float)
        capture = np.asarray(group["host_capture_group_fraction"], float)
        capture_error = np.asarray(
            group["host_capture_group_fraction_err"], float
        )
        finite = (
            selected & np.isfinite(z) & np.isfinite(capture)
            & np.isfinite(capture_error) & (capture_error >= 0.0)
        )
        for label, redshift_mask in (
            ("low", z < 1.0), ("high", z >= 1.0)
        ):
            use = finite & redshift_mask
            if np.count_nonzero(use) < 100:
                raise ValueError(
                    f"Too few local eBOSS JAXSedFit capture donors in {label}-z bin."
                )
            output[label] = {
                "capture": np.clip(capture[use], 0.0, 1.0),
                "capture_error": capture_error[use],
            }
    return output


def draw_host_capture_fraction(
    redshift: np.ndarray,
    calibration: dict[str, dict[str, np.ndarray]],
    rng: np.random.Generator,
) -> np.ndarray:
    capture = np.empty(len(redshift), dtype=float)
    for label, use in (("low", redshift < 1.0), ("high", redshift >= 1.0)):
        pool = calibration[label]
        donor = rng.integers(0, len(pool["capture"]), size=np.count_nonzero(use))
        capture[use] = np.clip(
            pool["capture"][donor]
            + pool["capture_error"][donor] * rng.normal(size=len(donor)),
            0.0,
            1.0,
        )
    return capture


def validate_mock_host_capture_fraction(
    redshift: np.ndarray,
    capture: np.ndarray,
    calibration: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Validate mock PSF host-capture draws against their empirical mixture.

    ``host_capture_group_fraction`` is the posterior-derived fraction of
    integrated host light entering the SDSS PSF.  It is the quantity applied
    to mock ugriz host flux.  In particular, this deliberately does not use
    ``1 - f_AGN_psf``: that is non-variable/total PSF flux, not host flux.
    """

    redshift = np.asarray(redshift, float)
    capture = np.asarray(capture, float)
    if redshift.shape != capture.shape:
        raise ValueError("Host-capture redshift and draw arrays must match.")
    if np.any(~np.isfinite(capture)) or np.any((capture < 0.0) | (capture > 1.0)):
        raise ValueError("Mock host-capture draws must be finite and within [0, 1].")
    diagnostics: dict[str, Any] = {}
    failures = []
    for bin_index, (label, use) in enumerate(
        (("low", redshift < 1.0), ("high", redshift >= 1.0))
    ):
        pool = calibration[label]
        source = np.asarray(pool["capture"], float)
        source_error = np.asarray(pool["capture_error"], float)
        if source.shape != source_error.shape or len(source) == 0:
            raise ValueError(f"Invalid JAXSedFit host-capture pool for {label}-z.")
        if np.count_nonzero(use) == 0:
            raise ValueError(f"No mock host-capture draws in {label}-z bin.")

        # Compare with the same posterior-summary mixture used by the mock
        # sampler, using an independent deterministic Monte Carlo reference.
        reference_rng = np.random.default_rng(910_247 + bin_index)
        donor = reference_rng.integers(
            0, len(source), size=HOST_CAPTURE_REFERENCE_DRAWS
        )
        reference = np.clip(
            source[donor]
            + source_error[donor]
            * reference_rng.normal(size=HOST_CAPTURE_REFERENCE_DRAWS),
            0.0,
            1.0,
        )
        mock_quantiles = np.quantile(capture[use], HOST_CAPTURE_QUANTILE_LEVELS)
        reference_quantiles = np.quantile(reference, HOST_CAPTURE_QUANTILE_LEVELS)
        difference = mock_quantiles - reference_quantiles
        diagnostics[label] = {
            "n_mock": int(np.count_nonzero(use)),
            "n_jaxsedfit_donors": int(len(source)),
            "quantile_levels": HOST_CAPTURE_QUANTILE_LEVELS.tolist(),
            "mock_quantiles": mock_quantiles.tolist(),
            "reference_mixture_quantiles": reference_quantiles.tolist(),
            "difference": difference.tolist(),
            "max_abs_difference": float(np.max(np.abs(difference))),
        }
        if np.any(np.abs(difference) > HOST_CAPTURE_QUANTILE_TOLERANCE):
            failures.append(label)
    if failures:
        raise ValueError(
            "Mock PSF host-capture draws fail JAXSedFit distribution closure in "
            f"redshift bins {failures}; quantile tolerance="
            f"{HOST_CAPTURE_QUANTILE_TOLERANCE}."
        )
    return diagnostics


def _load_qsogen(qsogen_path: Path):
    required = ("qsosed.py", "qsosed_emlines_20210625.dat", "S0_template_norm.sed", "pl_ext_comp_03.sph")
    missing = [name for name in required if not (qsogen_path / name).is_file()]
    if missing:
        raise ValueError(f"qsogen checkout is missing {missing}.")
    spec = importlib.util.spec_from_file_location("qvc_external_qsosed", qsogen_path / "qsosed.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    params = {
        "plslp1": -0.349, "plslp2": 0.593, "plstep": -1.0, "plbrk1": 3880.0,
        "tbb": 1243.6, "plbrk3": 1200.0, "bbnorm": 3.961, "scal_emline": -0.9936,
        "emline_type": None, "scal_halpha": 1.0, "scal_lya": 1.0, "scal_nlr": 1.0,
        "emline_template": np.genfromtxt(qsogen_path / required[1], unpack=True),
        "galaxy_template": np.genfromtxt(qsogen_path / required[2], unpack=True),
        "reddening_curve": np.genfromtxt(qsogen_path / required[3], unpack=True),
        "zlum_lumval": np.array([[0.23, 0.34, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.3, 3.7, 4.13, 4.5],
                                   [-21.76, -22.9, -24.1, -25.4, -26.0, -26.6, -27.1, -27.6, -27.9, -28.1, -28.4, -28.6, -28.9]]),
        "M_i": -27.0, "beslope": 0.183, "benorm": -27.0, "bcnorm": False,
        "lyForest": True, "lylim": 912.0, "gflag": True, "fragal": 0.244, "gplind": 0.684,
    }
    return module.Quasar_sed, params


def _filter_curves(qsogen_path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    curves = []
    for name in ("SDSS_u", "SDSS_g", "SDSS_r", "SDSS_i", "SDSS_z", "WISE_W1", "WISE_W2"):
        wave, response = np.genfromtxt(qsogen_path / "filters" / f"{name}.filter", unpack=True)
        order = np.argsort(wave)
        curves.append((np.asarray(wave[order], float), np.clip(np.asarray(response[order], float), 0, None)))
    return curves


def _absolute_m2500(m_hd: float, redshift: float) -> float:
    return float(m_hd - COSMOLOGY.distmod(redshift).value)


def _project_component(
    rest_wave: np.ndarray, luminosity_lambda: np.ndarray, redshift: float,
    curves: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    dl_cm = float(COSMOLOGY.luminosity_distance(redshift).to_value("cm"))
    output = np.empty(len(curves), float)
    for index, (wave, response) in enumerate(curves):
        rest_filter_wave = wave / (1 + redshift)
        active = response > 0.0
        if (
            np.min(rest_filter_wave[active]) < rest_wave[0]
            or np.max(rest_filter_wave[active]) > rest_wave[-1]
        ):
            raise RuntimeError(
                f"SED wavelength grid does not cover targeting filter {index}."
            )
        observed_flambda = np.interp(
            rest_filter_wave, rest_wave, luminosity_lambda
        ) / (4 * np.pi * dl_cm**2 * (1 + redshift))
        numerator = np.trapezoid(observed_flambda * response * wave, wave)
        denominator = C_ANGSTROM_S * np.trapezoid(response / wave, wave)
        output[index] = numerator / denominator / NANOMAGGY_CGS
    return output


def qsogen_paired_flux(
    m_hd: float, redshift: float, realization: int, *, QuasarSED, params: dict[str, Any],
    curves: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return paired nuclear and integrated-host AB nanomaggy fluxes."""
    values = dict(params)
    # A small native-template quadrature, paired exactly across host branches.
    values["emline_type"] = (-1.0, 0.0, 1.0, None)[realization % 4]
    values["plslp1"] = params["plslp1"] + (-0.18, -0.06, 0.06, 0.18)[(realization // 4) % 4]
    rest_wave = np.logspace(np.log10(600.0), np.log10(60_000.0), 16_001)
    absolute = _absolute_m2500(m_hd, redshift)
    lnu_2500 = 10 ** ((AB_ZEROPOINT_LNU - absolute) / 2.5)
    target_llambda = lnu_2500 * C_ANGSTROM_S / 2500.0**2

    # First construct only the nucleus.  Its normalized L3000 fixes qsogen's
    # documented M_i control variable; no empirical M2500 offset is used.
    values["gflag"] = False
    values["M_i"] = -27.0
    nuclear_sed = QuasarSED(
        z=redshift, LogL3000=None, wavlen=rest_wave, ebv=0.0, params=values
    )
    nuclear_shape = np.asarray(nuclear_sed.flux, float)
    nuclear_at_2500 = float(np.interp(2500.0, rest_wave, nuclear_shape))
    if not np.isfinite(nuclear_at_2500) or nuclear_at_2500 <= 0:
        raise RuntimeError("qsogen produced nonpositive nuclear 2500-A flux.")
    scale = target_llambda / nuclear_at_2500
    nuclear = nuclear_shape * scale
    log_l3000 = math.log10(3000.0 * np.interp(3000.0, rest_wave, nuclear))
    values["M_i"] = (35.3 - log_l3000) / 0.4
    values["gflag"] = True
    host_sed = QuasarSED(
        z=redshift, LogL3000=None, wavlen=rest_wave, ebv=0.0, params=values
    )
    host_shape = np.asarray(host_sed.host_galaxy_flux, float)
    second_nuclear_shape = np.asarray(host_sed.flux, float) - host_shape
    second_scale = target_llambda / float(
        np.interp(2500.0, rest_wave, second_nuclear_shape)
    )
    nuclear = second_nuclear_shape * second_scale
    host = host_shape * second_scale
    # m_HD is the LF nuclear coordinate and is never regenerated from total light.
    recovered = AB_ZEROPOINT_LNU - 2.5 * math.log10(
        np.interp(2500.0, rest_wave, nuclear) * 2500.0**2 / C_ANGSTROM_S
    ) + float(COSMOLOGY.distmod(redshift).value)
    if abs(recovered - m_hd) > 1e-12:
        raise RuntimeError(f"qsogen nuclear normalization changed m_HD by {recovered - m_hd:.3g}.")
    return (
        _project_component(rest_wave, nuclear, redshift, curves),
        _project_component(rest_wave, host, redshift, curves),
        target_llambda,
    )


def wise_ab_to_vega_nanomaggy(flux: Any) -> np.ndarray:
    """Convert physical AB nanomaggies to the DR16Q WISE Vega convention."""

    value = np.asarray(flux, dtype=float)
    if value.shape[-1] != 2:
        raise ValueError("WISE conversion requires W1/W2 on the final axis.")
    return value * np.power(10.0, 0.4 * WISE_AB_MINUS_VEGA)


def _old_completeness_hash(
    *, light_curve_h5: Path, spectra_fit_h5: Path, lf_mock: Path,
    spectra_metadata: Path,
    magnitude_convention: str, completeness_magnitude: str,
    z_range: tuple[float, float], sdss_target_selection: str,
) -> str:
    observed, _ = load_agn_data(
        str(light_curve_h5), apply_cut=True, spectra_fit_h5=[str(spectra_fit_h5)],
        magnitude_convention=magnitude_convention,
        completeness_magnitude=completeness_magnitude,
        sdss_target_metadata_h5=str(spectra_metadata),
        z_range=z_range, sdss_target_selection=sdss_target_selection,
        completeness_stratification="none", plot_diagnostics=False,
        plot_path="/tmp/qvc-color-prepare", cut_report_path=None,
    )
    observed = prepare_completeness_magnitude_columns(observed, completeness_magnitude)
    model = get_completeness_function_2d(observed, sim_file=str(lf_mock), plot=False)[0]
    return hash_completeness_2d(model)


def _mock_grid(lf_mock: Path, z_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    # Nodes include the exact HD support boundaries, so interpolation never
    # silently removes the first/last half-bin.
    m_grid = np.linspace(
        COMPLETENESS_MAG_EDGE_MIN,
        COMPLETENESS_MAG_EDGE_MAX,
        COMPLETENESS_N_MAG_BINS + 1,
    )
    n_z = max(2, int(np.ceil((z_range[1] - z_range[0]) / 0.1)) + 1)
    z_grid = np.linspace(z_range[0], z_range[1], n_z)
    with h5py.File(lf_mock, "r") as handle:
        for required in ("apparent_mag_2500", "z"):
            if required not in handle:
                raise ValueError(f"LF mock is missing {required!r}.")
    return m_grid, z_grid


def prepare_catalog(
    output_path: Path, *, lf_mock: Path, dr16q: Path, spectra_metadata: Path,
    qsogen_path: Path, light_curve_h5: Path, spectra_fit_h5: Path,
    draws_per_cell: int = 256, sed_realizations: int = 16, seed: int = 12345,
    z_range: tuple[float, float] = (0.44, 3.16),
    magnitude_convention: str = "dereddened", completeness_magnitude: str = "attenuated",
    sdss_target_selection: str = "eboss-color-sensitivity",
) -> str:
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace prepared catalog {output_path}.")
    if draws_per_cell < 200 or sed_realizations < 1:
        raise ValueError("Preparation requires at least 200 draws/cell and one qsogen realization.")
    rng = np.random.default_rng(seed)
    with h5py.File(spectra_metadata, "r") as handle:
        catalog = handle["catalog"]
        names = _text(catalog["sdss_name"])
        observed = _read_observed_photometry(dr16q, names)
        opportunity, trial, color, channel = decode_eboss_trials(catalog)
        photometry_ok = _finite_photometry(observed)
        magnitude_ok = targeting_magnitude_eligible(observed)
        use = trial & photometry_ok & magnitude_ok
        if np.count_nonzero(use) < 40 or np.unique(color[use]).size != 2:
            raise ValueError("Insufficient clean local eBOSS alternative trials.")
        observing_pool = empirical_observing_pool(observed, opportunity)
        softening_phot = _feature_photometry({
            key: np.asarray(value)[opportunity & photometry_ok]
            for key, value in observed.items()
        })
        sdss_softening = np.median(softening_phot["sdss_error"], axis=0)
        wise_softening = np.median(softening_phot["wise_error"], axis=0)
        subset = {key: np.asarray(value)[use] for key, value in observed.items()}
        features, feature_names, patterns = program_features(
            subset, sdss_softening, wise_softening
        )
        success = color[use]
        closure = leave_one_channel_out_closure(
            "eboss", feature_names, features, success, channel[use], strict=True
        )
        training = (
            features, feature_names, patterns, success, channel[use], names[use]
        )
        eligible_names = names[opportunity]
    capture_calibration = load_host_capture_calibration(
        spectra_fit_h5, eligible_names
    )
    m_grid, z_grid = _mock_grid(lf_mock, z_range)
    QuasarSED, qsogen_params = _load_qsogen(qsogen_path)
    curves = _filter_curves(qsogen_path)
    n_cells = len(m_grid) * len(z_grid)
    n = n_cells * draws_per_cell
    m_hd = np.empty(n); redshift = np.empty(n); m_bin = np.empty(n, int); z_bin = np.empty(n, int)
    for cell_index in range(n_cells):
        start = cell_index * draws_per_cell; stop = start + draws_per_cell
        i, j = divmod(cell_index, len(z_grid))
        # Conditional C(m,z) uses fixed LF coordinates.  Equal weights are
        # quadrature weights over qsogen/noise realizations, not observed-row
        # LF weights.
        m_hd[start:stop], redshift[start:stop] = m_grid[i], z_grid[j]
        m_bin[start:stop], z_bin[start:stop] = i, j
    nuclear_flux = np.empty((n, 7))
    integrated_host_flux = np.empty((n, 7))
    luminosity = np.empty(n)
    cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, float]] = {}
    for row in range(n):
        key = (m_bin[row], z_bin[row], row % sed_realizations)
        if key not in cache:
            cache[key] = qsogen_paired_flux(
                float(m_grid[key[0]]), float(z_grid[key[1]]), key[2],
                QuasarSED=QuasarSED, params=qsogen_params, curves=curves,
            )
        nuclear, host, llambda = cache[key]
        nuclear_flux[row], integrated_host_flux[row] = nuclear, host
        luminosity[row] = llambda
    nuclear_flux[:, 5:] = wise_ab_to_vega_nanomaggy(nuclear_flux[:, 5:])
    integrated_host_flux[:, 5:] = wise_ab_to_vega_nanomaggy(
        integrated_host_flux[:, 5:]
    )
    host_capture = draw_host_capture_fraction(redshift, capture_calibration, rng)
    noiseless_host = nuclear_flux.copy()
    noiseless_host[:, :5] += host_capture[:, None] * integrated_host_flux[:, :5]
    noiseless_host[:, 5:] += integrated_host_flux[:, 5:]
    noiseless_nohost = nuclear_flux.copy()
    host_capture_validation = validate_mock_host_capture_fraction(
        redshift, host_capture, capture_calibration,
    )

    pool_size = len(observing_pool["sdss_error_raw"])
    state = rng.integers(0, pool_size, size=n)
    extinction = observing_pool["extinction"][state]
    wise_missing = observing_pool["wise_missing"][state]
    extinction_factor = np.power(10.0, -0.4 * extinction)
    raw_host = noiseless_host.copy()
    raw_nohost = noiseless_nohost.copy()
    raw_host[:, :5] *= extinction_factor
    raw_nohost[:, :5] *= extinction_factor
    epsilon = rng.normal(size=(n, 7))
    paired_error = np.column_stack((
        observing_pool["sdss_error_raw"][state],
        observing_pool["wise_error"][state],
    ))
    noisy_host, noisy_nohost = apply_paired_flux_noise(
        raw_host, raw_nohost, paired_error, epsilon
    )
    assert_paired_nuclear_state(m_hd, m_hd.copy(), luminosity, luminosity.copy())
    old_hash = _old_completeness_hash(
        light_curve_h5=light_curve_h5, spectra_fit_h5=spectra_fit_h5,
        spectra_metadata=spectra_metadata, lf_mock=lf_mock,
        magnitude_convention=magnitude_convention, completeness_magnitude=completeness_magnitude,
        z_range=z_range, sdss_target_selection=sdss_target_selection,
    )
    cut_manifest = {
        "intrinsic_support": ["Wang Type-1 LF m2500/z support"],
        "target_eligibility": [
            "local main-eBOSS provenance",
            "g_dered<22 or r_dered<22",
            "finite signed-flux representation",
        ],
        "target_selection": [
            "eBOSS CORE success among one clean alternative-channel trial"
        ],
        "downstream_survival": [],
        "hd_analysis": ["Hubble quality cuts (used only to hash C_old)"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text_dtype = h5py.string_dtype("utf-8")
    with h5py.File(output_path, "x") as output:
        output.attrs["schema"] = PREPARED_CATALOG_SCHEMA
        output.attrs["hubble_cut_configuration_json"] = json.dumps(
            {key: os.environ.get(key) for key in HUBBLE_CUT_ENV_KEYS},
            sort_keys=True,
        )
        output.attrs["cut_manifest_json"] = json.dumps(cut_manifest, sort_keys=True)
        output.attrs["opportunity_rules_json"] = json.dumps(
            {
                "program": "eboss",
                "eligibility": "local photometry/provenance proxy; no morphology",
                "survey": "eboss",
                "programname": "eboss",
                "ambiguous_target_combinations": "excluded",
            }, sort_keys=True)
        output.attrs["old_completeness_hash"] = old_hash
        output.attrs["target_provenance_json"] = json.dumps(
            {"assignment": "eBOSS target masks plus main-program provenance",
             "definition_version": TARGET_BIT_DEFINITION_VERSION,
             "definition_source": "https://www.sdss.org/dr17/algorithms/bitmasks/",
             "color_bits": EBOSS_COLOR_BITS,
             "alternative_channel_bits": EBOSS_ALT_CHANNEL_BITS,
             "disqualify_bits": EBOSS_DISQUALIFY_BITS}, sort_keys=True)
        output.attrs["input_catalog_hashes_json"] = json.dumps(
            {"lf_mock": _sha256(lf_mock), "dr16q": _sha256(dr16q),
             "spectra_metadata": _sha256(spectra_metadata), "light_curve": _sha256(light_curve_h5),
             "spectra_fit": _sha256(spectra_fit_h5)}, sort_keys=True)
        qsogen_commit = subprocess.run(
            ["git", "-C", str(qsogen_path), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True).stdout.strip()
        output.attrs["qsogen_configuration_json"] = json.dumps(
            {"path": str(qsogen_path.resolve()), "commit": qsogen_commit,
             "host_counterfactual": "eBOSS photometry-only paired physical photometry",
             "m_hd": "fixed LF nuclear m2500 by construction",
             "sed_realizations": sed_realizations,
             "rest_wavelength_angstrom": [600.0, 60000.0],
             "M_i_relation": "LogL3000=-0.4*M_i+35.3",
             "WISE_AB_minus_Vega": WISE_AB_MINUS_VEGA.tolist()}, sort_keys=True)
        output.attrs["noise_model_json"] = json.dumps(
            {"kind": "paired empirical error-vector bootstrap; no fitted flux dependence",
             "bands": BANDS, "pool_size": pool_size, "seed": seed}, sort_keys=True)
        output.attrs["host_capture_calibration_json"] = json.dumps(
            {
                "source": "JAXSedFit catalog posterior summaries",
                "redshift_bins": ["z<1", "z>=1"],
                "ugriz_capture": "host_capture_group_fraction empirical posterior-summary mixture",
                "WISE_capture": "integrated host",
                "validation_quantity": "scalar fraction of integrated host light entering the SDSS PSF",
                "excluded_quantity": "f_AGN_psf is variable-AGN/total PSF and is not a host fraction",
                "quantile_tolerance": HOST_CAPTURE_QUANTILE_TOLERANCE,
                "validation": host_capture_validation,
            }, sort_keys=True)
        output.attrs["closure_diagnostics_json"] = json.dumps(
            closure, sort_keys=True
        )
        output.attrs["feature_transform_json"] = json.dumps(
            {
                "feature_names": feature_names,
                "sdss_softening_nanomaggy": sdss_softening.tolist(),
                "wise_softening_vega_nanomaggy": wise_softening.tolist(),
                "galactic_extinction": "DR16Q EXTINCTION",
            }, sort_keys=True)
        output.attrs["lf_mock_path"] = str(lf_mock.resolve())
        output.create_dataset("magnitude_grid", data=m_grid)
        output.create_dataset("redshift_grid", data=z_grid)
        features, feature_names, patterns, success, channel, trial_names = training
        train = output.create_group("training/eboss")
        train.create_dataset("features", data=features, compression="gzip")
        train.create_dataset("success", data=success.astype(np.int8))
        train.create_dataset(
            "marks", data=np.where(success, "both", "alt_only").astype(object),
            dtype=text_dtype,
        )
        train.create_dataset("patterns", data=np.asarray(patterns, dtype=object), dtype=text_dtype)
        train.create_dataset("alternative_channel", data=np.asarray(channel, dtype=object), dtype=text_dtype)
        train.create_dataset("sdss_name", data=np.asarray(trial_names, dtype=object), dtype=text_dtype)
        train.attrs["feature_names_json"] = json.dumps(feature_names)

        host_phot = {
            "sdss_flux": noisy_host[:, :5], "sdss_error": paired_error[:, :5],
            "extinction": extinction, "wise_flux": noisy_host[:, 5:],
            "wise_error": paired_error[:, 5:], "wise_missing": wise_missing,
        }
        nohost_phot = {
            "sdss_flux": noisy_nohost[:, :5], "sdss_error": paired_error[:, :5],
            "extinction": extinction, "wise_flux": noisy_nohost[:, 5:],
            "wise_error": paired_error[:, 5:], "wise_missing": wise_missing,
        }
        fh, names_h, ph = program_features(host_phot, sdss_softening, wise_softening)
        fn, names_n, pn = program_features(nohost_phot, sdss_softening, wise_softening)
        if names_h != feature_names or names_n != feature_names:
            raise RuntimeError("Observed/mock eBOSS feature schema mismatch.")
        mock = output.create_group("mock/eboss")
        for name, value in (
            ("features_host", fh), ("features_nohost", fn),
            ("m_bin", m_bin), ("z_bin", z_bin), ("weights", np.ones(n)),
            ("m_hd_host", m_hd), ("m_hd_nohost", m_hd),
            ("luminosity_host", luminosity), ("luminosity_nohost", luminosity),
            ("observing_state_id", state), ("noise_normal", epsilon),
            ("host_capture_fraction", host_capture),
        ):
            mock.create_dataset(name, data=value, compression="gzip")
        mock.create_dataset("patterns_host", data=np.asarray(ph, dtype=object), dtype=text_dtype)
        mock.create_dataset("patterns_nohost", data=np.asarray(pn, dtype=object), dtype=text_dtype)
    return _sha256(output_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lf-mock", type=Path)
    parser.add_argument("--lf-model", default="wang2026_type1_lade_a")
    parser.add_argument("--lf-oversample", type=float, default=4.0)
    parser.add_argument("--lf-max-rows", type=int, default=2_000_000)
    parser.add_argument("--dr16q", type=Path, required=True)
    parser.add_argument("--spectra-metadata", type=Path, required=True)
    parser.add_argument("--qsogen-path", type=Path, required=True)
    parser.add_argument("--light-curve-h5", type=Path, required=True)
    parser.add_argument("--spectra-fit-h5", type=Path, required=True)
    parser.add_argument("--draws-per-cell", type=int, default=256)
    parser.add_argument("--sed-realizations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--z-range", type=float, nargs=2, default=(0.44, 3.16))
    parser.add_argument("--magnitude-convention", choices=("dereddened", "attenuated"), default="dereddened")
    parser.add_argument("--completeness-magnitude", choices=("dereddened", "attenuated"), default="attenuated")
    parser.add_argument(
        "--sdss-target-selection", default="eboss-color-sensitivity"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    lf_mock = args.lf_mock
    if lf_mock is None:
        # Use exactly the Hubble loader and footprint estimator, then delegate
        # to the existing LF cache generator.  The resulting path is stored in
        # the prepared catalog and reused by every matched Hubble mode.
        from qvc.hubble.hubble_fit import (
            estimate_sky_box_area_deg2,
            generate_fresh_completeness_sim_file,
        )

        _, parent = load_agn_data(
            str(args.light_curve_h5), apply_cut=True,
            spectra_fit_h5=[str(args.spectra_fit_h5)],
            magnitude_convention=args.magnitude_convention,
            completeness_magnitude=args.completeness_magnitude,
            sdss_target_metadata_h5=str(args.spectra_metadata),
            z_range=tuple(args.z_range),
            sdss_target_selection=args.sdss_target_selection,
            completeness_stratification="none", plot_diagnostics=False,
            plot_path="/tmp/qvc-color-prepare", cut_report_path=None,
        )
        lf_mock = Path(generate_fresh_completeness_sim_file(
            "/tmp/qvc-color-prepare", area_deg2=estimate_sky_box_area_deg2(parent),
            seed=args.seed, oversample=args.lf_oversample, max_rows=args.lf_max_rows,
            lf_model=args.lf_model, z_range=tuple(args.z_range),
            completeness_magnitude=args.completeness_magnitude,
        ))
    digest = prepare_catalog(
        args.output, lf_mock=lf_mock, dr16q=args.dr16q,
        spectra_metadata=args.spectra_metadata, qsogen_path=args.qsogen_path,
        light_curve_h5=args.light_curve_h5, spectra_fit_h5=args.spectra_fit_h5,
        draws_per_cell=args.draws_per_cell, sed_realizations=args.sed_realizations,
        seed=args.seed, z_range=tuple(args.z_range),
        magnitude_convention=args.magnitude_convention,
        completeness_magnitude=args.completeness_magnitude,
        sdss_target_selection=args.sdss_target_selection,
    )
    print(json.dumps({"prepared_catalog": str(args.output), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
