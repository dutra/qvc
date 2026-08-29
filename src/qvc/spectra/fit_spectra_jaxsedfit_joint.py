#!/usr/bin/env python3
"""Jointly fit quasar broadband SEDs and SDSS spectra with jaxsedfit.

This is the SED-first companion to :mod:`qvc.spectra.fit_spectra`.  It reuses
that module's QVC sample selection, DR16Q matching, SDSS spectrum cache, and
light-curve PSF photometry, but fits a single physical SED jointly to:

* a caller-provided saved broadband-SED table;
* QVC's error-weighted SDSS light-curve PSF magnitudes; and
* the native-pixel SDSS spectrum through jaxsedfit's ``jaxqsofit`` backend.

The original ``fit_spectra.py`` remains the spectrum-first production path.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import multiprocessing as mp
import traceback
import warnings
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import numpyro.distributions as dist
import pandas as pd

from qvc.mcmc_diagnostics import (
    compute_numpyro_summary,
    convergence_fields,
    print_numpyro_summary_dict,
)
from qvc.provenance import (
    build_run_record,
    fingerprint_path,
    merge_history,
    read_hdf5_provenance,
    runtime_state,
    write_hdf5_provenance,
)
from tqdm import tqdm

from qvc.spectra import fit_spectra as legacy
from qvc.spectra.catalog_hdf5 import (
    ALPHA_NU_BLUE_WAVELENGTH_ANGSTROM,
    ALPHA_NU_RED_WAVELENGTH_ANGSTROM,
    F_HOST_2500_PSF_DRAW_COUNT,
    GRAHSP_ATTENUATION_BREAK_ANGSTROM,
    GRAHSP_ATTENUATION_NORMALIZATION,
    GRAHSP_ATTENUATION_OPTICAL_INDEX,
    JOINT_POSTERIOR_DRAW_COUNT,
    JOINT_POSTERIOR_DRAW_FIELDS,
    JOINT_POSTERIOR_SCALAR_SUMMARY_FIELDS,
    JOINT_PSF_PHOTOMETRY_BANDS,
    JOINT_PSF_PHOTOMETRY_DRAW_COUNT,
    PSF_AGN_FRACTION_DRAW_COUNT,
    write_spectra_catalog_hdf5,
)


C_ANGSTROM_PER_SECOND = 2.99792458e18
AB_ZEROPOINT_MJY = 3.631e6
METER_PER_MEGAPARSEC = 3.085677581491367e22
POSTERIOR_BUNDLE_FORMAT = "jaxsedfit_samples_meta_v2"
PSF_AGN_FRACTION_BANDS = tuple(legacy.SDSS_BANDS)
# Legacy marker retained for the standalone old-run repair utility only. It is
# never passed into the main-based fit configuration.
QVC_PSF_HOST_CAPTURE_GROUP = "qvc_sdss_psf"
HOST_FRACTION_REST_WAVELENGTH_ANGSTROM = 2500.0
SDSS_TYPICAL_PSF_FWHM_ARCSEC = 1.4
SDSS_STATIC_PSF_FWHM_ARCSEC = {
    "u": 1.53,
    "g": 1.44,
    "r": 1.32,
    "i": 1.26,
    "z": 1.29,
}
SDSS_LEGACY_FIBER_DIAMETER_ARCSEC = 3.0
SDSS_BOSS_FIBER_DIAMETER_ARCSEC = 2.0
HOST_CAPTURE_BUNDLE_ATTR = "qvc_host_capture_model"
HOST_CAPTURE_BUNDLE_MARKER = "static_sdss_psf_fwhm"
HOST_CAPTURE_PSF_FWHM_ATTR = "qvc_f_host_2500_psf_fwhm_arcsec"
HOST_CAPTURE_QVC_FWHM_ATTR = "qvc_sdss_psf_fwhm_arcsec"


class IncompatibleHostCaptureResumeError(RuntimeError):
    """Raised when a resume bundle uses a different host-capture model."""


def _fits_table_scalar(hdul, column_name):
    """Return the first scalar found for a column in a spectrum FITS file."""
    for hdu in hdul[1:]:
        data = getattr(hdu, "data", None)
        names = getattr(getattr(data, "dtype", None), "names", None) or ()
        if column_name not in names or len(data) == 0:
            continue
        value = data[column_name][0]
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return value.item() if isinstance(value, np.generic) else value
    return None


def sdss_spectrum_aperture_diameter_arcsec(hdul):
    """Return the fiber diameter for an SDSS-I--IV spectrum.

    SDSS-I/II Legacy and SEGUE-2 used the original 3-arcsec fibers. BOSS and
    eBOSS used the upgraded 2-arcsec fibers. The released spectra identify
    those families through SURVEY/INSTRUMENT and RUN2D metadata.
    """
    survey = str(_fits_table_scalar(hdul, "SURVEY") or "").strip().lower()
    instrument = str(_fits_table_scalar(hdul, "INSTRUMENT") or "").strip().lower()
    if survey in {"sdss", "segue", "segue1", "segue2"} or instrument == "sdss":
        return SDSS_LEGACY_FIBER_DIAMETER_ARCSEC
    if survey in {"boss", "eboss"} or instrument == "boss":
        return SDSS_BOSS_FIBER_DIAMETER_ARCSEC

    # In the combined SDSS-I--IV releases, 26 is Legacy, 103 is the SDSS
    # stellar-cluster reduction, and 104 is SEGUE-2. Versioned v5_* runs are
    # the BOSS/eBOSS reduction family.
    run2d = _fits_table_scalar(hdul, "RUN2D")
    if run2d is None and len(hdul):
        run2d = getattr(hdul[0], "header", {}).get("RUN2D")
    run2d = str(run2d or "").strip().lower()
    if run2d in {"26", "103", "104"}:
        return SDSS_LEGACY_FIBER_DIAMETER_ARCSEC
    if run2d.startswith("v5_"):
        return SDSS_BOSS_FIBER_DIAMETER_ARCSEC

    raise ValueError(
        "Cannot determine whether this spectrum used the 3-arcsec SDSS "
        "fibers or the 2-arcsec BOSS/eBOSS fibers from its FITS metadata "
        f"(SURVEY={survey!r}, INSTRUMENT={instrument!r}, RUN2D={run2d!r})."
    )


class M2500ReconstructionError(RuntimeError):
    """Raised when posterior draws cannot reproduce physical m2500 values."""


def _host_capture_resume_message(path, detail):
    return (
        f"Cannot resume spectral posterior bundle {path}: {detail}. "
        "This run requires static SDSS ugriz PSF FWHMs and JAXSEDFit's fitted "
        "host-capture scale, without QVC missing-PSF fraction parameters. Run "
        "fresh spectral inference when the saved model contract differs."
    )


JOINT_CHI2_SITES = (
    "sed_chi2",
    "sed_n_eff",
    "sed_reduced_chi2",
    "spectroscopy_chi2",
    "spectroscopy_n_eff",
    "spectroscopy_reduced_chi2",
    "joint_chi2",
    "joint_n_eff",
    "joint_reduced_chi2",
)
# Scalar deterministic sites that the legacy CSV writer received in-memory
# from JAXSEDFit, but compact v2 posterior bundles intentionally do not retain.
# Requesting them during prediction lets resume jobs reproduce the same flat
# catalog without expanding the saved posterior bundle or changing JAXSEDFit.
LEGACY_CSV_SCALAR_PREDICTION_SITES = (
    "ebv_agn",
    "ebv_gal",
    "formed_stellar_mass",
    "gal_lgmet_fit",
    "gal_lgmet_scatter_fit",
    "hot_fcov",
    "jqf_line_OIII_wing_log_fwhm",
    "jqf_line_broad_center_value_0",
    "jqf_line_broad_center_value_1",
    "jqf_line_broad_center_value_2",
    "jqf_line_broad_center_value_3",
    "jqf_line_coronal_center",
    "jqf_line_coronal_ion_log_fwhm",
    "jqf_line_dmu_independent_group",
    "jqf_line_high_ion_center",
    "jqf_line_high_ion_log_fwhm",
    "jqf_line_log_broad_fwhm",
    "jqf_line_log_fwhm_delta_group",
    "jqf_line_low_ion_log_fwhm",
    "jqf_line_nlr_center",
    "log_agn_amp_fit",
    "log_sfh_tau_over_age",
    "mass_metallicity_relation_logprior",
    "pl_cutoff",
    "sfh_age_gyr_fit",
    "sfh_tau_gyr_fit",
    "surviving_mass_fraction",
    "systematics_width",
    "uv_slope",
)
HUBBLE_MAGNITUDE_SITES = (
    "m_2500_dereddened",
    "m_2500_attenuated_model",
)
M2500_POSTERIOR_SITES = (
    "log_agn_amp",
    "pl_slope",
    "pl_bend_loc",
    "pl_bend_width",
    "uv_slope",
    "pl_cutoff",
    "ebv_gal",
    "ebv_agn",
)
M2500_REQUIRED_POSTERIOR_SITES = (
    *M2500_POSTERIOR_SITES,
)
M2500_CATALOG_SITES = (
    "m_2500_dereddened",
    "m_2500_attenuated_model",
    "a_2500_galaxy",
    "a_2500_internal",
    "a_2500_total",
)
ALPHA_NU_CATALOG_SITES = (
    "alpha_nu_intrinsic_1450_2500",
    "alpha_nu_attenuated_1450_2500",
)
DERIVED_HUBBLE_CATALOG_SITES = (
    *M2500_CATALOG_SITES,
    *ALPHA_NU_CATALOG_SITES,
)
DERIVED_SPECTRAL_CONVERGENCE_SITES = (
    *HUBBLE_MAGNITUDE_SITES,
    "a_2500_total",
    *ALPHA_NU_CATALOG_SITES,
)
SPECTRAL_CONVERGENCE_DISPLAY_SITES = (
    "ebv_gal",
    "log_ebv_gal",
    "ebv_agn",
    "log_ebv_agn",
    "pl_slope",
    "log_agn_amp",
    *DERIVED_SPECTRAL_CONVERGENCE_SITES,
)


def flambda_1e17_to_mjy(wave_angstrom, flux_1e17):
    """Convert SDSS ``1e-17 erg s-1 cm-2 A-1`` fluxes to mJy."""
    wave = np.asarray(wave_angstrom, dtype=float)
    flux = np.asarray(flux_1e17, dtype=float)
    return flux * 1e-17 * wave**2 / C_ANGSTROM_PER_SECOND / 1e-26


def ab_mag_to_mjy(magnitude):
    return AB_ZEROPOINT_MJY * 10.0 ** (-0.4 * np.asarray(magnitude, dtype=float))


def ab_mag_err_to_mjy_err(magnitude, magnitude_error):
    flux = ab_mag_to_mjy(magnitude)
    return flux * np.log(10.0) * 0.4 * np.asarray(magnitude_error, dtype=float)


def _sample_draws(samples, name, default=None):
    value = (samples or {}).get(name, default)
    if value is None:
        raise KeyError(f"JAXSEDFit posterior lacks required site {name!r}.")
    return np.asarray(value, dtype=float).reshape(-1)


def _broadcast_draws(*values):
    size = max(np.asarray(value).size for value in values)
    out = []
    for value in values:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 1:
            arr = np.full(size, arr.item(), dtype=float)
        elif arr.size != size:
            raise M2500ReconstructionError(
                "Posterior sites used for m2500 have incompatible draw counts."
            )
        out.append(arr)
    return out


def _validate_m2500_draws(draws):
    """Validate the exact attenuation identities on paired posterior draws."""

    required = (
        "m_2500_dereddened_draws",
        "m_2500_attenuated_model_draws",
        "a_2500_galaxy_draws",
        "a_2500_internal_draws",
        "a_2500_total_draws",
    )
    missing = [name for name in required if name not in draws]
    if missing:
        raise M2500ReconstructionError(
            f"m2500 reconstruction lacks required draws: {missing}."
        )

    arrays = {
        name: np.asarray(draws[name], dtype=float).reshape(-1)
        for name in required
    }
    sizes = {values.size for values in arrays.values()}
    if sizes == {0} or len(sizes) != 1:
        raise M2500ReconstructionError(
            "m2500 reconstruction produced empty or misaligned posterior draws."
        )
    nonfinite = [
        name for name, values in arrays.items() if not np.all(np.isfinite(values))
    ]
    if nonfinite:
        raise M2500ReconstructionError(
            f"m2500 reconstruction produced nonfinite draws for {nonfinite}."
        )

    attenuation = (
        arrays["a_2500_galaxy_draws"]
        + arrays["a_2500_internal_draws"]
    )
    if np.any(attenuation < -1e-12):
        raise M2500ReconstructionError(
            "m2500 reconstruction produced negative attenuation."
        )
    if np.all(attenuation == 0.0):
        raise M2500ReconstructionError(
            "m2500 reconstruction produced zero attenuation for every draw."
        )
    identities = (
        (
            arrays["a_2500_total_draws"],
            attenuation,
            "A_2500,total != A_2500,galaxy + A_2500,internal",
        ),
        (
            arrays["m_2500_attenuated_model_draws"]
            - arrays["m_2500_dereddened_draws"],
            attenuation,
            "m_2500,attenuated - m_2500,dereddened != A_2500,total",
        ),
    )
    for actual, expected, message in identities:
        if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
            raise M2500ReconstructionError(message)
    return draws


def posterior_samples_for_m2500(samples, prediction=None):
    """Combine latent and regenerated deterministic draws used by m2500.

    Compact JAXSEDFit posterior bundles intentionally retain only latent
    sites.  In particular, the deterministic ``ebv_gal`` and ``ebv_agn``
    draws are regenerated by ``predict``.  Silently replacing those missing
    sites with zero makes resumed fits report an attenuated magnitude that is
    identical to the dereddened magnitude, so require the physical inputs and
    prefer their regenerated values when available.
    """

    combined = dict(samples or {})
    if prediction is not None:
        missing_dust = [
            name for name in ("ebv_gal", "ebv_agn") if name not in prediction
        ]
        if missing_dust:
            raise M2500ReconstructionError(
                "JAXSEDFit prediction did not regenerate required physical "
                f"dust sites: {missing_dust}."
            )
    prediction = prediction or {}
    for name in M2500_POSTERIOR_SITES:
        if name in prediction:
            combined[name] = prediction[name]

    missing = [
        name for name in M2500_REQUIRED_POSTERIOR_SITES if name not in combined
    ]
    if missing:
        raise M2500ReconstructionError(
            "JAXSEDFit posterior and prediction lack required m2500 sites: "
            f"{missing}."
        )
    return combined


def _intrinsic_disk_luminosity_lambda(samples, wavelength_angstrom):
    """Evaluate the unattenuated JAXSEDFit bent-disk ``L_lambda`` exactly."""
    log_agn_amp = _sample_draws(samples, "log_agn_amp")
    pl_slope = _sample_draws(samples, "pl_slope")
    pl_bend_loc = _sample_draws(samples, "pl_bend_loc", 1000.0)
    pl_bend_width = _sample_draws(samples, "pl_bend_width", 10.0)
    uv_slope = _sample_draws(samples, "uv_slope", 0.0)
    pl_cutoff = _sample_draws(samples, "pl_cutoff", 100_000.0)
    (
        log_agn_amp,
        pl_slope,
        pl_bend_loc,
        pl_bend_width,
        uv_slope,
        pl_cutoff,
    ) = _broadcast_draws(
        log_agn_amp,
        pl_slope,
        pl_bend_loc,
        pl_bend_width,
        uv_slope,
        pl_cutoff,
    )

    wave = float(wavelength_angstrom)
    if not np.isfinite(wave) or wave <= 0.0:
        raise ValueError("Disk wavelength must be positive and finite.")
    pivot = 5100.0
    norm = np.exp(log_agn_amp) / pivot
    width = np.maximum(pl_bend_width, 1e-6)
    exponent = 1.0 / width
    mean_slope = (uv_slope + pl_slope + 2.0) / 2.0
    slope_change = (pl_slope - uv_slope) * width / 2.0
    pivot_ratio = pivot / pl_bend_loc
    divisor = 1.0 / (pivot_ratio**exponent + pivot_ratio**-exponent)
    wave_ratio = wave / pl_bend_loc
    luminosity_lambda = (
        norm
        * (wave / pivot) ** mean_slope
        * ((wave_ratio**exponent + wave_ratio**-exponent) * divisor) ** slope_change
        * (pivot / wave)
    )
    cutoff_factor = -np.expm1(-np.maximum(pl_cutoff, 0.0) / wave)
    return np.where(pl_cutoff > 0.0, luminosity_lambda * cutoff_factor, luminosity_lambda)


def _intrinsic_disk_luminosity_lambda_2500(samples):
    """Compatibility wrapper for the exact 2500-Angstrom disk continuum."""

    return _intrinsic_disk_luminosity_lambda(
        samples, ALPHA_NU_RED_WAVELENGTH_ANGSTROM
    )


def estimate_m2500_dereddened(samples, redshift, *, h0=70.0, om0=0.3):
    """Return intrinsic monochromatic AGN m2500 draws and attenuation diagnostics.

    This is the rest-frame monochromatic apparent AB magnitude at exactly
    2500 Angstrom, not a filter-integrated magnitude. It is built from the
    unattenuated accretion-disk luminosity, excluding host starlight, lines,
    Fe II, Balmer continuum, and torus emission. Inputs were already corrected
    for Milky-Way extinction when ``Observation.apply_mw_deredden`` is enabled.
    The returned attenuation includes both JAXSEDFit's foreground host-galaxy
    ``ebv_gal`` and nuclear ``ebv_agn`` terms.
    """
    from astropy.cosmology import FlatLambdaCDM

    luminosity_lambda = _intrinsic_disk_luminosity_lambda_2500(samples)
    luminosity_nu = luminosity_lambda * 2500.0**2 / C_ANGSTROM_PER_SECOND
    distance_m = (
        FlatLambdaCDM(H0=float(h0), Om0=float(om0))
        .luminosity_distance(float(redshift))
        .value
        * METER_PER_MEGAPARSEC
    )
    # As in fit_spectra.py, this deliberately uses the rest-frame spectral
    # convention
    #
    #   f_nu,restconv(2500) = L_nu(2500)/(4*pi*D_L^2)
    #                         = f_nu,obs[2500*(1+z)]/(1+z).
    #
    # Consequently m_2500_dereddened is the K-corrected MONOCHROMATIC
    # rest-frame apparent AB magnitude satisfying
    #
    #   m_2500_dereddened = M_2500_dereddened + distance modulus.
    #
    # It is not a filter-integrated magnitude and is not the directly observed
    # AB magnitude at 2500*(1+z), which would be brighter by
    # 2.5*log10(1+z).
    flux_nu_mjy = luminosity_nu / (4.0 * np.pi * distance_m**2) / 1e-29
    intrinsic_mag = -2.5 * np.log10(flux_nu_mjy / AB_ZEROPOINT_MJY)

    ebv_gal = _sample_draws(samples, "ebv_gal")
    ebv_agn = _sample_draws(samples, "ebv_agn")
    ebv_gal, ebv_agn, intrinsic_mag = _broadcast_draws(
        ebv_gal, ebv_agn, intrinsic_mag
    )
    # Match JAXSEDFit's native GRAHSP bi-attenuation law exactly.  Its
    # optical branch is norm * (wave / break)^index with norm=1.2.
    curve_2500 = GRAHSP_ATTENUATION_NORMALIZATION * (
        2500.0 / GRAHSP_ATTENUATION_BREAK_ANGSTROM
    ) ** GRAHSP_ATTENUATION_OPTICAL_INDEX
    attenuation_gal = ebv_gal * curve_2500
    attenuation_agn = ebv_agn * curve_2500
    attenuation_total = attenuation_gal + attenuation_agn
    return _validate_m2500_draws(
        {
            "m_2500_dereddened_draws": intrinsic_mag,
            "m_2500_attenuated_model_draws": (
                intrinsic_mag + attenuation_total
            ),
            "a_2500_galaxy_draws": attenuation_gal,
            "a_2500_internal_draws": attenuation_agn,
            "a_2500_total_draws": attenuation_total,
        }
    )


def summarize_m2500_dereddened(samples, redshift, *, h0=70.0, om0=0.3):
    draws = estimate_m2500_dereddened(
        samples, redshift, h0=h0, om0=om0
    )
    out = {}
    for draw_name, values in draws.items():
        name = draw_name.removesuffix("_draws")
        median, err, err_lower, err_upper = legacy.sym_percentile(values)
        out[name] = float(median)
        out[f"{name}_err"] = float(err)
        out[f"{name}_err_lower"] = float(err_lower)
        out[f"{name}_err_upper"] = float(err_upper)
    return out


def estimate_alpha_nu_1450_2500(samples, *, m2500_draws=None):
    """Return intrinsic and attenuated disk-only UV secant slopes.

    The convention is ``L_nu proportional to nu**alpha_nu``.  Both slopes are
    calculated from the same bent JAXSEDFit accretion-disk continuum at exactly
    1450 and 2500 Angstrom.  Host light, dust emission, torus, Fe II, Balmer
    continuum, and emission lines are deliberately excluded.  The attenuated
    slope applies the model's foreground-host plus nuclear GRAHSP screens; it
    does not reapply Milky-Way extinction to already corrected input data.
    """

    blue_wave = ALPHA_NU_BLUE_WAVELENGTH_ANGSTROM
    red_wave = ALPHA_NU_RED_WAVELENGTH_ANGSTROM
    blue_lambda = _intrinsic_disk_luminosity_lambda(samples, blue_wave)
    red_lambda = _intrinsic_disk_luminosity_lambda(samples, red_wave)
    blue_nu = np.asarray(blue_lambda, dtype=float) * blue_wave**2
    red_nu = np.asarray(red_lambda, dtype=float) * red_wave**2
    if (
        blue_nu.size == 0
        or blue_nu.shape != red_nu.shape
        or not np.all(np.isfinite(blue_nu))
        or not np.all(np.isfinite(red_nu))
        or np.any(blue_nu <= 0.0)
        or np.any(red_nu <= 0.0)
    ):
        raise M2500ReconstructionError(
            "UV alpha_nu reconstruction produced invalid intrinsic disk draws."
        )

    denominator = np.log10(red_wave / blue_wave)
    intrinsic = np.log10(blue_nu / red_nu) / denominator
    if m2500_draws is None:
        ebv_gal = _sample_draws(samples, "ebv_gal")
        ebv_agn = _sample_draws(samples, "ebv_agn")
        ebv_gal, ebv_agn, intrinsic = _broadcast_draws(
            ebv_gal, ebv_agn, intrinsic
        )
        curve_2500 = GRAHSP_ATTENUATION_NORMALIZATION * (
            red_wave / GRAHSP_ATTENUATION_BREAK_ANGSTROM
        ) ** GRAHSP_ATTENUATION_OPTICAL_INDEX
        a_2500_total = (ebv_gal + ebv_agn) * curve_2500
    else:
        a_2500_total = np.asarray(
            m2500_draws["a_2500_total_draws"], dtype=float
        ).reshape(-1)
        intrinsic, a_2500_total = _broadcast_draws(intrinsic, a_2500_total)
    attenuation_ratio = (
        blue_wave / red_wave
    ) ** GRAHSP_ATTENUATION_OPTICAL_INDEX
    attenuated = (
        intrinsic
        - 0.4
        * a_2500_total
        * (attenuation_ratio - 1.0)
        / denominator
    )
    if not np.all(np.isfinite(intrinsic)) or not np.all(np.isfinite(attenuated)):
        raise M2500ReconstructionError(
            "UV alpha_nu reconstruction produced nonfinite posterior draws."
        )
    return {
        "alpha_nu_intrinsic_1450_2500_draws": np.asarray(intrinsic, dtype=float),
        "alpha_nu_attenuated_1450_2500_draws": np.asarray(attenuated, dtype=float),
    }


def estimate_joint_hubble_posterior_draws(
    samples,
    redshift,
    *,
    h0=70.0,
    om0=0.3,
):
    """Return every disk/magnitude/attenuation draw stored jointly in v3."""

    m2500 = estimate_m2500_dereddened(
        samples, redshift, h0=h0, om0=om0
    )
    alpha = estimate_alpha_nu_1450_2500(samples, m2500_draws=m2500)
    draws = {**m2500, **alpha}
    sizes = {np.asarray(value).size for value in draws.values()}
    if len(sizes) != 1 or not sizes or next(iter(sizes)) < 1:
        raise M2500ReconstructionError(
            "Joint Hubble posterior products have incompatible draw counts."
        )
    return draws


def summarize_joint_hubble_posterior_draws(draws):
    """Summarize full posterior draws while retaining compact-draw covariance."""

    out = {}
    for draw_name, values in draws.items():
        name = draw_name.removesuffix("_draws")
        median, err, err_lower, err_upper = legacy.sym_percentile(values)
        out[name] = float(median)
        out[f"{name}_err"] = float(err)
        out[f"{name}_err_lower"] = float(err_lower)
        out[f"{name}_err_upper"] = float(err_upper)
    return out


def summarize_alpha_nu_1450_2500(samples):
    """Return scalar summaries for the two explicit UV slope definitions."""

    return summarize_joint_hubble_posterior_draws(
        estimate_alpha_nu_1450_2500(samples)
    )


def empty_hubble_convergence_summary():
    """Return the stable convergence schema used by the Hubble workflow."""

    return {
        f"{name}_rhat": np.nan
        for name in DERIVED_SPECTRAL_CONVERGENCE_SITES
    }


def _reshape_flat_samples_by_chain(samples, num_chains):
    """Reconstruct NumPyro's chain-major grouping from flattened draws."""

    num_chains = int(num_chains)
    if num_chains < 1 or not samples:
        raise ValueError("A positive chain count and posterior samples are required.")
    grouped = {}
    draw_count = None
    for name, value in samples.items():
        arr = np.asarray(value)
        if arr.ndim == 0 or arr.shape[0] % num_chains:
            raise ValueError(
                f"Cannot reconstruct {num_chains} chains for posterior site {name!r} "
                f"with shape {arr.shape}."
            )
        site_draw_count = arr.shape[0] // num_chains
        if draw_count is None:
            draw_count = site_draw_count
        elif site_draw_count != draw_count:
            raise ValueError("Posterior sites have inconsistent draw counts.")
        grouped[name] = arr.reshape((num_chains, site_draw_count) + arr.shape[1:])
    return grouped


def _fresh_grouped_nuts_samples(fit_result, prediction=None):
    """Return scientific, chain-grouped draws from a fresh JAXSEDFit result."""

    fitter = getattr(fit_result, "fitter", None)
    nuts_result = getattr(fitter, "nuts_result", None)
    mcmc = nuts_result.get("mcmc") if isinstance(nuts_result, dict) else None
    if mcmc is None:
        raise ValueError("Fresh fit does not expose a NumPyro MCMC result.")
    grouped = mcmc.get_samples(group_by_chain=True)
    scientific_samples = getattr(fit_result, "samples", None) or {}
    if prediction is not None:
        scientific_samples = posterior_samples_for_m2500(
            scientific_samples,
            prediction,
        )
    scientific_names = set(scientific_samples)
    grouped_scientific = {
        name: np.asarray(value)
        for name, value in grouped.items()
        if name in scientific_names
    }
    if not grouped_scientific:
        raise ValueError("Fresh fit exposes no scientific NumPyro samples.")

    # NumPyro's MCMC object may omit deterministic transforms (for example,
    # ebv_gal derived from log_ebv_gal).  JAXSEDFit retains those transforms in
    # its flattened scientific samples, whose ordering is chain-major.
    chain_count = np.asarray(next(iter(grouped_scientific.values()))).shape[0]
    flat_grouped = _reshape_flat_samples_by_chain(scientific_samples, chain_count)
    for name, value in flat_grouped.items():
        grouped_scientific.setdefault(name, value)
    return grouped_scientific


def summarize_m2500_convergence(
    grouped_samples,
    redshift,
    *,
    h0=70.0,
    om0=0.3,
    heading=None,
):
    """Compute the legacy m2500-only view of spectral convergence fields."""

    convergence = summarize_spectral_convergence(
        grouped_samples,
        redshift,
        h0=h0,
        om0=om0,
        heading=heading,
    )
    return {
        f"{name}_rhat": convergence[f"{name}_rhat"]
        for name in HUBBLE_MAGNITUDE_SITES
    }


def _scalar_grouped_samples(grouped_samples):
    """Return chain-grouped posterior sites representing scalar quantities."""

    scalar_samples = {}
    chain_shape = None
    for name, value in (grouped_samples or {}).items():
        arr = np.asarray(value)
        if arr.ndim < 2:
            continue
        if chain_shape is None:
            chain_shape = arr.shape[:2]
        if arr.shape[:2] != chain_shape:
            continue
        if np.prod(arr.shape[2:], dtype=int) != 1:
            continue
        scalar_samples[str(name)] = arr.reshape(chain_shape)
    return scalar_samples, chain_shape


def summarize_spectral_convergence(
    grouped_samples,
    redshift,
    *,
    h0=70.0,
    om0=0.3,
    heading=None,
    print_summary=True,
):
    """Save R-hat for every scalar spectral parameter and m2500 draw.

    Scalar sites are discovered from the posterior instead of maintained as a
    hand-picked list.  This includes both sampled and deterministic scalar
    quantities such as ``ebv_gal``, ``ebv_agn``, their log parameterizations,
    ``pl_slope``, continuum normalization, host, dust, calibration, and
    systematics parameters.  Array-valued line/model sites are excluded
    because they do not have an unambiguous flat scalar catalog column.
    """

    if not grouped_samples:
        return empty_hubble_convergence_summary()
    summary_samples, chain_shape = _scalar_grouped_samples(grouped_samples)
    if chain_shape is None:
        return empty_hubble_convergence_summary()
    derived = estimate_joint_hubble_posterior_draws(
        grouped_samples,
        redshift,
        h0=h0,
        om0=om0,
    )
    summary_samples.update(
        {
            name: np.asarray(derived[f"{name}_draws"], dtype=float).reshape(
                chain_shape
            )
            for name in DERIVED_SPECTRAL_CONVERGENCE_SITES
        }
    )
    summary_dict = compute_numpyro_summary(
        summary_samples,
        group_by_chain=True,
        prob=0.90,
    )
    display_summary = {
        name: summary_dict[name]
        for name in SPECTRAL_CONVERGENCE_DISPLAY_SITES
        if name in summary_dict
    }
    if print_summary:
        print_numpyro_summary_dict(display_summary, heading=heading)
    fields = convergence_fields(
        summary_dict,
        {name: name for name in summary_samples},
    )
    return {name: value for name, value in fields.items() if name.endswith("_rhat")}


def load_saved_sed_photometry(
    path,
    *,
    require_positive_flux=False,
    reject_invalid=False,
):
    """Load normalized long-form SED photometry from a saved table.

    Required columns are an object identifier, ``filter_name``, ``flux_mjy``,
    and ``flux_err_mjy``. CSV, Parquet, Feather, ECSV, and FITS tables are
    supported. Optional ``is_upper_limit``, ``psf_fwhm_arcsec``, and
    ``photometry_method`` columns are preserved.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        phot = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        phot = pd.read_parquet(path)
    elif suffix in {".feather", ".arrow"}:
        phot = pd.read_feather(path)
    elif suffix in {".ecsv", ".fits", ".fit", ".fz"}:
        from astropy.table import Table

        phot = Table.read(path).to_pandas()
    else:
        raise ValueError(
            f"Unsupported saved SED format {suffix!r}. Use CSV, Parquet, "
            "Feather, ECSV, or FITS."
        )

    aliases = {
        "object_id": "source_id",
        "objectId": "source_id",
        "filter": "filter_name",
        "flux": "flux_mjy",
        "flux_err": "flux_err_mjy",
    }
    for old, new in aliases.items():
        if new not in phot and old in phot:
            phot[new] = phot[old]
    required = {"source_id", "filter_name", "flux_mjy", "flux_err_mjy"}
    missing = required - set(phot)
    if missing:
        raise ValueError(f"{path} lacks saved-SED columns: {sorted(missing)}")

    phot = phot.copy()
    phot["source_id"] = phot["source_id"].map(legacy.normalize_object_id)
    phot["filter_name"] = phot["filter_name"].astype(str).str.strip()
    for col in ("flux_mjy", "flux_err_mjy"):
        phot[col] = pd.to_numeric(phot[col], errors="coerce")
    if "is_upper_limit" not in phot:
        phot["is_upper_limit"] = False
    else:
        values = phot["is_upper_limit"]
        if values.dtype == object:
            phot["is_upper_limit"] = values.astype(str).str.lower().isin(
                {"1", "true", "t", "yes", "y"}
            )
        else:
            phot["is_upper_limit"] = values.astype(bool)
    good = (
        (phot["source_id"] != "")
        & (phot["filter_name"] != "")
        & np.isfinite(phot["flux_mjy"])
        & np.isfinite(phot["flux_err_mjy"])
        & (phot["flux_err_mjy"] > 0)
    )
    if require_positive_flux:
        good &= phot["flux_mjy"] > 0
    if reject_invalid and not bool(good.all()):
        invalid_count = int((~good).sum())
        raise ValueError(
            f"{path} contains {invalid_count} invalid photometry row(s); "
            "object IDs and filters must be nonempty, and fluxes and errors "
            "must be finite and strictly positive."
        )
    return phot.loc[good].copy()


def load_sdss_psf_photometry_overrides(path):
    """Load strict, object-keyed SDSS ugriz flux overrides for experiments."""
    phot = load_saved_sed_photometry(
        path,
        require_positive_flux=True,
        reject_invalid=True,
    )
    expected_filters = {f"{band}_sdss" for band in legacy.SDSS_BANDS}
    unexpected = sorted(set(phot["filter_name"]) - expected_filters)
    if unexpected:
        raise ValueError(
            f"{path} contains non-SDSS override filters: {unexpected}."
        )
    duplicates = phot.duplicated(["source_id", "filter_name"], keep=False)
    if bool(duplicates.any()):
        duplicate_rows = phot.loc[duplicates, ["source_id", "filter_name"]]
        raise ValueError(
            f"{path} contains duplicate SDSS override rows: "
            f"{duplicate_rows.to_dict(orient='records')[:10]}."
        )
    for source_id, rows in phot.groupby("source_id", sort=False):
        actual_filters = set(rows["filter_name"])
        if actual_filters != expected_filters:
            missing = sorted(expected_filters - actual_filters)
            raise ValueError(
                f"{path} must contain exactly one ugriz SDSS override for "
                f"object_id={source_id}; missing filters: {missing}."
            )
    phot = phot.copy()
    phot["catalog"] = "sdss_dr16_notebook"
    phot["band"] = phot["filter_name"].str.removesuffix("_sdss")
    phot["photometry_method"] = "psf"
    phot["psf_fwhm_arcsec"] = phot["band"].map(SDSS_STATIC_PSF_FWHM_ARCSEC)
    phot["is_upper_limit"] = False
    return phot


def load_sdss_spectrum_selection_overrides(path):
    """Load strict per-object SDSS spectrum selections for experiments."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        frame = pd.read_csv(path)
    elif suffix == ".ecsv":
        from astropy.table import Table

        frame = Table.read(path).to_pandas()
    else:
        raise ValueError(
            f"Unsupported SDSS spectrum selection format {suffix!r}; "
            "use CSV or ECSV."
        )

    aliases = {"source_id": "object_id", "fiberid": "fiber"}
    for old, new in aliases.items():
        if new not in frame and old in frame:
            frame[new] = frame[old]
    required = {"object_id", "plate", "mjd", "fiber", "z"}
    missing = required - set(frame)
    if missing:
        raise ValueError(
            f"{path} lacks SDSS spectrum selection columns: {sorted(missing)}"
        )

    frame = frame.loc[:, sorted(required)].copy()
    frame["object_id"] = frame["object_id"].map(legacy.normalize_object_id)
    if (frame["object_id"] == "").any():
        raise ValueError(f"{path} contains an empty object_id.")
    duplicates = frame["object_id"].duplicated(keep=False)
    if bool(duplicates.any()):
        duplicate_ids = frame.loc[duplicates, "object_id"].unique().tolist()
        raise ValueError(
            f"{path} contains duplicate SDSS spectrum selections for "
            f"object IDs: {duplicate_ids[:10]}."
        )
    for column in ("plate", "mjd", "fiber", "z"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    numeric = frame[["plate", "mjd", "fiber", "z"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{path} contains non-finite SDSS spectrum metadata.")
    for column in ("plate", "mjd", "fiber"):
        values = frame[column].to_numpy(dtype=float)
        if np.any(values <= 0) or np.any(values != np.floor(values)):
            raise ValueError(
                f"{path} requires positive integer values for {column}."
            )
        frame[column] = values.astype(int)
    if np.any(frame["z"].to_numpy(dtype=float) <= 0):
        raise ValueError(f"{path} requires a strictly positive spectrum redshift.")
    return frame


def add_qvc_psf_photometry(rec, phot):
    """Add selected ugriz fluxes with static, typical SDSS PSF FWHMs."""
    phot = phot.copy()
    if "filter_name" in phot:
        sdss_names = {f"{band}_sdss" for band in legacy.SDSS_BANDS}
        phot = phot.loc[~phot["filter_name"].astype(str).isin(sdss_names)]

    override_rows = rec.get("_joint_sdss_psf_photometry")
    if override_rows is not None:
        rows = [dict(row) for row in override_rows]
    else:
        rows = []
        missing = []
        for band in legacy.SDSS_BANDS:
            mag = legacy.safe_float(rec.get(f"psf_mag_{band}"))
            mag_err = legacy.safe_float(rec.get(f"psf_mag_err_{band}"))
            if not (np.isfinite(mag) and np.isfinite(mag_err) and mag_err > 0):
                missing.append(band)
                continue
            rows.append(
                {
                    "source_id": str(rec["object_id"]),
                    "catalog": "qvc_light_curve",
                    "band": band,
                    "filter_name": f"{band}_sdss",
                    "flux_mjy": float(ab_mag_to_mjy(mag)),
                    "flux_err_mjy": float(ab_mag_err_to_mjy_err(mag, mag_err)),
                    "psf_fwhm_arcsec": SDSS_STATIC_PSF_FWHM_ARCSEC[band],
                    "photometry_method": "psf",
                    "is_upper_limit": False,
                }
            )
        if missing:
            raise ValueError(
                "Joint SED fitting requires finite QVC light-curve PSF magnitudes "
                f"and positive errors in all ugriz bands; missing/invalid: {missing}. "
                "The z-band requirement applies even when z was excluded from the "
                "variability fit."
            )
    if rows:
        phot = pd.concat([phot, pd.DataFrame(rows)], ignore_index=True, sort=False)
    limit_values = phot.get(
        "is_upper_limit", pd.Series(False, index=phot.index)
    ).to_numpy()
    phot["is_upper_limit"] = np.where(pd.isna(limit_values), False, limit_values).astype(bool)
    return phot


def build_joint_config(
    rec,
    phot,
    lam,
    flux,
    err,
    resolving_power,
    args,
    *,
    aperture_diameter_arcsec,
):
    """Construct one validated jaxsedfit joint SED+spectrum configuration."""
    from jaxsedfit import (
        AGNConfig,
        FilterSet,
        FitConfig,
        GalaxyConfig,
        InferenceConfig,
        JAXSEDFit,
        LikelihoodConfig,
        Observation,
        OutputConfig,
        PhotometryData,
        SpectroscopyData,
    )
    from jaxsedfit.filters import load_filter_curves

    del JAXSEDFit  # imported here as an early API/version check
    phot = add_qvc_psf_photometry(rec, phot)
    if len(phot) == 0:
        raise RuntimeError("No usable broadband photometry is available.")

    filter_names = phot["filter_name"].astype(str).tolist()
    spec_flux_mjy = flambda_1e17_to_mjy(lam, flux)
    spec_err_mjy = flambda_1e17_to_mjy(lam, err)
    wave_rf = np.asarray(lam) / (1.0 + float(rec["z"]))
    spec_good = (
        np.isfinite(spec_flux_mjy)
        & np.isfinite(spec_err_mjy)
        & (spec_err_mjy > 0)
        & (wave_rf >= args.wave_min)
        & (wave_rf <= args.wave_max)
    )
    if not np.any(spec_good):
        raise RuntimeError("No spectral pixels remain inside the rest wavelength range.")

    psf_fwhm = [
        None if not np.isfinite(legacy.safe_float(value)) else float(value)
        for value in phot.get("psf_fwhm_arcsec", pd.Series(np.nan, index=phot.index))
    ]
    method_values = phot.get(
        "photometry_method", pd.Series("catalog", index=phot.index)
    )
    methods = [None if pd.isna(value) else str(value) for value in method_values]
    config = FitConfig(
        observation=Observation(
            object_id=joint_saved_name(rec),
            redshift=float(rec["z"]),
            redshift_mode="fixed",
            ra=float(rec["ra"]),
            dec=float(rec["dec"]),
            apply_mw_deredden=not args.no_deredden,
        ),
        photometry=PhotometryData(
            filter_names=filter_names,
            fluxes=phot["flux_mjy"].astype(float).tolist(),
            errors=phot["flux_err_mjy"].astype(float).tolist(),
            is_upper_limit=phot["is_upper_limit"].astype(bool).tolist(),
            psf_fwhm_arcsec=psf_fwhm,
            photometry_method=methods,
        ),
        filters=FilterSet(curves=load_filter_curves(filter_names)),
        spectroscopy=SpectroscopyData(
            wave_obs=np.asarray(lam)[spec_good].astype(float).tolist(),
            fluxes=spec_flux_mjy[spec_good].astype(float).tolist(),
            errors=spec_err_mjy[spec_good].astype(float).tolist(),
            mask=[True] * int(np.sum(spec_good)),
            instrument="SDSS",
            aperture_diameter_arcsec=float(aperture_diameter_arcsec),
            epoch_mjd=float(rec["mjd"]),
            resolving_power=float(resolving_power),
        ),
        galaxy=GalaxyConfig(
            dsps_ssp_fn=args.dsps_ssp_fn,
            n_wave=args.sed_n_wave,
            rest_wave_min=100.0,
            rest_wave_max=3.0e6,
            fit_host_kinematics=True,
        ),
        agn=AGNConfig(
            agn_type=1,
            fit_lines=args.fit_lines,
            tied_lines=args.fit_lines,
            use_smart_line_priors=True,
            fit_feii=args.fit_fe,
            fit_balmer_continuum=args.fit_bc,
            line_flux_scale_mjy=args.line_flux_scale_mjy,
        ),
        likelihood=LikelihoodConfig(
            use_host_capture_model=True,
            systematics_width=args.photometry_systematics,
            spectrum_systematics_width=args.spectrum_systematics,
            spectrum_student_t_df=args.spectrum_student_t_df,
            spectrum_weight_mode="resolution_elements",
            fit_spectrum_scale=True,
            spectrum_scale_prior_sigma_dex=args.spectrum_scale_prior_sigma_dex,
        ),
        inference=InferenceConfig(
            method=args.fit_method,
            map_steps=args.optax_steps,
            learning_rate=args.optax_lr,
            plot_init=bool(getattr(args, "plot_init", False)),
            seed=args.seed,
            num_warmup=args.nuts_warmup,
            num_samples=args.nuts_samples,
            num_chains=args.nuts_chains,
            target_accept_prob=args.nuts_target_accept,
            dense_mass=args.dense_mass,
        ),
        output=OutputConfig(
            output_dir=str(args.output_dir),
            fig_path=str(sed_figure_path(args.fig_dir, rec)),
            plot_fig=False,
            save_fig=args.save_fig,
            save_result=args.save_jaxsedfit_samples,
            show_plot=False,
        ),
    )
    config.galaxy.host_sfh_model = "delayed_burst"
    config.prior_config.host.log_sfh_burst_age_gyr = dist.Uniform(
        np.log(0.01), np.log(0.5)
    )
    config.prior_config.host.log_sfh_burst_tau_gyr = dist.Uniform(
        np.log(0.01), np.log(0.2)
    )
    config.validate()
    return config, phot


def summarize_samples(samples):
    """Flatten scalar posterior sites into median/error catalog columns."""
    out = {}
    for name, value in (samples or {}).items():
        arr = np.asarray(value)
        if arr.size == 0:
            continue
        # Only scalar-valued sites have an unambiguous flat-table contract.
        if arr.ndim > 1 and np.prod(arr.shape[1:]) != 1:
            continue
        draws = arr.reshape(arr.shape[0], -1)[:, 0] if arr.ndim else arr.reshape(1)
        if not np.all(np.isfinite(draws)):
            continue
        median, err, _, _ = legacy.sym_percentile(draws)
        out[str(name)] = float(median)
        out[f"{name}_err"] = float(err)
    return out


def predict_catalog_posterior(fitter, *, kind, **prediction_kwargs):
    """Predict standard products plus every scalar needed by v3 derivations.

    JAXSEDFit's public prediction products intentionally omit some scalar
    deterministics that were present in the original in-memory fit samples.
    Request those sites through the public API so QVC can reproduce legacy CSV
    columns and exact bent-disk physical parameters from compact resume bundles,
    including log-parameterized configurations.
    """
    return fitter.predict(
        kind=kind,
        extra_return_sites=LEGACY_CSV_SCALAR_PREDICTION_SITES,
        required_return_sites=M2500_POSTERIOR_SITES,
        **prediction_kwargs,
    )


def add_sdss_psf_host_fraction_prediction(
    prediction,
    *,
    rest_wavelength=HOST_FRACTION_REST_WAVELENGTH_ANGSTROM,
    psf_fwhm_arcsec=SDSS_TYPICAL_PSF_FWHM_ARCSEC,
):
    """Add posterior PSF host fractions using main's fitted host scale.

    The typical SDSS FWHM is used only for this derived prediction. It is not
    supplied to the fitted photometry and therefore cannot change the
    likelihood or sampled latent sites.
    """
    required = (
        "rest_wave",
        "host_total_rest_sed",
        "dust_rest_sed",
        "agn_rest_sed",
        "log_host_capture_scale_arcsec_fit",
    )
    missing = [name for name in required if name not in prediction]
    if missing:
        raise ValueError(
            "JAXSEDFit full prediction lacks sites required for the direct "
            f"PSF host fraction: {missing}."
        )

    host_total = np.asarray(prediction["host_total_rest_sed"], dtype=float)
    dust = np.asarray(prediction["dust_rest_sed"], dtype=float)
    agn = np.asarray(prediction["agn_rest_sed"], dtype=float)
    if host_total.ndim != 2 or dust.shape != host_total.shape or agn.shape != host_total.shape:
        raise ValueError(
            "Rest-frame component predictions must be matching (draw, wavelength) arrays."
        )
    n_draws, n_wave = host_total.shape
    wave = np.asarray(prediction["rest_wave"], dtype=float)
    if wave.ndim == 1:
        if wave.size != n_wave:
            raise ValueError("rest_wave does not match the component wavelength axis.")
        wave = np.broadcast_to(wave, host_total.shape)
    elif wave.shape != host_total.shape:
        raise ValueError(
            "rest_wave must be one-dimensional or match the component arrays."
        )
    if not np.isfinite(rest_wavelength) or float(rest_wavelength) <= 0.0:
        raise ValueError("The requested rest wavelength must be positive and finite.")
    if not np.isfinite(psf_fwhm_arcsec) or float(psf_fwhm_arcsec) <= 0.0:
        raise ValueError("The adopted PSF FWHM must be positive and finite.")

    host_2500 = np.empty(n_draws, dtype=float)
    agn_2500 = np.empty(n_draws, dtype=float)
    for draw_index in range(n_draws):
        draw_wave = wave[draw_index]
        if (
            not np.all(np.isfinite(draw_wave))
            or np.any(np.diff(draw_wave) <= 0.0)
            or float(rest_wavelength) < draw_wave[0]
            or float(rest_wavelength) > draw_wave[-1]
        ):
            raise ValueError(
                "rest_wave must be finite, strictly increasing, and span the "
                f"requested {float(rest_wavelength):g} Angstrom wavelength."
            )
        host_2500[draw_index] = np.interp(
            rest_wavelength,
            draw_wave,
            host_total[draw_index] + dust[draw_index],
        )
        agn_2500[draw_index] = np.interp(
            rest_wavelength,
            draw_wave,
            agn[draw_index],
        )

    log_scale = np.asarray(
        prediction["log_host_capture_scale_arcsec_fit"], dtype=float
    )
    if log_scale.ndim == 0:
        log_scale = np.full(n_draws, float(log_scale), dtype=float)
    elif log_scale.shape[0] == n_draws and np.prod(log_scale.shape[1:]) == 1:
        log_scale = log_scale.reshape(n_draws)
    else:
        raise ValueError(
            "log_host_capture_scale_arcsec_fit must provide one value per draw."
        )
    effective_radius = (
        np.sqrt(2.0) * float(psf_fwhm_arcsec) / 2.354820045
    )
    host_scale = np.exp(log_scale)
    capture = effective_radius**2 / (effective_radius**2 + host_scale**2)
    captured_host = capture * host_2500
    denominator = agn_2500 + captured_host
    fractions = np.where(denominator > 0.0, captured_host / denominator, np.nan)
    if not np.all(np.isfinite(fractions)) or np.any((fractions < 0.0) | (fractions > 1.0)):
        raise ValueError("Derived f_host_2500_psf draws are not finite within [0, 1].")

    result = dict(prediction)
    result["component_host_fraction"] = fractions[:, None]
    result["component_host_capture_fraction"] = capture[:, None]
    return result


def summarize_catalog_posterior(samples, prediction):
    """Summarize scalar latent and deterministic posterior sites.

    Resume-ready JAXSEDFit bundles intentionally persist only the latent sites
    needed to reproduce a fit.  Derived sites such as ``fracAGN_5100_fit`` are
    regenerated by :meth:`JAXSEDFit.predict`, so catalog construction must
    summarize both mappings to preserve the flat-table schema written by the
    original CSV pipeline.
    """
    missing = [name for name in ("fracAGN_5100_fit",) if name not in prediction]
    if missing:
        raise ValueError(
            "JAXSEDFit prediction lacks required scalar catalog sites: "
            f"{missing}"
        )

    out = summarize_samples(samples)
    # Prefer regenerated prediction sites when a legacy bundle contains an
    # older cached copy of the same deterministic quantity.
    out.update(summarize_samples(prediction))
    return out


def summarize_joint_chi2(prediction):
    """Summarize the required joint-fit chi-square diagnostic sites."""
    missing = [name for name in JOINT_CHI2_SITES if name not in prediction]
    if missing:
        raise ValueError(
            "JAXSEDFit prediction lacks required chi-square sites: "
            f"{missing}"
        )
    return summarize_samples({name: prediction[name] for name in JOINT_CHI2_SITES})


def empty_joint_chi2_summary():
    """Return a stable value/error schema for joint-fit chi-square fields."""
    return {
        key: np.nan
        for name in JOINT_CHI2_SITES
        for key in (name, f"{name}_err")
    }


def empty_psf_agn_fraction_summary(bands=PSF_AGN_FRACTION_BANDS):
    """Return the stable per-band schema for joint-fit PSF AGN fractions."""
    return {
        key: np.nan
        for band in bands
        for key in (f"f_AGN_psf_{band}", f"f_AGN_psf_{band}_err")
    }


def empty_host_2500_psf_summary():
    """Return the stable direct 2500-Angstrom host-fraction schema."""
    return {
        "f_host_2500_psf": np.nan,
        "f_host_2500_psf_err": np.nan,
        "f_host_2500_psf_err_lower": np.nan,
        "f_host_2500_psf_err_upper": np.nan,
    }


def empty_alpha_nu_1450_2500_summary():
    """Return the stable scalar schema for both explicit UV slopes."""

    return {
        key: np.nan
        for name in ALPHA_NU_CATALOG_SITES
        for key in (
            name,
            f"{name}_err",
            f"{name}_err_lower",
            f"{name}_err_upper",
        )
    }


def empty_joint_hubble_posterior_summary():
    """Return the stable scalar schema paired with the v3 draw fields."""

    return {
        key: np.nan
        for name in DERIVED_HUBBLE_CATALOG_SITES
        for key in (
            name,
            f"{name}_err",
            f"{name}_err_lower",
            f"{name}_err_upper",
        )
    }


def summarize_host_2500_psf(prediction):
    """Summarize direct posterior PSF host fractions at rest-frame 2500 A."""
    if "component_host_fraction" not in prediction:
        raise ValueError(
            "JAXSEDFit prediction lacks direct monochromatic component_host_fraction."
        )
    fractions = np.asarray(prediction["component_host_fraction"], dtype=float)
    if fractions.ndim != 2 or fractions.shape[1] != 1 or fractions.shape[0] < 1:
        raise ValueError(
            "Direct 2500-Angstrom host-fraction predictions must have shape "
            f"(draw, 1); received {fractions.shape}."
        )
    draws = fractions[:, 0]
    if not np.all(np.isfinite(draws)):
        raise ValueError(
            "Direct 2500-Angstrom host-fraction prediction contains nonfinite draws."
        )
    if np.any((draws < 0.0) | (draws > 1.0)):
        raise ValueError(
            "Direct 2500-Angstrom host-fraction prediction contains draws outside "
            "the physical interval [0, 1]."
        )
    median, err, err_lower, err_upper = legacy.sym_percentile(draws)
    if not (np.isfinite(median) and np.isfinite(err) and 0.0 <= median <= 1.0):
        raise ValueError("Direct 2500-Angstrom host-fraction summary is invalid.")
    return {
        "f_host_2500_psf": float(median),
        "f_host_2500_psf_err": float(err),
        "f_host_2500_psf_err_lower": float(err_lower),
        "f_host_2500_psf_err_upper": float(err_upper),
    }


def extract_compact_host_2500_psf_draws(
    prediction,
    *,
    object_id,
    seed,
    draw_count=F_HOST_2500_PSF_DRAW_COUNT,
):
    """Return deterministic compact draws of direct PSF host fraction at 2500 A."""

    fractions = np.asarray(prediction.get("component_host_fraction"), dtype=float)
    if fractions.ndim != 2 or fractions.shape[1] != 1 or fractions.shape[0] < 1:
        raise ValueError(
            "Direct 2500-Angstrom host-fraction predictions must have shape "
            f"(draw, 1); received {fractions.shape}."
        )
    draws = fractions[:, 0]
    if not np.all(np.isfinite(draws)):
        raise ValueError(
            "Direct 2500-Angstrom host-fraction prediction contains nonfinite draws."
        )
    if np.any((draws < 0.0) | (draws > 1.0)):
        raise ValueError(
            "Direct 2500-Angstrom host-fraction prediction contains draws outside "
            "the physical interval [0, 1]."
        )
    if len(draws) > draw_count:
        digest = hashlib.sha256(f"{int(seed)}:{object_id}".encode("utf-8")).digest()
        object_seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(object_seed)
        chosen = np.sort(rng.choice(len(draws), size=draw_count, replace=False))
        draws = draws[chosen]

    valid_count = min(len(draws), int(draw_count))
    compact = np.full(int(draw_count), np.nan, dtype=np.float32)
    compact[:valid_count] = draws[:valid_count].astype(np.float32)
    return compact, valid_count


def deterministic_compact_posterior_indices(
    source_draw_count,
    *,
    object_id,
    seed,
    draw_count=JOINT_POSTERIOR_DRAW_COUNT,
):
    """Choose reproducible original posterior indices without replacement."""

    source_draw_count = int(source_draw_count)
    draw_count = int(draw_count)
    if source_draw_count < 1 or draw_count < 1:
        raise ValueError("Positive source and compact posterior draw counts are required.")
    if source_draw_count <= draw_count:
        return np.arange(source_draw_count, dtype=np.int32)
    digest = hashlib.sha256(f"{int(seed)}:{object_id}".encode("utf-8")).digest()
    object_seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(object_seed)
    return np.sort(
        rng.choice(source_draw_count, size=draw_count, replace=False)
    ).astype(np.int32)


def extract_compact_joint_posterior_draws(
    prediction,
    derived_draws,
    *,
    object_id,
    seed,
    draw_count=JOINT_POSTERIOR_DRAW_COUNT,
):
    """Compact every Hubble quantity at one shared set of posterior indices."""

    fractions = np.asarray(prediction.get("component_host_fraction"), dtype=float)
    if fractions.ndim != 2 or fractions.shape[1] != 1 or fractions.shape[0] < 1:
        raise ValueError(
            "Direct 2500-Angstrom host-fraction predictions must have shape "
            f"(draw, 1); received {fractions.shape}."
        )
    source_draw_count = int(fractions.shape[0])
    full = {"f_host_2500_psf": fractions[:, 0]}
    for name in JOINT_POSTERIOR_DRAW_FIELDS:
        if name == "f_host_2500_psf":
            continue
        draw_name = f"{name}_draws"
        if draw_name not in derived_draws:
            raise ValueError(
                f"Joint posterior derivation lacks required field {draw_name!r}."
            )
        values = np.asarray(derived_draws[draw_name], dtype=float).reshape(-1)
        if values.size != source_draw_count:
            raise ValueError(
                f"Joint posterior field {draw_name!r} has {values.size} draws; "
                f"expected {source_draw_count}."
            )
        full[name] = values
    nonfinite = [name for name, values in full.items() if not np.all(np.isfinite(values))]
    if nonfinite:
        raise ValueError(
            f"Joint posterior products contain nonfinite full draws for {nonfinite}."
        )
    if np.any((full["f_host_2500_psf"] < 0.0) | (full["f_host_2500_psf"] > 1.0)):
        raise ValueError("Direct f_host_2500_psf posterior draws are outside [0, 1].")

    selected = deterministic_compact_posterior_indices(
        source_draw_count,
        object_id=object_id,
        seed=seed,
        draw_count=draw_count,
    )
    valid_count = int(selected.size)
    compact = {
        name: np.full(int(draw_count), np.nan, dtype=np.float32)
        for name in JOINT_POSTERIOR_DRAW_FIELDS
    }
    for name, values in full.items():
        compact[name][:valid_count] = values[selected].astype(np.float32)
    posterior_index = np.full(int(draw_count), -1, dtype=np.int32)
    posterior_index[:valid_count] = selected
    return compact, valid_count, posterior_index, source_draw_count


def extract_aligned_joint_psf_photometry_draws(
    prediction,
    filter_names,
    posterior_index,
    valid_count,
    *,
    bands=JOINT_PSF_PHOTOMETRY_BANDS,
    draw_count=JOINT_PSF_PHOTOMETRY_DRAW_COUNT,
):
    """Select fitted total-PSF ugriz fluxes at the authoritative v3 indices."""

    total = np.asarray(prediction.get("pred_fluxes"), dtype=float)
    if total.ndim != 2 or total.shape[0] < 1:
        raise ValueError(
            "JAXSEDFit pred_fluxes must have shape (posterior draw, filter)."
        )
    names = [str(name) for name in filter_names]
    if len(names) != total.shape[1]:
        raise ValueError(
            "JAXSEDFit filter-name count does not match pred_fluxes width: "
            f"{len(names)} != {total.shape[1]}."
        )
    filter_indices = []
    for band in bands:
        matches = [index for index, name in enumerate(names) if name == band]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {band!r} total-PSF prediction, "
                f"found {len(matches)}."
            )
        filter_indices.append(matches[0])

    count = int(valid_count)
    indices = np.asarray(posterior_index, dtype=np.int64)
    if indices.shape != (int(draw_count),):
        raise ValueError(
            f"posterior_index has shape {indices.shape}; expected {(int(draw_count),)}."
        )
    if count < 1 or count > int(draw_count):
        raise ValueError(
            f"valid_count must be between 1 and {int(draw_count)}."
        )
    selected_indices = indices[:count]
    if (
        np.any(selected_indices < 0)
        or np.any(selected_indices >= total.shape[0])
        or (count > 1 and np.any(np.diff(selected_indices) <= 0))
        or not np.all(indices[count:] == -1)
    ):
        raise ValueError(
            "posterior_index is not a valid strictly increasing, -1-padded "
            "selection from pred_fluxes."
        )
    selected = total[np.ix_(selected_indices, filter_indices)]
    if not np.all(np.isfinite(selected)) or np.any(selected <= 0.0):
        raise ValueError(
            "Selected fitted total-PSF ugriz fluxes must be finite and positive."
        )
    compact = np.full(
        (int(draw_count), len(bands)),
        np.nan,
        dtype=np.float32,
    )
    compact[:count] = selected.astype(np.float32)
    return compact


def joint_psf_photometry_prediction_provenance(prediction_source):
    """Capture the exact JAXSEDFit code state behind fitted photometry."""

    dependency = runtime_state().get("dependencies", {}).get("JAXSEDFit", {})
    git = dependency.get("git", {}) if isinstance(dependency, dict) else {}
    commit = str(git.get("commit", "")).strip() or "unavailable"
    return {
        "prediction_source": str(prediction_source),
        "jaxsedfit_git_commit": commit,
        "jaxsedfit_git_dirty": git.get("dirty"),
        "jaxsedfit_module_path": str(dependency.get("module_path", "")),
        "jaxsedfit_version": str(dependency.get("version", "")),
        "posterior_alignment": "joint_posterior_draws/posterior_index",
        "prediction_site": "pred_fluxes",
    }


def extract_compact_psf_agn_fraction_draws(
    prediction,
    filter_names,
    *,
    object_id,
    seed,
    bands=PSF_AGN_FRACTION_BANDS,
    draw_count=PSF_AGN_FRACTION_DRAW_COUNT,
):
    """Return deterministic, jointly indexed PSF AGN-fraction draws."""

    total = np.asarray(prediction["pred_fluxes"], dtype=float)
    variable = np.asarray(prediction["variable_agn_fluxes"], dtype=float)
    if total.ndim != 2 or variable.shape != total.shape:
        raise ValueError("PSF AGN-fraction predictions must be matching 2D arrays.")

    filter_names = [str(name) for name in filter_names]
    indices = []
    for band in bands:
        matches = [i for i, name in enumerate(filter_names) if name == f"{band}_sdss"]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {band + '_sdss'!r} prediction, found {len(matches)}."
            )
        indices.append(matches[0])

    selected_total = total[:, indices]
    selected_variable = variable[:, indices]
    valid = (
        np.all(np.isfinite(selected_total) & (selected_total > 0.0), axis=1)
        & np.all(np.isfinite(selected_variable), axis=1)
    )
    fractions = np.clip(selected_variable[valid] / selected_total[valid], 0.0, 1.0)
    if len(fractions) > draw_count:
        digest = hashlib.sha256(f"{int(seed)}:{object_id}".encode("utf-8")).digest()
        object_seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(object_seed)
        chosen = np.sort(rng.choice(len(fractions), size=draw_count, replace=False))
        fractions = fractions[chosen]

    valid_count = min(len(fractions), int(draw_count))
    compact = np.full((int(draw_count), len(bands)), np.nan, dtype=np.float32)
    compact[:valid_count] = fractions[:valid_count].astype(np.float32)
    return compact, valid_count


def validate_m2500_catalog_rows(rows):
    """Reject incomplete or physically inconsistent successful fit rows.

    Exact attenuation identities are checked on paired draws upstream.  This
    catalog-level check instead protects the stable fresh/resumed schema and
    catches the historical resume failure where positive EBV posteriors were
    silently replaced by zero while deriving m2500.
    """

    successful = [row for row in rows if bool(row.get("fit_ok", False))]
    for row in successful:
        missing = [name for name in DERIVED_HUBBLE_CATALOG_SITES if name not in row]
        if missing:
            raise M2500ReconstructionError(
                "Successful spectral fit "
                f"{row.get('object_id')!r} ({row.get('execution_mode')!r}) "
                f"lacks required m2500 catalog fields or alpha_nu fields: {missing}."
            )
        try:
            values = np.asarray(
                [row[name] for name in DERIVED_HUBBLE_CATALOG_SITES],
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            raise M2500ReconstructionError(
                "Successful spectral fit "
                f"{row.get('object_id')!r} has nonnumeric m2500/alpha_nu fields."
            ) from exc
        if not np.all(np.isfinite(values)):
            raise M2500ReconstructionError(
                "Successful spectral fit "
                f"{row.get('object_id')!r} has nonfinite m2500/alpha_nu fields."
            )
        attenuation_values = np.asarray(
            [row[name] for name in M2500_CATALOG_SITES[2:]], dtype=float
        )
        if (
            np.any(attenuation_values < -1e-12)
            or float(row["m_2500_attenuated_model"])
            < float(row["m_2500_dereddened"]) - 1e-12
        ):
            raise M2500ReconstructionError(
                "Successful spectral fit "
                f"{row.get('object_id')!r} has physically inconsistent "
                "m2500 attenuation summaries."
            )

    resumed = [
        row for row in successful if str(row.get("execution_mode")) == "resumed"
    ]
    if not resumed:
        return

    attenuation = np.asarray(
        [row["a_2500_total"] for row in resumed], dtype=float
    )
    magnitude_delta = np.asarray(
        [
            row["m_2500_attenuated_model"] - row["m_2500_dereddened"]
            for row in resumed
        ],
        dtype=float,
    )
    ebv = np.asarray(
        [
            [row.get("ebv_gal", np.nan), row.get("ebv_agn", np.nan)]
            for row in resumed
        ],
        dtype=float,
    )
    log_ebv = np.asarray(
        [
            [row.get("log_ebv_gal", np.nan), row.get("log_ebv_agn", np.nan)]
            for row in resumed
        ],
        dtype=float,
    )
    positive_ebv = (
        np.all(np.isfinite(ebv), axis=1) & (np.sum(ebv, axis=1) > 0.0)
    ) | np.any(np.isfinite(log_ebv), axis=1)
    broken = (
        positive_ebv
        & ((attenuation == 0.0) | (magnitude_delta == 0.0))
    )
    if np.any(broken):
        object_ids = [
            str(row.get("object_id"))
            for row, is_broken in zip(resumed, broken)
            if is_broken
        ]
        qualifier = "All" if np.all(broken) else "Some"
        raise M2500ReconstructionError(
            f"{qualifier} successful resumed fits have positive EBV "
            "posteriors but zero reconstructed A_2500 or identical "
            "attenuated/dereddened m2500; refusing to write the catalog "
            f"(object_ids={object_ids[:10]})."
        )


def write_joint_fit_results_hdf5(path, rows, *, provenance=None):
    """Write worker rows and their private, jointly indexed v3 payloads."""

    catalog_rows = []
    draws = []
    counts = []
    joint_draws = {name: [] for name in JOINT_POSTERIOR_DRAW_FIELDS}
    joint_counts = []
    joint_indices = []
    joint_source_counts = []
    joint_psf_photometry = []
    joint_psf_provenance = None
    selection_seeds = set()
    for row in rows:
        catalog_rows.append({key: value for key, value in row.items() if not key.startswith("_")})
        draws.append(
            np.asarray(
                row.get(
                    "_psf_agn_fraction_draws",
                    np.full((PSF_AGN_FRACTION_DRAW_COUNT, len(PSF_AGN_FRACTION_BANDS)), np.nan),
                ),
                dtype=np.float32,
            )
        )
        counts.append(int(row.get("_psf_agn_fraction_valid_count", 0)))
        payload = row.get("_joint_posterior_draws", {})
        for name in JOINT_POSTERIOR_DRAW_FIELDS:
            joint_draws[name].append(
                np.asarray(
                    payload.get(
                        name,
                        np.full(JOINT_POSTERIOR_DRAW_COUNT, np.nan),
                    ),
                    dtype=np.float32,
                )
            )
        joint_counts.append(int(row.get("_joint_posterior_valid_count", 0)))
        joint_indices.append(
            np.asarray(
                row.get(
                    "_joint_posterior_index",
                    np.full(JOINT_POSTERIOR_DRAW_COUNT, -1),
                ),
                dtype=np.int32,
            )
        )
        joint_source_counts.append(
            int(row.get("_joint_posterior_source_draw_count", 0))
        )
        if "_joint_posterior_selection_seed" not in row:
            raise ValueError("Worker row lacks _joint_posterior_selection_seed.")
        selection_seeds.add(int(row["_joint_posterior_selection_seed"]))
        psf_photometry = row.get("_joint_psf_photometry_draws")
        if psf_photometry is None:
            joint_psf_photometry.append(None)
        else:
            joint_psf_photometry.append(
                np.asarray(psf_photometry, dtype=np.float32)
            )
            row_provenance = row.get("_joint_psf_photometry_provenance")
            if row_provenance is None:
                raise ValueError(
                    "Worker row with joint PSF photometry lacks provenance."
                )
            if joint_psf_provenance is None:
                joint_psf_provenance = dict(row_provenance)
            elif dict(row_provenance) != joint_psf_provenance:
                raise ValueError(
                    "Worker rows use mixed joint PSF photometry provenance."
                )
    draw_array = np.stack(draws, axis=0) if draws else np.empty(
        (0, PSF_AGN_FRACTION_DRAW_COUNT, len(PSF_AGN_FRACTION_BANDS)), dtype=np.float32
    )
    if len(selection_seeds) > 1:
        raise ValueError(
            f"Worker rows use mixed joint posterior selection seeds: {selection_seeds}."
        )
    selection_seed = next(iter(selection_seeds), 0)
    joint_draw_arrays = {
        name: np.stack(values, axis=0)
        if values
        else np.empty((0, JOINT_POSTERIOR_DRAW_COUNT), dtype=np.float32)
        for name, values in joint_draws.items()
    }
    joint_index_array = (
        np.stack(joint_indices, axis=0)
        if joint_indices
        else np.empty((0, JOINT_POSTERIOR_DRAW_COUNT), dtype=np.int32)
    )
    normalized_psf_photometry = []
    for row_index, (value, count) in enumerate(
        zip(joint_psf_photometry, joint_counts, strict=True)
    ):
        if value is None:
            if int(count) > 0:
                raise ValueError(
                    "Successful worker row with joint posterior draws lacks "
                    f"joint PSF photometry at row {row_index}."
                )
            value = np.full(
                (
                    JOINT_PSF_PHOTOMETRY_DRAW_COUNT,
                    len(JOINT_PSF_PHOTOMETRY_BANDS),
                ),
                np.nan,
                dtype=np.float32,
            )
        normalized_psf_photometry.append(value)
    joint_psf_photometry_array = (
        np.stack(normalized_psf_photometry, axis=0)
        if normalized_psf_photometry
        else np.empty(
            (
                0,
                JOINT_PSF_PHOTOMETRY_DRAW_COUNT,
                len(JOINT_PSF_PHOTOMETRY_BANDS),
            ),
            dtype=np.float32,
        )
    )
    if joint_psf_provenance is None:
        joint_psf_provenance = joint_psf_photometry_prediction_provenance(
            "fit_attempt_no_valid_draws"
        )
    catalog_frame = pd.DataFrame(catalog_rows)
    if catalog_frame.empty and len(catalog_frame.columns) == 0:
        # Preserve the v3 scalar schema even for a deliberately empty shard.
        catalog_frame = pd.DataFrame(
            {
                "fit_ok": pd.Series(dtype=bool),
                "mw_deredden_applied": pd.Series(dtype=bool),
                "joint_posterior_draw_source": pd.Series(dtype=str),
                **{
                    name: pd.Series(dtype=float)
                    for name in JOINT_POSTERIOR_SCALAR_SUMMARY_FIELDS
                },
            }
        )
    write_spectra_catalog_hdf5(
        path,
        catalog_frame,
        draw_array,
        np.asarray(counts, dtype=np.int16),
        joint_posterior_draws=joint_draw_arrays,
        joint_posterior_valid_count=np.asarray(joint_counts, dtype=np.int16),
        joint_posterior_index=joint_index_array,
        joint_posterior_source_draw_count=np.asarray(
            joint_source_counts, dtype=np.int32
        ),
        joint_posterior_selection_seed=selection_seed,
        joint_psf_photometry_draws=joint_psf_photometry_array,
        joint_psf_photometry_provenance=joint_psf_provenance,
        provenance=provenance,
    )


def summarize_psf_agn_fractions(
    prediction,
    filter_names,
    *,
    bands=PSF_AGN_FRACTION_BANDS,
):
    """Summarize posterior variable-AGN/total PSF fractions in SDSS bands."""
    arrays = {}
    for name in ("pred_fluxes", "variable_agn_fluxes"):
        value = prediction.get(name)
        if value is None:
            raise ValueError(
                f"JAXSEDFit prediction is missing required component {name!r}."
            )
        arrays[name] = np.asarray(value, dtype=float)

    total_fluxes = arrays["pred_fluxes"]
    variable_agn_fluxes = arrays["variable_agn_fluxes"]
    if total_fluxes.ndim != 2:
        raise ValueError(
            "JAXSEDFit component predictions must have shape (draw, filter); "
            f"received {total_fluxes.shape}."
        )
    if variable_agn_fluxes.shape != total_fluxes.shape:
        raise ValueError(
            "JAXSEDFit total and variable-AGN component predictions must "
            "have identical shapes."
        )

    filter_names = [str(name) for name in filter_names]
    if len(filter_names) != total_fluxes.shape[1]:
        raise ValueError(
            "JAXSEDFit filter-name count does not match the prediction width: "
            f"{len(filter_names)} != {total_fluxes.shape[1]}."
        )

    out = empty_psf_agn_fraction_summary(bands)
    for band in bands:
        filter_name = f"{band}_sdss"
        indices = [
            index for index, name in enumerate(filter_names) if name == filter_name
        ]
        if len(indices) != 1:
            raise ValueError(
                f"Expected exactly one {filter_name!r} prediction, found {len(indices)}."
            )
        index = indices[0]
        total = total_fluxes[:, index]
        variable_agn = variable_agn_fluxes[:, index]
        valid = np.isfinite(total) & (total > 0.0) & np.isfinite(variable_agn)
        if not np.any(valid):
            raise ValueError(
                f"No valid posterior PSF AGN-fraction draws for band {band!r}."
            )
        fractions = np.clip(variable_agn[valid] / total[valid], 0.0, 1.0)
        median, err, _, _ = legacy.sym_percentile(fractions)
        out[f"f_AGN_psf_{band}"] = float(median)
        out[f"f_AGN_psf_{band}_err"] = float(err)
    return out


def joint_saved_name(rec):
    """Return the stable JAXSEDFit observation name for one QVC record."""
    return f"z{float(rec['z']):.3f}_{rec['sdss_name']}_joint"


def posterior_bundle_path(directory, rec):
    """Return the expected posterior-bundle path for one QVC record."""
    return Path(directory) / f"{joint_saved_name(rec)}_samples.h5"


def sed_figure_path(fig_dir, rec):
    """Return the shared fresh/resumed SED figure path."""
    return Path(fig_dir) / f"{joint_saved_name(rec)}.png"


def verify_new_posterior_bundle(path):
    """Require a newly written bundle to use the current compact schema."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Expected saved posterior bundle was not written: {path}")
    with h5py.File(path, "r") as handle:
        actual = handle.attrs.get("posterior_bundle_format")
    if isinstance(actual, bytes):
        actual = actual.decode("utf-8")
    if actual != POSTERIOR_BUNDLE_FORMAT:
        raise ValueError(
            f"New posterior bundle {path} uses schema {actual!r}; "
            f"expected {POSTERIOR_BUNDLE_FORMAT!r}."
        )
    return path


def annotate_posterior_bundle(path, args, rec, *, event_type="fit", source_path=None):
    """Attach QVC provenance after JAXSEDFit has finished writing its bundle."""
    path = verify_new_posterior_bundle(path)
    inputs = {
        "input_catalog": getattr(args, "fpath_in", None),
        "dr16q_catalog": getattr(args, "dr16q_fits", None),
        "sed_photometry": getattr(args, "sed_photometry_path", None),
        "sdss_psf_photometry": getattr(args, "sdss_psf_photometry_path", None),
        "sdss_spectrum_selection": getattr(
            args, "sdss_spectrum_selection_path", None
        ),
        "dsps_ssp": getattr(args, "dsps_ssp_fn", None),
    }
    record = build_run_record(
        "qvc.spectra.fit_spectra_jaxsedfit_joint",
        args,
        object_id=str(rec["object_id"]),
        input_paths=inputs,
        event_type=event_type,
    )
    previous = None
    if source_path:
        try:
            previous = read_hdf5_provenance(source_path)
        except (OSError, ValueError):
            previous = None
        record["source_bundle"] = fingerprint_path(source_path)
        if previous is None:
            record["source_bundle"]["provenance"] = "unavailable"
    write_hdf5_provenance(path, merge_history(record, previous))
    with h5py.File(path, "r+") as handle:
        missing_path = "samples/missing_psf_host_capture_fraction"
        scale_path = "samples/log_host_capture_scale_arcsec"
        if missing_path in handle:
            raise ValueError(
                "New posterior bundle unexpectedly contains "
                "missing_psf_host_capture_fraction samples despite static "
                "PSF FWHMs."
            )
        if scale_path not in handle:
            raise ValueError(
                "New posterior bundle lacks log_host_capture_scale_arcsec samples."
            )
        handle.attrs[HOST_CAPTURE_BUNDLE_ATTR] = HOST_CAPTURE_BUNDLE_MARKER
        handle.attrs[HOST_CAPTURE_PSF_FWHM_ATTR] = SDSS_TYPICAL_PSF_FWHM_ARCSEC
        handle.attrs[HOST_CAPTURE_QVC_FWHM_ATTR] = np.asarray(
            [SDSS_STATIC_PSF_FWHM_ARCSEC[band] for band in PSF_AGN_FRACTION_BANDS],
            dtype=float,
        )
    return path


def validate_resume_host_capture_fitter(fitter, path):
    """Validate the static-FWHM QVC host-capture model after loading."""
    likelihood = getattr(getattr(fitter, "config", None), "likelihood", None)
    if not bool(getattr(likelihood, "use_host_capture_model", False)):
        raise IncompatibleHostCaptureResumeError(
            _host_capture_resume_message(
                path,
                "saved configuration has host-capture modeling disabled",
            )
        )
    photometry = getattr(getattr(fitter, "config", None), "photometry", None)
    filter_names = list(getattr(photometry, "filter_names", ()) or ())
    expected_filters = {f"{band}_sdss" for band in PSF_AGN_FRACTION_BANDS}
    qvc_indices = [
        index for index, name in enumerate(filter_names) if str(name) in expected_filters
    ]
    if len(qvc_indices) != len(expected_filters) or {
        str(filter_names[index]) for index in qvc_indices
    } != expected_filters:
        raise IncompatibleHostCaptureResumeError(
            _host_capture_resume_message(
                path,
                "configuration does not contain exactly one QVC measurement "
                "for each of ugriz",
            )
        )
    methods = list(getattr(photometry, "photometry_method", ()) or ())
    scales = list(getattr(photometry, "psf_fwhm_arcsec", ()) or ())
    if len(methods) != len(filter_names) or len(scales) != len(filter_names):
        raise IncompatibleHostCaptureResumeError(
            _host_capture_resume_message(path, "photometry metadata is incomplete")
        )
    invalid_metadata = [
        str(filter_names[index])
        for index in qvc_indices
        if str(methods[index]).strip().lower() != "psf"
        or not np.isclose(
            legacy.safe_float(scales[index]),
            SDSS_STATIC_PSF_FWHM_ARCSEC[str(filter_names[index])[0]],
        )
    ]
    if invalid_metadata:
        raise IncompatibleHostCaptureResumeError(
            _host_capture_resume_message(
                path,
                "QVC measurements do not use the expected static SDSS PSF FWHMs: "
                f"{invalid_metadata}",
            )
        )
    samples = getattr(fitter, "samples", None) or {}
    if "missing_psf_host_capture_fraction" in samples:
        raise IncompatibleHostCaptureResumeError(
            _host_capture_resume_message(
                path,
                "posterior unexpectedly contains missing-PSF host-capture fractions",
            )
        )
    scales = np.asarray(samples.get("log_host_capture_scale_arcsec", []))
    if scales.size == 0 or not np.all(np.isfinite(scales)):
        raise IncompatibleHostCaptureResumeError(
            _host_capture_resume_message(
                path, "posterior lacks a finite fitted host-capture scale"
            )
        )


def preflight_resume_host_capture_bundles(records, args):
    """Reject missing, old, or mixed-model resume bundles before workers start."""
    failures = []
    for rec in records:
        path = posterior_bundle_path(args.resume, rec)
        if not path.is_file():
            failures.append(f"{path}: missing")
            continue
        try:
            with h5py.File(path, "r") as handle:
                unannotated_accepted = False
                marker = handle.attrs.get(HOST_CAPTURE_BUNDLE_ATTR)
                if isinstance(marker, bytes):
                    marker = marker.decode("utf-8")
                if marker != HOST_CAPTURE_BUNDLE_MARKER:
                    allow_unannotated = bool(
                        getattr(args, "allow_unannotated_resume_bundle", False)
                    )
                    if marker not in (None, "") or not allow_unannotated:
                        raise ValueError(
                            f"host-capture model marker is {marker!r}; "
                            f"expected {HOST_CAPTURE_BUNDLE_MARKER!r}"
                        )
                    unannotated_accepted = True
                fwhm = handle.attrs.get(HOST_CAPTURE_PSF_FWHM_ATTR)
                missing_unannotated_fwhm = unannotated_accepted and fwhm is None
                if not missing_unannotated_fwhm and (
                    not np.isfinite(legacy.safe_float(fwhm))
                    or not np.isclose(
                        float(fwhm), SDSS_TYPICAL_PSF_FWHM_ARCSEC
                    )
                ):
                    raise ValueError(
                        f"prediction PSF FWHM is {fwhm!r}; expected "
                        f"{SDSS_TYPICAL_PSF_FWHM_ARCSEC} arcsec"
                    )
                qvc_fwhm = handle.attrs.get(HOST_CAPTURE_QVC_FWHM_ATTR)
                expected_qvc_fwhm = np.asarray(
                    [
                        SDSS_STATIC_PSF_FWHM_ARCSEC[band]
                        for band in PSF_AGN_FRACTION_BANDS
                    ],
                    dtype=float,
                )
                missing_unannotated_qvc_fwhm = (
                    unannotated_accepted and qvc_fwhm is None
                )
                if not missing_unannotated_qvc_fwhm and (
                    np.asarray(qvc_fwhm).shape != expected_qvc_fwhm.shape
                    or not np.allclose(np.asarray(qvc_fwhm, dtype=float), expected_qvc_fwhm)
                ):
                    raise ValueError(
                        f"QVC SDSS PSF FWHMs are {qvc_fwhm!r}; expected "
                        f"{expected_qvc_fwhm.tolist()} arcsec in ugriz order"
                    )
                fraction_path = "samples/missing_psf_host_capture_fraction"
                if fraction_path in handle:
                    raise ValueError("unexpected missing-PSF host-capture samples")
                if "samples/log_host_capture_scale_arcsec" not in handle:
                    raise ValueError("missing fitted host-capture scale samples")
                if unannotated_accepted:
                    warnings.warn(
                        "Explicitly accepting an unannotated resume bundle after "
                        f"structural sample validation: {path}. The loaded "
                        "JAXSEDFit configuration will still be checked before "
                        "posterior prediction.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        except (OSError, ValueError) as exc:
            failures.append(f"{path}: {exc}")
        if len(failures) >= 10:
            break
    if failures:
        detail = "; ".join(failures)
        raise IncompatibleHostCaptureResumeError(
            _host_capture_resume_message(
                args.resume,
                "resume preflight found incompatible bundles (first failures: "
                f"{detail})",
            )
        )


def _base_result(rec, args, *, execution_mode, resumed_from_path=""):
    """Build the stable output row shared by fresh and resumed execution."""
    result = {
        key: rec.get(key)
        for key in (
            "object_id",
            "sdss_name",
            "plate",
            "fiber",
            "mjd",
            "z",
            "ra",
            "dec",
            "loglbol",
        )
    }
    result.update(
        {
            # The production input catalog historically misspelled this field
            # as SDSSS_RUN2D.  Emit one canonical column from every worker so
            # fresh and resumed shards retain the survey-reduction metadata
            # needed by the Hubble sample filter.
            "SDSS_RUN2D": rec.get("SDSS_RUN2D", rec.get("SDSSS_RUN2D")),
            "fit_ok": False,
            "error_message": "",
            "fit_backend": "jaxsedfit_joint",
            "execution_mode": execution_mode,
            "resumed_from_run": str(getattr(args, "resume_run_name", "") or ""),
            "resumed_from_path": str(resumed_from_path or ""),
            "resume_error_message": "",
            "fit_result_path": "",
            "sed_fig_path": "",
            "spectrum_fig_path": "",
            "mw_deredden_applied": not bool(
                getattr(args, "no_deredden", False)
            ),
            "joint_posterior_draw_source": str(execution_mode),
        }
    )
    result.update(empty_joint_chi2_summary())
    result.update(empty_psf_agn_fraction_summary())
    result.update(empty_host_2500_psf_summary())
    result.update(empty_joint_hubble_posterior_summary())
    result.update(empty_hubble_convergence_summary())
    result["_psf_agn_fraction_draws"] = np.full(
        (PSF_AGN_FRACTION_DRAW_COUNT, len(PSF_AGN_FRACTION_BANDS)),
        np.nan,
        dtype=np.float32,
    )
    result["_psf_agn_fraction_valid_count"] = 0
    result["_joint_posterior_draws"] = {
        name: np.full(JOINT_POSTERIOR_DRAW_COUNT, np.nan, dtype=np.float32)
        for name in JOINT_POSTERIOR_DRAW_FIELDS
    }
    result["_joint_posterior_valid_count"] = 0
    result["_joint_posterior_index"] = np.full(
        JOINT_POSTERIOR_DRAW_COUNT, -1, dtype=np.int32
    )
    result["_joint_posterior_source_draw_count"] = 0
    result["_joint_posterior_selection_seed"] = int(
        getattr(args, "seed", 0)
    )
    result["_joint_psf_photometry_draws"] = None
    result["_joint_psf_photometry_provenance"] = None
    return result


def _clear_compact_posterior_payloads(result):
    """Ensure a failed worker row cannot retain a partially built draw payload."""

    result["_psf_agn_fraction_draws"] = np.full(
        (PSF_AGN_FRACTION_DRAW_COUNT, len(PSF_AGN_FRACTION_BANDS)),
        np.nan,
        dtype=np.float32,
    )
    result["_psf_agn_fraction_valid_count"] = 0
    result["_joint_posterior_draws"] = {
        name: np.full(JOINT_POSTERIOR_DRAW_COUNT, np.nan, dtype=np.float32)
        for name in JOINT_POSTERIOR_DRAW_FIELDS
    }
    result["_joint_posterior_valid_count"] = 0
    result["_joint_posterior_index"] = np.full(
        JOINT_POSTERIOR_DRAW_COUNT, -1, dtype=np.int32
    )
    result["_joint_posterior_source_draw_count"] = 0
    result["_joint_psf_photometry_draws"] = None
    result["_joint_psf_photometry_provenance"] = None


def _uses_nuts(config, method=None):
    configured = getattr(getattr(config, "inference", None), "method", "")
    return "nuts" in str(method or configured).lower()


def save_spectrum_figure(fitter, rec, fig_dir):
    """Save the jaxqsofit spectral decomposition beside the joint SED figure."""
    from matplotlib import pyplot as plt

    fig_path = (
        Path(fig_dir)
        / f"z{float(rec['z']):.3f}_{rec['sdss_name']}_spectrum.png"
    )
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig = fitter.plot_spectrum(show_plot=False, plot_residual=False)
    if fig is None:
        raise RuntimeError("JAXSEDFit did not return a spectrum figure.")
    try:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"Saved spectrum plot: {fig_path}")
    return fig_path


def initialization_figure_path(fig_dir, rec, stage):
    """Return the stable output path for one Optax initialization stage."""
    return Path(fig_dir) / f"{joint_saved_name(rec)}_init_{stage}.png"


def fit_with_saved_initialization_plots(fitter, rec, args):
    """Run the fit while saving initialization plots without GUI windows."""
    if not bool(getattr(args, "plot_init", False)):
        return fitter.fit(progress_bar=args.progress)

    title_stages = {
        "Stage 1 continuum/host MAP initialization": "stage1",
        "Stage 2 smooth spectral-feature MAP initialization": "stage2",
        "Stage 3 full MAP initialization": "stage3",
        "Full MAP initialization": "map",
    }
    original_plot_sed = fitter.plot_sed

    def save_instead_of_showing(*call_args, **call_kwargs):
        stage = title_stages.get(str(call_kwargs.get("title", "")))
        if stage is None:
            return original_plot_sed(*call_args, **call_kwargs)
        output_path = initialization_figure_path(args.fig_dir, rec, stage)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        call_kwargs["output_path"] = output_path
        call_kwargs["show"] = False
        figure = original_plot_sed(*call_args, **call_kwargs)
        if not output_path.is_file():
            raise FileNotFoundError(
                f"Initialization figure was not written: {output_path}"
            )
        print(f"Saved initialization plot: {output_path}")
        return figure

    fitter.plot_sed = save_instead_of_showing
    try:
        return fitter.fit(progress_bar=args.progress)
    finally:
        fitter.plot_sed = original_plot_sed


def release_jaxsedfit_memory(fitter=None, fit_result=None):
    """Discard one fit's large state and clear process-wide JAX caches."""
    states = []
    for owner in (fitter, fit_result):
        state = getattr(owner, "_fit_state", None)
        if state is None:
            state = getattr(owner, "_state", None)
        if state is not None and all(state is not item for item in states):
            states.append(state)

    for state in states:
        figure = getattr(state, "figure", None)
        if figure is not None:
            try:
                from matplotlib import pyplot as plt

                plt.close(figure)
            except Exception:
                pass
        for name in (
            "map_result",
            "nuts_result",
            "ns_result",
            "samples",
            "predictive",
            "predictive_cache",
            "summary",
            "figure",
            "plot_cache",
        ):
            try:
                setattr(state, name, None)
            except Exception:
                pass

    if fit_result is not None:
        for name in (
            "fitter",
            "samples",
            "median",
            "summary",
            "figure",
            "_state",
            "_spectrum",
        ):
            try:
                setattr(fit_result, name, None)
            except Exception:
                pass

    if fitter is not None:
        reset = getattr(fitter, "_reset_fit_state", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                pass
        for name in ("context", "config"):
            try:
                setattr(fitter, name, None)
            except Exception:
                pass

    gc.collect()
    try:
        import jax

        jax.clear_caches()
    except Exception:
        # Cleanup must not replace a completed fit or mask its original error.
        pass
    gc.collect()


def run_one_fit(
    rec,
    args,
    *,
    execution_mode="fresh",
    resumed_from_path="",
    resume_error_message="",
):
    """Run a complete joint fit and return its stable flat result row."""
    result = _base_result(
        rec,
        args,
        execution_mode=execution_mode,
        resumed_from_path=resumed_from_path,
    )
    result["resume_error_message"] = str(resume_error_message or "")
    fitter = None
    fit_result = None
    try:
        hdul = legacy.load_spec_from_cache(
            rec["plate"], rec["fiber"], rec["mjd"], cache_dir=args.cache_dir
        )
        if hdul is None:
            hdul = legacy.fetch_spectrum_fits(
                args, rec["plate"], rec["fiber"], rec["mjd"], cache_dir=args.cache_dir
            )
        try:
            lam, flux, err, resolving_power = legacy.get_spectrum_arrays(hdul)
            aperture_diameter_arcsec = sdss_spectrum_aperture_diameter_arcsec(hdul)
        finally:
            hdul.close()

        phot = pd.DataFrame(rec.get("_joint_photometry", []))
        config, used_phot = build_joint_config(
            rec,
            phot,
            lam,
            flux,
            err,
            resolving_power,
            args,
            aperture_diameter_arcsec=aperture_diameter_arcsec,
        )
        from jaxsedfit import JAXSEDFit

        fitter = JAXSEDFit(config)
        fit_result = fit_with_saved_initialization_plots(fitter, rec, args)
        prediction = predict_catalog_posterior(
            fitter,
            kind="plot",
        )
        prediction = add_sdss_psf_host_fraction_prediction(prediction)
        m2500_samples = posterior_samples_for_m2500(
            fit_result.samples,
            prediction,
        )
        derived_draws = estimate_joint_hubble_posterior_draws(
            m2500_samples,
            rec["z"],
            h0=config.galaxy.cosmology_h0,
            om0=config.galaxy.cosmology_om0,
        )
        result.update(summarize_catalog_posterior(fit_result.samples, prediction))
        result.update(summarize_joint_chi2(prediction))
        result.update(summarize_host_2500_psf(prediction))
        (
            result["_joint_posterior_draws"],
            result["_joint_posterior_valid_count"],
            result["_joint_posterior_index"],
            result["_joint_posterior_source_draw_count"],
        ) = extract_compact_joint_posterior_draws(
            prediction,
            derived_draws,
            object_id=rec["object_id"],
            seed=args.seed,
        )
        if bool(
            getattr(
                getattr(config, "observation", None),
                "apply_mw_deredden",
                not bool(getattr(args, "no_deredden", False)),
            )
        ):
            result["_joint_psf_photometry_draws"] = (
                extract_aligned_joint_psf_photometry_draws(
                    prediction,
                    used_phot["filter_name"].astype(str).tolist(),
                    result["_joint_posterior_index"],
                    result["_joint_posterior_valid_count"],
                )
            )
            result["_joint_psf_photometry_provenance"] = (
                joint_psf_photometry_prediction_provenance(
                    "fresh_fit_prediction"
                )
            )
        result.update(
            summarize_psf_agn_fractions(
                prediction,
                used_phot["filter_name"].astype(str).tolist(),
            )
        )
        (
            result["_psf_agn_fraction_draws"],
            result["_psf_agn_fraction_valid_count"],
        ) = extract_compact_psf_agn_fraction_draws(
            prediction,
            used_phot["filter_name"].astype(str).tolist(),
            object_id=rec["object_id"],
            seed=args.seed,
        )
        result.update(summarize_joint_hubble_posterior_draws(derived_draws))
        result["mw_deredden_applied"] = bool(
            getattr(
                getattr(config, "observation", None),
                "apply_mw_deredden",
                not bool(getattr(args, "no_deredden", False)),
            )
        )
        result["joint_posterior_draw_source"] = "fresh_fit"
        if _uses_nuts(config, getattr(fit_result, "method", None)):
            try:
                result.update(
                    summarize_spectral_convergence(
                        _fresh_grouped_nuts_samples(
                            fit_result,
                            prediction,
                        ),
                        rec["z"],
                        h0=config.galaxy.cosmology_h0,
                        om0=config.galaxy.cosmology_om0,
                        heading=(
                            f"[{rec['object_id']}] NumPyro spectral "
                            "posterior summary:"
                        ),
                        print_summary=getattr(
                            args, "print_convergence_summary", True
                        ),
                    )
                )
            except Exception as exc:
                if args.verbose:
                    print(
                        f"Spectral convergence unavailable for "
                        f"object_id={rec['object_id']}: {exc}"
                    )
        result["n_photometry"] = int(len(used_phot))
        result["photometry_filters"] = ",".join(used_phot["filter_name"].astype(str))
        result["fit_result_path"] = str(fit_result.path or "")
        if args.save_jaxsedfit_samples:
            annotate_posterior_bundle(
                result["fit_result_path"],
                args,
                rec,
                event_type=execution_mode,
                source_path=resumed_from_path or None,
            )
        result["sed_fig_path"] = (
            str(sed_figure_path(args.fig_dir, rec)) if args.save_fig else ""
        )
        result["spectrum_fig_path"] = (
            str(save_spectrum_figure(fitter, rec, args.fig_dir))
            if args.save_fig
            else ""
        )
        result["fit_ok"] = True
    except M2500ReconstructionError:
        raise
    except Exception as exc:
        _clear_compact_posterior_payloads(result)
        result["error_message"] = str(exc)
        if args.verbose:
            traceback.print_exc()
    finally:
        release_jaxsedfit_memory(fitter, fit_result)
    return result


def _run_resumed_fit(rec, args, source_path):
    """Regenerate one object from saved draws and write current-schema outputs."""
    from jaxsedfit import JAXSEDFit

    fitter = None
    try:
        fitter = JAXSEDFit.load(source_path)
        return _complete_resumed_fit(rec, args, source_path, fitter)
    finally:
        release_jaxsedfit_memory(fitter)


def _complete_resumed_fit(rec, args, source_path, fitter):
    """Build current catalog products from one already-loaded posterior."""
    from matplotlib import pyplot as plt

    result = _base_result(
        rec,
        args,
        execution_mode="resumed",
        resumed_from_path=source_path,
    )
    fitter.predictive = None
    validate_resume_host_capture_fitter(fitter, source_path)
    config = fitter.config
    saved_name = str(config.observation.object_id)
    expected_name = joint_saved_name(rec)
    if saved_name != expected_name:
        raise ValueError(
            f"Posterior bundle observation ID {saved_name!r} does not match "
            f"selected object {expected_name!r}."
        )

    prediction = predict_catalog_posterior(
        fitter,
        kind="plot",
    )
    prediction = add_sdss_psf_host_fraction_prediction(prediction)
    m2500_samples = posterior_samples_for_m2500(
        fitter.samples,
        prediction,
    )
    derived_draws = estimate_joint_hubble_posterior_draws(
        m2500_samples,
        config.observation.redshift,
        h0=config.galaxy.cosmology_h0,
        om0=config.galaxy.cosmology_om0,
    )
    result.update(summarize_catalog_posterior(fitter.samples, prediction))
    result.update(summarize_joint_chi2(prediction))
    result.update(summarize_host_2500_psf(prediction))
    (
        result["_joint_posterior_draws"],
        result["_joint_posterior_valid_count"],
        result["_joint_posterior_index"],
        result["_joint_posterior_source_draw_count"],
    ) = extract_compact_joint_posterior_draws(
        prediction,
        derived_draws,
        object_id=rec["object_id"],
        seed=args.seed,
    )
    if bool(
        getattr(
            getattr(config, "observation", None),
            "apply_mw_deredden",
            not bool(getattr(args, "no_deredden", False)),
        )
    ):
        result["_joint_psf_photometry_draws"] = (
            extract_aligned_joint_psf_photometry_draws(
                prediction,
                config.photometry.filter_names,
                result["_joint_posterior_index"],
                result["_joint_posterior_valid_count"],
            )
        )
        result["_joint_psf_photometry_provenance"] = (
            joint_psf_photometry_prediction_provenance(
                "saved_posterior_bundle_prediction"
            )
        )
    result.update(
        summarize_psf_agn_fractions(
            prediction,
            config.photometry.filter_names,
        )
    )
    (
        result["_psf_agn_fraction_draws"],
        result["_psf_agn_fraction_valid_count"],
    ) = extract_compact_psf_agn_fraction_draws(
        prediction,
        config.photometry.filter_names,
        object_id=rec["object_id"],
        seed=args.seed,
    )
    result.update(summarize_joint_hubble_posterior_draws(derived_draws))
    result["mw_deredden_applied"] = bool(
        getattr(
            config.observation,
            "apply_mw_deredden",
            not bool(getattr(args, "no_deredden", False)),
        )
    )
    result["joint_posterior_draw_source"] = "resume_bundle_reprocess"
    if _uses_nuts(config):
        try:
            result.update(
                summarize_spectral_convergence(
                    _reshape_flat_samples_by_chain(
                        m2500_samples,
                        config.inference.num_chains,
                    ),
                    config.observation.redshift,
                    h0=config.galaxy.cosmology_h0,
                    om0=config.galaxy.cosmology_om0,
                    heading=(
                        f"[{rec['object_id']}] NumPyro spectral "
                        "posterior summary:"
                    ),
                    print_summary=getattr(
                        args, "print_convergence_summary", True
                    ),
                )
            )
        except Exception as exc:
            if args.verbose:
                print(
                    f"Spectral convergence unavailable for "
                    f"object_id={rec['object_id']}: {exc}"
                )
    result["n_photometry"] = int(len(config.photometry.filter_names))
    result["photometry_filters"] = ",".join(
        str(name) for name in config.photometry.filter_names
    )

    if args.save_jaxsedfit_samples:
        result_path = fitter.save(args.output_dir)
        result["fit_result_path"] = str(
            annotate_posterior_bundle(
                result_path,
                args,
                rec,
                event_type="resume",
                source_path=source_path,
            )
        )
    else:
        # The immutable source bundle remains the posterior backing this row.
        # Retain that useful reference when a catalog-only resume deliberately
        # avoids writing a many-gigabyte duplicate bundle set.
        result["fit_result_path"] = str(source_path)

    if args.save_fig:
        sed_path = sed_figure_path(args.fig_dir, rec)
        sed_path.parent.mkdir(parents=True, exist_ok=True)
        sed_fig = fitter.plot_sed(output_path=sed_path, show=False)
        if sed_fig is not None:
            plt.close(sed_fig)
        if not sed_path.is_file():
            raise FileNotFoundError(f"Resumed SED figure was not written: {sed_path}")
        result["sed_fig_path"] = str(sed_path)
        result["spectrum_fig_path"] = str(
            save_spectrum_figure(fitter, rec, args.fig_dir)
        )

    result["fit_ok"] = True
    return result


def _remove_incomplete_resumed_outputs(rec, args):
    """Remove only the exact new-run artifacts a failed resume may have written."""
    paths = [
        posterior_bundle_path(args.output_dir, rec),
        sed_figure_path(args.fig_dir, rec),
        Path(args.fig_dir)
        / f"z{float(rec['z']):.3f}_{rec['sdss_name']}_spectrum.png",
    ]
    for path in paths:
        path.unlink(missing_ok=True)


def run_hybrid_fit(rec, args):
    """Resume one selected object, retaining fallback only for non-model errors."""
    source_path = posterior_bundle_path(args.resume, rec)
    if not source_path.is_file():
        raise IncompatibleHostCaptureResumeError(
            _host_capture_resume_message(source_path, "expected bundle is missing")
        )

    try:
        return _run_resumed_fit(rec, args, source_path)
    except (IncompatibleHostCaptureResumeError, M2500ReconstructionError):
        raise
    except Exception as exc:
        resume_error = f"{type(exc).__name__}: {exc}"
        _remove_incomplete_resumed_outputs(rec, args)
        if bool(getattr(args, "resume_only", False)):
            raise RuntimeError(
                "Strict --resume-only reconstruction failed for "
                f"object_id={rec.get('object_id')} from {source_path}; "
                "a fresh Optax/NUTS fit was not started. "
                f"Original error: {resume_error}"
            ) from exc
        if args.verbose:
            print(
                f"Resume failed for object_id={rec.get('object_id')} from "
                f"{source_path}; running a fresh fit. {resume_error}"
            )
            traceback.print_exc()
        return run_one_fit(
            rec,
            args,
            execution_mode="fresh_resume_failed",
            resumed_from_path=source_path,
            resume_error_message=resume_error,
        )


def load_prepared_resume_records(path, object_ids):
    """Load already cross-matched records for inexpensive local resume batches."""
    path = Path(path)
    frame = pd.read_csv(path, dtype={"object_id": str})
    required = {
        "object_id",
        "sdss_name",
        "plate",
        "fiber",
        "mjd",
        "z",
        "ra",
        "dec",
    }
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Prepared resume records {path} are missing columns: "
            f"{sorted(missing_columns)}"
        )
    frame = frame.copy()
    frame["object_id"] = frame["object_id"].map(legacy.normalize_object_id)
    if (frame["object_id"] == "").any():
        raise ValueError(f"Prepared resume records {path} contain empty object IDs.")
    duplicates = frame.loc[
        frame["object_id"].duplicated(keep=False), "object_id"
    ].unique()
    if len(duplicates):
        raise ValueError(
            f"Prepared resume records {path} contain duplicate object IDs: "
            f"{duplicates[:10].tolist()}"
        )

    requested = [legacy.normalize_object_id(value) for value in object_ids or []]
    requested = [value for value in requested if value]
    requested = list(dict.fromkeys(requested))
    if requested:
        indexed = frame.set_index("object_id", drop=False)
        missing_ids = [value for value in requested if value not in indexed.index]
        if missing_ids:
            raise ValueError(
                f"Prepared resume records {path} lack {len(missing_ids)} requested "
                f"object ID(s); first missing: {missing_ids[:10]}"
            )
        frame = indexed.loc[requested].reset_index(drop=True)

    for column in ("plate", "fiber", "mjd", "z", "ra", "dec"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(
                f"Prepared resume records {path} contain non-finite {column!r}."
            )
        frame[column] = numeric
    if (frame["sdss_name"].astype(str).str.strip() == "").any():
        raise ValueError(
            f"Prepared resume records {path} contain empty SDSS names."
        )
    return frame.to_dict(orient="records")


def build_records(args):
    if getattr(args, "resume_records_path", None):
        records = load_prepared_resume_records(
            args.resume_records_path,
            args.filter_object_id,
        )
    else:
        records = legacy.build_records(args)
    spectrum_override_path = getattr(args, "sdss_spectrum_selection_path", None)
    if spectrum_override_path:
        selections = load_sdss_spectrum_selection_overrides(
            spectrum_override_path
        ).set_index("object_id")
        missing_ids = [
            str(rec["object_id"])
            for rec in records
            if str(rec["object_id"]) not in selections.index
        ]
        if missing_ids:
            raise ValueError(
                f"SDSS spectrum selection file {spectrum_override_path} lacks "
                f"selected object ID(s): {missing_ids[:10]}."
            )
        for rec in records:
            selected = selections.loc[str(rec["object_id"])]
            for column in ("plate", "mjd", "fiber"):
                rec[column] = int(selected[column])
            rec["z"] = float(selected["z"])
    if args.resume and bool(getattr(args, "resume_only", False)):
        # All photometry and spectroscopy needed for posterior prediction are
        # embedded in the saved bundle. Avoid re-reading the large SED table in
        # every strict, restartable local resume call. Ordinary hybrid resume
        # still prepares these inputs because it may fall back to a fresh fit.
        return records
    phot = load_saved_sed_photometry(args.sed_photometry_path)
    grouped = (
        {str(key): value.to_dict(orient="records") for key, value in phot.groupby("source_id")}
        if len(phot)
        else {}
    )
    for rec in records:
        rec["_joint_photometry"] = grouped.get(str(rec["object_id"]), [])
    override_path = getattr(args, "sdss_psf_photometry_path", None)
    if override_path:
        overrides = load_sdss_psf_photometry_overrides(override_path)
        override_groups = {
            str(key): value.to_dict(orient="records")
            for key, value in overrides.groupby("source_id")
        }
        missing_ids = [
            str(rec["object_id"])
            for rec in records
            if str(rec["object_id"]) not in override_groups
        ]
        if missing_ids:
            raise ValueError(
                f"SDSS PSF override file {override_path} lacks selected object "
                f"ID(s): {missing_ids[:10]}."
            )
        for rec in records:
            rec["_joint_sdss_psf_photometry"] = override_groups[
                str(rec["object_id"])
            ]
    return records


def run_fit(args):
    records = build_records(args)
    if not records:
        raise RuntimeError("No records to process.")
    if args.resume:
        preflight_resume_host_capture_bundles(records, args)
    worker = partial(run_hybrid_fit if args.resume else run_one_fit, args=args)
    if args.resume:
        description = (
            "Joint SED+spectrum resume-only"
            if args.resume_only
            else "Joint SED+spectrum resume/fallback"
        )
    else:
        description = "Joint SED+spectrum fits"
    if args.nproc <= 1:
        rows = [
            worker(rec)
            for rec in tqdm(
                records,
                desc=description,
                disable=not args.catalog_progress,
            )
        ]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.nproc) as pool:
            rows = list(
                tqdm(
                    pool.imap(worker, records),
                    total=len(records),
                    desc=description,
                    disable=not args.catalog_progress,
                )
            )
    provenance = build_run_record(
        "qvc.spectra.fit_spectra_jaxsedfit_joint",
        args,
        input_paths={
            "input_catalog": args.fpath_in,
            "sed_photometry": args.sed_photometry_path,
            "sdss_psf_photometry": getattr(
                args, "sdss_psf_photometry_path", None
            ),
            "sdss_spectrum_selection": getattr(
                args, "sdss_spectrum_selection_path", None
            ),
            "dr16q_catalog": args.dr16q_fits,
            "prepared_resume_records": getattr(args, "resume_records_path", None),
        },
        event_type="catalog_shard",
    )
    validate_m2500_catalog_rows(rows)
    write_joint_fit_results_hdf5(args.fpath_out, rows, provenance=provenance)
    print(f"Wrote {len(rows)} rows to {args.fpath_out}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Jointly fit saved SED photometry and SDSS spectra with jaxsedfit."
    )
    parser.add_argument("fpath_out", nargs="?")
    parser.add_argument("--fpath-in", dest="fpath_in")
    parser.add_argument("--fpath-out", dest="fpath_out_opt")
    parser.add_argument("--mode", choices=("download", "fit"), required=True)
    parser.add_argument("--dr16q-fits", default="data/dr16q_prop_May01_2024.fits")
    parser.add_argument("--cache-dir", default="data/spectra_cache")
    parser.add_argument(
        "--sed-photometry-path",
        help=(
            "Saved long-form SED table. Requires source_id/object_id, "
            "filter_name, flux_mjy, and flux_err_mjy."
        ),
    )
    parser.add_argument(
        "--sdss-psf-photometry-path",
        help=(
            "Optional experiment-only long-form table containing exactly one "
            "finite, positive ugriz SDSS PSF flux and error per selected object. "
            "When supplied, these rows replace the default QVC light-curve means."
        ),
    )
    parser.add_argument(
        "--sdss-spectrum-selection-path",
        help=(
            "Optional experiment-only CSV/ECSV selecting one exact SDSS "
            "plate/MJD/fiber and spectrum redshift for every selected object. "
            "When supplied, these values replace the DR16Q spectrum match."
        ),
    )
    parser.add_argument("--output-dir", default="results/jaxsedfit_joint")
    parser.add_argument("--fig-dir", default="plots/jaxsedfit_joint")
    parser.add_argument(
        "--resume",
        help=(
            "Directory containing compatible per-object *_samples.h5 bundles. "
            "Bundles must use the static SDSS ugriz PSF FWHMs and contain the "
            "fitted host-capture scale without missing-PSF fraction samples."
        ),
    )
    parser.add_argument(
        "--resume-run-name",
        default="",
        help="Old run identifier recorded in output provenance columns.",
    )
    parser.add_argument(
        "--resume-only",
        action="store_true",
        help=(
            "Fail if reconstruction from a saved posterior bundle fails; never "
            "fall back to a fresh Optax/NUTS fit. Requires --resume."
        ),
    )
    parser.add_argument(
        "--allow-unannotated-resume-bundle",
        action="store_true",
        help=(
            "Explicitly allow a structurally compatible main-branch bundle "
            "that is missing only QVC model-marker attributes. The loaded "
            "configuration is still validated. Requires --resume."
        ),
    )
    parser.add_argument(
        "--resume-records-path",
        help=(
            "CSV of already DR16Q-cross-matched records used to avoid repeating "
            "the expensive catalog preparation in local resume batches. Valid "
            "only with --resume."
        ),
    )
    parser.add_argument("--max-sep", type=float, default=1.0)
    parser.add_argument("--N", type=int)
    parser.add_argument("--skip", type=int)
    parser.add_argument("--filter_sdss_name", nargs="+")
    parser.add_argument("--filter_object_id", nargs="+")
    parser.add_argument("--fit-method", choices=("optax", "nuts", "optax+nuts"), default="optax+nuts")
    parser.add_argument("--dsps-ssp-fn", default="data/ssp_data_continuum_fsps_v3.2_lgmet_age.h5")
    parser.add_argument("--wave-min", type=float, default=1250.0)
    parser.add_argument("--wave-max", type=float, default=8000.0)
    parser.add_argument("--sed-n-wave", type=int, default=512)
    parser.add_argument("--photometry-systematics", type=float, default=0.08)
    parser.add_argument("--spectrum-systematics", type=float, default=0.08)
    parser.add_argument("--spectrum-student-t-df", type=float, default=5.0)
    parser.add_argument("--spectrum-scale-prior-sigma-dex", type=float, default=0.1)
    parser.add_argument("--line-flux-scale-mjy", type=float, default=0.1)
    parser.add_argument("--optax-steps", type=int, default=4000)
    parser.add_argument("--optax-lr", type=float, default=5e-3)
    parser.add_argument("--nuts-warmup", type=int, default=250)
    parser.add_argument("--nuts-samples", type=int, default=250)
    parser.add_argument("--nuts-chains", type=int, default=1)
    parser.add_argument("--nuts-target-accept", type=float, default=0.9)
    parser.add_argument("--dense-mass", choices=("blocks", "true", "false"), default="blocks")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--no-deredden", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--plot-init",
        action="store_true",
        help=(
            "Save each staged Optax initialization SED beside the final plot "
            "without opening GUI windows."
        ),
    )
    parser.set_defaults(catalog_progress=True)
    parser.add_argument(
        "--no-catalog-progress",
        dest="catalog_progress",
        action="store_false",
        help=(
            "Disable the outer per-process catalog progress bar. Useful when a "
            "parent driver provides one persistent run-level progress bar."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(print_convergence_summary=True)
    parser.add_argument(
        "--no-print-convergence-summary",
        dest="print_convergence_summary",
        action="store_false",
        help=(
            "Compute and save convergence fields without printing a posterior "
            "summary table for every object."
        ),
    )
    parser.set_defaults(fit_lines=True, fit_fe=True, fit_bc=True, save_fig=True, save_jaxsedfit_samples=True)
    parser.add_argument("--no-fit-lines", dest="fit_lines", action="store_false")
    parser.add_argument("--no-fit-fe", dest="fit_fe", action="store_false")
    parser.add_argument("--no-fit-bc", dest="fit_bc", action="store_false")
    parser.add_argument("--no-save-fig", dest="save_fig", action="store_false")
    parser.add_argument("--no-save-jaxsedfit-samples", dest="save_jaxsedfit_samples", action="store_false")
    args = parser.parse_args(argv)
    if args.fpath_out is None:
        args.fpath_out = args.fpath_out_opt
    if args.mode == "fit" and not args.fpath_out:
        parser.error("fpath_out is required for --mode fit.")
    if args.mode == "fit" and Path(args.fpath_out).suffix.lower() not in {".h5", ".hdf5"}:
        parser.error("fpath_out must end in .h5 or .hdf5 for joint spectral fits.")
    if args.mode == "fit" and not args.sed_photometry_path:
        parser.error("--sed-photometry-path is required for --mode fit.")
    if args.mode == "fit" and args.no_deredden:
        parser.error(
            "--no-deredden is incompatible with qvc_spectra_catalog_v3: "
            "mandatory fitted PSF colors are defined after Milky-Way "
            "foreground correction."
        )
    if not (args.fpath_in or args.filter_object_id):
        parser.error("--fpath-in or --filter_object_id is required.")
    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.is_dir():
            parser.error(f"--resume directory does not exist: {resume_dir}")
        if resume_dir.resolve() == Path(args.output_dir).resolve():
            parser.error("--resume and --output-dir must refer to different directories.")
        args.resume = str(resume_dir)
        if not args.resume_run_name:
            args.resume_run_name = resume_dir.parent.name
        if args.resume_records_path:
            if not args.resume_only:
                parser.error(
                    "--resume-records-path requires --resume-only because the "
                    "prepared records intentionally omit fresh-fit photometry."
                )
            resume_records_path = Path(args.resume_records_path)
            if not resume_records_path.is_file():
                parser.error(
                    "--resume-records-path does not exist: "
                    f"{resume_records_path}"
                )
            args.resume_records_path = str(resume_records_path)
    elif (
        args.resume_only
        or args.allow_unannotated_resume_bundle
        or args.resume_records_path
    ):
        parser.error(
            "--resume-only, --allow-unannotated-resume-bundle, and "
            "--resume-records-path require --resume."
        )
    args.dense_mass = {"true": True, "false": False}.get(args.dense_mass, args.dense_mass)
    return args


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)
    if args.mode == "download":
        legacy.run_download(args)
    else:
        run_fit(args)


if __name__ == "__main__":
    main()
