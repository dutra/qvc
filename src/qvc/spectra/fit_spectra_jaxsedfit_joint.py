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
import hashlib
import multiprocessing as mp
import traceback
from functools import partial
from pathlib import Path

import h5py
import numpy as np
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
    write_hdf5_provenance,
)
from tqdm import tqdm

from qvc.spectra import fit_spectra as legacy
from qvc.spectra.catalog_hdf5 import (
    PSF_AGN_FRACTION_DRAW_COUNT,
    write_spectra_catalog_hdf5,
)


C_ANGSTROM_PER_SECOND = 2.99792458e18
AB_ZEROPOINT_MJY = 3.631e6
METER_PER_MEGAPARSEC = 3.085677581491367e22
GRAHSP_ATTENUATION_BREAK_ANGSTROM = 11_000.0
POSTERIOR_BUNDLE_FORMAT = "jaxsedfit_samples_meta_v2"
PSF_AGN_FRACTION_BANDS = tuple(legacy.SDSS_BANDS)
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
DERIVED_SPECTRAL_CONVERGENCE_SITES = (
    *HUBBLE_MAGNITUDE_SITES,
    "a_2500_total",
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
            raise ValueError("Posterior sites have incompatible draw counts.")
        out.append(arr)
    return out


def _intrinsic_disk_luminosity_lambda_2500(samples):
    """Evaluate the unattenuated JAXSEDFit disk L_lambda at 2500 Angstrom."""
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

    wave = 2500.0
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

    ebv_gal = _sample_draws(samples, "ebv_gal", 0.0)
    ebv_agn = _sample_draws(samples, "ebv_agn", 0.0)
    ebv_gal, ebv_agn, intrinsic_mag = _broadcast_draws(
        ebv_gal, ebv_agn, intrinsic_mag
    )
    curve_2500 = (2500.0 / GRAHSP_ATTENUATION_BREAK_ANGSTROM) ** -1.2
    attenuation_gal = ebv_gal * curve_2500
    attenuation_agn = ebv_agn * curve_2500
    attenuation_total = attenuation_gal + attenuation_agn
    return {
        "m_2500_dereddened_draws": intrinsic_mag,
        "m_2500_attenuated_model_draws": (
            intrinsic_mag + attenuation_total
        ),
        "a_2500_galaxy_draws": attenuation_gal,
        "a_2500_internal_draws": attenuation_agn,
        "a_2500_total_draws": attenuation_total,
    }


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


def empty_hubble_convergence_summary():
    """Return the stable convergence schema used by the Hubble workflow."""

    return {
        f"{name}_rhat": np.nan
        for name in HUBBLE_MAGNITUDE_SITES
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


def _fresh_grouped_nuts_samples(fit_result):
    """Return scientific, chain-grouped draws from a fresh JAXSEDFit result."""

    fitter = getattr(fit_result, "fitter", None)
    nuts_result = getattr(fitter, "nuts_result", None)
    mcmc = nuts_result.get("mcmc") if isinstance(nuts_result, dict) else None
    if mcmc is None:
        raise ValueError("Fresh fit does not expose a NumPyro MCMC result.")
    grouped = mcmc.get_samples(group_by_chain=True)
    scientific_samples = getattr(fit_result, "samples", None) or {}
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
    derived = estimate_m2500_dereddened(
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
    print_numpyro_summary_dict(display_summary, heading=heading)
    fields = convergence_fields(
        summary_dict,
        {name: name for name in summary_samples},
    )
    return {name: value for name, value in fields.items() if name.endswith("_rhat")}


def load_saved_sed_photometry(path):
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
    return phot.loc[good].copy()


def add_qvc_psf_photometry(rec, phot):
    """Replace saved SDSS rows with QVC light-curve mean ugriz PSF fluxes."""
    phot = phot.copy()
    if "filter_name" in phot:
        sdss_names = {f"{band}_sdss" for band in legacy.SDSS_BANDS}
        phot = phot.loc[~phot["filter_name"].astype(str).isin(sdss_names)]

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
                "psf_fwhm_arcsec": np.nan,
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


def build_joint_config(rec, phot, lam, flux, err, resolving_power, args):
    """Construct one validated jaxsedfit joint SED+spectrum configuration."""
    from jaxsedfit import (
        AGNConfig,
        FilterSet,
        FitConfig,
        GalaxyConfig,
        InferenceConfig,
        JaxQSOFitConfig,
        JAXSEDFit,
        LikelihoodConfig,
        Observation,
        OutputConfig,
        PhotometryData,
        SpectroscopyConfig,
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
    methods = phot.get(
        "photometry_method", pd.Series("catalog", index=phot.index)
    ).astype(str).tolist()

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
            aperture_diameter_arcsec=3.0,
            epoch_mjd=float(rec["mjd"]),
        ),
        spectroscopy_config=SpectroscopyConfig(
            enabled=True,
            backend="jaxqsofit",
            fit_scale=True,
            scale_prior_sigma_dex=args.spectrum_scale_prior_sigma_dex,
            systematics_width=args.spectrum_systematics,
            student_t_df=args.spectrum_student_t_df,
            likelihood_weight_mode="resolution_elements",
            resolving_power=float(resolving_power),
            jaxqsofit=JaxQSOFitConfig(
                use_spectral_lines=args.fit_lines,
                use_tied_lines=args.fit_lines,
                use_spectral_smart_priors=True,
                use_spectral_feii=args.fit_fe,
                use_spectral_balmer_continuum=args.fit_bc,
                line_flux_scale_mjy=args.line_flux_scale_mjy,
            ),
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
            # The detailed spectrum backend owns Fe II and Balmer features.
            fit_balmer_continuum=False,
        ),
        likelihood=LikelihoodConfig(
            use_host_capture_model=True,
            systematics_width=args.photometry_systematics,
        ),
        inference=InferenceConfig(
            method=args.fit_method,
            map_steps=args.optax_steps,
            learning_rate=args.optax_lr,
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


def predict_catalog_posterior(fitter, *, kind):
    """Predict the standard products plus legacy scalar catalog sites.

    JAXSEDFit's public prediction products intentionally omit some scalar
    deterministics that were present in the original in-memory fit samples.
    Extend the return-site selection only for this prediction call so QVC can
    reproduce the legacy CSV columns from compact resume bundles.
    """
    original_return_sites = getattr(fitter, "_predictive_return_sites", None)
    if original_return_sites is None:
        raise RuntimeError(
            "Installed JAXSEDFit does not expose predictive return-site "
            "selection required to rebuild the legacy spectra catalog."
        )

    instance_vars = getattr(fitter, "__dict__", {})
    had_instance_override = "_predictive_return_sites" in instance_vars
    previous_instance_override = instance_vars.get("_predictive_return_sites")

    def catalog_return_sites(prediction_kind):
        return list(
            dict.fromkeys(
                (
                    *original_return_sites(prediction_kind),
                    *LEGACY_CSV_SCALAR_PREDICTION_SITES,
                )
            )
        )

    fitter._predictive_return_sites = catalog_return_sites
    try:
        return fitter.predict(kind=kind)
    finally:
        if had_instance_override:
            fitter._predictive_return_sites = previous_instance_override
        else:
            del fitter._predictive_return_sites


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


def write_joint_fit_results_hdf5(path, rows, *, provenance=None):
    """Write worker result rows and their private fraction-draw payloads."""

    catalog_rows = []
    draws = []
    counts = []
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
    draw_array = np.stack(draws, axis=0) if draws else np.empty(
        (0, PSF_AGN_FRACTION_DRAW_COUNT, len(PSF_AGN_FRACTION_BANDS)), dtype=np.float32
    )
    write_spectra_catalog_hdf5(
        path,
        pd.DataFrame(catalog_rows),
        draw_array,
        np.asarray(counts, dtype=np.int16),
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
    return path


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
        }
    )
    result.update(empty_joint_chi2_summary())
    result.update(empty_psf_agn_fraction_summary())
    result.update(empty_hubble_convergence_summary())
    result["_psf_agn_fraction_draws"] = np.full(
        (PSF_AGN_FRACTION_DRAW_COUNT, len(PSF_AGN_FRACTION_BANDS)),
        np.nan,
        dtype=np.float32,
    )
    result["_psf_agn_fraction_valid_count"] = 0
    return result


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
    fig = fitter.plot_jaxqsofit_spectrum(show_plot=False, plot_residual=False)
    if fig is None:
        raise RuntimeError("JAXSEDFit did not return a spectrum figure.")
    try:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"Saved spectrum plot: {fig_path}")
    return fig_path


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
        finally:
            hdul.close()

        phot = pd.DataFrame(rec.get("_joint_photometry", []))
        config, used_phot = build_joint_config(
            rec, phot, lam, flux, err, resolving_power, args
        )
        from jaxsedfit import JAXSEDFit

        fitter = JAXSEDFit(config)
        fit_result = fitter.fit(progress_bar=args.progress)
        prediction = predict_catalog_posterior(fitter, kind="photometry")
        result.update(summarize_catalog_posterior(fit_result.samples, prediction))
        result.update(summarize_joint_chi2(prediction))
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
        result.update(
            summarize_m2500_dereddened(
                fit_result.samples,
                rec["z"],
                h0=config.galaxy.cosmology_h0,
                om0=config.galaxy.cosmology_om0,
            )
        )
        if _uses_nuts(config, getattr(fit_result, "method", None)):
            try:
                result.update(
                    summarize_spectral_convergence(
                        _fresh_grouped_nuts_samples(fit_result),
                        rec["z"],
                        h0=config.galaxy.cosmology_h0,
                        om0=config.galaxy.cosmology_om0,
                        heading=(
                            f"[{rec['object_id']}] NumPyro spectral "
                            "posterior summary:"
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
    except Exception as exc:
        result["error_message"] = str(exc)
        if args.verbose:
            traceback.print_exc()
    return result


def _run_resumed_fit(rec, args, source_path):
    """Regenerate one object from saved draws and write current-schema outputs."""
    from matplotlib import pyplot as plt
    from jaxsedfit import JAXSEDFit

    result = _base_result(
        rec,
        args,
        execution_mode="resumed",
        resumed_from_path=source_path,
    )
    fitter = JAXSEDFit.load(source_path)
    fitter.predictive = None
    config = fitter.config
    saved_name = str(config.observation.object_id)
    expected_name = joint_saved_name(rec)
    if saved_name != expected_name:
        raise ValueError(
            f"Posterior bundle observation ID {saved_name!r} does not match "
            f"selected object {expected_name!r}."
        )

    prediction = predict_catalog_posterior(fitter, kind="plot")
    result.update(summarize_catalog_posterior(fitter.samples, prediction))
    result.update(summarize_joint_chi2(prediction))
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
    result.update(
        summarize_m2500_dereddened(
            fitter.samples,
            config.observation.redshift,
            h0=config.galaxy.cosmology_h0,
            om0=config.galaxy.cosmology_om0,
        )
    )
    if _uses_nuts(config):
        try:
            result.update(
                summarize_spectral_convergence(
                    _reshape_flat_samples_by_chain(
                        fitter.samples,
                        config.inference.num_chains,
                    ),
                    config.observation.redshift,
                    h0=config.galaxy.cosmology_h0,
                    om0=config.galaxy.cosmology_om0,
                    heading=(
                        f"[{rec['object_id']}] NumPyro spectral "
                        "posterior summary:"
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
    """Resume one selected object when possible, otherwise fit it from scratch."""
    source_path = posterior_bundle_path(args.resume, rec)
    if not source_path.is_file():
        return run_one_fit(
            rec,
            args,
            execution_mode="fresh_missing_bundle",
            resumed_from_path=source_path,
        )

    try:
        return _run_resumed_fit(rec, args, source_path)
    except Exception as exc:
        resume_error = f"{type(exc).__name__}: {exc}"
        if args.verbose:
            print(
                f"Resume failed for object_id={rec.get('object_id')} from "
                f"{source_path}; running a fresh fit. {resume_error}"
            )
            traceback.print_exc()
        _remove_incomplete_resumed_outputs(rec, args)
        return run_one_fit(
            rec,
            args,
            execution_mode="fresh_resume_failed",
            resumed_from_path=source_path,
            resume_error_message=resume_error,
        )


def build_records(args):
    records = legacy.build_records(args)
    phot = load_saved_sed_photometry(args.sed_photometry_path)
    grouped = (
        {str(key): value.to_dict(orient="records") for key, value in phot.groupby("source_id")}
        if len(phot)
        else {}
    )
    for rec in records:
        rec["_joint_photometry"] = grouped.get(str(rec["object_id"]), [])
    return records


def run_fit(args):
    records = build_records(args)
    if not records:
        raise RuntimeError("No records to process.")
    worker = partial(run_hybrid_fit if args.resume else run_one_fit, args=args)
    description = (
        "Joint SED+spectrum resume/fallback"
        if args.resume
        else "Joint SED+spectrum fits"
    )
    if args.nproc <= 1:
        rows = [worker(rec) for rec in tqdm(records, desc=description)]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.nproc) as pool:
            rows = list(tqdm(pool.imap(worker, records), total=len(records)))
    provenance = build_run_record(
        "qvc.spectra.fit_spectra_jaxsedfit_joint",
        args,
        input_paths={
            "input_catalog": args.fpath_in,
            "sed_photometry": args.sed_photometry_path,
            "dr16q_catalog": args.dr16q_fits,
        },
        event_type="catalog_shard",
    )
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
    parser.add_argument("--output-dir", default="results/jaxsedfit_joint")
    parser.add_argument("--fig-dir", default="plots/jaxsedfit_joint")
    parser.add_argument(
        "--resume",
        help=(
            "Directory containing old per-object *_samples.h5 bundles. "
            "Missing or unusable bundles fall back to complete fresh fits."
        ),
    )
    parser.add_argument(
        "--resume-run-name",
        default="",
        help="Old run identifier recorded in output provenance columns.",
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
    parser.add_argument("--verbose", action="store_true")
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
