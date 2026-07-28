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
import multiprocessing as mp
import traceback
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from qvc.spectra import fit_spectra as legacy


C_ANGSTROM_PER_SECOND = 2.99792458e18
AB_ZEROPOINT_MJY = 3.631e6


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
            object_id=f"z{float(rec['z']):.3f}_{rec['sdss_name']}_joint",
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
            fig_path=str(Path(args.fig_dir) / f"z{float(rec['z']):.3f}_{rec['sdss_name']}_joint.png"),
            plot_fig=False,
            save_fig=args.save_fig,
            save_result=args.save_jaxsedfit_samples,
            show_plot=False,
        ),
    )
    config.validate()
    return config, phot


def summarize_samples(samples):
    """Flatten scalar posterior sites into median/error CSV columns."""
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


def run_one_fit(rec, args):
    result = {
        key: rec.get(key)
        for key in ("object_id", "sdss_name", "plate", "fiber", "mjd", "z", "ra", "dec", "loglbol")
    }
    result.update({"fit_ok": False, "error_message": "", "fit_backend": "jaxsedfit_joint"})
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

        phot = pd.DataFrame(rec.pop("_joint_photometry"))
        config, used_phot = build_joint_config(
            rec, phot, lam, flux, err, resolving_power, args
        )
        from jaxsedfit import JAXSEDFit

        fit_result = JAXSEDFit(config).fit(progress_bar=args.progress)
        result.update(summarize_samples(fit_result.samples))
        result["n_photometry"] = int(len(used_phot))
        result["photometry_filters"] = ",".join(used_phot["filter_name"].astype(str))
        result["fit_result_path"] = str(fit_result.path or "")
        result["fit_ok"] = True
    except Exception as exc:
        result["error_message"] = str(exc)
        if args.verbose:
            traceback.print_exc()
    return result


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
    worker = partial(run_one_fit, args=args)
    if args.nproc <= 1:
        rows = [worker(rec) for rec in tqdm(records, desc="Joint SED+spectrum fits")]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.nproc) as pool:
            rows = list(tqdm(pool.imap(worker, records), total=len(records)))
    Path(args.fpath_out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.fpath_out, index=False)
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
    parser.add_argument("--optax-steps", type=int, default=2000)
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
    if args.mode == "fit" and not args.sed_photometry_path:
        parser.error("--sed-photometry-path is required for --mode fit.")
    if not (args.fpath_in or args.filter_object_id):
        parser.error("--fpath-in or --filter_object_id is required.")
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
