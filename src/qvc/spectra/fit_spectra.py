#!/usr/bin/env python3
"""
Fit SDSS quasar spectra with jaxqsofit.

This module runs the current spectra workflow:

1. read the quasar sample from the input HDF5 file,
2. match objects to DR16Q,
3. download/cache the SDSS spectra,
4. run one jaxqsofit fit per object,
5. save a flat summary CSV plus the native jaxqsofit outputs.

There is no grid of fits and no collect/select stage.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import traceback
from collections.abc import Mapping
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from speclite import filters as speclite_filters
from tqdm import tqdm

num_cores = os.environ.get("NUM_CORES", max((os.cpu_count() or 2) - 2, 1))
try:
    num_cores = int(num_cores)
except ValueError:
    print(f"Invalid NUM_CORES value '{num_cores}', falling back to os.cpu_count().")
    num_cores = os.cpu_count() or 1

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_cores}"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from qvc.hubble.hubble_utils import match_radec, read_quasars_from_hdf5_flat, resolve_qvc_data_path
from jaxqsofit import (
    BALConfig,
    ContinuumConfig,
    FitConfig,
    HostConfig,
    InferenceConfig,
    JAXQSOFit,
    LineConfig,
    Observation,
    OutputConfig,
    PreprocessingConfig,
    PSFPhotometryData,
    SpectroscopyData,
    build_default_prior_config,
)
from jaxqsofit.custom_components import normalize_custom_line_components
from jaxqsofit.model import (
    _broad_line_mask,
    _evaluate_custom_line_component_jax,
    _extract_line_table_from_prior_config,
    _many_gauss_lnlam,
    build_tied_line_meta_from_linelist,
    reconstruct_posterior_components,
)


COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
SDSS_BANDS = ("u", "g", "r", "i", "z")
SDSS_CAL_BANDS = ("u", "g", "r", "i")
_SDSS_FILTER_CACHE = None


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def sym_percentile(x, p=[16, 50, 84], axis=0):
    lower, median, upper = np.percentile(x, p, axis=axis)
    err = 0.5 * (upper - lower)   # optional symmetric equivalent
    err_lower = median - lower
    err_upper = upper - median
    return median, err, err_lower, err_upper

def safe_float(x, default=np.nan):
    try:
        x = np.asarray(x).squeeze()
        return float(x)
    except Exception:
        return default


def normalize_object_id(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return text[:-2] if text.endswith(".0") else text


def coerce_scalar(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (str, bool, int, float)):
        return x
    try:
        arr = np.asarray(x)
        if arr.ndim == 0 or arr.size == 1:
            return arr.reshape(-1)[0].item()
    except Exception:
        pass
    return None


def serialize_any(x):
    if x is None:
        return None
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (str, bool, int, float)):
        return x
    if isinstance(x, dict):
        return {str(k): serialize_any(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [serialize_any(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    try:
        arr = np.asarray(x)
        return arr.tolist()
    except Exception:
        return repr(x)


def prior_config_for_fit_config(prior_config):
    """
    Normalize prior configs for current jaxqsofit FitConfig validation.

    Older jaxqsofit versions accepted low-level model-prior mappings directly.
    Current versions expect structured PriorConfig sections, with raw model
    priors carried under "_model_priors".
    """
    if not isinstance(prior_config, Mapping):
        return prior_config
    if len(prior_config) == 0:
        return None
    structured_keys = {"continuum", "host", "lines", "feii", "psf", "student_t_df", "_model_priors"}
    if any(key in prior_config for key in structured_keys):
        return prior_config
    return {"_model_priors": dict(prior_config)}


def sdss_spec_filename(plate: int, mjd: int, fiber: int) -> str:
    return f"spec-{int(plate):04d}-{int(mjd):05d}-{int(fiber):04d}.fits"


def sdss_cache_file_path(cache_dir: str | Path, plate: int, mjd: int, fiber: int) -> Path:
    return Path(cache_dir) / sdss_spec_filename(plate=plate, mjd=mjd, fiber=fiber)


def load_sdss_spec_from_cache(cache_dir: str | Path, plate: int, mjd: int, fiber: int):
    cache_file = sdss_cache_file_path(cache_dir=cache_dir, plate=plate, mjd=mjd, fiber=fiber)
    if cache_file.exists():
        return fits.open(cache_file, memmap=False)
    return None


def sdss_bands_affected_by_lya(z, buffer=0.0):
    """
    SDSS ugriz bands whose rest-frame blue edge is below Ly-alpha.
    """
    edges_obs = {
        "u": (3055.11, 4030.64),
        "g": (3797.64, 5553.04),
        "r": (5418.23, 6994.42),
        "i": (6692.41, 8400.32),
        "z": (7964.70, 10873.33),
    }

    cutoff = 1216.0 + float(buffer)
    affected = []
    for band, (lo_obs, _hi_obs) in edges_obs.items():
        if (lo_obs / (1.0 + float(z))) < cutoff:
            affected.append(band)
    return affected


def get_sdss_filters():
    global _SDSS_FILTER_CACHE
    if _SDSS_FILTER_CACHE is None:
        filters = speclite_filters.load_filters(*[f"sdss2010-{b}" for b in SDSS_BANDS])
        _SDSS_FILTER_CACHE = {band: filt for band, filt in zip(SDSS_BANDS, filters)}
    return _SDSS_FILTER_CACHE


def get_filter_wavelength_angstrom(filt):
    """Return filter wavelengths as a float array in Angstrom."""
    wave = filt.wavelength
    if hasattr(wave, "to_value"):
        return np.asarray(wave.to_value(u.AA), dtype=float)
    return np.asarray(wave, dtype=float)


def build_psf_photometry_inputs(rec):
    """
    Build PSF-photometry inputs for jaxqsofit from mean-corrected multiband values.
    """
    z = safe_float(rec.get("z"))
    dropped_bands = set(sdss_bands_affected_by_lya(z)) if np.isfinite(z) else set()

    psf_bands_all = []
    psf_mags_all = []
    psf_mag_errs_all = []

    for band in SDSS_CAL_BANDS:
        if band in dropped_bands:
            continue

        mag = safe_float(rec.get(f"mean_corrected_{band}"))
        if not np.isfinite(mag):
            continue

        mag_err = safe_float(rec.get(f"mean_{band}_err"))
        if not (np.isfinite(mag_err) and mag_err > 0.0):
            continue

        print(
            f"Band: {band} -- LC Mean: {rec.get(f'lc_mean_{band}')}, "
            f"Fit Mean: {rec.get(f'mean_{band}')}, Corrected Mag: {mag}"
        )

        psf_bands_all.append(band)
        psf_mags_all.append(float(mag))
        psf_mag_errs_all.append(float(mag_err))

    return psf_bands_all, psf_mags_all, psf_mag_errs_all


def _draw_vector(values, *, n_use, fill_value):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.full(n_use, fill_value, dtype=float)
    if arr.size >= n_use:
        return arr[:n_use]
    pad_value = float(np.nanmedian(arr)) if np.any(np.isfinite(arr)) else fill_value
    return np.pad(arr, (0, n_use - arr.size), mode="constant", constant_values=pad_value)


def _draw_matrix(values, *, n_use, n_cols, fill_value=0.0):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.full((n_use, n_cols), float(arr), dtype=float)
    if arr.ndim == 1:
        row = np.full((n_cols,), fill_value, dtype=float)
        m = min(n_cols, arr.size)
        row[:m] = arr[:m]
        return np.repeat(row[None, :], n_use, axis=0)
    if arr.shape[0] >= n_use:
        arr = arr[:n_use]
    else:
        pad_row = np.nanmedian(arr, axis=0) if np.any(np.isfinite(arr)) else np.full((arr.shape[1],), fill_value, dtype=float)
        arr = np.concatenate([arr, np.repeat(pad_row[None, :], n_use - arr.shape[0], axis=0)], axis=0)
    if arr.shape[1] != n_cols:
        out = np.full((n_use, n_cols), fill_value, dtype=float)
        m = min(n_cols, arr.shape[1])
        out[:, :m] = arr[:, :m]
        arr = out
    return np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)


def _line_template_strengths(tied_line_meta):
    if tied_line_meta is None or tied_line_meta.get("n_lines", 0) == 0:
        return np.zeros((0,), dtype=float)
    fgroup = np.asarray(tied_line_meta.get("fgroup", []), dtype=int)
    flux_ratio = np.asarray(tied_line_meta.get("flux_ratio", np.ones(len(fgroup))), dtype=float)
    amp_init_group = np.asarray(
        tied_line_meta.get("amp_init_group", np.ones(int(np.max(fgroup)) + 1 if len(fgroup) else 0)),
        dtype=float,
    )
    if amp_init_group.ndim == 0:
        amp_init_group = np.asarray([float(amp_init_group)], dtype=float)
    base = np.ones(len(fgroup), dtype=float)
    valid = (fgroup >= 0) & (fgroup < amp_init_group.size)
    base[valid] = amp_init_group[fgroup[valid]]
    return np.nan_to_num(base * flux_ratio, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_take_group_values(matrix, group_ids, *, row_index, default):
    group_ids = np.asarray(group_ids, dtype=int)
    out = np.full(group_ids.shape, default, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or group_ids.size == 0:
        return out
    valid = (group_ids >= 0) & (group_ids < matrix.shape[1])
    if np.any(valid):
        out[valid] = matrix[row_index, group_ids[valid]]
    return np.nan_to_num(out, nan=default, posinf=default, neginf=default)


def _nanmedian_or_default(values, default):
    arr = np.asarray(values, dtype=float)
    good = np.isfinite(arr)
    if np.any(good):
        return float(np.nanmedian(arr[good]))
    return float(default)


def _infer_family_draw(values, family_mask, *, default, require_positive=False):
    values = np.asarray(values, dtype=float)
    family_mask = np.asarray(family_mask, dtype=bool)
    if values.ndim != 2 or values.shape[1] != family_mask.size or not np.any(family_mask):
        return np.full(values.shape[0] if values.ndim == 2 else 0, float(default), dtype=float)
    family_vals = values[:, family_mask]
    valid = np.isfinite(family_vals)
    if require_positive:
        valid &= family_vals > 0.0
    per_draw = np.full(family_vals.shape[0], np.nan, dtype=float)
    any_valid = np.any(valid, axis=1)
    if np.any(any_valid):
        per_draw[any_valid] = np.nanmedian(np.where(valid[any_valid], family_vals[any_valid], np.nan), axis=1)
    fallback = _nanmedian_or_default(np.where(valid, family_vals, np.nan), default=default)
    per_draw[~np.isfinite(per_draw)] = fallback
    return np.nan_to_num(per_draw, nan=fallback, posinf=fallback, neginf=fallback)


def _reconstruct_line_psf_draws_on_wave(q, wave_out, n_use, *, return_components=False):
    samples = getattr(q, "numpyro_samples", None)
    if not isinstance(samples, dict) or len(samples) == 0:
        empty = np.zeros((n_use, len(wave_out)), dtype=float)
        if return_components:
            return {"broad": empty.copy(), "narrow": empty.copy(), "total": empty}
        return empty

    prior_config = getattr(q, "_fit_prior_config", None)
    if isinstance(prior_config, dict):
        prior_config = dict(prior_config)
    else:
        prior_config = build_default_prior_config(np.asarray(getattr(q, "flux", []), dtype=float))

    line_table = _extract_line_table_from_prior_config(prior_config)
    native_wave = np.asarray(getattr(q, "wave", []), dtype=float)
    if native_wave.ndim != 1 or native_wave.size < 2 or not np.all(np.isfinite(native_wave)):
        native_wave = np.asarray(wave_out, dtype=float)
    tied_line_meta = build_tied_line_meta_from_linelist(line_table, wave_out) if line_table is not None else None
    native_tied_line_meta = build_tied_line_meta_from_linelist(line_table, native_wave) if line_table is not None else None
    custom_line_components = normalize_custom_line_components(getattr(q, "_fit_custom_line_components", ()))

    if (tied_line_meta is None or tied_line_meta.get("n_lines", 0) == 0) and len(custom_line_components) == 0:
        empty = np.zeros((n_use, len(wave_out)), dtype=float)
        if return_components:
            return {"broad": empty.copy(), "narrow": empty.copy(), "total": empty}
        return empty

    wave_out = np.asarray(wave_out, dtype=float)
    lnwave = np.log(wave_out)
    out_broad = np.zeros((n_use, wave_out.size), dtype=float)
    out_narrow = np.zeros((n_use, wave_out.size), dtype=float)

    fit_poly = bool(getattr(q, "_fit_fit_poly", False))
    fit_poly_order = int(getattr(q, "_fit_fit_poly_order", 2))
    w0 = 0.5 * (wave_out[0] + wave_out[-1])
    x_poly = (wave_out - w0) / max(w0, 1.0)

    line_dmu_group = np.asarray(samples.get("line_dmu_group", np.zeros((n_use, 0))), dtype=float)
    line_sig_group = np.asarray(samples.get("line_sig_group", np.zeros((n_use, 0))), dtype=float)
    line_amp_group = np.asarray(samples.get("line_amp_group", np.zeros((n_use, 0))), dtype=float)
    if line_dmu_group.ndim == 1:
        line_dmu_group = line_dmu_group[:, None]
    if line_sig_group.ndim == 1:
        line_sig_group = line_sig_group[:, None]
    if line_amp_group.ndim == 1:
        line_amp_group = line_amp_group[:, None]
    if line_dmu_group.shape[0] < n_use:
        line_dmu_group = _draw_matrix(line_dmu_group, n_use=n_use, n_cols=line_dmu_group.shape[1] if line_dmu_group.ndim == 2 else 0)
    else:
        line_dmu_group = line_dmu_group[:n_use]
    if line_sig_group.shape[0] < n_use:
        line_sig_group = _draw_matrix(line_sig_group, n_use=n_use, n_cols=line_sig_group.shape[1] if line_sig_group.ndim == 2 else 0)
    else:
        line_sig_group = line_sig_group[:n_use]
    if line_amp_group.shape[0] < n_use:
        line_amp_group = _draw_matrix(line_amp_group, n_use=n_use, n_cols=line_amp_group.shape[1] if line_amp_group.ndim == 2 else 0)
    else:
        line_amp_group = line_amp_group[:n_use]

    scale_psf_draws = _draw_vector(
        getattr(q, "pred_out", {}).get("scale_psf", getattr(q, "scale_psf", np.nan)),
        n_use=n_use,
        fill_value=1.0,
    )
    eta_psf_draws = _draw_vector(
        getattr(q, "pred_out", {}).get("eta_psf", getattr(q, "eta_psf", np.nan)),
        n_use=n_use,
        fill_value=1.0,
    )

    if tied_line_meta is not None and tied_line_meta.get("n_lines", 0) > 0:
        ln_lambda0 = np.asarray(tied_line_meta["ln_lambda0"], dtype=float)
        line_lambda = np.exp(ln_lambda0)
        broad_mask = np.asarray(_broad_line_mask(tied_line_meta.get("names", [])), dtype=float)
        is_broad = broad_mask > 0.5
        native_names = list(native_tied_line_meta.get("names", [])) if native_tied_line_meta is not None else []
        native_name_to_idx = {name: idx for idx, name in enumerate(native_names)}
        full_names = list(tied_line_meta.get("names", []))
        use_native_line = np.array(
            [
                (name in native_name_to_idx)
                and (native_wave[0] <= line_lambda[idx] <= native_wave[-1])
                for idx, name in enumerate(full_names)
            ],
            dtype=bool,
        )

        native_amps = np.zeros((n_use, len(native_names)), dtype=float)
        native_sigs = np.zeros((n_use, len(native_names)), dtype=float)
        native_dmus = np.zeros((n_use, len(native_names)), dtype=float)
        native_templates = np.zeros((len(native_names),), dtype=float)
        native_broad_mask = np.zeros((len(native_names),), dtype=bool)
        if native_tied_line_meta is not None and native_tied_line_meta.get("n_lines", 0) > 0:
            native_vgroup = np.asarray(native_tied_line_meta["vgroup"], dtype=int)
            native_wgroup = np.asarray(native_tied_line_meta["wgroup"], dtype=int)
            native_fgroup = np.asarray(native_tied_line_meta["fgroup"], dtype=int)
            native_flux_ratio = np.asarray(native_tied_line_meta["flux_ratio"], dtype=float)
            native_templates = _line_template_strengths(native_tied_line_meta)
            native_broad_mask = np.asarray(_broad_line_mask(native_names), dtype=float) > 0.5
            for i in range(n_use):
                native_dmus[i] = _safe_take_group_values(line_dmu_group, native_vgroup, row_index=i, default=0.0)
                native_sigs[i] = _safe_take_group_values(line_sig_group, native_wgroup, row_index=i, default=0.0)
                native_amp_base = _safe_take_group_values(line_amp_group, native_fgroup, row_index=i, default=0.0)
                native_amps[i] = native_amp_base * native_flux_ratio

        family_norm = {}
        family_sig = {}
        family_dmu = {}
        for family_name, family_mask in (("broad", native_broad_mask), ("narrow", ~native_broad_mask)):
            if family_mask.size == 0 or not np.any(family_mask):
                family_norm[family_name] = np.zeros(n_use, dtype=float)
                family_sig[family_name] = np.zeros(n_use, dtype=float)
                family_dmu[family_name] = np.zeros(n_use, dtype=float)
                continue
            family_templates = native_templates[family_mask]
            ratio_draws = np.full((n_use, np.count_nonzero(family_mask)), np.nan, dtype=float)
            positive_templates = np.isfinite(family_templates) & (family_templates > 0.0)
            if np.any(positive_templates):
                ratio_draws[:, positive_templates] = native_amps[:, family_mask][:, positive_templates] / family_templates[positive_templates]
            family_norm[family_name] = _infer_family_draw(ratio_draws, np.ones(ratio_draws.shape[1], dtype=bool), default=0.0, require_positive=True)
            family_sig[family_name] = _infer_family_draw(native_sigs, family_mask, default=0.0, require_positive=True)
            family_dmu[family_name] = _infer_family_draw(native_dmus, family_mask, default=0.0, require_positive=False)

        full_templates = _line_template_strengths(tied_line_meta)
        for i in range(n_use):
            dmu = np.zeros_like(ln_lambda0)
            sigs = np.zeros_like(ln_lambda0)
            amps = np.zeros_like(ln_lambda0)
            if np.any(use_native_line):
                for full_idx in np.where(use_native_line)[0]:
                    native_idx = native_name_to_idx[full_names[full_idx]]
                    dmu[full_idx] = native_dmus[i, native_idx]
                    sigs[full_idx] = native_sigs[i, native_idx]
                    amps[full_idx] = native_amps[i, native_idx]
            fallback_mask = ~use_native_line
            if np.any(fallback_mask):
                fallback_broad = fallback_mask & is_broad
                fallback_narrow = fallback_mask & ~is_broad
                if np.any(fallback_broad):
                    amps[fallback_broad] = family_norm["broad"][i] * full_templates[fallback_broad]
                    sigs[fallback_broad] = family_sig["broad"][i]
                    dmu[fallback_broad] = family_dmu["broad"][i]
                if np.any(fallback_narrow):
                    amps[fallback_narrow] = family_norm["narrow"][i] * full_templates[fallback_narrow]
                    sigs[fallback_narrow] = family_sig["narrow"][i]
                    dmu[fallback_narrow] = family_dmu["narrow"][i]
            mus = ln_lambda0 + dmu
            line_broad = np.asarray(_many_gauss_lnlam(lnwave, amps * broad_mask, mus, sigs), dtype=float)
            line_narrow = np.asarray(_many_gauss_lnlam(lnwave, amps * (1.0 - broad_mask), mus, sigs), dtype=float)
            out_broad[i] += scale_psf_draws[i] * line_broad
            out_narrow[i] += scale_psf_draws[i] * eta_psf_draws[i] * line_narrow

    if len(custom_line_components) > 0:
        def _sample_line_value(samples_dict, key, default=0.0, draw_index=0):
            vals = np.asarray(samples_dict.get(key, np.full((n_use,), default)), dtype=float).reshape(-1)
            if vals.size == 0:
                return default
            if draw_index < vals.size:
                return float(vals[draw_index])
            return float(np.nanmedian(vals)) if np.any(np.isfinite(vals)) else default

        for i in range(n_use):
            for comp in custom_line_components:
                custom_line = np.asarray(
                    _evaluate_custom_line_component_jax(
                        wave_out,
                        samples,
                        comp,
                        lambda sdict, key, default=0.0, draw_index=i: _sample_line_value(sdict, key, default=default, draw_index=draw_index),
                    ),
                    dtype=float,
                )
                if comp.line_kind == "broad":
                    out_broad[i] += scale_psf_draws[i] * custom_line
                else:
                    out_narrow[i] += scale_psf_draws[i] * eta_psf_draws[i] * custom_line

    if fit_poly:
        poly = np.ones((n_use, wave_out.size), dtype=float)
        for k in range(1, fit_poly_order + 1):
            key = f"poly_c{k}"
            if key not in samples:
                continue
            coeff = _draw_vector(samples[key], n_use=n_use, fill_value=0.0)
            poly += coeff[:, None] * (x_poly[None, :] ** k)
        poly = np.clip(poly, 0.2, 5.0)
        out_broad *= poly
        out_narrow *= poly

    out_broad = np.nan_to_num(out_broad, nan=0.0, posinf=0.0, neginf=0.0)
    out_narrow = np.nan_to_num(out_narrow, nan=0.0, posinf=0.0, neginf=0.0)
    out_total = out_broad + out_narrow
    if return_components:
        return {"broad": out_broad, "narrow": out_narrow, "total": out_total}
    return out_total


def _prepare_psf_bandpass_fraction_inputs(q, bands, n_draws):
    """Return shared reconstructed draws needed for per-band PSF fractions."""

    z = safe_float(getattr(q, "z", np.nan))
    native_wave = np.asarray(getattr(q, "wave", []), dtype=float)
    pred_out = getattr(q, "pred_out", {}) or {}
    if (not np.isfinite(z)) or native_wave.ndim != 1 or native_wave.size < 2:
        return None
    if not hasattr(q, "reconstruct_posterior_spectrum"):
        raise RuntimeError(
            "JAXQSOFit object does not expose reconstruct_posterior_spectrum(); "
            "cannot compute bandpass PSF fractions without posterior reconstruction."
        )

    filters = get_sdss_filters()
    bands = [str(band) for band in bands if str(band) in filters]
    if len(bands) == 0:
        return None

    wave_rf_chunks = []
    filter_specs = {}
    for band in bands:
        filt = filters[band]
        filt_wave = get_filter_wavelength_angstrom(filt)
        filt_trans = np.asarray(filt.response, dtype=float)
        if filt_wave.size == 0 or not np.any(np.isfinite(filt_trans)):
            continue
        valid = np.isfinite(filt_wave) & np.isfinite(filt_trans)
        if np.count_nonzero(valid) < 2:
            continue
        filt_wave = np.asarray(filt_wave[valid], dtype=float)
        filt_trans = np.asarray(filt_trans[valid], dtype=float)
        wave_rf_band = filt_wave / (1.0 + z)
        wave_rf_chunks.append(wave_rf_band)
        filter_specs[band] = (wave_rf_band, filt_trans)

    if len(wave_rf_chunks) == 0:
        return {"bands": bands, "filter_specs": filter_specs, "empty": True}

    wave_rf_concat = np.concatenate(wave_rf_chunks)
    wave_rf = np.unique(wave_rf_concat)
    if wave_rf.size < 2:
        return {"bands": bands, "filter_specs": filter_specs, "empty": True}

    try:
        recon = q.reconstruct_posterior_spectrum(
            wave_out=wave_rf,
            n_draws=n_draws,
            return_components=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "reconstruct_posterior_spectrum() failed while computing bandpass "
            f"PSF fractions: {exc}"
        ) from exc

    wave_rf = np.asarray(recon["wave"], dtype=float)
    recon_draws = recon["draws"]
    pl_draws = np.nan_to_num(np.asarray(recon_draws["PL"], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    host_draws = np.nan_to_num(
        np.asarray(recon_draws.get("host", np.zeros_like(pl_draws)), dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    agn_cont_draws = np.zeros_like(pl_draws)
    for key, draws in recon_draws.items():
        if key in {"host", "continuum"}:
            continue
        agn_cont_draws = agn_cont_draws + np.nan_to_num(
            np.asarray(draws, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
    n_use = pl_draws.shape[0]

    scale_psf_draws = _draw_vector(
        pred_out.get("scale_psf", getattr(q, "scale_psf", np.nan)),
        n_use=n_use,
        fill_value=1.0,
    )
    eta_psf_draws = _draw_vector(
        pred_out.get("eta_psf", getattr(q, "eta_psf", np.nan)),
        n_use=n_use,
        fill_value=1.0,
    )

    pl_psf_draws = scale_psf_draws[:, None] * pl_draws
    agn_psf_draws = scale_psf_draws[:, None] * agn_cont_draws
    host_psf_draws = scale_psf_draws[:, None] * eta_psf_draws[:, None] * host_draws
    try:
        line_psf_draws = _reconstruct_line_psf_draws_on_wave(
            q, wave_rf, n_use=n_use, return_components=True
        )
    except TypeError:
        legacy_line_psf_draws = _reconstruct_line_psf_draws_on_wave(q, wave_rf, n_use=n_use)
        line_psf_draws = {
            "broad": np.zeros_like(legacy_line_psf_draws, dtype=float),
            "narrow": np.zeros_like(legacy_line_psf_draws, dtype=float),
            "total": np.asarray(legacy_line_psf_draws, dtype=float),
        }

    return {
        "bands": bands,
        "filter_specs": filter_specs,
        "wave_rf": wave_rf,
        "pl_psf_draws": pl_psf_draws,
        "agn_psf_draws": agn_psf_draws,
        "host_psf_draws": host_psf_draws,
        "line_broad_psf_draws": line_psf_draws["broad"],
        "line_narrow_psf_draws": line_psf_draws["narrow"],
        "line_total_psf_draws": line_psf_draws["total"],
        "empty": False,
    }


def estimate_pl_psf_bandpass_fractions(q, bands=SDSS_BANDS, n_draws=128):
    """Return spectra-derived PSF PL/total fractions through each SDSS bandpass."""

    shared = _prepare_psf_bandpass_fraction_inputs(q, bands=bands, n_draws=n_draws)
    if shared is None:
        return {}
    bands = shared["bands"]
    if shared["empty"]:
        return {band: (np.nan, np.nan) for band in bands}

    wave_rf = shared["wave_rf"]
    filter_specs = shared["filter_specs"]
    pl_psf_draws = shared["pl_psf_draws"]
    total_psf_draws = shared["agn_psf_draws"] + shared["host_psf_draws"] + shared["line_total_psf_draws"]
    n_use = pl_psf_draws.shape[0]

    out = {}
    for band in bands:
        if band not in filter_specs:
            out[band] = (np.nan, np.nan)
            continue
        wave_rf_band, filt_trans = filter_specs[band]
        trans = np.interp(wave_rf, wave_rf_band, filt_trans, left=0.0, right=0.0)
        if not np.any(trans > 0):
            out[band] = (np.nan, np.nan)
            continue
        num_pl = np.trapezoid(pl_psf_draws * trans[None, :] * wave_rf[None, :], wave_rf, axis=1)
        num_total = np.trapezoid(total_psf_draws * trans[None, :] * wave_rf[None, :], wave_rf, axis=1)
        frac = np.full(n_use, np.nan, dtype=float)
        good = np.isfinite(num_pl) & np.isfinite(num_total) & (num_total > 0)
        frac[good] = np.clip(num_pl[good] / num_total[good], 0.0, 1.0)
        if np.any(np.isfinite(frac)):
            median, err, _, _ = sym_percentile(frac[np.isfinite(frac)])
            out[band] = (float(median), float(err))
        else:
            out[band] = (np.nan, np.nan)
    return out


def estimate_agn_psf_bandpass_fractions(q, bands=SDSS_BANDS, n_draws=128):
    """Return spectra-derived PSF variable-AGN/total fractions through each SDSS bandpass."""

    shared = _prepare_psf_bandpass_fraction_inputs(q, bands=bands, n_draws=n_draws)
    if shared is None:
        return {}

    bands = shared["bands"]
    if shared["empty"]:
        return {band: (np.nan, np.nan) for band in bands}

    wave_rf = shared["wave_rf"]
    filter_specs = shared["filter_specs"]
    variable_agn_psf_draws = shared["agn_psf_draws"] + shared["line_broad_psf_draws"]
    total_psf_draws = variable_agn_psf_draws + shared["host_psf_draws"] + shared["line_narrow_psf_draws"]
    n_use = variable_agn_psf_draws.shape[0]

    out = {}
    for band in bands:
        if band not in filter_specs:
            out[band] = (np.nan, np.nan)
            continue
        wave_rf_band, filt_trans = filter_specs[band]
        trans = np.interp(wave_rf, wave_rf_band, filt_trans, left=0.0, right=0.0)
        if not np.any(trans > 0):
            out[band] = (np.nan, np.nan)
            continue
        num_agn = np.trapezoid(
            variable_agn_psf_draws * trans[None, :] * wave_rf[None, :],
            wave_rf,
            axis=1,
        )
        num_total = np.trapezoid(total_psf_draws * trans[None, :] * wave_rf[None, :], wave_rf, axis=1)
        frac = np.full(n_use, np.nan, dtype=float)
        good = np.isfinite(num_agn) & np.isfinite(num_total) & (num_total > 0)
        frac[good] = np.clip(num_agn[good] / num_total[good], 0.0, 1.0)
        if np.any(np.isfinite(frac)):
            median, err, _, _ = sym_percentile(frac[np.isfinite(frac)])
            out[band] = (float(median), float(err))
        else:
            out[band] = (np.nan, np.nan)
    return out


def effective_decompose_host_flag(z, requested=True):
    """Disable host decomposition for high-redshift spectra."""
    z = safe_float(z)
    return bool(requested) and (not np.isfinite(z) or z <= 1.5)


def effective_fit_bal_flag(z):
    """Enable BAL components only for high-redshift spectra."""
    z = safe_float(z)
    return bool(np.isfinite(z) and z > 2.0)


def effective_fit_bc_flag(z, requested=True):
    """Disable Balmer-continuum fitting for high-redshift spectra."""
    z = safe_float(z)
    return bool(requested) and (not np.isfinite(z) or z <= 1.5)


def estimate_m2500_from_model(q):
    """Estimate reddened and intrinsic apparent mags at rest-frame 2500A from PL draws."""
    if not hasattr(q, "numpyro_samples") or q.numpyro_samples is None:
        return np.nan, np.nan, np.nan, np.nan

    s = q.numpyro_samples

    if "PL_norm" not in s or "PL_slope" not in s:
        return np.nan, np.nan, np.nan, np.nan

    pl_norm = np.asarray(s["PL_norm"], dtype=float).reshape(-1)
    pl_slope = np.asarray(s["PL_slope"], dtype=float).reshape(-1)
    if pl_norm.size == 0 or pl_slope.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    n = min(pl_norm.size, pl_slope.size)
    pl_norm = pl_norm[:n]
    pl_slope = pl_slope[:n]

    if "scale_psf" in s:
        scale_psf = np.asarray(s["scale_psf"], dtype=float).reshape(-1)
        if scale_psf.size == 1:
            scale_psf = np.full((n,), float(scale_psf[0]), dtype=float)
        else:
            scale_psf = scale_psf[:n]
    else:
        scale_psf = np.full_like(pl_norm, float(getattr(q, "scale_psf", np.nan)), dtype=float)
        if not np.isfinite(scale_psf).any():
            scale_psf = np.ones_like(pl_norm, dtype=float)

    if "reddening_ebv" in s:
        reddening_ebv = np.asarray(s["reddening_ebv"], dtype=float).reshape(-1)
        if reddening_ebv.size == 1:
            reddening_ebv = np.full((n,), float(reddening_ebv[0]), dtype=float)
        else:
            reddening_ebv = reddening_ebv[:n]
    else:
        reddening_ebv = np.zeros((n,), dtype=float)

    prior_config = getattr(q, "_fit_prior_config", {}) or {}
    pl_pivot = float(np.asarray(prior_config.get("PL_pivot", np.nan), dtype=float))
    if not np.isfinite(pl_pivot) or pl_pivot <= 0.0:
        wave = np.asarray(getattr(q, "wave", []), dtype=float)
        if wave.size > 0 and np.all(np.isfinite(wave)):
            pl_pivot = float(0.5 * (wave[0] + wave[-1]))
        else:
            pl_pivot = 2500.0

    # NumPy equivalent of jaxqsofit.model._smc_like_reddening_jax at 2500A.
    reddening_uv_ref = float(prior_config.get("reddening_uv_ref", 2500.0))
    reddening_alpha = float(prior_config.get("reddening_alpha", 1.2))
    reddening_uv_ref = max(reddening_uv_ref, 1e-8)
    k_lambda_2500 = (2500.0 / reddening_uv_ref) ** (-reddening_alpha)
    reddening_atten_2500 = 10.0 ** (-0.4 * np.maximum(reddening_ebv, 0.0) * k_lambda_2500)

    f_lambda_2500_intrinsic = scale_psf * pl_norm * (2500.0 / pl_pivot) ** pl_slope
    f_lambda_2500_reddened = f_lambda_2500_intrinsic * reddening_atten_2500

    c_A_s = 2.99792458e18
    f_nu_reddened = (f_lambda_2500_reddened * 1e-17) * (2500.0**2) / c_A_s
    f_nu_intrinsic = (f_lambda_2500_intrinsic * 1e-17) * (2500.0**2) / c_A_s

    valid_reddened = np.isfinite(f_nu_reddened) & (f_nu_reddened > 0.0)
    valid_intrinsic = np.isfinite(f_nu_intrinsic) & (f_nu_intrinsic > 0.0)

    if np.any(valid_reddened):
        m_2500_samples = -2.5 * np.log10(f_nu_reddened[valid_reddened]) - 48.60
        m50, m_err, _, _ = sym_percentile(m_2500_samples)
    else:
        m50, m_err = np.nan, np.nan

    if np.any(valid_intrinsic):
        m_2500_intrinsic_samples = -2.5 * np.log10(f_nu_intrinsic[valid_intrinsic]) - 48.60
        m50_intrinsic, m_err_intrinsic, _, _ = sym_percentile(m_2500_intrinsic_samples)
    else:
        m50_intrinsic, m_err_intrinsic = np.nan, np.nan

    return m50, m_err, m50_intrinsic, m_err_intrinsic


def posterior_component_fraction_at_wave(q, numerator_key, denominator_key, wave0):
    """Return posterior median/error for a model-component fraction at one wavelength."""
    wave = np.asarray(getattr(q, "wave", []), dtype=float)
    if wave.ndim != 1 or wave.size == 0 or not np.all(np.isfinite(wave)):
        return np.nan, np.nan

    pred_out = getattr(q, "pred_out", None)
    if not isinstance(pred_out, dict):
        return np.nan, np.nan

    numerator = pred_out.get(numerator_key)
    if isinstance(denominator_key, (tuple, list)):
        denominator = None
        for key in denominator_key:
            value = pred_out.get(key)
            if value is None:
                return np.nan, np.nan
            value = np.asarray(value, dtype=float)
            denominator = value if denominator is None else denominator + value
    else:
        denominator = pred_out.get(denominator_key)
    if numerator is None or denominator is None:
        return np.nan, np.nan

    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    if numerator.ndim == 1:
        numerator = numerator[None, :]
    if denominator.ndim == 1:
        denominator = denominator[None, :]

    if numerator.shape != denominator.shape or numerator.shape[-1] != wave.size:
        return np.nan, np.nan

    idx = int(np.argmin(np.abs(wave - float(wave0))))
    numerator_vals = numerator[:, idx]
    denominator_vals = denominator[:, idx]
    frac = np.divide(
        numerator_vals,
        denominator_vals,
        out=np.full(numerator_vals.shape, np.nan, dtype=float),
        where=np.isfinite(numerator_vals) & np.isfinite(denominator_vals) & (denominator_vals != 0.0),
    )
    good = np.isfinite(frac)
    if not np.any(good):
        return np.nan, np.nan

    median, err, _, _ = sym_percentile(frac[good])
    return safe_float(median), safe_float(err)


def posterior_component_integrated_fraction(q, numerator_key, denominator_key, *, positive_only=True):
    """Return posterior median/error for a ratio of integrated model components."""
    wave = np.asarray(getattr(q, "wave", []), dtype=float)
    if wave.ndim != 1 or wave.size < 2 or not np.all(np.isfinite(wave)):
        return np.nan, np.nan

    pred_out = getattr(q, "pred_out", None)
    if not isinstance(pred_out, dict):
        return np.nan, np.nan

    numerator = pred_out.get(numerator_key)
    denominator = pred_out.get(denominator_key)
    if numerator is None or denominator is None:
        return np.nan, np.nan

    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    if numerator.ndim == 1:
        numerator = numerator[None, :]
    if denominator.ndim == 1:
        denominator = denominator[None, :]

    if numerator.shape != denominator.shape or numerator.shape[-1] != wave.size:
        return np.nan, np.nan

    if positive_only:
        numerator = np.where(np.isfinite(numerator), np.maximum(numerator, 0.0), np.nan)
        denominator = np.where(np.isfinite(denominator), np.maximum(denominator, 0.0), np.nan)

    numerator_int = np.trapezoid(numerator, wave, axis=1)
    denominator_int = np.trapezoid(denominator, wave, axis=1)
    frac = np.divide(
        numerator_int,
        denominator_int,
        out=np.full(numerator_int.shape, np.nan, dtype=float),
        where=np.isfinite(numerator_int) & np.isfinite(denominator_int) & (denominator_int > 0.0),
    )
    good = np.isfinite(frac)
    if not np.any(good):
        return np.nan, np.nan

    median, err, _, _ = sym_percentile(frac[good])
    return safe_float(median), safe_float(err)


def reconstructed_component_fraction_at_wave(
    q,
    component_key,
    reference_key,
    wave0,
    *,
    apply_poly=False,
    n_draws=None,
):
    """Return a posterior component/reference fraction from a tiny reconstructed window."""
    wave_native = np.asarray(getattr(q, "wave", []), dtype=float)
    if wave_native.ndim != 1 or wave_native.size < 2 or not np.all(np.isfinite(wave_native)):
        return np.nan, np.nan

    dw = float(np.nanmedian(np.diff(wave_native)))
    if not np.isfinite(dw) or dw <= 0.0:
        dw = max(abs(float(wave0)) * 1.0e-3, 1.0e-3)
    delta = max(dw, abs(float(wave0)) * 1.0e-3, 1.0e-3)
    wave_out = np.array([float(wave0) - delta, float(wave0), float(wave0) + delta], dtype=float)
    if wave_out[0] <= 0.0:
        eps = max(abs(float(wave0)) * 1.0e-4, 1.0e-3)
        wave_out = np.array([max(float(wave0) - eps, 1.0e-6), float(wave0), float(wave0) + eps], dtype=float)

    prior_config = getattr(q, "_fit_prior_config", None)
    if isinstance(prior_config, dict):
        prior_config = dict(prior_config)
    else:
        prior_config = build_default_prior_config(np.asarray(getattr(q, "flux", []), dtype=float))

    pl_pivot = safe_float(prior_config.get("PL_pivot", np.nan))
    if not np.isfinite(pl_pivot) or pl_pivot <= 0.0:
        prior_config["PL_pivot"] = float(0.5 * (wave_native[0] + wave_native[-1]))

    age_grid_gyr = getattr(q, "_fit_fsps_age_grid", None)
    logzsol_grid = getattr(q, "_fit_fsps_logzsol_grid", None)
    if age_grid_gyr is None or logzsol_grid is None:
        fsps_grid = getattr(q, "fsps_grid", None)
        age_grid_gyr = getattr(fsps_grid, "age_grid_gyr", None)
        logzsol_grid = getattr(fsps_grid, "logzsol_grid", None)
    if age_grid_gyr is None or logzsol_grid is None:
        return np.nan, np.nan

    recon = reconstruct_posterior_components(
        wave_out=wave_out,
        samples=getattr(q, "numpyro_samples", {}),
        pred_out=getattr(q, "pred_out", None),
        age_grid_gyr=age_grid_gyr,
        logzsol_grid=logzsol_grid,
        dsps_ssp_fn=getattr(q, "_fit_dsps_ssp_fn", "tempdata.h5"),
        prior_config=prior_config,
        fit_poly=bool(apply_poly) and bool(getattr(q, "_fit_fit_poly", False)),
        fit_reddening=bool(getattr(q, "_fit_fit_reddening", False)),
        fit_poly_order=int(getattr(q, "_fit_fit_poly_order", 2)),
        fe_uv_wave=np.asarray(getattr(q, "fe_uv_wave", []), dtype=float),
        fe_uv_flux=np.asarray(getattr(q, "fe_uv_flux", []), dtype=float),
        fe_op_wave=np.asarray(getattr(q, "fe_op_wave", []), dtype=float),
        fe_op_flux=np.asarray(getattr(q, "fe_op_flux", []), dtype=float),
        custom_components=getattr(q, "_fit_custom_components", ()),
        n_draws=n_draws,
        return_components=True,
    )
    if component_key not in recon["draws"] or reference_key not in recon["draws"]:
        return np.nan, np.nan

    idx = int(np.argmin(np.abs(np.asarray(recon["wave"], dtype=float) - float(wave0))))
    numerator_vals = np.asarray(recon["draws"][component_key], dtype=float)[:, idx]
    denominator_vals = np.asarray(recon["draws"][reference_key], dtype=float)[:, idx]
    frac = np.divide(
        numerator_vals,
        denominator_vals,
        out=np.full(numerator_vals.shape, np.nan, dtype=float),
        where=np.isfinite(numerator_vals) & np.isfinite(denominator_vals) & (denominator_vals != 0.0),
    )
    good = np.isfinite(frac)
    if not np.any(good):
        return np.nan, np.nan

    median, err, _, _ = sym_percentile(frac[good])
    return safe_float(median), safe_float(err)


def estimate_host_center_fraction(q):
    """Return the posterior host/continuum fraction at the spectrum midpoint."""
    wave = np.asarray(getattr(q, "wave", []), dtype=float)
    if wave.ndim != 1 or wave.size == 0 or not np.all(np.isfinite(wave)):
        return np.nan, np.nan

    lam_obs = np.asarray(getattr(q, "lam", []), dtype=float)
    if lam_obs.ndim == 1 and lam_obs.size == wave.size and np.all(np.isfinite(lam_obs)):
        z = safe_float(getattr(q, "z", np.nan), np.nan)
        z_scale = max(1.0 + z, 1e-8) if np.isfinite(z) else 1.0
        wave_center = float(0.5 * (lam_obs[0] + lam_obs[-1])) / z_scale
    else:
        wave_center = float(0.5 * (wave[0] + wave[-1]))

    median, err = posterior_component_fraction_at_wave(
        q,
        numerator_key="gal_model",
        denominator_key="continuum_model",
        wave0=wave_center,
    )
    if np.isfinite(median):
        return median, err

    fallback = safe_float(getattr(q, "frac_host_pivot", np.nan))
    if np.isfinite(fallback):
        return fallback, np.nan
    return np.nan, np.nan


def estimate_host_2500_fraction(q):
    """Return host/continuum at rest-frame 2500 A from posterior components."""
    wave0 = 2500.0
    wave = np.asarray(getattr(q, "wave", []), dtype=float)
    if wave.ndim == 1 and wave.size > 0 and np.all(np.isfinite(wave)) and (wave[0] <= wave0 <= wave[-1]):
        median, err = posterior_component_fraction_at_wave(
            q,
            numerator_key="gal_model",
            denominator_key="continuum_model",
            wave0=wave0,
        )
        if np.isfinite(median):
            return median, err

    try:
        median, err = reconstructed_component_fraction_at_wave(
            q,
            component_key="host",
            reference_key="continuum",
            wave0=wave0,
            apply_poly=False,
        )
        if np.isfinite(median):
            return median, err
    except Exception:
        pass

    fallback = safe_float(getattr(q, "frac_host_2500", np.nan))
    if np.isfinite(fallback) and fallback >= 0.0:
        return fallback, np.nan
    return np.nan, np.nan


def estimate_host_psf_2500_fraction(q):
    """Return PSF-space host/(AGN+host) at rest-frame 2500 A from posterior components."""
    wave0 = 2500.0
    wave = np.asarray(getattr(q, "wave", []), dtype=float)
    if wave.ndim == 1 and wave.size > 0 and np.all(np.isfinite(wave)) and (wave[0] <= wave0 <= wave[-1]):
        median, err = posterior_component_fraction_at_wave(
            q,
            numerator_key="gal_model_psf",
            denominator_key=("agn_model_psf", "gal_model_psf"),
            wave0=wave0,
        )
        if np.isfinite(median):
            return median, err

    fallback = safe_float(getattr(q, "frac_host_psf_2500", np.nan))
    if np.isfinite(fallback) and fallback >= 0.0:
        return fallback, np.nan
    return np.nan, np.nan


def estimate_pl_2500_fraction(q):
    """Return power-law/total-model at rest-frame 2500 A from posterior components."""
    wave0 = 2500.0
    wave = np.asarray(getattr(q, "wave", []), dtype=float)
    if wave.ndim == 1 and wave.size > 0 and np.all(np.isfinite(wave)) and (wave[0] <= wave0 <= wave[-1]):
        median, err = posterior_component_fraction_at_wave(
            q,
            numerator_key="f_pl_model",
            denominator_key="model",
            wave0=wave0,
        )
        if np.isfinite(median):
            return median, err

    # No reconstruction fallback here: the available reconstruction helper only
    # rebuilds continuum components, so it cannot preserve the requested
    # "PL / total spectrum (including lines)" definition outside native coverage.
    return np.nan, np.nan


def _format_value_with_err(value, err):
    value = safe_float(value)
    err = safe_float(err)
    if not np.isfinite(value):
        return None
    if np.isfinite(err):
        return f"{value:.4f} +/- {err:.4f}"
    return f"{value:.4f}"


def format_spectrum_diagnostics(result):
    lines = []
    object_id = str(result.get("object_id", "")).strip() or "unknown"
    lines.append(f"Spectrum diagnostics for object_id={object_id}")

    context_parts = []
    sdss_name = str(result.get("sdss_name", "")).strip()
    if sdss_name:
        context_parts.append(f"sdss_name={sdss_name}")
    z = safe_float(result.get("z"))
    if np.isfinite(z):
        context_parts.append(f"z={z:.5f}")
    bands_used = str(result.get("bands_used", "")).strip()
    if bands_used:
        context_parts.append(f"bands_used={bands_used}")
    lines.append("  Context: " + (", ".join(context_parts) if context_parts else "not available"))

    psf_constant_entries = []
    psf_pl_entries = []
    for band in SDSS_BANDS:
        value_str = _format_value_with_err(result.get(f"f_AGN_psf_{band}"), result.get(f"f_AGN_psf_{band}_err"))
        if value_str is not None:
            psf_constant_entries.append(f"    {band}: {value_str}")
        value_str = _format_value_with_err(result.get(f"f_PL_psf_{band}"), result.get(f"f_PL_psf_{band}_err"))
        if value_str is not None:
            psf_pl_entries.append(f"    {band}: {value_str}")
    lines.append("  PSF variable-AGN fractions:")
    if psf_constant_entries:
        lines.extend(psf_constant_entries)
    else:
        lines.append("    not available")
    lines.append("  PSF pure-PL fractions:")
    if psf_pl_entries:
        lines.extend(psf_pl_entries)
    else:
        lines.append("    not available")

    broader_specs = [
        ("f_PL", "f_PL_err", "f_PL"),
        ("f_host_2500", "f_host_2500_err", "f_host_2500"),
        ("frac_host_psf_2500", "frac_host_psf_2500_err", "frac_host_psf_2500"),
        ("f_host_center", "f_host_center_err", "f_host_center"),
        ("apparent_mag_2500", "apparent_mag_2500_err", "apparent_mag_2500"),
        ("apparent_mag_2500_intrinsic", "apparent_mag_2500_intrinsic_err", "apparent_mag_2500_intrinsic"),
    ]
    broader_entries = []
    for key, err_key, label in broader_specs:
        value_str = _format_value_with_err(result.get(key), result.get(err_key))
        if value_str is not None:
            broader_entries.append(f"    {label}: {value_str}")
    lines.append("  Broader diagnostics:")
    if broader_entries:
        lines.extend(broader_entries)
    else:
        lines.append("    not available")

    return "\n".join(lines)


def print_spectrum_diagnostics(result):
    print(format_spectrum_diagnostics(result))


def compute_derived_results(result, q, args):
    """
    Populate old fit_spectra-compatible columns from jaxqsofit outputs when possible.
    """
    # result["best"] = True
    # result["decomp_host"] = bool(args.decompose_host)
    # result["BC"] = bool(args.fit_bc)
    # result["poly"] = bool(args.fit_poly)
    # result["npca_qso"] = 10 if args.decompose_host else -1

    # result["redchi2_conti_full"] = safe_float(result.get("chi2_per_pixel"))
    # result["redchi"] = safe_float(result.get("chi2_per_pixel"))

    # if "PL_slope_blue" not in result:
    #     result["PL_slope_blue"] = safe_float(result.get("PL_slope"))
    # if "PL_slope_red" not in result:
    #     result["PL_slope_red"] = safe_float(result.get("PL_slope"))

    if "f_host_5100" not in result:
        result["f_host_5100"] = safe_float(result.get("frac_host_5100"))

    bi = safe_float(getattr(q, "bi", np.nan))
    bi_err = safe_float(getattr(q, "bi_err", np.nan))
    if bool(result.get("fit_bal_effective", False)) and not np.isfinite(bi):
        try:
            bi, bi_err = q.balnicity_index()
        except Exception as exc:
            if bool(getattr(args, "verbose", False)):
                print(
                    f"[WARNING] Failed to compute BAL BI for object_id={result.get('object_id')}: {exc}"
                )
            bi, bi_err = np.nan, np.nan
    result["bi"] = safe_float(bi)
    result["bi_err"] = safe_float(bi_err)
    decompose_host_eff = bool(getattr(q, "_fit_decompose_host", getattr(args, "decompose_host", True)))
    result["decompose_host_effective"] = decompose_host_eff

    f_pl_2500, f_pl_2500_err = estimate_pl_2500_fraction(q)
    result["f_PL"] = safe_float(f_pl_2500)
    result["f_PL_err"] = safe_float(f_pl_2500_err)

    for band, (median, err) in estimate_pl_psf_bandpass_fractions(q, bands=SDSS_BANDS).items():
        result[f"f_PL_psf_{band}"] = safe_float(median)
        result[f"f_PL_psf_{band}_err"] = safe_float(err)
    for band, (median, err) in estimate_agn_psf_bandpass_fractions(q, bands=SDSS_BANDS).items():
        result[f"f_AGN_psf_{band}"] = safe_float(median)
        result[f"f_AGN_psf_{band}_err"] = safe_float(err)

    if decompose_host_eff:
        f_host_2500, f_host_2500_err = estimate_host_2500_fraction(q)
        result["f_host_2500"] = safe_float(f_host_2500)
        result["f_host_2500_err"] = safe_float(f_host_2500_err)

        f_host_psf_2500, f_host_psf_2500_err = estimate_host_psf_2500_fraction(q)
        result["frac_host_psf_2500"] = safe_float(f_host_psf_2500)
        result["frac_host_psf_2500_err"] = safe_float(f_host_psf_2500_err)

        # Host/continuum fraction at the center of the fitted spectrum.
        m50, m_err = estimate_host_center_fraction(q)
        result["f_host_center"] = safe_float(m50)
        result["f_host_center_err"] = safe_float(m_err)
    else:
        result["f_host_2500"] = 0.0
        result["f_host_2500_err"] = 0.0
        result["frac_host_psf_2500"] = 0.0
        result["frac_host_psf_2500_err"] = 0.0
        result["f_host_center"] = 0.0
        result["f_host_center_err"] = 0.0

    # Narrow-line fraction, defined as integrated narrow-line flux over integrated
    # total continuum flux so it remains well-defined even without host decomposition.
    m50, m_err = posterior_component_integrated_fraction(
        q,
        numerator_key="line_model_narrow",
        denominator_key="continuum_model",
        positive_only=True,
    )
    result["f_na"] = safe_float(m50)
    result["f_na_err"] = safe_float(m_err)

    m50, m_err = posterior_component_integrated_fraction(
        q,
        numerator_key="line_model_broad",
        denominator_key="continuum_model",
        positive_only=True,
    )
    result["f_br"] = safe_float(m50)
    result["f_br_err"] = safe_float(m_err)

    # BC fraction, defined against the total continuum at 3000 A.
    m50, m_err = posterior_component_fraction_at_wave(
        q,
        numerator_key="f_bc_model",
        denominator_key="continuum_model",
        wave0=3000.0,
    )
    result["f_bc_3000"] = safe_float(m50)
    result["f_bc_3000_err"] = safe_float(m_err)

    # FeUV fraction, also defined against the total continuum at 3000 A.
    m50, m_err = posterior_component_fraction_at_wave(
        q,
        numerator_key="f_fe_mgii_model",
        denominator_key="continuum_model",
        wave0=3000.0,
    )
    result["f_fe_uv_3000"] = safe_float(m50)
    result["f_fe_uv_3000_err"] = safe_float(m_err)

    z = safe_float(result.get("z"))
    m2500 = np.nan
    m2500_err = np.nan
    m2500_intrinsic = np.nan
    m2500_intrinsic_err = np.nan

    m2500, m2500_err, m2500_intrinsic, m2500_intrinsic_err = estimate_m2500_from_model(q)

    result["apparent_mag_2500"] = m2500
    result["apparent_mag_2500_err"] = m2500_err
    result["apparent_mag_2500_intrinsic"] = m2500_intrinsic
    result["apparent_mag_2500_intrinsic_err"] = m2500_intrinsic_err


# -----------------------------------------------------------------------------
# sample building and cross-match
# -----------------------------------------------------------------------------

def load_quasar_core_list(fpath_in):
    return read_quasars_from_hdf5_flat(fpath_in)


def build_sample_df_from_object_ids(object_ids):
    requested_ids = [normalize_object_id(obj_id) for obj_id in (object_ids or [])]
    requested_ids = [obj_id for obj_id in requested_ids if obj_id]
    requested_ids = list(dict.fromkeys(requested_ids))
    if len(requested_ids) == 0:
        raise ValueError("No non-empty object_id values were provided.")

    catalog_path = resolve_qvc_data_path("data/S82/Catalog.parquet")
    cat = pd.read_parquet(catalog_path)
    required_cols = {"objectId", "RA", "DEC"}
    missing = required_cols - set(cat.columns)
    if missing:
        raise ValueError(
            f"S82 catalog at {catalog_path} is missing required column(s): {sorted(missing)}"
        )

    lookup = (
        cat.loc[:, ["objectId", "RA", "DEC"]]
        .dropna(subset=["objectId", "RA", "DEC"])
        .assign(object_id=lambda d: d["objectId"].map(normalize_object_id))
    )
    lookup = lookup[lookup["object_id"] != ""]
    lookup = (
        lookup.drop_duplicates(subset=["object_id"], keep="first")
        .set_index("object_id")
        .loc[:, ["RA", "DEC"]]
    )

    selected = lookup.reindex(requested_ids)
    missing_ids = [
        oid
        for oid, ra, dec in zip(requested_ids, selected["RA"].tolist(), selected["DEC"].tolist())
        if not (np.isfinite(safe_float(ra)) and np.isfinite(safe_float(dec)))
    ]
    if missing_ids:
        preview = ", ".join(missing_ids[:20])
        raise ValueError(
            "Could not resolve RA/DEC from S82 Catalog.parquet for "
            f"{len(missing_ids)} object_id(s). First missing: {preview}"
        )

    return pd.DataFrame(
        {
            "object_id": requested_ids,
            "ra": selected["RA"].astype(float).to_numpy(),
            "dec": selected["DEC"].astype(float).to_numpy(),
        }
    )


def _psf_calibration_bands_for_z(z):
    dropped_bands = set(sdss_bands_affected_by_lya(z)) if np.isfinite(safe_float(z)) else set()
    return [band for band in SDSS_CAL_BANDS if band not in dropped_bands]


def _format_row_identity(row):
    object_id = normalize_object_id(row.get("object_id_norm", row.get("object_id")))
    sdss_name = str(row.get("sdss_name", "")).strip()
    if sdss_name:
        return f"object_id={object_id}, sdss_name={sdss_name}"
    return f"object_id={object_id}"


def load_lc_mean_by_object_id(object_ids):
    from qvc.light_curve.multiband_generate_lc import concat_light_curves

    requested_ids = [normalize_object_id(obj_id) for obj_id in object_ids]
    requested_ids = [obj_id for obj_id in requested_ids if obj_id]
    if len(requested_ids) == 0:
        return {}

    # concat_light_curves intersects caller-provided ids with S82 integer ids.
    # Pass both string and int forms to maximize matching robustness.
    filter_values = []
    for obj_id in requested_ids:
        filter_values.append(obj_id)
        try:
            filter_values.append(int(obj_id))
        except Exception:
            pass

    lc_objects = concat_light_curves(filter_object_ids=filter_values, progress_bar=False)
    by_object_id = {}
    for obj in lc_objects:
        norm_id = normalize_object_id(obj.get("object_id"))
        if not norm_id:
            continue
        mags_mean = obj.get("mags_mean", [])
        row = {f"lc_mean_{band}": np.nan for band in SDSS_BANDS}
        for band, mag in zip(SDSS_BANDS, mags_mean):
            mag_val = safe_float(mag)
            if np.isfinite(mag_val):
                row[f"lc_mean_{band}"] = float(mag_val)
        by_object_id[norm_id] = row
    return by_object_id


def _apply_lc_mean_correction(sample_df):
    sample_df = sample_df.copy()
    for band in SDSS_BANDS:
        for col in (f"lc_mean_{band}", f"mean_corrected_{band}"):
            if col not in sample_df.columns:
                sample_df[col] = np.nan
    object_ids = sample_df["object_id"].astype(str).tolist()
    by_object_id = load_lc_mean_by_object_id(object_ids)
    for idx, norm_id in enumerate(sample_df["object_id_norm"].tolist()):
        row = by_object_id.get(norm_id)
        if row is None:
            continue
        for band in SDSS_BANDS:
            sample_df.at[idx, f"lc_mean_{band}"] = row.get(f"lc_mean_{band}", np.nan)

    missing = []
    for idx, row in sample_df.iterrows():
        z = safe_float(row.get("z"))
        for band in _psf_calibration_bands_for_z(z):
            lc_col = f"lc_mean_{band}"
            mean_col = f"mean_{band}"
            err_col = f"mean_{band}_err"
            lc_mean = safe_float(row.get(lc_col))
            fit_mean = safe_float(row.get(mean_col))
            fit_mean_err = safe_float(row.get(err_col))
            if not np.isfinite(lc_mean):
                missing.append(f"{_format_row_identity(row)} band={band}: missing finite {lc_col} from concat_light_curves()")
            if not np.isfinite(fit_mean):
                missing.append(f"{_format_row_identity(row)} band={band}: missing finite {mean_col} from H5/input")
            if not (np.isfinite(fit_mean_err) and fit_mean_err > 0.0):
                missing.append(f"{_format_row_identity(row)} band={band}: missing positive finite {err_col} from H5/input")
            if np.isfinite(lc_mean) and np.isfinite(fit_mean):
                sample_df.at[idx, f"mean_corrected_{band}"] = float(lc_mean + fit_mean)
    if missing:
        preview = "; ".join(missing[:10])
        more = "" if len(missing) <= 10 else f"; ... {len(missing) - 10} more"
        raise ValueError(f"Cannot build PSF photometry inputs. {preview}{more}")
    return sample_df


def prepare_sample_df(
    sample_df,
    filter_sdss_name=None,
    filter_object_id=None,
    N=None,
    skip=None,
    use_h5_mean_correction=False,
):
    sample_df = sample_df.copy()
    sample_df["object_id"] = sample_df["object_id"].astype(str).str.strip()
    sample_df["object_id_norm"] = sample_df["object_id"].map(normalize_object_id)

    exclusion_sdss_names = {
        "221120.38+010905.6",  # wrong redshift
        "235133.07+005537.0",  # wrong redshift
        "024555.35+005332.6",  # weird spectrum
    }
    if "sdss_name" in sample_df.columns:
        sample_df = sample_df[~sample_df["sdss_name"].astype(str).isin(exclusion_sdss_names)]

    if filter_sdss_name is not None:
        sample_df = sample_df[sample_df["sdss_name"].astype(str).isin(filter_sdss_name)]
    if filter_object_id is not None:
        ids = [normalize_object_id(x) for x in filter_object_id]
        ids = [x for x in ids if x]
        sample_df = sample_df[sample_df["object_id_norm"].isin(ids)]

    if skip is not None:
        sample_df = sample_df.iloc[int(skip):]
    if N is not None:
        sample_df = sample_df.iloc[: int(N)]

    sample_df = sample_df.reset_index(drop=True)
    if use_h5_mean_correction:
        print(
            "--use-h5-mean-correction is deprecated for spectra PSF correction; "
            "using concat_light_curves() LC means plus H5/input mean_* offsets."
        )
    sample_df = _apply_lc_mean_correction(sample_df)

    sample_df["object_id"] = sample_df["object_id_norm"]
    sample_df = sample_df.drop(columns=["object_id_norm"])
    return sample_df



def match_to_dr16q(sample_df, dr16q_fits, max_sep_arcsec=1.0):
    cols = ["RA", "DEC", "SDSS_NAME", "PLATE", "FIBERID", "MJD", "Z_SYS", "LOGLBOL"]
    data_cat = Table.read(dr16q_fits, hdu=1)[cols].to_pandas()
    df_matched, unmatched = match_radec(
        sample_df,
        data_cat,
        populate_cols=["SDSS_NAME", "PLATE", "FIBERID", "MJD", "Z_SYS", "LOGLBOL", "RA", "DEC"],
        ra_col_a="ra",
        dec_col_a="dec",
        ra_col_b="RA",
        dec_col_b="DEC",
        max_sep_arcsec=max_sep_arcsec,
        add_prefix=False,
    )

    df_matched = df_matched.copy()
    df_matched["plate"] = df_matched["PLATE"].astype(int)
    df_matched["fiber"] = df_matched["FIBERID"].astype(int)
    df_matched["mjd"] = df_matched["MJD"].astype(int)
    df_matched["z"] = df_matched["Z_SYS"].astype(float)
    df_matched["loglbol"] = df_matched["LOGLBOL"].astype(float)
    df_matched["sdss_name"] = df_matched["SDSS_NAME"].astype(str).str.strip()
    df_matched["object_id"] = df_matched["object_id"].astype(str).str.strip()
    df_matched["ra"] = df_matched["RA"].astype(float)
    df_matched["dec"] = df_matched["DEC"].astype(float)

    print(f"Matched {len(df_matched)} objects to DR16Q. Unmatched: {len(unmatched)}")
    return df_matched



def build_records(args):
    if args.fpath_in is None:
        sample_df = build_sample_df_from_object_ids(args.filter_object_id)
    elif str(args.fpath_in).lower().endswith(".csv"):
        sample_df = pd.read_csv(args.fpath_in)
    else:
        sample_df = load_quasar_core_list(args.fpath_in)

    print("build_records filtering on: ", args.filter_object_id)
    sample_df = prepare_sample_df(
        sample_df,
        filter_sdss_name=args.filter_sdss_name,
        filter_object_id=args.filter_object_id,
        N=args.N,
        skip=args.skip,
        use_h5_mean_correction=bool(args.use_h5_mean_correction),
    )
    print(f"Sample size after filtering: {len(sample_df)}")
    if len(sample_df) > 0:
        print("First few filtered object_ids:", sample_df["object_id"].head().tolist())

    df_matched = match_to_dr16q(sample_df, args.dr16q_fits, args.max_sep)
    return [row.to_dict() for _, row in df_matched.iterrows()]


def fetch_dustmaps(args):
    import dustmaps.sfd
    dustmaps.sfd.fetch()

# -----------------------------------------------------------------------------
# SDSS spectrum cache
# -----------------------------------------------------------------------------

def fetch_spectrum_fits(args, plate, fiber, mjd, cache_dir="data/spectra_cache"):

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = sdss_cache_file_path(
        cache_dir=cache_dir,
        plate=int(plate),
        mjd=int(mjd),
        fiber=int(fiber),
    )
    spec_file = sdss_spec_filename(plate=int(plate), mjd=int(mjd), fiber=int(fiber))

    if cache_file.exists():
        return fits.open(cache_file, memmap=False)
    elif args.mode != "download":
        raise RuntimeError(f"Cache file {cache_file} ({spec_file}) not found in non-download mode.")

    from astroquery.sdss import SDSS
    spec = SDSS.get_spectra(plate=int(plate), fiberID=int(fiber), mjd=int(mjd))
    if spec is None or len(spec) == 0:
        raise RuntimeError(f"No SDSS spectrum found for {spec_file}")

    hdul = spec[0]
    hdul.writeto(cache_file, overwrite=True)
    return fits.open(cache_file, memmap=False)



def load_spec_from_cache(plate, fiber, mjd, cache_dir="data/spectra_cache"):
    return load_sdss_spec_from_cache(
        cache_dir=cache_dir,
        plate=int(plate),
        mjd=int(mjd),
        fiber=int(fiber),
    )



def get_spectrum_arrays(hdul):
    tb = hdul[1].data
    lam = np.asarray(10 ** tb["loglam"], dtype=float)
    flux = np.asarray(tb["flux"], dtype=float)

    ivar = np.asarray(tb["ivar"], dtype=float)
    err = np.full_like(flux, np.nan, dtype=float)
    good_ivar = np.isfinite(ivar) & (ivar > 0)
    err[good_ivar] = 1.0 / np.sqrt(ivar[good_ivar])

    good = np.isfinite(lam) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    return lam[good], flux[good], err[good]


# -----------------------------------------------------------------------------
# saving
# -----------------------------------------------------------------------------


def extract_named_results(q):
    out = {}

    if hasattr(q, "conti_result_name") and hasattr(q, "conti_result"):
        for name, value in zip(q.conti_result_name, q.conti_result):
            out[str(name)] = coerce_scalar(value)

    if hasattr(q, "line_result_name") and hasattr(q, "line_result"):
        for name, value in zip(q.line_result_name, q.line_result):
            out[str(name)] = coerce_scalar(value)

    return out



def extract_scalar_attrs(q):
    out = {}
    for key, value in q.__dict__.items():
        if key.startswith("_"):
            continue
        if key in out:
            continue
        scalar = coerce_scalar(value)
        if scalar is not None:
            out[str(key)] = scalar
    return out



def extract_fit_stats(q):
    out = {
        "chi2": np.nan,
        "chi2_per_pixel": np.nan,
        "wrms": np.nan,
        "n_pixels": 0,
        "wave_min_rf": np.nan,
        "wave_max_rf": np.nan,
    }

    resid = np.asarray(q.flux) - np.asarray(q.model_total)
    sigma = np.asarray(q.err)
    mask = np.asarray(q.wave, dtype=float) >= 1215.67
    resid = resid[mask]

    s = q.numpyro_samples
    frac_j = safe_float(np.median(np.asarray(s.get("frac_jitter", 0.0))), 0.0)
    add_j = safe_float(np.median(np.asarray(s.get("add_jitter", 0.0))), 0.0)
    sigma = np.sqrt(sigma**2 + (frac_j * np.abs(np.asarray(q.model_total))) ** 2 + add_j**2)

    good = np.isfinite(resid) & np.isfinite(sigma) & (sigma > 0)

    z = resid[good] / sigma[good]
    out["wrms"] = float(np.sqrt(np.mean(z**2)))

    out["chi2"] = float(np.sum(z**2))
    out["chi2_per_pixel"] = float(np.mean(z**2))
    out["n_pixels"] = int(np.sum(good))

    out["wave_min_rf"] = safe_float(np.min(q.wave))
    out["wave_max_rf"] = safe_float(np.max(q.wave))

    return out


# -----------------------------------------------------------------------------
# fitting
# -----------------------------------------------------------------------------

def run_one_fit(rec, args):
    result = {
        "object_id": str(rec["object_id"]),
        "sdss_name": str(rec["sdss_name"]),
        "plate": int(rec["plate"]),
        "fiber": int(rec["fiber"]),
        "mjd": int(rec["mjd"]),
        "z": float(rec["z"]),
        "ra": float(rec["ra"]),
        "dec": float(rec["dec"]),
        "loglbol": safe_float(rec.get("loglbol")),
        "fit_ok": False,
        "error_message": "",
        "delta_mag_u": -1e9,
        "delta_mag_g": -1e9,
        "delta_mag_r": -1e9,
        "delta_mag_i": -1e9,
        "delta_mag_z": -1e9,
        "mag_synth_u": -1e9,
        "mag_synth_g": -1e9,
        "mag_synth_r": -1e9,
        "mag_synth_i": -1e9,
        "mag_synth_z": -1e9,
        "mean_corrected_u": -1e9,
        "mean_corrected_g": -1e9,
        "mean_corrected_r": -1e9,
        "mean_corrected_i": -1e9,
        "mean_corrected_z": -1e9,
        "delta_m_flux_recal": 0.0,
        "sigma_dm": 0.0,
        "dm_i": 0.0,
        "flux_scale": 1.0,
        "bands_used": "",
        "numpyro_sample_count": 0,
        "bi": np.nan,
        "bi_err": np.nan,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    result["result_dir"] = args.output_dir
    os.makedirs(args.fig_dir, exist_ok=True)
    result["fig_dir"] = args.fig_dir if args.save_fig else None

    try:
        hdul = load_spec_from_cache(
            rec["plate"],
            rec["fiber"],
            rec["mjd"],
            cache_dir=args.cache_dir,
        )
        if hdul is None:
            hdul = fetch_spectrum_fits(
                args,
                rec["plate"],
                rec["fiber"],
                rec["mjd"],
                cache_dir=args.cache_dir,
            )

        try:
            lam, flux, err = get_spectrum_arrays(hdul)
        finally:
            hdul.close()
        if len(lam) == 0:
            raise RuntimeError("Spectrum has no good pixels after ivar filtering.")

        fit_name = f"z{rec['z']:.3f}_{rec['sdss_name']}"
        prior_config = build_default_prior_config(flux)
        decompose_host_eff = effective_decompose_host_flag(rec["z"], requested=args.decompose_host)
        fit_bc_eff = effective_fit_bc_flag(rec["z"], requested=args.fit_bc)
        fit_bal_eff = effective_fit_bal_flag(rec["z"])
        result["decompose_host_effective"] = bool(decompose_host_eff)
        result["fit_bc_effective"] = bool(fit_bc_eff)
        result["fit_bal_effective"] = bool(fit_bal_eff)
        psf_bands_all, psf_mags_all, psf_mag_errs_all = build_psf_photometry_inputs(rec)
        result["bands_used"] = "".join(psf_bands_all)
        for band in SDSS_BANDS:
            mag = safe_float(rec.get(f"mean_corrected_{band}"))
            if np.isfinite(mag):
                result[f"mean_corrected_{band}"] = float(mag)

        if args.resume:
            q = JAXQSOFit.load_from_samples(
                filename=fit_name,  # important: matches OutputConfig(save_name=...)
                output_path=str(args.output_dir),
                kwargs_plot={
                    "save_fig_path": args.fig_dir,
                    "show_plot": False,
                },
                plot_diagnostics=args.plot_mcmc_diagnostics,
                diagnostics_kwargs={"save_fig_path": args.fig_dir},
            )
            if len(psf_bands_all) > 0 and not bool(getattr(q, "use_psf_phot", False)):
                raise RuntimeError(
                    f"Saved samples for {fit_name} were fit without PSF photometry, "
                    f"but current inputs provide PSF bands {''.join(psf_bands_all)}. "
                    "Rerun without --resume to refit with PSF mag correction."
                )
        else:
            psf_photometry = None
            if len(psf_bands_all) > 0:
                psf_photometry = PSFPhotometryData(
                    magnitudes=psf_mags_all,
                    magnitude_errors=psf_mag_errs_all,
                    filter_names=tuple(psf_bands_all),
                )

            config = FitConfig(
                observation=Observation(
                    object_id=fit_name,
                    redshift=float(rec["z"]),
                    ra=float(rec["ra"]),
                    dec=float(rec["dec"]),
                    apply_mw_deredden=not args.no_deredden,
                ),
                spectroscopy=SpectroscopyData(
                    wave_obs=lam,
                    fluxes=flux,
                    errors=err,
                ),
                psf_photometry=psf_photometry,
                preprocessing=PreprocessingConfig(
                    wave_range=(args.wave_min, args.wave_max),
                    mask_lya_forest=args.mask_lya_forest,
                ),
                continuum=ContinuumConfig(
                    fit_power_law=args.fit_pl,
                    fit_feii=args.fit_fe,
                    fit_balmer_continuum=fit_bc_eff,
                    fit_polynomial_tilt=args.fit_poly,
                    fit_reddening=True,
                ),
                bal=BALConfig(
                    enabled=fit_bal_eff,
                ),
                host=HostConfig(
                    enabled=decompose_host_eff,
                    dsps_ssp_fn=args.dsps_ssp_fn,
                ),
                lines=LineConfig(
                    enabled=args.fit_lines,
                ),
                inference=InferenceConfig(
                    method=args.fit_method,
                    map_steps=args.optax_steps,
                    learning_rate=args.optax_lr,
                    num_warmup=args.nuts_warmup,
                    num_samples=args.nuts_samples,
                    num_chains=args.nuts_chains,
                    target_accept_prob=args.nuts_target_accept,
                ),
                output=OutputConfig(
                    output_path=str(args.output_dir),
                    save_name=fit_name,
                    save_result=args.save_jaxqsofit_samples,
                    plot_fig=args.save_fig,
                    save_fig=args.save_fig,
                    show_plot=False,
                ),
                prior_config=prior_config_for_fit_config(prior_config),
            )
            q = JAXQSOFit(config)
            q.fit(
                verbose=args.verbose,
                kwargs_plot={
                    "save_fig_path": args.fig_dir,
                    "plot_residual": args.plot_residual,
                    "show_plot": False,
                },
            )
            if args.plot_mcmc_diagnostics:
                q.plot_mcmc_diagnostics(save_fig_path=args.fig_dir)

        result.update(extract_named_results(q))
        result.update(extract_scalar_attrs(q))
        result.update(extract_fit_stats(q))
        compute_derived_results(result, q, args)
        result["fit_ok"] = True
        print_spectrum_diagnostics(result)

        return result

    except Exception as exc:
        result["error_message"] = str(exc)
        if args.verbose:
            traceback.print_exc()
        return result


# -----------------------------------------------------------------------------
# runners
# -----------------------------------------------------------------------------

def run_download(args):
    records = build_records(args)
    errors = []

    for rec in tqdm(records, desc="Downloading spectra"):
        try:
            hdul = fetch_spectrum_fits(
                args,
                rec["plate"],
                rec["fiber"],
                rec["mjd"],
                cache_dir=args.cache_dir,
            )
            hdul.close()
        except Exception as exc:
            errors.append((rec["sdss_name"], str(exc)))

    print(f"Tried to download {len(records)} spectra. Errors: {len(errors)}")
    if errors:
        for name, msg in errors[:10]:
            print(f"  {name}: {msg}")



def run_fit(args):
    records = build_records(args)
    if len(records) == 0:
        filter_ids = [normalize_object_id(x) for x in (args.filter_object_id or [])]
        filter_ids = [x for x in filter_ids if x]
        hint = ""
        if filter_ids:
            preview = ", ".join(filter_ids[:10])
            hint = (
                " "
                f"Requested --filter_object_id values (first up to 10): {preview}. "
                "These IDs may be absent from the input catalog/H5."
            )
        raise RuntimeError(f"No records to process.{hint}")

    worker = partial(run_one_fit, args=args)

    if int(args.nproc) <= 1:
        rows = [worker(rec) for rec in tqdm(records, desc="Fitting spectra")]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=int(args.nproc)) as pool:
            rows = list(tqdm(pool.imap(worker, records), total=len(records), desc="Fitting spectra"))

    df = pd.DataFrame(rows)
    Path(args.fpath_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.fpath_out, index=False)
    print(f"Wrote {len(df)} rows to {args.fpath_out}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fit SDSS spectra with jaxqsofit.")

    # Keep output positional for batch scripts, but require explicit input flag.
    p.add_argument("fpath_out", nargs="?", help="Output CSV with one row per fitted object.")
    p.add_argument("--fpath-in", dest="fpath_in", default=None, help="Input HDF5/CSV quasar catalog.")
    p.add_argument("--fpath-out", dest="fpath_out_opt", default=None, help="Output CSV with one row per fitted object.")
    p.add_argument("--mode", choices=["download", "fit", "fetch-dustmaps"], required=True)

    p.add_argument("--dr16q-fits", default="data/dr16q_prop_May01_2024.fits")
    p.add_argument("--cache-dir", default="data/spectra_cache")
    p.add_argument("--output-dir", default="results/jaxqsofit")
    p.add_argument("--max-sep", type=float, default=1.0, help="Cross-match radius in arcsec.")
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--skip", type=int, default=None)
    p.add_argument("--filter_sdss_name", nargs="+", default=None)
    p.add_argument("--filter_object_id", nargs="+", default=None)
    p.add_argument("--fit-method", choices=["optax", "nuts", "optax+nuts"], default="optax+nuts")
    p.add_argument("--dsps-ssp-fn", default="data/ssp_data_continuum_fsps_v3.2_lgmet_age.h5", help="Path to the DSPS SSP HDF5 file.")
    p.add_argument("--no-deredden", action="store_true")

    p.add_argument("--wave-min", type=float, default=1250.0, help="Rest-frame minimum wavelength.")
    p.add_argument("--wave-max", type=float, default=8000.0, help="Rest-frame maximum wavelength.")

    p.add_argument("--optax-steps", type=int, default=600)
    p.add_argument("--optax-lr", type=float, default=1e-2)
    p.add_argument("--nuts-warmup", type=int, default=50)
    p.add_argument("--nuts-samples", type=int, default=50)
    p.add_argument("--nuts-chains", type=int, default=1)
    p.add_argument("--nuts-target-accept", type=float, default=0.9)

    p.set_defaults(fit_lines=True)
    p.add_argument("--fit-lines", dest="fit_lines", action="store_true")
    p.add_argument("--no-fit-lines", dest="fit_lines", action="store_false")

    p.set_defaults(decompose_host=True)
    p.add_argument("--decompose-host", dest="decompose_host", action="store_true")
    p.add_argument("--no-decompose-host", dest="decompose_host", action="store_false")

    p.set_defaults(fit_pl=True)
    p.add_argument("--fit-pl", dest="fit_pl", action="store_true")
    p.add_argument("--no-fit-pl", dest="fit_pl", action="store_false")

    p.set_defaults(fit_fe=True)
    p.add_argument("--fit-fe", dest="fit_fe", action="store_true")
    p.add_argument("--no-fit-fe", dest="fit_fe", action="store_false")

    p.set_defaults(fit_bc=True)
    p.add_argument("--fit-bc", dest="fit_bc", action="store_true")
    p.add_argument("--no-fit-bc", dest="fit_bc", action="store_false")

    p.set_defaults(fit_poly=True)
    p.add_argument("--fit-poly", dest="fit_poly", action="store_true")
    p.add_argument("--no-fit-poly", dest="fit_poly", action="store_false")

    p.set_defaults(mask_lya_forest=True)
    p.add_argument("--mask-lya-forest", dest="mask_lya_forest", action="store_true")
    p.add_argument("--no-mask-lya-forest", dest="mask_lya_forest", action="store_false")

    p.set_defaults(fit_poly_edge_flex=True)
    p.add_argument("--fit-poly-edge-flex", dest="fit_poly_edge_flex", action="store_true")
    p.add_argument("--no-fit-poly-edge-flex", dest="fit_poly_edge_flex", action="store_false")

    p.set_defaults(save_jaxqsofit_samples=True)
    p.add_argument("--save-jaxqsofit-samples", dest="save_jaxqsofit_samples", action="store_true")
    p.add_argument("--no-save-jaxqsofit-samples", dest="save_jaxqsofit_samples", action="store_false")
    

    p.add_argument("--nproc", type=int, default=1, help="Use spawn multiprocessing when nproc > 1.")
    p.add_argument("--plot-residual", dest="plot_residual", action="store_true", default=False, help="Plot residuals in fit figures.")
    p.add_argument("--plot_mcmc_diagnostics", action="store_true", default=False, help="Plot trace/corner MCMC diagnostics when posterior samples are available.")
    p.add_argument("--disable_rescale_flux", "--disable-rescale-flux", dest="disable_rescale_flux", action="store_true", help="Disable magnitude-based flux rescaling.")
    p.add_argument(
        "--use-h5-mean-correction",
        action="store_true",
        default=False,
        help="Use legacy H5 mean correction (mags_mean_* + mean_*) instead of light-curve-derived means.",
    )
    p.set_defaults(save_fig=True)
    p.add_argument("--save-fig", dest="save_fig", action="store_true")
    p.add_argument("--no-save-fig", dest="save_fig", action="store_false")
    p.add_argument("--fig-dir", default="plots/jaxqsofit/", help="Path to save figures")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume mode: load the saved samples from jaxqsofit.")
    p.add_argument("--dustmaps-data-dir", default="results/dustmaps", help="Directory to store dustmaps data (used for fetch-dustmaps mode)")
    args = p.parse_args()

    if args.fpath_out is None and args.fpath_out_opt is not None:
        args.fpath_out = args.fpath_out_opt

    # fit and download need an input source (catalog file or object_id filter);
    # fit also needs output CSV.
    if args.mode in {"fit", "download"} and not (args.fpath_in or args.filter_object_id):
        p.error("--mode fit/download requires at least one of --fpath-in or --filter_object_id.")
    if args.mode == "fit" and not args.fpath_out:
        p.error("fpath_out is required for --mode fit.")

    return args



def main():
    args = parse_args()

    # Set dustmaps location only when it will be used.
    #if args.mode == "fetch-dustmaps" or (args.mode == "fit" and not args.no_deredden):
        #from dustmaps.config import config
        #config["data_dir"] = args.dustmaps_data_dir

    if args.mode == "download":
        run_download(args)
    elif args.mode == "fit":
        run_fit(args)
    elif args.mode == "fetch-dustmaps":
        fetch_dustmaps(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
