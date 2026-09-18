"""Compact paired light-curve posterior draws for catalog HDF5 products."""

from __future__ import annotations

import hashlib

import h5py
import numpy as np


LIGHT_CURVE_POSTERIOR_DRAW_GROUP = "light_curve_posterior_draws"
LIGHT_CURVE_POSTERIOR_DRAW_FORMAT = "qvc_light_curve_posterior_draws_v3"
LIGHT_CURVE_POSTERIOR_DRAW_COUNT = 128
LIGHT_CURVE_POSTERIOR_BANDS = ("u", "g", "r", "i", "z")
LIGHT_CURVE_POSTERIOR_DRAW_SELECTION = "sha256_seed_object_id_uniform_without_replacement_v1"
LIGHT_CURVE_POSTERIOR_DRAW_PAYLOAD_KEY = "_light_curve_posterior_draw_payload"
LIGHT_CURVE_LOG_SIGMA_DRAW_COL = "light_curve_log_sigma_uv_draws"
LIGHT_CURVE_LOG_TAU_RF_DRAW_COL = "light_curve_log_tau_uv_rf_draws"
LIGHT_CURVE_POSTERIOR_VALID_COUNT_COL = "light_curve_posterior_valid_count"


def sigma_band_dataset(band):
    return f"log_sigma_band_{band}"


def tau_cont_band_dataset(band):
    return f"log_tau_cont_band_{band}_rf"


def draw_dataset_names():
    names = ["log_sigma_uv", "log_tau_uv_rf", "posterior_index"]
    names.extend(sigma_band_dataset(band) for band in LIGHT_CURVE_POSTERIOR_BANDS)
    names.extend(tau_cont_band_dataset(band) for band in LIGHT_CURVE_POSTERIOR_BANDS)
    return tuple(names)


def deterministic_posterior_indices(
    valid_indices, *, object_id, seed, draw_count=LIGHT_CURVE_POSTERIOR_DRAW_COUNT
):
    """Choose reproducible paired indices without replacement."""
    valid_indices = np.asarray(valid_indices, dtype=np.int32)
    if len(valid_indices) <= int(draw_count):
        return valid_indices
    digest = hashlib.sha256(f"{int(seed)}:{object_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))
    positions = np.sort(rng.choice(len(valid_indices), size=int(draw_count), replace=False))
    return valid_indices[positions]


def compact_log_sigma_tau_posterior_draws(
    log_sigma_uv,
    log_tau_uv,
    *,
    log_sigma_band,
    log_tau_cont_band_rf,
    bands,
    redshift,
    object_id,
    selection_seed=0,
    disk_order=0,
):
    """Return one fixed-width paired UV and per-band continuum payload."""
    sigma_raw = np.asarray(log_sigma_uv, dtype=float).reshape(-1)
    tau_raw = np.asarray(log_tau_uv, dtype=float).reshape(-1)
    if sigma_raw.shape != tau_raw.shape:
        raise ValueError(
            "Paired log_sigma_uv/log_tau_uv posterior shapes differ: "
            f"{sigma_raw.shape} versus {tau_raw.shape}."
        )
    redshift = float(redshift)
    if not np.isfinite(redshift) or redshift <= -1.0:
        raise ValueError(f"Invalid redshift for posterior draws: {redshift!r}.")
    bands = tuple(str(band) for band in bands)
    unknown = sorted(set(bands) - set(LIGHT_CURVE_POSTERIOR_BANDS))
    if unknown or len(set(bands)) != len(bands):
        raise ValueError(f"Invalid light-curve posterior bands: {bands}.")

    sigma_by_band = {}
    tau_by_band = {}
    finite = np.isfinite(sigma_raw) & np.isfinite(tau_raw)
    for band in bands:
        try:
            sigma_values = np.asarray(log_sigma_band[band], dtype=float).reshape(-1)
            tau_values = np.asarray(log_tau_cont_band_rf[band], dtype=float).reshape(-1)
        except KeyError as exc:
            raise KeyError(f"Missing per-band posterior draws for {band!r}.") from exc
        if sigma_values.shape != sigma_raw.shape or tau_values.shape != sigma_raw.shape:
            raise ValueError(
                f"Per-band posterior shape mismatch for {band!r}: "
                f"sigma={sigma_values.shape}, tau={tau_values.shape}, expected={sigma_raw.shape}."
            )
        sigma_by_band[band] = sigma_values
        tau_by_band[band] = tau_values
        finite &= np.isfinite(sigma_values) & np.isfinite(tau_values)

    finite_indices = np.flatnonzero(finite).astype(np.int32)
    if not len(finite_indices):
        raise ValueError("No jointly finite UV and per-band continuum draws.")
    selected = deterministic_posterior_indices(
        finite_indices, object_id=str(object_id), seed=selection_seed
    )
    count = len(selected)

    def padded(values, *, dtype=np.float32, fill=np.nan):
        output = np.full(LIGHT_CURVE_POSTERIOR_DRAW_COUNT, fill, dtype=dtype)
        output[: len(values)] = np.asarray(values, dtype=dtype)
        return output

    payload = {
        "log_sigma_uv": padded(sigma_raw[selected] / np.log(10.0)),
        "log_tau_uv_rf": padded(
            tau_raw[selected] / np.log(10.0) - np.log10(1.0 + redshift)
        ),
        "posterior_index": padded(selected, dtype=np.int32, fill=-1),
        "band_present": np.asarray(
            [band in bands for band in LIGHT_CURVE_POSTERIOR_BANDS], dtype=bool
        ),
        "valid_count": np.int16(count),
        "finite_source_draw_count": np.int32(len(finite_indices)),
        "source_draw_count": np.int32(len(sigma_raw)),
        "disk_order": np.int16(disk_order),
        "selection_seed": int(selection_seed),
        "format": LIGHT_CURVE_POSTERIOR_DRAW_FORMAT,
    }
    for band in LIGHT_CURVE_POSTERIOR_BANDS:
        payload[sigma_band_dataset(band)] = (
            padded(sigma_by_band[band][selected]) if band in bands else padded([])
        )
        payload[tau_cont_band_dataset(band)] = (
            padded(tau_by_band[band][selected]) if band in bands else padded([])
        )
    return payload


def stack_light_curve_posterior_draw_payloads(payloads):
    """Stack optional per-object payloads on the catalog row axis."""
    payloads = list(payloads)
    n_rows = len(payloads)
    shape = (n_rows, LIGHT_CURVE_POSTERIOR_DRAW_COUNT)
    stacked = {
        name: np.full(
            shape,
            -1 if name == "posterior_index" else np.nan,
            dtype=np.int32 if name == "posterior_index" else np.float32,
        )
        for name in draw_dataset_names()
    }
    stacked.update(
        band_present=np.zeros((n_rows, len(LIGHT_CURVE_POSTERIOR_BANDS)), dtype=bool),
        valid_count=np.zeros(n_rows, dtype=np.int16),
        finite_source_draw_count=np.zeros(n_rows, dtype=np.int32),
        source_draw_count=np.zeros(n_rows, dtype=np.int32),
        disk_order=np.zeros(n_rows, dtype=np.int16),
    )
    seeds = set()
    for row_index, payload in enumerate(payloads):
        if payload is None:
            continue
        if payload.get("format") != LIGHT_CURVE_POSTERIOR_DRAW_FORMAT:
            raise ValueError("Only the v3 light-curve posterior format is supported.")
        for name in draw_dataset_names():
            values = np.asarray(payload[name])
            if values.shape != (LIGHT_CURVE_POSTERIOR_DRAW_COUNT,):
                raise ValueError(
                    f"Posterior payload {name!r} has shape {values.shape}; "
                    f"expected ({LIGHT_CURVE_POSTERIOR_DRAW_COUNT},)."
                )
            stacked[name][row_index] = values
        present = np.asarray(payload["band_present"], dtype=bool)
        if present.shape != (len(LIGHT_CURVE_POSTERIOR_BANDS),):
            raise ValueError(f"Invalid band_present shape {present.shape}.")
        stacked["band_present"][row_index] = present
        for name in ("valid_count", "finite_source_draw_count", "source_draw_count", "disk_order"):
            stacked[name][row_index] = payload[name]
        seeds.add(int(payload.get("selection_seed", 0)))
    if len(seeds) > 1:
        raise ValueError(f"Light-curve posterior payloads use mixed selection seeds: {seeds}.")
    stacked["selection_seed"] = seeds.pop() if seeds else 0
    stacked["format"] = LIGHT_CURVE_POSTERIOR_DRAW_FORMAT
    return stacked


def write_light_curve_posterior_draw_group(hdf, payload):
    """Write a stacked v3 compact draw payload into an open HDF5 file."""
    if payload.get("format") != LIGHT_CURVE_POSTERIOR_DRAW_FORMAT:
        raise ValueError("Only the v3 light-curve posterior format is supported.")
    group = hdf.create_group(LIGHT_CURVE_POSTERIOR_DRAW_GROUP)
    group.attrs.update(
        format=LIGHT_CURVE_POSTERIOR_DRAW_FORMAT,
        draw_count=LIGHT_CURVE_POSTERIOR_DRAW_COUNT,
        draw_selection=LIGHT_CURVE_POSTERIOR_DRAW_SELECTION,
        selection_seed=int(payload["selection_seed"]),
        row_alignment="root_catalog_leading_axis",
        posterior_index_semantics="zero_based_per_object_sample_file_flattened_axis",
        bands=",".join(LIGHT_CURVE_POSTERIOR_BANDS),
        logarithm_base=10,
        log_sigma_uv_definition="continuum_rms_at_rest_2500A",
        log_tau_uv_rf_definition="continuum_only_disk_convolved_integral_timescale_at_rest_2500A",
        log_sigma_band_definition="continuum_rms_at_band_rest_wavelength",
        log_tau_cont_band_rf_definition=(
            "continuum_only_disk_convolved_integral_timescale_at_band_rest_wavelength"
        ),
    )
    for name in draw_dataset_names():
        dtype = np.int32 if name == "posterior_index" else np.float32
        group.create_dataset(
            name, data=np.asarray(payload[name], dtype=dtype), dtype=dtype,
            compression="gzip", shuffle=True,
        )
    for name, dtype in (
        ("band_present", bool),
        ("valid_count", np.int16),
        ("finite_source_draw_count", np.int32),
        ("source_draw_count", np.int32),
        ("disk_order", np.int16),
    ):
        group.create_dataset(
            name, data=np.asarray(payload[name], dtype=dtype), dtype=dtype,
            compression="gzip", shuffle=True,
        )


def read_light_curve_posterior_draw_group(hdf, *, incompatible_as_missing=False):
    """Read and validate the sole supported v3 compact draw group."""
    if LIGHT_CURVE_POSTERIOR_DRAW_GROUP not in hdf:
        return None
    group = hdf[LIGHT_CURVE_POSTERIOR_DRAW_GROUP]
    if not isinstance(group, h5py.Group):
        raise ValueError(f"{LIGHT_CURVE_POSTERIOR_DRAW_GROUP!r} must be an HDF5 group.")
    payload_format = group.attrs.get("format", "")
    if isinstance(payload_format, bytes):
        payload_format = payload_format.decode()
    if (
        payload_format != LIGHT_CURVE_POSTERIOR_DRAW_FORMAT
        or int(group.attrs.get("draw_count", -1)) != LIGHT_CURVE_POSTERIOR_DRAW_COUNT
    ):
        if incompatible_as_missing:
            return None
        raise ValueError(
            "Unsupported light-curve posterior draw group; regenerate the catalog "
            f"with {LIGHT_CURVE_POSTERIOR_DRAW_FORMAT}."
        )
    required = draw_dataset_names() + (
        "band_present", "valid_count", "finite_source_draw_count", "source_draw_count", "disk_order"
    )
    missing = [name for name in required if name not in group]
    if missing:
        if incompatible_as_missing:
            return None
        raise KeyError(f"Incomplete {LIGHT_CURVE_POSTERIOR_DRAW_GROUP} group; missing {missing}.")
    payload = {name: np.asarray(group[name][...]) for name in required}
    payload["selection_seed"] = int(group.attrs.get("selection_seed", 0))
    payload["format"] = LIGHT_CURVE_POSTERIOR_DRAW_FORMAT
    n_rows = len(payload["valid_count"])
    expected = (n_rows, LIGHT_CURVE_POSTERIOR_DRAW_COUNT)
    for name in draw_dataset_names():
        if payload[name].shape != expected:
            if incompatible_as_missing:
                return None
            raise ValueError(
                f"Embedded posterior {name!r} has shape {payload[name].shape}; expected {expected}."
            )
    expected_bands = (n_rows, len(LIGHT_CURVE_POSTERIOR_BANDS))
    if payload["band_present"].shape != expected_bands:
        if incompatible_as_missing:
            return None
        raise ValueError(
            f"Embedded band_present has shape {payload['band_present'].shape}; "
            f"expected {expected_bands}."
        )
    return payload
