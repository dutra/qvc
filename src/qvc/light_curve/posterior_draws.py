"""Compact paired light-curve posterior draws for catalog HDF5 products."""

from __future__ import annotations

import hashlib

import h5py
import numpy as np


LIGHT_CURVE_POSTERIOR_DRAW_GROUP = "light_curve_posterior_draws"
LIGHT_CURVE_POSTERIOR_DRAW_FORMAT = "qvc_light_curve_posterior_draws_v1"
LIGHT_CURVE_POSTERIOR_DRAW_COUNT = 64
LIGHT_CURVE_POSTERIOR_DRAW_SELECTION = (
    "sha256_seed_object_id_uniform_without_replacement_v1"
)
LIGHT_CURVE_POSTERIOR_DRAW_PAYLOAD_KEY = (
    "_light_curve_posterior_draw_payload"
)
LIGHT_CURVE_LOG_SIGMA_DRAW_COL = "light_curve_log_sigma_uv_draws"
LIGHT_CURVE_LOG_TAU_RF_DRAW_COL = "light_curve_log_tau_uv_rf_draws"
LIGHT_CURVE_POSTERIOR_VALID_COUNT_COL = "light_curve_posterior_valid_count"


def deterministic_posterior_indices(
    valid_indices,
    *,
    object_id,
    seed,
    draw_count=LIGHT_CURVE_POSTERIOR_DRAW_COUNT,
):
    """Choose reproducible paired indices without replacement."""

    valid_indices = np.asarray(valid_indices, dtype=np.int32)
    if len(valid_indices) <= int(draw_count):
        return valid_indices
    digest = hashlib.sha256(
        f"{int(seed)}:{object_id}".encode("utf-8")
    ).digest()
    object_seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(object_seed)
    chosen_positions = np.sort(
        rng.choice(len(valid_indices), size=int(draw_count), replace=False)
    )
    return valid_indices[chosen_positions]


def compact_log_sigma_tau_posterior_draws(
    log_sigma_uv,
    log_tau_uv,
    *,
    redshift,
    object_id,
    selection_seed=0,
):
    """Return one fixed-width paired base-10/rest-frame draw payload."""

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

    finite_indices = np.flatnonzero(
        np.isfinite(sigma_raw) & np.isfinite(tau_raw)
    ).astype(np.int32)
    if not len(finite_indices):
        raise ValueError("No finite paired log_sigma_uv/log_tau_uv draws.")
    selected = deterministic_posterior_indices(
        finite_indices,
        object_id=str(object_id),
        seed=selection_seed,
    )
    count = len(selected)

    sigma_compact = np.full(
        LIGHT_CURVE_POSTERIOR_DRAW_COUNT, np.nan, dtype=np.float32
    )
    tau_compact = np.full(
        LIGHT_CURVE_POSTERIOR_DRAW_COUNT, np.nan, dtype=np.float32
    )
    index_compact = np.full(
        LIGHT_CURVE_POSTERIOR_DRAW_COUNT, -1, dtype=np.int32
    )
    sigma_compact[:count] = (
        sigma_raw[selected] / np.log(10.0)
    ).astype(np.float32)
    tau_compact[:count] = (
        tau_raw[selected] / np.log(10.0) - np.log10(1.0 + redshift)
    ).astype(np.float32)
    index_compact[:count] = selected
    return {
        "log_sigma_uv": sigma_compact,
        "log_tau_uv_rf": tau_compact,
        "posterior_index": index_compact,
        "valid_count": np.int16(count),
        "finite_source_draw_count": np.int32(len(finite_indices)),
        "source_draw_count": np.int32(len(sigma_raw)),
        "selection_seed": int(selection_seed),
    }


def stack_light_curve_posterior_draw_payloads(payloads):
    """Stack optional per-object payloads on the catalog row axis."""

    payloads = list(payloads)
    n_rows = len(payloads)
    shape = (n_rows, LIGHT_CURVE_POSTERIOR_DRAW_COUNT)
    stacked = {
        "log_sigma_uv": np.full(shape, np.nan, dtype=np.float32),
        "log_tau_uv_rf": np.full(shape, np.nan, dtype=np.float32),
        "posterior_index": np.full(shape, -1, dtype=np.int32),
        "valid_count": np.zeros(n_rows, dtype=np.int16),
        "finite_source_draw_count": np.zeros(n_rows, dtype=np.int32),
        "source_draw_count": np.zeros(n_rows, dtype=np.int32),
    }
    seeds = set()
    for row_index, payload in enumerate(payloads):
        if payload is None:
            continue
        for name in ("log_sigma_uv", "log_tau_uv_rf", "posterior_index"):
            values = np.asarray(payload[name])
            if values.shape != (LIGHT_CURVE_POSTERIOR_DRAW_COUNT,):
                raise ValueError(
                    f"Posterior payload {name!r} has shape {values.shape}; "
                    f"expected ({LIGHT_CURVE_POSTERIOR_DRAW_COUNT},)."
                )
            stacked[name][row_index] = values
        for name in (
            "valid_count",
            "finite_source_draw_count",
            "source_draw_count",
        ):
            stacked[name][row_index] = payload[name]
        seeds.add(int(payload.get("selection_seed", 0)))
    if len(seeds) > 1:
        raise ValueError(
            f"Light-curve posterior payloads use mixed selection seeds: {seeds}."
        )
    stacked["selection_seed"] = seeds.pop() if seeds else 0
    return stacked


def write_light_curve_posterior_draw_group(hdf, payload):
    """Write a stacked compact draw payload into an open HDF5 file."""

    group = hdf.create_group(LIGHT_CURVE_POSTERIOR_DRAW_GROUP)
    group.attrs["format"] = LIGHT_CURVE_POSTERIOR_DRAW_FORMAT
    group.attrs["draw_count"] = LIGHT_CURVE_POSTERIOR_DRAW_COUNT
    group.attrs["draw_selection"] = LIGHT_CURVE_POSTERIOR_DRAW_SELECTION
    group.attrs["selection_seed"] = int(payload["selection_seed"])
    group.attrs["row_alignment"] = "root_catalog_leading_axis"
    group.attrs["posterior_index_semantics"] = (
        "zero_based_per_object_sample_file_flattened_axis"
    )
    group.attrs["logarithm_base"] = 10
    group.attrs["log_sigma_uv_definition"] = (
        "saved_log_sigma_uv_divided_by_ln10"
    )
    group.attrs["log_tau_uv_rf_definition"] = (
        "saved_log_tau_uv_divided_by_ln10_minus_log10_1_plus_z"
    )
    for name in ("log_sigma_uv", "log_tau_uv_rf"):
        group.create_dataset(
            name,
            data=np.asarray(payload[name], dtype=np.float32),
            dtype=np.float32,
            compression="gzip",
            shuffle=True,
        )
    group.create_dataset(
        "posterior_index",
        data=np.asarray(payload["posterior_index"], dtype=np.int32),
        dtype=np.int32,
        compression="gzip",
        shuffle=True,
    )
    for name, dtype in (
        ("valid_count", np.int16),
        ("finite_source_draw_count", np.int32),
        ("source_draw_count", np.int32),
    ):
        group.create_dataset(
            name,
            data=np.asarray(payload[name], dtype=dtype),
            dtype=dtype,
            compression="gzip",
            shuffle=True,
        )


def read_light_curve_posterior_draw_group(hdf):
    """Read and minimally validate the compact draw group, or return None."""

    if LIGHT_CURVE_POSTERIOR_DRAW_GROUP not in hdf:
        return None
    group = hdf[LIGHT_CURVE_POSTERIOR_DRAW_GROUP]
    if not isinstance(group, h5py.Group):
        raise ValueError(
            f"{LIGHT_CURVE_POSTERIOR_DRAW_GROUP!r} must be an HDF5 group."
        )
    required = (
        "log_sigma_uv",
        "log_tau_uv_rf",
        "posterior_index",
        "valid_count",
        "finite_source_draw_count",
        "source_draw_count",
    )
    missing = [name for name in required if name not in group]
    if missing:
        raise KeyError(
            f"Incomplete {LIGHT_CURVE_POSTERIOR_DRAW_GROUP} group; missing {missing}."
        )
    payload = {name: np.asarray(group[name][...]) for name in required}
    payload["selection_seed"] = int(group.attrs.get("selection_seed", 0))
    n_rows = len(payload["valid_count"])
    expected = (n_rows, LIGHT_CURVE_POSTERIOR_DRAW_COUNT)
    for name in ("log_sigma_uv", "log_tau_uv_rf", "posterior_index"):
        if payload[name].shape != expected:
            raise ValueError(
                f"Embedded posterior {name!r} has shape {payload[name].shape}; "
                f"expected {expected}."
            )
    return payload
