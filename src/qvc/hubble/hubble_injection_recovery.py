"""Synthetic injection/recovery utilities for the AGN Hubble relation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from astropy.cosmology import FlatLambdaCDM
import h5py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from qvc.hubble.hubble_completeness_refactored import (
    Completeness2D,
    get_completeness_function_2d,
    get_relative_selection_function_2d,
)
from qvc.hubble.hubble_likelihood import log_likelihood, sigma_lens_from_dc
from qvc.hubble.hubble_model import (
    AgnPivotContext,
    M_model_agn,
    agn_model_pack_obs,
    agn_model_pack_params,
    build_agn_pivot_context,
    get_model_params,
)


INJECTION_SCHEMA_VERSION = 1
SELECTION_MODELS = ("none", "logistic-magnitude", "logistic-sigma-tau")
PREDICTOR_NOISE_MODES = ("noiseless", "realistic")
COMPLETENESS_CANDIDATES = ("none", "oracle", "current-2d", "relative-2d")
RECOVERY_PARAMETERS = ("M0_agn", "alpha_agn", "beta_agn", "log_f")
RECOVERY_ERROR_SCALES = {
    "M0_agn": 0.08,
    "alpha_agn": 0.20,
    "beta_agn": 0.20,
    "log_f": 0.12,
}
DIAGNOSTIC_Z_EDGES = (0.44, 0.60, 0.90, 1.20, 1.50, 2.00, 2.50, 3.16)
TRUTH_AT_FIT_KEYS = ("H0", "Om0", *RECOVERY_PARAMETERS)
INJECTION_PARENT_COLUMNS = (
    "object_id",
    "z",
    "z_err",
    "apparent_mag_2500",
    "apparent_mag_2500_err",
    "log_sigma_uv",
    "log_tau_uv_rf",
    "log_sigma_uv_std_psd",
    "log_tau_uv_rf_std_psd",
    "log_sigma_uv_log_tau_uv_rf_cov_psd",
    "log_sigma_uv_latent",
    "log_tau_uv_rf_latent",
    "absolute_mag_2500_latent",
    "selection_probability",
    "selected",
)


def _finite_float(value: Any, *, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _finite_pair(values: Any, *, name: str, ordered: bool = False) -> tuple[float, float]:
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly two numeric values.") from exc
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two numeric values.")
    pair = tuple(_finite_float(value, name=f"{name}[{i}]") for i, value in enumerate(values))
    if ordered and pair[0] >= pair[1]:
        raise ValueError(f"{name} must be strictly increasing; got {pair!r}.")
    return pair


@dataclass(frozen=True)
class HubbleInjectionTruth:
    H0: float
    Om0: float
    M0_agn: float
    alpha_agn: float
    beta_agn: float
    log_f: float
    reference_pivots: tuple[float, float]

    def __post_init__(self):
        for name in ("H0", "Om0", "M0_agn", "alpha_agn", "beta_agn", "log_f"):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name=name))
        if self.H0 <= 0.0:
            raise ValueError("H0 must be positive.")
        if not (0.0 < self.Om0 < 1.0):
            raise ValueError("Om0 must lie strictly between zero and one.")
        object.__setattr__(
            self,
            "reference_pivots",
            _finite_pair(self.reference_pivots, name="reference_pivots"),
        )


@dataclass(frozen=True)
class HubbleInjectionConfig:
    seed: int
    n_parent: int
    z_range: tuple[float, float]
    selection_model: str
    predictor_noise: str

    def __post_init__(self):
        if isinstance(self.seed, bool):
            raise ValueError("seed must be an integer.")
        try:
            seed = int(self.seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("seed must be an integer.") from exc
        if seed != self.seed or seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        try:
            n_parent = int(self.n_parent)
        except (TypeError, ValueError) as exc:
            raise ValueError("n_parent must be an integer.") from exc
        if n_parent != self.n_parent or n_parent < 8:
            raise ValueError("n_parent must be an integer greater than or equal to 8.")
        z_range = _finite_pair(self.z_range, name="z_range", ordered=True)
        selection_model = str(self.selection_model)
        if selection_model not in SELECTION_MODELS:
            raise ValueError(
                f"selection_model must be one of {SELECTION_MODELS}; got {selection_model!r}."
            )
        predictor_noise = str(self.predictor_noise)
        if predictor_noise not in PREDICTOR_NOISE_MODES:
            raise ValueError(
                f"predictor_noise must be one of {PREDICTOR_NOISE_MODES}; "
                f"got {predictor_noise!r}."
            )
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "n_parent", n_parent)
        object.__setattr__(self, "z_range", z_range)
        object.__setattr__(self, "selection_model", selection_model)
        object.__setattr__(self, "predictor_noise", predictor_noise)


@dataclass(frozen=True)
class InjectionDataset:
    parent: pd.DataFrame
    selected: pd.DataFrame
    selected_mask: np.ndarray
    selection_probability: np.ndarray
    pivot_context: AgnPivotContext
    truth_at_fit_pivot: dict[str, float]
    dataset_id: str
    config: HubbleInjectionConfig
    truth: HubbleInjectionTruth


@dataclass(frozen=True)
class RecoveryResult:
    candidate: str
    backend: str
    estimates: dict[str, float]
    truth: dict[str, float]
    metrics: dict[str, float]
    residuals: pd.DataFrame
    runtime_seconds: float = 0.0
    samples: np.ndarray | None = None
    model_labels: tuple[str, ...] = ()


def _stratified_redshifts(
    rng: np.random.Generator,
    n_parent: int,
    z_range: tuple[float, float],
) -> np.ndarray:
    internal = [
        value
        for value in DIAGNOSTIC_Z_EDGES
        if z_range[0] < value < z_range[1]
    ]
    edges = np.asarray((z_range[0], *internal, z_range[1]), dtype=float)
    bin_index = np.arange(n_parent, dtype=int) % (len(edges) - 1)
    z = rng.uniform(edges[bin_index], edges[bin_index + 1])
    return z[rng.permutation(n_parent)]


def _reference_pivot_context(
    object_ids: np.ndarray,
    z_range: tuple[float, float],
    reference_pivots: tuple[float, float],
) -> AgnPivotContext:
    return AgnPivotContext(
        observable_names=("log_sigma_uv", "log_tau_uv_rf"),
        values=reference_pivots,
        z_range=z_range,
        reference_object_ids=tuple(object_ids.astype(str)),
    )


def _dataset_digest(
    parent: pd.DataFrame,
    selected_mask: np.ndarray,
    selection_probability: np.ndarray,
    config: HubbleInjectionConfig,
    truth: HubbleInjectionTruth,
) -> str:
    digest = hashlib.sha256()
    metadata = {
        "schema_version": INJECTION_SCHEMA_VERSION,
        "config": asdict(config),
        "truth": asdict(truth),
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    digest.update(pd.util.hash_pandas_object(parent, index=True).to_numpy().tobytes())
    digest.update(np.asarray(selected_mask, dtype=np.uint8).tobytes())
    digest.update(np.asarray(selection_probability, dtype="<f8").tobytes())
    return digest.hexdigest()


def _truth_at_pivot(
    truth: HubbleInjectionTruth,
    pivot_context: AgnPivotContext,
) -> dict[str, float]:
    fit_pivots = pivot_context.as_dict()
    return {
        "H0": truth.H0,
        "Om0": truth.Om0,
        "M0_agn": (
            truth.M0_agn
            + truth.alpha_agn
            * (fit_pivots["log_sigma_uv"] - truth.reference_pivots[0])
            + truth.beta_agn
            * (fit_pivots["log_tau_uv_rf"] - truth.reference_pivots[1])
        ),
        "alpha_agn": truth.alpha_agn,
        "beta_agn": truth.beta_agn,
        "log_f": truth.log_f,
    }


def generate_hubble_injection(
    config: HubbleInjectionConfig,
    truth: HubbleInjectionTruth,
) -> InjectionDataset:
    """Generate one reproducible parent and selected AGN Hubble-table sample."""

    if not isinstance(config, HubbleInjectionConfig):
        raise TypeError("config must be a HubbleInjectionConfig.")
    if not isinstance(truth, HubbleInjectionTruth):
        raise TypeError("truth must be a HubbleInjectionTruth.")

    rng = np.random.default_rng(config.seed)
    n = config.n_parent
    object_ids = np.asarray([f"inj_{config.seed}_{i:06d}" for i in range(n)], dtype=str)
    z = _stratified_redshifts(rng, n, config.z_range)

    latent_standard = rng.multivariate_normal(
        mean=np.zeros(2),
        cov=np.array([[1.0, 0.25], [0.25, 1.0]], dtype=float),
        size=n,
    )
    z_centered = z - float(np.mean(config.z_range))
    log_sigma_latent = -0.70 + 0.18 * latent_standard[:, 0] + 0.035 * z_centered
    log_tau_latent = 2.48 + 0.28 * latent_standard[:, 1] + 0.12 * z_centered

    if config.predictor_noise == "noiseless":
        sigma_err = np.zeros(n, dtype=float)
        tau_err = np.zeros(n, dtype=float)
        sigma_tau_cov = np.zeros(n, dtype=float)
        log_sigma_observed = log_sigma_latent.copy()
        log_tau_observed = log_tau_latent.copy()
    else:
        sigma_err = np.full(n, 0.04, dtype=float)
        tau_err = np.full(n, 0.07, dtype=float)
        sigma_tau_cov = np.full(n, 0.25 * sigma_err[0] * tau_err[0], dtype=float)
        measurement_standard = rng.multivariate_normal(
            mean=np.zeros(2),
            cov=np.array([[1.0, 0.25], [0.25, 1.0]], dtype=float),
            size=n,
        )
        log_sigma_observed = log_sigma_latent + sigma_err * measurement_standard[:, 0]
        log_tau_observed = log_tau_latent + tau_err * measurement_standard[:, 1]

    obs = {
        "log_sigma_uv": log_sigma_latent,
        "log_tau_uv_rf": log_tau_latent,
        "log_sigma_uv_std_psd": sigma_err,
        "log_tau_uv_rf_std_psd": tau_err,
        "log_sigma_uv_log_tau_uv_rf_cov_psd": sigma_tau_cov,
    }
    reference_context = _reference_pivot_context(
        object_ids,
        config.z_range,
        truth.reference_pivots,
    )
    params_arr = agn_model_pack_params(asdict(truth))
    obs_arr, _, pivot_arr = agn_model_pack_obs(obs, pivot_context=reference_context)
    absolute_mag = M_model_agn(params_arr, obs_arr, pivot_arr)

    cosmo = FlatLambdaCDM(H0=truth.H0, Om0=truth.Om0)
    m_err = np.full(n, 0.05, dtype=float)
    lensing_err = sigma_lens_from_dc(z, cosmo)
    total_response_sigma = np.sqrt(
        m_err**2 + lensing_err**2 + np.exp(truth.log_f) ** 2
    )
    apparent_mag = (
        absolute_mag
        + cosmo.distmod(z).value
        + rng.normal(0.0, total_response_sigma, size=n)
    )

    if config.selection_model == "none":
        selection_probability = np.ones(n, dtype=float)
    else:
        magnitude_limit = 22.35 - 0.12 * (z - 1.5)
        selection_logit = (magnitude_limit - apparent_mag) / 0.30
        if config.selection_model == "logistic-sigma-tau":
            selection_logit = (
                selection_logit
                + 1.20
                * (log_sigma_observed - truth.reference_pivots[0])
                / 0.18
                - 0.80
                * (log_tau_observed - truth.reference_pivots[1])
                / 0.28
            )
        selection_probability = expit(selection_logit)
    if np.any(~np.isfinite(selection_probability)) or np.any(
        (selection_probability < 0.0) | (selection_probability > 1.0)
    ):
        raise ValueError("Generated selection probabilities must be finite and in [0, 1].")
    selected_mask = (
        np.ones(n, dtype=bool)
        if config.selection_model == "none"
        else rng.random(n) < selection_probability
    )
    if not np.any(selected_mask):
        raise ValueError("The injected selection produced an empty fitted sample.")

    parent = pd.DataFrame(
        {
            "object_id": object_ids,
            "z": z,
            "z_err": np.zeros(n, dtype=float),
            "apparent_mag_2500": apparent_mag,
            "apparent_mag_2500_err": m_err,
            "log_sigma_uv": log_sigma_observed,
            "log_tau_uv_rf": log_tau_observed,
            "log_sigma_uv_std_psd": sigma_err,
            "log_tau_uv_rf_std_psd": tau_err,
            "log_sigma_uv_log_tau_uv_rf_cov_psd": sigma_tau_cov,
            "log_sigma_uv_latent": log_sigma_latent,
            "log_tau_uv_rf_latent": log_tau_latent,
            "absolute_mag_2500_latent": absolute_mag,
            "selection_probability": selection_probability,
            "selected": selected_mask,
        }
    )
    selected = parent.loc[selected_mask].copy()
    pivot_context = build_agn_pivot_context(selected, config.z_range)
    truth_at_fit_pivot = _truth_at_pivot(truth, pivot_context)
    dataset_id = _dataset_digest(
        parent,
        selected_mask,
        selection_probability,
        config,
        truth,
    )
    return InjectionDataset(
        parent=parent,
        selected=selected,
        selected_mask=selected_mask,
        selection_probability=selection_probability,
        pivot_context=pivot_context,
        truth_at_fit_pivot=truth_at_fit_pivot,
        dataset_id=dataset_id,
        config=config,
        truth=truth,
    )


def _decode_hdf_strings(values: Any) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in np.asarray(values)
        ],
        dtype=object,
    )


def _validate_loaded_dataset(dataset: InjectionDataset) -> None:
    if tuple(dataset.parent.columns) != INJECTION_PARENT_COLUMNS:
        raise ValueError(
            "Injection parent columns do not match the required ordered schema: "
            f"{INJECTION_PARENT_COLUMNS!r}."
        )
    n_parent = len(dataset.parent)
    if not dataset.parent.index.equals(pd.RangeIndex(n_parent)):
        raise ValueError(
            "Injection parent index must be the canonical zero-based RangeIndex."
        )
    if n_parent != dataset.config.n_parent:
        raise ValueError(
            f"Injection parent length {n_parent} does not match config n_parent="
            f"{dataset.config.n_parent}."
        )
    selected_mask = np.asarray(dataset.selected_mask)
    probability = np.asarray(dataset.selection_probability, dtype=float)
    if selected_mask.dtype != np.bool_ or selected_mask.shape != (n_parent,):
        raise ValueError("selected_mask must be a one-dimensional boolean parent-length array.")
    if probability.shape != (n_parent,) or np.any(~np.isfinite(probability)):
        raise ValueError(
            "selection_probability must be a finite one-dimensional parent-length array."
        )
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("selection_probability values must lie in [0, 1].")
    if not np.any(selected_mask):
        raise ValueError("Injection selected_mask must contain at least one selected object.")
    if not np.array_equal(
        dataset.parent["selected"].to_numpy(dtype=bool),
        selected_mask,
    ):
        raise ValueError("parent selected column does not match selected_mask.")
    if not np.array_equal(
        dataset.parent["selection_probability"].to_numpy(dtype=float),
        probability,
    ):
        raise ValueError(
            "parent selection_probability column does not match stored probabilities."
        )
    expected_selected = dataset.parent.loc[selected_mask].copy()
    try:
        pd.testing.assert_frame_equal(dataset.selected, expected_selected, check_exact=True)
    except AssertionError as exc:
        raise ValueError("Selected dataframe does not exactly match parent[selected_mask].") from exc
    object_ids = dataset.parent["object_id"].astype(str)
    if object_ids.duplicated().any() or (object_ids.str.strip() == "").any():
        raise ValueError("Injection object_id values must be unique and nonempty.")
    numeric = dataset.parent.drop(columns=["object_id"]).to_numpy(dtype=float)
    if np.any(~np.isfinite(numeric)):
        raise ValueError("Injection parent numeric values must all be finite.")
    uncertainty_columns = (
        "apparent_mag_2500_err",
        "log_sigma_uv_std_psd",
        "log_tau_uv_rf_std_psd",
    )
    if np.any(
        dataset.parent.loc[:, uncertainty_columns].to_numpy(dtype=float) < 0.0
    ):
        raise ValueError("Injection measurement uncertainties must be nonnegative.")
    sigma_err = dataset.parent["log_sigma_uv_std_psd"].to_numpy(dtype=float)
    tau_err = dataset.parent["log_tau_uv_rf_std_psd"].to_numpy(dtype=float)
    covariance = dataset.parent[
        "log_sigma_uv_log_tau_uv_rf_cov_psd"
    ].to_numpy(dtype=float)
    covariance_determinant = (sigma_err * tau_err) ** 2 - covariance**2
    covariance_tolerance = (
        32.0
        * np.finfo(float).eps
        * np.maximum(1.0, (sigma_err * tau_err) ** 2)
    )
    if np.any(covariance_determinant < -covariance_tolerance):
        raise ValueError(
            "Injection predictor covariance matrices must be positive semidefinite."
        )
    expected_pivot = build_agn_pivot_context(dataset.selected, dataset.config.z_range)
    if dataset.pivot_context != expected_pivot:
        raise ValueError("Stored pivot metadata does not match the selected fitted sample.")
    if tuple(dataset.truth_at_fit_pivot) != TRUTH_AT_FIT_KEYS:
        raise ValueError(
            f"truth_at_fit_pivot keys must be ordered as {TRUTH_AT_FIT_KEYS!r}."
        )
    if np.any(
        ~np.isfinite(
            np.asarray(
                [dataset.truth_at_fit_pivot[key] for key in TRUTH_AT_FIT_KEYS],
                dtype=float,
            )
        )
    ):
        raise ValueError("truth_at_fit_pivot values must all be finite.")
    expected_truth_at_fit_pivot = _truth_at_pivot(
        dataset.truth,
        dataset.pivot_context,
    )
    if dataset.truth_at_fit_pivot != expected_truth_at_fit_pivot:
        raise ValueError(
            "truth_at_fit_pivot is incompatible with the injection truth and "
            "stored pivot context."
        )
    expected_id = _dataset_digest(
        dataset.parent,
        selected_mask,
        probability,
        dataset.config,
        dataset.truth,
    )
    if dataset.dataset_id != expected_id:
        raise ValueError(
            "Injection dataset_id does not match the strict content hash."
        )


def save_injection_hdf5(dataset: InjectionDataset, path: str | Path) -> None:
    """Persist an injection with a strict, versioned schema."""

    if not isinstance(dataset, InjectionDataset):
        raise TypeError("dataset must be an InjectionDataset.")
    _validate_loaded_dataset(dataset)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = INJECTION_SCHEMA_VERSION
        handle.attrs["dataset_id"] = dataset.dataset_id
        handle.attrs["config_json"] = json.dumps(asdict(dataset.config), sort_keys=True)
        handle.attrs["truth_json"] = json.dumps(asdict(dataset.truth), sort_keys=True)
        handle.attrs["parent_column_order_json"] = json.dumps(INJECTION_PARENT_COLUMNS)
        handle.attrs["pivot_rule"] = dataset.pivot_context.rule

        parent_group = handle.create_group("parent")
        for column in INJECTION_PARENT_COLUMNS:
            values = dataset.parent[column].to_numpy()
            if column == "object_id":
                parent_group.create_dataset(
                    column,
                    data=values.astype(object),
                    dtype=string_dtype,
                )
            else:
                parent_group.create_dataset(column, data=values)
        handle.create_dataset(
            "selected_mask",
            data=np.asarray(dataset.selected_mask, dtype=np.bool_),
        )
        handle.create_dataset(
            "selection_probability",
            data=np.asarray(dataset.selection_probability, dtype=float),
        )
        handle.create_dataset(
            "pivot_observable_names",
            data=np.asarray(dataset.pivot_context.observable_names, dtype=object),
            dtype=string_dtype,
        )
        handle.create_dataset(
            "pivot_values",
            data=np.asarray(dataset.pivot_context.values, dtype=float),
        )
        handle.create_dataset(
            "pivot_z_range",
            data=np.asarray(dataset.pivot_context.z_range, dtype=float),
        )
        handle.create_dataset(
            "pivot_reference_object_ids",
            data=np.asarray(dataset.pivot_context.reference_object_ids, dtype=object),
            dtype=string_dtype,
        )
        handle.create_dataset(
            "truth_at_fit_names",
            data=np.asarray(TRUTH_AT_FIT_KEYS, dtype=object),
            dtype=string_dtype,
        )
        handle.create_dataset(
            "truth_at_fit_values",
            data=np.asarray(
                [dataset.truth_at_fit_pivot[key] for key in TRUTH_AT_FIT_KEYS],
                dtype=float,
            ),
        )


def load_injection_hdf5(path: str | Path) -> InjectionDataset:
    """Load only the current complete injection schema; no migration is attempted."""

    path = Path(path)
    required_attrs = {
        "schema_version",
        "dataset_id",
        "config_json",
        "truth_json",
        "parent_column_order_json",
        "pivot_rule",
    }
    required_items = {
        "parent",
        "selected_mask",
        "selection_probability",
        "pivot_observable_names",
        "pivot_values",
        "pivot_z_range",
        "pivot_reference_object_ids",
        "truth_at_fit_names",
        "truth_at_fit_values",
    }
    try:
        with h5py.File(path, "r") as handle:
            missing_attrs = sorted(required_attrs - set(handle.attrs))
            missing_items = sorted(required_items - set(handle))
            if missing_attrs or missing_items:
                raise ValueError(
                    "Injection HDF5 is missing required metadata: "
                    f"attrs={missing_attrs}, items={missing_items}."
                )
            if int(handle.attrs["schema_version"]) != INJECTION_SCHEMA_VERSION:
                raise ValueError(
                    "Unsupported injection schema_version="
                    f"{handle.attrs['schema_version']!r}; expected "
                    f"{INJECTION_SCHEMA_VERSION}."
                )
            stored_columns = tuple(json.loads(handle.attrs["parent_column_order_json"]))
            if stored_columns != INJECTION_PARENT_COLUMNS:
                raise ValueError(
                    "Injection parent column order is incompatible with the current schema."
                )
            parent_group = handle["parent"]
            missing_columns = sorted(set(INJECTION_PARENT_COLUMNS) - set(parent_group))
            if missing_columns:
                raise ValueError(
                    f"Injection HDF5 is missing required parent columns: {missing_columns}."
                )
            parent_data: dict[str, np.ndarray] = {}
            for column in INJECTION_PARENT_COLUMNS:
                values = parent_group[column][:]
                parent_data[column] = (
                    _decode_hdf_strings(values)
                    if column == "object_id"
                    else np.asarray(values)
                )
            parent = pd.DataFrame(parent_data, columns=INJECTION_PARENT_COLUMNS)
            selected_mask = np.asarray(handle["selected_mask"][:], dtype=np.bool_)
            selection_probability = np.asarray(
                handle["selection_probability"][:],
                dtype=float,
            )
            if selected_mask.shape != (len(parent),):
                raise ValueError(
                    "selected_mask must be a one-dimensional boolean "
                    "parent-length array."
                )
            if selection_probability.shape != (len(parent),):
                raise ValueError(
                    "selection_probability must be a one-dimensional "
                    "parent-length array."
                )
            config = HubbleInjectionConfig(**json.loads(handle.attrs["config_json"]))
            truth = HubbleInjectionTruth(**json.loads(handle.attrs["truth_json"]))
            pivot_context = AgnPivotContext(
                observable_names=tuple(
                    _decode_hdf_strings(handle["pivot_observable_names"][:])
                ),
                values=tuple(np.asarray(handle["pivot_values"][:], dtype=float)),
                z_range=tuple(np.asarray(handle["pivot_z_range"][:], dtype=float)),
                reference_object_ids=tuple(
                    _decode_hdf_strings(handle["pivot_reference_object_ids"][:])
                ),
                rule=str(handle.attrs["pivot_rule"]),
            )
            truth_names = tuple(_decode_hdf_strings(handle["truth_at_fit_names"][:]))
            truth_values = tuple(
                np.asarray(handle["truth_at_fit_values"][:], dtype=float)
            )
            if truth_names != TRUTH_AT_FIT_KEYS or len(truth_values) != len(
                TRUTH_AT_FIT_KEYS
            ):
                raise ValueError(
                    "Injection truth-at-fit metadata is incomplete or reordered."
                )
            truth_at_fit_pivot = dict(zip(truth_names, truth_values))
            dataset_id = str(handle.attrs["dataset_id"])
    except OSError as exc:
        raise ValueError(f"Cannot load injection HDF5 {path}: {exc}") from exc

    dataset = InjectionDataset(
        parent=parent,
        selected=parent.loc[selected_mask].copy(),
        selected_mask=selected_mask,
        selection_probability=selection_probability,
        pivot_context=pivot_context,
        truth_at_fit_pivot=truth_at_fit_pivot,
        dataset_id=dataset_id,
        config=config,
        truth=truth,
    )
    try:
        _validate_loaded_dataset(dataset)
    except (AssertionError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Invalid injection HDF5 content: {exc}") from exc
    return dataset


def _write_parent_mock_hdf5(dataset: InjectionDataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "z",
            data=dataset.parent["z"].to_numpy(dtype=float),
        )
        handle.create_dataset(
            "apparent_mag_2500",
            data=dataset.parent["apparent_mag_2500"].to_numpy(dtype=float),
        )
        handle.attrs["mock_count_scale"] = 1.0
        handle.attrs["source"] = "qvc_hubble_injection_recovery"
        handle.attrs["dataset_id"] = dataset.dataset_id


def _oracle_logistic_completeness(dataset: InjectionDataset):
    if dataset.config.selection_model != "logistic-magnitude":
        raise ValueError(
            "The oracle candidate is defined only for selection_model="
            "'logistic-magnitude'; no lower-dimensional oracle is assumed for "
            f"{dataset.config.selection_model!r}."
        )
    mag_centers = np.linspace(15.0, 30.0, 601)
    z_centers = np.linspace(0.0, 4.0, 81)
    mm, zz = np.meshgrid(mag_centers, z_centers, indexing="ij")
    magnitude_limit = 22.35 - 0.12 * (zz - 1.5)
    completeness_map = expit((magnitude_limit - mm) / 0.30)
    model = Completeness2D(mag_centers, z_centers, completeness_map)
    setattr(model, "_recovery_candidate_name", "oracle")
    return (
        model,
        mag_centers,
        z_centers,
        float(np.diff(mag_centers).mean()),
        float(np.diff(z_centers).mean()),
        0.0,
    )


def build_completeness_candidate(
    name: str,
    dataset: InjectionDataset,
    workdir: str | Path,
):
    """Build one strict production-likelihood completeness candidate."""

    name = str(name)
    if name not in COMPLETENESS_CANDIDATES:
        raise ValueError(
            f"Completeness candidate must be one of {COMPLETENESS_CANDIDATES}; "
            f"got {name!r}."
        )
    if not isinstance(dataset, InjectionDataset):
        raise TypeError("dataset must be an InjectionDataset.")
    _validate_loaded_dataset(dataset)
    workdir = Path(workdir)
    if name == "none":
        return None
    if name == "oracle":
        return _oracle_logistic_completeness(dataset)

    mock_path = workdir / f"parent_mock_{dataset.dataset_id[:12]}.h5"
    _write_parent_mock_hdf5(dataset, mock_path)
    builder = (
        get_relative_selection_function_2d
        if name == "relative-2d"
        else get_completeness_function_2d
    )
    completeness_params = builder(
        dataset.selected,
        sim_file=str(mock_path),
        plot=False,
    )
    setattr(
        completeness_params[0],
        "_recovery_candidate_name",
        name,
    )
    return completeness_params


def _selected_agn_data(dataset: InjectionDataset) -> dict[str, np.ndarray]:
    fields = (
        "object_id",
        "z",
        "z_err",
        "apparent_mag_2500",
        "apparent_mag_2500_err",
        "log_sigma_uv",
        "log_tau_uv_rf",
        "log_sigma_uv_std_psd",
        "log_tau_uv_rf_std_psd",
        "log_sigma_uv_log_tau_uv_rf_cov_psd",
    )
    return {
        field: dataset.selected[field].to_numpy()
        for field in fields
    }


def _full_theta(
    free_values: np.ndarray,
    *,
    model_labels: tuple[str, ...],
    fixed_values: dict[str, float],
) -> np.ndarray:
    free = dict(zip(RECOVERY_PARAMETERS, np.asarray(free_values, dtype=float)))
    return np.asarray(
        [
            free[label] if label in free else fixed_values[label]
            for label in model_labels
        ],
        dtype=float,
    )


def _initial_standardization_guess(dataset: InjectionDataset) -> np.ndarray:
    selected = dataset.selected
    pivots = dataset.pivot_context.as_dict()
    cosmo = FlatLambdaCDM(
        H0=dataset.truth_at_fit_pivot["H0"],
        Om0=dataset.truth_at_fit_pivot["Om0"],
    )
    design = np.column_stack(
        [
            np.ones(len(selected), dtype=float),
            selected["log_sigma_uv"].to_numpy(dtype=float)
            - pivots["log_sigma_uv"],
            selected["log_tau_uv_rf"].to_numpy(dtype=float)
            - pivots["log_tau_uv_rf"],
        ]
    )
    absolute_mag_observed = (
        selected["apparent_mag_2500"].to_numpy(dtype=float)
        - cosmo.distmod(selected["z"].to_numpy(dtype=float)).value
    )
    coefficients, _, rank, _ = np.linalg.lstsq(
        design,
        absolute_mag_observed,
        rcond=None,
    )
    if rank != design.shape[1] or np.any(~np.isfinite(coefficients)):
        raise RuntimeError(
            "Cannot initialize recovery: the injected sigma/tau design matrix "
            "is rank deficient or nonfinite."
        )
    residual = absolute_mag_observed - design @ coefficients
    scatter = max(float(np.std(residual, ddof=design.shape[1])), 0.25)
    return np.asarray([*coefficients, np.log(scatter)], dtype=float)


def _candidate_name(completeness_params) -> str:
    if completeness_params is None:
        return "none"
    if not isinstance(completeness_params, tuple) or len(completeness_params) < 2:
        raise TypeError(
            "completeness_params must be None or a production completeness tuple."
        )
    name = getattr(completeness_params[0], "_recovery_candidate_name", None)
    if name not in COMPLETENESS_CANDIDATES:
        raise ValueError(
            "Completeness tuple is missing a valid recovery candidate identity."
        )
    return str(name)


def _recovery_metrics(
    dataset: InjectionDataset,
    estimates: dict[str, float],
    truth: dict[str, float],
    residuals: pd.DataFrame,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "n_parent": float(len(dataset.parent)),
        "n_selected": float(len(dataset.selected)),
        "retention_fraction": float(np.mean(dataset.selected_mask)),
        "raw_residual_mean": float(np.mean(residuals["raw_residual"])),
        "debiased_residual_mean": float(np.mean(residuals["debiased_residual"])),
        "raw_residual_rms": float(
            np.sqrt(np.mean(np.square(residuals["raw_residual"])))
        ),
        "debiased_residual_rms": float(
            np.sqrt(np.mean(np.square(residuals["debiased_residual"])))
        ),
    }
    z = residuals["z"].to_numpy(dtype=float)
    debiased = residuals["debiased_residual"].to_numpy(dtype=float)
    metrics["residual_z_slope"] = float(np.polyfit(z, debiased, 1)[0])
    parameter_errors = []
    normalized_parameter_errors = []
    for name in RECOVERY_PARAMETERS:
        error = float(estimates[name] - truth[name])
        metrics[f"{name}_error"] = error
        parameter_errors.append(error)
        normalized_parameter_errors.append(error / RECOVERY_ERROR_SCALES[name])
    metrics["parameter_rmse_unscaled"] = float(
        np.sqrt(np.mean(np.square(parameter_errors)))
    )
    metrics["parameter_rmse"] = float(
        np.sqrt(np.mean(np.square(normalized_parameter_errors)))
    )
    edges = np.asarray(
        [
            dataset.config.z_range[0],
            *[
                value
                for value in DIAGNOSTIC_Z_EDGES
                if dataset.config.z_range[0] < value < dataset.config.z_range[1]
            ],
            dataset.config.z_range[1],
        ],
        dtype=float,
    )
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (z >= lo) & (z <= hi if hi == edges[-1] else z < hi)
        label = f"z{lo:.2f}_{hi:.2f}".replace(".", "p")
        metrics[f"n_{label}"] = float(np.count_nonzero(mask))
        metrics[f"debiased_residual_mean_{label}"] = (
            float(np.mean(debiased[mask])) if np.any(mask) else np.nan
        )
    return metrics


def evaluate_recovery(
    dataset: InjectionDataset,
    result: RecoveryResult,
) -> dict[str, float]:
    """Recompute recovery metrics from a result and its object-level residuals."""

    if not isinstance(dataset, InjectionDataset):
        raise TypeError("dataset must be an InjectionDataset.")
    if not isinstance(result, RecoveryResult):
        raise TypeError("result must be a RecoveryResult.")
    return _recovery_metrics(
        dataset,
        result.estimates,
        result.truth,
        result.residuals,
    )


def fit_fixed_cosmology(
    dataset: InjectionDataset,
    completeness_params,
) -> RecoveryResult:
    """Recover AGN standardization with cosmology fixed at the injected truth."""

    if not isinstance(dataset, InjectionDataset):
        raise TypeError("dataset must be an InjectionDataset.")
    _validate_loaded_dataset(dataset)
    candidate = _candidate_name(completeness_params)
    agn_data = _selected_agn_data(dataset)
    priors, labels, _ = get_model_params("FlatLambdaCDM", only_agn=True)
    model_labels = tuple(labels)
    fixed_values = {
        "H0": dataset.truth_at_fit_pivot["H0"],
        "Om0": dataset.truth_at_fit_pivot["Om0"],
    }
    bounds = []
    for name in RECOVERY_PARAMETERS:
        low, high = priors[name]
        epsilon = 1e-8 * max(1.0, abs(low), abs(high))
        bounds.append((float(low + epsilon), float(high - epsilon)))

    def objective(free_values):
        theta = _full_theta(
            free_values,
            model_labels=model_labels,
            fixed_values=fixed_values,
        )
        loglike, _ = log_likelihood(
            theta,
            agn_data=agn_data,
            pantheon_data={},
            _sna_L=None,
            _sna_Lower=True,
            _sna_LogdetCov=None,
            cosmo_model="FlatLambdaCDM",
            completeness_params=completeness_params,
            z_pivot_agn=1.5,
            agn_pivot_context=dataset.pivot_context,
            only_agn=True,
            use_full_cov=False,
        )
        return -float(loglike) if np.isfinite(loglike) else 1e100

    started = time.perf_counter()
    initial = _initial_standardization_guess(dataset)
    starts = (
        initial,
        initial + np.asarray([0.15, 0.30, -0.30, 0.05]),
        initial + np.asarray([-0.15, -0.30, 0.30, -0.05]),
    )
    fits = [
        minimize(
            objective,
            np.clip(start, [bound[0] for bound in bounds], [bound[1] for bound in bounds]),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-7},
        )
        for start in starts
    ]
    successful = [
        fit
        for fit in fits
        if fit.success and np.isfinite(fit.fun) and np.all(np.isfinite(fit.x))
    ]
    if not successful:
        messages = [str(fit.message) for fit in fits]
        raise RuntimeError(
            "Fixed-cosmology recovery optimization failed for every start: "
            f"{messages}."
        )
    best = min(successful, key=lambda fit: float(fit.fun))
    estimates = dict(
        zip(RECOVERY_PARAMETERS, np.asarray(best.x, dtype=float))
    )
    estimates.update(fixed_values)
    residuals = _residual_frame_from_estimates(
        dataset,
        estimates,
        completeness_params,
    )
    result_truth = {
        key: float(dataset.truth_at_fit_pivot[key])
        for key in TRUTH_AT_FIT_KEYS
    }
    metrics = _recovery_metrics(
        dataset,
        estimates,
        result_truth,
        residuals,
    )
    return RecoveryResult(
        candidate=candidate,
        backend="fast",
        estimates=estimates,
        truth=result_truth,
        metrics=metrics,
        residuals=residuals,
        runtime_seconds=float(time.perf_counter() - started),
        model_labels=model_labels,
    )


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_recovery_artifacts(
    dataset: InjectionDataset,
    results: list[RecoveryResult] | tuple[RecoveryResult, ...],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the stable comparison artifacts used by the CLI and regressions."""

    if not isinstance(dataset, InjectionDataset):
        raise TypeError("dataset must be an InjectionDataset.")
    results = tuple(results)
    if not results:
        raise ValueError("At least one RecoveryResult is required.")
    if any(not isinstance(result, RecoveryResult) for result in results):
        raise TypeError("results must contain only RecoveryResult objects.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "injection": output_dir / "injection.h5",
        "manifest": output_dir / "manifest.json",
        "recovery": output_dir / "recovery.csv",
        "residuals": output_dir / "residuals.csv",
        "metrics": output_dir / "metrics.json",
        "diagnostics": output_dir / "recovery_diagnostics.pdf",
    }
    save_injection_hdf5(dataset, paths["injection"])

    manifest = {
        "schema_version": INJECTION_SCHEMA_VERSION,
        "dataset_id": dataset.dataset_id,
        "config": asdict(dataset.config),
        "truth": asdict(dataset.truth),
        "truth_at_fit_pivot": dataset.truth_at_fit_pivot,
        "pivot_context": {
            "observable_names": dataset.pivot_context.observable_names,
            "values": dataset.pivot_context.values,
            "z_range": dataset.pivot_context.z_range,
            "reference_object_ids": dataset.pivot_context.reference_object_ids,
            "rule": dataset.pivot_context.rule,
        },
        "candidates": [result.candidate for result in results],
        "backends": [result.backend for result in results],
    }
    paths["manifest"].write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    recovery_rows = []
    for result in results:
        for parameter in TRUTH_AT_FIT_KEYS:
            estimate = float(result.estimates[parameter])
            truth_value = float(result.truth[parameter])
            scale = RECOVERY_ERROR_SCALES.get(parameter, np.nan)
            recovery_rows.append(
                {
                    "dataset_id": dataset.dataset_id,
                    "candidate": result.candidate,
                    "backend": result.backend,
                    "parameter": parameter,
                    "truth": truth_value,
                    "estimate": estimate,
                    "error": estimate - truth_value,
                    "error_scale": scale,
                    "normalized_error": (
                        (estimate - truth_value) / scale
                        if np.isfinite(scale)
                        else np.nan
                    ),
                    "runtime_seconds": result.runtime_seconds,
                }
            )
    pd.DataFrame(recovery_rows).to_csv(paths["recovery"], index=False)

    residual_frames = []
    for result in results:
        frame = result.residuals.copy()
        frame.insert(0, "dataset_id", dataset.dataset_id)
        frame.insert(1, "candidate", result.candidate)
        frame.insert(2, "backend", result.backend)
        residual_frames.append(frame)
    pd.concat(residual_frames, axis=0, ignore_index=True).to_csv(
        paths["residuals"],
        index=False,
    )

    metrics_payload = {
        "schema_version": INJECTION_SCHEMA_VERSION,
        "dataset_id": dataset.dataset_id,
        "results": [
            {
                "candidate": result.candidate,
                "backend": result.backend,
                "runtime_seconds": result.runtime_seconds,
                "metrics": result.metrics,
            }
            for result in results
        ],
    }
    paths["metrics"].write_text(
        json.dumps(
            _json_safe(metrics_payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    import matplotlib.pyplot as plt

    fig, (ax_parameters, ax_residuals) = plt.subplots(
        2,
        1,
        figsize=(9.0, 8.0),
        constrained_layout=True,
    )
    x = np.arange(len(RECOVERY_PARAMETERS), dtype=float)
    width = min(0.22, 0.75 / len(results))
    center_offset = 0.5 * (len(results) - 1)
    truth_values = np.asarray(
        [dataset.truth_at_fit_pivot[name] for name in RECOVERY_PARAMETERS],
        dtype=float,
    )
    for i, result in enumerate(results):
        estimates = np.asarray(
            [result.estimates[name] for name in RECOVERY_PARAMETERS],
            dtype=float,
        )
        ax_parameters.scatter(
            x + (i - center_offset) * width,
            estimates - truth_values,
            label=result.candidate,
        )
        ax_residuals.scatter(
            result.residuals["z"],
            result.residuals["debiased_residual"],
            s=7,
            alpha=0.25,
            label=result.candidate,
        )
    ax_parameters.axhline(0.0, color="black", linewidth=1.0)
    ax_parameters.set_xticks(x, RECOVERY_PARAMETERS)
    ax_parameters.set_ylabel("estimate - truth")
    ax_parameters.legend()
    ax_residuals.axhline(0.0, color="black", linewidth=1.0)
    ax_residuals.set_xlabel("redshift")
    ax_residuals.set_ylabel("debiased residual [mag]")
    ax_residuals.legend()
    fig.savefig(paths["diagnostics"])
    plt.close(fig)
    return paths


def _synthetic_pantheon_sample(
    dataset: InjectionDataset,
    *,
    n_calibrators: int = 12,
    n_hubble_flow: int = 72,
) -> pd.DataFrame:
    rng = np.random.default_rng(dataset.config.seed + 10_000)
    cosmo = FlatLambdaCDM(
        H0=dataset.truth.H0,
        Om0=dataset.truth.Om0,
    )
    M0_sn = -19.25
    calibrator_mu = rng.uniform(29.0, 34.0, size=n_calibrators)
    calibrator_z = rng.uniform(0.002, 0.009, size=n_calibrators)
    hubble_z = np.linspace(0.015, 1.1, n_hubble_flow)
    hubble_z += rng.normal(0.0, 0.001, size=n_hubble_flow)
    hubble_z = np.clip(hubble_z, 0.011, None)
    error = 0.08
    calibrator_m = M0_sn + calibrator_mu + rng.normal(
        0.0,
        error,
        size=n_calibrators,
    )
    hubble_m = (
        M0_sn
        + cosmo.distmod(hubble_z).value
        + rng.normal(0.0, error, size=n_hubble_flow)
    )
    return pd.DataFrame(
        {
            "zHD": np.concatenate([calibrator_z, hubble_z]),
            "m_b_corr": np.concatenate([calibrator_m, hubble_m]),
            "IS_CALIBRATOR": np.concatenate(
                [
                    np.ones(n_calibrators, dtype=int),
                    np.zeros(n_hubble_flow, dtype=int),
                ]
            ),
            "CEPH_DIST": np.concatenate(
                [calibrator_mu, np.full(n_hubble_flow, -9.0)]
            ),
            "MU_SH0ES_ERR_DIAG": np.full(
                n_calibrators + n_hubble_flow,
                error,
            ),
        }
    )


def _residual_frame_from_estimates(
    dataset: InjectionDataset,
    estimates: dict[str, float],
    completeness_params,
) -> pd.DataFrame:
    agn_data = _selected_agn_data(dataset)
    _, labels, _ = get_model_params("FlatLambdaCDM", only_agn=True)
    theta = np.asarray([estimates[label] for label in labels], dtype=float)
    _, blob = log_likelihood(
        theta,
        agn_data=agn_data,
        pantheon_data={},
        _sna_L=None,
        _sna_Lower=True,
        _sna_LogdetCov=None,
        cosmo_model="FlatLambdaCDM",
        completeness_params=completeness_params,
        z_pivot_agn=1.5,
        agn_pivot_context=dataset.pivot_context,
        only_agn=True,
        use_full_cov=False,
    )
    params_arr = agn_model_pack_params(estimates)
    obs_arr, _, pivot_arr = agn_model_pack_obs(
        agn_data,
        pivot_context=dataset.pivot_context,
    )
    absolute_mag_model = M_model_agn(params_arr, obs_arr, pivot_arr)
    cosmo = FlatLambdaCDM(H0=estimates["H0"], Om0=estimates["Om0"])
    raw_residual = (
        agn_data["apparent_mag_2500"]
        - absolute_mag_model
        - cosmo.distmod(agn_data["z"]).value
    )
    dmi = np.asarray(blob[1], dtype=float)
    return pd.DataFrame(
        {
            "object_id": agn_data["object_id"].astype(str),
            "z": np.asarray(agn_data["z"], dtype=float),
            "selection_probability": dataset.selected[
                "selection_probability"
            ].to_numpy(dtype=float),
            "raw_residual": raw_residual,
            "dmi": dmi,
            "selection_sigma": np.asarray(blob[2], dtype=float),
            "debiased_residual": raw_residual - dmi,
        },
        index=dataset.selected.index,
    )


def run_joint_sampler_recovery(
    dataset: InjectionDataset,
    candidate: str,
    output_dir: str | Path,
) -> RecoveryResult:
    """Run the real joint CPU Dynesty pipeline as an opt-in recovery check."""

    if not isinstance(dataset, InjectionDataset):
        raise TypeError("dataset must be an InjectionDataset.")
    _validate_loaded_dataset(dataset)
    candidate = str(candidate)
    if candidate not in ("none", "current-2d", "relative-2d"):
        raise ValueError(
            "The production backend supports only candidates 'none', "
            "'current-2d', and 'relative-2d'; an explicit prebuilt oracle is "
            "not accepted by run_mcmc_pipeline."
        )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    df_pantheon = _synthetic_pantheon_sample(dataset)
    mock_path = None
    if candidate in ("current-2d", "relative-2d"):
        mock_path = output_dir / f"parent_mock_{dataset.dataset_id[:12]}.h5"
        _write_parent_mock_hdf5(dataset, mock_path)

    from qvc.hubble import hubble_fit

    previous_cwd = Path.cwd()
    previous_result_dir = os.environ.get("QVC_RESULT_DIR")
    previous_num_cores = hubble_fit.num_cores
    started = time.perf_counter()
    try:
        os.chdir(output_dir)
        os.environ["QVC_RESULT_DIR"] = str(output_dir / "results")
        hubble_fit.num_cores = min(2, previous_num_cores)
        (
            flat_samples,
            model_labels,
            _,
            _,
            logZ,
            logZerr,
            _,
            _,
            _,
        ) = hubble_fit.run_mcmc_pipeline(
            df_agn=dataset.selected,
            df_agn_all=dataset.parent,
            df_pantheon=df_pantheon,
            _sna_L=None,
            _sna_Lower=True,
            _sna_LogdetCov=None,
            agn_pivot_context=dataset.pivot_context,
            cosmo_model="FlatLambdaCDM",
            only_sna=False,
            only_agn=False,
            completeness=candidate in ("current-2d", "relative-2d"),
            use_full_cov=False,
            resume=False,
            speed="fastest",
            z_range=dataset.config.z_range,
            prefix="production",
            checkpoint_file_override=str(
                output_dir / f"posterior_{candidate}.h5"
            ),
            completeness_sim_file=(
                str(mock_path) if mock_path is not None else None
            ),
            completeness_mode=(
                "2d_relative_support" if candidate == "relative-2d" else "2d"
            ),
            minimal_plots=True,
            df_agn_completeness=dataset.selected,
        )
    finally:
        hubble_fit.num_cores = previous_num_cores
        os.chdir(previous_cwd)
        if previous_result_dir is None:
            os.environ.pop("QVC_RESULT_DIR", None)
        else:
            os.environ["QVC_RESULT_DIR"] = previous_result_dir

    flat_samples = np.asarray(flat_samples, dtype=float)
    model_labels = tuple(str(label) for label in model_labels)
    if (
        flat_samples.ndim != 2
        or flat_samples.shape[1] != len(model_labels)
        or np.any(~np.isfinite(flat_samples))
    ):
        raise RuntimeError(
            "Production recovery returned incomplete or nonfinite posterior samples."
        )
    posterior_median = np.median(flat_samples, axis=0)
    all_estimates = dict(zip(model_labels, posterior_median))
    estimates = {
        key: float(all_estimates[key])
        for key in TRUTH_AT_FIT_KEYS
    }
    completeness_params = build_completeness_candidate(
        candidate,
        dataset,
        output_dir / "postfit_completeness",
    )
    residuals = _residual_frame_from_estimates(
        dataset,
        estimates,
        completeness_params,
    )
    result_truth = {
        key: float(dataset.truth_at_fit_pivot[key])
        for key in TRUTH_AT_FIT_KEYS
    }
    metrics = _recovery_metrics(
        dataset,
        estimates,
        result_truth,
        residuals,
    )
    metrics["logZ"] = float(logZ)
    metrics["logZerr"] = float(logZerr)
    return RecoveryResult(
        candidate=candidate,
        backend="production",
        estimates=estimates,
        truth=result_truth,
        metrics=metrics,
        residuals=residuals,
        runtime_seconds=float(time.perf_counter() - started),
        samples=flat_samples,
        model_labels=model_labels,
    )
