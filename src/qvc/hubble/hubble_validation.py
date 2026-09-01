"""Synthetic fixed-truth validation helpers for the AGN Hubble pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
from scipy.special import expit

from qvc.hubble.completeness_mock_catalog import mock_lf_grid_per_zbin
from qvc.hubble.cuts import (
    COMPLETENESS_MAG_2500_MAX,
    COMPLETENESS_MAG_2500_MIN,
    COMPLETENESS_MAP_Z_EDGE_MAX,
    COMPLETENESS_MAP_Z_EDGE_MIN,
)
from qvc.hubble.hubble_completeness_refactored import (
    COMPLETENESS_MAG_COL,
    COMPLETENESS_MAG_ERR_COL,
)
from qvc.hubble.hubble_likelihood import sigma_lens_from_dc


ARM_NAMES = (
    "all",
    "selected_uncorrected",
    "selected_oracle",
    "selected_estimated",
)
CORNER_PARAMETERS = ("alpha_agn", "beta_agn", "Om0", "w0", "wa")
SEED_STREAMS = (
    "population",
    "scatter",
    "selection",
    "calibration_population",
    "calibration_scatter",
    "calibration_selection",
    "inference_all",
    "inference_selected_uncorrected",
    "inference_selected_oracle",
    "inference_selected_estimated",
)


@dataclass(frozen=True)
class ValidationTruth:
    h0: float = 70.0
    om0: float = 0.30
    w0: float = -1.0
    wa: float = 0.0
    alpha_agn: float = 7.0
    beta_agn: float = -1.0
    m0_agn: float = -23.0
    intrinsic_scatter_mag: float = 0.5
    log_sigma_pivot: float = -0.8
    log_sigma_scale: float = 0.2
    log_tau_pivot: float = 2.7
    log_tau_scale: float = 0.4

    def validate(self) -> None:
        values = np.asarray(list(asdict(self).values()), dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError("All injected truth values must be finite.")
        if self.h0 <= 0.0 or not 0.0 < self.om0 < 1.0:
            raise ValueError("Injected H0 and Om0 must satisfy H0>0 and 0<Om0<1.")
        if self.intrinsic_scatter_mag <= 0.0:
            raise ValueError("intrinsic_scatter_mag must be positive.")
        if self.log_sigma_scale <= 0.0 or self.log_tau_scale <= 0.0:
            raise ValueError("Predictor scales must be positive.")
        coefficient_norm = np.hypot(
            self.alpha_agn * self.log_sigma_scale,
            self.beta_agn * self.log_tau_scale,
        )
        if coefficient_norm <= 0.0:
            raise ValueError("At least one injected predictor coefficient must be nonzero.")

    def parameter_truths(self) -> dict[str, float]:
        return {
            "M0_agn": self.m0_agn,
            "alpha_agn": self.alpha_agn,
            "beta_agn": self.beta_agn,
            "log_f": float(np.log(self.intrinsic_scatter_mag)),
            "H0": self.h0,
            "Om0": self.om0,
            "w0": self.w0,
            "wa": self.wa,
        }


class AnalyticSigmoidCompleteness:
    """Pickle-safe analytic magnitude-only completeness for oracle fits."""

    magnitude_support_mode = "hard-cut"

    def __init__(
        self,
        m50: float,
        width: float,
        magnitude_support=(COMPLETENESS_MAG_2500_MIN, COMPLETENESS_MAG_2500_MAX),
    ):
        self.m50 = float(m50)
        self.width = float(width)
        self.magnitude_support = tuple(float(value) for value in magnitude_support)
        if not np.isfinite(self.m50) or not np.isfinite(self.width) or self.width <= 0.0:
            raise ValueError("The sigmoid midpoint must be finite and width positive.")
        if len(self.magnitude_support) != 2 or self.magnitude_support[0] >= self.magnitude_support[1]:
            raise ValueError("magnitude_support must contain two increasing values.")

    def __call__(self, magnitude, redshift=None):
        del redshift
        magnitude = np.asarray(magnitude, dtype=float)
        probability = expit(-(magnitude - self.m50) / self.width)
        inside = (magnitude >= self.magnitude_support[0]) & (
            magnitude <= self.magnitude_support[1]
        )
        return np.where(inside, probability, 0.0)

    @property
    def grid(self):
        return {"m50": self.m50, "width": self.width}


def analytic_completeness_params(m50: float, width: float):
    """Return the completeness tuple consumed by the Hubble likelihood."""

    model = AnalyticSigmoidCompleteness(m50, width)
    magnitude_grid = np.linspace(*model.magnitude_support, 501)
    redshift_grid = np.linspace(0.0, 4.5, 46)
    return (
        model,
        magnitude_grid,
        redshift_grid,
        float(np.diff(magnitude_grid[:2])[0]),
        float(np.diff(redshift_grid[:2])[0]),
    )


def sigmoid_detection_probability(
    magnitude,
    m50: float,
    width: float,
    magnitude_support=(COMPLETENESS_MAG_2500_MIN, COMPLETENESS_MAG_2500_MAX),
):
    """Evaluate the hard-supported injected detection probability stably."""

    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("Selection width must be finite and positive.")
    support = np.asarray(magnitude_support, dtype=float)
    if support.shape != (2,) or not np.all(np.isfinite(support)) or support[0] >= support[1]:
        raise ValueError("magnitude_support must contain two finite increasing values.")
    magnitude = np.asarray(magnitude, dtype=float)
    probability = expit(-(magnitude - float(m50)) / float(width))
    inside = (magnitude >= support[0]) & (magnitude <= support[1])
    return np.where(inside, probability, 0.0)


def derive_seed_ledger(master_seed: int, realization: int) -> dict[str, int]:
    """Derive stable, independent RNG streams for one realization."""

    sequence = np.random.SeedSequence([int(master_seed), int(realization)])
    children = sequence.spawn(len(SEED_STREAMS))
    return {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(SEED_STREAMS, children)
    }


def project_absolute_magnitude_to_predictors(
    absolute_magnitude,
    orthogonal_coordinate,
    truth: ValidationTruth,
):
    """Encode luminosity along alpha/beta and scatter only orthogonally."""

    truth.validate()
    absolute_magnitude = np.asarray(absolute_magnitude, dtype=float)
    orthogonal_coordinate = np.asarray(orthogonal_coordinate, dtype=float)
    absolute_magnitude, orthogonal_coordinate = np.broadcast_arrays(
        absolute_magnitude, orthogonal_coordinate
    )
    a = truth.alpha_agn * truth.log_sigma_scale
    b = truth.beta_agn * truth.log_tau_scale
    norm = float(np.hypot(a, b))
    delta = absolute_magnitude - truth.m0_agn
    standardized_sigma = delta * a / norm**2 - orthogonal_coordinate * b / norm
    standardized_tau = delta * b / norm**2 + orthogonal_coordinate * a / norm
    log_sigma = truth.log_sigma_pivot + truth.log_sigma_scale * standardized_sigma
    log_tau = truth.log_tau_pivot + truth.log_tau_scale * standardized_tau
    return log_sigma, log_tau


def inject_catalog_observables(
    redshift,
    absolute_magnitude,
    *,
    truth: ValidationTruth,
    rng: np.random.Generator,
    cosmology,
    object_id_start: int = 0,
    object_id_prefix: str = "validation",
) -> pd.DataFrame:
    """Create the exact DataFrame schema needed by the AGN-only likelihood."""

    truth.validate()
    redshift = np.asarray(redshift, dtype=float)
    absolute_magnitude = np.asarray(absolute_magnitude, dtype=float)
    if redshift.shape != absolute_magnitude.shape or redshift.ndim != 1:
        raise ValueError("redshift and absolute_magnitude must be matching 1D arrays.")
    if np.any(~np.isfinite(redshift)) or np.any(redshift <= 0.0):
        raise ValueError("redshift must contain finite positive values.")
    if np.any(~np.isfinite(absolute_magnitude)):
        raise ValueError("absolute_magnitude must contain finite values.")

    orthogonal = rng.normal(size=redshift.size)
    log_sigma, log_tau = project_absolute_magnitude_to_predictors(
        absolute_magnitude, orthogonal, truth
    )
    intrinsic_residual = rng.normal(
        loc=0.0, scale=truth.intrinsic_scatter_mag, size=redshift.size
    )
    lens_sigma = sigma_lens_from_dc(redshift, cosmology)
    lens_residual = rng.normal(loc=0.0, scale=lens_sigma, size=redshift.size)
    apparent_magnitude = (
        absolute_magnitude
        + cosmology.distmod(redshift).value
        + intrinsic_residual
        + lens_residual
    )
    object_ids = [
        f"{object_id_prefix}_{index:09d}"
        for index in range(object_id_start, object_id_start + redshift.size)
    ]
    zeros = np.zeros(redshift.size, dtype=float)
    frame = pd.DataFrame(
        {
            "object_id": object_ids,
            "z": redshift,
            "z_err": zeros,
            "apparent_mag_2500": apparent_magnitude,
            "apparent_mag_2500_err": zeros,
            "m_2500_dereddened": apparent_magnitude,
            "m_2500_dereddened_err": zeros,
            "m_2500_attenuated_model": apparent_magnitude,
            "m_2500_attenuated_model_err": zeros,
            COMPLETENESS_MAG_COL: apparent_magnitude,
            COMPLETENESS_MAG_ERR_COL: zeros,
            "log_sigma_uv": log_sigma,
            "log_tau_uv_rf": log_tau,
            "log_sigma_uv_std_psd": zeros,
            "log_tau_uv_rf_std_psd": zeros,
            "log_sigma_uv_log_tau_uv_rf_cov_psd": zeros,
            "injected_absolute_mag_2500": absolute_magnitude,
            "injected_intrinsic_residual_mag": intrinsic_residual,
            "injected_lensing_sigma_mag": lens_sigma,
            "injected_lensing_residual_mag": lens_residual,
            "injected_orthogonal_coordinate": orthogonal,
        }
    )
    frame.attrs.update(
        {
            "completeness_magnitude": "dereddened",
            "completeness_magnitude_source": "m_2500_dereddened",
            "completeness_magnitude_err_source": "m_2500_dereddened_err",
            "completeness_magnitude_support_mode": "hard-cut",
        }
    )
    return frame


def apply_sigmoid_selection(
    frame: pd.DataFrame,
    *,
    m50: float,
    width: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Attach probabilities/uniforms and return the detected rows."""

    result = frame.copy()
    probability = sigmoid_detection_probability(
        result["apparent_mag_2500"].to_numpy(dtype=float), m50, width
    )
    uniform = rng.random(len(result))
    result["injected_detection_probability"] = probability
    result["injected_detection_uniform"] = uniform
    result["injected_detected"] = uniform < probability
    result.attrs.update(frame.attrs)
    detected = result.loc[result["injected_detected"]].copy()
    detected.attrs.update(result.attrs)
    return result, detected


def sample_lf_chunk(
    lf_grid,
    cosmology,
    *,
    rng: np.random.Generator,
    area_deg2: float,
    z_range=(0.1, 4.0),
    apparent_magnitude_support=(14.0, 28.0),
    alpha_nu=-0.5,
    alpha_nu_scatter=0.3,
):
    """Draw one Poisson LF chunk and return z and intrinsic M2500."""

    sampled = mock_lf_grid_per_zbin(
        lf_grid,
        float(area_deg2),
        float(alpha_nu),
        float(alpha_nu_scatter),
        cosmology,
        z_range=tuple(float(value) for value in z_range),
        m2500_support=tuple(float(value) for value in apparent_magnitude_support),
        rng=rng,
        return_global=True,
        return_alpha=True,
    )
    redshift = np.asarray(sampled[4], dtype=float)
    apparent_m2500 = np.asarray(sampled[6], dtype=float)
    # The LF helper concatenates redshift bins in order.  Shuffle before any
    # fixed-size truncation so "first N" remains a draw from the full LF.
    order = rng.permutation(redshift.size)
    redshift = redshift[order]
    apparent_m2500 = apparent_m2500[order]
    absolute_m2500 = apparent_m2500 - cosmology.distmod(redshift).value
    return redshift, absolute_m2500


def generate_matched_fit_catalogs(
    lf_grid,
    cosmology,
    *,
    truth: ValidationTruth,
    n_fit: int,
    m50: float,
    selection_width: float,
    population_rng: np.random.Generator,
    scatter_rng: np.random.Generator,
    selection_rng: np.random.Generator,
    area_deg2: float = 20.0,
    z_range=(0.1, 4.0),
    apparent_magnitude_support=(14.0, 28.0),
):
    """Generate equal-size unselected and selected catalogs from one stream."""

    n_fit = int(n_fit)
    if n_fit <= 0:
        raise ValueError("n_fit must be positive.")
    parent_parts = []
    detected_parts = []
    object_id_start = 0
    for _ in range(100):
        redshift, absolute_magnitude = sample_lf_chunk(
            lf_grid,
            cosmology,
            rng=population_rng,
            area_deg2=area_deg2,
            z_range=z_range,
            apparent_magnitude_support=apparent_magnitude_support,
        )
        if redshift.size == 0:
            continue
        chunk = inject_catalog_observables(
            redshift,
            absolute_magnitude,
            truth=truth,
            rng=scatter_rng,
            cosmology=cosmology,
            object_id_start=object_id_start,
            object_id_prefix="fit",
        )
        object_id_start += len(chunk)
        chunk, detected = apply_sigmoid_selection(
            chunk, m50=m50, width=selection_width, rng=selection_rng
        )
        parent_parts.append(chunk)
        detected_parts.append(detected)
        if sum(len(part) for part in parent_parts) >= n_fit and sum(
            len(part) for part in detected_parts
        ) >= n_fit:
            break
    else:
        raise RuntimeError("Could not generate enough LF objects after 100 chunks.")

    full_parent = pd.concat(parent_parts, ignore_index=True)
    full_detected = pd.concat(detected_parts, ignore_index=True)
    parent = full_parent.iloc[:n_fit].copy()
    selected = full_detected.iloc[:n_fit].copy()
    parent.attrs.update(parent_parts[0].attrs)
    selected.attrs.update(parent_parts[0].attrs)
    generation_metadata = {
        "n_parent_generated": int(len(full_parent)),
        "n_detected_generated": int(len(full_detected)),
        "detection_fraction": float(len(full_detected) / len(full_parent)),
    }
    parent.attrs.update(generation_metadata)
    selected.attrs.update(generation_metadata)
    if len(parent) != n_fit or len(selected) != n_fit:
        raise RuntimeError("Generated validation catalogs have incorrect sizes.")
    return parent, selected


def generate_calibration_catalog(
    lf_grid,
    cosmology,
    *,
    truth: ValidationTruth,
    n_parent: int,
    m50: float,
    selection_width: float,
    population_rng: np.random.Generator,
    scatter_rng: np.random.Generator,
    selection_rng: np.random.Generator,
    area_deg2: float = 20.0,
    z_range=(0.1, 4.0),
    apparent_magnitude_support=(14.0, 28.0),
):
    """Generate an independent parent/detected pair for map estimation."""

    parts = []
    object_id_start = 0
    for _ in range(100):
        redshift, absolute_magnitude = sample_lf_chunk(
            lf_grid,
            cosmology,
            rng=population_rng,
            area_deg2=area_deg2,
            z_range=z_range,
            apparent_magnitude_support=apparent_magnitude_support,
        )
        if redshift.size == 0:
            continue
        part = inject_catalog_observables(
            redshift,
            absolute_magnitude,
            truth=truth,
            rng=scatter_rng,
            cosmology=cosmology,
            object_id_start=object_id_start,
            object_id_prefix="calibration",
        )
        object_id_start += len(part)
        parts.append(part)
        if sum(len(value) for value in parts) >= int(n_parent):
            break
    else:
        raise RuntimeError("Could not generate the requested calibration parent.")
    parent = pd.concat(parts, ignore_index=True).iloc[: int(n_parent)].copy()
    parent.attrs.update(parts[0].attrs)
    parent, detected = apply_sigmoid_selection(
        parent, m50=m50, width=selection_width, rng=selection_rng
    )
    return parent, detected


def write_completeness_parent_hdf5(
    frame: pd.DataFrame,
    path: Path,
    *,
    z_range=(0.1, 4.0),
) -> Path:
    """Write the minimal HDF5 schema used by the empirical map builder."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    magnitude = frame["apparent_mag_2500"].to_numpy(dtype=float)
    redshift = frame["z"].to_numpy(dtype=float)
    # The production map uses a fixed 0--4.5 redshift grid and validates that
    # the mock declares that full interpolation domain.  These two entries sit
    # outside the magnitude histogram, so they establish numerical domain
    # coverage without changing any parent-count bin.
    magnitude = np.concatenate((magnitude, [17.0, 100.0]))
    redshift = np.concatenate(
        (redshift, [COMPLETENESS_MAP_Z_EDGE_MIN, COMPLETENESS_MAP_Z_EDGE_MAX])
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "apparent_mag_2500",
            data=magnitude,
            compression="gzip",
        )
        handle.create_dataset(
            "z", data=redshift, compression="gzip"
        )
        handle.attrs["mock_count_scale"] = 1.0
        handle.attrs["mock_redshift_min"] = float(COMPLETENESS_MAP_Z_EDGE_MIN)
        handle.attrs["mock_redshift_max"] = float(COMPLETENESS_MAP_Z_EDGE_MAX)
        handle.attrs["generator"] = "qvc.hubble.hubble_validation"
        handle.attrs["support_sentinels_outside_magnitude_map"] = 2
    return path


def config_fingerprint(config: dict) -> str:
    """Return a stable campaign configuration digest."""

    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def posterior_summary_row(
    flat_samples,
    model_labels: Iterable[str],
    *,
    arm: str,
    realization: int,
    checkpoint_file: Path,
    truth: ValidationTruth,
    n_fit: int,
    n_parent_generated: int,
    detection_fraction: float,
    parameter_truths: dict[str, float] | None = None,
) -> dict:
    """Flatten posterior quantiles and truth metadata into one CSV row."""

    flat_samples = np.asarray(flat_samples, dtype=float)
    labels = tuple(str(label) for label in model_labels)
    if flat_samples.ndim != 2 or flat_samples.shape[1] != len(labels):
        raise ValueError("flat_samples and model_labels have incompatible shapes.")
    quantiles = np.percentile(flat_samples, [2.5, 16.0, 50.0, 84.0, 97.5], axis=0)
    row = {
        "realization": int(realization),
        "arm": str(arm),
        "status": "complete",
        "checkpoint_file": str(checkpoint_file),
        "posterior_sample_count": int(flat_samples.shape[0]),
        "n_fit": int(n_fit),
        "n_parent_generated": int(n_parent_generated),
        "detection_fraction": float(detection_fraction),
    }
    truth_values = truth.parameter_truths() if parameter_truths is None else dict(parameter_truths)
    for label, values in zip(labels, quantiles.T):
        for suffix, value in zip(("q025", "q16", "q50", "q84", "q975"), values):
            row[f"{label}_{suffix}"] = float(value)
        if label in truth_values:
            row[f"truth_{label}"] = float(truth_values[label])
    return row


def ensemble_summary(recovery: pd.DataFrame) -> pd.DataFrame:
    """Compute bias, pull, and interval coverage from successful fit rows."""

    rows = []
    complete = recovery.loc[recovery["status"] == "complete"].copy()
    parameters = sorted(
        column.removesuffix("_q50")
        for column in complete.columns
        if column.endswith("_q50")
        and f"truth_{column.removesuffix('_q50')}" in complete.columns
    )
    for arm in ARM_NAMES:
        arm_rows = complete.loc[complete["arm"] == arm]
        for parameter in parameters:
            estimate = arm_rows[f"{parameter}_q50"].to_numpy(dtype=float)
            truth_value = arm_rows[f"truth_{parameter}"].to_numpy(dtype=float)
            q025 = arm_rows[f"{parameter}_q025"].to_numpy(dtype=float)
            q16 = arm_rows[f"{parameter}_q16"].to_numpy(dtype=float)
            q84 = arm_rows[f"{parameter}_q84"].to_numpy(dtype=float)
            q975 = arm_rows[f"{parameter}_q975"].to_numpy(dtype=float)
            finite = np.isfinite(estimate) & np.isfinite(truth_value)
            estimate, truth_value = estimate[finite], truth_value[finite]
            q025, q16, q84, q975 = q025[finite], q16[finite], q84[finite], q975[finite]
            residual = estimate - truth_value
            sigma = 0.5 * (q84 - q16)
            valid_pull = np.isfinite(sigma) & (sigma > 0.0)
            pulls = residual[valid_pull] / sigma[valid_pull]
            rows.append(
                {
                    "arm": arm,
                    "parameter": parameter,
                    "n_success": int(estimate.size),
                    "mean_bias": float(np.mean(residual)) if residual.size else np.nan,
                    "median_bias": float(np.median(residual)) if residual.size else np.nan,
                    "rmse": float(np.sqrt(np.mean(residual**2))) if residual.size else np.nan,
                    "median_absolute_error": float(np.median(np.abs(residual))) if residual.size else np.nan,
                    "pull_mean": float(np.mean(pulls)) if pulls.size else np.nan,
                    "pull_std": float(np.std(pulls, ddof=1)) if pulls.size > 1 else np.nan,
                    "coverage_68": float(np.mean((q16 <= truth_value) & (truth_value <= q84))) if residual.size else np.nan,
                    "coverage_95": float(np.mean((q025 <= truth_value) & (truth_value <= q975))) if residual.size else np.nan,
                }
            )
    return pd.DataFrame(
        rows,
        columns=(
            "arm",
            "parameter",
            "n_success",
            "mean_bias",
            "median_bias",
            "rmse",
            "median_absolute_error",
            "pull_mean",
            "pull_std",
            "coverage_68",
            "coverage_95",
        ),
    )


def write_dataframe_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV by replacement so interrupted campaigns keep valid state."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def collect_recovery_fragments(campaign_dir: Path) -> pd.DataFrame:
    """Collect seed-local recovery rows into the canonical campaign table."""

    campaign_dir = Path(campaign_dir)
    recovery_path = campaign_dir / "recovery.csv"
    existing = pd.read_csv(recovery_path) if recovery_path.is_file() else pd.DataFrame()
    fragment_paths = sorted((campaign_dir / "runs").glob("seed_*/recovery.csv"))
    if not fragment_paths:
        if not existing.empty:
            return existing
        return pd.DataFrame(columns=("realization", "arm", "status"))

    fragments = []
    required = {"realization", "arm", "status"}
    for path in fragment_paths:
        fragment = pd.read_csv(path)
        missing = sorted(required - set(fragment.columns))
        if missing:
            raise ValueError(f"Recovery fragment {path} is missing columns: {missing}")
        fragments.append(fragment)
    fragmented = pd.concat(fragments, ignore_index=True, sort=False)
    duplicates = fragmented.duplicated(["realization", "arm"], keep=False)
    if duplicates.any():
        pairs = (
            fragmented.loc[duplicates, ["realization", "arm"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(f"Duplicate realization/arm recovery rows: {pairs}")
    if not existing.empty:
        missing = sorted(required - set(existing.columns))
        if missing:
            raise ValueError(f"Campaign recovery table {recovery_path} is missing columns: {missing}")
        fragment_keys = set(
            zip(fragmented["realization"].astype(int), fragmented["arm"].astype(str))
        )
        existing_keys = list(
            zip(existing["realization"].astype(int), existing["arm"].astype(str))
        )
        existing = existing.loc[
            [key not in fragment_keys for key in existing_keys]
        ].copy()
    recovery = pd.concat([existing, fragmented], ignore_index=True, sort=False)
    recovery = recovery.sort_values(["realization", "arm"], kind="stable").reset_index(drop=True)
    write_dataframe_atomic(recovery, recovery_path)
    return recovery


def incomplete_recovery_report(recovery: pd.DataFrame, configuration: dict) -> pd.DataFrame:
    """Return failed and absent realization/arm pairs for a campaign."""

    columns = ("realization", "arm", "status", "error_type", "error_message")
    required_config = {"seed_start", "n_runs", "arms"}
    if not required_config.issubset(configuration):
        return pd.DataFrame(columns=columns)

    failed = recovery.loc[recovery.get("status", pd.Series(dtype=str)) != "complete"].copy()
    for column in columns:
        if column not in failed:
            failed[column] = pd.NA
    failed = failed.loc[:, columns]

    observed = set()
    if {"realization", "arm"}.issubset(recovery.columns):
        observed = set(
            zip(
                recovery["realization"].astype(int),
                recovery["arm"].astype(str),
            )
        )
    missing_rows = []
    for realization in range(
        int(configuration["seed_start"]),
        int(configuration["seed_start"]) + int(configuration["n_runs"]),
    ):
        for arm in configuration["arms"]:
            if (realization, str(arm)) not in observed:
                missing_rows.append(
                    {
                        "realization": realization,
                        "arm": str(arm),
                        "status": "missing",
                        "error_type": pd.NA,
                        "error_message": pd.NA,
                    }
                )
    report = pd.concat(
        [failed, pd.DataFrame(missing_rows, columns=columns)],
        ignore_index=True,
    )
    if report.empty:
        return pd.DataFrame(columns=columns)
    return report.sort_values(["realization", "arm"], kind="stable").reset_index(drop=True)


__all__ = [
    "ARM_NAMES",
    "CORNER_PARAMETERS",
    "AnalyticSigmoidCompleteness",
    "ValidationTruth",
    "analytic_completeness_params",
    "apply_sigmoid_selection",
    "config_fingerprint",
    "collect_recovery_fragments",
    "derive_seed_ledger",
    "ensemble_summary",
    "generate_calibration_catalog",
    "generate_matched_fit_catalogs",
    "inject_catalog_observables",
    "incomplete_recovery_report",
    "posterior_summary_row",
    "project_absolute_magnitude_to_predictors",
    "sigmoid_detection_probability",
    "write_completeness_parent_hdf5",
    "write_dataframe_atomic",
]
