"""Offline eBOSS photometry-only host-selection sensitivity products.

This module is deliberately separated from the Hubble likelihood.  It accepts
historically transformed targeting features, fits one eBOSS colour head,
evaluates paired host/no-host mocks, and writes immutable two-
dimensional completeness maps.  Cosmology code may load the maps, but never
the features, target flags, qsogen draws, support model, or fitted parameters.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.special import expit

from qvc.hubble.hubble_completeness_refactored import Completeness2D


ARTIFACT_SCHEMA = "qvc_eboss_color_completeness_v6"
PREPARED_CATALOG_SCHEMA = "qvc_eboss_color_prepared_v6"
COLOR_PROGRAMS = ("eboss",)
COMPLETENESS_MODES = ("old", "host-removal")
MIN_EFFECTIVE_SAMPLE_SIZE = 200.0
MIN_COLOR_HOST = 1.0e-6
MAX_OUT_OF_SUPPORT_WEIGHT = 0.05
MHD_PAIR_ATOL = 1.0e-12


EBOSS_FEATURE_NAMES = (
    "i_psf", "u-g", "g-r", "r-i", "i-z", "i-W1", "W1-W2",
    "log_depth_sdss", "log_depth_wise", "missing_W1", "missing_W2",
)


def signed_luptitude(flux: Any, softening: Any) -> np.ndarray:
    """Return finite 22.5-zero-point luptitudes for signed nanomaggy flux."""

    value = np.asarray(flux, dtype=float)
    b = np.asarray(softening, dtype=float)
    if np.any(~np.isfinite(value)) or np.any(~np.isfinite(b)) or np.any(b <= 0.0):
        raise ValueError("Signed luptitudes require finite flux and positive softening.")
    return 22.5 - (2.5 / np.log(10.0)) * (
        np.arcsinh(value / (2.0 * b)) + np.log(b)
    )


def eboss_target_features(
    sdss_flux: Any,
    sdss_error: Any,
    wise_flux: Any,
    wise_error: Any,
    missingness: Any,
    sdss_softening: Any,
    wise_softening: Any,
) -> np.ndarray:
    """Low-capacity signed-flux-safe projection of XDQSOz+WISE inputs."""

    sf, se = np.asarray(sdss_flux, float), np.asarray(sdss_error, float)
    wf, we = np.asarray(wise_flux, float), np.asarray(wise_error, float)
    missing = np.asarray(missingness, bool)
    if sf.shape[-1] != 5 or se.shape != sf.shape or wf.shape[-1] != 2 or we.shape != wf.shape:
        raise ValueError("eBOSS targeting requires aligned ugriz and W1/W2 flux/error arrays.")
    if missing.shape != wf.shape:
        raise ValueError("WISE missingness must match W1/W2 flux shape.")
    if np.any(~np.isfinite(sf)) or np.any(~np.isfinite(se)) or np.any(se <= 0.0):
        raise ValueError("eBOSS ugriz fluxes/errors must be finite with positive errors.")
    if np.any(~np.isfinite(we)) or np.any(we <= 0.0):
        raise ValueError("eBOSS WISE placeholder errors must be finite and positive.")
    sdss_mag = signed_luptitude(sf, sdss_softening)
    wise_mag = signed_luptitude(np.where(missing, 0.0, wf), wise_softening)
    return np.column_stack((
        sdss_mag[:, 3],
        sdss_mag[:, 0] - sdss_mag[:, 1],
        sdss_mag[:, 1] - sdss_mag[:, 2],
        sdss_mag[:, 2] - sdss_mag[:, 3],
        sdss_mag[:, 3] - sdss_mag[:, 4],
        sdss_mag[:, 3] - wise_mag[:, 0],
        wise_mag[:, 0] - wise_mag[:, 1],
        np.log10(np.median(se, axis=-1)),
        np.log10(np.median(we, axis=-1)),
        missing.astype(float),
    ))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_1d_finite(name: str, value: Any, *, minimum_size: int = 1) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size < minimum_size or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array.")
    return array


def validate_cut_manifest(manifest: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    stages = (
        "intrinsic_support",
        "target_eligibility",
        "target_selection",
        "downstream_survival",
        "hd_analysis",
    )
    if set(manifest) != set(stages):
        raise ValueError(f"Cut manifest must contain exactly {stages}.")
    normalized = {stage: [str(item) for item in manifest[stage]] for stage in stages}
    flattened = [item for stage in stages for item in normalized[stage]]
    duplicates = sorted({item for item in flattened if flattened.count(item) > 1})
    if duplicates:
        raise ValueError(f"Each cut must belong to exactly one stage: {duplicates}.")
    return normalized


def decode_target_marks(
    *, color_targeted: Any, alternative_targeted: Any, color_eligible: Any
) -> np.ndarray:
    """Return conservative colour/alternative targeting marks.

    Ineligible rows are ``unknown`` even if a catalogue happens to carry a
    colour bit: they cannot be Bernoulli trials for that program's colour head.
    """

    color = np.asarray(color_targeted, dtype=bool)
    alt = np.asarray(alternative_targeted, dtype=bool)
    eligible = np.asarray(color_eligible, dtype=bool)
    if color.shape != alt.shape or color.shape != eligible.shape:
        raise ValueError("Target flags and eligibility must have identical shapes.")
    out = np.full(color.shape, "unknown", dtype="U10")
    out[eligible & color & ~alt] = "color_only"
    out[eligible & ~color & alt] = "alt_only"
    out[eligible & color & alt] = "both"
    return out


def assert_paired_nuclear_state(
    m_hd_host: Any,
    m_hd_nohost: Any,
    luminosity_host: Any,
    luminosity_nohost: Any,
    *,
    atol: float = MHD_PAIR_ATOL,
) -> None:
    for label, host, nohost in (
        ("m_hd", m_hd_host, m_hd_nohost),
        ("intrinsic nuclear luminosity", luminosity_host, luminosity_nohost),
    ):
        left = np.asarray(host, dtype=float)
        right = np.asarray(nohost, dtype=float)
        if left.shape != right.shape or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise ValueError(f"Paired {label} arrays must have identical finite shapes.")
        if not np.allclose(left, right, rtol=0.0, atol=float(atol)):
            maximum = float(np.max(np.abs(left - right)))
            raise ValueError(f"Host synthesis modified {label}; maximum difference={maximum:.3g}.")


def apply_paired_flux_noise(
    flux_host: Any,
    flux_nohost: Any,
    empirical_error: Any,
    standard_normal: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one empirical error vector and one noise draw to both branches.

    The empirical error vector is the latent observing/depth state.  It is
    deliberately not adjusted as a function of either branch's flux: doing so
    would require a calibrated photon-noise model and would extrapolate beyond
    the survey measurements at the bright end.
    """

    host = np.asarray(flux_host, dtype=float)
    nohost = np.asarray(flux_nohost, dtype=float)
    error = np.asarray(empirical_error, dtype=float)
    epsilon = np.asarray(standard_normal, dtype=float)
    if host.shape != nohost.shape or host.shape != error.shape or host.shape != epsilon.shape:
        raise ValueError("Paired fluxes, empirical errors, and standard-normal draws must match.")
    if not np.all(np.isfinite(error)) or np.any(error <= 0.0):
        raise ValueError("Empirical photometric errors must be finite and positive.")
    perturbation = error * epsilon
    return host + perturbation, nohost + perturbation


@dataclass(frozen=True)
class LogisticColorHead:
    program: str
    feature_names: tuple[str, ...]
    median: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    l2: float

    def __post_init__(self) -> None:
        if self.program not in COLOR_PROGRAMS:
            raise ValueError(f"Unknown color-head program {self.program!r}.")
        forbidden = {"z", "redshift", "spec_z"}
        if any(str(name).strip().lower() in forbidden for name in self.feature_names):
            raise ValueError("Explicit redshift is forbidden from S_C features.")
        width = len(self.feature_names)
        for name, value in (("median", self.median), ("scale", self.scale), ("coefficients", self.coefficients)):
            array = np.asarray(value, dtype=float)
            if array.shape != (width,) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain one finite value per feature.")
        if np.any(np.asarray(self.scale) <= 0.0) or not np.isfinite(self.intercept):
            raise ValueError("Color-head scaling and intercept are invalid.")

    @classmethod
    def fit(
        cls,
        program: str,
        feature_names: Sequence[str],
        features: Any,
        success: Any,
        *,
        l2: float = 1.0,
    ) -> "LogisticColorHead":
        x = np.asarray(features, dtype=float)
        y = np.asarray(success, dtype=float)
        names = tuple(str(name) for name in feature_names)
        if x.ndim != 2 or x.shape[1] != len(names) or y.shape != (x.shape[0],):
            raise ValueError("Training feature/label shapes are inconsistent.")
        if x.shape[0] < 20 or not np.all(np.isfinite(x)) or not np.all(np.isin(y, (0.0, 1.0))):
            raise ValueError("Color-head training requires >=20 finite binary trials.")
        if np.unique(y).size != 2:
            raise ValueError("Color-head training requires successes and failures.")
        median = np.median(x, axis=0)
        q25, q75 = np.percentile(x, [25.0, 75.0], axis=0)
        scale = q75 - q25
        scale = np.where(scale > 1.0e-12, scale, 1.0)
        xs = (x - median) / scale

        def objective(theta):
            logits = theta[0] + xs @ theta[1:]
            loss = np.sum(np.logaddexp(0.0, logits) - y * logits)
            loss += 0.5 * float(l2) * np.sum(theta[1:] ** 2)
            probability = expit(logits)
            gradient = np.concatenate((
                [np.sum(probability - y)],
                xs.T @ (probability - y) + float(l2) * theta[1:],
            ))
            return float(loss), gradient

        initial = np.zeros(x.shape[1] + 1, dtype=float)
        initial[0] = np.log(np.mean(y) / (1.0 - np.mean(y)))
        result = minimize(objective, initial, jac=True, method="L-BFGS-B")
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"Color-head optimization failed: {result.message}")
        return cls(program, names, median, scale, result.x[1:], float(result.x[0]), float(l2))

    def predict(self, features: Any) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        if x.shape[-1] != len(self.feature_names) or not np.all(np.isfinite(x)):
            raise ValueError("Color-head evaluation features are invalid.")
        value = expit(self.intercept + ((x - self.median) / self.scale) @ self.coefficients)
        return np.clip(np.asarray(value, dtype=float), 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "program": self.program,
            "feature_names": list(self.feature_names),
            "median": np.asarray(self.median).tolist(),
            "scale": np.asarray(self.scale).tolist(),
            "coefficients": np.asarray(self.coefficients).tolist(),
            "intercept": self.intercept,
            "l2": self.l2,
        }


@dataclass(frozen=True)
class EmpiricalFeatureSupport:
    median: np.ndarray
    scale: np.ndarray
    training_scaled: np.ndarray
    k: int
    threshold: float
    allowed_patterns: tuple[str, ...]

    @classmethod
    def fit(cls, features: Any, patterns: Sequence[str]) -> "EmpiricalFeatureSupport":
        x = np.asarray(features, dtype=float)
        pattern = np.asarray(patterns).astype(str)
        if x.ndim != 2 or x.shape[0] < 20 or pattern.shape != (x.shape[0],):
            raise ValueError("Support training arrays are inconsistent or too small.")
        if not np.all(np.isfinite(x)):
            raise ValueError("Support features must be finite after program transformation.")
        median = np.median(x, axis=0)
        q25, q75 = np.percentile(x, [25.0, 75.0], axis=0)
        scale = np.where((q75 - q25) > 1.0e-12, q75 - q25, 1.0)
        scaled = (x - median) / scale
        k = min(x.shape[0] - 1, max(20, int(np.ceil(np.sqrt(x.shape[0])))))
        distances, _ = cKDTree(scaled).query(scaled, k=k + 1)
        threshold = float(np.percentile(distances[:, -1], 99.0))
        return cls(median, scale, scaled, k, threshold, tuple(sorted(set(pattern))))

    def contains(self, features: Any, patterns: Sequence[str]) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        pattern = np.asarray(patterns).astype(str)
        if x.ndim != 2 or x.shape[1] != self.training_scaled.shape[1] or pattern.shape != (x.shape[0],):
            raise ValueError("Support evaluation arrays have invalid shapes.")
        finite = np.all(np.isfinite(x), axis=1)
        distance = np.full(x.shape[0], np.inf, dtype=float)
        if np.any(finite):
            scaled = (x[finite] - self.median) / self.scale
            queried, _ = cKDTree(self.training_scaled).query(scaled, k=self.k)
            distance[finite] = queried[:, -1] if queried.ndim == 2 else queried
        pattern_ok = np.isin(pattern, np.asarray(self.allowed_patterns))
        return finite & pattern_ok & (distance <= self.threshold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "median": np.asarray(self.median).tolist(),
            "scale": np.asarray(self.scale).tolist(),
            "k": self.k,
            "threshold": self.threshold,
            "allowed_patterns": list(self.allowed_patterns),
        }


def leave_one_channel_out_closure(
    program: str,
    feature_names: Sequence[str],
    features: Any,
    success: Any,
    channels: Sequence[str],
    *,
    l2: float = 1.0,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Acceptance diagnostic for alternative-channel transport."""

    x = np.asarray(features, float)
    y = np.asarray(success, float)
    channel = np.asarray(channels).astype(str)
    if channel.shape != y.shape or x.shape[0] != y.size:
        raise ValueError("Alternative-channel closure arrays are inconsistent.")
    diagnostics = []
    testable = [name for name in sorted(set(channel)) if np.count_nonzero(channel == name) >= 10]
    if len(testable) < 2:
        return [{
            "status": "not_testable",
            "reason": "fewer than two alternative channels have at least ten trials",
            "channel_counts": {name: int(np.count_nonzero(channel == name)) for name in sorted(set(channel))},
        }]
    for held_out in testable:
        test = channel == held_out
        train = ~test
        if np.count_nonzero(test) < 10 or np.count_nonzero(train) < 20 or np.unique(y[train]).size != 2:
            raise ValueError(f"Cannot run leave-one-channel-out closure for {program}/{held_out}.")
        head = LogisticColorHead.fit(program, feature_names, x[train], y[train], l2=l2)
        probability = head.predict(x[test])
        residual = float(np.sum(y[test] - probability))
        sigma = float(np.sqrt(np.sum(probability * (1.0 - probability))))
        z_score = residual / max(sigma, 1.0e-12)
        seed_material = f"{program}:{held_out}:closure-v1".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        bootstrap_residual = np.empty(4096, dtype=float)
        for start in range(0, 4096, 256):
            draw = rng.random((min(256, 4096 - start), probability.size)) < probability
            bootstrap_residual[start:start + len(draw)] = np.sum(draw - probability, axis=1)
        p_value = float(
            (1 + np.count_nonzero(np.abs(bootstrap_residual) >= abs(residual)))
            / (1 + bootstrap_residual.size)
        )
        failed = bool(abs(z_score) > 5.0 or p_value < 0.01)
        row = {"channel": held_out, "n": int(np.count_nonzero(test)),
               "z_score": z_score, "bootstrap_p": p_value,
               "status": "failed" if failed else "passed"}
        diagnostics.append(row)
        if failed and strict:
            raise ValueError(
                f"Alternative-channel color-head closure failed for {program}/{held_out}: "
                f"z={z_score:.2f}, held-out p={p_value:.3g}."
            )
    return diagnostics


def weighted_effective_sample_size(weights: Any) -> float:
    weights = _as_1d_finite("weights", weights)
    if np.any(weights < 0.0) or np.sum(weights) <= 0.0:
        raise ValueError("Importance weights must be nonnegative with positive sum.")
    return float(np.sum(weights) ** 2 / np.sum(weights**2))


def marginalize_paired_color_completeness(
    *,
    m_bin: Any,
    z_bin: Any,
    n_m: int,
    n_z: int,
    weights: Any,
    probability_host: Any,
    probability_nohost: Any,
    supported_host: Any,
    supported_nohost: Any,
) -> dict[str, np.ndarray]:
    m_index = np.asarray(m_bin, dtype=int)
    z_index = np.asarray(z_bin, dtype=int)
    weight = _as_1d_finite("weights", weights)
    ph = _as_1d_finite("probability_host", probability_host)
    pn = _as_1d_finite("probability_nohost", probability_nohost)
    sh = np.asarray(supported_host, dtype=bool)
    sn = np.asarray(supported_nohost, dtype=bool)
    shape = weight.shape
    if any(array.shape != shape for array in (m_index, z_index, ph, pn, sh, sn)):
        raise ValueError("Mock marginalization arrays must share one object axis.")
    if np.any(weight < 0.0) or np.any((ph < 0.0) | (ph > 1.0)) or np.any((pn < 0.0) | (pn > 1.0)):
        raise ValueError("Mock weights/probabilities are outside their bounds.")
    result = {
        name: np.full((n_m, n_z), np.nan, dtype=float)
        for name in (
            "C_color_host", "C_color_nohost", "uncertainty_host", "uncertainty_nohost",
            "n_eff_host", "n_eff_nohost", "out_of_support_fraction_host",
            "out_of_support_fraction_nohost",
        )
    }
    support_host_grid = np.zeros((n_m, n_z), dtype=bool)
    support_nohost_grid = np.zeros((n_m, n_z), dtype=bool)
    for i in range(n_m):
        for j in range(n_z):
            use = (m_index == i) & (z_index == j)
            if not np.any(use) or np.sum(weight[use]) <= 0.0:
                continue
            w = weight[use]
            total = np.sum(w)
            neff = weighted_effective_sample_size(w)
            for branch, probability, supported in (("host", ph[use], sh[use]), ("nohost", pn[use], sn[use])):
                mean = float(np.sum(w * probability) / total)
                variance = float(np.sum(w * (probability - mean) ** 2) / total)
                oos = float(np.sum(w * ~supported) / total)
                result[f"C_color_{branch}"][i, j] = mean
                result[f"uncertainty_{branch}"][i, j] = np.sqrt(max(variance, 0.0) / neff)
                result[f"n_eff_{branch}"][i, j] = neff
                result[f"out_of_support_fraction_{branch}"][i, j] = oos
                (support_host_grid if branch == "host" else support_nohost_grid)[i, j] = oos <= MAX_OUT_OF_SUPPORT_WEIGHT
    result["support_host"] = support_host_grid
    result["support_nohost"] = support_nohost_grid
    result["delta_C_host_color"] = result["C_color_host"] - result["C_color_nohost"]
    return result


def _artifact_digest(metadata_json: str, arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(metadata_json.encode("utf-8"))
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def hash_completeness_2d(model: Completeness2D) -> str:
    """Content hash for the referenced empirical LF/count baseline map."""

    m = np.asarray(model.mag_centers, dtype=float)
    z = np.asarray(model.z_centers, dtype=float)
    mm, zz = np.meshgrid(m, z, indexing="ij")
    values = np.asarray(model(mm, zz), dtype=float)
    return _artifact_digest(
        _canonical_json({"magnitude_support": list(map(float, model.magnitude_support))}),
        {"magnitude_grid": m, "redshift_grid": z, "C_old": values},
    )


@dataclass(frozen=True)
class ColorCompletenessArtifact:
    magnitude_grid: np.ndarray
    redshift_grid: np.ndarray
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]
    content_hash: str

    @property
    def valid_mask(self) -> np.ndarray:
        arrays = self.arrays
        return (
            np.asarray(arrays["support_host"], dtype=bool)
            & np.asarray(arrays["support_nohost"], dtype=bool)
            & (np.asarray(arrays["n_eff_host"], dtype=float) >= MIN_EFFECTIVE_SAMPLE_SIZE)
            & (np.asarray(arrays["n_eff_nohost"], dtype=float) >= MIN_EFFECTIVE_SAMPLE_SIZE)
            & np.isfinite(np.asarray(arrays["uncertainty_host"], dtype=float))
            & np.isfinite(np.asarray(arrays["uncertainty_nohost"], dtype=float))
        )

    def validate(self) -> None:
        magnitude = _as_1d_finite("magnitude_grid", self.magnitude_grid, minimum_size=2)
        redshift = _as_1d_finite("redshift_grid", self.redshift_grid, minimum_size=2)
        if np.any(np.diff(magnitude) <= 0.0) or np.any(np.diff(redshift) <= 0.0):
            raise ValueError("Artifact axes must be strictly increasing.")
        shape = (magnitude.size, redshift.size)
        required = {
            "C_color_host", "C_color_nohost", "delta_C_host_color",
            "uncertainty_host", "uncertainty_nohost", "n_eff_host", "n_eff_nohost",
            "support_host", "support_nohost", "out_of_support_fraction_host",
            "out_of_support_fraction_nohost",
        }
        missing = required - set(self.arrays)
        if missing:
            raise ValueError(f"Color artifact is missing datasets: {sorted(missing)}.")
        for name in required:
            if np.asarray(self.arrays[name]).shape != shape:
                raise ValueError(f"Artifact dataset {name!r} has the wrong shape.")
        metadata_json = _canonical_json(dict(self.metadata))
        digest_arrays = {"magnitude_grid": magnitude, "redshift_grid": redshift, **{name: np.asarray(value) for name, value in self.arrays.items()}}
        if _artifact_digest(metadata_json, digest_arrays) != self.content_hash:
            raise ValueError("Color-completeness artifact content hash mismatch.")
        host = np.asarray(self.arrays["C_color_host"], dtype=float)
        nohost = np.asarray(self.arrays["C_color_nohost"], dtype=float)
        if np.any(~np.isfinite(host)) or np.any(~np.isfinite(nohost)) or np.any((host < 0.0) | (host > 1.0)) or np.any((nohost < 0.0) | (nohost > 1.0)):
            raise ValueError("Color completeness probabilities must be finite in [0,1].")
        if not np.allclose(self.arrays["delta_C_host_color"], host - nohost, rtol=0.0, atol=1.0e-12):
            raise ValueError("Stored delta_C_host_color is inconsistent.")


def write_color_completeness_artifact(
    path: str | Path,
    *,
    magnitude_grid: Any,
    redshift_grid: Any,
    products: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing artifact {output}.")
    magnitude = np.asarray(magnitude_grid, dtype=float)
    redshift = np.asarray(redshift_grid, dtype=float)
    arrays = {name: np.asarray(value) for name, value in products.items()}
    metadata_json = _canonical_json(dict(metadata))
    digest_arrays = {"magnitude_grid": magnitude, "redshift_grid": redshift, **arrays}
    content_hash = _artifact_digest(metadata_json, digest_arrays)
    candidate = ColorCompletenessArtifact(magnitude, redshift, arrays, dict(metadata), content_hash)
    candidate.validate()
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "x") as handle:
        handle.attrs["schema"] = ARTIFACT_SCHEMA
        handle.attrs["content_hash_sha256"] = content_hash
        handle.attrs["metadata_json"] = metadata_json
        handle.create_dataset("magnitude_grid", data=magnitude)
        handle.create_dataset("redshift_grid", data=redshift)
        for name, value in arrays.items():
            handle.create_dataset(name, data=value)
    return content_hash


def read_color_completeness_artifact(path: str | Path) -> ColorCompletenessArtifact:
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("schema", "")) != ARTIFACT_SCHEMA:
            raise ValueError("Unsupported color-completeness artifact schema.")
        metadata = json.loads(str(handle.attrs["metadata_json"]))
        arrays = {
            name: np.asarray(handle[name])
            for name in handle
            if name not in {"magnitude_grid", "redshift_grid"}
        }
        artifact = ColorCompletenessArtifact(
            np.asarray(handle["magnitude_grid"], dtype=float),
            np.asarray(handle["redshift_grid"], dtype=float),
            arrays,
            metadata,
            str(handle.attrs["content_hash_sha256"]),
        )
    artifact.validate()
    return artifact


def _required_stencil_mask(artifact: ColorCompletenessArtifact, magnitude: Any, redshift: Any) -> np.ndarray:
    m = np.asarray(magnitude, dtype=float)
    z = np.asarray(redshift, dtype=float)
    if m.shape != z.shape or not np.all(np.isfinite(m)) or not np.all(np.isfinite(z)):
        raise ValueError("HD preflight coordinates must be matching finite arrays.")
    mg, zg = artifact.magnitude_grid, artifact.redshift_grid
    if np.any((m < mg[0]) | (m > mg[-1]) | (z < zg[0]) | (z > zg[-1])):
        raise ValueError("Color-dependent completeness cannot extrapolate outside its grid.")
    i1 = np.searchsorted(mg, m, side="right").clip(1, len(mg) - 1)
    j1 = np.searchsorted(zg, z, side="right").clip(1, len(zg) - 1)
    valid = artifact.valid_mask
    return valid[i1 - 1, j1 - 1] & valid[i1, j1 - 1] & valid[i1 - 1, j1] & valid[i1, j1]


def color_support_stencil_mask(
    artifact: ColorCompletenessArtifact, magnitude: Any, redshift: Any
) -> np.ndarray:
    """Return the explicit analysis mask for valid four-cell interpolation stencils.

    Coordinates outside the artifact grid return ``False``.  This helper is
    used before all matched Hubble modes; it never enters the likelihood.
    """

    m = np.asarray(magnitude, dtype=float)
    z = np.asarray(redshift, dtype=float)
    if m.shape != z.shape:
        raise ValueError("Color-support coordinates must share one shape.")
    output = np.zeros(m.shape, dtype=bool)
    finite = np.isfinite(m) & np.isfinite(z)
    inside = finite & (m >= artifact.magnitude_grid[0]) & (m <= artifact.magnitude_grid[-1]) \
        & (z >= artifact.redshift_grid[0]) & (z <= artifact.redshift_grid[-1])
    if np.any(inside):
        output[inside] = _required_stencil_mask(artifact, m[inside], z[inside])
    return output


def build_hubble_completeness_map(
    mode: str,
    *,
    old_completeness: Completeness2D,
    artifact: ColorCompletenessArtifact | None = None,
    hd_magnitude: Any | None = None,
    hd_redshift: Any | None = None,
    completeness_floor: float = 1.0e-12,
) -> Completeness2D:
    if mode not in COMPLETENESS_MODES:
        raise ValueError(f"Unknown completeness mode {mode!r}; expected {COMPLETENESS_MODES}.")
    if mode == "old":
        return old_completeness
    if artifact is None:
        raise ValueError(f"Completeness mode {mode!r} requires a color artifact.")
    artifact.validate()
    expected_old_hash = str(artifact.metadata.get("old_completeness_hash", ""))
    if not expected_old_hash:
        raise ValueError("Color artifact does not reference C_old by content hash.")
    actual_old_hash = hash_completeness_2d(old_completeness)
    if expected_old_hash != actual_old_hash:
        raise ValueError("Color artifact references a different C_old completeness map.")
    if hd_magnitude is not None or hd_redshift is not None:
        if hd_magnitude is None or hd_redshift is None or not np.all(_required_stencil_mask(artifact, hd_magnitude, hd_redshift)):
            raise ValueError("At least one HD object requires an unsupported color-map interpolation stencil.")
    host = np.asarray(artifact.arrays["C_color_host"], dtype=float)
    if np.any(host <= MIN_COLOR_HOST):
        raise ValueError("Host-removal ratio has C_color_host <= 1e-6.")
    nohost = np.asarray(artifact.arrays["C_color_nohost"], dtype=float)
    m_mesh, z_mesh = np.meshgrid(
        artifact.magnitude_grid, artifact.redshift_grid, indexing="ij"
    )
    old = np.asarray(old_completeness(m_mesh, z_mesh), dtype=float)
    raw_values = old * nohost / host
    clipped_mask = (raw_values < float(completeness_floor)) | (raw_values > 1.0)
    values = np.clip(raw_values, float(completeness_floor), 1.0)
    model = Completeness2D(
        artifact.magnitude_grid,
        artifact.redshift_grid,
        values,
        magnitude_support=(artifact.magnitude_grid[0], artifact.magnitude_grid[-1]),
    )
    model.artifact_content_hash = artifact.content_hash
    model.old_completeness_hash = actual_old_hash
    model.completeness_mode = mode
    model.clipped_cell_mask = clipped_mask
    return model


def _read_text_array(dataset) -> np.ndarray:
    return np.asarray(dataset.asstr()).astype(str)


def build_artifact_from_prepared_catalog(
    input_path: str | Path,
    output_path: str | Path,
    *,
    l2: float = 1.0,
    plot_dir: str | Path | None = None,
) -> str:
    """Train one strict eBOSS head and evaluate paired host/no-host mocks."""

    with h5py.File(input_path, "r") as handle:
        if str(handle.attrs.get("schema", "")) != PREPARED_CATALOG_SCHEMA:
            raise ValueError(
                "Unsupported prepared color-catalog schema; regenerate it with "
                "the empirical paired-error preparation command."
            )
        for attr in (
            "cut_manifest_json", "opportunity_rules_json", "old_completeness_hash",
            "target_provenance_json", "input_catalog_hashes_json",
            "hubble_cut_configuration_json", "host_capture_calibration_json",
            "feature_transform_json", "closure_diagnostics_json",
        ):
            if attr not in handle.attrs:
                raise ValueError(f"Prepared catalog is missing required attribute {attr!r}.")
        cut_manifest = validate_cut_manifest(json.loads(str(handle.attrs["cut_manifest_json"])))
        opportunity_rules = json.loads(str(handle.attrs["opportunity_rules_json"]))
        if opportunity_rules.get("program") != "eboss":
            raise ValueError("Prepared opportunity rules must select only eBOSS.")
        magnitude_grid = np.asarray(handle["magnitude_grid"], float)
        redshift_grid = np.asarray(handle["redshift_grid"], float)
        if "training/eboss" not in handle or "mock/eboss" not in handle:
            raise ValueError("Prepared catalog is missing the eBOSS groups.")
        training = handle["training/eboss"]
        mock = handle["mock/eboss"]
        required_training = {
            "features", "success", "marks", "patterns", "alternative_channel"
        }
        missing = required_training - set(training)
        if missing:
            raise ValueError(f"eBOSS training group is missing {sorted(missing)}.")
        marks = _read_text_array(training["marks"])
        success = np.asarray(training["success"], float)
        if np.any(~np.isin(marks, ("alt_only", "both"))):
            raise ValueError("eBOSS training contains non-alternative trials.")
        if not np.array_equal(success.astype(bool), marks == "both"):
            raise ValueError("eBOSS success must encode both vs alt_only.")
        feature_names = tuple(json.loads(str(training.attrs["feature_names_json"])))
        if feature_names != EBOSS_FEATURE_NAMES:
            raise ValueError("Prepared eBOSS feature schema is not the frozen v6 schema.")
        features = np.asarray(training["features"], float)
        patterns = _read_text_array(training["patterns"])
        channel = _read_text_array(training["alternative_channel"])
        closure = leave_one_channel_out_closure(
            "eboss", feature_names, features, success, channel, l2=l2, strict=True
        )
        head = LogisticColorHead.fit(
            "eboss", feature_names, features, success, l2=l2
        )
        support = EmpiricalFeatureSupport.fit(features, patterns)

        required_mock = {
            "features_host", "features_nohost", "patterns_host", "patterns_nohost",
            "m_bin", "z_bin", "weights", "m_hd_host", "m_hd_nohost",
            "luminosity_host", "luminosity_nohost", "observing_state_id",
            "noise_normal", "host_capture_fraction",
        }
        missing = required_mock - set(mock)
        if missing:
            raise ValueError(f"eBOSS mock group is missing {sorted(missing)}.")
        assert_paired_nuclear_state(
            mock["m_hd_host"], mock["m_hd_nohost"],
            mock["luminosity_host"], mock["luminosity_nohost"],
        )
        fh = np.asarray(mock["features_host"], float)
        fn = np.asarray(mock["features_nohost"], float)
        if fh.shape != fn.shape or fh.shape[1] != len(feature_names):
            raise ValueError("Mock feature schema differs between eBOSS branches.")
        part = {
            "m_bin": np.asarray(mock["m_bin"], int),
            "z_bin": np.asarray(mock["z_bin"], int),
            "weights": np.asarray(mock["weights"], float),
            "probability_host": head.predict(fh),
            "probability_nohost": head.predict(fn),
            "supported_host": support.contains(
                fh, _read_text_array(mock["patterns_host"])
            ),
            "supported_nohost": support.contains(
                fn, _read_text_array(mock["patterns_nohost"])
            ),
        }

        metadata = {
            "schema": ARTIFACT_SCHEMA,
            "input_catalog_sha256": _file_sha256(input_path),
            "old_completeness_hash": str(handle.attrs["old_completeness_hash"]),
            "hubble_cut_configuration": json.loads(
                str(handle.attrs["hubble_cut_configuration_json"])
            ),
            "cut_manifest": cut_manifest,
            "opportunity_rules": opportunity_rules,
            "target_provenance": json.loads(str(handle.attrs["target_provenance_json"])),
            "input_catalog_hashes": json.loads(str(handle.attrs["input_catalog_hashes_json"])),
            "lf_mock_path": str(handle.attrs.get("lf_mock_path", "")),
            "qsogen_configuration_json": str(handle.attrs.get("qsogen_configuration_json", "{}")),
            "noise_model_json": str(handle.attrs.get("noise_model_json", "{}")),
            "host_capture_calibration": json.loads(
                str(handle.attrs["host_capture_calibration_json"])
            ),
            "feature_transform": json.loads(
                str(handle.attrs["feature_transform_json"])
            ),
            "photometry_representation": (
                "eBOSS effective projected XDQSOz+WISE signed "
                "luptitudes/colors/depth/missingness"
            ),
            "heads_frozen": {"eboss": head.to_dict()},
            "support_frozen": {"eboss": support.to_dict()},
            "closure_diagnostics": {"eboss": closure},
            "closure_policy": (
                "strict when at least two clean alternative channels are testable; "
                "otherwise conditional on the available alternative channel"
            ),
            "host_counterfactual": (
                "eBOSS photometry-only color-head; fixed nuclear m_hd; morphology absent"
            ),
        }
    products = marginalize_paired_color_completeness(
        n_m=len(magnitude_grid), n_z=len(redshift_grid), **part
    )
    digest = write_color_completeness_artifact(
        output_path, magnitude_grid=magnitude_grid, redshift_grid=redshift_grid,
        products=products, metadata=metadata,
    )
    if plot_dir is not None:
        write_color_completeness_plots(
            magnitude_grid, redshift_grid, products, plot_dir
        )
    return digest


def write_color_completeness_plots(
    magnitude_grid: Any,
    redshift_grid: Any,
    products: Mapping[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    """Write required frozen-map and support closure diagnostics."""

    from matplotlib import pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    panels = (
        ("C_color_host", r"$C_{\rm color,host}$"),
        ("C_color_nohost", r"$C_{\rm color,nohost}$"),
        ("delta_C_host_color", r"$\Delta C_{\rm host}^{\rm color}$"),
        ("host_nohost_ratio", r"$C_{\rm color,nohost}/C_{\rm color,host}$"),
    )
    plot_products = dict(products)
    plot_products["host_nohost_ratio"] = (
        np.asarray(products["C_color_nohost"])
        / np.asarray(products["C_color_host"])
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    extent = [float(np.min(redshift_grid)), float(np.max(redshift_grid)),
              float(np.min(magnitude_grid)), float(np.max(magnitude_grid))]
    valid = (
        np.asarray(products["support_host"], bool)
        & np.asarray(products["support_nohost"], bool)
        & (np.asarray(products["n_eff_host"]) >= MIN_EFFECTIVE_SAMPLE_SIZE)
        & (np.asarray(products["n_eff_nohost"]) >= MIN_EFFECTIVE_SAMPLE_SIZE)
    )
    for ax, (name, label) in zip(axes.flat, panels):
        values = np.ma.masked_where(~valid, np.asarray(plot_products[name]))
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("lightgray")
        image = ax.imshow(values, origin="lower", aspect="auto", extent=extent, cmap=cmap)
        ax.set(
            xlabel="redshift", ylabel=r"nuclear $m_{2500}$",
            title=f"{label} (gray = unsupported)",
        )
        fig.colorbar(image, ax=ax)
    map_path = output / "program_color_completeness_maps.png"
    fig.savefig(map_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    support_panels = (
        ("out_of_support_fraction_host", "host out-of-support weight"),
        ("out_of_support_fraction_nohost", "no-host out-of-support weight"),
        ("valid", "valid interpolation cell"),
    )
    support_values = dict(products)
    support_values["valid"] = valid.astype(float)
    for ax, (name, label) in zip(axes, support_panels):
        image = ax.imshow(
            np.asarray(support_values[name]), origin="lower", aspect="auto",
            extent=extent, vmin=0.0, vmax=1.0,
        )
        ax.set(xlabel="redshift", ylabel=r"nuclear $m_{2500}$", title=label)
        fig.colorbar(image, ax=ax)
    support_path = output / "program_color_feature_support.png"
    fig.savefig(support_path, dpi=180)
    plt.close(fig)

    slice_panels = (
        ("C_color_host", r"$C_{\rm color,host}$"),
        ("C_color_nohost", r"$C_{\rm color,nohost}$"),
        ("host_nohost_ratio", r"$C_{\rm color,nohost}/C_{\rm color,host}$"),
        ("delta_C_host_color", r"$\Delta C_{\rm host}^{\rm color}$"),
    )
    indices = np.linspace(
        0, len(magnitude_grid) - 1, min(5, len(magnitude_grid)), dtype=int
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (name, ylabel) in zip(axes.flat, slice_panels):
        values = np.asarray(plot_products[name])
        for index in indices:
            ax.plot(
                redshift_grid,
                np.where(valid[index], values[index], np.nan),
                label=f"m2500={magnitude_grid[index]:.2f}",
            )
        if name == "delta_C_host_color":
            ax.axhline(0.0, color="black", lw=0.8)
        elif name == "host_nohost_ratio":
            ax.axhline(1.0, color="black", lw=0.8)
        ax.set(xlabel="redshift", ylabel=ylabel)
    axes.flat[-1].legend(fontsize=8)
    slice_path = output / "program_color_redshift_slices.png"
    fig.savefig(slice_path, dpi=180)
    plt.close(fig)
    return [map_path, support_path, slice_path]


def plot_matched_hubble_residual_change(
    old_csv: str | Path,
    host_removal_csv: str | Path,
    output_path: str | Path,
) -> Path:
    """Plot host-removal minus old residuals for one exactly matched HD sample."""

    import pandas as pd
    from matplotlib import pyplot as plt

    required = {"object_id", "z", "residuals"}
    old = pd.read_csv(old_csv, dtype={"object_id": str})
    host_removal = pd.read_csv(host_removal_csv, dtype={"object_id": str})
    for label, frame in (("old", old), ("host-removal", host_removal)):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{label} residual table is missing {sorted(missing)}.")
        if frame["object_id"].duplicated().any():
            raise ValueError(f"{label} residual table has duplicate object IDs.")
    if set(old["object_id"]) != set(host_removal["object_id"]):
        raise ValueError("Matched Hubble modes do not contain identical object IDs.")
    joined = old[list(required)].merge(
        host_removal[list(required)],
        on="object_id",
        how="inner",
        suffixes=("_old", "_host_removal"),
        validate="one_to_one",
    )
    if not np.allclose(joined["z_old"], joined["z_host_removal"], rtol=0.0, atol=1e-12):
        raise ValueError("Matched Hubble modes have inconsistent object redshifts.")
    z = joined["z_old"].to_numpy(dtype=float)
    delta = (
        joined["residuals_host_removal"].to_numpy(dtype=float)
        - joined["residuals_old"].to_numpy(dtype=float)
    )
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(delta)):
        raise ValueError("Matched Hubble residual change contains non-finite values.")
    edges = np.linspace(float(np.min(z)), float(np.max(z)), 11)
    bin_index = np.clip(np.digitize(z, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for index in range(len(edges) - 1):
        use = bin_index == index
        if not np.any(use):
            continue
        values = delta[use]
        rows.append({
            "z_low": edges[index],
            "z_high": edges[index + 1],
            "z_mean": float(np.mean(z[use])),
            "n": int(np.count_nonzero(use)),
            "delta_residual_mean": float(np.mean(values)),
            "delta_residual_sem": float(
                np.std(values, ddof=1) / np.sqrt(values.size)
                if values.size > 1 else 0.0
            ),
        })
    summary = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output.with_suffix(".csv"), index=False)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(z, delta, s=8, alpha=0.18, color="tab:blue", rasterized=True)
    ax.errorbar(
        summary["z_mean"],
        summary["delta_residual_mean"],
        yerr=summary["delta_residual_sem"],
        fmt="o-",
        color="black",
        capsize=2,
        label="bin mean ± SEM",
    )
    ax.axhline(0.0, color="tab:red", lw=1.0)
    ax.set(
        xlabel="redshift",
        ylabel="Δ residual (host-removal − old)",
        title=f"Matched eBOSS photometry-only sensitivity (N={len(joined)})",
    )
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or validate eBOSS photometry-only color artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("artifact", type=Path)
    validate.add_argument("--require-program", choices=COLOR_PROGRAMS)
    build = subparsers.add_parser("build")
    build.add_argument("prepared_catalog", type=Path)
    build.add_argument("artifact", type=Path)
    build.add_argument("--l2", type=float, default=1.0)
    build.add_argument("--plot-dir", type=Path)
    compare = subparsers.add_parser("plot-residual-change")
    compare.add_argument("old_csv", type=Path)
    compare.add_argument("host_removal_csv", type=Path)
    compare.add_argument("output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "build":
        digest = build_artifact_from_prepared_catalog(
            args.prepared_catalog, args.artifact, l2=args.l2, plot_dir=args.plot_dir
        )
        print(_canonical_json({"artifact": str(args.artifact), "content_hash_sha256": digest}))
        return 0
    if args.command == "plot-residual-change":
        output = plot_matched_hubble_residual_change(
            args.old_csv, args.host_removal_csv, args.output
        )
        print(_canonical_json({"residual_change_plot": str(output)}))
        return 0
    artifact = read_color_completeness_artifact(args.artifact)
    if args.require_program is not None:
        actual = str(
            artifact.metadata.get("opportunity_rules", {}).get("program", "")
        )
        if actual != args.require_program:
            raise ValueError(
                f"Artifact program is {actual!r}, expected {args.require_program!r}."
            )
    print(_canonical_json({"artifact": str(args.artifact), "content_hash_sha256": artifact.content_hash, "shape": list(artifact.arrays["C_color_host"].shape), "valid_cells": int(np.count_nonzero(artifact.valid_mask))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
