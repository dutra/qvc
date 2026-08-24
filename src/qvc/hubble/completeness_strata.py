"""Named SDSS targeting strata for modular Hubble completeness models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qvc.hubble.cuts import build_sdss_target_selection_mask
from qvc.hubble.hubble_completeness_refactored import (
    COMPLETENESS_FHOST_COL,
    COMPLETENESS_MAG_COL,
    COMPLETENESS_MAG_ERR_COL,
    get_completeness_function_2d,
    get_completeness_function_3d_fhost,
    get_completeness_function_4d_fhost_alpha,
    make_dm_function,
)


COMPLETENESS_STRATUM_COL = "completeness_stratum"
COMPLETENESS_STRATUM_CODE_COL = "completeness_stratum_code"
COMPLETENESS_STRATIFICATION_CHOICES = (
    "none",
    "sdss-survey",
    "sdss-survey-var",
    "sdss-survey-var-no-boss",
    "sdss-clean-var-core",
)


@dataclass(frozen=True)
class CompletenessStratumDefinition:
    """One mutually exclusive stratum backed by an SDSS selection preset."""

    name: str
    sdss_selection: str


@dataclass(frozen=True)
class CompletenessStratificationPreset:
    """An ordered, reproducible completeness-stratification definition."""

    name: str
    strata: tuple[CompletenessStratumDefinition, ...]
    exclude_unassigned: bool = True

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "exclude_unassigned": bool(self.exclude_unassigned),
            "strata": [
                {"name": item.name, "sdss_selection": item.sdss_selection}
                for item in self.strata
            ],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        )


COMPLETENESS_STRATIFICATION_PRESETS = {
    "sdss-survey": CompletenessStratificationPreset(
        name="sdss-survey",
        strata=(
            CompletenessStratumDefinition("legacy-sdss", "legacy-sdss"),
            CompletenessStratumDefinition("boss", "boss"),
            CompletenessStratumDefinition("eboss", "eboss"),
        ),
    ),
    "sdss-survey-var": CompletenessStratificationPreset(
        name="sdss-survey-var",
        strata=(
            CompletenessStratumDefinition("legacy-sdss", "legacy-sdss"),
            CompletenessStratumDefinition("boss", "boss"),
            CompletenessStratumDefinition(
                "eboss-var-s82-inclusive", "eboss-var-s82-inclusive"
            ),
            CompletenessStratumDefinition(
                "eboss-non-var-s82", "eboss-non-var-s82"
            ),
        ),
    ),
    "sdss-survey-var-no-boss": CompletenessStratificationPreset(
        name="sdss-survey-var-no-boss",
        strata=(
            CompletenessStratumDefinition("legacy-sdss", "legacy-sdss"),
            CompletenessStratumDefinition(
                "eboss-var-s82-inclusive", "eboss-var-s82-inclusive"
            ),
            CompletenessStratumDefinition(
                "eboss-non-var-s82", "eboss-non-var-s82"
            ),
        ),
    ),
    "sdss-clean-var-core": CompletenessStratificationPreset(
        name="sdss-clean-var-core",
        strata=(
            CompletenessStratumDefinition("legacy-sdss", "legacy-sdss"),
            CompletenessStratumDefinition("boss", "boss"),
            CompletenessStratumDefinition(
                "eboss-var-not-core", "eboss-var-s82-only"
            ),
            CompletenessStratumDefinition(
                "eboss-var-and-core", "eboss-var-s82-core-only"
            ),
        ),
    ),
}


def normalize_completeness_stratification(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized not in COMPLETENESS_STRATIFICATION_CHOICES:
        raise ValueError(
            f"Unknown completeness stratification {value!r}; choose one of "
            f"{COMPLETENESS_STRATIFICATION_CHOICES}."
        )
    return normalized


def get_completeness_stratification_preset(
    value: str,
) -> CompletenessStratificationPreset | None:
    normalized = normalize_completeness_stratification(value)
    if normalized == "none":
        return None
    return COMPLETENESS_STRATIFICATION_PRESETS[normalized]


@dataclass(frozen=True)
class CompletenessStratumAssignment:
    preset_name: str
    definition_json: str
    retained_mask: np.ndarray
    labels: np.ndarray
    codes: np.ndarray
    criteria: tuple[str, ...]
    counts: tuple[int, ...]

    @property
    def stratum_names(self) -> tuple[str, ...]:
        preset = COMPLETENESS_STRATIFICATION_PRESETS[self.preset_name]
        return tuple(item.name for item in preset.strata)


def assign_completeness_strata(
    df: pd.DataFrame,
    stratification: str,
    *,
    require_nonempty: bool = True,
) -> CompletenessStratumAssignment | None:
    """Assign one stable stratum code to each retained row."""

    preset = get_completeness_stratification_preset(stratification)
    if preset is None:
        return None

    labels = np.full(len(df), "", dtype=object)
    codes = np.full(len(df), -1, dtype=np.int16)
    criteria = []
    counts = []
    for code, definition in enumerate(preset.strata):
        mask, criterion = build_sdss_target_selection_mask(
            df, definition.sdss_selection
        )
        mask = np.asarray(mask, dtype=bool)
        overlap = mask & (codes >= 0)
        if np.any(overlap):
            examples = df.loc[overlap, "object_id"].astype(str).head(5).tolist()
            raise ValueError(
                f"Completeness preset {preset.name!r} has overlapping stratum "
                f"{definition.name!r}; example object_id(s): {examples}."
            )
        labels[mask] = definition.name
        codes[mask] = code
        criteria.append(criterion)
        counts.append(int(np.count_nonzero(mask)))

    if require_nonempty:
        empty = [
            definition.name
            for definition, count in zip(preset.strata, counts)
            if count == 0
        ]
        if empty:
            raise ValueError(
                f"Completeness preset {preset.name!r} produced empty stratum(s) "
                f"{empty}; choose another preset or provide complete SDSS metadata."
            )

    retained = codes >= 0
    if not preset.exclude_unassigned and np.any(~retained):
        raise ValueError(
            f"Completeness preset {preset.name!r} left "
            f"{np.count_nonzero(~retained)} row(s) unassigned."
        )
    return CompletenessStratumAssignment(
        preset_name=preset.name,
        definition_json=preset.canonical_json(),
        retained_mask=retained,
        labels=labels,
        codes=codes,
        criteria=tuple(criteria),
        counts=tuple(counts),
    )


def annotate_completeness_strata(
    df: pd.DataFrame,
    assignment: CompletenessStratumAssignment,
) -> pd.DataFrame:
    """Return retained rows annotated with stable labels and integer codes."""

    if len(df) != len(assignment.retained_mask):
        raise ValueError("Completeness stratum assignment length does not match dataframe.")
    retained = assignment.retained_mask
    out = df.loc[retained].copy()
    out[COMPLETENESS_STRATUM_COL] = assignment.labels[retained].astype(str)
    out[COMPLETENESS_STRATUM_CODE_COL] = assignment.codes[retained].astype(np.int16)
    out.attrs.update(df.attrs)
    out.attrs["completeness_stratification"] = assignment.preset_name
    out.attrs["completeness_stratification_definition"] = assignment.definition_json
    return out


@dataclass(frozen=True)
class StratifiedCompletenessBundle:
    """Ordered collection of one existing completeness tuple per stratum."""

    preset_name: str
    definition_json: str
    stratum_names: tuple[str, ...]
    params_by_stratum: tuple[tuple[Any, ...], ...]

    def params_for_code(self, code: int) -> tuple[Any, ...]:
        try:
            return self.params_by_stratum[int(code)]
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid completeness stratum code {code!r}.") from exc


def is_stratified_completeness(value: Any) -> bool:
    return isinstance(value, StratifiedCompletenessBundle)


def _validate_map_input_columns(df: pd.DataFrame, completeness_mode: str) -> None:
    missing = {COMPLETENESS_MAG_COL, COMPLETENESS_MAG_ERR_COL} - set(df.columns)
    if missing:
        raise KeyError(
            "Completeness requires prepared 2500-A magnitude columns: "
            f"{sorted(missing)}."
        )
    if completeness_mode in {"3d_fhost", "4d_fhost_alpha"}:
        if COMPLETENESS_FHOST_COL not in df.columns:
            raise KeyError(
                f"completeness_mode={completeness_mode!r} requires "
                f"{COMPLETENESS_FHOST_COL!r}."
            )
        if not np.all(np.isfinite(df[COMPLETENESS_FHOST_COL].to_numpy(float))):
            raise ValueError(
                f"completeness_mode={completeness_mode!r} requires finite "
                f"{COMPLETENESS_FHOST_COL}."
            )
    if completeness_mode == "4d_fhost_alpha":
        if "alpha_lambda" not in df.columns:
            raise KeyError("4D completeness requires 'alpha_lambda'.")
        if not np.all(np.isfinite(df["alpha_lambda"].to_numpy(float))):
            raise ValueError("4D completeness requires finite alpha_lambda.")


def build_single_completeness_params(
    df_observed: pd.DataFrame,
    df_parent: pd.DataFrame,
    *,
    completeness_mode: str,
    completeness_sim_file: str,
    plot: bool,
    plot_path: str | Path,
) -> tuple[Any, ...]:
    """Build one legacy completeness tuple without stratum-specific logic."""

    _validate_map_input_columns(df_observed, completeness_mode)
    if completeness_mode == "4d_fhost_alpha":
        return get_completeness_function_4d_fhost_alpha(
            df_observed,
            sim_file=completeness_sim_file,
            plot=plot,
            plot_path=str(plot_path),
            df_agn_fhost_population=df_parent,
        )
    if completeness_mode == "3d_fhost":
        return get_completeness_function_3d_fhost(
            df_observed,
            sim_file=completeness_sim_file,
            plot=plot,
            plot_path=str(plot_path),
            df_agn_fhost_population=df_parent,
        )
    return get_completeness_function_2d(
        df_observed,
        sim_file=completeness_sim_file,
        plot=plot,
        plot_path=str(plot_path),
    )


def _validate_bundle_grids(params_by_stratum: list[tuple[Any, ...]]) -> None:
    reference = params_by_stratum[0]
    reference_mode = getattr(reference[0], "mode", "2d")
    reference_grids = [np.asarray(value) for value in reference[1:]]
    for params in params_by_stratum[1:]:
        if getattr(params[0], "mode", "2d") != reference_mode:
            raise ValueError("Completeness strata produced different map modes.")
        # Grid arrays precede scalar widths and model metadata. Compare all
        # consecutive one-dimensional numeric arrays shared by both tuples.
        for ref_value, value in zip(reference_grids, params[1:]):
            ref_arr = np.asarray(ref_value)
            arr = np.asarray(value)
            if ref_arr.ndim != 1 or arr.ndim != 1:
                break
            if ref_arr.shape != arr.shape or not np.allclose(ref_arr, arr):
                raise ValueError("Completeness strata produced incompatible grids.")


def build_completeness_params(
    df_observed: pd.DataFrame,
    df_parent: pd.DataFrame,
    *,
    completeness_mode: str,
    completeness_sim_file: str,
    plot: bool,
    plot_path: str | Path,
    stratification: str = "none",
) -> tuple[Any, ...] | StratifiedCompletenessBundle:
    """Build a legacy map or an ordered bundle of per-stratum maps."""

    preset = get_completeness_stratification_preset(stratification)
    if preset is None:
        return build_single_completeness_params(
            df_observed,
            df_parent,
            completeness_mode=completeness_mode,
            completeness_sim_file=completeness_sim_file,
            plot=plot,
            plot_path=plot_path,
        )

    for frame_name, frame in (("observed", df_observed), ("parent", df_parent)):
        missing = {
            COMPLETENESS_STRATUM_COL,
            COMPLETENESS_STRATUM_CODE_COL,
        } - set(frame.columns)
        if missing:
            raise KeyError(
                f"Stratified completeness {frame_name} dataframe is missing {sorted(missing)}."
            )

    params_by_stratum = []
    for code, definition in enumerate(preset.strata):
        observed = df_observed[
            df_observed[COMPLETENESS_STRATUM_CODE_COL].to_numpy() == code
        ]
        parent = df_parent[
            df_parent[COMPLETENESS_STRATUM_CODE_COL].to_numpy() == code
        ]
        if observed.empty or parent.empty:
            raise ValueError(
                f"Cannot build completeness stratum {definition.name!r}: "
                f"observed={len(observed)}, parent={len(parent)}."
            )
        stratum_plot_path = Path(plot_path) / "strata" / definition.name
        params_by_stratum.append(
            build_single_completeness_params(
                observed,
                parent,
                completeness_mode=completeness_mode,
                completeness_sim_file=completeness_sim_file,
                plot=plot,
                plot_path=stratum_plot_path,
            )
        )
    _validate_bundle_grids(params_by_stratum)
    return StratifiedCompletenessBundle(
        preset_name=preset.name,
        definition_json=preset.canonical_json(),
        stratum_names=tuple(item.name for item in preset.strata),
        params_by_stratum=tuple(params_by_stratum),
    )


class StratifiedDebiasInterpolator:
    """Dispatch existing dmi interpolators by a stable string stratum label."""

    def __init__(self, interpolators: dict[str, Any]):
        self.interpolators = dict(interpolators)

    def evaluate_stratified(self, points: np.ndarray, strata: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        strata = np.asarray(strata).astype(str)
        if points.ndim != 2 or strata.shape != (points.shape[0],):
            raise ValueError("Stratified interpolation points/labels have incompatible shapes.")
        out = np.full(points.shape[0], np.nan, dtype=float)
        unknown = sorted(set(strata) - set(self.interpolators))
        if unknown:
            raise ValueError(f"Unknown completeness stratum label(s): {unknown}.")
        for name, interpolator in self.interpolators.items():
            mask = strata == name
            if np.any(mask):
                out[mask] = np.asarray(interpolator(points[mask]), dtype=float)
        return out


def make_stratified_dm_function(
    df: pd.DataFrame,
    values: np.ndarray,
) -> Any:
    """Return the legacy interpolator or one interpolator per dataframe stratum."""

    values = np.asarray(values, dtype=float)
    if values.shape != (len(df),):
        raise ValueError(f"Debias values have shape {values.shape}, expected {(len(df),)}.")
    common_kwargs = {
        "f_host_2500_psf": (
            df[COMPLETENESS_FHOST_COL].to_numpy()
            if COMPLETENESS_FHOST_COL in df.columns
            else None
        ),
        "alpha_lambda": (
            df["alpha_lambda"].to_numpy() if "alpha_lambda" in df.columns else None
        ),
    }
    if COMPLETENESS_STRATUM_COL not in df.columns:
        return make_dm_function(
            df[COMPLETENESS_MAG_COL].to_numpy(),
            df["z"].to_numpy(),
            values,
            **common_kwargs,
        )

    interpolators = {}
    labels = df[COMPLETENESS_STRATUM_COL].astype(str).to_numpy()
    for name in dict.fromkeys(labels.tolist()):
        mask = labels == name
        interpolators[name] = make_dm_function(
            df.loc[mask, COMPLETENESS_MAG_COL].to_numpy(),
            df.loc[mask, "z"].to_numpy(),
            values[mask],
            f_host_2500_psf=(
                df.loc[mask, COMPLETENESS_FHOST_COL].to_numpy()
                if COMPLETENESS_FHOST_COL in df.columns
                else None
            ),
            alpha_lambda=(
                df.loc[mask, "alpha_lambda"].to_numpy()
                if "alpha_lambda" in df.columns
                else None
            ),
        )
    return StratifiedDebiasInterpolator(interpolators)


def write_completeness_stratum_counts(
    *,
    preset_name: str,
    before_cuts: pd.DataFrame,
    after_quality_cuts: pd.DataFrame,
    fitted: pd.DataFrame,
    output_path: str | Path,
    cut_summary_path: str | Path | None = None,
) -> pd.DataFrame | None:
    """Write deterministic population counts for an active preset."""

    preset = get_completeness_stratification_preset(preset_name)
    if preset is None:
        return None
    rows = []
    for code, definition in enumerate(preset.strata):
        row = {
            "completeness_stratification": preset.name,
            "completeness_stratum_code": code,
            "completeness_stratum": definition.name,
            "sdss_selection": definition.sdss_selection,
        }
        for label, frame in (
            ("before_cuts", before_cuts),
            ("after_quality_cuts", after_quality_cuts),
            ("fitted", fitted),
        ):
            if COMPLETENESS_STRATUM_CODE_COL not in frame.columns:
                raise KeyError(
                    f"Cannot count {label}: missing {COMPLETENESS_STRATUM_CODE_COL!r}."
                )
            row[label] = int(
                np.count_nonzero(
                    frame[COMPLETENESS_STRATUM_CODE_COL].to_numpy(dtype=int)
                    == code
                )
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    if cut_summary_path is not None:
        cut_summary_path = Path(cut_summary_path)
        start = "\n# completeness-stratification-counts:start\n"
        end = "# completeness-stratification-counts:end\n"
        block = (
            start
            + summary.to_string(index=False)
            + "\n"
            + end
        )
        existing = (
            cut_summary_path.read_text(encoding="utf-8")
            if cut_summary_path.exists()
            else ""
        )
        if start in existing and end in existing:
            before, remainder = existing.split(start, 1)
            _, after = remainder.split(end, 1)
            existing = before.rstrip("\n") + "\n" + after.lstrip("\n")
        cut_summary_path.parent.mkdir(parents=True, exist_ok=True)
        cut_summary_path.write_text(existing.rstrip("\n") + block, encoding="utf-8")
    return summary
