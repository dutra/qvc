"""Paired diagnostics for the luminosity-function Hubble sweep."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qvc.hubble.completeness_mock_catalog import COMPLETENESS_LF_MODELS
from qvc.hubble.hubble_model import get_model_params


REPO_ROOT = Path(__file__).resolve().parents[3]
STYLE_PATH = Path(__file__).with_name("style.mplstyle")
DIAGNOSTIC_FIGURE_STEM = "lf_selection_correction_and_hubble_residuals"
PARAMETER_FIGURE_STEM = "lf_parameter_comparison"
SUMMARY_FILENAME = "lf_residual_sensitivity_summary.csv"
BINNED_FILENAME = "lf_residual_sensitivity_binned.csv"
README_FILENAME = "lf_residual_sensitivity_README.txt"
Z_EDGES = np.array([0.44, 0.80, 1.20, 1.60, 2.00, 2.50, 3.16])
Z_CENTERS = 0.5 * (Z_EDGES[:-1] + Z_EDGES[1:])
DIAGNOSTIC_LABELS = {
    "shen": "Shen et al. (2020)",
    "shen_type1_intrinsic": "Shen et al. (2020), Type 1 intrinsic",
    "shen_type1_attenuated": "Shen et al. (2020), Type 1 attenuated",
    "wang2026_type1_lade_a": "Wang et al. (2026)",
    "palanque2016_ple_lede": "Palanque-Delabrouille et al. (2016)",
    "kulkarni2019_type1_model1": "Kulkarni et al. (2019), M1",
    "kulkarni2019_type1_model2": "Kulkarni et al. (2019), M2",
    "kulkarni2019_type1_model3": "Kulkarni et al. (2019), M3",
}
COLORS = {
    "shen": "#000000",
    "shen_type1_intrinsic": "#999999",
    "shen_type1_attenuated": "#56B4E9",
    "wang2026_type1_lade_a": "#0072B2",
    "palanque2016_ple_lede": "#E69F00",
    "kulkarni2019_type1_model1": "#009E73",
    "kulkarni2019_type1_model2": "#CC79A7",
    "kulkarni2019_type1_model3": "#D55E00",
}
MARKERS = {
    "shen": "o",
    "shen_type1_intrinsic": "X",
    "shen_type1_attenuated": "*",
    "wang2026_type1_lade_a": "s",
    "palanque2016_ple_lede": "^",
    "kulkarni2019_type1_model1": "D",
    "kulkarni2019_type1_model2": "v",
    "kulkarni2019_type1_model3": "P",
}
PARAMETER_COMPARISON_NAMES = (
    "alpha_agn",
    "beta_agn",
    "M0_agn",
    "H0",
    "Om0",
    "w0",
    "wa",
)
PARAMETER_COMPARISON_LABELS = {
    "alpha_agn": r"$\alpha_{\rm AGN}$",
    "beta_agn": r"$\beta_{\rm AGN}$",
    "M0_agn": r"$M^0_{\rm AGN}$",
    "H0": r"$H_0$",
    "Om0": r"$\Omega_{m,0}$",
    "w0": r"$w_0$",
    "wa": r"$w_a$",
}
PARAMETER_COMPARISON_LF_LABELS = {
    "shen": "Shen+20",
    "shen_type1_intrinsic": "Shen+20, Type 1 intrinsic",
    "shen_type1_attenuated": "Shen+20, Type 1 attenuated",
    "wang2026_type1_lade_a": "Wang+26, LADE-A",
    "palanque2016_ple_lede": "Palanque+16, PLE+LEDE",
    "kulkarni2019_type1_model1": "Kulkarni+19, M1",
    "kulkarni2019_type1_model2": "Kulkarni+19, M2",
    "kulkarni2019_type1_model3": "Kulkarni+19, M3",
}


def _normalized_ids(values) -> np.ndarray:
    normalized = []
    for value in np.asarray(values):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        normalized.append(str(value).strip())
    return np.asarray(normalized)


def _fit_selection_mask(values) -> np.ndarray:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.bool_):
        return values.astype(bool)
    normalized = np.char.lower(np.char.strip(values.astype(str)))
    valid = np.isin(normalized, ("true", "false"))
    if not np.all(valid):
        raise ValueError("is_fit_selection must contain only true/false values.")
    return normalized == "true"


def _sole_match(directory: Path, pattern: str, *, description: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {description} under {directory}, "
            f"found {len(matches)}."
        )
    return matches[0]


def _bootstrap_median_interval(values, rng, *, draws: int):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty redshift bin.")
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))]
    medians = np.median(samples, axis=1)
    return np.median(values), *np.quantile(medians, [0.16, 0.84])


def _binned_medians(
    z,
    values,
    rng,
    *,
    bootstrap_draws: int,
    interval_kind: str = "bootstrap_median",
):
    if interval_kind not in {
        "bootstrap_median",
        "object_distribution",
        "paired_distribution",
    }:
        raise ValueError(f"Unknown interval kind: {interval_kind!r}.")
    rows = []
    for index, (lower, upper) in enumerate(zip(Z_EDGES[:-1], Z_EDGES[1:])):
        include_upper = index == len(Z_CENTERS) - 1
        mask = (z >= lower) & ((z <= upper) if include_upper else (z < upper))
        bin_values = np.asarray(values[mask], dtype=float)
        if interval_kind == "bootstrap_median":
            median, lo, hi = _bootstrap_median_interval(
                bin_values, rng, draws=bootstrap_draws
            )
        else:
            finite = bin_values[np.isfinite(bin_values)]
            if finite.size == 0:
                raise ValueError("Cannot summarize an empty redshift bin.")
            median = np.median(finite)
            lo, hi = np.quantile(finite, [0.16, 0.84])
        rows.append((Z_CENTERS[index], int(mask.sum()), median, lo, hi))
    return np.asarray(rows, dtype=float)


def _load_sweep_data(
    base_prefix: str,
    diagrams: Sequence[tuple[str, Path]],
    *,
    repo_root: Path,
):
    diagram_paths = dict(diagrams)
    expected_models = tuple(model for model, _ in diagrams)
    if len(diagram_paths) != len(expected_models):
        raise ValueError("Diagnostic run identifiers must be unique.")
    if not expected_models or expected_models[0] != "shen":
        raise ValueError("Diagnostic diagrams must start with the Shen reference run.")
    missing_labels = [
        model for model in expected_models if model not in DIAGNOSTIC_LABELS
    ]
    if missing_labels:
        raise ValueError(
            "Diagnostic labels are missing for runs: " + ", ".join(missing_labels)
        )

    residuals = {}
    corrections = {}
    m0_agn = {}
    _, parameter_names, _ = get_model_params("Flatw0waCDM")
    m0_index = parameter_names.index("M0_agn")

    for model in expected_models:
        residual_path = diagram_paths[model].with_name("hubble_plot_residuals.csv")
        if not residual_path.is_file():
            raise FileNotFoundError(f"Missing residual table: {residual_path}")
        frame = pd.read_csv(residual_path)
        required = {"object_id", "is_fit_selection", "z", "residuals"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                f"Residual table {residual_path} is missing columns: {missing}."
            )
        frame["object_id"] = _normalized_ids(frame["object_id"])
        frame = frame.loc[_fit_selection_mask(frame["is_fit_selection"])].set_index(
            "object_id"
        )
        if not frame.index.is_unique:
            raise ValueError(f"Residual table {residual_path} has duplicate object IDs.")
        residuals[model] = frame

        posterior_directory = (
            repo_root
            / "results"
            / "hubble_posteriors"
            / f"{base_prefix}_{model}"
        )
        posterior_path = _sole_match(
            posterior_directory, "*.h5", description="posterior H5 file"
        )
        with h5py.File(posterior_path, "r") as handle:
            for dataset in (
                "object_id_fit_selection",
                "dmi_posterior_median",
                "flat_samples",
            ):
                if dataset not in handle:
                    raise ValueError(
                        f"Posterior file {posterior_path} lacks dataset {dataset!r}."
                    )
            ids = _normalized_ids(handle["object_id_fit_selection"][:])
            dmi = np.asarray(handle["dmi_posterior_median"][:], dtype=float)
            if ids.shape != dmi.shape:
                raise ValueError(
                    f"Posterior object IDs and dmi differ in shape in {posterior_path}."
                )
            corrections[model] = pd.Series(dmi, index=ids, name="dmi")
            if not corrections[model].index.is_unique:
                raise ValueError(f"Posterior file {posterior_path} has duplicate IDs.")
            samples = np.asarray(handle["flat_samples"][:], dtype=float)
            if samples.ndim != 2 or samples.shape[1] != len(parameter_names):
                raise ValueError(
                    f"Posterior file {posterior_path} has flat_samples shape "
                    f"{samples.shape}; expected (*, {len(parameter_names)})."
                )
            m0_agn[model] = float(np.median(samples[:, m0_index]))

    reference_ids = set(residuals["shen"].index)
    for model in expected_models:
        residual_ids = set(residuals[model].index)
        correction_ids = set(corrections[model].index)
        if residual_ids != reference_ids or correction_ids != reference_ids:
            raise RuntimeError(
                "LF diagnostics require identical fit-selection object IDs; "
                f"model {model!r} has {len(residual_ids)} residual IDs and "
                f"{len(correction_ids)} correction IDs versus "
                f"{len(reference_ids)} for Shen."
            )
    ids = pd.Index(sorted(reference_ids))
    if len(ids) == 0:
        raise RuntimeError("LF diagnostics found no common fit-selection objects.")
    return residuals, corrections, m0_agn, ids


def _save_figure(fig, output_directory: Path, stem: str) -> tuple[Path, Path]:
    pdf_path = output_directory / f"{stem}.pdf"
    png_path = output_directory / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    return pdf_path, png_path


def load_lf_parameter_summaries(
    base_prefix: str,
    *,
    repo_root: Path = REPO_ROOT,
    models: Sequence[str] = COMPLETENESS_LF_MODELS,
) -> pd.DataFrame:
    """Load median and central-68% summaries for each LF posterior."""
    _, model_labels, _ = get_model_params("Flatw0waCDM")
    missing_parameters = [
        name for name in PARAMETER_COMPARISON_NAMES if name not in model_labels
    ]
    if missing_parameters:
        raise RuntimeError(
            "Flatw0waCDM parameterization lacks requested parameters: "
            + ", ".join(missing_parameters)
        )
    parameter_indices = {
        name: model_labels.index(name) for name in PARAMETER_COMPARISON_NAMES
    }

    models = tuple(models)
    if not models or models[0] != "shen" or len(set(models)) != len(models):
        raise ValueError(
            "Parameter-comparison runs must be unique and start with 'shen'."
        )
    missing_labels = [
        model for model in models if model not in PARAMETER_COMPARISON_LF_LABELS
    ]
    if missing_labels:
        raise ValueError(
            "Parameter-comparison labels are missing for runs: "
            + ", ".join(missing_labels)
        )

    rows = []
    for model in models:
        posterior_directory = (
            Path(repo_root)
            / "results"
            / "hubble_posteriors"
            / f"{base_prefix}_{model}"
        )
        posterior_path = _sole_match(
            posterior_directory, "*.h5", description="posterior H5 file"
        )
        with h5py.File(posterior_path, "r") as handle:
            if "flat_samples" not in handle:
                raise ValueError(
                    f"Posterior file {posterior_path} lacks dataset 'flat_samples'."
                )
            samples = np.asarray(handle["flat_samples"][:], dtype=float)
        if samples.ndim != 2 or samples.shape[1] != len(model_labels):
            raise ValueError(
                f"Posterior file {posterior_path} has flat_samples shape "
                f"{samples.shape}; expected (*, {len(model_labels)})."
            )
        if samples.shape[0] == 0:
            raise ValueError(f"Posterior file {posterior_path} has no samples.")

        for parameter, index in parameter_indices.items():
            values = samples[:, index]
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Posterior file {posterior_path} has non-finite samples "
                    f"for {parameter!r}."
                )
            lower, median, upper = np.quantile(values, [0.16, 0.50, 0.84])
            rows.append(
                {
                    "model": model,
                    "parameter": parameter,
                    "median": median,
                    "interval_16": lower,
                    "interval_84": upper,
                }
            )
    return pd.DataFrame(rows)


def generate_lf_parameter_comparison(
    base_prefix: str,
    output_directory: Path,
    *,
    repo_root: Path = REPO_ROOT,
    models: Sequence[str] = COMPLETENESS_LF_MODELS,
) -> Mapping[str, Path]:
    """Plot fitted parameters across LF choices with Shen as the reference."""
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    summaries = load_lf_parameter_summaries(
        base_prefix, repo_root=Path(repo_root), models=models
    )
    models = tuple(models)
    x = np.arange(len(models))

    with plt.style.context(STYLE_PATH):
        figure, axes = plt.subplots(
            len(PARAMETER_COMPARISON_NAMES),
            1,
            figsize=(8.0, 13.5),
            sharex=True,
            constrained_layout=True,
        )
        for axis, parameter in zip(
            axes, PARAMETER_COMPARISON_NAMES, strict=True
        ):
            parameter_rows = summaries.loc[
                summaries["parameter"].eq(parameter)
            ].set_index("model").loc[list(models)]
            medians = parameter_rows["median"].to_numpy(dtype=float)
            lower = parameter_rows["interval_16"].to_numpy(dtype=float)
            upper = parameter_rows["interval_84"].to_numpy(dtype=float)
            shen_median = float(parameter_rows.loc["shen", "median"])
            axis.axhline(
                shen_median,
                color="black",
                linestyle="--",
                linewidth=1.5,
                zorder=1,
            )
            axis.errorbar(
                x,
                medians,
                yerr=np.vstack((medians - lower, upper - medians)),
                fmt="o",
                color="#0072B2",
                markersize=6,
                linewidth=1.5,
                capsize=3,
                zorder=2,
            )
            axis.set_ylabel(PARAMETER_COMPARISON_LABELS[parameter])
            axis.margins(x=0.05, y=0.16)

        axes[-1].set_xticks(
            x,
            [PARAMETER_COMPARISON_LF_LABELS[model] for model in models],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        parameter_pdf, parameter_png = _save_figure(
            figure, output_directory, PARAMETER_FIGURE_STEM
        )
    return {
        "parameter_pdf": parameter_pdf,
        "parameter_png": parameter_png,
    }


def _append_binned_rows(
    rows, *, model, quantity, stats, interval_kind="bootstrap_median"
):
    for row, (lower, upper) in zip(stats, zip(Z_EDGES[:-1], Z_EDGES[1:])):
        rows.append(
            {
                "model": model,
                "quantity": quantity,
                "z_min": lower,
                "z_max": upper,
                "n": int(row[1]),
                "median": row[2],
                "interval_16": row[3],
                "interval_84": row[4],
                "interval_kind": interval_kind,
            }
        )


def _selection_correction_and_residual_figure(
    residuals,
    corrections,
    m0_agn,
    ids,
    output_directory,
    rng,
    binned_rows,
    *,
    bootstrap_draws,
    models,
):
    reference_frame = residuals["shen"].loc[ids]
    z = reference_frame["z"].to_numpy(float)
    reference_dmi = corrections["shen"].loc[ids].to_numpy(float)
    reference_residual = reference_frame["residuals"].to_numpy(float)
    fig, axes = plt.subplots(4, 1, figsize=(10, 16), constrained_layout=True)
    absolute_offsets = np.linspace(-0.035, 0.035, len(models))
    offsets = np.linspace(-0.03, 0.03, len(models) - 1)

    for offset, model in zip(absolute_offsets, models):
        dmi = corrections[model].loc[ids].to_numpy(float)
        stats = _binned_medians(
            z,
            dmi,
            rng,
            bootstrap_draws=bootstrap_draws,
            interval_kind="object_distribution",
        )
        axes[0].errorbar(
            stats[:, 0] + offset,
            stats[:, 2],
            yerr=np.vstack(
                (stats[:, 2] - stats[:, 3], stats[:, 4] - stats[:, 2])
            ),
            color=COLORS[model],
            marker=MARKERS[model],
            ms=4.7,
            lw=2.4,
            label=DIAGNOSTIC_LABELS[model],
        )
        _append_binned_rows(
            binned_rows,
            model=model,
            quantity="dmi",
            stats=stats,
            interval_kind="object_distribution",
        )

    for offset, model in zip(offsets, models[1:]):
        delta_dmi = corrections[model].loc[ids].to_numpy(float) - reference_dmi
        stats = _binned_medians(
            z,
            delta_dmi,
            rng,
            bootstrap_draws=bootstrap_draws,
            interval_kind="paired_distribution",
        )
        axes[1].errorbar(
            stats[:, 0] + offset,
            stats[:, 2],
            yerr=np.vstack((stats[:, 2] - stats[:, 3], stats[:, 4] - stats[:, 2])),
            color=COLORS[model],
            marker=MARKERS[model],
            ms=4.7,
            lw=2.4,
            label=DIAGNOSTIC_LABELS[model],
        )
        _append_binned_rows(
            binned_rows,
            model=model,
            quantity="delta_dmi_vs_shen",
            stats=stats,
            interval_kind="paired_distribution",
        )

        compensated = delta_dmi + (m0_agn[model] - m0_agn["shen"])
        compensated_stats = _binned_medians(
            z,
            compensated,
            rng,
            bootstrap_draws=bootstrap_draws,
            interval_kind="paired_distribution",
        )
        residual_delta = (
            residuals[model].loc[ids, "residuals"].to_numpy(float)
            - reference_residual
        )
        residual_stats = _binned_medians(
            z,
            residual_delta,
            rng,
            bootstrap_draws=bootstrap_draws,
            interval_kind="paired_distribution",
        )
        axes[2].errorbar(
            compensated_stats[:, 0] + offset,
            compensated_stats[:, 2],
            yerr=np.vstack(
                (
                    compensated_stats[:, 2] - compensated_stats[:, 3],
                    compensated_stats[:, 4] - compensated_stats[:, 2],
                )
            ),
            color=COLORS[model],
            marker=MARKERS[model],
            ms=4.7,
            lw=2.4,
        )
        axes[3].errorbar(
            residual_stats[:, 0] + offset,
            residual_stats[:, 2],
            yerr=np.vstack(
                (
                    residual_stats[:, 2] - residual_stats[:, 3],
                    residual_stats[:, 4] - residual_stats[:, 2],
                )
            ),
            color=COLORS[model],
            marker=MARKERS[model],
            ms=4.7,
            lw=2.4,
        )
        _append_binned_rows(
            binned_rows,
            model=model,
            quantity="delta_dmi_plus_delta_M0_vs_shen",
            stats=compensated_stats,
            interval_kind="paired_distribution",
        )
        _append_binned_rows(
            binned_rows,
            model=model,
            quantity="delta_residual_vs_shen",
            stats=residual_stats,
            interval_kind="paired_distribution",
        )
    axes[0].set_ylabel(r"Selection correction, $d m_i$ (mag)")
    axes[0].set_ylim(top=0.65)
    axes[0].legend(ncol=2, frameon=False, loc="upper center")
    axes[1].set_ylabel(
        r"Selection-correction difference, $\Delta d m_i$" "\n"
        r"(w.r.t. Shen et al.) (mag)"
    )
    axes[1].set_ylim(top=0.1)
    axes[2].set_ylabel(
        r"$\Delta d m_i + \Delta M^0_{\rm AGN}$" "\n"
        r"(w.r.t. Shen et al.) (mag)"
    )
    axes[3].set_ylabel(
        r"Final $\Delta$ Hubble residual" "\n"
        r"(w.r.t. Shen et al.) (mag)"
    )
    for axis in axes:
        axis.set_xlim(Z_EDGES[0], Z_EDGES[-1])
        axis.set_xlabel("Redshift")
        axis.text(
            0.02,
            0.04,
            f"N = {len(ids):,}",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
        )
    return _save_figure(fig, output_directory, DIAGNOSTIC_FIGURE_STEM)


def _write_summary(
    residuals, corrections, m0_agn, ids, output_directory, *, models
):
    reference_residual = residuals["shen"].loc[ids, "residuals"].to_numpy(float)
    reference_dmi = corrections["shen"].loc[ids].to_numpy(float)
    rows = []
    for model in models:
        values = residuals[model].loc[ids, "residuals"].to_numpy(float)
        delta_residual = values - reference_residual
        delta_dmi = corrections[model].loc[ids].to_numpy(float) - reference_dmi
        delta_m0 = m0_agn[model] - m0_agn["shen"]
        rows.append(
            {
                "model": model,
                "label": DIAGNOSTIC_LABELS[model],
                "n_paired": len(ids),
                "residual_rms_mag": np.sqrt(np.mean(values**2)),
                "delta_residual_rms_vs_shen_mag": np.sqrt(
                    np.mean(delta_residual**2)
                ),
                "median_abs_delta_residual_vs_shen_mag": np.median(
                    np.abs(delta_residual)
                ),
                "p90_abs_delta_residual_vs_shen_mag": np.quantile(
                    np.abs(delta_residual), 0.90
                ),
                "median_delta_dmi_vs_shen_mag": np.median(delta_dmi),
                "delta_M0_agn_vs_shen_mag": delta_m0,
                "median_delta_dmi_plus_delta_M0_mag": (
                    np.median(delta_dmi) + delta_m0
                ),
                "centered_delta_dmi_rms_mag": np.sqrt(
                    np.mean((delta_dmi - np.median(delta_dmi)) ** 2)
                ),
            }
        )
    path = output_directory / SUMMARY_FILENAME
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def generate_lf_comparison_diagnostics(
    base_prefix: str,
    diagrams: Sequence[tuple[str, Path]],
    output_directory: Path,
    *,
    repo_root: Path = REPO_ROOT,
    bootstrap_draws: int = 3000,
) -> Mapping[str, Path]:
    """Generate paired LF residual and selection-correction diagnostics."""

    if bootstrap_draws < 1:
        raise ValueError("bootstrap_draws must be positive.")
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    residuals, corrections, m0_agn, ids = _load_sweep_data(
        base_prefix, diagrams, repo_root=Path(repo_root)
    )
    models = tuple(model for model, _ in diagrams)
    rng = np.random.default_rng(20260831)
    binned_rows = []
    with plt.style.context(STYLE_PATH):
        diagnostic_pdf, diagnostic_png = _selection_correction_and_residual_figure(
            residuals,
            corrections,
            m0_agn,
            ids,
            output_directory,
            rng,
            binned_rows,
            bootstrap_draws=bootstrap_draws,
            models=models,
        )
    summary_path = _write_summary(
        residuals,
        corrections,
        m0_agn,
        ids,
        output_directory,
        models=models,
    )
    binned_path = output_directory / BINNED_FILENAME
    pd.DataFrame(binned_rows).to_csv(binned_path, index=False)
    readme_path = output_directory / README_FILENAME
    readme_path.write_text(
        "Automatically generated paired diagnostics for the luminosity-function "
        f"Hubble sweep. All statistics use the same {len(ids)} fit-selection "
        "object IDs in every run. In the absolute-value panel, points are the "
        "median and error bars are the 16th-84th percentiles of the actual "
        "object-level dmi values within each redshift bin. In delta panels, "
        "error bars are the "
        "16th-84th percentiles of the actual paired object differences within "
        "each bin; plotted points remain the medians. The figure uses "
        "src/qvc/hubble/style.mplstyle.\n",
        encoding="utf-8",
    )
    return {
        "diagnostic_pdf": diagnostic_pdf,
        "diagnostic_png": diagnostic_png,
        "summary_csv": summary_path,
        "binned_csv": binned_path,
        "readme": readme_path,
    }
