"""Posterior-predictive closure diagnostics for completeness corrections."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qvc.hubble.hubble_completeness_refactored import COMPLETENESS_FHOST_COL
from qvc.hubble.completeness_strata import (
    COMPLETENESS_STRATUM_CODE_COL,
    COMPLETENESS_STRATUM_COL,
    StratifiedCompletenessBundle,
)
from qvc.hubble.hubble_likelihood import (
    agn_selection_prediction,
    completeness_loglike,
)
from qvc.hubble.latent_alpha_completeness import (
    JOINT_DRAW_INPUT_COUNT,
    LatentAlphaConfig,
    deterministic_joint_draw_indices,
)


@dataclass(frozen=True)
class CompletenessClosureResult:
    """Per-redshift-bin completeness-closure results."""

    summary: pd.DataFrame
    all_bins_pass: bool
    seed: int
    n_posterior_draws: int
    latent_alpha_marginalized_to_c3: bool = False


def _as_draw_matrix(values, *, n_objects, name):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != n_objects:
        raise ValueError(
            f"{name} must have shape (draw, object) or (object,); got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _evaluate_completeness(
    completeness_model,
    magnitude,
    redshift,
    *,
    f_host_2500_psf=None,
    alpha_lambda=None,
):
    mode = getattr(completeness_model, "mode", "2d")
    if mode == "4d_fhost_alpha":
        if f_host_2500_psf is None or alpha_lambda is None:
            raise ValueError(
                "4D completeness closure requires f_host_2500_psf and alpha_lambda."
            )
        values = completeness_model(
            magnitude,
            redshift,
            f_host_2500_psf,
            alpha_lambda,
        )
    elif mode == "3d_fhost":
        if f_host_2500_psf is None:
            raise ValueError(
                "3D completeness closure requires f_host_2500_psf."
            )
        f_host = np.asarray(f_host_2500_psf, dtype=float)
        if f_host.ndim == 2:
            values = np.mean(
                completeness_model(
                    np.asarray(magnitude, dtype=float)[:, None],
                    np.asarray(redshift, dtype=float)[:, None],
                    f_host,
                ),
                axis=1,
            )
        else:
            values = completeness_model(magnitude, redshift, f_host)
    else:
        values = completeness_model(magnitude, redshift)
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def simulate_completeness_closure(
    *,
    model_magnitude_draws,
    sigma_draws,
    redshift,
    completeness_model,
    magnitude_grid,
    correction_completeness_model=None,
    f_host_2500_psf=None,
    alpha_lambda=None,
    redshift_bins=None,
    seed=12345,
    max_abs_mean_zscore=4.0,
    min_detected_per_bin=100,
    latent_alpha_marginalized_to_c3=False,
):
    """Simulate selection and test recovery of zero corrected residual.

    Each input row is one posterior draw of the model-predicted selection
    magnitude and its total Gaussian scatter.  Synthetic magnitudes are drawn,
    accepted with the current completeness probability, and corrected with the
    exact conditional-mean calculation used by the Hubble likelihood.

    ``correction_completeness_model`` is primarily a diagnostic hook: omitting
    it tests self-consistency, while supplying a different model exposes the
    residual pattern produced by a miscalibrated correction.
    """

    redshift = np.asarray(redshift, dtype=float)
    if redshift.ndim != 1 or redshift.size == 0 or not np.all(np.isfinite(redshift)):
        raise ValueError("redshift must be a nonempty finite one-dimensional array.")
    n_objects = redshift.size
    model_draws = _as_draw_matrix(
        model_magnitude_draws,
        n_objects=n_objects,
        name="model_magnitude_draws",
    )
    sigma = _as_draw_matrix(sigma_draws, n_objects=n_objects, name="sigma_draws")
    if sigma.shape[0] == 1 and model_draws.shape[0] > 1:
        sigma = np.broadcast_to(sigma, model_draws.shape)
    if sigma.shape != model_draws.shape:
        raise ValueError(
            "sigma_draws must have the same number of posterior draws as "
            "model_magnitude_draws, or contain one broadcastable draw."
        )
    if np.any(sigma <= 0.0):
        raise ValueError("sigma_draws must be strictly positive.")

    magnitude_grid = np.asarray(magnitude_grid, dtype=float)
    if (
        magnitude_grid.ndim != 1
        or magnitude_grid.size < 2
        or not np.all(np.isfinite(magnitude_grid))
        or np.any(np.diff(magnitude_grid) <= 0.0)
    ):
        raise ValueError("magnitude_grid must be finite and strictly increasing.")
    if redshift_bins is None:
        redshift_bins = np.arange(0.4, 3.41, 0.2)
    redshift_bins = np.asarray(redshift_bins, dtype=float)
    if (
        redshift_bins.ndim != 1
        or redshift_bins.size < 2
        or np.any(np.diff(redshift_bins) <= 0.0)
    ):
        raise ValueError("redshift_bins must be a strictly increasing 1D array.")

    min_detected_per_bin = int(min_detected_per_bin)
    if min_detected_per_bin < 1:
        raise ValueError("min_detected_per_bin must be positive.")
    max_abs_mean_zscore = float(max_abs_mean_zscore)
    if not np.isfinite(max_abs_mean_zscore) or max_abs_mean_zscore <= 0.0:
        raise ValueError("max_abs_mean_zscore must be finite and positive.")

    correction_model = correction_completeness_model or completeness_model
    detection_support = getattr(
        completeness_model,
        "magnitude_support",
        (float(magnitude_grid[0]), float(magnitude_grid[-1])),
    )
    correction_support = getattr(
        correction_model,
        "magnitude_support",
        (float(magnitude_grid[0]), float(magnitude_grid[-1])),
    )
    f_host = None if f_host_2500_psf is None else np.asarray(f_host_2500_psf, dtype=float)
    alpha = None if alpha_lambda is None else np.asarray(alpha_lambda, dtype=float)
    for name, value in (("f_host_2500_psf", f_host), ("alpha_lambda", alpha)):
        if value is None:
            continue
        shape_ok = value.shape == redshift.shape
        if name == "f_host_2500_psf":
            shape_ok = shape_ok or (
                value.ndim == 2 and value.shape[0] == redshift.size
            )
        if not shape_ok or not np.all(np.isfinite(value)):
            raise ValueError(
                f"{name} must match redshift (optionally with a posterior-draw "
                "axis for host fraction) and contain only finite values."
            )

    rng = np.random.default_rng(seed)
    simulated_redshift = []
    raw_residuals = []
    corrected_residuals = []
    corrected_sigmas = []
    n_parent_by_object = np.zeros(n_objects, dtype=int)
    n_detected_by_object = np.zeros(n_objects, dtype=int)

    for model_magnitude, total_sigma in zip(model_draws, sigma):
        simulated_magnitude = rng.normal(model_magnitude, total_sigma)
        p_detect = _evaluate_completeness(
            completeness_model,
            simulated_magnitude,
            redshift,
            f_host_2500_psf=f_host,
            alpha_lambda=alpha,
        )
        p_detect = np.where(
            (simulated_magnitude >= detection_support[0])
            & (simulated_magnitude <= detection_support[1]),
            p_detect,
            0.0,
        )
        detected = rng.random(n_objects) < p_detect
        n_parent_by_object += 1
        n_detected_by_object += detected

        _, correction_blob = completeness_loglike(
            m_obs=np.clip(
                simulated_magnitude,
                correction_support[0],
                correction_support[1],
            ),
            m_obs_err=total_sigma,
            m_model=model_magnitude,
            mu_err=total_sigma,
            z=redshift,
            completeness_model=correction_model,
            m_grid=magnitude_grid,
            magnitude_support=correction_support,
            sigma_completeness=0.0,
            f_host_2500_psf=f_host,
            alpha_lambda=alpha,
        )
        correction = np.asarray(correction_blob[1], dtype=float)
        conditional_sigma = np.asarray(correction_blob[2], dtype=float)
        raw = simulated_magnitude - model_magnitude

        simulated_redshift.append(redshift[detected])
        raw_residuals.append(raw[detected])
        corrected_residuals.append((raw - correction)[detected])
        corrected_sigmas.append(conditional_sigma[detected])

    z_selected = np.concatenate(simulated_redshift)
    raw_selected = np.concatenate(raw_residuals)
    corrected_selected = np.concatenate(corrected_residuals)
    sigma_selected = np.concatenate(corrected_sigmas)

    rows = []
    for index, (z_lo, z_hi) in enumerate(zip(redshift_bins[:-1], redshift_bins[1:])):
        in_parent_bin = (redshift >= z_lo) & (redshift < z_hi)
        in_selected_bin = (z_selected >= z_lo) & (z_selected < z_hi)
        raw_bin = raw_selected[in_selected_bin]
        corrected_bin = corrected_selected[in_selected_bin]
        sigma_bin = sigma_selected[in_selected_bin]
        n_parent = int(np.sum(n_parent_by_object[in_parent_bin]))
        n_detected = int(corrected_bin.size)

        mean_raw = float(np.mean(raw_bin)) if n_detected else np.nan
        mean_corrected = float(np.mean(corrected_bin)) if n_detected else np.nan
        corrected_std = (
            float(np.std(corrected_bin, ddof=1)) if n_detected > 1 else np.nan
        )
        mean_err = corrected_std / np.sqrt(n_detected) if n_detected > 1 else np.nan
        mean_zscore = (
            mean_corrected / mean_err
            if np.isfinite(mean_err) and mean_err > 0.0
            else np.nan
        )
        valid_sigma = np.isfinite(sigma_bin) & (sigma_bin > 0.0)
        n_chi2 = int(np.count_nonzero(valid_sigma))
        reduced_chi2 = (
            float(
                np.sum(np.square(corrected_bin[valid_sigma] / sigma_bin[valid_sigma]))
                / max(n_chi2 - 1, 1)
            )
            if n_chi2
            else np.nan
        )
        enough = n_detected >= min_detected_per_bin
        bin_pass = bool(
            enough
            and np.isfinite(mean_zscore)
            and abs(mean_zscore) <= max_abs_mean_zscore
        )
        rows.append(
            {
                "bin_index": index,
                "z_lo": float(z_lo),
                "z_hi": float(z_hi),
                "n_parent": n_parent,
                "n_detected": n_detected,
                "detection_fraction": n_detected / n_parent if n_parent else np.nan,
                "mean_raw_residual": mean_raw,
                "mean_corrected_residual": mean_corrected,
                "std_corrected_residual": corrected_std,
                "mean_corrected_residual_err": mean_err,
                "mean_corrected_zscore": mean_zscore,
                "reduced_chi2_corrected": reduced_chi2,
                "bin_pass": bin_pass,
            }
        )

    summary = pd.DataFrame(rows)
    return CompletenessClosureResult(
        summary=summary,
        all_bins_pass=bool(summary["bin_pass"].all()),
        seed=int(seed),
        n_posterior_draws=int(model_draws.shape[0]),
        latent_alpha_marginalized_to_c3=bool(
            latent_alpha_marginalized_to_c3
        ),
    )


def _optional_data_column(data, name):
    if hasattr(data, "get"):
        value = data.get(name)
    else:
        value = None
    return None if value is None else np.asarray(value, dtype=float)


def _data_column(data, name):
    if hasattr(data, "columns"):
        if name not in data.columns:
            raise KeyError(name)
        return np.asarray(data[name])
    if name not in data:
        raise KeyError(name)
    return np.asarray(data[name])


def simulate_hubble_posterior_closure(
    *,
    posterior_samples,
    agn_data,
    cosmo_model,
    z_pivot_agn,
    agn_pivot_context,
    completeness_params,
    redshift_bins=None,
    seed=12345,
    max_posterior_draws=100,
    max_abs_mean_zscore=4.0,
    min_detected_per_bin=100,
    only_agn=False,
    use_planck_h0_prior=False,
    use_planck_om_prior=False,
    use_alpha_lambda_term=False,
    use_eta_sigma_term=False,
    use_redshift_log_f_term=False,
    use_redshift_mu_term=False,
    latent_alpha_config=None,
):
    """Run closure using the exact selection prediction for posterior draws."""

    samples = np.asarray(posterior_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("posterior_samples must be a nonempty finite 2D array.")
    max_posterior_draws = int(max_posterior_draws)
    if max_posterior_draws < 1:
        raise ValueError("max_posterior_draws must be positive.")
    n_keep = min(samples.shape[0], max_posterior_draws)
    draw_indices = np.linspace(0, samples.shape[0] - 1, n_keep, dtype=int)
    selected_samples = samples[draw_indices]
    if latent_alpha_config is not None and not isinstance(
        latent_alpha_config, LatentAlphaConfig
    ):
        raise TypeError("latent_alpha_config must be a LatentAlphaConfig.")

    model_draws = []
    sigma_draws = []
    for theta in selected_samples:
        prediction = agn_selection_prediction(
            theta,
            agn_data=agn_data,
            cosmo_model=cosmo_model,
            z_pivot_agn=z_pivot_agn,
            agn_pivot_context=agn_pivot_context,
            only_agn=only_agn,
            use_planck_h0_prior=use_planck_h0_prior,
            use_planck_om_prior=use_planck_om_prior,
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
            use_redshift_log_f_term=use_redshift_log_f_term,
            use_redshift_mu_term=use_redshift_mu_term,
            require_selection_fields=True,
            latent_alpha_config=latent_alpha_config,
        )
        model_draws.append(prediction["selection_model_magnitude"])
        sigma_draws.append(prediction["selection_total_error"])

    if isinstance(completeness_params, StratifiedCompletenessBundle):
        codes = _data_column(agn_data, COMPLETENESS_STRATUM_CODE_COL).astype(int)
        redshift = _data_column(agn_data, "z").astype(float)
        if latent_alpha_config is None:
            f_host = _optional_data_column(agn_data, COMPLETENESS_FHOST_COL)
        else:
            raw_host = _data_column(agn_data, "f_host_2500_psf_draws")
            if raw_host.ndim == 1 and raw_host.dtype == object:
                raw_host = np.stack(raw_host)
            raw_host = np.asarray(raw_host, dtype=float)
            if raw_host.shape != (redshift.size, JOINT_DRAW_INPUT_COUNT):
                raise ValueError(
                    "Latent-alpha closure requires aligned host draws with "
                    f"shape {(redshift.size, JOINT_DRAW_INPUT_COUNT)}."
                )
            f_host = raw_host[:, deterministic_joint_draw_indices()]
        alpha = _optional_data_column(agn_data, "alpha_lambda")
        summaries = []
        all_pass = True
        for code, (name, params) in enumerate(
            zip(
                completeness_params.stratum_names,
                completeness_params.params_by_stratum,
            )
        ):
            mask = codes == code
            if not np.any(mask):
                raise ValueError(
                    f"Completeness closure has no fitted objects in stratum {name!r}."
                )
            completeness_model, magnitude_grid = params[:2]
            stratum_result = simulate_completeness_closure(
                model_magnitude_draws=np.asarray(model_draws, dtype=float)[:, mask],
                sigma_draws=np.asarray(sigma_draws, dtype=float)[:, mask],
                redshift=redshift[mask],
                completeness_model=completeness_model,
                magnitude_grid=magnitude_grid,
                f_host_2500_psf=None if f_host is None else f_host[mask],
                alpha_lambda=None if alpha is None else alpha[mask],
                redshift_bins=redshift_bins,
                seed=int(seed) + code,
                max_abs_mean_zscore=max_abs_mean_zscore,
                min_detected_per_bin=min_detected_per_bin,
                latent_alpha_marginalized_to_c3=(
                    latent_alpha_config is not None
                ),
            )
            summary = stratum_result.summary.copy()
            summary.insert(0, COMPLETENESS_STRATUM_COL, name)
            summary.insert(0, COMPLETENESS_STRATUM_CODE_COL, code)
            summaries.append(summary)
            all_pass = all_pass and stratum_result.all_bins_pass
        return CompletenessClosureResult(
            summary=pd.concat(summaries, ignore_index=True),
            all_bins_pass=bool(all_pass),
            seed=int(seed),
            n_posterior_draws=int(len(selected_samples)),
            latent_alpha_marginalized_to_c3=(
                latent_alpha_config is not None
            ),
        )

    completeness_model, magnitude_grid = completeness_params[:2]
    if latent_alpha_config is None:
        closure_f_host = _optional_data_column(
            agn_data, COMPLETENESS_FHOST_COL
        )
    else:
        raw_host = _data_column(agn_data, "f_host_2500_psf_draws")
        if raw_host.ndim == 1 and raw_host.dtype == object:
            raw_host = np.stack(raw_host)
        raw_host = np.asarray(raw_host, dtype=float)
        expected_shape = (
            np.asarray(agn_data["z"]).size,
            JOINT_DRAW_INPUT_COUNT,
        )
        if raw_host.shape != expected_shape:
            raise ValueError(
                "Latent-alpha closure requires aligned host draws with "
                f"shape {expected_shape}."
            )
        closure_f_host = raw_host[:, deterministic_joint_draw_indices()]
    return simulate_completeness_closure(
        model_magnitude_draws=np.asarray(model_draws, dtype=float),
        sigma_draws=np.asarray(sigma_draws, dtype=float),
        redshift=np.asarray(agn_data["z"], dtype=float),
        completeness_model=completeness_model,
        magnitude_grid=magnitude_grid,
        f_host_2500_psf=closure_f_host,
        alpha_lambda=_optional_data_column(agn_data, "alpha_lambda"),
        redshift_bins=redshift_bins,
        seed=seed,
        max_abs_mean_zscore=max_abs_mean_zscore,
        min_detected_per_bin=min_detected_per_bin,
        latent_alpha_marginalized_to_c3=(latent_alpha_config is not None),
    )


def write_completeness_closure_diagnostics(result, plot_path):
    """Write the closure table, verdict metadata, and diagnostic plot."""

    from matplotlib import pyplot as plt

    diagnostics_dir = Path(plot_path) / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = diagnostics_dir / "completeness_closure_summary.csv"
    metadata_path = diagnostics_dir / "completeness_closure_metadata.json"
    plot_file = diagnostics_dir / "completeness_closure.pdf"
    result.summary.to_csv(summary_path, index=False)
    per_stratum_verdicts = (
        {
            str(name): bool(group["bin_pass"].all())
            for name, group in result.summary.groupby(
                COMPLETENESS_STRATUM_COL, sort=False
            )
        }
        if COMPLETENESS_STRATUM_COL in result.summary.columns
        else {}
    )
    metadata_path.write_text(
        json.dumps(
            {
                "all_bins_pass": bool(result.all_bins_pass),
                "n_posterior_draws": int(result.n_posterior_draws),
                "seed": int(result.seed),
                "n_bins": int(len(result.summary)),
                "n_bins_passed": int(result.summary["bin_pass"].sum()),
                "per_stratum_verdicts": per_stratum_verdicts,
                "latent_alpha_marginalized_to_c3": bool(
                    result.latent_alpha_marginalized_to_c3
                ),
                "latent_alpha_closure_semantics": (
                    "response_integrated_over_parent_equals_host_aware_C3"
                    if result.latent_alpha_marginalized_to_c3
                    else "not_applicable"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    def write_plot(summary, output_file, title):
        z_mid = 0.5 * (
            summary["z_lo"].to_numpy(dtype=float)
            + summary["z_hi"].to_numpy(dtype=float)
        )
        fig, (ax_residual, ax_chi2) = plt.subplots(
            2, 1, figsize=(8.0, 7.5), sharex=True
        )
        ax_residual.plot(
            z_mid,
            summary["mean_raw_residual"],
            color="tab:blue",
            marker="o",
            label="Selected, uncorrected",
        )
        ax_residual.errorbar(
            z_mid,
            summary["mean_corrected_residual"],
            yerr=summary["mean_corrected_residual_err"],
            color="tab:red",
            marker="o",
            capsize=3,
            label="Selected, corrected",
        )
        ax_residual.axhline(0.0, color="black", linewidth=1.0)
        ax_residual.set_ylabel("Mean residual (mag)")
        ax_residual.legend(frameon=False)
        ax_residual.grid(alpha=0.25)
        ax_chi2.plot(
            z_mid,
            summary["reduced_chi2_corrected"],
            color="tab:red",
            marker="o",
        )
        ax_chi2.axhline(1.0, color="magenta", linewidth=1.2)
        ax_chi2.set_xlabel("Redshift-bin midpoint")
        ax_chi2.set_ylabel(r"Corrected $\chi^2_\nu$")
        ax_chi2.grid(alpha=0.25)
        fig.suptitle(title)
        fig.tight_layout()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close(fig)

    summary = result.summary
    per_stratum_plots = {}
    if COMPLETENESS_STRATUM_COL in summary.columns:
        # The top-level PDF is a compact aggregate view; detailed closure
        # verdicts remain independent in the per-stratum directories.
        aggregate = summary.groupby("bin_index", sort=True, as_index=False).agg(
            z_lo=("z_lo", "first"),
            z_hi=("z_hi", "first"),
            mean_raw_residual=("mean_raw_residual", "mean"),
            mean_corrected_residual=("mean_corrected_residual", "mean"),
            mean_corrected_residual_err=("mean_corrected_residual_err", "mean"),
            reduced_chi2_corrected=("reduced_chi2_corrected", "mean"),
        )
        write_plot(
            aggregate,
            plot_file,
            "Stratified completeness posterior-predictive closure: "
            + ("PASS" if result.all_bins_pass else "FAIL"),
        )
        for name, group in summary.groupby(COMPLETENESS_STRATUM_COL, sort=False):
            stratum_plot = (
                diagnostics_dir
                / "strata"
                / str(name)
                / "completeness_closure.pdf"
            )
            stratum_pass = bool(group["bin_pass"].all())
            write_plot(
                group,
                stratum_plot,
                f"Completeness closure: {name} ({'PASS' if stratum_pass else 'FAIL'})",
            )
            per_stratum_plots[str(name)] = stratum_plot
    else:
        write_plot(
            summary,
            plot_file,
            "Completeness posterior-predictive closure: "
            + ("PASS" if result.all_bins_pass else "FAIL"),
        )
    return {
        "summary_csv": summary_path,
        "metadata_json": metadata_path,
        "plot_pdf": plot_file,
        "per_stratum_plot_pdfs": per_stratum_plots,
    }
