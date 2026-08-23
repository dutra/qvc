"""Posterior-predictive closure diagnostics for completeness corrections."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qvc.hubble.hubble_completeness_refactored import COMPLETENESS_FHOST_COL
from qvc.hubble.hubble_likelihood import (
    agn_selection_prediction,
    completeness_loglike,
)


@dataclass(frozen=True)
class CompletenessClosureResult:
    """Per-redshift-bin completeness-closure results."""

    summary: pd.DataFrame
    all_bins_pass: bool
    seed: int
    n_posterior_draws: int


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
        values = completeness_model(magnitude, redshift, f_host_2500_psf)
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
        if value is not None and (value.shape != redshift.shape or not np.all(np.isfinite(value))):
            raise ValueError(f"{name} must match redshift and contain only finite values.")

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
    )


def _optional_data_column(data, name):
    if hasattr(data, "get"):
        value = data.get(name)
    else:
        value = None
    return None if value is None else np.asarray(value, dtype=float)


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
            require_selection_fields=True,
        )
        model_draws.append(prediction["selection_model_magnitude"])
        sigma_draws.append(prediction["selection_total_error"])

    completeness_model, magnitude_grid = completeness_params[:2]
    return simulate_completeness_closure(
        model_magnitude_draws=np.asarray(model_draws, dtype=float),
        sigma_draws=np.asarray(sigma_draws, dtype=float),
        redshift=np.asarray(agn_data["z"], dtype=float),
        completeness_model=completeness_model,
        magnitude_grid=magnitude_grid,
        f_host_2500_psf=_optional_data_column(agn_data, COMPLETENESS_FHOST_COL),
        alpha_lambda=_optional_data_column(agn_data, "alpha_lambda"),
        redshift_bins=redshift_bins,
        seed=seed,
        max_abs_mean_zscore=max_abs_mean_zscore,
        min_detected_per_bin=min_detected_per_bin,
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
    metadata_path.write_text(
        json.dumps(
            {
                "all_bins_pass": bool(result.all_bins_pass),
                "n_posterior_draws": int(result.n_posterior_draws),
                "seed": int(result.seed),
                "n_bins": int(len(result.summary)),
                "n_bins_passed": int(result.summary["bin_pass"].sum()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    summary = result.summary
    z_mid = 0.5 * (
        summary["z_lo"].to_numpy(dtype=float)
        + summary["z_hi"].to_numpy(dtype=float)
    )
    fig, (ax_residual, ax_chi2) = plt.subplots(
        2,
        1,
        figsize=(8.0, 7.5),
        sharex=True,
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
    fig.suptitle(
        "Completeness posterior-predictive closure: "
        + ("PASS" if result.all_bins_pass else "FAIL")
    )
    fig.tight_layout()
    fig.savefig(plot_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {
        "summary_csv": summary_path,
        "metadata_json": metadata_path,
        "plot_pdf": plot_file,
    }
