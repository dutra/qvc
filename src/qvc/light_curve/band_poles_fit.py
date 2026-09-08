"""Independent inference and derived-parameter plumbing for the band-pole model."""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import distributions as dist
from qvc.light_curve.multiband_model_shared_latent_band_poles_blr import (
    VARIANT,
    DEFAULT_TRANSITION,
    continuum_effective_timescale,
    SharedLatentBandPolesBLRQS,
    make_multiband_shared_latent_band_poles_blr_model,
)


def build_explicit_band_poles_params(raw, lam_rf, *, lam_lya_rf=None, disk_order=3):
    from qvc.light_curve.fit_light_curves import build_explicit_model_params_relflux

    raw = dict(raw)
    log_s_uv = jnp.asarray(raw["log_tau_slow_uv_driver"])
    log_f = jnp.asarray(raw["log_tau_fast_center0"])
    eta = jnp.asarray(raw["eta_tau"])
    # Reuse generic amplitude, nuisance and lag conversions, then supply the
    # independent model's poles. No old kernel or SLB builder is invoked.
    raw["log_tau_slow_center0"] = log_s_uv
    params = build_explicit_model_params_relflux(
        raw, lam_rf, lam_lya_rf=lam_lya_rf, shared_latent=True
    )
    log_s_band = log_s_uv[..., None] + eta[..., None] * jnp.log(
        jnp.asarray(lam_rf) / 2500.0
    )
    log_f_band = jnp.broadcast_to(log_f[..., None], log_s_band.shape)
    params["tau_fast_driver"] = jnp.exp(log_f)
    params["tau_slow_uv_driver"] = jnp.exp(log_s_uv)
    params.pop("tau_slow_driver", None)
    params["tau_fast_band"] = jnp.exp(log_f_band)
    params["tau_slow_band"] = jnp.exp(log_s_band)
    params["log_kernel_param"] = jnp.concatenate([log_f_band, log_s_band], axis=-1)
    params["log_tau_fast_uv"] = log_f
    params["log_tau_slow_center0"] = log_s_uv + eta * jnp.log(
        params["lambda_center_rf"] / 2500.0
    )
    params["band_pole_below_common"] = (log_s_band < log_f[..., None]).astype(
        jnp.float64
    )
    lag_uv = jnp.asarray(params["lag0"]) * (2500.0 / params["lambda_center_rf"]) ** (
        4.0 / 3.0
    )
    fast, slow, lag_uv = jnp.broadcast_arrays(jnp.exp(log_f), jnp.exp(log_s_uv), lag_uv)
    times = jax.vmap(
        lambda f, s, l: continuum_effective_timescale(f, s, l, disk_order=disk_order)
    )(fast.reshape(-1), slow.reshape(-1), lag_uv.reshape(-1))
    params["log_tau_uv"] = jnp.log(times).reshape(fast.shape)
    return params


def add_band_poles_prediction_params(samples, lam_rf, *, lam_lya_rf=None, disk_order=3):
    from qvc.light_curve.fraction_marginalization import (
        scale_prediction_samples_by_fraction,
    )

    out = dict(samples)
    explicit = build_explicit_band_poles_params(
        out, lam_rf, lam_lya_rf=lam_lya_rf, disk_order=disk_order
    )
    out.update({k: np.asarray(v) for k, v in explicit.items()})
    # This is the raw reference-flux amplitude coordinate used by the builder.
    out["log_sigma_center0"] = np.asarray(explicit["log_sigma_center0_relflux"])
    return scale_prediction_samples_by_fraction(out)


def build_single_object_model_band_poles(
    obj_dict,
    lam_rf,
    log_jitter_mean,
    *,
    lam_lya_rf=None,
    disable_linear_trend=False,
    disable_lag_blr=False,
    disable_lag_bc=False,
    drop_band_lyman_alpha=False,
    tau_fast_truncated=False,
    n_blr_terms=1,
    use_erlang=True,
    shared_latent=False,
    erlang_order=3,
    disk_order=3,
    use_fast_solver=False,
    drw_parameterization=False,
    enforce_positive_flux_guard=False,
    enable_seeing_dependence=False,
    psf_fraction_mode=None,
    eta_prior_profile="modified",
    band_poles_transition=DEFAULT_TRANSITION,
):
    """Return the relative-flux quasi-separable model for one object."""
    from qvc.light_curve.fit_light_curves import (
        _compute_log_jitter_mean_grid,
        _get_object_active_noise_calibration_masks,
        resolve_eta_prior_profile,
        compute_flux_line_ratio_offsets,
        compute_lambda_center_rf,
        compute_log_igm_transmission_band,
        empirical_logmeanexp,
        eta_sigma_prior,
        eta_tau_prior,
        fit_logit_normal,
        has_jitter,
        lag0_prior,
        linear_trend_band_offset_raw_prior_relflux,
        linear_trend_prior_relflux,
        log_sigma_center0_relflux_prior,
        log_tau_fast_separation_raw_prior,
        log_tau_slow_center0_prior,
        mag_residual_to_relative_flux,
        magerr_residual_to_relative_fluxerr,
        mean_prior_relflux,
        ordered_log_tau_fast,
        reference_flux_from_mean_magnitudes,
        sample_flux_line_latent_params,
        sample_linear_trend_with_band_offsets,
        scale_variable_relflux_amplitudes,
        select_fraction_draws_for_bands,
        zero_mean,
    )

    if (
        drw_parameterization
        or use_fast_solver
        or enforce_positive_flux_guard
        or not use_erlang
    ):
        raise ValueError(
            "Band-pole model requires its own cascaded quasi-separable kernel"
        )
    eta_prior_profile = resolve_eta_prior_profile(eta_prior_profile, VARIANT)
    if n_blr_terms != 1:
        raise ValueError(
            "model_variant='shared_latent_band_poles_blr' supports only n_blr_terms=1."
        )
    t, bidx = obj_dict["X"]
    y = obj_dict["y"]
    yerr = obj_dict["yerr"]
    survey_idx = jnp.asarray(obj_dict["survey_idx"], dtype=jnp.int32)
    z = float(obj_dict["z"])
    B = int(len(lam_rf))
    lambda_center_rf = compute_lambda_center_rf(lam_rf)
    if lam_lya_rf is None:
        lam_lya_rf = lam_rf
    lam_lya_rf = jnp.asarray(lam_lya_rf, dtype=lam_rf.dtype)
    bands = tuple((str(b) for b in obj_dict.get("bands", [])))
    if bands:
        log_igm_transmission_band = compute_log_igm_transmission_band(bands, z)
    else:
        log_igm_transmission_band = jnp.zeros(B, dtype=lam_rf.dtype)
    baseline_flux_by_band = reference_flux_from_mean_magnitudes(obj_dict["mags_means"])
    fraction_draws = None
    logit_normal_mean = None
    logit_normal_scale_tril = None
    if psf_fraction_mode is not None:
        if psf_fraction_mode not in {"empirical", "logit-normal"}:
            raise ValueError(
                "Flux-likelihood PSF fraction mode must be 'empirical' or 'logit-normal'."
            )
        fraction_draws = select_fraction_draws_for_bands(obj_dict, bands)
        if psf_fraction_mode == "logit-normal":
            logit_normal_mean, _, logit_normal_scale_tril = fit_logit_normal(
                fraction_draws
            )
    if "y_relflux_fit" in obj_dict and "yerr_relflux_fit" in obj_dict:
        y_relflux = jnp.asarray(obj_dict["y_relflux_fit"], dtype=float)
        yerr_relflux = jnp.asarray(obj_dict["yerr_relflux_fit"], dtype=float)
    else:
        y_relflux = mag_residual_to_relative_flux(y)
        yerr_relflux = magerr_residual_to_relative_fluxerr(y, yerr)
    bidx_np = np.asarray(bidx)
    yerr_relflux_np = np.asarray(yerr_relflux, dtype=float)
    log_jitter_mean_relflux, log_jitter_active_mask_relflux = (
        _compute_log_jitter_mean_grid(
            yerr_relflux_np, bidx_np, np.asarray(survey_idx, dtype=np.int32), B
        )
    )
    _, survey_offset_active_mask = _get_object_active_noise_calibration_masks(
        obj_dict, B
    )

    def model():
        eta_sigma = numpyro.sample("eta_sigma", eta_sigma_prior(eta_prior_profile))
        eta_tau = numpyro.sample("eta_tau", eta_tau_prior(eta_prior_profile))
        tau_center_prior_fn = log_tau_slow_center0_prior
        tau_center_prior = tau_center_prior_fn(0.0, z, 2500.0)
        log_tau_slow_center0 = numpyro.sample(
            "log_tau_slow_uv_driver", tau_center_prior
        )
        log_tau_separation_raw = numpyro.sample(
            "log_tau_separation_raw", log_tau_fast_separation_raw_prior()
        )
        log_tau_fast_center0 = numpyro.deterministic(
            "log_tau_fast_center0",
            ordered_log_tau_fast(log_tau_slow_center0, log_tau_separation_raw),
        )
        log_sigma_center0 = numpyro.sample(
            "log_sigma_center0",
            log_sigma_center0_relflux_prior(eta_sigma, lambda_center_rf),
        )
        linear_trend, linear_trend_band_offset, _linear_trend_band = (
            sample_linear_trend_with_band_offsets(
                B=B,
                disable_linear_trend=disable_linear_trend,
                trend_prior_dist=linear_trend_prior_relflux(t_ref=t, z=z),
                band_offset_raw_prior_dist=linear_trend_band_offset_raw_prior_relflux(),
                shared_linear_trend=True,
            )
        )
        lag0 = numpyro.sample("lag0", lag0_prior(z=z))
        log_lag0 = numpyro.deterministic("log_lag0", jnp.log(lag0))
        lag_beta = numpyro.deterministic("lag_beta", jnp.asarray(4.0 / 3.0))
        line_ratio_offsets = compute_flux_line_ratio_offsets(
            lam_rf,
            lambda_center_rf=lambda_center_rf,
            eta_sigma=eta_sigma,
            log_igm_transmission_band=log_igm_transmission_band,
        )
        (
            mean,
            dlog_amp_blr,
            dlog_amp_blr2,
            log_lag_blr,
            log_lag_blr2,
            log_jitter,
            survey_delta_mag,
            seeing_mean_slope,
            seeing_scatter_slope,
            dlog_amp_bc,
            log_lag_ratio_bc_to_blr,
        ) = sample_flux_line_latent_params(
            B=B,
            z=z,
            log_lag0=log_lag0,
            log_jitter_mean=log_jitter_mean_relflux,
            log_jitter_active_mask=log_jitter_active_mask_relflux,
            survey_offset_active_mask=survey_offset_active_mask,
            seeing_active_mask=(
                obj_dict.get("seeing_active_mask") if enable_seeing_dependence else None
            ),
            line_ratio_offsets=line_ratio_offsets,
            mean_prior_dist=mean_prior_relflux(),
            disable_lag_blr=disable_lag_blr,
            disable_lag_bc=True,
            n_blr_terms=n_blr_terms,
        )
        _ = numpyro.deterministic(
            "log_tau_fake", float(obj_dict.get("log_tau_fake", -99.0))
        )
        _ = numpyro.deterministic(
            "log_sigma_fake", float(obj_dict.get("log_sigma_fake", -99.0))
        )
        raw_params = dict(
            log_tau_slow_center0=log_tau_slow_center0,
            log_tau_slow_uv_driver=log_tau_slow_center0,
            eta_tau=eta_tau,
            log_tau_fast_center0=log_tau_fast_center0,
            log_sigma_center0=log_sigma_center0,
            lambda_center_rf=lambda_center_rf,
            linear_trend=linear_trend,
            linear_trend_band_offset=linear_trend_band_offset,
            mean=mean,
            dlog_amp_blr=dlog_amp_blr,
            dlog_amp_blr2=dlog_amp_blr2,
            log_lag_blr=log_lag_blr,
            log_lag_blr2=log_lag_blr2,
            log_jitter=log_jitter,
            survey_delta_mag=survey_delta_mag,
            seeing_mean_slope=seeing_mean_slope,
            seeing_scatter_slope=seeing_scatter_slope,
            lag0=lag0,
            lag_beta=lag_beta,
            log_igm_transmission_band=log_igm_transmission_band,
            eta_sigma=eta_sigma,
        )
        if dlog_amp_bc is not None:
            raw_params["dlog_amp_bc"] = dlog_amp_bc
            raw_params["log_lag_ratio_bc_to_blr"] = log_lag_ratio_bc_to_blr
        params = build_explicit_band_poles_params(
            raw_params, lam_rf, lam_lya_rf=lam_lya_rf, disk_order=disk_order
        )
        numpyro.deterministic("lambda_center_rf", params["lambda_center_rf"])
        numpyro.deterministic(
            "log_sigma_center0_relflux", params["log_sigma_center0_relflux"]
        )
        numpyro.deterministic("log_sigma_uv_relflux", params["log_sigma_uv_relflux"])
        numpyro.deterministic("log_sigma_uv", params["log_sigma_uv"])
        numpyro.deterministic("log_tau_uv", params["log_tau_uv"])
        numpyro.deterministic("log_tau_fast_uv", params["log_tau_fast_uv"])
        numpyro.deterministic("tau_fast", params["tau_fast_band"])
        numpyro.deterministic("tau_slow", params["tau_slow_band"])
        numpyro.deterministic("tau_fast_driver", params["tau_fast_driver"])
        numpyro.deterministic("tau_slow_uv_driver", params["tau_slow_uv_driver"])
        numpyro.deterministic(
            "band_pole_below_common", params["band_pole_below_common"]
        )
        numpyro.deterministic("amp_cont_relflux", params["amp_cont_relflux"])
        numpyro.deterministic("amp_bc_relflux", params["amp_bc_relflux"])
        numpyro.deterministic("amp_blr_relflux", params["amp_blr_relflux"])
        numpyro.deterministic("amp_blr2_relflux", params["amp_blr2_relflux"])
        numpyro.deterministic("amp_cont", params["amp_cont"])
        numpyro.deterministic("amp_bc", params["amp_bc"])
        numpyro.deterministic("amp_blr", params["amp_blr"])
        numpyro.deterministic("amp_blr2", params["amp_blr2"])
        numpyro.deterministic(
            "log_igm_transmission_band", params["log_igm_transmission_band"]
        )
        numpyro.deterministic("igm_transmission_band", params["igm_transmission_band"])
        numpyro.deterministic("lag_disk", params["lag_disk"])
        numpyro.deterministic("lag_bc", params["lag_bc"])
        numpyro.deterministic("lag_blr", params["lag_blr"])
        numpyro.deterministic("lag_blr2", params["lag_blr2"])
        numpyro.deterministic("F0_cont_band", baseline_flux_by_band)
        model_factory = make_multiband_shared_latent_band_poles_blr_model
        m = model_factory(
            X=(t, bidx),
            y=y_relflux,
            yerr=yerr_relflux,
            n_band=B,
            survey_idx=survey_idx,
            baseline_flux_by_band=baseline_flux_by_band,
            zero_mean=zero_mean,
            has_jitter=has_jitter,
            seeing_covariate=(
                obj_dict.get("seeing_covariate") if enable_seeing_dependence else None
            ),
            **{
                "disk_order": disk_order,
                "blr_order": erlang_order,
                "transition": band_poles_transition,
            },
        )
        if psf_fraction_mode == "empirical":
            component_loglikes = jax.vmap(
                lambda fractions: m.log_prob(
                    scale_variable_relflux_amplitudes(params, fractions)
                )
            )(jnp.asarray(fraction_draws))
            numpyro.deterministic(
                "psf_agn_fraction_responsibility", jax.nn.softmax(component_loglikes)
            )
            numpyro.factor("loglike", empirical_logmeanexp(component_loglikes))
        elif psf_fraction_mode == "logit-normal":
            fraction_logits = numpyro.sample(
                "psf_agn_fraction_logit",
                dist.MultivariateNormal(
                    loc=jnp.asarray(logit_normal_mean),
                    scale_tril=jnp.asarray(logit_normal_scale_tril),
                ),
            )
            fractions = numpyro.deterministic(
                "psf_agn_fraction", jax.nn.sigmoid(fraction_logits)
            )
            numpyro.factor(
                "loglike",
                m.log_prob(scale_variable_relflux_amplitudes(params, fractions)),
            )
        else:
            numpyro.factor("loglike", m.log_prob(params))

    return model


def posterior_band_poles_moments(flat_samples, bands, *, disk_order=3, blr_order=3):
    """Exact draw-wise moments from flattened catalog/prediction samples."""
    f = jnp.asarray(flat_samples["tau_fast_driver"])
    s = jnp.asarray(flat_samples["tau_slow_uv_driver"])
    by_band = lambda key: jnp.column_stack([flat_samples[f"{key}_{b}"] for b in bands])

    def one(f, s, sb, ld, lb, ac, ab):
        kernel = SharedLatentBandPolesBLRQS(
            tau_fast=jnp.atleast_1d(f),
            tau_slow_uv=jnp.atleast_1d(s),
            tau_slow_band=sb,
            lag_disk=ld,
            lag_blr=lb,
            amp_cont=ac,
            amp_blr=ab,
            disk_order=disk_order,
            blr_order=blr_order,
        )
        return (
            kernel.effective_timescales(),
            kernel.stationary_rms(),
            kernel.continuum_effective_timescales(),
        )

    total_tau, total_rms, cont_tau = jax.jit(jax.vmap(one))(
        f,
        s,
        by_band("tau_slow_band"),
        by_band("lag_disk"),
        by_band("lag_blr"),
        by_band("amp_cont_relflux"),
        by_band("amp_blr_relflux"),
    )
    ld_uv = jnp.asarray(flat_samples["lag0"]) * (
        2500.0 / jnp.asarray(flat_samples["lambda_center_rf"])
    ) ** (4.0 / 3.0)
    uv = jax.jit(
        jax.vmap(
            lambda tf, ts, l: continuum_effective_timescale(
                tf, ts, l, disk_order=disk_order
            )
        )
    )(f, s, ld_uv)
    return {
        k: np.asarray(v)
        for k, v in dict(
            tau_total=total_tau,
            rms_total=total_rms,
            tau_continuum=cont_tau,
            tau_uv=uv,
            crossing=by_band("tau_slow_band") < f[:, None],
        ).items()
    }


def continuum_structure_function(
    samples, band, tau_rf, *, z, disk_order=3, transition=DEFAULT_TRANSITION
):
    """Continuum disk-filtered SF at median parameters, in legacy RMS units."""
    med = lambda key: float(np.median(np.asarray(samples[key])))
    kernel = SharedLatentBandPolesBLRQS(
        tau_fast=jnp.atleast_1d(med("tau_fast_driver")),
        tau_slow_uv=jnp.atleast_1d(med("tau_slow_uv_driver")),
        tau_slow_band=jnp.atleast_1d(med(f"tau_slow_band_{band}")),
        lag_disk=jnp.atleast_1d(med(f"lag_disk_{band}")),
        lag_blr=jnp.ones(1),
        amp_cont=jnp.atleast_1d(med(f"amp_cont_{band}")),
        amp_blr=jnp.zeros(1),
        disk_order=disk_order,
        blr_order=1,
        transition=transition,
    )
    P = kernel.stationary_covariance()
    h = kernel.observation_model((0.0, 0))
    cov = jax.jit(
        lambda times: jnp.einsum(
            "i,nij,j->n", h, kernel._transition_matrices(times), P @ h
        )
    )(jnp.asarray(tau_rf) * (1.0 + z))
    return np.sqrt(np.maximum(2.0 * (float(h @ P @ h) - np.asarray(cov)), 0.0))
