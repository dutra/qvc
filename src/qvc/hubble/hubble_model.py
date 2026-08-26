import numpy as np
import pandas as pd
from scipy.special import expit
from collections import OrderedDict
from dataclasses import dataclass
from qvc.hubble.latent_alpha_completeness import (
    RESPONSE_COEFFICIENT_PRIOR_SIGMA,
    response_coefficient_names,
    response_coefficient_prior_specs,
)
from qvc.hubble.fitted_color_completeness import COLOR_STRENGTH_PARAMETER

AGN_ALPHA_LAMBDA_PARAM = "gamma_alpha_lambda"
AGN_ALPHA_LAMBDA_OBS = "alpha_lambda"
AGN_ALPHA_LAMBDA_ERR = "alpha_lambda_err"
AGN_ETA_SIGMA_PARAM = "gamma_eta_sigma"
AGN_ETA_SIGMA_OBS = "eta_sigma"
AGN_ETA_SIGMA_ERR = "eta_sigma_err"
AGN_LOGF_Z_PARAM = "gamma_log_f_z"
AGN_MU_Z_PARAM = "gamma_mu_z"
AGN_INTRINSIC_SCATTER_MAG_CENTER = 2.5 * 0.2  # 0.2 dex in luminosity = 0.5 mag
AGN_LOG_F_PRIOR_HALF_WIDTH = 1.6
AGN_LOG_F_PRIOR = (
    np.log(AGN_INTRINSIC_SCATTER_MAG_CENTER) - AGN_LOG_F_PRIOR_HALF_WIDTH,
    np.log(AGN_INTRINSIC_SCATTER_MAG_CENTER) + AGN_LOG_F_PRIOR_HALF_WIDTH,
)
PLANCK_H0_PRIOR = (67.37 - 0.54, 67.37 + 0.54)
PLANCK_OM0_PRIOR = (0.315 - 0.007, 0.315 + 0.007)
AGN_PIVOT_RULE = "rounded_median_v1"
LATENT_ALPHA_BETA_PARAM = "beta_alpha_L"
LATENT_ALPHA_RESPONSE_PARAM_PREFIX = "alpha_sel_"
LATENT_ALPHA_RESPONSE_PRIOR_SIGMA = RESPONSE_COEFFICIENT_PRIOR_SIGMA
LATENT_ALPHA_LUMINOSITY_MODES = ("off", "fixed", "joint")


def latent_alpha_response_parameter_names(magnitude_interaction=False):
    """Return the authoritative latent-alpha surface coefficient order.

    The first index is the Legendre redshift order.  ``linear`` and
    ``quadratic`` multiply the standardized alpha coordinate and its centered
    square, respectively.  Optional ``*_magnitude`` coefficients multiply the
    same terms by the standardized apparent-magnitude coordinate.
    """

    return response_coefficient_names(bool(magnitude_interaction))


def normalize_latent_alpha_luminosity_mode(mode):
    normalized = str(mode).strip().lower()
    if normalized not in LATENT_ALPHA_LUMINOSITY_MODES:
        raise ValueError(
            "Invalid latent-alpha luminosity mode "
            f"{mode!r}; expected one of {LATENT_ALPHA_LUMINOSITY_MODES}."
        )
    return normalized


def get_agn_model_spec(use_alpha_lambda_term=False, use_eta_sigma_term=False):
    req_params = (
        "M0_agn",
        "alpha_agn",
        "beta_agn",
    )
    req_obs = (
        "log_sigma_uv",
        "log_tau_uv_rf",
    )
    req_errs = (
        "log_sigma_uv_std_psd",
        "log_tau_uv_rf_std_psd",
        "log_sigma_uv_log_tau_uv_rf_cov_psd",
    )
    if use_alpha_lambda_term:
        req_params += (AGN_ALPHA_LAMBDA_PARAM,)
        req_obs += (AGN_ALPHA_LAMBDA_OBS,)
        req_errs += (AGN_ALPHA_LAMBDA_ERR,)
    if use_eta_sigma_term:
        req_params += (AGN_ETA_SIGMA_PARAM,)
        req_obs += (AGN_ETA_SIGMA_OBS,)
        req_errs += (AGN_ETA_SIGMA_ERR,)
    return req_params, req_obs, req_errs


# Keep the default non-alpha model as the module-level import contract.
agn_model_req_params, agn_model_req_obs, agn_model_req_errs = get_agn_model_spec(
    use_alpha_lambda_term=False,
    use_eta_sigma_term=False,
)
agn_model_pidx = {k: i for i, k in enumerate(agn_model_req_params)}
agn_model_oidx = {k: i for i, k in enumerate(agn_model_req_obs)}
agn_model_eidx = {k: i for i, k in enumerate(agn_model_req_errs)}

def _require(keys, provided, where):
    miss = set(keys) - set(provided)
    if miss:
        raise KeyError(f"Missing {where}: {sorted(miss)}")


def infer_model_option_flags(cosmo_model, sample_dim, only_sna=False, only_agn=False):
    combos = []
    redshift_mu_options = (False,) if only_sna else (False, True)
    for use_alpha_lambda_term in (False, True):
        for use_eta_sigma_term in (False, True):
            for use_redshift_log_f_term in (False, True):
                for use_redshift_mu_term in redshift_mu_options:
                    _, labels, _ = get_model_params(
                        cosmo_model,
                        only_sna=only_sna,
                        only_agn=only_agn,
                        use_planck_h0_prior=False,
                        use_alpha_lambda_term=use_alpha_lambda_term,
                        use_eta_sigma_term=use_eta_sigma_term,
                        use_redshift_log_f_term=use_redshift_log_f_term,
                        use_redshift_mu_term=use_redshift_mu_term,
                    )
                    combos.append(
                        (
                            len(labels),
                            use_alpha_lambda_term,
                            use_eta_sigma_term,
                            use_redshift_log_f_term,
                            use_redshift_mu_term,
                        )
                    )
    matches = [combo for combo in combos if combo[0] == sample_dim]
    if len(matches) == 1:
        (_, use_alpha_lambda_term, use_eta_sigma_term,
         use_redshift_log_f_term, use_redshift_mu_term) = matches[0]
        return {
            "use_alpha_lambda_term": use_alpha_lambda_term,
            "use_eta_sigma_term": use_eta_sigma_term,
            "use_redshift_log_f_term": use_redshift_log_f_term,
            "use_redshift_mu_term": use_redshift_mu_term,
        }
    expected = sorted({n for n, _, _, _, _ in combos})
    raise ValueError(
        f"Could not infer model option flags for sample_dim={sample_dim}, "
        f"cosmo_model={cosmo_model!r}. Expected one of {expected} columns."
    )


def resolve_model_option_flags(
    cosmo_model,
    sample_dim,
    *,
    only_sna=False,
    only_agn=None,
    use_planck_h0_prior=False,
    use_alpha_lambda_term=None,
    use_eta_sigma_term=None,
    use_redshift_log_f_term=None,
    use_redshift_mu_term=None,
):
    combos = []
    redshift_mu_options = (False,) if only_sna else (False, True)
    only_agn_options = (False,) if only_sna and only_agn is None else (
        (False, True) if only_agn is None else (bool(only_agn),)
    )
    for only_agn_flag in only_agn_options:
        for alpha_flag in (False, True):
            for eta_flag in (False, True):
                for logf_flag in (False, True):
                    for mu_flag in redshift_mu_options:
                        _, labels, _ = get_model_params(
                            cosmo_model,
                            only_sna=only_sna,
                            only_agn=only_agn_flag,
                            use_planck_h0_prior=use_planck_h0_prior,
                            use_alpha_lambda_term=alpha_flag,
                            use_eta_sigma_term=eta_flag,
                            use_redshift_log_f_term=logf_flag,
                            use_redshift_mu_term=mu_flag,
                        )
                        combos.append(
                            {
                                "sample_dim": len(labels),
                                "only_agn": only_agn_flag,
                                "use_alpha_lambda_term": alpha_flag,
                                "use_eta_sigma_term": eta_flag,
                                "use_redshift_log_f_term": logf_flag,
                                "use_redshift_mu_term": mu_flag,
                            }
                        )

    matches = [combo for combo in combos if combo["sample_dim"] == sample_dim]
    if use_alpha_lambda_term is not None:
        matches = [
            combo for combo in matches
            if combo["use_alpha_lambda_term"] == use_alpha_lambda_term
        ]
    if use_eta_sigma_term is not None:
        matches = [
            combo for combo in matches
            if combo["use_eta_sigma_term"] == use_eta_sigma_term
        ]
    if use_redshift_log_f_term is not None:
        matches = [
            combo for combo in matches
            if combo["use_redshift_log_f_term"] == use_redshift_log_f_term
        ]
    if use_redshift_mu_term is not None:
        matches = [
            combo for combo in matches
            if combo["use_redshift_mu_term"] == use_redshift_mu_term
        ]
    if only_agn is None and len(matches) > 1:
        non_agn_matches = [combo for combo in matches if not combo["only_agn"]]
        if len(non_agn_matches) == 1:
            matches = non_agn_matches

    if len(matches) == 1:
        return {
            "only_agn": matches[0]["only_agn"],
            "use_alpha_lambda_term": matches[0]["use_alpha_lambda_term"],
            "use_eta_sigma_term": matches[0]["use_eta_sigma_term"],
            "use_redshift_log_f_term": matches[0]["use_redshift_log_f_term"],
            "use_redshift_mu_term": matches[0]["use_redshift_mu_term"],
        }

    expected = sorted({combo["sample_dim"] for combo in combos})
    requested = {
        "use_alpha_lambda_term": use_alpha_lambda_term,
        "use_eta_sigma_term": use_eta_sigma_term,
        "use_redshift_log_f_term": use_redshift_log_f_term,
        "use_redshift_mu_term": use_redshift_mu_term,
    }
    if len(matches) > 1:
        matching_configs = [
            {
                "use_alpha_lambda_term": combo["use_alpha_lambda_term"],
                "use_eta_sigma_term": combo["use_eta_sigma_term"],
                "use_redshift_log_f_term": combo["use_redshift_log_f_term"],
                "use_redshift_mu_term": combo["use_redshift_mu_term"],
                "only_agn": combo["only_agn"],
            }
            for combo in matches
        ]
        raise ValueError(
            f"Ambiguous model option flags for sample_dim={sample_dim}, "
            f"cosmo_model={cosmo_model!r}. Matching configurations: "
            f"{matching_configs}. Pass explicit use_alpha_lambda_term, "
            f"use_eta_sigma_term, use_redshift_log_f_term, and/or use_redshift_mu_term."
        )

    raise ValueError(
        f"Could not resolve model option flags for sample_dim={sample_dim}, "
        f"cosmo_model={cosmo_model!r}, requested={requested}. Expected one of "
        f"{expected} columns."
    )


def infer_use_alpha_lambda_term(cosmo_model, sample_dim, only_sna=False, only_agn=False):
    return infer_model_option_flags(
        cosmo_model, sample_dim, only_sna=only_sna, only_agn=only_agn
    )["use_alpha_lambda_term"]


def infer_use_eta_sigma_term(cosmo_model, sample_dim, only_sna=False, only_agn=False):
    return infer_model_option_flags(
        cosmo_model, sample_dim, only_sna=only_sna, only_agn=only_agn
    )["use_eta_sigma_term"]


def infer_use_redshift_log_f_term(cosmo_model, sample_dim, only_sna=False, only_agn=False):
    return infer_model_option_flags(
        cosmo_model, sample_dim, only_sna=only_sna, only_agn=only_agn
    )["use_redshift_log_f_term"]


def infer_use_redshift_mu_term(cosmo_model, sample_dim, only_sna=False, only_agn=False):
    return infer_model_option_flags(
        cosmo_model, sample_dim, only_sna=only_sna, only_agn=only_agn
    )["use_redshift_mu_term"]


def evaluate_log_f(params_dict, z, z_pivot, use_redshift_log_f_term=False):
    z = np.asarray(z, dtype=float)
    log_f0 = float(params_dict["log_f"])
    if not use_redshift_log_f_term:
        return np.full_like(z, log_f0, dtype=float)
    gamma_f = float(params_dict[AGN_LOGF_Z_PARAM])
    return log_f0 + gamma_f * np.log10((1.0 + z) / (1.0 + float(z_pivot)))


def evaluate_mu_redshift_term(params_dict, z, z_pivot, use_redshift_mu_term=False):
    """Evaluate the pivoted mean AGN Hubble-modulus evolution in magnitudes."""
    z = np.asarray(z, dtype=float)
    z_pivot = float(z_pivot)
    if not use_redshift_mu_term:
        return np.zeros_like(z, dtype=float)
    if np.any(~np.isfinite(z)) or np.any(z <= -1.0):
        raise ValueError("AGN mean-evolution redshifts must be finite and greater than -1.")
    if not np.isfinite(z_pivot) or z_pivot <= -1.0:
        raise ValueError("AGN mean-evolution z_pivot must be finite and greater than -1.")
    gamma_mu_z = float(params_dict[AGN_MU_Z_PARAM])
    return gamma_mu_z * np.log10((1.0 + z) / (1.0 + z_pivot))


def agn_model_pack_params(params_dict, use_alpha_lambda_term=False, use_eta_sigma_term=False):
    req_params, _, _ = get_agn_model_spec(
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    _require(req_params, params_dict, "params")
    params = np.array([params_dict[k] for k in req_params], dtype=float)
    return params


def _fixed_pivot_from_observable(key, values):
    pivot = float(np.nanmedian(np.asarray(values, dtype=float)))
    if key == "log_sigma_uv":
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            rounded_linear_pivot = np.round(np.power(10.0, pivot), 1)
    elif key == "log_tau_uv_rf":
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            rounded_linear_pivot = (
                np.round(np.power(10.0, pivot) / 100.0) * 100.0
            )
    else:
        return pivot

    if not np.isfinite(rounded_linear_pivot) or rounded_linear_pivot <= 0.0:
        raise ValueError(
            f"Rounded linear AGN pivot for observable {key!r} must be finite "
            f"and positive; got {rounded_linear_pivot!r}."
        )
    return float(np.log10(rounded_linear_pivot))


def _normalize_reference_object_ids(values, *, where):
    raw_ids = tuple(values)
    if len(raw_ids) == 0:
        raise ValueError(f"{where} must not be empty.")

    normalized_ids = []
    for object_id in raw_ids:
        missing = pd.isna(object_id)
        if np.ndim(missing) != 0 or bool(missing):
            raise ValueError(
                f"{where} must contain only present scalar object IDs."
            )
        normalized = (
            object_id.decode("utf-8")
            if isinstance(object_id, (bytes, np.bytes_))
            else str(object_id)
        )
        if not normalized.strip():
            raise ValueError(f"{where} must not contain empty object IDs.")
        normalized_ids.append(normalized)
    return tuple(normalized_ids)


@dataclass(frozen=True)
class AgnPivotContext:
    """Immutable definition of the observable coordinate system for one AGN fit."""

    observable_names: tuple
    values: tuple
    z_range: tuple
    reference_object_ids: tuple
    rule: str = AGN_PIVOT_RULE

    def __post_init__(self):
        names = tuple(str(name) for name in self.observable_names)
        values = tuple(float(value) for value in self.values)
        z_range = tuple(float(value) for value in self.z_range)
        object_ids = _normalize_reference_object_ids(
            self.reference_object_ids,
            where="AgnPivotContext.reference_object_ids",
        )
        rule = str(self.rule)

        if len(names) == 0:
            raise ValueError("AgnPivotContext.observable_names must not be empty.")
        if len(set(names)) != len(names):
            raise ValueError(
                "AgnPivotContext.observable_names must be unique; "
                f"got {names!r}."
            )
        if len(values) != len(names):
            raise ValueError(
                "AgnPivotContext values/name length mismatch: "
                f"{len(values)} values for {len(names)} names."
            )
        if not np.all(np.isfinite(np.asarray(values, dtype=float))):
            raise ValueError("AgnPivotContext.values must all be finite.")
        if len(z_range) != 2 or not np.all(np.isfinite(np.asarray(z_range, dtype=float))):
            raise ValueError(
                "AgnPivotContext.z_range must contain exactly two finite values."
            )
        if z_range[0] > z_range[1]:
            raise ValueError(
                "AgnPivotContext.z_range must be ordered as (minimum, maximum); "
                f"got {z_range!r}."
            )
        if rule != AGN_PIVOT_RULE:
            raise ValueError(
                f"Unsupported AGN pivot rule {rule!r}; expected {AGN_PIVOT_RULE!r}."
            )

        object.__setattr__(self, "observable_names", names)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "z_range", z_range)
        object.__setattr__(self, "reference_object_ids", object_ids)
        object.__setattr__(self, "rule", rule)

    def as_array(self, use_alpha_lambda_term=False, use_eta_sigma_term=False):
        """Return values in the canonical order for the requested model."""

        _, expected_names, _ = get_agn_model_spec(
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
        )
        if self.observable_names != expected_names:
            raise ValueError(
                "AGN pivot observables do not match the active model: "
                f"stored={self.observable_names!r}, expected={expected_names!r}."
            )
        return np.asarray(self.values, dtype=float)

    def as_dict(self):
        return dict(zip(self.observable_names, self.values))


def build_agn_pivot_context(
    df_agn,
    z_range,
    use_alpha_lambda_term=False,
    use_eta_sigma_term=False,
):
    """Compute the one AGN observable pivot context used by an entire fit."""

    _, req_obs, _ = get_agn_model_spec(
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    required = ("z", "object_id") + req_obs
    _require(required, df_agn, "AGN pivot reference data")

    try:
        z = np.asarray(df_agn["z"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("AGN pivot reference column 'z' must be numeric.") from exc
    if np.any(~np.isfinite(z)):
        raise ValueError(
            "AGN pivot reference column 'z' must contain only finite values."
        )
    z_range = tuple(float(value) for value in z_range)
    if len(z_range) != 2 or not np.all(np.isfinite(z_range)) or z_range[0] > z_range[1]:
        raise ValueError(
            "z_range must contain two finite ordered values; "
            f"got {z_range!r}."
        )
    fit_mask = (z >= z_range[0]) & (z <= z_range[1])
    if not np.any(fit_mask):
        raise ValueError(
            "Cannot compute AGN pivots: no reference objects fall inside "
            f"inclusive z_range={z_range!r}."
        )

    pivot_values = []
    for name in req_obs:
        try:
            values = np.asarray(df_agn[name], dtype=float)[fit_mask]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"AGN pivot reference observable {name!r} must be numeric."
            ) from exc
        bad = ~np.isfinite(values)
        if np.any(bad):
            raise ValueError(
                f"AGN pivot reference observable {name!r} contains "
                f"{int(np.count_nonzero(bad))} nonfinite fitted value(s)."
            )
        pivot = _fixed_pivot_from_observable(name, values)
        if not np.isfinite(pivot):
            raise ValueError(
                f"Computed nonfinite AGN pivot for observable {name!r}."
            )
        pivot_values.append(pivot)

    object_ids = _normalize_reference_object_ids(
        np.asarray(df_agn["object_id"], dtype=object)[fit_mask],
        where="AGN pivot reference object_id values",
    )
    return AgnPivotContext(
        observable_names=req_obs,
        values=tuple(pivot_values),
        z_range=z_range,
        reference_object_ids=object_ids,
    )


def agn_model_pack_obs(
    obs_dict,
    use_alpha_lambda_term=False,
    use_eta_sigma_term=False,
    *,
    pivot_context,
):
    _, req_obs, req_errs = get_agn_model_spec(
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    _require(req_obs, obs_dict, "observables")
    _require(req_errs, obs_dict, "errors")
    obs = np.array([obs_dict[k] for k in req_obs], dtype=float)
    err = np.array([obs_dict[k] for k in req_errs], dtype=float)
    validate_agn_observable_uncertainties(
        err,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        object_ids=(
            np.asarray(obs_dict["object_id"], dtype=object)
            if "object_id" in obs_dict
            else None
        ),
    )
    if not isinstance(pivot_context, AgnPivotContext):
        raise TypeError(
            "pivot_context must be an AgnPivotContext; "
            f"got {type(pivot_context).__name__}."
        )
    pivots = pivot_context.as_array(
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    return obs, err, pivots

def hinge(x, a, b, x0):
    return a + b * np.maximum(0.0, x - x0)

def logistic(x, A, k, x0):
     return A * expit(k*(x - x0))

def M_model_agn(params_arr, obs_arr, pivots_array, use_alpha_lambda_term=False, use_eta_sigma_term=False):
    req_params, req_obs, _ = get_agn_model_spec(
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    pidx = {k: i for i, k in enumerate(req_params)}
    oidx = {k: i for i, k in enumerate(req_obs)}

    M0_agn = params_arr[pidx["M0_agn"]]
    alpha_agn = params_arr[pidx["alpha_agn"]]
    beta_agn = params_arr[pidx["beta_agn"]]
    #gamma_agn = params_arr[agn_model_pidx["gamma_agn"]]

    log_sigma_uv = obs_arr[oidx["log_sigma_uv"]]
    log_tau_uv_rf = obs_arr[oidx["log_tau_uv_rf"]]
    log_sigma_uv_pivot = pivots_array[oidx["log_sigma_uv"]]
    log_tau_uv_rf_pivot = pivots_array[oidx["log_tau_uv_rf"]]

    #dm_psf_correction = obs_arr[agn_model_oidx["dm_psf_correction"]]
    #dm_psf_correction_pivot = pivots_array[agn_model_oidx["dm_psf_correction"]]
    # PL_slope_blue = obs_arr[agn_model_oidx["PL_slope_blue"]]

    # A = params_arr[agn_model_pidx["A"]]
    # k = params_arr[agn_model_pidx["k"]]
    # x0 = params_arr[agn_model_pidx["x0"]]

    M_pred = (
        M0_agn
        + alpha_agn * (log_sigma_uv - log_sigma_uv_pivot)
        + beta_agn  * (log_tau_uv_rf - log_tau_uv_rf_pivot)
        #+ gamma_agn * (dm_psf_correction - dm_psf_correction_pivot)
        #+ logistic(PL_slope_blue, A, k, x0)
    )
    if use_alpha_lambda_term:
        gamma_alpha_lambda = params_arr[pidx[AGN_ALPHA_LAMBDA_PARAM]]
        alpha_lambda = obs_arr[oidx[AGN_ALPHA_LAMBDA_OBS]]
        alpha_lambda_pivot = pivots_array[oidx[AGN_ALPHA_LAMBDA_OBS]]
        M_pred = M_pred + gamma_alpha_lambda * (alpha_lambda - alpha_lambda_pivot)
    if use_eta_sigma_term:
        gamma_eta_sigma = params_arr[pidx[AGN_ETA_SIGMA_PARAM]]
        eta_sigma = obs_arr[oidx[AGN_ETA_SIGMA_OBS]]
        eta_sigma_pivot = pivots_array[oidx[AGN_ETA_SIGMA_OBS]]
        M_pred = M_pred + gamma_eta_sigma * (eta_sigma - eta_sigma_pivot)
    return M_pred


def M_model_agn_posterior_samples(
    params_samples,
    obs_arr,
    pivots_array,
    use_alpha_lambda_term=False,
    use_eta_sigma_term=False,
):
    """Evaluate the affine AGN magnitude relation for all samples at once.

    Parameters
    ----------
    params_samples
        Posterior parameter matrix in canonical AGN-model order, shaped
        ``(n_samples, n_parameters)``.
    obs_arr
        Canonically ordered observable arrays, shaped
        ``(n_observables, n_objects)``.

    Returns
    -------
    numpy.ndarray
        Predicted absolute magnitudes shaped ``(n_samples, n_objects)``.
    """
    req_params, req_obs, _ = get_agn_model_spec(
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    samples = np.asarray(params_samples, dtype=float)
    observables = np.asarray(obs_arr, dtype=float)
    pivots = np.asarray(pivots_array, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != len(req_params):
        raise ValueError(
            "params_samples must have shape "
            f"(n_samples, {len(req_params)}); got {samples.shape}"
        )
    if observables.ndim != 2 or observables.shape[0] != len(req_obs):
        raise ValueError(
            "obs_arr must have shape "
            f"({len(req_obs)}, n_objects); got {observables.shape}"
        )
    if pivots.shape != (len(req_obs),):
        raise ValueError(
            f"pivots_array must have shape ({len(req_obs)},); got {pivots.shape}"
        )

    pidx = {name: index for index, name in enumerate(req_params)}
    oidx = {name: index for index, name in enumerate(req_obs)}
    predicted = np.broadcast_to(
        samples[:, pidx["M0_agn"], None],
        (samples.shape[0], observables.shape[1]),
    ).copy()
    coefficient_terms = [
        ("alpha_agn", "log_sigma_uv"),
        ("beta_agn", "log_tau_uv_rf"),
    ]
    if use_alpha_lambda_term:
        coefficient_terms.append(
            (AGN_ALPHA_LAMBDA_PARAM, AGN_ALPHA_LAMBDA_OBS)
        )
    if use_eta_sigma_term:
        coefficient_terms.append((AGN_ETA_SIGMA_PARAM, AGN_ETA_SIGMA_OBS))
    for parameter_name, observable_name in coefficient_terms:
        predicted += (
            samples[:, pidx[parameter_name], None]
            * (
                observables[oidx[observable_name]][None, :]
                - pivots[oidx[observable_name]]
            )
        )
    return predicted


def _format_invalid_agn_locations(mask, object_ids):
    indices = np.flatnonzero(np.atleast_1d(mask))
    if object_ids is None:
        return f"indices {indices[:5].tolist()}"
    identifiers = np.atleast_1d(np.asarray(object_ids, dtype=object))
    if identifiers.size != np.atleast_1d(mask).size:
        raise ValueError(
            "object_ids must contain one value per AGN uncertainty column; "
            f"got {identifiers.size} identifiers for "
            f"{np.atleast_1d(mask).size} columns."
        )
    return f"object_id values {identifiers[indices[:5]].astype(str).tolist()}"


def validate_agn_observable_uncertainties(
    err_arr,
    use_alpha_lambda_term=False,
    use_eta_sigma_term=False,
    *,
    object_ids=None,
):
    """Validate canonical AGN observable errors and their covariance.

    The first two uncertainty rows are standard deviations and the third is
    their covariance.  A valid 2x2 covariance matrix must satisfy
    ``abs(covariance) <= sigma_std * tau_std``.  Optional observable-error
    rows are also required to be finite and nonnegative.
    """
    _, _, req_errs = get_agn_model_spec(
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    errors = np.asarray(err_arr, dtype=float)
    if errors.ndim not in (1, 2) or errors.shape[0] != len(req_errs):
        raise ValueError(
            "err_arr must have shape "
            f"({len(req_errs)},) or ({len(req_errs)}, n_objects); "
            f"got {errors.shape}"
        )

    eidx = {name: index for index, name in enumerate(req_errs)}
    error_names = [
        "log_sigma_uv_std_psd",
        "log_tau_uv_rf_std_psd",
    ]
    if use_alpha_lambda_term:
        error_names.append(AGN_ALPHA_LAMBDA_ERR)
    if use_eta_sigma_term:
        error_names.append(AGN_ETA_SIGMA_ERR)

    for name in error_names:
        values = errors[eidx[name]]
        invalid = ~np.isfinite(values) | (values < 0.0)
        if np.any(invalid):
            locations = _format_invalid_agn_locations(invalid, object_ids)
            raise ValueError(
                f"AGN uncertainty {name!r} must be finite and nonnegative; "
                f"invalid value(s) at {locations}."
            )

    covariance_name = "log_sigma_uv_log_tau_uv_rf_cov_psd"
    covariance = errors[eidx[covariance_name]]
    invalid_covariance = ~np.isfinite(covariance)
    if np.any(invalid_covariance):
        locations = _format_invalid_agn_locations(
            invalid_covariance, object_ids
        )
        raise ValueError(
            f"AGN covariance {covariance_name!r} must be finite; "
            f"invalid value(s) at {locations}."
        )

    sigma_std = errors[eidx["log_sigma_uv_std_psd"]]
    tau_std = errors[eidx["log_tau_uv_rf_std_psd"]]
    covariance_bound = sigma_std * tau_std
    roundoff_tolerance = (
        64.0
        * np.finfo(float).eps
        * np.maximum.reduce(
            [
                np.ones_like(covariance_bound),
                np.abs(covariance),
                covariance_bound,
            ]
        )
    )
    invalid_psd = np.abs(covariance) > (
        covariance_bound + roundoff_tolerance
    )
    if np.any(invalid_psd):
        locations = _format_invalid_agn_locations(invalid_psd, object_ids)
        raise ValueError(
            "AGN sigma/tau covariance violates "
            "|covariance| <= sigma_std * tau_std at "
            f"{locations}."
        )
    return errors


def _clip_roundoff_negative_variance(components, *, where):
    component_array = np.asarray(list(components.values()), dtype=float)
    variance = np.sum(component_array, axis=0)
    if np.any(~np.isfinite(variance)):
        raise ValueError(f"{where} produced nonfinite propagated variance.")
    absolute_scale = np.sum(np.abs(component_array), axis=0)
    tolerance = (
        64.0
        * np.finfo(float).eps
        * np.maximum(np.finfo(float).tiny, absolute_scale)
    )
    materially_negative = variance < -tolerance
    if np.any(materially_negative):
        locations = np.flatnonzero(np.atleast_1d(materially_negative))
        raise ValueError(
            f"{where} produced materially negative propagated variance "
            f"at indices {locations[:5].tolist()}."
        )
    roundoff_zero = np.abs(variance) <= tolerance
    return np.where(roundoff_zero, 0.0, np.maximum(variance, 0.0))


def M_model_agn_observable_variance_posterior(
    params_samples,
    err_arr,
    use_alpha_lambda_term=False,
    use_eta_sigma_term=False,
):
    """Average per-object observable variance over posterior coefficients.

    This intentionally excludes the posterior variance of the global
    relation parameters themselves.  That uncertainty is correlated between
    objects and belongs in a model/posterior band, not in independent data
    error bars.
    """
    req_params, _, req_errs = get_agn_model_spec(
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    samples = np.asarray(params_samples, dtype=float)
    errors = np.asarray(err_arr, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != len(req_params):
        raise ValueError(
            "params_samples must have shape "
            f"(n_samples, {len(req_params)}); got {samples.shape}"
        )
    if samples.shape[0] == 0:
        raise ValueError("params_samples must contain at least one sample")
    if np.any(~np.isfinite(samples)):
        raise ValueError("params_samples must contain only finite values")
    if errors.ndim != 2 or errors.shape[0] != len(req_errs):
        raise ValueError(
            "err_arr must have shape "
            f"({len(req_errs)}, n_objects); got {errors.shape}"
        )
    validate_agn_observable_uncertainties(
        errors,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )

    pidx = {name: index for index, name in enumerate(req_params)}
    eidx = {name: index for index, name in enumerate(req_errs)}
    alpha = samples[:, pidx["alpha_agn"]]
    beta = samples[:, pidx["beta_agn"]]
    sigma_std = errors[eidx["log_sigma_uv_std_psd"]]
    tau_std = errors[eidx["log_tau_uv_rf_std_psd"]]
    sigma_tau_cov = errors[
        eidx["log_sigma_uv_log_tau_uv_rf_cov_psd"]
    ]
    components = {
        "sigma": np.mean(np.square(alpha)) * np.square(sigma_std),
        "tau": np.mean(np.square(beta)) * np.square(tau_std),
        "covariance": (
            2.0 * np.mean(alpha * beta) * sigma_tau_cov
        ),
    }
    if use_alpha_lambda_term:
        gamma_alpha_lambda = samples[:, pidx[AGN_ALPHA_LAMBDA_PARAM]]
        alpha_lambda_err = errors[eidx[AGN_ALPHA_LAMBDA_ERR]]
        components["alpha_lambda"] = (
            np.mean(np.square(gamma_alpha_lambda))
            * np.square(alpha_lambda_err)
        )
    if use_eta_sigma_term:
        gamma_eta_sigma = samples[:, pidx[AGN_ETA_SIGMA_PARAM]]
        eta_sigma_err = errors[eidx[AGN_ETA_SIGMA_ERR]]
        components["eta_sigma"] = (
            np.mean(np.square(gamma_eta_sigma))
            * np.square(eta_sigma_err)
        )
    variance = _clip_roundoff_negative_variance(
        components,
        where="Posterior AGN observable-error propagation",
    )
    return variance, components


def M_model_agn_err(
    params_arr,
    obs_arr,
    err_arr,
    pivots_array,
    check_negative=False,
    use_alpha_lambda_term=False,
    use_eta_sigma_term=False,
):
    req_params, _, req_errs = get_agn_model_spec(
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    params_arr = np.asarray(params_arr, dtype=float)
    if params_arr.shape != (len(req_params),):
        raise ValueError(
            f"params_arr must have shape ({len(req_params)},); "
            f"got {params_arr.shape}"
        )
    if np.any(~np.isfinite(params_arr)):
        raise ValueError("params_arr must contain only finite values")
    err_arr = validate_agn_observable_uncertainties(
        err_arr,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    pidx = {k: i for i, k in enumerate(req_params)}
    eidx = {k: i for i, k in enumerate(req_errs)}

    alpha_agn = params_arr[pidx["alpha_agn"]]
    beta_agn = params_arr[pidx["beta_agn"]]
    
    log_sigma_uv_std_psd = err_arr[eidx["log_sigma_uv_std_psd"]]
    log_tau_uv_rf_std_psd = err_arr[eidx["log_tau_uv_rf_std_psd"]]
    log_sigma_uv_log_tau_uv_rf_cov_psd = err_arr[eidx["log_sigma_uv_log_tau_uv_rf_cov_psd"]]

    # gamma_agn   = params_arr[agn_model_pidx["gamma_agn"]]
    # dm_psf_correction_err = err_arr[agn_model_eidx["dm_psf_correction_err"]]
    components = {
        "sigma": (alpha_agn * log_sigma_uv_std_psd) ** 2,
        "tau": (beta_agn * log_tau_uv_rf_std_psd) ** 2,
        "covariance": (
            2
            * alpha_agn
            * beta_agn
            * log_sigma_uv_log_tau_uv_rf_cov_psd
        ),
    }
    if use_alpha_lambda_term:
        gamma_alpha_lambda = params_arr[pidx[AGN_ALPHA_LAMBDA_PARAM]]
        alpha_lambda_err = err_arr[eidx[AGN_ALPHA_LAMBDA_ERR]]
        components["alpha_lambda"] = (
            gamma_alpha_lambda * alpha_lambda_err
        ) ** 2
    if use_eta_sigma_term:
        gamma_eta_sigma = params_arr[pidx[AGN_ETA_SIGMA_PARAM]]
        eta_sigma_err = err_arr[eidx[AGN_ETA_SIGMA_ERR]]
        components["eta_sigma"] = (gamma_eta_sigma * eta_sigma_err) ** 2
    r = _clip_roundoff_negative_variance(
        components,
        where="AGN observable-error propagation",
    )
    if check_negative:
        return np.sqrt(r), None
    return np.sqrt(r)


def get_model_params(
    cosmo_model,
    only_sna=False,
    only_agn=False,
    use_planck_h0_prior=False,
    use_planck_om_prior=False,
    use_alpha_lambda_term=False,
    use_eta_sigma_term=False,
    use_redshift_log_f_term=False,
    use_redshift_mu_term=False,
    use_latent_alpha_completeness=False,
    latent_alpha_luminosity_mode="off",
    latent_alpha_beta_prior=(-0.5, 0.5),
    latent_alpha_magnitude_interaction=False,
    use_fitted_color_completeness=False,
):
    if only_sna and only_agn:
        raise ValueError("only_sna and only_agn cannot both be True.")
    latent_alpha_luminosity_mode = normalize_latent_alpha_luminosity_mode(
        latent_alpha_luminosity_mode
    )
    if use_latent_alpha_completeness and only_sna:
        raise ValueError("Latent-alpha completeness requires an AGN likelihood.")
    if use_fitted_color_completeness and only_sna:
        raise ValueError("Fitted-color completeness requires an AGN likelihood.")
    if use_fitted_color_completeness and use_latent_alpha_completeness:
        raise ValueError(
            "Fitted-color and latent-alpha completeness cannot be enabled "
            "simultaneously."
        )
    
    priors = OrderedDict([
        ("M0_sn",       (-20, -18)),    # SN absolute magnitude, MLE: ~-19.3

        ("M0_agn",   (-26.0, -18.0)),
        ("alpha_agn", (-20,  20.0)),
        ("beta_agn",  (-20.0,  20.0)),
        (AGN_ALPHA_LAMBDA_PARAM, (-20.0, 20.0)),
        (AGN_ETA_SIGMA_PARAM, (-20.0, 20.0)),
        
        # ("A",    (-5.0,  5.0)),
        # ("k",    (0,  20.0)),
        # ("x0",   (-2.0,  1.0)),

        #("gamma_agn", (-100.0, 100.0)),
        # ("A_red",    (-5.0,  0.0)),   # expect negative (e.g. ~ -2)
        # ("k_red",    (0.1,  5.0)),    # >0 (e.g. ~ 1–3 per dex)
        # ("x0_red",   (-5.0,  5.0)),    # bend near where trend starts

        ("log_f",     AGN_LOG_F_PRIOR),
        (AGN_LOGF_Z_PARAM, (-10.0, 10.0)),
        (AGN_MU_Z_PARAM, (-10.0, 10.0)),
        #("sigma_b",   (-1,  1)),

        ("H0",       PLANCK_H0_PRIOR if use_planck_h0_prior else (60.0, 80.0)),
        ("Om0",      PLANCK_OM0_PRIOR if use_planck_om_prior else (0.0, 1.0)),
        
    ])
    if not use_alpha_lambda_term:
        priors.pop(AGN_ALPHA_LAMBDA_PARAM)
    if not use_eta_sigma_term:
        priors.pop(AGN_ETA_SIGMA_PARAM)
    if not use_redshift_log_f_term:
        priors.pop(AGN_LOGF_Z_PARAM)
    if not use_redshift_mu_term or only_sna:
        priors.pop(AGN_MU_Z_PARAM)
    if only_agn:
        priors.pop("M0_sn")

    # Selection-surface parameters belong to the selected-data model, not the
    # AGN standardization relation.  Insert them immediately after the AGN
    # scatter/evolution terms and before cosmology for a stable checkpoint
    # order.  They retain tuple bounds for existing consumers; the samplers
    # recognize their prefix and apply the truncated-Gaussian density.
    if use_latent_alpha_completeness:
        insertion = OrderedDict()
        response_specs = response_coefficient_prior_specs(
            latent_alpha_magnitude_interaction
        )
        for label in latent_alpha_response_parameter_names(
            latent_alpha_magnitude_interaction
        ):
            spec = response_specs[label]
            insertion[label] = (float(spec["low"]), float(spec["high"]))
        if latent_alpha_luminosity_mode == "joint":
            try:
                beta_low, beta_high = map(float, latent_alpha_beta_prior)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "latent_alpha_beta_prior must contain two numeric bounds."
                ) from exc
            if not np.isfinite(beta_low) or not np.isfinite(beta_high) or beta_low >= beta_high:
                raise ValueError(
                    "latent_alpha_beta_prior must contain finite ordered bounds."
                )
            insertion[LATENT_ALPHA_BETA_PARAM] = (beta_low, beta_high)

        rebuilt = OrderedDict()
        for key, value in priors.items():
            rebuilt[key] = value
            if key == "log_f":
                rebuilt.update(insertion)
        priors = rebuilt

    if use_fitted_color_completeness:
        rebuilt = OrderedDict()
        for key, value in priors.items():
            rebuilt[key] = value
            if key == "log_f":
                rebuilt[COLOR_STRENGTH_PARAMETER] = (-1.0, 1.0)
        priors = rebuilt

    # Select cosmological parameters based on model
    if cosmo_model == 'FlatLambdaCDM':
        pass
    elif cosmo_model == 'FlatwCDM':
        priors |= OrderedDict([
            ("w0",          (-3.0, 1.0))
        ])
    elif cosmo_model == 'Flatw0waCDM':
        priors |= OrderedDict([
            ("w0", (-3.0, 1.0)),   # covers phantom (<-1), Λ (-1), quintessence (> -1), and even w>0
            ("wa", (-30, 1))    # symmetric variation
        ])
    elif cosmo_model == 'FlatwpwaCDM':
        priors |= OrderedDict([
            ("wp", (-10.0, 1.0)),   # covers phantom (<-1), Λ (-1), quintessence (> -1), and even w>0
            ("wa", (-50, 500))    # symmetric variation
        ])

    else:
        raise ValueError("cosmo_model must be 'FlatwCDM' or 'Flatw0waCDM'")

    model_labels = list(priors.keys())
    
    # Map model_labels to LaTeX-compatible labels
    latex_labels = {
        "gamma_sn": r"$\gamma_{\rm SN}$",
        "tau_Ms": r"$\tau_{M_s}$",
        "M0_sn": r"$M^0_{\rm SN}$",
        "M0_agn": r"$M^0_{\rm AGN}$",
        "alpha_agn": r"$\alpha_{\rm AGN}$",
        "beta_agn": r"$\beta_{\rm AGN}$",
        AGN_ALPHA_LAMBDA_PARAM: r"$\gamma_{\alpha_\lambda}$",
        AGN_ETA_SIGMA_PARAM: r"$\gamma_{\eta_\sigma}$",
        "gamma_agn": r"$\gamma_{\rm AGN}$",
        "log_f": r"$\log f$",
        AGN_LOGF_Z_PARAM: r"$\gamma_{\log f,z}$",
        AGN_MU_Z_PARAM: r"$\gamma_{\mu,z}$",
        LATENT_ALPHA_BETA_PARAM: r"$\beta_{\alpha L}$",
        COLOR_STRENGTH_PARAMETER: r"$s_{\rm color}$",
        "sigma_b": r"$\sigma_{\rm b}$",
        "H0": r"$H_0$",
        "Om0": r"$\Omega_{m,0}$",
        "w0": r"$w_0$",
        "wp": r"$w_p$",
        "wa": r"$w_a$"
    }
    for order in range(4):
        latex_labels[
            f"{LATENT_ALPHA_RESPONSE_PARAM_PREFIX}z_p{order}_linear"
        ] = rf"$s^{{(1)}}_{{{order}}}$"
        latex_labels[
            f"{LATENT_ALPHA_RESPONSE_PARAM_PREFIX}z_p{order}_quadratic"
        ] = rf"$s^{{(2)}}_{{{order}}}$"
        latex_labels[
            f"{LATENT_ALPHA_RESPONSE_PARAM_PREFIX}mag_z_p{order}_linear"
        ] = rf"$s^{{(1m)}}_{{{order}}}$"
        latex_labels[
            f"{LATENT_ALPHA_RESPONSE_PARAM_PREFIX}mag_z_p{order}_quadratic"
        ] = rf"$s^{{(2m)}}_{{{order}}}$"
    model_labels_latex = [latex_labels.get(label, label) for label in model_labels]
    
    return priors, model_labels, model_labels_latex
