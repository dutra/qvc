import numpy as np
from scipy.special import expit
from collections import OrderedDict

AGN_ALPHA_LAMBDA_PARAM = "gamma_alpha_lambda"
AGN_ALPHA_LAMBDA_OBS = "alpha_lambda"
AGN_ALPHA_LAMBDA_ERR = "alpha_lambda_err"
AGN_ETA_SIGMA_PARAM = "gamma_eta_sigma"
AGN_ETA_SIGMA_OBS = "eta_sigma"
AGN_ETA_SIGMA_ERR = "eta_sigma_err"
AGN_LOGF_Z_PARAM = "gamma_log_f_z"
AGN_INTRINSIC_SCATTER_MAG_CENTER = 2.5 * 0.2  # 0.2 dex in luminosity = 0.5 mag
AGN_LOG_F_PRIOR = (np.log(AGN_INTRINSIC_SCATTER_MAG_CENTER) - 0.8,
                   np.log(AGN_INTRINSIC_SCATTER_MAG_CENTER) + 0.8)


def get_agn_model_spec(use_alpha_lambda_term=False, use_eta_sigma_term=False):
    req_params = (
        "M0_agn",
        "alpha_agn",
        "beta_agn",
    )
    req_obs = (
        "log_sigma_hat0",
        "log_sigma_uv",
        "log_tau_uv_rf",
    )
    req_errs = (
        "log_sigma_hat0_err",
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


def infer_model_option_flags(cosmo_model, sample_dim, only_sna=False):
    combos = []
    for use_alpha_lambda_term in (False, True):
        for use_eta_sigma_term in (False, True):
            for use_redshift_log_f_term in (False, True):
                _, labels, _ = get_model_params(
                    cosmo_model,
                    only_sna=only_sna,
                    use_alpha_lambda_term=use_alpha_lambda_term,
                    use_eta_sigma_term=use_eta_sigma_term,
                    use_redshift_log_f_term=use_redshift_log_f_term,
                )
                combos.append(
                    (
                        len(labels),
                        use_alpha_lambda_term,
                        use_eta_sigma_term,
                        use_redshift_log_f_term,
                    )
                )
    matches = [combo for combo in combos if combo[0] == sample_dim]
    if len(matches) == 1:
        _, use_alpha_lambda_term, use_eta_sigma_term, use_redshift_log_f_term = matches[0]
        return {
            "use_alpha_lambda_term": use_alpha_lambda_term,
            "use_eta_sigma_term": use_eta_sigma_term,
            "use_redshift_log_f_term": use_redshift_log_f_term,
        }
    expected = sorted({n for n, _, _ in combos})
    raise ValueError(
        f"Could not infer model option flags for sample_dim={sample_dim}, "
        f"cosmo_model={cosmo_model!r}. Expected one of {expected} columns."
    )


def resolve_model_option_flags(
    cosmo_model,
    sample_dim,
    *,
    only_sna=False,
    use_alpha_lambda_term=None,
    use_eta_sigma_term=None,
    use_redshift_log_f_term=None,
):
    combos = []
    for alpha_flag in (False, True):
        for eta_flag in (False, True):
            for logf_flag in (False, True):
                _, labels, _ = get_model_params(
                    cosmo_model,
                    only_sna=only_sna,
                    use_alpha_lambda_term=alpha_flag,
                    use_eta_sigma_term=eta_flag,
                    use_redshift_log_f_term=logf_flag,
                )
                combos.append(
                    {
                        "sample_dim": len(labels),
                        "use_alpha_lambda_term": alpha_flag,
                        "use_eta_sigma_term": eta_flag,
                        "use_redshift_log_f_term": logf_flag,
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

    if len(matches) == 1:
        return {
            "use_alpha_lambda_term": matches[0]["use_alpha_lambda_term"],
            "use_eta_sigma_term": matches[0]["use_eta_sigma_term"],
            "use_redshift_log_f_term": matches[0]["use_redshift_log_f_term"],
        }

    expected = sorted({combo["sample_dim"] for combo in combos})
    requested = {
        "use_alpha_lambda_term": use_alpha_lambda_term,
        "use_eta_sigma_term": use_eta_sigma_term,
        "use_redshift_log_f_term": use_redshift_log_f_term,
    }
    if len(matches) > 1:
        matching_configs = [
            {
                "use_alpha_lambda_term": combo["use_alpha_lambda_term"],
                "use_eta_sigma_term": combo["use_eta_sigma_term"],
                "use_redshift_log_f_term": combo["use_redshift_log_f_term"],
            }
            for combo in matches
        ]
        raise ValueError(
            f"Ambiguous model option flags for sample_dim={sample_dim}, "
            f"cosmo_model={cosmo_model!r}. Matching configurations: "
            f"{matching_configs}. Pass explicit use_alpha_lambda_term, "
            f"use_eta_sigma_term, and/or use_redshift_log_f_term."
        )

    raise ValueError(
        f"Could not resolve model option flags for sample_dim={sample_dim}, "
        f"cosmo_model={cosmo_model!r}, requested={requested}. Expected one of "
        f"{expected} columns."
    )


def infer_use_alpha_lambda_term(cosmo_model, sample_dim, only_sna=False):
    return infer_model_option_flags(
        cosmo_model, sample_dim, only_sna=only_sna
    )["use_alpha_lambda_term"]


def infer_use_eta_sigma_term(cosmo_model, sample_dim, only_sna=False):
    return infer_model_option_flags(
        cosmo_model, sample_dim, only_sna=only_sna
    )["use_eta_sigma_term"]


def infer_use_redshift_log_f_term(cosmo_model, sample_dim, only_sna=False):
    return infer_model_option_flags(
        cosmo_model, sample_dim, only_sna=only_sna
    )["use_redshift_log_f_term"]


def evaluate_log_f(params_dict, z, z_pivot, use_redshift_log_f_term=False):
    z = np.asarray(z, dtype=float)
    log_f0 = float(params_dict["log_f"])
    if not use_redshift_log_f_term:
        return np.full_like(z, log_f0, dtype=float)
    gamma_f = float(params_dict[AGN_LOGF_Z_PARAM])
    return log_f0 + gamma_f * np.log10((1.0 + z) / (1.0 + float(z_pivot)))


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
        return float(np.log10(max(np.round(10.0**pivot, 1), 1e-8)))
    if key == "log_tau_uv_rf":
        return float(np.log10(max(np.round(10.0**pivot / 100.0) * 100.0, 1e-8)))
    return pivot


def agn_model_pack_obs(obs_dict, use_alpha_lambda_term=False, use_eta_sigma_term=False):
    _, req_obs, req_errs = get_agn_model_spec(
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    _require(req_obs, obs_dict, "observables")
    _require(req_errs, obs_dict, "errors")
    obs = np.array([obs_dict[k] for k in req_obs], dtype=float)
    err = np.array([obs_dict[k] for k in req_errs], dtype=float)
    pivots = {k: _fixed_pivot_from_observable(k, obs_dict[k]) for k in req_obs}
    # pivots["log_tau_uv_rf"] = np.log10(500)
    # pivots["log_sigma_uv"]  = np.log10(0.2)
    pivots = np.array([pivots[k] for k in req_obs], dtype=float)
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
        # + alpha_agn * (log_sigma_hat0 - log_sigma_hat0_pivot)
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
    pidx = {k: i for i, k in enumerate(req_params)}
    eidx = {k: i for i, k in enumerate(req_errs)}

    alpha_agn = params_arr[pidx["alpha_agn"]]
    beta_agn = params_arr[pidx["beta_agn"]]
    
    log_sigma_uv_std_psd = err_arr[eidx["log_sigma_uv_std_psd"]]
    log_tau_uv_rf_std_psd = err_arr[eidx["log_tau_uv_rf_std_psd"]]
    log_sigma_uv_log_tau_uv_rf_cov_psd = err_arr[eidx["log_sigma_uv_log_tau_uv_rf_cov_psd"]]

    # gamma_agn   = params_arr[agn_model_pidx["gamma_agn"]]
    # dm_psf_correction_err = err_arr[agn_model_eidx["dm_psf_correction_err"]]

    # log_sigma_hat0_err = err_arr[agn_model_eidx["log_sigma_hat0_err"]]

    r = (
          (alpha_agn * log_sigma_uv_std_psd)**2
        + (beta_agn  * log_tau_uv_rf_std_psd)**2
        + 2 * alpha_agn * beta_agn * log_sigma_uv_log_tau_uv_rf_cov_psd
        #+ (gamma_agn * dm_psf_correction_err)**2
        # (log_sigma_hat0_err * alpha_agn)**2
    )
    if use_alpha_lambda_term:
        gamma_alpha_lambda = params_arr[pidx[AGN_ALPHA_LAMBDA_PARAM]]
        alpha_lambda_err = err_arr[eidx[AGN_ALPHA_LAMBDA_ERR]]
        r = r + (gamma_alpha_lambda * alpha_lambda_err) ** 2
    if use_eta_sigma_term:
        gamma_eta_sigma = params_arr[pidx[AGN_ETA_SIGMA_PARAM]]
        eta_sigma_err = err_arr[eidx[AGN_ETA_SIGMA_ERR]]
        r = r + (gamma_eta_sigma * eta_sigma_err) ** 2
    if check_negative:
        if np.any(r < 0):
            idx = np.where(r < 0)
            return np.full_like(r, -1), idx
        return np.sqrt(r), None
    else:
        return np.sqrt(r)


def get_model_params(
    cosmo_model,
    only_sna=False,
    use_alpha_lambda_term=False,
    use_eta_sigma_term=False,
    use_redshift_log_f_term=False,
):
    
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
        #("sigma_b",   (-1,  1)),

        ("H0",       (60.0, 80.0)),
        #("H0",       (67.37-0.54, 67.37+0.54)),  # Planck 2018 TT,TE,EE+lowE+lensing
        ("Om0",      (0.0, 1.0)),
        #("Om0",      (0.32, 0.34)),
        
    ])
    if not use_alpha_lambda_term:
        priors.pop(AGN_ALPHA_LAMBDA_PARAM)
    if not use_eta_sigma_term:
        priors.pop(AGN_ETA_SIGMA_PARAM)
    if not use_redshift_log_f_term:
        priors.pop(AGN_LOGF_Z_PARAM)

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
        "sigma_b": r"$\sigma_{\rm b}$",
        "H0": r"$H_0$",
        "Om0": r"$\Omega_{m,0}$",
        "w0": r"$w_0$",
        "wp": r"$w_p$",
        "wa": r"$w_a$"
    }
    model_labels_latex = [latex_labels.get(label, label) for label in model_labels]
    
    return priors, model_labels, model_labels_latex
