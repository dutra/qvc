import numpy as np
from scipy.special import expit
from collections import OrderedDict

# --- ONE SOURCE OF TRUTH (orders) ---
agn_model_req_params = ("M0_agn", "alpha_agn", "beta_agn", 
                        #"gamma_agn",
                          #"A", "k", "x0"
                        )
agn_model_req_obs     = (
                        "log_sigma_hat0",
                        "log_sigma_uv", "log_tau_uv_rf",
                        #'dm_psf_correction',
                        #"PL_slope_blue"
                         )
agn_model_req_errs   = (
                        "log_sigma_hat0_err",
                        "log_sigma_uv_std_psd", "log_tau_uv_rf_std_psd", "log_sigma_uv_log_tau_uv_rf_cov_psd",
                        #"dm_psf_correction_err",
                        )

# Build index maps once (module import time)
agn_model_pidx = {k:i for i,k in enumerate(agn_model_req_params)}
agn_model_oidx = {k:i for i,k in enumerate(agn_model_req_obs)}
agn_model_eidx = {k:i for i,k in enumerate(agn_model_req_errs)}

def _require(keys, provided, where):
    miss = set(keys) - set(provided)
    if miss:
        raise KeyError(f"Missing {where}: {sorted(miss)}")

def agn_model_pack_params(params_dict):
    _require(agn_model_req_params, params_dict, "params")
    params = np.array([params_dict[k] for k in agn_model_req_params], dtype=float)
    return params


def agn_model_pack_obs(obs_dict):
    _require(agn_model_req_obs,  obs_dict, "observables")
    _require(agn_model_req_errs, obs_dict, "errors")
    obs = np.array([obs_dict[k] for k in agn_model_req_obs],  dtype=float)
    err = np.array([obs_dict[k] for k in agn_model_req_errs], dtype=float)
    pivots = {k: float(np.mean(obs_dict[k])) for k in agn_model_req_obs}
    # pivots["log_tau_uv_rf"] = np.log10(500)
    # pivots["log_sigma_uv"]  = np.log10(0.2)
    pivots = np.array([pivots[k] for k in agn_model_req_obs], dtype=float)
    return obs, err, pivots

def hinge(x, a, b, x0):
    return a + b * np.maximum(0.0, x - x0)

def logistic(x, A, k, x0):
     return A * expit(k*(x - x0))

def M_model_agn(params_arr, obs_arr, pivots_array):
    M0_agn    = params_arr[agn_model_pidx["M0_agn"]]
    alpha_agn = params_arr[agn_model_pidx["alpha_agn"]]
    beta_agn  = params_arr[agn_model_pidx["beta_agn"]]
    #gamma_agn = params_arr[agn_model_pidx["gamma_agn"]]

    log_sigma_uv  = obs_arr[agn_model_oidx["log_sigma_uv"]]
    log_tau_uv_rf = obs_arr[agn_model_oidx["log_tau_uv_rf"]]
    log_sigma_uv_pivot  = pivots_array[agn_model_oidx["log_sigma_uv"]]
    log_tau_uv_rf_pivot = pivots_array[agn_model_oidx["log_tau_uv_rf"]]

    #dm_psf_correction = obs_arr[agn_model_oidx["dm_psf_correction"]]
    #dm_psf_correction_pivot = pivots_array[agn_model_oidx["dm_psf_correction"]]
    # PL_slope_blue = obs_arr[agn_model_oidx["PL_slope_blue"]]

    # A = params_arr[agn_model_pidx["A"]]
    # k = params_arr[agn_model_pidx["k"]]
    # x0 = params_arr[agn_model_pidx["x0"]]

    return (
        M0_agn
        + alpha_agn * (log_sigma_uv - log_sigma_uv_pivot)
        + beta_agn  * (log_tau_uv_rf - log_tau_uv_rf_pivot)
        #+ gamma_agn * (dm_psf_correction - dm_psf_correction_pivot)
        #+ logistic(PL_slope_blue, A, k, x0)
        # + alpha_agn * (log_sigma_hat0 - log_sigma_hat0_pivot)
    )

def M_model_agn_err(params_arr, obs_arr, err_arr, pivots_array, check_negative=False):
    alpha_agn   = params_arr[agn_model_pidx["alpha_agn"]]
    beta_agn    = params_arr[agn_model_pidx["beta_agn"]]
    
    log_sigma_uv_std_psd  = err_arr[agn_model_eidx["log_sigma_uv_std_psd"]]
    log_tau_uv_rf_std_psd = err_arr[agn_model_eidx["log_tau_uv_rf_std_psd"]]
    log_sigma_uv_log_tau_uv_rf_cov_psd = err_arr[agn_model_eidx["log_sigma_uv_log_tau_uv_rf_cov_psd"]]

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
    if check_negative:
        if np.any(r < 0):
            idx = np.where(r < 0)
            return np.full_like(r, -1), idx
        return np.sqrt(r), None
    else:
        return np.sqrt(r)


def get_model_params(cosmo_model, only_sna=False):
    
    priors = OrderedDict([
        ("M0_sn",       (-20, -18)),    # SN absolute magnitude, MLE: ~-19.3

        ("M0_agn",   (-26.0, -18.0)),
        ("alpha_agn", (-20,  20.0)),
        ("beta_agn",  (-20.0,  20.0)),
        
        # ("A",    (-5.0,  5.0)),
        # ("k",    (0,  20.0)),
        # ("x0",   (-2.0,  1.0)),

        #("gamma_agn", (-100.0, 100.0)),
        # ("A_red",    (-5.0,  0.0)),   # expect negative (e.g. ~ -2)
        # ("k_red",    (0.1,  5.0)),    # >0 (e.g. ~ 1–3 per dex)
        # ("x0_red",   (-5.0,  5.0)),    # bend near where trend starts

        ("log_f",     (-5.0,  3.0)),
        #("sigma_b",   (-1,  1)),

        ("H0",       (60.0, 80.0)),
        #("H0",       (67.37-0.54, 67.37+0.54)),  # Planck 2018 TT,TE,EE+lowE+lensing
        ("Om0",      (0.0, 1.0)),
        #("Om0",      (0.32, 0.34)),
        
    ])

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
        "gamma_agn": r"$\gamma_{\rm AGN}$",
        "log_f": r"$\log f$",
        "sigma_b": r"$\sigma_{\rm b}$",
        "H0": r"$H_0$",
        "Om0": r"$\Omega_{m,0}$",
        "w0": r"$w_0$",
        "wp": r"$w_p$",
        "wa": r"$w_a$"
    }
    model_labels_latex = [latex_labels.get(label, label) for label in model_labels]
    
    return priors, model_labels, model_labels_latex
