import numpy as np
from scipy.special import expit
from collections import OrderedDict

# --- ONE SOURCE OF TRUTH (orders) ---
agn_model_req_params = ("M0_agn", "alpha_agn", "beta_agn", #"gamma_agn",
                        )
agn_model_req_obs     = ("log_sigma_UV", "log_tau_UV_RF",) #"log_conti_a_0",)
agn_model_req_errs   = (
                        #"log_sigma_UV_err", "log_tau_UV_RF_err", "cov_log_sigma_UV_log_tau_UV_RF",
                        "log_sigma_UV_std_psd", "log_tau_UV_RF_std_psd", "log_sigma_UV_log_tau_UV_RF_cov_psd"
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
    #pivots["log_tau_UV_RF"] = np.log10(500)
    #pivots["log_sigma_UV"]  = np.log10(0.3)
    pivots = np.array([pivots[k] for k in agn_model_req_obs], dtype=float)
    return obs, err, pivots



def M_model_agn(params_arr, obs_arr, pivots_array):
    M0_agn    = params_arr[agn_model_pidx["M0_agn"]]
    alpha_agn = params_arr[agn_model_pidx["alpha_agn"]]
    beta_agn  = params_arr[agn_model_pidx["beta_agn"]]

    log_sigma_UV  = obs_arr[agn_model_oidx["log_sigma_UV"]]
    log_tau_UV_RF = obs_arr[agn_model_oidx["log_tau_UV_RF"]]

    log_sigma_UV_pivot  = pivots_array[agn_model_oidx["log_sigma_UV"]]
    log_tau_UV_RF_pivot = pivots_array[agn_model_oidx["log_tau_UV_RF"]]

    # --- new ebv correction params ---
    # gamma_agn = params_arr[agn_model_pidx["gamma_agn"]]
    # log_conti_a_0 = obs_arr[agn_model_oidx["log_conti_a_0"]]
    # log_conti_a_0_pivot = pivots_array[agn_model_oidx["log_conti_a_0"]]

    return (
        M0_agn
        + alpha_agn * (log_sigma_UV - log_sigma_UV_pivot)
        + beta_agn  * (log_tau_UV_RF - log_tau_UV_RF_pivot)
        #+ gamma_agn * (log_conti_a_0 - log_conti_a_0_pivot)
    )

def M_model_agn_err(params_arr, obs_arr, err_arr, pivots_array, check_negative=False):
    alpha_agn   = params_arr[agn_model_pidx["alpha_agn"]]
    beta_agn    = params_arr[agn_model_pidx["beta_agn"]]

    # log_sigma_UV_err  = err_arr[agn_model_eidx["log_sigma_UV_err"]]
    # log_tau_UV_RF_err = err_arr[agn_model_eidx["log_tau_UV_RF_err"]]
    # cov_log_sigma_tau = err_arr[agn_model_eidx["cov_log_sigma_UV_log_tau_UV_RF"]]

    log_sigma_UV_err  = err_arr[agn_model_eidx["log_sigma_UV_std_psd"]]
    log_tau_UV_RF_err = err_arr[agn_model_eidx["log_tau_UV_RF_std_psd"]]
    cov_log_sigma_tau = err_arr[agn_model_eidx["log_sigma_UV_log_tau_UV_RF_cov_psd"]]

    r = ((alpha_agn * log_sigma_UV_err)**2
        + (beta_agn  * log_tau_UV_RF_err)**2
        + 2 * alpha_agn * beta_agn * cov_log_sigma_tau
    )
    if check_negative:
        if np.any(r < 0):
            idx = np.where(r < 0)
            return np.full_like(r, -1), idx
        return np.sqrt(r), None
    else:
        return np.sqrt(r)

# def _sigmoid(x, A, k, x0):
#     return A / (1.0 + np.exp(-k * (x - x0)))

# def _anchored_sigmoid(x, A, k, x0, x_piv):
#     # anchored so effect is exactly zero at x_piv
#     s  = 1.0 / (1.0 + np.exp(-k * (x      - x0)))
#     sp = 1.0 / (1.0 + np.exp(-k * (x_piv  - x0)))
#     return A * (s - sp)

# def M_model_agn(params_arr, obs_arr, pivots_array):
#     M0_agn    = params_arr[agn_model_pidx["M0_agn"]]
#     alpha_agn = params_arr[agn_model_pidx["alpha_agn"]]
#     beta_agn  = params_arr[agn_model_pidx["beta_agn"]]

#     # --- new reddening correction params ---
#     A_red  = params_arr[agn_model_pidx["A_red"]]   # expect negative (e.g. ~ -2)
#     k_red  = params_arr[agn_model_pidx["k_red"]]   # >0 (e.g. ~ 1–3 per dex)
#     x0_red = params_arr[agn_model_pidx["x0_red"]]  # bend near where trend starts

#     log_sigma_UV  = obs_arr[agn_model_oidx["log_sigma_UV"]]
#     log_tau_UV_RF = obs_arr[agn_model_oidx["log_tau_UV_RF"]]
#     x_red         = obs_arr[agn_model_oidx["log_reddening_integral"]]  # NEW

#     log_sigma_UV_pivot  = pivots_array[agn_model_oidx["log_sigma_UV"]]
#     log_tau_UV_RF_pivot = pivots_array[agn_model_oidx["log_tau_UV_RF"]]
#     x_red_pivot         = pivots_array[agn_model_oidx["log_reddening_integral"]]  # NEW

#     # anchored sigmoid term (zero at small-reddening pivot)
#     red_term = _anchored_sigmoid(x_red, A_red, np.abs(k_red) + 1e-6, x0_red, x_red_pivot)

#     return (
#         M0_agn
#         + alpha_agn * (log_sigma_UV - log_sigma_UV_pivot)
#         + beta_agn  * (log_tau_UV_RF - log_tau_UV_RF_pivot)
#         + red_term
#     )




# def broken_power_law_err(x, x_err, x_break, d1, d2, ds):
#     u = ds * (x - x_break)
#     with np.errstate(over='ignore', under='ignore'):
#         ten_u = np.power(10, u)
#     ten_u = np.clip(ten_u, 1e-10, 1e10)  # prevent infs

#     df_dx = d1 + (d2 - d1) * ten_u / (1 + ten_u)
#     return np.abs(df_dx) * x_err

# def broken_power_law(x, x_break, d1, d2, ds):
#     """Broken power law defined to be zero at x_break.
#     That decorrelates d1, d2 from M0_agn.
#     """
#     #print(f"broken_power_law: x={x}, x_break={x_break}, d1={d1}, d2={d2}, ds={ds}")
#     delta = x - x_break
#     term = (d2 - d1) / ds * np.log10(1 + 10**(ds * delta))
#     offset = (d2 - d1) / ds * np.log10(2)  # value of the term when delta = 0
#     return d1 * delta + term - offset

# # Broken power law model
# def M_model_agn(M0_agn, log_sigma_UV_break, eta_A1_agn, eta_A2_agn, eta_break_agn, beta_agn, log_sigma_UV, log_tau_UV_RF):
#     """AGN model with broken power law in log_sigma_UV."""
#     bpl = broken_power_law(log_sigma_UV, log_sigma_UV_break, eta_A1_agn, eta_A2_agn, ds=eta_break_agn)
#     return M0_agn + bpl + beta_agn * (log_tau_UV_RF - log_tau_UV_RF_pivot)

# # keep this same(ish) signature as M_model_agn + x_err
# def M_model_agn_err(M0_agn, log_sigma_UV_break, eta_A1_agn, eta_A2_agn, eta_break_agn, beta_agn,
#                     log_sigma_UV, log_sigma_UV_err, log_tau_UV_RF_err):
#     err_bpl = broken_power_law_err(log_sigma_UV, log_sigma_UV_err, log_sigma_UV_break, eta_A1_agn, eta_A2_agn, ds=eta_break_agn)    
#     return np.sqrt(err_bpl**2 + (beta_agn * log_tau_UV_RF_err)**2)


def get_model_params(cosmo_model, only_sna=False):
    
    priors = OrderedDict([
        ("M0_sn",       (-20, -18)),    # SN absolute magnitude, MLE: ~-19.3

        ("M0_agn",   (-26.0, -18.0)),
        ("alpha_agn", (0.0,  20.0)),
        ("beta_agn",  (-20.0,  0.0)),
        
        #("gamma_agn", (-50.0, 50.0)),
        # ("A_red",    (-5.0,  0.0)),   # expect negative (e.g. ~ -2)
        # ("k_red",    (0.1,  5.0)),    # >0 (e.g. ~ 1–3 per dex)
        # ("x0_red",   (-5.0,  5.0)),    # bend near where trend starts

        ("log_f",     (-5.0,  3.0)),

        ("H0",       (60.0, 80.0)),
        ("Om0",      (0.0, 1.0)),
    ])

    # Select cosmological parameters based on model
    if cosmo_model == 'FlatLambdaCDM':
        pass
    elif cosmo_model == 'FlatwCDM':
        priors |= OrderedDict([
            ("w0",          (-10.0, 1.0))
        ])
    elif cosmo_model == 'Flatw0waCDM':
        priors |= OrderedDict([
            ("w0", (-10.0, 1.0)),   # covers phantom (<-1), Λ (-1), quintessence (> -1), and even w>0
            ("wa", (-50, 20))    # symmetric variation
        ])
    elif cosmo_model == 'FlatwpwaCDM':
        priors |= OrderedDict([
            ("wp", (-10.0, 1.0)),   # covers phantom (<-1), Λ (-1), quintessence (> -1), and even w>0
            ("wa", (-50, 20))    # symmetric variation
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
        "H0": r"$H_0$",
        "Om0": r"$\Omega_{m,0}$",
        "w0": r"$w_0$",
        "wp": r"$w_p$",
        "wa": r"$w_a$"
    }
    model_labels_latex = [latex_labels.get(label, label) for label in model_labels]
    
    return priors, model_labels, model_labels_latex