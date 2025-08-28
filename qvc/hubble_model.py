import numpy as np
from scipy.special import expit
from collections import OrderedDict


# --- Reference constants and pivot values ---
log_sigma_UV_pivot = -0.72 # TODO make this a parameter
log_tau_UV_RF_pivot = 2.65  # TODO make this a parameter
bwb_beta_pivot = 0.14
#M0_agn_offset = -5.179  # TODO make this a parameter
#z_agn_pivot = 1.2 # TODO make this a parameter
#alpha_nu_pivot = -0.86
alpha_nu_pivot = 0

def broken_power_law_err(x, x_err, x_break, d1, d2, ds):
    u = ds * (x - x_break)
    with np.errstate(over='ignore', under='ignore'):
        ten_u = np.power(10, u)
    ten_u = np.clip(ten_u, 1e-10, 1e10)  # prevent infs

    df_dx = d1 + (d2 - d1) * ten_u / (1 + ten_u)
    return np.abs(df_dx) * x_err

def broken_power_law(x, x_break, d1, d2, ds):
    """Broken power law defined to be zero at x_break.
    That decorrelates d1, d2 from M0_agn.
    """
    #print(f"broken_power_law: x={x}, x_break={x_break}, d1={d1}, d2={d2}, ds={ds}")
    delta = x - x_break
    term = (d2 - d1) / ds * np.log10(1 + 10**(ds * delta))
    offset = (d2 - d1) / ds * np.log10(2)  # value of the term when delta = 0
    return d1 * delta + term - offset

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

# # Linear model
k_logistic = 1.0  # steepness of the logistic function
def M_model_agn(M0_agn, alpha_agn, beta_agn, gamma_agn, log_sigma_UV, log_tau_UV_RF, alpha_nu):
    logistic_alpha_nu = expit(k_logistic * (alpha_nu - alpha_nu_pivot)) - 0.5
    return (M0_agn + alpha_agn * (log_sigma_UV - log_sigma_UV_pivot)
            + beta_agn * (log_tau_UV_RF - log_tau_UV_RF_pivot)
            + gamma_agn * logistic_alpha_nu
    )
            # gamma_agn * (alpha_nu - alpha_nu_pivot)
    
def M_model_agn_err(M0_agn, alpha_agn, beta_agn, gamma_agn,
                    log_sigma_UV, log_sigma_UV_err, log_tau_UV_RF_err, alpha_nu, alpha_nu_err):
    # Derivative of expit(x) = expit(x) * (1 - expit(x))
    x = k_logistic * (alpha_nu - alpha_nu_pivot)
    logistic_alpha_nu = expit(x)
    logistic_derivative = k_logistic * logistic_alpha_nu * (1 - logistic_alpha_nu)
    d_alpha_nu_term = gamma_agn * logistic_derivative
    err = np.sqrt((alpha_agn * log_sigma_UV_err)**2 + (beta_agn * log_tau_UV_RF_err)**2 
                  + (d_alpha_nu_term * alpha_nu_err)**2
    )
                  #+ (gamma_agn * alpha_nu_err)**2)
    #mask = err > 5
    #print(f"Errors associated with log_sigma_UV_err > 5 mag: ", (alpha_agn * log_sigma_UV_err)[mask])
    #print(f"Errors associated with log_tau_UV_RF_err > 5 mag: ",  (beta_agn * log_tau_UV_RF_err)[mask])
    #print(f"Errors associated with bwb_beta_err > 5 mag: ", (gamma_agn * bwb_beta_err)[mask])
    return err


def get_model_params(cosmo_model, only_sna=False):
    
    priors = OrderedDict([
        ("M0_sn",       (-21, -18)),    # SN absolute magnitude, MLE: ~-19.3

        ("M0_agn",   (-24.0, -17.0)),
        ("alpha_agn", (-10.0,  10.0)),
        ("beta_agn",  (-5.0,  5.0)),
        ("gamma_agn", (-10.0,  10.0)),
        ("log_f",     (-5.0,  0.3)),

        ("H0",       (60.0, 80.0)),
        ("Om0",      (0.1, 0.9)),
    ])

    # Select cosmological parameters based on model
    if cosmo_model == 'FlatLambdaCDM':
        pass
    elif cosmo_model == 'FlatwCDM':
        priors |= OrderedDict([
            ("w0",          (-10, 0))
        ])
    elif cosmo_model == 'Flatw0waCDM':
        priors |= OrderedDict([
            ("w0", (-10.0, 5.0)),   # covers phantom (<-1), Λ (-1), quintessence (> -1), and even w>0
            ("wa", (-100, 10))    # symmetric variation
        ])

    else:
        raise ValueError("cosmo_model must be 'FlatwCDM' or 'Flatw0waCDM'")

    model_labels = list(priors.keys())
    
    # Map model_labels to LaTeX-compatible labels
    latex_labels = {
        "gamma_sn": r"$\gamma_{\rm SN}$ (mag)",
        "tau_Ms": r"$\tau_{M_s}$",
        "M0_sn": r"$M^0_{\rm SN}$ (mag)",
        "M0_agn": r"$M^0_{\rm AGN}$ (mag)",
        "alpha_agn": r"$\alpha_{\rm AGN}$ (mag/dex)",
        "beta_agn": r"$\beta_{\rm AGN}$ (mag/dex)",
        "gamma_agn": r"$\gamma_{\rm AGN}$ (mag/dex)",
        "log_f": r"$\log f$",
        "H0": r"$H_0$ (km\,s$^{-1}$\,{\rm Mpc}^{-1})",
        "Om0": r"$\Omega_{m,0}$",
        "w0": r"$w_0$",
        "wp": r"$w_p$",
        "wa": r"$w_a$"
    }
    model_labels_latex = [latex_labels.get(label, label) for label in model_labels]
    
    return priors, model_labels, model_labels_latex