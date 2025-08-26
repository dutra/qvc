import numpy as np
from scipy.special import expit
from collections import OrderedDict


# --- Reference constants and pivot values ---
log_sigma_UV_pivot = -0.72 # TODO make this a parameter
log_tau_UV_RF_pivot = 2.65  # TODO make this a parameter
bwb_beta_pivot = 0.14
#M0_agn_offset = -5.179  # TODO make this a parameter
#z_agn_pivot = 1.2 # TODO make this a parameter
alpha_nu_pivot = -1

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
def M_model_agn(M0_agn, alpha_agn, beta_agn, gamma_agn, log_sigma_UV, log_tau_UV_RF, bwb_beta):
    return M0_agn + alpha_agn * (log_sigma_UV - log_sigma_UV_pivot) + beta_agn * (log_tau_UV_RF - log_tau_UV_RF_pivot)# + gamma_agn * (bwb_beta - bwb_beta_pivot)
    
def M_model_agn_err(M0_agn, alpha_agn, beta_agn, gamma_agn,
                    log_sigma_UV, log_sigma_UV_err, log_tau_UV_RF_err, bwb_beta_err):
    err = np.sqrt((alpha_agn * log_sigma_UV_err)**2 + (beta_agn * log_tau_UV_RF_err)**2)# + (gamma_agn * bwb_beta_err)**2)
    #mask = err > 5
    #print(f"Errors associated with log_sigma_UV_err > 5 mag: ", (alpha_agn * log_sigma_UV_err)[mask])
    #print(f"Errors associated with log_tau_UV_RF_err > 5 mag: ",  (beta_agn * log_tau_UV_RF_err)[mask])
    #print(f"Errors associated with bwb_beta_err > 5 mag: ", (gamma_agn * bwb_beta_err)[mask])
    return err


def M_model_SN(m_b_corr, host_logmass, M0_sn, gamma_sn, tau_Ms):
    """
    Brout+2022 SN model using standardized m_b_corr from Pantheon+SH0ES.
    """
    # Stable logistic correction in log-space, zero at logM = 10
    delta_host = gamma_sn * expit(-(host_logmass - 10) / tau_Ms) - gamma_sn / 2
    return m_b_corr - M0_sn + delta_host

def get_model_params(cosmo_model, only_sna=False):
    
    priors = OrderedDict([
        ("M0_sn",       (-22, -17)),    # SN absolute magnitude, MLE: ~-19.3

        ("M0_agn",   (-26.0, -16.0)),
        ("alpha_agn", (-5.0,  5.0)),
        ("beta_agn",  (-5.0,  5.0)),
        ("gamma_agn", (-5.0,  5.0)),
        ("log_f",     (-5.0,  0.3)),

        ("H0",       (60.0, 85.0)),
        ("Om0",      (0.05, 0.90)),
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