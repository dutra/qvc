import numpy as np
from scipy.special import expit
from collections import OrderedDict


# --- Reference constants and pivot values ---
log_sigma_UV_pivot = -0.8 # TODO make this a parameter
log_tau_UV_RF_pivot = 2.9  # TODO make this a parameter
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
    return M0_agn + alpha_agn * (log_sigma_UV - log_sigma_UV_pivot) + beta_agn * (log_tau_UV_RF - log_tau_UV_RF_pivot) + gamma_agn * (bwb_beta - bwb_beta_pivot)
    
def M_model_agn_err(M0_agn, alpha_agn, gamma_agn, beta_agn,
                    log_sigma_UV, log_sigma_UV_err, log_tau_UV_RF_err, bwb_beta_err):
    return np.sqrt((alpha_agn * log_sigma_UV_err)**2 + (beta_agn * log_tau_UV_RF_err)**2 + (gamma_agn * bwb_beta_err)**2)


def M_model_SN(m_b_corr, host_logmass, M0_sn, gamma_sn, tau_Ms):
    """
    Brout+2022 SN model using standardized m_b_corr from Pantheon+SH0ES.
    """
    # Stable logistic correction in log-space, zero at logM = 10
    delta_host = gamma_sn * expit(-(host_logmass - 10) / tau_Ms) - gamma_sn / 2
    return m_b_corr - M0_sn + delta_host

def get_model_params(cosmo_model):

    priors = OrderedDict([
        ("gamma_sn",    (-0.1, 0.1)),     # Host mass step usually ~0.05
        ("tau_Ms",      (0.01, 0.2)),     # LOG Width of sigmoid transition usually ~0.043
        ("M0_sn",       (-20, -19)),    # SN absolute magnitude, MLE: ~-19.3
        ("M0_agn", (-24, -17)),         # M0_agn
        ("alpha_agn",   (-10, 10)),         # AGN sigma correlation
        ("beta_agn",    (-10, 10)),         # AGN tau correlation
        ("gamma_agn",    (-10, 10)),         
        ("log_f",       (-3, 0.5)),
        ("H0",          (65, 80)),
        #("Om0",         (0.32, 0.324)),
        ("Om0",         (0.2, 0.7)),
    ])

    # Select cosmological parameters based on model
    if cosmo_model == 'FlatLambdaCDM':
        pass
    elif cosmo_model == 'FlatwCDM':
        priors |= OrderedDict([
            ("w0",          (-10, 10))
        ])
    elif cosmo_model == 'Flatw0waCDM':
        priors |= OrderedDict([
            ("wp", (-20.0, 1.0)),   # covers phantom (<-1), Λ (-1), quintessence (> -1), and even w>0
            ("wa", (-20.0, 20.0))    # symmetric variation
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
        "log_sigma_UV_break": r"$\log_{10}\sigma_{\rm 0,break}$",
        "eta_A1_agn": r"$\eta_{A1, \rm AGN}$",
        "eta_A2_agn": r"$\eta_{A2, \rm AGN}$",
        "eta_break_agn": r"$\eta_{\rm break, AGN}$",
        "beta_agn": r"$\beta_{\rm AGN}$",
        "gamma_agn": r"$\gamma_{\rm AGN}$",
        "log_f": r"$\log f$",
        "H0": r"$H_0$",
        "Om0": r"$\Omega_{m,0}$",
        "wp": r"$w_p$",
        "wa": r"$w_a$"
    }
    model_labels_latex = [latex_labels.get(label, label) for label in model_labels]
    
    return priors, model_labels, model_labels_latex