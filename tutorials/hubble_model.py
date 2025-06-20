import numpy as np
from scipy.special import expit
from collections import OrderedDict


def K_corr(z, alpha_nu=-0.5):
    """K-correction for magnitude (assuming f_ν ~ ν^{alpha_nu})."""
    return -2.5 * (1 + alpha_nu) * np.log10(1 + z)

# --- Reference constants and pivot values ---
#-log_sigma_hat_pivot = -0.638 # TODO make this a parameter
log_tau_UV_RF_pivot = 3.057  # TODO make this a parameter
#M0_agn_offset = -5.179  # TODO make this a parameter
#z_agn_pivot = 1.2 # TODO make this a parameter

# --- AGN model ---
# def M_model_agn(M0_agn, alpha_agn, log_sigma_hat_UV):
#     return M0_agn - 26 + alpha_agn * 2 * (log_sigma_hat_UV - log_sigma_hat_pivot)

# def M_model_agn(M0_sn, delta_M_agn, alpha_agn, beta_agn, log_sigma_hat_UV, log_tau_UV_RF):
#     return M0_sn - M0_agn_offset + alpha_agn * 2 * (log_sigma_hat_UV - log_sigma_hat_pivot) + beta_agn * (log_tau_UV_RF - log_tau_UV_RF_pivot)


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
    delta = x - x_break
    term = (d2 - d1) / ds * np.log10(1 + 10**(ds * delta))
    offset = (d2 - d1) / ds * np.log10(2)  # value of the term when delta = 0
    return d1 * delta + term - offset

def M_model_agn(M0_agn, log_sigma_hat_sq_break, eta_A1_agn, eta_A2_agn, eta_breaK_agn, beta_agn, log_sigma_hat_UV, log_tau_UV_RF):
    """AGN model with broken power law in log_sigma_hat_UV."""
    # eta_A1_agn = 3.2
    # eta_A2_agn = 10
    # log_sigma_hat_sq_break = -1.1
    # M0_agn = -23.9

    bpl = broken_power_law(2*log_sigma_hat_UV, log_sigma_hat_sq_break, eta_A1_agn, eta_A2_agn, ds=eta_breaK_agn)
    return M0_agn + bpl + beta_agn * (log_tau_UV_RF - log_tau_UV_RF_pivot)

# keep this same(ish) signature as M_model_agn + x_err
def M_model_agn_err(M0_agn, log_sigma_hat_sq_break, eta_A1_agn, eta_A2_agn, eta_breaK_agn, beta_agn, log_sigma_hat_UV, log_sigma_hat_UV_err, log_tau_UV_RF_err):
    err_bpl = broken_power_law_err(2*log_sigma_hat_UV, 2*log_sigma_hat_UV_err, log_sigma_hat_sq_break, eta_A1_agn, eta_A2_agn, ds=eta_breaK_agn)    
    #return err_bpl
    return np.sqrt(err_bpl**2 + (beta_agn * log_tau_UV_RF_err)**2)

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
        ("delta_M0_agn", (-5, -2)),         # M0_agn = M0_sn + delta_M0, MLE: ~-5.179
        ("log_sigma_hat_sq_break", (-1.3, -0.5)), # AGN broken power law break point
        ("eta_A1_agn",  (-10, 10)),          # AGN broken power law slope 1
        ("eta_A2_agn",  (-10, 10)),          # AGN broken power law slope 2
        ("eta_break_agn", (0, 10)),      # AGN broken power law slope transition
        ("beta_agn",    (-3, 0)),         # AGN tau correlation
        ("log_f",       (-1, 0.5)),
        ("H0",          (65, 80)),
        #("Om0",         (0.32, 0.324)),
        ("Om0",         (0.25, 0.7)),
    ])

    # Select cosmological parameters based on model
    if cosmo_model == 'FlatLambdaCDM':
        pass
    elif cosmo_model == 'FlatwCDM':
        priors |= OrderedDict([
            ("w0",          (-2, -0.5))
        ])
    elif cosmo_model == 'Flatw0waCDM':
        priors |= OrderedDict([
            ("w0",          (-2, -0.5)),
            ("wa",          (-2, 0))
        ])

    else:
        raise ValueError("cosmo_model must be 'FlatwCDM' or 'Flatw0waCDM'")

    model_labels = list(priors.keys())
    
    # Map model_labels to LaTeX-compatible labels
    latex_labels = {
        "gamma_sn": r"$\gamma_{\rm SN}$",
        "tau_Ms": r"$\tau_{M_s}$",
        "M0_sn": r"$M^0_{\rm SN}$",
        "delta_M0_agn": r"$\Delta M^0_{\rm AGN}$",
        "log_sigma_hat_sq_break": r"$\log_{10}\hat{\sigma}^2_{\rm break}$",
        "eta_A1_agn": r"$\eta_{A1, \rm AGN}$",
        "eta_A2_agn": r"$\eta_{A2, \rm AGN}$",
        "eta_break_agn": r"$\eta_{\rm break, AGN}$",
        "beta_agn": r"$\beta_{\rm AGN}$",
        "log_f": r"$\log f$",
        "H0": r"$H_0$",
        "Om0": r"$\Omega_{m,0}$",
        "w0": r"$w_0$",
        "wa": r"$w_a$"
    }
    model_labels_latex = [latex_labels.get(label, label) for label in model_labels]
    
    return priors, model_labels, model_labels_latex