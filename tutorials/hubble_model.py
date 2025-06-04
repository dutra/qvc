import numpy as np
from scipy.special import expit


def K_corr(z, alpha_nu=-0.5):
    """K-correction for magnitude (assuming f_ν ~ ν^{alpha_nu})."""
    return -2.5 * (1 + alpha_nu) * np.log10(1 + z)

# --- Reference constants and pivot values ---
log_sigma_hat_pivot = -0.638 # TODO make this a parameter
log_tau_UV_RF_pivot = 3.110  # TODO make this a parameter
M0_agn_offset = 5.162  # TODO make this a parameter
z_agn_pivot = 2 # TODO make this a parameter

# --- AGN model ---
# def M_model_agn(M0_agn, alpha_agn, log_sigma_hat_UV):
#     return M0_agn - 26 + alpha_agn * 2 * (log_sigma_hat_UV - log_sigma_hat_pivot)

def M_model_agn(M0_sn, delta_M_agn, alpha_agn, beta_agn, log_sigma_hat_UV, log_tau_UV_RF):
    return M0_sn - M0_agn_offset + alpha_agn * 2 * (log_sigma_hat_UV - log_sigma_hat_pivot) + beta_agn * (log_tau_UV_RF - log_tau_UV_RF_pivot)

def M_model_SN(m_b, x1, c, bias, host_logmass, alpha_sn, beta_sn, M0_sn, gamma_sn, tau_Ms):
    """Brout+ 2022 SN model with stable host correction."""
    # Stable logistic function: delta_host = gamma / (1 + exp(...)) = gamma * sigmoid(...)
    S = 1e10
    host_mass = 10**host_logmass
    delta_host = gamma_sn * expit(-(host_mass - S) / tau_Ms) - gamma_sn/2
    return m_b + alpha_sn * x1 - beta_sn * c - M0_sn + delta_host - bias # bias may be already included in m_b


def get_model_params(cosmo_model):
    # Select cosmological parameters based on model
    if cosmo_model == 'FlatLambdaCDM':
        cosmo_params = ['H0', 'Om0']
    elif cosmo_model == 'FlatwCDM':
        cosmo_params = ['H0', 'Om0', 'w0']
    elif cosmo_model == 'Flatw0waCDM':
        cosmo_params = ['H0', 'Om0', 'w0', 'wa']
    else:
        raise ValueError("cosmo_model must be 'FlatwCDM' or 'Flatw0waCDM'")
    # Model parameters: AGN correlation + SN calibration + cosmology
    model_labels = ['alpha_sn', 'beta_sn', 'gamma_sn', 'tau_Ms', 'M0_sn', 
                    'delta_M_agn', 'alpha_agn',
                    'beta_agn', 'log_f'] + cosmo_params

    # --- Priors ---
    priors = {
        "alpha_sn":    (-.2, .2),        # SN stretch coefficient
        "beta_sn":     (0.6, 1.2),         # SN color-luminosity
        "gamma_sn":    (-0.1, 0.1),    # Host mass step
        "tau_Ms":   (0.65, 0.8),        # Host mass transition
        "M0_sn":    (-21, -17),
        "delta_M_agn": (-2, 2),  # AGN small offset for individual AGN, making this larger creates weird degeneracies
        "alpha_agn": (0, 5),         # AGN variability correlation
        "beta_agn":  (-5, 0),        # AGN variability correlation
        "log_f":    (-3, .5),
        "H0":       (60, 80),
        "Om0":      (0.2, 0.4),
        "w0":       (-3, 0),
        "wa":       (-5, 0)
    }
    return priors, model_labels