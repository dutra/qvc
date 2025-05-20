import numpy as np
from scipy.special import expit
from scipy import stats
from scipy.signal import fftconvolve
from astropy.cosmology import FlatwCDM, Flatw0waCDM


def K_corr(z, alpha_nu=-0.5):
    """K-correction for magnitude (assuming f_ν ~ ν^{alpha_nu})."""
    return -2.5 * (1 + alpha_nu) * np.log10(1 + z)

# --- Reference constants and pivot values ---
sigma_pivot = -0.8
tau_pivot = 2.0

# --- AGN model ---
def M_model_agn(M0_agn, alpha_agn, log_sigma_UV, log_tau_UV_RF):
    return M0_agn + alpha_agn * 2 * (log_sigma_UV - sigma_pivot) - (log_tau_UV_RF - tau_pivot)

# --- SN model (Brout+ 2022 Eq. 1 and 2) ---
# SN calibration: anchor absolute magnitude from SH0ES (Riess et al. 2022:contentReference[oaicite:0]{index=0})
# M_anchor = -19.253  # SH0ES-calibrated SN Ia absolute magnitude (reference value)

def M_model_SN(m_b, x1, c, bias, host_logmass, alpha, beta, M0, gamma, tau_Ms):
    """Brout+ 2022 SN model with stable host correction."""
    # Stable logistic function: delta_host = gamma / (1 + exp(...)) = gamma * sigmoid(...)
    S = 1e10
    host_mass = 10**host_logmass
    delta_host = gamma * expit(-(host_mass - S) / tau_Ms) - gamma/2
    return m_b + alpha * x1 - beta * c - M0 - bias + delta_host


def get_model_params(cosmo_model):
    # Select cosmological parameters based on model
    if cosmo_model == 'FlatwCDM':
        cosmo_params = ['H0', 'Om0', 'w0']
    elif cosmo_model == 'Flatw0waCDM':
        cosmo_params = ['H0', 'Om0', 'w0', 'wa']
    else:
        raise ValueError("cosmo_model must be 'FlatwCDM' or 'Flatw0waCDM'")
    # Model parameters: AGN correlation + SN calibration + cosmology
    model_labels = ['alpha', 'beta', 'gamma', 'tau_Ms', 'M0_sn', 'alpha_agn', 'M0_agn', 'log_f'] + cosmo_params

    # --- Priors ---
    priors = {
        "alpha":    (-1, 1),        # SN stretch coefficient
        "beta":     (0, 5),         # SN color-luminosity
        "gamma":    (-0.5, 0.5),    # Host mass step
        "tau_Ms":   (0.5, 1.5),        # Host mass transition
        "M0_sn":    (-21, -17),
        "alpha_agn": (-10, 10),     # AGN variability correlation
        "M0_agn":   (-30, -10),
        "log_f":    (-3, 1),
        "H0":       (60, 80),
        "Om0":      (0.2, 0.7),
        "w0":       (-3, 0),
        "wa":       (-3, 3)
    }
    return priors, model_labels