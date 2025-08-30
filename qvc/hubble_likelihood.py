
from dynesty.utils import resample_equal
from scipy.linalg import cho_solve
from astropy.cosmology import FlatwCDM, Flatw0waCDM, FlatLambdaCDM, FlatwpwaCDM
from scipy import stats
from scipy.signal import fftconvolve
import numpy as np

from hubble_model import get_model_params, M_model_agn, M_model_agn_err, agn_model_pack_params, agn_model_pack_obs

def completeness_loglike(m_model, mu_err, z, completeness2d, m_grid, sigma_completeness, tiny=1e-300):
    """
    m_model : array (N_obj,) model-predicted apparent magnitudes
    mu_err  : array (N_obj,) Gaussian sigma for each magnitude
    z       : array (N_obj,) redshifts
    m_grid  : array (N_grid,) magnitude grid (e.g., the map's mag_centers)
    """
    m_grid = np.asarray(m_grid)
    z      = np.asarray(z)
    m_model = np.asarray(m_model)
    mu_err  = np.asarray(mu_err)

    # Gaussian *pdf* over the real line, evaluated on m_grid
    # Do NOT renormalize row-wise over m_grid.
    sigma = np.maximum(mu_err, 1e-9)  # avoid zero-sigma
    pdf = stats.norm.pdf(m_grid[None, :],
        loc=m_model[:, None],
        scale=np.sqrt(sigma[:, None]**2 + sigma_completeness**2)) # If not adding scatter to mags_true
        #scale=sigma[:, None])

    # p_detect(m, z)
    p_det = completeness2d(m_grid[None, :], z[:, None])  # shape (N_obj, N_grid)
    wpdf = pdf * p_det

    # ∫ pdf(m) * p_det(m, z) dm  (outside-grid p_det=0 by construction)
    integrals = np.trapz(wpdf, m_grid, axis=1)
    integrals = np.clip(integrals, tiny, 1.0)            # numerical guard

    # Average
    m_integrals = np.trapz(wpdf * m_grid[None, :], m_grid, axis=1)
    m_integrals = np.clip(m_integrals, tiny, None)        # numerical guard (can be > 1; units=mag)
    dmi = m_integrals / integrals - m_model

    # return a 2xN float blob (consistent shape)
    blob = np.vstack([integrals.astype(float), dmi.astype(float)])
    return np.sum(np.log(integrals)), blob
    #return np.sum(np.log(integrals)), (integrals, dmi)

# --- Log-likelihood ---
def empty_blob(N_obj):
    # FIX: always return (2, N_obj) float array
    return np.zeros((2, N_obj), dtype=float)

def log_likelihood_pantheon_cephdist(params, pantheon_data, _sna_L, _sna_Lower, _sna_LogdetCov,
                                     cosmo, use_full_cov):
    # SN model: compute host mass correction
    #delta_host = params['gamma_sn'] * expit(-( _pantheon_data['HOST_LOGMASS'] - 10) / params['tau_Ms']) - params['gamma_sn'] / 2

    # Start with cosmological prediction
    sn_mu_model = cosmo.distmod(pantheon_data['zHD']).value

    # Apply Cepheid distances for calibrator hosts
    mask_calib = pantheon_data['IS_CALIBRATOR'] == 1
    sn_mu_model[mask_calib] = pantheon_data['CEPH_DIST'][mask_calib]

    # Residuals: observed standardized SN magnitude minus theoretical prediction
    res_snia = pantheon_data['m_b_corr'] - (sn_mu_model + params['M0_sn'])# - delta_host)
    # Compute main SN likelihood (with or without covariance)
    if use_full_cov:
        quad_form = res_snia.T @ cho_solve((_sna_L, _sna_Lower), res_snia)
        ll_snia = -0.5 * quad_form - 0.5 * _sna_LogdetCov - 0.5 * len(res_snia) * np.log(2 * np.pi)
    else:
        sigma = pantheon_data['MU_SH0ES_ERR_DIAG']
        ll_snia = np.sum(stats.norm.logpdf(res_snia, scale=sigma))
    return ll_snia

def log_likelihood_pantheon_mush0es(params, pantheon_data, _sna_L, _sna_Lower, _sna_LogdetCov, 
                                    cosmo, use_full_cov):
    # ---------------------------------------------------------------------
    # SN LIKELIHOOD — Approach with SH0ES-anchored, cosmology-agnostic data
    # ---------------------------------------------------------------------
    # Data vector: absolutely calibrated, Tripp-corrected SN distance moduli
    mu_sn_data = pantheon_data['MU_SH0ES']     # from Pantheon+SH0ES.dat
    z_sn       = pantheon_data['zHD']          # Hubble-diagram redshift

    # Model prediction for the shared Hubble diagram:
    # - If you run "cosmology-free", replace this with your spline/GP μ(z).
    # - If you're comparing to a cosmological model, keep cosmo.distmod below.
    mu_sn_model = cosmo.distmod(z_sn).value

    # Residuals (no M0, no host step, no bias terms — already included in MU_SH0ES)
    res_snia = mu_sn_data - mu_sn_model

    # Likelihood with the FULL SH0ES covariance (SN + Cepheid systematics)
    if use_full_cov:
        # These should be set at data-load time from Pantheon+SH0ES_STAT+SYS.cov:
        #   _sna_L, _sna_Lower, _sna_LogdetCov
        # where cho_solve((_sna_L, _sna_Lower), x) == (C^{-1} x)
        quad_form = res_snia.T @ cho_solve((_sna_L, _sna_Lower), res_snia)
        ll_snia   = -0.5 * (quad_form + _sna_LogdetCov + len(res_snia) * np.log(2 * np.pi))
    else:
        # Diagnostic/plotting only — README warns not to fit parameters with this
        sigma_diag = pantheon_data['MU_SH0ES_ERR_DIAG']
        ll_snia    = np.sum(stats.norm.logpdf(res_snia, scale=sigma_diag))
    return ll_snia

def log_likelihood(theta, *, agn_data, pantheon_data, 
                   _sna_L, _sna_Lower, _sna_LogdetCov,
                   cosmo_model, completeness_params,
                   only_sna=False, use_full_cov=False, use_mu_sh0es=False):
    priors, model_labels, model_labels_latex = get_model_params(cosmo_model, only_sna=only_sna)
    model_priors = {key: priors[key] for key in model_labels}
    params = dict(zip(model_labels, theta))

    # We'll need N_obj to create a fixed-shape blob for ALL branches
    N_obj = len(agn_data['z'])  # used for consistent blobs

    # Prior bounds
    for key, (low, high) in model_priors.items():
        if low > high:
            raise ValueError(f"For key {key} prior: Low {low} > high {high}")
        if not (low < params[key] < high):
            return -np.inf, empty_blob(N_obj)

    # Cosmology (you can ignore if your background is non-parametric; here it's kept for AGN and/or SN-vs-cosmo fits)
    if cosmo_model == 'FlatwCDM':
        cosmo = FlatwCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'])
    elif cosmo_model == 'Flatw0waCDM':
        cosmo = Flatw0waCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'], wa=params['wa'])
        if params['w0'] + params['wa'] >= 0:  # "no early DE" guard
            return -np.inf, empty_blob(N_obj)
    elif cosmo_model == 'FlatwpwaCDM':
        cosmo = FlatwpwaCDM(H0=params['H0'], Om0=params['Om0'], wp=params['wp'], wa=params['wa'], zp=z_pivot_agn)
    elif cosmo_model == 'FlatLambdaCDM':
        cosmo = FlatLambdaCDM(H0=params['H0'], Om0=params['Om0'])

    if use_mu_sh0es:
        ll_snia = log_likelihood_pantheon_mush0es(params, pantheon_data, 
                                                  _sna_L, _sna_Lower, _sna_LogdetCov,
                                                  cosmo, use_full_cov)
    else:
        ll_snia = log_likelihood_pantheon_cephdist(params, pantheon_data, 
                                                   _sna_L, _sna_Lower, _sna_LogdetCov,
                                                    cosmo, use_full_cov)
    
    if only_sna:
        return ll_snia, empty_blob(N_obj)

    # ------------------------
    # AGN likelihood
    # ------------------------
    z = agn_data['z']
    m_obs = agn_data['apparent_mag_2500']
    m_err = agn_data['apparent_mag_2500_err']

    agn_params_arr = agn_model_pack_params(params)
    agn_obs_arr, agn_err_arr, agn_pivot_arr = agn_model_pack_obs(agn_data)

    M_pred = M_model_agn(agn_params_arr, agn_obs_arr, agn_pivot_arr)
    M_pred_err = M_model_agn_err(agn_params_arr, agn_obs_arr, agn_err_arr, agn_pivot_arr)
    
    mu_err = np.sqrt(
        m_err**2 +
        M_pred_err**2 +
        (0.055 * z)**2 +
        np.exp(params['log_f'])**2
    )
    mu_pred = m_obs - M_pred 

    mu_cosmo = cosmo.distmod(z).value

    ll_agn = np.sum(stats.norm.logpdf(mu_pred - mu_cosmo, scale=mu_err))

    m_model = M_pred + mu_cosmo  # model-predicted magnitude

    ll_completeness = 0.0
    comp_blob = empty_blob(N_obj)
    if completeness_params is not None:
        completeness2d, mag_centers, _, _, _, completeness_scatter = completeness_params
        ll_completeness, comp_blob = completeness_loglike(
            m_model=m_model, mu_err=mu_err, z=z,
            completeness2d=completeness2d, m_grid=mag_centers,
            sigma_completeness=completeness_scatter
        )

    return ll_snia + ll_agn - ll_completeness, comp_blob
