
from scipy.linalg import cho_solve
from astropy.cosmology import FlatwCDM, Flatw0waCDM, FlatLambdaCDM, FlatwpwaCDM
from scipy import stats
import numpy as np

from hubble_model import get_model_params, M_model_agn, M_model_agn_err, agn_model_pack_params, agn_model_pack_obs

def completeness_loglike(m_obs, m_obs_err, m_model, mu_err, z, completeness2d, m_grid, sigma_completeness=0.0, tiny=1e-300):
    """
    Compute log-likelihood contribution from magnitude-limited sample selection.

    m_obs   : array (N_obj,) observed apparent magnitudes
    m_model : array (N_obj,) model-predicted apparent magnitudes
    mu_err  : array (N_obj,) Gaussian sigma for each magnitude
    z       : array (N_obj,) redshifts
    m_grid  : array (N_grid,) magnitude grid (e.g., the map's mag_centers)
    sigma_completeness : float, additional uncertainty in completeness
    """
    m_grid = np.asarray(m_grid)
    z      = np.asarray(z)
    m_model = np.asarray(m_model)
    mu_err  = np.asarray(mu_err)

    # shared pieces
    sig = np.sqrt(mu_err[:, None]**2 + float(sigma_completeness)**2)   # (N,1)

    # completeness on grid for each object
    p_det = completeness2d(m_grid[None, :], z[:, None])                 # (N,G)

    # Model-centered selection factor: Z_i
    pdf_model = stats.norm.pdf(m_grid[None, :], loc=m_model[:, None], scale=sig)  # (N,G)
    wpdf_model = pdf_model * p_det

    Z = np.trapz(wpdf_model, m_grid, axis=1)                            # (N,)
    Z = np.clip(Z, tiny, None)                                          # guard denom

    # Debias for plotting (the scatter is mostly in M, not Malmquist)
    m_Z = np.trapz(wpdf_model * m_grid[None, :], m_grid, axis=1)
    m_Z = np.clip(m_Z, tiny, None)
    E = m_Z / Z
    dmi_obs = E - m_model
    dmi_obs[z<0.2] = 0.0

    blob = np.vstack([Z.astype(float), dmi_obs.astype(float)])
 
    # ---- NEW: mask out the specified redshift range ----
    mask_in = (0.44 < z) & (z < 3.16)
    #mask_in = z < 100
    loglike_terms = np.log(Z)
    loglike_terms[~mask_in] = 0.0  # zero out the contribution

    return np.sum(loglike_terms), blob

# --- Log-likelihood ---
def empty_blob(N_obj):
    # FIX: always return (2, N_obj) float array
    return np.zeros((2, N_obj), dtype=float)

def log_likelihood_pantheon_cephdist(params, pantheon_data, _sna_L, _sna_Lower, _sna_LogdetCov,
                                     cosmo, use_full_cov):
    """
    Uses only SNe with (zHD > 0.01) OR IS_CALIBRATOR == True.
    For calibrators, replaces cosmological μ with Cepheid host distances.
    """
    # --- selection mask ---
    # also applied when loading pantheon data in hubble_utils.py
    is_calib_bool = np.asarray(pantheon_data['IS_CALIBRATOR'], dtype=bool)
    mask = (pantheon_data['zHD'] > 0.01) | is_calib_bool

    # --- subset data ---
    zHD = pantheon_data['zHD'][mask]
    m_b_corr = pantheon_data['m_b_corr'][mask]
    is_calib_sel = is_calib_bool[mask]

    # --- cosmological / Cepheid μ ---
    sn_mu_model = cosmo.distmod(zHD).value
    if np.any(is_calib_sel):
        sn_mu_model[is_calib_sel] = pantheon_data['CEPH_DIST'][mask][is_calib_sel]

    # --- residuals ---
    res_snia = m_b_corr - (sn_mu_model + params['M0_sn'])

    # --- likelihood ---
    if use_full_cov:
        # Expect Cholesky & logdet for the *masked* subset
        n = res_snia.size
        if _sna_L is None or _sna_LogdetCov is None:
            raise ValueError("Full-cov mode requires _sna_L and _sna_LogdetCov for the masked subset.")
        # basic dimension check to catch mismatches early
        if _sna_L.shape[0] != n or _sna_L.shape[1] != n:
            raise ValueError(
                f"Covariance Cholesky shape {_sna_L.shape} does not match masked data length {n}. "
                "Pass the covariance for the same mask."
            )
        quad_form = res_snia.T @ cho_solve((_sna_L, _sna_Lower), res_snia)
        ll_snia = -0.5 * quad_form - 0.5 * _sna_LogdetCov - 0.5 * n * np.log(2 * np.pi)
    else:
        sigma = pantheon_data['MU_SH0ES_ERR_DIAG'][mask]
        ll_snia = np.sum(stats.norm.logpdf(res_snia, scale=sigma))

    return ll_snia


def log_likelihood(theta, *, agn_data, pantheon_data, 
                   _sna_L, _sna_Lower, _sna_LogdetCov,
                   cosmo_model, completeness_params,
                   z_pivot_agn,
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
        # if params['w0'] + params['wa'] >= 0:  # "no early DE" guard
        #     return -np.inf, empty_blob(N_obj)
    elif cosmo_model == 'FlatwpwaCDM':
        cosmo = FlatwpwaCDM(H0=params['H0'], Om0=params['Om0'], wp=params['wp'], wa=params['wa'], zp=z_pivot_agn)
    elif cosmo_model == 'FlatLambdaCDM':
        cosmo = FlatLambdaCDM(H0=params['H0'], Om0=params['Om0'])

    ll_snia = log_likelihood_pantheon_cephdist(params, pantheon_data, 
                                                _sna_L, _sna_Lower, _sna_LogdetCov,
                                                cosmo, use_full_cov)
    
    if only_sna:
        return ll_snia, empty_blob(N_obj)

    # ------------------------
    # AGN likelihood
    # ------------------------
    z = agn_data['z']
    z_err = agn_data['z_err']
    m_obs = agn_data['apparent_mag_2500']
    m_err = agn_data['apparent_mag_2500_err']

    agn_params_arr = agn_model_pack_params(params)
    agn_obs_arr, agn_err_arr, agn_pivot_arr = agn_model_pack_obs(agn_data)

    M_pred = M_model_agn(agn_params_arr, agn_obs_arr, agn_pivot_arr)
    M_pred_err, idx = M_model_agn_err(agn_params_arr, agn_obs_arr, agn_err_arr, agn_pivot_arr, check_negative=True)
    if np.any(M_pred_err < 0):
        print(f"[ERROR] Negative AGN model error at indices: {idx}. Returning -inf log-likelihood.")
        print("For object_id: ", agn_data['object_id'][idx])
        raise ValueError("Negative AGN model error encountered.")
        # return -np.inf, empty_blob(N_obj)
    
    mu_err = np.sqrt(
        m_err**2 +
        M_pred_err**2 +
        z_err**2 +
        (0.055 * z)**2 +
        np.exp(params['log_f'])**2
    )
    mu_pred = m_obs - M_pred 

    mu_cosmo = cosmo.distmod(z).value

    ll_agn_terms = stats.norm.logpdf(mu_pred - mu_cosmo, scale=mu_err)
    mask_in = (0.44 < z) & (z < 3.16)
    #mask_in = z < 100
    ll_agn_terms[~mask_in] = 0.0  # zero out the contribution
    
    ll_agn = np.sum(ll_agn_terms)

    m_model = M_pred + mu_cosmo  # model-predicted magnitude

    ll_completeness = 0.0
    comp_blob = empty_blob(N_obj)
    if completeness_params is not None:
        completeness2d, mag_centers, _, _, _, completeness_scatter = completeness_params
        ll_completeness, comp_blob = completeness_loglike(
            m_obs=m_obs,
            m_obs_err=m_err,
            m_model=m_model, mu_err=mu_err, z=z,
            completeness2d=completeness2d, m_grid=mag_centers,
            sigma_completeness=completeness_scatter
        )
    ll = ll_snia + ll_agn - ll_completeness

    return ll, comp_blob
