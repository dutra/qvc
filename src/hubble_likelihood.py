
from scipy.linalg import cho_solve
from astropy.cosmology import FlatwCDM, Flatw0waCDM, FlatLambdaCDM, FlatwpwaCDM
from scipy import stats
import numpy as np

from numpy import trapezoid as trapz

#from hubble_utils import loglike_cmb_theta_simple
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

    Z = trapz(wpdf_model, m_grid, axis=1)                            # (N,)
    Z = np.clip(Z, tiny, None)                                          # guard denom

    # Debias for plotting (the scatter is mostly in M, not Malmquist)
    m_Z = trapz(wpdf_model * m_grid[None, :], m_grid, axis=1)
    m_Z = np.clip(m_Z, tiny, None)
    E = m_Z / Z
    dmi_obs = E - m_model
    dmi_obs[z<0.2] = 0.0

    blob = np.vstack([Z.astype(float), dmi_obs.astype(float)])
    loglike_terms = np.log(Z)

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

# --- Weak-lensing scatter from comoving distance (Shah+2022 arxiv:2203.09865) ---
def sigma_lens_from_dc(z, cosmo, amp=0.06, z_ref=1.0, power=3/2):
    """
    Return sigma_lens (mag) = amp * [dC(z)/dC(z_ref)]**power
    using comoving distances from the provided cosmology.
    """
    z = np.atleast_1d(z)
    dc   = cosmo.comoving_distance(z).value        # Mpc (units cancel in the ratio)
    dc_1 = float(cosmo.comoving_distance(z_ref).value)
    ratio = np.clip(dc / dc_1, 0.0, None)
    return amp * ratio**power

def log_likelihood(theta, *, agn_data, pantheon_data, 
                   _sna_L, _sna_Lower, _sna_LogdetCov,
                   cosmo_model, completeness_params,
                   z_pivot_agn,
                   agn_calibrators_data=None,
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
    
    sigma_lens = sigma_lens_from_dc(z, cosmo)   # vector (same shape as z)

    mu_err_sq = (
        m_err**2 +
        M_pred_err**2 +
        z_err**2 +
        #(0.055 * z)**2 +
        sigma_lens**2 +
        #(np.exp(params['log_f']) + params['sigma_b'] * (1+z))**2
        np.exp(params['log_f'])**2
    )    

    mu_err = np.sqrt(mu_err_sq)
    mu_pred = m_obs - M_pred 

    mu_cosmo = cosmo.distmod(z).value

    ll_agn_terms = stats.norm.logpdf(mu_pred - mu_cosmo, scale=mu_err)    
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

    # ll_cmb, _ = loglike_cmb_theta_simple(cosmo)
    
    ll = ll_snia + ll_agn - ll_completeness
    
    return ll, comp_blob

def log_likelihood_nearbylcs(
    theta, *, 
    agn_data,                 # main AGN sample
    agn_calibrators_data,     # separate table with AGN_IS_CALIBRATOR, MU_CAL, MU_CAL_ERR
    pantheon_data,            # unused here; kept for API symmetry
    _sna_L, _sna_Lower, _sna_LogdetCov,
    cosmo_model, completeness_params,
    z_pivot_agn,
    only_sna=False, use_full_cov=False, use_mu_sh0es=False
):
    """
    AGN likelihood with separate calibrators table.

    - Non-calibrator AGN: compare mu_pred to mu_cosmo (standard), apply your z-window.
    - Calibrators: use *only* agn_calibrators_data (no merge with agn_data):
        replace mu_cosmo with MU_CAL and use MU_CAL_ERR (plus model & intrinsic terms).
    """

    priors, model_labels, model_labels_latex = get_model_params(cosmo_model, only_sna=only_sna)
    model_priors = {key: priors[key] for key in model_labels}
    params = dict(zip(model_labels, theta))

    # Fixed-size blob for downstream consumers
    N_obj = len(agn_data['z'])

    # ---- Priors ----
    for key, (low, high) in model_priors.items():
        if low > high:
            raise ValueError(f"For key {key} prior: Low {low} > high {high}")
        if not (low < params[key] < high):
            return -np.inf, empty_blob(N_obj)

    # ---- Cosmology ----
    if cosmo_model == 'FlatwCDM':
        cosmo = FlatwCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'])
    elif cosmo_model == 'Flatw0waCDM':
        cosmo = Flatw0waCDM(H0=params['H0'], Om0=params['Om0'], w0=params['w0'], wa=params['wa'])
    elif cosmo_model == 'FlatwpwaCDM':
        cosmo = FlatwpwaCDM(H0=params['H0'], Om0=params['Om0'], wp=params['wp'], wa=params['wa'], zp=z_pivot_agn)
    elif cosmo_model == 'FlatLambdaCDM':
        cosmo = FlatLambdaCDM(H0=params['H0'], Om0=params['Om0'])

    # ========================
    # 1) NON-CALIBRATOR AGN
    # ========================
    # Exclude objects present as calibrators (where AGN_IS_CALIBRATOR==True) from agn_data
    cal_mask_tbl = np.asarray(agn_calibrators_data['AGN_IS_CALIBRATOR'], dtype=bool)
    cal_ids = set(np.asarray(agn_calibrators_data['object_id'])[cal_mask_tbl].astype(str).tolist())

    ids_agn = np.asarray(agn_data['object_id']).astype(str)
    mask_noncal = np.array([oid not in cal_ids for oid in ids_agn], dtype=bool)

    z_nc     = agn_data['z'][mask_noncal]
    z_err_nc = agn_data['z_err'][mask_noncal]
    m_obs_nc = agn_data['apparent_mag_2500'][mask_noncal]
    m_err_nc = agn_data['apparent_mag_2500_err'][mask_noncal]

    agn_params_arr = agn_model_pack_params(params)

    # pack obs/errs for the non-calibrator subset
    agn_obs_arr_nc, agn_err_arr_nc, agn_pivot_arr_nc = agn_model_pack_obs(
        {k: v[mask_noncal] for k, v in agn_data.items()}
    )

    M_pred_nc = M_model_agn(agn_params_arr, agn_obs_arr_nc, agn_pivot_arr_nc)
    M_pred_err_nc, idx_nc = M_model_agn_err(
        agn_params_arr, agn_obs_arr_nc, agn_err_arr_nc, agn_pivot_arr_nc, check_negative=True
    )
    if np.any(M_pred_err_nc < 0):
        print(f"[ERROR] Negative AGN model error at indices (non-cal): {idx_nc}.")
        raise ValueError("Negative AGN model error (non-calibrators).")

    mu_pred_nc  = m_obs_nc - M_pred_nc
    mu_cosmo_nc = cosmo.distmod(z_nc).value

    sigma_lens = sigma_lens_from_dc(z_nc, cosmo)   # vector (same shape as z)

    mu_err_nc = np.sqrt(
        m_err_nc**2 +
        M_pred_err_nc**2 +
        z_err_nc**2 +
        sigma_lens**2 +
        #(0.055 * z_nc)**2 +
        np.exp(params['log_f'])**2
    )

    ll_agn_terms_nc = stats.norm.logpdf(mu_pred_nc - mu_cosmo_nc, scale=mu_err_nc)
    ll_agn_noncal = np.sum(ll_agn_terms_nc)

    # ========================
    # 2) CALIBRATOR AGN (agn_calibrators_data ONLY)
    # ========================
    # Use only rows where AGN_IS_CALIBRATOR is True
    cal_ids_tbl = np.asarray(agn_calibrators_data['object_id']).astype(str)[cal_mask_tbl]
    if cal_ids_tbl.size > 0:
        m_obs_c    = agn_calibrators_data['apparent_mag_2500'][cal_mask_tbl]
        m_err_c    = agn_calibrators_data['apparent_mag_2500_err'][cal_mask_tbl]
        mu_cal     = agn_calibrators_data['MU_CAL'][cal_mask_tbl]
        mu_cal_err = agn_calibrators_data['MU_CAL_ERR'][cal_mask_tbl]

        # Pack obs/errs from the calibrator table itself
        # (Assumes packers accept dict-like with the same column names as agn_data)
        agn_obs_arr_c, agn_err_arr_c, agn_pivot_arr_c = agn_model_pack_obs(
            {k: agn_calibrators_data[k][cal_mask_tbl] for k in agn_calibrators_data.keys()}
        )

        M_pred_c = M_model_agn(agn_params_arr, agn_obs_arr_c, agn_pivot_arr_c)
        M_pred_err_c, idx_c = M_model_agn_err(
            agn_params_arr, agn_obs_arr_c, agn_err_arr_c, agn_pivot_arr_c, check_negative=True
        )
        if np.any(M_pred_err_c < 0):
            print(f"[ERROR] Negative AGN model error at indices (calibrators): {idx_c}.")
            raise ValueError("Negative AGN model error (calibrators).")

        mu_pred_c = m_obs_c - M_pred_c

        # For calibrators: drop z-terms; use provided MU_CAL_ERR
        mu_err_c = np.sqrt(
            m_err_c**2 +
            M_pred_err_c**2 +
            np.exp(params['log_f'])**2 +
            mu_cal_err**2
        )

        ll_agn_cal = np.sum(stats.norm.logpdf(mu_pred_c - mu_cal, scale=mu_err_c))
    else:
        raise ValueError("No calibrator AGN found in agn_calibrators_data where AGN_IS_CALIBRATOR is True.")
        ll_agn_cal = 0.0

    # ========================
    # 3) COMPLETENESS (non-calibrators only)
    # ========================
    ll_completeness = 0.0
    comp_blob = empty_blob(N_obj)
    if completeness_params is not None and np.any(mask_noncal):
        completeness2d, mag_centers, _, _, _, completeness_scatter, _, _ = completeness_params
        # model-predicted magnitude for non-calibrators (cosmo-anchored for selection)
        m_model_nc = M_pred_nc + mu_cosmo_nc
        ll_completeness, comp_blob = completeness_loglike(
            m_obs=m_obs_nc,
            m_obs_err=m_err_nc,
            m_model=m_model_nc, mu_err=mu_err_nc, z=z_nc,
            completeness2d=completeness2d, m_grid=mag_centers,
            sigma_completeness=completeness_scatter
        )

    # ========================
    # 4) Total
    # ========================
    ll = ll_agn_noncal + ll_agn_cal - ll_completeness
    return ll, comp_blob
