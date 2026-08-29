
from scipy.linalg import cho_solve
from astropy.cosmology import FlatwCDM, Flatw0waCDM, FlatLambdaCDM, FlatwpwaCDM
import numpy as np

#from qvc.hubble.hubble_utils import loglike_cmb_theta_simple
from qvc.hubble.hubble_model import (
    get_model_params,
    M_model_agn,
    M_model_agn_err,
    agn_model_pack_params,
    agn_model_pack_obs,
    evaluate_log_f,
)
from qvc.hubble.hubble_completeness_refactored import (
    COMPLETENESS_FHOST_COL,
    COMPLETENESS_MAG_COL,
    COMPLETENESS_MAG_ERR_COL,
)

_LOG_2PI = np.log(2.0 * np.pi)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)

SELECTION_ATTENUATION_MODES = ("fixed-offset", "joint-posterior")
JOINT_DEREDDENED_MAG_DRAWS_COL = "m_2500_dereddened_draws"
JOINT_ATTENUATED_MAG_DRAWS_COL = "m_2500_attenuated_model_draws"
JOINT_POSTERIOR_VALID_COUNT_COL = "joint_posterior_valid_count"


def normalize_selection_attenuation_mode(mode):
    normalized = str(mode).strip().lower()
    if normalized not in SELECTION_ATTENUATION_MODES:
        raise ValueError(
            f"Invalid selection_attenuation_mode={mode!r}; expected one of "
            f"{SELECTION_ATTENUATION_MODES}."
        )
    return normalized


def _normal_logpdf_sum(residuals, sigma):
    residuals = np.asarray(residuals, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return float(np.sum(-0.5 * (residuals / sigma) ** 2 - np.log(sigma) - 0.5 * _LOG_2PI))


def _attenuated_selection_inputs(
    agn_data,
    *,
    hubble_magnitude,
    hubble_magnitude_error,
    hubble_model_magnitude,
    hubble_total_error,
):
    """Express the selection integral in the configured 2500-A magnitude."""
    missing = {
        COMPLETENESS_MAG_COL,
        COMPLETENESS_MAG_ERR_COL,
    } - set(agn_data)
    if missing:
        raise KeyError(
            "Completeness likelihood requires explicitly prepared magnitude "
            f"fields {sorted(missing)}."
        )
    selection_magnitude = np.asarray(
        agn_data[COMPLETENESS_MAG_COL],
        dtype=float,
    )
    selection_magnitude_error = np.asarray(
        agn_data[COMPLETENESS_MAG_ERR_COL],
        dtype=float,
    )
    hubble_magnitude = np.asarray(hubble_magnitude, dtype=float)
    hubble_magnitude_error = np.asarray(hubble_magnitude_error, dtype=float)
    hubble_model_magnitude = np.asarray(hubble_model_magnitude, dtype=float)
    hubble_total_error = np.asarray(hubble_total_error, dtype=float)

    attenuation_offset = selection_magnitude - hubble_magnitude
    selection_model_magnitude = hubble_model_magnitude + attenuation_offset
    non_magnitude_variance = np.clip(
        hubble_total_error**2 - hubble_magnitude_error**2,
        0.0,
        None,
    )
    selection_total_error = np.sqrt(
        non_magnitude_variance + selection_magnitude_error**2
    )
    return (
        selection_magnitude,
        selection_magnitude_error,
        selection_model_magnitude,
        selection_total_error,
    )


def _array_cache_token(arr):
    if arr is None:
        return None
    arr = np.asarray(arr)
    return (id(arr), arr.shape, arr.dtype.str)


def _validate_magnitude_support(magnitude_support):
    """Return a finite, ordered hard-selection interval."""

    values = np.asarray(magnitude_support, dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError(
            "magnitude_support must contain exactly two finite numeric bounds."
        )
    lower, upper = (float(value) for value in values)
    if lower >= upper:
        raise ValueError(
            "magnitude_support must satisfy lower < upper; "
            f"got ({lower}, {upper})."
        )
    return lower, upper


def _validate_observed_magnitude_support(m_obs, magnitude_support):
    """Require selected observed magnitudes to lie within the hard support."""

    values = np.asarray(m_obs, dtype=float)
    lower, upper = _validate_magnitude_support(magnitude_support)
    outside = ~np.isfinite(values) | (values < lower) | (values > upper)
    if np.any(outside):
        raise ValueError(
            "All observed selection magnitudes must lie within the hard "
            f"support ({lower}, {upper}); found "
            f"{int(np.count_nonzero(outside))} outside."
        )
    return values


def _magnitude_integration_grid(m_grid, magnitude_support):
    """Insert exact physical cut edges around the calibrated center grid."""

    centers = np.asarray(m_grid, dtype=float)
    if (
        centers.ndim != 1
        or centers.size < 2
        or not np.all(np.isfinite(centers))
        or np.any(np.diff(centers) <= 0.0)
    ):
        raise ValueError(
            "m_grid must be finite, strictly increasing, and contain at least two points."
        )
    lower, upper = _validate_magnitude_support(magnitude_support)
    calibrated_lower = centers[0] - 0.5 * (centers[1] - centers[0])
    calibrated_upper = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    tolerance = 32.0 * np.finfo(float).eps * max(
        1.0, abs(lower), abs(upper), abs(calibrated_lower), abs(calibrated_upper)
    )
    if lower < calibrated_lower - tolerance or upper > calibrated_upper + tolerance:
        raise ValueError(
            "magnitude_support must lie within the calibrated magnitude-bin edges; "
            f"got ({lower}, {upper}) versus ({calibrated_lower}, {calibrated_upper})."
        )
    interior = centers[(centers > lower) & (centers < upper)]
    return np.concatenate(([lower], interior, [upper]))


def _cached_magnitude_integration_grid(completeness_model, m_grid, magnitude_support):
    support = _validate_magnitude_support(magnitude_support)
    key = (_array_cache_token(m_grid), support)
    cache = getattr(completeness_model, "_likelihood_magnitude_grid_cache", None)
    if cache is None:
        cache = {}
        setattr(completeness_model, "_likelihood_magnitude_grid_cache", cache)
    if key not in cache:
        cache[key] = _magnitude_integration_grid(m_grid, support)
    return cache[key]


def _cached_completeness_pdet(
    completeness_model,
    m_grid,
    z,
    *,
    f_host_2500_psf=None,
    alpha_lambda=None,
):
    """Evaluate edge-clamped p(detect), caching across sampler calls."""

    mode = getattr(completeness_model, "mode", "2d")
    key = (
        mode,
        _array_cache_token(m_grid),
        _array_cache_token(z),
        _array_cache_token(f_host_2500_psf),
        _array_cache_token(alpha_lambda),
    )
    cache = getattr(completeness_model, "_likelihood_pdet_cache", None)
    if cache is None:
        cache = {}
        setattr(completeness_model, "_likelihood_pdet_cache", cache)
    if key in cache:
        return cache[key]

    map_mag_centers = np.asarray(
        getattr(completeness_model, "mag_centers", m_grid), dtype=float
    )
    map_m_grid = np.clip(m_grid, map_mag_centers[0], map_mag_centers[-1])

    if mode == "4d_fhost_alpha":
        if f_host_2500_psf is None or alpha_lambda is None:
            raise ValueError("f_host_2500_psf and alpha_lambda are required for 4D host/color completeness.")
        p_det = completeness_model(
            map_m_grid[None, :],
            z[:, None],
            np.asarray(f_host_2500_psf)[:, None],
            np.asarray(alpha_lambda)[:, None],
        )
    elif mode == "3d_fhost":
        if f_host_2500_psf is None:
            raise ValueError("f_host_2500_psf is required for 3D host-aware completeness.")
        p_det = completeness_model(map_m_grid[None, :], z[:, None], np.asarray(f_host_2500_psf)[:, None])
    else:
        p_det = completeness_model(map_m_grid[None, :], z[:, None])

    p_det = np.asarray(p_det, dtype=float)
    cache[key] = p_det
    return p_det


def completeness_loglike(
    m_obs,
    m_obs_err,
    m_model,
    mu_err,
    z,
    completeness_model,
    m_grid,
    magnitude_support,
    sigma_completeness=0.0,
    tiny=1e-300,
    f_host_2500_psf=None,
    alpha_lambda=None,
):
    """
    Compute log-likelihood contribution from magnitude-limited sample selection.

    m_obs   : array (N_obj,) observed apparent magnitudes
    m_model : array (N_obj,) model-predicted apparent magnitudes
    mu_err  : array (N_obj,) Gaussian sigma for each magnitude
    z       : array (N_obj,) redshifts
    m_grid  : array (N_grid,) magnitude grid (e.g., the map's mag_centers)
    sigma_completeness : float, optional physical scatter in the selection variable.
        This should not be set from the completeness-map smoothing bandwidth.
    """
    m_grid = np.asarray(m_grid, dtype=float)
    magnitude_support = _validate_magnitude_support(magnitude_support)
    integration_grid = _cached_magnitude_integration_grid(
        completeness_model, m_grid, magnitude_support
    )
    _validate_observed_magnitude_support(m_obs, magnitude_support)
    z      = np.asarray(z)
    m_model = np.asarray(m_model)
    mu_err  = np.asarray(mu_err)

    # shared pieces
    sig = np.sqrt(mu_err[:, None]**2 + float(sigma_completeness)**2)   # (N,1)

    p_det = _cached_completeness_pdet(
        completeness_model,
        integration_grid,
        z,
        f_host_2500_psf=f_host_2500_psf,
        alpha_lambda=alpha_lambda,
    )

    # Model-centered selection factor: Z_i
    dx = (integration_grid[None, :] - m_model[:, None]) / sig
    pdf_model = np.exp(-0.5 * dx**2) * (_INV_SQRT_2PI / sig)  # (N,G)
    wpdf_model = pdf_model * p_det

    Z = np.trapezoid(wpdf_model, integration_grid, axis=1)
    m_Z = np.trapezoid(
        wpdf_model * integration_grid[None, :], integration_grid, axis=1
    )
    m2_Z = np.trapezoid(
        wpdf_model * integration_grid[None, :] ** 2, integration_grid, axis=1
    )

    Z = np.clip(Z, tiny, None)                                          # guard denom

    # Debias for plotting (the scatter is mostly in M, not Malmquist)
    # If the selection integral is effectively zero, the conditional
    # expectation is undefined. In that case keep the debias correction at
    # zero instead of manufacturing huge magnitude shifts from tiny/tiny.
    valid_Z = Z > (100.0 * tiny)
    E = np.where(valid_Z, m_Z / Z, m_model)
    E2 = np.where(valid_Z, m2_Z / Z, m_model**2)
    dmi_obs = E - m_model
    sigma_sel = np.sqrt(np.clip(E2 - E**2, 0.0, None))

    blob = np.vstack([Z.astype(float), dmi_obs.astype(float), sigma_sel.astype(float)])
    loglike_terms = np.log(Z)

    return np.sum(loglike_terms), blob


def _joint_attenuation_draw_arrays(agn_data, hubble_magnitude):
    required = {
        JOINT_DEREDDENED_MAG_DRAWS_COL,
        JOINT_ATTENUATED_MAG_DRAWS_COL,
        JOINT_POSTERIOR_VALID_COUNT_COL,
    }
    missing = sorted(required - set(agn_data))
    if missing:
        raise KeyError(
            "joint-posterior attenuation selection requires aligned spectra "
            f"v3 fields {missing}."
        )

    def matrix(column):
        raw = np.asarray(agn_data[column])
        if raw.ndim == 1 and raw.dtype == object:
            raw = np.stack(raw)
        return np.asarray(raw, dtype=float)

    dereddened = matrix(JOINT_DEREDDENED_MAG_DRAWS_COL)
    attenuated = matrix(JOINT_ATTENUATED_MAG_DRAWS_COL)
    counts_raw = np.asarray(agn_data[JOINT_POSTERIOR_VALID_COUNT_COL])
    hubble_magnitude = np.asarray(hubble_magnitude, dtype=float)
    if dereddened.ndim != 2 or attenuated.shape != dereddened.shape:
        raise ValueError("Paired joint-posterior magnitude draws must have identical 2D shapes.")
    n_objects, n_draws = dereddened.shape
    if counts_raw.shape != (n_objects,) or hubble_magnitude.shape != (n_objects,):
        raise ValueError("Joint-posterior fields must share the Hubble object axis.")
    if not np.all(np.isfinite(counts_raw)) or not np.all(counts_raw == np.floor(counts_raw)):
        raise ValueError("Joint-posterior valid counts must be exact integers.")
    counts = counts_raw.astype(int)
    if np.any((counts <= 0) | (counts > n_draws)):
        raise ValueError("Joint-posterior valid counts are outside the stored draw axis.")
    valid = np.arange(n_draws)[None, :] < counts[:, None]
    if np.any(~np.isfinite(dereddened[valid])) or np.any(~np.isfinite(attenuated[valid])):
        raise ValueError("Valid joint-posterior magnitude draws must be finite.")
    if np.any(attenuated[valid] < dereddened[valid] - 1e-12):
        raise ValueError("Attenuated magnitude draws cannot be brighter than paired dereddened draws.")
    return dereddened, attenuated, counts


def joint_posterior_completeness_loglike_for_data(
    *, completeness_params, agn_data, hubble_magnitude,
    hubble_magnitude_error, hubble_model_magnitude, hubble_total_error, z,
):
    """Marginalize the simple 2D selection integral over paired spectra-v3 draws."""
    model, m_grid = completeness_params[:2]
    m_grid = np.asarray(m_grid, dtype=float)
    magnitude_support = getattr(
        model, "magnitude_support", (float(m_grid[0]), float(m_grid[-1]))
    )
    integration_grid = _cached_magnitude_integration_grid(
        model, m_grid, magnitude_support
    )
    _validate_observed_magnitude_support(
        agn_data[COMPLETENESS_MAG_COL], magnitude_support
    )
    hubble_magnitude = np.asarray(hubble_magnitude, dtype=float)
    model_magnitude = np.asarray(hubble_model_magnitude, dtype=float)
    total_error = np.asarray(hubble_total_error, dtype=float)
    magnitude_error = np.asarray(hubble_magnitude_error, dtype=float)
    external_error = np.sqrt(np.clip(total_error**2 - magnitude_error**2, 0.0, None))
    if np.any(~np.isfinite(external_error)) or np.any(external_error <= 0.0):
        raise ValueError("Joint-posterior external errors must be finite and positive.")
    dereddened, attenuated, counts = _joint_attenuation_draw_arrays(
        agn_data, hubble_magnitude
    )
    valid = np.arange(dereddened.shape[1])[None, :] < counts[:, None]
    epsilon = dereddened - hubble_magnitude[:, None]
    attenuation = attenuated - dereddened
    centers = model_magnitude[:, None] + epsilon + attenuation
    p_det = _cached_completeness_pdet(
        model, integration_grid, np.asarray(z, dtype=float)
    )
    sigma = external_error[:, None, None]
    dx = (integration_grid[None, None, :] - centers[:, :, None]) / sigma
    weighted = np.exp(-0.5 * dx**2) * (_INV_SQRT_2PI / sigma)
    weighted *= p_det[:, None, :]
    weighted = np.where(valid[:, :, None], weighted, 0.0)
    component_z = np.trapezoid(weighted, integration_grid, axis=2)
    hubble_grid = integration_grid[None, None, :] - attenuation[:, :, None]
    component_mz = np.trapezoid(
        weighted * hubble_grid, integration_grid, axis=2
    )
    component_m2z = np.trapezoid(
        weighted * hubble_grid**2, integration_grid, axis=2
    )
    norm = counts.astype(float)
    z_raw = np.sum(component_z, axis=1) / norm
    mz = np.sum(component_mz, axis=1) / norm
    m2z = np.sum(component_m2z, axis=1) / norm
    z_safe = np.clip(z_raw, 1e-300, None)
    usable = z_raw > 1e-298
    expectation = np.where(usable, mz / z_safe, model_magnitude)
    expectation2 = np.where(usable, m2z / z_safe, model_magnitude**2)
    blob = np.vstack([
        z_safe,
        expectation - model_magnitude,
        np.sqrt(np.clip(expectation2 - expectation**2, 0.0, None)),
    ])
    return float(np.sum(np.log(z_safe))), blob



# --- Log-likelihood ---
def empty_blob(N_obj):
    # FIX: always return (3, N_obj) float array
    return np.zeros((3, N_obj), dtype=float)


def pantheon_distance_modulus(cosmo, z_hd, z_hel):
    """Return the Pantheon+ distance modulus for separate HD/heliocentric z.

    Pantheon+ uses the Hubble-diagram redshift for the comoving-distance
    integral and the heliocentric redshift for the photon redshift factor:

        D_L = (1 + z_hel) D_C(z_hd).

    All cosmologies supported by this pipeline are flat, so ``D_C`` is also
    the transverse comoving distance.
    """
    z_hd = np.asarray(z_hd, dtype=float)
    z_hel = np.asarray(z_hel, dtype=float)
    if z_hd.shape != z_hel.shape:
        raise ValueError(
            "Pantheon zHD and zHEL must have identical shapes; "
            f"got {z_hd.shape} and {z_hel.shape}."
        )
    if not np.all(np.isfinite(z_hd)):
        raise ValueError("Pantheon zHD must contain only finite values.")
    if not np.all(np.isfinite(z_hel)):
        raise ValueError("Pantheon zHEL must contain only finite values.")

    dc_mpc = np.asarray(cosmo.comoving_distance(z_hd).value, dtype=float)
    dl_mpc = (1.0 + z_hel) * dc_mpc
    if not np.all(np.isfinite(dl_mpc)) or np.any(dl_mpc <= 0.0):
        raise ValueError(
            "Pantheon mixed-redshift luminosity distances must be finite and positive."
        )
    return 5.0 * np.log10(dl_mpc) + 25.0


def log_likelihood_pantheon_cephdist(params, pantheon_data, _sna_L, _sna_Lower, _sna_LogdetCov,
                                     cosmo, use_full_cov, use_ceph_dist_calibration=True):
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
    try:
        zHEL = pantheon_data['zHEL'][mask]
    except KeyError as exc:
        raise KeyError(
            "Pantheon likelihood requires the zHEL field; "
            "zHD is not a valid fallback for the photon redshift factor."
        ) from exc
    m_b_corr = pantheon_data['m_b_corr'][mask]
    is_calib_sel = is_calib_bool[mask]

    # --- cosmological / Cepheid μ ---
    sn_mu_model = pantheon_distance_modulus(cosmo, zHD, zHEL)
    if use_ceph_dist_calibration:
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
        ll_snia = _normal_logpdf_sum(res_snia, sigma)

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


def sigma_mu_from_z_err(z, z_err, cosmo):
    """
    Project redshift uncertainty onto distance-modulus uncertainty with a
    central finite difference in z.
    """
    z = np.asarray(z, dtype=float)
    z_err = np.asarray(z_err, dtype=float)
    z_lo = np.maximum(z - z_err, 1e-8)
    z_hi = np.maximum(z + z_err, z_lo + 1e-8)
    mu_lo = cosmo.distmod(z_lo).value
    mu_hi = cosmo.distmod(z_hi).value
    sigma_mu = 0.5 * np.abs(mu_hi - mu_lo)
    return np.where(np.isfinite(z_err) & (z_err > 0.0), sigma_mu, 0.0)

def log_likelihood(theta, *, agn_data, pantheon_data, 
                   _sna_L, _sna_Lower, _sna_LogdetCov,
                   cosmo_model, completeness_params,
                   z_pivot_agn,
                   agn_pivot_context,
                   agn_calibrators_data=None,
                   use_planck_h0_prior=False,
                   use_planck_om_prior=False,
                   use_ceph_dist_calibration=True,
                   use_alpha_lambda_term=False,
                   use_eta_sigma_term=False,
                   use_redshift_log_f_term=False,
                   early_de_guard=False,
                   only_sna=False,
                   only_agn=False,
                   use_full_cov=False,
                   selection_attenuation_mode="fixed-offset"):
    selection_attenuation_mode = normalize_selection_attenuation_mode(
        selection_attenuation_mode
    )
    priors, model_labels, model_labels_latex = get_model_params(
        cosmo_model,
        only_sna=only_sna,
        only_agn=only_agn,
        use_planck_h0_prior=use_planck_h0_prior,
        use_planck_om_prior=use_planck_om_prior,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
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
        if early_de_guard and params['w0'] + params['wa'] >= 0:  # "no early DE" guard
            return -np.inf, empty_blob(N_obj)
    elif cosmo_model == 'FlatLambdaCDM':
        cosmo = FlatLambdaCDM(H0=params['H0'], Om0=params['Om0'])

    if only_agn:
        ll_snia = 0.0
    else:
        ll_snia = log_likelihood_pantheon_cephdist(
            params,
            pantheon_data,
            _sna_L,
            _sna_Lower,
            _sna_LogdetCov,
            cosmo,
            use_full_cov,
            use_ceph_dist_calibration=use_ceph_dist_calibration,
        )
    
    if only_sna:
        return ll_snia, empty_blob(N_obj)

    # ------------------------
    # AGN likelihood
    # ------------------------
    z = agn_data['z']
    z_err = agn_data['z_err']
    m_obs = agn_data['apparent_mag_2500']
    m_err = agn_data['apparent_mag_2500_err']

    agn_params_arr = agn_model_pack_params(
        params,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    agn_obs_arr, agn_err_arr, agn_pivot_arr = agn_model_pack_obs(
        agn_data,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        pivot_context=agn_pivot_context,
    )

    M_pred = M_model_agn(
        agn_params_arr,
        agn_obs_arr,
        agn_pivot_arr,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    M_pred_err, idx = M_model_agn_err(
        agn_params_arr,
        agn_obs_arr,
        agn_err_arr,
        agn_pivot_arr,
        check_negative=True,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    if np.any(M_pred_err < 0):
        print(f"[ERROR] Negative AGN model error at indices: {idx}. Returning -inf log-likelihood.")
        print("For object_id: ", agn_data['object_id'][idx])
        raise ValueError("Negative AGN model error encountered.")
        # return -np.inf, empty_blob(N_obj)
    
    sigma_lens = sigma_lens_from_dc(z, cosmo)   # vector (same shape as z)
    sigma_mu_z = sigma_mu_from_z_err(z, z_err, cosmo)

    log_f_eff = evaluate_log_f(
        params, z, z_pivot=z_pivot_agn, use_redshift_log_f_term=use_redshift_log_f_term
    )
    mu_err_sq = (
        m_err**2 +
        M_pred_err**2 +
        sigma_mu_z**2 +
        #(0.055 * z)**2 +
        sigma_lens**2 +
        #(np.exp(params['log_f']) + params['sigma_b'] * (1+z))**2
        np.exp(log_f_eff)**2
    )    

    mu_err = np.sqrt(mu_err_sq)
    mu_pred = m_obs - M_pred 

    mu_cosmo = cosmo.distmod(z).value

    ll_agn = _normal_logpdf_sum(mu_pred - mu_cosmo, mu_err)

    m_model = M_pred + mu_cosmo  # model-predicted magnitude

    ll_completeness = 0.0
    comp_blob = empty_blob(N_obj)
    if completeness_params is not None:
        completeness_model = completeness_params[0]
        mag_centers = completeness_params[1]
        (
            selection_magnitude,
            selection_magnitude_error,
            selection_model_magnitude,
            selection_total_error,
        ) = _attenuated_selection_inputs(
            agn_data,
            hubble_magnitude=m_obs,
            hubble_magnitude_error=m_err,
            hubble_model_magnitude=m_model,
            hubble_total_error=mu_err,
        )
        if selection_attenuation_mode == "fixed-offset":
            ll_completeness, comp_blob = completeness_loglike(
            m_obs=selection_magnitude,
            m_obs_err=selection_magnitude_error,
            m_model=selection_model_magnitude,
            mu_err=selection_total_error,
            z=z,
            completeness_model=completeness_model, m_grid=mag_centers,
            magnitude_support=getattr(
                completeness_model,
                "magnitude_support",
                (float(mag_centers[0]), float(mag_centers[-1])),
            ),
            sigma_completeness=0.0,
            f_host_2500_psf=agn_data.get(COMPLETENESS_FHOST_COL),
            alpha_lambda=agn_data.get("alpha_lambda"),
            )
        else:
            ll_completeness, comp_blob = joint_posterior_completeness_loglike_for_data(
                completeness_params=completeness_params,
                agn_data=agn_data,
                hubble_magnitude=m_obs,
                hubble_magnitude_error=m_err,
                hubble_model_magnitude=m_model,
                hubble_total_error=mu_err,
                z=z,
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
    agn_pivot_context,
    use_planck_h0_prior=False,
    use_planck_om_prior=False,
    use_ceph_dist_calibration=True,
    use_alpha_lambda_term=False,
    use_eta_sigma_term=False,
    use_redshift_log_f_term=False,
    early_de_guard=False,
    only_sna=False,
    only_agn=False,
    use_full_cov=False,
    selection_attenuation_mode="fixed-offset",
):
    """
    AGN likelihood with separate calibrators table.

    - Non-calibrator AGN: compare mu_pred to mu_cosmo (standard), apply your z-window.
    - Calibrators: use *only* agn_calibrators_data (no merge with agn_data):
        replace mu_cosmo with MU_CAL and use MU_CAL_ERR (plus model & intrinsic terms).
    """

    selection_attenuation_mode = normalize_selection_attenuation_mode(
        selection_attenuation_mode
    )
    priors, model_labels, model_labels_latex = get_model_params(
        cosmo_model,
        only_sna=only_sna,
        only_agn=only_agn,
        use_planck_h0_prior=use_planck_h0_prior,
        use_planck_om_prior=use_planck_om_prior,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
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
        if early_de_guard and params['w0'] + params['wa'] >= 0:  # "no early DE" guard
            return -np.inf, empty_blob(N_obj)
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

    agn_params_arr = agn_model_pack_params(
        params,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )

    # pack obs/errs for the non-calibrator subset
    agn_obs_arr_nc, agn_err_arr_nc, agn_pivot_arr_nc = agn_model_pack_obs(
        {k: v[mask_noncal] for k, v in agn_data.items()},
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        pivot_context=agn_pivot_context,
    )

    M_pred_nc = M_model_agn(
        agn_params_arr,
        agn_obs_arr_nc,
        agn_pivot_arr_nc,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    M_pred_err_nc, idx_nc = M_model_agn_err(
        agn_params_arr,
        agn_obs_arr_nc,
        agn_err_arr_nc,
        agn_pivot_arr_nc,
        check_negative=True,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    if np.any(M_pred_err_nc < 0):
        print(f"[ERROR] Negative AGN model error at indices (non-cal): {idx_nc}.")
        raise ValueError("Negative AGN model error (non-calibrators).")

    mu_pred_nc  = m_obs_nc - M_pred_nc
    mu_cosmo_nc = cosmo.distmod(z_nc).value

    sigma_lens = sigma_lens_from_dc(z_nc, cosmo)   # vector (same shape as z)
    sigma_mu_z_nc = sigma_mu_from_z_err(z_nc, z_err_nc, cosmo)

    log_f_eff_nc = evaluate_log_f(
        params, z_nc, z_pivot=z_pivot_agn, use_redshift_log_f_term=use_redshift_log_f_term
    )
    mu_err_nc = np.sqrt(
        m_err_nc**2 +
        M_pred_err_nc**2 +
        sigma_mu_z_nc**2 +
        sigma_lens**2 +
        #(0.055 * z_nc)**2 +
        np.exp(log_f_eff_nc)**2
    )

    ll_agn_noncal = _normal_logpdf_sum(mu_pred_nc - mu_cosmo_nc, mu_err_nc)

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
            {k: agn_calibrators_data[k][cal_mask_tbl] for k in agn_calibrators_data.keys()},
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
            pivot_context=agn_pivot_context,
        )

        M_pred_c = M_model_agn(
            agn_params_arr,
            agn_obs_arr_c,
            agn_pivot_arr_c,
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
        )
        M_pred_err_c, idx_c = M_model_agn_err(
            agn_params_arr,
            agn_obs_arr_c,
            agn_err_arr_c,
            agn_pivot_arr_c,
            check_negative=True,
            use_alpha_lambda_term=use_alpha_lambda_term,
            use_eta_sigma_term=use_eta_sigma_term,
        )
        if np.any(M_pred_err_c < 0):
            print(f"[ERROR] Negative AGN model error at indices (calibrators): {idx_c}.")
            raise ValueError("Negative AGN model error (calibrators).")

        mu_pred_c = m_obs_c - M_pred_c

        # For calibrators: drop z-terms; use provided MU_CAL_ERR
        log_f_eff_c = evaluate_log_f(
            params,
            agn_calibrators_data['z'][cal_mask_tbl],
            z_pivot=z_pivot_agn,
            use_redshift_log_f_term=use_redshift_log_f_term,
        )
        mu_err_c = np.sqrt(
            m_err_c**2 +
            M_pred_err_c**2 +
            np.exp(log_f_eff_c)**2 +
            mu_cal_err**2
        )

        ll_agn_cal = _normal_logpdf_sum(mu_pred_c - mu_cal, mu_err_c)
    else:
        raise ValueError("No calibrator AGN found in agn_calibrators_data where AGN_IS_CALIBRATOR is True.")
        ll_agn_cal = 0.0

    # ========================
    # 3) COMPLETENESS (non-calibrators only)
    # ========================
    ll_completeness = 0.0
    comp_blob = empty_blob(N_obj)
    if completeness_params is not None and np.any(mask_noncal):
        completeness_model = completeness_params[0]
        mag_centers = completeness_params[1]
        # model-predicted magnitude for non-calibrators (cosmo-anchored for selection)
        m_model_nc = M_pred_nc + mu_cosmo_nc
        agn_data_nc = {
            key: np.asarray(value)[mask_noncal]
            for key, value in agn_data.items()
        }
        (
            selection_magnitude_nc,
            selection_magnitude_error_nc,
            selection_model_magnitude_nc,
            selection_total_error_nc,
        ) = _attenuated_selection_inputs(
            agn_data_nc,
            hubble_magnitude=m_obs_nc,
            hubble_magnitude_error=m_err_nc,
            hubble_model_magnitude=m_model_nc,
            hubble_total_error=mu_err_nc,
        )
        if selection_attenuation_mode == "fixed-offset":
            ll_completeness, comp_blob = completeness_loglike(
                m_obs=selection_magnitude_nc,
                m_obs_err=selection_magnitude_error_nc,
                m_model=selection_model_magnitude_nc,
                mu_err=selection_total_error_nc,
                z=z_nc,
                completeness_model=completeness_model, m_grid=mag_centers,
                magnitude_support=getattr(
                    completeness_model,
                    "magnitude_support",
                    (float(mag_centers[0]), float(mag_centers[-1])),
                ),
                sigma_completeness=0.0,
                f_host_2500_psf=agn_data.get(COMPLETENESS_FHOST_COL, None)[mask_noncal] if agn_data.get(COMPLETENESS_FHOST_COL, None) is not None else None,
                alpha_lambda=agn_data.get("alpha_lambda", None)[mask_noncal] if agn_data.get("alpha_lambda", None) is not None else None,
            )
        else:
            ll_completeness, comp_blob = joint_posterior_completeness_loglike_for_data(
                completeness_params=completeness_params,
                agn_data=agn_data_nc,
                hubble_magnitude=m_obs_nc,
                hubble_magnitude_error=m_err_nc,
                hubble_model_magnitude=m_model_nc,
                hubble_total_error=mu_err_nc,
                z=z_nc,
            )

    # ========================
    # 4) Total
    # ========================
    ll = ll_agn_noncal + ll_agn_cal - ll_completeness
    return ll, comp_blob
