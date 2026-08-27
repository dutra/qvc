
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
    evaluate_mu_redshift_term,
)
from qvc.hubble.hubble_completeness_refactored import (
    COMPLETENESS_MAG_COL,
    COMPLETENESS_MAG_ERR_COL,
)
from qvc.hubble.completeness_strata import (
    COMPLETENESS_STRATUM_CODE_COL,
    StratifiedCompletenessBundle,
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
    """Return a finite, ordered ``(lower, upper)`` hard-cut interval."""

    try:
        support = np.asarray(magnitude_support, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "magnitude_support must contain exactly two finite numeric bounds."
        ) from exc
    if support.shape != (2,) or not np.all(np.isfinite(support)):
        raise ValueError(
            "magnitude_support must contain exactly two finite numeric bounds."
        )
    lower, upper = (float(value) for value in support)
    if lower >= upper:
        raise ValueError(
            "magnitude_support must satisfy lower < upper; "
            f"got ({lower}, {upper})."
        )
    return lower, upper


def _validate_observed_magnitude_support(m_obs, magnitude_support):
    """Require every supplied selected magnitude to satisfy the hard cuts."""

    m_obs = np.asarray(m_obs, dtype=float)
    outside_mask = (
        ~np.isfinite(m_obs)
        | (m_obs < magnitude_support[0])
        | (m_obs > magnitude_support[1])
    )
    if np.any(outside_mask):
        raise ValueError(
            "All observed selection magnitudes must lie within the hard "
            f"support {magnitude_support}; found "
            f"{int(np.count_nonzero(outside_mask))} outside."
        )
    return m_obs


def _magnitude_integration_grid(m_grid, magnitude_support):
    """Restrict a center grid to hard cuts, inserting exact cut endpoints.

    The map nodes are histogram-bin centers.  The calibrated domain therefore
    extends by half of the adjacent bin spacing beyond the first and last node.
    """

    m_grid = np.asarray(m_grid, dtype=float)
    if (
        m_grid.ndim != 1
        or m_grid.size < 2
        or not np.all(np.isfinite(m_grid))
        or np.any(np.diff(m_grid) <= 0.0)
    ):
        raise ValueError(
            "m_grid must be a finite, strictly increasing one-dimensional "
            "array with at least two points."
        )

    lower, upper = _validate_magnitude_support(magnitude_support)
    calibrated_lower = float(m_grid[0] - 0.5 * (m_grid[1] - m_grid[0]))
    calibrated_upper = float(m_grid[-1] + 0.5 * (m_grid[-1] - m_grid[-2]))
    edge_tolerance = 32.0 * np.finfo(float).eps * max(
        1.0,
        abs(lower),
        abs(upper),
        abs(calibrated_lower),
        abs(calibrated_upper),
    )
    if (
        lower < calibrated_lower - edge_tolerance
        or upper > calibrated_upper + edge_tolerance
    ):
        raise ValueError(
            "magnitude_support must lie within the calibrated magnitude-bin "
            "edges; "
            f"got support=({lower}, {upper}) and "
            f"edges=({calibrated_lower}, {calibrated_upper})."
        )
    if abs(lower - m_grid[0]) <= edge_tolerance:
        lower = float(m_grid[0])
    if abs(upper - m_grid[-1]) <= edge_tolerance:
        upper = float(m_grid[-1])
    if lower == m_grid[0] and upper == m_grid[-1]:
        return m_grid

    interior = m_grid[(m_grid > lower) & (m_grid < upper)]
    return np.concatenate(([lower], interior, [upper]))


def _cached_magnitude_integration_grid(
    completeness_model,
    m_grid,
    magnitude_support,
):
    """Return one stable bounded grid per map/grid/support combination."""

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
):
    """Evaluate p(detect) on the fixed likelihood grid, caching across calls.

    Integration coordinates may reach the histogram edges.  Map lookup
    coordinates are clipped to the center grid so the outer bin values extend
    constantly across their remaining half bins.
    """

    key = (
        _array_cache_token(m_grid),
        _array_cache_token(z),
    )
    cache = getattr(completeness_model, "_likelihood_pdet_cache", None)
    if cache is None:
        cache = {}
        setattr(completeness_model, "_likelihood_pdet_cache", cache)
    if key in cache:
        return cache[key]

    map_mag_centers = getattr(completeness_model, "mag_centers", None)
    if map_mag_centers is None:
        map_m_grid = m_grid
    else:
        map_mag_centers = np.asarray(map_mag_centers, dtype=float)
        map_m_grid = np.clip(
            m_grid,
            map_mag_centers[0],
            map_mag_centers[-1],
        )

    if getattr(completeness_model, "mode", "2d") != "2d":
        raise ValueError("The Hubble likelihood accepts only frozen 2D completeness maps.")
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
):
    """
    Compute log-likelihood contribution from magnitude-limited sample selection.

    m_obs   : array (N_obj,) observed apparent magnitudes
    m_model : array (N_obj,) model-predicted apparent magnitudes
    mu_err  : array (N_obj,) Gaussian sigma for each magnitude
    z       : array (N_obj,) redshifts
    m_grid  : array (N_grid,) magnitude grid (e.g., the map's mag_centers)
    magnitude_support : (lower, upper) finite hard cuts in the same magnitude
        convention as ``m_obs`` and ``m_model``.
    sigma_completeness : float, optional physical scatter in the selection variable.
        This should not be set from the completeness-map smoothing bandwidth.
    """
    m_grid = np.asarray(m_grid, dtype=float)
    magnitude_support = _validate_magnitude_support(magnitude_support)
    integration_grid = _cached_magnitude_integration_grid(
        completeness_model,
        m_grid,
        magnitude_support,
    )
    _validate_observed_magnitude_support(m_obs, magnitude_support)
    z = np.asarray(z)
    m_model = np.asarray(m_model)
    mu_err = np.asarray(mu_err)

    # shared pieces
    sig = np.sqrt(mu_err[:, None]**2 + float(sigma_completeness)**2)   # (N,1)

    p_det = _cached_completeness_pdet(
        completeness_model,
        integration_grid,
        z,
    )

    # Model-centered selection factor: Z_i
    dx = (integration_grid[None, :] - m_model[:, None]) / sig
    pdf_model = np.exp(-0.5 * dx**2) * (_INV_SQRT_2PI / sig)  # (N,G)
    wpdf_model = pdf_model * p_det

    Z = np.trapezoid(wpdf_model, integration_grid, axis=1)  # (N,)
    m_Z = np.trapezoid(
        wpdf_model * integration_grid[None, :],
        integration_grid,
        axis=1,
    )
    m2_Z = np.trapezoid(
        wpdf_model * integration_grid[None, :] ** 2,
        integration_grid,
        axis=1,
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
    """Return validated paired v3 magnitude draws and their valid mask."""

    required = {
        JOINT_DEREDDENED_MAG_DRAWS_COL,
        JOINT_ATTENUATED_MAG_DRAWS_COL,
        JOINT_POSTERIOR_VALID_COUNT_COL,
    }
    missing = sorted(required - set(agn_data))
    if missing:
        raise KeyError(
            "joint-posterior attenuation selection requires aligned v3 "
            f"spectral posterior fields {missing}."
        )
    def draw_matrix(column):
        raw = np.asarray(agn_data[column])
        if raw.ndim == 1 and raw.dtype == object:
            try:
                raw = np.stack(raw)
            except ValueError as exc:
                raise ValueError(
                    f"Could not stack aligned posterior field {column!r}."
                ) from exc
        return np.asarray(raw, dtype=float)

    dereddened = draw_matrix(JOINT_DEREDDENED_MAG_DRAWS_COL)
    attenuated = draw_matrix(JOINT_ATTENUATED_MAG_DRAWS_COL)
    counts_raw = np.asarray(
        agn_data[JOINT_POSTERIOR_VALID_COUNT_COL]
    )
    hubble_magnitude = np.asarray(hubble_magnitude, dtype=float)
    if dereddened.ndim != 2 or attenuated.shape != dereddened.shape:
        raise ValueError(
            "joint-posterior magnitude draws must be aligned two-dimensional "
            "arrays with identical shapes."
        )
    n_objects, n_draws = dereddened.shape
    if hubble_magnitude.shape != (n_objects,) or counts_raw.shape != (n_objects,):
        raise ValueError(
            "joint-posterior magnitude draws, valid counts, and Hubble "
            "magnitudes must share the same object axis."
        )
    if not np.issubdtype(counts_raw.dtype, np.integer):
        if not np.all(np.isfinite(counts_raw)) or not np.all(
            counts_raw == np.floor(counts_raw)
        ):
            raise ValueError("joint posterior valid counts must be exact integers.")
    counts = counts_raw.astype(int)
    if np.any((counts <= 0) | (counts > n_draws)):
        raise ValueError(
            "joint posterior valid counts must be between one and the stored "
            "draw-axis length."
        )
    valid = np.arange(n_draws)[None, :] < counts[:, None]
    if np.any(~np.isfinite(dereddened[valid])) or np.any(
        ~np.isfinite(attenuated[valid])
    ):
        raise ValueError("joint-posterior valid magnitude draws must be finite.")
    if np.any(attenuated[valid] < dereddened[valid] - 1e-12):
        raise ValueError(
            "joint-posterior attenuated magnitude draws cannot be brighter "
            "than their paired dereddened draws."
        )
    if np.any(~np.isfinite(hubble_magnitude)):
        raise ValueError("joint-posterior Hubble magnitudes must be finite.")
    return dereddened, attenuated, counts, valid


def completeness_loglike_joint_posterior(
    *,
    hubble_magnitude,
    hubble_model_magnitude,
    external_error,
    z,
    dereddened_draws,
    attenuated_draws,
    valid_draw_counts,
    completeness_model,
    m_grid,
    magnitude_support,
    tiny=1e-300,
):
    """Marginalize an attenuated completeness integral over paired v3 draws."""

    hubble_magnitude = np.asarray(hubble_magnitude, dtype=float)
    hubble_model_magnitude = np.asarray(hubble_model_magnitude, dtype=float)
    external_error = np.asarray(external_error, dtype=float)
    z = np.asarray(z, dtype=float)
    dereddened_draws = np.asarray(dereddened_draws, dtype=float)
    attenuated_draws = np.asarray(attenuated_draws, dtype=float)
    valid_draw_counts = np.asarray(valid_draw_counts, dtype=int)
    n_objects, n_draws = dereddened_draws.shape
    expected_vector = (n_objects,)
    for name, values in (
        ("hubble_magnitude", hubble_magnitude),
        ("hubble_model_magnitude", hubble_model_magnitude),
        ("external_error", external_error),
        ("z", z),
        ("valid_draw_counts", valid_draw_counts),
    ):
        if values.shape != expected_vector:
            raise ValueError(f"{name} has shape {values.shape}; expected {expected_vector}.")
    if attenuated_draws.shape != dereddened_draws.shape:
        raise ValueError("Paired joint-posterior magnitude draws have different shapes.")
    if np.any(~np.isfinite(external_error)) or np.any(external_error <= 0.0):
        raise ValueError("joint-posterior external errors must be finite and positive.")
    valid = np.arange(n_draws)[None, :] < valid_draw_counts[:, None]

    m_grid = np.asarray(m_grid, dtype=float)
    magnitude_support = _validate_magnitude_support(magnitude_support)
    integration_grid = _cached_magnitude_integration_grid(
        completeness_model, m_grid, magnitude_support
    )
    p_det = _cached_completeness_pdet(
        completeness_model, integration_grid, z
    )

    safe_dereddened = np.where(valid, dereddened_draws, hubble_magnitude[:, None])
    safe_attenuated = np.where(valid, attenuated_draws, hubble_magnitude[:, None])
    epsilon = safe_dereddened - hubble_magnitude[:, None]
    attenuation = safe_attenuated - safe_dereddened
    centers = hubble_model_magnitude[:, None] + epsilon + attenuation
    sig = external_error[:, None, None]
    dx = (integration_grid[None, None, :] - centers[:, :, None]) / sig
    pdf = np.exp(-0.5 * dx**2) * (_INV_SQRT_2PI / sig)
    weighted_pdf = pdf * p_det[:, None, :]
    weighted_pdf = np.where(valid[:, :, None], weighted_pdf, 0.0)

    component_Z = np.trapezoid(weighted_pdf, integration_grid, axis=2)
    hubble_grid = integration_grid[None, None, :] - attenuation[:, :, None]
    component_mZ = np.trapezoid(
        weighted_pdf * hubble_grid, integration_grid, axis=2
    )
    component_m2Z = np.trapezoid(
        weighted_pdf * hubble_grid**2, integration_grid, axis=2
    )
    draw_norm = valid_draw_counts.astype(float)
    Z_raw = np.sum(component_Z, axis=1) / draw_norm
    m_Z = np.sum(component_mZ, axis=1) / draw_norm
    m2_Z = np.sum(component_m2Z, axis=1) / draw_norm
    Z = np.clip(Z_raw, tiny, None)
    valid_Z = Z_raw > (100.0 * tiny)
    expectation = np.where(valid_Z, m_Z / Z, hubble_model_magnitude)
    expectation2 = np.where(
        valid_Z, m2_Z / Z, hubble_model_magnitude**2
    )
    dmi = expectation - hubble_model_magnitude
    sigma_selection = np.sqrt(
        np.clip(expectation2 - expectation**2, 0.0, None)
    )
    blob = np.vstack([Z, dmi, sigma_selection]).astype(float)
    return float(np.sum(np.log(Z))), blob


def joint_posterior_completeness_loglike_for_data(
    *,
    completeness_params,
    agn_data,
    hubble_magnitude,
    hubble_magnitude_error,
    hubble_model_magnitude,
    hubble_total_error,
    z,
):
    """Evaluate paired-draw attenuation marginalization, including strata."""

    if COMPLETENESS_MAG_COL not in agn_data:
        raise KeyError(
            "joint-posterior attenuation selection requires the prepared "
            f"selection magnitude field {COMPLETENESS_MAG_COL!r}."
        )
    selection_magnitude = np.asarray(
        agn_data[COMPLETENESS_MAG_COL], dtype=float
    )
    dereddened, attenuated, counts, _ = _joint_attenuation_draw_arrays(
        agn_data, hubble_magnitude
    )
    hubble_total_error = np.asarray(hubble_total_error, dtype=float)
    hubble_magnitude_error = np.asarray(hubble_magnitude_error, dtype=float)
    external_error = np.sqrt(
        np.clip(
            hubble_total_error**2 - hubble_magnitude_error**2,
            0.0,
            None,
        )
    )
    n_objects = len(np.asarray(z))

    def evaluate(params, mask):
        model, mag_centers = params[:2]
        support = getattr(
            model,
            "magnitude_support",
            (float(mag_centers[0]), float(mag_centers[-1])),
        )
        _validate_observed_magnitude_support(
            selection_magnitude[mask], support
        )
        return completeness_loglike_joint_posterior(
            hubble_magnitude=np.asarray(hubble_magnitude)[mask],
            hubble_model_magnitude=np.asarray(hubble_model_magnitude)[mask],
            external_error=external_error[mask],
            z=np.asarray(z)[mask],
            dereddened_draws=dereddened[mask],
            attenuated_draws=attenuated[mask],
            valid_draw_counts=counts[mask],
            completeness_model=model,
            m_grid=mag_centers,
            magnitude_support=support,
        )

    if not isinstance(completeness_params, StratifiedCompletenessBundle):
        return evaluate(completeness_params, np.ones(n_objects, dtype=bool))
    if COMPLETENESS_STRATUM_CODE_COL not in agn_data:
        raise KeyError(
            f"Stratified completeness requires {COMPLETENESS_STRATUM_CODE_COL!r}."
        )
    codes = np.asarray(agn_data[COMPLETENESS_STRATUM_CODE_COL], dtype=int)
    blob = empty_blob(n_objects)
    total = 0.0
    for code, params in enumerate(completeness_params.params_by_stratum):
        mask = codes == code
        if not np.any(mask):
            raise ValueError(
                f"No object for completeness stratum {completeness_params.stratum_names[code]!r}."
            )
        group_loglike, group_blob = evaluate(params, mask)
        total += group_loglike
        blob[:, mask] = group_blob
    return total, blob



# --- Log-likelihood ---
def empty_blob(N_obj):
    # FIX: always return (3, N_obj) float array
    return np.zeros((3, N_obj), dtype=float)


def completeness_loglike_for_data(
    *,
    completeness_params,
    agn_data,
    m_obs,
    m_obs_err,
    m_model,
    mu_err,
    z,
):
    """Evaluate a frozen two-dimensional completeness map.

    The likelihood accepts only a precomputed C(m_HD, z) interpolator. Color
    features, target flags, qsogen draws, support models, and fitted color-head
    parameters are intentionally absent from this interface.
    """
    n_objects = len(np.asarray(z))
    if completeness_params is None:
        return 0.0, empty_blob(n_objects)

    def evaluate(params, mask):
        model, mag_centers = params[:2]
        if hasattr(model, "fhost_centers") or hasattr(model, "alpha_centers"):
            raise TypeError("The Hubble likelihood accepts only frozen 2D completeness maps.")
        support = getattr(model, "magnitude_support", (float(mag_centers[0]), float(mag_centers[-1])))
        return completeness_loglike(
            m_obs=np.asarray(m_obs)[mask],
            m_obs_err=np.asarray(m_obs_err)[mask],
            m_model=np.asarray(m_model)[mask],
            mu_err=np.asarray(mu_err)[mask],
            z=np.asarray(z)[mask],
            completeness_model=model,
            m_grid=mag_centers,
            magnitude_support=support,
            sigma_completeness=0.0,
        )

    if not isinstance(completeness_params, StratifiedCompletenessBundle):
        return evaluate(completeness_params, np.ones(n_objects, dtype=bool))
    if COMPLETENESS_STRATUM_CODE_COL not in agn_data:
        raise KeyError(f"Stratified completeness requires {COMPLETENESS_STRATUM_CODE_COL!r}.")
    codes = np.asarray(agn_data[COMPLETENESS_STRATUM_CODE_COL], dtype=int)
    if codes.shape != (n_objects,):
        raise ValueError("Completeness stratum codes have the wrong shape.")
    if np.any((codes < 0) | (codes >= len(completeness_params.stratum_names))):
        raise ValueError("Completeness stratum codes are outside the configured bundle.")
    blob = empty_blob(n_objects)
    total = 0.0
    for code, params in enumerate(completeness_params.params_by_stratum):
        mask = codes == code
        if not np.any(mask):
            raise ValueError(f"No object for completeness stratum {completeness_params.stratum_names[code]!r}.")
        group_loglike, group_blob = evaluate(params, mask)
        total += float(group_loglike)
        blob[:, mask] = group_blob
    return total, blob

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


def sigma_mu_model_from_z_err(
    z,
    z_err,
    cosmo,
    params,
    *,
    z_pivot,
    use_redshift_mu_term=False,
):
    """Project redshift uncertainty through cosmology plus mean evolution."""
    if not use_redshift_mu_term:
        return sigma_mu_from_z_err(z, z_err, cosmo)
    z = np.asarray(z, dtype=float)
    z_err = np.asarray(z_err, dtype=float)
    z_lo = np.maximum(z - z_err, 1e-8)
    z_hi = np.maximum(z + z_err, z_lo + 1e-8)
    mu_lo = cosmo.distmod(z_lo).value + evaluate_mu_redshift_term(
        params, z_lo, z_pivot, use_redshift_mu_term=True
    )
    mu_hi = cosmo.distmod(z_hi).value + evaluate_mu_redshift_term(
        params, z_hi, z_pivot, use_redshift_mu_term=True
    )
    sigma_mu = 0.5 * np.abs(mu_hi - mu_lo)
    return np.where(np.isfinite(z_err) & (z_err > 0.0), sigma_mu, 0.0)


def _cosmology_from_agn_params(cosmo_model, params, z_pivot_agn):
    if cosmo_model == "FlatwCDM":
        return FlatwCDM(H0=params["H0"], Om0=params["Om0"], w0=params["w0"])
    if cosmo_model == "Flatw0waCDM":
        return Flatw0waCDM(
            H0=params["H0"],
            Om0=params["Om0"],
            w0=params["w0"],
            wa=params["wa"],
        )
    if cosmo_model == "FlatwpwaCDM":
        return FlatwpwaCDM(
            H0=params["H0"],
            Om0=params["Om0"],
            wp=params["wp"],
            wa=params["wa"],
            zp=z_pivot_agn,
        )
    if cosmo_model == "FlatLambdaCDM":
        return FlatLambdaCDM(H0=params["H0"], Om0=params["Om0"])
    raise ValueError(f"Unsupported cosmology model {cosmo_model!r}.")


def agn_selection_prediction(
    theta,
    *,
    agn_data,
    cosmo_model,
    z_pivot_agn,
    agn_pivot_context,
    only_agn=False,
    use_planck_h0_prior=False,
    use_planck_om_prior=False,
    use_alpha_lambda_term=False,
    use_eta_sigma_term=False,
    use_redshift_log_f_term=False,
    use_redshift_mu_term=False,
    require_selection_fields=True,
):
    """Return one posterior draw's exact AGN selection-space prediction."""

    _, model_labels, _ = get_model_params(
        cosmo_model,
        only_sna=False,
        only_agn=only_agn,
        use_planck_h0_prior=use_planck_h0_prior,
        use_planck_om_prior=use_planck_om_prior,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
        use_redshift_mu_term=use_redshift_mu_term,
    )
    params = dict(zip(model_labels, np.asarray(theta, dtype=float)))
    cosmo = _cosmology_from_agn_params(cosmo_model, params, z_pivot_agn)

    z = np.asarray(agn_data["z"], dtype=float)
    z_err = np.asarray(agn_data["z_err"], dtype=float)
    m_obs = np.asarray(agn_data["apparent_mag_2500"], dtype=float)
    m_err = np.asarray(agn_data["apparent_mag_2500_err"], dtype=float)
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
    M_pred_err, negative_indices = M_model_agn_err(
        agn_params_arr,
        agn_obs_arr,
        agn_err_arr,
        agn_pivot_arr,
        check_negative=True,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
    )
    if np.any(M_pred_err < 0.0):
        object_ids = np.asarray(agn_data["object_id"])
        raise ValueError(
            "Negative AGN model error for object_id values "
            f"{object_ids[negative_indices].tolist()}."
        )

    sigma_lens = sigma_lens_from_dc(z, cosmo)
    sigma_mu_z = sigma_mu_model_from_z_err(
        z,
        z_err,
        cosmo,
        params,
        z_pivot=z_pivot_agn,
        use_redshift_mu_term=use_redshift_mu_term,
    )
    log_f_eff = evaluate_log_f(
        params,
        z,
        z_pivot=z_pivot_agn,
        use_redshift_log_f_term=use_redshift_log_f_term,
    )
    total_error = np.sqrt(
        m_err**2
        + M_pred_err**2
        + sigma_mu_z**2
        + sigma_lens**2
        + np.exp(log_f_eff) ** 2
    )
    mu_cosmo = np.asarray(cosmo.distmod(z).value, dtype=float)
    delta_mu_z = evaluate_mu_redshift_term(
        params,
        z,
        z_pivot_agn,
        use_redshift_mu_term=use_redshift_mu_term,
    )
    mu_model = mu_cosmo + delta_mu_z
    model_magnitude = np.asarray(M_pred + mu_model, dtype=float)
    selection_fields = {COMPLETENESS_MAG_COL, COMPLETENESS_MAG_ERR_COL}
    if selection_fields.issubset(agn_data):
        (
            selection_magnitude,
            selection_magnitude_error,
            selection_model_magnitude,
            selection_total_error,
        ) = _attenuated_selection_inputs(
            agn_data,
            hubble_magnitude=m_obs,
            hubble_magnitude_error=m_err,
            hubble_model_magnitude=model_magnitude,
            hubble_total_error=total_error,
        )
    elif require_selection_fields:
        missing = sorted(selection_fields - set(agn_data))
        raise KeyError(
            "Completeness likelihood requires explicitly prepared magnitude "
            f"fields {missing}."
        )
    else:
        selection_magnitude = m_obs
        selection_magnitude_error = m_err
        selection_model_magnitude = model_magnitude
        selection_total_error = total_error
    return {
        "params": params,
        "cosmology": cosmo,
        "M_pred": np.asarray(M_pred, dtype=float),
        "M_pred_err": np.asarray(M_pred_err, dtype=float),
        "mu_pred": m_obs - M_pred,
        "mu_cosmo": mu_cosmo,
        "delta_mu_z": delta_mu_z,
        "mu_model": mu_model,
        "model_magnitude": model_magnitude,
        "total_error": total_error,
        "selection_magnitude": selection_magnitude,
        "selection_magnitude_error": selection_magnitude_error,
        "selection_model_magnitude": selection_model_magnitude,
        "selection_total_error": selection_total_error,
    }

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
                   use_redshift_mu_term=False,
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
        use_redshift_mu_term=use_redshift_mu_term,
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

    prediction = agn_selection_prediction(
        theta,
        agn_data=agn_data,
        cosmo_model=cosmo_model,
        z_pivot_agn=z_pivot_agn,
        agn_pivot_context=agn_pivot_context,
        only_agn=only_agn,
        use_planck_h0_prior=use_planck_h0_prior,
        use_planck_om_prior=use_planck_om_prior,
        use_alpha_lambda_term=use_alpha_lambda_term,
        use_eta_sigma_term=use_eta_sigma_term,
        use_redshift_log_f_term=use_redshift_log_f_term,
        use_redshift_mu_term=use_redshift_mu_term,
        require_selection_fields=completeness_params is not None,
    )
    z = np.asarray(agn_data["z"], dtype=float)
    ll_agn = _normal_logpdf_sum(
        prediction["mu_pred"] - prediction["mu_model"],
        prediction["total_error"],
    )

    ll_completeness = 0.0
    comp_blob = empty_blob(N_obj)
    if completeness_params is not None:
        if selection_attenuation_mode == "fixed-offset":
            ll_completeness, comp_blob = completeness_loglike_for_data(
                completeness_params=completeness_params,
                agn_data=agn_data,
                m_obs=prediction["selection_magnitude"],
                m_obs_err=prediction["selection_magnitude_error"],
                m_model=prediction["selection_model_magnitude"],
                mu_err=prediction["selection_total_error"],
                z=z,
            )
        else:
            ll_completeness, comp_blob = (
                joint_posterior_completeness_loglike_for_data(
                    completeness_params=completeness_params,
                    agn_data=agn_data,
                    hubble_magnitude=np.asarray(
                        agn_data["apparent_mag_2500"], dtype=float
                    ),
                    hubble_magnitude_error=np.asarray(
                        agn_data["apparent_mag_2500_err"], dtype=float
                    ),
                    hubble_model_magnitude=prediction["model_magnitude"],
                    hubble_total_error=prediction["total_error"],
                    z=z,
                )
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
    use_redshift_mu_term=False,
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
        use_redshift_mu_term=use_redshift_mu_term,
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
    delta_mu_z_nc = evaluate_mu_redshift_term(
        params,
        z_nc,
        z_pivot_agn,
        use_redshift_mu_term=use_redshift_mu_term,
    )
    mu_model_nc = mu_cosmo_nc + delta_mu_z_nc

    sigma_lens = sigma_lens_from_dc(z_nc, cosmo)   # vector (same shape as z)
    sigma_mu_z_nc = sigma_mu_model_from_z_err(
        z_nc,
        z_err_nc,
        cosmo,
        params,
        z_pivot=z_pivot_agn,
        use_redshift_mu_term=use_redshift_mu_term,
    )

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

    ll_agn_noncal = _normal_logpdf_sum(mu_pred_nc - mu_model_nc, mu_err_nc)

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
        z_c = np.asarray(agn_calibrators_data['z'][cal_mask_tbl], dtype=float)
        delta_mu_z_c = evaluate_mu_redshift_term(
            params,
            z_c,
            z_pivot_agn,
            use_redshift_mu_term=use_redshift_mu_term,
        )
        sigma_delta_mu_z_c = np.zeros_like(z_c)
        if use_redshift_mu_term:
            calibrator_z_err = agn_calibrators_data.get(
                'z_err', np.zeros(len(agn_calibrators_data['z']), dtype=float)
            )
            z_err_c = np.asarray(calibrator_z_err, dtype=float)[cal_mask_tbl]
            z_lo_c = np.maximum(z_c - z_err_c, 1e-8)
            z_hi_c = np.maximum(z_c + z_err_c, z_lo_c + 1e-8)
            delta_lo_c = evaluate_mu_redshift_term(
                params, z_lo_c, z_pivot_agn, use_redshift_mu_term=True
            )
            delta_hi_c = evaluate_mu_redshift_term(
                params, z_hi_c, z_pivot_agn, use_redshift_mu_term=True
            )
            sigma_delta_mu_z_c = np.where(
                np.isfinite(z_err_c) & (z_err_c > 0.0),
                0.5 * np.abs(delta_hi_c - delta_lo_c),
                0.0,
            )

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
            mu_cal_err**2 +
            sigma_delta_mu_z_c**2
        )

        ll_agn_cal = _normal_logpdf_sum(
            mu_pred_c - (mu_cal + delta_mu_z_c), mu_err_c
        )
    else:
        raise ValueError("No calibrator AGN found in agn_calibrators_data where AGN_IS_CALIBRATOR is True.")
        ll_agn_cal = 0.0

    # ========================
    # 3) COMPLETENESS (non-calibrators only)
    # ========================
    ll_completeness = 0.0
    comp_blob = empty_blob(N_obj)
    if completeness_params is not None and np.any(mask_noncal):
        # model-predicted magnitude for non-calibrators (cosmo-anchored for selection)
        m_model_nc = M_pred_nc + mu_model_nc
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
            ll_completeness, noncal_blob = completeness_loglike_for_data(
                completeness_params=completeness_params,
                agn_data=agn_data_nc,
                m_obs=selection_magnitude_nc,
                m_obs_err=selection_magnitude_error_nc,
                m_model=selection_model_magnitude_nc,
                mu_err=selection_total_error_nc,
                z=z_nc,
            )
        else:
            ll_completeness, noncal_blob = (
                joint_posterior_completeness_loglike_for_data(
                    completeness_params=completeness_params,
                    agn_data=agn_data_nc,
                    hubble_magnitude=m_obs_nc,
                    hubble_magnitude_error=m_err_nc,
                    hubble_model_magnitude=m_model_nc,
                    hubble_total_error=mu_err_nc,
                    z=z_nc,
                )
            )
        comp_blob[:, mask_noncal] = noncal_blob

    # ========================
    # 4) Total
    # ========================
    ll = ll_agn_noncal + ll_agn_cal - ll_completeness
    return ll, comp_blob
