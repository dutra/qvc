import numpy as np
import h5py
import os
from scipy import stats
from scipy.stats import norm, sigmaclip, multivariate_normal
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import interp1d

prefix = os.environ.get("PREFIX", "")

class SimpleCompleteness2D:
    """
    Simple analytic completeness: sigmoid dropoff in apparent magnitude.
    Completeness is 1 at bright mags, drops to 0 at faint mags.
    Constant in redshift (z).
    """
    def __init__(self, mag_lim=24.0, width=0.2):
        self.mag_lim = mag_lim
        self.width = width

    def __call__(self, mag, z=None):
        mag = np.asarray(mag)
        return 1.0 / (1.0 + np.exp((mag - self.mag_lim) / self.width))

    @property
    def grid(self):
        return dict(mag_lim=self.mag_lim, width=self.width)

def get_completeness_function_2d_simple(*args, mag_lim=24.0, width=0.2, plot=False, **kwargs):
    """
    Drop-in replacement that returns a simple analytic completeness function:
        p(detect | mag, z) = 1 / (1 + exp((mag - mag_lim) / width))

    Ignores all data inputs. Hard-coded mag grid: 16 to 30.
    """
    completeness2d = SimpleCompleteness2D(mag_lim=mag_lim, width=width)

    # Fixed mag grid: 16 to 30
    mag_min, mag_max = 16.0, 30.0
    mag_centers = np.linspace(mag_min, mag_max, 300)
    z_centers = np.linspace(0.0, 3.0, 30)  # dummy grid for API compatibility
    dm = mag_centers[1] - mag_centers[0]
    dz = z_centers[1] - z_centers[0]

    if plot:
        completeness_vals = completeness2d(mag_centers)
        os.makedirs(f"plots/hubble/{prefix}/completeness", exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.plot(mag_centers, completeness_vals, label="Completeness")
        plt.axvline(mag_lim, color='r', linestyle='--', label=f"mag_lim = {mag_lim}")
        plt.xlabel("Apparent Magnitude")
        plt.ylabel("p(detect)")
        plt.title("Simple Analytic Completeness Function")
        plt.ylim(-0.05, 1.05)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/hubble/{prefix}/completeness/simple_completeness_function.png", dpi=200)
        plt.close()

    return completeness2d, mag_centers, z_centers, dm, dz

class Completeness2D:
    """
    Interpolates p(detect | m, z) on a (mag, z) grid.
    - Outside the grid, returns 0 (via RegularGridInterpolator fill_value=0).
    """
    def __init__(self, mag_centers, z_centers, completeness_map):
        self.mag_centers = np.asarray(mag_centers)
        self.z_centers   = np.asarray(z_centers)

        # Ensure finite in [0,1]
        C = np.nan_to_num(completeness_map, nan=0.0, posinf=0.0, neginf=0.0)
        C = np.clip(C, 0.0, 1.0).astype(float)

        self.mag_min, self.mag_max = float(self.mag_centers[0]),  float(self.mag_centers[-1])
        self.z_min,   self.z_max   = float(self.z_centers[0]),    float(self.z_centers[-1])

        # No clipping before interpolation; rely on fill_value=0 for out-of-bounds
        self._interp = RegularGridInterpolator(
            (self.mag_centers, self.z_centers),
            C,
            bounds_error=False,
            fill_value=0.0,
        )

    def __call__(self, mag, z):
        mag = np.asarray(mag)
        z   = np.asarray(z)
        m_b, z_b = np.broadcast_arrays(mag, z)
        pts = np.column_stack([m_b.ravel(), z_b.ravel()])
        vals = self._interp(pts)
        return vals.reshape(m_b.shape)

    @property
    def grid(self):
        return dict(mag_centers=self.mag_centers, z_centers=self.z_centers)

import numpy as np

def m2500_from_mi_broken(
    mi_obs, z,
    alpha_lam_blue, alpha_lam_red,
    lambda_i_eff_obs=7480.0,   # Å, SDSS i effective wavelength (observer-frame)
    lambda_break=5000.0        # Å, rest-frame break
):
    raise NotImplementedError("This function has been moved to qvc/hubble_utils.py")

    """
    Convert OBSERVER-frame i-band AB magnitude (mi_obs) to REST-frame m_2500 (AB)
    assuming a broken power law in f_lambda with a rest-frame break.
    """
    mi_obs         = np.asarray(mi_obs, dtype=float)
    z              = np.asarray(z, dtype=float)
    alpha_lam_blue = np.asarray(alpha_lam_blue, dtype=float)
    alpha_lam_red  = np.asarray(alpha_lam_red,  dtype=float)
    lambda_break   = np.asarray(lambda_break,   dtype=float)

    lam_i_rest = float(lambda_i_eff_obs) / (1.0 + z)

    ratio_blue = (2500.0 / lam_i_rest) ** (2.0 + alpha_lam_blue)
    ratio_red  = ((2500.0 / lam_i_rest) ** (2.0 + alpha_lam_red) *
                  (2500.0 / lambda_break) ** (alpha_lam_blue - alpha_lam_red))

    use_blue = lam_i_rest <= lambda_break
    ratio = np.where(use_blue, ratio_blue, ratio_red)

    m2500_rest = mi_obs - 2.5 * np.log10((1.0 + z) * ratio)
    return m2500_rest



import numpy as np
from types import SimpleNamespace

def fit_m2500_predictor(
    df_agn,
    lambda_i_eff=7480.0,              # Å, REST-frame i effective wavelength
    valid_mag_range=(1.0, 30.0),
    method="huber",                   # 'ridge' | 'elasticnet' | 'huber' | 'wls' | 'gp_resid'
    sample_weight=None,               # optional 1/var weights for WLS (same length as df_agn)
    verbose=True,
):
    """
    Physics-aware predictor with selectable calibration back-end.

    Returns
    -------
    predictor : SimpleNamespace with fields
        lambda_i_eff, A,B,D,C_pivot, y0, alpha0, sigma, r2, backend, and .predict(...)
        If method == 'gp_resid', also includes gp_ (sklearn GP) and scaler_.
    info : dict with arrays and diagnostics
    """

    # ---- Pull columns
    z    = df_agn["z"].astype(float)
    
    # old
    mi_rest   = df_agn['apparent_mag_i_rest'].astype(float)

    # mew
    # mi_obs   = df_agn['apparent_mag_i_obs'].astype(float)
    # alpha_lam_red = df_agn['PL_slope_red'].astype(float)
    # alpha_lam_blue = df_agn['PL_slope_blue'].astype(float)
    # lambda_break = df_agn['PL_break_wave'].astype(float)

    # lambda_i_eff_obs = float(lambda_i_eff)
    # alpha_i = np.where(lambda_i_eff_obs/(1.0+z) <= lambda_break, alpha_lam_blue, alpha_lam_red)

    #mi_rest = mi_obs - 2.5 * (alpha_i + 1.0) * np.log10(1.0 + z)
    
    mi = mi_rest
    mi_err = df_agn['apparent_mag_i_rest_err'].astype(float)
    y    = df_agn["apparent_mag_2500"].astype(float)
    y_err = df_agn["apparent_mag_2500_err"].astype(float)
    abl  = df_agn["PL_slope_blue"].astype(float)
    ard  = df_agn["PL_slope_red"].astype(float)
    lbrk = df_agn["PL_break_wave"].astype(float)

    # ---- Mask
    lo, hi = valid_mag_range
    mask = (
        y.notna() & mi.notna() & z.notna() &
        abl.notna() & ard.notna() & lbrk.notna() &
        y.between(lo, hi) & mi.between(lo, hi)
    ).to_numpy()
    if mask.sum() < 6:
        raise ValueError("Not enough valid objects (need >= 6).")

    mi_arr   = mi.to_numpy()[mask]
    y_arr    = y.to_numpy()[mask]
    z_arr    = z.to_numpy()[mask]
    abl_arr  = abl.to_numpy()[mask]
    ard_arr  = ard.to_numpy()[mask]
    lbrk_arr = lbrk.to_numpy()[mask]

    w = None

    # ---- Physics prediction
    # y_phys = m2500_from_mi_broken(
    #     mi_arr, z_arr, abl_arr, ard_arr,
    #     lambda_i_eff_obs=float(lambda_i_eff),
    #     lambda_break=lbrk_arr
    # )
    y_phys = mi_arr

    # ---- Pivots and design matrix
    y0     = float(np.mean(y_phys))
    alpha0 = float(np.mean(abl_arr))
    dyp    = (y_phys - y0)
    dblue  = (abl_arr - alpha0)
    X_lin  = np.column_stack([dyp, dalph, dyp**2])   # no constant here
    ones   = np.ones_like(dyp)

    # Helper to compute diagnostics
    def _diagnostics(y_true, y_pred, p):
        resid = y_true - y_pred
        rss = float(np.sum(resid**2 if w is None else (w * resid**2)))
        tss = float(np.sum((y_true - np.average(y_true, weights=None if w is None else w))**2 if w is None else (w * (y_true - np.average(y_true, weights=w))**2)))
        r2  = 1.0 - rss / tss if tss > 0 else np.nan
        dof = max(len(y_true) - p, 1)
        # If weighted, σ is approximate RMS of residuals (weights ignored in denom for simplicity)
        sigma = float(np.sqrt(np.sum(resid**2) / dof))
        return resid, r2, sigma

    # ---- Backends
    backend = method.lower()

    if backend == "ridge":
        from sklearn.linear_model import RidgeCV
        alphas = np.logspace(-4, 2, 100)
        # add constant explicitly, keep fit_intercept=False
        X = np.column_stack([X_lin, ones])
        ridge = RidgeCV(alphas=alphas, fit_intercept=False, store_cv_results=True)
        ridge.fit(X, y_arr, sample_weight=w)
        A, B, D, C_pivot = ridge.coef_.astype(float)
        y_fit = X @ ridge.coef_.astype(float)
        resid, r2, sigma = _diagnostics(y_arr, y_fit, p=X.shape[1])
        alpha_cv = float(ridge.alpha_)
        se = None  # (optional) could compute from (X^T X)^{-1} if desired


    elif backend == "huber":
        from sklearn.linear_model import HuberRegressor
        # let Huber learn an intercept; do NOT add 'ones' column
        huber = HuberRegressor(alpha=0.0, fit_intercept=True)  # alpha=0.0 ~ no L2 shrink
        huber.fit(X_lin, y_arr, sample_weight=w)
        A, B, D = huber.coef_.astype(float)
        C_pivot = float(huber.intercept_)
        y_fit = huber.predict(X_lin)
        resid, r2, sigma = _diagnostics(y_arr, y_fit, p=X_lin.shape[1] + 1)
        alpha_cv = None
        se = None

    else:
        raise ValueError("method must be one of: 'ridge', 'elasticnet', 'huber', 'wls', 'gp_resid'.")

    # ---- Build predictor wrapper
    pred = SimpleNamespace(
        backend=backend,
        lambda_i_eff=float(lambda_i_eff),
        A=float(A), B=float(B), D=float(D), C_pivot=float(C_pivot),
        y0=float(y0), alpha0=float(alpha0),
        sigma=float(sigma), r2=float(r2),
        alpha_cv=alpha_cv, se=se
    )

    def _predict(mi_in, z_in, alpha_lam_blue, alpha_lam_red, lambda_break,
                 *, A=A, B=B, D=D, C_pivot=C_pivot, y0=y0, alpha0=alpha0,
                 lambda_i_eff=float(lambda_i_eff), backend=backend, pred_ns=pred):

        dyp_   = (np.asarray(mi_in, dtype=float) - y0)
        dalph_ = (np.asarray(alpha_lam_blue, dtype=float) - alpha0)
        y_lin  = A * dyp_ + B * dalph_ + D * (dyp_ ** 2) + C_pivot
        return y_lin

    pred.predict = _predict
    if verbose:
        print(f"Coefficients: A={A:.4f}, B={B:.4f}, D={D:.4f}, C_pivot={C_pivot:.4f}")
        print(f"Sigma: {sigma:.4f}")

    info = dict(
        mask=mask, y=y_arr, mi=mi_arr, z=z_arr,
        alpha_lam_blue=abl_arr, alpha_lam_red=ard_arr, lambda_break=lbrk_arr,
        y_phys=y_phys,
        pivots=dict(y0=y0, alpha0=alpha0),
        coefs=dict(A=float(A), B=float(B), D=float(D), C_pivot=float(C_pivot)),
        backend=backend,
        y_fit=y_fit,
        residuals=(y_arr - y_fit),
        resid=resid,
        r2=float(r2),
        sigma=float(sigma),
        alpha_cv=alpha_cv,
        se=se,
    )
    return pred, info


def get_completeness_function_2d_old(
    df_agn,
    sim_file="data/mock_mag_z.h5",
    n_mag_bins=40, n_z_bins=20,
    sigma_z=0.5,                  # smoothing in BIN units along z
    smooth_counts=True,
    plot=False,
    method="ridge"
):
    """
    Build p(detect | m, z) using the *ridge-calibrated broken-law* predictor.

    Steps:
      1) Fit calibrated predictor (physics -> ridge y ≈ A*y_phys + B).
      2) Predict 'true' m_2500 for the simulation using MEDIANS of (PL_slope_blue, PL_slope_red, PL_break_wave)
         from the fitted mask, apply the same calibration A,B.
      3) Smooth raw counts (not the ratio) with Gaussian sigma=(sigma_fit, sigma_z),
         where sigma_fit is the fitted RMS scatter (mag).
      4) Return Completeness2D and grid info.
    """
    import os, h5py
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    predictor, info = fit_m2500_predictor(
        df_agn,
        lambda_i_eff=7480.0,
        valid_mag_range=(1.0, 30.0),
        method=method,
        verbose=True,
    )

    y_obs  = np.asarray(info["y"], dtype=float)
    z_obs  = np.asarray(info["z"], dtype=float)
    y_fit  = np.asarray(info["y_fit"], dtype=float)
    sigma_fit = float(predictor.sigma)

    # Simulation inputs: rest-frame mi and z
    with h5py.File(sim_file, "r") as f:
        mi_true = np.asarray(f["apparent_mag_i_rest"][:], dtype=float)
        z_true  = np.asarray(f["z"][:], dtype=float)

    # Use medians of PL parameters from training mask
    msk = info["mask"]

    y_true = predictor.predict(
        mi_in=mi_true, 
        z_in=z_true,
        alpha_lam_blue=-1.5,
        alpha_lam_red=-1.5,
        lambda_break=5000.0
    )

    # Filter finite
    ok_obs  = np.isfinite(y_obs) & np.isfinite(z_obs)
    ok_true = np.isfinite(y_true) & np.isfinite(z_true)
    m_obs,  z_obs  = y_obs[ok_obs],  z_obs[ok_obs]
    m_true, z_true = y_true[ok_true], z_true[ok_true]

    # Grid
    mag_min, mag_max = 16.0, 28.0
    z_min,   z_max   = float(np.min(z_true)), 5.0
    mag_edges = np.linspace(mag_min, mag_max, n_mag_bins + 1)
    z_edges   = np.linspace(z_min,  z_max,    n_z_bins   + 1)
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    z_centers   = 0.5 * (z_edges[:-1]   + z_edges[1:])
    dm = float(mag_centers[1] - mag_centers[0]) if len(mag_centers) > 1 else float(mag_edges[-1] - mag_edges[0])
    dz = float(z_centers[1] - z_centers[0])     if len(z_centers)   > 1 else float(z_edges[-1] - z_edges[0])

    # 2D histograms on [mag, z]
    H_true, _, _ = np.histogram2d(m_true, z_true, bins=[mag_edges, z_edges])
    H_obs,  _, _ = np.histogram2d(m_obs,  z_obs,  bins=[mag_edges, z_edges])

    # Smooth COUNTS (not ratio); mag smoothing uses sigma_fit (mag)
    if smooth_counts:
        sig_mag_pix = max(sigma_fit / dm, 1e-6)      # scatter is in mag
        H_true_s = gaussian_filter(H_true, sigma=(float(sig_mag_pix), float(sigma_z)),
                                   mode="constant", cval=0.0)
        H_obs_s  = gaussian_filter(H_obs,  sigma=(float(sig_mag_pix), float(sigma_z)),
                                   mode="constant", cval=0.0)
    else:
        H_true_s, H_obs_s = H_true, H_obs

    eps = 1e-12
    C = H_obs_s / (H_true_s + eps)
    C[H_true_s < eps] = 0.0
    C = np.clip(C, 0.0, 1.0)

    if plot:
        try:
            plot_dir = f"plots/hubble/{prefix}/completeness"
        except NameError:
            plot_dir = "plots/hubble/completeness"
        os.makedirs(plot_dir, exist_ok=True)

        plt.figure(figsize=(7, 6))
        lo, hi = float(np.min(y_obs)), float(np.max(y_obs))
        plt.scatter(y_obs, y_fit, s=4, alpha=0.4, label="Data")
        plt.plot([lo, hi], [lo, hi], "r--", label="y = y_fit")
        plt.xlabel("Observed $m_{2500}$")
        plt.ylabel("Predicted $m_{2500}$")
        plt.title("Observed vs Predicted $m_{2500}$")
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "y_vs_yfit.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(7, 5))
        im = plt.imshow(
            np.log10(np.clip(C.T, 1e-12, None)), origin="lower", aspect="auto",
            extent=[mag_edges[0], mag_edges[-1], z_edges[0], z_edges[-1]], cmap="viridis"
        )
        plt.ylabel(r"$z$")
        plt.xlabel(r"$m_{2500\,\text{\AA}} \; (\mathrm{mag})$")
        cbar = plt.colorbar(im); cbar.set_label(r"Completeness $p(I{=}1\,|\,m,z)$")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "completeness_map.png"), dpi=200)
        plt.savefig(os.path.join(plot_dir, "completeness_map.pdf"), dpi=600)
        plt.close()

    return Completeness2D(mag_centers, z_centers, C), mag_centers, z_centers, dm, dz, sigma_fit, info, None, None

def get_completeness_function_2d(
    df_agn,
    sim_file="data/mock_mag_z.h5",
    n_mag_bins=40, n_z_bins=20,
    sigma_z=0.5,                  # smoothing in BIN units along z
    smooth_counts=True,
    plot=False,
):
    """
    Build p(detect | m, z)
    """
    import os, h5py
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    # Simulation inputs: rest-frame mi and z
    with h5py.File(sim_file, "r") as f:
        m_true = np.asarray(f["apparent_mag_i_rest"][:], dtype=float)
        z_true  = np.asarray(f["z"][:], dtype=float)

    # Filter finite
    z_obs = df_agn["z"].to_numpy(dtype=float)
    m_obs = df_agn["apparent_mag_2500"].to_numpy(dtype=float)

    ok_obs  = np.isfinite(m_obs) & np.isfinite(z_obs)
    ok_true = np.isfinite(m_true) & np.isfinite(z_true)
    m_obs,  z_obs  = m_obs[ok_obs],  z_obs[ok_obs]
    m_true, z_true = m_true[ok_true], z_true[ok_true]

    # Grid
    mag_min, mag_max = 16.0, 28.0
    z_min,   z_max   = float(np.min(z_true)), 5.0
    mag_edges = np.linspace(mag_min, mag_max, n_mag_bins + 1)
    z_edges   = np.linspace(z_min,  z_max,    n_z_bins   + 1)
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    z_centers   = 0.5 * (z_edges[:-1]   + z_edges[1:])
    dm = float(mag_centers[1] - mag_centers[0]) if len(mag_centers) > 1 else float(mag_edges[-1] - mag_edges[0])
    dz = float(z_centers[1] - z_centers[0])     if len(z_centers)   > 1 else float(z_edges[-1] - z_edges[0])

    # 2D histograms on [mag, z]
    H_true, _, _ = np.histogram2d(m_true, z_true, bins=[mag_edges, z_edges])
    H_obs,  _, _ = np.histogram2d(m_obs,  z_obs,  bins=[mag_edges, z_edges])

    # Smooth COUNTS (not ratio); mag smoothing uses sigma_fit (mag)
    if smooth_counts:
        sigma_fit = 0.1  # fixed smoothing value
        sig_mag_pix = max(sigma_fit / dm, 1e-6)      # scatter is in mag
        H_true_s = gaussian_filter(H_true, sigma=(float(sig_mag_pix), float(sigma_z)),
                                   mode="constant", cval=0.0)
        H_obs_s  = gaussian_filter(H_obs,  sigma=(float(sig_mag_pix), float(sigma_z)),
                                   mode="constant", cval=0.0)
    else:
        H_true_s, H_obs_s = H_true, H_obs

    eps = 1e-12
    C = H_obs_s / (H_true_s + eps)
    C[H_true_s < eps] = 0.0
    C = np.clip(C, 0.0, 1.0)

    if plot:
        try:
            plot_dir = f"plots/hubble/{prefix}/completeness"
        except NameError:
            plot_dir = "plots/hubble/completeness"
        os.makedirs(plot_dir, exist_ok=True)


        plt.figure(figsize=(7, 5))
        im = plt.imshow(
            np.log10(np.clip(C.T, 1e-12, None)), origin="lower", aspect="auto",
            extent=[mag_edges[0], mag_edges[-1], z_edges[0], z_edges[-1]], cmap="viridis"
        )
        plt.ylabel(r"$z$")
        plt.xlabel(r"$m_{2500\,\text{\AA}} \; (\mathrm{mag})$")
        cbar = plt.colorbar(im); cbar.set_label(r"Completeness $p(I{=}1\,|\,m,z)$")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "completeness_map.png"), dpi=200)
        plt.savefig(os.path.join(plot_dir, "completeness_map.pdf"), dpi=600)
        plt.close()

    return Completeness2D(mag_centers, z_centers, C), mag_centers, z_centers, dm, dz, sigma_fit, None, None, None

import numpy as np
from scipy.interpolate import RegularGridInterpolator

def make_dm_function(m, z, dm, m_bins=40, z_bins=40, *, method='linear'):
    """
    Build a 2D interpolator dm(m,z) defined on bin midpoints.
    Queries are always clipped to the grid range (no extrapolation).
    """
    # Remove non-finite values
    m = np.asarray(m)
    z = np.asarray(z)
    dm = np.asarray(dm)
    #mask = np.isfinite(m) & np.isfinite(z) & np.isfinite(dm)
    m, z, dm = m[np.isfinite(m)], z[np.isfinite(z)], dm[np.isfinite(dm)]

    # Build bin edges
    m_edges = np.linspace(m.min(), m.max(), m_bins) if np.isscalar(m_bins) else np.asarray(m_bins)
    z_edges = np.linspace(z.min(), z.max(), z_bins) if np.isscalar(z_bins) else np.asarray(z_bins)

    # 2D binning: means per cell
    counts, _, _ = np.histogram2d(z, m, bins=[z_edges, m_edges])
    sums,   _, _ = np.histogram2d(z, m, bins=[z_edges, m_edges], weights=dm)
    mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

    # Grid points are the bin midpoints
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
    m_mid = 0.5 * (m_edges[:-1] + m_edges[1:])

    # Core interpolator (will return NaN outside, we’ll clip inputs before calling it)
    interp_core = RegularGridInterpolator(
        (z_mid, m_mid), mean,
        method=method, bounds_error=False, fill_value=np.nan
    )

    # Clipping wrapper
    z_lo, z_hi = z_mid.min(), z_mid.max()
    m_lo, m_hi = m_mid.min(), m_mid.max()

    def interp_clipped(pts):
        pts = np.asarray(pts)
        arr = np.atleast_2d(pts).astype(float)
        arr[:, 0] = np.clip(arr[:, 0], z_lo, z_hi)
        arr[:, 1] = np.clip(arr[:, 1], m_lo, m_hi)
        out = interp_core(arr)
        return out if np.ndim(pts) > 1 else out[0]

    return interp_clipped

def estimate_m50(bin_edges, true_counts, det_counts, ax=None):
    """
    Estimate the magnitude where completeness drops to 50%.
    Parameters
    ----------
    bin_edges : array-like
        Magnitude bin edges (len B+1).
    true_counts : array-like
        True/input counts per bin (len B).
    det_counts : array-like
        Detected counts per bin (len B).
    ax : matplotlib Axes, optional
        Axis to plot on. If None, a new figure is created.
    Returns
    -------
    m50 : float
        Magnitude at 50% completeness.
    w : float
        Width parameter of the logistic fit.
    """
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    completeness = det_counts / np.maximum(true_counts, 1)
    # Logistic function
    def logistic(m, m50, w):
        return 1.0 / (1.0 + np.exp((m - m50) / w))
    # Initial guess: halfway point
    m_guess = bin_centers[np.argmin(np.abs(completeness - 0.5))]
    w_guess = 0.3
    popt, pcov = curve_fit(logistic, bin_centers, completeness, p0=[m_guess, w_guess], bounds=([bin_edges.min(), 0.01],[bin_edges.max(), 5.0]))
    m50, w = popt
    # Plot
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(bin_centers, completeness, label="Observed completeness", color="k")
    m_fit = np.linspace(bin_edges.min(), bin_edges.max(), 200)
    ax.plot(m_fit, logistic(m_fit, *popt), 'r-', label=f"Fit: m50={m50:.2f}")
    ax.axhline(0.5, color="gray", ls="--")
    ax.axvline(m50, color="red", ls="--")
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Completeness")
    ax.set_ylim(0, 1.05)
    ax.legend()
    return m50, w