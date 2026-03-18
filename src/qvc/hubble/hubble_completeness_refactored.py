import json
import numpy as np
import h5py
import os
from astropy.cosmology import FlatLambdaCDM
from scipy import stats
from scipy.stats import norm, sigmaclip, multivariate_normal
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.special import expit, logit
from functools import partial

from qvc.hubble.hubble_utils import convert_M2500_to_logL2500, resolve_qvc_data_path


COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)

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

def get_completeness_function_2d_simple(*args, mag_lim=24.0, width=0.2, plot=False, plot_path=None, **kwargs):
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
        base_plot_path = plot_path or "plots/hubble"
        completeness_path = os.path.join(base_plot_path, "completeness")
        os.makedirs(completeness_path, exist_ok=True)

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
        plt.savefig(os.path.join(completeness_path, "simple_completeness_function.pdf"), dpi=600)
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

    @property
    def mode(self):
        return "2d"


class Completeness3D:
    """
    Interpolates p(detect | m, z, f_host) on a (mag, z, f_host) grid.
    - Outside the grid, returns 0.
    """

    def __init__(self, mag_centers, z_centers, fhost_centers, completeness_cube):
        self.mag_centers = np.asarray(mag_centers)
        self.z_centers = np.asarray(z_centers)
        self.fhost_centers = np.asarray(fhost_centers)

        C = np.nan_to_num(completeness_cube, nan=0.0, posinf=0.0, neginf=0.0)
        C = np.clip(C, 0.0, 1.0).astype(float)

        self.mag_min, self.mag_max = float(self.mag_centers[0]), float(self.mag_centers[-1])
        self.z_min, self.z_max = float(self.z_centers[0]), float(self.z_centers[-1])
        self.fhost_min, self.fhost_max = float(self.fhost_centers[0]), float(self.fhost_centers[-1])

        self._interp = RegularGridInterpolator(
            (self.mag_centers, self.z_centers, self.fhost_centers),
            C,
            bounds_error=False,
            fill_value=0.0,
        )

    def __call__(self, mag, z, f_host):
        mag = np.asarray(mag)
        z = np.asarray(z)
        f_host = np.asarray(f_host)
        # The completeness cube is defined on bin centers, but f_host is a
        # bounded physical variable on [0, 1]. Clip to the nearest supported
        # center so values very close to 0 or 1 do not spuriously get
        # fill_value=0 from the interpolator.
        f_host = np.clip(f_host, self.fhost_min, self.fhost_max)
        m_b, z_b, f_b = np.broadcast_arrays(mag, z, f_host)
        pts = np.column_stack([m_b.ravel(), z_b.ravel(), f_b.ravel()])
        vals = self._interp(pts)
        return vals.reshape(m_b.shape)

    @property
    def grid(self):
        return dict(
            mag_centers=self.mag_centers,
            z_centers=self.z_centers,
            fhost_centers=self.fhost_centers,
        )

    @property
    def mode(self):
        return "3d_fhost"


class Completeness4D:
    """
    Interpolates p(detect | m, z, f_host, alpha_lambda) on a regular grid.
    - Outside the grid, returns 0.
    """

    def __init__(self, mag_centers, z_centers, fhost_centers, alpha_centers, completeness_hypercube):
        self.mag_centers = np.asarray(mag_centers)
        self.z_centers = np.asarray(z_centers)
        self.fhost_centers = np.asarray(fhost_centers)
        self.alpha_centers = np.asarray(alpha_centers)

        C = np.nan_to_num(completeness_hypercube, nan=0.0, posinf=0.0, neginf=0.0)
        C = np.clip(C, 0.0, 1.0).astype(float)

        self.mag_min, self.mag_max = float(self.mag_centers[0]), float(self.mag_centers[-1])
        self.z_min, self.z_max = float(self.z_centers[0]), float(self.z_centers[-1])
        self.fhost_min, self.fhost_max = float(self.fhost_centers[0]), float(self.fhost_centers[-1])
        self.alpha_min, self.alpha_max = float(self.alpha_centers[0]), float(self.alpha_centers[-1])

        self._interp = RegularGridInterpolator(
            (self.mag_centers, self.z_centers, self.fhost_centers, self.alpha_centers),
            C,
            bounds_error=False,
            fill_value=0.0,
        )

    def __call__(self, mag, z, f_host, alpha_lambda):
        mag = np.asarray(mag)
        z = np.asarray(z)
        f_host = np.asarray(f_host)
        alpha_lambda = np.asarray(alpha_lambda)
        f_host = np.clip(f_host, self.fhost_min, self.fhost_max)
        alpha_lambda = np.clip(alpha_lambda, self.alpha_min, self.alpha_max)
        m_b, z_b, f_b, a_b = np.broadcast_arrays(mag, z, f_host, alpha_lambda)
        pts = np.column_stack([m_b.ravel(), z_b.ravel(), f_b.ravel(), a_b.ravel()])
        vals = self._interp(pts)
        return vals.reshape(m_b.shape)

    @property
    def grid(self):
        return dict(
            mag_centers=self.mag_centers,
            z_centers=self.z_centers,
            fhost_centers=self.fhost_centers,
            alpha_centers=self.alpha_centers,
        )

    @property
    def mode(self):
        return "4d_fhost_alpha"


_FHOST_CLIP_EPS = 1e-3
_ALPHA_TRUE_MEAN = -1.5
_ALPHA_TRUE_SIGMA = 0.5
_ALPHA_MIN = -4.0
_ALPHA_MAX = 0.5


def generalized_sigmoid_fhost(logL2500, x0, k, nu):
    arg = np.clip(k * (np.asarray(logL2500, dtype=float) - x0), -60.0, 60.0)
    return 1.0 / np.power(1.0 + np.exp(arg), nu)


def apparent_mag_to_logL2500(m2500, z, cosmo):
    m2500 = np.asarray(m2500, dtype=float)
    z = np.asarray(z, dtype=float)
    M2500 = m2500 - cosmo.distmod(z).value
    return convert_M2500_to_logL2500(M2500)


def fit_fhost_center_l2500_model(
    df_agn,
    *,
    fit_logL_max=45.5,
    clip_eps=_FHOST_CLIP_EPS,
    cosmo=COSMO,
):
    required = {"z", "apparent_mag_2500", "f_host_center"}
    if not required.issubset(df_agn.columns):
        missing = ", ".join(sorted(required - set(df_agn.columns)))
        raise KeyError(f"Missing required columns for f_host model fit: {missing}")

    z = np.asarray(df_agn["z"], dtype=float)
    m2500 = np.asarray(df_agn["apparent_mag_2500"], dtype=float)
    f_host = np.asarray(df_agn["f_host_center"], dtype=float)
    logL2500 = apparent_mag_to_logL2500(m2500, z, cosmo)

    fit_mask = (
        np.isfinite(logL2500)
        & np.isfinite(f_host)
        & np.isfinite(z)
        & (z > 0.0)
        & (f_host >= 0.0)
        & (f_host <= 1.0)
        & (logL2500 <= fit_logL_max)
    )
    if np.count_nonzero(fit_mask) < 8:
        raise ValueError("Need at least 8 finite rows to fit the f_host_center(log L_2500) model.")

    x_fit = logL2500[fit_mask]
    y_fit = np.clip(f_host[fit_mask], clip_eps, 1.0 - clip_eps)
    p0 = (float(np.nanmedian(x_fit)), 2.0, 1.0)
    bounds = (
        [float(np.nanmin(x_fit)), 0.01, 0.1],
        [float(np.nanmax(x_fit)), 20.0, 10.0],
    )
    popt, _ = curve_fit(
        generalized_sigmoid_fhost,
        x_fit,
        y_fit,
        p0=p0,
        bounds=bounds,
        maxfev=20000,
    )

    mean_fit = np.clip(generalized_sigmoid_fhost(x_fit, *popt), clip_eps, 1.0 - clip_eps)
    residual_logit = logit(y_fit) - logit(mean_fit)
    sigma_host_logit = float(np.nanstd(residual_logit, ddof=1))
    if not np.isfinite(sigma_host_logit):
        sigma_host_logit = 0.0
    sigma_host_logit = max(sigma_host_logit, 1e-6)

    return {
        "x0": float(popt[0]),
        "k": float(popt[1]),
        "nu": float(popt[2]),
        "sigma_host_logit": sigma_host_logit,
        "fit_logL_max": float(fit_logL_max),
        "clip_eps": float(clip_eps),
        "n_fit": int(np.count_nonzero(fit_mask)),
    }


def predict_fhost_center_from_logL2500(logL2500, model):
    clip_eps = float(model.get("clip_eps", _FHOST_CLIP_EPS))
    mean = generalized_sigmoid_fhost(logL2500, model["x0"], model["k"], model["nu"])
    return np.clip(mean, clip_eps, 1.0 - clip_eps)


def sample_fhost_center_from_logL2500(logL2500, model, rng):
    mean = predict_fhost_center_from_logL2500(logL2500, model)
    sigma_host_logit = float(model.get("sigma_host_logit", 0.0))
    if sigma_host_logit <= 0.0:
        return mean
    sampled_logit = logit(mean) + rng.normal(0.0, sigma_host_logit, size=np.shape(mean))
    clip_eps = float(model.get("clip_eps", _FHOST_CLIP_EPS))
    return np.clip(expit(sampled_logit), clip_eps, 1.0 - clip_eps)

def predicted_new_loglbol(df_agn, loglbol):

    # Data
    x = np.asarray(df_agn['LOGLBOL'], dtype=float)   # predictor
    y = np.asarray(df_agn['log_lbol'], dtype=float)  # response

    # Drop NaNs/Infs
    m = np.isfinite(x) & np.isfinite(y)
    x_fit, y_fit = x[m], y[m]

    # Quadratic fit: y ≈ c2*x^2 + c1*x + c0
    c2, c1, c0 = np.polyfit(x_fit, y_fit, deg=2)

    def predict_log_lbol(LOGLBOL):
        """
        Vectorized predictor: given LOGLBOL, return predicted log_lbol.
        """
        X = np.asarray(LOGLBOL, dtype=float)
        return c2 * X**2 + c1 * X + c0

    predicted_loglbol = predict_log_lbol(loglbol)
    return predicted_loglbol

def get_completeness_function_2d(
    df_agn,
    sim_file="data/nov9_mock_mag_z_moresources.h5",
    #sim_file="data/dec4_mock_mag_z_ananna.h5",
    n_mag_bins=30, n_z_bins=40,
    smooth_counts=True,
    plot=False,
    plot_path=None,
    fill_along_mag=False,
    fill_along_z=False,
):
    """
    Build p(detect | m, z)
    """
    import os, h5py
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    import pandas as pd
    from astropy.cosmology import FlatLambdaCDM
    # Simulation inputs: apparent-magnitude proxy at rest-frame 2500 A and z
    sim_file = resolve_qvc_data_path(sim_file)
    with h5py.File(sim_file, "r") as f:
        if "apparent_mag_2500" in f:
            m_true = np.asarray(f["apparent_mag_2500"][:], dtype=float)
        else:
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
    mag_min, mag_max = 18.5, 24.0
    z_min,   z_max   = 0.0, 4.0
    mag_edges = np.linspace(mag_min, mag_max, n_mag_bins + 1)
    z_edges   = np.linspace(z_min,  z_max,    n_z_bins   + 1)
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    z_centers   = 0.5 * (z_edges[:-1]   + z_edges[1:])
    dm = float(mag_centers[1] - mag_centers[0]) if len(mag_centers) > 1 else float(mag_edges[-1] - mag_edges[0])
    dz = float(z_centers[1] - z_centers[0])     if len(z_centers)   > 1 else float(z_edges[-1] - z_edges[0])
    # 2D histograms on [mag, z]
    H_true, _, _ = np.histogram2d(m_true, z_true, bins=[mag_edges, z_edges])
    H_obs,  _, _ = np.histogram2d(m_obs,  z_obs,  bins=[mag_edges, z_edges])
    if smooth_counts:
        # --- Choose physical smoothing widths (recommended) ---
        sigma_mag = 0.2    # mag, for completeness-map smoothing along magnitude
        sigma_z_abs = 0.2  # absolute redshift, for smoothing along z
        print(f"Smoothing counts with sigma_mag={sigma_mag} mag (sigma_mag/dm={sigma_mag/dm}), sigma_z={sigma_z_abs} absolute z")
        # Convert physical -> pixel for the Gaussian filter
        sig_mag_pix = max(float(sigma_mag/dm), 1e-6)
        sig_z_pix   = max(float(sigma_z_abs/dz), 1e-6)
        H_true_s = gaussian_filter(H_true, sigma=(sig_mag_pix, sig_z_pix),
                                mode="nearest")
        H_obs_s  = gaussian_filter(H_obs,  sigma=(sig_mag_pix, sig_z_pix),
                                mode="nearest")
    else:
        sigma_mag = 0.0
        H_true_s, H_obs_s = H_true, H_obs
    eps = 1e-12
    C = H_obs_s / (H_true_s + eps)
    C[H_true_s < eps] = 0.0
    C = np.clip(C, 0.0, 1.0)

    if fill_along_mag:
    # fill non-decreasing completeness along mag (for each z)
        tol = 1e-12
        for j in range(C.shape[1]):              # each z-column
            vmax = C[:, j].max()
            i = C.shape[0] - 1 - np.argmax(C[::-1, j] >= vmax - tol)  # rightmost max
            C[:i+1, j] = vmax

    if fill_along_z:
        # fill high z
        C = np.maximum.accumulate(C[:, ::-1], axis=1)[:, ::-1]
        C = np.clip(C, 0.0, 1.0)

    # --- Gentle boost for faint, low-z objects ---
    # Controls (tune if needed):
    # faint_mag_start = 20.0   # start boosting at m >= this (fainter than this)
    # z_boost_max     = 1.0    # only z < this gets boosted
    # boost_strength  = 1.0  # 0.0–0.5 is a mild nudge; 0.2 ~ "slightly more complete"

    # # Build smooth 1D ramps on the bin centers
    # wm = np.clip((mag_centers - faint_mag_start) / (mag_max - faint_mag_start), 0.0, 1.0)
    # wz = np.clip((z_boost_max - z_centers) / z_boost_max, 0.0, 1.0)
    # W  = np.outer(wm, wz)  # shape (n_mag_bins, n_z_bins)

    # # Nudge completeness upward only where there's room (1 - C)
    # C = C + boost_strength * W * (1.0 - C)
    # C = np.clip(C, 0.0, 1.0)

    if plot:
        base_plot_path = plot_path or "plots/hubble"
        plot_dir = os.path.join(base_plot_path, "completeness")
        os.makedirs(plot_dir, exist_ok=True)
        # Plot completeness map
        plt.figure(figsize=(7, 5))
        im = plt.imshow(
            np.log10(np.clip(C.T, 1e-12, None)), origin="lower", aspect="auto",
            extent=[mag_edges[0], mag_edges[-1], z_edges[0], z_edges[-1]], cmap="viridis",
            vmin=-4, vmax=0
        )
        plt.ylabel(r'$z$')
        # plt.xlabel(r'$L_{2500\,\mathrm{\AA}}$ (erg s$^{-1}$)')
        #plt.xlabel(r"$m_{2500\,\mathrm{\AA}} \; (\mathrm{mag})$")
        plt.xlabel(r'$m_{2500\,\mathrm{\AA}}$ (mag)')

        cbar = plt.colorbar(im); 
        cbar.set_label(r"Completeness $\log\,p(I{=}1\,|\,m,z)$")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "completeness_map.pdf"), dpi=600)
        plt.close()
        # Plot H_obs
        plt.figure(figsize=(7, 5))
        im = plt.imshow(
            np.log10(np.clip(H_obs.T, 1e-12, None)), origin="lower", aspect="auto",
            extent=[mag_edges[0], mag_edges[-1], z_edges[0], z_edges[-1]], cmap="plasma"
        )
        plt.ylabel(r"$z$")
        plt.xlabel(r"$m_{2500\,\text{\AA}} \; (\mathrm{mag})$")
        cbar = plt.colorbar(im); cbar.set_label(r"$\log\,H_{\rm obs}$")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "H_obs_map.pdf"), dpi=600)
        plt.close()
        # Plot H_true
        plt.figure(figsize=(7, 5))
        im = plt.imshow(
            np.log10(np.clip(H_true.T, 1e-12, None)), origin="lower", aspect="auto",
            extent=[mag_edges[0], mag_edges[-1], z_edges[0], z_edges[-1]], cmap="cividis"
        )
        plt.ylabel(r"$z$")
        plt.xlabel(r"Apparent Magnitude $m_{i,\mathrm{rest}} \; (\mathrm{mag})$")
        cbar = plt.colorbar(im); cbar.set_label(r"$\log\,H_{\rm true}$")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "H_true_map.pdf"), dpi=600)
        plt.close()
    return Completeness2D(mag_centers, z_centers, C), mag_centers, z_centers, dm, dz, sigma_mag


def _plot_completeness_vs_fhost_slices(
    completeness3d,
    mag_centers,
    z_centers,
    fhost_centers,
    *,
    plot_dir,
    mag_slices=(19.5, 21.0, 22.5),
    z_slices=(0.5, 1.5, 2.5),
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(mag_slices), figsize=(5.0 * len(mag_slices), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, mag0 in zip(axes, mag_slices):
        mag_eval = np.full_like(fhost_centers, np.clip(mag0, mag_centers[0], mag_centers[-1]), dtype=float)
        for z0 in z_slices:
            z_eval = np.full_like(fhost_centers, np.clip(z0, z_centers[0], z_centers[-1]), dtype=float)
            p = completeness3d(mag_eval, z_eval, fhost_centers)
            ax.plot(fhost_centers, p, lw=2, label=fr"$z={z0:.1f}$")
        ax.set_title(fr"$m_{{2500}}={mag0:.1f}$")
        ax.set_xlabel(r"$f_{\rm host,center}$")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"$p(\mathrm{detect}\mid m_{2500}, z, f_{\rm host})$")
    axes[-1].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "completeness_vs_fhost_slices.pdf"), dpi=300)
    plt.close(fig)


def _plot_completeness_vs_alpha_slices(
    completeness4d,
    mag_centers,
    z_centers,
    fhost_centers,
    alpha_centers,
    *,
    plot_dir,
    mag_slices=(19.5, 21.0, 22.5),
    z_slices=(0.5, 1.5, 2.5),
    fhost_slice=0.05,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(mag_slices), figsize=(5.0 * len(mag_slices), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    fhost_eval = np.full_like(alpha_centers, np.clip(fhost_slice, fhost_centers[0], fhost_centers[-1]), dtype=float)
    for ax, mag0 in zip(axes, mag_slices):
        mag_eval = np.full_like(alpha_centers, np.clip(mag0, mag_centers[0], mag_centers[-1]), dtype=float)
        for z0 in z_slices:
            z_eval = np.full_like(alpha_centers, np.clip(z0, z_centers[0], z_centers[-1]), dtype=float)
            p = completeness4d(mag_eval, z_eval, fhost_eval, alpha_centers)
            ax.plot(alpha_centers, p, lw=2, label=fr"$z={z0:.1f}$")
        ax.set_title(fr"$m_{{2500}}={mag0:.1f}$, $f_{{\rm host}}={fhost_slice:.2f}$")
        ax.set_xlabel(r"$\alpha_{\lambda}$")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"$p(\mathrm{detect}\mid m_{2500}, z, f_{\rm host}, \alpha_{\lambda})$")
    axes[-1].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "completeness_vs_alpha_slices.pdf"), dpi=300)
    plt.close(fig)


def get_completeness_function_3d_fhost(
    df_agn,
    sim_file="data/nov9_mock_mag_z_moresources.h5",
    n_mag_bins=30,
    n_z_bins=40,
    n_fhost_bins=20,
    smooth_counts=True,
    plot=False,
    plot_path=None,
    fit_logL_max=45.5,
    sigma_mag=0.2,
    sigma_z_abs=0.2,
    sigma_fhost=0.05,
):
    """
    Build p(detect | m, z, f_host_center) using a one-shot host-fraction parent model.
    """
    import matplotlib.pyplot as plt

    if "f_host_center" not in df_agn.columns:
        raise KeyError("df_agn must contain 'f_host_center' for 3D host-aware completeness.")

    sim_file = resolve_qvc_data_path(sim_file)
    with h5py.File(sim_file, "r") as f:
        if "apparent_mag_2500" in f:
            m_true = np.asarray(f["apparent_mag_2500"][:], dtype=float)
        else:
            m_true = np.asarray(f["apparent_mag_i_rest"][:], dtype=float)
        z_true = np.asarray(f["z"][:], dtype=float)

    z_obs = df_agn["z"].to_numpy(dtype=float)
    m_obs = df_agn["apparent_mag_2500"].to_numpy(dtype=float)
    fhost_obs = df_agn["f_host_center"].to_numpy(dtype=float)

    ok_obs = (
        np.isfinite(m_obs)
        & np.isfinite(z_obs)
        & np.isfinite(fhost_obs)
        & (fhost_obs >= 0.0)
        & (fhost_obs <= 1.0)
    )
    ok_true = np.isfinite(m_true) & np.isfinite(z_true)
    m_obs, z_obs, fhost_obs = m_obs[ok_obs], z_obs[ok_obs], fhost_obs[ok_obs]
    m_true, z_true = m_true[ok_true], z_true[ok_true]

    host_model = fit_fhost_center_l2500_model(df_agn.loc[ok_obs], fit_logL_max=fit_logL_max, cosmo=COSMO)
    logL_true = apparent_mag_to_logL2500(m_true, z_true, COSMO)
    rng = np.random.default_rng(12345)
    fhost_true = sample_fhost_center_from_logL2500(logL_true, host_model, rng)

    mag_min, mag_max = 18.5, 24.0
    z_min, z_max = 0.0, 4.0
    fhost_min, fhost_max = 0.0, 1.0
    mag_edges = np.linspace(mag_min, mag_max, n_mag_bins + 1)
    z_edges = np.linspace(z_min, z_max, n_z_bins + 1)
    fhost_edges = np.linspace(fhost_min, fhost_max, n_fhost_bins + 1)
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    fhost_centers = 0.5 * (fhost_edges[:-1] + fhost_edges[1:])
    dm = float(mag_centers[1] - mag_centers[0]) if len(mag_centers) > 1 else float(mag_edges[-1] - mag_edges[0])
    dz = float(z_centers[1] - z_centers[0]) if len(z_centers) > 1 else float(z_edges[-1] - z_edges[0])
    dfh = float(fhost_centers[1] - fhost_centers[0]) if len(fhost_centers) > 1 else float(fhost_edges[-1] - fhost_edges[0])

    H_true, _ = np.histogramdd((m_true, z_true, fhost_true), bins=[mag_edges, z_edges, fhost_edges])
    H_obs, _ = np.histogramdd((m_obs, z_obs, fhost_obs), bins=[mag_edges, z_edges, fhost_edges])

    if smooth_counts:
        sig_mag_pix = max(float(sigma_mag / dm), 1e-6)
        sig_z_pix = max(float(sigma_z_abs / dz), 1e-6)
        sig_fhost_pix = max(float(sigma_fhost / dfh), 1e-6)
        H_true_s = gaussian_filter(H_true, sigma=(sig_mag_pix, sig_z_pix, sig_fhost_pix), mode="nearest")
        H_obs_s = gaussian_filter(H_obs, sigma=(sig_mag_pix, sig_z_pix, sig_fhost_pix), mode="nearest")
    else:
        H_true_s, H_obs_s = H_true, H_obs

    eps = 1e-12
    C = H_obs_s / (H_true_s + eps)
    C[H_true_s < eps] = 0.0
    C = np.clip(C, 0.0, 1.0)

    completeness3d = Completeness3D(mag_centers, z_centers, fhost_centers, C)

    if plot:
        base_plot_path = plot_path or "plots/hubble"
        plot_dir = os.path.join(base_plot_path, "completeness")
        os.makedirs(plot_dir, exist_ok=True)

        with open(os.path.join(plot_dir, "fhost_center_l2500_model.json"), "w") as handle:
            json.dump(host_model, handle, indent=2)

        _plot_completeness_vs_fhost_slices(
            completeness3d,
            mag_centers,
            z_centers,
            fhost_centers,
            plot_dir=plot_dir,
        )

        fig, ax = plt.subplots(figsize=(7, 5))
        c_slice = C.mean(axis=2)
        im = ax.imshow(
            np.log10(np.clip(c_slice.T, 1e-12, None)),
            origin="lower",
            aspect="auto",
            extent=[mag_edges[0], mag_edges[-1], z_edges[0], z_edges[-1]],
            cmap="viridis",
            vmin=-4,
            vmax=0,
        )
        ax.set_ylabel(r"$z$")
        ax.set_xlabel(r"$m_{2500\,\mathrm{\AA}}$ (mag)")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r"Mean over $f_{\rm host}$: $\log\,p(I{=}1\,|\,m,z,f_{\rm host})$")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "completeness_map_mean_fhost.pdf"), dpi=600)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        c_slice_mf = C.mean(axis=1)
        im = ax.imshow(
            np.log10(np.clip(c_slice_mf.T, 1e-12, None)),
            origin="lower",
            aspect="auto",
            extent=[mag_edges[0], mag_edges[-1], fhost_edges[0], fhost_edges[-1]],
            cmap="viridis",
            vmin=-4,
            vmax=0,
        )
        ax.set_xlabel(r"$m_{2500\,\mathrm{\AA}}$ (mag)")
        ax.set_ylabel(r"$f_{\rm host,center}$")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r"Mean over $z$: $\log\,p(I{=}1\,|\,m,z,f_{\rm host})$")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "completeness_map_m2500_fhost.pdf"), dpi=600)
        plt.close(fig)

    return (
        completeness3d,
        mag_centers,
        z_centers,
        fhost_centers,
        dm,
        dz,
        dfh,
        sigma_mag,
        host_model,
    )


def get_completeness_function_4d_fhost_alpha(
    df_agn,
    sim_file="data/nov9_mock_mag_z_moresources.h5",
    n_mag_bins=30,
    n_z_bins=40,
    n_fhost_bins=20,
    n_alpha_bins=20,
    smooth_counts=True,
    plot=False,
    plot_path=None,
    fit_logL_max=45.5,
    sigma_mag=0.2,
    sigma_z_abs=0.2,
    sigma_fhost=0.05,
    sigma_alpha=0.15,
):
    """
    Build p(detect | m, z, f_host_center, alpha_lambda) using a one-shot host model
    and a fixed mock alpha_lambda population.
    """
    import matplotlib.pyplot as plt

    required = {"f_host_center", "alpha_lambda", "apparent_mag_2500", "z"}
    if not required.issubset(df_agn.columns):
        missing = ", ".join(sorted(required - set(df_agn.columns)))
        raise KeyError(f"df_agn must contain columns for 4D completeness: {missing}")

    sim_file = resolve_qvc_data_path(sim_file)
    with h5py.File(sim_file, "r") as f:
        if "apparent_mag_2500" in f:
            m_true = np.asarray(f["apparent_mag_2500"][:], dtype=float)
        else:
            m_true = np.asarray(f["apparent_mag_i_rest"][:], dtype=float)
        z_true = np.asarray(f["z"][:], dtype=float)

    z_obs = df_agn["z"].to_numpy(dtype=float)
    m_obs = df_agn["apparent_mag_2500"].to_numpy(dtype=float)
    fhost_obs = df_agn["f_host_center"].to_numpy(dtype=float)
    alpha_obs = df_agn["alpha_lambda"].to_numpy(dtype=float)

    ok_obs = (
        np.isfinite(m_obs)
        & np.isfinite(z_obs)
        & np.isfinite(fhost_obs)
        & np.isfinite(alpha_obs)
        & (fhost_obs >= 0.0)
        & (fhost_obs <= 1.0)
        & (alpha_obs >= _ALPHA_MIN)
        & (alpha_obs <= _ALPHA_MAX)
    )
    ok_true = np.isfinite(m_true) & np.isfinite(z_true)
    m_obs, z_obs, fhost_obs, alpha_obs = m_obs[ok_obs], z_obs[ok_obs], fhost_obs[ok_obs], alpha_obs[ok_obs]
    m_true, z_true = m_true[ok_true], z_true[ok_true]

    host_model = fit_fhost_center_l2500_model(df_agn.loc[ok_obs], fit_logL_max=fit_logL_max, cosmo=COSMO)
    logL_true = apparent_mag_to_logL2500(m_true, z_true, COSMO)
    rng = np.random.default_rng(12345)
    fhost_true = sample_fhost_center_from_logL2500(logL_true, host_model, rng)
    alpha_true = np.clip(rng.normal(_ALPHA_TRUE_MEAN, _ALPHA_TRUE_SIGMA, size=np.shape(m_true)), _ALPHA_MIN, _ALPHA_MAX)

    mag_min, mag_max = 18.5, 24.0
    z_min, z_max = 0.0, 4.0
    fhost_min, fhost_max = 0.0, 1.0
    alpha_min, alpha_max = _ALPHA_MIN, _ALPHA_MAX
    mag_edges = np.linspace(mag_min, mag_max, n_mag_bins + 1)
    z_edges = np.linspace(z_min, z_max, n_z_bins + 1)
    fhost_edges = np.linspace(fhost_min, fhost_max, n_fhost_bins + 1)
    alpha_edges = np.linspace(alpha_min, alpha_max, n_alpha_bins + 1)
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    fhost_centers = 0.5 * (fhost_edges[:-1] + fhost_edges[1:])
    alpha_centers = 0.5 * (alpha_edges[:-1] + alpha_edges[1:])
    dm = float(mag_centers[1] - mag_centers[0]) if len(mag_centers) > 1 else float(mag_edges[-1] - mag_edges[0])
    dz = float(z_centers[1] - z_centers[0]) if len(z_centers) > 1 else float(z_edges[-1] - z_edges[0])
    dfh = float(fhost_centers[1] - fhost_centers[0]) if len(fhost_centers) > 1 else float(fhost_edges[-1] - fhost_edges[0])
    da = float(alpha_centers[1] - alpha_centers[0]) if len(alpha_centers) > 1 else float(alpha_edges[-1] - alpha_edges[0])

    H_true, _ = np.histogramdd((m_true, z_true, fhost_true, alpha_true), bins=[mag_edges, z_edges, fhost_edges, alpha_edges])
    H_obs, _ = np.histogramdd((m_obs, z_obs, fhost_obs, alpha_obs), bins=[mag_edges, z_edges, fhost_edges, alpha_edges])

    if smooth_counts:
        sig_mag_pix = max(float(sigma_mag / dm), 1e-6)
        sig_z_pix = max(float(sigma_z_abs / dz), 1e-6)
        sig_fhost_pix = max(float(sigma_fhost / dfh), 1e-6)
        sig_alpha_pix = max(float(sigma_alpha / da), 1e-6)
        H_true_s = gaussian_filter(H_true, sigma=(sig_mag_pix, sig_z_pix, sig_fhost_pix, sig_alpha_pix), mode="nearest")
        H_obs_s = gaussian_filter(H_obs, sigma=(sig_mag_pix, sig_z_pix, sig_fhost_pix, sig_alpha_pix), mode="nearest")
    else:
        H_true_s, H_obs_s = H_true, H_obs

    eps = 1e-12
    C = H_obs_s / (H_true_s + eps)
    C[H_true_s < eps] = 0.0
    C = np.clip(C, 0.0, 1.0)

    completeness4d = Completeness4D(mag_centers, z_centers, fhost_centers, alpha_centers, C)

    if plot:
        base_plot_path = plot_path or "plots/hubble"
        plot_dir = os.path.join(base_plot_path, "completeness")
        os.makedirs(plot_dir, exist_ok=True)

        with open(os.path.join(plot_dir, "fhost_center_l2500_model.json"), "w") as handle:
            json.dump(host_model, handle, indent=2)

        _plot_completeness_vs_fhost_slices(
            lambda mag_eval, z_eval, f_eval: completeness4d(mag_eval, z_eval, f_eval, np.full_like(f_eval, _ALPHA_TRUE_MEAN)),
            mag_centers,
            z_centers,
            fhost_centers,
            plot_dir=plot_dir,
        )
        _plot_completeness_vs_alpha_slices(
            completeness4d,
            mag_centers,
            z_centers,
            fhost_centers,
            alpha_centers,
            plot_dir=plot_dir,
        )

        fig, ax = plt.subplots(figsize=(7, 5))
        c_slice = C.mean(axis=(2, 3))
        im = ax.imshow(
            np.log10(np.clip(c_slice.T, 1e-12, None)),
            origin="lower",
            aspect="auto",
            extent=[mag_edges[0], mag_edges[-1], z_edges[0], z_edges[-1]],
            cmap="viridis",
            vmin=-4,
            vmax=0,
        )
        ax.set_ylabel(r"$z$")
        ax.set_xlabel(r"$m_{2500\,\mathrm{\AA}}$ (mag)")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r"Mean over $f_{\rm host}, \alpha_{\lambda}$: $\log\,p(I{=}1\,|\,m,z,f_{\rm host},\alpha_{\lambda})$")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "completeness_map_mean_fhost_alpha.pdf"), dpi=600)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        c_slice_ma = C.mean(axis=(1, 2))
        im = ax.imshow(
            np.log10(np.clip(c_slice_ma.T, 1e-12, None)),
            origin="lower",
            aspect="auto",
            extent=[mag_edges[0], mag_edges[-1], alpha_edges[0], alpha_edges[-1]],
            cmap="viridis",
            vmin=-4,
            vmax=0,
        )
        ax.set_xlabel(r"$m_{2500\,\mathrm{\AA}}$ (mag)")
        ax.set_ylabel(r"$\alpha_{\lambda}$")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r"Mean over $z, f_{\rm host}$: $\log\,p(I{=}1\,|\,m,z,f_{\rm host},\alpha_{\lambda})$")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "completeness_map_m2500_alpha_lambda.pdf"), dpi=600)
        plt.close(fig)

    return (
        completeness4d,
        mag_centers,
        z_centers,
        fhost_centers,
        alpha_centers,
        dm,
        dz,
        dfh,
        da,
        sigma_mag,
        host_model,
        dict(alpha_mean=_ALPHA_TRUE_MEAN, alpha_sigma=_ALPHA_TRUE_SIGMA, alpha_min=_ALPHA_MIN, alpha_max=_ALPHA_MAX),
    )


import numpy as np
from scipy.interpolate import NearestNDInterpolator, RegularGridInterpolator

def make_dm_function(m, z, dm, m_bins=40, z_bins=40):
    """
    Build a 2D interpolator dm(m,z) defined on bin midpoints.
    Queries are clipped to the populated grid range and empty cells are filled
    from the nearest populated bin before interpolation.
    """
    m = np.asarray(m)
    z = np.asarray(z)
    dm = np.asarray(dm)
    mask = np.isfinite(m) & np.isfinite(z) & np.isfinite(dm)
    m, z, dm = m[mask], z[mask], dm[mask]
    if m.size == 0:
        raise ValueError("make_dm_function requires at least one finite (m, z, dm) point.")

    # Build bin edges
    m_edges = np.linspace(m.min(), m.max(), int(m_bins) + 1) if np.isscalar(m_bins) else np.asarray(m_bins)
    z_edges = np.linspace(z.min(), z.max(), int(z_bins) + 1) if np.isscalar(z_bins) else np.asarray(z_bins)

    # 2D binning: means per cell
    counts, _, _ = np.histogram2d(z, m, bins=[z_edges, m_edges])
    sums,   _, _ = np.histogram2d(z, m, bins=[z_edges, m_edges], weights=dm)
    mean = np.full_like(sums, np.nan, dtype=float)
    mean[counts > 0] = sums[counts > 0] / counts[counts > 0]

    # Grid points are the bin midpoints
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
    m_mid = 0.5 * (m_edges[:-1] + m_edges[1:])

    valid = np.isfinite(mean)
    if not np.any(valid):
        raise ValueError("make_dm_function could not populate any finite bias bins.")

    if np.any(~valid):
        zz, mm = np.meshgrid(z_mid, m_mid, indexing="ij")
        nearest_fill = NearestNDInterpolator(
            np.column_stack([zz[valid], mm[valid]]),
            mean[valid],
        )
        filled = mean.copy()
        filled[~valid] = nearest_fill(np.column_stack([zz[~valid], mm[~valid]]))
    else:
        filled = mean

    if len(z_mid) < 2 or len(m_mid) < 2:
        z0 = float(z_mid[0])
        m0 = float(m_mid[0])
        dm0 = float(filled[0, 0])

        def interp_single_bin(pts):
            pts = np.asarray(pts)
            arr = np.atleast_2d(pts).astype(float)
            out = np.full(arr.shape[0], dm0, dtype=float)
            return out if np.ndim(pts) > 1 else out[0]

        return interp_single_bin

    interp_core = RegularGridInterpolator(
        (z_mid, m_mid), filled,
        method="linear", bounds_error=False, fill_value=None
    )

    z_lo, z_hi = float(z_mid.min()), float(z_mid.max())
    m_lo, m_hi = float(m_mid.min()), float(m_mid.max())

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
