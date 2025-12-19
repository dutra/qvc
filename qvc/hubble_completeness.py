import numpy as np
import h5py
import os
from scipy import stats
from scipy.stats import norm, sigmaclip, multivariate_normal
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import interp1d
from functools import partial

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
        sigma_z_abs = 0.5  # absolute redshift, for smoothing along z
        print(f"Smoothing counts with sigma_mag={sigma_mag} mag (sigma_mag/dm={sigma_mag/dm}), sigma_z={sigma_z_abs} absolute z")
        # Convert physical -> pixel for the Gaussian filter
        sig_mag_pix = max(float(sigma_mag/dm), 1e-6)
        sig_z_pix   = max(float(sigma_z_abs/dz), 1e-6)
        H_true_s = gaussian_filter(H_true, sigma=(sig_mag_pix, sig_z_pix),
                                mode="constant", cval=0.0)
        H_obs_s  = gaussian_filter(H_obs,  sigma=(sig_mag_pix, sig_z_pix),
                                mode="constant", cval=0.0)
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
        plot_dir = f"plots/hubble/{prefix}/completeness"
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
        plt.savefig(os.path.join(plot_dir, "completeness_map.png"), dpi=200)
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
        plt.savefig(os.path.join(plot_dir, "H_obs_map.png"), dpi=200)
        #plt.savefig(os.path.join(plot_dir, "H_obs_map.pdf"), dpi=600)
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
        plt.savefig(os.path.join(plot_dir, "H_true_map.png"), dpi=200)
        #plt.savefig(os.path.join(plot_dir, "H_true_map.pdf"), dpi=600)
        plt.close()
    return Completeness2D(mag_centers, z_centers, C), mag_centers, z_centers, dm, dz, sigma_mag


import numpy as np
from scipy.interpolate import RegularGridInterpolator

def make_dm_function(m, z, dm, m_bins=40, z_bins=40):
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

    interp_core = RegularGridInterpolator(
        (z_mid, m_mid), mean,
        method='nearest', bounds_error=False, fill_value=None
    )

    # Clipping wrapper
    # z_lo, z_hi = z_mid.min(), z_mid.max()
    # m_lo, m_hi = m_mid.min(), m_mid.max()

    # def interp_clipped(pts):
    #     pts = np.asarray(pts)
    #     arr = np.atleast_2d(pts).astype(float)
    #     arr[:, 0] = np.clip(arr[:, 0], z_lo, z_hi)
    #     arr[:, 1] = np.clip(arr[:, 1], m_lo, m_hi)
    #     out = interp_core(arr)
    #     return out if np.ndim(pts) > 1 else out[0]

    return interp_core

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
