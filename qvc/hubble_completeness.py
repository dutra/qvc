import numpy as np
import h5py

from scipy import stats
from scipy.stats import norm, sigmaclip, multivariate_normal
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import interp1d

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

    Ignores all data inputs. Hard-coded mag grid: 16 to 26.
    """
    completeness2d = SimpleCompleteness2D(mag_lim=mag_lim, width=width)

    # Fixed mag grid: 16 to 26
    mag_min, mag_max = 16.0, 26.0
    mag_centers = np.linspace(mag_min, mag_max, 300)
    z_centers = np.linspace(0.0, 3.0, 30)  # dummy grid for API compatibility
    dm = mag_centers[1] - mag_centers[0]
    dz = z_centers[1] - z_centers[0]

    if plot:
        completeness_vals = completeness2d(mag_centers)
        os.makedirs("plots/completeness", exist_ok=True)

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
        plt.savefig("plots/completeness/simple_completeness_function.png", dpi=200)
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

def get_completeness_function_2d(
    df_agn,
    sim_file="data/mock_mag_z.h5",
    n_mag_bins=20, n_z_bins=60,
    #mag_min=15, mag_max=24,
    sigma_mag=0.5, sigma_z=0.5,
    smooth_counts=True,
    plot=False,
):
    """
    Build p(detect | m, z) from a simulated 'true' set and an observed set.

    Key changes:
      • Smooth counts, not the ratio.
      • No normalization by max; result already in [0,1].
      • Use fill_value=0 outside grid (no pre-clipping).
    """
    # --- Load simulated (true) sample
    with h5py.File(sim_file, 'r') as f:
        mags_true_i = f['apparent_mag_i_rest'][:]
        z_true = f['z'][:]

    mask = (
        np.isfinite(df_agn['apparent_mag_2500'].values) &
        np.isfinite(df_agn['apparent_mag_i_rest'].values) &
        (df_agn['apparent_mag_2500'].values > 18) & (df_agn['apparent_mag_2500'].values < 26) &
        (df_agn['apparent_mag_i_rest'].values > 15) & (df_agn['apparent_mag_i_rest'].values < 25)
    )
    y = df_agn['apparent_mag_2500'].values[mask]
    x = df_agn['apparent_mag_i_rest'].values[mask]

    # Fit line
    x_pivot = np.mean(x)
    print(f"Pivot point for x: {x_pivot}")
    slope, intercept = np.polyfit(x-x_pivot, y, 1)
    y_fit = slope * (x-x_pivot) + intercept

    calculated_mags_true_2500 = (mags_true_i-x_pivot)*slope + intercept

    mags_true = calculated_mags_true_2500

    # Estimate scatter as std of residuals
    residuals = y - y_fit
    scatter = np.std(residuals)
    print(f"Scatter in m_2500 - m_i fit (std of residuals): {scatter:.2f}")
    
    # --- Observed sample
    mags_obs = np.asarray(df_agn["apparent_mag_2500"].values)
    z_obs    = np.asarray(df_agn["z"].values)

    if plot:
        from matplotlib import pyplot as plt
        # Observed
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, alpha=0.7, label='Data')
        plt.plot(x, y_fit, color='red', label=f'Best fit: y={slope:.2f}x+{intercept:.2f}')
        plt.xlabel('apparent_mag_i (total AGN, host-subtracted, rest)')
        plt.ylabel('apparent_mag_2500 (continuum-only, rest)')
        #plt.title('apparent_mag_i vs apparent_mag_2500')
        plt.grid(True)
        plt.legend()
        plt.savefig("plots/completeness/mag2500_vs_magi_fit.png", dpi=200)
        plt.close()
    
    # --- Clean NaNs/Infs
    mask_true = np.isfinite(mags_true) & np.isfinite(z_true)
    mags_true, z_true = mags_true[mask_true], z_true[mask_true]

    mask_obs = np.isfinite(mags_obs) & np.isfinite(z_obs)
    mags_obs, z_obs = mags_obs[mask_obs], z_obs[mask_obs]

    # --- Bin edges and centers
    z_min, z_max = float(np.min(z_true)), 4.0
    if z_max - z_min < 1e-3:
        z_min -= 0.01
        z_max += 0.01

    #mag_min = np.min(mags_true) - 0.5
    #mag_max = np.max(mags_true) + 0.5
    mag_min = 16
    mag_max = 26
    print(f"Using mag range: {mag_min:.2f} to {mag_max:.2f}")

    mag_edges = np.linspace(mag_min, mag_max, n_mag_bins + 1)
    z_edges   = np.linspace(z_min,  z_max,  n_z_bins   + 1)
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    z_centers   = 0.5 * (z_edges[:-1]   + z_edges[1:])
    print(f"Using z range: {z_centers[0]:.2f} to {z_centers[-1]:.2f}")

    # --- 2D histograms
    H_true, _, _ = np.histogram2d(mags_true, z_true, bins=[mag_edges, z_edges])
    H_obs,  _, _ = np.histogram2d(mags_obs,  z_obs,  bins=[mag_edges, z_edges])

    # --- Smooth COUNTS (not the ratio)
    if smooth_counts:
        # 'constant' with cval=0 avoids propagating edge values outward
        H_true_s = gaussian_filter(H_true, sigma=(sigma_mag, sigma_z), mode="constant", cval=0.0)
        H_obs_s  = gaussian_filter(H_obs,  sigma=(sigma_mag, sigma_z), mode="constant", cval=0.0)
    else:
        H_true_s, H_obs_s = H_true, H_obs

    # --- Completeness ratio with small epsilon
    eps = 1e-12
    C = H_obs_s / (H_true_s + eps)
    C[H_true_s < eps] = 0.0                     # no true support -> undefined -> 0
    C = np.clip(C, 0.0, 1.0)

    # --- Optional diagnostic plot
    if plot:
        import matplotlib.pyplot as plt
        plt.imshow(
            np.log10(C.T), origin="lower", aspect="auto",
            extent=[mag_edges[0], mag_edges[-1], z_edges[0], z_edges[-1]]
        )
        plt.xlabel("Apparent Magnitude")
        plt.ylabel("Redshift")
        plt.title("Completeness Map p(detect | m, z)")
        cbar = plt.colorbar()
        cbar.set_label("p(detect)")
        plt.tight_layout()
        plt.savefig("plots/completeness/completeness_map.png", dpi=200)
        #plt.show()
        plt.close()

        plt.imshow(
            np.log10(H_true_s.T), origin="lower", aspect="auto",
            extent=[mag_edges[0], mag_edges[-1], z_edges[0], z_edges[-1]]
        )
        plt.xlabel("Apparent Magnitude")
        plt.ylabel("Redshift")
        plt.title("Completeness Map p(detect | m, z)")
        cbar = plt.colorbar()
        cbar.set_label("p(detect)")
        plt.tight_layout()
        plt.savefig("plots/completeness/H_true_s.png", dpi=200)
        plt.close()

        plt.imshow(
            np.log10(H_obs_s.T), origin="lower", aspect="auto",
            extent=[mag_edges[0], mag_edges[-1], z_edges[0], z_edges[-1]]
        )
        plt.xlabel("Apparent Magnitude")
        plt.ylabel("Redshift")
        plt.title("Completeness Map p(detect | m, z)")
        cbar = plt.colorbar()
        cbar.set_label("p(detect)")
        plt.tight_layout()
        plt.savefig("plots/completeness/H_obs_s.png", dpi=200)

    # bin widths (uniform by construction)
    dm = float(mag_centers[1] - mag_centers[0]) if len(mag_centers) > 1 else float(mag_edges[-1] - mag_edges[0])
    dz = float(z_centers[1] - z_centers[0])     if len(z_centers)   > 1 else float(z_edges[-1] - z_edges[0])

    return Completeness2D(mag_centers, z_centers, C), mag_centers, z_centers, dm, dz, scatter

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
    mask = np.isfinite(m) & np.isfinite(z) & np.isfinite(dm)
    m, z, dm = m[mask], z[mask], dm[mask]

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