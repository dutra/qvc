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

class LogisticCompleteness2500:
    """
    Logistic completeness in apparent_mag_2500 with a z-dependent faint limit
    derived from an observed-band depth. Can use per-object alpha_lambda.
    """
    def __init__(self,
                 mag_lim_obs=24.5,
                 width=1.0,
                 lam_obs=7480.0,
                 alpha_nu=-0.4,     # used only if alpha_lambda is not supplied at call time
                 delta_m=0.0,
                 floor=1e-12):
        self.mag_lim_obs = float(mag_lim_obs)
        self.width       = float(width)
        self.lam_obs     = float(lam_obs)
        self.alpha_nu    = float(alpha_nu)
        self.delta_m     = float(delta_m)
        self.floor       = float(floor)

    @staticmethod
    def _lambda_to_nu(alpha_lambda):
        # alpha_nu = -(alpha_lambda + 2)
        return -(alpha_lambda + 2.0)

    def mlim_2500(self, z, alpha_lambda=None):
        """
        Effective m_lim at 2500Å, optionally using per-object alpha_lambda.
        z: array-like
        alpha_lambda: None or array-like with same shape as z
        """
        z = np.asarray(z)
        if alpha_lambda is None:
            alpha_nu = self.alpha_nu
        else:
            alpha_lambda = np.asarray(alpha_lambda)
            alpha_nu = self._lambda_to_nu(alpha_lambda)

        x = ((1.0 + z) * self.lam_obs) / 2500.0
        mlim = self.mag_lim_obs - 2.5 * alpha_nu * np.log10(x) + self.delta_m
        return mlim

    def __call__(self, mag_2500, z, alpha_lambda=None):
        """
        p(detect | m_2500, z, alpha_lambda)
        """
        m = np.asarray(mag_2500)
        mlim = self.mlim_2500(z, alpha_lambda=alpha_lambda)
        p = 1.0 / (1.0 + np.exp((m - mlim) / self.width))
        return np.clip(p, self.floor, 1.0)

    @property
    def grid(self):
        return dict(mag_lim_obs=self.mag_lim_obs, width=self.width,
                    lam_obs=self.lam_obs, alpha_nu=self.alpha_nu,
                    delta_m=self.delta_m)


def get_completeness_function_2d_simple(*args,
                                        mag_lim_obs=24.5,
                                        width=1.0,
                                        lam_obs=7480.0,
                                        alpha_nu=-0.4,
                                        delta_m=-1.33,
                                        alpha_lambda=None,   # <— NEW
                                        plot=False,
                                        **kwargs):
    """
    Returns a z-dependent completeness function that can use per-object alpha_lambda.
    """
    completeness2d = LogisticCompleteness2500(mag_lim_obs=mag_lim_obs,
                                              width=width,
                                              lam_obs=lam_obs,
                                              delta_m=delta_m,
                                              alpha_nu=alpha_nu)

    # dummies for backward-compatibility; fast path doesn't need grids
    mag_centers = np.array([0.0])
    z_centers   = np.array([0.0])
    dm = dz = 0.0

    if plot:
        import matplotlib.pyplot as plt, os
        os.makedirs("plots/completeness", exist_ok=True)

        z_plot = np.linspace(0, 4, 200)
        if alpha_lambda is None:
            mlim = completeness2d.mlim_2500(z_plot)  # uses scalar alpha_nu
        else:
            # use a representative slope for the plot
            alpha_lambda_med = np.median(np.asarray(alpha_lambda))
            mlim = completeness2d.mlim_2500(z_plot, alpha_lambda=np.full_like(z_plot, alpha_lambda_med))

        plt.figure(figsize=(8,5))
        plt.plot(z_plot, mlim)
        plt.xlabel("Redshift z"); plt.ylabel(r"$m_{\mathrm{lim},2500}(z)$")
        plt.title("Effective 2500Å magnitude limit vs z")
        plt.grid(True); plt.tight_layout()
        plt.savefig("plots/completeness/eff_mlim_2500_vs_z.png", dpi=200)
        plt.close()

    return completeness2d, mag_centers, z_centers, dm, dz

class Completeness2D:
    def __init__(self, mag_centers, z_centers, completeness_map):
        self.mag_centers = mag_centers
        self.z_centers = z_centers

        # Clip NaNs and store minimum finite completeness
        completeness_map_clean = np.nan_to_num(completeness_map, nan=0.0)
        self.min_completeness_value = float(np.nanmin(completeness_map_clean))

        self.mag_min = mag_centers[0]
        self.mag_max = mag_centers[-1]
        self.z_min = z_centers[0]
        self.z_max = z_centers[-1]

        self.interp_fn = RegularGridInterpolator(
            (mag_centers, z_centers),
            completeness_map_clean,
            bounds_error=False,
            fill_value=0.0
        )

    def __call__(self, mag, z):
        mag = np.asarray(mag)
        z = np.asarray(z)
        mag_b, z_b = np.broadcast_arrays(mag, z)

        mag_clipped = np.clip(mag_b, self.mag_min, self.mag_max)
        z_clipped = np.clip(z_b, self.z_min, self.z_max)

        pts = np.column_stack([mag_clipped.ravel(), z_clipped.ravel()])
        vals = self.interp_fn(pts)
        return vals.reshape(mag_b.shape)

    def get_completeness_map(self):
        return self.interp_fn.values


def get_completeness_function_2d(df_agn,
                                 sim_file="data/sampled_apparent_magnitudes_redshift_vol3.h5",
                                 n_mag_bins=20, n_z_bins=30,
                                 mag_min=16, mag_max=24,
                                 sigma_mag=1.0, sigma_z=0.7,
                                 normalize=True,
                                 plot=False):
    # --- Load simulated (true) sample
    mags_true_list, z_true_list = [], []
    with h5py.File(sim_file, "r") as f:
        for name in f["redshift_bin"]:
            ds = f["redshift_bin"][name]
            mags = ds[()]
            z_bin = ds.attrs["redshift"]
            mags_true_list.append(mags)
            z_true_list.append(np.full_like(mags, z_bin, dtype=float))

    mags_true = np.concatenate(mags_true_list)
    z_true = np.concatenate(z_true_list)

    # --- Load observed sample
    mags_obs = df_agn['apparent_mag_2500'].values
    z_obs = df_agn['z'].values

    # --- Clean NaNs/Infs
    mask_true = np.isfinite(mags_true) & np.isfinite(z_true)
    mags_true = mags_true[mask_true]
    z_true = z_true[mask_true]

    mask_obs = np.isfinite(mags_obs) & np.isfinite(z_obs)
    mags_obs = mags_obs[mask_obs]
    z_obs = z_obs[mask_obs]

    # --- Bin edges and centers
    z_min, z_max = np.min(z_true), np.max(z_true)
    if z_max - z_min < 1e-3:
        z_min -= 0.01
        z_max += 0.01

    mag_edges = np.linspace(mag_min, mag_max, n_mag_bins + 1)
    z_edges = np.linspace(z_min, z_max, n_z_bins + 1)
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    # --- Histogram both samples
    hist_true, _, _ = np.histogram2d(mags_true, z_true, bins=[mag_edges, z_edges])
    hist_obs, _, _ = np.histogram2d(mags_obs, z_obs, bins=[mag_edges, z_edges])

    # --- Compute completeness ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        completeness = np.zeros_like(hist_true, dtype=float)
        valid = hist_true > 0
        completeness[valid] = hist_obs[valid] / hist_true[valid]

    # --- Optional smoothing
    completeness_smoothed = gaussian_filter(completeness, sigma=(sigma_mag, sigma_z), mode='nearest')

    # --- Clip to [0, 1] and normalize if requested
    completeness_smoothed = np.clip(completeness_smoothed, 0.0, 1.0)
    if normalize and np.nanmax(completeness_smoothed) > 0:
        completeness_smoothed /= np.nanmax(completeness_smoothed)

    # --- Compute bin widths for completeness convolution
    dm = mag_centers[1] - mag_centers[0]
    dz = z_centers[1] - z_centers[0]

    # --- Optional diagnostic plot
    if plot:
        import matplotlib.pyplot as plt
        plt.imshow(completeness_smoothed.T, origin='lower', aspect='auto',
                   extent=[mag_edges[0], mag_edges[-1], z_edges[0], z_edges[-1]])
        plt.xlabel('Apparent Magnitude')
        plt.ylabel('Redshift')
        plt.title('Completeness Map (Smoothed)')
        plt.colorbar(label='p(detect | m, z)')
        plt.tight_layout()
        plt.savefig("plots/completeness/completeness_map_2d.png", dpi=200)
        plt.close()
        #plt.show()

    return Completeness2D(mag_centers, z_centers, completeness_smoothed), mag_centers, z_centers, dm, dz
