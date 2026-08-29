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
from qvc.hubble.cuts import (
    COMPLETENESS_MAG_EDGE_MAX,
    COMPLETENESS_MAG_EDGE_MIN,
    COMPLETENESS_N_MAG_BINS,
)


COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)
COMPLETENESS_MAG_COL = "completeness_m_2500"
COMPLETENESS_MAG_ERR_COL = "completeness_m_2500_err"
COMPLETENESS_FHOST_COL = "f_host_2500_psf"
COMPLETENESS_FHOST_ERR_COL = "f_host_2500_psf_err"
VALID_COMPLETENESS_MAGNITUDES = ("dereddened", "attenuated")
_COMPLETENESS_MAGNITUDE_SOURCES = {
    "dereddened": ("m_2500_dereddened", "m_2500_dereddened_err"),
    "attenuated": (
        "m_2500_attenuated_model",
        "m_2500_attenuated_model_err",
    ),
}


def normalize_completeness_magnitude(completeness_magnitude):
    """Validate the physical m_2500 definition used for completeness."""
    normalized = str(completeness_magnitude).strip().lower()
    if normalized not in VALID_COMPLETENESS_MAGNITUDES:
        raise ValueError(
            f"Invalid completeness_magnitude={completeness_magnitude!r}. "
            f"Expected one of {VALID_COMPLETENESS_MAGNITUDES}."
        )
    return normalized


def prepare_completeness_magnitude_columns(
    df,
    completeness_magnitude="dereddened",
):
    """Return a dataframe with stable aliases for the selected m_2500 fields."""
    choice = normalize_completeness_magnitude(completeness_magnitude)
    magnitude_source, error_source = _COMPLETENESS_MAGNITUDE_SOURCES[choice]
    missing = {
        magnitude_source,
        error_source,
    } - set(df.columns)
    if missing:
        raise KeyError(
            f"Completeness magnitude choice {choice!r} requires columns "
            f"{sorted(missing)}."
        )
    prepared = df.copy()
    prepared[COMPLETENESS_MAG_COL] = prepared[magnitude_source]
    prepared[COMPLETENESS_MAG_ERR_COL] = prepared[error_source]
    prepared.attrs.update(df.attrs)
    prepared.attrs["completeness_magnitude"] = choice
    prepared.attrs["completeness_magnitude_source"] = magnitude_source
    prepared.attrs["completeness_magnitude_err_source"] = error_source
    return prepared


def resolve_completeness_magnitude_column(df):
    """Require the explicitly prepared completeness-magnitude alias."""
    if COMPLETENESS_MAG_COL in df.columns:
        return COMPLETENESS_MAG_COL
    raise KeyError(
        "Completeness magnitude has not been prepared. Expected explicit "
        f"column {COMPLETENESS_MAG_COL!r}; call "
        "prepare_completeness_magnitude_columns(df, choice) first."
    )


def evaluate_dm_interp(
    dm_interp,
    z,
    m2500,
    *,
    f_host_2500_psf=None,
    f_host_2500=None,
    alpha_lambda=None,
):
    """Evaluate debias correction using the richest available feature set."""
    z = np.asarray(z, dtype=float)
    m2500 = np.asarray(m2500, dtype=float)
    cols = [z, m2500]
    # Completeness-derived debiasing should use the PSF host fraction.  Keep the
    # old keyword only for external compatibility; internal callers pass PSF.
    fhost = f_host_2500_psf if f_host_2500_psf is not None else f_host_2500
    if fhost is not None:
        cols.append(np.asarray(fhost, dtype=float))
        if alpha_lambda is not None:
            cols.append(np.asarray(alpha_lambda, dtype=float))
    pts = np.column_stack(cols)
    finite = np.all(np.isfinite(pts), axis=1)
    out = np.full(pts.shape[0], np.nan, dtype=float)
    if np.any(finite):
        out[finite] = np.asarray(dm_interp(pts[finite]), dtype=float)
    return out


def _binned_mean_interp_1d(z, y, *, z_bins=40):
    """Fallback 1D trend estimate from z-binned means."""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(z) & np.isfinite(y)
    z = z[mask]
    y = y[mask]
    if z.size == 0:
        raise ValueError("_binned_mean_interp_1d requires at least one finite (z, y) point.")

    z_edges = np.linspace(z.min(), z.max(), int(z_bins) + 1) if np.isscalar(z_bins) else np.asarray(z_bins, dtype=float)
    counts, _ = np.histogram(z, bins=z_edges)
    sums, _ = np.histogram(z, bins=z_edges, weights=y)
    mean = np.full(counts.shape, np.nan, dtype=float)
    mean[counts > 0] = sums[counts > 0] / counts[counts > 0]

    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
    valid = np.isfinite(mean)
    if not np.any(valid):
        raise ValueError("_binned_mean_interp_1d could not populate any finite bins.")

    if np.any(~valid):
        nearest_fill = RegularGridInterpolator(
            (z_mid[valid],),
            mean[valid],
            bounds_error=False,
            fill_value=None,
        )
        filled = mean.copy()
        filled[~valid] = nearest_fill(z_mid[~valid, None])
    else:
        filled = mean

    if len(z_mid) < 2:
        y0 = float(filled[0])

        def interp_single_bin(z_query):
            zq = np.asarray(z_query, dtype=float)
            out = np.full(np.atleast_1d(zq).shape, y0, dtype=float)
            return out if np.ndim(z_query) > 0 else out[0]

        return interp_single_bin

    z_lo, z_hi = float(z_mid.min()), float(z_mid.max())

    def interp_zonly(z_query):
        zq = np.asarray(z_query, dtype=float)
        out = np.interp(np.clip(np.atleast_1d(zq), z_lo, z_hi), z_mid, filled)
        return out if np.ndim(z_query) > 0 else out[0]

    return interp_zonly


def build_lowess_trend_1d(
    z,
    y,
    yerr=None,
    *,
    frac=0.25,
    it=1,
    min_points=10,
):
    """
    Build a local 1D LOWESS-style trend evaluator with optional inverse-variance weights.

    The returned callable clamps to the nearest endpoint value outside the fitted z range.
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    if yerr is None:
        base_w = np.ones_like(y, dtype=float)
    else:
        yerr = np.asarray(yerr, dtype=float)
        base_w = np.divide(
            1.0,
            yerr**2,
            out=np.zeros_like(yerr, dtype=float),
            where=np.isfinite(yerr) & (yerr > 0),
        )

    mask = np.isfinite(z) & np.isfinite(y) & np.isfinite(base_w) & (base_w > 0)
    z = z[mask]
    y = y[mask]
    base_w = base_w[mask]

    if z.size == 0:
        raise ValueError("build_lowess_trend_1d requires at least one finite weighted point.")

    order = np.argsort(z)
    z = z[order]
    y = y[order]
    base_w = base_w[order]

    z_unique, inverse = np.unique(z, return_inverse=True)
    w_unique = np.bincount(inverse, weights=base_w)
    y_unique = np.divide(
        np.bincount(inverse, weights=base_w * y),
        w_unique,
        out=np.zeros_like(w_unique, dtype=float),
        where=w_unique > 0,
    )

    n = z_unique.size
    if n == 1:
        y0 = float(y_unique[0])

        def constant_eval(z_query):
            zq = np.asarray(z_query, dtype=float)
            out = np.full(np.atleast_1d(zq).shape, y0, dtype=float)
            return out if np.ndim(z_query) > 0 else out[0]

        return constant_eval

    frac = float(np.clip(frac, 1.0 / n, 1.0))
    span = min(n, max(2, int(np.ceil(frac * n)), int(min_points)))
    robust_w = np.ones(n, dtype=float)

    def _local_predict(z_query, robust_weights):
        zq = np.atleast_1d(np.asarray(z_query, dtype=float))
        pred = np.full(zq.shape, np.nan, dtype=float)
        eps = max(np.finfo(float).eps, 1e-12 * max(1.0, float(z_unique[-1] - z_unique[0])))

        for i, x0 in enumerate(zq):
            dist = np.abs(z_unique - x0)
            bandwidth = np.partition(dist, span - 1)[span - 1]
            bandwidth = max(float(bandwidth), eps)
            u = dist / bandwidth
            kernel = np.where(u < 1.0, (1.0 - u**3) ** 3, 0.0)
            kernel[dist == 0.0] = 1.0
            w = kernel * w_unique * robust_weights
            good = np.isfinite(w) & (w > 0)

            if np.count_nonzero(good) < 2:
                pred[i] = np.average(y_unique, weights=w_unique * robust_weights)
                continue

            x = z_unique[good] - x0
            yy = y_unique[good]
            ww = w[good]
            s0 = np.sum(ww)
            s1 = np.sum(ww * x)
            s2 = np.sum(ww * x * x)
            t0 = np.sum(ww * yy)
            t1 = np.sum(ww * x * yy)
            denom = s0 * s2 - s1 * s1
            if (not np.isfinite(denom)) or abs(denom) < eps * max(1.0, s0 * s2):
                pred[i] = t0 / s0
            else:
                pred[i] = (s2 * t0 - s1 * t1) / denom

        return pred

    for _ in range(max(int(it), 0)):
        fitted = _local_predict(z_unique, robust_w)
        resid = y_unique - fitted
        scale = np.nanmedian(np.abs(resid))
        if (not np.isfinite(scale)) or scale <= 0:
            break
        u = resid / (6.0 * scale)
        robust_w = np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0)
        if not np.any(robust_w > 0):
            robust_w = np.ones(n, dtype=float)
            break

    y_left = float(_local_predict(np.array([z_unique[0]]), robust_w)[0])
    y_right = float(_local_predict(np.array([z_unique[-1]]), robust_w)[0])

    def evaluate(z_query):
        zq = np.asarray(z_query, dtype=float)
        zq_1d = np.atleast_1d(zq)
        out = _local_predict(zq_1d, robust_w)
        out = np.where(zq_1d < z_unique[0], y_left, out)
        out = np.where(zq_1d > z_unique[-1], y_right, out)
        return out if np.ndim(z_query) > 0 else out[0]

    return evaluate


def _can_use_lowess_trend_1d(z, y=None, yerr=None, *, min_points=10):
    """Return True when the inputs support a stable LOWESS fit."""
    z = np.asarray(z, dtype=float)
    mask = np.isfinite(z)

    if y is not None:
        y = np.asarray(y, dtype=float)
        mask &= np.isfinite(y)

    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)
        mask &= np.isfinite(yerr) & (yerr > 0)

    if np.count_nonzero(mask) < max(int(min_points), 2):
        return False

    return np.unique(z[mask]).size >= 2


def build_smooth_trend_1d(
    z,
    y,
    yerr=None,
    *,
    frac=0.25,
    it=1,
    min_points=10,
    fallback_bins=40,
):
    """
    Build a smooth 1D trend with LOWESS when the input supports it,
    otherwise fall back to a binned interpolation.
    """
    if _can_use_lowess_trend_1d(z, y=y, yerr=yerr, min_points=min_points):
        return build_lowess_trend_1d(
            z,
            y,
            yerr=yerr,
            frac=frac,
            it=it,
            min_points=min_points,
        )
    return _binned_mean_interp_1d(z, y, z_bins=fallback_bins)

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


def _estimate_mock_count_scale(H_obs_s, H_true_s, mag_centers, *, complete_mag_max=20.0, eps=1e-12):
    """
    Estimate one global mock-count normalization from bright bins that should be
    close to complete.
    """
    H_obs_s = np.asarray(H_obs_s, dtype=float)
    H_true_s = np.asarray(H_true_s, dtype=float)
    mag_centers = np.asarray(mag_centers, dtype=float)
    bright = np.isfinite(mag_centers) & (mag_centers <= float(complete_mag_max))
    if not np.any(bright):
        return 1.0

    bright_shape = (bright.size,) + (1,) * (H_true_s.ndim - 1)
    bright_mask = bright.reshape(bright_shape)
    obs_sum = float(np.sum(np.where(bright_mask, H_obs_s, 0.0)))
    true_sum = float(np.sum(np.where(bright_mask, H_true_s, 0.0)))
    if not np.isfinite(obs_sum) or not np.isfinite(true_sum) or obs_sum <= 0.0 or true_sum <= eps:
        return 1.0
    return max(obs_sum / true_sum, eps)


def _scaled_completeness_ratio(
    H_obs_s,
    H_true_s,
    mag_centers,
    *,
    label,
    count_scale=None,
    complete_mag_max=20.0,
    eps=1e-12,
):
    """Compute completeness after one global bright-end normalization."""
    if count_scale is None:
        count_scale = _estimate_mock_count_scale(
            H_obs_s,
            H_true_s,
            mag_centers,
            complete_mag_max=complete_mag_max,
            eps=eps,
        )
        scale_source = f"bright-end m<={complete_mag_max:.2f}"
    else:
        count_scale = max(float(count_scale), eps)
        scale_source = "mock H5 metadata"
    C_raw = H_obs_s / (count_scale * H_true_s + eps)
    finite = np.isfinite(C_raw)
    n_finite = int(np.count_nonzero(finite))
    n_gt1 = int(np.count_nonzero(finite & (C_raw > 1.0)))
    print(
        f"{label}: mock count scale={count_scale:.4g} ({scale_source}); "
        f"C_raw > 1 in {n_gt1}/{n_finite} finite bins ({n_gt1 / max(n_finite, 1):.2%}) before clipping."
    )
    C = np.asarray(C_raw, dtype=float)
    C[H_true_s < eps] = 0.0
    C = np.clip(C, 0.0, 1.0)
    return C

class Completeness2D:
    """
    Interpolates p(detect | m, z) on a (mag, z) grid.
    - Bright magnitudes are clipped to the bright grid edge; faint magnitudes
      outside the grid return 0.
    - Redshifts outside the grid are linearly extrapolated from the nearest
      two redshift planes. Extrapolated probabilities are bounded to [0, 1].
    """
    def __init__(self, mag_centers, z_centers, completeness_map):
        self.mag_centers = np.asarray(mag_centers)
        self.z_centers   = np.asarray(z_centers)

        # Ensure finite in [0,1]
        C = np.nan_to_num(completeness_map, nan=0.0, posinf=0.0, neginf=0.0)
        C = np.clip(C, 0.0, 1.0).astype(float)

        self.mag_min, self.mag_max = float(self.mag_centers[0]),  float(self.mag_centers[-1])
        self.z_min,   self.z_max   = float(self.z_centers[0]),    float(self.z_centers[-1])

        # ``fill_value=None`` enables multilinear extrapolation. __call__ masks
        # unsupported faint magnitudes explicitly, so only redshift extrapolates.
        self._interp = RegularGridInterpolator(
            (self.mag_centers, self.z_centers),
            C,
            bounds_error=False,
            fill_value=None,
        )
        self._warned_oob = False

    def __call__(self, mag, z):
        mag_raw = np.asarray(mag, dtype=float)
        z_raw = np.asarray(z, dtype=float)
        oob = (
            (mag_raw < self.mag_min)
            | (mag_raw > self.mag_max)
            | (z_raw < self.z_min)
            | (z_raw > self.z_max)
        )
        if np.any(oob) and not self._warned_oob:
            print(
                "[WARNING] Completeness2D received objects outside the "
                f"grid range m=[{self.mag_min:.2f}, {self.mag_max:.2f}], "
                f"z=[{self.z_min:.2f}, {self.z_max:.2f}]. "
                f"count={int(np.count_nonzero(oob))}"
            )
            self._warned_oob = True
        mag = np.maximum(np.asarray(mag, dtype=float), self.mag_min)
        z = np.asarray(z, dtype=float)
        m_b, z_b = np.broadcast_arrays(mag, z)
        pts = np.column_stack([m_b.ravel(), z_b.ravel()])
        finite = np.all(np.isfinite(pts), axis=1) & (pts[:, 0] <= self.mag_max)
        vals = np.zeros(pts.shape[0], dtype=float)
        if np.any(finite):
            vals[finite] = np.clip(self._interp(pts[finite]), 0.0, 1.0)
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
    - Bright magnitudes are clipped to the bright grid edge; faint magnitudes
      outside the grid still return 0.
    - Redshifts outside the grid are linearly extrapolated from the nearest
      two redshift planes. Extrapolated probabilities are bounded to [0, 1].
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
            fill_value=None,
        )
        self._warned_oob = False

    def __call__(self, mag, z, f_host):
        mag_raw = np.asarray(mag, dtype=float)
        z_raw = np.asarray(z, dtype=float)
        f_host_raw = np.asarray(f_host, dtype=float)
        oob = (
            (mag_raw < self.mag_min)
            | (mag_raw > self.mag_max)
            | (z_raw < self.z_min)
            | (z_raw > self.z_max)
            | (f_host_raw < self.fhost_min)
            | (f_host_raw > self.fhost_max)
        )
        if np.any(oob) and not self._warned_oob:
            print(
                "[WARNING] Completeness3D received objects outside the "
                f"grid range m=[{self.mag_min:.2f}, {self.mag_max:.2f}], "
                f"z=[{self.z_min:.2f}, {self.z_max:.2f}], "
                f"f_host=[{self.fhost_min:.3f}, {self.fhost_max:.3f}]. "
                f"count={int(np.count_nonzero(oob))}"
            )
            self._warned_oob = True
        mag = np.maximum(np.asarray(mag, dtype=float), self.mag_min)
        z = np.asarray(z, dtype=float)
        f_host = np.asarray(f_host)
        # The completeness cube is defined on bin centers, but f_host is a
        # bounded physical variable on [0, 1]. Clip to the nearest supported
        # center so values very close to 0 or 1 do not spuriously get
        # fill_value=0 from the interpolator.
        f_host = np.clip(f_host, self.fhost_min, self.fhost_max)
        m_b, z_b, f_b = np.broadcast_arrays(mag, z, f_host)
        pts = np.column_stack([m_b.ravel(), z_b.ravel(), f_b.ravel()])
        finite = np.all(np.isfinite(pts), axis=1) & (pts[:, 0] <= self.mag_max)
        vals = np.zeros(pts.shape[0], dtype=float)
        if np.any(finite):
            vals[finite] = np.clip(self._interp(pts[finite]), 0.0, 1.0)
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
    - Bright magnitudes are clipped to the bright grid edge; faint magnitudes
      outside the grid still return 0.
    - Redshifts outside the grid are linearly extrapolated from the nearest
      two redshift planes. Extrapolated probabilities are bounded to [0, 1].
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
            fill_value=None,
        )
        self._warned_oob = False

    def __call__(self, mag, z, f_host, alpha_lambda):
        mag_raw = np.asarray(mag, dtype=float)
        z_raw = np.asarray(z, dtype=float)
        f_host_raw = np.asarray(f_host, dtype=float)
        alpha_raw = np.asarray(alpha_lambda, dtype=float)
        oob = (
            (mag_raw < self.mag_min)
            | (mag_raw > self.mag_max)
            | (z_raw < self.z_min)
            | (z_raw > self.z_max)
            | (f_host_raw < self.fhost_min)
            | (f_host_raw > self.fhost_max)
            | (alpha_raw < self.alpha_min)
            | (alpha_raw > self.alpha_max)
        )
        if np.any(oob) and not self._warned_oob:
            print(
                "[WARNING] Completeness4D received objects outside the "
                f"grid range m=[{self.mag_min:.2f}, {self.mag_max:.2f}], "
                f"z=[{self.z_min:.2f}, {self.z_max:.2f}], "
                f"f_host=[{self.fhost_min:.3f}, {self.fhost_max:.3f}], "
                f"alpha_lambda=[{self.alpha_min:.2f}, {self.alpha_max:.2f}]. "
                f"count={int(np.count_nonzero(oob))}"
            )
            self._warned_oob = True
        mag = np.maximum(np.asarray(mag, dtype=float), self.mag_min)
        z = np.asarray(z, dtype=float)
        f_host = np.asarray(f_host)
        alpha_lambda = np.asarray(alpha_lambda)
        f_host = np.clip(f_host, self.fhost_min, self.fhost_max)
        alpha_lambda = np.clip(alpha_lambda, self.alpha_min, self.alpha_max)
        m_b, z_b, f_b, a_b = np.broadcast_arrays(mag, z, f_host, alpha_lambda)
        pts = np.column_stack([m_b.ravel(), z_b.ravel(), f_b.ravel(), a_b.ravel()])
        finite = np.all(np.isfinite(pts), axis=1) & (pts[:, 0] <= self.mag_max)
        vals = np.zeros(pts.shape[0], dtype=float)
        if np.any(finite):
            vals[finite] = np.clip(self._interp(pts[finite]), 0.0, 1.0)
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
_ALPHA_MIN_SIGMA = 0.05


def _finite_float_attr(attrs, *keys):
    for key in keys:
        if key not in attrs:
            continue
        try:
            value = float(attrs[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def _robust_alpha_sigma(alpha, *, fallback=_ALPHA_TRUE_SIGMA, min_sigma=_ALPHA_MIN_SIGMA):
    alpha = np.asarray(alpha, dtype=float)
    alpha = alpha[np.isfinite(alpha)]
    sigma = np.nan
    if alpha.size >= 2:
        p16, p84 = np.nanpercentile(alpha, [16.0, 84.0])
        sigma = 0.5 * (p84 - p16)
        if (not np.isfinite(sigma)) or sigma <= 0.0:
            sigma = np.nanstd(alpha, ddof=1)
    if (not np.isfinite(sigma)) or sigma <= 0.0:
        sigma = fallback
    return float(max(sigma, min_sigma))


def _alpha_lambda_model_from_values(
    alpha_lambda,
    *,
    source,
    count_key="n_fit",
    fallback_mean=_ALPHA_TRUE_MEAN,
    fallback_sigma=_ALPHA_TRUE_SIGMA,
):
    alpha = np.asarray(alpha_lambda, dtype=float)
    mask = np.isfinite(alpha) & (alpha >= _ALPHA_MIN) & (alpha <= _ALPHA_MAX)
    alpha = alpha[mask]
    if alpha.size == 0:
        mean = float(fallback_mean)
        sigma = float(fallback_sigma)
    else:
        mean = float(np.nanmedian(alpha))
        sigma = _robust_alpha_sigma(alpha, fallback=fallback_sigma)
    return {
        "alpha_mean": mean,
        "alpha_sigma": sigma,
        "alpha_min": float(_ALPHA_MIN),
        "alpha_max": float(_ALPHA_MAX),
        "source": source,
        count_key: int(alpha.size),
    }


def fit_alpha_lambda_population(df_agn):
    """Fit the fallback alpha_lambda parent from the usable observed sample."""
    if "alpha_lambda" not in df_agn.columns:
        return {
            "alpha_mean": float(_ALPHA_TRUE_MEAN),
            "alpha_sigma": float(_ALPHA_TRUE_SIGMA),
            "alpha_min": float(_ALPHA_MIN),
            "alpha_max": float(_ALPHA_MAX),
            "source": "default_alpha_lambda",
            "n_fit": 0,
        }
    return _alpha_lambda_model_from_values(
        df_agn["alpha_lambda"].to_numpy(dtype=float),
        source="observed_alpha_lambda",
        count_key="n_fit",
    )


def sample_alpha_lambda_population(size, model, rng):
    mean = float(model.get("alpha_mean", _ALPHA_TRUE_MEAN))
    sigma = float(model.get("alpha_sigma", _ALPHA_TRUE_SIGMA))
    sigma = max(sigma, _ALPHA_MIN_SIGMA)
    return np.clip(rng.normal(mean, sigma, size=size), _ALPHA_MIN, _ALPHA_MAX)


def _read_mock_alpha_lambda(h5file):
    for key in ("alpha_lambda", "PL_slope"):
        if key in h5file:
            return np.asarray(h5file[key][:], dtype=float), f"mock_h5_dataset:{key}"
    if "alpha_nu" in h5file:
        alpha_nu = np.asarray(h5file["alpha_nu"][:], dtype=float)
        return -alpha_nu - 2.0, "mock_h5_dataset:alpha_nu"
    return None, None


def _read_mock_fhost_2500_psf(h5file):
    for key in (COMPLETENESS_FHOST_COL, "frac_host_psf_2500", "fhost_2500_psf"):
        if key in h5file:
            return np.asarray(h5file[key][:], dtype=float), f"mock_h5_dataset:{key}"
    return None, None


def _alpha_lambda_model_from_h5_attrs(attrs):
    alpha_mean = _finite_float_attr(attrs, "alpha_lambda_parent_mean", "alpha_lambda_mean")
    alpha_sigma = _finite_float_attr(attrs, "alpha_lambda_parent_sigma", "alpha_lambda_sigma")

    alpha_nu_mean = _finite_float_attr(attrs, "alpha_nu_parent_mean", "alpha_nu_input_mean", "alpha_nu_mean")
    alpha_nu_sigma = _finite_float_attr(attrs, "alpha_nu_parent_sigma", "alpha_nu_input_sigma", "alpha_nu_sigma")
    if alpha_mean is None and alpha_nu_mean is not None:
        alpha_mean = -alpha_nu_mean - 2.0
    if alpha_sigma is None and alpha_nu_sigma is not None:
        alpha_sigma = abs(alpha_nu_sigma)
    if alpha_mean is None:
        return None
    if alpha_sigma is None or not np.isfinite(alpha_sigma) or alpha_sigma <= 0.0:
        alpha_sigma = _ALPHA_TRUE_SIGMA
    return {
        "alpha_mean": float(np.clip(alpha_mean, _ALPHA_MIN, _ALPHA_MAX)),
        "alpha_sigma": float(max(alpha_sigma, _ALPHA_MIN_SIGMA)),
        "alpha_min": float(_ALPHA_MIN),
        "alpha_max": float(_ALPHA_MAX),
        "source": "mock_h5_attrs",
        "n_fit": 0,
    }


def generalized_sigmoid_fhost(logL2500, x0, k, nu):
    arg = np.clip(k * (np.asarray(logL2500, dtype=float) - x0), -60.0, 60.0)
    return 1.0 / np.power(1.0 + np.exp(arg), nu)


def apparent_mag_to_logL2500(m2500, z, cosmo):
    m2500 = np.asarray(m2500, dtype=float)
    z = np.asarray(z, dtype=float)
    M2500 = m2500 - cosmo.distmod(z).value
    return convert_M2500_to_logL2500(M2500)


def fit_fhost_2500_l2500_model(
    df_agn,
    *,
    f_host_col="f_host_2500",
    fit_logL_max=45.5,
    clip_eps=_FHOST_CLIP_EPS,
    cosmo=COSMO,
):
    required = {"z", f_host_col}
    if not required.issubset(df_agn.columns):
        missing = ", ".join(sorted(required - set(df_agn.columns)))
        raise KeyError(f"Missing required columns for f_host model fit: {missing}")

    z = np.asarray(df_agn["z"], dtype=float)
    magnitude_col = resolve_completeness_magnitude_column(df_agn)
    m2500 = np.asarray(df_agn[magnitude_col], dtype=float)
    f_host = np.asarray(df_agn[f_host_col], dtype=float)
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
        raise ValueError("Need at least 8 finite rows to fit the f_host_2500(log L_2500) model.")

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


def predict_fhost_2500_from_logL2500(logL2500, model):
    clip_eps = float(model.get("clip_eps", _FHOST_CLIP_EPS))
    mean = generalized_sigmoid_fhost(logL2500, model["x0"], model["k"], model["nu"])
    return np.clip(mean, clip_eps, 1.0 - clip_eps)


def sample_fhost_2500_from_logL2500(logL2500, model, rng):
    mean = predict_fhost_2500_from_logL2500(logL2500, model)
    sigma_host_logit = float(model.get("sigma_host_logit", 0.0))
    if sigma_host_logit <= 0.0:
        return mean
    sampled_logit = logit(mean) + rng.normal(0.0, sigma_host_logit, size=np.shape(mean))
    clip_eps = float(model.get("clip_eps", _FHOST_CLIP_EPS))
    return np.clip(expit(sampled_logit), clip_eps, 1.0 - clip_eps)


def _fit_fhost_population_model(df_agn, df_agn_fhost_population, *, fit_logL_max):
    """Fit the host-fraction parent on the supplied population frame."""

    fit_df = df_agn if df_agn_fhost_population is None else df_agn_fhost_population
    host_model = fit_fhost_2500_l2500_model(
        fit_df,
        f_host_col=COMPLETENESS_FHOST_COL,
        fit_logL_max=fit_logL_max,
        cosmo=COSMO,
    )
    host_model["observed_fit_source"] = (
        "fit_sample_f_host_2500_psf_vs_l2500"
        if df_agn_fhost_population is None
        else "precut_f_host_2500_psf_vs_l2500"
    )
    host_model["n_observed_population"] = int(len(fit_df))
    return host_model


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
    n_mag_bins=COMPLETENESS_N_MAG_BINS, n_z_bins=40,
    smooth_counts=True,
    plot=False,
    plot_path=None,
    fill_along_mag=False,
    fill_along_z=False,
    z_range=None,
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
        mock_count_scale = f.attrs.get("mock_count_scale")
        declared_z_min = float(f.attrs.get("mock_redshift_min", np.nan))
        declared_z_max = float(f.attrs.get("mock_redshift_max", np.nan))

    # Filter finite
    z_obs = df_agn["z"].to_numpy(dtype=float)
    magnitude_col = resolve_completeness_magnitude_column(df_agn)
    m_obs = df_agn[magnitude_col].to_numpy(dtype=float)
    ok_obs  = np.isfinite(m_obs) & np.isfinite(z_obs)
    ok_true = np.isfinite(m_true) & np.isfinite(z_true)
    m_obs,  z_obs  = m_obs[ok_obs],  z_obs[ok_obs]
    m_true, z_true = m_true[ok_true], z_true[ok_true]
    # Grid
    mag_min, mag_max = COMPLETENESS_MAG_EDGE_MIN, COMPLETENESS_MAG_EDGE_MAX
    if z_range is None:
        z_min, z_max = 0.0, 4.0
    else:
        z_min, z_max = (float(z_range[0]), float(z_range[1]))
        if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min < 0.0 or z_min >= z_max:
            raise ValueError(
                "Completeness z_range must be finite, non-negative, and increasing."
            )
        if np.isfinite(declared_z_min) and np.isfinite(declared_z_max):
            support_min, support_max = declared_z_min, declared_z_max
        else:
            support_min = float(np.nanmin(z_true))
            support_max = float(np.nanmax(z_true))
        tolerance = max(1e-6, 0.01 * (z_max - z_min))
        if support_min > z_min + tolerance or support_max < z_max - tolerance:
            raise ValueError(
                "Completeness mock redshift support does not cover the plotting "
                f"sample: mock=[{support_min:.6g}, {support_max:.6g}], "
                f"required=[{z_min:.6g}, {z_max:.6g}]. Provide a compatible "
                "mock or omit --completeness_sim_file to generate one."
            )
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
    C = _scaled_completeness_ratio(
        H_obs_s,
        H_true_s,
        mag_centers,
        label="Completeness2D",
        count_scale=mock_count_scale,
        eps=eps,
    )

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
        C_plot = gaussian_filter(C, sigma=(1, 1), mode="nearest")
        plt.figure(figsize=(7, 5))
        im = plt.imshow(
            np.log10(np.clip(C_plot.T, 1e-12, None)), origin="lower", aspect="auto",
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
        ax.set_xlabel(r"$f_{\rm host,2500}$")
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


def _plot_fixed_slice_completeness_map_4d(
    completeness4d,
    mag_edges,
    z_edges,
    mag_centers,
    z_centers,
    fhost_centers,
    alpha_centers,
    *,
    plot_dir,
    fhost_slice=0.05,
    alpha_slice=-1.5,
):
    import matplotlib.pyplot as plt

    fhost0 = float(np.clip(fhost_slice, fhost_centers[0], fhost_centers[-1]))
    alpha0 = float(np.clip(alpha_slice, alpha_centers[0], alpha_centers[-1]))
    M, Z = np.meshgrid(mag_centers, z_centers, indexing="ij")
    C_slice = completeness4d(M, Z, np.full_like(M, fhost0), np.full_like(M, alpha0))

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        np.log10(np.clip(C_slice.T, 1e-12, None)),
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
    cbar.set_label(
        rf"$\log\,p(I{{=}}1\,|\,m,z,f_{{\rm host}}={fhost0:.2f},\alpha_{{\lambda}}={alpha0:.2f})$"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "completeness_map_fixed_fhost_alpha.pdf"), dpi=600)
    plt.close(fig)


def _plot_weighted_marginal_completeness_map_4d(
    C,
    mag_edges,
    z_edges,
    fhost_true,
    alpha_true,
    fhost_edges,
    alpha_edges,
    *,
    plot_dir,
):
    import matplotlib.pyplot as plt

    H_latent, _ = np.histogramdd((fhost_true, alpha_true), bins=[fhost_edges, alpha_edges])
    W = H_latent.astype(float)
    if np.sum(W) <= 0:
        W = np.ones_like(W, dtype=float)
    W /= np.sum(W)

    c_weighted = np.tensordot(C, W, axes=([2, 3], [0, 1]))

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        np.log10(np.clip(c_weighted.T, 1e-12, None)),
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
    cbar.set_label(r"Population-weighted $\log\,p(I{=}1\,|\,m,z)$")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "completeness_map_weighted_fhost_alpha.pdf"), dpi=600)
    plt.close(fig)


def _plot_mock_latent_summary_maps(
    m_true,
    z_true,
    fhost_true,
    alpha_true,
    mag_edges,
    z_edges,
    *,
    plot_dir,
):
    import matplotlib.pyplot as plt

    def _binned_mean(x, y, value):
        counts, _, _ = np.histogram2d(x, y, bins=[mag_edges, z_edges])
        sums, _, _ = np.histogram2d(x, y, bins=[mag_edges, z_edges], weights=value)
        mean = np.full_like(sums, np.nan, dtype=float)
        mean[counts > 0] = sums[counts > 0] / counts[counts > 0]
        return mean

    def _binned_std(x, y, value):
        counts, _, _ = np.histogram2d(x, y, bins=[mag_edges, z_edges])
        sums, _, _ = np.histogram2d(x, y, bins=[mag_edges, z_edges], weights=value)
        sums2, _, _ = np.histogram2d(x, y, bins=[mag_edges, z_edges], weights=value**2)
        mean = np.full_like(sums, np.nan, dtype=float)
        var = np.full_like(sums, np.nan, dtype=float)
        mask = counts > 0
        mean[mask] = sums[mask] / counts[mask]
        var[mask] = np.maximum(sums2[mask] / counts[mask] - mean[mask] ** 2, 0.0)
        return np.sqrt(var)

    products = [
        (_binned_mean(m_true, z_true, fhost_true), "mock_fhost_mean_map.pdf", r"Mean $f_{\rm host}$"),
        (_binned_std(m_true, z_true, fhost_true), "mock_fhost_std_map.pdf", r"$\sigma(f_{\rm host})$"),
        (_binned_mean(m_true, z_true, alpha_true), "mock_alpha_lambda_mean_map.pdf", r"Mean $\alpha_{\lambda}$"),
        (_binned_std(m_true, z_true, alpha_true), "mock_alpha_lambda_std_map.pdf", r"$\sigma(\alpha_{\lambda})$"),
    ]

    for arr, filename, label in products:
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(
            arr.T,
            origin="lower",
            aspect="auto",
            extent=[mag_edges[0], mag_edges[-1], z_edges[0], z_edges[-1]],
            cmap="viridis",
        )
        ax.set_ylabel(r"$z$")
        ax.set_xlabel(r"$m_{2500\,\mathrm{\AA}}$ (mag)")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(label)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, filename), dpi=600)
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
    df_agn_fhost_population=None,
):
    """
    Build p(detect | m, z, f_host_2500_psf), preferring mock PSF host
    fractions when available.
    """
    import matplotlib.pyplot as plt

    if COMPLETENESS_FHOST_COL not in df_agn.columns:
        raise KeyError(
            f"df_agn must contain {COMPLETENESS_FHOST_COL!r} for 3D host-aware completeness."
        )

    sim_file = resolve_qvc_data_path(sim_file)
    with h5py.File(sim_file, "r") as f:
        if "apparent_mag_2500" in f:
            m_true = np.asarray(f["apparent_mag_2500"][:], dtype=float)
        else:
            m_true = np.asarray(f["apparent_mag_i_rest"][:], dtype=float)
        z_true = np.asarray(f["z"][:], dtype=float)
        fhost_true_raw, fhost_source = _read_mock_fhost_2500_psf(f)
        if fhost_true_raw is not None and fhost_true_raw.shape != m_true.shape:
            print(
                "[WARNING] Ignoring mock PSF f_host_2500 dataset because its shape "
                f"{fhost_true_raw.shape} does not match apparent magnitude shape {m_true.shape}."
            )
            fhost_true_raw, fhost_source = None, None
        mock_count_scale = f.attrs.get("mock_count_scale")

    z_obs = df_agn["z"].to_numpy(dtype=float)
    magnitude_col = resolve_completeness_magnitude_column(df_agn)
    m_obs = df_agn[magnitude_col].to_numpy(dtype=float)
    fhost_obs = df_agn[COMPLETENESS_FHOST_COL].to_numpy(dtype=float)

    ok_obs = (
        np.isfinite(m_obs)
        & np.isfinite(z_obs)
        & np.isfinite(fhost_obs)
        & (fhost_obs >= 0.0)
        & (fhost_obs <= 1.0)
    )
    if fhost_true_raw is None:
        ok_true = np.isfinite(m_true) & np.isfinite(z_true)
    else:
        ok_true = (
            np.isfinite(m_true)
            & np.isfinite(z_true)
            & np.isfinite(fhost_true_raw)
            & (fhost_true_raw >= 0.0)
            & (fhost_true_raw <= 1.0)
        )
    m_obs, z_obs, fhost_obs = m_obs[ok_obs], z_obs[ok_obs], fhost_obs[ok_obs]
    m_true, z_true = m_true[ok_true], z_true[ok_true]

    host_model = _fit_fhost_population_model(
        df_agn.loc[ok_obs],
        df_agn_fhost_population,
        fit_logL_max=fit_logL_max,
    )
    rng = np.random.default_rng(12345)
    if fhost_true_raw is None:
        logL_true = apparent_mag_to_logL2500(m_true, z_true, COSMO)
        fhost_true = sample_fhost_2500_from_logL2500(logL_true, host_model, rng)
        host_model["source"] = "observed_fhost_model"
    else:
        fhost_true = np.clip(fhost_true_raw[ok_true], 0.0, 1.0)
        host_model["source"] = fhost_source
        host_model["n_mock"] = int(np.size(fhost_true))
        host_model["mock_fhost_mean"] = float(np.nanmean(fhost_true)) if np.size(fhost_true) else np.nan
        host_model["mock_fhost_sigma"] = float(np.nanstd(fhost_true, ddof=1)) if np.size(fhost_true) > 1 else 0.0

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
    C = _scaled_completeness_ratio(
        H_obs_s,
        H_true_s,
        mag_centers,
        label="Completeness3D",
        count_scale=mock_count_scale,
        eps=eps,
    )

    completeness3d = Completeness3D(mag_centers, z_centers, fhost_centers, C)

    if plot:
        base_plot_path = plot_path or "plots/hubble"
        plot_dir = os.path.join(base_plot_path, "completeness")
        os.makedirs(plot_dir, exist_ok=True)

        with open(os.path.join(plot_dir, "fhost_2500_psf_l2500_model.json"), "w") as handle:
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
        ax.set_ylabel(r"$f_{\rm host,2500}$")
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
    n_fhost_bins=12,
    n_alpha_bins=12,
    smooth_counts=True,
    plot=False,
    plot_path=None,
    fit_logL_max=45.5,
    sigma_mag=0.2,
    sigma_z_abs=0.2,
    sigma_fhost=0.1,
    sigma_alpha=0.35,
    df_agn_fhost_population=None,
):
    """
    Build p(detect | m, z, f_host_2500_psf, alpha_lambda), preferring mock
    PSF host fractions and alpha_lambda populations when available.
    """
    import matplotlib.pyplot as plt

    required = {
        COMPLETENESS_FHOST_COL,
        "alpha_lambda",
        "z",
    }
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
        fhost_true_raw, fhost_source = _read_mock_fhost_2500_psf(f)
        if fhost_true_raw is not None and fhost_true_raw.shape != m_true.shape:
            print(
                "[WARNING] Ignoring mock PSF f_host_2500 dataset because its shape "
                f"{fhost_true_raw.shape} does not match apparent magnitude shape {m_true.shape}."
            )
            fhost_true_raw, fhost_source = None, None
        alpha_true_raw, alpha_source = _read_mock_alpha_lambda(f)
        if alpha_true_raw is not None and alpha_true_raw.shape != m_true.shape:
            print(
                "[WARNING] Ignoring mock alpha_lambda dataset because its shape "
                f"{alpha_true_raw.shape} does not match apparent magnitude shape {m_true.shape}."
            )
            alpha_true_raw, alpha_source = None, None
        alpha_attr_model = _alpha_lambda_model_from_h5_attrs(f.attrs)
        mock_count_scale = f.attrs.get("mock_count_scale")

    z_obs = df_agn["z"].to_numpy(dtype=float)
    magnitude_col = resolve_completeness_magnitude_column(df_agn)
    m_obs = df_agn[magnitude_col].to_numpy(dtype=float)
    fhost_obs = df_agn[COMPLETENESS_FHOST_COL].to_numpy(dtype=float)
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
    if fhost_true_raw is not None:
        ok_true &= (
            np.isfinite(fhost_true_raw)
            & (fhost_true_raw >= 0.0)
            & (fhost_true_raw <= 1.0)
        )
    if alpha_true_raw is not None:
        ok_true &= (
            np.isfinite(alpha_true_raw)
            & (alpha_true_raw >= _ALPHA_MIN)
            & (alpha_true_raw <= _ALPHA_MAX)
        )
    m_obs, z_obs, fhost_obs, alpha_obs = m_obs[ok_obs], z_obs[ok_obs], fhost_obs[ok_obs], alpha_obs[ok_obs]
    m_true, z_true = m_true[ok_true], z_true[ok_true]
    if alpha_true_raw is not None:
        alpha_true = np.clip(alpha_true_raw[ok_true], _ALPHA_MIN, _ALPHA_MAX)
        alpha_model = _alpha_lambda_model_from_values(
            alpha_true,
            source=alpha_source,
            count_key="n_mock",
        )
    else:
        alpha_model = alpha_attr_model or fit_alpha_lambda_population(df_agn.loc[ok_obs])

    host_model = _fit_fhost_population_model(
        df_agn.loc[ok_obs],
        df_agn_fhost_population,
        fit_logL_max=fit_logL_max,
    )
    rng = np.random.default_rng(12345)
    if fhost_true_raw is None:
        logL_true = apparent_mag_to_logL2500(m_true, z_true, COSMO)
        fhost_true = sample_fhost_2500_from_logL2500(logL_true, host_model, rng)
        host_model["source"] = "observed_fhost_model"
    else:
        fhost_true = np.clip(fhost_true_raw[ok_true], 0.0, 1.0)
        host_model["source"] = fhost_source
        host_model["n_mock"] = int(np.size(fhost_true))
        host_model["mock_fhost_mean"] = float(np.nanmean(fhost_true)) if np.size(fhost_true) else np.nan
        host_model["mock_fhost_sigma"] = float(np.nanstd(fhost_true, ddof=1)) if np.size(fhost_true) > 1 else 0.0
    if alpha_true_raw is None:
        alpha_true = sample_alpha_lambda_population(np.shape(m_true), alpha_model, rng)

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
    C = _scaled_completeness_ratio(
        H_obs_s,
        H_true_s,
        mag_centers,
        label="Completeness4D",
        count_scale=mock_count_scale,
        eps=eps,
    )

    completeness4d = Completeness4D(mag_centers, z_centers, fhost_centers, alpha_centers, C)

    if plot:
        base_plot_path = plot_path or "plots/hubble"
        plot_dir = os.path.join(base_plot_path, "completeness")
        os.makedirs(plot_dir, exist_ok=True)

        with open(os.path.join(plot_dir, "fhost_2500_psf_l2500_model.json"), "w") as handle:
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

        _plot_fixed_slice_completeness_map_4d(
            completeness4d,
            mag_edges,
            z_edges,
            mag_centers,
            z_centers,
            fhost_centers,
            alpha_centers,
            plot_dir=plot_dir,
            fhost_slice=float(np.nanmedian(fhost_obs)),
            alpha_slice=float(np.nanmedian(alpha_obs)),
        )

        _plot_weighted_marginal_completeness_map_4d(
            C,
            mag_edges,
            z_edges,
            fhost_true,
            alpha_true,
            fhost_edges,
            alpha_edges,
            plot_dir=plot_dir,
        )

        _plot_mock_latent_summary_maps(
            m_true,
            z_true,
            fhost_true,
            alpha_true,
            mag_edges,
            z_edges,
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
        cbar.set_label(r"Unweighted mean over $f_{\rm host}, \alpha_{\lambda}$: $\log\,p(I{=}1\,|\,m,z,f_{\rm host},\alpha_{\lambda})$")
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
        alpha_model,
    )


import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RegularGridInterpolator


def make_dm_function(
    m,
    z,
    dm,
    m_bins=40,
    z_bins=40,
    f_host_2500_psf=None,
    f_host_2500=None,
    alpha_lambda=None,
    *,
    lowess_frac=0.25,
    lowess_it=1,
    lowess_min_points=10,
):
    """
    Build a mean debias interpolator for plotting/debiased magnitudes.

    The returned callable accepts the historical point-array interface used by
    the plotting code:
      [z, m_2500]
      [z, m_2500, f_host_2500_psf]
      [z, m_2500, f_host_2500_psf, alpha_lambda]

    If host/color columns are provided, they are included in the interpolation.
    Internal completeness callers use the PSF host fraction; the old
    f_host_2500 keyword remains only as a compatibility alias.
    """
    m = np.asarray(m, dtype=float)
    z = np.asarray(z, dtype=float)
    dm = np.asarray(dm, dtype=float)
    mask = np.isfinite(m) & np.isfinite(z) & np.isfinite(dm)
    fhost = f_host_2500_psf if f_host_2500_psf is not None else f_host_2500
    if fhost is not None:
        fhost = np.asarray(fhost, dtype=float)
        mask &= np.isfinite(fhost)
    if alpha_lambda is not None:
        alpha_lambda = np.asarray(alpha_lambda, dtype=float)
        mask &= np.isfinite(alpha_lambda)

    m, z, dm = m[mask], z[mask], dm[mask]
    if fhost is not None:
        fhost = fhost[mask]
    if alpha_lambda is not None:
        alpha_lambda = alpha_lambda[mask]
    if z.size == 0:
        raise ValueError("make_dm_function requires at least one finite (m, z, dm) point.")

    if fhost is not None:
        feature_cols = [z, m, fhost]
        if alpha_lambda is not None:
            feature_cols.append(alpha_lambda)
        train_pts = np.column_stack(feature_cols)
        center = np.median(train_pts, axis=0)
        scale = 0.5 * (
            np.percentile(train_pts, 84, axis=0)
            - np.percentile(train_pts, 16, axis=0)
        )
        scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
        train_pts_scaled = (train_pts - center) / scale
        try:
            linear_interp = LinearNDInterpolator(train_pts_scaled, dm, fill_value=np.nan)
        except Exception as exc:
            print(
                "Warning: failed to build LinearNDInterpolator for debias correction; "
                f"falling back to nearest-neighbor interpolation only. Error: {exc}"
            )
            linear_interp = None
        nearest_interp = NearestNDInterpolator(train_pts_scaled, dm)

        def interp_features(pts):
            pts = np.asarray(pts, dtype=float)
            arr = np.atleast_2d(pts)
            expected_dim = train_pts.shape[1]
            if arr.shape[1] < expected_dim:
                raise ValueError(
                    f"dm_interp expected at least {expected_dim} columns, got {arr.shape[1]}."
                )
            finite = np.all(np.isfinite(arr[:, :expected_dim]), axis=1)
            out = np.full(arr.shape[0], np.nan, dtype=float)
            if not np.any(finite):
                return out if np.ndim(pts) > 1 else out[0]
            arr_scaled = (arr[:, :expected_dim] - center) / scale
            arr_scaled_finite = arr_scaled[finite]
            if linear_interp is None:
                finite_out = np.full(arr_scaled_finite.shape[0], np.nan, dtype=float)
            else:
                finite_out = np.asarray(linear_interp(arr_scaled_finite), dtype=float)
            missing = ~np.isfinite(finite_out)
            if np.any(missing):
                finite_out[missing] = np.asarray(nearest_interp(arr_scaled_finite[missing]), dtype=float)
            out[finite] = finite_out
            return out if np.ndim(pts) > 1 else out[0]

        return interp_features

    trend = build_smooth_trend_1d(
        z,
        dm,
        frac=lowess_frac,
        it=lowess_it,
        min_points=lowess_min_points,
        fallback_bins=z_bins,
    )

    def interp_zonly(pts):
        pts = np.asarray(pts)
        arr = np.atleast_2d(pts).astype(float)
        out = np.asarray(trend(arr[:, 0]), dtype=float)
        return out if np.ndim(pts) > 1 else out[0]

    return interp_zonly

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
