import argparse
import gzip
import importlib
import io
import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq


COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)
L_SUN_ERG_S = 3.828e33
L0 = 1e10 * L_SUN_ERG_S
LOG10_MAG_JACOBIAN = np.log10(0.4)
NU_2500_HZ = 2.99792458e18 / 2500.0
AB_ABSOLUTE_MAG_ZEROPOINT = 51.59477721004232


def log_nu_lnu_to_ab_absolute_magnitude(log_nu_lnu, frequency_hz):
    """Convert log10(nu L_nu / erg s^-1) to monochromatic absolute AB magnitude."""
    log_lnu = np.asarray(log_nu_lnu, dtype=float) - np.log10(float(frequency_hz))
    return AB_ABSOLUTE_MAG_ZEROPOINT - 2.5 * log_lnu


def _candidate_existing_path(*candidates):
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.exists():
            return path
    return None


def _discover_qvc_root() -> Path:
    """Find the repository root by walking upward to the pyproject file."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]


def default_shen_pubtools_path():
    qvc_root = _discover_qvc_root()
    return _candidate_existing_path(
        os.environ.get("SHEN_PUBTOOLS_PATH"),
        os.environ.get("HOPKINS_PUBTOOLS_PATH"),
        qvc_root / "quasarlf" / "pubtools",
        qvc_root.parent / "quasarlf" / "pubtools",
        qvc_root.parent / "quasarlf" / "pubtools",
        qvc_root / "external" / "quasarlf" / "pubtools",
    )


def default_ananna_xlf_path():
    qvc_root = _discover_qvc_root()
    return _candidate_existing_path(
        os.environ.get("ANANNA_XLF_PATH"),
        qvc_root / "ananna_xlf" / "final_sol_all.npy.gz",
        qvc_root.parent / "ananna_xlf" / "final_sol_all.npy.gz",
        Path.home() / "ananna_xlf" / "final_sol_all.npy.gz",
    )


@contextmanager
def _temporary_cwd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _configure_shen_paths(shen_config, pubtools_path):
    """Point Shen's module-level paths at the selected pubtools checkout."""
    homepath = f"{Path(pubtools_path).resolve()}{os.sep}"
    shen_config.homepath = homepath
    shen_config.datapath = f"{homepath}data{os.sep}"
    return f"{homepath}obdata_copy{os.sep}"


def bolometric_correction_shen20(
    L_bol,
    c1=4.073,
    k1=-0.026,
    c2=12.60,
    k2=0.278,
):
    x = L_bol / L0
    return c1 * x**k1 + c2 * x**k2


def lbol_from_loglx_shen20(loglx_array):
    loglx_array = np.asarray(loglx_array, dtype=float)
    lx_array = 10.0**loglx_array
    lbol_array = np.empty_like(lx_array)

    for i, lx in enumerate(lx_array):
        def equation(lb):
            return lb - bolometric_correction_shen20(lb) * lx

        lbol_array[i] = brentq(equation, lx, 1e4 * lx)

    return np.log10(lbol_array), lbol_array


def build_shen_lf(pubtools_path):
    if pubtools_path is None:
        pubtools_path = default_shen_pubtools_path()
    if pubtools_path is None:
        raise FileNotFoundError(
            "Shen pubtools path not found. Pass --shen-pubtools-path or set SHEN_PUBTOOLS_PATH."
        )
    pubtools_path = Path(pubtools_path).expanduser().resolve()
    if not pubtools_path.exists():
        raise FileNotFoundError(f"Shen pubtools path not found: {pubtools_path}")

    sys.path.insert(0, str(pubtools_path))
    added_obdata_path = None
    try:
        with _temporary_cwd(pubtools_path):
            silent_stream = io.StringIO()
            with redirect_stdout(silent_stream), redirect_stderr(silent_stream):
                config_path = pubtools_path / "config.py"
                if config_path.is_file():
                    shen_config = importlib.import_module("config")
                    added_obdata_path = _configure_shen_paths(
                        shen_config,
                        pubtools_path,
                    )
                    sys.path.insert(0, added_obdata_path)
                from utilities import return_qlf_in_band

                z_bins = np.linspace(0.0, 8.0, 40)
                qlf_values = [
                    return_qlf_in_band(redshift=z, nu=NU_2500_HZ, model="B")
                    for z in z_bins
                ]
    finally:
        if added_obdata_path is not None:
            try:
                sys.path.remove(added_obdata_path)
            except ValueError:
                pass
        try:
            sys.path.remove(str(pubtools_path))
        except ValueError:
            pass

    luminosities = np.asarray(qlf_values[0][0], dtype=float)
    phi_log10 = np.asarray([qlf[1] for qlf in qlf_values], dtype=float) + LOG10_MAG_JACOBIAN
    # Evaluate the physical 2500 A channel rather than Shen's nu=0 identity
    # channel.  This includes the SED/bolometric-correction scatter and the
    # N_H-dependent extinction model, so the mock parent is the optically
    # detectable population rather than the total intrinsic bolometric QLF.
    # m_grid is monochromatic rest-frame absolute AB M_2500, converted from
    # Shen's nu*L_nu(2500 A); it is not apparent, bolometric, or band-integrated.
    m_grid = log_nu_lnu_to_ab_absolute_magnitude(luminosities, NU_2500_HZ)
    return phi_log10, m_grid, z_bins


def build_ananna_lf(ananna_xlf_path):
    if ananna_xlf_path is None:
        ananna_xlf_path = default_ananna_xlf_path()
    if ananna_xlf_path is None:
        raise FileNotFoundError(
            "Ananna XLF file not found. Pass --ananna-xlf-path or set ANANNA_XLF_PATH."
        )
    ananna_xlf_path = Path(ananna_xlf_path).expanduser().resolve()
    if not ananna_xlf_path.exists():
        raise FileNotFoundError(f"Ananna XLF file not found: {ananna_xlf_path}")

    lumbins = np.linspace(41.0, 47.0, 150)
    zbin1 = np.arange(0.002, 0.1, 0.005)
    zbin2 = np.arange(0.1, 1.0, 0.05)
    zbin3 = np.arange(1.0, 5.05, 0.05)
    zbins = np.concatenate([zbin1, zbin2, zbin3])
    nhbins = np.linspace(20.0, 26.0, 80)

    with gzip.GzipFile(ananna_xlf_path, "r") as handle:
        loading_matr = np.load(handle)

    lum_func = RegularGridInterpolator(
        (zbins, lumbins, nhbins),
        loading_matr[0],
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    mask_unobs = (nhbins >= 20.0) & (nhbins < 22.0)
    nh_unobs = nhbins[mask_unobs]
    z_bins = np.linspace(0.01, 4.5, 20)
    lx_grid = lumbins

    phi_unobs_2d = np.zeros((len(z_bins), len(lx_grid)))
    for i, z in enumerate(z_bins):
        zz, ll, nh = np.meshgrid(np.array([z]), lx_grid, nh_unobs, indexing="ij")
        pts = np.column_stack([zz.ravel(), ll.ravel(), nh.ravel()])
        phi_3d = lum_func(pts).reshape(1, len(lx_grid), len(nh_unobs))
        phi_unobs_2d[i, :] = np.trapezoid(phi_3d[0], x=nh_unobs, axis=-1)

    loglbol_grid, _ = lbol_from_loglx_shen20(lx_grid)
    jac = np.empty_like(lx_grid)
    jac[1:-1] = (lx_grid[2:] - lx_grid[:-2]) / (loglbol_grid[2:] - loglbol_grid[:-2])
    jac[0] = (lx_grid[1] - lx_grid[0]) / (loglbol_grid[1] - loglbol_grid[0])
    jac[-1] = (lx_grid[-1] - lx_grid[-2]) / (loglbol_grid[-1] - loglbol_grid[-2])

    phi_bol_lin = np.clip(phi_unobs_2d, 1e-40, None) * jac[None, :]
    phi_bol_log10 = np.log10(np.clip(phi_bol_lin, 1e-40, None)) + LOG10_MAG_JACOBIAN
    m_grid = 91.0 - 2.5 * loglbol_grid
    return phi_bol_log10, m_grid, z_bins


def mock_m_per_zbin(
    phi_log10,
    m_grid,
    z_bins,
    area_deg2,
    alpha_nu,
    dalpha_nu,
    cosmo,
    *,
    z_res=512,
    m_scatter=0.0,
    kcorr_zref=2.0,
    completeness=None,
    m_lim=None,
    thinning_probability=1.0,
    rng=None,
    return_z=False,
    return_global=False,
    return_alpha=False,
    verbose=False,
):
    rng = np.random.default_rng() if rng is None else rng
    thinning_probability = float(thinning_probability)
    if not np.isfinite(thinning_probability) or not (0.0 < thinning_probability <= 1.0):
        raise ValueError(
            "thinning_probability must be finite and in (0, 1], "
            f"got {thinning_probability}."
        )

    m_grid = np.asarray(m_grid, dtype=float)
    order = np.argsort(m_grid)
    m_grid = m_grid[order]
    n_mag = len(m_grid)

    z_bins = np.asarray(z_bins, dtype=float)
    z_mids = 0.5 * (z_bins[:-1] + z_bins[1:])

    phi = np.asarray(phi_log10, dtype=float)
    if phi.shape == (len(z_mids), len(order)):
        z_support = z_mids
    elif phi.shape == (len(z_bins), len(order)):
        z_support = z_bins
    elif phi.shape == (len(order), len(z_mids)):
        phi = phi.T
        z_support = z_mids
    elif phi.shape == (len(order), len(z_bins)):
        phi = phi.T
        z_support = z_bins
    else:
        raise ValueError("phi_log10 has incompatible shape for the supplied z and magnitude grids.")

    phi = phi[:, order]
    rgi = RegularGridInterpolator(
        (z_support, m_grid),
        phi,
        method="linear",
        bounds_error=False,
        fill_value=-np.inf,
    )

    edges = np.empty(n_mag + 1)
    edges[1:-1] = 0.5 * (m_grid[1:] + m_grid[:-1])
    edges[0] = m_grid[0] - 0.5 * (m_grid[1] - m_grid[0])
    edges[-1] = m_grid[-1] + 0.5 * (m_grid[-1] - m_grid[-2])
    dm = np.diff(edges)

    area_sr = area_deg2 * (np.pi / 180.0) ** 2
    per_z_m = []
    per_z_m_rest = []
    per_z = []
    per_z_alpha_lambda = []
    nexp_per_bin = np.zeros(len(z_bins) - 1)
    nsel_per_bin = np.zeros(len(z_bins) - 1, dtype=int)

    for i, (z1, z2) in enumerate(zip(z_bins[:-1], z_bins[1:])):
        z = np.linspace(z1, z2, z_res)
        dvdz = cosmo.differential_comoving_volume(z).to_value(u.Mpc**3 / u.sr) * area_sr
        zz = np.repeat(z, n_mag)
        mm = np.tile(m_grid, z.size)
        logphi_flat = rgi(np.column_stack([zz, mm]))
        phi_flat = np.where(np.isfinite(logphi_flat), 10.0**logphi_flat, 0.0)
        phi_zm = phi_flat.reshape(z.size, n_mag)
        phi_int_z = np.sum(phi_zm * dm[None, :], axis=1)

        nexp = np.trapezoid(phi_int_z * dvdz, z)
        nexp_per_bin[i] = nexp
        n_draw = rng.poisson(nexp * thinning_probability)
        if n_draw == 0 or not np.isfinite(nexp) or nexp <= 0:
            per_z_m.append(np.empty(0, dtype=float))
            per_z_m_rest.append(np.empty(0, dtype=float))
            per_z.append(np.empty(0, dtype=float))
            per_z_alpha_lambda.append(np.empty(0, dtype=float))
            continue

        wz = phi_int_z * dvdz
        cdf_z = np.cumsum(wz)
        if cdf_z[-1] <= 0 or not np.isfinite(cdf_z[-1]):
            per_z_m.append(np.empty(0, dtype=float))
            per_z_m_rest.append(np.empty(0, dtype=float))
            per_z.append(np.empty(0, dtype=float))
            per_z_alpha_lambda.append(np.empty(0, dtype=float))
            continue
        cdf_z /= cdf_z[-1]
        z_samp = np.interp(rng.random(n_draw), cdf_z, z)

        zzs = np.repeat(z_samp, n_mag)
        mm = np.tile(m_grid, z_samp.size)
        logphi_flat = rgi(np.column_stack([zzs, mm]))
        phi_flat = np.where(np.isfinite(logphi_flat), 10.0**logphi_flat, 0.0)
        phi_m_given_z = phi_flat.reshape(z_samp.size, n_mag)
        w_m = phi_m_given_z * dm[None, :]
        row_sum = w_m.sum(axis=1)
        valid = row_sum > 0
        z_samp = z_samp[valid]
        w_m = w_m[valid]
        row_sum = row_sum[valid]
        if z_samp.size == 0:
            per_z_m.append(np.empty(0, dtype=float))
            per_z_m_rest.append(np.empty(0, dtype=float))
            per_z.append(np.empty(0, dtype=float))
            per_z_alpha_lambda.append(np.empty(0, dtype=float))
            continue

        cdf_m = np.cumsum(w_m, axis=1) / row_sum[:, None]
        u_rand = rng.random(z_samp.size)
        idx = np.sum(cdf_m < u_rand[:, None], axis=1)
        idx = np.clip(idx, 0, n_mag - 1)
        cdf_lo = np.zeros(z_samp.size, dtype=float)
        use_lo = idx > 0
        cdf_lo[use_lo] = cdf_m[np.arange(z_samp.size)[use_lo], idx[use_lo] - 1]
        cdf_hi = cdf_m[np.arange(z_samp.size), idx]
        m_lo = edges[idx]
        m_hi = edges[idx + 1]
        t = (u_rand - cdf_lo) / (cdf_hi - cdf_lo + 1e-12)
        m_abs = m_lo + t * (m_hi - m_lo)

        dm_s = 5.0 * np.log10(cosmo.luminosity_distance(z_samp).to_value(u.pc)) - 5.0
        alpha_nu_samp = alpha_nu + rng.normal(0.0, dalpha_nu, size=z_samp.shape)
        alpha_lambda_samp = -alpha_nu_samp - 2.0
        m_intrinsic = m_abs

        if kcorr_zref is None:
            kcorr = -2.5 * (1.0 + alpha_nu_samp) * np.log10(1.0 + z_samp)
        else:
            kcorr = -2.5 * (1.0 + alpha_nu_samp) * np.log10((1.0 + z_samp) / (1.0 + kcorr_zref))

        m_obs = m_intrinsic + dm_s + kcorr
        lam_i_obs = 7480.0
        lam_2500 = 2500.0
        lam_i_rest = lam_i_obs / (1.0 + kcorr_zref)
        color_term = -2.5 * alpha_nu_samp * np.log10(lam_i_rest / lam_2500)
        m_2500_abs = m_intrinsic + color_term
        m_2500_obs = m_2500_abs + dm_s

        if m_scatter > 0:
            m_obs = m_obs + rng.normal(0.0, m_scatter, size=m_obs.size)
            m_2500_obs = m_2500_obs + rng.normal(0.0, m_scatter, size=m_2500_obs.size)

        if m_lim is not None and m_obs.size > 0:
            keep = m_obs < m_lim
            m_obs = m_obs[keep]
            m_2500_obs = m_2500_obs[keep]
            z_samp = z_samp[keep]
            alpha_lambda_samp = alpha_lambda_samp[keep]

        if completeness is not None and m_obs.size > 0:
            p = np.clip(completeness(m_obs, z_samp), 0.0, 1.0)
            keep = rng.random(m_obs.size) < p
            m_obs = m_obs[keep]
            m_2500_obs = m_2500_obs[keep]
            z_samp = z_samp[keep]
            alpha_lambda_samp = alpha_lambda_samp[keep]

        per_z_m.append(m_obs)
        per_z_m_rest.append(m_2500_obs)
        per_z.append(z_samp)
        per_z_alpha_lambda.append(alpha_lambda_samp)
        nsel_per_bin[i] = m_obs.size

    if not (return_z or return_global):
        out = (per_z_m, nexp_per_bin)
        if return_alpha:
            out = out + (per_z_alpha_lambda,)
        return out

    out = (per_z_m, nexp_per_bin, per_z, nsel_per_bin)
    if return_global:
        nonempty = [i for i, arr in enumerate(per_z_m) if len(arr)]
        if nonempty:
            z_all = np.concatenate([per_z[i] for i in nonempty])
            m_all = np.concatenate([per_z_m[i] for i in nonempty])
            m_rest_all = np.concatenate([per_z_m_rest[i] for i in nonempty])
            bin_index = np.concatenate([np.full(len(per_z_m[i]), i, dtype=int) for i in nonempty])
            alpha_lambda_all = np.concatenate([per_z_alpha_lambda[i] for i in nonempty])
        else:
            z_all = np.empty(0, dtype=float)
            m_all = np.empty(0, dtype=float)
            m_rest_all = np.empty(0, dtype=float)
            bin_index = np.empty(0, dtype=int)
            alpha_lambda_all = np.empty(0, dtype=float)
        out = out + (z_all, m_all, m_rest_all, bin_index)
        if return_alpha:
            out = out + (alpha_lambda_all,)
    elif return_alpha:
        out = out + (per_z_alpha_lambda,)
    return out


def save_mock_catalog(
    output_path,
    z_all,
    m_all,
    m_2500_all,
    m_limit=None,
    *,
    alpha_lambda_all=None,
    thinning_probability=1.0,
    rng=None,
    area_deg2=None,
    alpha_nu_parent_mean=None,
    alpha_nu_parent_sigma=None,
):
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    thinning_probability = float(thinning_probability)
    if not np.isfinite(thinning_probability) or not (0.0 < thinning_probability <= 1.0):
        raise ValueError(
            "thinning_probability must be finite and in (0, 1], "
            f"got {thinning_probability}."
        )
    mask = np.ones_like(z_all, dtype=bool)
    if m_limit is not None:
        mask &= m_all < m_limit
    alpha_lambda_all = None if alpha_lambda_all is None else np.asarray(alpha_lambda_all, dtype=float)
    if alpha_lambda_all is not None and alpha_lambda_all.shape != np.shape(z_all):
        raise ValueError(
            "alpha_lambda_all must have the same shape as z_all; "
            f"got {alpha_lambda_all.shape} and {np.shape(z_all)}."
        )
    n_before_thin = int(np.count_nonzero(mask))
    n_after_thin = int(np.count_nonzero(mask))
    with h5py.File(output_path, "w") as h5file:
        h5file.create_dataset("z", data=z_all[mask])
        h5file.create_dataset("apparent_mag_i", data=m_all[mask])
        # This is an apparent-magnitude proxy at rest-frame 2500 A, not rest-frame i-band.
        h5file.create_dataset("apparent_mag_2500", data=m_2500_all[mask])
        # Keep the legacy key so existing completeness readers do not break.
        h5file.create_dataset("apparent_mag_i_rest", data=m_2500_all[mask])
        if alpha_lambda_all is not None:
            alpha_saved = alpha_lambda_all[mask]
            h5file.create_dataset("alpha_lambda", data=alpha_saved)
            h5file.create_dataset("alpha_nu", data=-alpha_saved - 2.0)
            finite_alpha = alpha_saved[np.isfinite(alpha_saved)]
            if finite_alpha.size > 0:
                h5file.attrs["alpha_lambda_mean"] = float(np.nanmean(finite_alpha))
                h5file.attrs["alpha_lambda_sigma"] = float(np.nanstd(finite_alpha, ddof=1)) if finite_alpha.size > 1 else 0.0
        h5file.attrs["thinning_probability"] = thinning_probability
        h5file.attrs["mock_count_scale"] = 1.0 / thinning_probability
        if alpha_nu_parent_mean is not None and np.isfinite(alpha_nu_parent_mean):
            h5file.attrs["alpha_nu_parent_mean"] = float(alpha_nu_parent_mean)
            h5file.attrs["alpha_lambda_parent_mean"] = float(-alpha_nu_parent_mean - 2.0)
        if alpha_nu_parent_sigma is not None and np.isfinite(alpha_nu_parent_sigma):
            h5file.attrs["alpha_nu_parent_sigma"] = float(abs(alpha_nu_parent_sigma))
            h5file.attrs["alpha_lambda_parent_sigma"] = float(abs(alpha_nu_parent_sigma))
        if area_deg2 is not None and np.isfinite(area_deg2):
            h5file.attrs["area_deg2"] = float(area_deg2)
    print(
        "Saved mock catalog with "
        f"{n_after_thin} / {n_before_thin} sources after m_lim cut "
        f"(p_keep={thinning_probability:.4g}, mock_count_scale={1.0 / thinning_probability:.4g})"
    )


def plot_mock_catalog(z_all, m_values, plot_path, title, ylabel, bin_index):
    plot_path = Path(plot_path).expanduser().resolve()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(z_all, m_values, c=bin_index, cmap="viridis", s=3, linewidths=0, rasterized=True)
    fig.colorbar(sc, ax=ax, label="Redshift Bin Index")
    ax.set_xlabel("Redshift")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, 5)
    ax.set_ylim(15, 30)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate mock completeness catalogs from Shen or Ananna luminosity functions.")
    parser.add_argument("--lf-model", choices=["shen", "ananna"], required=True, help="Luminosity-function model to use.")
    parser.add_argument("--output", required=True, help="Output HDF5 file path.")
    parser.add_argument("--area-deg2", type=float, default=None, help="Survey area in deg^2. Defaults: 5 for Shen, 50 for Ananna.")
    parser.add_argument("--alpha-nu", type=float, default=-0.5, help="Mean spectral index for K-correction.")
    parser.add_argument("--dalpha-nu", type=float, default=0.3, help="Scatter in the spectral index.")
    parser.add_argument("--m-limit", type=float, default=28.0, help="Apparent-magnitude cut applied before saving.")
    parser.add_argument("--m-scatter", type=float, default=0.0, help="Additional Gaussian apparent-magnitude scatter.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--z-res", type=int, default=512, help="Redshift resolution inside each z bin.")
    parser.add_argument("--plot", action="store_true", help="Save a diagnostic scatter plot.")
    parser.add_argument("--plot-path", default=None, help="Optional plot output path.")
    parser.add_argument("--plot-rest", action="store_true", help="Plot rest-frame 2500A apparent magnitudes instead of observed survey-band magnitudes.")
    parser.add_argument(
        "--shen-pubtools-path",
        default=None,
        help="Path to the Shen QLF pubtools directory. If omitted, use SHEN_PUBTOOLS_PATH or common repo-relative locations.",
    )
    parser.add_argument(
        "--ananna-xlf-path",
        default=None,
        help="Path to the gzipped Ananna XLF matrix. If omitted, use ANANNA_XLF_PATH or common local locations.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-bin sampling diagnostics.")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.lf_model == "shen":
        phi_log10, m_grid, z_bins = build_shen_lf(args.shen_pubtools_path)
        area_deg2 = 5.0 if args.area_deg2 is None else args.area_deg2
    else:
        phi_log10, m_grid, z_bins = build_ananna_lf(args.ananna_xlf_path)
        area_deg2 = 50.0 if args.area_deg2 is None else args.area_deg2

    _, nexp, _, nsel, z_all, m_all, m_rest_all, bin_index, alpha_lambda_all = mock_m_per_zbin(
        phi_log10,
        m_grid,
        z_bins,
        area_deg2,
        args.alpha_nu,
        args.dalpha_nu,
        COSMO,
        z_res=args.z_res,
        m_scatter=args.m_scatter,
        kcorr_zref=2.0,
        m_lim=args.m_limit,
        rng=rng,
        return_z=True,
        return_global=True,
        return_alpha=True,
        verbose=args.verbose,
    )

    save_mock_catalog(
        args.output,
        z_all,
        m_all,
        m_rest_all,
        m_limit=args.m_limit,
        alpha_lambda_all=alpha_lambda_all,
        alpha_nu_parent_mean=args.alpha_nu,
        alpha_nu_parent_sigma=args.dalpha_nu,
    )
    print(f"Saved mock catalog to {args.output}")
    print(f"Generated {len(z_all)} total mock sources before save cut.")

    if args.plot:
        if args.plot_path is None:
            stem = Path(args.output).with_suffix("")
            plot_path = f"{stem}_{args.lf_model}.pdf"
        else:
            plot_path = args.plot_path
        if args.plot_rest:
            plot_mock_catalog(
                z_all,
                m_rest_all,
                plot_path,
                f"Mock Survey from {args.lf_model.title()} LF",
                "Apparent Magnitude at 2500A",
                bin_index,
            )
        else:
            plot_mock_catalog(
                z_all,
                m_all,
                plot_path,
                f"Mock Survey from {args.lf_model.title()} LF",
                "Observed-band Apparent Magnitude",
                bin_index,
            )
        print(f"Saved diagnostic plot to {plot_path}")


if __name__ == "__main__":
    main()
