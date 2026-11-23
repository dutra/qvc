"""Component luminosities derived from saved spectral parameter medians.

These are functions of parameter medians, not medians of derived draws.
See scripts/plot_spectral_components_vs_redshift.py for model conventions.
"""

import numpy as np
import pandas as pd

# Shared y range [log10(L/W)] of every luminosity panel. It holds the
# 0.5-99.5 percentile range of all six pre-cut distributions (33.3-40.1);
# points outside are drawn as edge arrows.
LUMINOSITY_YLIM = (33.0, 40.5)
# Raw catalog columns the derived luminosities need.
LUMINOSITY_INPUTS = (
    "log_agn_amp", "fracAGN_5100_fit", "spectral_feii_norm", "spectral_balmer_norm",
    "spectral_balmer_tau", "spectral_feature_amplitude_scale", "log_dust_luminosity_fit",
    "fcov",
)
# Clip bounds of fracAGN_5100_fit in the fit (jaxsedfit model.py:3888-3892).
FRAC_AGN_5100_CLIP = (1.0e-4, 0.999)
FEII_PIVOT_A = 4575.0
# Mean f_lambda of the BruhweilerVerner08 template over rest 4434-4684 A,
# relative to the pixel nearest 4575 A that the fit normalises to 1.
FEII_WINDOW_MEAN = 0.2759
BALMER_PIVOT_A = 3000.0
BALMER_EDGE_A = 3646.0
BALMER_TE_K = 15000.0
C_ANGSTROM_PER_S = 2.99792458e18
MJY_TO_W_M2_HZ = 1.0e-29


def _ylabel(sub, where):
    return rf"$\log_{{10}}(L_{{\rm {sub}}}/{{\rm W}})$" + f"\n{where}"


# (derived column, legend label, y-axis spec), all log10(L / W).
AMPLITUDE_PANELS = (
    ("log_L_feii_4575", r"Fe II",
     {"scale": "linear", "ylim": LUMINOSITY_YLIM,
      "ylabel": _ylabel("FeII", r"mean $\lambda L_\lambda$, 4434-4684 $\rm\AA$, as fitted")}),
    ("log_L_bc_3000", r"Balmer continuum",
     {"scale": "linear", "ylim": LUMINOSITY_YLIM,
      "ylabel": _ylabel("BC", r"$\nu L_\nu$ at 3000 $\rm\AA$, as fitted")}),
    ("log_L_disc_5100", r"AGN continuum (disc)",
     {"scale": "linear", "ylim": LUMINOSITY_YLIM,
      "ylabel": _ylabel("disc", r"$\lambda L_\lambda$ at 5100 $\rm\AA$, intrinsic")}),
    ("log_L_host_5100", r"Galaxy (stars)",
     {"scale": "linear", "ylim": LUMINOSITY_YLIM,
      "ylabel": _ylabel("host", r"$\lambda L_\lambda$ at 5100 $\rm\AA$, intrinsic")}),
    ("log_dust_luminosity_fit", "Galaxy dust\n(energy balance)",
     {"scale": "linear", "ylim": LUMINOSITY_YLIM,
      "ylabel": _ylabel("dust", "integrated")}),
    ("log_L_torus_12um", r"AGN torus",
     {"scale": "linear", "ylim": LUMINOSITY_YLIM,
      "ylabel": _ylabel("torus", r"$\lambda L_\lambda$ at 12 $\mu$m")}),
)


def _luminosity_distance_m(z: np.ndarray) -> np.ndarray:
    """d_L in m with the SED fit's cosmology, astropy FlatLambdaCDM(70, 0.3)."""
    import astropy.units as u
    from astropy.cosmology import FlatLambdaCDM

    z = np.asarray(z, dtype=float)
    out = np.full(z.shape, np.nan)
    ok = np.isfinite(z) & (z >= 0.0)
    out[ok] = FlatLambdaCDM(H0=70.0, Om0=0.3).luminosity_distance(z[ok]).to_value(u.m)
    return out


def _log_nu_l_nu(fnu_mjy, rest_wavelength_a, z, d_l_m):
    """log10(nu L_nu / W) at a rest wavelength from the observed f_nu there."""
    nu_obs = C_ANGSTROM_PER_S / (rest_wavelength_a * (1.0 + z))
    with np.errstate(divide="ignore", invalid="ignore"):
        lum = 4.0 * np.pi * d_l_m**2 * nu_obs * np.asarray(fnu_mjy, dtype=float) * MJY_TO_W_M2_HZ
        return np.where(lum > 0.0, np.log10(np.where(lum > 0.0, lum, 1.0)), np.nan)


def _planck_lambda(wavelength_a, temperature_k):
    h, c, k = 6.62607015e-27, 2.99792458e10, 1.380649e-16
    lam = wavelength_a * 1e-8
    return (2.0 * h * c**2 / lam**5) / np.expm1(h * c / (lam * k * temperature_k))


def add_component_luminosities(frame: pd.DataFrame) -> dict:
    """Add the derived log10(L/W) columns of AMPLITUDE_PANELS; return counts."""
    num = {name: pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
           for name in LUMINOSITY_INPUTS}
    z = pd.to_numeric(frame["z"], errors="coerce").to_numpy(dtype=float)
    d_l = _luminosity_distance_m(z)
    log_amp = num["log_agn_amp"]  # ln(lambda L_lambda,5100 / W), intrinsic disc
    scale = num["spectral_feature_amplitude_scale"]
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["log_L_disc_5100"] = log_amp / np.log(10.0)
        frac = num["fracAGN_5100_fit"]
        lo, hi = FRAC_AGN_5100_CLIP
        at_lo = np.isfinite(frac) & (frac <= lo * (1.0 + 1e-6))
        at_hi = np.isfinite(frac) & (frac >= hi * (1.0 - 1e-9))
        valid = np.isfinite(frac) & ~at_lo & ~at_hi
        frac_ok = np.where(valid, frac, 0.5)
        frame["log_L_host_5100"] = np.where(
            valid, (log_amp + np.log((1.0 - frac_ok) / frac_ok)) / np.log(10.0), np.nan)
        frame["log_L_feii_4575"] = _log_nu_l_nu(
            num["spectral_feii_norm"] / scale * FEII_WINDOW_MEAN, FEII_PIVOT_A, z, d_l)
        tau = num["spectral_balmer_tau"]
        shape = (-np.expm1(-tau * (BALMER_PIVOT_A / BALMER_EDGE_A) ** 3)
                 * _planck_lambda(BALMER_PIVOT_A, BALMER_TE_K)
                 / _planck_lambda(BALMER_EDGE_A, BALMER_TE_K))
        frame["log_L_bc_3000"] = _log_nu_l_nu(
            num["spectral_balmer_norm"] / scale * shape, BALMER_PIVOT_A, z, d_l)
        fcov = num["fcov"]
        frame["log_L_torus_12um"] = np.where(
            fcov > 0.0, np.log10(2.5 * np.where(fcov > 0.0, fcov, 1.0)) + frame["log_L_disc_5100"],
            np.nan)
    return {"frac_nan": int(np.sum(~np.isfinite(frac))), "frac_at_min": int(at_lo.sum()),
            "frac_at_max": int(at_hi.sum())}


