#!/usr/bin/env python
"""Compare SDSS S82 band variability amplitudes with a line-dilution model.

The script measures a model-free variability amplitude in each SDSS band and
compares the band ratios with a toy quasar spectrum in which emission lines,
Fe II, the Balmer continuum and (optionally) host light dilute the continuum
variability amplitude in magnitudes.

1. For every quasar with at least ``--min-epochs`` good epochs in u, g, r and
   i, the excess variance is the light-curve variance minus the mean squared
   photometric error. SDSS observes ugri within minutes, so all bands share
   the same epochs and baseline.
2. In each redshift bin, the ratio log10(sigma_b / sigma_r) uses the median
   excess variance of each band. Errors come from a bootstrap over quasars.
3. The model spectrum is a power-law continuum, Gaussian broad and narrow
   lines with Vanden Berk et al. (2001) composite equivalent widths, the
   JAXSEDFit Fe II template, a Grandi (1982) Balmer continuum and a DSPS
   host. Each component is projected through the JAXSEDFit SDSS filter
   curves (photon weighted) with a mean IGM transmission. The dilution of
   band b is log10(variable flux / total flux), where each component's
   variable flux is its responsivity times its flux.
4. Component strengths and responsivities are grid-fit to the observed
   ratios, with one free offset per band ratio for the intrinsic slope.

Example::

    python scripts/plot_band_amplitude_dilution.py \\
        --output plots/hubble/audit/band_amplitude_ratios_vs_z_full_toy.pdf
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BANDS = ("u", "g", "r", "i")
RATIO_BANDS = ("u", "g", "i")
SDSS_FILTER_IDS = {0: "u", 1: "g", 2: "r", 3: "i"}
# Vanden Berk et al. (2001) composite equivalent widths (A) and FWHM (km/s).
COMPOSITE_LINES = (
    (1216.0, 95.0, 4000.0),   # Lya + N V
    (1400.0, 8.1, 4000.0),    # Si IV + O IV]
    (1549.0, 23.8, 4000.0),   # C IV
    (1909.0, 21.2, 4000.0),   # C III]
    (2800.0, 32.3, 4000.0),   # Mg II
    (3727.0, 1.6, 500.0),     # [O II]
    (4102.0, 5.0, 3000.0),    # Hdelta
    (4340.0, 12.6, 3000.0),   # Hgamma
    (4861.0, 46.2, 3000.0),   # Hbeta
    (4959.0, 8.0, 500.0),     # [O III]
    (5007.0, 24.0, 500.0),    # [O III]
    (6563.0, 194.5, 3000.0),  # Halpha
)
CONTINUUM_ALPHA_LAMBDA = -1.56
BALMER_EDGE = 3646.0
BALMER_TE = 1.5e4
DEFAULT_GRID = {
    "ew_fe": (40.0, 70.0, 100.0, 130.0),
    "f_bc": (0.10, 0.15, 0.20, 0.30),
    "r_line": (0.3, 0.5, 0.7),
    "r_bc": (0.0, 0.3, 0.6),
    "r_fe": (0.0, 0.2),
    "host": (False, True),
}
MODEL_FLOOR_DEX = 0.01


def jaxsedfit_resource_dir() -> Path:
    """Return the JAXSEDFit package resource directory."""
    import jaxsedfit

    return Path(jaxsedfit.__file__).resolve().parent / "resources"


def excess_variance_table(lc: pd.DataFrame, *, min_epochs: int = 20, clip_mad: float = 5.0) -> pd.DataFrame:
    """Return per-object, per-band excess variance from raw SDSS epochs.

    ``lc`` needs the columns ``objectId``, ``psMag``, ``psMagErr`` and
    ``filterID`` (0-3 for ugri). Epochs farther than ``clip_mad`` robust
    standard deviations from the band median are removed first.
    """
    lc = lc[lc.filterID.isin(SDSS_FILTER_IDS)]
    lc = lc[(lc.psMagErr > 0) & (lc.psMagErr < 0.3) & (lc.psMag > 14) & (lc.psMag < 24.5)].copy()
    lc["band"] = lc.filterID.map(SDSS_FILTER_IDS)
    grouped = lc.groupby(["objectId", "band"]).psMag
    median = grouped.transform("median")
    mad = grouped.transform(lambda x: np.median(np.abs(x - np.median(x))))
    lc = lc[np.abs(lc.psMag - median) < clip_mad * 1.4826 * mad + 0.05]
    agg = lc.groupby(["objectId", "band"]).agg(
        n=("psMag", "size"),
        var=("psMag", "var"),
        err2=("psMagErr", lambda e: float(np.mean(np.square(e)))),
        mag=("psMag", "median"),
    )
    agg = agg[agg.n >= min_epochs]
    agg["xs"] = agg["var"] - agg["err2"]
    wide = agg.reset_index().pivot(index="objectId", columns="band", values=["xs", "mag"])
    wide.columns = [f"{name}_{band}" for name, band in wide.columns]
    return wide.reset_index()


def binned_band_ratios(table: pd.DataFrame, edges, *, n_boot: int = 300, seed: int = 1) -> pd.DataFrame:
    """Return log10(sigma_b / sigma_r) per redshift bin from median excess variances."""
    rng = np.random.default_rng(seed)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = table[(table.z > lo) & (table.z <= hi)]
        row = {"z": round(0.5 * (lo + hi), 4), "N": len(sub)}
        for band in RATIO_BANDS:
            a = sub[f"xs_{band}"].to_numpy(float)
            r = sub["xs_r"].to_numpy(float)

            def ratio(index):
                num, den = np.median(a[index]), np.median(r[index])
                return 0.5 * np.log10(num / den) if num > 0 and den > 0 else np.nan

            if len(sub) < 40:
                row[band], row[f"{band}_err"] = np.nan, np.nan
                continue
            row[band] = ratio(np.arange(len(sub)))
            boot = [ratio(rng.integers(0, len(sub), len(sub))) for _ in range(n_boot)]
            row[f"{band}_err"] = float(np.nanstd(boot))
        rows.append(row)
    return pd.DataFrame(rows)


class DilutionToy:
    """Composite quasar spectrum projected through SDSS filters.

    Every component's band flux is linear in its strength, so the band
    integrals are computed once per redshift and reused for every parameter
    set.
    """

    def __init__(self, resource_dir, z_grid, *, ssp_path=None, host_fraction=None):
        resource_dir = Path(resource_dir)
        self.z_grid = np.round(np.asarray(z_grid, dtype=float), 4)
        self.wave = np.arange(700.0, 9500.0, 1.0)
        wr = self.wave
        cont = (wr / 3000.0) ** CONTINUUM_ALPHA_LAMBDA
        lines = np.zeros_like(wr)
        for center, ew, fwhm in COMPOSITE_LINES:
            sigma = center * fwhm / 2.998e5 / 2.355
            profile = np.exp(-0.5 * ((wr - center) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
            lines += ew * np.interp(center, wr, cont) * profile
        fe = np.loadtxt(resource_dir / "templates" / "Fe_d11-m20-20.5.txt")
        fe_raw = np.interp(wr, fe[:, 0], fe[:, 1], left=0.0, right=0.0)
        uv = (wr >= 2200.0) & (wr <= 3090.0)
        fe_unit = fe_raw / np.trapezoid(fe_raw[uv] / cont[uv], wr[uv])  # UV Fe II EW = 1 A
        lam_cm = wr * 1e-8
        planck = 1.0 / lam_cm**5 / np.expm1(1.4388 / (lam_cm * BALMER_TE))
        balmer = np.where(wr <= BALMER_EDGE, planck * (1.0 - np.exp(-((wr / BALMER_EDGE) ** 3))), 0.0)
        balmer = balmer / np.interp(3000.0, wr, balmer) * np.interp(3000.0, wr, cont)  # = continuum at 3000 A
        self.components = {"cont": cont, "line": lines, "fe": fe_unit, "bc": balmer}
        if ssp_path is not None:
            self.components["host"] = self._host_sed(ssp_path)
        self.host_fraction = host_fraction
        window = (wr >= 4000.0) & (wr <= 5000.0)
        self.window = {k: np.trapezoid(v[window], wr[window]) for k, v in self.components.items()}
        self.filters = {
            band: np.loadtxt(resource_dir / "filters" / "sloan" / "sdss" / f"SLOAN_SDSS.{band}.dat")
            for band in BANDS
        }
        self.integrals = {z: self._band_integrals(z) for z in self.z_grid}

    def _host_sed(self, ssp_path, *, tau_gyr=1.5, age_gyr=6.0, metallicity=0.0142):
        """Delayed-tau, solar-metallicity DSPS host spectrum (f_lambda shape)."""
        with h5py.File(ssp_path, "r") as handle:
            wave = handle["ssp_wave"][()]
            lgmet = handle["ssp_lgmet"][()]
            ages = 10 ** handle["ssp_lg_age_gyr"][()]
            flux = handle["ssp_flux"][()]
        imet = int(np.argmin(np.abs(lgmet - np.log10(metallicity))))
        lookback = age_gyr - ages
        weights = np.where(ages <= age_gyr, lookback * np.exp(-lookback / tau_gyr), 0.0) * np.gradient(ages)
        fnu = (weights[:, None] * flux[imet]).sum(axis=0)
        return np.interp(self.wave, wave, fnu / wave**2)

    def _igm(self, z):
        trans = np.ones_like(self.wave)
        forest = self.wave < 1216.0
        trans[forest] = np.exp(-0.0018 * (self.wave[forest] * (1 + z) / 1216.0) ** 3.92)
        trans[self.wave < 912.0] = 0.0
        return trans

    def _band_integrals(self, z):
        trans = self._igm(z)
        obs = self.wave * (1 + z)
        out = {}
        for band, curve in self.filters.items():
            weight = np.interp(obs, curve[:, 0], curve[:, 1], left=0.0, right=0.0) * obs
            usable = np.trapezoid(weight * trans, obs) / np.trapezoid(weight, obs) > 0.3
            out[band] = {
                name: (np.trapezoid(weight * sed * trans, obs) if usable else np.nan)
                for name, sed in self.components.items()
            }
        return out

    def log_dilution(self, params) -> pd.DataFrame:
        """Return log10(variable / total band flux) for each z and band."""
        use_host = bool(params.get("host", False)) and "host" in self.components
        rows = {}
        for z in self.z_grid:
            host_scale = 0.0
            if use_host:
                f_host = float(self.host_fraction(z))
                agn_window = (
                    self.window["cont"] + self.window["line"]
                    + params["ew_fe"] * self.window["fe"] + params["f_bc"] * self.window["bc"]
                )
                host_scale = f_host / (1.0 - f_host) * agn_window / self.window["host"]
            row = {}
            for band, c in self.integrals[z].items():
                total = c["cont"] + c["line"] + params["ew_fe"] * c["fe"] + params["f_bc"] * c["bc"]
                if use_host:
                    total = total + host_scale * c["host"]
                variable = (
                    c["cont"] + params["r_line"] * c["line"]
                    + params["r_fe"] * params["ew_fe"] * c["fe"] + params["r_bc"] * params["f_bc"] * c["bc"]
                )
                row[band] = np.log10(variable / total)
            rows[z] = row
        return pd.DataFrame(rows).T


def chi2_with_offsets(observed: pd.DataFrame, dilution: pd.DataFrame):
    """Return chi2 and the best per-ratio offsets between model and data."""
    obs = observed.set_index(np.round(observed.z, 4))
    total, offsets = 0.0, {}
    for band in RATIO_BANDS:
        y = obs[band]
        err = np.sqrt(obs[f"{band}_err"] ** 2 + MODEL_FLOOR_DEX**2)
        model = (dilution[band] - dilution["r"]).reindex(obs.index)
        ok = np.isfinite(y) & np.isfinite(model) & np.isfinite(err)
        offset = np.sum(((y - model) / err**2)[ok]) / np.sum((1 / err**2)[ok])
        total += float(np.sum((((y - model - offset) / err) ** 2)[ok]))
        offsets[band] = float(offset)
    return total, offsets


def fit_grid(toy: DilutionToy, observed: pd.DataFrame, grid=DEFAULT_GRID):
    """Return (chi2, params, offsets) for the best grid point."""
    if "host" not in toy.components:
        grid = {**grid, "host": (False,)}
    best = None
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        chi2, offsets = chi2_with_offsets(observed, toy.log_dilution(params))
        if best is None or chi2 < best[0]:
            best = (chi2, params, offsets)
    return best


def plot(observed, curves, output):
    """Plot observed band ratios with model curves (label -> dilution, offsets)."""
    from jaxsedfit import mplstyle

    plt.style.use(mplstyle.style_path())
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True, gridspec_kw=dict(hspace=0.05))
    styles = (("--", "C0"), (":", "C1"), ("-", "C3"))
    for ax, band in zip(axes, RATIO_BANDS):
        ax.errorbar(observed.z, observed[band], yerr=observed[f"{band}_err"], fmt="o", ms=4,
                    color="k", label="SDSS S82 excess RMS")
        for (label, (dilution, offsets)), (ls, color) in zip(curves.items(), styles):
            model = dilution[band] - dilution["r"] + offsets[band]
            ax.plot(model.index, model, ls, color=color, label=label)
        ax.set_ylabel(r"$\log_{10}(\sigma_{%s}/\sigma_r)$" % band)
    axes[0].legend(fontsize=8, loc="lower left")
    axes[-1].set_xlabel(r"$z$")
    axes[-1].set_xlim(float(observed.z.min()) - 0.05, float(observed.z.max()) + 0.05)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    from qvc.hubble.hubble_utils import resolve_qvc_data_path

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sdss-lc", default="S82/dr16s82_sdssLCRaw.parquet")
    parser.add_argument("--catalog", default="S82/Catalog.parquet")
    parser.add_argument("--ssp", default="ssp_data_continuum_fsps_v3.2_lgmet_age.h5",
                        help="DSPS SSP table for the host template. Use 'none' to skip the host.")
    parser.add_argument("--r-max", type=float, default=21.0, help="Keep quasars with median r below this.")
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--z-max", type=float, default=3.2)
    parser.add_argument("--output", default="plots/hubble/audit/band_amplitude_ratios_vs_z_full_toy.pdf")
    args = parser.parse_args(argv)

    import pyarrow.parquet as pq

    lc = pq.read_table(resolve_qvc_data_path(args.sdss_lc),
                       columns=["objectId", "psMag", "psMagErr", "filterID"]).to_pandas()
    table = excess_variance_table(lc, min_epochs=args.min_epochs)
    catalog = pd.read_parquet(resolve_qvc_data_path(args.catalog), columns=["objectId", "Z_DR16Q", "qg_fragal"])
    catalog["objectId"] = catalog.objectId.astype(str)
    table["objectId"] = table.objectId.astype(str)
    table = table.merge(catalog, on="objectId").rename(columns={"Z_DR16Q": "z"})
    table = table.dropna(subset=[f"xs_{b}" for b in BANDS])
    table = table[table.mag_r < args.r_max]
    print(f"Quasars with ugri and r < {args.r_max}: {len(table)}")

    edges = np.round(np.arange(0.3, args.z_max + 1e-9, 0.1), 2)
    observed = binned_band_ratios(table, edges)
    observed = observed.dropna(subset=list(RATIO_BANDS), how="all")

    zb = pd.cut(table.z, edges)
    host_med = table.groupby(zb, observed=True).qg_fragal.median()
    host_z = np.array([interval.mid for interval in host_med.index])
    host_fraction = lambda z: float(np.interp(z, host_z, host_med.values))  # noqa: E731

    ssp = None if str(args.ssp).lower() == "none" else resolve_qvc_data_path(args.ssp)
    if ssp is not None and not Path(ssp).exists():
        print(f"DSPS table {ssp} not found. Fitting without a host component.")
        ssp = None
    toy = DilutionToy(jaxsedfit_resource_dir(), observed.z, ssp_path=ssp, host_fraction=host_fraction)

    lines_only = dict(ew_fe=0.0, f_bc=0.0, r_line=0.5, r_bc=0.0, r_fe=0.0, host=False)
    constant_fe_bc = dict(ew_fe=80.0, f_bc=0.15, r_line=0.5, r_bc=0.0, r_fe=0.0, host=False)
    best_chi2, best, _ = fit_grid(toy, observed)
    summary = {"n_points": int(observed[list(RATIO_BANDS)].notna().sum().sum()), "best_params": best, "chi2": {}}
    curves = {}
    for label, params in (
        ("Lines only", lines_only),
        ("Lines + Fe II + BC (constant)", constant_fe_bc),
        ("Best fit" + (" with host" if best["host"] else ", no host"), best),
    ):
        dilution = toy.log_dilution(params)
        chi2, offsets = chi2_with_offsets(observed, dilution)
        summary["chi2"][label] = chi2
        curves[label] = (dilution, offsets)
    if "host" in toy.components:
        summary["chi2"]["Best fit + host"] = chi2_with_offsets(
            observed, toy.log_dilution({**best, "host": True})
        )[0]
    print(json.dumps(summary, indent=2, default=float))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot(observed, curves, output)
    observed.to_csv(output.with_suffix(".csv"), index=False)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=float))
    print(output)


if __name__ == "__main__":
    main()
