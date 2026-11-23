"""Plot saved no-Cepheid SN fits without using SH0ES distance moduli.

Run with the QVC environment; no inference is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM, FlatwCDM, Flatw0waCDM
from astropy.cosmology.realizations import Planck18
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=256)
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("--draws must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = args.data_dir / "Pantheon+SH0ES.dat"
    cov_path = args.data_dir / "Pantheon+SH0ES_STAT+SYS.cov"
    catalog = pd.read_csv(catalog_path, sep=r"\s+")
    mask = (catalog.zHD > 0.01) | catalog.IS_CALIBRATOR.astype(bool)
    covariance = np.loadtxt(cov_path, skiprows=1).reshape(len(catalog), len(catalog))
    errors = np.sqrt(np.diag(covariance)[mask])
    sn = catalog.loc[mask].reset_index(drop=True)
    z = sn.zHD.to_numpy()
    # Put points onto the zHEL=zHD convention used by the smooth model curve.
    correction = 5 * np.log10((1 + sn.zHEL.to_numpy()) / (1 + z))
    magnitude = sn.m_b_corr.to_numpy() - correction
    grid = np.unique(np.concatenate((np.geomspace(1e-4, 5.2, 400),
                                     np.linspace(1e-4, 5.2, 500))))
    style = Path(__file__).resolve().parents[1] / "src/qvc/hubble/style.mplstyle"
    plt.style.use(style)
    plt.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans"})
    manifest = {"catalog": str(catalog_path.resolve()), "catalog_sha256": sha256(catalog_path),
                "covariance_sha256": sha256(cov_path), "n_sn_rows": len(sn),
                "n_calibrator_rows": int(sn.IS_CALIBRATOR.astype(bool).sum()),
                "distance_convention": "m_b_corr - M0_sn - 5 log10((1+zHEL)/(1+zHD))",
                "error_bars": "sqrt(diag(selected STAT+SYS covariance)); correlations not shown",
                "residuals": "median of matched posterior draws of m_b_corr-M0_sn-mu(zHD,zHEL)",
                "binning": "Binned points omitted from all panels",
                "reference": {"model": "FlatLambdaCDM", "H0": 73.6, "Om0": 0.334,
                              "source": "Brout et al. 2022, Table 3; doi:10.3847/1538-4357/ac8e04"},
                "runs": []}
    models = {"FlatLambdaCDM": FlatLambdaCDM, "FlatwCDM": FlatwCDM,
              "Flatw0waCDM": Flatw0waCDM}
    for model, cls in models.items():
        checkpoint = args.checkpoint_dir / f"{model}_sna.h5"
        with h5py.File(checkpoint) as f:
            tag = f["run_tag"][()].decode()
            if "_sna_" not in tag or not any(
                prior_tag in tag
                for prior_tag in ("_nocephdist_planckh0", "_nocephdist_pantheonh0")
            ):
                raise ValueError(f"Not a saved no-Cepheid SN-only run: {tag}")
            labels = [s.decode() for s in f["model_labels"][()]]
            samples = f["flat_samples"][()]
            priors = json.loads(f["prior_bounds_json"][()].decode())
        indices = np.linspace(0, len(samples) - 1, min(args.draws, len(samples)), dtype=int)
        selected = samples[indices]
        params = {key: selected[:, i] for i, key in enumerate(labels)}
        curves, residuals = [], []
        for i in range(len(selected)):
            kwargs = {key: params[key][i] for key in ("H0", "Om0", "w0", "wa") if key in params}
            cosmo = cls(**kwargs)
            curves.append(cosmo.distmod(grid).value)
            residuals.append(magnitude - params["M0_sn"][i] - cosmo.distmod(z).value)
        low, mid, high = np.percentile(curves, [16, 50, 84], axis=0)
        residual = np.median(residuals, axis=0)
        mu = magnitude - np.median(samples[:, labels.index("M0_sn")])
        assert np.all(np.isfinite([low, mid, high])) and np.all(np.isfinite(residual))
        # Match the repository's plot_hubble layout and visual conventions.
        fig = plt.figure(figsize=(9, 7))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
        ax = fig.add_subplot(gs[0])
        res = fig.add_subplot(gs[1], sharex=ax)
        inset = inset_axes(ax, width="40%", height="40%", loc="lower right", borderpad=1.5)
        labels_latex = {"FlatLambdaCDM": r"flat $\Lambda$CDM model",
                        "FlatwCDM": r"flat $w$CDM model",
                        "Flatw0waCDM": r"flat $w_0w_a$CDM model"}
        for axis in (ax, inset):
            axis.errorbar(z, mu, yerr=errors, fmt="s", markersize=2,
                          color="#0A84FF", linestyle="none", lw=0.8,
                          alpha=0.7, zorder=1, label="SN Ia")
            axis.plot(grid, mid, color="m", lw=2.4 if axis is ax else 1.4,
                      zorder=5, label=labels_latex[model])
            axis.fill_between(grid, low, high, color="m", alpha=0.25, zorder=4)
        planck = Planck18.distmod(grid).value
        ax.plot(grid, planck, color="#F0B000", lw=1.2, ls="--", zorder=5,
                label=r"flat $\Lambda$CDM (Planck 2018)")
        pantheon_reference = FlatLambdaCDM(H0=73.6, Om0=0.334).distmod(grid).value
        for axis in (ax, inset):
            axis.plot(grid, pantheon_reference, color="#ff7f0e", lw=1.5,
                      ls=(0, (6, 2, 1, 2)), zorder=6,
                      label=r"flat $\Lambda$CDM (Pantheon+SH0ES)")

        res.axhline(0, color="m", lw=3, zorder=1)
        res.plot(grid, planck - mid, color="#F0B000", ls="--", lw=2)
        res.plot(grid, pantheon_reference - mid, color="#ff7f0e", lw=1.5,
                 ls=(0, (6, 2, 1, 2)))
        ax.set_ylim(26, 51)
        ax.set_xlim(-0.2, z.max() + 0.3)
        inset.set_xscale("log")
        inset.set_xlim(0.02, 5.2)
        inset.set_ylim(32, 51)
        inset.set_xlabel(r"$z$", fontsize=12, labelpad=-10)
        inset.set_ylabel(r"$\mu$ (mag)", fontsize=12)
        inset.tick_params(which="major", labelsize=10)
        ax.set_ylabel(r"$\mu$ (mag)")
        ax.yaxis.set_label_coords(-0.075, 0.5)
        ax.legend(loc="lower left", bbox_to_anchor=(0.08, 0.08), frameon=False, fontsize=10)
        res.set_ylabel(r"$\Delta\mu$ (mag)")
        res.set_xlabel(r"$z$")
        res.set_ylim(-0.7, 0.7)
        res.set_yticks([-0.5, 0, 0.5])
        res.text(0.02, 0.10, f"RMS: {np.sqrt(np.mean(residual ** 2)):.3f} mag",
                 transform=res.transAxes, fontsize=9,
                 bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"), zorder=20)
        for axis in (ax, inset, res):
            axis.minorticks_on()
            axis.tick_params(which="minor", direction="in", length=4, top=True, right=True, width=2)
            axis.tick_params(which="major", direction="in", length=8, top=True, right=True)
        ax.tick_params(axis="x", which="both", labelbottom=False)
        stem = args.output_dir / f"hubble_{model}_sna_noceph"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        table = sn[["CID", "zHD", "zHEL", "IS_CALIBRATOR"]].copy()
        table["mu_noceph_hd"] = mu
        table["sigma_stat_sys_diag"] = errors
        table["residual_posterior_median"] = residual
        table.to_csv(stem.with_suffix(".csv"), index=False)
        manifest["runs"].append({"checkpoint": str(checkpoint.resolve()),
                                 "checkpoint_sha256": sha256(checkpoint), "run_tag": tag,
                                 "prior_bounds": priors, "posterior_draws": len(samples),
                                 "plotted_draw_indices": indices.tolist()})
        print(f"Saved {stem}.pdf", flush=True)
    (args.output_dir / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
