"""Compare saved debiased AGN-only residuals on their common fit selection."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    "nov21_agnonly_lcdm_quick_brightcut_mean_om01_20261121_154625",
    "nov21_continuum_agnonly_lcdm_om01_standard",
]


def main():
    paths = [ROOT / "plots/hubble" / run / "FlatLambdaCDM_agn/hubble_plot_residuals.csv"
             for run in RUNS]
    frames = []
    for path in paths:
        d = pd.read_csv(path, dtype={"object_id": str})
        good = (d.is_fit_selection & d.in_fit_z_range
                & np.isfinite(d.residuals) & np.isfinite(d.clipping_sigma)
                & (d.clipping_sigma > 0))
        d = d.loc[good].set_index("object_id", verify_integrity=True)
        frames.append(d)
    common = frames[0].index.intersection(frames[1].index)
    frames = [d.loc[common] for d in frames]
    assert np.allclose(frames[0].z, frames[1].z)
    plt.style.use(ROOT / "src/qvc/hubble/style.mplstyle")
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    edges = np.linspace(0.44, 3.16, 17)
    labels = ["Previous LC fit (with redshift evolution)",
              "New LC fit (without redshift evolution)"]
    rows = []
    for d, label, color, marker, line in zip(
        frames, labels, ["#0072B2", "#D55E00"], ["o", "s"], ["--", "-"]
    ):
        points = []
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            b = d[(d.z >= lo) & ((d.z < hi) if i < 15 else (d.z <= hi))]
            if len(b) < 3:
                continue
            w = 1 / b.clipping_sigma.to_numpy()**2
            mean = np.average(b.residuals, weights=w)
            sem = np.sqrt(1 / w.sum())
            center = (lo + hi) / 2
            points.append((center, mean, sem))
            rows.append(dict(sample=label, z_low=lo, z_high=hi, n=len(b),
                             z_center=center, mean_residual=mean, sem=sem))
        p = np.asarray(points)
        ax.errorbar(p[:, 0], p[:, 1], yerr=p[:, 2], color=color,
                    marker=marker, linestyle=line, capsize=3, label=label)
    ax.axhline(0, color="0.35", lw=1, zorder=0)
    ax.set(xlabel="Redshift $z$", ylabel=r"$\Delta\mu$ (mag)", xlim=(0.4, 3.2))
    ax.set_title(f"Debiased AGN-only flat LCDM: {len(common):,} matched objects", fontsize=15)
    ax.legend(frameon=False, fontsize=12)
    fig.supxlabel("Residuals relative to each fit's own cosmology; inverse-variance weighted means",
                  fontsize=10)
    output = ROOT / "plots/diagnostics/hubble_residual_previous_vs_new_lc"
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(output.with_suffix("." + ext))
    pd.DataFrame(rows).to_csv(output.with_suffix(".csv"), index=False)
    output.with_suffix(".json").write_text(json.dumps(dict(
        inputs=[str(p) for p in paths], matched_objects=len(common),
        object_ids=list(common), bin_edges=edges.tolist(),
        uncertainty="sqrt(1 / sum(1 / clipping_sigma**2)); no cross-object covariance",
        caveat="Previous fit includes gamma_mean_z; new fit does not. Both use their own fitted cosmology."
    ), indent=2) + "\n")
    print(f"Matched {len(common)} objects; saved {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
