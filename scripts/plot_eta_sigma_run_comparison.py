"""Compare the Nov 21 run with the exact fiducial fitting sample."""
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = "nov21_1110am_chisq_slb_psffractioncontinuum_onlyspectrafiltered_iters3disk3_svi10000lr0003w1000s500_specaug31w500s500_6ad2c7f_chisq.h5"


def main():
    with h5py.File(ROOT / "results/data" / RUN) as handle:
        new = pd.DataFrame({key: handle[key][:] for key in ("object_id", "z", "eta_sigma")})
    new["object_id"] = new.object_id.map(
        lambda x: x.decode() if isinstance(x, bytes) else str(x)
    )
    fiducial = pd.read_csv(
        ROOT / "plots/hubble/fiducial/Flatw0waCDM_joint/agn_table_all_fields.csv",
        dtype={"object_id": str},
    )
    with h5py.File(ROOT / "results/hubble_posteriors/fiducial/Flatw0waCDM_joint.h5") as handle:
        ids = {x.decode() if isinstance(x, bytes) else str(x)
               for x in handle["object_id_fit_selection"][:]}
    fiducial = fiducial[fiducial.object_id.isin(ids)]
    if len(fiducial) != len(ids) or fiducial.object_id.nunique() != len(ids):
        raise ValueError("Fiducial table must match every checkpoint fitting ID exactly once")
    common_ids = ids.intersection(new.object_id)
    print(f"Fiducial objects absent from new run: {len(ids - common_ids)}")
    new = new[new.object_id.isin(common_ids)]
    fiducial = fiducial[fiducial.object_id.isin(common_ids)]
    if len(new) != len(common_ids) or new.object_id.nunique() != len(common_ids):
        raise ValueError("New run must contain each matched ID exactly once")
    new = new.set_index("object_id").loc[fiducial.object_id].reset_index()
    if not np.allclose(new.z, fiducial.z):
        raise ValueError("Matched sources have inconsistent redshifts")
    print(f"Matched the same {len(common_ids)} fiducial objects in both runs")

    plt.style.use(ROOT / "src/qvc/hubble/style.mplstyle")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True,
                             layout="constrained")
    bins = np.arange(0, 4.01, 0.2)
    rows = []
    for ax, (label, data, color) in zip(axes, [
        ("Nov 21 run: matched objects", new, "#0072B2"),
        ("Fiducial: matched objects", fiducial, "#D55E00"),
    ]):
        finite = np.isfinite(data.z) & np.isfinite(data.eta_sigma)
        z = data.loc[finite, "z"].to_numpy()
        eta = data.loc[finite, "eta_sigma"].to_numpy()
        print(f"{label}: {len(z)}/{len(data)} finite sources")
        ax.scatter(z, eta, s=7, color=color, alpha=0.20, linewidths=0,
                   rasterized=True)
        points = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (z >= lo) & (z < hi)
            if mask.sum() < 20:
                continue
            p16, median, p84 = np.percentile(eta[mask], [16, 50, 84])
            center = np.median(z[mask])
            points.append((center, median))
            rows.append(dict(sample=label, z_low=lo, z_high=hi,
                             n=int(mask.sum()), z_median=center,
                             eta_median=median, eta_p16=p16, eta_p84=p84))
        points = np.asarray(points)
        ax.plot(points[:, 0], points[:, 1], "o-", color="black", ms=4,
                label="Median in redshift bins")
        ax.set_title(f"{label} ($N={len(z):,}$)")
        ax.set_xlabel("Redshift $z$")
        ax.set_box_aspect(1)
        ax.legend(loc="upper right", fontsize=12, frameon=False)
    axes[0].set_ylabel(r"$\eta_\sigma$")
    axes[0].set_xlim(0, 3.8)
    values = np.concatenate([new.eta_sigma.to_numpy(), fiducial.eta_sigma.to_numpy()])
    values = values[np.isfinite(values)]
    axes[0].set_ylim(values.min() - 0.12, values.max() + 0.12)
    out = ROOT / "plots/diagnostics/eta_sigma_nov21_vs_fiducial"
    out.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{extension}"))
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
    plt.close(fig)
    print(out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
