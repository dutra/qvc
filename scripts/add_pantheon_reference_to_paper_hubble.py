"""Add the Brout+2022 Table 3 flat-LCDM reference to the saved fiducial PDF.

Retains the original PDF content, axes, points, curves, and annotations.
Verifies the posterior median against the original vector model path before
using it as the residual reference. No inference or data recalibration.
"""
import hashlib
import argparse
import io
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "plots/hubble/fiducial/Flatw0waCDM_joint"
SOURCE = DIRECTORY / "hubble_diagram_debiased.pdf"
CHECKPOINT = ROOT / "results/hubble_posteriors/fiducial/Flatw0waCDM_joint.h5"
OUTPUT = DIRECTORY / "hubble_diagram_debiased_pantheon_sh0es.pdf"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    global DIRECTORY, SOURCE, CHECKPOINT, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agn-run", help="AGN-only run name; otherwise use the fiducial paper fit")
    options = parser.parse_args()
    if options.agn_run:
        DIRECTORY = ROOT / "plots/hubble" / options.agn_run / "Flatw0waCDM_agn"
        SOURCE = DIRECTORY / "hubble_diagram_debiased.pdf"
        CHECKPOINT = ROOT / "results/hubble_posteriors" / options.agn_run / "Flatw0waCDM_agn.h5"
        OUTPUT = DIRECTORY / "hubble_diagram_debiased_pantheon_sh0es.pdf"
    reader = PdfReader(SOURCE)
    page = reader.pages[0]
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    operations = ContentStream(page.get_contents(), reader).operations
    rectangles, magenta_paths = set(), []
    color, rect, points = None, None, []
    for args, operator in operations:
        if operator == b"re":
            rect = tuple(map(float, args))
            rectangles.add(rect)
        if operator == b"RG":
            color = tuple(map(float, args))
        if operator in (b"m", b"l"):
            points.append(tuple(map(float, args)))
        if operator in (b"S", b"f", b"f*", b"B", b"n"):
            if operator == b"S" and color == (0.75, 0., 0.75) and len(points) > 5:
                magenta_paths.append((rect, np.array(points)))
            points = []
    assert len(rectangles) == 3, "Unexpected source layout"
    main_rect = max(rectangles, key=lambda r: r[2] * r[3])
    residual_rect = min(rectangles, key=lambda r: r[1])
    inset_rect = next(r for r in rectangles if r not in (main_rect, residual_rect))
    data = pd.read_csv(DIRECTORY / "hubble_plot_residuals.csv")
    xlim = (-0.2, float(data.z.max()) + 0.3)
    with h5py.File(CHECKPOINT) as f:
        samples = f["flat_samples"][()]
        labels = [x.decode() for x in f["model_labels"][()]]
    indices = np.linspace(0, len(samples) - 1, min(len(samples), 500), dtype=int)
    grid = np.linspace(1e-4, 5.2, 500)
    curves = []
    for row in samples[indices]:
        params = dict(zip(labels, row))
        cosmo = Flatw0waCDM(**{key: params[key] for key in ("H0", "Om0", "w0", "wa")})
        curves.append(cosmo.distmod(grid).value)
    median = np.median(curves, axis=0)
    original_path = next(points for rect, points in magenta_paths if rect == main_rect)
    z_path = xlim[0] + (original_path[:, 0] - main_rect[0]) / main_rect[2] * np.ptp(xlim)
    mu_path = 26 + (original_path[:, 1] - main_rect[1]) / main_rect[3] * 25
    valid = (z_path >= grid[0] + 1e-6) & (z_path <= xlim[1])
    max_difference = float(np.max(np.abs(mu_path[valid] - np.interp(z_path[valid], grid, median))))
    assert max_difference < 1e-4, f"Saved posterior does not reproduce PDF curve: {max_difference} mag"

    reference = FlatLambdaCDM(H0=73.6, Om0=0.334)
    reference_mu = reference.distmod(grid).value
    # A transparent full-page Matplotlib overlay preserves source geometry.
    fig = plt.figure(figsize=(width / 72, height / 72))
    fig.patch.set_alpha(0)
    color = "#ff7f0e"
    for rect, limits, logarithmic, y in (
        (main_rect, (26, 51), False, reference_mu),
        (inset_rect, (32, 51), True, reference_mu),
        (residual_rect, (-0.7, 0.7), False, reference_mu - median),
    ):
        x, bottom, w, h = rect
        ax = fig.add_axes([x / width, bottom / height, w / width, h / height])
        ax.set_axis_off()
        ax.patch.set_alpha(0)
        ax.set_xlim((0.02, 5.2) if logarithmic else xlim)
        ax.set_ylim(limits)
        if logarithmic:
            ax.set_xscale("log")
        ax.plot(grid, y, color=color, lw=1.5, ls=(0, (6, 2, 1, 2)), clip_on=True)
    # Insert the reference after Planck and move the three data entries down.
    legend_ax = fig.add_axes([0, 0, 1, 1])
    legend_ax.set_axis_off()
    legend_ax.set_xlim(0, width)
    legend_ax.set_ylim(0, height)
    legend_ax.add_patch(plt.Rectangle((102, 157), 150, 46 if options.agn_run else 61,
                                      facecolor="white", edgecolor="none"))
    reference_y = 195.22 if options.agn_run else 210.0
    legend_ax.plot([107.393, 127.393], [reference_y, reference_y], color=color,
                   lw=1.5, ls=(0, (6, 2, 1, 2)))
    legend_ax.text(135.393, reference_y, r"flat $\Lambda$CDM (Pantheon+SH0ES)",
                   fontsize=10, va="center")
    data_entries = (
        (195.22, "AGN", "o", "#888888", 3, 0.6),
        (180.44, "AGN (z-binned)", "o", "red", 5, 0.98),
        (165.66, "SN Ia", "s", "#0A84FF", 2, 0.7),
    )
    if options.agn_run:
        data_entries = tuple((y - 14.78, *entry) for y, *entry in data_entries[:2])
    for y, label, marker, marker_color, size, alpha in data_entries:
        legend_ax.errorbar(117.393, y, yerr=5, fmt=marker, color=marker_color,
                          markersize=size, alpha=alpha, capsize=2,
                          elinewidth=1.2 if label == "AGN (z-binned)" else 0.8)
        legend_ax.text(135.393, y, label, fontsize=10, va="center")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", transparent=True)
    plt.close(fig)
    buffer.seek(0)
    page.merge_page(PdfReader(buffer).pages[0])
    writer = PdfWriter()
    writer.add_page(page)
    writer.add_metadata({"/Title": "Fiducial w0wa Hubble diagram with Pantheon+SH0ES flat LCDM",
                         "/Subject": "Brout et al. 2022 Table 3: H0=73.6 km/s/Mpc, Om0=0.334"})
    with OUTPUT.open("wb") as handle:
        writer.write(handle)
    provenance = {
        "source_pdf": str(SOURCE), "source_sha256": digest(SOURCE),
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": digest(CHECKPOINT),
        "source": "https://doi.org/10.3847/1538-4357/ac8e04", "table": 3,
        "reference": {"model": "FlatLambdaCDM", "H0": 73.6, "Om0": 0.334},
        "residual": "mu_Pantheon_SH0ES - original_w0wa_posterior_median",
        "posterior_pdf_agreement_max_mag": max_difference,
        "posterior_draw_indices": indices.tolist(),
        "original_content": "Preserved via PDF vector overlay except reordered legend; no inference rerun",
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(OUTPUT)
    print(f"Original model curve reproduced within {max_difference:.3g} mag")


if __name__ == "__main__":
    main()
