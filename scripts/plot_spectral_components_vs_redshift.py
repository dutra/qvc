#!/usr/bin/env python
"""Plot every stored spectral component fraction against redshift.

The script reads the flat ``catalog/`` group of a spectral-fit catalog and
draws one panel per component fraction with
:func:`qvc.hubble.hubble_plotting.plot_spectral_fractions_cut_sample`.
It writes a pre-cut figure (every catalog row) and, when a selection is given,
a post-cut figure restricted to it: either ``--posterior`` (the posterior's
``object_id_fit_selection``) or ``--object-ids`` (a CSV with an ``object_id``
column). Both fraction figures share the pre-cut redshift range.

It also writes ``<amplitude-stem>_precut.pdf`` and ``_postcut.pdf``: one
luminosity per continuum component of the joint SED + spectrum fit, all as
log10(L / W) on one shared linear y range so component strengths can be
compared by eye. The post-cut amplitude figure uses the post-cut redshift span.
``--figures`` selects which set to write.

Fractions (definitions from
``qvc.spectra.fit_spectra_jaxsedfit_joint.summarize_spectral_component_fractions``
and ``add_sdss_psf_host_fraction_prediction``):

- ``f_bc_3000``, ``f_fe_uv_3000``: Balmer continuum and UV Fe II over the
  spectral continuum at the fitted pixel nearest rest 3000 A. The continuum is
  the fibre-captured host plus accretion disc plus torus (the joint SED model
  rendered on the spectrum), plus Fe II and Balmer continuum. Lines are
  excluded. The pixel is the nearest one even when rest 3000 A is outside the
  spectrum, so above z ~ 2.1 (SDSS) or z ~ 2.5 (BOSS) these are ratios at the
  red edge of the spectrum, at a shorter rest wavelength. The Balmer
  continuum is only fitted when the spectrum covers rest 3000-4000 A, so
  ``f_bc_3000`` is exactly zero (drawn as down-arrows) above z ~ 1.3 (SDSS)
  or z ~ 1.6 (BOSS), and for most objects at z < 0.2.
- ``1 - f_bc_3000 - f_fe_uv_3000`` (derived here): the disc + fibre-captured
  host + torus share of that same continuum, at the same pixel. Where the
  Balmer continuum is not fitted this is just 1 - f_fe_uv_3000.
- ``f_br``, ``f_na``: broad and narrow line flux over the same continuum, both
  integrated in f_lambda over the fitted wavelength range.
- ``f_host_2500_psf``: PSF-captured host over captured host plus AGN at rest
  2500 A, from the SED components.

Component luminosities (derived here from the catalog's posterior medians;
JAXSEDFit model at the fit's commit, jaxsedfit_dutra de8866c, file:line refer
to that commit). L means lambda L_lambda = nu L_nu at the stated rest
wavelength unless marked integrated. Flux -> luminosity uses the fit's own
convention, lambda L_lambda(rest) = 4 pi d_L^2 nu_obs f_nu,obs (model.py:1435
``_redshift_to_obs`` and model.py:2744), with d_L from astropy
FlatLambdaCDM(H0=70, Om0=0.3), the fit's ``galaxy.cosmology_h0/om0`` defaults
(config.py:204-205, preload.py:2079-2081). Because every value is a function
of per-parameter medians, it approximates (not equals) the posterior median of
the derived luminosity.

- AGN disc: ``log_agn_amp / ln 10``. ``log_agn_amp`` is the sampled natural
  log of the intrinsic (unattenuated) disc lambda L_lambda at rest 5100 A in W
  (model.py:4094: L_lambda,5100 = exp(log_agn_amp) / 5100 A).
- Galaxy: intrinsic host stellar lambda L_lambda at rest 5100 A, inverted from
  ``fracAGN_5100_fit`` = L_disc / (L_disc + L_host) at 5100 A (model.py:3875,
  3888-3892), where L_host is the interpolated unattenuated SSP mixture of the
  whole galaxy (no aperture capture, no nebular or dust emission;
  model.py:2144/2383) and L_disc the intrinsic disc only (no torus). Hence
  L_host = L_disc (1 - f) / f. The fit clips f to [1e-4, 0.999]; rows at a
  clip bound (host unconstrained relative to the disc) and NaN rows are NaN.
- Fe II: ``spectral_feii_norm / spectral_feature_amplitude_scale`` is the
  model's f_nu (mJy, SED/photometry flux calibration) of the Fe II template at
  rest 4575 A: the sampled amplitude lives in the observed spectrum's scale
  and is divided by the spectrum scale before being added to the SED-calibrated
  continuum (spectral_components.py:699-703; the total is multiplied by the
  spectrum scale only for the spectrum likelihood, model.py:4997-5000). The
  BruhweilerVerner08 template is normalised to 1 in f_lambda at its pixel
  nearest rest 4575 A before broadening (preload.py:652-660) and converted to
  f_nu with pivot 4575 A (spectral_components.py:493-516). That single
  normalisation pixel is a narrow spike, so the bare amplitude overstates the
  Fe II level by about 0.5 dex. The panel therefore multiplies by
  ``FEII_WINDOW_MEAN`` = 0.276, the template's mean f_lambda over rest
  4434-4684 A relative to that pixel, and shows the mean lambda L_lambda of
  the optical Fe II blend in that window. It is not corrected for the
  internal (host/nuclear) attenuation of the continuum it is added to. Above
  z ~ 1 rest 4575 A is outside the SDSS spectrum and the amplitude is fixed
  by the UV Fe II.
- Balmer continuum: (``spectral_balmer_norm`` / scale) * S(3000 A) is the
  BC f_nu at its pivot, rest 3000 A, with S = (1 - exp(-tau (3000/3646)^3))
  B_lambda(3000 A, 15000 K) / B_lambda(3646 A, 15000 K) the unbroadened
  edge-normalised shape (spectral_model.py:1568-1659, spectral_components.py
  729-753; tau = ``spectral_balmer_tau``). Velocity broadening is neglected at
  3000 A, 650 A from the edge. Same calibration and attenuation caveats as
  Fe II. NaN where the BC is not fitted (spectrum lacks rest 3000-4000 A).
- Galaxy dust: ``log_dust_luminosity_fit``, already log10 of the integrated
  host dust luminosity in W (energy balance: the host light absorbed by the
  galaxy attenuation plus nebular terms).
- AGN torus: log10(2.5 * fcov * lambda L_5100,disc / W), the torus
  lambda L_lambda at its 12 micron normalisation wavelength before the
  silicate feature is applied (model.py:1108-1109, 70, 1117; called with
  l_agn = exp(log_agn_amp), model.py:4010).

Example::

    python scripts/plot_spectral_components_vs_redshift.py \\
        --catalog results/data/jaxqsofit/sep25_0443pm_spectrafit_24f6f85_aug31_w500s500_d01c034_rerun.h5 \\
        --posterior results/hubble_posteriors/fiducial/FlatLambdaCDM_joint.h5 \\
        --plot-path plots/hubble/fiducial

    # Quality cuts without any redshift cut (5,403 objects), amplitudes only:
    python scripts/plot_spectral_components_vs_redshift.py \\
        --catalog results/data/jaxqsofit/sep25_0443pm_spectrafit_24f6f85_aug31_w500s500_d01c034_rerun.h5 \\
        --object-ids plots/hubble/fiducial/diagnostics/postcut_no_z_cuts_object_ids.csv \\
        --plot-path plots/hubble/fiducial --figures amplitudes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use(Path(__file__).resolve().parents[1] / "src/qvc/hubble/style.mplstyle")

from qvc.hubble.hubble_plotting import plot_spectral_fractions_cut_sample  # noqa: E402

REMAINING_CONTINUUM = "f_disc_host_3000"
# (column, label) in plotting order. Columns absent from the catalog are skipped.
COMPONENT_PANELS = (
    ("f_bc_3000", r"$f_{\rm BC,3000\,\AA}$"),
    ("f_fe_uv_3000", r"$f_{\rm FeII,3000\,\AA}$"),
    (REMAINING_CONTINUUM, "$1-f_{\\rm BC}-f_{\\rm FeII}$\n(disc+host+torus)"),
    ("f_br", r"$f_{\rm broad\ lines}$"),
    ("f_na", r"$f_{\rm narrow\ lines}$"),
    ("f_host_2500_psf", r"$f_{\rm host,2500\,\AA}^{\rm PSF}$"),
)
Y_FLOOR = 2e-6  # Lower y limit used by plot_spectral_fractions_cut_sample.

from qvc.hubble.spectral_components import (
    AMPLITUDE_PANELS, FRAC_AGN_5100_CLIP, LUMINOSITY_INPUTS,
    add_component_luminosities,
)


def _as_str(values) -> np.ndarray:
    return np.array(
        [v.decode() if isinstance(v, bytes) else str(v) for v in values], dtype=object
    )


def load_catalog(path: Path) -> pd.DataFrame:
    """Read ``object_id``, ``z`` and every component-fraction and amplitude column."""
    wanted = {
        "object_id", "z",
        *(name for name, _ in COMPONENT_PANELS),
        *LUMINOSITY_INPUTS,
    }
    with h5py.File(path, "r") as handle:
        group = handle["catalog"]
        data = {name: group[name][()] for name in wanted if name in group}
    if "object_id" not in data or "z" not in data:
        raise KeyError(f"{path} catalog/ lacks object_id or z")
    frame = pd.DataFrame(data)
    frame["object_id"] = _as_str(frame["object_id"])
    if {"f_bc_3000", "f_fe_uv_3000"}.issubset(frame.columns):
        # Same 3000 A denominator, so the remainder is the disc+host+torus share.
        frame[REMAINING_CONTINUUM] = 1.0 - frame["f_bc_3000"] - frame["f_fe_uv_3000"]
    if set(LUMINOSITY_INPUTS).issubset(frame.columns):
        frame.attrs["luminosity_counts"] = add_component_luminosities(frame)
    return frame


def load_selection(path: Path) -> np.ndarray:
    """Read the post-cut ``object_id_fit_selection`` from a posterior file."""
    with h5py.File(path, "r") as handle:
        if "object_id_fit_selection" not in handle:
            raise KeyError(f"{path} has no object_id_fit_selection dataset")
        return _as_str(handle["object_id_fit_selection"][()])


def load_selection_csv(path: Path) -> np.ndarray:
    """Read post-cut ids from a CSV with an ``object_id`` column."""
    ids = pd.read_csv(path, dtype={"object_id": str})
    if "object_id" not in ids.columns:
        raise KeyError(f"{path} has no object_id column")
    return ids["object_id"].astype(str).to_numpy(dtype=object)


def panel_summary(frame: pd.DataFrame, panels) -> str:
    lines = []
    for column, _ in panels:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        lines.append(
            f"  {column:18s} finite={finite.size:6d} below_floor={np.sum(finite < Y_FLOOR):5d} "
            f"negative={np.sum(finite < 0):5d} above_1={np.sum(finite > 1):5d} "
            f"median={np.median(finite) if finite.size else np.nan:.4g}"
        )
    return "\n".join(lines)


def amplitude_summary(frame: pd.DataFrame, panels) -> str:
    lines = []
    for column, _, _ in panels:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            q = np.percentile(finite, [0, 0.5, 50, 99.5, 100])
            stats = " ".join(f"{name}={value:.4g}" for name, value in
                             zip(("min", "p0.5", "median", "p99.5", "max"), q))
        else:
            stats = "no finite values"
        lines.append(f"  {column:24s} finite={finite.size:6d} nan={values.size - finite.size:6d} {stats}")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--catalog", type=Path, required=True,
                        help="Spectral-fit catalog h5 with a flat catalog/ group.")
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument("--posterior", type=Path, default=None,
                                 help="Hubble posterior h5 holding object_id_fit_selection.")
    selection_group.add_argument("--object-ids", type=Path, default=None,
                                 help="CSV with an object_id column giving the post-cut sample.")
    parser.add_argument("--plot-path", default="plots/hubble",
                        help="Figures go to <plot-path>/diagnostics/.")
    parser.add_argument("--stem", default="spectral_components_vs_redshift",
                        help="Output names are <stem>_precut.pdf and <stem>_postcut.pdf.")
    parser.add_argument("--amplitude-stem", default="spectral_component_amplitudes_vs_redshift",
                        help="Amplitude figure names are <amplitude-stem>_precut.pdf and _postcut.pdf.")
    parser.add_argument("--amplitude-columns", type=int, default=3, choices=range(1, 7),
                        help="Number of columns in the amplitude figures (default: 3).")
    parser.add_argument("--figures", choices=("all", "fractions", "amplitudes"), default="all",
                        help="Which figure set to write.")
    args = parser.parse_args(argv)

    frame = load_catalog(args.catalog)
    panels = [(name, label) for name, label in COMPONENT_PANELS if name in frame.columns]
    if not panels:
        raise SystemExit("No component-fraction columns found in the catalog.")

    z = pd.to_numeric(frame["z"], errors="coerce").to_numpy(dtype=float)
    z = z[np.isfinite(z)]
    pad = 0.02 * (z.max() - z.min()) if z.max() > z.min() else 0.1
    xlim = (max(0.0, z.min() - pad), z.max() + pad)

    samples = [("precut", frame)]
    print(f"catalog rows: {len(frame)}; panels: {[name for name, _ in panels]}")
    if args.posterior is not None or args.object_ids is not None:
        selection = (load_selection(args.posterior) if args.posterior is not None
                     else load_selection_csv(args.object_ids))
        postcut = frame[frame["object_id"].isin(set(selection))]
        missing = len(set(selection)) - postcut["object_id"].nunique()
        print(f"post-cut ids: {len(selection)} ({len(set(selection))} unique); "
              f"matched in catalog: {len(postcut)}; missing: {missing}")
        samples.append(("postcut", postcut))

    amplitude_panels = [panel for panel in AMPLITUDE_PANELS if panel[0] in frame.columns]
    skipped = [panel[0] for panel in AMPLITUDE_PANELS if panel[0] not in frame.columns]
    if skipped:
        print(f"amplitude columns missing from the catalog (skipped): {skipped}")

    for tag, sample in samples:
        if args.figures in ("all", "fractions"):
            print(f"{tag} ({len(sample)} rows):\n{panel_summary(sample, panels)}")
            path = plot_spectral_fractions_cut_sample(
                sample, plot_path=args.plot_path, filename=f"{args.stem}_{tag}.pdf",
                xlim=xlim, panels=panels, max_columns=len(panels),
            )
            print(f"wrote {path}")
        if args.figures in ("all", "amplitudes") and amplitude_panels:
            frac = pd.to_numeric(sample["fracAGN_5100_fit"], errors="coerce").to_numpy(dtype=float)
            print(f"{tag} fracAGN_5100_fit: nan={np.sum(~np.isfinite(frac))} "
                  f"at_min={np.sum(frac <= FRAC_AGN_5100_CLIP[0] * (1 + 1e-6))} "
                  f"at_max={np.sum(frac >= FRAC_AGN_5100_CLIP[1] * (1 - 1e-9))} (host NaN there)")
            print(f"{tag} luminosities ({len(sample)} rows):\n"
                  f"{amplitude_summary(sample, amplitude_panels)}")
            # xlim=None: each sample gets its own redshift span.
            path = plot_spectral_fractions_cut_sample(
                sample, plot_path=args.plot_path,
                filename=f"{args.amplitude_stem}_{tag}.pdf", xlim=None,
                panels=[(name, label) for name, label, _ in amplitude_panels],
                panel_axes=[{**spec, "box_aspect": 1,
                             "legend_fontsize": matplotlib.rcParams["legend.fontsize"]}
                            for _, _, spec in amplitude_panels],
                max_columns=args.amplitude_columns,
                figsize=(6 * min(args.amplitude_columns, len(amplitude_panels)),
                         6 * (-(-len(amplitude_panels) // args.amplitude_columns))),
            )
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
