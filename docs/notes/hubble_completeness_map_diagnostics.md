# Hubble completeness controls and outputs

## Launcher configuration

`run_hubble.xonsh` selects the September 9 10:33 SLB catalog and the August 31
spectral catalog. It defaults to `standard`; set `QVC_HUBBLE_SPEED=quick` for
the saved September 12 quick configuration. It fits Flatw0waCDM and
FlatLambdaCDM jointly with SNe, with the default cosmological prior,
unrounded observable pivots, and no sigma-clipping pass. It preserves the
baseline continuum timescale and catalog covariance interface from preview.
No light-curve or spectra fitting changes are required.

The launcher selects attenuated completeness magnitudes from 17 through 26
(inclusive). The map retains padded support [16.5, 27.5] and z=[0, 4.5].
The fit interval z=[0.44, 3.16] is applied separately from map construction.
The historical saved run name includes `magcut1724` and `chisq30`, but its
checkpoint specifies 17–26 and no g-band variability threshold.

Launcher Tier 1 limits are SED chi-square <=1.5, spectroscopy <=1.1, joint
<=1.2, LOO <=1.0, zero divergences, spectral R-hat <=1.1, and light-curve
R-hat <=1.05. Tier 2 requires spectral S/N >=3, log timescale >=1.5,
magnitude uncertainty <=0.5 mag, and PSF host fraction in [0, 1]. Point-count,
information, and extinction cuts are disabled in this launcher.

`QVC_CUT_LIGHT_CURVE_N_POINTS_MIN`, `QVC_CUT_SN_MEDIAN_ALL_MIN`,
`QVC_CUT_ETA_SIGMA_KL_MIN`, and `QVC_CUT_F_HOST_2500_PSF_MAX` configure the
added inclusive Tier 2 cuts. `none` disables each entire cut, including
column/finiteness requirements. Point counts retain the existing sum over
available bands excluding u. The library defaults remain 400 points, S/N 3,
information 0.05, and PSF host fraction <=0.90; launcher overrides are applied
before importing scientific modules. Magnitude bounds use
`QVC_CUT_COMPLETENESS_MAG_2500_MIN/MAX` (library defaults 17/27).

## Grid and smoothing

The launcher defaults to requested bin widths of 0.2 mag and 0.2 in redshift,
with Gaussian count smoothing sigma_mag=0.1 mag and sigma_z=0.3. Over the
fixed padded coverage this yields 55 x 23 bins, with actual redshift width
4.5/23. Observed and mock counts are smoothed separately before division.
The original Wang Type-1 LADE-A mock parent and varying spectral-index
conversion are retained.

Override the launcher with `--completeness-mag-bin-width`,
`--completeness-z-bin-width`, `--completeness-smooth-sigma-mag`, and
`--completeness-smooth-sigma-z`, or corresponding environment variables
`QVC_HUBBLE_COMPLETENESS_MAG_BIN_WIDTH`, `QVC_HUBBLE_COMPLETENESS_Z_BIN_WIDTH`,
`QVC_HUBBLE_COMPLETENESS_SMOOTH_SIGMA_MAG`, and
`QVC_HUBBLE_COMPLETENESS_SMOOTH_SIGMA_Z`. Explicit CLI values win. Direct
scientific CLI grid settings are resolved before importing grid constants.
Smoothing widths are not physical magnitude errors in the likelihood.

## Outputs

Completeness plotting is opt-in with `--plot-completeness`. The launcher
enables it and defaults to minimal plots; `QVC_HUBBLE_MINIMAL_PLOTS=false`
restores the full plot suite. Minimal mode retains the debiased Hubble diagram,
cosmological corner, and fresh-run Dynesty posterior products.

The completeness suite includes raw counts, count comparisons, relative maps
with logarithmic colors, and bright-cut diagnostics. Counts-comparison CSVs
show observed/mock ratios by magnitude and redshift; they do not establish
that departures from a plateau are necessarily selection-model errors.
Stage timing messages identify computation and export costs.

Plots use `plots/hubble/<prefix>/<cosmology>_<data-combination>/`. Hubble
bins use 16 equal-width intervals over the fitted redshift range and display
bins with at least three objects. Annotations distinguish fitted and displayed
counts and the existing scatter/chi-square definitions; the residual panel
uses limits of +/-0.7 mag. The fit-quality summary is also printed as a table.
Posterior completeness replay uses at most 256 evenly spaced posterior rows,
plus any rows required for matched plotting draws. Sampling is unchanged.

For reproduction, compare exact fitted IDs, cut grids, priors, and likelihood
values at fixed posterior points. The saved September 12 quick run has 4,509
fitted AGN. Its sampler was unseeded, so fresh posterior samples need not be
identical. Preserve the original catalogs and mock for comparisons, and use
a new output prefix to avoid overwriting saved products.

## Port validation

Using the existing September 12 mock and catalogs, the port reproduced all
5,606 quality-selected objects and the exact ordered 4,509 fitted IDs in both
saved cosmology checkpoints. Bright-cut thresholds, map peaks, and prior
bounds matched. At three saved posterior points per cosmology, the AGN
log-likelihood and every per-object normalization/debias moment matched the
source branch exactly. This checks the AGN likelihood without rerunning the
unseeded sampler. The SNe likelihood is unchanged from preview.
