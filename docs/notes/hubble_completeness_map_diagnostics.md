# Hubble sample-selection controls

The hard-cut science magnitude support defaults to inclusive [17, 27],
within padded map edges [16.5, 27.5]. Narrower science bounds can be set with
QVC_CUT_COMPLETENESS_MAG_2500_MIN/MAX. The map retains fixed redshift coverage
[0, 4.5], separate from the fitting redshift range.

Tier 2 adds configurable spectral S/N, information, and PSF host-fraction cuts,
and makes the existing point-count cut configurable. Defaults are 400 points,
S/N >=3, eta_sigma_kl >=0.05, and PSF host fraction in [0, 0.90]. Set
QVC_CUT_LIGHT_CURVE_N_POINTS_MIN, QVC_CUT_SN_MEDIAN_ALL_MIN,
QVC_CUT_ETA_SIGMA_KL_MIN, and QVC_CUT_F_HOST_2500_PSF_MAX respectively.
Use none to disable a cut including required-column and finiteness checks.
Point counts retain the existing sum over available bands excluding u.
Enabled thresholds are inclusive and are recorded for resume validation.

## Completeness grid and smoothing

The library defaults to 0.1-mag and 0.1-redshift target bin widths. Use
--completeness-mag-bin-width and --completeness-z-bin-width or the corresponding
QVC_HUBBLE_COMPLETENESS_MAG_BIN_WIDTH and QVC_HUBBLE_COMPLETENESS_Z_BIN_WIDTH
environment variables. The grid preserves padded physical coverage and uses
integer bin counts. Direct CLI overrides are resolved before scientific imports.

--completeness-smooth-sigma-mag and --completeness-smooth-sigma-z set Gaussian
count smoothing widths (defaults 0.1 mag and 0.3 in z), with corresponding
QVC_HUBBLE_COMPLETENESS_SMOOTH_SIGMA_MAG/Z environment variables. Counts are
smoothed separately before division. These are not physical magnitude errors
in the likelihood. Checkpoints record the resolved support and smoothing.

Counts-comparison PDF/CSV diagnostics expose observed and mock histograms
and the map values by magnitude and redshift. The mock parent and original
magnitude-dependent completeness remain active when a bright cut is applied.
