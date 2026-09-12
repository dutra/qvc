# Bright-subsample selection

`--bright-subsample-completeness-min 0.1 --bright-subsample-margin 0.25`
selects AGN at least 0.25 mag brighter than the faintest grid center reaching
10% of the completeness peak at each redshift. Thresholds are interpolated
in redshift. The options support the NumPy/Dynesty `--run single` and `--run full` pipelines
with AGN and 2D completeness. Omit the minimum to disable the cut.
`--bright-subsample-absolute` interprets the minimum as an absolute map value.

The map is constructed from the completeness population before the bright
cut and reused for both cosmologies. The cut filters the fit/plot sample
before its shared observable pivots are calculated. The map population is
forwarded separately to the fitting pipeline so selection does not change
its numerator.

For both likelihood normalization and posterior debiasing, the effective
selection function is `S_eff(m,z) = S(m,z) * [m <= m_cut(z)]`. The original
completeness remains active inside the cut. This does not make the retained
sample fully complete or establish a causal explanation for residual trends.
The existing per-object likelihood normalization is retained.

The threshold grid and per-redshift retained counts are written to
`bright_subsample_thresholds.csv` and `bright_subsample_counts.csv`. With
`--plot-completeness`, the threshold diagnostic is also plotted. Checkpoint
names include `_brightsub-rel0p1-dm0p25`; checkpoint metadata records the full
cut and resume rejects mismatched definitions. Plot directories use the
cosmological model and data combination; use a distinct prefix for each run.


`--run full` forwards the selected LF (including Wang Type-1), attenuation mode,
pre-cut completeness population, and bright-cut definition to each AGN fit.
The shared-pivot resume check verifies the bright-cut definition before reuse.
Its SNe-only comparison disables AGN completeness, completeness plots, and the
bright cut. Direct `--only_sna` runs also ignore these AGN selection options.
The original selection and completeness settings still apply to joint and
AGN-only fits; JAX bright-selected fits remain unsupported.

Comparison plots are written to `plots/hubble/<prefix>/model_compare/`.
Use a distinct prefix when retaining comparison plots from different runs.
Comparison HDF5 filenames retain the speed, redshift/sample settings, LF,
attenuation mode, map variant, and bright-cut tag. Individual cosmology plot
directories keep their existing layout.
