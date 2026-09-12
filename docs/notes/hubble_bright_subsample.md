# Bright-subsample selection

`--bright-subsample-completeness-min 0.1 --bright-subsample-margin 0.25`
selects AGN at least 0.25 mag brighter than the faintest grid center reaching
10% of the completeness peak at each redshift. Thresholds are interpolated
in redshift. The options require the NumPy/Dynesty `--run single` pipeline,
AGN, and 2D completeness. Omit the minimum to disable the cut.
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
