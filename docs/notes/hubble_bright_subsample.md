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

All completeness maps, audits, posterior correction plots, and bright-cut
artifacts live under `plots/hubble/<prefix>/<cosmology_mode>/completeness/`.
No completeness folder is generated directly under the prefix. Each AGN
cosmology gets its own complete set, using the same original bright selection
and completeness parent; SNe-only fits produce none. The shared mock catalog
used to derive the bright cut is generated under the first AGN model and
reused by subsequent models. Existing output directories are not migrated.
Python callers can supply `bright_subsample_plot_data=BrightSubsamplePlotData(...)`
to `run_single` or `run_all` to export the original pre-bright sample counts
and selection plot. Omitting it retains threshold-only export for callers
that do not have the original selection inputs.

Relative to each model directory, the threshold grid and retained counts are written to
`completeness/bright_subsample_thresholds.csv` and
`completeness/bright_subsample_counts.csv`. With `--plot-completeness`, the
threshold diagnostic is written to `completeness/bright_subsample_cut.pdf`.
The redshift-binned cut report remains at
`plots/hubble/<prefix>/diagnostics/cut_diagnostics_by_z.csv`. Checkpoint
metadata records the full cut and the detailed `run_tag` (including
`_brightsub-rel0p1-dm0p25`); resume rejects mismatched cut definitions. Plot directories use the
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

Posterior files use the same cosmology/mode name as the plot folder, under
`hubble_posteriors/<prefix>/` in the configured results directory:
`Flatw0waCDM_joint.h5`, `Flatw0waCDM_agn.h5`, or `Flatw0waCDM_sna.h5`.
Two-pass fits retain `_pass1.h5` and `_pass2.h5` suffixes; JAX files use
`<cosmology>_<mode>_jax.h5`. Automatic resume searches these shorter names.
Existing files are not renamed; use an explicit resume path for legacy names.
Use separate prefixes to retain different configurations. Detailed settings
remain in checkpoint metadata, including `run_tag`, rather than filenames.

Use `--skip-debiased-residual-plot` (alias `--skip_debiased_residual_plot`) to
skip only the `Plotting debiased residuals...` partial-control diagnostic in
NumPy/Dynesty `single` and `full` runs. This skips
`full_residuals_debiased_partial_controls.pdf` and its auxiliary
`partial_control_residuals.csv` and `partial_control_parameter_index.csv`.
Other residual plots, Hubble residual CSVs, clipping, and inference remain
unchanged. The default is off; minimal mode already skips this diagnostic.
Existing output files are not deleted.
