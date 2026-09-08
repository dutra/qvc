# Generate only the light-curve fit figure with PSD

Append `--only-light-curve-plot` (alias `--only_light_curve_plot`) to a light-curve fitting command or to `python hpc_scripts/sfitlc.py ...`.

The flag implies `--plot`. It generates the combined light-curve and PSD PDF in `plots/multiband/<prefix>/light_curves_fits/`, suppressing other figures and their plot-only calculations. Catalog diagnostics and fitting still run. Existing PSD and component overlay options remain available.

It cannot be combined with `--disable_combined_plot` or `--disable_plot_psd`. With the HPC launcher, catalog merging still runs, but automatic Stone, MacLeod, Suberlak, and same-length comparison figures are suppressed. Existing `--resume` semantics are unchanged. Without this flag, existing plotting behavior is unchanged.

Object-selection inputs are unchanged: this feature does not add HDF5 support to `--chisq-csv`.
