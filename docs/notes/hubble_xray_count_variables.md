# X-ray counts exported to LaTeX

Full model-comparison output includes `NumAGNXrayMatched` and
`NumAGNAlphaOXPlotted` alongside the existing AGN counts. With the fiducial
prefix, these are `\resultFiducialNumAGNXrayMatched` and
`\resultFiducialNumAGNAlphaOXPlotted`.

The counts refer to the selected diagnostic sample, including sources outside
the fit redshift range, not the full pre-cut catalog. The match count uses an
explicit `xray_matched` flag from the positional match, including counterparts
without usable broad-band flux. The plotted count comes directly from the
alphaOX plotting mask (finite alphaOX, residual, residual uncertainty and z;
missing x errors are allowed). It counts points passed to the plot, including
any beyond the displayed axis limits. Skipped alphaOX plots report zero.

The unqualified variables refer to Flatw0waCDM when present, otherwise the first
requested model. A comment in the generated TeX file identifies that model.
Model-specific variables are also exported, for example
`\resultFiducialFlatwZerowaCDMNumAGNAlphaOXPlotted`, so model-dependent clipping
or missing residuals cannot silently mix counts.

Values are generated afresh with the configured X-ray catalog; existing saved
TeX files and PDFs are not rewritten by this code change. This does not activate
the newly rebuilt Chandra catalog or alter matching/selection thresholds.
