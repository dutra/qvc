# Original SLB light-curve improvements

`--eta-prior-profile modified_tight` is opt-in. It uses underlying Normal location/scale (-0.8, 0.25) truncated to [-1.5, 0] for eta_sigma, and (0.5, 0.25) truncated to [0, 1.5] for eta_tau in models with wavelength-dependent drivers. Original SLB has no eta_tau. Existing defaults are unchanged.

Original SLB fits export `log_tau_slow_rf_ess` and `log_tau_slow_rf_rhat` in result catalogs and, when sample saving is enabled, as sample HDF5 attributes. These use the actual chain-by-draw log10 rest-frame slow-driver timescale and NumPyro ESS/split R-hat. One-chain split R-hat does not measure agreement between independent chains. Reloading flat samples reuses saved attributes; legacy files without them retain NaN diagnostics. These fields do not enable Hubble regressor selection or change covariance handling or posterior payload versions.

`load_obj_samples_from_hdf5(..., return_metadata=True)` returns `(samples, attributes)`; the default still returns the sample dictionary. Existing scalar diagnostic datasets remain datasets; the new slow-driver diagnostics are attributes, not posterior draws.
