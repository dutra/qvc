
# System requirements
## Hardware Requirements
A modern laptop capable of running a python environment is enough to run all steps in this README.

For the necessary data, 44 GB of disk space will be required.

In optimal performance, 16+ GB of RAM and a modern multi-core CPU is recomended.

## OS Requirements
These routines were tested on
- Linux: Arch Linux with Kernel v6.12.62-1
- Mac OSX: <Colin put your Mac info here>

## Software Requirements
A python enviornment is necessary. This was tested with `python v3.13.5` and `python v<Colin>`.

We highly recommend a python virtual environment such as [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)

The demos should only take a couple hours to run. The full light curve and hubble diagram fitting were run in Yale's HPC clusters, consuming roughly ~100,000 CPU hours.


## Installation guide
The installation step will typically take less than an hour in a modern machine. Downloading the necessary data (next step) may take longer depending on the user's network speed.

### Clone the repository
`git clone https://github.com/dutra/qvc.git`  
`cd qvc`  
`git checkout preview`  

### Installing eztaox
`cd eztaox`  
`pip install .`  

## Set up main conda jax CPU environment (suggested)
From the root directory,  
`conda env create -n jaxcpu python=3.13.5 -f environment.yml`

`conda activate jaxcpu`

If you are managing your environment through other means (e.g.: venv), you can use `pip` to install the requirements:  
`python -m pip install -r requirements.txt`

This environment will be the main environment, used by the light curve multiband fitting and hubble diagram fitting procedures.

### Installing PyQSOFit
In the qvc root directory:  
`git clone https://github.com/dutra/PyQSOFit.git`  

`cd PyQSOFit`  

We highly suggest you create a separate conda virtual environment for PyQSOFit and the fit spectra procedure. When installing `speclite`, it will downgrade numpy and potentially break several packages. Make sure to be inside the `PyQSOFit` directory.  
`conda env create -n pyqsofit python=3.12.9 -f environment.yml`

If you are managing your environment through other means (e.g.: venv), you can use `pip` to install the requirements:  
`python -m pip install -r requirements.txt`

Finally,  
`pip install .`  

## Download the data
Download the demo data from <url>
and extract it inside the `src` folder.

# Demo

## Light Curve Multiband fitting
The Light Curve multiband fitting can be run by specifying an Object ID.  
Included in the demo data is the light curve for the object 1465126, which will reproduce Figure 1.

First, `cd` into the `src` folder. Then run  

```bash
export PREFIX=demo
export SUFFIX=1465126
python multiband_fit.py --plot \
    --progress --nwarm 1000 --nsamp 500 --nchains 4 \
    --max_tree_depth 14 \
    --disable_fhost \
    --bwb \
    --filter_object_id 1465126
```

The results will be written under the folder `src/results/data/demo`, while plots will be under `src/plots/multiband/demo` and samples under `src/results/samples/demo`. 
The expected output is Figure 1 under `src/plots/multiband/demo`.

The `multiband_fit.py` script accepts other options such as `--rf_length_cut <days>`.  
For a list of all options supported, run `python multiband_fit.py --help`.  

## Spectra Fitting
In this step, we will utilize the light curve from the previous step and fit its spectra.

First, make sure you are in the `speclite` environment with  
```bash
conda activate pyqsofit
```

*Do not install speclite in your main environment*. It will downgrade numpy and may break several packages.

Fitting an AGN spectra involves several steps. First, all spectra cache needs be downloaded with  
```bash
python fit_spectra.py --download
```

Then, a run collection several different combinations of spectral fit (e.g.: balmer continuum, iron, host, quasar templates, etc) can be run with
 ```bash
python fit_spectra.py results/data/demo/1465126.h5  results/data/demo/1465126_collect.csv --mode collect --MC_samples 1
```

Next, a run selecting the best fits (chi squared) and penalties to avoid overfitting balmer continuum and host:
```bash
python fit_spectra.py results/data/demo/1465126_collect.csv results/data/demo/1465126_select.csv --mode select
```

Finally, a run to use the selected fits to sample and compute errors:  
```bash
python fit_spectra.py results/data/demo/1465126.h5 results/data/demo/1465126.csv --single_csv results/data/demo/1465126_select.csv --mode single --MC_samples 50
```

The generated file `results/data/demo/1465126.csv` contains all the results from fitting the AGN spectra, including the apparent magnitude at the 2500 AA restframe wavelenght. It can be used by the Hubble Diagram fitting procedure in the argument `--spectra_fit_csv`.


## Hubble diagram fitting
A subset of the Hubble Diagram fitting and plots can be obtained in notebook `hubble_diagram_plots.ipynb`.  The expected output are figures similar to Figures 2, 3 and 7.

In order to reproduce all hubble diagram plots found in the manuscript, we will use the saved full dynesty checkpoints downloaded in the step _Download the data_. The HD fitting procedure can be run with the command

```bash
PREFIX=demo \
python hubble_fit.py --resume \
    --cosmo_models FlatLambdaCDM FlatwCDM Flatw0waCDM \
    --run full \
    --speed production \
    --spectra_fit_csv "results/data/nov12a_11c_single_scratch_nov10a_carma_removemix_fixmeanband_no1pluszflux2L_freeiron_mc50_best.csv" \
    --sdss_mags_csv "results/data/nov2_sdss_mags.csv"  \
    --zquery_csv "results/data/sep19_chisq_zquery.csv" \
    --z_range 0.44 3.16 \
    "results/data/nov10a_single_chisq_carma_mixscalar_nozband_highertaufastlim_removemix_fixband_lagblrband_chisq_spl_nofhost_bwb_lmc-6_N1w1000s200t14ch4.h5"
```

where the `--resume` flag will load the dynesty checkpoint.

The restricted redshift fit (z = 1 - 3.16) can be run with

```bash
PREFIX=demo_zonecut \
python hubble_fit.py --resume \
    --cosmo_models FlatLambdaCDM FlatwCDM Flatw0waCDM \
    --run full \
    --speed production \
    --spectra_fit_csv "results/data/nov12a_11c_single_scratch_nov10a_carma_removemix_fixmeanband_no1pluszflux2L_freeiron_mc50_best.csv" \
    --sdss_mags_csv "results/data/nov2_sdss_mags.csv"  \
    --zquery_csv "results/data/sep19_chisq_zquery.csv" \
    --z_range 1.0 3.16 \
    "results/data/nov10a_single_chisq_carma_mixscalar_nozband_highertaufastlim_removemix_fixband_lagblrband_chisq_spl_nofhost_bwb_lmc-6_N1w1000s200t14ch4.h5"
```

Similarly as `multiband_fit.py`, `hubble_fit.py` will use the environment variable `PREFIX`.  
All hubble diagram relevant plots, tables, and results will be generate under `src/plots/hubble/`.

The expected outputs are Figures 2, 3, 4, 5, 6, 7, 8, A1 and A2, under `src/plots/hubble/` for the cosmology models FlatLambdaCDM FlatwCDM Flatw0waCDM.
It will also generated Table 4 and display a summary of the cosmological model with highest evidence (Flatw0waCDM).

The expected time to run is less than one hour.

## Appendix figures
- Appendix Figure B3 can be reproduced in the notebook `appendix_band_vs_wavelength.ipynb`  
- Appendix Figures C4, C5, D6, D7 can be reproduced in notebook `appendix_sigma_tau.ipynb`  
- Appendix Figure E8 can be reproduced in notebook `appendix_lags.ipynb`  

# Instructions for use
## `multiband_fit.py`

The `multiband_fit.py` script accepts other options such as `--rf_length_cut <days>`.  

For help organizing the batch runs in Yale's HPC, `multiband_fit.py` can use the environment varibles `PREFIX` and `SUFFIX`, which default to `test`.  
The results will be written under the folder `results/data/<prefix>`, while plots will be under `plots/multiband/<prefix>` and samples under `results/samples/<prefix>`.  

For a list of all options supported, run `python multiband_fit.py --help`. 

### Usage

```bash
multiband_fit.py [-h]
  [--filter_object_id FILTER_OBJECT_ID [FILTER_OBJECT_ID ...]]
  [--N N]
  [--skip SKIP]
  [--filter_file FILTER_FILE]
  [--plot]
  [--progress]
  [--nwarm NWARM]
  [--nsamp NSAMP]
  [--nchains NCHAINS]
  [--inject_fake]
  [--bwb]
  [--max_tree_depth MAX_TREE_DEPTH]
  [--load_sample_file]
  [--disable_poly1]
  [--jax_trace]
  [--rf_length_cut RF_LENGTH_CUT]
  [--exact_same_length]
  [--load_stone_lcs]
  [--free_eta_break]
  [--disable_corner_plot]
  [--couple_sigma_tau]
  [--disable_lag_blr]
  [--sigma_tau_uniform]
  [--lmc LMC]
  [--disable_plot_psd]
  [--inject_random_fake_etas]
  [--fhost_csv FHOST_CSV]
  [--disable_fhost]
  [--broken_pl]
  [--log_sigma_eta_tau_sigma LOG_SIGMA_ETA_TAU_SIGMA]
  [--beta_tau BETA_TAU]
  [--disable_band_drop]
  [--load_nearby_lc_csv LOAD_NEARBY_LC_CSV]
  [--load_yu_priors]
  [--disable_sigma_tau_plane_cut]
  [--tau_fast_truncated]
```

### Options

* `-h, --help` — Show this help message and exit.
* `--filter_object_id FILTER_OBJECT_ID [FILTER_OBJECT_ID ...]` — List of object IDs to filter.
* `--N N` — Number of objects to process.
* `--skip SKIP` — Number of objects to skip.
* `--filter_file FILTER_FILE` — Path to file containing object IDs.
* `--plot` — Enable plotting of results.
* `--progress` — Show progress bar.
* `--nwarm NWARM` — Warmup steps for MCMC.
* `--nsamp NSAMP` — Samples per chain for MCMC.
* `--nchains NCHAINS` — Number of chains (>=1).
* `--inject_fake` — Inject fake light curves.
* `--bwb` — Enable BWB model.
* `--max_tree_depth MAX_TREE_DEPTH` — NUTS max tree depth.
* `--load_sample_file` — Load saved samples (debug).
* `--disable_poly1` — Disable trend.
* `--jax_trace` — Enable JAX trace (compile profile).
* `--rf_length_cut RF_LENGTH_CUT` — Rest-frame cut (days).
* `--exact_same_length` — Exact same RF length cut.
* `--load_stone_lcs` — Use Stone LCs.
* `--free_eta_break` — Free `eta_break`, `lam_s`.
* `--disable_corner_plot` — Disable corner plot.
* `--couple_sigma_tau` — Couple sigma/tau prior.
* `--disable_lag_blr` — Disable BLR lag model.
* `--sigma_tau_uniform` — Uniform priors for sigma/tau.
* `--lmc LMC` — LMC Q groups (0/1/2/3).
* `--disable_plot_psd` — Disable PSD sub-plot.
* `--inject_random_fake_etas` — Randomize fake etas.
* `--fhost_csv FHOST_CSV` — CSV with columns: `object_id,f_host_2500`.
* `--disable_fhost` — Set all `f_host_2500=0`.
* `--broken_pl` — Use broken power law.
* `--log_sigma_eta_tau_sigma LOG_SIGMA_ETA_TAU_SIGMA` — Stddev for `log_sigma_eta_tau` priors.
* `--beta_tau BETA_TAU` — `beta_tau` for fake curves.
* `--disable_band_drop` — Disable Lya band drop.
* `--load_nearby_lc_csv LOAD_NEARBY_LC_CSV` — CSV listing nearby LCs to load.
* `--load_yu_priors` — Use Yu+2023 priors.
* `--disable_sigma_tau_plane_cut` — Disable sigma–tau plane cut.
* `--tau_fast_truncated` — Truncated prior for `tau_fast0`.



## Merging the results
Each `multiband_fit.py` run will produce one `.h5` file per object under the folder `results/data/<prefix>`.  
The hubble diagram script loads a single `.h5` with multiple light curve fit results.   
You can merge multiple `.h5` files into one by using the utility
`python merge_results.py <prefix>`.  
That will produce a file `results/data/<prefix>.h5` that can be loaded with the hubble diagram utility.  

For a full list of options, run `python merge_results.py --help`.

### Usage

```bash
merge_results.py [-h] [--base-dir BASE_DIR] [--exclude-jobs [EXCLUDE_JOBS ...]] [--expected EXPECTED] [--skip-populate-sdss] [--in-format {auto,h5,csv}] [--out OUT] [--out-format {h5,csv}] [--dedup-keys [DEDUP_KEYS ...]] prefix
```

**Purpose:** Merge CSV or HDF5 shards (`job*.{h5,csv}`) found in `<base_dir>/<prefix>/` and write a single merged output as either `.h5` or `.csv`.

### Positional arguments
* `prefix` — Subdirectory under `--base-dir` containing `job*.{h5,csv}`. Also used for default output name.

### Options

* `-h, --help` — Show this help message and exit.
* `--base-dir, -b BASE_DIR` — Base directory that contains `<prefix>/job*.{h5,csv}`. Default: `results/data`
* `--exclude-jobs, -x [EXCLUDE_JOBS ...]` — Space-separated list of job IDs to exclude (from filenames like `job57.h5` / `job57.csv`).
* `--expected, -N EXPECTED` — Expected number of objects per input shard (rows for CSV, top-level objects for H5). If set, non-matching shards are skipped.
* `--skip-populate-sdss` — Skip `populate_sdss_fields` before writing.
* `--in-format {auto,h5,csv}` — Force input format; default auto-detect from file extension.
* `--out OUT` — Explicit output path. If omitted, defaults to `<base_dir>/<prefix>.<ext>`, where `<ext>` is derived from `--out-format`.
* `--out-format {h5,csv}` — Output format. If omitted and `--out` is given, inferred from its extension. If both omitted, defaults to `.h5`.
* `--dedup-keys [DEDUP_KEYS ...]` — Keys to use for de-duplication across shards (**last occurrence wins**). Set to `''` to disable. Default: `object_id`

## Hubble Diagram Fitting

### Usage

```bash
hubble_fit.py [-h] [--force_populate_fields] [--cosmo_models {FlatwCDM,Flatw0waCDM,FlatLambdaCDM,FlatwpwaCDM} [{FlatwCDM,Flatw0waCDM,FlatLambdaCDM,FlatwpwaCDM} ...]] [--disable_completeness] [--disable_full_covariance] [--resume [RESUME]]
              [--run {full,single}] [--speed {production,test,fast,dev}] [--N N] [--only_sna] [--use_mu_sh0es] [--spectra_fit_csv SPECTRA_FIT_CSV [SPECTRA_FIT_CSV ...]] [--zquery_csv ZQUERY_CSV] [--no_cuts] [--z_pivot_agn Z_PIVOT_AGN]
              [--skip_plots] [--fhost_cut FHOST_CUT] [--exclude_object_ids_csv EXCLUDE_OBJECT_IDS_CSV [EXCLUDE_OBJECT_IDS_CSV ...]] [--residuals_sigma_clip RESIDUALS_SIGMA_CLIP] [--residuals_csv RESIDUALS_CSV] [--agn_calibrators AGN_CALIBRATORS]
              [--redchi2_cut REDCHI2_CUT] [--iron_frac_cut IRON_FRAC_CUT] [--sdss_mags_csv SDSS_MAGS_CSV] [--result_prefix RESULT_PREFIX] [--z_range Z_RANGE Z_RANGE] [--pickled]
              agn_data_filepath
```

**Purpose:** Run Hubble fit pipeline.

### Positional arguments

* `agn_data_filepath` — Path to AGN data file.

### Options

* `-h, --help` — Show this help message and exit.
* `--force_populate_fields` — Force populate fields.
* `--cosmo_models {FlatwCDM,Flatw0waCDM,FlatLambdaCDM,FlatwpwaCDM} [{FlatwCDM,Flatw0waCDM,FlatLambdaCDM,FlatwpwaCDM} ...]` — Cosmological models list (default: `FlatwCDM`).
* `--disable_completeness` — Enable completeness correction (default: `True`).
* `--disable_full_covariance` — Use full covariance matrix for SNIa likelihood (default: `False`).
* `--resume [RESUME]` — Resume previous MCMC run (default: `False`). If a string is provided, it is used as the checkpoint file.
* `--run {full,single}` — Run mode: compare_models, compare_sna, full, or single (default: `single`).
* `--speed {production,test,fast,dev}` — Sampling speed: production, test, or fast (default: `production`).
* `--N N` — Number of AGNs to run (default: all).
* `--only_sna` — Run SNIa-only fit (default: `False`).
* `--use_mu_sh0es` — Use `MU_SH0ES` for SNIa fit (default: `False`).
* `--spectra_fit_csv SPECTRA_FIT_CSV [SPECTRA_FIT_CSV ...]` — Path(s) to spectra fit CSV file(s).
* `--zquery_csv ZQUERY_CSV` — Path to zquery CSV file.
* `--no_cuts` — Disable AGN data cuts (default: `False`).
* `--z_pivot_agn Z_PIVOT_AGN` — Pivot redshift for AGN standardization (default: `1.5`).
* `--skip_plots` — Skip plotting steps (default: `False`).
* `--fhost_cut FHOST_CUT` — Optional `fhost` cut value (default: `10`).
* `--exclude_object_ids_csv EXCLUDE_OBJECT_IDS_CSV [EXCLUDE_OBJECT_IDS_CSV ...]` — Path(s) to CSV file(s) containing object IDs to exclude.
* `--residuals_sigma_clip RESIDUALS_SIGMA_CLIP` — Optional residual cut value to exclude outliers (default: `None`).
* `--residuals_csv RESIDUALS_CSV` — Path to CSV file containing residuals for outlier exclusion (default: `None`).
* `--agn_calibrators AGN_CALIBRATORS` — Path to H5 or CSV file containing AGN data to use as calibrators (default: `None`).
* `--redchi2_cut REDCHI2_CUT` — Optional reduced chi-squared cut value to exclude outliers (default: `None`).
* `--iron_frac_cut IRON_FRAC_CUT` — Optional iron fraction cut value to exclude outliers (default: `None`).
* `--sdss_mags_csv SDSS_MAGS_CSV` — Path to CSV file containing SDSS magnitudes (default: `None`).
* `--result_prefix RESULT_PREFIX` — Prefix for result files (default: empty).
* `--z_range Z_RANGE Z_RANGE` — Redshift range for AGN data (default: `[0.44, 3.16]`).
* `--pickled` — Use pickled data file (default: `False`).

## Spectra Fitting

First, make sure you are in the `speclite` environment with  
```bash
conda activate pyqsofit
```

*Do not install speclite in your main environment*. It will downgrade numpy and may break several packages.

Fitting an AGN spectra involves several steps. First, all spectra cache needs be downloaded with  
```bash
python fit_spectra.py --download
```

Then, a run collection all different combinations of spectral fit (e.g.: balmer continuum, iron, host, quasar templates, etc) can be run with
 ```bash
python fit_spectra.py <file_in_lightcurves>  <file_out_collect> --mode collect --MC_samples 1
```

where `<file_in`> is a `.h5` file with light curve fits and `<file_out>` will be the generated csv. This is necessary since `fit_spectra.py` corrects for the mean variability amplitude when computing an apparent magnitude.

Next, a run selecting the best fits (chi squared) and penalties to avoid overfitting balmer continuum and host:
```bash
python fit_spectra.py <file_out_collect> <file_out_select> --mode select
```

Finally, a run to use the selected fits to sample and compute errors:  
```bash
python fit_spectra.py <file_in_lightcurves> <spectra_file_out> --single_csv <file_out_select> --mode single --MC_samples 50
```

The generated `<spectra_file_out>` will be used by the Hubble Diagram fitting procedure in the argument `--spectra_fit_csv`.

### Usage

```bash
fit_spectra.py [-h] --mode {collect,single,select,download} [--single_csv SINGLE_CSV] [--dr16q-fits DR16Q_FITS] [--cache-dir CACHE_DIR] [--max-sep MAX_SEP] [--N N] [--skip SKIP] [--filter_object_id FILTER_OBJECT_ID [FILTER_OBJECT_ID ...]]
               [--filter_sdss_name FILTER_SDSS_NAME [FILTER_SDSS_NAME ...]] [--spectral_fit_csv SPECTRAL_FIT_CSV] [--allow_partial_band_overlap] [--enable_BC] [--nproc NPROC] [--MC_samples MC_SAMPLES] [--enable_poly] [--disable_rescale_flux]
               fpath_in fpath_out
```

**Purpose:** DR16Q crossmatch, optional SDSS spectrum download, QSOFit processing → CSV (collect/select).

### Positional arguments

* `fpath_in` — Path to HDF5 with mag means (input to build the sample).
* `fpath_out` — Output CSV file for all QSOFit runs (collect) / Input CSV (select).

### Options

* `-h, --help` — Show this help message and exit.
* `--mode {collect,single,select,download}` — `collect`: run all configs, write one CSV; `select`: read CSV, mark `best=True` per `object_id`.
* `--single_csv SINGLE_CSV` — Optional CSV with the run configuration for single mode.
* `--dr16q-fits DR16Q_FITS` — Path to DR16Q FITS catalog.
* `--cache-dir CACHE_DIR` — Directory for cached spectra FITS.
* `--max-sep MAX_SEP` — Max match separation in arcsec.
* `--N N` — Optional limit on number of rows before matching.
* `--skip SKIP` — Optional number of rows to skip at start.
* `--filter_object_id FILTER_OBJECT_ID [FILTER_OBJECT_ID ...]` — List of object IDs to filter.
* `--filter_sdss_name FILTER_SDSS_NAME [FILTER_SDSS_NAME ...]` — List of `sdss_name`s to filter.
* `--spectral_fit_csv SPECTRAL_FIT_CSV` — Optional CSV of external spectral-fit results to merge into output (by `object_id` if present, else `sdss_name`).
* `--allow_partial_band_overlap` — Allow bands with partial wavelength overlap when computing synthetic mags.
* `--enable_BC` — Include `BC=True` runs (otherwise only `BC=False`).
* `--nproc NPROC` — Parallel worker processes for QSOFit.
* `--MC_samples MC_SAMPLES` — Number of Monte Carlo samples per object (`0` to disable MC).
* `--enable_poly` — Include polynomial component in continuum fit to account for dust reddening.
* `--disable_rescale_flux` — Disable absolute flux rescaling based on photometric mags.

## Yale HPC Notes
The full light curve and hubble diagram fitting were run in Yale's HPC clusters, consuming roughly ~100,000 CPU hours.

Several scripts useful for starting and managing slurm jobs can be found under the folder `hpc_scripts`.
