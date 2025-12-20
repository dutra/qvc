
# QVC: Demo and Reproducibility Guide

This repository provides end-to-end tooling for:
1. Multi-band AGN light-curve fitting,
2. Spectral fitting with PyQSOFit, and
3. Hubble-diagram fitting and figure generation.

A lightweight demo workflow is included to reproduce key figures and validate installation.

---

## System Requirements

### Hardware
- A modern laptop capable of running a Python environment is sufficient to run the demo workflow.
- Disk space: **~44 GB** is required for the demo data.
- For improved performance, we recommend **16+ GB RAM** and a modern multi-core CPU.

> Note: Full-scale light-curve and Hubble-diagram production runs were executed on Yale HPC resources and required on the order of **~100,000 CPU-hours**.

### Operating Systems
These routines have been tested on:
- **Linux**: Arch Linux (kernel `v6.12.62-1`)
- **macOS**: _TBD (Colin: please fill in version and hardware details)_

### Software
- A Python environment is required.
- Tested with **Python `3.13.5`** (and **Python `<Colin>`** for macOS).

We recommend using a virtual environment manager such as **Conda**:
- https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html

The list of all python dependencies (installed in the next step) are listed under [environment.yml](environment.yml)

---

## Installation

Installation typically completes in **under one hour** on a modern machine. Downloading the demo data may take longer depending on network speed.

### 1) Create the main (JAX CPU) environment (recommended)

From the repository root:

```bash
conda env create -n jaxcpu -f environment.yml
conda activate jaxcpu
```

This **main environment** is used for:

* Multi-band light-curve fitting
* Hubble-diagram fitting


### 2) Clone the repository
```bash
git clone https://github.com/dutra/qvc.git
cd qvc
git checkout preview
```

### 3) Install `eztaox`

```bash
cd eztaox
pip install .
```

---

## PyQSOFit Environment (for Spectral Fitting)

Spectral fitting relies on PyQSOFit and additional dependencies. We strongly recommend using a **separate environment**.

### 1) Create a dedicated environment

**Important:** Installing `speclite` can downgrade NumPy and may break packages in your main environment. Keep PyQSOFit isolated.

From inside the `PyQSOFit` directory:

```bash
conda env create -n pyqsofit python=3.12.9 -f environment.yml
conda activate pyqsofit
```

### 2) Clone PyQSOFit

From the `qvc` repository root:

```bash
git clone https://github.com/dutra/PyQSOFit.git
cd PyQSOFit
```


### 3) Install PyQSOFit

```bash
pip install .
```

---

## Download the Demo Data

1. Download the demo data from: **`<url>`**
2. Extract it into the `src/` directory.

---

# Demo Workflow

## 1) Multi-band Light-Curve Fitting

The multi-band fit can be run for a specific Object ID. The demo data includes Object ID **1465126**, which reproduces **Figure 1**.

From the repository root:

```bash
cd src
```

Run:

```bash
export PREFIX=demo
python multiband_fit.py --plot \
    --progress --nwarm 1000 --nsamp 500 --nchains 4 \
    --max_tree_depth 14 \
    --disable_fhost \
    --bwb \
    --filter_object_id 1465126
```

Outputs:

* Fit results: `src/results/data/demo/`
* Plots: `src/plots/multiband/demo/`
* Samples: `src/results/samples/demo/`

The expected output is **Figure 1** under `src/plots/multiband/demo/`.

Additional options (example): `--rf_length_cut <days>`
For the full list of options:

```bash
python multiband_fit.py --help
```

---

## 2) Spectral Fitting

This step uses the light-curve output from the previous section and fits the corresponding spectra.

### Activate the PyQSOFit environment

```bash
conda activate pyqsofit
```

**Do not install `speclite` in your main environment.** It can downgrade NumPy and destabilize other dependencies.

### Download the spectral cache

```bash
python fit_spectra.py --download
```

### Run the spectral fitting pipeline

1. **Collect**: run multiple template/continuum configurations

```bash
python fit_spectra.py results/data/demo/1465126.h5 results/data/demo/1465126_collect.csv \
    --mode collect --MC_samples 1
```

2. **Select**: choose best fits (via chi-squared) with penalties to avoid overfitting Balmer continuum and host components

```bash
python fit_spectra.py results/data/demo/1465126_collect.csv results/data/demo/1465126_select.csv \
    --mode select
```

3. **Single**: sample the selected fit(s) to estimate uncertainties

```bash
python fit_spectra.py results/data/demo/1465126.h5 results/data/demo/1465126.csv \
    --single_csv results/data/demo/1465126_select.csv \
    --mode single --MC_samples 50
```

The output file:

* `results/data/demo/1465126.csv`

This contains spectral-fit results, including the apparent magnitude at **rest-frame 2500 Å**, and can be passed to the Hubble-diagram fitting step via `--spectra_fit_csv`.

---

## 3) Hubble-Diagram Fitting

A subset of Hubble-diagram fitting and plotting can be run via:

* `hubble_diagram_plots.ipynb`

Expected outputs resemble **Figures 2, 3, and 7**.

To reproduce all Hubble-diagram plots from the manuscript, use the saved **dynesty** checkpoints downloaded with the demo data. The full fitting procedure can be run with:

```bash
PREFIX=demo \
python hubble_fit.py --resume \
    --cosmo_models FlatLambdaCDM FlatwCDM Flatw0waCDM \
    --run full \
    --speed production \
    --spectra_fit_csv "results/data/nov12a_11c_single_scratch_nov10a_carma_removemix_fixmeanband_no1pluszflux2L_freeiron_mc50_best.csv" \
    --sdss_mags_csv "results/data/nov2_sdss_mags.csv" \
    --zquery_csv "results/data/sep19_chisq_zquery.csv" \
    --z_range 0.44 3.16 \
    "results/data/nov10a_single_chisq_carma_mixscalar_nozband_highertaufastlim_removemix_fixband_lagblrband_chisq_spl_nofhost_bwb_lmc-6_N1w1000s200t14ch4.h5"
```

The `--resume` flag loads the dynesty checkpoint.

### Restricted redshift fit

For the restricted redshift interval (**z = 1.0–3.16**):

```bash
PREFIX=demo_zonecut \
python hubble_fit.py --resume \
    --cosmo_models FlatLambdaCDM FlatwCDM Flatw0waCDM \
    --run full \
    --speed production \
    --spectra_fit_csv "results/data/nov12a_11c_single_scratch_nov10a_carma_removemix_fixmeanband_no1pluszflux2L_freeiron_mc50_best.csv" \
    --sdss_mags_csv "results/data/nov2_sdss_mags.csv" \
    --zquery_csv "results/data/sep19_chisq_zquery.csv" \
    --z_range 1.0 3.16 \
    "results/data/nov10a_single_chisq_carma_mixscalar_nozband_highertaufastlim_removemix_fixband_lagblrband_chisq_spl_nofhost_bwb_lmc-6_N1w1000s200t14ch4.h5"
```

As with `multiband_fit.py`, `hubble_fit.py` uses the `PREFIX` environment variable.

Outputs:

* All Hubble-diagram plots, tables, and results are generated under: `src/plots/hubble/`

Expected figures:

* **Figures 2–8, A1, and A2** under `src/plots/hubble/` for:

  * `FlatLambdaCDM`
  * `FlatwCDM`
  * `Flatw0waCDM`

Additional outputs:

* **Table 4**
* A printed summary of the cosmological model with the highest evidence (expected: `Flatw0waCDM`)

Typical runtime (demo checkpoints): **< 1 hour**.

---

## Appendix Figures

The following notebooks reproduce appendix figures:

* Appendix Figure **B3**: `appendix_band_vs_wavelength.ipynb`
* Appendix Figures **C4, C5, D6, D7**: `appendix_sigma_tau.ipynb`
* Appendix Figure **E8**: `appendix_lags.ipynb`


# Instructions for use
## `multiband_fit.py`

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
