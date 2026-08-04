# QVC: Demo and Reproducibility Guide

This repository provides end-to-end tooling for:

1. Multi-band AGN light-curve fitting
2. Spectral fitting with PyQSOFit
3. Hubble-diagram fitting and figure generation

The demo workflow downloads the needed data, fits one light curve, fits one spectrum, and recreates the publication Hubble runs with saved posteriors.

The resulting figures are found under `docker-workdir/plots/` when run with Docker, or under `plots/` for a local run.

Note: For speed, the light curve fitting and spectra fitting will run with a minimal number of warmup and sampling steps, and the produced plots may have minor deviations from the published figures. The published figures were run with longer warmup and sampling steps in Yale's HPC Clusters.

## Quick Start

```bash
git clone https://github.com/dutra/qvc.git
cd qvc
```

### Docker
For convenience, we include a self-contained Docker (https://www.docker.com) container.

```bash
docker build -t qvc-demo .
mkdir -p "$(pwd)/docker-workdir"
```

Run the full replication workflow:

```bash
docker run --rm \
  -v "$(pwd)/docker-workdir:/work/qvc-demo" \
  qvc-demo all
```

Available commands:

```text
all
setup
light-curve
spectra
hubble
```

The Docker setup step downloads the data automatically into `docker-workdir`, including the generated completeness mock used by the Hubble stage. Generated `results/`, `plots/`, and dustmaps files are written there as well.

## Local Install

You may also install the package using pip.
First, create and activate a Python environment. We recommend Conda (https://www.anaconda.com/download).

```bash
conda create -n jaxcpu -c conda-forge python=3.12.11 pip
conda activate jaxcpu
pip install -e .
```

Run the full demo workflow:

```bash
bash scripts/run_demo.sh all
```

You can also run individual stages:

```bash
bash scripts/run_demo.sh setup
bash scripts/run_demo.sh light-curve
bash scripts/run_demo.sh spectra
bash scripts/run_demo.sh hubble
```

Alternatively, you may call each command directly. Note again that these commands will run a minimal number of warmup and sampling steps for speed; the produced plots may have minor deviations from the published figures. The published figures were run with longer warmup and sampling steps in Yale's HPC Clusters.

All plots and results will be generated under `plots/` and `results/`, respectively.

### Setup

Downloads the demo inputs, cached spectrum, Hubble inputs, generated completeness mock, DSPS data, Pantheon+ files, and dustmaps data needed by the local workflow.

Expected output: downloaded inputs under `data/`, `results/data/`, `results/cosmo/`, `results/hubble_posteriors/`, and dustmaps under `results/dustmaps/`.

```bash
python -m qvc.setup_data
```

### Light Curve Fitting

Fits the demo AGN light curve for object `1452887` with reduced SVI/NUTS sampling and writes the local light-curve result files and plots.

Expected output: results under `results/data/demo/` and figures under `plots/multiband/demo/`.

```bash
PREFIX=demo SUFFIX=demo python -m qvc.light_curve.fit_light_curves \
  --filter_object_id 1452887 \
  --svi_steps 100 \
  --nwarm 100 \
  --nsamp 100 \
  --nchains 1 \
  --progress \
  --plot \
  --fit_method "svi+nuts" \
  --corner_plot_mode "full"
```

### Spectra Fitting

Fits the matching cached SDSS spectrum with `qvc.spectra.fit_spectra`, saving the output CSV, fit figures, and MCMC diagnostic plots.

Expected output: the fit CSV at `results/data/spectra/demo.csv`, result artifacts under `results/spectra/demo/`, and figures under `plots/spectra/demo/`.

```bash
python -m qvc.spectra.fit_spectra \
  --mode fit \
  "results/data/spectra/demo.csv" \
  --cache-dir "data/spectra_cache_all" \
  --output-dir "results/spectra/demo" \
  --fig-dir "plots/spectra/demo" \
  --verbose \
  --save-fig \
  --nuts-warmup 100 \
  --nuts-samples 100 \
  --nuts-chains 1 \
  --filter_object_id 1452887 \
  --plot_mcmc_diagnostics \
  --nproc 1
```

An experimental SED-first companion jointly fits a saved broadband-SED table
and the SDSS spectrum with JAXSEDFit. It keeps the native `fit_spectra` sample
matching and spectrum cache, always replaces saved SDSS points with the QVC
light-curve PSF `ugriz` means (including `z` even when it is excluded from the
variability fit), and uses JAXSEDFit's `jaxqsofit` joint spectral backend:
The summary CSV includes `m_2500_dereddened` from the intrinsic AGN disk plus
separate `a_2500_galaxy` and `a_2500_internal` attenuation estimates.

```bash
python -m qvc.spectra.fit_spectra_jaxsedfit_joint \
  --mode fit \
  "results/data/spectra/demo_joint.csv" \
  --cache-dir "data/spectra_cache_all" \
  --sed-photometry-path "/path/to/saved_sed_photometry.parquet" \
  --output-dir "results/jaxsedfit_joint/demo" \
  --fig-dir "plots/jaxsedfit_joint/demo" \
  --dsps-ssp-fn "../jaxqsofit/tempdata.h5" \
  --filter_object_id 1452887 \
  --progress \
  --nproc 1
```

### Hubble Diagram Fitting

Resumes the saved fiducial and restricted Hubble posterior checkpoints and regenerates the comparison outputs and plots.

Expected output: results under `results/` and figures under `plots/hubble/paper_hubble_final_production/` and `plots/hubble/paper_hubble_final_production_restricted/`.

Set `QVC_HUBBLE_MAGNITUDE_CONVENTION` explicitly to either `dereddened` or
`attenuated` before running either command. There is no default. Hubble fitting
accepts only CSV output from `fit_spectra_jaxsedfit_joint.py`.

#### Fiducial fit

```bash
python -m qvc.hubble.hubble_fit --resume \
  --cosmo_models FlatLambdaCDM FlatwCDM Flatw0waCDM \
  --run full \
  --speed production \
  --spectra_fit_csv "results/data/jul29_2206_9dad47e_sedfit.csv" \
  --magnitude-convention "$QVC_HUBBLE_MAGNITUDE_CONVENTION" \
  --completeness_sim_file "results/data/mock_completeness_catalog_fresh.h5" \
  --z_range 0.44 3.16 \
  --result_prefix "fiducial" \
  --prefix "paper_hubble_final_production" \
  --sigma_clip_threshold 3.0 \
  "results/data/lc_data_all.h5"
```

#### Restricted fit

```bash
python -m qvc.hubble.hubble_fit --resume \
  --cosmo_models FlatLambdaCDM FlatwCDM Flatw0waCDM \
  --run full \
  --speed production \
  --spectra_fit_csv "results/data/jul29_2206_9dad47e_sedfit.csv" \
  --magnitude-convention "$QVC_HUBBLE_MAGNITUDE_CONVENTION" \
  --completeness_sim_file "results/data/mock_completeness_catalog_fresh.h5" \
  --z_range 1.0 3.16 \
  --result_prefix "restricted" \
  --prefix "paper_hubble_final_production_restricted" \
  --sigma_clip_threshold 3.0 \
  "results/data/lc_data_all.h5"
```


## System Notes

- Tested with Python `3.12.11`.
- Tested on an AMD x86_64 desktop, MacBook Air M1, and Yale HPC.
- All required Python packages and versions are listed under `requirements.txt` and will be installed automatically with the `pip install -e .` command.
- Typical local install time is on the order of 10 minutes and less than 1 hour on a normal desktop computer, depending on hardware and internet speed.
- Local demo runtime is on the order of a few hours on a normal computer.
- Full-scale production runs were executed on Yale HPC and required over `100,000` CPU-hours.

## Optional Pubtools Build

We generated the Hubble completeness mock catalog with Shen et al. `quasarlf/pubtools`. For reproduction, setup downloads the generated catalog at `results/data/mock_completeness_catalog_fresh.h5`, so Docker users do not need to compile `pubtools`.

Only build `pubtools` if you want to regenerate that mock catalog:

```bash
git clone https://bitbucket.org/ShenXuejian/quasarlf.git
export SHEN_PUBTOOLS_PATH=<directory of quasarlf>/quasarlf/pubtools
```

Follow the `pubtools/clib` instructions in that repository to compile the shared libraries.

## Useful Scripts

`scripts/run_demo.sh` runs setup plus the light-curve, spectra, and Hubble stages.

`scripts/docker_entrypoint.sh` is the Docker entrypoint and accepts `all`, `setup`, `light-curve`, `spectra`, and `hubble`.

`scripts/create_master_input_list.py` exports a master CSV of S82 light curves joined to DR17 spectroscopy metadata.

`scripts/copy_paper_assets.sh` copies generated manuscript assets into paper-output locations.

## Appendix Data

To fetch appendix datasets:

```bash
python -m qvc.setup_data --appendix
```

The Stone, MacLeod, and same-length comparison plots were generated using Yale's HPC by running the `hpc_scripts/sfitlc.py` script.

## HPC Scripts
The `hpc_scripts` folder contains Slurm/Yale-HPC-oriented helpers. Treat these as templates: partitions, paths, Conda environments, and account-specific settings may need local edits.

- `hpc_scripts/smanage.py` inspects, cancels, holds, or resumes Slurm jobs by job-name glob.
  ```bash
  python hpc_scripts/smanage.py status "multiband_*"
  ```
- `hpc_scripts/sfitlc.py` generates and submits multiband light-curve Slurm arrays for `chisq`, `stone`, `macleod`, and `samelength` samples, including merge jobs.
  ```bash
  python hpc_scripts/sfitlc.py --fit stone --N 1
  ```
- `hpc_scripts/sfitspectra.xsh` generates and submits spectra-fitting Slurm arrays from a chi-square selected object list, with optional exclusion filtering.
  ```bash
  xonsh hpc_scripts/sfitspectra.xsh
  ```
  To retry only tasks whose latest attempt ended unsuccessfully, pass the full
  original job name:
  ```bash
  xonsh hpc_scripts/sfitspectra.xsh \
    --retry aug03_1853_f3ea5a6_svi4000_N2000_PRS103to107
  ```
  Retry mode requires the original generated
  `hpc_scripts/submit/jaxqsofit/submit_<job>.sbatch` script and matching
  `<job>_object_ids.txt` manifest. It searches Slurm accounting records from
  the last seven days, preserves the original task and chunk numbers, and
  applies the scheduler resources currently configured in `sfitspectra.xsh`.
  Every object in an unsuccessful chunk is rerun; completed chunks are left
  untouched.
- `hpc_scripts/shubble_grid.xsh` generates and submits Hubble grid sweeps over `N` and `zmax`.
  ```bash
  xonsh hpc_scripts/shubble_grid.xsh
  ```
