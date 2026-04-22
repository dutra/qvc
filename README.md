
# QVC: Demo and Reproducibility Guide

This repository provides end-to-end tooling for:
1. Multi-band AGN light-curve fitting,
2. Spectral fitting with PyQSOFit, and
3. Hubble-diagram fitting and figure generation.

A lightweight demo workflow is included to reproduce key figures and validate installation.

## Docker Demo

A code-only Docker workflow is included for the lightweight demo. The image copies only the QVC code at build time. When the container starts, it downloads the required demo assets into a writable work directory and then runs:

1. demo data setup,
2. one light-curve fit for object `1465126`,
3. one spectra fit for the same object,
4. one fast single-model Hubble run using the downloaded AGN data.

Build the image from the repository root:

```bash
docker build -t qvc-demo .
```

Run the full demo and persist downloads/results on the host:

```bash
mkdir -p "$(pwd)/docker-workdir"
docker run --rm \
  -v "$(pwd)/docker-workdir:/work/qvc-demo" \
  qvc-demo
```

Run setup only:

```bash
docker run --rm \
  -v "$(pwd)/docker-workdir:/work/qvc-demo" \
  qvc-demo setup
```

Available container commands are:

```text
all
setup
light-curve
spectra
hubble
```

Notes:

* The container is CPU-only and sets `QT_QPA_PLATFORM=offscreen`, `MPLBACKEND=Agg`, and `JAX_PLATFORM_NAME=cpu`.
* Downloads, generated `results/`, generated `plots/`, and dustmaps files are written under `/work/qvc-demo` by default.
* The default Docker Hubble run uses `FlatLambdaCDM`, `--run single`, `--speed fast`, and `--disable_completeness` so it does not require the external Shen `pubtools` build.

---

## System Requirements

### Hardware
- A modern laptop capable of running a Python environment is sufficient to run the demo workflow.
- Disk space: **~50 MB** is required for the demo fitting a single light curve and fitting a single spectra. For the hubble diagram fitting, all posteriors must be downloaded and **~50MB** is required.

- For improved performance, we recommend **16+ GB RAM** and a modern multi-core CPU.

> Note: Full-scale light-curve and Hubble-diagram production runs were executed on Yale HPC and required on the order of **~100,000 CPU-hours**.

### Operating Systems
These routines have been tested on:
- **Linux**: Arch Linux (kernel `v6.12.62-1`)
- **macOS**: Macbook Pro M1

### Software
- A Python environment is required.
- Tested with **Python** `3.12.11`.

We recommend using a virtual environment manager such as **Conda**:
- https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html

---

## Installation

Installation typically completes in **a few minutes** on a modern machine.

### 1) Create the main (JAX CPU) environment (recommended)

From the repository root:

```bash
conda create -n jaxcpu -c conda-forge python=3.12.11
conda activate jaxcpu
```

This **main environment** is used for:
* Spectra fitting
* Multi-band light-curve fitting
* Hubble-diagram fitting

Make sure to always activate the conda environment before working with QVC.

### 3) Compile _quasarlf/pubtools_
We must compile the pubtools from
```
Xuejian Shen, Philip F Hopkins, Claude-André Faucher-Giguère, D M Alexander, Gordon T Richards, Nicholas P Ross, R C Hickox, The bolometric quasar luminosity function at z = 0–7, Monthly Notices of the Royal Astronomical Society, Volume 495, Issue 3, July 2020, Pages 3252–3275, https://doi.org/10.1093/mnras/staa1381
```
found at [https://bitbucket.org/ShenXuejian/quasarlf](https://bitbucket.org/ShenXuejian/quasarlf/)

```
git clone https://bitbucket.org/ShenXuejian/quasarlf.git
```

Follow the directions under `pubtools/clib` to compile `convolve.so` and `specialuse/convolve_ao.so`.
Export the `pubtools` folder as a shell environment variable:

```
export SHEN_PUBTOOLS_PATH=<directory of quasarlf>/quasarlf/pubtools
```

### 4) Clone the QVC repository and install
```bash
git clone https://github.com/dutra/qvc.git
cd qvc
pip install -e .
```

### 5) Fetch and setup the data files
Run 
```bash
python -m qvc.setup_data
```

By default, we only include a single Light Curve (object id 1465126) data under `data/S82` and corresponding spectra under `data/spectra_cache`.
If you have other light curve catalos or spectra, please extract them there.

To fetch the appendix datasets instead, run:

```bash
python -m qvc.setup_data --appendix
```

This downloads:

* S82 DRW fits from MacLeod et al 2010 extracted under `data/MacLeod2010`
* Fits from Stone 2021 extracted under `data/Stone2021`.
* SDSS information under `data/SDSS_DR17`. Note: This download may take a long time and it is only required when running `create_master_input_list.py` script (see next section).

## Useful Scripts

Useful repo-level scripts live under `scripts/` and are intended to be run from the repository root after `pip install -e .`.

`scripts/create_master_input_list.py` exports a master CSV of S82 light curves joined to DR17 spectroscopy metadata, with optional cuts on `variability_chi_sq_red_g` and `RUN2D`.

```bash
/home/dutra/.conda/envs/jaxcpu4/bin/python scripts/create_master_input_list.py \
  --output-csv results/data/master_input_list.csv \
  --variability_chisq_cut 20 \
  --run2d_cut v5_6_0
```

`scripts/run_demo.sh` runs the lightweight local demo workflow that sets up data and executes the example fit pipeline.

```bash
bash scripts/run_demo.sh
```

`scripts/docker_entrypoint.sh` is the container entrypoint used by the Docker demo image and supports `all`, `setup`, `light-curve`, `spectra`, and `hubble`.

```bash
bash scripts/docker_entrypoint.sh setup
```

`scripts/copy_paper_assets.sh` copies generated manuscript assets into the expected paper-output locations.

```bash
bash scripts/copy_paper_assets.sh
```

## Multi-band Light-Curve Fitting

The multi-band fit can be run for a specific Object ID. The demo data includes Object ID **1465126**, which reproduces **Figure 1**.
Run the multi-band light curve fitting with:

```bash
python -m qvc.light_curve.fit_light_curves \
 --plot \
 --progress --nwarm 200 --nsamp 100 --nchains 4 \
 --max_tree_depth 8 \
 --filter_object_id 1465126
```

For speed we run the demo with 200 warm up steps, 100 sampling steps, and a max tree depth of 8.
Figure 1 in the manuscript was produced with 2000 warmup steps, 500 sample steps, and max tree depth of 14.

Outputs:

* Fit results: `results/data/demo/`
* Plots: `plots/multiband/demo/`
* Samples: `results/samples/demo/`

The expected output is **Figure 1** under `plots/multiband/demo/`.

Additional options (example): `--rf_length_cut <days>`
For the full list of options:

```bash
python -m qvc.light_curve.fit_light_curves --help
```
---

## Hubble-Diagram Fitting

A subset of Hubble-diagram fitting and plotting can be run via:

* `notebooks/hubble_diagram_plots.ipynb`

Expected outputs resemble **Figures 2, 3, and 7**.

To reproduce the full manuscript-style Hubble-diagram pipeline, run:

```bash
xonsh run_hubble_paper.xonsh
```

`run_hubble_paper.xonsh` is the entrypoint for the full paper pipeline. It orchestrates the required `qvc.hubble.hubble_fit` runs and prefixes used for the paper outputs.

If you need a lower-level manual invocation of the Hubble fitter instead of the full pipeline script, you can still run `python -m qvc.hubble.hubble_fit ...` directly with the desired arguments.

The pipeline currently sets `speed = "test"` inside `run_hubble_paper.xonsh`. More generally, `hubble_fit` can be run with `--speed fast` for a lighter run that uses a minimum number of `dynesty` live points for a reasonable result in a few hours on a laptop. For our published results, we used `--speed production`, which uses about 500 live points for a fuller exploration of the likelihood.

---
## Spectral Fitting

This step uses the light-curve output from the previous section and fits the corresponding spectra.

```bash
python -m qvc.spectra.fit_spectra \
  --mode fit \
  results/data/demo/1465126.h5 \
  results/data/demo/spectra_1465126.csv \
  --cache-dir data/spectra_cache \
  --verbose \
  --save-fig \
  --nuts-warmup 200 \
  --nuts-samples 100 \
  --nuts-chains 1 \
  --filter_object_id 1465126 
```

---

## Appendix Figures

The following notebooks reproduce appendix figures:

* Appendix Figure **B3**: `notebooks/appendix_band_vs_wavelength.ipynb`
* Appendix Figures **C4, C5, D6, D7**: `notebooks/appendix_sigma_tau.ipynb`
* Appendix Figure **E8**: `notebooks/appendix_lag.ipynb`

## Appendix Completeness
The completeness catalogs are included in the data demo file (`qvc_data_demo`).  
The completeness catalogs can be regenerated from the completeness tooling in `notebooks/hubble_diagram_colin.ipynb` if necessary.
 


## Yale HPC Notes
The full light curve and hubble diagram fitting were run in Yale's HPC clusters, consuming roughly ~100,000 CPU hours.

Several scripts useful for starting and managing slurm jobs can be found under the folder `hpc_scripts`.
