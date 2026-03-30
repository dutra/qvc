
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

Installation typically completes in **a few minutes** on a modern machine. Downloading the demo data may take longer depending on network speed.

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

### 2) Install EzTaoX
For now, we need to install EzTaoX from the repository (the one installed with `pip install eztaox` is out-of-date):
```bash
git clone https://github.com/LSST-AGN-Variability/EzTaoX
pip install EzTaoX
```

### 3) Clone the QVC repository and install
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

3) Hubble-Diagram Fitting

A subset of Hubble-diagram fitting and plotting can be run via:

* `notebooks/hubble_diagram_plots.ipynb`

Expected outputs resemble **Figures 2, 3, and 7**.

To reproduce all Hubble-diagram plots from the manuscript, the full fitting procedure can be run with:

```bash
python -m qvc.hubble.hubble_fit \ 
    --cosmo_models FlatLambdaCDM FlatwCDM Flatw0waCDM \
    --run full \ 
    --wrms_cut 1.2 \ 
    --fhost_cut 0.1 \ 
    --iron_frac_cut 0.001 \ 
    --bc_frac_cut 0.001 \ 
    --variability_chi_sq_cut 10.0 \ 
    --speed fast \ 
    --spectra_fit_csv results/data/jaxqsofit_mar15c.csv \ 
    --z_range 0.44 3.16 \ 
    --result_prefix fiducial \ 
    --prefix demo \ 
    "results/data/nov10a_single_chisq_carma_mixscalar_nozband_highertaufastlim_removemix_fixband_lagblrband_chisq_spl_nofhost_bwb_lmc-6_N1w1000s200t14ch4.h5"
```

---
# TBD

## 3) Spectral Fitting
TBD 

This step uses the light-curve output from the previous section and fits the corresponding spectra.


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
