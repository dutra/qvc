# QVC: Demo and Reproducibility Guide

This repository provides end-to-end tooling for:

1. Multi-band AGN light-curve fitting
2. Spectral fitting with PyQSOFit
3. Hubble-diagram fitting and figure generation

The demo workflow downloads the needed data, fits one light curve, fits one spectrum, and recreates the publication Hubble runs with saved posteriors.

The resulting figures are found under the `docker-workdir/plots` folder (if ran with docker) or `plots` folder.

Note: For speed, the light curve fitting and spectra fitting will run with a minimal number of warmup and sampling steps, and the produced plots may have minor deviations from the published figures. The published figures were ran with longer warmup and sampling steps in Yale's HPC Clusters.

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

## System Notes

- Tested with Python `3.12.11`.
- The Docker image is CPU-only and sets `QT_QPA_PLATFORM=offscreen`, `MPLBACKEND=Agg`, and `JAX_PLATFORM_NAME=cpu`.
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

Appendix notebooks live under `notebooks/`.

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
- `hpc_scripts/sfitspectra.xsh` generates and submits spectra-fitting Slurm arrays from a chi-square selected object list, with optional H5 membership and exclusion filtering.
  ```bash
  xonsh hpc_scripts/sfitspectra.xsh
  ```
- `hpc_scripts/shubble_grid.xsh` generates and submits Hubble grid sweeps over `N` and `zmax`.
  ```bash
  xonsh hpc_scripts/shubble_grid.xsh
  ```
