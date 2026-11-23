# QVC: Demo and Reproducibility Guide

This repository provides end-to-end tooling for:

1. Multi-band AGN light-curve fitting
2. Joint SED and spectral fitting with JAXSEDFit
3. Hubble-diagram fitting and figure generation

The quick-start workflow downloads the published Zenodo summaries and recreates the Hubble runs with saved posteriors. The single-object light-curve and spectrum setup modes download their own inputs (see below).

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

Run the published Hubble replication workflow:

```bash
docker run --rm \
  -v "$(pwd)/docker-workdir:/work/qvc-demo" \
  qvc-demo hubble
```

Available commands:

```text
all
setup
light-curve
spectra
hubble
```

The `hubble` command runs `setup_data --paper` automatically inside `docker-workdir`, then strictly resumes the published checkpoints. It generates the Wang completeness mock locally and copies the paper figures and tables into `docker-workdir/plots/paper/`. If CSC is unavailable, the two X-ray figures are omitted from the copy; any older destination copies remain unchanged. Results and plots remain in the mounted directory. Set `QVC_HUBBLE_SKIP_PAPER_ASSET_COPY=1` explicitly to skip copying. The image defaults to `hubble` when no command is supplied.

`light-curve` automatically downloads the single-object bundle for SDSS J233151.91+003258.3 (object `1448770`) and fits it. `spectra` installs the same object’s photometry, its cached spectrum, stellar templates, and dust maps. It uses the installed `jaxsedfit` backend to generate both the joint SED and spectrum figures; standalone `jaxqsofit` is not required. `all` still requires preinstalled raw inputs. Full `setup` also downloads the fitting dependencies and dustmaps into the mounted directory.

## Local Install

You may also install the package using pip.
First, create and activate a Python environment. We recommend Conda (https://www.anaconda.com/download).

```bash
conda create -n jaxcpu -c conda-forge python=3.12.11 pip
conda activate jaxcpu
pip install -e .
pip install --force-reinstall --no-deps "jaxsedfit @ git+https://github.com/burke86/jaxsedfit.git"
```

JAXSEDFit follows the upstream default branch without a commit pin. The final
command refreshes it even when upstream has not changed its package version;
repeat it when updating an existing environment. Catalog manifests record the
installed revision for provenance.

Run the published Hubble replication workflow:

```bash
bash scripts/run_demo.sh hubble
```

Both the light-curve and spectrum commands install their own demo inputs:

```bash
bash scripts/run_demo.sh light-curve
bash scripts/run_demo.sh hubble
bash scripts/run_demo.sh spectra
```

Alternatively, you may call each command directly. Note again that these commands will run a minimal number of warmup and sampling steps for speed; the produced plots may have minor deviations from the published figures. The published figures were run with longer warmup and sampling steps in Yale's HPC Clusters.

All plots and results will be generated under `plots/` and `results/`, respectively.

The paper launcher (`run_hubble_paper.xonsh`) fixes both the fiducial and
restricted samples at `--agn-sigma-pivot 0.2 --agn-tau-pivot 200` (magnitudes and
rest-frame days). Other launchers retain sample-median pivots. Supply both options
together with finite, positive values; they cannot be combined with
`--disable-pivot-rounding`. Checkpoint resumes require matching pivot values and
rules. The paper launcher resumes the published `fiducial` and `restricted`
checkpoints; missing checkpoints fail rather than starting a new fit.

### Setup

Downloads the cached spectrum, DR16Q catalog, Hubble inputs, legacy completeness mock, DSPS data, Pantheon+ files, CSC catalog, and dustmaps. For the single-object raw-photometry bundle alone, use `python -m qvc.setup_data --light-curve`. Use `python -m qvc.setup_data --spectra` to also install its cached spectrum, DSPS templates, and dust maps.

Expected output: downloaded inputs under `data/`, `results/data/`, `results/cosmo/`, `results/hubble_posteriors/`, and dustmaps under `results/dustmaps/`.

```bash
python -m qvc.setup_data
```

For the published paper's Hubble resume only:

```bash
python -m qvc.setup_data --paper
QVC_HUBBLE_SKIP_PAPER_ASSET_COPY=1 python -m xonsh --no-rc run_hubble_paper.xonsh
```

This downloads the checksum-verified inputs from
[Zenodo record 23021880](https://zenodo.org/records/23021880):
`results/data/light_curve_data.h5`, `results/data/spectra_data.h5`, and the
12 checkpoints under `results/hubble_posteriors/{fiducial,restricted}/`.
The paper-only setup also downloads the external Pantheon+ data and covariance
and queries the official CSC 2.1 TAP service for the paper's X-ray region
(-1.251 to +1.251 degrees declination, all right ascensions). It validates the
response and produces
`results/diagnostics/chandra_precut_sep13/csc21_region_download.vot`, the CSC
catalog read by the paper launcher. Existing valid catalogs are reused.
CSC network failures (including HTTP errors and timeouts) produce a warning;
setup continues and retries CSC on its next invocation. When the catalog is
missing, Hubble fitting continues with the same sample and fit settings,
but omits CSC matching and X-ray-dependent diagnostics. Setup reports that CSC
is unavailable. Invalid catalog responses and failures downloading required
inputs still stop setup. Source matching, when available, uses the catalog's
positional-error ellipses during the Hubble analysis.
The launcher regenerates the Wang completeness mock; raw fitting catalogs,
per-object plot archives, and `Table_7.csv` are not required for this resume.
The command above skips the full manuscript asset-copy step, which needs assets outside this download. Omit `QVC_HUBBLE_SKIP_PAPER_ASSET_COPY=1` only when those assets are available.

### Raw inputs for optional refitting

The Zenodo HDF5 files contain fit summaries, not individual photometric epochs.
The fitting loaders use these files under `data/S82/`:

- `Catalog.parquet`
- `dr16s82_sdssLCRaw.parquet`
- `dr16s82_ps1LCRaw.parquet`
- `dr16s82_ZuberLCRaw.parquet`

For SDSS J233151.91+003258.3 (object `1448770`), download the raw inputs with:

```bash
python -m qvc.setup_data --light-curve
```

This installs the four Parquet files above, three matching seeing sidecars, and
a one-row `data/dr16q_prop_May01_2024.fits` from the
[Google Drive bundle](https://drive.google.com/file/d/1MghLBBfCaS-7e8jHyI7yWzKja3vpVP85/view).
The archive checksum and file list are verified before installation. Existing
complete inputs are reused; a partially populated destination is rejected to
avoid mixing or overwriting catalogs. Set `QVC_DATA_DIR` to a separate empty
directory when needed. Other objects still require their original raw inputs.
The bundle contains no posterior samples: `--resume` requires a separately
saved per-object sample file under `results/samples/<PREFIX>/`.

### Light Curve Fitting

Fits the demo AGN light curve for SDSS J233151.91+003258.3, object `1448770` with reduced SVI/NUTS sampling and writes the local light-curve result files and plots.

Expected output: results under `results/data/demo/` and figures under `plots/multiband/demo/`.

```bash
PREFIX=demo SUFFIX=demo python -m qvc.light_curve.fit_light_curves \
  --filter_object_id 1448770 \
  --svi_steps 100 \
  --nwarm 100 \
  --nsamp 100 \
  --nchains 1 \
  --progress \
  --plot \
  --fit_method "svi+nuts" \
  --corner_plot_mode "full"
```

The experimental shared-latent response model is available through a
dedicated entry point:

```bash
PREFIX=latent_demo SUFFIX=latent_demo \
python -m qvc.light_curve.fit_light_curves_shared_latent_blr \
  --filter_object_id 1448770 \
  --nwarm 500 \
  --nsamp 250 \
  --nchains 2 \
  --plot
```

It uses one unit-RMS DHO driver for every band. Each observed continuum is a
narrow Erlang-convolved disk response whose centroid follows a shared
`lambda^(4/3)` law, while each band's generic delayed component has its own
lag and unit-RMS Erlang response. The same model can be selected from the main
entry point with `--model_variant shared_latent_blr`. `--disk_order` controls
the disk-response width (default 3); `--erlang_order` controls the delayed
response width (default 3). No emission-line, Fe, or diffuse-continuum
wavelength template is imposed. The shared driver is wavelength-independent,
so this variant has no `eta_tau`; `--eta_prior_profile modified` changes only
its `eta_sigma` prior.

For this variant, `log_tau_uv_rf` is the exact rest-frame integral-correlation
timescale of a synthetic continuum-only disk response evaluated at 2500 A.
It excludes every delayed/BLR component. `log_tau_driver_rf` is the intrinsic
shared-driver integral timescale, `log10((tau_fast + tau_slow)/(1 + z))`, with
the individual pole diagnostics retained as `log_tau_driver_fast_rf` and
`log_tau_driver_slow_rf`.

For output compatibility with the other light-curve models,
`log_tau_band_<band>_RF` is the rest-frame effective timescale of that band's
complete disk-plus-delayed light curve. It is the exact integral of the
normalized autocorrelation function, not a fitted response centroid. The same
value is also written as `log_tau_effective_<band>_RF`.

Seeing-dependent photometric contamination can be enabled with
`--enable_seeing_dependence` for any GP choice, including
`shared_latent_blr` and the CARMA(2,1) parameterization. For each band/survey
combination with at least three usable epoch-level PSF-FWHM measurements and
nonzero seeing range, this fits both a centered mean-flux slope and a
seeing-dependent extra-scatter slope. It is disabled by default.

### Joint SED and Spectrum Fitting

The `spectra` demo fits SDSS J233151.91+003258.3 (object `1448770`,
plate 9202, MJD 57713, fiber 9) with `qvc.spectra.fit_spectra_jaxsedfit_joint`.
A single joint fit produces both the broadband SED figure and the resolved
spectrum figure, using the `jaxsedfit` dependency already installed in Docker.
The older standalone `fit_spectra` command is not used by this demo.

```bash
bash scripts/run_demo.sh spectra
```

Setup downloads the single-object photometry/DR16Q bundle and the
[spectrum and SED bundle](https://drive.google.com/file/d/1RRRLpcsvros4K-AKW8dAwHBEhVK8y2FT/view),
which includes the 10-row UV/near-IR/WISE SED table, installs its packaged provenance,
and prepares the stellar templates and dust maps. SDSS ugriz photometry comes
from the object's light-curve means. No additional SED download is needed.
To install just these inputs, run `python -m qvc.setup_data --spectra`.

Default outputs:

- `results/data/spectra/demo.h5`: fit catalog.
- `results/spectra/demo/`: saved joint posterior bundle.
- `plots/spectra/demo/z1.010_233151.91+003258.3_joint.pdf`: joint SED figure.
- `plots/spectra/demo/z1.010_233151.91+003258.3_spectrum.pdf`: spectrum figure.

A short execution check (not a convergence test):

```bash
DEMO_OPTAX_STEPS=10 DEMO_NUTS_WARMUP=5 DEMO_NUTS_SAMPLES=5 \
  bash scripts/run_demo.sh spectra
```

Pass these variables with Docker's `-e` option for the same short run in a
container. Their defaults are 100 optimization steps, 100 warmup steps, and
100 posterior draws in one chain. Set `DEMO_PREFIX` to keep separate runs.
For another object, supply its input data and set `DEMO_OBJECT_ID` and
`QVC_SED_PHOTOMETRY_PATH` to that object's saved broadband table.

### Hubble Diagram Fitting

Resumes the saved fiducial and restricted Hubble posterior checkpoints and regenerates the comparison outputs and plots.

`--resume` remains strict and fails when a requested checkpoint is absent. Use
`--resume-if-exists` with the same optional per-model H5 paths to resume files
that exist and start fresh fits for files that do not. For `run_hubble.xonsh`,
set `QVC_HUBBLE_RESUME_IF_EXISTS=true` to check the default checkpoint paths, or
set it to an explicit checkpoint path; do not combine it with
`QVC_HUBBLE_RESUME`.

Expected output: results under `results/` and figures under `plots/hubble/fiducial/` and `plots/hubble/restricted/`.

Both Docker and local execution use `run_hubble_paper.xonsh`, with the same
Zenodo input names, cut settings, fixed pivots, Wang completeness model, and
`fiducial`/`restricted` checkpoint directories. The default magnitude convention
is `dereddened`; the completeness magnitude is `attenuated`. Preserve these
settings when reproducing the published checkpoints.

```bash
bash scripts/run_demo.sh hubble
```

For an already downloaded dataset, the equivalent direct command is:

```bash
QVC_HUBBLE_SKIP_PAPER_ASSET_COPY=1 python -m xonsh --no-rc run_hubble_paper.xonsh
```

The launcher reads `results/data/light_curve_data.h5` and
`results/data/spectra_data.h5`. The latter is the versioned minimal spectral
HDF5 schema accepted by the Hubble loader; a legacy spectral CSV is not needed.
It uses redshift ranges 0.44–3.16 (fiducial) and 0.7–3.16 (restricted).


## System Notes

- Tested with Python `3.12.11`.
- Tested on an AMD x86_64 desktop, MacBook Air M1, and Yale HPC.
- All required Python packages and versions are listed under `pyproject.toml` and will be installed automatically with the `pip install -e .` command.
- Typical local install time is on the order of 10 minutes and less than 1 hour on a normal desktop computer, depending on hardware and internet speed.
- Local demo runtime is on the order of a few hours on a normal computer.
- Full-scale production runs were executed on Yale HPC and required over `100,000` CPU-hours.

## Optional Pubtools Build

The published Hubble launcher generates its Wang completeness mock using the Python implementation and does not need `pubtools`. Full setup also retains the legacy Shen mock at `results/data/mock_completeness_catalog_fresh.h5`.

Only build `pubtools` if you want to regenerate that mock catalog:

```bash
git clone https://bitbucket.org/ShenXuejian/quasarlf.git
export SHEN_PUBTOOLS_PATH=<directory of quasarlf>/quasarlf/pubtools
```

Follow the `pubtools/clib` instructions in that repository to compile the shared libraries.

## Useful Scripts

`scripts/run_demo.sh hubble` runs paper setup and Hubble reproduction. Its `all` command additionally runs light-curve and spectrum fitting when raw inputs are present.

`scripts/docker_entrypoint.sh` is the Docker entrypoint and accepts `all`, `setup`, `light-curve`, `spectra`, and `hubble`.

`scripts/create_master_input_list.py` exports a master CSV of S82 light curves joined to DR17 spectroscopy metadata.

`scripts/copy_paper_assets.sh` copies generated manuscript assets into paper-output locations.

## Appendix Data

To fetch appendix datasets:

```bash
python -m qvc.setup_data --appendix
```

The Hubble launcher supports optional redshift evolution in the mean AGN
absolute-magnitude relation and in its intrinsic scatter:

```bash
QVC_HUBBLE_MODE=agn-only-no-bright-cut \
xonsh run_hubble.xonsh --fit_redshift_mean_term --fit_redshift_log_f_term
```

Either flag can be used alone. The mean term adds
`gamma_mean_z * log10((1+z)/(1+z_pivot))` to predicted absolute magnitude,
where its pivot is the median redshift of the fit's immutable reference sample.
The scatter term evolves `log_f(z)` using the existing scatter pivot.
Both gamma priors are uniform on `[-20, 20]`. Default launcher output prefixes
gain `_meanz` and/or `_logfz`; explicit prefix overrides remain unchanged.
These options use the Dynesty backend. The AGN-only preset retains completeness
correction but disables the bright-subsample cut and uses `tails` magnitude
support with the 14–32 guard.

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
  xonsh hpc_scripts/sfitspectra.xsh \
    --chisq-csv data/jul14_master_input_file_chisqgt20_nested_N8000.csv \
    --description nested_N8000
  ```
  Add `--fit-bal` to enable JAXSEDFit's built-in N V, Si IV, and C IV BAL
  absorption components in the default `fit_spectra_jaxsedfit_joint.py`
  backend. It can be combined with `--resume` for posterior bundles produced
  by a BAL-enabled run; the saved JAXSEDFit configuration is validated before
  reuse.
  ```bash
  xonsh hpc_scripts/sfitspectra.xsh \
    --chisq-csv data/jul14_master_input_file_chisqgt20_nested_N8000.csv \
    --description nested_N8000_BAL \
    --fit-bal
  ```
  To regenerate catalogs and plots without any fresh inference, add
  `--resume OLD_RUN_NAME --resume-only`. Saved bundles are read from
  `results/data/jaxqsofit/OLD_RUN_NAME/all/`; outputs go into a new run.
  Individual failures (including missing bundles) produce `fit_ok=False`
  catalog rows and processing continues. A chunk with any successful objects
  exits successfully; an entirely failed chunk writes its failure catalog
  before exiting nonzero. Inspect `fit_ok` and `error_message` even for
  completed Slurm tasks.

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
