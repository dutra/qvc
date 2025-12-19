# README

This project is divided into two main components:
a. Light curve fitting
b. Hubble diagram fitting

## Setting up eztaox
### Cloning the project:

``git clone git@github.com:dutra/eztaox.git``

``cd eztaox``

``git checkout preview``

### Create new conda env
Assuming conda is installed, load it in your system. On Yale's HPC, you will need to use
``module load miniconda``

## Set up a CPU environment (suggested)
`conda create --name jaxcpu python=3.13.5 jax jaxopt jupyter jupyterlab numpyro tinygp matplotlib astropy jaxopt h5py tqdm ipykernel pandas corner scipy`
`conda activate jaxcpu`


## [Deprecated] Set up a GPU environment
``conda create -n jaxgpu python=3.12.11 jupyter jupyterlab "jax[cuda12]" numpyro tinygp matplotlib astropy jaxopt h5py tqdm ipykernel pandas corner scipy ``

``conda activate jaxgpu``


## Installing eztaox for development
delete the dependencies in dependencies array in eztaox/pyproject.toml

in eztaox folder, run

``pip install .``

you may need to install other packages with pip, make a note of them:

pip install <package>

## Upload data

``cd eztaox/tutorials/data ``

``curl -O http://quasar.astro.illinois.edu/paper_data/DR16Q/dr16q_prop_May01_2024.fits.gz gunzip dr16q_prop_May01_2024.fits.gz``

Also under tutorials/data, upload the parquet files from https://www.dropbox.com/scl/fo/7a6wz93al26xm8xzv2355/ANJTBMmTSwMywlr0wSO4esM?rlkey=66tshxe23ezyjiohcrpjkdne9&e=1&st=9od96bti&dl=0

## Load the conda environment every time before working:

``module load miniconda``

``conda activate jaxcpu``



# PyQSOFit

# Light Multiband fitting
The Light multiband fitting can be run by specifying an Object ID.
The Light Curve figure included in the manuscript is 1465126.

``
python multiband_fit.py --plot \
    --progress --nwarm 1000 --nsamp 500 --nchains 4 \
    --max_tree_depth 14 \
    --disable_fhost \
    --bwb \
    --filter_object_id 1465126
``

The `multiband_fit.py` script accepts other options such as `--rf_length_cut <days>`.
For a list of all options supported, run `python multiband_fit.py --help`.

For help organizing the batch runs in Yale's HPC, `multiband_fit.py` can use the environment varibles `PREFIX` and `SUFFIX`, which default to `test`.
The results will be written under the folder `results/data/<prefix>`, while plots will be under `plots/multiband/<prefix>` and samples under `results/samples/<prefix>`.

## Merging the results
Each `multiband_fit.py` run will produce one `.h5` file per object under the folder `results/data/<prefix>`.
The hubble diagram script loads a single `.h5` with multiple light curve fit results. 
You can merge multiple `.h5` files into one by using the utility
`python merge_results.py <prefix>`.
That will produce a file `results/data/<prefix>.h5` that can be loaded with the hubble diagram utility.

# Hubble diagram fitting
The hubble diagram fitting procedure can be ran with the command

```
python hubble_fit.py \
    --cosmo_models FlatLambdaCDM FlatwCDM Flatw0waCDM \
    --run <run_speed> \
    --speed <speed> \
    --spectra_fit_csv <spectra_fit_csv> \
    --sdss_mags_csv <sdss_mags_csv>  \
    --zquery_csv <zquery_csv>  \
    --z_range 0.44 3.16 \
    --result_prefix "" \
    <h5_file>
```

For more information and a list of all available options, `python hubble_fit.py --help`.
In particular,
the `<run_speed>` can be chosen from `fast`, `dev`, `test` and `production`; each increases the number of initial and dynamic live points and tighten the dlogz for the Bayesian evidence.
For example, the `production` option uses an `2000` as effective number of samples, starts with `1000` or `50*ndim` whichever is greater (where `ndim` is the number of dimensions to fit), `500` for the number of batch live points and a `dlogz` of 0.01.

Similarly as `multiband_fit.py`, `hubble_fit.py` will use the environment variable `PREFIX`.
All hubble diagram relevant plots, tables, and results will be generate under `plots/hubble/<prefix>`.

To reproduce the results in the submitted manuscripts, please run it with
```
python hubble_fit.py \
    --cosmo_models FlatLambdaCDM FlatwCDM Flatw0waCDM \
    --run full \
    --speed production \
    --spectra_fit_csv "results/data/nov12a_11c_single_scratch_nov10a_carma_removemix_fixmeanband_no1pluszflux2L_freeiron_mc50_best.csv" \
    --sdss_mags_csv "results/data/nov2_sdss_mags.csv"  \
    --zquery_csv "results/data/sep19_chisq_zquery.csv"  \
    --z_range 0.44 3.16 \
    "results/data/nov10a_single_chisq_carma_mixscalar_nozband_highertaufastlim_removemix_fixband_lagblrband_chisq_spl_nofhost_bwb_lmc-6_N1w1000s200t14ch4.h5"
```


# Yale HPC Notes
## Show your jobs status:
`squeue -u <netid>`

## Show a specific job status:
`squeue -j <jobid>`

## Hold a job(set priority to 0):
`scontrol hold <jobid>`
Undo with `scontrol release <joid>`
