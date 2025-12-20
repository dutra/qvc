# README

## Installation guide
## Set up a conda CPU environment (suggested)
`conda create --name jaxcpu python=3.13.5 jax jaxopt jupyter jupyterlab numpyro tinygp matplotlib astropy jaxopt h5py tqdm ipykernel pandas corner scipy`
`conda activate jaxcpu`

### Installing eztaox
`git clone https://github.com/dutra/eztaox.git`

`git checkout preview`
`cd eztaox`
`pip install .`

### Installing PyQSOFit
In the qvc root directory:
`git clone https://github.com/dutra/PyQSOFit.git`
`cd PyQSOFit`
`pip install .`

## Download the data
Download the demo data from 
and extract it in the main qvc folder.


# PyQSOFit


# Light Multiband fitting
The Light multiband fitting can be run by specifying an Object ID.
Included in the demo data is the light curve for the object 1465126, which will reproduce Figure 1.

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
A subset of the Hubble Diagram fitting can be obtained in notebook `hubble_diagram_plots.ipynb`.

In order to reproduce all hubble diagram plots, the HD fitting procedure can be run with the command
```
PREFIX=dec6a_nov10a_oct12a_preview_fhost0_production_carma_spl_zsigma05_redchi212 \
python hubble_fit.py --resume\
    --cosmo_models FlatLambdaCDM FlatwCDM Flatw0waCDM \
    --run full \
    --speed production \
    --spectra_fit_csv "results/data/nov12a_11c_single_scratch_nov10a_carma_removemix_fixmeanband_no1pluszflux2L_freeiron_mc50_best.csv" \
    --sdss_mags_csv "results/data/nov2_sdss_mags.csv"  \
    --zquery_csv "results/data/sep19_chisq_zquery.csv"  \
    --z_range 0.44 3.16 \
    "results/data/nov10a_single_chisq_carma_mixscalar_nozband_highertaufastlim_removemix_fixband_lagblrband_chisq_spl_nofhost_bwb_lmc-6_N1w1000s200t14ch4.h5"
```
where the `--resume` flag will load the dynesty checkpoint.

Similarly as `multiband_fit.py`, `hubble_fit.py` will use the environment variable `PREFIX`.
All hubble diagram relevant plots, tables, and results will be generate under `plots/hubble/<prefix>`.

# Appendix figures
- Appendix Figure B3 can be reproduced in the notebook `appendix_band_vs_wavelength.ipynb`
- Appendix Figures C4, C5, D6, D7 can be reproduced in notebook `appendix_sigma_tau.ipynb`
- Appendix Figure E8 can be reproduced in notebook `appendix_lags.ipynb`

# Yale HPC Notes
## Show your jobs status:
`squeue -u <netid>`

## Show a specific job status:
`squeue -j <jobid>`

## Hold a job(set priority to 0):
`scontrol hold <jobid>`
Undo with `scontrol release <joid>`
