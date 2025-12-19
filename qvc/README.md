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


## Install eztaox for dev
delete the dependencies in dependencies array in eztaox/pyproject.toml

in eztaox folder, run

``pip install .``

you may need to install other packages with pip, make a note of them:

pip install <package>

## Upload data

``cd eztaox/tutorials/data ``

``curl -O http://quasar.astro.illinois.edu/paper_data/DR16Q/dr16q_prop_May01_2024.fits.gz gunzip dr16q_prop_May01_2024.fits.gz``

Also under tutorials/data, upload the parquet files from https://www.dropbox.com/scl/fo/7a6wz93al26xm8xzv2355/ANJTBMmTSwMywlr0wSO4esM?rlkey=66tshxe23ezyjiohcrpjkdne9&e=1&st=9od96bti&dl=0

## every time you want to work:

``module load miniconda``

``conda activate jaxcpu``


## Running

### Light Multiband fitting
#### Specifying objects per id

pip install -U tfp-nightly

``JAX_ENABLE_X64=True PREFIX=test SUFFIX=joint python multiband_fit.py --progress --plot --nwarm 50 --nsamp 20 --f_host_shen11 --bwb --filter_object_id 1406548 1412797``

#### Running a batch of N
``JAX_ENABLE_X64=True PREFIX=test SUFFIX=joint python multiband_fit.py  --progress --plot --filter_file data/aug4_sample_chisqg10_ebv005sn3.csv --nwarm 1000 --nsamp 500 --nchains 2 --max_tree_depth 6 --f_host_shen11 --bwb --job_N 20 --job_id 0
``

# PyQSOFit
```
def _L_conti(self, wave, pp, waves=np.array([1350, 3000, 5100])):
        """
        Calculate continuum Luminoisity at given waves
        """
        waves = np.array(waves)

        # Add these lines:
        minw = np.min([np.min(wave), 2300])
        maxw = np.max([np.max(wave), 2700])
        wave = np.linspace(minw, maxw, 2000)  # ensure the waves are within the range of the spectrum

        L = np.full(len(waves), -1.0)  # to save the luminosity results
        valid_idx = np.where((waves < np.max(wave)) & (waves > np.min(wave)), True, False)
        conti_flux = self.PL(waves[valid_idx], pp) + self.F_poly_conti(waves[valid_idx], pp[11:])
        Llam = waves[valid_idx] * self.flux2L(conti_flux, self.z)
        Llam[Llam <= 0] = 1e-1  # to make the log of these invalid values to be -1.
        L[valid_idx] = np.log10(Llam)

        return L
```

# Yale HPC Notes
## Show your jobs status:
`squeue -u <netid>`

## Show a specific job status:
`squeue -j <jobid>`

## Hold a job(set priority to 0):
`scontrol hold <jobid>`
Undo with `scontrol release <joid>`
