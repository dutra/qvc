import os
import multiprocessing

num_cores = os.environ.get("NUM_CORES", os.cpu_count()-2)
try:
    num_cores = int(num_cores)
except ValueError:
    print(f"Invalid NUM_CORES value '{num_cores}', ignoring.")
    num_cores = os.cpu_count()-2

if multiprocessing.current_process().name == "MainProcess":
    print(f"CPU Num Cores: {num_cores}")
# Make each XLA/Eigen CPU device single-threaded
os.environ["XLA_FLAGS"] = (
    f"--xla_force_host_platform_device_count={num_cores} --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=1"
)
# Avoid extra per-process threadpools from BLAS/OMP/NumExpr:
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.pop("NUMEXPR_MAX_THREADS", None)
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.environ["JAX_PLATFORM_NAME"] = "cpu"
prefix = os.environ.get('PREFIX', "test")
suffix = os.environ.get('SUFFIX', "test")

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
from jax import lax
import jax.numpy as jnp

import numpy as np
import math
import pandas as pd
from tqdm import tqdm
import numpyro
numpyro.set_host_device_count(num_cores)  # Tell NumPyro how many to use
numpyro.enable_x64()
numpyro.enable_validation(True)  # checks distribution params & support

from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import median_abs_deviation


from numpyro import infer
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer.reparam import LocScaleReparam

from tinygp import kernels

import warnings
#warnings.filterwarnings("ignore", category=RuntimeWarning)

import logging

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

from multiband_fit_utils import *
from multiband_fit_plotting import *
from multiband_generate_lc import *
from multiband_models import *
from multiband_model_lmc import MyMultiVarModel_BLR_LMC

# define params
zero_mean = False
has_jitter = True
has_lag = True

universal_params = (
    'eta_A1_mean','eta_A2_mean','eta_tau1_mean','eta_tau2_mean',
    'eta_break','lam_s',
    'sigma_eta_A1','sigma_eta_A2','sigma_eta_tau1','sigma_eta_tau2',
    'log_sigma_eta_A1','log_sigma_eta_A2','log_sigma_eta_tau1','log_sigma_eta_tau2',
    'mu_log_tau_rf','sigma_log_tau_rf','mu_log_sigma_hat0','sigma_log_sigma_hat0',
    'delta_eta_tau', 'mu_eta_tau',
    # LMC hypers
    'gate_log_temp', 'lmc_sep_raw', 'lmc_sep_left_raw', 'lmc_sep_right_raw', 'lmc_span_raw',
    'lmc_mu_raw', 'lmc_delta_raw', 'lmc_sep', 'lmc_sep_left', 'lmc_sep_right', 'lmc_span',
)

def inv_softplus(y):
    # numerically stable inverse softplus
    return jnp.where(y > 20.0, y, jnp.log(jnp.expm1(y)))

def build_model(batch_data, zs, lam_rfs, f_host_value, log_jitter_mean, log_tau_fake_in, log_sigma_fake_in, 
                bwb=True, disable_poly1=False, d_eta=True, disable_lag_blr=False, free_eta_break=False,
                couple_sigma_tau=False, sigma_tau_uniform=False, inject_fake=False, 
                lmc_q_groups=None, sample_lmc_hypers=False, eta_tau_normal=False):
    # Precompute and capture constants in the closure so they are treated as
    # static by JAX/NumPyro. This prevents unnecessary retracing/recompilation
    # when running MCMC, as these values do not change between runs.
    batch_size = len(batch_data)
    nBands = 5  # or use from config

    # Reference centers for object-level priors
    log_tau_drw0_c = jnp.log(10**2.5 * (1 + zs))
    log_lag_blr_c  = jnp.log(10**1.5 * (1 + zs))

    # use a non‑centered parameterization to avoid Neal’s funnel
    @handlers.reparam(config={
        "eta_A1": LocScaleReparam(centered=0.0),
        "eta_A2": LocScaleReparam(centered=0.0),
        "eta_tau1": LocScaleReparam(centered=0.0),
        "eta_tau2": LocScaleReparam(centered=0.0),
    })
    def numpyro_joint_model():

        # Initialize parameters
        # Global "universal" means for eta
        # eta_A1_mean = numpyro.sample("eta_A1_mean", dist.TruncatedNormal(-0.5, 0.4, high=0.0))
        # eta_A2_mean = numpyro.sample("eta_A2_mean", dist.TruncatedNormal(-0.5, 0.4, high=0.0))
        # eta_tau1_mean = numpyro.sample("eta_tau1_mean", dist.TruncatedNormal(-0.5, 0.4))
        # eta_tau2_mean = numpyro.sample("eta_tau2_mean", dist.TruncatedNormal(0.1, 0.4, low=0.0))

        # eta_A1_mean = numpyro.sample("eta_A1_mean", dist.Normal(-0.5, 0.4))
        # eta_A2_mean = numpyro.sample("eta_A2_mean", dist.Normal(-0.5, 0.4))
        # eta_tau1_mean = numpyro.sample("eta_tau1_mean", dist.Normal(-0.5, 0.4))
        # eta_tau2_mean = numpyro.sample("eta_tau2_mean", dist.Normal(0.1, 0.4))

        eta_A1_mean = numpyro.sample("eta_A1_mean", dist.Uniform(-1.0, 0.0))
        eta_A2_mean = numpyro.sample("eta_A2_mean", dist.Uniform(-1.0, 0.0))
        eta_tau1_mean = numpyro.sample("eta_tau1_mean", dist.Uniform(-1.0, 5.0))
        eta_tau2_mean = numpyro.sample("eta_tau2_mean", dist.Uniform(-1.0, 5.0))

<<<<<<< HEAD
        # Symmetric, order-agnostic priors for global tau-slopes
        #mu_eta_tau = numpyro.sample("mu_eta_tau", dist.Normal(0.5, 2.0))   # broad center near what you expect
        #delta_eta_tau = numpyro.sample("delta_eta_tau", dist.Normal(0.0, 2.0))  # symmetric around 0

        #eta_tau1_mean = numpyro.deterministic("eta_tau1_mean", mu_eta_tau + 0.5 * delta_eta_tau)
        #eta_tau2_mean = numpyro.deterministic("eta_tau2_mean", mu_eta_tau - 0.5 * delta_eta_tau)

=======
        eta_tau1_mean = numpyro.sample("eta_tau1_mean", dist.Uniform(-1.0, 5.0))
        eta_tau2_mean = numpyro.sample("eta_tau2_mean", dist.Uniform(-1.0, 5.0))
>>>>>>> 6e94c52750396bef1ecfca07b81f5f69c90caec4

        if free_eta_break:
            print("[INFO] Free eta_break and lam_s.")
            s = 0.4
            median = 0.1
            mu = jnp.log(median)
            sigma = jnp.sqrt(jnp.log((1 + jnp.sqrt(1 + 4*(s/median)**2)) / 2))
            eta_break = numpyro.sample("eta_break", dist.LogNormal(mu, sigma))
            lam_s = numpyro.sample("lam_s", dist.Normal(2500.0, 100.0)) # Hard to constrain
        else:
            eta_break = numpyro.deterministic("eta_break", 0.1)
            lam_s = numpyro.deterministic("lam_s", 2500.0)

        # Recommended defaults:
        # - separations are on a *raw* scale; effective sep = softplus(raw) + min_sep_ln (enforced in transform)
        # keep warm default as prior center
        gate_log_temp = numpyro.sample("gate_log_temp", dist.Normal(jnp.log(0.9), 0.35))

        if sample_lmc_hypers:

            if lmc_q_groups == 2:
                # sample a single separation DOF around your current target (~0.35)
                sep_soft_target_q2 = 0.35
                lmc_sep_raw = numpyro.sample("lmc_sep_raw", dist.Normal(inv_softplus(sep_soft_target_q2), 0.6))
                lmc_sep = jax.nn.softplus(lmc_sep_raw)

                numpyro.deterministic("lmc_sep", lmc_sep)
            elif lmc_q_groups == 3:
                sep_left_target  = 0.30   # in ln-days (post-softplus space)
                sep_right_target = 0.42

                raw_left0  = inv_softplus(sep_left_target)
                raw_right0 = inv_softplus(sep_right_target)
                mu_raw0    = 0.5 * (raw_left0 + raw_right0)
                delta0_raw = (raw_right0 - raw_left0)

                numpyro.deterministic("lmc_mu_raw", mu_raw0)

                # Sample ONLY the contrast, centered on the desired asymmetry
                delta_raw = numpyro.sample("lmc_delta_raw", dist.Normal(delta0_raw, 0.5))  # 0.4–0.6 is a good range

                lmc_sep_left_raw  = mu_raw0 - 0.5 * delta_raw
                lmc_sep_right_raw = mu_raw0 + 0.5 * delta_raw
                numpyro.deterministic("lmc_sep_left_raw",  lmc_sep_left_raw)
                numpyro.deterministic("lmc_sep_right_raw", lmc_sep_right_raw)

                # Monitor in separation space (post-softplus) for plots/debug
                lmc_sep_left  = jax.nn.softplus(lmc_sep_left_raw)
                lmc_sep_right = jax.nn.softplus(lmc_sep_right_raw)
                numpyro.deterministic("lmc_sep_left",  lmc_sep_left)
                numpyro.deterministic("lmc_sep_right", lmc_sep_right)        
            elif (lmc_q_groups is not None) and (lmc_q_groups > 3):
                lmc_span_raw = numpyro.sample("lmc_span_raw",
                                              dist.Normal(0.0, 1.0))
        else:
            # Deterministic “priors” (fixed values). These are in RF ln-days logic, but
            # only the raw values are set here; the transform applies min_sep_ln.

            # target RF separation factor ≈ 2.5 → Δ_target = ln(2.5) ≈ 0.916 ln-days
            # In the transform: sep = softplus(raw) + min_sep_ln, so we pick raw so that
            # softplus(raw) ≈ 0.7 (leaves room above min_sep to avoid hard edges).
            sep_soft_target = jnp.array(0.70)
            raw_from_soft = jnp.log(jnp.expm1(jnp.maximum(sep_soft_target, 1e-6)))

            if lmc_q_groups == 2:
                # OLD target was effectively ~0.7 (large, forces hard split)
                # NEW: smaller, overlaps clusters -> connected posterior
                sep_soft_target_q2 = 0.35
                lmc_sep_raw = inv_softplus(sep_soft_target_q2)
                lmc_sep = jax.nn.softplus(lmc_sep_raw)

                numpyro.deterministic("lmc_sep_raw", lmc_sep_raw)
                numpyro.deterministic("lmc_sep", lmc_sep)
            elif lmc_q_groups == 3:
                # Slight asymmetry avoids global left↔right flips
                sep_left_target  = 0.30
                sep_right_target = 0.42

                lmc_sep_left_raw  = inv_softplus(sep_left_target)
                lmc_sep_right_raw = inv_softplus(sep_right_target)

                lmc_sep_left  = jax.nn.softplus(lmc_sep_left_raw)
                lmc_sep_right = jax.nn.softplus(lmc_sep_right_raw)

                numpyro.deterministic("lmc_sep_left_raw",  lmc_sep_left_raw)
                numpyro.deterministic("lmc_sep_right_raw", lmc_sep_right_raw)
                numpyro.deterministic("lmc_sep_left",  lmc_sep_left)
                numpyro.deterministic("lmc_sep_right", lmc_sep_right)
            elif (lmc_q_groups is not None) and (lmc_q_groups > 3):
                # modest total span; transform will ensure ≥ (Q−1)*min_sep_ln anyway
                lmc_span_raw = numpyro.deterministic("lmc_span_raw", raw_from_soft)



        # Population-level scatter (how much objects can deviate) 
        log_sigma_eta_A1 = numpyro.sample("log_sigma_eta_A1", dist.Normal(jnp.log(0.1), 0.2))
        log_sigma_eta_A2 = numpyro.sample("log_sigma_eta_A2", dist.Normal(jnp.log(0.1), 0.2))
        log_sigma_eta_tau1 = numpyro.sample("log_sigma_eta_tau1", dist.Normal(jnp.log(0.1), 0.2))
        log_sigma_eta_tau2 = numpyro.sample("log_sigma_eta_tau2", dist.Normal(jnp.log(0.1), 0.2))

        sigma_eta_A1 = numpyro.deterministic("sigma_eta_A1", jnp.exp(log_sigma_eta_A1))
        sigma_eta_A2 = numpyro.deterministic("sigma_eta_A2", jnp.exp(log_sigma_eta_A2))
        sigma_eta_tau1 = numpyro.deterministic("sigma_eta_tau1", jnp.exp(log_sigma_eta_tau1))
        sigma_eta_tau2 = numpyro.deterministic("sigma_eta_tau2", jnp.exp(log_sigma_eta_tau2))

        with numpyro.plate("objects", batch_size):
            # Object-level parameters (shape: [B])

            # Fake
            log_tau_fake = numpyro.deterministic("log_tau_fake", log_tau_fake_in)
            log_sigma_fake = numpyro.deterministic("log_sigma_fake", log_sigma_fake_in)

            # Variability k-corrections
            if d_eta:
                eta_A1 = numpyro.sample("eta_A1", dist.Normal(eta_A1_mean, sigma_eta_A1))
                eta_A2 = numpyro.sample("eta_A2", dist.Normal(eta_A2_mean, sigma_eta_A2))
                eta_tau1 = numpyro.sample("eta_tau1", dist.Normal(eta_tau1_mean, sigma_eta_tau1))
                eta_tau2 = numpyro.sample("eta_tau2", dist.Normal(eta_tau2_mean, sigma_eta_tau2))
            # Or, use deterministic to set them to the universal means
            else:
                eta_A1 = numpyro.deterministic("eta_A1", jnp.full(batch_size, eta_A1_mean))
                eta_A2 = numpyro.deterministic("eta_A2", jnp.full(batch_size, eta_A2_mean))
                eta_tau1 = numpyro.deterministic("eta_tau1", jnp.full(batch_size, eta_tau1_mean))
                eta_tau2 = numpyro.deterministic("eta_tau2", jnp.full(batch_size, eta_tau2_mean))

            # --- Core kernel parameters (hierarchical & identified) ---
            #log_tau_drw0 = numpyro.sample("log_tau_drw0", dist.TruncatedNormal(log_tau_drw0_c, 2.0, low=jnp.log(10**1.5)))
            # log_tau_drw0 = numpyro.sample("log_tau_drw0", dist.TruncatedNormal(log_tau_drw0_c, 1.5, low=jnp.log(10**1.5)))
            # log_sigma0 = numpyro.sample("log_sigma0", dist.Normal(-0.8, 1.0))
            # log_sigma_hat0 = numpyro.deterministic("log_sigma_hat0", log_sigma0 - 0.5 * log_tau_drw0)

            log_tau_drw0_high = 10.0 * jnp.log(10)
            if inject_fake:
                log_tau_drw0_low = 0.0
            else:
                log_tau_drw0_low = 1.5 * jnp.log(10)

            if sigma_tau_uniform:
                print("[INFO] Using Uniform prior on log_sigma0 and log_tau_drw0.")
                log_tau_drw0 = numpyro.sample("log_tau_drw0", dist.Uniform(log_tau_drw0_low, log_tau_drw0_high))
            else:
                print("[INFO] Using Normal prior on log_sigma0 and log_tau_drw0.")
                log_tau_drw0 = numpyro.sample("log_tau_drw0",
                    dist.TruncatedNormal(log_tau_drw0_c, 1.2*jnp.log(10), low=log_tau_drw0_low, high=log_tau_drw0_high))
            
            if couple_sigma_tau:
                # Coupled prior: log_sigma depends on log_tau
                # Put prior on standardized amplitude; derive log_sigma0 from it
                print("[WARNING] couple_sigma_tau=True: log_sigma0 is coupled to log_tau_drw0 via log_sigma_hat0 prior.")
                if sigma_tau_uniform:
                    log_sigma_hat0 = numpyro.sample("log_sigma_hat0", dist.Uniform(-5*jnp.log(10), -0.5*jnp.log(10)))
                else:
                    log_sigma_hat0 = numpyro.sample("log_sigma_hat0", dist.Normal(-0.6*jnp.log(10) - 0.5*log_tau_drw0_c, 2.0*jnp.log(10)))
                log_sigma0 = numpyro.deterministic("log_sigma0", log_sigma_hat0 + 0.5 * log_tau_drw0)
            else:
                # Uncoupled prior: log_sigma independent of log_tau
                print("[WARNING] couple_sigma_tau=False: log_sigma0 is independent of log_tau_drw0.")
                if sigma_tau_uniform:
                    log_sigma0 = numpyro.sample("log_sigma0", dist.Uniform(-2.0*jnp.log(10), 0.2*jnp.log(10)))
                else:
                    log_sigma0 = numpyro.sample("log_sigma0", dist.Normal(-0.6*jnp.log(10), 1.0*jnp.log(10)))
                log_sigma_hat0 = numpyro.deterministic("log_sigma_hat0", log_sigma0 - 0.5 * log_tau_drw0)

            # Host galaxy dilution
            alpha_host = numpyro.sample("alpha_host", dist.Normal(1.0, 0.1)) # alpha_lam
            alpha_agn = numpyro.sample("alpha_agn", dist.Normal(-1.5, 0.3)) # alpha_lam
            f_host = numpyro.deterministic("f_host", f_host_value)

            # Mean function detrending
            if disable_poly1:
                poly1 = numpyro.deterministic("poly1", jnp.zeros_like(log_sigma0))                
            else:
                poly1 = numpyro.sample("poly1", dist.Normal(0.0, 0.1))

            # Disk lags
            #lag0 = numpyro.sample("lag0", dist.TruncatedNormal(10.0, 5.0, low=0))
            #lag_beta = numpyro.sample("lag_beta", dist.TruncatedNormal(4/3, 0.2, low=0))

            log_lag0 = numpyro.sample(
                "log_lag0",
                dist.TruncatedNormal(jnp.log(0.2) + log_tau_drw0, 1.0,
                                    low=jnp.log(0.03), high=jnp.log(4_000.0))
            )
            lag0_tilde = numpyro.deterministic("lag0_tilde", jnp.exp(log_lag0))
            lag_beta = numpyro.sample("lag_beta", dist.Normal(4/3, 0.2))

            # Bluer when brighter (BWB) strength
            if bwb:
                bwb_log_alpha = numpyro.sample("bwb_log_alpha", dist.Normal(0.5, 0.2))
                bwb_log_beta = numpyro.sample("bwb_log_beta", dist.Normal(4.0, 1.0))
                bwb_alpha = numpyro.deterministic("bwb_alpha", jnp.exp(bwb_log_alpha))
                bwb_beta = numpyro.deterministic("bwb_beta", jnp.exp(bwb_log_beta))
            else:
                bwb_alpha = numpyro.deterministic("bwb_alpha", jnp.zeros(batch_size))
                bwb_beta = numpyro.deterministic("bwb_beta", jnp.ones(batch_size))

        with numpyro.plate("objects", batch_size, dim=-2):
            with numpyro.plate("band", nBands, dim=-1):
                # Parameters with shape [B, nBands]
                # Means in each band
                mean = numpyro.sample("mean", dist.Normal(jnp.full(nBands, 0.0), 0.2))

                # BLR amplitudes and lags
                if disable_lag_blr:
                    print("[WARNING] BLR lag model disabled.")
                    log_amp_delta_blr = numpyro.deterministic("log_amp_delta_blr", jnp.full((batch_size, nBands), -1e9))
                    log_lag_blr = numpyro.deterministic("log_lag_blr", jnp.full((batch_size, nBands), -9.0))
                else:
                    print("[WARNING] BLR lag model enabled.")
                    log_amp_delta_blr = numpyro.sample("log_amp_delta_blr", dist.Normal(jnp.full(nBands, -1.0), 3.0))
                    log_lag_blr = numpyro.sample(
                        "log_lag_blr",
                        dist.Uniform(jnp.log(0.2), jnp.log(5000.0))
                    )

                width_blr = numpyro.deterministic(
                    "width_blr",
                    0.2 * jnp.exp(log_tau_drw0_c)[:, None] * jnp.ones((batch_size, nBands))
                )
                width_cont = numpyro.deterministic(
                    "width_cont",
                    0.2 * jnp.exp(log_tau_drw0_c)[:, None] * jnp.ones((batch_size, nBands))
                )

                # Jitter
                log_jitter = numpyro.sample("log_jitter", dist.Normal(log_jitter_mean, 1.0))
                        

        def run_batch(obj, i):

            # t = obj[:, 0]
            # b = obj[:, 1].astype(t.dtype)
            # tie_eps = 10.0 * jnp.finfo(t.dtype).eps  # same magnitude used in the kernel
            # key = t + b * tie_eps
            # sort_idx   = jnp.argsort(key)
            # obj_sorted = obj[sort_idx]
            obj_sorted = obj  # data already sorted

            # Collect params for object i
            params = {
                "log_tau_drw0": log_tau_drw0[i],
                "log_sigma0": log_sigma0[i],
                "alpha_host": alpha_host[i],
                "alpha_agn": alpha_agn[i],
                "f_host": f_host[i],
                "poly1": poly1[i],
                "mean": mean[i],
                "log_amp_delta_blr": log_amp_delta_blr[i],
                "log_lag_blr": log_lag_blr[i],
                "log_jitter": log_jitter[i],
                #"lag0": lag0[i],
                "lag0_tilde": lag0_tilde[i],
                "lag_beta": lag_beta[i],
                "bwb_alpha": bwb_alpha[i],
                "bwb_beta": bwb_beta[i],
                "width_blr": width_blr[i],
                "width_cont": width_cont[i],
                # Fake
                "log_tau_fake": log_tau_fake[i],
                "log_sigma_fake": log_sigma_fake[i],
                # power law
                "eta_A1": eta_A1[i],
                "eta_A2": eta_A2[i],
                "eta_tau1": eta_tau1[i],
                "eta_tau2": eta_tau2[i],
                "eta_break": eta_break,
                "lam_s": lam_s,

                # ---- LMC hypers passed to the Model (used in my_tau_drw_transform) ----
                "gate_log_temp": gate_log_temp,
                **({"lmc_sep_raw": lmc_sep_raw} if lmc_q_groups == 2 else {}),
                **({"lmc_sep_left_raw":  lmc_sep_left_raw,
                    "lmc_sep_right_raw": lmc_sep_right_raw} if lmc_q_groups == 3 else {}),
                **({"lmc_span_raw": lmc_span_raw} if (lmc_q_groups is not None and lmc_q_groups > 3) else {}),

            }

            m = Model(
                X=(obj_sorted[:, 0], obj_sorted[:, 1]),
                y=obj_sorted[:, 2],
                yerr=obj_sorted[:, 3],
                kernel=kernels.quasisep.Exp(jnp.array([1, 1])),
                zero_mean=zero_mean, has_jitter=has_jitter, has_lag=has_lag,
                lam_rf=lam_rfs[i], z=zs[i], q_groups=lmc_q_groups,
                use_bwb=bwb
            )
            return m.log_prob(params)

        log_probs = jax.vmap(run_batch, in_axes=(0, 0))(batch_data, jnp.arange(batch_size))
        nan_mask = jnp.isnan(log_probs)
        idx_all = jnp.arange(batch_size)               # static shape
        idx_padded = jnp.where(nan_mask, idx_all, -1)  # -1 where not NaN
        lax.cond(
            jnp.any(nan_mask),
            lambda _: jax.debug.print(
                "NaNs at indices (=-1 means none): {}\nvalues: {}",
                idx_padded, jnp.where(nan_mask, log_probs, 0.0)
            ),
            lambda _: None,
            operand=None
        )
        # if jnp.any(jnp.isnan(log_probs)):
        #     jax.debug.print("Warning: NaN detected in log_probs:", log_probs)
        #log_probs = jnp.where(jnp.isfinite(log_probs), log_probs, -1e20)
        numpyro.factor("loglike", log_probs.sum())
    
    return numpyro_joint_model


def make_lc(Model, data, bands=['u', 'g', 'r', 'i', 'z'], inject_fake=False):
    times = data['times']
    mags = data['mags']
    magerrs = data['magerrs']

    dropped_bands = bands_bluer_than_lyman_alpha(data['z'])
    data['dropped_bands'] = dropped_bands

    logging.info(
        f"Excluding only blue bands {dropped_bands} "
        f"for quasar {data['object_id']} at z={data['z']}"
    )

    # Sort bands
    bands = list(sorted(bands, key=lambda b: ['u', 'g', 'r', 'i', 'z', 'y'].index(b)))

    if len(bands) == 0:
        print(f"No bands for quasar {data['object_id']}, skipping.", flush=True)
        return None

    # Combine data across bands
    all_times = np.concatenate([times[b] for b in bands])
    all_mags = np.concatenate([mags[b] for b in bands])
    all_magerrs = np.concatenate([magerrs[b] for b in bands])
    band_idx = np.concatenate([np.full(len(times[b]), i) for i, b in enumerate(bands)])
    band_idx = band_idx.astype(np.int64, copy=False)
    
    if len(all_times) == 0:
        print(f"No magnitudes for quasar {data['object_id']}, skipping.", flush=True)
        return None

    # --- stable pre-sort by the kernel's coord_to_sortable: t + eps * band ---
    band_idx = band_idx.astype(np.int64, copy=False)
    tie_eps  = 10.0 * np.finfo(all_times.dtype).eps
    key      = all_times + band_idx.astype(all_times.dtype) * tie_eps
    order    = np.argsort(key, kind="mergesort")   # stable

    all_times   = all_times[order]
    all_mags    = all_mags[order]
    all_magerrs = all_magerrs[order]
    band_idx    = band_idx[order]

    def exp_filter_to_tau(x, t, tau):
        """
        First-order exponential smoother that 'slows' the shared latent x(t)
        toward an effective time constant ~ tau (days).
        y[i] = (1 - a_i) * y[i-1] + a_i * x[i],  a_i = 1 - exp(-Δt_i / tau)
        """
        if x.size == 0:
            return x
        y = np.empty_like(x, dtype=float)
        y[0] = x[0]
        dt = np.diff(t)
        # clip to avoid under/overflow
        a = 1.0 - np.exp(-np.clip(dt / max(tau, 1e-9), 0.0, 1e6))
        for i in range(1, x.size):
            y[i] = (1.0 - a[i-1]) * y[i-1] + a[i-1] * x[i]
        return y

    # Inject fake DRW
    if inject_fake:
        alpha_sigma = -0.5  # σ(λ) ∝ λ^α
        beta_tau = 0.0      # τ(λ) ∝ λ^β

        # ---- FIX 1: build per-band arrays (B,), not per-observation ----
        lam_rf_bands = np.asarray([lambda_pivot[band] for band in bands], dtype=float) / (1.0 + float(data['z']))
        lam_ref = 2500.0  # Å

        # deterministic seed per object
        key = jax.random.PRNGKey(0)
        key = jax.random.fold_in(key, int(data['object_id']))
        key, k_tau0, k_sig0, k_latent, k_noise = jax.random.split(key, 5)

        # Base logs
        log_tau0_rf = jax.random.uniform(k_tau0, minval=0.5,  maxval=3.0)   # log10 tau_rest (d)
        log_sigma0  = jax.random.uniform(k_sig0, minval=-1.0, maxval=0.0)   # log10 sigma (mag)
        tau0_rf   = 10.0**float(log_tau0_rf)
        sigma0    = 10.0**float(log_sigma0)
        one_plus_z = float(1.0 + data['z'])

        # Per-band target τ, σ from wavelength laws (rest-frame → observed τ)
        tau_rf_band  = tau0_rf * (lam_rf_bands / lam_ref)**beta_tau        # (B,)
        tau_obs_band = tau_rf_band * one_plus_z                            # (B,)
        sigma_band   = sigma0   * (lam_rf_bands / lam_ref)**alpha_sigma    # (B,)

        # Choose a latent τ to drive everyone (e.g., geometric mean of band τ)
        tau_latent_obs = float(np.exp(np.mean(np.log(np.clip(tau_obs_band, 1e-6, None)))))
        sigma_latent   = 1.0  # unit scale; bands will rescale

        print(
            f"Injecting SHARED latent for object {data['object_id']}: "
            f"log_tau0_rf={float(log_tau0_rf):.3f}, log_sigma0={float(log_sigma0):.3f}, "
            f"tau_latent_obs≈{tau_latent_obs:.3g} d"
        )

        # Work on a time-sorted view so filtering is causal
        order = np.argsort(all_times)
        times_sorted     = all_times[order]
        mags_err_sorted  = all_magerrs[order]
        bands_sorted     = band_idx[order]  # (N,) int indices into `bands`

        # Sample ONE latent DRW on all timestamps (no measurement noise here)
        latent = sample_drw_tinygp(
            k_latent,
            times_sorted,
            tau=tau_latent_obs,
            sigma=sigma_latent,
            noise=1.0e-6,     # keep the latent clean; add obs noise per band later
            mean=0.0
        )[0]
        latent = np.array(latent)

        # Allocate output
        mags_sorted = np.empty_like(times_sorted, dtype=float)

        # Split keys for band-wise noise
        uniq_bands = np.unique(np.asarray(bands_sorted))
        noise_keys = jax.random.split(k_noise, len(uniq_bands))

        for bk, b in zip(noise_keys, uniq_bands):
            b = int(b)  # ---- FIX 2: use band index to index (B,) arrays ----
            m = (bands_sorted == b)
            t_b = np.asarray(times_sorted[m], dtype=float)
            x_b = np.asarray(latent[m],       dtype=float)

            # Filter the latent to impose target τ for this band
            y_b = exp_filter_to_tau(x_b, t_b, tau=float(tau_obs_band[b]))

            # Standardize per-band and scale to target σ(λ)
            y_b = y_b - np.mean(y_b)
            std = np.std(y_b)
            std = std if std > 1e-12 else 1.0
            y_b = (y_b / std) * float(sigma_band[b])

            # Add observational noise
            eps = jax.random.normal(bk, shape=(y_b.size,))
            y_b = y_b + np.array(eps) * mags_err_sorted[m]

            mags_sorted[m] = y_b

            print(
                f"  band {b}: λ_rf={lam_rf_bands[b]:.0f}Å, "
                f"τ_rf={tau_rf_band[b]:.3g} d, τ_obs={tau_obs_band[b]:.3g} d, "
                f"σ={sigma_band[b]:.3g}"
            )

        # Undo sorting
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        all_mags = mags_sorted[inv]
        ###################################################

    # Remove NaNs
    mask = np.isfinite(all_mags) & np.isfinite(all_magerrs) & np.isfinite(all_times)
    all_times, all_mags, all_magerrs, band_idx = (
        all_times[mask],
        all_mags[mask],
        all_magerrs[mask],
        band_idx[mask]
    )

    if len(all_times) == 0:
        print(f"No finite magnitudes for quasar {data['object_id']}, skipping.", flush=True)
        return None

    # --- Outlier rejection ---
    window_size = 6
    mask_outlier = np.ones(len(all_times), dtype=bool)

    for band in np.unique(band_idx):
        band_mask = band_idx == band
        idx_band = np.where(band_mask)[0]
        band_y = all_mags[band_mask]

        if len(band_y) < 2 * window_size + 1:
            continue

        windows = sliding_window_view(band_y, 2 * window_size + 1)
        centers = band_y[window_size:-window_size]
        medians = np.nanmean(windows, axis=1)
        mads = median_abs_deviation(windows, axis=1, nan_policy="omit")

        is_outlier = np.abs(centers - medians) > 2.5 * mads
        mask_outlier[idx_band[window_size:-window_size][is_outlier]] = False

    # Apply outlier mask
    all_times = all_times[mask_outlier]
    all_mags = all_mags[mask_outlier]
    all_magerrs = all_magerrs[mask_outlier]
    band_idx = band_idx[mask_outlier]

    # Calculate time length in observed frame and rest frame
    if len(all_times) > 0:
        t_obs_length = np.max(all_times) - np.min(all_times)
        t_rf_length = t_obs_length / (1.0 + data['z'])
    else:
        t_obs_length = 0.0
        t_rf_length = 0.0
    print(f"Observed-frame length: {t_obs_length:.2f} days, Rest-frame length: {t_rf_length:.2f} days")

    # --- Center magnitudes per band AFTER outlier rejection ---
    mags_means = np.array([
        np.nanmean(all_mags[band_idx == i]) for i in range(len(bands))
    ])
    mags_stds = np.array([
        np.nanstd(all_mags[band_idx == i]) for i in range(len(bands))
    ])
    print("Band stats (μ±σ):", ",".join(
        f"{band}:{mags_means[i]:.1f}±{mags_stds[i]:.2f}"
        for i, band in enumerate(bands)
    ))

    for i in range(len(bands)):
        band_mask = band_idx == i
        all_mags[band_mask] -= np.nanmean(all_mags[band_mask])
        # Mask dropped bands
        if bands[i] in dropped_bands:
            all_magerrs[band_mask] = 999.0

    # Define arrays for model
    X = (
        jnp.array(all_times) - jnp.min(all_times),
        jnp.array(band_idx)
    )
    y = jnp.array(all_mags)
    yerr = jnp.array(all_magerrs)
    t = jnp.array(all_times)

    batch_dict = {
        'X': X,
        'y': y,
        'yerr': yerr,
        'z': data['z'],
        'band_idx': band_idx,
        'mags_means': mags_means,
        'mags_stds': mags_stds,
        'log_tau_fake': -99.0,
        'log_sigma_fake': -99.0,
        't_obs_length': t_obs_length,
        't_rf_length': t_rf_length,
    }
    if inject_fake:
        batch_dict['log_tau_fake'] = np.log(10**log_tau0_rf)
        batch_dict['log_sigma_fake'] = np.log(10**log_sigma0)
        batch_dict['alpha_sigma'] = alpha_sigma
        batch_dict['beta_tau'] = beta_tau
    return batch_dict

                    
if __name__ == '__main__': 
    
    logging.info("Starting multiband fit")

    parser = argparse.ArgumentParser(description="Process quasars with optional filtering.")
    parser.add_argument("--filter_object_id", nargs="+", help="List of object IDs to filter.")
    parser.add_argument("--N", type=int, help="Number of objects to process.")
    parser.add_argument("--skip", type=int, help="Number of objects to skip.")
    parser.add_argument("--chunk_size", type=int, default=500, help="Chunk size for processing objects.")
    parser.add_argument("--filter_file", type=str, help="Path to the file containing object IDs to filter.")
    parser.add_argument("--plot", action="store_true", help="Enable plotting of results.")
    parser.add_argument("--ignore_existing", action="store_true", help="Ignore sources already in the HDF5 file.")
    parser.add_argument("--create_lc", action="store_true", help="Only create LC file and exit.")
    parser.add_argument("--progress", action="store_true", help="Show progress bar.")
    parser.add_argument("--nwarm", type=int, default=500, help="Number of warmup steps for MCMC.")
    parser.add_argument("--nsamp", type=int, default=250, help="Number of samples for MCMC.")
    parser.add_argument("--nchains", type=int, default=-1, help="Number of chains for MCMC.")
    parser.add_argument("--inject_fake", action="store_true", help="Use randomly sampled light curves with no correlation.")
    parser.add_argument("--bwb", action="store_true", help="Use BWB model.")
    parser.add_argument("--d_eta", action="store_true", help="Vary eta for each quasar with prior.")
    parser.add_argument("--max_tree_depth", type=int, default=8, help="Max tree depth param for NUTS sampler.")
    parser.add_argument("--load_sample_file", action="store_true", help="Load samples from previously ran job.")
    parser.add_argument("--disable_poly1", action="store_true", help="Disable Mean function detrending.")
    parser.add_argument("--jax_trace", action="store_true", help="Enable jax tracing.")
    parser.add_argument("--rf_length_cut", type=int, default=-1, help="Cut light curves to same rest-frame length.")
    parser.add_argument('--exact_same_length', action='store_true', help="Cut light curves to exact same rest-frame length.")
    parser.add_argument("--alpha_lam_csv", type=str, default=None, help="Path to CSV file containing alpha_lam values per object.")
    parser.add_argument("--load_stone_lcs", action="store_true", default=False, help="Load Stone light curves instead of default.")
    parser.add_argument("--free_eta_break", action="store_true", default=False, help="Allow eta_break to be a free parameter.")
    parser.add_argument("--disable_corner_plot", action="store_true", default=False, help="Disable corner plot generation.")
    parser.add_argument("--couple_sigma_tau", action="store_true", default=False, help="Use coupled prior for sigma and tau.")
    parser.add_argument("--disable_lag_blr", action="store_true", default=False, help="Disable BLR lag model.")
    parser.add_argument("--sigma_tau_uniform", action="store_true", default=False, help="Use uniform priors for sigma and tau.")
    parser.add_argument("--lmc", type=int, default=0, choices=[0, 1, 2, 3], help="Number of LMC Q groups (0 disables LMC, 1/2/3 controls Q).")
    parser.add_argument("--sample_lmc_hypers", action="store_true", default=False, help="Sample LMC hyperparameters instead of using fixed values.")
    parser.add_argument("--disable_plot_psd", action="store_true", default=False, help="Disable PSD plot generation.")
    parser.add_argument("--eta_tau_normal", action="store_true", default=False, help="Use uniform prior for eta_tau1 and eta_tau2.")
    args = parser.parse_args()
    print("Args: ", args)

    if args.filter_object_id is not None and len(args.filter_object_id) > 0:
        print(f"Filtering object IDs: {len(args.filter_object_id)}")

    if args.load_stone_lcs:
        objs = load_stone_lcs(filter_object_ids=args.filter_object_id)
        print(f"Loaded {len(objs)} Stone light curves.")
    else:
        objs = concat_light_curves(filter_object_ids=args.filter_object_id, progress_bar=args.progress, N=args.N, skip=args.skip)
    print(f"Loaded {len(objs)} objects from concat_light_curves")

    objs = populate_sdss_fields(objs, progress_bar=args.progress)
    
    if args.rf_length_cut > 0:
        objs = cut_light_curve_restframe_window(objs, n_days=args.rf_length_cut, same_length=args.exact_same_length)
        print(f"After restframe cut, {len(objs)} objects remain.")
    
    # # --- Precompute f_host_shen11 prior means ---
    # if args.f_host_shen11:
    #     # Host flux empirical relation
    #     #logl5100 = jnp.array([obj['LOGL5100'] for obj in batch_data])
    #     logl5100 = jnp.array([obj['LOGLBOL'] - jnp.log10(9.26) for obj in batch_data])

    #     x = logl5100 - 44.0
    #     f_host = 0.8052 - 1.5502 * x + 0.9121 * jnp.power(x, 2) - 0.1577 * jnp.power(x, 3)
    #     f_host = jnp.clip(f_host, 0.0, None)
    #     f_host_value = jnp.where(logl5100 < 45.053, f_host, 0.0)
    # else:
    #     batch_size = len(batch_data)
    #     f_host_value = jnp.zeros(batch_size)

    if args.alpha_lam_csv is not None:
        # Load CSV with columns: object_id, alpha_lambda, f_host_5100
        alpha_df = pd.read_csv(args.alpha_lam_csv, dtype={"object_id": str})
        alpha_map = alpha_df.set_index("object_id")[["alpha_lambda", "f_host_5100"]].to_dict(orient="index")
        # Populate objs with alpha_lambda and f_host_5100 by object_id
        for obj in objs:
            oid = str(obj["object_id"])
            if oid in alpha_map and alpha_map[oid]["f_host_5100"] >= 0:
                obj["f_host_5100"] = alpha_map[oid]["f_host_5100"]
            else:
                obj["f_host_5100"] = 0.0  # Default if not found or invalid
    else:
        print("[WARNING] Not using alpha_lam_csv, setting f_host_5100=0.0 for all objects.")
        for obj in objs:
            obj["f_host_5100"] = 0.0 

    for obj in objs:
        print(f"Object {obj['object_id']}: f_host_5100 = {obj['f_host_5100']}")

    #objs = populate_sdss_fields(objs)
    if args.lmc > 0:
        print(f"\033[93m[WARNING] Using LMC model (Q={args.lmc}) instead of DRW.\033[0m")
        Model = MyMultiVarModel_BLR_LMC
    else:
        Model = MyMultiVarModel

    # After loading objs
    logging.info("--- Joint fitting ---")
    batch_data = []
    if args.load_stone_lcs:
        bands = ['g', 'r', 'i']
    else:
        bands = ['u', 'g', 'r', 'i', 'z']
    for i, obj in enumerate(objs):
        # Prepare each object's data for the joint model
        result = make_lc(Model, obj, bands=bands, inject_fake=args.inject_fake)
        if result is None:
            continue
        obj['i'] = i
        obj |= result
        # Run bestP for each object
        lam_rf = jnp.array([lambda_pivot[band] for band in bands]) / (1 + obj['z'])

        batch_data.append({
            'object_id': obj['object_id'],
            'X': obj['X'],
            'y': obj['y'],
            'yerr': obj['yerr'],
            'band_idx': obj['band_idx'],
            'z': obj['z'],
            # add any other fields needed by your model
            'LOGLBOL': obj['LOGLBOL'],
            'mags_means': obj['mags_means'],
            'mags_stds': obj['mags_stds'],
            'lam_rf': lam_rf,
            'f_host_5100': obj['f_host_5100'],
            'log_tau_fake': obj['log_tau_fake'],
            'log_sigma_fake': obj['log_sigma_fake'],
            'alpha_sigma': obj.get('alpha_sigma', None),
            'beta_tau': obj.get('beta_tau', None),
            'dropped_bands': obj['dropped_bands'],
            't_obs_length': obj['t_obs_length'],
            't_rf_length': obj['t_rf_length'],
        })

    num_objects = len(batch_data)
    logging.info(f"Running joint fit on {len(batch_data)} objects...")


    padded_batch_data = pad_batch(batch_data, nBands=5)
    # Re-sort *after* padding by the same key the kernel uses
    padded_batch_data = [
        jnp.array(resort_by_kernel_key(np.asarray(obj)))
        for obj in padded_batch_data
    ]
    # NEW: stack into shape (B, Nmax, 4) so vmap sees a batch dimension
    batch_array = jnp.stack(padded_batch_data, axis=0)
    
    #init_strategy = numpyro.infer.init_to_sample()
    init_strategy = numpyro.infer.init_to_median()
    logging.info("Done with numpyro.infer.init_to_median")

    # --- Precompute log_jitter prior means ---
    zs = jnp.array([obj['z'] for obj in batch_data])
    lam_rfs = jnp.array([obj['lam_rf'] for obj in batch_data])

    log_jitter_mean = jnp.stack([
        jnp.array(jnp.full(5, 1e-6) + jnp.log(jnp.mean(jnp.array(obj[:,3][obj[:,3] < 10])))) for obj in padded_batch_data
    ])  # shape (B, nBands)
    # log_jitter_mean = jnp.stack([safe_log_jitter_mean(obj) for obj in padded_batch_data])
    assert jnp.isfinite(log_jitter_mean).all(), "Non-finite log_jitter_mean"

    f_host_value = jnp.array([obj["f_host_5100"] for obj in batch_data])

    log_tau_fake = jnp.array([obj['log_tau_fake'] for obj in batch_data])
    log_sigma_fake = jnp.array([obj['log_sigma_fake'] for obj in batch_data])



    numpyro_joint_model = build_model(batch_array, zs, lam_rfs, f_host_value, log_jitter_mean, log_tau_fake, log_sigma_fake, 
                                      bwb=args.bwb, disable_poly1=args.disable_poly1, d_eta=args.d_eta,
                                      disable_lag_blr=args.disable_lag_blr, 
                                      free_eta_break=args.free_eta_break,
                                      couple_sigma_tau=args.couple_sigma_tau, sigma_tau_uniform=args.sigma_tau_uniform,
                                      inject_fake=args.inject_fake, lmc_q_groups=args.lmc, sample_lmc_hypers=args.sample_lmc_hypers,
                                      eta_tau_normal=args.eta_tau_normal)

    nuts_kernel = NUTS(numpyro_joint_model, init_strategy=init_strategy, dense_mass=True, 
                       max_tree_depth=args.max_tree_depth,
                       target_accept_prob=0.9)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=args.nwarm,
        num_samples=args.nsamp,
        num_chains=args.nchains,
        progress_bar=args.progress,
        chain_method="parallel",
    )

    if args.load_sample_file:
        logging.warning(f"Loading samples from saved file")
        samples_flat = load_all_samples_from_hdf5()
    else: # run MCMC sampler
        if args.jax_trace:
            from jax.profiler import StepTraceAnnotation, start_trace, stop_trace
            # 1) Tiny compile pass OFF-trace (same chain_method/nchains to match shapes)
            compile_mcmc = MCMC(
                NUTS(numpyro_joint_model, init_strategy=init_strategy,
                    dense_mass=True, max_tree_depth=args.max_tree_depth),
                num_warmup=5,
                num_samples=1,
                num_chains=1,
                progress_bar=False,
                chain_method="vectorized",
            )
            compile_mcmc.run(jax.random.PRNGKey(0))  # triggers JIT compile

            # 2) Real run ON-trace
            trace_dir = getattr(args, "jax_trace_dir", "./jax_tb")
            start_trace(trace_dir)
            with StepTraceAnnotation("MCMC_run"):
                mcmc.run(jax.random.PRNGKey(1))
            stop_trace()
        else:
            # Plain run, no tracing
            mcmc.run(jax.random.PRNGKey(0))
        samples_flat = mcmc.get_samples(group_by_chain=False)
        samples_per_chain = mcmc.get_samples(group_by_chain=True)
        save_all_samples_to_hdf5(samples_flat)
    
    #ns = NestedSampler(numpyro_joint_model)
    #ns.run(jax.random.PRNGKey(0), Model, batch_data)
    #samples_flat = ns.get_samples(jax.random.PRNGKey(0), num_samples=1000)

    logging.info("Done with MCMC run")

    # Save and plot the results
    results = []
    for i, obj in enumerate(batch_data):
        logging.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        logging.info(f"Quasar {i+1}/{len(batch_data)} Object ID: {obj['object_id']}")

        obj_flat_samples = select_samples_for_object(samples_flat, i, universal_params=universal_params)
        obj_flat_samples_flatten_per_band = flatten_flat_samples_per_band(obj_flat_samples)
        save_obj_samples_to_hdf5(obj_flat_samples_flatten_per_band, obj['object_id'])

        obj_samples_per_chain = select_samples_for_object_per_chain(samples_per_chain, i, universal_params=universal_params)
        obj_samples_per_chain_flatten_per_band = flatten_per_chain_samples_per_band(obj_samples_per_chain)
        diagnostics = diagnostics_for_per_chain_samples(obj_samples_per_chain_flatten_per_band)
        
        # Add the object-specific parameters
        result = process_samples(obj_flat_samples_flatten_per_band, obj)

        # Plotting
        if args.plot:
            plot_mcmc_traces(obj_flat_samples_flatten_per_band, obj)
            m = Model(
                obj['X'], obj['y'], obj['yerr'], 
                kernels.quasisep.Exp(jnp.array([1, 1])),
                zero_mean=zero_mean, has_jitter=has_jitter, has_lag=has_lag,
                lam_rf=obj['lam_rf'], z=obj['z'], use_bwb=args.bwb, q_groups=args.lmc
            )
            save_combined_plot(obj_flat_samples, m, obj['X'], obj['y'], obj['yerr'], obj['band_idx'], result, 
                               bands=bands, plot_psd=(not args.disable_plot_psd))
            plot_correlation_matrix(obj_flat_samples_flatten_per_band, obj)
            plot_all_histograms(obj_flat_samples_flatten_per_band, obj)
            if not args.disable_corner_plot:
                #plot_posterior(obj_flat_samples_flatten_per_band, obj)
                plot_posterior_fast(obj_flat_samples_flatten_per_band, obj)
            plot_broken_power_law(obj_flat_samples, obj)
            #dump_mcmc_diagnostics(mcmc, obj, i, len(batch_data))
        # If inject_fake, compare injected vs recovered sigma and tau
        if args.inject_fake:
            injected_log_tau = obj['log_tau_fake']
            injected_log_sigma = obj['log_sigma_fake']
            # Use median of posterior for recovered values
            recovered_log_tau = np.median(obj_flat_samples_flatten_per_band['log_tau_drw0'])
            recovered_log_sigma = np.median(obj_flat_samples_flatten_per_band['log_sigma0'])

            tau_p16, tau_p84 = np.percentile(obj_flat_samples_flatten_per_band['log_tau_drw0'], [16, 84])
            sigma_p16, sigma_p84 = np.percentile(obj_flat_samples_flatten_per_band['log_sigma0'], [16, 84])
            tau_in_bounds = tau_p16 <= injected_log_tau <= tau_p84
            sigma_in_bounds = sigma_p16 <= injected_log_sigma <= sigma_p84

            alpha_sigma = obj['alpha_sigma']
            beta_tau = obj['beta_tau']
            eta_A1 = np.median(obj_flat_samples_flatten_per_band['eta_A1'])
            eta_A2 = np.median(obj_flat_samples_flatten_per_band['eta_A2'])
            eta_tau1 = np.median(obj_flat_samples_flatten_per_band['eta_tau1'])
            eta_tau2 = np.median(obj_flat_samples_flatten_per_band['eta_tau2'])
            eta_A1_p16, eta_A1_p84 = np.percentile(obj_flat_samples_flatten_per_band['eta_A1'], [16, 84])
            eta_A2_p16, eta_A2_p84 = np.percentile(obj_flat_samples_flatten_per_band['eta_A2'], [16, 84])
            eta_tau1_p16, eta_tau1_p84 = np.percentile(obj_flat_samples_flatten_per_band['eta_tau1'], [16, 84])
            eta_tau2_p16, eta_tau2_p84 = np.percentile(obj_flat_samples_flatten_per_band['eta_tau2'], [16, 84])
            eta_A1_in_bounds = eta_A1_p16 <= alpha_sigma <= eta_A1_p84
            eta_A2_in_bounds = eta_A2_p16 <= alpha_sigma <= eta_A2_p84

            eta_tau1_in_bounds = eta_tau1_p16 <= beta_tau <= eta_tau1_p84
            eta_tau2_in_bounds = eta_tau2_p16 <= beta_tau <= eta_tau2_p84
            # Determine color: green if all in bounds, else red
            all_in_bounds = all([
                tau_in_bounds, sigma_in_bounds,
                eta_A1_in_bounds, eta_A2_in_bounds,
                eta_tau1_in_bounds, eta_tau2_in_bounds
            ])
            color = "\033[92m" if all_in_bounds else "\033[91m"
            eta_A1_rhat = diagnostics.get('eta_A1_rhat', np.nan)
            eta_A2_rhat = diagnostics.get('eta_A2_rhat', np.nan)
            eta_tau1_rhat = diagnostics.get('eta_tau1_rhat', np.nan)
            eta_tau2_rhat = diagnostics.get('eta_tau2_rhat', np.nan)
            print(
                f"{color}[FAKE INJECT] Object {obj['object_id']}:\n"
                f"  log10_tau:    injected = {injected_log_tau/np.log(10):.3f}, "
                f"recovered = {recovered_log_tau/np.log(10):.3f} ± {(tau_p84-tau_p16)/2/np.log(10):.3f} "
                f"(16th = {tau_p16/np.log(10):.3f}, 84th = {tau_p84/np.log(10):.3f}, in bounds: {tau_in_bounds})\n"
                f"  log10_sigma:  injected = {injected_log_sigma/np.log(10):.3f}, "
                f"recovered = {recovered_log_sigma/np.log(10):.3f} ± {(sigma_p84-sigma_p16)/2/np.log(10):.3f} "
                f"(16th = {sigma_p16/np.log(10):.3f}, 84th = {sigma_p84/np.log(10):.3f}, in bounds: {sigma_in_bounds})\n"
                f"  alpha_sigma: injected = {alpha_sigma:.3f}, "
                f"eta_A1 = {eta_A1:.3f} ± {(eta_A1_p84-eta_A1_p16)/2:.3f} (in bounds: {eta_A1_in_bounds}, rhat={eta_A1_rhat:.3f}), "
                f"eta_A2 = {eta_A2:.3f} ± {(eta_A2_p84-eta_A2_p16)/2:.3f} (in bounds: {eta_A2_in_bounds}, rhat={eta_A2_rhat:.3f})\n"
                f"  beta_tau: injected = {beta_tau:.3f}, "
                f"eta_tau1 = {eta_tau1:.3f} ± {(eta_tau1_p84-eta_tau1_p16)/2:.3f} (in bounds: {eta_tau1_in_bounds}, rhat={eta_tau1_rhat:.3f}), "
                f"eta_tau2 = {eta_tau2:.3f} ± {(eta_tau2_p84-eta_tau2_p16)/2:.3f} (in bounds: {eta_tau2_in_bounds}, rhat={eta_tau2_rhat:.3f})\033[0m"
            )
        final_result_obj = obj | result | diagnostics | dict(prefix=prefix, suffix=suffix)
        results.append(final_result_obj)
        logging.info("--------------------------------------------------------------")
    
    save_quasar_list_hdf5(results, ignored_keys=['X', 'y', 'yerr', 'band_idx'])
        
    sys.exit("Exiting the program as requested.")
