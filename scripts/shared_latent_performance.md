# Shared-latent performance improvements

The original `multiband_model_shared_latent_blr.py` is unchanged. Its SHA256
before and after implementation is
`7b64e74d47f8252c65fbb3af36c5e5f19b192d42aecaa73c472307935537e9c4`.
The existing SLB test file and convenience entry point are also unchanged.

## What changes

Both shared-latent variants use the shared adapter's new `pred_training_mean`
for refinement. For training covariance `C = K + D`, this computes
`alpha = C^-1 (y_corrected - mean)` and `y_corrected - D alpha`. It restores the
inverse GP sort followed by the inverse constructor sort. Survey and seeing
offsets remain subtracted exactly as in the previous prediction path. Every
refinement iteration, including the final diagnostic update, is retained.

`pred(params, X)` still returns the conditional mean and standard deviation in
the requested coordinate order. It now uses sequential blocks of 64 queries.
Within each block, sorted queries form the second coordinate argument of the
forward-aware rectangular QSM. Multiplication by a block-sized identity gives
the training-to-query cross covariance. A triangular solve supplies its
contribution to the marginal variance. This never allocates an N-training
identity or a full conditional covariance. Padding, ties and extrapolation are
supported; padding is removed and query order restored before returning.

The variance includes tinygp's original default prediction diagonal,
`sqrt(finfo(float64).eps) = 1.4901161193847656e-8`. No new variance clipping is
introduced. Joint conditional covariance and joint posterior sampling APIs
are unchanged; drawing independently from the marginal blocks would be wrong.

LOO uses the diagonal and multiplication of a QSM precision matrix, with the
same `1e-10 * max(nanmedian(diag(C)), 1)` regularization as before. Residual
standardization, chi-squared, finite filtering and the lag-bin calculations are
unchanged. Lag-bin pair arrays are still allocated; this change specifically
removes the dense covariance/precision stage. Other model variants retain
their previous prediction and LOO implementations.

Only the band-pole kernel changes: a row-wise triangular Lyapunov solve
replaces the scalar recurrence, and `jax.checkpoint` wraps the local `expm`
fallback. The diagonal equation doubles off-diagonal row coefficients, not
the already doubled diagonal. Neither change requires invertible stationary
covariance. The rate threshold, analytic formulas, causal direction, float64
arithmetic, `max_squarings=64` and explicit `expm` backend are retained.

Band-pole posterior moments use stable cached JIT functions and sequential
batches of 128 draws. Total and continuum moments share a stationary
covariance; the exact separate UV response is retained. The same moment
payload goes to catalog summaries and sample serialization. NumPyro's
chain/draw axes are restored after one augmentation; fixed `bc_weight` remains
a wavelength vector. Empirical fraction resampling still triggers fresh
augmentation, so fractions are applied once. No posterior values are cached
as static arguments.

There are no new run flags. Priors, seeds, schedules, backend provenance,
Hubble quantities and saved field definitions remain unchanged. Single-scatter
assembly, chain-scheduling changes and inference-wide compilation reuse are
not included.

## Reproduce validation

Use the project's scientific environment (measured here with `jaxcpu4`, JAX
0.6.2, CPU, float64). On this checkout:

```bash
export PYTHONPATH=src:/home/dutra/dev/eztaox/src
export NUM_CORES=3
export MPLCONFIGDIR=/tmp/qvc-mpl
PYTHON=/home/dutra/.conda/envs/jaxcpu4/bin/python

$PYTHON -m pytest -q \
  tests/test_shared_latent_performance.py \
  tests/test_multiband_model_shared_latent_band_poles_blr.py \
  tests/test_multiband_model_shared_latent_blr.py \
  tests/test_multiband_model_dho_blr_erlang.py \
  tests/test_fit_light_curves_outlier_rejection.py \
  tests/test_fit_light_curves_recenter.py \
  tests/test_lc_fraction_marginalization.py \
  tests/test_multiband_fit_plotting.py \
  tests/test_svi_warm_start.py
sha256sum src/qvc/light_curve/multiband_model_shared_latent_blr.py
git diff --check
```

Tests compare covariance, transitions, semigroup identities, solves,
likelihoods, predictions, gradients and moments against dense calculations.
They cover the two transition backends, orders 1/2/3/5, wavelength slopes,
crossings, coincident poles, fallback boundaries and long-separation gradients.
Additional tests exercise both sorts, nuisance corrections, partial query and
posterior batches, prediction noise, full LOO diagnostic outputs, fraction
scaling, chain shape, moment reuse and Hubble fields. Short real SVI/NUTS smoke
tests exercise both refinement schedules; they are not scientific fits.

## Reproduce benchmarks

`scripts/benchmark_shared_latent_performance.py` runs one isolated process per
invocation. The default baseline revision is the pre-change commit
`3904cdda87f33d92b728058214e7c865463c39ea`. It reads baseline methods through
`git show`; it never checks out or edits source. A Git checkout containing that
commit is required for baseline modes.

```bash
for n in 128 512; do
  for mode in baseline row checkpoint optimized; do
    $PYTHON scripts/benchmark_shared_latent_performance.py \
      --stage likelihood --variant analytic --mode "$mode" --n "$n" \
      --output "/tmp/${mode}_likelihood_${n}.json"
  done
done
for mode in baseline optimized; do
  $PYTHON scripts/benchmark_shared_latent_performance.py \
    --stage likelihood --case fallback --mode "$mode" --n 512
  for variant in slb analytic; do
    $PYTHON scripts/benchmark_shared_latent_performance.py \
      --stage training --variant "$variant" --mode "$mode" --n 512
    for q in 600 800; do
      $PYTHON scripts/benchmark_shared_latent_performance.py \
        --stage prediction --variant "$variant" --mode "$mode" --n 512 --queries "$q"
      $PYTHON scripts/benchmark_shared_latent_performance.py \
        --stage plot --variant "$variant" --mode "$mode" --n 512 --queries "$q"
    done
    for n in 512 2048; do
      $PYTHON scripts/benchmark_shared_latent_performance.py \
        --stage loo --variant "$variant" --mode "$mode" --n "$n"
    done
  done
  $PYTHON scripts/benchmark_shared_latent_performance.py \
    --stage posterior --mode "$mode" --draws 750
done
```

Each process uses the same three CPUs, data and parameters. Timing synchronizes
JAX results and reports the median of five warmed calls separately from the
first call. Linux peak RSS includes compilation and is distinct from compiler
temporary-memory estimates. Posterior baseline calls recreate JIT closures,
so their repeated-call timings include that repeated compilation.

`plot` calls the real four-band plotting prediction helper, including display
conversion, at 600 or 800 points per band. It excludes Matplotlib, PSD and corner
rendering. `loo` calls the full production LOO diagnostic, including GP
construction and pair bins. These isolated peaks are not additive and cannot
predict an entire long-running worker's high-water mark or sampling ESS/sec.

## Measurements on this machine (2026-09-07)

CPU affinity was restricted to the same three CPUs for each isolated benchmark. The 512-point kernel comparison was repeated after the broad regression process exited, using nine warmed repetitions. Other rows use five repetitions. Results and individual first-call times are under `results/diagnostics/shared_latent_performance_implementation/`; `summary.json` summarizes the production-stage comparisons. Files marked `fused_probe` are earlier fused microbenchmarks and are **not** used in the tables below.

| Model | Stage | Observations | Peak RSS, MiB (before → after) | Peak saved | Warm time, ms (before → after) | Speed ratio |
|---|---|---:|---:|---:|---:|---:|
| SLB | Training mean | 512 | 2612 → 320 | 2292 MiB (87.7%) | 502.11 → 2.71 | 185.50× |
| SLB | Marginals (600 queries) | 512 | 3022 → 578 | 2444 MiB (80.9%) | 605.97 → 42.21 | 14.36× |
| SLB | Marginals (800 queries) | 512 | 3813 → 561 | 3252 MiB (85.3%) | 804.50 → 53.77 | 14.96× |
| SLB | Plot helper (600/band) | 512 | 2782 → 751 | 2031 MiB (73.0%) | 3251.24 → 165.21 | 19.68× |
| SLB | Plot helper (800/band) | 512 | 3498 → 716 | 2782 MiB (79.5%) | 4231.68 → 187.49 | 22.57× |
| SLB | Full LOO | 512 | 783 → 696 | 87 MiB (11.2%) | 213.74 → 141.48 | 1.51× |
| SLB | Full LOO | 2048 | 1785 → 945 | 840 MiB (47.0%) | 1834.50 → 179.24 | 10.23× |
| Band poles | Training mean | 512 | 4580 → 375 | 4205 MiB (91.8%) | 900.37 → 4.61 | 195.41× |
| Band poles | Marginals (600 queries) | 512 | 5195 → 632 | 4563 MiB (87.8%) | 1045.04 → 66.44 | 15.73× |
| Band poles | Marginals (800 queries) | 512 | 6683 → 653 | 6030 MiB (90.2%) | 1422.86 → 87.97 | 16.17× |
| Band poles | Plot helper (600/band) | 512 | 5280 → 808 | 4471 MiB (84.7%) | 3863.24 → 260.76 | 14.82× |
| Band poles | Plot helper (800/band) | 512 | 6816 → 806 | 6010 MiB (88.2%) | 5336.13 → 333.90 | 15.98× |
| Band poles | Full LOO | 512 | 1252 → 1141 | 111 MiB (8.9%) | 1533.17 → 1539.42 | 1.00× |
| Band poles | Full LOO | 2048 | 2467 → 1562 | 905 MiB (36.7%) | 3343.94 → 1571.73 | 2.13× |
| Band poles | Posterior moments (750 draws) | — | 391 → 334 | 57 MiB (14.6%) | 1117.54 → 29.19 | 38.28× |

Kernel-only comparisons isolate the row solve and checkpoint changes. Peak RSS covers both likelihood and gradient compilations in each worker; compiler temporary memory below is for the gradient executable only.

| Case | N | Change | Warm likelihood, ms | Warm value+gradient, ms | Compiler temporary MiB | Peak RSS MiB |
|---|---:|---|---:|---:|---:|---:|
| normal | 128 | baseline | 1.12 | 9.34 | 34.0 | 679 |
| normal | 128 | row | 1.40 | 7.64 | 33.5 | 680 |
| normal | 128 | checkpoint | 1.06 | 5.94 | 8.1 | 684 |
| normal | 128 | optimized | 1.01 | 4.34 | 7.5 | 686 |
| normal | 512 | baseline | 3.92 | 24.63 | 133.0 | 796 |
| normal | 512 | row | 3.89 | 22.31 | 132.5 | 794 |
| normal | 512 | checkpoint | 3.95 | 16.14 | 29.2 | 709 |
| normal | 512 | optimized | 4.01 | 14.55 | 28.6 | 713 |
| fallback | 512 | baseline | 47.33 | 168.23 | 133.0 | 797 |
| fallback | 512 | row | 46.29 | 168.66 | 132.5 | 793 |
| fallback | 512 | checkpoint | 47.04 | 203.65 | 29.2 | 715 |
| fallback | 512 | optimized | 47.72 | 206.54 | 28.6 | 715 |

The ordinary 512-point gradient is about **1.69× faster**, with about **10% less process peak RSS** and **78% less compiler temporary memory**. The row solver alone saves almost no process RSS here. Small 128-point process peaks do not improve consistently.

When every band/reference pole is near the common pole, checkpointing makes the gradient about **23% slower** (168 → 207 ms), while still reducing peak RSS about 10% and temporary memory about 78%. This is a deliberate memory/recomputation tradeoff. The all-fallback synthetic stress case does not measure how often fallback occurs in a real posterior. The fallback remains necessary for numerical correctness.

The full 512-observation band-pole LOO diagnostic has essentially unchanged warmed runtime. Larger LOO problems benefit more. In particular, the much larger speed/RSS savings from a fused standalone precision microbenchmark must not be substituted for these full-diagnostic measurements.

Kernel changes affect repeated SVI/NUTS likelihood gradients. Training means run once per refinement iteration; marginal predictions run during plotting. LOO and moment summaries are post-fit work at the object level. Posterior compilation reuse removes repeated compilation and batching bounds draw-dependent work; it does not change the sampler or its chain schedule. No ESS/sec or production-run wall-time claim is made.

Validation completed across the regression runs: **315 unique tests passed; 3 inapplicable SLB-only fallback combinations were skipped**. The 290 existing tests comprise 284 passing shared/kernel regressions plus six refinement cases rerun successfully after correcting fixed-vector chain reshaping. The dedicated suite contributes 18 passing tests, and plotting/SVI regressions contribute seven. The initial import and chain-reshape failures were corrected before this final validation. Numerical agreement is checked at float64 tolerances, not bitwise equality.
