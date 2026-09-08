# Experimental shared-forcing, band-dependent continuum poles

Select `--model_variant shared_latent_band_poles_blr` in the normal light-curve
CLI. The original SLB kernel, adapter, convenience entry point, and tests are
unchanged. This is a separate model, not a replacement for an existing run.

```sh
PYTHONPATH=src NUM_CORES=3 python -m qvc.light_curve.fit_light_curves \
  <your existing data-selection and fitting arguments> \
  --model_variant shared_latent_band_poles_blr \
  --band_poles_transition analytic \
  --disk_order 3 --erlang_order 3 \
  --eta_prior_profile modified \
  --flux_linearized_refinement_strategy svi_then_nuts \
  --flux_linearized_refinement_iters 3
```

The orders and refinement count are example pilot settings, not a convergence
recommendation. Both `nuts_each` and `svi_then_nuts` refinement are supported.
`svi_then_nuts` requires the existing `svi+nuts` fitting backend. The existing
`hpc_scripts/sfitlc.py` forwards these flags unchanged. No jobs are submitted by
the implementation, tests, or benchmark. `--fast_solver`,
`--dho_drw_parameterization`, and `--enforce_positive_flux_guard` are incompatible
and rejected. One BLR response per band is supported, with configurable Erlang
order. The existing Balmer-continuum component remains disabled in this variant.

`--band_poles_transition analytic` is the default for new fits. Select
`--band_poles_transition expm` for the original full matrix exponential;
`--band-poles-transition` is an equivalent spelling. This flag is rejected for
other variants. Both choices describe the same stochastic model and priors.

For paired runs, use identical data arguments and fitting settings with distinct
prefixes. This bash/zsh example assumes `DATA_ARGS` is an array containing your
existing data-file and object-selection arguments:

```sh
common=(--model_variant shared_latent_band_poles_blr
        --fit_method svi+nuts --disk_order 3 --erlang_order 3
        --eta_prior_profile modified
        --flux_linearized_refinement_strategy svi_then_nuts
        --flux_linearized_refinement_iters 1
        --nwarm 500 --nsamp 250 --nchains 3)
PREFIX=band_poles_analytic SUFFIX=compare NUM_CORES=3 PYTHONPATH=src \
  python -m qvc.light_curve.fit_light_curves "${DATA_ARGS[@]}" "${common[@]}" \
  --band_poles_transition analytic
PREFIX=band_poles_expm SUFFIX=compare NUM_CORES=3 PYTHONPATH=src \
  python -m qvc.light_curve.fit_light_curves "${DATA_ARGS[@]}" "${common[@]}" \
  --band_poles_transition expm
```

Keep input object ordering identical: the fitter folds the object index into
its fixed initial random seed. Identical seeds do not guarantee identical NUTS
trajectories under floating-point differences; compare posterior estimates and
diagnostics. These commands are documentation only and were not executed.

## Definition and priors

One stationary unit-variance OU state F has common pole f. Each band has a
relaxation `dZ_b/dt = (F-Z_b)/s_b`, with
`s_b = s_2500 (lambda_rest,b / 2500 Angstrom)^eta_tau`. An additional reference
relaxation uses `s_2500`. Band-specific Erlang disk chains are driven by their
band relaxation; all BLR chains are driven by the reference relaxation.
Disk lags retain the fixed wavelength exponent 4/3. Each filtered component is
normalized by its own stationary standard deviation before applying the
existing relative-flux RMS amplitude. Continuum–BLR cross terms are retained.

The new model defaults to the modified profile: `eta_tau ~ Normal(0.5, 0.5)`
and `eta_sigma ~ Normal(-0.8, 0.5)`. Explicit `--eta_prior_profile default`
selects the existing truncated slope priors. All older model defaults are
preserved. The reference slow-pole prior is a truncated Normal in log10
rest-frame days, location 2.5, scale 1.2, bounds [1,4]. The common-pole gap is
`ln f = ln s_2500 - softplus(g)`, with the existing raw-gap Normal calibrated
to a ratio of 150 and width 0.4 dex. This orders f below the UV reference pole,
but does **not** order f below every band pole. Band-pole crossings are allowed
without clipping or swapping. `eta_tau` describes the relaxation-pole law, not
the exact slope of the disk-filtered integral timescales.

## Numerical construction

The new kernel lives in
`src/qvc/light_curve/multiband_model_shared_latent_band_poles_blr.py`; its
independent inference builder and posterior conversion live in
`src/qvc/light_curve/band_poles_fit.py`.

The state dimension is `B+2+B*(disk_order+blr_order)` (30 for four bands and
orders 3/3). A is lower triangular with negative diagonal; Q has only
`Q[0,0]=2/f`. The stationary covariance solves
`A P + P A.T + Q = 0` by row-major triangular recurrence:

```
P[i,j] = (-Q[i,j] - sum(k<i, A[i,k]*P[k,j])
                  - sum(k<j, P[i,k]*A[j,k])) / (A[i,i]+A[j,j])
```

Both transitions operate in this same float64 cascaded state basis. `expm`
uses `expm(A*dt, max_squarings=64)` unchanged. `analytic` assembles exponential
driver terms and Erlang blocks. If R_j(lambda,t) is a chain state's response
to `exp(-lambda*t)`, with rates `a=1/f`, `b=1/s`, its initial-Z coefficient is
`R_j(b,t)` and initial-F coefficient is `b*(R_j(b,t)-R_j(a,t))/(a-b)`.
The scalar convolution uses a 26-term Taylor series for
`abs((q-lambda)*t) <= 1`, preserving coincident-response-rate derivatives.

When `abs(a-b) <= 1e-3*max(a,b)`, analytic mode instead evaluates a local
`(response_order+2)` matrix exponential for that chain's two input columns.
The predicate depends on rates and sits outside the time vmap, so ordinary
likelihoods skip this branch at runtime. Batched parameter evaluations can
cause JAX to evaluate both branches, so the inactive denominator is also safe.
No singular OU-difference state basis is introduced. Coincident states can
make P singular; the solver needs neither its inverse nor artificial latent
jitter. Observational uncertainties still enter the usual GP diagonal.

For a later observation i and earlier observation j, covariance is
`h_i exp(A*dt) P h_j.T`. Independent symmetric and rectangular QSM methods use
this forward convention. The training coordinates must be sorted by physical
time (the adapter sorts them); query order is arbitrary. Ties and extrapolation
are supported. Causal coordinates are selected before exponentiation, so an
unused negative-time exponential cannot contaminate prediction gradients.
Likelihoods still use tinygp's QuasisepSolver. Its generic dense state-matrix
algebra is linear in observation count at fixed state dimension; this does not
make dense matrix exponentials cheap.

## Saved scientific quantities

All raw log samples use natural logs and observer-frame time units. Catalog
log summaries use base 10; `_rf`/`_RF` times are rest-frame days.

* `log_tau_slow_uv_driver` samples the UV **pole**, independently of eta_tau.
  `tau_slow_uv_driver`, `tau_fast_driver`, and `tau_slow_band` store the
  observer-frame poles. Catalog `log_tau_slow_uv_driver_rf` and
  `log_tau_driver_fast_rf` make the reference/common poles explicit.
* `log_tau_uv` is derived for every draw from a temporary response at exactly
  2500 Angstrom, including the UV disk chain and excluding BLR. It is never a
  nearest-band approximation. With continuum loading h its integral time is
  `-h A^{-1} P h.T / (h P h.T)`; only A is inverted/solved, never P.
  Public `log_tau_uv_rf = ln(tau_UV_obs)/ln(10) - log10(1+z)` is the Hubble
  regressor. It generally differs from the reference slow pole.
* `log_sigma_uv` retains the continuum RMS at 2500 Angstrom in the existing
  magnitude-equivalent units: relative-flux RMS times `2.5/ln(10)`. This is
  the existing external convention, not the exact RMS of a logarithmically
  transformed Gaussian process.
* `log_tau_cont_band_<b>_RF` summarizes each continuum-only integral moment;
  `log_tau_band_<b>_RF` / `log_tau_effective_<b>_RF` include the BLR cross terms.
  `log_sigma_total_rms_band_<b>` is the total-band RMS; continuum RMS remains
  `log_sigma_rms_band_<b>`. These summaries are computed from exact moments
  for paired posterior draws, not from a median-parameter kernel.
  Saved sample files additionally contain the per-draw arrays
  `tau_continuum_band_obs`, `tau_total_band_obs`, and `rms_total_band_relflux`.
* `band_pole_below_common_fraction_<b>` is the fraction of draws with `s_b<f`.
  Eta_tau samples and their posterior/KL summaries are retained.
* Sigma–tau uncertainties and covariance use paired UV draws and the existing
  percentile-based covariance regularization. Both embedded draws and sample
  HDF5 files carry the existing continuum-integral semantics marker, so merge
  and Hubble readers recognize them without reinterpreting legacy catalogs.

The new variant's model structure-function diagnostic uses its disk-filtered
continuum at median physical parameters. It is distinct from the exact
draw-wise integral moments and excludes BLR, matching its continuum RMS scale.

Output metadata records the distinct variant, resolved prior profile, pole
definitions, BLR reference driver, and selected transition implementation.
Sample files also record response orders. Do not label the sampled UV pole as
the Hubble regressor when examining raw chains.

`band_poles_transition` and `transition_implementation` identify the fitting
backend; `prediction_band_poles_transition` and
`prediction_transition_implementation` identify the prediction backend.
With `--resume` and no explicit backend, the saved backend is used. Historical
band-pole files without the selector are recognized as `expm`. An explicit
resume/replot override changes prediction evaluation but preserves the fitting
backend label, and the resume policy leaves the source sample file intact.

## Reproducible verification

Use the repository's scientific environment with JAX x64, NumPyro, tinygp,
SciPy, and EzTaoX available. On the development machine this was `jaxcpu4`,
with `PYTHONPATH=src:/home/dutra/dev/eztaox/src` and `MPLCONFIGDIR=/tmp/qvc-mpl`.

```sh
NUM_CORES=3 PYTHONPATH=src python -m pytest -q \
  tests/test_multiband_model_shared_latent_band_poles_blr.py \
  tests/test_multiband_model_shared_latent_blr.py \
  tests/test_multiband_model_dho_blr_erlang.py \
  tests/test_fit_light_curves_recenter.py \
  tests/test_multiband_fit_plotting.py \
  tests/test_lc_fraction_marginalization.py \
  tests/test_merge_results.py tests/test_sfitlc.py

PYTHONPATH=src python scripts/benchmark_band_poles.py \
  --counts 128 512 --cpus 3 --repeats 5 \
  --output results/diagnostics/band_poles_benchmark.json
```

The dedicated tests compare with a separate parallel-OU dense oracle where
that basis is nonsingular, and independent frequency integration at pole
coincidences/crossings. They cover QSM covariance, likelihood, triangular
solves, conditional means/variances, Lyapunov residuals, transition and
semigroup identities, moments, parameter derivatives, zero-slope SLB
equivalence, priors, serialization, plotting PSD, and both refinement dispatch
paths. The actual synthetic SVI→NUTS smoke has only two warmup steps/four
draws; it verifies execution, not convergence or parameter recovery.

Initial model validation: the combined regression run passed 278 tests.
After the final diagnostics/serialization additions, the dedicated suite
passed all 23 tests; the targeted new and existing structure-function checks
passed all 3 tests. SHA-256 checks confirmed the protected SLB source and test
files remained byte-for-byte unchanged. AST comparisons also confirmed the
original SLB-capable builder and relative-flux parameter conversion were
unchanged. `git diff --check` passed.

Analytic-backend validation: the combined suite passed **326 tests**, including
all **69 dedicated band-pole tests**. These additionally cover both backends,
orders 1/2/3/5, gradients across the local-fallback threshold, runtime proof
that normal time batches skip local expm, CLI aliases, and resume provenance.
The HPC backend-flag forwarding check also passed after its final update.

## Measured transition-backend cost (2026-09-07)

JAX 0.6.2 CPU, float64, three-CPU affinity, four bands, orders 3/3, synthetic
times over 5000 observer days. Median of five warmed calls; separate process
for each row. Both models use the same five shared physical parameters; the
new model additionally differentiates eta_tau. Peak RSS includes compilation
and both compiled functions, not just live likelihood arrays.

| Observations | Kernel | Likelihood | Likelihood + gradient | Peak process RSS |
|---:|---|---:|---:|---:|
| 128 | SLB | 0.847 ms | 4.21 ms | 418 MiB |
| 128 | Band poles analytic | 1.35 ms | 7.62 ms | 695 MiB |
| 128 | Band poles expm | 15.82 ms | 74.03 ms | 473 MiB |
| 512 | SLB | 2.55 ms | 10.97 ms | 443 MiB |
| 512 | Band poles analytic | 4.04 ms | 22.76 ms | 803 MiB |
| 512 | Band poles expm | 60.09 ms | 288.36 ms | 836 MiB |

Analytic mode is about 10–13 times faster for warmed likelihood-plus-gradient
calls than full expm here (12–15 times for likelihood alone). Compilation is
longer: first value-and-gradient calls took about 7.5 s for analytic versus
1.8–2.0 s for expm. Peak RSS includes both compiled functions and compiler
memory, explaining why analytic is not uniformly lower-memory.

A separate stress case sets `f=2`, `s_2500=2.0002`, and `eta_tau=0`, activating
all eight local chain fallbacks:

| Observations | Backend | Likelihood | Likelihood + gradient | Peak process RSS |
|---:|---|---:|---:|---:|
| 128 | analytic, all fallbacks | 3.54 ms | 39.67 ms | 696 MiB |
| 128 | expm | 15.58 ms | 72.92 ms | 474 MiB |
| 512 | analytic, all fallbacks | 13.10 ms | 149.16 ms | 805 MiB |
| 512 | expm | 59.10 ms | 289.50 ms | 835 MiB |

These are kernel costs, not full NUTS runtime predictions; leapfrog counts, chains, nuisance
parameters, and cadence matter. Start with a small recovery/predictive pilot
before considering a catalog run. No claim of improved Stone agreement or
scientific identifiability follows from solver correctness alone.

The raw measurements from this verification are saved locally in
`results/diagnostics/band_poles_implementation/analytic_benchmark.json`.
The earlier expm-only measurements remain in `benchmark.json` in that folder.
