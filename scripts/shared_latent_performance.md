# Original SLB prediction benchmark

Run each case in a fresh process with `PYTHONPATH=src python scripts/benchmark_shared_latent_performance.py --mode optimized --stage prediction` and repeat with `--mode baseline`. Stages are likelihood, training, prediction, plot, and loo. JSON records first-call and repeated timings, peak RSS, compiler temporary memory where available, and the SLB kernel hash. Baseline uses dense conditioning or the existing dense LOO calculation. No production performance claim is implied by synthetic timings.

Predictions compute marginal means and variances in blocks of 64; these are not joint posterior draws. Training refinement computes only the conditional mean. LOO uses the same historical diagonal regularization through QSM precision operations.
