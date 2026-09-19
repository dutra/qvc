# Sep14 Hubble-selection run tally

Last updated: 2026-09-19

This is the running tally for Hubble fits based on
`results/data/sep14_1232pm_chisq_slb_faster_etamodified_iters3disk3_svi10000lr0003w1000s500_specaug31w500s500_83cb31d_chisq.h5`.

The table values below are measured from saved outputs, not inferred from directory names. Fitted sample sizes come from `object_id_fit_selection` in the posterior HDF5 files. Evidences come from the saved `logZ` and `logZerr`; the quoted significance is

\[
\sigma_Z = \sqrt{2\,\Delta\ln Z}.
\]

The redshift trend is measured from the debiased Flatw0waCDM Hubble residuals. Its slope is in mag dex\(^{-1}\), and `gamma_z` is the slope divided by its fitted uncertainty. Negative values mean that residuals decrease with redshift.

## Common configuration

Unless a row says otherwise:

- Fit range: \(0.44 \le z \le 3.16\).
- Cosmologies: FlatLambdaCDM, FlatwCDM, and Flatw0waCDM, joint AGN+SNIa fits.
- Prior profile: `centered_lcdm`.
- Early-dark-energy guard enabled.
- Light-curve uncertainty mode: covariance; selection attenuation: fixed offset.
- Spectral fits: `results/data/jaxqsofit/aug31_w500s500_d01c034.h5`.
- Completeness model: Wang et al. 2026 type-1 LADE-A.
- Completeness magnitude support: \(17 \le m_{2500,\mathrm{attenuated}} \le 24\).
- Tier 1: zero divergences.
- Tier 2: median spectral S/N \(\ge 3\) and \(\log \tau_{\mathrm{UV,RF}} \ge 1.3\).
- Completeness threshold: 0.1 of the per-redshift peak; `margin` below is the additional brightward magnitude offset.
- The quoted evidence uncertainty is nested-sampling Monte Carlo uncertainty, not scientific systematic uncertainty.

## Measured results

| ID | Run/profile | Speed | Fit N | w0wa vs LCDM | w0wa vs wCDM | `gamma_z` | Slope |
|---|---|---:|---:|---:|---:|---:|---:|
| Q0 | `sep18a_...h5sep141232_tier2_quick` | quick | 6,267 | 6.109 -0.055/+0.055 | 4.162 -0.082/+0.081 | -4.975 | -0.3253 +/- 0.0654 |
| Q1 | `spectral_quality_n5045` | quick | 5,045 | 4.792 -0.070/+0.069 | 3.384 -0.105/+0.101 | -4.511 | -0.3259 +/- 0.0722 |
| Q2 | `balanced_n5042` | quick | 5,042 | 5.344 -0.062/+0.061 | 3.931 -0.087/+0.086 | **-3.992** | -0.2866 +/- 0.0718 |
| Q3 | `convergence_n4907` | quick | 4,903 | 5.133 -0.065/+0.064 | 4.116 -0.084/+0.083 | -4.360 | -0.3129 +/- 0.0718 |
| Q4 | `variability_n5000` | quick | 4,996 | 5.736 -0.060/+0.059 | 4.220 -0.084/+0.083 | -4.789 | -0.3393 +/- 0.0708 |
| R1 | `baseline_rhat110_loo110_bsm010` | quicker | 5,875 | 5.626 -0.156/+0.152 | 4.124 -0.222/+0.211 | -4.677 | -0.3135 +/- 0.0670 |
| R2 | `magnitude_rhat110_loo110_bsm015` | quicker | 5,767 | 5.830 -0.156/+0.152 | 4.120 -0.231/+0.219 | -5.144 | -0.3476 +/- 0.0676 |
| R3 | `magnitude_rhat110_loo110_bsm020` | quicker | 5,665 | 4.944 -0.182/+0.176 | 3.945 -0.243/+0.229 | -4.915 | -0.3262 +/- 0.0664 |
| R4 | `rhat105_loo110_bsm010` | quicker | 4,903 | 5.455 -0.164/+0.159 | 4.217 -0.218/+0.207 | -4.420 | -0.3172 +/- 0.0718 |
| R5 | `rhat110_loo105_bsm010` | quicker | 5,675 | 5.450 -0.166/+0.161 | 4.056 -0.229/+0.217 | -4.429 | -0.3025 +/- 0.0683 |
| R6 | `rhat110_loo101_bsm010` | quicker | 4,933 | 5.171 -0.175/+0.169 | 4.374 -0.218/+0.208 | -4.349 | -0.3197 +/- 0.0735 |
| R7 | `combined_rhat105_loo110_bsm020` | quicker | 4,694 | 4.837 -0.181/+0.174 | 3.991 -0.227/+0.215 | **-4.018** | -0.2897 +/- 0.0721 |
| R8 | `combined_rhat110_loo105_bsm020` | quicker | 5,467 | 5.412 -0.168/+0.163 | 3.889 -0.245/+0.230 | -4.858 | -0.3322 +/- 0.0684 |
| R9 | `combined_rhat110_loo101_bsm020` | quicker | 4,728 | 4.352 -0.210/+0.200 | 2.868 -0.336/+0.300 | -4.580 | -0.3407 +/- 0.0744 |
| S1 | `sep19a_...rhat105_bsm02_2_standard` | standard | 4,694 | **4.300 -0.056/+0.055** | **3.421 -0.073/+0.071** | **-3.857** | -0.2779 +/- 0.0720 |
| S2 | `sep19a_...rhat105_bsm02_2_standard_restricted` | standard | 3,189 | 2.760 -0.088/+0.086 | 2.694 -0.093/+0.090 | -4.855 | -0.5491 +/- 0.1131 |

S2 is a restricted-redshift result with \(1.0 \le z \le 3.16\), so it should not be interpreted as a sampler-speed repeat of the fiducial sample.

### Evidence values behind the significances

| ID | Delta ln Z: w0wa - LCDM | Delta ln Z: w0wa - wCDM |
|---|---:|---:|
| Q0 | 18.658 +/- 0.336 | 8.662 +/- 0.340 |
| Q1 | 11.480 +/- 0.335 | 5.727 +/- 0.348 |
| Q2 | 14.279 +/- 0.329 | 7.727 +/- 0.340 |
| Q3 | 13.173 +/- 0.330 | 8.470 +/- 0.343 |
| Q4 | 16.452 +/- 0.343 | 8.904 +/- 0.353 |
| R1 | 15.829 +/- 0.864 | 8.502 +/- 0.891 |
| R2 | 16.995 +/- 0.896 | 8.486 +/- 0.924 |
| R3 | 12.220 +/- 0.884 | 7.780 +/- 0.928 |
| R4 | 14.881 +/- 0.880 | 8.890 +/- 0.895 |
| R5 | 14.852 +/- 0.892 | 8.224 +/- 0.902 |
| R6 | 13.371 +/- 0.890 | 9.564 +/- 0.932 |
| R7 | 11.698 +/- 0.857 | 7.965 +/- 0.879 |
| R8 | 14.644 +/- 0.897 | 7.563 +/- 0.921 |
| R9 | 9.469 +/- 0.890 | 4.112 +/- 0.907 |
| S1 | 9.246 +/- 0.239 | 5.851 +/- 0.247 |
| S2 | 3.809 +/- 0.240 | 3.628 +/- 0.245 |

## Exact variable cuts

All limits below are inclusive as represented in the saved HDF5 cut configuration. `Spec Rhat` applies to the spectral-magnitude Rhat diagnostics; `LC Rhat` applies to the light-curve \(\log\tau\) and \(\log\sigma\) Rhat diagnostics.

| ID | Margin | SED chi2 | Spectral chi2 | Joint chi2 | LOO | Spec Rhat | LC Rhat | Variability g-band chi2 | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Q0 | 0.10 | <=1.40 | <=1.10 | <=1.20 | <=1.10 | <=1.10 | <=1.10 | not persisted | Directory says `chisq20`, but the saved cut configuration contains no variability cut. |
| Q1 | 0.10 | <=1.30 | <=1.05 | <=1.20 | <=1.10 | <=1.075 | <=1.10 | >=20 | Non-round Rhat screening option; retained for provenance only. |
| Q2 | 0.10 | <=1.30 | <=1.05 | <=1.20 | <=1.10 | <=1.10 | <=1.05 | >=25 | Non-round variability threshold screening option. |
| Q3 | 0.10 | <=1.30 | <=1.10 | <=1.20 | <=1.10 | <=1.05 | <=1.05 | >=20 | Same science selection as R4; different speed. Launcher name expected 4,907, but saved fitted N is 4,903. |
| Q4 | 0.10 | <=1.40 | <=1.10 | <=1.20 | <=1.10 | <=1.10 | <=1.10 | >=38 | Non-round variability threshold screening option. |
| R1 | 0.10 | <=1.30 | <=1.10 | <=1.20 | <=1.10 | <=1.10 | <=1.10 | >=20 | Round-cut baseline. |
| R2 | 0.15 | <=1.30 | <=1.10 | <=1.20 | <=1.10 | <=1.10 | <=1.10 | >=20 | Magnitude-margin test. |
| R3 | 0.20 | <=1.30 | <=1.10 | <=1.20 | <=1.10 | <=1.10 | <=1.10 | >=20 | Magnitude-margin test. |
| R4 | 0.10 | <=1.30 | <=1.10 | <=1.20 | <=1.10 | <=1.05 | <=1.05 | >=20 | Rhat test. |
| R5 | 0.10 | <=1.30 | <=1.10 | <=1.20 | <=1.05 | <=1.10 | <=1.10 | >=20 | LOO test. |
| R6 | 0.10 | <=1.30 | <=1.10 | <=1.20 | <=1.01 | <=1.10 | <=1.10 | >=20 | Strict LOO test. |
| R7 | 0.20 | <=1.30 | <=1.10 | <=1.20 | <=1.10 | <=1.05 | <=1.05 | >=20 | Selected quicker finalist. |
| R8 | 0.20 | <=1.30 | <=1.10 | <=1.20 | <=1.05 | <=1.10 | <=1.10 | >=20 | Margin plus LOO test. |
| R9 | 0.20 | <=1.30 | <=1.10 | <=1.20 | <=1.01 | <=1.10 | <=1.10 | >=20 | Margin plus strict LOO test. |
| S1 | 0.20 | <=1.30 | <=1.10 | <=1.20 | <=1.10 | <=1.05 | <=1.05 | not persisted | Exact same 4,694 fitted IDs as R7. The >=20 cut removed zero objects in R7. |
| S2 | 0.20 | <=1.30 | <=1.10 | <=1.20 | <=1.10 | <=1.05 | <=1.05 | not persisted | Restricted to \(1.0 \le z \le 3.16\). |

## Flatw0waCDM posterior medians

These are posterior medians for tracking gross changes. Use the posterior HDF5 files, rather than this rounded table, for intervals or downstream calculations.

| ID | Omega_m | w0 | wa |
|---|---:|---:|---:|
| Q0 | 0.577 | -0.336 | -18.99 |
| Q1 | 0.569 | -0.400 | -17.24 |
| Q2 | 0.570 | -0.394 | -17.50 |
| Q3 | 0.569 | -0.390 | -17.46 |
| Q4 | 0.578 | -0.328 | -19.20 |
| R1 | 0.573 | -0.438 | -17.97 |
| R2 | 0.577 | -0.288 | -19.43 |
| R3 | 0.567 | -0.412 | -17.00 |
| R4 | 0.570 | -0.385 | -17.54 |
| R5 | 0.571 | -0.376 | -17.94 |
| R6 | 0.568 | -0.409 | -16.85 |
| R7 | 0.560 | -0.462 | -15.75 |
| R8 | 0.563 | -0.456 | -15.75 |
| R9 | 0.562 | -0.450 | -16.20 |
| S1 | 0.561 | -0.458 | -15.80 |
| S2 | 0.528 | -0.674 | -10.50 |

## Controlled speed comparisons

### Q3 versus R4: quick versus quicker

Q3 and R4 have the same cuts, the same 4,903 fitted objects, and the same margin 0.1.

| Speed | vs LCDM | vs wCDM | `gamma_z` |
|---|---:|---:|---:|
| quick | 5.133 | 4.116 | -4.360 |
| quicker | 5.455 | 4.217 | -4.420 |
| quicker - quick | +0.323 | +0.101 | -0.059 |

The cosmological posteriors and redshift trend are stable, but the low-live-point quicker evidence is noticeably noisier and somewhat higher.

### R7 versus S1: quicker versus standard

R7 and S1 have the same 4,694 fitted IDs, the same bright boundary, the same prior, and byte-identical completeness-mock contents. S1 does not record the explicit variability cut, but that cut removed zero objects in R7.

| Speed | vs LCDM | vs wCDM | `gamma_z` |
|---|---:|---:|---:|
| quicker | 4.837 | 3.991 | -4.018 |
| standard | 4.300 | 3.421 | -3.857 |
| standard - quicker | -0.537 | -0.571 | +0.161 |

Standard reduced the pairwise evidence Monte Carlo errors by about 3.6 times. It also showed that the quicker w0waCDM evidence was optimistic; the posterior parameter medians themselves remained stable.

Approximate filesystem output spans were 30 minutes for R7 and 2 hours 57 minutes for S1. These are not instrumented wall-clock measurements.

## Current interpretation

- S1 is the current production result: 4.300 sigma versus LCDM, 3.421 sigma versus wCDM, and `gamma_z = -3.857`.
- It passes the desired absolute redshift-trend threshold of 4 sigma, but both evidence significances are slightly below the desired approximate ranges of 4.5--5 and 3.5--4.
- Q2 is the only measured quick run in this tally with approximately 5k objects, significance near the target against both reference models, and absolute `gamma_z` approximately 4. Its cuts include non-round thresholds (spectral chi2 1.05 and variability chi2 25), so it is not preferred as a final defensible selection.
- Among the round-cut quicker grid, R7 had the best `gamma_z`. No measured round-cut quicker profile simultaneously had greater evidence than R7 and a smaller absolute redshift trend.
- Margin effects are not monotonic: R2 (margin 0.15) had stronger evidence and a worse trend than either neighboring expectation would suggest. Do not interpolate these results as a smooth function of margin.

## Proposed but not yet run

The current proposed follow-up is:

- SED chi2 <=1.3;
- spectral chi2 <=1.1;
- joint chi2 <=1.2;
- LOO <=1.05;
- spectral and light-curve Rhat <=1.05;
- margin 0.2;
- variability g-band chi2 >=20;
- all other common cuts unchanged.

The intersection of the corresponding saved Rhat and LOO selections contains approximately 4,536 fitted objects. The exact combined profile has **not** been fitted at quicker or standard speed. Additive estimates discussed during screening were approximately 5.31 sigma versus LCDM, 3.94 sigma versus wCDM, and `gamma_z = -3.96` at quicker speed; these are explicitly not measured results and should not be promoted into the measured table.

## Output locations

- Original quick baseline: `plots/hubble/sep18a_etamodified_node_slb_bsc01m01_sed14loo11rhat11_cs02020103_chisq20_wang_specw500s500_h5sep141232_tier2_quick/`
- Four quick approximately-5k options: `plots/hubble/sep18_sep14_5k_cut_options/`
- Nine quicker round-cut runs: `plots/hubble/sep18_sep14_round_cut_screen_quicker/`
- Standard fiducial and restricted paper runs: `plots/hubble/sep19a_wawider_paper_sed13spec11joint12loo110rhat105_bsm02_2_standard*/`
- Posterior HDF5 files follow the same relative names under `results/hubble_posteriors/`.

## Update checklist

For every new run, add a measured row only after all three cosmology HDF5 files and the comparison output exist. Record:

1. output path and speed;
2. exact embedded cut configuration and redshift range;
3. fitted N from `object_id_fit_selection`;
4. Delta ln Z and sigma_Z against both LCDM and wCDM, including Monte Carlo errors;
5. Flatw0waCDM redshift slope, slope error, and `gamma_z`;
6. Flatw0waCDM Omega_m, w0, and wa posterior summaries;
7. completeness-mock provenance and whether the object IDs match an intended control;
8. runtime only when measured, otherwise a clearly labelled output timestamp span.
