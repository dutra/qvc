# Hubble sample-selection controls

The hard-cut science magnitude support defaults to inclusive [17, 27],
within padded map edges [16.5, 27.5]. Narrower science bounds can be set with
QVC_CUT_COMPLETENESS_MAG_2500_MIN/MAX. The map retains fixed redshift coverage
[0, 4.5], separate from the fitting redshift range.

Tier 2 adds configurable spectral S/N, information, and PSF host-fraction cuts,
and makes the existing point-count cut configurable. Defaults are 400 points,
S/N >=3, eta_sigma_kl >=0.05, and PSF host fraction in [0, 0.90]. Set
QVC_CUT_LIGHT_CURVE_N_POINTS_MIN, QVC_CUT_SN_MEDIAN_ALL_MIN,
QVC_CUT_ETA_SIGMA_KL_MIN, and QVC_CUT_F_HOST_2500_PSF_MAX respectively.
Use none to disable a cut including required-column and finiteness checks.
Point counts retain the existing sum over available bands excluding u.
Enabled thresholds are inclusive and are recorded for resume validation.
