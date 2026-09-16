"""Population mean fits and whole-quasar bootstrap envelopes."""
import numpy as np
import pandas as pd
import pytest

from qvc.hubble.sigma_tau_lambda_fit import (
    SDSS_LAMBDA_PIVOT, fit_sigma_tau_lambda_broken_pl, log_broken_pl,
    slope_population_band,
)


def frame():
    z = np.array([1., 1.2, 1.4, 1.6])
    df = pd.DataFrame({'z': z, 'log_sigma_uv': [0., 2., -3., 5.],
                       'log_tau_uv_rf': [1., 3., -1., 4.]})
    blue = np.array([.01, .02, .04, .3])
    red = np.array([.02, .03, .07, .4])
    for band in 'ugri':
        x = np.log10(SDSS_LAMBDA_PIVOT[band] / (1 + z) / 2500)
        y = np.minimum(x, 0) * blue + np.maximum(x, 0) * red
        df[f'log_tau_band_{band}_RF'] = df.log_tau_uv_rf + y
        df[f'log_sigma_band_{band}'] = df.log_sigma_uv - 10 * y
        df[f'log_tau_band_{band}_RF_err'] = [1e-9, 1e9, np.nan, 0.]
    return df, blue, red


def test_population_means_and_bootstrap_quantiles():
    df, blue, red = frame()
    result = fit_sigma_tau_lambda_broken_pl(df, include_plot_payload=True)
    for target, factor in [('tau', 1), ('sigma', -10)]:
        data = result[f'{target}_data']; fit = result[f'fit_{target}']
        x = np.log10(data['lam'] / 2500)
        h = np.column_stack([np.minimum(x, 0), np.maximum(x, 0)])
        slopes = factor*np.array([blue, red]).T
        if target == 'tau':
            logs = np.log(slopes)
            expected = np.exp(logs.mean(0) + .5*logs.var(0))
        else:
            expected = slopes.mean(0)
        np.testing.assert_allclose([fit['d1'], fit['d2']], expected)
        np.testing.assert_allclose(fit['object_slopes'], slopes)
        np.testing.assert_allclose(fit['mean_slope_percentiles'],
                                  np.percentile(fit['bootstrap_mean_slopes'], [16,84], axis=0).T)
        assert fit['intercept'] == 0
        assert log_broken_pl(2500,2500,fit['d1'],fit['d2']) == 0
    changed = df.copy()
    for band in 'ugri':
        changed[f'log_tau_band_{band}_RF_err'] = 3.
    again = fit_sigma_tau_lambda_broken_pl(changed)
    assert again['eta_tau_blue'] == result['eta_tau_blue']
    # Marginal absolute levels cannot alter a direct ratio fit.
    changed.log_tau_uv_rf += 100
    for band in 'ugri':
        changed[f'log_tau_band_{band}_RF'] += 100
    np.testing.assert_allclose(fit_sigma_tau_lambda_broken_pl(changed)['eta_tau_blue'],result['eta_tau_blue'])


def test_band_order_anchor_sign_and_missing_side():
    df, _, _ = frame()
    df.loc[0,'log_tau_band_i_RF'] = np.nan
    fit = fit_sigma_tau_lambda_broken_pl(df)['fit_tau']
    grid = np.array([1000., 2500., 5000.])
    lo, hi = slope_population_band(fit, grid, lam_s=2500)
    assert np.all(lo <= hi)
    assert lo[1] == hi[1] == 0
    assert hi[0] < 0 and lo[2] > 0
    np.testing.assert_allclose(lo[0],np.log10(.4)*fit['empirical_population_quantiles'][0,2])
    with pytest.raises(ValueError, match='both sides'):
        fit_sigma_tau_lambda_broken_pl(df, bands=('u',))


def test_paired_bootstrap_reproducible_and_rejects_nonpositive_tau():
    df, _, _ = frame()
    # All objects have proportional red/blue slopes: resampling individual
    # sides independently would break this exact relation.
    for band in 'ugri':
        x = np.log10(SDSS_LAMBDA_PIVOT[band] / (1+df.z.to_numpy()) / 2500)
        blue = np.array([.01,.02,.04,.3])
        df[f'log_tau_band_{band}_RF'] = df.log_tau_uv_rf + np.where(x<0,blue,2*blue)*x
    a = fit_sigma_tau_lambda_broken_pl(df, bootstrap_replicates=80)['fit_tau']
    b = fit_sigma_tau_lambda_broken_pl(df, bootstrap_replicates=80)['fit_tau']
    np.testing.assert_array_equal(a['bootstrap_mean_slopes'],b['bootstrap_mean_slopes'])
    np.testing.assert_allclose(a['bootstrap_mean_slopes'][:,1],2*a['bootstrap_mean_slopes'][:,0])
    for band in 'ugri':
        df[f'log_tau_band_{band}_RF'] = df.log_tau_uv_rf
    with pytest.raises(ValueError, match='strictly positive'):
        fit_sigma_tau_lambda_broken_pl(df, bootstrap_replicates=10)


def test_tau_uses_continuum_median_parameters_instead_of_stored_total_tau():
    from qvc.light_curve.multiband_model_shared_latent_blr import (
        continuum_effective_timescale,
    )

    df, _, _ = frame()
    df["tau_fast_driver"] = [20.0, 30.0, 40.0, 50.0]
    df["tau_slow_driver"] = [100.0, 120.0, 150.0, 180.0]
    df["lag0"] = [15.0, 18.0, 21.0, 24.0]
    df["lambda_center_rf"] = [2200.0, 2300.0, 2400.0, 2600.0]
    # Deliberately incompatible stored total-tau values must be ignored.
    for band in "ugri":
        lam_rf = SDSS_LAMBDA_PIVOT[band] / (1.0 + df["z"].to_numpy())
        df[f"lag_disk_{band}"] = (
            df["lag0"] * (lam_rf / df["lambda_center_rf"]) ** (4.0 / 3.0)
        )
        df[f"log_tau_band_{band}_RF"] = 99.0
    df["log_tau_uv_rf"] = -99.0

    result = fit_sigma_tau_lambda_broken_pl(
        df,
        bands=tuple("ugri"),
        include_plot_payload=True,
        bootstrap_replicates=20,
    )
    data = result["tau_data"]
    assert data["value_definition"] == "continuum_median_parameter_approximation"

    expected = []
    for band in "ugri":
        for row in range(len(df)):
            reference_lag = df.loc[row, "lag0"] * (
                2500.0 / df.loc[row, "lambda_center_rf"]
            ) ** (4.0 / 3.0)
            reference = float(
                continuum_effective_timescale(
                    df.loc[row, "tau_fast_driver"],
                    df.loc[row, "tau_slow_driver"],
                    reference_lag,
                    disk_order=3,
                )
            )
            band_tau = float(
                continuum_effective_timescale(
                    df.loc[row, "tau_fast_driver"],
                    df.loc[row, "tau_slow_driver"],
                    df.loc[row, f"lag_disk_{band}"],
                    disk_order=3,
                )
            )
            expected.append(np.log10(band_tau / reference))
    np.testing.assert_allclose(data["y_res"], expected, rtol=2e-5, atol=2e-6)
