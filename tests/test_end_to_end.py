import os
import sys
from pathlib import Path

import h5py
import numpy as np
import jax.numpy as jnp
from jax import device_get, random
from jax.tree_util import tree_map
from numpyro.infer import MCMC, NUTS


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.hubble import hubble_plotting, hubble_utils
from qvc.light_curve.fit_light_curves import build_single_object_model, make_lc
from qvc.light_curve.multiband_fit_utils import (
    flatten_flat_samples_per_band,
    lambda_pivot,
    process_samples,
)


def _write_test_quasars_hdf5(path, quasars):
    path.parent.mkdir(parents=True, exist_ok=True)
    string_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(path, "w") as hdf:
        for quasar in quasars:
            group = hdf.create_group(str(quasar["object_id"]))
            for key, value in quasar.items():
                if key == "object_id":
                    continue
                if isinstance(value, dict):
                    sub_group = group.create_group(key)
                    for sub_key, sub_value in value.items():
                        arr = np.asarray(sub_value)
                        if arr.dtype.kind in {"U", "S", "O"}:
                            arr = arr.astype(string_dt)
                        sub_group.create_dataset(sub_key, data=arr)
                else:
                    arr = np.asarray(value)
                    if arr.dtype.kind in {"U", "S", "O"}:
                        arr = arr.astype(string_dt)
                    group.attrs[key] = arr


def _make_fake_public_object():
    bands = ("g", "r", "i", "z")
    return {
        "object_id": "101",
        "z": 1.35,
        "times": {
            band: np.linspace(58000.0, 58540.0, 12, dtype=float)
            for band in bands
        },
        "mags": {
            "g": np.linspace(20.0, 20.25, 12, dtype=float),
            "r": np.linspace(19.7, 19.92, 12, dtype=float),
            "i": np.linspace(19.5, 19.72, 12, dtype=float),
            "z": np.linspace(19.3, 19.55, 12, dtype=float),
        },
        "magerrs": {band: np.full(12, 0.05, dtype=float) for band in bands},
        "cadence": {band: 7.0 for band in bands},
        "cadence_err": {band: 0.5 for band in bands},
        "number_points": {band: 12 for band in bands},
    }


def test_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    for name in (
        "plot_alpha_lambda_vs_l2500_by_redshift",
        "plot_alpha_lambda_histogram",
        "plot_blr_lag_vs_amp_by_band",
        "plot_blr_lag_vs_redshift_by_band",
        "plot_f_host_center_vs_l2500",
        "plot_Mi_relation",
        "plot_cut_diagnostics",
        "plot_m2500_vs_z_colorpanels",
        "plot_sigma_uv_host_correction",
        "plot_tau_sigma_vs_wu_catalog",
        "plot_tau_sigma_vs_redshift",
    ):
        monkeypatch.setattr(hubble_plotting, name, lambda *args, **kwargs: None)

    monkeypatch.setattr(hubble_utils, "populate_xray", lambda df: df)

    obj = _make_fake_public_object()
    lc = make_lc(obj, bands=["g", "r", "i", "z"], inject_fake=True, drop_band_lyman_alpha=False)

    assert lc is not None
    assert lc["bands"] == ["g", "r", "i"]
    obj = obj | lc

    bands = obj["bands"]
    lam_rf = jnp.array([lambda_pivot[b] for b in bands], dtype=float) / (1.0 + float(obj["z"]))
    bidx = np.asarray(obj["band_idx"])
    yerr = np.asarray(obj["yerr"])
    log_jitter_mean = np.array(
        [
            np.log(np.mean(yerr[(bidx == i) & np.isfinite(yerr) & (yerr < 10)]))
            for i in range(len(bands))
        ],
        dtype=float,
    )
    numpyro_model = build_single_object_model(
        obj,
        lam_rf,
        log_jitter_mean=jnp.array(log_jitter_mean),
        disable_poly1=False,
        disable_lag_blr=False,
        drop_band_lyman_alpha=False,
        sigma_tau_uniform=False,
        tau_fast_truncated=False,
    )

    nuts = NUTS(
        numpyro_model,
        dense_mass=False,
        max_tree_depth=2,
        target_accept_prob=0.8,
    )
    mcmc = MCMC(
        nuts,
        num_warmup=5,
        num_samples=8,
        num_chains=1,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(0))

    samples_flat = mcmc.get_samples(group_by_chain=False)
    samples_flat = tree_map(lambda x: np.asarray(device_get(x)), samples_flat)
    flat_per_band = flatten_flat_samples_per_band(samples_flat, bands=bands)
    result = process_samples(flat_per_band, obj, bands=bands)

    quasar = {
        "object_id": obj["object_id"],
        "z": float(obj["z"]),
        "mags_mean": np.asarray(obj["mags_means"], dtype=float),
        "dropped_bands": np.asarray(obj["dropped_bands"], dtype="U"),
        "t_rf_length": float(obj["t_rf_length"]),
        "t_obs_length": float(obj["t_obs_length"]),
        "ebv_wu": 0.01,
        "apparent_mag_2500": 20.2,
        "alpha_lambda": -1.45,
        "alpha_lambda_err": 0.08,
        "ra": 150.0,
        "dec": 2.0,
        "cadence": obj["cadence"],
        "cadence_err": obj["cadence_err"],
        "number_points": obj["number_points"],
        "log_sigma_UV": float(result["log_sigma_UV"]),
        "log_sigma_UV_err": float(result["log_sigma_UV_err"]),
        "log_tau_UV_RF": float(result["log_tau_UV_RF"]),
        "log_tau_UV_RF_err": float(result["log_tau_UV_RF_err"]),
        "log_sigma_UV_log_tau_UV_RF_cov_psd": float(result["log_sigma_UV_log_tau_UV_RF_cov_psd"]),
        "log_sigma_UV_std_psd": float(result["log_sigma_UV_std_psd"]),
        "log_tau_UV_RF_std_psd": float(result["log_tau_UV_RF_std_psd"]),
        "log_jitter_u": -9.0,
        "log_amp_delta_blr_u": -9.0,
        "log_jitter_g": float(np.percentile(flat_per_band["log_jitter_g"], 50)),
        "log_jitter_r": float(np.percentile(flat_per_band["log_jitter_r"], 50)),
        "log_jitter_i": float(np.percentile(flat_per_band["log_jitter_i"], 50)),
        "log_amp_delta_blr_g": float(np.percentile(flat_per_band["log_amp_delta_blr_g"], 50)),
        "log_amp_delta_blr_r": float(np.percentile(flat_per_band["log_amp_delta_blr_r"], 50)),
        "log_amp_delta_blr_i": float(np.percentile(flat_per_band["log_amp_delta_blr_i"], 50)),
    }

    h5_path = tmp_path / "data" / "fake_light_curve_end_to_end.h5"
    _write_test_quasars_hdf5(h5_path, [quasar])

    df = hubble_utils.load_agn_data(
        h5_path,
        spectra_fit_csv=None,
        lc_info_csv=None,
        only_load=True,
        apply_cut=False,
        plot_path=str(tmp_path / "figures"),
    )

    assert len(df) == 1
    row = df.iloc[0]
    assert row["object_id"] == obj["object_id"]
    assert row["len_dropped_bands"] == 1
    assert np.isclose(row["t_rf_length"], obj["t_rf_length"])
    assert np.isclose(row["log_sigma_UV"], result["log_sigma_UV"])
    assert np.isclose(row["log_tau_UV_RF"], result["log_tau_UV_RF"])
    assert np.isclose(row["mags_mean_g"], obj["mags_means"][0])
    assert np.isclose(row["mags_mean_r"], obj["mags_means"][1])
    assert np.isclose(row["mags_mean_i"], obj["mags_means"][2])
