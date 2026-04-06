from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from qvc.light_curve.multiband_model_dho_blr import (
    ContiBLR_SHO_Wrapper,
    OverdampedSHOBaseQS,
    make_linear_mean_func,
)


def mag_to_relative_flux(mag):
    mag = jnp.asarray(mag, dtype=float)
    return jnp.power(10.0, -0.4 * mag)


def magerr_to_relative_fluxerr(mag, magerr):
    mag = jnp.asarray(mag, dtype=float)
    magerr = jnp.asarray(magerr, dtype=float)
    flux = mag_to_relative_flux(mag)
    return flux * (0.4 * jnp.log(10.0)) * magerr


def relative_flux_to_mag(flux):
    flux = jnp.asarray(flux, dtype=float)
    return -2.5 * jnp.log10(jnp.clip(flux, 1e-300, None))


@dataclass
class FluxHybridMultibandModel:
    X: tuple[jnp.ndarray, jnp.ndarray]
    y: jnp.ndarray
    yerr: jnp.ndarray
    n_band: int
    zero_mean: bool = False
    has_jitter: bool = True
    stability_jitter: float = 1e-6

    def __post_init__(self):
        self.t = jnp.asarray(self.X[0], dtype=float)
        self.band = jnp.asarray(self.X[1], dtype=jnp.int32)
        self.y = jnp.asarray(self.y, dtype=float)
        self.yerr = jnp.asarray(self.yerr, dtype=float)
        self.mean_func = make_linear_mean_func(self.t, zero_mean=self.zero_mean)
        self.f0_cont_band = self._compute_baseline_flux_by_band()

    def _compute_baseline_flux_by_band(self):
        baselines = []
        for band_index in range(self.n_band):
            mask = self.band == band_index
            if bool(jnp.any(mask)):
                median_mag = jnp.nanmedian(self.y[mask])
            else:
                median_mag = 0.0
            baselines.append(mag_to_relative_flux(median_mag))
        return jnp.asarray(baselines, dtype=float)

    def _continuum_kernel(self, params):
        zeros = jnp.zeros_like(jnp.asarray(params["amp_cont"]))
        base_kernel = OverdampedSHOBaseQS(
            tau_fast=jnp.asarray(params["tau_fast_band"], dtype=float),
            tau_slow=jnp.asarray(params["tau_slow_band"], dtype=float),
        )
        return ContiBLR_SHO_Wrapper(
            kernel=base_kernel,
            params={
                "amp_cont": jnp.asarray(params["amp_cont"], dtype=float),
                "amp_bc": zeros,
                "amp_blr": zeros,
                "amp_blr2": zeros,
                "lag_disk": jnp.asarray(params["lag_disk"], dtype=float),
                "lag_bc": zeros,
                "lag_blr": zeros,
                "lag_blr2": zeros,
            },
        )

    def build_augmented_coords(self, params, *, include_bc: bool, include_blr2: bool):
        t_obs = self.t
        b_obs = self.band

        coords = {
            "obs": (t_obs, b_obs),
        }
        order = ["obs"]

        coords["blr"] = (t_obs - jnp.asarray(params["lag_blr"], dtype=float)[b_obs], b_obs)
        order.append("blr")

        if include_blr2:
            coords["blr2"] = (t_obs - jnp.asarray(params["lag_blr2"], dtype=float)[b_obs], b_obs)
            order.append("blr2")

        if include_bc:
            coords["bc"] = (t_obs - jnp.asarray(params["lag_bc"], dtype=float)[b_obs], b_obs)
            order.append("bc")

        t_aug = jnp.concatenate([coords[key][0] for key in order])
        b_aug = jnp.concatenate([coords[key][1] for key in order])

        index = {}
        start = 0
        n_obs = t_obs.shape[0]
        for key in order:
            index[key] = slice(start, start + n_obs)
            start += n_obs

        return (t_aug, b_aug), index

    def latent_covariance(self, params, *, include_bc: bool, include_blr2: bool):
        X_aug, index = self.build_augmented_coords(
            params,
            include_bc=include_bc,
            include_blr2=include_blr2,
        )
        kernel = self._continuum_kernel(params)
        K = kernel(X_aug, X_aug)
        K = K + self.stability_jitter * jnp.eye(K.shape[0], dtype=K.dtype)
        return K, X_aug, index

    def mean_vector(self, params, X_aug):
        mean_params = {"mean": params["mean"], "poly1": params.get("poly1", 0.0)}
        return jnp.asarray(self.mean_func(mean_params, X_aug), dtype=float)

    def total_flux_and_model_mag(
        self,
        params,
        latent_cont_aug,
        X_aug,
        index,
        *,
        include_bc: bool,
        include_blr2: bool,
        f_host_band=None,
    ):
        latent_cont_aug = jnp.asarray(latent_cont_aug, dtype=float)
        mean_aug = self.mean_vector(params, X_aug)
        m_cont_aug = mean_aug + latent_cont_aug
        band_aug = jnp.asarray(X_aug[1], dtype=jnp.int32)
        f0_aug = self.f0_cont_band[band_aug]
        f_cont_aug = f0_aug * mag_to_relative_flux(m_cont_aug)
        delta_f_aug = f_cont_aug - f0_aug

        band_obs = self.band
        f0_obs = self.f0_cont_band[band_obs]
        total_flux = f0_obs + delta_f_aug[index["obs"]]

        amp_cont_obs = jnp.maximum(jnp.asarray(params["amp_cont"], dtype=float)[band_obs], 1e-12)
        amp_blr_obs = jnp.asarray(params["amp_blr"], dtype=float)[band_obs]
        total_flux = total_flux + (amp_blr_obs / amp_cont_obs) * delta_f_aug[index["blr"]]

        if include_blr2:
            amp_blr2_obs = jnp.asarray(params["amp_blr2"], dtype=float)[band_obs]
            total_flux = total_flux + (amp_blr2_obs / amp_cont_obs) * delta_f_aug[index["blr2"]]

        if include_bc:
            amp_bc_obs = jnp.asarray(params["amp_bc"], dtype=float)[band_obs]
            total_flux = total_flux + (amp_bc_obs / amp_cont_obs) * delta_f_aug[index["bc"]]

        if f_host_band is not None:
            total_flux = total_flux + jnp.asarray(f_host_band, dtype=float)[band_obs]

        positive_flux = total_flux > 1e-12
        model_mag = relative_flux_to_mag(jnp.clip(total_flux / f0_obs, 1e-12, None))

        return {
            "total_flux": total_flux,
            "positive_flux": positive_flux,
            "model_mag": model_mag,
            "f0_cont_band": self.f0_cont_band,
            "m_cont_aug": m_cont_aug,
        }


def make_multiband_dho_blr_flux_model(
    X,
    y,
    yerr,
    n_band=None,
    *,
    zero_mean=False,
    has_jitter=True,
):
    if n_band is None:
        n_band = int(jnp.max(jnp.asarray(X[1], dtype=jnp.int32))) + 1

    return FluxHybridMultibandModel(
        X=X,
        y=y,
        yerr=yerr,
        n_band=n_band,
        zero_mean=zero_mean,
        has_jitter=has_jitter,
    )


__all__ = [
    "FluxHybridMultibandModel",
    "mag_to_relative_flux",
    "magerr_to_relative_fluxerr",
    "make_multiband_dho_blr_flux_model",
    "relative_flux_to_mag",
]
