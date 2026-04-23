from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from tinygp.helpers import JAXArray
from tinygp.kernels import quasisep as qs
import tinygp.kernels.quasisep as tkq
from tinygp import GaussianProcess

from eztaox.kernels import direct, quasisep
from eztaox.models import MultiVarModel


def _safe_pos(x, eps=1e-12):
    return jnp.maximum(x, eps)


def _mag_from_relflux_scale(dtype=None):
    return jnp.asarray(2.5 / jnp.log(10.0), dtype=dtype)


def stabilize_total_flux_ratio(total_flux_ratio, *, min_ratio=1e-12, softness=0.0):
    total_flux_ratio = jnp.asarray(total_flux_ratio, dtype=float)
    min_ratio = jnp.asarray(min_ratio, dtype=total_flux_ratio.dtype)
    softness = jnp.asarray(softness, dtype=total_flux_ratio.dtype)
    hard_floor = jnp.maximum(total_flux_ratio, min_ratio)
    softness_safe = jnp.maximum(softness, jnp.asarray(1e-12, dtype=total_flux_ratio.dtype))
    scaled = (total_flux_ratio - min_ratio) / softness_safe
    soft_floor = min_ratio + softness_safe * jax.nn.softplus(scaled)
    return jnp.where(softness > 0.0, soft_floor, hard_floor)


def mag_residual_to_relative_flux(y_mag):
    y_mag = jnp.asarray(y_mag, dtype=float)
    return jnp.power(10.0, -0.4 * y_mag) - 1.0


def magerr_residual_to_relative_fluxerr(y_mag, yerr_mag):
    y_mag = jnp.asarray(y_mag, dtype=float)
    yerr_mag = jnp.asarray(yerr_mag, dtype=float)
    deriv = (0.4 * jnp.log(10.0)) * jnp.power(10.0, -0.4 * y_mag)
    return jnp.abs(deriv) * yerr_mag


def relative_flux_to_mag_residual(rel_flux, *, min_total_flux_ratio=1e-12, floor_softness=0.0):
    rel_flux = jnp.asarray(rel_flux, dtype=float)
    total_flux_ratio = stabilize_total_flux_ratio(
        1.0 + rel_flux,
        min_ratio=min_total_flux_ratio,
        softness=floor_softness,
    )
    return -2.5 * jnp.log10(total_flux_ratio)


def relative_flux_std_to_mag_std(
    rel_flux,
    rel_flux_std,
    *,
    min_total_flux_ratio=1e-12,
    floor_softness=0.0,
):
    rel_flux = jnp.asarray(rel_flux, dtype=float)
    rel_flux_std = jnp.asarray(rel_flux_std, dtype=float)
    total_flux_ratio = stabilize_total_flux_ratio(
        1.0 + rel_flux,
        min_ratio=min_total_flux_ratio,
        softness=floor_softness,
    )
    scale = _mag_from_relflux_scale(rel_flux.dtype) / total_flux_ratio
    return scale * rel_flux_std


def _interp_relflux_basis(grid_t, grid_y, query_t):
    grid_t = jnp.asarray(grid_t, dtype=float)
    grid_y = jnp.asarray(grid_y, dtype=float)
    query_t = jnp.asarray(query_t, dtype=float)
    return jnp.interp(query_t, grid_t, grid_y, left=0.0, right=0.0)


class OverdampedSHOBaseQS(qs.Quasisep):
    """Exact single-driver overdamped-SHO latent driver with bandwise fast/slow poles."""

    tau_fast: jnp.ndarray
    tau_slow: jnp.ndarray

    def coord_to_sortable(self, X):
        t, b = X
        return t + 1e-9 * jnp.asarray(b, dtype=jnp.int32)

    def _ordered_taus(self):
        raw_fast = _safe_pos(jnp.asarray(self.tau_fast))
        raw_slow = _safe_pos(jnp.asarray(self.tau_slow))
        tau_fast = jnp.minimum(raw_fast, raw_slow)
        tau_slow = jnp.maximum(raw_fast, raw_slow)
        tau_slow = jnp.maximum(tau_slow, tau_fast * (1.0 + 1e-6))
        return tau_fast, tau_slow

    def _B(self):
        return self.tau_fast.shape[0]

    def _driver_loading(self):
        tau_fast, tau_slow = self._ordered_taus()
        return jnp.sqrt(2.0 / jnp.maximum(tau_fast + tau_slow, 1e-12))

    def _obs_scale(self):
        tau_fast, tau_slow = self._ordered_taus()
        return (tau_fast + tau_slow) / jnp.maximum(tau_slow - tau_fast, 1e-12)

    def design_matrix(self):
        tau_fast, tau_slow = self._ordered_taus()
        lam_fast = 1.0 / tau_fast
        lam_slow = 1.0 / tau_slow
        A_fast = -jnp.diag(lam_fast)
        A_slow = -jnp.diag(lam_slow)
        zero = jnp.zeros_like(A_fast)
        return jnp.block([[A_fast, zero], [zero, A_slow]])

    def stationary_covariance(self):
        tau_fast, tau_slow = self._ordered_taus()
        tau_all = jnp.concatenate([tau_fast, tau_slow])
        beta = self._driver_loading()
        beta_all = jnp.concatenate([beta, beta])
        ti = tau_all[:, None]
        tj = tau_all[None, :]
        bi = beta_all[:, None]
        bj = beta_all[None, :]
        P = (bi * bj) * (ti * tj) / jnp.maximum(ti + tj, 1e-12)
        return 0.5 * (P + P.T)

    def observation_model(self, X: JAXArray) -> JAXArray:
        _t, b = X
        b = jnp.asarray(b, dtype=jnp.int32)
        B = self._B()
        obs_scale = self._obs_scale()

        h = jnp.zeros(2 * B, dtype=obs_scale.dtype)
        h = h.at[b].set(obs_scale[b])
        h = h.at[B + b].set(-obs_scale[b])
        return h

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        t1, _ = X1
        t2, _ = X2
        dt = t2 - t1
        tau_fast, tau_slow = self._ordered_taus()
        Phi_fast = jnp.diag(jnp.exp(-dt / tau_fast))
        Phi_slow = jnp.diag(jnp.exp(-dt / tau_slow))
        zero = jnp.zeros_like(Phi_fast)
        return jnp.block([[Phi_fast, zero], [zero, Phi_slow]])


class ContiBLR_SHO_Wrapper(qs.Wrapper):
    """Multiband wrapper around a shared latent overdamped-SHO base kernel."""

    params: dict[str, JAXArray]

    def coord_to_sortable(self, X):
        t, b = X
        return t + 1e-9 * jnp.asarray(b, dtype=jnp.int32)

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel.transition_matrix(X1, X2)

    def _lagged_obs(self, b, lag):
        h0 = self.kernel.observation_model((0.0, b))
        phi = self.kernel.transition_matrix((0.0, b), (lag, b))
        return h0 @ phi

    def observation_model(self, X: JAXArray) -> JAXArray:
        _t, b = X
        b = jnp.asarray(b, dtype=jnp.int32)

        amp_cont_all = jnp.asarray(self.params["amp_cont"])
        lag_disk_all = jnp.asarray(self.params["lag_disk"])
        amp_cont = _safe_pos(amp_cont_all)[b]
        amp_bc = jnp.maximum(
            jnp.asarray(self.params.get("amp_bc", jnp.zeros_like(amp_cont_all))),
            0.0,
        )[b]
        amp_blr = _safe_pos(jnp.asarray(self.params["amp_blr"]))[b]
        amp_blr2 = _safe_pos(jnp.asarray(self.params["amp_blr2"]))[b]
        lag_disk = jnp.maximum(lag_disk_all, 0.0)[b]
        lag_bc = jnp.maximum(
            jnp.asarray(self.params.get("lag_bc", jnp.zeros_like(lag_disk_all))),
            0.0,
        )[b]
        lag_blr = jnp.maximum(jnp.asarray(self.params["lag_blr"]), 0.0)[b]
        lag_blr2 = jnp.maximum(jnp.asarray(self.params["lag_blr2"]), 0.0)[b]

        h_cont = self._lagged_obs(b, lag_disk)
        h_bc = self._lagged_obs(b, lag_disk + lag_bc)
        h_blr = self._lagged_obs(b, lag_disk + lag_blr)
        h_blr2 = self._lagged_obs(b, lag_disk + lag_blr2)
        return amp_cont * h_cont + amp_bc * h_bc + amp_blr * h_blr + amp_blr2 * h_blr2


def qs_psd(kernel, omega, b: int, sigma_n2: float = 0.0):
    """One-sided cyclic-frequency PSD density evaluated at angular omega."""

    A = kernel.design_matrix()
    P = kernel.stationary_covariance()
    Qc = -(A @ P + P @ A.T)
    I = jnp.eye(A.shape[0], dtype=A.dtype)
    h = kernel.observation_model((jnp.array(0.0), jnp.array(int(b))))

    def one_w(w):
        v = jnp.linalg.solve((-1j * w) * I - A.T, h)
        return (v.conj().T @ (Qc @ v)).real + sigma_n2

    return 2.0 * jax.vmap(one_w)(omega)


class ContiBLR_SHO_Model(MultiVarModel):
    """MultiVarModel with a convenience PSD method for plotting."""

    survey_idx: JAXArray | None = None
    nBand: int
    n_band: int
    t_in_bands: list[JAXArray]
    concat_inds_in_bands: list[JAXArray]

    def __init__(
        self,
        X,
        y,
        yerr,
        base_kernel,
        nBand,
        multiband_kernel=None,
        mean_func=None,
        amp_scale_func=None,
        lag_func=None,
        *,
        survey_idx=None,
        **kwargs,
    ):
        t = jnp.asarray(X[0])
        band = jnp.asarray(X[1], dtype=jnp.int32)
        y = jnp.asarray(y)
        yerr = jnp.asarray(yerr)
        inds = jnp.argsort(t)

        self.X = (t[inds], band[inds])
        self.diag = (yerr**2)[inds]
        self.y = y[inds]
        self.base_kernel_def = jax.flatten_util.ravel_pytree(base_kernel)[1]
        is_quasisep_kernel = isinstance(base_kernel, qs.Quasisep)
        self.nBand = nBand
        self.n_band = nBand

        # Some EzTaoX releases precompute per-band sorted indices with jnp.unique()
        # in MultiVarModel.__init__. That is not valid when this qvc model is built
        # from traced arrays inside NumPyro. qvc uses the simpler lag_transform()
        # below, so these fast-path caches are intentionally left empty.
        self.t_in_bands = []
        self.concat_inds_in_bands = []

        if multiband_kernel is None:
            if is_quasisep_kernel:
                multiband_kernel = quasisep.MultibandLowRank
            else:
                multiband_kernel = direct.MultibandLowRank
        self.multiband_kernel = multiband_kernel
        self.mean_func = mean_func
        self.amp_scale_func = amp_scale_func
        self.lag_func = lag_func
        self.zero_mean = kwargs.get("zero_mean", True)
        self.has_jitter = kwargs.get("has_jitter", False)
        self.has_lag = kwargs.get("has_lag", False)

        if survey_idx is None:
            self.survey_idx = None
        else:
            self.survey_idx = jnp.asarray(survey_idx, dtype=jnp.int32)[inds]

    def lag_transform(
        self, has_lag: bool, params: dict[str, JAXArray], X: JAXArray
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray]:
        if has_lag is False:
            lags = jnp.zeros(self.nBand)
        elif self.lag_func is not None:
            lags = self.lag_func(params)
        else:
            lags = self._default_lag_func(params)

        t, band = X
        new_t = t - lags[band]
        return (new_t, band), jnp.argsort(new_t)

    def my_lag_transform(
        self, X: JAXArray, has_lag: bool, params: dict[str, JAXArray]
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray]:
        return self.lag_transform(has_lag, params, X)

    def my_amp_transform(self, params: dict[str, JAXArray]) -> JAXArray:
        return jnp.log(_safe_pos(jnp.asarray(params["amp_cont"])))

    def mean_to_display(self, mean_vals):
        return mean_vals

    def prediction_to_display(self, pred_result):
        return pred_result

    def _jitter_diag(self, params, band):
        log_jitter = jnp.asarray(params["log_jitter"])
        if self.survey_idx is not None and log_jitter.ndim == 2:
            return jnp.exp(log_jitter[band, self.survey_idx]) ** 2
        return (jnp.exp(jnp.atleast_1d(log_jitter)) ** 2)[band]

    def _survey_offset_in_model_units(self, params, band):
        survey_delta_mag = params.get("survey_delta_mag")
        if survey_delta_mag is None or self.survey_idx is None:
            return jnp.zeros_like(jnp.asarray(band, dtype=float))
        survey_delta_mag = jnp.asarray(survey_delta_mag, dtype=float)
        if survey_delta_mag.ndim != 2:
            return jnp.zeros_like(jnp.asarray(band, dtype=float))
        return survey_delta_mag[band, self.survey_idx]

    def _observed_y_sorted(self, params, inds):
        band = jnp.asarray(self.X[1], dtype=jnp.int32)
        survey_offset = self._survey_offset_in_model_units(params, band)
        return (self.y - survey_offset)[inds]

    def _build_gp(
        self, params: dict[str, JAXArray]
    ) -> tuple[GaussianProcess, JAXArray]:
        log_amp_scales = self.get_amp_scale(params)
        means = partial(self.get_mean, self.zero_mean, params)

        X, inds = self.lag_transform(self.has_lag, params, self.X)
        t = X[0]
        band = X[1]

        diags = self.diag
        if self.has_jitter is True:
            diags = self.diag + self._jitter_diag(params, band)

        new_params = params.copy()
        new_params["amplitudes"] = jnp.exp(log_amp_scales)
        kernel = self.multiband_kernel(
            params=new_params,
            kernel=self.base_kernel_def(jnp.exp(new_params["log_kernel_param"])),
        )

        gp_kwargs = {
            "diag": diags[inds],
            "mean": means,
        }
        if isinstance(kernel, tkq.Quasisep):
            gp_kwargs["assume_sorted"] = True

        return (
            GaussianProcess(
                kernel,
                (t[inds], band[inds]),
                **gp_kwargs,
            ),
            inds,
        )

    def psd(self, params, omega, b: int = 0, sigma_n2: float = 0.0):
        gp, _ = self._build_gp(params)
        return qs_psd(kernel=gp.kernel, omega=omega, b=b, sigma_n2=sigma_n2)

    @eqx.filter_jit
    def log_prob(self, params: dict[str, JAXArray]) -> JAXArray:
        gp, inds = self._build_gp(params)
        return gp.log_probability(y=self._observed_y_sorted(params, inds)) + self.log_prior(params)

    def sample(self, params: dict[str, JAXArray]) -> None:
        gp, inds = self._build_gp(params)
        numpyro.sample("gp", gp.numpyro_dist(), obs=self._observed_y_sorted(params, inds))

    def aic(self, params: dict[str, JAXArray]) -> JAXArray:
        k = len(jax.flatten_util.ravel_pytree(params)[0])
        gp, inds = self._build_gp(params)
        log_likelihood = gp.log_probability(y=self._observed_y_sorted(params, inds))
        return 2 * k - 2 * log_likelihood

    def bic(self, params: dict[str, JAXArray]) -> JAXArray:
        n = self.y.size
        k = len(jax.flatten_util.ravel_pytree(params)[0])
        gp, inds = self._build_gp(params)
        log_likelihood = gp.log_probability(y=self._observed_y_sorted(params, inds))
        return jnp.log(n) * k - 2 * log_likelihood

    @eqx.filter_jit
    def pred(
        self, params: dict[str, JAXArray], X: JAXArray
    ) -> tuple[JAXArray, JAXArray]:
        new_X, _ = self.lag_transform(self.has_lag, params, X)
        gp, inds = self._build_gp(params)
        _, cond = gp.condition(self._observed_y_sorted(params, inds), new_X)
        return cond.loc, jnp.sqrt(cond.variance)


class ContiBLRRelativeFlux_SHO_Wrapper(ContiBLR_SHO_Wrapper):
    """Relative-flux QS wrapper using flux-additive prompt and delayed amplitudes."""

    def observation_model(self, X: JAXArray) -> JAXArray:
        _t, b = X
        b = jnp.asarray(b, dtype=jnp.int32)

        amp_cont_all = jnp.asarray(
            self.params["amp_cont_relflux"]
            if "amp_cont_relflux" in self.params
            else self.params["amp_cont"]
        )
        lag_disk_all = jnp.asarray(self.params["lag_disk"])
        amp_cont = _safe_pos(amp_cont_all)[b]
        amp_bc = jnp.maximum(
            jnp.asarray(
                self.params["amp_bc_relflux"]
                if "amp_bc_relflux" in self.params
                else self.params.get("amp_bc", jnp.zeros_like(amp_cont_all))
            ),
            0.0,
        )[b]
        amp_blr = _safe_pos(
            jnp.asarray(
                self.params["amp_blr_relflux"]
                if "amp_blr_relflux" in self.params
                else self.params["amp_blr"]
            )
        )[b]
        amp_blr2 = _safe_pos(
            jnp.asarray(
                self.params["amp_blr2_relflux"]
                if "amp_blr2_relflux" in self.params
                else self.params.get("amp_blr2", jnp.zeros_like(amp_cont_all))
            )
        )[b]
        lag_disk = jnp.maximum(lag_disk_all, 0.0)[b]
        lag_bc = jnp.maximum(
            jnp.asarray(self.params.get("lag_bc", jnp.zeros_like(lag_disk_all))),
            0.0,
        )[b]
        lag_blr = jnp.maximum(jnp.asarray(self.params["lag_blr"]), 0.0)[b]
        lag_blr2 = jnp.maximum(
            jnp.asarray(self.params.get("lag_blr2", jnp.zeros_like(lag_disk_all))),
            0.0,
        )[b]

        h_cont = self._lagged_obs(b, lag_disk)
        h_bc = self._lagged_obs(b, lag_disk + lag_bc)
        h_blr = self._lagged_obs(b, lag_disk + lag_blr)
        h_blr2 = self._lagged_obs(b, lag_disk + lag_blr2)
        return amp_cont * h_cont + amp_bc * h_bc + amp_blr * h_blr + amp_blr2 * h_blr2


class ContiBLRFluxLinearized_SHO_Wrapper(ContiBLRRelativeFlux_SHO_Wrapper):
    """Backward-compatible alias for the relative-flux QS wrapper."""


class ContiBLRRelativeFlux_SHO_Model(ContiBLR_SHO_Model):
    """QS model fit in relative-flux residual units with mag-display helpers."""

    def my_amp_transform(self, params: dict[str, JAXArray]) -> JAXArray:
        amp_cont_relflux = jnp.asarray(
            params["amp_cont_relflux"]
            if "amp_cont_relflux" in params
            else params["amp_cont"]
        )
        return jnp.log(_safe_pos(_mag_from_relflux_scale(amp_cont_relflux.dtype) * amp_cont_relflux))

    def _survey_offset_in_model_units(self, params, band):
        survey_delta_mag = params.get("survey_delta_mag")
        if survey_delta_mag is None or self.survey_idx is None:
            return jnp.zeros_like(jnp.asarray(band, dtype=float))
        survey_delta_mag = jnp.asarray(survey_delta_mag, dtype=float)
        if survey_delta_mag.ndim != 2:
            return jnp.zeros_like(jnp.asarray(band, dtype=float))
        return mag_residual_to_relative_flux(survey_delta_mag[band, self.survey_idx])

    def mean_to_display(self, mean_vals):
        return relative_flux_to_mag_residual(mean_vals)

    def prediction_to_display(self, pred_result):
        if len(pred_result) == 2:
            mu_rel, std_rel = pred_result
            return (
                relative_flux_to_mag_residual(mu_rel),
                relative_flux_std_to_mag_std(mu_rel, std_rel),
            )

        mu_rel, std_rel, mu_cont_rel, std_cont_rel, mu_blr_rel, std_blr_rel = pred_result
        return (
            relative_flux_to_mag_residual(mu_rel),
            relative_flux_std_to_mag_std(mu_rel, std_rel),
            relative_flux_to_mag_residual(mu_cont_rel),
            relative_flux_std_to_mag_std(mu_cont_rel, std_cont_rel),
            relative_flux_to_mag_residual(mu_blr_rel),
            relative_flux_std_to_mag_std(mu_blr_rel, std_blr_rel),
        )

    def psd(self, params, omega, b: int = 0, sigma_n2: float = 0.0):
        gp, _ = self._build_gp(params)
        relflux_psd = qs_psd(kernel=gp.kernel, omega=omega, b=b, sigma_n2=sigma_n2)
        scale = _mag_from_relflux_scale(relflux_psd.dtype)
        return relflux_psd * scale**2


class ContiBLRFluxLinearized_SHO_Model(ContiBLRRelativeFlux_SHO_Model):
    """Backward-compatible alias for the relative-flux QS model."""


class TwoStageFluxMixDisplayModel:
    """Display/prediction wrapper for the staged flux-mixing approximation."""

    def __init__(
        self,
        *,
        continuum_model,
        basis_grid_t,
        basis_relflux_norm,
        t_ref,
        zero_mean=False,
        prediction_samples=None,
        min_total_flux_ratio=1e-12,
        floor_softness=0.0,
    ):
        self.continuum_model = continuum_model
        self.basis_grid_t = jnp.asarray(basis_grid_t, dtype=float)
        self.basis_relflux_norm = jnp.asarray(basis_relflux_norm, dtype=float)
        self.mean_func = make_linear_mean_func(t_ref, zero_mean=zero_mean)
        self.zero_mean = zero_mean
        self.prediction_samples = prediction_samples
        self.min_total_flux_ratio = float(min_total_flux_ratio)
        self.floor_softness = float(floor_softness)
        self.X = continuum_model.X
        self.has_lag = False
        self.nBand = getattr(continuum_model, "nBand", int(self.basis_relflux_norm.shape[0]))

    def prediction_to_display(self, pred_result):
        return pred_result

    def my_amp_transform(self, params: dict[str, JAXArray]) -> JAXArray:
        return jnp.log(_safe_pos(jnp.asarray(params["amp_cont"])))

    def lag_transform(
        self, has_lag: bool, params: dict[str, JAXArray], X: JAXArray
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray]:
        del has_lag, params
        t = jnp.asarray(X[0], dtype=float)
        band = jnp.asarray(X[1], dtype=jnp.int32)
        return (t, band), jnp.arange(t.shape[0], dtype=jnp.int32)

    def my_lag_transform(
        self, X: JAXArray, has_lag: bool, params: dict[str, JAXArray]
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray]:
        return self.lag_transform(has_lag, params, X)

    def _predict_mean(self, params: dict[str, JAXArray], X: JAXArray) -> JAXArray:
        t_query = jnp.asarray(X[0], dtype=float)
        band_idx = jnp.asarray(X[1], dtype=jnp.int32)
        mean_vals = self.mean_func(params, (t_query, band_idx))

        amp_cont_all = jnp.asarray(params["amp_cont"], dtype=float)
        amp_blr_all = jnp.asarray(params.get("amp_blr", jnp.zeros_like(amp_cont_all)), dtype=float)
        amp_bc_all = jnp.asarray(params.get("amp_bc", jnp.zeros_like(amp_cont_all)), dtype=float)
        lag_blr_all = jnp.asarray(params.get("lag_blr", jnp.zeros_like(amp_cont_all)), dtype=float)
        lag_bc_all = jnp.asarray(params.get("lag_bc", jnp.zeros_like(amp_cont_all)), dtype=float)

        def one_point(tt, bb):
            basis_band = self.basis_relflux_norm[bb]
            prompt = amp_cont_all[bb] * _interp_relflux_basis(self.basis_grid_t, basis_band, tt)
            blr = amp_blr_all[bb] * _interp_relflux_basis(
                self.basis_grid_t,
                basis_band,
                tt - jnp.maximum(lag_blr_all[bb], 0.0),
            )
            bc = amp_bc_all[bb] * _interp_relflux_basis(
                self.basis_grid_t,
                basis_band,
                tt - jnp.maximum(lag_bc_all[bb], 0.0),
            )
            return prompt + blr + bc

        rel_flux = jax.vmap(one_point)(t_query, band_idx)
        return mean_vals + relative_flux_to_mag_residual(
            rel_flux,
            min_total_flux_ratio=self.min_total_flux_ratio,
            floor_softness=self.floor_softness,
        )

    def pred(
        self, params: dict[str, JAXArray], X: JAXArray
    ) -> tuple[JAXArray, JAXArray]:
        mean_pred = self._predict_mean(params, X)
        if not self.prediction_samples:
            return mean_pred, jnp.zeros_like(mean_pred)

        n_draws = len(next(iter(self.prediction_samples.values())))
        if n_draws == 0:
            return mean_pred, jnp.zeros_like(mean_pred)

        pred_stack = []
        for i in range(n_draws):
            sample_params = {
                key: jnp.asarray(value[i])
                for key, value in self.prediction_samples.items()
            }
            pred_stack.append(np.asarray(self._predict_mean(sample_params, X), dtype=float))
        pred_stack = np.asarray(pred_stack, dtype=float)
        std_pred = np.std(pred_stack, axis=0, ddof=1 if pred_stack.shape[0] > 1 else 0)
        return mean_pred, jnp.asarray(std_pred, dtype=float)

    def psd(self, params, omega, b: int = 0, sigma_n2: float = 0.0):
        amp_cont = jnp.asarray(params["amp_cont"], dtype=float)
        zeros = jnp.zeros_like(amp_cont)
        psd_params = dict(params)
        psd_params["amp_blr"] = zeros
        psd_params["amp_blr2"] = zeros
        psd_params["amp_bc"] = zeros
        psd_params["lag_blr"] = zeros
        psd_params["lag_blr2"] = zeros
        psd_params["lag_bc"] = zeros
        return self.continuum_model.psd(psd_params, omega, b=b, sigma_n2=sigma_n2)


def make_linear_mean_func(t_ref, zero_mean=False):
    """Return a simple per-band linear mean function."""

    t_ref = jnp.asarray(t_ref)
    t_center = 0.5 * (jnp.min(t_ref) + jnp.max(t_ref))
    t_std = jnp.maximum(jnp.std(t_ref), 1e-6)

    def mean_func(params, X):
        if zero_mean:
            return jnp.zeros_like(X[0], dtype=t_ref.dtype)

        band_idx = jnp.asarray(X[1], dtype=jnp.int32)
        mean = params["mean"][band_idx] if "mean" in params else 0.0
        if "linear_trend_band" in params:
            linear_trend = jnp.asarray(params["linear_trend_band"], dtype=t_ref.dtype)[band_idx]
        else:
            linear_trend = params["linear_trend"] if "linear_trend" in params else 0.0
            if "linear_trend_band_offset" in params:
                linear_trend = linear_trend + jnp.asarray(
                    params["linear_trend_band_offset"],
                    dtype=t_ref.dtype,
                )[band_idx]
        time_scaled = (X[0] - t_center) / t_std
        return mean + linear_trend * time_scaled

    return mean_func


def make_multiband_dho_blr_model(
    X,
    y,
    yerr,
    n_band=None,
    *,
    survey_idx=None,
    zero_mean=False,
    has_jitter=True,
):
    """Construct the multiband DHO+BLR model."""

    if n_band is None:
        n_band = int(jnp.max(jnp.asarray(X[1], dtype=jnp.int32))) + 1

    t = jnp.asarray(X[0])

    def amp_scale_func(_params):
        return jnp.zeros(n_band)

    return ContiBLR_SHO_Model(
        X,
        y,
        yerr,
        base_kernel=OverdampedSHOBaseQS(
            tau_fast=jnp.full(n_band, 10.0),
            tau_slow=jnp.full(n_band, 100.0),
        ),
        nBand=n_band,
        multiband_kernel=ContiBLR_SHO_Wrapper,
        mean_func=make_linear_mean_func(t, zero_mean=zero_mean),
        amp_scale_func=amp_scale_func,
        survey_idx=survey_idx,
        zero_mean=zero_mean,
        has_jitter=has_jitter,
        has_lag=False,
    )


def make_multiband_dho_blr_flux_linearized_model(
    X,
    y,
    yerr,
    n_band=None,
    *,
    survey_idx=None,
    baseline_flux_by_band=None,
    zero_mean=False,
    has_jitter=True,
):
    """Construct the relative-flux multiband DHO+BLR QS model."""

    return make_multiband_dho_blr_relative_flux_model(
        X,
        y,
        yerr,
        n_band=n_band,
        survey_idx=survey_idx,
        baseline_flux_by_band=baseline_flux_by_band,
        zero_mean=zero_mean,
        has_jitter=has_jitter,
    )


def make_multiband_dho_blr_relative_flux_model(
    X,
    y,
    yerr,
    n_band=None,
    *,
    survey_idx=None,
    baseline_flux_by_band=None,
    zero_mean=False,
    has_jitter=True,
):
    """Construct the relative-flux multiband DHO+BLR model."""

    if n_band is None:
        n_band = int(jnp.max(jnp.asarray(X[1], dtype=jnp.int32))) + 1

    t = jnp.asarray(X[0])

    def amp_scale_func(_params):
        return jnp.zeros(n_band)

    return ContiBLRRelativeFlux_SHO_Model(
        X,
        y,
        yerr,
        base_kernel=OverdampedSHOBaseQS(
            tau_fast=jnp.full(n_band, 10.0),
            tau_slow=jnp.full(n_band, 100.0),
        ),
        nBand=n_band,
        multiband_kernel=ContiBLRRelativeFlux_SHO_Wrapper,
        mean_func=make_linear_mean_func(t, zero_mean=zero_mean),
        amp_scale_func=amp_scale_func,
        survey_idx=survey_idx,
        zero_mean=zero_mean,
        has_jitter=has_jitter,
        has_lag=False,
    )


def make_multiband_dho_blr_fluxmix_fast_display_model(
    *,
    continuum_model,
    basis_grid_t,
    basis_relflux_norm,
    t_ref,
    zero_mean=False,
    prediction_samples=None,
    min_total_flux_ratio=1e-12,
    floor_softness=0.0,
):
    """Construct the staged flux-mixing display model."""

    return TwoStageFluxMixDisplayModel(
        continuum_model=continuum_model,
        basis_grid_t=basis_grid_t,
        basis_relflux_norm=basis_relflux_norm,
        t_ref=t_ref,
        zero_mean=zero_mean,
        prediction_samples=prediction_samples,
        min_total_flux_ratio=min_total_flux_ratio,
        floor_softness=floor_softness,
    )


__all__ = [
    "ContiBLRRelativeFlux_SHO_Model",
    "ContiBLRRelativeFlux_SHO_Wrapper",
    "ContiBLR_SHO_Model",
    "ContiBLRFluxLinearized_SHO_Model",
    "ContiBLRFluxLinearized_SHO_Wrapper",
    "ContiBLR_SHO_Wrapper",
    "OverdampedSHOBaseQS",
    "mag_residual_to_relative_flux",
    "magerr_residual_to_relative_fluxerr",
    "make_linear_mean_func",
    "make_multiband_dho_blr_model",
    "make_multiband_dho_blr_flux_linearized_model",
    "make_multiband_dho_blr_fluxmix_fast_display_model",
    "make_multiband_dho_blr_relative_flux_model",
    "qs_psd",
    "stabilize_total_flux_ratio",
    "relative_flux_std_to_mag_std",
    "relative_flux_to_mag_residual",
    "TwoStageFluxMixDisplayModel",
]
