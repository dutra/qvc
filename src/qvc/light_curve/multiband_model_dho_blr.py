import jax
import jax.numpy as jnp
from tinygp.helpers import JAXArray
from tinygp.kernels import quasisep as qs

from eztaox.models import MultiVarModel


def _safe_pos(x, eps=1e-12):
    return jnp.maximum(x, eps)


def _mag_from_relflux_scale(dtype=None):
    return jnp.asarray(2.5 / jnp.log(10.0), dtype=dtype)


def mag_residual_to_relative_flux(y_mag):
    y_mag = jnp.asarray(y_mag, dtype=float)
    return jnp.power(10.0, -0.4 * y_mag) - 1.0


def magerr_residual_to_relative_fluxerr(y_mag, yerr_mag):
    y_mag = jnp.asarray(y_mag, dtype=float)
    yerr_mag = jnp.asarray(yerr_mag, dtype=float)
    deriv = (0.4 * jnp.log(10.0)) * jnp.power(10.0, -0.4 * y_mag)
    return jnp.abs(deriv) * yerr_mag


def relative_flux_to_mag_residual(rel_flux):
    rel_flux = jnp.asarray(rel_flux, dtype=float)
    return -2.5 * jnp.log10(_safe_pos(1.0 + rel_flux))


def relative_flux_std_to_mag_std(rel_flux, rel_flux_std):
    rel_flux = jnp.asarray(rel_flux, dtype=float)
    rel_flux_std = jnp.asarray(rel_flux_std, dtype=float)
    scale = _mag_from_relflux_scale(rel_flux.dtype) / _safe_pos(1.0 + rel_flux)
    return scale * rel_flux_std


class OverdampedSHOBaseQS(qs.Quasisep):
    """Per-band overdamped-SHO latent driver with shared-noise fast/slow blocks."""

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

    def design_matrix(self):
        tau_fast, tau_slow = self._ordered_taus()
        lam_fast = 1.0 / tau_fast
        lam_slow = 1.0 / tau_slow
        A_fast = -jnp.diag(lam_fast)
        A_slow = -jnp.diag(lam_slow)
        zero = jnp.zeros_like(A_fast)
        return jnp.block([[A_fast, zero], [zero, A_slow]])

    def stationary_covariance(self):
        def _P(tau):
            tau = _safe_pos(tau)
            ti = tau[:, None]
            tj = tau[None, :]
            P = 2.0 * jnp.sqrt(ti * tj) / jnp.maximum(ti + tj, 1e-12)
            return 0.5 * (P + P.T)

        tau_fast, tau_slow = self._ordered_taus()
        P_fast = _P(tau_fast)
        P_slow = _P(tau_slow)
        zero = jnp.zeros_like(P_fast)
        return jnp.block([[P_fast, zero], [zero, P_slow]])

    def observation_model(self, X: JAXArray) -> JAXArray:
        _t, b = X
        b = jnp.asarray(b, dtype=jnp.int32)
        B = self._B()
        tau_fast, tau_slow = self._ordered_taus()
        denom = jnp.maximum(tau_slow[b] - tau_fast[b], 1e-12)
        c_fast = -tau_fast[b] / denom
        c_slow = tau_slow[b] / denom

        h = jnp.zeros(2 * B, dtype=tau_fast.dtype)
        h = h.at[b].set(c_fast)
        h = h.at[B + b].set(c_slow)
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
    """One-sided PSD for a quasiseparable kernel with multiband observations."""

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

    def psd(self, params, omega, b: int = 0, sigma_n2: float = 0.0):
        gp, _ = self._build_gp(params)
        return qs_psd(kernel=gp.kernel, omega=omega, b=b, sigma_n2=sigma_n2)


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
        poly1 = params["poly1"] if "poly1" in params else 0.0
        time_scaled = (X[0] - t_center) / t_std
        return mean + poly1 * time_scaled

    return mean_func


def make_multiband_dho_blr_model(
    X,
    y,
    yerr,
    n_band=None,
    *,
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
        zero_mean=zero_mean,
        has_jitter=has_jitter,
        has_lag=False,
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
    "make_multiband_dho_blr_relative_flux_model",
    "qs_psd",
    "relative_flux_std_to_mag_std",
    "relative_flux_to_mag_residual",
]
