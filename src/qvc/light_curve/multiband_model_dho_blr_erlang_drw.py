"""Causal Erlang model with a DRW-style CARMA(2,1) continuum."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import expm
from tinygp.solvers.quasisep.core import DiagQSM, StrictLowerTriQSM, SymmQSM

from qvc.light_curve.dho_drw_parameterization import IntegratedTimescaleDHOBaseQS
from qvc.light_curve.multiband_dho_core import make_linear_mean_func
from qvc.light_curve.multiband_model_dho_blr_erlang import (
    DEFAULT_ERLANG_ORDER,
    ContiBLRErlangRelativeFluxModel,
    ErlangResponseDHOQS,
)

POSITIVE_FLUX_N_SIGMA = 4.0
POSITIVE_FLUX_MARGIN_SOFTNESS = 0.01
_CRITICAL_DAMPING_X = 1e-4
_PHI_SERIES_RADIUS = 4.0
_PHI_SERIES_TERMS = 32
_CRITICAL_TAYLOR_TERMS = 2


def _complex_pair_multiply(a_real, a_imag, b_real, b_imag):
    return (
        a_real * b_real - a_imag * b_imag,
        a_real * b_imag + a_imag * b_real,
    )


def _sinc(x):
    """Stable ``sin(x) / x`` with finite derivatives at the origin."""

    x = jnp.asarray(x)
    use_series = jnp.abs(x) <= 1e-4
    x_safe = jnp.where(use_series, 1.0, x)
    regular = jnp.sin(x_safe) / x_safe
    x2 = x**2
    series = 1.0 - x2 / math.factorial(3) + x2**2 / math.factorial(5)
    return jnp.where(use_series, series, regular)


def _scaled_phi_pairs(z_real, z_imag, decay, max_order):
    """Return ``exp(-decay) * phi_j(z)`` for ``j=1..max_order``.

    Complex values are represented by real/imaginary pairs so the common
    overdamped and critical paths stay real-valued.  The power-series branch
    avoids cancellation near continuum/response pole collisions; the closed
    form is efficient and stable away from them.
    """

    z_real = jnp.asarray(z_real)
    z_imag = jnp.asarray(z_imag, dtype=z_real.dtype)
    decay = jnp.asarray(decay, dtype=z_real.dtype)
    use_series = jnp.hypot(z_real, z_imag) <= _PHI_SERIES_RADIUS

    series_real = jnp.where(use_series, z_real, 0.0)
    series_imag = jnp.where(use_series, z_imag, 0.0)
    closed_real = jnp.where(use_series, 1.0, z_real)
    closed_imag = jnp.where(use_series, 0.0, z_imag)
    exp_decay = jnp.exp(-decay)
    exp_real = jnp.exp(closed_real - decay) * jnp.cos(closed_imag)
    exp_imag = jnp.exp(closed_real - decay) * jnp.sin(closed_imag)

    coefficients = jnp.asarray(
        [
            [
                1.0 / math.factorial(power + order)
                for order in range(1, int(max_order) + 1)
            ]
            for power in range(_PHI_SERIES_TERMS)
        ],
        dtype=z_real.dtype,
    )
    series_shape = series_real.shape + (int(max_order),)
    series_value_real = jnp.broadcast_to(coefficients[-1], series_shape)
    series_value_imag = jnp.zeros(series_shape, dtype=z_real.dtype)

    def horner_step(index, state):
        value_real, value_imag = state
        power = _PHI_SERIES_TERMS - 2 - index
        value_real, value_imag = _complex_pair_multiply(
            value_real,
            value_imag,
            series_real[..., None],
            series_imag[..., None],
        )
        return value_real + coefficients[power], value_imag

    series_value_real, series_value_imag = jax.lax.fori_loop(
        0,
        _PHI_SERIES_TERMS - 1,
        horner_step,
        (series_value_real, series_value_imag),
    )
    series_value_real = exp_decay[..., None] * series_value_real
    series_value_imag = exp_decay[..., None] * series_value_imag

    power_real = jnp.ones_like(closed_real)
    power_imag = jnp.zeros_like(closed_imag)
    partial_real = jnp.ones_like(closed_real)
    partial_imag = jnp.zeros_like(closed_imag)
    values_real = []
    values_imag = []

    for order in range(1, int(max_order) + 1):
        power_real, power_imag = _complex_pair_multiply(
            power_real,
            power_imag,
            closed_real,
            closed_imag,
        )
        numerator_real = exp_real - exp_decay * partial_real
        numerator_imag = exp_imag - exp_decay * partial_imag
        denominator = power_real**2 + power_imag**2
        closed_value_real = (
            numerator_real * power_real + numerator_imag * power_imag
        ) / denominator
        closed_value_imag = (
            numerator_imag * power_real - numerator_real * power_imag
        ) / denominator

        values_real.append(
            jnp.where(
                use_series,
                series_value_real[..., order - 1],
                closed_value_real,
            )
        )
        values_imag.append(
            jnp.where(
                use_series,
                series_value_imag[..., order - 1],
                closed_value_imag,
            )
        )

        partial_real = partial_real + power_real / math.factorial(order)
        partial_imag = partial_imag + power_imag / math.factorial(order)

    return jnp.stack(values_real, axis=-1), jnp.stack(values_imag, axis=-1)


def _critical_response_derivative(phi_values, decay, order, derivative):
    """Derivative of the dimensionless scalar Erlang response at criticality."""

    result = jnp.zeros_like(decay)
    for offset in range(int(derivative) + 1):
        rising = math.factorial(order + offset - 1) / math.factorial(order - 1)
        coefficient = (
            (-1.0) ** offset
            * math.comb(derivative, offset)
            * rising
        )
        result = result + coefficient * phi_values[..., order + offset - 1]
    return jnp.power(decay, order) * result


def _integrated_erlang_scatter_indices(n_band, order):
    """Static indices for assembling band-local transitions into global state."""

    n_band = int(n_band)
    order = int(order)
    n_driver = 2 * n_band
    rows = []
    columns = []
    for band in range(n_band):
        rows.extend((band, band, n_band + band, n_band + band))
        columns.extend((band, n_band + band, band, n_band + band))
    for band in range(n_band):
        for row in range(order):
            for column in range(row + 1):
                rows.append(n_driver + band * order + row)
                columns.append(n_driver + band * order + column)
    for band in range(n_band):
        for row in range(order):
            rows.append(n_driver + band * order + row)
            columns.append(band)
    for band in range(n_band):
        for row in range(order):
            rows.append(n_driver + band * order + row)
            columns.append(n_band + band)
    return np.asarray(rows, dtype=np.int32), np.asarray(columns, dtype=np.int32)


def _integrated_erlang_transitions(kernel, dt):
    """Exact batched CARMA(2,1)-plus-Erlang state transitions."""

    dt = jnp.atleast_1d(jnp.asarray(dt))
    omega0 = jnp.asarray(kernel.carma_omega0)
    damping = jnp.asarray(kernel.carma_damping, dtype=omega0.dtype)
    obs_position = jnp.asarray(kernel.carma_obs_position, dtype=omega0.dtype)
    obs_velocity = jnp.asarray(kernel.carma_obs_velocity, dtype=omega0.dtype)
    n_band = int(omega0.shape[0])
    order = int(kernel.order)
    n_state = n_band * (2 + order)

    time = dt[:, None]
    omega0 = omega0[None, :]
    half_damping = 0.5 * damping[None, :]
    discriminant = half_damping**2 - omega0**2
    response_rate = kernel._response_rates()[None, :].astype(omega0.dtype)
    response_decay = response_rate * time
    centered_decay = (response_rate - half_damping) * time
    critical_x = discriminant * time**2
    use_critical = jnp.abs(critical_x) <= _CRITICAL_DAMPING_X
    use_overdamped = (discriminant >= 0.0) & ~use_critical
    use_underdamped = (discriminant < 0.0) & ~use_critical

    # Continuum transition.  Stable real roots avoid overflow/cancellation in
    # the strongly overdamped regime; sine/cosine covers the underdamped one.
    over_root = jnp.sqrt(jnp.maximum(discriminant, 1e-300))
    slow_rate = omega0**2 / (half_damping + over_root)
    fast_rate = half_damping + over_root
    exp_slow = jnp.exp(-slow_rate * time)
    exp_fast = jnp.exp(-fast_rate * time)
    over_c = 0.5 * (exp_slow + exp_fast)
    over_r = over_root * time
    over_r_safe = jnp.where(use_overdamped, over_r, 1.0)
    over_s = time * (exp_slow - exp_fast) / (2.0 * over_r_safe)

    under_root = jnp.sqrt(jnp.maximum(-discriminant, 1e-300))
    under_angle = under_root * time
    under_angle_safe = jnp.where(use_underdamped, under_angle, 1.0)
    under_exp = jnp.exp(-half_damping * time)
    under_c = under_exp * jnp.cos(under_angle)
    under_s = time * under_exp * _sinc(under_angle)

    critical_exp = jnp.exp(-half_damping * time)
    critical_c = critical_exp * (
        1.0
        + critical_x / math.factorial(2)
        + critical_x**2 / math.factorial(4)
        + critical_x**3 / math.factorial(6)
    )
    critical_s = time * critical_exp * (
        1.0
        + critical_x / math.factorial(3)
        + critical_x**2 / math.factorial(5)
        + critical_x**3 / math.factorial(7)
    )
    continuum_c = jnp.where(discriminant >= 0.0, over_c, under_c)
    continuum_s = jnp.where(discriminant >= 0.0, over_s, under_s)
    continuum_c = jnp.where(use_critical, critical_c, continuum_c)
    continuum_s = jnp.where(use_critical, critical_s, continuum_s)
    e00 = continuum_c + half_damping * continuum_s
    e01 = continuum_s
    e10 = -(omega0**2) * continuum_s
    e11 = continuum_c - half_damping * continuum_s

    # Evaluate two regular modes plus the critical center in one pair batch.
    # The first slot is either the slow real mode or the positive-frequency
    # complex mode, so the inactive damping regime incurs no extra phi work.
    max_phi_order = order + 2 * _CRITICAL_TAYLOR_TERMS + 1
    mode_real = jnp.stack(
        [
            jnp.where(
                use_overdamped,
                centered_decay + over_r,
                centered_decay,
            ),
            jnp.where(
                use_overdamped,
                centered_decay - over_r,
                centered_decay,
            ),
            centered_decay,
        ],
        axis=-1,
    )
    mode_imag = jnp.stack(
        [
            jnp.zeros_like(centered_decay),
            jnp.zeros_like(centered_decay),
            jnp.zeros_like(centered_decay),
        ],
        axis=-1,
    )
    mode_imag = mode_imag.at[..., 0].set(
        jnp.where(use_underdamped, under_angle, 0.0)
    )
    mode_decay = jnp.broadcast_to(response_decay[..., None], mode_real.shape)
    phi_real, phi_imag = _scaled_phi_pairs(
        mode_real,
        mode_imag,
        mode_decay,
        max_phi_order,
    )
    response_powers = jnp.stack(
        [jnp.power(response_decay, index) for index in range(1, order + 1)],
        axis=-1,
    )
    over_plus = phi_real[..., 0, :order] * response_powers
    over_minus = phi_real[..., 1, :order] * response_powers
    over_u = 0.5 * (over_plus + over_minus)
    over_w = (over_plus - over_minus) / (2.0 * over_r_safe[..., None])

    under_u = phi_real[..., 0, :order] * response_powers
    under_w = (
        phi_imag[..., 0, :order]
        * response_powers
        / under_angle_safe[..., None]
    )

    critical_phi = phi_real[..., 2, :]
    critical_u = []
    critical_w = []
    for response_order in range(1, order + 1):
        derivatives = [
            _critical_response_derivative(
                critical_phi,
                response_decay,
                response_order,
                derivative,
            )
            for derivative in range(2 * _CRITICAL_TAYLOR_TERMS + 2)
        ]
        critical_u.append(
            derivatives[0]
            + critical_x * derivatives[2] / math.factorial(2)
            + critical_x**2 * derivatives[4] / math.factorial(4)
        )
        critical_w.append(
            derivatives[1]
            + critical_x * derivatives[3] / math.factorial(3)
            + critical_x**2 * derivatives[5] / math.factorial(5)
        )
    critical_u = jnp.stack(critical_u, axis=-1)
    critical_w = jnp.stack(critical_w, axis=-1)

    response_u = jnp.where(
        (discriminant >= 0.0)[..., None],
        over_u,
        under_u,
    )
    response_w = jnp.where(
        (discriminant >= 0.0)[..., None],
        over_w,
        under_w,
    )
    response_u = jnp.where(use_critical[..., None], critical_u, response_u)
    response_w = jnp.where(use_critical[..., None], critical_w, response_w)
    response_v = time[..., None] * response_w

    obs_position = obs_position[None, :, None]
    obs_velocity = obs_velocity[None, :, None]
    half_damping = half_damping[..., None]
    omega_squared = (omega0**2)[..., None]
    h0 = (
        obs_position * (response_u + half_damping * response_v)
        - obs_velocity * omega_squared * response_v
    )
    h1 = (
        obs_position * response_v
        + obs_velocity * (response_u - half_damping * response_v)
    )

    toeplitz = [jnp.exp(-response_decay)]
    for diagonal in range(1, order):
        toeplitz.append(
            toeplitz[-1] * response_decay / float(diagonal)
        )
    toeplitz = jnp.stack(toeplitz, axis=-1)

    values = []
    for band in range(n_band):
        values.extend(
            (e00[:, band], e01[:, band], e10[:, band], e11[:, band])
        )
    for band in range(n_band):
        for row in range(order):
            for column in range(row + 1):
                values.append(toeplitz[:, band, row - column])
    for band in range(n_band):
        for row in range(order):
            values.append(h0[:, band, row])
    for band in range(n_band):
        for row in range(order):
            values.append(h1[:, band, row])
    values = jnp.stack(values, axis=1)
    rows, columns = _integrated_erlang_scatter_indices(n_band, order)
    transitions = jnp.zeros(
        (dt.shape[0], n_state, n_state),
        dtype=values.dtype,
    )
    return transitions.at[:, rows, columns].set(values)


class ErlangResponseIntegratedDHOQS(ErlangResponseDHOQS):
    """Erlang response driven by a CARMA(2,1) continuum."""

    carma_omega0: jnp.ndarray | None = None
    carma_damping: jnp.ndarray | None = None
    carma_obs_position: jnp.ndarray | None = None
    carma_obs_velocity: jnp.ndarray | None = None

    def _base(self):
        return IntegratedTimescaleDHOBaseQS(
            self.carma_omega0,
            self.carma_damping,
            self.carma_obs_position,
            self.carma_obs_velocity,
        )

    def _dimensions(self):
        B = int(self.carma_omega0.shape[0])
        return B, 2 * B, 2 * B + B * int(self.order)

    def design_matrix(self):
        base = self._base()
        A0 = base.design_matrix()
        B, n0, _ = self._dimensions()
        order = int(self.order)
        dtype = A0.dtype
        n_response = B * order
        rates = self._response_rates().astype(dtype)
        state_rates = jnp.repeat(rates, order)

        response = -jnp.diag(state_rates)
        sub_rows = jnp.arange(1, n_response, dtype=jnp.int32)
        within_chain = (sub_rows % order) != 0
        response = response.at[sub_rows, sub_rows - 1].set(
            jnp.where(within_chain, state_rates[sub_rows], 0.0)
        )

        bands = jnp.arange(B, dtype=jnp.int32)
        driver_loadings = jax.vmap(
            lambda b: base.observation_model((jnp.asarray(0.0), b))
        )(bands)
        driver = jnp.zeros((n_response, n0), dtype=dtype)
        chain_starts = bands * order
        driver = driver.at[chain_starts].set(rates[:, None] * driver_loadings)

        zero_top_right = jnp.zeros((n0, n_response), dtype=dtype)
        return jnp.block([[A0, zero_top_right], [driver, response]])

    def transition_matrices_from_dt(self, dt):
        """Return exact transitions for a vector of nonnegative time gaps."""

        return _integrated_erlang_transitions(self, dt)

    def _transition_matrix_expm(self, X1, X2):
        """General matrix-exponential oracle retained for regression tests."""

        t1, _ = X1
        t2, _ = X2
        dt = t2 - t1
        A = self.design_matrix()
        B, _n0, n = self._dimensions()
        idx = self._band_state_indices()
        A_blocks = A[idx[:, :, None], idx[:, None, :]]
        transition_blocks = jax.vmap(lambda block: expm(block * dt))(A_blocks)
        transition = jnp.zeros((n, n), dtype=A.dtype)
        bands = jnp.arange(B)
        return transition.at[
            idx[bands][:, :, None],
            idx[bands][:, None, :],
        ].set(transition_blocks)

    def transition_matrix(self, X1, X2):
        t1, _ = X1
        t2, _ = X2
        return self.transition_matrices_from_dt(
            jnp.reshape(t2 - t1, (1,))
        )[0]

    def to_symm_qsm(self, X):
        """Build the causal QSM with one batched structured transition call."""

        stationary_covariance = self.stationary_covariance()
        time = jnp.asarray(X[0])
        dt = jnp.diff(time, prepend=time[0])
        transitions = self.transition_matrices_from_dt(dt)
        observations = jax.vmap(self.observation_model)(X)
        diagonal = jnp.einsum(
            "ni,ij,nj->n",
            observations,
            stationary_covariance,
            observations,
        )
        lower_p = jax.vmap(lambda h, transition: h @ transition)(
            observations,
            transitions,
        )
        lower_q = observations @ stationary_covariance.T
        return SymmQSM(
            diag=DiagQSM(d=diagonal),
            lower=StrictLowerTriQSM(
                p=lower_p,
                q=lower_q,
                a=transitions,
            ),
        )


class ContiBLRErlangIntegratedDHOModel(ContiBLRErlangRelativeFluxModel):
    """Relative-flux wrapper for the integrated-timescale DHO kernel."""

    def _build_kernel(self, params):
        tau_drw = jnp.asarray(params["tau_drw_band"])
        amp_cont = jnp.asarray(
            params["amp_cont_relflux"] if "amp_cont_relflux" in params else params["amp_cont"]
        )
        amp_blr_rms = jnp.asarray(
            params["amp_blr_relflux"] if "amp_blr_relflux" in params else params["amp_blr"]
        )
        agn_fraction = jnp.asarray(
            params.get("agn_fraction_by_band", jnp.ones_like(amp_cont))
        )
        amp_cont = amp_cont * agn_fraction
        if amp_blr_rms.ndim > agn_fraction.ndim:
            agn_fraction_blr = agn_fraction[..., None]
        else:
            agn_fraction_blr = agn_fraction
        amp_blr_rms = amp_blr_rms * agn_fraction_blr
        base = IntegratedTimescaleDHOBaseQS.from_drw(
            tau_drw,
            jnp.asarray(params["quality_factor"]),
            jnp.asarray(params["tau_perturb_band"]),
        )
        lag_blr = jnp.asarray(params["lag_blr"])

        # Parameterize the line component by its stationary output RMS instead
        # of the Erlang filter's DC gain.  The latter becomes arbitrarily weak
        # for lag >> tau_drw and creates a gain-lag ridge in which enormous
        # coefficients have innocuous covariance.  Unit-response
        # normalization removes that ridge while retaining the same kernel
        # family and an interpretable continuum/line RMS ratio.
        unit_response = ErlangResponseIntegratedDHOQS(
            tau_fast=jnp.full_like(tau_drw, 0.5),
            tau_slow=jnp.full_like(tau_drw, 0.5),
            lag_blr=lag_blr,
            amp_cont=jnp.zeros_like(amp_cont),
            amp_blr=jnp.ones_like(amp_blr_rms),
            order=self.erlang_order,
            carma_omega0=base.omega0,
            carma_damping=base.damping,
            carma_obs_position=base.obs_position,
            carma_obs_velocity=base.obs_velocity,
        )
        bands = jnp.arange(self.nBand, dtype=jnp.int32)
        zero = jnp.asarray(0.0, dtype=tau_drw.dtype)
        unit_response_var = jax.vmap(
            lambda band: unit_response.evaluate((zero, band), (zero, band))
        )(bands)
        amp_blr_dc_gain = amp_blr_rms / jnp.sqrt(
            jnp.maximum(unit_response_var, 1e-12)
        )
        return ErlangResponseIntegratedDHOQS(
            tau_fast=jnp.full_like(tau_drw, 0.5),
            tau_slow=jnp.full_like(tau_drw, 0.5),
            lag_blr=lag_blr,
            amp_cont=amp_cont,
            amp_blr=amp_blr_dc_gain,
            order=self.erlang_order,
            carma_omega0=base.omega0,
            carma_damping=base.damping,
            carma_obs_position=base.obs_position,
            carma_obs_velocity=base.obs_velocity,
        )

    def positive_flux_margin(self, params):
        """Return the per-band four-sigma margin above zero total flux."""
        kernel = self._build_kernel(params)
        bands = jnp.arange(self.nBand, dtype=jnp.int32)
        zero = jnp.asarray(0.0, dtype=self.y.dtype)
        variance = jax.vmap(
            lambda band: kernel.evaluate((zero, band), (zero, band))
        )(bands)
        stationary_std = jnp.sqrt(jnp.maximum(variance, 0.0))

        mean_function, coordinates, _, _ = self._likelihood_inputs(params)
        means = jax.vmap(mean_function)(coordinates)
        observed_bands = jnp.asarray(coordinates[1], dtype=jnp.int32)
        min_mean = jax.vmap(
            lambda band: jnp.min(
                jnp.where(observed_bands == band, means, jnp.inf)
            )
        )(bands)
        return (
            1.0
            + min_mean
            - jnp.asarray(POSITIVE_FLUX_N_SIGMA, dtype=self.y.dtype)
            * stationary_std
        )

    def positive_flux_log_penalty(self, params):
        """Smooth diagnostic penalty retained for reporting and tests."""

        margin = self.positive_flux_margin(params)
        scaled_violation = -margin / jnp.asarray(
            POSITIVE_FLUX_MARGIN_SOFTNESS,
            dtype=self.y.dtype,
        )
        return -0.5 * jnp.sum(jax.nn.softplus(scaled_violation) ** 2)

    def _log_prob_impl(self, params):
        """Unjitted objective for composition into NumPyro's outer JIT."""

        # Keep the boundary differentiable so AutoNormal SVI can initialize;
        # at the chosen softness, even a 0.1 flux-ratio violation costs about
        # 50 log-probability units per affected band.
        from qvc.light_curve.fast_quasisep import block_diagonal_log_probability

        means, coordinates, diagonal, indices = self._likelihood_inputs(params)
        mean = jax.vmap(means)(coordinates)
        residual = self._observed_y_sorted(params, indices) - mean
        likelihood = block_diagonal_log_probability(
            self._build_kernel(params),
            coordinates,
            diagonal,
            residual,
            sort_time=coordinates[0],
        )
        return likelihood + self.positive_flux_log_penalty(params)

    @eqx.filter_jit
    def log_prob(self, params):
        """Standalone compiled objective used by diagnostics and tests."""

        return self._log_prob_impl(params)


def make_multiband_dho_blr_flux_linearized_erlang_drw_model(
    X,
    y,
    yerr,
    n_band=None,
    *,
    survey_idx=None,
    baseline_flux_by_band=None,
    zero_mean=False,
    has_jitter=True,
    erlang_order=DEFAULT_ERLANG_ORDER,
):
    """Construct the all-regime CARMA(2,1) plus causal Erlang response model."""

    del baseline_flux_by_band
    if n_band is None:
        n_band = int(jnp.max(jnp.asarray(X[1], dtype=jnp.int32))) + 1
    t = jnp.asarray(X[0])
    return ContiBLRErlangIntegratedDHOModel(
        X,
        y,
        yerr,
        base_kernel=IntegratedTimescaleDHOBaseQS.from_drw(
            tau_drw=jnp.full(n_band, 100.0),
            quality_factor=jnp.full(n_band, 0.5),
            tau_perturb=jnp.full(n_band, 2.0),
        ),
        nBand=n_band,
        mean_func=make_linear_mean_func(t, zero_mean=zero_mean),
        has_lag=False,
        has_jitter=has_jitter,
        zero_mean=zero_mean,
        survey_idx=survey_idx,
        erlang_order=erlang_order,
        use_fast_solver=False,
    )


__all__ = [
    "ContiBLRErlangIntegratedDHOModel",
    "ErlangResponseIntegratedDHOQS",
    "POSITIVE_FLUX_MARGIN_SOFTNESS",
    "POSITIVE_FLUX_N_SIGMA",
    "make_multiband_dho_blr_flux_linearized_erlang_drw_model",
]
