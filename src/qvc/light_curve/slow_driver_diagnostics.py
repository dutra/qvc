"""Chain-aware convergence diagnostics for the original SLB slow driver."""
import numpy as np

SLOW_FIELD = "log_tau_slow_rf"
SLOW_DIAGNOSTIC_FIELDS = (SLOW_FIELD + "_ess", SLOW_FIELD + "_rhat")

def slow_pole_draws(samples, model_variant, redshift):
    keys = {'shared_latent_blr': 'tau_slow_driver'}
    if model_variant not in keys:
        raise ValueError(f'Slow-pole regressor is unsupported for {model_variant!r}')
    if keys[model_variant] not in samples:
        raise ValueError(f'Missing explicit {keys[model_variant]} posterior; legacy poles cannot be inferred')
    if model_variant == 'shared_latent_blr' and 'eta_tau' in samples:
        if np.any(np.asarray(samples['eta_tau']) != 0):
            raise ValueError('Legacy SLB samples with nonzero eta_tau require a refit')
    z = float(redshift)
    pole = np.asarray(samples[keys[model_variant]], dtype=float)
    if not np.isfinite(z) or z <= -1 or np.any(~np.isfinite(pole)) or np.any(pole <= 0):
        raise ValueError('Slow poles must be finite and positive, and redshift must exceed -1')
    return np.log10(pole) - np.log10(1 + z)

def slow_pole_convergence(samples_per_chain, model_variant, redshift, *, saved_diagnostics=None):
    """Chain-aware NumPyro diagnostics, or explicit unavailable values."""
    result = {SLOW_FIELD + '_ess': np.nan, SLOW_FIELD + '_rhat': np.nan}
    if samples_per_chain is None:
        for key in result:
            if saved_diagnostics is not None and key in saved_diagnostics:
                value = np.asarray(saved_diagnostics[key], dtype=float)
                if value.ndim != 0:
                    raise ValueError(f"Saved {key} must be scalar")
                result[key] = float(value)
        return result
    draws = slow_pole_draws(samples_per_chain, model_variant, redshift)
    if draws.ndim != 2 or draws.shape[1] < 4 or np.ptp(draws) == 0:
        return result
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    result[SLOW_FIELD + '_ess'] = float(effective_sample_size(draws))
    result[SLOW_FIELD + '_rhat'] = float(split_gelman_rubin(draws))
    return result
