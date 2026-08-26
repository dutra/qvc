import numpy as np
import pandas as pd
import pytest

from qvc.hubble import hubble_fit
from qvc.hubble.latent_alpha_completeness import (
    LatentAlphaConfig,
    latent_alpha_config_hash,
)


def _wang_config(*, mu=-0.5, magnitude_interactions=False):
    return LatentAlphaConfig.for_lf(
        lf_model="wang2026_type1_lade_a",
        requested_luminosity_state="attenuated",
        mode="off",
        mu=mu,
        include_magnitude_interactions=magnitude_interactions,
    )


def _resume_replot_payload(config=None):
    payload = {
        "flat_samples": np.ones((3, 2), dtype=float),
        "object_id_fit_selection": np.array(["agn_a", "agn_b"]),
        "dmi_max_w": np.array([10.0, 20.0]),
        "dmi_posterior_median": np.array([11.0, 21.0]),
        "dmi_posterior_sigma": np.array([0.1, 0.2]),
        "integrals_max_w": np.array([100.0, 200.0]),
    }
    if config is not None:
        payload["latent_alpha_config_json"] = config.to_json()
    return payload


def test_resume_replot_requires_exact_latent_alpha_config_json():
    expected = _wang_config()
    changed_parent = _wang_config(mu=-0.4)
    current = pd.DataFrame({"object_id": ["agn_b"]})

    remapped = hubble_fit._remap_resume_replot_checkpoint(
        _resume_replot_payload(expected),
        "posterior.h5",
        current,
        ndim=2,
        expected_latent_alpha_config=expected,
    )
    np.testing.assert_array_equal(
        remapped["object_id_fit_selection"], np.array(["agn_b"])
    )

    with pytest.raises(RuntimeError, match="exact latent_alpha_config_json"):
        hubble_fit._remap_resume_replot_checkpoint(
            _resume_replot_payload(expected),
            "posterior.h5",
            current,
            ndim=2,
            expected_latent_alpha_config=changed_parent,
        )

    with pytest.raises(RuntimeError, match="exact latent_alpha_config_json"):
        hubble_fit._remap_resume_replot_checkpoint(
            _resume_replot_payload(),
            "posterior.h5",
            current,
            ndim=2,
            expected_latent_alpha_config=expected,
        )


def test_resume_replot_rejects_latent_checkpoint_for_nonlatent_run():
    current = pd.DataFrame({"object_id": ["agn_a"]})
    with pytest.raises(RuntimeError, match="exact latent_alpha_config_json"):
        hubble_fit._remap_resume_replot_checkpoint(
            _resume_replot_payload(_wang_config()),
            "posterior.h5",
            current,
            ndim=2,
            expected_latent_alpha_config=None,
        )


def test_multi_cosmology_comparison_tag_contains_full_config_hash_identity():
    baseline = _wang_config()
    changed_parent = _wang_config(mu=-0.4)
    changed_surface = _wang_config(magnitude_interactions=True)

    def tag(config):
        return hubble_fit.make_multi_cosmology_comparison_tag(
            "single_compare",
            only_agn=True,
            speed="quick",
            N=None,
            z_range=(0.44, 3.16),
            completeness=True,
            completeness_mode="3d_fhost_latent_alpha",
            completeness_magnitude="attenuated",
            latent_alpha_config=config,
        )

    tags = {tag(config) for config in (baseline, changed_parent, changed_surface)}
    assert len(tags) == 3
    for config in (baseline, changed_parent, changed_surface):
        config_tag = tag(config)
        assert "_alat-off-attenuated" in config_tag
        assert latent_alpha_config_hash(config)[:10] in config_tag

    assert hubble_fit._latent_alpha_run_tag_suffix(baseline) in hubble_fit.make_run_tag(
        "FlatLambdaCDM",
        False,
        "quick",
        None,
        (0.44, 3.16),
        only_agn=True,
        completeness_mode="3d_fhost_latent_alpha",
        completeness_magnitude="attenuated",
        latent_alpha_config=baseline,
    )
