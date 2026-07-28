from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from qvc.hubble.hubble_model import (
    AGN_PIVOT_RULE,
    AgnPivotContext,
    agn_model_pack_obs,
    build_agn_pivot_context,
)


def _pivot_frame():
    return pd.DataFrame(
        {
            "object_id": ["low", "left", "middle", "right", "high"],
            "z": [0.1, 0.2, 0.6, 1.0, 1.1],
            "log_sigma_uv": np.log10([9.9, 0.14, 0.26, 0.36, 0.001]),
            "log_tau_uv_rf": np.log10([99999.0, 149.0, 251.0, 899.0, 1.0]),
            "alpha_lambda": [50.0, -1.9, -1.5, -1.1, -50.0],
            "eta_sigma": [50.0, 0.2, 0.6, 1.0, -50.0],
        }
    )


def _base_context():
    return AgnPivotContext(
        observable_names=("log_sigma_uv", "log_tau_uv_rf"),
        values=(np.log10(0.3), np.log10(300.0)),
        z_range=(0.2, 1.0),
        reference_object_ids=("left", "middle", "right"),
    )


def _packed_observables():
    return {
        "log_sigma_uv": np.array([-0.5, -0.3]),
        "log_tau_uv_rf": np.array([2.1, 2.4]),
        "log_sigma_uv_std_psd": np.array([0.04, 0.05]),
        "log_tau_uv_rf_std_psd": np.array([0.08, 0.09]),
        "log_sigma_uv_log_tau_uv_rf_cov_psd": np.array([0.001, 0.002]),
    }


def test_builder_preserves_rounded_sigma_tau_and_optional_medians():
    frame = _pivot_frame()

    context = build_agn_pivot_context(
        frame,
        (0.2, 1.0),
        use_alpha_lambda_term=True,
        use_eta_sigma_term=True,
    )

    assert context.observable_names == (
        "log_sigma_uv",
        "log_tau_uv_rf",
        "alpha_lambda",
        "eta_sigma",
    )
    np.testing.assert_allclose(
        context.values,
        [np.log10(0.3), np.log10(300.0), -1.5, 0.6],
        rtol=0.0,
        atol=1e-14,
    )
    assert context.z_range == (0.2, 1.0)
    assert context.reference_object_ids == ("left", "middle", "right")
    assert context.rule == AGN_PIVOT_RULE


@pytest.mark.parametrize(
    ("observable", "linear_value", "expected_pivot"),
    [
        ("log_sigma_uv", 0.051, np.log10(0.1)),
        ("log_tau_uv_rf", 51.0, np.log10(100.0)),
    ],
)
def test_builder_accepts_rounded_pivots_just_above_zero_boundary(
    observable,
    linear_value,
    expected_pivot,
):
    frame = pd.DataFrame(
        {
            "object_id": ["object"],
            "z": [0.5],
            "log_sigma_uv": [np.log10(0.3)],
            "log_tau_uv_rf": [np.log10(300.0)],
        }
    )
    frame.loc[0, observable] = np.log10(linear_value)

    context = build_agn_pivot_context(frame, (0.2, 1.0))

    assert context.as_dict()[observable] == pytest.approx(expected_pivot)


@pytest.mark.parametrize(
    ("observable", "log_value"),
    [
        ("log_sigma_uv", np.log10(0.05)),
        ("log_tau_uv_rf", np.log10(50.0)),
        ("log_sigma_uv", -400.0),
        ("log_tau_uv_rf", -400.0),
    ],
)
def test_builder_rejects_boundary_or_underflow_pivots_rounded_to_zero(
    observable,
    log_value,
):
    frame = pd.DataFrame(
        {
            "object_id": ["object"],
            "z": [0.5],
            "log_sigma_uv": [np.log10(0.3)],
            "log_tau_uv_rf": [np.log10(300.0)],
        }
    )
    frame.loc[0, observable] = log_value

    with pytest.raises(
        ValueError,
        match=rf"Rounded linear AGN pivot.*{observable!s}.*finite and positive",
    ):
        build_agn_pivot_context(frame, (0.2, 1.0))


@pytest.mark.parametrize("observable", ["log_sigma_uv", "log_tau_uv_rf"])
def test_builder_rejects_nonfinite_rounded_linear_pivots(observable):
    frame = pd.DataFrame(
        {
            "object_id": ["object"],
            "z": [0.5],
            "log_sigma_uv": [np.log10(0.3)],
            "log_tau_uv_rf": [np.log10(300.0)],
        }
    )
    frame.loc[0, observable] = 400.0

    with pytest.raises(
        ValueError,
        match=rf"Rounded linear AGN pivot.*{observable!s}.*finite and positive",
    ):
        build_agn_pivot_context(frame, (0.2, 1.0))


def test_builder_is_deterministic_inclusive_and_preserves_duplicate_ids():
    reference = pd.DataFrame(
        {
            "object_id": ["left", "duplicate", "duplicate", "right"],
            "z": [0.2, 0.4, 0.4, 1.0],
            "log_sigma_uv": np.log10([0.1, 0.2, 0.2, 0.4]),
            "log_tau_uv_rf": np.log10([100.0, 200.0, 200.0, 400.0]),
        }
    )

    first = build_agn_pivot_context(reference, (0.2, 1.0))
    second = build_agn_pivot_context(reference.copy(deep=True), (0.2, 1.0))

    assert first == second
    assert first.reference_object_ids == ("left", "duplicate", "duplicate", "right")


def test_out_of_range_extremes_do_not_change_context():
    in_range = _pivot_frame().query("0.2 <= z <= 1.0").copy()
    extended = pd.concat(
        [
            in_range,
            pd.DataFrame(
                {
                    "object_id": ["extreme-low", "extreme-high"],
                    "z": [-100.0, 100.0],
                    "log_sigma_uv": [-1000.0, 1000.0],
                    "log_tau_uv_rf": [1000.0, -1000.0],
                    "alpha_lambda": [-1000.0, 1000.0],
                    "eta_sigma": [1000.0, -1000.0],
                }
            ),
        ],
        ignore_index=True,
    )

    base_context = build_agn_pivot_context(
        in_range,
        (0.2, 1.0),
        use_alpha_lambda_term=True,
        use_eta_sigma_term=True,
    )
    extended_context = build_agn_pivot_context(
        extended,
        (0.2, 1.0),
        use_alpha_lambda_term=True,
        use_eta_sigma_term=True,
    )

    assert extended_context == base_context


@pytest.mark.parametrize(
    ("frame_transform", "kwargs", "error_type", "message"),
    [
        (
            lambda frame: frame.drop(columns=["log_sigma_uv"]),
            {},
            KeyError,
            "log_sigma_uv",
        ),
        (
            lambda frame: frame.drop(columns=["alpha_lambda"]),
            {"use_alpha_lambda_term": True},
            KeyError,
            "alpha_lambda",
        ),
        (
            lambda frame: frame.assign(
                log_sigma_uv=["0.0", "bad", "0.0", "0.0", "0.0"]
            ),
            {},
            ValueError,
            "log_sigma_uv.*numeric",
        ),
        (
            lambda frame: frame.assign(
                log_tau_uv_rf=[0.0, np.nan, 0.0, 0.0, 0.0]
            ),
            {},
            ValueError,
            "log_tau_uv_rf.*nonfinite",
        ),
        (
            lambda frame: frame.assign(z=["bad"] * len(frame)),
            {},
            ValueError,
            "'z'.*numeric",
        ),
        (
            lambda frame: frame.assign(z=np.full(len(frame), 2.0)),
            {},
            ValueError,
            "no reference objects",
        ),
        (
            lambda frame: frame.assign(
                object_id=["low", pd.NA, "middle", "right", "high"]
            ),
            {},
            ValueError,
            "object_id.*present scalar",
        ),
    ],
)
def test_builder_rejects_missing_nonnumeric_nonfinite_or_empty_fit_data(
    frame_transform,
    kwargs,
    error_type,
    message,
):
    frame = frame_transform(_pivot_frame())

    with pytest.raises(error_type, match=message):
        build_agn_pivot_context(frame, (0.2, 1.0), **kwargs)


@pytest.mark.parametrize(
    "z_range",
    [
        (),
        (0.2,),
        (0.2, 1.0, 2.0),
        (1.0, 0.2),
        (np.nan, 1.0),
        (0.2, np.inf),
    ],
)
def test_builder_rejects_invalid_redshift_range(z_range):
    with pytest.raises(ValueError, match="z_range"):
        build_agn_pivot_context(_pivot_frame(), z_range)


def test_pack_observables_requires_context_and_returns_its_exact_pivots():
    obs_dict = _packed_observables()
    context = _base_context()

    with pytest.raises(TypeError, match="pivot_context"):
        agn_model_pack_obs(obs_dict)

    obs, errors, pivots = agn_model_pack_obs(
        obs_dict,
        pivot_context=context,
    )

    np.testing.assert_array_equal(
        obs,
        np.array([obs_dict["log_sigma_uv"], obs_dict["log_tau_uv_rf"]]),
    )
    np.testing.assert_array_equal(
        errors,
        np.array(
            [
                obs_dict["log_sigma_uv_std_psd"],
                obs_dict["log_tau_uv_rf_std_psd"],
                obs_dict["log_sigma_uv_log_tau_uv_rf_cov_psd"],
            ]
        ),
    )
    np.testing.assert_array_equal(pivots, np.asarray(context.values))


def test_pack_observables_rejects_none_and_model_incompatible_contexts():
    obs_dict = _packed_observables()

    with pytest.raises(TypeError, match="AgnPivotContext"):
        agn_model_pack_obs(obs_dict, pivot_context=None)

    reordered = AgnPivotContext(
        observable_names=("log_tau_uv_rf", "log_sigma_uv"),
        values=(np.log10(300.0), np.log10(0.3)),
        z_range=(0.2, 1.0),
        reference_object_ids=("left", "middle", "right"),
    )
    with pytest.raises(ValueError, match="active model"):
        agn_model_pack_obs(obs_dict, pivot_context=reordered)

    unknown = AgnPivotContext(
        observable_names=("log_sigma_uv", "not_a_model_observable"),
        values=(np.log10(0.3), 0.0),
        z_range=(0.2, 1.0),
        reference_object_ids=("left", "middle", "right"),
    )
    with pytest.raises(ValueError, match="active model"):
        agn_model_pack_obs(obs_dict, pivot_context=unknown)


def test_pack_observables_rejects_context_missing_enabled_optional_terms():
    obs_dict = {
        **_packed_observables(),
        "alpha_lambda": np.array([-1.5, -1.4]),
        "alpha_lambda_err": np.array([0.08, 0.08]),
        "eta_sigma": np.array([0.5, 0.6]),
        "eta_sigma_err": np.array([0.04, 0.04]),
    }

    with pytest.raises(ValueError, match="active model"):
        agn_model_pack_obs(
            obs_dict,
            use_alpha_lambda_term=True,
            pivot_context=_base_context(),
        )
    with pytest.raises(ValueError, match="active model"):
        agn_model_pack_obs(
            obs_dict,
            use_eta_sigma_term=True,
            pivot_context=_base_context(),
        )


def test_context_is_immutable_and_copies_mutable_constructor_inputs():
    names = ["log_sigma_uv", "log_tau_uv_rf"]
    values = [np.log10(0.3), np.log10(300.0)]
    z_range = [0.2, 1.0]
    object_ids = ["same", "same"]
    context = AgnPivotContext(names, values, z_range, object_ids)

    names[0] = "changed"
    values[0] = -99.0
    z_range[0] = -99.0
    object_ids[0] = "changed"

    assert context.observable_names == ("log_sigma_uv", "log_tau_uv_rf")
    assert context.values == (np.log10(0.3), np.log10(300.0))
    assert context.z_range == (0.2, 1.0)
    assert context.reference_object_ids == ("same", "same")
    with pytest.raises(FrozenInstanceError):
        context.values = (0.0, 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"observable_names": ()},
        {"observable_names": ("log_sigma_uv", "log_sigma_uv")},
        {"values": (0.0,)},
        {"values": (0.0, np.nan)},
        {"z_range": (0.2,)},
        {"z_range": (1.0, 0.2)},
        {"z_range": (0.2, np.inf)},
        {"reference_object_ids": ()},
        {"reference_object_ids": (None,)},
        {"reference_object_ids": (np.nan,)},
        {"reference_object_ids": (pd.NA,)},
        {"rule": "unknown_rule"},
    ],
)
def test_context_rejects_invalid_metadata(kwargs):
    valid = {
        "observable_names": ("log_sigma_uv", "log_tau_uv_rf"),
        "values": (np.log10(0.3), np.log10(300.0)),
        "z_range": (0.2, 1.0),
        "reference_object_ids": ("object",),
        "rule": AGN_PIVOT_RULE,
    }
    valid.update(kwargs)

    with pytest.raises(ValueError):
        AgnPivotContext(**valid)
