import numpy as np
import pandas as pd
import pytest

from qvc.hubble import hubble_fit
from qvc.hubble.hubble_model import AgnPivotContext, build_agn_pivot_context


def _reference_frame():
    return pd.DataFrame(
        {
            "object_id": ["q0", "q1", "q2"],
            "z": [0.5, 1.0, 1.5],
            "log_sigma_uv": [-1.0, -0.7, -0.4],
            "log_tau_uv_rf": [2.0, 2.5, 3.0],
        }
    )


def _context():
    return build_agn_pivot_context(_reference_frame(), (0.5, 1.5))


def _checkpoint_payload(context=None):
    context = _context() if context is None else context
    return {
        **hubble_fit._agn_pivot_checkpoint_payload(context),
        "sigma_clip_pass_stage": "single",
        "object_id_fit_selection": np.asarray(
            context.reference_object_ids,
            dtype=str,
        ),
    }


def _prepare_resume_replot_context(current, checkpoint):
    return hubble_fit._prepare_shared_agn_pivot_context(
        current,
        cosmo_models=["FlatLambdaCDM"],
        resume_by_model={"FlatLambdaCDM": str(checkpoint)},
        z_range=(0.5, 1.5),
        N=None,
        uniform_redshift_distribution=False,
        only_sna=False,
        only_agn=False,
        speed="fastest",
        completeness=False,
        completeness_mode="2d",
        disable_ceph_dist_calibration=False,
        use_planck_h0_prior=False,
        use_planck_om_prior=False,
        use_alpha_lambda_term=False,
        use_eta_sigma_term=False,
        use_redshift_log_f_term=False,
        disable_sigma_clip_pass=True,
        resume_stage="both",
        prefix="unit",
        resume_replot_with_cuts=True,
    )


def test_checkpoint_pivot_context_round_trip(tmp_path):
    checkpoint = tmp_path / "pivot.h5"
    expected = _context()
    hubble_fit.save_chains(
        checkpoint,
        flat_samples=np.ones((2, 2)),
        **_checkpoint_payload(expected),
    )

    loaded = hubble_fit.load_chains(checkpoint)
    actual = hubble_fit._load_agn_pivot_context_from_checkpoint(
        loaded,
        checkpoint_file=checkpoint,
    )

    assert actual == expected


def test_stage_checkpoint_preserves_pivot_context_exactly(tmp_path):
    checkpoint = tmp_path / "stage.h5"
    expected = _context()
    hubble_fit.save_chains(
        checkpoint,
        flat_samples=np.ones((2, 2)),
        **_checkpoint_payload(expected),
    )

    hubble_fit._write_stage_checkpoint(
        checkpoint,
        sigma_clip_pass_stage="pass1",
        df_agn_initial_fit_selection=_reference_frame(),
    )
    loaded = hubble_fit.load_chains(checkpoint)
    actual = hubble_fit._load_agn_pivot_context_from_checkpoint(
        loaded,
        checkpoint_file=checkpoint,
    )

    assert actual == expected
    np.testing.assert_array_equal(
        hubble_fit._normalize_object_id_array(
            loaded["object_id_initial_fit_selection"],
            field_name="object_id_initial_fit_selection",
            checkpoint_file=checkpoint,
        ),
        np.asarray(expected.reference_object_ids),
    )
    hubble_fit._validate_agn_pivot_checkpoint_reference_provenance(
        actual,
        loaded,
        checkpoint_file=checkpoint,
    )


@pytest.mark.parametrize(
    ("stage", "provenance_field"),
    [
        ("single", "object_id_fit_selection"),
        ("pass1", "object_id_initial_fit_selection"),
        ("pass2", "object_id_initial_fit_selection"),
    ],
)
def test_checkpoint_reference_provenance_accepts_exact_order_and_multiplicity(
    stage,
    provenance_field,
):
    context = _context()
    payload = {
        **_checkpoint_payload(context),
        "sigma_clip_pass_stage": stage,
        provenance_field: np.asarray(context.reference_object_ids),
    }

    assert (
        hubble_fit._validate_agn_pivot_checkpoint_reference_provenance(
            context,
            payload,
            checkpoint_file="exact.h5",
        )
        is context
    )


@pytest.mark.parametrize("stage", ["single", "pass1", "pass2"])
def test_checkpoint_reference_provenance_rejects_missing_required_field(stage):
    context = _context()
    payload = {
        **hubble_fit._agn_pivot_checkpoint_payload(context),
        "sigma_clip_pass_stage": stage,
    }

    with pytest.raises(RuntimeError, match="missing required immutable.*provenance"):
        hubble_fit._validate_agn_pivot_checkpoint_reference_provenance(
            context,
            payload,
            checkpoint_file="missing-provenance.h5",
        )


def test_checkpoint_reference_provenance_rejects_missing_stage():
    context = _context()
    payload = _checkpoint_payload(context)
    payload.pop("sigma_clip_pass_stage")

    with pytest.raises(RuntimeError, match="sigma_clip_pass_stage.*No legacy fallback"):
        hubble_fit._validate_agn_pivot_checkpoint_reference_provenance(
            context,
            payload,
            checkpoint_file="missing-stage.h5",
        )


def test_checkpoint_reference_provenance_rejects_missing_object_id_element():
    context = _context()
    payload = _checkpoint_payload(context)
    payload["object_id_fit_selection"] = np.asarray(
        ["q0", np.nan, "q2"],
        dtype=object,
    )

    with pytest.raises(RuntimeError, match="missing or non-scalar object ID"):
        hubble_fit._validate_agn_pivot_checkpoint_reference_provenance(
            context,
            payload,
            checkpoint_file="missing-object-id.h5",
        )


@pytest.mark.parametrize(
    "provenance_ids",
    [
        ("q1", "q0", "q2"),
        ("q0", "q1"),
        ("q0", "q1", "q1"),
    ],
)
@pytest.mark.parametrize("stage", ["single", "pass1", "pass2"])
def test_checkpoint_reference_provenance_rejects_reordered_partial_or_duplicate_ids(
    stage,
    provenance_ids,
):
    context = _context()
    provenance_field = (
        "object_id_fit_selection"
        if stage == "single"
        else "object_id_initial_fit_selection"
    )
    payload = {
        **_checkpoint_payload(context),
        "sigma_clip_pass_stage": stage,
        provenance_field: np.asarray(provenance_ids),
    }

    with pytest.raises(RuntimeError, match="including order and multiplicity"):
        hubble_fit._validate_agn_pivot_checkpoint_reference_provenance(
            context,
            payload,
            checkpoint_file="incompatible-provenance.h5",
        )


@pytest.mark.parametrize("missing_key", hubble_fit.AGN_PIVOT_CHECKPOINT_KEYS)
def test_checkpoint_rejects_each_missing_pivot_field(missing_key):
    payload = _checkpoint_payload()
    payload.pop(missing_key)

    with pytest.raises(RuntimeError, match="missing required immutable pivot metadata"):
        hubble_fit._load_agn_pivot_context_from_checkpoint(
            payload,
            checkpoint_file="missing.h5",
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("agn_pivot_observable_names", ["log_tau_uv_rf", "log_sigma_uv"]),
        ("agn_pivot_observable_names", ["log_sigma_uv", "log_sigma_uv"]),
        ("agn_pivot_values", [np.nan, 2.5]),
        ("agn_pivot_values", [1.0]),
        ("agn_pivot_z_range", [0.5, np.inf]),
        ("agn_pivot_z_range", [1.5, 0.5]),
        ("agn_pivot_reference_object_ids", []),
        ("agn_pivot_reference_object_ids", ["q0", np.nan, "q2"]),
        ("agn_pivot_rule", "unknown-rule"),
    ],
)
def test_checkpoint_rejects_invalid_or_incompatible_pivot_metadata(key, value):
    payload = _checkpoint_payload()
    payload[key] = value

    with pytest.raises(RuntimeError, match="invalid or incompatible pivot metadata"):
        hubble_fit._load_agn_pivot_context_from_checkpoint(
            payload,
            checkpoint_file="invalid.h5",
        )


@pytest.mark.parametrize(
    "reference_ids",
    [
        ("q1", "q0", "q2"),
        ("q0", "q1"),
        ("q0", "q1", "q1"),
    ],
)
def test_checkpoint_context_rejects_reordered_partial_or_extra_duplicate_reference(
    reference_ids,
):
    base = _context()
    context = AgnPivotContext(
        observable_names=base.observable_names,
        values=base.values,
        z_range=base.z_range,
        reference_object_ids=reference_ids,
        rule=base.rule,
    )

    with pytest.raises(ValueError, match="reference object IDs"):
        hubble_fit._validate_agn_pivot_context_for_reference(
            context,
            _reference_frame(),
            z_range=(0.5, 1.5),
        )


def test_resume_replot_can_use_stored_context_with_different_current_objects(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "stored.h5"
    stored = _context()
    payload = {
        **_checkpoint_payload(stored),
        "sigma_clip_pass_stage": "pass2",
        "object_id_initial_fit_selection": np.asarray(
            stored.reference_object_ids,
            dtype=str,
        ),
    }
    hubble_fit.save_chains(checkpoint, **payload)
    current = _reference_frame().copy()
    current["object_id"] = ["new0", "new1", "new2"]
    current["log_sigma_uv"] += 10.0
    current["log_tau_uv_rf"] -= 10.0
    monkeypatch.setattr(
        hubble_fit,
        "build_agn_pivot_context",
        lambda *args, **kwargs: pytest.fail(
            "resume-replot must not recompute the pivot context"
        ),
    )

    actual = _prepare_resume_replot_context(current, checkpoint)

    assert actual == stored


@pytest.mark.parametrize(
    "provenance_ids",
    [
        None,
        ("q1", "q0", "q2"),
        ("q0", "q1"),
        ("q0", "q1", "q1"),
    ],
)
def test_resume_replot_rejects_missing_or_incompatible_two_pass_provenance(
    tmp_path,
    provenance_ids,
):
    checkpoint = tmp_path / "invalid-two-pass.h5"
    stored = _context()
    payload = {
        **_checkpoint_payload(stored),
        "sigma_clip_pass_stage": "pass2",
    }
    if provenance_ids is not None:
        payload["object_id_initial_fit_selection"] = np.asarray(
            provenance_ids,
            dtype=str,
        )
    hubble_fit.save_chains(checkpoint, **payload)

    current = _reference_frame().copy()
    current["object_id"] = ["new0", "new1", "new2"]
    expected_message = (
        "missing required immutable.*provenance"
        if provenance_ids is None
        else "including order and multiplicity"
    )
    with pytest.raises(RuntimeError, match=expected_message):
        _prepare_resume_replot_context(current, checkpoint)


def test_multi_cosmology_fresh_context_is_computed_once(monkeypatch):
    calls = []
    real_builder = build_agn_pivot_context

    def counted_builder(*args, **kwargs):
        calls.append(args[0]["object_id"].astype(str).tolist())
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(hubble_fit, "build_agn_pivot_context", counted_builder)
    context = hubble_fit._prepare_shared_agn_pivot_context(
        _reference_frame(),
        cosmo_models=["FlatLambdaCDM", "FlatwCDM"],
        resume_by_model={"FlatLambdaCDM": False, "FlatwCDM": False},
        z_range=(0.5, 1.5),
        N=None,
        uniform_redshift_distribution=False,
        only_sna=False,
        only_agn=False,
        speed="fastest",
        completeness=False,
        completeness_mode="2d",
        disable_ceph_dist_calibration=False,
        use_planck_h0_prior=False,
        use_planck_om_prior=False,
        use_alpha_lambda_term=False,
        use_eta_sigma_term=False,
        use_redshift_log_f_term=False,
        disable_sigma_clip_pass=True,
        resume_stage="both",
        prefix="unit",
    )

    assert isinstance(context, AgnPivotContext)
    assert calls == [["q0", "q1", "q2"]]


def test_fit_selection_applies_n_after_inclusive_redshift_cut():
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "object_id": ["low", "high"],
                    "z": [0.1, 2.0],
                    "log_sigma_uv": [-20.0, 20.0],
                    "log_tau_uv_rf": [-20.0, 20.0],
                }
            ),
            _reference_frame(),
        ],
        ignore_index=True,
    )

    first = hubble_fit._select_agn_fit_selection(
        frame,
        z_range=(0.5, 1.5),
        N=2,
        uniform_redshift_distribution=False,
    )
    second = hubble_fit._select_agn_fit_selection(
        frame,
        z_range=(0.5, 1.5),
        N=2,
        uniform_redshift_distribution=False,
    )

    assert len(first) == 2
    assert first["z"].between(0.5, 1.5, inclusive="both").all()
    assert first["object_id"].tolist() == second["object_id"].tolist()


def test_multi_cosmology_resume_loads_context_without_computing(
    tmp_path,
    monkeypatch,
):
    context = _context()
    checkpoints = {}
    for model in ("FlatLambdaCDM", "FlatwCDM"):
        checkpoint = tmp_path / f"{model}.h5"
        hubble_fit.save_chains(checkpoint, **_checkpoint_payload(context))
        checkpoints[model] = str(checkpoint)
    monkeypatch.setattr(
        hubble_fit,
        "build_agn_pivot_context",
        lambda *args, **kwargs: pytest.fail(
            "resumed contexts must be loaded, not recomputed"
        ),
    )

    actual = hubble_fit._prepare_shared_agn_pivot_context(
        _reference_frame(),
        cosmo_models=list(checkpoints),
        resume_by_model=checkpoints,
        z_range=(0.5, 1.5),
        N=None,
        uniform_redshift_distribution=False,
        only_sna=False,
        only_agn=False,
        speed="fastest",
        completeness=False,
        completeness_mode="2d",
        disable_ceph_dist_calibration=False,
        use_planck_h0_prior=False,
        use_planck_om_prior=False,
        use_alpha_lambda_term=False,
        use_eta_sigma_term=False,
        use_redshift_log_f_term=False,
        disable_sigma_clip_pass=True,
        resume_stage="both",
        prefix="unit",
    )

    assert actual == context


def test_multi_cosmology_resume_rejects_different_contexts(tmp_path):
    first = _context()
    second = AgnPivotContext(
        observable_names=first.observable_names,
        values=(first.values[0] + 0.1, first.values[1]),
        z_range=first.z_range,
        reference_object_ids=first.reference_object_ids,
    )
    checkpoints = {}
    for model, context in (
        ("FlatLambdaCDM", first),
        ("FlatwCDM", second),
    ):
        checkpoint = tmp_path / f"{model}.h5"
        hubble_fit.save_chains(checkpoint, **_checkpoint_payload(context))
        checkpoints[model] = str(checkpoint)

    with pytest.raises(RuntimeError, match="do not share one identical"):
        hubble_fit._prepare_shared_agn_pivot_context(
            _reference_frame(),
            cosmo_models=list(checkpoints),
            resume_by_model=checkpoints,
            z_range=(0.5, 1.5),
            N=None,
            uniform_redshift_distribution=False,
            only_sna=False,
            only_agn=False,
            speed="fastest",
            completeness=False,
            completeness_mode="2d",
            disable_ceph_dist_calibration=False,
            use_planck_h0_prior=False,
            use_planck_om_prior=False,
            use_alpha_lambda_term=False,
            use_eta_sigma_term=False,
            use_redshift_log_f_term=False,
            disable_sigma_clip_pass=True,
            resume_stage="both",
            prefix="unit",
        )
