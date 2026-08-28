from collections import OrderedDict
from types import ModuleType
import sys

import numpy as np

from qvc.mcmc_diagnostics import (
    compute_numpyro_summary,
    convergence_fields,
    print_numpyro_summary_dict,
)


def test_summary_is_computed_once_then_printed_and_persisted(monkeypatch, capsys):
    calls = []
    expected = {
        "log_sigma_uv": OrderedDict(
            mean=np.array(1.0),
            std=np.array(0.1),
            median=np.array(1.0),
            **{
                "5.0%": np.array(0.8),
                "95.0%": np.array(1.2),
                "n_eff": np.array(123.0),
                "r_hat": np.array(1.01),
            },
        )
    }
    diagnostics_module = ModuleType("numpyro.diagnostics")

    def fake_summary(samples, *, prob, group_by_chain):
        calls.append((samples, prob, group_by_chain))
        return expected

    diagnostics_module.summary = fake_summary
    numpyro_module = ModuleType("numpyro")
    numpyro_module.diagnostics = diagnostics_module
    monkeypatch.setitem(sys.modules, "numpyro", numpyro_module)
    monkeypatch.setitem(sys.modules, "numpyro.diagnostics", diagnostics_module)

    result = compute_numpyro_summary(
        {"log_sigma_uv": np.ones((2, 4))},
        group_by_chain=True,
    )
    print_numpyro_summary_dict(result, heading="posterior")
    fields = convergence_fields(result, {"log_sigma_uv": "log_sigma_uv"})

    assert len(calls) == 1
    assert fields == {
        "log_sigma_uv_rhat": 1.01,
        "log_sigma_uv_ess": 123.0,
    }
    output = capsys.readouterr().out
    assert "posterior" in output
    assert "n_eff" in output
    assert "r_hat" in output


def test_missing_summary_produces_nan_convergence_fields(capsys):
    fields = convergence_fields(
        {},
        {
            "log_sigma_uv": "log_sigma_uv",
            "log_tau_uv_rf": "log_tau_uv",
        },
    )
    print_numpyro_summary_dict({}, heading="posterior")

    assert set(fields) == {
        "log_sigma_uv_rhat",
        "log_sigma_uv_ess",
        "log_tau_uv_rf_rhat",
        "log_tau_uv_rf_ess",
    }
    assert all(np.isnan(value) for value in fields.values())
    assert "Summary unavailable" in capsys.readouterr().out


def test_short_summary_failure_returns_empty_mapping(monkeypatch):
    diagnostics_module = ModuleType("numpyro.diagnostics")

    def unavailable(*args, **kwargs):
        raise AssertionError("too few draws")

    diagnostics_module.summary = unavailable
    numpyro_module = ModuleType("numpyro")
    numpyro_module.diagnostics = diagnostics_module
    monkeypatch.setitem(sys.modules, "numpyro", numpyro_module)
    monkeypatch.setitem(sys.modules, "numpyro.diagnostics", diagnostics_module)

    assert compute_numpyro_summary({"x": np.ones((1, 2))}) == {}
