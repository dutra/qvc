import argparse
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
os.chdir(SRC)
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from validate_erlang_end_to_end_recovery import resolve_kernel_models


def _args(**overrides):
    values = {
        "kernel_model": None,
        "injection_kernel_model": None,
        "recovery_kernel_model": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_split_kernel_models_support_drw_injection_carma21_recovery():
    args = _args(
        injection_kernel_model="drw",
        recovery_kernel_model="carma21",
    )
    resolve_kernel_models(args)
    assert args.injection_kernel_model == "drw"
    assert args.recovery_kernel_model == "carma21"


def test_kernel_models_default_to_previous_carma21_behavior():
    args = _args()
    resolve_kernel_models(args)
    assert args.injection_kernel_model == "carma21"
    assert args.recovery_kernel_model == "carma21"


def test_legacy_kernel_model_still_sets_both_paths():
    args = _args(kernel_model="legacy")
    resolve_kernel_models(args)
    assert args.injection_kernel_model == "legacy"
    assert args.recovery_kernel_model == "legacy"


def test_conflicting_compatibility_and_split_flags_are_rejected():
    args = _args(kernel_model="legacy", recovery_kernel_model="carma21")
    with pytest.raises(ValueError, match="conflicts"):
        resolve_kernel_models(args)
