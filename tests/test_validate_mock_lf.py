import importlib.util
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qvc-validate-mock-lf-matplotlib")

from qvc.hubble.empirical_luminosity_functions import (
    KULKARNI2019_TYPE1_MODEL_IDS,
    build_empirical_lf,
)


VALIDATOR_PATH = ROOT / "scripts" / "validate_mock_lf.py"


def _load_validator_module():
    spec = importlib.util.spec_from_file_location("qvc_validate_mock_lf", VALIDATOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validator_uses_all_final_qvc_kulkarni_models_without_stale_loader():
    source = VALIDATOR_PATH.read_text(encoding="utf-8")

    assert "new_load_kk18_lf_shape" not in source
    assert "return_kk18_lf_fitted" not in source
    assert "Kulkarni2019_QLF" not in source

    validator = _load_validator_module()
    assert tuple(validator.KULKARNI2019_PLOT_STYLES) == tuple(
        KULKARNI2019_TYPE1_MODEL_IDS
    )

    m2500 = np.array([-27.0, -24.0, -21.0])
    redshift = 2.4
    evaluated = validator.evaluate_kulkarni2019_modes(m2500, redshift)
    np.testing.assert_allclose(
        validator.m1450_to_m2500(validator.m2500_to_m1450(m2500)),
        m2500,
        rtol=0.0,
        atol=2e-14,
    )
    assert tuple(evaluated) == tuple(KULKARNI2019_TYPE1_MODEL_IDS)

    native_m1450 = validator.m2500_to_m1450(m2500)
    for model_id in KULKARNI2019_TYPE1_MODEL_IDS:
        direct = build_empirical_lf(
            model_id,
            native_m1450,
            np.array([redshift]),
            validator.COSMO,
        )
        np.testing.assert_array_equal(evaluated[model_id], direct.phi_log10[0])
        assert np.all(np.isfinite(evaluated[model_id]))
