import numpy as np
import pandas as pd
import pytest

from qvc.hubble import hubble_utils
from qvc.hubble.hubble_bright_subsample import BrightSubsampleCut
from qvc.hubble.hubble_model import AgnPivotContext


@pytest.mark.parametrize("prefix", ["fiducial", "restricted"])
@pytest.mark.parametrize("relative", [True, False])
@pytest.mark.parametrize(
    "threshold, margin, expected",
    [(0.1, 0.15, ("0.1", "10", "0.15")),
     (0.125, 0.25, ("0.125", "12.5", "0.25"))],
)
def test_bright_subsample_tex_exports(tmp_path, monkeypatch, prefix, relative,
                                     threshold, margin, expected):
    monkeypatch.setattr(hubble_utils, "fit_sigma_tau_lambda_broken_pl", lambda *a, **k: None)
    frame = pd.DataFrame({"z": [1.0, 4.0]})
    pivot = AgnPivotContext(
        observable_names=("log_sigma_uv", "log_tau_uv_rf"),
        values=(-1.0, np.log10(200)), reference_object_ids=("obj1",),
        z_range=(0.44, 3.16),
    )

    def export(**kwargs):
        hubble_utils.write_results_tex_variables(
            frame, frame, pd.DataFrame({"IS_CALIBRATOR": [], "zHD": []}),
            (0.44, 3.16), {}, {}, None, str(tmp_path),
            result_prefix=prefix, agn_pivot_context=pivot, **kwargs,
        )
        return (tmp_path / f"param_results_{prefix}.tex").read_text()

    disabled = export()
    cut = BrightSubsampleCut(
        completeness_min=threshold, margin=margin, relative=relative,
        z_grid=(0.5, 3.0), threshold_grid=(21.0, 22.0),
    )
    enabled = export(bright_subsample_cut=cut)
    for suffix, value in zip(
        ("CompletenessMin", "CompletenessPercent", "Margin"), expected
    ):
        command = rf"\newcommand{{\result{prefix.capitalize()}BrightSubsample{suffix}}}"
        assert command + rf"{{\ensuremath{{{value}}}}}" in enabled
        assert command + r"{\ensuremath{\mathrm{N/A}}}" in disabled
    mode = "relative-to-peak" if relative else "absolute"
    assert f"% Bright-subsample threshold: {mode} completeness" in enabled
    assert "% Bright-subsample selection: disabled" in disabled

    def existing_lines(tex):
        return [line for line in tex.splitlines()
                if "BrightSubsample" not in line and not line.startswith("% Bright-subsample")]

    assert existing_lines(enabled) == existing_lines(disabled)
    assert rf"\result{prefix.capitalize()}NumAGNFitted}}{{\ensuremath{{1}}}}" in enabled
