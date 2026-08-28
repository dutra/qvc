"""Fit QVC light curves with the shared-driver disk plus delayed model.

This is a convenience entry point for ``fit_light_curves``. It selects
``--model_variant shared_latent_blr`` unless the caller supplied an explicit
model variant, while retaining all standard light-curve fitting arguments.
"""

from __future__ import annotations

import sys

from qvc.light_curve.fit_light_curves import main as _main


def main():
    if "--model_variant" not in sys.argv[1:]:
        sys.argv[1:1] = ["--model_variant", "shared_latent_blr"]
    _main()


if __name__ == "__main__":
    main()
