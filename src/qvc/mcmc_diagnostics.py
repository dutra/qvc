"""Shared NumPyro posterior-summary helpers.

The helpers in this module deliberately separate computation from rendering so
that persisted convergence diagnostics are taken from the exact same summary
dictionary that is shown to the user.
"""

from __future__ import annotations

from itertools import product

import numpy as np


def compute_numpyro_summary(samples, *, group_by_chain=True, prob=0.90):
    """Return one NumPyro summary, or an empty mapping when unavailable."""

    if not samples:
        return {}
    try:
        from numpyro.diagnostics import summary

        return summary(samples, prob=prob, group_by_chain=group_by_chain)
    except (AssertionError, ValueError):
        # Split R-hat needs at least four draws.  Short smoke fits and
        # non-MCMC inference should remain serializable.
        return {}


def print_numpyro_summary_dict(summary_dict, *, heading=None):
    """Render a precomputed NumPyro summary without recomputing diagnostics."""

    if heading:
        print(f"\n{heading}")
    if not summary_dict:
        print("Summary unavailable: at least four posterior draws are required.")
        print()
        return

    first_stats = next(iter(summary_dict.values()))
    columns = [""] + list(first_stats.keys())
    row_names = {}
    for name, stats in summary_dict.items():
        shape = np.asarray(stats["mean"]).shape
        row_names[name] = name + "[" + ",".join(str(size - 1) for size in shape) + "]"
    max_len = max(max(map(len, row_names.values())), 10)
    name_format = "{:>" + str(max_len) + "}"
    header_format = name_format + " {:>9}" * (len(columns) - 1)
    row_format = name_format + " {:>9.2f}" * (len(columns) - 1)

    print()
    print(header_format.format(*columns))
    for name, stats in summary_dict.items():
        shape = np.asarray(stats["mean"]).shape
        if not shape:
            print(row_format.format(name, *stats.values()))
            continue
        for idx in product(*map(range, shape)):
            idx_text = "[{}]".format(",".join(map(str, idx)))
            print(row_format.format(name + idx_text, *[np.asarray(v)[idx] for v in stats.values()]))
    print()


def convergence_fields(summary_dict, field_map):
    """Extract scalar NumPyro R-hat/ESS values under flat catalog names.

    Parameters
    ----------
    summary_dict
        Output from :func:`compute_numpyro_summary`.
    field_map
        Mapping from output quantity name to its site name in ``summary_dict``.
    """

    out = {}
    for output_name, summary_name in field_map.items():
        rhat = np.nan
        ess = np.nan
        stats = summary_dict.get(summary_name)
        if stats is not None:
            rhat_arr = np.asarray(stats.get("r_hat", np.nan), dtype=float)
            ess_arr = np.asarray(stats.get("n_eff", np.nan), dtype=float)
            if rhat_arr.size == 1:
                rhat = float(rhat_arr.reshape(-1)[0])
            if ess_arr.size == 1:
                ess = float(ess_arr.reshape(-1)[0])
        out[f"{output_name}_rhat"] = rhat
        out[f"{output_name}_ess"] = ess
    return out
