#!/usr/bin/env python3
"""Summarize the completed quick benchmark runs and plot marginal densities."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path('results/pantheon_reproduction')
fluid = json.loads((root/'fluid_quick/summary.json').read_text())
ppf = json.loads((root/'ppf_quick/summary.json').read_text())
reference = np.loadtxt(root/'reference/chain.txt.gz')[32000:, :5]
runs = [('Released chain', reference, 'black')]
for name, result, color in [('Fluid quick', fluid, 'tab:blue'), ('PPF quick', ppf, 'tab:orange')]:
    directory = 'fluid_quick' if result['dark_energy'] == 'fluid' else 'ppf_quick'
    chain = np.load(root/directory/'chain.npz')['chain'][result['burn']:].reshape(-1, 5)
    runs.append((name, chain, color))
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
for ax, index, label in zip(axes, [0, 2, 3], ['Om0', 'w0', 'wa']):
    edges = np.linspace(min(np.min(x[:, index]) for _, x, _ in runs), max(np.max(x[:, index]) for _, x, _ in runs), 50)
    for name, samples, color in runs:
        ax.hist(samples[:, index], bins=edges, density=True, histtype='step', color=color, label=name, linewidth=1.5)
    ax.set_xlabel(label)
    ax.tick_params(direction='in', top=True, right=True)
axes[0].set_ylabel('Marginal density')
axes[-1].legend(fontsize=8)
fig.savefig(root/'comparison.png', dpi=180)

def interval(s):
    return f"{s['median']:.3f} [{s['q16']:.3f}, {s['q84']:.3f}]"

lines = ['# Unanchored Pantheon+ quick reproduction', '',
         'Values are medians and central 68% intervals. These runs are implementation checks, not converged final inference.', '',
         '| Parameter | Released chain | New fluid run | New PPF run |',
         '|---|---:|---:|---:|']
for label in ['Om0', 'w0', 'wa', 'h', 'M']:
    lines.append('| '+label+' | '+' | '.join(interval(s[label]) for s in [fluid['reference']['summary'], fluid['summary'], ppf['summary']])+' |')
lines += ['', '## Configuration and checks', '',
          '- 1,590 light-curve measurements with zHD > 0.01; full selected STAT+SYS covariance.',
          '- Free M, no Cepheid-distance substitution, no external H0 likelihood, no AGN parameters.',
          '- Uniform bounds from the released chain header; h = H0 / (100 km/s/Mpc). The h posterior is not an absolute SN-only H0 measurement.',
          '- Fixed omega_b = 0.048, flat geometry, neutrino settings from the chain header with the legacy mass conversion.',
          '- Public fluid model rejects evolving phantom/crossing models; PPF changes that restriction. CAMB also restricts early-time dark energy and positive matter densities.',
          f"- Largest absolute log-likelihood difference at 24 released-chain points: {fluid['reference']['max_abs_loglike_difference']:.6f}. This is close agreement, not a bitwise historical-backend reproduction.",
          '- Three regression tests passed: covariance selection, released theory-method/precision comparison, prior bounds and crossing behavior.', '',
          '## Sampling diagnostics', '']
for name, result in [('Fluid', fluid), ('PPF', ppf)]:
    lines.append(f"- {name}: {result['walkers']} walkers, {result['steps']} steps, {result['burn']} discarded; {result['seconds']:.1f} seconds; mean acceptance {result['acceptance_fraction']:.3f}; post-burn length/tau range {min(result['post_burn_steps_over_tau']):.1f}–{max(result['post_burn_steps_over_tau']):.1f}. A customary target is >50 tau per walker; longer runs are needed.")
lines += ['', '## Public chain versus paper', '',
          'The paper quotes w0 approximately -0.93 +/- 0.15 and wa approximately -0.1 (+0.9/-2.0) for Pantheon+ alone. The downloaded public chain has much narrower wa support and different quantiles; reproducing it is not reproducing those paper numbers. Its header bounds wa to [-2, 2], whereas the quoted lower endpoint is approximately -2.1. The fluid backend further excludes the crossing region. PPF is a diagnostic alternative, not an established reconstruction of the paper configuration.', '',
          'Paper: https://arxiv.org/abs/2202.04077',
          'Reference release: https://github.com/PantheonPlusSH0ES/DataRelease/tree/c447f0fea703fcd0fff57de5000947b5ca81286b/Pantheon%2B_Data/5_COSMOLOGY/chains/Pantheon%2BOnly', '',
          'Reproducing the printed paper intervals requires resolving this public-chain/configuration discrepancy, rather than tuning the fit to the printed central values.', '',
          'Inputs and hashes: reference/manifest.json. Full diagnostics and split-chain summaries: fluid_quick/summary.json and ppf_quick/summary.json.']
(root/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
