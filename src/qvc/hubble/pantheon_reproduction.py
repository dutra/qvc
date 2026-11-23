"""Isolated, unanchored Pantheon+ benchmark against the public flat-w0wa chain.

Requires camb and dynesty (or emcee when explicitly selected). This deliberately does not inherit QVC H0 priors,
AGN parameters, Cepheid distances, or the calibrator-inclusive sample mask.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from qvc.hubble.dynesty_settings import SPEED_CHOICES, get_dynesty_speed_settings, prior_transform_dynesty

LABELS = ('Om0', 'h', 'w0', 'wa', 'M')
BOUNDS = np.array([[.01, .9], [.55, .91], [-2., .5], [-2., 2.], [-20., -18.]])


def load_data(directory):
    directory = Path(directory)
    frame = pd.read_csv(directory / 'Pantheon+SH0ES.dat', sep=r'\s+')
    mask = (frame.zHD > .01).to_numpy()
    with open(directory / 'Pantheon+SH0ES_STAT+SYS.cov') as stream:
        n = int(stream.readline())
        covariance = np.loadtxt(stream).reshape(n, n)
    if n != len(frame):
        raise ValueError('Covariance and catalogue dimensions differ')
    return frame.loc[mask].copy(), covariance[np.ix_(mask, mask)]


class Likelihood:
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('camb', None)
        return state

    def __setstate__(self, state):
        import camb
        self.__dict__.update(state)
        self.camb = camb

    def __init__(self, directory, dark_energy='fluid'):
        import camb
        self.camb = camb
        self.dark_energy = dark_energy
        self.data, self.covariance = load_data(directory)
        self.factor = cho_factor(self.covariance, lower=True)
        self.z = self.data.zHD.to_numpy()
        self.zhel = self.data.zHEL.to_numpy()
        self.observed = self.data.m_b_corr.to_numpy()

    def theory(self, theta):
        om, h, w, wa, magnitude = theta
        p = self.camb.CAMBparams()
        # Fixed parameters in the released chain header; legacy neutrino mass
        # conversion. Contemporary CAMB is validated against saved log posteriors.
        p.set_cosmology(H0=100*h, ombh2=.048*h*h,
                        omch2=(om-.048)*h*h-.00083,
                        mnu=.00083*94.07, num_massive_neutrinos=3, nnu=3.046)
        p.set_dark_energy(w=w, wa=wa, dark_energy_model=self.dark_energy)
        background = self.camb.get_background(p, no_thermo=True)
        da = background.angular_diameter_distance(self.z)
        return 5*np.log10((1+self.z)*(1+self.zhel)*da)+25+magnitude

    def __call__(self, theta):
        theta = np.asarray(theta)
        if not np.all(np.isfinite(theta)) or np.any(theta < BOUNDS[:, 0]) or np.any(theta > BOUNDS[:, 1]):
            return -np.inf
        try:
            residual = self.observed - self.theory(theta)
        except (self.camb.CAMBError, ValueError):
            return -np.inf
        value = -.5 * residual @ cho_solve(self.factor, residual, check_finite=False)
        return float(value) if np.isfinite(value) else -np.inf


def summarize(samples):
    return {label: dict(zip(('q16', 'median', 'q84'), map(float, np.percentile(samples[:, i], [16, 50, 84]))))
            for i, label in enumerate(LABELS)}


def reference_check(likelihood, directory):
    chain = np.loadtxt(Path(directory)/'chain.txt.gz')
    # Header says 16 walkers. Remove first 2000 ensemble iterations, including
    # initialization transients; report the choice rather than hiding burn-in.
    retained = chain[32000:]
    retained = retained[np.isfinite(retained[:, -1])]
    chosen = retained[np.linspace(0, len(retained)-1, 24, dtype=int)]
    differences = np.array([likelihood(row[:5])-(row[-1]-row[-2]) for row in chosen])
    return dict(total_rows=len(chain), discarded_rows=32000,
                summary=summarize(retained[:, :5]),
                max_abs_loglike_difference=float(np.max(np.abs(differences))),
                loglike_differences=differences.tolist(),
                crossing_w_minus_one_count=int(np.sum((1+retained[:, 2])*(1+retained[:, 2]+retained[:, 3]) < 0)))


def run_dynesty(args, likelihood, reference):
    import camb
    import dynesty
    from dynesty import utils as dyfunc
    settings = get_dynesty_speed_settings(args.speed, len(LABELS))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in args.reference.iterdir() if p.is_file()}
    config = dict(sampler='dynesty', dark_energy=args.dark_energy,
                  parameter_order=list(LABELS), uniform_bounds=BOUNDS.tolist(),
                  sample_count=len(likelihood.data), external_H0_prior=False,
                  cepheid_distances=False, speed=args.speed, settings=settings,
                  bound='multi', sample='rwalk', update_interval=10*len(LABELS),
                  seed=args.seed, camb_version=camb.__version__,
                  dynesty_version=dynesty.__version__, reference_hashes=hashes)
    config_path = args.output/'config.json'
    checkpoint = args.output/'dynesty.checkpoint'
    if args.resume:
        if not config_path.exists() or json.loads(config_path.read_text()) != config:
            raise ValueError('Resume configuration or reference inputs differ from the saved run')
        sampler = dynesty.DynamicNestedSampler.restore(str(checkpoint))
    else:
        if config_path.exists() or checkpoint.exists():
            raise ValueError('Existing dynesty run: use --resume or choose a fresh output directory')
        config_path.write_text(json.dumps(config, indent=2)+'\n')
        priors = dict(zip(LABELS, BOUNDS))
        sampler = dynesty.DynamicNestedSampler(
            likelihood, partial(prior_transform_dynesty, priors=priors, model_labels=LABELS),
            len(LABELS), bound='multi', sample='rwalk', update_interval=10*len(LABELS),
            rstate=np.random.default_rng(args.seed))
    started = time.monotonic()
    sampler.run_nested(**settings, print_progress=True, resume=args.resume,
                       maxcall=args.maxcall, checkpoint_file=str(checkpoint), checkpoint_every=60)
    result = sampler.results
    weights = np.exp(result.logwt-result.logz[-1])
    weights /= weights.sum()
    flat = dyfunc.resample_equal(result.samples, weights, rstate=np.random.default_rng(args.seed))
    np.savez_compressed(args.output/'posterior.npz', samples=result.samples,
                        weights=weights, flat_samples=flat, logl=result.logl,
                        logwt=result.logwt, logz=result.logz, logzerr=result.logzerr, labels=LABELS)
    summary = dict(**config, reference=reference, summary=summarize(flat),
                   seconds_this_invocation=time.monotonic()-started,
                   likelihood_calls=int(np.sum(result.ncall)),
                   effective_samples=float(1/np.sum(weights**2)),
                   logz=float(result.logz[-1]), logzerr=float(result.logzerr[-1]),
                   maxcall=args.maxcall,
                   caveat='A maxcall-limited run is a smoke check, not convergence evidence. '
                          'Log evidence omits the parameter-independent Gaussian normalization; '
                          'do not compare it directly with the joint pipeline evidence.')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    import camb
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path('results/pantheon_reproduction/reference'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dark-energy', choices=('fluid', 'ppf'), default='fluid')
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--burn', type=int, default=600)
    parser.add_argument('--walkers', type=int, default=32)
    parser.add_argument('--seed', type=int, default=271828)
    parser.add_argument('--sampler', choices=('emcee', 'dynesty'), default='dynesty')
    parser.add_argument('--speed', choices=SPEED_CHOICES, default='production')
    parser.add_argument('--resume', action='store_true', help='Resume a saved dynesty checkpoint')
    parser.add_argument('--maxcall', type=int, help='Bound dynesty calls for smoke checks only')
    args = parser.parse_args()
    if args.sampler == 'emcee' and (not 0 <= args.burn < args.steps or args.walkers < 12):
        parser.error('Require 0 <= burn < steps and at least 12 walkers')
    if args.sampler == 'emcee' and (args.resume or args.maxcall is not None):
        parser.error('--resume and --maxcall apply only to dynesty')
    args.output.mkdir(parents=True, exist_ok=True)
    likelihood = Likelihood(args.reference, args.dark_energy)
    reference = reference_check(likelihood, args.reference)
    if reference['max_abs_loglike_difference'] > .05:
        raise RuntimeError(f'Reference likelihood mismatch: {reference}')
    print('Reference check:', json.dumps(reference), flush=True)
    if args.sampler == 'dynesty':
        run_dynesty(args, likelihood, reference)
        return
    import emcee
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    initial = []
    # Independent initialization: do not seed from the published posterior.
    while len(initial) < args.walkers:
        h = rng.uniform(.60, .86)
        theta = np.array([rng.uniform(.15, .36), h, rng.uniform(-.99, -.70),
                          rng.uniform(0, .40), -19.25+5*np.log10(h/.73)+rng.normal(0, .015)])
        if np.isfinite(likelihood(theta)):
            initial.append(theta)
    sampler = emcee.EnsembleSampler(args.walkers, 5, likelihood)
    started = time.monotonic()
    for i, _ in enumerate(sampler.sample(initial, iterations=args.steps)):
        if (i+1) % 100 == 0:
            print(f'{i+1}/{args.steps}, elapsed {time.monotonic()-started:.1f}s', flush=True)
    chain = sampler.get_chain()
    np.savez_compressed(args.output/'chain.npz', chain=chain, log_probability=sampler.get_log_prob(), labels=LABELS)
    flat = sampler.get_chain(discard=args.burn, flat=True)
    tau = emcee.autocorr.integrated_time(chain[args.burn:], quiet=True)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.reference.iterdir() if p.is_file()}
    result = dict(sample_count=len(likelihood.data), dark_energy=args.dark_energy,
                  parameter_order=LABELS, uniform_bounds=BOUNDS.tolist(),
                  external_H0_prior=False, cepheid_distances=False,
                  camb_version=camb.__version__, emcee_version=emcee.__version__,
                  seed=args.seed, steps=args.steps, burn=args.burn, walkers=args.walkers,
                  seconds=time.monotonic()-started, reference=reference,
                  summary=summarize(flat), first_half=summarize(chain[args.burn:(args.steps+args.burn)//2].reshape(-1, 5)),
                  second_half=summarize(chain[(args.steps+args.burn)//2:].reshape(-1, 5)),
                  acceptance_fraction=float(np.mean(sampler.acceptance_fraction)),
                  autocorrelation_times=tau.tolist(), effective_samples=(len(flat)/tau).tolist(),
                  post_burn_steps_over_tau=((args.steps-args.burn)/tau).tolist(),
                  reference_hashes=hashes,
                  caveat='Quick run; convergence requires substantially more than 50 autocorrelation times per walker. Contemporary CAMB is an approximate historical-backend match; inspect fixed-point errors.')
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
