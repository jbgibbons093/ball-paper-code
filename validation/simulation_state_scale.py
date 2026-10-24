"""Describe the simulated true states of the 30 primary datasets (Supplementary Methods 3.1 and Table 1 notes).

Regenerates each primary dataset from the finalized arguments (seeds 2027-2029, RDoC shares 0.10 and 0.25, five
settings) and reports, across the 30 datasets:
- the standard deviation of the true composite state over test-patient sessions, with and without the accumulated
  direct RDoC contribution
- the share of within-patient variance carried by the slow component in the linear setting
- the fraction of days with an observed RDoC profile
- the number of treatments per patient

Uses simulated data only.
"""
import argparse
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'validation'))
import direct_rdoc_benchmark as B  # noqa: E402

SEEDS = (2027, 2028, 2029)
SHARES = (0.10, 0.25)
CELLS = ('linear', 'interaction', 'nonlinear', 'heterogeneous', 'missingness')
# Values from validation/finalized_simulation_arguments.json.
ARGS = dict(n=150, t=84, slow_fraction=0.75, delta_ar=0.3, rdoc_active_dims=3, rdoc_beta_seed=31011, rdoc_min_abs=0.35,
            missing_daily_probability=0.55, missing_mnar_gamma=0.7, missing_proxy_noise=0.4,
            missing_density_multiplier=1.5, anchor_observation='gaussian', irt_n_items=9, irt_discrimination=1.5,
            interaction_strength=1.5, nonlinear_strength=0.75, heterogeneity_strength=0.5, n_subtypes_for_scale=3)


def describe(values):
    values = np.asarray(values, dtype=float)
    return f'median {np.median(values):.3f}, range {values.min():.3f} to {values.max():.3f}'


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    args = types.SimpleNamespace(**ARGS)
    sd, sd_without_rdoc, observed, slow_share, treatments = [], [], [], [], []
    for seed in SEEDS:
        for share in SHARES:
            for cell in CELLS:
                data = B.H.generate_dataset(B._base_config(args, seed, share, cell))
                data = B._inject_benchmark_cell(data, cell, share, args)
                comp = data.components
                test = set(data.individuals.loc[data.individuals.split == 'test', 'id'])
                c = comp[comp.id.isin(test)]
                sd.append(c.L.std())
                sd_without_rdoc.append((c.L - c.rdoc_direct_level).std())
                observed.append(comp.proxy_observed.mean())
                treatments.extend(data.treatments.groupby('id').size().tolist())
                if cell == 'linear':
                    for _, g in comp.groupby('id'):
                        fast = 0.5 * ((g.delta_d - g.rdoc_direct_level) + (g.delta_p - g.rdoc_direct_level))
                        slow_share.append(g.slow.var() / (g.slow.var() + fast.var()))
    print(f'Datasets: {len(sd)}')
    print(f'True composite-state SD over test-patient sessions: {describe(sd)}')
    print(f'Same SD without the accumulated RDoC contribution: {describe(sd_without_rdoc)}')
    print(f'Fraction of days with an observed RDoC profile: {describe(observed)}')
    print(f'Within-patient slow-component share, linear setting: {describe(slow_share)}')
    print(f'Treatments per patient: {min(treatments)} to {max(treatments)}')


if __name__ == '__main__':
    main()
