"""Further seeds (App. F, repeated route generations): headline numbers from the released tables.

seeds/seeds_summary.csv        per model and seed: route outcomes and Composition score on the 240 routes of the first
                               view of every scene (seed = 0, 1, 2, pooled)
seeds/seed_spread.csv          per route completed under all three seeds: pairwise distances of the three ends, their
                               spread (root mean square), the end error of each seed (strides of seed 0); summary rows
seeds/repeatability.csv        agreement of route outcomes and correlation of error vectors between seeds
seeds/route_coverage.csv       outcome of every route under each seed

usage: python3 scoring/seed_tables.py DATA_RELEASE_DIR
"""
import itertools, sys
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau

UNRANKED = {'alaya-world'}      # stride near the measurement noise (Table 2, dagger)
MIN_ROUTES = 40                 # models with at least this many routes completed under all three seeds


def main(rel):
    D = Path(rel) / 'seeds'
    S = pd.read_csv(D / 'seeds_summary.csv', dtype={'seed': str})
    C = S[S.seed != 'pooled'].pivot(index='model', columns='seed', values='composition')
    rng = C.max(axis=1) - C.min(axis=1)
    ranked = rng.drop(index=[m for m in UNRANKED if m in rng.index])
    print('Composition score per seed\n' + C.assign(range=rng).round(1).to_string())
    print(f'\nvariation across seeds, {len(ranked)} ranked models: median {ranked.median():.1f}, '
          f'max {ranked.max():.1f} ({ranked.idxmax()})')
    R = C.drop(index=[m for m in UNRANKED if m in C.index])
    for a, b in itertools.combinations(C.columns, 2):
        print(f'Kendall tau, seeds {a} and {b}, {len(R)} ranked models: {kendalltau(R[a], R[b])[0]:.3f}')
    P = pd.read_csv(D / 'seed_spread.csv')
    Q = P[P.row_type == 'route']
    g = Q.groupby('model').agg(routes=('spread', 'size'), spread=('spread', 'median'), error=('error_median', 'median'),
                               error_over_spread=('error_over_spread', 'median'))
    print('\nRoutes completed under all three seeds: median spread of the three ends, median end error (strides of '
          'seed 0), and the median over routes of error / spread\n' + g.round(2).to_string())
    k = g[(g.routes >= MIN_ROUTES) & ~g.index.isin(UNRANKED)]
    print(f'\nend error over spread, {len(k)} ranked models with >= {MIN_ROUTES} such routes: '
          f'{k.error_over_spread.min():.2f} to {k.error_over_spread.max():.2f}')


if __name__ == '__main__':
    main(sys.argv[1])
