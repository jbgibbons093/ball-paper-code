"""Share of analytic sessions that precede a patient's first note (Supplementary Methods 4.10).

These sessions receive RDoC scores of 0 in the RDoC transition refit. The script reads the session table and the RDoC
proxy table from the protected empirical workspace through BALL's own loaders, then prints aggregate counts only.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import BALL  # noqa: E402

BALL._install_virtual_package()
E = sys.modules['empirical.fit_empirical']


def main():
    sessions = E._coerce_sessions(pd.read_csv(E.OUT / 'sessions.csv'))
    splits, _, _ = E._primary_empirical_partitions(sessions)
    ids = {int(k) for k in splits}
    rdoc = E._load_rdoc_sessions()
    rdoc = rdoc[rdoc.id.isin(ids)].drop_duplicates(['id', 'day'], keep='last')   # the refit keys scores by (id, day)
    merged = sessions[sessions.id.isin(ids)][['id', 'day']].merge(rdoc[['id', 'day', 'rdoc_observed']],
                                                                   on=['id', 'day'], how='left')
    observed = merged.rdoc_observed.fillna(False).astype(bool)
    print(f'Analytic sessions: {len(merged)}; before a first note: {int((~observed).sum())} '
          f'({100 * (1 - observed.mean()):.1f}%)')


if __name__ == '__main__':
    main()
