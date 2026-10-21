"""Check that the session measures enter the clinical dataset from the previous recorded session, on their
declared scale (stimulation intensity divided by 100). Aggregates only."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import BALL  # noqa: E402

BALL._install_virtual_package()
E = sys.modules['empirical.fit_empirical']
PROTECTED = Path(os.environ['BALL_PROTECTED_ROOT'])
measurement = json.loads((PROTECTED / 'publication_ready_20260809_v12_uniform_cadence/measurement_calibration.json')
                         .read_text(encoding='utf-8-sig'))   # any valid calibration; this check concerns covariates
sessions = E._coerce_sessions(pd.read_csv(E.OUT / 'sessions.csv'))
anchors = pd.read_csv(E.OUT / 'anchors_sparse_14d.csv')
features = E._load_comorbidity_sessions(PROTECTED / 'session_comorbidity_features.csv')
data, _ = E._empirical_neural_dataset(anchors, sessions, None, features, measurement)

first = len(E.EMPIRICAL_FEATURE_NAMES) - len(E.SESSION_MEASURE_COLS)
report = {'p_daily': int(data.config.p_daily), 'feature_names': E.EMPIRICAL_FEATURE_NAMES[first:]}
daily = data.daily.sort_values(['id', 't'])
by_patient = {int(pid): g.sort_values(['session', 'day']).reset_index(drop=True) for pid, g in sessions.groupby('id')}
mismatches, first_available, checked = 0, 0, 0
for pid, rows in daily.groupby('id'):
    source = by_patient[int(pid)]
    n = len(source)
    for k, col in enumerate(E.SESSION_MEASURE_COLS):
        x = rows[f'X{first + k}'].to_numpy()[:n]
        available = rows[f'input_X{first + k}'].to_numpy()[:n]
        first_available += int(available[0])
        expected = source[col].to_numpy(dtype=float)[:-1] / E.SESSION_MEASURE_DIVISORS.get(col, 1.0)
        observed = x[1:]
        ok = available[1:] & np.isfinite(expected)
        mismatches += int((~np.isclose(observed[ok], expected[ok])).sum())
        checked += int(ok.sum())
ranges = {}
for k, name in enumerate(report['feature_names']):
    values = daily.loc[daily[f'input_X{first + k}'], f'X{first + k}']
    ranges[name] = [round(float(q), 3) for q in values.quantile([0, .01, .5, .99, 1])]
report.update(values_checked=checked, lag_mismatches=mismatches, first_session_marked_available=first_available,
              availability_fraction={name: round(float(daily[f'input_X{first + k}'].mean()), 3)
                                     for k, name in enumerate(report['feature_names'])},
              available_value_quantiles_0_1_50_99_100=ranges)
print(json.dumps(report, indent=1))
assert mismatches == 0 and first_available == 0
assert max(max(r) for r in ranges.values()) <= 8, 'a session measure is far above the scale of the others'
