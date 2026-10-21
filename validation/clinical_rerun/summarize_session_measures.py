"""Aggregate summary of the five session measures in the cohort after clinic exclusions (Supplementary Table 2 rows).

Reads the session table, keeps the clinic-exclusive cohort used by every clinical analysis, and writes counts only.
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'empirical'))
from revise_uncertainty import E   # noqa: E402  (installs BALL's virtual package)

sessions = E._coerce_sessions(pd.read_csv(E.OUT / 'sessions.csv'))
splits, _, _ = E._primary_empirical_partitions(sessions)
cohort = sessions[sessions.id.isin(set(splits))]
values = {col: pd.to_numeric(cohort[col], errors='coerce') for col in E.SESSION_MEASURE_COLS}
intensity = values['mt_standardized']   # percent of motor threshold in the source


def sessions_and_patients(mask):
    return {'sessions': int(mask.sum()), 'patients': int(cohort.loc[mask, 'id'].nunique())}


summary = {
    'patients': int(cohort.id.nunique()),
    'sessions': int(len(cohort)),
    'sessions_with_every_measure_recorded': int(pd.concat(values, axis=1).notna().all(axis=1).sum()),
    'intensity_percent_motor_threshold_quartiles': [float(q) for q in intensity.quantile([.25, .5, .75])],
    'taking_medication': sessions_and_patients(values['taking_meds_current'].eq(1)),
    'current_suicidal_ideation_noted': sessions_and_patients(values['si_current'].gt(0)),
    'current_suicidal_ideation_values': {str(int(k)): int(n) for k, n in values['si_current'].value_counts().sort_index().items()},
    'recent_suicidal_ideation_noted': sessions_and_patients(values['si_recent'].eq(1)),
    'course_after_first': sessions_and_patients(values['course_number'].gt(1)),
    'course_number_max': int(values['course_number'].max()),
}
(HERE / 'aggregates/session_measure_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
