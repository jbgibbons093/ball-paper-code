"""Aggregate-only check of the proposed clinic-exclusive temporal partition."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import BALL
BALL._install_virtual_package()
E = sys.modules['empirical.fit_empirical']


def dated_map(sessions):
    dates = pd.to_datetime(sessions.date, errors='coerce')
    spans = sessions.assign(_date=dates).groupby('id')._date.agg(['min', 'max'])
    cutoff = pd.Timestamp('2025-01-01')
    mapping = {int(pid): 'train' for pid, row in spans.iterrows()
               if pd.notna(row['max']) and row['max'] < cutoff}
    mapping.update({int(pid): 'test' for pid, row in spans.iterrows()
                    if pd.notna(row['min']) and row['min'] >= cutoff})
    return mapping


def interval_roles(mapping):
    ids = np.asarray(sorted(pid for pid, role in mapping.items() if role == 'test'))
    np.random.default_rng(20250101).shuffle(ids)
    return {int(pid): ('calibration' if n < len(ids)//2 else 'test')
            for n, pid in enumerate(ids)}


def audit(require_live_fix=False):
    sessions = pd.read_csv(E.OUT/'sessions.csv',
                          usecols=['id', 'split', 'date', 'FacilityName'])
    sessions['clinic'] = sessions.FacilityName.map(E._canonical_empirical_clinic)
    primary, _, _ = E._primary_empirical_partitions(sessions)
    eligible = {int(pid) for pid, role in primary.items() if role == 'train'}
    old_source = sessions[sessions['split'].isin(['train', 'val'])]
    new_source = sessions[sessions.id.isin(eligible)]
    old = dated_map(old_source)
    corrected = dated_map(new_source)
    live = E._generalization_split_maps(sessions)['forward_2025']
    retained = pd.read_csv(E.OUT/'anchors_sparse_14d.csv', usecols=['id', 'role'])
    modeled_ids = set(retained.loc[retained.role.eq('anchor'), 'id'].astype(int))
    source = sessions.groupby('id')['split'].first()
    dominant = sessions.groupby('id').clinic.agg(lambda values: values.value_counts().index[0])
    clinic_map = (pd.DataFrame({'clinic': dominant, 'split': source})
                  .drop_duplicates().set_index('clinic')['split'].to_dict())
    allowed_clinics = {clinic for clinic, role in clinic_map.items() if role in ['train', 'val']}
    result = {'cutoff': '2025-01-01', 'partition_seed': 20250101,
              'live_matches_corrected_reference': live == corrected,
              'later_patient_set_preserved': {p for p, r in old.items() if r == 'test'} ==
                                            {p for p, r in corrected.items() if r == 'test'},
              'later_calibration_test_assignments_preserved': interval_roles(old) == interval_roles(corrected),
              'primary_eligible_development_patients': len(eligible)}
    for label, mapping, source_frame in [('old', old, old_source), ('corrected', corrected, new_source)]:
        train_ids = {pid for pid, role in mapping.items() if role == 'train'}
        later_ids = {pid for pid, role in mapping.items() if role == 'test'}
        modeled = source_frame[source_frame.id.isin(train_ids & modeled_ids)]
        outside = modeled[~modeled.clinic.isin(allowed_clinics)]
        roles = interval_roles(mapping)
        result[label] = {
            'source_training_patients': len(train_ids),
            'modeled_training_patients': int(modeled.id.nunique()),
            'source_later_patients': len(later_ids),
            'modeled_later_patients': len(later_ids & modeled_ids),
            'training_source_training_partition_patients': int(sum(source[p] == 'train' for p in train_ids)),
            'training_source_selection_partition_patients': int(sum(source[p] == 'val' for p in train_ids)),
            'source_spanning_patients': int(source_frame.id.nunique()-len(mapping)),
            'actual_training_clinics': int(modeled.clinic.nunique()),
            'training_patients_outside_primary_eligibility': len(set(modeled.id)-eligible),
            'training_sessions_outside_development_clinics': len(outside),
            'training_patients_with_outside_sessions': int(outside.id.nunique()),
            'training_outside_clinics': int(outside.clinic.nunique()),
            'assigned_calibration_patients': sum(role == 'calibration' for role in roles.values()),
            'assigned_final_test_patients': sum(role == 'test' for role in roles.values())
        }
    assert result['later_patient_set_preserved']
    assert result['later_calibration_test_assignments_preserved']
    assert result['corrected']['training_sessions_outside_development_clinics'] == 0
    assert result['corrected']['training_patients_outside_primary_eligibility'] == 0
    for pid, role in corrected.items():
        dates = pd.to_datetime(new_source.loc[new_source.id.eq(pid), 'date'])
        assert (dates.max() < pd.Timestamp('2025-01-01') if role == 'train'
                else dates.min() >= pd.Timestamp('2025-01-01'))
    if require_live_fix and not result['live_matches_corrected_reference']:
        raise AssertionError('Apply the documented shared BALL.py temporal partition patch before fitting.')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--require-live-fix', action='store_true')
    args = p.parse_args()
    result = audit(args.require_live_fix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
