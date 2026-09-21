"""Causal questionnaire-history predictors for interval sensitivities."""
from __future__ import annotations
import numpy as np
import pandas as pd

HALF={'PHQ9':13.5,'BDI':31.5,'GAD7':10.5}
EXTRA_FEATURES=['same_instrument_gap','last_score_change','last_score_change_per_day',
    'recent_score_mean','recent_score_sd','recent_score_range','recent_score_slope',
    'prior_history_span','same_instrument_count_14d','same_instrument_count_28d',
    'recall_overlap_fraction']

def add_history(events,retained,queries=None):
    """Reconstruct history using earlier sessions, including policy feedback.

    `queries` contains actually requested scores only. All values at the target
    session remain unavailable, even if another instrument was retained there.
    Neither the target column nor event-row ordering affects these predictors.
    """
    past=retained[['id','session','day','anchor','value']].copy()
    past['score']=(past.value-past.anchor.map(HALF))/past.anchor.map(HALF)
    if queries is not None and len(queries):
        q=queries[['id','session','day','anchor','target']].rename(columns={'target':'score'})
        past=pd.concat([past.drop(columns='value'),q],ignore_index=True)
    if past.duplicated(['id','session','anchor']).any():
        raise ValueError('History contains duplicate questionnaire events.')
    groups={key:g.sort_values('session') for key,g in past.groupby(['id','anchor'])}
    records=[]
    for r in events.itertuples():
        g=groups.get((r.id,r.anchor))
        g=g[g.session<r.session] if g is not None else pd.DataFrame()
        if len(g)==0:
            records.append({x:(0. if x.startswith('same_instrument_count') else np.nan) for x in EXTRA_FEATURES})
            continue
        recent=g.tail(3);values=recent.score.to_numpy();days=recent.day.to_numpy()
        gap=float(r.day-g.day.iloc[-1]);span=float(g.day.iloc[-1]-g.day.iloc[0])
        change=float(values[-1]-values[-2]) if len(values)>1 else np.nan
        dt=float(days[-1]-days[-2]) if len(days)>1 else 0.
        centered=days-days.mean();denominator=float(np.square(centered).sum())
        slope=float(np.dot(centered,values-values.mean())/denominator) if denominator>0 else np.nan
        records.append(dict(same_instrument_gap=gap,last_score_change=change,
            last_score_change_per_day=change/dt if dt>0 else np.nan,
            recent_score_mean=float(values.mean()),recent_score_sd=float(values.std()),
            recent_score_range=float(np.ptp(values)),recent_score_slope=slope,prior_history_span=span,
            same_instrument_count_14d=int((g.day>=r.day-14).sum()),
            same_instrument_count_28d=int((g.day>=r.day-28).sum()),
            recall_overlap_fraction=float(np.clip((14-gap)/14,0,1))))
    result=events.drop(columns=EXTRA_FEATURES,errors='ignore').copy()
    return pd.concat([result.reset_index(drop=True),pd.DataFrame(records)],axis=1)
