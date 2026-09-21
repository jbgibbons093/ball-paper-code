"""Aggregate policy replays and recalibrate intervals after policy feedback."""
from pathlib import Path
import argparse
import pickle
import json
import numpy as np
import pandas as pd
from conditional_intervals import predict,qpatient,pmean,band

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--frozen-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--interval-run',type=Path)
    p.add_argument('--refined',action='store_true')
    args=p.parse_args()
    interval_run=args.interval_run or args.run
    interval_predict=predict
    if args.refined:
        from refine_intervals import predict as interval_predict
        from interval_history import add_history
    measurement=json.loads((args.frozen_run/'measurement_calibration.json').read_text())
    scales=measurement['development_data']['normalized_score_sd_by_instrument']
    metrics=[];severity=[];pairs=[]
    fixed={}
    for file in args.run.glob('adaptive_events_*d_fixed_schedule.csv'):
        frame=pd.read_csv(file);frame=frame[frame.role.eq('test')].copy()
        frame['sqerror']=((frame.ball-frame.target)/frame.anchor.map(scales))**2
        fixed[int(frame.cadence.iloc[0])]=frame
    for file in sorted(args.run.glob('adaptive_events_*.csv')):
        d=pd.read_csv(file);cad=int(d.cadence.iloc[0]);policy=d.policy.iloc[0];test_parts=[]
        if args.refined:
            retained=pd.read_csv(Path(__file__).resolve().parent/'derived'/f'anchors_sparse_{cad}d.csv')
            retained=retained[retained.role.eq('anchor')]
            d=add_history(d,retained,d[d['query']])
        for inst,g in d.groupby('anchor'):
            c=g[g.role.eq('calibration')];t=g[g.role.eq('test')].copy()
            with (interval_run/f'interval_models_{cad}d_{inst}.pkl').open('rb') as f:models=pickle.load(f)
            cl,ch=interval_predict(models,c);tl,th=interval_predict(models,t)
            q=qpatient(c,np.maximum(cl-c.target,c.target-ch))
            t['lower']=np.clip(tl-q,-1,1);t['upper']=np.clip(th+q,-1,1)
            t['covered']=(t.target>=t.lower)&(t.target<=t.upper)
            t['width_fraction']=np.maximum(t.upper-t.lower,0)/2
            t['sqerror']=((t.ball-t.target)/scales[inst])**2
            test_parts.append(t)
        t=pd.concat(test_parts,ignore_index=True)
        if args.refined:
            if interval_run.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
                raise ValueError('Patient-level outputs must remain outside the source repository.')
            t.to_csv(interval_run/f'adaptive_intervals_{cad}d_{policy}.csv',index=False)
        baseline=fixed[cad]
        match=t.merge(baseline,on=['id','session','anchor'],suffixes=('','_base'))
        assert len(match)==len(t)==len(baseline)
        np.testing.assert_allclose(match.target,match.target_base)
        if policy=='fixed_schedule':
            replay=pd.read_csv(args.run/f'replayed_events_{cad}d.csv')
            comparison=t.merge(replay[replay.role.eq('test')],on=['id','session','anchor'],suffixes=('','_replay'))
            np.testing.assert_allclose(comparison.ball,comparison.ball_replay,atol=3e-6)
            np.testing.assert_allclose(comparison.sd,comparison.sd_replay,atol=3e-6)
        per=match.groupby('id')[['sqerror','sqerror_base']].mean()
        rng=np.random.default_rng(20260918)
        indices=rng.integers(0,len(per),size=(2000,len(per)))
        values=per.to_numpy()
        diff=np.sqrt(values[indices,0].mean(axis=1))-np.sqrt(values[indices,1].mean(axis=1))
        lo,hi=np.quantile(diff,[.025,.975])
        qevents=t[t['query']]
        metrics.append(dict(cadence=cad,policy=policy,n_events=len(t),n_patients=t.id.nunique(),
            extra_measurements=int(t['query'].sum()),extra_measurement_fraction=float(t['query'].mean()),
            extra_measurements_per_patient=float(t.groupby('id')['query'].sum().mean()),
            sessions_with_extra_measurement=qevents[['id','session']].drop_duplicates().shape[0],
            rmse=float(np.sqrt(pmean(t,t.sqerror))),coverage=pmean(t,t.covered),
            mean_width_fraction=float(t.width_fraction.mean()),
            paired_rmse_change=float(np.sqrt(per.sqerror.mean())-np.sqrt(per.sqerror_base.mean())),
            paired_rmse_change_ci_low=float(lo),paired_rmse_change_ci_high=float(hi)))
        for definition,column in [('observed_score','target'),('first_retained_score','baseline_score'),('last_retained_score','last_score')]:
            for b in range(3):
                g=t[t[column].notna() & (band(t[column].to_numpy())==b)]
                if len(g):severity.append(dict(cadence=cad,policy=policy,definition=definition,band=b,
                    coverage=pmean(g,g.covered),rmse=float(np.sqrt(pmean(g,g.sqerror))),
                    n_events=len(g),n_patients=g.id.nunique()))
    args.output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(metrics).to_csv(args.output/'adaptive_reassessment_metrics.csv',index=False)
    pd.DataFrame(severity).to_csv(args.output/'adaptive_severity_metrics.csv',index=False)
    print(pd.DataFrame(metrics).to_string(index=False))

if __name__=='__main__':main()
