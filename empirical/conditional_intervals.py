"""Development-selected questionnaire intervals for frozen BALL predictions.

The latent point estimator is unchanged. Questionnaire residual distributions
are fitted on withheld development events. Calibration patients only determine
conformal corrections; test patients never select a model or threshold.
"""
from pathlib import Path
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

FEATURES=['ball','sd','gap','prior_count','same_instrument_prior_count','last_score',
          'baseline_score','same_channel_prior_count','day','window_n_sessions']
GRID=[{'max_leaf_nodes':7,'max_iter':100,'min_samples_leaf':100},
      {'max_leaf_nodes':15,'max_iter':100,'min_samples_leaf':100},
      {'max_leaf_nodes':7,'max_iter':200,'min_samples_leaf':100}]

def weights(g):return 1/g.groupby('id').id.transform('size').to_numpy()
def pmean(g,x):return float(pd.Series(np.asarray(x),index=g.index).groupby(g.id).mean().mean())
def qpatient(g,scores,alpha=.05):
    n=g.id.nunique(); rank=np.ceil((n+1)*(1-alpha))
    if rank>n:return np.inf
    s=np.asarray(scores); w=weights(g); order=np.argsort(s,kind='stable')
    return float(s[order[min(np.searchsorted(np.cumsum(w[order]),rank),len(s)-1)]])

def fit(g,params):
    y=g.target-g.ball;w=weights(g)*g.id.nunique()**-1*len(g)
    models=[]
    for quantile in [.025,.975]:
        model=HistGradientBoostingRegressor(loss='quantile',quantile=quantile,
                learning_rate=.05,l2_regularization=1.,early_stopping=False,
                random_state=20260918,**params)
        model.fit(g[FEATURES],y,sample_weight=w);models.append(model)
    return models

def predict(models,g):
    a=models[0].predict(g[FEATURES])+g.ball.to_numpy()
    b=models[1].predict(g[FEATURES])+g.ball.to_numpy()
    return np.minimum(a,b),np.maximum(a,b)

def interval_score(y,lo,hi):
    return hi-lo+40*np.maximum(lo-y,0)+40*np.maximum(y-hi,0)

def band(x):return np.select([x<=-.5,x<=.5],[0,1],default=2)

def severity_inverted_interval(lo,hi,corrections):
    """Invert every candidate score band before observing a test outcome.

    The convex hull includes the complete bandwise prediction set. Thus no
    unknown test score is used to select its correction or its interval.
    """
    lowers=[];uppers=[]
    for b,(a,z) in enumerate([(-1.,-.5),(-.5,.5),(.5,1.)]):
        left=np.maximum(lo-corrections[b],a)
        right=np.minimum(hi+corrections[b],z)
        ok=left<=right
        lowers.append(np.where(ok,left,np.inf));uppers.append(np.where(ok,right,-np.inf))
    left=np.min(lowers,axis=0);right=np.max(uppers,axis=0)
    empty=left>right
    # Empty candidate sets remain empty for diagnostics rather than being
    # silently turned into full-score intervals.
    return left,right,empty

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--aggregate-output',type=Path,required=True)
    args=parser.parse_args()
    args.aggregate_output.mkdir(parents=True,exist_ok=True)
    specification={'features':FEATURES,'candidate_grid':GRID,'selection':'minimum patient-balanced interval score in development source-val clinics',
        'refit':'all development patients; withheld questionnaire outcomes only','calibration':'reserved calibration patients',
        'point_estimator':'unchanged BALL student','target':'future observed questionnaire score',
        'nominal_coverage':.95,'severity_bands':[-.5,.5],
        'severity_inversion':'evaluate all candidate outcome bands and take convex hull; never choose using test outcome',
        'seed':20260918}
    (args.run/'interval_specification.json').write_text(json.dumps(specification,indent=2))
    summaries=[];selections=[];groups=[]
    for source in sorted(args.run.glob('replayed_events_*d.csv')):
        d=pd.read_csv(source)
        cadence=int(d.cadence.iloc[0]);parts=[]
        for instrument,g in d.groupby('anchor'):
            dev=g[g.role.eq('development')]
            a=dev[dev.source_split.eq('train')];v=dev[dev.source_split.eq('val')]
            c=g[g.role.eq('calibration')];t=g[g.role.eq('test')]
            assert not (set(a.id)&set(v.id) or set(dev.id)&set(c.id) or set(dev.id)&set(t.id) or set(c.id)&set(t.id))
            scores=[]
            for number,params in enumerate(GRID):
                models=fit(a,params);lo,hi=predict(models,v)
                value=pmean(v,interval_score(v.target,np.clip(lo,-1,1),np.clip(hi,-1,1)))
                scores.append(value)
                selections.append(dict(cadence=cadence,instrument=instrument,candidate=number,validation_interval_score=value,**params))
            choice=int(np.argmin(scores));models=fit(dev,GRID[choice])
            with (args.run/f'interval_models_{cadence}d_{instrument}.pkl').open('wb') as f:pickle.dump(models,f)
            cl,ch=predict(models,c);tl,th=predict(models,t)
            cs=np.maximum(cl-c.target.to_numpy(),c.target.to_numpy()-ch)
            q=qpatient(c,cs)
            old_q=qpatient(c,abs(c.target-c.ball)/c.sd)
            history_c=band(c.last_score.fillna(c.ball).to_numpy())
            history_t=band(t.last_score.fillna(t.ball).to_numpy())
            history_q={b:qpatient(c[history_c==b],cs[history_c==b]) for b in range(3)}
            outcome_c=band(c.target.to_numpy())
            outcome_q={b:qpatient(c[outcome_c==b],cs[outcome_c==b]) for b in range(3)}
            invlo,invhi,empty=severity_inverted_interval(tl,th,outcome_q)
            methods={
                'covariance_corrected_symmetric':(t.ball-old_q*t.sd,t.ball+old_q*t.sd),
                'conditional_quantiles':(tl-q,th+q),
                'conditional_quantiles_history_groups':(tl-np.array([history_q[b] for b in history_t]),th+np.array([history_q[b] for b in history_t])),
                'conditional_quantiles_score_band_inversion':(invlo,invhi),
            }
            corrections={'selected_candidate':choice,'global':q,'history':history_q,'outcome':outcome_q,
                         'n_development_patients':dev.id.nunique(),'n_calibration_patients':c.id.nunique()}
            (args.run/f'interval_corrections_{cadence}d_{instrument}.json').write_text(json.dumps(corrections,indent=2))
            for method,(lower,upper) in methods.items():
                lo=np.clip(np.asarray(lower),-1,1);hi=np.clip(np.asarray(upper),-1,1)
                # Preserve genuine empty sets after clipping.
                is_empty=np.asarray(lower)>np.asarray(upper)
                coverage=(t.target.to_numpy()>=lo)&(t.target.to_numpy()<=hi)&~is_empty
                width=np.maximum(hi-lo,0)/2; width[is_empty]=0
                out=t[['cadence','id','session','anchor','target','ball','last_score','baseline_score','day']].copy()
                out['method']=method;out['lower']=lo;out['upper']=hi;out['covered']=coverage;out['width_fraction']=width
                parts.append(out)
                summaries.append(dict(cadence=cadence,instrument=instrument,method=method,coverage=pmean(t,coverage),
                    width_fraction=float(width.mean()),interval_score=pmean(t,interval_score(t.target,lo,hi)),
                    n_events=len(t),n_patients=t.id.nunique(),empty_fraction=float(is_empty.mean())))
            print(f'{cadence}d {instrument}: fitted development-selected intervals; candidate {choice}',flush=True)
        scored=pd.concat(parts,ignore_index=True)
        scored.to_csv(args.run/f'revised_intervals_{cadence}d.csv',index=False)
        for definition,column in [('observed_score','target'),('first_retained_score','baseline_score'),('last_retained_score','last_score')]:
            valid=scored[scored[column].notna()].copy();valid['band']=band(valid[column].to_numpy())
            for (method,b),g in valid.groupby(['method','band']):
                groups.append(dict(cadence=cadence,definition=definition,method=method,band=int(b),
                    coverage=pmean(g,g.covered),width_fraction=float(g.width_fraction.mean()),n_events=len(g),n_patients=g.id.nunique()))
    pd.DataFrame(summaries).to_csv(args.aggregate_output/'revised_interval_metrics.csv',index=False)
    pd.DataFrame(groups).to_csv(args.aggregate_output/'revised_severity_metrics.csv',index=False)
    pd.DataFrame(selections).to_csv(args.aggregate_output/'development_interval_selection.csv',index=False)

if __name__=='__main__':
    with threadpool_limits(limits=8):main()
