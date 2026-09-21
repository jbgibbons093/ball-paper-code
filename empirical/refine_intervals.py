"""Development-selected refinement of BALL questionnaire prediction intervals.

The stored specification separates auxiliary interval fitting, development
calibration for candidate selection, and development validation by patient.
Reserved calibration/test patients never choose a candidate. The original BALL
point estimates are retained. These intervals concern observed questionnaires,
not uncertainty about a directly observed latent state.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from conditional_intervals import FEATURES,GRID,weights,pmean,qpatient,interval_score
from interval_history import EXTRA_FEATURES,HALF,add_history

SEED=20260918
LOWER_QUANTILES=[.005,.015,.025,.035,.045]
QUANTILES=LOWER_QUANTILES+[round(q+.95,6) for q in LOWER_QUANTILES]
VARIANTS={
    'infit_central':{'training':'infit','history':False,'tail_mode':'central'},
    'crossfit_central':{'training':'crossfit','history':False,'tail_mode':'central'},
    'crossfit_history_central':{'training':'crossfit','history':True,'tail_mode':'central'},
    'crossfit_history_shortest':{'training':'crossfit','history':True,'tail_mode':'shortest'},
}

def specification():
    return {'version':'interval_refinement_v1','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'seed':SEED,'nominal_coverage':.95,'base_features':FEATURES,'extra_history_features':EXTRA_FEATURES,
        'grid':GRID,'variants':VARIANTS,'quantiles':QUANTILES,
        'shortest_interval':'choose the narrowest of five estimated 95-percent quantile intervals for each history, then calibrate its endpoints',
        'quantile_rearrangement':'sort predicted quantiles to remove crossing; clip base bounds to legal questionnaire range',
        'development_fit':'80 percent of source-training patients, deterministic patient split',
        'development_calibration':'remaining 20 percent of source-training patients',
        'development_validation':'source-validation clinics; common patient-excluded BALL predictions for all candidates',
        'selection':'minimum patient-balanced interval score after development calibration; ties follow listed variant then grid order',
        'refit':'selected specification on all development events, using the stated infit or crossfit predictions',
        'reserved_calibration':'only final conformal endpoint correction',
        'reserved_test':'one evaluation after candidate selection, never model selection',
        'primary_point_predictions':'unchanged frozen BALL ensemble',
        'crossfit':'two patient-exclusive folds stratified by development clinic; five ensemble members, original 120/120 epochs',
        'reporting':'sensitivity analysis; preserve prior outputs and do not add new sensitivity discussion'}

def patient_selection_roles(d):
    train_ids=np.asarray(sorted(d.loc[d.source_split.eq('train'),'id'].unique()),dtype=int)
    rng=np.random.default_rng(SEED);rng.shuffle(train_ids)
    ncal=max(1,int(np.ceil(.2*len(train_ids))))
    mapping={int(pid):('calibration' if i<ncal else 'fit') for i,pid in enumerate(train_ids)}
    mapping.update({int(pid):'validation' for pid in d.loc[d.source_split.eq('val'),'id'].unique()})
    return mapping

def fit_bank(g,params,features,quantiles,cache):
    digest=hashlib.sha256(pd.util.hash_pandas_object(g[['id','session','anchor','target']+features],index=False).values.tobytes()).hexdigest()
    fingerprint={'data_hash':digest,'features':features,'params':params,'quantiles':quantiles}
    if cache.exists():
        with cache.open('rb') as f:bundle=pickle.load(f)
        if bundle['fingerprint']!=fingerprint:raise ValueError(f'Incompatible cached interval fit: {cache.name}')
        return bundle
    sample_weight=weights(g)*len(g)/g.id.nunique()
    models=[]
    for q in quantiles:
        model=HistGradientBoostingRegressor(loss='quantile',quantile=q,learning_rate=.05,
            l2_regularization=1.,early_stopping=False,random_state=SEED,**params)
        model.fit(g[features],g.target-g.ball,sample_weight=sample_weight)
        models.append(model)
    bundle={'models':models,'features':features,'quantiles':quantiles,'fingerprint':fingerprint}
    with cache.open('wb') as f:pickle.dump(bundle,f)
    return bundle

def predict(bundle,g,tail_mode=None):
    mode=tail_mode or bundle['tail_mode']
    q=np.asarray(bundle['quantiles'])
    values=np.column_stack([m.predict(g[bundle['features']]) for m in bundle['models']])+g.ball.to_numpy()[:,None]
    values=np.clip(np.sort(values,axis=1),-1,1)
    if mode=='central':
        return values[:,np.flatnonzero(np.isclose(q,.025))[0]],values[:,np.flatnonzero(np.isclose(q,.975))[0]]
    lowers=values[:,:5];uppers=values[:,5:]
    best=np.argmin(uppers-lowers,axis=1);rows=np.arange(len(g))
    return lowers[rows,best],uppers[rows,best]

def calibrate(bundle,c,t,tail_mode=None):
    cl,ch=predict(bundle,c,tail_mode);tl,th=predict(bundle,t,tail_mode)
    correction=qpatient(c,np.maximum(cl-c.target.to_numpy(),c.target.to_numpy()-ch))
    lo=np.clip(tl-correction,-1,1);hi=np.clip(th+correction,-1,1)
    # Negative conformal corrections may produce empty intervals. Preserve
    # their meaning instead of swapping endpoints or inflating them silently.
    return lo,hi,correction

def metric(g,lo,hi):
    covered=(g.target.to_numpy()>=lo)&(g.target.to_numpy()<=hi)&(lo<=hi)
    return {'coverage':pmean(g,covered),'width_fraction':float(np.maximum(hi-lo,0).mean()/2),
        'patient_balanced_width_fraction':pmean(g,np.maximum(hi-lo,0)/2),
        'interval_score':pmean(g,interval_score(g.target.to_numpy(),lo,hi)),
        'n_events':len(g),'n_patients':int(g.id.nunique()),'empty_fraction':float((lo>hi).mean())}

def fit_cadence(args,cadence):
    source=pd.read_csv(args.previous_run/f'replayed_events_{cadence}d.csv')
    source=source[source.role.eq('development')].copy()
    oof=pd.read_csv(args.crossfit_run/f'oof_events_{cadence}d.csv')
    keys=['id','session','anchor']
    match=source[keys+['target']].merge(oof[keys+['target']],on=keys,suffixes=('_old','_oof'),validate='one_to_one')
    assert len(match)==len(source)==len(oof)
    np.testing.assert_allclose(match.target_old,match.target_oof)
    retained=pd.read_csv(args.anchors/f'anchors_sparse_{cadence}d.csv')
    retained=retained[retained.role.eq('anchor')]
    source=add_history(source,retained);oof=add_history(oof,retained)
    roles=patient_selection_roles(source)
    a_ids={k for k,v in roles.items() if v=='fit'};c_ids={k for k,v in roles.items() if v=='calibration'}
    v_ids={k for k,v in roles.items() if v=='validation'}
    assert not(a_ids&c_ids or a_ids&v_ids or c_ids&v_ids)
    (args.output/f'interval_development_roles_{cadence}d.json').write_text(json.dumps(roles,sort_keys=True))
    choices={};scores=[]
    for inst in sorted(oof.anchor.unique()):
        oo=oof[oof.anchor.eq(inst)];old=source[source.anchor.eq(inst)]
        cal=oo[oo.id.isin(c_ids)];val=oo[oo.id.isin(v_ids)]
        candidates=[]
        for name,spec in VARIANTS.items():
            base=old if spec['training']=='infit' else oo
            fit=base[base.id.isin(a_ids)]
            features=FEATURES+(EXTRA_FEATURES if spec['history'] else [])
            quantiles=QUANTILES if spec['history'] else [.025,.975]
            # Rich central and shortest variants share the identical quantile bank.
            bank_name=f"{spec['training']}_{'history' if spec['history'] else 'base'}"
            for number,params in enumerate(GRID):
                cache=args.output/f'dev_bank_{cadence}d_{inst}_{bank_name}_{number}.pkl'
                bundle=fit_bank(fit,params,features,quantiles,cache)
                lo,hi,q=calibrate(bundle,cal,val,spec['tail_mode'])
                row={'cadence':cadence,'instrument':inst,'variant':name,'grid':number,
                    'development_correction':q,**metric(val,lo,hi),**params}
                scores.append(row);candidates.append(row)
            print(f'{cadence}d {inst} {name}: development candidates evaluated',flush=True)
        choice=min(candidates,key=lambda x:x['interval_score'])
        choices[inst]={'variant':choice['variant'],'grid':choice['grid'],
            'development_interval_score':choice['interval_score']}
        # Refit every variant's development-selected complexity for a complete
        # sensitivity comparison, with one overall development-selected result.
        for name,spec in VARIANTS.items():
            pick=min([x for x in candidates if x['variant']==name],key=lambda x:x['interval_score'])
            base=old if spec['training']=='infit' else oo
            features=FEATURES+(EXTRA_FEATURES if spec['history'] else [])
            quantiles=QUANTILES if spec['history'] else [.025,.975]
            bank_name=f"{spec['training']}_{'history' if spec['history'] else 'base'}"
            bundle=fit_bank(base,GRID[pick['grid']],features,quantiles,
                args.output/f'final_bank_{cadence}d_{inst}_{bank_name}_{pick["grid"]}.pkl')
            bundle={**bundle,'tail_mode':spec['tail_mode'],'variant':name,'grid':pick['grid']}
            with (args.output/f'refined_models_{cadence}d_{inst}_{name}.pkl').open('wb') as f:pickle.dump(bundle,f)
            if name==choice['variant']:
                with (args.output/f'interval_models_{cadence}d_{inst}.pkl').open('wb') as f:pickle.dump(bundle,f)
    (args.output/f'interval_choices_{cadence}d.json').write_text(json.dumps(choices,indent=2))
    pd.DataFrame(scores).to_csv(args.aggregate_output/f'development_refinement_selection_{cadence}d.csv',index=False)
    print(f'{cadence}d interval specification selected and frozen without reserved outcomes',flush=True)

def evaluate_cadence(args,cadence):
    d=pd.read_csv(args.previous_run/f'replayed_events_{cadence}d.csv')
    d=d[d.role.isin(['calibration','test'])].copy()
    retained=pd.read_csv(args.anchors/f'anchors_sparse_{cadence}d.csv')
    retained=retained[retained.role.eq('anchor')]
    d=add_history(d,retained)
    choices=json.loads((args.output/f'interval_choices_{cadence}d.json').read_text())
    parts=[];summaries=[];corrections=[]
    for inst,g in d.groupby('anchor'):
        c=g[g.role.eq('calibration')];t=g[g.role.eq('test')]
        for name in VARIANTS:
            with (args.output/f'refined_models_{cadence}d_{inst}_{name}.pkl').open('rb') as f:bundle=pickle.load(f)
            lo,hi,q=calibrate(bundle,c,t)
            row={'cadence':cadence,'instrument':inst,'method':name,'selected':name==choices[inst]['variant'],**metric(t,lo,hi)}
            summaries.append(row)
            out=t[['cadence','id','session','anchor','target','ball','last_score','baseline_score','day']].copy()
            out['method']=name;out['lower']=lo;out['upper']=hi
            out['covered']=(t.target.to_numpy()>=lo)&(t.target.to_numpy()<=hi)&(lo<=hi)
            out['width_fraction']=np.maximum(hi-lo,0)/2;parts.append(out)
            cl,ch=predict(bundle,c);tl,th=predict(bundle,t)
            corrections.append({'cadence':cadence,'instrument':inst,'variant':name,'q_normalized':q,
                'q_native_per_endpoint':q*HALF[inst],'n_calibration_patients':c.id.nunique(),
                'calibration_base_coverage':metric(c,cl,ch)['coverage'],
                'test_base_width_fraction':float((th-tl).mean()/2)})
            if name==choices[inst]['variant']:
                chosen=out.copy();chosen['method']='conditional_quantiles';parts.append(chosen)
                summaries.append({**row,'method':'conditional_quantiles'})
                (args.output/f'interval_corrections_{cadence}d_{inst}.json').write_text(json.dumps(
                    {'variant':name,'selected_grid':bundle['grid'],'global':q,'n_calibration_patients':int(c.id.nunique())},indent=2))
    pd.concat(parts,ignore_index=True).to_csv(args.output/f'revised_intervals_{cadence}d.csv',index=False)
    pd.DataFrame(summaries).to_csv(args.aggregate_output/f'refinement_metrics_{cadence}d.csv',index=False)
    pd.DataFrame(corrections).to_csv(args.aggregate_output/f'refinement_corrections_{cadence}d.csv',index=False)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--previous-run',type=Path,required=True)
    p.add_argument('--crossfit-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--aggregate-output',type=Path,required=True)
    p.add_argument('--anchors',type=Path,default=Path(__file__).resolve().parent/'derived')
    p.add_argument('--cadences',type=int,nargs='+',default=[7,14,21,28])
    p.add_argument('--stage',choices=['specification','fit','evaluate','all'],default='all')
    args=p.parse_args()
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError('Patient-level and fitted-model outputs must stay outside the source repository.')
    args.output.mkdir(parents=True,exist_ok=True);args.aggregate_output.mkdir(parents=True,exist_ok=True)
    spec=args.output/'refinement_specification.json'
    if not spec.exists():spec.write_text(json.dumps(specification(),indent=2))
    if args.stage=='specification':return
    for cadence in args.cadences:
        if args.stage in ['fit','all'] and not (args.output/f'interval_choices_{cadence}d.json').exists():
            fit_cadence(args,cadence)
        if args.stage in ['evaluate','all']:evaluate_cadence(args,cadence)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
