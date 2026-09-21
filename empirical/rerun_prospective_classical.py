"""Matched prospective classical rerun; all patient rows stay protected."""
from pathlib import Path
import argparse, copy, hashlib, json, pickle
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from revise_uncertainty import setup, E, REPO
from prospective_classical import fit_models, predict_patient

REMOVED=('locf','causal_anchor_mean','interpolation')

def main():
    p=argparse.ArgumentParser()
    for name in ['frozen-run','comorbidity-features','output','previous-run']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cadences',type=int,nargs='+',default=[7,14,21,28])
    p.add_argument('--temporal',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.output.resolve().is_relative_to(REPO.resolve()):raise ValueError('Protected output required.')
    specification=dict(cadences=args.cadences,temporal=args.temporal,
        inference='forward Gaussian filtering, fixed issued estimates for recall averaging',
        questionnaire_availability='strictly earlier session, all instruments distinct',
        markov_groups='prior observed questionnaire counts; tertiles fitted in development sessions',
        fitting='complete development sequences only; original empirical iteration and ridge settings',
        source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ['prospective_classical.py','rerun_prospective_classical.py']})
    sp=args.output/'specification.json'
    if sp.exists():assert json.loads(sp.read_text())==specification
    else:sp.write_text(json.dumps(specification,indent=2))
    for cad in args.cadences:
        dest=args.output/f'corrected_comparator_events_{cad}d.csv'
        if dest.exists():print(f'{cad}d complete cache retained',flush=True);continue
        print(f'{cad}d constructing matched inputs',flush=True)
        data,days,anchors,sessions,splits,roles,measurement,*_=setup(args,cad)
        modelpath=args.output/f'models_{cad}d.pkl'
        if modelpath.exists():models,settings,train_ids=pickle.loads(modelpath.read_bytes())
        else:
            print(f'{cad}d fitting development parameters',flush=True)
            models,settings,train_ids=fit_models(data,E)
            modelpath.write_bytes(pickle.dumps((models,settings,train_ids)))
        assert set(train_ids).isdisjoint(pid for pid,role in roles.items() if role in ['test','calibration'])
        if args.temporal:
            rows=pd.read_csv(args.previous_run/f'replayed_events_{cad}d.csv')
            rows=rows[rows.role.eq('test')].copy()
            rows['z_true']=rows.target
            rows['pooling_sd_normalized']=rows.anchor.map(measurement['development_data']['normalized_score_sd_by_instrument'])
            rows['instrument_intercept']=rows.intercept;rows['instrument_loading']=rows.loading
            rows['scale_half_range']=rows.half_range;rows['scale_center']=rows.center
            rows['evaluation_role']=rows.role
        else:rows=pd.read_csv(args.previous_run/f'corrected_comparator_events_{cad}d.csv',low_memory=False)
        rows=rows.drop(columns=[c for c in rows if c.startswith(REMOVED) or
            (c.startswith('ball_teacher') and not c.startswith('ball_teacher_matching_only'))])
        predictions={}
        for ch,channel in enumerate(['Y1','Y2']):
            for short,method in [('lg','s0_direct_lgssm'),('markov','markov_direct_transition')]:
                print(f'{cad}d forward filtering {channel} {short}',flush=True)
                predictions[(ch,method)]={int(pid):predict_patient(data,int(pid),models[channel][short],channel,settings,E)[0]
                    for pid in rows.id.unique()}
        # Audit full pipeline using fixed fitted parameters and changed future inputs.
        audit=[]
        for pid in sorted(rows.loc[rows.evaluation_role.eq('test'),'id'].unique())[:3]:
            n=len(days[int(pid)]);cut=max(1,n//2)
            altered=copy.copy(data)
            altered.anchors=data.anchors.copy();altered.components=data.components.copy();altered.daily=data.daily.copy()
            future=(altered.anchors.id.eq(pid)&altered.anchors.t.ge(cut))
            altered.anchors.loc[future,'value']=99.;altered.anchors.loc[future,'observed']=False
            for col in altered.daily:
                if col.startswith('X'):altered.daily.loc[altered.daily.id.eq(pid)&altered.daily.t.gt(cut),col]=99.
            for col in ['a','recent_treatment','treatment_burden','dt']+[f'B{j}' for j in range(data.config.q)]:
                altered.components.loc[altered.components.id.eq(pid)&altered.components.t.ge(cut),col]=0 if col=='a' else 99.
            # Current B is legitimately observed; empirical B is excluded and fixed zero.
            for ch,channel in enumerate(['Y1','Y2']):
                for short,method in [('lg','s0_direct_lgssm'),('markov','markov_direct_transition')]:
                    changed=predict_patient(altered,int(pid),models[channel][short],channel,settings,E)[0]
                    err=float(np.max(np.abs(changed[:cut+1]-predictions[(ch,method)][int(pid)][:cut+1])))
                    assert err<1e-10,(cad,method,channel,err)
                    audit.append(dict(channel=channel,method=method,max_prefix_change=err))
        sessionmap={int(pid):g.sort_values(['session','day']).session.to_numpy() for pid,g in sessions.groupby('id') if int(pid) in days}
        windows=anchors.set_index(['id','session','anchor'])[['window_start_day','window_end_day']]
        for method in ['s0_direct_lgssm','markov_direct_transition']:
            values=[]
            for r in rows.itertuples():
                w=windows.loc[(r.id,r.session,r.anchor)]
                endpoint=int(np.flatnonzero(sessionmap[r.id]==r.session)[0])
                idx=np.flatnonzero((days[r.id]>=w.window_start_day)&(days[r.id]<=w.window_end_day))
                idx=idx[idx<=endpoint]
                ch=measurement['instruments'][r.anchor]['channel']
                values.append(r.instrument_intercept+r.instrument_loading*predictions[(ch,method)][r.id][idx].mean())
            rows[method]=values
            rows[method+'_native']=rows[method]*rows.scale_half_range+rows.scale_center
            rows[method+'_native_clipped']=np.clip(rows[method+'_native'],0,2*rows.scale_half_range)
            rows[method+'_clipped']=(rows[method+'_native_clipped']-rows.scale_center)/rows.scale_half_range
        assert np.isfinite(rows[['s0_direct_lgssm','markov_direct_transition']]).all().all()
        rows.to_csv(dest,index=False)
        (args.output/f'audit_{cad}d.json').write_text(json.dumps(dict(training_patients=len(train_ids),
            evaluation_rows=len(rows),test_patients=int(rows.loc[rows.evaluation_role.eq('test'),'id'].nunique()),
            training_excludes_calibration_and_test=True,future_perturbation_checks=audit),indent=2))
        for method in ['ball','s0_direct_lgssm','markov_direct_transition']:
            test=rows[rows.evaluation_role.eq('test')]
            error=((test[method]-test.z_true)/test.pooling_sd_normalized)**2
            print(json.dumps(dict(cadence=cad,method=method,rmse=float(np.sqrt(error.groupby(test.id).mean().mean())))),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
