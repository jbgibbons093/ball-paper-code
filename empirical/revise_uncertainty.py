"""Replay frozen BALL ensembles for development-only uncertainty revision.

Patient-level inputs, caches and predictions must stay in the protected output
directory supplied on the command line. This does not retrain the latent model.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import BALL
BALL._install_virtual_package()
E=sys.modules['empirical.fit_empirical']
M=sys.modules['simulations.src.methods.ball_ssm']

def setup(args,cadence):
    frozen=args.frozen_run
    measurement=json.loads((frozen/'measurement_calibration.json').read_text())
    sessions=E._coerce_sessions(pd.read_csv(E.OUT/'sessions.csv'))
    splits,roles,flow=E._primary_empirical_partitions(sessions)
    features=E._load_comorbidity_sessions(args.comorbidity_features)
    anchors=pd.read_csv(E.OUT/f'anchors_sparse_{cadence}d.csv')
    if getattr(args,'temporal',False):
        splits=E._generalization_split_maps(sessions)['forward_2025']
        sessions=sessions[sessions['split'].isin(['train','val'])].copy()
        test_ids=np.asarray(sorted(pid for pid,role in splits.items() if role=='test'))
        np.random.default_rng(20250101).shuffle(test_ids)
        roles={int(pid):('calibration' if i<len(test_ids)//2 else 'test') for i,pid in enumerate(test_ids)}
        cache=args.output/'temporal_measurement_calibration.json'
        if cache.exists():measurement=json.loads(cache.read_text())
        else:
            source=sessions.groupby('id')['split'].first().to_dict()
            train_map={pid:'train' for pid,role in splits.items() if role=='train' and source[pid]=='train'}
            measurement=E._fit_development_measurement_calibration(pd.read_csv(E.OUT/'questionnaire_events_all.csv'),train_map)
            cache.write_text(json.dumps(measurement,indent=2))
    data,days=E._empirical_neural_dataset(anchors,sessions,None,features,measurement,patient_splits=splits)
    cfg=M.SSMConfig(d_model=96,n_heads=4,n_layers=3,ensemble_size=5,
                    use_alpha_slow=False,rdoc_drift_head=False,delta_phi=.9,
                    anchor_weight=10.,student_anchor_weight=2.,max_individuals=data.config.n)
    train,stats=M.build_ssm_batch(data,'train',torch.device('cpu'),cfg.max_individuals,
                                 causal_impute=True,include_rdoc_in_encoder=cfg.encoder_use_rdoc)
    test,_=M.build_ssm_batch(data,'test',torch.device('cpu'),cfg.max_individuals,
                            feature_stats=stats,causal_impute=True,include_rdoc_in_encoder=cfg.encoder_use_rdoc)
    return data,days,anchors,sessions,splits,roles,measurement,cfg,train,test,stats

def load_member(args,cadence,member,data,train,cfg,device):
    label=f'{cadence}d_generalization_forward_2025' if getattr(args,'temporal',False) else f'{cadence}d_primary_ball'
    root=args.frozen_run/'member_checkpoints'/label
    gen,net=M._make_models(data,train,cfg,device,causal=True)
    teacher=torch.load(root/f'teacher_member_{member:02d}.pt',map_location=device,weights_only=True)
    gen.load_state_dict(teacher['generative_state'],strict=True)
    metadata=json.loads((root/f'student_member_{member:02d}.json').read_text())
    seed=int(metadata['member_seed'])+500
    pointer=json.loads((root/f'training_state_student_distilled_seed_{seed}.latest.json').read_text())
    assert pointer['phase']=='complete'
    state=torch.load(root/pointer['checkpoint_file'],map_location=device,weights_only=False)
    net.load_state_dict(state['network_state'],strict=True)
    net.eval();gen.eval()
    return gen,net

def gpu_batch(batch,indices,device):
    part=M._subbatch(batch,torch.as_tensor(indices,dtype=torch.long))
    return dataclasses.replace(part,**{f:getattr(part,f).to(device) for f in M._B_FIELDS})

def encode_chunks(net,batch,device,batch_size=32):
    pieces=[]
    with torch.no_grad():
        for start in range(0,len(batch.ids),batch_size):
            part=gpu_batch(batch,np.arange(start,min(start+batch_size,len(batch.ids))),device)
            h=net.encode(part)
            pieces.append(h.cpu().numpy())
    return np.concatenate(pieces)

def replay(args, prepared=None, verify_frozen=True):
    device=torch.device(args.device)
    torch.set_num_threads(8)
    args.output.mkdir(parents=True,exist_ok=True)
    for cadence in args.cadences:
        destination=args.output/f'replayed_events_{cadence}d.csv'
        if destination.exists():
            print(f'{cadence}d existing replay cache retained',flush=True);continue
        print(f'{cadence}d building frozen inputs',flush=True)
        context=prepared[cadence] if prepared is not None else setup(args,cadence)
        data,days,anchors,sessions,splits,roles,measurement,cfg,train,test,stats=context
        events=anchors[anchors.evaluation_role.eq('withheld_eligible') & anchors.id.isin(data.individuals.id)].copy()
        events=events.sort_values(['id','session','anchor']).reset_index(drop=True)
        session_map={int(pid):g.sort_values(['session','day']).session.to_numpy() for pid,g in sessions.groupby('id') if int(pid) in days}
        patient_location={int(pid):('train',i) for i,pid in enumerate(train.ids)}
        patient_location.update({int(pid):('test',i) for i,pid in enumerate(test.ids)})
        window=[];old_window=[];positions=[]
        for r in events.itertuples():
            sid=session_map[int(r.id)]; t=int(np.flatnonzero(sid==r.session)[0]);positions.append(t)
            old=np.flatnonzero((days[int(r.id)]>=r.window_start_day)&(days[int(r.id)]<=r.window_end_day))
            idx=old[old<=t]
            assert len(idx)>0
            window.append(idx);old_window.append(old)
        member_mean=[];member_variance=[];old_member_mean=[];old_member_variance=[]
        for member in range(5):
            gen,net=load_member(args,cadence,member,data,train,cfg,device)
            h_train=encode_chunks(net,train,device,args.batch_size)
            h_test=encode_chunks(net,test,device,args.batch_size)
            valid=train.valid.numpy().astype(bool)
            h_valid=h_train[valid]
            covariance=cfg.laplace_scale/(cfg.laplace_prior_precision+np.square(h_valid).sum(axis=0))
            bias_variance=cfg.laplace_scale/(cfg.laplace_prior_precision+len(h_valid))
            arrays={}
            for split,h in [('train',h_train),('test',h_test)]:
                with torch.no_grad():
                    tensor=torch.from_numpy(h).to(device)
                    mu=net.mu_head(tensor).cpu().numpy()[...,-2:]
                    var=net.logvar_head(tensor).clamp(-8.,4.).exp().cpu().numpy()[...,-2:]
                arrays[split]=(mu,var,h)
            mm=[];vv=[];oldm=[];oldv=[]
            for r,idx,oidx in zip(events.itertuples(),window,old_window):
                split,i=patient_location[int(r.id)]
                channel=measurement['instruments'][r.anchor]['channel']
                mu,var,h=arrays[split]
                n=len(idx);h_bar=h[i,idx].mean(axis=0)
                mm.append(mu[i,idx,channel].mean())
                # The same final-layer weights and bias are shared by all
                # sessions. Average features before applying their covariance.
                vv.append(var[i,idx,channel].sum()/n**2+(h_bar**2*covariance).sum()+bias_variance)
                oldm.append(mu[i,oidx,channel].mean())
                oldv.append((var[i,oidx,channel]+(h[i,oidx]**2*covariance).sum(axis=-1)+bias_variance).sum()/len(oidx)**2)
            member_mean.append(mm);member_variance.append(vv)
            old_member_mean.append(oldm);old_member_variance.append(oldv)
            # Session arrays are retained for reproducibility; never published.
            np.savez_compressed(args.output/f'session_member_{cadence}d_{member}.npz',
                                train_ids=np.asarray(train.ids),test_ids=np.asarray(test.ids),
                                train_mean=arrays['train'][0],test_mean=arrays['test'][0],
                                train_variance=arrays['train'][1],test_variance=arrays['test'][1],
                                weight_covariance=covariance,bias_variance=bias_variance)
            del gen,net,h_train,h_test,arrays
            torch.cuda.empty_cache()
            print(f'{cadence}d replayed ensemble member {member+1}/5',flush=True)
        member_mean=np.asarray(member_mean); member_variance=np.asarray(member_variance)
        old_member_mean=np.asarray(old_member_mean)
        rows=events[['id','session','day','anchor','value','window_start_day','window_end_day']].copy()
        rows['cadence']=cadence
        rows['role']=rows.id.map(lambda pid:roles.get(int(pid),'development'))
        rows['source_split']=rows.id.map(sessions.groupby('id').split.first())
        rows['latent_mean']=member_mean.mean(axis=0)
        # Members represent shared fitted functions, so disagreement must be
        # computed after each member's recall-window average.
        rows['latent_variance']=member_variance.mean(axis=0)+member_mean.var(axis=0)
        rows['old_window_latent_mean']=old_member_mean.mean(axis=0)
        rows['window_n_sessions']=[len(x) for x in window]
        rows['excluded_later_same_day_sessions']=[len(o)-len(n) for o,n in zip(old_window,window)]
        for key in ['intercept','loading','observation_sd','channel']:
            rows[key]=rows.anchor.map({k:v[key] for k,v in measurement['instruments'].items()})
        rows['center']=rows.anchor.map({k:v['midpoint'] for k,v in measurement['transforms'].items()})
        rows['half_range']=rows.anchor.map({k:v['half_range'] for k,v in measurement['transforms'].items()})
        rows['ball']=rows.intercept+rows.loading*rows.latent_mean
        rows['sd']=np.sqrt(rows.loading**2*rows.latent_variance+rows.observation_sd**2)
        rows['target']=(rows.value-rows.center)/rows.half_range
        rows['old_window_ball']=rows.intercept+rows.loading*rows.old_window_latent_mean
        # Prospective history predictors for conditional interval estimation.
        prior_by={int(pid):g.sort_values(['session','anchor']) for pid,g in anchors[anchors.role.eq('anchor')].groupby('id')}
        histories=[]
        for r in rows.itertuples():
            prior=prior_by[int(r.id)]
            prior=prior[prior.session<r.session]
            same=prior[prior.anchor.eq(r.anchor)]
            same_channel=prior[prior.anchor.map({k:v['channel'] for k,v in measurement['instruments'].items()}).eq(r.channel)]
            histories.append(dict(prior_count=len(prior),same_instrument_prior_count=len(same),
                gap=float(r.day-prior.day.max()) if len(prior) else np.nan,
                last_score=(float(same.value.iloc[-1])-r.center)/r.half_range if len(same) else np.nan,
                baseline_score=(float(same.value.iloc[0])-r.center)/r.half_range if len(same) else np.nan,
                same_channel_prior_count=len(same_channel)))
        rows=pd.concat([rows,pd.DataFrame(histories)],axis=1)
        # Fits made before the September 2026 duplicate-day correction averaged the whole calendar recall window.
        # Later fits average only the sessions through the target. The replay must reproduce one exactly.
        errors={}
        if not verify_frozen:
            # Fresh patient-excluded ensembles have no original prediction
            # artifact to reproduce. Their partition audit is performed by
            # the cross-fitting runner before this replay.
            pass
        elif getattr(args,'temporal',False):
            verify=rows[rows.role.eq('test')].copy()
            scale=verify.anchor.map(measurement['development_data']['normalized_score_sd_by_instrument'])
            frozen=pd.read_csv(args.frozen_run/'empirical_generalization_complete.csv')
            expected=float(frozen[frozen.method.eq('ball')].rmse.iloc[0])
            for definition,column in [('calendar_window','old_window_ball'),('sessions_through_target','ball')]:
                se=((verify[column]-verify.target)/scale)**2
                errors[definition]=abs(float(np.sqrt(se.groupby(verify.id).mean().mean()))-expected)
        else:
            frozen=pd.read_csv(args.frozen_run/'empirical_prediction_rows_complete.csv',
                               usecols=['cadence','id','session','anchor','ball'],low_memory=False)
            verify=rows.merge(frozen[frozen.cadence.eq(cadence)],on=['cadence','id','session','anchor'],suffixes=('','_frozen'))
            errors={'calendar_window':float(np.max(abs(verify.old_window_ball-verify.ball_frozen))),
                    'sessions_through_target':float(np.max(abs(verify.ball-verify.ball_frozen)))}
        matched=min(errors,key=errors.get) if errors else None
        if verify_frozen:
            assert errors[matched]<2e-5,('Frozen checkpoint replay failed',errors)
        rows.to_csv(destination,index=False)
        print(json.dumps({'cadence':cadence,'events':len(rows),'frozen_point_max_error':errors.get(matched),
                          'frozen_window_definition':matched,'frozen_errors_by_definition':errors,
                          'frozen_prediction_comparison_performed':bool(verify_frozen),
                          'later_same_day_window_fraction':float((rows.excluded_later_same_day_sessions>0).mean())}),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--frozen-run',type=Path,required=True)
    p.add_argument('--comorbidity-features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cadences',type=int,nargs='+',default=[7,14,21,28])
    p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--temporal',action='store_true')
    args=p.parse_args()
    if args.output.resolve().is_relative_to(REPO.resolve()):
        raise ValueError('Patient-level output must stay outside the source repository.')
    if args.temporal and args.cadences!=[14]:raise ValueError('Frozen temporal models use the 14-day schedule.')
    replay(args)

if __name__=='__main__':main()
