"""Prospective replay of extra measurement decisions with causal feedback.

Only genuine withheld questionnaire events are query opportunities. Every score
is predicted before its value can be revealed, and a query enters the encoder
at the next recorded session. This evaluates an observed-opportunity policy,
not unobserved questionnaires or actual clinical resource utilization.
"""
from pathlib import Path
import argparse
import dataclasses
import json
import sys
import numpy as np
import pandas as pd
import torch

from revise_uncertainty import setup,load_member,M
from uncertainty_math import recall_moments,causal_window_positions,expected_fractional_variance_reduction

POLICIES={'fixed_schedule':None,'periodic_7d':0.,
          'latent_gain_25pct':.25,'latent_gain_50pct':.5,'latent_gain_75pct':.75}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--frozen-run',type=Path,required=True)
    p.add_argument('--comorbidity-features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cadence',type=int,default=14)
    p.add_argument('--device',default='cuda')
    p.add_argument('--policies',nargs='+',choices=list(POLICIES),default=list(POLICIES))
    # Random requests matched on count: each eligible opportunity is requested with this probability,
    # ignoring BALL's uncertainty. Replicates are written to a subfolder so policy summaries skip them.
    p.add_argument('--random-probability',type=float)
    p.add_argument('--random-replicates',type=int,default=0)
    p.add_argument('--random-first-replicate',type=int,default=0)
    p.add_argument('--random-seed',type=int,default=20261017)
    p.add_argument('--random-folder',default='random_matched')
    p.add_argument('--member-archives',type=Path,help='folder holding session_member archives, if not --output')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError('Patient-level outputs must be outside the source repository.')
    torch.set_num_threads(8);device=torch.device(args.device)
    data,days,anchors,sessions,splits,roles,measurement,cfg,train,test,stats=setup(args,args.cadence)
    if data.metadata.get('encoder_feature_scaling')=='none':
        assert np.all(stats[0]==0) and np.all(stats[1]==1)
    B,T=test.valid.shape;ids=list(map(int,test.ids));idrow={pid:i for i,pid in enumerate(ids)}
    instruments=list(data.config.anchor_instrument_names)
    instidx={a:i for i,a in enumerate(instruments)}
    channels=np.array([measurement['instruments'][a]['channel'] for a in instruments])
    models=[];covariances=[];biases=[]
    for member in range(5):
        _,net=load_member(args,args.cadence,member,data,train,cfg,device)
        models.append(net)
        archive=np.load((args.member_archives or args.output)/f'session_member_{args.cadence}d_{member}.npz')
        covariances.append(archive['weight_covariance']);biases.append(float(archive['bias_variance']))
    session_map={int(pid):g.sort_values(['session','day']).session.to_numpy() for pid,g in sessions.groupby('id') if int(pid) in idrow}
    position={(pid,int(s)):i for pid,values in session_map.items() for i,s in enumerate(values)}
    events=anchors[anchors.id.isin(ids)].copy()
    events['position']=[position[(int(r.id),int(r.session))] for r in events.itertuples()]
    targets=events[events.evaluation_role.eq('withheld_eligible')]
    by_pos={int(k):g for k,g in targets.groupby('position')}
    retained_by_pos={int(k):g for k,g in events[events.role.eq('anchor')].groupby('position')}
    specification={'minimum_days_since_same_instrument':7,'variance_reduction_thresholds':POLICIES,
       'decision_score':'loading squared times recall-mean latent variance divided by predictive questionnaire variance',
       'interpretation':'Gaussian approximation to expected fractional latent-variance reduction from one questionnaire',
       'feedback':'queried value becomes visible at next recorded session only',
       'opportunities':'genuine withheld eligible questionnaire events',
       'evaluation':'all common withheld events, scored before any reveal; queried events remain in evaluation',
       'model_members':5,'no_test_outcome_used_for_decision':True,
       'baseline':'original retained questionnaires are always available on their original schedule'}
    spec_path=args.output/f'adaptive_specification_{args.cadence}d.json'
    if args.random_probability is None or not spec_path.exists():
        spec_path.write_text(json.dumps(specification,indent=2))

    def replay(dest,policy,decide,extra):
        batch=dataclasses.replace(test,**{f:getattr(test,f).clone().to(device) for f in M._B_FIELDS})
        mu_history=np.zeros((5,B,T,2),dtype=np.float32)
        var_history=np.zeros_like(mu_history)
        h_history=np.zeros((5,B,T,cfg.d_model),dtype=np.float32)
        last_day=np.full((B,3),-np.inf);last_score=np.full((B,3),np.nan)
        baseline_score=np.full((B,3),np.nan);prior_count=np.zeros((B,3),dtype=int)
        records=[]
        for t in range(T):
            active=np.flatnonzero(test.valid[:,t].numpy())
            if not len(active):continue
            idx=torch.as_tensor(active,device=device,dtype=torch.long)
            part=dataclasses.replace(batch,**{f:getattr(batch,f)[idx,:t+1] for f in M._B_FIELDS})
            with torch.no_grad():
                for k,net in enumerate(models):
                    h=net.encode(part)[:,-1,:]
                    h_history[k,active,t]=h.cpu().numpy()
                    mu_history[k,active,t]=net.mu_head(h)[:,-2:].cpu().numpy()
                    var_history[k,active,t]=net.logvar_head(h).clamp(-8.,4.).exp()[:,-2:].cpu().numpy()
            newly_selected=[]
            for r in by_pos.get(t,pd.DataFrame()).itertuples():
                i=idrow[int(r.id)];j=instidx[r.anchor];ch=int(channels[j])
                window=causal_window_positions(days[int(r.id)],r.window_start_day,r.window_end_day,t)
                assert len(window)>0
                # np.take preserves the member, time, feature axis order.
                latent_mean,latent_var=recall_moments(
                    np.take(mu_history[:,i,:,ch],window,axis=1),
                    np.take(var_history[:,i,:,ch],window,axis=1),
                    np.take(h_history[:,i],window,axis=1),covariances,biases)
                pars=measurement['instruments'][r.anchor]
                transform=measurement['transforms'][r.anchor]
                signal_variance=pars['loading']**2*latent_var
                observation_variance=pars['observation_sd']**2
                gain=expected_fractional_variance_reduction(latent_var,pars['loading'],pars['observation_sd'])
                eligible=(r.day-last_day[i,j])>=7
                query=decide(gain,eligible)
                # No r.value appears above this line: the decision is made first.
                target=(r.value-transform['midpoint'])/transform['half_range']
                prediction=pars['intercept']+pars['loading']*latent_mean
                records.append(dict(cadence=args.cadence,policy=policy,id=int(r.id),session=int(r.session),day=int(r.day),
                    anchor=r.anchor,role=roles[int(r.id)],target=target,ball=prediction,sd=np.sqrt(signal_variance+observation_variance),
                    gain=gain,query=query,eligible_after_cooldown=eligible,prior_count=int(prior_count[i].sum()),
                    same_instrument_prior_count=int(prior_count[i,j]),same_channel_prior_count=int(prior_count[i,channels==ch].sum()),
                    gap=float(r.day-np.max(last_day[i])) if np.isfinite(last_day[i]).any() else np.nan,
                    last_score=last_score[i,j],baseline_score=baseline_score[i,j],window_n_sessions=len(window),**extra))
                if query:newly_selected.append((r,i,j,target))
            # Reveal all selected outcomes together after every prediction at
            # this session. Shifted input prevents same-session cross-instrument leakage.
            updates=[]
            for r in retained_by_pos.get(t,pd.DataFrame()).itertuples():
                i=idrow[int(r.id)];j=instidx[r.anchor];transform=measurement['transforms'][r.anchor]
                updates.append((r,i,j,(r.value-transform['midpoint'])/transform['half_range']))
            updates.extend(newly_selected)
            for r,i,j,value in updates:
                last_day[i,j]=r.day;last_score[i,j]=value;prior_count[i,j]+=1
                if not np.isfinite(baseline_score[i,j]):baseline_score[i,j]=value
                if t+1<T:
                    batch.enc_features[i,t+1,-6+j]=(value-float(stats[0][-6+j]))/float(stats[1][-6+j])
                    batch.enc_features[i,t+1,-3+j]=(1.-float(stats[0][-3+j]))/float(stats[1][-3+j])
            if t%20==0:print(f'{args.cadence}d {policy}: session position {t+1}/{T}',flush=True)
        result=pd.DataFrame(records)
        assert not result.duplicated(['id','session','anchor']).any()
        assert len(result)==len(targets)
        result.to_csv(dest,index=False)
        print(f'{args.cadence}d {policy}: complete, {len(result)} predictions, {int(result["query"].sum())} extra measurements',flush=True)

    for policy in args.policies if args.random_probability is None else []:
        dest=args.output/f'adaptive_events_{args.cadence}d_{policy}.csv'
        if dest.exists():print(f'{policy} existing replay retained',flush=True);continue
        threshold=POLICIES[policy]
        replay(dest,policy,lambda gain,eligible,threshold=threshold:threshold is not None and eligible and gain>=threshold,{})
    if args.random_probability is not None:
        folder=args.output/args.random_folder;folder.mkdir(exist_ok=True)
        for rep in range(args.random_first_replicate,args.random_first_replicate+args.random_replicates):
            dest=folder/f'adaptive_events_{args.cadence}d_random_matched_r{rep:03d}.csv'
            if dest.exists():print(f'random replicate {rep} existing replay retained',flush=True);continue
            rng=np.random.default_rng([args.random_seed,args.cadence,rep])
            replay(dest,'random_matched',lambda gain,eligible,rng=rng:bool(eligible and rng.random()<args.random_probability),
                   dict(replicate=rep,request_probability=args.random_probability))

if __name__=='__main__':main()
