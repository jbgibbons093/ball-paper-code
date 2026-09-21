"""Causal inference helpers for the classical simulation comparators."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'empirical'))
from prospective_classical import forward_filter, update_gaussian, prior_counts


def history_strata(data, n_strata, fit_ids):
    counts={int(pid):prior_counts(data.anchors[data.anchors.id.eq(pid)],len(g))
        for pid,g in data.components.groupby('id')}
    edges=np.quantile(np.concatenate([v for pid,v in counts.items() if pid in fit_ids]),
        np.arange(1,n_strata)/n_strata) if n_strata>1 else np.empty(0)
    return {pid:np.searchsorted(edges,v,side='right') for pid,v in counts.items()}


def event_strata(frame, strata):
    return pd.Series([int(np.asarray(strata[int(r.id)])[int(r.t)])
        if np.ndim(strata[int(r.id)]) else int(strata[int(r.id)])
        for r in frame.itertuples()],index=frame.index)


def filter_person(data,pid,theta,readout,args,*,anchor_name,persistence,
                  innovation_sd,stratum,n_strata,helpers):
    F=helpers
    comp,anchors,b,daily,observed,sd,action,recent,burden=F.person_arrays(data,pid,readout,anchor_name=anchor_name)
    n=len(comp)
    subtype=int(comp.subtype.iloc[0]) if 'subtype' in comp else 0
    features=F.transition_features(b,action,recent,burden,data.config.n_treatment_types,
        basis=getattr(args,'s0_basis','linear'),subtype=subtype,n_subtypes=data.config.n_subtypes)
    groups=np.broadcast_to(np.asarray(stratum,int),(n,))[1:]
    drift=features@theta[:features.shape[1]]
    if n_strata:drift+=theta[features.shape[1]+groups]
    rho=np.broadcast_to(np.asarray(persistence,float),(n,))[1:]
    gaps=comp.dt.to_numpy(float)[:-1] if 'dt' in comp else np.ones(n-1)
    persistence_steps,scale=F._classical_gap_factors(rho,gaps)
    variance=innovation_sd**2*np.maximum((1-rho**(2*gaps))/np.maximum(1-rho**2,1e-8),0.)
    use_irt=str(getattr(data.config,'anchor_observation','gaussian')).lower()=='irt'
    def assimilate(mean,cov,r):
        # Only this now-available questionnaire enters the update. Iterated
        # Fisher scoring relinearizes its likelihood against the same prior.
        end=min(int(r.t),int(r.window_end),len(mean)-1)
        idx=np.arange(max(0,int(r.window_start)),end+1)
        if not use_irt:
            constraint=F.anchor_constraint(r,len(mean),data,args,None)
            if constraint is None:return mean,cov
            coeff,target,obs_sd=constraint
            return update_gaussian(mean,cov,np.asarray([i for i,w in coeff]),
                np.asarray([w for i,w in coeff]),target,obs_sd**2)
        state=F.init_irt_state(data,args);state['pid']=pid
        point=float(mean[idx].mean())
        updated,cupdated=mean,cov
        for _ in range(4):
            state['zbar'][(pid,anchor_name,end)]=(point-state['loc'])/state['scale']
            constraint=F.anchor_constraint(r,len(mean),data,args,state)
            if constraint is None:return mean,cov
            coeff,target,obs_sd=constraint
            updated,cupdated=update_gaussian(mean,cov,np.asarray([i for i,w in coeff]),
                np.asarray([w for i,w in coeff]),target,obs_sd**2)
            point=float(updated[idx].mean())
        return updated,cupdated
    mu,var=forward_filter(drift*scale,persistence_steps,variance,daily,observed,
        sd**2,anchors,args.prior_sd**2,observation_update=assimilate)
    # The trailing covariance fields are used only in development fitting;
    # forward predictions never enter the parameter-estimation routine.
    return mu,b,action,recent,burden,np.r_[gaps,1.],features,var,np.zeros(n-1)
