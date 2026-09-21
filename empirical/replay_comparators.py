"""Apply the same prospective recall-window endpoint to all forward comparators."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from revise_uncertainty import setup,E

def main():
    p=argparse.ArgumentParser();p.add_argument('--frozen-run',type=Path,required=True)
    p.add_argument('--comorbidity-features',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cadences',type=int,nargs='+',default=[7,14,21,28]);args=p.parse_args()
    old=pd.read_csv(args.frozen_run/'empirical_prediction_rows_complete.csv',low_memory=False)
    for cad in args.cadences:
        dest=args.output/f'corrected_comparator_events_{cad}d.csv'
        if dest.exists():print(f'{cad}d comparison cache retained',flush=True);continue
        data,days,anchors,sessions,splits,roles,measurement,cfg,train,test,stats=setup(args,cad)
        rows=old[old.cadence.eq(cad)].copy().reset_index(drop=True)
        all_ids=sorted(data.individuals.id.astype(int));test_ids=list(map(int,test.ids))
        allmap={pid:i for i,pid in enumerate(all_ids)};testmap={pid:i for i,pid in enumerate(test_ids)}
        session_map={int(pid):g.sort_values(['session','day']).session.to_numpy() for pid,g in sessions.groupby('id') if int(pid) in days}
        windows=anchors.set_index(['id','session','anchor'])[['window_start_day','window_end_day']]
        current=[];prior=[]
        for r in rows.itertuples():
            w=windows.loc[(r.id,r.session,r.anchor)]
            idx=np.flatnonzero((days[r.id]>=w.window_start_day)&(days[r.id]<=w.window_end_day))
            endpoint=int(np.flatnonzero(session_map[r.id]==r.session)[0])
            current.append(idx[idx<=endpoint]);prior.append(idx)
        root=args.frozen_run/'member_checkpoints'
        methods={'ball_direct_causal':(f'{cad}d_direct_causal','direct_causal')}
        if cad==14:
            methods.update({'ball_no_dense_ehr':('14d_no_dense_ehr','student'),
                'ball_anchor_only':('14d_anchor_only','student'),
                'ball_session_balanced':('14d_session_balanced','student'),
                'ball_inherited_dynamics_only':('14d_distillation_decomposition','inherited_dynamics_only'),
                'ball_teacher_matching_only':('14d_distillation_decomposition','teacher_matching_only')})
        arrays={};idmaps={}
        for name,(folder,arm) in methods.items():
            files=list((root/folder).glob(f'{arm}_member_*.npz'))
            assert len(files)==5,(folder,arm,len(files))
            arrays[name]=np.mean([np.stack([z['z_d'],z['z_p']],axis=-1) for z in [np.load(f) for f in sorted(files)]],axis=0)
            idmaps[name]=testmap
        files=list((root/f'{cad}d_exponential_decay_gru').glob('member_*_prediction.npy'))
        assert len(files)==5
        arrays['exponential_decay_gru']=np.mean([np.load(f) for f in sorted(files)],axis=0)
        idmaps['exponential_decay_gru']=allmap
        print(f'{cad}d reproducing frozen Gaussian-process fit',flush=True)
        gps={}
        for ch,name in enumerate(['Y1','Y2']):
            gps[ch],metadata=E._fit_empirical_gp_channel(data,name,days,fixed_auxiliary_weight=.15)
        diagnostics={}
        for name in list(arrays)+['gp_causal_filter']:
            values=[];previous=[]
            for r,new,oldidx in zip(rows.itertuples(),current,prior):
                ch=measurement['instruments'][r.anchor]['channel']
                if name=='gp_causal_filter':arr=gps[ch][r.id]
                else:arr=arrays[name][idmaps[name][r.id],:,ch]
                values.append(r.instrument_intercept+r.instrument_loading*float(arr[new].mean()))
                previous.append(r.instrument_intercept+r.instrument_loading*float(arr[oldidx].mean()))
            delta=float(np.max(abs(np.asarray(previous)-rows[name])))
            assert delta<2e-5,(name,delta)
            diagnostics[name]=delta
            rows[name]=values
        newball=pd.read_csv(args.output/f'replayed_events_{cad}d.csv').set_index(['id','session','anchor'])
        keys=pd.MultiIndex.from_frame(rows[['id','session','anchor']])
        rows['ball']=newball.loc[keys,'ball'].to_numpy();rows['ball_sd']=newball.loc[keys,'sd'].to_numpy()
        if cad==14:
            frozen_cad=old[old.cadence.eq(cad)]
            np.testing.assert_allclose(frozen_cad.ball_full_decomposition,frozen_cad.ball)
            rows['ball_full_decomposition']=rows.ball
        for name in list(arrays)+['gp_causal_filter','ball']:
            rows[f'{name}_native']=rows[name]*rows.scale_half_range+rows.scale_center
            rows[f'{name}_native_clipped']=np.clip(rows[f'{name}_native'],0,2*rows.scale_half_range)
            rows[f'{name}_clipped']=(rows[f'{name}_native_clipped']-rows.scale_center)/rows.scale_half_range
        if cad==14:
            for suffix in ['_native','_native_clipped','_clipped']:
                rows['ball_full_decomposition'+suffix]=rows['ball'+suffix]
        rows.to_csv(dest,index=False)
        print(json.dumps({'cadence':cad,'frozen_old_window_reproduction_max_errors':diagnostics}),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=8):main()
