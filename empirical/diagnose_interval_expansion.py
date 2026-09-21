"""Aggregate diagnosis of endpoint corrections, without discarding patients."""
from pathlib import Path
import argparse
import pickle
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from conditional_intervals import predict,qpatient,pmean
from interval_history import HALF

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for file in sorted(args.run.glob('replayed_events_*d.csv')):
        d=pd.read_csv(file);cad=int(d.cadence.iloc[0])
        for inst,g in d.groupby('anchor'):
            with (args.run/f'interval_models_{cad}d_{inst}.pkl').open('rb') as f:models=pickle.load(f)
            c=g[g.role.eq('calibration')]
            cl,ch=predict(models,c);scores=np.maximum(cl-c.target.to_numpy(),c.target.to_numpy()-ch)
            q=qpatient(c,scores)
            influence=[]
            for pid in c.id.unique():
                mask=c.id.to_numpy()!=pid
                influence.append(qpatient(c[mask],scores[mask]))
            for role,sub in g.groupby('role'):
                lo,hi=predict(models,sub)
                covered=(sub.target>=lo)&(sub.target<=hi)
                a=np.clip(lo-q,-1,1);b=np.clip(hi+q,-1,1)
                rows.append(dict(cadence=cad,instrument=inst,role=role,
                    n_patients=sub.id.nunique(),n_events=len(sub),
                    base_coverage=pmean(sub,covered),
                    base_width_fraction=float((np.clip(hi,-1,1)-np.clip(lo,-1,1)).mean()/2),
                    calibrated_coverage=pmean(sub,(sub.target>=a)&(sub.target<=b)),
                    calibrated_width_fraction=float((b-a).mean()/2),
                    q_normalized=q,q_native_per_endpoint=q*HALF[inst],
                    q_leave_one_patient_out_min=min(influence),q_leave_one_patient_out_max=max(influence),
                    largest_single_patient_reduction_native=(q-min(influence))*HALF[inst]))
    pd.DataFrame(rows).to_csv(args.output/'calibration_expansion_diagnostic.csv',index=False)
    print('Aggregate calibration expansion diagnostics saved.')

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
