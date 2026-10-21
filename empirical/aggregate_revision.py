"""Synchronize aggregate tables from corrected prospective prediction rows."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from revise_uncertainty import E

SCHEDULE_COMPARISONS = 4   # the four prespecified questionnaire schedules
ADJUSTED_ALPHA = .05/SCHEDULE_COMPARISONS   # Bonferroni adjustment across schedules

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--frozen-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--interval-run',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    interval_run=args.interval_run or args.run
    frozen_overall=pd.read_csv(args.frozen_run/'empirical_recovery_overall_complete.csv')
    overall=[];pairwise=[];instrument_rows=[];conditional=[];context=[];classification=[]
    for cad in [7,14,21,28]:
        path=args.run/f'corrected_comparator_events_{cad}d.csv'
        if not path.exists():raise RuntimeError(f'Missing synchronized {cad}d comparison replay.')
        ev=pd.read_csv(path,low_memory=False);ev=ev[ev.evaluation_role.eq('test')].copy()
        if cad==14:
            ev['ball_full_decomposition']=ev.ball
            for suffix in ['_native','_native_clipped','_clipped']:ev['ball_full_decomposition'+suffix]=ev['ball'+suffix]
        methods=frozen_overall[frozen_overall.cadence.eq(cad)].method.tolist()
        methods=[m for m in methods if not m.startswith(('locf','causal_anchor_mean')) and m not in ['interpolation','ball_teacher']]
        errors={}
        for method in methods:
            se=((ev[method]-ev.z_true)/ev.pooling_sd_normalized)**2
            pe=se.groupby(ev.id).mean();errors[method]=pe
            overall.append(dict(cadence=cad,method=method,rmse=float(np.sqrt(pe.mean())),n=len(ev),n_patients=ev.id.nunique()))
            for inst,g in ev.groupby('anchor'):
                native=((g[method]-g.z_true)*g.scale_half_range)**2
                standardized=((g[method]-g.z_true)/g.pooling_sd_normalized)**2
                instrument_rows.append(dict(cadence=cad,method=method,instrument=inst,
                    rmse=float(np.sqrt(native.groupby(g.id).mean().mean())),
                    rmse_development_sd_units=float(np.sqrt(standardized.groupby(g.id).mean().mean())),n=len(g),n_patients=g.id.nunique()))
            for kind,column in [('cold_start','cold_start'),('prior_anchor_count','prior_anchor_band'),('severity','severity_band'),
                    ('treatment_week','treatment_week'),('sex','sex'),('age','age_band'),('clinic','clinic'),('gap','gap_bin')]:
                for label,g in ev.groupby(column):
                    context.append(dict(cadence=cad,method=method,stratum_type=kind,stratum=str(label),
                        rmse=float(np.sqrt(se.loc[g.index].groupby(g.id).mean().mean())),n=len(g),n_patients=g.id.nunique()))
        patients=errors['ball'].index
        draws=np.random.default_rng(101).integers(0,len(patients),size=(2000,len(patients)))
        ball=errors['ball'].to_numpy()
        for method in methods:
            if method=='ball':continue
            other=errors[method].reindex(patients).to_numpy()
            delta=np.sqrt(ball[draws].mean(axis=1))-np.sqrt(other[draws].mean(axis=1))
            lo,hi=np.quantile(delta,[.025,.975])
            adj_lo,adj_hi=np.quantile(delta,[ADJUSTED_ALPHA/2,1-ADJUSTED_ALPHA/2])
            pairwise.append(dict(cadence=cad,reference='ball',comparator=method,
                rmse_difference=float(np.sqrt(ball.mean())-np.sqrt(other.mean())),ci_lo=lo,ci_hi=hi,
                ci_lo_bonferroni_schedules=adj_lo,ci_hi_bonferroni_schedules=adj_hi,
                n_patients=len(patients),bootstrap_replicates=2000))
        classification.extend(E._workload_classification_rows(ev,cad,[m for m in ['ball','ball_direct_causal','ball_no_dense_ehr','ball_anchor_only','gp_causal_filter','exponential_decay_gru','causal_anchor_mean'] if m in methods]))
        intervals=pd.read_csv(interval_run/f'revised_intervals_{cad}d.csv')
        intervals=intervals[intervals.method.eq('conditional_quantiles')]
        ev=ev.merge(intervals[['id','session','anchor','lower','upper','covered','width_fraction']],on=['id','session','anchor'],validate='one_to_one')
        for kind,column in [('instrument','anchor'),('cold_start','cold_start'),('prior_anchor_count','prior_anchor_band'),
                ('severity','severity_band'),('treatment_week','treatment_week'),('sex','sex'),('age','age_band'),('clinic','clinic'),('gap','gap_bin')]:
            for label,g in ev.groupby(column):
                conditional.append(dict(cadence=cad,stratum_type=kind,stratum=str(label),
                    coverage=float(g.groupby('id').covered.mean().mean()),
                    mean_width_fraction_of_legal_range=float(g.width_fraction.mean()),n=len(g),n_patients=g.id.nunique()))
    for name,rows in [('empirical_recovery_overall',overall),('empirical_pairwise_bootstrap',pairwise),
            ('empirical_recovery_by_instrument',instrument_rows),('empirical_recovery_by_context',context),
            ('empirical_calibration_conditional',conditional),('empirical_workload_classification',classification)]:
        pd.DataFrame(rows).to_csv(args.output/(name+'_complete.csv'),index=False)
    print(pd.DataFrame(overall).pivot(index='cadence',columns='method',values='rmse').round(4).to_string())

if __name__=='__main__':main()
