"""Aggregate interval refinement, matched comparisons, and display inputs."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from conditional_intervals import pmean
from interval_history import HALF

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--previous-run',type=Path,required=True)
    p.add_argument('--frozen-run',type=Path,required=True)
    p.add_argument('--classical-run',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    cal=pd.concat([pd.read_csv(out/f'refinement_metrics_{cad}d.csv') for cad in [7,14,21,28]],ignore_index=True)
    cal.to_csv(out/'revised_interval_metrics.csv',index=False)
    primary=cal[cal.method.eq('conditional_quantiles')].copy()
    primary['mean_width_fraction_of_legal_range']=primary.width_fraction
    primary.to_csv(out/'empirical_calibration_complete.csv',index=False)
    pair=pd.read_csv(out/'empirical_pairwise_bootstrap_complete.csv')
    rows=[]
    for cad,g in pair.groupby('cadence'):
        row={'cadence':cad}
        for name,alias in [('ball_direct_causal','direct_causal')]:
            r=g[g.comparator.eq(name)].iloc[0]
            row['ball_minus_'+alias+'_rmse']=r.rmse_difference
            row[alias+'_ci_lo']=r.ci_lo;row[alias+'_ci_hi']=r.ci_hi
            row[alias+'_ci_lo_bonferroni']=r.ci_lo_bonferroni_schedules;row[alias+'_ci_hi_bonferroni']=r.ci_hi_bonferroni_schedules
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'empirical_bootstrap_diff_complete.csv',index=False)
    temporal=pd.read_csv(args.frozen_run/'empirical_generalization_complete.csv')
    temporal=temporal[~temporal.method.str.startswith(('locf','causal_anchor_mean'))&~temporal.method.eq('interpolation')].copy()
    d=pd.read_csv(args.previous_run/'temporal/replayed_events_14d.csv');d=d[d.role.eq('test')]
    measurement=json.loads((args.previous_run/'temporal/temporal_measurement_calibration.json').read_text())
    scale=d.anchor.map(measurement['development_data']['normalized_score_sd_by_instrument'])
    se=((d.ball-d.target)/scale)**2
    rmse=np.sqrt(se.groupby(d.id).mean().mean())
    temporal.loc[temporal.method.eq('ball'),['rmse','rmse_raw','rmse_support_clipped']]=rmse
    ints=pd.read_csv(args.run/'temporal/revised_intervals_14d.csv');ints=ints[ints.method.eq('conditional_quantiles')]
    temporal.loc[temporal.method.eq('ball'),'patient_balanced_conformal_coverage']=ints.groupby(['anchor','id']).covered.mean().groupby('anchor').mean().mean()
    temporal.loc[temporal.method.eq('ball'),'mean_interval_width_fraction_of_legal_range']=ints.groupby(['anchor','id']).width_fraction.mean().groupby('anchor').mean().mean()
    if args.classical_run:
        e=pd.read_csv(args.classical_run/'temporal/corrected_comparator_events_14d.csv')
        base=temporal.loc[temporal.method.eq('ball')].iloc[0].to_dict()
        added=[];paired=[]
        bse=(((e.ball-e.z_true)/e.pooling_sd_normalized)**2).groupby(e.id).mean()
        draws=np.random.default_rng(101).integers(0,len(bse),size=(2000,len(bse)))
        for method in ['s0_direct_lgssm','markov_direct_transition']:
            error=(((e[method]-e.z_true)/e.pooling_sd_normalized)**2).groupby(e.id).mean().reindex(bse.index)
            raw=float(np.sqrt(error.mean()))
            clipped=(((e[method+'_clipped']-e.z_true)/e.pooling_sd_normalized)**2).groupby(e.id).mean()
            row=dict(base,method=method,rmse=raw,rmse_raw=raw,rmse_support_clipped=float(np.sqrt(clipped.mean())),
                patient_balanced_conformal_coverage=np.nan,mean_interval_width_fraction_of_legal_range=np.nan,
                selected_daily_persistence=np.nan,selected_questionnaire_weight=np.nan,selected_student_questionnaire_weight=np.nan)
            added.append(row)
            delta=np.sqrt(bse.to_numpy()[draws].mean(axis=1))-np.sqrt(error.to_numpy()[draws].mean(axis=1))
            lo,hi=np.quantile(delta,[.025,.975])
            paired.append(dict(comparator=method,rmse_difference=np.sqrt(bse.mean())-raw,ci_lo=lo,ci_hi=hi,n_patients=len(bse)))
        temporal=pd.concat([temporal,pd.DataFrame(added)],ignore_index=True)
        pd.DataFrame(paired).to_csv(out/'temporal_pairwise_bootstrap.csv',index=False)
    temporal.to_csv(out/'empirical_generalization_complete.csv',index=False)
    clinical=[];summaries=[];changes=[];selected=[]
    cutoff={'PHQ9':20,'BDI':29,'GAD7':15}
    for setting,cadences in [('primary',[7,14,21,28]),('temporal',[14])]:
        run=args.run if setting=='primary' else args.run/'temporal'
        oldrun=args.previous_run if setting=='primary' else args.previous_run/'temporal'
        for cad in cadences:
            d=pd.read_csv(run/f'revised_intervals_{cad}d.csv')
            old=pd.read_csv(oldrun/f'revised_intervals_{cad}d.csv')
            old=old[old.method.eq('conditional_quantiles')].copy();old['method']='previous_conditional_quantiles'
            combined=pd.concat([old,d],ignore_index=True)
            choices=json.loads((run/f'interval_choices_{cad}d.json').read_text())
            for inst,choice in choices.items():selected.append(dict(setting=setting,cadence=cad,instrument=inst,**choice))
            for method,g in combined.groupby('method'):
                summaries.append(dict(setting=setting,cadence=cad,method=method,coverage=pmean(g,g.covered),
                    width_fraction=float(g.width_fraction.mean()),n_events=len(g),n_patients=g.id.nunique()))
            chosen=d[d.method.eq('conditional_quantiles')].copy()
            matched=chosen.merge(old,on=['id','session','anchor'],suffixes=('','_old'),validate='one_to_one')
            assert len(matched)==len(chosen)==len(old)
            np.testing.assert_allclose(matched.target,matched.target_old)
            np.testing.assert_allclose(matched.ball,matched.ball_old)
            for inst,g in matched.groupby('anchor'):
                per=g.groupby('id')[['width_fraction','width_fraction_old','covered','covered_old']].mean()
                idx=np.random.default_rng(20260918).integers(0,len(per),size=(2000,len(per)))
                delta=per.width_fraction-per.width_fraction_old
                boot=delta.to_numpy()[idx].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])
                covlo,covhi=np.quantile(per.covered.to_numpy()[idx].mean(axis=1),[.025,.975])
                changes.append(dict(setting=setting,cadence=cad,instrument=inst,
                    old_width_fraction=g.width_fraction_old.mean(),new_width_fraction=g.width_fraction.mean(),
                    relative_width_change=g.width_fraction.mean()/g.width_fraction_old.mean()-1,
                    patient_balanced_width_change=delta.mean(),width_change_ci_lo=lo,width_change_ci_hi=hi,
                    old_coverage=per.covered_old.mean(),new_coverage=per.covered.mean(),
                    new_coverage_ci_lo=covlo,new_coverage_ci_hi=covhi,
                    n_patients=len(per),n_events=len(g)))
            if setting=='primary':
                chosen=chosen[chosen.baseline_score.notna()].copy()
                raw=chosen.baseline_score*chosen.anchor.map(HALF)+chosen.anchor.map(HALF)
                chosen['severe']=np.rint(raw)>=chosen.anchor.map(cutoff)
                scaling=pd.read_csv(args.previous_run/f'corrected_comparator_events_{cad}d.csv',
                    usecols=['id','session','anchor','pooling_sd_normalized','evaluation_role'])
                chosen=chosen.merge(scaling[scaling.evaluation_role.eq('test')],on=['id','session','anchor'],validate='one_to_one')
                chosen['sqerror']=((chosen.ball-chosen.target)/chosen.pooling_sd_normalized)**2
                for severe,g in chosen.groupby('severe'):
                    clinical.append(dict(cadence=cad,severe=bool(severe),coverage=pmean(g,g.covered),
                        width_fraction=g.width_fraction.mean(),rmse=np.sqrt(pmean(g,g.sqerror)),n_events=len(g),n_patients=g.id.nunique()))
    pd.DataFrame(clinical).to_csv(out/'baseline_clinical_severity.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'refinement_sensitivity_summary.csv',index=False)
    pd.DataFrame(changes).to_csv(out/'refinement_matched_changes.csv',index=False)
    pd.DataFrame(selected).to_csv(out/'refinement_selected_specifications.csv',index=False)
    print('Synchronized aggregate refinement and display inputs saved.')

if __name__=='__main__':main()
