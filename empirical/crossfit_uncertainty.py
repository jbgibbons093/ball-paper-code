"""Patient-excluded BALL predictions for an uncertainty sensitivity analysis.

Fresh five-member ensembles use the original architecture and training budget.
Only development patients enter these fits. Original fitted models and their
calibration/test predictions remain unchanged. Private artifacts stay outside
the source repository.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import revise_uncertainty as R

E,M=R.E,R.M
SEED=20260918

def fold_assignment(sessions,ids):
    """Deterministic two-fold patient split, stratified by development clinic."""
    s=sessions[sessions.id.isin(ids)].copy()
    s['clinic']=s.FacilityName.map(E._canonical_empirical_clinic)
    clinics=s.groupby('id').clinic.agg(lambda x:x.value_counts().index[0])
    assignment={}
    rng=np.random.default_rng(SEED)
    for _,g in clinics.groupby(clinics):
        p=np.asarray(sorted(g.index),dtype=int);rng.shuffle(p)
        assignment.update({int(pid):int(i%2) for i,pid in enumerate(p)})
    return assignment

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--frozen-run',type=Path,required=True)
    p.add_argument('--comorbidity-features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cadences',type=int,nargs='+',default=[7,14,21,28])
    p.add_argument('--temporal',action='store_true')
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if args.output.resolve().is_relative_to(R.REPO.resolve()):
        raise ValueError('Private training artifacts must stay outside the source repository.')
    if args.temporal and args.cadences!=[14]:raise ValueError('Temporal analysis uses 14 days.')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    sessions=E._coerce_sessions(pd.read_csv(E.OUT/'sessions.csv'))
    base_splits,heldout_roles,_=E._primary_empirical_partitions(sessions)
    if args.temporal:
        base_splits=E._generalization_split_maps(sessions)['forward_2025']
        sessions=sessions[sessions['split'].isin(['train','val'])].copy()
    development={int(k) for k,v in base_splits.items() if v=='train'}
    # Remove reserved patients before vocabulary, observation links, tensor
    # construction, or uncertainty training can see them.
    sessions=sessions[sessions.id.isin(development)].copy()
    folds=fold_assignment(sessions,development)
    source=sessions.groupby('id')['split'].first().to_dict()
    features=E._load_comorbidity_sessions(args.comorbidity_features)
    features=features[features.id.isin(development)].copy()
    questionnaires=pd.read_csv(E.OUT/'questionnaire_events_all.csv')
    questionnaires=questionnaires[questionnaires.id.isin(development)].copy()
    (args.output/'crossfit_patient_folds.json').write_text(json.dumps(folds,sort_keys=True))
    specification={'seed':SEED,'n_folds':2,'fold_unit':'patient, stratified by canonical development clinic',
        'ensemble_members':5,'teacher_epochs':120,'student_epochs':120,'anchor_warmup':60,'kl_warmup':60,
        'measurement_fit':'source-training patients in the training fold only; reused across cadences',
        'preprocessing':'training fold only','reserved_calibration_or_test_in_fitting':False,
        'primary_frozen_point_estimator_changed':False,'cadences':args.cadences,'temporal':args.temporal}
    (args.output/'crossfit_specification.json').write_text(json.dumps(specification,indent=2))
    for fold in range(2):
        folder=args.output/f'fold_{fold}';folder.mkdir(exist_ok=True)
        mapping={pid:('test' if f==fold else 'train') for pid,f in folds.items()}
        train_ids={pid for pid,v in mapping.items() if v=='train'}
        test_ids=set(mapping)-train_ids
        assert train_ids.isdisjoint(test_ids) and train_ids|test_ids==development
        measurement_path=folder/'measurement_calibration.json'
        if measurement_path.exists():measurement=json.loads(measurement_path.read_text())
        else:
            print(f'fold {fold}: fitting observation links on excluded-patient development partition',flush=True)
            fit_map={pid:'train' for pid in train_ids if source[pid]=='train'}
            measurement=E._fit_development_measurement_calibration(questionnaires,fit_map)
            measurement['fit_scope']='source-training patients in this cross-fitting training fold only'
            measurement_path.write_text(json.dumps(measurement,indent=2))
        for cadence in args.cadences:
            final=folder/f'oof_events_{cadence}d.csv'
            if final.exists():print(f'fold {fold} {cadence}d complete cache retained',flush=True);continue
            anchors=pd.read_csv(E.OUT/f'anchors_sparse_{cadence}d.csv')
            anchors=anchors[anchors.id.isin(development)].copy()
            data,days=E._empirical_neural_dataset(anchors,sessions,None,features,measurement,patient_splits=mapping)
            cfg=M.SSMConfig(d_model=96,n_heads=4,n_layers=3,teacher_epochs=120,student_epochs=120,
                anchor_warmup_epochs=60,kl_warmup_epochs=60,batch_size=32,ensemble_size=5,
                max_individuals=data.config.n,seed=E.EMPIRICAL_SEED,rdoc_drift_head=False,use_alpha_slow=False,
                delta_phi=.9,anchor_weight=10.,student_anchor_weight=2.,progress_every_epochs=30,
                diagnostics_every_epochs=20,state_checkpoint_every_epochs=30)
            signature=hashlib.sha256(json.dumps({'mapping':mapping,'measurement':measurement,'cadence':cadence,
                'anchors_sha256':hashlib.sha256(pd.util.hash_pandas_object(anchors,index=True).values.tobytes()).hexdigest(),
                'specification':specification},sort_keys=True).encode()).hexdigest()
            cfg=E._empirical_checkpointed_config(cfg,folder/'member_checkpoints',f'{cadence}d_primary_ball',signature)
            print(f'fold {fold} {cadence}d: training 5 members, {len(train_ids)} training and {len(test_ids)} excluded patients',flush=True)
            result=M.fit_ball_ssm(data,cfg,device=args.device,causal=True,prediction_split='test')
            E._assert_result_training_valid(result,f'crossfit_{fold}_{cadence}',5)
            del result
            torch.cuda.empty_cache()
            train,stats=M.build_ssm_batch(data,'train',torch.device('cpu'),cfg.max_individuals,
                causal_impute=True,include_rdoc_in_encoder=cfg.encoder_use_rdoc)
            test,_=M.build_ssm_batch(data,'test',torch.device('cpu'),cfg.max_individuals,
                feature_stats=stats,causal_impute=True,include_rdoc_in_encoder=cfg.encoder_use_rdoc)
            roles={pid:('development' if pid in test_ids else 'crossfit_training') for pid in mapping}
            context=(data,days,anchors,sessions,mapping,roles,measurement,cfg,train,test,stats)
            replay_args=SimpleNamespace(frozen_run=folder,output=folder,cadences=[cadence],device=args.device,
                batch_size=32,temporal=False)
            R.replay(replay_args,prepared={cadence:context},verify_frozen=False)
            rows=pd.read_csv(folder/f'replayed_events_{cadence}d.csv')
            rows=rows[rows.role.eq('development')].copy();rows['crossfit_fold']=fold
            assert set(rows.id).issubset(test_ids) and set(rows.id).isdisjoint(train_ids)
            rows.to_csv(final,index=False)
            print(f'fold {fold} {cadence}d: excluded-patient predictions saved',flush=True)
    for cadence in args.cadences:
        d=pd.concat([pd.read_csv(args.output/f'fold_{f}'/f'oof_events_{cadence}d.csv') for f in range(2)],ignore_index=True)
        assert not d.duplicated(['id','session','anchor']).any()
        d.to_csv(args.output/f'oof_events_{cadence}d.csv',index=False)
    print('CROSS-FITTING COMPLETE',flush=True)

if __name__=='__main__':main()
