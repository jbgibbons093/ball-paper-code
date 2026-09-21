"""Check actual saved networks for strict next-session questionnaire feedback."""
import argparse
import dataclasses
import json
from pathlib import Path
import numpy as np
import torch
from revise_uncertainty import setup,load_member,M

def main():
    p=argparse.ArgumentParser();p.add_argument('--frozen-run',type=Path,required=True)
    p.add_argument('--comorbidity-features',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(8);device=torch.device('cuda')
    data,days,anchors,sessions,splits,roles,measurement,cfg,train,test,stats=setup(args,14)
    indices=np.flatnonzero(test.valid.sum(axis=1).numpy()>=30)[:4]
    base=M._subbatch(test,torch.as_tensor(indices))
    base=dataclasses.replace(base,**{f:getattr(base,f).to(device) for f in M._B_FIELDS})
    t=15;modified=base.enc_features.clone()
    modified[:,t+1:,-6:-3]+=0.5
    modified[:,t+1:,-3:]=1.
    changed=dataclasses.replace(base,enc_features=modified)
    checks=[]
    with torch.no_grad():
        for k in range(5):
            _,net=load_member(args,14,k,data,train,cfg,device)
            before,_=net(base);after,_=net(changed)
            earlier=float((before[:,:t+1]-after[:,:t+1]).abs().max())
            later=float((before[:,t+1:30]-after[:,t+1:30]).abs().max())
            assert earlier<1e-6,(k,earlier)
            assert later>1e-6,(k,later)
            checks.append(dict(member=k,max_pre_reveal_change=earlier,max_post_reveal_change=later))
    report={'causality_passed':True,'members':checks,'patient_identifiers_exported':False,
            'test':'Changed future questionnaire inputs after the target; prior and current estimates stayed fixed.'}
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
