"""Future perturbations must not alter issued classical simulation estimates."""
from pathlib import Path
import copy,sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import direct_rdoc_benchmark as B
import direct_rdoc_fair_comparator as F
from prospective_simulation_filter import history_strata,filter_person
from test_direct_rdoc_input_contract import _args

class SimulationCausalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args=_args();cls.args.n=24;cls.args.t=30;cls.args.s0_basis='linear'
        cls.args.prior_sd=5.0
        cls.data=B.make_data(cls.args,2027,.1,'missingness')
        cls.train=set(cls.data.individuals.loc[cls.data.individuals.split.eq('train'),'id'].astype(int))
        cls.pid=int(cls.data.individuals.loc[cls.data.individuals.split.eq('test'),'id'].iloc[0])
    def test_gaussian_and_markov_future_invariance(self):
        data=self.data;pid=self.pid;cut=15
        other=copy.copy(data);other.anchors=data.anchors.copy();other.components=data.components.copy();other.daily=data.daily.copy()
        mask=other.anchors.id.eq(pid)&other.anchors.t.ge(cut)
        other.anchors.loc[mask,'value']=999.;other.anchors.loc[mask,'observed']=False
        for col in [c for c in other.components if c.startswith('B')]+['a','dt']:
            if col not in other.components:continue
            other.components.loc[other.components.id.eq(pid)&other.components.t.ge(cut),col]=0 if col=='a' else 99.
        for col in [c for c in other.daily if c.startswith('X')]:
            other.daily.loc[other.daily.id.eq(pid)&other.daily.t.gt(cut),col]=99.
        for nstrata in [0,3]:
            groups=history_strata(data,max(nstrata,1),self.train)
            changed=history_strata(other,max(nstrata,1),self.train)
            np.testing.assert_array_equal(groups[pid][:cut+1],changed[pid][:cut+1])
            for channel in ['Y1','Y2']:
                readout=F.fit_daily_readout(data,self.train,1.,anchor_name=channel)
                arrays=F.person_arrays(data,pid,readout,anchor_name=channel)
                _,_,b,_,_,_,action,recent,burden=arrays
                nf=F.transition_features(b,action,recent,burden,data.config.n_treatment_types,basis='linear').shape[1]
                theta=np.linspace(-.02,.02,nf+nstrata)
                kwargs=dict(anchor_name=channel,persistence=.6,innovation_sd=.75,n_strata=nstrata,helpers=F)
                x=filter_person(data,pid,theta,readout,self.args,stratum=groups[pid],**kwargs)[0]
                y=filter_person(other,pid,theta,readout,self.args,stratum=changed[pid],**kwargs)[0]
                np.testing.assert_allclose(x[:cut+1],y[:cut+1],atol=1e-12,rtol=0.)

if __name__=='__main__':unittest.main()
