import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'empirical'))
from prospective_classical import forward_filter, prior_counts, pattern_groups

class ProspectiveClassicalTests(unittest.TestCase):
    def anchors(self):
        return pd.DataFrame([dict(t=1,observed=True,window_start=0,window_end=1,
            value=4.,loading=1.,intercept=0.,observation_sd=.5),
            dict(t=1,observed=True,window_start=0,window_end=1,
            value=2.,loading=2.,intercept=0.,observation_sd=.5)])
    def run_filter(self,anchors,daily=None):
        return forward_filter(np.zeros(4),np.ones(4),np.ones(4),
            np.zeros(5) if daily is None else daily,np.ones(5,dtype=bool),2.,anchors,25.)[0]
    def test_delayed_all_same_session_instruments(self):
        a=self.anchors();x=self.run_filter(a);a.value=99.;y=self.run_filter(a)
        np.testing.assert_array_equal(x[:2],y[:2]);self.assertNotEqual(x[2],y[2])
    def test_future_structured_values_do_not_change_issued_estimates(self):
        x=self.run_filter(self.anchors());y=self.run_filter(self.anchors(),np.array([0.,0.,0.,99.,99.]))
        np.testing.assert_array_equal(x[:3],y[:3])
    def test_future_pattern_and_value_do_not_change_earlier_predictions(self):
        a=self.anchors();b=pd.concat([a,a.assign(t=4,value=99.,window_end=4)])
        np.testing.assert_array_equal(self.run_filter(a),self.run_filter(b))
        np.testing.assert_array_equal(pattern_groups(a,5,[1,3]),pattern_groups(b,5,[1,3]))
        np.testing.assert_array_equal(prior_counts(a,5),[0,0,2,2,2])
    def test_recall_observation_matches_prefix_gaussian_solution(self):
        a=self.anchors();actual=self.run_filter(a)
        for end in range(5):
            n=end+1;rows=[];ys=[]
            def add(v,y,sd):rows.append(np.asarray(v)/sd);ys.append(y/sd)
            e=np.eye(n);add(e[0],0.,5.)
            for k in range(n):add(e[k],0.,np.sqrt(2.))
            for k in range(1,n):add(e[k]-e[k-1],0.,1.)
            for r in a[a.t<end].itertuples():
                row=np.zeros(n);row[r.window_start:r.window_end+1]=r.loading/(r.window_end-r.window_start+1)
                add(row,r.value,r.observation_sd)
            expected=np.linalg.lstsq(np.asarray(rows),ys,rcond=None)[0][-1]
            self.assertAlmostEqual(actual[end],expected,places=10)

if __name__=='__main__':unittest.main()
