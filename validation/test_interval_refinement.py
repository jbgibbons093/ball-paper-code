"""Information-boundary checks for interval-history refinement."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'empirical'))
from interval_history import add_history,EXTRA_FEATURES
from refine_intervals import predict,QUANTILES

class ConstantQuantile:
    def __init__(self,value):self.value=value
    def predict(self,x):return np.full(len(x),self.value)

class QuantileIntervalTests(unittest.TestCase):
    def test_shortest_pair_depends_only_on_predictors(self):
        values=[-.9,-.8,-.7,-.6,-.5,.1,.15,.2,.4,.6]
        bundle={'models':[ConstantQuantile(v) for v in values],
            'features':['ball'],'quantiles':QUANTILES,'tail_mode':'shortest'}
        frame=pd.DataFrame({'ball':[0.,0.],'target':[-1.,1.]})
        lo,hi=predict(bundle,frame)
        np.testing.assert_allclose(lo,[-.7,-.7])
        np.testing.assert_allclose(hi,[.2,.2])
        frame['target']=[100.,-100.]
        a,b=predict(bundle,frame)
        np.testing.assert_equal(a,lo);np.testing.assert_equal(b,hi)

    def test_crossed_quantiles_are_ordered_and_bounded(self):
        bundle={'models':[ConstantQuantile(3.),ConstantQuantile(-3.)],
            'features':['ball'],'quantiles':[.025,.975],'tail_mode':'central'}
        lo,hi=predict(bundle,pd.DataFrame({'ball':[0.]}))
        np.testing.assert_equal(lo,[-1.]);np.testing.assert_equal(hi,[1.])

class CausalHistoryTests(unittest.TestCase):
    def setUp(self):
        self.past=pd.DataFrame([
            [1,1,0,'PHQ9',10],[1,2,7,'PHQ9',14],[1,3,14,'PHQ9',25],
            [1,4,21,'PHQ9',27],[1,2,7,'GAD7',21]],
            columns=['id','session','day','anchor','value'])
        self.events=pd.DataFrame([[1,3,14,'PHQ9',0.]],columns=['id','session','day','anchor','target'])

    def test_target_and_future_scores_cannot_change_history(self):
        a=add_history(self.events,self.past)
        p=self.past.copy();p.loc[p.session>=3,'value']=0
        e=self.events.copy();e['target']=-1
        b=add_history(e,p)
        np.testing.assert_allclose(a[EXTRA_FEATURES],b[EXTRA_FEATURES],equal_nan=True)

    def test_instrument_specific_gap_and_change(self):
        a=add_history(self.events,self.past).iloc[0]
        self.assertEqual(a.same_instrument_gap,7)
        self.assertAlmostEqual(a.last_score_change,4/13.5)
        self.assertAlmostEqual(a.recall_overlap_fraction,.5)
        self.assertEqual(a.same_instrument_count_14d,2)

    def test_feedback_only_after_query_session(self):
        retained=self.past[self.past.session.eq(1)]
        query=pd.DataFrame([[1,3,14,'PHQ9',1.]],columns=['id','session','day','anchor','target'])
        same=add_history(self.events,retained,query)
        without=add_history(self.events,retained)
        np.testing.assert_allclose(same[EXTRA_FEATURES],without[EXTRA_FEATURES],equal_nan=True)
        later=self.events.copy();later['session']=4;later['day']=21
        result=add_history(later,retained,query).iloc[0]
        self.assertEqual(result.same_instrument_gap,7)
        self.assertEqual(result.same_instrument_count_28d,2)

if __name__=='__main__':unittest.main()
