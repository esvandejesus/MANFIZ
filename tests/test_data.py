import unittest
import numpy as np
from manfiz import NARXSpec, Standardizer
from manfiz.benchmarks import simulate, make_training_runs, make_run
from manfiz.premises import BellPremises


class DataTests(unittest.TestCase):
    def test_narx_causality_and_experiment_boundaries(self):
        spec=NARXSpec(output_lags=(1,2),input_lags=(1,),discard=2)
        u=np.arange(8.)[:,None]; y=np.column_stack([10+u[:,0],100+u[:,0]])
        a=spec.transform(u,y)
        np.testing.assert_array_equal(a.X[0],[11,10,101,100,1])
        np.testing.assert_array_equal(a.y[0],[12,102])
        altered=y.copy(); altered[5:]+=1000
        np.testing.assert_array_equal(spec.transform(u,altered).X[:4],a.X[:4])
        both=spec.transform_runs([(u,y),(u+100,y+1000)])
        np.testing.assert_array_equal(both.X[6],[1011,1010,1101,1100,101])
        np.testing.assert_array_equal(both.sample_indices,np.tile(np.arange(2,8),2))

    def test_normalization_training_only_and_constant_features(self):
        x=np.array([[1.,8.],[2.,8.],[3.,8.]])
        scale=Standardizer.fit(x); before=scale.mean.copy()
        np.testing.assert_allclose(scale.transform(x)[:,0],[-1,0,1])
        scale.transform([[100.,8.]])
        np.testing.assert_array_equal(before,scale.mean)
        np.testing.assert_array_equal(scale.transform(x)[:,1],0)
        np.testing.assert_allclose(scale.inverse(scale.transform(x)),x)

    def test_simulator_early_steps_hand_computed(self):
        u=np.array([[.2,-.3,.1],[.5,.4,-.2],[.1,.2,.3],[.4,-.1,.2]])
        y=simulate(1,u)
        self.assertAlmostEqual(y[2,0],.30*.5+.18*(-.3)-.15*(-.2)+.06*.5*.4)
        self.assertAlmostEqual(y[2,1],.20*.2-.26*.4+.24*.1+.05*.4**2+.04*.5*(-.2))
        y=simulate(2,u)
        self.assertAlmostEqual(y[2,1],.22*np.sin(1.3*.4)+.18*np.tanh(-.2)+.08*.2**2-.05*.4*(-.2))
        y=simulate(3,u)
        x1=.18*np.tanh(.2)+.10*(-.3)*.1
        x2=.15*np.sin(-.3)+.12*.1; x3=.10*.2**2-.08*.1**2
        self.assertAlmostEqual(y[1,0],x1+.15*x2**2)
        self.assertAlmostEqual(y[1,1],x2+.20*np.tanh(x3)+.05*x1*x3)

    def test_noise_bounds_and_reproducibility(self):
        a=make_run(1,n_samples=100,noise_std=.02,phase_seed=123,noise_seed=321)
        b=make_run(1,n_samples=100,noise_std=.02,phase_seed=123,noise_seed=321)
        np.testing.assert_array_equal(a.y,b.y)
        self.assertLessEqual(abs(a.y-a.y_true).max(),np.sqrt(3)*.02)
        runs=make_training_runs(2,n_samples=100)
        self.assertEqual(len(set(r.noise_seed for r in runs)),6)

    def test_log_memberships_extreme_inputs_stay_normalized(self):
        p=BellPremises.initialize(np.array([[-2.,-1.],[2.,1.]]),2)
        w=p.weights(np.array([[0.,0.],[1e100,-1e100]]))
        np.testing.assert_allclose(w.sum(1),1.,atol=1e-12)
        self.assertTrue(np.all(np.isfinite(w)))
        self.assertTrue(np.all(w >= 0))


if __name__=='__main__':
    unittest.main()
