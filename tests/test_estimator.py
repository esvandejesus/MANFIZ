import itertools
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from manfiz import MANFIZ, FitConfig, PremiseConfig, ReductionConfig, NARXSpec


class EstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(82)
        cls.x=rng.uniform(-1,1,(180,3))
        cls.y=np.column_stack([.4*cls.x[:,0]+.2*cls.x[:,1]+.3*np.sin(2*cls.x[:,2]),
                               -.1*cls.x[:,0]+.5*cls.x[:,1]**2-.2*cls.x[:,2]])
        cls.y+=rng.uniform(-.01,.01,cls.y.shape)
        cls.groups=np.repeat(np.arange(3),60)
        cfg=FitConfig(premise=PremiseConfig(max_iter=100,max_evaluations=200),
                      reduction=ReductionConfig(q_target=8,q_max=36),
                      batch_size=8,max_buffer_size=24,pe_ratio_min=1e-5,min_samples_per_group=30)
        cls.model=MANFIZ([2],config=cfg).fit(cls.x,cls.y,groups=cls.groups,sensor_bound=.01)
        cls.model.calibrate_validation(cls.x[:20],cls.y[:20])
        cls.model.calibrate_noise(cls.x[:20],cls.y[:20],sensor_bound=.02,
                                  regressor_bound=[.03,.02,0.])

    def test_fresh_joint_training_and_generic_dimensions(self):
        model=self.model; p=model.fit_report_['premise']
        self.assertTrue(model.fit_report_['trained_from_scratch'])
        self.assertLess(p['final_objective'],p['initial_objective'])
        self.assertGreater(np.linalg.norm(p['final_parameters']-p['initial_parameters']),0)
        self.assertEqual(model.beta_.shape,(2,4,2))
        self.assertTrue(all(n > 0 for n in model.fit_report_['accepted_updates']))
        for out in model.outputs_:
            self.assertLess(max(out['witness_membership_errors']),1e-8)
            self.assertTrue(all(r.shape[1] <= 44 for r in out['generators']))

    def test_noise_propagation_covers_all_box_vertices(self):
        model=self.model; x=self.x[3:4].copy()
        clean=model.predict_interval(x,mode='parameters')
        # Worst-case clean output on each boundary; exhaust every vertex of
        # the two uncertain regressor coordinates and current output noise.
        d=model.noise_calibration_['clean_defect']; by=model.noise_calibration_['sensor_bound']
        for signs in itertools.product((-1,1),repeat=2):
            measured=x.copy(); measured[0,:2]+=np.array(signs)*np.array([.03,.02])
            interval=model.predict_interval(measured,mode='noise')
            for sign_clean,sign_noise in itertools.product((-1,1),repeat=2):
                target=clean.center+sign_clean*(clean.radius+d)+sign_noise*by
                self.assertTrue(np.all(target <= interval.upper+1e-12))
                self.assertTrue(np.all(target >= interval.lower-1e-12))

    def test_noise_on_premise_columns_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'noise-free premise'):
            self.model.calibrate_noise(self.x[:4],self.y[:4],sensor_bound=.01,regressor_bound=.01)

    def test_reuse_with_different_training_targets_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'same X,y'):
            self.model.fit_zonotopes(self.x,self.y+1,groups=self.groups,sensor_bound=.01)

    def test_save_load_exact_for_all_interval_modes_and_no_pickle(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.npz'; self.model.save(path)
            restored=MANFIZ.load(path)
            for mode in ('parameters','validation','noise'):
                a=self.model.predict_interval(self.x[:10],mode=mode)
                b=restored.predict_interval(self.x[:10],mode=mode)
                np.testing.assert_array_equal(a.center,b.center)
                np.testing.assert_array_equal(a.radius,b.radius)
            with np.load(path,allow_pickle=False) as data:
                arrays={k:data[k] for k in data.files}
            meta=json.loads(arrays['metadata'].item()); meta['format_version']=999
            arrays['metadata']=np.asarray(json.dumps(meta)); np.savez(path,**arrays)
            with self.assertRaisesRegex(ValueError,'format/version'):
                MANFIZ.load(path)

    def test_chunked_radii_agree(self):
        a=self.model.predict_interval(self.x,mode='noise',chunk_size=11)
        b=self.model.predict_interval(self.x,mode='noise',chunk_size=1000)
        np.testing.assert_allclose(a.center,b.center,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(a.radius,b.radius,rtol=1e-13,atol=1e-13)

    def test_free_run_does_not_read_future_output_measurements(self):
        class LinearModel:
            def predict(self,x,nominal=False):
                return (.7*x[:,0]+.2*x[:,1])[:,None]
        spec=NARXSpec(output_lags=(1,),input_lags=(1,),discard=1)
        u=np.arange(20.)[:,None]; first=np.zeros((20,1)); second=first.copy(); second[1:]=999
        a=spec.free_run(LinearModel(),u,first)
        b=spec.free_run(LinearModel(),u,second)
        np.testing.assert_array_equal(a,b)
        self.assertAlmostEqual(a[1,0],.2)


if __name__=='__main__':
    unittest.main()
