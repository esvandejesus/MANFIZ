import itertools
import unittest
import numpy as np
from manfiz import Zonotope, ReductionConfig
from manfiz.zonotopes import reduce_fixed, reduce_adaptive, measurement_update, weighted_radius
from manfiz.calibration import minimax_correction, verify_witness


class GeometryTests(unittest.TestCase):
    def test_outer_reduction_support_and_hard_budget(self):
        rng=np.random.default_rng(41); r=rng.normal(size=(3,32))
        w=np.array([[2.,.3,0],[.3,1.,.1],[0,.1,.5]])
        reduced=reduce_fixed(r,9,w)
        directions=rng.normal(size=(1000,3))
        self.assertEqual(reduced.shape,(3,9))
        self.assertTrue(np.all(abs(directions@r).sum(1) <= abs(directions@reduced).sum(1)+1e-12))
        z=Zonotope(np.arange(3.),r).reduce(9,w)
        np.testing.assert_array_equal(z.center,np.arange(3.))
        self.assertAlmostEqual(weighted_radius(r,w)**2,np.trace(w@r@r.T),places=10)

    def test_adaptive_selection_matches_explicit_candidate_search(self):
        rng=np.random.default_rng(9); r=rng.normal(size=(3,27))*.05
        c=rng.normal(size=(5,3)); f=np.full(5,.04); w=np.diag([1.,3.,.8])
        for cycle_cap in (1.005,.00001):
            cfg=ReductionConfig(q_target=5,q_max=20,max_cycle_ratio=cycle_cap)
            actual,_,info=reduce_adaptive(r,c,f,cfg,w)
            ref=weighted_radius(r,w); feasible=[]; scores=[]
            for q in range(5,21):
                rr=reduce_fixed(r,q,w)
                _,post,_=measurement_update(np.zeros(3),rr,c,np.zeros(5),f)
                a=weighted_radius(rr,w)/ref; b=weighted_radius(post,w)/ref
                feasible.append(a <= cfg.max_inflation*(1+cfg.inflation_tolerance) and
                                b <= cfg.max_cycle_ratio*(1+cfg.cycle_tolerance))
                scores.append(max(a/cfg.max_inflation-1,0)**2+max(b/cfg.max_cycle_ratio-1,0)**2)
            if any(feasible):
                expected=5+feasible.index(True)
            else:
                best=min(scores)
                expected=5+next(i for i,v in enumerate(scores) if v <= best+cfg.tie_tolerance*max(1,abs(best)))
            self.assertEqual(info['selected_budget'],expected)
            self.assertEqual(info['fallback'],not any(feasible))
            np.testing.assert_allclose(actual,reduce_fixed(r,expected,w),atol=1e-14)

    def test_measurement_update_contains_every_feasible_prior_vertex(self):
        center=np.array([.2,-.3]); r=np.array([[1.,.1],[.2,.8]])
        c=np.array([[1.,.3],[-.2,.7]]); z=np.array([.3,-.1]); f=np.array([1.4,1.2])
        newc,newr,diag=measurement_update(center,r,c,z,f)
        points=[center+r@np.array(sign) for sign in itertools.product((-1,1),repeat=2)]
        points=[p for p in points if np.all(abs(c@p-z) <= f)]
        self.assertGreater(len(points),0)
        for p in points:
            self.assertLess(max(verify_witness([newc],[newr],p)),1e-8)
        self.assertLess(diag['joseph_error'],1e-12)

    def test_joint_feasibility_is_stricter_than_pointwise_enclosure(self):
        # Each target can choose a different delta in [-1,1], but one constant
        # delta cannot explain both signs with a sensor bound below one.
        h=np.ones((2,1)); e=np.array([-1.,1.])
        lp=minimax_correction(h,e,np.array([1.]))
        self.assertAlmostEqual(lp['minimum_error'],1.)
        self.assertLess(lp['dual_gap'],1e-10)
        negative=minimax_correction(h,np.array([-2.,-2.]))
        self.assertAlmostEqual(negative['witness'][0],-2.)
        self.assertAlmostEqual(negative['minimum_error'],0.)

    def test_linear_map_and_minkowski_support(self):
        z=Zonotope([1.,2.],np.diag([.1,.2])); other=Zonotope([-.4,.2],np.diag([.3,.1]))
        total=z.minkowski_sum(other).linear_map([[2.,-1.]],offset=[.5])
        self.assertAlmostEqual(total.center[0],-.5)
        np.testing.assert_allclose(total.interval,([-1.6],[.6]))


if __name__=='__main__':
    unittest.main()
