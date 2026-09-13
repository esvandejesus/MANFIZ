"""Small archived MATLAB oracle. Never imported by model training."""
from pathlib import Path
import unittest
import numpy as np
from manfiz.benchmarks import simulate
from manfiz.premises import BellPremises,affine,evaluate_center


class MatlabReferenceTests(unittest.TestCase):
    def test_original_simulator_trajectories(self):
        with np.load(Path(__file__).parent/'fixtures/matlab_reference.npz',allow_pickle=False) as f:
            for s in (1,2,3):
                for run in range(1,7):
                    np.testing.assert_allclose(simulate(s,f[f's{s}_u{run}']),f[f's{s}_truth{run}'],rtol=0,atol=1e-12)

    def test_original_stored_centers_with_original_parameters(self):
        with np.load(Path(__file__).parent/'fixtures/matlab_reference.npz',allow_pickle=False) as f:
            for s in (1,2,3):
                xn=(f[f's{s}_X']-f[f's{s}_muX'])/f[f's{s}_sigmaX']
                p=BellPremises((2,2,2),f[f's{s}_pars'],f[f's{s}_rules'])
                center=evaluate_center(affine(xn),p.weights(xn[:,[4,6,8]]),f[f's{s}_beta'])
                center=center*f[f's{s}_sigmaY']+f[f's{s}_muY']
                np.testing.assert_allclose(center,f[f's{s}_expected_center'],rtol=0,atol=1e-12)


if __name__=='__main__':
    unittest.main()
