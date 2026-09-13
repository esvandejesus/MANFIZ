import unittest
import numpy as np
from manfiz import FitConfig,PremiseConfig,ReductionConfig
from manfiz.premises import BellPremises


class ConfigurationTests(unittest.TestCase):
    def test_fractional_budgets_and_memberships_are_rejected(self):
        with self.assertRaises(ValueError):
            PremiseConfig(max_iter=10.5).validate()
        with self.assertRaises(ValueError):
            ReductionConfig(q_max=1200.1).validate(11)
        with self.assertRaises(ValueError):
            FitConfig(batch_size=16.5).validate(11)
        with self.assertRaises(ValueError):
            BellPremises.initialize(np.array([[-1.],[1.]]),2.5)

    def test_config_roundtrip_preserves_all_settings(self):
        from dataclasses import asdict
        cfg=FitConfig(premise=PremiseConfig(max_iter=37),reduction=ReductionConfig(q_max=800),
                      parameter_drift=.001)
        recovered=FitConfig.from_dict(asdict(cfg))
        self.assertEqual(asdict(cfg),asdict(recovered))


if __name__=='__main__':
    unittest.main()
