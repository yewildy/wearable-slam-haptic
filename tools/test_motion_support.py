import unittest
import numpy as np
from motion_support import SupportedDistancePolicy


class SupportTests(unittest.TestCase):
    def policy(self):
        return SupportedDistancePolicy(backend='reference')

    def test_isolated_near_does_not_override_group(self):
        p = self.policy()
        points = np.array([[.1,0,0],[1,0,0],[1,.01,0],[1,0,.01],[1,.01,.01]])
        r = p.process(points,'livox_frame',1.)
        self.assertTrue(r['valid'])
        self.assertAlmostEqual(r['distance_m'],1.)
        self.assertEqual(r['isolated_removed_points'],1)
        self.assertEqual(r['self_removed_points'],0)

    def test_dense_near_remains_uncertain(self):
        r = self.policy().process(np.array([[.1,0,0],[.1,.01,0],[.1,0,.01],[.1,.01,.01]]),'livox_frame',1.)
        self.assertEqual(r['state'],'NEAR_FIELD_UNCERTAIN')
        self.assertEqual(r['isolated_removed_points'],0)

    def test_all_sparse_is_unknown_not_free(self):
        r = self.policy().process(np.array([[.1,0,0],[.3,0,0]]),'livox_frame',1.)
        self.assertFalse(r['valid'])
        self.assertIsNone(r['strength'])
        self.assertEqual(r['isolated_removed_points'],2)

    def test_thin_target_can_be_missed(self):
        r = self.policy().process(np.array([[.3,0,0],[2,0,0],[2,.01,0],[2,0,.01],[2,.01,.01]]),'livox_frame',1.)
        self.assertTrue(r['valid'])
        self.assertAlmostEqual(r['distance_m'],2.)
        self.assertTrue(r['thin_object_miss_risk'])

    def test_duplicate_stamp_rejected(self):
        p=self.policy(); xyz=np.zeros((0,3))
        p.process(xyz,'livox_frame',1.)
        self.assertEqual(p.process(xyz,'livox_frame',1.)['state'],'TIMESTAMP_NONINCREASING')

    def test_fast_scipy_support_matches_reference(self):
        try:import scipy
        except ImportError:self.skipTest('SciPy not installed')
        rng=np.random.default_rng(42)
        points=np.vstack([rng.normal(0,.05,(100,3)),np.zeros((4,3)),[[1,0,0]]])
        fast=SupportedDistancePolicy(backend='scipy')
        slow=self.policy()
        np.testing.assert_array_equal(fast._support(points),slow._support(points))
