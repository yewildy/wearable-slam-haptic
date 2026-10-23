"""去噪后的点必须仍属于足够大的连通组；不是人体身份验证。"""
import unittest
import numpy as np
from motion_support import SupportedDistancePolicy


def group(distance):
    return np.array([[distance, 0, 0], [distance, .01, 0],
                     [distance, 0, .01], [distance, .01, .01]])


def star():
    return np.array([[.1, 0, 0], [.1, .07, 0], [.1, -.07, 0], [.1, 0, .07]])


class GroupRegressionTests(unittest.TestCase):
    def test_pruned_star_does_not_override_far_group(self):
        policy = SupportedDistancePolicy(backend='reference')
        report = policy.process(np.vstack([star(), group(.8)]), 'livox_frame', 1.)
        self.assertEqual(report['state'], 'DISTANCE_CANDIDATE')
        self.assertAlmostEqual(report['distance_m'], .8)
        self.assertEqual(report['isolated_removed_points'], 4)
        self.assertEqual(report['postfilter_small_group_removed_points'], 1)

    def test_pruned_star_alone_is_unknown(self):
        report = SupportedDistancePolicy(backend='reference').process(star(), 'livox_frame', 1.)
        self.assertFalse(report['valid'])
        self.assertIsNone(report['strength'])
        self.assertEqual(report['state'], 'NO_REMAINING_OBSERVATIONS')

    def test_new_dense_near_group_survives(self):
        report = SupportedDistancePolicy(backend='reference').process(group(.1), 'livox_frame', 1.)
        self.assertEqual(report['state'], 'NEAR_FIELD_UNCERTAIN')
        self.assertEqual(report['isolated_removed_points'], 0)

    def test_connected_chain_is_not_recursively_eroded(self):
        policy = SupportedDistancePolicy(backend='reference')
        points = np.array([[.1 + i*.07, 0, 0] for i in range(4)])
        np.testing.assert_array_equal(policy._group_mask(points), [True]*4)

    def test_empty_and_small_disconnected_groups(self):
        policy = SupportedDistancePolicy(backend='reference')
        self.assertEqual(len(policy._group_mask(np.empty((0, 3)))), 0)
        points = np.vstack([group(.1)[:3], group(.8)[:2], group(1.5)])
        np.testing.assert_array_equal(policy._group_mask(points), [False]*5 + [True]*4)

    def test_duplicate_returns_remain_counted_as_returns(self):
        policy = SupportedDistancePolicy(backend='reference')
        np.testing.assert_array_equal(policy._group_mask(np.tile([.1, 0, 0], (4, 1))), [True]*4)

    def test_sparse_graph_matches_full_reference(self):
        try:
            import scipy
        except ImportError:
            self.skipTest('SciPy not installed')
        fast = SupportedDistancePolicy(backend='scipy')
        reference = SupportedDistancePolicy(backend='reference')
        rng = np.random.default_rng(20261024)
        for count in (0, 1, 2, 3, 4, 20, 200):
            for _ in range(5):
                points = rng.uniform(0, .4, (count, 3))
                np.testing.assert_array_equal(fast._group_mask(points), reference._group_mask(points))
        chain = np.array([[i*.08, 0, 0] for i in range(4)])
        np.testing.assert_array_equal(fast._group_mask(chain), reference._group_mask(chain))
        duplicates = np.tile([.1, 0, 0], (20, 1))
        np.testing.assert_array_equal(fast._group_mask(duplicates), reference._group_mask(duplicates))


class PipelineGroupTests(unittest.TestCase):
    def test_warmed_pipeline_rejects_star_but_alerts_new_group(self):
        try:
            import scipy
        except ImportError:
            self.skipTest('SciPy not installed')
        from motion_pipeline import MotionEngine
        engine = MotionEngine()
        for i in range(110):
            engine.process(group(.8), 'livox_frame', 1. + i*.1)
        result = engine.process(np.vstack([group(.8), star()]), 'livox_frame', 12.)
        self.assertAlmostEqual(result['distance_m'], .8)
        self.assertLess(result['strength'], 1.)
        result = engine.process(np.vstack([group(.8), group(.1)]), 'livox_frame', 12.1)
        self.assertEqual(result['state'], 'NEAR_FIELD_ALERT')
        self.assertEqual(result['strength'], 1.)


if __name__ == '__main__':
    unittest.main()
