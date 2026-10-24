"""复用查询必须与完整组检查一致，不能把上帧邻居当作本帧证据。"""
import unittest
from unittest.mock import patch
import numpy as np
from distance_policy import Settings
from motion_support import SupportedDistancePolicy


class QueryReuseTests(unittest.TestCase):
    def setUp(self):
        try:
            import scipy
        except ImportError:
            self.skipTest('SciPy not installed')

    def test_dense_cloud_with_isolated_return_builds_only_one_tree(self):
        from scipy.spatial import cKDTree
        rng = np.random.default_rng(42)
        points = np.vstack([rng.uniform(0, .02, (100, 3))+[1., 0, 0], [[.1, 0, 0]]])
        with patch('scipy.spatial.cKDTree', wraps=cKDTree) as constructor:
            report = SupportedDistancePolicy().process(points, 'livox_frame', 1.)
        self.assertEqual(constructor.call_count, 1)
        self.assertEqual(report['isolated_removed_points'], 1)

    def test_randomized_pipeline_matches_reference(self):
        rng = np.random.default_rng(20261024)
        for minimum_neighbors in (1, 3, 5):
            settings = Settings(min_neighbors=minimum_neighbors)
            for count in (0, 1, 3, 4, 20, 200):
                for _ in range(3):
                    points = rng.uniform(-.2, .2, (count, 3))+[.3, 0, 0]
                    fast = SupportedDistancePolicy(settings)
                    reference = SupportedDistancePolicy(settings, backend='reference')
                    actual = fast.process(points, 'livox_frame', 1., now=1.)
                    expected = reference.process(points, 'livox_frame', 1., now=1.)
                    self.assertEqual(actual, expected)
                    np.testing.assert_array_equal(fast.filtered_points, reference.filtered_points)

    def test_pending_tail_connected_to_certified_group_survives(self):
        points = np.array([[.3, 0, 0], [.31, 0, 0], [.32, 0, 0], [.33, 0, 0],
                           [.40, 0, 0], [.47, 0, 0]])
        policy = SupportedDistancePolicy()
        expected = policy._group_mask(points)
        actual = policy._group_mask(points, np.array([True]*4+[False]*2))
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(actual.all())

    def test_no_cross_frame_certification(self):
        policy = SupportedDistancePolicy()
        dense = np.tile([.1, 0, 0], (8, 1))
        self.assertEqual(policy.process(dense, 'livox_frame', 1.)['isolated_removed_points'], 0)
        star = np.array([[.1, 0, 0], [.1, .07, 0], [.1, -.07, 0], [.1, 0, .07]])
        result = policy.process(star, 'livox_frame', 2.)
        self.assertEqual(result['state'], 'NO_REMAINING_OBSERVATIONS')
        self.assertEqual(result['postfilter_small_group_removed_points'], 1)


if __name__ == '__main__':
    unittest.main()
