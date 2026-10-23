"""核对静止条件的含义：固定重复近点应排除，但不能屏蔽任意新近点。"""
import unittest
import numpy as np
from rolling_near import RollingNear
from rolling_distance import rolling_result
from distance_policy import DistancePolicy


class StationaryContract(unittest.TestCase):
    def test_all_fixed_near_points_removed_at_ten_seconds(self):
        rng=np.random.RandomState(23)
        xyz=rng.uniform(-.25,.25,(200,3))
        r=RollingNear()
        for i in range(101):mask,detail=r.process(xyz,1+i*.1)
        self.assertTrue(detail['rolling_ready'])
        self.assertEqual(int(mask.sum()),len(xyz))

    def test_fixed_scene_distance_is_outside_half_meter(self):
        points=np.array([[.1,0,0],[.3,0,0],[.5,0,0],
                         [.8,0,0],[.8,.002,0],[.8,0,.002],[.8,.002,.002]])
        r,p=RollingNear(),DistancePolicy(backend='reference')
        for i in range(110):
            answer=rolling_result(p,r,points,'livox_frame',1+i*.1,0.,len(points))
        self.assertTrue(answer['valid']);self.assertGreater(answer['distance_m'],.5)
        self.assertEqual(answer['rolling_removed_points'],3)

    def test_no_outside_points_is_unknown_not_fabricated_distance(self):
        points=np.array([[.1,0,0],[.3,0,0]])
        r,p=RollingNear(),DistancePolicy(backend='reference')
        for i in range(110):
            answer=rolling_result(p,r,points,'livox_frame',1+i*.1,0.,len(points))
        self.assertEqual(answer['rolling_removed_points'],2)
        self.assertFalse(answer['valid']);self.assertIsNone(answer['distance_m'])

    def test_sensor_static_does_not_suppress_a_new_near_target(self):
        base=np.array([[.3,0,0],[.8,0,0],[.8,.002,0],[.8,0,.002],[.8,.002,.002]])
        r,p=RollingNear(),DistancePolicy(backend='reference')
        for i in range(110):
            rolling_result(p,r,base,'livox_frame',1+i*.1,0.,len(base))
        points=np.vstack([base,[.1,0,0]])
        answer=rolling_result(p,r,points,'livox_frame',12.,0.,len(points))
        self.assertEqual(answer['state'],'NEAR_FIELD_ALERT')
        self.assertAlmostEqual(answer['distance_m'],.1)
        self.assertEqual(answer['strength'],1.)

    def test_repeated_sparse_hits_not_automatically_stable(self):
        r=RollingNear()
        for i in range(110):
            points=np.array([[.1,0,0]]) if i%10==0 else np.empty((0,3))
            r.process(points,1+i*.1)
        mask,detail=r.process([[.1,0,0]],12.)
        self.assertTrue(detail['rolling_ready']);self.assertFalse(mask[0])


if __name__=='__main__':unittest.main()
