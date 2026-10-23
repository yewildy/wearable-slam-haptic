import json
import unittest
import numpy as np
from rolling_near import RollingNear
from rolling_distance import rolling_result
from distance_policy import DistancePolicy
from strength_view_model import checked_report, ViewModel


def group(x):
    return [[x,0,0],[x,.002,0],[x,0,.002],[x,.002,.002]]


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.r=RollingNear();self.p=DistancePolicy(backend='reference')

    def send(self, points, stamp, age=.15, count=None):
        points=np.asarray(points,dtype=float).reshape(-1,3)
        return rolling_result(self.p,self.r,points,'livox_frame',stamp,age,
                              len(points) if count is None else count,stamp)

    def warm(self):
        for i in range(110):result=self.send(group(.1)+group(1.5),1+i*.1)
        return result

    def test_static_filter_then_strength(self):
        r=self.warm()
        self.assertTrue(r['valid']);self.assertEqual(r['rolling_removed_points'],4)
        self.assertAlmostEqual(r['distance_m'],1.5);self.assertAlmostEqual(r['strength'],.25)
        self.assertEqual(r['self_removed_points'],0)
        self.assertEqual(r['rolling_identity'],'unconfirmed_temporal_stability_only')
        checked_report(json.dumps(r))

    def test_moving_object_new_position_alerts_immediately(self):
        self.warm()
        r=self.send(group(.1)+group(.3)+group(1.5),12.)
        self.assertEqual(r['rolling_removed_points'],4)
        self.assertTrue(r['valid']);self.assertAlmostEqual(r['distance_m'],.3)
        self.assertEqual(r['strength'],1.)

    def test_unmatched_very_near_return_warns_not_discarded(self):
        self.warm()
        r=self.send(group(.1)+[[.15,0,0]]+group(1.5),12.)
        self.assertEqual(r['state'],'NEAR_FIELD_ALERT');self.assertEqual(r['strength'],1.)
        self.assertTrue(r['near_distance_is_uncertain']);checked_report(json.dumps(r))

    def test_gap_reenters_warmup(self):
        self.warm();r=self.send(group(.1)+group(1.5),13.)
        self.assertFalse(r['valid']);self.assertEqual(r['state'],'ROLLING_WARMUP')

    def test_timestamp_failure_clears_history(self):
        self.warm();r=self.send(group(.1),12.,age=1.)
        self.assertFalse(r['valid']);self.assertEqual(r['state'],'STALE_OR_FUTURE_HEADER')
        self.assertFalse(self.r.frames)

    def test_wrong_count_clears_history(self):
        self.warm();r=self.send(group(.1),12.,count=10)
        self.assertEqual(r['state'],'POINT_COUNT_MISMATCH');self.assertFalse(self.r.frames)

    def test_view_stale_clears_experimental_diagnostics(self):
        v=ViewModel();v.receive(json.dumps(self.warm()),now=1.)
        r=v.snapshot(now=2.)
        self.assertIsNone(r['candidate_strength']);self.assertIsNone(r['rolling_removed_points'])

    def test_cannot_filter_beyond_radius(self):
        for i in range(110):r=self.send(group(.51),1+i*.1)
        self.assertEqual(r['rolling_removed_points'],0);self.assertTrue(r['valid'])

    def test_stationary_external_can_be_suppressed_counterexample(self):
        for i in range(110):r=self.send(group(.4)+group(1.5),1+i*.1)
        self.assertEqual(r['rolling_removed_points'],4)
        self.assertAlmostEqual(r['distance_m'],1.5)
        self.assertEqual(r['masked_space_occupancy'],'unknown_not_free')


if __name__=='__main__':unittest.main()
