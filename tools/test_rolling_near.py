import unittest
import numpy as np
from rolling_near import RollingNear


class RollingTests(unittest.TestCase):
    def feed(self, r, points, start=1, count=105):
        for i in range(count):
            result = r.process(points, start+i*.1)
        return result

    def test_warmup(self):
        removed, d = self.feed(RollingNear(), [[.1,0,0]], count=90)
        self.assertFalse(d['rolling_ready']); self.assertFalse(removed.any())

    def test_static_and_far(self):
        p=np.array([[.1,0,0],[.6,0,0],[0,0,0],[np.nan,0,0]])
        old=p.copy(); removed,d=self.feed(RollingNear(),p)
        self.assertEqual(removed.tolist(),[True,False,False,False])
        np.testing.assert_equal(p,old)

    def test_new_object_not_immediately_removed(self):
        r=RollingNear();self.feed(r,[[.1,0,0]])
        removed,_=r.process([[.1,0,0],[.3,0,0]],11.5)
        self.assertEqual(removed.tolist(),[True,False])

    def test_gap_and_reverse_reset(self):
        for next_time in (12.,1.):
            r=RollingNear();self.feed(r,[[.1,0,0]])
            removed,d=r.process([[.1,0,0]],next_time)
            self.assertFalse(d['rolling_ready']);self.assertFalse(removed.any())

    def test_expiry(self):
        r=RollingNear();self.feed(r,[[.1,0,0]])
        self.feed(r,[[.3,0,0]],start=11.5,count=105)
        removed,d=r.process([[.1,0,0],[.3,0,0]],22.)
        self.assertEqual(removed.tolist(),[False,True])

    def test_static_external_also_removed(self):
        # 明确的反例：算法不知道这是一面静止外墙，不能宣称自身识别成功。
        removed,d=self.feed(RollingNear(),[[.4,0,0]])
        self.assertTrue(removed[0])
        self.assertEqual(d['rolling_identity'],'unconfirmed_temporal_stability_only')

    def test_duplicate_points_are_one_vote(self):
        r=RollingNear()
        for i in range(105):
            p=[[.1,0,0]]*500 if i<10 else [[.3,0,0]]
            r.process(p,1+i*.1)
        removed,_=r.process([[.1,0,0]],11.5)
        self.assertFalse(removed.any())

    def test_wrong_frame(self):
        r=RollingNear();self.feed(r,[[.1,0,0]])
        with self.assertRaises(ValueError):r.process([[.1,0,0]],11.5,'map')
        self.assertFalse(r.frames)

    def test_intermittent_nonrepetitive_hits(self):
        r=RollingNear()
        for i in range(110):
            r.process([[.1,0,0]] if i%3==0 else [],1+i*.1) if i%3==0 else r.process(np.empty((0,3)),1+i*.1)
        removed,d=r.process([[.1,0,0]],12.)
        self.assertTrue(d['rolling_ready']);self.assertTrue(removed[0])

    def test_boundary_only_within_radius(self):
        removed,_=self.feed(RollingNear(),[[.5,0,0],[.50001,0,0]])
        self.assertEqual(removed.tolist(),[True,False])

    def test_lower_rate_does_not_toggle_warmup(self):
        r=RollingNear()
        for i in range(160):
            removed,d=r.process([[.1,0,0]],1+i*.14)
            if i*.14>=10.:
                self.assertTrue(d['rolling_ready'])
                self.assertTrue(removed[0])

    def test_small_boundary_tolerance_and_new_farther_point(self):
        r=RollingNear();self.feed(r,[[.119,0,0]])
        removed,_=r.process([[.121,0,0],[.15,0,0],[.3,0,0]],11.5)
        self.assertEqual(removed.tolist(),[True,False,False])

    def test_external_inside_tolerance_can_be_suppressed(self):
        r=RollingNear();self.feed(r,[[.119,0,0]])
        removed,_=r.process([[.125,0,0]],11.5)
        self.assertTrue(removed[0])


if __name__=='__main__':unittest.main()
