import importlib.util,unittest
import numpy as np
from motion_background import BackgroundContinuity
from motion_pipeline import MotionEngine

SCIPY=importlib.util.find_spec('scipy') is not None


def checked(x=0.,angle=0.):
    m=np.eye(4);m[0,3]=x
    c,s=np.cos(angle),np.sin(angle);m[:2,:2]=[[c,-s],[s,c]]
    return m,'world'


@unittest.skipUnless(SCIPY,'Device SciPy required')
class BackgroundTests(unittest.TestCase):
    def seed(self):
        b=BackgroundContinuity();b.apply(np.array([[.4,0,0]]),[True],checked(),1.);return b

    def test_two_centimeter_ego_move_continues_old_ignored_point(self):
        b=self.seed();mask,d=b.apply(np.array([[.38,0,0]]),[False],checked(.02),1.1)
        self.assertTrue(mask[0]);self.assertEqual(d['rolling_continuity_removed_points'],1)

    def test_significantly_closer_still_alerts(self):
        b=self.seed();mask,d=b.apply(np.array([[.36,0,0]]),[False],checked(.04),1.1)
        self.assertFalse(mask[0])

    def test_new_object_not_old_world_point(self):
        b=self.seed();mask,_=b.apply(np.array([[.3,0,0]]),[False],checked(.02),1.1)
        self.assertFalse(mask[0])

    def test_cumulative_ego_motion_not_refreshed_by_continuation(self):
        b=self.seed()
        for i in range(1,9):mask,_=b.apply(np.array([[.4-i*.01,0,0]]),[False],checked(i*.01),1+i*.1)
        self.assertFalse(mask[0])

    def test_expiry_missing_pose_rotation_and_radius(self):
        cases=[(np.array([[.4,0,0]]),checked(),12.),
               (np.array([[.4,0,0]]),None,1.1),
               (np.array([[.4*np.cos(.2),-.4*np.sin(.2),0]]),checked(angle=.2),1.1),
               (np.array([[.51,0,0]]),checked(-.11),1.1)]
        for xyz,p,t in cases:
            mask,_=self.seed().apply(xyz,[False],p,t);self.assertFalse(mask[0])

    def test_pipeline_small_self_move_does_not_realert_old_background(self):
        e=MotionEngine()
        def group(x):return [[x,0,0],[x,.002,0],[x,0,.002],[x,.002,.002]]
        for i in range(110):
            t=1+i*.1;e.process(np.array(group(.1)+group(.4)+group(1.5)), 'livox_frame',t,
                              pose=dict(matrix=checked()[0],frame='world',stamp=t))
        r=e.process(np.array(group(.1)+group(.38)+group(1.48)), 'livox_frame',12.,
                    pose=dict(matrix=checked(.02)[0],frame='world',stamp=12.))
        self.assertTrue(r['valid']);self.assertGreater(r['distance_m'],.5)
        self.assertGreater(r['rolling_continuity_removed_points'],0)
        self.assertLess(r['strength'],1.)
