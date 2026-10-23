import importlib.util
import unittest
import math
import numpy as np
from motion_tracking import MotionSettings,MotionTracker,checked_pose,clusters
from motion_pose import fixed_extrinsic
from motion_pipeline import MotionEngine
from distance_policy import distance_strength,Settings

SCIPY=importlib.util.find_spec('scipy') is not None


def pose(stamp,x=0.,angle=0.):
    c,s=math.cos(angle),math.sin(angle)
    m=np.array([[c,-s,0,x],[s,c,0,0],[0,0,1,0],[0,0,0,1.]])
    return dict(matrix=m,frame='world',stamp=stamp)


def det(d,center=None,extent=.02):
    p=np.array([d,0,0] if center is None else center,dtype=float)
    return dict(center=p,closest=p,distance=d,points=4,extent=extent)


def curve(d):return distance_strength(d,Settings())


class ValidationTests(unittest.TestCase):
    def test_parameters_reject_invalid(self):
        for values in ({'max_tracks':True},{'decay_tau_s':0},{'speed_gain':float('nan')},
                       {'stopped_floor':1.1},{'history':2},{'voxel_m':.2}):
            with self.assertRaises(ValueError):MotionSettings(**values)

    def test_pose_validation(self):
        for change in ('old','reflection','bad_bottom','missing_frame'):
            p=pose(1.)
            if change=='old':p['stamp']=.5
            if change=='reflection':p['matrix'][0,0]=-1.
            if change=='bad_bottom':p['matrix'][3,0]=1.
            if change=='missing_frame':p['frame']=''
            with self.assertRaises(ValueError):checked_pose(p,1.,MotionSettings())

    def test_fixed_extrinsic_composition(self):
        e=fixed_extrinsic([0,-1,0,1,0,0,0,0,1],[.1,.2,.3])
        np.testing.assert_allclose((e@np.array([1,0,0,1]))[:3],[.1,1.2,.3])
        with self.assertRaises(ValueError):fixed_extrinsic(np.eye(3),[0,0,0],True)


@unittest.skipUnless(SCIPY,'Device SciPy required for proven association/clustering')
class TrackerTests(unittest.TestCase):
    def step(self,tracker,d,stamp,with_pose=True,**kwargs):
        return tracker.process([det(d,**kwargs)],stamp,curve,pose(stamp) if with_pose else None)['tracks'][0]

    def test_fast_approach_stronger_at_same_distance(self):
        answers=[]
        for speed in (.1,1.):
            tracker=MotionTracker()
            for i in range(11):r=self.step(tracker,1.5+speed*(1-i*.1),1+i*.1)
            self.assertAlmostEqual(r['closing_speed_mps'],speed,places=5)
            answers.append(r['strength'])
        self.assertGreater(answers[1],answers[0])

    def test_stopping_decays_but_not_to_zero(self):
        t=MotionTracker()
        for i in range(11):r=self.step(t,1.8-i*.1,1+i*.1)
        peak=r['strength']
        for i in range(1,101):r=self.step(t,.8,2+i*.1)
        self.assertLess(r['strength'],peak)
        self.assertGreaterEqual(r['strength'],curve(.8)*.4)
        self.assertLess(r['dwell_factor'],.5)

    def test_tiny_velocity_noise_does_not_disable_decay(self):
        t=MotionTracker()
        for i in range(110):r=self.step(t,.8-1e-9*i*.1,1+i*.1)
        self.assertLess(r['dwell_factor'],.5)
        self.assertAlmostEqual(r['strength'],r['base_strength']*r['dwell_factor'],places=6)

    def test_close_target_has_habituation_floor(self):
        t=MotionTracker()
        for i in range(110):r=self.step(t,.3,1+i*.1)
        self.assertGreaterEqual(r['strength'],.6)

    def test_no_pose_does_not_enable_habituation(self):
        t=MotionTracker()
        for i in range(110):r=self.step(t,.8,1+i*.1,False)
        self.assertEqual(r['dwell_factor'],1.)
        self.assertAlmostEqual(r['strength'],curve(.8))

    def test_motion_restarts_after_habituation(self):
        t=MotionTracker()
        for i in range(100):r=self.step(t,1.,1+i*.1)
        low=r['strength']
        for i in range(1,6):r=self.step(t,1.-i*.06,10.9+i*.1)
        self.assertGreater(r['strength'],low)
        self.assertEqual(r['dwell_factor'],1.)

    def test_target_jump_does_not_create_huge_speed(self):
        t=MotionTracker()
        for i in range(8):r=self.step(t,3.,1+i*.1)
        old=r['track_id'];new=self.step(t,.7,1.8)
        self.assertNotEqual(old,new['track_id']);self.assertIsNone(new['closing_speed_mps'])

    def test_timestamp_gap_and_reverse_reset(self):
        for stamp in (4.,1.):
            t=MotionTracker()
            for i in range(8):self.step(t,1.,1+i*.1)
            r=self.step(t,1.,stamp)
            self.assertIsNone(r['closing_speed_mps'])

    def test_ego_translation_keeps_static_world_target(self):
        t=MotionTracker();identity=None
        for i in range(10):
            stamp=1+i*.1;x=i*.01
            r=t.process([det(1-x)],stamp,curve,pose(stamp,x))['tracks'][0]
            identity=r['track_id'] if identity is None else identity
            self.assertEqual(r['track_id'],identity)
        self.assertAlmostEqual(r['coordinate_speed_mps'],0.,places=6)
        self.assertAlmostEqual(r['closing_speed_mps'],.1,places=6)

    def test_ego_rotation_keeps_static_world_target(self):
        t=MotionTracker();identity=None
        for i in range(12):
            stamp=1+i*.1;a=i*.1
            local=[math.cos(a),-math.sin(a),0]
            r=t.process([det(1.,local)],stamp,curve,pose(stamp,angle=a))['tracks'][0]
            identity=r['track_id'] if identity is None else identity
            self.assertEqual(identity,r['track_id'])
        self.assertAlmostEqual(r['coordinate_speed_mps'],0.,places=6)

    def test_previous_ignored_background_has_no_novelty_boost(self):
        t=MotionTracker();t.process([],1.,curve,pose(1.),np.array([[.4,0,0]]))
        r=t.process([det(.2)],1.1,curve,pose(1.1,.2))['tracks'][0]
        self.assertTrue(r['familiar_background']);self.assertEqual(r['novelty_boost'],0.)
        self.assertEqual(r['strength'],1.)  # 旧背景也不能抹掉真实近距离警报。

    def test_world_memory_expiry_and_limit(self):
        t=MotionTracker(MotionSettings(max_memory_cells=4))
        t.process([],1.,curve,pose(1.),np.array([[x,0,0] for x in np.arange(10)*.1]))
        self.assertLessEqual(len(t.world_memory),4)
        # 模拟正常时间推进，不用长断流重置代替TTL检查。
        for i in range(1,220):t.process([],1+i*.1,curve,pose(1+i*.1))
        self.assertFalse(t.world_memory)

    def test_pose_mode_change_resets_velocity(self):
        t=MotionTracker()
        for i in range(8):self.step(t,1.,1+i*.1)
        r=self.step(t,1.,1.8,False)
        self.assertIsNone(r['closing_speed_mps'])

    def test_pose_jump_clears_old_memory(self):
        t=MotionTracker();t.process([],1.,curve,pose(1.),np.array([[.4,0,0]]))
        r=t.process([det(1.)],1.1,curve,pose(1.1,2.))
        self.assertEqual(r['pose_state'],'POSE_JUMP_TRACKS_RESET')
        self.assertEqual(r['background_memory_cells'],0)

    def test_ambiguous_association_does_not_estimate_speed(self):
        t=MotionTracker();t.process([det(1.)],1.,curve,pose(1.))
        r=t.process([det(.98),det(1.02)],1.1,curve,pose(1.1))
        self.assertGreater(r['association_uncertain_count'],0)
        self.assertTrue(all(q['closing_speed_mps'] is None for q in r['tracks']))

    def test_wide_wall_centroid_is_not_object_speed(self):
        t=MotionTracker()
        for i in range(8):r=self.step(t,1.-i*.01,1+i*.1,extent=2.)
        self.assertIsNone(r['closing_speed_mps']);self.assertTrue(r['wide_surface_velocity_rejected'])

    def test_fast_far_target_can_dominate_nearer_static_target(self):
        t=MotionTracker()
        for i in range(10):
            ds=[det(.8,[.8,0,0]),det(3.2-i*.2,[0,3.2-i*.2,0])]
            r=t.process(ds,1+i*.1,curve,pose(1+i*.1))
        selected=max(r['tracks'],key=lambda q:q['strength'])
        self.assertGreater(selected['distance_m'],.8)

    def test_speed_alone_does_not_always_override_nearer_target(self):
        t=MotionTracker()
        for i in range(10):
            ds=[det(.8,[.8,0,0]),det(2.5-i*.1,[0,2.5-i*.1,0])]
            r=t.process(ds,1+i*.1,curve,pose(1+i*.1))
        self.assertAlmostEqual(max(r['tracks'],key=lambda q:q['strength'])['distance_m'],.8)

    def test_cluster_limits_and_minimum_support(self):
        p=np.array([[.8,0,0],[.8,.002,0],[.8,0,.002],[.8,.002,.002]])
        self.assertEqual(len(clusters(p,MotionSettings())),1)
        self.assertEqual(len(clusters(p[:3],MotionSettings())),0)
        with self.assertRaises(ValueError):clusters(np.arange(12).reshape(-1,3),MotionSettings(max_voxels=1))
        with self.assertRaises(ValueError):clusters(np.array([[0,0,0],[.1,0,0],[.2,0,0]]),MotionSettings(max_pairs=1,cluster_radius_m=.3))

    def test_short_occlusion_preserves_id_without_outputting_unseen_track(self):
        t=MotionTracker()
        for i in range(8):r=self.step(t,1.,1+i*.1)
        identity=r['track_id']
        missing=t.process([],1.8,curve,pose(1.8))
        self.assertEqual(missing['tracks'],[])
        r=self.step(t,1.,1.9)
        self.assertEqual(r['track_id'],identity)


@unittest.skipUnless(SCIPY,'Device SciPy required')
class PipelineTests(unittest.TestCase):
    def test_second_stage_unknown_and_near_alert_are_preserved(self):
        engine=MotionEngine()
        far=np.array([[1.5,0,0],[1.5,.002,0],[1.5,0,.002],[1.5,.002,.002]])
        for i in range(110):r=engine.process(far,'livox_frame',1+i*.1,pose=pose(1+i*.1))
        self.assertTrue(r['valid'])
        points=np.vstack([far,[[.1,0,0],[.1,.01,0],[.1,0,.01],[.1,.01,.01]]])
        r=engine.process(points,'livox_frame',12.,pose=pose(12.))
        self.assertEqual(r['state'],'NEAR_FIELD_ALERT');self.assertEqual(r['strength'],1.)
        self.assertEqual(r['motion_override'],'UNTRUSTED_NEAR_FIELD_NOT_DECAYED')
        bad=engine.process(far,'livox_frame',12.1,age=2.,pose=pose(12.1))
        self.assertFalse(bad['valid']);self.assertIsNone(bad['strength'])

    def test_invalid_pose_falls_back_to_distance_not_fabricated_world(self):
        e=MotionEngine();p=np.array([[1.5,0,0],[1.5,.002,0],[1.5,0,.002],[1.5,.002,.002]])
        for i in range(110):r=e.process(p,'livox_frame',1+i*.1,pose=pose(1+i*.1-.2))
        self.assertTrue(r['valid']);self.assertFalse(r['motion_valid'])
        self.assertEqual(r['pose_state'],'MOTION_REJECTED_DISTANCE_FALLBACK')

    def test_clock_rollback_requires_new_warmup_and_recovers(self):
        e=MotionEngine();p=np.array([[1.5,0,0],[1.5,.002,0],[1.5,0,.002],[1.5,.002,.002]])
        for i in range(110):e.process(p,'livox_frame',1+i*.1,pose=pose(1+i*.1))
        r=e.process(p,'livox_frame',1.,pose=pose(1.))
        self.assertFalse(r['valid']);self.assertEqual(r['state'],'TIMESTAMP_NONINCREASING')
        for i in range(111):r=e.process(p,'livox_frame',1.1+i*.1,pose=pose(1.1+i*.1))
        self.assertTrue(r['valid'])

    def test_clustering_budget_failure_retains_distance(self):
        e=MotionEngine(MotionSettings(max_voxels=1))
        p=np.array([[.8,0,0],[.8,.002,0],[.8,0,.002],[.8,.002,.002],
                    [2.,0,0],[2.,.002,0],[2.,0,.002],[2.,.002,.002]])
        original=p.copy()
        for i in range(110):r=e.process(p,'livox_frame',1+i*.1,pose=pose(1+i*.1))
        self.assertTrue(r['valid']);self.assertFalse(r['motion_valid'])
        self.assertAlmostEqual(r['strength'],curve(.8))
        np.testing.assert_equal(p,original)

    def test_tracked_close_object_is_not_learned_away_after_stop(self):
        e=MotionEngine()
        def points(d):return np.array([[.1,0,0],[.1,.002,0],[.1,0,.002],[.1,.002,.002],
                                       [d,0,0],[d,.002,0],[d,0,.002],[d,.002,.002]])
        for i in range(110):e.process(points(1.),'livox_frame',1+i*.1,pose=pose(1+i*.1))
        for i in range(1,15):e.process(points(1.-i*.05),'livox_frame',11.9+i*.1,pose=pose(11.9+i*.1))
        protected=0
        for i in range(1,160):
            stamp=13.3+i*.1;r=e.process(points(.3),'livox_frame',stamp,pose=pose(stamp))
            protected=max(protected,r.get('rolling_protected_track_points',0))
        self.assertTrue(r['valid']);self.assertAlmostEqual(r['distance_m'],.3)
        self.assertGreaterEqual(r['strength'],.6);self.assertGreater(protected,0)

    def test_guard_does_not_unmask_neighboring_bracket_points(self):
        e=MotionEngine()
        for i in range(12):
            t=1+i*.1
            d=.8 if i<6 else .8-(i-5)*.1
            e.motion.process([det(d)],t,curve,pose(t))
        # 让最后几帧保持0.2m，得到可靠速度；曾在远处的目标中心不能保护10cm外的支架。
        for i in range(1,12):e.motion.process([det(.2)],2.1+i*.1,curve,pose(2.1+i*.1))
        guard=e._track_guard(pose(3.3),3.3)
        self.assertIsNotNone(guard)
        result=guard(np.array([[.1,0,0],[.2,0,0]]),np.array([True,True]))
        self.assertEqual(result.tolist(),[False,True])

    def test_pose_loss_after_tracking_reenters_unknown_warmup(self):
        e=MotionEngine();p=np.array([[1.5,0,0],[1.5,.002,0],[1.5,0,.002],[1.5,.002,.002]])
        for i in range(110):e.process(p,'livox_frame',1+i*.1,pose=pose(1+i*.1))
        r=e.process(p,'livox_frame',12.,pose=None)
        self.assertFalse(r['valid']);self.assertEqual(r['state'],'ROLLING_WARMUP')

    def test_fast_far_warning_has_consistent_state(self):
        e=MotionEngine()
        def p(d):return np.array([[d,0,0],[d,.002,0],[d,0,.002],[d,.002,.002]])
        for i in range(110):e.process(p(4.6),'livox_frame',1+i*.1,pose=pose(1+i*.1))
        for i in range(1,11):
            stamp=11.9+i*.1;r=e.process(p(4.6-i*.2),'livox_frame',stamp,pose=pose(stamp))
        self.assertGreater(r['distance_m'],2.)
        self.assertGreater(r['strength'],0.)
        self.assertEqual(r['state'],'DISTANCE_CANDIDATE');self.assertGreater(r['level'],0)


if __name__=='__main__':unittest.main()
