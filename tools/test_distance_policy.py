"""人工点测试。不能代替身体标注、稀疏物体真值或穿戴试验。"""
import copy
import json
import os
import unittest
from types import SimpleNamespace
import numpy as np
from distance_policy import DistancePolicy, Settings, distance_strength, empty_profile, supported_points
from distance_node import cloud_result


def cluster(distance):
    return np.array([[distance, 0., 0.], [distance, .01, 0.],
                     [distance, 0., .01], [distance, .01, .01]])


def reviewed():
    p = empty_profile('livox_frame')
    p.update(self_voxels=[[10,0,0]], self_identity_confirmed=True,
             confirmation_basis='reviewer_declaration_only', review_basis='synthetic fixture only')
    return p


class PolicyTests(unittest.TestCase):
    def policy(self, **kw):
        return DistancePolicy(backend=os.environ.get('DISTANCE_TEST_BACKEND', 'reference'), **kw)

    def run_cloud(self, xyz, **kw):
        return self.policy().process(xyz, 'livox_frame', 100., now=0., **kw)

    def test_01_curve_anchors(self):
        for d, expected in [(3,0),(2,0),(1.5,.25),(1,.5),(.75,.75),(.5,1),(.3,1)]:
            self.assertAlmostEqual(distance_strength(d,Settings()),expected)

    def test_02_monotonic_curve(self):
        values=[distance_strength(float(d),Settings()) for d in np.linspace(0,6,301)]
        self.assertTrue(all(0<=v<=1 for v in values))
        self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))

    def test_03_invalid_curve(self):
        for d in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): distance_strength(d,Settings())

    def test_04_parameter_checks(self):
        for args in [dict(min_neighbors=True),dict(release_frames=0),dict(stale_timeout=0),dict(hysteresis=-1),
                     dict(support_radius=0),dict(expected_frame=''),dict(strong_distance=1.5),dict(max_range=float('nan'))]:
            with self.assertRaises(ValueError): Settings(**args)

    def test_05_dense_cluster(self):
        r=self.run_cloud(cluster(1.5))
        self.assertTrue(r['valid']); self.assertAlmostEqual(r['strength'],.25)
        self.assertEqual(r['supported_points'],4)

    def test_06_four_bands(self):
        for d,l in [(.4,3),(.8,2),(1.5,1),(3,0)]:
            self.assertEqual(self.run_cloud(cluster(d))['level'],l)

    def test_07_sparse_close_not_quiet(self):
        r=self.run_cloud(np.vstack([cluster(1), [.3,0,0]]))
        self.assertEqual(r['state'],'UNSUPPORTED_CLOSER_RETURN')
        self.assertIsNone(r['strength']); self.assertEqual(r['candidate_strength'],.5)

    def test_08_near_field_unknown(self):
        r=self.run_cloud(np.vstack([cluster(1), [.1,0,0]]))
        self.assertEqual(r['state'],'NEAR_FIELD_UNCERTAIN'); self.assertIsNone(r['strength'])

    def test_09_sparse_far_does_not_override_nearer_cluster(self):
        r=self.run_cloud(np.vstack([cluster(1), [3,0,0]]))
        self.assertTrue(r['valid']); self.assertEqual(r['strength'],.5)

    def test_10_empty(self):
        self.assertEqual(self.run_cloud(np.empty((0,3)))['state'],'NO_REMAINING_OBSERVATIONS')

    def test_11_invalid_separate(self):
        r=self.run_cloud(np.vstack([cluster(1),[0,0,0],[np.nan,0,0],[np.inf,0,0]]))
        self.assertEqual(r['invalid_points'],3); self.assertEqual(r['self_removed_points'],0)

    def test_12_far_only(self):
        r=self.run_cloud(np.array([[6.,0.,0.]]))
        self.assertEqual(r['state'],'NO_NEAR_CANDIDATE'); self.assertEqual(r['strength'],0)
        self.assertIsNone(r['distance_m'])

    def test_13_no_sparse_support(self):
        r=self.run_cloud(np.array([[1.,0.,0.]]))
        self.assertFalse(r['valid']); self.assertIsNone(r['strength'])

    def test_14_frame_reject(self):
        r=self.policy().process(cluster(1),'map',1.)
        self.assertEqual(r['state'],'FRAME_MISMATCH')

    def test_15_count_reject(self):
        self.assertEqual(self.run_cloud(cluster(1),declared_count=2)['state'],'POINT_COUNT_MISMATCH')

    def test_16_timestamp_reject(self):
        for stamp in (0.,float('nan'),True):
            self.assertEqual(self.policy().process(cluster(1),'livox_frame',stamp)['state'],'INVALID_TIMESTAMP')

    def test_17_header_age(self):
        for age in (.6,-.2,float('nan')):
            self.assertEqual(self.run_cloud(cluster(1),source_age_s=age)['state'],'STALE_OR_FUTURE_HEADER')

    def test_18_duplicate_and_backward(self):
        p=self.policy(); p.process(cluster(1),'livox_frame',10.)
        for stamp in (10.,9.):
            self.assertEqual(p.process(cluster(1),'livox_frame',stamp)['state'],'TIMESTAMP_NONINCREASING')
        self.assertTrue(p.process(cluster(1),'livox_frame',11.)['valid'])

    def test_19_watchdog(self):
        p=self.policy()
        self.assertEqual(p.watchdog(0.)['state'],'STALE_OR_NO_DATA')
        p.process(cluster(1),'livox_frame',1.,now=2.)
        self.assertIsNone(p.watchdog(2.4))
        self.assertIsNone(p.watchdog(2.6)['strength'])

    def test_20_release_latch(self):
        p=self.policy()
        levels=[p.process(cluster(d),'livox_frame',float(i+1))['level'] for i,d in enumerate([.4,.55,.7,.7])]
        self.assertEqual(levels,[3,3,3,2])

    def test_21_strength_is_current_not_latched(self):
        p=self.policy(); p.process(cluster(.4),'livox_frame',1.)
        r=p.process(cluster(.55),'livox_frame',2.)
        self.assertEqual(r['level'],3); self.assertAlmostEqual(r['strength'],.95)

    def test_22_mask_requires_profile(self):
        with self.assertRaises(ValueError): self.policy(apply_self_mask=True)
        with self.assertRaises(ValueError): self.policy(apply_self_mask='true')

    def test_23_unreviewed_still_kept(self):
        r=self.policy(profile=reviewed()).process(np.vstack([cluster(1),[.205,.005,.005]]),'livox_frame',1.)
        self.assertEqual(r['self_removed_points'],0); self.assertFalse(r['valid'])

    def test_24_reviewed_local_removal(self):
        r=self.policy(profile=reviewed(),apply_self_mask=True).process(np.vstack([cluster(1),[.205,.005,.005]]),'livox_frame',1.)
        self.assertEqual(r['self_removed_points'],1); self.assertEqual(r['strength'],.5)
        self.assertEqual(r['masked_space_occupancy'],'unknown_not_free')

    def test_25_same_voxel_external_also_removed(self):
        r=self.policy(profile=reviewed(),apply_self_mask=True).process(np.array([[.206,.006,.006]]),'livox_frame',1.)
        self.assertEqual(r['self_removed_points'],1); self.assertIsNone(r['strength'])

    def test_26_no_claim_no_profile(self):
        r=self.run_cloud(cluster(1))
        self.assertFalse(r['self_mask_enabled']); self.assertEqual(r['identity_basis'],'not_applied')

    def test_27_mask_no_declaration_rejected(self):
        p=reviewed();p['self_identity_confirmed']=False
        with self.assertRaises(ValueError):self.policy(profile=p,apply_self_mask=True)

    def test_28_nonmutation(self):
        xyz=np.vstack([cluster(1),[.205,.005,.005]])
        before=xyz.copy(); p=reviewed(); prior=copy.deepcopy(p)
        self.policy(profile=p,apply_self_mask=True).process(xyz,'livox_frame',1.)
        np.testing.assert_array_equal(before,xyz);self.assertEqual(prior,p)

    def test_29_json_has_no_nonfinite_number(self):
        for xyz in (cluster(1),np.empty((0,3)),np.array([[np.nan,0,0]])):
            json.dumps(self.run_cloud(xyz),allow_nan=False)

    def test_30_reference_neighbor_semantics(self):
        xyz=np.array([[1.,0,0],[1.,0,0],[1.,0,0],[1.,0,0]])
        np.testing.assert_array_equal(supported_points(xyz,.08,3,'reference'),[True]*4)
        np.testing.assert_array_equal(supported_points(xyz[:3],.08,3,'reference'),[False]*3)

    def test_31_bad_shape(self):
        self.assertEqual(self.run_cloud(np.array([1.,0,0]))['state'],'INVALID_COORDINATE_ARRAY')

    def test_32_adapter(self):
        msg=SimpleNamespace(points=[SimpleNamespace(x=x,y=y,z=z) for x,y,z in cluster(1)],
                            point_num=4,header=SimpleNamespace(frame_id='livox_frame',stamp=SimpleNamespace(to_sec=lambda:10.)))
        r=cloud_result(self.policy(),msg,10.,0.)
        self.assertEqual(r['strength'],.5)
        self.assertEqual(cloud_result(self.policy(),msg,11.)['state'],'STALE_OR_FUTURE_HEADER')

    def test_33_backends_match(self):
        try:
            import scipy.spatial
        except ImportError:
            self.skipTest('SciPy absent locally; exact device backend checked on Jetson')
        xyz=np.random.default_rng(77).normal(size=(300,3))*.1+[1,0,0]
        np.testing.assert_array_equal(supported_points(xyz,.08,3,'scipy'),supported_points(xyz,.08,3,'reference'))


if __name__=='__main__': unittest.main()
