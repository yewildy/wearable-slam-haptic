import copy
import unittest

import numpy as np

from audit_mask_risk import audit, checked_boxes, checked_sample, public_summary, removed_by_boxes, synthetic_risks


def fixture():
    sample = {'xyz': np.array([[-.3, 0, 0], [-.31, 0, 0], [-.32, 0, 0], [-.45, 0, 0]], dtype=np.float32),
              'frame_id': np.arange(4), 'frame_stamps': np.arange(1., 5.),
              'frame_names': np.array(['fixture']*4)}
    profile = {'self_mask_enabled': False, 'self_identity_confirmed': False,
               'self_mask_frame': 'fixture', 'self_boxes': [{'min': [-.49, -.1, -.1], 'max': [-.2, .1, .1]}]}
    return sample, profile


class RiskTests(unittest.TestCase):
    def test_external_thin_object_can_be_fully_deleted(self):
        cases = {row['case']: row for row in synthetic_risks()}
        self.assertEqual(cases['external_thin_pole_inside_box']['removed_points'], 9)
        self.assertEqual(cases['external_thin_pole_crossing_box']['remaining_points'], 2)

    def test_exact_faces_and_next_float_outside(self):
        cases = {row['case']: row for row in synthetic_risks()}
        self.assertEqual(cases['six_exact_faces']['removed_points'], 6)
        self.assertEqual(cases['six_one_float32_step_outside']['removed_points'], 0)

    def test_overlapping_boxes_do_not_double_count(self):
        sample, profile = fixture()
        boxes = profile['self_boxes']*2
        self.assertEqual(int(removed_by_boxes(sample['xyz'], boxes).sum()), 4)

    def test_later_frames_are_not_used_in_fit(self):
        report = audit(*fixture())
        self.assertEqual(report['prefix_fit']['removed_near_points'], 2)
        self.assertEqual(report['same_scene_heldout']['remaining_near_points'], 1)
        self.assertFalse(report['temporal_split']['independent_scene'])

    def test_disabled_candidate_not_mutated(self):
        sample, profile = fixture()
        before = copy.deepcopy(profile)
        audit(sample, profile)
        self.assertEqual(profile, before)

    def test_frame_change_rejected(self):
        sample, profile = fixture()
        sample['frame_names'][3] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'frames must match'):
            audit(sample, profile)

    def test_legacy_without_frames_rejected(self):
        sample, profile = fixture()
        del sample['frame_names']
        with self.assertRaisesRegex(ValueError, 'current capture'):
            audit(sample, profile)

    def test_invalid_points_rejected(self):
        for xyz in ((0, 0, 0), (float('nan'), 0, 0), (.8, 0, 0)):
            sample, profile = fixture()
            sample['xyz'][0] = xyz
            with self.assertRaises(ValueError):
                audit(sample, profile)

    def test_bad_indices_rejected(self):
        for value in (-1, 4):
            sample, profile = fixture()
            sample['frame_id'][0] = value
            with self.assertRaises(ValueError):
                audit(sample, profile)

    def test_duplicate_timestamp_rejected(self):
        sample, profile = fixture()
        sample['frame_stamps'][1] = 1
        with self.assertRaises(ValueError):
            audit(sample, profile)

    def test_no_near_points_in_partition_rejected(self):
        sample, profile = fixture()
        sample['frame_id'][:] = 0
        with self.assertRaisesRegex(ValueError, 'Both temporal'):
            audit(sample, profile)

    def test_invalid_bounds_rejected(self):
        sample, profile = fixture()
        for value in (True, float('inf'), 1001):
            bad = copy.deepcopy(profile)
            bad['self_boxes'][0]['min'][0] = value
            with self.assertRaises(ValueError):
                audit(sample, bad)
        with self.assertRaises(ValueError):
            checked_boxes([{'min': [1, 0, 0], 'max': [1+1e-10, 1, 1]}])

    def test_profile_must_remain_candidate_disabled(self):
        for key in ('self_mask_enabled', 'self_identity_confirmed'):
            sample, profile = fixture()
            profile[key] = True
            with self.assertRaises(ValueError):
                audit(sample, profile)

    def test_empty_frame_is_counted_without_claiming_free_space(self):
        sample, profile = fixture()
        sample['frame_id'] = np.array([0, 0, 2, 2])
        report = audit(sample, profile)
        self.assertEqual(report['prefix_fit']['frames_without_near_points'], 1)
        self.assertEqual(report['same_scene_heldout']['frames_without_near_points'], 1)

    def test_public_report_omits_scene_geometry(self):
        report = public_summary(audit(*fixture()))
        self.assertNotIn('prefix_fitted_box', report)
        self.assertEqual(report['mask_provenance'], 'candidate_unconfirmed_disabled_input')
        self.assertIn('unknown occupancy', ' '.join(report['limitations']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
