"""检查审核来源、默认保留和局部体素过滤边界。人工标签不当作物理真值。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from review_self_mask import checked_candidate, make_profile, preview, make_page, digest


def candidate():
    return {'schema_version': 1, 'kind': 'initialization_voxel_candidates',
            'self_mask_enabled': False, 'self_identity_confirmed': False, 'mask_applied': False,
            'frame': 'sensor', 'parameters': {'voxel_m': .02, 'seconds': 10},
            'candidate_voxels': 2, 'frames_used': 101,
            'candidate_cells': [
                {'voxel_index': [10, 0, 0], 'center_m': [.21, .01, .01], 'frame_fraction': 1},
                {'voxel_index': [-11, 0, 0], 'center_m': [-.21, .01, .01], 'frame_fraction': 1}]}


def review():
    return {'schema_version': 1, 'candidate_sha256': 'test_sha', 'review_basis': 'synthetic fixture only',
            'decisions': [{'voxel_index': [10, 0, 0], 'decision': 'SELF', 'reason': 'synthetic selected cell'},
                          {'voxel_index': [-11, 0, 0], 'decision': 'UNSURE', 'reason': ''}]}


class ReviewTest(unittest.TestCase):
    def test_profile_remains_disabled(self):
        profile = make_profile(candidate(), review(), 'test_sha')
        self.assertFalse(profile['self_mask_enabled'])
        self.assertEqual(profile['confirmation_basis'], 'reviewer_declaration_only')
        self.assertEqual(profile['self_voxels'], [[10, 0, 0]])

    def test_default_preserves_valid_points(self):
        profile = make_profile(candidate(), review(), 'test_sha')
        result = preview(np.array([[.205, .01, .01]]), 'sensor', profile)
        self.assertFalse(result['self_removed'].any())
        self.assertTrue(result['kept'].all())

    def test_explicit_preview_removes_only_reviewed_cells(self):
        profile = make_profile(candidate(), review(), 'test_sha')
        points = np.array([[.205, .01, .01], [-.205, .01, .01], [.4, 0, 0], [.1, 0, 0]])
        original = points.copy()
        result = preview(points, 'sensor', profile, enabled=True)
        np.testing.assert_array_equal(result['self_removed'], [True, False, False, False])
        np.testing.assert_array_equal(points, original)
        self.assertEqual(result['masked_space_occupancy'], 'unknown_not_free')

    def test_external_and_unsure_are_retained(self):
        for choice in ('EXTERNAL', 'UNSURE'):
            decisions = review()
            decisions['decisions'][0]['decision'] = choice
            profile = make_profile(candidate(), decisions, 'test_sha')
            self.assertFalse(profile['self_identity_confirmed'])
            self.assertFalse(preview(np.array([[.205, .01, .01]]), 'sensor', profile, True)['self_removed'].any())

    def test_negative_cell(self):
        decisions = review()
        decisions['decisions'][1].update(decision='SELF', reason='synthetic negative cell')
        profile = make_profile(candidate(), decisions, 'test_sha')
        self.assertTrue(preview(np.array([[-.205, .01, .01]]), 'sensor', profile, True)['self_removed'][0])

    def test_no_blanket_direction_or_radius_exclusion(self):
        profile = make_profile(candidate(), review(), 'test_sha')
        points = np.array([[.205, .01, .01], [.205, .06, .01], [.225, .01, .01], [.105, .01, .01]])
        np.testing.assert_array_equal(preview(points, 'sensor', profile, True)['self_removed'], [True, False, False, False])

    def test_voxel_boundary_is_half_open(self):
        profile = make_profile(candidate(), review(), 'test_sha')
        boundary = 11*.02
        points = np.array([[np.nextafter(boundary, -np.inf), .01, .01], [boundary, .01, .01]])
        np.testing.assert_array_equal(preview(points, 'sensor', profile, True)['self_removed'], [True, False])

    def test_invalid_points_are_not_claimed_as_self(self):
        profile = make_profile(candidate(), review(), 'test_sha')
        points = np.array([[np.nan, 0, 0], [np.inf, 1, 1], [0, 0, 0], [1e100, 1, 1]])
        result = preview(points, 'sensor', profile, True)
        np.testing.assert_array_equal(result['invalid'], [True, True, True, False])
        self.assertFalse(result['self_removed'].any())

    def test_wrong_candidate_digest(self):
        with self.assertRaises(ValueError): make_profile(candidate(), review(), 'wrong_sha')

    def test_missing_or_duplicate_review_rows(self):
        for change in ('missing', 'duplicate', 'unknown'):
            data = review()
            if change == 'missing': data['decisions'].pop()
            if change == 'duplicate': data['decisions'][1] = data['decisions'][0]
            if change == 'unknown': data['decisions'][1]['voxel_index'] = [99, 0, 0]
            with self.subTest(change=change), self.assertRaises(ValueError):
                make_profile(candidate(), data, 'test_sha')

    def test_self_requires_both_reasons(self):
        for location in ('global', 'cell'):
            data = review()
            if location == 'global': data['review_basis'] = ' '
            else: data['decisions'][0]['reason'] = ''
            with self.subTest(location=location), self.assertRaises(ValueError):
                make_profile(candidate(), data, 'test_sha')

    def test_bad_candidate_schema(self):
        for field, value in [('kind', 'other'), ('self_mask_enabled', True), ('candidate_voxels', 999)]:
            data = candidate()
            data[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): checked_candidate(data)

    def test_frame_mismatch(self):
        profile = make_profile(candidate(), review(), 'test_sha')
        with self.assertRaises(ValueError): preview(np.zeros((1, 3)), 'wrong', profile, True)

    def test_undeclared_profile_cannot_exclude(self):
        profile = make_profile(candidate(), review(), 'test_sha')
        profile['self_identity_confirmed'] = False
        with self.assertRaises(ValueError): preview(np.array([[.205, .01, .01]]), 'sensor', profile, True)

    def test_external_object_inside_accepted_cell_is_also_removed(self):
        # 反例：相同空间格中的外物无法仅靠几何身份区分，不能声称误删问题消失。
        profile = make_profile(candidate(), review(), 'test_sha')
        self.assertTrue(preview(np.array([[.209, .012, .01]]), 'sensor', profile, True)['self_removed'][0])

    def test_page_matches_source_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            npz, report, html = root/'points.npz', root/'candidate.json', root/'review.html'
            np.savez_compressed(npz, xyz=np.tile([.205, .01, .01], (121, 1)), frame_id=np.arange(121),
                                frame_stamps=100+np.arange(121)*.1, frame_names=np.array(['sensor']*121))
            data = candidate()
            data['input_sha256'] = digest(npz)
            report.write_text(json.dumps(data), encoding='utf-8')
            make_page(report, npz, html)
            self.assertIn('candidate_sha256', html.read_text(encoding='utf-8'))
            self.assertNotIn('__REVIEW_DATA__', html.read_text(encoding='utf-8'))
            with self.assertRaises(FileExistsError): make_page(report, npz, html)
            data['input_sha256'] = 'wrong'
            report.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaises(ValueError): make_page(report, npz, root/'bad.html')
            self.assertFalse((root/'bad.html').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
