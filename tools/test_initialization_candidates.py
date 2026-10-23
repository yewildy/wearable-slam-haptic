"""人工数据检查统计逻辑；不将静态背景当作已确认自身。"""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from initialization_candidates import analyze, run


def sample():
    n = 121
    return {'xyz': np.tile([-.21, .01, .01], (n, 1)),
            'frame_id': np.arange(n), 'frame_stamps': 100 + np.arange(n) * .1,
            'frame_names': np.array(['sensor'] * n)}


class CandidatesTest(unittest.TestCase):
    def test_stationary_object_is_only_candidate(self):
        result = analyze(sample(), 'sensor')
        self.assertEqual(result['frames_used'], 101)
        self.assertEqual(result['candidate_voxels'], 1)
        for field in ('self_mask_enabled', 'self_identity_confirmed', 'mask_applied'):
            self.assertIs(result[field], False)
        self.assertNotIn('self_boxes', result)

    def test_duplicate_points_do_not_inflate_occupancy(self):
        data = sample()
        data['xyz'] = np.repeat(data['xyz'], 20, axis=0)
        data['frame_id'] = np.repeat(data['frame_id'], 20)
        result = analyze(data, 'sensor')
        self.assertEqual(result['candidate_cells'][0]['frames_seen'], 101)
        self.assertEqual(result['candidate_cells'][0]['frame_fraction'], 1)

    def test_missing_near_returns_stay_in_denominator(self):
        data = sample()
        data['xyz'], data['frame_id'] = data['xyz'][::2], data['frame_id'][::2]
        self.assertEqual(analyze(data, 'sensor')['candidate_voxels'], 0)

    def test_no_points_in_initial_window(self):
        data = sample()
        data['xyz'], data['frame_id'] = data['xyz'][110:], data['frame_id'][110:]
        self.assertEqual(analyze(data, 'sensor')['candidate_voxels'], 0)

    def test_temporal_segment_rejects_late_only_surface(self):
        data = sample()
        data['xyz'], data['frame_id'] = data['xyz'][20:], data['frame_id'][20:]
        self.assertEqual(analyze(data, 'sensor')['candidate_voxels'], 0)

    def test_negative_voxel_floor(self):
        self.assertEqual(analyze(sample(), 'sensor')['candidate_cells'][0]['voxel_index'], [-11, 0, 0])

    def test_future_points_not_used(self):
        data = sample()
        data['xyz'][101:] = [.4, .02, .02]
        self.assertEqual(analyze(data, 'sensor')['candidate_voxels'], 1)

    def test_reordered_points_produce_identical_statistics(self):
        data = sample()
        other = copy.deepcopy(data)
        other['xyz'], other['frame_id'] = other['xyz'][::-1], other['frame_id'][::-1]
        self.assertEqual(analyze(data, 'sensor'), analyze(other, 'sensor'))

    def test_short_recording_rejected(self):
        with self.assertRaises(ValueError):
            analyze(sample(), 'sensor', seconds=15)

    def test_gaps_rejected_including_window_end(self):
        for start in (45, 99):
            data = sample()
            data['frame_stamps'][start:] += 1
            with self.subTest(start=start), self.assertRaises(ValueError):
                analyze(data, 'sensor')

    def test_bad_schema(self):
        for change in ('frame', 'stamp', 'index', 'nan', 'zero', 'missing_names'):
            data = sample()
            if change == 'frame': data['frame_names'][0] = 'other'
            if change == 'stamp': data['frame_stamps'][1] = data['frame_stamps'][0]
            if change == 'index': data['frame_id'][0] = 999
            if change == 'nan': data['xyz'][0, 0] = np.nan
            if change == 'zero': data['xyz'][0] = 0
            if change == 'missing_names': del data['frame_names']
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze(data, 'sensor')

    def test_bad_parameters(self):
        for params in ({'seconds': 0}, {'voxel_m': np.nan}, {'occupancy': 1.1},
                       {'min_frames': True}, {'min_frames': 1000}, {'max_gap': -1}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                analyze(sample(), 'sensor', **params)

    def test_file_outputs_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root/'near.npz', root/'result'
            np.savez_compressed(source, **sample())
            before = source.read_bytes()
            run(source, target, 'sensor')
            self.assertTrue((target/'candidate_review.json').is_file())
            self.assertTrue((target/'candidate_voxels.csv').is_file())
            with self.assertRaises(FileExistsError): run(source, target, 'sensor')
            self.assertEqual(source.read_bytes(), before)

    def test_invalid_input_creates_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root/'near.npz', root/'result'
            data = sample()
            data['frame_stamps'][1] = 0
            np.savez_compressed(source, **data)
            with self.assertRaises(ValueError): run(source, target, 'sensor')
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
