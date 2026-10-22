import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

script = Path(__file__).parent/'scripts/capture_near_cloud.py'
if not script.exists():
    script = Path.home()/'haptic_ws/src/haptic_feedback/scripts/capture_near_cloud.py'
spec = importlib.util.spec_from_file_location('capture_near_cloud', script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def frame(points=((.25, .01, .0),), stamp=1., name='test_lidar', declared=None):
    return SimpleNamespace(points=[SimpleNamespace(x=x,y=y,z=z,tag=0) for x,y,z in points],
        point_num=len(points) if declared is None else declared,
        header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda:stamp),frame_id=name))


class FakeBag:
    frames = []
    def __init__(self, *args): pass
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def get_type_and_topic_info(self):
        return SimpleNamespace(topics={'/livox/lidar':SimpleNamespace(message_count=len(self.frames))})
    def read_messages(self, topics):
        if topics == ['/livox/lidar']:
            yield from (('/livox/lidar',f,None) for f in self.frames)


class AnalysisTests(unittest.TestCase):
    def analyze(self, frames):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        output = Path(temp.name)
        with patch.dict(sys.modules, {'rosbag':SimpleNamespace(Bag=FakeBag)}), \
             patch.object(FakeBag,'frames',frames), contextlib.redirect_stdout(io.StringIO()):
            report = module.analyze(Path('synthetic.bag'),output)
        return report, output

    def assertBlocked(self, frames, reason):
        report, output = self.analyze(frames)
        self.assertFalse(report['data_quality']['candidate_eligible'])
        self.assertIn(reason,report['data_quality']['candidate_block_reasons'])
        self.assertFalse((output/'candidate_mask.yaml').exists())
        self.assertTrue((output/'near_returns.npz').exists())

    def test_valid_candidate_disabled_and_preserves_frame_metadata(self):
        report, output = self.analyze([frame(stamp=1),frame(stamp=1.1)])
        self.assertTrue(report['data_quality']['candidate_eligible'])
        self.assertAlmostEqual(report['data_quality']['header_gap_seconds']['median'],.1)
        candidate = json.loads((output/'candidate_mask.yaml').read_text())
        self.assertFalse(candidate['self_mask_enabled'])
        self.assertFalse(candidate['self_identity_confirmed'])
        with np.load(output/'near_returns.npz') as data:
            self.assertEqual(data['frame_names'].tolist(),['test_lidar','test_lidar'])

    def test_zero_points_no_candidate(self):
        self.assertBlocked([frame(points=((0,0,0),))],'no_nonzero_near_points')

    def test_empty_frame_no_candidate(self):
        self.assertBlocked([frame(points=())],'no_nonzero_near_points')

    def test_nonfinite_points_filtered_and_counted(self):
        report, _ = self.analyze([frame(points=((float('nan'),0,0),(float('inf'),0,0),(.25,0,0)))])
        self.assertEqual(report['counts']['nonfinite_points'],2)
        self.assertEqual(report['near_points_0_to_0_5_m'],1)

    def test_mixed_frame_blocks_candidate(self):
        self.assertBlocked([frame(),frame(stamp=1.1,name='different')],'mixed_coordinate_frames')

    def test_blank_frame_blocks_candidate(self):
        self.assertBlocked([frame(name='')],'empty_coordinate_frame')

    def test_zero_timestamp_blocks_candidate(self):
        self.assertBlocked([frame(stamp=0)],'invalid_header_timestamps')

    def test_nonfinite_timestamp_blocks_candidate(self):
        report, output = self.analyze([frame(stamp=float('nan'))])
        self.assertFalse((output/'candidate_mask.yaml').exists())
        self.assertIsNone(report['lidar_header_duration_seconds'])
        json.dumps(report, allow_nan=False)

    def test_duplicate_stamp_blocks_candidate(self):
        self.assertBlocked([frame(),frame()],'non_increasing_header_timestamps')

    def test_backward_stamp_blocks_candidate(self):
        self.assertBlocked([frame(stamp=2),frame(stamp=1)],'non_increasing_header_timestamps')

    def test_count_mismatch_blocks_candidate(self):
        self.assertBlocked([frame(declared=4)],'declared_point_count_mismatch')

    def test_voxel_fraction_counts_frames_not_points(self):
        _, output = self.analyze([frame(points=((.25,0,0),)*20),frame(stamp=1.1)])
        import csv
        with (output/'near_voxels.csv').open() as stream:
            fractions=[float(row['frame_fraction']) for row in csv.DictReader(stream)]
        self.assertTrue(all(0 < f <= 1 for f in fractions))

    def test_missing_lidar_refuses_analysis(self):
        with self.assertRaisesRegex(RuntimeError,'No /livox/lidar'):
            self.analyze([])


if __name__=='__main__':
    unittest.main(verbosity=2)
