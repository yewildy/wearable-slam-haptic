#!/usr/bin/env python3
"""Verify exclusion mechanics on a saved bag, not physical self identity.

The accepted profile below is an in-memory TEST FIXTURE simulating review.
It must never be saved/deployed as a real wearer mask. No ROS nodes are started.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(bag_path, candidate_path, tools_dir):
    import rosbag
    tools_dir = Path(tools_dir)
    review = load_module('review_verification_api', tools_dir/'review_self_mask.py')
    candidate_bytes = Path(candidate_path).read_bytes()
    candidate = json.loads(candidate_bytes)
    frame, width, keys = review.checked_candidate(candidate)
    if not keys:
        raise ValueError('No candidate cells available for a nontrivial exclusion trial')
    fingerprint = hashlib.sha256(candidate_bytes).hexdigest()
    # Actual unreviewed state: all regions remain UNSURE and therefore retained.
    unreviewed = {'schema_version': 1, 'candidate_sha256': fingerprint,
                  'review_basis': 'No physical review performed',
                  'decisions': [{'voxel_index': list(key), 'decision': 'UNSURE', 'reason': ''} for key in keys]}
    disabled = review.make_profile(candidate, unreviewed, fingerprint)
    # Software-only mock review exercises the real filter. The True declaration
    # inside this fixture is simulated input, NOT a claim that points are self.
    mock_review = {'schema_version': 1, 'candidate_sha256': fingerprint,
                   'review_basis': 'SOFTWARE TEST FIXTURE ONLY. Physical identity is unverified.',
                   'decisions': [{'voxel_index': list(key), 'decision': 'SELF',
                                  'reason': 'Simulated acceptance to test a candidate cell; NOT physical self review.'}
                                 for key in keys]}
    fixture = review.make_profile(candidate, mock_review, fingerprint)
    rows = []
    key_set = set(keys)
    guard = hashlib.sha256(Path(bag_path).read_bytes()).hexdigest()
    start = time.perf_counter()
    with rosbag.Bag(str(bag_path), 'r') as bag:
        for _, message, _ in bag.read_messages(topics=['/livox/lidar']):
            if message.header.frame_id != frame:
                raise ValueError('Full bag contains a coordinate frame mismatch')
            xyz = np.array([(p.x, p.y, p.z) for p in message.points], dtype=np.float32).reshape(-1, 3)
            if message.point_num != len(xyz):
                raise ValueError('Declared point count mismatch')
            original = xyz.copy()
            finite = np.isfinite(xyz).all(axis=1)
            valid = finite & ~np.all(xyz == 0, axis=1)
            before = review.preview(xyz, frame, disabled, enabled=True)
            after = review.preview(xyz, frame, fixture, enabled=True)
            # Independent geometric expectation, using exact index membership.
            expected = np.zeros(len(xyz), dtype=bool)
            nearby = valid & np.all(np.abs(xyz) <= .6, axis=1)
            cells = np.floor(xyz[nearby].astype(float)/width).astype(np.int64)
            expected[nearby] = [tuple(cell) in key_set for cell in cells]
            assert not before['self_removed'].any(), 'Unreviewed default excluded points'
            np.testing.assert_array_equal(after['self_removed'], expected)
            np.testing.assert_array_equal(after['kept'], valid & ~expected)
            np.testing.assert_array_equal(xyz, original)
            rows.append({'frame': len(rows), 'stamp': message.header.stamp.to_sec(),
                         'input': len(xyz), 'valid': int(valid.sum()),
                         'invalid': int((~valid).sum()), 'removed': int(expected.sum()),
                         'kept_valid': int(after['kept'].sum()), 'unreviewed_removed': 0,
                         'outside_candidate_removed': int((after['self_removed'] & ~expected).sum())})
    if len(rows) <= candidate['frames_used']:
        raise ValueError('No later frames available to test beyond initialization')
    stamps = np.array([row['stamp'] for row in rows])
    if not np.isfinite(stamps).all() or np.any(stamps <= 0) or np.any(np.diff(stamps) <= 0):
        raise ValueError('Bag timestamps must be finite positive and strictly increasing')
    assert hashlib.sha256(Path(bag_path).read_bytes()).hexdigest() == guard
    assert Path(candidate_path).read_bytes() == candidate_bytes
    def totals(part):
        return {'frames': len(part), 'input_points': sum(row['input'] for row in part),
                'valid_points': sum(row['valid'] for row in part),
                'invalid_points': sum(row['invalid'] for row in part),
                'test_fixture_removed': sum(row['removed'] for row in part),
                'kept_valid': sum(row['kept_valid'] for row in part),
                'frames_with_exclusion': sum(row['removed'] > 0 for row in part),
                'unreviewed_removed': sum(row['unreviewed_removed'] for row in part),
                'outside_candidate_removed': sum(row['outside_candidate_removed'] for row in part)}
    split = candidate['frames_used']
    # Synthetic controls are not mixed into the original bag.
    centers = (np.array(keys[:12], dtype=float)+.5)*width
    excluded = review.preview(centers, frame, fixture, enabled=True)['self_removed']
    outside_keys = [key for key in ((x, 0, 0) for x in range(-25, 26)) if key not in key_set][:12]
    outside = (np.array(outside_keys, dtype=float)+.5)*width
    retained = review.preview(outside, frame, fixture, enabled=True)['kept']
    assert excluded.all() and retained.all()
    return {'schema_version': 1, 'analysis_kind': 'candidate_exclusion_software_fixture',
            'candidate_voxels': len(keys), 'sample_header_span_seconds': float(stamps[-1]-stamps[0]),
            'initialization': totals(rows[:split]), 'later_same_recording': totals(rows[split:]),
            'all_frames': totals(rows),
            'synthetic_controls': {'inside_selected_cells_removed': int(excluded.sum()),
                                   'outside_selected_cells_retained': int(retained.sum())},
            'actual_filter_function_exercised': True, 'original_bag_unchanged': True,
            'candidate_unchanged': True, 'live_default_changed': False,
            'test_fixture_saved_as_profile': False, 'physical_self_identity_verified': False,
            'real_external_object_ground_truth_available': False,
            'elapsed_offline_verification_seconds': time.perf_counter()-start,
            'limitations': ['Mock accepted regions only; actual candidates remain unreviewed.',
                           'Later frames are from the same recording, not an independent scene.',
                           'An external object in a selected cell would also be excluded.',
                           'No realtime pipeline, wearer, accuracy, false-negative or latency validation.'],
            'per_frame': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--tools-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = verify(args.bag, args.candidate, args.tools_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'exclusion_private.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    summary = {key: value for key, value in report.items() if key != 'per_frame'}
    (args.output/'exclusion_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
