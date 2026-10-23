"""Compare saved sensor-local near returns without starting ROS or hardware.

Saved samples contain no pose or far-field truth. Added targets and identity poses
are explicitly synthetic controls, never reconstructed real-world observations.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def group(distance):
    return np.array([[distance, 0, 0], [distance, .01, 0],
                     [distance, 0, .01], [distance, .01, .01]])


def pose(stamp):
    return dict(matrix=np.eye(4), frame='synthetic_world', stamp=stamp)


def summarize(rows):
    mature = [r for r in rows if r.get('rolling_ready')]
    fields = ('rolling_raw_removed_points', 'rolling_removed_points',
              'rolling_protected_track_points', 'rolling_continuity_removed_points',
              'isolated_removed_points', 'near_untrusted_points')
    return dict(frames=len(rows), mature_frames=len(mature),
                mature_states=dict(Counter(r['state'] for r in mature)),
                valid_near_frames=sum(r['valid'] and r.get('distance_m') is not None
                                      and r['distance_m'] < .5 for r in mature),
                stage_totals={f: sum(r.get(f, 0) for r in mature) for f in fields},
                last={k: rows[-1].get(k) for k in ('state', 'distance_m', 'strength') + fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code))
    from distance_policy import DistancePolicy
    from rolling_distance import rolling_result
    from rolling_near import RollingNear
    from motion_pipeline import MotionEngine

    sample = np.load(args.sample, allow_pickle=False)
    input_hash = hashlib.sha256(args.sample.read_bytes()).hexdigest()
    stamps = sample['stamps']; offsets = sample['offsets']; points = sample['xyz']
    assert len(offsets) == len(stamps)+1 and offsets[0] == 0 and offsets[-1] == len(points)
    assert np.all(np.diff(stamps) > 0) and np.all(np.diff(stamps) <= .5)
    finite = np.isfinite(points).all(axis=1)
    max_range = float(np.linalg.norm(points[finite], axis=1).max())
    assert max_range <= .5+1e-8, 'This audit expects a near-only crop'
    rows = {}; summaries = {}
    for variant in ('r4_near_crop', 'v3_near_crop_no_pose',
                    'v3_crop_plus_synthetic_far_no_pose', 'v3_crop_plus_synthetic_far_identity_pose'):
        engine = MotionEngine(); old_policy = DistancePolicy(); old_rolling = RollingNear()
        records = []
        for i, original_stamp in enumerate(stamps):
            stamp = float(original_stamp-stamps[0]+1.)
            xyz = points[offsets[i]:offsets[i+1]]
            if 'plus_synthetic' in variant:
                xyz = np.vstack([xyz, group(.8)])
            if variant.startswith('r4'):
                report = rolling_result(old_policy, old_rolling, xyz, 'livox_frame', stamp, 0., len(xyz))
            else:
                report = engine.process(xyz, 'livox_frame', stamp,
                                        pose=pose(stamp) if 'identity_pose' in variant else None)
            report = {k: report.get(k) for k in (
                'state', 'valid', 'distance_m', 'strength', 'rolling_ready',
                'rolling_raw_removed_points', 'rolling_removed_points',
                'rolling_protected_track_points', 'rolling_continuity_removed_points',
                'isolated_removed_points', 'near_untrusted_points') if k in report}
            records.append(report)
        rows[variant] = records; summaries[variant] = summarize(records)

    controls = {}
    for name in ('fixed_near_plus_far', 'approached_then_stopped'):
        engine = MotionEngine(); result = None
        for i in range(350):
            stamp = 1.+i*.1
            if name == 'fixed_near_plus_far':
                xyz = np.vstack([group(.1), group(.8)])
            else:
                moving = 1. if i < 110 else max(.3, 1.-(i-109)*.05)
                xyz = np.vstack([group(.1), group(moving)])
            result = engine.process(xyz, 'livox_frame', stamp, pose=pose(stamp))
        controls[name] = {k: result.get(k) for k in (
            'state', 'distance_m', 'strength', 'rolling_raw_removed_points',
            'rolling_removed_points', 'rolling_protected_track_points')}
    assert controls['fixed_near_plus_far']['distance_m'] == .8
    assert abs(controls['approached_then_stopped']['distance_m']-.3) < 1e-9
    assert controls['approached_then_stopped']['rolling_protected_track_points'] > 0
    assert hashlib.sha256(args.sample.read_bytes()).hexdigest() == input_hash
    args.output.mkdir(exist_ok=False)
    result = dict(sample_frames=len(stamps), duration_s=float(stamps[-1]-stamps[0]),
                  sample_max_range_m=max_range, sample_has_pose=False,
                  sample_has_far_field=False, input_unchanged=True,
                  variants=summaries, synthetic_controls=controls,
                  production_code_changed=False)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (args.output/'per_frame_private.json').write_text(json.dumps(rows, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
