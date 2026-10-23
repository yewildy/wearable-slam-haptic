#!/usr/bin/env python3
"""只读已有完整bag，逐帧计算并写新的CSV/摘要。不启动ROS master或驱动。"""
import argparse
from collections import Counter
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import time
from distance_node import cloud_result
from distance_policy import DistancePolicy, Settings


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def replay(bag_path, output, frame, profile=None, apply=False, topic='/livox/lidar'):
    import rosbag
    bag_path, output = Path(bag_path), Path(output)
    before = sha(bag_path)
    policy = DistancePolicy(Settings(expected_frame=frame), profile, apply)
    output.mkdir(parents=True, exist_ok=False)
    counts, frames, valid, elapsed_total, values = Counter(), 0, 0, 0., []
    fields = ['frame', 'stamp', 'state', 'valid', 'distance_m', 'strength', 'level', 'candidate_distance_m',
              'candidate_strength', 'input_points', 'invalid_points', 'self_removed_points',
              'near_untrusted_points', 'unsupported_closer_return', 'supported_points', 'processing_ms']
    with (output/'frames.csv').open('x', newline='', encoding='utf-8') as stream, rosbag.Bag(str(bag_path), 'r') as bag:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for _, msg, _ in bag.read_messages(topics=[topic]):
            if getattr(msg, '_type', '') != 'livox_ros_driver2/CustomMsg':
                raise ValueError('Expected livox_ros_driver2/CustomMsg; no silent message conversion')
            started = time.monotonic()
            # 离线age=0按记录顺序重算；不伪装实时延迟或实时断流检测。
            result = cloud_result(policy, msg, msg.header.stamp.to_sec())
            elapsed = (time.monotonic()-started)*1000
            elapsed_total += elapsed
            row = {name:result.get(name) for name in fields}
            row.update(frame=frames, stamp=msg.header.stamp.to_sec(), processing_ms=elapsed)
            writer.writerow(row)
            counts[result['state']] += 1
            frames += 1
            valid += int(result['valid'])
            if result['strength'] is not None:
                values.append(result['strength'])
    if not frames:
        raise ValueError('No matching cloud messages')
    if sha(bag_path) != before:
        raise ValueError('Original bag changed during replay')
    summary = dict(frames=frames, valid_distance_decisions=valid, states=dict(counts),
                   strength_min=min(values) if values else None, strength_max=max(values) if values else None,
                   mean_compute_ms=elapsed_total/frames, settings=asdict(policy.settings),
                   mask_enabled=apply, original_unchanged=True, motor_control=False,
                   validation_scope='same saved stationary recording; no ground truth or live timing')
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bag', type=Path)
    p.add_argument('--frame', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--reviewed-profile', type=Path)
    p.add_argument('--apply-self-mask', action='store_true')
    args = p.parse_args()
    profile = json.loads(args.reviewed_profile.read_text(encoding='utf-8')) if args.reviewed_profile else None
    print(json.dumps(replay(args.bag, args.output, args.frame, profile, args.apply_self_mask), indent=2))


if __name__ == '__main__':
    main()
