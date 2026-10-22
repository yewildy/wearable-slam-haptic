#!/usr/bin/env python3
"""Record a bounded raw sample and inspect persistent near returns; never auto-mask."""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np


def describe(values):
    return {'min': float(np.min(values)), 'median': float(np.median(values)),
            'max': float(np.max(values))} if len(values) else None


def candidate_quality(timestamps, frame_names, count_mismatches, near_count):
    stamps = np.asarray(timestamps, dtype=np.float64)
    finite = np.isfinite(stamps)
    deltas = np.diff(stamps)
    valid_deltas = deltas[np.isfinite(deltas)]
    invalid_stamps = int((~finite | (stamps <= 0)).sum())
    non_increasing = int((valid_deltas <= 0).sum())
    reasons = []
    if not near_count: reasons.append('no_nonzero_near_points')
    if len(set(frame_names)) != 1: reasons.append('mixed_coordinate_frames')
    if any(not name.strip() for name in frame_names): reasons.append('empty_coordinate_frame')
    if invalid_stamps: reasons.append('invalid_header_timestamps')
    if non_increasing: reasons.append('non_increasing_header_timestamps')
    if count_mismatches: reasons.append('declared_point_count_mismatch')
    return {'candidate_eligible': not reasons, 'candidate_block_reasons': reasons,
            'invalid_header_timestamp_frames': invalid_stamps,
            'non_increasing_header_transitions': non_increasing,
            'declared_count_mismatch_frames': count_mismatches,
            'header_gap_seconds': describe(valid_deltas),
            'note': 'Eligibility is data consistency only, not confirmation of self geometry or data freshness.'}


def write_pcd(path, xyz):
    with path.open('x') as stream:
        stream.write('# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\n'
                     'TYPE F F F\nCOUNT 1 1 1\nWIDTH %d\nHEIGHT 1\n'
                     'VIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA ascii\n' % (len(xyz), len(xyz)))
        np.savetxt(stream, xyz, fmt='%.6f')


def analyze(bag_path, output):
    import rosbag
    frames = 0
    counts = collections.Counter()
    tags = collections.Counter()
    voxel_frames = collections.Counter()
    near_chunks, frame_chunks, timestamps = [], [], []
    frame_names = set()
    per_frame_names = []
    count_mismatches = 0
    baseline_states = collections.Counter()
    baseline_distances = []
    last_cloud = np.empty((0, 3), dtype=np.float32)
    voxel_size = 0.02
    with rosbag.Bag(str(bag_path), 'r') as bag:
        topic_counts = {k: v.message_count for k, v in bag.get_type_and_topic_info().topics.items()}
        for _, message, _ in bag.read_messages(topics=['/livox/lidar']):
            xyz = np.array([(p.x, p.y, p.z) for p in message.points], dtype=np.float32).reshape(-1, 3)
            raw_tags = np.array([p.tag for p in message.points], dtype=np.uint8)
            counts['input_points'] += len(xyz)
            count_mismatches += int(message.point_num != len(xyz))
            finite = np.isfinite(xyz).all(axis=1)
            zero = finite & (xyz == 0).all(axis=1)
            counts['nonfinite_points'] += int((~finite).sum())
            counts['zero_points'] += int(zero.sum())
            valid = finite & ~zero
            last_cloud = xyz[valid]
            lengths = np.linalg.norm(xyz[valid], axis=1)
            near = (lengths > 0) & (lengths <= 0.5)
            cloud = xyz[valid][near]
            near_chunks.append(cloud)
            frame_chunks.append(np.full(len(cloud), frames, dtype=np.int32))
            tags.update(int(x) for x in raw_tags[valid][near])
            voxels = np.floor(cloud/voxel_size).astype(np.int32)
            voxel_frames.update(tuple(x) for x in np.unique(voxels, axis=0))
            timestamps.append(message.header.stamp.to_sec())
            frame_names.add(message.header.frame_id)
            per_frame_names.append(message.header.frame_id)
            frames += 1
        for _, message, _ in bag.read_messages(topics=['/haptic/proximity/diagnostics']):
            if message.status:
                status = message.status[0]
                baseline_states[status.message] += 1
                fields = {v.key: v.value for v in status.values}
                value = float(fields.get('distance_m', 'nan'))
                if np.isfinite(value): baseline_distances.append(value)
    if not frames:
        raise RuntimeError('No /livox/lidar frames saved; no analysis produced')
    near_xyz = np.concatenate(near_chunks)
    frame_ids = np.concatenate(frame_chunks)
    quality = candidate_quality(timestamps, per_frame_names, count_mismatches, len(near_xyz))
    np.savez_compressed(output/'near_returns.npz', xyz=near_xyz, frame_id=frame_ids,
                        frame_stamps=np.array(timestamps), frame_names=np.array(per_frame_names),
                        voxel_size_m=voxel_size)
    write_pcd(output/'last_valid_cloud.pcd', last_cloud)
    write_pcd(output/'near_returns.pcd', near_xyz)
    distances = np.linalg.norm(near_xyz, axis=1)
    sectors = np.floor((np.degrees(np.arctan2(near_xyz[:, 1], near_xyz[:, 0]))+180)/30).astype(int)
    sectors = np.clip(sectors, 0, 11)
    with (output/'near_voxels.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['x_center_m','y_center_m','z_center_m','frames_seen','frame_fraction'])
        for key, number in voxel_frames.most_common():
            writer.writerow([*((np.array(key)+0.5)*voxel_size), number, number/frames])
    radial_edges = np.array([0, .1, .15, .19, .2, .21, .25, .3, .4, .5])
    histogram, _ = np.histogram(distances, bins=radial_edges)
    report = {'frames': frames, 'frame_names': sorted(frame_names), 'topic_counts': topic_counts,
              'lidar_header_duration_seconds': float(timestamps[-1]-timestamps[0])
                  if np.isfinite(timestamps[0]) and np.isfinite(timestamps[-1]) else None,
              'data_quality': quality,
              'counts': dict(counts), 'near_points_0_to_0_5_m': len(near_xyz),
              'near_xyz_m': {axis: describe(near_xyz[:, i]) for i, axis in enumerate('xyz')},
              'radial_bins_m': radial_edges.tolist(), 'radial_counts': histogram.tolist(),
              'azimuth_bins_start_deg': list(range(-180, 180, 30)),
              'azimuth_counts': np.bincount(sectors, minlength=12).tolist(),
              'near_raw_tag_counts': {str(k): v for k, v in tags.items()},
              'voxel_size_m': voxel_size,
              'voxels_seen_in_at_least_80_percent_of_frames': sum(v/frames >= .8 for v in voxel_frames.values()),
              'mask_applied': False, 'self_identity_confirmed': False,
              'recorded_baseline_states': dict(baseline_states),
              'recorded_baseline_distance_m': describe(baseline_distances),
              'note': 'Persistent is not self: static surroundings also persist. Tag bytes are not classified.'}
    (output/'analysis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if quality['candidate_eligible']:
        candidate = {'self_mask_enabled': False, 'self_mask_frame': next(iter(frame_names)),
                     'self_boxes': [{'min': (near_xyz.min(axis=0)-.02).astype(float).tolist(),
                                     'max': (near_xyz.max(axis=0)+.02).astype(float).tolist()}],
                     'profile_source': 'observed_near_returns_with_2cm_margin',
                     'self_identity_confirmed': False,
                     'warning': 'Candidate only. Static surroundings may be included; do not enable as a calibrated body mask.'}
        (output/'candidate_mask.yaml').write_text(json.dumps(candidate, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=15)
    parser.add_argument('--output-root', type=Path, default=Path.home()/'bags/haptic')
    parser.add_argument('--bag', type=Path, help='Analyze an existing bag without recording')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 60:
        parser.error('seconds must be within 1..60')
    import rospy
    from livox_ros_driver2.msg import CustomMsg
    output = args.output_root/('cloud_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    bag_path = args.bag
    if bag_path is None:
        rospy.init_node('capture_near_cloud', anonymous=True, disable_signals=True)
        rospy.wait_for_message('/livox/lidar', CustomMsg, timeout=10)
        params = rospy.get_param('/haptic_proximity_node', {})
        (output/'parameters.json').write_text(json.dumps(params, indent=2), encoding='utf-8')
        import rospkg
        package = Path(rospkg.RosPack().get_path('haptic_feedback'))
        hashes = {}
        for name in ['src/pcl_proximity_node.cpp','launch/pcl_proximity.launch','config/self_mask.yaml']:
            path = package/name
            if path.is_file(): hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (output/'source_hashes.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
        bag_path = output/'raw.bag'
        with (output/'record.log').open('x') as log:
            process = subprocess.Popen(['rosbag','record','--buffsize=128','-O',str(bag_path),
                '/livox/lidar','/livox/imu','/haptic/proximity/diagnostics'],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                time.sleep(args.seconds)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                    raise RuntimeError('Recorder did not finish normally; preserve partial files')
            if process.returncode not in (0, 130, -signal.SIGINT):
                raise RuntimeError('Recorder exited abnormally; inspect record.log')
        rospy.signal_shutdown('Recording complete')
    analyze(bag_path, output)
    print('SAVED_DIRECTORY='+str(output))


if __name__ == '__main__':
    main()
