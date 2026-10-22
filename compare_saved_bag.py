"""Paired offline PCL replay. Never connects to the live ROS graph."""
import argparse
import collections
import copy
import datetime
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time

import numpy as np

from replay_candidate import inspect_bag, stop_owned_group, trial_profile, wait_ready


PARAMETERS = {'min_range': .2, 'max_range': 5., 'support_radius': .08, 'min_neighbors': 3,
              'weak_distance': 2., 'medium_distance': 1., 'strong_distance': .5,
              'hysteresis': .1, 'release_frames': 2, 'stale_timeout': .5}
METRICS = ['distance_m', 'self_masked_points', 'near_untrusted_points', 'range_points',
           'supported_points', 'removed_by_support', 'processing_ms']


def finite_number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def describe(values):
    if not values:
        return {'count': 0, 'min': None, 'mean': None, 'median': None, 'p95': None, 'max': None}
    return {'count': len(values), 'min': min(values), 'mean': statistics.mean(values),
            'median': statistics.median(values), 'p95': float(np.percentile(values, 95)), 'max': max(values)}


def indexed_rows(rows, expected_keys):
    expected = set(expected_keys)
    if len(expected) != len(expected_keys):
        raise ValueError('Expected headers are not unique')
    indexed, duplicates, unexpected = {}, 0, 0
    for row in rows:
        key = tuple(row['key'])
        if key not in expected:
            unexpected += 1
        elif key in indexed:
            duplicates += 1
        else:
            indexed[key] = row
    return indexed, duplicates, unexpected


def summarize(rows, expected_keys):
    indexed, duplicates, unexpected = indexed_rows(rows, expected_keys)
    matched = list(indexed.values())
    metrics = {}
    for name in METRICS:
        values = [finite_number(row['values'].get(name)) for row in matched]
        metrics[name] = describe([v for v in values if v is not None])
    unknown = sum(str(row['values'].get('level')) in ('UNKNOWN', '255') for row in matched)
    return {'frames_expected': len(expected_keys), 'frames_matched': len(matched),
            'missing_frames': len(expected_keys)-len(matched), 'duplicate_outputs': duplicates,
            'unexpected_outputs': unexpected,
            'complete_unique_headers': len(matched) == len(expected_keys) and duplicates == unexpected == 0,
            'states': dict(collections.Counter(row['state'] for row in matched)),
            'unknown_level_frames': unknown,
            'invalid_or_missing_level_frames': sum(str(row['values'].get('level')) not in
                ('0', '1', '2', '3', 'UNKNOWN', '255') for row in matched),
            'near_field_uncertain_frames': sum(row['values'].get('near_field_uncertain') == 'true' for row in matched),
            'frame_mismatch_outputs': sum(row['state'] == 'SELF_MASK_FRAME_MISMATCH' for row in rows),
            'unsupported_closer_return_frames': sum(row['values'].get('unsupported_closer_return') == 'true' for row in matched),
            'metrics': metrics}


def compare_rows(baseline, candidate, expected_keys):
    left, _, _ = indexed_rows(baseline, expected_keys)
    right, _, _ = indexed_rows(candidate, expected_keys)
    shared = sorted(left.keys() & right.keys())
    deltas = []
    for key in shared:
        a = finite_number(left[key]['values'].get('distance_m'))
        b = finite_number(right[key]['values'].get('distance_m'))
        if a is not None and b is not None:
            deltas.append(b-a)
    transitions = collections.Counter(left[key]['state']+' -> '+right[key]['state'] for key in shared)
    return {'paired_headers': len(shared), 'finite_distance_pairs': len(deltas),
            'candidate_minus_baseline_distance_m': describe(deltas), 'state_transitions': dict(transitions)}


def public_summary(report):
    return copy.deepcopy({key: report[key] for key in ['schema_version', 'scope', 'frames_expected',
        'header_duration_seconds', 'header_gap_seconds', 'replay_rate', 'parameters', 'mask_identity',
        'validation_scope', 'phases', 'paired', 'temporary_master_stopped', 'limitations']})


def read_expected(path, frame):
    import rosbag
    keys, stamps = [], []
    with rosbag.Bag(str(path), 'r') as bag:
        for _, message, _ in bag.read_messages(topics=['/livox/lidar']):
            stamp = message.header.stamp.to_nsec()
            if message.header.frame_id != frame or stamp <= 0:
                raise ValueError('Invalid frame name or timestamp in saved bag')
            if stamps and stamp <= stamps[-1]:
                raise ValueError('Saved headers must be strictly increasing for unique pairing')
            stamps.append(stamp)
            keys.append((frame, stamp))
    return keys, stamps


def main():
    import rospy
    import yaml
    from diagnostic_msgs.msg import DiagnosticArray
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='Contains saved raw.bag and disabled candidate_mask.yaml')
    args = parser.parse_args()
    directory = args.directory.resolve()
    raw = directory/'raw.bag'
    original_profile = yaml.safe_load((directory/'candidate_mask.yaml').read_text())
    if not isinstance(original_profile, dict) or original_profile.get('self_mask_enabled') is not False:
        raise ValueError('Input candidate must remain disabled')
    candidate = trial_profile(original_profile)
    metadata = inspect_bag(raw, candidate['self_mask_frame'])
    expected, stamps = read_expected(raw, candidate['self_mask_frame'])
    if len(expected) != metadata['frames_expected']:
        raise ValueError('Bag metadata count differs from readable frames')
    port = 11328
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', port))
    os.environ['ROS_MASTER_URI'] = 'http://127.0.0.1:'+str(port)
    os.environ['ROS_IP'] = '127.0.0.1'
    os.environ.pop('ROS_HOSTNAME', None)
    output = directory/('comparison_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(exist_ok=False)
    master = detector = player = None
    capture = {'active': False, 'rows': [], 'watchdog_messages': 0}
    phase_rows, phases, snapshots = {}, {}, {}

    def master_ready():
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=.1):
                return True
        except OSError:
            return False

    def receive(message):
        if not capture['active'] or not message.status:
            return
        if not message.header.frame_id:
            capture['watchdog_messages'] += 1
            return
        status = message.status[0]
        capture['rows'].append({'key': (message.header.frame_id, message.header.stamp.to_nsec()),
                                'state': status.message, 'values': {v.key: v.value for v in status.values}})

    with (output/'comparison.log').open('x') as log:
        try:
            master = subprocess.Popen(['roscore', '-p', str(port)], stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            wait_ready(master_ready, master, 10, 'Isolated master')
            rospy.init_node('compare_saved_bag', disable_signals=True)
            subscriber = rospy.Subscriber('/haptic/proximity/diagnostics', DiagnosticArray, receive, queue_size=1000)
            for phase, enabled in [('baseline', False), ('candidate_trial', True)]:
                name = 'offline_compare_'+phase
                params = dict(PARAMETERS, input_topic='/haptic/offline_compare/lidar',
                              self_mask_enabled=enabled, self_mask_frame=candidate['self_mask_frame'],
                              self_boxes=candidate['self_boxes'])
                for key, value in params.items():
                    rospy.set_param('/'+name+'/'+key, value)
                detector = subprocess.Popen(['rosrun', 'haptic_feedback', 'pcl_proximity_node', '__name:='+name],
                                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                wait_ready(lambda: subscriber.get_num_connections() > 0, detector, 8, 'Detector')
                snapshots[phase] = rospy.get_param('/'+name)
                capture.update(active=True, rows=[], watchdog_messages=0)
                started = time.monotonic()
                player = subprocess.Popen(['rosbag', 'play', '--wait-for-subscribers', '-r', '1', str(raw),
                    '--topics', '/livox/lidar', '/livox/lidar:=/haptic/offline_compare/lidar'],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                timeout = max(45., min(180., metadata['bag_duration_seconds']*2+10))
                if player.wait(timeout=timeout) != 0:
                    raise RuntimeError('Offline bag player failed')
                elapsed = time.monotonic()-started
                deadline = time.monotonic()+2
                while time.monotonic() < deadline:
                    if len({tuple(row['key']) for row in capture['rows']} & set(expected)) == len(expected):
                        break
                    time.sleep(.02)
                capture['active'] = False
                phase_rows[phase] = list(capture['rows'])
                phases[phase] = summarize(phase_rows[phase], expected)
                phases[phase]['playback_process_wall_seconds'] = elapsed
                phases[phase]['watchdog_messages_during_capture_window'] = capture['watchdog_messages']
                phases[phase]['mask_enabled_for_isolated_trial'] = enabled
                stop_owned_group(player)
                player = None
                stop_owned_group(detector)
                detector = None
                deadline = time.monotonic()+3
                while subscriber.get_num_connections() and time.monotonic() < deadline:
                    time.sleep(.05)
                if subscriber.get_num_connections():
                    raise RuntimeError('Previous detector still connected')
        finally:
            capture['active'] = False
            try:
                if rospy.core.is_initialized():
                    rospy.signal_shutdown('Comparison complete')
            finally:
                try:
                    stop_owned_group(player)
                finally:
                    try:
                        stop_owned_group(detector)
                    finally:
                        stop_owned_group(master)
    deadline = time.monotonic()+5
    while master_ready() and time.monotonic() < deadline:
        time.sleep(.05)
    if master_ready():
        raise RuntimeError('Owned test master remains reachable')
    report = {'schema_version': 1, 'scope': 'paired_full_cloud_offline_PCL_replay',
              'frames_expected': len(expected), 'header_duration_seconds': (stamps[-1]-stamps[0])/1e9,
              'header_gap_seconds': describe([(b-a)/1e9 for a, b in zip(stamps, stamps[1:])]),
              'replay_rate': 1., 'parameters': PARAMETERS,
              'mask_identity': 'unconfirmed_candidate_isolated_trial_only_live_default_unchanged',
              'validation_scope': 'same_static_recording_used_to_fit_candidate_not_independent_validation',
              'phases': phases, 'paired': compare_rows(phase_rows['baseline'], phase_rows['candidate_trial'], expected),
              'temporary_master_stopped': True,
              'limitations': ['No target distance reference, self labels or real false-alarm/miss-rate measurement',
                              'Unknown level and near-field uncertainty are different and reported separately',
                              'Masked or occluded space remains unknown occupancy, not evidence of free space',
                              'Processing time is callback work, not sensor-to-alert latency or a controlled benchmark',
                              'Playback wall time includes player startup; single sequential runs are order/load dependent',
                              'No live sensor, camera, mapping, motor or default-profile change'],
              'private_parameters': snapshots, 'private_frame_rows': phase_rows}
    (output/'comparison_private.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    public = public_summary(report)
    (output/'comparison_summary.json').write_text(json.dumps(public, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(public, indent=2, allow_nan=False))
    print('SAVED_DIRECTORY='+str(output))
    if not all(phase['complete_unique_headers'] for phase in phases.values()):
        raise RuntimeError('Incomplete/duplicate output: report retained but comparison not accepted')


if __name__ == '__main__':
    main()
