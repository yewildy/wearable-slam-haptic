"""Small offline support-radius study; no live inputs or parameter tuning in place."""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np

from compare_saved_bag import PARAMETERS, finite_number, indexed_rows, read_expected, summarize
from replay_candidate import stop_owned_group, wait_ready

RADII = (.04, .08, .12)


def select_indices(total, limit=15):
    if type(total) is not int or total < 1 or type(limit) is not int or limit < 1:
        raise ValueError('Positive integer frame count and limit required')
    return np.linspace(0, total-1, min(total, limit), dtype=int).tolist()


def check_near_export(message, index, export):
    required = {'xyz', 'frame_id', 'frame_stamps', 'frame_names'}
    if not required.issubset(export):
        raise ValueError('Current near export with per-frame names required')
    if export['frame_names'][index] != message.header.frame_id:
        raise ValueError('Saved near-export frame mismatch')
    if export['frame_stamps'][index] != message.header.stamp.to_sec():
        raise ValueError('Saved near-export timestamp mismatch')
    xyz = np.array([(p.x, p.y, p.z) for p in message.points], dtype=np.float32).reshape(-1, 3)
    valid = np.isfinite(xyz).all(axis=1) & ~np.all(xyz == 0, axis=1)
    xyz = xyz[valid]
    distance = np.linalg.norm(xyz, axis=1)
    near = xyz[(distance > 0) & (distance <= .5)]
    expected = export['xyz'][export['frame_id'] == index]
    if not np.array_equal(near, expected):
        raise ValueError('Saved near export differs from raw selected frame')
    return len(near)


def support_trend(phases, expected):
    if len(phases) != 3 or [p['radius_m'] for p in phases] != list(RADII):
        raise ValueError('Expected the fixed three-radius study in ascending order')
    if not expected:
        raise ValueError('No selected frames')
    indexed = []
    for phase in phases:
        by_key, duplicates, unexpected = indexed_rows(phase['rows'], expected)
        if len(by_key) != len(expected) or duplicates or unexpected:
            raise ValueError('Incomplete, duplicate or unexpected frame results')
        indexed.append(by_key)
    counts_monotonic, distance_checks = [], []
    for key in expected:
        counts = [finite_number(p[key]['values'].get('supported_points')) for p in indexed]
        if any(c is None or c < 0 or c != int(c) for c in counts):
            raise ValueError('Missing or invalid supported-point count')
        counts_monotonic.append(counts[0] <= counts[1] <= counts[2])
        distances = [finite_number(p[key]['values'].get('distance_m')) for p in indexed]
        if all(d is not None for d in distances):
            distance_checks.append(distances[0]+1e-6 >= distances[1] and distances[1]+1e-6 >= distances[2])
    return {'paired_frames': len(expected), 'nondecreasing_supported_counts_all_frames': all(counts_monotonic),
            'finite_distance_triplets': len(distance_checks),
            'nonincreasing_nearest_supported_distance': all(distance_checks) if distance_checks else None,
            'accuracy_improvement_established': False}


def public_summary(report):
    return copy.deepcopy({key: report[key] for key in ['scope', 'sample_selection', 'parameters_fixed',
        'radii_m', 'near_export_selected_frames_exact_match', 'saved_sample_results', 'support_trend',
        'synthetic_results', 'temporary_master_stopped', 'live_defaults_changed', 'limitations']})


def main():
    import rosbag
    import rospy
    from diagnostic_msgs.msg import DiagnosticArray
    from livox_ros_driver2.msg import CustomMsg, CustomPoint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('near_export', type=Path)
    parser.add_argument('binary', type=Path, help='Explicit isolated-build executable')
    parser.add_argument('output_root', type=Path)
    args = parser.parse_args()
    if not args.binary.is_file():
        raise ValueError('Isolated binary missing')
    with np.load(args.near_export, allow_pickle=False) as data:
        export = {key: data[key] for key in data.files}
    frame = str(export['frame_names'][0])
    expected_all, _ = read_expected(args.bag, frame)
    indices = select_indices(len(expected_all))
    selected, near_counts = [], []
    with rosbag.Bag(str(args.bag), 'r') as bag:
        info = bag.get_type_and_topic_info().topics['/livox/lidar']
        if info.msg_type != 'livox_ros_driver2/CustomMsg' or info.message_count != len(expected_all):
            raise ValueError('Saved topic metadata mismatch')
        for index, (_, message, _) in enumerate(bag.read_messages(topics=['/livox/lidar'])):
            if index in indices:
                near_counts.append(check_near_export(message, index, export))
                selected.append(message)
    expected = [expected_all[i] for i in indices]
    port = 11330
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', port))
    os.environ['ROS_MASTER_URI'] = 'http://127.0.0.1:'+str(port)
    os.environ['ROS_IP'] = '127.0.0.1'
    os.environ.pop('ROS_HOSTNAME', None)
    root = args.output_root/('support_radius_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    root.mkdir(parents=True, exist_ok=False)
    master = detector = None
    capture, messages = [], {}
    phases, summary, synthetic, snapshots = [], [], [], []

    def master_ready():
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=.1):
                return True
        except OSError:
            return False

    def receive(message):
        if message.status and message.header.frame_id:
            status = message.status[0]
            row = {'key': (message.header.frame_id, message.header.stamp.to_nsec()),
                   'state': status.message, 'values': {v.key: v.value for v in status.values}}
            capture.append(row)
            messages[row['key']] = row

    def send(message):
        key = (message.header.frame_id, message.header.stamp.to_nsec())
        publisher.publish(message)
        wait_ready(lambda: key in messages, detector, 4, 'Matched offline diagnostic')
        row = messages[key]
        assert row['values']['self_mask_review_state'] == 'DISABLED'
        return row

    background = [(1.5, i*.005, 0) for i in range(6)]
    fixtures = [('dense_target', [(.6, i*.005, 0) for i in range(6)]+background),
                ('sparse_square_target', [(.6, y, z) for y in [-.03, .03] for z in [-.03, .03]]+background),
                ('near_return_borrowing_wall_support', [(.5, 0, 0)]+[(.59, i*.005, 0) for i in range(6)])]
    with (root/'study.log').open('x') as log:
        try:
            master = subprocess.Popen(['roscore', '-p', str(port)], stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            wait_ready(master_ready, master, 10, 'Isolated master')
            rospy.init_node('support_radius_study', disable_signals=True)
            sub = rospy.Subscriber('/haptic/proximity/diagnostics', DiagnosticArray, receive, queue_size=1000)
            publisher = rospy.Publisher('/haptic/test/radius_cloud', CustomMsg, queue_size=1)
            for radius in RADII:
                name = 'radius_'+str(round(radius*1000))
                params = dict(PARAMETERS, support_radius=radius, input_topic='/haptic/test/radius_cloud',
                              self_mask_enabled=False, self_identity_confirmed=False, self_boxes=[])
                for key, value in params.items():
                    rospy.set_param('/'+name+'/'+key, value)
                detector = subprocess.Popen([str(args.binary.resolve()), '__name:='+name], stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                wait_ready(lambda: publisher.get_num_connections() > 0 and sub.get_num_connections() > 0,
                           detector, 8, 'Detector')
                capture.clear()
                messages.clear()
                snapshots.append(rospy.get_param('/'+name))
                for message in selected:
                    send(message)
                rows = list(capture)
                phases.append({'radius_m': radius, 'rows': rows})
                stats = summarize(rows, expected)
                stats['metrics'].pop('processing_ms')
                summary.append({'radius_m': radius, 'statistics': stats})
                for label, points in fixtures:
                    for _ in range(4):
                        message = CustomMsg()
                        message.header.frame_id = 'radius_fixture'
                        message.header.stamp = rospy.Time.now()
                        for xyz in points:
                            p = CustomPoint()
                            p.x, p.y, p.z = xyz
                            message.points.append(p)
                        message.point_num = len(message.points)
                        result = send(message)
                    data = result['values']
                    synthetic.append({'radius_m': radius, 'fixture': label, 'state_after_four_inputs': result['state'],
                                      'distance_m': finite_number(data['distance_m']),
                                      'supported_points': int(data['supported_points']),
                                      'removed_by_support': int(data['removed_by_support'])})
                stop_owned_group(detector)
                detector = None
                deadline = time.monotonic()+3
                while (publisher.get_num_connections() or sub.get_num_connections()) and time.monotonic() < deadline:
                    time.sleep(.05)
                assert publisher.get_num_connections() == sub.get_num_connections() == 0
        finally:
            try:
                if rospy.core.is_initialized():
                    rospy.signal_shutdown('Support-radius study complete')
            finally:
                try:
                    stop_owned_group(detector)
                finally:
                    stop_owned_group(master)
    with socket.socket() as probe:
        assert probe.connect_ex(('127.0.0.1', port)) != 0
    report = {'scope': 'three_radius_offline_subset_and_synthetic_counterexamples',
              'sample_selection': {'total_saved_frames': len(expected_all), 'selected_count': len(indices),
                                   'selected_indices': indices, 'selection': 'fixed evenly spaced indices including endpoints',
                                   'independent_scene': False},
              'parameters_fixed': {key: value for key, value in PARAMETERS.items() if key != 'support_radius'},
              'radii_m': RADII, 'near_export_selected_frames_exact_match': len(selected),
              'saved_sample_results': summary, 'support_trend': support_trend(phases, expected),
              'synthetic_results': synthetic, 'temporary_master_stopped': True, 'live_defaults_changed': False,
              'limitations': ['Only 15 or fewer selected frames from the existing static sample, not independent scenes',
                              'Mask disabled for this comparison; min_range remains 0.2 m',
                              'Increasing support radius can retain sparse objects and also support isolated nearer returns',
                              'More retained points or a smaller nearest distance is not proof of improved accuracy',
                              'Request/response replay, not recorded-time playback, a speed benchmark or latency measurement',
                              'Synthetic cases report the fourth input to settle the existing release hysteresis',
                              'No real target labels, false-alarm/miss rates, motion or wearable safety validation'],
              'private_phase_rows': phases, 'private_parameters': snapshots, 'private_selected_near_counts': near_counts,
              'private_binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest()}
    (root/'radius_private.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    public = public_summary(report)
    (root/'support_radius_summary.json').write_text(json.dumps(public, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(public, indent=2, allow_nan=False))
    print('SAVED_DIRECTORY='+str(root))
    assert report['support_trend']['nondecreasing_supported_counts_all_frames']
    assert all(row['statistics']['complete_unique_headers'] for row in summary)


if __name__ == '__main__':
    main()
