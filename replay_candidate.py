import argparse
import collections
import copy
import datetime
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import time

def trial_profile(profile):
    if not isinstance(profile, dict) or profile.get('self_identity_confirmed') is not False:
        raise ValueError('Expected an explicitly unconfirmed candidate profile')
    frame = profile.get('self_mask_frame')
    if not isinstance(frame, str) or not frame.strip():
        raise ValueError('Candidate frame must be nonempty')
    boxes = profile.get('self_boxes')
    if not isinstance(boxes, list) or not boxes:
        raise ValueError('Candidate requires at least one geometry box')
    for box in boxes:
        if not isinstance(box, dict):
            raise ValueError('Invalid geometry box')
        for key in ['min', 'max']:
            values = box.get(key)
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError('Each bound requires three coordinates')
            if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1000 for v in values):
                raise ValueError('Bounds must be finite numeric coordinates')
        if any(a >= b for a, b in zip(box['min'], box['max'])):
            raise ValueError('Box min must be less than max')
    result = copy.deepcopy(profile)
    result['self_mask_enabled'] = True
    return result


def inspect_bag(path, expected_frame):
    if not path.is_file():
        raise ValueError('raw.bag is missing')
    import rosbag
    with rosbag.Bag(str(path), 'r') as bag:
        info = bag.get_type_and_topic_info().topics.get('/livox/lidar')
        if info is None or info.message_count == 0:
            raise ValueError('No recorded /livox/lidar messages')
        if info.msg_type != 'livox_ros_driver2/CustomMsg':
            raise ValueError('Expected livox_ros_driver2/CustomMsg')
        _, message, _ = next(bag.read_messages(topics=['/livox/lidar']))
        if message.header.frame_id != expected_frame:
            raise ValueError('Candidate frame does not match the recorded lidar frame')
        return {'frames_expected': info.message_count,
                'bag_duration_seconds': max(0., bag.get_end_time()-bag.get_start_time())}


def wait_ready(ready, process, seconds, label):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(label+' process exited before becoming ready')
        if ready():
            return
        time.sleep(.05)
    raise RuntimeError(label+' readiness timed out')


def stop_owned_group(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    for sig, delay in [(signal.SIGTERM, 12), (signal.SIGKILL, 5)]:
        try:
            process.wait(timeout=delay)
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                return
    process.wait(timeout=5)


def summarize(rows, expected, frame):
    report = {'replay_only': True, 'self_identity_confirmed': False,
              'frames_expected': expected, 'frames_processed': len(rows),
              'frame': frame, 'frames_by_name': dict(collections.Counter(n for n, _, _ in rows)),
              'states': dict(collections.Counter(s for _, s, _ in rows))}
    report['frame_mismatch_outputs'] = sum(n != frame for n, _, _ in rows)
    report['complete_frame_count'] = len(rows) == expected
    for key in ['distance_m', 'self_masked_points', 'near_untrusted_points', 'processing_ms']:
        values = [float(v[key]) for _, _, v in rows if key in v and math.isfinite(float(v[key]))]
        if values:
            report[key] = {'min': min(values), 'mean': statistics.mean(values), 'max': max(values)}
    return report


def main():
    import yaml
    parser = argparse.ArgumentParser(description='Replay a candidate geometry profile on an isolated ROS master')
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    profile = trial_profile(yaml.safe_load((root/'candidate_mask.yaml').read_text()))
    bag_path = root/'raw.bag'
    metadata = inspect_bag(bag_path, profile['self_mask_frame'])
    port = 11326
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', port))
    os.environ['ROS_MASTER_URI'] = f'http://127.0.0.1:{port}'
    os.environ['ROS_IP'] = '127.0.0.1'
    os.environ.pop('ROS_HOSTNAME', None)
    import rospy
    from diagnostic_msgs.msg import DiagnosticArray
    output = root/('replay_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(exist_ok=False)
    trial = output/'replay_only_mask.yaml'
    trial.write_text(yaml.safe_dump(profile))
    rows = []
    def receive(message):
        if message.status and message.header.frame_id:
            status = message.status[0]
            rows.append((message.header.frame_id, status.message, {v.key: v.value for v in status.values}))
    launch = player = None
    with (output/'replay_candidate.log').open('x') as log:
        try:
            launch = subprocess.Popen(['roslaunch', '-p', str(port), 'haptic_feedback',
                'pcl_proximity.launch', 'self_mask_config:='+str(trial)],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            def master_ready():
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=.1): return True
                except OSError: return False
            wait_ready(master_ready, launch, 10, 'Master')
            rospy.init_node('replay_candidate_report', disable_signals=True)
            subscriber = rospy.Subscriber('/haptic/proximity/diagnostics', DiagnosticArray, receive, queue_size=100)
            wait_ready(lambda: subscriber.get_num_connections() > 0, launch, 5, 'Detector')
            player = subprocess.Popen(['rosbag', 'play', '--wait-for-subscribers', '-r', '1', str(bag_path),
                '--topics', '/livox/lidar'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            timeout = max(45., min(180., metadata['bag_duration_seconds']*2+10))
            if player.wait(timeout=timeout) != 0:
                raise RuntimeError('Bag player exited unsuccessfully')
            deadline = time.monotonic()+2
            while len(rows) < metadata['frames_expected'] and time.monotonic() < deadline:
                time.sleep(.02)
            subscriber.unregister()
            if not rows:
                raise RuntimeError('No frame diagnostics received')
            report = summarize(rows, metadata['frames_expected'], profile['self_mask_frame'])
            report['mask_box'] = profile['self_boxes']
        finally:
            try:
                if rospy.core.is_initialized(): rospy.signal_shutdown('Replay complete')
            finally:
                try: stop_owned_group(player)
                finally: stop_owned_group(launch)
    deadline = time.monotonic()+5
    while master_ready() and time.monotonic() < deadline: time.sleep(.05)
    if master_ready():
        raise RuntimeError('Temporary master remains reachable; inspect the owned launch log')
    report['temporary_master_stopped'] = True
    (output/'replay_candidate_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print('SAVED_DIRECTORY='+str(output))


if __name__ == '__main__':
    main()
