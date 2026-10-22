import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time

import rospy
from diagnostic_msgs.msg import DiagnosticArray
from livox_ros_driver2.msg import CustomMsg, CustomPoint

ROOT = Path('/home/eno/haptic_ws/logs/20261021/pcl_proximity_tests')
ROOT.mkdir(parents=True, exist_ok=True)
PORT = 11324
with socket.socket() as probe:
    probe.bind(('127.0.0.1', PORT))
os.environ['ROS_MASTER_URI'] = f'http://127.0.0.1:{PORT}'
os.environ['ROS_IP'] = '127.0.0.1'
os.environ.pop('ROS_HOSTNAME', None)
latest = None
condition = threading.Condition()
results = []


def receive(msg):
    global latest
    with condition:
        latest = msg
        condition.notify_all()


def cluster(x):
    return [(x, 0.01*i, 0.0) for i in range(6)]


def publish_case(seq, points, expected, **counts):
    message = CustomMsg()
    message.header.seq = seq
    message.header.frame_id = f'synthetic_pcl_test_{seq}'
    for x, y, z in points:
        p = CustomPoint()
        p.x, p.y, p.z = x, y, z
        message.points.append(p)
    message.point_num = len(message.points)
    for _ in range(4):
        message.header.stamp = rospy.Time.now()
        publisher.publish(message)
        time.sleep(0.1)
    deadline = time.monotonic()+5
    with condition:
        while latest is None or latest.header.frame_id != message.header.frame_id:
            if time.monotonic() > deadline:
                raise RuntimeError(f'No output for {seq}')
            condition.wait(0.1)
        status = latest.status[0]
        fields = {item.key: item.value for item in status.values}
    assert status.message == expected, (seq, status.message, expected)
    for key, value in counts.items():
        assert fields[key] == str(value), (seq, key, fields[key], value)
    results.append({'case': seq, 'state': status.message, 'level': fields['level'],
                    'distance_m': None if fields['distance_m'] == 'nan' else float(fields['distance_m'])})
    return fields


process = None
with (ROOT/'launch.log').open('w') as log:
    try:
        process = subprocess.Popen(['roslaunch', '-p', str(PORT), 'haptic_feedback',
                                    'pcl_proximity.launch', 'input_topic:=/haptic/test/cloud'],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(100):
            try:
                with socket.create_connection(('127.0.0.1', PORT), timeout=0.1): break
            except OSError:
                if process.poll() is not None: raise RuntimeError('Launch failed')
                time.sleep(0.1)
        rospy.init_node('test_pcl_proximity', disable_signals=True)
        subscriber = rospy.Subscriber('/haptic/proximity/diagnostics', DiagnosticArray, receive, queue_size=30)
        publisher = rospy.Publisher('/haptic/test/cloud', CustomMsg, queue_size=1)
        for _ in range(100):
            if publisher.get_num_connections(): break
            time.sleep(0.05)
        assert publisher.get_num_connections(), 'Node never subscribed'
        time.sleep(0.7)
        assert latest.status[0].message == 'STALE_OR_NO_DATA'
        publish_case(1, [], 'NO_SUPPORTED_POINTS', level='UNKNOWN')
        publish_case(2, [(0,0,0), (float('nan'),0,0)], 'NO_SUPPORTED_POINTS', zero_points=1, nonfinite_points=1)
        publish_case(3, [(0.1,0,0)], 'NEAR_FIELD_UNCERTAIN', near_untrusted_points=1, level='UNKNOWN')
        publish_case(4, [(0.3,0,0)], 'NO_SUPPORTED_POINTS', removed_by_support=1)
        fields = publish_case(5, cluster(0.4)+[(0,0,0),(0.25,0,0)], 'STRONG', level=3, zero_points=1,
                              supported_points=6, removed_by_support=1, unsupported_closer_return='true')
        assert abs(float(fields['distance_m'])-0.4) < 1e-5
        publish_case(6, cluster(0.53), 'STRONG', level=3)
        publish_case(7, cluster(0.8), 'MEDIUM', level=2)
        publish_case(8, cluster(1.5), 'WEAK', level=1)
        publish_case(9, cluster(3.0), 'NO_NEAR_CANDIDATE', level=0)
        publish_case(10, cluster(1.5), 'WEAK', level=1)
        publish_case(11, cluster(0.4)+[(0.1,0,0)], 'STRONG;NEAR_FIELD_UNCERTAIN', level=3)
        publish_case(12, cluster(6.0), 'NO_SUPPORTED_POINTS', far_excluded_points=6)
        time.sleep(0.8)
        assert latest.status[0].message == 'STALE_OR_NO_DATA'
        fields = {v.key:v.value for v in latest.status[0].values}
        assert fields['level'] == 'UNKNOWN' and math.isnan(float(fields['distance_m']))
        publish_case(13, cluster(0.4), 'STRONG', level=3)
        invalid = subprocess.run(['rosrun','haptic_feedback','pcl_proximity_node','__name:=invalid_params_test',
                                  '_min_range:=0.6'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
        assert invalid.returncode != 0, 'Invalid configuration accepted'
        report = {'synthetic_only': True, 'cases': results, 'parameter_validation_passed': True,
                  'startup_and_dropout_unknown_passed': True, 'physical_motor_tested': False,
                  'isolated_master_port': PORT}
    finally:
        if rospy.core.is_initialized(): rospy.signal_shutdown('Test complete')
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try: process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
with socket.socket() as probe:
    assert probe.connect_ex(('127.0.0.1',PORT)) != 0, 'Temporary master left running'
report['temporary_master_stopped'] = True
(ROOT/'verification.json').write_text(json.dumps(report, indent=2),encoding='utf-8')
print(json.dumps(report, indent=2))
