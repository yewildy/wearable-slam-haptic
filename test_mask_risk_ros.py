"""Synthetic counterexamples on an owned ROS master; no hardware topics used."""
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import rospy
from diagnostic_msgs.msg import DiagnosticArray
from livox_ros_driver2.msg import CustomMsg, CustomPoint

from replay_candidate import stop_owned_group, wait_ready


def main():
    port = 11327
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', port))
    os.environ['ROS_MASTER_URI'] = 'http://127.0.0.1:'+str(port)
    os.environ['ROS_IP'] = '127.0.0.1'
    os.environ.pop('ROS_HOSTNAME', None)
    root = Path.home()/'haptic_ws/logs/20261022'/('mask_risk_'+datetime.datetime.now().strftime('%H%M%S_%f'))
    root.mkdir(parents=True, exist_ok=False)
    received, results = {}, []
    master = detector = None

    def connected():
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=.1):
                return True
        except OSError:
            return False

    def receive(message):
        if message.header.frame_id:
            received[message.header.stamp.to_nsec()] = message

    background = [(.8, .01*i, 0) for i in range(6)]
    inside = [(-.3, 0, -.08+.02*i) for i in range(9)]
    crossing = [(-.3, 0, -.12+.03*i) for i in range(9)]
    outside = [(-.401, .01*i, 0) for i in range(6)]

    with (root/'run.log').open('x') as log:
        try:
            master = subprocess.Popen(['roscore', '-p', str(port)], stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            wait_ready(connected, master, 10, 'Isolated master')
            rospy.init_node('mask_risk_test', disable_signals=True)
            subscriber = rospy.Subscriber('/haptic/proximity/diagnostics', DiagnosticArray, receive, queue_size=100)
            publisher = rospy.Publisher('/haptic/test/risk_cloud', CustomMsg, queue_size=1)
            for enabled in (False, True):
                name = 'mask_risk_'+str(enabled).lower()
                params = {'input_topic': '/haptic/test/risk_cloud', 'self_mask_enabled': enabled,
                          'self_mask_frame': 'risk_fixture',
                          'self_boxes': [{'min': [-.4, -.1, -.1], 'max': [-.2, .1, .1]}]}
                for key, value in params.items():
                    rospy.set_param('/'+name+'/'+key, value)
                detector = subprocess.Popen(['rosrun', 'haptic_feedback', 'pcl_proximity_node', '__name:='+name],
                                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                wait_ready(lambda: publisher.get_num_connections() > 0 and subscriber.get_num_connections() > 0,
                           detector, 8, 'Detector')
                for label, points, masked in [('inside_thin_pole', inside, 9),
                                               ('crossing_thin_pole', crossing, 7),
                                               ('outside_box', outside, 0)]:
                    message = CustomMsg()
                    message.header.frame_id = 'risk_fixture'
                    for xyz in points+background:
                        p = CustomPoint()
                        p.x, p.y, p.z = xyz
                        message.points.append(p)
                    message.point_num = len(message.points)
                    for _ in range(4):
                        message.header.stamp = rospy.Time.now()
                        publisher.publish(message)
                        time.sleep(.12)
                    stamp = message.header.stamp.to_nsec()
                    wait_ready(lambda: stamp in received, detector, 3, 'Matching diagnostic')
                    status = received[stamp].status[0]
                    values = {v.key: v.value for v in status.values}
                    expected_state = 'MEDIUM' if enabled and masked else 'STRONG'
                    assert status.message == expected_state, (enabled, label, status.message)
                    assert int(values['self_masked_points']) == (masked if enabled else 0), values
                    if enabled and label == 'crossing_thin_pole':
                        assert int(values['removed_by_support']) == 2, values
                    if enabled and masked:
                        assert abs(float(values['distance_m'])-.8) < 1e-5, values
                    results.append({'mask_enabled': enabled, 'case': label, 'state': status.message,
                                    'distance_m': float(values['distance_m']),
                                    'masked_points': int(values['self_masked_points']),
                                    'removed_by_support': int(values['removed_by_support']),
                                    'supported_points': int(values['supported_points'])})
                stop_owned_group(detector)
                detector = None
                deadline = time.monotonic()+3
                while publisher.get_num_connections() and time.monotonic() < deadline:
                    time.sleep(.05)
                assert publisher.get_num_connections() == 0
        finally:
            try:
                if rospy.core.is_initialized():
                    rospy.signal_shutdown('Synthetic mask risk complete')
            finally:
                try:
                    stop_owned_group(detector)
                finally:
                    stop_owned_group(master)
    with socket.socket() as probe:
        assert probe.connect_ex(('127.0.0.1', port)) != 0
    report = {'synthetic_only': True, 'cases': results, 'temporary_master_stopped': True,
              'live_lidar_used': False, 'physical_motor_tested': False,
              'interpretation': 'A legitimate external pole inside a mask is lost; outside remnants may also fail support. Not real-world miss rate.'}
    (root/'mask_risk_ros_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('VERIFICATION_FILE='+str(root/'mask_risk_ros_verification.json'))


if __name__ == '__main__':
    main()
