"""Checks metadata in a separately built binary, on an owned synthetic ROS graph."""
import argparse
import datetime
import json
import math
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path, help='Explicit isolated build executable, not the live binary')
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise ValueError('Isolated binary is missing')
    port = 11329
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', port))
    os.environ['ROS_MASTER_URI'] = 'http://127.0.0.1:'+str(port)
    os.environ['ROS_IP'] = '127.0.0.1'
    os.environ.pop('ROS_HOSTNAME', None)
    root = Path.home()/'haptic_ws/logs/20261023'/('provenance_'+datetime.datetime.now().strftime('%H%M%S_%f'))
    root.mkdir(parents=True, exist_ok=False)
    messages, results, signatures = {}, [], {}
    master = detector = None

    def ready():
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=.1):
                return True
        except OSError:
            return False

    def receive(message):
        messages[(message.header.frame_id, message.header.stamp.to_nsec())] = message

    def values(message):
        return {v.key: v.value for v in message.status[0].values}

    def metadata(message, enabled, confirmed):
        data = values(message)
        expected = 'DISABLED' if not enabled else 'USER_CONFIRMED' if confirmed else 'CANDIDATE_UNCONFIRMED'
        assert data['self_mask_review_state'] == expected, data
        assert data['self_identity_confirmed'] == str(confirmed).lower(), data
        assert data['self_mask_confirmation_basis'] == ('explicit_configuration_only' if confirmed else 'not_confirmed'), data
        assert data['masked_space_occupancy'] == ('unknown_not_free' if enabled else 'not_inferred_by_mask'), data
        return data

    def send(frame='provenance_fixture'):
        message = CustomMsg()
        message.header.frame_id = frame
        for x in [-.3, .8]:
            for i in range(6):
                p = CustomPoint()
                p.x, p.y, p.z = x, i*.01, 0
                message.points.append(p)
        message.point_num = len(message.points)
        for _ in range(4):
            message.header.stamp = rospy.Time.now()
            publisher.publish(message)
            time.sleep(.1)
        key = (frame, message.header.stamp.to_nsec())
        wait_ready(lambda: key in messages, detector, 3, 'Matching cloud diagnostic')
        return messages[key]

    with (root/'run.log').open('x') as log:
        try:
            master = subprocess.Popen(['roscore', '-p', str(port)], stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            wait_ready(ready, master, 10, 'Isolated master')
            rospy.init_node('test_mask_provenance', disable_signals=True)
            sub = rospy.Subscriber('/haptic/proximity/diagnostics', DiagnosticArray, receive, queue_size=100)
            publisher = rospy.Publisher('/haptic/test/provenance_cloud', CustomMsg, queue_size=1)
            for label, enabled, confirmed in [('default_absent', False, None), ('candidate', True, False),
                                               ('declared_confirmed', True, True), ('disabled_declared', False, True)]:
                name = 'provenance_'+label
                params = {'input_topic': '/haptic/test/provenance_cloud', 'self_mask_enabled': enabled,
                          'self_mask_frame': 'provenance_fixture',
                          'self_boxes': [{'min': [-.4, -.1, -.1], 'max': [-.2, .1, .1]}]}
                if confirmed is not None:
                    params['self_identity_confirmed'] = confirmed
                for key, value in params.items():
                    rospy.set_param('/'+name+'/'+key, value)
                messages.clear()
                detector = subprocess.Popen([str(binary), '__name:='+name], stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True)
                wait_ready(lambda: publisher.get_num_connections() > 0 and sub.get_num_connections() > 0,
                           detector, 8, 'Detector')
                first = send()
                data = metadata(first, enabled, bool(confirmed))
                assert first.status[0].message == ('MEDIUM' if enabled else 'STRONG')
                assert abs(float(data['distance_m'])-(.8 if enabled else .3)) < 1e-5
                signature = {key: data[key] for key in ['distance_m', 'level', 'self_masked_points', 'range_points',
                                                       'supported_points', 'removed_by_support']}
                signatures[label] = signature
                results.append({'case': label+'_normal', 'review_state': data['self_mask_review_state'],
                                'state': first.status[0].message})
                if enabled:
                    mismatch = send('wrong_frame')
                    bad = metadata(mismatch, enabled, bool(confirmed))
                    assert mismatch.status[0].message == 'SELF_MASK_FRAME_MISMATCH'
                    assert bad['level'] == 'UNKNOWN' and math.isnan(float(bad['distance_m']))
                    results.append({'case': label+'_frame_mismatch', 'review_state': bad['self_mask_review_state'],
                                    'state': mismatch.status[0].message})
                    metadata(send(), enabled, bool(confirmed))
                before = set(messages)
                wait_ready(lambda: any(key[0] == '' and key not in before for key in messages),
                           detector, 3, 'Stale watchdog')
                stale = next(message for key, message in messages.items() if key[0] == '' and key not in before)
                stale_data = metadata(stale, enabled, bool(confirmed))
                assert stale.status[0].message == 'STALE_OR_NO_DATA' and stale_data['level'] == 'UNKNOWN'
                results.append({'case': label+'_stale', 'review_state': stale_data['self_mask_review_state'],
                                'state': stale.status[0].message})
                stop_owned_group(detector)
                detector = None
                deadline = time.monotonic()+3
                while (publisher.get_num_connections() or sub.get_num_connections()) and time.monotonic() < deadline:
                    time.sleep(.05)
                assert publisher.get_num_connections() == sub.get_num_connections() == 0
            assert signatures['candidate'] == signatures['declared_confirmed']
            assert signatures['default_absent'] == signatures['disabled_declared']
            for i, bad in enumerate(['true', 1, []]):
                name = 'provenance_bad_'+str(i)
                rospy.set_param('/'+name+'/self_identity_confirmed', bad)
                result = subprocess.run([str(binary), '__name:='+name], stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout=8, text=True)
                (root/('invalid_'+str(i)+'.log')).write_text(result.stdout, encoding='utf-8')
                assert result.returncode == 1, (i, repr(bad), result.returncode, result.stdout)
                _, _, system = rospy.get_master().getSystemState()
                assert not any(topic == '/haptic/proximity/diagnostics' and '/'+name in nodes
                               for topic, nodes in system[0]), 'Invalid node advertised diagnostics'
                results.append({'case': 'reject_nonboolean_'+str(i), 'rejected': True})
        finally:
            try:
                if rospy.core.is_initialized():
                    rospy.signal_shutdown('Provenance checks complete')
            finally:
                try:
                    stop_owned_group(detector)
                finally:
                    stop_owned_group(master)
    with socket.socket() as probe:
        assert probe.connect_ex(('127.0.0.1', port)) != 0
    report = {'synthetic_only': True, 'isolated_build_only': True, 'live_deployment_performed': False,
              'cases': results, 'metadata_does_not_change_filter_results_in_fixtures': True,
              'temporary_master_stopped': True, 'physical_identity_verified': False,
              'confirmation_meaning': 'Explicit configuration assertion only; not safety or physical verification'}
    (root/'provenance_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('VERIFICATION_FILE='+str(root/'provenance_verification.json'))


if __name__ == '__main__':
    main()
