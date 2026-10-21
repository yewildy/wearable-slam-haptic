import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import rospy
from livox_ros_driver2.msg import CustomMsg
from std_msgs.msg import Float32, UInt32


ROOT = Path('/home/eno/haptic_ws/logs/20261021/distance_demo')
PORT = 11323
with socket.socket() as probe:
    probe.bind(('127.0.0.1', PORT))
os.environ['ROS_MASTER_URI'] = f'http://127.0.0.1:{PORT}'
os.environ['ROS_IP'] = '127.0.0.1'
os.environ.pop('ROS_HOSTNAME', None)
ROOT.mkdir(parents=True, exist_ok=True)
process = None
with (ROOT / 'launch.log').open('w') as log:
    try:
        process = subprocess.Popen(
            ['roslaunch', '-p', str(PORT), 'haptic_feedback', 'demo.launch'],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 20
        while True:
            try:
                with socket.create_connection(('127.0.0.1', PORT), timeout=.2):
                    break
            except OSError:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Demo master did not start')
                time.sleep(.2)
        rospy.init_node('haptic_demo_test', disable_signals=True)
        raw = rospy.wait_for_message('/haptic/demo/lidar', CustomMsg, timeout=10)
        counts = [rospy.wait_for_message('/haptic/point_count', UInt32, timeout=5).data for _ in range(3)]
        assert raw.point_num == len(raw.points) == 4
        assert [point.x for point in raw.points] == [1., 2., 3., 4.]
        assert counts == [4, 4, 4]
        distances = [rospy.wait_for_message('/haptic/nearest_distance', Float32, timeout=5).data for _ in range(3)]
        assert distances == [1.0, 1.0, 1.0]
        published = dict(rospy.get_published_topics('/'))
        assert '/livox/lidar' not in published
        assert published['/haptic/demo/lidar'] == 'livox_ros_driver2/CustomMsg'
        assert published['/haptic/point_count'] == 'std_msgs/UInt32'
        assert published['/haptic/nearest_distance'] == 'std_msgs/Float32'
        info = subprocess.check_output(['rosnode', 'info', '/haptic_feedback_node'], text=True)
        assert '/haptic/demo/lidar [livox_ros_driver2/CustomMsg]' in info
        report = {
            'synthetic_data_only': True, 'physical_lidar_tested': False,
            'demo_input': '/haptic/demo/lidar', 'output': '/haptic/point_count',
            'input_type': 'livox_ros_driver2/CustomMsg', 'output_type': 'std_msgs/UInt32',
            'received_counts': counts, 'live_lidar_topic_not_published': True,
            'distance_topic': '/haptic/nearest_distance', 'distance_type': 'std_msgs/Float32',
            'received_distances_m': distances,
            'isolated_master_port': PORT, 'motor_control_enabled': False,
        }
    finally:
        rospy.signal_shutdown('Done')
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
try:
    with socket.create_connection(('127.0.0.1', PORT), timeout=.2):
        raise RuntimeError('Temporary master was left running')
except ConnectionRefusedError:
    report['temporary_master_stopped'] = True
(ROOT / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
