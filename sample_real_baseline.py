import csv
import json
import math
from pathlib import Path
import statistics
import threading
import time

import rospy
from livox_ros_driver2.msg import CustomMsg
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, UInt32

ROOT = Path('/home/eno/haptic_ws/logs/20261021/real_baseline')
ROOT.mkdir(parents=True, exist_ok=True)
rows, outputs, counts, imu_times = [], [], [], []
lock = threading.Lock()


def cloud(message):
    now = time.monotonic()
    finite = [(math.hypot(p.x, p.y, p.z), p) for p in message.points
              if all(math.isfinite(v) for v in (p.x, p.y, p.z))]
    nearest, point = min(finite, key=lambda pair: pair[0]) if finite else (float('nan'), None)
    positive = [d for d, _ in finite if d > 0]
    row = {
        'arrival_monotonic': now, 'header_seconds': message.header.stamp.to_sec(),
        'frame_id': message.header.frame_id, 'declared_points': message.point_num,
        'points': len(message.points), 'nonfinite_points': len(message.points)-len(finite),
        'zero_points': sum(d == 0 for d, _ in finite), 'raw_min_m': nearest,
        'positive_min_m': min(positive, default=float('nan')),
        'nearest_x': point.x if point else None, 'nearest_y': point.y if point else None,
        'nearest_z': point.z if point else None, 'nearest_tag': point.tag if point else None,
    }
    with lock:
        rows.append(row)


rospy.init_node('haptic_real_baseline_check', anonymous=True)
subs = [
    rospy.Subscriber('/livox/lidar', CustomMsg, cloud, queue_size=5, buff_size=8*1024*1024),
    rospy.Subscriber('/haptic/nearest_distance', Float32, lambda m: outputs.append(m.data), queue_size=30),
    rospy.Subscriber('/haptic/point_count', UInt32, lambda m: counts.append(m.data), queue_size=30),
    rospy.Subscriber('/livox/imu', Imu, lambda m: imu_times.append(time.monotonic()), queue_size=500),
]
rospy.sleep(15)
for sub in subs:
    sub.unregister()
with lock:
    observed = list(rows)
assert len(observed) >= 20, 'Too few real frames'
assert len(outputs) >= 20 and len(counts) >= 20, 'Distance node output missing'
info = rospy.get_published_topics('/')
assert '/haptic/demo/lidar' not in dict(info), 'Mock publisher still present'
values = [r['raw_min_m'] for r in observed if math.isfinite(r['raw_min_m'])]
positive = [r['positive_min_m'] for r in observed if math.isfinite(r['positive_min_m'])]
header_times = [r['header_seconds'] for r in observed]
gaps = [b-a for a,b in zip(header_times,header_times[1:])]
arrivals = [r['arrival_monotonic'] for r in observed]
result = {
    'physical_lidar_tested': True, 'motor_control_enabled': False,
    'test_kind': 'observational_data_link_and_raw_distance_only', 'sample_seconds_requested': 15,
    'cloud_frames': len(observed), 'cloud_observed_hz': (len(observed)-1)/(arrivals[-1]-arrivals[0]),
    'frame_ids': sorted(set(r['frame_id'] for r in observed)),
    'points_per_frame_min': min(r['points'] for r in observed),
    'points_per_frame_max': max(r['points'] for r in observed),
    'points_per_frame_mean': statistics.mean(r['points'] for r in observed),
    'declared_count_mismatches': sum(r['declared_points'] != r['points'] for r in observed),
    'nonfinite_points': sum(r['nonfinite_points'] for r in observed),
    'zero_coordinate_points': sum(r['zero_points'] for r in observed),
    'raw_nearest_min_m': min(values, default=None), 'raw_nearest_max_m': max(values, default=None),
    'positive_nearest_min_m_diagnostic_only': min(positive, default=None),
    'positive_nearest_max_m_diagnostic_only': max(positive, default=None),
    'header_non_increasing_steps': sum(g <= 0 for g in gaps),
    'header_max_gap_seconds': max(gaps, default=None),
    'distance_messages': len(outputs), 'count_messages': len(counts),
    'distance_output_min_m': min(outputs), 'distance_output_max_m': max(outputs),
    'imu_messages': len(imu_times),
    'imu_observed_hz': (len(imu_times)-1)/(imu_times[-1]-imu_times[0]) if len(imu_times)>1 else None,
    'mock_topic_absent': True, 'ground_truth_distance_measured': False,
    'obstacle_detection_validated': False, 'full_scan_saved': False,
}
with (ROOT/'frame_summary_private.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(observed[0]))
    writer.writeheader()
    writer.writerows(observed)
(ROOT/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
rospy.signal_shutdown('Observational sample complete')
