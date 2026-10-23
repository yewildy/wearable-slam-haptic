#!/usr/bin/env python3
"""ROS1收发适配层。只订阅已有驱动；不启动建图/驱动，不控制马达。"""
import json
import threading
import time
import numpy as np
from distance_policy import DistancePolicy, Settings


def cloud_result(policy, msg, ros_now, now=None):
    points = np.asarray([(p.x, p.y, p.z) for p in msg.points], dtype=np.float64).reshape(-1, 3)
    stamp = msg.header.stamp.to_sec()
    return policy.process(points, msg.header.frame_id, stamp, ros_now-stamp, msg.point_num, now)


def main():
    import rospy
    from livox_ros_driver2.msg import CustomMsg
    from std_msgs.msg import Float32, String
    rospy.init_node('haptic_distance_v2')
    values = {name:rospy.get_param('~'+name, field.default) for name, field in Settings.__dataclass_fields__.items()}
    profile_path = rospy.get_param('~reviewed_profile', '')
    profile = None
    if profile_path:
        from pathlib import Path
        profile = json.loads(Path(profile_path).expanduser().read_text(encoding='utf-8'))
    policy = DistancePolicy(Settings(**values), profile, rospy.get_param('~apply_self_mask', False))
    from rolling_near import RollingNear
    from rolling_distance import rolling_result
    rolling = RollingNear() if rospy.get_param('~rolling_near_experiment', False) else None
    # 不静默退回耗时的参考算法；设备缺SciPy时明确报错，不安装/升级库。
    import scipy.spatial
    prefix = '/haptic/distance_v2/'
    pubs = {name:rospy.Publisher(prefix+name, kind, queue_size=1) for name, kind in
            [('strength', Float32), ('distance', Float32), ('state', String), ('report', String)]}
    lock = threading.Lock()
    callback_started = [None]

    def publish(result):
        pubs['strength'].publish(float('nan') if result['strength'] is None else result['strength'])
        pubs['distance'].publish(float('nan') if result['distance_m'] is None else result['distance_m'])
        pubs['state'].publish(result['state'])
        pubs['report'].publish(json.dumps(result, allow_nan=False, ensure_ascii=True))

    def callback(msg):
        with lock:
            started = time.monotonic()
            callback_started[0] = started
            try:
                age = rospy.Time.now().to_sec()-msg.header.stamp.to_sec()
                if rolling is None:
                    result = cloud_result(policy, msg, rospy.Time.now().to_sec(), started)
                else:
                    points = np.asarray([(p.x,p.y,p.z) for p in msg.points], dtype=float).reshape(-1,3)
                    result = rolling_result(policy,rolling,points,msg.header.frame_id,
                                            msg.header.stamp.to_sec(),age,msg.point_num,started)
                result['source_age_s'] = age
            except (ValueError, TypeError, KeyError) as error:
                rospy.logerr_throttle(2., 'Distance input rejected: %s', error)
                result = policy.unknown('INPUT_OR_PROFILE_ERROR')
            elapsed = time.monotonic()-started
            if elapsed > policy.settings.stale_timeout:
                result = policy.unknown('PROCESSING_TOO_SLOW')
                if rolling is not None:rolling.reset()
            elif result['valid'] and rospy.Time.now().to_sec()-msg.header.stamp.to_sec() > policy.settings.stale_timeout:
                result = policy.unknown('STALE_AT_PUBLICATION')
            result.update(input_stamp=msg.header.stamp.to_sec(), input_frame=msg.header.frame_id, processing_ms=1000*elapsed)
            publish(result)
            callback_started[0] = None

    def watchdog_loop():
        # wall time工作，不因/use_sim_time停止而停止断流检查。
        while not rospy.is_shutdown():
            if lock.acquire(blocking=False):
                try:
                    report = policy.watchdog()
                    if report is not None:
                        if rolling is not None:rolling.reset()
                        publish(report)
                finally:
                    lock.release()
            elif callback_started[0] is not None and time.monotonic()-callback_started[0] > policy.settings.stale_timeout:
                publish(policy._report('PROCESSING_TOO_SLOW'))
            time.sleep(.1)

    rospy.Subscriber(rospy.get_param('~input_topic', '/livox/lidar'), CustomMsg, callback,
                     queue_size=1, buff_size=16*1024*1024, tcp_nodelay=True)
    thread = threading.Thread(target=watchdog_loop, daemon=True)
    thread.start()
    rospy.logwarn('SOFTWARE PROTOTYPE: all visible directions, no ground filter, mask=%s, no motor', policy.apply_self_mask)
    rospy.spin()


if __name__ == '__main__':
    main()
