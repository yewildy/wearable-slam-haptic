"""滚动过滤后的原型距离提醒；不把时间稳定性称为人体识别。"""
import math
import numpy as np


def rolling_result(policy, rolling, xyz, frame, stamp, age, count, now=None):
    xyz = np.asarray(xyz)
    if (frame != policy.settings.expected_frame or frame != 'livox_frame'
            or not math.isfinite(stamp) or stamp <= 0
            or not math.isfinite(age) or age > policy.settings.stale_timeout
            or age < -policy.settings.future_tolerance
            or xyz.ndim != 2 or xyz.shape[1] != 3
            or type(count) is not int or count != len(xyz)
            or (rolling.last_stamp is not None and stamp <= rolling.last_stamp)):
        rolling.reset()
        return policy.process(xyz,frame,stamp,age,count,now)
    removed, detail = rolling.process(xyz,stamp,frame)
    report = policy.process(xyz[~removed],frame,stamp,age,now=now)
    report.update(detail, original_input_points=len(xyz), experimental=True,
                  masked_space_occupancy='unknown_not_free', motor_control=False)
    if not detail['rolling_ready']:
        report.update(valid=False,state='ROLLING_WARMUP',distance_m=None,strength=None,level=None)
    elif report['state'] == 'NEAR_FIELD_UNCERTAIN':
        # 未解释近回波触发保守提醒，不把它们静默丢掉，也不声称近场量距精度。
        report.update(valid=True,state='NEAR_FIELD_ALERT',distance_m=report['raw_nearest_m'],
                      strength=1.,level=3,near_distance_is_uncertain=True)
    return report
