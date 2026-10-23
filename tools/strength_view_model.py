"""只读显示模型；显示断流和非法报告时不沿用旧有效强度。"""
import json
import math
import threading
import time

REASONS = {
    'WAITING_FOR_REPORT': '等待提醒程序的数据',
    'DISPLAY_STALE': '提醒程序断流',
    'INVALID_REPORT': '收到的提醒数据格式有误',
    'DISTANCE_CANDIDATE': '正在根据距离计算提醒',
    'NO_NEAR_CANDIDATE': '本帧没有近距离候选',
    'STALE_OR_NO_DATA': '没有新点云数据',
    'NEAR_FIELD_UNCERTAIN': '近场回波未确认',
    'UNSUPPORTED_CLOSER_RETURN': '更近处存在未确认的孤立回波',
    'NO_SUPPORTED_POINTS': '没有足够的邻域支持点',
    'NO_REMAINING_OBSERVATIONS': '没有剩余有效观测',
    'FRAME_MISMATCH': '输入坐标系不匹配',
    'POINT_COUNT_MISMATCH': '消息点数不一致',
    'INVALID_TIMESTAMP': '点云时间戳无效',
    'TIMESTAMP_NONINCREASING': '时间戳重复或倒退',
    'STALE_OR_FUTURE_HEADER': '点云时间过旧或超前',
    'STALE_AT_PUBLICATION': '计算完成时数据已过旧',
    'PROCESSING_TOO_SLOW': '计算超时',
    'INVALID_COORDINATE_ARRAY': '点云坐标格式有误',
    'INPUT_OR_PROFILE_ERROR': '输入数据或审核配置有误',
}


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def checked_report(text):
    if not isinstance(text, str) or len(text) > 100000:
        raise ValueError('Invalid report text')
    report = json.loads(text)
    if not isinstance(report, dict) or type(report.get('valid')) is not bool:
        raise ValueError('Missing validity flag')
    state = report.get('state')
    if not isinstance(state, str) or not state or len(state) > 100:
        raise ValueError('Invalid state')
    if type(report.get('self_mask_enabled')) is not bool:
        raise ValueError('Missing mask state')
    strength, distance = report.get('strength'), report.get('distance_m')
    if report['valid']:
        if state not in ('DISTANCE_CANDIDATE', 'NO_NEAR_CANDIDATE'):
            raise ValueError('Unexpected valid state')
        if not number(strength) or not 0 <= strength <= 1:
            raise ValueError('Invalid strength')
        if distance is None:
            if state != 'NO_NEAR_CANDIDATE' or strength != 0:
                raise ValueError('Missing valid distance')
        elif not number(distance) or not 0 < distance <= 1000000:
            raise ValueError('Invalid distance')
    elif strength is not None or distance is not None:
        raise ValueError('Unknown result must not contain an effective strength or distance')
    for name in ('candidate_distance_m', 'raw_nearest_m', 'processing_ms'):
        value = report.get(name)
        if value is not None and (not number(value) or not 0 <= value <= 1000000):
            raise ValueError('Invalid diagnostic value')
    for name in ('input_points', 'self_removed_points', 'near_untrusted_points', 'supported_points'):
        value = report.get(name)
        if value is not None and (type(value) is not int or not 0 <= value <= 1000000000):
            raise ValueError('Invalid count')
    return report


class ViewModel:
    def __init__(self, timeout=.75):
        if not number(timeout) or timeout <= 0:
            raise ValueError('Invalid display timeout')
        self.timeout, self.lock, self.report, self.received = timeout, threading.Lock(), None, None

    def receive(self, text, now=None):
        now = time.monotonic() if now is None else now
        try:
            report = checked_report(text)
        except (ValueError, TypeError, RecursionError):
            report = dict(valid=False, state='INVALID_REPORT', strength=None, distance_m=None,
                          self_mask_enabled=False, mask_state_known=False)
        with self.lock:
            self.report, self.received = report, now

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if self.report is None:
                return dict(valid=False, state='WAITING_FOR_REPORT', strength=None, distance_m=None,
                            self_mask_enabled=False, mask_state_known=False, age=None)
            result = dict(self.report)
            result['age'] = max(0., now-self.received)
        if now-self.received > self.timeout or now < self.received:
            result.update(valid=False, state='DISPLAY_STALE', strength=None, distance_m=None,
                          candidate_distance_m=None, raw_nearest_m=None, mask_state_known=False)
        return result
