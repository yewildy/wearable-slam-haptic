#!/usr/bin/env python3
"""第二步纯计算：已审查局部过滤、邻域支持、距离到软件强度，不控制硬件。"""
from dataclasses import dataclass
import math
import time
import numpy as np
from review_self_mask import preview


@dataclass(frozen=True)
class Settings:
    expected_frame: str = 'livox_frame'
    min_range: float = .2
    max_range: float = 5.
    support_radius: float = .08
    min_neighbors: int = 3  # 不包含查询点自己；相同坐标的其他回波仍按点计。
    strong_distance: float = .5
    medium_distance: float = 1.
    weak_distance: float = 2.
    hysteresis: float = .1
    release_frames: int = 2
    stale_timeout: float = .5
    future_tolerance: float = .1

    def __post_init__(self):
        if not isinstance(self.expected_frame, str) or not self.expected_frame.strip():
            raise ValueError('expected_frame must be a nonempty sensor-local frame')
        names = ('min_range', 'max_range', 'support_radius', 'strong_distance',
                 'medium_distance', 'weak_distance', 'hysteresis', 'stale_timeout', 'future_tolerance')
        if any(type(getattr(self, k)) not in (int, float) or not math.isfinite(getattr(self, k)) for k in names):
            raise ValueError('Parameters must be finite numbers, not booleans')
        if not (0 < self.min_range < self.strong_distance < self.medium_distance <
                self.weak_distance < self.max_range and 0 < self.support_radius <= self.max_range
                and self.hysteresis >= 0 and self.stale_timeout > 0 and self.future_tolerance >= 0):
            raise ValueError('Invalid distance or timing thresholds')
        if any(type(getattr(self, k)) is not int or getattr(self, k) < 1 for k in ('min_neighbors', 'release_frames')):
            raise ValueError('Counts must be positive integers')


def empty_profile(frame):
    return dict(schema_version=1, kind='reviewed_sensor_voxels', self_mask_enabled=False,
                frame=frame, voxel_m=.02, self_voxels=[], self_identity_confirmed=False,
                confirmation_basis='not_confirmed', review_basis='', masked_space_occupancy='unknown_not_free')


def distance_strength(distance, settings):
    """目标值：<=0.5m为1，1m为0.5，>=2m为0，中间分段线性。"""
    if not math.isfinite(distance) or distance < 0:
        raise ValueError('A finite nonnegative distance is required')
    if distance <= settings.strong_distance:
        return 1.
    if distance <= settings.medium_distance:
        return .5 + .5*(settings.medium_distance-distance)/(settings.medium_distance-settings.strong_distance)
    return max(0., .5*(settings.weak_distance-distance)/(settings.weak_distance-settings.medium_distance))


def supported_points(xyz, radius, neighbors, backend='scipy'):
    """在距离ROI内查邻居。SciPy用于设备；小数组参考算法只供本机验证。"""
    if not len(xyz):
        return np.zeros(0, dtype=bool)
    if backend == 'scipy':
        from scipy.spatial import cKDTree
        counts = cKDTree(xyz).query_ball_point(xyz, radius, eps=0., return_length=True)
    elif backend == 'reference':
        if len(xyz) > 4000:
            raise ValueError('Reference backend is limited to small teaching/test arrays')
        counts = np.empty(len(xyz), dtype=np.int64)
        for first in range(0, len(xyz), 128):
            delta = xyz[first:first+128, None, :] - xyz[None, :, :]
            counts[first:first+128] = np.count_nonzero(np.sum(delta*delta, axis=2) <= radius*radius, axis=1)
    else:
        raise ValueError('Unknown support backend')
    return counts-1 >= neighbors


class DistancePolicy:
    """有状态部分只有时间检查和等级释放迟滞，没有速度估计。"""
    def __init__(self, settings=None, profile=None, apply_self_mask=False, backend='scipy'):
        self.settings = settings or Settings()
        if type(apply_self_mask) is not bool:
            raise ValueError('apply_self_mask must be boolean')
        if apply_self_mask and profile is None:
            raise ValueError('Explicit reviewed profile required to enable masking')
        self.profile = profile if profile is not None else empty_profile(self.settings.expected_frame)
        self.apply_self_mask, self.backend = apply_self_mask, backend
        preview(np.empty((0, 3)), self.settings.expected_frame, self.profile, enabled=apply_self_mask)
        self.level, self.release_count, self.last_stamp, self.last_receipt = None, 0, None, None

    def unknown(self, reason, **details):
        self.level, self.release_count = None, 0
        return self._report(reason, **details)

    def _report(self, reason, **details):
        return dict(state=reason, valid=False, distance_m=None, strength=None, level=None,
                    candidate_distance_m=None, candidate_strength=None, motor_control=False,
                    self_mask_enabled=self.apply_self_mask,
                    identity_basis='reviewer_declaration_only' if self.apply_self_mask and self.profile['self_voxels'] else 'not_applied',
                    masked_space_occupancy='unknown_not_free', scope='all_visible_sensor_directions_no_ground_filter',
                    **details)

    def watchdog(self, now=None):
        now = time.monotonic() if now is None else now
        if self.last_receipt is None or now-self.last_receipt > self.settings.stale_timeout:
            return self.unknown('STALE_OR_NO_DATA')
        return None

    def support_mask(self, xyz):
        return supported_points(xyz, self.settings.support_radius, self.settings.min_neighbors, self.backend)

    def _level(self, distance):
        s = self.settings
        band = 3 if distance <= s.strong_distance else 2 if distance <= s.medium_distance else 1 if distance <= s.weak_distance else 0
        if self.level is None or band >= self.level:
            self.level, self.release_count = band, 0
        else:
            boundary = {3:s.strong_distance, 2:s.medium_distance, 1:s.weak_distance}[self.level]
            self.release_count = self.release_count+1 if distance > boundary+s.hysteresis else 0
            if self.release_count >= s.release_frames:
                self.level, self.release_count = band, 0
        return self.level

    def process(self, points, frame, stamp, source_age_s=0., declared_count=None, now=None):
        """输入必须是传感器局部XYZ。age由ROS适配层提供；离线传0不表示测得延迟。"""
        s = self.settings
        self.last_receipt = time.monotonic() if now is None else now
        if frame != s.expected_frame:
            return self.unknown('FRAME_MISMATCH')
        if type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp <= 0:
            return self.unknown('INVALID_TIMESTAMP')
        if not math.isfinite(source_age_s) or source_age_s > s.stale_timeout or source_age_s < -s.future_tolerance:
            return self.unknown('STALE_OR_FUTURE_HEADER')
        if self.last_stamp is not None and stamp <= self.last_stamp:
            return self.unknown('TIMESTAMP_NONINCREASING')
        self.last_stamp = stamp
        xyz = np.asarray(points)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.dtype.kind not in 'fi':
            return self.unknown('INVALID_COORDINATE_ARRAY')
        if declared_count is not None and (type(declared_count) is not int or declared_count != len(xyz)):
            return self.unknown('POINT_COUNT_MISMATCH')
        masks = preview(xyz, frame, self.profile, enabled=self.apply_self_mask)
        remaining = xyz[masks['kept']].astype(np.float64)
        ranges = np.hypot(np.hypot(remaining[:, 0], remaining[:, 1]), remaining[:, 2])
        near = ranges < s.min_range
        roi = (ranges >= s.min_range) & (ranges <= s.max_range)
        candidate_xyz, candidate_ranges = remaining[roi], ranges[roi]
        supported = self.support_mask(candidate_xyz)
        nearest = float(candidate_ranges[supported].min()) if supported.any() else None
        raw_nearest = float(ranges.min()) if len(ranges) else None
        unsupported_closer = bool((~supported).any() and
                                  (nearest is None or np.any(candidate_ranges[~supported] < nearest-1e-6)))
        result = self._report('NO_SUPPORTED_POINTS')
        result.update(input_points=len(xyz), invalid_points=int(masks['invalid'].sum()),
                      self_removed_points=int(masks['self_removed'].sum()), kept_valid_points=len(remaining),
                      near_untrusted_points=int(near.sum()), range_points=len(candidate_xyz),
                      far_points=int(np.count_nonzero(ranges > s.max_range)),
                      supported_points=int(supported.sum()), unsupported_points=int((~supported).sum()),
                      unsupported_closer_return=unsupported_closer, raw_nearest_m=raw_nearest,
                      candidate_distance_m=nearest,
                      candidate_strength=distance_strength(nearest, s) if nearest is not None else None)
        if near.any():
            result['state'] = 'NEAR_FIELD_UNCERTAIN'
        elif unsupported_closer:
            result['state'] = 'UNSUPPORTED_CLOSER_RETURN'
        elif nearest is not None:
            result.update(state='DISTANCE_CANDIDATE' if nearest <= s.weak_distance else 'NO_NEAR_CANDIDATE',
                          valid=True, distance_m=nearest, strength=distance_strength(nearest, s), level=self._level(nearest))
        elif len(remaining) and result['far_points'] == len(remaining):
            result.update(state='NO_NEAR_CANDIDATE', valid=True, strength=0., level=self._level(s.max_range))
        elif not len(remaining):
            result['state'] = 'NO_REMAINING_OBSERVATIONS'
        if not result['valid']:
            self.level, self.release_count = None, 0
        return result


def main():
    """在任何有NumPy的电脑上演示映射，不需要ROS、SciPy或雷达。"""
    import json
    s = Settings()
    for d in (2.5, 2., 1.5, 1., .75, .5, .3):
        print(json.dumps(dict(distance_m=d, strength=distance_strength(d, s))))


if __name__ == '__main__':
    main()
