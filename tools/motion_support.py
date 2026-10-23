"""滚动过滤后独立去噪；邻域支持不是自身身份判断。"""
import math
import numpy as np
from distance_policy import DistancePolicy, supported_points


class SupportedDistancePolicy(DistancePolicy):
    def support_mask(self, xyz):
        # 原始剩余帧上已做一次邻域支持；不在删点后的子集反复侵蚀物体边缘。
        return np.ones(len(xyz), dtype=bool)

    def _support(self, xyz):
        if self.backend != 'scipy' or not len(xyz):
            return supported_points(xyz, self.settings.support_radius,
                                    self.settings.min_neighbors, self.backend)
        from scipy.spatial import cKDTree
        # 只需知道第4近点是否在半径内，不枚举稠密区域的全部邻居。
        distances, _ = cKDTree(xyz).query(xyz, k=self.settings.min_neighbors+1)
        return distances[:, -1] <= self.settings.support_radius

    def process(self, points, frame, stamp, source_age_s=0., declared_count=None, now=None):
        xyz = np.asarray(points)
        self.filtered_points = np.empty((0, 3))
        valid = (xyz.ndim == 2 and xyz.shape[1] == 3 and xyz.dtype.kind in 'fi'
                 and frame == self.settings.expected_frame
                 and type(stamp) in (int, float) and math.isfinite(stamp) and stamp > 0
                 and math.isfinite(source_age_s)
                 and -self.settings.future_tolerance <= source_age_s <= self.settings.stale_timeout
                 and (self.last_stamp is None or stamp > self.last_stamp)
                 and (declared_count is None or type(declared_count) is int and declared_count == len(xyz)))
        if not valid:
            return super().process(points, frame, stamp, source_age_s, declared_count, now)
        finite = np.isfinite(xyz).all(axis=1)
        ranges = np.linalg.norm(xyz, axis=1)
        eligible = finite & (ranges > 0) & (ranges <= self.settings.max_range)
        active = np.flatnonzero(eligible)
        # 单次半径离群点过滤。邻域支持来自本帧原始剩余点，不作递归k-core。
        active = active[self._support(xyz[active])]
        keep = ~eligible
        keep[active] = True
        self.filtered_points = xyz[keep]
        report = super().process(self.filtered_points, frame, stamp, source_age_s, now=now)
        report.update(isolated_removed_points=int(np.count_nonzero(~keep)),
                      isolation_input_points=len(xyz),
                      isolation_policy='8cm_3_other_neighbors_single_pass',
                      thin_object_miss_risk=True)
        return report
